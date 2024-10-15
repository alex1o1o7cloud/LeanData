import Mathlib

namespace NUMINAMATH_CALUDE_max_product_l3948_394896

def digits : Finset Nat := {3, 5, 6, 8, 9}

def is_valid_pair (a b c d e : Nat) : Prop :=
  {a, b, c, d, e} = digits ∧ 
  100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c < 1000 ∧
  10 ≤ 10 * d + e ∧ 10 * d + e < 100

def product (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product :
  ∀ a b c d e : Nat, is_valid_pair a b c d e →
    product a b c d e ≤ product 9 5 3 8 6 :=
by sorry

end NUMINAMATH_CALUDE_max_product_l3948_394896


namespace NUMINAMATH_CALUDE_prime_between_squares_l3948_394854

theorem prime_between_squares : ∃ p : ℕ, 
  Prime p ∧ 
  (∃ n : ℕ, p = n^2 + 9) ∧ 
  (∃ m : ℕ, p = (m+1)^2 - 8) ∧ 
  p = 73 := by
sorry

end NUMINAMATH_CALUDE_prime_between_squares_l3948_394854


namespace NUMINAMATH_CALUDE_canoe_row_probability_value_l3948_394883

def oar_probability : ℚ := 3/5

/-- The probability of being able to row a canoe with two independent oars -/
def canoe_row_probability : ℚ :=
  oar_probability * oar_probability +  -- Both oars work
  oar_probability * (1 - oar_probability) +  -- Left works, right breaks
  (1 - oar_probability) * oar_probability  -- Left breaks, right works

theorem canoe_row_probability_value :
  canoe_row_probability = 21/25 := by
  sorry

end NUMINAMATH_CALUDE_canoe_row_probability_value_l3948_394883


namespace NUMINAMATH_CALUDE_field_trip_problem_l3948_394871

/-- Given a field trip with vans and buses, calculates the number of people in each van. -/
def peoplePerVan (numVans : ℕ) (numBuses : ℕ) (peoplePerBus : ℕ) (totalPeople : ℕ) : ℕ :=
  (totalPeople - numBuses * peoplePerBus) / numVans

theorem field_trip_problem :
  peoplePerVan 6 8 18 180 = 6 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_problem_l3948_394871


namespace NUMINAMATH_CALUDE_inequality_proof_l3948_394810

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^3 + b^3 + c^3 = 3) : 
  1/(a^4 + 3) + 1/(b^4 + 3) + 1/(c^4 + 3) ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3948_394810


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l3948_394872

theorem arithmetic_sequence_count :
  ∀ (a l d : ℝ) (n : ℕ),
    a = 2.5 →
    l = 62.5 →
    d = 5 →
    l = a + (n - 1) * d →
    n = 13 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l3948_394872


namespace NUMINAMATH_CALUDE_floor_sqrt_20_squared_l3948_394812

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_20_squared_l3948_394812


namespace NUMINAMATH_CALUDE_minimize_expression_l3948_394814

theorem minimize_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 50 / n ≥ 8.1667 ∧
  ((n : ℝ) / 3 + 50 / n = 8.1667 ↔ n = 12) :=
sorry

end NUMINAMATH_CALUDE_minimize_expression_l3948_394814


namespace NUMINAMATH_CALUDE_x_equals_five_l3948_394880

theorem x_equals_five (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : 5 * x^2 + 15 * x * y = x^3 + 2 * x^2 * y + 3 * x * y^2) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_five_l3948_394880


namespace NUMINAMATH_CALUDE_second_quadrant_points_characterization_l3948_394853

def second_quadrant_points : Set (ℤ × ℤ) :=
  {p | p.1 < 0 ∧ p.2 > 0 ∧ p.2 ≤ p.1 + 4}

theorem second_quadrant_points_characterization :
  second_quadrant_points = {(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-3, 1)} := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_points_characterization_l3948_394853


namespace NUMINAMATH_CALUDE_investment_time_calculation_l3948_394840

/-- Represents the investment scenario of two partners A and B --/
structure Investment where
  a_capital : ℝ
  a_time : ℝ
  b_capital : ℝ
  b_time : ℝ
  profit_ratio : ℝ

/-- Theorem stating the time B's investment was effective --/
theorem investment_time_calculation (i : Investment) 
  (h1 : i.a_capital = 27000)
  (h2 : i.b_capital = 36000)
  (h3 : i.a_time = 12)
  (h4 : i.profit_ratio = 2/1) :
  i.b_time = 4.5 := by
  sorry

#check investment_time_calculation

end NUMINAMATH_CALUDE_investment_time_calculation_l3948_394840


namespace NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_21_l3948_394851

theorem constant_term_expansion (x : ℝ) : 
  (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7) = x^7 + 2*x^6 + 2*x^5 + 3*x^4 + x^5 + 2*x^4 + x^3 + 7*x^3 + 7*x^2 + 21 := by
  sorry

theorem constant_term_is_21 : 
  (λ x : ℝ => (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7)) 0 = 21 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_21_l3948_394851


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l3948_394850

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (feet_per_step : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (length_steps : ℝ) * feet_per_step * (width_steps : ℝ) * feet_per_step * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  expected_potato_yield 18 25 3 (3/4) = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l3948_394850


namespace NUMINAMATH_CALUDE_count_divisible_numbers_count_divisible_numbers_proof_l3948_394857

theorem count_divisible_numbers : ℕ → Prop :=
  fun n => 
    (∃ (S : Finset ℕ), 
      (∀ x ∈ S, 1000 ≤ x ∧ x ≤ 3000 ∧ 12 ∣ x ∧ 18 ∣ x ∧ 24 ∣ x) ∧
      (∀ x, 1000 ≤ x ∧ x ≤ 3000 ∧ 12 ∣ x ∧ 18 ∣ x ∧ 24 ∣ x → x ∈ S) ∧
      S.card = n) →
    n = 28

theorem count_divisible_numbers_proof : count_divisible_numbers 28 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_count_divisible_numbers_proof_l3948_394857


namespace NUMINAMATH_CALUDE_x_varies_as_z_to_four_thirds_l3948_394867

/-- Given that x varies directly as the fourth power of y and y varies as the cube root of z,
    prove that x varies as the (4/3)th power of z. -/
theorem x_varies_as_z_to_four_thirds
  (h1 : ∃ (k : ℝ), ∀ (x y : ℝ), x = k * y^4)
  (h2 : ∃ (j : ℝ), ∀ (y z : ℝ), y = j * z^(1/3))
  : ∃ (m : ℝ), ∀ (x z : ℝ), x = m * z^(4/3) := by
  sorry

end NUMINAMATH_CALUDE_x_varies_as_z_to_four_thirds_l3948_394867


namespace NUMINAMATH_CALUDE_loan_amounts_correct_l3948_394878

-- Define the total loan amount in tens of thousands of yuan
def total_loan : ℝ := 68

-- Define the total annual interest in tens of thousands of yuan
def total_interest : ℝ := 8.42

-- Define the annual interest rate for Type A loan
def rate_A : ℝ := 0.12

-- Define the annual interest rate for Type B loan
def rate_B : ℝ := 0.13

-- Define the amount of Type A loan in tens of thousands of yuan
def loan_A : ℝ := 42

-- Define the amount of Type B loan in tens of thousands of yuan
def loan_B : ℝ := 26

theorem loan_amounts_correct : 
  loan_A + loan_B = total_loan ∧ 
  rate_A * loan_A + rate_B * loan_B = total_interest := by
  sorry

end NUMINAMATH_CALUDE_loan_amounts_correct_l3948_394878


namespace NUMINAMATH_CALUDE_ellipse_equation_l3948_394808

/-- An ellipse with one focus at (1, 0) and eccentricity √2/2 has the equation x^2/2 + y^2 = 1 -/
theorem ellipse_equation (e : ℝ × ℝ → Prop) :
  (∃ (a b c : ℝ), 
    -- One focus is at (1, 0)
    c = 1 ∧
    -- Eccentricity is √2/2
    c / a = Real.sqrt 2 / 2 ∧
    -- Standard form of ellipse equation
    b^2 = a^2 - c^2 ∧
    (∀ (x y : ℝ), e (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1)) →
  (∀ (x y : ℝ), e (x, y) ↔ x^2 / 2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3948_394808


namespace NUMINAMATH_CALUDE_least_lcm_a_c_l3948_394828

theorem least_lcm_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 12) (h2 : Nat.lcm b c = 15) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 20 ∧ (∀ (x y : ℕ), Nat.lcm x b = 12 → Nat.lcm b y = 15 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
sorry

end NUMINAMATH_CALUDE_least_lcm_a_c_l3948_394828


namespace NUMINAMATH_CALUDE_problem_statement_l3948_394832

theorem problem_statement (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (heq1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2012)
  (heq2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2012) :
  (a*b)^2012 - (c*d)^2012 = -2012 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3948_394832


namespace NUMINAMATH_CALUDE_probability_four_primes_in_six_rolls_l3948_394894

/-- The probability of getting exactly 4 prime numbers in 6 rolls of a fair 8-sided die -/
theorem probability_four_primes_in_six_rolls (die : Finset ℕ) 
  (h_die : die = {1, 2, 3, 4, 5, 6, 7, 8}) 
  (h_prime : {n ∈ die | Nat.Prime n} = {2, 3, 5, 7}) : 
  (Nat.choose 6 4 * (4 / 8)^4 * (4 / 8)^2) = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_primes_in_six_rolls_l3948_394894


namespace NUMINAMATH_CALUDE_base3_sum_equality_l3948_394838

/-- Converts a list of digits in base 3 to its decimal representation -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 3 * acc) 0

/-- Converts a decimal number to its base 3 representation -/
def toBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 3) ((m % 3) :: acc)
    go n []

/-- The sum of the given numbers in base 3 is equal to 112010 in base 3 -/
theorem base3_sum_equality : 
  let a := [2]
  let b := [1, 1]
  let c := [2, 0, 2]
  let d := [1, 0, 0, 2]
  let e := [2, 2, 1, 1, 1]
  let sum := [0, 1, 0, 2, 1, 1]
  toBase3 (toDecimal a + toDecimal b + toDecimal c + toDecimal d + toDecimal e) = sum := by
  sorry

end NUMINAMATH_CALUDE_base3_sum_equality_l3948_394838


namespace NUMINAMATH_CALUDE_speed_increase_from_weight_cut_l3948_394892

/-- Proves that the speed increase from weight cut is 10 mph given the initial conditions --/
theorem speed_increase_from_weight_cut 
  (original_speed : ℝ) 
  (supercharge_increase_percent : ℝ)
  (final_speed : ℝ) :
  original_speed = 150 →
  supercharge_increase_percent = 30 →
  final_speed = 205 →
  final_speed - (original_speed * (1 + supercharge_increase_percent / 100)) = 10 := by
sorry

end NUMINAMATH_CALUDE_speed_increase_from_weight_cut_l3948_394892


namespace NUMINAMATH_CALUDE_truncated_cube_edges_l3948_394813

/-- Represents a cube with truncated corners -/
structure TruncatedCube where
  initialEdges : Nat
  vertices : Nat
  newEdgesPerVertex : Nat

/-- Calculates the total number of edges in a truncated cube -/
def totalEdges (c : TruncatedCube) : Nat :=
  c.initialEdges + c.vertices * c.newEdgesPerVertex

/-- Theorem stating that a cube with truncated corners has 36 edges -/
theorem truncated_cube_edges :
  ∀ (c : TruncatedCube),
  c.initialEdges = 12 ∧ c.vertices = 8 ∧ c.newEdgesPerVertex = 3 →
  totalEdges c = 36 := by
  sorry

end NUMINAMATH_CALUDE_truncated_cube_edges_l3948_394813


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l3948_394827

-- Part 1
theorem part_one : 12 - (-11) - 1 = 22 := by sorry

-- Part 2
theorem part_two : -1^4 / (-3)^2 / (9/5) = -5/81 := by sorry

-- Part 3
theorem part_three : -8 * (1/2 - 3/4 + 5/8) = -3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l3948_394827


namespace NUMINAMATH_CALUDE_average_monthly_bill_l3948_394843

/-- The average monthly bill for a family over 6 months, given the average for the first 4 months and the last 2 months. -/
theorem average_monthly_bill (avg_first_four : ℝ) (avg_last_two : ℝ) : 
  avg_first_four = 30 → avg_last_two = 24 → 
  (4 * avg_first_four + 2 * avg_last_two) / 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_bill_l3948_394843


namespace NUMINAMATH_CALUDE_inequality_proof_l3948_394846

theorem inequality_proof (x : ℝ) (h1 : 0 < x) (h2 : x < 20) :
  Real.sqrt x + Real.sqrt (20 - x) ≤ 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3948_394846


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_865_over_21_l3948_394885

theorem sqrt_fraction_sum_equals_sqrt_865_over_21 :
  Real.sqrt (9 / 49 + 16 / 9) = Real.sqrt 865 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_equals_sqrt_865_over_21_l3948_394885


namespace NUMINAMATH_CALUDE_max_min_f_l3948_394834

def f (x y : ℝ) : ℝ := 3 * |x + y| + |4 * y + 9| + |7 * y - 3 * x - 18|

theorem max_min_f :
  ∀ x y : ℝ, x^2 + y^2 ≤ 5 →
  (∀ x' y' : ℝ, x'^2 + y'^2 ≤ 5 → f x' y' ≤ f x y) → f x y = 27 + 6 * Real.sqrt 5 ∧
  (∀ x' y' : ℝ, x'^2 + y'^2 ≤ 5 → f x y ≤ f x' y') → f x y = 27 - 3 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_max_min_f_l3948_394834


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l3948_394819

-- Define the sets P and Q
def P : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def Q : Set ℝ := {x | (1 + x) / (x - 3) ≤ 0}

-- State the theorem
theorem union_of_P_and_Q : P ∪ Q = {x : ℝ | -2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l3948_394819


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_relation_l3948_394806

/-- Given a right triangle with sides 15, 36, and 39, and a circumscribed circle,
    where an altitude from the right angle divides one non-triangular region into
    areas A and B, and C is the largest non-triangular region, prove that A + B + 270 = C -/
theorem circumscribed_circle_area_relation (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  (15 : ℝ) ^ 2 + 36 ^ 2 = 39 ^ 2 →
  A < B →
  B < C →
  A + B + 270 = C := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_relation_l3948_394806


namespace NUMINAMATH_CALUDE_select_parents_count_l3948_394864

/-- The number of ways to select 4 parents out of 12 (6 couples), 
    such that exactly one pair of the chosen 4 are a couple -/
def selectParents : ℕ := sorry

/-- The total number of couples -/
def totalCouples : ℕ := 6

/-- The total number of parents -/
def totalParents : ℕ := 12

/-- The number of parents to be selected -/
def parentsToSelect : ℕ := 4

theorem select_parents_count : 
  selectParents = 240 := by sorry

end NUMINAMATH_CALUDE_select_parents_count_l3948_394864


namespace NUMINAMATH_CALUDE_factorization_equality_l3948_394820

/-- Proves that the factorization of 3x(x - 5) + 4(x - 5) - 2x^2 is (x - 15)(x + 4) for all real x -/
theorem factorization_equality (x : ℝ) : 3*x*(x - 5) + 4*(x - 5) - 2*x^2 = (x - 15)*(x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3948_394820


namespace NUMINAMATH_CALUDE_problem_G2_1_l3948_394817

theorem problem_G2_1 (a : ℚ) :
  137 / a = 0.1234234234235 → a = 1110 := by sorry

end NUMINAMATH_CALUDE_problem_G2_1_l3948_394817


namespace NUMINAMATH_CALUDE_harry_cookies_per_batch_l3948_394865

/-- Calculates the number of cookies in a batch given the total chips, number of batches, and chips per cookie. -/
def cookies_per_batch (total_chips : ℕ) (num_batches : ℕ) (chips_per_cookie : ℕ) : ℕ :=
  (total_chips / num_batches) / chips_per_cookie

/-- Proves that the number of cookies in a batch is 3 given the specified conditions. -/
theorem harry_cookies_per_batch :
  cookies_per_batch 81 3 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_harry_cookies_per_batch_l3948_394865


namespace NUMINAMATH_CALUDE_tom_payment_is_nine_l3948_394805

/-- The original price of the rare robot in dollars -/
def original_price : ℝ := 3

/-- The multiplier for the selling price -/
def price_multiplier : ℝ := 3

/-- The amount Tom should pay in dollars -/
def tom_payment : ℝ := original_price * price_multiplier

/-- Theorem stating that Tom should pay $9.00 for the rare robot -/
theorem tom_payment_is_nine : tom_payment = 9 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_is_nine_l3948_394805


namespace NUMINAMATH_CALUDE_f_positive_iff_l3948_394829

/-- The function f(x) = 2x + 5 -/
def f (x : ℝ) : ℝ := 2 * x + 5

/-- Theorem: f(x) > 0 if and only if x > -5/2 -/
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x > -5/2 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_iff_l3948_394829


namespace NUMINAMATH_CALUDE_potato_chips_count_l3948_394881

/-- The number of potato chips one potato can make -/
def potato_chips_per_potato (total_potatoes wedge_potatoes wedges_per_potato : ℕ) 
  (chip_wedge_difference : ℕ) : ℕ :=
let remaining_potatoes := total_potatoes - wedge_potatoes
let chip_potatoes := remaining_potatoes / 2
let total_wedges := wedge_potatoes * wedges_per_potato
let total_chips := total_wedges + chip_wedge_difference
total_chips / chip_potatoes

/-- Theorem stating that one potato can make 20 potato chips under given conditions -/
theorem potato_chips_count : 
  potato_chips_per_potato 67 13 8 436 = 20 := by
  sorry

end NUMINAMATH_CALUDE_potato_chips_count_l3948_394881


namespace NUMINAMATH_CALUDE_adjacent_supplementary_angles_l3948_394887

theorem adjacent_supplementary_angles (angle1 angle2 : ℝ) : 
  (angle1 + angle2 = 180) → (angle1 = 80) → (angle2 = 100) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_supplementary_angles_l3948_394887


namespace NUMINAMATH_CALUDE_two_roots_theorem_l3948_394836

theorem two_roots_theorem (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (x₁ - a) * (x₁ - b) + (x₁ - a) * (x₁ - c) + (x₁ - b) * (x₁ - c) = 0 ∧
    (x₂ - a) * (x₂ - b) + (x₂ - a) * (x₂ - c) + (x₂ - b) * (x₂ - c) = 0 ∧
    a < x₁ ∧ x₁ < b ∧ b < x₂ ∧ x₂ < c :=
by sorry

end NUMINAMATH_CALUDE_two_roots_theorem_l3948_394836


namespace NUMINAMATH_CALUDE_power_function_range_l3948_394890

def power_function (x : ℝ) (m : ℕ+) : ℝ := x^(3*m.val - 9)

theorem power_function_range (m : ℕ+) 
  (h1 : ∀ (x : ℝ), x > 0 → ∀ (y : ℝ), y > x → power_function y m < power_function x m)
  (h2 : ∀ (x : ℝ), power_function x m = power_function (-x) m) :
  {a : ℝ | (a + 1)^(m.val/3) < (3 - 2*a)^(m.val/3)} = {a : ℝ | a < 2/3} := by
sorry

end NUMINAMATH_CALUDE_power_function_range_l3948_394890


namespace NUMINAMATH_CALUDE_canteen_distance_l3948_394826

/-- Given a right triangle with legs 450 and 600 rods, prove that a point on the hypotenuse 
    that is equidistant from both ends of the hypotenuse is 468.75 rods from each end. -/
theorem canteen_distance (a b c x : ℝ) (h1 : a = 450) (h2 : b = 600) 
  (h3 : c^2 = a^2 + b^2) (h4 : x^2 = a^2 + (b - x)^2) : x = 468.75 := by
  sorry

end NUMINAMATH_CALUDE_canteen_distance_l3948_394826


namespace NUMINAMATH_CALUDE_prime_counting_upper_bound_l3948_394870

open Real

/-- The prime counting function π(n) -/
noncomputable def prime_counting (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n > 55, π(n) < 3 ln 2 * (n / ln n) -/
theorem prime_counting_upper_bound (n : ℕ) (h : n > 55) :
  (prime_counting n : ℝ) < 3 * log 2 * (n / log n) := by
  sorry

end NUMINAMATH_CALUDE_prime_counting_upper_bound_l3948_394870


namespace NUMINAMATH_CALUDE_percentage_increase_problem_l3948_394822

theorem percentage_increase_problem : 
  let initial := 100
  let after_first_increase := initial * (1 + 0.2)
  let final := after_first_increase * (1 + 0.5)
  final = 180 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_problem_l3948_394822


namespace NUMINAMATH_CALUDE_ludwig_earnings_proof_l3948_394811

/-- Ludwig's weekly work schedule and earnings --/
def ludwig_weekly_earnings : ℕ :=
  let full_day_salary : ℕ := 10
  let full_days : ℕ := 4
  let half_days : ℕ := 3
  let full_day_earnings : ℕ := full_day_salary * full_days
  let half_day_earnings : ℕ := (full_day_salary / 2) * half_days
  full_day_earnings + half_day_earnings

/-- Theorem: Ludwig's weekly earnings are $55 --/
theorem ludwig_earnings_proof : ludwig_weekly_earnings = 55 := by
  sorry

end NUMINAMATH_CALUDE_ludwig_earnings_proof_l3948_394811


namespace NUMINAMATH_CALUDE_shell_division_impossibility_l3948_394855

theorem shell_division_impossibility : ¬ ∃ (n : ℕ), 
  (637 - n) % 3 = 0 ∧ (n + 1 : ℕ) = (637 - n) / 3 := by
  sorry

end NUMINAMATH_CALUDE_shell_division_impossibility_l3948_394855


namespace NUMINAMATH_CALUDE_celenes_borrowed_books_l3948_394879

/-- Represents the problem of determining the number of books Celine borrowed -/
theorem celenes_borrowed_books :
  let daily_charge : ℚ := 0.5
  let days_for_first_book : ℕ := 20
  let days_in_may : ℕ := 31
  let total_paid : ℚ := 41
  let num_books_whole_month : ℕ := 2

  let charge_first_book : ℚ := daily_charge * days_for_first_book
  let charge_per_whole_month_book : ℚ := daily_charge * days_in_may
  let charge_whole_month_books : ℚ := num_books_whole_month * charge_per_whole_month_book
  let total_charge : ℚ := charge_first_book + charge_whole_month_books

  total_charge = total_paid →
  num_books_whole_month + 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_celenes_borrowed_books_l3948_394879


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3948_394873

theorem arithmetic_sequence_sum : 
  ∀ (a l : ℤ) (d : ℤ) (n : ℕ),
    a = 162 →
    d = -6 →
    l = 48 →
    n > 0 →
    l = a + (n - 1) * d →
    (n : ℤ) * (a + l) / 2 = 2100 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3948_394873


namespace NUMINAMATH_CALUDE_model_parameters_l3948_394868

/-- Given a model y = c * e^(k * x) where c > 0, and its logarithmic transformation
    z = ln y resulting in the linear regression equation z = 2x - 1,
    prove that k = 2 and c = 1/e. -/
theorem model_parameters (c : ℝ) (k : ℝ) :
  c > 0 →
  (∀ x y z : ℝ, y = c * Real.exp (k * x) → z = Real.log y → z = 2 * x - 1) →
  k = 2 ∧ c = 1 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_model_parameters_l3948_394868


namespace NUMINAMATH_CALUDE_power_mod_eleven_l3948_394830

theorem power_mod_eleven : 5^2023 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l3948_394830


namespace NUMINAMATH_CALUDE_cube_sum_geq_triple_sum_products_l3948_394831

theorem cube_sum_geq_triple_sum_products
  (a b c : ℝ)
  (ha : a ≥ 0)
  (hb : b ≥ 0)
  (hc : c ≥ 0)
  (h_sum_squares : a^2 + b^2 + c^2 ≥ 3) :
  (a + b + c)^3 ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_triple_sum_products_l3948_394831


namespace NUMINAMATH_CALUDE_inverse_proposition_l3948_394889

theorem inverse_proposition : 
  (∀ a b : ℝ, a^2 + b^2 ≠ 0 → a = 0 ∧ b = 0) ↔ 
  (∀ a b : ℝ, a = 0 ∧ b = 0 → a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l3948_394889


namespace NUMINAMATH_CALUDE_length_of_24_l3948_394899

def length_of_integer (k : ℕ) : ℕ := sorry

theorem length_of_24 :
  let k : ℕ := 24
  length_of_integer k = 4 := by sorry

end NUMINAMATH_CALUDE_length_of_24_l3948_394899


namespace NUMINAMATH_CALUDE_water_in_bucket_l3948_394861

theorem water_in_bucket (initial_amount : ℝ) (poured_out : ℝ) : 
  initial_amount = 0.8 → poured_out = 0.2 → initial_amount - poured_out = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_water_in_bucket_l3948_394861


namespace NUMINAMATH_CALUDE_set_membership_implies_m_value_l3948_394893

theorem set_membership_implies_m_value (m : ℚ) : 
  let A : Set ℚ := {m + 2, 2 * m^2 + m}
  3 ∈ A → m = -3/2 := by sorry

end NUMINAMATH_CALUDE_set_membership_implies_m_value_l3948_394893


namespace NUMINAMATH_CALUDE_heartsuit_five_three_l3948_394860

def heartsuit (x y : ℤ) : ℤ := 4 * x - 2 * y

theorem heartsuit_five_three : heartsuit 5 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_five_three_l3948_394860


namespace NUMINAMATH_CALUDE_brendan_grass_cutting_l3948_394821

/-- Brendan's grass cutting capacity over a week -/
theorem brendan_grass_cutting (initial_capacity : ℝ) (increase_percentage : ℝ) (days_in_week : ℕ) :
  initial_capacity = 8 →
  increase_percentage = 0.5 →
  days_in_week = 7 →
  (initial_capacity + initial_capacity * increase_percentage) * days_in_week = 84 := by
  sorry

end NUMINAMATH_CALUDE_brendan_grass_cutting_l3948_394821


namespace NUMINAMATH_CALUDE_monkey_to_snake_ratio_l3948_394824

/-- Represents the number of animals in John's zoo --/
structure ZooAnimals where
  snakes : ℕ
  monkeys : ℕ
  lions : ℕ
  pandas : ℕ
  dogs : ℕ

/-- Conditions for John's zoo --/
def zoo_conditions (z : ZooAnimals) : Prop :=
  z.snakes = 15 ∧
  z.lions = z.monkeys - 5 ∧
  z.pandas = z.lions + 8 ∧
  z.dogs * 3 = z.pandas ∧
  z.snakes + z.monkeys + z.lions + z.pandas + z.dogs = 114

/-- Theorem stating the ratio of monkeys to snakes is 2:1 --/
theorem monkey_to_snake_ratio (z : ZooAnimals) (h : zoo_conditions z) :
  z.monkeys = 2 * z.snakes := by
  sorry

end NUMINAMATH_CALUDE_monkey_to_snake_ratio_l3948_394824


namespace NUMINAMATH_CALUDE_x_sixth_power_equals_one_l3948_394866

theorem x_sixth_power_equals_one (x : ℝ) (h : 1 + x + x^2 + x^3 + x^4 + x^5 = 0) : x^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_power_equals_one_l3948_394866


namespace NUMINAMATH_CALUDE_store_sales_l3948_394852

-- Define the prices and quantities of each pencil type
def eraser_price : ℚ := 0.8
def regular_price : ℚ := 0.5
def short_price : ℚ := 0.4
def mechanical_price : ℚ := 1.2
def novelty_price : ℚ := 1.5

def eraser_quantity : ℕ := 200
def regular_quantity : ℕ := 40
def short_quantity : ℕ := 35
def mechanical_quantity : ℕ := 25
def novelty_quantity : ℕ := 15

-- Define the total sales function
def total_sales : ℚ :=
  eraser_price * eraser_quantity +
  regular_price * regular_quantity +
  short_price * short_quantity +
  mechanical_price * mechanical_quantity +
  novelty_price * novelty_quantity

-- Theorem statement
theorem store_sales : total_sales = 246.5 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_l3948_394852


namespace NUMINAMATH_CALUDE_line_parametric_to_cartesian_l3948_394849

/-- Given a line with parametric equations x = 1 + t/2 and y = 2 + (√3/2)t,
    its Cartesian equation is √3x - y + 2 - √3 = 0 --/
theorem line_parametric_to_cartesian :
  ∀ (x y t : ℝ),
  (x = 1 + t / 2 ∧ y = 2 + (Real.sqrt 3 / 2) * t) ↔
  (Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_parametric_to_cartesian_l3948_394849


namespace NUMINAMATH_CALUDE_jacket_purchase_price_l3948_394876

theorem jacket_purchase_price 
  (markup_rate : ℝ)
  (discount_rate : ℝ)
  (gross_profit : ℝ)
  (h_markup : markup_rate = 0.4)
  (h_discount : discount_rate = 0.2)
  (h_profit : gross_profit = 16) :
  ∃ (purchase_price selling_price : ℝ),
    selling_price = purchase_price + markup_rate * selling_price ∧
    gross_profit = (1 - discount_rate) * selling_price - purchase_price ∧
    purchase_price = 48 := by
  sorry

end NUMINAMATH_CALUDE_jacket_purchase_price_l3948_394876


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3948_394842

/-- The y-coordinate of the vertex of the parabola y = -2x^2 + 16x + 72 is 104 -/
theorem parabola_vertex_y_coordinate :
  let f (x : ℝ) := -2 * x^2 + 16 * x + 72
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≤ f x₀ ∧ f x₀ = 104 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3948_394842


namespace NUMINAMATH_CALUDE_wire_ratio_l3948_394818

theorem wire_ratio (a b : ℝ) (h : a > 0) (k : b > 0) : 
  (a^2 / 16 = b^2 / (4 * Real.pi)) → a / b = 2 / Real.sqrt Real.pi := by
sorry

end NUMINAMATH_CALUDE_wire_ratio_l3948_394818


namespace NUMINAMATH_CALUDE_arrange_digits_eq_sixteen_l3948_394816

/-- The number of ways to arrange the digits of 45,550 to form a 5-digit number, where numbers cannot begin with 0 -/
def arrange_digits : ℕ :=
  let digits : Multiset ℕ := {0, 4, 5, 5, 5}
  let non_zero_positions := 4  -- Number of valid positions for 0 (2nd to 5th)
  let remaining_digits := 4    -- Number of digits to arrange after placing 0
  let repeated_digit := 3      -- Number of 5's
  non_zero_positions * (remaining_digits.factorial / repeated_digit.factorial)

theorem arrange_digits_eq_sixteen : arrange_digits = 16 := by
  sorry

end NUMINAMATH_CALUDE_arrange_digits_eq_sixteen_l3948_394816


namespace NUMINAMATH_CALUDE_ninety_degrees_possible_l3948_394858

-- Define a pentagon with angles in arithmetic progression
def Pentagon (a d : ℝ) : Prop :=
  a > 60 ∧  -- smallest angle > 60 degrees
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 540  -- sum of angles in pentagon

-- Theorem statement
theorem ninety_degrees_possible (a d : ℝ) :
  Pentagon a d → ∃ k : ℕ, k < 5 ∧ a + k*d = 90 := by
  sorry


end NUMINAMATH_CALUDE_ninety_degrees_possible_l3948_394858


namespace NUMINAMATH_CALUDE_satisfactory_grade_fraction_l3948_394825

/-- Represents the grades in a science class -/
inductive Grade
  | A
  | B
  | C
  | D
  | F

/-- Returns true if the grade is satisfactory (A, B, or C) -/
def isSatisfactory (g : Grade) : Bool :=
  match g with
  | Grade.A => true
  | Grade.B => true
  | Grade.C => true
  | _ => false

/-- Represents the distribution of grades in the class -/
def gradeDistribution : List (Grade × Nat) :=
  [(Grade.A, 8), (Grade.B, 6), (Grade.C, 4), (Grade.D, 2), (Grade.F, 6)]

/-- Theorem: The fraction of satisfactory grades is 9/13 -/
theorem satisfactory_grade_fraction :
  let totalGrades := (gradeDistribution.map (·.2)).sum
  let satisfactoryGrades := (gradeDistribution.filter (isSatisfactory ·.1)).map (·.2) |>.sum
  (satisfactoryGrades : ℚ) / totalGrades = 9 / 13 := by
  sorry


end NUMINAMATH_CALUDE_satisfactory_grade_fraction_l3948_394825


namespace NUMINAMATH_CALUDE_steve_blank_questions_l3948_394823

def total_questions : ℕ := 60
def word_problems : ℕ := 20
def add_sub_problems : ℕ := 25
def algebra_problems : ℕ := 10
def geometry_problems : ℕ := 5

def steve_word : ℕ := 15
def steve_add_sub : ℕ := 22
def steve_algebra : ℕ := 8
def steve_geometry : ℕ := 3

theorem steve_blank_questions :
  total_questions - (steve_word + steve_add_sub + steve_algebra + steve_geometry) = 12 :=
by sorry

end NUMINAMATH_CALUDE_steve_blank_questions_l3948_394823


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3948_394803

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x| ≥ a * x) → |a| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3948_394803


namespace NUMINAMATH_CALUDE_solve_for_x_l3948_394863

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 10) (h2 : y = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3948_394863


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l3948_394888

theorem complex_arithmetic_equality : (469157 * 9999)^2 / 53264 + 3758491 = 413303758491 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l3948_394888


namespace NUMINAMATH_CALUDE_afternoon_sales_proof_l3948_394833

/-- A salesman sells pears in the morning and afternoon. -/
structure PearSales where
  morning : ℝ
  afternoon : ℝ

/-- The total amount of pears sold in a day. -/
def total_sales (s : PearSales) : ℝ := s.morning + s.afternoon

/-- Theorem: Given a salesman who sold twice as much pears in the afternoon than in the morning,
    and sold 390 kilograms in total that day, the amount sold in the afternoon is 260 kilograms. -/
theorem afternoon_sales_proof (s : PearSales) 
    (h1 : s.afternoon = 2 * s.morning) 
    (h2 : total_sales s = 390) : 
    s.afternoon = 260 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_sales_proof_l3948_394833


namespace NUMINAMATH_CALUDE_positive_root_of_cubic_l3948_394877

theorem positive_root_of_cubic (x : ℝ) : 
  x = 2 + Real.sqrt 3 → x > 0 ∧ x^3 - 4*x^2 - 2*x - Real.sqrt 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_root_of_cubic_l3948_394877


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3948_394815

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence satisfying
    certain conditions, the 9th term equals 0. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_third : a 3 = 6)
  (h_sum : a 1 + a 11 = 6) :
  a 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3948_394815


namespace NUMINAMATH_CALUDE_union_of_sets_l3948_394807

theorem union_of_sets : 
  let A : Set ℕ := {1, 3}
  let B : Set ℕ := {1, 2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3948_394807


namespace NUMINAMATH_CALUDE_negation_equivalence_l3948_394898

variable (m : ℝ)

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - m*x - m < 0) ↔ (∀ x : ℝ, x^2 - m*x - m ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3948_394898


namespace NUMINAMATH_CALUDE_correlated_relationships_l3948_394859

/-- Represents a relationship between two variables -/
structure Relationship where
  has_correlation : Bool

/-- The relationship between carbon content in molten steel and smelting time -/
def steel_relationship : Relationship :=
  ⟨true⟩

/-- The relationship between a point on a curve and its coordinates -/
def curve_point_relationship : Relationship :=
  ⟨false⟩

/-- The relationship between citrus yield and temperature -/
def citrus_yield_relationship : Relationship :=
  ⟨true⟩

/-- The relationship between tree cross-section diameter and height -/
def tree_relationship : Relationship :=
  ⟨true⟩

/-- The relationship between a person's age and wealth -/
def age_wealth_relationship : Relationship :=
  ⟨true⟩

/-- The list of all relationships -/
def all_relationships : List Relationship :=
  [steel_relationship, curve_point_relationship, citrus_yield_relationship, tree_relationship, age_wealth_relationship]

theorem correlated_relationships :
  (all_relationships.filter (·.has_correlation)).length = 4 :=
sorry

end NUMINAMATH_CALUDE_correlated_relationships_l3948_394859


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l3948_394874

/-- Calculates the additional charge per 2/5 of a mile for a taxi service -/
theorem taxi_charge_calculation (initial_fee : ℚ) (total_distance : ℚ) (total_charge : ℚ) :
  initial_fee = 2.05 →
  total_distance = 3.6 →
  total_charge = 5.20 →
  (total_charge - initial_fee) / (total_distance / (2/5)) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_calculation_l3948_394874


namespace NUMINAMATH_CALUDE_largest_consecutive_integers_sum_sixty_consecutive_integers_sum_largest_n_is_sixty_l3948_394884

theorem largest_consecutive_integers_sum (n : ℕ) : 
  (∃ a : ℕ, a > 0 ∧ n * (2 * a + n - 1) = 4020) → n ≤ 60 :=
by
  sorry

theorem sixty_consecutive_integers_sum : 
  ∃ a : ℕ, a > 0 ∧ 60 * (2 * a + 60 - 1) = 4020 :=
by
  sorry

theorem largest_n_is_sixty : 
  ∀ n : ℕ, (∃ a : ℕ, a > 0 ∧ n * (2 * a + n - 1) = 4020) → n ≤ 60 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_integers_sum_sixty_consecutive_integers_sum_largest_n_is_sixty_l3948_394884


namespace NUMINAMATH_CALUDE_sequence_inequality_l3948_394839

theorem sequence_inequality (n : ℕ) : n / (n + 2) < (n + 1) / (n + 3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3948_394839


namespace NUMINAMATH_CALUDE_prove_some_number_l3948_394886

theorem prove_some_number (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 35 * some_number * 35) :
  some_number = 21 := by
sorry

end NUMINAMATH_CALUDE_prove_some_number_l3948_394886


namespace NUMINAMATH_CALUDE_power_sum_equation_l3948_394837

theorem power_sum_equation : 
  let x : ℚ := 1/2
  2^(0 : ℤ) + x^(-2 : ℤ) = 5 := by sorry

end NUMINAMATH_CALUDE_power_sum_equation_l3948_394837


namespace NUMINAMATH_CALUDE_volumetric_contraction_of_mixed_liquids_l3948_394841

/-- Proves that the volumetric contraction when mixing two liquids with given properties is 21 cm³ -/
theorem volumetric_contraction_of_mixed_liquids :
  let density1 : ℝ := 1.7
  let mass1 : ℝ := 400
  let density2 : ℝ := 1.2
  let mass2 : ℝ := 600
  let total_mass : ℝ := mass1 + mass2
  let mixed_density : ℝ := 1.4
  let volume1 : ℝ := mass1 / density1
  let volume2 : ℝ := mass2 / density2
  let total_volume : ℝ := volume1 + volume2
  let actual_volume : ℝ := total_mass / mixed_density
  let contraction : ℝ := total_volume - actual_volume
  contraction = 21 := by sorry

end NUMINAMATH_CALUDE_volumetric_contraction_of_mixed_liquids_l3948_394841


namespace NUMINAMATH_CALUDE_log_27_3_l3948_394845

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_27_3 : log 27 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_log_27_3_l3948_394845


namespace NUMINAMATH_CALUDE_inverse_variation_problems_l3948_394856

/-- Two real numbers vary inversely if their product is constant -/
def VaryInversely (r s : ℝ) : Prop :=
  ∃ k : ℝ, ∀ r' s', r' * s' = k

theorem inverse_variation_problems
  (h : VaryInversely r s)
  (h1 : r = 1500 ↔ s = 0.25) :
  (r = 3000 → s = 0.125) ∧ (s = 0.15 → r = 2500) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problems_l3948_394856


namespace NUMINAMATH_CALUDE_highest_score_l3948_394809

theorem highest_score (total_innings : ℕ) (overall_average : ℚ) (score_difference : ℕ) (average_without_extremes : ℚ) :
  total_innings = 46 →
  overall_average = 63 →
  score_difference = 150 →
  average_without_extremes = 58 →
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (total_innings : ℚ) * overall_average = (total_innings - 2 : ℚ) * average_without_extremes + highest_score + lowest_score ∧
    highest_score = 248 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_l3948_394809


namespace NUMINAMATH_CALUDE_average_after_17th_is_40_l3948_394882

/-- Represents a batsman's performance -/
structure Batsman where
  totalRunsBefore : ℕ  -- Total runs before the 17th inning
  inningsBefore : ℕ    -- Number of innings before the 17th inning (16)
  runsIn17th : ℕ       -- Runs scored in the 17th inning (88)
  averageIncrease : ℕ  -- Increase in average after 17th inning (3)

/-- Calculate the average score after the 17th inning -/
def averageAfter17th (b : Batsman) : ℚ :=
  (b.totalRunsBefore + b.runsIn17th) / (b.inningsBefore + 1)

/-- The main theorem to prove -/
theorem average_after_17th_is_40 (b : Batsman) 
    (h1 : b.inningsBefore = 16)
    (h2 : b.runsIn17th = 88) 
    (h3 : b.averageIncrease = 3)
    (h4 : averageAfter17th b = (b.totalRunsBefore / b.inningsBefore) + b.averageIncrease) :
  averageAfter17th b = 40 := by
  sorry


end NUMINAMATH_CALUDE_average_after_17th_is_40_l3948_394882


namespace NUMINAMATH_CALUDE_antecedent_value_l3948_394835

/-- Given a ratio of 4:6 and a consequent of 30, prove the antecedent is 20 -/
theorem antecedent_value (ratio_antecedent ratio_consequent consequent : ℕ) 
  (h1 : ratio_antecedent = 4)
  (h2 : ratio_consequent = 6)
  (h3 : consequent = 30) :
  ratio_antecedent * consequent / ratio_consequent = 20 := by
  sorry

end NUMINAMATH_CALUDE_antecedent_value_l3948_394835


namespace NUMINAMATH_CALUDE_hostel_expenditure_hostel_expenditure_proof_l3948_394862

/-- Calculates the new total expenditure of a hostel after adding more students --/
theorem hostel_expenditure 
  (initial_students : ℕ) 
  (budget_decrease : ℚ) 
  (expenditure_increase : ℚ) 
  (new_students : ℕ) : ℚ :=
  let new_total_students := initial_students + new_students
  let total_expenditure_increase := 
    (new_total_students : ℚ) * (initial_students : ℚ) * budget_decrease / initial_students + expenditure_increase
  total_expenditure_increase

/-- Proves that the new total expenditure is 5775 given the initial conditions --/
theorem hostel_expenditure_proof :
  hostel_expenditure 100 10 400 32 = 5775 := by
  sorry

end NUMINAMATH_CALUDE_hostel_expenditure_hostel_expenditure_proof_l3948_394862


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l3948_394875

theorem sum_of_squares_theorem (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_power : x^4 + y^4 + z^4 = x^6 + y^6 + z^6) :
  x^2 + y^2 + z^2 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l3948_394875


namespace NUMINAMATH_CALUDE_negative_one_in_M_l3948_394800

def M : Set ℝ := {x | x^2 - 1 = 0}

theorem negative_one_in_M : (-1 : ℝ) ∈ M := by sorry

end NUMINAMATH_CALUDE_negative_one_in_M_l3948_394800


namespace NUMINAMATH_CALUDE_queue_waiting_times_l3948_394891

/-- Represents a queue with Slowpokes and Quickies -/
structure Queue where
  m : ℕ  -- number of Slowpokes
  n : ℕ  -- number of Quickies
  a : ℕ  -- time taken by Quickies
  b : ℕ  -- time taken by Slowpokes

/-- Calculates the minimum total waiting time for a given queue -/
def min_waiting_time (q : Queue) : ℕ :=
  q.a * (q.n.choose 2) + q.a * q.m * q.n + q.b * (q.m.choose 2)

/-- Calculates the maximum total waiting time for a given queue -/
def max_waiting_time (q : Queue) : ℕ :=
  q.a * (q.n.choose 2) + q.b * q.m * q.n + q.b * (q.m.choose 2)

/-- Calculates the expected total waiting time for a given queue -/
def expected_waiting_time (q : Queue) : ℚ :=
  (q.m + q.n).choose 2 * (q.b * q.m + q.a * q.n) / (q.m + q.n)

/-- Theorem stating the properties of the queue waiting times -/
theorem queue_waiting_times (q : Queue) :
  (min_waiting_time q ≤ max_waiting_time q) ∧
  (↑(min_waiting_time q) ≤ expected_waiting_time q) ∧
  (expected_waiting_time q ≤ max_waiting_time q) :=
sorry

end NUMINAMATH_CALUDE_queue_waiting_times_l3948_394891


namespace NUMINAMATH_CALUDE_total_cards_l3948_394844

-- Define the number of people
def num_people : ℕ := 4

-- Define the number of cards each person has
def cards_per_person : ℕ := 14

-- Theorem: The total number of cards is 56
theorem total_cards : num_people * cards_per_person = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l3948_394844


namespace NUMINAMATH_CALUDE_ethyne_bond_count_l3948_394847

/-- Represents a chemical bond in a molecule -/
inductive Bond
  | Sigma
  | Pi

/-- Represents the ethyne (acetylene) molecule -/
structure Ethyne where
  /-- The number of carbon atoms in ethyne -/
  carbon_count : Nat
  /-- The number of hydrogen atoms in ethyne -/
  hydrogen_count : Nat
  /-- The structure of ethyne is linear -/
  is_linear : Bool
  /-- Each carbon atom forms a triple bond with the other carbon atom -/
  has_carbon_triple_bond : Bool
  /-- Each carbon atom forms a single bond with a hydrogen atom -/
  has_carbon_hydrogen_single_bond : Bool

/-- Counts the number of sigma bonds in ethyne -/
def count_sigma_bonds (e : Ethyne) : Nat :=
  sorry

/-- Counts the number of pi bonds in ethyne -/
def count_pi_bonds (e : Ethyne) : Nat :=
  sorry

/-- Theorem stating the number of sigma and pi bonds in ethyne -/
theorem ethyne_bond_count (e : Ethyne) :
  e.carbon_count = 2 ∧
  e.hydrogen_count = 2 ∧
  e.is_linear ∧
  e.has_carbon_triple_bond ∧
  e.has_carbon_hydrogen_single_bond →
  count_sigma_bonds e = 3 ∧ count_pi_bonds e = 2 :=
by sorry

end NUMINAMATH_CALUDE_ethyne_bond_count_l3948_394847


namespace NUMINAMATH_CALUDE_train_meeting_correct_l3948_394801

/-- Represents the properties of two trains meeting between two cities -/
structure TrainMeeting where
  normal_meet_time : ℝ  -- in hours
  early_a_distance : ℝ  -- in km
  early_b_distance : ℝ  -- in km
  early_time : ℝ        -- in hours

/-- The solution to the train meeting problem -/
def train_meeting_solution (tm : TrainMeeting) : ℝ × ℝ × ℝ :=
  let distance := 660
  let speed_a := 115
  let speed_b := 85
  (distance, speed_a, speed_b)

/-- Theorem stating that the given solution satisfies the train meeting conditions -/
theorem train_meeting_correct (tm : TrainMeeting) 
  (h1 : tm.normal_meet_time = 3 + 18/60)
  (h2 : tm.early_a_distance = 14)
  (h3 : tm.early_b_distance = 9)
  (h4 : tm.early_time = 3) :
  let (distance, speed_a, speed_b) := train_meeting_solution tm
  (speed_a + speed_b) * tm.normal_meet_time = distance ∧
  speed_a * (tm.normal_meet_time + 24/60) = distance - tm.early_a_distance + speed_b * tm.early_time ∧
  speed_b * (tm.normal_meet_time + 36/60) = distance - tm.early_b_distance + speed_a * tm.early_time :=
by
  sorry

end NUMINAMATH_CALUDE_train_meeting_correct_l3948_394801


namespace NUMINAMATH_CALUDE_range_of_a_l3948_394897

theorem range_of_a (a : ℝ) : 
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → a ∈ Set.Iic (-1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3948_394897


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3948_394869

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_first_two : a 1 + a 2 = 324
  sum_third_fourth : a 3 + a 4 = 36

/-- The theorem to be proved -/
theorem geometric_sequence_sum (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3948_394869


namespace NUMINAMATH_CALUDE_max_NPMK_is_8010_l3948_394804

/-- Represents a three-digit number MMK where M and K are digits and M = K + 1 -/
def MMK (M K : ℕ) : Prop :=
  M ≥ 1 ∧ M ≤ 9 ∧ K ≥ 0 ∧ K ≤ 8 ∧ M = K + 1

/-- Represents the result of multiplying MMK by M -/
def NPMK (M K : ℕ) : ℕ := (100 * M + 10 * M + K) * M

/-- The theorem stating that the maximum value of NPMK is 8010 -/
theorem max_NPMK_is_8010 :
  ∀ M K : ℕ, MMK M K → NPMK M K ≤ 8010 ∧ ∃ M K : ℕ, MMK M K ∧ NPMK M K = 8010 := by
  sorry

end NUMINAMATH_CALUDE_max_NPMK_is_8010_l3948_394804


namespace NUMINAMATH_CALUDE_triangle_satisfies_conditions_l3948_394802

/-- Triangle ABC with given properties --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  euler_line : ℝ → ℝ → Prop

/-- The specific triangle we're considering --/
def our_triangle : Triangle where
  A := (-4, 0)
  B := (0, 4)
  C := (0, -2)
  euler_line := fun x y => x - y + 2 = 0

/-- Theorem stating that the given triangle satisfies the conditions --/
theorem triangle_satisfies_conditions (t : Triangle) : 
  t.A = (-4, 0) ∧ 
  t.B = (0, 4) ∧ 
  t.C = (0, -2) ∧ 
  (∀ x y, t.euler_line x y ↔ x - y + 2 = 0) →
  t = our_triangle :=
sorry

end NUMINAMATH_CALUDE_triangle_satisfies_conditions_l3948_394802


namespace NUMINAMATH_CALUDE_problem_solution_l3948_394848

theorem problem_solution (n k : ℕ) : 
  (1/2)^n * (1/81)^k = 1/18^22 → k = 11 → n = 22 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3948_394848


namespace NUMINAMATH_CALUDE_liquid_X_percentage_l3948_394895

/-- The percentage of liquid X in solution P -/
def percentage_X_in_P : ℝ := sorry

/-- The percentage of liquid X in solution Q -/
def percentage_X_in_Q : ℝ := 0.015

/-- The weight of solution P in grams -/
def weight_P : ℝ := 200

/-- The weight of solution Q in grams -/
def weight_Q : ℝ := 800

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 0.013

theorem liquid_X_percentage :
  percentage_X_in_P * weight_P + percentage_X_in_Q * weight_Q =
  percentage_X_in_mixture * (weight_P + weight_Q) ∧
  percentage_X_in_P = 0.005 := by sorry

end NUMINAMATH_CALUDE_liquid_X_percentage_l3948_394895
