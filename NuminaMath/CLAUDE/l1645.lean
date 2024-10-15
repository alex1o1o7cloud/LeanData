import Mathlib

namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1645_164577

/-- Sequence sum -/
def S (n : ℕ) : ℕ := n^2

/-- Main sequence -/
def a (n : ℕ) : ℝ := 2*n - 1

/-- Arithmetic subsequence -/
def isArithmeticSubsequence (f : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, f (n + 1) - f n = d

/-- Geometric subsequence -/
def isGeometricSubsequence (f : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, f (n + 1) = q * f n

/-- Arithmetic sequence -/
def isArithmeticSequence (f : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, f (n + 1) - f n = d

/-- Geometric sequence -/
def isGeometricSequence (f : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ ∀ n : ℕ, f (n + 1) = q * f n

theorem problem_1 : isArithmeticSubsequence (λ n => a (3*n)) := by sorry

theorem problem_2 (a : ℕ → ℤ) (d : ℤ) (h1 : d ≠ 0) (h2 : isArithmeticSequence a d) 
  (h3 : a 5 = 6) (h4 : isGeometricSubsequence (λ n => a (2*n + 1))) :
  ∃ n1 : ℕ, n1 ∈ ({6, 8, 11} : Set ℕ) := by sorry

theorem problem_3 (a : ℕ → ℝ) (q : ℝ) (h1 : isGeometricSequence a q) 
  (h2 : ∃ f : ℕ → ℕ, Infinite {n : ℕ | n ∈ Set.range f} ∧ isArithmeticSubsequence (λ n => a (f n))) :
  q = -1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1645_164577


namespace NUMINAMATH_CALUDE_exponent_division_l1645_164597

theorem exponent_division (a : ℝ) (m n : ℕ) (h : m > n) :
  a^m / a^n = a^(m - n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1645_164597


namespace NUMINAMATH_CALUDE_bicycle_stock_decrease_l1645_164532

/-- The monthly decrease in bicycle stock -/
def monthly_decrease : ℕ := sorry

/-- The number of months between January 1 and October 1 -/
def months : ℕ := 9

/-- The total decrease in bicycle stock from January 1 to October 1 -/
def total_decrease : ℕ := 36

/-- Theorem stating that the monthly decrease in bicycle stock is 4 -/
theorem bicycle_stock_decrease : monthly_decrease = 4 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_stock_decrease_l1645_164532


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1645_164571

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 5)*(x - 6) = -53 + k*x + x^2) ↔ (k = -1 ∨ k = -25) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1645_164571


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_including_31_l1645_164536

theorem unique_x_with_three_prime_divisors_including_31 :
  ∀ (x n : ℕ),
    x = 8^n - 1 →
    (∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 31 ∧ q ≠ 31 ∧ x = 31 * p * q) →
    (∀ (r : ℕ), Prime r ∧ r ∣ x → r = 31 ∨ r = p ∨ r = q) →
    x = 32767 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_including_31_l1645_164536


namespace NUMINAMATH_CALUDE_vector_difference_norm_l1645_164527

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_difference_norm (a b : V)
  (ha : ‖a‖ = 6)
  (hb : ‖b‖ = 8)
  (hab : ‖a + b‖ = ‖a - b‖) :
  ‖a - b‖ = 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_norm_l1645_164527


namespace NUMINAMATH_CALUDE_car_dealership_ratio_l1645_164567

/-- Given a car dealership with economy cars, luxury cars, and sport utility vehicles,
    where the ratio of economy to luxury cars is 3:2 and the ratio of economy cars
    to sport utility vehicles is 4:1, prove that the ratio of luxury cars to sport
    utility vehicles is 8:3. -/
theorem car_dealership_ratio (E L S : ℚ) 
    (h1 : E / L = 3 / 2)
    (h2 : E / S = 4 / 1) :
    L / S = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_ratio_l1645_164567


namespace NUMINAMATH_CALUDE_average_of_5_8_N_l1645_164506

theorem average_of_5_8_N (N : ℝ) (h : 8 < N ∧ N < 20) : 
  let avg := (5 + 8 + N) / 3
  avg = 8 ∨ avg = 10 := by
sorry

end NUMINAMATH_CALUDE_average_of_5_8_N_l1645_164506


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1645_164539

/-- Proves that the cost price of an article is $975, given that it was sold at $1170 with a 20% profit. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 1170 →
  profit_percentage = 20 →
  selling_price = (100 + profit_percentage) / 100 * 975 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1645_164539


namespace NUMINAMATH_CALUDE_faucet_fill_time_l1645_164586

/-- Proves that given four faucets can fill a 120-gallon tub in 8 minutes, 
    eight faucets will fill a 30-gallon tub in 60 seconds. -/
theorem faucet_fill_time : 
  ∀ (faucets_1 faucets_2 : ℕ) 
    (tub_1 tub_2 : ℝ) 
    (time_1 : ℝ) 
    (time_2 : ℝ),
  faucets_1 = 4 →
  faucets_2 = 8 →
  tub_1 = 120 →
  tub_2 = 30 →
  time_1 = 8 →
  (faucets_1 : ℝ) * tub_2 * time_1 = faucets_2 * tub_1 * time_2 →
  time_2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_faucet_fill_time_l1645_164586


namespace NUMINAMATH_CALUDE_optimal_allocation_l1645_164574

/-- Represents the farming problem with given conditions -/
structure FarmingProblem where
  totalLand : ℝ
  riceYield : ℝ
  peanutYield : ℝ
  riceCost : ℝ
  peanutCost : ℝ
  ricePrice : ℝ
  peanutPrice : ℝ
  availableInvestment : ℝ

/-- Calculates the profit for a given allocation of land -/
def profit (p : FarmingProblem) (riceLand : ℝ) (peanutLand : ℝ) : ℝ :=
  (p.ricePrice * p.riceYield - p.riceCost) * riceLand +
  (p.peanutPrice * p.peanutYield - p.peanutCost) * peanutLand

/-- Checks if a land allocation is valid according to the problem constraints -/
def isValidAllocation (p : FarmingProblem) (riceLand : ℝ) (peanutLand : ℝ) : Prop :=
  riceLand ≥ 0 ∧ peanutLand ≥ 0 ∧
  riceLand + peanutLand ≤ p.totalLand ∧
  p.riceCost * riceLand + p.peanutCost * peanutLand ≤ p.availableInvestment

/-- The main theorem stating that the given allocation maximizes profit -/
theorem optimal_allocation (p : FarmingProblem) 
  (h : p = {
    totalLand := 2,
    riceYield := 6000,
    peanutYield := 1500,
    riceCost := 3600,
    peanutCost := 1200,
    ricePrice := 3,
    peanutPrice := 5,
    availableInvestment := 6000
  }) :
  ∀ x y, isValidAllocation p x y → profit p x y ≤ profit p (3/2) (1/2) :=
sorry


end NUMINAMATH_CALUDE_optimal_allocation_l1645_164574


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1645_164541

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
    (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
    (∀ (d e f : ℕ+), (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (d * Real.sqrt 3 + e * Real.sqrt 11) / f) → c ≤ f)) →
  a + b + c = 113 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1645_164541


namespace NUMINAMATH_CALUDE_angle_triple_complement_l1645_164599

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l1645_164599


namespace NUMINAMATH_CALUDE_function_shift_and_overlap_l1645_164551

theorem function_shift_and_overlap (f : ℝ → ℝ) :
  (∀ x, f (x - π / 12) = Real.cos (π / 2 - 2 * x)) →
  (∀ x, f x = Real.sin (2 * x - π / 6)) :=
by sorry

end NUMINAMATH_CALUDE_function_shift_and_overlap_l1645_164551


namespace NUMINAMATH_CALUDE_school_bus_distance_l1645_164547

/-- Calculates the total distance traveled by a school bus under specific conditions -/
theorem school_bus_distance : 
  let initial_velocity := 0
  let acceleration := 2
  let acceleration_time := 30
  let constant_speed_time := 20 * 60
  let deceleration := 1
  let final_velocity := acceleration * acceleration_time
  let distance_constant_speed := final_velocity * constant_speed_time
  let distance_deceleration := final_velocity^2 / (2 * deceleration)
  distance_constant_speed + distance_deceleration = 73800 := by
  sorry

end NUMINAMATH_CALUDE_school_bus_distance_l1645_164547


namespace NUMINAMATH_CALUDE_average_of_first_n_naturals_l1645_164554

theorem average_of_first_n_naturals (n : ℕ) : 
  (n * (n + 1)) / (2 * n) = 10 → n = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_n_naturals_l1645_164554


namespace NUMINAMATH_CALUDE_stratified_sampling_admin_count_l1645_164587

theorem stratified_sampling_admin_count 
  (total_employees : ℕ) 
  (admin_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 120) 
  (h2 : admin_employees = 40) 
  (h3 : sample_size = 24) : 
  ℕ :=
  by
    sorry

#check stratified_sampling_admin_count

end NUMINAMATH_CALUDE_stratified_sampling_admin_count_l1645_164587


namespace NUMINAMATH_CALUDE_range_of_a_l1645_164533

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (a * x^2 - x + 1/(16*a))

def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → Real.sqrt (2*x + 1) < 1 + a*x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → a ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1645_164533


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1645_164500

/-- The equation of a line perpendicular to 2x - y + 4 = 0 and passing through (-2, 1) is x + 2y = 0 -/
theorem perpendicular_line_equation :
  let given_line : ℝ → ℝ → Prop := λ x y => 2 * x - y + 4 = 0
  let point : ℝ × ℝ := (-2, 1)
  let perpendicular_line : ℝ → ℝ → Prop := λ x y => x + 2 * y = 0
  (∀ x y, perpendicular_line x y ↔ 
    (∃ m b, y = m * x + b ∧ 
            m * 2 = -1 ∧ 
            point.2 = m * point.1 + b)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1645_164500


namespace NUMINAMATH_CALUDE_card_sum_difference_l1645_164520

theorem card_sum_difference (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n > 4)
  (h_a : ∀ m ∈ Finset.range (2*n + 5), ⌊a m⌋ = m) :
  ∃ (i j k l : ℕ), i ∈ Finset.range (2*n + 5) ∧ 
                   j ∈ Finset.range (2*n + 5) ∧ 
                   k ∈ Finset.range (2*n + 5) ∧ 
                   l ∈ Finset.range (2*n + 5) ∧
                   i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
                   |a i + a j - a k - a l| < 1 / (n - Real.sqrt (n / 2)) :=
sorry

end NUMINAMATH_CALUDE_card_sum_difference_l1645_164520


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_large_seat_capacity_l1645_164548

/-- Represents a Ferris wheel with small and large seats -/
structure FerrisWheel where
  small_seats : Nat
  large_seats : Nat
  small_seat_capacity : Nat
  large_seat_capacity : Nat

/-- Calculates the total number of people who can ride on large seats -/
def large_seat_capacity (fw : FerrisWheel) : Nat :=
  fw.large_seats * fw.large_seat_capacity

/-- Theorem stating that the capacity of large seats in the given Ferris wheel is 84 -/
theorem paradise_park_ferris_wheel_large_seat_capacity :
  let fw := FerrisWheel.mk 3 7 16 12
  large_seat_capacity fw = 84 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_large_seat_capacity_l1645_164548


namespace NUMINAMATH_CALUDE_units_digit_of_x_l1645_164530

def has_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem units_digit_of_x (p x : ℕ) (h1 : p * x = 32^10)
  (h2 : has_units_digit p 6) (h3 : x % 4 = 0) :
  has_units_digit x 1 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_x_l1645_164530


namespace NUMINAMATH_CALUDE_equal_numbers_product_l1645_164525

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 22 →
  b = 18 →
  c = 32 →
  d = e →
  d * e = 196 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l1645_164525


namespace NUMINAMATH_CALUDE_age_puzzle_l1645_164592

/-- The age of a person satisfying a specific age-related equation --/
theorem age_puzzle : ∃ A : ℕ, 5 * (A + 5) - 5 * (A - 5) = A ∧ A = 50 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l1645_164592


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l1645_164579

theorem divisibility_by_twelve (n : Nat) : n < 10 → (3150 + n) % 12 = 0 ↔ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l1645_164579


namespace NUMINAMATH_CALUDE_product_equality_l1645_164570

theorem product_equality : 100 * 29.98 * 2.998 * 1000 = (2998 : ℝ) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1645_164570


namespace NUMINAMATH_CALUDE_kennedy_benedict_house_difference_l1645_164516

theorem kennedy_benedict_house_difference (kennedy_house : ℕ) (benedict_house : ℕ)
  (h1 : kennedy_house = 10000)
  (h2 : benedict_house = 2350) :
  kennedy_house - 4 * benedict_house = 600 :=
by sorry

end NUMINAMATH_CALUDE_kennedy_benedict_house_difference_l1645_164516


namespace NUMINAMATH_CALUDE_geometric_progression_sum_inequality_l1645_164526

/-- An increasing positive geometric progression -/
def IsIncreasingPositiveGP (b : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 1 ∧ ∀ n, b n > 0 ∧ b (n + 1) = b n * q

theorem geometric_progression_sum_inequality 
  (b : ℕ → ℝ) 
  (h_gp : IsIncreasingPositiveGP b) 
  (h_sum : b 4 + b 3 - b 2 - b 1 = 5) : 
  b 6 + b 5 ≥ 20 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_inequality_l1645_164526


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1645_164593

/-- A function f that represents the inequality to be proven -/
noncomputable def f (a b : ℝ) : ℝ := sorry

/-- Theorem stating the inequality and equality conditions -/
theorem inequality_and_equality_conditions (a b : ℝ) (ha : a ≥ 3) (hb : b ≥ 3) :
  f a b ≥ 0 ∧ (f a b = 0 ↔ (a = 3 ∧ b ≥ 3) ∨ (b = 3 ∧ a ≥ 3)) := by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1645_164593


namespace NUMINAMATH_CALUDE_teapot_teacup_discount_l1645_164590

/-- Represents the payment amount for a purchase of teapots and teacups under different discount methods -/
def payment_amount (x : ℝ) : Prop :=
  let teapot_price : ℝ := 20
  let teacup_price : ℝ := 5
  let num_teapots : ℝ := 4
  let discount_rate : ℝ := 0.92
  let y1 : ℝ := teapot_price * num_teapots + teacup_price * (x - num_teapots)
  let y2 : ℝ := (teapot_price * num_teapots + teacup_price * x) * discount_rate
  (4 ≤ x ∧ x < 34 → y1 < y2) ∧
  (x = 34 → y1 = y2) ∧
  (x > 34 → y1 > y2)

theorem teapot_teacup_discount (x : ℝ) (h : x ≥ 4) : payment_amount x := by
  sorry

end NUMINAMATH_CALUDE_teapot_teacup_discount_l1645_164590


namespace NUMINAMATH_CALUDE_salary_problem_l1645_164517

theorem salary_problem (total_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ)
  (h_total : total_salary = 5000)
  (h_a_spend : a_spend_rate = 0.95)
  (h_b_spend : b_spend_rate = 0.85)
  (h_equal_savings : (1 - a_spend_rate) * a_salary = (1 - b_spend_rate) * b_salary)
  (h_total_sum : a_salary + b_salary = total_salary) :
  a_salary = 3750 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l1645_164517


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1645_164544

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 - k) + y^2 / (k - 1) = 1

-- Theorem statement
theorem hyperbola_condition (k : ℝ) :
  k > 3 → is_hyperbola k ∧ ¬(∀ k', is_hyperbola k' → k' > 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1645_164544


namespace NUMINAMATH_CALUDE_local_max_at_two_l1645_164553

/-- The function f(x) = x(x-c)² has a local maximum at x=2 if and only if c = 6 -/
theorem local_max_at_two (c : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), x * (x - c)^2 ≤ 2 * (2 - c)^2) ↔ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_local_max_at_two_l1645_164553


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1645_164568

theorem complex_modulus_problem (z : ℂ) : 2 + z * Complex.I = z - 2 * Complex.I → Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1645_164568


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_sqrt_8_simplification_sqrt_1_3_simplification_sqrt_4_simplification_l1645_164594

-- Define what it means for a quadratic radical to be simplest
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ≠ x → (∃ n : ℕ, x = Real.sqrt n) → 
    ¬(∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x = a * Real.sqrt b ∧ b < y)

-- State the theorem
theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 6) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 8) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/3)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 4) :=
by sorry

-- Define the simplification rules
theorem sqrt_8_simplification : Real.sqrt 8 = 2 * Real.sqrt 2 := by sorry
theorem sqrt_1_3_simplification : Real.sqrt (1/3) = Real.sqrt 3 / 3 := by sorry
theorem sqrt_4_simplification : Real.sqrt 4 = 2 := by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_sqrt_8_simplification_sqrt_1_3_simplification_sqrt_4_simplification_l1645_164594


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1645_164598

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 30) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1645_164598


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l1645_164521

-- Define the function
def f (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem function_satisfies_conditions :
  (f 1 = 3) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :=
by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l1645_164521


namespace NUMINAMATH_CALUDE_civilisation_meaning_l1645_164581

/-- The meaning of a word -/
def word_meaning (word : String) : String :=
  sorry

/-- Theorem: The meaning of "civilisation (n.)" is "civilization" -/
theorem civilisation_meaning : word_meaning "civilisation (n.)" = "civilization" :=
  sorry

end NUMINAMATH_CALUDE_civilisation_meaning_l1645_164581


namespace NUMINAMATH_CALUDE_polynomial_roots_l1645_164505

theorem polynomial_roots : 
  let p (x : ℚ) := 6*x^5 + 29*x^4 - 71*x^3 - 10*x^2 + 24*x + 8
  (p (-2) = 0) ∧ 
  (p (1/2) = 0) ∧ 
  (p 1 = 0) ∧ 
  (p (4/3) = 0) ∧ 
  (p (-2/3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1645_164505


namespace NUMINAMATH_CALUDE_log_product_problem_l1645_164529

theorem log_product_problem (c d : ℕ+) : 
  (d.val - c.val - 1 = 435) →  -- Number of terms is 435
  (Real.log d.val / Real.log c.val = 3) →  -- Value of the product is 3
  (c.val + d.val = 130) := by
sorry

end NUMINAMATH_CALUDE_log_product_problem_l1645_164529


namespace NUMINAMATH_CALUDE_collinear_points_t_value_l1645_164580

/-- Given three points A, B, and C in a 2D plane, this function checks if they are collinear --/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if points A(1, 2), B(-3, 4), and C(2, t) are collinear, then t = 3/2 --/
theorem collinear_points_t_value :
  ∀ t : ℝ, are_collinear (1, 2) (-3, 4) (2, t) → t = 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_t_value_l1645_164580


namespace NUMINAMATH_CALUDE_mike_tv_and_games_time_l1645_164503

/-- Given Mike's TV and video game habits, prove the total time spent on both activities in a week. -/
theorem mike_tv_and_games_time (tv_hours_per_day : ℕ) (video_game_days_per_week : ℕ) : 
  tv_hours_per_day = 4 →
  video_game_days_per_week = 3 →
  (tv_hours_per_day * 7 + video_game_days_per_week * (tv_hours_per_day / 2)) = 34 := by
sorry


end NUMINAMATH_CALUDE_mike_tv_and_games_time_l1645_164503


namespace NUMINAMATH_CALUDE_sophomore_allocation_l1645_164576

theorem sophomore_allocation (total_students : ℕ) (sophomores : ℕ) (total_spots : ℕ) :
  total_students = 800 →
  sophomores = 260 →
  total_spots = 40 →
  (sophomores : ℚ) / total_students * total_spots = 13 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_allocation_l1645_164576


namespace NUMINAMATH_CALUDE_monica_cookies_problem_l1645_164596

/-- The number of cookies Monica's father ate -/
def father_cookies : ℕ := 6

/-- The total number of cookies Monica made -/
def total_cookies : ℕ := 30

/-- The number of cookies Monica has left -/
def remaining_cookies : ℕ := 8

theorem monica_cookies_problem :
  (∃ (f : ℕ),
    f = father_cookies ∧
    total_cookies = f + (f / 2) + (f / 2 + 2) + remaining_cookies) :=
by
  sorry

end NUMINAMATH_CALUDE_monica_cookies_problem_l1645_164596


namespace NUMINAMATH_CALUDE_sum_is_composite_l1645_164524

theorem sum_is_composite (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ (a : ℕ) + b + c + d = x * y :=
by
  sorry

end NUMINAMATH_CALUDE_sum_is_composite_l1645_164524


namespace NUMINAMATH_CALUDE_basketball_match_probabilities_l1645_164528

/-- Represents the probability of a team winning a single game -/
structure GameProbability where
  teamA : ℝ
  teamB : ℝ
  sum_to_one : teamA + teamB = 1

/-- Calculates the probability of team A winning by a score of 2 to 1 -/
def prob_A_wins_2_1 (p : GameProbability) : ℝ :=
  2 * p.teamA * p.teamB * p.teamA

/-- Calculates the probability of team B winning the match -/
def prob_B_wins (p : GameProbability) : ℝ :=
  p.teamB * p.teamB + 2 * p.teamA * p.teamB * p.teamB

/-- The main theorem stating the probabilities for the given scenario -/
theorem basketball_match_probabilities (p : GameProbability) 
  (hA : p.teamA = 0.6) (hB : p.teamB = 0.4) :
  prob_A_wins_2_1 p = 0.288 ∧ prob_B_wins p = 0.352 := by
  sorry


end NUMINAMATH_CALUDE_basketball_match_probabilities_l1645_164528


namespace NUMINAMATH_CALUDE_first_knife_price_is_50_l1645_164502

/-- Represents the daily sales data for a door-to-door salesman --/
structure SalesData where
  houses_visited : ℕ
  purchase_rate : ℚ
  expensive_knife_price : ℕ
  weekly_revenue : ℕ
  work_days : ℕ

/-- Calculates the price of the first set of knives based on the given sales data --/
def calculate_knife_price (data : SalesData) : ℚ :=
  let buyers := data.houses_visited * data.purchase_rate
  let expensive_knife_buyers := buyers / 2
  let weekly_expensive_knife_revenue := expensive_knife_buyers * data.expensive_knife_price * data.work_days
  let weekly_first_knife_revenue := data.weekly_revenue - weekly_expensive_knife_revenue
  let weekly_first_knife_sales := expensive_knife_buyers * data.work_days
  weekly_first_knife_revenue / weekly_first_knife_sales

/-- Theorem stating that the price of the first set of knives is $50 --/
theorem first_knife_price_is_50 (data : SalesData)
  (h1 : data.houses_visited = 50)
  (h2 : data.purchase_rate = 1/5)
  (h3 : data.expensive_knife_price = 150)
  (h4 : data.weekly_revenue = 5000)
  (h5 : data.work_days = 5) :
  calculate_knife_price data = 50 := by
  sorry

end NUMINAMATH_CALUDE_first_knife_price_is_50_l1645_164502


namespace NUMINAMATH_CALUDE_min_distance_theorem_l1645_164562

theorem min_distance_theorem (a b x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x + y + Real.sqrt ((a - x)^2 + (b - y)^2) ≥ Real.sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_theorem_l1645_164562


namespace NUMINAMATH_CALUDE_log_equation_solution_l1645_164565

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1645_164565


namespace NUMINAMATH_CALUDE_even_function_symmetric_about_y_axis_l1645_164543

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem even_function_symmetric_about_y_axis (f : ℝ → ℝ) (h : even_function f) :
  ∀ x y : ℝ, f x = y ↔ f (-x) = y :=
by sorry

end NUMINAMATH_CALUDE_even_function_symmetric_about_y_axis_l1645_164543


namespace NUMINAMATH_CALUDE_other_root_is_one_l1645_164578

theorem other_root_is_one (a : ℝ) : 
  (2^2 - a*2 + 2 = 0) → 
  ∃ x, x ≠ 2 ∧ x^2 - a*x + 2 = 0 ∧ x = 1 := by
sorry

end NUMINAMATH_CALUDE_other_root_is_one_l1645_164578


namespace NUMINAMATH_CALUDE_frac_5_13_150th_digit_l1645_164535

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def nth_digit_after_decimal (n d : ℕ) (k : ℕ) : ℕ := sorry

theorem frac_5_13_150th_digit :
  nth_digit_after_decimal 5 13 150 = 5 := by sorry

end NUMINAMATH_CALUDE_frac_5_13_150th_digit_l1645_164535


namespace NUMINAMATH_CALUDE_sweetest_sugar_water_l1645_164531

-- Define the initial sugar water concentration
def initial_concentration : ℚ := 25 / 125

-- Define Student A's final concentration (remains the same)
def concentration_A : ℚ := initial_concentration

-- Define Student B's added solution
def added_solution_B : ℚ := 20 / 50

-- Define Student C's added solution
def added_solution_C : ℚ := 2 / 5

-- Theorem statement
theorem sweetest_sugar_water :
  added_solution_C > concentration_A ∧
  added_solution_C > added_solution_B :=
sorry

end NUMINAMATH_CALUDE_sweetest_sugar_water_l1645_164531


namespace NUMINAMATH_CALUDE_quadratic_solution_l1645_164591

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9 : ℝ) - 36 = 0) → b = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1645_164591


namespace NUMINAMATH_CALUDE_largest_common_term_l1645_164559

def first_sequence (n : ℕ) : ℕ := 3 + 8 * n

def second_sequence (m : ℕ) : ℕ := 5 + 9 * m

theorem largest_common_term :
  ∃ (n m : ℕ),
    first_sequence n = second_sequence m ∧
    first_sequence n = 131 ∧
    first_sequence n ≤ 150 ∧
    ∀ (k l : ℕ), first_sequence k = second_sequence l → first_sequence k ≤ 150 → first_sequence k ≤ 131 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l1645_164559


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l1645_164561

theorem loss_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1500 →
  selling_price = 1290 →
  (cost_price - selling_price) / cost_price * 100 = 14 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l1645_164561


namespace NUMINAMATH_CALUDE_parabola_coefficients_l1645_164545

/-- A parabola with coefficients a, b, and c in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℚ × ℚ := sorry

/-- Check if a point lies on the parabola -/
def lies_on (p : Parabola) (x y : ℚ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Check if the parabola has a vertical axis of symmetry -/
def has_vertical_axis (p : Parabola) : Prop := sorry

theorem parabola_coefficients :
  ∀ p : Parabola,
    vertex p = (5, -3) →
    has_vertical_axis p →
    lies_on p 2 4 →
    p.a = 7/9 ∧ p.b = -70/9 ∧ p.c = 140/9 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l1645_164545


namespace NUMINAMATH_CALUDE_arithmetic_sequence_calculation_l1645_164514

theorem arithmetic_sequence_calculation : 
  let n := 2023
  let sum_to_n (k : ℕ) := k * (k + 1) / 2
  let diff_from_one_to (k : ℕ) := 1 - (sum_to_n k - 1)
  (diff_from_one_to (n - 1)) * (sum_to_n n - 1) - 
  (diff_from_one_to n) * (sum_to_n (n - 1) - 1) = n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_calculation_l1645_164514


namespace NUMINAMATH_CALUDE_point_coordinates_l1645_164584

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_coordinates :
  ∀ (x y : ℝ),
  fourth_quadrant x y →
  |x| = 3 →
  |y| = 5 →
  (x = 3 ∧ y = -5) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l1645_164584


namespace NUMINAMATH_CALUDE_overall_rate_relation_l1645_164510

/-- Given three deposit amounts and their respective interest rates, 
    this theorem proves the relation for the overall annual percentage rate. -/
theorem overall_rate_relation 
  (P1 P2 P3 : ℝ) 
  (R1 R2 R3 : ℝ) 
  (h1 : P1 * (1 + R1)^2 + P2 * (1 + R2)^2 + P3 * (1 + R3)^2 = 2442)
  (h2 : P1 * (1 + R1)^3 + P2 * (1 + R2)^3 + P3 * (1 + R3)^3 = 2926) :
  ∃ R : ℝ, (1 + R)^3 / (1 + R)^2 = 2926 / 2442 :=
sorry

end NUMINAMATH_CALUDE_overall_rate_relation_l1645_164510


namespace NUMINAMATH_CALUDE_wedding_attendance_l1645_164566

theorem wedding_attendance (actual_attendance : ℕ) (show_up_rate : ℚ) : 
  actual_attendance = 209 → show_up_rate = 95/100 → 
  ∃ expected_attendance : ℕ, expected_attendance = 220 ∧ 
  (↑actual_attendance : ℚ) = show_up_rate * expected_attendance := by
sorry

end NUMINAMATH_CALUDE_wedding_attendance_l1645_164566


namespace NUMINAMATH_CALUDE_circle_tangent_condition_l1645_164557

-- Define a circle using its equation coefficients
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

-- Define what it means for a circle to be tangent to the x-axis at the origin
def tangent_to_x_axis_at_origin (c : Circle) : Prop :=
  ∃ (y : ℝ), y ≠ 0 ∧ 0^2 + y^2 + c.D*0 + c.E*y + c.F = 0 ∧
  ∀ (x : ℝ), x ≠ 0 → (∀ (y : ℝ), x^2 + y^2 + c.D*x + c.E*y + c.F ≠ 0)

-- The main theorem
theorem circle_tangent_condition (c : Circle) :
  tangent_to_x_axis_at_origin c ↔ c.E ≠ 0 ∧ c.D = 0 ∧ c.F = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_condition_l1645_164557


namespace NUMINAMATH_CALUDE_ellipse_to_hyperbola_l1645_164542

/-- Given an ellipse with equation x²/8 + y²/5 = 1 where its foci are its vertices,
    prove that the equation of the hyperbola with foci at the vertices of the ellipse
    is x²/3 - y²/5 = 1 -/
theorem ellipse_to_hyperbola (x y : ℝ) :
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    (x^2 / 8 + y^2 / 5 = 1) ∧
    (c^2 = a^2 + b^2) ∧
    (c = 2 * a)) →
  (∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (x^2 / 3 - y^2 / 5 = 1) ∧
    (c'^2 = a'^2 + b'^2) ∧
    (c' = 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_to_hyperbola_l1645_164542


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_negative_a_l1645_164518

/-- Given a real-valued function f(x) = ax³ + ln x, prove that if there exists a positive real number x
    such that the derivative of f at x is zero, then a is negative. -/
theorem tangent_perpendicular_implies_negative_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 3 * a * x^2 + 1 / x = 0) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_negative_a_l1645_164518


namespace NUMINAMATH_CALUDE_exists_congruent_triangle_with_same_color_on_sides_l1645_164515

/-- A color type with 1992 different colors -/
inductive Color : Type
| mk : Fin 1992 → Color

/-- A point in the plane -/
structure Point : Type :=
  (x y : ℝ)

/-- A triangle in the plane -/
structure Triangle : Type :=
  (a b c : Point)

/-- A coloring of the plane -/
def Coloring : Type := Point → Color

/-- A predicate to check if a point is on a line segment -/
def OnSegment (p q r : Point) : Prop := sorry

/-- A predicate to check if two triangles are congruent -/
def Congruent (t1 t2 : Triangle) : Prop := sorry

/-- Main theorem -/
theorem exists_congruent_triangle_with_same_color_on_sides
  (coloring : Coloring)
  (all_colors_used : ∀ c : Color, ∃ p : Point, coloring p = c)
  (t : Triangle) :
  ∃ t' : Triangle, Congruent t t' ∧
    ∃ (p1 p2 p3 : Point) (c : Color),
      OnSegment p1 t'.a t'.b ∧
      OnSegment p2 t'.b t'.c ∧
      OnSegment p3 t'.c t'.a ∧
      coloring p1 = c ∧
      coloring p2 = c ∧
      coloring p3 = c :=
sorry

end NUMINAMATH_CALUDE_exists_congruent_triangle_with_same_color_on_sides_l1645_164515


namespace NUMINAMATH_CALUDE_cost_750_candies_l1645_164522

/-- The cost of buying a given number of chocolate candies with a possible discount -/
def total_cost (candies_per_box : ℕ) (box_cost : ℚ) (num_candies : ℕ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let num_boxes := (num_candies + candies_per_box - 1) / candies_per_box
  let cost_before_discount := num_boxes * box_cost
  let discount := if num_candies > discount_threshold then discount_rate * cost_before_discount else 0
  cost_before_discount - discount

/-- The total cost to buy 750 chocolate candies is $180 -/
theorem cost_750_candies :
  total_cost 30 8 750 (1/10) 500 = 180 := by
  sorry

end NUMINAMATH_CALUDE_cost_750_candies_l1645_164522


namespace NUMINAMATH_CALUDE_three_digit_numbers_from_five_cards_l1645_164573

theorem three_digit_numbers_from_five_cards : 
  let n : ℕ := 5  -- number of cards
  let r : ℕ := 3  -- number of digits in the formed number
  Nat.factorial n / Nat.factorial (n - r) = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_from_five_cards_l1645_164573


namespace NUMINAMATH_CALUDE_root_sum_ratio_l1645_164595

theorem root_sum_ratio (k₁ k₂ : ℝ) : 
  (∃ a b : ℝ, (k₁ * (a^2 - a) + a + 7 = 0 ∧ k₂ * (b^2 - b) + b + 7 = 0) ∧
              (a / b + b / a = 5 / 6)) →
  k₁ / k₂ + k₂ / k₁ = 433 / 36 := by
sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l1645_164595


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l1645_164575

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x

theorem derivative_f_at_one :
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l1645_164575


namespace NUMINAMATH_CALUDE_line_properties_l1645_164523

def line_equation (x y : ℝ) : Prop := y = -x + 5

theorem line_properties :
  let angle_with_ox : ℝ := 135
  let intersection_point : ℝ × ℝ := (0, 5)
  let point_A : ℝ × ℝ := (2, 3)
  let point_B : ℝ × ℝ := (2, -3)
  (∀ x y, line_equation x y → 
    (Real.tan (angle_with_ox * π / 180) = -1 ∧ 
     line_equation (intersection_point.1) (intersection_point.2))) ∧
  line_equation point_A.1 point_A.2 ∧
  ¬line_equation point_B.1 point_B.2 :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l1645_164523


namespace NUMINAMATH_CALUDE_expression_equality_1_expression_equality_2_l1645_164560

-- Part 1
theorem expression_equality_1 : 
  2 * Real.sin (45 * π / 180) - (π - Real.sqrt 5) ^ 0 + (1/2)⁻¹ + |Real.sqrt 2 - 1| = 2 * Real.sqrt 2 := by
  sorry

-- Part 2
theorem expression_equality_2 (a b : ℝ) : 
  (2*a + 3*b) * (3*a - 2*b) = 6*a^2 + 5*a*b - 6*b^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_1_expression_equality_2_l1645_164560


namespace NUMINAMATH_CALUDE_sam_watermelons_l1645_164537

theorem sam_watermelons (grown : ℕ) (eaten : ℕ) (h1 : grown = 4) (h2 : eaten = 3) :
  grown - eaten = 1 := by
  sorry

end NUMINAMATH_CALUDE_sam_watermelons_l1645_164537


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1645_164555

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x > 1}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem complement_union_theorem : (U \ A) ∪ B = {x : ℝ | x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1645_164555


namespace NUMINAMATH_CALUDE_min_intersection_size_l1645_164563

theorem min_intersection_size (total students_green_eyes students_own_lunch : ℕ)
  (h_total : total = 25)
  (h_green : students_green_eyes = 15)
  (h_lunch : students_own_lunch = 18)
  : ∃ (intersection : ℕ), 
    intersection ≤ students_green_eyes ∧ 
    intersection ≤ students_own_lunch ∧
    intersection ≥ students_green_eyes + students_own_lunch - total ∧
    intersection = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_intersection_size_l1645_164563


namespace NUMINAMATH_CALUDE_graduation_chairs_l1645_164556

/-- Calculates the total number of chairs needed for a graduation ceremony. -/
def chairs_needed (graduates : ℕ) (parents_per_graduate : ℕ) (teachers : ℕ) : ℕ :=
  graduates + (graduates * parents_per_graduate) + teachers + (teachers / 2)

/-- Proves that 180 chairs are needed for the given graduation ceremony. -/
theorem graduation_chairs : chairs_needed 50 2 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_graduation_chairs_l1645_164556


namespace NUMINAMATH_CALUDE_expression_evaluation_l1645_164550

theorem expression_evaluation : -3 * 5 - (-4 * -2) + (-15 * -3) / 3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1645_164550


namespace NUMINAMATH_CALUDE_probability_of_five_consecutive_heads_l1645_164507

/-- Represents a sequence of 8 coin flips -/
def CoinFlipSequence := Fin 8 → Bool

/-- Returns true if the given sequence has at least 5 consecutive heads -/
def hasAtLeastFiveConsecutiveHeads (seq : CoinFlipSequence) : Bool :=
  sorry

/-- The total number of possible outcomes when flipping a coin 8 times -/
def totalOutcomes : Nat := 2^8

/-- The number of outcomes with at least 5 consecutive heads -/
def successfulOutcomes : Nat := 13

theorem probability_of_five_consecutive_heads :
  (Nat.card {seq : CoinFlipSequence | hasAtLeastFiveConsecutiveHeads seq} : ℚ) / totalOutcomes = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_five_consecutive_heads_l1645_164507


namespace NUMINAMATH_CALUDE_quadratic_root_implies_s_value_l1645_164504

theorem quadratic_root_implies_s_value 
  (r s : ℝ) 
  (h : (4 + 3*I : ℂ) = -r/(2*2) + (r^2/(2*2)^2 - s/2).sqrt) : 
  s = 50 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_s_value_l1645_164504


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1645_164588

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1645_164588


namespace NUMINAMATH_CALUDE_mars_inhabitable_area_l1645_164546

/-- The fraction of Mars' surface that is not covered by water -/
def mars_land_fraction : ℚ := 3/5

/-- The fraction of Mars' land that is inhabitable -/
def mars_inhabitable_land_fraction : ℚ := 2/3

/-- The fraction of Mars' surface that Martians can inhabit -/
def mars_inhabitable_fraction : ℚ := mars_land_fraction * mars_inhabitable_land_fraction

theorem mars_inhabitable_area :
  mars_inhabitable_fraction = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_mars_inhabitable_area_l1645_164546


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1645_164509

/-- The quadratic function y = x^2 + ax + a - 2 -/
def f (a x : ℝ) : ℝ := x^2 + a*x + a - 2

theorem quadratic_function_properties (a : ℝ) :
  -- The function always has two distinct real roots
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
  -- The distance between the roots is minimized when a = 2
  (∀ b : ℝ, ∃ x₁ x₂ : ℝ, f b x₁ = 0 ∧ f b x₂ = 0 → 
    |x₁ - x₂| ≥ |(-2 : ℝ) - 2|) ∧
  -- When both roots are in the interval (-2, 2), a is in the interval (-2/3, 2)
  (∀ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ -2 < x₁ ∧ x₁ < 2 ∧ -2 < x₂ ∧ x₂ < 2 → 
    -2/3 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1645_164509


namespace NUMINAMATH_CALUDE_eighth_iteration_is_zero_l1645_164501

-- Define the function g based on the graph
def g : ℕ → ℕ
| 0 => 0
| 1 => 8
| 2 => 5
| 3 => 0
| 4 => 7
| 5 => 3
| 6 => 9
| 7 => 2
| 8 => 1
| 9 => 4
| _ => 0  -- Default case for numbers not explicitly shown in the graph

-- Define the iteration of g
def iterate_g (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => iterate_g n (g x)

-- Theorem statement
theorem eighth_iteration_is_zero : iterate_g 8 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_eighth_iteration_is_zero_l1645_164501


namespace NUMINAMATH_CALUDE_skew_to_common_line_relationships_l1645_164538

-- Define the concept of a line in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- or any other suitable representation
  -- This is just a placeholder structure

-- Define the concept of skew lines
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not parallel and do not intersect
  sorry

-- Define the possible positional relationships
inductive PositionalRelationship
  | Parallel
  | Intersecting
  | Skew

-- Theorem statement
theorem skew_to_common_line_relationships 
  (a b l : Line3D) 
  (ha : are_skew a l) 
  (hb : are_skew b l) : 
  ∃ (r : PositionalRelationship), 
    (r = PositionalRelationship.Parallel) ∨ 
    (r = PositionalRelationship.Intersecting) ∨ 
    (r = PositionalRelationship.Skew) :=
sorry

end NUMINAMATH_CALUDE_skew_to_common_line_relationships_l1645_164538


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1645_164585

-- Define the inequality
def inequality (x : ℝ) : Prop := (5*x + 3)/(x - 1) ≤ 3

-- Define the solution set
def solution_set : Set ℝ := {x | -3 ≤ x ∧ x < 1}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x ∧ x ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1645_164585


namespace NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l1645_164512

theorem consecutive_four_product_plus_one_is_square (x : ℤ) :
  ∃ y : ℤ, x * (x + 1) * (x + 2) * (x + 3) + 1 = y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l1645_164512


namespace NUMINAMATH_CALUDE_face_mask_profit_l1645_164534

/-- Calculate the total profit from selling face masks --/
theorem face_mask_profit (num_boxes : ℕ) (masks_per_box : ℕ) (total_cost : ℚ) (selling_price : ℚ) :
  num_boxes = 3 →
  masks_per_box = 20 →
  total_cost = 15 →
  selling_price = 1/2 →
  (num_boxes * masks_per_box : ℚ) * selling_price - total_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_face_mask_profit_l1645_164534


namespace NUMINAMATH_CALUDE_streamer_hourly_rate_l1645_164513

/-- A streamer's weekly schedule and earnings --/
structure StreamerSchedule where
  daysOff : ℕ
  hoursPerStreamDay : ℕ
  weeklyEarnings : ℕ

/-- Calculate the hourly rate of a streamer --/
def hourlyRate (s : StreamerSchedule) : ℚ :=
  s.weeklyEarnings / ((7 - s.daysOff) * s.hoursPerStreamDay)

/-- Theorem stating that given the specific conditions, the hourly rate is $10 --/
theorem streamer_hourly_rate :
  let s : StreamerSchedule := {
    daysOff := 3,
    hoursPerStreamDay := 4,
    weeklyEarnings := 160
  }
  hourlyRate s = 10 := by
  sorry

end NUMINAMATH_CALUDE_streamer_hourly_rate_l1645_164513


namespace NUMINAMATH_CALUDE_bandages_left_in_box_l1645_164569

/-- The number of bandages in a box before use -/
def initial_bandages : ℕ := 24 - 8

/-- The number of bandages used on the left knee -/
def left_knee_bandages : ℕ := 2

/-- The number of bandages used on the right knee -/
def right_knee_bandages : ℕ := 3

/-- The total number of bandages used -/
def total_used_bandages : ℕ := left_knee_bandages + right_knee_bandages

theorem bandages_left_in_box : initial_bandages - total_used_bandages = 11 := by
  sorry

end NUMINAMATH_CALUDE_bandages_left_in_box_l1645_164569


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1645_164564

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y :=
by
  use (-2)
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1645_164564


namespace NUMINAMATH_CALUDE_circular_arcs_in_regular_ngon_l1645_164572

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A point inside a regular n-gon -/
def PointInside (E : RegularNGon n) (P : ℝ × ℝ) : Prop := sorry

/-- A circular arc inside a regular n-gon -/
def CircularArcInside (E : RegularNGon n) (arc : ℝ × ℝ → ℝ × ℝ → Prop) : Prop := sorry

/-- The angle between two circular arcs at their intersection point -/
def AngleBetweenArcs (arc1 arc2 : ℝ × ℝ → ℝ × ℝ → Prop) (P : ℝ × ℝ) : ℝ := sorry

theorem circular_arcs_in_regular_ngon (n : ℕ) (E : RegularNGon n) (P₁ P₂ : ℝ × ℝ) 
  (h₁ : PointInside E P₁) (h₂ : PointInside E P₂) :
  ∃ (arc1 arc2 : ℝ × ℝ → ℝ × ℝ → Prop),
    CircularArcInside E arc1 ∧ 
    CircularArcInside E arc2 ∧
    arc1 P₁ P₂ ∧ 
    arc2 P₁ P₂ ∧
    AngleBetweenArcs arc1 arc2 P₁ ≥ (1 - 2 / n) * π ∧
    AngleBetweenArcs arc1 arc2 P₂ ≥ (1 - 2 / n) * π :=
sorry

end NUMINAMATH_CALUDE_circular_arcs_in_regular_ngon_l1645_164572


namespace NUMINAMATH_CALUDE_pet_food_difference_l1645_164540

theorem pet_food_difference (dog_food : ℕ) (cat_food : ℕ) 
  (h1 : dog_food = 600) (h2 : cat_food = 327) : 
  dog_food - cat_food = 273 := by
  sorry

end NUMINAMATH_CALUDE_pet_food_difference_l1645_164540


namespace NUMINAMATH_CALUDE_action_figure_value_l1645_164519

theorem action_figure_value (n : ℕ) (known_value : ℕ) (discount : ℕ) (total_earned : ℕ) :
  n = 5 →
  known_value = 20 →
  discount = 5 →
  total_earned = 55 →
  ∃ (other_value : ℕ),
    other_value * (n - 1) + known_value = total_earned + n * discount ∧
    other_value = 15 := by
  sorry

end NUMINAMATH_CALUDE_action_figure_value_l1645_164519


namespace NUMINAMATH_CALUDE_triangle_two_solutions_l1645_164583

theorem triangle_two_solutions (a b : ℝ) (A : ℝ) (ha : a = Real.sqrt 3) (hb : b = 3) (hA : A = π / 6) :
  (b * Real.sin A < a) ∧ (a < b) → ∃ (B C : ℝ), 0 < B ∧ 0 < C ∧ A + B + C = π ∧
  a = b * Real.sin C / Real.sin A ∧ 
  b = a * Real.sin B / Real.sin A :=
sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_l1645_164583


namespace NUMINAMATH_CALUDE_wendy_albums_l1645_164582

theorem wendy_albums (total_pictures : ℕ) (pictures_in_one_album : ℕ) (pictures_per_album : ℕ) 
  (h1 : total_pictures = 45)
  (h2 : pictures_in_one_album = 27)
  (h3 : pictures_per_album = 2) :
  (total_pictures - pictures_in_one_album) / pictures_per_album = 9 := by
  sorry

end NUMINAMATH_CALUDE_wendy_albums_l1645_164582


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_6_or_8_l1645_164511

/-- The number of four-digit numbers -/
def total_four_digit_numbers : ℕ := 9000

/-- The number of digits that are not 6 or 8 for the first digit -/
def first_digit_choices : ℕ := 7

/-- The number of digits that are not 6 or 8 for the other digits -/
def other_digit_choices : ℕ := 8

/-- The number of four-digit numbers without 6 or 8 -/
def numbers_without_6_or_8 : ℕ := first_digit_choices * other_digit_choices * other_digit_choices * other_digit_choices

theorem four_digit_numbers_with_6_or_8 :
  total_four_digit_numbers - numbers_without_6_or_8 = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_6_or_8_l1645_164511


namespace NUMINAMATH_CALUDE_total_cookies_sum_l1645_164508

/-- The number of cookies Kristy baked -/
def total_cookies : ℕ := sorry

/-- The number of cookies Kristy ate -/
def kristy_ate : ℕ := 2

/-- The number of cookies Kristy gave to her brother -/
def brother_got : ℕ := 1

/-- The number of cookies taken by the first friend -/
def first_friend_took : ℕ := 3

/-- The number of cookies taken by the second friend -/
def second_friend_took : ℕ := 5

/-- The number of cookies taken by the third friend -/
def third_friend_took : ℕ := 5

/-- The number of cookies left -/
def cookies_left : ℕ := 6

/-- Theorem stating that the total number of cookies is the sum of all distributed and remaining cookies -/
theorem total_cookies_sum : 
  total_cookies = kristy_ate + brother_got + first_friend_took + 
                  second_friend_took + third_friend_took + cookies_left :=
by sorry

end NUMINAMATH_CALUDE_total_cookies_sum_l1645_164508


namespace NUMINAMATH_CALUDE_problem_solution_l1645_164552

theorem problem_solution (x : ℝ) : (400 * 7000 : ℝ) = 28000 * (100 ^ x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1645_164552


namespace NUMINAMATH_CALUDE_magazine_subscription_l1645_164558

theorem magazine_subscription (total_students : ℕ) 
  (boys_first_half : ℕ) (girls_first_half : ℕ)
  (boys_second_half : ℕ) (girls_second_half : ℕ)
  (boys_whole_year : ℕ) 
  (h1 : total_students = 56)
  (h2 : boys_first_half = 25)
  (h3 : girls_first_half = 15)
  (h4 : boys_second_half = 26)
  (h5 : girls_second_half = 25)
  (h6 : boys_whole_year = 23) :
  girls_first_half - (girls_first_half + girls_second_half - (total_students - (boys_first_half + boys_second_half - boys_whole_year))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_magazine_subscription_l1645_164558


namespace NUMINAMATH_CALUDE_circles_tangent_line_parallel_l1645_164589

-- Define the types for points, lines, and circles
variable (Point Line Circle : Type)

-- Define the necessary relations and operations
variable (tangent_circles : Circle → Circle → Prop)
variable (tangent_circle_line : Circle → Line → Point → Prop)
variable (tangent_circles_at : Circle → Circle → Point → Prop)
variable (on_line : Point → Line → Prop)
variable (between : Point → Point → Point → Prop)
variable (intersection : Line → Line → Point)
variable (line_through : Point → Point → Line)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem circles_tangent_line_parallel 
  (Γ Γ₁ Γ₂ : Circle) (l : Line) 
  (A A₁ A₂ B₁ B₂ C D₁ D₂ : Point) :
  tangent_circles Γ Γ₁ →
  tangent_circles Γ Γ₂ →
  tangent_circles Γ₁ Γ₂ →
  tangent_circle_line Γ l A →
  tangent_circle_line Γ₁ l A₁ →
  tangent_circle_line Γ₂ l A₂ →
  tangent_circles_at Γ Γ₁ B₁ →
  tangent_circles_at Γ Γ₂ B₂ →
  tangent_circles_at Γ₁ Γ₂ C →
  between A₁ A A₂ →
  D₁ = intersection (line_through A₁ C) (line_through A₂ B₂) →
  D₂ = intersection (line_through A₂ C) (line_through A₁ B₁) →
  parallel (line_through D₁ D₂) l :=
by sorry

end NUMINAMATH_CALUDE_circles_tangent_line_parallel_l1645_164589


namespace NUMINAMATH_CALUDE_smallest_n_for_euler_totient_equation_l1645_164549

def euler_totient (n : ℕ) : ℕ := sorry

theorem smallest_n_for_euler_totient_equation : 
  ∀ n : ℕ, n > 0 → euler_totient n = (2^5 * n) / 47 → n ≥ 59895 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_euler_totient_equation_l1645_164549
