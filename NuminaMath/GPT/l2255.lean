import Mathlib

namespace NUMINAMATH_GPT_grant_total_earnings_l2255_225511

def earnings_first_month : ℕ := 350
def earnings_second_month : ℕ := 2 * earnings_first_month + 50
def earnings_third_month : ℕ := 4 * (earnings_first_month + earnings_second_month)
def total_earnings : ℕ := earnings_first_month + earnings_second_month + earnings_third_month

theorem grant_total_earnings : total_earnings = 5500 := by
  sorry

end NUMINAMATH_GPT_grant_total_earnings_l2255_225511


namespace NUMINAMATH_GPT_triangle_sides_fraction_sum_eq_one_l2255_225536

theorem triangle_sides_fraction_sum_eq_one
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2 + a * b) :
  a / (b + c) + b / (c + a) = 1 :=
sorry

end NUMINAMATH_GPT_triangle_sides_fraction_sum_eq_one_l2255_225536


namespace NUMINAMATH_GPT_average_age_of_other_9_students_l2255_225528

variable (total_students : ℕ) (total_average_age : ℝ) (group1_students : ℕ) (group1_average_age : ℝ) (age_student12 : ℝ) (group2_students : ℕ)

theorem average_age_of_other_9_students 
  (h1 : total_students = 16) 
  (h2 : total_average_age = 16) 
  (h3 : group1_students = 5) 
  (h4 : group1_average_age = 14) 
  (h5 : age_student12 = 42) 
  (h6 : group2_students = 9) : 
  (group1_students * group1_average_age + group2_students * 16 + age_student12) / total_students = total_average_age := by
  sorry

end NUMINAMATH_GPT_average_age_of_other_9_students_l2255_225528


namespace NUMINAMATH_GPT_difference_between_m_and_n_l2255_225583

theorem difference_between_m_and_n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 10 * 2^m = 2^n + 2^(n + 2)) :
  n - m = 1 :=
sorry

end NUMINAMATH_GPT_difference_between_m_and_n_l2255_225583


namespace NUMINAMATH_GPT_min_value_of_f_l2255_225559

noncomputable def f (x a : ℝ) := Real.exp (x - a) - Real.log (x + a) - 1

theorem min_value_of_f (a : ℝ) : 
  (0 < a) → (∃ x : ℝ, f x a = 0) ↔ a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l2255_225559


namespace NUMINAMATH_GPT_find_a_n_l2255_225526

def S (n : ℕ) : ℕ := 2^(n+1) - 1

def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2^n

theorem find_a_n (n : ℕ) : a_n n = if n = 1 then 3 else 2^n :=
  sorry

end NUMINAMATH_GPT_find_a_n_l2255_225526


namespace NUMINAMATH_GPT_ratio_of_crates_l2255_225575

/-
  Gabrielle sells eggs. On Monday she sells 5 crates of eggs. On Tuesday she sells 2 times as many
  crates of eggs as Monday. On Wednesday she sells 2 fewer crates than Tuesday. On Thursday she sells
  some crates of eggs. She sells a total of 28 crates of eggs for the 4 days. Prove the ratio of the 
  number of crates she sells on Thursday to the number she sells on Tuesday is 1/2.
-/

theorem ratio_of_crates 
    (mon_crates : ℕ) 
    (tue_crates : ℕ) 
    (wed_crates : ℕ) 
    (thu_crates : ℕ) 
    (total_crates : ℕ) 
    (h_mon : mon_crates = 5) 
    (h_tue : tue_crates = 2 * mon_crates) 
    (h_wed : wed_crates = tue_crates - 2) 
    (h_total : total_crates = mon_crates + tue_crates + wed_crates + thu_crates) 
    (h_total_val : total_crates = 28): 
  (thu_crates / tue_crates : ℚ) = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_crates_l2255_225575


namespace NUMINAMATH_GPT_brainiacs_like_both_l2255_225534

theorem brainiacs_like_both
  (R M B : ℕ)
  (h1 : R = 2 * M)
  (h2 : R + M - B = 96)
  (h3 : M - B = 20) : B = 18 := by
  sorry

end NUMINAMATH_GPT_brainiacs_like_both_l2255_225534


namespace NUMINAMATH_GPT_optimality_theorem_l2255_225545

def sequence_1 := "[[[a1, a2], a3], a4]" -- 22 symbols sequence
def sequence_2 := "[[a1, a2], [a3, a4]]" -- 16 symbols sequence

def optimal_sequence := sequence_2

theorem optimality_theorem : optimal_sequence = "[[a1, a2], [a3, a4]]" :=
by
  sorry

end NUMINAMATH_GPT_optimality_theorem_l2255_225545


namespace NUMINAMATH_GPT_number_subtracted_from_15n_l2255_225589

theorem number_subtracted_from_15n (m n : ℕ) (h_pos_n : 0 < n) (h_pos_m : 0 < m) (h_eq : m = 15 * n - 1) (h_remainder : m % 5 = 4) : 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_subtracted_from_15n_l2255_225589


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l2255_225508

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 = 2) 
  (h2 : a 5 = 1 / 4) : 
  ( ∃ a1 : ℝ, a n = a1 * q ^ (n - 1)) 
    :=
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l2255_225508


namespace NUMINAMATH_GPT_data_set_variance_l2255_225507

def data_set : List ℕ := [2, 4, 5, 3, 6]

noncomputable def mean (l : List ℕ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℕ) : ℝ :=
  let m : ℝ := mean l
  (l.map (fun x => (x - m) ^ 2)).sum / l.length

theorem data_set_variance : variance data_set = 2 := by
  sorry

end NUMINAMATH_GPT_data_set_variance_l2255_225507


namespace NUMINAMATH_GPT_range_of_f_l2255_225542

-- Define the function f
def f (x : ℕ) : ℤ := 2 * (x : ℤ) - 3

-- Define the domain
def domain : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the expected range
def expected_range : Finset ℤ := {-1, 1, 3, 5, 7}

-- Prove the range of f given the domain
theorem range_of_f : domain.image f = expected_range :=
  sorry

end NUMINAMATH_GPT_range_of_f_l2255_225542


namespace NUMINAMATH_GPT_tire_mileage_problem_l2255_225585

/- Definitions -/
def total_miles : ℕ := 45000
def enhancement_ratio : ℚ := 1.2
def total_tire_miles : ℚ := 180000

/- Question as theorem -/
theorem tire_mileage_problem
  (x y : ℚ)
  (h1 : y = enhancement_ratio * x)
  (h2 : 4 * x + y = total_tire_miles) :
  (x = 34615 ∧ y = 41538) :=
sorry

end NUMINAMATH_GPT_tire_mileage_problem_l2255_225585


namespace NUMINAMATH_GPT_num_positive_integers_n_l2255_225569

theorem num_positive_integers_n (n : ℕ) : 
  (∃ n, ( ∃ k : ℕ, n = 2015 * k^2 ∧ ∃ m, m^2 = 2015 * n) ∧ 
          (∃ k : ℕ, n = 2015 * k^2 ∧  ∃ l : ℕ, 2 * 2015 * k^2 = l * (1 + k^2)))
  →
  n = 5 := sorry

end NUMINAMATH_GPT_num_positive_integers_n_l2255_225569


namespace NUMINAMATH_GPT_cos_660_degrees_is_one_half_l2255_225503

noncomputable def cos_660_eq_one_half : Prop :=
  (Real.cos (660 * Real.pi / 180) = 1 / 2)

theorem cos_660_degrees_is_one_half : cos_660_eq_one_half :=
by
  sorry

end NUMINAMATH_GPT_cos_660_degrees_is_one_half_l2255_225503


namespace NUMINAMATH_GPT_problem_statement_l2255_225555

theorem problem_statement
  (c d : ℕ)
  (h_factorization : ∀ x, x^2 - 18 * x + 72 = (x - c) * (x - d))
  (h_c_nonnegative : c ≥ 0)
  (h_d_nonnegative : d ≥ 0)
  (h_c_greater_d : c > d) :
  4 * d - c = 12 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2255_225555


namespace NUMINAMATH_GPT_sum_of_cubes_ages_l2255_225516

theorem sum_of_cubes_ages (d t h : ℕ) 
  (h1 : 4 * d + t = 3 * h) 
  (h2 : 4 * h ^ 2 = 2 * d ^ 2 + t ^ 2) 
  (h3 : Nat.gcd d (Nat.gcd t h) = 1)
  : d ^ 3 + t ^ 3 + h ^ 3 = 155557 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_ages_l2255_225516


namespace NUMINAMATH_GPT_number_of_terms_in_product_l2255_225552

theorem number_of_terms_in_product 
  (a b c d e f g h i : ℕ) :
  (a + b + c + d) * (e + f + g + h + i) = 20 :=
sorry

end NUMINAMATH_GPT_number_of_terms_in_product_l2255_225552


namespace NUMINAMATH_GPT_vitya_catches_up_in_5_minutes_l2255_225578

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end NUMINAMATH_GPT_vitya_catches_up_in_5_minutes_l2255_225578


namespace NUMINAMATH_GPT_total_shirts_l2255_225538

def hazel_shirts : ℕ := 6
def razel_shirts : ℕ := 2 * hazel_shirts

theorem total_shirts : hazel_shirts + razel_shirts = 18 := by
  sorry

end NUMINAMATH_GPT_total_shirts_l2255_225538


namespace NUMINAMATH_GPT_problem_inequality_l2255_225554

theorem problem_inequality (a b : ℝ) (n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : 1 < a * b) (h_n : 2 ≤ n) :
  (a + b)^n > a^n + b^n + 2^n - 2 :=
sorry

end NUMINAMATH_GPT_problem_inequality_l2255_225554


namespace NUMINAMATH_GPT_another_divisor_l2255_225541

theorem another_divisor (n : ℕ) (h1 : n = 44402) (h2 : ∀ d ∈ [12, 48, 74, 100], (n + 2) % d = 0) : 
  199 ∣ (n + 2) := 
by 
  sorry

end NUMINAMATH_GPT_another_divisor_l2255_225541


namespace NUMINAMATH_GPT_stephanie_gas_payment_l2255_225517

variables (electricity_bill : ℕ) (gas_bill : ℕ) (water_bill : ℕ) (internet_bill : ℕ)
variables (electricity_paid : ℕ) (gas_paid_fraction : ℚ) (water_paid_fraction : ℚ) (internet_paid : ℕ)
variables (additional_gas_payment : ℕ) (remaining_payment : ℕ) (expected_remaining : ℕ)

def stephanie_budget : Prop :=
  electricity_bill = 60 ∧
  electricity_paid = 60 ∧
  gas_bill = 40 ∧
  gas_paid_fraction = 3/4 ∧
  water_bill = 40 ∧
  water_paid_fraction = 1/2 ∧
  internet_bill = 25 ∧
  internet_paid = 4 * 5 ∧
  remaining_payment = 30 ∧
  expected_remaining = 
    (gas_bill - gas_paid_fraction * gas_bill) +
    (water_bill - water_paid_fraction * water_bill) + 
    (internet_bill - internet_paid) - 
    additional_gas_payment ∧
  expected_remaining = remaining_payment

theorem stephanie_gas_payment : additional_gas_payment = 5 :=
by sorry

end NUMINAMATH_GPT_stephanie_gas_payment_l2255_225517


namespace NUMINAMATH_GPT_wheat_distribution_l2255_225512

def mill1_rate := 19 / 3 -- quintals per hour
def mill2_rate := 32 / 5 -- quintals per hour
def mill3_rate := 5     -- quintals per hour

def total_wheat := 1330 -- total wheat in quintals

theorem wheat_distribution :
    ∃ (x1 x2 x3 : ℚ), 
    x1 = 475 ∧ x2 = 480 ∧ x3 = 375 ∧ 
    x1 / mill1_rate = x2 / mill2_rate ∧ x2 / mill2_rate = x3 / mill3_rate ∧ 
    x1 + x2 + x3 = total_wheat :=
by {
  sorry
}

end NUMINAMATH_GPT_wheat_distribution_l2255_225512


namespace NUMINAMATH_GPT_Jerome_money_left_l2255_225598

-- Definitions based on conditions
def J_half := 43              -- Half of Jerome's money
def to_Meg := 8               -- Amount Jerome gave to Meg
def to_Bianca := to_Meg * 3   -- Amount Jerome gave to Bianca

-- Total initial amount of Jerome's money
def J_initial : ℕ := J_half * 2

-- Amount left after giving money to Meg
def after_Meg : ℕ := J_initial - to_Meg

-- Amount left after giving money to Bianca
def after_Bianca : ℕ := after_Meg - to_Bianca

-- Statement to be proved
theorem Jerome_money_left : after_Bianca = 54 :=
by
  sorry

end NUMINAMATH_GPT_Jerome_money_left_l2255_225598


namespace NUMINAMATH_GPT_chickens_and_rabbits_l2255_225556

theorem chickens_and_rabbits (c r : ℕ) 
    (h1 : c = 2 * r - 5)
    (h2 : 2 * c + r = 92) : ∃ c r : ℕ, (c = 2 * r - 5) ∧ (2 * c + r = 92) := 
by 
    -- proof steps
    sorry

end NUMINAMATH_GPT_chickens_and_rabbits_l2255_225556


namespace NUMINAMATH_GPT_beta_value_l2255_225502

variable {α β : Real}
open Real

theorem beta_value :
  cos α = 1 / 7 ∧ cos (α + β) = -11 / 14 ∧ 0 < α ∧ α < π / 2 ∧ π / 2 < α + β ∧ α + β < π → 
  β = π / 3 := 
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_beta_value_l2255_225502


namespace NUMINAMATH_GPT_cars_with_neither_feature_l2255_225596

theorem cars_with_neither_feature 
  (total_cars : ℕ) 
  (power_steering : ℕ) 
  (power_windows : ℕ) 
  (both_features : ℕ) 
  (h1 : total_cars = 65) 
  (h2 : power_steering = 45) 
  (h3 : power_windows = 25) 
  (h4 : both_features = 17)
  : total_cars - (power_steering + power_windows - both_features) = 12 :=
by
  sorry

end NUMINAMATH_GPT_cars_with_neither_feature_l2255_225596


namespace NUMINAMATH_GPT_problem1_problem2_l2255_225579

-- Problem (1)
theorem problem1 : (Real.sqrt 12 + (-1 / 3)⁻¹ + (-2)^2 = 2 * Real.sqrt 3 + 1) :=
  sorry

-- Problem (2)
theorem problem2 (a : Real) (h : a ≠ 2) :
  (2 * a / (a^2 - 4) / (1 + (a - 2) / (a + 2)) = 1 / (a - 2)) :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2255_225579


namespace NUMINAMATH_GPT_positive_square_root_of_256_l2255_225531

theorem positive_square_root_of_256 (y : ℝ) (hy_pos : y > 0) (hy_squared : y^2 = 256) : y = 16 :=
by
  sorry

end NUMINAMATH_GPT_positive_square_root_of_256_l2255_225531


namespace NUMINAMATH_GPT_find_number_l2255_225564

theorem find_number (N : ℝ) (h : (1 / 2) * (3 / 5) * N = 36) : N = 120 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2255_225564


namespace NUMINAMATH_GPT_lily_milk_remaining_l2255_225565

def lilyInitialMilk : ℚ := 4
def milkGivenAway : ℚ := 7 / 3
def milkLeft : ℚ := 5 / 3

theorem lily_milk_remaining : lilyInitialMilk - milkGivenAway = milkLeft := by
  sorry

end NUMINAMATH_GPT_lily_milk_remaining_l2255_225565


namespace NUMINAMATH_GPT_copper_zinc_ratio_l2255_225572

theorem copper_zinc_ratio (total_weight : ℝ) (zinc_weight : ℝ) 
  (h_total_weight : total_weight = 70) (h_zinc_weight : zinc_weight = 31.5) : 
  (70 - 31.5) / 31.5 = 77 / 63 :=
by
  have h_copper_weight : total_weight - zinc_weight = 38.5 :=
    by rw [h_total_weight, h_zinc_weight]; norm_num
  sorry

end NUMINAMATH_GPT_copper_zinc_ratio_l2255_225572


namespace NUMINAMATH_GPT_polynomial_factorization_l2255_225523

theorem polynomial_factorization (x : ℝ) : x - x^3 = x * (1 - x) * (1 + x) := 
by sorry

end NUMINAMATH_GPT_polynomial_factorization_l2255_225523


namespace NUMINAMATH_GPT_shaded_area_calculation_l2255_225543

-- Define the dimensions of the grid and the size of each square
def gridWidth : ℕ := 9
def gridHeight : ℕ := 7
def squareSize : ℕ := 2

-- Define the number of 2x2 squares horizontally and vertically
def numSquaresHorizontally : ℕ := gridWidth / squareSize
def numSquaresVertically : ℕ := gridHeight / squareSize

-- Define the area of one 2x2 square and one shaded triangle within it
def squareArea : ℕ := squareSize * squareSize
def shadedTriangleArea : ℕ := squareArea / 2

-- Define the total number of 2x2 squares
def totalNumSquares : ℕ := numSquaresHorizontally * numSquaresVertically

-- Define the total area of shaded regions
def totalShadedArea : ℕ := totalNumSquares * shadedTriangleArea

-- The theorem to be proved
theorem shaded_area_calculation : totalShadedArea = 24 := by
  sorry    -- Placeholder for the proof

end NUMINAMATH_GPT_shaded_area_calculation_l2255_225543


namespace NUMINAMATH_GPT_inequality_solution_l2255_225510

section
variables (a x : ℝ)

theorem inequality_solution (h : a < 0) :
  (ax^2 + (1 - a) * x - 1 > 0 ↔
     (-1 < a ∧ a < 0 ∧ 1 < x ∧ x < -1/a) ∨
     (a = -1 ∧ false) ∨
     (a < -1 ∧ -1/a < x ∧ x < 1)) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l2255_225510


namespace NUMINAMATH_GPT_interest_rate_l2255_225553

noncomputable def compoundInterest (P : ℕ) (r : ℕ) (t : ℕ) : ℚ :=
  P * ((1 + r / 100 : ℚ) ^ t) - P

noncomputable def simpleInterest (P : ℕ) (r : ℕ) (t : ℕ) : ℚ :=
  P * r * t / 100

theorem interest_rate (P t : ℕ) (D : ℚ) (r : ℕ) :
  P = 10000 → t = 2 → D = 49 →
  compoundInterest P r t - simpleInterest P r t = D → r = 7 := by
  sorry

end NUMINAMATH_GPT_interest_rate_l2255_225553


namespace NUMINAMATH_GPT_distinct_rational_numbers_l2255_225530

theorem distinct_rational_numbers (m : ℚ) :
  abs m < 100 ∧ (∃ x : ℤ, 4 * x^2 + m * x + 15 = 0) → 
  ∃ n : ℕ, n = 48 :=
sorry

end NUMINAMATH_GPT_distinct_rational_numbers_l2255_225530


namespace NUMINAMATH_GPT_sin_double_angle_eq_half_l2255_225550

theorem sin_double_angle_eq_half (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : Real.sin (π / 2 + 2 * α) = Real.cos (π / 4 - α)) : 
  Real.sin (2 * α) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_eq_half_l2255_225550


namespace NUMINAMATH_GPT_minimum_abs_a_plus_b_l2255_225535

theorem minimum_abs_a_plus_b {a b : ℤ} (h1 : |a| < |b|) (h2 : |b| ≤ 4) : ∃ (a b : ℤ), |a| + b = -4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_abs_a_plus_b_l2255_225535


namespace NUMINAMATH_GPT_trig_identity_l2255_225568

theorem trig_identity (α : ℝ) (h : Real.tan α = 2 / 3) : 
  Real.sin (2 * α) - Real.cos (π - 2 * α) = 17 / 13 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l2255_225568


namespace NUMINAMATH_GPT_monthly_payment_l2255_225588

theorem monthly_payment (price : ℝ) (discount_rate : ℝ) (down_payment : ℝ) (months : ℕ) (monthly_payment : ℝ) :
  price = 480 ∧ discount_rate = 0.05 ∧ down_payment = 150 ∧ months = 3 ∧
  monthly_payment = (price * (1 - discount_rate) - down_payment) / months →
  monthly_payment = 102 :=
by
  sorry

end NUMINAMATH_GPT_monthly_payment_l2255_225588


namespace NUMINAMATH_GPT_number_of_nickels_is_3_l2255_225586

-- Defining the problem conditions
def total_coins := 8
def total_value := 53 -- in cents
def at_least_one_penny := 1
def at_least_one_nickel := 1
def at_least_one_dime := 1

-- Stating the proof problem
theorem number_of_nickels_is_3 : ∃ (pennies nickels dimes : Nat), 
  pennies + nickels + dimes = total_coins ∧ 
  pennies ≥ at_least_one_penny ∧ 
  nickels ≥ at_least_one_nickel ∧ 
  dimes ≥ at_least_one_dime ∧ 
  pennies + 5 * nickels + 10 * dimes = total_value ∧ 
  nickels = 3 := sorry

end NUMINAMATH_GPT_number_of_nickels_is_3_l2255_225586


namespace NUMINAMATH_GPT_red_robin_team_arrangements_l2255_225515

theorem red_robin_team_arrangements :
  let boys := 3
  let girls := 4
  let choose2 (n : ℕ) (k : ℕ) := Nat.choose n k
  let permutations (n : ℕ) := Nat.factorial n
  let waysToPositionBoys := choose2 boys 2 * permutations 2
  let waysToPositionRemainingMembers := permutations (boys - 2 + girls)
  waysToPositionBoys * waysToPositionRemainingMembers = 720 :=
by
  let boys := 3
  let girls := 4
  let choose2 (n : ℕ) (k : ℕ) := Nat.choose n k
  let permutations (n : ℕ) := Nat.factorial n
  let waysToPositionBoys := choose2 boys 2 * permutations 2
  let waysToPositionRemainingMembers := permutations (boys - 2 + girls)
  have : waysToPositionBoys * waysToPositionRemainingMembers = 720 := 
    by sorry -- Proof omitted here
  exact this

end NUMINAMATH_GPT_red_robin_team_arrangements_l2255_225515


namespace NUMINAMATH_GPT_speed_limit_of_friend_l2255_225514

theorem speed_limit_of_friend (total_distance : ℕ) (christina_speed : ℕ) (christina_time_min : ℕ) (friend_time_hr : ℕ) 
(h1 : total_distance = 210)
(h2 : christina_speed = 30)
(h3 : christina_time_min = 180)
(h4 : friend_time_hr = 3)
(h5 : total_distance = (christina_speed * (christina_time_min / 60)) + (christina_speed * friend_time_hr)) :
  (total_distance - christina_speed * (christina_time_min / 60)) / friend_time_hr = 40 := 
by
  sorry

end NUMINAMATH_GPT_speed_limit_of_friend_l2255_225514


namespace NUMINAMATH_GPT_complement_of_A_with_respect_to_U_l2255_225529

-- Definitions
def U : Set ℤ := {-2, -1, 1, 3, 5}
def A : Set ℤ := {-1, 3}

-- Statement of the problem
theorem complement_of_A_with_respect_to_U :
  (U \ A) = {-2, 1, 5} := 
by
  sorry

end NUMINAMATH_GPT_complement_of_A_with_respect_to_U_l2255_225529


namespace NUMINAMATH_GPT_net_marble_change_l2255_225590

/-- Josh's initial number of marbles. -/
def initial_marbles : ℕ := 20

/-- Number of marbles Josh lost. -/
def lost_marbles : ℕ := 16

/-- Number of marbles Josh found. -/
def found_marbles : ℕ := 8

/-- Number of marbles Josh traded away. -/
def traded_away_marbles : ℕ := 5

/-- Number of marbles Josh received in a trade. -/
def received_in_trade_marbles : ℕ := 9

/-- Number of marbles Josh gave away. -/
def gave_away_marbles : ℕ := 3

/-- Number of marbles Josh received from his cousin. -/
def received_from_cousin_marbles : ℕ := 4

/-- Final number of marbles Josh has after all transactions. -/
def final_marbles : ℕ :=
  initial_marbles - lost_marbles + found_marbles - traded_away_marbles + received_in_trade_marbles
  - gave_away_marbles + received_from_cousin_marbles

theorem net_marble_change : (final_marbles : ℤ) - (initial_marbles : ℤ) = -3 := 
by
  sorry

end NUMINAMATH_GPT_net_marble_change_l2255_225590


namespace NUMINAMATH_GPT_xiangming_payment_methods_count_l2255_225576

def xiangming_payment_methods : Prop :=
  ∃ x y z : ℕ, 
    x + y + z ≤ 10 ∧ 
    x + 2 * y + 5 * z = 18 ∧ 
    ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ z > 0) ∨ (y > 0 ∧ z > 0))

theorem xiangming_payment_methods_count : 
  xiangming_payment_methods → ∃! n, n = 11 :=
by sorry

end NUMINAMATH_GPT_xiangming_payment_methods_count_l2255_225576


namespace NUMINAMATH_GPT_positive_number_percent_l2255_225524

theorem positive_number_percent (x : ℝ) (h : 0.01 * x^2 = 9) (hx : 0 < x) : x = 30 :=
sorry

end NUMINAMATH_GPT_positive_number_percent_l2255_225524


namespace NUMINAMATH_GPT_student_B_speed_l2255_225597

theorem student_B_speed (d : ℝ) (ratio : ℝ) (t_diff : ℝ) (sB : ℝ) : 
  d = 12 → ratio = 1.2 → t_diff = 1/6 → 
  (d / sB - t_diff = d / (ratio * sB)) → 
  sB = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_student_B_speed_l2255_225597


namespace NUMINAMATH_GPT_range_of_expression_l2255_225594

theorem range_of_expression (x y : ℝ) (h1 : x * y = 1) (h2 : 3 ≥ x ∧ x ≥ 4 * y ∧ 4 * y > 0) :
  ∃ A B, A = 4 ∧ B = 5 ∧ ∀ z, z = (x^2 + 4 * y^2) / (x - 2 * y) → 4 ≤ z ∧ z ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_expression_l2255_225594


namespace NUMINAMATH_GPT_delivery_cost_l2255_225548

theorem delivery_cost (base_fee : ℕ) (limit : ℕ) (extra_fee : ℕ) 
(item_weight : ℕ) (total_cost : ℕ) 
(h1 : base_fee = 13) (h2 : limit = 5) (h3 : extra_fee = 2) 
(h4 : item_weight = 7) (h5 : total_cost = 17) : 
  total_cost = base_fee + (item_weight - limit) * extra_fee := 
by
  sorry

end NUMINAMATH_GPT_delivery_cost_l2255_225548


namespace NUMINAMATH_GPT_find_a_l2255_225577

theorem find_a (a : ℝ) 
  (line_through : ∃ (p1 p2 : ℝ × ℝ), p1 = (a-2, -1) ∧ p2 = (-a-2, 1)) 
  (perpendicular : ∀ (l1 l2 : ℝ × ℝ), l1 = (2, 3) → l2 = (-1/a, 1) → false) : 
  a = -2/3 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_l2255_225577


namespace NUMINAMATH_GPT_derivative_equals_l2255_225539

noncomputable def func (x : ℝ) : ℝ :=
  (3 / (8 * Real.sqrt 2) * Real.log ((Real.sqrt 2 + Real.tanh x) / (Real.sqrt 2 - Real.tanh x)))
  - (Real.tanh x / (4 * (2 - (Real.tanh x)^2)))

theorem derivative_equals :
  ∀ x : ℝ, deriv func x = 1 / (2 + (Real.cosh x)^2)^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_derivative_equals_l2255_225539


namespace NUMINAMATH_GPT_topsoil_cost_is_112_l2255_225551

noncomputable def calculate_topsoil_cost (length width depth_in_inches : ℝ) (cost_per_cubic_foot : ℝ) : ℝ :=
  let depth_in_feet := depth_in_inches / 12
  let volume := length * width * depth_in_feet
  volume * cost_per_cubic_foot

theorem topsoil_cost_is_112 :
  calculate_topsoil_cost 8 4 6 7 = 112 :=
by
  sorry

end NUMINAMATH_GPT_topsoil_cost_is_112_l2255_225551


namespace NUMINAMATH_GPT_volume_of_larger_prism_is_correct_l2255_225580

noncomputable def volume_of_larger_solid : ℝ :=
  let A := (0, 0, 0)
  let B := (2, 0, 0)
  let C := (2, 2, 0)
  let D := (0, 2, 0)
  let E := (0, 0, 2)
  let F := (2, 0, 2)
  let G := (2, 2, 2)
  let H := (0, 2, 2)
  let P := (1, 1, 1)
  let Q := (1, 0, 1)
  
  -- Assume the plane equation here divides the cube into equal halves
  -- Calculate the volume of one half of the cube
  let volume := 2 -- This represents the volume of the larger solid

  volume

theorem volume_of_larger_prism_is_correct :
  volume_of_larger_solid = 2 :=
sorry

end NUMINAMATH_GPT_volume_of_larger_prism_is_correct_l2255_225580


namespace NUMINAMATH_GPT_major_axis_length_l2255_225584

theorem major_axis_length {r : ℝ} (h_r : r = 1) (h_major : ∃ (minor_axis : ℝ), minor_axis = 2 * r ∧ 1.5 * minor_axis = major_axis) : major_axis = 3 :=
by
  sorry

end NUMINAMATH_GPT_major_axis_length_l2255_225584


namespace NUMINAMATH_GPT_area_square_II_l2255_225522

theorem area_square_II (a b : ℝ) :
  let diag_I := 2 * (a + b)
  let area_I := (a + b) * (a + b) * 2
  let area_II := area_I * 3
  area_II = 6 * (a + b) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_area_square_II_l2255_225522


namespace NUMINAMATH_GPT_sara_gave_dan_pears_l2255_225500

theorem sara_gave_dan_pears :
  ∀ (original_pears left_pears given_to_dan : ℕ),
    original_pears = 35 →
    left_pears = 7 →
    given_to_dan = original_pears - left_pears →
    given_to_dan = 28 :=
by
  intros original_pears left_pears given_to_dan h_original h_left h_given
  rw [h_original, h_left] at h_given
  exact h_given

end NUMINAMATH_GPT_sara_gave_dan_pears_l2255_225500


namespace NUMINAMATH_GPT_length_real_axis_l2255_225581

theorem length_real_axis (x y : ℝ) : 
  (x^2 / 4 - y^2 / 12 = 1) → 4 = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_length_real_axis_l2255_225581


namespace NUMINAMATH_GPT_total_passengers_per_day_l2255_225560

-- Define the conditions
def airplanes : ℕ := 5
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day : ℕ := 2

-- Define the proof problem
theorem total_passengers_per_day : 
  (airplanes * rows_per_airplane * seats_per_row * flights_per_day) = 1400 := 
by 
  sorry

end NUMINAMATH_GPT_total_passengers_per_day_l2255_225560


namespace NUMINAMATH_GPT_store_profit_loss_l2255_225505

theorem store_profit_loss :
  ∃ (x y : ℝ), (1 + 0.25) * x = 135 ∧ (1 - 0.25) * y = 135 ∧ (135 - x) + (135 - y) = -18 :=
by
  sorry

end NUMINAMATH_GPT_store_profit_loss_l2255_225505


namespace NUMINAMATH_GPT_intersection_M_N_l2255_225592

def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N := {x : ℝ | x^2 - 3*x ≤ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2255_225592


namespace NUMINAMATH_GPT_simplify_complex_expression_l2255_225593

open Complex

theorem simplify_complex_expression :
  let a := (4 : ℂ) + 6 * I
  let b := (4 : ℂ) - 6 * I
  ((a / b) - (b / a) = (24 * I) / 13) := by
  sorry

end NUMINAMATH_GPT_simplify_complex_expression_l2255_225593


namespace NUMINAMATH_GPT_area_of_circle_portion_l2255_225574

theorem area_of_circle_portion :
  (∀ x y : ℝ, (x^2 + 6 * x + y^2 = 50) → y ≤ x - 3 → y ≤ 0 → (y^2 + (x + 3)^2 ≤ 59)) →
  (∃ area : ℝ, area = (59 * Real.pi / 4)) :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_portion_l2255_225574


namespace NUMINAMATH_GPT_equivalent_polar_point_representation_l2255_225532

/-- Representation of a point in polar coordinates -/
structure PolarPoint :=
  (r : ℝ)
  (θ : ℝ)

theorem equivalent_polar_point_representation :
  ∀ (p1 p2 : PolarPoint), p1 = PolarPoint.mk (-1) (5 * Real.pi / 6) →
    (p2 = PolarPoint.mk 1 (11 * Real.pi / 6) → p1.r + Real.pi = p2.r ∧ p1.θ = p2.θ) :=
by
  intros p1 p2 h1 h2
  sorry

end NUMINAMATH_GPT_equivalent_polar_point_representation_l2255_225532


namespace NUMINAMATH_GPT_budget_for_bulbs_l2255_225521

theorem budget_for_bulbs (num_crocus_bulbs : ℕ) (cost_per_crocus : ℝ) (budget : ℝ)
  (h1 : num_crocus_bulbs = 22)
  (h2 : cost_per_crocus = 0.35)
  (h3 : budget = num_crocus_bulbs * cost_per_crocus) :
  budget = 7.70 :=
sorry

end NUMINAMATH_GPT_budget_for_bulbs_l2255_225521


namespace NUMINAMATH_GPT_garden_snake_length_l2255_225537

theorem garden_snake_length :
  ∀ (garden_snake boa_constrictor : ℝ),
    boa_constrictor * 7.0 = garden_snake →
    boa_constrictor = 1.428571429 →
    garden_snake = 10.0 :=
by
  intros garden_snake boa_constrictor H1 H2
  sorry

end NUMINAMATH_GPT_garden_snake_length_l2255_225537


namespace NUMINAMATH_GPT_cos_x_plus_2y_eq_one_l2255_225561

theorem cos_x_plus_2y_eq_one (x y a : ℝ) 
  (hx : -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4)
  (hy : -Real.pi / 4 ≤ y ∧ y ≤ Real.pi / 4)
  (h_eq1 : x^3 + Real.sin x - 2 * a = 0)
  (h_eq2 : 4 * y^3 + (1 / 2) * Real.sin (2 * y) + a = 0) : 
  Real.cos (x + 2 * y) = 1 := 
sorry -- Proof goes here

end NUMINAMATH_GPT_cos_x_plus_2y_eq_one_l2255_225561


namespace NUMINAMATH_GPT_initial_cost_renting_car_l2255_225520

theorem initial_cost_renting_car
  (initial_cost : ℝ)
  (miles_monday : ℝ := 620)
  (miles_thursday : ℝ := 744)
  (cost_per_mile : ℝ := 0.50)
  (total_spent : ℝ := 832)
  (total_miles : ℝ := miles_monday + miles_thursday)
  (expected_initial_cost : ℝ := 150) :
  total_spent = initial_cost + cost_per_mile * total_miles → initial_cost = expected_initial_cost :=
by
  sorry

end NUMINAMATH_GPT_initial_cost_renting_car_l2255_225520


namespace NUMINAMATH_GPT_lg_sum_geometric_seq_l2255_225549

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem lg_sum_geometric_seq (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 * a 5 * a 8 = 1) :
  Real.log (a 4) + Real.log (a 6) = 0 := 
sorry

end NUMINAMATH_GPT_lg_sum_geometric_seq_l2255_225549


namespace NUMINAMATH_GPT_determine_asymptotes_l2255_225595

noncomputable def asymptotes_of_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (2 * a = 2 * Real.sqrt 2) ∧ (2 * b = 2) → 
  (∀ x y : ℝ, (y = x * (Real.sqrt 2 / 2) ∨ y = -x * (Real.sqrt 2 / 2)))

theorem determine_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a = 2 * Real.sqrt 2) ∧ (2 * b = 2) → 
  asymptotes_of_hyperbola a b ha hb :=
by
  intros h
  sorry

end NUMINAMATH_GPT_determine_asymptotes_l2255_225595


namespace NUMINAMATH_GPT_min_moves_to_find_treasure_l2255_225563

theorem min_moves_to_find_treasure (cells : List ℕ) (h1 : cells = [5, 5, 5]) : 
  ∃ n, n = 2 ∧ (∀ moves, moves ≥ n → true) := sorry

end NUMINAMATH_GPT_min_moves_to_find_treasure_l2255_225563


namespace NUMINAMATH_GPT_foodAdditivesPercentage_l2255_225525

-- Define the given percentages
def microphotonicsPercentage : ℕ := 14
def homeElectronicsPercentage : ℕ := 24
def microorganismsPercentage : ℕ := 29
def industrialLubricantsPercentage : ℕ := 8

-- Define degrees representing basic astrophysics
def basicAstrophysicsDegrees : ℕ := 18

-- Define the total degrees in a circle
def totalDegrees : ℕ := 360

-- Define the total budget percentage
def totalBudgetPercentage : ℕ := 100

-- Prove that the remaining percentage for food additives is 20%
theorem foodAdditivesPercentage :
  let basicAstrophysicsPercentage := (basicAstrophysicsDegrees * totalBudgetPercentage) / totalDegrees
  let totalKnownPercentage := microphotonicsPercentage + homeElectronicsPercentage + microorganismsPercentage + industrialLubricantsPercentage + basicAstrophysicsPercentage
  totalBudgetPercentage - totalKnownPercentage = 20 :=
by
  let basicAstrophysicsPercentage := (basicAstrophysicsDegrees * totalBudgetPercentage) / totalDegrees
  let totalKnownPercentage := microphotonicsPercentage + homeElectronicsPercentage + microorganismsPercentage + industrialLubricantsPercentage + basicAstrophysicsPercentage
  sorry

end NUMINAMATH_GPT_foodAdditivesPercentage_l2255_225525


namespace NUMINAMATH_GPT_ellipse_sum_l2255_225540

-- Define the givens
def h : ℤ := -3
def k : ℤ := 5
def a : ℤ := 7
def b : ℤ := 4

-- State the theorem to be proven
theorem ellipse_sum : h + k + a + b = 13 := by
  sorry

end NUMINAMATH_GPT_ellipse_sum_l2255_225540


namespace NUMINAMATH_GPT_bee_count_l2255_225518

theorem bee_count (initial_bees additional_bees : ℕ) (h_init : initial_bees = 16) (h_add : additional_bees = 9) :
  initial_bees + additional_bees = 25 :=
by
  sorry

end NUMINAMATH_GPT_bee_count_l2255_225518


namespace NUMINAMATH_GPT_number_of_vip_children_l2255_225544

theorem number_of_vip_children (total_attendees children_percentage children_vip_percentage : ℕ) :
  total_attendees = 400 →
  children_percentage = 75 →
  children_vip_percentage = 20 →
  (total_attendees * children_percentage / 100) * children_vip_percentage / 100 = 60 :=
by
  intros h_total h_children_pct h_vip_pct
  sorry

end NUMINAMATH_GPT_number_of_vip_children_l2255_225544


namespace NUMINAMATH_GPT_subtraction_of_tenths_l2255_225557

theorem subtraction_of_tenths (a b : ℝ) (n : ℕ) (h1 : a = (1 / 10) * 6000) (h2 : b = (1 / 10 / 100) * 6000) : (a - b) = 594 := by
sorry

end NUMINAMATH_GPT_subtraction_of_tenths_l2255_225557


namespace NUMINAMATH_GPT_add_pure_water_to_achieve_solution_l2255_225573

theorem add_pure_water_to_achieve_solution
  (w : ℝ) (h_salt_content : 0.15 * 40 = 6) (h_new_concentration : 6 / (40 + w) = 0.1) :
  w = 20 :=
sorry

end NUMINAMATH_GPT_add_pure_water_to_achieve_solution_l2255_225573


namespace NUMINAMATH_GPT_chickens_cheaper_than_buying_eggs_l2255_225570

theorem chickens_cheaper_than_buying_eggs :
  ∃ W, W ≥ 80 ∧ 80 + W ≤ 2 * W :=
by
  sorry

end NUMINAMATH_GPT_chickens_cheaper_than_buying_eggs_l2255_225570


namespace NUMINAMATH_GPT_meaningful_expression_l2255_225519

theorem meaningful_expression (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 1))) → x > 1 :=
by sorry

end NUMINAMATH_GPT_meaningful_expression_l2255_225519


namespace NUMINAMATH_GPT_fraction_of_groups_with_a_and_b_l2255_225567

/- Definitions based on the conditions -/
def total_persons : ℕ := 6
def group_size : ℕ := 3
def person_a : ℕ := 1  -- arbitrary assignment for simplicity
def person_b : ℕ := 2  -- arbitrary assignment for simplicity

/- Hypotheses based on conditions -/
axiom six_persons (n : ℕ) : n = total_persons
axiom divided_into_two_groups (grp_size : ℕ) : grp_size = group_size
axiom a_and_b_included (a b : ℕ) : a = person_a ∧ b = person_b

/- The theorem to prove -/
theorem fraction_of_groups_with_a_and_b
    (total_groups : ℕ := Nat.choose total_persons group_size)
    (groups_with_a_b : ℕ := Nat.choose 4 1) :
    groups_with_a_b / total_groups = 1 / 5 :=
by
    sorry

end NUMINAMATH_GPT_fraction_of_groups_with_a_and_b_l2255_225567


namespace NUMINAMATH_GPT_max_value_inequality_am_gm_inequality_l2255_225599

-- Given conditions and goals as Lean statements
theorem max_value_inequality (x : ℝ) : (|x - 1| + |x - 2| ≥ 1) := sorry

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (1/a) + (1/(2*b)) + (1/(3*c)) = 1) : (a + 2*b + 3*c) ≥ 9 := sorry

end NUMINAMATH_GPT_max_value_inequality_am_gm_inequality_l2255_225599


namespace NUMINAMATH_GPT_find_y_l2255_225546

theorem find_y (x y : ℕ) (h1 : 24 * x = 173 * y) (h2 : 173 * y = 1730) : y = 10 :=
by 
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_find_y_l2255_225546


namespace NUMINAMATH_GPT_cos_beta_l2255_225562

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h_cos_α : Real.cos α = 3/5) (h_cos_alpha_plus_beta : Real.cos (α + β) = -5/13) : 
  Real.cos β = 33/65 :=
by
  sorry

end NUMINAMATH_GPT_cos_beta_l2255_225562


namespace NUMINAMATH_GPT_StatementA_incorrect_l2255_225566

def f (n : ℕ) : ℕ := (n.factorial)^2

def g (x : ℕ) : ℕ := f (x + 1) / f x

theorem StatementA_incorrect (x : ℕ) (h : x = 1) : g x ≠ 4 := sorry

end NUMINAMATH_GPT_StatementA_incorrect_l2255_225566


namespace NUMINAMATH_GPT_solve_inequality_l2255_225582

theorem solve_inequality :
  {x : ℝ | (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 2)} =
  {x : ℝ | (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x)} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2255_225582


namespace NUMINAMATH_GPT_g_at_seven_equals_92_l2255_225533

def g (n : ℕ) : ℕ := n^2 + 2*n + 29

theorem g_at_seven_equals_92 : g 7 = 92 :=
by
  sorry

end NUMINAMATH_GPT_g_at_seven_equals_92_l2255_225533


namespace NUMINAMATH_GPT_malou_average_score_l2255_225591

def quiz1_score := 91
def quiz2_score := 90
def quiz3_score := 92

def sum_of_scores := quiz1_score + quiz2_score + quiz3_score
def number_of_quizzes := 3

theorem malou_average_score : sum_of_scores / number_of_quizzes = 91 :=
by
  sorry

end NUMINAMATH_GPT_malou_average_score_l2255_225591


namespace NUMINAMATH_GPT_conic_is_parabola_l2255_225571

-- Define the main equation
def main_equation (x y : ℝ) : Prop :=
  y^4 - 6 * x^2 = 3 * y^2 - 2

-- Definition of parabola condition
def is_parabola (x y : ℝ) : Prop :=
  ∃ a b c : ℝ, y^2 = a * x + b ∧ a ≠ 0

-- The theorem statement.
theorem conic_is_parabola :
  ∀ x y : ℝ, main_equation x y → is_parabola x y :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_conic_is_parabola_l2255_225571


namespace NUMINAMATH_GPT_polynomial_perfect_square_l2255_225547

theorem polynomial_perfect_square (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 1 = (x^2 + 5 * x + 5)^2 :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_perfect_square_l2255_225547


namespace NUMINAMATH_GPT_batteries_manufactured_l2255_225558

theorem batteries_manufactured (gather_time create_time : Nat) (robots : Nat) (hours : Nat) (total_batteries : Nat) :
  gather_time = 6 →
  create_time = 9 →
  robots = 10 →
  hours = 5 →
  total_batteries = (hours * 60 / (gather_time + create_time)) * robots →
  total_batteries = 200 :=
by
  intros h_gather h_create h_robots h_hours h_batteries
  simp [h_gather, h_create, h_robots, h_hours] at h_batteries
  exact h_batteries

end NUMINAMATH_GPT_batteries_manufactured_l2255_225558


namespace NUMINAMATH_GPT_find_b_l2255_225501

theorem find_b (α β b : ℤ)
  (h1: α > 1)
  (h2: β < -1)
  (h3: ∃ x : ℝ, α * x^2 + β * x - 2 = 0)
  (h4: ∃ x : ℝ, x^2 + bx - 2 = 0)
  (hb: ∀ root1 root2 : ℝ, root1 * root2 = -2 ∧ root1 + root2 = -b) :
  b = 0 := 
sorry

end NUMINAMATH_GPT_find_b_l2255_225501


namespace NUMINAMATH_GPT_hannah_speed_l2255_225509

theorem hannah_speed :
  ∃ H : ℝ, 
    (∀ t : ℝ, (t = 6) → d = 130) ∧ 
    (∀ t : ℝ, (t = 11) → d = 130) → 
    (d = 37 * 5 + H * 5) → 
    H = 15 := 
by 
  sorry

end NUMINAMATH_GPT_hannah_speed_l2255_225509


namespace NUMINAMATH_GPT_cats_in_village_l2255_225587

theorem cats_in_village (C : ℕ) (h1 : 1 / 3 * C = (1 / 4) * (1 / 3) * C)
  (h2 : (1 / 12) * C = 10) : C = 120 :=
sorry

end NUMINAMATH_GPT_cats_in_village_l2255_225587


namespace NUMINAMATH_GPT_area_of_given_circle_is_4pi_l2255_225513

-- Define the given equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 - 12 * x + 18 * y + 27 = 0

-- Define the area of the circle to be proved
noncomputable def area_of_circle : ℝ := 4 * Real.pi

-- Statement of the theorem to be proved in Lean
theorem area_of_given_circle_is_4pi :
  (∃ x y : ℝ, circle_equation x y) → area_of_circle = 4 * Real.pi :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_area_of_given_circle_is_4pi_l2255_225513


namespace NUMINAMATH_GPT_correct_operation_l2255_225527

theorem correct_operation (a b : ℝ) :
  ¬ (a^2 + a^3 = a^6) ∧
  ¬ ((a*b)^2 = a*(b^2)) ∧
  ¬ ((a + b)^2 = a^2 + b^2) ∧
  ((a + b)*(a - b) = a^2 - b^2) := 
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2255_225527


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2255_225504

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n * a₁ + (n * (n - 1) / 2) * d

/-- Condition for the sum of the first 5 terms -/
def S5 (a₁ d : ℚ) : Prop := S 5 a₁ d = 6

/-- Condition for the second term of the sequence -/
def a2 (a₁ d : ℚ) : Prop := a₁ + d = 1

/-- The main theorem to be proved -/
theorem arithmetic_sequence_common_difference (a₁ d : ℚ) (hS5 : S5 a₁ d) (ha2 : a2 a₁ d) : d = 1 / 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2255_225504


namespace NUMINAMATH_GPT_find_two_heaviest_l2255_225506

theorem find_two_heaviest (a b c d : ℝ) : 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) →
  ∃ x y : ℝ, (x ≠ y) ∧ (x = max (max (max a b) c) d) ∧ (y = max (max (min (max a b) c) d) d) :=
by sorry

end NUMINAMATH_GPT_find_two_heaviest_l2255_225506
