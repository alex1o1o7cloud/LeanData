import Mathlib

namespace NUMINAMATH_CALUDE_estimation_greater_than_exact_l897_89735

theorem estimation_greater_than_exact 
  (a b d : ℕ+) 
  (a' b' d' : ℝ)
  (h_a : a' > a ∧ a' < a + 1)
  (h_b : b' < b ∧ b' > b - 1)
  (h_d : d' < d ∧ d' > d - 1) :
  Real.sqrt (a' / b') - Real.sqrt d' > Real.sqrt (a / b) - Real.sqrt d :=
sorry

end NUMINAMATH_CALUDE_estimation_greater_than_exact_l897_89735


namespace NUMINAMATH_CALUDE_right_triangles_common_hypotenuse_l897_89739

theorem right_triangles_common_hypotenuse (AC AD CD : ℝ) (hAC : AC = 16) (hAD : AD = 32) (hCD : CD = 14) :
  let AB := Real.sqrt (AD^2 - (AC + CD)^2)
  let BC := Real.sqrt (AB^2 + AC^2)
  BC = Real.sqrt 380 := by
sorry

end NUMINAMATH_CALUDE_right_triangles_common_hypotenuse_l897_89739


namespace NUMINAMATH_CALUDE_sin_cos_identity_l897_89755

theorem sin_cos_identity : 
  Real.sin (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (138 * π / 180) * Real.cos (72 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l897_89755


namespace NUMINAMATH_CALUDE_no_primes_for_d_10_l897_89773

theorem no_primes_for_d_10 : ¬∃ (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  (q * r) ∣ (p^2 + 10) ∧
  (r * p) ∣ (q^2 + 10) ∧
  (p * q) ∣ (r^2 + 10) :=
by
  sorry

-- Note: The case for d = 11 is not included as the solution was inconclusive

end NUMINAMATH_CALUDE_no_primes_for_d_10_l897_89773


namespace NUMINAMATH_CALUDE_bennys_work_days_l897_89776

theorem bennys_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days_worked : ℕ) : 
  hours_per_day = 3 →
  total_hours = 18 →
  days_worked * hours_per_day = total_hours →
  days_worked = 6 := by
sorry

end NUMINAMATH_CALUDE_bennys_work_days_l897_89776


namespace NUMINAMATH_CALUDE_valid_sequences_10_l897_89747

def T : ℕ → ℕ
  | 0 => 0  -- We define T(0) as 0 for completeness
  | 1 => 2
  | 2 => 4
  | (n + 3) => T (n + 2) + T (n + 1)

def valid_sequences (n : ℕ) : ℕ := T n

theorem valid_sequences_10 : valid_sequences 10 = 178 := by
  sorry

#eval valid_sequences 10

end NUMINAMATH_CALUDE_valid_sequences_10_l897_89747


namespace NUMINAMATH_CALUDE_estimate_larger_than_original_l897_89733

theorem estimate_larger_than_original (x y ε : ℝ) 
  (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) : 
  (x + ε) - (y - ε) > x - y := by
  sorry

end NUMINAMATH_CALUDE_estimate_larger_than_original_l897_89733


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l897_89795

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) ↔ m ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l897_89795


namespace NUMINAMATH_CALUDE_total_cards_proof_l897_89729

/-- The number of baseball cards Carlos has -/
def carlos_cards : ℕ := 20

/-- The difference in cards between Carlos and Matias -/
def difference : ℕ := 6

/-- The number of baseball cards Matias has -/
def matias_cards : ℕ := carlos_cards - difference

/-- The number of baseball cards Jorge has -/
def jorge_cards : ℕ := matias_cards

/-- The total number of baseball cards -/
def total_cards : ℕ := carlos_cards + matias_cards + jorge_cards

theorem total_cards_proof : total_cards = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_proof_l897_89729


namespace NUMINAMATH_CALUDE_sum_of_abc_l897_89732

theorem sum_of_abc (a b c : ℝ) 
  (h1 : a^2*b + a^2*c + b^2*a + b^2*c + c^2*a + c^2*b + 3*a*b*c = 30)
  (h2 : a^2 + b^2 + c^2 = 13) : 
  a + b + c = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_abc_l897_89732


namespace NUMINAMATH_CALUDE_power_function_linear_intersection_min_value_l897_89738

theorem power_function_linear_intersection_min_value (m n k b : ℝ) : 
  (2 * m - 1 = 1) →  -- Condition for power function
  (n - 2 = 0) →      -- Condition for power function
  (k > 0) →          -- Given condition for k
  (b > 0) →          -- Given condition for b
  (k * m + b = n) →  -- Linear function passes through (m, n)
  (∀ k' b' : ℝ, k' > 0 → b' > 0 → k' * m + b' = n → 4 / k' + 1 / b' ≥ 9 / 2) ∧ 
  (∃ k' b' : ℝ, k' > 0 ∧ b' > 0 ∧ k' * m + b' = n ∧ 4 / k' + 1 / b' = 9 / 2) :=
by sorry

end NUMINAMATH_CALUDE_power_function_linear_intersection_min_value_l897_89738


namespace NUMINAMATH_CALUDE_fruit_distribution_l897_89766

theorem fruit_distribution (total_strawberries total_grapes : ℕ) 
  (leftover_strawberries leftover_grapes : ℕ) :
  total_strawberries = 66 →
  total_grapes = 49 →
  leftover_strawberries = 6 →
  leftover_grapes = 4 →
  ∃ (B : ℕ), 
    B > 0 ∧
    (total_strawberries - leftover_strawberries) % B = 0 ∧
    (total_grapes - leftover_grapes) % B = 0 ∧
    B = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_distribution_l897_89766


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l897_89725

theorem complex_sum_of_parts (a b : ℝ) (h : (Complex.mk a b) = Complex.mk 1 (-1)) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l897_89725


namespace NUMINAMATH_CALUDE_max_value_theorem_l897_89703

theorem max_value_theorem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_constraint : x^2 - x*y + 2*y^2 = 8) :
  x^2 + x*y + 2*y^2 ≤ (72 + 32*Real.sqrt 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l897_89703


namespace NUMINAMATH_CALUDE_davids_math_marks_l897_89752

theorem davids_math_marks (english physics chemistry biology average : ℕ) 
  (h1 : english = 91)
  (h2 : physics = 82)
  (h3 : chemistry = 67)
  (h4 : biology = 85)
  (h5 : average = 78)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 65 := by
  sorry

#check davids_math_marks

end NUMINAMATH_CALUDE_davids_math_marks_l897_89752


namespace NUMINAMATH_CALUDE_fallen_pages_count_l897_89783

/-- Represents a page number as a triple of digits -/
structure PageNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a PageNumber to its numerical value -/
def PageNumber.toNat (p : PageNumber) : Nat :=
  p.hundreds * 100 + p.tens * 10 + p.ones

/-- Checks if a PageNumber is even -/
def PageNumber.isEven (p : PageNumber) : Prop :=
  p.toNat % 2 = 0

/-- Checks if a PageNumber is a permutation of another PageNumber -/
def PageNumber.isPermutationOf (p1 p2 : PageNumber) : Prop :=
  (p1.hundreds = p2.hundreds ∨ p1.hundreds = p2.tens ∨ p1.hundreds = p2.ones) ∧
  (p1.tens = p2.hundreds ∨ p1.tens = p2.tens ∨ p1.tens = p2.ones) ∧
  (p1.ones = p2.hundreds ∨ p1.ones = p2.tens ∨ p1.ones = p2.ones)

theorem fallen_pages_count 
  (first_page last_page : PageNumber)
  (h_first : first_page.toNat = 143)
  (h_perm : last_page.isPermutationOf first_page)
  (h_even : last_page.isEven)
  (h_greater : last_page.toNat > first_page.toNat) :
  last_page.toNat - first_page.toNat + 1 = 172 := by
  sorry

end NUMINAMATH_CALUDE_fallen_pages_count_l897_89783


namespace NUMINAMATH_CALUDE_guide_is_native_l897_89797

/-- Represents the two tribes on the island -/
inductive Tribe
  | Native
  | Alien

/-- Represents a person on the island -/
structure Person where
  tribe : Tribe

/-- Represents the claim a person makes about their tribe -/
def claim (p : Person) : Tribe :=
  match p.tribe with
  | Tribe.Native => Tribe.Native
  | Tribe.Alien => Tribe.Native

/-- Represents the report a guide makes about another person's claim -/
def report (guide : Person) (other : Person) : Tribe :=
  match guide.tribe with
  | Tribe.Native => claim other
  | Tribe.Alien => claim other

theorem guide_is_native (guide : Person) (other : Person) :
  report guide other = Tribe.Native → guide.tribe = Tribe.Native :=
by
  sorry

#check guide_is_native

end NUMINAMATH_CALUDE_guide_is_native_l897_89797


namespace NUMINAMATH_CALUDE_janice_earnings_l897_89711

/-- Calculates the total earnings for Janice's work week -/
def calculate_earnings (days_worked : ℕ) (daily_rate : ℕ) (overtime_rate : ℕ) (overtime_shifts : ℕ) : ℕ :=
  days_worked * daily_rate + overtime_shifts * overtime_rate

/-- Proves that Janice's earnings for the week equal $195 -/
theorem janice_earnings : calculate_earnings 5 30 15 3 = 195 := by
  sorry

end NUMINAMATH_CALUDE_janice_earnings_l897_89711


namespace NUMINAMATH_CALUDE_limit_Sn_divided_by_n2Bn_l897_89709

-- Define the set A
def A (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

-- Define B_n as the number of subsets of A
def B_n (n : ℕ) : ℕ := 2^n

-- Define S_n as the sum of elements in non-empty proper subsets of A
def S_n (n : ℕ) : ℕ := (n * (n + 1) / 2) * (2^(n - 1) - 1)

-- State the theorem
theorem limit_Sn_divided_by_n2Bn (ε : ℝ) (ε_pos : ε > 0) :
  ∃ N : ℕ, ∀ n ≥ N, |S_n n / (n^2 * B_n n : ℝ) - 1/4| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_Sn_divided_by_n2Bn_l897_89709


namespace NUMINAMATH_CALUDE_quadratic_and_squared_equation_solutions_l897_89788

theorem quadratic_and_squared_equation_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 - 3 * x₁ + 1 = 0) ∧ (2 * x₂^2 - 3 * x₂ + 1 = 0) ∧ x₁ = 1/2 ∧ x₂ = 1) ∧
  (∃ y₁ y₂ : ℝ, ((y₁ - 2)^2 = (2 * y₁ + 3)^2) ∧ ((y₂ - 2)^2 = (2 * y₂ + 3)^2) ∧ y₁ = -5 ∧ y₂ = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_squared_equation_solutions_l897_89788


namespace NUMINAMATH_CALUDE_exp_greater_than_linear_l897_89785

theorem exp_greater_than_linear (x : ℝ) (h : x > 0) : Real.exp x ≥ Real.exp 1 * x := by
  sorry

end NUMINAMATH_CALUDE_exp_greater_than_linear_l897_89785


namespace NUMINAMATH_CALUDE_solve_for_a_l897_89794

theorem solve_for_a : ∃ a : ℝ, 
  (2 * 1 - a * (-1) = 3) ∧ (a = 1) := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l897_89794


namespace NUMINAMATH_CALUDE_private_teacher_cost_l897_89779

/-- Calculates the amount each parent must pay for a private teacher --/
theorem private_teacher_cost 
  (former_salary : ℕ) 
  (raise_percentage : ℚ) 
  (num_kids : ℕ) 
  (h1 : former_salary = 45000)
  (h2 : raise_percentage = 1/5)
  (h3 : num_kids = 9) : 
  (former_salary * (1 + raise_percentage)) / num_kids = 6000 := by
  sorry

end NUMINAMATH_CALUDE_private_teacher_cost_l897_89779


namespace NUMINAMATH_CALUDE_rain_in_tel_aviv_l897_89774

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of rain on any given day in Tel Aviv. -/
def rain_probability : ℝ := 0.5

/-- The number of randomly chosen days. -/
def total_days : ℕ := 6

/-- The number of rainy days we're interested in. -/
def rainy_days : ℕ := 4

theorem rain_in_tel_aviv :
  binomial_probability total_days rainy_days rain_probability = 0.234375 := by
  sorry

end NUMINAMATH_CALUDE_rain_in_tel_aviv_l897_89774


namespace NUMINAMATH_CALUDE_tailor_cut_difference_l897_89727

theorem tailor_cut_difference (skirt_cut pants_cut : ℝ) 
  (h1 : skirt_cut = 0.75)
  (h2 : pants_cut = 0.5) : 
  skirt_cut - pants_cut = 0.25 := by
sorry

end NUMINAMATH_CALUDE_tailor_cut_difference_l897_89727


namespace NUMINAMATH_CALUDE_inequality_proof_l897_89719

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (a * b / Real.sqrt (c^2 + 3)) + (b * c / Real.sqrt (a^2 + 3)) + (c * a / Real.sqrt (b^2 + 3)) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l897_89719


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l897_89754

/-- Given plane vectors a and b satisfying certain conditions, prove that the magnitude of their sum is √21. -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) : 
  ‖a‖ = 2 → 
  ‖b‖ = 3 → 
  a - b = (Real.sqrt 2, Real.sqrt 3) →
  ‖a + b‖ = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l897_89754


namespace NUMINAMATH_CALUDE_equation_solution_l897_89745

theorem equation_solution : 
  ∃ x : ℝ, (6 * x^2 + 111 * x + 1) / (2 * x + 37) = 3 * x + 1 ∧ x = -18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l897_89745


namespace NUMINAMATH_CALUDE_equation_pattern_l897_89792

theorem equation_pattern (n : ℕ) (hn : n ≥ 1) :
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_equation_pattern_l897_89792


namespace NUMINAMATH_CALUDE_leahs_birdseed_supply_l897_89701

/-- Represents the number of weeks Leah can feed her birds without going back to the store -/
def weeks_of_feed (boxes_bought : ℕ) (boxes_in_pantry : ℕ) (parrot_consumption : ℕ) (cockatiel_consumption : ℕ) (grams_per_box : ℕ) : ℕ :=
  let total_boxes := boxes_bought + boxes_in_pantry
  let total_grams := total_boxes * grams_per_box
  let weekly_consumption := parrot_consumption + cockatiel_consumption
  total_grams / weekly_consumption

/-- Theorem stating that Leah can feed her birds for 12 weeks without going back to the store -/
theorem leahs_birdseed_supply : weeks_of_feed 3 5 100 50 225 = 12 := by
  sorry

end NUMINAMATH_CALUDE_leahs_birdseed_supply_l897_89701


namespace NUMINAMATH_CALUDE_prime_divisibility_problem_l897_89798

theorem prime_divisibility_problem (p n : ℕ) : 
  p.Prime → 
  n > 0 → 
  n ≤ 2 * p → 
  (n ^ (p - 1) ∣ (p - 1) ^ n + 1) → 
  ((p = 2 ∧ n = 2) ∨ (p = 3 ∧ n = 3) ∨ n = 1) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_problem_l897_89798


namespace NUMINAMATH_CALUDE_house_coloring_l897_89706

theorem house_coloring (n : ℕ) (h : n ≥ 2) :
  ∃ (f : Fin n → Fin n) (c : Fin n → Fin 3),
    (∀ i : Fin n, f i ≠ i) ∧
    (∀ i j : Fin n, i ≠ j → f i ≠ f j) ∧
    (∀ i : Fin n, c i ≠ c (f i)) :=
by sorry

end NUMINAMATH_CALUDE_house_coloring_l897_89706


namespace NUMINAMATH_CALUDE_at_least_eight_nonzero_digits_l897_89708

/-- Given a natural number n, returns a number consisting of n repeating 9's -/
def repeating_nines (n : ℕ) : ℕ := 10^n - 1

/-- Counts the number of non-zero digits in the decimal representation of a natural number -/
def count_nonzero_digits (k : ℕ) : ℕ := sorry

theorem at_least_eight_nonzero_digits 
  (k : ℕ) (n : ℕ) (h1 : k > 0) (h2 : k % repeating_nines n = 0) : 
  count_nonzero_digits k ≥ 8 := by sorry

end NUMINAMATH_CALUDE_at_least_eight_nonzero_digits_l897_89708


namespace NUMINAMATH_CALUDE_ann_found_blocks_l897_89749

/-- Given that Ann initially had 9 blocks and ended up with 53 blocks,
    prove that she found 44 blocks. -/
theorem ann_found_blocks (initial_blocks : ℕ) (final_blocks : ℕ) :
  initial_blocks = 9 →
  final_blocks = 53 →
  final_blocks - initial_blocks = 44 := by
  sorry

end NUMINAMATH_CALUDE_ann_found_blocks_l897_89749


namespace NUMINAMATH_CALUDE_f_derivative_at_sqrt2_over_2_l897_89796

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem f_derivative_at_sqrt2_over_2 :
  (deriv f) (Real.sqrt 2 / 2) = -3/2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_sqrt2_over_2_l897_89796


namespace NUMINAMATH_CALUDE_staff_distribution_ways_l897_89761

/-- The number of ways to distribute n indistinguishable objects among k distinct containers,
    with each container receiving at least min_per_container and at most max_per_container objects. -/
def distribute_objects (n : ℕ) (k : ℕ) (min_per_container : ℕ) (max_per_container : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 90 ways to distribute 5 staff members among 3 schools
    with each school receiving at least 1 and at most 2 staff members. -/
theorem staff_distribution_ways : distribute_objects 5 3 1 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_staff_distribution_ways_l897_89761


namespace NUMINAMATH_CALUDE_number_puzzle_l897_89753

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 15) = 75 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l897_89753


namespace NUMINAMATH_CALUDE_simplify_expression_value_given_condition_value_given_equations_l897_89710

-- Part 1
theorem simplify_expression (x y : ℝ) :
  3*(x+y)^2 - 7*(x+y) + 8*(x+y)^2 + 6*(x+y) = 11*(x+y)^2 - (x+y) := by sorry

-- Part 2
theorem value_given_condition (a : ℝ) (h : a^2 + 2*a = 3) :
  3*a^2 + 6*a - 14 = -5 := by sorry

-- Part 3
theorem value_given_equations (a b c d : ℝ) 
  (h1 : a - 3*b = 3) (h2 : 2*b + c = 5) (h3 : c - 4*d = -7) :
  (a - 2*b) - (3*b - c) - (c + 4*d) = -9 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_value_given_condition_value_given_equations_l897_89710


namespace NUMINAMATH_CALUDE_wrench_handle_length_l897_89770

/-- Represents the inverse relationship between force and handle length -/
def inverse_relation (force : ℝ) (length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

theorem wrench_handle_length
  (force₁ : ℝ) (length₁ : ℝ) (force₂ : ℝ) (length₂ : ℝ)
  (h_inverse : inverse_relation force₁ length₁ ∧ inverse_relation force₂ length₂)
  (h_force₁ : force₁ = 300)
  (h_length₁ : length₁ = 12)
  (h_force₂ : force₂ = 400) :
  length₂ = 9 := by
  sorry

end NUMINAMATH_CALUDE_wrench_handle_length_l897_89770


namespace NUMINAMATH_CALUDE_square_triangle_area_l897_89778

theorem square_triangle_area (x : ℝ) : 
  x > 0 →
  (3 * x) ^ 2 + (4 * x) ^ 2 + (1 / 2) * (3 * x) * (4 * x) = 962 →
  x = Real.sqrt 31 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_area_l897_89778


namespace NUMINAMATH_CALUDE_election_votes_count_l897_89748

theorem election_votes_count :
  ∀ (total_votes : ℕ) (harold_percentage : ℚ) (jacquie_percentage : ℚ),
    harold_percentage = 60 / 100 →
    jacquie_percentage = 1 - harold_percentage →
    (harold_percentage * total_votes : ℚ) - (jacquie_percentage * total_votes : ℚ) = 24 →
    total_votes = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_election_votes_count_l897_89748


namespace NUMINAMATH_CALUDE_toms_profit_l897_89750

/-- Calculate Tom's total profit from lawn mowing and side jobs --/
theorem toms_profit (small_lawns : ℕ) (small_price : ℕ)
                    (medium_lawns : ℕ) (medium_price : ℕ)
                    (large_lawns : ℕ) (large_price : ℕ)
                    (gas_expense : ℕ) (maintenance_expense : ℕ)
                    (weed_jobs : ℕ) (weed_price : ℕ)
                    (hedge_jobs : ℕ) (hedge_price : ℕ)
                    (rake_jobs : ℕ) (rake_price : ℕ) :
  small_lawns = 4 →
  small_price = 12 →
  medium_lawns = 3 →
  medium_price = 15 →
  large_lawns = 1 →
  large_price = 20 →
  gas_expense = 17 →
  maintenance_expense = 5 →
  weed_jobs = 2 →
  weed_price = 10 →
  hedge_jobs = 3 →
  hedge_price = 8 →
  rake_jobs = 1 →
  rake_price = 12 →
  (small_lawns * small_price + medium_lawns * medium_price + large_lawns * large_price +
   weed_jobs * weed_price + hedge_jobs * hedge_price + rake_jobs * rake_price) -
  (gas_expense + maintenance_expense) = 147 :=
by sorry

end NUMINAMATH_CALUDE_toms_profit_l897_89750


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l897_89715

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l897_89715


namespace NUMINAMATH_CALUDE_probability_sum_23_l897_89772

/-- Represents a 20-faced die with specific numbered faces and one blank face -/
structure SpecialDie :=
  (numbered_faces : List Nat)
  (blank_face : Unit)

/-- Defines Die A with faces 1-18, 20, and one blank -/
def dieA : SpecialDie :=
  { numbered_faces := List.range 18 ++ [20],
    blank_face := () }

/-- Defines Die B with faces 1-7, 9-20, and one blank -/
def dieB : SpecialDie :=
  { numbered_faces := List.range 7 ++ List.range' 9 20,
    blank_face := () }

/-- Calculates the probability of rolling a sum of 23 with two specific dice -/
def probabilitySum23 (d1 d2 : SpecialDie) : Rat :=
  sorry

theorem probability_sum_23 :
  probabilitySum23 dieA dieB = 7 / 200 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_23_l897_89772


namespace NUMINAMATH_CALUDE_tangent_line_at_ln2_max_k_for_f_greater_g_l897_89723

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x / (Real.exp x - 1)

def g (k : ℕ) (x : ℝ) : ℝ := k / (x + 1)

def tangent_line (x : ℝ) : ℝ := -2 * x + 2 * Real.log 2 + 2

theorem tangent_line_at_ln2 (x : ℝ) (h : x > 0) :
  tangent_line x = -2 * x + 2 * Real.log 2 + 2 :=
sorry

theorem max_k_for_f_greater_g :
  ∃ (k : ℕ), k = 3 ∧ 
  (∀ (x : ℝ), x > 0 → f x > g k x) ∧
  (∀ (k' : ℕ), k' > k → ∃ (x : ℝ), x > 0 ∧ f x ≤ g k' x) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_ln2_max_k_for_f_greater_g_l897_89723


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l897_89789

theorem mod_equivalence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -4378 [ZMOD 6] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l897_89789


namespace NUMINAMATH_CALUDE_patricia_hair_length_l897_89704

/-- Given Patricia's hair growth scenario, prove the desired hair length after donation -/
theorem patricia_hair_length 
  (current_length : ℕ) 
  (donation_length : ℕ) 
  (growth_needed : ℕ) 
  (h1 : current_length = 14)
  (h2 : donation_length = 23)
  (h3 : growth_needed = 21) : 
  current_length + growth_needed - donation_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_patricia_hair_length_l897_89704


namespace NUMINAMATH_CALUDE_unique_zero_of_increasing_cubic_l897_89799

/-- Given an increasing function f(x) = x^3 + bx + c on [-1, 1] with f(1/2) * f(-1/2) < 0,
    prove that f has exactly one zero in [-1, 1]. -/
theorem unique_zero_of_increasing_cubic {b c : ℝ} :
  let f : ℝ → ℝ := λ x ↦ x^3 + b*x + c
  (∀ x y, x ∈ [-1, 1] → y ∈ [-1, 1] → x < y → f x < f y) →
  f (1/2) * f (-1/2) < 0 →
  ∃! x, x ∈ [-1, 1] ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_of_increasing_cubic_l897_89799


namespace NUMINAMATH_CALUDE_agent_007_encryption_possible_l897_89787

theorem agent_007_encryption_possible : ∃ (m n : ℕ), (0.07 : ℝ) = 1 / m + 1 / n := by
  sorry

end NUMINAMATH_CALUDE_agent_007_encryption_possible_l897_89787


namespace NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l897_89746

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 145 → (∀ a b : ℕ, a^2 - b^2 = 145 → x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 433 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l897_89746


namespace NUMINAMATH_CALUDE_debt_payment_difference_l897_89741

/-- Given a debt paid in 40 installments with specific conditions, 
    prove the difference between later and earlier payments. -/
theorem debt_payment_difference (first_payment : ℝ) (average_payment : ℝ) 
    (h1 : first_payment = 410)
    (h2 : average_payment = 442.5) : 
    ∃ (difference : ℝ), 
      20 * first_payment + 20 * (first_payment + difference) = 40 * average_payment ∧ 
      difference = 65 := by
  sorry

end NUMINAMATH_CALUDE_debt_payment_difference_l897_89741


namespace NUMINAMATH_CALUDE_honey_ratio_proof_l897_89771

/-- Given the conditions of James' honey production and jar requirements, 
    prove that the ratio of honey his friend is bringing jars for to the total honey produced is 1:2 -/
theorem honey_ratio_proof (hives : ℕ) (honey_per_hive : ℝ) (jar_capacity : ℝ) (james_jars : ℕ) 
  (h1 : hives = 5)
  (h2 : honey_per_hive = 20)
  (h3 : jar_capacity = 0.5)
  (h4 : james_jars = 100) :
  (↑james_jars : ℝ) / ((↑hives * honey_per_hive) / jar_capacity) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_honey_ratio_proof_l897_89771


namespace NUMINAMATH_CALUDE_line_segment_length_l897_89705

/-- Given points A, B, C, D, and E on a line in that order, prove that CD = 3 cm -/
theorem line_segment_length (A B C D E : ℝ) : 
  (B - A = 2) → 
  (C - A = 5) → 
  (D - B = 6) → 
  (∃ x, E - D = x) → 
  (E - B = 8) → 
  (E - A < 12) → 
  (D - C = 3) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l897_89705


namespace NUMINAMATH_CALUDE_remaining_amount_is_correct_l897_89718

-- Define the base-8 representation of Max's savings
def max_savings_base8 : ℕ := 5273

-- Define the cost of the ticket in base 10
def ticket_cost : ℕ := 1500

-- Function to convert from base 8 to base 10
def base8_to_base10 (n : ℕ) : ℕ :=
  let digits := n.digits 8
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (8^i)) 0

-- Theorem to prove
theorem remaining_amount_is_correct :
  base8_to_base10 max_savings_base8 - ticket_cost = 1247 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_is_correct_l897_89718


namespace NUMINAMATH_CALUDE_no_integer_satisfies_condition_l897_89713

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_integer_satisfies_condition : 
  ¬ ∃ n : ℕ+, (n : ℕ) % sum_of_digits n = 0 → sum_of_digits (n * sum_of_digits n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_satisfies_condition_l897_89713


namespace NUMINAMATH_CALUDE_smallest_total_books_satisfying_conditions_l897_89736

/-- Represents the number of books for each subject -/
structure BookCounts where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Checks if the given book counts satisfy the required ratios -/
def satisfiesRatios (books : BookCounts) : Prop :=
  3 * books.chemistry = 2 * books.physics ∧
  4 * books.biology = 3 * books.chemistry

/-- Calculates the total number of books -/
def totalBooks (books : BookCounts) : ℕ :=
  books.physics + books.chemistry + books.biology

/-- Theorem stating the smallest possible total number of books satisfying the conditions -/
theorem smallest_total_books_satisfying_conditions :
  ∃ (books : BookCounts),
    satisfiesRatios books ∧
    totalBooks books > 3000 ∧
    ∀ (other : BookCounts),
      satisfiesRatios other → totalBooks other > 3000 →
      totalBooks books ≤ totalBooks other :=
by sorry

end NUMINAMATH_CALUDE_smallest_total_books_satisfying_conditions_l897_89736


namespace NUMINAMATH_CALUDE_smallest_fraction_given_inequalities_l897_89757

theorem smallest_fraction_given_inequalities :
  ∀ r s : ℤ, 3 * r ≥ 2 * s - 3 → 4 * s ≥ r + 12 → 
  (∃ r' s' : ℤ, r' * s = r * s' ∧ s' > 0 ∧ r' * 2 = s') →
  ∀ r' s' : ℤ, r' * s = r * s' ∧ s' > 0 → r' * 2 ≤ s' :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_given_inequalities_l897_89757


namespace NUMINAMATH_CALUDE_truck_journey_l897_89742

theorem truck_journey (north_distance east_distance total_distance : ℝ) 
  (h1 : north_distance = 40)
  (h2 : total_distance = 50)
  (h3 : total_distance^2 = north_distance^2 + east_distance^2) :
  east_distance = 30 := by
  sorry

end NUMINAMATH_CALUDE_truck_journey_l897_89742


namespace NUMINAMATH_CALUDE_sand_remaining_l897_89714

/-- Calculates the remaining amount of sand in a truck after transit -/
theorem sand_remaining (initial_sand lost_sand : ℝ) :
  initial_sand ≥ 0 →
  lost_sand ≥ 0 →
  lost_sand ≤ initial_sand →
  initial_sand - lost_sand = initial_sand - lost_sand :=
by
  sorry

#check sand_remaining 4.1 2.4

end NUMINAMATH_CALUDE_sand_remaining_l897_89714


namespace NUMINAMATH_CALUDE_value_of_m_l897_89724

/-- A function f(x) is a direct proportion function with respect to x if f(x) = kx for some constant k ≠ 0 -/
def IsDirectProportionFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- A function f(x) passes through the second and fourth quadrants if f(x) < 0 for x > 0 and f(x) > 0 for x < 0 -/
def PassesThroughSecondAndFourthQuadrants (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x < 0) ∧ (∀ x < 0, f x > 0)

/-- The main theorem -/
theorem value_of_m (m : ℝ) :
  IsDirectProportionFunction (fun x ↦ (m - 2) * x^(m^2 - 8)) ∧
  PassesThroughSecondAndFourthQuadrants (fun x ↦ (m - 2) * x^(m^2 - 8)) →
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_m_l897_89724


namespace NUMINAMATH_CALUDE_system_solution_l897_89707

theorem system_solution (x y k : ℝ) : 
  x + 2*y = k - 1 →
  2*x + y = 5*k + 4 →
  x + y = 5 →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l897_89707


namespace NUMINAMATH_CALUDE_initial_group_size_l897_89790

theorem initial_group_size (X : ℕ) : 
  X - 6 + 5 - 2 + 3 = 13 → X = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_group_size_l897_89790


namespace NUMINAMATH_CALUDE_divisibility_by_five_l897_89717

theorem divisibility_by_five (n : ℕ) : 
  (2^(3*n + 5) + 3^(n + 1)) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l897_89717


namespace NUMINAMATH_CALUDE_polygon_perimeter_l897_89775

/-- The perimeter of a polygon formed by removing a right triangle from a rectangle. -/
theorem polygon_perimeter (rectangle_length : ℝ) (rectangle_width : ℝ) (triangle_height : ℝ) :
  rectangle_length = 10 →
  rectangle_width = 6 →
  triangle_height = 4 →
  ∃ (polygon_perimeter : ℝ),
    polygon_perimeter = 22 + 2 * Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_polygon_perimeter_l897_89775


namespace NUMINAMATH_CALUDE_james_tylenol_intake_l897_89737

/-- Calculates the total milligrams of Tylenol taken per day given the number of tablets per dose,
    milligrams per tablet, hours between doses, and hours in a day. -/
def tylenolPerDay (tabletsPerDose : ℕ) (mgPerTablet : ℕ) (hoursBetweenDoses : ℕ) (hoursInDay : ℕ) : ℕ :=
  let mgPerDose := tabletsPerDose * mgPerTablet
  let dosesPerDay := hoursInDay / hoursBetweenDoses
  mgPerDose * dosesPerDay

/-- Proves that James takes 3000 mg of Tylenol per day given the specified conditions. -/
theorem james_tylenol_intake : tylenolPerDay 2 375 6 24 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_james_tylenol_intake_l897_89737


namespace NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l897_89722

/-- The speed of a train excluding stoppages, given its speed including stoppages and stop time. -/
theorem train_speed_excluding_stoppages 
  (speed_with_stops : ℝ) 
  (stop_time : ℝ) 
  (h1 : speed_with_stops = 30) 
  (h2 : stop_time = 24) : 
  speed_with_stops * (60 - stop_time) / 60 = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l897_89722


namespace NUMINAMATH_CALUDE_lowest_price_pet_food_l897_89751

/-- Calculates the final price of a pet food container after two consecutive discounts -/
def final_price (msrp : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  msrp * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the lowest possible price of a $35 pet food container
    after a maximum 30% discount and an additional 20% discount is $19.60 -/
theorem lowest_price_pet_food :
  final_price 35 0.3 0.2 = 19.60 := by
  sorry

end NUMINAMATH_CALUDE_lowest_price_pet_food_l897_89751


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_l897_89756

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_f_at_pi : 
  deriv f π = -(1 / π^2) := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_l897_89756


namespace NUMINAMATH_CALUDE_roots_sum_fourth_powers_l897_89765

theorem roots_sum_fourth_powers (c d : ℝ) : 
  c^2 - 6*c + 8 = 0 → 
  d^2 - 6*d + 8 = 0 → 
  c^4 + c^3*d + d^3*c + d^4 = 432 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_fourth_powers_l897_89765


namespace NUMINAMATH_CALUDE_prob_two_red_or_blue_is_one_fifth_l897_89720

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability of drawing two marbles sequentially without replacement
    where both marbles are either red or blue -/
def prob_two_red_or_blue (counts : MarbleCounts) : ℚ :=
  let total := counts.red + counts.blue + counts.green
  let red_or_blue := counts.red + counts.blue
  (red_or_blue / total) * ((red_or_blue - 1) / (total - 1))

/-- Theorem stating that the probability of drawing two red or blue marbles
    from a bag with 4 red, 3 blue, and 8 green marbles is 1/5 -/
theorem prob_two_red_or_blue_is_one_fifth :
  prob_two_red_or_blue ⟨4, 3, 8⟩ = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_or_blue_is_one_fifth_l897_89720


namespace NUMINAMATH_CALUDE_man_speed_man_speed_is_6_l897_89759

/-- Calculates the speed of a man running opposite to a train --/
theorem man_speed (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (5/18)
  let relative_speed := train_length / passing_time
  let man_speed_mps := relative_speed - train_speed_mps
  let man_speed_kmph := man_speed_mps * (18/5)
  man_speed_kmph

/-- The speed of the man is 6 kmph --/
theorem man_speed_is_6 :
  man_speed 220 60 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_is_6_l897_89759


namespace NUMINAMATH_CALUDE_challenge_probabilities_l897_89793

/-- A challenge with 3 equally difficult questions -/
structure Challenge where
  num_questions : ℕ := 3
  num_chances : ℕ := 3
  correct_prob : ℝ := 0.7

/-- The probability of passing the challenge on the second attempt -/
def prob_pass_second_attempt (c : Challenge) : ℝ :=
  (1 - c.correct_prob) * c.correct_prob

/-- The overall probability of passing the challenge -/
def prob_pass_challenge (c : Challenge) : ℝ :=
  1 - (1 - c.correct_prob) ^ c.num_chances

/-- Theorem stating the probabilities for the given challenge -/
theorem challenge_probabilities (c : Challenge) :
  prob_pass_second_attempt c = 0.21 ∧ prob_pass_challenge c = 0.973 := by
  sorry

#check challenge_probabilities

end NUMINAMATH_CALUDE_challenge_probabilities_l897_89793


namespace NUMINAMATH_CALUDE_semicircular_window_perimeter_l897_89716

/-- The perimeter of a semicircular window with diameter d is (π * d) / 2 + d -/
theorem semicircular_window_perimeter (d : ℝ) (h : d = 63) :
  (π * d) / 2 + d = (π * 63) / 2 + 63 :=
by sorry

end NUMINAMATH_CALUDE_semicircular_window_perimeter_l897_89716


namespace NUMINAMATH_CALUDE_apple_tree_bearing_time_l897_89791

def time_to_bear_fruit (age_planted : ℕ) (age_first_apple : ℕ) : ℕ :=
  age_first_apple - age_planted

theorem apple_tree_bearing_time :
  let age_planted : ℕ := 4
  let age_first_apple : ℕ := 11
  time_to_bear_fruit age_planted age_first_apple = 7 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_bearing_time_l897_89791


namespace NUMINAMATH_CALUDE_inequality_reversal_l897_89730

theorem inequality_reversal (a b : ℝ) (h : a > b) : ∃ m : ℝ, ¬(m * a < m * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l897_89730


namespace NUMINAMATH_CALUDE_reflect_H_twice_l897_89726

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the parallelogram EFGH
def E : Point2D := ⟨3, 6⟩
def F : Point2D := ⟨5, 10⟩
def G : Point2D := ⟨7, 6⟩
def H : Point2D := ⟨5, 2⟩

-- Define reflection across x-axis
def reflectX (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

-- Define reflection across y = x + 2
def reflectYXPlus2 (p : Point2D) : Point2D :=
  ⟨p.y - 2, p.x + 2⟩

-- Theorem statement
theorem reflect_H_twice (h : Point2D) :
  h = H →
  reflectYXPlus2 (reflectX h) = ⟨-4, 7⟩ :=
by sorry

end NUMINAMATH_CALUDE_reflect_H_twice_l897_89726


namespace NUMINAMATH_CALUDE_jungkook_has_smallest_number_l897_89764

def yoongi_number : ℕ := 7
def jungkook_number : ℕ := 6
def yuna_number : ℕ := 9

theorem jungkook_has_smallest_number :
  jungkook_number ≤ yoongi_number ∧ jungkook_number ≤ yuna_number :=
sorry

end NUMINAMATH_CALUDE_jungkook_has_smallest_number_l897_89764


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_fourths_l897_89786

theorem tan_thirteen_pi_fourths : Real.tan (13 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_fourths_l897_89786


namespace NUMINAMATH_CALUDE_tangent_line_at_2_2_increasing_intervals_decreasing_interval_l897_89721

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2_2 :
  ∃ (a b c : ℝ), a * 2 + b * 2 + c = 0 ∧
  ∀ (x y : ℝ), y = f x → (y - f 2) = f_derivative 2 * (x - 2) →
  a * x + b * y + c = 0 :=
sorry

-- Theorem for increasing intervals
theorem increasing_intervals :
  ∀ x, (x < -1 ∨ x > 1) → f_derivative x > 0 :=
sorry

-- Theorem for decreasing interval
theorem decreasing_interval :
  ∀ x, -1 < x ∧ x < 1 → f_derivative x < 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_2_increasing_intervals_decreasing_interval_l897_89721


namespace NUMINAMATH_CALUDE_joan_has_nine_balloons_l897_89712

/-- The number of blue balloons that Sally has -/
def sally_balloons : ℕ := 5

/-- The number of blue balloons that Jessica has -/
def jessica_balloons : ℕ := 2

/-- The total number of blue balloons -/
def total_balloons : ℕ := 16

/-- The number of blue balloons that Joan has -/
def joan_balloons : ℕ := total_balloons - (sally_balloons + jessica_balloons)

theorem joan_has_nine_balloons : joan_balloons = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_nine_balloons_l897_89712


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_l897_89767

/-- Represents the daily sales and profit scenario for a clothing item -/
structure ClothingSales where
  initialSales : ℕ  -- Initial daily sales
  initialProfit : ℕ  -- Initial profit per piece in yuan
  salesIncrease : ℕ  -- Increase in daily sales per yuan of price reduction
  targetProfit : ℕ  -- Target daily profit in yuan

/-- Calculates the daily profit given a price reduction -/
def dailyProfit (cs : ClothingSales) (priceReduction : ℕ) : ℕ :=
  (cs.initialProfit - priceReduction) * (cs.initialSales + cs.salesIncrease * priceReduction)

/-- Theorem stating that price reductions of 4 or 36 yuan achieve the target profit -/
theorem price_reduction_achieves_target (cs : ClothingSales) 
  (h1 : cs.initialSales = 20)
  (h2 : cs.initialProfit = 44)
  (h3 : cs.salesIncrease = 5)
  (h4 : cs.targetProfit = 1600) :
  (dailyProfit cs 4 = cs.targetProfit) ∧ (dailyProfit cs 36 = cs.targetProfit) :=
by
  sorry

#eval dailyProfit { initialSales := 20, initialProfit := 44, salesIncrease := 5, targetProfit := 1600 } 4
#eval dailyProfit { initialSales := 20, initialProfit := 44, salesIncrease := 5, targetProfit := 1600 } 36

end NUMINAMATH_CALUDE_price_reduction_achieves_target_l897_89767


namespace NUMINAMATH_CALUDE_count_1000_digit_integers_l897_89728

/-- Represents the count of n-digit numbers ending with 1 or 9 -/
def b (n : ℕ) : ℕ := sorry

/-- Represents the count of n-digit numbers ending with 3 or 7 -/
def c (n : ℕ) : ℕ := sorry

/-- Represents the count of n-digit numbers ending with 5 -/
def d (n : ℕ) : ℕ := sorry

/-- All digits are odd -/
axiom all_digits_odd : ∀ n, b n + c n + d n > 0

/-- Adjacent digits differ by 2 -/
axiom adjacent_digits_differ_by_two :
  ∀ n, b (n + 1) = c n ∧ c (n + 1) = 2 * d n + b n ∧ d (n + 1) = c n

/-- Base cases -/
axiom base_cases : b 1 = 2 ∧ c 1 = 2 ∧ d 1 = 1

/-- The main theorem -/
theorem count_1000_digit_integers :
  b 1000 + c 1000 + d 1000 = 8 * 3^499 :=
sorry

end NUMINAMATH_CALUDE_count_1000_digit_integers_l897_89728


namespace NUMINAMATH_CALUDE_no_consecutive_product_for_nine_power_minus_seven_l897_89777

theorem no_consecutive_product_for_nine_power_minus_seven :
  ∀ n : ℕ, ¬∃ k : ℕ, 9^n - 7 = k * (k + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_consecutive_product_for_nine_power_minus_seven_l897_89777


namespace NUMINAMATH_CALUDE_original_number_proof_l897_89702

theorem original_number_proof : 
  ∀ x : ℝ, ((x / 8) * 16 + 20) / 4 = 34 → x = 58 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l897_89702


namespace NUMINAMATH_CALUDE_solution_when_a_neg3_m_0_range_of_a_for_real_roots_range_of_m_when_a_0_l897_89768

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a + 3
def g (m : ℝ) (x : ℝ) : ℝ := m*x + 5 - 2*m

-- Question 1
theorem solution_when_a_neg3_m_0 :
  {x : ℝ | f (-3) x - g 0 x = 0} = {-1, 5} := by sorry

-- Question 2
theorem range_of_a_for_real_roots :
  {a : ℝ | ∃ x ∈ Set.Icc (-1) 1, f a x = 0} = Set.Icc (-8) 0 := by sorry

-- Question 3
theorem range_of_m_when_a_0 :
  {m : ℝ | ∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Icc 1 4, f 0 x₁ = g m x₂} =
  Set.Iic (-3) ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_solution_when_a_neg3_m_0_range_of_a_for_real_roots_range_of_m_when_a_0_l897_89768


namespace NUMINAMATH_CALUDE_cos_sin_negative_225_deg_l897_89731

theorem cos_sin_negative_225_deg : Real.cos (-225 * π / 180) + Real.sin (-225 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_negative_225_deg_l897_89731


namespace NUMINAMATH_CALUDE_sum_of_integers_l897_89743

theorem sum_of_integers (x y : ℕ+) (h1 : x.val^2 + y.val^2 = 181) (h2 : x.val * y.val = 90) :
  x.val + y.val = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l897_89743


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l897_89769

theorem wire_cutting_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece = 10 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 35 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l897_89769


namespace NUMINAMATH_CALUDE_four_pepperoni_slices_left_l897_89762

/-- Represents the pizza sharing scenario -/
structure PizzaSharing where
  total_people : ℕ
  pepperoni_slices : ℕ
  cheese_slices : ℕ
  cheese_left : ℕ
  pepperoni_only_eaters : ℕ

/-- Calculate the number of pepperoni slices left -/
def pepperoni_left (ps : PizzaSharing) : ℕ :=
  let cheese_eaten := ps.cheese_slices - ps.cheese_left
  let slices_per_person := cheese_eaten / (ps.total_people - ps.pepperoni_only_eaters)
  let pepperoni_eaten := slices_per_person * (ps.total_people - ps.pepperoni_only_eaters) + 
                         slices_per_person * ps.pepperoni_only_eaters
  ps.pepperoni_slices - pepperoni_eaten

/-- Theorem stating that given the conditions, 4 pepperoni slices are left -/
theorem four_pepperoni_slices_left : 
  ∀ (ps : PizzaSharing), 
  ps.total_people = 4 ∧ 
  ps.pepperoni_slices = 16 ∧ 
  ps.cheese_slices = 16 ∧ 
  ps.cheese_left = 7 ∧ 
  ps.pepperoni_only_eaters = 1 →
  pepperoni_left ps = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_pepperoni_slices_left_l897_89762


namespace NUMINAMATH_CALUDE_measure_8_and_5_cm_l897_89758

-- Define the marks on the ruler
def ruler_marks : List ℕ := [0, 7, 11]

-- Define a function to check if a length can be measured
def can_measure (length : ℕ) : Prop :=
  ∃ (a b c : ℤ), a * ruler_marks[1] + b * ruler_marks[2] + c * (ruler_marks[2] - ruler_marks[1]) = length

-- Theorem statement
theorem measure_8_and_5_cm :
  can_measure 8 ∧ can_measure 5 :=
by sorry

end NUMINAMATH_CALUDE_measure_8_and_5_cm_l897_89758


namespace NUMINAMATH_CALUDE_square_sum_formula_l897_89744

theorem square_sum_formula (x y c a : ℝ) 
  (h1 : x * y = 2 * c) 
  (h2 : 1 / x^2 + 1 / y^2 = 3 * a) : 
  (x + y)^2 = 12 * a * c^2 + 4 * c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_formula_l897_89744


namespace NUMINAMATH_CALUDE_additional_area_codes_l897_89763

/-- The number of available signs for area codes -/
def num_signs : ℕ := 124

/-- The number of 2-letter area codes -/
def two_letter_codes : ℕ := num_signs * (num_signs - 1)

/-- The number of 3-letter area codes -/
def three_letter_codes : ℕ := num_signs * (num_signs - 1) * (num_signs - 2)

/-- The additional number of area codes created with the 3-letter system compared to the 2-letter system -/
theorem additional_area_codes :
  three_letter_codes - two_letter_codes = 1845396 :=
by sorry

end NUMINAMATH_CALUDE_additional_area_codes_l897_89763


namespace NUMINAMATH_CALUDE_teaching_team_formation_l897_89740

def chinese_teachers : ℕ := 2
def math_teachers : ℕ := 2
def english_teachers : ℕ := 4
def team_size : ℕ := 5

def ways_to_form_team : ℕ := 
  Nat.choose english_teachers 1 + 
  (Nat.choose chinese_teachers 1 * Nat.choose english_teachers 2) +
  (Nat.choose math_teachers 1 * Nat.choose english_teachers 2) +
  (Nat.choose chinese_teachers 1 * Nat.choose math_teachers 1 * Nat.choose english_teachers 3)

theorem teaching_team_formation :
  ways_to_form_team = 44 :=
by sorry

end NUMINAMATH_CALUDE_teaching_team_formation_l897_89740


namespace NUMINAMATH_CALUDE_function_machine_output_l897_89780

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 15 then
    step1 + 10
  else
    step1 - 3

theorem function_machine_output : function_machine 12 = 33 := by
  sorry

end NUMINAMATH_CALUDE_function_machine_output_l897_89780


namespace NUMINAMATH_CALUDE_symmetric_points_range_l897_89700

noncomputable section

open Real

def e : ℝ := Real.exp 1

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x

def g (x : ℝ) : ℝ := exp x

def h (x : ℝ) : ℝ := log x

theorem symmetric_points_range (a : ℝ) :
  (∃ x y : ℝ, 1/e ≤ x ∧ x ≤ e ∧ 1/e ≤ y ∧ y ≤ e ∧
    f a x = g y ∧ f a y = g x) →
  1 ≤ a ∧ a ≤ e + 1/e :=
by sorry

end

end NUMINAMATH_CALUDE_symmetric_points_range_l897_89700


namespace NUMINAMATH_CALUDE_remainder_theorem_l897_89784

theorem remainder_theorem : 
  (2^300 + 405) % (2^150 + 2^75 + 1) = 404 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l897_89784


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l897_89782

/-- Convert a list of bits (0s and 1s) to a natural number -/
def binaryToNat (bits : List Nat) : Nat :=
  bits.foldl (fun acc b => 2 * acc + b) 0

/-- The theorem to prove -/
theorem binary_arithmetic_equality :
  let a := binaryToNat [1, 1, 0, 1]
  let b := binaryToNat [1, 1, 1, 0]
  let c := binaryToNat [1, 0, 1, 1]
  let d := binaryToNat [1, 0, 0, 1]
  let e := binaryToNat [1, 0, 1]
  a + b - c + d - e = binaryToNat [1, 0, 0, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l897_89782


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l897_89734

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l897_89734


namespace NUMINAMATH_CALUDE_final_S_equals_3_pow_10_l897_89781

/-- Represents the state of the program at each iteration --/
structure ProgramState where
  S : ℕ
  i : ℕ

/-- The initial state of the program --/
def initial_state : ProgramState := { S := 1, i := 1 }

/-- The transition function for each iteration of the loop --/
def iterate (state : ProgramState) : ProgramState :=
  { S := state.S * 3, i := state.i + 1 }

/-- The final state after the loop completes --/
def final_state : ProgramState :=
  (iterate^[10]) initial_state

/-- The theorem stating that the final value of S is equal to 3^10 --/
theorem final_S_equals_3_pow_10 : final_state.S = 3^10 := by
  sorry

end NUMINAMATH_CALUDE_final_S_equals_3_pow_10_l897_89781


namespace NUMINAMATH_CALUDE_binomial_13_choose_10_l897_89760

theorem binomial_13_choose_10 : Nat.choose 13 10 = 286 := by
  sorry

end NUMINAMATH_CALUDE_binomial_13_choose_10_l897_89760
