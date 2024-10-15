import Mathlib

namespace NUMINAMATH_GPT_initial_population_l838_83834

theorem initial_population (P : ℝ) : 
  (0.9 * P * 0.85 = 2907) → P = 3801 := by
  sorry

end NUMINAMATH_GPT_initial_population_l838_83834


namespace NUMINAMATH_GPT_no_maximum_value_l838_83868

-- Define the conditions and the expression in Lean
def expression (a b c d : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2 + a*b + c*d

def condition (a b c d : ℝ) : Prop := a * d - b * c = 1

theorem no_maximum_value : ¬ ∃ M, ∀ a b c d, condition a b c d → expression a b c d ≤ M := by
  sorry

end NUMINAMATH_GPT_no_maximum_value_l838_83868


namespace NUMINAMATH_GPT_compute_difference_of_squares_l838_83837

theorem compute_difference_of_squares :
    75^2 - 25^2 = 5000 :=
by
  sorry

end NUMINAMATH_GPT_compute_difference_of_squares_l838_83837


namespace NUMINAMATH_GPT_george_painting_combinations_l838_83831

namespace Combinations

/-- George's painting problem -/
theorem george_painting_combinations :
  let colors := 10
  let colors_to_pick := 3
  let textures := 2
  ((colors) * (colors - 1) * (colors - 2) / (colors_to_pick * (colors_to_pick - 1) * 1)) * (textures ^ colors_to_pick) = 960 :=
by
  sorry

end Combinations

end NUMINAMATH_GPT_george_painting_combinations_l838_83831


namespace NUMINAMATH_GPT_product_of_real_numbers_triple_when_added_to_their_reciprocal_l838_83874

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_product_of_real_numbers_triple_when_added_to_their_reciprocal_l838_83874


namespace NUMINAMATH_GPT_a_pow_10_plus_b_pow_10_l838_83818

theorem a_pow_10_plus_b_pow_10 (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (hn : ∀ n ≥ 3, a^(n) + b^(n) = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^10 + b^10 = 123 :=
by
  sorry

end NUMINAMATH_GPT_a_pow_10_plus_b_pow_10_l838_83818


namespace NUMINAMATH_GPT_expression_value_l838_83830

theorem expression_value (a b : ℕ) (h₁ : a = 2023) (h₂ : b = 2020) :
  ((
     (3 / (a - b) + (3 * a) / (a^3 - b^3) * ((a^2 + a * b + b^2) / (a + b))) * ((2 * a + b) / (a^2 + 2 * a * b + b^2))
  ) * (3 / (a + b))) = 3 :=
by
  -- Use the provided conditions
  rw [h₁, h₂]
  -- Execute the following steps as per the mathematical solution steps 
  sorry

end NUMINAMATH_GPT_expression_value_l838_83830


namespace NUMINAMATH_GPT_find_consecutive_numbers_l838_83835

theorem find_consecutive_numbers (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c)
    (h_lcm : Nat.lcm a (Nat.lcm b c) = 660) : a = 10 ∧ b = 11 ∧ c = 12 := 
    sorry

end NUMINAMATH_GPT_find_consecutive_numbers_l838_83835


namespace NUMINAMATH_GPT_complete_the_square_l838_83858

theorem complete_the_square (x : ℝ) : x^2 + 2*x - 3 = 0 ↔ (x + 1)^2 = 4 :=
by sorry

end NUMINAMATH_GPT_complete_the_square_l838_83858


namespace NUMINAMATH_GPT_find_numbers_l838_83890

/-- Given the sums of three pairs of numbers, we prove the individual numbers. -/
theorem find_numbers (x y z : ℕ) (h1 : x + y = 40) (h2 : y + z = 50) (h3 : z + x = 70) :
  x = 30 ∧ y = 10 ∧ z = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l838_83890


namespace NUMINAMATH_GPT_smallest_n_correct_l838_83836

/-- The first term of the geometric sequence. -/
def a₁ : ℚ := 5 / 6

/-- The second term of the geometric sequence. -/
def a₂ : ℚ := 25

/-- The common ratio for the geometric sequence. -/
def r : ℚ := a₂ / a₁

/-- The nth term of the geometric sequence. -/
def a_n (n : ℕ) : ℚ := a₁ * r^(n - 1)

/-- The smallest n such that the nth term is divisible by 10^7. -/
def smallest_n : ℕ := 8

theorem smallest_n_correct :
  ∀ n : ℕ, (a₁ * r^(n - 1)) ∣ (10^7 : ℚ) ↔ n = smallest_n := 
sorry

end NUMINAMATH_GPT_smallest_n_correct_l838_83836


namespace NUMINAMATH_GPT_max_diff_y_l838_83875

theorem max_diff_y (x y z : ℕ) (h₁ : 4 < x) (h₂ : x < z) (h₃ : z < y) (h₄ : y < 10) (h₅ : y - x = 5) : y = 9 :=
sorry

end NUMINAMATH_GPT_max_diff_y_l838_83875


namespace NUMINAMATH_GPT_at_least_one_nonzero_l838_83850

theorem at_least_one_nonzero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_nonzero_l838_83850


namespace NUMINAMATH_GPT_pool_capacity_is_800_l838_83872

-- Definitions for the given problem conditions
def fill_time_all_valves : ℝ := 36
def fill_time_first_valve : ℝ := 180
def fill_time_second_valve : ℝ := 240
def third_valve_more_than_first : ℝ := 30
def third_valve_more_than_second : ℝ := 10
def leak_rate : ℝ := 20

-- Function definition for the capacity of the pool
def capacity (W : ℝ) : Prop :=
  let V1 := W / fill_time_first_valve
  let V2 := W / fill_time_second_valve
  let V3 := (W / fill_time_first_valve) + third_valve_more_than_first
  let effective_rate := V1 + V2 + V3 - leak_rate
  (W / fill_time_all_valves) = effective_rate

-- Proof statement that the capacity of the pool is 800 cubic meters
theorem pool_capacity_is_800 : capacity 800 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_pool_capacity_is_800_l838_83872


namespace NUMINAMATH_GPT_comb_5_1_eq_5_l838_83809

theorem comb_5_1_eq_5 : Nat.choose 5 1 = 5 :=
by
  sorry

end NUMINAMATH_GPT_comb_5_1_eq_5_l838_83809


namespace NUMINAMATH_GPT_smallest_n_l838_83865

theorem smallest_n (n : ℕ) (h : 503 * n % 48 = 1019 * n % 48) : n = 4 := by
  sorry

end NUMINAMATH_GPT_smallest_n_l838_83865


namespace NUMINAMATH_GPT_select_best_athlete_l838_83808

theorem select_best_athlete
  (avg_A avg_B avg_C avg_D: ℝ)
  (var_A var_B var_C var_D: ℝ)
  (h_avg_A: avg_A = 185)
  (h_avg_B: avg_B = 180)
  (h_avg_C: avg_C = 185)
  (h_avg_D: avg_D = 180)
  (h_var_A: var_A = 3.6)
  (h_var_B: var_B = 3.6)
  (h_var_C: var_C = 7.4)
  (h_var_D: var_D = 8.1) :
  (avg_A > avg_B ∧ avg_A > avg_D ∧ var_A < var_C) →
  (avg_A = 185 ∧ var_A = 3.6) :=
by
  sorry

end NUMINAMATH_GPT_select_best_athlete_l838_83808


namespace NUMINAMATH_GPT_exists_large_natural_with_high_digit_sum_l838_83800

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem exists_large_natural_with_high_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (factorial n) ≥ 10 ^ 100 :=
by sorry

end NUMINAMATH_GPT_exists_large_natural_with_high_digit_sum_l838_83800


namespace NUMINAMATH_GPT_number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l838_83856

theorem number_of_positive_integers_with_erased_digit_decreased_by_nine_times : 
  ∃ n : ℕ, 
  ∀ (m a k : ℕ),
  (m + 10^k * a + 10^(k + 1) * n = 9 * (m + 10^k * n)) → 
  m < 10^k ∧ n > 0 ∧ n < m ∧  m ≠ 0 → 
  (m + 10^k * n  = 9 * (m - a) ) ∧ 
  (m % 10 = 5 ∨ m % 10 = 0) → 
  n = 28 :=
by
  sorry

end NUMINAMATH_GPT_number_of_positive_integers_with_erased_digit_decreased_by_nine_times_l838_83856


namespace NUMINAMATH_GPT_power_of_power_eq_512_l838_83899

theorem power_of_power_eq_512 : (2^3)^3 = 512 := by
  sorry

end NUMINAMATH_GPT_power_of_power_eq_512_l838_83899


namespace NUMINAMATH_GPT_multiply_polynomials_l838_83804

def polynomial_multiplication (x : ℝ) : Prop :=
  (x^4 + 24*x^2 + 576) * (x^2 - 24) = x^6 - 13824

theorem multiply_polynomials (x : ℝ) : polynomial_multiplication x :=
by
  sorry

end NUMINAMATH_GPT_multiply_polynomials_l838_83804


namespace NUMINAMATH_GPT_motorcyclist_average_speed_l838_83803

theorem motorcyclist_average_speed :
  ∀ (t : ℝ), 120 / t = 60 * 3 → 
  3 * t / 4 = 45 :=
by
  sorry

end NUMINAMATH_GPT_motorcyclist_average_speed_l838_83803


namespace NUMINAMATH_GPT_evaluate_function_at_neg_one_l838_83839

def f (x : ℝ) : ℝ := -2 * x^2 + 1

theorem evaluate_function_at_neg_one : f (-1) = -1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_function_at_neg_one_l838_83839


namespace NUMINAMATH_GPT_minimum_discount_l838_83885

theorem minimum_discount (C M : ℝ) (profit_margin : ℝ) (x : ℝ) 
  (hC : C = 800) (hM : M = 1200) (hprofit_margin : profit_margin = 0.2) :
  (M * x - C ≥ C * profit_margin) → (x ≥ 0.8) :=
by
  -- Here, we need to solve the inequality given the conditions
  sorry

end NUMINAMATH_GPT_minimum_discount_l838_83885


namespace NUMINAMATH_GPT_simple_interest_rate_l838_83848

theorem simple_interest_rate (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (h1 : SI = 130) (h2 : P = 780) (h3 : T = 4) :
  R = 4.17 :=
sorry

end NUMINAMATH_GPT_simple_interest_rate_l838_83848


namespace NUMINAMATH_GPT_complete_the_square_l838_83893

-- Define the initial condition
def initial_eqn (x : ℝ) : Prop := x^2 - 6 * x + 5 = 0

-- Theorem statement for completing the square
theorem complete_the_square (x : ℝ) : initial_eqn x → (x - 3)^2 = 4 :=
by sorry

end NUMINAMATH_GPT_complete_the_square_l838_83893


namespace NUMINAMATH_GPT_nat_divisibility_l838_83895

theorem nat_divisibility (n : ℕ) : 27 ∣ (10^n + 18 * n - 1) :=
  sorry

end NUMINAMATH_GPT_nat_divisibility_l838_83895


namespace NUMINAMATH_GPT_number_of_uncertain_events_is_three_l838_83870

noncomputable def cloudy_day_will_rain : Prop := sorry
noncomputable def fair_coin_heads : Prop := sorry
noncomputable def two_students_same_birth_month : Prop := sorry
noncomputable def olympics_2008_in_beijing : Prop := true

def is_uncertain (event: Prop) : Prop :=
  event ∧ ¬(event = true ∨ event = false)

theorem number_of_uncertain_events_is_three :
  is_uncertain cloudy_day_will_rain ∧
  is_uncertain fair_coin_heads ∧
  is_uncertain two_students_same_birth_month ∧
  ¬is_uncertain olympics_2008_in_beijing →
  3 = 3 :=
by sorry

end NUMINAMATH_GPT_number_of_uncertain_events_is_three_l838_83870


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l838_83811

theorem polynomial_coeff_sum (A B C D : ℤ) 
  (h : ∀ x : ℤ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l838_83811


namespace NUMINAMATH_GPT_expand_expression_l838_83892

theorem expand_expression (x y : ℝ) :
  (x + 3) * (4 * x - 5 * y) = 4 * x ^ 2 - 5 * x * y + 12 * x - 15 * y :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l838_83892


namespace NUMINAMATH_GPT_milk_fraction_in_cup1_is_one_third_l838_83888

-- Define the initial state of the cups
structure CupsState where
  cup1_tea : ℚ  -- amount of tea in cup1
  cup1_milk : ℚ -- amount of milk in cup1
  cup2_tea : ℚ  -- amount of tea in cup2
  cup2_milk : ℚ -- amount of milk in cup2

def initial_cups_state : CupsState := {
  cup1_tea := 8,
  cup1_milk := 0,
  cup2_tea := 0,
  cup2_milk := 8
}

-- Function to transfer a fraction of tea from cup 1 to cup 2
def transfer_tea (s : CupsState) (frac : ℚ) : CupsState := {
  cup1_tea := s.cup1_tea * (1 - frac),
  cup1_milk := s.cup1_milk,
  cup2_tea := s.cup2_tea + s.cup1_tea * frac,
  cup2_milk := s.cup2_milk
}

-- Function to transfer a fraction of the mixture from cup 2 to cup 1
def transfer_mixture (s : CupsState) (frac : ℚ) : CupsState := {
  cup1_tea := s.cup1_tea + (frac * s.cup2_tea),
  cup1_milk := s.cup1_milk + (frac * s.cup2_milk),
  cup2_tea := s.cup2_tea * (1 - frac),
  cup2_milk := s.cup2_milk * (1 - frac)
}

-- Define the state after each transfer
def state_after_tea_transfer := transfer_tea initial_cups_state (1 / 4)
def final_state := transfer_mixture state_after_tea_transfer (1 / 3)

-- Prove the fraction of milk in the first cup is 1/3
theorem milk_fraction_in_cup1_is_one_third : 
  (final_state.cup1_milk / (final_state.cup1_tea + final_state.cup1_milk)) = 1 / 3 :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_milk_fraction_in_cup1_is_one_third_l838_83888


namespace NUMINAMATH_GPT_binary_mul_1101_111_eq_1001111_l838_83863

theorem binary_mul_1101_111_eq_1001111 :
  let n1 := 0b1101 -- binary representation of 13
  let n2 := 0b111  -- binary representation of 7
  let product := 0b1001111 -- binary representation of 79
  n1 * n2 = product :=
by
  sorry

end NUMINAMATH_GPT_binary_mul_1101_111_eq_1001111_l838_83863


namespace NUMINAMATH_GPT_problem_l838_83855

theorem problem (m : ℝ) (h : m^2 + 3 * m = -1) : m - 1 / (m + 1) = -2 :=
by
  sorry

end NUMINAMATH_GPT_problem_l838_83855


namespace NUMINAMATH_GPT_mh_range_l838_83843

theorem mh_range (x m : ℝ) (h : 1 / 3 < x ∧ x < 1 / 2) (hx : |x - m| < 1) : 
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 := 
sorry

end NUMINAMATH_GPT_mh_range_l838_83843


namespace NUMINAMATH_GPT_smallest_common_multiple_l838_83864

theorem smallest_common_multiple (n : ℕ) (h1 : n > 0) (h2 : 8 ∣ n) (h3 : 6 ∣ n) : n = 24 :=
by sorry

end NUMINAMATH_GPT_smallest_common_multiple_l838_83864


namespace NUMINAMATH_GPT_min_value_a_plus_2b_minus_3c_l838_83805

theorem min_value_a_plus_2b_minus_3c
  (a b c : ℝ)
  (h : ∀ (x y : ℝ), x + 2 * y - 3 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ x + 2 * y + 3) :
  ∃ m : ℝ, m = a + 2 * b - 3 * c ∧ m = -4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_a_plus_2b_minus_3c_l838_83805


namespace NUMINAMATH_GPT_solve_x_l838_83822

noncomputable def diamond (a b : ℝ) : ℝ := a / b

axiom diamond_assoc (a b c : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) : 
  diamond a (diamond b c) = a / (b / c)

axiom diamond_id (a : ℝ) (a_nonzero : a ≠ 0) : diamond a a = 1

theorem solve_x (x : ℝ) (h₁ : 1008 ≠ 0) (h₂ : 12 ≠ 0) (h₃ : x ≠ 0) : diamond 1008 (diamond 12 x) = 50 → x = 25 / 42 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_l838_83822


namespace NUMINAMATH_GPT_tram_length_proof_l838_83816
-- Import the necessary library

-- Define the conditions
def tram_length : ℕ := 32 -- The length of the tram we want to prove

-- The main theorem to be stated
theorem tram_length_proof (L : ℕ) (v : ℕ) 
  (h1 : v = L / 4)  -- The tram passed by Misha in 4 seconds
  (h2 : v = (L + 64) / 12)  -- The tram passed through a tunnel of 64 meters in 12 seconds
  : L = tram_length :=
by
  sorry

end NUMINAMATH_GPT_tram_length_proof_l838_83816


namespace NUMINAMATH_GPT_arrange_letters_l838_83802

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrange_letters : factorial 7 / (factorial 3 * factorial 2 * factorial 2) = 210 := 
by
  sorry

end NUMINAMATH_GPT_arrange_letters_l838_83802


namespace NUMINAMATH_GPT_ray_climbs_l838_83869

theorem ray_climbs (n : ℕ) (h1 : n % 3 = 1) (h2 : n % 5 = 3) (h3 : n % 7 = 1) (h4 : n > 15) : n = 73 :=
sorry

end NUMINAMATH_GPT_ray_climbs_l838_83869


namespace NUMINAMATH_GPT_direct_proportion_point_l838_83819

theorem direct_proportion_point (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : y₁ = k * x₁) (hx₁ : x₁ = -1) (hy₁ : y₁ = 2) (hx₂ : x₂ = 1) (hy₂ : y₂ = -2) 
  : y₂ = k * x₂ := 
by
  -- sorry will skip the proof
  sorry

end NUMINAMATH_GPT_direct_proportion_point_l838_83819


namespace NUMINAMATH_GPT_negation_of_proposition_l838_83838

theorem negation_of_proposition (x y : ℝ) :
  (¬ (x + y = 1 → xy ≤ 1)) ↔ (x + y ≠ 1 → xy > 1) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l838_83838


namespace NUMINAMATH_GPT_find_fg_satisfy_l838_83873

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) / 2
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := (Real.sin x - Real.cos x) / 2 + c

theorem find_fg_satisfy (c : ℝ) : ∀ x y : ℝ,
  Real.sin x + Real.cos y = f x + f y + g x c - g y c := 
by 
  intros;
  rw [f, g, g, f];
  sorry

end NUMINAMATH_GPT_find_fg_satisfy_l838_83873


namespace NUMINAMATH_GPT_functions_of_same_family_count_l838_83813

theorem functions_of_same_family_count : 
  (∃ (y : ℝ → ℝ), ∀ x, y x = x^2) ∧ 
  (∃ (range_set : Set ℝ), range_set = {1, 2}) → 
  ∃ n, n = 9 :=
by
  sorry

end NUMINAMATH_GPT_functions_of_same_family_count_l838_83813


namespace NUMINAMATH_GPT_fraction_human_habitable_surface_l838_83817

variable (fraction_water_coverage : ℚ)
variable (fraction_inhabitable_remaining_land : ℚ)
variable (fraction_reserved_for_agriculture : ℚ)

def fraction_inhabitable_land (f_water : ℚ) (f_inhabitable : ℚ) : ℚ :=
  (1 - f_water) * f_inhabitable

def fraction_habitable_land (f_inhabitable_land : ℚ) (f_reserved : ℚ) : ℚ :=
  f_inhabitable_land * (1 - f_reserved)

theorem fraction_human_habitable_surface 
  (h1 : fraction_water_coverage = 3/5)
  (h2 : fraction_inhabitable_remaining_land = 2/3)
  (h3 : fraction_reserved_for_agriculture = 1/2) :
  fraction_habitable_land 
    (fraction_inhabitable_land fraction_water_coverage fraction_inhabitable_remaining_land)
    fraction_reserved_for_agriculture = 2/15 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_human_habitable_surface_l838_83817


namespace NUMINAMATH_GPT_smallest_prime_number_conditions_l838_83853

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |> List.sum -- Summing the digits in base 10

def is_prime (n : ℕ) : Prop := Nat.Prime n

def smallest_prime_number (n : ℕ) : Prop :=
  is_prime n ∧ sum_of_digits n = 17 ∧ n > 200 ∧
  (∀ m : ℕ, is_prime m ∧ sum_of_digits m = 17 ∧ m > 200 → n ≤ m)

theorem smallest_prime_number_conditions (p : ℕ) : 
  smallest_prime_number p ↔ p = 197 :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_number_conditions_l838_83853


namespace NUMINAMATH_GPT_lines_parallel_iff_m_eq_1_l838_83842

-- Define the two lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y = 2 - m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * m * x + 4 * y = -16

-- Parallel lines condition
def parallel_condition (m : ℝ) : Prop := (1 * 4 - 2 * m * (1 + m) = 0) ∧ (1 * 16 - 2 * m * (m - 2) ≠ 0)

-- The theorem to prove
theorem lines_parallel_iff_m_eq_1 (m : ℝ) : l1 m = l2 m → parallel_condition m → m = 1 :=
by 
  sorry

end NUMINAMATH_GPT_lines_parallel_iff_m_eq_1_l838_83842


namespace NUMINAMATH_GPT_union_complement_with_B_l838_83825

namespace SetTheory

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Definition of the complement of A relative to U in Lean
def C_U (A U : Set ℕ) : Set ℕ := U \ A

-- Theorem statement
theorem union_complement_with_B (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {0, 1, 2, 3}) (hB : B = {2, 3, 4}) : 
  (C_U A U) ∪ B = {2, 3, 4} :=
by
  -- Proof goes here
  sorry

end SetTheory

end NUMINAMATH_GPT_union_complement_with_B_l838_83825


namespace NUMINAMATH_GPT_barbara_spent_total_l838_83861

variables (cost_steaks cost_chicken total_spent per_pound_steak per_pound_chicken : ℝ)
variables (weight_steaks weight_chicken : ℝ)

-- Defining the given conditions
def conditions :=
  per_pound_steak = 15 ∧
  weight_steaks = 4.5 ∧
  cost_steaks = per_pound_steak * weight_steaks ∧

  per_pound_chicken = 8 ∧
  weight_chicken = 1.5 ∧
  cost_chicken = per_pound_chicken * weight_chicken

-- Proving the total spent by Barbara is $79.50
theorem barbara_spent_total 
  (h : conditions per_pound_steak weight_steaks cost_steaks per_pound_chicken weight_chicken cost_chicken) : 
  total_spent = 79.5 :=
sorry

end NUMINAMATH_GPT_barbara_spent_total_l838_83861


namespace NUMINAMATH_GPT_range_of_m_l838_83824

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - 2 * m * x + m + 3
noncomputable def g (x : ℝ) : ℝ := 2^(x - 2)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ↔ -4 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l838_83824


namespace NUMINAMATH_GPT_y1_gt_y2_for_line_through_points_l838_83806

theorem y1_gt_y2_for_line_through_points (x1 y1 x2 y2 k b : ℝ) 
  (h_line_A : y1 = k * x1 + b) 
  (h_line_B : y2 = k * x2 + b) 
  (h_k_neq_0 : k ≠ 0)
  (h_k_pos : k > 0)
  (h_b_nonneg : b ≥ 0)
  (h_x1_gt_x2 : x1 > x2) : 
  y1 > y2 := 
  sorry

end NUMINAMATH_GPT_y1_gt_y2_for_line_through_points_l838_83806


namespace NUMINAMATH_GPT_combined_share_of_A_and_C_l838_83826

-- Definitions based on the conditions
def total_money : Float := 15800
def charity_investment : Float := 0.10 * total_money
def savings_investment : Float := 0.08 * total_money
def remaining_money : Float := total_money - charity_investment - savings_investment

def ratio_A : Nat := 5
def ratio_B : Nat := 9
def ratio_C : Nat := 6
def ratio_D : Nat := 5
def sum_of_ratios : Nat := ratio_A + ratio_B + ratio_C + ratio_D

def share_A : Float := (ratio_A.toFloat / sum_of_ratios.toFloat) * remaining_money
def share_C : Float := (ratio_C.toFloat / sum_of_ratios.toFloat) * remaining_money
def combined_share_A_C : Float := share_A + share_C

-- Statement to be proven
theorem combined_share_of_A_and_C : combined_share_A_C = 5700.64 := by
  sorry

end NUMINAMATH_GPT_combined_share_of_A_and_C_l838_83826


namespace NUMINAMATH_GPT_polynomial_value_at_n_plus_1_l838_83821

theorem polynomial_value_at_n_plus_1 
  (f : ℕ → ℝ) 
  (n : ℕ)
  (hdeg : ∃ m, m = n) 
  (hvalues : ∀ k (hk : k ≤ n), f k = k / (k + 1)) : 
  f (n + 1) = (n + 1 + (-1) ^ (n + 1)) / (n + 2) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_value_at_n_plus_1_l838_83821


namespace NUMINAMATH_GPT_abs_sum_of_first_six_a_sequence_terms_l838_83860

def a_sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => -5
  | n+1 => a_sequence n + 2

theorem abs_sum_of_first_six_a_sequence_terms :
  |a_sequence 0| + |a_sequence 1| + |a_sequence 2| + |a_sequence 3| + |a_sequence 4| + |a_sequence 5| = 18 := sorry

end NUMINAMATH_GPT_abs_sum_of_first_six_a_sequence_terms_l838_83860


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_unique_l838_83846

theorem right_triangle_hypotenuse_unique :
  ∃ (a b c : ℚ) (d e : ℕ), 
    (c^2 = a^2 + b^2) ∧
    (a = 10 * e + d) ∧
    (c = 10 * d + e) ∧
    (d + e = 11) ∧
    (d ≠ e) ∧
    (a = 56) ∧
    (b = 33) ∧
    (c = 65) :=
by {
  sorry
}

end NUMINAMATH_GPT_right_triangle_hypotenuse_unique_l838_83846


namespace NUMINAMATH_GPT_solve_expression_l838_83845

theorem solve_expression (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 3 * (x + 4) = 6 :=
sorry

end NUMINAMATH_GPT_solve_expression_l838_83845


namespace NUMINAMATH_GPT_center_temperature_l838_83840

-- Define the conditions as a structure
structure SquareSheet (f : ℝ × ℝ → ℝ) :=
  (temp_0: ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x, 0) = 0 ∧ f (0, x) = 0 ∧ f (1, x) = 0)
  (temp_100: ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x, 1) = 100)
  (no_radiation_loss: True) -- Just a placeholder since this condition is theoretical in nature

-- Define the claim as a theorem
theorem center_temperature (f : ℝ × ℝ → ℝ) (h : SquareSheet f) : f (0.5, 0.5) = 25 :=
by
  sorry -- Proof is not required and skipped

end NUMINAMATH_GPT_center_temperature_l838_83840


namespace NUMINAMATH_GPT_total_right_handed_players_is_correct_l838_83867

variable (total_players : ℕ)
variable (throwers : ℕ)
variable (left_handed_non_throwers_ratio : ℕ)
variable (total_right_handed_players : ℕ)

theorem total_right_handed_players_is_correct
  (h1 : total_players = 61)
  (h2 : throwers = 37)
  (h3 : left_handed_non_throwers_ratio = 1 / 3)
  (h4 : total_right_handed_players = 53) :
  total_right_handed_players = throwers + (total_players - throwers) -
    left_handed_non_throwers_ratio * (total_players - throwers) :=
by
  sorry

end NUMINAMATH_GPT_total_right_handed_players_is_correct_l838_83867


namespace NUMINAMATH_GPT_find_common_difference_l838_83894

variable {a : ℕ → ℤ}  -- Define the arithmetic sequence as a function from natural numbers to integers
variable (d : ℤ)      -- Define the common difference

-- Assume the conditions given in the problem
axiom h1 : a 2 = 14
axiom h2 : a 5 = 5

theorem find_common_difference (n : ℕ) : d = -3 :=
by {
  -- This part will be filled in by the actual proof
  sorry
}

end NUMINAMATH_GPT_find_common_difference_l838_83894


namespace NUMINAMATH_GPT_gcd_72_168_gcd_98_280_f_at_3_l838_83828

/-- 
Prove that the GCD of 72 and 168 using the method of mutual subtraction is 24.
-/
theorem gcd_72_168 : Nat.gcd 72 168 = 24 :=
sorry

/-- 
Prove that the GCD of 98 and 280 using the Euclidean algorithm is 14.
-/
theorem gcd_98_280 : Nat.gcd 98 280 = 14 :=
sorry

/-- 
Prove that the value of f(3) where f(x) = x^5 + x^3 + x^2 + x + 1 is 283 using Horner's method.
-/
def f (x : ℕ) : ℕ := x^5 + x^3 + x^2 + x + 1

theorem f_at_3 : f 3 = 283 :=
sorry

end NUMINAMATH_GPT_gcd_72_168_gcd_98_280_f_at_3_l838_83828


namespace NUMINAMATH_GPT_solve_for_x_l838_83827

theorem solve_for_x (x : ℚ) : 
  x + 5 / 6 = 11 / 18 - 2 / 9 → x = -4 / 9 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l838_83827


namespace NUMINAMATH_GPT_g_is_correct_l838_83841

-- Define the given polynomial equation
def poly_lhs (x : ℝ) : ℝ := 2 * x^5 - x^3 + 4 * x^2 + 3 * x - 5
def poly_rhs (x : ℝ) : ℝ := 7 * x^3 - 4 * x + 2

-- Define the function g(x)
def g (x : ℝ) : ℝ := -2 * x^5 + 6 * x^3 - 4 * x^2 - x + 7

-- The theorem to be proven
theorem g_is_correct : ∀ x : ℝ, poly_lhs x + g x = poly_rhs x :=
by
  intro x
  unfold poly_lhs poly_rhs g
  sorry

end NUMINAMATH_GPT_g_is_correct_l838_83841


namespace NUMINAMATH_GPT_grandfather_time_difference_l838_83810

-- Definitions based on the conditions
def treadmill_days : ℕ := 4
def miles_per_day : ℕ := 2
def monday_speed : ℕ := 6
def tuesday_speed : ℕ := 3
def wednesday_speed : ℕ := 4
def thursday_speed : ℕ := 3
def walk_speed : ℕ := 3

-- The theorem statement
theorem grandfather_time_difference :
  let monday_time := (miles_per_day : ℚ) / monday_speed
  let tuesday_time := (miles_per_day : ℚ) / tuesday_speed
  let wednesday_time := (miles_per_day : ℚ) / wednesday_speed
  let thursday_time := (miles_per_day : ℚ) / thursday_speed
  let actual_total_time := monday_time + tuesday_time + wednesday_time + thursday_time
  let walk_total_time := (treadmill_days * miles_per_day : ℚ) / walk_speed
  (walk_total_time - actual_total_time) * 60 = 80 := sorry

end NUMINAMATH_GPT_grandfather_time_difference_l838_83810


namespace NUMINAMATH_GPT_four_digit_number_sum_l838_83866

theorem four_digit_number_sum (x y z w : ℕ) (h1 : 1001 * x + 101 * y + 11 * z + 2 * w = 2003)
  (h2 : x = 1) : (x = 1 ∧ y = 9 ∧ z = 7 ∧ w = 8) ↔ (1000 * x + 100 * y + 10 * z + w = 1978) :=
by sorry

end NUMINAMATH_GPT_four_digit_number_sum_l838_83866


namespace NUMINAMATH_GPT_question_correct_statements_l838_83847

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (f : ℝ → ℝ) : ∀ x y : ℝ, f (x + y) = f x + f y
axiom periodicity (f : ℝ → ℝ) : f 2 = 0

theorem question_correct_statements : 
  (∀ x : ℝ, f (x + 2) = f x) ∧ -- ensuring the function is periodic
  (∀ x : ℝ, f x = -f (-x)) ∧ -- ensuring the function is odd
  (∀ x : ℝ, f (x+2) = -f (-x)) :=  -- ensuring symmetry about point (1,0)
by
  -- We'll prove this using the conditions given and properties derived from it
  sorry 

end NUMINAMATH_GPT_question_correct_statements_l838_83847


namespace NUMINAMATH_GPT_geo_seq_ratio_l838_83851

theorem geo_seq_ratio (S : ℕ → ℝ) (r : ℝ) (hS : ∀ n, S n = (1 - r^(n+1)) / (1 - r))
  (hS_ratio : S 10 / S 5 = 1 / 2) : S 15 / S 5 = 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_geo_seq_ratio_l838_83851


namespace NUMINAMATH_GPT_relative_error_comparison_l838_83898

theorem relative_error_comparison :
  let e₁ := 0.05
  let l₁ := 25.0
  let e₂ := 0.4
  let l₂ := 200.0
  let relative_error (e l : ℝ) : ℝ := (e / l) * 100
  (relative_error e₁ l₁ = relative_error e₂ l₂) :=
by
  sorry

end NUMINAMATH_GPT_relative_error_comparison_l838_83898


namespace NUMINAMATH_GPT_set_intersection_complement_l838_83832

theorem set_intersection_complement (U : Set ℝ) (A B : Set ℝ) 
  (hU : U = Set.univ) 
  (hA : ∀ x : ℝ, A x ↔ x^2 - x - 6 ≤ 0) 
  (hB : ∀ x : ℝ, B x ↔ Real.log x / Real.log (1/2) ≥ -1) :
  A ∩ (U \ B) = (Set.Icc (-2 : ℝ) 0 ∪ Set.Ioc 2 3) :=
by
  ext x
  -- Proof here would follow
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l838_83832


namespace NUMINAMATH_GPT_intersection_of_lines_l838_83820

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 5 * x - 2 * y = 8 ∧ 6 * x + 3 * y = 21 ∧ x = 22 / 9 ∧ y = 19 / 9 :=
by 
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l838_83820


namespace NUMINAMATH_GPT_mean_score_l838_83887

theorem mean_score (M SD : ℝ) (h1 : 58 = M - 2 * SD) (h2 : 98 = M + 3 * SD) : M = 74 :=
by
  sorry

end NUMINAMATH_GPT_mean_score_l838_83887


namespace NUMINAMATH_GPT_max_sum_of_ABC_l838_83876

/-- Theorem: The maximum value of A + B + C for distinct positive integers A, B, and C such that A * B * C = 2023 is 297. -/
theorem max_sum_of_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : A * B * C = 2023) :
  A + B + C ≤ 297 :=
sorry

end NUMINAMATH_GPT_max_sum_of_ABC_l838_83876


namespace NUMINAMATH_GPT_ian_saves_per_day_l838_83891

-- Let us define the given conditions
def total_saved : ℝ := 0.40 -- Ian saved a total of $0.40
def days : ℕ := 40 -- Ian saved for 40 days

-- Now, we need to prove that Ian saved 0.01 dollars/day
theorem ian_saves_per_day (h : total_saved = 0.40 ∧ days = 40) : total_saved / days = 0.01 :=
by
  sorry

end NUMINAMATH_GPT_ian_saves_per_day_l838_83891


namespace NUMINAMATH_GPT_probability_white_given_popped_is_7_over_12_l838_83849

noncomputable def probability_white_given_popped : ℚ :=
  let P_W := 0.4
  let P_Y := 0.4
  let P_R := 0.2
  let P_popped_given_W := 0.7
  let P_popped_given_Y := 0.5
  let P_popped_given_R := 0
  let P_popped := P_popped_given_W * P_W + P_popped_given_Y * P_Y + P_popped_given_R * P_R
  (P_popped_given_W * P_W) / P_popped

theorem probability_white_given_popped_is_7_over_12 : probability_white_given_popped = 7 / 12 := 
  by
    sorry

end NUMINAMATH_GPT_probability_white_given_popped_is_7_over_12_l838_83849


namespace NUMINAMATH_GPT_students_with_green_eyes_l838_83889

-- Define the variables and given conditions
def total_students : ℕ := 36
def students_with_red_hair (y : ℕ) : ℕ := 3 * y
def students_with_both : ℕ := 12
def students_with_neither : ℕ := 4

-- Define the proof statement
theorem students_with_green_eyes :
  ∃ y : ℕ, 
  (students_with_red_hair y + y - students_with_both + students_with_neither = total_students) ∧
  (students_with_red_hair y ≠ y) → y = 11 :=
by
  sorry

end NUMINAMATH_GPT_students_with_green_eyes_l838_83889


namespace NUMINAMATH_GPT_find_m_abc_inequality_l838_83883

-- Define properties and the theorem for the first problem
def f (x m : ℝ) := m - |x - 2|

theorem find_m (m : ℝ) : (∀ x, f (x + 2) m ≥ 0 ↔ x ∈ Set.Icc (-1 : ℝ) 1) → m = 1 := by
  intros h
  sorry

-- Define properties and the theorem for the second problem
theorem abc_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) → (a + 2 * b + 3 * c ≥ 9) := by
  intros h
  sorry

end NUMINAMATH_GPT_find_m_abc_inequality_l838_83883


namespace NUMINAMATH_GPT_total_books_in_library_l838_83882

theorem total_books_in_library :
  ∃ (total_books : ℕ),
  (∀ (books_per_floor : ℕ), books_per_floor - 2 = 20 → 
  total_books = (28 * 6 * books_per_floor)) ∧ total_books = 3696 :=
by
  sorry

end NUMINAMATH_GPT_total_books_in_library_l838_83882


namespace NUMINAMATH_GPT_remainder_2_pow_19_div_7_l838_83862

theorem remainder_2_pow_19_div_7 :
  2^19 % 7 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_2_pow_19_div_7_l838_83862


namespace NUMINAMATH_GPT_hexagon_perimeter_l838_83823

-- Definitions of the conditions
def side_length : ℕ := 5
def number_of_sides : ℕ := 6

-- The perimeter of the hexagon
def perimeter : ℕ := side_length * number_of_sides

-- Proof statement
theorem hexagon_perimeter : perimeter = 30 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_l838_83823


namespace NUMINAMATH_GPT_tax_rate_l838_83896

noncomputable def payroll_tax : Float := 300000
noncomputable def tax_paid : Float := 200
noncomputable def tax_threshold : Float := 200000

theorem tax_rate (tax_rate : Float) : 
  (payroll_tax - tax_threshold) * tax_rate = tax_paid → tax_rate = 0.002 := 
by
  sorry

end NUMINAMATH_GPT_tax_rate_l838_83896


namespace NUMINAMATH_GPT_train_length_proof_l838_83881

/-- Given a train's speed of 45 km/hr, time to cross a bridge of 30 seconds, and the bridge length of 225 meters, prove that the length of the train is 150 meters. -/
theorem train_length_proof (speed_km_hr : ℝ) (time_sec : ℝ) (bridge_length_m : ℝ) (train_length_m : ℝ)
    (h_speed : speed_km_hr = 45) (h_time : time_sec = 30) (h_bridge_length : bridge_length_m = 225) :
  train_length_m = 150 :=
by
  sorry

end NUMINAMATH_GPT_train_length_proof_l838_83881


namespace NUMINAMATH_GPT_actual_area_of_lawn_l838_83878

-- Definitions and conditions
variable (blueprint_area : ℝ)
variable (side_on_blueprint : ℝ)
variable (actual_side_length : ℝ)

-- Given conditions
def blueprint_conditions := 
  blueprint_area = 300 ∧ 
  side_on_blueprint = 5 ∧ 
  actual_side_length = 15

-- Prove the actual area of the lawn
theorem actual_area_of_lawn (blueprint_area : ℝ) (side_on_blueprint : ℝ) (actual_side_length : ℝ) (x : ℝ) :
  blueprint_conditions blueprint_area side_on_blueprint actual_side_length →
  (x = 27000000 ∧ x / 10000 = 2700) :=
by
  sorry

end NUMINAMATH_GPT_actual_area_of_lawn_l838_83878


namespace NUMINAMATH_GPT_sum_of_squares_l838_83852

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l838_83852


namespace NUMINAMATH_GPT_inequality_selection_l838_83854

theorem inequality_selection (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  1/a + 4/b ≥ 9/(a + b) :=
sorry

end NUMINAMATH_GPT_inequality_selection_l838_83854


namespace NUMINAMATH_GPT_cube_numbers_not_all_even_cube_numbers_not_all_divisible_by_3_l838_83884

-- Define the initial state of the cube vertices
def initial_cube : ℕ → ℕ
| 0 => 1  -- The number at vertex 0 is 1
| _ => 0  -- The numbers at other vertices are 0

-- Define the edge addition operation
def edge_add (v1 v2 : ℕ → ℕ) (edge : ℕ × ℕ) : ℕ → ℕ :=
  λ x => if x = edge.1 ∨ x = edge.2 then v1 x + 1 else v1 x

-- Condition: one can add one to the numbers at the ends of any edge
axiom edge_op : ∀ (v : ℕ → ℕ) (e : ℕ × ℕ), ℕ → ℕ

-- Defining the problem in Lean
theorem cube_numbers_not_all_even :
  ¬ (∃ (v : ℕ → ℕ), ∀ x, v x % 2 = 0) :=
by
  -- Proof not required
  sorry

theorem cube_numbers_not_all_divisible_by_3 :
  ¬ (∃ (v : ℕ → ℕ), ∀ x, v x % 3 = 0) :=
by
  -- Proof not required
  sorry

end NUMINAMATH_GPT_cube_numbers_not_all_even_cube_numbers_not_all_divisible_by_3_l838_83884


namespace NUMINAMATH_GPT_inequality_proof_l838_83829

variables (a b c d e f : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
variable (hcond : |sqrt (a * d) - sqrt (b * c)| ≤ 1)

theorem inequality_proof :
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l838_83829


namespace NUMINAMATH_GPT_area_of_triangle_ABC_eq_3_l838_83857

variable {n : ℕ}

def arithmetic_seq (a_1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => a_1 + n * d

def sum_arithmetic_seq (a_1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => (n + 1) * a_1 + (n * (n + 1) / 2) * d

def f (n : ℕ) : ℤ := sum_arithmetic_seq 4 6 n

def point_A (n : ℕ) : ℤ × ℤ := (n, f n)
def point_B (n : ℕ) : ℤ × ℤ := (n + 1, f (n + 1))
def point_C (n : ℕ) : ℤ × ℤ := (n + 2, f (n + 2))

def area_of_triangle (A B C : ℤ × ℤ) : ℤ :=
  (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)).natAbs / 2

theorem area_of_triangle_ABC_eq_3 : 
  ∀ (n : ℕ), area_of_triangle (point_A n) (point_B n) (point_C n) = 3 := 
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_eq_3_l838_83857


namespace NUMINAMATH_GPT_distance_between_stripes_l838_83897

/-
Problem statement:
Given:
1. The street has parallel curbs 30 feet apart.
2. The length of the curb between the stripes is 10 feet.
3. Each stripe is 60 feet long.

Prove:
The distance between the stripes is 5 feet.
-/

-- Definitions:
def distance_between_curbs : ℝ := 30
def length_between_stripes_on_curb : ℝ := 10
def length_of_each_stripe : ℝ := 60

-- Theorem statement:
theorem distance_between_stripes :
  ∃ d : ℝ, (length_between_stripes_on_curb * distance_between_curbs = length_of_each_stripe * d) ∧ d = 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_stripes_l838_83897


namespace NUMINAMATH_GPT_largest_prime_factor_13231_l838_83844

-- Define the conditions
def is_prime (n : ℕ) : Prop := ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

-- State the problem as a theorem in Lean 4
theorem largest_prime_factor_13231 (H1 : 13231 = 121 * 109) 
    (H2 : is_prime 109)
    (H3 : 121 = 11^2) :
    ∃ p, is_prime p ∧ p ∣ 13231 ∧ ∀ q, is_prime q ∧ q ∣ 13231 → q ≤ p :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_factor_13231_l838_83844


namespace NUMINAMATH_GPT_part_I_part_II_l838_83812

namespace ArithmeticGeometricSequences

-- Definitions of sequences and their properties
def a1 : ℕ := 1
def b1 : ℕ := 2
def b (n : ℕ) : ℕ := 2 * 3 ^ (n - 1) -- General term of the geometric sequence

-- Definitions from given conditions
def a (n : ℕ) : ℕ := 3 * n - 2 -- General term of the arithmetic sequence

-- Sum of the first n terms of the geometric sequence
def S (n : ℕ) : ℕ := if n = 0 then 0 else 3 ^ n - 1

-- Theorem statement
theorem part_I (n : ℕ) : 
  (a1 = 1) ∧ 
  (b1 = 2) ∧ 
  (∀ n > 0, b n > 0) ∧ 
  (∀ b2 : ℕ, 2 * (1 + b2 / 2) = 2 + b2) ∧ 
  (∀ b2 a2 : ℕ, (1 + b2 / 2)^2 = b2 * ((a 3) + 2)) →
  (a n = 3 * n - 2) ∧ 
  (b n = 2 * 3 ^ (n - 1)) :=
  sorry

theorem part_II (n : ℕ) (m : ℝ) :
  (a1 = 1) ∧ 
  (b1 = 2) ∧ 
  (∀ n > 0, b n > 0) ∧ 
  (∀ b2 : ℕ, 2 * (1 + b2 / 2) = 2 + b2) ∧ 
  (∀ b2 a2 : ℕ, (1 + b2 / 2)^2 = b2 * ((a 3) + 2)) → 
  (∀ n > 0, S n + a n > m) → 
  (m < 3) :=
  sorry

end ArithmeticGeometricSequences

end NUMINAMATH_GPT_part_I_part_II_l838_83812


namespace NUMINAMATH_GPT_small_box_dolls_l838_83859

theorem small_box_dolls (x : ℕ) : 
  (5 * 7 + 9 * x = 71) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_small_box_dolls_l838_83859


namespace NUMINAMATH_GPT_geometric_sequence_sum_l838_83814

theorem geometric_sequence_sum (S : ℕ → ℝ) (a₄_to_a₁₂_sum : ℝ):
  (S 3 = 2) → (S 6 = 6) → a₄_to_a₁₂_sum = (S 12 - S 3)  :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l838_83814


namespace NUMINAMATH_GPT_find_smaller_number_l838_83880

theorem find_smaller_number (x y : ℤ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l838_83880


namespace NUMINAMATH_GPT_polynomial_roots_a_ge_five_l838_83877

theorem polynomial_roots_a_ge_five (a b c : ℤ) (h_a_pos : a > 0)
    (h_distinct_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
        a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) : a ≥ 5 := sorry

end NUMINAMATH_GPT_polynomial_roots_a_ge_five_l838_83877


namespace NUMINAMATH_GPT_mary_income_percent_of_juan_l838_83801

variable (J : ℝ)
variable (T : ℝ)
variable (M : ℝ)

-- Conditions
def tim_income := T = 0.60 * J
def mary_income := M = 1.40 * T

-- Theorem to prove that Mary's income is 84 percent of Juan's income
theorem mary_income_percent_of_juan : tim_income J T → mary_income T M → M = 0.84 * J :=
by
  sorry

end NUMINAMATH_GPT_mary_income_percent_of_juan_l838_83801


namespace NUMINAMATH_GPT_total_new_bottles_l838_83833

theorem total_new_bottles (initial_bottles : ℕ) (recycle_ratio : ℕ) (bonus_ratio : ℕ) (final_bottles : ℕ) :
  initial_bottles = 625 →
  recycle_ratio = 5 →
  bonus_ratio = 20 →
  final_bottles = 163 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_total_new_bottles_l838_83833


namespace NUMINAMATH_GPT_dima_picks_more_berries_l838_83871

theorem dima_picks_more_berries (N : ℕ) (dima_fastness : ℕ) (sergei_fastness : ℕ) (dima_rate : ℕ) (sergei_rate : ℕ) :
  N = 450 → dima_fastness = 2 * sergei_fastness →
  dima_rate = 1 → sergei_rate = 2 →
  let dima_basket : ℕ := N / 2
  let sergei_basket : ℕ := (2 * N) / 3
  dima_basket > sergei_basket ∧ (dima_basket - sergei_basket) = 50 := 
by {
  sorry
}

end NUMINAMATH_GPT_dima_picks_more_berries_l838_83871


namespace NUMINAMATH_GPT_totalWheelsInStorageArea_l838_83886

def numberOfBicycles := 24
def numberOfTricycles := 14
def wheelsPerBicycle := 2
def wheelsPerTricycle := 3

theorem totalWheelsInStorageArea :
  numberOfBicycles * wheelsPerBicycle + numberOfTricycles * wheelsPerTricycle = 90 :=
by
  sorry

end NUMINAMATH_GPT_totalWheelsInStorageArea_l838_83886


namespace NUMINAMATH_GPT_find_c_for_radius_of_circle_l838_83879

theorem find_c_for_radius_of_circle :
  ∃ c : ℝ, (∀ x y : ℝ, x^2 + 8 * x + y^2 - 6 * y + c = 0 → (x + 4)^2 + (y - 3)^2 = 25 - c) ∧
  (∀ x y : ℝ, (x + 4)^2 + (y - 3)^2 = 25 → c = 0) :=
sorry

end NUMINAMATH_GPT_find_c_for_radius_of_circle_l838_83879


namespace NUMINAMATH_GPT_cliff_shiny_igneous_l838_83815

variables (I S : ℕ)

theorem cliff_shiny_igneous :
  I = S / 2 ∧ I + S = 270 → I / 3 = 30 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_cliff_shiny_igneous_l838_83815


namespace NUMINAMATH_GPT_complex_fraction_l838_83807

theorem complex_fraction (h : (1 : ℂ) - I = 1 - (I : ℂ)) :
  ((1 - I) * (1 - (2 * I))) / (1 + I) = -2 - I := 
by
  sorry

end NUMINAMATH_GPT_complex_fraction_l838_83807
