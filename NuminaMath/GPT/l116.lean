import Mathlib

namespace bert_bought_300_stamps_l116_116729

theorem bert_bought_300_stamps (x : ℝ) 
(H1 : x / 2 + x = 450) : x = 300 :=
by
  sorry

end bert_bought_300_stamps_l116_116729


namespace find_first_number_in_sequence_l116_116555

theorem find_first_number_in_sequence :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℚ),
    (a3 = a2 * a1) ∧ 
    (a4 = a3 * a2) ∧ 
    (a5 = a4 * a3) ∧ 
    (a6 = a5 * a4) ∧ 
    (a7 = a6 * a5) ∧ 
    (a8 = a7 * a6) ∧ 
    (a9 = a8 * a7) ∧ 
    (a10 = a9 * a8) ∧ 
    (a8 = 36) ∧ 
    (a9 = 324) ∧ 
    (a10 = 11664) ∧ 
    (a1 = 59049 / 65536) := 
sorry

end find_first_number_in_sequence_l116_116555


namespace first_term_of_geo_series_l116_116398

-- Define the conditions
def common_ratio : ℚ := 1 / 4
def sum_S : ℚ := 40

-- Define the question to be proven
theorem first_term_of_geo_series (a : ℚ) (h : sum_S = a / (1 - common_ratio)) : a = 30 := 
by
  sorry

end first_term_of_geo_series_l116_116398


namespace exterior_angle_of_parallel_lines_l116_116961

theorem exterior_angle_of_parallel_lines (A B C x y : ℝ) (hAx : A = 40) (hBx : B = 90) (hCx : C = 40)
  (h_parallel : true)
  (h_triangle : x = 180 - A - C)
  (h_exterior_angle : y = 180 - x) :
  y = 80 := 
by
  sorry

end exterior_angle_of_parallel_lines_l116_116961


namespace term_61_is_201_l116_116605

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (a5 : ℤ)

-- Define the general formula for the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ :=
  a5 + (n - 5) * d

-- Given variables and conditions:
axiom h1 : a5 = 33
axiom h2 : d = 3

theorem term_61_is_201 :
  arithmetic_sequence a5 d 61 = 201 :=
by
  -- proof here
  sorry

end term_61_is_201_l116_116605


namespace monthly_income_of_P_l116_116647

theorem monthly_income_of_P (P Q R : ℝ) 
    (h1 : (P + Q) / 2 = 2050) 
    (h2 : (Q + R) / 2 = 5250) 
    (h3 : (P + R) / 2 = 6200) : 
    P = 3000 :=
by
  sorry

end monthly_income_of_P_l116_116647


namespace probability_XOX_OXO_l116_116421

open Nat

/-- Setting up the math problem to be proved -/
def X : Finset ℕ := {1, 2, 3, 4}
def O : Finset ℕ := {5, 6, 7}

def totalArrangements : ℕ := choose 7 4

def favorableArrangements : ℕ := 1

theorem probability_XOX_OXO : (favorableArrangements : ℚ) / (totalArrangements : ℚ) = 1 / 35 := by
  have h_total : totalArrangements = 35 := by sorry
  have h_favorable : favorableArrangements = 1 := by sorry
  rw [h_total, h_favorable]
  norm_num

end probability_XOX_OXO_l116_116421


namespace find_a_l116_116039

theorem find_a (a : ℝ) (h1 : 0 < a)
  (c1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (c2 : ∀ x y : ℝ, x^2 + y^2 + 2 * a * y - 6 = 0)
  (h_chord : (2 * Real.sqrt 3) = 2 * Real.sqrt 3) :
  a = 1 := 
sorry

end find_a_l116_116039


namespace percentage_of_boys_currently_l116_116813

variables (B G : ℕ)

theorem percentage_of_boys_currently
  (h1 : B + G = 50)
  (h2 : B + 50 = 95) :
  (B * 100) / 50 = 90 :=
by
  sorry

end percentage_of_boys_currently_l116_116813


namespace pirate_treasure_l116_116513

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l116_116513


namespace square_side_length_square_area_l116_116188

theorem square_side_length 
  (d : ℝ := 4) : (s : ℝ) = 2 * Real.sqrt 2 :=
  sorry

theorem square_area 
  (s : ℝ := 2 * Real.sqrt 2) : (A : ℝ) = 8 :=
  sorry

end square_side_length_square_area_l116_116188


namespace ratio_tends_to_zero_as_n_tends_to_infinity_l116_116620

def smallest_prime_not_dividing (n : ℕ) : ℕ :=
  -- Function to find the smallest prime not dividing n
  sorry

theorem ratio_tends_to_zero_as_n_tends_to_infinity :
  ∀ ε > 0, ∃ N, ∀ n > N, (smallest_prime_not_dividing n : ℝ) / (n : ℝ) < ε := by
  sorry

end ratio_tends_to_zero_as_n_tends_to_infinity_l116_116620


namespace remainder_of_3_pow_800_mod_17_l116_116227

theorem remainder_of_3_pow_800_mod_17 : (3^800) % 17 = 1 := by
  sorry

end remainder_of_3_pow_800_mod_17_l116_116227


namespace find_reciprocal_square_sum_of_roots_l116_116649

theorem find_reciprocal_square_sum_of_roots :
  ∃ (a b c : ℝ), 
    (a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (a^3 - 6 * a^2 - a + 3 = 0) ∧ 
    (b^3 - 6 * b^2 - b + 3 = 0) ∧ 
    (c^3 - 6 * c^2 - c + 3 = 0) ∧ 
    (a + b + c = 6) ∧
    (a * b + b * c + c * a = -1) ∧
    (a * b * c = -3)) 
    → (1 / a^2 + 1 / b^2 + 1 / c^2 = 37 / 9) :=
sorry

end find_reciprocal_square_sum_of_roots_l116_116649


namespace range_zero_of_roots_l116_116909

theorem range_zero_of_roots (x y z w : ℝ) (h1 : x + y + z + w = 0) 
                            (h2 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 :=
  sorry

end range_zero_of_roots_l116_116909


namespace number_of_valid_pairs_l116_116026

theorem number_of_valid_pairs (a b : ℝ) :
  (∃ x y : ℤ, a * (x : ℝ) + b * (y : ℝ) = 2 ∧ (x : ℝ)^2 + (y : ℝ)^2 = 65) →
  ∃! pairs_count : ℕ, pairs_count = 72 :=
by
  sorry

end number_of_valid_pairs_l116_116026


namespace compare_abc_l116_116424

theorem compare_abc 
  (a : ℝ := 1 / 11) 
  (b : ℝ := Real.sqrt (1 / 10)) 
  (c : ℝ := Real.log (11 / 10)) : 
  b > c ∧ c > a := 
by
  sorry

end compare_abc_l116_116424


namespace square_perimeter_eq_16_l116_116470

theorem square_perimeter_eq_16 (s : ℕ) (h : s^2 = 4 * s) : 4 * s = 16 :=
by {
  sorry
}

end square_perimeter_eq_16_l116_116470


namespace compute_result_l116_116006

noncomputable def compute_expr : ℚ :=
  8 * (2 / 7)^4

theorem compute_result : compute_expr = 128 / 2401 := 
by 
  sorry

end compute_result_l116_116006


namespace negation_example_l116_116479

theorem negation_example : ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ ∃ x : ℝ, x > 1 ∧ x^2 ≤ 1 := by
  sorry

end negation_example_l116_116479


namespace largest_integer_not_sum_of_multiple_of_30_and_composite_l116_116682

theorem largest_integer_not_sum_of_multiple_of_30_and_composite :
  ∃ n : ℕ, ∀ a b : ℕ, 0 ≤ b ∧ b < 30 ∧ b.prime ∧ (∀ k < a, (b + 30 * k).prime)
    → (30 * a + b = n) ∧
      (∀ m : ℕ, ∀ c d : ℕ, 0 ≤ d ∧ d < 30 ∧ d.prime ∧ (∀ k < c, (d + 30 * k).prime) 
        → (30 * c + d ≤ n)) ∧
      n = 93 :=
by
  sorry

end largest_integer_not_sum_of_multiple_of_30_and_composite_l116_116682


namespace profit_calculation_more_profitable_method_l116_116886

def profit_end_of_month (x : ℝ) : ℝ :=
  0.3 * x - 900

def profit_beginning_of_month (x : ℝ) : ℝ :=
  0.26 * x

theorem profit_calculation (x : ℝ) (h₁ : profit_end_of_month x = 0.3 * x - 900)
  (h₂ : profit_beginning_of_month x = 0.26 * x) :
  profit_end_of_month x = 0.3 * x - 900 ∧ profit_beginning_of_month x = 0.26 * x :=
by 
  sorry

theorem more_profitable_method (x : ℝ) (hx : x = 20000)
  (h_beg : profit_beginning_of_month x = 0.26 * x)
  (h_end : profit_end_of_month x = 0.3 * x - 900) :
  profit_beginning_of_month x > profit_end_of_month x ∧ profit_beginning_of_month x = 5200 :=
by 
  sorry

end profit_calculation_more_profitable_method_l116_116886


namespace find_f_10_l116_116692

variable (f : ℝ → ℝ)
variable (y : ℝ)

-- Conditions
def condition1 : Prop := ∀ x, f x = 2 * x^2 + y
def condition2 : Prop := f 2 = 30

-- Theorem to prove
theorem find_f_10 (h1 : condition1 f y) (h2 : condition2 f) : f 10 = 222 := 
sorry

end find_f_10_l116_116692


namespace survived_more_than_died_l116_116938

-- Define the given conditions
def total_trees : ℕ := 13
def trees_died : ℕ := 6
def trees_survived : ℕ := total_trees - trees_died

-- The proof statement
theorem survived_more_than_died :
  trees_survived - trees_died = 1 := 
by
  -- This is where the proof would go
  sorry

end survived_more_than_died_l116_116938


namespace find_t_l116_116125

theorem find_t (t : ℝ) (h : (1 / (t+3) + 3 * t / (t+3) - 4 / (t+3)) = 5) : t = -9 :=
by
  sorry

end find_t_l116_116125


namespace incorrect_option_C_l116_116893

-- Definitions of increasing and decreasing functions
def increasing (f : ℝ → ℝ) := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≤ f x₂
def decreasing (f : ℝ → ℝ) := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≥ f x₂

-- The incorrectness of option C
theorem incorrect_option_C (f g : ℝ → ℝ) 
  (h₁ : increasing f) 
  (h₂ : decreasing g) : ¬ increasing (fun x => f x + g x) := 
sorry

end incorrect_option_C_l116_116893


namespace min_value_PA_PF_l116_116304

noncomputable def minimum_value_of_PA_and_PF_minimum 
  (x y : ℝ)
  (A : ℝ × ℝ)
  (F : ℝ × ℝ) : ℝ :=
  if ((A = (-1, 8)) ∧ (F = (0, 1)) ∧ (x^2 = 4 * y)) then 9 else 0

theorem min_value_PA_PF 
  (A : ℝ × ℝ := (-1, 8))
  (F : ℝ × ℝ := (0, 1))
  (P : ℝ × ℝ)
  (hP : P.1^2 = 4 * P.2) :
  minimum_value_of_PA_and_PF_minimum P.1 P.2 A F = 9 :=
by
  sorry

end min_value_PA_PF_l116_116304


namespace pumpkin_count_sunshine_orchard_l116_116175

def y (x : ℕ) : ℕ := 3 * x^2 + 12

theorem pumpkin_count_sunshine_orchard :
  y 14 = 600 :=
by
  sorry

end pumpkin_count_sunshine_orchard_l116_116175


namespace interval_solution_l116_116578

-- Let the polynomial be defined
def polynomial (x : ℝ) : ℝ := x^3 - 12 * x^2 + 30 * x

-- Prove the inequality for the specified intervals
theorem interval_solution :
  { x : ℝ | polynomial x > 0 } = { x : ℝ | (0 < x ∧ x < 5) ∨ x > 6 } :=
by
  sorry

end interval_solution_l116_116578


namespace probability_of_shortening_exactly_one_digit_l116_116877

theorem probability_of_shortening_exactly_one_digit :
  let n := 2014 in
  let p := 0.1 in
  let q := 0.9 in
  (n.choose 1) * p ^ 1 * q ^ (n - 1) = 201.4 * 1.564 * 10^(-90) :=
by sorry

end probability_of_shortening_exactly_one_digit_l116_116877


namespace remainder_when_sum_divided_by_5_l116_116090

theorem remainder_when_sum_divided_by_5 (f y : ℤ) (k m : ℤ) 
  (hf : f = 5 * k + 3) (hy : y = 5 * m + 4) : 
  (f + y) % 5 = 2 := 
by {
  sorry
}

end remainder_when_sum_divided_by_5_l116_116090


namespace sum_of_possible_values_l116_116822

theorem sum_of_possible_values 
  (x y : ℝ) 
  (h : x * y - x / y^2 - y / x^2 = 3) :
  (x = 0 ∨ y = 0 → False) → 
  ((x - 1) * (y - 1) = 1 ∨ (x - 1) * (y - 1) = 4) → 
  ((x - 1) * (y - 1) = 1 → (x - 1) * (y - 1) = 1) → 
  ((x - 1) * (y - 1) = 4 → (x - 1) * (y - 1) = 4) → 
  (1 + 4 = 5) := 
by 
  sorry

end sum_of_possible_values_l116_116822


namespace ball_travel_distance_five_hits_l116_116394

def total_distance_traveled (h₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  let descents := List.range (n + 1) |>.map (λ i => h₀ * r ^ i)
  let ascents := List.range n |>.map (λ i => h₀ * r ^ (i + 1))
  (descents.sum + ascents.sum)

theorem ball_travel_distance_five_hits :
  total_distance_traveled 120 (3 / 4) 5 = 612.1875 :=
by
  sorry

end ball_travel_distance_five_hits_l116_116394


namespace cistern_capacity_l116_116237

theorem cistern_capacity (C : ℝ) (h1 : C / 20 > 0) (h2 : C / 24 > 0) (h3 : 4 - C / 20 = C / 24) : C = 480 / 11 :=
by sorry

end cistern_capacity_l116_116237


namespace f_at_pi_over_2_eq_1_l116_116979

noncomputable def f (ω : ℝ) (b x : ℝ) : ℝ := sin (ω * x + π / 4) + b

theorem f_at_pi_over_2_eq_1 (ω : ℝ) (b : ℝ) (T : ℝ) (hω_pos : ω > 0)
  (hT_period : T = 2 * π / ω) (hT_range : 2 * π / 3 < T ∧ T < π)
  (h_symm : f ω b (3 * π / 2) = 2) :
  f ω b (π / 2) = 1 :=  
sorry

end f_at_pi_over_2_eq_1_l116_116979


namespace mode_of_shoe_sizes_is_25_5_l116_116701

def sales_data := [(24, 2), (24.5, 5), (25, 3), (25.5, 6), (26, 4)]

theorem mode_of_shoe_sizes_is_25_5 
  (h : ∀ x ∈ sales_data, 2 ≤ x.1 ∧ 
        (∀ y ∈ sales_data, x.2 ≤ y.2 → x.1 = 25.5 ∨ x.2 < 6)) : 
  (∃ s, s ∈ sales_data ∧ s.1 = 25.5 ∧ s.2 = 6) :=
sorry

end mode_of_shoe_sizes_is_25_5_l116_116701


namespace prove_k_in_terms_of_x_l116_116646

variables {A B k x : ℝ}

-- given conditions
def positive_numbers (A B : ℝ) := A > 0 ∧ B > 0
def ratio_condition (A B k : ℝ) := A = k * B
def percentage_condition (A B x : ℝ) := A = B + (x / 100) * B

-- proof statement
theorem prove_k_in_terms_of_x (A B k x : ℝ) (h1 : positive_numbers A B) (h2 : ratio_condition A B k) (h3 : percentage_condition A B x) (h4 : k > 1) :
  k = 1 + x / 100 :=
sorry

end prove_k_in_terms_of_x_l116_116646


namespace quarters_difference_nickels_eq_l116_116400

variable (q : ℕ)

def charles_quarters := 7 * q + 2
def richard_quarters := 3 * q + 7
def quarters_difference := charles_quarters q - richard_quarters q
def money_difference_in_nickels := 5 * quarters_difference q

theorem quarters_difference_nickels_eq :
  money_difference_in_nickels q = 20 * (q - 5/4) :=
by
  sorry

end quarters_difference_nickels_eq_l116_116400


namespace shorten_by_one_expected_length_l116_116884

-- Definition of the random sequence and binomial distribution properties
def digit_sequence {n : ℕ} (p : ℝ) (len : ℕ) : ℝ := 
  let q := 1 - p in
  (len.choose 1) * (p^1 * q^(len - 1))

-- Part (a) statement
theorem shorten_by_one 
  (len : ℕ) 
  (p : ℝ := 0.1) 
  (q := 1 - p) 
  (x := len - 1) : 
  digit_sequence p x = 1.564e-90 :=
by sorry

-- Part (b) statement
theorem expected_length 
  (initial_len : ℕ) 
  (p : ℝ := 0.1) (removals := initial_len - 1) 
  (expected_removed : ℝ := removals * p) : 
  (initial_len : ℝ) - expected_removed = 1813.6 :=
by sorry

end shorten_by_one_expected_length_l116_116884


namespace coprime_repeating_decimal_sum_l116_116790

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l116_116790


namespace haley_seeds_in_big_garden_l116_116443

def seeds_in_big_garden (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - (small_gardens * seeds_per_small_garden)

theorem haley_seeds_in_big_garden :
  let total_seeds := 56
  let small_gardens := 7
  let seeds_per_small_garden := 3
  seeds_in_big_garden total_seeds small_gardens seeds_per_small_garden = 35 :=
by
  sorry

end haley_seeds_in_big_garden_l116_116443


namespace pirates_treasure_l116_116484

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l116_116484


namespace union_A_B_l116_116435

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def C : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_A_B : A ∪ B = C := 
by sorry

end union_A_B_l116_116435


namespace David_total_swim_time_l116_116015

theorem David_total_swim_time :
  let t_freestyle := 48
  let t_backstroke := t_freestyle + 4
  let t_butterfly := t_backstroke + 3
  let t_breaststroke := t_butterfly + 2
  t_freestyle + t_backstroke + t_butterfly + t_breaststroke = 212 :=
by
  sorry

end David_total_swim_time_l116_116015


namespace least_number_division_remainder_4_l116_116220

theorem least_number_division_remainder_4 : 
  ∃ n : Nat, (n % 6 = 4) ∧ (n % 130 = 4) ∧ (n % 9 = 4) ∧ (n % 18 = 4) ∧ n = 2344 :=
by
  sorry

end least_number_division_remainder_4_l116_116220


namespace max_value_of_y_l116_116417

noncomputable def max_value_of_function : ℝ := 1 + Real.sqrt 2

theorem max_value_of_y : ∀ x : ℝ, (2 * Real.sin x * (Real.sin x + Real.cos x)) ≤ max_value_of_function :=
by
  -- Proof goes here
  sorry

example : ∃ x : ℝ, (2 * Real.sin x * (Real.sin x + Real.cos x)) = max_value_of_function :=
by
  -- Proof goes here
  sorry

end max_value_of_y_l116_116417


namespace fraction_less_than_thirty_percent_l116_116544

theorem fraction_less_than_thirty_percent (x : ℚ) (hx : x * 180 = 36) (hx_lt : x < 0.3) : x = 1 / 5 := 
by
  sorry

end fraction_less_than_thirty_percent_l116_116544


namespace primes_up_to_floor_implies_all_primes_l116_116332

/-- Define the function f. -/
def f (x p : ℕ) : ℕ := x^2 + x + p

/-- Define the initial prime condition. -/
def primes_up_to_floor_sqrt_p_over_3 (p : ℕ) : Prop :=
  ∀ x, x ≤ Nat.floor (Nat.sqrt (p / 3)) → Nat.Prime (f x p)

/-- Define the property we want to prove. -/
def all_primes_up_to_p_minus_2 (p : ℕ) : Prop :=
  ∀ x, x ≤ p - 2 → Nat.Prime (f x p)

/-- The main theorem statement. -/
theorem primes_up_to_floor_implies_all_primes
  (p : ℕ) (h : primes_up_to_floor_sqrt_p_over_3 p) : all_primes_up_to_p_minus_2 p :=
sorry

end primes_up_to_floor_implies_all_primes_l116_116332


namespace train_average_speed_l116_116717

theorem train_average_speed (x : ℝ) (h1 : x > 0) :
  let d1 := x
  let d2 := 2 * x
  let s1 := 50
  let s2 := 20
  let t1 := d1 / s1
  let t2 := d2 / s2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  average_speed = 25 := 
by
  sorry

end train_average_speed_l116_116717


namespace gambler_final_amount_l116_116239

theorem gambler_final_amount :
  let initial_money := 100
  let win_multiplier := (3/2 : ℚ)
  let loss_multiplier := (1/2 : ℚ)
  let final_multiplier := (win_multiplier * loss_multiplier)^4
  let final_amount := initial_money * final_multiplier
  final_amount = (8100 / 256) :=
by
  sorry

end gambler_final_amount_l116_116239


namespace area_of_right_triangle_l116_116161

theorem area_of_right_triangle
  (BC AC : ℝ)
  (h1 : BC * AC = 16) : 
  0.5 * BC * AC = 8 := by 
  sorry

end area_of_right_triangle_l116_116161


namespace expected_girls_left_of_boys_l116_116366

theorem expected_girls_left_of_boys : 
  (∑ i in (finset.range 7), ((i+1) : ℝ) / 17) = 7 / 11 :=
sorry

end expected_girls_left_of_boys_l116_116366


namespace unique_a_for_three_distinct_real_solutions_l116_116023

theorem unique_a_for_three_distinct_real_solutions (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 - 2 * x + 1 - 3 * |x|) ∧
  ((∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0) ∧
  (∀ x4 : ℝ, f x4 = 0 → (x4 = x1 ∨ x4 = x2 ∨ x4 = x3) )) ) ↔
  a = 1 / 4 :=
sorry

end unique_a_for_three_distinct_real_solutions_l116_116023


namespace find_number_l116_116594

theorem find_number (p q N : ℝ) (h1 : N / p = 8) (h2 : N / q = 18) (h3 : p - q = 0.20833333333333334) : N = 3 :=
sorry

end find_number_l116_116594


namespace horner_rule_polynomial_polynomial_value_at_23_l116_116360

def polynomial (x : ℤ) : ℤ := 7 * x ^ 3 + 3 * x ^ 2 - 5 * x + 11

def horner_polynomial (x : ℤ) : ℤ := x * ((7 * x + 3) * x - 5) + 11

theorem horner_rule_polynomial (x : ℤ) : polynomial x = horner_polynomial x :=
by 
  -- The proof steps would go here,
  -- demonstrating that polynomial x = horner_polynomial x.
  sorry

-- Instantiation of the theorem for a specific value of x
theorem polynomial_value_at_23 : polynomial 23 = horner_polynomial 23 :=
by 
  -- Using the previously established theorem
  apply horner_rule_polynomial

end horner_rule_polynomial_polynomial_value_at_23_l116_116360


namespace side_length_of_cube_l116_116205

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end side_length_of_cube_l116_116205


namespace gcd_372_684_is_12_l116_116651

theorem gcd_372_684_is_12 :
  Nat.gcd 372 684 = 12 :=
sorry

end gcd_372_684_is_12_l116_116651


namespace treasure_coins_l116_116505

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l116_116505


namespace fever_above_threshold_l116_116944

-- Definitions as per conditions
def normal_temp : ℤ := 95
def temp_increase : ℤ := 10
def fever_threshold : ℤ := 100

-- Calculated new temperature
def new_temp := normal_temp + temp_increase

-- The proof statement, asserting the correct answer
theorem fever_above_threshold : new_temp - fever_threshold = 5 := 
by 
  sorry

end fever_above_threshold_l116_116944


namespace equation1_solution_equation2_solution_l116_116645

variable (x : ℝ)

theorem equation1_solution :
  ((2 * x - 5) / 6 - (3 * x + 1) / 2 = 1) → (x = -2) :=
by
  sorry

theorem equation2_solution :
  (3 * x - 7 * (x - 1) = 3 - 2 * (x + 3)) → (x = 5) :=
by
  sorry

end equation1_solution_equation2_solution_l116_116645


namespace pirate_treasure_l116_116509

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l116_116509


namespace number_of_pines_possible_l116_116210

-- Definitions based on conditions in the problem
def total_trees : ℕ := 101
def at_least_one_between_poplars (poplars : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (poplars[i] - poplars[j]) > 1
def at_least_two_between_birches (birches : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (birches[i] - birches[j]) > 2
def at_least_three_between_pines (pines : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (pines[i] - pines[j]) > 3

-- Proving the number of pines planted is either 25 or 26
theorem number_of_pines_possible (poplars birches pines : List ℕ)
  (h1 : length (poplars ++ birches ++ pines) = total_trees)
  (h2 : at_least_one_between_poplars poplars)
  (h3 : at_least_two_between_birches birches)
  (h4 : at_least_three_between_pines pines) :
  length pines = 25 ∨ length pines = 26 :=
sorry

end number_of_pines_possible_l116_116210


namespace find_n_l116_116802

theorem find_n (x n : ℝ) (h₁ : x = 1) (h₂ : 5 / (n + 1 / x) = 1) : n = 4 :=
sorry

end find_n_l116_116802


namespace min_value_of_a_l116_116818

theorem min_value_of_a (a : ℝ) (h : a > 0) (h₁ : ∀ x : ℝ, |x - a| + |1 - x| ≥ 1) : a ≥ 2 := 
sorry

end min_value_of_a_l116_116818


namespace measles_cases_1993_l116_116155

theorem measles_cases_1993 :
  ∀ (cases_1970 cases_1986 cases_2000 : ℕ)
    (rate1 rate2 : ℕ),
  cases_1970 = 600000 →
  cases_1986 = 30000 →
  cases_2000 = 600 →
  rate1 = 35625 →
  rate2 = 2100 →
  cases_1986 - 7 * rate2 = 15300 :=
by {
  sorry
}

end measles_cases_1993_l116_116155


namespace remainder_of_polynomial_division_l116_116122

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 7 * x^4 - 16 * x^3 + 3 * x^2 - 5 * x - 20

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := 2 * x - 4

-- The remainder theorem sets x to 2 and evaluates P(x)
theorem remainder_of_polynomial_division : P 2 = -34 :=
by
  -- We will substitute x=2 directly into P(x)
  sorry

end remainder_of_polynomial_division_l116_116122


namespace chris_money_left_over_l116_116257

-- Define the constants based on the conditions given in the problem.
def video_game_cost : ℕ := 60
def candy_cost : ℕ := 5
def earnings_per_hour : ℕ := 8
def hours_worked : ℕ := 9

-- Define the intermediary results based on the problem's conditions.
def total_cost : ℕ := video_game_cost + candy_cost
def total_earnings : ℕ := earnings_per_hour * hours_worked

-- Define the final result to be proven.
def total_leftover : ℕ := total_earnings - total_cost

-- State the proof problem as a Lean theorem.
theorem chris_money_left_over : total_leftover = 7 := by
  sorry

end chris_money_left_over_l116_116257


namespace female_democrats_count_l116_116661

variable (F M : ℕ)
def total_participants : Prop := F + M = 720
def female_democrats (D_F : ℕ) : Prop := D_F = 1 / 2 * F
def male_democrats (D_M : ℕ) : Prop := D_M = 1 / 4 * M
def total_democrats (D_F D_M : ℕ) : Prop := D_F + D_M = 1 / 3 * 720

theorem female_democrats_count
  (F M D_F D_M : ℕ)
  (h1 : total_participants F M)
  (h2 : female_democrats F D_F)
  (h3 : male_democrats M D_M)
  (h4 : total_democrats D_F D_M) :
  D_F = 120 :=
sorry

end female_democrats_count_l116_116661


namespace quadratic_roots_equal_integral_l116_116012

theorem quadratic_roots_equal_integral (c : ℝ) (h : (6^2 - 4 * 3 * c) = 0) : 
  ∃ x : ℝ, (3 * x^2 - 6 * x + c = 0) ∧ (x = 1) := 
by sorry

end quadratic_roots_equal_integral_l116_116012


namespace sum_of_first_53_odd_numbers_l116_116693

theorem sum_of_first_53_odd_numbers :
  let first_term := 1
  let last_term := first_term + (53 - 1) * 2
  let sum := 53 / 2 * (first_term + last_term)
  sum = 2809 :=
by
  let first_term := 1
  let last_term := first_term + (53 - 1) * 2
  have last_term_val : last_term = 105 := by
    sorry
  let sum := 53 / 2 * (first_term + last_term)
  have sum_val : sum = 2809 := by
    sorry
  exact sum_val

end sum_of_first_53_odd_numbers_l116_116693


namespace intersection_of_A_and_B_l116_116174

open Set

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x^2 - x ≤ 0}
  let B := ({0, 1, 2} : Set ℝ)
  A ∩ B = ({0, 1} : Set ℝ) :=
by
  sorry

end intersection_of_A_and_B_l116_116174


namespace last_trip_l116_116638

def initial_order : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

def boatCapacity : Nat := 4  -- Including Snow White

def adjacentPairsQuarrel (adjPairs : List (String × String)) : Prop :=
  ∀ (d1 d2 : String), (d1, d2) ∈ adjPairs → (d2, d1) ∈ adjPairs → False

def canRow (person : String) : Prop := person = "Snow White"

noncomputable def final_trip (remainingDwarfs : List String) (allTrips : List (List String)) : List String := ["Grumpy", "Bashful", "Doc"]

theorem last_trip (adjPairs : List (String × String))
  (h_adj : adjacentPairsQuarrel adjPairs)
  (h_row : canRow "Snow White")
  (dwarfs_order : List String = initial_order)
  (without_quarrels : ∀ trip : List String, trip ∈ allTrips → 
    ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → (d1, d2) ∈ adjPairs → 
    ("Snow White" ∈ trip) → True) :
  final_trip ["Grumpy", "Bashful", "Doc"] allTrips = ["Grumpy", "Bashful", "Doc"] :=
sorry

end last_trip_l116_116638


namespace boys_love_marbles_l116_116156

def total_marbles : ℕ := 26
def marbles_per_boy : ℕ := 2
def num_boys_love_marbles : ℕ := total_marbles / marbles_per_boy

theorem boys_love_marbles : num_boys_love_marbles = 13 := by
  rfl

end boys_love_marbles_l116_116156


namespace numerator_of_fraction_l116_116951

noncomputable def repeating_fraction : ℚ := 175 / 333

theorem numerator_of_fraction
  (h1 : repeating_fraction = 525 / 999) 
  (h2 : (81 : ℕ) % 3 ≠ 0 → 5) : 
  repeating_fraction.num = 175 := 
sorry

end numerator_of_fraction_l116_116951


namespace find_x_from_conditions_l116_116821

theorem find_x_from_conditions (a b x y s : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) :
  s = (4 * a)^(4 * b) ∧ s = a^b * y^b ∧ y = 4 * x → x = 64 * a^3 :=
by
  sorry

end find_x_from_conditions_l116_116821


namespace abs_c_eq_181_l116_116185

theorem abs_c_eq_181
  (a b c : ℤ)
  (h_gcd : Int.gcd a (Int.gcd b c) = 1)
  (h_eq : a * (Complex.mk 3 2)^4 + b * (Complex.mk 3 2)^3 + c * (Complex.mk 3 2)^2 + b * (Complex.mk 3 2) + a = 0) :
  |c| = 181 :=
sorry

end abs_c_eq_181_l116_116185


namespace runner_loop_time_l116_116358

-- Define times for the meetings using the provided conditions
def meeting_time_ab : ℕ := 15  -- time from A to B
def meeting_time_bc : ℕ := 25  -- time from B to C

-- Noncomputable definition since the exact number of runners is not determined computationally.
noncomputable def time_for_one_loop : ℕ :=
  let total_time := 2 * meeting_time_ab + 2 * meeting_time_bc in
  total_time

-- The theorem states the problem to be proven
theorem runner_loop_time (a b : ℕ) (h_a : a = 15) (h_b : b = 25) : 
  let t_total := 2 * a + 2 * b in
  t_total = 80 :=
  by
    sorry

end runner_loop_time_l116_116358


namespace tan_of_neg_23_over_3_pi_l116_116079

theorem tan_of_neg_23_over_3_pi : (Real.tan (- 23 / 3 * Real.pi) = Real.sqrt 3) :=
by
  sorry

end tan_of_neg_23_over_3_pi_l116_116079


namespace find_t_l116_116652

theorem find_t (t : ℝ) : 
  (∃ (m b : ℝ), (∀ x y, (y = m * x + b) → ((x = 1 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ (x = 5 ∧ y = 19))) ∧ (28 = 28 * m + b) ∧ (t = 28 * m + b)) → 
  t = 88 :=
by
  sorry

end find_t_l116_116652


namespace factor_expression_l116_116742

theorem factor_expression (a : ℝ) : 
  49 * a ^ 3 + 245 * a ^ 2 + 588 * a = 49 * a * (a ^ 2 + 5 * a + 12) :=
by
  sorry

end factor_expression_l116_116742


namespace plane_equidistant_from_B_and_C_l116_116753

-- Define points B and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def B : Point3D := { x := 4, y := 1, z := 0 }
def C : Point3D := { x := 2, y := 0, z := 3 }

-- Define the predicate for a plane equation
def plane_eq (a b c d : ℝ) (P : Point3D) : Prop :=
  a * P.x + b * P.y + c * P.z + d = 0

-- The problem statement
theorem plane_equidistant_from_B_and_C :
  ∃ D : ℝ, plane_eq (-2) (-1) 3 D { x := B.x, y := B.y, z := B.z } ∧
            plane_eq (-2) (-1) 3 D { x := C.x, y := C.y, z := C.z } :=
sorry

end plane_equidistant_from_B_and_C_l116_116753


namespace treasure_coins_l116_116504

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l116_116504


namespace pirates_treasure_l116_116488

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l116_116488


namespace circle_diameter_l116_116260

open Real

theorem circle_diameter (r_D : ℝ) (r_C : ℝ) (h_D : r_D = 10) (h_ratio: (π * (r_D ^ 2 - r_C ^ 2)) / (π * r_C ^ 2) = 4) : 2 * r_C = 4 * sqrt 5 :=
by sorry

end circle_diameter_l116_116260


namespace johns_remaining_money_l116_116056

theorem johns_remaining_money (H1 : ∃ (n : ℕ), n = 5376) (H2 : 5376 = 5 * 8^3 + 3 * 8^2 + 7 * 8^1 + 6) :
  (2814 - 1350 = 1464) :=
by {
  sorry
}

end johns_remaining_money_l116_116056


namespace bridget_apples_l116_116395

variable (x : ℕ)

-- Conditions as definitions
def apples_after_splitting : ℕ := x / 2
def apples_after_giving_to_cassie : ℕ := apples_after_splitting x - 5
def apples_after_finding_hidden : ℕ := apples_after_giving_to_cassie x + 2
def final_apples : ℕ := apples_after_finding_hidden x
def bridget_keeps : ℕ := 6

-- Proof statement
theorem bridget_apples : x / 2 - 5 + 2 = bridget_keeps → x = 18 := by
  intros h
  sorry

end bridget_apples_l116_116395


namespace range_of_a_l116_116153

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, abs (2 * x - 1) + abs (x + 2) ≥ a^2 + (1 / 2) * a + 2) →
  -1 ≤ a ∧ a ≤ (1 / 2) := by
sorry

end range_of_a_l116_116153


namespace treasures_coins_count_l116_116496

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l116_116496


namespace car_cost_l116_116084

-- Define the weekly allowance in the first year
def first_year_allowance_weekly : ℕ := 50

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Calculate the total first year savings
def first_year_savings : ℕ := first_year_allowance_weekly * weeks_in_year

-- Define the hourly wage and weekly hours worked in the second year
def hourly_wage : ℕ := 9
def weekly_hours_worked : ℕ := 30

-- Calculate the total second year earnings
def second_year_earnings : ℕ := hourly_wage * weekly_hours_worked * weeks_in_year

-- Define the weekly spending in the second year
def weekly_spending : ℕ := 35

-- Calculate the total second year spending
def second_year_spending : ℕ := weekly_spending * weeks_in_year

-- Calculate the total second year savings
def second_year_savings : ℕ := second_year_earnings - second_year_spending

-- Calculate the total savings after two years
def total_savings : ℕ := first_year_savings + second_year_savings

-- Define the additional amount needed
def additional_amount_needed : ℕ := 2000

-- Calculate the total cost of the car
def total_cost_of_car : ℕ := total_savings + additional_amount_needed

-- Theorem statement
theorem car_cost : total_cost_of_car = 16820 := by
  -- The proof is omitted; it is enough to state the theorem
  sorry

end car_cost_l116_116084


namespace pirates_treasure_l116_116486

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l116_116486


namespace minimum_sum_of_dimensions_l116_116344

-- Define the problem as a Lean 4 statement
theorem minimum_sum_of_dimensions (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 2184) : 
  x + y + z = 36 := 
sorry

end minimum_sum_of_dimensions_l116_116344


namespace Tony_fever_l116_116947

theorem Tony_fever :
  ∀ (normal_temp sickness_increase fever_threshold : ℕ),
    normal_temp = 95 →
    sickness_increase = 10 →
    fever_threshold = 100 →
    (normal_temp + sickness_increase) - fever_threshold = 5 :=
by
  intros normal_temp sickness_increase fever_threshold h1 h2 h3
  sorry

end Tony_fever_l116_116947


namespace necklaces_made_l116_116021

theorem necklaces_made (total_beads : ℕ) (beads_per_necklace : ℕ) (h1 : total_beads = 18) (h2 : beads_per_necklace = 3) : total_beads / beads_per_necklace = 6 := 
by {
  sorry
}

end necklaces_made_l116_116021


namespace Nigel_initial_amount_l116_116461

-- Defining the initial amount Olivia has
def Olivia_initial : ℕ := 112

-- Defining the amount left after buying the tickets
def amount_left : ℕ := 83

-- Defining the cost per ticket and the number of tickets bought
def cost_per_ticket : ℕ := 28
def number_of_tickets : ℕ := 6

-- Calculating the total cost of the tickets
def total_cost : ℕ := cost_per_ticket * number_of_tickets

-- Calculating the total amount Olivia spent
def Olivia_spent : ℕ := Olivia_initial - amount_left

-- Defining the total amount they spent
def total_spent : ℕ := total_cost

-- Main theorem to prove that Nigel initially had $139
theorem Nigel_initial_amount : ∃ (n : ℕ), (n + Olivia_initial - Olivia_spent = total_spent) → n = 139 :=
by {
  sorry
}

end Nigel_initial_amount_l116_116461


namespace lily_of_the_valley_bushes_needed_l116_116842

theorem lily_of_the_valley_bushes_needed 
  (r l : ℕ) (h_radius : r = 20) (h_length : l = 400) : 
  l / (2 * r) = 10 := 
by 
  sorry

end lily_of_the_valley_bushes_needed_l116_116842


namespace minimum_ab_ge_four_l116_116764

variable (a b : ℝ)
variables (ha : 0 < a) (hb : 0 < b)
variable (h : 1 / a + 4 / b = Real.sqrt (a * b))

theorem minimum_ab_ge_four : a * b ≥ 4 := by
  sorry

end minimum_ab_ge_four_l116_116764


namespace value_of_f_x_plus_5_l116_116619

open Function

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem value_of_f_x_plus_5 (x : ℝ) : f (x + 5) = 3 * x + 16 :=
by
  sorry

end value_of_f_x_plus_5_l116_116619


namespace ending_number_of_range_l116_116481

theorem ending_number_of_range (n : ℕ) (h : ∃ k, 1 ≤ k ∧ k ≤ 5 ∧ n = 29 + 11 * k) : n = 77 := by
  sorry

end ending_number_of_range_l116_116481


namespace hyperbola_property_l116_116766

def hyperbola := {x : ℝ // ∃ y : ℝ, x^2 - y^2 / 8 = 1}

def is_on_left_branch (M : hyperbola) : Prop :=
  M.1 < 0

def focus1 : ℝ := -3
def focus2 : ℝ := 3

def distance (a b : ℝ) : ℝ := abs (a - b)

theorem hyperbola_property (M : hyperbola) (hM : is_on_left_branch M) :
  distance M.1 focus1 + distance focus1 focus2 - distance M.1 focus2 = 4 :=
  sorry

end hyperbola_property_l116_116766


namespace sum_of_digits_joeys_age_l116_116972

-- Given conditions
variables (C : ℕ) (J : ℕ := C + 2) (Z : ℕ := 1)

-- Define the condition that the sum of Joey's and Chloe's ages will be an integral multiple of Zoe's age.
def sum_is_multiple_of_zoe (n : ℕ) : Prop :=
  ∃ k : ℕ, (J + C) = k * Z

-- Define the problem of finding the sum of digits the first time Joey's age alone is a multiple of Zoe's age.
def sum_of_digits_first_multiple (J Z : ℕ) : ℕ :=
  (J / 10) + (J % 10)

-- The theorem we need to prove
theorem sum_of_digits_joeys_age : (sum_of_digits_first_multiple J Z = 1) :=
sorry

end sum_of_digits_joeys_age_l116_116972


namespace functional_equation_to_odd_function_l116_116984

variables (f : ℝ → ℝ)

theorem functional_equation_to_odd_function (h : ∀ x y : ℝ, f (x + y) = f x + f y) :
  f 0 = 0 ∧ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end functional_equation_to_odd_function_l116_116984


namespace possible_slopes_l116_116705

theorem possible_slopes (m : ℝ) :
    (∃ x y : ℝ, (y = m * x - 3) ∧ (4 * x^2 + 25 * y^2 = 100)) ↔ 
    m ∈ (Set.Ioo (-∞) (-Real.sqrt (2 / 110)) ∪ Set.Ioo (Real.sqrt (2 / 110)) ∞) := 
  sorry

end possible_slopes_l116_116705


namespace solution_set_correct_l116_116031

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then 2^(-x) - 4 else 2^(x) - 4

theorem solution_set_correct : 
  (∀ x, f x = f |x|) → 
  (∀ x, f x = 2^(-x) - 4 ∨ f x = 2^(x) - 4) → 
  { x | f (x - 2) > 0 } = { x | x < 0 ∨ x > 4 } :=
by
  intro h1 h2
  sorry

end solution_set_correct_l116_116031


namespace cow_drink_pond_l116_116965

variable (a b c : ℝ)
variable (condition1 : a + 3 * c = 51 * b)
variable (condition2 : a + 30 * c = 60 * b)

theorem cow_drink_pond :
  a + 3 * c = 51 * b →
  a + 30 * c = 60 * b →
  (9 * 17) / (7 * 2) = 75 := sorry
start

end cow_drink_pond_l116_116965


namespace first_term_geometric_series_l116_116397

variable (a : ℝ)
variable (r : ℝ := 1/4)
variable (S : ℝ := 80)

theorem first_term_geometric_series 
  (h1 : r = 1/4) 
  (h2 : S = 80)
  : a = 60 :=
by 
  sorry

end first_term_geometric_series_l116_116397


namespace lowest_possible_price_l116_116245

-- Definitions based on the provided conditions
def regular_discount_range : Set Real := {x | 0.10 ≤ x ∧ x ≤ 0.30}
def additional_discount : Real := 0.20
def retail_price : Real := 35.00

-- Problem statement transformed into Lean
theorem lowest_possible_price :
  ∃ d ∈ regular_discount_range, (retail_price * (1 - d)) * (1 - additional_discount) = 19.60 :=
by
  sorry

end lowest_possible_price_l116_116245


namespace minimum_value_of_expression_l116_116819

noncomputable def min_value (a b : ℝ) : ℝ := 1 / a + 3 / b

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) : min_value a b ≥ 16 := 
sorry

end minimum_value_of_expression_l116_116819


namespace part1_part2_part3_l116_116160

/-- Proof for part (1): If the point P lies on the x-axis, then m = -1. -/
theorem part1 (m : ℝ) (hx : 3 * m + 3 = 0) : m = -1 := 
by {
  sorry
}

/-- Proof for part (2): If point P lies on a line passing through A(-5, 1) and parallel to the y-axis, 
then the coordinates of point P are (-5, -12). -/
theorem part2 (m : ℝ) (hy : 2 * m + 5 = -5) : (2 * m + 5, 3 * m + 3) = (-5, -12) := 
by {
  sorry
}

/-- Proof for part (3): If point P is moved 2 right and 3 up to point M, 
and point M lies in the third quadrant with a distance of 7 from the y-axis, then the coordinates of M are (-7, -15). -/
theorem part3 (m : ℝ) 
  (hc : 2 * m + 7 = -7)
  (config : 3 * m + 6 < 0) : (2 * m + 7, 3 * m + 6) = (-7, -15) := 
by {
  sorry
}

end part1_part2_part3_l116_116160


namespace false_proposition_among_given_l116_116110

theorem false_proposition_among_given (a b c : Prop) : 
  (a = ∀ x : ℝ, ∃ y : ℝ, x = y) ∧
  (b = (∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2)) ∧
  (c = ∀ α β : ℝ, α = β ∧ ∃ P : Type, ∃ vertices : P, α = β ) → ¬c := by
  sorry

end false_proposition_among_given_l116_116110


namespace value_of_x_l116_116685

theorem value_of_x (x : ℝ) : 8^4 + 8^4 + 8^4 = 2^x → x = Real.log 3 / Real.log 2 + 12 :=
by
  sorry

end value_of_x_l116_116685


namespace solution_range_l116_116002

-- Given conditions from the table
variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom h₁ : f a b c 1.1 = -0.59
axiom h₂ : f a b c 1.2 = 0.84
axiom h₃ : f a b c 1.3 = 2.29
axiom h₄ : f a b c 1.4 = 3.76

theorem solution_range (a b c : ℝ) : 
  ∃ x : ℝ, 1.3 < x ∧ x < 1.4 ∧ f a b c x = 3 :=
sorry

end solution_range_l116_116002


namespace log_inequality_solution_l116_116294

variable {a x : ℝ}

theorem log_inequality_solution (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  (1 + Real.log (a ^ x - 1) / Real.log 2 ≤ Real.log (4 - a ^ x) / Real.log 2) →
  ((1 < a ∧ x ≤ Real.log (7 / 4) / Real.log a) ∨ (0 < a ∧ a < 1 ∧ x ≥ Real.log (7 / 4) / Real.log a)) :=
sorry

end log_inequality_solution_l116_116294


namespace average_mark_of_first_class_is_40_l116_116083

open Classical

noncomputable def average_mark_first_class (n1 n2 : ℕ) (m2 : ℕ) (a : ℚ) : ℚ :=
  let x := (a * (n1 + n2) - n2 * m2) / n1
  x

theorem average_mark_of_first_class_is_40 : average_mark_first_class 30 50 90 71.25 = 40 := by
  sorry

end average_mark_of_first_class_is_40_l116_116083


namespace quadratic_discriminant_correct_l116_116747

def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem quadratic_discriminant_correct :
  discriminant 5 (5 + 1/2) (-1/2) = 161 / 4 :=
by
  -- let's prove the equality directly
  sorry

end quadratic_discriminant_correct_l116_116747


namespace anna_coaching_days_l116_116399

/-- The total number of days from January 1 to September 4 in a non-leap year -/
def total_days_in_non_leap_year_up_to_sept4 : ℕ :=
  let days_in_january := 31
  let days_in_february := 28
  let days_in_march := 31
  let days_in_april := 30
  let days_in_may := 31
  let days_in_june := 30
  let days_in_july := 31
  let days_in_august := 31
  let days_up_to_sept4 := 4
  days_in_january + days_in_february + days_in_march + days_in_april +
  days_in_may + days_in_june + days_in_july + days_in_august + days_up_to_sept4

theorem anna_coaching_days : total_days_in_non_leap_year_up_to_sept4 = 247 :=
by
  -- Proof omitted
  sorry

end anna_coaching_days_l116_116399


namespace calculate_non_defective_m3_percentage_l116_116601

def percentage_non_defective_m3 : ℝ := 93

theorem calculate_non_defective_m3_percentage 
  (P : ℝ) -- Total number of products
  (P_pos : 0 < P) -- Total number of products is positive
  (percentage_m1 : ℝ := 0.40)
  (percentage_m2 : ℝ := 0.30)
  (percentage_m3 : ℝ := 0.30)
  (defective_m1 : ℝ := 0.03)
  (defective_m2 : ℝ := 0.01)
  (total_defective : ℝ := 0.036) :
  percentage_non_defective_m3 = 93 :=
by sorry -- The actual proof is omitted

end calculate_non_defective_m3_percentage_l116_116601


namespace yulgi_allowance_l116_116957

theorem yulgi_allowance (Y G : ℕ) (h₁ : Y + G = 6000) (h₂ : (Y + G) - (Y - G) = 4800) (h₃ : Y > G) : Y = 3600 :=
sorry

end yulgi_allowance_l116_116957


namespace repeating_decimal_fraction_l116_116800

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l116_116800


namespace meeting_at_centroid_l116_116724

theorem meeting_at_centroid :
  let A := (2, 9)
  let B := (-3, -4)
  let C := (6, -1)
  let centroid := ((2 - 3 + 6) / 3, (9 - 4 - 1) / 3)
  centroid = (5 / 3, 4 / 3) := sorry

end meeting_at_centroid_l116_116724


namespace compute_expression_l116_116898

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end compute_expression_l116_116898


namespace algebra_sum_l116_116191

-- Given conditions
def letterValue (ch : Char) : Int :=
  let pos := ch.toNat - 'a'.toNat + 1
  match pos % 6 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 0 => -2
  | _ => 0  -- This case is actually unreachable.

def wordValue (w : List Char) : Int :=
  w.foldl (fun acc ch => acc + letterValue ch) 0

theorem algebra_sum : wordValue ['a', 'l', 'g', 'e', 'b', 'r', 'a'] = 0 :=
  sorry

end algebra_sum_l116_116191


namespace cos_double_angle_zero_l116_116133

open Real

theorem cos_double_angle_zero (α : ℝ) (h : sin (π / 6 - α) = cos (π / 6 + α)) : cos (2 * α) = 0 :=
by
  sorry

end cos_double_angle_zero_l116_116133


namespace sum_of_fraction_components_l116_116796

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l116_116796


namespace count_students_neither_math_physics_chemistry_l116_116460

def total_students := 150

def students_math := 90
def students_physics := 70
def students_chemistry := 40

def students_math_and_physics := 20
def students_math_and_chemistry := 15
def students_physics_and_chemistry := 10
def students_all_three := 5

theorem count_students_neither_math_physics_chemistry :
  (total_students - 
   (students_math + students_physics + students_chemistry - 
    students_math_and_physics - students_math_and_chemistry - 
    students_physics_and_chemistry + students_all_three)) = 5 := by
  sorry

end count_students_neither_math_physics_chemistry_l116_116460


namespace probability_of_no_defective_pencils_l116_116157

open Nat

-- Define the total number of ways to select 5 pencils out of 15
def total_ways_to_choose : ℕ := (choose 15 5)

-- Define the number of ways to choose 5 non-defective pencils from 11 non-defective
def non_defective_ways_to_choose : ℕ := (choose 11 5)

-- Define the probability as a rational number
def probability_none_defective : ℚ := (non_defective_ways_to_choose : ℚ) / (total_ways_to_choose : ℚ)

theorem probability_of_no_defective_pencils :
  probability_none_defective = 154 / 1001 :=
by
  -- We assume this proof can be constructed from provided combinatorial calculations
  -- Add the necessary definitions to calculate total ways and non-defective ways
  sorry

end probability_of_no_defective_pencils_l116_116157


namespace combined_tax_rate_l116_116003

-- Definitions and conditions
def tax_rate_mork : ℝ := 0.45
def tax_rate_mindy : ℝ := 0.20
def income_ratio_mindy_to_mork : ℝ := 4

-- Theorem statement
theorem combined_tax_rate :
  ∀ (M : ℝ), (tax_rate_mork * M + tax_rate_mindy * (income_ratio_mindy_to_mork * M)) / (M + income_ratio_mindy_to_mork * M) = 0.25 :=
by
  intros M
  sorry

end combined_tax_rate_l116_116003


namespace initial_students_began_contest_l116_116600

theorem initial_students_began_contest
  (n : ℕ)
  (first_round_fraction : ℚ)
  (second_round_fraction : ℚ)
  (remaining_students : ℕ) :
  first_round_fraction * second_round_fraction * n = remaining_students →
  remaining_students = 18 →
  first_round_fraction = 0.3 →
  second_round_fraction = 0.5 →
  n = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_students_began_contest_l116_116600


namespace problem_solution_l116_116028

theorem problem_solution (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = a * x^2 + (b - 3) * x + 3) →
  (∀ x : ℝ, f x = f (-x)) →
  (a^2 - 2 = -a) →
  a + b = 4 :=
by
  intros h1 h2 h3
  sorry

end problem_solution_l116_116028


namespace band_formation_max_l116_116247

-- Define the conditions provided in the problem
theorem band_formation_max (m r x : ℕ) (h1 : m = r * x + 5)
  (h2 : (r - 3) * (x + 2) = m) (h3 : m < 100) :
  m = 70 :=
sorry

end band_formation_max_l116_116247


namespace circumcircle_radius_l116_116032

theorem circumcircle_radius (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) :
  let s₁ := a^2 + b^2
  let s₂ := c^2
  s₁ = s₂ → 
  (c / 2) = 6.5 :=
by
  sorry

end circumcircle_radius_l116_116032


namespace vectors_parallel_opposite_directions_l116_116306

theorem vectors_parallel_opposite_directions
  (a b : ℝ × ℝ)
  (h₁ : a = (-1, 2))
  (h₂ : b = (2, -4)) :
  b = (-2 : ℝ) • a ∧ b = -2 • a :=
by
  sorry

end vectors_parallel_opposite_directions_l116_116306


namespace total_cost_of_pens_and_notebooks_l116_116550

theorem total_cost_of_pens_and_notebooks (a b : ℝ) : 5 * a + 8 * b = 5 * a + 8 * b := 
by 
  sorry

end total_cost_of_pens_and_notebooks_l116_116550


namespace heavy_cream_cost_l116_116628

theorem heavy_cream_cost
  (cost_strawberries : ℕ)
  (cost_raspberries : ℕ)
  (total_cost : ℕ)
  (cost_heavy_cream : ℕ) :
  (cost_strawberries = 3 * 2) →
  (cost_raspberries = 5 * 2) →
  (total_cost = 20) →
  (cost_heavy_cream = total_cost - (cost_strawberries + cost_raspberries)) →
  cost_heavy_cream = 4 :=
by
  sorry

end heavy_cream_cost_l116_116628


namespace value_of_expression_l116_116591

theorem value_of_expression (x : ℝ) (h : x ^ 2 - 3 * x + 1 = 0) : 
  x ≠ 0 → (x ^ 2) / (x ^ 4 + x ^ 2 + 1) = 1 / 8 :=
by 
  intros h1 
  sorry

end value_of_expression_l116_116591


namespace smallest_largest_number_in_list_l116_116241

theorem smallest_largest_number_in_list :
  ∃ (a b c d e : ℕ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ 
  (a + b + c + d + e = 50) ∧ (e - a = 20) ∧ 
  (c = 6) ∧ (b = 6) ∧ 
  (e = 20) :=
by
  sorry

end smallest_largest_number_in_list_l116_116241


namespace john_initial_candies_l116_116326

theorem john_initial_candies : ∃ x : ℕ, (∃ (x3 : ℕ), x3 = ((x - 2) / 2) ∧ x3 = 6) ∧ x = 14 := by
  sorry

end john_initial_candies_l116_116326


namespace pirates_treasure_l116_116524

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l116_116524


namespace cameron_list_count_l116_116254

-- Definitions
def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b
def is_perfect_square (n : ℕ) : Prop := ∃ m, n = m * m
def is_perfect_cube (n : ℕ) : Prop := ∃ m, n = m * m * m

-- The main statement
theorem cameron_list_count :
  let smallest_square := 25
  let smallest_cube := 125
  (∀ n : ℕ, is_multiple_of n 25 → smallest_square ≤ n → n ≤ smallest_cube) →
  ∃ count : ℕ, count = 5 :=
by 
  sorry

end cameron_list_count_l116_116254


namespace f_800_l116_116618

noncomputable def f : ℕ → ℕ := sorry

axiom axiom1 : ∀ x y : ℕ, 0 < x → 0 < y → f (x * y) = f x + f y
axiom axiom2 : f 10 = 10
axiom axiom3 : f 40 = 14

theorem f_800 : f 800 = 26 :=
by
  -- Apply the conditions here
  sorry

end f_800_l116_116618


namespace age_difference_l116_116390

theorem age_difference
  (A B : ℕ)
  (hB : B = 48)
  (h_condition : A + 10 = 2 * (B - 10)) :
  A - B = 18 :=
by
  sorry

end age_difference_l116_116390


namespace min_sum_of_diagonals_l116_116069

theorem min_sum_of_diagonals (x y : ℝ) (α : ℝ) (hx : 0 < x) (hy : 0 < y) (hα : 0 < α ∧ α < π) (h_area : x * y * Real.sin α = 2) : x + y ≥ 2 * Real.sqrt 2 :=
sorry

end min_sum_of_diagonals_l116_116069


namespace g_inv_g_inv_14_l116_116840

noncomputable def g (x : ℝ) := 3 * x - 4
noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end g_inv_g_inv_14_l116_116840


namespace nat_pair_solution_l116_116573

theorem nat_pair_solution (x y : ℕ) : 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by
  sorry

end nat_pair_solution_l116_116573


namespace move_right_by_three_units_l116_116829

theorem move_right_by_three_units :
  (-1 + 3 = 2) :=
  by { sorry }

end move_right_by_three_units_l116_116829


namespace at_most_n_diameters_l116_116929

theorem at_most_n_diameters {n : ℕ} (h : n ≥ 3) (points : Fin n → ℝ × ℝ) (d : ℝ) 
  (hd : ∀ i j, dist (points i) (points j) ≤ d) :
  ∃ (diameters : Fin n → Fin n), 
    (∀ i, dist (points i) (points (diameters i)) = d) ∧
    (∀ i j, (dist (points i) (points j) = d) → 
      (∃ k, k = i ∨ k = j → diameters k = if k = i then j else i)) :=
sorry

end at_most_n_diameters_l116_116929


namespace total_dollar_amount_l116_116698

/-- Definitions of base 5 numbers given in the problem -/
def pearls := 1 * 5^0 + 2 * 5^1 + 3 * 5^2 + 4 * 5^3
def silk := 1 * 5^0 + 1 * 5^1 + 1 * 5^2 + 1 * 5^3
def spices := 1 * 5^0 + 2 * 5^1 + 2 * 5^2
def maps := 0 * 5^0 + 1 * 5^1

/-- The theorem to prove the total dollar amount in base 10 -/
theorem total_dollar_amount : pearls + silk + spices + maps = 808 :=
by
  sorry

end total_dollar_amount_l116_116698


namespace compute_fraction_power_l116_116901

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end compute_fraction_power_l116_116901


namespace restaurant_made_correct_amount_l116_116552

noncomputable def restaurant_revenue : ℝ := 
  let price1 := 8
  let qty1 := 10
  let price2 := 10
  let qty2 := 5
  let price3 := 4
  let qty3 := 20
  let total_sales := qty1 * price1 + qty2 * price2 + qty3 * price3
  let discount := 0.10
  let discounted_total := total_sales * (1 - discount)
  let sales_tax := 0.05
  let final_amount := discounted_total * (1 + sales_tax)
  final_amount

theorem restaurant_made_correct_amount : restaurant_revenue = 198.45 := by
  sorry

end restaurant_made_correct_amount_l116_116552


namespace pirates_treasure_l116_116491

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l116_116491


namespace number_of_poles_needed_l116_116104

def length := 90
def width := 40
def distance_between_poles := 5

noncomputable def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem number_of_poles_needed (l w d : ℕ) : perimeter l w / d = 52 :=
by
  rw [perimeter]
  sorry

end number_of_poles_needed_l116_116104


namespace sum_f_84_eq_1764_l116_116975

theorem sum_f_84_eq_1764 (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, 0 < n → f n < f (n + 1))
  (h2 : ∀ m n : ℕ, 0 < m → 0 < n → f (m * n) = f m * f n)
  (h3 : ∀ m n : ℕ, 0 < m → 0 < n → m ≠ n → m^n = n^m → (f m = n ∨ f n = m)) :
  f 84 = 1764 :=
by
  sorry

end sum_f_84_eq_1764_l116_116975


namespace gcd_of_90_and_405_l116_116272

def gcd_90_405 : ℕ := Nat.gcd 90 405

theorem gcd_of_90_and_405 : gcd_90_405 = 45 :=
by
  -- proof goes here
  sorry

end gcd_of_90_and_405_l116_116272


namespace augmented_matrix_solution_l116_116581

theorem augmented_matrix_solution (a b : ℝ) 
    (h1 : (∀ (x y : ℝ), (a * x = 2 ∧ y = b ↔ x = 2 ∧ y = 1))) : 
    a + b = 2 :=
by
  sorry

end augmented_matrix_solution_l116_116581


namespace one_cow_empties_pond_in_75_days_l116_116966

-- Define the necessary variables and their types
variable (c a b : ℝ) -- c represents daily water inflow from the spring
                      -- a represents the total volume of the pond
                      -- b represents the daily consumption per cow

-- Define the conditions
def condition1 : Prop := a + 3 * c = 3 * 17 * b
def condition2 : Prop := a + 30 * c = 30 * 2 * b

-- Target statement we want to prove
theorem one_cow_empties_pond_in_75_days (h1 : condition1 c a b) (h2 : condition2 c a b) :
  ∃ t : ℝ, t = 75 := 
sorry -- Proof to be provided


end one_cow_empties_pond_in_75_days_l116_116966


namespace subscription_difference_is_4000_l116_116720

-- Given definitions
def total_subscription (A B C : ℕ) : Prop :=
  A + B + C = 50000

def subscription_B (x : ℕ) : ℕ :=
  x + 5000

def subscription_A (x y : ℕ) : ℕ :=
  x + 5000 + y

def profit_ratio (profit_C total_profit x : ℕ) : Prop :=
  (profit_C : ℚ) / total_profit = (x : ℚ) / 50000

-- Prove that A subscribed Rs. 4,000 more than B
theorem subscription_difference_is_4000 (x y : ℕ)
  (h1 : total_subscription (subscription_A x y) (subscription_B x) x)
  (h2 : profit_ratio 8400 35000 x) :
  y = 4000 :=
sorry

end subscription_difference_is_4000_l116_116720


namespace largest_hole_leakage_rate_l116_116226

theorem largest_hole_leakage_rate (L : ℝ) (h1 : 600 = (L + L / 2 + L / 6) * 120) : 
  L = 3 :=
sorry

end largest_hole_leakage_rate_l116_116226


namespace distance_to_store_l116_116337

noncomputable def D : ℝ := 4

theorem distance_to_store :
  (1/3) * (D/2 + D/10 + D/10) = 56/60 :=
by
  sorry

end distance_to_store_l116_116337


namespace set_intersection_l116_116034

def A := {x : ℝ | -5 < x ∧ x < 2}
def B := {x : ℝ | |x| < 3}

theorem set_intersection : {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | -3 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l116_116034


namespace find_x_from_ratio_l116_116439

theorem find_x_from_ratio (x y k: ℚ) 
  (h1 : ∀ x y, (5 * x - 3) / (y + 20) = k) 
  (h2 : 5 * 1 - 3 = 2 * 22) (hy : y = 5) : 
  x = 58 / 55 := 
by 
  sorry

end find_x_from_ratio_l116_116439


namespace part1_am_eq_ln_am1_minus_1_part2_am_le_am1_minus_2_part3_k_is_3_l116_116134

noncomputable def f (x : ℝ) := Real.log x
noncomputable def deriv_f (x : ℝ) := 1 / x

theorem part1_am_eq_ln_am1_minus_1 (a_n : ℕ → ℝ) (m : ℕ) (h : m ≥ 2) :
  a_n m = Real.log (a_n (m - 1)) - 1 :=
sorry

theorem part2_am_le_am1_minus_2 (a_n : ℕ → ℝ) (m : ℕ) (h : m ≥ 2) :
  a_n m ≤ a_n (m - 1) - 2 :=
sorry

theorem part3_k_is_3 (a_n : ℕ → ℝ) :
  ∃ k : ℕ, k = 3 ∧ ∀ n : ℕ, n ≤ k → (a_n n) - (a_n (n - 1)) = (a_n 2) - (a_n 1) :=
sorry

end part1_am_eq_ln_am1_minus_1_part2_am_le_am1_minus_2_part3_k_is_3_l116_116134


namespace find_f_105_5_l116_116295

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom product_condition : ∀ x : ℝ, f x * f (x + 2) = -1
axiom specific_interval : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x = x

theorem find_f_105_5 : f 105.5 = 2.5 :=
by
  sorry

end find_f_105_5_l116_116295


namespace number_of_second_graders_l116_116081

-- Define the number of kindergartners, first graders, and total students
def k : ℕ := 14
def f : ℕ := 24
def t : ℕ := 42

-- Define the number of second graders
def s : ℕ := t - (k + f)

-- The theorem to prove
theorem number_of_second_graders : s = 4 := by
  -- We can use sorry here since we are not required to provide the proof
  sorry

end number_of_second_graders_l116_116081


namespace find_m_over_n_l116_116588

variable (a b : ℝ × ℝ)
variable (m n : ℝ)
variable (n_nonzero : n ≠ 0)

axiom a_def : a = (1, 2)
axiom b_def : b = (-2, 3)
axiom collinear : ∃ k : ℝ, m • a - n • b = k • (a + 2 • b)

theorem find_m_over_n : m / n = -1 / 2 := by
  sorry

end find_m_over_n_l116_116588


namespace problem_equivalence_l116_116540

theorem problem_equivalence : 4 * 299 + 3 * 299 + 2 * 299 + 298 = 2989 := by
  sorry

end problem_equivalence_l116_116540


namespace triangle_PR_eq_8_l116_116451

open Real

theorem triangle_PR_eq_8 (P Q R M : ℝ) 
  (PQ QR PM : ℝ) 
  (hPQ : PQ = 6) (hQR : QR = 10) (hPM : PM = 5) 
  (M_midpoint : M = (Q + R) / 2) :
  dist P R = 8 :=
by
  sorry

end triangle_PR_eq_8_l116_116451


namespace seq_100_gt_14_l116_116843

variable {a : ℕ → ℝ}

axiom seq_def (n : ℕ) : a 0 = 1 ∧ (∀ n ≥ 0, a (n + 1) = a n + 1 / a n)

theorem seq_100_gt_14 : a 100 > 14 :=
by
  -- Establish sequence definition
  have h1 : a 0 = 1 := (seq_def 0).left,
  have h2 : ∀ n ≥ 0, a (n + 1) = a n + 1 / a n := (seq_def 0).right,
  sorry

end seq_100_gt_14_l116_116843


namespace quad_eq_diagonals_theorem_l116_116427

noncomputable def quad_eq_diagonals (a b c d m n : ℝ) (A C : ℝ) : Prop :=
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C)

theorem quad_eq_diagonals_theorem (a b c d m n A C : ℝ) :
  quad_eq_diagonals a b c d m n A C :=
by
  sorry

end quad_eq_diagonals_theorem_l116_116427


namespace range_of_a_l116_116583

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (hf_odd : ∀ x, f (-x) = -f x)
  (hf_expr_pos : ∀ x, x > 0 → f x = -x^2 + ax - 1 - a)
  (hf_monotone : ∀ x y, x < y → f y ≤ f x) :
  -1 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l116_116583


namespace bruce_total_amount_paid_l116_116691

-- Definitions for quantities and rates
def quantity_of_grapes : Nat := 8
def rate_per_kg_grapes : Nat := 70
def quantity_of_mangoes : Nat := 11
def rate_per_kg_mangoes : Nat := 55

-- Calculate individual costs
def cost_of_grapes := quantity_of_grapes * rate_per_kg_grapes
def cost_of_mangoes := quantity_of_mangoes * rate_per_kg_mangoes

-- Calculate total amount paid
def total_amount_paid := cost_of_grapes + cost_of_mangoes

-- Statement to prove
theorem bruce_total_amount_paid : total_amount_paid = 1165 := by
  -- Proof is intentionally left as a placeholder
  sorry

end bruce_total_amount_paid_l116_116691


namespace person_A_boxes_average_unit_price_after_promotion_l116_116871

-- Definitions based on the conditions.
def unit_price (x: ℕ) (y: ℕ) : ℚ := y / x

def person_A_spent : ℕ := 2400
def person_B_spent : ℕ := 3000
def promotion_discount : ℕ := 20
def boxes_difference : ℕ := 10

-- Main proofs
theorem person_A_boxes (unit_price: ℕ → ℕ → ℚ) 
  (person_A_spent person_B_spent boxes_difference: ℕ): 
  ∃ x, unit_price person_A_spent x = unit_price person_B_spent (x + boxes_difference) 
  ∧ x = 40 := 
by {
  sorry
}

theorem average_unit_price_after_promotion (unit_price: ℕ → ℕ → ℚ) 
  (promotion_discount: ℕ) (person_A_spent person_B_spent: ℕ) 
  (boxes_A_promotion boxes_B: ℕ): 
  person_A_spent / (boxes_A_promotion * 2) + 20 = 48 
  ∧ person_B_spent / (boxes_B * 2) + 20 = 50 :=
by {
  sorry
}

end person_A_boxes_average_unit_price_after_promotion_l116_116871


namespace range_of_a_l116_116584

theorem range_of_a 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : r > 0) 
  (cos_le_zero : (3 * a - 9) / r ≤ 0) 
  (sin_gt_zero : (a + 2) / r > 0) : 
  -2 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l116_116584


namespace pirate_treasure_l116_116510

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l116_116510


namespace room_length_l116_116477

theorem room_length
  (width : ℝ)
  (cost_rate : ℝ)
  (total_cost : ℝ)
  (h_width : width = 4)
  (h_cost_rate : cost_rate = 850)
  (h_total_cost : total_cost = 18700) :
  ∃ L : ℝ, L = 5.5 ∧ total_cost = cost_rate * (L * width) :=
by
  sorry

end room_length_l116_116477


namespace eval_g_inv_g_inv_14_l116_116839

variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)

axiom g_def : ∀ x, g x = 3 * x - 4
axiom g_inv_def : ∀ y, g_inv y = (y + 4) / 3

theorem eval_g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by
    sorry

end eval_g_inv_g_inv_14_l116_116839


namespace integral_substitution_l116_116696

open Set Function Filter

variables {a b : ℝ} {φ : ℝ → ℝ} (f : ℝ → ℝ)

-- Assumptions
-- 1. φ = φ(y) is a smooth function on [a, b] such that φ(a) < φ(b)
-- 2. f = f(x) is a Borel measurable function integrable on [φ(a), φ(b)]
-- Define "φ smooth" as having continuous derivative, which is continuous differentiability
def φ_smooth (a b : ℝ) (φ : ℝ → ℝ) : Prop := Continuous φ ∧ Continuous (λ y, deriv φ y)

-- Prove: ∫ (x in φ(a)..φ(b)), f(x) dx = ∫ (y in a..b), f(φ(y)) * deriv φ y dy
theorem integral_substitution (ha : φ_smooth a b φ) (hφa : φ a < φ b) 
  {μ : MeasureTheory.Measure ℝ}
  (μab : μ (Ioc a b) < ∞)
  (h_int_f : MeasureTheory.IntegrableOn f (Ioc (φ a) (φ b)) μ) : 
  ∫ x in (Set.interval (φ a) (φ b)), f x ∂μ = 
  ∫ y in (Set.interval a b), f (φ y) * deriv φ y ∂μ :=
by
  sorry

end integral_substitution_l116_116696


namespace pirate_treasure_l116_116512

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l116_116512


namespace total_milk_consumed_l116_116448

theorem total_milk_consumed (regular_milk : ℝ) (soy_milk : ℝ) (H1 : regular_milk = 0.5) (H2: soy_milk = 0.1) :
    regular_milk + soy_milk = 0.6 :=
  by
  sorry

end total_milk_consumed_l116_116448


namespace reduced_price_l116_116248

theorem reduced_price (P R : ℝ) (Q : ℝ) 
  (h1 : R = 0.80 * P) 
  (h2 : 600 = Q * P) 
  (h3 : 600 = (Q + 4) * R) : 
  R = 30 :=
by
  sorry

end reduced_price_l116_116248


namespace pirates_treasure_l116_116487

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l116_116487


namespace solution_set_of_inequality_l116_116025

theorem solution_set_of_inequality :
  {x : ℝ | (x-1)*(2-x) ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l116_116025


namespace highest_more_than_lowest_by_37_5_percent_l116_116846

variables (highest_price lowest_price : ℝ)

theorem highest_more_than_lowest_by_37_5_percent
  (h_highest : highest_price = 22)
  (h_lowest : lowest_price = 16) :
  ((highest_price - lowest_price) / lowest_price) * 100 = 37.5 :=
by
  sorry

end highest_more_than_lowest_by_37_5_percent_l116_116846


namespace probability_of_shortening_exactly_one_digit_l116_116878

theorem probability_of_shortening_exactly_one_digit :
  let n := 2014 in
  let p := 0.1 in
  let q := 0.9 in
  (n.choose 1) * p ^ 1 * q ^ (n - 1) = 201.4 * 1.564 * 10^(-90) :=
by sorry

end probability_of_shortening_exactly_one_digit_l116_116878


namespace evaluate_g_at_3_l116_116590

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 + 3 * x - 2

theorem evaluate_g_at_3 : g 3 = 79 := by
  sorry

end evaluate_g_at_3_l116_116590


namespace pirates_treasure_l116_116490

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l116_116490


namespace sum_of_squares_ineq_l116_116352

theorem sum_of_squares_ineq (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum_sq : a^2 + b^2 + c^2 = 3) :
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 :=
sorry

end sum_of_squares_ineq_l116_116352


namespace calculate_fraction_l116_116530

theorem calculate_fraction :
  (2019 + 1981)^2 / 121 = 132231 := 
  sorry

end calculate_fraction_l116_116530


namespace repeating_decimal_fraction_l116_116798

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l116_116798


namespace pirates_treasure_l116_116494

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l116_116494


namespace max_height_of_rock_l116_116553

theorem max_height_of_rock : 
    ∃ t_max : ℝ, (∀ t : ℝ, -5 * t^2 + 25 * t + 10 ≤ -5 * t_max^2 + 25 * t_max + 10) ∧ (-5 * t_max^2 + 25 * t_max + 10 = 165 / 4) := 
sorry

end max_height_of_rock_l116_116553


namespace age_difference_is_12_l116_116537

noncomputable def age_difference (x : ℕ) : ℕ :=
  let older := 3 * x
  let younger := 2 * x
  older - younger

theorem age_difference_is_12 :
  ∃ x : ℕ, 3 * x + 2 * x = 60 ∧ age_difference x = 12 :=
by
  sorry

end age_difference_is_12_l116_116537


namespace repeated_decimal_to_fraction_l116_116779

theorem repeated_decimal_to_fraction :
  ∀ (a b : ℕ), (0.35.recursor = 35 / 99 ∧ Nat.gcd 35 99 = 1) → a + b = 134 := by
  sorry

end repeated_decimal_to_fraction_l116_116779


namespace shorten_by_one_expected_length_l116_116883

-- Definition of the random sequence and binomial distribution properties
def digit_sequence {n : ℕ} (p : ℝ) (len : ℕ) : ℝ := 
  let q := 1 - p in
  (len.choose 1) * (p^1 * q^(len - 1))

-- Part (a) statement
theorem shorten_by_one 
  (len : ℕ) 
  (p : ℝ := 0.1) 
  (q := 1 - p) 
  (x := len - 1) : 
  digit_sequence p x = 1.564e-90 :=
by sorry

-- Part (b) statement
theorem expected_length 
  (initial_len : ℕ) 
  (p : ℝ := 0.1) (removals := initial_len - 1) 
  (expected_removed : ℝ := removals * p) : 
  (initial_len : ℝ) - expected_removed = 1813.6 :=
by sorry

end shorten_by_one_expected_length_l116_116883


namespace complex_division_l116_116419

theorem complex_division (i : ℂ) (h : i * i = -1) : 3 / (1 - i) ^ 2 = (3 / 2) * i :=
by
  sorry

end complex_division_l116_116419


namespace sequence_difference_l116_116607

theorem sequence_difference
  (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a (n + 1) - a n - n = 0) :
  a 2017 - a 2016 = 2016 :=
by
  sorry

end sequence_difference_l116_116607


namespace problem_statement_l116_116609

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (A B C D : V)
  (BC CD : V)
  (AC AB AD : V)

theorem problem_statement
  (h1 : BC = 2 • CD)
  (h2 : BC = AC - AB) :
  AD = (3 / 2 : ℝ) • AC - (1 / 2 : ℝ) • AB :=
sorry

end problem_statement_l116_116609


namespace paving_stone_width_l116_116545

theorem paving_stone_width
  (courtyard_length : ℝ) (courtyard_width : ℝ)
  (num_paving_stones : ℕ) (paving_stone_length : ℝ)
  (courtyard_area : ℝ) (paving_stone_area : ℝ)
  (width : ℝ)
  (h1 : courtyard_length = 30)
  (h2 : courtyard_width = 16.5)
  (h3 : num_paving_stones = 99)
  (h4 : paving_stone_length = 2.5)
  (h5 : courtyard_area = courtyard_length * courtyard_width)
  (h6 : courtyard_area = 495)
  (h7 : paving_stone_area = courtyard_area / num_paving_stones)
  (h8 : paving_stone_area = 5)
  (h9 : paving_stone_area = paving_stone_length * width) :
  width = 2 := by
  sorry

end paving_stone_width_l116_116545


namespace find_number_l116_116700

theorem find_number (x : ℝ) :
  (7 * (x + 10) / 5) - 5 = 44 → x = 25 :=
by
  sorry

end find_number_l116_116700


namespace track_time_is_80_l116_116357

noncomputable def time_to_complete_track
  (a b : ℕ) 
  (meetings : a = 15 ∧ b = 25) : ℕ :=
a + b

theorem track_time_is_80 (a b : ℕ) (meetings : a = 15 ∧ b = 25) : time_to_complete_track a b meetings = 80 := by
  sorry

end track_time_is_80_l116_116357


namespace permutations_five_three_eq_sixty_l116_116543

theorem permutations_five_three_eq_sixty : (Nat.factorial 5) / (Nat.factorial (5 - 3)) = 60 := 
by
  sorry

end permutations_five_three_eq_sixty_l116_116543


namespace tan_alpha_l116_116291

theorem tan_alpha (α : ℝ) (h1 : Real.sin (Real.pi - α) = 3/5) (h2 : Real.pi / 2 < α ∧ α < Real.pi) : Real.tan α = -3/4 := 
  sorry

end tan_alpha_l116_116291


namespace find_a_l116_116928

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb 2 (x^2 + a)

theorem find_a (a : ℝ) : f 3 a = 1 → a = -7 :=
by
  intro h
  unfold f at h
  sorry

end find_a_l116_116928


namespace simplify_fraction_product_l116_116635

theorem simplify_fraction_product : 
  (21 / 28) * (14 / 33) * (99 / 42) = 1 := 
by 
  sorry

end simplify_fraction_product_l116_116635


namespace weight_of_empty_jar_l116_116532

variable (W : ℝ) -- Weight of the empty jar
variable (w : ℝ) -- Weight of water for one-fifth of the jar

-- Conditions
variable (h1 : W + w = 560)
variable (h2 : W + 4 * w = 740)

-- Theorem statement
theorem weight_of_empty_jar (W w : ℝ) (h1 : W + w = 560) (h2 : W + 4 * w = 740) : W = 500 := 
by
  sorry

end weight_of_empty_jar_l116_116532


namespace cardinality_union_l116_116329

open Finset

theorem cardinality_union (A B : Finset ℕ) (h : 2 ^ A.card + 2 ^ B.card - 2 ^ (A ∩ B).card = 144) : (A ∪ B).card = 8 := 
by 
  sorry

end cardinality_union_l116_116329


namespace triangle_inequality_l116_116967

theorem triangle_inequality
  (A B C : ℝ)
  (hA : 0 < A)
  (hB : 0 < B)
  (hC : 0 < C)
  (hABC : A + B + C = Real.pi) :
  Real.sin (3 * A / 2) + Real.sin (3 * B / 2) + Real.sin (3 * C / 2) ≤
  Real.cos ((A - B) / 2) + Real.cos ((B - C) / 2) + Real.cos ((C - A) / 2) :=
by
  sorry

end triangle_inequality_l116_116967


namespace how_many_necklaces_given_away_l116_116565

-- Define the initial conditions
def initial_necklaces := 50
def broken_necklaces := 3
def bought_necklaces := 5
def final_necklaces := 37

-- Define the question proof statement
theorem how_many_necklaces_given_away : 
  (initial_necklaces - broken_necklaces + bought_necklaces - final_necklaces) = 15 :=
by sorry

end how_many_necklaces_given_away_l116_116565


namespace find_x_l116_116761

noncomputable def xSolution (x : ℝ) : Prop := 
  (0 < x ∧ x < π / 2) ∧ 
  (∃ a : ℝ, (log (cos x) / log 2)).frac = 0) ∧ 
  (∃ b : ℝ, (log (sqrt (tan x))).frac = 0) ∧ 
  (((log (cos x) / log 2).intPart + (log (sqrt (tan x))).intPart) = 1) → 
  x = Real.arcsin ((sqrt 5 - 1) / 2)

-- This theorem states that x satisfies various conditions and should equal arcsin((sqrt 5 - 1) / 2)
theorem find_x : ∃ x : ℝ, xSolution x := by
  sorry

end find_x_l116_116761


namespace maria_needs_flour_l116_116551

-- Definitions from conditions
def cups_of_flour_per_cookie (c : ℕ) (f : ℚ) : ℚ := f / c

def total_cups_of_flour (cps_per_cookie : ℚ) (num_cookies : ℕ) : ℚ := cps_per_cookie * num_cookies

-- Given values
def cookies_20 := 20
def flour_3 := 3
def cookies_100 := 100

theorem maria_needs_flour :
  total_cups_of_flour (cups_of_flour_per_cookie cookies_20 flour_3) cookies_100 = 15 :=
by
  sorry -- Proof is omitted

end maria_needs_flour_l116_116551


namespace max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_l116_116952

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem max_ab_bc_ca : ab + bc + ca ≤ 1 / 3 :=
by sorry

theorem a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2 :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ 1 / 2 :=
by sorry

end max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_l116_116952


namespace train_overtake_distance_l116_116216

/--
 Train A leaves the station traveling at 30 miles per hour.
 Two hours later, Train B leaves the same station traveling in the same direction at 42 miles per hour.
 Prove that Train A is overtaken by Train B 210 miles from the station.
-/
theorem train_overtake_distance
    (speed_A : ℕ) (speed_B : ℕ) (delay_B : ℕ)
    (hA : speed_A = 30)
    (hB : speed_B = 42)
    (hDelay : delay_B = 2) :
    ∃ d : ℕ, d = 210 ∧ ∀ t : ℕ, (speed_B * t = (speed_A * t + speed_A * delay_B) → d = speed_B * t) :=
by
  sorry

end train_overtake_distance_l116_116216


namespace m_divides_n_l116_116060

theorem m_divides_n 
  (m n : ℕ) 
  (hm_pos : 0 < m) 
  (hn_pos : 0 < n) 
  (h : 5 * m + n ∣ 5 * n + m) 
  : m ∣ n :=
sorry

end m_divides_n_l116_116060


namespace repeating_decimal_sum_l116_116784

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (∀ d, d ∣ a ∧ d ∣ b ↔ d = 1) ∧ 
  (a + b = 134) ∧ (∃ x : ℚ, x = 35 / 99 ∧ (x = 0.353535...)) :=
begin
  sorry
end

end repeating_decimal_sum_l116_116784


namespace expected_value_of_girls_left_of_boys_l116_116372

def num_girls_to_left_of_all_boys (boys girls : ℕ) : ℚ :=
  (boys + girls : ℚ) / (boys + 1)

theorem expected_value_of_girls_left_of_boys :
  num_girls_to_left_of_all_boys 10 7 = 7 / 11 := by
  sorry

end expected_value_of_girls_left_of_boys_l116_116372


namespace tree_sidewalk_space_l116_116958

theorem tree_sidewalk_space (num_trees : ℕ) (tree_distance: ℝ) (total_road_length: ℝ): 
  num_trees = 13 → 
  tree_distance = 12 → 
  total_road_length = 157 → 
  (total_road_length - tree_distance * (num_trees - 1)) / num_trees = 1 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end tree_sidewalk_space_l116_116958


namespace ellen_bakes_6_balls_of_dough_l116_116412

theorem ellen_bakes_6_balls_of_dough (rising_time baking_time total_time : ℕ) (h_rise : rising_time = 3) (h_bake : baking_time = 2) (h_total : total_time = 20) :
  ∃ n : ℕ, (rising_time + baking_time) + rising_time * (n - 1) = total_time ∧ n = 6 :=
by sorry

end ellen_bakes_6_balls_of_dough_l116_116412


namespace no_matrix_adds_three_to_second_column_l116_116917

theorem no_matrix_adds_three_to_second_column :
  ¬ ∃ (M : Matrix (Fin 2) (Fin 2) ℚ), 
    ∀ (X : Matrix (Fin 2) (Fin 2) ℚ), 
    M.mul X = X + (Matrix.vecCons (Matrix.vecCons 0 3 : Fin 2 → ℚ) (Matrix.vecCons 0 3 : Fin 2 → ℚ)) :=
begin
  sorry
end

end no_matrix_adds_three_to_second_column_l116_116917


namespace dan_licks_l116_116907

/-- 
Given that Michael takes 63 licks, Sam takes 70 licks, David takes 70 licks, 
Lance takes 39 licks, and the average number of licks for all five people is 60, 
prove that Dan takes 58 licks to get to the center of a lollipop.
-/
theorem dan_licks (D : ℕ) 
  (M : ℕ := 63) 
  (S : ℕ := 70) 
  (Da : ℕ := 70) 
  (L : ℕ := 39)
  (avg : ℕ := 60) :
  ((M + S + Da + L + D) / 5 = avg) → D = 58 :=
by sorry

end dan_licks_l116_116907


namespace negation_example_l116_116091

theorem negation_example :
  (¬ ∀ x y : ℝ, |x + y| > 3) ↔ (∃ x y : ℝ, |x + y| ≤ 3) :=
by
  sorry

end negation_example_l116_116091


namespace greatest_int_lt_neg_31_div_6_l116_116362

theorem greatest_int_lt_neg_31_div_6 : ∃ (n : ℤ), n < -31 / 6 ∧ ∀ m : ℤ, m < -31 / 6 → m ≤ n := 
sorry

end greatest_int_lt_neg_31_div_6_l116_116362


namespace final_segment_distance_l116_116906

theorem final_segment_distance :
  let north_distance := 2
  let east_distance := 1
  let south_distance := 1
  let net_north := north_distance - south_distance
  let net_east := east_distance
  let final_distance := Real.sqrt (net_north ^ 2 + net_east ^ 2)
  final_distance = Real.sqrt 2 :=
by
  sorry

end final_segment_distance_l116_116906


namespace intersection_A_B_range_m_l116_116144

-- Define set A when m = 3 as given
def A_set (x : ℝ) : Prop := 3 - 2 * x - x^2 ≥ 0
def A (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1

-- Define set B when m = 3 as given
def B_set (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def B (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 4

-- The intersection of A and B should be: -2 ≤ x ≤ 1
theorem intersection_A_B : ∀ (x : ℝ), A x ∧ B x ↔ (-2 ≤ x ∧ x ≤ 1) := sorry

-- Define A for general m > 0
def A_set_general (x : ℝ) : Prop := 3 - 2 * x - x^2 ≥ 0

-- Define B for general m
def B_set_general (x : ℝ) (m : ℝ) : Prop := (x - 1)^2 ≤ m^2

-- Prove the range for m such that A ⊆ B
theorem range_m (m : ℝ) (h : m > 0) : (∀ x, A_set_general x → B_set_general x m) ↔ m ≥ 4 := sorry

end intersection_A_B_range_m_l116_116144


namespace pirates_treasure_l116_116521

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l116_116521


namespace Jim_remaining_distance_l116_116053

theorem Jim_remaining_distance (t d r : ℕ) (h₁ : t = 1200) (h₂ : d = 923) (h₃ : r = t - d) : r = 277 := 
by 
  -- Proof steps would go here
  sorry

end Jim_remaining_distance_l116_116053


namespace third_smallest_four_digit_in_pascals_triangle_l116_116529

theorem third_smallest_four_digit_in_pascals_triangle : 
  ∃ n k, (nat.choose n k) = 1002 ∧ (∀ k' < k, (nat.choose n k') < 1002) ∧ 
         (∃ n1 k1 n2 k2, (nat.choose n1 k1) = 1000 ∧ (nat.choose n2 k2) = 1001 ∧ 
                         (∀ k1' < k1, (nat.choose n1 k1') < 1000) ∧ 
                         (∀ k2' < k2, (nat.choose n2 k2') < 1001)) := 
by
  sorry

end third_smallest_four_digit_in_pascals_triangle_l116_116529


namespace smallest_unrepresentable_integer_l116_116024

theorem smallest_unrepresentable_integer :
  ∃ n : ℕ, (∀ a b c d : ℕ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) → 
  n ≠ (2^a - 2^b) / (2^c - 2^d)) ∧ n = 11 :=
by
  sorry

end smallest_unrepresentable_integer_l116_116024


namespace cost_to_fill_half_of_CanB_l116_116004

theorem cost_to_fill_half_of_CanB (r h : ℝ) (C_cost : ℝ) (VC VB : ℝ) 
(h1 : VC = 2 * VB) 
(h2 : VB = Real.pi * r^2 * h) 
(h3 : VC = Real.pi * (2 * r)^2 * (h / 2)) 
(h4 : C_cost = 16):
  C_cost / 4 = 4 :=
by
  sorry

end cost_to_fill_half_of_CanB_l116_116004


namespace compute_f_pi_over_2_l116_116978

noncomputable def f (x : ℝ) (ω : ℝ) (b : ℝ) : ℝ := sin (ω * x + (π / 4)) + b

theorem compute_f_pi_over_2
  (ω b : ℝ) 
  (h1 : ω > 0)
  (T : ℝ) 
  (h2 : (2 * π / 3) < T ∧ T < π)
  (h3 : T = 2 * π / ω)
  (h4 : f (3 * π / 2) ω b = 2):
  f (π / 2) ω b = 1 :=
sorry

end compute_f_pi_over_2_l116_116978


namespace repeating_decimal_35_as_fraction_l116_116789

noncomputable def repeating_decimal_as_fraction (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ a b : ℕ, (0.\overline{35}).repr = n / d ∧ Int.gcd a b = 1 ∧ a + b = 134

theorem repeating_decimal_35_as_fraction :
  repeating_decimal_as_fraction 35 99 := sorry

end repeating_decimal_35_as_fraction_l116_116789


namespace primes_or_prime_squares_l116_116569

theorem primes_or_prime_squares (n : ℕ) (h1 : n > 1)
  (h2 : ∀ d, d ∣ n → d > 1 → (d - 1) ∣ (n - 1)) : 
  (∃ p, Nat.Prime p ∧ (n = p ∨ n = p * p)) :=
by
  sorry

end primes_or_prime_squares_l116_116569


namespace tile_calc_proof_l116_116827

noncomputable def total_tiles (length width : ℕ) : ℕ :=
  let border_tiles_length := (2 * (length - 4)) * 2
  let border_tiles_width := (2 * (width - 4)) * 2
  let total_border_tiles := (border_tiles_length + border_tiles_width) * 2 - 8
  let inner_length := (length - 4)
  let inner_width := (width - 4)
  let inner_area := inner_length * inner_width
  let inner_tiles := inner_area / 4
  total_border_tiles + inner_tiles

theorem tile_calc_proof :
  total_tiles 15 20 = 144 :=
by
  sorry

end tile_calc_proof_l116_116827


namespace snow_white_last_trip_l116_116640

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end snow_white_last_trip_l116_116640


namespace tony_fever_temperature_above_threshold_l116_116949

theorem tony_fever_temperature_above_threshold 
  (n : ℕ) (i : ℕ) (f : ℕ) 
  (h1 : n = 95) (h2 : i = 10) (h3 : f = 100) : 
  n + i - f = 5 :=
by
  sorry

end tony_fever_temperature_above_threshold_l116_116949


namespace cos_beta_zero_l116_116432

theorem cos_beta_zero (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) (h3 : Real.cos α = 1 / 2) (h4 : Real.cos (α + β) = -1 / 2) : Real.cos β = 0 :=
sorry

end cos_beta_zero_l116_116432


namespace sum_of_squares_of_real_solutions_l116_116734

theorem sum_of_squares_of_real_solutions :
  (∀ x : ℝ, |x^2 - 3 * x + 1 / 400| = 1 / 400)
  → ((0^2 : ℝ) + 3^2 + (9 - 1 / 100) = 999 / 100) := sorry

end sum_of_squares_of_real_solutions_l116_116734


namespace sufficient_and_necessary_condition_for_positive_sum_l116_116762

variable (q : ℤ) (a1 : ℤ)

def geometric_sequence (n : ℕ) : ℤ := a1 * q ^ (n - 1)

def sum_of_first_n_terms (n : ℕ) : ℤ :=
  if q = 1 then a1 * n else (a1 * (1 - q ^ n)) / (1 - q)

theorem sufficient_and_necessary_condition_for_positive_sum :
  (a1 > 0) ↔ (sum_of_first_n_terms q a1 2017 > 0) :=
sorry

end sufficient_and_necessary_condition_for_positive_sum_l116_116762


namespace julia_bill_ratio_l116_116828

-- Definitions
def saturday_miles_b (s_b : ℕ) (s_su : ℕ) := s_su = s_b + 4
def sunday_miles_j (s_su : ℕ) (t : ℕ) (s_j : ℕ) := s_j = t * s_su
def total_weekend_miles (s_b : ℕ) (s_su : ℕ) (s_j : ℕ) := s_b + s_su + s_j = 36

-- Proof statement
theorem julia_bill_ratio (s_b s_su s_j : ℕ) (h1 : saturday_miles_b s_b s_su) (h3 : total_weekend_miles s_b s_su s_j) (h_su : s_su = 10) : (2 * s_su = s_j) :=
by
  sorry  -- proof

end julia_bill_ratio_l116_116828


namespace part1_part2_l116_116303

variable (m x : ℝ)

-- Condition: mx - 3 > 2x + m
def inequality1 := m * x - 3 > 2 * x + m

-- Part (1) Condition: x < (m + 3) / (m - 2)
def solution_set_part1 := x < (m + 3) / (m - 2)

-- Part (2) Condition: 2x - 1 > 3 - x
def inequality2 := 2 * x - 1 > 3 - x

theorem part1 (h : ∀ x, inequality1 m x → solution_set_part1 m x) : m < 2 :=
sorry

theorem part2 (h1 : ∀ x, inequality1 m x ↔ inequality2 x) : m = 17 :=
sorry

end part1_part2_l116_116303


namespace hyungjun_initial_ribbon_length_l116_116589

noncomputable def initial_ribbon_length (R: ℝ) : Prop :=
  let used_for_first_box := R / 2 + 2000
  let remaining_after_first := R - used_for_first_box
  let used_for_second_box := (remaining_after_first / 2) + 2000
  remaining_after_first - used_for_second_box = 0

theorem hyungjun_initial_ribbon_length : ∃ R: ℝ, initial_ribbon_length R ∧ R = 12000 :=
  by
  exists 12000
  unfold initial_ribbon_length
  simp
  sorry

end hyungjun_initial_ribbon_length_l116_116589


namespace gcd_90_405_l116_116271

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l116_116271


namespace value_of_expression_l116_116409

theorem value_of_expression (a b c k : ℕ) (h_a : a = 30) (h_b : b = 25) (h_c : c = 4) (h_k : k = 3) : 
  (a - (b - k * c)) - ((a - k * b) - c) = 66 :=
by
  rw [h_a, h_b, h_c, h_k]
  simp
  sorry

end value_of_expression_l116_116409


namespace gecko_sales_ratio_l116_116895

theorem gecko_sales_ratio (x : ℕ) (h1 : 86 + x = 258) : 86 / Nat.gcd 172 86 = 1 ∧ 172 / Nat.gcd 172 86 = 2 := by
  sorry

end gecko_sales_ratio_l116_116895


namespace find_subtracted_value_l116_116244

theorem find_subtracted_value (N V : ℕ) (h1 : N = 1376) (h2 : N / 8 - V = 12) : V = 160 :=
by
  sorry

end find_subtracted_value_l116_116244


namespace min_value_of_expression_l116_116437

theorem min_value_of_expression (a b : ℝ) (h1 : 1 < a) (h2 : 0 < b) (h3 : a + 2 * b = 2) : 
  4 * (1 + Real.sqrt 2) ≤ (2 / (a - 1) + a / b) :=
by
  sorry

end min_value_of_expression_l116_116437


namespace selection_schemes_count_l116_116339

theorem selection_schemes_count :
  let total_teachers := 9
  let select_from_total := Nat.choose 9 3
  let select_all_male := Nat.choose 5 3
  let select_all_female := Nat.choose 4 3
  select_from_total - (select_all_male + select_all_female) = 420 := by
    sorry

end selection_schemes_count_l116_116339


namespace tins_per_case_is_24_l116_116714

def total_cases : ℕ := 15
def damaged_percentage : ℝ := 0.05
def remaining_tins : ℕ := 342

theorem tins_per_case_is_24 (x : ℕ) (h : (1 - damaged_percentage) * (total_cases * x) = remaining_tins) : x = 24 :=
  sorry

end tins_per_case_is_24_l116_116714


namespace circle_rolling_start_point_l116_116556

theorem circle_rolling_start_point (x : ℝ) (h1 : ∃ x, (x + 2 * Real.pi = -1) ∨ (x - 2 * Real.pi = -1)) :
  x = -1 - 2 * Real.pi ∨ x = -1 + 2 * Real.pi :=
by
  sorry

end circle_rolling_start_point_l116_116556


namespace correct_sum_is_1826_l116_116688

-- Define the four-digit number representation
def four_digit (A B C D : ℕ) := 1000 * A + 100 * B + 10 * C + D

-- Condition: Yoongi confused the units digit (9 as 6)
-- The incorrect number Yoongi used
def incorrect_number (A B C : ℕ) := four_digit A B C 6

-- The correct number
def correct_number (A B C : ℕ) := four_digit A B C 9

-- The sum obtained by Yoongi
def yoongi_sum (A B C : ℕ) := incorrect_number A B C + 57

-- The correct sum 
def correct_sum (A B C : ℕ) := correct_number A B C + 57

-- Condition: Yoongi's sum is 1823
axiom yoongi_sum_is_1823 (A B C: ℕ) : yoongi_sum A B C = 1823

-- Proof Problem: Prove that the correct sum is 1826
theorem correct_sum_is_1826 (A B C : ℕ) : correct_sum A B C = 1826 := by
  -- The proof goes here
  sorry

end correct_sum_is_1826_l116_116688


namespace which_two_students_donated_l116_116722

theorem which_two_students_donated (A B C D : Prop) 
  (h1 : A ∨ D) 
  (h2 : ¬(A ∧ D)) 
  (h3 : (A ∧ B) ∨ (A ∧ D) ∨ (B ∧ D))
  (h4 : ¬(A ∧ B ∧ D)) 
  : B ∧ D :=
sorry

end which_two_students_donated_l116_116722


namespace min_value_expr_l116_116919

open Real

theorem min_value_expr : ∃ x y : ℝ, 
  let expr := (sqrt (2 * (1 + cos (2 * x))) - sqrt (3 - sqrt 2) * sin x + 1) *
               (3 + 2 * sqrt (7 - sqrt 2) * cos y - cos (2 * y))
  in expr = -9 :=
by sorry

end min_value_expr_l116_116919


namespace problem_statement_l116_116667

variable {x y z : ℝ}

theorem problem_statement (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
    (h₁ : x^2 - y^2 = y * z) (h₂ : y^2 - z^2 = x * z) : 
    x^2 - z^2 = x * y := 
by
  sorry

end problem_statement_l116_116667


namespace choose_starters_l116_116706

theorem choose_starters :
  let totalPlayers := 16
  let numberOfTwins := 2
  let playersExcludingTwins := totalPlayers - numberOfTwins
  Nat.choose totalPlayers 6 - Nat.choose playersExcludingTwins 6 = 5005 :=
by
  let totalPlayers := 16
  let numberOfTwins := 2
  let playersExcludingTwins := totalPlayers - numberOfTwins
  sorry

end choose_starters_l116_116706


namespace apples_in_boxes_l116_116229

theorem apples_in_boxes (apples_per_box : ℕ) (number_of_boxes : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_box = 12) (h2 : number_of_boxes = 90) : total_apples = 1080 :=
by
  sorry

end apples_in_boxes_l116_116229


namespace sum_of_fraction_components_l116_116794

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l116_116794


namespace common_sum_of_4x4_matrix_l116_116347

open Matrix

-- Define the set of integers from -12 to 3 inclusive
def intSeq : List ℤ := List.range (3 - (-12) + 1) |>.map (λ x => x - 12)

def validMatrix (m : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  let rowsEqualSum := ∀ i, (Finset.univ : Finset (Fin 4)).sum (λ j => m i j) = -18
  let colsEqualSum := ∀ j, (Finset.univ : Finset (Fin 4)).sum (λ i => m i j) = -18
  let diag1Sum := (Finset.univ : Finset (Fin 4)).sum (λ k => m k k) = -18
  let diag2Sum := (Finset.univ : Finset (Fin 4)).sum (λ k => m k (Fin 3 - k)) = -18
  rowsEqualSum ∧ colsEqualSum ∧ diag1Sum ∧ diag2Sum

theorem common_sum_of_4x4_matrix : ∃ m : Matrix (Fin 4) (Fin 4) ℤ, validMatrix m ∧ (Matrix.toList m).permute (intSeq) :=
by 
  sorry

end common_sum_of_4x4_matrix_l116_116347


namespace system_infinite_solutions_a_eq_neg2_l116_116586

theorem system_infinite_solutions_a_eq_neg2 
  (x y a : ℝ)
  (h1 : 2 * x + 2 * y = -1)
  (h2 : 4 * x + a^2 * y = a) 
  (infinitely_many_solutions : ∃ (a : ℝ), ∀ (c : ℝ), 4 * x + a^2 * y = c) :
  a = -2 :=
by
  sorry

end system_infinite_solutions_a_eq_neg2_l116_116586


namespace probability_to_form_computers_l116_116055

def letters_in_campus : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def letters_in_threads : Finset Char := {'T', 'H', 'R', 'E', 'A', 'D', 'S'}
def letters_in_glow : Finset Char := {'G', 'L', 'O', 'W'}
def letters_in_computers : Finset Char := {'C', 'O', 'M', 'P', 'U', 'T', 'E', 'R', 'S'}

noncomputable def probability_campus : ℚ := 1 / Nat.choose 6 3
noncomputable def probability_threads : ℚ := 1 / Nat.choose 7 5
noncomputable def probability_glow : ℚ := 1 / (Nat.choose 4 2 / Nat.choose 3 1)

noncomputable def overall_probability : ℚ :=
  probability_campus * probability_threads * probability_glow

theorem probability_to_form_computers :
  overall_probability = 1 / 840 := by
  sorry

end probability_to_form_computers_l116_116055


namespace cars_on_river_road_l116_116654

theorem cars_on_river_road (B C : ℕ) (h_ratio : B / C = 1 / 3) (h_fewer : C = B + 40) : C = 60 :=
sorry

end cars_on_river_road_l116_116654


namespace probability_two_red_or_blue_correct_l116_116384

noncomputable def probability_two_red_or_blue_sequential : ℚ := 1 / 5

theorem probability_two_red_or_blue_correct :
  let total_marbles := 15
  let red_blue_marbles := 7
  let first_draw_prob := (7 : ℚ) / 15
  let second_draw_prob := (6 : ℚ) / 14
  first_draw_prob * second_draw_prob = probability_two_red_or_blue_sequential :=
by
  sorry

end probability_two_red_or_blue_correct_l116_116384


namespace largest_positive_integer_not_sum_of_multiple_30_and_composite_l116_116676

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

noncomputable def largest_not_sum_of_multiple_30_and_composite : ℕ :=
  211

theorem largest_positive_integer_not_sum_of_multiple_30_and_composite {m : ℕ} :
  m = largest_not_sum_of_multiple_30_and_composite ↔ 
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ k : ℕ, (b = k * 30) ∨ is_composite b) → m ≠ 30 * a + b) :=
sorry

end largest_positive_integer_not_sum_of_multiple_30_and_composite_l116_116676


namespace pirates_treasure_l116_116485

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l116_116485


namespace arithmetic_seq_sum_x_y_l116_116861

theorem arithmetic_seq_sum_x_y :
  ∃ (x y : ℕ), (∀ n : ℕ, n > 0 → a_n = 3 + (n - 1) * 5) ∧ x + 33 = 33 ∧ x = 28 → x + y = 61 :=
by
  sorry

end arithmetic_seq_sum_x_y_l116_116861


namespace fourth_derivative_l116_116575

noncomputable def f (x : ℝ) : ℝ := (5 * x - 8) * 2^(-x)

theorem fourth_derivative (x : ℝ) : 
  deriv (deriv (deriv (deriv f))) x = 2^(-x) * (Real.log 2)^4 * (5 * x - 9) :=
sorry

end fourth_derivative_l116_116575


namespace largest_integer_not_sum_of_30_and_composite_l116_116673

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def is_not_sum_of_30_and_composite (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ b < 30 ∧ is_composite(b) ∧ n = 30 * a + b

theorem largest_integer_not_sum_of_30_and_composite :
  ∃ n : ℕ, is_not_sum_of_30_and_composite(n) ∧ ∀ m : ℕ, is_not_sum_of_30_and_composite(m) → m ≤ n :=
  ⟨93, sorry⟩

end largest_integer_not_sum_of_30_and_composite_l116_116673


namespace treasure_coins_l116_116506

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l116_116506


namespace combination_equality_l116_116541

theorem combination_equality : 
  Nat.choose 5 2 + Nat.choose 5 3 = 20 := 
by 
  sorry

end combination_equality_l116_116541


namespace quad_side_difference_l116_116709

theorem quad_side_difference (a b c d s x y : ℝ)
  (h1 : a = 80) (h2 : b = 100) (h3 : c = 150) (h4 : d = 120)
  (semiperimeter : s = (a + b + c + d) / 2)
  (h5 : x + y = c) 
  (h6 : (|x - y| = 30)) : 
  |x - y| = 30 :=
sorry

end quad_side_difference_l116_116709


namespace num_values_satisfying_g_g_x_eq_4_l116_116346

def g (x : ℝ) : ℝ := sorry

theorem num_values_satisfying_g_g_x_eq_4 
  (h1 : g (-2) = 4)
  (h2 : g (2) = 4)
  (h3 : g (4) = 4)
  (h4 : ∀ x, g (x) ≠ -2)
  (h5 : ∃! x, g (x) = 2) 
  (h6 : ∃! x, g (x) = 4) 
  : ∃! x1 x2, g (g x1) = 4 ∧ g (g x2) = 4 ∧ x1 ≠ x2 :=
by
  sorry

end num_values_satisfying_g_g_x_eq_4_l116_116346


namespace find_A_from_complement_l116_116772

-- Define the universal set U
def U : Set ℕ := {0, 1, 2}

-- Define the complement of set A in U
variable (A : Set ℕ)
def complement_U_A : Set ℕ := {n | n ∈ U ∧ n ∉ A}

-- Define the condition given in the problem
axiom h : complement_U_A A = {2}

-- State the theorem to be proven
theorem find_A_from_complement : A = {0, 1} :=
sorry

end find_A_from_complement_l116_116772


namespace balls_is_perfect_square_l116_116851

open Classical -- Open classical logic for nonconstructive proofs

-- Define a noncomputable function to capture the main proof argument
noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem balls_is_perfect_square {a v : ℕ} (h : (2 * a * v) = (a + v) * (a + v - 1))
  : is_perfect_square (a + v) :=
sorry

end balls_is_perfect_square_l116_116851


namespace combined_rate_mpg_900_over_41_l116_116000

-- Declare the variables and conditions
variables {d : ℕ} (h_d_pos : d > 0)

def combined_mpg (d : ℕ) : ℚ :=
  let anna_car_gasoline := (d : ℚ) / 50
  let ben_car_gasoline  := (d : ℚ) / 20
  let carl_car_gasoline := (d : ℚ) / 15
  let total_gasoline    := anna_car_gasoline + ben_car_gasoline + carl_car_gasoline
  ((3 : ℚ) * d) / total_gasoline

-- Define the theorem statement
theorem combined_rate_mpg_900_over_41 :
  ∀ d : ℕ, d > 0 → combined_mpg d = 900 / 41 :=
by
  intros d h_d_pos
  rw [combined_mpg]
  -- Steps following the solution
  sorry -- proof omitted

end combined_rate_mpg_900_over_41_l116_116000


namespace parallel_lines_iff_determinant_zero_l116_116937

theorem parallel_lines_iff_determinant_zero (a1 b1 c1 a2 b2 c2 : ℝ) :
  (a1 * b2 - a2 * b1 = 0) ↔ ((a1 * c2 - a2 * c1 = 0) → (b1 * c2 - b2 * c1 = 0)) := 
sorry

end parallel_lines_iff_determinant_zero_l116_116937


namespace shaded_area_percentage_l116_116531

theorem shaded_area_percentage (total_area shaded_area : ℕ) (h_total : total_area = 49) (h_shaded : shaded_area = 33) : 
  (shaded_area : ℚ) / total_area = 33 / 49 := 
by
  sorry

end shaded_area_percentage_l116_116531


namespace yellow_less_than_three_times_red_l116_116379

def num_red : ℕ := 40
def less_than_three_times (Y : ℕ) : Prop := Y < 120
def blue_half_yellow (Y B : ℕ) : Prop := B = Y / 2
def remaining_after_carlos (B : ℕ) : Prop := 40 + B = 90
def difference_three_times_red (Y : ℕ) : ℕ := 3 * num_red - Y

theorem yellow_less_than_three_times_red (Y B : ℕ) 
  (h1 : less_than_three_times Y) 
  (h2 : blue_half_yellow Y B) 
  (h3 : remaining_after_carlos B) : 
  difference_three_times_red Y = 20 := by
  sorry

end yellow_less_than_three_times_red_l116_116379


namespace slower_train_speed_l116_116856

-- Defining the conditions

def length_of_each_train := 80 -- in meters
def faster_train_speed := 52 -- in km/hr
def time_to_pass := 36 -- in seconds

-- Main statement: 
theorem slower_train_speed (v : ℝ) : 
    let relative_speed := (faster_train_speed - v) * (1000 / 3600) -- converting relative speed from km/hr to m/s
    let total_distance := 2 * length_of_each_train
    let speed_equals_distance_over_time := total_distance / time_to_pass 
    (relative_speed = speed_equals_distance_over_time) -> v = 36 :=
by
  intros
  sorry

end slower_train_speed_l116_116856


namespace p_is_sufficient_but_not_necessary_for_q_l116_116820

variable (x : ℝ)

def p := x > 1
def q := x > 0

theorem p_is_sufficient_but_not_necessary_for_q : (p x → q x) ∧ ¬(q x → p x) := by
  sorry

end p_is_sufficient_but_not_necessary_for_q_l116_116820


namespace repeated_decimal_to_fraction_l116_116780

theorem repeated_decimal_to_fraction :
  ∀ (a b : ℕ), (0.35.recursor = 35 / 99 ∧ Nat.gcd 35 99 = 1) → a + b = 134 := by
  sorry

end repeated_decimal_to_fraction_l116_116780


namespace simplify_expression_l116_116181

theorem simplify_expression (x : ℤ) : (3 * x) ^ 3 + (2 * x) * (x ^ 4) = 27 * x ^ 3 + 2 * x ^ 5 :=
by sorry

end simplify_expression_l116_116181


namespace treasure_coins_l116_116503

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l116_116503


namespace compute_fraction_power_l116_116009

theorem compute_fraction_power :
  8 * (2 / 7)^4 = 128 / 2401 :=
by
  sorry

end compute_fraction_power_l116_116009


namespace compute_fraction_power_l116_116902

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end compute_fraction_power_l116_116902


namespace area_difference_l116_116476

-- Define the original and new rectangle dimensions
def original_rect_area (length width : ℕ) : ℕ := length * width
def new_rect_area (length width : ℕ) : ℕ := (length - 2) * (width + 2)

-- Define the problem statement
theorem area_difference (a : ℕ) : new_rect_area a 5 - original_rect_area a 5 = 2 * a - 14 :=
by
  -- Insert proof here
  sorry

end area_difference_l116_116476


namespace non_similar_triangles_with_arithmetic_angles_l116_116777

theorem non_similar_triangles_with_arithmetic_angles : 
  ∃! (d : ℕ), d > 0 ∧ d ≤ 50 := 
sorry

end non_similar_triangles_with_arithmetic_angles_l116_116777


namespace first_group_people_count_l116_116386

def group_ice_cream (P : ℕ) : Prop :=
  let total_days_per_person1 := P * 10
  let total_days_per_person2 := 5 * 16
  total_days_per_person1 = total_days_per_person2

theorem first_group_people_count 
  (P : ℕ) 
  (H1 : group_ice_cream P) : 
  P = 8 := 
sorry

end first_group_people_count_l116_116386


namespace duration_of_period_l116_116888

/-- The duration of the period at which B gains Rs. 1125 by lending 
Rs. 25000 at rate of 11.5% per annum and borrowing the same 
amount at 10% per annum -/
theorem duration_of_period (principal : ℝ) (rate_borrow : ℝ) (rate_lend : ℝ) (gain : ℝ) : 
  ∃ (t : ℝ), principal = 25000 ∧ rate_borrow = 0.10 ∧ rate_lend = 0.115 ∧ gain = 1125 → 
  t = 3 :=
by
  sorry

end duration_of_period_l116_116888


namespace polynomial_roots_l116_116914

-- Problem statement: prove that the roots of the given polynomial are {-1, 3, 3}
theorem polynomial_roots : 
  (λ x => x^3 - 5 * x^2 + 3 * x + 9) = (λ x => (x + 1) * (x - 3) ^ 2) :=
by
  sorry

end polynomial_roots_l116_116914


namespace pirate_treasure_l116_116508

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l116_116508


namespace vodka_shot_size_l116_116164

theorem vodka_shot_size (x : ℝ) (h1 : 8 / 2 = 4) (h2 : 4 * x = 2 * 3) : x = 1.5 :=
by
  sorry

end vodka_shot_size_l116_116164


namespace sum_of_squares_of_two_numbers_l116_116355

theorem sum_of_squares_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 30) :
  x^2 + y^2 = 840 :=
by
  sorry

end sum_of_squares_of_two_numbers_l116_116355


namespace perpendicular_lines_l116_116152

theorem perpendicular_lines {a : ℝ} :
  a*(a-1) + (1-a)*(2*a+3) = 0 → (a = 1 ∨ a = -3) := 
by
  intro h
  sorry

end perpendicular_lines_l116_116152


namespace coffee_customers_l116_116726

theorem coffee_customers (C : ℕ) :
  let coffee_cost := 5
  let tea_ordered := 8
  let tea_cost := 4
  let total_revenue := 67
  (coffee_cost * C + tea_ordered * tea_cost = total_revenue) → C = 7 := by
  sorry

end coffee_customers_l116_116726


namespace gcd_90_405_l116_116268

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l116_116268


namespace length_segment_pq_l116_116177

theorem length_segment_pq 
  (P Q R S T : ℝ)
  (h1 : (dist P Q + dist P R + dist P S + dist P T = 67))
  (h2 : (dist Q P + dist Q R + dist Q S + dist Q T = 34)) :
  dist P Q = 11 :=
sorry

end length_segment_pq_l116_116177


namespace dorothy_needs_more_money_l116_116120

structure Person :=
  (age : ℕ)

def Discount (age : ℕ) : ℝ :=
  if age <= 11 then 0.5 else
  if age >= 65 then 0.8 else
  if 12 <= age && age <= 18 then 0.7 else 1.0

def ticketCost (age : ℕ) : ℝ :=
  (10 : ℝ) * Discount age

def specialExhibitCost : ℝ := 5

def totalCost (family : List Person) : ℝ :=
  (family.map (λ p => ticketCost p.age + specialExhibitCost)).sum

def salesTaxRate : ℝ := 0.1

def finalCost (family : List Person) : ℝ :=
  let total := totalCost family
  total + (total * salesTaxRate)

def dorothy_money_after_trip (dorothy_money : ℝ) (family : List Person) : ℝ :=
  dorothy_money - finalCost family

theorem dorothy_needs_more_money :
  dorothy_money_after_trip 70 [⟨15⟩, ⟨10⟩, ⟨40⟩, ⟨42⟩, ⟨65⟩] = -1.5 := by
  sorry

end dorothy_needs_more_money_l116_116120


namespace runner_time_l116_116356

-- Assumptions for the problem
variables (meet1 meet2 meet3 : ℕ) -- Times at which the runners meet

-- Given conditions per the problem
def conditions := (meet1 = 15 ∧ meet2 = 25)

-- Final statement proving the time taken to run the entire track
theorem runner_time (meet1 meet2 meet3 : ℕ) (h1 : meet1 = 15) (h2 : meet2 = 25) : 
  let total_time := 2 * meet1 + 2 * meet2 in
  total_time = 80 :=
by {
  sorry
}

end runner_time_l116_116356


namespace garbage_collection_l116_116013

theorem garbage_collection (Daliah Dewei Zane : ℝ) 
(h1 : Daliah = 17.5)
(h2 : Dewei = Daliah - 2)
(h3 : Zane = 4 * Dewei) :
Zane = 62 :=
sorry

end garbage_collection_l116_116013


namespace real_solution_count_l116_116921

theorem real_solution_count : 
  ∃ (n : ℕ), n = 1 ∧
    ∀ x : ℝ, 
      (3 * x / (x ^ 2 + 2 * x + 4) + 4 * x / (x ^ 2 - 4 * x + 4) = 1) ↔ (x = 2) :=
by
  sorry

end real_solution_count_l116_116921


namespace number_of_chords_l116_116926

theorem number_of_chords : (Nat.choose 10 3 + Nat.choose 10 4 + Nat.choose 10 5 + Nat.choose 10 6 + Nat.choose 10 7 + Nat.choose 10 8 + Nat.choose 10 9 + Nat.choose 10 10) = 968 :=
by
  sorry

end number_of_chords_l116_116926


namespace alternate_seating_boys_l116_116320

theorem alternate_seating_boys (B : ℕ) (girl : ℕ) (ways : ℕ)
  (h1 : girl = 1)
  (h2 : ways = 24)
  (h3 : ways = B - 1) :
  B = 25 :=
sorry

end alternate_seating_boys_l116_116320


namespace interval_monotonicity_no_zeros_min_a_l116_116299

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

theorem interval_monotonicity (a : ℝ) :
  a = 1 →
  (∀ x, 0 < x ∧ x ≤ 2 → f a x < f a (x+1)) ∧
  (∀ x, x ≥ 2 → f a x < f a (x-1)) :=
by
  sorry

theorem no_zeros_min_a : 
  (∀ x, x ∈ Set.Ioo 0 (1/2 : ℝ) → f a x ≠ 0) →
  a ≥ 2 - 4 * Real.log 2 :=
by
  sorry

end interval_monotonicity_no_zeros_min_a_l116_116299


namespace coordinates_of_C_l116_116832

theorem coordinates_of_C (A B : ℝ × ℝ) (hA : A = (-2, -1)) (hB : B = (4, 9)) :
    ∃ C : ℝ × ℝ, (dist C A) = 4 * dist C B ∧ C = (-0.8, 1) :=
sorry

end coordinates_of_C_l116_116832


namespace ellipse_to_parabola_standard_eq_l116_116769

theorem ellipse_to_parabola_standard_eq :
  ∀ (x y : ℝ), (x^2 / 25 + y^2 / 16 = 1) → (y^2 = 12 * x) :=
by
  sorry

end ellipse_to_parabola_standard_eq_l116_116769


namespace number_of_zeros_of_g_is_zero_l116_116579

noncomputable def f (x : ℝ) : ℝ := sorry

theorem number_of_zeros_of_g_is_zero (h1 : ∀ x, Continuous (f x))
    (h2 : ∀ x, Differentiable ℝ (f x))
    (h3 : ∀ x : ℝ, 0 < x → x * (deriv (deriv f)) x + f x > 0) :
    ∀ x : ℝ, 0 < x → x * f x + 1 ≠ 0 :=
begin
    intros x hx,
    sorry
end

end number_of_zeros_of_g_is_zero_l116_116579


namespace roots_separation_condition_l116_116580

theorem roots_separation_condition (m n p q : ℝ)
  (h_1 : ∃ (x1 x2 : ℝ), x1 + x2 = -m ∧ x1 * x2 = n ∧ x1 ≠ x2)
  (h_2 : ∃ (x3 x4 : ℝ), x3 + x4 = -p ∧ x3 * x4 = q ∧ x3 ≠ x4)
  (h_3 : (∀ x1 x2 x3 x4 : ℝ, x1 + x2 = -m ∧ x1 * x2 = n ∧ x3 + x4 = -p ∧ x3 * x4 = q → 
         (x3 - x1) * (x3 - x2) * (x4 - x1) * (x4 - x2) < 0)) : 
  (n - q)^2 + (m - p) * (m * q - n * p) < 0 :=
sorry

end roots_separation_condition_l116_116580


namespace snow_white_last_trip_l116_116642

universe u

-- Define the dwarf names as an enumerated type
inductive Dwarf : Type
| Happy | Grumpy | Dopey | Bashful | Sleepy | Doc | Sneezy
deriving DecidableEq, Repr

open Dwarf

-- Define conditions
def initial_lineup : List Dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- Define the condition of adjacency
def adjacent (d1 d2 : Dwarf) : Prop :=
  List.pairwise_adjacent (· = d2) initial_lineup d1

-- Define the boat capacity
def boat_capacity : Fin 4 := by sorry

-- Snow White is the only one who can row
def snow_white_rows : Prop := true

-- No quarrels condition
def no_quarrel_without_snow_white (group : List Dwarf) : Prop :=
  ∀ d1 d2, d1 ∈ group → d2 ∈ group → ¬ adjacent d1 d2

-- Objective: Transfer all dwarfs without quarrels
theorem snow_white_last_trip (trips : List (List Dwarf)) :
  ∃ last_trip : List Dwarf,
    last_trip = [Grumpy, Bashful, Doc] ∧
    no_quarrel_without_snow_white (initial_lineup.diff (trips.join ++ last_trip)) :=
sorry

end snow_white_last_trip_l116_116642


namespace side_length_of_square_l116_116464

theorem side_length_of_square (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (a b c : ℝ) (h_leg1 : a = 12) (h_leg2 : b = 9) (h_right : c^2 = a^2 + b^2) :
  ∃ s : ℝ, s = 45/8 :=
by 
  -- Given the right triangle with legs 12 cm and 9 cm, the length of the side of the square is 45/8 cm
  let s := 45/8
  use s
  sorry

end side_length_of_square_l116_116464


namespace bills_fraction_l116_116231
-- Lean 4 code

theorem bills_fraction (total_stickers : ℕ) (andrews_fraction : ℚ) (total_given_away : ℕ)
  (andrews_stickers : ℕ) (remaining_stickers : ℕ)
  (bills_stickers : ℕ) :
  total_stickers = 100 →
  andrews_fraction = 1/5 →
  andrews_stickers = 1/5 * 100 →
  total_given_away = 44 →
  andrews_stickers = 20 →
  remaining_stickers = total_stickers - andrews_stickers →
  bills_stickers = total_given_away - andrews_stickers →
  bills_stickers = 24 →
  bills_stickers / remaining_stickers = 3 / 10 :=
begin
  sorry
end

end bills_fraction_l116_116231


namespace point_b_not_inside_circle_a_l116_116830

theorem point_b_not_inside_circle_a (a : ℝ) : a < 5 → ¬ (1 < a ∧ a < 5) :=
by
  sorry

end point_b_not_inside_circle_a_l116_116830


namespace chris_money_left_l116_116258

def video_game_cost : ℕ := 60
def candy_cost : ℕ := 5
def babysitting_rate : ℕ := 8
def hours_worked : ℕ := 9
def earnings : ℕ := babysitting_rate * hours_worked
def total_cost : ℕ := video_game_cost + candy_cost
def money_left : ℕ := earnings - total_cost

theorem chris_money_left
  (h1 : video_game_cost = 60)
  (h2 : candy_cost = 5)
  (h3 : babysitting_rate = 8)
  (h4 : hours_worked = 9) :
  money_left = 7 :=
by
  -- The detailed proof is omitted.
  sorry

end chris_money_left_l116_116258


namespace periodic_even_l116_116017

noncomputable def f : ℝ → ℝ := sorry  -- We assume the existence of such a function.

variables {α β : ℝ}  -- acute angles of a right triangle

-- Function properties
theorem periodic_even (h_periodic: ∀ x: ℝ, f (x + 2) = f x)
  (h_even: ∀ x: ℝ, f (-x) = f x)
  (h_decreasing: ∀ x y: ℝ, -3 ≤ x ∧ x < y ∧ y ≤ -2 → f x > f y)
  (h_inc_interval_0_1: ∀ x y: ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y)
  (ha: 0 < α ∧ α < π / 2)
  (hb: 0 < β ∧ β < π / 2)
  (h_sum_right_triangle: α + β = π / 2): f (Real.sin α) > f (Real.cos β) :=
sorry

end periodic_even_l116_116017


namespace tan_2theta_l116_116771

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.cos x

theorem tan_2theta (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.tan (2 * θ) = -4 / 3 := 
by 
  sorry

end tan_2theta_l116_116771


namespace cost_of_softball_l116_116480

theorem cost_of_softball 
  (original_budget : ℕ)
  (dodgeball_cost : ℕ)
  (num_dodgeballs : ℕ)
  (increase_rate : ℚ)
  (num_softballs : ℕ)
  (new_budget : ℕ)
  (softball_cost : ℕ)
  (h0 : original_budget = num_dodgeballs * dodgeball_cost)
  (h1 : increase_rate = 0.20)
  (h2 : new_budget = original_budget + increase_rate * original_budget)
  (h3 : new_budget = num_softballs * softball_cost) :
  softball_cost = 9 :=
by
  sorry

end cost_of_softball_l116_116480


namespace gcd_of_90_and_405_l116_116274

def gcd_90_405 : ℕ := Nat.gcd 90 405

theorem gcd_of_90_and_405 : gcd_90_405 = 45 :=
by
  -- proof goes here
  sorry

end gcd_of_90_and_405_l116_116274


namespace compute_fraction_power_l116_116903

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end compute_fraction_power_l116_116903


namespace triangle_sides_inequality_l116_116044

theorem triangle_sides_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
    (a/(b + c - a) + b/(c + a - b) + c/(a + b - c)) ≥ ((b + c - a)/a + (c + a - b)/b + (a + b - c)/c) ∧
    ((b + c - a)/a + (c + a - b)/b + (a + b - c)/c) ≥ 3 :=
by
  sorry

end triangle_sides_inequality_l116_116044


namespace max_a3_in_arith_geo_sequences_l116_116425

theorem max_a3_in_arith_geo_sequences
  (a1 a2 a3 : ℝ) (b1 b2 b3 : ℝ)
  (h1 : a1 + a2 + a3 = 15)
  (h2 : a2 = ((a1 + a3) / 2))
  (h3 : b1 * b2 * b3 = 27)
  (h4 : (a1 + b1) * (a3 + b3) = (a2 + b2) ^ 2)
  (h5 : a1 + b1 > 0)
  (h6 : a2 + b2 > 0)
  (h7 : a3 + b3 > 0) :
  a3 ≤ 59 := sorry

end max_a3_in_arith_geo_sequences_l116_116425


namespace necessary_but_not_sufficient_condition_l116_116324

theorem necessary_but_not_sufficient_condition (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 1 → a n ^ 2 = a (n - 1) * a (n + 1)) →
  ¬ (∀ n : ℕ, n > 0 → ∃ r : ℝ, a n = a 1 * r ^ (n - 1)) :=
sorry

end necessary_but_not_sufficient_condition_l116_116324


namespace probability_of_triangle_segments_from_15gon_l116_116105

/-- A proof problem that calculates the probability that three randomly selected segments 
    from a regular 15-gon inscribed in a circle form a triangle with positive area. -/
theorem probability_of_triangle_segments_from_15gon : 
  let n := 15
  let total_segments := (n * (n - 1)) / 2 
  let total_combinations := total_segments * (total_segments - 1) * (total_segments - 2) / 6 
  let valid_probability := 943 / 1365
  valid_probability = (total_combinations - count_violating_combinations) / total_combinations :=
sorry

end probability_of_triangle_segments_from_15gon_l116_116105


namespace factorization_of_polynomial_l116_116414

theorem factorization_of_polynomial (x : ℝ) : 12 * x^2 - 40 * x + 25 = (2 * Real.sqrt 3 * x - 5)^2 :=
  sorry

end factorization_of_polynomial_l116_116414


namespace smallest_positive_integer_l116_116221

theorem smallest_positive_integer (a : ℤ)
  (h1 : a % 2 = 1)
  (h2 : a % 3 = 2)
  (h3 : a % 4 = 3)
  (h4 : a % 5 = 4) :
  a = 59 :=
sorry

end smallest_positive_integer_l116_116221


namespace total_handshakes_l116_116853

theorem total_handshakes (gremlins imps unfriendly_gremlins : ℕ) 
    (handshakes_among_friendly : ℕ) (handshakes_friendly_with_unfriendly : ℕ) 
    (handshakes_between_imps_and_gremlins : ℕ) 
    (h_friendly : gremlins = 30) (h_imps : imps = 20) 
    (h_unfriendly : unfriendly_gremlins = 10) 
    (h_handshakes_among_friendly : handshakes_among_friendly = 190) 
    (h_handshakes_friendly_with_unfriendly : handshakes_friendly_with_unfriendly = 200)
    (h_handshakes_between_imps_and_gremlins : handshakes_between_imps_and_gremlins = 600) : 
    handshakes_among_friendly + handshakes_friendly_with_unfriendly + handshakes_between_imps_and_gremlins = 990 := 
by 
    sorry

end total_handshakes_l116_116853


namespace fuel_ethanol_problem_l116_116396

theorem fuel_ethanol_problem (x : ℝ) (h : 0.12 * x + 0.16 * (200 - x) = 28) : x = 100 := 
by
  sorry

end fuel_ethanol_problem_l116_116396


namespace coneCannotBeQuadrilateral_l116_116561

-- Define types for our geometric solids
inductive Solid
| Cylinder
| Cone
| FrustumCone
| Prism

-- Define a predicate for whether the cross-section can be a quadrilateral
def canBeQuadrilateral (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => true
  | Solid.Cone => false
  | Solid.FrustumCone => true
  | Solid.Prism => true

-- The theorem we need to prove
theorem coneCannotBeQuadrilateral : canBeQuadrilateral Solid.Cone = false := by
  sorry

end coneCannotBeQuadrilateral_l116_116561


namespace one_third_eleven_y_plus_three_l116_116150

theorem one_third_eleven_y_plus_three (y : ℝ) : 
  (1/3) * (11 * y + 3) = 11 * y / 3 + 1 :=
by
  sorry

end one_third_eleven_y_plus_three_l116_116150


namespace max_area_of_triangle_ABC_l116_116812

noncomputable def max_area_triangle_ABC: ℝ :=
  let QA := 3
  let QB := 4
  let QC := 5
  let BC := 6
  -- Given these conditions, prove the maximum area of triangle ABC
  19

theorem max_area_of_triangle_ABC 
  (QA QB QC BC : ℝ) 
  (h1 : QA = 3) 
  (h2 : QB = 4) 
  (h3 : QC = 5) 
  (h4 : BC = 6) 
  (h5 : QB * QB + BC * BC = QC * QC) -- The right angle condition at Q
  : max_area_triangle_ABC = 19 :=
by sorry

end max_area_of_triangle_ABC_l116_116812


namespace cube_side_length_l116_116193

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (h0 : s ≠ 0) : s = 6 :=
sorry

end cube_side_length_l116_116193


namespace no_empty_boxes_prob_l116_116630

def P (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem no_empty_boxes_prob :
  let num_balls := 3
  let num_boxes := 3
  let total_outcomes := num_boxes ^ num_balls
  let favorable_outcomes := P num_balls num_boxes
  let probability := favorable_outcomes / total_outcomes
  probability = 2 / 9 :=
by
  sorry

end no_empty_boxes_prob_l116_116630


namespace solve_inequality_l116_116182

theorem solve_inequality (k x : ℝ) :
  (x^2 > (k + 1) * x - k) ↔ 
  (if k > 1 then (x < 1 ∨ x > k)
   else if k = 1 then (x ≠ 1)
   else (x < k ∨ x > 1)) :=
by
  sorry

end solve_inequality_l116_116182


namespace no_n_such_that_n_times_s_is_20222022_l116_116621

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem
theorem no_n_such_that_n_times_s_is_20222022 :
  ∀ n : ℕ, n * sum_of_digits n ≠ 20222022 :=
by
  sorry

end no_n_such_that_n_times_s_is_20222022_l116_116621


namespace nancy_carrots_l116_116992

def carrots_total 
  (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

theorem nancy_carrots : 
  carrots_total 12 2 21 = 31 :=
by
  -- Add the proof here
  sorry

end nancy_carrots_l116_116992


namespace final_price_of_coat_is_correct_l116_116387

-- Define the conditions as constants
def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

-- Define the discounted amount calculation
def discount_amount : ℝ := original_price * discount_rate

-- Define the sale price after the discount
def sale_price : ℝ := original_price - discount_amount

-- Define the tax amount calculation on the sale price
def tax_amount : ℝ := sale_price * tax_rate

-- Define the total selling price
def total_selling_price : ℝ := sale_price + tax_amount

-- The theorem that needs to be proven
theorem final_price_of_coat_is_correct : total_selling_price = 96.6 :=
by
  sorry

end final_price_of_coat_is_correct_l116_116387


namespace _l116_116119

noncomputable theorem distinct_pos_numbers_sum_to_22
  (a b c : ℝ) 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a ≠ b)
  (h5 : b ≠ c)
  (h6 : a ≠ c)
  (eq1 : a^2 + b * c = 115)
  (eq2 : b^2 + a * c = 127)
  (eq3 : c^2 + a * b = 115) :
  a + b + c = 22 :=
  sorry

end _l116_116119


namespace Tony_fever_l116_116945

theorem Tony_fever :
  ∀ (normal_temp sickness_increase fever_threshold : ℕ),
    normal_temp = 95 →
    sickness_increase = 10 →
    fever_threshold = 100 →
    (normal_temp + sickness_increase) - fever_threshold = 5 :=
by
  intros normal_temp sickness_increase fever_threshold h1 h2 h3
  sorry

end Tony_fever_l116_116945


namespace altered_solution_detergent_volume_l116_116381

-- Definitions from conditions
def original_ratio_bleach : ℚ := 2
def original_ratio_detergent : ℚ := 40
def original_ratio_water : ℚ := 100

def altered_ratio_bleach : ℚ := original_ratio_bleach * 3
def altered_ratio_detergent : ℚ := original_ratio_detergent
def altered_ratio_water : ℚ := original_ratio_water / 2

def altered_total_water : ℚ := 300
def ratio_parts_per_liter : ℚ := altered_total_water / altered_ratio_water

-- Statement to prove
theorem altered_solution_detergent_volume : 
  altered_ratio_detergent * ratio_parts_per_liter = 60 := 
by 
  -- Here you'd write the proof, but we skip it as per instructions
  sorry

end altered_solution_detergent_volume_l116_116381


namespace correct_calculation_l116_116533

-- Define the variables used in the problem
variables (a x y : ℝ)

-- The main theorem statement
theorem correct_calculation : (2 * x * y^2 - x * y^2 = x * y^2) :=
by sorry

end correct_calculation_l116_116533


namespace number_of_five_dollar_bills_l116_116629

theorem number_of_five_dollar_bills (total_money denomination expected_bills : ℕ) 
  (h1 : total_money = 45) 
  (h2 : denomination = 5) 
  (h3 : expected_bills = total_money / denomination) : 
  expected_bills = 9 :=
by
  sorry

end number_of_five_dollar_bills_l116_116629


namespace find_minimum_value_M_l116_116121

theorem find_minimum_value_M : (∃ (M : ℝ), (∀ (x : ℝ), -x^2 + 2 * x ≤ M) ∧ M = 1) := 
sorry

end find_minimum_value_M_l116_116121


namespace gcd_xyx_xyz_square_of_nat_l116_116987

theorem gcd_xyx_xyz_square_of_nat 
  (x y z : ℕ)
  (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) :
  ∃ n : ℕ, (Nat.gcd x (Nat.gcd y z)) * x * y * z = n ^ 2 :=
by
  sorry

end gcd_xyx_xyz_square_of_nat_l116_116987


namespace find_a_l116_116752

open Nat

-- Define the conditions and the proof goal
theorem find_a (a b : ℕ) (h1 : 2019 = a^2 - b^2) (h2 : a < 1000) : a = 338 :=
sorry

end find_a_l116_116752


namespace division_addition_l116_116403

theorem division_addition :
  (-150 + 50) / (-50) = 2 := by
  sorry

end division_addition_l116_116403


namespace tony_fever_temperature_above_threshold_l116_116950

theorem tony_fever_temperature_above_threshold 
  (n : ℕ) (i : ℕ) (f : ℕ) 
  (h1 : n = 95) (h2 : i = 10) (h3 : f = 100) : 
  n + i - f = 5 :=
by
  sorry

end tony_fever_temperature_above_threshold_l116_116950


namespace abs_diff_kth_power_l116_116281

theorem abs_diff_kth_power (k : ℕ) (a b : ℤ) (x y : ℤ)
  (hk : 2 ≤ k)
  (ha : a ≠ 0) (hb : b ≠ 0)
  (hab_odd : (a + b) % 2 = 1)
  (hxy : 0 < |x - y| ∧ |x - y| ≤ 2)
  (h_eq : a^k * x - b^k * y = a - b) :
  ∃ m : ℤ, |a - b| = m^k :=
sorry

end abs_diff_kth_power_l116_116281


namespace no_positive_sequence_exists_l116_116019

theorem no_positive_sequence_exists:
  ¬ (∃ (b : ℕ → ℝ), (∀ n, b n > 0) ∧ (∀ m : ℕ, (∑' k, b ((k + 1) * m)) = (1 / m))) :=
by
  sorry

end no_positive_sequence_exists_l116_116019


namespace length_of_bridge_l116_116716

theorem length_of_bridge
  (train_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_kmph : ℝ)
  (conversion_factor : ℝ)
  (bridge_length : ℝ) :
  train_length = 100 →
  crossing_time = 12 →
  train_speed_kmph = 120 →
  conversion_factor = 1 / 3.6 →
  bridge_length = 299.96 :=
by
  sorry

end length_of_bridge_l116_116716


namespace evaluate_expression_l116_116413

theorem evaluate_expression (x : ℝ) : (1 - x^2) * (1 + x^4) = 1 - x^2 + x^4 - x^6 :=
by
  sorry

end evaluate_expression_l116_116413


namespace correlation_coefficient_value_relation_between_gender_and_electric_car_expectation_X_value_l116_116253

-- Definition 1: Variance and regression coefficients and correlation coefficient calculation
noncomputable def correlation_coefficient : ℝ := 4.7 * (Real.sqrt (2 / 50))

-- Theorem 1: Correlation coefficient computation
theorem correlation_coefficient_value :
  correlation_coefficient = 0.94 :=
sorry

-- Definition 2: Chi-square calculation for independence test
noncomputable def chi_square : ℝ :=
  (100 * ((30 * 35 - 20 * 15)^2 : ℝ)) / (50 * 50 * 45 * 55)

-- Theorem 2: Chi-square test result
theorem relation_between_gender_and_electric_car :
  chi_square > 6.635 :=
sorry

-- Definition 3: Probability distribution and expectation calculation
def probability_distribution : Finset ℚ :=
{(21/55), (28/55), (6/55)}

noncomputable def expectation_X : ℚ :=
(0 * (21/55) + 1 * (28/55) + 2 * (6/55))

-- Theorem 3: Expectation of X calculation
theorem expectation_X_value :
  expectation_X = 8/11 :=
sorry

end correlation_coefficient_value_relation_between_gender_and_electric_car_expectation_X_value_l116_116253


namespace frenchwoman_present_l116_116564

theorem frenchwoman_present
    (M_F M_R W_R : ℝ)
    (condition_1 : M_F > M_R + W_R)
    (condition_2 : W_R > M_F + M_R) 
    : false :=
by
  -- We would assume the opposite of what we know to lead to a contradiction here.
  -- This is a placeholder to indicate the proof should lead to a contradiction.
  sorry

end frenchwoman_present_l116_116564


namespace probability_of_not_adjacent_to_edge_is_16_over_25_l116_116391

def total_squares : ℕ := 100
def perimeter_squares : ℕ := 36
def non_perimeter_squares : ℕ := total_squares - perimeter_squares
def probability_not_adjacent_to_edge : ℚ := non_perimeter_squares / total_squares

theorem probability_of_not_adjacent_to_edge_is_16_over_25 :
  probability_not_adjacent_to_edge = 16 / 25 := by
  sorry

end probability_of_not_adjacent_to_edge_is_16_over_25_l116_116391


namespace problems_left_to_grade_l116_116873

-- Defining all the conditions
def worksheets_total : ℕ := 14
def worksheets_graded : ℕ := 7
def problems_per_worksheet : ℕ := 2

-- Stating the proof problem
theorem problems_left_to_grade : 
  (worksheets_total - worksheets_graded) * problems_per_worksheet = 14 := 
by
  sorry

end problems_left_to_grade_l116_116873


namespace cost_price_per_meter_l116_116538

-- Definitions for conditions
def total_length : ℝ := 9.25
def total_cost : ℝ := 416.25

-- The theorem to be proved
theorem cost_price_per_meter : total_cost / total_length = 45 := by
  sorry

end cost_price_per_meter_l116_116538


namespace bag_of_potatoes_weight_l116_116385

variable (W : ℝ)

-- Define the condition given in the problem.
def condition : Prop := W = 12 / (W / 2)

-- Define the statement we want to prove.
theorem bag_of_potatoes_weight : condition W → W = 24 := by
  intro h
  sorry

end bag_of_potatoes_weight_l116_116385


namespace triangle_side_lengths_l116_116321

theorem triangle_side_lengths
  (x y z : ℕ)
  (h1 : x > y)
  (h2 : y > z)
  (h3 : x + y + z = 240)
  (h4 : 3 * x - 2 * (y + z) = 5 * z + 10)
  (h5 : x < y + z) :
  (x = 113 ∧ y = 112 ∧ z = 15) ∨
  (x = 114 ∧ y = 110 ∧ z = 16) ∨
  (x = 115 ∧ y = 108 ∧ z = 17) ∨
  (x = 116 ∧ y = 106 ∧ z = 18) ∨
  (x = 117 ∧ y = 104 ∧ z = 19) ∨
  (x = 118 ∧ y = 102 ∧ z = 20) ∨
  (x = 119 ∧ y = 100 ∧ z = 21) := by
  sorry

end triangle_side_lengths_l116_116321


namespace fever_above_threshold_l116_116942

-- Definitions as per conditions
def normal_temp : ℤ := 95
def temp_increase : ℤ := 10
def fever_threshold : ℤ := 100

-- Calculated new temperature
def new_temp := normal_temp + temp_increase

-- The proof statement, asserting the correct answer
theorem fever_above_threshold : new_temp - fever_threshold = 5 := 
by 
  sorry

end fever_above_threshold_l116_116942


namespace touching_line_eq_l116_116535

theorem touching_line_eq (f : ℝ → ℝ) (f_def : ∀ x, f x = 3 * x^4 - 4 * x^3) : 
  ∃ l : ℝ → ℝ, (∀ x, l x = - (8 / 9) * x - (4 / 27)) ∧ 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (f x₁ = l x₁ ∧ f x₂ = l x₂) :=
by sorry

end touching_line_eq_l116_116535


namespace function_inverse_necessary_not_sufficient_l116_116233

theorem function_inverse_necessary_not_sufficient (f : ℝ → ℝ) :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, g (f x) = x ∧ f (g x) = x) →
  ¬ (∀ (x y : ℝ), x < y → f x < f y) :=
by
  sorry

end function_inverse_necessary_not_sufficient_l116_116233


namespace remainder_eq_27_l116_116748

def p (x : ℝ) : ℝ := x^4 + 2 * x^2 + 3
def a : ℝ := -2
def remainder := p (-2)
theorem remainder_eq_27 : remainder = 27 :=
by
  sorry

end remainder_eq_27_l116_116748


namespace number_of_algebra_textbooks_l116_116073

theorem number_of_algebra_textbooks
  (x y n : ℕ)
  (h₁ : x * n + y = 2015)
  (h₂ : y * n + x = 1580) :
  y = 287 := 
sorry

end number_of_algebra_textbooks_l116_116073


namespace samantha_interest_l116_116466

-- Definitions based on problem conditions
def P : ℝ := 2000
def r : ℝ := 0.08
def n : ℕ := 5

-- Compound interest calculation
noncomputable def A : ℝ := P * (1 + r) ^ n
noncomputable def Interest : ℝ := A - P

-- Theorem statement with Lean 4
theorem samantha_interest : Interest = 938.656 := 
by 
  sorry

end samantha_interest_l116_116466


namespace sum_of_coefficients_binomial_expansion_l116_116750

theorem sum_of_coefficients_binomial_expansion :
  (∑ k in Finset.range 8, Nat.choose 7 k) = 128 :=
by
  sorry

end sum_of_coefficients_binomial_expansion_l116_116750


namespace donation_to_second_orphanage_l116_116634

variable (total_donation : ℝ) (first_donation : ℝ) (third_donation : ℝ)

theorem donation_to_second_orphanage :
  total_donation = 650 ∧ first_donation = 175 ∧ third_donation = 250 →
  (total_donation - first_donation - third_donation = 225) := by
  sorry

end donation_to_second_orphanage_l116_116634


namespace probability_of_drawing_ball_1_is_2_over_5_l116_116099

noncomputable def probability_of_drawing_ball_1 : ℚ :=
  let total_balls := [1, 2, 3, 4, 5]
  let draw_pairs := [ (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5) ]
  let favorable_pairs := [ (1, 2), (1, 3), (1, 4), (1, 5) ]
  (favorable_pairs.length : ℚ) / (draw_pairs.length : ℚ)

theorem probability_of_drawing_ball_1_is_2_over_5 :
  probability_of_drawing_ball_1 = 2 / 5 :=
by sorry

end probability_of_drawing_ball_1_is_2_over_5_l116_116099


namespace real_roots_condition_l116_116420

theorem real_roots_condition (a : ℝ) (h : a ≠ -1) : 
    (∃ x : ℝ, x^2 + a * x + (a + 1)^2 = 0) ↔ a ∈ Set.Icc (-2 : ℝ) (-2 / 3) :=
sorry

end real_roots_condition_l116_116420


namespace certain_number_any_number_l116_116447

theorem certain_number_any_number (k : ℕ) (n : ℕ) (h1 : 5^k - k^5 = 1) (h2 : 15^k ∣ n) : true :=
by
  sorry

end certain_number_any_number_l116_116447


namespace total_cost_of_books_l116_116308

theorem total_cost_of_books
  (C1 : ℝ)
  (C2 : ℝ)
  (H1 : C1 = 285.8333333333333)
  (H2 : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 2327.5 :=
by
  sorry

end total_cost_of_books_l116_116308


namespace cylindrical_tank_volume_l116_116776

theorem cylindrical_tank_volume (d h : ℝ) (d_eq_20 : d = 20) (h_eq_10 : h = 10) : 
  π * ((d / 2) ^ 2) * h = 1000 * π :=
by
  sorry

end cylindrical_tank_volume_l116_116776


namespace correct_value_l116_116035

theorem correct_value : ∀ (x : ℕ),  (x / 6 = 12) → (x * 7 = 504) :=
  sorry

end correct_value_l116_116035


namespace maximum_value_squared_l116_116169

theorem maximum_value_squared (a b : ℝ) (h₁ : 0 < b) (h₂ : b ≤ a) :
  (∃ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧ a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a + x)^2 + (b - y)^2) →
  (a / b)^2 ≤ 4 / 3 := 
sorry

end maximum_value_squared_l116_116169


namespace compute_fraction_power_l116_116904

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end compute_fraction_power_l116_116904


namespace shuttle_speed_in_km_per_sec_l116_116557

variable (speed_mph : ℝ) (miles_to_km : ℝ) (hour_to_sec : ℝ)

theorem shuttle_speed_in_km_per_sec
  (h_speed_mph : speed_mph = 18000)
  (h_miles_to_km : miles_to_km = 1.60934)
  (h_hour_to_sec : hour_to_sec = 3600) :
  (speed_mph * miles_to_km) / hour_to_sec = 8.046 := by
sorry

end shuttle_speed_in_km_per_sec_l116_116557


namespace inequality_must_be_true_l116_116940

theorem inequality_must_be_true (a b : ℝ) (h : a > b ∧ b > 0) :
  a + 1 / b > b + 1 / a :=
sorry

end inequality_must_be_true_l116_116940


namespace fraction_of_quarters_in_1790s_l116_116465

theorem fraction_of_quarters_in_1790s (total_coins : ℕ) (coins_in_1790s : ℕ) :
  total_coins = 30 ∧ coins_in_1790s = 7 → 
  (coins_in_1790s : ℚ) / total_coins = 7 / 30 :=
by
  sorry

end fraction_of_quarters_in_1790s_l116_116465


namespace children_per_block_l116_116725

theorem children_per_block {children total_blocks : ℕ} 
  (h_total_blocks : total_blocks = 9) 
  (h_total_children : children = 54) : 
  (children / total_blocks = 6) :=
by
  -- Definitions from conditions
  have h1 : total_blocks = 9 := h_total_blocks
  have h2 : children = 54 := h_total_children

  -- Goal to prove
  -- children / total_blocks = 6
  sorry

end children_per_block_l116_116725


namespace probability_shortening_exactly_one_digit_l116_116882
open Real

theorem probability_shortening_exactly_one_digit :
  ∀ (sequence : List ℕ),
    (sequence.length = 2015) ∧
    (∀ (x ∈ sequence), x = 0 ∨ x = 9) →
      (P (λ seq, (length (shrink seq) = 2014)) sequence = 1.564e-90) := sorry

end probability_shortening_exactly_one_digit_l116_116882


namespace distribution_of_balls_into_boxes_l116_116146

noncomputable def partitions_of_6_into_4_boxes : ℕ := 9

theorem distribution_of_balls_into_boxes :
  let balls := 6
  let boxes := 4
  let ways := partitions_of_6_into_4_boxes
  ways = 9 :=
by
  let balls := 6
  let boxes := 4
  let ways := partitions_of_6_into_4_boxes
  sorry

end distribution_of_balls_into_boxes_l116_116146


namespace find_f_value_l116_116982

theorem find_f_value (ω b : ℝ) (hω : ω > 0) (hb : b = 2)
  (hT1 : 2 < ω) (hT2 : ω < 3)
  (hsymm : ∃ k : ℤ, (3 * π / 2) * ω + (π / 4) = k * π) :
  (sin ((5 / 2 : ℝ) * (π / 2) + (π / 4)) + 2 = 1) :=
by
  calc
    sin ((5 / 2 : ℝ) * (π / 2) + (π / 4)) + 2 = sin (5 * π / 4 + π / 4) + 2 : by sorry
    ... = sin (3 * π / 2) + 2 : by sorry
    ... = -1 + 2 : by sorry
    ... = 1 : by sorry

end find_f_value_l116_116982


namespace max_diff_x_y_l116_116623

theorem max_diff_x_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : 
  x - y ≤ Real.sqrt (4 / 3) := 
by
  sorry

end max_diff_x_y_l116_116623


namespace school_boys_number_l116_116158

theorem school_boys_number (B G : ℕ) (h1 : B / G = 5 / 13) (h2 : G = B + 80) : B = 50 :=
by
  sorry

end school_boys_number_l116_116158


namespace cube_side_length_l116_116201

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end cube_side_length_l116_116201


namespace max_min_value_l116_116301

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x - 2)

theorem max_min_value (M m : ℝ) (hM : M = f 3) (hm : m = f 4) : (m * m) / M = 8 / 3 := by
  sorry

end max_min_value_l116_116301


namespace find_prime_triplet_l116_116751

theorem find_prime_triplet (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  3 * p^4 - 5 * q^4 - 4 * r^2 = 26 ↔ (p, q, r) = (5, 3, 19) :=
by
  sorry

end find_prime_triplet_l116_116751


namespace ishaan_age_eq_6_l116_116264

-- Variables for ages
variable (I : ℕ) -- Ishaan's current age

-- Constants for ages
def daniel_current_age := 69
def years := 15
def daniel_future_age := daniel_current_age + years

-- Lean theorem statement
theorem ishaan_age_eq_6 
    (h1 : daniel_current_age = 69)
    (h2 : daniel_future_age = 4 * (I + years)) : 
    I = 6 := by
  sorry

end ishaan_age_eq_6_l116_116264


namespace sixth_employee_salary_l116_116075

def salaries : List Real := [1000, 2500, 3100, 3650, 1500]

def mean_salary_of_six : Real := 2291.67

theorem sixth_employee_salary : 
  let total_five := salaries.sum 
  let total_six := mean_salary_of_six * 6
  (total_six - total_five) = 2000.02 :=
by
  sorry

end sixth_employee_salary_l116_116075


namespace thomas_percentage_l116_116728

/-- 
Prove that if Emmanuel gets 100 jelly beans out of a total of 200 jelly beans, and 
Barry and Emmanuel share the remainder in a 4:5 ratio, then Thomas takes 10% 
of the jelly beans.
-/
theorem thomas_percentage (total_jelly_beans : ℕ) (emmanuel_jelly_beans : ℕ)
  (barry_ratio : ℕ) (emmanuel_ratio : ℕ) (thomas_percentage : ℕ) :
  total_jelly_beans = 200 → emmanuel_jelly_beans = 100 → barry_ratio = 4 → emmanuel_ratio = 5 →
  thomas_percentage = 10 :=
by
  intros;
  sorry

end thomas_percentage_l116_116728


namespace sum_of_fraction_components_l116_116795

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l116_116795


namespace porche_project_time_l116_116833

theorem porche_project_time :
  let total_time := 180
  let math_time := 45
  let english_time := 30
  let science_time := 50
  let history_time := 25
  let homework_time := math_time + english_time + science_time + history_time 
  total_time - homework_time = 30 :=
by
  sorry

end porche_project_time_l116_116833


namespace pirates_treasure_l116_116523

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l116_116523


namespace probability_of_selecting_letter_a_l116_116834

def total_ways := Nat.choose 5 2
def ways_to_select_a := 4
def probability_of_selecting_a := (ways_to_select_a : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_letter_a :
  probability_of_selecting_a = 2 / 5 :=
by
  -- proof steps will be filled in here
  sorry

end probability_of_selecting_letter_a_l116_116834


namespace treasures_coins_count_l116_116498

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l116_116498


namespace anika_sequence_correct_l116_116183

noncomputable def anika_sequence : ℚ :=
  let s0 := 1458
  let s1 := s0 * 3
  let s2 := s1 / 2
  let s3 := s2 * 3
  let s4 := s3 / 2
  let s5 := s4 * 3
  s5

theorem anika_sequence_correct :
  anika_sequence = (3^9 : ℚ) / 2 := by
  sorry

end anika_sequence_correct_l116_116183


namespace largest_integer_not_sum_of_30_and_composite_l116_116674

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def is_not_sum_of_30_and_composite (n : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ b < 30 ∧ is_composite(b) ∧ n = 30 * a + b

theorem largest_integer_not_sum_of_30_and_composite :
  ∃ n : ℕ, is_not_sum_of_30_and_composite(n) ∧ ∀ m : ℕ, is_not_sum_of_30_and_composite(m) → m ≤ n :=
  ⟨93, sorry⟩

end largest_integer_not_sum_of_30_and_composite_l116_116674


namespace ticket_cost_l116_116001

open Real

-- Variables for ticket prices
variable (A C S : ℝ)

-- Given conditions
def cost_condition : Prop :=
  C = A / 2 ∧ S = A - 1.50 ∧ 6 * A + 5 * C + 3 * S = 40.50

-- The goal is to prove that the total cost for 10 adult tickets, 8 child tickets,
-- and 4 senior tickets is 64.38
theorem ticket_cost (h : cost_condition A C S) : 10 * A + 8 * C + 4 * S = 64.38 :=
by
  -- Implementation of the proof would go here
  sorry

end ticket_cost_l116_116001


namespace number_of_pines_l116_116209

theorem number_of_pines (total_trees : ℕ) (T B S : ℕ) (h_total : total_trees = 101)
  (h_poplar_spacing : ∀ T1 T2, T1 ≠ T2 → (T2 - T1) > 1)
  (h_birch_spacing : ∀ B1 B2, B1 ≠ B2 → (B2 - B1) > 2)
  (h_pine_spacing : ∀ S1 S2, S1 ≠ S2 → (S2 - S1) > 3) :
  S = 25 ∨ S = 26 :=
sorry

end number_of_pines_l116_116209


namespace equally_spaced_markings_number_line_l116_116041

theorem equally_spaced_markings_number_line 
  (steps : ℕ) (distance : ℝ) (z_steps : ℕ) (z : ℝ)
  (h1 : steps = 4)
  (h2 : distance = 16)
  (h3 : z_steps = 2) :
  z = (distance / steps) * z_steps :=
by
  sorry

end equally_spaced_markings_number_line_l116_116041


namespace jacques_initial_gumballs_l116_116166

def joanna_initial_gumballs : ℕ := 40
def each_shared_gumballs_after_purchase : ℕ := 250

theorem jacques_initial_gumballs (J : ℕ) (h : 2 * (joanna_initial_gumballs + J + 4 * (joanna_initial_gumballs + J)) = 2 * each_shared_gumballs_after_purchase) : J = 60 :=
by
  sorry

end jacques_initial_gumballs_l116_116166


namespace extra_interest_l116_116702

def principal : ℝ := 7000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def interest (P R T : ℝ) : ℝ := P * R * T

theorem extra_interest :
  interest principal rate1 time - interest principal rate2 time = 840 := by
  sorry

end extra_interest_l116_116702


namespace two_digit_numbers_l116_116560

theorem two_digit_numbers (n m : ℕ) (Hn : 1 ≤ n ∧ n ≤ 9) (Hm : n < m ∧ m ≤ 9) :
  ∃ (count : ℕ), count = 36 :=
by
  sorry

end two_digit_numbers_l116_116560


namespace determine_exponent_l116_116234

theorem determine_exponent (m : ℕ) (hm : m > 0) (h_symm : ∀ x : ℝ, x^m - 3 = (-(x))^m - 3)
  (h_decr : ∀ (x y : ℝ), 0 < x ∧ x < y → x^m - 3 > y^m - 3) : m = 1 := 
sorry

end determine_exponent_l116_116234


namespace simplify_expression_l116_116622

theorem simplify_expression (x y z : ℤ) (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) :
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 :=
by
  sorry

end simplify_expression_l116_116622


namespace cube_side_length_l116_116197

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end cube_side_length_l116_116197


namespace required_average_for_tickets_l116_116896

theorem required_average_for_tickets 
  (june_score : ℝ) (patty_score : ℝ) (josh_score : ℝ) (henry_score : ℝ)
  (num_children : ℝ) (total_score : ℝ) (average_score : ℝ) (S : ℝ)
  (h1 : june_score = 97) (h2 : patty_score = 85) (h3 : josh_score = 100) 
  (h4 : henry_score = 94) (h5 : num_children = 4) 
  (h6 : total_score = june_score + patty_score + josh_score + henry_score)
  (h7 : average_score = total_score / num_children) 
  (h8 : average_score = 94)
  : S ≤ 94 :=
sorry

end required_average_for_tickets_l116_116896


namespace problem_a_problem_b_l116_116876

open Probability

noncomputable def sequenceShortenedByExactlyOneDigitProbability : ℝ :=
  1.564 * 10^(-90)

theorem problem_a (n : ℕ) (p : ℝ) (k : ℕ) (initialLength : ℕ)
  (h_n : n = 2014) (h_p : p = 0.1) (h_k : k = 1) (h_initialLength : initialLength = 2015) :
  (binomialₓ n k p * (1 - p)^(n - k)) = sequenceShortenedByExactlyOneDigitProbability := 
sorry

noncomputable def expectedLengthNewSequence : ℝ :=
  1813.6

theorem problem_b (n : ℕ) (p : ℝ) (initialLength : ℕ) 
  (h_n : n = 2014) (h_p : p = 0.1) (h_initialLength : initialLength = 2015) :
  (initialLength - n * p) = expectedLengthNewSequence :=
sorry

end problem_a_problem_b_l116_116876


namespace correct_option_is_B_l116_116138

variable (f : ℝ → ℝ)
variable (h0 : f 0 = 2)
variable (h1 : ∀ x : ℝ, deriv f x > f x + 1)

theorem correct_option_is_B : 3 * Real.exp (1 : ℝ) < f 2 + 1 := sorry

end correct_option_is_B_l116_116138


namespace james_new_friends_l116_116968

-- Definitions and assumptions based on the conditions provided
def initial_friends := 20
def lost_friends := 2
def friends_after_loss : ℕ := initial_friends - lost_friends
def friends_upon_arrival := 19

-- Definition of new friends made
def new_friends : ℕ := friends_upon_arrival - friends_after_loss

-- Statement to prove
theorem james_new_friends :
  new_friends = 1 :=
by
  -- Solution proof would be inserted here
  sorry

end james_new_friends_l116_116968


namespace sum_remainders_mod_15_l116_116868

theorem sum_remainders_mod_15 (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
    (a + b + c) % 15 = 6 :=
by
  sorry

end sum_remainders_mod_15_l116_116868


namespace no_students_unable_to_partner_l116_116483

def students_males_females :=
  let males_6th_class1 : Nat := 17
  let females_6th_class1 : Nat := 13
  let males_6th_class2 : Nat := 14
  let females_6th_class2 : Nat := 18
  let males_6th_class3 : Nat := 15
  let females_6th_class3 : Nat := 17
  let males_7th_class : Nat := 22
  let females_7th_class : Nat := 20

  let total_males := males_6th_class1 + males_6th_class2 + males_6th_class3 + males_7th_class
  let total_females := females_6th_class1 + females_6th_class2 + females_6th_class3 + females_7th_class

  total_males == total_females

theorem no_students_unable_to_partner : students_males_females = true := by
  -- Skipping the proof
  sorry

end no_students_unable_to_partner_l116_116483


namespace repeating_decimal_sum_l116_116782

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (∀ d, d ∣ a ∧ d ∣ b ↔ d = 1) ∧ 
  (a + b = 134) ∧ (∃ x : ℚ, x = 35 / 99 ∧ (x = 0.353535...)) :=
begin
  sorry
end

end repeating_decimal_sum_l116_116782


namespace selling_price_l116_116094

theorem selling_price 
  (cost_price : ℝ) 
  (profit_percentage : ℝ) 
  (h_cost : cost_price = 192) 
  (h_profit : profit_percentage = 0.25) : 
  ∃ selling_price : ℝ, selling_price = cost_price * (1 + profit_percentage) := 
by {
  sorry
}

end selling_price_l116_116094


namespace combined_height_is_320_cm_l116_116335

-- Define Maria's height in inches
def Maria_height_in_inches : ℝ := 54

-- Define Ben's height in inches
def Ben_height_in_inches : ℝ := 72

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℝ := 2.54

-- Define the combined height of Maria and Ben in centimeters
def combined_height_in_cm : ℝ := (Maria_height_in_inches + Ben_height_in_inches) * inch_to_cm

-- State and prove that the combined height is 320.0 cm
theorem combined_height_is_320_cm : combined_height_in_cm = 320.0 := by
  sorry

end combined_height_is_320_cm_l116_116335


namespace minimum_n_for_i_pow_n_eq_neg_i_l116_116292

open Complex

theorem minimum_n_for_i_pow_n_eq_neg_i : ∃ (n : ℕ), 0 < n ∧ (i^n = -i) ∧ ∀ (m : ℕ), 0 < m ∧ (i^m = -i) → n ≤ m :=
by
  sorry

end minimum_n_for_i_pow_n_eq_neg_i_l116_116292


namespace initial_deadline_l116_116547

theorem initial_deadline (D : ℝ) :
  (∀ (n : ℝ), (10 * 20) / 4 = n / 1) → 
  (∀ (m : ℝ), 8 * 75 = m * 3) →
  (∀ (d1 d2 : ℝ), d1 = 20 ∧ d2 = 93.75 → D = d1 + d2) →
  D = 113.75 :=
by {
  sorry
}

end initial_deadline_l116_116547


namespace min_a_plus_b_l116_116648

open Real

theorem min_a_plus_b (a b : ℕ) (h_a_pos : a > 1) (h_ab : ∃ a b, (a^2 * b - 1) / (a * b^2) = 1 / 2024) :
  a + b = 228 :=
sorry

end min_a_plus_b_l116_116648


namespace coefficient_x3_in_expansion_l116_116472

theorem coefficient_x3_in_expansion :
  let x := Polynomial.C
  let expr := (x^2 - x + 1)^10 in
  coeff expr 3 = -210 :=
by
  sorry

end coefficient_x3_in_expansion_l116_116472


namespace percentage_rotten_apples_l116_116996

theorem percentage_rotten_apples
  (total_apples : ℕ)
  (smell_pct : ℚ)
  (non_smelling_rotten_apples : ℕ)
  (R : ℚ) :
  total_apples = 200 →
  smell_pct = 0.70 →
  non_smelling_rotten_apples = 24 →
  0.30 * (R / 100 * total_apples) = non_smelling_rotten_apples →
  R = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end percentage_rotten_apples_l116_116996


namespace intersection_complement_A_U_B_l116_116305

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def set_A : Set ℕ := {2, 4, 6}
def set_B : Set ℕ := {1, 3, 5, 7}

theorem intersection_complement_A_U_B :
  set_A ∩ (universal_set \ set_B) = {2, 4, 6} :=
by {
  sorry
}

end intersection_complement_A_U_B_l116_116305


namespace max_k_value_l116_116140

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_k_value :
  (∀ x : ℝ, 0 < x → (∃ k : ℝ, k * x = Real.log x ∧ k ≤ f x)) ∧
  (∀ x : ℝ, 0 < x → f x ≤ 1 / Real.exp 1) ∧
  (∀ x : ℝ, 0 < x → (k = f x → k ≤ 1 / Real.exp 1)) := 
sorry

end max_k_value_l116_116140


namespace baoh2_formation_l116_116266

noncomputable def moles_of_baoh2_formed (moles_bao : ℕ) (moles_h2o : ℕ) : ℕ :=
  if moles_bao = moles_h2o then moles_bao else sorry

theorem baoh2_formation :
  moles_of_baoh2_formed 3 3 = 3 :=
by sorry

end baoh2_formation_l116_116266


namespace range_of_a_l116_116478

noncomputable def func (x a : ℝ) : ℝ := -x^2 - 2 * a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → func x a ≤ a^2) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ func x a = a^2) →
  -1 ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l116_116478


namespace sum_of_ages_l116_116990

theorem sum_of_ages {a b c : ℕ} (h1 : a * b * c = 72) (h2 : b < a) (h3 : a < c) : a + b + c = 13 :=
sorry

end sum_of_ages_l116_116990


namespace arithmetic_seq_sum_x_y_l116_116860

theorem arithmetic_seq_sum_x_y :
  ∃ (x y : ℕ), (∀ n : ℕ, n > 0 → a_n = 3 + (n - 1) * 5) ∧ x + 33 = 33 ∧ x = 28 → x + y = 61 :=
by
  sorry

end arithmetic_seq_sum_x_y_l116_116860


namespace sum_of_squares_l116_116567

theorem sum_of_squares (a b c : ℝ) (h_arith : a + b + c = 30) (h_geom : a * b * c = 216) 
(h_harm : 1/a + 1/b + 1/c = 3/4) : a^2 + b^2 + c^2 = 576 := 
by 
  sorry

end sum_of_squares_l116_116567


namespace range_of_2a_plus_b_range_of_a_minus_b_range_of_a_div_b_l116_116132

variable {a b : ℝ}

theorem range_of_2a_plus_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -10 < 2*a + b ∧ 2*a + b < 19 :=
by
  sorry

theorem range_of_a_minus_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -9 < a - b ∧ a - b < 6 :=
by
  sorry

theorem range_of_a_div_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -2 < a / b ∧ a / b < 4 :=
by
  sorry

end range_of_2a_plus_b_range_of_a_minus_b_range_of_a_div_b_l116_116132


namespace my_op_five_four_l116_116309

-- Define the operation a * b
def my_op (a b : ℤ) := a^2 + a * b - b^2

-- Define the theorem to prove 5 * 4 = 29 given the defined operation my_op
theorem my_op_five_four : my_op 5 4 = 29 := 
by 
sorry

end my_op_five_four_l116_116309


namespace check_double_root_statements_l116_116151

-- Condition Definitions
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ r : ℝ, a * r^2 + b * r + c = 0 ∧ a * (2 * r)^2 + b * (2 * r) + c = 0

-- Statement ①
def statement_1 : Prop := ¬is_double_root_equation 1 2 (-8)

-- Statement ②
def statement_2 : Prop := is_double_root_equation 1 (-3) 2

-- Statement ③
def statement_3 (m n : ℝ) : Prop := 
  (∃ r : ℝ, (r - 2) * (m * r + n) = 0 ∧ (m * (2 * r) + n = 0) ∧ r = 2) → 4 * m^2 + 5 * m * n + n^2 = 0

-- Statement ④
def statement_4 (p q : ℝ) : Prop := 
  (p * q = 2 → is_double_root_equation p 3 q)

-- Main proof problem statement
theorem check_double_root_statements (m n p q : ℝ) : 
  statement_1 ∧ statement_2 ∧ statement_3 m n ∧ statement_4 p q :=
by
  sorry

end check_double_root_statements_l116_116151


namespace angle_y_is_80_l116_116964

def parallel (m n : ℝ) : Prop := sorry

def angle_at_base (θ : ℝ) := θ = 40
def right_angle (θ : ℝ) := θ = 90
def exterior_angle (θ1 θ2 : ℝ) := θ1 + θ2 = 180

theorem angle_y_is_80 (m n : ℝ) (θ1 θ2 θ3 θ_ext : ℝ) :
  parallel m n →
  angle_at_base θ1 →
  right_angle θ2 →
  angle_at_base θ3 →
  exterior_angle θ_ext θ3 →
  θ_ext = 80 := by
  sorry

end angle_y_is_80_l116_116964


namespace largest_a_pow_b_l116_116915

theorem largest_a_pow_b (a b : ℕ) (h_pos_a : 1 < a) (h_pos_b : 1 < b) (h_eq : a^b * b^a + a^b + b^a = 5329) : 
  a^b = 64 :=
by
  sorry

end largest_a_pow_b_l116_116915


namespace gcd_90_405_l116_116277

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l116_116277


namespace find_x_l116_116280

theorem find_x (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 48) : x = 8 :=
sorry

end find_x_l116_116280


namespace maximum_squares_formation_l116_116995

theorem maximum_squares_formation (total_matchsticks : ℕ) (triangles : ℕ) (used_for_triangles : ℕ) (remaining_matchsticks : ℕ) (squares : ℕ):
  total_matchsticks = 24 →
  triangles = 6 →
  used_for_triangles = 13 →
  remaining_matchsticks = total_matchsticks - used_for_triangles →
  squares = remaining_matchsticks / 4 →
  squares = 4 :=
by
  sorry

end maximum_squares_formation_l116_116995


namespace real_solutions_eq_l116_116745

theorem real_solutions_eq :
  ∀ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12) → (x = 10 ∨ x = -1) :=
by
  sorry

end real_solutions_eq_l116_116745


namespace incorrect_expression_l116_116036

variable (x y : ℝ)

theorem incorrect_expression (h : x > y) (hnx : x < 0) (hny : y < 0) : x^2 - 3 ≤ y^2 - 3 := by
sorry

end incorrect_expression_l116_116036


namespace number_of_mandatory_questions_correct_l116_116807

-- Definitions and conditions
def num_mandatory_questions (x : ℕ) (k : ℕ) (y : ℕ) (m : ℕ) : Prop :=
  (3 * k - 2 * (x - k) + 5 * m = 49) ∧
  (k + m = 15) ∧
  (y = 25 - x)

-- Proof statement
theorem number_of_mandatory_questions_correct :
  ∃ x k y m : ℕ, num_mandatory_questions x k y m ∧ x = 13 :=
by
  sorry

end number_of_mandatory_questions_correct_l116_116807


namespace length_of_diagonal_AC_l116_116602

-- Definitions based on the conditions
variable (AB BC CD DA AC : ℝ)
variable (angle_ADC : ℝ)

-- Conditions
def conditions : Prop :=
  AB = 12 ∧ BC = 12 ∧ CD = 15 ∧ DA = 15 ∧ angle_ADC = 120

theorem length_of_diagonal_AC (h : conditions AB BC CD DA angle_ADC) : AC = 15 :=
sorry

end length_of_diagonal_AC_l116_116602


namespace average_weight_of_section_B_l116_116849

theorem average_weight_of_section_B
  (num_students_A : ℕ) (num_students_B : ℕ)
  (avg_weight_A : ℝ) (avg_weight_class : ℝ)
  (total_students : ℕ := num_students_A + num_students_B)
  (total_weight_class : ℝ := total_students * avg_weight_class)
  (total_weight_A : ℝ := num_students_A * avg_weight_A)
  (total_weight_B : ℝ := total_weight_class - total_weight_A)
  (avg_weight_B : ℝ := total_weight_B / num_students_B) :
  num_students_A = 50 →
  num_students_B = 40 →
  avg_weight_A = 50 →
  avg_weight_class = 58.89 →
  avg_weight_B = 70.0025 :=
by intros; sorry

end average_weight_of_section_B_l116_116849


namespace rearrange_marked_squares_l116_116449

theorem rearrange_marked_squares (n k : ℕ) (h : n > 1) (h' : k ≤ n + 1) :
  ∃ (f g : Fin n → Fin n), true := sorry

end rearrange_marked_squares_l116_116449


namespace area_of_region_l116_116731

theorem area_of_region :
  ∫ y in (0:ℝ)..(1:ℝ), y ^ (2 / 3) = 3 / 5 :=
by
  sorry

end area_of_region_l116_116731


namespace max_neg_integers_l116_116173

theorem max_neg_integers (
  a b c d e f g h : ℤ
) (h_a : a ≠ 0) (h_c : c ≠ 0) (h_e : e ≠ 0)
  (h_ineq : (a * b^2 + c * d * e^3) * (f * g^2 * h + f^3 - g^2) < 0)
  (h_abs : |d| < |f| ∧ |f| < |h|)
  : ∃ s, s = 5 ∧ ∀ (neg_count : ℕ), neg_count ≤ s := 
sorry

end max_neg_integers_l116_116173


namespace symmetric_circle_l116_116314

theorem symmetric_circle :
  ∀ (C D : Type) (hD : ∀ x y : ℝ, (x + 2)^2 + (y - 6)^2 = 1) (hline : ∀ x y : ℝ, x - y + 5 = 0), 
  (∀ x y : ℝ, (x - 1)^2 + (y - 3)^2 = 1) := 
by sorry

end symmetric_circle_l116_116314


namespace treasures_coins_count_l116_116501

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l116_116501


namespace solve_problem_l116_116117

open Real

noncomputable def problem_statement : ℝ :=
  2 * log (sqrt 2) + (log 5 / log 2) * log 2

theorem solve_problem : problem_statement = 1 := by
  sorry

end solve_problem_l116_116117


namespace cost_per_treat_l116_116815

def treats_per_day : ℕ := 2
def days_in_month : ℕ := 30
def total_spent : ℝ := 6.0

theorem cost_per_treat : (total_spent / (treats_per_day * days_in_month : ℕ)) = 0.10 :=
by 
  sorry

end cost_per_treat_l116_116815


namespace C5_properties_l116_116913


def C5 : SimpleGraph (Fin 5) :=
{ adj := λ i j, 
    i = j + 1 ∨ i + 1 = j ∨ (i = 0 ∧ j = 4) ∨ (i = 4 ∧ j = 0),
  symm := by
    finish,
  loopless := by
    finish
}

noncomputable def C5_chromatic_number : ℕ :=
  chromatic_number C5

def C5_no_3_cliques : Prop :=
  ∀ (v1 v2 v3 : Fin 5), (C5.adj v1 v2 ∧ C5.adj v2 v3 ∧ C5.adj v3 v1) → false

theorem C5_properties :
  C5_chromatic_number = 3 ∧ C5_no_3_cliques :=
begin
  sorry, -- Proof goes here
end

end C5_properties_l116_116913


namespace rate_for_gravelling_roads_l116_116711

variable (length breadth width cost : ℕ)
variable (rate per_square_meter : ℕ)

def total_area_parallel_length : ℕ := length * width
def total_area_parallel_breadth : ℕ := (breadth * width) - (width * width)
def total_area : ℕ := total_area_parallel_length length width + total_area_parallel_breadth breadth width

def rate_per_square_meter := cost / total_area length breadth width

theorem rate_for_gravelling_roads :
  (length = 70) →
  (breadth = 30) →
  (width = 5) →
  (cost = 1900) →
  rate_per_square_meter length breadth width cost = 4 := by
  intros; exact sorry

end rate_for_gravelling_roads_l116_116711


namespace wendi_owns_rabbits_l116_116671

/-- Wendi's plot of land is 200 feet by 900 feet. -/
def area_land_in_feet : ℕ := 200 * 900

/-- One rabbit can eat enough grass to clear ten square yards of lawn area per day. -/
def rabbit_clear_per_day : ℕ := 10

/-- It would take 20 days for all of Wendi's rabbits to clear all the grass off of her grassland property. -/
def days_to_clear : ℕ := 20

/-- Convert feet to yards (3 feet in a yard). -/
def feet_to_yards (feet : ℕ) : ℕ := feet / 3

/-- Calculate the total area of the land in square yards. -/
def area_land_in_yards : ℕ := (feet_to_yards 200) * (feet_to_yards 900)

theorem wendi_owns_rabbits (total_area : ℕ := area_land_in_yards)
                            (clear_area_per_rabbit : ℕ := rabbit_clear_per_day * days_to_clear) :
  total_area / clear_area_per_rabbit = 100 := 
sorry

end wendi_owns_rabbits_l116_116671


namespace max_distance_on_curve_and_ellipse_l116_116136

noncomputable def max_distance_between_P_and_Q : ℝ :=
  6 * Real.sqrt 2

theorem max_distance_on_curve_and_ellipse :
  ∃ P Q, (P ∈ { p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 2 }) ∧ 
         (Q ∈ { q : ℝ × ℝ | q.1^2 / 10 + q.2^2 = 1 }) ∧ 
         (dist P Q = max_distance_between_P_and_Q) := 
sorry

end max_distance_on_curve_and_ellipse_l116_116136


namespace point_A_outside_circle_l116_116297

noncomputable def circle_radius := 6
noncomputable def distance_OA := 8

theorem point_A_outside_circle : distance_OA > circle_radius :=
by
  -- Solution will go here
  sorry

end point_A_outside_circle_l116_116297


namespace symmetric_intersection_range_l116_116597

theorem symmetric_intersection_range (k m p : ℝ)
  (intersection_symmetric : ∀ (x y : ℝ), 
    (x = k*y - 1 ∧ (x^2 + y^2 + k*x + m*y + 2*p = 0)) → 
    (y = x)) 
  : p < -3/2 := 
sorry

end symmetric_intersection_range_l116_116597


namespace sum_mod_15_l116_116865

theorem sum_mod_15 
  (d e f : ℕ) 
  (hd : d % 15 = 11)
  (he : e % 15 = 12)
  (hf : f % 15 = 13) : 
  (d + e + f) % 15 = 6 :=
by
  sorry

end sum_mod_15_l116_116865


namespace expected_number_of_girls_left_of_all_boys_l116_116369

noncomputable def expected_girls_left_of_all_boys (boys girls : ℕ) : ℚ :=
    if boys = 10 ∧ girls = 7 then (7 : ℚ) / 11 else 0

theorem expected_number_of_girls_left_of_all_boys 
    (boys girls : ℕ) (h_boys : boys = 10) (h_girls : girls = 7) :
    expected_girls_left_of_all_boys boys girls = (7 : ℚ) / 11 :=
by
  rw [expected_girls_left_of_all_boys, if_pos]
  { simp }
  { exact ⟨h_boys, h_girls⟩ }

end expected_number_of_girls_left_of_all_boys_l116_116369


namespace Clare_has_more_pencils_than_Jeanine_l116_116452

def Jeanine_initial_pencils : ℕ := 250
def Clare_initial_pencils : ℤ := (-3 : ℤ) * Jeanine_initial_pencils / 5
def Jeanine_pencils_given_Abby : ℕ := (2 : ℕ) * Jeanine_initial_pencils / 7
def Jeanine_pencils_given_Lea : ℕ := (5 : ℕ) * Jeanine_initial_pencils / 11
def Clare_pencils_after_squaring : ℤ := Clare_initial_pencils ^ 2
def Clare_pencils_after_Jeanine_share : ℤ := Clare_pencils_after_squaring + (-1) * Jeanine_initial_pencils / 4

def Jeanine_final_pencils : ℕ := Jeanine_initial_pencils - Jeanine_pencils_given_Abby - Jeanine_pencils_given_Lea

theorem Clare_has_more_pencils_than_Jeanine :
  Clare_pencils_after_Jeanine_share - Jeanine_final_pencils = 22372 :=
sorry

end Clare_has_more_pencils_than_Jeanine_l116_116452


namespace length_of_bridge_l116_116235

noncomputable def speed_kmh_to_mps (speed_kmh : ℕ) : ℝ := speed_kmh * 1000 / 3600

def total_distance_covered (speed_mps : ℝ) (time_s : ℕ) : ℝ := speed_mps * time_s

def bridge_length (total_distance : ℝ) (train_length : ℝ) : ℝ := total_distance - train_length

theorem length_of_bridge (train_length : ℝ) (time_s : ℕ) (speed_kmh : ℕ) :
  bridge_length (total_distance_covered (speed_kmh_to_mps speed_kmh) time_s) train_length = 299.9 :=
by
  have speed_mps := speed_kmh_to_mps speed_kmh
  have total_distance := total_distance_covered speed_mps time_s
  have length_of_bridge := bridge_length total_distance train_length
  sorry

end length_of_bridge_l116_116235


namespace relay_race_arrangements_l116_116633

theorem relay_race_arrangements :
  let boys := 6
      girls := 2
      totalSelections := 4
      selectedBoys := 3
      selectedGirls := 1 in 
  (nat.choose girls selectedGirls) * (nat.choose boys selectedBoys) * selectedBoys *
  (nat.perm 3 3) = 720 := sorry

end relay_race_arrangements_l116_116633


namespace seven_solutions_l116_116444

theorem seven_solutions: ∃ (pairs : List (ℕ × ℕ)), 
  (∀ (x y : ℕ), (x < y) → ((1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 2007) ↔ (x, y) ∈ pairs) 
  ∧ pairs.length = 7 :=
sorry

end seven_solutions_l116_116444


namespace julia_played_with_kids_on_Monday_l116_116816

theorem julia_played_with_kids_on_Monday (k_wednesday : ℕ) (k_monday : ℕ)
  (h1 : k_wednesday = 4) (h2 : k_monday = k_wednesday + 2) : k_monday = 6 := 
by
  sorry

end julia_played_with_kids_on_Monday_l116_116816


namespace joe_initial_money_l116_116455

theorem joe_initial_money (cost_notebook cost_book money_left : ℕ) 
                          (num_notebooks num_books : ℕ)
                          (h1 : cost_notebook = 4) 
                          (h2 : cost_book = 7)
                          (h3 : num_notebooks = 7) 
                          (h4 : num_books = 2) 
                          (h5 : money_left = 14) :
  (num_notebooks * cost_notebook + num_books * cost_book + money_left) = 56 := by
  sorry

end joe_initial_money_l116_116455


namespace max_residents_per_apartment_l116_116811

theorem max_residents_per_apartment (total_floors : ℕ) (floors_with_6_apts : ℕ) (floors_with_5_apts : ℕ)
  (rooms_per_6_floors : ℕ) (rooms_per_5_floors : ℕ) (max_residents : ℕ) : 
  total_floors = 12 ∧ floors_with_6_apts = 6 ∧ floors_with_5_apts = 6 ∧ 
  rooms_per_6_floors = 6 ∧ rooms_per_5_floors = 5 ∧ max_residents = 264 → 
  264 / (6 * 6 + 6 * 5) = 4 := sorry

end max_residents_per_apartment_l116_116811


namespace room_total_space_l116_116612

-- Definitions based on the conditions
def bookshelf_space : ℕ := 80
def reserved_space : ℕ := 160
def number_of_shelves : ℕ := 3

-- The theorem statement
theorem room_total_space : 
  (number_of_shelves * bookshelf_space) + reserved_space = 400 := 
by
  sorry

end room_total_space_l116_116612


namespace cube_side_length_l116_116200

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end cube_side_length_l116_116200


namespace calculate_value_l116_116364

def a : ℕ := 2500
def b : ℕ := 2109
def d : ℕ := 64

theorem calculate_value : (a - b) ^ 2 / d = 2389 := by
  sorry

end calculate_value_l116_116364


namespace max_x_plus_y_l116_116614

-- Define the conditions as hypotheses in a Lean statement
theorem max_x_plus_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^4 = (x - 1) * (y^3 - 23) - 1) :
  x + y ≤ 7 ∧ (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^4 = (x - 1) * (y^3 - 23) - 1 ∧ x + y = 7) :=
by
  sorry

end max_x_plus_y_l116_116614


namespace greatest_prime_factor_of_5_pow_7_plus_6_pow_6_l116_116672

def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry -- Implementation of finding greatest prime factor goes here

theorem greatest_prime_factor_of_5_pow_7_plus_6_pow_6 : 
  greatest_prime_factor (5^7 + 6^6) = 211 := 
by 
  sorry -- Proof of the theorem goes here

end greatest_prime_factor_of_5_pow_7_plus_6_pow_6_l116_116672


namespace digit_B_divisibility_l116_116665

theorem digit_B_divisibility :
  ∃ B : ℕ, B < 10 ∧ (2 * 100 + B * 10 + 9) % 13 = 0 ↔ B = 0 :=
by
  sorry

end digit_B_divisibility_l116_116665


namespace repeating_decimal_fraction_l116_116801

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l116_116801


namespace probability_of_collinear_dots_in_5x5_grid_l116_116808

def collinear_dots_probability (total_dots chosen_dots collinear_sets : ℕ) : ℚ :=
  (collinear_sets : ℚ) / (Nat.choose total_dots chosen_dots)

theorem probability_of_collinear_dots_in_5x5_grid :
  collinear_dots_probability 25 4 12 = 12 / 12650 := by
  sorry

end probability_of_collinear_dots_in_5x5_grid_l116_116808


namespace tangent_line_equation_l116_116910

open Real

noncomputable def circle_center : ℝ × ℝ := (2, 1)
noncomputable def tangent_point : ℝ × ℝ := (4, 3)

def circle_equation (x y : ℝ) : Prop := (x - circle_center.1)^2 + (y - circle_center.2)^2 = 1

theorem tangent_line_equation :
  ∀ (x y : ℝ), ( (x = 4 ∧ y = 3) ∨ circle_equation x y ) → 2 * x + 2 * y - 7 = 0 :=
sorry

end tangent_line_equation_l116_116910


namespace sequence_and_sum_l116_116285

-- Given conditions as definitions
def a₁ : ℕ := 1

def recurrence (a_n a_n1 : ℕ) (n : ℕ) : Prop := (a_n1 = 3 * a_n * (1 + (1 / n : ℝ)))

-- Stating the theorem
theorem sequence_and_sum (a : ℕ → ℕ) (S : ℕ → ℝ) :
  (a 1 = a₁) →
  (∀ n, recurrence (a n) (a (n + 1)) n) →
  (∀ n, a n = n * 3 ^ (n - 1)) ∧
  (∀ n, S n = (2 * n - 1) * 3 ^ n / 4 + 1 / 4) :=
by
  sorry

end sequence_and_sum_l116_116285


namespace solve_trig_equation_proof_l116_116755

noncomputable def solve_trig_equation (θ : ℝ) : Prop :=
  2 * Real.cos θ ^ 2 - 5 * Real.cos θ + 2 = 0 ∧ (θ = 60 / 180 * Real.pi)

theorem solve_trig_equation_proof (θ : ℝ) :
  solve_trig_equation θ :=
sorry

end solve_trig_equation_proof_l116_116755


namespace initial_amount_l116_116610

theorem initial_amount (P : ℝ) :
  (P * 1.0816 - P * 1.08 = 3.0000000000002274) → P = 1875.0000000001421 :=
by
  sorry

end initial_amount_l116_116610


namespace ending_number_of_SetB_l116_116180

-- Definition of Set A
def SetA : Set ℕ := {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

-- Definition of Set B
def SetB_ends_at (n : ℕ) : Set ℕ := {i | 6 ≤ i ∧ i ≤ n}

-- The main theorem statement
theorem ending_number_of_SetB : ∃ n, SetA ∩ SetB_ends_at n = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15} ∧ 10 ∈ SetA ∩ SetB_ends_at n := 
sorry

end ending_number_of_SetB_l116_116180


namespace ramesh_paid_price_l116_116631

variable (P : ℝ) (P_paid : ℝ)

-- conditions
def discount_price (P : ℝ) : ℝ := 0.80 * P
def additional_cost : ℝ := 125 + 250
def total_cost_with_discount (P : ℝ) : ℝ := discount_price P + additional_cost
def selling_price_without_discount (P : ℝ) : ℝ := 1.10 * P
def given_selling_price : ℝ := 18975

-- the theorem to prove
theorem ramesh_paid_price :
  (∃ P : ℝ, selling_price_without_discount P = given_selling_price ∧ total_cost_with_discount P = 14175) :=
by
  sorry

end ramesh_paid_price_l116_116631


namespace number_of_packs_l116_116595

-- Given conditions
def cost_per_pack : ℕ := 11
def total_money : ℕ := 110

-- Statement to prove
theorem number_of_packs :
  total_money / cost_per_pack = 10 := by
  sorry

end number_of_packs_l116_116595


namespace square_side_length_l116_116222

theorem square_side_length (A : ℝ) (h : A = 25) : ∃ s : ℝ, s * s = A ∧ s = 5 :=
by
  sorry

end square_side_length_l116_116222


namespace inverse_proportion_order_l116_116315

theorem inverse_proportion_order (k : ℝ) (y1 y2 y3 : ℝ) 
  (h1 : k > 0) 
  (ha : y1 = k / (-3)) 
  (hb : y2 = k / (-2)) 
  (hc : y3 = k / 2) : 
  y2 < y1 ∧ y1 < y3 := 
sorry

end inverse_proportion_order_l116_116315


namespace cars_meet_first_time_l116_116130

-- Definitions based on conditions
def car (t : ℕ) (v : ℕ) : ℕ := t * v
def car_meet (t : ℕ) (v1 v2 : ℕ) : Prop := ∃ n, v1 * t + v2 * t = n

-- Given conditions
variables (v_A v_B v_C v_D : ℕ) (pairwise_different : v_A ≠ v_B ∧ v_B ≠ v_C ∧ v_C ≠ v_D ∧ v_D ≠ v_A)
variables (t1 t2 t3 : ℕ) (time_AC : t1 = 7) (time_BD : t1 = 7) (time_AB : t2 = 53)
variables (condition1 : car_meet t1 v_A v_C) (condition2 : car_meet t1 v_B v_D)
variables (condition3 : ∃ k, (v_A - v_B) * t2 = k)

-- Theorem statement
theorem cars_meet_first_time : ∃ t, (t = 371) := sorry

end cars_meet_first_time_l116_116130


namespace curved_surface_area_cone_l116_116768

variable (a α β : ℝ) (l := a * Real.sin α) (r := a * Real.cos β)

theorem curved_surface_area_cone :
  π * r * l = π * a^2 * Real.sin α * Real.cos β := by
  sorry

end curved_surface_area_cone_l116_116768


namespace largest_non_sum_217_l116_116679

def is_prime (n : ℕ) : Prop := sorry -- This would be some definition of primality

noncomputable def largest_non_sum_of_composite (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ b ∧ b < 30 ∧ (∀ i : ℕ, i < a → is_prime (b + 30 * i)) →
  n ≤ 30 * a + b

theorem largest_non_sum_217 : ∃ n, largest_non_sum_of_composite n ∧ n = 217 := sorry

end largest_non_sum_217_l116_116679


namespace expected_girls_left_of_boys_l116_116367

theorem expected_girls_left_of_boys : 
  (∑ i in (finset.range 7), ((i+1) : ℝ) / 17) = 7 / 11 :=
sorry

end expected_girls_left_of_boys_l116_116367


namespace companyKW_price_percentage_l116_116261

theorem companyKW_price_percentage (A B P : ℝ) (h1 : P = 1.40 * A) (h2 : P = 2.00 * B) : 
  P / ((P / 1.40) + (P / 2.00)) * 100 = 82.35 :=
by sorry

end companyKW_price_percentage_l116_116261


namespace correct_algorithm_option_l116_116847

def OptionA := ("Sequential structure", "Flow structure", "Loop structure")
def OptionB := ("Sequential structure", "Conditional structure", "Nested structure")
def OptionC := ("Sequential structure", "Conditional structure", "Loop structure")
def OptionD := ("Flow structure", "Conditional structure", "Loop structure")

-- The correct structures of an algorithm are sequential, conditional, and loop.
def algorithm_structures := ("Sequential structure", "Conditional structure", "Loop structure")

theorem correct_algorithm_option : algorithm_structures = OptionC := 
by 
  -- This would be proven by logic and checking the options; omitted here with 'sorry'
  sorry

end correct_algorithm_option_l116_116847


namespace circle_radius_five_iff_l116_116116

noncomputable def circle_eq_radius (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + 8*x + y^2 + 4*y - k = 0

def is_circle_with_radius (r : ℝ) (x y : ℝ) (k : ℝ) : Prop :=
  circle_eq_radius x y k ↔ r = 5 ∧ k = 5

theorem circle_radius_five_iff (k : ℝ) :
  (∃ x y : ℝ, circle_eq_radius x y k) ↔ k = 5 :=
sorry

end circle_radius_five_iff_l116_116116


namespace find_total_coins_l116_116514

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l116_116514


namespace pants_cost_l116_116826

/-- Given:
- 3 skirts with each costing $20.00
- 5 blouses with each costing $15.00
- The total spending is $180.00
- A discount on pants: buy 1 pair get 1 pair 1/2 off

Prove that each pair of pants costs $30.00 before the discount. --/
theorem pants_cost (cost_skirt cost_blouse total_amount : ℤ) (pants_discount: ℚ) (total_cost: ℤ) :
  cost_skirt = 20 ∧ cost_blouse = 15 ∧ total_amount = 180 
  ∧ pants_discount * 2 = 1 
  ∧ total_cost = 3 * cost_skirt + 5 * cost_blouse + 3/2 * pants_discount → 
  pants_discount = 30 := by
  sorry

end pants_cost_l116_116826


namespace sequence_and_sum_l116_116284

-- Given conditions as definitions
def a₁ : ℕ := 1

def recurrence (a_n a_n1 : ℕ) (n : ℕ) : Prop := (a_n1 = 3 * a_n * (1 + (1 / n : ℝ)))

-- Stating the theorem
theorem sequence_and_sum (a : ℕ → ℕ) (S : ℕ → ℝ) :
  (a 1 = a₁) →
  (∀ n, recurrence (a n) (a (n + 1)) n) →
  (∀ n, a n = n * 3 ^ (n - 1)) ∧
  (∀ n, S n = (2 * n - 1) * 3 ^ n / 4 + 1 / 4) :=
by
  sorry

end sequence_and_sum_l116_116284


namespace expected_value_of_girls_left_of_boys_l116_116374

def num_girls_to_left_of_all_boys (boys girls : ℕ) : ℚ :=
  (boys + girls : ℚ) / (boys + 1)

theorem expected_value_of_girls_left_of_boys :
  num_girls_to_left_of_all_boys 10 7 = 7 / 11 := by
  sorry

end expected_value_of_girls_left_of_boys_l116_116374


namespace conic_sections_union_l116_116072

theorem conic_sections_union :
  ∀ (x y : ℝ), (y^4 - 4*x^4 = 2*y^2 - 1) ↔ 
               (y^2 - 2*x^2 = 1) ∨ (y^2 + 2*x^2 = 1) := 
by
  sorry

end conic_sections_union_l116_116072


namespace no_such_primes_l116_116894

theorem no_such_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_three : p > 3) (hq_gt_three : q > 3) (hq_div_p2_minus_1 : q ∣ (p^2 - 1)) 
  (hp_div_q2_minus_1 : p ∣ (q^2 - 1)) : false := 
sorry

end no_such_primes_l116_116894


namespace minute_first_catch_hour_l116_116109

theorem minute_first_catch_hour :
  ∃ (t : ℚ), t = 60 * (1 + (5 / 11)) :=
sorry

end minute_first_catch_hour_l116_116109


namespace sum_remainders_mod_15_l116_116867

theorem sum_remainders_mod_15 (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
    (a + b + c) % 15 = 6 :=
by
  sorry

end sum_remainders_mod_15_l116_116867


namespace cube_side_length_l116_116202

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end cube_side_length_l116_116202


namespace inequality_solution_l116_116577

theorem inequality_solution (x : ℝ) :
  (abs ((x^2 - 5 * x + 4) / 3) < 1) ↔ 
  ((5 - Real.sqrt 21) / 2 < x) ∧ (x < (5 + Real.sqrt 21) / 2) := 
sorry

end inequality_solution_l116_116577


namespace problem_inequality_l116_116027

variable {α : Type*} [LinearOrder α]

def M (x y : α) : α := max x y
def m (x y : α) : α := min x y

theorem problem_inequality (a b c d e : α) (h : a < b) (h1 : b < c) (h2 : c < d) (h3 : d < e) : 
  M (M a (m b c)) (m d (m a e)) = b := sorry

end problem_inequality_l116_116027


namespace team_not_losing_probability_l116_116186

theorem team_not_losing_probability
  (p_center_forward : ℝ) (p_winger : ℝ) (p_attacking_midfielder : ℝ)
  (rate_center_forward : ℝ) (rate_winger : ℝ) (rate_attacking_midfielder : ℝ)
  (h_center_forward : p_center_forward = 0.2) (h_winger : p_winger = 0.5) (h_attacking_midfielder : p_attacking_midfielder = 0.3)
  (h_rate_center_forward : rate_center_forward = 0.4) (h_rate_winger : rate_winger = 0.2) (h_rate_attacking_midfielder : rate_attacking_midfielder = 0.2) :
  (p_center_forward * (1 - rate_center_forward) + p_winger * (1 - rate_winger) + p_attacking_midfielder * (1 - rate_attacking_midfielder)) = 0.76 :=
by
  sorry

end team_not_losing_probability_l116_116186


namespace gcd_90_405_l116_116279

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l116_116279


namespace negation_example_l116_116348

theorem negation_example : ¬ (∀ x : ℝ, x^2 ≥ Real.log 2) ↔ ∃ x : ℝ, x^2 < Real.log 2 :=
by
  sorry

end negation_example_l116_116348


namespace treasures_coins_count_l116_116499

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l116_116499


namespace probability_shortening_exactly_one_digit_l116_116881
open Real

theorem probability_shortening_exactly_one_digit :
  ∀ (sequence : List ℕ),
    (sequence.length = 2015) ∧
    (∀ (x ∈ sequence), x = 0 ∨ x = 9) →
      (P (λ seq, (length (shrink seq) = 2014)) sequence = 1.564e-90) := sorry

end probability_shortening_exactly_one_digit_l116_116881


namespace find_norm_b_projection_of_b_on_a_l116_116137

open Real EuclideanSpace

noncomputable def a : ℝ := 4

noncomputable def angle_ab : ℝ := π / 4  -- 45 degrees in radians

noncomputable def inner_prod_condition (b : ℝ) : ℝ := 
  (1 / 2 * a) * (2 * a) + 
  (1 / 2 * a) * (-3 * b) + 
  b * (2 * a) + 
  b * (-3 * b) - 12

theorem find_norm_b (b : ℝ) (hb : inner_prod_condition b = 0) : b = sqrt 2 :=
  sorry

theorem projection_of_b_on_a (b : ℝ) (hb : inner_prod_condition b = 0) : 
  (b * cos angle_ab) = 1 :=
  sorry

end find_norm_b_projection_of_b_on_a_l116_116137


namespace units_digit_in_base_7_l116_116991

theorem units_digit_in_base_7 (n m : ℕ) (h1 : n = 312) (h2 : m = 57) : (n * m) % 7 = 4 :=
by
  sorry

end units_digit_in_base_7_l116_116991


namespace cube_side_length_l116_116192

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (h0 : s ≠ 0) : s = 6 :=
sorry

end cube_side_length_l116_116192


namespace pictures_at_the_museum_l116_116378

theorem pictures_at_the_museum (M : ℕ) (zoo_pics : ℕ) (deleted_pics : ℕ) (remaining_pics : ℕ)
    (h1 : zoo_pics = 15) (h2 : deleted_pics = 31) (h3 : remaining_pics = 2) (h4 : zoo_pics + M = deleted_pics + remaining_pics) :
    M = 18 := 
sorry

end pictures_at_the_museum_l116_116378


namespace inequality_of_trig_function_l116_116016

theorem inequality_of_trig_function 
  (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
sorry

end inequality_of_trig_function_l116_116016


namespace div_of_floats_l116_116858

theorem div_of_floats : (0.2 : ℝ) / (0.005 : ℝ) = 40 := 
by
  sorry

end div_of_floats_l116_116858


namespace find_P20_l116_116282

theorem find_P20 (a b : ℝ) (P : ℝ → ℝ) (hP : ∀ x, P x = x^2 + a * x + b) 
  (h_condition : P 10 + P 30 = 40) : P 20 = -80 :=
by {
  -- Additional statements to structure the proof can go here
  sorry
}

end find_P20_l116_116282


namespace part1_solution_set_part2_range_of_m_l116_116925

noncomputable def f (x : ℝ) : ℝ := abs (x + 2) * abs (x - 3)

theorem part1_solution_set :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} :=
sorry

theorem part2_range_of_m (m : ℝ) :
  (∃ x : ℝ, f x ≤ abs (3 * m - 2)) → m ∈ Set.Iic (-1) ∪ Set.Ici (7 / 3) :=
sorry

end part1_solution_set_part2_range_of_m_l116_116925


namespace train_stoppage_time_l116_116092

theorem train_stoppage_time
  (D : ℝ) -- Distance in kilometers
  (T_no_stop : ℝ := D / 300) -- Time without stoppages in hours
  (T_with_stop : ℝ := D / 200) -- Time with stoppages in hours
  (T_stop : ℝ := T_with_stop - T_no_stop) -- Time lost due to stoppages in hours
  (T_stop_minutes : ℝ := T_stop * 60) -- Time lost due to stoppages in minutes
  (stoppage_per_hour : ℝ := T_stop_minutes / (D / 300)) -- Time stopped per hour of travel
  : stoppage_per_hour = 30 := sorry

end train_stoppage_time_l116_116092


namespace odd_factor_form_l116_116625

theorem odd_factor_form (n : ℕ) (x y : ℕ) (h_n : n > 0) (h_gcd : Nat.gcd x y = 1) :
  ∀ p, p ∣ (x ^ (2 ^ n) + y ^ (2 ^ n)) ∧ Odd p → ∃ k > 0, p = 2^(n+1) * k + 1 := 
by
  sorry

end odd_factor_form_l116_116625


namespace avg_highway_mpg_l116_116562

noncomputable def highway_mpg (total_distance : ℕ) (fuel : ℕ) : ℝ :=
  total_distance / fuel
  
theorem avg_highway_mpg :
  highway_mpg 305 25 = 12.2 :=
by
  sorry

end avg_highway_mpg_l116_116562


namespace sequence_a100_gt_14_l116_116844

theorem sequence_a100_gt_14 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, 1 ≤ n → a (n+1) = a n + 1 / a n) :
  a 100 > 14 :=
by sorry

end sequence_a100_gt_14_l116_116844


namespace age_of_youngest_child_l116_116243

theorem age_of_youngest_child
  (total_bill : ℝ)
  (mother_charge : ℝ)
  (child_charge_per_year : ℝ)
  (children_total_years : ℝ)
  (twins_age : ℕ)
  (youngest_child_age : ℕ)
  (h_total_bill : total_bill = 13.00)
  (h_mother_charge : mother_charge = 6.50)
  (h_child_charge_per_year : child_charge_per_year = 0.65)
  (h_children_bill : total_bill - mother_charge = children_total_years * child_charge_per_year)
  (h_children_age : children_total_years = 10)
  (h_youngest_child : youngest_child_age = 10 - 2 * twins_age) :
  youngest_child_age = 2 ∨ youngest_child_age = 4 :=
by
  sorry

end age_of_youngest_child_l116_116243


namespace polynomial_divisibility_l116_116179

theorem polynomial_divisibility (a b x y : ℤ) : 
  ∃ k : ℤ, (a * x + b * y)^3 + (b * x + a * y)^3 = k * (a + b) * (x + y) := by
  sorry

end polynomial_divisibility_l116_116179


namespace expected_number_of_girls_left_of_all_boys_l116_116370

noncomputable def expected_girls_left_of_all_boys (boys girls : ℕ) : ℚ :=
    if boys = 10 ∧ girls = 7 then (7 : ℚ) / 11 else 0

theorem expected_number_of_girls_left_of_all_boys 
    (boys girls : ℕ) (h_boys : boys = 10) (h_girls : girls = 7) :
    expected_girls_left_of_all_boys boys girls = (7 : ℚ) / 11 :=
by
  rw [expected_girls_left_of_all_boys, if_pos]
  { simp }
  { exact ⟨h_boys, h_girls⟩ }

end expected_number_of_girls_left_of_all_boys_l116_116370


namespace prime_number_solution_l116_116415

theorem prime_number_solution (X Y : ℤ) (h_prime : Prime (X^4 + 4 * Y^4)) :
  (X = 1 ∧ Y = 1) ∨ (X = -1 ∧ Y = -1) :=
sorry

end prime_number_solution_l116_116415


namespace grade3_trees_count_l116_116959

-- Declare the variables and types
variables (x y : ℕ)

-- Given conditions as definitions
def students_equation := (2 * x + y = 100)
def trees_equation := (9 * x + (13 / 2) * y = 566)
def avg_trees_grade3 := 4

-- Assert the problem statement
theorem grade3_trees_count (hx : students_equation x y) (hy : trees_equation x y) : 
  (avg_trees_grade3 * x = 84) :=
sorry

end grade3_trees_count_l116_116959


namespace probability_of_ending_at_multiple_of_3_l116_116453

noncomputable def probability_ends_at_multiple_of_3 : ℚ :=
let prob_start_multiple_3 := (5 / 15 : ℚ), -- Probability of starting at a multiple of 3
    prob_start_one_more_3 := (4 / 15 : ℚ), -- Probability of starting one more than a multiple of 3
    prob_start_one_less_3 := (5 / 15 : ℚ), -- Probability of starting one less than a multiple of 3
    prob_LL := (1 / 16 : ℚ),               -- Probability of "LL" outcome
    prob_RR := (9 / 16 : ℚ) in             -- Probability of "RR" outcome
  prob_start_multiple_3 * prob_LL +
  prob_start_one_more_3 * prob_RR +
  prob_start_one_less_3 * prob_LL

theorem probability_of_ending_at_multiple_of_3 :
  probability_ends_at_multiple_of_3 = (7 / 30 : ℚ) :=
sorry

end probability_of_ending_at_multiple_of_3_l116_116453


namespace roots_quadratic_fraction_l116_116148

theorem roots_quadratic_fraction :
  (∀ (x : ℝ), (Polynomial.eval x (Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 2 * Polynomial.X - Polynomial.C 4) = 0) → 
  let x1 := (-2 + Real.sqrt (4 + 16)) / (2 * 1)
  let x2 := (-2 - Real.sqrt (4 + 16)) / (2 * 1)
  (x1 + x2) / (x1 * x2) = -1 / 2) := 
sorry

end roots_quadratic_fraction_l116_116148


namespace system_of_equations_solution_l116_116655

theorem system_of_equations_solution :
  ∃ x y : ℝ, (2 * x + y = 6) ∧ (x - y = 3) ∧ (x = 3) ∧ (y = 0) :=
by
  sorry

end system_of_equations_solution_l116_116655


namespace arithmetical_puzzle_l116_116162

theorem arithmetical_puzzle (S I X T W E N : ℕ) 
  (h1 : S = 1) 
  (h2 : N % 2 = 0) 
  (h3 : (1 * 100 + I * 10 + X) * 3 = T * 1000 + W * 100 + E * 10 + N) 
  (h4 : ∀ (a b c d e f : ℕ), 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
        d ≠ e ∧ d ≠ f ∧
        e ≠ f) :
  T = 5 := sorry

end arithmetical_puzzle_l116_116162


namespace vertices_form_vertical_line_l116_116974

theorem vertices_form_vertical_line (a b k d : ℝ) (ha : 0 < a) (hk : 0 < k) :
  ∃ x, ∀ t : ℝ, ∃ y, (x = -b / (2 * a) ∧ y = - (b^2) / (4 * a) + k * t + d) :=
sorry

end vertices_form_vertical_line_l116_116974


namespace linear_function_no_second_quadrant_l116_116190

theorem linear_function_no_second_quadrant (x y : ℝ) (h : y = 2 * x - 3) :
  ¬ ((x < 0) ∧ (y > 0)) :=
by {
  sorry
}

end linear_function_no_second_quadrant_l116_116190


namespace blue_to_red_ratio_l116_116307

variable (B R : ℕ)

-- Conditions
def total_mugs : ℕ := B + R + 12 + 4
def yellow_mugs : ℕ := 12
def red_mugs : ℕ := yellow_mugs / 2
def other_color_mugs : ℕ := 4

theorem blue_to_red_ratio :
  total_mugs = 40 → R = red_mugs → (B / Nat.gcd B R) = 3 ∧ (R / Nat.gcd B R) = 1 :=
by
  intros h_total M_red
  sorry

end blue_to_red_ratio_l116_116307


namespace ratio_B_over_A_eq_one_l116_116475

theorem ratio_B_over_A_eq_one (A B : ℤ) (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 3 → 
  (A : ℝ) / (x + 3) + (B : ℝ) / (x * (x - 3)) = (x^3 - 3*x^2 + 15*x - 9) / (x^3 + x^2 - 9*x)) :
  (B : ℝ) / (A : ℝ) = 1 :=
sorry

end ratio_B_over_A_eq_one_l116_116475


namespace weight_of_each_soda_crate_l116_116410

-- Definitions based on conditions
def bridge_weight_limit := 20000
def empty_truck_weight := 12000
def number_of_soda_crates := 20
def dryer_weight := 3000
def number_of_dryers := 3
def fully_loaded_truck_weight := 24000
def soda_weight := 1000
def produce_weight := 2 * soda_weight
def total_cargo_weight := fully_loaded_truck_weight - empty_truck_weight

-- Lean statement to prove the weight of each soda crate
theorem weight_of_each_soda_crate :
  number_of_soda_crates * ((total_cargo_weight - (number_of_dryers * dryer_weight)) / 3) / number_of_soda_crates = 50 :=
by
  sorry

end weight_of_each_soda_crate_l116_116410


namespace valid_pic4_valid_pic5_l116_116831

-- Define the type for grid coordinates
structure Coord where
  x : ℕ
  y : ℕ

-- Define the function to check if two coordinates are adjacent by side
def adjacent (a b : Coord) : Prop :=
  (a.x = b.x ∧ (a.y = b.y + 1 ∨ a.y = b.y - 1)) ∨
  (a.y = b.y ∧ (a.x = b.x + 1 ∨ a.x = b.x - 1))

-- Define the coordinates for the pictures №4 and №5
def pic4_coords : List (ℕ × Coord) :=
  [(1, ⟨0, 0⟩), (2, ⟨1, 0⟩), (4, ⟨2, 0⟩), (3, ⟨0, 1⟩),
   (5, ⟨1, 1⟩), (6, ⟨2, 1⟩), (7, ⟨2, 2⟩), (8, ⟨1, 3⟩)]

def pic5_coords : List (ℕ × Coord) :=
  [(1, ⟨0, 0⟩), (2, ⟨0, 1⟩), (3, ⟨0, 2⟩), (4, ⟨0, 3⟩), (5, ⟨1, 3⟩)]

-- Define the validity condition for a picture
def valid_picture (coords : List (ℕ × Coord)) : Prop :=
  ∀ (n : ℕ) (c1 c2 : Coord), (n, c1) ∈ coords → (n + 1, c2) ∈ coords → adjacent c1 c2

-- The theorem to prove that pictures №4 and №5 are valid configurations
theorem valid_pic4 : valid_picture pic4_coords := sorry

theorem valid_pic5 : valid_picture pic5_coords := sorry

end valid_pic4_valid_pic5_l116_116831


namespace rice_bag_weight_l116_116240

theorem rice_bag_weight (r f : ℕ) (total_weight : ℕ) (h1 : 20 * r + 50 * f = 2250) (h2 : r = 2 * f) : r = 50 := 
by
  sorry

end rice_bag_weight_l116_116240


namespace expected_value_girls_left_of_boys_l116_116376

theorem expected_value_girls_left_of_boys :
  let boys := 10
      girls := 7
      students := boys + girls in
  (∀ (lineup : Finset (Fin students)), let event := { l : Finset (Fin students) | ∃ g : Fin girls, g < boys - 1} in
       ProbabilityTheory.expectation (λ p, (lineup ∩ event).card)) = 7 / 11 := 
sorry

end expected_value_girls_left_of_boys_l116_116376


namespace locus_of_midpoint_l116_116930

theorem locus_of_midpoint {P Q M : ℝ × ℝ} (hP_on_circle : P.1^2 + P.2^2 = 13)
  (hQ_perpendicular_to_y_axis : Q.1 = P.1) (h_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1^2 / (13 / 4)) + (M.2^2 / 13) = 1 := 
sorry

end locus_of_midpoint_l116_116930


namespace length_of_side_b_l116_116154

theorem length_of_side_b (B C : ℝ) (c b : ℝ) (hB : B = 45 * Real.pi / 180) (hC : C = 60 * Real.pi / 180) (hc : c = 1) :
  b = Real.sqrt 6 / 3 :=
by
  sorry

end length_of_side_b_l116_116154


namespace sugar_per_bar_l116_116238

theorem sugar_per_bar (bars_per_minute : ℕ) (sugar_per_2_minutes : ℕ)
  (h1 : bars_per_minute = 36)
  (h2 : sugar_per_2_minutes = 108) :
  (sugar_per_2_minutes / (bars_per_minute * 2) : ℚ) = 1.5 := 
by 
  sorry

end sugar_per_bar_l116_116238


namespace cube_side_length_l116_116195

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (h0 : s ≠ 0) : s = 6 :=
sorry

end cube_side_length_l116_116195


namespace interest_rate_first_part_eq_3_l116_116063

variable (T P1 P2 r2 I : ℝ)
variable (hT : T = 3400)
variable (hP1 : P1 = 1300)
variable (hP2 : P2 = 2100)
variable (hr2 : r2 = 5)
variable (hI : I = 144)

theorem interest_rate_first_part_eq_3 (r : ℝ) (h : (P1 * r) / 100 + (P2 * r2) / 100 = I) : r = 3 :=
by
  -- leaning in the proof
  sorry

end interest_rate_first_part_eq_3_l116_116063


namespace fever_above_threshold_l116_116943

-- Definitions as per conditions
def normal_temp : ℤ := 95
def temp_increase : ℤ := 10
def fever_threshold : ℤ := 100

-- Calculated new temperature
def new_temp := normal_temp + temp_increase

-- The proof statement, asserting the correct answer
theorem fever_above_threshold : new_temp - fever_threshold = 5 := 
by 
  sorry

end fever_above_threshold_l116_116943


namespace distinct_divisor_sum_l116_116835

theorem distinct_divisor_sum (n : ℕ) (x : ℕ) (h : x < n.factorial) :
  ∃ (k : ℕ) (d : Fin k → ℕ), (k ≤ n) ∧ (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ n.factorial) ∧ (x = Finset.sum Finset.univ d) :=
sorry

end distinct_divisor_sum_l116_116835


namespace simplify_and_evaluate_expression_l116_116468

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 5 + 1) : 
  ( ( (x^2 - 1) / x ) / (1 + 1 / x) ) = Real.sqrt 5 :=
by 
  sorry

end simplify_and_evaluate_expression_l116_116468


namespace trig_proof_1_trig_proof_2_l116_116756

variables {α : ℝ}

-- Given condition
def tan_alpha (a : ℝ) := Real.tan a = -3

-- Proof problem statement
theorem trig_proof_1 (h : tan_alpha α) :
  (3 * Real.sin α - 3 * Real.cos α) / (6 * Real.cos α + Real.sin α) = -4 := sorry

theorem trig_proof_2 (h : tan_alpha α) :
  1 / (Real.sin α * Real.cos α + 1 + Real.cos (2 * α)) = -10 := sorry

end trig_proof_1_trig_proof_2_l116_116756


namespace brothers_percentage_fewer_trees_l116_116048

theorem brothers_percentage_fewer_trees (total_trees initial_days brother_days : ℕ) (trees_per_day : ℕ) (total_brother_trees : ℕ) (percentage_fewer : ℕ):
  initial_days = 2 →
  brother_days = 3 →
  trees_per_day = 20 →
  total_trees = 196 →
  total_brother_trees = total_trees - (trees_per_day * initial_days) →
  percentage_fewer = ((total_brother_trees / brother_days - trees_per_day) * 100) / trees_per_day →
  percentage_fewer = 60 :=
by
  sorry

end brothers_percentage_fewer_trees_l116_116048


namespace max_right_angles_in_triangular_prism_l116_116046

theorem max_right_angles_in_triangular_prism 
  (n_triangles : ℕ) 
  (n_rectangles : ℕ) 
  (max_right_angles_triangle : ℕ) 
  (max_right_angles_rectangle : ℕ)
  (h1 : n_triangles = 2)
  (h2 : n_rectangles = 3)
  (h3 : max_right_angles_triangle = 1)
  (h4 : max_right_angles_rectangle = 4) : 
  (n_triangles * max_right_angles_triangle + n_rectangles * max_right_angles_rectangle = 14) :=
by
  sorry

end max_right_angles_in_triangular_prism_l116_116046


namespace fraction_simplification_l116_116650

theorem fraction_simplification :
  ∃ (p q : ℕ), p = 2021 ∧ q ≠ 0 ∧ gcd p q = 1 ∧ (1011 / 1010) - (1010 / 1011) = (p : ℚ) / q := 
sorry

end fraction_simplification_l116_116650


namespace area_AOC_is_1_l116_116627

noncomputable def point := (ℝ × ℝ) -- Define a point in 2D space

def vector_add (v1 v2 : point) : point :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_zero : point := (0, 0)

def scalar_mul (r : ℝ) (v : point) : point :=
  (r * v.1, r * v.2)

def vector_eq (v1 v2 : point) : Prop := 
  v1.1 = v2.1 ∧ v1.2 = v2.2

variables (A B C O : point)
variable (area_ABC : ℝ)

-- Conditions:
-- Point O is a point inside triangle ABC with an area of 4
-- \(\overrightarrow {OA} + \overrightarrow {OB} + 2\overrightarrow {OC} = \overrightarrow {0}\)
axiom condition_area : area_ABC = 4
axiom condition_vector : vector_eq (vector_add (vector_add O A) (vector_add O B)) (scalar_mul (-2) O)

-- Theorem to prove: the area of triangle AOC is 1
theorem area_AOC_is_1 : (area_ABC / 4) = 1 := 
sorry

end area_AOC_is_1_l116_116627


namespace exterior_angle_of_parallel_lines_l116_116962

theorem exterior_angle_of_parallel_lines (A B C x y : ℝ) (hAx : A = 40) (hBx : B = 90) (hCx : C = 40)
  (h_parallel : true)
  (h_triangle : x = 180 - A - C)
  (h_exterior_angle : y = 180 - x) :
  y = 80 := 
by
  sorry

end exterior_angle_of_parallel_lines_l116_116962


namespace standard_eq_of_hyperbola_l116_116405

-- Definitions of required variables and parameters
variables (a b : ℝ) (e : ℝ) (c : ℝ)
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom e_def : e = 2
axiom hyperbola_eqn : (y : ℝ) * (y / a)^2 - (x : ℝ) * (x / b)^2 = 1

noncomputable def find_hyperbola_standard_eq : Prop :=
  hyperbola_eqn ∧ -- Hyperbola with given conditions
  x^2 = 8 * y ∧ -- Parabola equation
  e = c / a ∧  -- Eccentricity definition
  c = 2 ∧ -- Focus definition from parabola
  c^2 = a^2 + b^2 ∧ -- Relationship between a, b, and c in a hyperbola
  y^2 - x^2 / 3 = 1 -- Standard equation of hyperbola

-- The statement only - proof to be completed
theorem standard_eq_of_hyperbola :
  find_hyperbola_standard_eq :=
sorry

end standard_eq_of_hyperbola_l116_116405


namespace treasure_coins_l116_116502

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l116_116502


namespace customer_paid_l116_116349

def cost_price : ℝ := 7999.999999999999
def percentage_markup : ℝ := 0.10
def selling_price (cp : ℝ) (markup : ℝ) := cp + cp * markup

theorem customer_paid :
  selling_price cost_price percentage_markup = 8800 :=
by
  sorry

end customer_paid_l116_116349


namespace Tim_took_out_11_rulers_l116_116213

-- Define the initial number of rulers
def initial_rulers := 14

-- Define the number of rulers left in the drawer
def rulers_left := 3

-- Define the number of rulers taken by Tim
def rulers_taken := initial_rulers - rulers_left

-- Statement to prove that the number of rulers taken by Tim is indeed 11
theorem Tim_took_out_11_rulers : rulers_taken = 11 := by
  sorry

end Tim_took_out_11_rulers_l116_116213


namespace probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18_l116_116171

open Real

noncomputable def probability_event : ℝ :=
  ((327.61 - 324) / (361 - 324))

theorem probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18 :
  probability_event = 361 / 3700 :=
by
  -- Conditions and calculations supplied in the problem
  sorry

end probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18_l116_116171


namespace average_temperature_week_l116_116064

theorem average_temperature_week 
  (T_sun : ℝ := 40)
  (T_mon : ℝ := 50)
  (T_tue : ℝ := 65)
  (T_wed : ℝ := 36)
  (T_thu : ℝ := 82)
  (T_fri : ℝ := 72)
  (T_sat : ℝ := 26) :
  (T_sun + T_mon + T_tue + T_wed + T_thu + T_fri + T_sat) / 7 = 53 :=
by
  sorry

end average_temperature_week_l116_116064


namespace value_of_x_l116_116659

variable (x y z : ℕ)

theorem value_of_x : x = 10 :=
  assume h1 : x = y / 2,
  assume h2 : y = z / 4,
  assume h3 : z = 80,
  sorry

end value_of_x_l116_116659


namespace charge_increase_percentage_l116_116471

variable (P R G : ℝ)

def charge_relation_1 : Prop := P = 0.45 * R
def charge_relation_2 : Prop := P = 0.90 * G

theorem charge_increase_percentage (h1 : charge_relation_1 P R) (h2 : charge_relation_2 P G) : 
  (R/G - 1) * 100 = 100 :=
by
  sorry

end charge_increase_percentage_l116_116471


namespace angle_in_second_quadrant_l116_116927

theorem angle_in_second_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.sin α > 0) : 
    α ∈ Set.Ioo (π / 2) π := 
    sorry

end angle_in_second_quadrant_l116_116927


namespace snow_white_last_trip_dwarfs_l116_116643

-- Definitions for the conditions
def original_lineup := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]
def only_snow_white_can_row := True
def boat_capacity_snow_white_and_dwarfs := 3
def dwarfs_quarrel_if_adjacent (d1 d2 : String) : Prop :=
  let index_d1 := List.indexOf original_lineup d1
  let index_d2 := List.indexOf original_lineup d2
  abs (index_d1 - index_d2) = 1

-- Theorem to prove the correct answer
theorem snow_white_last_trip_dwarfs :
  let last_trip_dwarfs := ["Grumpy", "Bashful", "Sneezy"]
  ∃ (trip : List String), trip = last_trip_dwarfs ∧ 
  ∀ (d1 d2 : String), d1 ∈ trip → d2 ∈ trip → d1 ≠ d2 → ¬dwarfs_quarrel_if_adjacent d1 d2 :=
by
  sorry

end snow_white_last_trip_dwarfs_l116_116643


namespace merchant_product_quantities_l116_116699

theorem merchant_product_quantities
  (x p1 : ℝ)
  (h1 : 4000 = x * p1)
  (h2 : 8800 = 2 * x * (p1 + 4))
  (h3 : (8800 / (2 * x)) - (4000 / x) = 4):
  x = 100 ∧ 2 * x = 200 :=
by sorry

end merchant_product_quantities_l116_116699


namespace smallest_integer_sum_of_squares_and_cubes_infinite_integers_sum_of_squares_and_cubes_l116_116418

theorem smallest_integer_sum_of_squares_and_cubes :
  ∃ (n : ℕ) (a b c d : ℕ), n > 2 ∧ n = a^2 + b^2 ∧ n = c^3 + d^3 ∧
  ∀ (m : ℕ) (x y u v : ℕ), (m > 2 ∧ m = x^2 + y^2 ∧ m = u^3 + v^3) → n ≤ m := 
sorry

theorem infinite_integers_sum_of_squares_and_cubes :
  ∀ (k : ℕ), ∃ (n : ℕ) (a b c d : ℕ), n = 1 + 2^(6*k) ∧ n = a^2 + b^2 ∧ n = c^3 + d^3 :=
sorry

end smallest_integer_sum_of_squares_and_cubes_infinite_integers_sum_of_squares_and_cubes_l116_116418


namespace find_total_coins_l116_116519

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l116_116519


namespace constant_slope_ratio_fixed_line_intersection_l116_116433

noncomputable def ellipse_c_eq : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (eccentricity a b) = (sqrt 3 / 2) ∧ right_focus a b = (sqrt 3, 0) ∧
  ellipse a b = set_of (λ p : ℝ × ℝ, (p.1^2 / 4) + p.2^2 = 1)

theorem constant_slope_ratio (a b : ℝ) (e : elliptical e a b = (sqrt 3 / 2)) (focus : elliptical e (sqrt 3, 0)) : Prop :=
  ∀ {P Q A B D : ℝ × ℝ} (k1 k2 : ℝ),
    (P ∈ ellipse a b) →
    (Q ∈ ellipse a b) →
    A = (-a, 0) →
    B = (a, 0) →
    D = (1, 0) →
    (line_intersects_c A P k1 k2) →
    (line_intersects_c B Q k2 k1) →
    (non_zero_slope A P) →
    (non_zero_slope B Q) →
    (k1 / k2) = (1 / 3)

theorem fixed_line_intersection (a b : ℝ) (e : elliptical e a b = (sqrt 3 / 2)) (focus : elliptical e (sqrt 3, 0)) : Prop :=
  ∀ {P Q A B D : ℝ × ℝ} (M : ℝ × ℝ) (k1 k2 : ℝ),
    (P ∈ ellipse a b) →
    (Q ∈ ellipse a b) →
    A = (-a, 0) →
    B = (a, 0) →
    D = (1, 0) →
    (line_intersects_c A P k1 k2) →
    (line_intersects_c B Q k2 k1) →
    (non_zero_slope A P) →
    (non_zero_slope B Q) →
    intersection_point A P B Q M →
    (M.1 = 4)

end constant_slope_ratio_fixed_line_intersection_l116_116433


namespace length_of_longer_leg_of_smallest_triangle_l116_116411

theorem length_of_longer_leg_of_smallest_triangle 
  (hypotenuse_largest : ℝ) 
  (h1 : hypotenuse_largest = 10)
  (h45 : ∀ hyp, (hyp / Real.sqrt 2 / Real.sqrt 2 / Real.sqrt 2 / Real.sqrt 2) = hypotenuse_largest / 4) :
  (hypotenuse_largest / 4) = 5 / 2 := by
  sorry

end length_of_longer_leg_of_smallest_triangle_l116_116411


namespace man_rate_in_still_water_l116_116093

theorem man_rate_in_still_water (speed_with_stream speed_against_stream : ℝ)
  (h1 : speed_with_stream = 22) (h2 : speed_against_stream = 10) :
  (speed_with_stream + speed_against_stream) / 2 = 16 := by
  sorry

end man_rate_in_still_water_l116_116093


namespace trajectory_of_moving_circle_l116_116030

-- Definitions for the given circles C1 and C2
def Circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def Circle2 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Prove the trajectory of the center of the moving circle M
theorem trajectory_of_moving_circle (x y : ℝ) :
  ((∃ x_center y_center : ℝ, Circle1 x_center y_center ∧ Circle2 x_center y_center ∧ 
  -- Tangency conditions for Circle M
  (x - x_center)^2 + y^2 = (x_center - 2)^2 + y^2 ∧ (x - x_center)^2 + y^2 = (x_center + 2)^2 + y^2)) →
  (x = 0 ∨ x^2 - y^2 / 3 = 1) := 
sorry

end trajectory_of_moving_circle_l116_116030


namespace valid_integers_count_l116_116128

theorem valid_integers_count : 
  ∃ count : ℕ, count = 96 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → (Nat.factorial (n ^ 2 - 4) / (Nat.factorial n) ^ (n - 2)).denom = 1 → n ≥ 5) :=
by
  -- Mathematical proof skipped
  sorry

end valid_integers_count_l116_116128


namespace regular_polygon_sides_l116_116805

theorem regular_polygon_sides (θ : ℝ) (h : θ = 20) : 360 / θ = 18 := by
  sorry

end regular_polygon_sides_l116_116805


namespace fruit_basket_l116_116662

-- Define the quantities and their relationships
variables (O A B P : ℕ)

-- State the conditions
def condition1 : Prop := A = O - 2
def condition2 : Prop := B = 3 * A
def condition3 : Prop := P = B / 2
def condition4 : Prop := O + A + B + P = 28

-- State the theorem
theorem fruit_basket (h1 : condition1 O A) (h2 : condition2 A B) (h3 : condition3 B P) (h4 : condition4 O A B P) : O = 6 :=
sorry

end fruit_basket_l116_116662


namespace min_S_value_l116_116293

noncomputable def S (x y z : ℝ) : ℝ := (1 + z) / (2 * x * y * z)

theorem min_S_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x^2 + y^2 + z^2 = 1) :
  S x y z ≥ 4 := 
sorry

end min_S_value_l116_116293


namespace pirate_treasure_l116_116511

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l116_116511


namespace cds_probability_l116_116393

def probability (total favorable : ℕ) : ℚ := favorable / total

theorem cds_probability :
  probability 120 24 = 1 / 5 :=
by
  sorry

end cds_probability_l116_116393


namespace m_range_l116_116615

noncomputable def G (x : ℝ) (m : ℝ) : ℝ := (8 * x^2 + 22 * x + 5 * m) / 8

theorem m_range (m : ℝ) : 2.5 ≤ m ∧ m ≤ 3.5 ↔ m = 121 / 40 := by
  sorry

end m_range_l116_116615


namespace probability_of_exactly_one_shortening_l116_116879

-- Define the conditions and problem
def shorten_sequence_exactly_one_probability (n : ℕ) (p : ℝ) : ℝ :=
  nat.choose n 1 * p^1 * (1 - p)^(n - 1)

-- Given conditions
def problem_8a := shorten_sequence_exactly_one_probability 2014 0.1 ≈ 1.564e-90

-- Lean statement
theorem probability_of_exactly_one_shortening :
  problem_8a := sorry

end probability_of_exactly_one_shortening_l116_116879


namespace cube_side_length_l116_116203

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end cube_side_length_l116_116203


namespace intersection_eq_l116_116441

def U : Set ℝ := {x : ℝ | True}
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x ≥ 1}
def CU_N : Set ℝ := {x : ℝ | x < 1}

theorem intersection_eq : M ∩ CU_N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l116_116441


namespace pirates_treasure_l116_116522

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l116_116522


namespace work_done_days_l116_116383

theorem work_done_days (a_days : ℕ) (b_days : ℕ) (together_days : ℕ) (a_work_done : ℚ) (b_work_done : ℚ) (together_work : ℚ) : 
  a_days = 12 ∧ b_days = 15 ∧ together_days = 5 ∧ 
  a_work_done = 1/12 ∧ b_work_done = 1/15 ∧ together_work = 3/4 → 
  ∃ days : ℚ, a_days > 0 ∧ b_days > 0 ∧ together_days > 0 ∧ days = 3 := 
  sorry

end work_done_days_l116_116383


namespace area_of_region_l116_116574

noncomputable def region_area : ℝ :=
  sorry

theorem area_of_region :
  region_area = sorry := 
sorry

end area_of_region_l116_116574


namespace value_of_x_l116_116658

theorem value_of_x (x y z : ℝ) (h1 : x = (1 / 2) * y) (h2 : y = (1 / 4) * z) (h3 : z = 80) : x = 10 := by
  sorry

end value_of_x_l116_116658


namespace AC_diagonal_length_l116_116603

noncomputable def AC_length (AD DC : ℝ) (angle_ADC : ℝ) : ℝ :=
  Real.sqrt (AD^2 + DC^2 - 2 * AD * DC * Real.cos angle_ADC)

theorem AC_diagonal_length :
  let AD := 15
  let DC := 15
  let angle_ADC := 2 * Real.pi / 3 -- 120 degrees in radians
  AC_length AD DC angle_ADC = 15 :=
by
  have h : AC_length 15 15 (2 * Real.pi / 3) = Real.sqrt (15^2 + 15^2 - 2 * 15 * 15 * Real.cos (2 * Real.pi / 3)),
  { unfold AC_length },
  rw h,
  have h_cos : Real.cos (2 * Real.pi / 3) = -1 / 2,
  { sorry }, -- intermediate steps to find cosine of 120 degrees
  rw [h_cos, sq],
  norm_num,
  refl

end AC_diagonal_length_l116_116603


namespace calculate_f_g_l116_116941

noncomputable def f (x : ℕ) : ℕ := 4 * x + 3
noncomputable def g (x : ℕ) : ℕ := (x + 2) ^ 2

theorem calculate_f_g : f (g 3) = 103 :=
by 
  -- Proof omitted.
  sorry

end calculate_f_g_l116_116941


namespace value_of_x_l116_116686

theorem value_of_x (x : ℕ) : (8^4 + 8^4 + 8^4 = 2^x) → x = 13 :=
by
  sorry

end value_of_x_l116_116686


namespace general_formula_sum_first_n_terms_l116_116286

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

axiom a_initial : a 1 = 1
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n * (1 + 1 / n)

theorem general_formula : ∀ n : ℕ, n > 0 → a n = n * 3^(n - 1) :=
by
  sorry

theorem sum_first_n_terms : ∀ n : ℕ, S n = (2 * n - 1) * 3^n + 1 / 4 :=
by
  sorry

end general_formula_sum_first_n_terms_l116_116286


namespace shortest_distance_parabola_line_l116_116874

theorem shortest_distance_parabola_line :
  ∃ (P Q : ℝ × ℝ), P.2 = P.1^2 - 6 * P.1 + 15 ∧ Q.2 = 2 * Q.1 - 7 ∧
  ∀ (p q : ℝ × ℝ), p.2 = p.1^2 - 6 * p.1 + 15 → q.2 = 2 * q.1 - 7 → 
  dist p q ≥ dist P Q :=
sorry

end shortest_distance_parabola_line_l116_116874


namespace choir_min_students_l116_116249

theorem choir_min_students : ∃ n : ℕ, (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) ∧ n = 990 :=
by
  sorry

end choir_min_students_l116_116249


namespace sum_of_grid_numbers_l116_116020

theorem sum_of_grid_numbers (A E: ℕ) (S: ℕ) 
    (hA: A = 2) 
    (hE: E = 3)
    (h1: ∃ B : ℕ, 2 + B = S ∧ 3 + B = S)
    (h2: ∃ D : ℕ, 2 + D = S ∧ D + 3 = S)
    (h3: ∃ F : ℕ, 3 + F = S ∧ F + 3 = S)
    (h4: ∃ G H I: ℕ, 
         2 + G = S ∧ G + H = S ∧ H + C = S ∧ 
         3 + H = S ∧ E + I = S ∧ H + I = S):
  A + B + C + D + E + F + G + H + I = 22 := 
by 
  sorry

end sum_of_grid_numbers_l116_116020


namespace total_profit_at_100_max_profit_price_l116_116697

noncomputable def sales_volume (x : ℝ) : ℝ := 15 - 0.1 * x
noncomputable def floating_price (S : ℝ) : ℝ := 10 / S
noncomputable def supply_price (x : ℝ) : ℝ := 30 + floating_price (sales_volume x)
noncomputable def profit_per_set (x : ℝ) : ℝ := x - supply_price x
noncomputable def total_profit (x : ℝ) : ℝ := profit_per_set x * sales_volume x

-- Theorem 1: Total profit when each set is priced at 100 yuan is 340 ten thousand yuan
theorem total_profit_at_100 : total_profit 100 = 340 := by
  sorry

-- Theorem 2: The price per set that maximizes profit per set is 140 yuan
theorem max_profit_price : ∃ x, profit_per_set x = 100 ∧ x = 140 := by
  sorry

end total_profit_at_100_max_profit_price_l116_116697


namespace tangent_line_equation_l116_116189

theorem tangent_line_equation (P : ℝ × ℝ) (hP : P = (-4, -3)) :
  ∃ (a b c : ℝ), a * -4 + b * -3 + c = 0 ∧ a * a + b * b = (5:ℝ)^2 ∧ 
                 a = 4 ∧ b = 3 ∧ c = 25 := 
sorry

end tangent_line_equation_l116_116189


namespace simplify_expression_is_one_fourth_l116_116636

noncomputable def fourth_root (x : ℝ) : ℝ := x ^ (1 / 4)
noncomputable def square_root (x : ℝ) : ℝ := x ^ (1 / 2)
noncomputable def simplified_expression : ℝ := (fourth_root 81 - square_root 12.25) ^ 2

theorem simplify_expression_is_one_fourth : simplified_expression = 1 / 4 := 
by
  sorry

end simplify_expression_is_one_fourth_l116_116636


namespace rhombus_side_length_l116_116767

theorem rhombus_side_length (area d1 d2 side : ℝ) (h_area : area = 24)
(h_d1 : d1 = 6) (h_other_diag : d2 * 6 = 48) (h_side : side = Real.sqrt (3^2 + 4^2)) :
  side = 5 :=
by
  -- This is where the proof would go
  sorry

end rhombus_side_length_l116_116767


namespace like_terms_exponent_l116_116939

theorem like_terms_exponent (x y : ℝ) (n : ℕ) : 
  (∀ (a b : ℝ), a * x ^ 3 * y ^ (n - 1) = b * x ^ 3 * y ^ 1 → n = 2) :=
by
  sorry

end like_terms_exponent_l116_116939


namespace repeating_decimal_fraction_l116_116799

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l116_116799


namespace last_digit_of_sum_is_four_l116_116459

theorem last_digit_of_sum_is_four (x y z : ℕ)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 0 ≤ y ∧ y ≤ 9)
  (hz : 0 ≤ z ∧ z ≤ 9)
  (h : 1950 ≤ 200 * x + 11 * y + 11 * z ∧ 200 * x + 11 * y + 11 * z < 2000) :
  (200 * x + 11 * y + 11 * z) % 10 = 4 :=
sorry

end last_digit_of_sum_is_four_l116_116459


namespace a_n_is_square_of_rational_inequality_holds_l116_116129

namespace ArithmeticGeometricMeans

noncomputable def A (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def G (a b : ℝ) : ℝ := Real.sqrt (a * b)

def a_seq (n : ℕ) : ℕ → ℝ 
| 0       := 0
| 1       := 1
| (m+2) := A (A (a_seq m) (a_seq (m + 1))) (G (a_seq m) (a_seq (m + 1)))

theorem a_n_is_square_of_rational (n : ℕ) (h₀ : n > 0) : ∃ b_n : ℚ, (0 ≤ b_n) ∧ (a_seq n = b_n ^ 2) :=
sorry

theorem inequality_holds (n : ℕ) (h₀ : n > 0) : abs (a_seq n - 2 / 3) < 1 / 2^n :=
sorry

end ArithmeticGeometricMeans

end a_n_is_square_of_rational_inequality_holds_l116_116129


namespace locus_of_point_P_l116_116288

theorem locus_of_point_P (P : ℝ × ℝ) (M N : ℝ × ℝ)
  (hxM : M = (-2, 0))
  (hxN : N = (2, 0))
  (hxPM : P.fst ^ 2 + (P.snd - 0) ^ 2 = xPM)
  (hxPN : P.fst ^ 2 + (P.snd - 0) ^ 2 = xPN)
  : P.fst ^ 2 + P.snd ^ 2 = 4 ∧ P.fst ≠ 2 ∧ P.fst ≠ -2 :=
by
  -- proof omitted
  sorry

end locus_of_point_P_l116_116288


namespace part1_part2_l116_116334

variable (a m : ℝ)

def f (x : ℝ) : ℝ := 2 * |x - 1| - a

theorem part1 (h : ∃ x, f a x - 2 * |x - 7| ≤ 0) : a ≥ -12 :=
sorry

theorem part2 (h : ∀ x, f 1 x + |x + 7| ≥ m) : m ≤ 7 :=
sorry

end part1_part2_l116_116334


namespace neg_sqrt_17_estimate_l116_116741

theorem neg_sqrt_17_estimate : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end neg_sqrt_17_estimate_l116_116741


namespace inequality_proof_l116_116290

theorem inequality_proof
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y * z) / (Real.sqrt (2 * x^2 * (y + z))) + 
  (y^2 + z * x) / (Real.sqrt (2 * y^2 * (z + x))) + 
  (z^2 + x * y) / (Real.sqrt (2 * z^2 * (x + y))) ≥ 1 := 
sorry

end inequality_proof_l116_116290


namespace min_value_expression_l116_116918

theorem min_value_expression : ∀ (x y : ℝ), ∃ z : ℝ, z ≥ 3*x^2 + 2*x*y + 3*y^2 + 5 ∧ z = 5 :=
by
  sorry

end min_value_expression_l116_116918


namespace inequality_solution_set_range_of_m_l116_116302

noncomputable def f (x : ℝ) : ℝ := |x - 1|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m

theorem inequality_solution_set :
  {x : ℝ | f x + x^2 - 1 > 0} = {x : ℝ | x > 1 ∨ x < 0} :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x < g x m) → m > 4 :=
sorry

end inequality_solution_set_range_of_m_l116_116302


namespace perimeter_area_ratio_le_8_l116_116058

/-- Let \( S \) be a shape in the plane obtained as a union of finitely many unit squares.
    The perimeter of a single unit square is 4 and its area is 1.
    Prove that the ratio of the perimeter \( P \) and the area \( A \) of \( S \)
    is at most 8, i.e., \(\frac{P}{A} \leq 8\). -/
theorem perimeter_area_ratio_le_8
  (S : Set (ℝ × ℝ)) 
  (unit_square : ∀ (x y : ℝ), (x, y) ∈ S → (x + 1, y + 1) ∈ S ∧ (x + 1, y) ∈ S ∧ (x, y + 1) ∈ S ∧ (x, y) ∈ S)
  (P A : ℝ)
  (unit_square_perimeter : ∀ (x y : ℝ), (x, y) ∈ S → P = 4)
  (unit_square_area : ∀ (x y : ℝ), (x, y) ∈ S → A = 1) :
  P / A ≤ 8 :=
sorry

end perimeter_area_ratio_le_8_l116_116058


namespace side_length_of_cube_l116_116206

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end side_length_of_cube_l116_116206


namespace remaining_amount_is_9_l116_116010

-- Define the original prices of the books
def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00

-- Define the discount rate for the first two books
def discount_rate : ℝ := 0.25

-- Define the total cost without discount
def total_cost_without_discount := book1_price + book2_price + book3_price + book4_price

-- Calculate the discounts for the first two books
def book1_discount := book1_price * discount_rate
def book2_discount := book2_price * discount_rate

-- Calculate the discounted prices for the first two books
def discounted_book1_price := book1_price - book1_discount
def discounted_book2_price := book2_price - book2_discount

-- Calculate the total cost of the books with discounts applied
def total_cost_with_discount := discounted_book1_price + discounted_book2_price + book3_price + book4_price

-- Define the free shipping threshold
def free_shipping_threshold : ℝ := 50.00

-- Calculate the remaining amount Connor needs to spend
def remaining_amount_to_spend := free_shipping_threshold - total_cost_with_discount

-- State the theorem
theorem remaining_amount_is_9 : remaining_amount_to_spend = 9.00 := by
  -- we would provide the proof here
  sorry

end remaining_amount_is_9_l116_116010


namespace tips_fraction_of_salary_l116_116250

theorem tips_fraction_of_salary (S T x : ℝ) (h1 : T = x * S) 
  (h2 : T / (S + T) = 1 / 3) : x = 1 / 2 := by
  sorry

end tips_fraction_of_salary_l116_116250


namespace probability_at_least_one_l116_116571

theorem probability_at_least_one (p1 p2 : ℝ) (hp1 : 0 ≤ p1) (hp2 : 0 ≤ p2) (hp1p2 : p1 ≤ 1) (hp2p2 : p2 ≤ 1)
  (h0 : 0 ≤ 1 - p1) (h1 : 0 ≤ 1 - p2) (h2 : 1 - (1 - p1) ≥ 0) (h3 : 1 - (1 - p2) ≥ 0) :
  1 - (1 - p1) * (1 - p2) = 1 - (1 - p1) * (1 - p2) := by
  sorry

end probability_at_least_one_l116_116571


namespace max_value_of_sin2A_tan2B_l116_116325

-- Definitions for the trigonometric functions and angles in triangle ABC
variables {A B C : ℝ}

-- Condition: sin^2 A + sin^2 B = sin^2 C - sqrt 2 * sin A * sin B
def condition (A B C : ℝ) : Prop :=
  (Real.sin A) ^ 2 + (Real.sin B) ^ 2 = (Real.sin C) ^ 2 - Real.sqrt 2 * (Real.sin A) * (Real.sin B)

-- Question: Find the maximum value of sin 2A * tan^2 B
noncomputable def target (A B : ℝ) : ℝ :=
  Real.sin (2 * A) * (Real.tan B) ^ 2

-- The proof statement
theorem max_value_of_sin2A_tan2B (h : condition A B C) : ∃ (max_val : ℝ), max_val = 3 - 2 * Real.sqrt 2 ∧ ∀ (x : ℝ), target A x ≤ max_val := 
sorry

end max_value_of_sin2A_tan2B_l116_116325


namespace inequality_proof_l116_116312

theorem inequality_proof (x y : ℝ) (h : x * y < 0) : abs (x + y) < abs (x - y) :=
sorry

end inequality_proof_l116_116312


namespace math_proof_problem_l116_116765

namespace Proofs

-- Definition of the arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℤ) : Prop := 
  ∀ m n, a n = a m + (n - m) * (a (m + 1) - a m)

-- Conditions for the arithmetic sequence
def a_conditions (a : ℕ → ℤ) : Prop := 
  a 3 = -6 ∧ a 6 = 0

-- Definition of the geometric sequence {b_n}
def geometric_seq (b : ℕ → ℤ) : Prop := 
  ∃ q, ∀ n, b (n + 1) = q * b n

-- Conditions for the geometric sequence
def b_conditions (b a : ℕ → ℤ) : Prop := 
  b 1 = -8 ∧ b 2 = a 1 + a 2 + a 3

-- The general formula for {a_n}
def a_formula (a : ℕ → ℤ) :=
  ∀ n, a n = 2 * n - 12

-- The sum formula of the first n terms of {b_n}
def S_n_formula (b : ℕ → ℤ) (S_n : ℕ → ℤ) :=
  ∀ n, S_n n = 4 * (1 - 3^n)

-- The main theorem combining all
theorem math_proof_problem (a b : ℕ → ℤ) (S_n : ℕ → ℤ) :
  arithmetic_seq a →
  a_conditions a →
  geometric_seq b →
  b_conditions b a →
  (a_formula a ∧ S_n_formula b S_n) :=
by 
  sorry

end Proofs

end math_proof_problem_l116_116765


namespace total_water_capacity_l116_116887

-- Define the given conditions as constants
def numTrucks : ℕ := 5
def tanksPerTruck : ℕ := 4
def capacityPerTank : ℕ := 200

-- Define the claim as a theorem
theorem total_water_capacity :
  numTrucks * (tanksPerTruck * capacityPerTank) = 4000 :=
by
  sorry

end total_water_capacity_l116_116887


namespace multiplicative_inverse_l116_116732

theorem multiplicative_inverse (a b n : ℤ) (h₁ : a = 208) (h₂ : b = 240) (h₃ : n = 307) : 
  (a * b) % n = 1 :=
by
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end multiplicative_inverse_l116_116732


namespace num_three_digit_integers_with_zero_in_units_place_divisible_by_30_l116_116445

noncomputable def countThreeDigitMultiplesOf30WithZeroInUnitsPlace : ℕ :=
  let a := 120
  let d := 30
  let l := 990
  (l - a) / d + 1

theorem num_three_digit_integers_with_zero_in_units_place_divisible_by_30 :
  countThreeDigitMultiplesOf30WithZeroInUnitsPlace = 30 := by
  sorry

end num_three_digit_integers_with_zero_in_units_place_divisible_by_30_l116_116445


namespace sum_squares_inequality_l116_116354

theorem sum_squares_inequality {a b c : ℝ} 
  (h1 : a > 0)
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := 
sorry

end sum_squares_inequality_l116_116354


namespace max_imaginary_part_of_roots_l116_116559

noncomputable def find_phi : Prop :=
  ∃ z : ℂ, z^6 - z^4 + z^2 - 1 = 0 ∧ (∀ w : ℂ, w^6 - w^4 + w^2 - 1 = 0 → z.im ≤ w.im) ∧ z.im = Real.sin (Real.pi / 4)

theorem max_imaginary_part_of_roots : find_phi :=
sorry

end max_imaginary_part_of_roots_l116_116559


namespace mike_owes_correct_amount_l116_116970

variables (dollars_per_room rooms_cleaned total_amount : ℚ)

def mike_owes_jennifer (d : ℚ) (r : ℚ) : ℚ := d * r

theorem mike_owes_correct_amount :
  mike_owes_jennifer (13/3) (8/5) = 104/15 :=
by
  sorry

end mike_owes_correct_amount_l116_116970


namespace greatest_k_for_factorial_div_l116_116333

-- Definitions for conditions in the problem
def a : Nat := Nat.factorial 100
noncomputable def b (k : Nat) : Nat := 100^k

-- Statement to prove the greatest value of k for which b is a factor of a
theorem greatest_k_for_factorial_div (k : Nat) : 
  (∀ m : Nat, (m ≤ k → b m ∣ a) ↔ m ≤ 12) := 
by
  sorry

end greatest_k_for_factorial_div_l116_116333


namespace solution_of_modified_system_l116_116043

theorem solution_of_modified_system
  (a b x y : ℝ)
  (h1 : 2*a*3 + 3*4 = 18)
  (h2 : -3 + 5*b*4 = 17)
  : (x + y = 7 ∧ x - y = -1) → (2*a*(x+y) + 3*(x-y) = 18 ∧ (x+y) - 5*b*(x-y) = -17) → (x = (7 / 2) ∧ y = (-1 / 2)) :=
by
sorry

end solution_of_modified_system_l116_116043


namespace ordered_pair_correct_l116_116074

def find_ordered_pair (s m : ℚ) : Prop :=
  (∀ t : ℚ, (∃ x y : ℚ, x = -3 + t * m ∧ y = s + t * (-7) ∧ y = (3/4) * x + 5))
  ∧ s = 11/4 ∧ m = -28/3

theorem ordered_pair_correct :
  find_ordered_pair (11/4) (-28/3) :=
by
  sorry

end ordered_pair_correct_l116_116074


namespace months_passed_l116_116727

-- Let's define our conditions in mathematical terms
def received_bones (months : ℕ) : ℕ := 10 * months
def buried_bones : ℕ := 42
def available_bones : ℕ := 8
def total_bones (months : ℕ) : Prop := received_bones months = buried_bones + available_bones

-- We need to prove that the number of months (x) satisfies the condition
theorem months_passed (x : ℕ) : total_bones x → x = 5 :=
by
  sorry

end months_passed_l116_116727


namespace sector_area_is_nine_l116_116955

-- Given the conditions: the perimeter of the sector is 12 cm and the central angle is 2 radians
def sector_perimeter_radius (r : ℝ) :=
  4 * r = 12

def sector_angle : ℝ := 2

-- Prove that the area of the sector is 9 cm²
theorem sector_area_is_nine (r : ℝ) (s : ℝ) (h : sector_perimeter_radius r) (h_angle : sector_angle = 2) :
  s = 9 :=
by
  sorry

end sector_area_is_nine_l116_116955


namespace work_completion_time_l116_116313

theorem work_completion_time (a b c : ℕ) (ha : a = 36) (hb : b = 18) (hc : c = 6) : (1 / (1 / a + 1 / b + 1 / c) = 4) := by
  sorry

end work_completion_time_l116_116313


namespace general_term_formula_l116_116430

theorem general_term_formula (S : ℕ → ℤ) (a : ℕ → ℤ) : 
  (∀ n, S n = 3 * n ^ 2 - 2 * n) → 
  (∀ n ≥ 2, a n = S n - S (n - 1)) ∧ a 1 = S 1 → 
  ∀ n, a n = 6 * n - 5 := 
by
  sorry

end general_term_formula_l116_116430


namespace third_intermission_served_l116_116911

def total_served : ℚ :=  0.9166666666666666
def first_intermission : ℚ := 0.25
def second_intermission : ℚ := 0.4166666666666667

theorem third_intermission_served : first_intermission + second_intermission ≤ total_served →
  (total_served - (first_intermission + second_intermission)) = 0.25 :=
by
  sorry

end third_intermission_served_l116_116911


namespace joan_games_l116_116054

theorem joan_games (last_year_games this_year_games total_games : ℕ)
  (h1 : last_year_games = 9)
  (h2 : total_games = 13)
  : this_year_games = total_games - last_year_games → this_year_games = 4 := 
by
  intros h
  rw [h1, h2] at h
  exact h

end joan_games_l116_116054


namespace necessary_condition_l116_116172

theorem necessary_condition (a b : ℝ) (h : b ≠ 0) (h2 : a > b) (h3 : b > 0) : (1 / a < 1 / b) :=
sorry

end necessary_condition_l116_116172


namespace range_of_m_l116_116143

noncomputable def quadratic_function (m x : ℝ) : ℝ := (m-2) * x^2 + 2 * m * x - (3 - m)

theorem range_of_m (m : ℝ) (h_vertex_third_quadrant : (-(m) / (m-2) < 0) ∧ ((-5)*m + 6) / (m-2) < 0)
                   (h_parabola_opens_upwards : m - 2 > 0)
                   (h_intersects_negative_y_axis : m < 3) : 2 < m ∧ m < 3 :=
by {
    sorry
}

end range_of_m_l116_116143


namespace neg_sqrt_17_bounds_l116_116739

theorem neg_sqrt_17_bounds :
  (16 < 17) ∧ (17 < 25) ∧ (16 = 4^2) ∧ (25 = 5^2) ∧ (4 < Real.sqrt 17) ∧ (Real.sqrt 17 < 5) →
  (-5 < -Real.sqrt 17) ∧ (-Real.sqrt 17 < -4) :=
by
  sorry

end neg_sqrt_17_bounds_l116_116739


namespace gas_station_constant_l116_116068

structure GasStationData where
  amount : ℝ
  unit_price : ℝ
  price_per_yuan_per_liter : ℝ

theorem gas_station_constant (data : GasStationData) (h1 : data.amount = 116.64) (h2 : data.unit_price = 18) (h3 : data.price_per_yuan_per_liter = 6.48) : data.unit_price = 18 :=
sorry

end gas_station_constant_l116_116068


namespace relationship_between_a_b_c_l116_116310

noncomputable def a : ℝ := 1 / 3
noncomputable def b : ℝ := Real.sin (1 / 3)
noncomputable def c : ℝ := 1 / Real.pi

theorem relationship_between_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_between_a_b_c_l116_116310


namespace problem_l116_116061

theorem problem (m : ℕ) (h : m = 16^2023) : m / 8 = 2^8089 :=
by {
  sorry
}

end problem_l116_116061


namespace time_taken_by_Arun_to_cross_train_B_l116_116096

structure Train :=
  (length : ℕ)
  (speed_kmh : ℕ)

def to_m_per_s (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 1000) / 3600

def relative_speed (trainA trainB : Train) : ℕ :=
  to_m_per_s trainA.speed_kmh + to_m_per_s trainB.speed_kmh

def total_length (trainA trainB : Train) : ℕ :=
  trainA.length + trainB.length

def time_to_cross (trainA trainB : Train) : ℕ :=
  total_length trainA trainB / relative_speed trainA trainB

theorem time_taken_by_Arun_to_cross_train_B :
  time_to_cross (Train.mk 175 54) (Train.mk 150 36) = 13 :=
by
  sorry

end time_taken_by_Arun_to_cross_train_B_l116_116096


namespace vectors_parallel_solution_l116_116774

theorem vectors_parallel_solution (x : ℝ) (a b : ℝ × ℝ) (h1 : a = (2, x)) (h2 : b = (x, 8)) (h3 : ∃ k, b = (k * 2, k * x)) : x = 4 ∨ x = -4 :=
by
  sorry

end vectors_parallel_solution_l116_116774


namespace number_at_100th_row_1000th_column_l116_116255

axiom cell_numbering_rule (i j : ℕ) : ℕ

/-- 
  The cell located at the intersection of the 100th row and the 1000th column
  on an infinitely large chessboard, sequentially numbered with specific rules,
  will receive the number 900.
-/
theorem number_at_100th_row_1000th_column : cell_numbering_rule 100 1000 = 900 :=
sorry

end number_at_100th_row_1000th_column_l116_116255


namespace treasures_coins_count_l116_116497

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l116_116497


namespace focus_of_parabola_l116_116570

theorem focus_of_parabola (x y : ℝ) (h : y = 4 * x^2) : (0, 1 / 16) ∈ {p : ℝ × ℝ | ∃ x y, y = 4 * x^2 ∧ p = (0, 1 / (4 * (1 / y)))} :=
by
  sorry

end focus_of_parabola_l116_116570


namespace initial_volume_of_mixture_l116_116242

theorem initial_volume_of_mixture (p q : ℕ) (x : ℕ) (h_ratio1 : p = 5 * x) (h_ratio2 : q = 3 * x) (h_added : q + 15 = 6 * x) (h_new_ratio : 5 * (3 * x + 15) = 6 * 5 * x) : 
  p + q = 40 :=
by
  sorry

end initial_volume_of_mixture_l116_116242


namespace new_circumference_of_circle_l116_116103

theorem new_circumference_of_circle (w h : ℝ) (d_multiplier : ℝ) 
  (h_w : w = 7) (h_h : h = 24) (h_d_multiplier : d_multiplier = 1.5) : 
  (π * (d_multiplier * (Real.sqrt (w^2 + h^2)))) = 37.5 * π :=
by
  sorry

end new_circumference_of_circle_l116_116103


namespace inequalities_proof_l116_116426

variables (x y z : ℝ)

def p := x + y + z
def q := x * y + y * z + z * x
def r := x * y * z

theorem inequalities_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (p x y z) ^ 2 ≥ 3 * (q x y z) ∧
  (p x y z) ^ 3 ≥ 27 * (r x y z) ∧
  (p x y z) * (q x y z) ≥ 9 * (r x y z) ∧
  (q x y z) ^ 2 ≥ 3 * (p x y z) * (r x y z) ∧
  (p x y z) ^ 2 * (q x y z) + 3 * (p x y z) * (r x y z) ≥ 4 * (q x y z) ^ 2 ∧
  (p x y z) ^ 3 + 9 * (r x y z) ≥ 4 * (p x y z) * (q x y z) ∧
  (p x y z) * (q x y z) ^ 2 ≥ 2 * (p x y z) ^ 2 * (r x y z) + 3 * (q x y z) * (r x y z) ∧
  (p x y z) * (q x y z) ^ 2 + 3 * (q x y z) * (r x y z) ≥ 4 * (p x y z) ^ 2 * (r x y z) ∧
  2 * (q x y z) ^ 3 + 9 * (r x y z) ^ 2 ≥ 7 * (p x y z) * (q x y z) * (r x y z) ∧
  (p x y z) ^ 4 + 4 * (q x y z) ^ 2 + 6 * (p x y z) * (r x y z) ≥ 5 * (p x y z) ^ 2 * (q x y z) :=
by sorry

end inequalities_proof_l116_116426


namespace johns_payment_l116_116168

-- Define the value of the camera
def camera_value : ℕ := 5000

-- Define the rental fee rate per week as a percentage
def rental_fee_rate : ℝ := 0.1

-- Define the rental period in weeks
def rental_period : ℕ := 4

-- Define the friend's contribution rate as a percentage
def friend_contribution_rate : ℝ := 0.4

-- Theorem: Calculate how much John pays for the camera rental
theorem johns_payment :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let total_rental_fee := weekly_rental_fee * rental_period
  let friends_contribution := total_rental_fee * friend_contribution_rate
  let johns_payment := total_rental_fee - friends_contribution
  johns_payment = 1200 :=
by
  sorry

end johns_payment_l116_116168


namespace proof_problem_l116_116733

noncomputable def mean (l : List ℕ) : ℝ :=
  (l.sum : ℝ) / (l.length : ℝ)

noncomputable def median (l : List ℕ) : ℝ :=
  let sorted := l.qsort (≤)
  if l.length % 2 = 0 then 
    ((sorted.get ((l.length / 2) - 1)) + (sorted.get (l.length / 2))) / 2
  else 
    sorted.get (l.length / 2)

noncomputable def modes (l : List ℕ) : List ℕ :=
  let freq_map := l.foldl (λ m n => m.insert n (m.find n).get_or_else 0 + 1) ∅
  let max_freq := freq_map.fold (0, 0) (λ (k, v) (max_k, max_v) => if v > max_v then (k, v) else (max_k, max_v)).snd
  freq_map.toList.filter (λ (k, v) => v = max_freq).map Prod.fst

noncomputable def median_of_modes (l : List ℕ) : ℝ :=
  let modes_sorted := (modes l).qsort (≤)
  if modes_sorted.length % 2 = 0 then 
    ((modes_sorted.get ((modes_sorted.length / 2) - 1)) + (modes_sorted.get (modes_sorted.length / 2))) / 2
  else 
    modes_sorted.get (modes_sorted.length / 2)

theorem proof_problem :
  let dates := List.replicate 12 (List.range 1 29) ++ List.replicate 12 29 ++ List.replicate 12 30 ++ List.replicate 8 31
  let flat_dates := dates.join
  median_of_modes flat_dates < mean flat_dates ∧ mean flat_dates < median flat_dates := sorry


end proof_problem_l116_116733


namespace train_speed_conversion_l116_116558

/-- Define a function to convert kmph to m/s --/
def kmph_to_ms (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

/-- Theorem stating that 72 kmph is equivalent to 20 m/s --/
theorem train_speed_conversion : kmph_to_ms 72 = 20 :=
by
  sorry

end train_speed_conversion_l116_116558


namespace geometric_sequence_value_l116_116960

theorem geometric_sequence_value 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_nonzero_diff : d ≠ 0)
  (h_condition : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0)
  (h_geom_seq : ∀ n, b (n + 1) = b n * (b 1 / b 0))
  (h_b7_eq_a7 : b 7 = a 7) :
  b 6 * b 8 = 16 :=
sorry

end geometric_sequence_value_l116_116960


namespace outfits_count_l116_116872

def num_outfits (redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats : ℕ) : ℕ :=
  (redShirts * pairsPants * (greenHats + blueHats)) +
  (greenShirts * pairsPants * (redHats + blueHats)) +
  (blueShirts * pairsPants * (redHats + greenHats))

theorem outfits_count :
  ∀ (redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats : ℕ),
  redShirts = 4 → greenShirts = 4 → blueShirts = 4 →
  pairsPants = 7 →
  greenHats = 6 → redHats = 6 → blueHats = 6 →
  num_outfits redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats = 1008 :=
by
  intros redShirts greenShirts blueShirts pairsPants greenHats redHats blueHats
  intros hredShirts hgreenShirts hblueShirts hpairsPants hgreenHats hredHats hblueHats
  rw [hredShirts, hgreenShirts, hblueShirts, hpairsPants, hgreenHats, hredHats, hblueHats]
  sorry

end outfits_count_l116_116872


namespace find_number_l116_116687

variable (x : ℝ)

theorem find_number : ((x * 5) / 2.5 - 8 * 2.25 = 5.5) -> x = 11.75 :=
by
  intro h
  sorry

end find_number_l116_116687


namespace unique_bisecting_line_exists_l116_116718

noncomputable def triangle_area := 1 / 2 * 6 * 8
noncomputable def triangle_perimeter := 6 + 8 + 10

theorem unique_bisecting_line_exists :
  ∃ (line : ℝ → ℝ), 
    (∃ x y : ℝ, x + y = 12 ∧ x * y = 30 ∧ 
      1 / 2 * x * y * (24 / triangle_perimeter) = 12) ∧
    (∃ x' y' : ℝ, x' + y' = 12 ∧ x' * y' = 24 ∧ 
      1 / 2 * x' * y' * (24 / triangle_perimeter) = 12) ∧
    ((x = x' ∧ y = y') ∨ (x = y' ∧ y = x')) :=
sorry

end unique_bisecting_line_exists_l116_116718


namespace area_AKM_less_than_area_ABC_l116_116604

-- Define the rectangle ABCD
structure Rectangle :=
(A B C D : ℝ) -- Four vertices of the rectangle
(side_AB : ℝ) (side_BC : ℝ) (side_CD : ℝ) (side_DA : ℝ)

-- Define the arbitrary points K and M on sides BC and CD respectively
variables (B C D K M : ℝ)

-- Define the area of triangle function and area of rectangle function
def area_triangle (A B C : ℝ) : ℝ := sorry -- Assuming a function calculating area of triangle given 3 vertices
def area_rectangle (A B C D : ℝ) : ℝ := sorry -- Assuming a function calculating area of rectangle given 4 vertices

-- Assuming the conditions given in the problem statement
variables (A : ℝ) (rect : Rectangle)

-- Prove that the area of triangle AKM is less than the area of triangle ABC
theorem area_AKM_less_than_area_ABC : 
  ∀ (K M : ℝ), K ∈ [B,C] → M ∈ [C,D] →
    area_triangle A K M < area_triangle A B C := sorry

end area_AKM_less_than_area_ABC_l116_116604


namespace compute_expression_l116_116897

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end compute_expression_l116_116897


namespace compute_fraction_power_l116_116008

theorem compute_fraction_power :
  8 * (2 / 7)^4 = 128 / 2401 :=
by
  sorry

end compute_fraction_power_l116_116008


namespace jill_llamas_count_l116_116052

theorem jill_llamas_count : 
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  herd_after_sell = 18 := 
by
  -- Definitions for the conditions
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  -- Proof will be filled in here.
  sorry

end jill_llamas_count_l116_116052


namespace total_amount_spent_correct_l116_116389

noncomputable def total_amount_spent (mango_cost pineapple_cost cost_pineapple total_people : ℕ) : ℕ :=
  let pineapple_people := cost_pineapple / pineapple_cost
  let mango_people := total_people - pineapple_people
  let mango_cost_total := mango_people * mango_cost
  cost_pineapple + mango_cost_total

theorem total_amount_spent_correct :
  total_amount_spent 5 6 54 17 = 94 := by
  -- This is where the proof would go, but it's omitted per instructions
  sorry

end total_amount_spent_correct_l116_116389


namespace find_certain_age_l116_116422

theorem find_certain_age 
(Kody_age : ℕ) 
(Mohamed_age : ℕ) 
(certain_age : ℕ) 
(h1 : Kody_age = 32) 
(h2 : Mohamed_age = 2 * certain_age) 
(h3 : ∀ four_years_ago, four_years_ago = Kody_age - 4 → four_years_ago * 2 = Mohamed_age - 4) :
  certain_age = 30 := sorry

end find_certain_age_l116_116422


namespace intersection_slopes_l116_116704

theorem intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (4 / 41)) ∨ m ∈ Set.Ici (Real.sqrt (4 / 41)) := 
sorry

end intersection_slopes_l116_116704


namespace polynomial_identity_solution_l116_116744

theorem polynomial_identity_solution (P : Polynomial ℝ) :
  (∀ x : ℝ, x * P.eval (x - 1) = (x - 2) * P.eval x) ↔ (∃ a : ℝ, P = Polynomial.C a * (Polynomial.X ^ 2 - Polynomial.X)) :=
by
  sorry

end polynomial_identity_solution_l116_116744


namespace arithmetic_seq_sum_x_y_l116_116859

theorem arithmetic_seq_sum_x_y :
  ∃ (x y : ℕ), (∀ n : ℕ, n > 0 → a_n = 3 + (n - 1) * 5) ∧ x + 33 = 33 ∧ x = 28 → x + y = 61 :=
by
  sorry

end arithmetic_seq_sum_x_y_l116_116859


namespace marco_total_time_l116_116176

def marco_run_time (laps distance1 distance2 speed1 speed2 : ℕ ) : ℝ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  laps * (time1 + time2)

theorem marco_total_time :
  marco_run_time 7 150 350 3 4 = 962.5 :=
by
  sorry

end marco_total_time_l116_116176


namespace final_trip_theorem_l116_116641

/-- Define the lineup of dwarfs -/
inductive Dwarf where
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

open Dwarf

/-- Define the conditions -/
-- The dwarfs initially standing next to each other
def adjacent (d1 d2 : Dwarf) : Prop :=
  (d1 = Happy ∧ d2 = Grumpy) ∨
  (d1 = Grumpy ∧ d2 = Dopey) ∨
  (d1 = Dopey ∧ d2 = Bashful) ∨
  (d1 = Bashful ∧ d2 = Sleepy) ∨
  (d1 = Sleepy ∧ d2 = Doc) ∨
  (d1 = Doc ∧ d2 = Sneezy)

-- Snow White is the only one who can row
constant snowWhite_can_row : Prop := true

-- The boat can hold Snow White and up to 3 dwarfs
constant boat_capacity : ℕ := 4

-- Define quarrel if left without Snow White
def quarrel_without_snowwhite (d1 d2 : Dwarf) : Prop := adjacent d1 d2

-- Define the final trip setup
def final_trip (dwarfs : List Dwarf) : Prop :=
  dwarfs = [Grumpy, Bashful, Doc]

-- Theorem to prove the final trip
theorem final_trip_theorem : ∃ dwarfs, final_trip dwarfs :=
  sorry

end final_trip_theorem_l116_116641


namespace gcd_90_405_l116_116270

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l116_116270


namespace cube_side_length_l116_116199

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end cube_side_length_l116_116199


namespace tank_fraction_l116_116062

theorem tank_fraction (x : ℚ) : 
  let tank1_capacity := 7000
  let tank2_capacity := 5000
  let tank3_capacity := 3000
  let tank2_fraction := 4 / 5
  let tank3_fraction := 1 / 2
  let total_water := 10850
  tank1_capacity * x + tank2_capacity * tank2_fraction + tank3_capacity * tank3_fraction = total_water → 
  x = 107 / 140 := 
by {
  sorry
}

end tank_fraction_l116_116062


namespace pirates_treasure_l116_116495

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l116_116495


namespace cube_side_length_l116_116196

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end cube_side_length_l116_116196


namespace diane_coffee_purchase_l116_116265

theorem diane_coffee_purchase (c d : ℕ) (h1 : c + d = 7) (h2 : 90 * c + 60 * d % 100 = 0) : c = 6 :=
by
  sorry

end diane_coffee_purchase_l116_116265


namespace circle_placement_in_rectangle_l116_116710

theorem circle_placement_in_rectangle
  (L W : ℝ) (n : ℕ) (side_length diameter : ℝ)
  (h_dim : L = 20) (w_dim : W = 25)
  (h_squares : n = 120) (h_side_length : side_length = 1)
  (h_diameter : diameter = 1) :
  ∃ (x y : ℝ) (circle_radius : ℝ), 
    circle_radius = diameter / 2 ∧
    0 ≤ x ∧ x + diameter / 2 ≤ L ∧ 
    0 ≤ y ∧ y + diameter / 2 ≤ W ∧ 
    ∀ (i : ℕ) (hx : i < n) (sx sy : ℝ),
      0 ≤ sx ∧ sx + side_length ≤ L ∧
      0 ≤ sy ∧ sy + side_length ≤ W ∧
      dist (x, y) (sx + side_length / 2, sy + side_length / 2) ≥ diameter / 2 := 
sorry

end circle_placement_in_rectangle_l116_116710


namespace range_of_m_l116_116474

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) ↔ (m ≤ 2) :=
by
  sorry

end range_of_m_l116_116474


namespace painting_time_l116_116572

theorem painting_time (n₁ t₁ n₂ t₂ : ℕ) (h1 : n₁ = 8) (h2 : t₁ = 12) (h3 : n₂ = 6) (h4 : n₁ * t₁ = n₂ * t₂) : t₂ = 16 :=
by
  sorry

end painting_time_l116_116572


namespace eggs_left_l116_116549

def initial_eggs := 20
def mother_used := 5
def father_used := 3
def chicken1_laid := 4
def chicken2_laid := 3
def chicken3_laid := 2
def oldest_took := 2

theorem eggs_left :
  initial_eggs - (mother_used + father_used) + (chicken1_laid + chicken2_laid + chicken3_laid) - oldest_took = 19 := 
by
  sorry

end eggs_left_l116_116549


namespace problem_part1_problem_part2_l116_116585

noncomputable def f (m x : ℝ) := Real.log (m * x) - x + 1
noncomputable def g (m x : ℝ) := (x - 1) * Real.exp x - m * x

theorem problem_part1 (m : ℝ) (h : m > 0) (hf : ∀ x, f m x ≤ 0) : m = 1 :=
sorry

theorem problem_part2 (m : ℝ) (h : m > 0) :
  ∃ x₀, (∀ x, g m x ≤ g m x₀) ∧ (1 / 2 * Real.log (m + 1) < x₀ ∧ x₀ < m) :=
sorry

end problem_part1_problem_part2_l116_116585


namespace symmetric_circle_equation_l116_116438

theorem symmetric_circle_equation :
  (∀ x y : ℝ, (x - 1) ^ 2 + y ^ 2 = 1 ↔ x ^ 2 + (y + 1) ^ 2 = 1) :=
by sorry

end symmetric_circle_equation_l116_116438


namespace largest_prime_mersenne_below_500_l116_116885

def is_mersenne (m : ℕ) (n : ℕ) := m = 2^n - 1
def is_power_of_2 (n : ℕ) := ∃ (k : ℕ), n = 2^k

theorem largest_prime_mersenne_below_500 : ∀ (m : ℕ), 
  m < 500 →
  (∃ n, is_power_of_2 n ∧ is_mersenne m n ∧ Nat.Prime m) →
  m ≤ 3 := 
by
  sorry

end largest_prime_mersenne_below_500_l116_116885


namespace roberts_total_sales_l116_116632

theorem roberts_total_sales 
  (basic_salary : ℝ := 1250) 
  (commission_rate : ℝ := 0.10) 
  (savings_rate : ℝ := 0.20) 
  (monthly_expenses : ℝ := 2888) 
  (S : ℝ) : S = 23600 :=
by
  have total_earnings := basic_salary + commission_rate * S
  have used_for_expenses := (1 - savings_rate) * total_earnings
  have expenses_eq : used_for_expenses = monthly_expenses := sorry
  have expense_calc : (1 - savings_rate) * (basic_salary + commission_rate * S) = monthly_expenses := sorry
  have simplify_eq : 0.80 * (1250 + 0.10 * S) = 2888 := sorry
  have open_eq : 1000 + 0.08 * S = 2888 := sorry
  have isolate_S : 0.08 * S = 1888 := sorry
  have solve_S : S = 1888 / 0.08 := sorry
  have final_S : S = 23600 := sorry
  exact final_S

end roberts_total_sales_l116_116632


namespace arithmetic_sequence_sum_l116_116864

theorem arithmetic_sequence_sum :
  ∀ (x y : ℕ), (∀ (a b c : ℕ), b - a = c - b → c - b = 5) ∧
               (3 + 5 * 1 = 8) ∧
               (8 + 5 * 1 = 13) ∧
               (x + 5 * 1 = y) ∧
               (y + 5 * 1 = 33) →
               x + y = 51 :=
by
  intros x y h
  sorry

end arithmetic_sequence_sum_l116_116864


namespace compute_expression_l116_116899

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end compute_expression_l116_116899


namespace chris_money_left_over_l116_116256

-- Define the constants based on the conditions given in the problem.
def video_game_cost : ℕ := 60
def candy_cost : ℕ := 5
def earnings_per_hour : ℕ := 8
def hours_worked : ℕ := 9

-- Define the intermediary results based on the problem's conditions.
def total_cost : ℕ := video_game_cost + candy_cost
def total_earnings : ℕ := earnings_per_hour * hours_worked

-- Define the final result to be proven.
def total_leftover : ℕ := total_earnings - total_cost

-- State the proof problem as a Lean theorem.
theorem chris_money_left_over : total_leftover = 7 := by
  sorry

end chris_money_left_over_l116_116256


namespace average_marks_correct_l116_116095

def marks := [76, 65, 82, 62, 85]
def num_subjects := 5
def total_marks := marks.sum
def avg_marks := total_marks / num_subjects

theorem average_marks_correct : avg_marks = 74 :=
by sorry

end average_marks_correct_l116_116095


namespace find_b_for_smallest_c_l116_116124

theorem find_b_for_smallest_c (c b : ℝ) (h_c_pos : 0 < c) (h_b_pos : 0 < b)
  (polynomial_condition : ∀ x : ℝ, (x^4 - c*x^3 + b*x^2 - c*x + 1 = 0) → real) :
  c = 4 → b = 6 :=
by
  intros h_c_eq_4
  sorry

end find_b_for_smallest_c_l116_116124


namespace james_calories_per_minute_l116_116165

variable (classes_per_week : ℕ) (hours_per_class : ℝ) (total_calories_per_week : ℕ)

theorem james_calories_per_minute
  (h1 : classes_per_week = 3)
  (h2 : hours_per_class = 1.5)
  (h3 : total_calories_per_week = 1890) :
  total_calories_per_week / (classes_per_week * (hours_per_class * 60)) = 7 := 
by
  sorry

end james_calories_per_minute_l116_116165


namespace angle_and_ratio_range_l116_116317

variables {a b c A B C : ℝ}

-- Define the circumcenter condition
def circumcenter_inside_triangle (A B C : ℝ) := 
  A < π/2 ∧ B < π/2 ∧ C < π/2

-- Define the main condition
def main_condition (a b c A B C : ℝ) :=
  (b^2 - a^2 - c^2) * Real.sin (B + C) = Real.sqrt 3 * a * c * Real.cos (A + C)

theorem angle_and_ratio_range 
  (h1 : circumcenter_inside_triangle A B C)
  (h2 : main_condition a b c A B C) :
  (A = π / 3) ∧ (Real.sqrt 3 < (b + c) / a ∧ (b + c) / a ≤ 2) :=
by
  sorry

end angle_and_ratio_range_l116_116317


namespace fraction_to_decimal_l116_116905

theorem fraction_to_decimal : (7 / 32 : ℚ) = 0.21875 := 
by {
  sorry
}

end fraction_to_decimal_l116_116905


namespace parabola_equations_l116_116708

theorem parabola_equations (x y : ℝ) (h₁ : (0, 0) = (0, 0)) (h₂ : (-2, 3) = (-2, 3)) :
  (x^2 = 4 / 3 * y) ∨ (y^2 = - 9 / 2 * x) :=
sorry

end parabola_equations_l116_116708


namespace time_after_3577_minutes_l116_116088

-- Definitions
def startingTime : Nat := 6 * 60 -- 6:00 PM in minutes
def startDate : String := "2020-12-31"
def durationMinutes : Nat := 3577
def minutesInHour : Nat := 60
def hoursInDay : Nat := 24

-- Theorem to prove that 3577 minutes after 6:00 PM on December 31, 2020 is January 3 at 5:37 AM
theorem time_after_3577_minutes : 
  (durationMinutes + startingTime) % (hoursInDay * minutesInHour) = 5 * minutesInHour + 37 :=
  by
  sorry -- proof goes here

end time_after_3577_minutes_l116_116088


namespace sports_club_membership_l116_116159

theorem sports_club_membership :
  (17 + 21 - 10 + 2 = 30) :=
by
  sorry

end sports_club_membership_l116_116159


namespace find_total_coins_l116_116516

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l116_116516


namespace largest_non_sum_217_l116_116680

def is_prime (n : ℕ) : Prop := sorry -- This would be some definition of primality

noncomputable def largest_non_sum_of_composite (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ b ∧ b < 30 ∧ (∀ i : ℕ, i < a → is_prime (b + 30 * i)) →
  n ≤ 30 * a + b

theorem largest_non_sum_217 : ∃ n, largest_non_sum_of_composite n ∧ n = 217 := sorry

end largest_non_sum_217_l116_116680


namespace rational_solution_counts_l116_116920

theorem rational_solution_counts :
  (∃ (x y : ℚ), x^2 + y^2 = 2) ∧ 
  (¬ ∃ (x y : ℚ), x^2 + y^2 = 3) := 
by 
  sorry

end rational_solution_counts_l116_116920


namespace repeated_decimal_to_fraction_l116_116778

theorem repeated_decimal_to_fraction :
  ∀ (a b : ℕ), (0.35.recursor = 35 / 99 ∧ Nat.gcd 35 99 = 1) → a + b = 134 := by
  sorry

end repeated_decimal_to_fraction_l116_116778


namespace LilyUsed14Dimes_l116_116989

variable (p n d : ℕ)

theorem LilyUsed14Dimes
  (h1 : p + n + d = 50)
  (h2 : p + 5 * n + 10 * d = 200) :
  d = 14 := by
  sorry

end LilyUsed14Dimes_l116_116989


namespace total_pigs_correct_l116_116082

def initial_pigs : Float := 64.0
def incoming_pigs : Float := 86.0
def total_pigs : Float := 150.0

theorem total_pigs_correct : initial_pigs + incoming_pigs = total_pigs := by 
  sorry

end total_pigs_correct_l116_116082


namespace average_weight_women_l116_116482

variable (average_weight_men : ℕ) (number_of_men : ℕ)
variable (average_weight : ℕ) (number_of_women : ℕ)
variable (average_weight_all : ℕ) (total_people : ℕ)

theorem average_weight_women (h1 : average_weight_men = 190) 
                            (h2 : number_of_men = 8)
                            (h3 : average_weight_all = 160)
                            (h4 : total_people = 14) 
                            (h5 : number_of_women = 6):
  average_weight = 120 := 
by
  sorry

end average_weight_women_l116_116482


namespace physics_experiment_l116_116246

theorem physics_experiment (x : ℕ) (h : 1 + x + (x + 1) * x = 36) :
  1 + x + (x + 1) * x = 36 :=
  by                        
  exact h

end physics_experiment_l116_116246


namespace triangle_acd_area_l116_116323

noncomputable def area_of_triangle : ℝ := sorry

theorem triangle_acd_area (AB CD : ℝ) (h : CD = 3 * AB) (area_trapezoid: ℝ) (h1: area_trapezoid = 20) :
  area_of_triangle = 15 := 
sorry

end triangle_acd_area_l116_116323


namespace toby_peanut_butter_servings_l116_116854

theorem toby_peanut_butter_servings :
  let bread_calories := 100
  let peanut_butter_calories_per_serving := 200
  let total_calories := 500
  let bread_pieces := 1
  ∃ (servings : ℕ), total_calories = (bread_calories * bread_pieces) + (peanut_butter_calories_per_serving * servings) → servings = 2 := by
  sorry

end toby_peanut_butter_servings_l116_116854


namespace four_digit_number_divisible_by_9_l116_116463

theorem four_digit_number_divisible_by_9
    (a b c d e f g h i j : ℕ)
    (h₀ : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
               f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
               g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
               h ≠ i ∧ h ≠ j ∧
               i ≠ j )
    (h₁ : a + b + c + d + e + f + g + h + i + j = 45)
    (h₂ : 100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j) :
  ((1000 * g + 100 * h + 10 * i + j) % 9 = 0) := sorry

end four_digit_number_divisible_by_9_l116_116463


namespace disjoint_subsets_same_sum_l116_116670

theorem disjoint_subsets_same_sum (s : Finset ℕ) (h₁ : s.card = 10) (h₂ : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 100) :
  ∃ A B : Finset ℕ, A ⊆ s ∧ B ⊆ s ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end disjoint_subsets_same_sum_l116_116670


namespace valid_common_ratios_count_l116_116298

noncomputable def num_valid_common_ratios (a₁ : ℝ) (q : ℝ) : ℝ :=
  let a₅ := a₁ * q^4
  let a₃ := a₁ * q^2
  if 2 * a₅ = 4 * a₁ + (-2) * a₃ then 1 else 0

theorem valid_common_ratios_count (a₁ : ℝ) : 
  (num_valid_common_ratios a₁ 1) + (num_valid_common_ratios a₁ (-1)) = 2 :=
by sorry

end valid_common_ratios_count_l116_116298


namespace quadratic_roots_ratio_l116_116653

theorem quadratic_roots_ratio {m n p : ℤ} (h₀ : m ≠ 0) (h₁ : n ≠ 0) (h₂ : p ≠ 0)
  (h₃ : ∃ r1 r2 : ℤ, r1 * r2 = m ∧ n = 9 * r1 * r2 ∧ p = -(r1 + r2) ∧ m = -3 * (r1 + r2)) :
  n / p = -27 := by
  sorry

end quadratic_roots_ratio_l116_116653


namespace find_h_l116_116988

noncomputable def h (x : ℝ) : ℝ := -x^4 - 2 * x^3 + 4 * x^2 + 9 * x - 5

def f (x : ℝ) : ℝ := x^4 + 2 * x^3 - x^2 - 4 * x + 1

def p (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 4

theorem find_h (x : ℝ) : (f x) + (h x) = p x :=
by sorry

end find_h_l116_116988


namespace different_tea_packets_or_miscalculation_l116_116908

theorem different_tea_packets_or_miscalculation : 
  ∀ (n_1 n_2 : ℕ), 3 ≤ t_1 ∧ t_1 ≤ 4 ∧ 3 ≤ t_2 ∧ t_2 ≤ 4 ∧
  (74 = t_1 * x ∧ 105 = t_2 * y → x ≠ y) ∨ 
  (∃ (e_1 e_2 : ℕ), (e_1 + e_2 = 74) ∧ (e_1 + e_2 = 105) → false) :=
by
  -- Construction based on the provided mathematical problem
  sorry

end different_tea_packets_or_miscalculation_l116_116908


namespace ellipse_eccentricity_l116_116806

theorem ellipse_eccentricity (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (k + 8) + y^2 / 9 = 1) ∧ (∃ e : ℝ, e = 1 / 2) → 
  (k = 4 ∨ k = -5 / 4) := sorry

end ellipse_eccentricity_l116_116806


namespace probability_of_winning_pair_l116_116548

/--
A deck consists of five red cards and five green cards, with each color having cards labeled from A to E. 
Two cards are drawn from this deck.
A winning pair is defined as two cards of the same color or two cards of the same letter. 
Prove that the probability of drawing a winning pair is 5/9.
-/
theorem probability_of_winning_pair :
  let total_cards := 10
  let total_ways := Nat.choose total_cards 2
  let same_letter_ways := 5
  let same_color_red_ways := Nat.choose 5 2
  let same_color_green_ways := Nat.choose 5 2
  let same_color_ways := same_color_red_ways + same_color_green_ways
  let favorable_outcomes := same_letter_ways + same_color_ways
  favorable_outcomes / total_ways = 5 / 9 := by
  sorry

end probability_of_winning_pair_l116_116548


namespace find_alpha_l116_116436

noncomputable def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * (a 2 / a 1)

-- Given that {a_n} is a geometric sequence,
-- a_1 and a_8 are roots of the equation
-- x^2 - 2x * sin(alpha) - √3 * sin(alpha) = 0,
-- and (a_1 + a_8)^2 = 2 * a_3 * a_6 + 6,
-- prove that alpha = π / 3.
theorem find_alpha :
  ∃ α : ℝ,
  (∀ (a : ℕ → ℝ), isGeometricSequence a ∧ 
  (∃ (a1 a8 : ℝ), 
    (a1 + a8)^2 = 2 * a 3 * a 6 + 6 ∧
    a1 + a8 = 2 * Real.sin α ∧
    a1 * a8 = - Real.sqrt 3 * Real.sin α)) →
  α = Real.pi / 3 :=
by 
  sorry

end find_alpha_l116_116436


namespace baby_plants_produced_l116_116735

theorem baby_plants_produced (baby_plants_per_time: ℕ) (times_per_year: ℕ) (years: ℕ) (total_babies: ℕ) :
  baby_plants_per_time = 2 ∧ times_per_year = 2 ∧ years = 4 ∧ total_babies = baby_plants_per_time * times_per_year * years → 
  total_babies = 16 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end baby_plants_produced_l116_116735


namespace equation_of_line_intersection_l116_116773

theorem equation_of_line_intersection
  (h1 : ∀ x y : ℝ, x^2 + y^2 = 1)
  (h2 : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0) :
  ∀ x y : ℝ, x - 2*y + 1 = 0 :=
by
  sorry

end equation_of_line_intersection_l116_116773


namespace johns_payment_l116_116167

def camera_value : ℕ := 5000
def rental_fee_perc : ℕ := 10
def rental_duration : ℕ := 4
def friend_contrib_perc : ℕ := 40

theorem johns_payment :
  let total_rental_fee := (rental_fee_perc * camera_value / 100) * rental_duration,
      friend_payment := (friend_contrib_perc * total_rental_fee / 100),
      johns_payment := total_rental_fee - friend_payment
  in johns_payment = 1200 :=
by
  sorry

end johns_payment_l116_116167


namespace order_of_logs_l116_116617

open Real

noncomputable def a := log 10 / log 5
noncomputable def b := log 12 / log 6
noncomputable def c := 1 + log 2 / log 7

theorem order_of_logs : a > b ∧ b > c :=
by
  sorry

end order_of_logs_l116_116617


namespace correct_propositions_l116_116770

-- Define the propositions as conditions
def proposition1 (x : ℝ) : Prop := - (1 / x)

def proposition2 (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≤ -1 → (deriv (λ x:ℝ, x^2 + 2*a*x + 1) x) ≤ 0 → a ≤ 1

def proposition3 (m : ℝ) : Prop :=
  log 0.7 (2 * m) < log 0.7 (m - 1) → m < -1

def proposition4 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) → (f (1 - x) + f (x - 1) = 0)

theorem correct_propositions :
  ({proposition2} ∪ {proposition4} = {proposition2, proposition4}) ∧
  ({proposition1} ∪ {proposition3} = ∅) :=
by
  sorry

end correct_propositions_l116_116770


namespace find_playground_side_length_l116_116664

-- Define the conditions
def playground_side_length (x : ℝ) : Prop :=
  let perimeter_square := 4 * x
  let perimeter_garden := 2 * (12 + 9)
  let total_perimeter := perimeter_square + perimeter_garden
  total_perimeter = 150

-- State the main theorem to prove that the side length of the square fence around the playground is 27 yards
theorem find_playground_side_length : ∃ x : ℝ, playground_side_length x ∧ x = 27 :=
by
  exists 27
  sorry

end find_playground_side_length_l116_116664


namespace decimal_to_base9_l116_116406

theorem decimal_to_base9 (n : ℕ) (h : n = 1729) : 
  (2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0) = n :=
by sorry

end decimal_to_base9_l116_116406


namespace coprime_repeating_decimal_sum_l116_116793

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l116_116793


namespace max_subset_size_l116_116985

/-- A function to check if the sum of any two distinct elements of S is not divisible by 7 -/
def valid_subset (S : Finset ℕ) : Prop :=
  ∀ {a b}, a ∈ S → b ∈ S → a ≠ b → (a + b) % 7 ≠ 0

/-- The maximum possible number of elements in a subset of {1, 2, ..., 50} with no two distinct elements summing to a multiple of 7 is 23 -/
theorem max_subset_size (S : Finset ℕ) (hS : S ⊆ Finset.range 51) (hvalid : valid_subset S) :
  S.card ≤ 23 := sorry

end max_subset_size_l116_116985


namespace quadratic_bound_l116_116361

theorem quadratic_bound (a b c : ℝ) :
  (∀ (u : ℝ), |u| ≤ 10 / 11 → ∃ (v : ℝ), |u - v| ≤ 1 / 11 ∧ |a * v^2 + b * v + c| ≤ 1) →
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 2 := by
  sorry

end quadratic_bound_l116_116361


namespace sara_initial_savings_l116_116467

-- Given conditions as definitions
def save_rate_sara : ℕ := 10
def save_rate_jim : ℕ := 15
def weeks : ℕ := 820

-- Prove that the initial savings of Sara is 4100 dollars given the conditions
theorem sara_initial_savings : 
  ∃ S : ℕ, S + save_rate_sara * weeks = save_rate_jim * weeks → S = 4100 := 
sorry

end sara_initial_savings_l116_116467


namespace power_eq_l116_116113

theorem power_eq (a b c : ℝ) (h₁ : a = 81) (h₂ : b = 4 / 3) : (a ^ b) = 243 * (3 ^ (1 / 3)) := by
  sorry

end power_eq_l116_116113


namespace infinite_squares_and_circles_difference_l116_116011

theorem infinite_squares_and_circles_difference 
  (side_length : ℝ)
  (h₁ : side_length = 1)
  (square_area_sum : ℝ)
  (circle_area_sum : ℝ)
  (h_square_area : square_area_sum = (∑' n : ℕ, (side_length / 2^n)^2))
  (h_circle_area : circle_area_sum = (∑' n : ℕ, π * (side_length / 2^(n+1))^2 ))
  : square_area_sum - circle_area_sum = 2 - (π / 2) :=
by 
  sorry 

end infinite_squares_and_circles_difference_l116_116011


namespace hyperbola_real_axis_length_l116_116936

theorem hyperbola_real_axis_length (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : ∀ x y : ℝ, x = 1 → y = 2 → (x^2 / (a^2)) - (y^2 / (b^2)) = 1)
  (h_parabola : ∀ y : ℝ, y = 2 → (y^2) = 4 * 1)
  (h_focus : (1, 2) = (1, 2))
  (h_eq : a^2 + b^2 = 1) :
  2 * a = 2 * (Real.sqrt 2 - 1) :=
by 
-- Skipping the proof part
sorry

end hyperbola_real_axis_length_l116_116936


namespace Tony_fever_l116_116946

theorem Tony_fever :
  ∀ (normal_temp sickness_increase fever_threshold : ℕ),
    normal_temp = 95 →
    sickness_increase = 10 →
    fever_threshold = 100 →
    (normal_temp + sickness_increase) - fever_threshold = 5 :=
by
  intros normal_temp sickness_increase fever_threshold h1 h2 h3
  sorry

end Tony_fever_l116_116946


namespace equation_infinitely_many_solutions_iff_b_eq_neg9_l116_116018

theorem equation_infinitely_many_solutions_iff_b_eq_neg9 (b : ℤ) :
  (∀ x : ℤ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := 
by
  sorry

end equation_infinitely_many_solutions_iff_b_eq_neg9_l116_116018


namespace largest_integer_value_x_l116_116219

theorem largest_integer_value_x : ∀ (x : ℤ), (5 - 4 * x > 17) → x ≤ -4 := sorry

end largest_integer_value_x_l116_116219


namespace sum_of_abc_l116_116118

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (eq1 : a^2 + b * c = 115) (eq2 : b^2 + a * c = 127) (eq3 : c^2 + a * b = 115) :
  a + b + c = 22 := by
  sorry

end sum_of_abc_l116_116118


namespace weight_of_each_bag_of_flour_l116_116668

theorem weight_of_each_bag_of_flour
  (flour_weight_needed : ℕ)
  (cost_per_bag : ℕ)
  (salt_weight_needed : ℕ)
  (salt_cost_per_pound : ℚ)
  (promotion_cost : ℕ)
  (ticket_price : ℕ)
  (tickets_sold : ℕ)
  (total_made : ℕ)
  (profit : ℕ)
  (total_flour_cost : ℕ)
  (num_bags : ℕ)
  (weight_per_bag : ℕ)
  (calc_salt_cost : ℚ := salt_weight_needed * salt_cost_per_pound)
  (calc_total_earnings : ℕ := tickets_sold * ticket_price)
  (calc_total_cost : ℚ := calc_total_earnings - profit)
  (calc_flour_cost : ℚ := calc_total_cost - calc_salt_cost - promotion_cost)
  (calc_num_bags : ℚ := calc_flour_cost / cost_per_bag)
  (calc_weight_per_bag : ℚ := flour_weight_needed / calc_num_bags) :
  flour_weight_needed = 500 ∧
  cost_per_bag = 20 ∧
  salt_weight_needed = 10 ∧
  salt_cost_per_pound = 0.2 ∧
  promotion_cost = 1000 ∧
  ticket_price = 20 ∧
  tickets_sold = 500 ∧
  total_made = 8798 ∧
  profit = 10000 - total_made ∧
  calc_salt_cost = 2 ∧
  calc_total_earnings = 10000 ∧
  calc_total_cost = 1202 ∧
  calc_flour_cost = 200 ∧
  calc_num_bags = 10 ∧
  calc_weight_per_bag = 50 :=
by {
  sorry
}

end weight_of_each_bag_of_flour_l116_116668


namespace expected_number_of_girls_left_of_all_boys_l116_116371

noncomputable def expected_girls_left_of_all_boys (boys girls : ℕ) : ℚ :=
    if boys = 10 ∧ girls = 7 then (7 : ℚ) / 11 else 0

theorem expected_number_of_girls_left_of_all_boys 
    (boys girls : ℕ) (h_boys : boys = 10) (h_girls : girls = 7) :
    expected_girls_left_of_all_boys boys girls = (7 : ℚ) / 11 :=
by
  rw [expected_girls_left_of_all_boys, if_pos]
  { simp }
  { exact ⟨h_boys, h_girls⟩ }

end expected_number_of_girls_left_of_all_boys_l116_116371


namespace value_of_g_at_five_l116_116070

def g (x : ℕ) : ℕ := x^2 - 2 * x

theorem value_of_g_at_five : g 5 = 15 := by
  sorry

end value_of_g_at_five_l116_116070


namespace compute_expression_l116_116900

-- Defining notation for the problem expression and answer simplification
theorem compute_expression : 
    9 * (2/3)^4 = 16/9 := by 
  sorry

end compute_expression_l116_116900


namespace farmer_field_area_l116_116038

variable (x : ℕ) (A : ℕ)

def planned_days : Type := {x : ℕ // 120 * x = 85 * (x + 2) + 40}

theorem farmer_field_area (h : {x : ℕ // 120 * x = 85 * (x + 2) + 40}) : A = 720 :=
by
  sorry

end farmer_field_area_l116_116038


namespace geometric_progression_l116_116267

theorem geometric_progression :
  ∃ (b1 q : ℚ), 
    (b1 * q * (q^2 - 1) = -45/32) ∧ 
    (b1 * q^3 * (q^2 - 1) = -45/512) ∧ 
    ((b1 = 6 ∧ q = 1/4) ∨ (b1 = -6 ∧ q = -1/4)) :=
by
  sorry

end geometric_progression_l116_116267


namespace investment_rate_l116_116892

theorem investment_rate (total : ℝ) (invested_at_3_percent : ℝ) (rate_3_percent : ℝ) 
                        (invested_at_5_percent : ℝ) (rate_5_percent : ℝ) 
                        (desired_income : ℝ) (remaining : ℝ) (additional_income : ℝ) (r : ℝ) : 
  total = 12000 ∧ 
  invested_at_3_percent = 5000 ∧ 
  rate_3_percent = 0.03 ∧ 
  invested_at_5_percent = 4000 ∧ 
  rate_5_percent = 0.05 ∧ 
  desired_income = 600 ∧ 
  remaining = total - invested_at_3_percent - invested_at_5_percent ∧ 
  additional_income = desired_income - (invested_at_3_percent * rate_3_percent + invested_at_5_percent * rate_5_percent) ∧ 
  r = (additional_income / remaining) * 100 → 
  r = 8.33 := 
by
  sorry

end investment_rate_l116_116892


namespace largest_positive_integer_not_sum_of_multiple_30_and_composite_l116_116675

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

noncomputable def largest_not_sum_of_multiple_30_and_composite : ℕ :=
  211

theorem largest_positive_integer_not_sum_of_multiple_30_and_composite {m : ℕ} :
  m = largest_not_sum_of_multiple_30_and_composite ↔ 
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ k : ℕ, (b = k * 30) ∨ is_composite b) → m ≠ 30 * a + b) :=
sorry

end largest_positive_integer_not_sum_of_multiple_30_and_composite_l116_116675


namespace problem_inequality_l116_116997

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x^2 + y^2 + z^2 + x*y + y*z + z*x ≤ 1) : 
  (1/x - 1) * (1/y - 1) * (1/z - 1) ≥ 9 * Real.sqrt 6 - 19 :=
sorry

end problem_inequality_l116_116997


namespace cube_side_length_l116_116194

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (h0 : s ≠ 0) : s = 6 :=
sorry

end cube_side_length_l116_116194


namespace tan_945_equals_1_l116_116402

noncomputable def tan_circular (x : ℝ) : ℝ := Real.tan x

theorem tan_945_equals_1 :
  tan_circular 945 = 1 := 
by
  sorry

end tan_945_equals_1_l116_116402


namespace gcd_90_405_l116_116278

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l116_116278


namespace find_coefficients_l116_116576

theorem find_coefficients (c d : ℝ)
  (h : ∃ u v : ℝ, u ≠ v ∧ (u^3 + c * u^2 + 10 * u + 4 = 0) ∧ (v^3 + c * v^2 + 10 * v + 4 = 0)
     ∧ (u^3 + d * u^2 + 13 * u + 5 = 0) ∧ (v^3 + d * v^2 + 13 * v + 5 = 0)) :
  (c, d) = (7, 8) :=
by
  sorry

end find_coefficients_l116_116576


namespace find_total_coins_l116_116515

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l116_116515


namespace no_natural_m_n_exists_l116_116251

theorem no_natural_m_n_exists (m n : ℕ) : 
  (0.07 = (1 : ℝ) / m + (1 : ℝ) / n) → False :=
by
  -- Normally, the proof would go here, but it's not required by the prompt
  sorry

end no_natural_m_n_exists_l116_116251


namespace smallest_mult_to_cube_l116_116457

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_mult_to_cube (n : ℕ) (h : ∃ n, ∃ k, n * y = k^3) : n = 4500 := 
  sorry

end smallest_mult_to_cube_l116_116457


namespace gcd_of_90_and_405_l116_116275

def gcd_90_405 : ℕ := Nat.gcd 90 405

theorem gcd_of_90_and_405 : gcd_90_405 = 45 :=
by
  -- proof goes here
  sorry

end gcd_of_90_and_405_l116_116275


namespace eval_g_inv_g_inv_14_l116_116838

variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)

axiom g_def : ∀ x, g x = 3 * x - 4
axiom g_inv_def : ∀ y, g_inv y = (y + 4) / 3

theorem eval_g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by
    sorry

end eval_g_inv_g_inv_14_l116_116838


namespace P_plus_Q_l116_116330

theorem P_plus_Q (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 4 → (P / (x - 4) + Q * (x + 2) = (-4 * x^2 + 16 * x + 30) / (x - 4))) : P + Q = 42 :=
sorry

end P_plus_Q_l116_116330


namespace minimum_jumps_l116_116660

theorem minimum_jumps (a b : ℕ) (h : 2 * a + 3 * b = 2016) : a + b = 673 :=
sorry

end minimum_jumps_l116_116660


namespace y_coordinate_sum_of_circle_on_y_axis_l116_116568

-- Define the properties of the circle
def center := (-3, 1)
def radius := 8

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y - 1) ^ 2 = 64

-- Define the Lean theorem statement
theorem y_coordinate_sum_of_circle_on_y_axis 
  (h₁ : center = (-3, 1)) 
  (h₂ : radius = 8) 
  (h₃ : ∀ y : ℝ, circle_eq 0 y → (∃ y1 y2 : ℝ, y = y1 ∨ y = y2) ) : 
  ∃ y1 y2 : ℝ, (y1 + y2 = 2) ∧ (circle_eq 0 y1) ∧ (circle_eq 0 y2) := 
by 
  sorry

end y_coordinate_sum_of_circle_on_y_axis_l116_116568


namespace smallest_integer_gcd_6_l116_116683

theorem smallest_integer_gcd_6 : ∃ n : ℕ, n > 100 ∧ gcd n 18 = 6 ∧ ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n :=
by
  let n := 114
  have h1 : n > 100 := sorry
  have h2 : gcd n 18 = 6 := sorry
  have h3 : ∀ m : ℕ, (m > 100 ∧ gcd m 18 = 6) → m ≥ n := sorry
  exact ⟨n, h1, h2, h3⟩

end smallest_integer_gcd_6_l116_116683


namespace sum_of_squares_ineq_l116_116351

theorem sum_of_squares_ineq (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum_sq : a^2 + b^2 + c^2 = 3) :
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 :=
sorry

end sum_of_squares_ineq_l116_116351


namespace minimum_value_of_expression_l116_116760

theorem minimum_value_of_expression (x A B C : ℝ) (hx : x > 0) 
  (hA : A = x^2 + 1/x^2) (hB : B = x - 1/x) (hC : C = B * (A + 1)) : 
  ∃ m : ℝ, m = 6.4 ∧ m = A^3 / C :=
by {
  sorry
}

end minimum_value_of_expression_l116_116760


namespace snow_white_last_trip_l116_116644

-- Definitions based on the problem's conditions
inductive Dwarf
| Happy
| Grumpy
| Dopey
| Bashful
| Sleepy
| Doc
| Sneezy

def is_adjacent (d1 d2 : Dwarf) : Prop :=
  (d1, d2) ∈ [
    (Dwarf.Happy, Dwarf.Grumpy),
    (Dwarf.Grumpy, Dwarf.Happy),
    (Dwarf.Grumpy, Dwarf.Dopey),
    (Dwarf.Dopey, Dwarf.Grumpy),
    (Dwarf.Dopey, Dwarf.Bashful),
    (Dwarf.Bashful, Dwarf.Dopey),
    (Dwarf.Bashful, Dwarf.Sleepy),
    (Dwarf.Sleepy, Dwarf.Bashful),
    (Dwarf.Sleepy, Dwarf.Doc),
    (Dwarf.Doc, Dwarf.Sleepy),
    (Dwarf.Doc, Dwarf.Sneezy),
    (Dwarf.Sneezy, Dwarf.Doc)
  ]

def boat_capacity : ℕ := 3

variable (snowWhite : Prop)

-- The theorem to prove that the dwarfs Snow White will take in the last trip are Grumpy, Bashful and Doc
theorem snow_white_last_trip 
  (h1 : snowWhite)
  (h2 : boat_capacity = 3)
  (h3 : ∀ d1 d2, is_adjacent d1 d2 → snowWhite)
  : (snowWhite ∧ (Dwarf.Grumpy ∧ Dwarf.Bashful ∧ Dwarf.Doc)) :=
sorry

end snow_white_last_trip_l116_116644


namespace burger_meal_cost_l116_116401

theorem burger_meal_cost 
  (x : ℝ) 
  (h : 5 * (x + 1) = 35) : 
  x = 6 := 
sorry

end burger_meal_cost_l116_116401


namespace find_fx_at_pi_half_l116_116977

open Real

-- Conditions on the function f
noncomputable def f (x : ℝ) : ℝ := sin(ω * x + (π / 4)) + b

-- Variables
variables (ω b : ℝ) (hpos : ω > 0)
  (T : ℝ) (hT : (2 * π / 3) < T ∧ T < π)
  (hperiod : T = 2 * π / ω)
  (hsymm : ∀ x, f(3 * π / 2 - x) = 2 - (f(x - 3 * π / 2) - 2))

-- Proof statement
theorem find_fx_at_pi_half :
  f ω b (π / 2) = 1 :=
sorry

end find_fx_at_pi_half_l116_116977


namespace alarm_clock_shows_noon_in_14_minutes_l116_116252

-- Definitions based on given problem conditions
def clockRunsSlow (clock_time real_time : ℕ) : Prop :=
  clock_time = real_time * 56 / 60

def timeSinceSet : ℕ := 210 -- 3.5 hours in minutes
def correctClockShowsNoon : ℕ := 720 -- Noon in minutes (12*60)

-- Main statement to prove
theorem alarm_clock_shows_noon_in_14_minutes :
  ∃ minutes : ℕ, clockRunsSlow (timeSinceSet * 56 / 60) timeSinceSet ∧ correctClockShowsNoon - (480 + timeSinceSet * 56 / 60) = minutes ∧ minutes = 14 := 
by
  sorry

end alarm_clock_shows_noon_in_14_minutes_l116_116252


namespace find_second_number_l116_116536

theorem find_second_number (a b c : ℝ) (h1 : a + b + c = 3.622) (h2 : a = 3.15) (h3 : c = 0.458) : b = 0.014 :=
sorry

end find_second_number_l116_116536


namespace exist_line_l1_exist_line_l2_l116_116139

noncomputable def P : ℝ × ℝ := ⟨3, 2⟩

def line1_eq (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2_eq (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def perpend_line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

def line_l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0
def line_l2_case1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def line_l2_case2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem exist_line_l1 : ∃ (x y : ℝ), line1_eq x y ∧ line2_eq x y ∧ perpend_line_eq x y → line_l1 x y :=
by
  sorry

theorem exist_line_l2 : ∃ (x y : ℝ), line1_eq x y ∧ line2_eq x y ∧ ((line_l2_case1 x y) ∨ (line_l2_case2 x y)) :=
by
  sorry

end exist_line_l1_exist_line_l2_l116_116139


namespace total_cost_of_motorcycle_l116_116049

-- Definitions from conditions
def total_cost (x : ℝ) := 0.20 * x = 400

-- The theorem to prove
theorem total_cost_of_motorcycle (x : ℝ) (h : total_cost x) : x = 2000 := 
by
  sorry

end total_cost_of_motorcycle_l116_116049


namespace inequality_holds_if_and_only_if_l116_116824

noncomputable def absolute_inequality (x a : ℝ) : Prop :=
  |x - 3| + |x - 4| + |x - 5| < a

theorem inequality_holds_if_and_only_if (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, absolute_inequality x a) ↔ a > 4 := 
sorry

end inequality_holds_if_and_only_if_l116_116824


namespace power_function_passes_through_1_1_l116_116473

theorem power_function_passes_through_1_1 (a : ℝ) : (1 : ℝ) ^ a = 1 := 
by
  sorry

end power_function_passes_through_1_1_l116_116473


namespace ratio_area_rhombus_to_square_l116_116045

namespace Geometry

open Real

/-- Given a rhombus with an angle of 30 degrees,
    an inscribed circle, and a square inscribed in the circle,
    the ratio of the area of the rhombus to the area of the square is 4. -/
theorem ratio_area_rhombus_to_square (a : ℝ) (h₀: 0 < a) :
  let S1 := (a^2 * (1/2))
  let r := (a / 4)
  let b := (r * sqrt 2)
  let S2 := b^2 
  S1 / S2 = 4 :=
by
  sorry

end Geometry

end ratio_area_rhombus_to_square_l116_116045


namespace gcd_90_405_l116_116269

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l116_116269


namespace abel_overtake_kelly_chris_overtake_both_l116_116598

-- Given conditions and variables
variable (d : ℝ)  -- distance at which Abel overtakes Kelly
variable (d_c : ℝ)  -- distance at which Chris overtakes both Kelly and Abel
variable (t_k : ℝ)  -- time taken by Kelly to run d meters
variable (t_a : ℝ)  -- time taken by Abel to run (d + 3) meters
variable (t_c : ℝ)  -- time taken by Chris to run the required distance
variable (k_speed : ℝ := 9)  -- Kelly's speed
variable (a_speed : ℝ := 9.5)  -- Abel's speed
variable (c_speed : ℝ := 10)  -- Chris's speed
variable (head_start_k : ℝ := 3)  -- Kelly's head start over Abel
variable (head_start_c : ℝ := 2)  -- Chris's head start behind Abel
variable (lost_by : ℝ := 0.75)  -- Abel lost by distance

-- Proof problem for Abel overtaking Kelly
theorem abel_overtake_kelly 
  (hk : t_k = d / k_speed) 
  (ha : t_a = (d + head_start_k) / a_speed) 
  (h_lost : lost_by = 0.75):
  d + lost_by = 54.75 := 
sorry

-- Proof problem for Chris overtaking both Kelly and Abel
theorem chris_overtake_both 
  (hc : t_c = (d_c + 5) / c_speed)
  (h_56 : d_c = 56):
  d_c = c_speed * (56 / c_speed) :=
sorry

end abel_overtake_kelly_chris_overtake_both_l116_116598


namespace gcd_90_405_l116_116276

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end gcd_90_405_l116_116276


namespace probability_red_balls_fourth_draw_l116_116850

theorem probability_red_balls_fourth_draw :
  let p_red := 2 / 10
  let p_white := 8 / 10
  p_red * p_red * p_white * p_white * 3 / 10 + 
  p_red * p_white * p_red * p_white * 2 / 10 + 
  p_white * p_red * p_red * p_red = 0.0434 :=
sorry

end probability_red_balls_fourth_draw_l116_116850


namespace base_b_sum_correct_l116_116114

def sum_double_digit_numbers (b : ℕ) : ℕ :=
  (b * (b - 1) * (b ^ 2 - b + 1)) / 2

def base_b_sum (b : ℕ) : ℕ :=
  b ^ 2 + 12 * b + 5

theorem base_b_sum_correct : ∃ b : ℕ, sum_double_digit_numbers b = base_b_sum b ∧ b = 15 :=
by
  sorry

end base_b_sum_correct_l116_116114


namespace compute_f_pi_div_2_l116_116980

def f (x : ℝ) (ω : ℝ) (b : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4) + b

theorem compute_f_pi_div_2 :
  ∀ (b ω : ℝ),
    ω > 0 →
    (∃ T, T = 2 * Real.pi / ω ∧ (2 * Real.pi / 3 < T ∧ T < Real.pi)) →
    (∀ x : ℝ, Real.sin (ω * (3 * Real.pi / 2 - x) + Real.pi / 4) + 2 = f x ω 2) →
    f (Real.pi / 2) ω 2 = 1 :=
by
  intros b ω hω hT hSym
  sorry

end compute_f_pi_div_2_l116_116980


namespace complement_of_A_l116_116145

-- Definition of the universal set U and the set A
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | (x - 1) * (x + 2) > 0}

-- Theorem statement for the complement of A in U
theorem complement_of_A:
  (U \ A) = {x | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end complement_of_A_l116_116145


namespace rotated_squares_overlap_area_l116_116085

noncomputable def total_overlap_area (side_length : ℝ) : ℝ :=
  let base_area := side_length ^ 2
  3 * base_area

theorem rotated_squares_overlap_area : total_overlap_area 8 = 192 := by
  sorry

end rotated_squares_overlap_area_l116_116085


namespace repeating_decimal_sum_l116_116783

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (∀ d, d ∣ a ∧ d ∣ b ↔ d = 1) ∧ 
  (a + b = 134) ∧ (∃ x : ℚ, x = 35 / 99 ∧ (x = 0.353535...)) :=
begin
  sorry
end

end repeating_decimal_sum_l116_116783


namespace sequence_periodic_of_period_9_l116_116066

theorem sequence_periodic_of_period_9 (a : ℕ → ℤ) (h : ∀ n, a (n + 2) = |a (n + 1)| - a n) (h_nonzero : ∃ n, a n ≠ 0) :
  ∃ m, ∃ k, m > 0 ∧ k > 0 ∧ (∀ n, a (n + m + k) = a (n + m)) ∧ k = 9 :=
by
  sorry

end sequence_periodic_of_period_9_l116_116066


namespace expected_value_of_girls_left_of_boys_l116_116373

def num_girls_to_left_of_all_boys (boys girls : ℕ) : ℚ :=
  (boys + girls : ℚ) / (boys + 1)

theorem expected_value_of_girls_left_of_boys :
  num_girls_to_left_of_all_boys 10 7 = 7 / 11 := by
  sorry

end expected_value_of_girls_left_of_boys_l116_116373


namespace inequality_proof_l116_116624

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_proof_l116_116624


namespace num_integer_ks_l116_116956

theorem num_integer_ks (k : Int) :
  (∃ a b c d : Int, (2*x + a) * (x + b) = 2*x^2 - k*x + 6 ∨
                   (2*x + c) * (x + d) = 2*x^2 - k*x + 6) →
  ∃ ks : Finset Int, ks.card = 6 ∧ k ∈ ks :=
sorry

end num_integer_ks_l116_116956


namespace Liza_rent_l116_116993

theorem Liza_rent :
  (800 - R + 1500 - 117 - 100 - 70 = 1563) -> R = 450 :=
by
  intros h
  sorry

end Liza_rent_l116_116993


namespace jennifer_fruits_left_l116_116051

theorem jennifer_fruits_left:
  (apples = 2 * pears) →
  (cherries = oranges / 2) →
  (grapes = 3 * apples) →
  pears = 15 →
  oranges = 30 →
  pears_given = 3 →
  oranges_given = 5 →
  apples_given = 5 →
  cherries_given = 7 →
  grapes_given = 3 →
  (remaining_fruits =
    (pears - pears_given) +
    (oranges - oranges_given) +
    (apples - apples_given) +
    (cherries - cherries_given) +
    (grapes - grapes_given)) →
  remaining_fruits = 157 :=
by
  intros
  sorry

end jennifer_fruits_left_l116_116051


namespace point_outside_circle_l116_116040

theorem point_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, (a * x + b * y = 1) ∧ (x^2 + y^2 = 1)) :
  a^2 + b^2 > 1 := by
  sorry

end point_outside_circle_l116_116040


namespace tax_difference_is_correct_l116_116969

-- Define the original price and discount rate as constants
def original_price : ℝ := 50
def discount_rate : ℝ := 0.10

-- Define the state and local sales tax rates as constants
def state_sales_tax_rate : ℝ := 0.075
def local_sales_tax_rate : ℝ := 0.07

-- Calculate the discounted price
def discounted_price : ℝ := original_price * (1 - discount_rate)

-- Calculate state and local sales taxes after discount
def state_sales_tax : ℝ := discounted_price * state_sales_tax_rate
def local_sales_tax : ℝ := discounted_price * local_sales_tax_rate

-- Calculate the difference between state and local sales taxes
def tax_difference : ℝ := state_sales_tax - local_sales_tax

-- The proof to show that the difference is 0.225
theorem tax_difference_is_correct : tax_difference = 0.225 := by
  sorry

end tax_difference_is_correct_l116_116969


namespace solution_set_no_pos_ab_l116_116300

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

theorem solution_set :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 2 / 3 ≤ x ∧ x ≤ 4} :=
by sorry

theorem no_pos_ab :
  ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ 1 / a + 2 / b = 4 :=
by sorry

end solution_set_no_pos_ab_l116_116300


namespace smallest_three_digit_integer_l116_116363

theorem smallest_three_digit_integer (n : ℕ) : 
  100 ≤ n ∧ n < 1000 ∧ ¬ (n - 1 ∣ (n!)) ↔ n = 1004 := 
by
  sorry

end smallest_three_digit_integer_l116_116363


namespace orange_cost_l116_116994

-- Definitions based on the conditions
def dollar_per_pound := 5 / 6
def pounds : ℕ := 18
def total_cost := pounds * dollar_per_pound

-- The statement to be proven
theorem orange_cost : total_cost = 15 :=
by
  sorry

end orange_cost_l116_116994


namespace avg_salary_of_employees_is_1500_l116_116322

-- Definitions for conditions
def num_employees : ℕ := 20
def num_people_incl_manager : ℕ := 21
def manager_salary : ℝ := 4650
def salary_increase : ℝ := 150

-- Definition for average salary of employees excluding the manager
def avg_salary_employees (A : ℝ) : Prop :=
    21 * (A + salary_increase) = 20 * A + manager_salary

-- The target proof statement
theorem avg_salary_of_employees_is_1500 :
  ∃ A : ℝ, avg_salary_employees A ∧ A = 1500 := by
  -- Proof goes here
  sorry

end avg_salary_of_employees_is_1500_l116_116322


namespace volume_of_rectangular_prism_l116_116657

theorem volume_of_rectangular_prism (l w h : ℕ) (x : ℕ) 
  (h_ratio : l = 3 * x ∧ w = 2 * x ∧ h = x)
  (h_edges : 4 * l + 4 * w + 4 * h = 72) : 
  l * w * h = 162 := 
by
  sorry

end volume_of_rectangular_prism_l116_116657


namespace treasure_coins_l116_116507

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l116_116507


namespace largest_integer_not_sum_of_30_and_composite_l116_116678

theorem largest_integer_not_sum_of_30_and_composite : 
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b < 30 ∧ ¬ prime b ∧ (n = 30 * a + b) → n = 157 :=
by
  sorry

end largest_integer_not_sum_of_30_and_composite_l116_116678


namespace repeated_decimal_to_fraction_l116_116781

theorem repeated_decimal_to_fraction :
  ∀ (a b : ℕ), (0.35.recursor = 35 / 99 ∧ Nat.gcd 35 99 = 1) → a + b = 134 := by
  sorry

end repeated_decimal_to_fraction_l116_116781


namespace mutually_exclusive_event_l116_116102

def shooting_twice : Type := 
  { hit_first : Bool // hit_first = true ∨ hit_first = false }

def hitting_at_least_once (shoots : shooting_twice) : Prop :=
  shoots.1 ∨ (¬shoots.1 ∧ true)

def missing_both_times (shoots : shooting_twice) : Prop :=
  ¬shoots.1 ∧ (¬true ∨ true)

def mutually_exclusive (A : Prop) (B : Prop) : Prop :=
  A ∨ B → ¬ (A ∧ B)

theorem mutually_exclusive_event :
  ∀ shoots : shooting_twice, 
  mutually_exclusive (hitting_at_least_once shoots) (missing_both_times shoots) :=
by
  intro shoots
  unfold mutually_exclusive
  sorry

end mutually_exclusive_event_l116_116102


namespace remainder_317_l116_116469

theorem remainder_317 (y : ℤ)
  (h1 : 4 + y ≡ 9 [ZMOD 16])
  (h2 : 6 + y ≡ 8 [ZMOD 81])
  (h3 : 8 + y ≡ 49 [ZMOD 625]) :
  y ≡ 317 [ZMOD 360] := 
sorry

end remainder_317_l116_116469


namespace area_of_region_l116_116408

theorem area_of_region : 
  (∃ x y : ℝ, |5 * x - 10| + |4 * y + 20| ≤ 10) →
  ∃ area : ℝ, 
  area = 10 :=
sorry

end area_of_region_l116_116408


namespace number_of_new_players_l116_116350

-- Definitions based on conditions
def total_groups : Nat := 2
def players_per_group : Nat := 5
def returning_players : Nat := 6

-- Convert conditions to definition
def total_players : Nat := total_groups * players_per_group

-- Define what we want to prove
def new_players : Nat := total_players - returning_players

-- The proof problem statement
theorem number_of_new_players :
  new_players = 4 :=
by
  sorry

end number_of_new_players_l116_116350


namespace tickets_to_buy_l116_116230

theorem tickets_to_buy
  (ferris_wheel_cost : Float := 2.0)
  (roller_coaster_cost : Float := 7.0)
  (multiple_rides_discount : Float := 1.0)
  (newspaper_coupon : Float := 1.0) :
  (ferris_wheel_cost + roller_coaster_cost - multiple_rides_discount - newspaper_coupon = 7.0) :=
by
  sorry

end tickets_to_buy_l116_116230


namespace advertisement_revenue_l116_116100

theorem advertisement_revenue
  (cost_per_program : ℝ)
  (num_programs : ℕ)
  (selling_price_per_program : ℝ)
  (desired_profit : ℝ)
  (total_cost_production : ℝ)
  (total_revenue_sales : ℝ)
  (total_revenue_needed : ℝ)
  (revenue_from_advertisements : ℝ) :
  cost_per_program = 0.70 →
  num_programs = 35000 →
  selling_price_per_program = 0.50 →
  desired_profit = 8000 →
  total_cost_production = cost_per_program * num_programs →
  total_revenue_sales = selling_price_per_program * num_programs →
  total_revenue_needed = total_cost_production + desired_profit →
  revenue_from_advertisements = total_revenue_needed - total_revenue_sales →
  revenue_from_advertisements = 15000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end advertisement_revenue_l116_116100


namespace problem_a2_minus_b2_problem_a3_minus_b3_l116_116037

variable (a b : ℝ)
variable (h1 : a + b = 8)
variable (h2 : a - b = 4)

theorem problem_a2_minus_b2 :
  a^2 - b^2 = 32 := 
by
sorry

theorem problem_a3_minus_b3 :
  a^3 - b^3 = 208 := 
by
sorry

end problem_a2_minus_b2_problem_a3_minus_b3_l116_116037


namespace calc_xy_square_l116_116566

theorem calc_xy_square
  (x y z : ℝ)
  (h1 : 2 * x * (y + z) = 1 + y * z)
  (h2 : 1 / x - 2 / y = 3 / 2)
  (h3 : x + y + 1 / 2 = 0) :
  (x + y + z) ^ 2 = 1 :=
by
  sorry

end calc_xy_square_l116_116566


namespace gcd_of_90_and_405_l116_116273

def gcd_90_405 : ℕ := Nat.gcd 90 405

theorem gcd_of_90_and_405 : gcd_90_405 = 45 :=
by
  -- proof goes here
  sorry

end gcd_of_90_and_405_l116_116273


namespace neg_sqrt_17_estimate_l116_116740

theorem neg_sqrt_17_estimate : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end neg_sqrt_17_estimate_l116_116740


namespace arithmetic_sequence_sum_l116_116863

theorem arithmetic_sequence_sum :
  ∀ (x y : ℕ), (∀ (a b c : ℕ), b - a = c - b → c - b = 5) ∧
               (3 + 5 * 1 = 8) ∧
               (8 + 5 * 1 = 13) ∧
               (x + 5 * 1 = y) ∧
               (y + 5 * 1 = 33) →
               x + y = 51 :=
by
  intros x y h
  sorry

end arithmetic_sequence_sum_l116_116863


namespace possible_values_of_a_l116_116663

theorem possible_values_of_a :
  ∃ a b c : ℤ, 
    (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) ↔ 
    (a = 3 ∨ a = 7) :=
by
  sorry

end possible_values_of_a_l116_116663


namespace tan_half_angle_lt_l116_116065

theorem tan_half_angle_lt (x : ℝ) (h : 0 < x ∧ x ≤ π / 2) : 
  Real.tan (x / 2) < x := 
by
  sorry

end tan_half_angle_lt_l116_116065


namespace num_children_proof_l116_116050

-- Definitions and Main Problem
def legs_of_javier : ℕ := 2
def legs_of_wife : ℕ := 2
def legs_per_child : ℕ := 2
def legs_per_dog : ℕ := 4
def legs_of_cat : ℕ := 4
def num_dogs : ℕ := 2
def num_cats : ℕ := 1
def total_legs : ℕ := 22

-- Proof problem: Prove that the number of children (num_children) is equal to 3
theorem num_children_proof : ∃ num_children : ℕ, legs_of_javier + legs_of_wife + (num_children * legs_per_child) + (num_dogs * legs_per_dog) + (num_cats * legs_of_cat) = total_legs ∧ num_children = 3 :=
by
  -- Proof goes here
  sorry

end num_children_proof_l116_116050


namespace mixture_weight_l116_116080

theorem mixture_weight :
  let weight_a_per_liter := 900 -- in gm
  let weight_b_per_liter := 750 -- in gm
  let ratio_a := 3
  let ratio_b := 2
  let total_volume := 4 -- in liters
  let volume_a := (ratio_a / (ratio_a + ratio_b)) * total_volume
  let volume_b := (ratio_b / (ratio_a + ratio_b)) * total_volume
  let weight_a := volume_a * weight_a_per_liter
  let weight_b := volume_b * weight_b_per_liter
  let total_weight_gm := weight_a + weight_b
  let total_weight_kg := total_weight_gm / 1000 
  total_weight_kg = 3.36 :=
by
  sorry

end mixture_weight_l116_116080


namespace pirates_treasure_l116_116520

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l116_116520


namespace fraction_division_l116_116730

theorem fraction_division :
  (5 / 4) / (8 / 15) = 75 / 32 :=
sorry

end fraction_division_l116_116730


namespace xy_sufficient_not_necessary_l116_116078

theorem xy_sufficient_not_necessary (x y : ℝ) :
  (xy ≠ 6) → (x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → xy ≠ 6) := by
  sorry

end xy_sufficient_not_necessary_l116_116078


namespace prove_number_of_cows_l116_116721

-- Define the conditions: Cows, Sheep, Pigs, Total animals
variables (C S P : ℕ)

-- Condition 1: Twice as many sheep as cows
def condition1 : Prop := S = 2 * C

-- Condition 2: Number of Pigs is 3 times the number of sheep
def condition2 : Prop := P = 3 * S

-- Condition 3: Total number of animals is 108
def condition3 : Prop := C + S + P = 108

-- The theorem to prove
theorem prove_number_of_cows (h1 : condition1 C S) (h2 : condition2 S P) (h3 : condition3 C S P) : C = 12 :=
sorry

end prove_number_of_cows_l116_116721


namespace sum_squares_inequality_l116_116353

theorem sum_squares_inequality {a b c : ℝ} 
  (h1 : a > 0)
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := 
sorry

end sum_squares_inequality_l116_116353


namespace sum_of_four_consecutive_even_numbers_l116_116656

theorem sum_of_four_consecutive_even_numbers (n : ℤ) (h : n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344) :
  n + (n + 2) + (n + 4) + (n + 6) = 36 := sorry

end sum_of_four_consecutive_even_numbers_l116_116656


namespace part_I_solution_part_II_solution_l116_116758

-- Part (I) proof problem: Prove the solution set for a specific inequality
theorem part_I_solution (x : ℝ) : -6 < x ∧ x < 10 / 3 → |2 * x - 2| + x + 1 < 9 :=
by
  sorry

-- Part (II) proof problem: Prove the range of 'a' for a given inequality to hold
theorem part_II_solution (a : ℝ) : (-3 ≤ a ∧ a ≤ 17 / 3) →
  (∀ x : ℝ, x ≥ 2 → |a * x + a - 4| + x + 1 ≤ (x + 2)^2) :=
by
  sorry

end part_I_solution_part_II_solution_l116_116758


namespace Blair_17th_turn_l116_116454

/-
  Jo begins counting by saying "5". Blair then continues the sequence, each time saying a number that is 2 more than the last number Jo said. Jo increments by 1 each turn after Blair. They alternate turns.
  Prove that Blair says the number 55 on her 17th turn.
-/

def Jo_initial := 5
def increment_Jo := 1
def increment_Blair := 2

noncomputable def blair_sequence (n : ℕ) : ℕ :=
  Jo_initial + increment_Blair + (n - 1) * (increment_Jo + increment_Blair)

theorem Blair_17th_turn : blair_sequence 17 = 55 := by
    sorry

end Blair_17th_turn_l116_116454


namespace calculate_overhead_cost_l116_116837

noncomputable def overhead_cost (prod_cost revenue_cost : ℕ) (num_performances : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost - num_performances * prod_cost

theorem calculate_overhead_cost :
  overhead_cost 7000 16000 9 (9 * 16000) = 81000 :=
by
  sorry

end calculate_overhead_cost_l116_116837


namespace geometric_sequence_a5_l116_116606

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 * a 5 = 16) (h2 : a 4 = 8) (h3 : ∀ n, a n > 0) : a 5 = 16 := 
by
  sorry

end geometric_sequence_a5_l116_116606


namespace sum_of_angles_is_290_l116_116599

-- Given conditions
def angle_A : ℝ := 40
def angle_C : ℝ := 70
def angle_D : ℝ := 50
def angle_F : ℝ := 60

-- Calculate angle B (which is same as angle E)
def angle_B : ℝ := 180 - angle_A - angle_C
def angle_E := angle_B  -- by the condition that B and E are identical

-- Total sum of angles
def total_angle_sum : ℝ := angle_A + angle_B + angle_C + angle_D + angle_F

-- Theorem statement
theorem sum_of_angles_is_290 : total_angle_sum = 290 := by
  sorry

end sum_of_angles_is_290_l116_116599


namespace fraction_multiplication_l116_116218

theorem fraction_multiplication (x : ℚ) (h : x = 236 / 100) : x * 3 = 177 / 25 :=
by
  sorry

end fraction_multiplication_l116_116218


namespace range_of_a_l116_116141

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x >= 0 then -x + 3 * a else x^2 - a * x + 1

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → f a x1 ≥ f a x2) ↔ (0 <= a ∧ a <= 1/3) :=
by
  sorry

end range_of_a_l116_116141


namespace math_problem_l116_116340

/-- Lean translation of the mathematical problem.
Given \(a, b \in \mathbb{R}\) such that \(a^2 + b^2 = a^2 b^2\) and 
\( |a| \neq 1 \) and \( |b| \neq 1 \), prove that 
\[
\frac{a^7}{(1 - a)^2} - \frac{a^7}{(1 + a)^2} = 
\frac{b^7}{(1 - b)^2} - \frac{b^7}{(1 + b)^2}.
\]
-/
theorem math_problem 
  (a b : ℝ) 
  (h1 : a^2 + b^2 = a^2 * b^2) 
  (h2 : |a| ≠ 1) 
  (h3 : |b| ≠ 1) : 
  (a^7 / (1 - a)^2 - a^7 / (1 + a)^2) = 
  (b^7 / (1 - b)^2 - b^7 / (1 + b)^2) := 
by 
  -- Proof is omitted for this exercise.
  sorry

end math_problem_l116_116340


namespace polygon_largest_area_l116_116289

-- Definition for the area calculation of each polygon based on given conditions
def area_A : ℝ := 3 * 1 + 2 * 0.5
def area_B : ℝ := 6 * 1
def area_C : ℝ := 4 * 1 + 3 * 0.5
def area_D : ℝ := 5 * 1 + 1 * 0.5
def area_E : ℝ := 7 * 1

-- Theorem stating the problem
theorem polygon_largest_area :
  area_E = max (max (max (max area_A area_B) area_C) area_D) area_E :=
by
  -- The proof steps would go here.
  sorry

end polygon_largest_area_l116_116289


namespace repeating_decimal_35_as_fraction_l116_116788

noncomputable def repeating_decimal_as_fraction (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ a b : ℕ, (0.\overline{35}).repr = n / d ∧ Int.gcd a b = 1 ∧ a + b = 134

theorem repeating_decimal_35_as_fraction :
  repeating_decimal_as_fraction 35 99 := sorry

end repeating_decimal_35_as_fraction_l116_116788


namespace a_beats_b_by_32_meters_l116_116689

-- Define the known conditions.
def distance_a_in_t : ℕ := 224 -- Distance A runs in 28 seconds
def time_a : ℕ := 28 -- Time A takes to run 224 meters
def distance_b_in_t : ℕ := 224 -- Distance B runs in 32 seconds
def time_b : ℕ := 32 -- Time B takes to run 224 meters

-- Define the speeds.
def speed_a : ℕ := distance_a_in_t / time_a
def speed_b : ℕ := distance_b_in_t / time_b

-- Define the distances each runs in 32 seconds.
def distance_a_in_32_sec : ℕ := speed_a * 32
def distance_b_in_32_sec : ℕ := speed_b * 32

-- The proof statement
theorem a_beats_b_by_32_meters :
  distance_a_in_32_sec - distance_b_in_32_sec = 32 := 
sorry

end a_beats_b_by_32_meters_l116_116689


namespace expected_value_girls_left_of_boys_l116_116375

theorem expected_value_girls_left_of_boys :
  let boys := 10
      girls := 7
      students := boys + girls in
  (∀ (lineup : Finset (Fin students)), let event := { l : Finset (Fin students) | ∃ g : Fin girls, g < boys - 1} in
       ProbabilityTheory.expectation (λ p, (lineup ∩ event).card)) = 7 / 11 := 
sorry

end expected_value_girls_left_of_boys_l116_116375


namespace pine_count_25_or_26_l116_116211

-- Define the total number of trees
def total_trees : ℕ := 101

-- Define the constraints
def poplar_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 1 → t.nth i = some 1 → t.nth (i + 1) ≠ some 1

def birch_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 2 → t.nth i = some 2 → t.nth (i + 1) ≠ some 2 ∧ t.nth (i + 2) ≠ some 2

def pine_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 3 → t.nth i = some 3 → t.nth (i + 1) ≠ some 3 ∧ t.nth (i + 2) ≠ some 3 ∧ t.nth (i + 3) ≠ some 3

-- Define the number of pines
def number_of_pines (t : List ℕ) : ℕ := t.countp (λ x, x = 3)

-- The main theorem asserting the number of pines possible
theorem pine_count_25_or_26 (t : List ℕ) (htotal : t.length = total_trees) 
    (hpoplar : poplar_spacing t) (hbirch : birch_spacing t) (hpine : pine_spacing t) :
  number_of_pines t = 25 ∨ number_of_pines t = 26 := 
sorry

end pine_count_25_or_26_l116_116211


namespace cos_alpha_sub_beta_sin_alpha_l116_116423

open Real

variables (α β : ℝ)

-- Conditions:
-- 0 < α < π / 2
def alpha_in_first_quadrant := 0 < α ∧ α < π / 2

-- -π / 2 < β < 0
def beta_in_fourth_quadrant := -π / 2 < β ∧ β < 0

-- sin β = -5/13
def sin_beta := sin β = -5 / 13

-- tan(α - β) = 4/3
def tan_alpha_sub_beta := tan (α - β) = 4 / 3

-- Theorem statements (follows directly from the conditions and the equivalence):
theorem cos_alpha_sub_beta : alpha_in_first_quadrant α → beta_in_fourth_quadrant β → sin_beta β → tan_alpha_sub_beta α β → cos (α - β) = 3 / 5 := sorry

theorem sin_alpha : alpha_in_first_quadrant α → beta_in_fourth_quadrant β → sin_beta β → tan_alpha_sub_beta α β → sin α = 33 / 65 := sorry

end cos_alpha_sub_beta_sin_alpha_l116_116423


namespace integer_solutions_l116_116743

theorem integer_solutions (m n : ℤ) :
  m^3 - n^3 = 2 * m * n + 8 ↔ (m = 0 ∧ n = -2) ∨ (m = 2 ∧ n = 0) :=
sorry

end integer_solutions_l116_116743


namespace neg_sqrt_17_bounds_l116_116738

theorem neg_sqrt_17_bounds :
  (16 < 17) ∧ (17 < 25) ∧ (16 = 4^2) ∧ (25 = 5^2) ∧ (4 < Real.sqrt 17) ∧ (Real.sqrt 17 < 5) →
  (-5 < -Real.sqrt 17) ∧ (-Real.sqrt 17 < -4) :=
by
  sorry

end neg_sqrt_17_bounds_l116_116738


namespace max_value_ineq_l116_116076

theorem max_value_ineq (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : a + b + c = 1) :
  (a + 3 * b + 5 * c) * (a + b / 3 + c / 5) ≤ 9 / 5 :=
sorry

end max_value_ineq_l116_116076


namespace perimeter_of_tangents_triangle_l116_116669

theorem perimeter_of_tangents_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
    (4 * a * Real.sqrt (a * b)) / (a - b) = 4 * a * (Real.sqrt (a * b) / (a - b)) := 
sorry

end perimeter_of_tangents_triangle_l116_116669


namespace seminar_total_cost_l116_116554

theorem seminar_total_cost 
  (regular_fee : ℝ)
  (discount_rate : ℝ)
  (num_teachers : ℕ) 
  (food_allowance_per_teacher : ℝ)
  (total_cost : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_rate = 0.05)
  (h3 : num_teachers = 10) 
  (h4 : food_allowance_per_teacher = 10)
  (h5 : total_cost = regular_fee * num_teachers * (1 - discount_rate) + food_allowance_per_teacher * num_teachers) :
  total_cost = 1525 := 
sorry

end seminar_total_cost_l116_116554


namespace find_f_13_l116_116582

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

theorem find_f_13 (f : ℝ → ℝ) 
  (h_period : periodic f 1.5) 
  (h_val : f 1 = 20) 
  : f 13 = 20 :=
by
  sorry

end find_f_13_l116_116582


namespace range_of_a_l116_116933

theorem range_of_a (a : ℝ) : 
  (M = {x : ℝ | 2 * x + 1 < 3}) → 
  (N = {x : ℝ | x < a}) → 
  (M ∩ N = N) ↔ a ≤ 1 :=
by
  let M := {x : ℝ | 2 * x + 1 < 3}
  let N := {x : ℝ | x < a}
  simp [Set.subset_def]
  sorry

end range_of_a_l116_116933


namespace complex_fifth_roots_wrong_statement_l116_116170

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)
noncomputable def y : ℂ := Complex.exp (-2 * Real.pi * Complex.I / 5)

theorem complex_fifth_roots_wrong_statement :
  ¬(x^5 + y^5 = 1) :=
sorry

end complex_fifth_roots_wrong_statement_l116_116170


namespace triangle_other_side_length_l116_116428

theorem triangle_other_side_length (a b : ℝ) (c : ℝ) (h_a : a = 3) (h_b : b = 4) (h_right_angle : c * c = a * a + b * b ∨ a * a = c * c + b * b):
  c = Real.sqrt 7 ∨ c = 5 :=
by
  sorry

end triangle_other_side_length_l116_116428


namespace fraction_value_l116_116848

theorem fraction_value : (5 - Real.sqrt 4) / (5 + Real.sqrt 4) = 3 / 7 := by
  sorry

end fraction_value_l116_116848


namespace asia_discount_problem_l116_116563

theorem asia_discount_problem
  (originalPrice : ℝ)
  (storeDiscount : ℝ)
  (memberDiscount : ℝ)
  (finalPriceUSD : ℝ)
  (exchangeRate : ℝ)
  (finalDiscountPercentage : ℝ) :
  originalPrice = 300 →
  storeDiscount = 0.20 →
  memberDiscount = 0.10 →
  finalPriceUSD = 224 →
  exchangeRate = 1.10 →
  finalDiscountPercentage = 28 :=
by
  sorry

end asia_discount_problem_l116_116563


namespace minutes_sean_played_each_day_l116_116998

-- Define the given conditions
def t : ℕ := 1512                               -- Total minutes played by Sean and Indira
def i : ℕ := 812                                -- Total minutes played by Indira
def d : ℕ := 14                                 -- Number of days Sean played

-- Define the to-be-proved statement
theorem minutes_sean_played_each_day : (t - i) / d = 50 :=
by
  sorry

end minutes_sean_played_each_day_l116_116998


namespace clara_gave_10_stickers_l116_116005

-- Defining the conditions
def initial_stickers : ℕ := 100
def remaining_after_boy (B : ℕ) : ℕ := initial_stickers - B
def remaining_after_friends (B : ℕ) : ℕ := (remaining_after_boy B) / 2

-- Theorem stating that Clara gave 10 stickers to the boy
theorem clara_gave_10_stickers (B : ℕ) (h : remaining_after_friends B = 45) : B = 10 :=
by
  sorry

end clara_gave_10_stickers_l116_116005


namespace ratio_of_first_to_second_ball_l116_116613

theorem ratio_of_first_to_second_ball 
  (x y z : ℕ) 
  (h1 : 3 * x = 27) 
  (h2 : y = 18) 
  (h3 : z = 3 * x) : 
  x / y = 1 / 2 := 
sorry

end ratio_of_first_to_second_ball_l116_116613


namespace f_comp_g_eq_g_comp_f_iff_l116_116803

variable {R : Type} [CommRing R]

def f (m n : R) (x : R) : R := m * x ^ 2 + n
def g (p q : R) (x : R) : R := p * x + q

theorem f_comp_g_eq_g_comp_f_iff (m n p q : R) :
  (∀ x : R, f m n (g p q x) = g p q (f m n x)) ↔ n * (1 - p ^ 2) - q * (1 - m) = 0 :=
by
  sorry

end f_comp_g_eq_g_comp_f_iff_l116_116803


namespace coprime_repeating_decimal_sum_l116_116792

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l116_116792


namespace root_expression_value_l116_116983

theorem root_expression_value 
  (p q r s : ℝ)
  (h1 : p + q + r + s = 15)
  (h2 : p*q + p*r + p*s + q*r + q*s + r*s = 35)
  (h3 : p*q*r + p*q*s + q*r*s + p*r*s = 27)
  (h4 : p*q*r*s = 9)
  (h5 : ∀ x : ℝ, x^4 - 15*x^3 + 35*x^2 - 27*x + 9 = 0 → (x = p ∨ x = q ∨ x = r ∨ x = s)) :
  (p / (1 / p + q*r) + q / (1 / q + r*s) + r / (1 / r + s*p) + s / (1 / s + p*q) = 155 / 123) := 
sorry

end root_expression_value_l116_116983


namespace not_equal_fractions_l116_116869

theorem not_equal_fractions :
  ¬ ((14 / 12 = 7 / 6) ∧
     (1 + 1 / 6 = 7 / 6) ∧
     (21 / 18 = 7 / 6) ∧
     (1 + 2 / 12 = 7 / 6) ∧
     (1 + 1 / 3 = 7 / 6)) :=
by 
  sorry

end not_equal_fractions_l116_116869


namespace students_per_van_l116_116922

def number_of_boys : ℕ := 60
def number_of_girls : ℕ := 80
def number_of_vans : ℕ := 5

theorem students_per_van : (number_of_boys + number_of_girls) / number_of_vans = 28 := by
  sorry

end students_per_van_l116_116922


namespace feasible_measures_l116_116225

-- Conditions for the problem
def condition1 := "Replace iron filings with iron pieces"
def condition2 := "Use excess zinc pieces instead of iron pieces"
def condition3 := "Add a small amount of CuSO₄ solution to the dilute hydrochloric acid"
def condition4 := "Add CH₃COONa solid to the dilute hydrochloric acid"
def condition5 := "Add sulfuric acid of the same molar concentration to the dilute hydrochloric acid"
def condition6 := "Add potassium sulfate solution to the dilute hydrochloric acid"
def condition7 := "Slightly heat (without considering the volatilization of HCl)"
def condition8 := "Add NaNO₃ solid to the dilute hydrochloric acid"

-- The criteria for the problem
def isFeasible (cond : String) : Prop :=
  cond = condition1 ∨ cond = condition2 ∨ cond = condition3 ∨ cond = condition7

theorem feasible_measures :
  ∀ cond, 
  cond ≠ condition4 →
  cond ≠ condition5 →
  cond ≠ condition6 →
  cond ≠ condition8 →
  isFeasible cond :=
by
  intros
  sorry

end feasible_measures_l116_116225


namespace connie_total_markers_l116_116263

/-
Connie has 4 different types of markers: red, blue, green, and yellow.
She has twice as many red markers as green markers.
She has three times as many blue markers as red markers.
She has four times as many yellow markers as green markers.
She has 36 green markers.
Prove that the total number of markers she has is 468.
-/

theorem connie_total_markers
 (g r b y : ℕ) 
 (hg : g = 36) 
 (hr : r = 2 * g)
 (hb : b = 3 * r)
 (hy : y = 4 * g) :
 g + r + b + y = 468 := 
 by
  sorry

end connie_total_markers_l116_116263


namespace largest_whole_x_l116_116916

theorem largest_whole_x (x : ℕ) (h : 11 * x < 150) : x ≤ 13 :=
sorry

end largest_whole_x_l116_116916


namespace final_trip_l116_116639

-- Definitions for the conditions
def dwarf := String
def snow_white := "Snow White" : dwarf
def Happy : dwarf := "Happy"
def Grumpy : dwarf := "Grumpy"
def Dopey : dwarf := "Dopey"
def Bashful : dwarf := "Bashful"
def Sleepy : dwarf := "Sleepy"
def Doc : dwarf := "Doc"
def Sneezy : dwarf := "Sneezy"

-- The dwarfs lineup from left to right
def lineup : List dwarf := [Happy, Grumpy, Dopey, Bashful, Sleepy, Doc, Sneezy]

-- The boat can hold Snow White and up to 3 dwarfs
def boat_capacity (load : List dwarf) : Prop := snow_white ∈ load ∧ load.length ≤ 4

-- Any two dwarfs standing next to each other in the original lineup will quarrel if left without Snow White
def will_quarrel (d1 d2 : dwarf) : Prop :=
  (d1, d2) ∈ (lineup.zip lineup.tail)

-- The objective: Prove that on the last trip, Snow White will take Grumpy, Bashful, and Doc
theorem final_trip : ∃ load : List dwarf, 
  set.load ⊆ {snow_white, Grumpy, Bashful, Doc} ∧ boat_capacity load :=
  sorry

end final_trip_l116_116639


namespace non_zero_number_is_nine_l116_116694

theorem non_zero_number_is_nine (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5 * x) : x = 9 :=
by
  sorry

end non_zero_number_is_nine_l116_116694


namespace angle_D_measure_l116_116319

theorem angle_D_measure (B C E F D : ℝ) 
  (h₁ : B = 120)
  (h₂ : B + C = 180)
  (h₃ : E = 45)
  (h₄ : F = C) 
  (h₅ : D + E + F = 180) :
  D = 75 := sorry

end angle_D_measure_l116_116319


namespace alice_bob_coffee_shop_spending_l116_116924

theorem alice_bob_coffee_shop_spending (A B : ℝ) (h1 : B = 0.5 * A) (h2 : A = B + 15) : A + B = 45 :=
by
  sorry

end alice_bob_coffee_shop_spending_l116_116924


namespace second_markdown_percentage_l116_116107

theorem second_markdown_percentage (P : ℝ) (h1 : P > 0)
    (h2 : ∃ x : ℝ, x = 0.50 * P) -- First markdown
    (h3 : ∃ y : ℝ, y = 0.45 * P) -- Final price
    : ∃ X : ℝ, X = 10 := 
sorry

end second_markdown_percentage_l116_116107


namespace nat_divisibility_l116_116217

theorem nat_divisibility {n : ℕ} : (n + 1 ∣ n^2 + 1) ↔ (n = 0 ∨ n = 1) := 
sorry

end nat_divisibility_l116_116217


namespace total_area_is_8_units_l116_116108

-- Let s be the side length of the original square and x be the leg length of each isosceles right triangle
variables (s x : ℕ)

-- The side length of the smaller square is 8 units
axiom smaller_square_length : s - 2 * x = 8

-- The area of one isosceles right triangle
def area_triangle : ℕ := x * x / 2

-- There are four triangles
def total_area_triangles : ℕ := 4 * area_triangle x

-- The aim is to prove that the total area of the removed triangles is 8 square units
theorem total_area_is_8_units : total_area_triangles x = 8 :=
sorry

end total_area_is_8_units_l116_116108


namespace ratio_Mandy_to_Pamela_l116_116450

-- Definitions based on conditions in the problem
def exam_items : ℕ := 100
def Lowella_correct : ℕ := (35 * exam_items) / 100  -- 35% of 100
def Pamela_correct : ℕ := Lowella_correct + (20 * Lowella_correct) / 100 -- 20% more than Lowella
def Mandy_score : ℕ := 84

-- The proof problem statement
theorem ratio_Mandy_to_Pamela : Mandy_score / Pamela_correct = 2 := by
  sorry

end ratio_Mandy_to_Pamela_l116_116450


namespace complex_div_eq_l116_116825

open Complex

def z := 4 - 2 * I

theorem complex_div_eq :
  (z + I = 4 - I) →
  (z / (4 + 2 * I) = (3 - 4 * I) / 5) :=
by
  sorry

end complex_div_eq_l116_116825


namespace triangle_ABC_right_angle_l116_116135

def point := (ℝ × ℝ)
def line (P: point) := P.1 = 5 ∨ ∃ a: ℝ, P.1 - 5 = a * (P.2 + 2)
def parabola (P: point) := P.2 ^ 2 = 4 * P.1
def perpendicular_slopes (k1 k2: ℝ) := k1 * k2 = -1

theorem triangle_ABC_right_angle (A B C: point) (P: point) 
  (hA: A = (1, 2))
  (hP: P = (5, -2))
  (h_line: line B ∧ line C)
  (h_parabola: parabola B ∧ parabola C):
  (∃ k_AB k_AC: ℝ, perpendicular_slopes k_AB k_AC) →
  ∃k_AB k_AC: ℝ, k_AB * k_AC = -1 :=
by sorry

end triangle_ABC_right_angle_l116_116135


namespace correct_statement_is_D_l116_116870

-- Define each statement as a proposition
def statement_A (a b c : ℕ) : Prop := c ≠ 0 → (a * c = b * c → a = b)
def statement_B : Prop := 30.15 = 30 + 15/60
def statement_C : Prop := ∀ (radius : ℕ), (radius ≠ 0) → (360 * (2 / (2 + 3 + 4)) = 90)
def statement_D : Prop := 9 * 30 + 40/2 = 50

-- Define the theorem to state the correct statement (D)
theorem correct_statement_is_D : statement_D :=
sorry

end correct_statement_is_D_l116_116870


namespace side_length_of_cube_l116_116204

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end side_length_of_cube_l116_116204


namespace conversion_base_10_to_5_l116_116407

theorem conversion_base_10_to_5 : 
  (425 : ℕ) = 3 * 5^3 + 2 * 5^2 + 0 * 5^1 + 0 * 5^0 :=
by sorry

end conversion_base_10_to_5_l116_116407


namespace find_group_2018_l116_116775

-- Definition of the conditions
def group_size (n : Nat) : Nat := 3 * n - 2

def total_numbers (n : Nat) : Nat := 
  (3 * n * n - n) / 2

theorem find_group_2018 : ∃ n : Nat, total_numbers (n - 1) < 1009 ∧ total_numbers n ≥ 1009 ∧ n = 27 :=
  by
  -- This forms the structure for the proof
  sorry

end find_group_2018_l116_116775


namespace angle_y_is_80_l116_116963

def parallel (m n : ℝ) : Prop := sorry

def angle_at_base (θ : ℝ) := θ = 40
def right_angle (θ : ℝ) := θ = 90
def exterior_angle (θ1 θ2 : ℝ) := θ1 + θ2 = 180

theorem angle_y_is_80 (m n : ℝ) (θ1 θ2 θ3 θ_ext : ℝ) :
  parallel m n →
  angle_at_base θ1 →
  right_angle θ2 →
  angle_at_base θ3 →
  exterior_angle θ_ext θ3 →
  θ_ext = 80 := by
  sorry

end angle_y_is_80_l116_116963


namespace set_range_of_three_numbers_l116_116106

theorem set_range_of_three_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : (a + b + c) / 3 = 6) 
(h4 : b = 6) (h5 : c = 10) : c - a = 8 := by
  sorry

end set_range_of_three_numbers_l116_116106


namespace bus_stop_time_l116_116380

theorem bus_stop_time (v_exclude_stop v_include_stop : ℕ) (h1 : v_exclude_stop = 54) (h2 : v_include_stop = 36) : 
  ∃ t: ℕ, t = 20 :=
by
  sorry

end bus_stop_time_l116_116380


namespace cryptarithm_solutions_unique_l116_116836

/- Definitions corresponding to the conditions -/
def is_valid_digit (d : Nat) : Prop := d < 10

def is_six_digit_number (n : Nat) : Prop := n >= 100000 ∧ n < 1000000

def matches_cryptarithm (abcdef bcdefa : Nat) : Prop := abcdef * 3 = bcdefa

/- Prove that the two identified solutions are valid and no other solutions exist -/
theorem cryptarithm_solutions_unique :
  ∀ (A B C D E F : Nat),
  is_valid_digit A → is_valid_digit B → is_valid_digit C →
  is_valid_digit D → is_valid_digit E → is_valid_digit F →
  let abcdef := 100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F
  let bcdefa := 100000 * B + 10000 * C + 1000 * D + 100 * E + 10 * F + A
  is_six_digit_number abcdef →
  is_six_digit_number bcdefa →
  matches_cryptarithm abcdef bcdefa →
  (abcdef = 142857 ∨ abcdef = 285714) :=
by
  intros A B C D E F A_valid B_valid C_valid D_valid E_valid F_valid abcdef bcdefa abcdef_six_digit bcdefa_six_digit cryptarithm_match
  sorry

end cryptarithm_solutions_unique_l116_116836


namespace total_combined_rainfall_l116_116814

def mondayRainfall := 7 * 1
def tuesdayRainfall := 4 * 2
def wednesdayRate := 2 * 2
def wednesdayRainfall := 2 * wednesdayRate
def totalRainfall := mondayRainfall + tuesdayRainfall + wednesdayRainfall

theorem total_combined_rainfall : totalRainfall = 23 :=
by
  unfold totalRainfall mondayRainfall tuesdayRainfall wednesdayRainfall wednesdayRate
  sorry

end total_combined_rainfall_l116_116814


namespace no_transform_to_1998_power_7_l116_116719

theorem no_transform_to_1998_power_7 :
  ∀ n : ℕ, (exists m : ℕ, n = 7^m) ->
  ∀ k : ℕ, n = 10 * k + (n % 10) ->
  ¬ (∃ t : ℕ, (t = (1998 ^ 7))) := 
by sorry

end no_transform_to_1998_power_7_l116_116719


namespace find_sample_size_l116_116526

def sample_size (sample : List ℕ) : ℕ :=
  sample.length

theorem find_sample_size :
  sample_size (List.replicate 500 0) = 500 :=
by
  sorry

end find_sample_size_l116_116526


namespace largest_integer_not_sum_of_multiple_of_30_and_composite_l116_116681

theorem largest_integer_not_sum_of_multiple_of_30_and_composite :
  ∃ n : ℕ, ∀ a b : ℕ, 0 ≤ b ∧ b < 30 ∧ b.prime ∧ (∀ k < a, (b + 30 * k).prime)
    → (30 * a + b = n) ∧
      (∀ m : ℕ, ∀ c d : ℕ, 0 ≤ d ∧ d < 30 ∧ d.prime ∧ (∀ k < c, (d + 30 * k).prime) 
        → (30 * c + d ≤ n)) ∧
      n = 93 :=
by
  sorry

end largest_integer_not_sum_of_multiple_of_30_and_composite_l116_116681


namespace chris_money_left_l116_116259

def video_game_cost : ℕ := 60
def candy_cost : ℕ := 5
def babysitting_rate : ℕ := 8
def hours_worked : ℕ := 9
def earnings : ℕ := babysitting_rate * hours_worked
def total_cost : ℕ := video_game_cost + candy_cost
def money_left : ℕ := earnings - total_cost

theorem chris_money_left
  (h1 : video_game_cost = 60)
  (h2 : candy_cost = 5)
  (h3 : babysitting_rate = 8)
  (h4 : hours_worked = 9) :
  money_left = 7 :=
by
  -- The detailed proof is omitted.
  sorry

end chris_money_left_l116_116259


namespace sum_of_four_digit_integers_up_to_4999_l116_116223

theorem sum_of_four_digit_integers_up_to_4999 : 
  let a := 1000
  let l := 4999
  let n := l - a + 1
  let S := (n / 2) * (a + l)
  S = 11998000 := 
by
  sorry

end sum_of_four_digit_integers_up_to_4999_l116_116223


namespace number_of_pines_l116_116212

theorem number_of_pines (trees : List Nat) :
  (∑ t in trees, 1) = 101 ∧
  (∀ i, ∀ j, trees[i] = trees[j] → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 1))) ∧
  (∀ i, ∀ j, trees[i] = 2 → trees[j] = 2 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 2))) ∧
  (∀ i, ∀ j, trees[i] = 3 → trees[j] = 3 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 3))) →
  (∑ t in trees, if t = 3 then 1 else 0) = 25 ∨ (∑ t in trees, if t = 3 then 1 else 0) = 26 :=
by
  sorry

end number_of_pines_l116_116212


namespace compute_result_l116_116007

noncomputable def compute_expr : ℚ :=
  8 * (2 / 7)^4

theorem compute_result : compute_expr = 128 / 2401 := 
by 
  sorry

end compute_result_l116_116007


namespace max_value_E_X_E_Y_l116_116804

open MeasureTheory

-- Defining the random variables and their ranges
variables {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
variable (X : Ω → ℝ) (Y : Ω → ℝ)

-- Condition: 2 ≤ X ≤ 3
def condition1 : Prop := ∀ ω, 2 ≤ X ω ∧ X ω ≤ 3

-- Condition: XY = 1
def condition2 : Prop := ∀ ω, X ω * Y ω = 1

-- The theorem statement
theorem max_value_E_X_E_Y (h1 : condition1 X) (h2 : condition2 X Y) : 
  ∃ E_X E_Y, (E_X = ∫ ω, X ω ∂μ) ∧ (E_Y = ∫ ω, Y ω ∂μ) ∧ (E_X * E_Y = 25 / 24) := 
sorry

end max_value_E_X_E_Y_l116_116804


namespace joe_spent_on_food_l116_116971

theorem joe_spent_on_food :
  ∀ (initial_savings flight hotel remaining food : ℝ),
    initial_savings = 6000 →
    flight = 1200 →
    hotel = 800 →
    remaining = 1000 →
    food = initial_savings - remaining - (flight + hotel) →
    food = 3000 :=
by
  intros initial_savings flight hotel remaining food h₁ h₂ h₃ h₄ h₅
  sorry

end joe_spent_on_food_l116_116971


namespace find_other_number_l116_116695

def HCF (a b : ℕ) : ℕ := sorry
def LCM (a b : ℕ) : ℕ := sorry

theorem find_other_number (B : ℕ) 
 (h1 : HCF 24 B = 15) 
 (h2 : LCM 24 B = 312) 
 : B = 195 := 
by
  sorry

end find_other_number_l116_116695


namespace side_length_of_cube_l116_116207

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end side_length_of_cube_l116_116207


namespace tony_fever_temperature_above_threshold_l116_116948

theorem tony_fever_temperature_above_threshold 
  (n : ℕ) (i : ℕ) (f : ℕ) 
  (h1 : n = 95) (h2 : i = 10) (h3 : f = 100) : 
  n + i - f = 5 :=
by
  sorry

end tony_fever_temperature_above_threshold_l116_116948


namespace fraction_equiv_l116_116446

theorem fraction_equiv (m n : ℚ) (h : m / n = 3 / 4) : (m + n) / n = 7 / 4 :=
sorry

end fraction_equiv_l116_116446


namespace sum_mod_15_l116_116866

theorem sum_mod_15 
  (d e f : ℕ) 
  (hd : d % 15 = 11)
  (he : e % 15 = 12)
  (hf : f % 15 = 13) : 
  (d + e + f) % 15 = 6 :=
by
  sorry

end sum_mod_15_l116_116866


namespace repeating_decimal_sum_l116_116785

theorem repeating_decimal_sum :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (∀ d, d ∣ a ∧ d ∣ b ↔ d = 1) ∧ 
  (a + b = 134) ∧ (∃ x : ℚ, x = 35 / 99 ∧ (x = 0.353535...)) :=
begin
  sorry
end

end repeating_decimal_sum_l116_116785


namespace max_intersection_l116_116127

open Finset

def n (S : Finset α) : ℕ := (2 : ℕ) ^ S.card

theorem max_intersection (A B C : Finset ℕ)
  (h1 : A.card = 2016)
  (h2 : B.card = 2016)
  (h3 : n A + n B + n C = n (A ∪ B ∪ C)) :
  (A ∩ B ∩ C).card ≤ 2015 :=
sorry

end max_intersection_l116_116127


namespace exists_nat_n_l116_116462

theorem exists_nat_n (l : ℕ) (hl : l > 0) : ∃ n : ℕ, n^n + 47 ≡ 0 [MOD 2^l] := by
  sorry

end exists_nat_n_l116_116462


namespace value_of_g_at_five_l116_116071

def g (x : ℕ) : ℕ := x^2 - 2 * x

theorem value_of_g_at_five : g 5 = 15 := by
  sorry

end value_of_g_at_five_l116_116071


namespace functional_equation_solution_l116_116022

noncomputable def f : ℚ → ℚ := sorry

theorem functional_equation_solution :
  (∀ x y : ℚ, f (f x + x * f y) = x + f x * y) →
  (∀ x : ℚ, f x = x) :=
by
  intro h
  sorry

end functional_equation_solution_l116_116022


namespace max_y_value_l116_116434

theorem max_y_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = (x - y) / (x + 3 * y)) : y ≤ 1 / 3 :=
by
  sorry

end max_y_value_l116_116434


namespace minimize_intercepts_line_eqn_l116_116889

theorem minimize_intercepts_line_eqn (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : (1:ℝ)/a + (1:ℝ)/b = 1)
  (h2 : ∃ a b, a + b = 4 ∧ a = 2 ∧ b = 2) :
  ∀ (x y : ℝ), x + y - 2 = 0 :=
by 
  sorry

end minimize_intercepts_line_eqn_l116_116889


namespace zane_picked_up_62_pounds_l116_116014

variable (daliah : ℝ) (dewei : ℝ) (zane : ℝ)

def daliah_garbage : ℝ := 17.5
def dewei_garbage : ℝ := daliah - 2
def zane_garbage : ℝ := dewei * 4

theorem zane_picked_up_62_pounds (h_daliah : daliah = daliah_garbage) 
                                  (h_dewei : dewei = dewei_garbage) 
                                  (h_zane : zane = zane_garbage) : 
  zane = 62 :=
by 
  sorry

end zane_picked_up_62_pounds_l116_116014


namespace eagle_speed_l116_116111

theorem eagle_speed (E : ℕ) 
  (falcon_speed : ℕ := 46)
  (pelican_speed : ℕ := 33)
  (hummingbird_speed : ℕ := 30)
  (total_distance : ℕ := 248)
  (flight_time : ℕ := 2)
  (falcon_distance := falcon_speed * flight_time)
  (pelican_distance := pelican_speed * flight_time)
  (hummingbird_distance := hummingbird_speed * flight_time) :
  2 * E + falcon_distance + pelican_distance + hummingbird_distance = total_distance →
  E = 15 :=
by
  -- Proof will be provided here
  sorry

end eagle_speed_l116_116111


namespace math_problem_l116_116311

/-- Given a function definition f(x) = 2 * x * f''(1) + x^2,
    Prove that the second derivative f''(0) is equal to -4. -/
theorem math_problem (f : ℝ → ℝ) (h1 : ∀ x, f x = 2 * x * (deriv^[2] (f) 1) + x^2) :
  (deriv^[2] f) 0 = -4 :=
  sorry

end math_problem_l116_116311


namespace equal_intercepts_l116_116142

theorem equal_intercepts (a : ℝ) (h : ∃p, (a * p, 0) = (0, a - 2)) : a = 1 ∨ a = 2 :=
sorry

end equal_intercepts_l116_116142


namespace find_value_l116_116757

theorem find_value (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^2 + b^2 + a * b = 7 :=
by
  sorry

end find_value_l116_116757


namespace initial_alcohol_percentage_l116_116542

theorem initial_alcohol_percentage (P : ℚ) (initial_volume : ℚ) (added_alcohol : ℚ) (added_water : ℚ)
  (final_percentage : ℚ) (final_volume : ℚ) (alcohol_volume_in_initial_solution : ℚ) :
  initial_volume = 40 ∧ 
  added_alcohol = 3.5 ∧ 
  added_water = 6.5 ∧ 
  final_percentage = 0.11 ∧ 
  final_volume = 50 ∧ 
  alcohol_volume_in_initial_solution = (P / 100) * initial_volume ∧ 
  alcohol_volume_in_initial_solution + added_alcohol = final_percentage * final_volume
  → P = 5 :=
by
  sorry

end initial_alcohol_percentage_l116_116542


namespace treasures_coins_count_l116_116500

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l116_116500


namespace ping_pong_ball_probability_l116_116999
open Nat 

def total_balls : ℕ := 70

def multiples_of_4_count : ℕ := 17
def multiples_of_9_count : ℕ := 7
def multiples_of_4_and_9_count : ℕ := 1

def inclusion_exclusion_principle : ℕ :=
  multiples_of_4_count + multiples_of_9_count - multiples_of_4_and_9_count

def desired_outcomes_count : ℕ := inclusion_exclusion_principle

def probability : ℚ := desired_outcomes_count / total_balls

theorem ping_pong_ball_probability : probability = 23 / 70 :=
  sorry

end ping_pong_ball_probability_l116_116999


namespace priya_trip_time_l116_116338

noncomputable def time_to_drive_from_X_to_Z_at_50_mph : ℝ := 5

theorem priya_trip_time :
  (∀ (distance_YZ distance_XZ : ℝ), 
    distance_YZ = 60 * 2.0833333333333335 ∧
    distance_XZ = distance_YZ * 2 →
    time_to_drive_from_X_to_Z_at_50_mph = distance_XZ / 50 ) :=
sorry

end priya_trip_time_l116_116338


namespace volume_of_region_l116_116224

open MeasureTheory Set Filter

noncomputable theory
open_locale classical

def regionVolume : ℝ := 8

theorem volume_of_region :
  let S := {p : ℝ³ | |p.1 + p.2| ≤ 1 ∧ |p.1 + p.2 + p.3 - 2| ≤ 2} in
  volume S = regionVolume :=
sorry

end volume_of_region_l116_116224


namespace power_of_two_square_l116_116097

theorem power_of_two_square (n : ℕ) : ∃ k : ℕ, 2^6 + 2^9 + 2^n = k^2 ↔ n = 10 :=
by
  sorry

end power_of_two_square_l116_116097


namespace subset_0_in_X_l116_116042

-- Define the set X
def X : Set ℤ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Define the theorem to prove
theorem subset_0_in_X : {0} ⊆ X :=
by
  sorry

end subset_0_in_X_l116_116042


namespace geom_progr_sum_eq_l116_116845

variable (a b q : ℝ) (n p : ℕ)

theorem geom_progr_sum_eq (h : a * (1 - q ^ (n * p)) / (1 - q) = b * (1 - q ^ (n * p)) / (1 - q ^ p)) :
  b = a * (1 - q ^ p) / (1 - q) :=
by
  sorry

end geom_progr_sum_eq_l116_116845


namespace find_f_of_given_conditions_l116_116981

def f (ω x : ℝ) (b : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4) + b

theorem find_f_of_given_conditions (ω : ℝ) (b : ℝ)
  (h1 : ω > 0)
  (h2 : 2 < ω ∧ ω < 3)
  (h3 : f ω (3 * Real.pi / 2) b = 2)
  (h4 : b = 2)
  : f ω (Real.pi / 2) b = 1 := by
  sorry

end find_f_of_given_conditions_l116_116981


namespace greatest_integer_b_for_no_real_roots_l116_116416

theorem greatest_integer_b_for_no_real_roots :
  ∃ (b : ℤ), (b * b < 20) ∧ (∀ (c : ℤ), (c * c < 20) → c ≤ 4) :=
by
  sorry

end greatest_integer_b_for_no_real_roots_l116_116416


namespace pickup_carries_10_bags_per_trip_l116_116215

def total_weight : ℕ := 10000
def weight_one_bag : ℕ := 50
def number_of_trips : ℕ := 20
def total_bags : ℕ := total_weight / weight_one_bag
def bags_per_trip : ℕ := total_bags / number_of_trips

theorem pickup_carries_10_bags_per_trip : bags_per_trip = 10 := by
  sorry

end pickup_carries_10_bags_per_trip_l116_116215


namespace loss_percent_l116_116690

theorem loss_percent (CP SP : ℝ) (h_CP : CP = 600) (h_SP : SP = 550) :
  ((CP - SP) / CP) * 100 = 8.33 := by
  sorry

end loss_percent_l116_116690


namespace sufficient_condition_abs_sum_gt_one_l116_116759

theorem sufficient_condition_abs_sum_gt_one (x y : ℝ) (h : y ≤ -2) : |x| + |y| > 1 :=
  sorry

end sufficient_condition_abs_sum_gt_one_l116_116759


namespace angle_sum_at_point_l116_116365

theorem angle_sum_at_point (x : ℝ) (h : 170 + 3 * x = 360) : x = 190 / 3 :=
by
  sorry

end angle_sum_at_point_l116_116365


namespace juanitas_dessert_cost_is_correct_l116_116392

noncomputable def brownie_cost := 2.50
noncomputable def regular_scoop_cost := 1.00
noncomputable def premium_scoop_cost := 1.25
noncomputable def deluxe_scoop_cost := 1.50
noncomputable def syrup_cost := 0.50
noncomputable def nuts_cost := 1.50
noncomputable def whipped_cream_cost := 0.75
noncomputable def cherry_cost := 0.25
noncomputable def discount_tuesday := 0.10

noncomputable def total_cost_of_juanitas_dessert :=
    let discounted_brownie := brownie_cost * (1 - discount_tuesday)
    let ice_cream_cost := 2 * regular_scoop_cost + premium_scoop_cost
    let syrup_total := 2 * syrup_cost
    let additional_toppings := nuts_cost + whipped_cream_cost + cherry_cost
    discounted_brownie + ice_cream_cost + syrup_total + additional_toppings
   
theorem juanitas_dessert_cost_is_correct:
  total_cost_of_juanitas_dessert = 9.00 := by
  sorry

end juanitas_dessert_cost_is_correct_l116_116392


namespace inverse_proportion_l116_116067

theorem inverse_proportion (α β k : ℝ) (h1 : α * β = k) (h2 : α = 5) (h3 : β = 10) : (α = 25 / 2) → (β = 4) := by sorry

end inverse_proportion_l116_116067


namespace arithmetic_sequence_sum_l116_116862

theorem arithmetic_sequence_sum :
  ∀ (x y : ℕ), (∀ (a b c : ℕ), b - a = c - b → c - b = 5) ∧
               (3 + 5 * 1 = 8) ∧
               (8 + 5 * 1 = 13) ∧
               (x + 5 * 1 = y) ∧
               (y + 5 * 1 = 33) →
               x + y = 51 :=
by
  intros x y h
  sorry

end arithmetic_sequence_sum_l116_116862


namespace sum_of_squares_of_roots_l116_116123

theorem sum_of_squares_of_roots (s₁ s₂ : ℝ) (h1 : s₁ + s₂ = 10) (h2 : s₁ * s₂ = 9) : 
  s₁^2 + s₂^2 = 82 := by
  sorry

end sum_of_squares_of_roots_l116_116123


namespace problem_a_problem_b_l116_116875

open Probability

noncomputable def sequenceShortenedByExactlyOneDigitProbability : ℝ :=
  1.564 * 10^(-90)

theorem problem_a (n : ℕ) (p : ℝ) (k : ℕ) (initialLength : ℕ)
  (h_n : n = 2014) (h_p : p = 0.1) (h_k : k = 1) (h_initialLength : initialLength = 2015) :
  (binomialₓ n k p * (1 - p)^(n - k)) = sequenceShortenedByExactlyOneDigitProbability := 
sorry

noncomputable def expectedLengthNewSequence : ℝ :=
  1813.6

theorem problem_b (n : ℕ) (p : ℝ) (initialLength : ℕ) 
  (h_n : n = 2014) (h_p : p = 0.1) (h_initialLength : initialLength = 2015) :
  (initialLength - n * p) = expectedLengthNewSequence :=
sorry

end problem_a_problem_b_l116_116875


namespace find_value_of_xy_l116_116934

-- Define the given conditions and declaration of the proof statement
theorem find_value_of_xy (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h_distinct : x ≠ y) (h_eq : x^2 + 2 / x = y + 2 / y) : x * y = 2 :=
sorry

end find_value_of_xy_l116_116934


namespace average_weight_a_b_l116_116187

theorem average_weight_a_b (A B C : ℝ) 
    (h1 : (A + B + C) / 3 = 45) 
    (h2 : (B + C) / 2 = 44) 
    (h3 : B = 33) : 
    (A + B) / 2 = 40 := 
by 
  sorry

end average_weight_a_b_l116_116187


namespace probability_same_group_l116_116343

noncomputable def num_students : ℕ := 800
noncomputable def num_groups : ℕ := 4
noncomputable def group_size : ℕ := num_students / num_groups
noncomputable def amy := 0
noncomputable def ben := 1
noncomputable def clara := 2

theorem probability_same_group : ∃ p : ℝ, p = 1 / 16 :=
by
  let P_ben_with_amy : ℝ := group_size / num_students
  let P_clara_with_amy : ℝ := group_size / num_students
  let P_all_same := P_ben_with_amy * P_clara_with_amy
  use P_all_same
  sorry

end probability_same_group_l116_116343


namespace triangle_AD_eq_8sqrt2_l116_116163

/-- Given a triangle ABC where AB = 13, AC = 20, and
    D is the foot of the perpendicular from A to BC,
    with the ratio BD : CD = 3 : 4, prove that AD = 8√2. -/
theorem triangle_AD_eq_8sqrt2 
  (AB AC : ℝ) (BD CD AD : ℝ) 
  (h₁ : AB = 13)
  (h₂ : AC = 20)
  (h₃ : BD / CD = 3 / 4)
  (h₄ : BD^2 = AB^2 - AD^2)
  (h₅ : CD^2 = AC^2 - AD^2) :
  AD = 8 * Real.sqrt 2 :=
by
  sorry

end triangle_AD_eq_8sqrt2_l116_116163


namespace prob_neither_A_nor_B_l116_116208

theorem prob_neither_A_nor_B
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ)
  (h1 : P_A = 0.25) (h2 : P_B = 0.30) (h3 : P_A_and_B = 0.15) : 
  1 - (P_A + P_B - P_A_and_B) = 0.60 :=
by
  sorry

end prob_neither_A_nor_B_l116_116208


namespace prime_p_square_condition_l116_116115

theorem prime_p_square_condition (p : ℕ) (h_prime : Prime p) (h_square : ∃ n : ℤ, 5^p + 4 * p^4 = n^2) :
  p = 31 :=
sorry

end prime_p_square_condition_l116_116115


namespace fraction_pos_integer_l116_116149

theorem fraction_pos_integer (p : ℕ) (hp : 0 < p) : (∃ (k : ℕ), k = 1 + (2 * p + 53) / (3 * p - 8)) ↔ p = 3 := 
by
  sorry

end fraction_pos_integer_l116_116149


namespace cube_side_length_l116_116198

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end cube_side_length_l116_116198


namespace pirates_treasure_l116_116525

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l116_116525


namespace rectangle_diagonal_ratio_l116_116404

theorem rectangle_diagonal_ratio (s : ℝ) :
  let d := (Real.sqrt 2) * s
  let D := (Real.sqrt 10) * s
  D / d = Real.sqrt 5 :=
by
  let d := (Real.sqrt 2) * s
  let D := (Real.sqrt 10) * s
  sorry

end rectangle_diagonal_ratio_l116_116404


namespace bridget_bought_17_apples_l116_116112

noncomputable def total_apples (x : ℕ) : Prop :=
  (2 * x / 3) - 5 = 6

theorem bridget_bought_17_apples : ∃ x : ℕ, total_apples x ∧ x = 17 :=
  sorry

end bridget_bought_17_apples_l116_116112


namespace find_c_l116_116126

theorem find_c (x c : ℝ) (h : ((5 * x + 38 + c) / 5) = (x + 4) + 5) : c = 7 :=
by
  sorry

end find_c_l116_116126


namespace minimum_value_inequality_l116_116986

theorem minimum_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (Real.sqrt ((x^2 + 4 * y^2) * (2 * x^2 + 3 * y^2)) / (x * y)) ≥ 2 * Real.sqrt (2 * Real.sqrt 6) :=
sorry

end minimum_value_inequality_l116_116986


namespace base_number_eq_2_l116_116954

theorem base_number_eq_2 (x : ℝ) (n : ℕ) (h₁ : x^(2 * n) + x^(2 * n) + x^(2 * n) + x^(2 * n) = 4^28) (h₂ : n = 27) : x = 2 := by
  sorry

end base_number_eq_2_l116_116954


namespace old_conveyor_time_l116_116723

theorem old_conveyor_time (x : ℝ) : 
  (1 / x) + (1 / 15) = 1 / 8.75 → 
  x = 21 := 
by 
  intro h 
  sorry

end old_conveyor_time_l116_116723


namespace wade_customers_sunday_l116_116857

theorem wade_customers_sunday :
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let total_tips := 296
  let tips_friday := customers_friday * tips_per_customer
  let tips_saturday := customers_saturday * tips_per_customer
  let tips_fri_sat := tips_friday + tips_saturday
  let tips_sunday := total_tips - tips_fri_sat
  let customers_sunday := tips_sunday / tips_per_customer
  customers_sunday = 36 :=
by
  let tips_per_customer := 2
  let customers_friday := 28
  let customers_saturday := 3 * customers_friday
  let total_tips := 296
  let tips_friday := customers_friday * tips_per_customer
  let tips_saturday := customers_saturday * tips_per_customer
  let tips_fri_sat := tips_friday + tips_saturday
  let tips_sunday := total_tips - tips_fri_sat
  let customers_sunday := tips_sunday / tips_per_customer
  have h : customers_sunday = 36 := by sorry
  exact h

end wade_customers_sunday_l116_116857


namespace expected_girls_left_of_boys_l116_116368

theorem expected_girls_left_of_boys : 
  (∑ i in (finset.range 7), ((i+1) : ℝ) / 17) = 7 / 11 :=
sorry

end expected_girls_left_of_boys_l116_116368


namespace cistern_capacity_l116_116236

theorem cistern_capacity (C : ℝ) (h1 : C / 20 > 0) (h2 : C / 24 > 0) (h3 : 4 - C / 20 = C / 24) : C = 480 / 11 :=
by sorry

end cistern_capacity_l116_116236


namespace election_votes_l116_116809

theorem election_votes (P : ℕ) (M : ℕ) (V : ℕ) (hP : P = 60) (hM : M = 1300) :
  V = 6500 :=
by
  sorry

end election_votes_l116_116809


namespace min_value_one_over_a_plus_two_over_b_l116_116214

theorem min_value_one_over_a_plus_two_over_b :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 2) →
  ∃ (min_val : ℝ), min_val = (1 / a + 2 / b) ∧ min_val = 9 / 2 :=
by
  sorry

end min_value_one_over_a_plus_two_over_b_l116_116214


namespace find_number_l116_116592

theorem find_number (n : ℕ) (some_number : ℕ) 
  (h : (1/5 : ℝ)^n * (1/4 : ℝ)^(18 : ℕ) = 1 / (2 * (some_number : ℝ)^n))
  (hn : n = 35) : some_number = 10 := 
by 
  sorry

end find_number_l116_116592


namespace total_cakes_served_l116_116712

-- Conditions
def cakes_lunch : Nat := 6
def cakes_dinner : Nat := 9

-- Statement of the problem
theorem total_cakes_served : cakes_lunch + cakes_dinner = 15 := 
by
  sorry

end total_cakes_served_l116_116712


namespace quadratic_no_real_solutions_l116_116059

theorem quadratic_no_real_solutions (a : ℝ) (h₀ : 0 < a) (h₁ : a^3 = 6 * (a + 1)) : 
  ∀ x : ℝ, ¬ (x^2 + a * x + a^2 - 6 = 0) :=
by
  sorry

end quadratic_no_real_solutions_l116_116059


namespace pirates_treasure_l116_116493

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l116_116493


namespace quadratic_equation_correct_l116_116534

theorem quadratic_equation_correct :
    (∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^2 = 5)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x y : ℝ, x + 2 * y = 0)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^2 + 1/x = 0)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^3 + x^2 = 0)) :=
by
  sorry

end quadratic_equation_correct_l116_116534


namespace P_at_20_l116_116283

-- Define the polynomial structure and given conditions
noncomputable def P (x : ℝ) : ℝ := x^2 + (a : ℝ) * x + (b : ℝ)

-- The conditions as given in the problem
axiom condition1 : P(10) = 10^2 + 10 * a + b
axiom condition2 : P(30) = 30^2 + 30 * a + b
axiom condition3 : (P(10) + P(30)) = 40

-- Prove that P(20) = -80 given the conditions
theorem P_at_20 : ∃ (a b : ℝ), P (20) = -80 :=
by
  sorry

end P_at_20_l116_116283


namespace ab_product_l116_116817

theorem ab_product (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a + b) * (2 * b + a) = 4752) : a * b = 520 := 
by
  sorry

end ab_product_l116_116817


namespace tan_225_eq_1_l116_116262

theorem tan_225_eq_1 : Real.tan (225 * Real.pi / 180) = 1 := by
  -- Let's denote the point P on the unit circle for 225 degrees as given
  have P_coords : (Real.cos (225 * Real.pi / 180), Real.sin (225 * Real.pi / 180)) = (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2) := sorry,
  -- Compute the tangent using the coordinates of P
  rw [Real.tan_eq_sin_div_cos],
  rw [P_coords],
  simp,
  sorry

end tan_225_eq_1_l116_116262


namespace algebra_inequality_l116_116596

theorem algebra_inequality (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x ^ 2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end algebra_inequality_l116_116596


namespace prism_surface_area_is_14_l116_116666

-- Definition of the rectangular prism dimensions
def prism_length : ℕ := 3
def prism_width : ℕ := 1
def prism_height : ℕ := 1

-- Definition of the surface area of the rectangular prism
def surface_area (l w h : ℕ) : ℕ :=
  2 * (l * w + w * h + h * l)

-- Theorem statement: The surface area of the resulting prism is 14
theorem prism_surface_area_is_14 : surface_area prism_length prism_width prism_height = 14 :=
  sorry

end prism_surface_area_is_14_l116_116666


namespace compound_statement_false_l116_116935

theorem compound_statement_false (p q : Prop) (h : ¬ (p ∧ q)) : ¬ p ∨ ¬ q :=
sorry

end compound_statement_false_l116_116935


namespace teammates_score_l116_116318

def Lizzie_score := 4
def Nathalie_score := Lizzie_score + 3
def combined_Lizzie_Nathalie := Lizzie_score + Nathalie_score
def Aimee_score := 2 * combined_Lizzie_Nathalie
def total_team_score := 50
def total_combined_score := Lizzie_score + Nathalie_score + Aimee_score

theorem teammates_score : total_team_score - total_combined_score = 17 :=
by
  sorry

end teammates_score_l116_116318


namespace monthly_salary_l116_116232

variables (S : ℕ) (h1 : S * 20 / 100 * 96 / 100 = 4 * 250)

theorem monthly_salary : S = 6250 :=
by sorry

end monthly_salary_l116_116232


namespace find_a_l116_116458

theorem find_a (a b c : ℂ) (ha : a.im = 0)
  (h1 : a + b + c = 5)
  (h2 : a * b + b * c + c * a = 8)
  (h3 : a * b * c = 4) :
  a = 1 ∨ a = 2 :=
sorry

end find_a_l116_116458


namespace repeating_decimal_35_as_fraction_l116_116787

noncomputable def repeating_decimal_as_fraction (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ a b : ℕ, (0.\overline{35}).repr = n / d ∧ Int.gcd a b = 1 ∧ a + b = 134

theorem repeating_decimal_35_as_fraction :
  repeating_decimal_as_fraction 35 99 := sorry

end repeating_decimal_35_as_fraction_l116_116787


namespace isosceles_right_triangle_hypotenuse_l116_116810

noncomputable def hypotenuse_length : ℝ :=
  let a := Real.sqrt 363
  let c := Real.sqrt (2 * (a ^ 2))
  c

theorem isosceles_right_triangle_hypotenuse :
  ∀ (a : ℝ),
    (2 * (a ^ 2)) + (a ^ 2) = 1452 →
    hypotenuse_length = Real.sqrt 726 := by
  intro a h
  rw [hypotenuse_length]
  sorry

end isosceles_right_triangle_hypotenuse_l116_116810


namespace cost_of_one_box_of_paper_clips_l116_116736

theorem cost_of_one_box_of_paper_clips (p i : ℝ) 
  (h1 : 15 * p + 7 * i = 55.40) 
  (h2 : 12 * p + 10 * i = 61.70) : 
  p = 1.835 := 
by 
  sorry

end cost_of_one_box_of_paper_clips_l116_116736


namespace transfer_equation_correct_l116_116527

theorem transfer_equation_correct (x : ℕ) :
  46 + x = 3 * (30 - x) := 
sorry

end transfer_equation_correct_l116_116527


namespace no_zonk_probability_l116_116336

theorem no_zonk_probability (Z C G : ℕ) (total_boxes : ℕ := 3) (tables : ℕ := 3)
  (no_zonk_prob : ℚ := 2 / 3) : (no_zonk_prob ^ tables) = 8 / 27 :=
by
  -- Here we would prove the theorem, but for the purpose of this task, we skip the proof.
  sorry

end no_zonk_probability_l116_116336


namespace find_c_l116_116626

def p (x : ℝ) : ℝ := 3 * x - 8
def q (x : ℝ) (c : ℝ) : ℝ := 5 * x - c

theorem find_c (c : ℝ) (h : p (q 3 c) = 14) : c = 23 / 3 :=
by
  sorry

end find_c_l116_116626


namespace inequality_abc_l116_116331

theorem inequality_abc (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a^2 + b^2 = 1/2) :
  (1 / (1 - a) + 1 / (1 - b) >= 4)
  ∧ ((1 / (1 - a) + 1 / (1 - b) = 4) ↔ (a = 1/2 ∧ b = 1/2)) :=
by
  sorry

end inequality_abc_l116_116331


namespace females_watch_eq_seventy_five_l116_116715

-- Definition of conditions
def males_watch : ℕ := 85
def females_dont_watch : ℕ := 120
def total_watch : ℕ := 160
def total_dont_watch : ℕ := 180

-- Definition of the proof problem
theorem females_watch_eq_seventy_five :
  total_watch - males_watch = 75 :=
by
  sorry

end females_watch_eq_seventy_five_l116_116715


namespace part1_part2_l116_116932

-- Definition of sets A, B, and Proposition p for Part 1
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a = 0}
def p (a : ℝ) : Prop := ∀ x ∈ B a, x ∈ A

-- Part 1: Prove the range of a
theorem part1 (a : ℝ) : (p a) → 0 < a ∧ a ≤ 1 :=
  by sorry

-- Definition of sets A and C for Part 2
def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 3 > 0}
def necessary_condition (m : ℝ) : Prop := ∀ x ∈ A, x ∈ C m

-- Part 2: Prove the range of m
theorem part2 (m : ℝ) : necessary_condition m → m ≤ 7 / 2 :=
  by sorry

end part1_part2_l116_116932


namespace find_f_at_one_l116_116029

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + 2 * x - 8

theorem find_f_at_one (h_cond : f a b (-1) = 10) : f a b (1) = 14 := by
  sorry

end find_f_at_one_l116_116029


namespace coprime_repeating_decimal_sum_l116_116791

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l116_116791


namespace probability_of_exactly_one_shortening_l116_116880

-- Define the conditions and problem
def shorten_sequence_exactly_one_probability (n : ℕ) (p : ℝ) : ℝ :=
  nat.choose n 1 * p^1 * (1 - p)^(n - 1)

-- Given conditions
def problem_8a := shorten_sequence_exactly_one_probability 2014 0.1 ≈ 1.564e-90

-- Lean statement
theorem probability_of_exactly_one_shortening :
  problem_8a := sorry

end probability_of_exactly_one_shortening_l116_116880


namespace recording_time_is_one_hour_l116_116328

-- Define the recording interval and number of instances
def recording_interval : ℕ := 5 -- The device records data every 5 seconds
def number_of_instances : ℕ := 720 -- The device recorded 720 instances of data

-- Prove that the total recording time is 1 hour
theorem recording_time_is_one_hour : (recording_interval * number_of_instances) / 3600 = 1 := by
  sorry

end recording_time_is_one_hour_l116_116328


namespace range_of_m_l116_116442

noncomputable def f (x m : ℝ) := x^2 - 2 * m * x + 4

def P (m : ℝ) : Prop := ∀ x, 2 ≤ x → f x m ≥ f (2 : ℝ) m
def Q (m : ℝ) : Prop := ∀ x, 4 * x^2 + 4 * (m - 2) * x + 1 > 0

theorem range_of_m (m : ℝ) : (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m ≤ 1 ∨ (2 < m ∧ m < 3) := sorry

end range_of_m_l116_116442


namespace general_formula_sum_first_n_terms_l116_116287

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

axiom a_initial : a 1 = 1
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n * (1 + 1 / n)

theorem general_formula : ∀ n : ℕ, n > 0 → a n = n * 3^(n - 1) :=
by
  sorry

theorem sum_first_n_terms : ∀ n : ℕ, S n = (2 * n - 1) * 3^n + 1 / 4 :=
by
  sorry

end general_formula_sum_first_n_terms_l116_116287


namespace will_buy_toys_l116_116539

theorem will_buy_toys : 
  ∀ (initialMoney spentMoney toyCost : ℕ), 
  initialMoney = 83 → spentMoney = 47 → toyCost = 4 → 
  (initialMoney - spentMoney) / toyCost = 9 :=
by
  intros initialMoney spentMoney toyCost hInit hSpent hCost
  sorry

end will_buy_toys_l116_116539


namespace find_number_l116_116593

theorem find_number (X : ℝ) (h : 50 = 0.20 * X + 47) : X = 15 :=
sorry

end find_number_l116_116593


namespace shop_owner_cheat_percentage_l116_116713

def CP : ℝ := 100
def cheating_buying : ℝ := 0.15  -- 15% cheating
def actual_cost_price : ℝ := CP * (1 + cheating_buying)  -- $115
def profit_percentage : ℝ := 43.75

theorem shop_owner_cheat_percentage :
  ∃ x : ℝ, profit_percentage = ((CP - x * CP / 100 - actual_cost_price) / actual_cost_price * 100) ∧ x = 65.26 :=
by
  sorry

end shop_owner_cheat_percentage_l116_116713


namespace f_at_neg_one_l116_116923

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + 3 * x + 16

noncomputable def f_with_r (x : ℝ) (a r : ℝ) : ℝ := (x^3 + a * x^2 + 3 * x + 16) * (x - r)

theorem f_at_neg_one (a b c r : ℝ) (h1 : ∀ x, g x a = 0 → f_with_r x a r = 0)
  (h2 : a - r = 5) (h3 : 16 - 3 * r = 150) (h4 : -16 * r = c) :
  f_with_r (-1) a r = -1347 :=
by
  sorry

end f_at_neg_one_l116_116923


namespace number_of_albums_l116_116086

-- Definitions for the given conditions
def pictures_from_phone : ℕ := 7
def pictures_from_camera : ℕ := 13
def pictures_per_album : ℕ := 4

-- We compute the total number of pictures
def total_pictures : ℕ := pictures_from_phone + pictures_from_camera

-- Statement: Prove the number of albums is 5
theorem number_of_albums :
  total_pictures / pictures_per_album = 5 := by
  sorry

end number_of_albums_l116_116086


namespace Laran_large_posters_daily_l116_116057

/-
Problem statement:
Laran has started a poster business. She is selling 5 posters per day at school. Some posters per day are her large posters that sell for $10. The large posters cost her $5 to make. The remaining posters are small posters that sell for $6. They cost $3 to produce. Laran makes a profit of $95 per 5-day school week. How many large posters does Laran sell per day?
-/

/-
Mathematically equivalent proof problem:
Prove that the number of large posters Laran sells per day is 2, given the following conditions:
1) L + S = 5
2) 5L + 3S = 19
-/

variables (L S : ℕ)

-- Given conditions
def condition1 := L + S = 5
def condition2 := 5 * L + 3 * S = 19

-- Prove the desired statement
theorem Laran_large_posters_daily 
    (h1 : condition1 L S) 
    (h2 : condition2 L S) : 
    L = 2 := 
sorry

end Laran_large_posters_daily_l116_116057


namespace q1_q2_q3_l116_116382

-- (1) Given |a| = 3, |b| = 1, and a < b, prove a + b = -2 or -4.
theorem q1 (a b : ℚ) (h1 : |a| = 3) (h2 : |b| = 1) (h3 : a < b) : a + b = -2 ∨ a + b = -4 := sorry

-- (2) Given rational numbers a and b such that ab ≠ 0, prove the value of (a/|a|) + (b/|b|) is 2, -2, or 0.
theorem q2 (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) : (a / |a|) + (b / |b|) = 2 ∨ (a / |a|) + (b / |b|) = -2 ∨ (a / |a|) + (b / |b|) = 0 := sorry

-- (3) Given rational numbers a, b, c such that a + b + c = 0 and abc < 0, prove the value of (b+c)/|a| + (a+c)/|b| + (a+b)/|c| is -1.
theorem q3 (a b c : ℚ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : (b + c) / |a| + (a + c) / |b| + (a + b) / |c| = -1 := sorry

end q1_q2_q3_l116_116382


namespace find_total_coins_l116_116518

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l116_116518


namespace repeating_decimal_35_as_fraction_l116_116786

noncomputable def repeating_decimal_as_fraction (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ a b : ℕ, (0.\overline{35}).repr = n / d ∧ Int.gcd a b = 1 ∧ a + b = 134

theorem repeating_decimal_35_as_fraction :
  repeating_decimal_as_fraction 35 99 := sorry

end repeating_decimal_35_as_fraction_l116_116786


namespace income_of_A_l116_116077

theorem income_of_A (x y : ℝ) (hx₁ : 5 * x - 3 * y = 1600) (hx₂ : 4 * x - 2 * y = 1600) : 
  5 * x = 4000 :=
by
  sorry

end income_of_A_l116_116077


namespace smallest_integer_proof_l116_116684

def smallest_integer_with_gcd_18_6 : Nat :=
  let n := 114
  if n > 100 ∧  (Nat.gcd n 18) = 6 then n else 0

theorem smallest_integer_proof : smallest_integer_with_gcd_18_6 = 114 := 
  by
    unfold smallest_integer_with_gcd_18_6
    have h₁ : 114 > 100 := by decide
    have h₂ : Nat.gcd 114 18 = 6 := by decide
    simp [h₁, h₂]
    sorry

end smallest_integer_proof_l116_116684


namespace pure_imaginary_sol_l116_116316

theorem pure_imaginary_sol (m : ℝ) (h : (m^2 - m - 2) = 0 ∧ (m + 1) ≠ 0) : m = 2 :=
sorry

end pure_imaginary_sol_l116_116316


namespace O_is_incenter_l116_116131

variable {n : ℕ}
variable (A : Fin n → ℝ × ℝ)
variable (O : ℝ × ℝ)

-- Conditions
def inside_convex_ngon (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry
def angles_acute (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry
def angles_inequality (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry

-- This is the statement that we need to prove.
theorem O_is_incenter 
  (h1 : inside_convex_ngon O A)
  (h2 : angles_acute O A) 
  (h3 : angles_inequality O A) 
: sorry := sorry

end O_is_incenter_l116_116131


namespace solve_for_x_l116_116147

variable (a b c x y z : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem solve_for_x (h1 : (x * y) / (x + y) = a)
                   (h2 : (x * z) / (x + z) = b)
                   (h3 : (y * z) / (y + z) = c) :
                   x = (2 * a * b * c) / (a * c + b * c - a * b) :=
by 
  sorry

end solve_for_x_l116_116147


namespace tickets_spent_dunk_a_clown_booth_l116_116855

/-
The conditions given:
1. Tom bought 40 tickets.
2. Tom went on 3 rides.
3. Each ride costs 4 tickets.
-/
def total_tickets : ℕ := 40
def rides_count : ℕ := 3
def tickets_per_ride : ℕ := 4

/-
We aim to prove that Tom spent 28 tickets at the 'dunk a clown' booth.
-/
theorem tickets_spent_dunk_a_clown_booth :
  (total_tickets - rides_count * tickets_per_ride) = 28 :=
by
  sorry

end tickets_spent_dunk_a_clown_booth_l116_116855


namespace necessary_but_not_sufficient_l116_116033

noncomputable def isEllipseWithFociX (a b : ℝ) : Prop :=
  ∃ (C : ℝ → ℝ → Prop), (∀ (x y : ℝ), C x y ↔ (x^2 / a + y^2 / b = 1)) ∧ (a > b ∧ a > 0 ∧ b > 0)

theorem necessary_but_not_sufficient (a b : ℝ) :
  (∀ (C : ℝ → ℝ → Prop), (∀ x y : ℝ, C x y ↔ (x^2 / a + y^2 / b = 1))
    → ((a > b ∧ a > 0 ∧ b > 0) → isEllipseWithFociX a b))
  ∧ ¬ (a > b → ∀ (C : ℝ → ℝ → Prop), (∀ x y : ℝ, C x y ↔ (x^2 / a + y^2 / b = 1)) → isEllipseWithFociX a b) :=
sorry

end necessary_but_not_sufficient_l116_116033


namespace existence_of_solution_values_continuous_solution_value_l116_116616

noncomputable def functional_equation_has_solution (a : ℝ) (f : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧ f 1 = 1 ∧ ∀ x y, (x ≤ y → f ((x + y) / 2) = (1 - a) * f x + a * f y)

theorem existence_of_solution_values :
  {a : ℝ | ∃ f : ℝ → ℝ, functional_equation_has_solution a f} = {0, 1/2, 1} :=
sorry

theorem continuous_solution_value :
  {a : ℝ | ∃ (f : ℝ → ℝ) (hf : Continuous f), functional_equation_has_solution a f} = {1/2} :=
sorry

end existence_of_solution_values_continuous_solution_value_l116_116616


namespace triangle_XDE_area_l116_116608

theorem triangle_XDE_area 
  (XY YZ XZ : ℝ) (hXY : XY = 8) (hYZ : YZ = 12) (hXZ : XZ = 14)
  (D E : ℝ → ℝ) (XD XE : ℝ) (hXD : XD = 3) (hXE : XE = 9) :
  ∃ (A : ℝ), A = 1/2 * XD * XE * (15 * Real.sqrt 17 / 56) ∧ A = 405 * Real.sqrt 17 / 112 :=
  sorry

end triangle_XDE_area_l116_116608


namespace siblings_pizza_order_l116_116754

theorem siblings_pizza_order :
  let Alex := 1 / 6
  let Beth := 2 / 5
  let Cyril := 1 / 3
  let Dan := 1 - (Alex + Beth + Cyril)
  Dan > Alex ∧ Alex > Cyril ∧ Cyril > Beth := sorry

end siblings_pizza_order_l116_116754


namespace coincide_green_square_pairs_l116_116912

structure Figure :=
  (green_squares : ℕ)
  (red_triangles : ℕ)
  (blue_triangles : ℕ)

theorem coincide_green_square_pairs (f : Figure) (hs : f.green_squares = 4)
  (rt : f.red_triangles = 3) (bt : f.blue_triangles = 6)
  (gs_coincide : ∀ n, n ≤ f.green_squares ⟶ n = f.green_squares) 
  (rt_coincide : ∃ n, n = 2) (bt_coincide : ∃ n, n = 2) 
  (red_blue_pairs : ∃ n, n = 3) : 
  ∃ pairs, pairs = 4 :=
by 
  sorry

end coincide_green_square_pairs_l116_116912


namespace value_by_which_number_is_multiplied_l116_116089

theorem value_by_which_number_is_multiplied (x : ℝ) : (5 / 6) * x = 10 ↔ x = 12 := by
  sorry

end value_by_which_number_is_multiplied_l116_116089


namespace beautiful_39th_moment_l116_116047

def is_beautiful (h : ℕ) (mm : ℕ) : Prop :=
  (h + mm) % 12 = 0

def start_time := (7, 49)

noncomputable def find_39th_beautiful_moment : ℕ × ℕ :=
  (15, 45)

theorem beautiful_39th_moment :
  find_39th_beautiful_moment = (15, 45) :=
by
  sorry

end beautiful_39th_moment_l116_116047


namespace max_tries_needed_to_open_lock_l116_116703

-- Definitions and conditions
def num_buttons : ℕ := 9
def sequence_length : ℕ := 4
def opposite_trigrams : ℕ := 2  -- assumption based on the problem's example
def total_combinations : ℕ := 3024

theorem max_tries_needed_to_open_lock :
  (total_combinations - (8 * 1 * 7 * 6 + 8 * 6 * 1 * 6 + 8 * 6 * 4 * 1)) = 2208 :=
by
  sorry

end max_tries_needed_to_open_lock_l116_116703


namespace max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_l116_116953

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem max_ab_bc_ca : ab + bc + ca ≤ 1 / 3 :=
by sorry

theorem a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2 :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ 1 / 2 :=
by sorry

end max_ab_bc_ca_a2_div_bc_plus_b2_div_ca_plus_c2_div_ab_geq_1_over_2_l116_116953


namespace smallest_term_of_sequence_l116_116345

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n^2 - 28 * n

-- The statement that the 5th term is the smallest in the sequence
theorem smallest_term_of_sequence : ∀ n : ℕ, a 5 ≤ a n := by
  sorry

end smallest_term_of_sequence_l116_116345


namespace find_f_pi_over_2_l116_116976

noncomputable def f (x : ℝ) (ω : ℝ) (b : ℝ) : ℝ := Real.sin (ω * x + π / 4) + b

theorem find_f_pi_over_2 (ω : ℝ) (b : ℝ) (T : ℝ) :
  (ω > 0) →
  (f.period ℝ (λ x, f x ω b) T) →
  ((2 * π / 3 < T) ∧ (T < π)) →
  ((f (3 * π / 2) ω b = 2) ∧ 
    (f (3 * π / 2) ω b = f (3 * π / 2 - T) ω b) ∧
    (f (3 * π / 2) ω b = f (3 * π / 2 + T) ω b)) →
  f (π / 2) ω b = 1 :=
by
  sorry

end find_f_pi_over_2_l116_116976


namespace sequence_sum_identity_l116_116587

theorem sequence_sum_identity 
  (a_n b_n : ℕ → ℕ) 
  (S_n T_n : ℕ → ℕ)
  (h1 : ∀ n, b_n n - a_n n = 2^n + 1)
  (h2 : ∀ n, S_n n + T_n n = 2^(n+1) + n^2 - 2) : 
  ∀ n, 2 * T_n n = n * (n - 1) :=
by sorry

end sequence_sum_identity_l116_116587


namespace correct_statement_is_D_l116_116228

axiom three_points_determine_plane : Prop
axiom line_and_point_determine_plane : Prop
axiom quadrilateral_is_planar_figure : Prop
axiom two_intersecting_lines_determine_plane : Prop

theorem correct_statement_is_D : two_intersecting_lines_determine_plane = True := 
by sorry

end correct_statement_is_D_l116_116228


namespace find_total_coins_l116_116517

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l116_116517


namespace unique_factor_and_multiple_of_13_l116_116707

theorem unique_factor_and_multiple_of_13 (n : ℕ) (h1 : n ∣ 13) (h2 : 13 ∣ n) : n = 13 :=
sorry

end unique_factor_and_multiple_of_13_l116_116707


namespace postage_unformable_l116_116749

theorem postage_unformable (n : ℕ) (h₁ : n > 0) (h₂ : 110 = 7 * n - 7 - n) :
  n = 19 := 
sorry

end postage_unformable_l116_116749


namespace eliza_total_clothes_l116_116737

def time_per_blouse : ℕ := 15
def time_per_dress : ℕ := 20
def blouse_time : ℕ := 2 * 60   -- 2 hours in minutes
def dress_time : ℕ := 3 * 60    -- 3 hours in minutes

theorem eliza_total_clothes :
  (blouse_time / time_per_blouse) + (dress_time / time_per_dress) = 17 :=
by
  sorry

end eliza_total_clothes_l116_116737


namespace sequence_initial_term_l116_116429

theorem sequence_initial_term (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n + 1) = a n + n)
  (h2 : a 61 = 2010) : a 1 = 180 :=
by
  sorry

end sequence_initial_term_l116_116429


namespace pirates_treasure_l116_116492

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l116_116492


namespace point_symmetric_about_y_axis_l116_116763

theorem point_symmetric_about_y_axis (A B : ℝ × ℝ) 
  (hA : A = (1, -2)) 
  (hSym : B = (-A.1, A.2)) :
  B = (-1, -2) := 
by 
  sorry

end point_symmetric_about_y_axis_l116_116763


namespace total_candles_used_l116_116184

def cakes_baked : ℕ := 8
def cakes_given_away : ℕ := 2
def remaining_cakes : ℕ := cakes_baked - cakes_given_away
def candles_per_cake : ℕ := 6

theorem total_candles_used : remaining_cakes * candles_per_cake = 36 :=
by
  -- proof omitted
  sorry

end total_candles_used_l116_116184


namespace min_value_expression_l116_116823

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  8 * x^3 + 27 * y^3 + 64 * z^3 + (1 / (8 * x * y * z)) ≥ 4 :=
by
  sorry

end min_value_expression_l116_116823


namespace prisha_other_number_l116_116178

def prisha_numbers (a b : ℤ) : Prop :=
  3 * a + 2 * b = 105 ∧ (a = 15 ∨ b = 15)

theorem prisha_other_number (a b : ℤ) (h : prisha_numbers a b) : b = 30 :=
sorry

end prisha_other_number_l116_116178


namespace slopes_of_line_intersecting_ellipse_l116_116101

theorem slopes_of_line_intersecting_ellipse (m : ℝ) : 
  (m ∈ Set.Iic (-1 / Real.sqrt 624) ∨ m ∈ Set.Ici (1 / Real.sqrt 624)) ↔
  ∃ x y, y = m * x + 10 ∧ 4 * x^2 + 25 * y^2 = 100 :=
by
  sorry

end slopes_of_line_intersecting_ellipse_l116_116101


namespace max_value_4287_5_l116_116456

noncomputable def maximum_value_of_expression (x y : ℝ) := x * y * (105 - 2 * x - 5 * y)

theorem max_value_4287_5 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y < 105) :
  maximum_value_of_expression x y ≤ 4287.5 :=
sorry

end max_value_4287_5_l116_116456


namespace trapezoid_prob_l116_116359

noncomputable def trapezoid_probability_not_below_x_axis : ℝ :=
  let P := (4, 4)
  let Q := (-4, -4)
  let R := (-10, -4)
  let S := (-2, 4)
  -- Coordinates of intersection points
  let T := (0, 0)
  let U := (-6, 0)
  -- Compute the probability
  (16 * Real.sqrt 2 + 16) / (16 * Real.sqrt 2 + 40)

theorem trapezoid_prob :
  trapezoid_probability_not_below_x_axis = (16 * Real.sqrt 2 + 16) / (16 * Real.sqrt 2 + 40) :=
sorry

end trapezoid_prob_l116_116359


namespace minimum_apples_l116_116852

theorem minimum_apples (x : ℕ) : 
  (x ≡ 10 [MOD 3]) ∧ (x ≡ 11 [MOD 4]) ∧ (x ≡ 12 [MOD 5]) → x = 67 :=
sorry

end minimum_apples_l116_116852


namespace first_sculpture_weight_is_five_l116_116611

variable (w x y z : ℝ)

def hourly_wage_exterminator := 70
def daily_hours := 20
def price_per_pound := 20
def second_sculpture_weight := 7
def total_income := 1640

def income_exterminator := daily_hours * hourly_wage_exterminator
def income_sculptures := total_income - income_exterminator
def income_second_sculpture := second_sculpture_weight * price_per_pound
def income_first_sculpture := income_sculptures - income_second_sculpture

def weight_first_sculpture := income_first_sculpture / price_per_pound

theorem first_sculpture_weight_is_five :
  weight_first_sculpture = 5 := sorry

end first_sculpture_weight_is_five_l116_116611


namespace solve_for_x_l116_116341

theorem solve_for_x (x : ℝ) : 45 - 5 = 3 * x + 10 → x = 10 :=
by
  sorry

end solve_for_x_l116_116341


namespace evaluate_expression_at_one_l116_116637

theorem evaluate_expression_at_one : 
  (4 + (4 + x^2) / x) / ((x + 2) / x) = 3 := by
  sorry

end evaluate_expression_at_one_l116_116637


namespace sequence_value_l116_116431

theorem sequence_value (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → a n * a (n + 2) = a (n + 1) ^ 2)
  (h2 : a 7 = 16)
  (h3 : a 3 * a 5 = 4) : 
  a 3 = 1 := 
sorry

end sequence_value_l116_116431


namespace cuboid_height_l116_116746

-- Definition of variables
def length := 4  -- in cm
def breadth := 6  -- in cm
def surface_area := 120  -- in cm²

-- The formula for the surface area of a cuboid: S = 2(lb + lh + bh)
def surface_area_formula (l b h : ℝ) : ℝ := 2 * (l * b + l * h + b * h)

-- Given these values, we need to prove that the height h is 3.6 cm
theorem cuboid_height : 
  ∃ h : ℝ, surface_area = surface_area_formula length breadth h ∧ h = 3.6 :=
by
  sorry

end cuboid_height_l116_116746


namespace find_number_l116_116098

theorem find_number (x : ℝ) (h : 0.50 * x = 48 + 180) : x = 456 :=
sorry

end find_number_l116_116098


namespace sum_of_fraction_components_l116_116797

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l116_116797


namespace pieces_per_block_is_32_l116_116890

-- Define the number of pieces of junk mail given to each house
def pieces_per_house : ℕ := 8

-- Define the number of houses in each block
def houses_per_block : ℕ := 4

-- Calculate the total number of pieces of junk mail given to each block
def total_pieces_per_block : ℕ := pieces_per_house * houses_per_block

-- Prove that the total number of pieces of junk mail given to each block is 32
theorem pieces_per_block_is_32 : total_pieces_per_block = 32 := 
by sorry

end pieces_per_block_is_32_l116_116890


namespace minimum_value_f_b_eq_neg_a_maximum_value_ab_inequality_for_f_and_f_l116_116440

noncomputable def f (a b x : ℝ) := Real.exp x - a * x - b

theorem minimum_value_f_b_eq_neg_a (a : ℝ) (h : 0 < a) :
  ∃ m, m = 2 * a - a * Real.log a ∧ ∀ x : ℝ, f a (-a) x ≥ m :=
sorry

theorem maximum_value_ab (a b : ℝ) (h : ∀ x : ℝ, f a b x + a ≥ 0) :
  ab ≤ (1 / 2) * Real.exp 3 :=
sorry

theorem inequality_for_f_and_f' (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : b = -a) (h3 : f a b x1 = 0) (h4 : f a b x2 = 0) (h5 : x1 < x2)
  : f a (-a) (3 * Real.log a) > (Real.exp ((2 * x1 * x2) / (x1 + x2)) - a) :=
sorry

end minimum_value_f_b_eq_neg_a_maximum_value_ab_inequality_for_f_and_f_l116_116440


namespace poles_on_each_side_l116_116388

theorem poles_on_each_side (total_poles : ℕ) (sides_equal : ℕ)
  (h1 : total_poles = 104) (h2 : sides_equal = 4) : 
  (total_poles / sides_equal) = 26 :=
by
  sorry

end poles_on_each_side_l116_116388


namespace right_triangle_area_l116_116891

theorem right_triangle_area :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ a * a + b * b = c * c ∧ (1/2 : ℝ) * a * b = 6 := 
sorry

end right_triangle_area_l116_116891


namespace expected_value_girls_left_of_boys_l116_116377

theorem expected_value_girls_left_of_boys :
  let boys := 10
      girls := 7
      students := boys + girls in
  (∀ (lineup : Finset (Fin students)), let event := { l : Finset (Fin students) | ∃ g : Fin girls, g < boys - 1} in
       ProbabilityTheory.expectation (λ p, (lineup ∩ event).card)) = 7 / 11 := 
sorry

end expected_value_girls_left_of_boys_l116_116377


namespace largest_integer_not_sum_of_30_and_composite_l116_116677

theorem largest_integer_not_sum_of_30_and_composite : 
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b < 30 ∧ ¬ prime b ∧ (n = 30 * a + b) → n = 157 :=
by
  sorry

end largest_integer_not_sum_of_30_and_composite_l116_116677


namespace isosceles_triangle_perimeter_l116_116931

/-
Problem:
Given an isosceles triangle with side lengths 5 and 6, prove that the perimeter of the triangle is either 16 or 17.
-/

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5 ∨ a = 6) (h₂ : b = 5 ∨ b = 6) (h₃ : a ≠ b) : 
  (a + a + b = 16 ∨ a + a + b = 17) ∧ (b + b + a = 16 ∨ b + b + a = 17) :=
by
  sorry

end isosceles_triangle_perimeter_l116_116931


namespace g_inv_g_inv_14_l116_116841

noncomputable def g (x : ℝ) := 3 * x - 4
noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end g_inv_g_inv_14_l116_116841


namespace joseph_investment_after_two_years_l116_116327

noncomputable def initial_investment : ℝ := 1000
noncomputable def monthly_addition : ℝ := 100
noncomputable def yearly_interest_rate : ℝ := 0.10
noncomputable def time_in_years : ℕ := 2

theorem joseph_investment_after_two_years :
  let first_year_total := initial_investment + 12 * monthly_addition
  let first_year_interest := first_year_total * yearly_interest_rate
  let end_of_first_year_total := first_year_total + first_year_interest
  let second_year_total := end_of_first_year_total + 12 * monthly_addition
  let second_year_interest := second_year_total * yearly_interest_rate
  let end_of_second_year_total := second_year_total + second_year_interest
  end_of_second_year_total = 3982 := 
by
  sorry

end joseph_investment_after_two_years_l116_116327


namespace fraction_difference_l116_116087

theorem fraction_difference :
  (↑(1+4+7) / ↑(2+5+8)) - (↑(2+5+8) / ↑(1+4+7)) = - (9 / 20) :=
by
  sorry

end fraction_difference_l116_116087


namespace pirates_treasure_l116_116489

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l116_116489


namespace recurrence_relation_l116_116546

def u (n : ℕ) : ℕ := sorry

theorem recurrence_relation (n : ℕ) : 
  u (n + 1) = (n + 1) * u n - (n * (n - 1)) / 2 * u (n - 2) :=
sorry

end recurrence_relation_l116_116546


namespace find_f2_l116_116296

variable (f g : ℝ → ℝ) (a : ℝ)

-- Definitions based on conditions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x
def equation (f g : ℝ → ℝ) (a : ℝ) := ∀ x, f x + g x = a^x - a^(-x) + 2

-- Lean statement for the proof problem
theorem find_f2
  (h1 : is_odd f)
  (h2 : is_even g)
  (h3 : equation f g a)
  (h4 : g 2 = a) : f 2 = 15 / 4 :=
by
  sorry

end find_f2_l116_116296


namespace F_transformed_l116_116528

-- Define the coordinates of point F
def F : ℝ × ℝ := (1, 0)

-- Reflection over the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Reflection over the y-axis
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Reflection over the line y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

-- Point F after all transformations
def F_final : ℝ × ℝ :=
  reflect_y_eq_x (reflect_y (reflect_x F))

-- Statement to prove
theorem F_transformed : F_final = (0, -1) :=
  sorry

end F_transformed_l116_116528


namespace oranges_worth_as_much_as_bananas_l116_116342

-- Define the given conditions
def worth_same_bananas_oranges (bananas oranges : ℕ) : Prop :=
  (3 / 4 * 12 : ℝ) = 9 ∧ 9 = 6

/-- Prove how many oranges are worth as much as (2 / 3) * 9 bananas,
    given that (3 / 4) * 12 bananas are worth 6 oranges. -/
theorem oranges_worth_as_much_as_bananas :
  worth_same_bananas_oranges 12 6 →
  (2 / 3 * 9 : ℝ) = 4 :=
by
  sorry

end oranges_worth_as_much_as_bananas_l116_116342


namespace kaleb_bought_new_books_l116_116973

theorem kaleb_bought_new_books :
  ∀ (TotalBooksSold KalebHasNow InitialBooks NewBooksBought : ℕ), 
  TotalBooksSold = 17 →
  InitialBooks = 34 →
  KalebHasNow = 24 → 
  NewBooksBought = 24 - (34 - 17) := 
by
  intros TotalBooksSold KalebHasNow InitialBooks NewBooksBought hSold hInit hNow
  rw [hSold, hInit, hNow]
  exact rfl

end kaleb_bought_new_books_l116_116973
