import Mathlib

namespace NUMINAMATH_GPT_sound_pressure_level_l2126_212644

theorem sound_pressure_level (p_0 p_1 p_2 p_3 : ℝ) (h_p0 : 0 < p_0)
  (L_p : ℝ → ℝ)
  (h_gasoline : 60 ≤ L_p p_1 ∧ L_p p_1 ≤ 90)
  (h_hybrid : 50 ≤ L_p p_2 ∧ L_p p_2 ≤ 60)
  (h_electric : L_p p_3 = 40)
  (h_L_p : ∀ p, L_p p = 20 * Real.log (p / p_0))
  : p_2 ≤ p_1 ∧ p_1 ≤ 100 * p_2 :=
by
  sorry

end NUMINAMATH_GPT_sound_pressure_level_l2126_212644


namespace NUMINAMATH_GPT_tangent_line_at_point_l2126_212662

noncomputable def tangent_line_equation (x : ℝ) : Prop :=
  ∀ y : ℝ, y = x * (3 * Real.log x + 1) → (x = 1 ∧ y = 1) → y = 4 * x - 3

theorem tangent_line_at_point : tangent_line_equation 1 :=
sorry

end NUMINAMATH_GPT_tangent_line_at_point_l2126_212662


namespace NUMINAMATH_GPT_log_increasing_condition_log_increasing_not_necessary_l2126_212639

theorem log_increasing_condition (a : ℝ) (h : a > 2) : a > 1 :=
by sorry

theorem log_increasing_not_necessary (a : ℝ) : ∃ b, (b > 1 ∧ ¬(b > 2)) :=
by sorry

end NUMINAMATH_GPT_log_increasing_condition_log_increasing_not_necessary_l2126_212639


namespace NUMINAMATH_GPT_unanswered_questions_count_l2126_212668

-- Define the variables: c (correct), w (wrong), u (unanswered)
variables (c w u : ℕ)

-- Define the conditions based on the problem statement.
def total_questions (c w u : ℕ) : Prop := c + w + u = 35
def new_system_score (c u : ℕ) : Prop := 6 * c + 3 * u = 120
def old_system_score (c w : ℕ) : Prop := 5 * c - 2 * w = 55

-- Prove that the number of unanswered questions, u, equals 10
theorem unanswered_questions_count (c w u : ℕ) 
    (h1 : total_questions c w u)
    (h2 : new_system_score c u)
    (h3 : old_system_score c w) : u = 10 :=
by
  sorry

end NUMINAMATH_GPT_unanswered_questions_count_l2126_212668


namespace NUMINAMATH_GPT_first_number_is_38_l2126_212681

theorem first_number_is_38 (x y : ℕ) (h1 : x + 2 * y = 124) (h2 : y = 43) : x = 38 :=
by
  sorry

end NUMINAMATH_GPT_first_number_is_38_l2126_212681


namespace NUMINAMATH_GPT_haley_collected_cans_l2126_212636

theorem haley_collected_cans (C : ℕ) (h : C - 7 = 2) : C = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_haley_collected_cans_l2126_212636


namespace NUMINAMATH_GPT_min_stamps_l2126_212659

theorem min_stamps : ∃ (x y : ℕ), 5 * x + 7 * y = 35 ∧ x + y = 5 :=
by
  have : ∀ (x y : ℕ), 5 * x + 7 * y = 35 → x + y = 5 → True := sorry
  sorry

end NUMINAMATH_GPT_min_stamps_l2126_212659


namespace NUMINAMATH_GPT_optimal_rental_decision_optimal_purchase_decision_l2126_212660

-- Definitions of conditions
def monthly_fee_first : ℕ := 50000
def monthly_fee_second : ℕ := 10000
def probability_seizure : ℚ := 0.5
def moving_cost : ℕ := 70000
def months_first_year : ℕ := 12
def months_seizure : ℕ := 4
def months_after_seizure : ℕ := months_first_year - months_seizure
def purchase_cost : ℕ := 2000000
def installment_period : ℕ := 36

-- Proving initial rental decision
theorem optimal_rental_decision :
  let annual_cost_first := monthly_fee_first * months_first_year
  let annual_cost_second := (monthly_fee_second * months_seizure) + (monthly_fee_first * months_after_seizure) + moving_cost
  annual_cost_second < annual_cost_first := 
by
  sorry

-- Proving purchasing decision
theorem optimal_purchase_decision :
  let total_rent_cost_after_seizure := (monthly_fee_second * months_seizure) + moving_cost + (monthly_fee_first * (4 * months_first_year - months_seizure))
  let total_purchase_cost := purchase_cost
  total_purchase_cost < total_rent_cost_after_seizure :=
by
  sorry

end NUMINAMATH_GPT_optimal_rental_decision_optimal_purchase_decision_l2126_212660


namespace NUMINAMATH_GPT_nishita_common_shares_l2126_212665

def annual_dividend_preferred_shares (num_preferred_shares : ℕ) (par_value : ℕ) (dividend_rate_preferred : ℕ) : ℕ :=
  (dividend_rate_preferred * par_value * num_preferred_shares) / 100

def annual_dividend_common_shares (total_dividend : ℕ) (dividend_preferred : ℕ) : ℕ :=
  total_dividend - dividend_preferred

def number_of_common_shares (annual_dividend_common : ℕ) (par_value : ℕ) (annual_rate_common : ℕ) : ℕ :=
  annual_dividend_common / ((annual_rate_common * par_value) / 100)

theorem nishita_common_shares (total_annual_dividend : ℕ) (num_preferred_shares : ℕ)
                             (par_value : ℕ) (dividend_rate_preferred : ℕ)
                             (semi_annual_rate_common : ℕ) : 
                             (number_of_common_shares (annual_dividend_common_shares total_annual_dividend 
                             (annual_dividend_preferred_shares num_preferred_shares par_value dividend_rate_preferred)) 
                             par_value (semi_annual_rate_common * 2)) = 3000 :=
by
  -- Provide values specific to the problem
  let total_annual_dividend := 16500
  let num_preferred_shares := 1200
  let par_value := 50
  let dividend_rate_preferred := 10
  let semi_annual_rate_common := 3.5
  sorry

end NUMINAMATH_GPT_nishita_common_shares_l2126_212665


namespace NUMINAMATH_GPT_find_article_cost_l2126_212615

noncomputable def original_cost_price (C S : ℝ) :=
  (S = 1.25 * C) ∧
  (S - 6.30 = 1.04 * C)

theorem find_article_cost (C S : ℝ) (h : original_cost_price C S) : C = 30 :=
by sorry

end NUMINAMATH_GPT_find_article_cost_l2126_212615


namespace NUMINAMATH_GPT_dvds_rented_l2126_212619

def total_cost : ℝ := 4.80
def cost_per_dvd : ℝ := 1.20

theorem dvds_rented : total_cost / cost_per_dvd = 4 := 
by
  sorry

end NUMINAMATH_GPT_dvds_rented_l2126_212619


namespace NUMINAMATH_GPT_initial_students_per_class_l2126_212606

theorem initial_students_per_class
  (S : ℕ) 
  (parents chaperones left_students left_chaperones : ℕ)
  (teachers remaining_individuals : ℕ)
  (h1 : parents = 5)
  (h2 : chaperones = 2)
  (h3 : left_students = 10)
  (h4 : left_chaperones = 2)
  (h5 : teachers = 2)
  (h6 : remaining_individuals = 15)
  (h7 : 2 * S + parents + teachers - left_students - left_chaperones = remaining_individuals) :
  S = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_students_per_class_l2126_212606


namespace NUMINAMATH_GPT_smallest_integer_cube_ends_in_392_l2126_212624

theorem smallest_integer_cube_ends_in_392 : ∃ n : ℕ, (n > 0) ∧ (n^3 % 1000 = 392) ∧ ∀ m : ℕ, (m > 0) ∧ (m^3 % 1000 = 392) → n ≤ m :=
by 
  sorry

end NUMINAMATH_GPT_smallest_integer_cube_ends_in_392_l2126_212624


namespace NUMINAMATH_GPT_minimum_value_4x_minus_y_l2126_212642

theorem minimum_value_4x_minus_y (x y : ℝ) (h1 : x - y ≥ 0) (h2 : x + y - 4 ≥ 0) (h3 : x ≤ 4) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x' y' : ℝ), (x' - y' ≥ 0) → (x' + y' - 4 ≥ 0) → (x' ≤ 4) → 4 * x' - y' ≥ m :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_4x_minus_y_l2126_212642


namespace NUMINAMATH_GPT_smallest_numbers_l2126_212693

-- Define the problem statement
theorem smallest_numbers (m n : ℕ) :
  (∃ (m1 n1 m2 n2 : ℕ), 7 * m1^2 - 11 * n1^2 = 1 ∧ 7 * m2^2 - 11 * n2^2 = 5) ↔
  (7 * m^2 - 11 * n^2 = 1) ∨ (7 * m^2 - 11 * n^2 = 5) :=
by
  sorry

end NUMINAMATH_GPT_smallest_numbers_l2126_212693


namespace NUMINAMATH_GPT_factorize_expression_l2126_212664

variable (a : ℝ)

theorem factorize_expression : a^3 + 4 * a^2 + 4 * a = a * (a + 2)^2 := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2126_212664


namespace NUMINAMATH_GPT_polynomial_abs_sum_roots_l2126_212618

theorem polynomial_abs_sum_roots (p q r m : ℤ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2500) (h3 : p * q * r = -m) :
  |p| + |q| + |r| = 100 :=
sorry

end NUMINAMATH_GPT_polynomial_abs_sum_roots_l2126_212618


namespace NUMINAMATH_GPT_female_athletes_drawn_l2126_212621

theorem female_athletes_drawn (total_athletes male_athletes female_athletes sample_size : ℕ)
  (h_total : total_athletes = male_athletes + female_athletes)
  (h_team : male_athletes = 48 ∧ female_athletes = 36)
  (h_sample_size : sample_size = 35) :
  (female_athletes * sample_size) / total_athletes = 15 :=
by
  sorry

end NUMINAMATH_GPT_female_athletes_drawn_l2126_212621


namespace NUMINAMATH_GPT_value_of_3b_minus_a_l2126_212694

theorem value_of_3b_minus_a :
  ∃ (a b : ℕ), (a > b) ∧ (a >= 0) ∧ (b >= 0) ∧ (∀ x : ℝ, (x - a) * (x - b) = x^2 - 16 * x + 60) ∧ (3 * b - a = 8) := 
sorry

end NUMINAMATH_GPT_value_of_3b_minus_a_l2126_212694


namespace NUMINAMATH_GPT_part3_conclusion_l2126_212673

-- Definitions and conditions for the problem
def quadratic_function (a x : ℝ) : ℝ := (x - a)^2 + a - 1

-- Part 1: Given condition that (1, 2) lies on the graph of the quadratic function
def part1_condition (a : ℝ) := (quadratic_function a 1) = 2

-- Part 2: Given condition that the function has a minimum value of 2 for 1 ≤ x ≤ 4
def part2_condition (a : ℝ) := ∀ x, 1 ≤ x ∧ x ≤ 4 → quadratic_function a x ≥ 2

-- Part 3: Given condition (m, n) on the graph where m > 0 and m > 2a
def part3_condition (a m n : ℝ) := m > 0 ∧ m > 2 * a ∧ quadratic_function a m = n

-- Conclusion for Part 3: Prove that n > -5/4
theorem part3_conclusion (a m n : ℝ) (h : part3_condition a m n) : n > -5/4 := 
sorry  -- Proof required here

end NUMINAMATH_GPT_part3_conclusion_l2126_212673


namespace NUMINAMATH_GPT_farmer_john_pairs_l2126_212675

noncomputable def farmer_john_animals_pairing :
    Nat := 
  let cows := 5
  let pigs := 4
  let horses := 7
  let num_ways_cow_pig_pair := cows * pigs
  let num_ways_horses_remaining := Nat.factorial horses
  num_ways_cow_pig_pair * num_ways_horses_remaining

theorem farmer_john_pairs : farmer_john_animals_pairing = 100800 := 
by
  sorry

end NUMINAMATH_GPT_farmer_john_pairs_l2126_212675


namespace NUMINAMATH_GPT_coffee_pods_per_box_l2126_212667

theorem coffee_pods_per_box (d k : ℕ) (c e : ℝ) (h1 : d = 40) (h2 : k = 3) (h3 : c = 8) (h4 : e = 32) :
  ∃ b : ℕ, b = 30 :=
by
  sorry

end NUMINAMATH_GPT_coffee_pods_per_box_l2126_212667


namespace NUMINAMATH_GPT_stamps_ratio_l2126_212645

theorem stamps_ratio (orig_stamps_P : ℕ) (addie_stamps : ℕ) (final_stamps_P : ℕ) 
  (h₁ : orig_stamps_P = 18) (h₂ : addie_stamps = 72) (h₃ : final_stamps_P = 36) :
  (final_stamps_P - orig_stamps_P) / addie_stamps = 1 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_stamps_ratio_l2126_212645


namespace NUMINAMATH_GPT_min_attempts_sufficient_a_l2126_212678

theorem min_attempts_sufficient_a (n : ℕ) (h : n > 2)
  (good_batteries bad_batteries : ℕ)
  (h1 : good_batteries = n + 1)
  (h2 : bad_batteries = n)
  (total_batteries := 2 * n + 1) :
  (∃ attempts, attempts = n + 1) := sorry

end NUMINAMATH_GPT_min_attempts_sufficient_a_l2126_212678


namespace NUMINAMATH_GPT_general_term_formula_sum_of_b_first_terms_l2126_212684

variable (a₁ a₂ : ℝ)
variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (T : ℕ → ℝ)

-- Conditions
axiom h1 : a₁ * a₂ = 8
axiom h2 : a₁ + a₂ = 6
axiom increasing_geometric_sequence : ∀ n : ℕ, a (n+1) = a (n) * (a₂ / a₁)
axiom initial_conditions : a 1 = a₁ ∧ a 2 = a₂
axiom b_def : ∀ n, b n = 2 * a n + 3

-- To Prove
theorem general_term_formula : ∀ n: ℕ, a n = 2 ^ (n + 1) :=
sorry

theorem sum_of_b_first_terms (n : ℕ) : T n = 2 ^ (n + 2) - 4 + 3 * n :=
sorry

end NUMINAMATH_GPT_general_term_formula_sum_of_b_first_terms_l2126_212684


namespace NUMINAMATH_GPT_derivative_of_f_l2126_212687

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x

theorem derivative_of_f :
  ∀ x ≠ 0, deriv f x = ((-x * Real.sin x - Real.cos x) / (x^2)) := sorry

end NUMINAMATH_GPT_derivative_of_f_l2126_212687


namespace NUMINAMATH_GPT_total_donations_l2126_212699

-- Define the conditions
def started_donating_age : ℕ := 17
def current_age : ℕ := 71
def annual_donation : ℕ := 8000

-- Define the proof problem to show the total donation amount equals $432,000
theorem total_donations : (current_age - started_donating_age) * annual_donation = 432000 := 
by
  sorry

end NUMINAMATH_GPT_total_donations_l2126_212699


namespace NUMINAMATH_GPT_fifth_stack_33_l2126_212674

def cups_in_fifth_stack (a d : ℕ) : ℕ :=
a + 4 * d

theorem fifth_stack_33 
  (a : ℕ) 
  (d : ℕ) 
  (h_first_stack : a = 17) 
  (h_pattern : d = 4) : 
  cups_in_fifth_stack a d = 33 := by
  sorry

end NUMINAMATH_GPT_fifth_stack_33_l2126_212674


namespace NUMINAMATH_GPT_solve_for_a_l2126_212680

theorem solve_for_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : x - a * y = 3) : a = -1 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l2126_212680


namespace NUMINAMATH_GPT_value_of_expression_l2126_212608

-- Defining the given conditions as Lean definitions
def x : ℚ := 2 / 3
def y : ℚ := 5 / 2

-- The theorem statement to prove that the given expression equals the correct answer
theorem value_of_expression : (1 / 3) * x^7 * y^6 = 125 / 261 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2126_212608


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2126_212622

theorem point_in_fourth_quadrant (m : ℝ) : (m-1 > 0 ∧ 2-m < 0) ↔ m > 2 :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2126_212622


namespace NUMINAMATH_GPT_initial_red_marbles_l2126_212637

variable (r g : ℝ)

def red_green_ratio_initial (r g : ℝ) : Prop := r / g = 5 / 3
def red_green_ratio_new (r g : ℝ) : Prop := (r + 15) / (g - 9) = 3 / 1

theorem initial_red_marbles (r g : ℝ) (h₁ : red_green_ratio_initial r g) (h₂ : red_green_ratio_new r g) : r = 52.5 := sorry

end NUMINAMATH_GPT_initial_red_marbles_l2126_212637


namespace NUMINAMATH_GPT_regular_polygon_sides_l2126_212628

theorem regular_polygon_sides (O A B : Type) (angle_OAB : ℝ) 
  (h_angle : angle_OAB = 72) : 
  (360 / angle_OAB = 5) := 
by 
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2126_212628


namespace NUMINAMATH_GPT_gcd_of_60_and_75_l2126_212625

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end NUMINAMATH_GPT_gcd_of_60_and_75_l2126_212625


namespace NUMINAMATH_GPT_inequality_proof_l2126_212671

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l2126_212671


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2126_212638

noncomputable def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a 1 + n * d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 = 2)
  (h3 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2126_212638


namespace NUMINAMATH_GPT_half_abs_diff_squares_l2126_212670

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 21) (h₂ : b = 17) :
  (|a^2 - b^2| / 2) = 76 :=
by 
  sorry

end NUMINAMATH_GPT_half_abs_diff_squares_l2126_212670


namespace NUMINAMATH_GPT_trig_expression_value_l2126_212654

theorem trig_expression_value (α : ℝ) (h : Real.tan (Real.pi + α) = 2) : 
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) = 3 := 
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l2126_212654


namespace NUMINAMATH_GPT_cot_sum_simplified_l2126_212609

noncomputable def cot (x : ℝ) : ℝ := (Real.cos x) / (Real.sin x)

theorem cot_sum_simplified : cot (π / 24) + cot (π / 8) = 96 / (π^2) := 
by 
  sorry

end NUMINAMATH_GPT_cot_sum_simplified_l2126_212609


namespace NUMINAMATH_GPT_smallest_prime_with_composite_reverse_l2126_212686

def is_prime (n : Nat) : Prop := 
  n > 1 ∧ ∀ m : Nat, m > 1 ∧ m < n → n % m ≠ 0

def is_composite (n : Nat) : Prop :=
  n > 1 ∧ ∃ m : Nat, m > 1 ∧ m < n ∧ n % m = 0

def reverse_digits (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

theorem smallest_prime_with_composite_reverse :
  ∃ (n : Nat), 10 ≤ n ∧ n < 100 ∧ is_prime n ∧ (n / 10 = 3) ∧ is_composite (reverse_digits n) ∧
  (∀ m : Nat, 10 ≤ m ∧ m < n ∧ (m / 10 = 3) ∧ is_prime m → ¬is_composite (reverse_digits m)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_with_composite_reverse_l2126_212686


namespace NUMINAMATH_GPT_range_of_a_l2126_212669

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2126_212669


namespace NUMINAMATH_GPT_scientific_notation_of_1206_million_l2126_212607

theorem scientific_notation_of_1206_million :
  (1206 * 10^6 : ℝ) = 1.206 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_1206_million_l2126_212607


namespace NUMINAMATH_GPT_not_perfect_cube_l2126_212691

theorem not_perfect_cube (n : ℕ) : ¬ ∃ k : ℕ, k ^ 3 = 2 ^ (2 ^ n) + 1 :=
sorry

end NUMINAMATH_GPT_not_perfect_cube_l2126_212691


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2126_212600

variable {a : ℕ → ℕ}

def is_geometric_sequence_with_common_product (k : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem geometric_sequence_sum :
  is_geometric_sequence_with_common_product 27 a →
  a 1 = 1 →
  a 2 = 3 →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 +
   a 11 + a 12 + a 13 + a 14 + a 15 + a 16 + a 17 + a 18) = 78 :=
by
  intros h_geom h_a1 h_a2
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2126_212600


namespace NUMINAMATH_GPT_three_digit_addition_l2126_212653

theorem three_digit_addition (a b : ℕ) (h₁ : 307 = 300 + a * 10 + 7) (h₂ : 416 + 10 * (a * 1) + 7 = 700 + b * 10 + 3) (h₃ : (7 + b + 3) % 3 = 0) : a + b = 2 :=
by
  -- mock proof, since solution steps are not considered
  sorry

end NUMINAMATH_GPT_three_digit_addition_l2126_212653


namespace NUMINAMATH_GPT_describe_set_T_l2126_212635

-- Define the conditions for the set of points T
def satisfies_conditions (x y : ℝ) : Prop :=
  (x + 3 = 4 ∧ y < 7) ∨ (y - 3 = 4 ∧ x < 1)

-- Define the set T based on the conditions
def set_T := {p : ℝ × ℝ | satisfies_conditions p.1 p.2}

-- Statement to prove the geometric description of the set T
theorem describe_set_T :
  (∃ x y, satisfies_conditions x y) → ∃ p1 p2,
  (p1 = (1, t) ∧ t < 7 → satisfies_conditions 1 t) ∧
  (p2 = (t, 7) ∧ t < 1 → satisfies_conditions t 7) ∧
  (p1 ≠ p2) :=
sorry

end NUMINAMATH_GPT_describe_set_T_l2126_212635


namespace NUMINAMATH_GPT_part_1_part_2_l2126_212633

variable {a b : ℝ}

theorem part_1 (ha : a > 0) (hb : b > 0) : a^2 + 3 * b^2 ≥ 2 * b * (a + b) :=
sorry

theorem part_2 (ha : a > 0) (hb : b > 0) : a^3 + b^3 ≥ a * b^2 + a^2 * b :=
sorry

end NUMINAMATH_GPT_part_1_part_2_l2126_212633


namespace NUMINAMATH_GPT_part_I_part_II_l2126_212683

-- Problem conditions as definitions
variable (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : a + b = 1)

-- Statement for part (Ⅰ)
theorem part_I : (1 / a) + (1 / b) ≥ 4 :=
by
  sorry

-- Statement for part (Ⅱ)
theorem part_II : (1 / (a ^ 2016)) + (1 / (b ^ 2016)) ≥ 2 ^ 2017 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l2126_212683


namespace NUMINAMATH_GPT_certain_number_equation_l2126_212605

theorem certain_number_equation (x : ℤ) (h : 16 * x + 17 * x + 20 * x + 11 = 170) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_certain_number_equation_l2126_212605


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_min_area_line_eq_l2126_212652

section part_one

variable (m x y : ℝ)

def line_eq := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4

theorem line_passes_through_fixed_point :
  ∀ m, line_eq m 3 1 = 0 :=
sorry

end part_one

section part_two

variable (k x y : ℝ)

def line_eq_l1 (k : ℝ) := y = k * (x - 3) + 1

theorem min_area_line_eq :
  line_eq_l1 (-1/3) x y = (x + 3 * y - 6 = 0) :=
sorry

end part_two

end NUMINAMATH_GPT_line_passes_through_fixed_point_min_area_line_eq_l2126_212652


namespace NUMINAMATH_GPT_tangent_line_equation_l2126_212629

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 5

def point_A : ℝ × ℝ := (1, -2)

theorem tangent_line_equation :
  ∀ x y : ℝ, (y = 4 * x - 6) ↔ (fderiv ℝ f (point_A.1) x = 4) ∧ (y = f (point_A.1) + 4 * (x - point_A.1)) := by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l2126_212629


namespace NUMINAMATH_GPT_problem_statement_l2126_212616

-- Definitions based on problem conditions
def p (a b c : ℝ) : Prop := a > b → (a * c^2 > b * c^2)

def q : Prop := ∃ x_0 : ℝ, (x_0 > 0) ∧ (x_0 - 1 + Real.log x_0 = 0)

-- Main theorem
theorem problem_statement : (¬ (∀ a b c : ℝ, p a b c)) ∧ q :=
by sorry

end NUMINAMATH_GPT_problem_statement_l2126_212616


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l2126_212679

-- Proof for equation (1)
theorem solve_equation1 : ∃ x : ℝ, 2 * (2 * x + 1) - (3 * x - 4) = 2 := by
  exists -4
  sorry

-- Proof for equation (2)
theorem solve_equation2 : ∃ y : ℝ, (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 := by
  exists -1
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l2126_212679


namespace NUMINAMATH_GPT_boxes_with_neither_l2126_212610

-- Definitions translating the conditions from the problem
def total_boxes : Nat := 15
def boxes_with_markers : Nat := 8
def boxes_with_crayons : Nat := 4
def boxes_with_both : Nat := 3

-- The theorem statement to prove
theorem boxes_with_neither : total_boxes - (boxes_with_markers + boxes_with_crayons - boxes_with_both) = 6 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_boxes_with_neither_l2126_212610


namespace NUMINAMATH_GPT_dogs_in_garden_l2126_212640

theorem dogs_in_garden (D : ℕ) (ducks : ℕ) (total_feet : ℕ) (dogs_feet : ℕ) (ducks_feet : ℕ) 
  (h1 : ducks = 2) 
  (h2 : total_feet = 28)
  (h3 : dogs_feet = 4)
  (h4 : ducks_feet = 2) 
  (h_eq : dogs_feet * D + ducks_feet * ducks = total_feet) : 
  D = 6 := by
  sorry

end NUMINAMATH_GPT_dogs_in_garden_l2126_212640


namespace NUMINAMATH_GPT_Emilee_earnings_l2126_212682

theorem Emilee_earnings (J R_j T R_t E R_e : ℕ) :
  (R_j * J = 35) → 
  (R_t * T = 30) → 
  (R_j * J + R_t * T + R_e * E = 90) → 
  (R_e * E = 25) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_Emilee_earnings_l2126_212682


namespace NUMINAMATH_GPT_scissors_total_l2126_212632

theorem scissors_total (initial_scissors : ℕ) (additional_scissors : ℕ) (h1 : initial_scissors = 54) (h2 : additional_scissors = 22) : 
  initial_scissors + additional_scissors = 76 :=
by
  sorry

end NUMINAMATH_GPT_scissors_total_l2126_212632


namespace NUMINAMATH_GPT_two_point_five_one_million_in_scientific_notation_l2126_212603

theorem two_point_five_one_million_in_scientific_notation :
  (2.51 * 10^6 : ℝ) = 2.51e6 := 
sorry

end NUMINAMATH_GPT_two_point_five_one_million_in_scientific_notation_l2126_212603


namespace NUMINAMATH_GPT_exponential_monotonicity_example_l2126_212617

theorem exponential_monotonicity_example (m n : ℕ) (a b : ℝ) (h1 : a = 0.2 ^ m) (h2 : b = 0.2 ^ n) (h3 : m > n) : a < b :=
by
  sorry

end NUMINAMATH_GPT_exponential_monotonicity_example_l2126_212617


namespace NUMINAMATH_GPT_intersecting_lines_l2126_212641

def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem intersecting_lines (x y : ℝ) : x ≠ 0 → y ≠ 0 → 
  (diamond x y = diamond y x) ↔ (y = x ∨ y = -x) := 
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_l2126_212641


namespace NUMINAMATH_GPT_trigonometric_identity_cos24_cos36_sub_sin24_cos54_l2126_212657

theorem trigonometric_identity_cos24_cos36_sub_sin24_cos54  :
  (Real.cos (24 * Real.pi / 180) * Real.cos (36 * Real.pi / 180) - Real.sin (24 * Real.pi / 180) * Real.cos (54 * Real.pi / 180) = 1 / 2) := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_cos24_cos36_sub_sin24_cos54_l2126_212657


namespace NUMINAMATH_GPT_donna_paid_165_l2126_212601

def original_price : ℝ := 200
def discount_rate : ℝ := 0.25
def tax_rate : ℝ := 0.1

def sale_price := original_price * (1 - discount_rate)
def tax := sale_price * tax_rate
def total_amount_paid := sale_price + tax

theorem donna_paid_165 : total_amount_paid = 165 := by
  sorry

end NUMINAMATH_GPT_donna_paid_165_l2126_212601


namespace NUMINAMATH_GPT_markers_blue_l2126_212623

theorem markers_blue {total_markers red_markers blue_markers : ℝ} 
  (h_total : total_markers = 64.0) 
  (h_red : red_markers = 41.0) 
  (h_blue : blue_markers = total_markers - red_markers) : 
  blue_markers = 23.0 := 
by 
  sorry

end NUMINAMATH_GPT_markers_blue_l2126_212623


namespace NUMINAMATH_GPT_Q_at_1_eq_1_l2126_212695

noncomputable def Q (x : ℚ) : ℚ := x^4 - 16*x^2 + 16

theorem Q_at_1_eq_1 : Q 1 = 1 := by
  sorry

end NUMINAMATH_GPT_Q_at_1_eq_1_l2126_212695


namespace NUMINAMATH_GPT_subset_N_M_l2126_212620

def M : Set ℝ := { x | ∃ (k : ℤ), x = k / 2 + 1 / 3 }
def N : Set ℝ := { x | ∃ (k : ℤ), x = k + 1 / 3 }

theorem subset_N_M : N ⊆ M := 
  sorry

end NUMINAMATH_GPT_subset_N_M_l2126_212620


namespace NUMINAMATH_GPT_sum_of_ab_conditions_l2126_212626

theorem sum_of_ab_conditions (a b : ℝ) (h : a^3 + b^3 = 1 - 3 * a * b) : a + b = 1 ∨ a + b = -2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_ab_conditions_l2126_212626


namespace NUMINAMATH_GPT_tan_beta_value_l2126_212613

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = -3 / 4) (h2 : Real.tan (α + β) = 1) : Real.tan β = 7 :=
sorry

end NUMINAMATH_GPT_tan_beta_value_l2126_212613


namespace NUMINAMATH_GPT_garden_dimensions_l2126_212630

theorem garden_dimensions
  (w l : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : l * w = 600) : 
  w = 10 * Real.sqrt 3 ∧ l = 20 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_garden_dimensions_l2126_212630


namespace NUMINAMATH_GPT_sequence_an_general_formula_sum_bn_formula_l2126_212631

variable (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)

axiom seq_Sn_eq_2an_minus_n : ∀ n : ℕ, n > 0 → S n + n = 2 * a n

theorem sequence_an_general_formula (n : ℕ) (h : n > 0) :
  (∀ n > 0, a n + 1 = 2 * (a (n - 1) + 1)) ∧ (a n = 2^n - 1) :=
sorry

theorem sum_bn_formula (n : ℕ) (h : n > 0) :
  (∀ n > 0, b n = n * a n + n) → T n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end NUMINAMATH_GPT_sequence_an_general_formula_sum_bn_formula_l2126_212631


namespace NUMINAMATH_GPT_total_clouds_l2126_212661

theorem total_clouds (C B : ℕ) (h1 : C = 6) (h2 : B = 3 * C) : C + B = 24 := by
  sorry

end NUMINAMATH_GPT_total_clouds_l2126_212661


namespace NUMINAMATH_GPT_bisection_method_root_exists_bisection_method_next_calculation_l2126_212663

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_method_root_exists :
  (f 0 < 0) → (f 0.5 > 0) → ∃ x0 : ℝ, 0 < x0 ∧ x0 < 0.5 ∧ f x0 = 0 :=
by
  intro h0 h05
  sorry

theorem bisection_method_next_calculation :
  f 0.25 = (0.25)^3 + 3 * 0.25 - 1 :=
by
  calc
    f 0.25 = 0.25^3 + 3 * 0.25 - 1 := rfl

end NUMINAMATH_GPT_bisection_method_root_exists_bisection_method_next_calculation_l2126_212663


namespace NUMINAMATH_GPT_polygon_sides_14_l2126_212655

theorem polygon_sides_14 (n : ℕ) (θ : ℝ) 
  (h₀ : (n - 2) * 180 - θ = 2000) :
  n = 14 :=
sorry

end NUMINAMATH_GPT_polygon_sides_14_l2126_212655


namespace NUMINAMATH_GPT_remainder_1234_mul_2047_mod_600_l2126_212656

theorem remainder_1234_mul_2047_mod_600 : (1234 * 2047) % 600 = 198 := by
  sorry

end NUMINAMATH_GPT_remainder_1234_mul_2047_mod_600_l2126_212656


namespace NUMINAMATH_GPT_sally_picked_peaches_l2126_212658

theorem sally_picked_peaches (original_peaches total_peaches picked_peaches : ℕ)
  (h_orig : original_peaches = 13)
  (h_total : total_peaches = 55)
  (h_picked : picked_peaches = total_peaches - original_peaches) :
  picked_peaches = 42 :=
by
  sorry

end NUMINAMATH_GPT_sally_picked_peaches_l2126_212658


namespace NUMINAMATH_GPT_fib_divisibility_l2126_212650

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

theorem fib_divisibility (m n : ℕ) (hm : 1 ≤ m) (hn : 1 < n) : 
  (fib (m * n - 1) - fib (n - 1) ^ m) % fib n ^ 2 = 0 :=
sorry

end NUMINAMATH_GPT_fib_divisibility_l2126_212650


namespace NUMINAMATH_GPT_book_original_price_l2126_212627

-- Definitions for conditions
def selling_price := 56
def profit_percentage := 75

-- Statement of the theorem
theorem book_original_price : ∃ CP : ℝ, selling_price = CP * (1 + profit_percentage / 100) ∧ CP = 32 :=
by
  sorry

end NUMINAMATH_GPT_book_original_price_l2126_212627


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2126_212649

theorem hyperbola_eccentricity (a b m n e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_mn : m * n = 2 / 9)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) : e = 3 * Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2126_212649


namespace NUMINAMATH_GPT_tower_no_knights_l2126_212634

-- Define the problem conditions in Lean

variable {T : Type} -- Type for towers
variable {K : Type} -- Type for knights

variable (towers : Fin 9 → T)
variable (knights : Fin 18 → K)

-- Movement of knights: each knight moves to a neighboring tower every hour (either clockwise or counterclockwise)
variable (moves : K → (T → T))

-- Each knight stands watch at each tower exactly once over the course of the night
variable (stands_watch : ∀ k : K, ∀ t : T, ∃ hour : Fin 9, moves k t = towers hour)

-- Condition: at one time (say hour 1), each tower had at least two knights on watch
variable (time1 : Fin 9 → Fin 9 → ℕ) -- Number of knights at each tower at hour 1
variable (cond1 : ∀ i : Fin 9, 2 ≤ time1 1 i)

-- Condition: at another time (say hour 2), exactly five towers each had exactly one knight on watch
variable (time2 : Fin 9 → Fin 9 → ℕ) -- Number of knights at each tower at hour 2
variable (cond2 : ∃ seq : Fin 5 → Fin 9, (∀ i : Fin 5, time2 2 (seq i) = 1) ∧ ∀ j : Fin 4, i ≠ j → 1 ≠ seq j)

-- Prove: there exists a time when one of the towers had no knights at all
theorem tower_no_knights : ∃ hour : Fin 9, ∃ i : Fin 9, moves (knights i) (towers hour) = towers hour ∧ (∀ knight : K, moves knight (towers hour) ≠ towers hour) :=
sorry

end NUMINAMATH_GPT_tower_no_knights_l2126_212634


namespace NUMINAMATH_GPT_lara_bought_52_stems_l2126_212677

-- Define the conditions given in the problem:
def flowers_given_to_mom : ℕ := 15
def flowers_given_to_grandma : ℕ := flowers_given_to_mom + 6
def flowers_in_vase : ℕ := 16

-- The total number of stems of flowers Lara bought should be:
def total_flowers_bought : ℕ := flowers_given_to_mom + flowers_given_to_grandma + flowers_in_vase

-- The main theorem to prove the total number of flowers Lara bought is 52:
theorem lara_bought_52_stems : total_flowers_bought = 52 := by
  sorry

end NUMINAMATH_GPT_lara_bought_52_stems_l2126_212677


namespace NUMINAMATH_GPT_unique_solution_exists_l2126_212643

theorem unique_solution_exists :
  ∃ (x y : ℝ), x = -13 / 96 ∧ y = 13 / 40 ∧
    (x / Real.sqrt (x^2 + y^2) - 1/x = 7) ∧
    (y / Real.sqrt (x^2 + y^2) + 1/y = 4) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_exists_l2126_212643


namespace NUMINAMATH_GPT_intersection_A_complement_B_l2126_212696

open Set

noncomputable def A : Set ℝ := {2, 3, 4, 5, 6}
noncomputable def B : Set ℝ := {x | x^2 - 8 * x + 12 >= 0}
noncomputable def complement_B : Set ℝ := {x | 2 < x ∧ x < 6}

theorem intersection_A_complement_B :
  A ∩ complement_B = {3, 4, 5} :=
sorry

end NUMINAMATH_GPT_intersection_A_complement_B_l2126_212696


namespace NUMINAMATH_GPT_small_cubes_with_painted_faces_l2126_212698

-- Definitions based on conditions
def large_cube_edge : ℕ := 8
def small_cube_edge : ℕ := 2
def division_factor : ℕ := large_cube_edge / small_cube_edge
def total_small_cubes : ℕ := division_factor ^ 3

-- Proving the number of cubes with specific painted faces.
theorem small_cubes_with_painted_faces :
  (8 : ℤ) = 8 ∧ -- 8 smaller cubes with three painted faces
  (24 : ℤ) = 24 ∧ -- 24 smaller cubes with two painted faces
  (24 : ℤ) = 24 := -- 24 smaller cubes with one painted face
by
  sorry

end NUMINAMATH_GPT_small_cubes_with_painted_faces_l2126_212698


namespace NUMINAMATH_GPT_completion_days_for_B_l2126_212604

-- Conditions
def A_completion_days := 20
def B_completion_days (x : ℕ) := x
def project_completion_days := 20
def A_work_days := project_completion_days - 10
def B_work_days := project_completion_days
def A_work_rate := 1 / A_completion_days
def B_work_rate (x : ℕ) := 1 / B_completion_days x
def combined_work_rate (x : ℕ) := A_work_rate + B_work_rate x
def A_project_completed := A_work_days * A_work_rate
def B_project_remaining (x : ℕ) := 1 - A_project_completed
def B_project_completion (x : ℕ) := B_work_days * B_work_rate x

-- Proof statement
theorem completion_days_for_B (x : ℕ) 
  (h : B_project_completion x = B_project_remaining x ∧ combined_work_rate x > 0) :
  x = 40 :=
sorry

end NUMINAMATH_GPT_completion_days_for_B_l2126_212604


namespace NUMINAMATH_GPT_find_constant_a_l2126_212685

noncomputable def f (a t : ℝ) : ℝ := (t - 2)^2 - 4 - a

theorem find_constant_a :
  (∃ (a : ℝ),
    (∀ (t : ℝ), -1 ≤ t ∧ t ≤ 1 → |f a t| ≤ 4) ∧ 
    (∃ (t : ℝ), -1 ≤ t ∧ t ≤ 1 ∧ |f a t| = 4)) →
  a = 1 :=
sorry

end NUMINAMATH_GPT_find_constant_a_l2126_212685


namespace NUMINAMATH_GPT_average_height_plants_l2126_212697

theorem average_height_plants (h1 h3 : ℕ) (h1_eq : h1 = 27) (h3_eq : h3 = 9)
  (prop : ∀ (h2 h4 : ℕ), (h2 = h1 / 3 ∨ h2 = h1 * 3) ∧ (h3 = h2 / 3 ∨ h3 = h2 * 3) ∧ (h4 = h3 / 3 ∨ h4 = h3 * 3)) : 
  ((27 + h2 + 9 + h4) / 4 = 12) :=
by 
  sorry

end NUMINAMATH_GPT_average_height_plants_l2126_212697


namespace NUMINAMATH_GPT_stella_profit_l2126_212651

-- Definitions based on the conditions
def number_of_dolls := 6
def price_per_doll := 8
def number_of_clocks := 4
def price_per_clock := 25
def number_of_glasses := 8
def price_per_glass := 6
def number_of_vases := 3
def price_per_vase := 12
def number_of_postcards := 10
def price_per_postcard := 3
def cost_of_merchandise := 250

-- Calculations based on given problem and solution
def revenue_from_dolls := number_of_dolls * price_per_doll
def revenue_from_clocks := number_of_clocks * price_per_clock
def revenue_from_glasses := number_of_glasses * price_per_glass
def revenue_from_vases := number_of_vases * price_per_vase
def revenue_from_postcards := number_of_postcards * price_per_postcard
def total_revenue := revenue_from_dolls + revenue_from_clocks + revenue_from_glasses + revenue_from_vases + revenue_from_postcards
def profit := total_revenue - cost_of_merchandise

-- Main theorem statement
theorem stella_profit : profit = 12 := by
  sorry

end NUMINAMATH_GPT_stella_profit_l2126_212651


namespace NUMINAMATH_GPT_sum_of_transformed_numbers_l2126_212690

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_transformed_numbers_l2126_212690


namespace NUMINAMATH_GPT_exactly_one_gt_one_of_abc_eq_one_l2126_212648

theorem exactly_one_gt_one_of_abc_eq_one 
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) 
  (h_sum : a + b + c > 1 / a + 1 / b + 1 / c) : 
  (1 < a ∧ b < 1 ∧ c < 1) ∨ (a < 1 ∧ 1 < b ∧ c < 1) ∨ (a < 1 ∧ b < 1 ∧ 1 < c) :=
sorry

end NUMINAMATH_GPT_exactly_one_gt_one_of_abc_eq_one_l2126_212648


namespace NUMINAMATH_GPT_pants_to_shirts_ratio_l2126_212689

-- Conditions
def shirts : ℕ := 4
def total_clothes : ℕ := 16

-- Given P as the number of pants and S as the number of shorts
variable (P S : ℕ)

-- State the conditions as hypotheses
axiom shorts_half_pants : S = P / 2
axiom total_clothes_condition : 4 + P + S = 16

-- Question: Prove that the ratio of pants to shirts is 2
theorem pants_to_shirts_ratio : P = 2 * shirts :=
by {
  -- insert proof steps here
  sorry
}

end NUMINAMATH_GPT_pants_to_shirts_ratio_l2126_212689


namespace NUMINAMATH_GPT_edward_original_lawns_l2126_212646

-- Definitions based on conditions
def dollars_per_lawn : ℕ := 4
def lawns_forgotten : ℕ := 9
def dollars_earned : ℕ := 32

-- The original number of lawns to mow
def original_lawns_to_mow (L : ℕ) : Prop :=
  dollars_per_lawn * (L - lawns_forgotten) = dollars_earned

-- The proof problem statement
theorem edward_original_lawns : ∃ L : ℕ, original_lawns_to_mow L ∧ L = 17 :=
by
  sorry

end NUMINAMATH_GPT_edward_original_lawns_l2126_212646


namespace NUMINAMATH_GPT_greatest_divisor_l2126_212676

theorem greatest_divisor (d : ℕ) :
  (1657 % d = 6 ∧ 2037 % d = 5) → d = 127 := by
  sorry

end NUMINAMATH_GPT_greatest_divisor_l2126_212676


namespace NUMINAMATH_GPT_num_divisible_by_7_in_range_l2126_212688

theorem num_divisible_by_7_in_range (n : ℤ) (h : 1 ≤ n ∧ n ≤ 2015)
    : (∃ k, 1 ≤ k ∧ k ≤ 335 ∧ 3 ^ (6 * k) + (6 * k) ^ 3 ≡ 0 [MOD 7]) :=
sorry

end NUMINAMATH_GPT_num_divisible_by_7_in_range_l2126_212688


namespace NUMINAMATH_GPT_distance_between_parallel_sides_l2126_212614

-- Define the givens
def length_side_a : ℝ := 24  -- length of one parallel side
def length_side_b : ℝ := 14  -- length of the other parallel side
def area_trapezium : ℝ := 342  -- area of the trapezium

-- We need to prove that the distance between parallel sides (h) is 18 cm
theorem distance_between_parallel_sides (h : ℝ)
  (H1 :  area_trapezium = (1/2) * (length_side_a + length_side_b) * h) :
  h = 18 :=
by sorry

end NUMINAMATH_GPT_distance_between_parallel_sides_l2126_212614


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l2126_212612

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : ∀ n, S n / a n = (n + 1) / 2) :
  (a 2 / a 3 = 2 / 3) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l2126_212612


namespace NUMINAMATH_GPT_carlos_wins_one_game_l2126_212602

def games_Won_Laura : ℕ := 5
def games_Lost_Laura : ℕ := 4
def games_Won_Mike : ℕ := 7
def games_Lost_Mike : ℕ := 2
def games_Lost_Carlos : ℕ := 5
variable (C : ℕ) -- Carlos's wins

theorem carlos_wins_one_game :
  games_Won_Laura + games_Won_Mike + C = (games_Won_Laura + games_Lost_Laura + games_Won_Mike + games_Lost_Mike + C + games_Lost_Carlos) / 2 →
  C = 1 :=
by
  sorry

end NUMINAMATH_GPT_carlos_wins_one_game_l2126_212602


namespace NUMINAMATH_GPT_find_x_value_l2126_212666

theorem find_x_value (x : ℝ) (h : 0.65 * x = 0.20 * 552.50) : x = 170 :=
sorry

end NUMINAMATH_GPT_find_x_value_l2126_212666


namespace NUMINAMATH_GPT_power_evaluation_l2126_212647

theorem power_evaluation (x : ℕ) (h1 : 3^x = 81) : 3^(x+2) = 729 := by
  sorry

end NUMINAMATH_GPT_power_evaluation_l2126_212647


namespace NUMINAMATH_GPT_sequence_properties_l2126_212672

variable {Seq : Nat → ℕ}
-- Given conditions: Sn = an(an + 3) / 6
def Sn (n : ℕ) := Seq n * (Seq n + 3) / 6

theorem sequence_properties :
  (Seq 1 = 3) ∧ (Seq 2 = 9) ∧ (∀ n : ℕ, Seq (n+1) = 3 * (n + 1)) :=
by 
  have h1 : Sn 1 = (Seq 1 * (Seq 1 + 3)) / 6 := rfl
  have h2 : Sn 2 = (Seq 2 * (Seq 2 + 3)) / 6 := rfl
  sorry

end NUMINAMATH_GPT_sequence_properties_l2126_212672


namespace NUMINAMATH_GPT_dry_mixed_fruits_weight_l2126_212692

theorem dry_mixed_fruits_weight :
  ∀ (fresh_grapes_weight fresh_apples_weight : ℕ)
    (grapes_water_content fresh_grapes_dry_matter_perc : ℕ)
    (apples_water_content fresh_apples_dry_matter_perc : ℕ),
    fresh_grapes_weight = 400 →
    fresh_apples_weight = 300 →
    grapes_water_content = 65 →
    fresh_grapes_dry_matter_perc = 35 →
    apples_water_content = 84 →
    fresh_apples_dry_matter_perc = 16 →
    (fresh_grapes_weight * fresh_grapes_dry_matter_perc / 100) +
    (fresh_apples_weight * fresh_apples_dry_matter_perc / 100) = 188 := by
  sorry

end NUMINAMATH_GPT_dry_mixed_fruits_weight_l2126_212692


namespace NUMINAMATH_GPT_pow_div_pow_l2126_212611

variable (a : ℝ)
variable (A B : ℕ)

theorem pow_div_pow (a : ℝ) (A B : ℕ) : a^A / a^B = a^(A - B) :=
  sorry

example : a^6 / a^2 = a^4 :=
  pow_div_pow a 6 2

end NUMINAMATH_GPT_pow_div_pow_l2126_212611
