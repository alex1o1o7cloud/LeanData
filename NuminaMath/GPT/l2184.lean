import Mathlib

namespace NUMINAMATH_GPT_range_of_m_l2184_218481

noncomputable def has_two_solutions (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 = x₁ + m ∧ x₂^2 = x₂ + m 

theorem range_of_m (m : ℝ) : has_two_solutions m ↔ m > -(1/4) :=
sorry

end NUMINAMATH_GPT_range_of_m_l2184_218481


namespace NUMINAMATH_GPT_prob_not_negative_review_A_prob_two_positive_reviews_choose_platform_A_l2184_218457

-- Definitions of the problem conditions
def positive_reviews_A := 75
def neutral_reviews_A := 20
def negative_reviews_A := 5
def total_reviews_A := 100

def positive_reviews_B := 64
def neutral_reviews_B := 8
def negative_reviews_B := 8
def total_reviews_B := 80

-- Prove the probability that a buyer's evaluation on platform A is not a negative review
theorem prob_not_negative_review_A : 
  (1 - negative_reviews_A / total_reviews_A) = 19 / 20 := by
  sorry

-- Prove the probability that exactly 2 out of 4 (2 from A and 2 from B) buyers give a positive review
theorem prob_two_positive_reviews :
  ((positive_reviews_A / total_reviews_A) ^ 2 * (1 - positive_reviews_B / total_reviews_B) ^ 2 + 
  2 * (positive_reviews_A / total_reviews_A) * (1 - positive_reviews_A / total_reviews_A) * 
  (positive_reviews_B / total_reviews_B) * (1 - positive_reviews_B / total_reviews_B) +
  (1 - positive_reviews_A / total_reviews_A) ^ 2 * (positive_reviews_B / total_reviews_B) ^ 2) = 
  73 / 400 := by
  sorry

-- Choose platform A based on the given data
theorem choose_platform_A :
  let E_A := (5 * 0.75 + 3 * 0.2 + 1 * 0.05)
  let D_A := (5 - E_A) ^ 2 * 0.75 + (3 - E_A) ^ 2 * 0.2 + (1 - E_A) ^ 2 * 0.05
  let E_B := (5 * 0.8 + 3 * 0.1 + 1 * 0.1)
  let D_B := (5 - E_B) ^ 2 * 0.8 + (3 - E_B) ^ 2 * 0.1 + (1 - E_B) ^ 2 * 0.1
  (E_A = E_B) ∧ (D_A < D_B) → choose_platform = "Platform A" := by
  sorry

end NUMINAMATH_GPT_prob_not_negative_review_A_prob_two_positive_reviews_choose_platform_A_l2184_218457


namespace NUMINAMATH_GPT_tangent_at_point_l2184_218445

theorem tangent_at_point (a b : ℝ) :
  (∀ x : ℝ, (x^3 - x^2 - a * x + b) = 2 * x + 1) →
  (a + b = -1) :=
by
  intro tangent_condition
  sorry

end NUMINAMATH_GPT_tangent_at_point_l2184_218445


namespace NUMINAMATH_GPT_cos_difference_of_angles_l2184_218483

theorem cos_difference_of_angles (α β : ℝ) 
    (h1 : Real.cos (α + β) = 1 / 5) 
    (h2 : Real.tan α * Real.tan β = 1 / 2) : 
    Real.cos (α - β) = 3 / 5 := 
sorry

end NUMINAMATH_GPT_cos_difference_of_angles_l2184_218483


namespace NUMINAMATH_GPT_number_of_quarters_l2184_218404
-- Definitions of the coin values
def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def half_dollar_value := 50

-- Number of each type of coin used in the proof
variable (pennies nickels dimes quarters half_dollars : ℕ)

-- Conditions from step (a)
axiom one_penny : pennies > 0
axiom one_nickel : nickels > 0
axiom one_dime : dimes > 0
axiom one_quarter : quarters > 0
axiom one_half_dollar : half_dollars > 0
axiom total_coins : pennies + nickels + dimes + quarters + half_dollars = 11
axiom total_value : pennies * penny_value + nickels * nickel_value + dimes * dime_value + quarters * quarter_value + half_dollars * half_dollar_value = 163

-- The conclusion we want to prove
theorem number_of_quarters : quarters = 1 := 
sorry

end NUMINAMATH_GPT_number_of_quarters_l2184_218404


namespace NUMINAMATH_GPT_find_h2_l2184_218449

noncomputable def h (x : ℝ) : ℝ := 
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) - 1) / (x^15 - 1)

theorem find_h2 : h 2 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_h2_l2184_218449


namespace NUMINAMATH_GPT_arithmetic_sequence_S11_l2184_218443

theorem arithmetic_sequence_S11 (a1 d : ℝ) 
  (h1 : a1 + d + a1 + 3 * d + 3 * (a1 + 6 * d) + a1 + 8 * d = 24) : 
  let a2 := a1 + d
  let a4 := a1 + 3 * d
  let a7 := a1 + 6 * d
  let a9 := a1 + 8 * d
  let S11 := 11 * (a1 + 5 * d)
  S11 = 44 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_S11_l2184_218443


namespace NUMINAMATH_GPT_cleared_land_with_corn_is_630_acres_l2184_218476

-- Definitions based on given conditions
def total_land : ℝ := 6999.999999999999
def cleared_fraction : ℝ := 0.90
def potato_fraction : ℝ := 0.20
def tomato_fraction : ℝ := 0.70

-- Calculate the cleared land
def cleared_land : ℝ := cleared_fraction * total_land

-- Calculate the land used for potato and tomato
def potato_land : ℝ := potato_fraction * cleared_land
def tomato_land : ℝ := tomato_fraction * cleared_land

-- Define the land planted with corn
def corn_land : ℝ := cleared_land - (potato_land + tomato_land)

-- The theorem to be proved
theorem cleared_land_with_corn_is_630_acres : corn_land = 630 := by
  sorry

end NUMINAMATH_GPT_cleared_land_with_corn_is_630_acres_l2184_218476


namespace NUMINAMATH_GPT_age_difference_l2184_218424

theorem age_difference {A B C : ℕ} (h : A + B = B + C + 15) : A - C = 15 := 
by 
  sorry

end NUMINAMATH_GPT_age_difference_l2184_218424


namespace NUMINAMATH_GPT_range_S13_over_a14_l2184_218437

lemma a_n_is_arithmetic_progression (a S : ℕ → ℕ) (h1 : ∀ n, a n > 1)
  (h2 : ∀ n, S (n + 1) - S n - (1 / 2) * ((a (n + 1)) ^ 2 - (a n) ^ 2) = 1 / 2) :
  ∀ n, a (n + 1) = a n + 1 := 
sorry

theorem range_S13_over_a14 (a S : ℕ → ℕ) (h1 : ∀ n, a n > 1)
  (h2 : ∀ n, S (n + 1) - S n - (1 / 2) * ((a (n + 1)) ^ 2 - (a n) ^ 2) = 1 / 2)
  (h3 : a 1 > 4) :
  130 / 17 < (S 13 / a 14) ∧ (S 13 / a 14) < 13 := 
sorry

end NUMINAMATH_GPT_range_S13_over_a14_l2184_218437


namespace NUMINAMATH_GPT_population_increase_rate_correct_l2184_218446

variable (P0 P1 : ℕ)
variable (r : ℚ)

-- Given conditions
def initial_population := P0 = 200
def population_after_one_year := P1 = 220

-- Proof problem statement
theorem population_increase_rate_correct :
  initial_population P0 →
  population_after_one_year P1 →
  r = (P1 - P0 : ℚ) / P0 * 100 →
  r = 10 :=
by
  sorry

end NUMINAMATH_GPT_population_increase_rate_correct_l2184_218446


namespace NUMINAMATH_GPT_total_population_eq_51b_over_40_l2184_218452

variable (b g t : Nat)

-- Conditions
def boys_eq_four_times_girls (b g : Nat) : Prop := b = 4 * g
def girls_eq_ten_times_teachers (g t : Nat) : Prop := g = 10 * t

-- Statement to prove
theorem total_population_eq_51b_over_40 (b g t : Nat) 
  (h1 : boys_eq_four_times_girls b g) 
  (h2 : girls_eq_ten_times_teachers g t) : 
  b + g + t = (51 * b) / 40 := 
sorry

end NUMINAMATH_GPT_total_population_eq_51b_over_40_l2184_218452


namespace NUMINAMATH_GPT_seashells_count_l2184_218492

theorem seashells_count (mary_seashells : ℕ) (keith_seashells : ℕ) (cracked_seashells : ℕ) 
  (h_mary : mary_seashells = 2) (h_keith : keith_seashells = 5) (h_cracked : cracked_seashells = 9) :
  (mary_seashells + keith_seashells = 7) ∧ (cracked_seashells > mary_seashells + keith_seashells) → false := 
by {
  sorry
}

end NUMINAMATH_GPT_seashells_count_l2184_218492


namespace NUMINAMATH_GPT_engineer_progress_l2184_218465

theorem engineer_progress (x : ℕ) : 
  ∀ (road_length_in_km : ℝ) 
    (total_days : ℕ) 
    (initial_men : ℕ) 
    (completed_work_in_km : ℝ) 
    (additional_men : ℕ) 
    (new_total_men : ℕ) 
    (remaining_work_in_km : ℝ) 
    (remaining_days : ℕ),
    road_length_in_km = 10 → 
    total_days = 300 → 
    initial_men = 30 → 
    completed_work_in_km = 2 → 
    additional_men = 30 → 
    new_total_men = 60 → 
    remaining_work_in_km = 8 → 
    remaining_days = total_days - x →
  (4 * (total_days - x) = 8 * x) →
  x = 100 :=
by
  intros road_length_in_km total_days initial_men completed_work_in_km additional_men new_total_men remaining_work_in_km remaining_days
  intros h1 h2 h3 h4 h5 h6 h7 h8 h_eqn
  -- Proof
  sorry

end NUMINAMATH_GPT_engineer_progress_l2184_218465


namespace NUMINAMATH_GPT_other_endpoint_coordinates_sum_l2184_218456

noncomputable def other_endpoint_sum (x1 y1 x2 y2 xm ym : ℝ) : ℝ :=
  let x := 2 * xm - x1
  let y := 2 * ym - y1
  x + y

theorem other_endpoint_coordinates_sum :
  (other_endpoint_sum 6 (-2) 0 12 3 5) = 12 := by
  sorry

end NUMINAMATH_GPT_other_endpoint_coordinates_sum_l2184_218456


namespace NUMINAMATH_GPT_quarters_needed_l2184_218403

-- Define the cost of items in cents and declare the number of items to purchase.
def quarter_value : ℕ := 25
def candy_bar_cost : ℕ := 25
def chocolate_cost : ℕ := 75
def juice_cost : ℕ := 50

def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Theorem stating the number of quarters needed to buy the given items.
theorem quarters_needed : 
  (num_candy_bars * candy_bar_cost + num_chocolates * chocolate_cost + num_juice_packs * juice_cost) / quarter_value = 11 := 
sorry

end NUMINAMATH_GPT_quarters_needed_l2184_218403


namespace NUMINAMATH_GPT_prob_students_on_both_days_l2184_218460
noncomputable def probability_event_on_both_days: ℚ := by
  let total_days := 2
  let total_students := 4
  let prob_single_day := (1 / total_days : ℚ) ^ total_students
  let prob_all_same_day := 2 * prob_single_day
  let prob_both_days := 1 - prob_all_same_day
  exact prob_both_days

theorem prob_students_on_both_days : probability_event_on_both_days = 7 / 8 :=
by
  exact sorry

end NUMINAMATH_GPT_prob_students_on_both_days_l2184_218460


namespace NUMINAMATH_GPT_digits_sum_not_2001_l2184_218438

theorem digits_sum_not_2001 (a : ℕ) (n m : ℕ) 
  (h1 : 10^(n-1) ≤ a ∧ a < 10^n)
  (h2 : 3 * n - 2 ≤ m ∧ m < 3 * n + 1)
  : m + n ≠ 2001 := 
sorry

end NUMINAMATH_GPT_digits_sum_not_2001_l2184_218438


namespace NUMINAMATH_GPT_fraction_ordering_l2184_218421

noncomputable def t1 : ℝ := (100^100 + 1) / (100^90 + 1)
noncomputable def t2 : ℝ := (100^99 + 1) / (100^89 + 1)
noncomputable def t3 : ℝ := (100^101 + 1) / (100^91 + 1)
noncomputable def t4 : ℝ := (101^101 + 1) / (101^91 + 1)
noncomputable def t5 : ℝ := (101^100 + 1) / (101^90 + 1)
noncomputable def t6 : ℝ := (99^99 + 1) / (99^89 + 1)
noncomputable def t7 : ℝ := (99^100 + 1) / (99^90 + 1)

theorem fraction_ordering : t6 < t7 ∧ t7 < t2 ∧ t2 < t1 ∧ t1 < t3 ∧ t3 < t5 ∧ t5 < t4 := by
  sorry

end NUMINAMATH_GPT_fraction_ordering_l2184_218421


namespace NUMINAMATH_GPT_min_neg_signs_to_zero_sum_l2184_218499

-- Definition of the set of numbers on the clock face
def clock_face_numbers : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Sum of the clock face numbers
def sum_clock_face_numbers := clock_face_numbers.sum

-- Given condition that the sum of clock face numbers is 78
axiom sum_clock_face_numbers_is_78 : sum_clock_face_numbers = 78

-- Definition of the function to calculate the minimum number of negative signs needed
def min_neg_signs_needed (numbers : List ℤ) (target : ℤ) : ℕ :=
  sorry -- The implementation is omitted

-- Theorem stating the goal of our problem
theorem min_neg_signs_to_zero_sum : min_neg_signs_needed clock_face_numbers 39 = 4 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_min_neg_signs_to_zero_sum_l2184_218499


namespace NUMINAMATH_GPT_proof_problem_l2184_218495

noncomputable def problem_statement : Prop :=
  let p1 := ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0
  let p2 := ∀ x y : ℝ, x + y > 2 → x > 1 ∧ y > 1
  let p3 := ∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3
  let p4 := ∀ a b c : ℝ, a ≠ 0 ∧ b^2 - 4 * a * c > 0 → ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0
  p3 = true ∧ p1 = false ∧ p2 = false ∧ p4 = false

theorem proof_problem : problem_statement := 
sorry

end NUMINAMATH_GPT_proof_problem_l2184_218495


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_divisible_by_p_l2184_218494

theorem arithmetic_sequence_common_difference_divisible_by_p 
  (n : ℕ) (a : ℕ → ℕ) (h1 : n ≥ 2021) (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) 
  (h3 : a 1 > 2021) (h4 : ∀ i, 1 ≤ i → i ≤ n → Nat.Prime (a i)) : 
  ∀ p, Nat.Prime p → p < 2021 → ∃ d, (∀ m, 2 ≤ m → a m = a 1 + (m - 1) * d) ∧ p ∣ d := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_divisible_by_p_l2184_218494


namespace NUMINAMATH_GPT_length_of_ad_l2184_218436

theorem length_of_ad (AB CD AD BC : ℝ) 
  (h1 : AB = 10) 
  (h2 : CD = 2 * AB) 
  (h3 : AD = BC) 
  (h4 : AB + BC + CD + AD = 42) : AD = 6 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_length_of_ad_l2184_218436


namespace NUMINAMATH_GPT_parallelogram_area_twice_quadrilateral_l2184_218450

theorem parallelogram_area_twice_quadrilateral (a b : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π) :
  let quadrilateral_area := (1 / 2) * a * b * Real.sin α
  let parallelogram_area := a * b * Real.sin α
  parallelogram_area = 2 * quadrilateral_area :=
by
  let quadrilateral_area := (1 / 2) * a * b * Real.sin α
  let parallelogram_area := a * b * Real.sin α
  sorry

end NUMINAMATH_GPT_parallelogram_area_twice_quadrilateral_l2184_218450


namespace NUMINAMATH_GPT_arth_seq_val_a7_l2184_218484

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arth_seq_val_a7 {a : ℕ → ℝ} 
  (h_arith : arithmetic_sequence a)
  (h_positive : ∀ n : ℕ, 0 < a n)
  (h_eq : 2 * a 6 + 2 * a 8 = (a 7) ^ 2) :
  a 7 = 4 := 
by sorry

end NUMINAMATH_GPT_arth_seq_val_a7_l2184_218484


namespace NUMINAMATH_GPT_min_N_of_block_viewed_l2184_218418

theorem min_N_of_block_viewed (x y z N : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_factor : (x - 1) * (y - 1) * (z - 1) = 231) : 
  N = x * y * z ∧ N = 384 :=
by {
  sorry 
}

end NUMINAMATH_GPT_min_N_of_block_viewed_l2184_218418


namespace NUMINAMATH_GPT_decimal_to_binary_18_l2184_218442

theorem decimal_to_binary_18 : (18: ℕ) = 0b10010 := by
  sorry

end NUMINAMATH_GPT_decimal_to_binary_18_l2184_218442


namespace NUMINAMATH_GPT_product_divisible_by_10_probability_l2184_218416

noncomputable def probability_divisible_by_10 (n : ℕ) (h: n > 1) : ℝ :=
  1 - ((8^n + 5^n - 4^n : ℝ) / (9^n : ℝ))

theorem product_divisible_by_10_probability (n : ℕ) (h: n > 1) :
  probability_divisible_by_10 n h = 1 - ((8^n + 5^n - 4^n : ℝ) / (9^n : ℝ)) :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_product_divisible_by_10_probability_l2184_218416


namespace NUMINAMATH_GPT_housewife_spending_l2184_218462

theorem housewife_spending
    (R : ℝ) (P : ℝ) (M : ℝ)
    (h1 : R = 25)
    (h2 : R = 0.85 * P)
    (h3 : M / R - M / P = 3) :
  M = 450 :=
by
  sorry

end NUMINAMATH_GPT_housewife_spending_l2184_218462


namespace NUMINAMATH_GPT_factorization_of_polynomial_l2184_218427

theorem factorization_of_polynomial (x : ℂ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^2 - 3) * (x^4 + 3 * x^2 + 9) :=
by sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l2184_218427


namespace NUMINAMATH_GPT_right_triangle_and_mod_inverse_l2184_218417

theorem right_triangle_and_mod_inverse (a b c m : ℕ) (h1 : a = 48) (h2 : b = 55) (h3 : c = 73) (h4 : m = 4273) 
  (h5 : a^2 + b^2 = c^2) : ∃ x : ℕ, (480 * x) % m = 1 ∧ x = 1643 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_and_mod_inverse_l2184_218417


namespace NUMINAMATH_GPT_completing_the_square_l2184_218464

theorem completing_the_square (x : ℝ) : (x^2 - 6*x + 7 = 0) → ((x - 3)^2 = 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_completing_the_square_l2184_218464


namespace NUMINAMATH_GPT_maximum_value_of_linear_expression_l2184_218496

theorem maximum_value_of_linear_expression (m n : ℕ) (h_sum : (m*(m + 1) + n^2 = 1987)) : 3 * m + 4 * n ≤ 221 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_linear_expression_l2184_218496


namespace NUMINAMATH_GPT_number_of_cats_l2184_218466

-- Defining the context and conditions
variables (x y z : Nat)
variables (h1 : x + y + z = 29) (h2 : x = z)

-- Proving the number of cats
theorem number_of_cats (x y z : Nat) (h1 : x + y + z = 29) (h2 : x = z) :
  6 * x + 3 * y = 87 := by
  sorry

end NUMINAMATH_GPT_number_of_cats_l2184_218466


namespace NUMINAMATH_GPT_find_prime_pairs_l2184_218493

theorem find_prime_pairs (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : 0 < n) :
  p * (p + 1) + q * (q + 1) = n * (n + 1) ↔ (p = 3 ∧ q = 5 ∧ n = 6) ∨ (p = 5 ∧ q = 3 ∧ n = 6) ∨ (p = 2 ∧ q = 2 ∧ n = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_prime_pairs_l2184_218493


namespace NUMINAMATH_GPT_integer_values_abc_l2184_218458

theorem integer_values_abc {a b c : ℤ} :
  a^2 + b^2 + c^2 + 3 < a * b + 3 * b + 2 * c ↔ (a = 1 ∧ b = 2 ∧ c = 1) :=
by
  sorry -- Proof to be filled

end NUMINAMATH_GPT_integer_values_abc_l2184_218458


namespace NUMINAMATH_GPT_p_6_eq_163_l2184_218486

noncomputable def p (x : ℕ) : ℕ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + x^2 + x + 1

theorem p_6_eq_163 : p 6 = 163 :=
by
  sorry

end NUMINAMATH_GPT_p_6_eq_163_l2184_218486


namespace NUMINAMATH_GPT_margo_pairing_probability_l2184_218425

theorem margo_pairing_probability (students : Finset ℕ)
  (H_50_students : students.card = 50)
  (margo irma jess kurt : ℕ)
  (H_margo_in_students : margo ∈ students)
  (H_irma_in_students : irma ∈ students)
  (H_jess_in_students : jess ∈ students)
  (H_kurt_in_students : kurt ∈ students)
  (possible_partners : Finset ℕ := students.erase margo) :
  (3: ℝ) / 49 = ((3: ℝ) / (possible_partners.card: ℝ)) :=
by
  -- The actual steps of the proof will be here
  sorry

end NUMINAMATH_GPT_margo_pairing_probability_l2184_218425


namespace NUMINAMATH_GPT_mike_total_games_l2184_218478

theorem mike_total_games
  (non_working : ℕ)
  (price_per_game : ℕ)
  (total_earnings : ℕ)
  (h1 : non_working = 9)
  (h2 : price_per_game = 5)
  (h3 : total_earnings = 30) :
  non_working + (total_earnings / price_per_game) = 15 := 
by
  sorry

end NUMINAMATH_GPT_mike_total_games_l2184_218478


namespace NUMINAMATH_GPT_gift_wrapping_combinations_l2184_218441

theorem gift_wrapping_combinations :
    (10 * 3 * 4 * 5 = 600) :=
by
    sorry

end NUMINAMATH_GPT_gift_wrapping_combinations_l2184_218441


namespace NUMINAMATH_GPT_john_initial_running_time_l2184_218432

theorem john_initial_running_time (H : ℝ) (hH1 : 1.75 * H = 168 / 12)
: H = 8 :=
sorry

end NUMINAMATH_GPT_john_initial_running_time_l2184_218432


namespace NUMINAMATH_GPT_p2_div_q2_eq_4_l2184_218422

theorem p2_div_q2_eq_4 
  (p q : ℝ → ℝ)
  (h1 : ∀ x, p x = 12 * x)
  (h2 : ∀ x, q x = (x + 4) * (x - 1))
  (h3 : p 0 = 0)
  (h4 : p (-1) / q (-1) = -2) :
  (p 2 / q 2 = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_p2_div_q2_eq_4_l2184_218422


namespace NUMINAMATH_GPT_initial_men_count_l2184_218459

theorem initial_men_count 
  (M : ℕ)
  (h1 : 8 * M * 30 = (M + 77) * 6 * 50) :
  M = 63 :=
by
  sorry

end NUMINAMATH_GPT_initial_men_count_l2184_218459


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_sum_of_sequence_b_n_l2184_218415

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h₁ : a 2 = 3) 
  (h₂ : S 5 + a 3 = 30) 
  (h₃ : ∀ n, S n = (n * (a 1 + (n-1) * ((a 2) - (a 1)))) / 2 
                     ∧ a n = a 1 + (n-1) * ((a 2) - (a 1))) : 
  (∀ n, a n = 2 * n - 1 ∧ S n = n^2) := 
sorry

theorem sum_of_sequence_b_n (b : ℕ → ℝ) 
  (T : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h₁ : ∀ n, b n = (a (n+1)) / (S n * S (n+1))) 
  (h₂ : ∀ n, a n = 2 * n - 1 ∧ S n = n^2) : 
  (∀ n, T n = (1 - 1 / (n+1)^2)) := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_sum_of_sequence_b_n_l2184_218415


namespace NUMINAMATH_GPT_binomial_10_3_eq_120_l2184_218469

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_GPT_binomial_10_3_eq_120_l2184_218469


namespace NUMINAMATH_GPT_multiplication_difference_l2184_218401

theorem multiplication_difference :
  672 * 673 * 674 - 671 * 673 * 675 = 2019 := by
  sorry

end NUMINAMATH_GPT_multiplication_difference_l2184_218401


namespace NUMINAMATH_GPT_focus_of_parabola_l2184_218426

theorem focus_of_parabola : 
  ∃(h k : ℚ), ((∀ x : ℚ, -2 * x^2 - 6 * x + 1 = -2 * (x + 3 / 2)^2 + 11 / 2) ∧ 
  (∃ a : ℚ, (a = -2 / 8) ∧ (h = -3/2) ∧ (k = 11/2 + a)) ∧ 
  (h, k) = (-3/2, 43 / 8)) :=
sorry

end NUMINAMATH_GPT_focus_of_parabola_l2184_218426


namespace NUMINAMATH_GPT_original_volume_l2184_218470

theorem original_volume (V : ℝ) (h1 : V > 0) 
    (h2 : (1/16) * V = 0.75) : V = 12 :=
by sorry

end NUMINAMATH_GPT_original_volume_l2184_218470


namespace NUMINAMATH_GPT_min_n_for_constant_term_l2184_218489

theorem min_n_for_constant_term :
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, 3 * n = 5 * r) → ∃ n : ℕ, n = 5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_min_n_for_constant_term_l2184_218489


namespace NUMINAMATH_GPT_derivative_f_at_1_l2184_218419

noncomputable def f (x : Real) : Real := x^3 * Real.sin x

theorem derivative_f_at_1 : deriv f 1 = 3 * Real.sin 1 + Real.cos 1 := by
  sorry

end NUMINAMATH_GPT_derivative_f_at_1_l2184_218419


namespace NUMINAMATH_GPT_simplify_expr_1_simplify_expr_2_l2184_218413

theorem simplify_expr_1 (x y : ℝ) :
  12 * x - 6 * y + 3 * y - 24 * x = -12 * x - 3 * y :=
by
  sorry

theorem simplify_expr_2 (a b : ℝ) :
  (3 / 2) * (a^2 * b - 2 * (a * b^2)) - (1 / 2) * (a * b^2 - 4 * (a^2 * b)) + (a * b^2) / 2 = (7 / 2) * (a^2 * b) - 3 * (a * b^2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_1_simplify_expr_2_l2184_218413


namespace NUMINAMATH_GPT_jana_walking_distance_l2184_218447

theorem jana_walking_distance (t_walk_mile : ℝ) (speed : ℝ) (time : ℝ) (distance : ℝ) :
  t_walk_mile = 24 → speed = 1 / t_walk_mile → time = 36 → distance = speed * time → distance = 1.5 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jana_walking_distance_l2184_218447


namespace NUMINAMATH_GPT_tissue_actual_diameter_l2184_218412

theorem tissue_actual_diameter (magnification_factor : ℝ) (magnified_diameter : ℝ) 
(h1 : magnification_factor = 1000)
(h2 : magnified_diameter = 0.3) : 
  magnified_diameter / magnification_factor = 0.0003 :=
by sorry

end NUMINAMATH_GPT_tissue_actual_diameter_l2184_218412


namespace NUMINAMATH_GPT_birds_on_the_fence_l2184_218435

theorem birds_on_the_fence (x : ℕ) : 10 + 2 * x = 50 → x = 20 := by
  sorry

end NUMINAMATH_GPT_birds_on_the_fence_l2184_218435


namespace NUMINAMATH_GPT_digging_project_length_l2184_218434

theorem digging_project_length (Length_2 : ℝ) : 
  (100 * 25 * 30) = (75 * Length_2 * 50) → 
  Length_2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_digging_project_length_l2184_218434


namespace NUMINAMATH_GPT_fruit_selling_price_3640_l2184_218400

def cost_price := 22
def initial_selling_price := 38
def initial_quantity_sold := 160
def price_reduction := 3
def quantity_increase := 120
def target_profit := 3640

theorem fruit_selling_price_3640 (x : ℝ) :
  ((initial_selling_price - x - cost_price) * (initial_quantity_sold + (x / price_reduction) * quantity_increase) = target_profit) →
  x = 9 →
  initial_selling_price - x = 29 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_fruit_selling_price_3640_l2184_218400


namespace NUMINAMATH_GPT_exactly_one_first_class_probability_at_least_one_second_class_probability_l2184_218407

-- Definitions based on the problem statement:
def total_pens : ℕ := 6
def first_class_pens : ℕ := 4
def second_class_pens : ℕ := 2

def total_draws : ℕ := 2

-- Event for drawing exactly one first-class quality pen
def probability_one_first_class := ((first_class_pens.choose 1 * second_class_pens.choose 1) /
                                    (total_pens.choose total_draws) : ℚ)

-- Event for drawing at least one second-class quality pen
def probability_at_least_one_second_class := (1 - (first_class_pens.choose total_draws /
                                                   total_pens.choose total_draws) : ℚ)

-- Statements to prove the probabilities
theorem exactly_one_first_class_probability :
  probability_one_first_class = 8 / 15 :=
sorry

theorem at_least_one_second_class_probability :
  probability_at_least_one_second_class = 3 / 5 :=
sorry

end NUMINAMATH_GPT_exactly_one_first_class_probability_at_least_one_second_class_probability_l2184_218407


namespace NUMINAMATH_GPT_horses_for_camels_l2184_218487

noncomputable def cost_of_one_elephant : ℕ := 11000
noncomputable def cost_of_one_ox : ℕ := 7333 -- approx.
noncomputable def cost_of_one_horse : ℕ := 1833 -- approx.
noncomputable def cost_of_one_camel : ℕ := 4400

theorem horses_for_camels (H : ℕ) :
  (H * cost_of_one_horse = cost_of_one_camel) → H = 2 :=
by
  -- skipping proof details
  sorry

end NUMINAMATH_GPT_horses_for_camels_l2184_218487


namespace NUMINAMATH_GPT_renovation_costs_l2184_218406

theorem renovation_costs :
  ∃ (x y : ℝ), 
    8 * x + 8 * y = 3520 ∧
    6 * x + 12 * y = 3480 ∧
    x = 300 ∧
    y = 140 ∧
    300 * 12 > 140 * 24 :=
by sorry

end NUMINAMATH_GPT_renovation_costs_l2184_218406


namespace NUMINAMATH_GPT_spending_limit_l2184_218409

variable (n b total_spent limit: ℕ)

theorem spending_limit (hne: n = 34) (hbe: b = n + 5) (hts: total_spent = n + b) (hlo: total_spent = limit + 3) : limit = 70 := by
  sorry

end NUMINAMATH_GPT_spending_limit_l2184_218409


namespace NUMINAMATH_GPT_exists_multiple_with_all_digits_l2184_218498

theorem exists_multiple_with_all_digits (n : ℕ) :
  ∃ m : ℕ, (m % n = 0) ∧ (∀ d : ℕ, d < 10 → d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9) := 
sorry

end NUMINAMATH_GPT_exists_multiple_with_all_digits_l2184_218498


namespace NUMINAMATH_GPT_Linda_original_savings_l2184_218468

theorem Linda_original_savings (S : ℝ)
  (H1 : 3/4 * S + 1/4 * S = S)
  (H2 : 1/4 * S = 220) :
  S = 880 :=
sorry

end NUMINAMATH_GPT_Linda_original_savings_l2184_218468


namespace NUMINAMATH_GPT_correct_statement_l2184_218440

def degree (term : String) : ℕ :=
  if term = "1/2πx^2" then 2
  else if term = "-4x^2y" then 3
  else 0

def coefficient (term : String) : ℤ :=
  if term = "-4x^2y" then -4
  else if term = "3(x+y)" then 3
  else 0

def is_monomial (term : String) : Bool :=
  if term = "8" then true
  else false

theorem correct_statement : 
  (degree "1/2πx^2" ≠ 3) ∧ 
  (coefficient "-4x^2y" ≠ 4) ∧ 
  (is_monomial "8" = true) ∧ 
  (coefficient "3(x+y)" ≠ 3) := 
by
  sorry

end NUMINAMATH_GPT_correct_statement_l2184_218440


namespace NUMINAMATH_GPT_mass_of_CaSO4_formed_correct_l2184_218402

noncomputable def mass_CaSO4_formed 
(mass_CaO : ℝ) (mass_H2SO4 : ℝ)
(molar_mass_CaO : ℝ) (molar_mass_H2SO4 : ℝ) (molar_mass_CaSO4 : ℝ) : ℝ :=
  let moles_CaO := mass_CaO / molar_mass_CaO
  let moles_H2SO4 := mass_H2SO4 / molar_mass_H2SO4
  let limiting_reactant_moles := min moles_CaO moles_H2SO4
  limiting_reactant_moles * molar_mass_CaSO4

theorem mass_of_CaSO4_formed_correct :
  mass_CaSO4_formed 25 35 56.08 98.09 136.15 = 48.57 :=
by
  rw [mass_CaSO4_formed]
  sorry

end NUMINAMATH_GPT_mass_of_CaSO4_formed_correct_l2184_218402


namespace NUMINAMATH_GPT_layla_more_points_than_nahima_l2184_218491

theorem layla_more_points_than_nahima (layla_points : ℕ) (total_points : ℕ) (h1 : layla_points = 70) (h2 : total_points = 112) :
  layla_points - (total_points - layla_points) = 28 :=
by
  sorry

end NUMINAMATH_GPT_layla_more_points_than_nahima_l2184_218491


namespace NUMINAMATH_GPT_abs_inequality_solution_l2184_218431

theorem abs_inequality_solution :
  {x : ℝ | |x + 2| > 3} = {x : ℝ | x < -5} ∪ {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l2184_218431


namespace NUMINAMATH_GPT_remainder_of_31_pow_31_plus_31_div_32_l2184_218410

theorem remainder_of_31_pow_31_plus_31_div_32 :
  (31^31 + 31) % 32 = 30 := 
by 
  trivial -- Replace with actual proof

end NUMINAMATH_GPT_remainder_of_31_pow_31_plus_31_div_32_l2184_218410


namespace NUMINAMATH_GPT_solution_set_l2184_218429

open Real

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the differentiable function f

axiom differentiable_f : Differentiable ℝ f
axiom condition_f : ∀ x, f x > 0 ∧ x * (deriv (deriv (deriv f))) x > 0

theorem solution_set :
  {x : ℝ | 1 ≤ x ∧ x < 2} =
    {x : ℝ | f (sqrt (x + 1)) > sqrt (x - 1) * f (sqrt (x ^ 2 - 1))} :=
sorry

end NUMINAMATH_GPT_solution_set_l2184_218429


namespace NUMINAMATH_GPT_paused_time_l2184_218473

theorem paused_time (total_length remaining_length paused_at : ℕ) (h1 : total_length = 60) (h2 : remaining_length = 30) : paused_at = total_length - remaining_length :=
by
  sorry

end NUMINAMATH_GPT_paused_time_l2184_218473


namespace NUMINAMATH_GPT_kendra_change_and_discounts_l2184_218408

-- Define the constants and conditions
def wooden_toy_price : ℝ := 20.0
def hat_price : ℝ := 10.0
def tax_rate : ℝ := 0.08
def discount_wooden_toys_2_3 : ℝ := 0.10
def discount_wooden_toys_4_or_more : ℝ := 0.15
def discount_hats_2 : ℝ := 0.05
def discount_hats_3_or_more : ℝ := 0.10
def kendra_bill : ℝ := 250.0
def kendra_wooden_toys : ℕ := 4
def kendra_hats : ℕ := 5

-- Calculate the applicable discounts based on conditions
def discount_on_wooden_toys : ℝ :=
  if kendra_wooden_toys >= 2 ∧ kendra_wooden_toys <= 3 then
    discount_wooden_toys_2_3
  else if kendra_wooden_toys >= 4 then
    discount_wooden_toys_4_or_more
  else
    0.0

def discount_on_hats : ℝ :=
  if kendra_hats = 2 then
    discount_hats_2
  else if kendra_hats >= 3 then
    discount_hats_3_or_more
  else
    0.0

-- Main theorem statement
theorem kendra_change_and_discounts :
  let total_cost_before_discounts := kendra_wooden_toys * wooden_toy_price + kendra_hats * hat_price
  let wooden_toys_discount := discount_on_wooden_toys * (kendra_wooden_toys * wooden_toy_price)
  let hats_discount := discount_on_hats * (kendra_hats * hat_price)
  let total_discounts := wooden_toys_discount + hats_discount
  let total_cost_after_discounts := total_cost_before_discounts - total_discounts
  let tax := tax_rate * total_cost_after_discounts
  let total_cost_after_tax := total_cost_after_discounts + tax
  let change_received := kendra_bill - total_cost_after_tax
  (total_discounts = 17) → 
  (change_received = 127.96) ∧ 
  (wooden_toys_discount = 12) ∧ 
  (hats_discount = 5) :=
by
  sorry

end NUMINAMATH_GPT_kendra_change_and_discounts_l2184_218408


namespace NUMINAMATH_GPT_Marcus_pretzels_l2184_218467

theorem Marcus_pretzels (John_pretzels : ℕ) (Marcus_more_than_John : ℕ) (h1 : John_pretzels = 28) (h2 : Marcus_more_than_John = 12) : Marcus_more_than_John + John_pretzels = 40 :=
by
  sorry

end NUMINAMATH_GPT_Marcus_pretzels_l2184_218467


namespace NUMINAMATH_GPT_trigonometric_identity_l2184_218472

open Real 

theorem trigonometric_identity (x y : ℝ) (h₁ : P = x * cos y) (h₂ : Q = x * sin y) : 
  (P + Q) / (P - Q) + (P - Q) / (P + Q) = 2 * cos y / sin y := by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2184_218472


namespace NUMINAMATH_GPT_problem1_problem2_l2184_218414

-- Proof problem 1 statement in Lean 4
theorem problem1 :
  (1 : ℝ) * (Real.sqrt 2)^2 - |(1 : ℝ) - Real.sqrt 3| + Real.sqrt ((-3 : ℝ)^2) + Real.sqrt 81 = 15 - Real.sqrt 3 :=
by sorry

-- Proof problem 2 statement in Lean 4
theorem problem2 (x y : ℝ) :
  (x - 2 * y)^2 - (x + 2 * y + 3) * (x + 2 * y - 3) = -8 * x * y + 9 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2184_218414


namespace NUMINAMATH_GPT_find_factor_l2184_218463

variable (x : ℕ) (f : ℕ)

def original_number := x = 20
def resultant := f * (2 * x + 5) = 135

theorem find_factor (h1 : original_number x) (h2 : resultant x f) : f = 3 := by
  sorry

end NUMINAMATH_GPT_find_factor_l2184_218463


namespace NUMINAMATH_GPT_intersection_A_B_range_of_m_l2184_218485

-- Step 1: Define sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

def B : Set ℝ := {x | -1 < x ∧ x < 3}

def C (m : ℝ) : Set ℝ := {x | m < x ∧ x < 2 * m - 1}

-- Step 2: Lean statements for the proof

-- (1) Prove A ∩ B = {x | 1 < x < 3}
theorem intersection_A_B : (A ∩ B) = {x | 1 < x ∧ x < 3} :=
by
  sorry

-- (2) Prove the range of m such that C ∪ B = B is (-∞, 2]
theorem range_of_m (m : ℝ) : (C m ∪ B = B) ↔ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_range_of_m_l2184_218485


namespace NUMINAMATH_GPT_farmer_land_area_l2184_218475

-- Variables representing the total land, and the percentages and areas.
variable {T : ℝ} (h_cleared : 0.85 * T =  V) (V_10_percent : 0.10 * V + 0.70 * V + 0.05 * V + 500 = V)
variable {total_acres : ℝ} (correct_total_acres : total_acres = 3921.57)

theorem farmer_land_area (h_cleared : 0.85 * T = V) (h_planted : 0.85 * V = 500) : T = 3921.57 :=
by
  sorry

end NUMINAMATH_GPT_farmer_land_area_l2184_218475


namespace NUMINAMATH_GPT_intersection_complement_A_B_l2184_218490

open Set

variable (x : ℝ)

def U := ℝ
def A := {x | -2 ≤ x ∧ x ≤ 3}
def B := {x | x < -1 ∨ x > 4}

theorem intersection_complement_A_B :
  {x | -2 ≤ x ∧ x ≤ 3} ∩ compl {x | x < -1 ∨ x > 4} = {x | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_A_B_l2184_218490


namespace NUMINAMATH_GPT_worm_length_difference_l2184_218480

def worm_1_length : ℝ := 0.8
def worm_2_length : ℝ := 0.1
def difference := worm_1_length - worm_2_length

theorem worm_length_difference : difference = 0.7 := by
  sorry

end NUMINAMATH_GPT_worm_length_difference_l2184_218480


namespace NUMINAMATH_GPT_carpet_dimensions_problem_l2184_218474

def carpet_dimensions (width1 width2 : ℕ) (l : ℕ) :=
  ∃ x y : ℕ, width1 = 38 ∧ width2 = 50 ∧ l = l ∧ x = 25 ∧ y = 50

theorem carpet_dimensions_problem (l : ℕ) :
  carpet_dimensions 38 50 l :=
by
  sorry

end NUMINAMATH_GPT_carpet_dimensions_problem_l2184_218474


namespace NUMINAMATH_GPT_surface_area_is_33_l2184_218488

structure TShape where
  vertical_cubes : ℕ -- Number of cubes in the vertical line
  horizontal_cubes : ℕ -- Number of cubes in the horizontal line
  intersection_point : ℕ -- Intersection point in the vertical line
  
def surface_area (t : TShape) : ℕ :=
  let top_and_bottom := 9 + 9
  let side_vertical := (3 + 4) -- 3 for the top cube, 1 each for the other 4 cubes
  let side_horizontal := (4 - 1) * 2 -- each of 4 left and right minus intersection twice
  let intersection := 2
  top_and_bottom + side_vertical + side_horizontal + intersection

theorem surface_area_is_33 (t : TShape) (h1 : t.vertical_cubes = 5) (h2 : t.horizontal_cubes = 5) (h3 : t.intersection_point = 3) : 
  surface_area t = 33 := by
  sorry

end NUMINAMATH_GPT_surface_area_is_33_l2184_218488


namespace NUMINAMATH_GPT_sin_cos_alpha_beta_l2184_218451

theorem sin_cos_alpha_beta (α β : ℝ) 
  (h1 : Real.sin α = Real.cos β) 
  (h2 : Real.cos α = Real.sin (2 * β)) :
  Real.sin β ^ 2 + Real.cos α ^ 2 = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_alpha_beta_l2184_218451


namespace NUMINAMATH_GPT_lights_on_top_layer_l2184_218430

theorem lights_on_top_layer
  (x : ℕ)
  (H1 : x + 2 * x + 4 * x + 8 * x + 16 * x + 32 * x + 64 * x = 381) :
  x = 3 :=
  sorry

end NUMINAMATH_GPT_lights_on_top_layer_l2184_218430


namespace NUMINAMATH_GPT_jackson_holidays_l2184_218420

theorem jackson_holidays (holidays_per_month : ℕ) (months_in_year : ℕ) (holidays_per_year : ℕ) : 
  holidays_per_month = 3 → months_in_year = 12 → holidays_per_year = holidays_per_month * months_in_year → holidays_per_year = 36 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_jackson_holidays_l2184_218420


namespace NUMINAMATH_GPT_tangent_lines_count_l2184_218428

def f (x : ℝ) : ℝ := x^3

theorem tangent_lines_count :
  (∃ x : ℝ, deriv f x = 3) ∧ 
  (∃ y : ℝ, deriv f y = 3 ∧ y ≠ x) := 
by
  -- Since f(x) = x^3, its derivative is f'(x) = 3x^2
  -- We need to solve 3x^2 = 3
  -- Therefore, x^2 = 1 and x = ±1
  -- Thus, there are two tangent lines
  sorry

end NUMINAMATH_GPT_tangent_lines_count_l2184_218428


namespace NUMINAMATH_GPT_f_of_f_neg1_eq_neg1_f_x_eq_neg1_iff_x_eq_0_or_2_l2184_218439

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 1 else -x + 1

-- Prove f[f(-1)] = -1
theorem f_of_f_neg1_eq_neg1 : f (f (-1)) = -1 := sorry

-- Prove that if f(x) = -1, then x = 0 or x = 2
theorem f_x_eq_neg1_iff_x_eq_0_or_2 (x : ℝ) : f x = -1 ↔ x = 0 ∨ x = 2 := sorry

end NUMINAMATH_GPT_f_of_f_neg1_eq_neg1_f_x_eq_neg1_iff_x_eq_0_or_2_l2184_218439


namespace NUMINAMATH_GPT_yard_length_eq_250_l2184_218454

noncomputable def number_of_trees : ℕ := 26
noncomputable def distance_between_trees : ℕ := 10
noncomputable def number_of_gaps := number_of_trees - 1
noncomputable def length_of_yard := number_of_gaps * distance_between_trees

theorem yard_length_eq_250 : 
  length_of_yard = 250 := 
sorry

end NUMINAMATH_GPT_yard_length_eq_250_l2184_218454


namespace NUMINAMATH_GPT_intersection_sets_l2184_218497

theorem intersection_sets (M N : Set ℝ) :
  (M = {x | x * (x - 3) < 0}) → (N = {x | |x| < 2}) → (M ∩ N = {x | 0 < x ∧ x < 2}) :=
by
  intro hM hN
  rw [hM, hN]
  sorry

end NUMINAMATH_GPT_intersection_sets_l2184_218497


namespace NUMINAMATH_GPT_elizabeth_haircut_l2184_218461

theorem elizabeth_haircut (t s f : ℝ) (ht : t = 0.88) (hs : s = 0.5) : f = t - s := by
  sorry

end NUMINAMATH_GPT_elizabeth_haircut_l2184_218461


namespace NUMINAMATH_GPT_primes_eq_2_3_7_l2184_218405

theorem primes_eq_2_3_7 (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 ∨ p = 7 :=
by
  sorry

end NUMINAMATH_GPT_primes_eq_2_3_7_l2184_218405


namespace NUMINAMATH_GPT_find_A_l2184_218479

-- Definitions and conditions
def f (A B : ℝ) (x : ℝ) : ℝ := A * x - 3 * B^2 
def g (B C : ℝ) (x : ℝ) : ℝ := B * x + C

theorem find_A (A B C : ℝ) (hB : B ≠ 0) (hBC : B + C ≠ 0) :
  f A B (g B C 1) = 0 → A = (3 * B^2) / (B + C) :=
by
  -- Introduction of the hypotheses
  intro h
  sorry

end NUMINAMATH_GPT_find_A_l2184_218479


namespace NUMINAMATH_GPT_family_work_solution_l2184_218471

noncomputable def family_work_problem : Prop :=
  ∃ (M W : ℕ),
    M + W = 15 ∧
    (M * (9/120) + W * (6/180) = 1) ∧
    W = 3

theorem family_work_solution : family_work_problem :=
by
  sorry

end NUMINAMATH_GPT_family_work_solution_l2184_218471


namespace NUMINAMATH_GPT_total_sheep_flock_l2184_218455

-- Definitions and conditions based on the problem description
def crossing_rate : ℕ := 3 -- Sheep per minute
def sleep_duration : ℕ := 90 -- Duration of sleep in minutes
def sheep_counted_before_sleep : ℕ := 42 -- Sheep counted before falling asleep

-- Total sheep that crossed while Nicholas was asleep
def sheep_during_sleep := crossing_rate * sleep_duration 

-- Total sheep that crossed when Nicholas woke up
def total_sheep_after_sleep := sheep_counted_before_sleep + sheep_during_sleep

-- Prove the total number of sheep in the flock
theorem total_sheep_flock : (2 * total_sheep_after_sleep) = 624 :=
by
  sorry

end NUMINAMATH_GPT_total_sheep_flock_l2184_218455


namespace NUMINAMATH_GPT_complex_magnitude_equality_l2184_218448

open Complex Real

theorem complex_magnitude_equality :
  abs ((Complex.mk (5 * sqrt 2) (-5)) * (Complex.mk (2 * sqrt 3) 6)) = 60 :=
by
  sorry

end NUMINAMATH_GPT_complex_magnitude_equality_l2184_218448


namespace NUMINAMATH_GPT_total_tiles_correct_l2184_218477

-- Definitions for room dimensions
def room_length : ℕ := 24
def room_width : ℕ := 18

-- Definitions for tile dimensions
def border_tile_side : ℕ := 2
def inner_tile_side : ℕ := 1

-- Definitions for border and inner area calculations
def border_width : ℕ := 2 * border_tile_side
def inner_length : ℕ := room_length - border_width
def inner_width : ℕ := room_width - border_width

-- Calculation of the number of tiles needed
def border_area : ℕ := (room_length * room_width) - (inner_length * inner_width)
def num_border_tiles : ℕ := border_area / (border_tile_side * border_tile_side)
def inner_area : ℕ := inner_length * inner_width
def num_inner_tiles : ℕ := inner_area / (inner_tile_side * inner_tile_side)

-- Total number of tiles
def total_tiles : ℕ := num_border_tiles + num_inner_tiles

-- The proof statement
theorem total_tiles_correct : total_tiles = 318 := by
  -- Lean code to check the calculations, proof is omitted.
  sorry

end NUMINAMATH_GPT_total_tiles_correct_l2184_218477


namespace NUMINAMATH_GPT_points_per_enemy_l2184_218423

theorem points_per_enemy (total_enemies destroyed_enemies points_earned points_per_enemy : ℕ)
  (h1 : total_enemies = 8)
  (h2 : destroyed_enemies = total_enemies - 6)
  (h3 : points_earned = 10)
  (h4 : points_per_enemy = points_earned / destroyed_enemies) : 
  points_per_enemy = 5 := 
by
  sorry

end NUMINAMATH_GPT_points_per_enemy_l2184_218423


namespace NUMINAMATH_GPT_difference_in_girls_and_boys_l2184_218411

theorem difference_in_girls_and_boys (x : ℕ) (h1 : 3 + 4 = 7) (h2 : 7 * x = 49) : 4 * x - 3 * x = 7 := by
  sorry

end NUMINAMATH_GPT_difference_in_girls_and_boys_l2184_218411


namespace NUMINAMATH_GPT_hyperbola_perimeter_l2184_218482

-- Lean 4 statement
theorem hyperbola_perimeter (a b m : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F1 F2 : ℝ × ℝ) (A B : ℝ × ℝ)
  (hyperbola_eq : ∀ (x y : ℝ), (x,y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1})
  (line_through_F1 : ∀ (x y : ℝ), x = F1.1)
  (A_B_on_hyperbola : (A.1^2/a^2 - A.2^2/b^2 = 1) ∧ (B.1^2/a^2 - B.2^2/b^2 = 1))
  (dist_AB : dist A B = m)
  (dist_relations : dist A F2 + dist B F2 - (dist A F1 + dist B F1) = 4 * a) : 
  dist A F2 + dist B F2 + dist A B = 4 * a + 2 * m :=
sorry

end NUMINAMATH_GPT_hyperbola_perimeter_l2184_218482


namespace NUMINAMATH_GPT_number_of_blind_students_l2184_218453

variable (B D : ℕ)

-- Condition 1: The deaf-student population is 3 times the blind-student population.
axiom H1 : D = 3 * B

-- Condition 2: There are 180 students in total.
axiom H2 : B + D = 180

theorem number_of_blind_students : B = 45 :=
by
  -- Sorry is used to skip the proof steps. The theorem statement is correct and complete based on the conditions.
  sorry

end NUMINAMATH_GPT_number_of_blind_students_l2184_218453


namespace NUMINAMATH_GPT_no_distinct_natural_numbers_exist_l2184_218433

theorem no_distinct_natural_numbers_exist 
  (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ¬ (a + 1 / a = (1 / 2) * (b + 1 / b + c + 1 / c)) :=
sorry

end NUMINAMATH_GPT_no_distinct_natural_numbers_exist_l2184_218433


namespace NUMINAMATH_GPT_how_many_more_yellow_peaches_l2184_218444

-- Definitions
def red_peaches : ℕ := 7
def yellow_peaches_initial : ℕ := 15
def green_peaches : ℕ := 8
def combined_red_green_peaches := red_peaches + green_peaches
def required_yellow_peaches := 2 * combined_red_green_peaches
def additional_yellow_peaches_needed := required_yellow_peaches - yellow_peaches_initial

-- Theorem statement
theorem how_many_more_yellow_peaches :
  additional_yellow_peaches_needed = 15 :=
by
  sorry

end NUMINAMATH_GPT_how_many_more_yellow_peaches_l2184_218444
