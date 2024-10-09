import Mathlib

namespace relationship_among_g_a_0_f_b_l2128_212850

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x^2 - 3

theorem relationship_among_g_a_0_f_b (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  -- Function properties are non-trivial and are omitted.
  sorry

end relationship_among_g_a_0_f_b_l2128_212850


namespace find_integers_for_perfect_square_l2128_212859

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

theorem find_integers_for_perfect_square :
  {x : ℤ | is_perfect_square (x^4 + x^3 + x^2 + x + 1)} = {-1, 0, 3} :=
by
  sorry

end find_integers_for_perfect_square_l2128_212859


namespace distribute_coins_l2128_212820

theorem distribute_coins (x y : ℕ) (h₁ : x + y = 16) (h₂ : x^2 - y^2 = 16 * (x - y)) :
  x = 8 ∧ y = 8 :=
by {
  sorry
}

end distribute_coins_l2128_212820


namespace sum_a3_a4_a5_a6_l2128_212891

theorem sum_a3_a4_a5_a6 (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) : S 6 - S 2 = 40 :=
by
  sorry

end sum_a3_a4_a5_a6_l2128_212891


namespace phase_shift_correct_l2128_212856

-- Given the function y = 3 * sin (x - π / 5)
-- We need to prove that the phase shift is π / 5.

theorem phase_shift_correct :
  ∀ x : ℝ, 3 * Real.sin (x - Real.pi / 5) = 3 * Real.sin (x - C) →
  C = Real.pi / 5 :=
by
  sorry

end phase_shift_correct_l2128_212856


namespace triangle_area_l2128_212868

theorem triangle_area (x : ℝ) (h1 : 6 * x = 6) (h2 : 8 * x = 8) (h3 : 10 * x = 2 * 5) : 
  1 / 2 * 6 * 8 = 24 := 
sorry

end triangle_area_l2128_212868


namespace correct_systematic_sampling_l2128_212821

-- Definitions for conditions in a)
def num_bags := 50
def num_selected := 5
def interval := num_bags / num_selected

-- We encode the systematic sampling selection process
def systematic_sampling (n : Nat) (start : Nat) (interval: Nat) (count : Nat) : List Nat :=
  List.range count |>.map (λ i => start + i * interval)

-- Theorem to prove that the selection of bags should have an interval of 10
theorem correct_systematic_sampling :
  ∃ (start : Nat), systematic_sampling num_selected start interval num_selected = [7, 17, 27, 37, 47] := sorry

end correct_systematic_sampling_l2128_212821


namespace apple_lovers_l2128_212801

theorem apple_lovers :
  ∃ (x y : ℕ), 22 * x = 1430 ∧ 13 * (x + y) = 1430 ∧ y = 45 :=
by
  sorry

end apple_lovers_l2128_212801


namespace Fk_same_implies_eq_l2128_212841

def Q (n: ℕ) : ℕ :=
  -- Implementation of the square part of n
  sorry

def N (n: ℕ) : ℕ :=
  -- Implementation of the non-square part of n
  sorry

def Fk (k: ℕ) (n: ℕ) : ℕ :=
  -- Implementation of Fk function calculating the smallest positive integer bigger than kn such that Fk(n) * n is a perfect square
  sorry

theorem Fk_same_implies_eq (k: ℕ) (n m: ℕ) (hk: 0 < k) : Fk k n = Fk k m → n = m :=
  sorry

end Fk_same_implies_eq_l2128_212841


namespace num_int_values_satisfying_inequality_l2128_212804

theorem num_int_values_satisfying_inequality (x : ℤ) :
  (x^2 < 9 * x) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8) := 
sorry

end num_int_values_satisfying_inequality_l2128_212804


namespace jack_finishes_in_16_days_l2128_212867

noncomputable def pages_in_book : ℕ := 285
noncomputable def weekday_reading_rate : ℕ := 23
noncomputable def weekend_reading_rate : ℕ := 35
noncomputable def weekdays_per_week : ℕ := 5
noncomputable def weekends_per_week : ℕ := 2
noncomputable def weekday_skipped : ℕ := 1
noncomputable def weekend_skipped : ℕ := 1

noncomputable def pages_per_week : ℕ :=
  (weekdays_per_week - weekday_skipped) * weekday_reading_rate + 
  (weekends_per_week - weekend_skipped) * weekend_reading_rate

noncomputable def weeks_needed : ℕ :=
  pages_in_book / pages_per_week

noncomputable def pages_left_after_weeks : ℕ :=
  pages_in_book % pages_per_week

noncomputable def extra_days_needed (pages_left : ℕ) : ℕ :=
  if pages_left > weekend_reading_rate then 2
  else if pages_left > weekday_reading_rate then 2
  else 1

noncomputable def total_days_needed : ℕ :=
  weeks_needed * 7 + extra_days_needed (pages_left_after_weeks)

theorem jack_finishes_in_16_days : total_days_needed = 16 := by
  sorry

end jack_finishes_in_16_days_l2128_212867


namespace Steven_more_than_Jill_l2128_212843

variable (Jill Jake Steven : ℕ)

def Jill_peaches : Jill = 87 := by sorry
def Jake_peaches_more : Jake = Jill + 13 := by sorry
def Steven_peaches_more : Steven = Jake + 5 := by sorry

theorem Steven_more_than_Jill : Steven - Jill = 18 := by
  -- Proof steps to be filled
  sorry

end Steven_more_than_Jill_l2128_212843


namespace conditionD_necessary_not_sufficient_l2128_212858

variable (a b : ℝ)

-- Define each of the conditions as separate variables
def conditionA : Prop := |a| < |b|
def conditionB : Prop := 2 * a < 2 * b
def conditionC : Prop := a < b - 1
def conditionD : Prop := a < b + 1

-- Prove that condition D is necessary but not sufficient for a < b
theorem conditionD_necessary_not_sufficient : conditionD a b → (¬ conditionA a b ∨ ¬ conditionB a b ∨ ¬ conditionC a b) ∧ ¬(conditionD a b ↔ a < b) :=
by sorry

end conditionD_necessary_not_sufficient_l2128_212858


namespace range_of_expression_l2128_212802

variable {a b : ℝ}

theorem range_of_expression 
  (h₁ : -1 < a + b) (h₂ : a + b < 3)
  (h₃ : 2 < a - b) (h₄ : a - b < 4) :
  -9 / 2 < 2 * a + 3 * b ∧ 2 * a + 3 * b < 13 / 2 := 
sorry

end range_of_expression_l2128_212802


namespace cos_pi_over_2_minus_l2128_212851

theorem cos_pi_over_2_minus (A : ℝ) (h : Real.sin A = 1 / 2) : Real.cos (3 * Real.pi / 2 - A) = -1 / 2 :=
  sorry

end cos_pi_over_2_minus_l2128_212851


namespace sandy_remaining_puppies_l2128_212897

-- Definitions from the problem
def initial_puppies : ℕ := 8
def given_away_puppies : ℕ := 4

-- Theorem statement
theorem sandy_remaining_puppies : initial_puppies - given_away_puppies = 4 := by
  sorry

end sandy_remaining_puppies_l2128_212897


namespace chameleons_all_white_l2128_212887

theorem chameleons_all_white :
  ∀ (a b c : ℕ), a = 800 → b = 1000 → c = 1220 → 
  (a + b + c = 3020) → (a % 3 = 2) → (b % 3 = 1) → (c % 3 = 2) →
    ∃ k : ℕ, (k = 3020 ∧ (k % 3 = 1)) ∧ 
    (if k = b then a = 0 ∧ c = 0 else false) :=
by
  sorry

end chameleons_all_white_l2128_212887


namespace positive_divisors_8_fact_l2128_212822

-- Factorial function definition
def factorial : Nat → Nat
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Function to compute the number of divisors from prime factors
def numDivisors (factors : List (Nat × Nat)) : Nat :=
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

-- Known prime factorization of 8!
noncomputable def factors_8_fact : List (Nat × Nat) :=
  [(2, 7), (3, 2), (5, 1), (7, 1)]

-- Theorem statement
theorem positive_divisors_8_fact : numDivisors factors_8_fact = 96 :=
  sorry

end positive_divisors_8_fact_l2128_212822


namespace sum_first_n_terms_arithmetic_sequence_l2128_212817

theorem sum_first_n_terms_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (a 2 + a 4 = 10) ∧ (∀ n : ℕ, a (n + 1) - a n = 2) → 
  (∀ n : ℕ, S n = n^2) := by
  intro h
  sorry

end sum_first_n_terms_arithmetic_sequence_l2128_212817


namespace sum_first_12_terms_l2128_212825

theorem sum_first_12_terms (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n * a n)
  (h2 : a 6 + a 7 = 18) : 
  S 12 = 108 :=
sorry

end sum_first_12_terms_l2128_212825


namespace solve_system_l2128_212893

def system_of_equations_solution : Prop :=
  ∃ (x y : ℚ), 4 * x - 7 * y = -9 ∧ 5 * x + 3 * y = -11 ∧ (x, y) = (-(104 : ℚ) / 47, (1 : ℚ) / 47)

theorem solve_system : system_of_equations_solution :=
sorry

end solve_system_l2128_212893


namespace possible_values_of_a_l2128_212809

def A (a : ℝ) : Set ℝ := { x | 0 < x ∧ x < a }
def B : Set ℝ := { x | 1 < x ∧ x < 2 }
def complement_R (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem possible_values_of_a (a : ℝ) :
  (∃ x, x ∈ A a) →
  B ⊆ complement_R (A a) →
  0 < a ∧ a ≤ 1 :=
by 
  sorry

end possible_values_of_a_l2128_212809


namespace min_abs_sum_is_5_l2128_212874

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end min_abs_sum_is_5_l2128_212874


namespace distinct_solution_count_l2128_212832

theorem distinct_solution_count
  (n : ℕ)
  (x y : ℕ)
  (h1 : x ≠ y)
  (h2 : x ≠ 2 * y)
  (h3 : y ≠ 2 * x)
  (h4 : x^2 - x * y + y^2 = n) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card ≥ 12 ∧ ∀ (a b : ℕ), (a, b) ∈ pairs → a^2 - a * b + b^2 = n :=
sorry

end distinct_solution_count_l2128_212832


namespace math_books_together_l2128_212812

theorem math_books_together (math_books english_books : ℕ) (h_math_books : math_books = 2) (h_english_books : english_books = 2) : 
  ∃ ways, ways = 12 := by
  sorry

end math_books_together_l2128_212812


namespace possible_values_of_polynomial_l2128_212877

theorem possible_values_of_polynomial (x : ℝ) (h : x^2 - 7 * x + 12 < 0) : 
48 < x^2 + 7 * x + 12 ∧ x^2 + 7 * x + 12 < 64 :=
sorry

end possible_values_of_polynomial_l2128_212877


namespace green_more_than_blue_l2128_212865

theorem green_more_than_blue (B Y G : Nat) (h1 : B + Y + G = 108) (h2 : B * 7 = Y * 3) (h3 : B * 8 = G * 3) : G - B = 30 := by
  sorry

end green_more_than_blue_l2128_212865


namespace expansion_simplification_l2128_212823

variable (x y : ℝ)

theorem expansion_simplification :
  let a := 3 * x + 4
  let b := 2 * x + 6 * y + 7
  a * b = 6 * x ^ 2 + 18 * x * y + 29 * x + 24 * y + 28 :=
by
  sorry

end expansion_simplification_l2128_212823


namespace no_solution_if_and_only_if_l2128_212815

theorem no_solution_if_and_only_if (n : ℝ) : 
  ¬ ∃ (x y z : ℝ), 
    (n * x + y = 1) ∧ 
    (n * y + z = 1) ∧ 
    (x + n * z = 1) ↔ n = -1 :=
by
  sorry

end no_solution_if_and_only_if_l2128_212815


namespace triangle_sum_l2128_212873

def triangle (a b c : ℕ) : ℤ := a * b - c

theorem triangle_sum :
  triangle 2 3 5 + triangle 1 4 7 = -2 :=
by
  -- This is where the proof would go
  sorry

end triangle_sum_l2128_212873


namespace emery_total_alteration_cost_l2128_212838

-- Definition of the initial conditions
def num_pairs_of_shoes := 17
def cost_per_shoe := 29
def shoes_per_pair := 2

-- Proving the total cost
theorem emery_total_alteration_cost : num_pairs_of_shoes * shoes_per_pair * cost_per_shoe = 986 := by
  sorry

end emery_total_alteration_cost_l2128_212838


namespace evaluate_custom_operation_l2128_212869

def custom_operation (x y : ℕ) : ℕ := 2 * x - 4 * y

theorem evaluate_custom_operation :
  custom_operation 7 3 = 2 :=
by
  sorry

end evaluate_custom_operation_l2128_212869


namespace code_word_MEET_l2128_212890

def translate_GREAT_TIME : String → ℕ 
| "G" => 0
| "R" => 1
| "E" => 2
| "A" => 3
| "T" => 4
| "I" => 5
| "M" => 6
| _   => 0 -- Default case for simplicity, not strictly necessary

theorem code_word_MEET : translate_GREAT_TIME "M" = 6 ∧ translate_GREAT_TIME "E" = 2 ∧ translate_GREAT_TIME "T" = 4 →
  let MEET : ℕ := (translate_GREAT_TIME "M" * 1000) + 
                  (translate_GREAT_TIME "E" * 100) + 
                  (translate_GREAT_TIME "E" * 10) + 
                  (translate_GREAT_TIME "T")
  MEET = 6224 :=
sorry

end code_word_MEET_l2128_212890


namespace susie_large_rooms_count_l2128_212829

theorem susie_large_rooms_count:
  (∀ small_rooms medium_rooms large_rooms : ℕ,  
    (small_rooms = 4) → 
    (medium_rooms = 3) → 
    (large_rooms = x) → 
    (225 = small_rooms * 15 + medium_rooms * 25 + large_rooms * 35) → 
    x = 2) :=
by
  intros small_rooms medium_rooms large_rooms
  intros h1 h2 h3 h4
  sorry

end susie_large_rooms_count_l2128_212829


namespace fourth_proportional_segment_l2128_212862

theorem fourth_proportional_segment 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  : ∃ x : ℝ, x = (b * c) / a := 
by
  sorry

end fourth_proportional_segment_l2128_212862


namespace existence_of_x2_with_sum_ge_2_l2128_212813

variables (a b c x1 x2 : ℝ) (h_root1 : a * x1^2 + b * x1 + c = 0) (h_x1_pos : x1 > 0)

theorem existence_of_x2_with_sum_ge_2 :
  ∃ x2, (c * x2^2 + b * x2 + a = 0) ∧ (x1 + x2 ≥ 2) :=
sorry

end existence_of_x2_with_sum_ge_2_l2128_212813


namespace fraction_of_income_from_tips_l2128_212863

variable (S T I : ℝ)

/- Definition of the conditions -/
def tips_condition : Prop := T = (3 / 4) * S
def income_condition : Prop := I = S + T

/- The proof problem statement, asserting the desired result -/
theorem fraction_of_income_from_tips (h1 : tips_condition S T) (h2 : income_condition S T I) : T / I = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l2128_212863


namespace value_of_f_at_2019_l2128_212866

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_positive : ∀ x : ℝ, f x > 0)
variable (h_functional : ∀ x : ℝ, f (x + 2) = 1 / (f x))

theorem value_of_f_at_2019 : f 2019 = 1 :=
by
  sorry

end value_of_f_at_2019_l2128_212866


namespace cubic_eq_factorization_l2128_212857

theorem cubic_eq_factorization (a b c : ℝ) :
  (∃ m n : ℝ, (x^3 + a * x^2 + b * x + c = (x^2 + m) * (x + n))) ↔ (c = a * b) :=
sorry

end cubic_eq_factorization_l2128_212857


namespace sum_of_integers_l2128_212881

theorem sum_of_integers (x y : ℤ) (h_pos : 0 < y) (h_gt : x > y) (h_diff : x - y = 14) (h_prod : x * y = 48) : x + y = 20 :=
sorry

end sum_of_integers_l2128_212881


namespace distance_formula_example_l2128_212886

variable (x1 y1 x2 y2 : ℝ)

theorem distance_formula_example : dist (3, -1) (-4, 3) = Real.sqrt 65 :=
by
  let x1 := 3
  let y1 := -1
  let x2 := -4
  let y2 := 3
  sorry

end distance_formula_example_l2128_212886


namespace flowers_in_each_basket_l2128_212882

-- Definitions based on the conditions
def initial_flowers (d1 d2 : Nat) : Nat := d1 + d2
def grown_flowers (initial growth : Nat) : Nat := initial + growth
def remaining_flowers (grown dead : Nat) : Nat := grown - dead
def flowers_per_basket (remaining baskets : Nat) : Nat := remaining / baskets

-- Given conditions in Lean 4
theorem flowers_in_each_basket 
    (daughters_flowers : Nat) 
    (growth : Nat) 
    (dead : Nat) 
    (baskets : Nat) 
    (h_daughters : daughters_flowers = 5 + 5) 
    (h_growth : growth = 20) 
    (h_dead : dead = 10) 
    (h_baskets : baskets = 5) : 
    flowers_per_basket (remaining_flowers (grown_flowers (initial_flowers 5 5) growth) dead) baskets = 4 := 
sorry

end flowers_in_each_basket_l2128_212882


namespace lcm_of_18_and_36_l2128_212864

theorem lcm_of_18_and_36 : Nat.lcm 18 36 = 36 := 
by 
  sorry

end lcm_of_18_and_36_l2128_212864


namespace arccos_neg_one_eq_pi_l2128_212811

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := 
by
  sorry

end arccos_neg_one_eq_pi_l2128_212811


namespace value_independent_of_a_value_when_b_is_neg_2_l2128_212839

noncomputable def algebraic_expression (a b : ℝ) : ℝ :=
  3 * a^2 + (4 * a * b - a^2) - 2 * (a^2 + 2 * a * b - b^2)

theorem value_independent_of_a (a b : ℝ) : algebraic_expression a b = 2 * b^2 :=
by
  sorry

theorem value_when_b_is_neg_2 (a : ℝ) : algebraic_expression a (-2) = 8 :=
by
  sorry

end value_independent_of_a_value_when_b_is_neg_2_l2128_212839


namespace a_plus_b_values_l2128_212892

theorem a_plus_b_values (a b : ℝ) (h1 : abs a = 5) (h2 : abs b = 3) (h3 : abs (a - b) = b - a) : a + b = -2 ∨ a + b = -8 :=
sorry

end a_plus_b_values_l2128_212892


namespace halfway_between_l2128_212884

theorem halfway_between (a b : ℚ) (h1 : a = 1/12) (h2 : b = 1/15) : (a + b) / 2 = 3 / 40 := by
  -- proofs go here
  sorry

end halfway_between_l2128_212884


namespace range_of_k_l2128_212883

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x - 1 = 0) ↔ (k ≥ 0 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_l2128_212883


namespace empty_seats_l2128_212889

theorem empty_seats (total_seats : ℕ) (people_watching : ℕ) (h_total_seats : total_seats = 750) (h_people_watching : people_watching = 532) : 
  total_seats - people_watching = 218 :=
by
  sorry

end empty_seats_l2128_212889


namespace expression_not_equal_l2128_212875

variable (a b c : ℝ)

theorem expression_not_equal :
  (a - (b - c)) ≠ (a - b - c) :=
by sorry

end expression_not_equal_l2128_212875


namespace min_value_fraction_l2128_212861

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_ln : Real.log (a + b) = 0) :
  (2 / a + 3 / b) = 5 + 2 * Real.sqrt 6 :=
by
  sorry

end min_value_fraction_l2128_212861


namespace square_side_length_exists_l2128_212840

theorem square_side_length_exists
    (k : ℕ)
    (n : ℕ)
    (h_side_length_condition : n * n = k * (k - 7))
    (h_grid_lines : k > 7) :
    n = 12 ∨ n = 24 :=
by sorry

end square_side_length_exists_l2128_212840


namespace solve_problem_l2128_212895

noncomputable def problem_expression : ℝ :=
  4^(1/2) + Real.log (3^2) / Real.log 3

theorem solve_problem : problem_expression = 4 := by
  sorry

end solve_problem_l2128_212895


namespace cos_arcsin_eq_tan_arcsin_eq_l2128_212870

open Real

theorem cos_arcsin_eq (h : arcsin (3 / 5) = θ) : cos (arcsin (3 / 5)) = 4 / 5 := by
  sorry

theorem tan_arcsin_eq (h : arcsin (3 / 5) = θ) : tan (arcsin (3 / 5)) = 3 / 4 := by
  sorry

end cos_arcsin_eq_tan_arcsin_eq_l2128_212870


namespace three_number_relationship_l2128_212834

theorem three_number_relationship :
  let a := (0.7 : ℝ) ^ 6
  let b := 6 ^ (0.7 : ℝ)
  let c := Real.log 6 / Real.log 0.7
  c < a ∧ a < b :=
sorry

end three_number_relationship_l2128_212834


namespace regular_ticket_price_l2128_212849

variable (P : ℝ) -- Define the regular ticket price as a real number

-- Condition: Travis pays $1400 for his ticket after a 30% discount on a regular price P
axiom h : 0.70 * P = 1400

-- Theorem statement: Proving that the regular ticket price P equals $2000
theorem regular_ticket_price : P = 2000 :=
by 
  sorry

end regular_ticket_price_l2128_212849


namespace average_mb_per_hour_of_music_l2128_212879

/--
Given a digital music library:
- It contains 14 days of music.
- The first 7 days use 10,000 megabytes of disk space.
- The next 7 days use 14,000 megabytes of disk space.
- Each day has 24 hours.

Prove that the average megabytes per hour of music in this library is 71 megabytes.
-/
theorem average_mb_per_hour_of_music
  (days_total : ℕ) 
  (days_first : ℕ) 
  (days_second : ℕ) 
  (mb_first : ℕ) 
  (mb_second : ℕ) 
  (hours_per_day : ℕ) 
  (total_mb : ℕ) 
  (total_hours : ℕ) :
  days_total = 14 →
  days_first = 7 →
  days_second = 7 →
  mb_first = 10000 →
  mb_second = 14000 →
  hours_per_day = 24 →
  total_mb = mb_first + mb_second →
  total_hours = days_total * hours_per_day →
  total_mb / total_hours = 71 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end average_mb_per_hour_of_music_l2128_212879


namespace total_sampled_students_is_80_l2128_212814

-- Given conditions
variables (total_students num_freshmen num_sampled_freshmen : ℕ)
variables (total_students := 2400) (num_freshmen := 600) (num_sampled_freshmen := 20)

-- Define the proportion for stratified sampling.
def stratified_sampling (total_students num_freshmen num_sampled_freshmen total_sampled_students : ℕ) : Prop :=
  num_freshmen / total_students = num_sampled_freshmen / total_sampled_students

-- State the theorem: Prove the total number of students to be sampled from the entire school is 80.
theorem total_sampled_students_is_80 : ∃ n, stratified_sampling total_students num_freshmen num_sampled_freshmen n ∧ n = 80 := 
sorry

end total_sampled_students_is_80_l2128_212814


namespace bridge_length_correct_l2128_212818

def train_length : ℕ := 256
def train_speed_kmh : ℕ := 72
def crossing_time : ℕ := 20

noncomputable def convert_speed (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600 -- Conversion from km/h to m/s

noncomputable def bridge_length (train_length : ℕ) (speed_m : ℕ) (time_s : ℕ) : ℕ :=
  (speed_m * time_s) - train_length

theorem bridge_length_correct :
  bridge_length train_length (convert_speed train_speed_kmh) crossing_time = 144 :=
by
  sorry

end bridge_length_correct_l2128_212818


namespace day_of_week_150th_day_of_year_N_minus_1_l2128_212806

/-- Given that the 250th day of year N is a Friday and year N is a leap year,
    prove that the 150th day of year N-1 is a Friday. -/
theorem day_of_week_150th_day_of_year_N_minus_1
  (N : ℕ) 
  (H1 : (250 % 7 = 5) → true)  -- Condition that 250th day is five days after Sunday (Friday).
  (H2 : 366 % 7 = 2)           -- Condition that year N is a leap year with 366 days.
  (H3 : (N - 1) % 7 = (N - 1) % 7) -- Used for year transition check.
  : 150 % 7 = 5 := sorry       -- Proving that the 150th of year N-1 is Friday.

end day_of_week_150th_day_of_year_N_minus_1_l2128_212806


namespace width_of_field_l2128_212872

noncomputable def field_width 
  (field_length : ℝ) 
  (rope_length : ℝ)
  (grazing_area : ℝ) : ℝ :=
if field_length > 2 * rope_length 
then rope_length
else grazing_area

theorem width_of_field 
  (field_length : ℝ := 45)
  (rope_length : ℝ := 22)
  (grazing_area : ℝ := 380.132711084365) : field_width field_length rope_length grazing_area = rope_length :=
by 
  sorry

end width_of_field_l2128_212872


namespace mean_of_six_numbers_l2128_212845

theorem mean_of_six_numbers (sum : ℚ) (h : sum = 1/3) : (sum / 6 = 1/18) :=
by
  sorry

end mean_of_six_numbers_l2128_212845


namespace courtyard_paving_l2128_212899

noncomputable def length_of_brick (L : ℕ) := L = 12

theorem courtyard_paving  (courtyard_length : ℕ) (courtyard_width : ℕ) 
                           (brick_width : ℕ) (total_bricks : ℕ) 
                           (H1 : courtyard_length = 18) (H2 : courtyard_width = 12) 
                           (H3 : brick_width = 6) (H4 : total_bricks = 30000) 
                           : length_of_brick 12 := 
by 
  sorry

end courtyard_paving_l2128_212899


namespace no_six_consecutive010101_l2128_212831

def unit_digit (n: ℕ) : ℕ := n % 10

def sequence : ℕ → ℕ
| 0     => 1
| 1     => 0
| 2     => 1
| 3     => 0
| 4     => 1
| 5     => 0
| (n + 6) => unit_digit (sequence n + sequence (n + 1) + sequence (n + 2) + sequence (n + 3) + sequence (n + 4) + sequence (n + 5))

theorem no_six_consecutive010101 : ∀ n, ¬ (sequence n = 0 ∧ sequence (n + 1) = 1 ∧ sequence (n + 2) = 0 ∧ sequence (n + 3) = 1 ∧ sequence (n + 4) = 0 ∧ sequence (n + 5) = 1) :=
sorry

end no_six_consecutive010101_l2128_212831


namespace quadrilateral_area_inequality_equality_condition_l2128_212833

theorem quadrilateral_area_inequality 
  (a b c d S : ℝ) 
  (hS : S = 0.5 * a * c + 0.5 * b * d) 
  : S ≤ 0.5 * (a * c + b * d) :=
sorry

theorem equality_condition 
  (a b c d S : ℝ) 
  (hS : S = 0.5 * a * c + 0.5 * b * d)
  (h_perpendicular : ∃ (α β : ℝ), α = 90 ∧ β = 90) 
  : S = 0.5 * (a * c + b * d) :=
sorry

end quadrilateral_area_inequality_equality_condition_l2128_212833


namespace decompose_expression_l2128_212871

-- Define the variables a and b as real numbers
variables (a b : ℝ)

-- State the theorem corresponding to the proof problem
theorem decompose_expression : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) :=
by
  sorry

end decompose_expression_l2128_212871


namespace person_speed_in_kmph_l2128_212800

noncomputable def speed_calculation (distance_meters : ℕ) (time_minutes : ℕ) : ℝ :=
  let distance_km := (distance_meters : ℝ) / 1000
  let time_hours := (time_minutes : ℝ) / 60
  distance_km / time_hours

theorem person_speed_in_kmph :
  speed_calculation 1080 12 = 5.4 :=
by
  sorry

end person_speed_in_kmph_l2128_212800


namespace dan_has_more_balloons_l2128_212805

-- Constants representing the number of balloons Dan and Tim have
def dans_balloons : ℝ := 29.0
def tims_balloons : ℝ := 4.142857143

-- Theorem: The ratio of Dan's balloons to Tim's balloons is 7
theorem dan_has_more_balloons : dans_balloons / tims_balloons = 7 := 
by
  sorry

end dan_has_more_balloons_l2128_212805


namespace total_items_in_quiz_l2128_212810

theorem total_items_in_quiz (score_percent : ℝ) (mistakes : ℕ) (total_items : ℕ) 
  (h1 : score_percent = 80) 
  (h2 : mistakes = 5) :
  total_items = 25 :=
sorry

end total_items_in_quiz_l2128_212810


namespace min_disks_required_l2128_212827

/-- A structure to hold information about the file storage problem -/
structure FileStorageConditions where
  total_files : ℕ
  disk_capacity : ℝ
  num_files_1_6MB : ℕ
  num_files_1MB : ℕ
  num_files_0_5MB : ℕ

/-- Define specific conditions given in the problem -/
def storage_conditions : FileStorageConditions := {
  total_files := 42,
  disk_capacity := 2.88,
  num_files_1_6MB := 8,
  num_files_1MB := 16,
  num_files_0_5MB := 18 -- Derived from total_files - num_files_1_6MB - num_files_1MB
}

/-- Theorem stating the minimum number of disks required to store all files is 16 -/
theorem min_disks_required (c : FileStorageConditions)
  (h1 : c.total_files = 42)
  (h2 : c.disk_capacity = 2.88)
  (h3 : c.num_files_1_6MB = 8)
  (h4 : c.num_files_1MB = 16)
  (h5 : c.num_files_0_5MB = 18) :
  ∃ n : ℕ, n = 16 := by
  sorry

end min_disks_required_l2128_212827


namespace smallest_natural_with_properties_l2128_212816

theorem smallest_natural_with_properties :
  ∃ n : ℕ, (∃ N : ℕ, n = 10 * N + 6) ∧ 4 * (10 * N + 6) = 6 * 10^(5 : ℕ) + N ∧ n = 153846 := sorry

end smallest_natural_with_properties_l2128_212816


namespace prob_at_least_one_head_l2128_212842

theorem prob_at_least_one_head (n : ℕ) (hn : n = 3) : 
  1 - (1 / (2^n)) = 7 / 8 :=
by
  sorry

end prob_at_least_one_head_l2128_212842


namespace isosceles_triangle_base_angle_l2128_212819

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle_isosceles : a = b ∨ b = c ∨ c = a)
  (h_angle_sum : a + b + c = 180) (h_one_angle : a = 50 ∨ b = 50 ∨ c = 50) :
  a = 50 ∨ b = 50 ∨ c = 50 ∨ a = 65 ∨ b = 65 ∨ c = 65 :=
by
  sorry

end isosceles_triangle_base_angle_l2128_212819


namespace solve_for_y_l2128_212847

theorem solve_for_y (y : ℤ) : (2 / 3 - 3 / 5 : ℚ) = 5 / y → y = 75 :=
by
  sorry

end solve_for_y_l2128_212847


namespace number_of_positive_integers_l2128_212848

theorem number_of_positive_integers (n : ℕ) : ∃! k : ℕ, k = 5 ∧
  (∀ n : ℕ, (1 ≤ n) → (12 % (n + 1) = 0)) :=
sorry

end number_of_positive_integers_l2128_212848


namespace fruit_salad_weight_l2128_212807

theorem fruit_salad_weight (melon berries : ℝ) (h_melon : melon = 0.25) (h_berries : berries = 0.38) : melon + berries = 0.63 :=
by
  sorry

end fruit_salad_weight_l2128_212807


namespace andrew_total_homeless_shelter_donation_l2128_212898

-- Given constants and conditions
def bake_sale_total : ℕ := 400
def ingredients_cost : ℕ := 100
def piggy_bank_donation : ℕ := 10

-- Intermediate calculated values
def remaining_total : ℕ := bake_sale_total - ingredients_cost
def shelter_donation_from_bake_sale : ℕ := remaining_total / 2

-- Final goal statement
theorem andrew_total_homeless_shelter_donation :
  shelter_donation_from_bake_sale + piggy_bank_donation = 160 :=
by
  -- Proof to be provided.
  sorry

end andrew_total_homeless_shelter_donation_l2128_212898


namespace triangle_least_perimeter_l2128_212803

noncomputable def least_perimeter_of_triangle : ℕ :=
  let a := 7
  let b := 17
  let c := 13
  a + b + c

theorem triangle_least_perimeter :
  let a := 7
  let b := 17
  let c := 13
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  4 ∣ (a^2 + b^2 + c^2) - 2 * c^2 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →
  least_perimeter_of_triangle = 37 :=
by
  intros _ _ _ h
  sorry

end triangle_least_perimeter_l2128_212803


namespace probability_at_least_three_prime_dice_l2128_212880

-- Definitions from the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def p := 5 / 12
def q := 7 / 12
def binomial (n k : ℕ) := Nat.choose n k

-- The probability of at least three primes
theorem probability_at_least_three_prime_dice :
  (binomial 5 3 * p ^ 3 * q ^ 2) +
  (binomial 5 4 * p ^ 4 * q ^ 1) +
  (binomial 5 5 * p ^ 5 * q ^ 0) = 40625 / 622080 :=
by
  sorry

end probability_at_least_three_prime_dice_l2128_212880


namespace avg_starting_with_d_l2128_212888

-- Define c and d as positive integers
variables (c d : ℤ) (hc : c > 0) (hd : d > 0)

-- Define d as the average of the seven consecutive integers starting with c
def avg_starting_with_c (c : ℤ) : ℤ := (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7

-- Define the condition that d is the average of the seven consecutive integers starting with c
axiom d_is_avg_starting_with_c : d = avg_starting_with_c c

-- Prove that the average of the seven consecutive integers starting with d equals c + 6
theorem avg_starting_with_d (c d : ℤ) (hc : c > 0) (hd : d > 0) (h : d = avg_starting_with_c c) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 6 := by
  sorry

end avg_starting_with_d_l2128_212888


namespace problem_equivalent_proof_l2128_212894

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l2128_212894


namespace sum_lent_l2128_212853

theorem sum_lent (P : ℝ) (R : ℝ := 4) (T : ℝ := 8) (I : ℝ) (H1 : I = P - 204) (H2 : I = (P * R * T) / 100) : 
  P = 300 :=
by 
  sorry

end sum_lent_l2128_212853


namespace leaves_count_l2128_212896

theorem leaves_count {m n L : ℕ} (h1 : m + n = 10) (h2 : L = 5 * m + 2 * n) :
  ¬(L = 45 ∨ L = 39 ∨ L = 37 ∨ L = 31) :=
by
  sorry

end leaves_count_l2128_212896


namespace value_of_e_l2128_212846

variable (e : ℝ)
noncomputable def eq1 : Prop :=
  ((10 * 0.3 + 2) / 4 - (3 * 0.3 - e) / 18 = (2 * 0.3 + 4) / 3)

theorem value_of_e : eq1 e → e = 6 := by
  intro h
  sorry

end value_of_e_l2128_212846


namespace quadratic_one_solution_l2128_212808

theorem quadratic_one_solution (b d : ℝ) (h1 : b + d = 35) (h2 : b < d) (h3 : (24 : ℝ)^2 - 4 * b * d = 0) :
  (b, d) = (35 - Real.sqrt 649 / 2, 35 + Real.sqrt 649 / 2) := 
sorry

end quadratic_one_solution_l2128_212808


namespace flowers_per_bouquet_l2128_212835

-- Defining the problem parameters
def total_flowers : ℕ := 66
def wilted_flowers : ℕ := 10
def num_bouquets : ℕ := 7

-- The goal is to prove that the number of flowers per bouquet is 8
theorem flowers_per_bouquet :
  (total_flowers - wilted_flowers) / num_bouquets = 8 :=
by
  sorry

end flowers_per_bouquet_l2128_212835


namespace harkamal_total_payment_l2128_212830

def grapes_quantity : ℕ := 10
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55

def cost_of_grapes : ℕ := grapes_quantity * grapes_rate
def cost_of_mangoes : ℕ := mangoes_quantity * mangoes_rate

def total_amount_paid : ℕ := cost_of_grapes + cost_of_mangoes

theorem harkamal_total_payment : total_amount_paid = 1195 := by
  sorry

end harkamal_total_payment_l2128_212830


namespace find_j_l2128_212854

def original_number (a b k : ℕ) : ℕ := 10 * a + b
def sum_of_digits (a b : ℕ) : ℕ := a + b
def modified_number (b a : ℕ) : ℕ := 20 * b + a

theorem find_j
  (a b k j : ℕ)
  (h1 : original_number a b k = k * sum_of_digits a b)
  (h2 : modified_number b a = j * sum_of_digits a b) :
  j = (199 + k) / 10 :=
sorry

end find_j_l2128_212854


namespace A_share_of_profit_l2128_212826

-- Define the conditions
def A_investment : ℕ := 100
def A_months : ℕ := 12
def B_investment : ℕ := 200
def B_months : ℕ := 6
def total_profit : ℕ := 100

-- Calculate the weighted investments (directly from conditions)
def A_weighted_investment : ℕ := A_investment * A_months
def B_weighted_investment : ℕ := B_investment * B_months
def total_weighted_investment : ℕ := A_weighted_investment + B_weighted_investment

-- Prove A's share of the profit
theorem A_share_of_profit : (A_weighted_investment / total_weighted_investment : ℚ) * total_profit = 50 := by
  -- The proof will go here
  sorry

end A_share_of_profit_l2128_212826


namespace mosquito_drops_per_feed_l2128_212836

-- Defining the constants and conditions.
def drops_per_liter : ℕ := 5000
def liters_to_die : ℕ := 3
def mosquitoes_to_kill : ℕ := 750

-- The assertion we want to prove.
theorem mosquito_drops_per_feed :
  (drops_per_liter * liters_to_die) / mosquitoes_to_kill = 20 :=
by
  sorry

end mosquito_drops_per_feed_l2128_212836


namespace sum_of_positive_integer_factors_of_24_l2128_212860

-- Define the number 24
def n : ℕ := 24

-- Define the list of positive factors of 24
def pos_factors_of_24 : List ℕ := [1, 2, 4, 8, 3, 6, 12, 24]

-- Define the sum of the factors
def sum_of_factors : ℕ := pos_factors_of_24.sum

-- The theorem statement
theorem sum_of_positive_integer_factors_of_24 : sum_of_factors = 60 := by
  sorry

end sum_of_positive_integer_factors_of_24_l2128_212860


namespace simplify_fraction_l2128_212824

theorem simplify_fraction : 
    (3 ^ 1011 + 3 ^ 1009) / (3 ^ 1011 - 3 ^ 1009) = 5 / 4 := 
by
  sorry

end simplify_fraction_l2128_212824


namespace find_X_l2128_212855

theorem find_X : ∃ X : ℝ, 0.60 * X = 0.30 * 800 + 370 ∧ X = 1016.67 := by
  sorry

end find_X_l2128_212855


namespace triangular_difference_30_28_l2128_212837

noncomputable def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_difference_30_28 : triangular 30 - triangular 28 = 59 :=
by
  sorry

end triangular_difference_30_28_l2128_212837


namespace minimum_focal_chord_length_l2128_212852

theorem minimum_focal_chord_length (p : ℝ) (hp : p > 0) :
  ∃ l, (l = 2 * p) ∧ (∀ y x1 x2, y^2 = 2 * p * x1 ∧ y^2 = 2 * p * x2 → l = x2 - x1) := 
sorry

end minimum_focal_chord_length_l2128_212852


namespace yellow_curved_given_curved_l2128_212876

variable (P_green : ℝ) (P_yellow : ℝ) (P_straight : ℝ) (P_curved : ℝ)
variable (P_red_given_straight : ℝ) 

-- Given conditions
variables (h1 : P_green = 3 / 4) 
          (h2 : P_yellow = 1 / 4) 
          (h3 : P_straight = 1 / 2) 
          (h4 : P_curved = 1 / 2)
          (h5 : P_red_given_straight = 1 / 3)

-- To be proven
theorem yellow_curved_given_curved : (P_yellow * P_curved) / P_curved = 1 / 4 :=
by
sorry

end yellow_curved_given_curved_l2128_212876


namespace min_t_of_BE_CF_l2128_212844

theorem min_t_of_BE_CF (A B C E F: ℝ)
  (hE_midpoint_AC : ∃ D, D = (A + C) / 2 ∧ E = D)
  (hF_midpoint_AB : ∃ D, D = (A + B) / 2 ∧ F = D)
  (h_AB_AC_ratio : B - A = 2 / 3 * (C - A)) :
  ∃ t : ℝ, t = 7 / 8 ∧ ∀ (BE CF : ℝ), BE = dist B E ∧ CF = dist C F → BE / CF < t := by
  sorry

end min_t_of_BE_CF_l2128_212844


namespace find_number_l2128_212878

-- Define the conditions
variables (y : ℝ) (Some_number : ℝ) (x : ℝ)

-- State the given equation
def equation := 19 * (x + y) + Some_number = 19 * (-x + y) - 21

-- State the proposition to prove
theorem find_number (h : equation 1 y Some_number) : Some_number = -59 :=
sorry

end find_number_l2128_212878


namespace sum_of_roots_of_equation_l2128_212828

theorem sum_of_roots_of_equation : 
  (∀ x, 5 = (x^3 - 2*x^2 - 8*x) / (x + 2)) → 
  (∃ x1 x2, (5 = x1) ∧ (5 = x2) ∧ (x1 + x2 = 4)) := 
by
  sorry

end sum_of_roots_of_equation_l2128_212828


namespace parabola_shifted_left_and_down_l2128_212885

-- Define the original parabolic equation
def original_parabola (x : ℝ) : ℝ := 2 * x ^ 2 - 1

-- Define the transformed parabolic equation
def transformed_parabola (x : ℝ) : ℝ := 2 * (x + 1) ^ 2 - 3

-- Theorem statement
theorem parabola_shifted_left_and_down :
  ∀ x : ℝ, transformed_parabola x = 2 * (x + 1) ^ 2 - 3 :=
by 
  -- Proof Left as an exercise.
  sorry

end parabola_shifted_left_and_down_l2128_212885
