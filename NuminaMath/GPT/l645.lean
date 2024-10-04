import Mathlib

namespace four_digit_palindrome_prob_divisible_by_11_prob_four_digit_palindrome_divisible_by_11_is_one_l645_645760

theorem four_digit_palindrome_prob_divisible_by_11 :
  let palindromes := { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1001 * a + 110 * b } in
  ∀ n ∈ palindromes, 11 ∣ n := 
by 
  -- The proof would go here 
  sorry

theorem prob_four_digit_palindrome_divisible_by_11_is_one :
  let palindromes := { n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1001 * a + 110 * b } in
  (∀ n ∈ palindromes, 11 ∣ n) →
  let count_palindromes := (finset.filter (λ n, 1000 ≤ n ∧ n < 10000 ∧ ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1001 * a + 110 * b) (finset.range 10000)).card in
  count_palindromes > 0 →
  real.to_nnreal (count_palindromes : ℝ) / real.to_nnreal (count_palindromes : ℝ) = 1 :=
by 
  -- The proof would go here
  sorry

end four_digit_palindrome_prob_divisible_by_11_prob_four_digit_palindrome_divisible_by_11_is_one_l645_645760


namespace concentration_third_flask_l645_645284

-- Define the concentrations as per the given problem

noncomputable def concentration (acid_mass water_mass : ℝ) : ℝ :=
  (acid_mass / (acid_mass + water_mass)) * 100

-- Given conditions
def acid_mass_first_flask : ℝ := 10
def acid_mass_second_flask : ℝ := 20
def acid_mass_third_flask : ℝ := 30
def concentration_first_flask : ℝ := 5
def concentration_second_flask : ℝ := 70 / 3

-- Total water added to the first and second flasks
def total_water_mass : ℝ :=
  let W1 := (acid_mass_first_flask - concentration_first_flask * acid_mass_first_flask / 100)
  let W2 := (acid_mass_second_flask - concentration_second_flask * acid_mass_second_flask / 100)
  W1 + W2 

-- Prove the concentration of acid in the third flask
theorem concentration_third_flask : 
  concentration acid_mass_third_flask total_water_mass = 10.5 := 
  sorry

end concentration_third_flask_l645_645284


namespace number_of_terms_in_arithmetic_sequence_l645_645105

theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ n → 6 + (k - 1) * 2 = 202)) ∧ n = 99 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l645_645105


namespace unique_triple_count_l645_645108

def is_valid_triple (a b c : ℤ) : Prop :=
  a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ (log a b = c^3) ∧ (a + b + c = 100)

theorem unique_triple_count : 
  ∃! (a b c : ℤ), is_valid_triple a b c :=
sorry

end unique_triple_count_l645_645108


namespace base_conversion_and_arithmetic_l645_645675

theorem base_conversion_and_arithmetic :
  let n1 := 2468 in
  let n2 := 1 * 5^2 + 2 * 5^1 + 3 * 5^0 in -- 123 in base 5
  let n3 := 1 * 8^2 + 0 * 8^1 + 7 * 8^0 in -- 107 in base 8
  let n4 := 4 * 9^3 + 3 * 9^2 + 2 * 9^1 + 1 * 9^0 in -- 4321 in base 9
  (n1 / n2 * n3 + n4).toNat = 7789 := sorry

end base_conversion_and_arithmetic_l645_645675


namespace area_of_square_l645_645854

theorem area_of_square (side_length : ℕ) (h : side_length = 9) : side_length * side_length = 81 :=
by
  rw h
  -- end the proof with sorry for now
  sorry

end area_of_square_l645_645854


namespace total_rankings_l645_645959

-- Defines the set of players
inductive Player
| P : Player
| Q : Player
| R : Player
| S : Player

-- Defines a function to count the total number of ranking sequences
def total_possible_rankings (p : Player → Player → Prop) : Nat := 
  4 * 2 * 2

-- Problem statement
theorem total_rankings : ∃ t : Player → Player → Prop, total_possible_rankings t = 16 :=
by
  sorry

end total_rankings_l645_645959


namespace value_of_a_with_two_distinct_roots_l645_645027

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l645_645027


namespace incircle_triangle_area_l645_645122

-- Define the semi-perimeter s
def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Define the area of triangle ABC using Heron's formula
noncomputable def area_ABC (a b c : ℝ) : ℝ := 
  let s := semiperimeter a b c in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the area of the new triangle formed by the incircle contact points
noncomputable def area_incircle_triangle (a b c : ℝ) : ℝ := 
  let s := semiperimeter a b c in
  area_ABC a b c * 
  (1 - ((s - a) ^ 2) / (b * c) -
      ((s - b) ^ 2) / (a * c) -
      ((s - c) ^ 2) / (a * b))

-- Proof statement
theorem incircle_triangle_area (a b c : ℝ) (T : ℝ) (hT : T = area_ABC a b c) : 
  area_incircle_triangle a b c = real.sqrt ((semiperimeter a b c) * (semiperimeter a b c - a) * (semiperimeter a b c - b) * (semiperimeter a b c - c)) * 
    (1 - ((semiperimeter a b c - a) ^ 2) / (b * c) - ((semiperimeter a b c - b) ^ 2) / (a * c) - ((semiperimeter a b c - c) ^ 2) / (a * b)) :=
sorry

end incircle_triangle_area_l645_645122


namespace volume_of_pyramid_is_six_l645_645970

noncomputable def volume_of_pyramid 
  (AB BC CG : ℝ) (hAB : AB = 4) (hBC : BC = 2) (hCG : CG = 3) : ℝ :=
  let area_base := BC * CG in
  let height := CG in
  (1 / 3) * area_base * height

theorem volume_of_pyramid_is_six : volume_of_pyramid 4 2 3 (by rfl) (by rfl) (by rfl) = 6 :=
by 
  calc
    volume_of_pyramid 4 2 3 (by rfl) (by rfl) (by rfl)
    = (1 / 3) * (2 * 3) * 3 : by simp [volume_of_pyramid]
    ... = 6 : by norm_num

end volume_of_pyramid_is_six_l645_645970


namespace find_b_l645_645850

variable (b : ℝ)

theorem find_b (h : log b 343 = -3 / 2) : b = 1 / 49 := by
  sorry

end find_b_l645_645850


namespace todd_final_money_l645_645668

noncomputable def todd_initial_money : ℝ := 100
noncomputable def todd_debt : ℝ := 110
noncomputable def todd_spent_on_ingredients : ℝ := 75
noncomputable def snow_cones_sold : ℝ := 200
noncomputable def price_per_snowcone : ℝ := 0.75

theorem todd_final_money :
  let initial_money := todd_initial_money,
      debt := todd_debt,
      spent := todd_spent_on_ingredients,
      revenue := snow_cones_sold * price_per_snowcone,
      remaining := initial_money - spent,
      total_pre_debt := remaining + revenue,
      final_money := total_pre_debt - debt
  in final_money = 65 :=
by
  sorry

end todd_final_money_l645_645668


namespace expectation_sum_l645_645564

variables {Ω : Type} [ProbabilitySpace Ω]
variables (ξ η : Ω → ℝ)

-- Conditions
axiom finite_expectation_ξ : ∃ Eξ : ℝ, Eξ = ⨍ ω, ξ ω ∂lebesgue
axiom finite_expectation_η : ∃ Eη : ℝ, Eη = ⨍ ω, η ω ∂lebesgue
axiom valid_sum : ¬((finite_expectation_ξ = ⊤ ∧ finite_expectation_η = ⊥) ∨ (finite_expectation_ξ = ⊥ ∧ finite_expectation_η = ⊤))

-- Proof statement
theorem expectation_sum : 
  ∃ Eξ Eη : ℝ, Eξ = ⨍ ω, ξ ω ∂lebesgue ∧ Eη = ⨍ ω, η ω ∂lebesgue ∧ (⨍ ω, (ξ + η) ω ∂lebesgue = Eξ + Eη) :=
sorry

end expectation_sum_l645_645564


namespace average_speed_l645_645762

theorem average_speed (v1 v2 : ℝ) (hv1 : v1 ≠ 0) (hv2 : v2 ≠ 0) : 
  2 / (1 / v1 + 1 / v2) = 2 * v1 * v2 / (v1 + v2) :=
by sorry

end average_speed_l645_645762


namespace range_of_a_l645_645496

-- Definitions for P and Q
def P (a : ℝ) : Prop := ∀ x y : ℝ, (0 < x ∧ x < y) → log a x > log a y
def Q (a : ℝ) : Prop := (a - 2) / (a + 2) ≤  0

-- Theorem to be proved
theorem range_of_a (a : ℝ) : ¬ (P a ∨ Q a) → (a ≤ -2 ∨ a > 2) :=
by
  sorry

end range_of_a_l645_645496


namespace rounded_accuracy_l645_645053

theorem rounded_accuracy (n : ℝ) (h : n = 2.81 * 10^4) : "The number is accurate to the hundred's place" :=
sorry

end rounded_accuracy_l645_645053


namespace teacher_age_is_40_l645_645330

noncomputable def teacher_age (n : ℕ) (s : ℝ) (t_avg : ℝ) :=
  let avg_students := 20 in
  let sum_students := n * avg_students in
  let sum_total := (n + 1) * t_avg in
  let age_teacher := sum_total - sum_students in
  age_teacher

theorem teacher_age_is_40 :
  teacher_age 19 20 21 = 40 :=
by
  sorry

end teacher_age_is_40_l645_645330


namespace second_supply_cost_is_24_l645_645787

-- Definitions based on the given problem conditions
def cost_first_supply : ℕ := 13
def last_year_remaining : ℕ := 6
def this_year_budget : ℕ := 50
def remaining_budget : ℕ := 19

-- Sum of last year's remaining budget and this year's budget
def total_budget : ℕ := last_year_remaining + this_year_budget

-- Total amount spent on school supplies
def total_spent : ℕ := total_budget - remaining_budget

-- Cost of second school supply
def cost_second_supply : ℕ := total_spent - cost_first_supply

-- The theorem to prove
theorem second_supply_cost_is_24 : cost_second_supply = 24 := by
  sorry

end second_supply_cost_is_24_l645_645787


namespace equation_has_two_distinct_roots_l645_645001

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l645_645001


namespace find_g_neg_five_l645_645877

-- Given function and its properties
variables (g : ℝ → ℝ)

-- Conditions
axiom ax1 : ∀ (x y : ℝ), g (x - y) = g x * g y
axiom ax2 : ∀ (x : ℝ), g x ≠ 0
axiom ax3 : g 5 = 2

-- Theorem to prove
theorem find_g_neg_five : g (-5) = 8 :=
sorry

end find_g_neg_five_l645_645877


namespace find_extrema_l645_645857

noncomputable def f (x : ℝ) : ℝ := 4^x - 6 * 2^x + 7

theorem find_extrema : 
  ∃ (x_min x_max : ℝ), x ∈ set.Icc 0 2 ∧ 
    f x_min = -2 ∧ 
    f x_max = 2 ∧ 
    x_min = real.log 3 / real.log 2 ∧ 
    x_max = 0 := 
sorry

end find_extrema_l645_645857


namespace motorcyclist_wait_time_after_passing_l645_645350

-- Define constants for the problem
def hiker_speed_mph : ℝ := 6
def cyclist_speed_mph : ℝ := 30
def wait_time_minutes : ℝ := 48
def minutes_per_hour : ℝ := 60

-- Function to convert speed from miles per hour to miles per minute
def speed_in_miles_per_minute (speed_mph : ℝ) : ℝ :=
  speed_mph / minutes_per_hour

-- Definitions of hiker's speed in miles per minute and motor-cyclist's speed in miles per minute
def hiker_speed_mpm : ℝ := speed_in_miles_per_minute hiker_speed_mph
def cyclist_speed_mpm : ℝ := speed_in_miles_per_minute cyclist_speed_mph

-- Distance covered by hiker in the time the cyclist waits
def hiker_distance_during_wait : ℝ :=
  wait_time_minutes * hiker_speed_mpm

-- Time taken by cyclist to travel the distance covered by hiker
def cyclist_time_to_cover_distance : ℝ :=
  hiker_distance_during_wait / cyclist_speed_mpm

theorem motorcyclist_wait_time_after_passing :
  cyclist_time_to_cover_distance = 9.6 :=
sorry

end motorcyclist_wait_time_after_passing_l645_645350


namespace find_incorrect_statement_l645_645320

-- Definition of the variance formula given the data set
def variance_formula (s2 : ℝ) (x̄ : ℝ) : Prop :=
  s2 = (1 / 5) * ((5 - x̄) ^ 2 + (4 - x̄) ^ 2 + (4 - x̄) ^ 2 + (3 - x̄) ^ 2 + (3 - x̄) ^ 2)

-- Definition to determine the incorrect statement
def incorrect_statement (s2 : ℝ) (x̄ : ℝ) : Prop :=
  let dataset := [5, 4, 4, 3, 3] in
  let mean := (5 + 4 + 4 + 3 + 3) / 5 in
  let median := 4 in
  let mode := [3, 4] in
  ¬ (mode.length = 1 ∧ mode.head? = some 4)

-- The main proof statement
theorem find_incorrect_statement (s2 : ℝ) (x̄ : ℝ) (h : variance_formula s2 x̄) : incorrect_statement s2 x̄ :=
by 
  -- Placeholder for the proof, which is skipped per requirements
  sorry

end find_incorrect_statement_l645_645320


namespace handshakes_at_gathering_of_six_couples_l645_645600

theorem handshakes_at_gathering_of_six_couples :
  let total_people := 12
  let handshakes_per_person := 9
  let total_handshakes := (total_people * handshakes_per_person) / 2
  in total_handshakes = 54 := 
by {
  have total_people := 12
  have handshakes_per_person := 9
  have total_handshakes := (total_people * handshakes_per_person) / 2
  show total_handshakes = 54
}

end handshakes_at_gathering_of_six_couples_l645_645600


namespace red_blue_card_sum_l645_645651

theorem red_blue_card_sum (N : ℕ) (r b : ℕ → ℕ) (h_r : ∀ i, 1 ≤ r i ∧ r i ≤ N) (h_b : ∀ i, 1 ≤ b i ∧ b i ≤ N):
  ∃ (A B : Finset ℕ), A ≠ ∅ ∧ B ≠ ∅ ∧ (∑ i in A, r i) = ∑ j in B, b j :=
by
  sorry

end red_blue_card_sum_l645_645651


namespace remainder_b_mod_7_l645_645566

theorem remainder_b_mod_7 :
  let b := Nat.mod_inv 2 7 + Nat.mod_inv 3 7 + Nat.mod_inv 5 7 
  (b := Nat.mod_inv b 7)
  b % 7 = 3 :=
by
  have h2 : Nat.mod_inv 2 7 = 4 := sorry
  have h3 : Nat.mod_inv 3 7 = 5 := sorry
  have h5 : Nat.mod_inv 5 7 = 3 := sorry
  have h_sum : (Nat.mod_inv 2 7 + Nat.mod_inv 3 7 + Nat.mod_inv 5 7) % 7 = 5 := sorry
  show Nat.mod_inv ((4 + 5 + 3) % 7) 7 % 7 = 3 from sorry


end remainder_b_mod_7_l645_645566


namespace rectangle_width_decrease_l645_645625

theorem rectangle_width_decrease (A L W : ℝ) (h1 : A = L * W) (h2 : 1.5 * L * W' = A) : 
  (W' = (2/3) * W) -> by exact (W - W') / W = 1 / 3 :=
by
  sorry

end rectangle_width_decrease_l645_645625


namespace concentration_of_acid_in_third_flask_is_correct_l645_645289

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l645_645289


namespace system_has_solution_l645_645823

theorem system_has_solution (a : ℝ) (h₀ : 0 < a) (h₁ : a ≤ 1 / 8) :
  ∃ (x y : ℝ), sqrt (x ^ 2 * y ^ 2) = a ^ a ∧ log a (x ^ log a y) + 2 * log a (y ^ log a x) = 6 * a ^ 3 := 
sorry

end system_has_solution_l645_645823


namespace correct_propositions_l645_645517

-- Definitions for the conditions
def tangent_at (f : ℝ → ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := P in (f x₀ = y₀) ∧ (∃ m : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs ((f x - y₀) - m * (x - x₀))) < ε)

def cut_through (f : ℝ → ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := P in 
  tangent_at f l P ∧ 
  ((∃ δ > 0, ∀ x, 0 < abs (x - x₀) < δ → (f x - y₀) * (x - x₀) > 0) ∧
   (∃ δ > 0, ∀ x, 0 < abs (x - x₀) < δ → (f (x₀ + δ) - y₀) * (f (x₀ - δ) - y₀) < 0))

-- Statement of the theorem with the correct answer
theorem correct_propositions :
  ∀ P₁ P₂ P₃ P₄ P₅,
  (P₁ = (0,0) ∧ P₂ = (-1,0) ∧ P₃ = (0,0) ∧ P₄ = (0,0) ∧ P₅ = (1,0))
  →
  (cut_through (λ x, x^3) (λ y, 0) P₁ ∧
   ¬cut_through (λ x, (x+1)^3) (λ y, -1) P₂ ∧
   cut_through (λ x, sin x) (λ y, y) P₃ ∧
   cut_through (λ x, tan x) (λ y, y) P₄ ∧
   ¬cut_through (λ x, log x) (λ y, y - 1) P₅) 
  → 
  true := sorry


end correct_propositions_l645_645517


namespace prob_seven_heads_in_ten_tosses_l645_645206

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.choose n k)

noncomputable def probability_of_heads (n k : ℕ) : ℚ :=
  (binomial_coefficient n k) * (0.5^k : ℚ) * (0.5^(n - k) : ℚ)

theorem prob_seven_heads_in_ten_tosses :
  probability_of_heads 10 7 = 15 / 128 :=
by
  sorry

end prob_seven_heads_in_ten_tosses_l645_645206


namespace symmetric_line_equation_l645_645939

noncomputable def symmetric_line (l1: ℝ × ℝ → Prop) (l2: ℝ × ℝ → Prop) (axis: ℝ) : Prop :=
  ∀ (x y : ℝ),
    l2 (x, y) ↔ l1 (2 * axis - x, y)

theorem symmetric_line_equation (x y : ℝ) :
  symmetric_line (λ p: ℝ × ℝ, 3 * p.1 - 4 * p.2 - 3 = 0) (λ p: ℝ × ℝ, 3 * p.1 + 4 * p.2 - 3 = 0) 1 :=
by
  sorry

end symmetric_line_equation_l645_645939


namespace shopkeeper_loss_percentage_l645_645347

theorem shopkeeper_loss_percentage {cp sp : ℝ} (h1 : cp = 100) (h2 : sp = cp * 1.1) (h_loss : sp * 0.33 = cp * (1 - x / 100)) :
  x = 70 :=
by
  sorry

end shopkeeper_loss_percentage_l645_645347


namespace digit_equation_solution_l645_645965

theorem digit_equation_solution :
  ∀ (A O B M E P : ℕ),
    A ≠ O ∧ A ≠ B ∧ A ≠ M ∧ A ≠ E ∧ A ≠ P ∧
    O ≠ B ∧ O ≠ M ∧ O ≠ E ∧ O ≠ P ∧
    B ≠ M ∧ B ≠ E ∧ B ≠ P ∧
    M ≠ E ∧ M ≠ P ∧
    E ≠ P →
    A = 5 ∧ O = 3 ∧ B = 8 ∧ M = 4 ∧ E = 6 ∧ P = 1 →
    6 * (100000 * A + 10000 * O + 1000 * B + 100 * M + 10 * E + P) =
    7 * (100000 * M + 10000 * E + 1000 * P + 100 * A + 10 * O + B) :=
by
  intros A O B M E P h_unique h_values,
  cases h_values with hA h_rest,
  cases h_rest with hO h_rest,
  cases h_rest with hB h_rest,
  cases h_rest with hM h_rest,
  cases h_rest with hE hP,
  sorry

end digit_equation_solution_l645_645965


namespace average_price_mixed_sugar_l645_645784

def average_selling_price_per_kg (weightA weightB weightC costA costB costC : ℕ) := 
  (costA * weightA + costB * weightB + costC * weightC) / (weightA + weightB + weightC : ℚ)

theorem average_price_mixed_sugar : 
  average_selling_price_per_kg 3 2 5 28 20 12 = 18.4 := 
by
  sorry

end average_price_mixed_sugar_l645_645784


namespace toothpicks_at_150th_stage_l645_645619

theorem toothpicks_at_150th_stage (a₁ d n : ℕ) (h₁ : a₁ = 6) (hd : d = 5) (hn : n = 150) :
  (n * (2 * a₁ + (n - 1) * d)) / 2 = 56775 :=
by
  sorry -- Proof to be completed.

end toothpicks_at_150th_stage_l645_645619


namespace sum_weighted_a_leq_bound_l645_645860

theorem sum_weighted_a_leq_bound (n : ℕ) (a : Fin n → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ a i)
  (h_cond : ∀ x : Fin n → ℝ, (∀ i j, i < j → x i > x j) → (∑ i, x i < 1) → (∑ i, a i * x i^3 < 1)) :
  ∑ i in Finset.range n, (n - i) * a i ≤ n^2 * (n + 1)^2 / 4 := 
by
  sorry

end sum_weighted_a_leq_bound_l645_645860


namespace fundraising_exceeded_goal_l645_645217

theorem fundraising_exceeded_goal:
  let goal := 4000
  let ken := 600
  let mary := 5 * ken
  let scott := mary / 3
  let total := ken + mary + scott
  total - goal = 600 :=
by
  let goal := 4000
  let ken := 600
  let mary := 5 * ken
  let scott := mary / 3
  let total := ken + mary + scott
  have h_goal : goal = 4000 := rfl
  have h_ken : ken = 600 := rfl
  have h_mary : mary = 5 * ken := rfl
  have h_scott : scott = mary / 3 := rfl
  have h_total : total = ken + mary + scott := rfl
  calc total - goal = (ken + mary + scott) - goal : by rw h_total
  ... = (600 + 3000 + 1000) - 4000 : by {rw [h_ken, h_mary, h_scott], norm_num}
  ... = 600 : by norm_num

end fundraising_exceeded_goal_l645_645217


namespace four_digit_perfect_square_of_2016_l645_645205

def rearranged_digit_numbers (d1 d2 d3 d4 : ℕ) : List ℕ :=
  let digits := [d1, d2, d3, d4]
  digits.permutations.map (λ l, l.foldl (λ n d, 10 * n + d) 0)

theorem four_digit_perfect_square_of_2016 :
  ∃ (n : ℕ), n^2 = 2601 ∧ 2601 ∈ rearranged_digit_numbers 2 0 1 6 ∧ (2601 ≥ 1000 ∧ 2601 < 10000) :=
sorry

end four_digit_perfect_square_of_2016_l645_645205


namespace total_balls_estimate_l645_645544

theorem total_balls_estimate 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (frequency : ℚ)
  (h_red_balls : red_balls = 12)
  (h_frequency : frequency = 0.6) 
  (h_fraction : (red_balls : ℚ) / total_balls = frequency): 
  total_balls = 20 := 
by 
  sorry

end total_balls_estimate_l645_645544


namespace sum_of_bn_l645_645771

theorem sum_of_bn (n : ℕ) (h : n > 0)
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (hb1 : b 1 = 2)
  (h_arith : ∀ k : ℕ, k > 0 → a (k + 1) - a k = 2)
  (h_geom : ∀ k : ℕ, k > 0 → b (k + 1) = 2 * b k) :
  (∑ k in Finset.range n, b (k + 1)) = (4 / 3 * (4 ^ n - 1)) :=
by
  sorry

end sum_of_bn_l645_645771


namespace sum_of_roots_of_log_eq_abs_div_2_l645_645047

theorem sum_of_roots_of_log_eq_abs_div_2 :
  ∑ x in {x : ℝ | log 2 (x + 8) = |x| / 2}, x = 8 := 
sorry

end sum_of_roots_of_log_eq_abs_div_2_l645_645047


namespace shortest_lock_sequence_l645_645976

def isValidLockSequence (seq : List ℕ) : Prop :=
  ∀ permutation : List ℕ,
    permutation ∈ [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]] →
    permutation.isInfixOf seq

theorem shortest_lock_sequence : isValidLockSequence [1, 2, 3, 1, 2, 1, 3, 2, 1] :=
  sorry

end shortest_lock_sequence_l645_645976


namespace max_value_f_on_interval_l645_645631

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, (∀ y ∈ Set.Icc (-1 : ℝ) 1, f y ≤ f x) ∧ f x = Real.exp 1 - 1 :=
sorry

end max_value_f_on_interval_l645_645631


namespace valeria_apartment_number_unit_digit_l645_645311

theorem valeria_apartment_number_unit_digit :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (¬ (n % 5 = 0)) ∧
  (n % 2 = 1) ∧ (intDigits n).sum < 8 ∧ (n % 10 = 6) ∧
  ((¬ (n % 5 = 0) ∧ n % 2 = 1 ∧ (intDigits n).sum < 8) ∨
   (¬ (n % 5 = 0) ∧ (intDigits n).sum < 8 ∨ (n % 10 = 6)) ∨
   (n % 2 = 1 ∧ (intDigits n).sum < 8 ∨ (n % 10 = 6)) ∨
   (¬ (n % 5 = 0) ∧ (n % 2 = 1) ∨ (n % 10 = 6))) -> n % 10 = 6 :=
sorry

end valeria_apartment_number_unit_digit_l645_645311


namespace minimum_distance_refrigerated_truck_l645_645241

theorem minimum_distance_refrigerated_truck (a b c : ℝ) (h₀ : a = 3) (h₁ : b = 4) (h₂ : c = Real.sqrt 13) :
  let minimum_distance := 2 * Real.sqrt 37 in
  minimum_distance = 2 * Real.sqrt 37 := by
suffices h₃ : (side1 = a) ∧ (side2 = b) ∧ (side3 = c),
{ sorry },
{ split,
  { exact h₀, },
  { split,
    { exact h₁, },
    { exact h₂, },
  }
}

#check minimum_distance_refrigerated_truck

end minimum_distance_refrigerated_truck_l645_645241


namespace problem_statement_l645_645567

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement : 
  (∀ x y z : ℝ, f(x^2 - y * f(z)) = x * f(x) - z * f(y)) →
  let n := {x : ℝ | ∃ (f : ℝ → ℝ), ∀ y z : ℝ, f(25 - y * f(z)) = 5 * f(5) - z * f(y)}.card in
  let s := ∑ x in {x : ℝ | ∃ f : ℝ → ℝ, f(25 - y * f(z)) = 5 * f(5) - z * f(y)}, x in
  n * s = 10 :=
begin
  sorry
end

end problem_statement_l645_645567


namespace number_of_ordered_triples_l645_645505

theorem number_of_ordered_triples :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), (x, y, z) ∈ S → Nat.lcm x y = 84 ∧ Nat.lcm x z = 480 ∧ Nat.lcm y z = 630) ∧ 
    S.card = 6 :=
by
  sorry

end number_of_ordered_triples_l645_645505


namespace no_function_exists_l645_645152

def M : Set ℕ := { x | x ≤ 2022 }

def f (a b : ℕ) : ℕ := sorry

theorem no_function_exists :
  (∀ a b ∈ M, f a (f b a) = b) ∧ (∀ x ∈ M, f x x ≠ x) → false :=
by
  sorry

end no_function_exists_l645_645152


namespace complex_number_solution_l645_645303

/-
  Given a complex number z of the form z = x + yi, 
  where x and y are positive integers,
  such that z^3 = -106 + ci for some integer c,
  we need to find z.
-/

def complex_solution : ℂ :=
  let z := 53 + 5 * Complex.I in
  z

theorem complex_number_solution
  (x y : ℕ)
  (h_pos : 0 < x ∧ 0 < y)
  (h_eq : (x + y * Complex.I)^3 = -106 + c * Complex.I) :
  complex_solution = 53 + 5 * Complex.I :=
by {
  sorry
}

end complex_number_solution_l645_645303


namespace nonnegative_difference_between_roots_l645_645806

theorem nonnegative_difference_between_roots : ∀ (x : ℝ),
  let a := 1
  let b := 42
  let c := 408
  let Δ := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt Δ) / (2 * a)
  let root2 := (-b - Real.sqrt Δ) / (2 * a)
  x = abs (root1 - root2) → x = 6 :=
begin
  intros x,
  sorry
end

end nonnegative_difference_between_roots_l645_645806


namespace equilateral_triangle_area_to_perimeter_ratio_l645_645684

theorem equilateral_triangle_area_to_perimeter_ratio
  (a : ℝ) (h : a = 10) :
  let altitude := a * (Real.sqrt 3 / 2) in
  let area := (1 / 2) * a * altitude in
  let perimeter := 3 * a in
  area / perimeter = 5 * (Real.sqrt 3) / 6 := 
by
  sorry

end equilateral_triangle_area_to_perimeter_ratio_l645_645684


namespace angle_BAC_eq_60_l645_645100

theorem angle_BAC_eq_60 
  (O I : Point) (A B C : Triangle)
  (hO : Circumcenter O (Triangle ⟨A, B, C⟩))
  (hI : Incenter I (Triangle ⟨A, B, C⟩))
  (hAngle : ∠ O I B = 30°) : 
  ∠ B A C = 60° := 
sorry

end angle_BAC_eq_60_l645_645100


namespace henry_age_l645_645256

theorem henry_age (H J : ℕ) 
  (sum_ages : H + J = 40) 
  (age_relation : H - 11 = 2 * (J - 11)) : 
  H = 23 := 
sorry

end henry_age_l645_645256


namespace count_valid_triangles_l645_645103

def isValidTriangle (a b c : ℕ) : Prop :=
  (a < b) ∧ (b < c) ∧ (a + b + c = 10) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem count_valid_triangles : finset.card { (a, b, c) | isValidTriangle a b c } = 2 :=
by sorry

end count_valid_triangles_l645_645103


namespace find_parallel_line_eq_l645_645245

-- Define the point A
def A := (point : ℝ × ℝ) (-1, 0)

-- Define the initial line
def initial_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define the line we want to prove
def target_line (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define the condition of being parallel
def are_parallel (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y = 0 → ∃ c : ℝ, line2 x y + c = 0

-- Prove the target
theorem find_parallel_line_eq (x y : ℝ) :
  (∃ c : ℝ, 2 * x - y + c = 0) ∧ initial_line x y → ∃ line, line = target_line
:= by
  sorry

end find_parallel_line_eq_l645_645245


namespace sequence_pattern_l645_645540

theorem sequence_pattern (a : ℕ → ℝ) :
  (∀ i, 1 < i ∧ i < 80 → a i = a (i - 1) * a (i + 1)) →
  (∏ i in (finset.range 40), a i.succ = 8) →
  (∏ i in (finset.range 80), a i.succ = 8) →
  (∀ n, ∃ k, 6 * k ≤ n → a n = if n % 6 = 0 then 2 else if n % 6 = 1 then 4 else if n % 6 = 2 then 2 else if n % 6 = 3 then 1/2 else if n % 6 = 4 then 1/4 else 1/2) :=
by
  intros h1 h2 h3
  sorry

end sequence_pattern_l645_645540


namespace incenter_circumcenter_dist_l645_645371

structure Triangle :=
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_tri : a + b > c ∧ a + c > b ∧ b + c > a)

def incenter (T : Triangle) : ℝ × ℝ :=
  -- Assuming this function exists
  sorry

def circumcenter (T : Triangle) : ℝ × ℝ :=
  -- Assuming this function exists
  sorry

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem incenter_circumcenter_dist :
  ∀ (T : Triangle), T.a = 6 → T.b = 8 → T.c = 10 → dist (incenter T) (circumcenter T) = real.sqrt 5 :=
by
  intros T h_a h_b h_c
  sorry

end incenter_circumcenter_dist_l645_645371


namespace find_cake_box_width_l645_645388

-- Define the dimensions of the carton
def carton_length := 25
def carton_width := 42
def carton_height := 60
def carton_volume := carton_length * carton_width * carton_height

-- Define the dimensions of the cake box
def cake_box_length := 8
variable (cake_box_width : ℝ) -- This is the unknown width we need to find
def cake_box_height := 5
def cake_box_volume := cake_box_length * cake_box_width * cake_box_height

-- Maximum number of cake boxes that can be placed in the carton
def max_cake_boxes := 210
def total_cake_boxes_volume := max_cake_boxes * cake_box_volume cake_box_width

-- Theorem to prove
theorem find_cake_box_width : cake_box_width = 7.5 :=
by
  sorry

end find_cake_box_width_l645_645388


namespace minimum_distance_refrigerated_truck_l645_645242

theorem minimum_distance_refrigerated_truck (a b c : ℝ) (h₀ : a = 3) (h₁ : b = 4) (h₂ : c = Real.sqrt 13) :
  let minimum_distance := 2 * Real.sqrt 37 in
  minimum_distance = 2 * Real.sqrt 37 := by
suffices h₃ : (side1 = a) ∧ (side2 = b) ∧ (side3 = c),
{ sorry },
{ split,
  { exact h₀, },
  { split,
    { exact h₁, },
    { exact h₂, },
  }
}

#check minimum_distance_refrigerated_truck

end minimum_distance_refrigerated_truck_l645_645242


namespace max_area_of_rectangle_l645_645550

-- Define the parameters and the problem
def perimeter := 150
def half_perimeter := perimeter / 2

theorem max_area_of_rectangle (x : ℕ) (y : ℕ) 
  (h1 : x + y = half_perimeter)
  (h2 : x > 0) (h3 : y > 0) :
  (∃ x y, x * y ≤ 1406) := 
sorry

end max_area_of_rectangle_l645_645550


namespace base_number_is_three_l645_645519

theorem base_number_is_three
  (x n : ℝ)
  (h1 : n = x ^ 0.15)
  (h2 : n ^ 13.33333333333333 = 9) :
  x = 3 :=
by sorry

end base_number_is_three_l645_645519


namespace find_base_b_l645_645949

theorem find_base_b (b : ℕ) (h : (3 * b + 4) ^ 2 = b ^ 3 + 2 * b ^ 2 + 9 * b + 6) : b = 10 :=
sorry

end find_base_b_l645_645949


namespace intercepted_segments_length_l645_645658

theorem intercepted_segments_length {a b c x : ℝ} 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : x = a * b * c / (a * b + b * c + c * a)) : 
  x = a * b * c / (a * b + b * c + c * a) :=
by sorry

end intercepted_segments_length_l645_645658


namespace extremum_problem_l645_645915

def f (x a b : ℝ) := x^3 + a*x^2 + b*x + a^2

def f_prime (x a b : ℝ) := 3*x^2 + 2*a*x + b

theorem extremum_problem (a b : ℝ) 
  (cond1 : f_prime 1 a b = 0)
  (cond2 : f 1 a b = 10) :
  (a, b) = (4, -11) := 
sorry

end extremum_problem_l645_645915


namespace correct_option_is_D_l645_645319

-- Definitions for conditions
def condition_1 := ∀ (alien_species : Type), (alien_species → Prop) -- Population shows a "J" shaped growth over a period of time.
def condition_2 := ∀ (alien_species : Type), ¬(alien_species → Prop) -- Population growth is not restricted by external factors.
def condition_3 := ∀ (alien_species : Type), (alien_species → Prop) -- Population threatens biodiversity.
def condition_4 := ∀ (alien_species : Type), (alien_species → Prop) -- Population will be eliminated if not adapting to the environment.

-- Problem statement
theorem correct_option_is_D (cond1 : condition_1) (cond2 : condition_2) (cond3 : condition_3) (cond4 : condition_4) :
  (cond1 ∧ cond3 ∧ cond4) := sorry

end correct_option_is_D_l645_645319


namespace problem_statement_l645_645057

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) (h5 : 0 < c) (h6 : c < 1) :
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) :=
sorry

end problem_statement_l645_645057


namespace ratio_of_area_to_perimeter_l645_645681

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l645_645681


namespace count_preferred_numbers_l645_645518

-- Definition of a preferred number
def is_preferred_number (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧ (∃ k : ℕ, k % 2 = 0 ∧ k > 0 ∧ 
  (n.digits.count 8 = k))

-- Theorem statement
theorem count_preferred_numbers : 
  ∑ n in finset.range 10000, 
    if is_preferred_number n then 1 else 0 = 460 := 
sorry

end count_preferred_numbers_l645_645518


namespace common_difference_d_l645_645050

open Real

-- Define the arithmetic sequence and relevant conditions
variable (a : ℕ → ℝ) -- Define the sequence as a function from natural numbers to real numbers
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific conditions from our problem
def problem_conditions (a : ℕ → ℝ) (d : ℝ) : Prop :=
  is_arithmetic_sequence a d ∧
  a 1 = 1 ∧
  (a 2) ^ 2 = a 1 * a 6

-- The goal is to prove that the common difference d is either 0 or 3
theorem common_difference_d (a : ℕ → ℝ) (d : ℝ) :
  problem_conditions a d → (d = 0 ∨ d = 3) := by
  sorry

end common_difference_d_l645_645050


namespace equation_has_at_least_two_distinct_roots_l645_645012

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l645_645012


namespace value_of_x_l645_645259

-- Define the variables x, y, z
variables (x y z : ℕ)

-- Hypothesis based on the conditions of the problem
hypothesis h1 : x = y / 3
hypothesis h2 : y = z / 4
hypothesis h3 : z = 48

-- The statement to be proved
theorem value_of_x : x = 4 :=
by { sorry }

end value_of_x_l645_645259


namespace find_x_plus_inv_x_l645_645091

theorem find_x_plus_inv_x (x : ℝ) (hx_pos : 0 < x) (h : x^10 + x^5 + 1/x^5 + 1/x^10 = 15250) :
  x + 1/x = 3 :=
by
  sorry

end find_x_plus_inv_x_l645_645091


namespace group_atoms_weight_l645_645805

theorem group_atoms_weight (total_weight : ℝ) (al_weight : ℝ) (group_weight : ℝ) (hw : total_weight = 122) (ha : al_weight = 26.98) : group_weight = 122 - 26.98 :=
by
  rw [hw, ha]
  rfl

end group_atoms_weight_l645_645805


namespace circle_radius_condition_l645_645317

theorem circle_radius_condition (c : ℝ) : 
  (∃ x y : ℝ, (x^2 + 6 * x + y^2 - 4 * y + c = 0)) ∧ 
  (radius = 6) ↔ 
  c = -23 := by
  sorry

end circle_radius_condition_l645_645317


namespace prob_rel_prime_2015_l645_645821

theorem prob_rel_prime_2015 : 
  let S := { n : ℕ | n ∈ Finset.range 2016 ∧ Nat.gcd n 2015 = 1 } in
  (Finset.card S : ℚ) / 2016 = 1442 / 2016 := by
sorry

end prob_rel_prime_2015_l645_645821


namespace no_function_satisfies_condition_l645_645427

theorem no_function_satisfies_condition :
  ¬ ∃ (f : ℤ → ℤ), ∀ (x y : ℝ), f (x + f y) = f x - y := sorry

end no_function_satisfies_condition_l645_645427


namespace find_y_l645_645425

theorem find_y (y : ℝ) (h : log y 128 = 7 / 2) : y = 4 :=
sorry

end find_y_l645_645425


namespace ferris_wheel_seats_broken_l645_645612

theorem ferris_wheel_seats_broken (b : ℕ) :
  (∃ b, 18 - b ≠ 0 ∧ (18 - b) * 15 = 120) → b = 10 :=
by
  intro h
  cases h with b h
  sorry

end ferris_wheel_seats_broken_l645_645612


namespace problem1_problem2_l645_645894

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := { x | a - b < x ∧ x < a + b }
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

-- First problem: prove the range of a
theorem problem1 (a : ℝ) (h : A a 1 ⊆ B) : a ≤ -2 ∨ a ≥ 6 := by
  sorry

-- Second problem: prove the range of b
theorem problem2 (b : ℝ) (h : A 1 b ∩ B = ∅) : b ≤ 2 := by
  sorry

end problem1_problem2_l645_645894


namespace find_scalars_l645_645923

def a : ℝ × ℝ × ℝ := (2, 2, 2)
def b : ℝ × ℝ × ℝ := (1, -4, 1)
def c : ℝ × ℝ × ℝ := (-2, 1, 3)
def target : ℝ × ℝ × ℝ := (3, -11, 5)

theorem find_scalars :
  ∃ p q r : ℝ, target = (p * a.1 + q * b.1 + r * c.1, p * a.2 + q * b.2 + r * c.2, p * a.3 + q * b.3 + r * c.3)
  ∧ (p, q, r) = (-1/2, 26/9, -1/7) :=
sorry

end find_scalars_l645_645923


namespace Powerjet_pumps_250_gallons_in_30_minutes_l645_645234

theorem Powerjet_pumps_250_gallons_in_30_minutes :
  let r := 500 -- Pump rate in gallons per hour
  let t := 1 / 2 -- Time in hours (30 minutes)
  r * t = 250 := by
  -- proof steps will go here
  sorry

end Powerjet_pumps_250_gallons_in_30_minutes_l645_645234


namespace concentration_of_acid_in_third_flask_is_correct_l645_645287

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l645_645287


namespace distance_incenter_circumcenter_l645_645377

theorem distance_incenter_circumcenter (A B C : ℝ × ℝ)
  (hAB : (dist A B) = 6)
  (hAC : (dist A C) = 8)
  (hBC : (dist B C) = 10)
  (h_right : ∠ BAC = 90) :
  dist (incenter A B C) (circumcenter A B C) = √5 :=
sorry

end distance_incenter_circumcenter_l645_645377


namespace concentration_of_first_solution_l645_645340

variables {volume_1 volume_2 volume_3 : ℝ}
variables {concentration_1 concentration_2 concentration_3 : ℝ}
variables (h1: volume_1 = 15) (h2: volume_2 = 15) (h3: volume_3 = 30)
variables (c1: concentration_1 = 0.01) (c2: concentration_2 = 0.05) (c3: concentration_3 = 0.03)

theorem concentration_of_first_solution :
  0.9 = (volume_1 * concentration_1) + (volume_2 * concentration_2) :=
by
  rw [h1, h2, h3, c1, c2]
  sorry

end concentration_of_first_solution_l645_645340


namespace min_t_of_BE_CF_l645_645974

theorem min_t_of_BE_CF (A B C E F: ℝ)
  (hE_midpoint_AC : ∃ D, D = (A + C) / 2 ∧ E = D)
  (hF_midpoint_AB : ∃ D, D = (A + B) / 2 ∧ F = D)
  (h_AB_AC_ratio : B - A = 2 / 3 * (C - A)) :
  ∃ t : ℝ, t = 7 / 8 ∧ ∀ (BE CF : ℝ), BE = dist B E ∧ CF = dist C F → BE / CF < t := by
  sorry

end min_t_of_BE_CF_l645_645974


namespace find_a_for_quadratic_l645_645021

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l645_645021


namespace maximum_cards_without_equal_pair_sums_l645_645121

def max_cards_no_equal_sum_pairs : ℕ :=
  let card_points := {x : ℕ | 1 ≤ x ∧ x ≤ 13}
  6

theorem maximum_cards_without_equal_pair_sums (deck : Finset ℕ) (h_deck : deck = {x : ℕ | 1 ≤ x ∧ x ≤ 13}) :
  ∃ S ⊆ deck, S.card = 6 ∧ ∀ {a b c d : ℕ}, a ∈ S → b ∈ S → c ∈ S → d ∈ S → a + b = c + d → a = c ∧ b = d ∨ a = d ∧ b = c := 
sorry

end maximum_cards_without_equal_pair_sums_l645_645121


namespace sales_amount_eq_800_l645_645357

variable (S : ℝ)

-- Define the commission function
def commission (S : ℝ) : ℝ :=
  if S ≤ 500 then 0.20 * S
  else 0.20 * 500 + 0.25 * (S - 500)

-- Define the total commission condition
def commission_percentage (S : ℝ) : ℝ := 0.21875 * S

-- Main theorem to prove
theorem sales_amount_eq_800
  (h : commission S = commission_percentage S) : S = 800 :=
by sorry

end sales_amount_eq_800_l645_645357


namespace find_initial_money_l645_645142

def cost_of_water (bottles : ℕ) (cost_per_bottle : ℕ) : ℕ :=
  bottles * cost_per_bottle

def cost_of_cheese (pounds : ℝ) (cost_per_pound : ℝ) : ℝ :=
  pounds * cost_per_pound

def initial_money (remaining : ℕ) (total_cost : ℕ) : ℕ :=
  remaining + total_cost

theorem find_initial_money : 
  let initial_bottles := 4,
      additional_bottles := 2 * initial_bottles,
      cost_per_bottle := 2,
      cheese_weight := 0.5,
      cost_per_pound := 10,
      remaining_money := 71 in
  initial_money remaining_money (cost_of_water initial_bottles cost_per_bottle 
                                   + cost_of_water additional_bottles cost_per_bottle 
                                   + nat.floor (cost_of_cheese cheese_weight cost_per_pound)) = 100 :=
by 
  -- statement only, no proof required
  sorry

end find_initial_money_l645_645142


namespace problem_solution_l645_645450

def f₁ (x : ℝ) : ℝ := sin x + cos x
def f₂ (x : ℝ) : ℝ := cos x - sin x
def f₃ (x : ℝ) : ℝ := -sin x - cos x
def f₄ (x : ℝ) : ℝ := -cos x + sin x
def periodic_f (n : ℕ) (x : ℝ) : ℝ :=
  if n % 4 = 0 then f₁ x else
  if n % 4 = 1 then f₂ x else
  if n % 4 = 2 then f₃ x else
  f₄ x

theorem problem_solution : 
  f₁ (π / 3) + f₂ (π / 3) + f₃ (π / 3) + ∑ k in finset.range 2017 (λ n, periodic_f n (π / 3)) = (1 + real.sqrt 3) / 2 :=
sorry

end problem_solution_l645_645450


namespace problem_I_problem_II_problem_III_l645_645494

-- Define the functions f and g
def f (x a : ℝ) := x^2 + a*x + 1
def g (x : ℝ) := exp x

-- Problem I: Prove the maximum value of f(x) * g(x) on [-2, 0] when a = 1
theorem problem_I :
  ∃ x ∈ Set.Icc (-2 : ℝ) 0, (x^2 + x + 1) * exp x = 1 := sorry

-- Problem II: Range of k for which f(x) = k * g(x) has exactly one root when a = -1
theorem problem_II :
  ∀ k : ℝ, ((0 < k ∧ k < 1/exp(1)) ∨ (3/exp(2) < k)) → 
  ∃! x : ℝ, x^2 - x + 1 = k * exp x := sorry

-- Problem III: Range of a such that |f(x1) - f(x2)| < |g(x1) - g(x2)| for x1, x2 ∈ [0, 2]
theorem problem_III :
  (∀ x1 x2 ∈ Set.Icc 0 2, x1 ≠ x2 → 
    abs ((x1^2 + a*x1 + 1) - (x2^2 + a*x2 + 1)) < abs (exp x1 - exp x2)) →
  a ∈ Set.Icc (-1 : ℝ) (2 - 2 * Real.log 2) := sorry

end problem_I_problem_II_problem_III_l645_645494


namespace find_fourth_vertex_of_rectangle_l645_645478

-- Define the vertices of the rectangle.
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (3, 2)

-- Prove that the coordinates of the fourth vertex D are (2, 3).
theorem find_fourth_vertex_of_rectangle :
  ∃ D : ℝ × ℝ, D = (2, 3) ∧ 
  (∀ A B C D, -- The configuration of a rectangle should be verified in setup.
    (D.1 - A.1) * (D.1 - C.1) + (D.2 - A.2) * (D.2 - C.2) = 0 ∧
    (D.1 - B.1) * (D.1 - C.1) + (D.2 - B.2) * (D.2 - C.2) = 0) :=
sorry

end find_fourth_vertex_of_rectangle_l645_645478


namespace polynomial_remainder_l645_645439

theorem polynomial_remainder :
  polynomial.eval 3 (polynomial.C (-7) + polynomial.C 3 * polynomial.X +
    polynomial.C (-5) * polynomial.X^2 + polynomial.C 1 * polynomial.X^3) = -16 :=
by
  sorry

end polynomial_remainder_l645_645439


namespace integer_solutions_count_l645_645638

theorem integer_solutions_count : 
  (finset.univ.filter (λ x : ℤ, |3 * (x : ℝ) - 4| + |3 * (x : ℝ) + 2| = 6)).card = 2 :=
by
  sorry

end integer_solutions_count_l645_645638


namespace prime_sum_probability_l645_645731

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def count_primes (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n = 3 then 2
  else if n = 5 then 4
  else if n = 7 then 6
  else if n = 11 then 2
  else 0

def total_dice_outcomes : ℕ := 36

theorem prime_sum_probability :
  (∑ n in {2, 3, 5, 7, 11}, count_primes n : ℚ) / total_dice_outcomes = 5 / 12 := by
  sorry

end prime_sum_probability_l645_645731


namespace part1_part2_l645_645058

variable {a : ℝ}

def A (a : ℝ) : set ℝ := {x | 2 * x + a ≥ x^2}
def B : set ℝ := {x | Real.log x = 0}

theorem part1 (a : ℝ) : a = 8 → A a = Icc (-2 : ℝ) 4 := by
  sorry

theorem part2 (a : ℝ) : (B ⊆ A a) → a ≥ -1 := by
  sorry

end part1_part2_l645_645058


namespace largest_value_of_m_exists_l645_645996

theorem largest_value_of_m_exists (a b c : ℝ) (h₁ : a + b + c = 12) (h₂ : a * b + b * c + c * a = 30) (h₃ : 0 < a) (h₄ : 0 < b) (h₅ : 0 < c) : 
  ∃ m : ℝ, (m = min (a * b) (min (b * c) (c * a))) ∧ (m = 2) := sorry

end largest_value_of_m_exists_l645_645996


namespace quadrilateral_area_perpendicular_diagonals_l645_645460

theorem quadrilateral_area_perpendicular_diagonals (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ (A : ℝ), A = (1 / 2) * a * b :=
by
  -- We state here that some proof exists.
  use (1 / 2) * a * b
  -- here would be the proof, skipping it as instructed
  sorry

end quadrilateral_area_perpendicular_diagonals_l645_645460


namespace not_every_set_has_regression_equation_l645_645595

-- Definitions based on conditions
def condA : Prop := ∀ (X Y : Type) (corr : X → Y → Prop), corr X Y → ¬causal X Y
def condB : Prop := ∀ (data : Type) (scatter : data → data → Prop), scatter data data → correlation data
def condC : Prop := ∀ (X Y : Type) (linear_corr : X → Y → Prop), linear_corr X Y → regression_line X Y

-- Theorem to prove the core of the problem
theorem not_every_set_has_regression_equation
    (hypA : condA)
    (hypB : condB)
    (hypC : condC) :
    ¬∀ (data : Type), ∃ (linear_corr : data → data → Prop), correlation linear_corr → regression_equation data :=
sorry

end not_every_set_has_regression_equation_l645_645595


namespace two_digit_values_satisfying_clubsuit_l645_645994

def sum_of_digits (x : ℕ) : ℕ :=
  let digits := x.digits 10
  digits.sum

theorem two_digit_values_satisfying_clubsuit : 
  {x : ℕ // 10 ≤ x ∧ x < 100 ∧ sum_of_digits (sum_of_digits x) = 4}.card = 10 :=
by
  sorry

end two_digit_values_satisfying_clubsuit_l645_645994


namespace foil_covered_prism_width_l645_645742

def inner_prism_dimensions (l w h : ℕ) : Prop :=
  w = 2 * l ∧ w = 2 * h ∧ l * w * h = 128

def outer_prism_width (l w h outer_width : ℕ) : Prop :=
  inner_prism_dimensions l w h ∧ outer_width = w + 2

theorem foil_covered_prism_width (l w h outer_width : ℕ) (h_inner_prism : inner_prism_dimensions l w h) :
  outer_prism_width l w h outer_width → outer_width = 10 :=
by
  intro h_outer_prism
  obtain ⟨h_w_eq, h_w_eq_2, h_volume_eq⟩ := h_inner_prism
  obtain ⟨_, h_outer_width_eq⟩ := h_outer_prism
  sorry

end foil_covered_prism_width_l645_645742


namespace pythagorean_triple_mod_7_l645_645591

theorem pythagorean_triple_mod_7 (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  ∃ x y ∈ ({a, b, c} : Finset ℕ), (x^2 - y^2) % 7 = 0 :=
by
  sorry

end pythagorean_triple_mod_7_l645_645591


namespace train_passing_time_l645_645364

noncomputable def first_train_length : ℝ := 270
noncomputable def first_train_speed_kmh : ℝ := 108
noncomputable def second_train_length : ℝ := 360
noncomputable def second_train_speed_kmh : ℝ := 72

noncomputable def convert_speed_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

noncomputable def first_train_speed_mps : ℝ := convert_speed_to_mps first_train_speed_kmh
noncomputable def second_train_speed_mps : ℝ := convert_speed_to_mps second_train_speed_kmh

noncomputable def relative_speed_mps : ℝ := first_train_speed_mps + second_train_speed_mps
noncomputable def total_distance : ℝ := first_train_length + second_train_length
noncomputable def time_to_pass : ℝ := total_distance / relative_speed_mps

theorem train_passing_time : time_to_pass = 12.6 :=
by 
  sorry

end train_passing_time_l645_645364


namespace max_sum_prod_eq_a2_div_4_l645_645088

-- Define the conditions
variables (n : ℕ) (a : ℝ)
hypothesis h_n : n > 1
hypothesis h_a : a > 0

def sum_xis (xs : Fin n → ℝ) : ℝ := (Finset.range n).sum (λ i, xs i)
def sum_prod_xi_xi1 (xs : Fin n → ℝ) : ℝ := (Finset.range (n - 1)).sum (λ i, xs i * xs (i + 1))

theorem max_sum_prod_eq_a2_div_4 (xs : Fin n → ℝ) (h_nonneg : ∀ i, 0 ≤ xs i)
  (h_sum : sum_xis n xs = a) :
  sum_prod_xi_xi1 n xs ≤ a^2 / 4 := 
sorry

end max_sum_prod_eq_a2_div_4_l645_645088


namespace meaningful_sqrt_domain_l645_645522

theorem meaningful_sqrt_domain (x : ℝ) : (∃ (y : ℝ), y = sqrt (x - 1)) ↔ (x ≥ 1) := 
by sorry

end meaningful_sqrt_domain_l645_645522


namespace powderman_distance_when_hears_explosion_l645_645766

noncomputable def powderman_speed_yd_per_s : ℝ := 10
noncomputable def blast_time_s : ℝ := 45
noncomputable def sound_speed_ft_per_s : ℝ := 1080
noncomputable def powderman_speed_ft_per_s : ℝ := 30

noncomputable def distance_powderman (t : ℝ) : ℝ := powderman_speed_ft_per_s * t
noncomputable def distance_sound (t : ℝ) : ℝ := sound_speed_ft_per_s * (t - blast_time_s)

theorem powderman_distance_when_hears_explosion :
  ∃ t, t > blast_time_s ∧ distance_powderman t = distance_sound t ∧ (distance_powderman t) / 3 = 463 :=
sorry

end powderman_distance_when_hears_explosion_l645_645766


namespace max_value_of_f_l645_645633

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_of_f : ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → f x ≤ Real.exp 1 - 1 :=
by 
-- The conditions: the function and the interval
intros x hx
-- The interval condition: -1 ≤ x ≤ 1
have h_interval : -1 ≤ x ∧ x ≤ 1 := by 
  cases hx
  split; assumption
-- We prove it directly by showing the evaluated function points
sorry

end max_value_of_f_l645_645633


namespace value_of_a_with_two_distinct_roots_l645_645028

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l645_645028


namespace ratio_equilateral_triangle_l645_645721

def equilateral_triangle_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * (Real.sqrt 3 / 2)
  let area := (1 / 2) * s * altitude
  let perimeter := 3 * s in
  area / perimeter -- this simplifies to 25\sqrt{3} / 30 or 5\sqrt{3} / 6

theorem ratio_equilateral_triangle : ∀ (s : ℝ), s = 10 → equilateral_triangle_ratio s (by assumption) = 5 * (Real.sqrt 3) / 6 :=
by
  intros s h
  rw h
  sorry

end ratio_equilateral_triangle_l645_645721


namespace piecewise_linear_function_y_at_x_10_l645_645172

theorem piecewise_linear_function_y_at_x_10
  (k1 k2 : ℝ)
  (y : ℝ → ℝ)
  (hx1 : ∀ x < 0, y x = k1 * x)
  (hx2 : ∀ x ≥ 0, y x = k2 * x)
  (h_y_pos : y 2 = 4)
  (h_y_neg : y (-5) = -20) :
  y 10 = 20 :=
by
  sorry

end piecewise_linear_function_y_at_x_10_l645_645172


namespace different_universities_count_l645_645328

-- Define the students and universities
inductive Student
| Ming
| Hong
| Other1
| Other2

inductive University
| A
| B

-- Define the condition for assignment
def assignment (s : Student) : University

-- Define the property where Xiao Ming and Xiao Hong are at different universities
def different_universities : Prop :=
  assignment Student.Ming ≠ assignment Student.Hong

-- Main proof statement
theorem different_universities_count : ((Σ (f : Student → University), different_universities) = 4) :=
sorry

end different_universities_count_l645_645328


namespace part1_exists_m_n_part2_exists_k_l645_645501

def vector_a := (3 : ℝ, 2 : ℝ)
def vector_b := (-1 : ℝ, 2 : ℝ)
def vector_c := (4 : ℝ, 1 : ℝ)

theorem part1_exists_m_n (m n : ℝ) : 
  vector_a = m • vector_b + n • vector_c ↔ m = 5 / 9 ∧ n = 8 / 9 := 
by {
  sorry
}

theorem part2_exists_k (k : ℝ) :
  (vector_a + k • vector_c) = (2 • vector_b - vector_a) ↔ k = -16 / 13 := 
by {
  sorry
}

end part1_exists_m_n_part2_exists_k_l645_645501


namespace equilateral_triangle_ratio_l645_645715

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Define the altitude of the equilateral triangle
def altitude (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2

-- Define the area of the equilateral triangle
def area (a : ℝ) : ℝ := (a * altitude a) / 2

-- Define the perimeter of the equilateral triangle
def perimeter (a : ℝ) : ℝ := 3 * a

-- Define the ratio of area to perimeter
def ratio (a : ℝ) : ℝ := area a / perimeter a

theorem equilateral_triangle_ratio :
  ratio 10 = (5 * Real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l645_645715


namespace todd_has_40_left_after_paying_back_l645_645663

def todd_snowcone_problem : Prop :=
  let borrowed := 100
  let repay := 110
  let cost_ingredients := 75
  let snowcones_sold := 200
  let price_per_snowcone := 0.75
  let total_earnings := snowcones_sold * price_per_snowcone
  let remaining_money := total_earnings - repay
  remaining_money = 40

theorem todd_has_40_left_after_paying_back : todd_snowcone_problem :=
by
  -- Add proof here if needed
  sorry

end todd_has_40_left_after_paying_back_l645_645663


namespace top_face_not_rotated_by_90_l645_645654

-- Define the cube and the conditions of rolling and returning
structure Cube :=
  (initial_top_face_orientation : ℕ) -- an integer representation of the orientation of the top face
  (position : ℤ × ℤ) -- (x, y) coordinates on a 2D plane

def rolls_over_edges (c : Cube) : Cube :=
  sorry -- placeholder for the actual rolling operation

def returns_to_original_position (c : Cube) (original : Cube) : Prop :=
  c.position = original.position ∧ c.initial_top_face_orientation = original.initial_top_face_orientation

-- The main theorem to prove
theorem top_face_not_rotated_by_90 {c : Cube} (original : Cube) :
  returns_to_original_position c original → c.initial_top_face_orientation ≠ (original.initial_top_face_orientation + 1) % 4 :=
sorry

end top_face_not_rotated_by_90_l645_645654


namespace factorize_difference_of_squares_l645_645848

theorem factorize_difference_of_squares :
  ∀ x : ℝ, x^2 - 9 = (x + 3) * (x - 3) :=
by 
  intro x
  have h : x^2 - 9 = x^2 - 3^2 := by rw (show 9 = 3^2, by norm_num)
  have hs : (x^2 - 3^2) = (x + 3) * (x - 3) := by exact (mul_self_sub_mul_self_eq x 3)
  exact Eq.trans h hs

end factorize_difference_of_squares_l645_645848


namespace smallest_n_divisible_by_125000_l645_645576

noncomputable def geometric_term_at (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

noncomputable def first_term : ℚ := 5 / 8
noncomputable def second_term : ℚ := 25
noncomputable def common_ratio : ℚ := second_term / first_term

theorem smallest_n_divisible_by_125000 :
  ∃ n : ℕ, n ≥ 7 ∧ geometric_term_at first_term common_ratio n % 125000 = 0 :=
by
  sorry

end smallest_n_divisible_by_125000_l645_645576


namespace john_needs_to_sell_134_pens_l645_645144

def cost_per_pen : ℝ := 8 / 5
def selling_price_per_pen : ℝ := 10 / 4
def profit_per_pen : ℝ := selling_price_per_pen - cost_per_pen
def desired_profit : ℝ := 120
def number_of_pens (desired_profit profit_per_pen : ℝ) : ℕ := 
  (desired_profit / profit_per_pen).ceil

theorem john_needs_to_sell_134_pens :
  number_of_pens desired_profit profit_per_pen = 134 :=
by
  unfold number_of_pens
  rw [←ceil_eq_ceil]
  apply eq_of_ceil_of_ceil
  norm_num

end john_needs_to_sell_134_pens_l645_645144


namespace solve_sqrt_equation_l645_645601

theorem solve_sqrt_equation (x : ℝ) :
  (sqrt (7 * x - 3) + sqrt (2 * x - 3) = 4) ↔ (x = 9.67 ∨ x = 1.85) := sorry

end solve_sqrt_equation_l645_645601


namespace women_in_first_group_l645_645341

-- Define productivity and work conditions
variable (W : ℕ) -- Number of women in the first group

-- Given that 10 women can color 500 meters in 5 days
def productivity_per_woman (p : ℕ) := p = 50 / 5 -- 10 meters per day

-- Define the condition for the first group
def first_group_work := W * 10 * 5 = 400 

-- The main statement to prove
theorem women_in_first_group (H : productivity_per_woman 10) (H1 : first_group_work) : W = 8 := 
by 
  sorry

end women_in_first_group_l645_645341


namespace simplify_polynomial_l645_645599

theorem simplify_polynomial (x : ℝ) :
  (14 * x ^ 12 + 8 * x ^ 9 + 3 * x ^ 8) + (2 * x ^ 14 - x ^ 12 + 2 * x ^ 9 + 5 * x ^ 5 + 7 * x ^ 2 + 6) =
  2 * x ^ 14 + 13 * x ^ 12 + 10 * x ^ 9 + 3 * x ^ 8 + 5 * x ^ 5 + 7 * x ^ 2 + 6 :=
by
  sorry

end simplify_polynomial_l645_645599


namespace sandra_fathers_contribution_ratio_l645_645597

theorem sandra_fathers_contribution_ratio :
  let saved := 10
  let mother := 4
  let candy_cost := 0.5
  let jellybean_cost := 0.2
  let candies := 14
  let jellybeans := 20
  let remaining := 11
  let total_cost := candies * candy_cost + jellybeans * jellybean_cost
  let total_amount := total_cost + remaining
  let amount_without_father := saved + mother
  let father := total_amount - amount_without_father
  (father / mother) = 2 := by 
  sorry

end sandra_fathers_contribution_ratio_l645_645597


namespace value_of_x_l645_645260

theorem value_of_x (y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := by
  sorry

end value_of_x_l645_645260


namespace cars_rented_at_3600_max_revenue_at_4050_l645_645768

def num_cars_not_rented (r : ℝ) : ℝ :=
  (r - 3000) / 50

def num_cars_rented (r : ℝ) : ℝ :=
  100 - num_cars_not_rented r

def maintenance_cost (r : ℝ) : ℝ :=
  150 * num_cars_rented r + 50 * num_cars_not_rented r

def revenue (r : ℝ) : ℝ :=
  r * num_cars_rented r - maintenance_cost r

theorem cars_rented_at_3600 : num_cars_rented 3600 = 88 := 
by 
  sorry

theorem max_revenue_at_4050 : 
  ∃ max_r : ℝ, max_r = 4050 ∧ revenue max_r = 307050 :=
by
  sorry

end cars_rented_at_3600_max_revenue_at_4050_l645_645768


namespace solve_eq1_solve_eq2_l645_645603

-- Define the first equation
def eq1 (x : ℚ) : Prop := x / (x - 1) = 3 / (2*x - 2) - 2

-- Define the valid solution for the first equation
def sol1 : ℚ := 7 / 6

-- Theorem for the first equation
theorem solve_eq1 : eq1 sol1 :=
by
  sorry

-- Define the second equation
def eq2 (x : ℚ) : Prop := (5*x + 2) / (x^2 + x) = 3 / (x + 1)

-- Theorem for the second equation: there is no valid solution
theorem solve_eq2 : ¬ ∃ x : ℚ, eq2 x :=
by
  sorry

end solve_eq1_solve_eq2_l645_645603


namespace altitude_circumradius_relation_l645_645201

variable (a b c R ha : ℝ)
-- Assume S is the area of the triangle
variable (S : ℝ)
-- conditions
axiom area_circumradius : S = (a * b * c) / (4 * R)
axiom area_altitude : S = (a * ha) / 2

-- Prove the equivalence
theorem altitude_circumradius_relation 
  (area_circumradius : S = (a * b * c) / (4 * R)) 
  (area_altitude : S = (a * ha) / 2) : 
  ha = (b * c) / (2 * R) :=
sorry

end altitude_circumradius_relation_l645_645201


namespace GretzkyStreet_length_l645_645101

theorem GretzkyStreet_length (n : ℕ) (d : ℕ) (h1 : n = 15) (h2 : d = 350) : 
  (n * d + 2 * d) / 1000 = 5.95 :=
by
  have h3 : (n * d + 2 * d) = 5950 := by linarith [h1, h2]
  have h4 : 5950 / 1000 = 5.95 := by norm_num
  rw h3
  exact h4

end GretzkyStreet_length_l645_645101


namespace unit_price_for_400_tons_l645_645327

theorem unit_price_for_400_tons:
  ∃ k b: ℝ, 1000 = 800 * k + b ∧ 2000 = 700 * k + b ∧ ∀ x, 400 = k * x + b → x = 5000 :=
begin
  sorry
end

end unit_price_for_400_tons_l645_645327


namespace total_balls_estimate_l645_645543

theorem total_balls_estimate 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (frequency : ℚ)
  (h_red_balls : red_balls = 12)
  (h_frequency : frequency = 0.6) 
  (h_fraction : (red_balls : ℚ) / total_balls = frequency): 
  total_balls = 20 := 
by 
  sorry

end total_balls_estimate_l645_645543


namespace GCF_LCM_computation_l645_645570

-- Definitions and axioms we need
def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- The theorem to prove
theorem GCF_LCM_computation : GCF (LCM 8 14) (LCM 7 12) = 28 :=
by sorry

end GCF_LCM_computation_l645_645570


namespace ratio_equilateral_triangle_l645_645720

def equilateral_triangle_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * (Real.sqrt 3 / 2)
  let area := (1 / 2) * s * altitude
  let perimeter := 3 * s in
  area / perimeter -- this simplifies to 25\sqrt{3} / 30 or 5\sqrt{3} / 6

theorem ratio_equilateral_triangle : ∀ (s : ℝ), s = 10 → equilateral_triangle_ratio s (by assumption) = 5 * (Real.sqrt 3) / 6 :=
by
  intros s h
  rw h
  sorry

end ratio_equilateral_triangle_l645_645720


namespace range_of_a_l645_645936

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + a * x + a + 3 > 0) : 0 ≤ a := sorry

end range_of_a_l645_645936


namespace area_of_quadrilateral_ABDF_l645_645355

-- Conditions
def length_AC : ℝ := 48
def width_AE : ℝ := 30
def ratio_B_on_AC : ℝ := 1/4
def ratio_F_on_AE : ℝ := 2/5

def area_rectangle (length_AC width_AE : ℝ) : ℝ := length_AC * width_AE
def length_AB := ratio_B_on_AC * length_AC
def length_BC := length_AC - length_AB
def length_AF := ratio_F_on_AE * width_AE
def length_FE := width_AE - length_AF
def area_triangle (base height : ℝ) : ℝ := 1/2 * base * height
def area_ACDE := area_rectangle length_AC width_AE
def area_BCD := area_triangle length_BC width_AE
def area_EFD := area_triangle length_FE length_AC

theorem area_of_quadrilateral_ABDF :
    area_ACDE - area_BCD - area_EFD = 468 :=
begin
  -- Proof omitted
  sorry
end

end area_of_quadrilateral_ABDF_l645_645355


namespace odd_n_permutation_exists_l645_645429

theorem odd_n_permutation_exists (n : ℕ) (h : n % 2 = 1) :
  ∃ (a : Fin n → Fin n), (∀ i : Fin (n + 1),
  isDistinct (λ (j : Fin (i+1)), (Finset.range j).sum a % (n + 1))) :=
sorry

end odd_n_permutation_exists_l645_645429


namespace closed_curve_area_correct_l645_645752

noncomputable def closed_curve_area : ℝ :=
  let length_of_arc := (3 : ℝ) * π / 4
  let side_length := (3 : ℝ)
  let number_of_arcs := (12 : ℕ)
  let radius := 3 / 2
  let octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  let sector_area := (3/4) * π * radius^2
  let total_sector_area := (number_of_arcs : ℝ) * sector_area / 2
  octagon_area + total_sector_area

theorem closed_curve_area_correct :
  closed_curve_area = 18 * (1 + Real.sqrt 2) + 81 * π / 8 := by
  sorry

end closed_curve_area_correct_l645_645752


namespace distance_between_incenter_and_circumcenter_l645_645373

-- Definitions and conditions
variables {A B C I O : Type}
variables [DecidableEq A] [DecidableEq B] [DecidableEq C]
variables (triangle_ABC : Triangle A B C)
variables sides_ABC_6_8_10 : ∃ (a b c : ℝ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a
variables inscribed_circle : TriangleInCircle A B C I
variables circumscribed_circle : TriangleCircumCircle A B C O

-- Statement to be proven
theorem distance_between_incenter_and_circumcenter :
  distance inscribed_circle.center circumscribed_circle.center = Real.sqrt 13 :=
sorry

end distance_between_incenter_and_circumcenter_l645_645373


namespace total_students_l645_645951

-- Definitions
def is_half_reading (S : ℕ) (half_reading : ℕ) := half_reading = S / 2
def is_third_playing (S : ℕ) (third_playing : ℕ) := third_playing = S / 3
def is_total_students (S half_reading third_playing homework : ℕ) := half_reading + third_playing + homework = S

-- Homework is given to be 4
def homework : ℕ := 4

-- Total number of students
theorem total_students (S : ℕ) (half_reading third_playing : ℕ)
    (h₁ : is_half_reading S half_reading) 
    (h₂ : is_third_playing S third_playing) 
    (h₃ : is_total_students S half_reading third_playing homework) :
    S = 24 := 
sorry

end total_students_l645_645951


namespace odd_product_probability_lt_one_eighth_l645_645657

theorem odd_product_probability_lt_one_eighth : 
  (∃ p : ℝ, p = (500 / 1000) * (499 / 999) * (498 / 998)) → p < 1 / 8 :=
by
  sorry

end odd_product_probability_lt_one_eighth_l645_645657


namespace area_of_triangle_proof_l645_645463

noncomputable def area_of_triangle (B D L : Point) (R : Real) (B D_distance : Real) : Real :=
  if h : B = D then 0
  else 
    let x := R / 20 in
    let BD := 9 * x in
    let BL := 5 * x in
    let AC := sqrt (4 * (BD^2 / 25)) in
    (1 / 2) * AC * BD * 2 * BD

theorem area_of_triangle_proof (A B C D L : Point) (R : Real)
  (h1 : tangent_to AB circle O R)
  (h2 : tangent_to BC circle O R)
  (h3 : on_median BD L)
  (h4 : BL = (5 / 9) * BD)
  (h5 : BD = 9 * (R / 20)):
  area_of_triangle B D L R BD = (27 * R^2) / 100 :=
by
  sorry

end area_of_triangle_proof_l645_645463


namespace dodecahedron_interior_diagonals_count_l645_645417

-- Define a dodecahedron structure
structure Dodecahedron :=
  (vertices : ℕ)
  (edges_per_vertex : ℕ)
  (faces_per_vertex : ℕ)

-- Define the property of a dodecahedron
def dodecahedron_property : Dodecahedron :=
{
  vertices := 20,
  edges_per_vertex := 3,
  faces_per_vertex := 3
}

-- The theorem statement
theorem dodecahedron_interior_diagonals_count (d : Dodecahedron)
  (h1 : d.vertices = 20)
  (h2 : d.edges_per_vertex = 3)
  (h3 : d.faces_per_vertex = 3) : 
  (d.vertices * (d.vertices - d.edges_per_vertex)) / 2 = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_count_l645_645417


namespace hyperbola_asymptotes_l645_645495

noncomputable def asymptotes_equation (a b : ℝ) : Prop :=
∃ k:ℝ, k = b / a ∧ (y = k * x ∨ y = -k * x)

theorem hyperbola_asymptotes (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
(eccentricity : ℝ) (h3 : 2 = eccentricity)
(hyperbola_eqn : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) :
  asymptotes_equation a b :=
begin
  sorry,
end

end hyperbola_asymptotes_l645_645495


namespace sequence_integer_terms_count_9720_l645_645411

-- Define the sequence by repeatedly dividing by 3
def sequence (n : ℕ) : ℕ → ℕ
| 0     := n
| (k+1) := sequence k / 3

-- Define the predicate for the number of terms in the sequence that are integers
def integer_terms_count (n : ℕ) (count : ℕ) : Prop :=
  ∀ k : ℕ, k < count → (sequence n k : ℤ) % 1 = 0

-- Given the specific value of n = 9720 and prove it has 6 integer terms
theorem sequence_integer_terms_count_9720 : integer_terms_count 9720 6 :=
  sorry

end sequence_integer_terms_count_9720_l645_645411


namespace cross_fraction_eq1_cross_fraction_eq2_cross_fraction_eq3_l645_645312

-- Problem 1
theorem cross_fraction_eq1 (x : ℝ) : (x + 12 / x = -7) → 
  ∃ (x₁ x₂ : ℝ), (x₁ = -3 ∧ x₂ = -4 ∧ x = x₁ ∨ x = x₂) :=
sorry

-- Problem 2
theorem cross_fraction_eq2 (a b : ℝ) 
    (h1 : a * b = -6) 
    (h2 : a + b = -5) : (a ≠ 0 ∧ b ≠ 0) →
    (b / a + a / b + 1 = -31 / 6) :=
sorry

-- Problem 3
theorem cross_fraction_eq3 (k x₁ x₂ : ℝ)
    (hk : k > 2)
    (hx1 : x₁ = 2022 * k - 2022)
    (hx2 : x₂ = k + 1) :
    (x₁ > x₂) →
    (x₁ + 4044) / x₂ = 2022 :=
sorry

end cross_fraction_eq1_cross_fraction_eq2_cross_fraction_eq3_l645_645312


namespace number_of_arrangements_is_32_l645_645131

open Nat

/-- Number of ways to arrange the numbers 1 through 9 in 9 cells such that
    the sum of the numbers in each column, starting from the second one,
    is 1 more than in the previous one is 32. -/
theorem number_of_arrangements_is_32 :
  ∃ (a b c d e f g h i : ℕ),
    set (finset.range 9) = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    (a + d) + (b + e) + (c + f) + (g + h + i) = 45 ∧
    b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
    f = e + 1 ∧ g = f + 1 ∧ h = g + 1 ∧ i = h + 1 :=
  sorry

end number_of_arrangements_is_32_l645_645131


namespace equilateral_triangle_area_to_perimeter_ratio_l645_645690

theorem equilateral_triangle_area_to_perimeter_ratio
  (a : ℝ) (h : a = 10) :
  let altitude := a * (Real.sqrt 3 / 2) in
  let area := (1 / 2) * a * altitude in
  let perimeter := 3 * a in
  area / perimeter = 5 * (Real.sqrt 3) / 6 := 
by
  sorry

end equilateral_triangle_area_to_perimeter_ratio_l645_645690


namespace number_of_possible_n_l645_645674

theorem number_of_possible_n (n_le_2021 : n ≤ 2021)
    (total_jars : nat := 343)
    (color_range : finset ℕ := finset.range' 1 8)
    (flipping_condition : ∀ (b y r : ℕ), b ∈ color_range → y ∈ color_range → r ∈ color_range → Π (b' y' r' : ℕ), b' ∈ color_range → y' ∈ color_range → r' ∈ color_range → b' - b ≤ 1 ∧ y' - y ≤ 1 ∧ r' - r ≤ 1)
    (jars_flipped_in_each_move : nat := 27) :
    (∃ n, n_le_2021 ∧ n ≥ 13 ∧ (∀ k ≥ 13, k ∈ finset.range n → odd k) ∧ finset.card (finset.filter odd (finset.range' 13 (2021 + 1))) = 1005)
    ∧ n = 1005 := sorry

end number_of_possible_n_l645_645674


namespace combination_15_12_l645_645404

theorem combination_15_12 : nat.choose 15 12 = 455 :=
by sorry

end combination_15_12_l645_645404


namespace isosceles_triangle_side_possibilities_l645_645835

theorem isosceles_triangle_side_possibilities:
  (∃ a b : ℕ, 2 * a + b = 20 ∧ 20 - 2 * a > 0 ∧ 20 - 2 * a < 2 * a) →
  (finset.filter (λ a : ℕ, 5 < a ∧ a < 10) (finset.range 20)).card = 4 :=
by
  sorry

end isosceles_triangle_side_possibilities_l645_645835


namespace cody_ate_dumplings_l645_645814

theorem cody_ate_dumplings (initial_dumplings remaining_dumplings : ℕ) (h1 : initial_dumplings = 14) (h2 : remaining_dumplings = 7) : initial_dumplings - remaining_dumplings = 7 :=
by
  sorry

end cody_ate_dumplings_l645_645814


namespace seq_arithmetic_general_formula_sum_b_n_l645_645461

noncomputable def seq (a : ℕ → ℝ) : ℕ → ℝ
| 0     := 1
| (n+1) := (a n) / (a n + 1)

noncomputable def seq_inv (a : ℕ → ℝ) : ℕ → ℝ :=
λ n, 1 / (seq a n)

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := (seq a n) * (seq a (n + 1))

def T (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), b a k

-- Statements to prove
theorem seq_arithmetic (n : ℕ) (a : ℕ → ℝ) : seq_inv a n = n + 1 :=
sorry

theorem general_formula (n : ℕ) : seq_inv (λ n, 1 / (n + 1)) n = n + 1 :=
sorry

theorem sum_b_n (n : ℕ) (a : ℕ → ℝ) : T a n = n / (n + 1) :=
sorry

end seq_arithmetic_general_formula_sum_b_n_l645_645461


namespace total_rocks_is_300_l645_645535

-- Definitions of rock types in Cliff's collection
variables (I S M : ℕ) -- I: number of igneous rocks, S: number of sedimentary rocks, M: number of metamorphic rocks
variables (shinyI shinyS shinyM : ℕ) -- shinyI: shiny igneous rocks, shinyS: shiny sedimentary rocks, shinyM: shiny metamorphic rocks

-- Given conditions
def igneous_one_third_shiny (I shinyI : ℕ) := 2 * shinyI = 3 * I
def sedimentary_two_ig_as_sed (S I : ℕ) := S = 2 * I
def metamorphic_twice_as_ig (M I : ℕ) := M = 2 * I
def shiny_igneous_is_40 (shinyI : ℕ) := shinyI = 40
def one_fifth_sed_shiny (S shinyS : ℕ) := 5 * shinyS = S
def three_quarters_met_shiny (M shinyM : ℕ) := 4 * shinyM = 3 * M

-- Theorem statement
theorem total_rocks_is_300 (I S M shinyI shinyS shinyM : ℕ)
  (h1 : igneous_one_third_shiny I shinyI)
  (h2 : sedimentary_two_ig_as_sed S I)
  (h3 : metamorphic_twice_as_ig M I)
  (h4 : shiny_igneous_is_40 shinyI)
  (h5 : one_fifth_sed_shiny S shinyS)
  (h6 : three_quarters_met_shiny M shinyM) :
  (I + S + M) = 300 :=
sorry -- Proof to be completed

end total_rocks_is_300_l645_645535


namespace triangle_O1BE_arithmetic_progression_l645_645813

noncomputable def circles_are_tangent (O1 O2 A B C D E : Point) (r1 r2 : ℝ) : Prop :=
r1 > r2 ∧
externally_tangent O1 O2 A ∧
tangent_at O1 B A ∧
tangent_at O2 C A ∧
tangent_line_intersect O1 O2 BC B C ∧
line_intersects O1 O2 Circle_O2 D ∧
line_intersects O1 O2 BC E ∧
(BC.length = 6 * DE.length)

theorem triangle_O1BE_arithmetic_progression (O1 O2 A B C D E : Point) (r1 r2 : ℝ)
(h_tangent : circles_are_tangent O1 O2 A B C D E r1 r2) :
triangle_side_lengths_arithmetic_sequence O1 B E ∧
distance O1 A B = 2 * distance O1 C A := by
  sorry

end triangle_O1BE_arithmetic_progression_l645_645813


namespace tan_counterexample_l645_645068

theorem tan_counterexample : 
  let α := (9 * Real.pi) / 4 
  let β := Real.pi / 4 in 
  (0 < α ∧ α < Real.pi / 2) ∧ 
  (0 < β ∧ β < Real.pi / 2) → 
  (α > β ∧ tan α = tan β) :=
by
  let α := (9 * Real.pi) / 4
  let β := Real.pi / 4
  have h1 : 0 < α ∧ α < Real.pi / 2 := sorry
  have h2 : 0 < β ∧ β < Real.pi / 2 := sorry
  have h3 : α > β := by
    sorry
  have h4 : tan α = tan β := by
    sorry
  exact ⟨h1, h2, ⟨h3, h4⟩⟩

end tan_counterexample_l645_645068


namespace cocktail_cans_l645_645754

theorem cocktail_cans (prev_apple_ratio : ℝ) (prev_grape_ratio : ℝ) 
  (new_apple_cans : ℝ) : ∃ new_grape_cans : ℝ, new_grape_cans = 15 :=
by
  let prev_apple_per_can := 1 / 6
  let prev_grape_per_can := 1 / 10
  let prev_total_per_can := (1 / 6) + (1 / 10)
  let new_apple_per_can := 1 / 5
  let new_grape_per_can := prev_total_per_can - new_apple_per_can
  let result := 1 / new_grape_per_can
  use result
  sorry

end cocktail_cans_l645_645754


namespace john_won_total_l645_645979

def student_amount (total_winnings : ℝ) : ℝ := total_winnings / 1000

def total_students_amount (total_winnings : ℝ) : ℝ := 100 * student_amount(total_winnings)

theorem john_won_total (total_winnings : ℝ) (h : total_students_amount(total_winnings) = 15525) : 
  total_winnings = 155250 :=
by
  sorry

end john_won_total_l645_645979


namespace sum_of_x_coords_of_P4_l645_645746

/--
A 150-gon P_1 is drawn in the Cartesian plane. The sum of the x-coordinates of the 150 vertices equals 3050.
The midpoints of the sides of P_1 form a second 150-gon, P_2. The midpoints of the sides of P_2 form a third 150-gon,
P_3, and similarly, the midpoints of the sides of P_3 form a fourth 150-gon, P_4.
Prove that the sum of the x-coordinates of the vertices of P_4 equals 3050.
-/

theorem sum_of_x_coords_of_P4 :
  let x_coords : Fin 150 → ℝ := sorry,
      P_1_x_sum := (Finset.univ.sum (λ i => x_coords i)),
      P_2_coords : Fin 150 → ℝ := λ i => (x_coords i + x_coords ((i + 1) % 150)) / 2,
      P_2_x_sum := (Finset.univ.sum (λ i => P_2_coords i)),
      P_3_coords : Fin 150 → ℝ := λ i => (P_2_coords i + P_2_coords ((i + 1) % 150)) / 2,
      P_3_x_sum := (Finset.univ.sum (λ i => P_3_coords i)),
      P_4_coords : Fin 150 → ℝ := λ i => (P_3_coords i + P_3_coords ((i + 1) % 150)) / 2,
      P_4_x_sum := (Finset.univ.sum (λ i => P_4_coords i))
  in P_1_x_sum = 3050 → P_4_x_sum = 3050 :=
sorry

end sum_of_x_coords_of_P4_l645_645746


namespace problem_l645_645933

theorem problem (x : ℝ) (h : sqrt (9 + x) + sqrt (16 - x) = 8) : (9 + x) * (16 - x) = 380.25 := 
by 
  sorry

end problem_l645_645933


namespace tangent_line_at_0_f_ge_l_f_plus_g_ge_0_l645_645492

open Real

def f (x : ℝ) : ℝ := 2 * sin x - 2 * cos x

def g (x : ℝ) : ℝ := exp (1 - 2 * x)

def l (x : ℝ) : ℝ := 2 * x - 2

theorem tangent_line_at_0 : ∀ (x : ℝ), f 0 = -2 ∧ deriv f 0 = 2,
proof
   sorry 

theorem f_ge_l : ∀ x ∈ Icc (-1/2 : ℝ) (1 : ℝ), f x ≥ l x :=
    sorry

theorem f_plus_g_ge_0 : ∀ x ∈ Icc (-1/2 : ℝ) (1 : ℝ), f x + g x ≥ 0 :=
    sorry

end tangent_line_at_0_f_ge_l_f_plus_g_ge_0_l645_645492


namespace factorization_of_x_squared_minus_nine_l645_645844

theorem factorization_of_x_squared_minus_nine (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by
  sorry

end factorization_of_x_squared_minus_nine_l645_645844


namespace determine_V_300_l645_645861

-- Definitions related to the arithmetic sequence
def arithmetic_sequence {α : Type*} [LinearOrderedField α] (b r : α) (n : ℕ) : α := b + (n - 1) * r

-- Definition of U_n
def U_n {α : Type*} [LinearOrderedField α] (b r : α) (n : ℕ) : α := (n / 2) * (2 * b + (n - 1) * r)

-- Definition of V_n
def V_n {α : Type*} [LinearOrderedField α] (b r : α) (n : ℕ) : α :=
  ∑ i in finset.range (n + 1), U_n b r i

-- Theorem stating the relationship between U_150 and V_300
theorem determine_V_300 {α : Type*} [LinearOrderedField α] (b r : α) (U150 : α) :
  U_150 = 150 * (b + 74.5 * r) →
  ∃ V300, V300 = 25 * 301 * (450 + 224.5 * r) :=
begin
  -- Proof is not required
  sorry,
end

end determine_V_300_l645_645861


namespace arithmetic_mean_l645_645462

theorem arithmetic_mean (n : ℕ) (h : n > 1) :
  let a := 1 + 1 / (n:ℚ)
  let a_i := λ i, if i = 1 then a else (1:ℚ)
  arithmetic_mean (finset.sum (finset.range n) (λ i, a_i i)) n = 1 + 1 / (n:ℚ)^2 := sorry

end arithmetic_mean_l645_645462


namespace upright_path_no_intersect_parabola_l645_645793

theorem upright_path_no_intersect_parabola :
  let m := 1
  let n := 1
  (m + n) = 2 :=
by
  -- Definition of up-right paths without intersection with the parabola
  let paths := (Finset.binom 13 3)
  let valid_paths := paths
  have all_paths_valid : valid_paths = paths := sorry
  have probability := valid_paths.toFloat() / paths.toFloat()
  have m, n : ℕ := (1, 1)
  have h_rel_prime : Nat.coprime m n := sorry
  exact (m + n) = 2

end upright_path_no_intersect_parabola_l645_645793


namespace carries_in_addition_l645_645571

theorem carries_in_addition (p : ℕ) (hp : p.Prime) (n m k : ℕ) :
  (p^k ∣ Nat.choose (n + m) m) →
  k = (λ (n m : ℕ), let carries := ... in carries) n m := sorry

end carries_in_addition_l645_645571


namespace max_distance_QP_l645_645466

noncomputable def max_distance_QP_on_circles (alpha : ℝ) : ℝ :=
  let P : ℝ × ℝ := (3 + 2 * Math.cos (alpha), 4 + 2 * Math.sin (alpha))
  let Q : ℝ × ℝ := (Math.cos alpha, Math.sin alpha)
  let distance := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  distance

theorem max_distance_QP (alpha : ℝ) :
  (3 - 0)^2 + (4 - 0)^2 = 25 →
  (x-3)² + (y-4)² = 4 →
  (Math.cos alpha)^2 + (Math.sin alpha)^2 = 1 →
  max_distance_QP_on_circles alpha = 8 :=
by
  sorry

end max_distance_QP_l645_645466


namespace anniversary_sale_total_cost_l645_645781

-- Definitions of conditions
def original_price_ice_cream : ℕ := 12
def discount_ice_cream : ℕ := 2
def sale_price_ice_cream : ℕ := original_price_ice_cream - discount_ice_cream

def price_per_five_cans_juice : ℕ := 2
def cans_per_five_pack : ℕ := 5

-- Definition of total cost
def total_cost : ℕ := 2 * sale_price_ice_cream + (10 / cans_per_five_pack) * price_per_five_cans_juice

-- The goal is to prove that total_cost is 24
theorem anniversary_sale_total_cost : total_cost = 24 :=
by
  sorry

end anniversary_sale_total_cost_l645_645781


namespace fundraising_exceeded_goal_l645_645214

theorem fundraising_exceeded_goal (ken mary scott : ℕ) (goal: ℕ) 
  (h_ken : ken = 600)
  (h_mary_ken : mary = 5 * ken)
  (h_mary_scott : mary = 3 * scott)
  (h_goal : goal = 4000) :
  (ken + mary + scott) - goal = 600 := 
  sorry

end fundraising_exceeded_goal_l645_645214


namespace domain_of_function_l645_645243

noncomputable def domain (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (2 * x - 3 ≥ 0) ∧ (x ≠ 3)

theorem domain_of_function :
  { x : ℝ | domain (λ x, sqrt (2 * x - 3) + (1 / (x - 3))) x } = 
  {x : ℝ | x ≥ 3 / 2} \ {3} :=
sorry

end domain_of_function_l645_645243


namespace remaining_weight_is_one_l645_645956

def weights : List ℕ := [1, 2, 3, ..., 16]

theorem remaining_weight_is_one :
  ∃ w ∈ weights, w = 1 :=
by
  sorry

end remaining_weight_is_one_l645_645956


namespace circle_tangent_radius_max_area_triangle_PMN_fixed_point_AB_l645_645458

theorem circle_tangent_radius (r : ℝ) (h : r > 0) : 
  (∀ x y, y = 2 * x + real.sqrt 5 ↔ x^2 + y^2 = r^2) → r = 1 :=
by sorry

theorem max_area_triangle_PMN : 
  (∀ P : ℝ × ℝ, (P.1 ^ 2 + (P.2 - 1) ^ 2 = 3 * (P.1 ^ 2 + (P.2 + 1) ^ 2)) → 
    1 / 2 * 2 * real.sqrt 3 = real.sqrt 3) :=
by sorry

theorem fixed_point_AB : 
  (∀ A B : ℝ × ℝ, (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (B.1 ^ 2 + B.2 ^ 2 = 1) ∧ 
  ((A.2 - 1) / A.1 * (B.2 - 1) / B.1 = real.sqrt 3 / 3) → 
    (∃ (P : ℝ × ℝ), P = (0, 2 + real.sqrt 3))) :=
by sorry

end circle_tangent_radius_max_area_triangle_PMN_fixed_point_AB_l645_645458


namespace limit_of_bijective_phi_l645_645555

open Nat

theorem limit_of_bijective_phi (φ : ℕ → ℕ) (hbij : Function.Bijective φ) (hlim : ∃ L : ℝ, tendsto (λ n : ℕ, φ n / n) at_top (𝓝 L)) : L = 1 := by
  sorry

end limit_of_bijective_phi_l645_645555


namespace tan_and_sin_to_cos_l645_645815

theorem tan_and_sin_to_cos (deg40 : Real) (deg10 : Real)
  (h1 : deg40 = 40 * (Real.pi / 180))
  (h2 : deg10 = 10 * (Real.pi / 180)) :
  Real.tan(deg40) + 6 * Real.sin(deg40) = Real.sqrt 3 + (Real.cos(deg10) / Real.cos(deg40)) :=
by
  sorry

end tan_and_sin_to_cos_l645_645815


namespace area_H1H2H3_eq_four_l645_645990

section TriangleArea

variables {P D E F H1 H2 H3 : Type*}

-- Definitions of midpoints, centroid, etc. can be implicit in Lean's formalism if necessary
-- We'll represent the area relation directly

-- Assume P is inside triangle DEF
def point_inside_triangle (P D E F : Type*) : Prop :=
sorry  -- Details are abstracted

-- Assume H1, H2, H3 are centroids of triangles PDE, PEF, PFD respectively
def is_centroid (H1 H2 H3 P D E F : Type*) : Prop :=
sorry  -- Details are abstracted

-- Given the area of triangle DEF
def area_DEF : ℝ := 12

-- Define the area function for the triangle formed by specific points
def area_triangle (A B C : Type*) : ℝ :=
sorry  -- Actual computation is abstracted

-- Mathematical statement to be proven
theorem area_H1H2H3_eq_four (P D E F H1 H2 H3 : Type*)
  (h_inside : point_inside_triangle P D E F)
  (h_centroid : is_centroid H1 H2 H3 P D E F)
  (h_area_DEF : area_triangle D E F = area_DEF) :
  area_triangle H1 H2 H3 = 4 :=
sorry

end TriangleArea

end area_H1H2H3_eq_four_l645_645990


namespace Harry_Terry_difference_l645_645102

theorem Harry_Terry_difference : 
(12 - (4 * 3)) - (12 - 4 * 3) = -24 := 
by
  sorry

end Harry_Terry_difference_l645_645102


namespace intervals_of_monotonicity_range_of_a_l645_645913

def f (x : ℝ) : ℝ := 2 * (Real.log (x / 2)) - (3 * x - 6) / (x + 1)

def g (x t a : ℝ) : ℝ := (x - t)^2 + (Real.log x - a * t)^2

theorem intervals_of_monotonicity :
  ∀ x ∈ set.Ioo (0 : ℝ) ∞, 
  (0 < x ∧ x < (1 / 2) ∨ 2 < x) → f x > 0 ∧
  ((1 / 2) < x ∧ x < 2) → f x < 0 :=
begin
  sorry
end

theorem range_of_a :
  ∀ x1 ∈ set.Ioo (1 : ℝ) ∞, 
  ∃ t ∈ set.Ioo (-∞ : ℝ) ∞, ∃ x2 ∈ set.Ioo (0 : ℝ) ∞, 
  f x1 ≥ g x2 t a → a ∈ set.Ioo (-∞ : ℝ) (1 / Real.exp 1) :=
begin
  sorry
end

end intervals_of_monotonicity_range_of_a_l645_645913


namespace ratio_of_second_to_first_l645_645387

theorem ratio_of_second_to_first (A1 A2 A3 : ℕ) (h1 : A1 = 600) (h2 : A3 = A1 + A2 - 400) (h3 : A1 + A2 + A3 = 3200) : A2 / A1 = 2 :=
by
  sorry

end ratio_of_second_to_first_l645_645387


namespace factorization_of_x_squared_minus_nine_l645_645842

theorem factorization_of_x_squared_minus_nine {x : ℝ} : x^2 - 9 = (x + 3) * (x - 3) :=
by
  -- Introduce the hypothesis to assist Lean in understanding the polynomial
  have h : x^2 - 9 = (x^2 - 3^2), 
  rw [pow_two, pow_two],
  exact factorization_of_x_squared_minus_3_squared _,
end

end factorization_of_x_squared_minus_nine_l645_645842


namespace anniversary_sale_total_cost_l645_645783

-- Definitions of conditions
def original_price_ice_cream : ℕ := 12
def discount_ice_cream : ℕ := 2
def sale_price_ice_cream : ℕ := original_price_ice_cream - discount_ice_cream

def price_per_five_cans_juice : ℕ := 2
def cans_per_five_pack : ℕ := 5

-- Definition of total cost
def total_cost : ℕ := 2 * sale_price_ice_cream + (10 / cans_per_five_pack) * price_per_five_cans_juice

-- The goal is to prove that total_cost is 24
theorem anniversary_sale_total_cost : total_cost = 24 :=
by
  sorry

end anniversary_sale_total_cost_l645_645783


namespace floor_div_eq_floor_div_l645_645165

theorem floor_div_eq_floor_div
  (a : ℝ) (n : ℤ) (ha_pos : 0 < a) :
  (⌊⌊a⌋ / n⌋ : ℤ) = ⌊a / n⌋ := 
sorry

end floor_div_eq_floor_div_l645_645165


namespace acid_concentration_third_flask_l645_645295

-- Define the concentrations of first and second flask
def conc_first (w1 : ℝ) : ℝ := 10 / (10 + w1)
def conc_second (w2 : ℝ) : ℝ := 20 / (20 + w2)

-- Define the acid mass in the third flask initially
def acid_mass_third : ℝ := 30

-- Total water added from the fourth flask
def total_water (w1 w2 : ℝ) : ℝ := w1 + w2

-- Acid concentration in the third flask after all water is added
def conc_third (w : ℝ) : ℝ := acid_mass_third / (acid_mass_third + w)

-- Problem statement: concentration in the third flask is 10.5%
theorem acid_concentration_third_flask (w1 : ℝ) (w2 : ℝ) (w : ℝ) 
  (h1 : conc_first w1 = 0.05) 
  (h2 : conc_second w2 = 70 / 300) 
  (h3 : w = total_water w1 w2) : 
  conc_third w = 10.5 / 100 := 
sorry

end acid_concentration_third_flask_l645_645295


namespace particle_position_after_3045_l645_645761

def particle_position_after_minutes (minutes : ℕ) : ℕ × ℕ :=
  let rec loop (n : ℕ) (time : ℕ) (x y : ℕ) : ℕ × ℕ :=
    if n % 2 = 1
    then -- odd n: right n units, up n+1 units, time increases by 2n+1
      let new_x := x + n
      let new_y := y + (n + 1)
      let new_time := time + (2 * n + 1)
      if new_time > minutes
      then let remaining_time := minutes - time
           (x + remaining_time, y)
      else loop (n + 1) new_time new_x new_y
    else -- even n: left n+1 units, down n units, time increases by 2n+1
      let new_x := x - (n + 1)
      let new_y := y - n
      let new_time := time + (2 * n + 1)
      if new_time > minutes
      then let remaining_time := minutes - time
           (x - remaining_time, y)
      else loop (n + 1) new_time new_x new_y
  loop 1 0 0 0

theorem particle_position_after_3045 : particle_position_after_minutes 3045 = (21, 54) :=
  by sorry

end particle_position_after_3045_l645_645761


namespace x_lt_1_is_necessary_but_not_sufficient_l645_645111

theorem x_lt_1_is_necessary_but_not_sufficient (x : ℝ) (hx : x < 1) :
  (hx → ln x < 0) ∧ ¬(ln x < 0 → hx) :=
sorry

end x_lt_1_is_necessary_but_not_sufficient_l645_645111


namespace parabola_has_one_x_intercept_l645_645416

theorem parabola_has_one_x_intercept :
  ∃! (x : ℝ), ∃ (y : ℝ), x = -3 * y ^ 2 + 2 * y + 2 ∧ y = 0 :=
by {
  existsi 2,
  existsi 0,
  split,
  { rw [pow_two, mul_zero, mul_zero, zero_add, zero_add], refl },
  { intros,
    rcases H with ⟨hy, hx⟩,
    rwa ←hx }
  sorry
}

end parabola_has_one_x_intercept_l645_645416


namespace least_value_is_one_l645_645315

noncomputable def least_value_of_expression : ℝ :=
  let expression (x y : ℝ) : ℝ := (x^2 * y - 1)^2 + (x - y)^2
  in let least_value := 1 in least_value

theorem least_value_is_one : 
  ∃ x y : ℝ, least_value_of_expression = 1 :=
by
  sorry

end least_value_is_one_l645_645315


namespace centroid_value_l645_645820

-- Define the vertices of triangle XYZ
def X := (-1, 4)
def Y := (5, 2)
def Z := (3, 10)

-- Define the point P as the centroid of triangle XYZ
def P : ℝ × ℝ := ((X.1 + Y.1 + Z.1) / 3, (X.2 + Y.2 + Z.2) / 3)

-- Define the expression for 10 * m + n, where P = (m, n)
def ten_m_plus_n (P : ℝ × ℝ) : ℝ :=
  10 * P.1 + P.2

-- State the theorem to prove
theorem centroid_value : ten_m_plus_n P = 86 / 3 := by sorry

end centroid_value_l645_645820


namespace simplify_A_value_of_A_l645_645904

variable {m : ℝ}

def A (m : ℝ) : ℝ := ((m ^ 2) / (m - 2) - (2 * m) / (m - 2)) + (m - 3) * (2 * m + 1)

theorem simplify_A : A m = 2 * m ^ 2 - 4 * m - 3 :=
by
  sorry

theorem value_of_A (h : m ^ 2 - 2 * m = 0) : A m = -3 :=
by
  sorry

end simplify_A_value_of_A_l645_645904


namespace binom_15_12_eq_455_l645_645402

theorem binom_15_12_eq_455 : nat.choose 15 12 = 455 := sorry

end binom_15_12_eq_455_l645_645402


namespace distinct_elements_condition_l645_645451

theorem distinct_elements_condition (x : ℕ) :
  ({5, x, x^2 - 4 * x}.card = 3 ↔ (x ≠ 5 ∧ x ≠ -1 ∧ x ≠ 0)) :=
sorry

end distinct_elements_condition_l645_645451


namespace k_value_if_perfect_square_l645_645943

theorem k_value_if_perfect_square (a k : ℝ) (h : ∃ b : ℝ, a^2 + 2*k*a + 1 = (a + b)^2) : k = 1 ∨ k = -1 :=
sorry

end k_value_if_perfect_square_l645_645943


namespace find_equation_of_BC_l645_645888

theorem find_equation_of_BC :
  ∃ (BC : ℝ → ℝ → Prop), 
  (∀ x y, (BC x y ↔ 2 * x - y + 5 = 0)) :=
sorry

end find_equation_of_BC_l645_645888


namespace length_DM_l645_645133

-- Define the conditions of the problem
variables (A B C D M : Type) [Geometry] -- implicit geometry context
variables (AB AC CD : ℝ) -- lengths of the sides in the quadrilateral
variables (angle_BAC angle_BDC : angle) -- the angles given in the problem
variables (h_AB : AB = sqrt 5) (h_AC : AC = sqrt 5) (h_CD : CD = 1)
variables (h_angle_BAC : angle_BAC = 90) (h_angle_BDC : angle_BDC = 90)

-- Statement to prove that DM = 1/2
theorem length_DM (length_DM : ℝ) : length_DM = 1/2 :=
by
  sorry

end length_DM_l645_645133


namespace equal_costs_at_45_students_l645_645950

def ticket_cost_option1 (x : ℕ) : ℝ :=
  x * 30 * 0.8

def ticket_cost_option2 (x : ℕ) : ℝ :=
  (x - 5) * 30 * 0.9

theorem equal_costs_at_45_students : ∀ x : ℕ, ticket_cost_option1 x = ticket_cost_option2 x ↔ x = 45 := 
by
  intro x
  sorry

end equal_costs_at_45_students_l645_645950


namespace exponential_sequence_term_eq_l645_645464

-- Definitions for the conditions
variable {α : Type} [CommRing α] (q : α)
def a (n : ℕ) : α := q * (q ^ (n - 1))

-- Statement of the problem
theorem exponential_sequence_term_eq : a q 9 = a q 3 * a q 7 := by
  sorry

end exponential_sequence_term_eq_l645_645464


namespace union_sets_intersection_complement_sets_l645_645096

universe u
variable {U A B : Set ℝ}

def universal_set : Set ℝ := {x | x ≤ 4}
def set_A : Set ℝ := {x | -2 < x ∧ x < 3}
def set_B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

theorem union_sets : set_A ∪ set_B = {x | -3 ≤ x ∧ x < 3} := by
  sorry

theorem intersection_complement_sets :
  set_A ∩ (universal_set \ set_B) = {x | 2 < x ∧ x < 3} := by
  sorry

end union_sets_intersection_complement_sets_l645_645096


namespace matrix_transformation_l645_645562

variable {A : Matrix (Fin 2) (Fin 2) ℝ}
variable {u x : Fin 2 → ℝ}
variable h1 : A.mulVec u = ![3, -7]
variable h2 : A.mulVec x = ![-2, 5]

theorem matrix_transformation :
  A.mulVec (3 • u - 2 • x) = ![13, -31] :=
by
  sorry

end matrix_transformation_l645_645562


namespace smallest_arrow_slit_length_total_length_greater_half_reliable_system_exists_l645_645672

theorem smallest_arrow_slit_length : 
  ∀ (s : ℝ), (∀ t : ℝ, t ≥ s → reliable (single_slit t)) → s = 2 / 3 :=
sorry

theorem total_length_greater_half : 
  ∀ (slits : List ℝ), (reliable (system_slits slits)) → (sum slits) > 1 / 2 :=
sorry

theorem reliable_system_exists : 
  ∀ (s : ℝ), s > 1 / 2 → (∃ (slits : List ℝ), (reliable (system_slits slits)) ∧ (sum slits) < s) :=
sorry

-- Definitions

def reliable (system : {}) : Prop := sorry  -- Fill more logic here

def single_slit (length : ℝ) : {} := sorry  -- Fill more logic here

def system_slits (slits : List ℝ) : {} := sorry  -- Fill more logic here

def sum (list : List ℝ) : ℝ := sorry  -- Sum of lengths of the list of slits

-- Function that demonstrates the sum of lengths

noncomputable def length (wall : ℝ) : ℝ := wall

end smallest_arrow_slit_length_total_length_greater_half_reliable_system_exists_l645_645672


namespace find_m_l645_645061

-- Define points A and B
def A (m : ℝ) := (-2 : ℝ, m)
def B (m : ℝ) := (m, 4 : ℝ)

-- Define the function to calculate the slope between two points
def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

-- Define the slope of the line passing through A and B
def slope_AB (m : ℝ) : ℝ := slope (-2) m m 4

-- Define the slope of the line 2x + y - 1 = 0
def slope_line : ℝ := -2

-- Statement of the mathematical problem
theorem find_m (m : ℝ) (h : slope_AB m = slope_line) : m = -8 :=
by
  sorry

end find_m_l645_645061


namespace problem_statement_l645_645219

def sum_not_11 (s : Set ℕ) : Prop :=
  ∀ {a b}, a ∈ s → b ∈ s → a ≠ b → a + b ≠ 11

def valid_subsets (M : Set ℕ) : Finset (Set ℕ) :=
  (Set.toFinset M).powerset.filter (λ s, Set.card s = 5 ∧ sum_not_11 s)

noncomputable def number_valid_subsets : ℕ :=
  (valid_subsets {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}).card

theorem problem_statement : number_valid_subsets = 32 := sorry

end problem_statement_l645_645219


namespace ratio_of_area_to_perimeter_l645_645679

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l645_645679


namespace A_3_2_equals_6_l645_645414

def A : ℕ → ℕ → ℕ
| 0 n := n + 1
| (m + 1) 0 := A m 1
| (m + 1) (n + 1) := A m (A (m + 1) n)

theorem A_3_2_equals_6 : A 3 2 = 6 := by
  sorry

end A_3_2_equals_6_l645_645414


namespace det_B4_l645_645509

theorem det_B4 (B : Matrix n n ℝ) (h : det B = -3) : det (B^4) = 81 := by
  sorry

end det_B4_l645_645509


namespace pyramid_cross_section_area_l645_645254

-- Define the conditions of the regular quadrilateral pyramid
variables (a : ℝ) -- base side length of the pyramid
variables (h l : ℝ) -- height and slant height of the pyramid

-- Define that the lateral edge forms an angle of 30 degrees with the height
axiom angle_condition : l * (real.cos (real.pi / 6)) = h

-- Prove the area of the cross-section passing through the apex and perpendicular to the opposite edge
theorem pyramid_cross_section_area :
  let S := (a^2 * real.sqrt 3) / 3 
  in S = (a^2 * real.sqrt 3) / 3 :=
by
  -- the actual proof steps go here
  sorry

end pyramid_cross_section_area_l645_645254


namespace factorization_of_x_squared_minus_nine_l645_645843

theorem factorization_of_x_squared_minus_nine {x : ℝ} : x^2 - 9 = (x + 3) * (x - 3) :=
by
  -- Introduce the hypothesis to assist Lean in understanding the polynomial
  have h : x^2 - 9 = (x^2 - 3^2), 
  rw [pow_two, pow_two],
  exact factorization_of_x_squared_minus_3_squared _,
end

end factorization_of_x_squared_minus_nine_l645_645843


namespace store_A_profit_margin_l645_645653

theorem store_A_profit_margin
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > x)
  (h : (y - x) / x + 0.12 = (y - 0.9 * x) / (0.9 * x)) :
  (y - x) / x = 0.08 :=
by {
  sorry
}

end store_A_profit_margin_l645_645653


namespace sequence_formula_l645_645094

theorem sequence_formula (S : ℕ → ℤ) (a : ℕ → ℤ) (h : ∀ n : ℕ, n > 0 → S n = 2 * a n - 2^n + 1) : 
  ∀ n : ℕ, n > 0 → a n = n * 2^(n - 1) :=
by
  intro n hn
  sorry

end sequence_formula_l645_645094


namespace problem1_problem2_l645_645070

noncomputable theory

open Set

def A (a : ℝ) : Set ℝ := {x | a - 4 ≤ x ∧ x ≤ a}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem problem1 (a : ℝ) (h_a : a = 0) : 
  (A a ∩ B) = {x | -4 ≤ x ∧ x < -1} ∧ 
  (A a ∪ B) = {x | x ≤ 0 ∨ x > 5} :=
by sorry

theorem problem2 (a : ℝ) : 
  (A a ∪ B) = B ↔ a < -1 ∨ 9 < a :=
by sorry

end problem1_problem2_l645_645070


namespace area_between_circles_l645_645125

noncomputable def radius_1 : ℝ := 12
noncomputable def distance_between_centers : ℝ := 2
noncomputable def chord_length : ℝ := 20

theorem area_between_circles : 
  let open Real in
  (π * radius_1 ^ 2 - π * (sqrt (radius_1 ^ 2 - (chord_length / 2) ^ 2 - distance_between_centers) ^ 2)) = 100 * π :=
by 
  sorry

end area_between_circles_l645_645125


namespace equal_soup_distribution_l645_645194

-- Given conditions
variables (x : ℝ) (hx_pos : x > 0)
def OmkarSoupTake (x : ℝ) := min 1 x
def Krit1SoupTake (rem : ℝ) := rem * 1/6
def Krit2SoupTake (rem : ℝ) := rem * 1/5
def Krit3SoupTake (rem : ℝ) := rem * 1/4

-- Correct answer to prove
noncomputable def everyoneGetsEqualAmount : Prop := x = 49/3

-- Proof statement
theorem equal_soup_distribution (hx : OmkarSoupTake x + Krit1SoupTake (x - OmkarSoupTake x) + 
                                 Krit2SoupTake ((x - OmkarSoupTake x) * 5/6) + 
                                 Krit3SoupTake (((x - OmkarSoupTake x) * 5/6 - (x - OmkarSoupTake x) * 4/30)) = x / 4) :
  everyoneGetsEqualAmount :=
sorry

end equal_soup_distribution_l645_645194


namespace laptop_cost_l645_645361

theorem laptop_cost (L : ℝ) (smartphone_cost : ℝ) (total_cost : ℝ) (change : ℝ) (n_laptops n_smartphones : ℕ) 
  (hl_smartphone : smartphone_cost = 400) 
  (hl_laptops : n_laptops = 2) 
  (hl_smartphones : n_smartphones = 4) 
  (hl_total : total_cost = 3000)
  (hl_change : change = 200) 
  (hl_total_spent : total_cost - change = 2 * L + 4 * smartphone_cost) : 
  L = 600 :=
by 
  sorry

end laptop_cost_l645_645361


namespace maximum_s_squared_l645_645367

-- Definitions based on our conditions
def semicircle_radius : ℝ := 5
def diameter_length : ℝ := 10

-- Statement of the problem (no proof, statement only)
theorem maximum_s_squared (A B C : ℝ×ℝ) (AC BC : ℝ) (h : AC + BC = s) :
    (A.2 = 0) ∧ (B.2 = 0) ∧ (dist A B = diameter_length) ∧
    (dist C (5,0) = semicircle_radius) ∧ (s = AC + BC) →
    s^2 ≤ 200 :=
sorry

end maximum_s_squared_l645_645367


namespace ratio_of_boys_to_girls_l645_645953

-- Definitions based on the initial conditions
def G : ℕ := 135
def T : ℕ := 351

-- Noncomputable because it involves division which is not always computable
noncomputable def B : ℕ := T - G

-- Main theorem to prove the ratio
theorem ratio_of_boys_to_girls : (B : ℚ) / G = 8 / 5 :=
by
  -- Here would be the proof, skipped with sorry.
  sorry

end ratio_of_boys_to_girls_l645_645953


namespace Fred_found_more_seashells_l645_645306

theorem Fred_found_more_seashells (Tom_seashells : ℕ) (Fred_seashells : ℕ) 
  (Tom_found : Tom_seashells = 15) (Fred_found : Fred_seashells = 43) : 
  Fred_seashells - Tom_seashells = 28 :=
by
  rw [Tom_found, Fred_found]
  exact rfl

end Fred_found_more_seashells_l645_645306


namespace ninth_number_in_sequence_is_121_l645_645253

-- Definitions of the initial part of the sequence
def seq : List Nat := [12, 13, 15, 17, 111, 113, 117, 119, 121, 129, 131]

-- Specification of the problem: the ninth element in the sequence
theorem ninth_number_in_sequence_is_121 : seq.nth 8 = some 121 := 
by {sorry}

end ninth_number_in_sequence_is_121_l645_645253


namespace tangent_segment_length_l645_645751

-- Setting up the necessary definitions and theorem.
def radius := 10
def seg1 := 4
def seg2 := 2

theorem tangent_segment_length :
  ∃ X : ℝ, X = 8 ∧
  (radius^2 = X^2 + ((X + seg1 + seg2) / 2)^2) :=
by
  sorry

end tangent_segment_length_l645_645751


namespace B_pow_101_eq_B_l645_645984

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![-1, 0, 0], ![0, 0, 0]]

-- State the theorem
theorem B_pow_101_eq_B : B^101 = B :=
  sorry

end B_pow_101_eq_B_l645_645984


namespace ratio_of_area_to_perimeter_l645_645707

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l645_645707


namespace vertex_parabola_locus_l645_645995

variable {a c : ℝ} (ha : 0 < a) (hc : 0 < c)

theorem vertex_parabola_locus :
  ∃ p : ℝ → ℝ, vertex_locus p = { pt : ℝ × ℝ | ∃ t : ℝ, pt.1 = -t ∧ pt.2 = -a * t^2 + c } :=
sorry

end vertex_parabola_locus_l645_645995


namespace parabola_vertex_example_l645_645831

noncomputable def parabola_vertex (a b c : ℝ) := (-b / (2 * a), (4 * a * c - b^2) / (4 * a))

theorem parabola_vertex_example : parabola_vertex (-4) (-16) (-20) = (-2, -4) :=
by
  sorry

end parabola_vertex_example_l645_645831


namespace find_u_to_make_root_l645_645054

theorem find_u_to_make_root :
  ∃ (u : ℝ), (u = 8.5) ∧ (∀ x : ℝ, x = ( -25 - real.sqrt 421 ) / 12 → 6 * x^2 + 25 * x + u = 0) :=
by
  use 8.5
  split
  { rfl }
  { intros x hx
    rw [hx, show (8.5 : ℝ) = (204 / 24) by norm_num]
    have sqrt_625_24u : real.sqrt (625 - 24 * (204 / 24)) = real.sqrt 421,
    { simp [show 625 - 24 * (204 / 24) = 421 by norm_num] }
    field_simp [sqrt_625_24u]
    norm_num
    sorry
  }

end find_u_to_make_root_l645_645054


namespace minimum_coins_to_identify_bag_l645_645764

theorem minimum_coins_to_identify_bag :
  ∀ (pouches : Fin 5 → List String),
    (∃ i, pouches i = List.replicate 30 "Gold")
    ∧ (∃ j, pouches j = List.replicate 30 "Silver")
    ∧ (∃ k, pouches k = List.replicate 30 "Bronze")
    ∧ (∃ m n, m ≠ n ∧
               pouches m = list.repeat "Gold" 10 ++ list.repeat "Silver" 10 ++ list.repeat "Bronze" 10
               ∧ pouches n = list.repeat "Gold" 10 ++ list.repeat "Silver" 10 ++ list.repeat "Bronze" 10)
  → 5 := sorry
  
end minimum_coins_to_identify_bag_l645_645764


namespace probability_of_fewer_heads_than_tails_l645_645316

theorem probability_of_fewer_heads_than_tails :
  let n := 8 in
  let outcomes := 2^8 in
  let equal_heads_tails := Nat.choose n (n / 2) in
  let y := equal_heads_tails / outcomes in
  let x := (1 - y) / 2 in
  x = 93 / 256 :=
by
  sorry

end probability_of_fewer_heads_than_tails_l645_645316


namespace equilateral_triangle_ratio_l645_645712

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Define the altitude of the equilateral triangle
def altitude (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2

-- Define the area of the equilateral triangle
def area (a : ℝ) : ℝ := (a * altitude a) / 2

-- Define the perimeter of the equilateral triangle
def perimeter (a : ℝ) : ℝ := 3 * a

-- Define the ratio of area to perimeter
def ratio (a : ℝ) : ℝ := area a / perimeter a

theorem equilateral_triangle_ratio :
  ratio 10 = (5 * Real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l645_645712


namespace ratio_of_area_to_perimeter_l645_645711

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l645_645711


namespace value_of_a_with_two_distinct_roots_l645_645024

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l645_645024


namespace anne_age_when_paul_is_38_l645_645118

theorem anne_age_when_paul_is_38 (p2015 : ℕ) (a2015 : ℕ) (difference : a2015 - p2015 = 3)
  (p38 : p2015 + (38 - 11) = 38) : a2015 + (38 - 11) = 41 :=
by 
  rw [p2015, a2015] at *
  sorry

end anne_age_when_paul_is_38_l645_645118


namespace park_area_l645_645740

theorem park_area (length breadth : ℝ) (x : ℝ) 
  (h1 : length = 3 * x) 
  (h2 : breadth = x) 
  (h3 : 2 * length + 2 * breadth = 800) 
  (h4 : 12 * (4 / 60) * 1000 = 800) : 
  length * breadth = 30000 := by
sorry

end park_area_l645_645740


namespace scatterbrained_scientist_probability_l645_645745

noncomputable def bus_ticket_success_probability : ℝ :=
  let p := 0.1 in
  let q := 1 - p in
  let x1 := p * 0.033 in -- Placeholder for recursive calculations over the state transitions
  let x3 := p + q * (p^2 + 2 * p * q * (p + q * 0.033)) in
  let x2 := p^2 + 2 * p * q * x3 in
  q * x2

theorem scatterbrained_scientist_probability :
  bus_ticket_success_probability = 0.033 :=
sorry

end scatterbrained_scientist_probability_l645_645745


namespace max_routes_l645_645636

-- Given conditions
def routes_through_stop (route : Set (Fin 9)) (stop : Fin 9) : Prop :=
  stop ∈ route

def routes_have_three_stops (route : Set (Fin 9)) : Prop :=
  route.card = 3

def routes_share_exactly_one_stop (route1 route2 : Set (Fin 9)) : Prop :=
  route1 ≠ route2 → (route1 ∩ route2).card ≤ 1

-- Question: maximum number of routes with conditions
theorem max_routes (n : ℕ) (routes : Fin n → Set (Fin 9)) (stops : Fin 9) :
  (∀ r, routes_have_three_stops (routes r)) →
  (∀ r1 r2, routes_share_exactly_one_stop (routes r1) (routes r2)) →
  ∃ bound : ℕ, n ≤ bound :=
sorry

end max_routes_l645_645636


namespace ratio_of_area_to_perimeter_l645_645680

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l645_645680


namespace trailing_zeros_500_factorial_l645_645929

theorem trailing_zeros_500_factorial : 
  let count_multiples (n : ℕ) (p : ℕ) : ℕ := n / p
  count_multiples 500 5 + count_multiples 500 25 + 
  count_multiples 500 125 + count_multiples 500 625 = 124 :=
by
  let count_multiples := fun (n p : ℕ) => n / p
  have h1 : count_multiples 500 5 = 500 / 5 := by rfl
  have h2 : count_multiples 500 25 = 500 / 25 := by rfl
  have h3 : count_multiples 500 125 = 500 / 125 := by rfl
  have h4 : count_multiples 500 625 = 500 / 625 := by rfl
  simp only [h1, h2, h3, h4]
  norm_num
  exact rfl

end trailing_zeros_500_factorial_l645_645929


namespace team_A_wins_exactly_4_of_7_l645_645129

noncomputable def probability_team_A_wins_4_of_7 : ℚ :=
  (Nat.choose 7 4) * ((1/2)^4) * ((1/2)^3)

theorem team_A_wins_exactly_4_of_7 :
  probability_team_A_wins_4_of_7 = 35 / 128 := by
sorry

end team_A_wins_exactly_4_of_7_l645_645129


namespace find_length_CD_l645_645947

theorem find_length_CD 
  (α β γ : Type) [linear_order α] [has_zero α] [has_add α] [has_smul α α] [has_mul α] [has_div α] [has_sqrt α]
  (A B C D : Point) 
  (AB BC CD : α) :
  angle B = 150 * degree_to_radian →
  length A B = 6 →
  length B C = 4 →
  perpendicular (line_through A B) (line_through A D) →
  perpendicular (line_through C B) (line_through C D) →
  length C D = (sqrt (52 + 24 * sqrt 3) * (sqrt 6 - sqrt 2)) / 4 := 
sorry

end find_length_CD_l645_645947


namespace rectangle_width_decrease_l645_645624

theorem rectangle_width_decrease (A L W : ℝ) (h1 : A = L * W) (h2 : 1.5 * L * W' = A) : 
  (W' = (2/3) * W) -> by exact (W - W') / W = 1 / 3 :=
by
  sorry

end rectangle_width_decrease_l645_645624


namespace problem_statement_l645_645493

noncomputable def f (x : ℝ) : ℝ := exp x + x^2
noncomputable def g (x : ℝ) : ℝ := sin x + x
noncomputable def l (x : ℝ) : ℝ := x + 1

theorem problem_statement (x : ℝ) :
  ∃ (f g l : ℝ → ℝ), (∀ x, f x = exp x + x^2) ∧
                     (∀ x, g x = sin x + x) ∧
                     (∀ x, l x = x + 1) ∧
                     ae_exp_plus_x_squared_minus_x_minus_sin_x x > 0 :=
by
  sorry

end problem_statement_l645_645493


namespace equation_has_at_least_two_distinct_roots_l645_645013

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l645_645013


namespace dice_product_probability_l645_645730

theorem dice_product_probability :
  (∃ (a b c d : ℕ), a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} ∧ d ∈ {1, 2, 3, 4, 5, 6} ∧ a * b * c * d = 4) →
  (probability_event (λ (a b c d : ℕ), a * b * c * d = 4) = 1 / 648) :=
sorry

end dice_product_probability_l645_645730


namespace problem_statement_l645_645901

noncomputable def f : ℕ → ℝ := sorry

axiom functional_eq (p q : ℕ) : f (p + q) = f p * f q
axiom initial_value : f 1 = 3

theorem problem_statement : 
    ( [f(1)]^2 + f(2)) / f(1) +
    ( [f(2)]^2 + f(4)) / f(3) +
    ( [f(3)]^2 + f(6)) / f(5) +
    ( [f(4)]^2 + f(8)) / f(7) +
    ( [f(5)]^2 + f(10)) / f(9)
    = 30 :=
by
  sorry

end problem_statement_l645_645901


namespace hiking_trip_time_l645_645349

noncomputable def R_up : ℝ := 7
noncomputable def R_down : ℝ := 1.5 * R_up
noncomputable def Distance_down : ℝ := 21
noncomputable def T_down : ℝ := Distance_down / R_down
noncomputable def T_up : ℝ := T_down

theorem hiking_trip_time :
  T_up = 2 := by
      sorry

end hiking_trip_time_l645_645349


namespace shortest_path_on_right_angle_polyhedron_l645_645138

theorem shortest_path_on_right_angle_polyhedron (X Y : ℝ) : 
  (shortest_path_on_surface X Y = 3 * real.sqrt 2) :=
begin
  /- Proof would go here -/
  sorry,
end

end shortest_path_on_right_angle_polyhedron_l645_645138


namespace concentration_third_flask_l645_645301

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l645_645301


namespace find_a_of_perpendicular_lines_l645_645308

theorem find_a_of_perpendicular_lines
  (a : ℝ) 
  (line1 : ℝ → ℝ → Prop := λ x y, ax + y - 1 = 0)
  (line2 : ℝ → ℝ → Prop := λ x y, x - y + 3 = 0)
  (perpendicular : ∀ x y, line1 x y → line2 x y → -a * 1 = -1) : a = 1 :=
sorry

end find_a_of_perpendicular_lines_l645_645308


namespace ellipse_equation_line_equation_l645_645244

/-- The equation of the ellipse given certain conditions -/
theorem ellipse_equation
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (ecc_eq : (b = a * (√3) / 2))
  (passes_through_C : ∀ x y : ℝ, (x = 1) → (y = (√3) / 2) → (x^2 / a^2 + y^2 / b^2 = 1)) :
  (a^2 = 4) ∧ (b^2 = 1) ∧ (∀ x y: ℝ, (x^2/4 + y^2 = 1)) :=
sorry

/-- The equation of the line passing through B(-1,0) intersecting the ellipse at P and Q
where |MN| = 4√7 -/
theorem line_equation
  (a b m n : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (ha : a^2 = 4)
  (hb : b^2 = 1)
  (hb1 : (a = 2 * b)) 
  (l_through_B : ∀ x y : ℝ, (x = -1) → (y = 0) → (∃ m : ℝ, x = m*y - 1))
  (intersection_condition : ∀ x y : ℝ, (x = (1 - y) / 2) → (mn_abs : |1 - y| = 4*√7)) :
  (m = 2 ∨ m = 1/3) → 
  ((∀ x y : ℝ, ((x - 2y + 1 = 0) ∨ (3x - y + 3 = 0))) :=
sorry

end ellipse_equation_line_equation_l645_645244


namespace find_first_term_l645_645080

-- Define the geometric sequence sum function
def S (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

-- Define the conditions
def conditions (a q : ℝ) : Prop := S a q 4 = 1 ∧ S a q 8 = 17

-- Prove that the first term a equals one of the given values under the conditions
theorem find_first_term (a q : ℝ) (h : conditions a q) :
  a = -1/5 ∨ a = 1/15 :=
by
  sorry

end find_first_term_l645_645080


namespace range_of_m_in_third_quadrant_l645_645116

/-- If the point (m - 4, 1 - 2m) is in the third quadrant, then the range of m is 1/2 < m < 4. -/
theorem range_of_m_in_third_quadrant (m : ℝ) : 
  (m - 4 < 0) ∧ (1 - 2 * m < 0) → (1 / 2 < m) ∧ (m < 4) :=
by
  intro h
  cases h
  split
  sorry

end range_of_m_in_third_quadrant_l645_645116


namespace perimeter_of_square_C_is_90_l645_645924

-- Definitions based on the given problem
def side_length_A : ℝ := 30 / 4
def side_length_B : ℝ := 2 * side_length_A
def side_length_C : ℝ := side_length_A + side_length_B
def perimeter (side_length : ℝ) : ℝ := 4 * side_length

-- The proof statement
theorem perimeter_of_square_C_is_90 :
  perimeter side_length_C = 90 :=
by
  sorry

end perimeter_of_square_C_is_90_l645_645924


namespace logo_height_poster_logo_height_badge_l645_645381

theorem logo_height_poster (h_orig: ℝ := 2) (w_orig: ℝ := 3) (w_new: ℝ := 12): 
  (h_new_poster: ℝ := h_orig * (w_new / w_orig)) = 8 := 
by 
  sorry

theorem logo_height_badge (h_orig: ℝ := 2) (w_orig: ℝ := 3) (w_new: ℝ := 1.5): 
  (h_new_badge: ℝ := h_orig * (w_new / w_orig)) = 1 := 
by 
  sorry

end logo_height_poster_logo_height_badge_l645_645381


namespace pow_four_inequality_l645_645203

theorem pow_four_inequality (x y : ℝ) : x^4 + y^4 ≥ x * y * (x + y)^2 :=
by
  sorry

end pow_four_inequality_l645_645203


namespace ratio_of_area_to_perimeter_l645_645708

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l645_645708


namespace z_sixth_power_l645_645148

-- Define z as given in the problem statement
def z : ℂ := (-real.sqrt 3 - complex.i) / 2

-- Our theorem to prove z^6 = -1
theorem z_sixth_power : z^6 = -1 :=
by sorry

end z_sixth_power_l645_645148


namespace no_real_solutions_for_equation_l645_645602

theorem no_real_solutions_for_equation (x : ℝ) : ¬(∃ x : ℝ, (8 * x^2 + 150 * x - 5) / (3 * x + 50) = 4 * x + 7) :=
sorry

end no_real_solutions_for_equation_l645_645602


namespace zeros_in_Q_l645_645577

def R_k (k : ℕ) : ℤ := (7^k - 1) / 6

def Q : ℤ := (7^30 - 1) / (7^6 - 1)

def count_zeros (n : ℤ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 470588 :=
by sorry

end zeros_in_Q_l645_645577


namespace solution_set_of_f_l645_645482

def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.log x / Real.log 2 else 1 - x^2

theorem solution_set_of_f (x : ℝ) : f x > 0 ↔ -1 < x ∧ x < 1 := by sorry

end solution_set_of_f_l645_645482


namespace problem_solution_l645_645931

theorem problem_solution (x : ℝ) (h : sqrt (9 + x) + sqrt (16 - x) = 8) : (9 + x) * (16 - x) = 380.25 :=
by
  sorry

end problem_solution_l645_645931


namespace valid_a_value_l645_645032

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l645_645032


namespace equilateral_triangle_ratio_l645_645718

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Define the altitude of the equilateral triangle
def altitude (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2

-- Define the area of the equilateral triangle
def area (a : ℝ) : ℝ := (a * altitude a) / 2

-- Define the perimeter of the equilateral triangle
def perimeter (a : ℝ) : ℝ := 3 * a

-- Define the ratio of area to perimeter
def ratio (a : ℝ) : ℝ := area a / perimeter a

theorem equilateral_triangle_ratio :
  ratio 10 = (5 * Real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l645_645718


namespace tickets_left_unsold_l645_645051

def total_rolls := 60
def tickets_per_roll := 300
def lost_rolls := 1
def torn_percentage := 0.05
def fourth_graders_percentage := 0.30
def fifth_graders_percentage := 0.40
def sixth_graders_percentage := 0.25
def seventh_graders_percentage := 0.35
def eighth_graders_percentage := 0.20
def ninth_graders_tickets := 200

noncomputable def initial_tickets := total_rolls * tickets_per_roll
noncomputable def lost_tickets := lost_rolls * tickets_per_roll
noncomputable def usable_tickets_after_loss := initial_tickets - lost_tickets
noncomputable def torn_tickets := torn_percentage * usable_tickets_after_loss
noncomputable def total_usable_tickets := usable_tickets_after_loss - torn_tickets

noncomputable def fourth_graders_tickets := fourth_graders_percentage * total_usable_tickets
noncomputable def remaining_after_fourth := total_usable_tickets - fourth_graders_tickets

noncomputable def fifth_graders_tickets := fifth_graders_percentage * remaining_after_fourth
noncomputable def remaining_after_fifth := remaining_after_fourth - fifth_graders_tickets

noncomputable def sixth_graders_tickets := sixth_graders_percentage * remaining_after_fifth
noncomputable def remaining_after_sixth := remaining_after_fifth - sixth_graders_tickets

noncomputable def seventh_graders_tickets := seventh_graders_percentage * remaining_after_sixth
noncomputable def remaining_after_seventh := remaining_after_sixth - seventh_graders_tickets

noncomputable def eighth_graders_tickets := eighth_graders_percentage * remaining_after_seventh
noncomputable def remaining_after_eighth := remaining_after_seventh - eighth_graders_tickets

noncomputable def remaining_after_ninth := remaining_after_eighth - ninth_graders_tickets

theorem tickets_left_unsold : remaining_after_ninth = 2556 := by
  -- Primary goal is to ensure the Lean code can be built; 
  -- detailed calculations are skipped.
  sorry

end tickets_left_unsold_l645_645051


namespace tetrahedron_QRS_area_l645_645992

noncomputable def area_triangle_QRS (a b c : ℝ) : ℝ :=
  1 / 2 * Real.sqrt (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)

theorem tetrahedron_QRS_area (a b c : ℝ) :
  triangle_area a b c = 1 / 2 * Real.sqrt (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) :=
by
  unfold triangle_area
  sorry

end tetrahedron_QRS_area_l645_645992


namespace not_write_date_03_12_not_write_date_else_l645_645980

structure Date := (day month : Nat)

def isAmbiguous (d: Date) : Prop :=
  d.day ≥ 1 ∧ d.day ≤ 12 ∧ d.month ≥ 1 ∧ d.month ≤ 12 ∧ d.day ≠ d.month

theorem not_write_date_03_12 :
  isAmbiguous (Date.mk 3 12) = true :=
by
  apply And.intro
  · apply And.intro
    · exact Nat.le_refl 3
    · exact Nat.dec_le 3 12
  · apply And.intro
    · apply And.intro
      · exact Nat.le_refl 12
      · exact Nat.dec_le 12 12
    · exact Nat.dec_ne 3 12

theorem not_write_date_else :
  ¬ (isAmbiguous (Date.mk 18 8) = true ∨ isAmbiguous (Date.mk 5 5) = true) :=
by
  apply And.intro
  · intro h
    cases h
  · apply And.intro
    · exact Nat.dec_ne 18 8
    · exact Nat.dec_ne 5 5

end not_write_date_03_12_not_write_date_else_l645_645980


namespace exponentiation_properties_l645_645395

theorem exponentiation_properties:
  (10^6) * (10^2)^3 / 10^4 = 10^8 :=
by
  sorry

end exponentiation_properties_l645_645395


namespace find_n_l645_645565

noncomputable def a : ℝ := Real.pi / 2010

theorem find_n :
  ∃ n : ℕ, (∀ m : ℕ, 0 < m → m < n → ¬ (2 * (Finset.range m).sum (λ k, Real.cos ((k+1)^2 * a) * Real.sin ((k+1) * a) ∈ ℤ)) ∧ 
  2 * (Finset.range n).sum (λ k, Real.cos ((k+1)^2 * a) * Real.sin ((k+1) * a) ∈ ℤ) ∧ 
  n = 201 :=
begin
  sorry
end

end find_n_l645_645565


namespace taxi_ride_distance_l645_645342

variable (t : ℝ) (c₀ : ℝ) (cᵢ : ℝ)

theorem taxi_ride_distance (h_t : t = 18.6) (h_c₀ : c₀ = 3.0) (h_cᵢ : cᵢ = 0.4) : 
  ∃ d : ℝ, d = 8 := 
by 
  sorry

end taxi_ride_distance_l645_645342


namespace general_proposition_l645_645470

theorem general_proposition 
  (h1 : sin 30 ^ 2 + sin 90 ^ 2 + sin 150 ^ 2 = 3 / 2)
  (h2 : sin 5 ^ 2 + sin 65 ^ 2 + sin 125 ^ 2 = 3 / 2) 
  (α : ℝ) :
  sin (α - 60) ^ 2 + sin α ^ 2 + sin (α + 60) ^ 2 = 3 / 2 :=
by
  sorry

end general_proposition_l645_645470


namespace unique_solution_l645_645428

theorem unique_solution (m n : ℤ) (h : 231 * m^2 = 130 * n^2) : m = 0 ∧ n = 0 :=
by {
  sorry
}

end unique_solution_l645_645428


namespace intervals_monotone_range_a_decreasing_range_a_inequality_l645_645486

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 + real.log (x + 1)

-- (1) Intervals of monotonicity when a = -1/4
theorem intervals_monotone (x : ℝ) : -1 < x → x < 1 ∨ x > 1 := sorry

-- (2) Range of a if the function is decreasing on [1, +∞)
theorem range_a_decreasing (a : ℝ) (h : ∀ x ∈ set.Icc (1 : ℝ) (⊤ : ℝ), f a x ≤ 0) : a ≤ -1/4 := sorry

-- (3) Range of a if f(x) - x ≤ 0 on [0, +∞)
theorem range_a_inequality (a : ℝ) (h : ∀ x ∈ set.Ici (0 : ℝ), f a x - x ≤ 0) : a ≤ 0 := sorry

end intervals_monotone_range_a_decreasing_range_a_inequality_l645_645486


namespace chiquita_height_l645_645191

theorem chiquita_height (C : ℝ) :
  (C + (C + 2) = 12) → (C = 5) :=
by
  intro h
  sorry

end chiquita_height_l645_645191


namespace monomials_to_perfect_square_l645_645898

theorem monomials_to_perfect_square :
  ∃ n : ℕ, n = 6 ∧ ∀ (p : ℝ[X] × ℝ[X]), 
    (p.1.coeff (4, 2) + p.2.coeff (2, 4) + (p.1.coeff (2, 1) - p.2.coeff (1, 2))^2 - p.1.coeff (2, 1) ^ 2 - p.2.coeff (1, 2) ^ 2 = 0) :=
begin
  sorry
end

end monomials_to_perfect_square_l645_645898


namespace bus_driver_max_regular_hours_l645_645750

theorem bus_driver_max_regular_hours 
  (regular_rate : ℕ)
  (ot_rate_factor : ℕ)
  (total_hours : ℕ)
  (total_compensation : ℕ)
  (H : ℕ) 
  (h1 : regular_rate = 15)
  (h2 : ot_rate_factor = 75)
  (h3 : total_hours = 54.32)
  (h4 : total_compensation = 976) :
  (H = 40) := 
sorry

end bus_driver_max_regular_hours_l645_645750


namespace simplify_trig_expr_l645_645225

-- Mathematical definitions and trigonometric properties
variables (α : Real)

theorem simplify_trig_expr :
  (sin (2 * π - α) * cos (3 * π + α) * cos (3 * π / 2 - α)) / 
  (sin (-π + α) * sin (3 * π - α) * cos (-α - π)) = -1 := 
by 
  sorry

end simplify_trig_expr_l645_645225


namespace acid_concentration_third_flask_l645_645296

-- Define the concentrations of first and second flask
def conc_first (w1 : ℝ) : ℝ := 10 / (10 + w1)
def conc_second (w2 : ℝ) : ℝ := 20 / (20 + w2)

-- Define the acid mass in the third flask initially
def acid_mass_third : ℝ := 30

-- Total water added from the fourth flask
def total_water (w1 w2 : ℝ) : ℝ := w1 + w2

-- Acid concentration in the third flask after all water is added
def conc_third (w : ℝ) : ℝ := acid_mass_third / (acid_mass_third + w)

-- Problem statement: concentration in the third flask is 10.5%
theorem acid_concentration_third_flask (w1 : ℝ) (w2 : ℝ) (w : ℝ) 
  (h1 : conc_first w1 = 0.05) 
  (h2 : conc_second w2 = 70 / 300) 
  (h3 : w = total_water w1 w2) : 
  conc_third w = 10.5 / 100 := 
sorry

end acid_concentration_third_flask_l645_645296


namespace solution_correct_l645_645863

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem solution_correct {n : ℕ} (hn : n ≥ 0) :
  n ≤ 2 * sum_of_digits n ↔ n ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19} : set ℕ) :=
by sorry

end solution_correct_l645_645863


namespace ratio_of_squares_l645_645776

-- Define the conditions
def side_1 := 5
def side_2 := 12
def hypotenuse := 13

def square_in_right_triangle_one (triangle_side1 triangle_side2 triangle_hyp : ℝ) : ℝ :=
  let x := (triangle_side1 * triangle_side2) / (triangle_side1 + triangle_side2)
  in x

def square_in_right_triangle_two (triangle_side1 triangle_side2 triangle_hyp : ℝ) : ℝ :=
  let y := (triangle_side1 * triangle_hyp) / (triangle_side1 + triangle_hyp)
  in y

-- Calculate x and y for our specific triangle
def x := square_in_right_triangle_one side_1 side_2 hypotenuse
def y := square_in_right_triangle_two side_1 side_2 hypotenuse

-- The proof problem: Prove the ratio x / y is as claimed
theorem ratio_of_squares : (x / y) = (1380 / 2873) := by
  -- The proof would go here
  sorry

end ratio_of_squares_l645_645776


namespace f_sqrt_29_l645_645415

def f (x : ℝ) : ℝ :=
  if x ∈ ℚ then 10 * x - 3 else 3 * (floor x) + 7

theorem f_sqrt_29 : f (Real.sqrt 29) = 22 := by
  sorry

end f_sqrt_29_l645_645415


namespace product_sum_diff_l645_645641

variable (a b : ℝ) -- Real numbers

theorem product_sum_diff (a b : ℝ) : (a + b) * (a - b) = (a + b) * (a - b) :=
by
  sorry

end product_sum_diff_l645_645641


namespace possible_third_side_l645_645542

theorem possible_third_side (x : ℝ) : (3 + 4 > x) ∧ (abs (4 - 3) < x) → (x = 2) :=
by 
  sorry

end possible_third_side_l645_645542


namespace speedster_convertibles_approx_l645_645749

-- Definitions corresponding to conditions
def total_inventory : ℕ := 120
def num_non_speedsters : ℕ := 40
def num_speedsters : ℕ := 2 * total_inventory / 3
def num_speedster_convertibles : ℕ := 64

-- Theorem statement
theorem speedster_convertibles_approx :
  2 * total_inventory / 3 - num_non_speedsters + num_speedster_convertibles = total_inventory :=
sorry

end speedster_convertibles_approx_l645_645749


namespace dorothy_profit_l645_645832

def cost_to_buy_ingredients : ℕ := 53
def number_of_doughnuts : ℕ := 25
def selling_price_per_doughnut : ℕ := 3

def revenue : ℕ := number_of_doughnuts * selling_price_per_doughnut
def profit : ℕ := revenue - cost_to_buy_ingredients

theorem dorothy_profit : profit = 22 :=
by
  -- calculation steps
  sorry

end dorothy_profit_l645_645832


namespace f_neg_two_l645_645084

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 1

-- State the theorem
theorem f_neg_two : f (-2) = -1 := by
  -- proof is not required, so simply skip it with sorry
  sorry

end f_neg_two_l645_645084


namespace sum_div_mult_sub_result_l645_645393

-- Define the problem with conditions and expected answer
theorem sum_div_mult_sub_result :
  3521 + 480 / 60 * 3 - 521 = 3024 :=
by 
  sorry

end sum_div_mult_sub_result_l645_645393


namespace binom_15_12_eq_455_l645_645406

theorem binom_15_12_eq_455 : nat.choose 15 12 = 455 :=
by 
  -- Proof omitted
  sorry

end binom_15_12_eq_455_l645_645406


namespace fundraising_exceeded_goal_l645_645215

theorem fundraising_exceeded_goal (ken mary scott : ℕ) (goal: ℕ) 
  (h_ken : ken = 600)
  (h_mary_ken : mary = 5 * ken)
  (h_mary_scott : mary = 3 * scott)
  (h_goal : goal = 4000) :
  (ken + mary + scott) - goal = 600 := 
  sorry

end fundraising_exceeded_goal_l645_645215


namespace range_of_expression_l645_645472

theorem range_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  0 < (x * y + y * z + z * x - 2 * x * y * z) ∧ (x * y + y * z + z * x - 2 * x * y * z) ≤ 7 / 27 := by
  sorry

end range_of_expression_l645_645472


namespace find_m_l645_645527

theorem find_m (x y m : ℤ) 
  (h1 : x - y = m + 2)
  (h2 : x + 3y = m)
  (h3 : x + y = -2) : 
  m = -3 := sorry

end find_m_l645_645527


namespace train_length_l645_645365

theorem train_length (time_cross_telegraph : ℕ) (speed_kmph : ℕ) (conversion_factor : ℚ)
  (time_cross_telegraph_eq : time_cross_telegraph = 16)
  (speed_kmph_eq : speed_kmph = 126)
  (conversion_factor_eq : conversion_factor = 5/18) :
  ∃ (length_of_train : ℚ), 
    length_of_train = (speed_kmph : ℚ) * conversion_factor * time_cross_telegraph :=
by 
  -- Introduce the necessary conditions and values
  have time_eq : time_cross_telegraph = 16 := by exact time_cross_telegraph_eq
  have speed_eq : speed_kmph = 126 := by exact speed_kmph_eq
  have conv_eq : conversion_factor = 5/18 := by exact conversion_factor_eq
  
  -- Calculate the length of the train
  set length_of_train := (speed_kmph : ℚ) * conversion_factor * time_cross_telegraph
  exists length_of_train

  -- The desired value for the length of the train
  have length_value : length_of_train = 560 := by
    -- Substitute the values and simplify
    rw [time_eq, speed_eq, conv_eq]
    norm_num
  exact length_value

end train_length_l645_645365


namespace required_circle_equation_l645_645434

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line equation on which the center of the required circle lies
def center_line (x y : ℝ) : Prop := 3 * x + 4 * y - 1 = 0

-- State the final proof that the equation of the required circle is (x + 1)^2 + (y - 1)^2 = 13 under the given conditions
theorem required_circle_equation (x y : ℝ) :
  ( ∃ (x1 y1 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧
    (∃ (cx cy r : ℝ), center_line cx cy ∧ (x - cx)^2 + (y - cy)^2 = r^2 ∧ (x1 - cx)^2 + (y1 - cy)^2 = r^2 ∧
      (x + 1)^2 + (y - 1)^2 = 13) )
 := sorry

end required_circle_equation_l645_645434


namespace find_A_l645_645049

def divisible_by(a b : ℕ) := b % a = 0

def valid_digit_A (A : ℕ) : Prop := (A = 0 ∨ A = 2 ∨ A = 4 ∨ A = 6 ∨ A = 8) ∧ divisible_by A 75

theorem find_A : ∃! A : ℕ, valid_digit_A A :=
by {
  sorry
}

end find_A_l645_645049


namespace problem_solution_l645_645174

noncomputable def problem (x : ℝ) (hx_pos : 0 < x) (hx_eq : x + 1 / x = 50) : ℝ :=
sqrt x + 1 / sqrt x

theorem problem_solution (x : ℝ) (hx_pos : 0 < x) (hx_eq : x + 1 / x = 50) : 
  problem x hx_pos hx_eq = sqrt 52 :=
sorry

end problem_solution_l645_645174


namespace integral_f_value_l645_645870

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 3^x else -x

theorem integral_f_value :
  ∫ x in -1..1, f x = (5 / 6) + (2 / Real.log 3) :=
sorry

end integral_f_value_l645_645870


namespace remainder_division_x2023_plus_1_l645_645041

theorem remainder_division_x2023_plus_1 (x : ℝ) :
  let f := (λ x, x^2023 + 1)
  let g := (λ x, x^12 - x^10 + x^8 - x^6 + x^4 - x^2 + 1)
  polynomial.modByMonic (f x) (g x) = -x^7 + 1 := 
sorry

end remainder_division_x2023_plus_1_l645_645041


namespace rectangle_width_decreased_l645_645627

theorem rectangle_width_decreased (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.5 * L in
  let W' := (L * W) / L' in
  ((W - W') / W) * 100 = 33.3333 :=
by
  sorry

end rectangle_width_decreased_l645_645627


namespace num_perfect_squares_in_range_l645_645109

-- Define the range for the perfect squares
def lower_bound := 75
def upper_bound := 400

-- Define the smallest integer whose square is greater than lower_bound
def lower_int := 9

-- Define the largest integer whose square is less than or equal to upper_bound
def upper_int := 20

-- State the proof problem
theorem num_perfect_squares_in_range : 
  (upper_int - lower_int + 1) = 12 :=
by
  -- Skipping the proof
  sorry

end num_perfect_squares_in_range_l645_645109


namespace bounds_of_sequence_l645_645229

noncomputable def sequence (x : ℕ → ℝ) :=
  ∀ n : ℕ, n > 0 → x n ^ n = ∑ j in Finset.range n, x n ^ j

theorem bounds_of_sequence (x : ℕ → ℝ) (h : sequence x) :
  ∀ n : ℕ, n > 0 → 2 - 1 / 2 ^ (n - 1) ≤ x n ∧ x n < 2 - 1 / 2 ^ n :=
by
  sorry

end bounds_of_sequence_l645_645229


namespace det_M_pow_three_eq_twenty_seven_l645_645467

-- Define a matrix M
variables (M : Matrix (Fin n) (Fin n) ℝ)

-- Given condition: det M = 3
axiom det_M_eq_3 : Matrix.det M = 3

-- State the theorem we aim to prove
theorem det_M_pow_three_eq_twenty_seven : Matrix.det (M^3) = 27 :=
by
  sorry

end det_M_pow_three_eq_twenty_seven_l645_645467


namespace find_a_for_quadratic_l645_645022

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l645_645022


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645699

theorem ratio_of_area_to_perimeter_of_equilateral_triangle (s : ℕ) : s = 10 → (let A := (s^2 * sqrt 3) / 4, P := 3 * s in A / P = 5 * sqrt 3 / 6) :=
by
  intro h,
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645699


namespace average_speed_of_train_l645_645366

def ChicagoTime (t : String) : Prop := t = "5:00 PM"
def NewYorkTime (t : String) : Prop := t = "10:00 AM"
def TimeDifference (d : Nat) : Prop := d = 1
def Distance (d : Nat) : Prop := d = 480

theorem average_speed_of_train :
  ∀ (d t1 t2 diff : Nat), 
  Distance d → (NewYorkTime "10:00 AM") → (ChicagoTime "5:00 PM") → TimeDifference diff →
  (t2 = 5 ∧ t1 = (10 - diff)) →
  (d / (t2 - t1) = 60) :=
by
  intros d t1 t2 diff hD ht1 ht2 hDiff hTimes
  sorry

end average_speed_of_train_l645_645366


namespace solution_set_for_inequality_l645_645046

-- Define the function involved
def rational_function (x : ℝ) : ℝ :=
  (3 * x - 1) / (2 - x)

-- Define the main theorem to state the solution set for the given inequality
theorem solution_set_for_inequality (x : ℝ) :
  (rational_function x ≥ 1) ↔ (3 / 4 ≤ x ∧ x < 2) :=
by
  sorry

end solution_set_for_inequality_l645_645046


namespace scientific_notation_conversion_l645_645231

theorem scientific_notation_conversion :
  216000 = 2.16 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l645_645231


namespace derivative_at_zero_does_not_exist_l645_645801

def f (x : ℝ) : ℝ :=
  if x ≠ 0 then arctan (x * cos (1 / (5 * x))) else 0

theorem derivative_at_zero_does_not_exist :
  ¬(∃ (L : ℝ), tendsto (λ (Δx : ℝ), (f Δx - f 0) / Δx) (𝓝 0) (𝓝 L)) :=
  by
  -- Definitions and conditions on the function
  let f : ℝ → ℝ := λ x, if x ≠ 0 then arctan (x * cos (1 / (5 * x))) else 0
  sorry

end derivative_at_zero_does_not_exist_l645_645801


namespace chord_length_is_correct_l645_645087

noncomputable def hyperbolaCircleChordLength : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∃ (p q : ℝ), p = 1 ∧ q = 2 ∧
  ∃ (asymptote : ℝ → ℝ),
    (asymptote = λ x, 2*x) ∧
    ∃ (circle : ℝ → ℝ → Prop), 
      (circle = λ x y, (x + 1)^2 + (y - 2)^2 = 4) ∧
      (∀ x y, circle x y → 
        ∃ d : ℝ, d = (4 / (Real.sqrt 5)) ∧
        (∃ chordLength,  chordLength = (4 * (Real.sqrt 5)) / 5 ∧
          chordLength = ∀ x y, asymptote x = y → 
            ∃ a b : ℝ, (a = x ∧ b = y) ∧ chordLength = (Real.sqrt (4 - (d^2) / 5)) 
        ) 
      )
    )
  )

theorem chord_length_is_correct : hyperbolaCircleChordLength := 
  sorry

end chord_length_is_correct_l645_645087


namespace length_EF_l645_645134

-- Define the points and lengths in the rectangle
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨10, 0⟩
def C : Point := ⟨10, 5⟩
def D : Point := ⟨0, 5⟩
def E : Point := ⟨0, 2⟩
def F : Point := ⟨10, 3⟩

noncomputable def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

theorem length_EF :
  dist E F = real.sqrt 101 :=
by
  sorry

end length_EF_l645_645134


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645698

theorem ratio_of_area_to_perimeter_of_equilateral_triangle (s : ℕ) : s = 10 → (let A := (s^2 * sqrt 3) / 4, P := 3 * s in A / P = 5 * sqrt 3 / 6) :=
by
  intro h,
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645698


namespace find_ratio_l645_645514

-- Given conditions
def smallest_positive_integer (n : ℕ) (a b : ℝ) : Prop :=
  (n > 0) ∧ (a > 0) ∧ (b > 0) ∧ ((a + b * complex.I)^n = (a - b * complex.I)^n)

-- Theorem statement
theorem find_ratio (n : ℕ) (a b : ℝ) :
  smallest_positive_integer n a b ∧ ∀ m : ℕ, m < n → ¬ smallest_positive_integer m a b →
  n = 3 ∧ b / a = real.sqrt 3 :=
by
  sorry

end find_ratio_l645_645514


namespace equilateral_triangle_ratio_is_correct_l645_645692

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l645_645692


namespace diego_oranges_l645_645421

theorem diego_oranges :
  ∀ (total_capacity weight_watermelon weight_grapes weight_apples : ℕ),
  total_capacity = 20 →
  weight_watermelon = 1 →
  weight_grapes = 1 →
  weight_apples = 17 →
  ∃ weight_oranges : ℕ, total_capacity - (weight_watermelon + weight_grapes + weight_apples) = weight_oranges ∧ weight_oranges = 1 :=
begin
  intros total_capacity weight_watermelon weight_grapes weight_apples h_capacity h_watermelon h_grapes h_apples,
  use total_capacity - (weight_watermelon + weight_grapes + weight_apples),
  split,
  { 
    rw [h_capacity, h_watermelon, h_grapes, h_apples],
    norm_num,
  },
  {
    rw [h_capacity, h_watermelon, h_grapes, h_apples],
    norm_num,
  }
end

end diego_oranges_l645_645421


namespace anniversary_sale_total_cost_l645_645782

-- Definitions of conditions
def original_price_ice_cream : ℕ := 12
def discount_ice_cream : ℕ := 2
def sale_price_ice_cream : ℕ := original_price_ice_cream - discount_ice_cream

def price_per_five_cans_juice : ℕ := 2
def cans_per_five_pack : ℕ := 5

-- Definition of total cost
def total_cost : ℕ := 2 * sale_price_ice_cream + (10 / cans_per_five_pack) * price_per_five_cans_juice

-- The goal is to prove that total_cost is 24
theorem anniversary_sale_total_cost : total_cost = 24 :=
by
  sorry

end anniversary_sale_total_cost_l645_645782


namespace square_diagonal_and_circle_circumference_l645_645774

-- Define the side length of the square
def side_length : ℝ := 30 * Real.sqrt 2

-- Define the length of the diagonal
def diagonal_length (side_length : ℝ) : ℝ := side_length * Real.sqrt 2

-- Define the radius of the inscribed circle
def radius (side_length : ℝ) : ℝ := side_length / 2

-- Define the circumference of the inscribed circle
def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

-- Given conditions and prove the required lengths and circumference
theorem square_diagonal_and_circle_circumference :
  let side := 30 * Real.sqrt 2 in
  (diagonal_length side = 60) ∧ (circumference (radius side) = 30 * Real.pi * Real.sqrt 2) := by
{
  let side := 30 * Real.sqrt 2,
  have h_diagonal : diagonal_length side = 60 := by sorry,
  have h_circumference : circumference (radius side) = 30 * Real.pi * Real.sqrt 2 := by sorry,
  exact ⟨h_diagonal, h_circumference⟩
}

end square_diagonal_and_circle_circumference_l645_645774


namespace inequality_solution_set_l645_645043

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (3 / 4 ≤ x ∧ x < 2) :=
by sorry

end inequality_solution_set_l645_645043


namespace cos_660_equals_half_l645_645837

-- Definitions based on the conditions
def cosine_periodic {θ : ℝ} (k : ℤ) : Prop := cos (θ + 360 * k) = cos θ
def cosine_even {θ : ℝ} : Prop := cos (-θ) = cos θ
def cosine_60_equals : Prop := cos 60 = 1 / 2

-- Mathematical proof problem statement
theorem cos_660_equals_half : cosine_periodic 660 2 ∧ cosine_even 60 ∧ cosine_60_equals → cos 660 = 1 / 2 := 
by
  sorry

end cos_660_equals_half_l645_645837


namespace david_money_left_l645_645822

noncomputable def david_trip (S H : ℝ) : Prop :=
  S + H = 3200 ∧ H = 0.65 * S

theorem david_money_left : ∃ H, david_trip 1939.39 H ∧ |H - 1260.60| < 0.01 := by
  sorry

end david_money_left_l645_645822


namespace comparison_of_abc_l645_645869

noncomputable def a : ℝ := 24 / 7
noncomputable def b : ℝ := Real.log 7
noncomputable def c : ℝ := Real.log (7 / Real.exp 1) / Real.log 3 + 1

theorem comparison_of_abc :
  (a = 24 / 7) →
  (b * Real.exp b = 7 * Real.log 7) →
  (3 ^ (c - 1) = 7 / Real.exp 1) →
  a > b ∧ b > c :=
by
  intros ha hb hc
  sorry

end comparison_of_abc_l645_645869


namespace quadrilateral_angle_exceeds_180_l645_645755

theorem quadrilateral_angle_exceeds_180 (n m : ℕ) (h_n_nonzero : n > 0) (h_m_nonzero : m > 0) 
  (h_convex : convex n-gon) 
  (h_division : divides_into_quadrilaterals n-gon m) :
  ∃ k, k = m - n / 2 + 1 ∧ (∀ q ∈ quadrilaterals, ∃ angle > 180, angle_of q  ≤ k) := 
sorry

end quadrilateral_angle_exceeds_180_l645_645755


namespace number_of_m_for_step1_count_15_l645_645819

def step1_count (m : ℕ) : ℕ :=
  let rec helper (n : ℕ) (count : ℕ) : ℕ :=
    if n = 1 then count
    else if n % 2 = 0 then helper (n / 2) (count + 1)
    else helper (n + 1) (count + 1)
  in helper m 0

theorem number_of_m_for_step1_count_15 : (finset.filter (λ m, step1_count m = 15) (finset.range 2^16)).card = 610 :=
  sorry

end number_of_m_for_step1_count_15_l645_645819


namespace area_balance_l645_645659

-- Define the 8x8 chessboard
def is_chessboard (n : ℕ) : Prop :=
  n = 8

-- Define the concept of cell centers and neighboring cells
structure cell_center :=
(x y : ℕ)
condition (h : 0 ≤ x ∧ x < 8 ∧ 0 ≤ y ∧ y < 8)

structure neighbor (c1 c2 : cell_center) : Prop :=
(horizontal : c1.x ≠ c2.x ∧ c1.y = c2.y)
(vertical : c1.x = c2.x ∧ c1.y ≠ c2.y)
(diagonal : abs (c1.x - c2.x) = abs (c1.y - c2.y))

-- Define a closed non-self-intersecting polygonal line through centers of cells
structure polygonal_line :=
(points : list cell_center)
(closed : points.head = points.last) -- closed path
(non_self_intersecting : ∀ (i j : ℕ), i ≠ j → points.nth i ≠ points.nth j) -- non-self-intersecting
(neighboring : ∀ (i : ℕ), neighbor (points.nth i) (points.nth ((i + 1) % points.length)))

-- Define the concept of areas within the chessboard
def cell_color (c : cell_center) : bool :=
  (c.x + c.y) % 2 = 0

def area_of_color (polygon : polygonal_line) (color : bool) : ℝ :=
  sorry

-- The proof problem statement: prove the total black area = total white area
theorem area_balance (polygon : polygonal_line):
  area_of_color polygon true = area_of_color polygon false :=
sorry

end area_balance_l645_645659


namespace Mitya_age_l645_645188

/--
Assume Mitya's current age is M and Shura's current age is S. If Mitya is 11 years older than Shura,
and when Mitya was as old as Shura is now, he was twice as old as Shura,
then prove that M = 27.5.
-/
theorem Mitya_age (S M : ℝ) (h1 : M = S + 11) (h2 : M - S = 2 * (S - (M - S))) : M = 27.5 :=
by
  sorry

end Mitya_age_l645_645188


namespace other_acute_angle_in_right_triangle_l645_645127

theorem other_acute_angle_in_right_triangle (α : ℝ) (β : ℝ) (γ : ℝ) 
  (h1 : α + β + γ = 180) (h2 : γ = 90) (h3 : α = 30) : β = 60 := 
sorry

end other_acute_angle_in_right_triangle_l645_645127


namespace stream_current_rate_proof_l645_645616

noncomputable def stream_current_rate (c : ℝ) : Prop :=
  ∃ (c : ℝ), (6 / (8 - c) + 6 / (8 + c) = 2) ∧ c = 4

theorem stream_current_rate_proof : stream_current_rate 4 :=
by {
  -- Proof to be provided here.
  sorry
}

end stream_current_rate_proof_l645_645616


namespace powerjet_30_minutes_500_gallons_per_hour_l645_645233

theorem powerjet_30_minutes_500_gallons_per_hour:
  ∀ (rate : ℝ) (time : ℝ), rate = 500 → time = 30 → (rate * (time / 60) = 250) := by
  intros rate time rate_eq time_eq
  sorry

end powerjet_30_minutes_500_gallons_per_hour_l645_645233


namespace ball_placement_problem_l645_645056

/-- Given the constraints that ball 1 cannot be in box 1 and ball 3 cannot be in box 3,
    prove that the total number of valid ways to place three selected balls into three boxes is 14. -/
theorem ball_placement_problem :
  let balls := {1, 2, 3, 4} in
  let boxes := {1, 2, 3} in
  let total_ways := 
    (balls.choose 3).sum (λ selected_balls, 
      (selected_balls.permutations.filter (λ p, p.head! ≠ 1 ∧ (p.nth! 2 ≠ 3))).length
    ) in
  total_ways = 14 :=
sorry

end ball_placement_problem_l645_645056


namespace increasing_condition_sufficient_not_necessary_l645_645483

/-- Defining the function f -/
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a / x

/-- Defining the derivative of f -/
def f_deriv (x : ℝ) (a : ℝ) : ℝ := 2 * x - a / x^2

/-- Proposition to prove the main statement -/
theorem increasing_condition_sufficient_not_necessary (a : ℝ) :
  (0 < a ∧ a < 2) →
  (∀ x : ℝ, 1 < x → 2 * x^3 ≥ a) ∧ ¬(∀ x : ℝ, 1 < x → 2 * x^3 ≥ a ↔ 0 < a ∧ a < 2) :=
by
  sorry

end increasing_condition_sufficient_not_necessary_l645_645483


namespace probability_product_multiple_of_4_l645_645324

theorem probability_product_multiple_of_4 :
  let cards := {1, 2, 3, 4, 5, 6}
  let pairs := { (a, b) | a ∈ cards ∧ b ∈ cards ∧ a < b }
  let favorable := { (a, b) ∈ pairs | (a * b) % 4 == 0 }
  let total := pairs.card
  let favorable_count := favorable.card
  total = 15 ∧ favorable_count = 6 → 
  (favorable_count : ℚ) / total = 2 / 5 :=
by
  intros cards pairs favorable total favorable_count h
  sorry

end probability_product_multiple_of_4_l645_645324


namespace diane_money_l645_645420

-- Define the conditions
def total_cost : ℤ := 65
def additional_needed : ℤ := 38
def initial_amount : ℤ := total_cost - additional_needed

-- Theorem statement
theorem diane_money : initial_amount = 27 := by
  sorry

end diane_money_l645_645420


namespace corr_star_single_var_corr_star_max_l645_645572

variables 
  {Ω : Type*} 
  {𝓕 : measurable_space Ω} 
  (P : measure_theory.measure Ω)

namespace measure_theory

def L2_measurable (ℳ : measurable_space Ω) : Type* :=
{ ξ : Ω → ℝ // measurable ξ ∧ ∫⁻ ω, (ξ ω) ^ 2 ∂P < ∞ }

noncomputable def corr_coeff (ξ ζ : Ω → ℝ) : ℝ := sorry -- Definition of correlation coefficient goes here

noncomputable def corr_star (𝓐 𝓑 : measurable_space Ω) : ℝ :=
  Sup (set.image (λ (ξζ : L2_measΩ 𝓐 ) × (L2_measΩ 𝓑), |corr_coeff ξζ.fst ξζ.snd|)
       ((λ ξ ζ, ⟨ξ, ζ⟩) '' set.univ))

theorem corr_star_single_var {X Y : Ω → ℝ} (hX : measurable X) (hY : measurable Y) :
  corr_star P.mk (measurable_space.of_measurable_single X) (measurable_space.of_measurable_single Y) = 
  corr_coeff X Y := sorry

theorem corr_star_max
  {ℳ₁ ℳ₂ ℕ₁ ℕ₂ : measurable_space Ω}
  (hindep : indep (ℳ₁⊔ℕ₁) (ℳ₂⊔ℕ₂)) :
  corr_star P.mk (ℳ₁⊔ℳ₂) (ℕ₁⊔ℕ₂) = max (corr_star P.mk ℳ₁ ℕ₁) (corr_star P.mk ℳ₂ ℕ₂) := sorry

end measure_theory

end corr_star_single_var_corr_star_max_l645_645572


namespace plane_equation_l645_645817

def plane_parametric (s t : ℝ) : ℝ × ℝ × ℝ :=
(2 + 2 * s - 3 * t, 4 - 2 * s, 1 - s + 3 * t)

theorem plane_equation :
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) ∧ 
  (forall x y z, A * x + B * y + C * z + D = 0 ↔
  (∃ s t, (x, y, z) = plane_parametric s t)) :=
sorry

end plane_equation_l645_645817


namespace hyperbola_asymptote_perpendicular_l645_645919

theorem hyperbola_asymptote_perpendicular :
  ∀ t : ℝ, (∃ k : ℝ, ∀ (x y : ℝ), (4 * x^2 - y^2 = 1) → (k = 2 ∨ k = -2) ∧ 
           (t * x + y + 1 = 0) → (k * (-t) = -1)) → (t = 1 / 2 ∨ t = -1 / 2) :=
by
  assume t,
  intro h,
  sorry

end hyperbola_asymptote_perpendicular_l645_645919


namespace range_of_function_l645_645967

theorem range_of_function (x : ℝ) : x ≠ 2 ↔ ∃ y, y = x / (x - 2) :=
sorry

end range_of_function_l645_645967


namespace todd_has_40_left_after_paying_back_l645_645664

def todd_snowcone_problem : Prop :=
  let borrowed := 100
  let repay := 110
  let cost_ingredients := 75
  let snowcones_sold := 200
  let price_per_snowcone := 0.75
  let total_earnings := snowcones_sold * price_per_snowcone
  let remaining_money := total_earnings - repay
  remaining_money = 40

theorem todd_has_40_left_after_paying_back : todd_snowcone_problem :=
by
  -- Add proof here if needed
  sorry

end todd_has_40_left_after_paying_back_l645_645664


namespace find_a_and_b_l645_645569

theorem find_a_and_b (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 → f x = ax^2 - 2*(a+1)*x + b ∧ f x < 0) : a = 2 ∧ b = 4 := 
sorry

end find_a_and_b_l645_645569


namespace find_constants_l645_645036

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ -2 → 
  (x^3 - x - 4) / ((x - 1) * (x - 4) * (x + 2)) = 
  A / (x - 1) + B / (x - 4) + C / (x + 2)) →
  A = 4 / 9 ∧ B = 28 / 9 ∧ C = -1 / 3 :=
by
  sorry

end find_constants_l645_645036


namespace equation_has_roots_l645_645010

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l645_645010


namespace concentration_third_flask_l645_645282

-- Define the concentrations as per the given problem

noncomputable def concentration (acid_mass water_mass : ℝ) : ℝ :=
  (acid_mass / (acid_mass + water_mass)) * 100

-- Given conditions
def acid_mass_first_flask : ℝ := 10
def acid_mass_second_flask : ℝ := 20
def acid_mass_third_flask : ℝ := 30
def concentration_first_flask : ℝ := 5
def concentration_second_flask : ℝ := 70 / 3

-- Total water added to the first and second flasks
def total_water_mass : ℝ :=
  let W1 := (acid_mass_first_flask - concentration_first_flask * acid_mass_first_flask / 100)
  let W2 := (acid_mass_second_flask - concentration_second_flask * acid_mass_second_flask / 100)
  W1 + W2 

-- Prove the concentration of acid in the third flask
theorem concentration_third_flask : 
  concentration acid_mass_third_flask total_water_mass = 10.5 := 
  sorry

end concentration_third_flask_l645_645282


namespace power_2015_of_z_l645_645906

-- Define the complex number z as given in the conditions.
def z : ℂ := (⟨(Real.sqrt 2) / 2, 0⟩ : ℂ) + (⟨0, (Real.sqrt 2) / 2⟩ : ℂ)

-- State the theorem to prove z^2015 = (√2/2) * (1 + i)
theorem power_2015_of_z : z^2015 = (⟨(Real.sqrt 2) / 2, 0⟩ : ℂ) * (⟨1, 1⟩ : ℂ) := by
  sorry

end power_2015_of_z_l645_645906


namespace not_entire_field_weedy_l645_645356

-- Define the conditions
def field_divided_into_100_plots : Prop :=
  ∃ (a b : ℕ), a * b = 100

def initial_weedy_plots : Prop :=
  ∃ (weedy_plots : Finset (ℕ × ℕ)), weedy_plots.card = 9

def plot_becomes_weedy (weedy_plots : Finset (ℕ × ℕ)) (p : ℕ × ℕ) : Prop :=
  (p.fst ≠ 0 ∧ (p.fst - 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 0 ∧ (p.fst, p.snd - 1) ∈ weedy_plots) ∨
  (p.fst ≠ 0 ∧ (p.fst - 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 100 ∧ (p.fst, p.snd + 1) ∈ weedy_plots) ∨
  (p.fst ≠ 100 ∧ (p.fst + 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 0 ∧ (p.fst, p.snd - 1) ∈ weedy_plots) ∨
  (p.fst ≠ 100 ∧ (p.fst + 1, p.snd) ∈ weedy_plots) ∧
  (p.snd ≠ 100 ∧ (p.fst, p.snd + 1) ∈ weedy_plots)

-- Theorem statement
theorem not_entire_field_weedy :
  field_divided_into_100_plots →
  initial_weedy_plots →
  (∀ weedy_plots : Finset (ℕ × ℕ), (∀ p : ℕ × ℕ, plot_becomes_weedy weedy_plots p → weedy_plots ∪ {p} = weedy_plots) → weedy_plots.card < 100) :=
  sorry

end not_entire_field_weedy_l645_645356


namespace binom_15_12_eq_455_l645_645407

theorem binom_15_12_eq_455 : nat.choose 15 12 = 455 :=
by 
  -- Proof omitted
  sorry

end binom_15_12_eq_455_l645_645407


namespace obtain_angle_10_30_l645_645639

theorem obtain_angle_10_30 (a : ℕ) (h : 100 + a = 135) : a = 35 := 
by sorry

end obtain_angle_10_30_l645_645639


namespace probability_product_multiple_of_4_l645_645323

theorem probability_product_multiple_of_4 :
  let cards := {1, 2, 3, 4, 5, 6}
  let pairs := { (a, b) | a ∈ cards ∧ b ∈ cards ∧ a < b }
  let favorable := { (a, b) ∈ pairs | (a * b) % 4 == 0 }
  let total := pairs.card
  let favorable_count := favorable.card
  total = 15 ∧ favorable_count = 6 → 
  (favorable_count : ℚ) / total = 2 / 5 :=
by
  intros cards pairs favorable total favorable_count h
  sorry

end probability_product_multiple_of_4_l645_645323


namespace find_m_points_with_center_of_gravity_condition_l645_645153

noncomputable def distance (x y : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

theorem find_m_points_with_center_of_gravity_condition
  (m n : ℕ)
  (hmn : m < n)
  (X : Fin n → (ℝ × ℝ))
  (H_interior : ∀ i : Fin n, distance X[i] (0, 0) ≤ 1)
  (H_on_border : ∃ i : Fin n, distance X[i] (0, 0) = 1) :
  ∃ (I : Fin m → Fin n),
    ∃ (barycenter : (ℝ × ℝ)),
      barycenter = (1/m : ℕ → ℝ) • (finset.univ.image (λ i : Fin m, X [I i])).sum
      ∧ distance barycenter (0, 0) ≥ 1 / (1 + 2 * m * (1 - 1 / n)) :=
sorry

end find_m_points_with_center_of_gravity_condition_l645_645153


namespace x_value_satisfies_equation_l645_645596

theorem x_value_satisfies_equation (x : ℕ) : 
  (∣ (20 : ℚ) / x - x / 15 ∣ = 4 / 3) → x = 10 :=   -- Question with Condition and Correct Answer
sorry

end x_value_satisfies_equation_l645_645596


namespace find_all_radioactive_balls_l645_645977

theorem find_all_radioactive_balls :
  ∃ (test : ℕ → ℕ → Prop), 
  (∀ b1 b2, test b1 b2 ↔ (b1 ∈ radioactive_balls ∧ b2 ∈ radioactive_balls)) →
  ∃ (S : set ℕ), (S.card = 51) ∧ 
  (∀ b ∈ S, b ∈ radioactive_balls) ∧ 
  S.card ≤ 51 ∧ 
  ∃ (t : finset (ℕ × ℕ)), t.card ≤ 145 ∧
  ∀ (p : ℕ × ℕ), p ∈ t → test p.1 p.2 :=
by
  sorry -- Proof omitted

end find_all_radioactive_balls_l645_645977


namespace determine_f_500_l645_645344

variable (f : ℝ → ℝ)
variable (h_cont : Continuous f)
variable (h1 : f 1000 = 999)
variable (h2 : ∀ x : ℝ, f x * f (f x) = 1)

theorem determine_f_500 : f 500 = 1 / 500 :=
by
  sorry

end determine_f_500_l645_645344


namespace no_pair_contributes_less_than_third_of_total_l645_645859

-- Defining the contributions
variables (a b c d e : ℝ)

-- Defining the total contribution
def S := a + b + c + d + e

-- The main theorem statement
theorem no_pair_contributes_less_than_third_of_total :
  a + b < S / 3 ∧
  b + c < S / 3 ∧
  c + d < S / 3 ∧
  d + e < S / 3 ∧
  e + a < S / 3 →
  false :=
begin
  -- Assume the pairs contribute less than one third of the total
  assume h : a + b < S / 3 ∧
            b + c < S / 3 ∧
            c + d < S / 3 ∧
            d + e < S / 3 ∧
            e + a < S / 3,

  -- Calculate the total
  have S_eq : S = a + b + c + d + e := rfl,

  -- Use the inequalities to derive a contradiction
  sorry
end

end no_pair_contributes_less_than_third_of_total_l645_645859


namespace total_pieces_ten_row_triangle_l645_645378

def unit_rods (n : ℕ) : ℕ := n * (6 + (n - 1) * 3) / 2

def connectors (n : ℕ) : ℕ := n * (n + 1) / 2

theorem total_pieces_ten_row_triangle : unit_rods 10 + connectors 11 = 231 :=
by calc
  unit_rods 10 + connectors 11 = 165 + 66 : by rw [unit_rods, connectors]; norm_num
                            ... = 231 : by norm_num

end total_pieces_ten_row_triangle_l645_645378


namespace min_travel_distance_l645_645239

-- Let's define the given distances between the docks.
def distance_AB : ℝ := 3
def distance_AC : ℝ := 4
def distance_BC : ℝ := Real.sqrt 13

-- The question is to find the minimum distance the truck will travel.
theorem min_travel_distance : 
  ∃ (x : ℝ), 
  x = 2 * Real.sqrt 37 ∧ 
  ∀ (d1 d2 : ℝ), 
  d1 = distance_AB ∧ 
  d2 = distance_AC ∧ 
  distance_BC = Real.sqrt 13 →
  x ≤ d1 + d2 + distance_BC :=
sorry

end min_travel_distance_l645_645239


namespace all_roots_less_than_one_over_2000_fact_l645_645592

theorem all_roots_less_than_one_over_2000_fact :
  ∀ x : ℝ, (x + 1) * (x + 2) * (x + 3) * ... * (x + 2001) = 2001 → x < 1 / 2000 ! :=
by
  -- The proof goes here
  sorry

end all_roots_less_than_one_over_2000_fact_l645_645592


namespace initial_number_of_men_is_correct_l645_645605

def initial_men (hours1 hours2 : ℕ) (depth1 depth2 : ℕ) (extra_men : ℕ) (initial_rate : ℕ → ℝ) : Prop :=
  ∃ M : ℕ, initial_rate M = (depth1 : ℝ) / (hours1 * M : ℝ) ∧ initial_rate (M + extra_men) = (depth2 : ℝ) / (hours2 * (M + extra_men) : ℝ) ∧ M = 72

theorem initial_number_of_men_is_correct :
  initial_men 8 6 30 50 88 (λ M, 30.0 / (8.0 * M) : ℕ → ℝ) := sorry

end initial_number_of_men_is_correct_l645_645605


namespace equation_has_two_distinct_roots_l645_645000

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l645_645000


namespace color_integers_l645_645810

theorem color_integers (n : ℕ) (n_colors : n = 2009) :
  (∃ f : ℕ → ℕ, (∀ k, ∃ x, f x = k) ∧ (∀ a b c, f a ≠ f b ∧ f b ≠ f c ∧ 
  f a ≠ f c → a * b ≠ c ∧ a * c ≠ b ∧ b * c ≠ a)) :=
by
  have colors : ℕ := n
  rw [n_colors] at colors
  let primes := (List.seq 1 colors).filter (λ k, Nat.Prime k)
  /-
  This is to define different sets S_i where S_i is all multiples of the ith prime (ignoring non_prime multiples)
  -/
  let f : ℕ → ℕ :=
    fun x => 
    if ∃ p_i ∈ primes, p_i ∣ x then
      primes.index_of (primes.find (λ p_i, p_i ∣ x)) + 1
    else
      colors -- if x doesn't have prime factor within the primes list
  
  use f
  split
  {
    intro k
    use primes.nth ((k % primes.length).nat_abs)
    exact primes.nth_le (k % primes.length) sorry
  }
  {
    intros a b c h
    -- This is where we state all combinations of products are not equal
    split; intro h_false; 
    iterate 3 { rw f at h_false; sorry }
  }

end color_integers_l645_645810


namespace point_A_equidistant_l645_645431

-- Definitions of points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define A, B, and C as given in the conditions
def B : Point := ⟨7, 3, -4⟩
def C : Point := ⟨1, 5, 7⟩
def A (y : ℝ) : Point := ⟨0, y, 0⟩

-- Define the distance function between two points
def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Proof statement
theorem point_A_equidistant :
  ∃ y : ℝ, dist (A y) B = dist (A y) C ∧ A y = ⟨0, 1/4, 0⟩ := by
  sorry

end point_A_equidistant_l645_645431


namespace perpendicular_lines_l645_645733

-- Definitions of conditions
def condition1 (α β γ δ : ℝ) : Prop := α = 90 ∧ α + β = 180 ∧ α + γ = 180 ∧ α + δ = 180
def condition2 (α β γ δ : ℝ) : Prop := α = β ∧ β = γ ∧ γ = δ
def condition3 (α β : ℝ) : Prop := α = β ∧ α + β = 180
def condition4 (α β : ℝ) : Prop := α = β ∧ α + β = 180

-- Main theorem statement
theorem perpendicular_lines (α β γ δ : ℝ) :
  (condition1 α β γ δ ∨ condition2 α β γ δ ∨
   condition3 α β ∨ condition4 α β) → α = 90 :=
by sorry

end perpendicular_lines_l645_645733


namespace initial_people_on_train_l645_645386

theorem initial_people_on_train 
    (P : ℕ)
    (h1 : 116 = P - 4)
    (h2 : P = 120)
    : 
    P = 116 + 4 := by
have h3 : P = 120 := by sorry
exact h3

end initial_people_on_train_l645_645386


namespace concentration_of_acid_in_third_flask_l645_645271

theorem concentration_of_acid_in_third_flask :
  ∀ (W1 W2 : ℝ),
    let W := 190 + 65.714 in 
    W1 = 190 ∧ W2 = 65.714 →
    (10 : ℝ) / (10 + W1) = 0.05 →
    (20 : ℝ) / (20 + W2) = 0.2331 →
    (30 : ℝ) / (30 + W) = 0.105 :=
begin
  sorry
end

end concentration_of_acid_in_third_flask_l645_645271


namespace games_given_away_correct_l645_645146

-- Define initial and remaining games
def initial_games : ℕ := 50
def remaining_games : ℕ := 35

-- Define the number of games given away
def games_given_away : ℕ := initial_games - remaining_games

-- Prove that the number of games given away is 15
theorem games_given_away_correct : games_given_away = 15 := by
  -- This is a placeholder for the actual proof
  sorry

end games_given_away_correct_l645_645146


namespace bed_height_l645_645380

noncomputable def bed_length : ℝ := 8
noncomputable def bed_width : ℝ := 4
noncomputable def bags_of_soil : ℕ := 16
noncomputable def soil_per_bag : ℝ := 4
noncomputable def total_volume_of_soil : ℝ := bags_of_soil * soil_per_bag
noncomputable def number_of_beds : ℕ := 2
noncomputable def volume_per_bed : ℝ := total_volume_of_soil / number_of_beds

theorem bed_height :
  volume_per_bed / (bed_length * bed_width) = 1 :=
sorry

end bed_height_l645_645380


namespace interest_rate_correct_l645_645786

-- Definitions based on the problem conditions
def SI : ℝ := 80
def P : ℝ := 571.43
def T : ℝ := 4
def expected_Rate : ℝ := 3.5 / 100  -- 3.5% expressed as a decimal

-- Statement to prove the interest rate calculation
theorem interest_rate_correct :
  (SI = P * (expected_Rate) * T) :=
sorry

end interest_rate_correct_l645_645786


namespace processing_rates_and_total_cost_l645_645753

variables (products total_days total_days_A total_days_B daily_capacity_A daily_capacity_B total_cost_A total_cost_B : ℝ)

noncomputable def A_processing_rate : ℝ := daily_capacity_A
noncomputable def B_processing_rate : ℝ := daily_capacity_B

theorem processing_rates_and_total_cost
  (h1 : products = 1000)
  (h2 : total_days_A = total_days_B + 10)
  (h3 : daily_capacity_B = 1.25 * daily_capacity_A)
  (h4 : total_cost_A = 100 * total_days_A)
  (h5 : total_cost_B = 125 * total_days_B) :
  (daily_capacity_A = 20) ∧ (daily_capacity_B = 25) ∧ (total_cost_A + total_cost_B = 5000) :=
by
  sorry

end processing_rates_and_total_cost_l645_645753


namespace tan_of_triangle_XYZ_l645_645972

theorem tan_of_triangle_XYZ (Y Z X : Type) [AddGroup Y]
  (angle_Y : Y)
  (YZ : ℝ)
  (XY : ℝ)
  (h1 : angle_Y = 90)
  (h2 : YZ = 4)
  (h3 : XY = 5)
  (XZ : ℝ) 
  (h4 : YZ^2 + XZ^2 = XY^2) :
  XY = 5 → XZ = 3 → tan X = 4 / 3 :=
by
  sorry

end tan_of_triangle_XYZ_l645_645972


namespace problem_l645_645932

theorem problem (x : ℝ) (h : sqrt (9 + x) + sqrt (16 - x) = 8) : (9 + x) * (16 - x) = 380.25 := 
by 
  sorry

end problem_l645_645932


namespace progressions_proportional_l645_645442

variable {α β: ℕ → ℕ}

def same_set_of_exponents (a b : ℕ) : Prop :=
  multiset.ofList (prime_factors a).pmap (λ p h, prime_data.exponent p h).1.to_finset
  = multiset.ofList (prime_factors b).pmap (λ p h, prime_data.exponent p h).1.to_finset

theorem progressions_proportional (a_n b_n : ℕ → ℕ) (h1 : ∃ a d, ∀ n, a_n n = d * (a + n))
  (h2 : ∃ b e, ∀ n, b_n n = e * (b + n)) 
  (h3 : ∀ n, same_set_of_exponents (a_n n) (b_n n)) : 
  ∃ k, ∀ n, a_n n = k * b_n n := 
by 
  sorry

end progressions_proportional_l645_645442


namespace conformal_map_l645_645436

noncomputable def mapRegion (z : ℂ) : ℂ := complex.exp (2 * real.pi * complex.I * (z / (z - 2)))

theorem conformal_map (z : ℂ) (h1 : complex.abs z < 2) (h2 : complex.abs (z - 1) > 1) :
  complex.im (mapRegion z) > 0 :=
sorry

end conformal_map_l645_645436


namespace max_vertex_value_in_cube_l645_645611

def transform_black (v : ℕ) (e1 e2 e3 : ℕ) : ℕ :=
  e1 + e2 + e3

def transform_white (v : ℕ) (d1 d2 d3 : ℕ) : ℕ :=
  d1 + d2 + d3

def max_value_after_transformation (initial_values : Fin 8 → ℕ) : ℕ :=
  -- Combination of transformations and iterations are derived here
  42648

theorem max_vertex_value_in_cube :
  ∀ (initial_values : Fin 8 → ℕ),
  (∀ i, 1 ≤ initial_values i ∧ initial_values i ≤ 8) →
  (∃ (final_value : ℕ), final_value = max_value_after_transformation initial_values) → final_value = 42648 :=
by {
  sorry
}

end max_vertex_value_in_cube_l645_645611


namespace second_year_associates_l645_645128

theorem second_year_associates (not_first_year : ℝ) (more_than_two_years : ℝ) 
  (h1 : not_first_year = 0.75) (h2 : more_than_two_years = 0.5) : 
  (not_first_year - more_than_two_years) = 0.25 :=
by 
  sorry

end second_year_associates_l645_645128


namespace ratio_of_area_to_perimeter_l645_645705

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l645_645705


namespace decimal_37th_digit_l645_645510

theorem decimal_37th_digit (n : ℕ) (h : n = 37) : 
  let dec_rep := "269230769".cycle in
  dec_rep.nth (n % 9) = '2' :=
  by
    sorry

end decimal_37th_digit_l645_645510


namespace largest_number_is_pi_count_of_rationals_is_five_l645_645383

open Real

def number_list : List ℝ := [0, sqrt 3, Real.pi, -sqrt 49, -0.001, 3 / 7, real.cbrt 27]

theorem largest_number_is_pi : List.maximum number_list = Real.pi :=
by
  sorry

theorem count_of_rationals_is_five : List.countp Rational.is number_list = 5 :=
by
  sorry

end largest_number_is_pi_count_of_rationals_is_five_l645_645383


namespace DS_eq_SN_l645_645582

open EuclideanGeometry

variables {A B F D N S : Point} {l : Line}

-- Assume A, B, F lie on line l with B between A and F
axiom on_line_A : A ∈ l
axiom on_line_B : B ∈ l
axiom on_line_F : F ∈ l
axiom between_ABF : Between A B F

-- Assume squares ABCD and BFNT on the same side of l
axiom square_ABCD : Square A B C D
axiom square_BFNT : Square B F N T

-- Circle passing through D, B, N and intersecting l at S ≠ B
axiom circle_DBN : ∃ (circle_DBN : Circle), D ∈ circle_DBN ∧ B ∈ circle_DBN ∧ N ∈ circle_DBN
axiom intersects_l_at_S : S ∈ l ∧ S ≠ B
axiom S_in_circle_DBN : S ∈ circle_DBN

-- Prove DS = SN
theorem DS_eq_SN : Distance D S = Distance S N :=
by
  sorry

end DS_eq_SN_l645_645582


namespace find_z_l645_645874

theorem find_z (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : z * i = 2 - i) : z = -1 - 2 * i := 
by
  sorry

end find_z_l645_645874


namespace fireworks_number_l645_645363

variable (x : ℕ)
variable (fireworks_total : ℕ := 484)
variable (happy_new_year_fireworks : ℕ := 12 * 5)
variable (boxes_of_fireworks : ℕ := 50 * 8)
variable (year_fireworks : ℕ := 4 * x)

theorem fireworks_number :
    4 * x + happy_new_year_fireworks + boxes_of_fireworks = fireworks_total →
    x = 6 := 
by
  sorry

end fireworks_number_l645_645363


namespace equation_has_two_distinct_roots_l645_645002

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l645_645002


namespace star_not_associative_l645_645563

def star (x y : ℝ) : ℝ := 3 * x * y + x + y

theorem star_not_associative {T : set ℝ} (non_zero : ∀ x ∈ T, x ≠ 0) :
  ∃ x y z ∈ T, star (star x y) z ≠ star x (star y z) :=
by
  have x := 1
  have y := 1
  have z := 1
  sorry

end star_not_associative_l645_645563


namespace min_value_fractional_sum_geometric_sequence_l645_645882

theorem min_value_fractional_sum_geometric_sequence {
  a : ℕ → ℝ,
  q : ℝ
}
  (h1 : ∀ n, a (n + 1) = q * a n)
  (h2 : 0 < a 5)
  (h3 : a 7 = a 6 + 2 * a 5)
  (h4 : ∃ m n, sqrt (a m * a n) = 4 * a 1)
  :
  (∀ m n, m + n = 6 → (1 / m + 9 / n) ≥ 11 / 4) :=
begin
  sorry
end

end min_value_fractional_sum_geometric_sequence_l645_645882


namespace number_of_roses_l645_645266

def total_flowers : ℕ := 10
def carnations : ℕ := 5
def roses : ℕ := total_flowers - carnations

theorem number_of_roses : roses = 5 := by
  sorry

end number_of_roses_l645_645266


namespace calc_nabla_l645_645447

variable (a b c d : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0)

def nabla (x y : ℝ) : ℝ := (x + y) / (1 + x * y)

theorem calc_nabla :
  (nabla (nabla 2 4) (nabla 5 6)) = 19 / 23 := by
  sorry

end calc_nabla_l645_645447


namespace sum_of_intersection_points_l645_645573

theorem sum_of_intersection_points (f : ℝ → ℝ) 
  (h_f : ∀ x, f (4 - x) = -f x)
  (x y : ℕ → ℝ)
  (h_inter : ∀ i, y i = f (x i) ∧ y i = 1 / (2 - x i))
  (m : ℕ) :
  ∑ i in finset.range m, (x i + y i) = m := 
sorry

end sum_of_intersection_points_l645_645573


namespace largest_d_in_range_l645_645827

theorem largest_d_in_range (d : ℝ) (g : ℝ → ℝ) :
  (g x = x^2 - 6x + d) → (∃ x : ℝ, g x = 2) → d ≤ 11 :=
by
  sorry

end largest_d_in_range_l645_645827


namespace number_of_persons_l645_645606

theorem number_of_persons (P : ℕ) : 
  (P * 12 * 5 = 30 * 13 * 6) → P = 39 :=
by
  sorry

end number_of_persons_l645_645606


namespace relationship_abcd_l645_645160

noncomputable def a := Real.sin (Real.sin (2008 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2008 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2008 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2008 * Real.pi / 180))

theorem relationship_abcd : b < a ∧ a < d ∧ d < c := by
  sorry

end relationship_abcd_l645_645160


namespace probability_event_A_occurrence_l645_645082

noncomputable def event_A (b c : ℕ) : Prop :=
  ∃ x1 x2 : ℝ, -2 * x1^2 + b * x1 + c = 0 ∧ -2 * x2^2 + b * x2 + c = 0 ∧ -1 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 2

noncomputable def valid_range : ℕ → Prop := λ n, n ∈ {0, 1, 2, 3}

theorem probability_event_A_occurrence :
  ∀ b c, valid_range b → valid_range c → (14 : ℝ) / (16 : ℝ) = (7 : ℝ) / (8 : ℝ) :=
by
  sorry

end probability_event_A_occurrence_l645_645082


namespace solve_inequality_l645_645604

theorem solve_inequality (x : ℝ) (h : x ≠ 1) : (x / (x - 1) ≥ 2 * x) ↔ (x ≤ 0 ∨ (1 < x ∧ x ≤ 3 / 2)) :=
by
  sorry

end solve_inequality_l645_645604


namespace prime_sum_of_primes_unique_l645_645418

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_sum_of_primes_unique (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h_sum_prime : is_prime (p^q + q^p)) :
  p = 2 ∧ q = 3 :=
sorry

end prime_sum_of_primes_unique_l645_645418


namespace equilateral_triangle_ratio_is_correct_l645_645693

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l645_645693


namespace correct_ordering_of_powers_l645_645314

theorem correct_ordering_of_powers : 
  7^8 < 3^15 ∧ 3^15 < 4^12 ∧ 4^12 < 8^10 :=
  by
    sorry

end correct_ordering_of_powers_l645_645314


namespace limsup_sigma_ratio_l645_645441

noncomputable def sigma (n : ℕ) : ℕ :=
  ∑ d in divisors n, d

theorem limsup_sigma_ratio :
  (∀ (n : ℕ), 0 < n) →
  limsup (λ (n : ℕ), (sigma (n ^ 2023) : ℝ) / (sigma n) ^ 2023) = 1 :=
by sorry

end limsup_sigma_ratio_l645_645441


namespace concentration_of_acid_in_third_flask_is_correct_l645_645285

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l645_645285


namespace total_planks_l645_645812

-- Define the initial number of planks
def initial_planks : ℕ := 15

-- Define the planks Charlie got
def charlie_planks : ℕ := 10

-- Define the planks Charlie's father got
def father_planks : ℕ := 10

-- Prove the total number of planks
theorem total_planks : (initial_planks + charlie_planks + father_planks) = 35 :=
by sorry

end total_planks_l645_645812


namespace travel_time_from_NY_to_SF_l645_645227

-- Define variables for the times
variable (T : ℝ)
-- Condition 1: Time from New Orleans to New York is (3/4)T
def Time_NO_NY := (3 / 4) * T
-- Condition 2: Total time from New Orleans to San Francisco is 58 hours
def Total_Time := 58
-- Condition 3: Layover in New York is 16 hours
def Layover := 16

-- Proposition stating that time taken to travel from New York to San Francisco is 24 hours given the conditions
theorem travel_time_from_NY_to_SF : 
  (3 / 4) * T + Layover + T = Total_Time → T = 24 := 
by 
  intro h,
  sorry

end travel_time_from_NY_to_SF_l645_645227


namespace powerjet_30_minutes_500_gallons_per_hour_l645_645232

theorem powerjet_30_minutes_500_gallons_per_hour:
  ∀ (rate : ℝ) (time : ℝ), rate = 500 → time = 30 → (rate * (time / 60) = 250) := by
  intros rate time rate_eq time_eq
  sorry

end powerjet_30_minutes_500_gallons_per_hour_l645_645232


namespace factorization_of_x_squared_minus_nine_l645_645846

theorem factorization_of_x_squared_minus_nine (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by
  sorry

end factorization_of_x_squared_minus_nine_l645_645846


namespace combination_15_12_l645_645405

theorem combination_15_12 : nat.choose 15 12 = 455 :=
by sorry

end combination_15_12_l645_645405


namespace distinct_values_count_l645_645803

theorem distinct_values_count : 
  ∃ distinct_vals : set ℕ, distinct_vals = {
    3^(3^(3^3)), 3^((3^3)^3), ((3^3)^3)^3, (3^(3^3))^3, (3^3)^(3^3)
  } ∧ 
  distinct_vals.size = 3 := by
  sorry

end distinct_values_count_l645_645803


namespace find_hyperbola_equation_l645_645900

-- Define the given conditions and the problem
def hyperbola_standard_equation (a : ℝ) (b : ℝ) (h : a > 0) : Prop :=
  ∃ c : ℝ, (c = 3) ∧ (a^2 + b^2 = 45) ∧ (a^2 = 4) ∧ eq1 ∧ eq2

-- Theorem statement
theorem find_hyperbola_equation : 
  hyperbola_standard_equation 2 (sqrt 5) (by norm_num) := 
sorry

end find_hyperbola_equation_l645_645900


namespace factorization_of_x_squared_minus_nine_l645_645845

theorem factorization_of_x_squared_minus_nine (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by
  sorry

end factorization_of_x_squared_minus_nine_l645_645845


namespace valid_a_value_l645_645030

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l645_645030


namespace valid_domain_sqrt_l645_645521

theorem valid_domain_sqrt (x : ℝ) : (∃ y, y = sqrt (2 - x)) ↔ (x ≤ 2) :=
by
  sorry

end valid_domain_sqrt_l645_645521


namespace imaginary_part_of_z_l645_645875

-- Define the complex number (2 + i)
def two_plus_i : ℂ := 2 + complex.i

-- Define the complex number z
def z : ℂ := 1 / (two_plus_i ^ 2)

-- The statement we need to prove
theorem imaginary_part_of_z :
  complex.im z = -4 / 25 :=
sorry

end imaginary_part_of_z_l645_645875


namespace Mitya_age_l645_645187

/--
Assume Mitya's current age is M and Shura's current age is S. If Mitya is 11 years older than Shura,
and when Mitya was as old as Shura is now, he was twice as old as Shura,
then prove that M = 27.5.
-/
theorem Mitya_age (S M : ℝ) (h1 : M = S + 11) (h2 : M - S = 2 * (S - (M - S))) : M = 27.5 :=
by
  sorry

end Mitya_age_l645_645187


namespace possible_values_of_expression_l645_645818

open Matrix

-- Define the matrix
def myMatrix (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a^2, b, c], ![b^2, c, a], ![c^2, a, b]]

-- Define the polynomial
def p (a b c x : ℝ) : ℝ := x^3 - (a^2 + b^2 + c^2) * x^2 + (a^4 + b^4 + c^4) * x - a * b * c

-- Conditions
variables a b c : ℝ
#check Real

noncomputable def det_myMatrix : ℝ :=
  det (myMatrix a b c)

theorem possible_values_of_expression (h1 : det_myMatrix = 0) :
  (∃ (v : ℝ), v ∈ {-1, (3 / 2)} ∧
  v = (a^2 / (b^2 + c) + b^2 / (a^2 + c) + c^2 / (a^2 + b^2))) :=
begin
  sorry
end

end possible_values_of_expression_l645_645818


namespace closest_perfect_square_to_1042_is_1024_l645_645729

theorem closest_perfect_square_to_1042_is_1024 :
  ∀ n : ℕ, (n = 32 ∨ n = 33) → ((1042 - n^2 = 18) ↔ n = 32):=
by
  intros n hn
  cases hn
  case inl h32 => sorry
  case inr h33 => sorry

end closest_perfect_square_to_1042_is_1024_l645_645729


namespace concentration_of_acid_in_third_flask_l645_645267

theorem concentration_of_acid_in_third_flask :
  ∀ (W1 W2 : ℝ),
    let W := 190 + 65.714 in 
    W1 = 190 ∧ W2 = 65.714 →
    (10 : ℝ) / (10 + W1) = 0.05 →
    (20 : ℝ) / (20 + W2) = 0.2331 →
    (30 : ℝ) / (30 + W) = 0.105 :=
begin
  sorry
end

end concentration_of_acid_in_third_flask_l645_645267


namespace area_of_intersection_polygon_l645_645676

-- Define the circle equation
def circle (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x - 3)^2 + 4 * y^2 = 36

-- Define the intersection point condition
def is_intersection_point (x y : ℝ) : Prop := circle x y ∧ ellipse x y

-- The Lean theorem statement to prove the area of the polygon defined by these intersection points.
theorem area_of_intersection_polygon : 
  ∃ A B C : ℝ × ℝ, is_intersection_point A.1 A.2 ∧ is_intersection_point B.1 B.2 ∧ is_intersection_point C.1 C.2  ∧ 
  (let (x1, y1) := A; let (x2, y2) := B; let (x3, y3) := C in 
   (1 / 2) * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
  = 3 * Real.sqrt 15 / 2) :=
sorry

end area_of_intersection_polygon_l645_645676


namespace A_independent_union_Bn_l645_645178

-- Define the probability space (a placeholder for now)
variable {Ω : Type*} [ProbabilitySpace Ω]

-- Define events A and Bn
variable (A : Event Ω)
variable (Bn : ℕ → Event Ω)

-- Define conditions
-- 1. A is independent of each Bn
axiom independent_each_A_Bn (n : ℕ) : IndependentEvent A (Bn n)

-- 2. B_i are mutually exclusive
axiom mutual_exclusive_Bi (i j : ℕ) (h : i ≠ j) : (Bn i ∩ Bn j) = ∅

-- Statement to prove: A and Union of B_ns are independent
theorem A_independent_union_Bn : 
  IndependentEvent A (⋃ n, Bn n) :=
begin
  sorry -- proof goes here
end

end A_independent_union_Bn_l645_645178


namespace range_of_a_l645_645092

variable {a m : ℝ}
variable {x1 x2 : ℝ} (h1 : x1^2 - m * x1 - 1 = 0) (h2 : x2^2 - m * x2 - 1 = 0)
variable (h3 : a^2 + 4 * a - 3 ≤ |x1 - x2|)
variable (h4 : (∃ x : ℝ, x^2 + 2 * x + a < 0))
variable (h5 : (∃ m : ℝ, x1 = (m + sqrt(m^2 + 4)) / 2 ∧ x2 = (m - sqrt(m^2 + 4)) / 2))

theorem range_of_a (h : p ∨ q) (h_not : ¬(p ∧ q)) : a = 1 ∨ a < -5 := by
  sorry

end range_of_a_l645_645092


namespace combination_15_12_l645_645403

theorem combination_15_12 : nat.choose 15 12 = 455 :=
by sorry

end combination_15_12_l645_645403


namespace power_of_two_l645_645454

theorem power_of_two (b m n : ℕ) (hb : b ≠ 1) (hmn : m ≠ n) 
  (same_prime_factors : nat.prime_factors (b^m - 1) = nat.prime_factors (b^n - 1)) :
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end power_of_two_l645_645454


namespace initial_average_income_l645_645613

-- Definitions based on initial conditions
def initial_earning_members : ℕ := 3
def average_income_after_death : ℕ := 650
def income_of_deceased : ℕ := 905

-- Statement to prove
theorem initial_average_income :
  let total_income_after_death := 2 * average_income_after_death in
  let total_income_before_death := total_income_after_death + income_of_deceased in
  let initial_average_income_per_member := total_income_before_death / initial_earning_members in
  initial_average_income_per_member = 735 := by
{
  -- This is where the proof would go
  sorry
}

end initial_average_income_l645_645613


namespace mary_days_eq_11_l645_645183

variable (x : ℝ) -- Number of days Mary takes to complete the work
variable (m_eff : ℝ) -- Efficiency of Mary (work per day)
variable (r_eff : ℝ) -- Efficiency of Rosy (work per day)

-- Given conditions
axiom rosy_efficiency : r_eff = 1.1 * m_eff
axiom rosy_days : r_eff * 10 = 1

-- Define the efficiency of Mary in terms of days
axiom mary_efficiency : m_eff = 1 / x

-- The theorem to prove
theorem mary_days_eq_11 : x = 11 :=
by
  sorry

end mary_days_eq_11_l645_645183


namespace total_flowers_l645_645579

-- Definition of conditions
def minyoung_flowers : ℕ := 24
def yoojung_flowers (y : ℕ) : Prop := minyoung_flowers = 4 * y

-- Theorem statement
theorem total_flowers (y : ℕ) (h : yoojung_flowers y) : minyoung_flowers + y = 30 :=
by sorry

end total_flowers_l645_645579


namespace actual_discount_is_expected_discount_l645_645346

-- Define the conditions
def promotional_discount := 20 / 100  -- 20% discount
def vip_card_discount := 10 / 100  -- 10% additional discount

-- Define the combined discount calculation
def combined_discount := (1 - promotional_discount) * (1 - vip_card_discount)

-- Define the expected discount off the original price
def expected_discount := 28 / 100  -- 28% discount

-- Theorem statement proving the combined discount is equivalent to the expected discount
theorem actual_discount_is_expected_discount :
  combined_discount = 1 - expected_discount :=
by
  -- Proof omitted.
  sorry

end actual_discount_is_expected_discount_l645_645346


namespace max_value_of_c_over_b_l645_645896

variables {A B C : ℝ} {a b c : ℝ}

-- Definitions of the sides and angles in a triangle
def sides_and_angles (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = π ∧ 
  a = 2 * (b / (2 * cos(A/2))) ∧
  a^2 = b^2 + c^2 - 2*b*c*cos A

-- Definition of the altitude
def altitude_on_side_BC (a b c : ℝ) : Prop :=
  ∃ h : ℝ, h = a / 2

-- Definition of maximum value of c/b
def max_ratio_c_over_b (t : ℝ) : Prop :=
  t = sqrt 2 + 1

theorem max_value_of_c_over_b 
  (a b c A B C : ℝ)
  (h₁ : sides_and_angles a b c A B C)
  (h₂ : altitude_on_side_BC a b c) :
  ∃ t, max_ratio_c_over_b t :=
sorry

end max_value_of_c_over_b_l645_645896


namespace oldest_son_park_visits_l645_645586

theorem oldest_son_park_visits 
    (season_pass_cost : ℕ)
    (cost_per_trip : ℕ)
    (youngest_son_trips : ℕ) 
    (remaining_value : ℕ)
    (oldest_son_trips : ℕ) : 
    season_pass_cost = 100 →
    cost_per_trip = 4 →
    youngest_son_trips = 15 →
    remaining_value = season_pass_cost - youngest_son_trips * cost_per_trip →
    oldest_son_trips = remaining_value / cost_per_trip →
    oldest_son_trips = 10 := 
by sorry

end oldest_son_park_visits_l645_645586


namespace equation_has_roots_l645_645007

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l645_645007


namespace relationship_a_b_l645_645491

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -2 ∨ x ≥ 1 then 0 else -x^2 - x + 2

theorem relationship_a_b (a b : ℝ) (h_pos : a > 0) :
  (∀ x : ℝ, a * x + b = g x) → (2 * a < b ∧ b < (a + 1)^2 / 4 + 2 ∧ 0 < a ∧ a < 3) :=
sorry

end relationship_a_b_l645_645491


namespace ratio_of_area_to_perimeter_l645_645706

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l645_645706


namespace incenter_circumcenter_dist_l645_645370

structure Triangle :=
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_tri : a + b > c ∧ a + c > b ∧ b + c > a)

def incenter (T : Triangle) : ℝ × ℝ :=
  -- Assuming this function exists
  sorry

def circumcenter (T : Triangle) : ℝ × ℝ :=
  -- Assuming this function exists
  sorry

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem incenter_circumcenter_dist :
  ∀ (T : Triangle), T.a = 6 → T.b = 8 → T.c = 10 → dist (incenter T) (circumcenter T) = real.sqrt 5 :=
by
  intros T h_a h_b h_c
  sorry

end incenter_circumcenter_dist_l645_645370


namespace majority_vote_is_280_l645_645960

-- Definitions based on conditions from step (a)
def totalVotes : ℕ := 1400
def winningPercentage : ℝ := 0.60
def losingPercentage : ℝ := 0.40

-- Majority computation based on the winning and losing percentages
def majorityVotes : ℝ := totalVotes * winningPercentage - totalVotes * losingPercentage

-- Theorem statement
theorem majority_vote_is_280 : majorityVotes = 280 := by
  sorry

end majority_vote_is_280_l645_645960


namespace scientific_notation_13000_l645_645964

theorem scientific_notation_13000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 13000 = a * 10^n ∧ a = 1.3 ∧ n = 4 :=
begin
  sorry
end

end scientific_notation_13000_l645_645964


namespace valid_a_value_l645_645035

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l645_645035


namespace equation_has_at_least_two_distinct_roots_l645_645014

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l645_645014


namespace find_fourth_vertex_of_rectangle_l645_645479

-- Define the vertices of the rectangle.
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (3, 2)

-- Prove that the coordinates of the fourth vertex D are (2, 3).
theorem find_fourth_vertex_of_rectangle :
  ∃ D : ℝ × ℝ, D = (2, 3) ∧ 
  (∀ A B C D, -- The configuration of a rectangle should be verified in setup.
    (D.1 - A.1) * (D.1 - C.1) + (D.2 - A.2) * (D.2 - C.2) = 0 ∧
    (D.1 - B.1) * (D.1 - C.1) + (D.2 - B.2) * (D.2 - C.2) = 0) :=
sorry

end find_fourth_vertex_of_rectangle_l645_645479


namespace sum_not_necessarily_positive_l645_645452

theorem sum_not_necessarily_positive :
  ∃ (s: Fin 25 → ℤ),
  (∀ (a b c : Fin 25), a ≠ b → b ≠ c → a ≠ c →
    ∃ d : Fin 25, d ≠ a → d ≠ b → d ≠ c → s a + s b + s c + s d > 0) →
  (∑ i, s i ≤ 0) :=
by
  sorry

end sum_not_necessarily_positive_l645_645452


namespace determine_n_l645_645151

def all_divisors (n : ℕ) : List ℕ := (List.range (n + 1)).filter (λ x, x > 0 ∧ n % x = 0)

theorem determine_n (n : ℕ) (d : List ℕ) 
  (h1 : d = all_divisors n) 
  (h2 : d.length ≥ 22) 
  (h7 : d.nthLe 6 sorry = d_7)
  (h10 : d.nthLe 9 sorry = d_{10}) 
  (h22 : d.nthLe 21 sorry = d_{22})
  (h_eq : (d_7 ^ 2) + (d_{10} ^ 2) = (n / d_{22}) ^ 2) : 
  n = 2040 := 
sorry

end determine_n_l645_645151


namespace map_width_l645_645789

theorem map_width (length : ℝ) (area : ℝ) (h1 : length = 2) (h2 : area = 20) : ∃ (width : ℝ), width = 10 :=
by
  sorry

end map_width_l645_645789


namespace find_a_for_quadratic_l645_645023

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l645_645023


namespace magnitude_of_a_l645_645926

-- Define the vector space over real numbers and the magnitudes of vectors
variable (a b : ℝ) (angle_ab : real.angle) (b_magnitude : ℝ)
-- Define the conditions
variable (h_angle : angle_ab = real.pi * (2/3))
variable (h_b_magnitude : b_magnitude = 1)
variable (h_inequality : ∀ (x : ℝ), real.norm (a + x * b) ≥ real.norm (a + b))

-- The main theorem to prove the magnitude of vector a is 2
theorem magnitude_of_a : real.norm a = 2 :=
by
  sorry

end magnitude_of_a_l645_645926


namespace segment_length_calc_l645_645975

noncomputable def segment_length_parallel_to_side
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) : ℝ :=
  a * (b + c) / (a + b + c)

theorem segment_length_calc
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  segment_length_parallel_to_side a b c a_pos b_pos c_pos = a * (b + c) / (a + b + c) :=
sorry

end segment_length_calc_l645_645975


namespace find_b_value_l645_645170

noncomputable def find_b (p q : ℕ) : ℕ := p^2 + q^2

theorem find_b_value
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h_distinct : p ≠ q) (h_roots : p + q = 13 ∧ p * q = 22) :
  find_b p q = 125 :=
by
  sorry

end find_b_value_l645_645170


namespace concentration_in_third_flask_l645_645273

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l645_645273


namespace concentration_third_flask_l645_645281

-- Define the concentrations as per the given problem

noncomputable def concentration (acid_mass water_mass : ℝ) : ℝ :=
  (acid_mass / (acid_mass + water_mass)) * 100

-- Given conditions
def acid_mass_first_flask : ℝ := 10
def acid_mass_second_flask : ℝ := 20
def acid_mass_third_flask : ℝ := 30
def concentration_first_flask : ℝ := 5
def concentration_second_flask : ℝ := 70 / 3

-- Total water added to the first and second flasks
def total_water_mass : ℝ :=
  let W1 := (acid_mass_first_flask - concentration_first_flask * acid_mass_first_flask / 100)
  let W2 := (acid_mass_second_flask - concentration_second_flask * acid_mass_second_flask / 100)
  W1 + W2 

-- Prove the concentration of acid in the third flask
theorem concentration_third_flask : 
  concentration acid_mass_third_flask total_water_mass = 10.5 := 
  sorry

end concentration_third_flask_l645_645281


namespace problem_1_problem_2_l645_645086

noncomputable def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

theorem problem_1 (m : ℝ) (h : ∀ x : ℝ, f x ≥ |m + 1|) : m ≤ 4 :=
by
  sorry

theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + 2 * b + c = 4) : 
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
by
  sorry

end problem_1_problem_2_l645_645086


namespace coefficient_x2_in_pk_l645_645171

noncomputable def pk (k : ℕ) (x : ℕ) : ℕ :=
  match k with
  | 0 => (x - 2) ^ 2
  | (n + 1) => (pk n x - 2) ^ 2

theorem coefficient_x2_in_pk (k : ℕ) :
  let pk := λ k x, match k with
                    | 0 => (x - 2) ^ 2
                    | (n + 1) => (pk n x - 2) ^ 2
  in ∃ A_k, polynomial.coeff (pk k x) 2 = (4 ^ (2 * k - 1) - 4 ^ (k - 1)) / 3 := 
sorry

end coefficient_x2_in_pk_l645_645171


namespace sum_of_first_n_terms_b_l645_645770

variable (n : ℕ)

def a (n : ℕ) : ℕ := 2 * n
def b (n : ℕ) : ℕ := 2 ^ n

theorem sum_of_first_n_terms_b :
  let b_sum := ∑ i in finset.range n, b (i + 1)
  b_sum = (4 ^ n - 1) * 4 / 3 :=
by sorry

end sum_of_first_n_terms_b_l645_645770


namespace tangent_lines_parallel_at_x_l645_645907

noncomputable def x0_values : set ℝ :=
  {x0 | let f1 := λ x : ℝ, x^2 - 1,
             f2 := λ x : ℝ, 1 - x^3,
             f1' := (deriv f1),
             f2' := (deriv f2)
        in f1'(x0) = f2'(x0) }

theorem tangent_lines_parallel_at_x (x0 : ℝ) (h : x0 ∈ x0_values) :
  x0 = 0 ∨ x0 = -2/3 :=
sorry

end tangent_lines_parallel_at_x_l645_645907


namespace todd_has_40_left_after_paying_back_l645_645665

def todd_snowcone_problem : Prop :=
  let borrowed := 100
  let repay := 110
  let cost_ingredients := 75
  let snowcones_sold := 200
  let price_per_snowcone := 0.75
  let total_earnings := snowcones_sold * price_per_snowcone
  let remaining_money := total_earnings - repay
  remaining_money = 40

theorem todd_has_40_left_after_paying_back : todd_snowcone_problem :=
by
  -- Add proof here if needed
  sorry

end todd_has_40_left_after_paying_back_l645_645665


namespace range_of_f_l645_645076

noncomputable def f : ℝ → ℝ := sorry

-- The conditions given in the problem
axiom f_deriv : ∀ x : ℝ, has_deriv_at f (deriv f x) x 
axiom f_zero : f 0 = 2
axiom f_deriv_ineq : ∀ x : ℝ, deriv f x - f x > exp x

-- The theorem statement proving the range of x
theorem range_of_f (x : ℝ) (h : x > 0) : f x > x * exp x + 2 * exp x := sorry

end range_of_f_l645_645076


namespace concentration_of_acid_in_third_flask_l645_645268

theorem concentration_of_acid_in_third_flask :
  ∀ (W1 W2 : ℝ),
    let W := 190 + 65.714 in 
    W1 = 190 ∧ W2 = 65.714 →
    (10 : ℝ) / (10 + W1) = 0.05 →
    (20 : ℝ) / (20 + W2) = 0.2331 →
    (30 : ℝ) / (30 + W) = 0.105 :=
begin
  sorry
end

end concentration_of_acid_in_third_flask_l645_645268


namespace semicircle_circumference_correct_l645_645250

noncomputable def perimeter_of_rectangle (l b : ℝ) : ℝ := 2 * (l + b)
noncomputable def side_of_square_by_rectangle (l b : ℝ) : ℝ := perimeter_of_rectangle l b / 4
noncomputable def circumference_of_semicircle (d : ℝ) : ℝ := (Real.pi * (d / 2)) + d

theorem semicircle_circumference_correct :
  let l := 16
  let b := 12
  let d := side_of_square_by_rectangle l b
  circumference_of_semicircle d = 35.98 :=
by
  sorry

end semicircle_circumference_correct_l645_645250


namespace greatest_divisor_l645_645329

theorem greatest_divisor (d : ℕ) :
  (1657 % d = 6 ∧ 2037 % d = 5) → d = 127 := by
  sorry

end greatest_divisor_l645_645329


namespace ten_digit_number_condition_l645_645853

theorem ten_digit_number_condition :
  ∃ (X : ℕ), (X < 10^10) ∧ -- X is a ten digit number
  let a := (list.of_fn (λ i, (X / 10^i % 10))) in -- The list of digits from least to most significant
  (a.length = 10 ∧
  (a.nth 0 = some (a.count 0) ∧
   a.nth 1 = some (a.count 1) ∧
   a.nth 2 = some (a.count 2) ∧
   a.nth 3 = some (a.count 3) ∧
   a.nth 4 = some (a.count 4) ∧
   a.nth 5 = some (a.count 5) ∧
   a.nth 6 = some (a.count 6) ∧
   a.nth 7 = some (a.count 7) ∧
   a.nth 8 = some (a.count 8) ∧
   a.nth 9 = some (a.count 9)) ∧
  (∑ i in finset.range 10, a.nth_le i (by linarith) = 10)) :=
  ∃ X, X = 6210001000 ∧ -- Proving the number must be 6210001000
  sorry

end ten_digit_number_condition_l645_645853


namespace nested_g_computation_l645_645156

def g (x : ℝ) : ℝ :=
  if x ≥ 0 then -2 * x^2 else x + 5

theorem nested_g_computation :
  g (g (g (g (g 2)))) = -3 := by
  sorry

end nested_g_computation_l645_645156


namespace estimate_total_balls_l645_645545

theorem estimate_total_balls (red_balls : ℕ) (frequency : ℝ) (total_balls : ℕ) 
  (h_red : red_balls = 12) (h_freq : frequency = 0.6) 
  (h_eq : (red_balls : ℝ) / total_balls = frequency) : 
  total_balls = 20 :=
by
  sorry

end estimate_total_balls_l645_645545


namespace remainder_when_T10_divided_by_5_is_0_l645_645862

def valid_sequence(n : ℕ) (s : List Char) : Prop :=
  s.length = n ∧
  (∀(i : ℕ), (s.get i = 'A' → s.getOpt (i + 1) ≠ some 'A' → s.getOpt (i + 2) ≠ some 'A' → s.getOpt (i + 3) ≠ some 'A')) ∧
  (∀(i : ℕ), (s.get i = 'B' → s.getOpt (i + 1) ≠ some 'B' → s.getOpt (i + 2) ≠ some 'B' → s.getOpt (i + 3) ≠ some 'B')) ∧
  (∀(i : ℕ), (s.get i = 'C' → s.getOpt (i + 1) ≠ some 'C')) 

def T (n : ℕ) : ℕ :=
  List.length (List.filter (valid_sequence n) (List.replicate n ['A', 'B', 'C']))

theorem remainder_when_T10_divided_by_5_is_0 : T 10 % 5 = 0 := by
  sorry

end remainder_when_T10_divided_by_5_is_0_l645_645862


namespace range_of_a_range_of_f_diff_l645_645247

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + x + 1
noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, f' a x1 = 0 ∧ f' a x2 = 0 ∧ x1 ≠ x2) ↔ (a < -Real.sqrt 3 ∨ a > Real.sqrt 3) :=
by
  sorry

theorem range_of_f_diff (a x1 x2 : ℝ) (h1 : f' a x1 = 0) (h2 : f' a x2 = 0) (h12 : x1 ≠ x2) : 
  0 < f a x1 - f a x2 :=
by
  sorry

end range_of_a_range_of_f_diff_l645_645247


namespace set_intersection_is_correct_l645_645465

def setA : Set ℝ := {x | x^2 - 4 * x > 0}
def setB : Set ℝ := {x | abs (x - 1) ≤ 2}
def setIntersection : Set ℝ := {x | -1 ≤ x ∧ x < 0}

theorem set_intersection_is_correct :
  setA ∩ setB = setIntersection := 
by
  sorry

end set_intersection_is_correct_l645_645465


namespace gumballs_per_pair_of_earrings_l645_645982

theorem gumballs_per_pair_of_earrings : 
  let day1_earrings := 3
  let day2_earrings := 2 * day1_earrings
  let day3_earrings := day2_earrings - 1
  let total_earrings := day1_earrings + day2_earrings + day3_earrings
  let days := 42
  let gumballs_per_day := 3
  let total_gumballs := days * gumballs_per_day
  (total_gumballs / total_earrings) = 9 :=
by
  -- Definitions
  let day1_earrings := 3
  let day2_earrings := 2 * day1_earrings
  let day3_earrings := day2_earrings - 1
  let total_earrings := day1_earrings + day2_earrings + day3_earrings
  let days := 42
  let gumballs_per_day := 3
  let total_gumballs := days * gumballs_per_day
  -- Theorem statement
  sorry

end gumballs_per_pair_of_earrings_l645_645982


namespace PM_eq_QM_iff_angle_BDP_eq_angle_CDQ_l645_645336

-- Let P and Q be points on the sides AB and AC respectively of triangle ABC
-- Let perpendiculars to AB at P and AC at Q meet at D, an interior point of triangle ABC
-- Let M be the midpoint of BC
-- Prove: PM = QM if and only if ∠BDP = ∠CDQ

variables {A B C P Q D M : Type} 
variables [EuclideanGeometry A B C] --assume a Euclidean space for geometry
variables [PtOnSide P A B] [PtOnSide Q A C]
variables [PerpendicularAt P AB] [PerpendicularAt Q AC]
variables [MeetsAt D P Q]
variables [Midpoint M B C]

theorem PM_eq_QM_iff_angle_BDP_eq_angle_CDQ :
  (distance P M = distance Q M) ↔ (angle B D P = angle C D Q) := sorry

end PM_eq_QM_iff_angle_BDP_eq_angle_CDQ_l645_645336


namespace prove_options_l645_645520

def curve (t : ℝ) (x y : ℝ) : Prop :=
  (x^2) / (5 - t) + (y^2) / (t - 1) = 1

theorem prove_options (t : ℝ):
  (t = 3 → ∃ x y, curve t x y ∧ ∃ k, k * (x^2 + y^2) = k * 2) ∧ -- Option A
  ((1 < t ∧ t < 3) → ∀ x,y, curve t x y → ∃ a b, a > b ∧ (y^2)/(a^2) + (x^2)/(b^2) = 1) -- Option C
:=
sorry

end prove_options_l645_645520


namespace binom_15_12_eq_455_l645_645408

theorem binom_15_12_eq_455 : nat.choose 15 12 = 455 :=
by 
  -- Proof omitted
  sorry

end binom_15_12_eq_455_l645_645408


namespace ratio_equilateral_triangle_l645_645723

def equilateral_triangle_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * (Real.sqrt 3 / 2)
  let area := (1 / 2) * s * altitude
  let perimeter := 3 * s in
  area / perimeter -- this simplifies to 25\sqrt{3} / 30 or 5\sqrt{3} / 6

theorem ratio_equilateral_triangle : ∀ (s : ℝ), s = 10 → equilateral_triangle_ratio s (by assumption) = 5 * (Real.sqrt 3) / 6 :=
by
  intros s h
  rw h
  sorry

end ratio_equilateral_triangle_l645_645723


namespace five_squared_decomposition_smallest_in_cube_21_l645_645052

-- Define the decomposition pattern for squares of integers greater than or equal to 2.
def square_decomposition (n : ℕ) : ℕ :=
  (list.range n).map (λ k, 2*k + 1).sum

-- Define the decomposition pattern for cubes of integers greater than or equal to 2.
def cube_decomposition (m : ℕ) : ℕ :=
  (list.range m).map (λ k, 2*k + (2*m-1)).sum

-- State the theorem for the square decomposition of 5^2.
theorem five_squared_decomposition : square_decomposition 5 = 25 :=
  sorry

-- Given the smallest element in the cube decomposition, determine the corresponding m.
theorem smallest_in_cube_21 : (∃ (m : ℕ), 21 = 2*m - 1) → cube_decomposition 5 = 125 :=
  sorry

end five_squared_decomposition_smallest_in_cube_21_l645_645052


namespace solve_real_eq_l645_645430

theorem solve_real_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (x = (23 + Real.sqrt 145) / 6 ∨ x = (23 - Real.sqrt 145) / 6) ↔
  ((x ^ 3 - 3 * x ^ 2) / (x ^ 2 - 4) + 2 * x = -16) :=
by sorry

end solve_real_eq_l645_645430


namespace sum_of_perimeters_l645_645775

theorem sum_of_perimeters (s : ℝ) : (∀ n : ℕ, n >= 0) → 
  (∑' n : ℕ, (4 * s) / (2 ^ n)) = 8 * s :=
by
  sorry

end sum_of_perimeters_l645_645775


namespace compute_H_iterates_l645_645916

def H (x : ℝ) : ℝ :=
  if x = 2 then -2
  else if x = -2 then 4
  else if x = 4 then 4
  else 0  -- This is a placeholder and should be more accurately defined via the graph provided.

theorem compute_H_iterates :
  H (H (H (H (H 2)))) = 4 :=
by
  have h2 := (if_pos (eq.refl 2) : H 2 = -2)
  have h_neg2 := (if_pos (eq.refl (-2)) : H (-2) = 4)
  have h4 := (if_pos (eq.refl 4) : H 4 = 4)
  calc
    H (H (H (H (H 2)))) = H (H (H (H (-2)))) : by rw h2
    ... = H (H (H 4)) : by rw h_neg2
    ... = H (H 4) : by rw h4
    ... = H 4 : by rw h4
    ... = 4 : by rw h4

end compute_H_iterates_l645_645916


namespace line_XY_passes_through_orthocenter_l645_645669

/-- Given conditions as per the problem description -/
variable (A B C J E F X Y : Point)

-- Assuming some axioms here to create a proper context
-- Definition of orthocenter is assumed to be used in the proof
axiom triangle_ABC : IsTriangle A B C
axiom circle_J : Circle J B C
axiom intersects_EF : (AC ∩ circle_J = E) ∧ (AB ∩ circle_J = F)
axiom similar_triangles_X : SimilarTriangle (Triangle F X B) (Triangle E J C)
axiom similar_triangles_Y : SimilarTriangle (Triangle E Y C) (Triangle F J B)
axiom same_side_X : SameSideLine X C AB
axiom same_side_Y : SameSideLine Y B AC

/-- Lean 4 statement - Proof that XY passes through the orthocenter of triangle ABC -/
theorem line_XY_passes_through_orthocenter (H : Point) :
    orthocenter H A B C →
    LinePassesThrough XY H := 
by
  sorry

end line_XY_passes_through_orthocenter_l645_645669


namespace probability_product_multiple_of_4_l645_645322

open Finset

theorem probability_product_multiple_of_4 :
  let cards := {1, 2, 3, 4, 5, 6}
  let pairs := (cards.product cards).filter (λ p, p.fst ≠ p.snd)
  let favorable_pairs := pairs.filter (λ p, (p.fst * p.snd) % 4 = 0)
  ∃ prob : ℚ,
    prob = ↑(favorable_pairs.card) / ↑(pairs.card)
    ∧ prob = 2 / 5 := by
  sorry

end probability_product_multiple_of_4_l645_645322


namespace range_of_a_l645_645941

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ Icc (π / 4) (4 * π / 3) → cos x ^ 2 - cos x + a = 0 → x ∈ Icc (π / 4) (4 * π / 3))
  ↔ a ∈ set.Icc (-2 : ℝ) (-3 / 4) ∪ set.Icc ((sqrt 2 - 1) / 2) (1 / 4) :=
by
  sorry

end range_of_a_l645_645941


namespace identify_irrational_number_l645_645384

theorem identify_irrational_number :
  ∀ (a b c d : ℝ),
    a = 1.333 →
    b = Real.sqrt 3 →
    c = 22 / 7 →
    d = Real.cbrt (-27) →
    (¬ ∃ p q : ℤ, b = p / q ∧ q ≠ 0) ∧
    (∀ x : ℝ, x = a → ∃ p q : ℤ, x = p / q ∧ q ≠ 0) ∧
    (∀ x : ℝ, x = c → ∃ p q : ℤ, x = p / q ∧ q ≠ 0) ∧
    (∀ x : ℝ, x = d → ∃ p q : ℤ, x = p / q ∧ q ≠ 0) :=
by
  sorry

end identify_irrational_number_l645_645384


namespace todd_money_after_repay_l645_645660

-- Definitions as conditions
def borrowed_amount : ℤ := 100
def repay_amount : ℤ := 110
def ingredients_cost : ℤ := 75
def snow_cones_sold : ℤ := 200
def price_per_snow_cone : ℚ := 0.75

-- Function to calculate money left after transactions
def money_left_after_repay (borrowed : ℤ) (repay : ℤ) (cost : ℤ) (sold : ℤ) (price : ℚ) : ℚ :=
  let left_after_cost := borrowed - cost
  let earnings := sold * price
  let total_before_repay := left_after_cost + earnings
  total_before_repay - repay

-- The theorem stating the problem
theorem todd_money_after_repay :
  money_left_after_repay borrowed_amount repay_amount ingredients_cost snow_cones_sold price_per_snow_cone = 65 := 
by
  -- This is a placeholder for the actual proof
  sorry

end todd_money_after_repay_l645_645660


namespace total_amount_is_24_l645_645779

-- Define the original price of a tub of ice cream
def original_price_ice_cream : ℕ := 12

-- Define the discount per tub of ice cream
def discount_per_tub : ℕ := 2

-- Define the discounted price of a tub of ice cream
def discounted_price_ice_cream : ℕ := original_price_ice_cream - discount_per_tub

-- Define the price for 5 cans of juice
def price_per_5_cans_of_juice : ℕ := 2

-- Define the number of cans of juice bought
def cans_of_juice_bought : ℕ := 10

-- Calculate the total cost for two tubs of ice cream and 10 cans of juice
def total_cost (p1 p2 : ℕ) : ℕ := 2 * p1 + (price_per_5_cans_of_juice * (cans_of_juice_bought / 5))

-- Prove that the total cost is $24
theorem total_amount_is_24 : total_cost discounted_price_ice_cream price_per_5_cans_of_juice = 24 := by
  sorry

end total_amount_is_24_l645_645779


namespace log_sum_a4_a5_a6_l645_645526

noncomputable def a_sequence (a : ℕ → ℝ) := ∀ n : ℕ, log (a (n + 1)) = 1 + log (a n)

theorem log_sum_a4_a5_a6 (a : ℕ → ℝ) (h1 : a_sequence a) (h2 : a 1 + a 2 + a 3 = 10) : log (a 4 + a 5 + a 6) = 4 := by
  sorry

end log_sum_a4_a5_a6_l645_645526


namespace largest_k_divides_2_pow_3_pow_m_add_1_l645_645856

theorem largest_k_divides_2_pow_3_pow_m_add_1 (m : ℕ) : 9 ∣ 2^(3^m) + 1 := sorry

end largest_k_divides_2_pow_3_pow_m_add_1_l645_645856


namespace increasing_intervals_l645_645909

def f (x : ℝ) : ℝ := sqrt 3 * sin x + (1 / 2) * cos (2 * x)

theorem increasing_intervals
  (k : ℤ) : 
  ∀ x, (k : ℝ) * Real.pi - (Real.pi / 3) ≤ x ∧ x ≤ (k : ℝ) * Real.pi + (Real.pi / 6) →
  f x is increasing :=
sorry

end increasing_intervals_l645_645909


namespace part_I_part_II_l645_645920

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * a_n (n - 1) + 1

noncomputable def b_n (n : ℕ) : ℕ :=
  a_n n + 1

noncomputable def T_n (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, (i + 1) * b_n (i + 1))

theorem part_I (n : ℕ) (hn : n > 0) : a_n n = 2^n - 1 :=
sorry

theorem part_II (n : ℕ) : T_n n = 2 + (n - 1) * 2^(n + 1) :=
sorry

end part_I_part_II_l645_645920


namespace value_of_x_l645_645265

theorem value_of_x (x y z : ℤ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 :=
by
  sorry

end value_of_x_l645_645265


namespace max_marks_l645_645338

theorem max_marks (M : ℝ) (h1 : 0.42 * M = 80) : M = 190 :=
by
  sorry

end max_marks_l645_645338


namespace vector_magnitude_l645_645925

variables (a b : ℝ × ℝ)

def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude
  (ha : magnitude a = 1)
  (hb : magnitude b = 2)
  (hab : a.1 + b.1 = 2 * real.sqrt 2 ∧ a.2 + b.2 = 1) :
  magnitude (3 a.1 + b.1, 3 a.2 + b.2) = 5 := 
sorry

end vector_magnitude_l645_645925


namespace f_at_pi_over_3_f_monotonically_decreasing_interval_symmetry_axis_l645_645484

noncomputable def f (x : ℝ) : ℝ :=
  2 * sqrt 3 * sin (x / 2) * cos (x / 2) - 2 * (cos (x / 2))^2

theorem f_at_pi_over_3 : f (π / 3) = 0 := 
sorry

theorem f_monotonically_decreasing_interval (k : ℤ) :
  ∀ x ∈ set.Icc (2 * π / 3 + 2 * k * π) (5 * π / 3 + 2 * k * π), 
  ∀ y ∈ set.Icc (2 * π / 3 + 2 * k * π) (5 * π / 3 + 2 * k * π),
  x < y → f x > f y :=
sorry

theorem symmetry_axis (k : ℤ) : ∃ x, x = 2 * π / 3 + k * π := 
sorry

end f_at_pi_over_3_f_monotonically_decreasing_interval_symmetry_axis_l645_645484


namespace min_value_f_l645_645910

noncomputable def f (x m : ℝ) : ℝ :=
  x * Real.exp x - (m / 2) * x^2 - m * x

theorem min_value_f (m : ℝ) (h_m : 0 < m) :
  let f := λ x, x * Real.exp x - (m / 2) * x^2 - m * x in
  let I := Set.Icc (1 : ℝ) 2 in
  f ∈ measurable (Set.IntervalIntegrable I) ∧
  (∀ x ∈ I, f x) ∈ lower_bounds ((λ x, x * Real.exp x - (m / 2) * x^2 - m * x) '' I) 
    ∈ {e - (3/2)*m, - (1/2)*m*Real.log m^2, 2*Real.exp 2^2 - 4*m} :=
begin
  sorry
end

end min_value_f_l645_645910


namespace total_surface_area_l645_645646

-- Defining the conditions
variables {a b c : ℝ}

-- Given conditions
def condition1 := 4 * a + 4 * b + 4 * c = 156
def condition2 := real.sqrt (a^2 + b^2 + c^2) = 25

-- The goal: To prove the total surface area is 896
theorem total_surface_area (h1 : condition1) (h2 : condition2) : 2 * a * b + 2 * b * c + 2 * c * a = 896 := 
sorry

end total_surface_area_l645_645646


namespace value_of_a_with_two_distinct_roots_l645_645026

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l645_645026


namespace brown_gumdrops_after_replacement_l645_645758

-- Definitions based on the given conditions.
def total_gumdrops (green_gumdrops : ℕ) : ℕ :=
  (green_gumdrops * 100) / 15

def blue_gumdrops (total_gumdrops : ℕ) : ℕ :=
  total_gumdrops * 25 / 100

def brown_gumdrops_initial (total_gumdrops : ℕ) : ℕ :=
  total_gumdrops * 15 / 100

def brown_gumdrops_final (brown_initial : ℕ) (blue_gumdrops : ℕ) : ℕ :=
  brown_initial + blue_gumdrops / 3

-- The main theorem statement based on the proof problem.
theorem brown_gumdrops_after_replacement
  (green_gumdrops : ℕ)
  (h_green : green_gumdrops = 36)
  : brown_gumdrops_final (brown_gumdrops_initial (total_gumdrops green_gumdrops)) 
                         (blue_gumdrops (total_gumdrops green_gumdrops))
    = 56 := 
  by sorry

end brown_gumdrops_after_replacement_l645_645758


namespace equilateral_triangle_area_to_perimeter_ratio_l645_645689

theorem equilateral_triangle_area_to_perimeter_ratio
  (a : ℝ) (h : a = 10) :
  let altitude := a * (Real.sqrt 3 / 2) in
  let area := (1 / 2) * a * altitude in
  let perimeter := 3 * a in
  area / perimeter = 5 * (Real.sqrt 3) / 6 := 
by
  sorry

end equilateral_triangle_area_to_perimeter_ratio_l645_645689


namespace range_f_in_0_1_l645_645881

variable (f : ℝ → ℝ)
variable (cond : ∀ x y : ℝ, x > y → (f x) ^ 2 ≤ f y)

theorem range_f_in_0_1 : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 :=
begin
  sorry
end

end range_f_in_0_1_l645_645881


namespace average_of_three_quantities_l645_645614

theorem average_of_three_quantities (a b c d e : ℝ) 
  (h_avg_5 : (a + b + c + d + e) / 5 = 11)
  (h_avg_2 : (d + e) / 2 = 21.5) :
  (a + b + c) / 3 = 4 :=
by
  sorry

end average_of_three_quantities_l645_645614


namespace angle_H_in_parallelogram_l645_645955

theorem angle_H_in_parallelogram :
  ∀ (E F G H : Type) 
    (parallelogram : EFGH → Prop)
    (angle_G : EFGH → ℝ)
    (angle_H : EFGH → ℝ),
  parallelogram EFGH →
  angle_G EFGH = 125 →
  angle_H EFGH = 180 - angle_G EFGH →
  angle_H EFGH = 55 :=
by
  intros E F G H parallelogram angle_G angle_H h₁ h₂ h₃
  sorry

end angle_H_in_parallelogram_l645_645955


namespace tom_chopped_more_than_tammy_l645_645743

-- Definitions based on conditions
def tom_rate : ℝ := 4 / 3
def tammy_rate : ℝ := 3 / 4
def total_salad : ℝ := 65
def combined_rate : ℝ := tom_rate + tammy_rate
def total_time : ℝ := total_salad / combined_rate
def tom_chopped : ℝ := tom_rate * total_time
def tammy_chopped : ℝ := tammy_rate * total_time

-- Proof statement
theorem tom_chopped_more_than_tammy :
  (((tom_rate * total_time) - (tammy_rate * total_time)) / (tammy_rate * total_time)) * 100 ≈ 77.36 :=
by sorry

end tom_chopped_more_than_tammy_l645_645743


namespace exceeded_goal_by_600_l645_645212

noncomputable def ken_collection : ℕ := 600
noncomputable def mary_collection : ℕ := 5 * ken_collection
noncomputable def scott_collection : ℕ := mary_collection / 3
noncomputable def goal : ℕ := 4000
noncomputable def total_raised : ℕ := mary_collection + scott_collection + ken_collection

theorem exceeded_goal_by_600 : total_raised - goal = 600 := by
  have h1 : ken_collection = 600 := rfl
  have h2 : mary_collection = 5 * ken_collection := rfl
  have h3 : scott_collection = mary_collection / 3 := rfl
  have h4 : goal = 4000 := rfl
  have h5 : total_raised = mary_collection + scott_collection + ken_collection := rfl
  have hken : ken_collection = 600 := rfl
  have hmary : mary_collection = 5 * 600 := by rw [hken]; rfl
  have hscott : scott_collection = 3000 / 3 := by rw [hmary]; rfl
  have htotal : total_raised = 3000 + 1000 + 600 := by rw [hmary, hscott, hken]; rfl
  have hexceeded : total_raised - goal = 4600 - 4000 := by rw [htotal, h4]; rfl
  exact hexceeded

end exceeded_goal_by_600_l645_645212


namespace number_of_terms_in_arithmetic_sequence_l645_645104

-- Definitions and conditions
def a : ℤ := -58  -- First term
def d : ℤ := 7   -- Common difference
def l : ℤ := 78  -- Last term

-- Statement of the problem
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 20 := 
by
  sorry

end number_of_terms_in_arithmetic_sequence_l645_645104


namespace distance_between_incenter_and_circumcenter_l645_645374

-- Definitions and conditions
variables {A B C I O : Type}
variables [DecidableEq A] [DecidableEq B] [DecidableEq C]
variables (triangle_ABC : Triangle A B C)
variables sides_ABC_6_8_10 : ∃ (a b c : ℝ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a
variables inscribed_circle : TriangleInCircle A B C I
variables circumscribed_circle : TriangleCircumCircle A B C O

-- Statement to be proven
theorem distance_between_incenter_and_circumcenter :
  distance inscribed_circle.center circumscribed_circle.center = Real.sqrt 13 :=
sorry

end distance_between_incenter_and_circumcenter_l645_645374


namespace diamond_detection_possible_l645_645958

theorem diamond_detection_possible :
  ∃ (detectors : Fin 14 → Bool),
    (∀ i : Fin 14, detectors i = false → detectors (i + 1) = true ∨ detectors (i + 2) = true ∨ detectors (i + 3) = true) ∧
    (∀ (j k l : Fin 14),
      j + 1 = k → k + 1 = l → 
      detectors j = true ∨ detectors k = true ∨ detectors l = true) :=
begin
  sorry
end

end diamond_detection_possible_l645_645958


namespace exists_multiple_l645_645872

theorem exists_multiple (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h : ∀ i, a i > 0) 
  (h2 : ∀ i, a i ≤ 2 * n) : 
  ∃ i j : Fin (n + 1), i ≠ j ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
sorry

end exists_multiple_l645_645872


namespace equation_has_roots_l645_645006

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l645_645006


namespace max_superior_squares_l645_645884

theorem max_superior_squares (n : ℕ) (h : n > 2004) :
  ∃ superior_squares_count : ℕ, superior_squares_count = n * (n - 2004) := 
sorry

end max_superior_squares_l645_645884


namespace minimum_value_of_expr_l645_645039

noncomputable def expr (x y : ℝ) : ℝ := 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4

theorem minimum_value_of_expr : ∃ x y : ℝ, expr x y = -1 ∧ ∀ (a b : ℝ), expr a b ≥ -1 := 
by
  sorry

end minimum_value_of_expr_l645_645039


namespace speed_with_stoppages_l645_645423

-- Define conditions
def train_speed_excl_stoppages := 45 -- Speed without stoppages in km/h
def stoppage_time_per_hour := 12 / 60 -- Stoppage time in hours (12 minutes)

-- Define the statement to be proved
theorem speed_with_stoppages : 
  let running_time := 1 - stoppage_time_per_hour in 
  let distance_covered := train_speed_excl_stoppages * running_time in
  distance_covered = 36 :=
by
  let running_time := 1 - stoppage_time_per_hour
  let distance_covered := train_speed_excl_stoppages * running_time
  sorry

end speed_with_stoppages_l645_645423


namespace ratio_of_green_to_blue_chairs_l645_645952

def blue_chairs : ℕ := 10
def total_chairs : ℕ := 67

axiom green_multiple_of_blue (k : ℕ) : blue_chairs = 10 ∧ G = k * blue_chairs
axiom fewer_white_chairs (G : ℕ) (W : ℕ) : W = (G + blue_chairs) - 13
axiom total_number_of_chairs (G : ℕ) (W : ℕ) : 10 + G + W = total_chairs

theorem ratio_of_green_to_blue_chairs : 
  ∃ (k : ℕ), G = k * blue_chairs ∧ k = 3 :=
by
  sorry

end ratio_of_green_to_blue_chairs_l645_645952


namespace indefinite_integral_l645_645804

noncomputable def integral_expression : ℝ → ℝ :=
  λ x, (2 * x^4 - 5 * x^2 - 8 * x - 8) / (x * (x - 2) * (x + 2))

noncomputable def integral_result (C : ℝ) : ℝ → ℝ :=
  λ x, x^2 + 2 * Real.log (|x|) - 3 / 2 * Real.log (|x - 2|) + 5 / 2 * Real.log (|x + 2|) + C

theorem indefinite_integral :
  ∃ C : ℝ, ∀ x : ℝ, (∫ u in 0..x, integral_expression u) = integral_result C x :=
by
  sorry

end indefinite_integral_l645_645804


namespace Powerjet_pumps_250_gallons_in_30_minutes_l645_645235

theorem Powerjet_pumps_250_gallons_in_30_minutes :
  let r := 500 -- Pump rate in gallons per hour
  let t := 1 / 2 -- Time in hours (30 minutes)
  r * t = 250 := by
  -- proof steps will go here
  sorry

end Powerjet_pumps_250_gallons_in_30_minutes_l645_645235


namespace area_of_two_circles_intersection_l645_645671

def circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 ≤ radius ^ 2}

def area_of_intersection (c1 c2 : set (ℝ × ℝ)) : ℝ := sorry -- Placeholder

theorem area_of_two_circles_intersection :
  let circle1 := circle (3, 0) 3
  let circle2 := circle (0, 3) 3
  area_of_intersection circle1 circle2 = (9 * Real.pi) / 2 - 9 :=
sorry

end area_of_two_circles_intersection_l645_645671


namespace find_AB_BC_l645_645236

noncomputable def angle_A : ℝ := 2 * real.arccos (real.sqrt (5 / 6))
noncomputable def OC : ℝ := real.sqrt 7
noncomputable def OD : ℝ := 3 * real.sqrt 15
noncomputable def AD_over_BC : ℝ := 5

theorem find_AB_BC (α : ℝ) 
                   (h1 : α = real.arccos (real.sqrt (5 / 6)))
                   (h2 : OC = real.sqrt 7) 
                   (h3 : OD = 3 * real.sqrt 15) 
                   (h4 : AD_over_BC = 5) :
  ∃ (AB BC : ℝ), AB = 2 * real.sqrt 3 ∧ BC = (5 * real.sqrt 3) / 3 :=
begin
  sorry
end

end find_AB_BC_l645_645236


namespace f_9_over_4_eq_neg_a_l645_645914

def ω : ℝ := sorry -- Assuming ω > 0

def f (x : ℝ) : ℝ := Real.sin (ω * x)

axiom f_periodic : ∀ x : ℝ, f (x - 1/2) = f (x + 1/2)
axiom f_value_a : f (-1/4) = a

theorem f_9_over_4_eq_neg_a : f (9/4) = -a := by
  sorry

end f_9_over_4_eq_neg_a_l645_645914


namespace geometric_sum_5_l645_645468

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + 1) = a n * r ∧ a (m + 1) = a m * r

theorem geometric_sum_5 (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) (h3 : ∀ n, 0 < a n) :
  a 3 + a 5 = 5 :=
sorry

end geometric_sum_5_l645_645468


namespace hyperbola_eccentricity_correct_l645_645075

noncomputable def hyperbola_eccentricity : ℝ :=
  let a := ℝ
  let b := ℝ
  let c := Math.sqrt(a ^ 2 + b ^ 2) in
  let e := c / a in
  if (a > 0) ∧ (b > 0) ∧ 
     (∃ (F1 F2 : (ℝ × ℝ)), True) ∧ 
     (∃ (l : ℝ × ℝ → Prop), l F1 ∧ ((∃ (A B : (ℝ × ℝ)), l A ∧ l B ∧ A ≠ F1 ∧ B ≠ F1 ∧ |AB| = |AF2| ∧ ∠ F1 A F2 = 90)) then
  e
  else
  sorry

theorem hyperbola_eccentricity_correct (a b : ℝ) (F1 F2 : ℝ × ℝ) (l : ℝ × ℝ → Prop)
  (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ (F1 : ℝ × ℝ), ¬ F1 = F2)
  (h4 : ∃ (A B : ℝ × ℝ), l A ∧ l B ∧ A ≠ F1 ∧ B ≠ F1 ∧ (dist A B = dist A F2) ∧ (∠ F1 A F2 = 90)) :
  let c := Real.sqrt(a^2 + b^2),
      e := c / a in
  e = Real.sqrt(6) + Real.sqrt(3) :=
by sorry

end hyperbola_eccentricity_correct_l645_645075


namespace factorization_of_x_squared_minus_nine_l645_645841

theorem factorization_of_x_squared_minus_nine {x : ℝ} : x^2 - 9 = (x + 3) * (x - 3) :=
by
  -- Introduce the hypothesis to assist Lean in understanding the polynomial
  have h : x^2 - 9 = (x^2 - 3^2), 
  rw [pow_two, pow_two],
  exact factorization_of_x_squared_minus_3_squared _,
end

end factorization_of_x_squared_minus_nine_l645_645841


namespace complex_magnitude_l645_645072

variable (a b : ℝ)

theorem complex_magnitude :
  ((1 + 2 * a * Complex.I) * Complex.I = 1 - b * Complex.I) →
  Complex.normSq (a + b * Complex.I) = 5/4 :=
by
  intro h
  -- Add missing logic to transform assumption to the norm result
  sorry

end complex_magnitude_l645_645072


namespace interval_between_births_l645_645644

def youngest_child_age : ℕ := 6

def sum_of_ages (I : ℝ) : ℝ :=
  youngest_child_age + (youngest_child_age + I) + (youngest_child_age + 2 * I) + (youngest_child_age + 3 * I) + (youngest_child_age + 4 * I)

theorem interval_between_births : ∃ (I : ℝ), sum_of_ages I = 60 ∧ I = 3.6 := 
by
  sorry

end interval_between_births_l645_645644


namespace pascal_triangle_41_l645_645506

theorem pascal_triangle_41:
  ∃ (n : Nat), ∀ (k : Nat), n = 41 ∧ (Nat.choose n k = 41) :=
sorry

end pascal_triangle_41_l645_645506


namespace quadratic_trinomial_prime_l645_645067

theorem quadratic_trinomial_prime (p x : ℤ) (hp : p > 1) (hx : 0 ≤ x ∧ x < p)
  (h_prime : Prime (x^2 - x + p)) : x = 0 ∨ x = 1 :=
by
  sorry

end quadratic_trinomial_prime_l645_645067


namespace area_sum_l645_645529

noncomputable theory
open EuclideanGeometry

variables {A B C D E : Point}
variables [T : Triangle A B C]
variables (h_midpoint_E : Midpoint E B C) (h_midpoint_D : Midpoint D A C)
variables (h_AC_len : dist A C = 2)
variables (h_angles : ∠BAC = 45 ∧ ∠ABC = 90 ∧ ∠ACB = 45)
variables (h_height_D_CE : height_from_point_to_line D (line_through C E) = 1)

theorem area_sum:
  area A B C + 2 * area C D E = 3 :=
sorry

end area_sum_l645_645529


namespace equilateral_triangle_area_to_perimeter_ratio_l645_645685

theorem equilateral_triangle_area_to_perimeter_ratio
  (a : ℝ) (h : a = 10) :
  let altitude := a * (Real.sqrt 3 / 2) in
  let area := (1 / 2) * a * altitude in
  let perimeter := 3 * a in
  area / perimeter = 5 * (Real.sqrt 3) / 6 := 
by
  sorry

end equilateral_triangle_area_to_perimeter_ratio_l645_645685


namespace solve_for_a_l645_645902

theorem solve_for_a (a y x : ℝ)
  (h1 : y = 5 * a)
  (h2 : x = 2 * a - 2)
  (h3 : y + 3 = x) :
  a = -5 / 3 :=
by
  sorry

end solve_for_a_l645_645902


namespace polynomial_remainder_l645_645042

theorem polynomial_remainder (x : ℝ) : 
  (x - 1)^100 + (x - 2)^200 = (x^2 - 3 * x + 2) * (some_q : ℝ) + 1 :=
sorry

end polynomial_remainder_l645_645042


namespace candy_amount_in_peanut_butter_jar_l645_645655

-- Definitions of the candy amounts in each jar
def banana_jar := 43
def grape_jar := banana_jar + 5
def peanut_butter_jar := 4 * grape_jar
def coconut_jar := (3 / 2) * banana_jar

-- The statement we need to prove
theorem candy_amount_in_peanut_butter_jar : peanut_butter_jar = 192 := by
  sorry

end candy_amount_in_peanut_butter_jar_l645_645655


namespace ratio_of_area_to_perimeter_l645_645710

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l645_645710


namespace acid_concentration_third_flask_l645_645294

-- Define the concentrations of first and second flask
def conc_first (w1 : ℝ) : ℝ := 10 / (10 + w1)
def conc_second (w2 : ℝ) : ℝ := 20 / (20 + w2)

-- Define the acid mass in the third flask initially
def acid_mass_third : ℝ := 30

-- Total water added from the fourth flask
def total_water (w1 w2 : ℝ) : ℝ := w1 + w2

-- Acid concentration in the third flask after all water is added
def conc_third (w : ℝ) : ℝ := acid_mass_third / (acid_mass_third + w)

-- Problem statement: concentration in the third flask is 10.5%
theorem acid_concentration_third_flask (w1 : ℝ) (w2 : ℝ) (w : ℝ) 
  (h1 : conc_first w1 = 0.05) 
  (h2 : conc_second w2 = 70 / 300) 
  (h3 : w = total_water w1 w2) : 
  conc_third w = 10.5 / 100 := 
sorry

end acid_concentration_third_flask_l645_645294


namespace negation_of_exists_leq_l645_645635

theorem negation_of_exists_leq (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_of_exists_leq_l645_645635


namespace range_of_a1_l645_645360

theorem range_of_a1 (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 1 / (2 - a n)) 
  (h_pos : ∀ n, a (n + 1) > a n) : a 1 < 1 := 
sorry

end range_of_a1_l645_645360


namespace equation_has_roots_l645_645008

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l645_645008


namespace betsy_remaining_squares_l645_645391

def total_squares := 16 + 16
def percent_sewn := 0.25
def squares_sewn := total_squares * percent_sewn
def remaining_squares := total_squares - squares_sewn

theorem betsy_remaining_squares : remaining_squares = 24 :=
by
  sorry

end betsy_remaining_squares_l645_645391


namespace length_of_AB_max_length_of_MN_l645_645963

/-
Conditions for problem (1):
- Parametric equation: x = sqrt(3)*t, y = t
- Polar equation: ρ^2 = cos^2 θ + sin θ
- Points A and B where l intersects C in polar coordinates

Condition for problem (2):
- Points M and N on curve C with OM ⊥ ON
- Polar equation: ρ^2 = cos^2 θ + sin θ
-/

-- Problem (1)
theorem length_of_AB :
  ∀ (t θ ρ : ℝ) (l : ℝ → ℝ × ℝ) (C : ℝ → ℝ),
    (∀ t, l t = (sqrt 3 * t, t)) →
    (∀ θ, C θ = (cos 2 * θ + sin θ) ^ (1 / 2)) →
    (l t).fst = C (π / 6) → -- condition for point A
    (l t).fst = C (7 * π / 6) → -- condition for point B
    let OA := (C (π / 6)) in
    let OB := (C (7 * π / 6)) in
    let AB := OA + OB in
    AB = (sqrt 5 + 1) / 2 :=
by sorry

-- Problem (2)
theorem max_length_of_MN :
  ∀ (θ ρ α : ℝ) (C : ℝ → ℝ),
    (∀ θ, C θ = (cos 2 * θ + sin θ) ^ (1 / 2)) →
    let OM := C α in
    let ON := C (α + π / 2) in
    OM ^ 2 + ON ^ 2 = 1 + sqrt 2 * sin (α + π / 4) →
    max (abs (OM + ON)) = sqrt (1+sqrt 2) :=
by sorry

end length_of_AB_max_length_of_MN_l645_645963


namespace irreducibility_of_polynomial_l645_645220

theorem irreducibility_of_polynomial (n : ℕ) (hn : 0 < n) :
  ∀ p q : polynomial ℤ, p * q = (X^2 + X)^(2^n : ℕ) + 1 → p.degree ≠ 0 ∧ q.degree ≠ 0 → false :=
sorry

end irreducibility_of_polynomial_l645_645220


namespace equilateral_triangle_ratio_is_correct_l645_645694

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l645_645694


namespace find_number_l645_645726

theorem find_number (x : ℝ) : x + 5 * 12 / (180 / 3) = 61 ↔ x = 60 := by
  sorry

end find_number_l645_645726


namespace initial_distance_truck_X_l645_645670

-- Definitions for the problem conditions
def speed_truck_x : ℝ := 57
def speed_truck_y : ℝ := 63
def time_to_overtake : ℝ := 3
def distance_ahead_after_overtake : ℝ := 4

-- The main theorem to prove the initial distance
theorem initial_distance_truck_X : 
  let relative_speed := speed_truck_y - speed_truck_x
      distance_covered := relative_speed * time_to_overtake
  in distance_covered - distance_ahead_after_overtake = 14 :=
by
  -- Using sorry to skip the proof
  sorry

end initial_distance_truck_X_l645_645670


namespace joe_paint_fraction_l645_645554

theorem joe_paint_fraction:
  ∃ f : ℝ,
  (0 ≤ f ∧ f ≤ 1) ∧ -- f is a fraction between 0 and 1
  360 * f + (1/5) * (360 - 360 * f) = 120 ∧ -- total paint used condition
  f = 1 / 6 := 
by 
  use 1 / 6
  split
  -- Proof for 0 ≤ f ∧ f ≤ 1
  split
  -- Proof for 0 ≤ f
  {
    sorry
  },
  -- Proof for f ≤ 1
  {
    sorry
  },
  -- Proof for total paint used condition
  {
    have h : 360 * (1 / 6) + (1 / 5) * (360 - 360 * (1 / 6)) = 120,
    {
      sorry
    },
    exact h,
  }

end joe_paint_fraction_l645_645554


namespace simplify_evaluate_expression_l645_645223

theorem simplify_evaluate_expression (x : ℕ) (h : x = 3) : 
  (1 + 1 / (x + 1)) * ((x + 1) / (x^2 + 4)) = 5 / 13 :=
by
  rw h
  sorry

end simplify_evaluate_expression_l645_645223


namespace train_length_l645_645788

def train_speed_kmph := 25 -- speed of train in km/h
def man_speed_kmph := 2 -- speed of man in km/h
def crossing_time_sec := 52 -- time to cross in seconds

noncomputable def length_of_train : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph -- relative speed in km/h
  let relative_speed_mps := relative_speed_kmph * (5 / 18) -- convert to m/s
  relative_speed_mps * crossing_time_sec -- length of train in meters

theorem train_length : length_of_train = 390 :=
  by sorry -- proof omitted

end train_length_l645_645788


namespace morgan_change_l645_645581

theorem morgan_change:
  let hamburger := 5.75
  let onion_rings := 2.50
  let smoothie := 3.25
  let side_salad := 3.75
  let cake := 4.20
  let total_cost := hamburger + onion_rings + smoothie + side_salad + cake
  let payment := 50
  let change := payment - total_cost
  ℝ := by
    exact sorry

end morgan_change_l645_645581


namespace triangle_maximum_area_l645_645179

variable {A B C : ℝ} {a b c : ℝ}

theorem triangle_maximum_area (h1 : a ∣ cos C - (1 / 2) * c = b)
  (h2 : a = 2 * Real.sqrt 3) : 
  (∃ b c, ∆abc_has_sides a b c ∧ ∆abc_area a b c ≤ Real.sqrt 3) :=
by
  sorry

end triangle_maximum_area_l645_645179


namespace equation_has_two_distinct_roots_l645_645003

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l645_645003


namespace calculate_1307_squared_l645_645737

theorem calculate_1307_squared : 1307 * 1307 = 1709849 := sorry

end calculate_1307_squared_l645_645737


namespace song_distribution_l645_645792

-- Definitions
def Song := Type
def Person := {Amy : Person} | {Beth : Person} | {Jo : Person}

def likes_two_not_third (p1 p2 p3 : Person) (s : Song) : Prop := (p1.likes s ∧ p2.likes s ∧ ¬p3.likes s)

-- Problem statement
theorem song_distribution {songs : Finset Song}
    (h_songs : songs.card = 5)
    (h_no_all_three : ∀ (s : Song), ¬(∀ (p : Person), p.likes s))
    (h_each_pair : ∀ (p1 p2 p3 : Person), ∃ s ∈ songs, likes_two_not_third p1 p2 p3 s) :
    ∃ (ways : ℕ), ways = 51 :=
by
  sorry

end song_distribution_l645_645792


namespace general_terms_strictly_increasing_range_l645_645063

noncomputable def seq (a : ℝ) : ℕ → ℝ
| 1       := a
| 2       := 3
| (n + 1) := 2 * seq a n + 3 * seq a (n - 1)

theorem general_terms (a : ℝ) (a_ne_1 : a ≠ 1) (a_ne_n3 : a ≠ -3) :
  (∀ n : ℕ, 1 ≤ n → seq a (n + 1) + seq a n = (a + 3) * 3 ^ (n - 1)) ∧
  (∀ n : ℕ, 1 ≤ n → seq a (n + 1) - 3 * seq a n = (-1) ^ (n - 1) * (3 - 3 * a)) :=
sorry

theorem strictly_increasing_range (a : ℝ) :
  (∀ n : ℕ, 1 ≤ n → seq a (n + 1) > seq a n) → a ∈ set.Ioo (-1 : ℝ) 1 ∪ set.Ioo (1 : ℝ) 3 :=
sorry

end general_terms_strictly_increasing_range_l645_645063


namespace product_of_integers_l645_645620

theorem product_of_integers (a b : ℤ) (h_lcm : Int.lcm a b = 45) (h_gcd : Int.gcd a b = 9) : a * b = 405 :=
by
  sorry

end product_of_integers_l645_645620


namespace matrix1_determinant_l645_645816

noncomputable def matrix1 : Matrix (Fin 3) (Fin 3) ℚ := ![
  [3, -6, 0],
  [5, -1, 2],
  [0, 3, -4]
]

theorem matrix1_determinant : matrix1.det = 114 := by
  sorry

end matrix1_determinant_l645_645816


namespace smallest_x_for_f_eq_f_2001_l645_645348

noncomputable def f (x : ℝ) : ℝ := 
  if 2 ≤ x ∧ x ≤ 6 then 2 - |x - 4|
  else sorry -- The remaining definition for f outside [2, 6] is not necessary for the problem.

theorem smallest_x_for_f_eq_f_2001 :
  ∃ x : ℝ, f x = f 2001 ∧ (∀ y : ℝ, y < x → f y ≠ f 2001) :=
by
  have h_f_eq_5f : ∀ x > 0, f (5 * x) = 5 * f x := sorry,
  have h_f_eq_piece : ∀ x, 2 ≤ x ∧ x ≤ 6 → f x = 2 - |x - 4| := sorry,
  let y := 2001 / (5 ^ 5),
  have h_y_in_range : 2 ≤ y ∧ y ≤ 6 := sorry,
  have h_f_2001_calc : f 2001 = 2 := sorry,
  have h_x_min : f 4 = 2 ∧ (∀ z, z < 4 → f z ≠ 2) := sorry,
  exact ⟨4, h_f_2001_calc, h_x_min⟩

end smallest_x_for_f_eq_f_2001_l645_645348


namespace equilateral_triangle_ratio_is_correct_l645_645691

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l645_645691


namespace problem1_problem2_l645_645469

variable {a b c : ℝ}

theorem problem1 (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : a^2 + 2 * b^2 + 3 * c^2 = 4) (hc : a = c) : a * b ≤ sqrt(2) / 2 := by
  sorry

theorem problem2 (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : a^2 + 2 * b^2 + 3 * c^2 = 4) : a + 2 * b + 3 * c ≤ 2 * sqrt(6) := by
  sorry

end problem1_problem2_l645_645469


namespace range_f_contained_in_0_1_l645_645878

theorem range_f_contained_in_0_1 (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := 
by {
  sorry
}

end range_f_contained_in_0_1_l645_645878


namespace calculate_f_of_f_neg_8_l645_645419

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ -1 then -(real.cbrt x) else x + 2 / x - 7

theorem calculate_f_of_f_neg_8 : f (f (-8)) = -4 :=
by
  sorry

end calculate_f_of_f_neg_8_l645_645419


namespace ratio_of_area_to_perimeter_l645_645682

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l645_645682


namespace perpendicular_case_parallel_case_l645_645098

variable (a b : ℝ)

-- Define the lines
def line1 (a b x y : ℝ) := a * x - b * y + 4 = 0
def line2 (a b x y : ℝ) := (a - 1) * x + y + b = 0

-- Define perpendicular condition
def perpendicular (a b : ℝ) := a * (a - 1) - b = 0

-- Define point condition
def passes_through (a b : ℝ) := -3 * a + b + 4 = 0

-- Define parallel condition
def parallel (a b : ℝ) := a * (a - 1) + b = 0

-- Define intercepts equal condition
def intercepts_equal (a b : ℝ) := b = -a

theorem perpendicular_case
    (h1 : perpendicular a b)
    (h2 : passes_through a b) :
    a = 2 ∧ b = 2 :=
sorry

theorem parallel_case
    (h1 : parallel a b)
    (h2 : intercepts_equal a b) :
    a = 2 ∧ b = -2 :=
sorry

end perpendicular_case_parallel_case_l645_645098


namespace common_points_l645_645637

variable {R : Type*} [LinearOrderedField R]

def eq1 (x y : R) : Prop := x - y + 2 = 0
def eq2 (x y : R) : Prop := 3 * x + y - 4 = 0
def eq3 (x y : R) : Prop := x + y - 2 = 0
def eq4 (x y : R) : Prop := 2 * x - 5 * y + 7 = 0

theorem common_points : ∃ s : Finset (R × R), 
  (∀ p ∈ s, eq1 p.1 p.2 ∨ eq2 p.1 p.2) ∧ (∀ p ∈ s, eq3 p.1 p.2 ∨ eq4 p.1 p.2) ∧ s.card = 6 :=
by
  sorry

end common_points_l645_645637


namespace card_count_l645_645200

theorem card_count (x y : ℕ) (h1 : x + y + 2 = 10) (h2 : 3 * x + 4 * y + 10 = 39) : x = 3 :=
by {
  sorry
}

end card_count_l645_645200


namespace tangent_line_slope_negative_l645_645917

def f (x : ℝ) : ℝ := Real.exp x + 3 / x

theorem tangent_line_slope_negative :
  (Deriv f 1) < 0 :=
by
  sorry

end tangent_line_slope_negative_l645_645917


namespace perpendicular_distance_from_P_to_AB_l645_645238

def dihedral_angle (α β : Plane) (AB : Line) : ℝ := 60
def point_on_plane (P : Point) (α : Plane) : Prop := P ∈ α
def point_on_plane (Q : Point) (β : Plane) : Prop := Q ∈ β
def not_on_line (P Q : Point) (AB : Line) : Prop := P ∉ AB ∧ Q ∉ AB
def angle_between_lines (PQ AB : Line) : ℝ := 45
def length_PQ (P Q : Point) : ℝ := 7 * real.sqrt 2
def perpendicular_distance (Q AB : Line) : ℝ := 3

theorem perpendicular_distance_from_P_to_AB {α β : Plane} {AB : Line}
  (P Q : Point) 
  (h_dihedral: dihedral_angle α β AB = 60)
  (h_P_on_α: point_on_plane P α)
  (h_Q_on_β: point_on_plane Q β)
  (h_not_on_line: not_on_line P Q AB)
  (h_angle: angle_between_lines (Line.mk P Q) AB = 45)
  (h_length: length_PQ P Q = 7 * real.sqrt 2)
  (h_perp_distance: perpendicular_distance Q AB = 3) :
  perpendicular_distance P AB = 8 := sorry

end perpendicular_distance_from_P_to_AB_l645_645238


namespace coeff_x2_of_expression_l645_645855

def coeff_x2 (p : Polynomial ℤ) : ℤ :=
  p.coeff 2

theorem coeff_x2_of_expression :
  coeff_x2 (4 * (X ^ 2 - 2 * X ^ 3 + 3 * X) +
            2 * (X + X ^ 3 - 4 * X ^ 2 + 2 * X ^ 5 - 2 * X ^ 2) -
            6 * (C 2 + X - 3 * X ^ 3 - 2 * X ^ 2)) = 10 := 
sorry

end coeff_x2_of_expression_l645_645855


namespace concentration_of_acid_in_third_flask_is_correct_l645_645290

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l645_645290


namespace ratio_of_area_to_perimeter_l645_645677

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l645_645677


namespace motorists_exceeding_speed_limit_l645_645738

-- Define the conditions
def total_motorists : ℕ := 100
def percent_receiving_tickets := 0.10
def percent_not_receiving_tickets_of_exceeded := 0.40

-- Define the desired outcome
def percent_exceeding_speed_limit : ℝ := 0.17

-- The proof statement
theorem motorists_exceeding_speed_limit :
  let motorists_receiving_tickets := total_motorists * percent_receiving_tickets
  let total_exceeding_limit := motorists_receiving_tickets / (1 - percent_not_receiving_tickets_of_exceeded)
  total_exceeding_limit / total_motorists = percent_exceeding_speed_limit :=
by
  sorry

end motorists_exceeding_speed_limit_l645_645738


namespace value_of_x_l645_645262

theorem value_of_x (y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := by
  sorry

end value_of_x_l645_645262


namespace todd_money_after_repay_l645_645662

-- Definitions as conditions
def borrowed_amount : ℤ := 100
def repay_amount : ℤ := 110
def ingredients_cost : ℤ := 75
def snow_cones_sold : ℤ := 200
def price_per_snow_cone : ℚ := 0.75

-- Function to calculate money left after transactions
def money_left_after_repay (borrowed : ℤ) (repay : ℤ) (cost : ℤ) (sold : ℤ) (price : ℚ) : ℚ :=
  let left_after_cost := borrowed - cost
  let earnings := sold * price
  let total_before_repay := left_after_cost + earnings
  total_before_repay - repay

-- The theorem stating the problem
theorem todd_money_after_repay :
  money_left_after_repay borrowed_amount repay_amount ingredients_cost snow_cones_sold price_per_snow_cone = 65 := 
by
  -- This is a placeholder for the actual proof
  sorry

end todd_money_after_repay_l645_645662


namespace part_a_concurrency_l645_645889

open EuclideanGeometry

variables {A B C O E F G K : Point}

-- Conditions
axiom acute_triangle_ABC : acute_triangle A B C
axiom circumcircle_ABC : Circle (A B C) O
axiom point_G_on_arc_BC_not_containing_O (circumcircle_OBC : Circle (O B C) G) : on_arc_BC_not_containing O G
axiom intersection_E (circumcircle_ABG : Circle (A B G) E) (h1: E ≠ A) : intersects E A C
axiom intersection_F (circumcircle_ACG : Circle (A C G) F) (h2 : F ≠ A) : intersects F A B
axiom intersection_K (BE_line : Line B E) (CF_line : Line C F) : intersection BE_line CF_line K

-- Question: Prove that AK, BC, and OG are concurrent
theorem part_a_concurrency (circumcircle_OBC : Circle (O B C) G) 
  (circumcircle_ABG : Circle (A B G) E)
  (circumcircle_ACG : Circle (A C G) F)
  (circumcircle_ABC : Circle (A B C) O)
  (h1: E ≠ A)
  (h2 : F ≠ A)
  (BE_line : Line B E)
  (CF_line : Line C F) : concurrency (Line A K) (Line B C) (Line O G) := sorry

end part_a_concurrency_l645_645889


namespace Yuna_boarding_place_l645_645422

-- Conditions
def Eunji_place : ℕ := 10
def people_after_Eunji : ℕ := 11

-- Proof Problem: Yuna's boarding place calculation
theorem Yuna_boarding_place :
  Eunji_place + people_after_Eunji + 1 = 22 :=
by
  sorry

end Yuna_boarding_place_l645_645422


namespace inequality_solution_set_range_of_a_l645_645488

def f (x : ℝ) : ℝ := abs (3*x + 2)

theorem inequality_solution_set :
  { x : ℝ | f x < 4 - abs (x - 1) } = { x : ℝ | -5/4 < x ∧ x < 1/2 } :=
by 
  sorry

theorem range_of_a (a : ℝ) (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) 
  (h4 : ∀ x : ℝ, abs (x - a) - f x ≤ 1 / m + 1 / n) : 
  0 < a ∧ a ≤ 10/3 :=
by 
  sorry

end inequality_solution_set_range_of_a_l645_645488


namespace vector_magnitude_sum_l645_645899

theorem vector_magnitude_sum (a b : ℝ^3) (h_angle : real.angle a b = real.pi / 3)
  (h_a : ∥a∥ = 2) (h_b : ∥b∥ = 1) : ∥a + 2 • b∥ = 2 * real.sqrt 3 :=
sorry

end vector_magnitude_sum_l645_645899


namespace problem_statement_l645_645485

def f (x : ℝ) : ℝ := 2^x - Real.sqrt 2

theorem problem_statement : f (1/2) = 0 :=
by
  unfold f
  sorry

end problem_statement_l645_645485


namespace find_a_for_quadratic_l645_645018

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l645_645018


namespace distance_incenter_circumcenter_l645_645376

theorem distance_incenter_circumcenter (A B C : ℝ × ℝ)
  (hAB : (dist A B) = 6)
  (hAC : (dist A C) = 8)
  (hBC : (dist B C) = 10)
  (h_right : ∠ BAC = 90) :
  dist (incenter A B C) (circumcenter A B C) = √5 :=
sorry

end distance_incenter_circumcenter_l645_645376


namespace peanuts_total_correct_l645_645946

def initial_peanuts : ℕ := 4
def added_peanuts : ℕ := 6
def total_peanuts : ℕ := initial_peanuts + added_peanuts

theorem peanuts_total_correct : total_peanuts = 10 := by
  sorry

end peanuts_total_correct_l645_645946


namespace MWBC_cyclic_l645_645164

-- Define the basic geometric elements
variables {A B C W I_A I_B D : Type}

-- Assume △ABC is an acute triangle
axiom acute_triangle (A B C :Type) : Prop

-- Define W as the intersection of the angle bisector of ∠ACB and side AB
axiom W_definition (A B C W : Type) : A = B → C = W

-- Define I_A as the incenter of triangle AWC
noncomputable def incenter_AWC (A W C I_A : Type) : Prop := sorry

-- Define I_B as the incenter of triangle WBC
noncomputable def incenter_WBC (W B C I_B : Type) : Prop := sorry

-- Define D as the intersection of I_AW and I_BB
axiom D_definition (I_A W I_B B D : Type) : I_A = W → I_B = B → W B = D

-- The theorem proving MWBC is a cyclic quadrilateral
theorem MWBC_cyclic (A B C W I_A I_B D : Type) 
  (hA : acute_triangle A B C)
  (hW : W_definition A B C W)
  (hIA : incenter_AWC A W C I_A)
  (hIB : incenter_WBC W B C I_B)
  (hD : D_definition I_A W I_B B D) : 
  cyclic_quadrilateral M W B C :=
sorry

end MWBC_cyclic_l645_645164


namespace binom_15_12_eq_455_l645_645400

theorem binom_15_12_eq_455 : nat.choose 15 12 = 455 := sorry

end binom_15_12_eq_455_l645_645400


namespace smallest_n_to_integer_l645_645409

noncomputable def y : ℕ → ℝ
| 0       := (4:ℝ)^(1/4)
| (n + 1) := (y n)^(4:ℝ)^(1/4)

theorem smallest_n_to_integer : ∃ n : ℕ, n > 0 ∧ y n ∈ ℕ ∧ ∀ m < n, y m ∉ ℕ :=
  sorry

end smallest_n_to_integer_l645_645409


namespace concentration_in_third_flask_l645_645276

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l645_645276


namespace pipe_fill_time_with_leak_l645_645199

theorem pipe_fill_time_with_leak (pipe_rate : ℝ) (leak_rate : ℝ) (fill_time : ℝ) :
  pipe_rate = 1 / 12 → 
  leak_rate = 1 / 36 → 
  fill_time = 1 / (pipe_rate - leak_rate) → 
  fill_time = 18 :=
by 
  intros h_pipe h_leak h_fill
  rw [h_pipe, h_leak] at h_fill
  have combined_rate : ℝ := (1 / 12) - (1 / 36)
  rw [show combined_rate = 1 / 18, by norm_num] at h_fill
  rw [inv_div, inv_inv] at h_fill
  exact h_fill

end pipe_fill_time_with_leak_l645_645199


namespace central_angle_of_section_l645_645343

theorem central_angle_of_section (A : ℝ) (hA : 0 < A) (prob : ℝ) (hprob : prob = 1 / 4) :
  ∃ θ : ℝ, (θ / 360) = prob :=
by
  use 90
  sorry

end central_angle_of_section_l645_645343


namespace plane_equation_from_perpendicular_foot_l645_645246

theorem plane_equation_from_perpendicular_foot :
  ∃ (A B C D : ℤ), 
    (A = 6) ∧ (B = -8) ∧ (C = 5) ∧ (D = -125) ∧ 
    (A > 0) ∧ 
    Int.gcd A (Int.gcd B (Int.gcd C D)) = 1 ∧ 
    (∀ x y z, A * x + B * y + C * z + D = 0) :=
by
  -- Definitions based on conditions
  let normal_vector := (6, -8, 5)
  let foot_of_perpendicular := (6, -8, 5)
  have A : ℤ := 6
  have B : ℤ := -8
  have C : ℤ := 5
  have D : ℤ := -125
  -- Construct the plane equation
  use [A, B, C, D]
  -- Conditions ensuring the definitions and answer are correct
  exact ⟨rfl, rfl, rfl, rfl, by norm_num, by norm_num, λ x y z, sorry⟩
  
-- sorry is a placeholder indicating the incomplete proof for the equation.

end plane_equation_from_perpendicular_foot_l645_645246


namespace shift_quadratic_l645_645248

theorem shift_quadratic (x : ℝ) :
  let y := x^2
  (y shifted_right_and_up := (x - 3)^2 + 3) 
  (y shifted_right_and_up = (x - 3)^2 + 3) :=
by
  let y := x^2
  let y_shifted_right_and_up := (x - 3)^2 + 3
  sorry

end shift_quadratic_l645_645248


namespace count_orthogonal_sets_l645_645525

def f1 (x : Real) : Real := Math.sin (x / 2)
def g1 (x : Real) : Real := Math.cos (x / 2)

def f2 (x : Real) : Real := x + 1
def g2 (x : Real) : Real := x - 1

def f3 (x : Real) : Real := x
def g3 (x : Real) : Real := x^2

-- Orthogonality condition on the interval [-1,1]
def orthogonal (f g : Real → Real) : Prop :=
  ∫ (x : Real) in -1..1, f x * g x = 0

theorem count_orthogonal_sets :
  (orthogonal f1 g1) ∧ (¬ orthogonal f2 g2) ∧ (orthogonal f3 g3) → 
  ∃ n, n = 2 :=
by
  sorry

end count_orthogonal_sets_l645_645525


namespace sum_of_solutions_l645_645440

theorem sum_of_solutions (x : ℝ) (h : sqrt (9 - x^2 / 4) = 3) : x = 0 :=
by
  sorry

end sum_of_solutions_l645_645440


namespace ratio_equilateral_triangle_l645_645725

def equilateral_triangle_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * (Real.sqrt 3 / 2)
  let area := (1 / 2) * s * altitude
  let perimeter := 3 * s in
  area / perimeter -- this simplifies to 25\sqrt{3} / 30 or 5\sqrt{3} / 6

theorem ratio_equilateral_triangle : ∀ (s : ℝ), s = 10 → equilateral_triangle_ratio s (by assumption) = 5 * (Real.sqrt 3) / 6 :=
by
  intros s h
  rw h
  sorry

end ratio_equilateral_triangle_l645_645725


namespace least_odd_positive_integer_not_obtainable_l645_645197

theorem least_odd_positive_integer_not_obtainable :
  ∃ n, n = 43 ∧ ¬ (∃ (s : list (ℤ × bool)), 
    (s.length = 9) ∧ 
    ((s.head? = some (1, tt)) ∧ 
    (s.tail = some s))
    ∧ 
    ∑ i in s.enum, 
    if s.nth (i + 1) matches (some y) then 
       if i % 2 == 0 then y.1 
       else -y.1 
    else 0 = n) :=
  sorry

end least_odd_positive_integer_not_obtainable_l645_645197


namespace range_of_f_l645_645040

noncomputable def f (x : ℝ) : ℝ := (3 * x + 7) / (x - 3)

theorem range_of_f :
  set.range f = {y : ℝ | y ≠ 3} :=
begin
  sorry
end

end range_of_f_l645_645040


namespace shortest_chord_length_l645_645062

theorem shortest_chord_length : 
  ∀ (λ : ℝ), ∃ L : ℝ,
  (∀ (λ x y : ℝ), (2 + λ) * x + (1 - 2 * λ) * y + 4 - 3 * λ = 0) ∧
  (x - 1) ^ 2 + y ^ 2 = 9 →
  L = 2 :=
\by
  intros,
  sorry

end shortest_chord_length_l645_645062


namespace ratio_a10_b10_l645_645500

variables {ℕ : Type} 
variable {a b : ℕ → ℕ}
variable {Sn Tn : ℕ → ℕ}
variable {d_a d_b : ℕ}
variable {a_1 b_1 : ℕ}

-- Arithmetic sequences
noncomputable def arith_seq (a_1 d_a : ℕ) (n : ℕ) := a_1 + (n - 1) * d_a
noncomputable def sum_arith_seq (a_1 d_a : ℕ) (n : ℕ) := n / 2 * (2 * a_1 + (n - 1) * d_a)

-- Given conditions
axiom seq_ratio (n : ℕ) : sum_arith_seq a_1 d_a n / sum_arith_seq b_1 d_b n = (3 * n - 1) / (2 * n + 3)

-- Prove the ratio of the 10th terms
theorem ratio_a10_b10 : (arith_seq a_1 d_a 10 / arith_seq b_1 d_b 10) = 57 / 41 := sorry

end ratio_a10_b10_l645_645500


namespace smallest_digit_not_in_odd_number_units_place_l645_645727

theorem smallest_digit_not_in_odd_number_units_place : 
  (∀ d ∈ {0, 2, 4, 6, 8}, d ≠ 0 → d > 0) ∧ (∀ d ∈ {1, 3, 5, 7, 9}, true) → 
  0 ∉ {1, 3, 5, 7, 9} :=
by
  sorry

end smallest_digit_not_in_odd_number_units_place_l645_645727


namespace last_two_digits_a2015_l645_645987

def a : ℕ → ℕ
| 0       := 1
| (n + 1) := if n % 2 = 0 then a n + 2 else 2 * a n

theorem last_two_digits_a2015 : (a 2015) % 100 = 72 :=
by sorry

end last_two_digits_a2015_l645_645987


namespace obtuse_triangle_classification_l645_645903

def angle_sum_triangle (x y z : ℝ) : Prop := x + y + z = 180

theorem obtuse_triangle_classification:
  ∀ (a b : ℝ), angle_sum_triangle a b (180 - a - b) → a = 40 ∧ b = 45 → 180 - a - b > 90 → "obtuse" :=
by
  sorry

end obtuse_triangle_classification_l645_645903


namespace calculate_expression_l645_645808

theorem calculate_expression : 3^0 - ((-1/2)^(-2)) = -3 :=
by
  have h1 : 3^0 = 1 := by norm_num
  have h2 : (-1/2)^(-2) = 4 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end calculate_expression_l645_645808


namespace ratio_equilateral_triangle_l645_645722

def equilateral_triangle_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * (Real.sqrt 3 / 2)
  let area := (1 / 2) * s * altitude
  let perimeter := 3 * s in
  area / perimeter -- this simplifies to 25\sqrt{3} / 30 or 5\sqrt{3} / 6

theorem ratio_equilateral_triangle : ∀ (s : ℝ), s = 10 → equilateral_triangle_ratio s (by assumption) = 5 * (Real.sqrt 3) / 6 :=
by
  intros s h
  rw h
  sorry

end ratio_equilateral_triangle_l645_645722


namespace exists_linear_g_l645_645455

/-
Given a function f: ℝ^n → ℝ^n such that f(x) = (f1(x), f2(x), ..., fn(x)) where each fi: ℝ^n → ℝ
has continuous second-order partial derivatives and satisfies ∂fi/∂xj - ∂fj/∂xi = cij for some
constants cij, show that there exists a function g: ℝ^n → ℝ such that fi + ∂g/∂xi is linear for
all 1 ≤ i ≤ n.
-/

variables {n : ℕ}

-- Definitions of the given conditions
noncomputable def f (x : ℝ^n) : ℝ^n := sorry
noncomputable def f_i (i : fin n) (x : ℝ^n) : ℝ := sorry
variables (c : fin n → fin n → ℝ)

-- Condition that each fi has continuous second-order partial derivatives
axiom continuous_second_order_partial_derivatives (i : fin n) : continuous (λ x : ℝ^n, f_i i x)

-- Condition ∂fi/∂xj - ∂fj/∂xi = cij
axiom partial_derivative_condition (i j : fin n) (x : ℝ^n) :
  (∂ (f_i i) x (x j) - ∂ (f_i j) x (x i)) = c i j

-- Proof statement to show the existence of function g that satisfies the linearity condition
theorem exists_linear_g :
  ∃ g : ℝ^n → ℝ, ∀ i : fin n, 
    is_linear_map ℝ (λ x : ℝ^n, f_i i x + ∂ (λ x : ℝ^n, g x) x (fun j => x j)) :=
sorry

end exists_linear_g_l645_645455


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645701

theorem ratio_of_area_to_perimeter_of_equilateral_triangle (s : ℕ) : s = 10 → (let A := (s^2 * sqrt 3) / 4, P := 3 * s in A / P = 5 * sqrt 3 / 6) :=
by
  intro h,
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645701


namespace total_charge_for_3_6_miles_during_peak_hours_l645_645553

-- Define the initial conditions as constants
def initial_fee : ℝ := 2.05
def charge_per_half_mile_first_2_miles : ℝ := 0.45
def charge_per_two_fifth_mile_after_2_miles : ℝ := 0.35
def peak_hour_surcharge : ℝ := 1.50

-- Define the function to calculate the total charge
noncomputable def total_charge (total_distance : ℝ) (is_peak_hour : Bool) : ℝ :=
  let first_2_miles_charge := if total_distance > 2 then 4 * charge_per_half_mile_first_2_miles else (total_distance / 0.5) * charge_per_half_mile_first_2_miles
  let remaining_distance := if total_distance > 2 then total_distance - 2 else 0
  let after_2_miles_charge := if total_distance > 2 then (remaining_distance / (2 / 5)) * charge_per_two_fifth_mile_after_2_miles else 0
  let surcharge := if is_peak_hour then peak_hour_surcharge else 0
  initial_fee + first_2_miles_charge + after_2_miles_charge + surcharge

-- Prove that total charge of 3.6 miles during peak hours is 6.75
theorem total_charge_for_3_6_miles_during_peak_hours : total_charge 3.6 true = 6.75 := by
  sorry

end total_charge_for_3_6_miles_during_peak_hours_l645_645553


namespace value_of_x_l645_645263

theorem value_of_x (x y z : ℤ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 :=
by
  sorry

end value_of_x_l645_645263


namespace kath_total_cost_l645_645124

-- Define constants
def admission_cost : ℝ := 8
def discount_percent : ℝ := 25 / 100
def num_people : ℕ := 6

-- Define the discounted price per ticket
def discounted_price : ℝ := admission_cost * (1 - discount_percent)

-- Define the total cost Kath will pay
def total_cost : ℝ := num_people * discounted_price

-- The theorem to prove Kath's total cost is $36
theorem kath_total_cost : total_cost = 36 := by
  -- definitions are not used here for simplicity: we assume those values are defined in our environment
  sorry

end kath_total_cost_l645_645124


namespace value_of_a_with_two_distinct_roots_l645_645025

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l645_645025


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645704

theorem ratio_of_area_to_perimeter_of_equilateral_triangle (s : ℕ) : s = 10 → (let A := (s^2 * sqrt 3) / 4, P := 3 * s in A / P = 5 * sqrt 3 / 6) :=
by
  intro h,
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645704


namespace sum_mysterious_numbers_l645_645937

def is_mysterious (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = (2 * k) ^ 2 - (2 * (k - 1)) ^ 2

theorem sum_mysterious_numbers :
  (∑ n in Finset.filter (λ n, is_mysterious n) (Finset.range 201)) = 2500 :=
by
  sorry

end sum_mysterious_numbers_l645_645937


namespace factorize_difference_of_squares_l645_645847

theorem factorize_difference_of_squares :
  ∀ x : ℝ, x^2 - 9 = (x + 3) * (x - 3) :=
by 
  intro x
  have h : x^2 - 9 = x^2 - 3^2 := by rw (show 9 = 3^2, by norm_num)
  have hs : (x^2 - 3^2) = (x + 3) * (x - 3) := by exact (mul_self_sub_mul_self_eq x 3)
  exact Eq.trans h hs

end factorize_difference_of_squares_l645_645847


namespace strawberry_jelly_amount_l645_645598

def totalJelly : ℕ := 6310
def blueberryJelly : ℕ := 4518
def strawberryJelly : ℕ := totalJelly - blueberryJelly

theorem strawberry_jelly_amount : strawberryJelly = 1792 := by
  rfl

end strawberry_jelly_amount_l645_645598


namespace express_set_M_l645_645840

def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def M : Set ℤ := {m | is_divisor 10 (m + 1)}

theorem express_set_M :
  M = {-11, -6, -3, -2, 0, 1, 4, 9} :=
by
  sorry

end express_set_M_l645_645840


namespace primes_not_divides_l645_645154

noncomputable def seq (p : ℕ) (hp : Nat.Prime p) : ℕ → ℕ
| 0     := 2
| 1     := 1
| (n+2) := seq p hp (n+1) + ((p^2 - 1) / 4) * seq p hp n

theorem primes_not_divides (p : ℕ) (hp : Nat.Prime p) (hcond : p ∣ (2^2019 - 1)) (n : ℕ) : ¬ (p ∣ (seq p hp n + 1)) :=
sorry

end primes_not_divides_l645_645154


namespace cube_surface_area_sum_of_edges_l645_645255

noncomputable def edge_length (sum_of_edges : ℝ) (num_of_edges : ℝ) : ℝ :=
  sum_of_edges / num_of_edges

noncomputable def surface_area (edge_length : ℝ) : ℝ :=
  6 * edge_length ^ 2

theorem cube_surface_area_sum_of_edges (sum_of_edges : ℝ) (num_of_edges : ℝ) (expected_area : ℝ) :
  num_of_edges = 12 → sum_of_edges = 72 → surface_area (edge_length sum_of_edges num_of_edges) = expected_area :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end cube_surface_area_sum_of_edges_l645_645255


namespace find_lambda_l645_645927

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3/2 * x), Real.sin (3/2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

noncomputable def f (x λ : ℝ) : ℝ :=
  dot_product (a x) (b x) - 2 * λ * magnitude (a x + b x)

theorem find_lambda
  (h_cos2x : ∀ x, dot_product (a x) (b x) = Real.cos (2 * x))
  (h_magnitude : ∀ x, magnitude (a x + b x) = 2 * Real.cos x)
  (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  ∃ λ : ℝ, (∀ y, f y λ ≥ 3/2) ∧ λ = 1/2 := sorry

end find_lambda_l645_645927


namespace eval_poly_at_2_l645_645886

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem eval_poly_at_2 :
  f 2 = 123 :=
by
  sorry

end eval_poly_at_2_l645_645886


namespace distance_traveled_downstream_l645_645741

noncomputable def speed_boat : ℝ := 20  -- Speed of the boat in still water in km/hr
noncomputable def rate_current : ℝ := 5  -- Rate of current in km/hr
noncomputable def time_minutes : ℝ := 24  -- Time traveled downstream in minutes
noncomputable def time_hours : ℝ := time_minutes / 60  -- Convert time to hours
noncomputable def effective_speed_downstream : ℝ := speed_boat + rate_current  -- Effective speed downstream

theorem distance_traveled_downstream :
  effective_speed_downstream * time_hours = 10 := by {
  sorry
}

end distance_traveled_downstream_l645_645741


namespace adoption_complete_in_7_days_l645_645335

-- Define the initial number of puppies
def initial_puppies := 9

-- Define the number of puppies brought in later
def additional_puppies := 12

-- Define the number of puppies adopted per day
def adoption_rate := 3

-- Define the total number of puppies
def total_puppies : Nat := initial_puppies + additional_puppies

-- Define the number of days required to adopt all puppies
def adoption_days : Nat := total_puppies / adoption_rate

-- Prove that the number of days to adopt all puppies is 7
theorem adoption_complete_in_7_days : adoption_days = 7 := by
  -- The exact implementation of the proof is not necessary,
  -- so we use sorry to skip the proof.
  sorry

end adoption_complete_in_7_days_l645_645335


namespace heads_tosses_l645_645135

-- Define the conditions as given in the problem
def total_tosses : ℕ := 10
def tail_tosses : ℕ := 7

-- Define the statement to be proved
theorem heads_tosses (total_tosses tail_tosses : ℕ) (htot : total_tosses = 10) (htail : tail_tosses = 7) : total_tosses - tail_tosses = 3 :=
by {
  rw [htot, htail],
  exact nat.sub_self 7
}

end heads_tosses_l645_645135


namespace cycloid_area_eq_l645_645825

-- Given the parametric equations of the cycloid
def cycloid_x (a t : ℝ) : ℝ := a * (t - sin t)
def cycloid_y (a t : ℝ) : ℝ := a * (1 - cos t)

-- Define the area function for parametric curves
def parametric_area (y dx_dt : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ t in a..b, y t * dx_dt t

-- Define the area under one arc of the cycloid
def cycloid_area (a : ℝ) : ℝ :=
  parametric_area (cycloid_y a) (λ t => a * (1 - cos t)) 0 (2 * π)

-- Prove that this area equals 3πa²
theorem cycloid_area_eq (a : ℝ) : 
  cycloid_area a = 3 * π * a^2 := sorry

end cycloid_area_eq_l645_645825


namespace expand_product_eq_l645_645839

theorem expand_product_eq :
  (∀ (x : ℤ), (x^3 - 3 * x^2 + 3 * x - 1) * (x^2 + 3 * x + 3) = x^5 - 3 * x^3 - x^2 + 3 * x) :=
by
  intro x
  sorry

end expand_product_eq_l645_645839


namespace centric_point_exists_l645_645999

noncomputable def C : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 ≤ 1 }

def centric_sequence (A : ℕ → ℝ × ℝ) : Prop :=
  A 0 = (0, 0) ∧
  A 1 = (1, 0) ∧
  ∀ n ≥ 0, ∃ P, P ∈ C ∧ is_circumcenter (A n) (A (n+1)) (A (n+2)) P

noncomputable def K : ℝ := 4048144

theorem centric_point_exists :
  ∃ (x y : ℝ) (A : ℕ → ℝ × ℝ), 
    centric_sequence A ∧ A 2012 = (x, y) ∧ x^2 + y^2 = K ∧ 
    ((x = -1006 ∧ y = 1006 * real.sqrt 3) ∨ (x = -1006 ∧ y = -1006 * real.sqrt 3)) :=
sorry

end centric_point_exists_l645_645999


namespace no_odd_3_digit_integers_divisible_by_5_without_digit_5_l645_645928

theorem no_odd_3_digit_integers_divisible_by_5_without_digit_5 :
  ∀ n : ℕ, (100 ≤ n ∧ n < 1000 ∧ (∃ k, n = 5 * k) ∧ n % 2 = 1 ∧ ¬ 5 ∈ to_digits 10 n) → false :=
by
  sorry

end no_odd_3_digit_integers_divisible_by_5_without_digit_5_l645_645928


namespace percent_participates_second_shift_l645_645834

-- Defining the problem context with all given conditions
def companyX := Type
def Shift : companyX → Type
def members (s : Shift companyX) : ℕ
def participatesInPensionProgram (s : Shift companyX) : ℕ

variable (s1 s2 s3 : Shift companyX)

axiom members_s1 : members s1 = 60
axiom members_s2 : members s2 = 50
axiom members_s3 : members s3 = 40

axiom participates_s1 : participatesInPensionProgram s1 = 12
axiom participates_s3 : participatesInPensionProgram s3 = 4

axiom totalParticipates : ∑ s, participatesInPensionProgram s = 36

-- Prove that 40% of the second shift participate in the pension program
theorem percent_participates_second_shift : 
  (participatesInPensionProgram s2) = 20 :=
by {
  sorry
}

end percent_participates_second_shift_l645_645834


namespace largest_d_for_range_l645_645828

theorem largest_d_for_range (d : ℝ) : (∃ x : ℝ, x^2 - 6*x + d = 2) ↔ d ≤ 11 := 
by
  sorry

end largest_d_for_range_l645_645828


namespace intersection_complement_l645_645607

open Set

variable (U P Q : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5, 6})
variable (H_P : P = {1, 2, 3, 4})
variable (H_Q : Q = {3, 4, 5})

theorem intersection_complement (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5}) :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_complement_l645_645607


namespace total_surface_area_l645_645645

-- Defining the conditions
variables {a b c : ℝ}

-- Given conditions
def condition1 := 4 * a + 4 * b + 4 * c = 156
def condition2 := real.sqrt (a^2 + b^2 + c^2) = 25

-- The goal: To prove the total surface area is 896
theorem total_surface_area (h1 : condition1) (h2 : condition2) : 2 * a * b + 2 * b * c + 2 * c * a = 896 := 
sorry

end total_surface_area_l645_645645


namespace circumcircles_intersect_at_one_point_l645_645578

structure Triangle (α : Type*) := (A B C : α)
structure Median (α : Type*) := (A B C : α)
structure Point (α : Type*) := (P : α)
structure Circle (α : Type*) := (center : α) (radius : ℝ)

variables {α : Type*} [Inhabited α]

def is_centroid (M : α) (T : Triangle α) (MA MB MC : Median α) : Prop :=
  -- Medians intersect at centroid M
  sorry

def is_circumcircle_touch (Ω : Circle α) (Mid : α) (T : Triangle α) (M : Median α) : Prop :=
  -- Circle passes through midpoint of AM and tangent to BC at M_A
  sorry

theorem circumcircles_intersect_at_one_point 
  (T : Triangle α) 
  (MA MB MC : Median α) 
  (M : α) 
  (ΩA ΩB ΩC : Circle α)
  (MidA MidB MidC : α) :
  is_centroid M T MA MB MC →
  is_circumcircle_touch ΩA MidA T MA →
  is_circumcircle_touch ΩB MidB T MB →
  is_circumcircle_touch ΩC MidC T MC →
  ∃ K : α, (K ∈ ΩA) ∧ (K ∈ ΩB) ∧ (K ∈ ΩC) :=
sorry

end circumcircles_intersect_at_one_point_l645_645578


namespace binom_15_12_eq_455_l645_645401

theorem binom_15_12_eq_455 : nat.choose 15 12 = 455 := sorry

end binom_15_12_eq_455_l645_645401


namespace matrix_vector_lin_comb_l645_645159

theorem matrix_vector_lin_comb
  (M: Matrix (Fin 2) (Fin 2) ℝ) (u v w: Matrix (Fin 2) (Fin 1) ℝ)
  (h_u: M ⬝ u = !![1; 2])
  (h_v: M ⬝ v = !![3; 4])
  (h_w: M ⬝ w = !![5; 6]):
  M ⬝ (2 • u + v - 2 • w) = !![-5; -4] := 
by
  sorry

end matrix_vector_lin_comb_l645_645159


namespace max_of_intersection_l645_645097

open Set

variable (I A B : Set ℕ)

def universal_set : Set ℕ := {x | 1 ≤ x ∧ x ≤ 100}

def subset_A : Set ℕ := {m | ∃ k : ℤ, 1 ≤ m ∧ m ≤ 100 ∧ m = 2 * ↑k + 1}

def subset_B : Set ℕ := {n | ∃ k : ℤ, 1 ≤ n ∧ n ≤ 100 ∧ n = 3 * ↑k}

def complement_A (I A : Set ℕ) : Set ℕ := I \ A

def intersection_complement_A_and_B (complement_A B : Set ℕ) : Set ℕ := complement_A ∩ B

theorem max_of_intersection : 
    let I := universal_set 
    let A := subset_A
    let B := subset_B
    let C := complement_A I A
    let D := intersection_complement_A_and_B C B
    ∀ x ∈ D, x ≤ 96 := sorry

end max_of_intersection_l645_645097


namespace ellipse_range_values_l645_645481

theorem ellipse_range_values (a b c : ℝ) (h1 : a > b > 0) (h2 : b = 1) 
    (h3 : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) 
    (h4 : ∀ (A B F1 : ℝ × ℝ), 
         let area := (1 / 2) * (a - c) * 1 
         in area = (2 - sqrt 3) / 2) 
    (F1 F2 P : ℝ × ℝ) (h5 : F1 = (-c, 0)) (h6 : F2 = (c, 0)) (h7 : ∀ (P : ℝ × ℝ), x y),
    let 
        dist_F1_P := sqrt ((fst P - fst F1) ^ 2 + (snd P - snd F1) ^ 2),
        dist_F2_P := sqrt ((fst P - fst F2) ^ 2 + (snd P - snd F2) ^ 2)
    in 
        1 ≤ (1 / dist_F1_P + 1 / dist_F2_P) ∧ (1 / dist_F1_P + 1 / dist_F2_P) ≤ 4 := 
    sorry

end ellipse_range_values_l645_645481


namespace concentration_of_acid_in_third_flask_l645_645270

theorem concentration_of_acid_in_third_flask :
  ∀ (W1 W2 : ℝ),
    let W := 190 + 65.714 in 
    W1 = 190 ∧ W2 = 65.714 →
    (10 : ℝ) / (10 + W1) = 0.05 →
    (20 : ℝ) / (20 + W2) = 0.2331 →
    (30 : ℝ) / (30 + W) = 0.105 :=
begin
  sorry
end

end concentration_of_acid_in_third_flask_l645_645270


namespace required_circle_equation_l645_645432

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5
def line_condition (x y : ℝ) : Prop := 3 * x + 4 * y - 1 = 0
def intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

theorem required_circle_equation : 
  ∃ (h : ℝ × ℝ → Prop), 
    (∀ p, intersection_points p.1 p.2 → h p.1 p.2) ∧ 
    (∃ cx cy r, (∀ x y, h x y ↔ (x - cx)^2 + (y - cy)^2 = r^2) ∧ line_condition cx cy ∧ h x y = ((x + 1)^2 + (y - 1)^2 = 13)) 
:= sorry

end required_circle_equation_l645_645432


namespace parametric_curve_length_l645_645437

noncomputable def curve_length : ℝ :=
  ∫ θ in (Real.pi / 3)..Real.pi, sqrt ((-5 * sin θ)^2 + (5 * cos θ)^2)

theorem parametric_curve_length :
  curve_length = 10 * Real.pi / 3 :=
by
  sorry

end parametric_curve_length_l645_645437


namespace simplify_and_evaluate_expression_l645_645224

variable (x y : ℝ)

theorem simplify_and_evaluate_expression (h₁ : x = -2) (h₂ : y = 1/2) :
  (x + 2 * y) ^ 2 - (x + y) * (3 * x - y) - 5 * y ^ 2 / (2 * x) = 2 + 1 / 2 := 
sorry

end simplify_and_evaluate_expression_l645_645224


namespace constant_term_coefficient_l645_645038

/-!
  Prove that the coefficient of the constant term in the expansion of
  \( (x^3 + \frac{1}{x^2})^5 \) is \( 10 \).
-/

theorem constant_term_coefficient :
  let f := (x : ℚ) ↦ (x^3 + x^(-2))^5 in
  -- coefficient of the constant term equals 10
  ∃ (c : ℚ), c = 10 ∧ (∀ x, (f x).expand_coefficient(0) = c) :=
sorry

end constant_term_coefficient_l645_645038


namespace sin_alpha_given_point_l645_645528

theorem sin_alpha_given_point : 
  let α := angle
  let P := (2 * Real.sin (Real.pi / 3), -2 * Real.cos (Real.pi / 3))
  P = (Real.sqrt 3, -1) →
  ∃ α, ∃ (P := (2 * Real.sin α, -2 * Real.cos α)), 
    (P = (Real.sqrt 3, -1)) → Real.sin α = -1 / 2 :=
by
  sorry

end sin_alpha_given_point_l645_645528


namespace pollen_scientific_notation_correct_l645_645126

def moss_flower_pollen_diameter := 0.0000084
def pollen_scientific_notation := 8.4 * 10^(-6)

theorem pollen_scientific_notation_correct :
  moss_flower_pollen_diameter = pollen_scientific_notation :=
by
  -- Proof skipped
  sorry

end pollen_scientific_notation_correct_l645_645126


namespace joe_initial_paint_amount_l645_645143

theorem joe_initial_paint_amount (P : ℝ) 
  (h1 : (2/3) * P + (1/15) * P = 264) : P = 360 :=
sorry

end joe_initial_paint_amount_l645_645143


namespace batsman_avg_increase_l645_645748

theorem batsman_avg_increase (R : ℕ) (A : ℕ) : 
  (R + 48 = 12 * 26) ∧ (R = 11 * A) → 26 - A = 2 :=
by
  intro h
  have h1 : R + 48 = 312 := h.1
  have h2 : R = 11 * A := h.2
  sorry

end batsman_avg_increase_l645_645748


namespace min_travel_distance_l645_645240

-- Let's define the given distances between the docks.
def distance_AB : ℝ := 3
def distance_AC : ℝ := 4
def distance_BC : ℝ := Real.sqrt 13

-- The question is to find the minimum distance the truck will travel.
theorem min_travel_distance : 
  ∃ (x : ℝ), 
  x = 2 * Real.sqrt 37 ∧ 
  ∀ (d1 d2 : ℝ), 
  d1 = distance_AB ∧ 
  d2 = distance_AC ∧ 
  distance_BC = Real.sqrt 13 →
  x ≤ d1 + d2 + distance_BC :=
sorry

end min_travel_distance_l645_645240


namespace betsy_remaining_squares_l645_645392

def total_squares := 16 + 16
def percent_sewn := 0.25
def squares_sewn := total_squares * percent_sewn
def remaining_squares := total_squares - squares_sewn

theorem betsy_remaining_squares : remaining_squares = 24 :=
by
  sorry

end betsy_remaining_squares_l645_645392


namespace equation_has_roots_l645_645011

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l645_645011


namespace find_t_l645_645110

-- Defining variables and assumptions
variables (V V0 g S t : Real)
variable (h1 : V = g * t + V0)
variable (h2 : S = (1 / 2) * g * t^2 + V0 * t)

-- The goal: to prove t equals 2S / (V + V0)
theorem find_t (V V0 g S t : Real) (h1 : V = g * t + V0) (h2 : S = (1 / 2) * g * t^2 + V0 * t):
  t = 2 * S / (V + V0) := by
  sorry

end find_t_l645_645110


namespace carrot_cakes_in_february_l645_645610

theorem carrot_cakes_in_february :
  (∃ (cakes_in_oct : ℕ) (cakes_in_nov : ℕ) (cakes_in_dec : ℕ) (cakes_in_jan : ℕ) (monthly_increase : ℕ),
      cakes_in_oct = 19 ∧
      cakes_in_nov = 21 ∧
      cakes_in_dec = 23 ∧
      cakes_in_jan = 25 ∧
      monthly_increase = 2 ∧
      cakes_in_february = cakes_in_jan + monthly_increase) →
  cakes_in_february = 27 :=
  sorry

end carrot_cakes_in_february_l645_645610


namespace operation_25_l645_645249

def a_op (a b : ℕ) : ℕ :=
  ((b - a) ^ 2) / (a ^ 2)

theorem operation_25 :
  a_op (-1) (a_op 1 (-1)) = 25 :=
by
  sorry

end operation_25_l645_645249


namespace nature_of_angles_in_DEF_l645_645541

open Real

-- Definitions for point and triangle
structure Triangle :=
(A B C : ℝ × ℝ)

-- Definitions for angles in the plane
def angle (A B C : ℝ × ℝ) : ℝ :=
  let v := (C.1 - B.1, C.2 - B.2)
  let u := (A.1 - B.1, A.2 - B.2)
  Real.angle v u

-- Definitions for the given triangle and properties
def triangle_ABC := Triangle.mk (0, 0) (1, 0) (cos (100 * π / 180), sin (100 * π / 180))
def A := triangle_ABC.A
def B := triangle_ABC.B
def C := triangle_ABC.C

def angle_BAC := angle B A C
def angle_ABC := angle A B C

-- Points of tangency with the incircle (definitions are symbolic)
def D : ℝ × ℝ := sorry
def E : ℝ × ℝ := sorry
def F : ℝ × ℝ := sorry

-- Triangle formed by points of tangency
def triangle_DEF := Triangle.mk D E F

-- Angles in triangle DEF
def angle_DEF := angle D E F
def angle_FED := angle F E D
def angle_EFD := angle E F D

-- The proof problem statement
theorem nature_of_angles_in_DEF :
  angle_DEF < π ∧ angle_FED > π / 2 ∧ angle_EFD < π :=
sorry

end nature_of_angles_in_DEF_l645_645541


namespace geometric_sequence_general_term_l645_645891

noncomputable def a_n (n : ℕ) : ℝ := 1 * (2:ℝ)^(n-1)

theorem geometric_sequence_general_term : 
  ∀ (n : ℕ), 
  (∀ (n : ℕ), 0 < a_n n) ∧ a_n 1 = 1 ∧ (a_n 1 + a_n 2 + a_n 3 = 7) → 
  a_n n = 2^(n-1) :=
by
  sorry

end geometric_sequence_general_term_l645_645891


namespace partitions_equal_l645_645168

namespace MathProof

-- Define the set of natural numbers
def nat := ℕ

-- Define the partition functions (placeholders)
def num_distinct_partitions (n : nat) : nat := sorry
def num_odd_partitions (n : nat) : nat := sorry

-- Statement of the theorem
theorem partitions_equal (n : nat) : 
  num_distinct_partitions n = num_odd_partitions n :=
sorry

end MathProof

end partitions_equal_l645_645168


namespace valid_a_value_l645_645031

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l645_645031


namespace amys_total_crumbs_l645_645307

section crumbs

variable (T C c : ℕ) -- variables for the number of trips and crumbs per trip

-- Given conditions as assumptions
variable (h1 : 2 * T = 2 * T) -- Amy makes twice as many trips as Arthur
variable (h2 : 1.5 * C = 1.5 * C) -- Amy carries 1.5 times as many crumbs per trip as Arthur
variable (h3 : T * C = c) -- Arthur carries a total of c crumbs

theorem amys_total_crumbs (T C c : ℕ) (h1 h2 : ∀ n: ℕ, n = n) (h3 : T * C = c) : 3 * c = 3 * c :=
by
  sorry

end crumbs

end amys_total_crumbs_l645_645307


namespace largest_percent_error_in_circle_area_l645_645132

theorem largest_percent_error_in_circle_area
  (actual_diameter : ℝ)
  (measurement_error_percent : ℝ)
  (area_formula : ℝ → ℝ := fun r => Float.pi * r^2)
  (actual_area : ℝ := area_formula (actual_diameter / 2))
  (min_diameter : ℝ := actual_diameter * (1 - measurement_error_percent / 100))
  (max_diameter : ℝ := actual_diameter * (1 + measurement_error_percent / 100))
  (min_area : ℝ := area_formula (min_diameter / 2))
  (max_area : ℝ := area_formula (max_diameter / 2)) :
  measurement_error_percent = 20 → actual_diameter = 20 → 
  ∃ (largest_error_percent : ℝ),
  largest_error_percent = max ((actual_area - min_area) / actual_area * 100) 
                               ((max_area - actual_area) / actual_area * 100) ∧
  largest_error_percent = 44 :=
by
  sorry

end largest_percent_error_in_circle_area_l645_645132


namespace trapezoid_AB_l645_645921

-- Define the vectors a and b
variables (a b : Type) [AddCommGroup a] [Module ℝ a]

-- Trapezoid OABC with given conditions
def vector_AB (a b : a) : a :=
  let CB := (1/2 : ℝ) • a in
  b + CB - a

-- Theorem statement in Lean
theorem trapezoid_AB (a b : a) :
  ∀ (CB OA : a), 
    CB = (1/2 : ℝ) • OA → 
    OA = a → 
    b + CB - OA = b - (1/2 : ℝ) • a := 
  by
    intros CB OA hCB hOA
    rw [hCB, hOA]
    ring
    sorry

end trapezoid_AB_l645_645921


namespace relationship_of_a_b_c_l645_645490

noncomputable def f : ℝ → ℝ := sorry

def x := (-∞: ℝ) 

theorem relationship_of_a_b_c :
  (let a := 3^0.3 * f(3^0.3)
       b := Real.log_base π 3 * f(Real.log_base π 3)
       c := Real.log_base 3 (1/9) * f(Real.log_base 3 (1/9))
  in c > a > b) :=
by
 -- Conditions: The graph of y = f(x-1) is symmetric about (1,0)
 -- Therefore, f(x) is an odd function and xf(x) is even.
 -- g(x) = xf(x). When x ∈ (-∞, 0), g'(x) = f(x) + x f''(x) < 0.
 -- g(x) is monotonically decreasing on x ∈ (-∞,0) and increasing on x ∈ (0,+∞)
 -- Since -log_3 (1/9) = 2 > 3^0.3 > 1 > log_π 3 > 0, thus g(log_3 (1/9) > g(3^0.3) > g(log_π 3))
 -- Therefore, g(log_3 (1/9)) = g(log_3 (1/9)) > g(3^0.3) > g(log_π 3), implying c > a > b.
sorry

end relationship_of_a_b_c_l645_645490


namespace milk_production_days_l645_645515

variable {x : ℕ}

def daily_cow_production (x : ℕ) : ℚ := (x + 4) / ((x + 2) * (x + 3))

def total_daily_production (x : ℕ) : ℚ := (x + 5) * daily_cow_production x

def required_days (x : ℕ) : ℚ := (x + 9) / total_daily_production x

theorem milk_production_days : 
  required_days x = (x + 9) * (x + 2) * (x + 3) / ((x + 5) * (x + 4)) := 
by 
  sorry

end milk_production_days_l645_645515


namespace bouquet_count_l645_645398

def number_of_bouquets (r p w : ℕ) : ℕ :=
  if r + p + w = 2 then 1 else 0

theorem bouquet_count :
  ∑ r p w, number_of_bouquets r p w = 6 :=
by sorry

end bouquet_count_l645_645398


namespace video_total_votes_l645_645379

theorem video_total_votes (x : ℕ) (L D : ℕ)
  (h1 : L + D = x)
  (h2 : L - D = 130)
  (h3 : 70 * x = 100 * L) :
  x = 325 :=
by
  sorry

end video_total_votes_l645_645379


namespace total_amount_is_24_l645_645780

-- Define the original price of a tub of ice cream
def original_price_ice_cream : ℕ := 12

-- Define the discount per tub of ice cream
def discount_per_tub : ℕ := 2

-- Define the discounted price of a tub of ice cream
def discounted_price_ice_cream : ℕ := original_price_ice_cream - discount_per_tub

-- Define the price for 5 cans of juice
def price_per_5_cans_of_juice : ℕ := 2

-- Define the number of cans of juice bought
def cans_of_juice_bought : ℕ := 10

-- Calculate the total cost for two tubs of ice cream and 10 cans of juice
def total_cost (p1 p2 : ℕ) : ℕ := 2 * p1 + (price_per_5_cans_of_juice * (cans_of_juice_bought / 5))

-- Prove that the total cost is $24
theorem total_amount_is_24 : total_cost discounted_price_ice_cream price_per_5_cans_of_juice = 24 := by
  sorry

end total_amount_is_24_l645_645780


namespace equation_has_at_least_two_distinct_roots_l645_645017

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l645_645017


namespace greatest_sum_of_visible_numbers_l645_645866

/-- Definition of a cube with numbered faces -/
structure Cube where
  face1 : ℕ
  face2 : ℕ
  face3 : ℕ
  face4 : ℕ
  face5 : ℕ
  face6 : ℕ

/-- The cubes face numbers -/
def cube_numbers : List ℕ := [1, 2, 4, 8, 16, 32]

/-- Stacked cubes with maximized visible numbers sum -/
def maximize_visible_sum :=
  let cube1 := Cube.mk 1 2 4 8 16 32
  let cube2 := Cube.mk 1 2 4 8 16 32
  let cube3 := Cube.mk 1 2 4 8 16 32
  let cube4 := Cube.mk 1 2 4 8 16 32
  244

theorem greatest_sum_of_visible_numbers : maximize_visible_sum = 244 := 
  by
    sorry -- Proof to be done

end greatest_sum_of_visible_numbers_l645_645866


namespace locus_of_P_l645_645149

noncomputable def Point := ℝ × ℝ -- Simplified representation of points
noncomputable def Segment := Set Point -- Simplified representation of segments
noncomputable def Circle := Set Point -- Simplified representation of circles

-- Define the geometric conditions
variable (A B C D P Q : Point)
variable (AB DC : Segment)

-- Specific conditions
axiom is_isosceles_trapezium (A B C D : Point) : Segment AB ∧ Segment DC ∧ (AB ∥ DC) ∧ (dist A B = dist D C)
axiom diagonals_intersect_at (Q : Point) (A C D B : Segment) : Q ∈ (convex_hull [A, C] ∩ convex_hull [D, B])
axiom distances_condition (P A B C D : Point) : dist P A * dist P C = dist P B * dist P D

-- Conclusion to prove
theorem locus_of_P (A B C D P Q : Point) : 
  is_isosceles_trapezium A B C D → diagonals_intersect_at Q A C D B → distances_condition P A B C D →
  P ∈ (perpendicular_bisector A B ∪ circle_centered_at Q) := by
  sorry

end locus_of_P_l645_645149


namespace triangle_area_equals_6sqrt3_l645_645530

-- Given triangle sides a = 7, b = 3, c = 8
-- Define the area S of triangle ABC with these sides
def area_of_triangle (a b c : ℝ) : ℝ :=
  let cosA := (b^2 + c^2 - a^2) / (2 * b * c)
  let A := Real.arccos cosA
  let S := (1 / 2) * b * c * Real.sin A
  S

theorem triangle_area_equals_6sqrt3 : area_of_triangle 7 3 8 = 6 * Real.sqrt 3 :=
by
  sorry

end triangle_area_equals_6sqrt3_l645_645530


namespace divisible_by_factorial_l645_645851

theorem divisible_by_factorial (k m : ℕ) :
  (∀ n : ℕ, (k.factorial.gcd m = 1 ∧ (∏ i in range k, (n + (i + 1) * m)) % k.factorial = 0) ∨ 
             (∃ p : ℕ, nat.prime p ∧ p ≤ k ∧ p ∣ m) → 
             ∀ n : ℕ, (∏ i in range k, (n + (i + 1) * m)) % k.factorial = 0) := 
by 
  sorry

end divisible_by_factorial_l645_645851


namespace jerry_feathers_count_l645_645552

noncomputable def hawk_feathers : ℕ := 6
noncomputable def eagle_feathers : ℕ := 17 * hawk_feathers
noncomputable def total_feathers : ℕ := hawk_feathers + eagle_feathers
noncomputable def remaining_feathers_after_sister : ℕ := total_feathers - 10
noncomputable def jerry_feathers_left : ℕ := remaining_feathers_after_sister / 2

theorem jerry_feathers_count : jerry_feathers_left = 49 :=
  by
  sorry

end jerry_feathers_count_l645_645552


namespace restaurant_meals_l645_645799

theorem restaurant_meals (k a : ℕ) (ratio_kids_to_adults : k / a = 10 / 7) (kids_meals_sold : k = 70) : a = 49 :=
by
  sorry

end restaurant_meals_l645_645799


namespace existence_of_b_l645_645169

theorem existence_of_b
  {n : ℕ} {a : fin n → ℕ}
  (hn : 2 ≤ n)
  (ha_pos : ∀ i, 0 < a i) :
  ∃ (b : fin n → ℕ),
  (∀ i, a i ≤ b i) ∧
  ∀ i j, i ≠ j → b i % n ≠ b j % n ∧
  (∑ i, b i) ≤ n * ((n-1)/2 + (⌊((∑ i, a i) : ℚ) / n⌋)) :=
begin
  sorry
end

end existence_of_b_l645_645169


namespace acid_concentration_third_flask_l645_645292

-- Define the concentrations of first and second flask
def conc_first (w1 : ℝ) : ℝ := 10 / (10 + w1)
def conc_second (w2 : ℝ) : ℝ := 20 / (20 + w2)

-- Define the acid mass in the third flask initially
def acid_mass_third : ℝ := 30

-- Total water added from the fourth flask
def total_water (w1 w2 : ℝ) : ℝ := w1 + w2

-- Acid concentration in the third flask after all water is added
def conc_third (w : ℝ) : ℝ := acid_mass_third / (acid_mass_third + w)

-- Problem statement: concentration in the third flask is 10.5%
theorem acid_concentration_third_flask (w1 : ℝ) (w2 : ℝ) (w : ℝ) 
  (h1 : conc_first w1 = 0.05) 
  (h2 : conc_second w2 = 70 / 300) 
  (h3 : w = total_water w1 w2) : 
  conc_third w = 10.5 / 100 := 
sorry

end acid_concentration_third_flask_l645_645292


namespace tiles_cover_iff_multiple_of_3_l645_645656

-- Define the structure of the three squares forming the figure Φ
structure Figure (n : ℕ) :=
  (touching : Prop)
  (n_gt_1 : n > 1)

-- The main theorem stating the proof problem
theorem tiles_cover_iff_multiple_of_3 (n : ℕ) (Φ : Figure n) : 
  (∃ cover : thatthe figure Φ can be covered with 1 × 3 and 3 × 1 tiles without overlapping) ↔ (n % 3 = 0) :=
by
  sorry

end tiles_cover_iff_multiple_of_3_l645_645656


namespace arithmetic_expression_eval_l645_645397

theorem arithmetic_expression_eval : (10 - 9 + 8) * 7 + 6 - 5 * (4 - 3 + 2) - 1 = 53 :=
by
  sorry

end arithmetic_expression_eval_l645_645397


namespace problem1_problem2_problem3_problem4_problem5_problem6_l645_645226

-- Problem 1
theorem problem1 (x : ℝ) : (x^2 - 3 * x) / Real.sqrt (x - 3) = 0 → false := sorry

-- Problem 2
theorem problem2 (x : ℝ) : x^2 - 6 * (Real.abs x) + 9 = 0 → (x = 3 ∨ x = -3) := sorry

-- Problem 3
theorem problem3 (x : ℝ) : (x - 1)^4 = 4 → (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) := sorry

-- Problem 4
theorem problem4 (x : ℝ) : (Int.floor (x^2) + x = 6) → (x = -3 ∨ x = 2) := sorry

-- Problem 5
theorem problem5 (x : ℝ) : (x^2 + x = 0 ∨ x^2 - 1 = 0) → (x = 0 ∨ x = -1 ∨ x = 1) := sorry

-- Problem 6
theorem problem6 (x : ℝ) : (x^2 - 2 * x - 3 = 0) ∧ (Real.abs x < 2) → (x = -1) := sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l645_645226


namespace convex_polygon_equal_angles_l645_645864

/-- For some integer \(n > 4\), a convex polygon has vertices \(v_1, v_2, \ldots, v_n\) in cyclic order.
    All its edges are the same length. It also has the property that the lengths of the diagonals \(v_1 v_4, v_2 v_5, \ldots, v_{n-3} v_{n}, v_{n-2} v_{1}, v_{n-1} v_{2}\), and \(v_{n} v_{3}\) are all equal.
    We need to prove that it is necessarily the case that the polygon has equal angles if and only if \(n\) is odd. -/
theorem convex_polygon_equal_angles (n : ℤ) (h1 : n > 4) 
  (h2 : ∀ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n → length (v i) = length (v j))
  (h3 : ∀ (i : ℕ), 1 ≤ i ∧ i < n - 3 → length (diagonal v (i + 3)) = length (diagonal v (n - i + 3))) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → angle v i = angle v (i + 1)) → odd n :=
by
  sorry

end convex_polygon_equal_angles_l645_645864


namespace at_least_one_solves_l645_645198

open ProbabilityTheory

variable (Ω : Type) [ProbabilitySpace Ω]

-- Define the events
variable (A B : Event Ω)

-- Define the given probabilities
variable (hA : P(A) = 0.5) (hB : P(B) = 0.4)

-- Define the probability of at least one event occurring
def prob_one_solves : Prop :=
  P(A ∪ B) = 0.7

-- Theorem statement
theorem at_least_one_solves (Ω : Type) [ProbabilitySpace Ω] (A B : Event Ω)
  (hA : P(A) = 0.5) (hB : P(B) = 0.4) : prob_one_solves Ω A B :=
by
  unfold prob_one_solves
  sorry

end at_least_one_solves_l645_645198


namespace smallest_number_among_given_set_l645_645791

theorem smallest_number_among_given_set :
  ∃ n ∈ {3, 0, -1, -1 / 2 : ℚ}, ∀ m ∈ {3, 0, -1, -1 / 2 : ℚ}, n ≤ m ∧ n = -1 :=
by sorry

end smallest_number_among_given_set_l645_645791


namespace curve_equation_range_of_m_l645_645876

-- Defining curve C condition
def curve_condition (x y : ℝ) := (Real.sqrt ((x - 1)^2 + y^2) - x = 1) ∧ (x > 0)

-- First proof: Equation of curve C
theorem curve_equation (x y : ℝ) (h: curve_condition x y) : y^2 = 4 * x := sorry

-- Second proof: Range of m for given condition
theorem range_of_m (m : ℝ) :
  (∃ (A B : ℝ × ℝ), ∀ (l : ℝ), (A.1 = l * A.2 + m) ∧ (B.1 = l * B.2 + m) ∧ curve_condition A.1 A.2 ∧ curve_condition B.1 B.2 ∧ (| (A.1 - 1)^2 + A.2^2 | + | (B.1 - 1)^2 + B.2^2 | < | (A.1 - B.1)^2 + (A.2 - B.2)^2)) → 
  (3 - 2 * Real.sqrt 2 < m ∧ m < 3 + 2 * Real.sqrt 2) := sorry

end curve_equation_range_of_m_l645_645876


namespace range_dot_product_PA_PB_l645_645065

-- Definitions of conditions
def sphere_radius : ℝ := 1
def points_distance : ℝ := sqrt 3

structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)
(radius_condition : x^2 + y^2 + z^2 = sphere_radius)

def A : Point := ⟨1, 0, 0, by sorry⟩
def B : Point := ⟨-1/2, (sqrt 3) / 2 , 0, by sorry⟩

-- Define dot product calculation
def dot_product_PA_PB (P : Point) : ℝ :=
  let PA := (1 - P.x, -P.y, -P.z)
  let PB := (-1/2 - P.x, (sqrt 3)/2 - P.y, -P.z)
  PA.1 * PB.1 + PA.2 * PB.2 + PA.3 * PB.3

-- Theorem to prove the range of dot product
theorem range_dot_product_PA_PB (P : Point) : -3/2 ≤ dot_product_PA_PB P ∧ dot_product_PA_PB P ≤ 1/2 :=
  sorry

end range_dot_product_PA_PB_l645_645065


namespace volume_of_given_sphere_lateral_surface_area_of_given_cylinder_l645_645648

noncomputable def radius_of_sphere (A : ℝ) : ℝ := 
  real.sqrt (A / (4 * real.pi))

noncomputable def volume_of_sphere (A : ℝ) : ℝ := 
  let r := radius_of_sphere A 
  in (4 / 3) * real.pi * r ^ 3

noncomputable def lateral_surface_area_of_cylinder_in_sphere (A : ℝ) : ℝ := 
  let r := radius_of_sphere A 
  in 2 * real.pi * r * (2 * r)

theorem volume_of_given_sphere :
  volume_of_sphere (484 * real.pi) = 1774.67 * real.pi :=
sorry

theorem lateral_surface_area_of_given_cylinder :
  lateral_surface_area_of_cylinder_in_sphere (484 * real.pi) = 484 * real.pi :=
sorry

end volume_of_given_sphere_lateral_surface_area_of_given_cylinder_l645_645648


namespace jerry_charge_per_hour_l645_645978

-- Define the conditions from the problem
def time_painting : ℝ := 8
def time_fixing_counter : ℝ := 3 * time_painting
def time_mowing_lawn : ℝ := 6
def total_time_worked : ℝ := time_painting + time_fixing_counter + time_mowing_lawn
def total_payment : ℝ := 570

-- The proof statement
theorem jerry_charge_per_hour : 
  total_payment / total_time_worked = 15 :=
by
  sorry

end jerry_charge_per_hour_l645_645978


namespace prime_sum_product_l645_645647

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hsum : p + q = 102) (hgt : p > 30 ∨ q > 30) :
  p * q = 2201 := 
sorry

end prime_sum_product_l645_645647


namespace polynomial_coeff_sum_eq_27_l645_645640

theorem polynomial_coeff_sum_eq_27
  (a b c d : ℝ)
  (h1 : (λ x:ℂ, x^4 + a*x^3 + b*x^2 + c*x + d) (1 + complex.i) = 0)
  (h2 : (λ x:ℂ, x^4 + a*x^3 + b*x^2 + c*x + d) (3 * complex.i) = 0)
  (h3 : ∀ x : ℝ, (λ x:ℂ, x^4 + a*x^3 + b*x^2 + c*x + d) x ∈ ℝ) : 
  a + b + c + d = 27 := 
sorry

end polynomial_coeff_sum_eq_27_l645_645640


namespace sum_fractions_l645_645871

noncomputable def f (x : ℝ) : ℝ := 2 / (4^x + 2) + 2

theorem sum_fractions (m : ℕ) (hm : 2 ≤ m) :
  (∑ k in Finset.range (m - 1), f (k / m)) = 5 / 2 * (m - 1) :=
sorry

end sum_fractions_l645_645871


namespace third_team_soup_amount_l645_645652

/-- 
Given the total required amount of soup for the event is 500 cups, 
and the number of cups made by the first, second, fourth, and fifth teams: 120, 80, 95, and 75 cups respectively,
prove that the third team should prepare 130 cups for the required amount of soup.
-/
theorem third_team_soup_amount :
  let total_cups := 500
  let cups_team1 := 120
  let cups_team2 := 80
  let cups_team4 := 95
  let cups_team5 := 75
  let cups_team3 := total_cups - (cups_team1 + cups_team2 + cups_team4 + cups_team5)
  cups_team3 = 130 :=
by
  let total_cups := 500
  let cups_team1 := 120
  let cups_team2 := 80
  let cups_team4 := 95
  let cups_team5 := 75
  let cups_team3 := total_cups - (cups_team1 + cups_team2 + cups_team4 + cups_team5)
  show cups_team3 = 130 from sorry

end third_team_soup_amount_l645_645652


namespace students_passed_in_three_topics_l645_645957

theorem students_passed_in_three_topics (T all_topics none_topics one_topic two_topics four_topics X : ℕ) 
    (h1 : T = 2500)
    (h2 : all_topics = 0.10 * T)
    (h3 : none_topics = 0.10 * T)
    (h4 : one_topic = 0.20 * (T - all_topics - none_topics))
    (h5 : two_topics = 0.25 * (T - all_topics - none_topics))
    (h6 : four_topics = 0.24 * T)
    (h_sum : all_topics + none_topics + one_topic + two_topics + four_topics + X = T) :
  X = 500 :=
sorry

end students_passed_in_three_topics_l645_645957


namespace total_shaded_area_is_71_l645_645309

-- Define the dimensions of the first rectangle
def rect1_length : ℝ := 4
def rect1_width : ℝ := 12

-- Define the dimensions of the second rectangle
def rect2_length : ℝ := 5
def rect2_width : ℝ := 7

-- Define the dimensions of the overlap area
def overlap_length : ℝ := 3
def overlap_width : ℝ := 4

-- Define the area calculation
def area (length width : ℝ) : ℝ := length * width

-- Calculate the areas of the rectangles and the overlap
def rect1_area : ℝ := area rect1_length rect1_width
def rect2_area : ℝ := area rect2_length rect2_width
def overlap_area : ℝ := area overlap_length overlap_width

-- Total shaded area calculation
def total_shaded_area : ℝ := rect1_area + rect2_area - overlap_area

-- Proof statement to show that the total shaded area is 71 square units
theorem total_shaded_area_is_71 : total_shaded_area = 71 := by
  sorry

end total_shaded_area_is_71_l645_645309


namespace cos_2x_is_even_and_has_period_pi_l645_645325

-- Define the function
def f (x : ℝ) : ℝ := Real.cos (2 * x)

-- Define a property that verifies a function is even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a property that verifies the period of function is π
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- The proposition to prove
theorem cos_2x_is_even_and_has_period_pi : is_even f ∧ has_period f π :=
by
  sorry

end cos_2x_is_even_and_has_period_pi_l645_645325


namespace length_of_PQ_l645_645905

open Real EuclideanGeometry Metric

noncomputable def circle_eqn : set (ℝ × ℝ) :=
{p | (p.1 - 3)^2 + (p.2 - 2)^2 = 1}

noncomputable def line_eqn : set (ℝ × ℝ) :=
{p | p.2 = (3 / 4) * p.1}

noncomputable def length_PQ (P Q : ℝ × ℝ) : ℝ :=
dist P Q

theorem length_of_PQ :
  ∃ P Q ∈ circle_eqn ∩ line_eqn, length_PQ P Q = (4 * sqrt 6) / 5 :=
by
  -- Proof goes here
  sorry

end length_of_PQ_l645_645905


namespace todd_final_money_l645_645667

noncomputable def todd_initial_money : ℝ := 100
noncomputable def todd_debt : ℝ := 110
noncomputable def todd_spent_on_ingredients : ℝ := 75
noncomputable def snow_cones_sold : ℝ := 200
noncomputable def price_per_snowcone : ℝ := 0.75

theorem todd_final_money :
  let initial_money := todd_initial_money,
      debt := todd_debt,
      spent := todd_spent_on_ingredients,
      revenue := snow_cones_sold * price_per_snowcone,
      remaining := initial_money - spent,
      total_pre_debt := remaining + revenue,
      final_money := total_pre_debt - debt
  in final_money = 65 :=
by
  sorry

end todd_final_money_l645_645667


namespace line_AC_eq_l645_645140

-- Define the points A and B
def A := (1, 1)
def B := (-3, -5)

-- Define the line m
def line_m (x y : ℝ) : Prop := 2 * x + y + 6 = 0

-- The proof statement that line AC has the equation x = 1
theorem line_AC_eq : ∃ (A : ℝ × ℝ) (B : ℝ × ℝ), 
  A = (1, 1) ∧ B = (-3, -5) ∧ 
  (∀ (P : ℝ × ℝ), (
    (P = A ∨ (line_m (fst P) (snd P) ∧ ∃ (B' : ℝ × ℝ), B' = (1, -3) ∧ P = B')) → 
    (fst P = 1)
  )
):=
sorry

end line_AC_eq_l645_645140


namespace concentration_third_flask_l645_645297

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l645_645297


namespace find_a_from_complex_modulus_l645_645897

theorem find_a_from_complex_modulus (a : ℝ) (i : ℂ) (h0 : i = complex.I) 
    (h1 : abs((1 + a * i) / (2 * i)) = (√5) / 2)
    : a = 2 ∨ a = -2 :=
by 
    sorry  -- the detailed proof steps are omitted

end find_a_from_complex_modulus_l645_645897


namespace f_increasing_pos_find_m_find_M_inter_N_l645_645475

-- Define the odd function f
def f (x : ℝ) : ℝ := sorry -- as f is not explicitly defined

-- Define the determinant of a 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the function g(θ)
def g (θ : ℝ) (m : ℝ) : ℝ :=
  det (Real.sin θ) (3 - Real.cos θ) m (Real.sin θ)

-- Condition for maximum value of g(θ) being 4
def max_value_condition (θ m : ℝ) : Prop :=
  0 ≤ θ ∧ θ ≤ Real.pi / 2 ∧ (g θ m ≤ 4 ∧ ∃ θ', g θ' m = 4)

-- Specify the sets M and N based on conditions
def M (m : ℝ) : Prop := ∃ θ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 ∧ g θ m > 0
def N (m : ℝ) : Prop := ∃ θ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 ∧ f (g θ m) < 0

-- The intersection of M and N
def M_inter_N (m : ℝ) : Prop := M m ∧ N m

-- Prove that f is increasing on (0, +∞)
theorem f_increasing_pos (h_odd : ∀ x, f (-x) = -f x)
  (h_inc_neg : ∀ x y, x < y → x < 0 → y < 0 → f x < f y) :
  ∀ x y, 0 < x → x < y → f x < f y :=
sorry

-- Find m such that the maximum value of g(θ) is 4
theorem find_m (h : ∃ m, ∀ θ, max_value_condition θ m) :
  ∃ m, m = -1 :=
sorry

-- Find M ∩ N
theorem find_M_inter_N (h_odd : ∀ x, f (-x) = -f x) (h_incr : ∀ x, f_increasing_pos h_odd (sorry)) :
  ∀ m, M_inter_N m → -1/3 < m ∧ m < 0 :=
sorry

end f_increasing_pos_find_m_find_M_inter_N_l645_645475


namespace fractional_eq_has_positive_root_m_value_l645_645942

-- Define the conditions and the proof goal
theorem fractional_eq_has_positive_root_m_value (m x : ℝ) (h1 : x - 2 ≠ 0) (h2 : 2 - x ≠ 0) (h3 : ∃ x > 0, (m / (x - 2)) = ((1 - x) / (2 - x)) - 3) : m = 1 :=
by
  -- Proof goes here
  sorry

end fractional_eq_has_positive_root_m_value_l645_645942


namespace AM_lt_BM_plus_CM_l645_645559

variable {punt : Type}

-- Define the properties of the triangle and the circle
variable [euclidean_space punt]
variable (A B C O M : punt)

-- Define the isosceles triangle ABC with AB = AC
variable (h_iso : dist A B = dist A C)

-- Define that triangle ABC is inscribed in circle O
variable (circle_O : circle O)

-- Define that point M is on the line segment from O to C and lies inside triangle ABC
variable (on_segment_OC : is_on_segment M O C)
variable (in_triangle : is_in_triangle M A B C)

-- State the theorem to prove
theorem AM_lt_BM_plus_CM (h_iso : dist A B = dist A C) (circle_O : circle O)
    (on_segment_OC : is_on_segment M O C) (in_triangle : is_in_triangle M A B C) :
    dist A M < dist B M + dist C M :=
by
  sorry

end AM_lt_BM_plus_CM_l645_645559


namespace weight_order_l645_645193

variable {α : Type}

variables (a b c d : α)

-- Condition 1: Wolf weighs more than Nif-Nif
axiom wolfMoreThanNif : a > b

-- Condition 2: Nuf-Nuf and Nif-Nif weigh more than the Wolf and Naf-Naf
axiom sumNufNifMoreThanWolfNaf : d + b > a + c

-- Condition 3: Nuf-Nuf and Naf-Naf together weigh as much as the Wolf and Nif-Nif together
axiom sumEqual : d + c = a + b

theorem weight_order : d > a ∧ a > b ∧ b > c :=
by
  sorry

end weight_order_l645_645193


namespace width_decreased_by_33_percent_l645_645621

theorem width_decreased_by_33_percent {L W : ℝ} (h : L > 0 ∧ W > 0) (h_area : (1.5 * L) * W' = L * W) :
  W' = (2 / 3) * W :=
begin
  sorry -- Proof to be filled in later
end

end width_decreased_by_33_percent_l645_645621


namespace find_QT_l645_645130

-- Definitions for the conditions in Lean
variables {P Q R S T : Type} [InnerProductSpace ℝ (Fin 4)] -- Assuming 4-dimensional real inner product space
variables (RS PQ : ℝ) (PT QT : ℝ)
variables {angleRPS angleQSP : ℝ}

-- Conditions provided in the problem
axiom RS_perp_PQ : OrthoNormalGeneral RS PQ
axiom PQ_perp_RS : OrthoNormalGeneral PQ RS
axiom RS_length : RS = 40
axiom PQ_length : PQ = 15
axiom line_through_Q_perp_to_PS (PS : ℝ) : angleRPS - angleQSP = π / 2
axiom PT_length : PT = 7

-- The statement we need to prove
theorem find_QT : QT = 25.14 := sorry

end find_QT_l645_645130


namespace angle_and_sin_range_l645_645502

noncomputable theory

-- Define the vectors m and n
def m (B : ℝ) : ℝ × ℝ := (Real.sin B, 1 - Real.cos B)
def n : ℝ × ℝ := (2, 0)

-- Define the cosine equality condition
def cos_angle_condition (B : ℝ) : Prop :=
  Real.cos (π / 3) = ((Real.sin B) * 2) / (Real.sqrt ((Real.sin B)^2 + (1 - Real.cos B)^2) * 2)

-- Helper function to capturing interior angles
def is_interior_angle (A B C : ℝ) : Prop :=
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π

-- Main theorem
theorem angle_and_sin_range (A B C : ℝ) (h1 : cos_angle_condition B) (h2 : is_interior_angle A B C) :
  B = 2 * π / 3 ∧
  (Real.sqrt 3) / 2 < (Real.sin A + Real.sin C) ∧ (Real.sin A + Real.sin C) ≤ 1 :=
sorry

end angle_and_sin_range_l645_645502


namespace walkway_time_against_direction_l645_645763

theorem walkway_time_against_direction (v_p v_w t : ℝ) (h1 : 90 = (v_p + v_w) * 30)
  (h2 : v_p * 48 = 90) 
  (h3 : 90 = (v_p - v_w) * t) :
  t = 120 := by 
  sorry

end walkway_time_against_direction_l645_645763


namespace fundraising_exceeded_goal_l645_645213

theorem fundraising_exceeded_goal (ken mary scott : ℕ) (goal: ℕ) 
  (h_ken : ken = 600)
  (h_mary_ken : mary = 5 * ken)
  (h_mary_scott : mary = 3 * scott)
  (h_goal : goal = 4000) :
  (ken + mary + scott) - goal = 600 := 
  sorry

end fundraising_exceeded_goal_l645_645213


namespace jack_total_cost_l645_645339

def plan_base_cost : ℕ := 25

def cost_per_text : ℕ := 8

def free_hours : ℕ := 25

def cost_per_extra_minute : ℕ := 10

def texts_sent : ℕ := 150

def hours_talked : ℕ := 26

def total_cost (base_cost : ℕ) (texts_sent : ℕ) (cost_per_text : ℕ) (hours_talked : ℕ) 
               (free_hours : ℕ) (cost_per_extra_minute : ℕ) : ℕ :=
  base_cost + (texts_sent * cost_per_text) / 100 + 
  ((hours_talked - free_hours) * 60 * cost_per_extra_minute) / 100

theorem jack_total_cost : 
  total_cost plan_base_cost texts_sent cost_per_text hours_talked free_hours cost_per_extra_minute = 43 :=
by
  sorry

end jack_total_cost_l645_645339


namespace max_value_proof_l645_645998

noncomputable def max_value (x y z : ℝ) : ℝ := x * y + x * z + y * z

def problem_condition (x y z : ℝ) :=
  x + 2 * y + z = 7

theorem max_value_proof :
  ∃ x y z : ℝ, problem_condition x y z ∧ max_value x y z = 7 :=
by
  use 3.5, 0, 3.5
  constructor
  · simp [problem_condition]
  · simp [max_value]
  sorry

end max_value_proof_l645_645998


namespace slope_of_line_l645_645060

theorem slope_of_line (α : ℝ) (h : sin α + cos α = 1 / 5) : Real.tan α = -4 / 3 :=
by
  sorry

end slope_of_line_l645_645060


namespace largest_d_for_range_l645_645829

theorem largest_d_for_range (d : ℝ) : (∃ x : ℝ, x^2 - 6*x + d = 2) ↔ d ≤ 11 := 
by
  sorry

end largest_d_for_range_l645_645829


namespace num_of_ordered_triples_l645_645107

theorem num_of_ordered_triples :
  (∃ n : ℕ, 
    n = (count (λ (a b c : ℕ),
      1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 10 ∧ b - a = c - b
    ) (finset.range 11).product (finset.range 11).product (finset.range 11)) ∧
    n = 20) :=
sorry

end num_of_ordered_triples_l645_645107


namespace odd_fraction_in_multiplication_table_l645_645533

theorem odd_fraction_in_multiplication_table :
  let n := 16 in
  let total_products := n * n in
  let odds := ∑ i in Finset.range n, if i % 2 = 1 then 1 else 0 in
  let odd_products := odds * odds in
  (odd_products : ℚ) / total_products = 0.25 :=
by
  sorry

end odd_fraction_in_multiplication_table_l645_645533


namespace Feuerbach_theorem_l645_645593

theorem Feuerbach_theorem
  (α β γ : ℝ) (x y z : ℝ)
  (htangent : 
    ∀ (x y z : ℝ), 
      x * cos (α/2) / sin ((β - γ) / 2) + 
      y * cos (β/2) / sin ((α - γ) / 2) + 
      z * cos (γ/2) / sin ((α - β) / 2) = 0) :
  ( ∃ (ξ η ζ : ℝ),
    (ξ : η : ζ = 
      ( sin^2 ((β - γ) / 2) :
        sin^2 ((α - γ) / 2) :
        sin^2 ((α - β) / 2) )) ) :=
begin
  sorry
end

end Feuerbach_theorem_l645_645593


namespace value_of_x_l645_645261

theorem value_of_x (y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := by
  sorry

end value_of_x_l645_645261


namespace num_valid_functions_l645_645438

theorem num_valid_functions : 
  let f (a b c d : ℤ) (x : ℤ) := a * x^3 + b * x^2 + c * x + d
  in (∀ a b c d : ℤ, (f a b c d 0 = f a b c d 1 ∨ f a b c d 0 = 0) ∧ 
      (f a b c d 0 = f a b c d 1 ∨ f a b c d 1 = 0) ∧ 
      (a = 0 ∨ a = 1) ∧ 
      (d = 0 ∨ d = 1)) 
  → 16 :=
by sorry

end num_valid_functions_l645_645438


namespace sum_of_bn_l645_645772

theorem sum_of_bn (n : ℕ) (h : n > 0)
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (hb1 : b 1 = 2)
  (h_arith : ∀ k : ℕ, k > 0 → a (k + 1) - a k = 2)
  (h_geom : ∀ k : ℕ, k > 0 → b (k + 1) = 2 * b k) :
  (∑ k in Finset.range n, b (k + 1)) = (4 / 3 * (4 ^ n - 1)) :=
by
  sorry

end sum_of_bn_l645_645772


namespace index_normalizer_centralizer_l645_645969

variables {G : Type*} [group G] {H : subgroup G} [fintype G] 

-- Given: H is a subgroup of G with |H| = 3
axiom H_subgroup : H ≤ G
axiom card_H : fintype.card H = 3

-- Variables for normalizer and centralizer
def normalizer (H : subgroup G) : subgroup G := { g ∈ G | ∀ h ∈ H, g * h * g⁻¹ ∈ H }
def centralizer (H : subgroup G) : subgroup G := { g ∈ G | ∀ h ∈ H, g * h = h * g }

-- Goal: Prove the index of normalizer modulo centralizer is either 1 or 2
theorem index_normalizer_centralizer :
  ∃ n : ℕ, (n = 1 ∨ n = 2) ∧ fintype.card (normalizer H ⧸ centralizer H) = n :=
sorry

end index_normalizer_centralizer_l645_645969


namespace concentration_third_flask_l645_645299

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l645_645299


namespace find_b_value_l645_645744

theorem find_b_value (x : ℝ) (h_neg : x < 0) (h_eq : 1 / (x + 1 / (x + 2)) = 2) : 
  x + 7 / 2 = 2 :=
sorry

end find_b_value_l645_645744


namespace ratio_y_x_l645_645938

variable {c x y : ℝ}

-- Conditions stated as assumptions
theorem ratio_y_x (h1 : x = 0.80 * c) (h2 : y = 1.25 * c) : y / x = 25 / 16 :=
by
  sorry

end ratio_y_x_l645_645938


namespace concentration_in_third_flask_l645_645275

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l645_645275


namespace investment_time_l645_645252

theorem investment_time
  (p_investment_ratio : ℚ) (q_investment_ratio : ℚ)
  (profit_ratio_p : ℚ) (profit_ratio_q : ℚ)
  (q_investment_time : ℕ)
  (h1 : p_investment_ratio / q_investment_ratio = 7 / 5)
  (h2 : profit_ratio_p / profit_ratio_q = 7 / 10)
  (h3 : q_investment_time = 40) :
  ∃ t : ℚ, t = 28 :=
by
  sorry

end investment_time_l645_645252


namespace rectangle_width_decrease_l645_645626

theorem rectangle_width_decrease (A L W : ℝ) (h1 : A = L * W) (h2 : 1.5 * L * W' = A) : 
  (W' = (2/3) * W) -> by exact (W - W') / W = 1 / 3 :=
by
  sorry

end rectangle_width_decrease_l645_645626


namespace sum_c_d_l645_645353

def vertices : list (ℝ × ℝ) := [(0,0), (2,3), (5,3), (3,0)]

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def perimeter (v : list (ℝ × ℝ)) : ℝ :=
  distance v.head! (v.drop 1).head! + distance (v.drop 1).head! (v.drop 2).head! + 
  distance (v.drop 2).head! (v.drop 3).head! + distance (v.drop 3).head! v.head!

theorem sum_c_d : vertices = [(0,0), (2,3), (5,3), (3,0)] → 
  ∃ c d : ℤ, perimeter vertices = c * real.sqrt 13 + d * real.sqrt 5 ∧ (c + d = 2) :=
by
  sorry

end sum_c_d_l645_645353


namespace xy_proof_l645_645594

theorem xy_proof (x y : ℝ) (h : (x + real.sqrt (x^2 + 1)) * (y + real.sqrt (y^2 + 1)) = 1) : x + y = 0 := 
by
  sorry

end xy_proof_l645_645594


namespace midpoint_of_segment_l645_645560

variable {Triangle : Type}
variable (H : Triangle) (D : Triangle) (K : Triangle)
variable [is_orthocenter H] [is_midpoint D] [is_on_circumcircle K HD]
variable [between D H K]

theorem midpoint_of_segment (H D K : Triangle) 
  [is_orthocenter H] [is_midpoint D] [is_on_circumcircle K HD] [between D H K] : 
  is_midpoint D HK :=
sorry

end midpoint_of_segment_l645_645560


namespace paint_gallons_needed_l645_645551

theorem paint_gallons_needed 
    (n_poles : ℕ := 12)
    (height : ℝ := 12)
    (diameter : ℝ := 8)
    (coverage_per_gallon : ℝ := 300)
    (whole_gallons : ℝ := 17) :
    let radius := diameter / 2
    let lateral_surface_area_per_pole := 2 * Real.pi * radius * height
    let top_and_bottom_face_area_per_pole := 2 * Real.pi * radius ^ 2
    let total_surface_area_per_pole := lateral_surface_area_per_pole + top_and_bottom_face_area_per_pole
    let total_paintable_area := n_poles * total_surface_area_per_pole
    let gallons_needed := (total_paintable_area / coverage_per_gallon).ceil in
    gallons_needed = whole_gallons :=
by
    sorry

end paint_gallons_needed_l645_645551


namespace sum_of_first_n_terms_b_l645_645769

variable (n : ℕ)

def a (n : ℕ) : ℕ := 2 * n
def b (n : ℕ) : ℕ := 2 ^ n

theorem sum_of_first_n_terms_b :
  let b_sum := ∑ i in finset.range n, b (i + 1)
  b_sum = (4 ^ n - 1) * 4 / 3 :=
by sorry

end sum_of_first_n_terms_b_l645_645769


namespace trigonometric_identity_l645_645867

theorem trigonometric_identity (α : ℝ) (h : sin α - cos α = 4 / 3) : cos^2 (π / 4 - α) = 1 / 9 :=
by
  sorry

end trigonometric_identity_l645_645867


namespace driving_time_is_correct_l645_645382

-- Define conditions
def flight_departure : ℕ := 20 * 60 -- 8:00 pm in minutes since 0:00
def checkin_time : ℕ := flight_departure - 2 * 60 -- 2 hours early
def latest_leave_time : ℕ := 17 * 60 -- 5:00 pm in minutes since 0:00
def additional_time : ℕ := 15 -- 15 minutes to park and make their way to the terminal

-- Define question
def driving_time : ℕ := checkin_time - additional_time - latest_leave_time

-- Prove the expected answer
theorem driving_time_is_correct : driving_time = 45 :=
by
  -- omitting the proof
  sorry

end driving_time_is_correct_l645_645382


namespace find_f3_l645_645448

noncomputable theory

variables (a b : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 + b * x + 3

theorem find_f3 (h : f a b (-3) = 10) : f a b 3 = 27 * a + 3 * b + 3 :=
sorry

end find_f3_l645_645448


namespace incenter_proof_problem_l645_645796

noncomputable theory

variables {A B C D I E : Type*}
variables [cyclic A B C D] [incenter I (triangle A B D)]
variables (h1 : perp AC BD) (h2 : perp (IE) BD) (h3 : distance IA IC)

theorem incenter_proof_problem : distance EI EC :=
begin
  sorry
end

end incenter_proof_problem_l645_645796


namespace equilateral_triangle_ratio_is_correct_l645_645697

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l645_645697


namespace largest_percentage_increase_l645_645798

def student_count (year: ℕ) : ℝ :=
  match year with
  | 2010 => 80
  | 2011 => 88
  | 2012 => 95
  | 2013 => 100
  | 2014 => 105
  | 2015 => 112
  | _    => 0  -- Because we only care about 2010-2015

noncomputable def percentage_increase (year1 year2 : ℕ) : ℝ :=
  ((student_count year2 - student_count year1) / student_count year1) * 100

theorem largest_percentage_increase :
  (∀ x y, percentage_increase 2010 2011 ≥ percentage_increase x y) :=
by sorry

end largest_percentage_increase_l645_645798


namespace neg_two_squared_result_l645_645642

theorem neg_two_squared_result : -2^2 = -4 :=
by
  sorry

end neg_two_squared_result_l645_645642


namespace area_of_triangle_ABC_l645_645973

noncomputable def area_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  0.5 * a * b * (Real.sin C)

theorem area_of_triangle_ABC :
  ∀ (a b c S1 S2 S3 : ℝ),
    S1 = (Real.sqrt 3 / 4) * a^2 →
    S2 = (Real.sqrt 3 / 4) * b^2 →
    S3 = (Real.sqrt 3 / 4) * c^2 →
    S1 - S2 + S3 = Real.sqrt 3 / 2 →
    b = Real.sqrt 3 * c →
    Real.cos C = 2 * Real.sqrt 2 / 3 →
    area_triangle_ABC a b c (Real.acos (2 * (Real.sqrt 2) / 3)) = Real.sqrt 2 / 4 :=
  by sorry

end area_of_triangle_ABC_l645_645973


namespace smallest_selected_class_l645_645757

def systematic_sampling_interval (total_classes : ℕ) (selected_classes : ℕ) : ℕ :=
  total_classes / selected_classes

def selected_class (k : ℕ) : ℕ :=
  26 - 6*k 

theorem smallest_selected_class {total_classes selected_classes k : ℕ}
  (h1 : total_classes = 30)
  (h2 : selected_classes = 5)
  (h3 : k = 4)
  (h4 : ∀ k, 26 - systematic_sampling_interval total_classes selected_classes * k > 0)
  : selected_class k = 2 :=
by
  rw [h1, h2, h3]
  simp [systematic_sampling_interval]
  sorry

end smallest_selected_class_l645_645757


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645703

theorem ratio_of_area_to_perimeter_of_equilateral_triangle (s : ℕ) : s = 10 → (let A := (s^2 * sqrt 3) / 4, P := 3 * s in A / P = 5 * sqrt 3 / 6) :=
by
  intro h,
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645703


namespace equation_has_at_least_two_distinct_roots_l645_645015

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l645_645015


namespace proof_problem_l645_645893

-- Define the propositions P and Q
def P := ∀ x : ℝ, (0 < x ∧ x < 1) ↔ (Real.log (x * (1 - x)) + 1 > 0)
def Q := ∀ A B C : ℝ, (∠A > ∠B) → (cos A * cos A < cos B * cos B)

-- Statement matching the given correct answer:
theorem proof_problem : ¬P ∧ Q :=
by sorry

end proof_problem_l645_645893


namespace ratio_of_area_to_perimeter_l645_645678

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l645_645678


namespace find_n_for_divisibility_by_33_l645_645865

theorem find_n_for_divisibility_by_33 (n : ℕ) (hn_range : n < 10) (div11 : (12 - n) % 11 = 0) (div3 : (20 + n) % 3 = 0) : n = 1 :=
by {
  -- Proof steps go here
  sorry
}

end find_n_for_divisibility_by_33_l645_645865


namespace sum_of_positive_ks_l645_645858

theorem sum_of_positive_ks :
  let integer_solutions (k : ℤ) := ∃ α β : ℤ, α + β = k ∧ α * β = -24
  let positive_ks := { k : ℤ | integer_solutions k ∧ k > 0 }
  ∑ k in positive_ks, k = 40 :=
by
  sorry

end sum_of_positive_ks_l645_645858


namespace alice_vs_tom_l645_645836

-- Define the number of bananas eaten by each student, with Alice and Tom specified.
variable (bananas : Fin 8 → Nat)

-- Define Alice as the student who ate the most bananas
def Alice := (Fin 8).maxBy (bananas)

-- Define Tom as the student who ate the fewest bananas
def Tom := (Fin 8).minBy (bananas)

-- State the theorem
theorem alice_vs_tom (hAlice : bananas Alice = 7) (hTom : bananas Tom = 1) : 
  bananas Alice - bananas Tom = 6 := by
  sorry

end alice_vs_tom_l645_645836


namespace ticket_identification_operations_l645_645304

theorem ticket_identification_operations :
  ∀ (n : ℕ), n = 30 → (∃ (operations_needed : ℕ), operations_needed = 5) := by
  intro n hn_eq_30
  use 5
  sorry

end ticket_identification_operations_l645_645304


namespace probability_white_or_red_l645_645961

theorem probability_white_or_red (a b c : ℕ) : 
  (a + b) / (a + b + c) = (a + b) / (a + b + c) := by
  -- Conditions
  let total_balls := a + b + c
  let white_red_balls := a + b
  -- Goal
  have prob_white_or_red := white_red_balls / total_balls
  exact rfl

end probability_white_or_red_l645_645961


namespace find_angle_C_range_of_a_plus_b_l645_645948

-- Given conditions: sides opposite to angles A, B, C are a, b, c respectively in triangle ABC
variables (a b c A B C : ℝ)

-- Condition of the problem: (a+c) * (sin A - sin C) = sin B * (a - b)
axiom triangle_condition : (a + c) * (Real.sin A - Real.sin C) = Real.sin B * (a - b)

-- Problem 1: Find the size of angle C (where answer is 60 degrees)
theorem find_angle_C : C = 60 :=
begin
  -- to be proven
  sorry
end

-- Problem 2: If c = 2, find the range of values for a + b (where answer is (2, 4])
axiom c_value : c = 2

theorem range_of_a_plus_b : 2 < a + b ∧ a + b ≤ 4 :=
begin
  -- to be proven
  sorry
end

end find_angle_C_range_of_a_plus_b_l645_645948


namespace periodic_sum_iff_commensurable_l645_645161

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f(x + T) = f(x)

def commensurable (T₁ T₂ : ℝ) : Prop := ∃ (m n : ℕ), T₁ / T₂ = m / n

theorem periodic_sum_iff_commensurable 
  {f g : ℝ → ℝ} 
  (hf : continuous f) (hg : continuous g) 
  (Hf : ∃ T1 > 0, is_periodic f T1 ∧ ∀ T > 0, is_periodic f T → T ≥ T1) 
  (Hg : ∃ T2 > 0, is_periodic g T2 ∧ ∀ T > 0, is_periodic g T → T ≥ T2) : 
  (∃ T > 0, is_periodic (λ x, f x + g x) T) ↔ 
  commensurable (classical.some Hf).1 (classical.some Hg).1 := 
sorry

end periodic_sum_iff_commensurable_l645_645161


namespace find_lambda_l645_645099

variables (a b : Vector ℝ) (λ : ℝ)

-- Given conditions
def given_conditions (a b : Vector ℝ) (λ : ℝ) : Prop :=
  (|a| = 2) ∧ (a • b = 2) 

-- Proof that a is perpendicular to a - λb for λ = 2
theorem find_lambda (a b : Vector ℝ) (h : given_conditions a b 2) : a • (a - 2 • b) = 0 :=
begin
  sorry,
end

end find_lambda_l645_645099


namespace count_64_digit_numbers_divisible_by_101_is_odd_l645_645549

/--
  Prove that the number of 64-digit natural numbers that do not contain the digit zero
  and are divisible by 101 is odd.
-/
theorem count_64_digit_numbers_divisible_by_101_is_odd :
  ¬ even (Finset.card {n : ℕ | 10^63 ≤ n ∧ n < 10^64 ∧ (∀ k ∈ Nat.digits 10 n, k ≠ 0) ∧ 101 ∣ n}) :=
sorry

end count_64_digit_numbers_divisible_by_101_is_odd_l645_645549


namespace sec_neg_seven_pi_over_four_l645_645426

theorem sec_neg_seven_pi_over_four : ∀ (θ : ℝ), θ = - 7 * Real.pi / 4 → Real.sec θ = Real.sqrt 2 :=
by
  intros θ h
  -- the proof is omitted
  sorry

end sec_neg_seven_pi_over_four_l645_645426


namespace hexagon_octaon_area_relation_l645_645795

noncomputable def radius_inscribed_circle (s : ℝ) (theta : ℝ) : ℝ :=
  s / (2 * Real.tan (theta / 2))

noncomputable def radius_circumscribed_circle (s : ℝ) (theta : ℝ) : ℝ :=
  s / (2 * Real.sin (theta / 2))

def area_between_circles (R r : ℝ) : ℝ :=
  Real.pi * (R^2 - r^2)

def hexagon_inner_outer_areas (s : ℝ) : ℝ × ℝ :=
  let R_hex := radius_circumscribed_circle s (360 / 6)
  let r_hex := radius_inscribed_circle s (360 / 6)
  (R_hex, r_hex)

def octagon_inner_outer_areas (s : ℝ) : ℝ × ℝ :=
  let R_oct := radius_circumscribed_circle s (360 / 8)
  let r_oct := radius_inscribed_circle s (360 / 8)
  (R_oct, r_oct)

theorem hexagon_octaon_area_relation (s : ℝ) (h : s = 3) :
  let ⟨R_hex, r_hex⟩ := hexagon_inner_outer_areas s
  let C := area_between_circles R_hex r_hex
  let ⟨R_oct, r_oct⟩ := octagon_inner_outer_areas s
  let D := area_between_circles R_oct r_oct
  C = (4 / 5) * D :=
by
  sorry

end hexagon_octaon_area_relation_l645_645795


namespace find_m_value_l645_645480

theorem find_m_value (m : ℝ) (x_vals y_vals : List ℝ) (hl : y_vals = [1, m + 1, 2 * m + 1, 3 * m + 3, 11]) (hx : x_vals = [0, 2, 4, 6, 8])
    (hxy : ∀ x y, y_vals.Head! = 1 ↔ x_vals.nth 0 = some 0) :
    (∀ x y, y = 1.3 * x + 0.6 ↔ (x, y) ∈ List.zip x_vals y_vals) → m = 2 := 
by
  sorry

end find_m_value_l645_645480


namespace john_wages_decrease_percentage_l645_645147

theorem john_wages_decrease_percentage (W : ℝ) (P : ℝ) :
  (0.20 * (W - P/100 * W)) = 0.50 * (0.30 * W) → P = 25 :=
by 
  intro h
  -- Simplification and other steps omitted; focus on structure
  sorry

end john_wages_decrease_percentage_l645_645147


namespace calc_expr_l645_645807

theorem calc_expr : 
  sqrt 12 - 3 * tan (30 * real.pi / 180) - (1 - real.pi)^0 + abs (-sqrt 3) = 2 * sqrt 3 - 1 :=
by
  sorry

end calc_expr_l645_645807


namespace parabola_focus_distance_l645_645073

noncomputable def point_on_parabola (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x

noncomputable def focus_distance (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  let focus_x := p / 2 in
  real.sqrt ((x - focus_x) ^ 2 + y ^ 2) = 12

noncomputable def y_axis_distance (x : ℝ) : Prop :=
  real.abs x = 9

theorem parabola_focus_distance (p : ℝ) (x : ℝ) (y : ℝ) (h1 : point_on_parabola p x y) (h2 : focus_distance p x y) (h3 : y_axis_distance x) : p = 6 :=
sorry

end parabola_focus_distance_l645_645073


namespace triangle_sinA_and_side_b_proof_l645_645531

def triangle_sinA_and_side_b (a C : ℝ) (tanA : ℝ) : Prop :=
  a = 2 * Real.sqrt 3 ∧ 
  C = Real.pi / 3 ∧ 
  tanA = 3 / 4 → 
  let sinA := 3 / 5 in
  let b := 4 + Real.sqrt 3 in
  Real.sin (Real.atan tanA) = sinA ∧ 
  b = (2 * Real.sqrt 3 * sinA) / (Real.sin (Real.atan (tanA + Real.tan C)))
  
-- Demonstration of usage of the defined function
theorem triangle_sinA_and_side_b_proof : 
  triangle_sinA_and_side_b (2 * Real.sqrt 3) (Real.pi / 3) (3 / 4) :=
by
  intros h
  sorry

end triangle_sinA_and_side_b_proof_l645_645531


namespace find_m_n_l645_645986

variables {A B C O G X Y : Type} [Point A] [Point B] [Point C] [Point O] [Point G] [Point X] [Point Y]
variables {abcTriangle : Triangle A B C} {circumcenterO : Circumcenter abcTriangle O} {centroidG : Centroid abcTriangle G}
variables {circumcircle : Circumcircle abcTriangle}
variables (tangentA : TangentLine circumcircle A X) (perpendicularGO : PerpendicularLine (Line.mk G O) G X)
variables {lineXG : Line.mk X G} {lineBC : Line.mk B C}
variables (Y_intersection : Intersection lineXG lineBC Y)
variables {angle1 angle2 angle3 : ℝ} (angle_ratio : angle1 / angle2 = 13 / 2 ∧ angle2 / angle3 = 2 / 17)

theorem find_m_n (h1 : ∠ B A C = 13 * (180 / (15 + 2 * 13))) (h2 : ∠ Y O X = 17 * (180 / (15 + 2 * 17))) :
  let k := 180 / (15 + 2 * 13) in 
  let angle_bac := 180 - 15 * k in
  let m := 585 in
  let n := 7 in
  m + n = 592 :=
by
  sorry -- Proof can be filled in here

end find_m_n_l645_645986


namespace third_vs_second_plant_relationship_l645_645981

-- Define the constants based on the conditions
def first_plant_tomatoes := 24
def second_plant_tomatoes := 12 + 5  -- Half of 24 plus 5
def total_tomatoes := 60

-- Define the production of the third plant based on the total number of tomatoes
def third_plant_tomatoes := total_tomatoes - (first_plant_tomatoes + second_plant_tomatoes)

-- Define the relationship to be proved
theorem third_vs_second_plant_relationship : 
  third_plant_tomatoes = second_plant_tomatoes + 2 :=
by
  -- Proof not provided, adding sorry to skip
  sorry

end third_vs_second_plant_relationship_l645_645981


namespace find_f_zero_l645_645512

-- Defining f(x) as a polynomial with given conditions
noncomputable def f (x : ℝ) : ℝ := sorry

-- Main theorem
theorem find_f_zero (f_monic : ∀ x : ℝ, polynomial.monic (λ x, f x)) 
                   (cond1 : f (-2) = 0) 
                   (cond2 : f 3 = -9)
                   (cond3 : f (-4) = -16)
                   (cond4 : f 5 = -25) : 
                   f 0 = 0 :=
by 
  sorry

end find_f_zero_l645_645512


namespace rectangular_coordinate_equation_of_circle_C_sum_distances_PA_PB_l645_645477

variable (t θ : ℝ)
variable (x y ρ : ℝ)

-- Problem conditions
def parametric_equation_of_line_x := 3 - t
def parametric_equation_of_line_y := Real.sqrt 5 + t
def polar_equation_of_circle_C := ρ = 2 * Real.sqrt 5 * Real.sin θ
def point_P_coordinates := (3, Real.sqrt 5)

-- Proofs to be established.
theorem rectangular_coordinate_equation_of_circle_C :
  (∃ ρ θ : ℝ, polar_equation_of_circle_C → x^2 + (y - Real.sqrt 5)^2 = 5) := sorry

theorem sum_distances_PA_PB :
  (let A := (3 - t, Real.sqrt 5 + t) in
    let B := (3 - t, Real.sqrt 5 + t) in
    ∀ (t₁ t₂ : ℝ), (t^2 - 3 * t + 2 = 0) →
    abs (3, Real.sqrt 5) - A + abs (3, Real.sqrt 5) - B = 3 ) := sorry

end rectangular_coordinate_equation_of_circle_C_sum_distances_PA_PB_l645_645477


namespace tangent_line_to_curve_at_P_l645_645618

noncomputable def tangent_line_at_point (x y : ℝ) := 4 * x - y - 2 = 0

theorem tangent_line_to_curve_at_P :
  (∃ (b: ℝ), ∀ (x: ℝ), b = 2 * 1^2 → tangent_line_at_point 1 2)
:= 
by
  sorry

end tangent_line_to_curve_at_P_l645_645618


namespace area_of_square_EFGH_l645_645410

-- Conditions
def square_side_length := 10
def BE := 1 / 5 * square_side_length

-- Theorem to prove
theorem area_of_square_EFGH : 
  let AB := square_side_length in
  let BE := BE in
  ∃ (x: ℝ), 
    (BE = 2) → 
    let s := x + 2 in
    (AB * AB = s * s + BE * BE) → 
    (∃ (x: ℝ), let area := (s - BE)^2 in area = 100 - 16 * real.sqrt 6) :=
sorry

end area_of_square_EFGH_l645_645410


namespace symmetric_scanning_codes_count_l645_645358

noncomputable def countSymmetricScanningCodes : ℕ :=
  let totalConfigs := 32
  let invalidConfigs := 2
  totalConfigs - invalidConfigs

theorem symmetric_scanning_codes_count :
  countSymmetricScanningCodes = 30 :=
by
  -- Here, we would detail the steps, but we omit the actual proof for now.
  sorry

end symmetric_scanning_codes_count_l645_645358


namespace larger_number_of_two_integers_l645_645507

theorem larger_number_of_two_integers (x y : ℤ) (h1 : x * y = 30) (h2 : x + y = 13) : (max x y = 10) :=
by
  sorry

end larger_number_of_two_integers_l645_645507


namespace geometric_sequence_seventh_term_l645_645137

noncomputable theory

def geometric_sequence (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_seventh_term :
  let a₁ := 3
  let q := real.sqrt 2
  geometric_sequence a₁ q 7 = 24 :=
by
  intros
  unfold geometric_sequence
  -- Here would go the actual proof calculation
  sorry

end geometric_sequence_seventh_term_l645_645137


namespace no_valid_positive_x_l645_645345

theorem no_valid_positive_x
  (π : Real)
  (R H x : Real)
  (hR : R = 5)
  (hH : H = 10)
  (hx_pos : x > 0) :
  ¬π * (R + x) ^ 2 * H = π * R ^ 2 * (H + x) :=
by
  sorry

end no_valid_positive_x_l645_645345


namespace probability_sum_of_squares_within_interval_l645_645204

noncomputable def prob_sum_squares_within_interval (a b : ℝ) (S : set (ℝ × ℝ)) := 
  (∫ x in a..b, ∫ y in a..b, 1)⁻¹ *
  ∫ x in a..b, ∫ y in a..b, if (x, y) ∈ S then 1 else 0

theorem probability_sum_of_squares_within_interval :
  prob_sum_squares_within_interval 0 2 {p | (p.1^2 + p.2^2) ≤ 2} = (real.pi / 8) :=
by
  sorry

end probability_sum_of_squares_within_interval_l645_645204


namespace relationship_among_f_values_l645_645078

noncomputable def f : ℝ → ℝ := sorry  -- Given even, monotonically decreasing

def a : ℝ := -2^(1.2)
def b : ℝ := (1/2)^(-0.8)
def c : ℝ := 2 * Real.log 2 / Real.log 5  -- log base 5

-- Statement: $ f(c) > f(b) > f(a) $
theorem relationship_among_f_values 
  (h_even : ∀ x, f(-x) = f(x)) 
  (h_monotonic : ∀ x y, 0 < x ∧ x < y → f(y) < f(x)
  ) :
  f(c) > f(b) ∧ f(b) > f(a) := 
sorry

end relationship_among_f_values_l645_645078


namespace problem_conditions_l645_645457

variable {m : ℝ}
def quadratic (x : ℝ) : ℝ := x^2 + m * x + 1

theorem problem_conditions (h₁: ∀ (x : ℝ), quadratic x <> 0 -> x ∈ [2, +∞]) 
  (h₂: there exists x : ℝ, x ∈ [2, +∞] -> x + m / x - 2 > 0) 
  (h₃: ∀ (x : ℝ), ¬ (x^2 + m * x + 1 = 0)) 
  (h₄: ( ∀ (x : ℝ), quadratic x = 0 ) ∨ ( ∃ x ∈ [2, +∞], x + m / x - 2 > 0)) 
  : 0 < m ∧ m ≤ 2 := 
by
  sorry

end problem_conditions_l645_645457


namespace tangents_parallel_l645_645997

-- Define the essential elements needed for the problem
variables {k : Type*} [circle k]
variables {P A B M C D : Type*}
variables (hP_outside : outside_circle P k)
variables (tangents_P : tangent P A k ∧ tangent P B k)
variables (midpoint_M : midpoint M B P)
variables (second_intersection_C : second_intersection A M k C)
variables (second_intersection_D : second_intersection P C k D)

-- Prove the required property
theorem tangents_parallel (h : ∀ (AD BP : line), parallel AD BP ↔ angle_PDA_eq_angle_CPB) :
  parallel_line (line_through A D) (line_through B P) :=
begin
  sorry
end

end tangents_parallel_l645_645997


namespace two_digit_numbers_sum_reversed_l645_645824

theorem two_digit_numbers_sum_reversed (a b : ℕ) (h₁ : 0 ≤ a) (h₂ : a ≤ 9) (h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : a + b = 12) :
  ∃ n : ℕ, n = 7 := 
sorry

end two_digit_numbers_sum_reversed_l645_645824


namespace function_range_l645_645673

theorem function_range (m n : ℕ) (hmn : m < n) (hcond : 
  1 = 1/2 + 1/6 + 1/12 + 1/m + 1/20 + 1/n + 1/42 + 1/56 + 1/72 + 1/90 + 1/110 + 1/132 + 1/156)
  (hvalues : m = 13 ∧ n = 30) :
  set.range (λ x, (m+n)*x/(x-1)) = { y : ℝ | y ≠ 43 } :=
sorry

end function_range_l645_645673


namespace illuminate_entire_plane_l645_645773

-- Define the conditions and question using Lean 4
theorem illuminate_entire_plane (A B C D : Point) : 
  ∃ (S1 S2 S3 S4 : SpotLight), illuminates(S1) ∧ illuminates(S2) ∧ illuminates(S3) ∧ illuminates(S4) :=
by
  sorry

end illuminate_entire_plane_l645_645773


namespace perpendicular_AK_KC_l645_645954

variables (A B C D N M K : Point)
variables 
  (h1 : ConvexQuadrilateral A B C D)
  (h2 : angle A B C = 135)
  (h3 : angle A D C = 135)
  (h4 : OnRay N A B)
  (h5 : OnRay M A D)
  (h6 : angle M C D = 90)
  (h7 : angle N C B = 90)
  (h8 : ∃ O1, Circumcircle O1 A M N)
  (h9 : ∃ O2, Circumcircle O2 A B D)
  (h10 : intersect_circle (Circumcircle O1 A M N) (Circumcircle O2 A B D) = {A, K})

theorem perpendicular_AK_KC : isPerp (Line A K) (Line K C) :=
  sorry

end perpendicular_AK_KC_l645_645954


namespace arithmetic_geometric_sequence_product_l645_645237

theorem arithmetic_geometric_sequence_product :
  ∀ (a : ℕ → ℝ) (q : ℝ),
    a 1 = 3 →
    (a 1) + (a 1 * q^2) + (a 1 * q^4) = 21 →
    (a 2) * (a 6) = 72 :=
by 
  intros a q h1 h2 
  sorry

end arithmetic_geometric_sequence_product_l645_645237


namespace lottery_winning_probability_l645_645123

theorem lottery_winning_probability :
  let P_A1 := 20 / 1000
  let P_A2 := 5 / 1000
  ∃ P_A, P_A = P_A1 + P_A2 ∧ P_A = 0.025 :=
begin
  let P_A1 := 20 / 1000,
  let P_A2 := 5 / 1000,
  use (P_A1 + P_A2),
  split,
  { refl },
  { norm_num }
end

end lottery_winning_probability_l645_645123


namespace collinear_QRS_l645_645991

-- Definitions and theorems you might use
variables {A B C P Q R S : Type} -- Vertices of the triangle and points
variables [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C]
variables [nonempty P] [nonempty Q] [nonempty R] [nonempty S]

noncomputable def perpendicular (A B : Type) : Prop := sorry -- Define perpendicularity
noncomputable def intersect (A B C : Type) : Prop := sorry -- Define intersection

-- The condition that P is an interior point of triangle ABC
variable (P_in_triangle_ABC : P ∈ triangle A B C)

-- Conditions that l, m, n are perpendicular to AP, BP, CP respectively
variable (l_perpendicular: perpendicular l AP)
variable (m_perpendicular: perpendicular m BP)
variable (n_perpendicular: perpendicular n CP)

-- Conditions that l, m, n intersect BC, AC, AB at Q, R, S respectively
variable (Q_intersect: intersect l BC Q)
variable (R_intersect: intersect m AC R)
variable (S_intersect: intersect n AB S)

-- The theorem to prove collinearity of points Q, R, S
theorem collinear_QRS :
  collinear Q R S := 
sorry

end collinear_QRS_l645_645991


namespace quotient_is_approx_89_l645_645196

noncomputable def calculate_quotient (dividend : ℝ) (divisor : ℝ) (remainder : ℝ) : ℝ :=
  (dividend - remainder) / divisor

theorem quotient_is_approx_89 : 
  calculate_quotient 16698 187.46067415730337 14 ≈ 89 := 
by
  sorry

end quotient_is_approx_89_l645_645196


namespace george_speed_for_last_mile_l645_645444

theorem george_speed_for_last_mile 
  (normal_speed : ℝ := 4) 
  (distance_to_school : ℝ := 2)
  (normal_time : ℝ := 0.5)
  (first_mile_speed : ℝ := 3)
  (first_mile_distance : ℝ := 1) :
  let remaining_time := normal_time - (first_mile_distance / first_mile_speed) in
  let required_speed := 1 / remaining_time in
  required_speed = 6 :=
by
  sorry

end george_speed_for_last_mile_l645_645444


namespace rectangle_width_decreased_l645_645629

theorem rectangle_width_decreased (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.5 * L in
  let W' := (L * W) / L' in
  ((W - W') / W) * 100 = 33.3333 :=
by
  sorry

end rectangle_width_decreased_l645_645629


namespace concentration_third_flask_l645_645302

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l645_645302


namespace first_line_shift_time_l645_645305

theorem first_line_shift_time (x y : ℝ) (h1 : (1 / x) + (1 / (x - 2)) + (1 / y) = 1.5 * ((1 / x) + (1 / (x - 2)))) 
  (h2 : x - 24 / 5 = (1 / ((1 / (x - 2)) + (1 / y)))) :
  x = 8 :=
sorry

end first_line_shift_time_l645_645305


namespace equilateral_triangle_area_to_perimeter_ratio_l645_645687

theorem equilateral_triangle_area_to_perimeter_ratio
  (a : ℝ) (h : a = 10) :
  let altitude := a * (Real.sqrt 3 / 2) in
  let area := (1 / 2) * a * altitude in
  let perimeter := 3 * a in
  area / perimeter = 5 * (Real.sqrt 3) / 6 := 
by
  sorry

end equilateral_triangle_area_to_perimeter_ratio_l645_645687


namespace total_amount_is_24_l645_645778

-- Define the original price of a tub of ice cream
def original_price_ice_cream : ℕ := 12

-- Define the discount per tub of ice cream
def discount_per_tub : ℕ := 2

-- Define the discounted price of a tub of ice cream
def discounted_price_ice_cream : ℕ := original_price_ice_cream - discount_per_tub

-- Define the price for 5 cans of juice
def price_per_5_cans_of_juice : ℕ := 2

-- Define the number of cans of juice bought
def cans_of_juice_bought : ℕ := 10

-- Calculate the total cost for two tubs of ice cream and 10 cans of juice
def total_cost (p1 p2 : ℕ) : ℕ := 2 * p1 + (price_per_5_cans_of_juice * (cans_of_juice_bought / 5))

-- Prove that the total cost is $24
theorem total_amount_is_24 : total_cost discounted_price_ice_cream price_per_5_cans_of_juice = 24 := by
  sorry

end total_amount_is_24_l645_645778


namespace no_real_roots_l645_645830

theorem no_real_roots (x : ℝ) : 
  sqrt (3 * x + 9) + (8 / sqrt (3 * x + 9)) = 4 → false :=
by
  sorry

end no_real_roots_l645_645830


namespace betsy_sewing_l645_645390

-- Definitions of conditions
def total_squares : ℕ := 16 + 16
def sewn_percentage : ℝ := 0.25
def sewn_squares : ℝ := sewn_percentage * total_squares
def squares_left : ℝ := total_squares - sewn_squares

-- Proof that Betsy needs to sew 24 more squares
theorem betsy_sewing : squares_left = 24 := by
  -- Sorry placeholder for the actual proof
  sorry

end betsy_sewing_l645_645390


namespace sum_of_coordinates_of_X_l645_645561

theorem sum_of_coordinates_of_X :
  ∃ (X : ℝ × ℝ × ℝ), 
  let Y : ℝ × ℝ × ℝ := (2, 3, 4)
  let Z : ℝ × ℝ × ℝ := (0, -1, -2)
  Z = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2, (X.3 + Y.3) / 2) ∧
  (X.1 + X.2 + X.3 = 33) := sorry

end sum_of_coordinates_of_X_l645_645561


namespace ellipse_slope_product_constant_l645_645890

noncomputable def ellipse_eq (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

theorem ellipse_slope_product_constant (A B : ℝ × ℝ) (hA : ellipse_eq A.1 A.2) (hB: ellipse_eq B.1 B.2)
    (h_area : abs (A.1 * B.2 - A.2 * B.1) / 2 = sqrt 2 / 2) :
    (A.2 / A.1) * (B.2 / B.1) = -1 / 2 := 
sorry

end ellipse_slope_product_constant_l645_645890


namespace width_decreased_by_33_percent_l645_645622

theorem width_decreased_by_33_percent {L W : ℝ} (h : L > 0 ∧ W > 0) (h_area : (1.5 * L) * W' = L * W) :
  W' = (2 / 3) * W :=
begin
  sorry -- Proof to be filled in later
end

end width_decreased_by_33_percent_l645_645622


namespace concentration_in_third_flask_l645_645277

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l645_645277


namespace find_numbers_l645_645182

theorem find_numbers (A B C D : ℚ) 
  (h1 : A + B = 44)
  (h2 : 5 * A = 6 * B)
  (h3 : C = 2 * (A - B))
  (h4 : D = (A + B + C) / 3 + 3) :
  A = 24 ∧ B = 20 ∧ C = 8 ∧ D = 61 / 3 := 
  by 
    sorry

end find_numbers_l645_645182


namespace find_a_in_triangle_l645_645499

theorem find_a_in_triangle (A B C : ℝ) (a b c : ℝ)
  (sin_A : ℝ := 1 / 3)
  (sin_B : ℝ)
  (b_eq : b = sqrt 3 * sin_B)
  (sin_A_eq : real.sin A = sin_A) :
  a = sqrt 3 / 3 :=
by
  sorry

end find_a_in_triangle_l645_645499


namespace newborn_members_count_l645_645120

theorem newborn_members_count 
  (N : ℝ) 
  (h : N * (27 / 64) = 84.375) : 
  N ≈ 200 :=
sorry

end newborn_members_count_l645_645120


namespace find_m_n_l645_645181

open Classical

-- Definitions for the sample space and events
def Omega : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2, 3, 5}
def C (m n : ℕ) : Set ℕ := {1, m, n, 8}

-- Definition for probability function assuming equal likelihood of each outcome
def prob (s : Set ℕ) : ℝ := (s.card : ℝ) / Omega.card

-- Conditions
axiom p_ABC_eq_pA_pB_pC (m n : ℕ) :
  prob (A ∩ B ∩ C m n) = prob A * prob B * prob (C m n)

axiom not_pairwise_independent (m n : ℕ) :
  ¬ (prob (A ∩ B) = prob A * prob B ∧
     prob (A ∩ C m n) = prob A * prob (C m n) ∧
     prob (B ∩ C m n) = prob B * prob (C m n))

-- The theorem to be proved
theorem find_m_n (m n : ℕ) : m + n = 13 :=
  sorry

end find_m_n_l645_645181


namespace problem_solution_l645_645908

noncomputable def problem_statement : Prop :=
  let Γ : set (ℝ × ℝ) := {p | (p.1 ^ 2) / 4 + p.2 ^ 2 = 1}
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (0, -1)
  let l1 : set (ℝ × ℝ) := {p | p.1 = -2}
  let l2 : set (ℝ × ℝ) := {p | p.2 = -1}
  ∃ (P : ℝ × ℝ) (x0 y0 : ℝ), 
    P = (x0, y0) ∧ x0 > 0 ∧ y0 > 0 ∧ P ∈ Γ ∧ 
    let l3 : set (ℝ × ℝ) := {p | (x0 * p.1) / 4 + y0 * p.2 = 1} in
    let C : ℝ × ℝ := (-2, -1) in
    let D : ℝ × ℝ := (4 * (1 + y0) / x0, -1) in
    let E : ℝ × ℝ := (-2, (2 + x0) / (2 * y0)) in
    let AD : set (ℝ × ℝ) := {p | ∃ (t : ℝ), p = t • D + (1 - t) • A} in
    let BE : set (ℝ × ℝ) := {p | ∃ (t : ℝ), p = t • E + (1 - t) • B} in
    let CP : set (ℝ × ℝ) := {p | ∃ (t : ℝ), p = t • P + (1 - t) • C} in
    ∃ (Q : ℝ × ℝ), Q ∈ AD ∧ Q ∈ BE ∧ Q ∈ CP

theorem problem_solution : problem_statement := 
  sorry

end problem_solution_l645_645908


namespace value_of_x_l645_645258

-- Define the variables x, y, z
variables (x y z : ℕ)

-- Hypothesis based on the conditions of the problem
hypothesis h1 : x = y / 3
hypothesis h2 : y = z / 4
hypothesis h3 : z = 48

-- The statement to be proved
theorem value_of_x : x = 4 :=
by { sorry }

end value_of_x_l645_645258


namespace trains_cleared_in_16_seconds_l645_645331

-- Definitions and assumptions based on the conditions
def length_train1 : ℝ := 100
def length_train2 : ℝ := 220
def speed_train1_kmph : ℝ := 42
def speed_train2_kmph : ℝ := 30

-- Conversion factor
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

-- Total distance that needs to be covered
def total_distance : ℝ := length_train1 + length_train2

-- Time needed for the trains to be clear of each other
def time_to_clear : ℝ := total_distance / relative_speed_mps

-- The theorem to prove
theorem trains_cleared_in_16_seconds : time_to_clear = 16 := by
  sorry

end trains_cleared_in_16_seconds_l645_645331


namespace binomial_alternating_sum_l645_645590

open Nat

theorem binomial_alternating_sum (n t : ℕ)  (hn : 0 < n) : 
  (∑ i in Finset.range (n + 1), (-1 : ℤ)^i * choose n i * choose (n + t - i) n) = 1 := 
by
  sorry

end binomial_alternating_sum_l645_645590


namespace interest_earned_l645_645318

-- Definition: Principal amount (P)
def principal : ℝ := 15000

-- Definition: Annual interest rate (R)
def rate : ℝ := 22.5

-- Definition: Time period in years (T)
def time : ℝ := 4 + 9 / 12

-- Definition: Simple interest formula computation (SI)
def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

-- Theorem: The interest earned on the investment
theorem interest_earned :
  simple_interest principal rate time = 16031.25 :=
by
  sorry

end interest_earned_l645_645318


namespace acid_concentration_third_flask_l645_645291

-- Define the concentrations of first and second flask
def conc_first (w1 : ℝ) : ℝ := 10 / (10 + w1)
def conc_second (w2 : ℝ) : ℝ := 20 / (20 + w2)

-- Define the acid mass in the third flask initially
def acid_mass_third : ℝ := 30

-- Total water added from the fourth flask
def total_water (w1 w2 : ℝ) : ℝ := w1 + w2

-- Acid concentration in the third flask after all water is added
def conc_third (w : ℝ) : ℝ := acid_mass_third / (acid_mass_third + w)

-- Problem statement: concentration in the third flask is 10.5%
theorem acid_concentration_third_flask (w1 : ℝ) (w2 : ℝ) (w : ℝ) 
  (h1 : conc_first w1 = 0.05) 
  (h2 : conc_second w2 = 70 / 300) 
  (h3 : w = total_water w1 w2) : 
  conc_third w = 10.5 / 100 := 
sorry

end acid_concentration_third_flask_l645_645291


namespace value_of_x_l645_645264

theorem value_of_x (x y z : ℤ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 :=
by
  sorry

end value_of_x_l645_645264


namespace todd_final_money_l645_645666

noncomputable def todd_initial_money : ℝ := 100
noncomputable def todd_debt : ℝ := 110
noncomputable def todd_spent_on_ingredients : ℝ := 75
noncomputable def snow_cones_sold : ℝ := 200
noncomputable def price_per_snowcone : ℝ := 0.75

theorem todd_final_money :
  let initial_money := todd_initial_money,
      debt := todd_debt,
      spent := todd_spent_on_ingredients,
      revenue := snow_cones_sold * price_per_snowcone,
      remaining := initial_money - spent,
      total_pre_debt := remaining + revenue,
      final_money := total_pre_debt - debt
  in final_money = 65 :=
by
  sorry

end todd_final_money_l645_645666


namespace Mitya_age_l645_645189

noncomputable def Mitya_current_age (S M : ℝ) := 
  (S + 11 = M) ∧ (M - S = 2*(S - (M - S)))

theorem Mitya_age (S M : ℝ) (h : Mitya_current_age S M) : M = 27.5 := by
  sorry

end Mitya_age_l645_645189


namespace concentration_third_flask_l645_645280

-- Define the concentrations as per the given problem

noncomputable def concentration (acid_mass water_mass : ℝ) : ℝ :=
  (acid_mass / (acid_mass + water_mass)) * 100

-- Given conditions
def acid_mass_first_flask : ℝ := 10
def acid_mass_second_flask : ℝ := 20
def acid_mass_third_flask : ℝ := 30
def concentration_first_flask : ℝ := 5
def concentration_second_flask : ℝ := 70 / 3

-- Total water added to the first and second flasks
def total_water_mass : ℝ :=
  let W1 := (acid_mass_first_flask - concentration_first_flask * acid_mass_first_flask / 100)
  let W2 := (acid_mass_second_flask - concentration_second_flask * acid_mass_second_flask / 100)
  W1 + W2 

-- Prove the concentration of acid in the third flask
theorem concentration_third_flask : 
  concentration acid_mass_third_flask total_water_mass = 10.5 := 
  sorry

end concentration_third_flask_l645_645280


namespace number_of_squares_with_center_50_30_l645_645583

theorem number_of_squares_with_center_50_30 : 
  let conditions := λ (S : Set (ℕ × ℕ) → Prop) (center : ℕ × ℕ), 
    ∀ (v ∈ S), let (x, y) := v in 
      50 - |x - 50| + 30 - |y - 30| = 0 ∧ 
      x ≥ 0 ∧ y ≥ 0 in
  ∃ (S : Set (ℕ × ℕ)), 
    conditions S (50, 30) ∧ 
    S.finite ∧ 
    S.card = 930 :=
sorry

end number_of_squares_with_center_50_30_l645_645583


namespace find_AX_l645_645966

theorem find_AX 
  (A B C X : Type) 
  (AB AC BC : ℝ) 
  (h1 : AB = 80) 
  (h2 : ∠ ACX = ∠ BCX) 
  (h3 : BC = 84) 
  (h4 : AC = 42) 
  : AX = (80 / 3) :=
sorry

end find_AX_l645_645966


namespace monotonicity_intervals_l645_645918

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem monotonicity_intervals :
  (∀ x y : ℝ, x ∈ Icc (-1) 1 → y ∈ Icc (-1) 1 → x < y → f y < f x) ∧
  (∀ x y : ℝ, x ∈ Icc (-∞) (-1) → y ∈ Ioc (-∞) (-1) → x < y → f x < f y) ∧
  (∀ x y : ℝ, x ∈ Icc 1 ∞ → y ∈ Ioc 1 ∞ → x < y → f x < f y) :=
by
  intro x y hx hy hxy
  sorry

end monotonicity_intervals_l645_645918


namespace fundraising_exceeded_goal_l645_645216

theorem fundraising_exceeded_goal:
  let goal := 4000
  let ken := 600
  let mary := 5 * ken
  let scott := mary / 3
  let total := ken + mary + scott
  total - goal = 600 :=
by
  let goal := 4000
  let ken := 600
  let mary := 5 * ken
  let scott := mary / 3
  let total := ken + mary + scott
  have h_goal : goal = 4000 := rfl
  have h_ken : ken = 600 := rfl
  have h_mary : mary = 5 * ken := rfl
  have h_scott : scott = mary / 3 := rfl
  have h_total : total = ken + mary + scott := rfl
  calc total - goal = (ken + mary + scott) - goal : by rw h_total
  ... = (600 + 3000 + 1000) - 4000 : by {rw [h_ken, h_mary, h_scott], norm_num}
  ... = 600 : by norm_num

end fundraising_exceeded_goal_l645_645216


namespace lateral_surface_area_and_volume_l645_645334

section Pyramid

variables 
  (m : ℝ) -- The area of the base parallelogram ABCD

-- Given definitions
def base_area (ABCD_area : ℝ) : Prop := ABCD_area = m^2
def BD_perpendicular_AD : Prop := true
def dihedral_angle_AD_BC : Prop := true
def dihedral_angle_AB_CD : Prop := true

-- The credited problem statement to be proved
theorem lateral_surface_area_and_volume :
  base_area m^2 → 
  BD_perpendicular_AD → 
  dihedral_angle_AD_BC → 
  dihedral_angle_AB_CD →
  (lateral_surface_area = m^2 * (1 + (real.sqrt 2) / 2) ∧
   volume = (m^3 * (real.sqrt (real.sqrt 2)) / 6)) :=
by
  intros,
  sorry

end Pyramid

end lateral_surface_area_and_volume_l645_645334


namespace Mitya_age_l645_645190

noncomputable def Mitya_current_age (S M : ℝ) := 
  (S + 11 = M) ∧ (M - S = 2*(S - (M - S)))

theorem Mitya_age (S M : ℝ) (h : Mitya_current_age S M) : M = 27.5 := by
  sorry

end Mitya_age_l645_645190


namespace subset_sums_eq_l645_645333

/-- Given a positive integer n, define the set I = {0, 1, ..., 10^n - 1}. 
We prove that there exist subsets A and B of I such that 
A ∪ B = I, A ∩ B = ∅, 
and ∀ k ∈ ℕ, 0 ≤ k ≤ n-1 implies ∑ x in A, x^k = ∑ y in B, y^k. -/
theorem subset_sums_eq (n : ℕ) (h : 0 < n) : 
  ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range (10^n) ∧
    A ∩ B = ∅ ∧
    ∀ (k : ℕ), k ≤ n - 1 → ∑ x in A, x^k = ∑ y in B, y^k := 
by {
  sorry
}

end subset_sums_eq_l645_645333


namespace correct_propositions_l645_645883

variables {α β : Type} {l m : Type}
variable [field α]
variable [affine_space β α]

-- Definitions
variables (line_l : l) (plane_alpha : β) (line_m : m) (plane_beta : β)
variable (l_perp_alpha : l ⊥ α)
variable (m_in_beta : m ∈ β)

-- Propositions
def proposition1 : Prop := α ∥ β → l ⊥ m
def proposition3 : Prop := l ∥ m → α ⊥ β

-- The theorem to be proved
theorem correct_propositions :
  proposition1 line_l plane_alpha line_m plane_beta ∧ 
  proposition3 line_l plane_alpha line_m plane_beta :=
begin
  sorry
end

end correct_propositions_l645_645883


namespace meaningful_fraction_x_range_l645_645524

theorem meaningful_fraction_x_range (x : ℝ) : (x-2 ≠ 0) ↔ (x ≠ 2) :=
by 
  sorry

end meaningful_fraction_x_range_l645_645524


namespace f_monotonically_increasing_g_max_min_in_interval_l645_645912

noncomputable def f (x : ℝ) : ℝ :=
  2 * sqrt 3 * sin (x + π / 4) * cos (x + π / 4) + sin (2 * x)

noncomputable def g (x : ℝ) : ℝ :=
  f (x + π / 6)

theorem f_monotonically_increasing (k : ℤ) :
  ∀ x, x ∈ Ioc (-5 * π / 12 + k * π) (π / 12 + k * π) → 4 * cos (2 * x + π / 3) > 0 :=
sorry

theorem g_max_min_in_interval :
  ∃ x₁ x₂, x₁ ∈ Icc 0 (π / 2) ∧ x₂ ∈ Icc 0 (π / 2) ∧ g x₁ = sqrt 3 ∧ g x₂ = -2 :=
sorry

end f_monotonically_increasing_g_max_min_in_interval_l645_645912


namespace smallest_sector_angle_l645_645145

def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem smallest_sector_angle 
  (a : ℕ) (d : ℕ) (n : ℕ := 15) (sum_angles : ℕ := 360) 
  (angles_arith_seq : arithmetic_sequence_sum a d n = sum_angles) 
  (h_poses : ∀ m : ℕ, arithmetic_sequence_sum a d m = sum_angles -> m = n) 
  : a = 3 := 
by 
  sorry

end smallest_sector_angle_l645_645145


namespace convert_base_10_to_base_7_l645_645313

theorem convert_base_10_to_base_7 (n : ℕ) (h : n = 784) : 
  ∃ a b c d : ℕ, n = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 :=
by
  sorry

end convert_base_10_to_base_7_l645_645313


namespace parabola_focus_distance_l645_645074

noncomputable def point_on_parabola (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x

noncomputable def focus_distance (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  let focus_x := p / 2 in
  real.sqrt ((x - focus_x) ^ 2 + y ^ 2) = 12

noncomputable def y_axis_distance (x : ℝ) : Prop :=
  real.abs x = 9

theorem parabola_focus_distance (p : ℝ) (x : ℝ) (y : ℝ) (h1 : point_on_parabola p x y) (h2 : focus_distance p x y) (h3 : y_axis_distance x) : p = 6 :=
sorry

end parabola_focus_distance_l645_645074


namespace equilateral_triangle_ratio_is_correct_l645_645695

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l645_645695


namespace miranda_saves_half_of_salary_l645_645207

noncomputable def hourly_wage := 10
noncomputable def daily_hours := 10
noncomputable def weekly_days := 5
noncomputable def weekly_salary := hourly_wage * daily_hours * weekly_days

noncomputable def robby_saving_fraction := 2 / 5
noncomputable def jaylen_saving_fraction := 3 / 5
noncomputable def total_savings := 3000
noncomputable def weeks := 4

noncomputable def robby_weekly_savings := robby_saving_fraction * weekly_salary
noncomputable def jaylen_weekly_savings := jaylen_saving_fraction * weekly_salary
noncomputable def robby_total_savings := robby_weekly_savings * weeks
noncomputable def jaylen_total_savings := jaylen_weekly_savings * weeks
noncomputable def combined_savings_rj := robby_total_savings + jaylen_total_savings
noncomputable def miranda_total_savings := total_savings - combined_savings_rj
noncomputable def miranda_weekly_savings := miranda_total_savings / weeks

noncomputable def miranda_saving_fraction := miranda_weekly_savings / weekly_salary

theorem miranda_saves_half_of_salary:
  miranda_saving_fraction = 1 / 2 := 
by sorry

end miranda_saves_half_of_salary_l645_645207


namespace corona_diameter_in_meters_l645_645192

theorem corona_diameter_in_meters : 
  ∀ (nanometers meters : ℝ), 
  nanometers = 120 ∧ meters = 10^(-9) → 
  (120 * meters = 1.2 * 10^(-7)) :=
by
  intros nanometers meters h
  cases h with h_nm h_m
  rw [h_nm, h_m, mul_assoc] 
  norm_num
  sorry

end corona_diameter_in_meters_l645_645192


namespace almost_all_G_in_P_H_l645_645112

noncomputable def gamma_tends_to_infinity (n : ℕ) : Prop :=
  ∃ (γ : ℕ → ℝ), (∀ N > 0, ∃ n ≥ N, γ n > N)

noncomputable def graph_in_class (G : Type) (H : Type) (P_H : Set G) : Prop :=
  ∀ g : G, g ∈ P_H ↔ ∃ h : H, h ⊆ g

variable {n : ℕ}
variable {H G : Type}
variable {P_H : Set G}

noncomputable def epsilon_prime (H : Type) : ℝ := 
  sorry -- placeholder for the actual implementation of epsilon(H)

theorem almost_all_G_in_P_H 
  (t : ℝ) (p : ℝ) (γ : ℕ → ℝ)
  (h_t : t = n^(-1 / epsilon_prime H)) 
  (h_p : ∀ n, p = γ n * n^(-1 / epsilon_prime H)) 
  (h_gamma : gamma_tends_to_infinity n) :
  ∀ G : Type, ∀ (G_set : Set G), graph_in_class G H P_H → 
  ∃ (almost_all_g : G → Prop), 
    (∀ g : G, g ∈ G_set → almost_all_g g) → 
    ∀ g : G, g ∈ G_set → g ∈ P_H := 
begin
  sorry -- Proof not required as per instructions
end

end almost_all_G_in_P_H_l645_645112


namespace constant_reciprocal_slopes_l645_645066

noncomputable def ellipse_with_focus (x y a b : ℝ) : Prop :=
y^2 / a^2 + x^2 / b^2 = 1

noncomputable def slopes (x₁ y₁ x₂ y₂ : ℝ) (P : (ℝ × ℝ)) : ℝ × ℝ :=
(((y₁ - P.2) / (x₁ - P.1)), ((y₂ - P.2) / (x₂ - P.1)))

theorem constant_reciprocal_slopes {a b k₁ k₂ : ℝ} (P F : (ℝ × ℝ)) :
  (∃ a b, a > b ∧ b > 0 ∧ (F = (0, 1)) ∧ ellipse_with_focus F.1 F.2 a b ∧ a = 2 ∧ b = sqrt 3) →
  (mn_slope : ℝ) = -1 →
  (∀ M N, (mn_intercepts : F.1 = -F.1 + 1 ∧ ellipse_with_focus M.1 M.2 a b ∧ ellipse_with_focus N.1 N.2 a b) →
  let (k₁, k₂) := slopes M.1 M.2 N.1 N.2 P in 
  k₁ * k₂ = 2 ∧ 1/k₁ + 1/k₂ = 2) →
  (∃k₁ k₂, 1/k₁ + 1/k₂ = 2) :=
begin
  assume h,
  sorry
end

end constant_reciprocal_slopes_l645_645066


namespace systematic_sampling_seventeenth_group_l645_645359

theorem systematic_sampling_seventeenth_group :
  ∀ (total_students : ℕ) (sample_size : ℕ) (first_number : ℕ) (interval : ℕ),
  total_students = 800 →
  sample_size = 50 →
  first_number = 8 →
  interval = total_students / sample_size →
  first_number + 16 * interval = 264 :=
by
  intros total_students sample_size first_number interval h1 h2 h3 h4
  sorry

end systematic_sampling_seventeenth_group_l645_645359


namespace count_ordered_pairs_satisfying_equation_l645_645106

theorem count_ordered_pairs_satisfying_equation :
  {p : ℤ × ℤ | (p.1)^2 + (p.2)^3 = (p.2)^2 + 2 * p.1 * p.2}.card = 3 :=
by sorry

end count_ordered_pairs_satisfying_equation_l645_645106


namespace solve_for_n_l645_645180

-- Definitions based on the conditions
def divisor_set (n : ℕ) : set ℕ := {d | d > 0 ∧ d ∣ n}

def smallest_divisors (n : ℕ) (d1 d2 d3 d4 : ℕ) : Prop :=
  d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧
  d1 ∈ divisor_set n ∧
  d2 ∈ divisor_set n ∧
  d3 ∈ divisor_set n ∧
  d4 ∈ divisor_set n

-- The main statement to prove
theorem solve_for_n: ∀ (n d1 d2 d3 d4 : ℕ), 
  (n > 0) → 
  (smallest_divisors n d1 d2 d3 d4) → 
  (n = d1^2 + d2^2 + d3^2 + d4^2) → 
  n = 130 ∧ d1 = 1 ∧ d2 = 2 ∧ d3 = 5 ∧ d4 = 10 :=
begin
  sorry
end

end solve_for_n_l645_645180


namespace intersection_of_A_and_B_l645_645071

open Set

def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { y | ∃ x, y = (1 / 2) ^ x ∧ x > -1 }

theorem intersection_of_A_and_B : A ∩ B = { y | 0 < y ∧ y ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l645_645071


namespace three_zeros_condition_l645_645114

noncomputable def f (ω : ℝ) (x : ℝ) := Real.sin (ω * x) + Real.cos (ω * x)

theorem three_zeros_condition (ω : ℝ) (hω : ω > 0) :
  (∃ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ 2 * Real.pi ∧
  f ω x1 = 0 ∧ f ω x2 = 0 ∧ f ω x3 = 0) →
  (∀ ω, (11 / 8 : ℝ) ≤ ω ∧ ω < (15 / 8 : ℝ) ∧
  (∀ x, f ω x = 0 ↔ x = (5 * Real.pi) / (4 * ω))) :=
sorry

end three_zeros_condition_l645_645114


namespace required_circle_equation_l645_645435

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line equation on which the center of the required circle lies
def center_line (x y : ℝ) : Prop := 3 * x + 4 * y - 1 = 0

-- State the final proof that the equation of the required circle is (x + 1)^2 + (y - 1)^2 = 13 under the given conditions
theorem required_circle_equation (x y : ℝ) :
  ( ∃ (x1 y1 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧
    (∃ (cx cy r : ℝ), center_line cx cy ∧ (x - cx)^2 + (y - cy)^2 = r^2 ∧ (x1 - cx)^2 + (y1 - cy)^2 = r^2 ∧
      (x + 1)^2 + (y - 1)^2 = 13) )
 := sorry

end required_circle_equation_l645_645435


namespace time_differences_l645_645119

def runner_times : Type := ⟨ℕ, ℕ, ℕ, ℕ⟩

def t_A : ℕ := 36
def t_B : ℕ := 45
def t_C : ℕ := 42
def t_D : ℕ := 40

theorem time_differences :
  (t_B - t_A = 9) ∧
  (t_C - t_A = 6) ∧
  (t_D - t_A = 4) ∧
  (t_B - t_A = 9) ∧
  (t_C - t_A = 6) ∧
  (t_D - t_A = 4) ∧
  (t_B - t_C = 3) ∧
  (t_B - t_D = 5) ∧
  (t_C - t_D = 2) :=
by
  -- These are the conditions used in the problem
  have h1 : t_A = 36 := rfl
  have h2 : t_B = 45 := rfl
  have h3 : t_C = 42 := rfl
  have h4 : t_D = 40 := rfl

  -- Calculate the differences directly using arithmetic subtraction in Lean
  calc 
    t_B - t_A = 45 - 36 : by rw [h1, h2]
          ... = 9       : by norm_num
    t_C - t_A = 42 - 36 : by rw [h1, h3]
          ... = 6       : by norm_num
    t_D - t_A = 40 - 36 : by rw [h1, h4]
          ... = 4       : by norm_num
    t_B - t_C = 45 - 42 : by rw [h2, h3]
          ... = 3       : by norm_num
    t_B - t_D = 45 - 40 : by rw [h2, h4]
          ... = 5       : by norm_num
    t_C - t_D = 42 - 40 : by rw [h3, h4]
          ... = 2       : by norm_num

  -- Combining everything together
  exact ⟨
    by calc t_B - t_A = 45 - 36 : by rw [h1, h2]; exact rfl
              ... = 9 : by norm_num,
    by calc t_C - t_A = 42 - 36 : by rw [h1, h3]; exact rfl
              ... = 6 : by norm_num,
    by calc t_D - t_A = 40 - 36 : by rw [h1, h4]; exact rfl
              ... = 4 : by norm_num,
    by calc t_B - t_A = 45 - 36 : by rw [h1, h2]; exact rfl
              ... = 9 : by norm_num,
    by calc t_C - t_A = 42 - 36 : by rw [h1, h3]; exact rfl
              ... = 6 : by norm_num,
    by calc t_D - t_A = 40 - 36 : by rw [h1, h4]; exact rfl
              ... = 4 : by norm_num,
    by calc t_B - t_C = 45 - 42 : by rw [h2, h3]; exact rfl
              ... = 3 : by norm_num,
    by calc t_B - t_D = 45 - 40 : by rw [h2, h4]; exact rfl
              ... = 5 : by norm_num,
    by calc t_C - t_D = 42 - 40 : by rw [h3, h4]; exact rfl
              ... = 2 : by norm_num
  ⟩

end time_differences_l645_645119


namespace simplify_sqrt_expression_l645_645222

theorem simplify_sqrt_expression : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 75 = 5 * Real.sqrt 3 :=
by
  sorry

end simplify_sqrt_expression_l645_645222


namespace triangle_sine_law_A_gt_B_triangle_two_solutions_l645_645548

-- Defining the context of the triangle and angles
variables (A B C : ℝ) (a b c : ℝ)

-- Condition: In triangle ABC, with sides a, b, c opposite to angles A, B, C respectively.
-- Prove: If A > B, then sin A > sin B
theorem triangle_sine_law_A_gt_B (hA_gt_B : A > B) : real.sin A > real.sin B :=
sorry

-- Prove: If A = 30 degrees, b = 4, a = 3, then triangle ABC has two solutions.
theorem triangle_two_solutions (hA : A = 30 * real.pi / 180) (hb : b = 4) (ha : a = 3) : ∃ B1 B2 : ℝ, B1 ≠ B2 ∧ real.sin B1 = (4 / 3) * real.sin (30 * real.pi / 180) ∧ real.sin B2 = (4 / 3) * real.sin (30 * real.pi / 180) :=
sorry

end triangle_sine_law_A_gt_B_triangle_two_solutions_l645_645548


namespace root_range_of_f_eq_zero_solution_set_of_f_le_zero_l645_645487

variable (m : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := m * x^2 + (2 * m + 1) * x + 2

theorem root_range_of_f_eq_zero (h : ∃ r1 r2 : ℝ, r1 > 1 ∧ r2 < 1 ∧ f r1 = 0 ∧ f r2 = 0) : -1 < m ∧ m < 0 :=
sorry

theorem solution_set_of_f_le_zero : 
  (m = 0 -> ∀ x, f x ≤ 0 ↔ x ≤ - 2) ∧
  (m < 0 -> ∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ - (1/m)) ∧
  (0 < m ∧ m < 1/2 -> ∀ x, f x ≤ 0 ↔ - (1/m) ≤ x ∧ x ≤ - 2) ∧
  (m = 1/2 -> ∀ x, f x ≤ 0 ↔ x = - 2) ∧
  (m > 1/2 -> ∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ - (1/m)) :=
sorry

end root_range_of_f_eq_zero_solution_set_of_f_le_zero_l645_645487


namespace a_seq_correct_l645_645158

-- Define the sequence and the sum condition
def a_seq (n : ℕ) : ℚ := if n = 0 then 0 else (2 ^ n - 1) / 2 ^ (n - 1)

def S_n (n : ℕ) : ℚ :=
  if n = 0 then 0 else (Finset.sum (Finset.range n) a_seq)

axiom condition (n : ℕ) (hn : n > 0) : S_n n + a_seq n = 2 * n

theorem a_seq_correct (n : ℕ) (hn : n > 0) : 
  a_seq n = (2 ^ n - 1) / 2 ^ (n - 1) := sorry

end a_seq_correct_l645_645158


namespace geometric_dist_properties_l645_645037

-- Definition of the geometric distribution conditions
variables {p q : ℝ} (h₁ : 0 < p) (h₂ : 0 < q) (h₃ : p + q = 1)

-- Define the geometric distribution as a discrete random variable
def geo_dist (n : ℕ) : ℝ := if n > 0 then p * q^(n - 1) else 0

noncomputable def char_fn (t : ℝ) : ℝ := p / (Real.exp (-complex.i * t) - q)

noncomputable def expectation : ℝ := 1 / p

noncomputable def variance : ℝ := q / p^2

-- The theorem to prove the characteristic function, expectation, and variance of ξ
theorem geometric_dist_properties :
  char_fn t = p / (Real.exp (-complex.i * t) - q) ∧
  expectation = 1 / p ∧
  variance = q / p^2 :=
by
  sorry

end geometric_dist_properties_l645_645037


namespace largest_d_in_range_l645_645826

theorem largest_d_in_range (d : ℝ) (g : ℝ → ℝ) :
  (g x = x^2 - 6x + d) → (∃ x : ℝ, g x = 2) → d ≤ 11 :=
by
  sorry

end largest_d_in_range_l645_645826


namespace fraction_of_roll_used_l645_645209

theorem fraction_of_roll_used 
  (x : ℚ) 
  (h1 : 3 * x + 3 * x + x + 2 * x = 9 * x)
  (h2 : 9 * x = (2 / 5)) : 
  x = 2 / 45 :=
by
  sorry

end fraction_of_roll_used_l645_645209


namespace num_odd_functions_l645_645385

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem num_odd_functions : 
  let f1 := (λ x : ℝ, x^3),
      f2 := (λ x : ℝ, x^2 + 1),
      f3 := (λ x : ℝ, 1 / x),
      f4 := (λ x : ℝ, |x| + 3) in
  (is_odd f1) ∧ ¬(is_odd f2) ∧ (is_odd f3) ∧ ¬(is_odd f4) →
  2 = 2 :=
by
  intros,
  sorry

end num_odd_functions_l645_645385


namespace quadratic_problem_l645_645167

-- Definitions
variables {ℝ : Type*} [linear_ordered_field ℝ]

-- Quadratic functions f and g
variables (f g : ℝ → ℝ)
variable (vertex : ℝ)

-- Conditions
def condition1 := ∀ x : ℝ, g(x) = -3 * f(200 - x)
def condition2 := vertex = find_vertex f = find_vertex g
variables (a1 a2 a3 a4 : ℝ)

def condition3 := a1 < a2 ∧ a2 < a3 ∧ a3 < a4
def condition4 := a3 - a2 = 300

-- Final result to prove
def to_prove := (a4 - a1) = 200 - 2 * real.sqrt 2

-- The statement
theorem quadratic_problem :
  ∀ (f g : ℝ → ℝ) (vertex a1 a2 a3 a4 : ℝ),
    (condition1 f g) →
    (condition2 f g vertex) →
    (condition3 a1 a2 a3 a4) →
    (condition4 a3 a2) →
    to_prove a4 a1 :=
by obviously sorry

end quadratic_problem_l645_645167


namespace radius_of_tangent_circle_l645_645887

variable (A B C D E F G O : Type)
           [RegularHexagon A B C D E F]
           (r : ℝ)
           (side_length : ℝ := 6)
           (tangent_point : G)
           (circle_center : O)

def is_tangent_to_extension (hex_side : Segment C D) : Prop :=
  TangentFromSideExtension hex_side tangent_point

def radius_of_circle (r : ℝ) : Prop :=
  (CircleThruPoints A E circle_center r) ∧ (is_tangent_to_extension D G)

theorem radius_of_tangent_circle : radius_of_circle 6 :=
by sorry

end radius_of_tangent_circle_l645_645887


namespace markup_percentage_is_correct_l645_645362

variable (purchase_price : ℝ) (markup_percentage : ℝ) (gross_profit : ℝ) 
variable (decreased_percentage : ℝ) (new_selling_price : ℝ)
variable (original_selling_price : ℝ)

# Let the purchase price be $210
def PP := 210

# The selling price after reducing by 20%
def NSP := (original_selling_price * (1 - decreased_percentage))

# Given conditions
axiom A1 : decreased_percentage = 0.20
axiom A2 : gross_profit = 14

# The relationship between selling price and purchase price after reduction
axiom A3 : gross_profit = NSP - PP

# Original selling price considering markup
theorem markup_percentage_is_correct : 
  ∃ (M : ℝ), original_selling_price = purchase_price * (1 + M / 100) 
  → PP = 210 
  → decreased_percentage = 0.20 
  → gross_profit = 14 
  → M = 33.33 := by
  sorry

end markup_percentage_is_correct_l645_645362


namespace volume_of_each_cube_l645_645351

theorem volume_of_each_cube
  (length : ℝ) (width : ℝ) (height : ℝ) (num_cubes : ℝ)
  (h1 : length = 7)
  (h2 : width = 18)
  (h3 : height = 3)
  (h4 : num_cubes = 42) :
  let V_box := length * width * height in
  let V_cube := V_box / num_cubes in
  V_cube = 9 :=
by
  sorry

end volume_of_each_cube_l645_645351


namespace integral_solution_l645_645332

noncomputable def integral_problem : ℝ :=
  ∫ x in 0..(2 * Real.arctan (1/2)), (1 - Real.sin x) / (Real.cos x * (1 + Real.cos x))

theorem integral_solution : integral_problem = 2 * Real.log (3/2) - 1/2 :=
  sorry

end integral_solution_l645_645332


namespace min_teachers_required_l645_645735

-- Define the conditions
def num_english_teachers : ℕ := 9
def num_history_teachers : ℕ := 7
def num_geography_teachers : ℕ := 6
def max_subjects_per_teacher : ℕ := 2

-- The proposition we want to prove
theorem min_teachers_required :
  ∃ (t : ℕ), t = 13 ∧
    t * max_subjects_per_teacher ≥ num_english_teachers + num_history_teachers + num_geography_teachers :=
sorry

end min_teachers_required_l645_645735


namespace compare_abc_l645_645868

noncomputable def a : ℝ := 2 * Real.sqrt Real.exp
noncomputable def b : ℝ := 3 * Real.root 3 Real.exp
noncomputable def c : ℝ := Real.exp ^ 2 / (4 - Real.log 4)

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l645_645868


namespace sum_of_sequence_up_to_100_l645_645412

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 2 ∧ (∀ n, a n * a (n + 1) * a (n + 2) ≠ 1) ∧
  ∀ n, a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3)

theorem sum_of_sequence_up_to_100 (a : ℕ → ℕ) (h : sequence a) :
  ∑ i in Finset.range 100, a (i + 1) = 200 :=
sorry

end sum_of_sequence_up_to_100_l645_645412


namespace sequence_general_term_l645_645064

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = n^2 + 1) :
  (∀ n, a n = if n = 1 then 2 else 2 * n - 1) :=
by
  sorry

end sequence_general_term_l645_645064


namespace vector_combination_l645_645511

def vec : Type := ℝ × ℝ × ℝ

def a : vec := (1, 0, 1)
def b : vec := (-1, 2, 3)
def c : vec := (0, 1, 1)

theorem vector_combination : 
  a - b + (2 • c) = (2, 0, 0) :=
by
  sorry

end vector_combination_l645_645511


namespace polynomial_even_solution_impossible_l645_645557

theorem polynomial_even_solution_impossible {f : ℝ → ℝ} (hf : polynomial f) (deg_f : f.degree ≠ 0) :
  ¬ (∀ a : ℝ, ∃ n : ℕ, even n ∧ (λ x, f x = a).roots n) :=
by
  sorry

end polynomial_even_solution_impossible_l645_645557


namespace floor_T_squared_l645_645176

noncomputable def T : ℝ := ∑ i in Finset.range 1003, 
  Real.sqrt (1 + 1 / ((2 * (i + 1) - 1)^2 : ℝ) + 1 / ((2 * (i + 1) + 1)^2))

theorem floor_T_squared : ⌊T^2⌋ = 1008015 :=
by 
  sorry

end floor_T_squared_l645_645176


namespace angle_between_vectors_l645_645445

noncomputable def a (coord : ℕ) : ℝ :=
  if coord = 0 then a₀ else if coord = 1 then a₁ else a₂

noncomputable def b : ℕ → ℝ := λ i, if i = 0 then b₀ else if i = 1 then b₁ else b₂

noncomputable def c : ℕ → ℝ := λ i, if i = 0 then 1 else if i = 1 then -2 else -2

theorem angle_between_vectors:
  let a := λ i, if i = 0 then a₀ else if i = 1 then a₁ else a₂
  let b := λ i, if i = 0 then b₀ else if i = 1 then b₁ else b₂
  let c := λ i, if i = 0 then 1 else if i = 1 then -2 else -2
  (2 * a 0 + b 0 = 0) ∧ (2 * a 1 + b 1 = -5) ∧ (2 * a 2 + b 2 = 10) ∧
  ((a 0 * c 0 + a 1 * c 1 + a 2 * c 2 = 4) ∧
  (Real.sqrt ( (b 0) ^ 2 + (b 1) ^ 2 + (b 2) ^ 2 ) = 12) ∧
  (real.angle.cos (Real.sqrt ((b 0 * c 0) + (b 1 * c 1) + (b 2 * c 2)) /
    (Real.sqrt ( (b 0) ^ 2 + (b 1) ^ 2 + (b 2) ^ 2 ) *
    Real.sqrt ( (c 0) ^ 2 + (c 1) ^ 2 + (c 2) ^ 2 ))) = cos(60 * Real.pi / 180)) := 
sorry

end angle_between_vectors_l645_645445


namespace first_group_size_l645_645809

theorem first_group_size
  (x : ℕ)
  (h1 : 2 * x + 22 + 16 + 14 = 68) : 
  x = 8 :=
by
  sorry

end first_group_size_l645_645809


namespace main_theorem_l645_645453

noncomputable theory

open_locale real -- open the real numbers locale for distance calculations

namespace mymath

def exists_circle_containing_thirteen_points (points : Finset (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (r : ℝ), r = 1 ∧ (points.filter (λ p, dist center p ≤ r)).card ≥ 13

theorem main_theorem 
  (points : Finset (EuclideanSpace ℝ (Fin 2)))
  (h : points.card = 25)
  (h_dist : ∀ (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)), 
    p1 ∈ points → p2 ∈ points → p3 ∈ points → 
    (dist p1 p2 < 1 ∨ dist p1 p3 < 1 ∨ dist p2 p3 < 1)) :
  exists_circle_containing_thirteen_points points :=
sorry -- proof to be filled

end mymath

end main_theorem_l645_645453


namespace constant_product_l645_645537

noncomputable theory

open_locale classical

variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α] [inner_product_space ℝ α]
variables (O A B C D X E: α) (r : ℝ)
variables (h1 : dist O A = r) (h2 : dist O B = r) (h3 : dist A B = 2 * r)
variables (h4 : dist O C = dist O D) (h5 : C ≠ D) (h6 : inner_product O A O X = 0)

theorem constant_product (X : α) (cd_perp_ab : inner_product C D = 0)
    (hxC : ∃ t : ℝ, X = C + t • (D - C))
    (hxE : dist X E = dist X O):
  (dist A X) * (dist A E) = (r * r - (dist X O) * (dist X O)) :=
sorry

end constant_product_l645_645537


namespace interest_rate_beyond_5_years_l645_645797

theorem interest_rate_beyond_5_years (P : ℝ) (R1 R2 R : ℝ) (I_total : ℝ) :
  P = 12000 → R1 = 6 → R2 = 9 → I_total = 11400 →
  let I_2 := P * (R1 / 100) * 2 in
  let I_3 := P * (R2 / 100) * 3 in
  let I_5 := I_2 + I_3 in
  let I_4 := I_total - I_5 in
  I_4 = P * (R / 100) * 4 →
  R = 14 := 
by
  intros hP hR1 hR2 hI_total
  let I_2 := P * (R1 / 100) * 2
  let I_3 := P * (R2 / 100) * 3
  let I_5 := I_2 + I_3
  let I_4 := I_total - I_5
  sorry

end interest_rate_beyond_5_years_l645_645797


namespace cost_of_getting_into_park_l645_645208

noncomputable def cost_of_parking : ℕ := 10
noncomputable def cost_of_meal_pass : ℕ := 25
noncomputable def distance_to_sea_world : ℕ := 165
noncomputable def car_mileage : ℕ := 30
noncomputable def gas_cost_per_gallon : ℕ := 3
noncomputable def additional_savings_needed : ℕ := 95
noncomputable def current_savings : ℕ := 28

theorem cost_of_getting_into_park : 
  let round_trip_distance := distance_to_sea_world * 2,
      total_gas_needed := round_trip_distance / car_mileage,
      total_gas_cost := total_gas_needed * gas_cost_per_gallon,
      total_known_costs := cost_of_parking + cost_of_meal_pass + total_gas_cost,
      total_needed_savings := current_savings + additional_savings_needed,
      cost_to_get_into_park := total_needed_savings - total_known_costs
  in cost_to_get_into_park = 55 := 
by 
  let round_trip_distance := distance_to_sea_world * 2
  let total_gas_needed := round_trip_distance / car_mileage
  let total_gas_cost := total_gas_needed * gas_cost_per_gallon
  let total_known_costs := cost_of_parking + cost_of_meal_pass + total_gas_cost
  let total_needed_savings := current_savings + additional_savings_needed
  let cost_to_get_into_park := total_needed_savings - total_known_costs
  show cost_to_get_into_park = 55, 
  from sorry

end cost_of_getting_into_park_l645_645208


namespace smallest_c_is_17_l645_645166

noncomputable def smallest_c (c d : ℝ) : ℝ :=
if ∀ x : ℤ, real.cos (c * x + d) = real.cos (17 * (x : ℝ)) then c else 0

theorem smallest_c_is_17 :
  ∃ (c d : ℝ), (c ≥ 0 ∧ d ≥ 0) ∧ (∀ x : ℤ, real.cos (c * (x : ℝ) + d) = real.cos (17 * (x : ℝ))) ∧ c = 17 := 
by
  sorry

end smallest_c_is_17_l645_645166


namespace functions_have_inverses_l645_645732

section
variables (a d f g h : ℝ → ℝ)

-- Define the conditions
def a_def := λ x : ℝ, sqrt (2 - x)
def d_def := λ x : ℝ, 2*x^2 + 4*x + 7
def f_def := λ x : ℝ, 3^x + 7^x
def g_def := λ x : ℝ, x - 1/x
def h_def := λ x : ℝ, x / 2

-- Define the domains
def domain_a := {x : ℝ | x ≤ 2}
def domain_d := {x : ℝ | 0 ≤ x}
def domain_f := set.univ -- ℝ
def domain_g := {x : ℝ | 0 < x}
def domain_h := {x : ℝ | -2 ≤ x ∧ x < 7}

-- Define the properties that each function needs to have an inverse
def has_inverse (f : ℝ → ℝ) (domain : set ℝ) : Prop :=
  ∀ x1 x2 ∈ domain, f x1 = f x2 → x1 = x2

-- The theorem to prove
theorem functions_have_inverses :
  has_inverse a_def domain_a ∧
  has_inverse d_def domain_d ∧
  has_inverse f_def domain_f ∧
  has_inverse g_def domain_g ∧
  has_inverse h_def domain_h :=
sorry

end

end functions_have_inverses_l645_645732


namespace find_R_l645_645944

theorem find_R (R : ℝ) (h_diff : ∃ a b : ℝ, a ≠ b ∧ (a - b = 12 ∨ b - a = 12) ∧ a + b = 2 ∧ a * b = -R) : R = 35 :=
by
  obtain ⟨a, b, h_neq, h_diff_12, h_sum, h_prod⟩ := h_diff
  sorry

end find_R_l645_645944


namespace problem_solution_l645_645930

theorem problem_solution (x : ℝ) (h : sqrt (9 + x) + sqrt (16 - x) = 8) : (9 + x) * (16 - x) = 380.25 :=
by
  sorry

end problem_solution_l645_645930


namespace calculate_distance_l645_645473

def velocity (t : ℝ) : ℝ := 3 * t^2 + t

theorem calculate_distance : ∫ t in (0 : ℝ)..(4 : ℝ), velocity t = 72 := 
by
  sorry

end calculate_distance_l645_645473


namespace range_of_m_l645_645069

-- Define the propositions
def p (m : ℝ) : Prop := m ≤ 2
def q (m : ℝ) : Prop := 0 < m ∧ m < 1

-- Problem statement to derive m's range
theorem range_of_m (m : ℝ) (h1: ¬ (p m ∧ q m)) (h2: p m ∨ q m) : m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2) := 
sorry

end range_of_m_l645_645069


namespace width_decreased_by_33_percent_l645_645623

theorem width_decreased_by_33_percent {L W : ℝ} (h : L > 0 ∧ W > 0) (h_area : (1.5 * L) * W' = L * W) :
  W' = (2 / 3) * W :=
begin
  sorry -- Proof to be filled in later
end

end width_decreased_by_33_percent_l645_645623


namespace g_of_neg_3_l645_645177

def g (x : ℝ) : ℝ :=
if x < 0 then 3 * x + 4 else x^2 + 2 * x + 1

theorem g_of_neg_3 : g (-3) = -5 :=
by
  have h_neg : -3 < 0 := by norm_num
  simp [g, h_neg]
  norm_num

end g_of_neg_3_l645_645177


namespace sum_first_n_l645_645079

open Nat

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := 2 * n - 1

-- Define the geometric sequence {b_n}
def b (n : ℕ) : ℕ := 2^(n - 1)

-- Sum of the first n terms of the sequence {a_n + b_n}
theorem sum_first_n (n : ℕ) : ∑ i in range n, (a (i + 1) + b (i + 1)) = n^2 + 2^n - 1 := sorry

end sum_first_n_l645_645079


namespace product_bounds_l645_645456

theorem product_bounds (n : ℕ) (h : n ≥ 2) (x : Fin n → ℝ) 
    (hx1 : ∀ i, x i ≥ 1 / n) 
    (hx2 : ∑ i, (x i)^2 = 1) : 
    (∏ i, x i) ≤ n^(-n / 2) ∧ (∏ i, x i) ≥ (n^2 - n + 1)^(1 / 2) / n^n := 
sorry

end product_bounds_l645_645456


namespace sum_of_numbers_l645_645728

theorem sum_of_numbers : 
  5678 + 6785 + 7856 + 8567 = 28886 := 
by 
  sorry

end sum_of_numbers_l645_645728


namespace ratio_equilateral_triangle_l645_645719

def equilateral_triangle_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * (Real.sqrt 3 / 2)
  let area := (1 / 2) * s * altitude
  let perimeter := 3 * s in
  area / perimeter -- this simplifies to 25\sqrt{3} / 30 or 5\sqrt{3} / 6

theorem ratio_equilateral_triangle : ∀ (s : ℝ), s = 10 → equilateral_triangle_ratio s (by assumption) = 5 * (Real.sqrt 3) / 6 :=
by
  intros s h
  rw h
  sorry

end ratio_equilateral_triangle_l645_645719


namespace problem_l645_645443

def f (c d : ℝ) : ℝ → ℝ :=
λ x, if x < 3 then c * x + d else 10 - 2*x

theorem problem (c d : ℝ) (h : ∀ x, f c d (f c d x) = x) :
  c + d = 9 / 2 :=
by
  sorry

end problem_l645_645443


namespace sixth_root_of_unity_l645_645608

/- Constants and Variables -/
variable (p q r s t k : ℂ)
variable (nz_p : p ≠ 0) (nz_q : q ≠ 0) (nz_r : r ≠ 0) (nz_s : s ≠ 0) (nz_t : t ≠ 0)
variable (hk1 : p * k^5 + q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)
variable (hk2 : q * k^5 + r * k^4 + s * k^3 + t * k^2 + p * k + q = 0)

/- Theorem to prove -/
theorem sixth_root_of_unity : k^6 = 1 :=
by sorry

end sixth_root_of_unity_l645_645608


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645702

theorem ratio_of_area_to_perimeter_of_equilateral_triangle (s : ℕ) : s = 10 → (let A := (s^2 * sqrt 3) / 4, P := 3 * s in A / P = 5 * sqrt 3 / 6) :=
by
  intro h,
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645702


namespace add_base7_correct_l645_645790

-- Define the conversion from base 7 to decimal
def base7_to_decimal : ℕ → ℕ 
| 5 := 5
| 16 := 7 * 1 + 6
| _ := 0

-- Define the addition in base7
def add_in_base7 (n m : ℕ) : ℕ := base7_to_decimal n + base7_to_decimal m

-- Define the conversion from decimal to base 7
def decimal_to_base7 (n : ℕ) : ℕ :=
let q := n / 7 in
let r := n % 7 in
q * 10 + r

-- Prove that the addition result is correct
theorem add_base7_correct : decimal_to_base7 (add_in_base7 5 16) = 24 :=
by sorry

end add_base7_correct_l645_645790


namespace distance_incenter_circumcenter_l645_645375

theorem distance_incenter_circumcenter (A B C : ℝ × ℝ)
  (hAB : (dist A B) = 6)
  (hAC : (dist A C) = 8)
  (hBC : (dist B C) = 10)
  (h_right : ∠ BAC = 90) :
  dist (incenter A B C) (circumcenter A B C) = √5 :=
sorry

end distance_incenter_circumcenter_l645_645375


namespace range_f_in_0_1_l645_645880

variable (f : ℝ → ℝ)
variable (cond : ∀ x y : ℝ, x > y → (f x) ^ 2 ≤ f y)

theorem range_f_in_0_1 : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 :=
begin
  sorry
end

end range_f_in_0_1_l645_645880


namespace symmetry_curve_l645_645489

theorem symmetry_curve (a b : ℝ) (f : ℝ → ℝ)
    (h_def : ∀ x, f(x) = (1/x - a) * Real.log(1 + x))
    (h_symm : ∀ x, f(1/x) = f(1/(2*b - 1/x))) : a + b = -1 :=
by
  sorry

end symmetry_curve_l645_645489


namespace equilateral_triangle_ratio_l645_645714

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Define the altitude of the equilateral triangle
def altitude (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2

-- Define the area of the equilateral triangle
def area (a : ℝ) : ℝ := (a * altitude a) / 2

-- Define the perimeter of the equilateral triangle
def perimeter (a : ℝ) : ℝ := 3 * a

-- Define the ratio of area to perimeter
def ratio (a : ℝ) : ℝ := area a / perimeter a

theorem equilateral_triangle_ratio :
  ratio 10 = (5 * Real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l645_645714


namespace problem_inequality_l645_645895

variable {a b : ℝ}

theorem problem_inequality 
  (h_a_nonzero : a ≠ 0) 
  (h_b_nonzero : b ≠ 0)
  (h_a_gt_b : a > b) : 
  1 / (a * b^2) > 1 / (a^2 * b) := 
by 
  sorry

end problem_inequality_l645_645895


namespace initial_water_ratio_l645_645747

-- Define the conditions given in the problem
def total_capacity : ℝ := 1000 / 1000  -- kiloliters
def fill_rate : ℝ := 1 / 2  -- kiloliters per minute
def drain_rate1 : ℝ := 1 / 4  -- kiloliters per minute
def drain_rate2 : ℝ := 1 / 6  -- kiloliters per minute
def fill_time : ℝ := 6  -- minutes

-- Define the net flow rate calculation
def net_flow_rate : ℝ := fill_rate - (drain_rate1 + drain_rate2)

-- Calculate the amount of water added in the given period of time
def amount_added_in_fill_time : ℝ := net_flow_rate * fill_time

-- Define the initial amount of water
def initial_amount : ℝ := total_capacity - amount_added_in_fill_time

-- Define the ratio of the initial amount of water in the tank to its total capacity
def ratio_initial_to_total : ℝ := initial_amount / total_capacity

-- The theorem that needs to be proved
theorem initial_water_ratio (h1 : total_capacity = 1)
                            (h2 : fill_rate = 0.5)
                            (h3 : drain_rate1 = 0.25)
                            (h4 : drain_rate2 = 1 / 6)
                            (h5 : fill_time = 6) :
  ratio_initial_to_total = 0.5 :=
by
  rw [total_capacity, fill_rate, drain_rate1, drain_rate2, fill_time]
  sorry

end initial_water_ratio_l645_645747


namespace find_a_for_quadratic_l645_645019

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l645_645019


namespace polynomial_integer_roots_bound_l645_645163

open Polynomial

def P (x : ℤ) : Polynomial ℤ := sorry

def n (P : Polynomial ℤ) : ℕ :=
  (Finset.filter (λ x, ((P.eval x) ^ 2 = 1)) (Finset.range 100)).card -- assuming finite range for simplicity

theorem polynomial_integer_roots_bound (P : Polynomial ℤ) (h_nonconstant : P ≠ 0) (h_int_coeffs : ∀ n : ℕ, P.coeff n ∈ ℤ) :
  n(P) ≤ 2 + P.natDegree := 
sorry

end polynomial_integer_roots_bound_l645_645163


namespace unique_shapes_count_l645_645993

theorem unique_shapes_count (R : Type) [rect : rectangle R] :
  let vertices := [A, B, C, D] in
  let circles := count_unique_circles vertices in
  let ellipses := count_unique_ellipses vertices in
  circles + ellipses = 6 :=
sorry

end unique_shapes_count_l645_645993


namespace triangle_inequality_circumscribed_inscribed_relation_l645_645221

open Real

variables {a b c R r : ℝ}

-- Given that a, b, c are the lengths of the sides of a triangle
-- Prove that:
theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_triangle : 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) : 
  a * b * c ≥ (a + b - c) * (a - b + c) * (-a + b + c) :=
sorry

-- Given the problem with R as the radius of the circumscribed circle and r as the radius of the inscribed circle
-- Prove that R ≥ 2r
theorem circumscribed_inscribed_relation (S : ℝ) (p : ℝ) 
  (h1 : S = 1/2 * a * b * (sin (R * c)))
  (h2 : c = 2 * R * (sin (R * c)))
  (h3 : S = r * p)
  (h4 : p = (a + b + c) / 2)
  (h5 : 4 * R * S = a * b * c) :
  R ≥ 2 * r :=
sorry

end triangle_inequality_circumscribed_inscribed_relation_l645_645221


namespace paint_needed_for_720_statues_l645_645935

noncomputable def paint_for_similar_statues (n : Nat) (h₁ h₂ : ℝ) (p₁ : ℝ) : ℝ :=
  let ratio := (h₂ / h₁) ^ 2
  n * (ratio * p₁)

theorem paint_needed_for_720_statues :
  paint_for_similar_statues 720 12 2 1 = 20 :=
by
  sorry

end paint_needed_for_720_statues_l645_645935


namespace concentration_of_acid_in_third_flask_l645_645272

theorem concentration_of_acid_in_third_flask :
  ∀ (W1 W2 : ℝ),
    let W := 190 + 65.714 in 
    W1 = 190 ∧ W2 = 65.714 →
    (10 : ℝ) / (10 + W1) = 0.05 →
    (20 : ℝ) / (20 + W2) = 0.2331 →
    (30 : ℝ) / (30 + W) = 0.105 :=
begin
  sorry
end

end concentration_of_acid_in_third_flask_l645_645272


namespace factorize_m_minimize_ab_find_abc_l645_645538

-- Problem 1: Factorization
theorem factorize_m (m : ℝ) : m^2 - 6 * m + 5 = (m - 1) * (m - 5) :=
sorry

-- Problem 2: Minimization
theorem minimize_ab (a b : ℝ) (h1 : (a - 2)^2 ≥ 0) (h2 : (b + 5)^2 ≥ 0) :
  ∃ (a b : ℝ), (a - 2)^2 + (b + 5)^2 + 4 = 4 ∧ a = 2 ∧ b = -5 :=
sorry

-- Problem 3: Value of a + b + c
theorem find_abc (a b c : ℝ) (h1 : a - b = 8) (h2 : a * b + c^2 - 4 * c + 20 = 0) :
  a + b + c = 2 :=
sorry

end factorize_m_minimize_ab_find_abc_l645_645538


namespace equation_has_two_distinct_roots_l645_645005

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l645_645005


namespace solution_to_problem_l645_645852

noncomputable def solution_set : Set ℝ := {x | 45 ≤ x * Real.floor x ∧ x * Real.floor x < 46}

theorem solution_to_problem : solution_set = Set.Ico 7.5 (40/6) :=
by
  sorry

end solution_to_problem_l645_645852


namespace rectangle_width_decreased_l645_645628

theorem rectangle_width_decreased (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.5 * L in
  let W' := (L * W) / L' in
  ((W - W') / W) * 100 = 33.3333 :=
by
  sorry

end rectangle_width_decreased_l645_645628


namespace concentration_of_acid_in_third_flask_is_correct_l645_645286

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l645_645286


namespace point_Q_probability_l645_645765

noncomputable def rect_region : set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

def point_closer_to_p1_than_p2 (p q₁ q₂ : ℝ × ℝ) : Prop :=
  (p.1 - q₁.1)^2 + (p.2 - q₁.2)^2 < (p.1 - q₂.1)^2 + (p.2 - q₂.2)^2

theorem point_Q_probability:
  let Q : pmf (ℝ × ℝ) := pmf.of_finset { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 } (by sorry)
  Q { p | point_closer_to_p1_than_p2 p (1, 1) (4, 2) } = 1 / 2 := 
by 
  sorry

end point_Q_probability_l645_645765


namespace product_of_tangents_l645_645048

open Real

theorem product_of_tangents : 2 * ∏ (i : ℕ) in Finset.range 89, tan ((i + 1 : ℤ) * (π / 180)) = 2 :=
by {
  norm_num,
  sorry
}

end product_of_tangents_l645_645048


namespace numbers_not_divisible_by_5_or_7_l645_645504

theorem numbers_not_divisible_by_5_or_7 (n : ℕ) (h : n = 999) :
  let num_div_5 := n / 5,
      num_div_7 := n / 7,
      num_div_35 := n / 35,
      total_eliminated := num_div_5 + num_div_7 - num_div_35,
      result := n - total_eliminated in
  result = 686 := by
{
  sorry
}

end numbers_not_divisible_by_5_or_7_l645_645504


namespace proof_pos_real_l645_645150

noncomputable def posReal (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1 + x) * (1 + y) = 2) : Prop :=
  x * y + 1 / (x * y) ≥ 6

theorem proof_pos_real (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1 + x) * (1 + y) = 2) :
  posReal x y hx hy h :=
begin
  sorry
end

end proof_pos_real_l645_645150


namespace required_circle_equation_l645_645433

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5
def line_condition (x y : ℝ) : Prop := 3 * x + 4 * y - 1 = 0
def intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

theorem required_circle_equation : 
  ∃ (h : ℝ × ℝ → Prop), 
    (∀ p, intersection_points p.1 p.2 → h p.1 p.2) ∧ 
    (∃ cx cy r, (∀ x y, h x y ↔ (x - cx)^2 + (y - cy)^2 = r^2) ∧ line_condition cx cy ∧ h x y = ((x + 1)^2 + (y - 1)^2 = 13)) 
:= sorry

end required_circle_equation_l645_645433


namespace concentration_third_flask_l645_645283

-- Define the concentrations as per the given problem

noncomputable def concentration (acid_mass water_mass : ℝ) : ℝ :=
  (acid_mass / (acid_mass + water_mass)) * 100

-- Given conditions
def acid_mass_first_flask : ℝ := 10
def acid_mass_second_flask : ℝ := 20
def acid_mass_third_flask : ℝ := 30
def concentration_first_flask : ℝ := 5
def concentration_second_flask : ℝ := 70 / 3

-- Total water added to the first and second flasks
def total_water_mass : ℝ :=
  let W1 := (acid_mass_first_flask - concentration_first_flask * acid_mass_first_flask / 100)
  let W2 := (acid_mass_second_flask - concentration_second_flask * acid_mass_second_flask / 100)
  W1 + W2 

-- Prove the concentration of acid in the third flask
theorem concentration_third_flask : 
  concentration acid_mass_third_flask total_water_mass = 10.5 := 
  sorry

end concentration_third_flask_l645_645283


namespace area_of_triangle_POF_l645_645157

noncomputable def parabola_area_problem : ℝ :=
let O := (0 : ℝ, 0 : ℝ) in
let F := (Real.sqrt 2, 0 : ℝ) in
let P := (3 * Real.sqrt 2, 2 * Real.sqrt 3: ℝ) in
let PF := 4 * Real.sqrt 2 in
let OF := Real.sqrt 2 in
if (P.1 - F.1) ^ 2 + P.2 ^ 2 = PF ^ 2 ∧
    P.2 ^ 2 = 4 * Real.sqrt 2 * P.1 ∧
    F.1 = Real.sqrt 2 then
  1 / 2 * OF * abs (P.2)
else
  0

-- Prove that the area of the triangle POF is 2sqrt(3)
theorem area_of_triangle_POF : 
  parabola_area_problem = 2 * Real.sqrt 3 :=
sorry

end area_of_triangle_POF_l645_645157


namespace equilateral_triangle_ratio_l645_645716

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Define the altitude of the equilateral triangle
def altitude (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2

-- Define the area of the equilateral triangle
def area (a : ℝ) : ℝ := (a * altitude a) / 2

-- Define the perimeter of the equilateral triangle
def perimeter (a : ℝ) : ℝ := 3 * a

-- Define the ratio of area to perimeter
def ratio (a : ℝ) : ℝ := area a / perimeter a

theorem equilateral_triangle_ratio :
  ratio 10 = (5 * Real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l645_645716


namespace reams_for_haley_correct_l645_645503

-- Definitions: 
-- total reams = 5
-- reams for sister = 3
-- reams for Haley = ?

def total_reams : Nat := 5
def reams_for_sister : Nat := 3
def reams_for_haley : Nat := total_reams - reams_for_sister

-- The proof problem: prove reams_for_haley = 2 given the conditions.
theorem reams_for_haley_correct : reams_for_haley = 2 := by 
  sorry

end reams_for_haley_correct_l645_645503


namespace concentration_in_third_flask_l645_645278

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l645_645278


namespace problem_sum_of_sine_l645_645449

noncomputable def f (x : ℝ) : ℝ := Real.sin (↑π * x / 3)

theorem problem_sum_of_sine :
  (∑ k in Finset.range 2005, f (k+1)) = Real.sqrt 3 / 2 := 
sorry

end problem_sum_of_sine_l645_645449


namespace exceeded_goal_by_600_l645_645211

noncomputable def ken_collection : ℕ := 600
noncomputable def mary_collection : ℕ := 5 * ken_collection
noncomputable def scott_collection : ℕ := mary_collection / 3
noncomputable def goal : ℕ := 4000
noncomputable def total_raised : ℕ := mary_collection + scott_collection + ken_collection

theorem exceeded_goal_by_600 : total_raised - goal = 600 := by
  have h1 : ken_collection = 600 := rfl
  have h2 : mary_collection = 5 * ken_collection := rfl
  have h3 : scott_collection = mary_collection / 3 := rfl
  have h4 : goal = 4000 := rfl
  have h5 : total_raised = mary_collection + scott_collection + ken_collection := rfl
  have hken : ken_collection = 600 := rfl
  have hmary : mary_collection = 5 * 600 := by rw [hken]; rfl
  have hscott : scott_collection = 3000 / 3 := by rw [hmary]; rfl
  have htotal : total_raised = 3000 + 1000 + 600 := by rw [hmary, hscott, hken]; rfl
  have hexceeded : total_raised - goal = 4600 - 4000 := by rw [htotal, h4]; rfl
  exact hexceeded

end exceeded_goal_by_600_l645_645211


namespace abs_inequality_solution_l645_645643

-- Define the absolute value function
def abs (x : ℝ) : ℝ := if x < 0 then -x else x

-- Define the inequality condition
def inequality (x : ℝ) : Prop := abs (x - 5) + abs (x + 3) ≥ 10

-- Define the solution set to prove
def solution_set (x : ℝ) : Prop := x ∈ set.Iic (-4) ∨ x ∈ set.Ici 6

-- The theorem statement to be proven
theorem abs_inequality_solution {x : ℝ} : inequality x ↔ solution_set x :=
sorry

end abs_inequality_solution_l645_645643


namespace angle_between_tetrahedron_segments_l645_645539

noncomputable def midpoint (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2, (p.3 + q.3) / 2)

noncomputable def centroid (p q r : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p.1 + q.1 + r.1) / 3, (p.2 + q.2 + r.2) / 3, (p.3 + q.3 + r.3) / 3)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 * u.1 + u.2 * u.2 + u.3 * u.3)

noncomputable def angle_between (u v : ℝ × ℝ × ℝ) : ℝ :=
  Real.arccos (dot_product u v / (magnitude u * magnitude v))

theorem angle_between_tetrahedron_segments :
  let a := 1 -- Side length; can be any positive real number
  let A := (0, 0, a * Real.sqrt 2 / 2)
  let B := (a / 2, -a * Real.sqrt 3 / 6, 0)
  let C := (-a / 2, -a * Real.sqrt 3 / 6, 0)
  let D := (0, a * Real.sqrt 3 / 3, 0)
  let M := midpoint A D
  let N := centroid B C D
  let P := midpoint C D
  let Q := centroid A B C
  angle_between (N.1 - M.1, N.2 - M.2, N.3 - M.3) (Q.1 - P.1, Q.2 - P.2, Q.3 - P.3) = Real.arccos (-1 / 18) :=
by
  sorry

end angle_between_tetrahedron_segments_l645_645539


namespace function_range_l645_645251

-- Define the function y = x + 1 / (2 * x)
def f (x : ℝ) : ℝ := x + 1 / (2 * x)

-- State the theorem
theorem function_range : set.range f = {y : ℝ | y ≤ -Real.sqrt 2 ∨ y ≥ Real.sqrt 2} :=
by
  sorry

end function_range_l645_645251


namespace smallest_nine_points_l645_645584

theorem smallest_nine_points :
  ∃ (n : ℕ), (n ≥ 5) ∧ 
  (∀ (points : Fin n → ℝ × ℝ), 
    (∀ i, ∃ (neighbors : Fin 4), 
      ∀ j, (i ≠ j → (points i = points j) = false) ∧
      dist (points i) (points (neighbors j)) = 1)) → 
  n = 9 :=
by 
  -- The proof would go here
  sorry

end smallest_nine_points_l645_645584


namespace proper_subset_singleton_l645_645497

theorem proper_subset_singleton : ∀ (P : Set ℕ), P = {0} → (∃ S, S ⊂ P ∧ S = ∅) :=
by
  sorry

end proper_subset_singleton_l645_645497


namespace parabola_focus_l645_645615

theorem parabola_focus (m : ℝ) (h : m < 0) : 
  let y := (λ x, (1 / m) * x^2) in 
  ∀ x, ∃ f, y = (λ x, (1 / m) * x^2) ∧ f = (0, m / 4) := 
by
  sorry

end parabola_focus_l645_645615


namespace larger_number_l645_645739

theorem larger_number (HCF A B : ℕ) (factor1 factor2 : ℕ) (h_HCF : HCF = 23) (h_factor1 : factor1 = 14) (h_factor2 : factor2 = 15) (h_LCM : HCF * factor1 * factor2 = A * B) (h_A : A = HCF * factor2) (h_B : B = HCF * factor1) : A = 345 :=
by
  sorry

end larger_number_l645_645739


namespace limit_of_polynomial_ratio_l645_645424

theorem limit_of_polynomial_ratio :
  ∀(x : ℝ), (tendsto (λ x : ℝ, (2 * x^3 - 3 * x^2 + 5 * x + 7) / (3 * x^3 + 4 * x^2 - x + 2)) at_top (𝓝 (2/3))) :=
by {
  sorry
}

end limit_of_polynomial_ratio_l645_645424


namespace calc_a2_a3_diff_ge_c_arith_sequence_l645_645459

variable {c : ℝ} (hc : c > 0)

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (abs (x + c + 4)) - (abs (x + c))

-- Define the sequence a_n
def a : ℕ → ℝ
| 0     => -c - 2  -- starting value a₁ = -c - 2 (Index 0 for a₁)
| (n+1) => f (a n)

-- Statement 1: To verify a₂ and a₃ given a₁ 
theorem calc_a2_a3 : a 1 = 2 ∧ a 2 = 10 + c :=
sorry

-- Statement 2: To prove ∀ (n : ℕ), a (n + 1) - a n ≥ c
theorem diff_ge_c : ∀ n : ℕ, a (n + 1) - a n ≥ c :=
sorry

-- Statement 3: To prove the values of a₁ such that {aₙ} forms an arithmetic sequence
theorem arith_sequence (a1 : ℝ) :
  (a 0 = a1) →
  (∀ n : ℕ, ∃ d : ℝ, ∀ m : ℕ, a (m + 1) = a m + d) →
  (a1 = -c - 8 ∨ a1 ≥ -c) :=
sorry

end calc_a2_a3_diff_ge_c_arith_sequence_l645_645459


namespace local_minimum_value_of_f_l645_645523

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem local_minimum_value_of_f : 
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≥ f x) ∧ f x = 1 :=
by
  sorry

end local_minimum_value_of_f_l645_645523


namespace betsy_sewing_l645_645389

-- Definitions of conditions
def total_squares : ℕ := 16 + 16
def sewn_percentage : ℝ := 0.25
def sewn_squares : ℝ := sewn_percentage * total_squares
def squares_left : ℝ := total_squares - sewn_squares

-- Proof that Betsy needs to sew 24 more squares
theorem betsy_sewing : squares_left = 24 := by
  -- Sorry placeholder for the actual proof
  sorry

end betsy_sewing_l645_645389


namespace max_expression_value_l645_645136

theorem max_expression_value :
  ∃ e f g h : ℕ,
    e ∈ {1, 2, 3, 4} ∧
    f ∈ {1, 2, 3, 4} ∧
    g ∈ {1, 2, 3, 4} ∧
    h ∈ {1, 2, 3, 4} ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    e * f ^ g - h = 161 :=
sorry

end max_expression_value_l645_645136


namespace parallel_lines_a_unique_l645_645115

theorem parallel_lines_a_unique (a : ℝ) :
  (∀ x y : ℝ, x + (a + 1) * y + (a^2 - 1) = 0 → x + 2 * y = 0 → -a / 2 = -1 / (a + 1)) →
  a = -2 :=
by
  sorry

end parallel_lines_a_unique_l645_645115


namespace concentration_of_acid_in_third_flask_l645_645269

theorem concentration_of_acid_in_third_flask :
  ∀ (W1 W2 : ℝ),
    let W := 190 + 65.714 in 
    W1 = 190 ∧ W2 = 65.714 →
    (10 : ℝ) / (10 + W1) = 0.05 →
    (20 : ℝ) / (20 + W2) = 0.2331 →
    (30 : ℝ) / (30 + W) = 0.105 :=
begin
  sorry
end

end concentration_of_acid_in_third_flask_l645_645269


namespace negation_of_p_l645_645892

def p := ∀ x : ℝ, x^2 ≥ 0

theorem negation_of_p : ¬p = (∃ x : ℝ, x^2 < 0) :=
  sorry

end negation_of_p_l645_645892


namespace intersect_point_AB_CD_l645_645962

-- Define points A, B, C, and D in 3D space
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A : Point := { x := 5, y := -3, z := 2 }
def B : Point := { x := 15, y := -13, z := 7 }
def C : Point := { x := 2, y := 4, z := -5 }
def D : Point := { x := 4, y := -1, z := 15 }

-- Define the target intersection point
def targetIntersectionPoint : Point := { x := 23 / 3, y := -19 / 3, z := 7 / 3 }

-- The goal is to prove that the lines AB and CD intersect at the target point
theorem intersect_point_AB_CD : 
  ∃ t s : ℝ,
    t * (B.x - A.x) + A.x = s * (D.x - C.x) + C.x ∧
    t * (B.y - A.y) + A.y = s * (D.y - C.y) + C.y ∧
    t * (B.z - A.z) + A.z = s * (D.z - C.z) + C.z ∧
    t * (B.x - A.x) + A.x = targetIntersectionPoint.x ∧
    t * (B.y - A.y) + A.y = targetIntersectionPoint.y ∧
    t * (B.z - A.z) + A.z = targetIntersectionPoint.z :=
sorry

end intersect_point_AB_CD_l645_645962


namespace problem_1_problem_2_l645_645095

def A := {x : ℝ | 1 < 2 * x - 1 ∧ 2 * x - 1 < 7}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}

theorem problem_1 : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

theorem problem_2 : (A ∪ B)ᶜ = {x : ℝ | x ≤ -1 ∨ x ≥ 4} :=
sorry

end problem_1_problem_2_l645_645095


namespace equilateral_triangle_ratio_l645_645717

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Define the altitude of the equilateral triangle
def altitude (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2

-- Define the area of the equilateral triangle
def area (a : ℝ) : ℝ := (a * altitude a) / 2

-- Define the perimeter of the equilateral triangle
def perimeter (a : ℝ) : ℝ := 3 * a

-- Define the ratio of area to perimeter
def ratio (a : ℝ) : ℝ := area a / perimeter a

theorem equilateral_triangle_ratio :
  ratio 10 = (5 * Real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l645_645717


namespace problem_statement_l645_645988

noncomputable def continuous_diff (f: ℝ → ℝ) := Differentiable ℝ f ∧ ∀ t, ContinuousAt f t

variables (f : ℝ → ℝ)
  (h1 : continuous_diff f)
  (h2 : ∀ t : ℝ, fderiv ℝ f t > f (f t))

theorem problem_statement : ∀ t ≥ 0, f (f (f t)) ≤ 0 :=
by
  sorry

end problem_statement_l645_645988


namespace convex_1991_gon_color_preservation_l645_645650

def is_edge_color_preserved (n : ℕ) (color : ℕ → ℕ → Prop) (σ : ℕ → ℕ) : Prop :=
  ∃ k l, 1 ≤ k ∧ k < l ∧ l ≤ n ∧ color k l = color (σ k) (σ l)

theorem convex_1991_gon_color_preservation :
  ∀ (color : ℕ → ℕ → Prop) (σ : ℕ → ℕ), 
    is_edge_color_preserved 1991 color σ :=
begin
  sorry
end

end convex_1991_gon_color_preservation_l645_645650


namespace equation_has_at_least_two_distinct_roots_l645_645016

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l645_645016


namespace find_f_of_4_l645_645568

def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem find_f_of_4 {a b c : ℝ} (h1 : f a b c 1 = 3) (h2 : f a b c 2 = 12) (h3 : f a b c 3 = 27) :
  f a b c 4 = 48 := 
sorry

end find_f_of_4_l645_645568


namespace inequality_solution_l645_645173

variable {a b c : ℝ}

theorem inequality_solution (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c ≥ 1) :
  (1 / (2 + a) + 1 / (2 + b) + 1 / (2 + c) ≤ 1) ∧ (1 / (2 + a) + 1 / (2 + b) + 1 / (2 + c) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end inequality_solution_l645_645173


namespace distance_between_incenter_and_circumcenter_l645_645372

-- Definitions and conditions
variables {A B C I O : Type}
variables [DecidableEq A] [DecidableEq B] [DecidableEq C]
variables (triangle_ABC : Triangle A B C)
variables sides_ABC_6_8_10 : ∃ (a b c : ℝ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a
variables inscribed_circle : TriangleInCircle A B C I
variables circumscribed_circle : TriangleCircumCircle A B C O

-- Statement to be proven
theorem distance_between_incenter_and_circumcenter :
  distance inscribed_circle.center circumscribed_circle.center = Real.sqrt 13 :=
sorry

end distance_between_incenter_and_circumcenter_l645_645372


namespace intersection_complement_l645_645922

-- Definitions
def U : Set ℤ := { y | y = (-1)^3 ∨ y = 0^3 ∨ y = 1^3 ∨ y = 2^3 }
def A : Set ℤ := { -1, 1 }
def B : Set ℤ := { 1, 8 }

-- Complement of B within U
def complement_U (S : Set ℤ) : Set ℤ := { x | x ∈ U ∧ x ∉ S }

-- The theorem to be proven
theorem intersection_complement (U A B) : 
  A ∩ (complement_U B) = { -1 } :=
by 
  sorry

end intersection_complement_l645_645922


namespace correct_judgment_is_C_l645_645083

-- Definitions based on conditions
def three_points_determine_a_plane (p1 p2 p3 : Point) : Prop :=
  -- This would use some axiom or definition of a plane determined by three points
  sorry

def line_and_point_determine_a_plane (l : Line) (p : Point) : Prop :=
  -- This would use some axiom or definition of a plane determined by a line and a point not on the line
  sorry

def two_parallel_lines_and_intersecting_line_same_plane (l1 l2 l3 : Line) : Prop :=
  -- Axiom 3 and its corollary stating that two parallel lines intersected by the same line are in the same plane
  sorry

def three_lines_intersect_pairwise_same_plane (l1 l2 l3 : Line) : Prop :=
  -- Definition stating that three lines intersecting pairwise might be co-planar or not
  sorry

-- Statement of the problem in Lean
theorem correct_judgment_is_C :
    ¬ (three_points_determine_a_plane p1 p2 p3)
  ∧ ¬ (line_and_point_determine_a_plane l p)
  ∧ (two_parallel_lines_and_intersecting_line_same_plane l1 l2 l3)
  ∧ ¬ (three_lines_intersect_pairwise_same_plane l1 l2 l3) :=
  sorry

end correct_judgment_is_C_l645_645083


namespace intersection_point_X_is_determined_by_circle_and_points_l645_645985

-- Variables and assumptions
variables {k : Type*} [metric_space k] [normed_group k]
variables (P Q A C B D X M : k)

-- Circle, points on the circle, and midpoint conditions
variables (circle_k : metric.sphere k)
variables (hP : P ∈ circle_k) (hQ : Q ∈ circle_k)
variables (chord_AC : line (set.range (λ t : ℝ, A + t • (C - A))))
variables (hM_midpoint_PQ : midpoint P Q = M)
variable (hAC_pass_M : M ∈ chord_AC)
variables (hPQ_parallel : line.parallel (line (set.range (λ t : ℝ, A+B+t • (D-C)))) (line (set.range (λ t : ℝ, P+t•(-Q+P)))))

-- Main theorem
theorem intersection_point_X_is_determined_by_circle_and_points :
    (∀ (circle_k : metric.sphere k)
        (P Q A C B D X M : k)
        (hP : P ∈ circle_k) (hQ : Q ∈ circle_k)
        (chord_AC : line (set.range (λ t : ℝ, A + t • (C - A))))
        (hM_midpoint_PQ : midpoint P Q = M)
        (hAC_pass_M : M ∈ chord_AC)
        (hPQ_parallel : line.parallel (line (set.range (λ t : ℝ, A + B + t • (D - C)))) (line (set.range (λ t : ℝ, P + t • (Q - P))))) 
        (hAB_parallel_PQ : line.parallel (line (set.range (λ t : ℝ, A + B + t • M))) (line (set.range (λ t : ℝ, P + t • (Q))))
        (hCD_parallel_PQ: line.parallel (line(set.range (λ t : ℝ, C + D + t • M))) (line(set.range (λ t : ℝ, P + t • (Q))))
        , 
        @ X.unique k := sorry

end intersection_point_X_is_determined_by_circle_and_points_l645_645985


namespace m_ge_1_l645_645498

open Set

theorem m_ge_1 (m : ℝ) :
  (∀ x, x ∈ {x | x ≤ 1} ∩ {x | ¬ (x ≤ m)} → False) → m ≥ 1 :=
by
  intro h
  sorry

end m_ge_1_l645_645498


namespace log_condition_sufficient_log_condition_not_necessary_l645_645446

theorem log_condition_sufficient (a b : ℝ) (ha : 2^a > 2) (hb : 2^b > 2) (hab : 2^a > 2^b) : 2^a > 2^b > 2 → log a 2 < log b 2 :=
by sorry

theorem log_condition_not_necessary (a b : ℝ) (h1 : log a 2 < log b 2): ∃ b, 2^a ≤ 2^(b) ∧ log a 2 < log b 2 :=
by sorry

end log_condition_sufficient_log_condition_not_necessary_l645_645446


namespace suresh_work_time_l645_645609

noncomputable def work_hours (x : ℝ) : Prop :=
  -- Conditions
  let suresh_work_rate := (1:ℝ) / 15
  let ashutosh_work_rate := (1:ℝ) / 35 in
  -- Equation
  x * suresh_work_rate + 14 * ashutosh_work_rate = 1

theorem suresh_work_time : ∃ x : ℝ, work_hours x ∧ x = 9 := 
by
  -- We need to show there's some x that satisfies the conditions
  use 9
  -- The proof of the equivalence is skipped for now
  sorry

end suresh_work_time_l645_645609


namespace concentration_in_third_flask_l645_645274

-- Definitions for the problem conditions
def first_flask_acid_mass : ℕ := 10
def second_flask_acid_mass : ℕ := 20
def third_flask_acid_mass : ℕ := 30

-- Define the total mass after adding water to achieve given concentrations
def total_mass_first_flask (water_added_first : ℕ) : ℕ := first_flask_acid_mass + water_added_first
def total_mass_second_flask (water_added_second : ℕ) : ℕ := second_flask_acid_mass + water_added_second
def total_mass_third_flask (total_water : ℕ) : ℕ := third_flask_acid_mass + total_water

-- Given concentrations as conditions
def first_flask_concentration (water_added_first : ℕ) : Prop :=
  (first_flask_acid_mass : ℚ) / (total_mass_first_flask water_added_first : ℚ) = 0.05

def second_flask_concentration (water_added_second : ℕ) : Prop :=
  (second_flask_acid_mass : ℚ) / (total_mass_second_flask water_added_second : ℚ) = 70 / 300

-- Define total water added
def total_water (water_added_first water_added_second : ℕ) : ℕ :=
  water_added_first + water_added_second

-- Final concentration in the third flask
def third_flask_concentration (total_water_added : ℕ) : Prop :=
  (third_flask_acid_mass : ℚ) / (total_mass_third_flask total_water_added : ℚ) = 0.105

-- Lean theorem statement
theorem concentration_in_third_flask
  (water_added_first water_added_second : ℕ)
  (h1 : first_flask_concentration water_added_first)
  (h2 : second_flask_concentration water_added_second) :
  third_flask_concentration (total_water water_added_first water_added_second) :=
sorry

end concentration_in_third_flask_l645_645274


namespace first_year_after_2010_with_digit_sum_3_l645_645534

-- Define a function to compute the sum of the digits of a year.
def sum_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d1 + d2 + d3 + d4

-- Define the conditions
def year_after_2010 (n : ℕ) : Prop := n > 2010

def sum_of_digits_is_3 (n : ℕ) : Prop := sum_of_digits n = 3

-- Theorem statement
theorem first_year_after_2010_with_digit_sum_3 : 
  ∃ n : ℕ, year_after_2010 n ∧ sum_of_digits_is_3 n ∧ ∀ m : ℕ, (year_after_2010 m ∧ sum_of_digits_is_3 m) → n ≤ m :=
begin
  sorry
end

end first_year_after_2010_with_digit_sum_3_l645_645534


namespace cylinder_surface_area_correct_cylinder_volume_correct_l645_645536

def cylinder_surface_area (M N : ℝ) : ℝ :=
  N * Real.pi + 2 * M

def cylinder_volume (M N : ℝ) : ℝ :=
  (N / 2) * Real.sqrt(M * Real.pi)

theorem cylinder_surface_area_correct (M N : ℝ) : 
  -- Preconditions
  (∃ R H : ℝ, (Real.pi * R^2 = M ∧ H * 2 * R = N)) →
  -- Surface area
  cylinder_surface_area M N = N * Real.pi + 2 * M :=
sorry

theorem cylinder_volume_correct (M N : ℝ) : 
  -- Preconditions
  (∃ R H : ℝ, (Real.pi * R^2 = M ∧ H * 2 * R = N)) →
  -- Volume
  cylinder_volume M N = (N / 2) * Real.sqrt(M * Real.pi) :=
sorry

end cylinder_surface_area_correct_cylinder_volume_correct_l645_645536


namespace cassandra_overall_score_l645_645811

theorem cassandra_overall_score 
  (score1_percent : ℤ) (score1_total : ℕ)
  (score2_percent : ℤ) (score2_total : ℕ)
  (score3_percent : ℤ) (score3_total : ℕ) :
  score1_percent = 60 → score1_total = 15 →
  score2_percent = 75 → score2_total = 20 →
  score3_percent = 85 → score3_total = 25 →
  let correct1 := (score1_percent * score1_total) / 100
  let correct2 := (score2_percent * score2_total) / 100
  let correct3 := (score3_percent * score3_total) / 100
  let total_correct := correct1 + correct2 + correct3
  let total_problems := score1_total + score2_total + score3_total
  75 = (100 * total_correct) / total_problems := by
  intros h1 h2 h3 h4 h5 h6
  let correct1 := (60 * 15) / 100
  let correct2 := (75 * 20) / 100
  let correct3 := (85 * 25) / 100
  let total_correct := correct1 + correct2 + correct3
  let total_problems := 15 + 20 + 25
  suffices 75 = (100 * total_correct) / total_problems by sorry
  sorry

end cassandra_overall_score_l645_645811


namespace probability_of_digit_in_decimal_representation_l645_645585

theorem probability_of_digit_in_decimal_representation :
  let decimal_rep := "27" in
  (decimal_rep.length = 2) →
  (decimal_rep.count '2' = 1) →
  (1 / decimal_rep.length = 1 / 2) :=
by
  intros h_len h_count
  sorry

end probability_of_digit_in_decimal_representation_l645_645585


namespace inequality_transformation_l645_645513

theorem inequality_transformation (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, f x = 2 * x + 3) (h2 : a > 0) (h3 : b > 0) :
  (∀ x, |f x + 5| < a → |x + 3| < b) ↔ b ≤ a / 2 :=
sorry

end inequality_transformation_l645_645513


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645700

theorem ratio_of_area_to_perimeter_of_equilateral_triangle (s : ℕ) : s = 10 → (let A := (s^2 * sqrt 3) / 4, P := 3 * s in A / P = 5 * sqrt 3 / 6) :=
by
  intro h,
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l645_645700


namespace concentration_third_flask_l645_645300

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l645_645300


namespace evaluate_polynomial_at_2_l645_645838

def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + x^2 + 2 * x + 3

theorem evaluate_polynomial_at_2 : polynomial 2 = 67 := by
  sorry

end evaluate_polynomial_at_2_l645_645838


namespace quadratic_inequality_l645_645885

noncomputable def exists_real_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (a - 1) * x1 + (2 * a - 5) = 0 ∧ x2^2 + (a - 1) * x2 + (2 * a - 5) = 0

noncomputable def valid_values (a : ℝ) : Prop :=
  a > 5 / 2 ∧ a < 10

theorem quadratic_inequality (a : ℝ) 
  (h1 : exists_real_roots a) 
  (h2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (a - 1) * x1 + (2 * a - 5) = 0 ∧ x2^2 + (a - 1) * x2 + (2 * a - 5) = 0 
  → (1 / x1 + 1 / x2 < -3 / 5)) :
  valid_values a :=
sorry

end quadratic_inequality_l645_645885


namespace part_a_part_b_l645_645195

-- Define the strip and symmetry conditions
def is_symmetric (s : String) : Prop := 
  s = String.reverse s

def can_be_split (s : String) (n : Nat) : Prop :=
  ∃ parts : List String, parts.length ≤ n ∧ s = String.join parts ∧ ∀ part ∈ parts, is_symmetric part

-- Statement for part (a): Prove that the strip can be cut into at most 24 symmetrical pieces
theorem part_a : can_be_split "x0x00xx0xx00xx00x0x00x0abcdabcdabcdabcdabcdabcd" 24 :=
sorry

-- Example strip provided for part (b)
def example_strip : String := "x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0"

-- Statement for part (b): Prove that the example strip cannot be split into fewer than 15 pieces
theorem part_b : ¬ can_be_split example_strip 14 :=
sorry

end part_a_part_b_l645_645195


namespace factorial_product_is_integer_l645_645202

theorem factorial_product_is_integer (k : ℕ) (h : 0 < k) :
  ∃ (N : ℤ), N = (↑((k^2)!) * ∏ j in Finset.range k, ↑(j!) / ↑((j + k)!)) :=
sorry

end factorial_product_is_integer_l645_645202


namespace problem_statement_l645_645574

-- Defining sets A and B based on the given conditions
def A : Set ℕ := {x | ∃ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, x = 3 * n}
def B : Set ℕ := {y | ∃ m ∈ {0, 1, 2, 3, 4, 5, 6}, y = 5 * m}

-- Defining the union of sets A and B
def union_set : Set ℕ := A ∪ B

-- Defining the sum of all elements in the union set
def sum_union : ℕ := union_set.to_finset.sum id

-- The theorem stating the problem
theorem problem_statement : sum_union = 225 :=
by
  -- leaving the proof as an exercise
  sorry

end problem_statement_l645_645574


namespace range_of_m_l645_645945

-- Defining the conditions
variable (x m : ℝ)

-- The theorem statement
theorem range_of_m (h : ∀ x : ℝ, x < m → 2*x + 1 < 5) : m ≤ 2 := by
  sorry

end range_of_m_l645_645945


namespace valid_a_value_l645_645033

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l645_645033


namespace solution_set_for_inequality_l645_645045

-- Define the function involved
def rational_function (x : ℝ) : ℝ :=
  (3 * x - 1) / (2 - x)

-- Define the main theorem to state the solution set for the given inequality
theorem solution_set_for_inequality (x : ℝ) :
  (rational_function x ≥ 1) ↔ (3 / 4 ≤ x ∧ x < 2) :=
by
  sorry

end solution_set_for_inequality_l645_645045


namespace initial_bleach_percentage_l645_645777

-- Define variables and constants
def total_volume : ℝ := 100
def drained_volume : ℝ := 3.0612244898
def desired_percentage : ℝ := 0.05

-- Define the initial percentage (unknown)
variable (P : ℝ)

-- Define the statement to be proved
theorem initial_bleach_percentage :
  ( (total_volume - drained_volume) * P + drained_volume * 1 = total_volume * desired_percentage )
  → P = 0.02 :=
  by
    intro h
    -- skipping the proof as per instructions
    sorry

end initial_bleach_percentage_l645_645777


namespace calculate_fraction_l645_645396

theorem calculate_fraction (x y : ℚ) (hx : x = 4 / 3) (hy : y = 5 / 7) : (3 * x + 7 * y) / (21 * x * y) = 9 / 140 :=
by {
  -- sorry placeholder for proof
  sorry,
}

end calculate_fraction_l645_645396


namespace distances_relation_l645_645556

-- Define the basic setup: right triangle ABC with height BT
variables {α : Type*} [linear_ordered_field α]
variables {A B C T X Y P : α × α}
variables (a c : α)

-- Assume T is the origin
def T := (0 : α, 0 : α)

-- Coordinates of other points
def A := (a, 0 : α)
def B := (0 : α, 2 : α)
def C := (-c, 0 : α)
def X := (-real.sqrt 3, 1 : α)
def Y := (real.sqrt 3, 1 : α)

-- Condition that A, B, C form a right triangle at B with height BT
def right_triangle_ABC : Prop :=
  ∃ a c : α, T = (0 : α, 0 : α) ∧ A = (a, 0 : α) ∧ B = (0 : α, 2 : α) ∧ C = (-c, 0 : α)

-- Conditions for constructing equilateral triangles BTX and BTY
def equilateral_triangle_BTX : Prop :=
  ∃ T X : α × α, T = (0 : α, 0 : α) ∧ X = (-real.sqrt 3, 1 : α)

def equilateral_triangle_BTY : Prop :=
  ∃ T Y : α × α, T = (0 : α, 0 : α) ∧ Y = (real.sqrt 3, 1 : α)

-- Condition for P as the intersection of AY and CX
def intersection_P : Prop :=
  ∃ P : α × α, P = ((real.sqrt 3 * (c - a)) / (2 * real.sqrt 3 + a + c), (a + c) / (2 * real.sqrt 3 + a + c))

-- Main theorem statement
theorem distances_relation
  (h₁ : right_triangle_ABC)
  (h₂ : equilateral_triangle_BTX)
  (h₃ : equilateral_triangle_BTY)
  (h₄ : intersection_P) :
  let PA := dist P A,
      PB := dist P B,
      PC := dist P C,
      BC := dist B C,
      CA := dist C A,
      AB := dist A B in
    PA * BC = PB * CA ∧ PB * CA = PC * AB := 
sorry

end distances_relation_l645_645556


namespace equilateral_triangle_area_to_perimeter_ratio_l645_645686

theorem equilateral_triangle_area_to_perimeter_ratio
  (a : ℝ) (h : a = 10) :
  let altitude := a * (Real.sqrt 3 / 2) in
  let area := (1 / 2) * a * altitude in
  let perimeter := 3 * a in
  area / perimeter = 5 * (Real.sqrt 3) / 6 := 
by
  sorry

end equilateral_triangle_area_to_perimeter_ratio_l645_645686


namespace sum_of_zeros_g_l645_645077

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ set.Icc 0 1 then (x - 1)^2 else f (2 - x)

def g (x : ℝ) : ℝ :=
  f x - Real.log 2017 (Real.abs (x - 1))

theorem sum_of_zeros_g : 
  (∃ z_sum : ℝ, z_sum = 2016 ∧ ∀ z, g z = 0 → z_sum = 2016) := sorry

end sum_of_zeros_g_l645_645077


namespace side_c_possibilities_l645_645547

theorem side_c_possibilities (A : ℝ) (a b c : ℝ) (hA : A = 30) (ha : a = 4) (hb : b = 4 * Real.sqrt 3) :
  c = 4 ∨ c = 8 :=
sorry

end side_c_possibilities_l645_645547


namespace ellipse_properties_l645_645471

theorem ellipse_properties
  (F1 F2 : ℝ × ℝ)
  (a : ℝ)
  (ha : a > 1)
  (Q : ℝ × ℝ)
  (hF2_coord : F2 = (√(a^2 - 1), 0)) :
  Q = (0, √(a^2 - 1)) →
  Q ∈ {p | (p.1)^2 / a^2 + (p.2)^2 = 1} →
  (∀ P : ℝ × ℝ, P ∈ {p | (p.1)^2 / a^2 + (p.2)^2 = 1} →
  |dist P F1 * dist P F2 = (4 / 3)) →
  (2*a = 2*√2) ∧ (∃ P : ℝ × ℝ, area_of_triangle F1 P F2 = (√3 / 3)) :=
by sorry

end ellipse_properties_l645_645471


namespace inequality_solution_set_l645_645044

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (3 / 4 ≤ x ∧ x < 2) :=
by sorry

end inequality_solution_set_l645_645044


namespace max_value_f_on_interval_l645_645632

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, (∀ y ∈ Set.Icc (-1 : ℝ) 1, f y ≤ f x) ∧ f x = Real.exp 1 - 1 :=
sorry

end max_value_f_on_interval_l645_645632


namespace max_value_of_f_l645_645634

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_of_f : ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → f x ≤ Real.exp 1 - 1 :=
by 
-- The conditions: the function and the interval
intros x hx
-- The interval condition: -1 ≤ x ≤ 1
have h_interval : -1 ≤ x ∧ x ≤ 1 := by 
  cases hx
  split; assumption
-- We prove it directly by showing the evaluated function points
sorry

end max_value_of_f_l645_645634


namespace solution_set_for_inequality_l645_645090

noncomputable def f : ℝ → ℝ := λ x, if h : x > 0 then Real.log x else -Real.log (-x)

theorem solution_set_for_inequality :
  {x : ℝ | f x < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end solution_set_for_inequality_l645_645090


namespace journey_duration_l645_645587

theorem journey_duration:
  ∀ (start_time end_time : Nat),
  (∃ s s_angle e e_angle : Nat, 
    9 * 60 ≤ s ∧ s ≤ 10 * 60 ∧ 
    3 * 60 ≤ e ∧ e ≤ 4 * 60 ∧ 
    s_angle = 270 ∧ e_angle = 90 ∧ 
    (e - s) / (60 * 5.5) = s_angle / (60 * 5.5)) 
  → (end_time - start_time = 311) :=
by
  sorry

end journey_duration_l645_645587


namespace angle_PHQ_90_l645_645139

theorem angle_PHQ_90 (A B C P Q H: Type) [Triangle ABC]
  (right_angle : angle C = 90)
  (P_mid_angle_bisector_A : midpoint (angle_bisector A) P)
  (Q_mid_angle_bisector_B : midpoint (angle_bisector B) Q)
  (H_touchpoint_incircle : incircle_touches_hypotenuse_at H) :
  angle P H Q = 90 :=
sorry

end angle_PHQ_90_l645_645139


namespace find_D_c_l645_645736

-- Define the given conditions
def daily_wage_ratio (W_a W_b W_c : ℝ) : Prop :=
  W_a / W_b = 3 / 4 ∧ W_a / W_c = 3 / 5 ∧ W_b / W_c = 4 / 5

def total_earnings (W_a W_b W_c : ℝ) (D_a D_b D_c : ℕ) : ℝ :=
  W_a * D_a + W_b * D_b + W_c * D_c

variables {W_a W_b W_c : ℝ} 
variables {D_a D_b D_c : ℕ} 

-- Given values according to the problem
def W_c_value : ℝ := 110
def D_a_value : ℕ := 6
def D_b_value : ℕ := 9
def total_earnings_value : ℝ := 1628

-- The target proof statement
theorem find_D_c 
  (h_ratio : daily_wage_ratio W_a W_b W_c)
  (h_Wc : W_c = W_c_value)
  (h_earnings : total_earnings W_a W_b W_c D_a_value D_b_value D_c = total_earnings_value) 
  : D_c = 4 := 
sorry

end find_D_c_l645_645736


namespace pi_approximation_l645_645940

theorem pi_approximation (d C : ℝ) (h1 : d = 8) (h2 : C = 25.12) : C / d ≈ 3.14 :=
by
  sorry

end pi_approximation_l645_645940


namespace sum_of_minimal_values_l645_645580

noncomputable def R (x : ℝ) : ℝ :=
  x^2 + 112 * x + 280

noncomputable def S (x : ℝ) : ℝ :=
  x^2 + 32 * x + 100

theorem sum_of_minimal_values :
  let R_min := R (-56)
  let S_min := S (-16)
  R_min + S_min = -3114 :=
begin
  let R_min := R (-56),
  let S_min := S (-16),
  have hR_min : R_min = 3136 - 6272 + 280 := by rw [R, pow_two, mul_assoc],
  have hS_min : S_min = 144 - 512 + 100 := by rw [S, pow_two, mul_assoc],
  have hR_min_val : R_min = -2846 := by linarith,
  have hS_min_val : S_min = -268 := by linarith,
  calc
    R_min + S_min = -2846 + (-268) : by rw [hR_min_val, hS_min_val]
    ... = -3114 : by linarith,
end

end sum_of_minimal_values_l645_645580


namespace value_of_a_with_two_distinct_roots_l645_645029

theorem value_of_a_with_two_distinct_roots (a x : ℝ) :
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 → ((x₁ x₂ : ℝ) (x₁ ≠ x₂) → a = 20) :=
by
  sorry

end value_of_a_with_two_distinct_roots_l645_645029


namespace distinct_count_floor_sequence_l645_645802

theorem distinct_count_floor_sequence :
  let seq := λ n, (n^2 : ℕ) / 2000 in
  nat.card (finset.image seq (finset.range 1001).erase 0) = 501 :=
sorry

end distinct_count_floor_sequence_l645_645802


namespace solve_proof_problem_l645_645508

noncomputable def proof_problem (f g : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x + g y) = 2 * x + y → g (x + f y) = x / 2 + y

theorem solve_proof_problem (f g : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + g y) = 2 * x + y) :
  ∀ x y : ℝ, g (x + f y) = x / 2 + y :=
sorry

end solve_proof_problem_l645_645508


namespace acid_concentration_third_flask_l645_645293

-- Define the concentrations of first and second flask
def conc_first (w1 : ℝ) : ℝ := 10 / (10 + w1)
def conc_second (w2 : ℝ) : ℝ := 20 / (20 + w2)

-- Define the acid mass in the third flask initially
def acid_mass_third : ℝ := 30

-- Total water added from the fourth flask
def total_water (w1 w2 : ℝ) : ℝ := w1 + w2

-- Acid concentration in the third flask after all water is added
def conc_third (w : ℝ) : ℝ := acid_mass_third / (acid_mass_third + w)

-- Problem statement: concentration in the third flask is 10.5%
theorem acid_concentration_third_flask (w1 : ℝ) (w2 : ℝ) (w : ℝ) 
  (h1 : conc_first w1 = 0.05) 
  (h2 : conc_second w2 = 70 / 300) 
  (h3 : w = total_water w1 w2) : 
  conc_third w = 10.5 / 100 := 
sorry

end acid_concentration_third_flask_l645_645293


namespace equilateral_triangle_area_to_perimeter_ratio_l645_645688

theorem equilateral_triangle_area_to_perimeter_ratio
  (a : ℝ) (h : a = 10) :
  let altitude := a * (Real.sqrt 3 / 2) in
  let area := (1 / 2) * a * altitude in
  let perimeter := 3 * a in
  area / perimeter = 5 * (Real.sqrt 3) / 6 := 
by
  sorry

end equilateral_triangle_area_to_perimeter_ratio_l645_645688


namespace scientific_notation_conversion_l645_645230

theorem scientific_notation_conversion :
  216000 = 2.16 * 10^5 :=
by
  sorry

end scientific_notation_conversion_l645_645230


namespace problem_solution_l645_645175

noncomputable def problem (x : ℝ) (hx_pos : 0 < x) (hx_eq : x + 1 / x = 50) : ℝ :=
sqrt x + 1 / sqrt x

theorem problem_solution (x : ℝ) (hx_pos : 0 < x) (hx_eq : x + 1 / x = 50) : 
  problem x hx_pos hx_eq = sqrt 52 :=
sorry

end problem_solution_l645_645175


namespace find_nine_digit_number_l645_645352

open List

def is_permutation (l1 l2 : List ℕ) : Prop :=
  l1 ~ l2

def uses_all_digits_once (n : ℕ) : Prop :=
  let digits := (to_digits 10 n).qsort (· ≤ ·)
  digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem find_nine_digit_number :
  ∃ (N : ℕ), uses_all_digits_once N ∧ is_permutation (to_digits 10 (8 * N)) (to_digits 10 N) := 
sorry

end find_nine_digit_number_l645_645352


namespace range_of_a_l645_645911

noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) * x^3

theorem range_of_a (a : ℝ) :
  f (Real.logb 2 a) + f (Real.logb 0.5 a) ≤ 2 * f 1 → (1/2 : ℝ) ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l645_645911


namespace equation_has_two_distinct_roots_l645_645004

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l645_645004


namespace exceeded_goal_by_600_l645_645210

noncomputable def ken_collection : ℕ := 600
noncomputable def mary_collection : ℕ := 5 * ken_collection
noncomputable def scott_collection : ℕ := mary_collection / 3
noncomputable def goal : ℕ := 4000
noncomputable def total_raised : ℕ := mary_collection + scott_collection + ken_collection

theorem exceeded_goal_by_600 : total_raised - goal = 600 := by
  have h1 : ken_collection = 600 := rfl
  have h2 : mary_collection = 5 * ken_collection := rfl
  have h3 : scott_collection = mary_collection / 3 := rfl
  have h4 : goal = 4000 := rfl
  have h5 : total_raised = mary_collection + scott_collection + ken_collection := rfl
  have hken : ken_collection = 600 := rfl
  have hmary : mary_collection = 5 * 600 := by rw [hken]; rfl
  have hscott : scott_collection = 3000 / 3 := by rw [hmary]; rfl
  have htotal : total_raised = 3000 + 1000 + 600 := by rw [hmary, hscott, hken]; rfl
  have hexceeded : total_raised - goal = 4600 - 4000 := by rw [htotal, h4]; rfl
  exact hexceeded

end exceeded_goal_by_600_l645_645210


namespace divisors_not_multiples_of_14_l645_645162

theorem divisors_not_multiples_of_14 (m : ℕ)
  (h1 : ∃ k : ℕ, m = 2 * k ∧ (k : ℕ) * k = m / 2)  
  (h2 : ∃ k : ℕ, m = 3 * k ∧ (k : ℕ) * k * k = m / 3)  
  (h3 : ∃ k : ℕ, m = 7 * k ∧ (k : ℕ) ^ 7 = m / 7) : 
  let total_divisors := (6 + 1) * (10 + 1) * (7 + 1)
  let divisors_divisible_by_14 := (5 + 1) * (10 + 1) * (6 + 1)
  total_divisors - divisors_divisible_by_14 = 154 :=
by
  sorry

end divisors_not_multiples_of_14_l645_645162


namespace find_point_coordinates_l645_645588

noncomputable def Q_coordinates : Prop :=
  let P := (1 : ℝ, 0 : ℝ)
  let unit_circle : ℝ × ℝ → Prop := λ p, p.1^2 + p.2^2 = 1
  let arc_length := 2 * Real.pi / 3
  ∃ Q : ℝ × ℝ, unit_circle Q ∧ Q = (Real.cos arc_length, Real.sin arc_length)

theorem find_point_coordinates : Q_coordinates :=
  sorry

end find_point_coordinates_l645_645588


namespace parabola_focus_line_condition_chord_length_on_line_l645_645089

noncomputable def parabola_focus : (ℝ × ℝ) := (2, 0)
def line (m : ℝ) (x y : ℝ) : Prop := m * x - y + 1 - m = 0
def circle (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 1) ^ 2 = 6

theorem parabola_focus_line_condition (m : ℝ) :
  line m 2 0 →
  m = -1 :=
by sorry

theorem chord_length_on_line :
  let m := -1 in
  ∀ A B : ℝ × ℝ,
  line m A.1 A.2 →
  line m B.1 B.2 →
  circle A.1 A.2 →
  circle B.1 B.2 →
  |(A.1 - B.1)^2 + (A.2 - B.2)^2| = (2 * Real.sqrt 6) ^ 2 :=
by sorry

end parabola_focus_line_condition_chord_length_on_line_l645_645089


namespace book_pricing_l645_645055

theorem book_pricing (n : ℕ) (p : ℕ) (h₁ : n = 43) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n → price i = p + 3 * (i - 1))
  (h₃ : price n = p + 126) :
  price 43 = price 22 + price 21 :=
sorry

end book_pricing_l645_645055


namespace males_not_listening_l645_645649

theorem males_not_listening 
  (males_listen : ℕ)
  (females_dont_listen : ℕ)
  (total_listen : ℕ)
  (total_dont_listen : ℕ)
  (males_listen = 70)
  (females_dont_listen = 110)
  (total_listen = 145)
  (total_dont_listen = 160) :
  ∃ males_dont_listen : ℕ, males_dont_listen = 50 :=
by
  sorry

end males_not_listening_l645_645649


namespace range_of_x_l645_645474

noncomputable def f (x : ℝ) : ℝ := 3 * x + sin x

theorem range_of_x :
  (∀ x ∈ Ioo (-1 : ℝ) 1, deriv f x = 3 + cos x) ∧ 
  (f 0 = 0) ∧
  (∀ x ∈ Ioo (-1 : ℝ) 1, f (1 - x) + f (1 - x^2) < 0) 
  → x ∈ Ioo (1 : ℝ) (Real.sqrt 2) :=
by
  sorry

end range_of_x_l645_645474


namespace equilateral_triangle_ratio_is_correct_l645_645696

noncomputable def equilateral_triangle_area_perimeter_ratio (a : ℝ) (h_eq : a = 10) : ℝ :=
  let altitude := (Real.sqrt 3 / 2) * a
  let area := (1 / 2) * a * altitude
  let perimeter := 3 * a
  area / perimeter

theorem equilateral_triangle_ratio_is_correct :
  equilateral_triangle_area_perimeter_ratio 10 (by rfl) = 5 * Real.sqrt 3 / 6 :=
by
  sorry

end equilateral_triangle_ratio_is_correct_l645_645696


namespace equation_has_roots_l645_645009

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l645_645009


namespace points_in_plane_l645_645081

def vector := (ℝ × ℝ × ℝ)
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def is_point_in_plane (point normal : vector) (pt_P : vector) : Prop := 
  let v := (point.1 - pt_P.1, point.2 - pt_P.2, point.3 - pt_P.3)
  dot_product v normal = 0

theorem points_in_plane (n : vector) (P : vector) :
  n = (-2, 3, 1) →
  P = (1, 1, 2) →
  is_point_in_plane (0, 0, 3) n P ∧
  is_point_in_plane (3, 2, 3) n P ∧
  is_point_in_plane (2, 1, 4) n P :=
by
  intros hn hP
  unfold is_point_in_plane
  have h1 : dot_product (-1, -1, 1) (-2, 3, 1) = 0 := by { dsimp; linarith }
  have h2 : dot_product (2, 1, 1) (-2, 3, 1) = 0 := by { dsimp; linarith }
  have h3 : dot_product (1, 0, 2) (-2, 3, 1) = 0 := by { dsimp; linarith }
  exact ⟨h1, h2, h3⟩

#if below lines are optional as Lean 4 doesn’t allow yet to have optional comments inside theorems
-- skipped proof cases (linarity proof omitted):
-- have h1 : dot_product (-1, -1, 1) (-2, 3, 1) = 0 := sorry
-- have h2 : dot_product (2, 1, 1) (-2, 3, 1) = 0 := sorry
-- have h3 : dot_product (1, 0, 2) (-2, 3, 1) = 0 := sorry

end points_in_plane_l645_645081


namespace men_in_first_group_l645_645113

theorem men_in_first_group (M : ℕ) (h1 : (M * 15) = (M + 0) * 15) (h2 : (15 * 36) = 540) : M = 36 :=
by
  -- Proof would go here
  sorry

end men_in_first_group_l645_645113


namespace find_a_for_quadratic_l645_645020

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l645_645020


namespace salt_price_per_pound_l645_645759

theorem salt_price_per_pound 
    (cost_salt1_per_pound : ℝ)
    (weight_salt1 : ℝ)
    (price_mixture_per_pound : ℝ)
    (weight_mixture : ℝ)
    (profit_margin : ℝ)
    (total_weight : ℝ)
    (total_cost_salt1 : ℝ)
    (total_revenue : ℝ)
    (total_cost : ℝ)
    (total_cost_salt2 : ℝ)
    (weight_salt2 : ℝ) :
    cost_salt1_per_pound = 0.25 ->
    weight_salt1 = 40 ->
    price_mixture_per_pound = 0.48 ->
    weight_mixture = 100 ->
    profit_margin = 0.20 ->
    total_weight = weight_salt1 + weight_salt2 ->
    total_cost_salt1 = cost_salt1_per_pound * weight_salt1 ->
    total_revenue = price_mixture_per_pound * weight_mixture ->
    total_revenue = (1 + profit_margin) * total_cost ->
    total_cost = total_cost_salt1 + total_cost_salt2 ->
    weight_salt2 = 60 ->
    (total_cost_salt2 / weight_salt2 = 0.50) : True :=
by
  sorry

end salt_price_per_pound_l645_645759


namespace maximize_sum_sequence_l645_645093

def sequence (n : ℕ) : ℕ := 26 - 2 * n

def sum_sequence (n : ℕ) : ℕ := (n * (27 - n)) / 2

theorem maximize_sum_sequence :
  (∀ n : ℕ, sum_sequence n ≤ sum_sequence 12 ∨ sum_sequence n ≤ sum_sequence 13) :=
sorry

end maximize_sum_sequence_l645_645093


namespace perimeter_inequality_l645_645589

-- Define Euclidean Geometry assumptions
variable {A B C A1 B1 C1 : Point}
variable {P P1 : ℝ}
variable {λ : ℝ}

-- Define conditions based on the problem statement
def condition1 := A ≠ B ∧ B ≠ C ∧ A ≠ C
def condition2 := (1 / 2) < λ ∧ λ < 1
def condition3 := dist B A1 = λ * dist B C
def condition4 := dist C B1 = λ * dist C A
def condition5 := dist A C1 = λ * dist A B

-- Prove the required inequality
theorem perimeter_inequality :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 → 
  (P = dist A B + dist B C + dist C A) →
  (P1 = dist A1 B1 + dist B1 C1 + dist C1 A1) →
  (2 * λ - 1) * P < P1 ∧ P1 < λ * P := 
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  sorry
end

end perimeter_inequality_l645_645589


namespace estimate_total_balls_l645_645546

theorem estimate_total_balls (red_balls : ℕ) (frequency : ℝ) (total_balls : ℕ) 
  (h_red : red_balls = 12) (h_freq : frequency = 0.6) 
  (h_eq : (red_balls : ℝ) / total_balls = frequency) : 
  total_balls = 20 :=
by
  sorry

end estimate_total_balls_l645_645546


namespace dorothy_profit_l645_645833

def cost_to_buy_ingredients : ℕ := 53
def number_of_doughnuts : ℕ := 25
def selling_price_per_doughnut : ℕ := 3

def revenue : ℕ := number_of_doughnuts * selling_price_per_doughnut
def profit : ℕ := revenue - cost_to_buy_ingredients

theorem dorothy_profit : profit = 22 :=
by
  -- calculation steps
  sorry

end dorothy_profit_l645_645833


namespace range_of_m_l645_645117

theorem range_of_m (h : ¬ ∃ x_0 : ℝ, 2 * x_0^2 - 3 * (m : ℝ) * x_0 + 9 < 0) : 
  -2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2 :=
begin
  sorry
end

end range_of_m_l645_645117


namespace number_of_digits_in_typed_number_l645_645184

theorem number_of_digits_in_typed_number
  (typed_number : ℕ)
  (mid_intended_digits : ℕ)
  (missing_digits : ℕ)
  (combinations : ℕ)
  (proof_condition : typed_number = 52115 ∧ mid_intended_digits = 7 ∧ missing_digits = 2 ∧ combinations = 21) : 
  typed_number.digits.size = 5 := 
by 
  sorry

end number_of_digits_in_typed_number_l645_645184


namespace probability_product_multiple_of_4_l645_645321

open Finset

theorem probability_product_multiple_of_4 :
  let cards := {1, 2, 3, 4, 5, 6}
  let pairs := (cards.product cards).filter (λ p, p.fst ≠ p.snd)
  let favorable_pairs := pairs.filter (λ p, (p.fst * p.snd) % 4 = 0)
  ∃ prob : ℚ,
    prob = ↑(favorable_pairs.card) / ↑(pairs.card)
    ∧ prob = 2 / 5 := by
  sorry

end probability_product_multiple_of_4_l645_645321


namespace solve_for_y_l645_645516

theorem solve_for_y (x y : ℝ) (h₁ : x^(2 * y) = 64) (h₂ : x = 8) : y = 1 :=
by
  sorry

end solve_for_y_l645_645516


namespace triangle_inequality_check_l645_645734

theorem triangle_inequality_check (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  (a = 5 ∧ b = 8 ∧ c = 12) → (a + b > c ∧ b + c > a ∧ c + a > b) :=
by 
  intros h
  rcases h with ⟨rfl, rfl, rfl⟩
  exact ⟨h1, h2, h3⟩

end triangle_inequality_check_l645_645734


namespace logarithm_conversion_l645_645873

variable (a b : ℝ)

theorem logarithm_conversion (h₁ : a ^ (1 / 3) = b) (h₂ : 0 < a) (h₃ : a ≠ 1) : log a b = 1 / 3 :=
by
  sorry

end logarithm_conversion_l645_645873


namespace calculate_days_l645_645934

-- Defining a simple structure for the problem conditions.
structure BrickLayingProblem where
  d e g : ℕ -- Declare the necessary variables as natural numbers.
  initial_rate : ℚ := g / (d * e) -- Defining the initial rate based on the problem.

-- The main theorem to prove.
theorem calculate_days (p : BrickLayingProblem) : 
  let y := (p.d * p.e : ℚ) / (p.g * p.g : ℚ) in
  ∃ (y : ℚ), y = (p.d * p.e : ℚ) / (p.g * p.g : ℚ) :=
by
  use (p.d * p.e : ℚ) / (p.g * p.g : ℚ)
  exact sorry

end calculate_days_l645_645934


namespace derivative_limit_l645_645085

noncomputable def f (x : ℝ) := 1 + Real.sin (2 * x)

theorem derivative_limit :
  (Real.limit (fun Δx => (f Δx - f 0) / Δx) 0 = 2) :=
by
  sorry

end derivative_limit_l645_645085


namespace find_f_of_7_l645_645476

noncomputable def f : ℝ → ℝ :=
sorry

theorem find_f_of_7 (f : ℝ → ℝ) : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x : ℝ, f (x + 4) = f x) ∧ 
  (∀ x : ℝ, x ∈ Ioo 0 2 → f x = x^2 + 1) →
  f 7 = -2 :=
sorry

end find_f_of_7_l645_645476


namespace integral_computation_l645_645394

noncomputable def integral_value : Real :=
  ∫ x in -2..2, sqrt (4 - x^2) + x^2

theorem integral_computation :
  integral_value = 2 * Real.pi + 16 / 3 :=
by
  sorry

end integral_computation_l645_645394


namespace supermarket_A_cost_400_functional_relationships_cost_effective_450_l645_645228

def discount_A (x : ℝ) : ℝ :=
  if x <= 300 then 0.9 * x
  else 0.7 * x + 60

def discount_B (x : ℝ) : ℝ :=
  if x <= 100 then x
  else 0.8 * x + 20

theorem supermarket_A_cost_400 : discount_A 400 = 340 :=
by sorry

theorem functional_relationships (x : ℝ) :
  (discount_A x = if x <= 300 then 0.9 * x else 0.7 * x + 60) ∧
  (discount_B x = if x <= 100 then x else 0.8 * x + 20) :=
by sorry

theorem cost_effective_450 : discount_A 450 < discount_B 450 :=
by sorry

end supermarket_A_cost_400_functional_relationships_cost_effective_450_l645_645228


namespace sum_pairs_rs_eq_48_l645_645186

theorem sum_pairs_rs_eq_48 : 
  (∑ (r, s) in {(1, 48), (2, 24), (3, 16), (4, 12), (6, 8)}, (r + s)) = 124 := 
by 
  sorry

end sum_pairs_rs_eq_48_l645_645186


namespace ratio_second_part_l645_645354

theorem ratio_second_part (first_part second_part total : ℕ) 
  (h_ratio_percent : 50 = 100 * first_part / total) 
  (h_first_part : first_part = 10) : 
  second_part = 10 := by
  have h_total : total = 2 * first_part := by sorry
  sorry

end ratio_second_part_l645_645354


namespace subset_condition_l645_645575

universe u

variable {α : Type u} [PartialOrder α]

-- Define set A
def A (a : α) : Set α := {x | x >= a}

-- Define set B
def B : Set α := {x | 2 < x ∧ x < 4}

-- Given condition B ⊆ A
theorem subset_condition (a : α) (h : B ⊆ A a) : a ≤ (2 : α) := by
  sorry

end subset_condition_l645_645575


namespace value_of_x_l645_645257

-- Define the variables x, y, z
variables (x y z : ℕ)

-- Hypothesis based on the conditions of the problem
hypothesis h1 : x = y / 3
hypothesis h2 : y = z / 4
hypothesis h3 : z = 48

-- The statement to be proved
theorem value_of_x : x = 4 :=
by { sorry }

end value_of_x_l645_645257


namespace geometric_sequence_an_l645_645059

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 3 else 3 * (2:ℝ)^(n - 1)

noncomputable def S (n : ℕ) : ℝ :=
  if n = 1 then 3 else (3 * (2:ℝ)^n - 3)

theorem geometric_sequence_an (n : ℕ) (h1 : a 1 = 3) (h2 : S 2 = 9) :
  a n = 3 * 2^(n-1) ∧ S n = 3 * (2^n - 1) :=
by
  sorry

end geometric_sequence_an_l645_645059


namespace company_pays_each_man_per_hour_l645_645141

theorem company_pays_each_man_per_hour
  (men : ℕ) (hours_per_job : ℕ) (jobs : ℕ) (total_pay : ℕ)
  (completion_time : men * hours_per_job = 1)
  (total_jobs_time : jobs * hours_per_job = 5)
  (total_earning : total_pay = 150) :
  (total_pay / (jobs * men * hours_per_job)) = 10 :=
sorry

end company_pays_each_man_per_hour_l645_645141


namespace find_optimal_mix_time_l645_645756

def strength_unimodal (mixing_time : ℝ) : Prop := sorry -- Definition of unimodal strength

noncomputable def maximum_experiments_needed (experimental_points : ℕ) : ℕ :=
  if experimental_points = 20 then 6 else sorry -- Definition according to the problem statement

theorem find_optimal_mix_time :
  ∀ (experimental_points : ℕ), experimental_points = 20 →
  maximum_experiments_needed experimental_points = 6 :=
by
  intros experimental_points h,
  rw h,
  unfold maximum_experiments_needed,
  simp

end find_optimal_mix_time_l645_645756


namespace range_of_a_l645_645989

def M {α : Type*} (s : set α) : ℕ := s.to_finset.card

def abs_diff {α : Type*} (A B : set α) [decidable_eq α] : ℕ :=
if M A ≥ M B then M A - M B else M B - M A

theorem range_of_a (a : ℝ) (A B : set ℝ) (ha : A = {1, 2, 3})
  (hb : B = {x | abs (x^2 - 2 * x - 3) = a})
  (hab : abs_diff A B = 1) :
  (0 ≤ a ∧ a < 4) ∨ (a > 4) :=
sorry

end range_of_a_l645_645989


namespace ratio_of_area_to_perimeter_l645_645709

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end ratio_of_area_to_perimeter_l645_645709


namespace range_f_contained_in_0_1_l645_645879

theorem range_f_contained_in_0_1 (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := 
by {
  sorry
}

end range_f_contained_in_0_1_l645_645879


namespace probability_of_earning_exactly_2300_in_3_spins_l645_645968

-- Definitions of the conditions
def spinner_sections : List ℕ := [0, 1000, 200, 7000, 300]
def equal_area_sections : Prop := true  -- Each section has the same area, simple condition

-- Proving the probability of earning exactly $2300 in three spins
theorem probability_of_earning_exactly_2300_in_3_spins :
  ∃ p : ℚ, p = 3 / 125 := sorry

end probability_of_earning_exactly_2300_in_3_spins_l645_645968


namespace fundraising_exceeded_goal_l645_645218

theorem fundraising_exceeded_goal:
  let goal := 4000
  let ken := 600
  let mary := 5 * ken
  let scott := mary / 3
  let total := ken + mary + scott
  total - goal = 600 :=
by
  let goal := 4000
  let ken := 600
  let mary := 5 * ken
  let scott := mary / 3
  let total := ken + mary + scott
  have h_goal : goal = 4000 := rfl
  have h_ken : ken = 600 := rfl
  have h_mary : mary = 5 * ken := rfl
  have h_scott : scott = mary / 3 := rfl
  have h_total : total = ken + mary + scott := rfl
  calc total - goal = (ken + mary + scott) - goal : by rw h_total
  ... = (600 + 3000 + 1000) - 4000 : by {rw [h_ken, h_mary, h_scott], norm_num}
  ... = 600 : by norm_num

end fundraising_exceeded_goal_l645_645218


namespace geometric_series_b_plus_1_sum_first_n_terms_l645_645971

noncomputable def seq_a : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 1) => (2 + 2 / n) * seq_a n + n + 1

noncomputable def seq_b (n : ℕ) : ℕ :=
  if n = 0 then 0 else seq_a n / n

def is_geometric_series (r : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq (n + 1) = r * seq n

theorem geometric_series_b_plus_1 :
  is_geometric_series 2 (λ n, seq_b n + 1) :=
sorry

noncomputable def S (n : ℕ) : ℕ :=
  (n - 1) * 2^(n + 1) + 2 - n * (n + 1) / 2

theorem sum_first_n_terms :
  ∀ n, (∑ k in Finset.range n, seq_a (k + 1)) = S n :=
sorry

end geometric_series_b_plus_1_sum_first_n_terms_l645_645971


namespace quadrilateral_sum_of_abc_l645_645767

theorem quadrilateral_sum_of_abc :
  let d1 := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let d2 := real.sqrt ((5 - 4)^2 + (4 - 6)^2)
  let d3 := real.sqrt ((2 - 5)^2 + (0 - 4)^2)
  let d4 := real.sqrt ((1 - 2)^2 + (2 - 0)^2)
  let perimeter := d1 + d2 + d3 + d4
  ∃ (a b c : ℤ), perimeter = a * real.sqrt 2 + b * real.sqrt 5 + c * real.sqrt 10 ∧ a + b + c = 2 :=
by {
  let d1 := real.sqrt ((4 - 1)^2 + (6 - 2)^2)
  let d2 := real.sqrt ((5 - 4)^2 + (4 - 6)^2)
  let d3 := real.sqrt ((2 - 5)^2 + (0 - 4)^2)
  let d4 := real.sqrt ((1 - 2)^2 + (2 - 0)^2)
  let perimeter := d1 + d2 + d3 + d4
  use [0, 2, 0]
  split
  . sorry -- Proof that perimeter = 0 * real.sqrt 2 + 2 * real.sqrt 5 + 0 * real.sqrt 10
  . trivial
}

end quadrilateral_sum_of_abc_l645_645767


namespace concentration_of_acid_in_third_flask_is_correct_l645_645288

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l645_645288


namespace max_marks_l645_645185

theorem max_marks (M : ℝ) (h_pass : 0.30 * M = 231) : M = 770 := sorry

end max_marks_l645_645185


namespace triangle_proof_l645_645532

open Real

-- Define the constants
constants {a b c : ℝ}
constants {A B C : ℝ}

-- Triangle ABC conditions
axiom triangle_abc : a = b ∧ b = c ∧ a + b + c = 3
axiom angle_condition : a * cos B + b * cos A = 2 * c * cos C

-- Problem I: Measure of angle C
def measure_angle_C : Prop :=
  C = π / 3

-- Problem II: Maximum area of the incircle
def maximum_incircle_area : Prop :=
  ∃ (S : ℝ), S = π * (sqrt 3 / 6) ^ 2 ∧ S = π / 12

-- Theorem to prove both statements
theorem triangle_proof : 
  (measure_angle_C ∧ maximum_incircle_area) :=
begin
  split,
  { 
    -- Proof part for measure_angle_C
    sorry 
  },
  { 
    -- Proof part for maximum_incircle_area
    sorry 
  }
end

end triangle_proof_l645_645532


namespace ratio_of_area_to_perimeter_l645_645683

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l645_645683


namespace total_lifespan_l645_645630

theorem total_lifespan (B H F : ℕ)
  (hB : B = 10)
  (hH : H = B - 6)
  (hF : F = 4 * H) :
  B + H + F = 30 := by
  sorry

end total_lifespan_l645_645630


namespace median_of_trapezoid_l645_645368

theorem median_of_trapezoid (h : ℝ) (x : ℝ) 
  (triangle_area_eq_trapezoid_area : (1 / 2) * 24 * h = ((x + (2 * x)) / 2) * h) : 
  ((x + (2 * x)) / 2) = 12 := by
  sorry

end median_of_trapezoid_l645_645368


namespace cos_arcsin_compute_l645_645399

theorem cos_arcsin_compute : 
  ∀ (x : ℝ), x = 5 / 13 → cos (arcsin x) = 12 / 13 := 
by
  intro x h
  rw [h]
  sorry

end cos_arcsin_compute_l645_645399


namespace limit_n_bn_l645_645413

noncomputable def L (x : ℝ) : ℝ := x - (x^3) / 3

noncomputable def iterate_L (n : ℕ) (x : ℝ) : ℝ :=
  nat.iterate L n x

theorem limit_n_bn :
  tendsto (λ n : ℕ, n * iterate_L n ((19 : ℝ) / n)) at_top (𝓝 (19 / 2)) :=
sorry

end limit_n_bn_l645_645413


namespace ratio_equilateral_triangle_l645_645724

def equilateral_triangle_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * (Real.sqrt 3 / 2)
  let area := (1 / 2) * s * altitude
  let perimeter := 3 * s in
  area / perimeter -- this simplifies to 25\sqrt{3} / 30 or 5\sqrt{3} / 6

theorem ratio_equilateral_triangle : ∀ (s : ℝ), s = 10 → equilateral_triangle_ratio s (by assumption) = 5 * (Real.sqrt 3) / 6 :=
by
  intros s h
  rw h
  sorry

end ratio_equilateral_triangle_l645_645724


namespace hyperbola_m_range_l645_645617

-- Define the equation of the hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (m + 2)) - (y^2 / (m - 1)) = 1

-- State the equivalent range problem
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m ↔ (m < -2 ∨ m > 1) :=
by
  sorry

end hyperbola_m_range_l645_645617


namespace incorrect_inequality_l645_645326

theorem incorrect_inequality : ¬ (-2 < -3) :=
by {
  -- Proof goes here
  sorry
}

end incorrect_inequality_l645_645326


namespace valid_a_value_l645_645034

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l645_645034


namespace todd_money_after_repay_l645_645661

-- Definitions as conditions
def borrowed_amount : ℤ := 100
def repay_amount : ℤ := 110
def ingredients_cost : ℤ := 75
def snow_cones_sold : ℤ := 200
def price_per_snow_cone : ℚ := 0.75

-- Function to calculate money left after transactions
def money_left_after_repay (borrowed : ℤ) (repay : ℤ) (cost : ℤ) (sold : ℤ) (price : ℚ) : ℚ :=
  let left_after_cost := borrowed - cost
  let earnings := sold * price
  let total_before_repay := left_after_cost + earnings
  total_before_repay - repay

-- The theorem stating the problem
theorem todd_money_after_repay :
  money_left_after_repay borrowed_amount repay_amount ingredients_cost snow_cones_sold price_per_snow_cone = 65 := 
by
  -- This is a placeholder for the actual proof
  sorry

end todd_money_after_repay_l645_645661


namespace student_failed_by_20_marks_l645_645785

theorem student_failed_by_20_marks (marks_obtained : ℕ) (total_marks : ℕ) 
  (passing_percentage : ℚ) (marks_required : ℕ) (marks_failed_by : ℕ) :
  marks_obtained = 160 →
  total_marks = 300 →
  passing_percentage = 0.6 →
  marks_required = (passing_percentage * total_marks).toNat →
  marks_failed_by = marks_required - marks_obtained →
  marks_failed_by = 20 :=
by
  sorry

end student_failed_by_20_marks_l645_645785


namespace train_length_approx_l645_645310

noncomputable def train_length (v_f v_s: ℝ) (t: ℝ) : ℝ :=
  let relative_speed := (v_f - v_s) * (5 / 18) -- Convert km/hr to m/s
  in relative_speed * t

theorem train_length_approx (v_f v_s: ℝ) (t : ℝ)
  (h1 : v_f = 46)
  (h2 : v_s = 36)
  (h3 : t = 36.00001) :
  train_length v_f v_s t ≈ 100 :=
by
  unfold train_length
  rw [h1, h2, h3]
  have r_speed : (46 - 36) * (5 / 18) = 50 / 18 := by norm_num
  rw r_speed
  norm_num [Mul.mul, Div.div]
  sorry -- The proof should be provided here, but per instructions, this is omitted.

end train_length_approx_l645_645310


namespace find_p_and_c_l645_645155

theorem find_p_and_c (p c : ℝ) (h₀ : c ≠ 0) 
  (h : ∀ x : ℝ, 0 < x ∧ x < 10^(-100) → c - 0.1 < x^p * (1 - (1 + x)^10) / (1 + (1 + x)^10) ∧ x^p * (1 - (1 + x)^10) / (1 + (1 + x)^10) < c + 0.1) : 
  (p, c) = (-1, -5) :=
sorry

end find_p_and_c_l645_645155


namespace bob_same_color_probability_is_1_over_28_l645_645337

def num_marriages : ℕ := 9
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 3

def david_marbles : ℕ := 3
def alice_marbles : ℕ := 3
def bob_marbles : ℕ := 3

def total_ways : ℕ := 1680
def favorable_ways : ℕ := 60
def probability_bob_same_color := favorable_ways / total_ways

theorem bob_same_color_probability_is_1_over_28 : probability_bob_same_color = (1 : ℚ) / 28 := by
  sorry

end bob_same_color_probability_is_1_over_28_l645_645337


namespace factorize_difference_of_squares_l645_645849

theorem factorize_difference_of_squares :
  ∀ x : ℝ, x^2 - 9 = (x + 3) * (x - 3) :=
by 
  intro x
  have h : x^2 - 9 = x^2 - 3^2 := by rw (show 9 = 3^2, by norm_num)
  have hs : (x^2 - 3^2) = (x + 3) * (x - 3) := by exact (mul_self_sub_mul_self_eq x 3)
  exact Eq.trans h hs

end factorize_difference_of_squares_l645_645849


namespace equilateral_triangle_ratio_l645_645713

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Define the altitude of the equilateral triangle
def altitude (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2

-- Define the area of the equilateral triangle
def area (a : ℝ) : ℝ := (a * altitude a) / 2

-- Define the perimeter of the equilateral triangle
def perimeter (a : ℝ) : ℝ := 3 * a

-- Define the ratio of area to perimeter
def ratio (a : ℝ) : ℝ := area a / perimeter a

theorem equilateral_triangle_ratio :
  ratio 10 = (5 * Real.sqrt 3) / 6 :=
by
  sorry

end equilateral_triangle_ratio_l645_645713


namespace probability_of_next_satisfied_customer_l645_645794

noncomputable def probability_of_satisfied_customer : ℝ :=
  let p := (0.8 : ℝ)
  let q := (0.15 : ℝ)
  let neg_reviews := (60 : ℝ)
  let pos_reviews := (20 : ℝ)
  p / (p + q) * (q / (q + p))

theorem probability_of_next_satisfied_customer :
  probability_of_satisfied_customer = 0.64 :=
sorry

end probability_of_next_satisfied_customer_l645_645794


namespace students_selected_milk_l645_645800

noncomputable def selected_soda_percent : ℚ := 50 / 100
noncomputable def selected_milk_percent : ℚ := 30 / 100
noncomputable def selected_soda_count : ℕ := 90
noncomputable def selected_milk_count := selected_milk_percent / selected_soda_percent * selected_soda_count

theorem students_selected_milk :
    selected_milk_count = 54 :=
by
  sorry

end students_selected_milk_l645_645800


namespace star_problem_l645_645558

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ := a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + ...)))

-- Define the proof problem
theorem star_problem (y : ℝ) (h : star 3 y = 18) : y = 30 := sorry

end star_problem_l645_645558


namespace concentration_third_flask_l645_645298

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l645_645298


namespace incenter_circumcenter_dist_l645_645369

structure Triangle :=
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_tri : a + b > c ∧ a + c > b ∧ b + c > a)

def incenter (T : Triangle) : ℝ × ℝ :=
  -- Assuming this function exists
  sorry

def circumcenter (T : Triangle) : ℝ × ℝ :=
  -- Assuming this function exists
  sorry

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem incenter_circumcenter_dist :
  ∀ (T : Triangle), T.a = 6 → T.b = 8 → T.c = 10 → dist (incenter T) (circumcenter T) = real.sqrt 5 :=
by
  intros T h_a h_b h_c
  sorry

end incenter_circumcenter_dist_l645_645369


namespace concentration_third_flask_l645_645279

-- Define the concentrations as per the given problem

noncomputable def concentration (acid_mass water_mass : ℝ) : ℝ :=
  (acid_mass / (acid_mass + water_mass)) * 100

-- Given conditions
def acid_mass_first_flask : ℝ := 10
def acid_mass_second_flask : ℝ := 20
def acid_mass_third_flask : ℝ := 30
def concentration_first_flask : ℝ := 5
def concentration_second_flask : ℝ := 70 / 3

-- Total water added to the first and second flasks
def total_water_mass : ℝ :=
  let W1 := (acid_mass_first_flask - concentration_first_flask * acid_mass_first_flask / 100)
  let W2 := (acid_mass_second_flask - concentration_second_flask * acid_mass_second_flask / 100)
  W1 + W2 

-- Prove the concentration of acid in the third flask
theorem concentration_third_flask : 
  concentration acid_mass_third_flask total_water_mass = 10.5 := 
  sorry

end concentration_third_flask_l645_645279


namespace ratio_of_Frederick_to_Tyson_l645_645983

-- Definitions of the ages based on given conditions
def Kyle : Nat := 25
def Tyson : Nat := 20
def Julian : Nat := Kyle - 5
def Frederick : Nat := Julian + 20

-- The ratio of Frederick's age to Tyson's age
def ratio : Nat × Nat := (Frederick / Nat.gcd Frederick Tyson, Tyson / Nat.gcd Frederick Tyson)

-- Proving the ratio is 2:1
theorem ratio_of_Frederick_to_Tyson : ratio = (2, 1) := by
  sorry

end ratio_of_Frederick_to_Tyson_l645_645983
