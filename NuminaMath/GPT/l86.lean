import Mathlib

namespace primes_between_8000_and_12000_l86_86316

theorem primes_between_8000_and_12000 : 
  {p : ℕ | p.prime ∧ 8000 < p^2 ∧ p^2 < 12000}.card = 5 :=
by
  sorry

end primes_between_8000_and_12000_l86_86316


namespace pavan_distance_travelled_l86_86517

theorem pavan_distance_travelled (D : ℝ) (h1 : D / 60 + D / 50 = 11) : D = 300 :=
sorry

end pavan_distance_travelled_l86_86517


namespace compound_interest_principal_l86_86084

noncomputable def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem compound_interest_principal :
  ∀ (P r : ℝ), 
    compound_interest P r 2 = 8880 → 
    compound_interest P r 3 = 9261 → 
    P ≈ 8160 :=
by {
  intros P r h1 h2,
  sorry
}

end compound_interest_principal_l86_86084


namespace jori_water_left_l86_86742

theorem jori_water_left (initial_water : ℚ) (used_water : ℚ) : initial_water = 3 ∧ used_water = 5/4 → initial_water - used_water = 7/4 :=
by {
  intro h,
  cases h with h1 h2,
  rw [h1, h2],
  norm_num,
}

end jori_water_left_l86_86742


namespace eval_expr_eq_zero_l86_86165

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l86_86165


namespace part1_part2_part3_l86_86644

variable {a : ℕ → ℝ} -- Defining the infinite sequence {a_n}
variable {S : ℕ → ℝ} -- Defining the sum of the first n terms
variable {b : ℕ → ℝ} -- Defining the sequence {b_n}
variable {t : ℝ} (ht : t > 0) -- t is a positive integer
variable (h0 : ∀ n, a n ≠ 0) -- Each term in the sequence {a_n} is nonzero
variable (h1 : ∀ n, a n * a (n + 1) = S n) -- Condition a_n * a_(n + 1) = S_n
variable (h2 : ∀ n, b n = a n / (a n + t)) -- Definition of {b_n}

-- Part 1
theorem part1 : a 2018 = 1009 :=
sorry

-- Part 2
theorem part2 : 0 < a 1 ∧ a 1 < (1 + Real.sqrt 5) / 2 :=
sorry

-- Part 3
theorem part3 (ha1 : 0 < a 1 ∧ a 1 ∈ ℕ) : ∀ n, ∃ m k, b n = b m * b k :=
sorry

end part1_part2_part3_l86_86644


namespace tom_found_dimes_l86_86049

theorem tom_found_dimes :
  let quarters := 10
  let nickels := 4
  let pennies := 200
  let total_value := 5
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let value_pennies := 0.01 * pennies
  let total_other := value_quarters + value_nickels + value_pennies
  let value_dimes := total_value - total_other
  value_dimes / 0.10 = 3 := sorry

end tom_found_dimes_l86_86049


namespace find_X_in_grid_l86_86725

theorem find_X_in_grid (x : ℕ) (d : ℕ) : 
  (∀i, i ∈ {0, 1, 2, 3, 4, 5} → ∃y, (10 + i * d = y)) ∧ -- condition to ensure valid arithmetic sequence
  (10 + 5 * d = 25) ∧                                      -- sixth number is 25
  (d = 3) →
  x = 19 :=
by
  sorry

end find_X_in_grid_l86_86725


namespace eval_ceil_floor_l86_86172

-- Definitions of the ceiling and floor functions and their relevant properties
def ceil (x : ℝ) : ℤ := ⌈x⌉
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Mathematical statement to prove
theorem eval_ceil_floor : ceil (7 / 3) + floor (- (7 / 3)) = 0 :=
by
  -- Proof is omitted
  sorry

end eval_ceil_floor_l86_86172


namespace angle_B_in_triangle_ABC_l86_86333

theorem angle_B_in_triangle_ABC (a b : ℝ) (A B : ℝ) (h1 : a = sqrt 3 * b) (h2 : A = 120) :
  B = 30 :=
sorry

end angle_B_in_triangle_ABC_l86_86333


namespace probability_domain_R_l86_86279

def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def valid_pairs : Set (ℕ × ℕ) := {p | p.1 ∈ A ∧ p.2 ∈ A ∧ (p.1 ^ 2 - 4 * p.2) < 0}
def total_pairs : ℕ := Finset.card (A ×ˢ A : Finset (ℕ × ℕ))

theorem probability_domain_R :
  (valid_pairs.to_finset.card : ℚ) / total_pairs = 17 / 36 :=
by
  sorry

end probability_domain_R_l86_86279


namespace max_fraction_diagonals_sides_cyclic_pentagon_l86_86066

theorem max_fraction_diagonals_sides_cyclic_pentagon (a b c d e A B C D E : ℝ)
  (h1 : b * e + a * A = C * D)
  (h2 : c * a + b * B = D * E)
  (h3 : d * b + c * C = E * A)
  (h4 : e * c + d * D = A * B)
  (h5 : a * d + e * E = B * C) :
  (a * b * c * d * e) / (A * B * C * D * E) ≤ (5 * Real.sqrt 5 - 11) / 2 :=
sorry

end max_fraction_diagonals_sides_cyclic_pentagon_l86_86066


namespace minimize_prod_time_l86_86825

noncomputable def shortest_production_time
  (items : ℕ) 
  (workers : ℕ) 
  (shaping_time : ℕ) 
  (firing_time : ℕ) : ℕ := by
  sorry

-- The main theorem statement
theorem minimize_prod_time
  (items : ℕ := 75)
  (workers : ℕ := 13)
  (shaping_time : ℕ := 15)
  (drying_time : ℕ := 10)
  (firing_time : ℕ := 30)
  (optimal_time : ℕ := 325) :
  shortest_production_time items workers shaping_time firing_time = optimal_time := by
  sorry

end minimize_prod_time_l86_86825


namespace circle_equation_l86_86437

/-- 
Given a circle whose center is on the x-axis, with a radius of sqrt(2),
and it passes through the point (-2, 1), the equation of the circle 
is (x + 1)^2 + y^2 = 2 or (x + 3)^2 + y^2 = 2.
-/
theorem circle_equation (a : ℝ) (x y : ℝ) :
  (∃ a, (a, 0) ∈ set.univ ∧ 
        (∃ r, r = sqrt 2 ∧ 
               (∃ x y, (x - a)^2 + y^2 = r^2 ∧
                      (x = -2 ∧ y = 1)))) →
  ((a = -1 ∨ a = -3) →
  ((x - a)^2 + y^2 = 2 ↔
     (x + 1) ^ 2 + y ^ 2 = 2 ∨ 
     (x + 3) ^ 2 + y ^ 2 = 2)) :=
by
  sorry

end circle_equation_l86_86437


namespace sample_definition_l86_86048

-- Define the population size
def population_size : ℕ := 780

-- Define the sample size
def sample_size : ℕ := 80

-- Define what a sample is
def is_sample (sample : set ℕ) (population : set ℕ) : Prop :=
  sample ⊆ population ∧ sample.card = sample_size

-- The population consists of scores of all students
def population : set ℕ := {n | n < population_size}

-- The specific sample from the population
def sample : set ℕ := {n | n < sample_size}

theorem sample_definition : is_sample sample population :=
by
  sorry

end sample_definition_l86_86048


namespace max_three_digit_sum_l86_86320

theorem max_three_digit_sum :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ 101 * A + 11 * B + 11 * C = 986 := 
sorry

end max_three_digit_sum_l86_86320


namespace probability_factor_less_than_10_l86_86855

def is_factor (n : ℕ) (f : ℕ) : Prop := f ∣ n

def count_factors (n : ℕ) : ℕ :=
  (Finset.filter (λ f => is_factor n f) (Finset.range (n + 1))).card

def count_factors_less_than (n : ℕ) (m : ℕ) : ℕ :=
  (Finset.filter (λ f => is_factor n f ∧ f < m) (Finset.range (n + 1))).card

theorem probability_factor_less_than_10 {n : ℕ} (hn : n = 120) :
  (count_factors_less_than n 10).toRat / (count_factors n).toRat = 1/2 :=
by
  sorry

end probability_factor_less_than_10_l86_86855


namespace distance_focus_directrix_l86_86720

theorem distance_focus_directrix (p : ℝ) (x_1 : ℝ) (h1 : 0 < p) (h2 : x_1^2 = 2 * p)
  (h3 : 1 + p / 2 = 3) : p = 4 :=
by
  sorry

end distance_focus_directrix_l86_86720


namespace total_payment_l86_86484

theorem total_payment (manicure_cost : ℚ) (tip_percentage : ℚ) (h_manicure_cost : manicure_cost = 30) (h_tip_percentage : tip_percentage = 30) : 
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := 
by 
  sorry

end total_payment_l86_86484


namespace average_age_when_youngest_born_l86_86473

theorem average_age_when_youngest_born (n : ℕ) (current_average_age youngest age_difference total_ages : ℝ)
  (hc1 : n = 7)
  (hc2 : current_average_age = 30)
  (hc3 : youngest = 6)
  (hc4 : age_difference = youngest * 6)
  (hc5 : total_ages = n * current_average_age - age_difference) :
  total_ages / n = 24.857
:= sorry

end average_age_when_youngest_born_l86_86473


namespace polynomial_roots_cubic_identity_l86_86382

theorem polynomial_roots_cubic_identity :
  ∀ (p q r : ℝ), (∃ (h : polynomial ℝ), h = 3 * X^3 - 5 * X^2 + 12 * X - 7 ∧ h.roots = {p, q, r}) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = -35/3 :=
by
  intros p q r h
  sorry

end polynomial_roots_cubic_identity_l86_86382


namespace correct_filling_correct_filling2_correct_filling3_correct_filling4_correct_filling5_correct_filling6_correct_filling7_correct_filling9_correct_filling10_correct_filling11_correct_filling12_correct_filling13_correct_filling14_correct_filling15_correct_filling16_correct_filling17_correct_filling18_correct_filling19_correct_filling20_l86_86776

-- Definitions of conditions and correct answers
def tired (context : String) : Prop := context = "and the woman said she was absolutely exhausted"
def turn_off (context : String) : Prop := context = "trying to"
def family (context : String) : Prop := context = "daughter’s passion had turned to making sure her"
def before (context : String) : Prop := context = "the woman’s brother’s house the night"
def sound (context : String) : Prop := context = "woke up to the"
def help (context : String) : Prop := context = "daughter asked for"
def broken (context : String) : Prop := context = "faucet handle was"
def until (context : String) : Prop := context = "wouldn’t go to sleep"
def everyone (context : String) : Prop := context = "in the family was out in the yard helping"
def dig (context : String) : Prop := context = "the only way to"
def power (context : String) : Prop := context = "Young people have much more"
def adults (context : String) : Prop := context = "to initiate and accomplish change than what"
def dragged (context : String) : Prop := context = "families are"
def tend_to (context : String) : Prop := context = "We"
def big (context : String) : Prop := context = "events like Clean and Green Week"
def difference (context : String) : Prop := context = "Instead, a lot of small changes can make a(n)"
def convince (context : String) : Prop := context = "students"
def up (context : String) : Prop := context = "to turn off the faucet to save water, move the air conditioning temperature"
def entire (context : String) : Prop := context = "family may change its behaviors."

-- The proof statements
theorem correct_filling (context: String) :
  tired(context) -> "tired" = "tired" :=
begin
  intro h,
  refl,
end

theorem correct_filling2 (context: String) :
  turn_off(context) -> "turn off" = "turn off" :=
begin
  intro h,
  refl,
end

theorem correct_filling3 (context: String) :
  family(context) -> "family" = "family" :=
begin
  intro h,
  refl,
end

theorem correct_filling4 (context: String) :
  before(context) -> "before" = "before" :=
begin
  intro h,
  refl,
end

theorem correct_filling5 (context: String) :
  sound(context) -> "sound" = "sound" :=
begin
  intro h,
  refl,
end

theorem correct_filling6 (context: String) :
  help(context) -> "help" = "help" :=
begin
  intro h,
  refl,
end

theorem correct_filling7 (context: String) :
  broken(context) -> "broken" = "broken" :=
begin
  intro h,
  refl,
end

theorem correct_filling9 (context: String) :
  until(context) -> "until" = "until" :=
begin
  intro h,
  refl,
end

theorem correct_filling10 (context: String) :
  everyone(context) -> "everyone" = "everyone" :=
begin
  intro h,
  refl,
end

theorem correct_filling11 (context: String) :
  dig(context) -> "dig" = "dig" :=
begin
  intro h,
  refl,
end

theorem correct_filling12 (context: String) :
  power(context) -> "power" = "power" :=
begin
  intro h,
  refl,
end

theorem correct_filling13 (context: String) :
  adults(context) -> "adults" = "adults" :=
begin
  intro h,
  refl,
end

theorem correct_filling14 (context: String) :
  dragged(context) -> "dragged" = "dragged" :=
begin
  intro h,
  refl,
end

theorem correct_filling15 (context: String) :
  tend_to(context) -> "tend to" = "tend to" :=
begin
  intro h,
  refl,
end

theorem correct_filling16 (context: String) :
  big(context) -> "big" = "big" :=
begin
  intro h,
  refl,
end

theorem correct_filling17 (context: String) :
  difference(context) -> "difference" = "difference" :=
begin
  intro h,
  refl,
end

theorem correct_filling18 (context: String) :
  convince(context) -> "convince" = "convince" :=
begin
  intro h,
  refl,
end

theorem correct_filling19 (context: String) :
  up(context) -> "up" = "up" :=
begin
  intro h,
  refl,
end

theorem correct_filling20 (context: String) :
  entire(context) -> "entire" = "entire" :=
begin
  intro h,
  refl,
end

#eval (if tired "and the woman said she was absolutely exhausted" then "tired" else "incorrect")
#eval (if turn_off "trying to" then "turn off" else "incorrect")
#eval (if family "daughter’s passion had turned to making sure her" then "family" else "incorrect")
#eval (if before "the woman’s brother’s house the night" then "before" else "incorrect")
#eval (if sound "woke up to the" then "sound" else "incorrect")
#eval (if help "daughter asked for" then "help" else "incorrect")
#eval (if broken "faucet handle was" then "broken" else "incorrect")
#eval (if until "wouldn’t go to sleep" then "until" else "incorrect")
#eval (if everyone "in the family was out in the yard helping" then "everyone" else "incorrect")
#eval (if dig "the only way to" then "dig" else "incorrect")
#eval (if power "Young people have much more" then "power" else "incorrect")
#eval (if adults "to initiate and accomplish change than what" then "adults" else "incorrect")
#eval (if dragged "families are" then "dragged" else "incorrect")
#eval (if tend_to "We" then "tend to" else "incorrect")
#eval (if big "events like Clean and Green Week" then "big" else "incorrect")
#eval (if difference "Instead, a lot of small changes can make a(n)" then "difference" else "incorrect")
#eval (if convince "students" then "convince" else "incorrect")
#eval (if up "to turn off the faucet to save water, move the air conditioning temperature" then "up" else "incorrect")
#eval (if entire "family may change its behaviors." then "entire" else "incorrect")

end correct_filling_correct_filling2_correct_filling3_correct_filling4_correct_filling5_correct_filling6_correct_filling7_correct_filling9_correct_filling10_correct_filling11_correct_filling12_correct_filling13_correct_filling14_correct_filling15_correct_filling16_correct_filling17_correct_filling18_correct_filling19_correct_filling20_l86_86776


namespace classB_wins_l86_86120

noncomputable def classA_size : ℕ := 40
noncomputable def classB_size : ℕ := 38

noncomputable def classA_excellent (a : ℕ) : ℕ := (15 * a) / 100
noncomputable def classA_good (a : ℕ) : ℕ := (30 * a) / 100 + 4
noncomputable def classA_average (a : ℕ) : ℕ := (25 * a) / 100
noncomputable def classA_sufficient (a : ℕ) : ℕ := (20 * a) / 100

noncomputable def classA_average_grade : ℕ → ℚ :=
  λ a, (5 * classA_excellent a + 4 * classA_good a + 3 * classA_average a + 2 * classA_sufficient a) / a

noncomputable def classB_excellent (b a : ℕ) : ℕ := classA_excellent a - 1
noncomputable def classB_good (b : ℕ) : ℕ := b / 2 - 4
noncomputable def classB_remaining (b a : ℕ) : ℕ := b - classB_excellent b a - classB_good b
noncomputable def classB_average (b a : ℕ) : ℕ := (5 * classB_remaining b a) / 6
noncomputable def classB_sufficient (b a : ℕ) : ℕ := classB_remaining b a - classB_average b a

noncomputable def classB_average_grade : ℕ → ℕ → ℚ :=
  λ b a, (5 * classB_excellent b a + 4 * classB_good b + 3 * classB_average b a + 2 * classB_sufficient b a) / b

theorem classB_wins (a b : ℕ) (h₁ : a = classA_size) (h₂ : b = classB_size) :
  classB_average_grade b a > classA_average_grade a := by
  sorry

end classB_wins_l86_86120


namespace find_A_l86_86977

-- Definitions
def num_divisors (n : ℕ) : ℕ := ∑ d in finset.range(n+1), if n % d = 0 then 1 else 0

-- Conditions expressed as definitions
def condition_2A (A : ℕ) : Prop := num_divisors (2 * A) = 30
def condition_3A (A : ℕ) : Prop := num_divisors (3 * A) = 36
def condition_9A (A : ℕ) : Prop := num_divisors (9 * A) = 49

-- The main theorem
theorem find_A (A : ℕ) : condition_2A A ∧ condition_3A A ∧ condition_9A A → A = 1176 :=
by {
  sorry
}

end find_A_l86_86977


namespace ratio_of_coefficients_l86_86032

theorem ratio_of_coefficients (a b c : ℝ) (h1 : 0 < a) 
  (h2 : ∃ (a b c : ℝ), ((a * (x - (-2)) * (x - (-1))) = ax^2 + bx + c)) : 
  a : b : c = 1 : 3 : 2 :=
by
  sorry

end ratio_of_coefficients_l86_86032


namespace students_answered_both_questions_correctly_l86_86322

theorem students_answered_both_questions_correctly (P_A P_B P_A'_B' : ℝ) (h_P_A : P_A = 0.75) (h_P_B : P_B = 0.7) (h_P_A'_B' : P_A'_B' = 0.2) :
  ∃ P_A_B : ℝ, P_A_B = 0.65 := 
by
  sorry

end students_answered_both_questions_correctly_l86_86322


namespace num_valid_Ns_less_2000_l86_86314

theorem num_valid_Ns_less_2000 : 
  {N : ℕ | N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x^⟨floor x⟩ = N}.card = 412 := 
sorry

end num_valid_Ns_less_2000_l86_86314


namespace ratio_first_term_l86_86554

theorem ratio_first_term (x : ℕ) (r : ℕ × ℕ) (h₀ : r = (6 - x, 7 - x)) 
        (h₁ : x ≥ 3) (h₂ : r.1 < r.2) : r.1 < 4 :=
by
  sorry

end ratio_first_term_l86_86554


namespace daps_to_dips_equivalence_l86_86690

-- Define the conditions
def ratio1 : Prop := (4 / 3) * dops = daps
def ratio2 : Prop := (2 / 7) * dips = dops

-- Define the equivalence we need to prove
def equivalence : Prop := 16 * daps = 42 * dips

-- Main statement
theorem daps_to_dips_equivalence (h1 : ratio1) (h2 : ratio2) : equivalence :=
sorry

end daps_to_dips_equivalence_l86_86690


namespace angle_between_u_v_l86_86613

noncomputable def angle_between_vectors (u v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
real.arccos ((inner_product_space.to_dual.map u v) / ((EuclideanSpace.norm u) * (EuclideanSpace.norm v)))

noncomputable def u : EuclideanSpace ℝ (Fin 3) := ![3, -2, 2]
noncomputable def v : EuclideanSpace ℝ (Fin 3) := ![2, 1, -1]

theorem angle_between_u_v :
  angle_between_vectors u v = real.arccos (2 / real.sqrt 102) := sorry

end angle_between_u_v_l86_86613


namespace A_can_avoid_defeat_l86_86476

theorem A_can_avoid_defeat (a : Fin 100 → Fin 100 → ℝ) :
  ∃ (RA RB : Finset (Fin 100)), RA.card = 50 ∧ RB.card = 50 ∧ RA ∩ RB = ∅ ∧
  ∑ i in RA, (∑ j, a i j)^2 ≥ ∑ i in RB, (∑ j, a i j)^2 :=
by sorry

end A_can_avoid_defeat_l86_86476


namespace syllogism_premises_l86_86037

-- Definitions from conditions
def all_chinese_indomitable : Prop := ∀ (x : Person), chinese x → indomitable x
def people_from_yushu_chinese : ∀ (x : Person), yushu x → chinese x

-- The proof problem that major premise and minor premise correspond to statements ① and ②.
theorem syllogism_premises :
  (major_premise all_chinese_indomitable) ∧ (minor_premise people_from_yushu_chinese) :=
sorry

end syllogism_premises_l86_86037


namespace range_of_x_l86_86256

theorem range_of_x (n : ℕ) (x : ℝ) :
  (x^(n+1) ≥ ∑ k in Finset.range n, (n - k) * Nat.choose (n + 1) k * x^k) →
  (even n → (x ≥ n ∨ x = -1)) ∧ (odd n → (x ≥ n ∨ x ≤ -1)) :=
by
  intro h
  sorry

end range_of_x_l86_86256


namespace symmetric_fun_eq_l86_86815

noncomputable def g (x : ℝ) : ℝ := log x / log 2

theorem symmetric_fun_eq (f : ℝ → ℝ) (h : ∀ x, f(x) = -g(-x)) : 
  ∀ x, f(x) = -log (-x) / log 2 :=
by
  -- skipping the proof
  intros
  rw h
  refl
  sorry

end symmetric_fun_eq_l86_86815


namespace infinite_very_good_pairs_l86_86112

-- Defining what it means for a pair to be "good"
def is_good (m n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ m ↔ p ∣ n)

-- Defining what it means for a pair to be "very good"
def is_very_good (m n : ℕ) : Prop :=
  is_good m n ∧ is_good (m + 1) (n + 1)

-- The theorem to prove: infiniteness of very good pairs
theorem infinite_very_good_pairs : Infinite {p : ℕ × ℕ | is_very_good p.1 p.2} :=
  sorry

end infinite_very_good_pairs_l86_86112


namespace ceiling_floor_sum_l86_86154

theorem ceiling_floor_sum : (Int.ceil (7 / 3) + Int.floor (-(7 / 3)) = 0) := by
  sorry

end ceiling_floor_sum_l86_86154


namespace probability_calculation_l86_86508

/-- Define the possible outcomes of a die roll -/
def die_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- Define what it means for a number to be a multiple of 3 -/
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

/-- Define the total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- Define the set of outcomes where the product is a multiple of 3 -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  { (x, y) | x ∈ die_outcomes ∧ y ∈ die_outcomes ∧ is_multiple_of_3 (x * y) }

/-- Define the probability of the product being a multiple of 3 -/
def probability_multiple_of_3 : ℚ :=
  favorable_outcomes.card / total_outcomes

theorem probability_calculation :
  probability_multiple_of_3 = 5 / 9 :=
sorry

end probability_calculation_l86_86508


namespace total_residents_in_building_l86_86891

def arithmetic_sum (n : Nat) (a1 an : Int) : Int :=
  n * (a1 + an) / 2

def num_apartments_on_floor (a1 d n : Int) : Int :=
  a1 + (n - 1) * d

theorem total_residents_in_building (n : Int) (a1 : Int) (d : Int)
  (one_bedroom_residents : Int) (two_bedroom_residents : Int)
  (three_bedroom_residents : Int) :
  n = 25 →
  a1 = 3 →
  d = 2 →
  one_bedroom_residents = 2 →
  two_bedroom_residents = 4 →
  three_bedroom_residents = 5 →
  (let a25 := num_apartments_on_floor a1 d n in 
   let sn := arithmetic_sum n a1 a25 in
   (225 * one_bedroom_residents) + (225 * two_bedroom_residents) + (225 * three_bedroom_residents) = 2475) :=
begin
  intros h_n h_a1 h_d h_one h_two h_three,
  let a25 := num_apartments_on_floor a1 d n,
  let sn := arithmetic_sum n a1 a25,
  have : a25 = 51, from calc
    a25 = a1 + (n - 1) * d : rfl
    ... = 3 + (25 - 1) * 2 : by simp [h_n, h_a1, h_d]
    ... = 3 + 48 : by norm_num
    ... = 51 : by norm_num,
  have : sn = 675, from calc
    sn = n * (a1 + a25) / 2 : rfl
    ... = 25 * (3 + 51) / 2 : by simp [h_n, h_a1, this]
    ... = 25 * 54 / 2 : by norm_num
    ... = 12.5 * 54 : by norm_num
    ... = 675 : by norm_num,
  have num_each_type := 675 / 3,
  have : num_each_type = 225, by norm_num,
  calc
    225 * one_bedroom_residents + 225 * two_bedroom_residents + 225 * three_bedroom_residents
        = 225 * 2 + 225 * 4 + 225 * 5 : by congr; assumption
    ... = 450 + 900 + 1125 : by norm_num
    ... = 2475 : by norm_num
end

end total_residents_in_building_l86_86891


namespace count_N_less_than_2000_l86_86303

noncomputable def count_valid_N (N : ℕ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x < 2000 ∧ N < 2000 ∧ x ^ (⌊x⌋₊) = N

theorem count_N_less_than_2000 : 
  ∃ (count : ℕ), count = 412 ∧ 
  (∀ (N : ℕ), N < 2000 → (∃ x : ℝ, x > 0 ∧ x < 2000 ∧ x ^ (⌊x⌋₊) = N) ↔ N < count) :=
sorry

end count_N_less_than_2000_l86_86303


namespace lines_meet_at_circumcircle_l86_86372

variables {Point : Type} [metric_space Point]

-- Define the given points
variables {A B C H M A1 C1 A2 C2 : Point}

-- Define some required functions
noncomputable def is_orthocenter (P : Point) (Δ : set Point) : Prop := sorry
noncomputable def midpoint (P Q : Point) : Point := sorry
noncomputable def intersection (L1 L2 : set Point) : Point := sorry
noncomputable def projection (P Q R : Point) : Point := sorry
noncomputable def on_circumcircle (P Δ : set Point) : Prop := sorry

axiom
  (acute_ABC : set Point)
  (orthocenter_H : is_orthocenter H acute_ABC)
  (midpoint_M : M = midpoint A C)
  (intersect_A1 : A1 = intersection (line M H) (line A B))
  (intersect_C1 : C1 = intersection (line M H) (line B C))
  (proj_A2 : A2 = projection A1 B H)
  (proj_C2 : C2 = projection C1 B H)

theorem lines_meet_at_circumcircle :
  ∃ R : Point, on_circumcircle R acute_ABC ∧ (R ∈ line C A2) ∧ (R ∈ line A C2) := sorry

end lines_meet_at_circumcircle_l86_86372


namespace tiles_cover_the_floor_l86_86916

theorem tiles_cover_the_floor
  (n : ℕ)
  (h : 2 * n - 1 = 101)
  : n ^ 2 = 2601 := sorry

end tiles_cover_the_floor_l86_86916


namespace least_four_digit_with_factors_3_5_7_l86_86062

open Nat

-- Definitions for the conditions
def has_factors (n : ℕ) (factors : List ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Main theorem statement
theorem least_four_digit_with_factors_3_5_7
  (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000)
  (h3 : has_factors n [3, 5, 7]) :
  n = 1050 :=
sorry

end least_four_digit_with_factors_3_5_7_l86_86062


namespace ln_n_lt_8m_l86_86385

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := 
  Real.log x - m * x^2 + 2 * n * x

theorem ln_n_lt_8m (m : ℝ) (n : ℝ) (h₀ : 0 < n) (h₁ : ∀ x > 0, f x m n ≤ f 1 m n) : 
  Real.log n < 8 * m := 
sorry

end ln_n_lt_8m_l86_86385


namespace negation_of_at_most_two_is_at_least_three_l86_86430

-- Define 'at most two solutions' as a condition
def at_most_two_solutions (f : α → Prop) := ∃ s : set α, s.card ≤ 2 ∧ ∀ x, f x → x ∈ s

-- Define the negation of 'at most two solutions' which is 'at least three solutions'
def at_least_three_solutions (f : α → Prop) := ∃ s : set α, s.card ≥ 3 ∧ ∀ x, f x → x ∈ s

-- Proof statement: Negation of 'at most two solutions' implies 'at least three solutions'
theorem negation_of_at_most_two_is_at_least_three {α : Type*} {f : α → Prop} :
  ¬ at_most_two_solutions f → at_least_three_solutions f :=
by sorry

end negation_of_at_most_two_is_at_least_three_l86_86430


namespace apples_in_each_basket_after_sister_took_apples_l86_86736

theorem apples_in_each_basket_after_sister_took_apples 
  (total_apples : ℕ) 
  (number_of_baskets : ℕ) 
  (apples_taken_from_each : ℕ)
  (initial_apples_per_basket := total_apples / number_of_baskets)
  (final_apples_per_basket := initial_apples_per_basket - apples_taken_from_each) :
  total_apples = 64 → number_of_baskets = 4 → apples_taken_from_each = 3 → final_apples_per_basket = 13 := 
by 
  intros htotal hnumber htake
  rw [htotal, hnumber, htake]
  have initial_apples : initial_apples_per_basket = 16 := by norm_num
  rw initial_apples
  norm_num
  sorry

end apples_in_each_basket_after_sister_took_apples_l86_86736


namespace inequality_solution_set_l86_86460

theorem inequality_solution_set :
  {x : ℝ | (x - 3) / (x + 2) ≤ 0} = {x : ℝ | -2 < x ∧ x ≤ 3} :=
by
  sorry

end inequality_solution_set_l86_86460


namespace range_of_a_l86_86251

variables (a : ℝ) (p q : Prop)

-- Conditions
def cond1 : Prop := a > 0
def cond2 : Prop := a ≠ 1
def prop_p : Prop := ∀ x > 0, ∀ y, y = log a (x + 2 - a) → ∀ ε > 0, ∃ δ > 0, ∀ x₀, x₀ - x < δ → abs (log a (x₀ + 2 - a) - log a (x + 2 - a)) < ε
def prop_q : Prop := ∀ x₀ x₁, x₀^2 + 2*a*x₀ + 1 = 0 → x₁^2 + 2*a*x₁ + 1 = 0 → x₀ ≠ x₁

-- Theorem
theorem range_of_a (ha0 : cond1 a) (ha1 : cond2 a) (pq_disj : p ∨ q) (pq_conj : ¬ (p ∧ q)) 
  (hp : p = prop_p a) (hq : q = prop_q a) : a ∈ set.Ioi 2 :=
sorry

end range_of_a_l86_86251


namespace minimize_prod_time_l86_86824

noncomputable def shortest_production_time
  (items : ℕ) 
  (workers : ℕ) 
  (shaping_time : ℕ) 
  (firing_time : ℕ) : ℕ := by
  sorry

-- The main theorem statement
theorem minimize_prod_time
  (items : ℕ := 75)
  (workers : ℕ := 13)
  (shaping_time : ℕ := 15)
  (drying_time : ℕ := 10)
  (firing_time : ℕ := 30)
  (optimal_time : ℕ := 325) :
  shortest_production_time items workers shaping_time firing_time = optimal_time := by
  sorry

end minimize_prod_time_l86_86824


namespace airplane_time_in_air_l86_86931

-- Define conditions
def distance_seaport_island := 840  -- Total distance in km
def speed_icebreaker := 20          -- Speed of the icebreaker in km/h
def time_icebreaker := 22           -- Total time the icebreaker traveled in hours
def speed_airplane := 120           -- Speed of the airplane in km/h

-- Prove the time the airplane spent in the air
theorem airplane_time_in_air : (distance_seaport_island - speed_icebreaker * time_icebreaker) / speed_airplane = 10 / 3 := by
  -- This is where the proof steps would go, but we're placing sorry to skip it for now.
  sorry

end airplane_time_in_air_l86_86931


namespace number_of_valid_Ns_l86_86310

noncomputable def count_valid_N : ℕ :=
  (finset.range 2000).filter (λ N, ∃ x : ℝ, x^floor x = N).card

theorem number_of_valid_Ns :
  count_valid_N = 1287 :=
sorry

end number_of_valid_Ns_l86_86310


namespace recommended_hours_per_day_l86_86147

def current_hours_per_day : ℕ := 4
def lacks_hours_per_week : ℕ := 14
def days_per_week : ℕ := 7

theorem recommended_hours_per_day : ℕ :=
  let current_hours_per_week := current_hours_per_day * days_per_week
  let total_recommended_hours_per_week := current_hours_per_week + lacks_hours_per_week
  total_recommended_hours_per_week / days_per_week = 6 :=
by sorry

end recommended_hours_per_day_l86_86147


namespace max_sin_A_plus_sin_C_l86_86253

theorem max_sin_A_plus_sin_C 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : A + B + C = π)
  (h_sin_seq : ∃ r : ℝ, ∀ x y z, (x, y, z) = (sin A, sin B, sin C) → y^2 = x * z)
  (h_B_max : B = π / 3) :
  sin A + sin C = sqrt 3 :=
by sorry

end max_sin_A_plus_sin_C_l86_86253


namespace martha_cakes_required_l86_86388

-- Conditions
def number_of_children : ℝ := 3.0
def cakes_per_child : ℝ := 18.0

-- The main statement to prove
theorem martha_cakes_required:
  (number_of_children * cakes_per_child) = 54.0 := 
by
  sorry

end martha_cakes_required_l86_86388


namespace evaluate_expression_l86_86197

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l86_86197


namespace modulus_product_l86_86978

theorem modulus_product :
  (complex.abs ⟨5, -3⟩) * (complex.abs ⟨5, 3⟩) = 34 := 
by 
  sorry

end modulus_product_l86_86978


namespace linear_regression_intercept_l86_86899

theorem linear_regression_intercept :
  let x_values := [1, 2, 3, 4, 5]
  let y_values := [0.5, 0.8, 1.0, 1.2, 1.5]
  let x_mean := (x_values.sum / x_values.length : ℝ)
  let y_mean := (y_values.sum / y_values.length : ℝ)
  let slope := 0.24
  (x_mean = 3) →
  (y_mean = 1) →
  y_mean = slope * x_mean + 0.28 :=
by
  sorry

end linear_regression_intercept_l86_86899


namespace find_a_l86_86441

variable (p1 p2 : ℝ × ℝ)

def direction_vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

def scalar_multiply (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

theorem find_a :
  let v := direction_vector (-2, 5) (1, 0) in
  let c := 1 / v.2 in
  let scaled_v := scalar_multiply c v in
  scaled_v = (3 / 5, -1) :=
by
  let v := direction_vector (-2, 5) (1, 0)
  let c := 1 / v.2
  let scaled_v := scalar_multiply c v
  sorry

end find_a_l86_86441


namespace find_PA_PB_sum_l86_86215

-- Define the conditions and problem parameters

def x := Real.sqrt 3 * Real.cos (Real.pi / 2)
def y := Real.sqrt 3 * Real.sin (Real.pi / 2)
def P : (Real × Real) := (x, y)

def C_equation (phi : Real) : Prop :=
  x = Real.sqrt 5 * Real.cos phi ∧ y = Real.sqrt 15 * Real.sin phi

def t_1 := 4 + Real.sqrt 8  -- This is derived from solving t_1 + t_2 = 8 and 2 * t_1 * t_2 = 8
def t_2 := 4 - Real.sqrt 8  -- This is derived similarly

def PA := Real.sqrt ((t_1 - P.1)^2 + (t_1 - P.2)^2)
def PB := Real.sqrt ((t_2 - P.1)^2 + (t_2 - P.2)^2)

theorem find_PA_PB_sum :
  PA + PB = 6 := 
by
  sorry

end find_PA_PB_sum_l86_86215


namespace count_N_less_than_2000_l86_86305

noncomputable def count_valid_N (N : ℕ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x < 2000 ∧ N < 2000 ∧ x ^ (⌊x⌋₊) = N

theorem count_N_less_than_2000 : 
  ∃ (count : ℕ), count = 412 ∧ 
  (∀ (N : ℕ), N < 2000 → (∃ x : ℝ, x > 0 ∧ x < 2000 ∧ x ^ (⌊x⌋₊) = N) ↔ N < count) :=
sorry

end count_N_less_than_2000_l86_86305


namespace sum_n_fraction_result_sum_n_fraction_a_plus_b_l86_86832

noncomputable def sum_n_fraction : ℚ :=
  (∑ n in Finset.range 100, 1 / (n * (n + 1) * (n + 2)))

theorem sum_n_fraction_result :
  (∑ n in Finset.range 100, 1 / (n * (n + 1) * (n + 2))) = 2575 / 10302 :=
by
  sorry

theorem sum_n_fraction_a_plus_b :
  let a := 2575
  let b := 10302
  a + b = 12877 :=
by
  sorry

end sum_n_fraction_result_sum_n_fraction_a_plus_b_l86_86832


namespace sin_sum_inequality_l86_86524

theorem sin_sum_inequality (n : ℕ) (h : 0 < n) : 
    (\sum k in finset.range (n + 1), (abs (Real.sin (n + k : ℝ)) / (n + k))) > 1/6 :=
sorry

end sin_sum_inequality_l86_86524


namespace parallelogram_be_length_l86_86865

theorem parallelogram_be_length (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (parallelogram_ABCD : is_parallelogram ABCD) 
  (line_perpendicular_E : is_perpendicular_point_through_diag_intersection E F)
  (AB_eq_a : AB = a) (BC_eq_b : BC = b) (BF_eq_c : BF = c) : 
  BE = (b * c) / (a + 2 * c) :=
sorry

end parallelogram_be_length_l86_86865


namespace convert_563_base8_to_base3_l86_86145

theorem convert_563_base8_to_base3 :
  let n := 8^2 * 5 + 8^1 * 6 + 8^0 * 3 in
  base3_repr n = "111220" :=
by
  let n := 8^2 * 5 + 8^1 * 6 + 8^0 * 3
  have h1: n = 371 := by norm_num
  have h2: base3_repr 371 = "111220" := by norm_num
  exact (h1.trans h2.symm)

end convert_563_base8_to_base3_l86_86145


namespace martian_traffic_light_signals_l86_86892

/-
Define the problem with conditions:
1. A Martian traffic light consists of six identical bulbs arranged in 2 rows and 3 columns.
2. A rover driver can distinguish the number and positions of lit bulbs but not unlit ones.
3. Impossible to determine which bulb if only one is lit.
4. If no bulb is lit, the driver sees nothing.
-/

def num_distinguishable_signals (light : fin 2 × fin 3 → bool) : ℕ := -- For a 2x3 Martian traffic light

  -- Condition: Only the lit bulbs matter, not their exact positions
  let combinations := (finset.univ : finset (fin 2 × fin 3)).powerset.filter (λ s, s.nonempty) in
  -- Function to filter combinations removing the all-unlit case and consider signal distinguishability
  let count := combinations.card in
  44 -- The final count of the distinguishable patterns

theorem martian_traffic_light_signals : num_distinguishable_signals = 44 := sorry

end martian_traffic_light_signals_l86_86892


namespace sandwich_count_l86_86558

-- Define the given conditions
def meats : ℕ := 8
def cheeses : ℕ := 12
def cheese_combination_count : ℕ := Nat.choose cheeses 3

-- Define the total sandwich count based on the conditions
def total_sandwiches : ℕ := meats * cheese_combination_count

-- The theorem we want to prove
theorem sandwich_count : total_sandwiches = 1760 := by
  -- Mathematical steps here are omitted
  sorry

end sandwich_count_l86_86558


namespace son_l86_86108

theorem son's_age (S F : ℕ) (h₁ : F = 7 * (S - 8)) (h₂ : F / 4 = 14) : S = 16 :=
by {
  sorry
}

end son_l86_86108


namespace range_of_f_l86_86268

def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x

theorem range_of_f (m n : ℝ) (h : ∀ y ∈ set.Icc m n, f y ∈ set.Icc (-6 : ℝ) 2) : 0 ≤ m + n ∧ m + n ≤ 4 :=
sorry

end range_of_f_l86_86268


namespace probability_math_majors_consecutive_l86_86843

noncomputable def total_ways := Nat.choose 11 4 -- Number of ways to choose 5 persons out of 12 (fixing one)
noncomputable def favorable_ways := 12         -- Number of ways to arrange 5 math majors consecutively around a round table

theorem probability_math_majors_consecutive :
  (favorable_ways : ℚ) / total_ways = 2 / 55 :=
by
  sorry

end probability_math_majors_consecutive_l86_86843


namespace interest_rate_lent_l86_86113

theorem interest_rate_lent (P_borrowed : ℕ) (T_borrowed : ℕ) (R_borrowed : ℕ) (annual_gain : ℕ) :
    P_borrowed = 5000 → T_borrowed = 2 → R_borrowed = 4 → annual_gain = 50 →
    let SI_borrowed := (P_borrowed * R_borrowed * T_borrowed) / 100
    let annual_interest_paid := SI_borrowed / T_borrowed
    let annual_interest_received := annual_interest_paid + annual_gain
    let R_lent := (annual_interest_received * 100) / P_borrowed
    R_lent = 5 := by
  intros P_eq T_eq R_eq gain_eq
  rw [P_eq, T_eq, R_eq, gain_eq]
  let SI_borrowed := (5000 * 4 * 2) / 100
  have SI_eq : SI_borrowed = 400 := by norm_num
  rw [SI_eq]
  let annual_interest_paid := 400 / 2
  have paid_eq : annual_interest_paid = 200 := by norm_num
  rw [paid_eq]
  let annual_interest_received := 200 + 50
  have received_eq : annual_interest_received = 250 := by norm_num
  rw [received_eq]
  let R_lent := (250 * 100) / 5000
  have rate_eq : R_lent = 5 := by norm_num
  rw [rate_eq]
  exact rfl

end interest_rate_lent_l86_86113


namespace exists_five_distinct_real_roots_l86_86410

noncomputable def distinct_real_numbers (a : Fin 10 -> ℝ) : Prop :=
  Function.Injective a

theorem exists_five_distinct_real_roots :
  ∃ (a : Fin 10 -> ℝ), distinct_real_numbers a ∧
  (∃ (roots : Fin 5 -> ℝ), 
    (roots 0 = 0) ∧ (roots 1 = 1) ∧ (roots 2 = -1) ∧ (roots 3 = 2) ∧ (roots 4 = -2) ∧
    ∀ x, ((x - a 0) * (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) *
          (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8) * (x - a 9) =
          (x + a 0) * (x + a 1) * (x + a 2) * (x + a 3) * (x + a 4) *
          (x + a 5) * (x + a 6) * (x + a 7) * (x + a 8) * (x + a 9)) →
          (x = roots 0 ∨ x = roots 1 ∨ x = roots 2 ∨ x = roots 3 ∨ x = roots 4))) :=
by 
  sorry

end exists_five_distinct_real_roots_l86_86410


namespace least_N_l86_86406

theorem least_N :
  ∃ N : ℕ, 
    (N % 2 = 1) ∧ 
    (N % 3 = 2) ∧ 
    (N % 5 = 3) ∧ 
    (N % 7 = 4) ∧ 
    (∀ M : ℕ, 
      (M % 2 = 1) ∧ 
      (M % 3 = 2) ∧ 
      (M % 5 = 3) ∧ 
      (M % 7 = 4) → 
      N ≤ M) :=
  sorry

end least_N_l86_86406


namespace bracelet_pairing_impossible_l86_86970

/--
Elizabeth has 100 different bracelets, and each day she wears three of them to school. 
Prove that it is impossible for any pair of bracelets to appear together on her wrist exactly once.
-/
theorem bracelet_pairing_impossible : 
  (∃ (bracelet_set : Finset (Finset (Fin 100))), 
    (∀ (a b : Fin 100), a ≠ b → ∃ t ∈ bracelet_set, {a, b} ⊆ t) ∧ (∀ t ∈ bracelet_set, t.card = 3) ∧ (bracelet_set.card * 3 / 2 ≠ 99)) :=
sorry

end bracelet_pairing_impossible_l86_86970


namespace triangle_inequality_values_l86_86551

theorem triangle_inequality_values (x : ℕ) :
  x ≥ 2 ∧ x < 10 ↔ (x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9) :=
by sorry

end triangle_inequality_values_l86_86551


namespace least_four_digit_multiple_3_5_7_l86_86063

theorem least_four_digit_multiple_3_5_7 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ n = 1050 :=
by
  use 1050
  repeat {sorry}

end least_four_digit_multiple_3_5_7_l86_86063


namespace sum_of_nth_row_l86_86790

theorem sum_of_nth_row (n : ℕ) : 
  let row_sum := (λ n, (n + 1) * 2^n - 1)
  in row_sum n = (n + 1) * 2^n - 1 :=
by sorry

end sum_of_nth_row_l86_86790


namespace john_spent_at_candy_store_l86_86675

-- Conditions
def weekly_allowance : ℚ := 2.25
def spent_at_arcade : ℚ := (3 / 5) * weekly_allowance
def remaining_after_arcade : ℚ := weekly_allowance - spent_at_arcade
def spent_at_toy_store : ℚ := (1 / 3) * remaining_after_arcade
def remaining_after_toy_store : ℚ := remaining_after_arcade - spent_at_toy_store

-- Problem: Prove that John spent $0.60 at the candy store
theorem john_spent_at_candy_store : remaining_after_toy_store = 0.60 :=
by
  sorry

end john_spent_at_candy_store_l86_86675


namespace polynomial_roots_l86_86611

theorem polynomial_roots :
  (∀ x : ℝ, (x^3 - 2*x^2 - 5*x + 6 = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3)) :=
by
  sorry

end polynomial_roots_l86_86611


namespace count_valid_paths_l86_86093

-- Definitions of the boundary and path conditions
def on_boundary_or_outside (x y : Int) : Bool :=
  (x < -3 ∨ x > 3) ∨ (y < -3 ∨ y > 3)

-- Definition of a valid step
def valid_step (x y : Int) : Int × Int → Bool
| (x', y') => (x' = x + 1 ∧ y' = y) ∨ (x' = x ∧ y' = y + 1)

-- Define tuple for the path and the valid path condition
def valid_path (path : List (Int × Int)) : Prop :=
  (path.head = (-5, -5)) ∧
  (path.last = (5, 5)) ∧
  (∀ (pos : Int × Int) ∈ path, on_boundary_or_outside pos.1 pos.2) ∧
  (∀ (i : Nat) (pos : Int × Int) , i < path.length - 1 → pos = path.get! i → valid_step pos (path.get! (i + 1)))

-- The number of paths satisfying the conditions
def number_of_valid_paths : Nat :=
  sorry  -- Placeholder for the calculation of paths

-- The lean theorem statement
theorem count_valid_paths : number_of_valid_paths = 4252 :=
sorry

end count_valid_paths_l86_86093


namespace probability_of_grid_being_black_l86_86092

noncomputable def probability_grid_black_after_rotation : ℚ := sorry

theorem probability_of_grid_being_black:
  probability_grid_black_after_rotation = 429 / 21845 :=
sorry

end probability_of_grid_being_black_l86_86092


namespace infinite_a_no_exact_n_divisors_Prime_geq_5_has_no_exact_n_divisors_l86_86792

-- Main theorem statement
theorem infinite_a_no_exact_n_divisors :
  ∀ p : ℕ, Prime p → p ≥ 5 →
  ∀ n : ℕ, n ≥ 1 → ∃ a : ℕ, a = p^(p-1) ∧ ¬ has_exactly_n_divisors a n :=
by sorry

-- Helper predicate to express the number of divisors
def has_exactly_n_divisors (a n : ℕ) : Prop :=
  (divisor_count a = n)

-- Function to count the number of divisors of a natural number
noncomputable def divisor_count (a : ℕ) : ℕ :=
  if a = 0 then 0 else (nat.sigma % a).length

theorem Prime_geq_5_has_no_exact_n_divisors :
  ∀ p : ℕ, Prime p → p ≥ 5 →
  ∀ n : ℕ, n ≥ 1 → ¬has_exactly_n_divisors (p^(p-1)) n :=
by sorry

end infinite_a_no_exact_n_divisors_Prime_geq_5_has_no_exact_n_divisors_l86_86792


namespace race_distance_A_beats_C_l86_86894

variables (race_distance1 race_distance2 race_distance3 : ℕ)
           (distance_AB distance_BC distance_AC : ℕ)

theorem race_distance_A_beats_C :
  race_distance1 = 500 →
  race_distance2 = 500 →
  distance_AB = 50 →
  distance_BC = 25 →
  distance_AC = 58 →
  race_distance3 = 400 :=
by
  sorry

end race_distance_A_beats_C_l86_86894


namespace find_river_depth_l86_86544

-- Define the given conditions
def river_width : ℝ := 45
def flow_rate_kmph : ℝ := 6
def volume_per_minute : ℝ := 9000

-- Convert flow rate from kmph to meters per minute
def flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60

-- Define the expected depth
def expected_depth : ℝ := 2

-- Statement to prove: the depth of the river
theorem find_river_depth :
  let D := volume_per_minute / (river_width * flow_rate_mpm) in
  D = expected_depth :=
by
  sorry

end find_river_depth_l86_86544


namespace mapping_correct_statements_l86_86884

variable (A B : Type)
variable (f : A → B)

theorem mapping_correct_statements :
  (∀ a1 a2 : A, f a1 = f a2 → a1 = a2) ∧
  (∃ a1 a2 : A, a1 ≠ a2 ∧ f a1 = f a2) ∧
  ¬ (∃ b : B, ∃ a1 a2 : A, a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b) ∧
  ¬ (∀ b : B, ∃ a : A, f a = b) →
  {s | s = 1 ∨ s = 2}.card = 2 :=
begin
  sorry
end

end mapping_correct_statements_l86_86884


namespace proof_lean_problem_l86_86345

open Real

noncomputable def problem_conditions (a b c : ℝ) (A B C : ℝ) (BD : ℝ) : Prop :=
  let condition_1 := c * sin ((A + C) / 2) = b * sin C
  let condition_2 := BD = 1
  let condition_3 := b = sqrt 3
  let condition_4 := BD = b * sin A
  let condition_5 := A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0
  condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5

noncomputable def find_B (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (h : problem_conditions a b c A B C BD) : Prop :=
  B = π / 3

noncomputable def find_perimeter (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (h : problem_conditions a b c A B C BD) : ℝ := 
  a + b + c = 3 + sqrt 3

theorem proof_lean_problem (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (h : problem_conditions a b c A B C BD) : 
  find_B a b c A B C BD h ∧ find_perimeter a b c A B C BD h :=
sorry

end proof_lean_problem_l86_86345


namespace part_I_max_min_part_II_exists_b_l86_86270

section PartI
variable (x : ℝ)
noncomputable def f_part_I := -x^2 + 3 * x - Real.log x

theorem part_I_max_min : 
  let f := f_part_I in 
  ∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, (f 1 = 2) ∧ (f (1 / 2) = Real.log 2 + 5 / 4)  :=
by
  sorry

end PartI

section PartII
variable (x b : ℝ)
noncomputable def f_part_II := b * x - Real.log x

theorem part_II_exists_b (e : ℝ) (he : e = Real.exp 1) :
  ∃ b : ℝ, (b = Real.exp 2) ∧ ∀ x ∈ Set.Ioc 0 e, 
    (∀ y : ℝ, f_part_II y = 3 → (y = b)) :=
by
  sorry

end PartII

end part_I_max_min_part_II_exists_b_l86_86270


namespace part_a_l86_86077

theorem part_a :
  (∀ a b : ℕ, a.gcd b = 1 → ∃ (frac : ℚ), frac = (1 / 2) ∨ frac = (1 / 3) ∨ frac = (1 / 6)) →
  (∑ (frac : ℚ) in {1 / 2, 1 / 3, 1 / 6}, frac).denom = 1 ∧
  (∑ (frac : ℚ) in {2, 3, 6}, frac).denom = 1 :=
sorry

end part_a_l86_86077


namespace symmetric_function_m_value_l86_86814

theorem symmetric_function_m_value (m : ℝ) : (∀ x : ℝ, (f : ℝ → ℝ) (x) = log (abs (x + m)) ∧ f (1 - x) = f (x)) → m = -1 :=
by
  intro h
  let f : ℝ → ℝ := λ x, log (abs (x + m))
  have symmetry := h 1
  sorry

end symmetric_function_m_value_l86_86814


namespace digit_150_of_fraction_l86_86498

theorem digit_150_of_fraction :
  let decimal_rep := "414285"
  let n := 150
  let len := String.length decimal_rep
  (n % len = 0) →
  (String.to_list decimal_rep).get_last! = '5' :=
by
  let decimal_rep := "414285"
  let n := 150
  let len := String.length decimal_rep
  intros h
  exact sorry

end digit_150_of_fraction_l86_86498


namespace num_perfect_square_factors_of_450_l86_86290

theorem num_perfect_square_factors_of_450 :
  ∃ n : ℕ, n = 4 ∧ ∀ d : ℕ, d ∣ 450 → (∃ k : ℕ, d = k * k) → d = 1 ∨ d = 25 ∨ d = 9 ∨ d = 225 :=
by
  sorry

end num_perfect_square_factors_of_450_l86_86290


namespace johns_phone_price_l86_86920

-- Define Alan's phone price
def alans_price : ℝ := 2000

-- Define the percentage increase
def percentage_increase : ℝ := 2/100

-- Define John's phone price
def johns_price := alans_price * (1 + percentage_increase)

-- The main theorem
theorem johns_phone_price : johns_price = 2040 := by
  sorry

end johns_phone_price_l86_86920


namespace solve_equation_l86_86797

theorem solve_equation (x : ℝ) (h : sqrt (4 * x - 3) + 12 / sqrt (4 * x - 3) = 8) :
  x = 7 / 4 ∨ x = 39 / 4 :=
sorry

end solve_equation_l86_86797


namespace students_in_dexters_high_school_l86_86707

variables (D S N : ℕ)

theorem students_in_dexters_high_school :
  (D = 4 * S) ∧
  (D + S + N = 3600) ∧
  (N = S - 400) →
  D = 8000 / 3 := 
sorry

end students_in_dexters_high_school_l86_86707


namespace simplify_fraction_l86_86438

theorem simplify_fraction :
  ∀ k : ℤ, ∃ a b : ℤ, (a = 1 ∧ b = 2) ∧ 
  ( (4 * k + 8) / 4 = a * k + b ) ∧ 
  (a / b : ℚ = 1 / 2) :=
by
  intros k
  use [1, 2]
  split
  { split; refl },
  split
  { rw [int.div_eq_of_eq_mul_left, mul_add, mul_one, add_comm],
    ring },
  norm_num,
  sorry -- Completes the proof when needed

end simplify_fraction_l86_86438


namespace total_distance_of_bus_rides_l86_86851

theorem total_distance_of_bus_rides :
  let vince_distance   := 5 / 8
  let zachary_distance := 1 / 2
  let alice_distance   := 17 / 20
  let rebecca_distance := 2 / 5
  let total_distance   := vince_distance + zachary_distance + alice_distance + rebecca_distance
  total_distance = 19/8 := by
  sorry

end total_distance_of_bus_rides_l86_86851


namespace count_ball_box_arrangements_l86_86836

theorem count_ball_box_arrangements :
  ∃ (arrangements : ℕ), arrangements = 20 ∧
  (∃ f : Fin 5 → Fin 5,
    (∃! i1, f i1 = i1) ∧ (∃! i2, f i2 = i2) ∧
    ∀ i, ∃! j, f i = j) :=
sorry

end count_ball_box_arrangements_l86_86836


namespace number_of_correct_propositions_l86_86812

def isMeaningful (f : ℝ → ℝ) : Prop :=
  ∀ x, isDefined (f x)

def isLine (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y → (x = y)

def isParabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c

noncomputable def prop1 : Prop := ¬isMeaningful (λ x => sqrt (x - 2) + sqrt (1 - x))
def prop2 : Prop := ∀ x : ℝ, true
noncomputable def prop3 : Prop := ¬isLine (λ x => 2 * x)
noncomputable def prop4 : Prop := ¬isParabola (λ x => if x >= 0 then x^2 else -x^2)

theorem number_of_correct_propositions : (prop1 ∧ prop2 ∧ prop3 ∧ prop4) = true := sorry

end number_of_correct_propositions_l86_86812


namespace gcd_390_455_546_l86_86018

theorem gcd_390_455_546 : Nat.gcd (Nat.gcd 390 455) 546 = 13 := 
by
  sorry    -- this indicates the proof is not included

end gcd_390_455_546_l86_86018


namespace total_expenses_l86_86356

def tulips : ℕ := 250
def carnations : ℕ := 375
def roses : ℕ := 320
def cost_per_flower : ℕ := 2

theorem total_expenses :
  tulips + carnations + roses * cost_per_flower = 1890 := 
sorry

end total_expenses_l86_86356


namespace train_crossing_time_l86_86078

-- Define the problem conditions in Lean 4
def train_length : ℕ := 130
def bridge_length : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := (speed_kmph * 1000 / 3600)

-- The statement to prove
theorem train_crossing_time : (train_length + bridge_length) / speed_mps = 28 :=
by
  -- The proof starts here
  sorry

end train_crossing_time_l86_86078


namespace positive_factors_of_450_are_perfect_squares_eq_8_l86_86287

theorem positive_factors_of_450_are_perfect_squares_eq_8 :
  let n := 450 in
  let is_factor (m k : ℕ) : Prop := k ≠ 0 ∧ k ≤ m ∧ m % k = 0 in
  let is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n in
  let factors (m : ℕ) : List ℕ := List.filter (λ k, is_factor m k) (List.range (m + 1)) in
  let perfect_square_factors := List.filter is_perfect_square (factors n) in
  List.length perfect_square_factors = 8 :=
by 
  have h1 : n = 2 * 3^2 * 5^2 := by sorry
  have h2 : ∀ m, m ∈ factors n ↔ is_factor n m := by sorry
  have h3 : ∀ m, is_perfect_square m ↔ (∃ k, k * k = m) := by sorry
  have h4 : List.length perfect_square_factors = 8 := by sorry
  exact h4

end positive_factors_of_450_are_perfect_squares_eq_8_l86_86287


namespace evaluate_expression_l86_86199

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l86_86199


namespace winner_score_l86_86889

def highest_winner_score (num_judges : ℕ) (max_rank : ℕ) (num_competitors : ℕ) (max_rank_diff : ℕ) : ℕ :=
  let possible_ranks := list.range (max_rank + 1)
  let total_ranks := (list.range (num_judges * max_rank + 1)).filter (λ s, 
    ∃ rs : list ℕ, rs.length = num_judges ∧ 
    (∀ r ∈ rs, r ∈ possible_ranks) ∧ 
    list.maximum rs - list.minimum rs ≤ max_rank_diff ∧ rs.sum = s)
  total_ranks.maximum.getOrElse 0

theorem winner_score (h1 : highest_winner_score 9 20 20 3 ≤ 24) : true :=
by trivial

end winner_score_l86_86889


namespace leftover_space_along_wall_l86_86564

theorem leftover_space_along_wall :
  ∀ (wall_length desk_length bookcase_length : ℝ) (n : ℕ),
  wall_length = 15 ∧ desk_length = 2 ∧ bookcase_length = 1.5 ∧ wall_length / (desk_length + bookcase_length) = n + (1 / (desk_length + bookcase_length)) →
    wall_length - (desk_length * n + bookcase_length * n) = 1 :=
by
  intros wall_length desk_length bookcase_length n
  simp
  intro h
  cases h with h1 h
  cases h with h2 h
  cases h with h3 h4
  rw  [h1, h2, h3]
  have t : 15 / 3.5 = Real.to_nnreal 4.285714285714286 := sorry
  rw t at h4
  exact sorry

end leftover_space_along_wall_l86_86564


namespace total_days_to_complete_work_l86_86130

def Amit_rate : ℝ := 1 / 15
def Ananthu_rate : ℝ := 1 / 90
def Amit_worked_days : ℝ := 3

theorem total_days_to_complete_work : 
    (Amit_worked_days * Amit_rate) + (Ananthu_rate * ((1 - (Amit_worked_days * Amit_rate)) * 90)) = 75 := 
by 
    -- The proof steps go here.
    sorry

end total_days_to_complete_work_l86_86130


namespace henry_games_total_l86_86284

theorem henry_games_total
    (wins : ℕ)
    (losses : ℕ)
    (draws : ℕ)
    (hw : wins = 2)
    (hl : losses = 2)
    (hd : draws = 10) :
  wins + losses + draws = 14 :=
by
  -- The proof is omitted.
  sorry

end henry_games_total_l86_86284


namespace sequence_50th_term_l86_86400

theorem sequence_50th_term : 
  (∀ n, (∃ a, (n = a + 1) ∧ term n = sqrt (3 * a)) → term 50 = 7 * sqrt 3) :=
by sorry

end sequence_50th_term_l86_86400


namespace sequence_sum_bound_l86_86122

noncomputable def a : ℕ → ℕ
| 0       := 0        -- providing a default 0 case for 0 index
| (n + 1) := if n = 0 then 2 else (a n)^2 - (a n) + 1

theorem sequence_sum_bound :
  1 - 1/(2003^2003) < ∑ i in Finset.range 2003, (1 / a (i + 1)) ∧
  ∑ i in Finset.range 2003, (1 / a (i + 1)) < 1 :=
begin
  sorry
end

end sequence_sum_bound_l86_86122


namespace range_of_lambda_over_m_l86_86282

theorem range_of_lambda_over_m (λ m α : ℝ)
  (h₁ : λ + 2 = 2 * m)
  (h₂ : λ^2 - sqrt 3 * cos (2 * α) = m + sin (2 * α)) :
  -6 ≤ λ / m ∧ λ / m ≤ 1 := 
sorry

end range_of_lambda_over_m_l86_86282


namespace eval_ceil_floor_sum_l86_86205

theorem eval_ceil_floor_sum :
  (Int.ceil (7 / 3) + Int.floor (- (7 / 3))) = 0 :=
by
  sorry

end eval_ceil_floor_sum_l86_86205


namespace least_four_digit_with_factors_l86_86057

open Nat

theorem least_four_digit_with_factors (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000) 
  (h3 : 3 ∣ n) 
  (h4 : 5 ∣ n) 
  (h5 : 7 ∣ n) : n = 1050 :=
by
  sorry

end least_four_digit_with_factors_l86_86057


namespace value_of_f_neg3_add_f_2_l86_86246

def f (x : ℤ) : ℤ :=
  if x ≤ 0 then 4 * x else 2 ^ x

theorem value_of_f_neg3_add_f_2 : f (-3) + f 2 = -8 :=
by
  sorry

end value_of_f_neg3_add_f_2_l86_86246


namespace problem_D_l86_86784
open Real

theorem problem_D : 
  (¬ (∀ a b : ℝ, abs (a + b) < 1 → abs a + abs b < 1)) ∧ 
  (set_of (λ x : ℝ, abs (x + 1) - 2 ≥ 0) = { x | x ≤ -3 } ∪ { x | x ≥ 1 }) :=
by
  sorry

end problem_D_l86_86784


namespace oranges_in_bin_l86_86548

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (added : ℕ) (result : ℕ)
    (h_initial : initial = 40)
    (h_thrown_away : thrown_away = 25)
    (h_added : added = 21)
    (h_result : result = 36) : initial - thrown_away + added = result :=
by
  -- skipped proof steps
  exact sorry

end oranges_in_bin_l86_86548


namespace equation_of_line_pq_l86_86107

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

theorem equation_of_line_pq :
  let M := (1 : ℝ, -2 : ℝ)
  let P := (2 : ℝ, 0 : ℝ)
  let Q := (0 : ℝ, -4 : ℝ)
  M = midpoint P Q →
  ∃ a b c : ℝ, a * 2 + b * (-4) + c = 0 ∧ a / b = (1 - 2) / (-2 - 0)
  ∧ a = 2 ∧ b = -1 ∧ c = -4 :=
by
  sorry

end equation_of_line_pq_l86_86107


namespace isosceles_right_triangle_legs_l86_86822

theorem isosceles_right_triangle_legs (c : ℝ) (a b : ℝ) (h1 : c = 9.899494936611665) (h2 : a = b) (h3 : c^2 = a^2 + b^2) : a = 7 :=
by
  noncomputable def x : ℝ := 7
  have hx : a = x := sorry
  exact hx

end isosceles_right_triangle_legs_l86_86822


namespace probability_of_black_black_red_l86_86547

-- Definitions:
def deck_size := 52
def black_cards := 26
def red_cards := 26

-- The probability calculation specific to the problem.
def probability_first_two_black_third_red : ℚ :=
  (black_cards * (black_cards - 1) / (deck_size * (deck_size - 1))) *
  (red_cards / (deck_size - 2))

-- The main statement of the theorem.
theorem probability_of_black_black_red :
  probability_first_two_black_third_red = 13 / 102 :=
by
  -- skip the proof using sorry.
  sorry

end probability_of_black_black_red_l86_86547


namespace triangle_area_proof_l86_86861

noncomputable def area_of_triangle_ABC : ℝ :=
  let AL := 4
  let BL := 2 * Real.sqrt 15
  let CL := 5
  let A := (0, 0) -- Just placeholders for vertices
  let B := (4, 0)
  let C := (3, 1) in -- Placeholder
  -- Assuming we have a function to compute area from given sides' lengths
  triangle_area AL BL CL = ⅑ * Real.sqrt 231 
  
theorem triangle_area_proof:
  ∀ AL BL CL : ℝ,
    AL = 4 → BL = 2 * Real.sqrt 15 → CL = 5 → 
    triangle_area AL BL CL = 9 * Real.sqrt 231 / 4 := 
begin
  intros,
  sorry
end

end triangle_area_proof_l86_86861


namespace sum_of_conjugates_l86_86423

theorem sum_of_conjugates
  (α β γ : ℝ)
  (h : exp (complex.I * α) + exp (complex.I * β) + exp (complex.I * γ) = (1 / 3 : ℂ) + (1 / 2 : ℂ) * complex.I) :
  exp (-complex.I * α) + exp (-complex.I * β) + exp (-complex.I * γ) = (1 / 3 : ℂ) - (1 / 2 : ℂ) * complex.I :=
by
  sorry

end sum_of_conjugates_l86_86423


namespace problem_statement_l86_86758

noncomputable def a : ℕ → ℝ
| 0     := -3
| (n+1) := 2 * a n + 2 * b n + 2 * real.sqrt ((a n)^2 + (b n)^2)

noncomputable def b : ℕ → ℝ
| 0     := 2
| (n+1) := 2 * a n + 2 * b n - 2 * real.sqrt ((a n)^2 + (b n)^2)

theorem problem_statement : (1 / a 2023) + (1 / b 2023) = 1 / 3 := sorry

end problem_statement_l86_86758


namespace quadratic_inequality_solution_l86_86993

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 6 * x > 20 ↔ x ∈ Ioo (-∞ : ℝ) (-2) ∪ Ioo 10 (∞ : ℝ) :=
sorry

end quadratic_inequality_solution_l86_86993


namespace mean_not_91_l86_86091

-- Define the list of scores
def scores : List ℝ := [78, 85, 91, 98, 98]

-- Define the mean calculation
def mean (l : List ℝ) : ℝ := l.sum / l.length

-- Define the theorem stating that the mean is not 91
theorem mean_not_91 : mean scores ≠ 91 := by
  sorry

end mean_not_91_l86_86091


namespace ceil_floor_sum_l86_86184

theorem ceil_floor_sum : (Int.ceil (7 / 3 : ℝ) + Int.floor (- 7 / 3 : ℝ) = 0) := 
  by {
    -- referring to the proof steps
    have h1: Int.ceil (7 / 3 : ℝ) = 3,
    { sorry },
    have h2: Int.floor (- 7 / 3 : ℝ) = -3,
    { sorry },
    rw [h1, h2],
    norm_num
  }

end ceil_floor_sum_l86_86184


namespace special_set_exists_l86_86733

noncomputable def exists_special_set : Prop :=
  ∃ S : set ℕ, 
    (∀ a : ℕ, a ∈ S ↔ 
      (∃ x y ∈ S, x ≠ y ∧ a = x + y) ∨ 
      (∃ p q ∉ S, p ≠ q ∧ a = p + q))

-- To be proven
theorem special_set_exists : exists_special_set := sorry

end special_set_exists_l86_86733


namespace modulus_of_z_root_of_polynomial_l86_86265

noncomputable def z : ℂ := (1/2 : ℂ) / (1 + (complex.I)) + (-5/4 + (9/4) * complex.I)

theorem modulus_of_z : complex.abs(z) = real.sqrt 5 :=
by 
  sorry

theorem root_of_polynomial : 
  (∃ p q : ℝ, (z ^ 2 * 2 + z * p + q = 0 ∧ p = 4 ∧ q = 10)) :=
by 
  sorry

end modulus_of_z_root_of_polynomial_l86_86265


namespace smallest_circle_tangent_to_line_and_circle_l86_86811

-- Define the line equation as a condition
def line_eq (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the original circle equation as a condition
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 2 * y = 0

-- Define the smallest circle equation as a condition
def smallest_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- The main lemma to prove that the smallest circle's equation matches the expected result
theorem smallest_circle_tangent_to_line_and_circle :
  (∀ x y, line_eq x y → smallest_circle_eq x y) ∧ (∀ x y, circle_eq x y → smallest_circle_eq x y) :=
by
  sorry -- Proof is omitted, as instructed

end smallest_circle_tangent_to_line_and_circle_l86_86811


namespace linear_combination_exists_l86_86929

structure Vec2 :=
  (x : ℝ)
  (y : ℝ)

def e1_B : Vec2 := 
  {x := -1, y := 2}

def e2_B : Vec2 := 
  {x :=  3, y := 2}

def a : Vec2 := 
  {x := 3, y := -1}

theorem linear_combination_exists :
  ∃ (α β : ℝ), (α * e1_B.x + β * e2_B.x = a.x) ∧ (α * e1_B.y + β * e2_B.y = a.y): 
  sorry

end linear_combination_exists_l86_86929


namespace find_angle_B_find_perimeter_l86_86353

-- Definitions of the variables and conditions
variable {α : Type} [LinearOrder α] [Field α] [TrigonometricFunctions α] -- for trigonometric functions

-- Definitions for the angles and sides of triangle ABC
variables (A B C a b c : α)

-- Conditions provided in the problem
variable (h1 : c * sin ((A + C) / 2) = b * sin C)
variable (h2 : BD = 1)
variable (h3 : b = sqrt 3)

-- Proving the angle B
theorem find_angle_B : B = π / 3 :=
sorry

-- Proving the perimeter of triangle ABC
theorem find_perimeter (B_eq : B = π / 3) : a + b + c = 3 + sqrt 3 :=
sorry

end find_angle_B_find_perimeter_l86_86353


namespace tangent_PQ_incircle_l86_86747

theorem tangent_PQ_incircle
  (ABC : Triangle)
  (I : Point)
  (h_incenter : incenter I ABC)
  (B C : Point)
  (P Q : Point)
  (circle_B : Circle)
  (circle_C : Circle)
  (tangent_BI : tangent circle_B I (line_segment B I))
  (tangent_CI : tangent circle_C I (line_segment C I))
  (intersects_AB : intersects circle_B (side AB) P)
  (intersects_AC : intersects circle_C (side AC) Q) :
  tangent (line_segment P Q) (incircle ABC) :=
sorry

end tangent_PQ_incircle_l86_86747


namespace gardener_total_expenses_l86_86358

theorem gardener_total_expenses
  (tulips carnations roses : ℕ)
  (cost_per_flower : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : roses = 320)
  (h4 : cost_per_flower = 2) :
  (tulips + carnations + roses) * cost_per_flower = 1890 := 
by
  sorry

end gardener_total_expenses_l86_86358


namespace kona_additional_miles_l86_86629

def distance_apartment_to_bakery : ℕ := 9
def distance_bakery_to_grandmothers_house : ℕ := 24
def distance_grandmothers_house_to_apartment : ℕ := 27

theorem kona_additional_miles:
  let no_bakery_round_trip := 2 * distance_grandmothers_house_to_apartment in
  let with_bakery_round_trip := 
    distance_apartment_to_bakery +
    distance_bakery_to_grandmothers_house +
    distance_grandmothers_house_to_apartment in
  with_bakery_round_trip - no_bakery_round_trip = 33 :=
by
  let no_bakery_round_trip : ℕ := 54
  let with_bakery_round_trip : ℕ := 60
  calc
    no_bakery_round_trip = 27 + 27 := sorry
    with_bakery_round_trip = 9 + 24 + 27 := sorry
    with_bakery_round_trip - no_bakery_round_trip = 33 := sorry

end kona_additional_miles_l86_86629


namespace min_period_of_cos_sq_is_pi_l86_86447

-- Define the function \(\cos^2\) and its minimum positive period problem
def min_period_cos_sq : ℝ := 
  -- Given the function
  let y (x : ℝ) := Real.cos x ^ 2
  -- Minimum positive period
  θ
-- Prove the minimum period of \( y = \cos^2 x \) is \( \pi \)
theorem min_period_of_cos_sq_is_pi : min_period_cos_sq = π :=
by
  -- leaving proof as sorry
  sorry

end min_period_of_cos_sq_is_pi_l86_86447


namespace correct_interpretation_of_assignment_l86_86429

theorem correct_interpretation_of_assignment :
  ∀ (x : ℕ), (* the interpretation of *) (x = x + 1) = (* is equivalent to saying that *) (x = x + 1) :=
begin
  sorry
end

end correct_interpretation_of_assignment_l86_86429


namespace cube_root_of_neg_eight_eq_neg_two_l86_86001

theorem cube_root_of_neg_eight_eq_neg_two : real.cbrt (-8) = -2 :=
by
  sorry

end cube_root_of_neg_eight_eq_neg_two_l86_86001


namespace total_distance_correct_l86_86394

def d1 : ℕ := 350
def d2 : ℕ := 375
def d3 : ℕ := 275
def total_distance : ℕ := 1000

theorem total_distance_correct : d1 + d2 + d3 = total_distance := by
  sorry

end total_distance_correct_l86_86394


namespace solution_set_of_inequality_l86_86652

noncomputable def f : ℝ → ℝ := sorry

variable (x : ℝ)

theorem solution_set_of_inequality 
  (h₀ : ∀ x, f'' x - f x > 1)
  (h₁ : f 0 = 2017) :
  {x | f x > 2018 * Real.exp x - 1} = Ioi 0 := 
sorry

end solution_set_of_inequality_l86_86652


namespace a_1998_l86_86759

noncomputable def sequence : ℕ → ℕ
| 0     := 1
| (n+1) := sorry  -- This should be properly defined

theorem a_1998 :
  let a := sequence 1998 in
  ∀ i j k, a + 2 * a + 4 * a = (i : ℕ) + 2 * j + 4 * k → a = 1227096648 :=
begin
  sorry,
end

end a_1998_l86_86759


namespace ceiling_floor_sum_l86_86156

theorem ceiling_floor_sum : (Int.ceil (7 / 3) + Int.floor (-(7 / 3)) = 0) := by
  sorry

end ceiling_floor_sum_l86_86156


namespace greatest_integer_c_l86_86055

noncomputable def quadratic_function (c : ℤ) (x : ℝ) : ℝ := x^2 + (c : ℝ) * x + 18

theorem greatest_integer_c (c : ℤ) :
  (∀ (x: ℝ), quadratic_function c x ≠ -6) →
  c ≤ 9 :=
begin
  sorry,
end

end greatest_integer_c_l86_86055


namespace hexagon_interior_angles_sum_l86_86470

theorem hexagon_interior_angles_sum :
  let n := 6 in
  (n - 2) * 180 = 720 :=
by
  sorry

end hexagon_interior_angles_sum_l86_86470


namespace perimeter_of_larger_triangle_l86_86566

-- Define the conditions for the specific problem
def isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∧ a ≠ c ∧ b ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

def similar_triangle (t₁ t₂ : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ (t₂.1 = k * t₁.1 ∧ t₂.2 = k * t₁.2 ∧ t₂.3 = k * t₁.3)

-- Premises
variables {a b c side_shorter side_longer : ℝ}
variables {k : ℝ}

-- The given isosceles triangle
def smaller_triangle := (15 : ℝ, 15 : ℝ, 6 : ℝ)

-- The larger similar triangle with the shortest side 18 cm
def larger_triangle := (18 : ℝ, 45 : ℝ, 45 : ℝ)

-- Problem statement in Lean
theorem perimeter_of_larger_triangle : 
  isosceles_triangle 15 15 6 →
  similar_triangle smaller_triangle larger_triangle →
  larger_triangle.1 = k * smaller_triangle.1 →
  larger_triangle.2 = k * smaller_triangle.2 →
  larger_triangle.3 = k * smaller_triangle.3 →
  k = 3 →
  (larger_triangle.1 + larger_triangle.2 + larger_triangle.3) = 108 :=
by
  -- Here we would give a proof if required
  sorry

end perimeter_of_larger_triangle_l86_86566


namespace percent_neither_filler_nor_cheese_l86_86895

-- Define the given conditions as constants
def total_weight : ℕ := 200
def filler_weight : ℕ := 40
def cheese_weight : ℕ := 30

-- Definition of the remaining weight that is neither filler nor cheese
def neither_weight : ℕ := total_weight - filler_weight - cheese_weight

-- Calculation of the percentage of the burger that is neither filler nor cheese
def percentage_neither : ℚ := (neither_weight : ℚ) / (total_weight : ℚ) * 100

-- The theorem to prove
theorem percent_neither_filler_nor_cheese :
  percentage_neither = 65 := by
  sorry

end percent_neither_filler_nor_cheese_l86_86895


namespace triangle_sides_l86_86416

theorem triangle_sides (x : ℝ) (h : x > 1) :
  let a := x^4 + x^3 + 2x^2 + x + 1,
      b := 2x^3 + x^2 + 2x + 1,
      c := x^4 - 1
  in (a + b > c) ∧ (a + c > b) ∧ (b + c > a) :=
by
  let a := x^4 + x^3 + 2x^2 + x + 1
  let b := 2x^3 + x^2 + 2x + 1
  let c := x^4 - 1
  -- These are direct translations of the conditions that need to hold
  have h1 : a + b = x^4 + 3x^3 + 3x^2 + 3x + 2 := by sorry
  have h2 : a + c = 2x^4 + x^3 + 2x^2 + x := by sorry
  have h3 : b + c = x^4 + 2x^3 + x^2 + 2x := by sorry
  -- The required inequalities
  have I1 : x^4 + 3x^3 + 3x^2 + 3x + 2 > x^4 - 1 := by sorry
  have I2 : 2x^4 + x^3 + 2x^2 + x > 2x^3 + x^2 + 2x + 1 := by sorry
  have I3 : x^4 + 2x^3 + x^2 + 2x > x^4 + x^3 + 2x^2 + x + 1 := by sorry
  exact ⟨I1, I2, I3⟩

end triangle_sides_l86_86416


namespace sum_of_interior_angles_of_hexagon_l86_86465

theorem sum_of_interior_angles_of_hexagon
  (n : ℕ)
  (h : n = 6) :
  (n - 2) * 180 = 720 := by
  sorry

end sum_of_interior_angles_of_hexagon_l86_86465


namespace evaluation_of_expression_l86_86507

theorem evaluation_of_expression : (3^2 - 2^2 + 1^2) = 6 :=
by
  sorry

end evaluation_of_expression_l86_86507


namespace area_of_ABCD_is_12_l86_86788

noncomputable def area_of_quadrilateral 
  (A B C D : ℝ × ℝ)
  (right_angle_at_B : ∠ABC = 90)
  (right_angle_at_D : ∠ADC = 90)
  (diagonal_AC : AC_length = (A - C).norm)
  (BC_length : (B - C).norm = 4)
  (AD_length : (A - D).norm = 3)
  (AC_length_value : AC_length = 5) : ℝ :=
let AB := (A - B).norm in
let DC := (D - C).norm in
let area_ABC := 1 / 2 * AB * 4 in
let area_ADC := 1 / 2 * 3 * DC in
area_ABC + area_ADC

theorem area_of_ABCD_is_12
  (A B C D : ℝ × ℝ)
  (right_angle_at_B : ∠ABC = 90)
  (right_angle_at_D : ∠ADC = 90)
  (diagonal_AC : AC_length = (A - C).norm)
  (BC_length : (B - C).norm = 4)
  (AD_length : (A - D).norm = 3)
  (AC_length_value : AC_length = 5) : area_of_quadrilateral A B C D right_angle_at_B right_angle_at_D diagonal_AC BC_length AD_length AC_length_value = 12 :=
sorry

end area_of_ABCD_is_12_l86_86788


namespace smallest_integer_between_x_and_10000_l86_86046

theorem smallest_integer_between_x_and_10000 :
  (∃ (x : ℕ), x + 81 ≤ 10000 ∧ ∀ n, x ≤ n → n < x + 81 → (∀ d ∈ to_digits 10 n, d = 4 ∨ d = 5 ∨ d = 6)) →
  ∃ x, x = 4444 :=
by
  sorry

end smallest_integer_between_x_and_10000_l86_86046


namespace sum_of_ceil_sqrt_l86_86974

theorem sum_of_ceil_sqrt :
  (∑ n in Finset.range 45, ⌈Real.sqrt (n + 10)⌉) = 270 :=
by
  sorry

end sum_of_ceil_sqrt_l86_86974


namespace tan_div_sin_cos_sin_mul_cos_l86_86234

theorem tan_div_sin_cos (α : ℝ) (h : Real.tan α = 7) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 13 :=
by
  sorry

theorem sin_mul_cos (α : ℝ) (h : Real.tan α = 7) :
  Real.sin α * Real.cos α = 7 / 50 :=
by
  sorry

end tan_div_sin_cos_sin_mul_cos_l86_86234


namespace inradius_excircle_ratio_l86_86748

-- Defining basic setup and inradii and excircle radii
variables {A B C M : Point} -- Points corresponding to vertices and the interior point M on AB
variables {r r1 r2 ρ ρ1 ρ2 : ℝ} -- Radii of inradii and excicles
variables (h_conditions : M ∈ segment A B ∧ inradius A M C = r1 ∧ inradius B M C = r2 ∧ inradius A B C = r ∧ excircle_radius_opposite AM A M C = ρ1 ∧ excircle_radius_opposite BM B M C = ρ2 ∧ excircle_radius_opposite AB A B C = ρ)

-- Proof statement
theorem inradius_excircle_ratio (h_conditions : M ∈ segment A B ∧ inradius A M C = r1 ∧ inradius B M C = r2 ∧ inradius A B C = r ∧ excircle_radius_opposite AM A M C = ρ1 ∧ excircle_radius_opposite BM B M C = ρ2 ∧ excircle_radius_opposite AB A B C = ρ) :
  (r1 / ρ1) * (r2 / ρ2) = r / ρ :=
by
  sorry

end inradius_excircle_ratio_l86_86748


namespace min_value_of_expression_l86_86625

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ m : ℝ, m = 1 ∧ 
    ∀ a b c ∈ {x : ℝ | x > 0}, 
      ∃ (val : ℝ), val = (a / (a^2 + 8 * b * c).sqrt + 
                          b / (b^2 + 8 * a * c).sqrt + 
                          c / (c^2 + 8 * a * b).sqrt) ∧ 
                          val ≥ m :=
begin
  -- Proof of the theorem goes here
  sorry,
end

end min_value_of_expression_l86_86625


namespace car_meet_truck_l86_86529

structure ProblemData :=
  (car_speed : ℝ)
  (truck_speed : ℝ)
  (car_arrival_time : ℝ)
  (truck_arrival_time : ℝ)
  (meeting_time : ℝ)

def time_in_minutes (hours minutes : ℝ) : ℝ :=
  hours * 60 + minutes

def car_truck_meeting_time (d : ProblemData) : ℝ :=
  let t_diff := d.truck_arrival_time - d.car_arrival_time
  let t_meet := d.car_arrival_time + t_diff / 3
  t_meet

theorem car_meet_truck :
  ∀ (d : ProblemData),
    d.car_speed = 2 * d.truck_speed →
    d.car_arrival_time = time_in_minutes 8 30 →
    d.truck_arrival_time = time_in_minutes 15 0 →
    car_truck_meeting_time d = time_in_minutes 10 40 :=
by
  intros d h1 h2 h3
  calc
    car_truck_meeting_time d = time_in_minutes 10 40 : by sorry

end car_meet_truck_l86_86529


namespace problem_for_odd_p_l86_86991

variable (p : ℕ)

-- Assuming the conditions that p is an odd number greater than 1
axiom h1 : p > 1
axiom h2 : odd p

theorem problem_for_odd_p (h1 : p > 1) (h2 : odd p) :
  (p-1)^(1/2*(p-1)) - 1 % (p-2) = 0 := 
sorry

end problem_for_odd_p_l86_86991


namespace final_value_after_operations_l86_86588

theorem final_value_after_operations (x : ℝ) (h : x ≠ 0) (n : ℕ) : 
  let z := alternate_operations x n in
  z = x^((-1)^n * 2^(n / 2)) :=
sorry

noncomputable def alternate_operations (x : ℝ) (n : ℕ) : ℝ :=
  let rec_op := λ n, if n % 2 = 0 then (rec_op (n / 2))^2 else (rec_op (n - 1))⁻¹ in
  rec_op n

end final_value_after_operations_l86_86588


namespace rhombus_area_from_diagonals_roots_l86_86818

theorem rhombus_area_from_diagonals_roots :
  (∃ a b : ℝ, (a ^ 2 - 6 * a + 8 = 0) ∧ (b ^ 2 - 6 * b + 8 = 0) ∧ 
  (a ≠ b) ∧ (a * b = 8) ∧ (a ≠ 0) ∧ (b ≠ 0)) → 
  (1 / 2 * (∃ a b : ℝ, (a * b = 8)) = 4) :=
by
  sorry

end rhombus_area_from_diagonals_roots_l86_86818


namespace heidi_and_karl_painting_l86_86323

-- Given conditions
def heidi_paint_rate := 1 / 60 -- Rate at which Heidi paints, in walls per minute
def karl_paint_rate := 2 * heidi_paint_rate -- Rate at which Karl paints, in walls per minute
def painting_time := 20 -- Time spent painting, in minutes

-- Prove the amount of each wall painted
theorem heidi_and_karl_painting :
  (heidi_paint_rate * painting_time = 1 / 3) ∧ (karl_paint_rate * painting_time = 2 / 3) :=
sorry

end heidi_and_karl_painting_l86_86323


namespace tangent_circles_l86_86217

def center1 := (-1, 1)
def radius1 := Real.sqrt 2
def circle1 (x y : ℝ) := (x + 1) ^ 2 + (y - 1) ^ 2 = 2

def line1 (x y : ℝ) := x - y - 4 = 0

def center2 := (1, -1)
def radius2 := Real.sqrt 2
def circle2 (x y : ℝ) := (x - 1) ^ 2 + (y + 1) ^ 2 = 2

theorem tangent_circles : 
  (∀ (x y : ℝ), circle2 x y ↔ 
    (x - center2.1) ^ 2 + (y - center2.2) ^ 2 = radius2 ^ 2) ∧
  (∀ (x y : ℝ), |x - y - 4| / Real.sqrt 2 = radius2 ∧ x + y = 0 → 
    (x - center2.1) ^ 2 + (y - center2.2) ^ 2 = radius2 ^ 2) :=
by sorry

end tangent_circles_l86_86217


namespace train_passes_pole_in_10_seconds_l86_86514

theorem train_passes_pole_in_10_seconds :
  let L := 150 -- length of the train in meters
  let S_kmhr := 54 -- speed in kilometers per hour
  let S_ms := S_kmhr * 1000 / 3600 -- speed in meters per second
  (L / S_ms = 10) := 
by
  sorry

end train_passes_pole_in_10_seconds_l86_86514


namespace ceiling_plus_floor_eq_zero_l86_86192

theorem ceiling_plus_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (-7 / 3)) = 0 := by
  sorry

end ceiling_plus_floor_eq_zero_l86_86192


namespace least_four_digit_with_factors_l86_86059

open Nat

theorem least_four_digit_with_factors (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000) 
  (h3 : 3 ∣ n) 
  (h4 : 5 ∣ n) 
  (h5 : 7 ∣ n) : n = 1050 :=
by
  sorry

end least_four_digit_with_factors_l86_86059


namespace first_term_of_new_ratio_l86_86556

-- Given conditions as definitions
def original_ratio : ℚ := 6 / 7
def x (n : ℕ) : Prop := n ≥ 3

-- Prove that the first term of the ratio that the new ratio should be less than is 4
theorem first_term_of_new_ratio (n : ℕ) (h1 : x n) : ∃ b, (6 - n) / (7 - n) < 4 / b :=
by
  exists 5
  sorry

end first_term_of_new_ratio_l86_86556


namespace eq_of_frac_eq_and_neq_neg_one_l86_86693

theorem eq_of_frac_eq_and_neq_neg_one
  (a b c d : ℝ)
  (h : (a + b) / (c + d) = (b + c) / (a + d))
  (h_neq : (a + b) / (c + d) ≠ -1) :
  a = c :=
sorry

end eq_of_frac_eq_and_neq_neg_one_l86_86693


namespace smallest_n_for_integer_S_n_l86_86376

noncomputable def K : ℚ := ∑ i in (Finset.range 10).filter (λ i, i > 0), (1 / i)

noncomputable def D : ℕ := 2^3 * 3^2 * 5 * 7

def S_n (n : ℕ) : ℚ := (n * 10^(n - 1)) * K + 1

theorem smallest_n_for_integer_S_n : ∃ n : ℕ, S_n n = 1 ∧ ∀ m < n, S_n m ≠ 1 := sorry

end smallest_n_for_integer_S_n_l86_86376


namespace product_sequence_l86_86939

theorem product_sequence : 
  (∏ n in Finset.range 1001, (n + 3 : ℝ) / (n + 2)) = 1004 / 3 :=
by
  sorry

end product_sequence_l86_86939


namespace a_100_result_a_1983_result_l86_86867

variable a : ℕ → ℕ

axiom increasing_sequence : ∀ (n m : ℕ), n < m → a n < a m
axiom a_of_a_k : ∀ k : ℕ, a (a k) = 3 * k

theorem a_100_result : a 100 = 181 :=
by
  sorry

theorem a_1983_result : a 1983 = 3762 :=
by
  sorry

end a_100_result_a_1983_result_l86_86867


namespace eval_ceil_floor_sum_l86_86208

theorem eval_ceil_floor_sum :
  (Int.ceil (7 / 3) + Int.floor (- (7 / 3))) = 0 :=
by
  sorry

end eval_ceil_floor_sum_l86_86208


namespace probability_of_abs_le_2_l86_86933

/-- Define the interval [-1, 3] as a set -/
def interval_set : set ℝ := { x | x ≥ -1 ∧ x ≤ 3 }

/-- Define the range that satisfies the condition |x| ≤ 2 -/
def range_satisfying_abs : set ℝ := { x | x ≥ -2 ∧ x ≤ 2 }

/-- Define the subset of the interval [-1, 3] that also satisfies |x| ≤ 2 -/
def valid_range : set ℝ := interval_set ∩ range_satisfying_abs

/-- Define the length of an interval [a, b] -/
def interval_length (a b : ℝ) : ℝ := b - a

/-- The problem's statement: Prove the probability that |x| ≤ 2 for x in [-1, 3] is 3/4 -/
theorem probability_of_abs_le_2 : 
  (interval_length (-1) 2) / (interval_length (-1) 3) = 3 / 4 :=
by
  sorry

end probability_of_abs_le_2_l86_86933


namespace find_width_fabric_width_is_3_l86_86210

variable (Area Length : ℝ)
variable (Width : ℝ)

theorem find_width (h1 : Area = 24) (h2 : Length = 8) :
  Width = Area / Length :=
sorry

theorem fabric_width_is_3 (h1 : Area = 24) (h2 : Length = 8) :
  (Area / Length) = 3 :=
by
  have h : Area / Length = 3 := by sorry
  exact h

end find_width_fabric_width_is_3_l86_86210


namespace ceil_floor_sum_l86_86182

theorem ceil_floor_sum : (Int.ceil (7 / 3 : ℝ) + Int.floor (- 7 / 3 : ℝ) = 0) := 
  by {
    -- referring to the proof steps
    have h1: Int.ceil (7 / 3 : ℝ) = 3,
    { sorry },
    have h2: Int.floor (- 7 / 3 : ℝ) = -3,
    { sorry },
    rw [h1, h2],
    norm_num
  }

end ceil_floor_sum_l86_86182


namespace contest_end_time_l86_86531

theorem contest_end_time (start_time : Time := ⟨15, 0⟩ ) (duration_minutes : ℕ := 450) (break_minutes : ℕ := 15)
  : Time :=
by
  sorry

end contest_end_time_l86_86531


namespace least_four_digit_with_factors_3_5_7_l86_86061

open Nat

-- Definitions for the conditions
def has_factors (n : ℕ) (factors : List ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Main theorem statement
theorem least_four_digit_with_factors_3_5_7
  (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000)
  (h3 : has_factors n [3, 5, 7]) :
  n = 1050 :=
sorry

end least_four_digit_with_factors_3_5_7_l86_86061


namespace count_N_less_than_2000_l86_86302

noncomputable def count_valid_N (N : ℕ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x < 2000 ∧ N < 2000 ∧ x ^ (⌊x⌋₊) = N

theorem count_N_less_than_2000 : 
  ∃ (count : ℕ), count = 412 ∧ 
  (∀ (N : ℕ), N < 2000 → (∃ x : ℝ, x > 0 ∧ x < 2000 ∧ x ^ (⌊x⌋₊) = N) ↔ N < count) :=
sorry

end count_N_less_than_2000_l86_86302


namespace number_of_routes_P_to_Q_l86_86769

noncomputable def number_of_routes (paths_from_P_to_T : ℕ) (paths_from_T_to_Q : ℕ) : ℕ :=
  paths_from_P_to_T * paths_from_T_to_Q

theorem number_of_routes_P_to_Q :
  number_of_routes 4 4 = 16 := by
  simp [number_of_routes]
  sorry

end number_of_routes_P_to_Q_l86_86769


namespace non_attacking_rooks_ways_l86_86716

def total_chessboard_squares := 64

def non_attacking_rook_placements (total_squares : ℕ) : ℕ :=
  let first_rook_placements := total_squares in
  let second_rook_placements := total_squares - 15 in
  first_rook_placements * second_rook_placements

theorem non_attacking_rooks_ways:
  non_attacking_rook_placements total_chessboard_squares = 3136 :=
by
  sorry

end non_attacking_rooks_ways_l86_86716


namespace find_a_l86_86325

theorem find_a (a : ℝ) (h₁ : a > 1) (h₂ : (∀ x : ℝ, a^3 = 8)) : a = 2 :=
by
  sorry

end find_a_l86_86325


namespace lateral_surface_areas_equal_implies_m_value_l86_86129

variables {a m : ℚ}

def base_area (a : ℚ) := (real.sqrt 3 / 4) * a^2

def volume_prism (a m : ℚ) := base_area a * m

def lateral_surface_area_prism (a m : ℚ) := 3 * a * m

def total_surface_area_prism (a m : ℚ) :=
  lateral_surface_area_prism a m + 2 * base_area a

def slant_height_pyramid (a m : ℚ) :=
  real.sqrt (m^2 + (a^2 / 12))

def volume_pyramid (a m : ℚ) := (1 / 3) * base_area a * m

def lateral_surface_area_pyramid (a m : ℚ) :=
  3/2 * a * slant_height_pyramid a m

def total_surface_area_pyramid (a m : ℚ) :=
  lateral_surface_area_pyramid a m + base_area a

theorem lateral_surface_areas_equal_implies_m_value (a : ℚ) :
  lateral_surface_area_prism a m = lateral_surface_area_pyramid a m →
  m = a / 6 :=
by {
  sorry
}

end lateral_surface_areas_equal_implies_m_value_l86_86129


namespace Tim_total_payment_l86_86481

-- Define the context for the problem
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.3

-- Define the total amount paid as the sum of the manicure cost and the tip
def total_amount_paid (cost : ℝ) (tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- The theorem to be proven
theorem Tim_total_payment : total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end Tim_total_payment_l86_86481


namespace shortest_path_intersecting_generating_lines_l86_86456

theorem shortest_path_intersecting_generating_lines 
  (r l : ℝ) (h_r : r = 2 / 3) (h_l : l = 2) :
  ∃ length : ℝ, length = 2 * Real.sqrt 3 ∧ 
    (shortest_path (cone_with_base_radius r) (cone_with_slant_height l)) = length :=
  sorry

end shortest_path_intersecting_generating_lines_l86_86456


namespace range_of_ω_l86_86016

theorem range_of_ω (ω : ℝ) (hω_pos : ω > 0)
  (h_incr : ∀ x y, -π/2 ≤ x → x ≤ y → y ≤ 2π/3 → ω * x < ω * y → sin(ω * x) ≤ sin(ω * y)) :
  ω ≤ 3 / 4 :=
sorry

end range_of_ω_l86_86016


namespace product_polynomials_l86_86938

theorem product_polynomials (x : ℝ) : 
  (1 + x^3) * (1 - 2 * x + x^4) = 1 - 2 * x + x^3 - x^4 + x^7 :=
by sorry

end product_polynomials_l86_86938


namespace safe_numbers_count_l86_86227

noncomputable theory

def is_safe (q n : ℕ) : Prop := 
  ∀ k : ℤ, n ≠ k * q ∧ n ≠ k * q + 1 ∧ n ≠ k * q + 2 ∧ n ≠ k * q + 3 ∧ 
  n ≠ k * q - 1 ∧ n ≠ k * q - 2 ∧ n ≠ k * q - 3

def is_safe_9 (n : ℕ) : Prop := is_safe 9 n
def is_safe_10 (n : ℕ) : Prop := is_safe 10 n
def is_safe_14 (n : ℕ) : Prop := is_safe 14 n

def count_safe_numbers : ℕ :=
  let lcm = Nat.lcm (Nat.lcm 9 10) 14 in
  let valid_residue_classes := 2 * 3 * 6 in
  let cycles := 20000 / lcm in
  valid_residue_classes * cycles

-- Proposition: The number of positive integers less than or equal to 20,000 which are 
-- simultaneously 9-safe, 10-safe, and 14-safe is 1116.
theorem safe_numbers_count : count_safe_numbers = 1116 :=
begin
  sorry
end

end safe_numbers_count_l86_86227


namespace find_n_in_geom_series_l86_86035

noncomputable def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem find_n_in_geom_series :
  ∃ n : ℕ, geom_sum 1 (1/2) n = 31 / 16 :=
sorry

end find_n_in_geom_series_l86_86035


namespace determine_a_b_l86_86321

def piecewise_function (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x + a else b * x - 1

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem determine_a_b (a b : ℝ) :
  is_odd_function (piecewise_function a b) → a = 1 ∧ b = 1 := by
  sorry

end determine_a_b_l86_86321


namespace students_per_bus_l86_86838

def total_students : ℕ := 360
def number_of_buses : ℕ := 8

theorem students_per_bus : total_students / number_of_buses = 45 :=
by
  sorry

end students_per_bus_l86_86838


namespace complex_pure_imaginary_l86_86428

theorem complex_pure_imaginary (m : ℝ) :
  ((m^2 - 2 * m - 3) = 0) ∧ ((m^2 - 4 * m + 3) ≠ 0) → m = -1 :=
by
  sorry

end complex_pure_imaginary_l86_86428


namespace sum_power_inequality_l86_86646

variables {n : ℕ} (k : ℕ)
variables {x y : Fin n.succ → ℝ}

-- Conditions
def condition1 (x y : Fin n.succ → ℝ) : Prop :=
  (∀ i j : Fin n.succ, i < j → x i > x j) ∧ (∀ i j : Fin n.succ, i < j → y i > y j)

def condition2 (x y : Fin n.succ → ℝ) : Prop :=
  (∀ m : Fin (n.succ + 1), (∑ i in Finset.range m.val, x ⟨i, Nat.lt_of_lt_succ m.is_lt⟩) > (∑ i in Finset.range m.val, y ⟨i, Nat.lt_of_lt_succ m.is_lt⟩))

theorem sum_power_inequality (k : ℕ) (x y : Fin n.succ → ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (∑ i : Fin n.succ, (x i)^k) > (∑ i : Fin n.succ, (y i)^k) := 
sorry

end sum_power_inequality_l86_86646


namespace total_messages_l86_86701

theorem total_messages (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  sorry

end total_messages_l86_86701


namespace units_digit_of_2_pow_20_minus_1_l86_86857

theorem units_digit_of_2_pow_20_minus_1 : (2^20 - 1) % 10 = 5 := 
  sorry

end units_digit_of_2_pow_20_minus_1_l86_86857


namespace geometric_sequence_common_ratio_l86_86263

theorem geometric_sequence_common_ratio (a₁ : ℝ) (q : ℝ) (S_n : ℕ → ℝ)
  (h₁ : S_n 3 = a₁ + a₁ * q + a₁ * q ^ 2)
  (h₂ : S_n 2 = a₁ + a₁ * q)
  (h₃ : S_n 3 / S_n 2 = 3 / 2) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l86_86263


namespace count_positive_integers_N_number_of_N_l86_86294

theorem count_positive_integers_N : ∀ N : ℕ, N < 2000 → ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N :=
begin
  sorry
end

theorem number_of_N : {N : ℕ // N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N}.card = 412 :=
begin
  sorry
end

end count_positive_integers_N_number_of_N_l86_86294


namespace animath_extortion_l86_86821

noncomputable def max_extortion (n : ℕ) : ℕ :=
2^n - n - 1 

theorem animath_extortion (n : ℕ) :
  ∃ steps : ℕ, steps < (2^n - n - 1) :=
sorry

end animath_extortion_l86_86821


namespace no_factors_l86_86961

noncomputable def p := (λ x : ℝ, x^4 - 3 * x^2 + 5)
noncomputable def p1 := (λ x : ℝ, x^2 + 1)
noncomputable def p2 := (λ x : ℝ, x - 1)
noncomputable def p3 := (λ x : ℝ, x^2 + 5)
noncomputable def p4 := (λ x : ℝ, x^2 + 2 * x + 1)

theorem no_factors : ¬(∃ q : ℝ → ℝ, p = λ x, (p1 x) * (q x)) ∧ 
                     ¬(∃ q : ℝ → ℝ, p = λ x, (p2 x) * (q x)) ∧ 
                     ¬(∃ q : ℝ → ℝ, p = λ x, (p3 x) * (q x)) ∧ 
                     ¬(∃ q : ℝ → ℝ, p = λ x, (p4 x) * (q x)) := by
  sorry

end no_factors_l86_86961


namespace morning_rowers_count_l86_86422

def number_afternoon_rowers : ℕ := 7
def total_rowers : ℕ := 60

def number_morning_rowers : ℕ :=
  total_rowers - number_afternoon_rowers

theorem morning_rowers_count :
  number_morning_rowers = 53 := by
  sorry

end morning_rowers_count_l86_86422


namespace probability_increasing_function_l86_86254

open ProbabilityTheory

variable {a b : ℝ}

theorem probability_increasing_function :
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) →
  (probability_space.probability {ab : ℝ × ℝ | ab.1 ∈ Set.Ioo 0 1 ∧ ab.2 ∈ Set.Ioo 0 1 ∧ ab.1 ≥ 2 * ab.2}) = 1 / 4 :=
by
  sorry

end probability_increasing_function_l86_86254


namespace abs_inequality_solution_l86_86033

theorem abs_inequality_solution (x : ℝ) : (|x + 3| > x + 3) ↔ (x < -3) :=
by
  sorry

end abs_inequality_solution_l86_86033


namespace reflect_triangle_final_position_l86_86471

variables {x1 x2 x3 y1 y2 y3 : ℝ}

-- Definition of reflection in x-axis and y-axis
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

theorem reflect_triangle_final_position (x1 x2 x3 y1 y2 y3 : ℝ) :
  (reflect_y (reflect_x x1 y1).1 (reflect_x x1 y1).2) = (-x1, -y1) ∧
  (reflect_y (reflect_x x2 y2).1 (reflect_x x2 y2).2) = (-x2, -y2) ∧
  (reflect_y (reflect_x x3 y3).1 (reflect_x x3 y3).2) = (-x3, -y3) :=
by
  sorry

end reflect_triangle_final_position_l86_86471


namespace largest_smallest_difference_l86_86846

theorem largest_smallest_difference :
  let digits := {9, 4, 3, 5} 
  let largest := 95
  let smallest := 34
  largest - smallest = 61 :=
by
  let digits := {9, 4, 3, 5}
  let largest := 95
  let smallest := 34
  show largest - smallest = 61
  exact Eq.refl 61

end largest_smallest_difference_l86_86846


namespace total_amount_paid_l86_86486

-- Definitions based on the conditions in part (a)
def cost_of_manicure : ℝ := 30
def tip_percentage : ℝ := 0.30

-- Proof statement based on conditions and answer in part (c)
theorem total_amount_paid : cost_of_manicure + (cost_of_manicure * tip_percentage) = 39 := by
  sorry

end total_amount_paid_l86_86486


namespace eval_ceil_floor_l86_86171

-- Definitions of the ceiling and floor functions and their relevant properties
def ceil (x : ℝ) : ℤ := ⌈x⌉
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Mathematical statement to prove
theorem eval_ceil_floor : ceil (7 / 3) + floor (- (7 / 3)) = 0 :=
by
  -- Proof is omitted
  sorry

end eval_ceil_floor_l86_86171


namespace time_to_meet_l86_86965

-- Definitions based on conditions
def motorboat_speed_Serezha : ℝ := 20 -- km/h
def crossing_time_Serezha : ℝ := 0.5 -- hours (30 minutes)
def running_speed_Dima : ℝ := 6 -- km/h
def running_time_Dima : ℝ := 0.25 -- hours (15 minutes)
def combined_speed : ℝ := running_speed_Dima + running_speed_Dima -- equal speeds running towards each other
def distance_meet : ℝ := (running_speed_Dima * running_time_Dima) -- The distance they need to cover towards each other

-- Prove the time for them to meet
theorem time_to_meet : (distance_meet / combined_speed) = (7.5 / 60) :=
by
  sorry

end time_to_meet_l86_86965


namespace continuously_differentiable_function_l86_86887

noncomputable def f : ℝ → ℝ := sorry

def D (a : ℝ) : Set (ℝ × ℝ × ℝ) :=
  { p | let (x, y, z) := p in
        x^2 + y^2 + z^2 ≤ a^2 ∧ |y| ≤ x/√3 }

theorem continuously_differentiable_function {f : ℝ → ℝ} (hf : ContinuousDifferentiable f)
    (cond : ∀ a ≥ 0, 
              ∫∫∫ (x y z) in D a, x * f (a * y / sqrt (x^2 + y^2)) =
              (π * a^3 / 8) * (f a + sin a - 1)) :
  (∀ t, f ∈ class_C2) ∧ f 0 = 1 ∧ f' 0 = 0 ∧ f' 2 0 = 0 :=
  sorry

end continuously_differentiable_function_l86_86887


namespace amy_balloons_l86_86361

theorem amy_balloons (james_balloons amy_balloons : ℕ) (h1 : james_balloons = 232) (h2 : james_balloons = amy_balloons + 131) :
  amy_balloons = 101 :=
by
  sorry

end amy_balloons_l86_86361


namespace incorrect_major_premise_l86_86477

-- Defining the types for line and plane
variable {Line Plane : Type}

-- The major premise: if a line is parallel to a plane, then this line is parallel to all lines within the plane
def major_premise (l : Line) (p : Plane) : Prop :=
  l ∥ p → ∀ l' : Line, l' ⊆ p → l ∥ l'

-- Given conditions
variables (a b : Line) (p : Plane)
variable (not_in_plane : b ∉ a)
variable (subset_plane : a ⊆ a)
variable (line_parallel_plane : b ∥ a)

-- The goal is to prove the major premise is incorrect
theorem incorrect_major_premise : ¬ (major_premise b a) :=
sorry

end incorrect_major_premise_l86_86477


namespace negation_of_at_most_one_obtuse_l86_86449

-- Defining a predicate to express the concept of an obtuse angle
def is_obtuse (θ : ℝ) : Prop := θ > 90

-- Defining a triangle with three interior angles α, β, and γ
structure Triangle :=
  (α β γ : ℝ)
  (sum_angles : α + β + γ = 180)

-- Defining the condition that "At most, only one interior angle of a triangle is obtuse"
def at_most_one_obtuse (T : Triangle) : Prop :=
  (is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ ¬ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ is_obtuse T.β ∧ ¬ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ ¬ is_obtuse T.γ)

-- The theorem we want to prove: Negation of "At most one obtuse angle" is "At least two obtuse angles"
theorem negation_of_at_most_one_obtuse (T : Triangle) :
  ¬ at_most_one_obtuse T ↔ (is_obtuse T.α ∧ is_obtuse T.β) ∨ (is_obtuse T.α ∧ is_obtuse T.γ) ∨ (is_obtuse T.β ∧ is_obtuse T.γ) := by
  sorry

end negation_of_at_most_one_obtuse_l86_86449


namespace allison_greater_than_brian_and_noah_l86_86924

noncomputable def allison_faces : list ℕ := [6, 6, 6, 6, 6, 6]
noncomputable def brian_faces : list ℕ := [2, 2, 4, 4, 6, 6]
noncomputable def noah_faces : list ℕ := [4, 4, 4, 8, 8, 8]

noncomputable def probability_allison_greater : ℚ :=
  let p_brian := (brian_faces.countp (λ x, x < 6)) / (brian_faces.length) in
  let p_noah := (noah_faces.countp (λ x, x < 6)) / (noah_faces.length) in
  p_brian * p_noah

theorem allison_greater_than_brian_and_noah :
  probability_allison_greater = 1 / 3 := by
  sorry

end allison_greater_than_brian_and_noah_l86_86924


namespace cosine_sum_simplification_l86_86418

theorem cosine_sum_simplification (x : ℝ) (k : ℤ) : 
  cos ((6 * k + 1) * π / 3 + x) + cos ((6 * k - 1) * π / 3 + x) = cos x :=
by
  sorry

end cosine_sum_simplification_l86_86418


namespace number_of_new_trailers_l86_86842

noncomputable def average_age (total_age : ℕ) (num_trailers : ℕ) : ℕ :=
total_age / num_trailers

theorem number_of_new_trailers :
  ∀ (n : ℕ),
    let avg_age_old := 15 in
    let number_old := 30 in
    let years_elapsed := 3 in
    let current_age_old := avg_age_old + years_elapsed in
    let total_age_old := number_old * current_age_old in
    let current_age_new := years_elapsed in
    let avg_current_age := 12 in
    (average_age (total_age_old + n * current_age_new) (number_old + n) = avg_current_age) → n = 20 :=
sorry

end number_of_new_trailers_l86_86842


namespace limit_tangent_product_l86_86979

theorem limit_tangent_product : 
  (∃ l, tendsto (λ x: ℝ, (2 - x) * tan (π / 4 * x)) (𝓝 2) (𝓝 l) ∧ l = 4 / π) :=
sorry

end limit_tangent_product_l86_86979


namespace S_is_circle_iff_l86_86893

noncomputable def great_circle_distance (X Y : pts_on_sphere) : Real := sorry

noncomputable def S (A B : pts_on_sphere) (k : Real) : set pts_on_sphere := 
  {P | great_circle_distance A P + great_circle_distance B P = k}

theorem S_is_circle_iff (A B : pts_on_sphere) (k : Real) : 
  (∃ C : set pts_on_sphere, C = S A B k ∧ is_circle C) ↔ (k = π ∧ ¬antipodal A B) := 
sorry

end S_is_circle_iff_l86_86893


namespace balls_in_boxes_with_one_empty_l86_86681

-- Question setup:
-- 5 distinguishable balls, 4 distinguishable boxes, at least one box must remain empty.
def num_ways_distribute_balls (balls boxes : ℕ) : ℕ :=
  let unrestricted := boxes ^ balls in
  let one_empty := boxes * (boxes - 1) ^ balls in
  let two_empty := (boxes * (boxes - 1) // 2) * (boxes - 2) ^ balls in
  let three_empty := boxes * (boxes - 1) // 6 * (boxes - 3) ^ balls in
  unrestricted - one_empty + two_empty - three_empty

-- The final theorem we want to prove
theorem balls_in_boxes_with_one_empty :
  num_ways_distribute_balls 5 4 = 240 :=
by {
  -- Skipping the actual proof steps
  sorry,
}

end balls_in_boxes_with_one_empty_l86_86681


namespace cube_root_of_neg_eight_l86_86002

theorem cube_root_of_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by
  sorry

end cube_root_of_neg_eight_l86_86002


namespace equation_of_line_passing_through_center_and_perpendicular_l86_86615

def center_of_circle_eq (x y : ℝ) : Prop :=
  (x + 1) ^ 2 + y ^ 2 = 1

def perpendicular_to_line_eq (y x : ℝ) : Prop :=
  y = x + 1

theorem equation_of_line_passing_through_center_and_perpendicular :
  (∃ (x y : ℝ), center_of_circle_eq x y ∧ perpendicular_to_line_eq y x) → (∀ (x y : ℝ), x - y + 1 = 0 → x - y + 1 = 0) :=
begin
  sorry,
end

end equation_of_line_passing_through_center_and_perpendicular_l86_86615


namespace problem_a_b_c_l86_86998

theorem problem_a_b_c (a b c : ℝ) (h₁ : a = Real.log 3 / Real.log 2) 
                                   (h₂ : b = 2.1^1.1) 
                                   (h₃ : c = Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10) : b > a ∧ a > c :=
by
  sorry

end problem_a_b_c_l86_86998


namespace marshmallows_needed_l86_86570

theorem marshmallows_needed (total_campers : ℕ) (boys_percentage : ℝ) (girls_percentage : ℝ)
  (boys_toast_percentage : ℝ) (girls_toast_percentage : ℝ) (each_camper_gets_one : ℕ) :
  total_campers = 180 → boys_percentage = 0.60 → girls_percentage = 0.40 →
  boys_toast_percentage = 0.55 → girls_toast_percentage = 0.80 →
  each_camper_gets_one = 1 →
  let boys := total_campers * boys_percentage in
  let girls := total_campers * girls_percentage in
  let boys_toasting := (boys * boys_toast_percentage).to_nat in
  let girls_toasting := (girls * girls_toast_percentage).to_nat in
  boys_toasting + girls_toasting = 117 :=
by
  intros
  sorry

end marshmallows_needed_l86_86570


namespace count_positive_integers_N_number_of_N_l86_86295

theorem count_positive_integers_N : ∀ N : ℕ, N < 2000 → ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N :=
begin
  sorry
end

theorem number_of_N : {N : ℕ // N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N}.card = 412 :=
begin
  sorry
end

end count_positive_integers_N_number_of_N_l86_86295


namespace range_of_a_l86_86658

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^3 else -x^3

theorem range_of_a (a : ℝ) (h : f (3 * a - 1) ≥ 8 * f a) : a ∈ Set.Iic (1 / 5) ∪ Set.Ici 1 := by
  sorry

end range_of_a_l86_86658


namespace inequality_proof_l86_86628

theorem inequality_proof (x : ℝ) (hx1 : x ≥ -1/2) (hx2 : x ≠ 0) :
  (4 * x^2) / (1 - Real.sqrt (1 + 2 * x))^2 < 2 * x + 9 ↔ x ∈ Set.Ici (-1/2) ∩ Set.Iio (45/8) \ {0} := by
  split
  { intro h,
    split,
    { exact hx1 },
    { split,
      { use x,
        assumption, },
      { intro h1, exact hx2 h1 },
    },
  sorry
  { sorry }

end inequality_proof_l86_86628


namespace range_of_a_l86_86273

def f (x : Real) (a : Real) : Real :=
  if 0 ≤ x ∧ x < 1 then (1/3) * x^3 - a * x + 1 else a * Real.log x

theorem range_of_a (a : Real) (h₀ : 0 < a) (h₁ : a ≤ 4/3) :
  ∀ x : Real, f x a ≥ 0 :=
sorry

end range_of_a_l86_86273


namespace minimize_quadratic_expression_l86_86228

def quadratic_expression (c : ℝ) : ℝ :=
  (3 / 4) * c^2 - 6 * c + 4

theorem minimize_quadratic_expression :
  ∃ c : ℝ, (c = 4 ∧ ∀ x : ℝ, quadratic_expression c ≤ quadratic_expression x) :=
by
  sorry

end minimize_quadratic_expression_l86_86228


namespace min_PA_plus_PO_l86_86258

variables {F O A P B : Type} [metric_space F] [metric_space O] [metric_space A] [metric_space P] [metric_space B]
variable [has_dist A F]

def parabola (x y : ℝ) := y^2 = -8*x

def point F : Prod ℝ ℝ := (0, 2)
def point B := (4, 0)
def point A : Prod ℝ ℝ := (-2, 4)
def point O : Prod ℝ ℝ := (0, 0)
def point P : Set (Prod ℝ ℝ) := {p | parabola p.fst p.snd = -x}

def directrix (x : ℝ) := -2

def distance (p1 p2 : Prod ℝ ℝ) := Real.Sqrt ((p2.fst - p1.fst)^2 + (p2.snd - p1.snd)^2)

theorem min_PA_plus_PO {a b c d : ℝ} (hx : Point A = (-2, 4)) (hb : Point B = (4, 0)) 
(hf : Point F = (0, 2)) : 
  min ((distance P A) + (distance P O)) (distance A B) = 2 * Real.Sqrt 13 := 
sorry

end min_PA_plus_PO_l86_86258


namespace problem1_problem2_l86_86940

theorem problem1 : 24 - (-16) + (-25) - 15 = 0 :=
by
  sorry

theorem problem2 : (-81) + 2 * (1 / 4) * (4 / 9) / (-16) = -81 - (1 / 16) :=
by
  sorry

end problem1_problem2_l86_86940


namespace find_triangle_l86_86653

theorem find_triangle (q : ℝ) (triangle : ℝ) (h1 : 3 * triangle * q = 63) (h2 : 7 * (triangle + q) = 161) : triangle = 1 :=
sorry

end find_triangle_l86_86653


namespace officers_count_l86_86902

-- Define the number of members in the club
def num_members : ℕ := 12

-- Define the number of positions to be filled
def num_positions : ℕ := 5

-- Theorem statement: There are 95,040 ways to choose the officers.
theorem officers_count :
  (Finset.range num_members).card.choose num_positions = 95_040 := 
sorry

end officers_count_l86_86902


namespace sequence_periodicity_l86_86143

-- Define the sequence recursively
def sequence (u : ℕ → ℝ) (b : ℝ) :=
  u 1 = b ∧ (b > 1) ∧ (∀ n ≥ 1, u (n + 1) = -1 / (u n + 2))

-- Statement of the problem
theorem sequence_periodicity (b : ℝ) (h : b > 1) (u : ℕ → ℝ) 
  (H : sequence u b) : u 16 = b := sorry

end sequence_periodicity_l86_86143


namespace average_value_eq_l86_86576

variable (x : ℝ)

theorem average_value_eq :
  ( -4 * x + 0 + 4 * x + 12 * x + 20 * x ) / 5 = 6.4 * x :=
by
  sorry

end average_value_eq_l86_86576


namespace product_of_numbers_l86_86515

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 :=
by
  sorry

end product_of_numbers_l86_86515


namespace regular_decagon_triangle_probability_l86_86994

theorem regular_decagon_triangle_probability :
  let total_triangles := Nat.choose 10 3
  let favorable_triangles := 10
  let probability := favorable_triangles / total_triangles
  probability = (1 : ℚ) / 12 :=
by
  sorry

end regular_decagon_triangle_probability_l86_86994


namespace find_w1_w2_l86_86594

theorem find_w1_w2 :
  ∃ (w1 w2 : ℤ), 
    (w1 + w2 = 3) ∧ 
    (∃ (λ : ℤ), w1 = 4 * λ ∧ w2 = -5 * λ) ∧ 
    (w1 = -12 ∧ w2 = 15) :=
by 
  use -12, 15,
  split,
  { show -12 + 15 = 3, exact rfl },
  {
    use -3,
    split;
    exact rfl,
  }

end find_w1_w2_l86_86594


namespace combined_avg_score_l86_86871

-- Define the average scores
def avg_score_u : ℕ := 65
def avg_score_b : ℕ := 80
def avg_score_c : ℕ := 77

-- Define the ratio of the number of students
def ratio_u : ℕ := 4
def ratio_b : ℕ := 6
def ratio_c : ℕ := 5

-- Prove the combined average score
theorem combined_avg_score : (ratio_u * avg_score_u + ratio_b * avg_score_b + ratio_c * avg_score_c) / (ratio_u + ratio_b + ratio_c) = 75 :=
by
  sorry

end combined_avg_score_l86_86871


namespace binomial_expansion_five_l86_86983

open Finset

theorem binomial_expansion_five (a b : ℝ) : 
  (a + b)^5 = a^5 + 5 * a^4 * b + 10 * a^3 * b^2 + 10 * a^2 * b^3 + 5 * a * b^4 + b^5 := 
by sorry

end binomial_expansion_five_l86_86983


namespace num_foxes_l86_86338

structure Creature :=
  (is_squirrel : Bool)
  (is_fox : Bool)
  (is_salamander : Bool)

def Anna : Creature := sorry
def Bob : Creature := sorry
def Cara : Creature := sorry
def Daniel : Creature := sorry

def tells_truth (c : Creature) : Bool :=
  c.is_squirrel || (c.is_salamander && ¬c.is_fox)

def Anna_statement : Prop := Anna.is_fox ≠ Daniel.is_fox
def Bob_statement : Prop := tells_truth Bob ↔ Cara.is_salamander
def Cara_statement : Prop := tells_truth Cara ↔ Bob.is_fox
def Daniel_statement : Prop := tells_truth Daniel ↔ (Anna.is_squirrel ∧ Bob.is_squirrel ∧ Cara.is_squirrel ∨ Daniel.is_squirrel)

theorem num_foxes :
  (Anna.is_fox + Bob.is_fox + Cara.is_fox + Daniel.is_fox = 2) :=
  sorry

end num_foxes_l86_86338


namespace greatest_number_s_l86_86082

theorem greatest_number_s (n : ℕ) (h₁ : n = 64) (h₂ : ∀ k, 0 < k → k < n → 68 + 4 * k ∈ s) :
  ∃ m, is_greatest s m ∧ m = 320 :=
by
  sorry

end greatest_number_s_l86_86082


namespace evaluate_expression_l86_86200

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l86_86200


namespace total_payment_l86_86483

theorem total_payment (manicure_cost : ℚ) (tip_percentage : ℚ) (h_manicure_cost : manicure_cost = 30) (h_tip_percentage : tip_percentage = 30) : 
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := 
by 
  sorry

end total_payment_l86_86483


namespace sin_neg_10_over_3_pi_l86_86619

theorem sin_neg_10_over_3_pi : Real.sin (-10 / 3 * Real.pi) = Math.sqrt 3 / 2 :=
by
  sorry

end sin_neg_10_over_3_pi_l86_86619


namespace count_valid_N_l86_86300

theorem count_valid_N : 
  ∃ (N : ℕ), (N < 2000) ∧ (∃ (x : ℝ), x^⌊x⌋₊ = N) :=
begin
  sorry
end

end count_valid_N_l86_86300


namespace winning_candidate_percentage_l86_86527

def votes_candidate1 := 2500
def votes_candidate2 := 5000
def votes_candidate3 := 20000
def total_votes := votes_candidate1 + votes_candidate2 + votes_candidate3

theorem winning_candidate_percentage :
  (votes_candidate3: ℝ) / (total_votes: ℝ) * 100 ≈ 72.73 := by sorry

end winning_candidate_percentage_l86_86527


namespace triangles_and_parallel_conditions_l86_86354

variables {A B C P Q M N : Point}

-- Conditions
def is_on_side_of (p q r : Point) : Prop := -- p is on the line segment qr
sorry

def is_acute_triangle (a b c : Point) : Prop := -- triangle abc is acute
sorry

def is_altitude (a b c m : Point) : Prop := -- m is the foot of altitude from a to line bc
sorry

def is_circumsizable_quadrilateral (a p q c : Point) : Prop := -- quadrilateral apqc is cyclic
sorry

-- Prove that MN is parallel to AC
theorem triangles_and_parallel_conditions (h₁ : is_on_side_of P A B) (h₂ : is_on_side_of Q B C)
(h₃ : is_acute_triangle B P Q)
(h₄ : is_altitude P B Q M) (h₅ : is_altitude Q P B N)
(h₆ : is_circumsizable_quadrilateral A P Q C) :
parallel MN AC :=
sorry

end triangles_and_parallel_conditions_l86_86354


namespace negation_equivalence_l86_86026

theorem negation_equivalence : (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
  sorry

end negation_equivalence_l86_86026


namespace problem_statement_l86_86755

-- Define the roots of the quadratic as r and s
variables (r s : ℝ)

-- Given conditions
def root_condition (r s : ℝ) := (r + s = 2 * Real.sqrt 6) ∧ (r * s = 3)

theorem problem_statement (h : root_condition r s) : r^8 + s^8 = 93474 :=
sorry

end problem_statement_l86_86755


namespace desired_interest_rate_l86_86536

def nominalValue : ℝ := 20
def dividendRate : ℝ := 0.09
def marketValue : ℝ := 15

theorem desired_interest_rate : (dividendRate * nominalValue / marketValue) * 100 = 12 := by
  sorry

end desired_interest_rate_l86_86536


namespace smallest_rel_prime_210_l86_86987

theorem smallest_rel_prime_210 : ∃ (y : ℕ), y > 1 ∧ Nat.gcd y 210 = 1 ∧ (∀ z : ℕ, z > 1 ∧ Nat.gcd z 210 = 1 → y ≤ z) ∧ y = 11 :=
by {
  sorry -- proof to be filled in
}

end smallest_rel_prime_210_l86_86987


namespace evaluate_expression_l86_86196

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l86_86196


namespace shaded_region_area_l86_86211

theorem shaded_region_area (a b : ℕ) (H : a = 2) (K : b = 4) :
  let s := a + b
  let area_square_EFGH := s * s
  let area_smaller_square_FG := a * a
  let area_smaller_square_EF := b * b
  let shaded_area := area_square_EFGH - (area_smaller_square_FG + area_smaller_square_EF)
  shaded_area = 16 := 
by
  sorry

end shaded_region_area_l86_86211


namespace total_payment_l86_86485

theorem total_payment (manicure_cost : ℚ) (tip_percentage : ℚ) (h_manicure_cost : manicure_cost = 30) (h_tip_percentage : tip_percentage = 30) : 
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := 
by 
  sorry

end total_payment_l86_86485


namespace daily_earnings_r_l86_86081

theorem daily_earnings_r (p q r s : ℝ)
  (h1 : p + q + r + s = 300)
  (h2 : p + r = 120)
  (h3 : q + r = 130)
  (h4 : s + r = 200)
  (h5 : p + s = 116.67) : 
  r = 75 :=
by
  sorry

end daily_earnings_r_l86_86081


namespace collinear_iff_real_simple_ratio_l86_86411

theorem collinear_iff_real_simple_ratio 
  {a b c : ℂ} 
  (h : (a - b) / (a - c) ∈ ℝ) : 
  ∃ k : ℝ, b = a + k * (c - a) :=
sorry

end collinear_iff_real_simple_ratio_l86_86411


namespace find_f_3_l86_86999

def f (x : ℝ) : ℝ := x^2 + 4 * x + 8

theorem find_f_3 : f 3 = 29 := by
  sorry

end find_f_3_l86_86999


namespace find_speed_ratio_l86_86493

noncomputable def circular_track_speed_ratio (v_V v_P C : ℝ) (H1 : v_V > 0) (H2 : v_P > 0) : Prop :=
  let t_1 := C / (v_V + v_P)
  let t_2 := (C * (2 * v_V + v_P)) / (v_V * (v_V + v_P))
  ∃ (r : ℝ), r = (1 + Real.sqrt 5) / 2 ∧ v_V / v_P = r

theorem find_speed_ratio
  (v_V v_P C : ℝ) (H1 : v_V > 0) (H2 : v_P > 0)
  (meeting1 : v_V * (C / (v_V + v_P)) + v_P * (C / (v_V + v_P)) = C)
  (lap_vasya : v_V * ((C * (2 * v_V + v_P)) / (v_V * (v_V + v_P))) = C + v_V * (C / (v_V + v_P)))
  (lap_petya : v_P * ((C * (2 * v_P + v_V)) / (v_P * (v_V + v_P))) = C + v_P * (C / (v_V + v_P))) :
  ∃ (r : ℝ), r = (1 + Real.sqrt 5) / 2 ∧ v_V / v_P = r :=
  sorry

end find_speed_ratio_l86_86493


namespace tax_rate_correct_l86_86549
-- Import the necessary Lean library

-- Define the conditions and the problem
theorem tax_rate_correct :
  let non_taxable_amount := 600
  let total_value := 1720
  let tax_paid := 78.4
  let excess_value := total_value - non_taxable_amount
  let tax_rate := (tax_paid / excess_value) * 100
  in tax_rate = 7 :=
by
  sorry

end tax_rate_correct_l86_86549


namespace sum_of_interior_angles_of_hexagon_l86_86467

theorem sum_of_interior_angles_of_hexagon
  (n : ℕ)
  (h : n = 6) :
  (n - 2) * 180 = 720 := by
  sorry

end sum_of_interior_angles_of_hexagon_l86_86467


namespace ceiling_plus_floor_eq_zero_l86_86187

theorem ceiling_plus_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (-7 / 3)) = 0 := by
  sorry

end ceiling_plus_floor_eq_zero_l86_86187


namespace lowest_score_dropped_l86_86874

theorem lowest_score_dropped (scores : Fin 4 → ℝ)
  (h1 : (∑ i, scores i) / 4 = 90)
  (h2 : (∑ i in Finset.univ.erase (Finset.univ.min' Finset.univ_nonempty), scores i) / 3 = 95) :
  (Finset.univ.min' Finset.univ_nonempty • scores) = 75 := 
by
  sorry

end lowest_score_dropped_l86_86874


namespace PM_perpendicular_to_AB_l86_86756

open EuclideanGeometry

-- Let AB be the diameter of a semicircle
variables (A B : Point) (M : Point)
-- M is a point on AB
variable (M_on_AB : PointOn M (Segment A B))
-- Points C, D, E, F lie on the semicircle
variables (semicircle : Semicircle A B) (C D E F : Point)
-- Angles conditions
variable (angle_AMD_equals_EMB : ∠ A M D = ∠ E M B)
variable (angle_CMA_equals_FMB : ∠ C M A = ∠ F M B)
-- P is the intersection point of the lines CD and EF
variable (P : Point)
variable (P_intersection : Intersection (Line C D) (Line E F) P)

-- Prove that PM is perpendicular to AB
theorem PM_perpendicular_to_AB : Perpendicular (Line P M) (Line A B) :=
by sorry

end PM_perpendicular_to_AB_l86_86756


namespace difference_is_167_l86_86045

-- Define the number of boys and girls in each village
def A_village_boys : ℕ := 204
def A_village_girls : ℕ := 468
def B_village_boys : ℕ := 334
def B_village_girls : ℕ := 516
def C_village_boys : ℕ := 427
def C_village_girls : ℕ := 458
def D_village_boys : ℕ := 549
def D_village_girls : ℕ := 239

-- Define total number of boys and girls
def total_boys := A_village_boys + B_village_boys + C_village_boys + D_village_boys
def total_girls := A_village_girls + B_village_girls + C_village_girls + D_village_girls

-- Define the difference between total girls and total boys
def difference := total_girls - total_boys

-- The theorem to prove the difference is 167
theorem difference_is_167 : difference = 167 := by
  sorry

end difference_is_167_l86_86045


namespace square_divide_into_9_triangles_l86_86954

-- Definition of side length of the square
def side_length : ℝ := a

-- Definition of congruent right triangles and the different triangle
def is_congruent (t1 t2 : Triangle) : Prop := 
  t1.legs = t2.legs ∧ t1.hypotenuse = t2.hypotenuse

def right_triangle (t : Triangle) : Prop := 
  t.angle = 90 ∧ t.area = (side_length^2 / 4)

def different_triangle (t : Triangle) : Prop := 
  t.angle = 90 ∧ t.area ≠ (side_length^2 / 4)

-- Main theorem statement
theorem square_divide_into_9_triangles (a : ℝ) (square : Square) :
  ∃ triangles : List Triangle,
    length triangles = 9
    ∧ (∀ t ∈ triangles.take 8, right_triangle t)
    ∧ (∀ t1 t2 ∈ triangles.take 8, t1 ≠ t2 → is_congruent t1 t2)
    ∧ (different_triangle (triangles.nth_le 8 sorry)) :=
sorry

end square_divide_into_9_triangles_l86_86954


namespace min_marked_cells_15x15_l86_86505

def is_valid_grid (grid : Matrix (Fin 15) (Fin 15) ℕ) : Prop :=
∀ (i : Fin 15), (∃ j₁ : Fin 10, grid i (Fin.ofNat (i + j₁)) ≠ 0) ∧ (∀ j₁ : Fin 10, j₁ + 10 < 15 → grid (Fin.ofNat (i + j₁)) j₁ ≠ 0)

theorem min_marked_cells_15x15 : ∃ (cells : Finset (Fin 15 × Fin 15)), 
  is_valid_grid (λ i j, if (i, j) ∈ cells then 1 else 0) ∧ cells.card = 20 := sorry

end min_marked_cells_15x15_l86_86505


namespace distance_traveled_downstream_l86_86831

-- Conditions
def speed_boat_still_water : ℝ := 20
def rate_current : ℝ := 5
def time_minutes : ℝ := 27
def time_hours : ℝ := time_minutes / 60

-- Hypothesis
def effective_speed_downstream : ℝ := speed_boat_still_water + rate_current

-- Goal statement
theorem distance_traveled_downstream : effective_speed_downstream * time_hours = 11.25 := by
  sorry

end distance_traveled_downstream_l86_86831


namespace supermarkets_in_us_l86_86474

noncomputable def number_of_supermarkets_in_canada : ℕ := 35
noncomputable def number_of_supermarkets_total : ℕ := 84
noncomputable def diff_us_canada : ℕ := 14
noncomputable def number_of_supermarkets_in_us : ℕ := number_of_supermarkets_in_canada + diff_us_canada

theorem supermarkets_in_us : number_of_supermarkets_in_us = 49 := by
  sorry

end supermarkets_in_us_l86_86474


namespace find_caramel_boxes_l86_86778

-- Definitions for the conditions.
def chocolate_boxes : ℕ := 6
def pieces_per_box : ℕ := 9
def total_candies : ℕ := 90

-- The problem statement.
theorem find_caramel_boxes : 
  let chocolate_candies := chocolate_boxes * pieces_per_box in
  let caramel_candies := total_candies - chocolate_candies in
  let caramel_boxes := caramel_candies / pieces_per_box in
  caramel_boxes = 4 := 
by
  sorry

end find_caramel_boxes_l86_86778


namespace positive_difference_median_mode_l86_86504

-- Define the data from the stem and leaf plot
def data_set := [10, 11, 12, 13, 21, 21, 21, 22, 23, 28, 29, 32, 38, 39, 53, 58, 59]

-- Function to compute the median of a list of numbers
noncomputable def median (l : List Int) : Int :=
  let sorted := l.qsort (· < ·)
  sorted[(sorted.length / 2 : Nat)]

-- Function to compute the mode of a list of numbers
noncomputable def mode (l : List Int) : Int :=
  l.foldl (fun (modes : List (Int × Nat)) x =>
    let count := modes.lookup x |>.getD 0
    (modes.erase x).cons (x, count + 1)
  ) [].qsort (·.snd > ·.snd) |>.head! |>.fst

-- Positive difference between two numbers
def positive_difference (a b : Int) : Int :=
  if a > b then a - b else b - a

-- The statement to prove
theorem positive_difference_median_mode :
  positive_difference (median data_set) (mode data_set) = 1 := by
  sorry

end positive_difference_median_mode_l86_86504


namespace percentage_change_area_decrease_36_percent_l86_86880

-- Definitions of initial and new dimensions
variables (L W : ℝ) (initial_area new_area : ℝ)

-- Condition: initial area
def A_initial : ℝ := L * W

-- Condition: new length and width
def L_new : ℝ := 1.6 * L
def W_new : ℝ := 0.4 * W

-- Condition: new area
def A_new : ℝ := L_new * W_new

-- Theorem: percentage change in the area
theorem percentage_change_area_decrease_36_percent
  (L W : ℝ) (h_initial_area : A_initial = L * W)
  (h_L_new : L_new = 1.6 * L)
  (h_W_new : W_new = 0.4 * W)
  (h_new_area : A_new = L_new * W_new):
  ((A_new - A_initial) / A_initial) * 100 = -36 := 
by
  sorry

end percentage_change_area_decrease_36_percent_l86_86880


namespace correct_exponent_operation_l86_86071

theorem correct_exponent_operation (x : ℝ) : x ^ 3 * x ^ 2 = x ^ 5 :=
by sorry

end correct_exponent_operation_l86_86071


namespace suitcase_combinations_l86_86766

theorem suitcase_combinations :
  let multiples_of_4 := {x | 1 ≤ x ∧ x ≤ 40 ∧ x % 4 = 0}.card,
      odd_numbers := {y | 1 ≤ y ∧ y ≤ 40 ∧ y % 2 = 1}.card,
      multiples_of_5 := {z | 1 ≤ z ∧ z ≤ 40 ∧ z % 5 = 0}.card in
  multiples_of_4 * odd_numbers * multiples_of_5 = 1600 :=
by
  sorry

end suitcase_combinations_l86_86766


namespace length_A_l86_86746

noncomputable def A : ℝ × ℝ := (0, 15)
noncomputable def B : ℝ × ℝ := (0, 18)
noncomputable def C : ℝ × ℝ := (4, 10)

-- Prove the length of A'B' given the conditions
theorem length_A'B'_eq : 
  ∃ (A' B' : ℝ × ℝ), 
  (A'.1 = A'.2) ∧ (B'.1 = B'.2) ∧ 
  (4 - A'.1) * (C.2 - A'.2) = (10 - A'.2) * (C.1 - A'.1) ∧
  (4 - B'.1) * (C.2 - B'.2) = (10 - B'.2) * (C.1 - B'.1) ∧
  (real.sqrt ((A'.1 - B'.1) ^ 2 + (A'.2 - B'.2) ^ 2) = 2 * real.sqrt 2 / 3) :=
begin
  sorry
end

end length_A_l86_86746


namespace original_numbers_l86_86148

noncomputable def find_numbers (A B C D E F : Nat) : Prop :=
  A + B + C + D + E + F = 21 ∧ -- Sum of 1 to 6
  (D + E + B = 14) ∧ -- Condition involving D, E, B
  (A + C = 3)  -- Condition involving A, C

theorem original_numbers :
  ∃ (A B C D E F : Nat), find_numbers A B C D E F ∧
    (A = 1) ∧
    (B = 3) ∧
    (C = 2) ∧
    (D = 5) ∧
    (E = 6) ∧
    (F = 4) :=
by
  existsi 1
  existsi 3
  existsi 2
  existsi 5
  existsi 6
  existsi 4
  dsimp [find_numbers]
  split
  { norm_num, }
  split
  { norm_num, }
  split
  { norm_num, }
  split
  { norm_num, }
  split
  { norm_num, }
  { norm_num, }
  sorry

end original_numbers_l86_86148


namespace nell_gave_jeff_168_cards_l86_86770

theorem nell_gave_jeff_168_cards (initial_cards : ℕ) (gave_to_john : ℕ) (remaining_cards : ℕ) : initial_cards = 573 → gave_to_john = 195 → remaining_cards = 210 → (initial_cards - gave_to_john - remaining_cards = 168) :=
by
  intros
  simp
  sorry

end nell_gave_jeff_168_cards_l86_86770


namespace apples_in_each_basket_after_sister_took_apples_l86_86735

theorem apples_in_each_basket_after_sister_took_apples 
  (total_apples : ℕ) 
  (number_of_baskets : ℕ) 
  (apples_taken_from_each : ℕ)
  (initial_apples_per_basket := total_apples / number_of_baskets)
  (final_apples_per_basket := initial_apples_per_basket - apples_taken_from_each) :
  total_apples = 64 → number_of_baskets = 4 → apples_taken_from_each = 3 → final_apples_per_basket = 13 := 
by 
  intros htotal hnumber htake
  rw [htotal, hnumber, htake]
  have initial_apples : initial_apples_per_basket = 16 := by norm_num
  rw initial_apples
  norm_num
  sorry

end apples_in_each_basket_after_sister_took_apples_l86_86735


namespace magnitude_of_complex_power_l86_86574

noncomputable def z := 1 - Real.sqrt(3) * Complex.I
def n := 4

theorem magnitude_of_complex_power :
  Complex.abs (z ^ n) = 16 := by
  sorry

end magnitude_of_complex_power_l86_86574


namespace parallelogram_vector_expressions_l86_86726

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions based on the given conditions
variables (a b : V) -- Vectors a and b representing AM and AN respectively

-- The statement we need to prove, mapping the given problem to Lean
theorem parallelogram_vector_expressions
  (AB AD MN BD : V)
  (h1 : AB = (9/8) • a - (3/8) • b)
  (h2 : AD = (9/8) • b - (3/8) • a)
  (h3 : MN = b - a)
  (h4 : BD = (3/2) • b - (3/2) • a) :
  True := 
begin
  sorry -- Proof omitted by instructions
end

end parallelogram_vector_expressions_l86_86726


namespace digits_in_product_l86_86375

def number_of_digits (n : ℕ) : ℕ :=
if n = 0 then 1 else Nat.log10 n + 1

def a : ℕ := 5987456123789012345
def b : ℕ := 67823456789

theorem digits_in_product : number_of_digits (a * b) = 33 :=
by sorry

end digits_in_product_l86_86375


namespace intersection_point_l86_86985

def satisfies_first_line (p : ℝ × ℝ) : Prop :=
  8 * p.1 - 5 * p.2 = 40

def satisfies_second_line (p : ℝ × ℝ) : Prop :=
  6 * p.1 + 2 * p.2 = 14

theorem intersection_point :
  satisfies_first_line (75 / 23, -64 / 23) ∧ satisfies_second_line (75 / 23, -64 / 23) :=
by 
  sorry

end intersection_point_l86_86985


namespace roller_coaster_cost_l86_86967

variable (ferris_wheel_rides : Nat) (log_ride_rides : Nat) (rc_rides : Nat)
variable (ferris_wheel_cost : Nat) (log_ride_cost : Nat)
variable (initial_tickets : Nat) (additional_tickets : Nat)
variable (total_needed_tickets : Nat)

theorem roller_coaster_cost :
  ferris_wheel_rides = 2 →
  log_ride_rides = 7 →
  rc_rides = 3 →
  ferris_wheel_cost = 2 →
  log_ride_cost = 1 →
  initial_tickets = 20 →
  additional_tickets = 6 →
  total_needed_tickets = initial_tickets + additional_tickets →
  let total_ride_costs := ferris_wheel_rides * ferris_wheel_cost + log_ride_rides * log_ride_cost
  let rc_cost := (total_needed_tickets - total_ride_costs) / rc_rides
  rc_cost = 5 := by
  sorry

end roller_coaster_cost_l86_86967


namespace solve_for_x_l86_86795

theorem solve_for_x (x : ℝ) : 5 + 3.4 * x = 2.1 * x - 30 → x = -26.923 := 
by 
  sorry

end solve_for_x_l86_86795


namespace crayons_birthday_l86_86779

theorem crayons_birthday (C E : ℕ) (hC : C = 523) (hE : E = 457) (hDiff : C = E + 66) : C = 523 := 
by {
  -- proof would go here
  sorry
}

end crayons_birthday_l86_86779


namespace sequence_property_l86_86075

def sequence (f : ℕ → ℤ) (c : ℤ) : Prop :=
  f 1 = 1 ∧
  f 2 = c ∧
  ∀ n, 2 ≤ n → f (n + 1) = 2 * (f n) -  (f (n - 1)) + 2

theorem sequence_property (c : ℤ) (hc : 0 < c) :
  ∀ k : ℕ, ∃ r : ℕ, ∀ f : ℕ → ℤ, sequence f c → f k * f (k + 1) = f r := by
  sorry

end sequence_property_l86_86075


namespace ceiling_plus_floor_eq_zero_l86_86186

theorem ceiling_plus_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (-7 / 3)) = 0 := by
  sorry

end ceiling_plus_floor_eq_zero_l86_86186


namespace find_n_l86_86068

theorem find_n (n : ℤ) (h : n + (n + 1) + (n + 2) = 9) : n = 2 :=
by
  sorry

end find_n_l86_86068


namespace prod_ge_exp_l86_86238

/-- 
Given 5n real numbers r_{i}, s_{i}, t_{i}, u_{i}, v_{i} (where 1 ≤ i ≤ n) 
all greater than 1, let R = (1 / n) * sum (λ i, r_{i}), 
S = (1 / n) * sum (λ i, s_{i}), T = (1 / n) * sum (λ i, t_{i}), 
U = (1 / n) * sum (λ i, u_{i}), V = (1 / n) * sum (λ i, v_{i}). 
Prove that: 
  ∏ i in (finset.range n), (r_{i} * s_{i} * t_{i} * u_{i} * v_{i} + 1) / (r_{i} * s_{i} * t_{i} * u_{i} * v_{i} - 1) 
   ≥ (R * S * T * U * V + 1) / (R * S * T * U * V - 1) ^ n
-/
theorem prod_ge_exp {n : ℕ} {r s t u v : fin n → ℝ} 
  (hr : ∀ i, 1 < r i) (hs : ∀ i, 1 < s i) (ht : ∀ i, 1 < t i) (hu : ∀ i, 1 < u i) (hv : ∀ i, 1 < v i) :
  let R := (1 / n) * ∑ i, r i,
      S := (1 / n) * ∑ i, s i,
      T := (1 / n) * ∑ i, t i,
      U := (1 / n) * ∑ i, u i,
      V := (1 / n) * ∑ i, v i in
  (∏ i in finset.range n, (r i * s i * t i * u i * v i + 1) / (r i * s i * t i * u i * v i - 1)) ≥ 
  ((R * S * T * U * V + 1) / (R * S * T * U * V - 1)) ^ n := 
sorry

end prod_ge_exp_l86_86238


namespace exponential_function_passes_through_fixed_point_l86_86638

theorem exponential_function_passes_through_fixed_point {a : ℝ} (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : 
  (a^(2 - 2) + 3) = 4 :=
by
  sorry

end exponential_function_passes_through_fixed_point_l86_86638


namespace simplify_trig_expression_l86_86793

theorem simplify_trig_expression (x : ℝ) :
  (2 + 3 * sin x - 4 * cos x) / (2 + 3 * sin x + 4 * cos x) = tan (x / 2) :=
by
  -- Conditions for the proof
  have h1 : sin x = 2 * sin (x / 2) * cos (x / 2) := sorry
  have h2 : cos x = 1 - 2 * sin (x / 2) ^ 2 := sorry
  -- Proof of the statement using the conditions
  sorry

end simplify_trig_expression_l86_86793


namespace bryden_receive_amount_l86_86101

theorem bryden_receive_amount :
  let face_value := 0.25
  let percent_multiplier := 1500 / 100
  let num_quarters := 5
  let total_face_value := num_quarters * face_value
  let total_amount := percent_multiplier * total_face_value
  total_amount = 18.75 := 
by
  let face_value := 0.25
  let percent_multiplier := 1500 / 100
  let num_quarters := 5
  let total_face_value := num_quarters * face_value
  let total_amount := percent_multiplier * total_face_value
  have h1 : face_value = 0.25 := rfl
  have h2 : percent_multiplier = 15 := by norm_num
  have h3 : num_quarters = 5 := rfl
  have h4 : total_face_value = 5 * 0.25 := rfl
  have h5 : total_face_value = 1.25 := by norm_num
  have h6 : total_amount = 15 * 1.25 := rfl
  have h7 : total_amount = 18.75 := by norm_num
  exact h7

end bryden_receive_amount_l86_86101


namespace pyramid_total_blocks_l86_86563

-- Define the number of layers in the pyramid
def num_layers : ℕ := 8

-- Define the block multiplier for each subsequent layer
def block_multiplier : ℕ := 5

-- Define the number of blocks in the top layer
def top_layer_blocks : ℕ := 3

-- Define the total number of sandstone blocks
def total_blocks_pyramid : ℕ :=
  let rec total_blocks (layer : ℕ) (blocks : ℕ) :=
    if layer = 0 then blocks
    else blocks + total_blocks (layer - 1) (blocks * block_multiplier)
  total_blocks (num_layers - 1) top_layer_blocks

theorem pyramid_total_blocks :
  total_blocks_pyramid = 312093 :=
by
  -- Proof omitted
  sorry

end pyramid_total_blocks_l86_86563


namespace fraction_trip_l86_86706

/-- 
Given the following conditions:
1. Ratios 4/5 of the students left on a field trip with the first vehicle.
2. Of the students who stayed behind, 1/3 did not want to go on the field trip at all.
3. When another vehicle was located, 1/2 of the students who did want to go on the field trip but had been left behind were able to join.
Prove that the fraction of the class that eventually ended up going on the field trip is 13/15.

Parameters:
x : ℕ - the total number of students in the class.

Proof:
fraction_of_class_that_went_on_trip = (4/5) * x + 1/2 * (2/15) * x = 13/15
-/
theorem fraction_trip (x: ℕ) (h1: 4/5 * x ∈ ℝ) (h2: 1/3 ∈ ℝ) (h3: 1/2 ∈ ℝ)
  (h4: 3/15 ∈ ℝ) (h5: 13/15 ∈ ℝ) : 
  ((4/5 : ℝ) * (x : ℝ) + (1/2 : ℝ) * ((2/15 : ℝ) * (x : ℝ))) / (x : ℝ) = (13/15 : ℝ) := 
by 
sorry

end fraction_trip_l86_86706


namespace proof_calculate_expr_l86_86134

def calculate_expr : Prop :=
  (4 + 4 + 6) / 3 - 2 / 3 = 4

theorem proof_calculate_expr : calculate_expr := 
by 
  sorry

end proof_calculate_expr_l86_86134


namespace johns_phone_price_l86_86919

-- Define Alan's phone price
def alans_price : ℝ := 2000

-- Define the percentage increase
def percentage_increase : ℝ := 2/100

-- Define John's phone price
def johns_price := alans_price * (1 + percentage_increase)

-- The main theorem
theorem johns_phone_price : johns_price = 2040 := by
  sorry

end johns_phone_price_l86_86919


namespace problem_1_problem_2_problem_3_l86_86622

-- Problem 1
theorem problem_1 (x : ℝ) (h : 4.8 - 3 * x = 1.8) : x = 1 :=
by { sorry }

-- Problem 2
theorem problem_2 (x : ℝ) (h : (1 / 8) / (1 / 5) = x / 24) : x = 15 :=
by { sorry }

-- Problem 3
theorem problem_3 (x : ℝ) (h : 7.5 * x + 6.5 * x = 2.8) : x = 0.2 :=
by { sorry }

end problem_1_problem_2_problem_3_l86_86622


namespace subcommittees_with_teacher_l86_86452

theorem subcommittees_with_teacher (committee : Finset ℕ) (teachers : Finset ℕ) 
  (h1 : committee.card = 10) (h2 : teachers.card = 4) 
  (h3 : teachers ⊆ committee) : 
  (∑ t in (committee.subsetsOfCard 4), (if (¬ (t ∩ teachers).nonempty then 1 else 0)) = 195 := 
by sorry

end subcommittees_with_teacher_l86_86452


namespace strictly_increasing_function_l86_86745

open Nat

noncomputable def f : ℕ → ℕ := sorry

theorem strictly_increasing_function (h_inc: ∀ m n, m < n → f(m) < f(n))
  (h_f2 : f(2) = 2)
  (h_mul : ∀ m n, gcd m n = 1 → f(m * n) = f(m) * f(n)) :
  ∀ n, f(n) = n := 
sorry

end strictly_increasing_function_l86_86745


namespace additional_miles_proof_l86_86631

-- Define the distances
def distance_to_bakery : ℕ := 9
def distance_bakery_to_grandma : ℕ := 24
def distance_grandma_to_apartment : ℕ := 27

-- Define the total distances
def total_distance_with_bakery : ℕ := distance_to_bakery + distance_bakery_to_grandma + distance_grandma_to_apartment
def total_distance_without_bakery : ℕ := 2 * distance_grandma_to_apartment

-- Define the additional miles
def additional_miles_with_bakery : ℕ := total_distance_with_bakery - total_distance_without_bakery

-- Theorem statement
theorem additional_miles_proof : additional_miles_with_bakery = 6 :=
by {
  -- Here should be the proof, but we insert sorry to indicate it's skipped
  sorry
}

end additional_miles_proof_l86_86631


namespace jellybeans_left_l86_86835

variable (initial_jellybeans : ℕ) (total_kids : ℕ) (absent_kids : ℕ) (jellybeans_per_kid : ℕ)

theorem jellybeans_left (initial_jellybeans total_kids absent_kids jellybeans_per_kid : ℕ): 
  initial_jellybeans = 100 → total_kids = 24 → absent_kids = 2 → jellybeans_per_kid = 3 → 
  initial_jellybeans - (total_kids - absent_kids) * jellybeans_per_kid = 34 :=
by
  intros h_init h_total h_absent h_jellybeans_per_kid
  rw [h_init, h_total, h_absent, h_jellybeans_per_kid]
  repeat rfl
  sorry

end jellybeans_left_l86_86835


namespace odd_integers_between_3000_and_6000_l86_86677

theorem odd_integers_between_3000_and_6000 :
  let is_odd (n : ℕ) := n % 2 = 1
  ∧ ∀ n, 3000 ≤ n ∧ n < 6000 → is_odd n ∧ ∀ i j k, (i ≠ j ∧ j ≠ k ∧ k ≠ i)
  → ∃ d1 d2 d3 d4, n = 1000 * d1 + 100 * d2 + 10 * d3 + d4 
  → n ∈ finset.range 3000 6000
  → finset.card n = 840 := 
begin
  sorry
end

end odd_integers_between_3000_and_6000_l86_86677


namespace number_of_correct_propositions_is_zero_l86_86926

-- Definitions of the conditions (propositions) 
def proposition1 (a b : Line) (α : Plane) :=
  (lineFormsEqualAnglesWithPlane a α ∧ lineFormsEqualAnglesWithPlane b α) → a.parallel b

def proposition2 (a b : Line) (α : Plane) :=
  (a.parallel b ∧ a.parallel α) → b.parallel α

def proposition3 (l : Line) (s : SlantLine) (α : Plane) :=
  (havePerpendicularProjectionsWithinPlane l s α) → l.perpendicular s

def proposition4 (A B : Point) (α : Plane) :=
  (distanceFromPlane A α = distanceFromPlane B α) → (AB.line.parallel α)

-- Main theorem stating the problem and the answer
theorem number_of_correct_propositions_is_zero :
  (∃ (a b : Line) (α : Plane), ¬ proposition1 a b α) ∧
  (∃ (a b : Line) (α : Plane), ¬ proposition2 a b α) ∧
  (∃ (l s : Line) (α : Plane), ¬ proposition3 l s α) ∧
  (∃ (A B : Point) (α : Plane), ¬ proposition4 A B α) →
  (numberOfCorrectPropositions = 0) :=
by
  sorry

end number_of_correct_propositions_is_zero_l86_86926


namespace g_is_zero_l86_86136

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (4 * (Real.sin x)^4 + (Real.cos x)^2) - 
  Real.sqrt (4 * (Real.cos x)^4 + (Real.sin x)^2)

theorem g_is_zero (x : ℝ) : g x = 0 := 
  sorry

end g_is_zero_l86_86136


namespace find_n_solution_l86_86620

theorem find_n_solution (n : ℚ) (h : (1 / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4)) : n = -4 / 3 :=
by
  sorry

end find_n_solution_l86_86620


namespace equation_of_ellipse_product_of_slopes_l86_86664

noncomputable def line_l : ℝ → ℝ := λ x, x + real.sqrt 6

def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 5

structure Ellipse :=
(a b : ℝ)
(h : a > b)
(eccentricity : ℝ)
(eccentricity_condition : eccentricity = real.sqrt(3) / 3)
(equation : ℝ → ℝ → Prop := λ x y, y^2 / a^2 + x^2 / b^2 = 1)

def ellipse_E : Ellipse :=
{ a := real.sqrt 3,
  b := real.sqrt 2,
  h := (by linarith [real.sqrt 3, real.sqrt 2]),
  eccentricity := real.sqrt(3) / 3,
  eccentricity_condition := rfl }

-- The proof of the first statement: equation of ellipse E
theorem equation_of_ellipse :
  ∃ e : Ellipse, e.equation = λ x y, y^2 / 3 + x^2 / 2 = 1 := 
by { use ellipse_E, sorry }

-- Necessary lemma that ensures P is a point on circle_O
lemma point_on_circle (P : ℝ × ℝ) : circle_O P.1 P.2 ↔ P.1^2 + P.2^2 = 5 := 
by { intro h, exact h, }

-- The proof of the second statement: product of the slopes of tangent lines
theorem product_of_slopes (P : ℝ × ℝ) (hP : circle_O P.1 P.2) :
  let k1 k2 : ℝ in k1 * k2 = -1 :=
by { sorry }

end equation_of_ellipse_product_of_slopes_l86_86664


namespace AK_bisects_BC_l86_86133

section problem

variables {A B C D E F K O : Type*}

-- Define the basic setup of the triangle and excircle
axiom excircle_of_triangle : triangle A B C → ∃ O : circle, tangent O BC ∧ tangent O CA ∧ tangent O AB

-- Define the points D, E, F of tangency
axiom points_of_tangency : ∀ (O : circle), tangent_point O BC = D ∧ tangent_point O CA = E ∧ tangent_point O AB = F

-- Define the intersection of lines OD and EF at point K
axiom intersection : ∀ (O : circle), meets (line O D) (line E F) = K

-- Formal statement of the theorem
theorem AK_bisects_BC (A B C D E F K : Type*) [excircle_of_triangle A B C O]
  (tangents : points_of_tangency O D E F) (intersect : intersection O K) : bisects (line A K) (segment B C) :=
sorry

end problem

end AK_bisects_BC_l86_86133


namespace total_amount_l86_86925

theorem total_amount (a b c : ℕ) (h1 : a * 5 = b * 3) (h2 : c * 5 = b * 9) (h3 : b = 50) :
  a + b + c = 170 := by
  sorry

end total_amount_l86_86925


namespace office_distance_eq_10_l86_86870

noncomputable def distance_to_office (D T : ℝ) : Prop :=
  D = 10 * (T + 10 / 60) ∧ D = 15 * (T - 10 / 60)

theorem office_distance_eq_10 (D T : ℝ) (h : distance_to_office D T) : D = 10 :=
by
  sorry

end office_distance_eq_10_l86_86870


namespace tangent_line_to_circle_l86_86240

theorem tangent_line_to_circle (A : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) :
  A = (1, 0) → C = (3, 4) → r = 2 →
  (∃ k : ℝ, (∀ x y : ℝ, x = 1 ∨ 3 * x - 4 * y - 3 = 0) ∧ line.pass_through A C ∧ line.is_tangent l C r) :=
begin
  intros hA hC hr,
  sorry -- proof steps would go here.
end

end tangent_line_to_circle_l86_86240


namespace proportion_correct_l86_86684

theorem proportion_correct (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
sorry

end proportion_correct_l86_86684


namespace alpha_is_30_or_60_l86_86650

theorem alpha_is_30_or_60
  (α : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2) -- α is acute angle
  (a : ℝ × ℝ := (3 / 4, Real.sin α))
  (b : ℝ × ℝ := (Real.cos α, 1 / Real.sqrt 3))
  (h2 : a.1 * b.2 = a.2 * b.1)  -- a ∥ b
  : α = Real.pi / 6 ∨ α = Real.pi / 3 := 
sorry

end alpha_is_30_or_60_l86_86650


namespace evaluate_expression_l86_86195

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l86_86195


namespace sqrt_diff_eq_l86_86858

theorem sqrt_diff_eq : sqrt (64 + 81) - sqrt (49 - 36) = sqrt 145 - sqrt 13 :=
by
  sorry

end sqrt_diff_eq_l86_86858


namespace num_students_basketball_l86_86336

-- Definitions for conditions
def num_students_cricket : ℕ := 8
def num_students_both : ℕ := 5
def num_students_either : ℕ := 10

-- statement to be proven
theorem num_students_basketball : ∃ B : ℕ, B = 7 ∧ (num_students_either = B + num_students_cricket - num_students_both) := sorry

end num_students_basketball_l86_86336


namespace intersection_points_equidistant_l86_86968

-- Define the conditions succinctly in Lean 
variables {A B C A1 B1 C1 P Q R : Point}
variables {triangle_ABC : Triangle A B C}
variables {altitude_A : Line A A1} {altitude_B : Line B B1} {altitude_C : Line C C1}
variables {semicircle_BC : Semicircle (BC)} {semicircle_CA : Semicircle (CA)} {semicircle_AB : Semicircle (AB)}

-- Statement of the theorem in Lean
theorem intersection_points_equidistant 
  (acute_triangle : acute_angled triangle_ABC)
  (A_perpendicular : perp altitude_A (Line_of_segment B C))
  (B_perpendicular : perp altitude_B (Line_of_segment C A))
  (C_perpendicular : perp altitude_C (Line_of_segment A B))
  (semicircle_BC_constructed : semicircle_BC = Semicircle_of_diameter B C)
  (semicircle_CA_constructed : semicircle_CA = Semicircle_of_diameter C A)
  (semicircle_AB_constructed : semicircle_AB = Semicircle_of_diameter A B)
  (intersections_A : intersects (extension altitude_A) semicircle_CA P)
  (intersections_B : intersects (extension altitude_B) semicircle_AB Q)
  (intersections_C : intersects (extension altitude_C) semicircle_BC R) :
  dist A Q = dist A R := 
sorry

end intersection_points_equidistant_l86_86968


namespace coloring_problem_l86_86546

theorem coloring_problem 
  (n : ℕ) (hn : 0 < n) 
  (N : ℕ) (hN : N = n^2 + 1) 
  (colors : fin N → Type)
  (square : fin N → fin N → colors) :
  ∃ row : fin N, finset.card (finset.image (λ col, square row col) finset.univ) ≥ n + 1 ∨
  ∃ col : fin N, finset.card (finset.image (λ row, square row col) finset.univ) ≥ n + 1 :=
sorry

end coloring_problem_l86_86546


namespace arcsin_one_half_l86_86948

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l86_86948


namespace probability_X_geq_2_l86_86654

noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  sorry -- Assumes existence of CDF for normal distribution.

variables (σ : ℝ) (X : ℝ → ℝ)
  (hX : ∀ x, X x ~ Normal 1 σ)
  (h_prob : normal_cdf 1 σ 1 - normal_cdf 1 σ 0 = 0.3)

theorem probability_X_geq_2 : 
  (normal_cdf 1 σ 0 + 0.2 = normal_cdf 1 σ 2) :=
sorry  -- Skipping actual proof placeholder

end probability_X_geq_2_l86_86654


namespace find_max_area_l86_86335

noncomputable def max_area_similar_trngl_in_grid : Prop :=
∃ (ABC DEF: triangle) (grid: square),
  grid.side_length = 1 ∧
  grid.dimension = 5 ∧
  ABC.side_lengths = (1, sqrt 5, sqrt 10) ∧
  (∃ similarity_factor, DEF.side_lengths = (similarity_factor * 1, similarity_factor * sqrt 5, similarity_factor * sqrt 10) ∧
  similarity_factor = sqrt 5 ∧
  (area DEF = 2.5))

-- Proof is omitted as the problem only requires the Lean statement.
theorem find_max_area : max_area_similar_trngl_in_grid :=
sorry

end find_max_area_l86_86335


namespace find_cost_prices_l86_86896

def cost_price_type_A (C_A : ℝ) : Prop :=
  let S_A := 1.40 * C_A in
  let C'_A := 0.70 * C_A in
  let S'_A := S_A - 14.50 in
  S'_A = 1.05 * C_A

def cost_price_type_B (C_B : ℝ) : Prop :=
  let S_B := 1.45 * C_B in
  let C'_B := 0.65 * C_B in
  let S'_B := S_B - 18.75 in
  S'_B = 1.0075 * C_B

theorem find_cost_prices (C_A C_B : ℝ) :
  cost_price_type_A C_A ∧ cost_price_type_B C_B → 
  C_A = 41.43 ∧ C_B = 42.37 :=
by
  -- Proof is omitted
  sorry

end find_cost_prices_l86_86896


namespace range_of_a_l86_86636

variable {x : ℝ} (a : ℝ)

def f (x : ℝ) := x * Real.log x - a * x

def g (x : ℝ) := x^3 - x + 6

theorem range_of_a (h : ∀ x > 0, 2 * f x ≤ (3 * x^2 - 1) + 2) : a ∈ Set.Ici (-2) :=
by sorry

end range_of_a_l86_86636


namespace eval_ceil_floor_sum_l86_86207

theorem eval_ceil_floor_sum :
  (Int.ceil (7 / 3) + Int.floor (- (7 / 3))) = 0 :=
by
  sorry

end eval_ceil_floor_sum_l86_86207


namespace probability_odd_multiple_of_5_l86_86840

theorem probability_odd_multiple_of_5 :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ 1 ≤ c ∧ c ≤ 100 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c) % 2 = 1 ∧ (a * b * c) % 5 = 0) → 
  p = 3 / 125 := 
sorry

end probability_odd_multiple_of_5_l86_86840


namespace smallest_integer_consecutive_set_l86_86618

theorem smallest_integer_consecutive_set 
(n : ℤ) (h : 7 * n + 21 > 4 * n) : n > -7 :=
by
  sorry

end smallest_integer_consecutive_set_l86_86618


namespace rect_side_ratio_square_l86_86116

theorem rect_side_ratio_square (a b d : ℝ) (h1 : b = 2 * a) (h2 : d = a * Real.sqrt 5) : (b / a) ^ 2 = 4 := 
by sorry

end rect_side_ratio_square_l86_86116


namespace smallest_c_correct_l86_86959

noncomputable def smallest_c := sqrt 6003

theorem smallest_c_correct (x : ℝ) (h : x > smallest_c) :
  ∃ (a b c d : ℝ), a = log 2005 (log 2004 (log 2003 (log 2002 (x^2 - 4000)))) ∧
                   b = log 2004 (log 2003 (log 2002 (x^2 - 4000))) ∧
                   c = log 2003 (log 2002 (x^2 - 4000)) ∧
                   d = log 2002 (x^2 - 4000) ∧ 
                   a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
begin
  -- Proof to be provided here
  sorry
end

end smallest_c_correct_l86_86959


namespace inequality_solution_set_range_of_a_l86_86271

section
variable {x a : ℝ}

def f (x a : ℝ) := |2 * x - 5 * a| + |2 * x + 1|
def g (x : ℝ) := |x - 1| + 3

theorem inequality_solution_set :
  {x : ℝ | |g x| < 8} = {x : ℝ | -4 < x ∧ x < 6} :=
sorry

theorem range_of_a (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, f x₁ a = g x₂) :
  a ≥ 0.4 ∨ a ≤ -0.8 :=
sorry
end

end inequality_solution_set_range_of_a_l86_86271


namespace roots_and_ratio_l86_86834

-- Definition of the hypothesis and the final theorem to prove
theorem roots_and_ratio (p q r s : ℝ) (hp : p ≠ 0) :
  let root1 := -1
  let root2 := 3
  let root3 := 4
  let poly := λ x, p * x^3 + q * x^2 + r * x + s
  -- Condition 1: The polynomial has given roots
  poly root1 = 0 ∧ poly root2 = 0 ∧ poly root3 = 0 →
  -- Vieta's formulas based on the given roots
  let sum_roots := root1 + root2 + root3
  let two_products_sum := root1 * root2 + root1 * root3 + root2 * root3
  let product_roots := root1 * root2 * root3
  -- Vieta's relations
  sum_roots = -q / p ∧
  two_products_sum = r / p ∧
  product_roots = -s / p →
  -- The statement we want to prove
  r / s = -5 / 12 :=
by
  intros h1 h2
  sorry

end roots_and_ratio_l86_86834


namespace emily_gave_away_l86_86602

variable (x : ℕ)

def emily_initial_books : ℕ := 7

def emily_books_after_giving_away (x : ℕ) : ℕ := 7 - x

def emily_books_after_buying_more (x : ℕ) : ℕ :=
  7 - x + 14

def emily_final_books : ℕ := 19

theorem emily_gave_away : (emily_books_after_buying_more x = emily_final_books) → x = 2 := by
  sorry

end emily_gave_away_l86_86602


namespace triangle_DMC_area_l86_86731

noncomputable def Heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_DMC_area :
  let AB := 6
  let BC := 5
  let AC := 7
  let s_area := Heron_area AB BC AC
  in s_area = 6 * Real.sqrt 6 
  → ∃ D M C : ℝ, D = AC / 2 
  ∧ M = (D + √(s_area/4)) / 2 
  ∧ C = AC 
  ∧ ∃ area_DMC : ℝ, area_DMC = s_area / 4 
  ∧ area_DMC = 3 * Real.sqrt 6 / 2 := 
by
  intros
  sorry 

end triangle_DMC_area_l86_86731


namespace oil_bill_january_l86_86878

theorem oil_bill_january (F J : ℝ)
  (h1 : F / J = 5 / 4)
  (h2 : (F + 30) / J = 3 / 2) :
  J = 120 :=
sorry

end oil_bill_january_l86_86878


namespace chosen_number_l86_86909

theorem chosen_number (x : ℝ) (h1 : x / 9 - 100 = 10) : x = 990 :=
  sorry

end chosen_number_l86_86909


namespace hexagon_interior_angles_sum_l86_86468

theorem hexagon_interior_angles_sum :
  let n := 6 in
  (n - 2) * 180 = 720 :=
by
  sorry

end hexagon_interior_angles_sum_l86_86468


namespace segment_length_BD_l86_86727

theorem segment_length_BD
  (AB BD : ℝ)
  (angle_ABD angle_DBC angle_BCD : ℝ)
  (AD DE BE EC : ℝ)
  (BC BD : ℝ)
  (h1 : AB = BD)
  (h2 : angle_ABD = angle_DBC)
  (h3 : angle_BCD = real.pi / 2)
  (h4 : BE = 7)
  (h5 : EC = 5)
  (h6 : AD = DE)
  (h7 : BD = BC):

  BE + EC = 12 :=
begin
  -- proof steps here
  sorry
end

end segment_length_BD_l86_86727


namespace restore_original_salary_l86_86513

theorem restore_original_salary (orig_salary : ℝ) (reducing_percent : ℝ) (increasing_percent : ℝ) :
  reducing_percent = 20 → increasing_percent = 25 →
  (orig_salary * (1 - reducing_percent / 100)) * (1 + increasing_percent / 100 / (1 - reducing_percent / 100)) = orig_salary
:= by
  intros
  sorry

end restore_original_salary_l86_86513


namespace surface_integral_a_eq_pi_l86_86137

noncomputable def surface_integral_part_a : ℝ :=
  let dS (x y z : ℝ) := sqrt (1 + (x^2 + y^2 + z^2)) in
  ∫ x in Icc (-1:ℝ) 1, ∫ y in Icc (-sqrt (1 - x^2):ℝ) sqrt (1 - x^2), ∫ z in Icc (0:ℝ) sqrt (1 - x^2 - y^2), abs x * dS x y z

theorem surface_integral_a_eq_pi : surface_integral_part_a = real.pi :=
sorry

end surface_integral_a_eq_pi_l86_86137


namespace pyramid_surface_area_l86_86589

noncomputable def surface_area_of_pyramid (a b c : ℝ) (h : ℝ) : ℝ :=
  4 * (1/2 * c * h)

theorem pyramid_surface_area
  (T U V W : Type)
  (E : T × T × T × T → ℝ)
  (h_side_lengths : (E (T,U), E (U,V), E (V,W)) ∈ {(28, 28, 35), (28, 35, 35)})
  (h_no_equilateral : ¬(E (T,U) = E (U,V) ∧ E (U,V) = E (V,W)))
  : surface_area_of_pyramid 28 28 35 (Real.sqrt 1029) = 70 * Real.sqrt 1029 :=
begin
  sorry
end

end pyramid_surface_area_l86_86589


namespace arithmetic_sequence_fifth_term_l86_86719

theorem arithmetic_sequence_fifth_term :
  ∀ (a₁ d : ℤ) (n : ℕ), a₁ = 3 → d = 4 → n = 5 → a₁ + (n - 1) * d = 19 :=
by
  intros a₁ d n ha₁ hd hn
  rw [ha₁, hd, hn]
  norm_num
  sorry

end arithmetic_sequence_fifth_term_l86_86719


namespace tourists_left_l86_86533

noncomputable def tourists_remaining {initial remaining poisoned recovered : ℕ} 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : ℕ :=
  remaining - poisoned + recovered

theorem tourists_left 
  (initial remaining poisoned recovered : ℕ) 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : tourists_remaining h1 h2 h3 h4 h5 h6 = 16 :=
  by
  sorry

end tourists_left_l86_86533


namespace sum_of_x_coordinates_of_intersections_l86_86479

theorem sum_of_x_coordinates_of_intersections 
  : ∑ k in finset.range 18, (100 - 10 * k + (if k % 2 = 0 then 1 else 0)) = 867 := 
sorry

end sum_of_x_coordinates_of_intersections_l86_86479


namespace greatest_divisor_of_372_lt_50_and_factor_of_72_l86_86500

theorem greatest_divisor_of_372_lt_50_and_factor_of_72 : 
  ∃ x, (x ∣ 372 ∧ x < 50 ∧ x ∣ 72) ∧ ∀ y, (y ∣ 372 ∧ y < 50 ∧ y ∣ 72) → y ≤ x ∧ x = 12 :=
begin
  sorry,
end

end greatest_divisor_of_372_lt_50_and_factor_of_72_l86_86500


namespace slower_train_pass_time_l86_86051

noncomputable def time_to_pass (length_train : ℕ) (speed_faster_kmh : ℕ) (speed_slower_kmh : ℕ) : ℕ :=
  let speed_faster_mps := speed_faster_kmh * 5 / 18
  let speed_slower_mps := speed_slower_kmh * 5 / 18
  let relative_speed := speed_faster_mps + speed_slower_mps
  let distance := length_train
  distance * 18 / (relative_speed * 5)

theorem slower_train_pass_time :
  time_to_pass 500 45 15 = 300 :=
by
  sorry

end slower_train_pass_time_l86_86051


namespace bacon_vs_tomatoes_l86_86571

theorem bacon_vs_tomatoes :
  let (n_b : ℕ) := 337
  let (n_t : ℕ) := 23
  n_b - n_t = 314 := by
  let n_b := 337
  let n_t := 23
  have h1 : n_b = 337 := rfl
  have h2 : n_t = 23 := rfl
  sorry

end bacon_vs_tomatoes_l86_86571


namespace minimal_face_sum_of_larger_cube_l86_86537

-- Definitions
def num_small_cubes : ℕ := 27
def num_faces_per_cube : ℕ := 6

-- The goal: Prove the minimal sum of the integers shown on the faces of the larger cube
theorem minimal_face_sum_of_larger_cube (min_sum : ℤ) 
    (H : min_sum = 90) :
    min_sum = 90 :=
by {
  sorry
}

end minimal_face_sum_of_larger_cube_l86_86537


namespace willam_tax_payment_correct_l86_86976

noncomputable def willamFarmTax : ℝ :=
  let totalTax := 3840
  let willamPercentage := 0.2777777777777778
  totalTax * willamPercentage

-- Lean theorem statement for the problem
theorem willam_tax_payment_correct : 
  willamFarmTax = 1066.67 :=
by
  sorry

end willam_tax_payment_correct_l86_86976


namespace integral_inequality_l86_86369

noncomputable def integral_expr (a b c : ℝ) : ℝ :=
  ∫ x in 0..1, (1 - a * x)^3 + (1 - b * x)^3 + (1 - c * x)^3 - 3 * x

theorem integral_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 1) :
  integral_expr a b c ≥ ab + bc + ca - (3/2) * (a + b + c) - (3/4) * abc :=
sorry

end integral_inequality_l86_86369


namespace total_amount_paid_l86_86487

-- Definitions based on the conditions in part (a)
def cost_of_manicure : ℝ := 30
def tip_percentage : ℝ := 0.30

-- Proof statement based on conditions and answer in part (c)
theorem total_amount_paid : cost_of_manicure + (cost_of_manicure * tip_percentage) = 39 := by
  sorry

end total_amount_paid_l86_86487


namespace eval_expr_eq_zero_l86_86167

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l86_86167


namespace triangle_area_l86_86575

theorem triangle_area : 
  ∀ (A B C : ℝ × ℝ), 
  A = (0, 0) → 
  B = (4, 0) → 
  C = (2, 6) → 
  (1 / 2 : ℝ) * (4 : ℝ) * (6 : ℝ) = (12.0 : ℝ) := 
by 
  intros A B C hA hB hC
  simp [hA, hB, hC]
  norm_num

end triangle_area_l86_86575


namespace product_two_digit_numbers_l86_86454

theorem product_two_digit_numbers (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 777) : (a = 21 ∧ b = 37) ∨ (a = 37 ∧ b = 21) := 
  sorry

end product_two_digit_numbers_l86_86454


namespace ceiling_plus_floor_eq_zero_l86_86189

theorem ceiling_plus_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (-7 / 3)) = 0 := by
  sorry

end ceiling_plus_floor_eq_zero_l86_86189


namespace find_total_money_l86_86151

theorem find_total_money
  (d x T : ℝ)
  (h1 : d = 5 / 17)
  (h2 : x = 35)
  (h3 : d * T = x) :
  T = 119 :=
by sorry

end find_total_money_l86_86151


namespace part3_inequality_l86_86458

-- Part 3
theorem part3_inequality :
  (∑ i in range n, (prod (λ j, 1 - (1 / (j ^ 2))) (range i)) * (1 / ((i + 1) ^ 2))) < (1 / 2) :=
sorry

end part3_inequality_l86_86458


namespace area_RectanglePQR_is_12_l86_86119

open Real

def point := ℝ × ℝ
def P : point := (0, 4)
def Q : point := (3, 4)
def R : point := (3, 0)

axiom PR_hypotenuse : dist P R = 5

theorem area_RectanglePQR_is_12 (hpq : dist P Q = 3) (hqr : dist Q R = 4) : 
  1 / 2 * dist P Q * dist Q R = 12 :=
by
  -- Assuming distances between points
  have hq_eq_p : dist P Q = 3 := by simp [dist, P, Q]; linarith
  have hr_eq_q : dist Q R = 4 := by simp [dist, Q, R]; linarith
  -- Calculating the area of the rectangle
  have h_area : 1 / 2 * dist P Q * dist Q R = 1 / 2 * 3 * 4 := by
      rw [hq_eq_p, hr_eq_q]
      linarith
  exact h_area

end area_RectanglePQR_is_12_l86_86119


namespace length_BE_of_congruent_rectangles_l86_86723

theorem length_BE_of_congruent_rectangles :
  ∀ (A B C D J K H G E B' C' F : Point) (BE : ℝ),
  square A B C D →
  side_length A B 1 →
  rectangle J K H G →
  rectangle E B' C' F →
  congruent_rectangles J K H G E B' C' F →
  length BE = 2 - √3 :=
begin
  intros,
  
  sorry,
end

end length_BE_of_congruent_rectangles_l86_86723


namespace m_n_sum_l86_86379

noncomputable def m : ℝ :=
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(-2/3) + (1.5)^(-2)

noncomputable def n : ℝ :=
  logBase 3 (427 / 3) + Real.log10 25 + Real.log10 4 + 7^(Real.logBase 7 2)

theorem m_n_sum : m + n = 17 / 4 := by
  sorry

end m_n_sum_l86_86379


namespace f_even_and_g_odd_l86_86992

variable (a : ℝ)
variable (h_pos : a > 0)
variable (h_ne_one : a ≠ 1)

def f (x : ℝ) := a^x + a^(-x) + 1
def g (x : ℝ) := a^x - a^(-x)

theorem f_even_and_g_odd : 
  (∀ x : ℝ, f a (-x) = f a x) ∧ (∀ x : ℝ, g a (-x) = -g a x) :=
by
  sorry

end f_even_and_g_odd_l86_86992


namespace find_sum_lent_l86_86869

theorem find_sum_lent (P : ℝ) : 
  (∃ R T : ℝ, R = 4 ∧ T = 8 ∧ I = P - 170 ∧ I = (P * 8) / 25) → P = 250 :=
by
  sorry

end find_sum_lent_l86_86869


namespace range_of_a_decreasing_l86_86660

def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a_decreasing (a : ℝ) : (∀ x y : ℝ, x ∈ Set.Iic 4 → y ∈ Set.Iic 4 → x ≤ y → f x a ≥ f y a) ↔ a ≤ -3 :=
by
  sorry

end range_of_a_decreasing_l86_86660


namespace choir_average_age_l86_86708

-- Conditions
def women_count : ℕ := 12
def men_count : ℕ := 10
def avg_age_women : ℝ := 25.0
def avg_age_men : ℝ := 40.0

-- Expected Answer
def expected_avg_age : ℝ := 31.82

-- Proof Statement
theorem choir_average_age :
  ((women_count * avg_age_women) + (men_count * avg_age_men)) / (women_count + men_count) = expected_avg_age :=
by
  sorry

end choir_average_age_l86_86708


namespace reading_plans_count_l86_86813

theorem reading_plans_count :
  let novels := {"Dream of the Red Chamber", "Romance of the Three Kingdoms", "Water Margin", "Journey to the West"}
  let students := {1, 2, 3, 4, 5}
  (∃ f : students → novels, (∀ n, ∃ s, f s = n) ∧ (f 1 ≠ f 2)) →
  if h : ∃ f : students → novels, (∀ n, ∃ s, f s = n) ∧ (f 1 ≠ f 2) then 240 else 0 = 240 :=
begin
  sorry
end

end reading_plans_count_l86_86813


namespace count_valid_N_l86_86297

theorem count_valid_N : 
  ∃ (N : ℕ), (N < 2000) ∧ (∃ (x : ℝ), x^⌊x⌋₊ = N) :=
begin
  sorry
end

end count_valid_N_l86_86297


namespace region_area_l86_86852

theorem region_area {x y : ℝ} (h : x^2 + y^2 - 4*x + 2*y = -1) : 
  ∃ (r : ℝ), r = 4*pi := 
sorry

end region_area_l86_86852


namespace total_balance_payment_l86_86399

theorem total_balance_payment (initial_balance1 initial_balance2 initial_balance3 : ℝ) 
    (rate1 month2_rate1 month3_rate1 : ℝ) 
    (month2_rate2 month3_rate2 : ℝ) 
    (constant_rate3 : ℝ) 
    (time_period : ℕ)
    (h_balance1 : initial_balance1 = 150) 
    (h_rate1 : rate1 = 0.02) 
    (h_month2_rate1 : month2_rate1 = 0.01) 
    (h_month3_rate1 : month3_rate1 = 0.005) 
    (h_balance2 : initial_balance2 = 220) 
    (h_month2_rate2 : month2_rate2 = 0.01) 
    (h_month3_rate2 : month3_rate2 = 0.005) 
    (h_balance3 : initial_balance3 = 75) 
    (h_constant_rate3 : constant_rate3 = 0.015) 
    (h_time_period : time_period = 3) :
    (initial_balance1 * (1 + rate1) * (1 + month2_rate1) * (1 + month3_rate1) +
    initial_balance2 * (1 + month2_rate2) * (1 + month3_rate2) +
    initial_balance3 * (1 + constant_rate3 * time_period)).round 2 = 456.93 := by
  sorry

end total_balance_payment_l86_86399


namespace shape_formed_is_rectangle_l86_86641

/-- A rectangle ABCD with vertices at (0,0), (0,4), (6,4), (6,0) --/
structure Rectangle :=
(A B C D : (ℝ × ℝ))
(hA : A = (0,0))
(hB : B = (0,4))
(hC : C = (6,4))
(hD : D = (6,0))

/-- Lines forming from A and B at specified angles --/
def line_from_angle (angle : ℝ) (start : ℝ × ℝ) : ℝ → ℝ × ℝ :=
λ x, let y := tan angle * x in (x + start.1, y + start.2)

/-- Intersection points determined by these lines --/
def intersection_points : (ℝ × ℝ) :=
let l1 := line_from_angle (real.pi / 4) (0,0),
    l2 := line_from_angle ((5 * real.pi) / 12) (0,0),
    l3 := line_from_angle (- real.pi / 4) (0,4),
    l4 := line_from_angle (- (5 * real.pi) / 12) (0,4) in 
((2, 2), (x, (real.tan (5 * real.pi / 12) * x))

/-- The shape formed by the intersection points should be examined --/
theorem shape_formed_is_rectangle : Rectangle := 
sorry

end shape_formed_is_rectangle_l86_86641


namespace parallelogram_area_is_66_67_l86_86042

noncomputable def parallelogram_area (a : ℝ) : ℝ :=
  40 * a

theorem parallelogram_area_is_66_67 {a : ℝ}
  (h1 : sqrt (40^2 + (55 - a)^2) = 40 * a)
  (h2 : 1 = 1) : parallelogram_area a = 200 / 3 :=
sorry

end parallelogram_area_is_66_67_l86_86042


namespace derivative_at_pi_over_3_l86_86012

-- Define the function f
def f (x : ℝ) : ℝ := 1 + Real.sin x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := Real.cos x

-- State the theorem to find f' at π/3
theorem derivative_at_pi_over_3 : f' (Real.pi / 3) = 1 / 2 :=
by
  -- Proof skipped
  sorry

end derivative_at_pi_over_3_l86_86012


namespace stewart_farm_horse_food_l86_86029

theorem stewart_farm_horse_food (h1 : 56 % 7 = 0) (h2 : 230 > 0) (sheep : ℕ) (horse : ℕ) 
  (ratio_cond : sheep / 7 = horse) (sheep_cond : sheep = 56) : 
  horse * 230 = 1840 :=
by
  have h3 : sheep / 7 = 8, from
    calc
      sheep / 7 = 56 / 7 : by rw [sheep_cond]
      ... = 8 : by norm_num,
  have h4 : horse = 8, from
    calc
      horse = sheep / 7 : by rw [← ratio_cond]
      ... = 8 : by rw [h3],
  calc
    horse * 230 = 8 * 230 : by rw [h4]
    ... = 1840 : by norm_num

end stewart_farm_horse_food_l86_86029


namespace limit_of_sequence_l86_86373

noncomputable def x_n (n : ℕ) (α : ℝ) : ℝ :=
if n > 1 / α then ∑ k in finset.range (n + 1), real.sinh (k / n^2) else 0

theorem limit_of_sequence (α : ℝ) (hα : α > 0) : 
  filter.tendsto (λ n : ℕ, x_n n α) filter.at_top (nhds (1 / 2)) :=
sorry

end limit_of_sequence_l86_86373


namespace weight_of_replaced_person_l86_86710
-- We need a broad import to ensure all necessary libraries are available.

section weight_problem

-- Define the total weight of the original group
variable (W : ℝ)

-- Define the original average weight
def original_average_weight (W : ℝ) : ℝ := W / 10

-- Define the weight of the new man
def weight_new_man : ℝ := 75

-- Define the increased average weight condition
def increased_average_weight (W : ℝ) : ℝ :=
(original_average_weight W) + 3

-- Equation relating the total weight, weight of the person replaced, and the increased average weight
def weight_replacement_equation (W weight_replaced : ℝ) : Prop :=
(W - weight_replaced + weight_new_man) / 10 = increased_average_weight W

-- Theorem stating the weight of the replaced person
theorem weight_of_replaced_person (W : ℝ) (weight_replaced : ℝ) :
  (weight_replacement_equation W weight_replaced) → weight_replaced = 45 :=
begin
  sorry
end

end weight_problem

end weight_of_replaced_person_l86_86710


namespace num_odd_integers_between_3000_and_6000_l86_86679

-- Number of odd integers between 3000 and 6000 with four different digits
theorem num_odd_integers_between_3000_and_6000 : 
  (∃ (nums : Finset ℕ), 
    (∀ n ∈ nums, 3000 ≤ n ∧ n < 6000 ∧ (n % 2 = 1)) ∧ 
    (∀ n ∈ nums, (Nat.digits 10 n).Nodup) ∧ nums.card = 840) := 
begin
  -- at this point the proof steps would go here
  sorry
end

end num_odd_integers_between_3000_and_6000_l86_86679


namespace min_value_of_expression_l86_86262

theorem min_value_of_expression (x y : ℝ) (hposx : x > 0) (hposy : y > 0) (heq : 2 / x + 1 / y = 1) : 
  x + 2 * y ≥ 8 :=
sorry

end min_value_of_expression_l86_86262


namespace no_odd_total_given_ratio_l86_86337

theorem no_odd_total_given_ratio (T : ℕ) (hT1 : 50 < T) (hT2 : T < 150) (hT3 : T % 2 = 1) : 
  ∀ (B : ℕ), T ≠ 8 * B + B / 4 :=
sorry

end no_odd_total_given_ratio_l86_86337


namespace total_volume_of_ice_cream_l86_86019

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

noncomputable def volume_of_hemisphere (r : ℝ) : ℝ :=
  (2/3) * Real.pi * r^3

theorem total_volume_of_ice_cream :
  let r1 := 4 -- radius of large cone and hemisphere
  let h1 := 12 -- height of large cone
  let r2 := 2 -- radius of smaller cone
  let h2 := 3 -- height of smaller cone
  volume_of_cone r1 h1 + volume_of_hemisphere r1 + volume_of_cone r2 h2 = (332/3) * Real.pi :=
by {
  let V_cone := volume_of_cone r1 h1,
  let V_hemisphere := volume_of_hemisphere r1,
  let V_small_cone := volume_of_cone r2 h2,
  let V_total := V_cone + V_hemisphere + V_small_cone in 
  have h_cone : V_cone = (1/3) * Real.pi * r1^2 * h1 := by sorry,
  have h_hemisphere : V_hemisphere = (2/3) * Real.pi * r1^3 := by sorry,
  have h_small_cone : V_small_cone = (1/3) * Real.pi * r2^2 * h2 := by sorry,
  have rewrite_total : V_total = (332/3) * Real.pi := by sorry,
  exact rewrite_total
}

end total_volume_of_ice_cream_l86_86019


namespace tan60_reciprocal_l86_86030

theorem tan60_reciprocal :
  let tan60 := 60 * Real.pi / 180 in
  Real.tan tan60 = Real.sqrt 3 →
  -1 / Real.tan tan60 = -Real.sqrt 3 / 3 :=
by
  intro h
  sorry

end tan60_reciprocal_l86_86030


namespace coefficient_x2_expansion_l86_86149

theorem coefficient_x2_expansion (x : ℝ) : 
  let binomial_expansion := (x - 2) ^ 6 in
  ∃ c : ℝ, c • x^2 ∈ (binomial_expansion) ∧ c = 240 :=
by 
  sorry

end coefficient_x2_expansion_l86_86149


namespace compare_balances_l86_86139

noncomputable def Cedric_balance (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

noncomputable def Daniel_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem compare_balances :
  let P := 12000
  let r_C := 0.06
  let r_D := 0.08
  let n := 20
  let Cedric_A := Cedric_balance P r_C n
  let Daniel_A := Daniel_balance P r_D n
  abs (Cedric_A - Daniel_A) ≈ 7286 :=
by
  let P := 12000
  let r_C := 0.06
  let r_D := 0.08
  let n := 20
  let Cedric_A := Cedric_balance P r_C n
  let Daniel_A := Daniel_balance P r_D n
  have h := abs (Cedric_A - Daniel_A) ≈ 7286
  sorry

end compare_balances_l86_86139


namespace evaluate_modulus_complex_l86_86972

def modulus_complex : ℝ :=
  let a := 3
  let b := -2
  real.sqrt (a^2 + b^2)

theorem evaluate_modulus_complex :
  modulus_complex = real.sqrt 13 := by
  sorry

end evaluate_modulus_complex_l86_86972


namespace arcsin_one_half_l86_86946

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  -- Conditions
  have h1 : -Real.pi / 2 ≤ Real.pi / 6 ∧ Real.pi / 6 ≤ Real.pi / 2 := by
    -- Proof the range of pi/6 is within [-pi/2, pi/2]
    sorry
  have h2 : ∀ x, Real.sin x = 1 / 2 → x = Real.pi / 6 := by
    -- Proof sin(pi/6) = 1 / 2
    sorry
  show Real.arcsin (1 / 2) = Real.pi / 6
  -- Proof arcsin(1/2) = pi/6 based on the above conditions
  sorry

end arcsin_one_half_l86_86946


namespace faster_ship_speed_l86_86492

theorem faster_ship_speed :
  ∀ (x y : ℕ),
    (200 + 100 = 300) → -- Total distance covered for both directions
    (x + y) * 10 = 300 → -- Opposite direction equation
    (x - y) * 25 = 300 → -- Same direction equation
    x = 21 := 
by
  intros x y _ eq1 eq2
  sorry

end faster_ship_speed_l86_86492


namespace distance_between_points_l86_86982

theorem distance_between_points :
  let p1 := (-4, 17)
  let p2 := (12, -1)
  let distance := Real.sqrt ((12 - (-4))^2 + (-1 - 17)^2)
  distance = 2 * Real.sqrt 145 := sorry

end distance_between_points_l86_86982


namespace museum_paintings_discarded_l86_86908

def initial_paintings : ℕ := 2500
def percentage_to_discard : ℝ := 0.35
def paintings_discarded : ℝ := initial_paintings * percentage_to_discard

theorem museum_paintings_discarded : paintings_discarded = 875 :=
by
  -- Lean automatically simplifies this using basic arithmetic rules
  sorry

end museum_paintings_discarded_l86_86908


namespace solve_linear_eq_l86_86830

theorem solve_linear_eq (x : ℝ) : 3 * x - 6 = 0 ↔ x = 2 :=
sorry

end solve_linear_eq_l86_86830


namespace ceiling_plus_floor_eq_zero_l86_86188

theorem ceiling_plus_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (-7 / 3)) = 0 := by
  sorry

end ceiling_plus_floor_eq_zero_l86_86188


namespace coloring_method_exists_iff_odd_n_l86_86368

theorem coloring_method_exists_iff_odd_n (n : ℕ) (h1 : 1 < n) :
  (∃ (c : Π (s : finset (fin n)), s.card = 2 → fin n), 
    (∀ t : finset (fin n), t.card = 3 → ∃ (a b c : fin n), 
      {a, b, c} = t ∧ function.injective_on (λ x, c ⟨{a, b}.insert x, sorry⟩ sorry) t)) ↔ 
    odd n :=
sorry

end coloring_method_exists_iff_odd_n_l86_86368


namespace greatest_integer_n_l86_86984

theorem greatest_integer_n : ∃ (x : Fin 9 → ℕ), 
  (¬ (∀ i, x i = 0)) ∧ 
  (∀ (ε : Fin 9 → ℤ), (¬ ∀ i, ε i = 0) → ¬ (9^3 ∣ ∑ i, ε i * (x i))) :=
begin
  sorry
end

end greatest_integer_n_l86_86984


namespace sum_of_indices_l86_86655

theorem sum_of_indices (d : ℕ) (n : ℕ) (h_nonzero : d ≠ 0)
  (h_geom : ∀ i, i ∈ {1, 5, 17, 2 * 3^(i-1) - 1} → ∃ a₁, ∀ j, a₁ + (i - 1) * d = a₁ * 3^(j - 1)) :
  (∑ i in (finset.range n).map (λ i, 2 * 3^i - 1), i) = 3^n - n - 1 :=
by
  sorry

end sum_of_indices_l86_86655


namespace both_reunions_attendees_l86_86083

def total_attendees : ℕ := 150
def oates_attendees : ℕ := 70
def hall_attendees : ℕ := 52

theorem both_reunions_attendees :
  ∃ x : ℕ, (x = oates_attendees + hall_attendees - total_attendees) ∧ x = 28 :=
by
  use oates_attendees + hall_attendees - total_attendees
  split
  · reflexivity
  · sorry

end both_reunions_attendees_l86_86083


namespace find_roots_of_polynomial_l86_86609

def f (x : ℝ) := x^3 - 2*x^2 - 5*x + 6

theorem find_roots_of_polynomial :
  (f 1 = 0) ∧ (f (-2) = 0) ∧ (f 3 = 0) :=
by
  -- Proof will be written here
  sorry

end find_roots_of_polynomial_l86_86609


namespace cost_price_of_clothing_l86_86540

-- Definitions based on given conditions
def marked_price : ℝ := 132
def discount_rate : ℝ := 0.1
def profit_rate : ℝ := 0.1

-- Statement to prove
theorem cost_price_of_clothing : 
  let selling_price := marked_price * (1 - discount_rate) in
  let cost_price := selling_price / (1 + profit_rate) in
  cost_price = 108 := by
  let selling_price := marked_price * (1 - discount_rate)
  let cost_price := selling_price / (1 + profit_rate)
  sorry

end cost_price_of_clothing_l86_86540


namespace cake_cut_no_square_l86_86095

theorem cake_cut_no_square (n : ℕ) : 
  (initial_area : ℝ) (remaining_area : ℝ) (length width : ℝ → ℕ → ℝ) 
  (h_initial_square : length initial_area 0 = width initial_area 0) 
  (h_pieces_equal_area : ∀ i, length remaining_area i * width remaining_area i 
    = length remaining_area (i + 1) * width remaining_area (i + 1)) 
  (h_perpendicular_cuts : ∀ i, length remaining_area (i + 1) ≠ width remaining_area (i + 1)) :
  ∀ i, width remaining_area (i + 1) ≠ length remaining_area (i + 1) := 
by 
  intros 
  sorry

end cake_cut_no_square_l86_86095


namespace gardener_total_expenses_l86_86359

theorem gardener_total_expenses
  (tulips carnations roses : ℕ)
  (cost_per_flower : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : roses = 320)
  (h4 : cost_per_flower = 2) :
  (tulips + carnations + roses) * cost_per_flower = 1890 := 
by
  sorry

end gardener_total_expenses_l86_86359


namespace sodium_thiosulfate_properties_l86_86791

def thiosulfate_structure : Type := sorry
-- Define the structure of S2O3^{2-} with S-S bond
def has_s_s_bond (ion : thiosulfate_structure) : Prop := sorry
-- Define the formation reaction
def formed_by_sulfite_reaction (ion : thiosulfate_structure) : Prop := sorry

theorem sodium_thiosulfate_properties :
  ∃ (ion : thiosulfate_structure),
    has_s_s_bond ion ∧ formed_by_sulfite_reaction ion :=
by
  sorry

end sodium_thiosulfate_properties_l86_86791


namespace ceil_floor_sum_l86_86183

theorem ceil_floor_sum : (Int.ceil (7 / 3 : ℝ) + Int.floor (- 7 / 3 : ℝ) = 0) := 
  by {
    -- referring to the proof steps
    have h1: Int.ceil (7 / 3 : ℝ) = 3,
    { sorry },
    have h2: Int.floor (- 7 / 3 : ℝ) = -3,
    { sorry },
    rw [h1, h2],
    norm_num
  }

end ceil_floor_sum_l86_86183


namespace probability_non_black_ball_l86_86695

/--
Given the odds of drawing a black ball as 5:3,
prove that the probability of drawing a non-black ball from the bag is 3/8.
-/
theorem probability_non_black_ball (n_black n_non_black : ℕ) (h : n_black = 5) (h' : n_non_black = 3) :
  (n_non_black : ℚ) / (n_black + n_non_black) = 3 / 8 :=
by
  -- proof goes here
  sorry

end probability_non_black_ball_l86_86695


namespace units_digit_sum_base8_l86_86224

theorem units_digit_sum_base8 : 
  let n1 := 53 
  let n2 := 64 
  let sum_base8 := n1 + n2 
  (sum_base8 % 8) = 7 := 
by 
  sorry

end units_digit_sum_base8_l86_86224


namespace last_letter_of_80th_permutation_of_WORDS_is_R_l86_86425

def permutation_count (s : String) : Nat :=
  (s.length.factorial : Nat)

def nth_permutation (s : String) (n : Nat) : String :=
  sorry -- we will assume the existence of this function

def last_letter (s : String) : Char :=
  s.get ⟨s.length - 1, sorry⟩

theorem last_letter_of_80th_permutation_of_WORDS_is_R :
  last_letter (nth_permutation "WORDS" 80) = 'R' :=
sorry

end last_letter_of_80th_permutation_of_WORDS_is_R_l86_86425


namespace hyperbola_equation_l86_86218

theorem hyperbola_equation :
  ∃ (b : ℝ), (∀ (x y : ℝ), ((x = 2) ∧ (y = 2)) →
    ((x^2 / 5) - (y^2 / b^2) = 1)) ∧
    (∀ x y, (y = (2 / Real.sqrt 5) * x) ∨ (y = -(2 / Real.sqrt 5) * x) → 
    (∀ (a b : ℝ), (a = 2) → (b = 2) →
      (b^2 = 4) → ((5 * y^2 / 4) - x^2 = 1))) :=
sorry

end hyperbola_equation_l86_86218


namespace percentage_less_than_mean_plus_standard_deviation_l86_86512

-- Define the characteristics of the distribution
variables {m d : ℝ} -- mean and standard deviation
variables {P : ℝ → ℝ} -- probability density function of the distribution

-- Define conditions
def is_symmetric_about_mean (P : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, P (m + x) = P (m - x)

def within_one_standard_deviation (P : ℝ → ℝ) (m d : ℝ) : Prop :=
  ∫ x in (m - d)..(m + d), P x = 0.68

-- The theorem we want to prove
theorem percentage_less_than_mean_plus_standard_deviation
  (h_symm : is_symmetric_about_mean P m)
  (h_within_sd : within_one_standard_deviation P m d) :
  ∫ x in -∞..(m + d), P x = 0.84 :=
sorry 

end percentage_less_than_mean_plus_standard_deviation_l86_86512


namespace region_area_l86_86853

theorem region_area {x y : ℝ} (h : x^2 + y^2 - 4*x + 2*y = -1) : 
  ∃ (r : ℝ), r = 4*pi := 
sorry

end region_area_l86_86853


namespace table_ordered_rows_columns_l86_86872
   
   theorem table_ordered_rows_columns (m n : ℕ) (A : matrix (fin m) (fin n) ℝ) 
     (row_sorted : ∀ i : fin m, sorted (≤) (λ j, A i j))
     (col_sorted : ∀ j : fin n, sorted (≤) (λ i, A i j)) :
     ∀ i : fin m, sorted (≤) (λ j, A i j) :=
   sorry
   
end table_ordered_rows_columns_l86_86872


namespace number_of_primes_squared_between_8000_and_12000_l86_86319

-- Define the conditions
def n_min := 90 -- the smallest integer such that n^2 > 8000
def n_max := 109 -- the largest integer such that n^2 < 12000

-- Check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Find the count of primes whose squares lie between 8000 and 12000
def primes_in_range_count : ℕ := 
  (List.filter (λ n, is_prime n) (List.range (n_max - n_min + 1)).map (λ x, x + n_min)).length

-- Define the theorem to match the problem statement
theorem number_of_primes_squared_between_8000_and_12000 : primes_in_range_count = 5 := 
  by sorry

end number_of_primes_squared_between_8000_and_12000_l86_86319


namespace num_valid_Ns_less_2000_l86_86311

theorem num_valid_Ns_less_2000 : 
  {N : ℕ | N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x^⟨floor x⟩ = N}.card = 412 := 
sorry

end num_valid_Ns_less_2000_l86_86311


namespace distance_between_parabola_vertices_l86_86596

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_parabola_vertices :
  distance (0, 3) (0, -1) = 4 := 
by {
  -- Proof omitted here
  sorry
}

end distance_between_parabola_vertices_l86_86596


namespace primes_not_divisible_by_p2021_l86_86627

theorem primes_not_divisible_by_p2021 (p : Fin 2021 → ℕ) (hp : ∀ i : Fin 2021, Nat.Prime (p i)) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 2021 ∧ ¬ (p 2020) ∣ (∑ i in Finset.range 2020, (p i)^k - k - 2020) :=
by
  sorry

end primes_not_divisible_by_p2021_l86_86627


namespace ceil_floor_sum_l86_86179

theorem ceil_floor_sum : (Int.ceil (7 / 3 : ℝ) + Int.floor (- 7 / 3 : ℝ) = 0) := 
  by {
    -- referring to the proof steps
    have h1: Int.ceil (7 / 3 : ℝ) = 3,
    { sorry },
    have h2: Int.floor (- 7 / 3 : ℝ) = -3,
    { sorry },
    rw [h1, h2],
    norm_num
  }

end ceil_floor_sum_l86_86179


namespace solution_set_f_less_than_3x_plus_6_l86_86008

variable {x : ℝ}

theorem solution_set_f_less_than_3x_plus_6 (f : ℝ → ℝ) (h_domain : ∀ x, x ∈ ℝ)
    (h_derivative : ∀ x, deriv f x > 3) (h_initial : f (-1) = 3) :
    {x | f x < 3 * x + 6} = {x | x < -1} :=
by {
  sorry
}

end solution_set_f_less_than_3x_plus_6_l86_86008


namespace monotone_increasing_ratio_range_l86_86233

variable (ω : ℝ) (x : ℝ)
variable (A : ℝ) (a b c : ℝ)
variable (a_vec b_vec : ℝ × ℝ)

def f (x : ℝ) : ℝ := (sin (ω * x), cos (ω * x)) • (cos (ω * x), sqrt 3 * cos (ω * x)) - (sqrt 3 / 2) * ((sin (ω * x))^2 + (cos (ω * x))^2)

def monotone_increasing_intervals : Set (Set ℝ) :=
  { S | ∃ k : ℤ, S = Icc (k * π - 5 * π / 12) (k * π + π / 12) }

theorem monotone_increasing (ω_pos : ω > 0) (T_pi : ∀ x, f (x + π) = f x) :
  f = λ x, sin (2 * x + π / 3) → monotone_increasing_intervals ω = { Icc (k * π - 5 * π / 12) (k * π + π / 12) | k : ℤ } :=
sorry

theorem ratio_range (acute_A : 0 < A ∧ A < π / 2) (f_A_Half : f (A / 2) = sqrt 3 / 2) :
  (sqrt 3 / 2 < a / b) ∧ (a / b < sqrt 3) :=
sorry

end monotone_increasing_ratio_range_l86_86233


namespace train_around_probability_train_present_when_alex_arrives_l86_86040

noncomputable def trainArrivalTime : Set ℝ := Set.Icc 15 45
noncomputable def trainWaitTime (t : ℝ) : Set ℝ := Set.Icc t (t + 15)
noncomputable def alexArrivalTime : Set ℝ := Set.Icc 0 60

theorem train_around (t : ℝ) (h : t ∈ trainArrivalTime) :
  ∀ (x : ℝ), x ∈ alexArrivalTime → x ∈ trainWaitTime t ↔ 15 ≤ t ∧ t ≤ 45 ∧ t ≤ x ∧ x ≤ t + 15 :=
sorry

theorem probability_train_present_when_alex_arrives :
  let total_area := 60 * 60
  let favorable_area := 1 / 2 * (15 + 15) * 15
  (favorable_area / total_area) = 1 / 16 :=
sorry

end train_around_probability_train_present_when_alex_arrives_l86_86040


namespace sqrt_calculation_l86_86578

theorem sqrt_calculation : Real.sqrt ((5: ℝ)^2 - (4: ℝ)^2 - (3: ℝ)^2) = 0 := 
by
  -- The proof is skipped
  sorry

end sqrt_calculation_l86_86578


namespace sum_of_distances_l86_86041

-- Define the conditions of the problem
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 25 + y^2 / 4 = 1)

def foci (F1 F2 A B : ℝ × ℝ) (on_ellipse : (ℝ × ℝ) → Prop) : Prop :=
  on_ellipse (A.1, A.2) ∧ on_ellipse (B.1, B.2) ∧
  let len_A_B := ( (A.1 - B.1)^2 + (A.2 - B.2)^2 )^0.5 in
  len_A_B = 8

-- Main statement to prove
theorem sum_of_distances
  (F1 F2 A B : ℝ × ℝ)
  (on_ellipse : (ℝ × ℝ) → Prop := ellipse)
  (conditions: foci F1 F2 A B on_ellipse) :
  let d1 := ( (A.1 - F1.1)^2 + (A.2 - F1.2)^2 )^0.5 in
  let d2 := ( (B.1 - F1.1)^2 + (B.2 - F1.2)^2 )^0.5 in
  d1 + d2 = 10 :=
sorry

end sum_of_distances_l86_86041


namespace problem1_problem2_l86_86090

-- Problem (1)
theorem problem1 : (1 / 8) ^ (-1 / 3) - 3 * (Real.log 3 (Real.log 3 4))^2 * (Real.log 8 27) + 2 * (Real.log (1/6) (Real.sqrt 3)) - (Real.log 6 2) = -3 :=
by
  sorry

-- Problem (2)
theorem problem2 : 27 ^ (2 / 3) - 2 ^ (Real.log 2 3) * (Real.log 2 (1 / 8)) + 2 * Real.log 10 (Real.sqrt (3 + Real.sqrt 5) + Real.sqrt (3 - Real.sqrt 5)) = 19 :=
by
  sorry

end problem1_problem2_l86_86090


namespace smallest_number_of_green_points_l86_86342

theorem smallest_number_of_green_points (total_points : ℕ) (dist : ℝ) (n b : ℕ) :
  total_points = 2020 →
  dist = 2020 →
  b + n = total_points →
  b ≤ n * (n - 1) →
  n ≥ 45 :=
by
  intros h1 h2 h3 h4
  rw h1 at *
  rw h2 at *
  sorry

end smallest_number_of_green_points_l86_86342


namespace meal_cost_is_correct_l86_86935

def samosa_quantity : ℕ := 3
def samosa_price : ℝ := 2
def pakora_quantity : ℕ := 4
def pakora_price : ℝ := 3
def mango_lassi_quantity : ℕ := 1
def mango_lassi_price : ℝ := 2
def biryani_quantity : ℕ := 2
def biryani_price : ℝ := 5.5
def naan_quantity : ℕ := 1
def naan_price : ℝ := 1.5

def tip_rate : ℝ := 0.18
def sales_tax_rate : ℝ := 0.07

noncomputable def total_meal_cost : ℝ :=
  let subtotal := (samosa_quantity * samosa_price) + (pakora_quantity * pakora_price) +
                  (mango_lassi_quantity * mango_lassi_price) + (biryani_quantity * biryani_price) +
                  (naan_quantity * naan_price)
  let sales_tax := subtotal * sales_tax_rate
  let total_before_tip := subtotal + sales_tax
  let tip := total_before_tip * tip_rate
  total_before_tip + tip

theorem meal_cost_is_correct : total_meal_cost = 41.04 := by
  sorry

end meal_cost_is_correct_l86_86935


namespace rational_terms_in_expansion_l86_86617

noncomputable def number_of_rational_terms : Nat :=
  let n := 60
  let is_rational_term (r : Nat) : Bool := (60 - r) % 2 = 0
  (Finset.range (n + 1)).filter is_rational_term |>.card

theorem rational_terms_in_expansion : number_of_rational_terms = 16 := by
  -- Proof omitted
  sorry

end rational_terms_in_expansion_l86_86617


namespace divisors_count_in_range_l86_86286

theorem divisors_count_in_range (n : ℕ) (range : ℕ → Prop) (count_divisors : ℕ) : 
  n = 36432 ∧ range = (λ x, 1 ≤ x ∧ x ≤ 9) ∧ count_divisors = (Finset.filter (λ x, n % x = 0) (Finset.filter range (Finset.range 10))).card →
  count_divisors = 7 :=
by 
  intros
  sorry

end divisors_count_in_range_l86_86286


namespace surface_area_of_CXYZ_l86_86123

def equilateral_triangle_height (a : ℝ) : ℝ := (a * real.sqrt 3) / 2

def area_of_triangle (base height : ℝ) : ℝ := (base * height) / 2

def surface_area_CXYZ : ℝ :=
  2 * area_of_triangle 5 10 + 
  area_of_triangle 5 (equilateral_triangle_height 5) + 
  area_of_triangle 5 (real.sqrt 118.75)

theorem surface_area_of_CXYZ :
  surface_area_CXYZ = 50 + 25 * real.sqrt 3 / 4 + 5 * real.sqrt 118.75 / 2 :=
by sorry

end surface_area_of_CXYZ_l86_86123


namespace initial_population_of_first_village_l86_86552

theorem initial_population_of_first_village (P : ℕ) :
  (P - 1200 * 18) = (42000 + 800 * 18) → P = 78000 :=
by
  sorry

end initial_population_of_first_village_l86_86552


namespace scientific_notation_0_00000164_l86_86900

theorem scientific_notation_0_00000164 :
  0.00000164 = 1.64 * (10 : ℝ) ^ (-6) :=
sorry

end scientific_notation_0_00000164_l86_86900


namespace calculate_polynomial_value_l86_86138

theorem calculate_polynomial_value :
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 11036846832 := 
by 
  sorry

end calculate_polynomial_value_l86_86138


namespace conversion_proof_l86_86691
-- Define the conversion rates as conditions
def cond1 : Prop := 7 * knicks = 2 * knacks
def cond2 : Prop := 3 * knacks = 4 * knocks

-- Define the goal 
def goal : Prop := 24 * knocks = 63 * knicks

-- The final statement to be proved
theorem conversion_proof (h1 : cond1) (h2 : cond2) : goal :=
sorry

end conversion_proof_l86_86691


namespace unique_representation_l86_86786

-- Define set A: Non-negative integers whose base-4 representation only includes 0 and 2
def is_in_A (a : ℕ) : Prop :=
  ∀ d ∈ (Nat.digits 4 a), d = 0 ∨ d = 2

-- Define set B: Non-negative integers whose base-4 representation only includes 0 and 1
def is_in_B (b : ℕ) : Prop :=
  ∀ d ∈ (Nat.digits 4 b), d = 0 ∨ d = 1

-- The theorem to prove the existence of such sets A and B
theorem unique_representation (n : ℕ) : 
  ∃ (a b : ℕ), is_in_A a ∧ is_in_B b ∧ n = a + b := 
by 
  sorry

end unique_representation_l86_86786


namespace conic_section_is_circle_l86_86072

theorem conic_section_is_circle : ∀ (x y : ℝ), (x - 3)^2 + (y + 4)^2 = 49 → "C" :=
by
  intro x y h
  sorry

end conic_section_is_circle_l86_86072


namespace area_of_triangle_OBA_l86_86343

theorem area_of_triangle_OBA (A : ℝ × ℝ) (B : ℝ × ℝ) (O : ℝ × ℝ)
  (hA : A = (3, (real.pi / 3))) (hB : B = (4, (real.pi / 6))) : 
  ∀ (area : ℝ), 
  (area = 6) ↔ (area = 1 / 2 * 3 * 4 * real.sin(real.pi / 2)) := 
by {
  sorry
}

end area_of_triangle_OBA_l86_86343


namespace michael_income_l86_86344

def income_tax_calculation (q : ℝ) (I : ℝ) : ℝ :=
  if I ≤ 30000 then 0.01 * q * I
  else if I ≤ 50000 then 0.01 * q * 30000 + 0.01 * (q + 3) * (I - 30000)
  else 0.01 * q * 30000 + 0.01 * (q + 3) * (20000) + 0.01 * (q + 5) * (I - 50000)

def tax_paid (q : ℝ) (I : ℝ) : ℝ :=
  0.01 * (q + 0.45) * I

theorem michael_income (q : ℝ) :
  ∃ I, I = 55000 ∧ income_tax_calculation q I = tax_paid q I :=
begin
  sorry
end

end michael_income_l86_86344


namespace sum_of_even_coefficients_l86_86223

theorem sum_of_even_coefficients (f : ℕ → ℤ) : 
    (f (1) = 1) ∧ (f (-1) = 1) →
    (2 * sum_coefficients_of_even_powers (f x) = f (1) + f (-1)) := 
by
  sorry

end sum_of_even_coefficients_l86_86223


namespace johns_spent_amount_l86_86923

def original_price : ℝ := 2000
def increase_rate : ℝ := 0.02

theorem johns_spent_amount : 
  let increased_amount := original_price * increase_rate in
  let john_total := original_price + increased_amount in
  john_total = 2040 :=
by
  sorry

end johns_spent_amount_l86_86923


namespace sector_area_l86_86121

noncomputable def area_of_sector (r : ℝ) (theta : ℝ) : ℝ :=
  1 / 2 * r * r * theta

theorem sector_area (r : ℝ) (theta : ℝ) (h_r : r = Real.pi) (h_theta : theta = 2 * Real.pi / 3) :
  area_of_sector r theta = Real.pi^3 / 6 :=
by
  sorry

end sector_area_l86_86121


namespace numberOfRationalTermsInExpansion_l86_86503

variable (x y : ℚ)

noncomputable def numberOfRationalTerms : ℕ :=
  let F : ℕ → ℚ := λ k, binom(2000, k) * (5^(k/4)) * (7^((2000-k)/2)) * (x^k) * (y^(2000-k))
  let rationalTerms := filter (λ k, 5^(k / 4) ∈ ℚ ∧ 7^((2000 - k) / 2) ∈ ℚ) (list.range 2001)
  rationalTerms.length

theorem numberOfRationalTermsInExpansion :
  numberOfRationalTerms x y = 501 :=
by
  sorry

end numberOfRationalTermsInExpansion_l86_86503


namespace simplify_expr_l86_86419

-- Define the terms
def a : ℕ := 2 ^ 10
def b : ℕ := 5 ^ 6

-- Define the expression we need to simplify
def expr := (a * b : ℝ)^(1/3)

-- Define the simplified form
def c : ℕ := 200
def d : ℕ := 2
def simplified_expr := (c : ℝ) * (d : ℝ)^(1/3)

-- The statement we need to prove
theorem simplify_expr : expr = simplified_expr ∧ (c + d = 202) := by
  sorry

end simplify_expr_l86_86419


namespace binomial_expansion_correct_statements_l86_86722

theorem binomial_expansion_correct_statements :
    let x := (1 : ℝ) in
    let f := (2 * (x) ^ (1 / 2)) - (1 / x) in
    (∑ i in finset.range 8, binomial 7 i) = 128 ∧
    ((f)^7 |>.expand |>.sum = 1) ∧
    ¬(∃ T, x ^ ((7 - 3 * T) / 2) = 1) ∧
    (binomial 7 5 * (2 ^ 2) * (-1)^5) = -84 :=
by
  sorry

end binomial_expansion_correct_statements_l86_86722


namespace number_of_valid_Ns_l86_86306

noncomputable def count_valid_N : ℕ :=
  (finset.range 2000).filter (λ N, ∃ x : ℝ, x^floor x = N).card

theorem number_of_valid_Ns :
  count_valid_N = 1287 :=
sorry

end number_of_valid_Ns_l86_86306


namespace boy_distance_l86_86528

theorem boy_distance (minutes : ℕ) (speed : ℕ) (time_in_seconds : minutes * 60 = 2160) : 
  (distance : ℕ) = speed * 2160 :=
by
  -- Given conditions
  have h_time : minutes = 36 := rfl
  have h_speed : speed = 4 := rfl

  -- Calculate the total distance
  let distance := speed * 2160
  -- Prove the distance is correct
  calc distance = (speed * 2160) : by rfl
             ... = (4 * 2160) : by rw h_speed
             ... = 8640 : by norm_num

end boy_distance_l86_86528


namespace find_value_of_N_l86_86028

theorem find_value_of_N (x N : ℝ) (hx : x = 1.5) (h : (3 + 2 * x)^5 = (N + 3 * x)^4) : N = 1.5 := by
  -- Here we will assume that the proof is filled in and correct.
  sorry

end find_value_of_N_l86_86028


namespace simplify_expression_l86_86421

theorem simplify_expression (x y : ℝ) : 
  let i := Complex.I in
  (x + i * y + 1) * (x - i * y + 1) = (x + 1) ^ 2 - y ^ 2 :=
by
  sorry

end simplify_expression_l86_86421


namespace trig_identity_l86_86883

open Real

theorem trig_identity : sin (20 * π / 180) * cos (10 * π / 180) - cos (160 * π / 180) * sin (170 * π / 180) = 1 / 2 := 
by
  sorry

end trig_identity_l86_86883


namespace projection_circumcenter_l86_86340

variable {P A B C O : Point}

-- Assuming PA = PB = PC is a condition
axiom h1 : dist P A = dist P B
axiom h2 : dist P B = dist P C

-- Assuming O is the projection of P onto plane ABC
axiom h3 : is_projection O P (plane.mk A B C)

-- The theorem to prove: O is the circumcenter of triangle ABC
theorem projection_circumcenter :
  is_circumcenter O A B C :=
by {
  sorry
}

end projection_circumcenter_l86_86340


namespace k_2_sufficient_but_not_necessary_l86_86280

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (1, k^2 - 1) - (2, 1)

def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

theorem k_2_sufficient_but_not_necessary (k : ℝ) :
  k = 2 → perpendicular vector_a (vector_b k) ∧ ∃ k, not (k = 2) ∧ perpendicular vector_a (vector_b k) :=
by
  sorry

end k_2_sufficient_but_not_necessary_l86_86280


namespace exists_real_k_l86_86459

theorem exists_real_k (c : Fin 1998 → ℕ)
  (h1 : 0 ≤ c 1)
  (h2 : ∀ (m n : ℕ), 0 < m → 0 < n → m + n < 1998 → c m + c n ≤ c (m + n) ∧ c (m + n) ≤ c m + c n + 1) :
  ∃ k : ℝ, ∀ n : Fin 1998, 1 ≤ n → c n = Int.floor (n * k) :=
by
  sorry

end exists_real_k_l86_86459


namespace min_infection_sources_needed_l86_86839

theorem min_infection_sources_needed 
  (total_squares : ℕ) 
  (initial_black : ℕ)
  (infection_cond : ℕ → ℕ → Prop) : 
  (total_squares = 72 ∧ 
  infection_cond = (λ adj infected, infected ≥ 2) ∧ 
  initial_black = 4) → 
  ∃ (min_sources : ℕ), 
    (min_sources = initial_black ∧ 
    ∀ square, min_sources <= 72 ∧ 
    infection_cond (adjacent square infected) = infected) :=
by
  intros h
  cases h with htotal hsquares
  exists 4
  exact ⟨htotal, hsquares⟩
  sorry

end min_infection_sources_needed_l86_86839


namespace derivative_at_pi_over_3_l86_86014

-- Define the function f
def f (x : ℝ) : ℝ := 1 + Real.sin x

-- State the theorem to be proven
theorem derivative_at_pi_over_3 : 
  (Real.deriv f) (Real.pi / 3) = 1 / 2 :=
by 
  -- Proof goes here
  sorry

end derivative_at_pi_over_3_l86_86014


namespace socks_selection_l86_86682

/-!
  # Socks Selection Problem
  Prove the total number of ways to choose a pair of socks of different colors
  given:
  1. there are 5 white socks,
  2. there are 4 brown socks,
  3. there are 3 blue socks,
  is equal to 47.
-/

theorem socks_selection : 
  let white_socks := 5
  let brown_socks := 4
  let blue_socks := 3
  5 * 4 + 4 * 3 + 5 * 3 = 47 :=
by
  let white_socks := 5
  let brown_socks := 4
  let blue_socks := 3
  sorry

end socks_selection_l86_86682


namespace sequence_sum_l86_86667

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then n - 1 else n

-- The main statement to prove the sum of the sequence
theorem sequence_sum :
  (Finset.range 100).sum (λ n, a (n + 1)) = 5000 :=
sorry

end sequence_sum_l86_86667


namespace smallest_number_among_l86_86561

noncomputable def smallest_real among (a b c d : Real) : Real :=
  if a <= b && a <= c && a <= d then a
  else if b <= a && b <= c && b <= d then b
  else if c <= a && c <= b && c <= d then c
  else d

theorem smallest_number_among :=
  let r1 := -1.0
  let r2 := -|(-2.0)|
  let r3 := 0.0
  let r4 := Real.pi
  smallest_real r1 r2 r3 r4 = -2.0 :=
by
  -- placeholder for proof
  sorry

end smallest_number_among_l86_86561


namespace polynomial_evaluation_l86_86750

noncomputable def Q (x : ℝ) : ℝ :=
  x^4 + x^3 + 2 * x

theorem polynomial_evaluation :
  Q (3) = 114 := by
  -- We assume the conditions implicitly in this equivalence.
  sorry

end polynomial_evaluation_l86_86750


namespace polynomial_degree_15_l86_86054

noncomputable def polynomial_degree (x : ℝ) (a b c d e f g : ℝ) : ℕ :=
  polynomial.degree (polynomial.C a * polynomial.X^8 + polynomial.C b * polynomial.X^2 + polynomial.C c * polynomial.X^0) +
  polynomial.degree (polynomial.X^4 + polynomial.C d * polynomial.X^3 + polynomial.C e * polynomial.X^0) +
  polynomial.degree (polynomial.X^2 + polynomial.C f * polynomial.X^0) +
  polynomial.degree (polynomial.C 2 * polynomial.X + polynomial.C g * polynomial.X^0)

theorem polynomial_degree_15 (a b c d e f g : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
                              (h₃ : d ≠ 0) (h₄ : e ≠ 0) (h₅ : f ≠ 0) (h₆ : g ≠ 0) :
  polynomial_degree x a b c d e f g = 15 :=
sorry

end polynomial_degree_15_l86_86054


namespace arithmetic_sequence_a10_l86_86247

variable {a : ℕ → ℝ}

-- Given the sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, (n < m) → (a (m + 1) - a m = a (n + 1) - a n)

-- Conditions
theorem arithmetic_sequence_a10 (h_arith : is_arithmetic_sequence a) 
                                (h1 : a 6 + a 8 = 16)
                                (h2 : a 4 = 1) :
  a 10 = 15 :=
sorry

end arithmetic_sequence_a10_l86_86247


namespace target_l86_86250

variable (α : ℝ)

-- Given condition
def cond : Prop := cos (α - 30 * Real.pi / 180) + sin α = (3 / 5) * Real.sqrt 3

-- Target statement to prove
theorem target (h : cond α) : cos (60 * Real.pi / 180 - α) = 3 / 5 := 
by
  sorry

end target_l86_86250


namespace compute_value_l86_86950

open Real

theorem compute_value : 
  ((1 / 2 : ℝ) ^ (-2)) + log 2 - log (1 / 5) = 5 := 
by 
  sorry

end compute_value_l86_86950


namespace wholesale_cost_l86_86543

variable (W R P : ℝ)

-- Conditions
def retail_price := R = 1.20 * W
def employee_discount := P = 0.95 * R
def employee_payment := P = 228

-- Theorem statement
theorem wholesale_cost (H1 : retail_price R W) (H2 : employee_discount P R) (H3 : employee_payment P) : W = 200 :=
by
  sorry

end wholesale_cost_l86_86543


namespace parallel_lines_perpendicular_lines_l86_86847

variables {x₁ y₁ z₁ x₂ y₂ z₂ m₁ n₁ p₁ m₂ n₂ p₂ : ℝ} -- Declare the necessary variables in ℝ

-- Define the conditions for the given lines
def line1 (x y z : ℝ) : Prop := (x - x₁) / m₁ = (y - y₁) / n₁ ∧ (y - y₁) / n₁ = (z - z₁) / p₁
def line2 (x y z : ℝ) : Prop := (x - x₂) / m₂ = (y - y₂) / n₂ ∧ (y - y₂) / n₂ = (z - z₂) / p₂

-- Prove that the lines are parallel if and only if the direction vectors are proportional
theorem parallel_lines : 
  ∀ (x y z : ℝ), line1 x y z ∧ line2 x y z ↔ m₁ / m₂ = n₁ / n₂ ∧ n₁ / n₂ = p₁ / p₂ :=
by
  sorry

-- Prove that the lines are perpendicular if and only if the dot product of the direction vectors is zero
theorem perpendicular_lines :
  ∀ (x y z : ℝ), line1 x y z ∧ line2 x y z ↔ m₁ * m₂ + n₁ * n₂ + p₁ * p₂ = 0 :=
by
  sorry

end parallel_lines_perpendicular_lines_l86_86847


namespace largest_five_digit_divisible_by_8_l86_86854

def divisible_by_eight (n : ℤ) : Prop := n % 8 = 0

theorem largest_five_digit_divisible_by_8 :
  ∃ n : ℤ, 10000 ≤ n ∧ n ≤ 99999 ∧ divisible_by_eight n ∧ ∀ m : ℤ, 10000 ≤ m ∧ m ≤ 99999 ∧ divisible_by_eight m → m ≤ n :=
begin
  use 99992,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { unfold divisible_by_eight,
    norm_num },
  { intros m h1 h2 h3,
    have h4 := h2.trans (nat.le_of_lt_succ (show m < 99993, by linarith)),
    have h5 := h2.trans (nat.le_of_lt_succ (show m < 10000, by linarith)),
    norm_num at h4 h5,
    interval_cases m with l hl,
    repeat { split_ifs at hl; try { linarith } },
    all_goals { try { norm_num at hl } },
  },
  sorry
end

end largest_five_digit_divisible_by_8_l86_86854


namespace value_of_a_value_of_sin_A_plus_pi_div_3_l86_86386

variables {A B : ℝ}
constants a b c : ℝ
hypothesis hb : b = 3
hypothesis hc : c = 1
hypothesis hA : A = 2 * B

noncomputable def find_a : ℝ :=
  let sinA := Math.sin A
  in 2 * (Math.sqrt 3)

noncomputable def find_sin_A_plus_pi_div_3 (A : ℝ) : ℝ :=
  let cosA := -1 / 3
  let sinA := 2 * Math.sqrt 2 / 3
  in 1/2 * sinA + Math.sqrt 3/2 * cosA

theorem value_of_a :
  a = find_a :=
sorry

theorem value_of_sin_A_plus_pi_div_3 :
  (find_sin_A_plus_pi_div_3 A) = (2 * Math.sqrt 2 - Math.sqrt 3) / 6 :=
sorry

end value_of_a_value_of_sin_A_plus_pi_div_3_l86_86386


namespace max_sqrt3_yA_plus_xB_eq_one_l86_86249

theorem max_sqrt3_yA_plus_xB_eq_one
  (x_A y_A x_B y_B : ℝ)
  (hA : x_A^2 + y_A^2 = 1)
  (hB : x_B = cos (atan2 y_A x_A + π/3))
  (hBy : y_B = sin (atan2 y_A x_A + π/3)) :
  ∃ α : ℝ, ((α = sqrt 3 * y_A + x_B) ∧ (α ≤ 1) ∧ (∀ β : ℝ, β < 1 →  β ≤ α)) :=
by
  sorry

end max_sqrt3_yA_plus_xB_eq_one_l86_86249


namespace trapezoid_area_l86_86542

-- Define the conditions
def rectangle_side_lengths (a b : ℝ) : Prop :=
  a = 5 ∧ b = 8

-- Define the folded trapezoid area problem
theorem trapezoid_area (a b : ℝ) (h : rectangle_side_lengths a b) : 
  ∃ (area : ℝ), area = 55 / 2 :=
by
  use 55 / 2
  sorry

end trapezoid_area_l86_86542


namespace part1_part2_part3_l86_86637

-- Define conditions for vectors a and b
variables {a b : EuclideanSpace ℝ (Fin 3)} {t : ℝ}
def a_norm : ℝ := 2
def b_norm : ℝ := 1
def angle_ab : ℝ := Real.pi / 3

-- Given conditions
axiom a_dot_b : (a.dot b) = a_norm * b_norm * Real.cos angle_ab
axiom a_norm_val : ∥a∥ = a_norm
axiom b_norm_val : ∥b∥ = b_norm
axiom angle_val : angle.vectorAngle a b = angle_ab

-- Prove part 1: a · b = 1
theorem part1 : a.dot b = 1 := by
  sorry

-- Define vectors m and n
def m : EuclideanSpace ℝ (Fin 3) := a + t • b
def n : EuclideanSpace ℝ (Fin 3) := t • a + 2 • b

-- Define function f(t) = m · n
def f (t : ℝ) : ℝ := (m.dot n)

-- Prove part 2: f(t) = t^2 + 6t + 2
theorem part2 : f t = t^2 + 6 * t + 2 := by
  sorry

-- Define function g(t) = f(t) / t
def g (t : ℝ) : ℝ := f t / t

-- Prove part 3: The range of g(t) on [1, 3] is [2√2 + 6, 29/3]
theorem part3 : Set.range (g ∘ (λ t, (Set.Icc 1 3 : Set ℝ).val (Subtype.mk t (And.intro (Le.le.refl t) (by linarith))))) = Set.Icc (2 * Real.sqrt 2 + 6) (29 / 3) := by
  sorry

end part1_part2_part3_l86_86637


namespace interval_of_n_l86_86990

noncomputable def divides (a b : ℕ) : Prop := ∃ k, b = k * a

theorem interval_of_n (n : ℕ) (hn : 0 < n ∧ n < 2000)
  (h1 : divides n 9999)
  (h2 : divides (n + 4) 999999) :
  801 ≤ n ∧ n ≤ 1200 :=
sorry

end interval_of_n_l86_86990


namespace number_of_students_third_l86_86457

-- Define the ratio and the total number of samples.
def ratio_first : ℕ := 3
def ratio_second : ℕ := 3
def ratio_third : ℕ := 4
def total_sample : ℕ := 50

-- Define the condition that the sum of ratios equals the total proportion numerator.
def sum_ratios : ℕ := ratio_first + ratio_second + ratio_third

-- Final proposition: the number of students to be sampled from the third grade.
theorem number_of_students_third :
  (ratio_third * total_sample) / sum_ratios = 20 := by
  sorry

end number_of_students_third_l86_86457


namespace total_messages_equation_l86_86700

theorem total_messages_equation (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  exact h

end total_messages_equation_l86_86700


namespace sum_of_roots_l86_86255

variables {f g : ℝ → ℝ}

-- Conditions from the problem
axiom A1 : ∀ x, f x = f (-x)              -- f is even function.
axiom A2 : ∀ x, g x = -g (-x)             -- g is odd function.
axiom A3 : ∀ y, f y = 0 ↔ y = 0 ∨ y = x₁ ∨ y = x₂  -- Roots of f(x) = 0 where |x₁| = |x₂| ∈ (1, 2)
axiom A4 : ∀ y, g y = 0 ↔ y = 0 ∨ y = x₃ ∨ y = x₄  -- Roots of g(x) = 0 where |x₃| = |x₄| ∈ (0, 1)
axiom A5 : ∀ x, -1 ≤ f x ∧ f x ≤ 1        -- Range of f
axiom A6 : ∀ x, -2 ≤ g x ∧ g x ≤ 2        -- Range of g

-- Statement to prove
theorem sum_of_roots : 
  let a := 3 in
  let b := 9 in
  let c := 9 in
  let d := 9 in
  a + b + c + d = 30 :=
by 
  sorry

end sum_of_roots_l86_86255


namespace arithmetic_sequence_15th_term_l86_86590

/-- 
The arithmetic sequence with first term 1 and common difference 3.
The 15th term of this sequence is 43.
-/
theorem arithmetic_sequence_15th_term :
  ∀ (a1 d n : ℕ), a1 = 1 → d = 3 → n = 15 → (a1 + (n - 1) * d) = 43 :=
by
  sorry

end arithmetic_sequence_15th_term_l86_86590


namespace remainderOfSumOfSquares_l86_86221

-- Definitions based on the given problem's conditions
def sumOfSquaresMod (n : ℕ) (m : ℕ) : ℕ :=
  (Finset.sum (Finset.range n) (λ x, (x + 1) * (x + 1))) % m

-- The proof problem in Lean 4 statement
theorem remainderOfSumOfSquares :
  sumOfSquaresMod 150 5 = 0 :=
sorry

end remainderOfSumOfSquares_l86_86221


namespace monotonic_decreasing_interval_l86_86024

-- Define the function y = 3 * sin(2 * x + π / 6)
def f (x : Real) : Real := 3 * Real.sin (2 * x + Real.pi / 6)

-- State the theorem about the monotonic decreasing interval of the function f.
theorem monotonic_decreasing_interval (k : ℤ) : 
  ∀ (x : Real), 
    (k.toReal * Real.pi + Real.pi / 6 <= x) ∧ (x <= k.toReal * Real.pi + 2 * Real.pi / 3) 
    ↔ 
    (f' : Real → Real) (x) < 0 
      sorry

end monotonic_decreasing_interval_l86_86024


namespace count_positive_integers_N_number_of_N_l86_86293

theorem count_positive_integers_N : ∀ N : ℕ, N < 2000 → ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N :=
begin
  sorry
end

theorem number_of_N : {N : ℕ // N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N}.card = 412 :=
begin
  sorry
end

end count_positive_integers_N_number_of_N_l86_86293


namespace false_proposition_l86_86928

def proposition_A (k : ℝ) (K2 : ℝ) (X Y : Type) : Prop :=
  (k < K2) → (credibility_related (X Y))

def proposition_B (R2 : ℝ) : Prop :=
  R2 > 0 → (better_fitting (R2))

def proposition_C (corr : ℝ) : Prop :=
  (-1 ≤ corr) ∧ (corr ≤ 1) → (stronger_correlation (corr))

def proposition_D (height : ℝ) (categories : Type) : Prop :=
  (height > 0) → (freq_representation (categories))

theorem false_proposition (k K2 R2 corr height : ℝ) (X Y categories : Type):
  ¬proposition_A k K2 X Y ∧ 
  proposition_B R2 ∧ 
  proposition_C corr ∧ 
  proposition_D height categories :=
sorry

end false_proposition_l86_86928


namespace Jillian_collected_29_l86_86739

variable (Savannah_shells Clayton_shells total_friends friend_shells : ℕ)

def Jillian_shells : ℕ :=
  let total_shells := friend_shells * total_friends
  let others_shells := Savannah_shells + Clayton_shells
  total_shells - others_shells

theorem Jillian_collected_29 (h_savannah : Savannah_shells = 17) 
                             (h_clayton : Clayton_shells = 8) 
                             (h_friends : total_friends = 2) 
                             (h_friend_shells : friend_shells = 27) : 
  Jillian_shells Savannah_shells Clayton_shells total_friends friend_shells = 29 :=
by
  sorry

end Jillian_collected_29_l86_86739


namespace count_convex_polygons_at_least_3_sides_with_conditions_l86_86844

-- Definition of the problem conditions
def n : ℕ := 12
def min_vertices : ℕ := 3
def angle_condition : ℕ := 30

-- Define the problem statement as a Lean theorem
theorem count_convex_polygons_at_least_3_sides_with_conditions :
  ∃ (num_polygons : ℕ), num_polygons = 4017 ∧
  ∀ (polygon_set : finset (fin n)), polygon_set.card ≥ min_vertices →
  ∀ (points : multiset (fin n)), points ∈ polygon_set.powerset.val →
  ∀ (polygon : list (fin n)), polygon.length ≥ min_vertices → 
  (∀ i j, i ≠ j → (triangle_angle (polygon.nth_le i sorry) (polygon.nth_le j sorry)) ≥ angle_condition) :=
sorry

end count_convex_polygons_at_least_3_sides_with_conditions_l86_86844


namespace find_pairs_with_5_cats_l86_86768

theorem find_pairs_with_5_cats (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (p : {a, b, c, d} = {1, 2, 3, 4}) :
  ((a + b = 5) ∨ (a + c = 5) ∨ (a + d = 5)) ∧ ∃ x y ∈ {a, b, c, d}, x ≠ y ∧ x + y = 5 :=
by
  sorry

end find_pairs_with_5_cats_l86_86768


namespace length_CD_l86_86489

theorem length_CD (AD BC BD CD : ℝ) 
  (h1 : AD ∥ BC) 
  (h2 : BD = 2) 
  (h3 : ∠ DBA = 30) 
  (h4 : ∠ BDC = 60) 
  (h5 : BC / AD = 7 / 3) 
  : CD = 8 / 3 :=
by
  sorry

end length_CD_l86_86489


namespace geometric_problem_l86_86384

open Set

variable (Point : Type) [EuclideanGeometry Point]

def square (A B C D : Point) : Prop :=
  quadrilateral A B C D ∧ (d A B = d B C) ∧ (d B C = d C D) ∧ (d C D = d D A) ∧
  (angle A B C = 90) ∧ (angle B C D = 90) ∧ (angle C D A = 90) ∧ (angle D A B = 90)

def equilateral_triangle (A B C : Point) : Prop := (d A B = d B C) ∧ (d B C = d C A)

variable (A B C D E N F M P Q : Point)

theorem geometric_problem 
  (H1 : square A B C D) 
  (H2 : E ∈ line_segment A B)
  (H3 : N ∈ line_segment C D)
  (H4 : F ∈ line_segment B C)
  (H5 : M ∈ line_segment B C)
  (H6 : collinear P (line_through A N ∩ line_through D E)) 
  (H7 : collinear Q (line_through A M ∩ line_through E F))
  (H8 : equilateral_triangle A M N) 
  (H9 : equilateral_triangle D E F) 
  : d P Q = d F M :=
sorry

end geometric_problem_l86_86384


namespace number_of_non_congruent_rectangles_l86_86115

def valid_rectangles_with_even_area (perimeter: ℕ) : ℕ :=
  let target_sum := perimeter / 2 in
  let pairs := (finset.range target_sum).filter (λ w, (w * (target_sum - w)) % 2 = 0) in
  pairs.card

theorem number_of_non_congruent_rectangles : valid_rectangles_with_even_area 100 = 25 :=
by {
  let perimeter := 100,
  let target_sum := perimeter / 2,
  let filtered_cards := (finset.range target_sum).filter (λ w, (w * (target_sum - w)) % 2 = 0),
  
  have h_target_sum : target_sum = 50, 
    by refl,

  have h_filtered_cards : filtered_cards.card = 25,
    sorry,
  
  show valid_rectangles_with_even_area perimeter = 25,
    by {
      simp [valid_rectangles_with_even_area, target_sum, h_target_sum, h_filtered_cards],
    }
}

end number_of_non_congruent_rectangles_l86_86115


namespace APBR_cyclic_l86_86743

-- Let ω be the circumcircle of triangle ABC
variable (ω : Circle)
-- Let K and L be points on ω such that KL is a diameter
variables (K L : ω)
-- M is the midpoint of AB
variables (A B C M : Point) (H_midpoint : M = midpoint A B)
-- KL passes through M (diameter condition)
variable (H_M_on_KL : onLine M (line K L))
-- Circle through L and M meets CK at points P and Q, with Q on KP
variables (circle_LM : Circle) (P Q : Point) 
variables (H_circle_LM : circleThrough [L, M] circle_LM) 
variables (H_meets_CK : meets circle_LM (line C K) P Q)
variable (H_Q_on_KP : onLine Q (line K P))
-- LQ meets the circumcircle of △KMQ again at R
variables (H_circle_KMQ : Circle) (H_circumcircle_KMQ : circleThrough [K, M, Q] H_circle_KMQ) 
variables (R : Point) 
variables (H_meets_LQ : meets (line L Q) H_circle_KMQ R)

theorem APBR_cyclic : cyclic quadrilateral A P B R :=
by sorry

end APBR_cyclic_l86_86743


namespace expected_returned_balls_l86_86153

-- Define the probabilities for adjacent and non-adjacent swaps
def P_adjacency : ℝ := 2/3
def P_non_adjacency : ℝ := 1/3

-- Define calculations involving probabilities
def T_swap : ℝ := 4/8
def T_double_swap : ℝ := (4/8) * (1/8)
def T_no_swap : ℝ := (4/8) * (4/8)

def expected_balls_in_original_position (n : ℕ) : ℝ :=
  let P_original_one := T_double_swap + T_no_swap in
  n * P_original_one

theorem expected_returned_balls :
  expected_balls_in_original_position 8 = 2.5 :=
  by
  have P_original_one := 1/16 + 1/4
  have P1_calc : P_original_one = 5/16 := by sorry
  have E_calc : (8 : ℝ) * (5 / 16) = 2.5 := by sorry
  rw [P1_calc, E_calc]
  -- state the goal to show that the calculation adheres. Hence expected value results in:
  exact sorry

end expected_returned_balls_l86_86153


namespace B_completes_remaining_work_in_12_days_l86_86897

-- Definitions for conditions.
def work_rate_a := 1/15
def work_rate_b := 1/18
def days_worked_by_a := 5

-- Calculation of work done by A and the remaining work for B
def work_done_by_a := days_worked_by_a * work_rate_a
def remaining_work := 1 - work_done_by_a

-- Proof statement
theorem B_completes_remaining_work_in_12_days : 
  ∀ (work_rate_a work_rate_b : ℚ), 
    work_rate_a = 1/15 → 
    work_rate_b = 1/18 → 
    days_worked_by_a = 5 → 
    work_done_by_a = days_worked_by_a * work_rate_a → 
    remaining_work = 1 - work_done_by_a → 
    (remaining_work / work_rate_b) = 12 :=
by 
  intros 
  sorry

end B_completes_remaining_work_in_12_days_l86_86897


namespace fifth_term_arithmetic_sequence_l86_86642

theorem fifth_term_arithmetic_sequence : 
  ∀ (a₁ a₂ a₃ a₅ : ℤ), 
  a₁ = -1 ∧ a₃ = 5 ∧ a₂ = (a₁ + a₃) / 2 ∧
  a₅ = a₁ + 4 * (a₂ - a₁) → a₅ = 11 := 
by 
  intros a₁ a₂ a₃ a₅ h,
  have ha₁ : a₁ = -1 := h.1,
  have ha₃ : a₃ = 5 := h.2.1,
  have ha₂ : a₂ = (a₁ + a₃) / 2 := h.2.2.1,
  have ha₅ : a₅ = a₁ + 4 * (a₂ - a₁) := h.2.2.2,
  sorry

end fifth_term_arithmetic_sequence_l86_86642


namespace common_tangent_problem_l86_86126

-- Given Definitions
variables {A B C : Point}
variables {ω : Circle}
variables {γ1 γ2 : Circle}
variable {AB AC : Line}

-- Conditions
def triangle (A B C : Point) : Prop :=
AB < AC ∧ ∃C ∈ ω, ∃B ∈ ω

def circles_touch_lines (γ1 γ2 : Circle) (AB AC : Line) : Prop :=
∃ (O1 ∈ γ1) (O2 ∈ γ2),
  (touches γ1 AB ↔ touches γ2 AC) ∧ centers_on ω [γ1 O1, γ2 O2]

def centers_on (ω : Circle) : Prop :=
∀ (γ1 γ2 : Circle) ∈ ω, centers_on γ1.eq_center_on ω ∧ centers_on γ2.eq_center_on ω

-- Goal
theorem common_tangent_problem (h : triangle A B C ∧ circles_touch_lines γ1 γ2 AB AC ∧ centers_on ω γ1 γ2) :
  ∃γ1 γ2, common_external_tangent C γ1 γ2 :=
sorry -- We leave the proof as 'sorry'

end common_tangent_problem_l86_86126


namespace area_of_triangle_CDE_l86_86729

theorem area_of_triangle_CDE (A B C D E : Type*) 
  [linear_ordered_field A] [linear_ordered_field B] 
  [linear_ordered_field C] [linear_ordered_field D] [linear_ordered_field E]
  (AD BD AC : A) (area_ABC : B)
  (h1 : AD = 2 * BD) 
  (h2 : E = (AC + C)/2)
  (h3 : area_ABC = 36) :
  (12 / 2 = 6) :=
by 
  sorry

end area_of_triangle_CDE_l86_86729


namespace value_of_a_when_x_is_3_root_l86_86689

theorem value_of_a_when_x_is_3_root (a : ℝ) :
  (3 ^ 2 + 3 * a + 9 = 0) -> a = -6 := by
  intros h
  sorry

end value_of_a_when_x_is_3_root_l86_86689


namespace operations_equivalence_l86_86941

def word := List Char

def first_operation (w : word) : word :=
  w ++ ['δ']

def insert_a_anywhere (w : word) : List word :=
  List.concatMap (λ i, [w.take i ++ ['A'] ++ w.drop i]) (List.range (w.length + 1))

def apply_first_operation (w : word) : List word :=
  let inserted_a := insert_a_anywhere w
  List.map first_operation inserted_a

def apply_second_operation (w : word) : List word :=
  insert_a_anywhere w

theorem operations_equivalence (w : word) :
  let set1 := (apply_first_operation w).toFinset
  let set2 := (apply_second_operation w).toFinset
  set1 = set2 :=
by
  sorry

end operations_equivalence_l86_86941


namespace length_of_diagonal_PR_l86_86717

theorem length_of_diagonal_PR (PQ QR RS SP : ℝ) (angle_RSP : ℝ) (h1 : PQ = 12) (h2 : QR = 12) (h3 : RS = 20) (h4 : SP = 20) (h5 : angle_RSP = 90) : 
  real.sqrt (RS^2 + SP^2) = 20 * real.sqrt 2 :=
by 
  sorry

end length_of_diagonal_PR_l86_86717


namespace johns_spent_amount_l86_86922

def original_price : ℝ := 2000
def increase_rate : ℝ := 0.02

theorem johns_spent_amount : 
  let increased_amount := original_price * increase_rate in
  let john_total := original_price + increased_amount in
  john_total = 2040 :=
by
  sorry

end johns_spent_amount_l86_86922


namespace problem_statement_l86_86520

theorem problem_statement (a b : ℤ) (h1 : b = 7) (h2: a * b = 2 * (a + b) + 1) :
  b - a = 4 := by
  sorry

end problem_statement_l86_86520


namespace cylinder_height_l86_86104

theorem cylinder_height
  (V : ℝ → ℝ → ℝ) 
  (π : ℝ)
  (r h : ℝ)
  (vol_increase_height : ℝ)
  (vol_increase_radius : ℝ)
  (h_increase : ℝ)
  (r_increase : ℝ)
  (original_radius : ℝ) :
  V r h = π * r^2 * h → 
  vol_increase_height = π * r^2 * h_increase →
  vol_increase_radius = π * ((r + r_increase)^2 - r^2) * h →
  r = original_radius →
  vol_increase_height = 72 * π →
  vol_increase_radius = 72 * π →
  original_radius = 3 →
  r_increase = 2 →
  h_increase = 2 →
  h = 4.5 :=
by
  sorry

end cylinder_height_l86_86104


namespace starting_number_l86_86800

theorem starting_number (n : ℕ) (h1 : n % 11 = 3) (h2 : (n + 11) % 11 = 3) (h3 : (n + 22) % 11 = 3) 
  (h4 : (n + 33) % 11 = 3) (h5 : (n + 44) % 11 = 3) (h6 : n + 44 ≤ 50) : n = 3 := 
sorry

end starting_number_l86_86800


namespace triangle_area_l86_86864

theorem triangle_area {A B C L : Type*} {dist : A → A → ℝ} 
  (h1 : ∃ (d : A → A → ℝ), 
          d A B = d A L + d L B ∧ 
          d A C = d A L + d L C ∧ 
          d B C = d B A + d A C) 
  (hAL : dist A L = 4)
  (hBL : dist B L = 2 * Real.sqrt 15)
  (hCL : dist C L = 5) :
  ∃ (area : ℝ), 
    area = (9 * Real.sqrt 231) / 4 :=
by
  sorry

end triangle_area_l86_86864


namespace initial_elephants_l86_86047

theorem initial_elephants (E : ℕ) :
  (E + 35 + 135 + 125 = 315) → (5 * 35 / 7 = 25) → (5 * 25 = 125) → (135 = 125 + 10) →
  E = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_elephants_l86_86047


namespace minimum_value_of_f_l86_86448

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem minimum_value_of_f :
  ∃ x : ℝ, is_local_min f x ∧ f x = -2 := 
sorry

end minimum_value_of_f_l86_86448


namespace nth_inequality_l86_86275

theorem nth_inequality (n : ℕ) : 
  (1 + (Finset.sum (Finset.range n) (λ k, (1 : ℝ) / (k + 2)^2)) < (2 * n + 1) / (n + 1)) := 
by 
  sorry

end nth_inequality_l86_86275


namespace evaluate_expression_l86_86198

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l86_86198


namespace express_h_l86_86380

variable {R : Type} [LinearOrderedField R]

def f (h : R → R) (x : R) : R := (h(x + 1) + h(x - 1)) / 2
def g (h : R → R) (x : R) : R := (h(x + 4) + h(x - 4)) / 2

theorem express_h (h : R → R) (x : R) :
  h(x) = g h x - f h (x - 3) + f h (x - 1) + f h (x + 1) - f h (x + 3) := 
sorry

end express_h_l86_86380


namespace mapleton_math_team_combinations_l86_86011

open Nat

theorem mapleton_math_team_combinations (girls boys : ℕ) (team_size girl_on_team boy_on_team : ℕ)
    (h_girls : girls = 4) (h_boys : boys = 5) (h_team_size : team_size = 4)
    (h_girl_on_team : girl_on_team = 3) (h_boy_on_team : boy_on_team = 1) :
    (Nat.choose girls girl_on_team) * (Nat.choose boys boy_on_team) = 20 := by
  sorry

end mapleton_math_team_combinations_l86_86011


namespace ratio_of_logs_l86_86424

noncomputable def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem ratio_of_logs (a b: ℝ) (h1 : log_base 8 a = log_base 18 b) 
    (h2 : log_base 18 b = log_base 32 (a + b)) 
    (hpos : 0 < a ∧ 0 < b) :
    b / a = (3 + 2 * (Real.log 3 / Real.log 2)) / (1 + 2 * (Real.log 3 / Real.log 2) + 5) :=
by 
    sorry

end ratio_of_logs_l86_86424


namespace jori_water_left_l86_86741

theorem jori_water_left (initial_water : ℚ) (used_water : ℚ) : initial_water = 3 ∧ used_water = 5/4 → initial_water - used_water = 7/4 :=
by {
  intro h,
  cases h with h1 h2,
  rw [h1, h2],
  norm_num,
}

end jori_water_left_l86_86741


namespace number_of_valid_Ns_l86_86307

noncomputable def count_valid_N : ℕ :=
  (finset.range 2000).filter (λ N, ∃ x : ℝ, x^floor x = N).card

theorem number_of_valid_Ns :
  count_valid_N = 1287 :=
sorry

end number_of_valid_Ns_l86_86307


namespace tourists_left_l86_86535

theorem tourists_left (initial_tourists eaten_by_anacondas poisoned_fraction recover_fraction : ℕ) 
(h_initial : initial_tourists = 30) 
(h_eaten : eaten_by_anacondas = 2)
(h_poisoned_fraction : poisoned_fraction = 2)
(h_recover_fraction : recover_fraction = 7) :
  initial_tourists - eaten_by_anacondas - (initial_tourists - eaten_by_anacondas) / poisoned_fraction + (initial_tourists - eaten_by_anacondas) / poisoned_fraction / recover_fraction = 16 :=
by
  sorry

end tourists_left_l86_86535


namespace eval_ceil_floor_sum_l86_86206

theorem eval_ceil_floor_sum :
  (Int.ceil (7 / 3) + Int.floor (- (7 / 3))) = 0 :=
by
  sorry

end eval_ceil_floor_sum_l86_86206


namespace find_angle_B_find_perimeter_l86_86352

-- Definitions of the variables and conditions
variable {α : Type} [LinearOrder α] [Field α] [TrigonometricFunctions α] -- for trigonometric functions

-- Definitions for the angles and sides of triangle ABC
variables (A B C a b c : α)

-- Conditions provided in the problem
variable (h1 : c * sin ((A + C) / 2) = b * sin C)
variable (h2 : BD = 1)
variable (h3 : b = sqrt 3)

-- Proving the angle B
theorem find_angle_B : B = π / 3 :=
sorry

-- Proving the perimeter of triangle ABC
theorem find_perimeter (B_eq : B = π / 3) : a + b + c = 3 + sqrt 3 :=
sorry

end find_angle_B_find_perimeter_l86_86352


namespace ceiling_floor_sum_l86_86160

theorem ceiling_floor_sum : (Int.ceil (7 / 3) + Int.floor (-(7 / 3)) = 0) := by
  sorry

end ceiling_floor_sum_l86_86160


namespace max_notebooks_with_budget_l86_86387

/-- Define the prices and quantities of notebooks -/
def notebook_price : ℕ := 2
def four_pack_price : ℕ := 6
def seven_pack_price : ℕ := 9
def max_budget : ℕ := 15

def total_notebooks (single_packs four_packs seven_packs : ℕ) : ℕ :=
  single_packs + 4 * four_packs + 7 * seven_packs

theorem max_notebooks_with_budget : 
  ∃ (single_packs four_packs seven_packs : ℕ), 
    notebook_price * single_packs + 
    four_pack_price * four_packs + 
    seven_pack_price * seven_packs ≤ max_budget ∧ 
    booklet_price * single_packs + 
    four_pack_price * four_packs + 
    seven_pack_price * seven_packs + total_notebooks single_packs four_packs seven_packs = 11 := 
by
  sorry

end max_notebooks_with_budget_l86_86387


namespace fraction_of_green_marbles_l86_86703

/-- 
Suppose that initially 4 out of every 7 marbles in a basket are blue,
and the rest are green. If the number of green marbles is tripled 
while the number of blue marbles remains constant, prove that the 
new fraction of green marbles in the basket is 9 out of 13.
-/
theorem fraction_of_green_marbles (n : ℕ) (h : n > 0)
  (initial_blue_fraction : ℚ := 4 / 7) 
  (initial_green_fraction : ℚ := 3 / 7)
  (new_green_fraction : ℚ := 9 / 13) :
  let blue_marbles := initial_blue_fraction * n,
      green_marbles := initial_green_fraction * n,
      new_green_marbles := 3 * green_marbles,
      total_new_marbles := blue_marbles + new_green_marbles in
  new_green_fraction = new_green_marbles / total_new_marbles :=
by
  sorry

end fraction_of_green_marbles_l86_86703


namespace ceil_floor_sum_l86_86180

theorem ceil_floor_sum : (Int.ceil (7 / 3 : ℝ) + Int.floor (- 7 / 3 : ℝ) = 0) := 
  by {
    -- referring to the proof steps
    have h1: Int.ceil (7 / 3 : ℝ) = 3,
    { sorry },
    have h2: Int.floor (- 7 / 3 : ℝ) = -3,
    { sorry },
    rw [h1, h2],
    norm_num
  }

end ceil_floor_sum_l86_86180


namespace tourists_left_l86_86534

theorem tourists_left (initial_tourists eaten_by_anacondas poisoned_fraction recover_fraction : ℕ) 
(h_initial : initial_tourists = 30) 
(h_eaten : eaten_by_anacondas = 2)
(h_poisoned_fraction : poisoned_fraction = 2)
(h_recover_fraction : recover_fraction = 7) :
  initial_tourists - eaten_by_anacondas - (initial_tourists - eaten_by_anacondas) / poisoned_fraction + (initial_tourists - eaten_by_anacondas) / poisoned_fraction / recover_fraction = 16 :=
by
  sorry

end tourists_left_l86_86534


namespace proportion_correct_l86_86683

theorem proportion_correct (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
sorry

end proportion_correct_l86_86683


namespace largest_prime_factor_9879_l86_86986

noncomputable def prime_factors (n : ℕ) := {
  factors : list ℕ // (∀ p, p ∈ factors → nat.prime p) ∧ factors.product = n }

theorem largest_prime_factor_9879 :
  ∃ (p : ℕ), nat.prime p ∧ p = 89 ∧ ∀ q ∈ prime_factors 9879, q ≤ p :=
by {
  sorry
}

end largest_prime_factor_9879_l86_86986


namespace symmetric_circle_eq_l86_86614

theorem symmetric_circle_eq :
  ∀ (x y : ℝ),
  ((x + 2)^2 + y^2 = 5) →
  (x - y + 1 = 0) →
  (∃ (a b : ℝ), ((a + 1)^2 + (b + 1)^2 = 5)) := 
by
  intros x y h_circle h_line
  -- skip the proof
  sorry

end symmetric_circle_eq_l86_86614


namespace balls_probability_l86_86097

theorem balls_probability:
  let balls := List.range 1 14,
  let odd_balls := [1, 3, 5, 7, 9, 11, 13],
  let even_balls := [2, 4, 6, 8, 10, 12],
  let total_ways := Nat.choose 13 7,
  let favorable_ways := (Nat.choose 7 1 * Nat.choose 6 6) +
                        (Nat.choose 7 3 * Nat.choose 6 4) +
                        (Nat.choose 7 5 * Nat.choose 6 2) +
                        (Nat.choose 7 7 * Nat.choose 6 0),
  let probability := favorable_ways.toRat / total_ways.toRat
  in probability = 212 / 429 := sorry

end balls_probability_l86_86097


namespace magnitude_a_plus_3b_sine_of_angle_l86_86763

open Real

section Vector_Proofs
variables (a b : ℝ^3) -- Vectors a and b in R^3
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) -- Magnitude conditions
variables (hcond : ‖3 • a - b‖ = sqrt 5) -- Given condition

-- Part 1: Prove the magnitude of a + 3b is √15
theorem magnitude_a_plus_3b : ‖a + 3 • b‖ = sqrt 15 := sorry

-- Part 2: Prove the sine of the angle between (3a - b) and (a + 3b) is √33/9
theorem sine_of_angle : 
  let x := 3 • a - b,
      y := a + 3 • b,
      cos_theta := (x ⬝ y) / (‖x‖ * ‖y‖)
  in sin (acos cos_theta) = sqrt 33 / 9 := sorry
end Vector_Proofs

end magnitude_a_plus_3b_sine_of_angle_l86_86763


namespace least_four_digit_13_heavy_l86_86917

def is_13_heavy (n : ℕ) : Prop := n % 13 > 8

theorem least_four_digit_13_heavy : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ is_13_heavy n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ is_13_heavy m → n ≤ m :=
by
  use 1004
  split
  . exact Nat.le_refl 1004
  split
  . exact Nat.lt_of_sub_one_le (by norm_num)
  split
  . show is_13_heavy 1004
    calc 1004 % 13 = 9 := by norm_num
    show 9 > 8
    exact Nat.lt_of_sub_one_le (by norm_num)
  sorry

end least_four_digit_13_heavy_l86_86917


namespace min_seats_to_occupy_l86_86475

theorem min_seats_to_occupy (n : ℕ) (h_n : n = 150) : 
  ∃ (k : ℕ), k = 90 ∧ ∀ m : ℕ, m ≥ k → ∀ i : ℕ, i < n → ∃ j : ℕ, (j < n) ∧ ((j = i + 1) ∨ (j = i - 1)) :=
sorry

end min_seats_to_occupy_l86_86475


namespace zyx_syndrome_diagnosis_l86_86911

noncomputable def total_patients : ℕ := 52
noncomputable def female_percentage : ℚ := 0.60
noncomputable def male_percentage : ℚ := 1 - female_percentage
noncomputable def female_patients : ℕ := (female_percentage * total_patients).toNat
noncomputable def male_patients : ℕ := total_patients - female_patients
noncomputable def female_syndrome_rate : ℚ := 0.20
noncomputable def male_syndrome_rate : ℚ := 0.30
noncomputable def female_syndrome_patients : ℕ := (female_syndrome_rate * female_patients).toNat
noncomputable def male_syndrome_patients : ℕ := (male_syndrome_rate * male_patients).toNat
noncomputable def total_syndrome_patients : ℕ := female_syndrome_patients + male_syndrome_patients
noncomputable def diagnosis_accuracy : ℚ := 0.75
noncomputable def diagnosed_patients : ℕ := (diagnosis_accuracy * total_syndrome_patients).toNat

theorem zyx_syndrome_diagnosis : diagnosed_patients = 9 := by
  sorry

end zyx_syndrome_diagnosis_l86_86911


namespace tourists_left_l86_86532

noncomputable def tourists_remaining {initial remaining poisoned recovered : ℕ} 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : ℕ :=
  remaining - poisoned + recovered

theorem tourists_left 
  (initial remaining poisoned recovered : ℕ) 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : tourists_remaining h1 h2 h3 h4 h5 h6 = 16 :=
  by
  sorry

end tourists_left_l86_86532


namespace find_b_for_continuity_l86_86383

-- Define the piecewise function f
def f (b : ℝ) (x : ℝ) : ℝ :=
if x ≤ 3 then 3 * x^2 - 2 else b * x + 5

-- Define the statement that f is continuous at x = 3
theorem find_b_for_continuity (b : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ (x : ℝ), |x - 3| < δ → |f b x - f b 3| < ε) →
  b = 20 / 3 :=
by
  sorry

end find_b_for_continuity_l86_86383


namespace odd_function_a_increasing_function_a_l86_86274

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem odd_function_a (a : ℝ) :
  (∀ x : ℝ, f (-x) a = - (f x a)) → a = -1 :=
by sorry

theorem increasing_function_a (a : ℝ) :
  (∀ x : ℝ, (Real.exp x - a * Real.exp (-x)) ≥ 0) → a ∈ Set.Iic 0 :=
by sorry

end odd_function_a_increasing_function_a_l86_86274


namespace triangle_area_l86_86863

theorem triangle_area {A B C L : Type*} {dist : A → A → ℝ} 
  (h1 : ∃ (d : A → A → ℝ), 
          d A B = d A L + d L B ∧ 
          d A C = d A L + d L C ∧ 
          d B C = d B A + d A C) 
  (hAL : dist A L = 4)
  (hBL : dist B L = 2 * Real.sqrt 15)
  (hCL : dist C L = 5) :
  ∃ (area : ℝ), 
    area = (9 * Real.sqrt 231) / 4 :=
by
  sorry

end triangle_area_l86_86863


namespace triangle_arithmetic_sequence_area_d_1_T_6_l86_86031

theorem triangle_arithmetic_sequence_area_d_1_T_6 :
  ∃ a b c : ℝ, (a, b, c) = (3, 4, 5) ∧
    ( ∃ α β γ : ℝ, α ≈ 36.86989764584 ∧ β ≈ 53.13010235416 ∧ γ = 90.0 ) :=
begin
  sorry
end

end triangle_arithmetic_sequence_area_d_1_T_6_l86_86031


namespace smallest_set_size_l86_86545

noncomputable def point : Type := (ℝ × ℝ)

def symmetric_about_origin (S : set point) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S → (-x, -y) ∈ S

def symmetric_about_x_axis (S : set point) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S → (x, -y) ∈ S

def symmetric_about_y_axis (S : set point) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S → (-x, y) ∈ S

def symmetric_about_line_y_eq_x (S : set point) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S → (y, x) ∈ S

def symmetries (S : set point) : Prop :=
  symmetric_about_origin S ∧
  symmetric_about_x_axis S ∧
  symmetric_about_y_axis S ∧
  symmetric_about_line_y_eq_x S

def minimum_points_in_S (S : set point) : Prop :=
  (2, 3) ∈ S ∧ symmetries S ∧ S.card = 8

theorem smallest_set_size :
  ∃ S : set point, minimum_points_in_S S := sorry

end smallest_set_size_l86_86545


namespace projection_of_a_in_direction_of_b_l86_86672

def collinear (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

def projection (a b : ℝ × ℝ) : ℝ := 
  let dot_product := a.1 * b.1 + a.2 * b.2
  let b_norm_squared := b.1 ^ 2 + b.2 ^ 2
  if b_norm_squared = 0 then 0 else dot_product / b_norm_squared

theorem projection_of_a_in_direction_of_b :
  ∀ (t : ℝ),
    collinear (3, 4) (t, -6) →
    projection (3, 4) (t, -6) = -5 :=
by
  intros t h
  unfold collinear at h
  unfold projection
  sorry

end projection_of_a_in_direction_of_b_l86_86672


namespace percent_dimes_value_is_60_l86_86073

variable (nickels dimes : ℕ)
variable (value_nickel value_dime : ℕ)
variable (num_nickels num_dimes : ℕ)

def total_value (n d : ℕ) (v_n v_d : ℕ) := n * v_n + d * v_d

def percent_value_dimes (total d_value : ℕ) := (d_value * 100) / total

theorem percent_dimes_value_is_60 :
  num_nickels = 40 →
  num_dimes = 30 →
  value_nickel = 5 →
  value_dime = 10 →
  percent_value_dimes (total_value num_nickels num_dimes value_nickel value_dime) (num_dimes * value_dime) = 60 := 
by sorry

end percent_dimes_value_is_60_l86_86073


namespace additional_miles_proof_l86_86632

-- Define the distances
def distance_to_bakery : ℕ := 9
def distance_bakery_to_grandma : ℕ := 24
def distance_grandma_to_apartment : ℕ := 27

-- Define the total distances
def total_distance_with_bakery : ℕ := distance_to_bakery + distance_bakery_to_grandma + distance_grandma_to_apartment
def total_distance_without_bakery : ℕ := 2 * distance_grandma_to_apartment

-- Define the additional miles
def additional_miles_with_bakery : ℕ := total_distance_with_bakery - total_distance_without_bakery

-- Theorem statement
theorem additional_miles_proof : additional_miles_with_bakery = 6 :=
by {
  -- Here should be the proof, but we insert sorry to indicate it's skipped
  sorry
}

end additional_miles_proof_l86_86632


namespace chord_length_range_l86_86639

theorem chord_length_range {l : ℝ → ℝ → Prop} {r : ℝ}
  (h₁ : ∀ (x y : ℝ), l x y ↔ x + y - 1 = 0)
  (h₂ : r > 0)
  (h₃ : ∃ (p q : ℝ), (x^2 + y^2 = r^2 → p + q - 1 = 0) ∧ (√14 = abs (p - q)))
  (M N : ℝ → ℝ)
  (h₄ : ∀ (x y : ℝ), (x, y) ∈ M ↔ x^2 + y^2 = r^2)
  (h₅ : ∀ (x y : ℝ), (x, y) ∈ N ↔ x^2 + y^2 = r^2)
  (P : ℝ → ℝ)
  (l'' : ℝ → ℝ → Prop)
  (h₆ : ∀ (x y : ℝ), l'' x y ↔ (1 + 2 * m) * x + (m - 1) * y - 3 * m = 0)
  (h₇ : ∀ (m : ℝ), (x, y) ∈ P  ↔ ∀ (M N  : ℝ → ℝ), Untested (x y) P ↔ Untested (x y) N )
  (h₈ : ∀ (PM PN : ℝ), PM = PN)
  :  (√6 - √2 ≤ ℝ ) 

Proof:
  qed.

end chord_length_range_l86_86639


namespace lines_region_solution_l86_86956

theorem lines_region_solution (h s : ℕ) (h_pos : h > 0) (s_pos : s > 0)
  (not_parallel : ∀ i j : ℕ, i ≠ j → i < s.succ → j < s.succ → ¬(are_parallel i j))
  (not_concurrent : ∀ i j k : ℕ, i ≠ j → i ≠ k → j ≠ k → i < (h + s).succ → j < (h + s).succ → k < (h + s).succ → ¬(are_concurrent i j k)) :
  (if h + s = 1992 then (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) else false) := sorry

def are_parallel : ℕ → ℕ → Prop := sorry
def are_concurrent : ℕ → ℕ → ℕ → Prop := sorry

end lines_region_solution_l86_86956


namespace cube_root_of_neg_eight_eq_neg_two_l86_86000

theorem cube_root_of_neg_eight_eq_neg_two : real.cbrt (-8) = -2 :=
by
  sorry

end cube_root_of_neg_eight_eq_neg_two_l86_86000


namespace ceil_floor_sum_l86_86185

theorem ceil_floor_sum : (Int.ceil (7 / 3 : ℝ) + Int.floor (- 7 / 3 : ℝ) = 0) := 
  by {
    -- referring to the proof steps
    have h1: Int.ceil (7 / 3 : ℝ) = 3,
    { sorry },
    have h2: Int.floor (- 7 / 3 : ℝ) = -3,
    { sorry },
    rw [h1, h2],
    norm_num
  }

end ceil_floor_sum_l86_86185


namespace count_N_less_than_2000_l86_86301

noncomputable def count_valid_N (N : ℕ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x < 2000 ∧ N < 2000 ∧ x ^ (⌊x⌋₊) = N

theorem count_N_less_than_2000 : 
  ∃ (count : ℕ), count = 412 ∧ 
  (∀ (N : ℕ), N < 2000 → (∃ x : ℝ, x > 0 ∧ x < 2000 ∧ x ^ (⌊x⌋₊) = N) ↔ N < count) :=
sorry

end count_N_less_than_2000_l86_86301


namespace number_of_boys_l86_86906

theorem number_of_boys (B G : ℕ) 
    (h1 : B + G = 345) 
    (h2 : G = B + 69) : B = 138 :=
by
  sorry

end number_of_boys_l86_86906


namespace irreducible_poly_f_l86_86752

variable {R : Type*} [CommRing R]

noncomputable def poly_f (a : ℕ → ℤ) (n : ℕ) : Polynomial ℤ :=
  ∏ i in Finset.range n, Polynomial.X - Polynomial.C (a i) - 1

theorem irreducible_poly_f (a : ℕ → ℤ) (n : ℕ)
  (h_distinct : Function.Injective a) :
  Irreducible (poly_f a n) :=
sorry

end irreducible_poly_f_l86_86752


namespace oranges_after_selling_l86_86365

-- Definitions derived from the conditions
def oranges_picked := 37
def oranges_sold := 10
def oranges_left := 27

-- The theorem to prove that Joan is left with 27 oranges
theorem oranges_after_selling (h : oranges_picked - oranges_sold = oranges_left) : oranges_left = 27 :=
by
  -- Proof omitted
  sorry

end oranges_after_selling_l86_86365


namespace ceil_floor_sum_l86_86178

theorem ceil_floor_sum : (Int.ceil (7 / 3 : ℝ) + Int.floor (- 7 / 3 : ℝ) = 0) := 
  by {
    -- referring to the proof steps
    have h1: Int.ceil (7 / 3 : ℝ) = 3,
    { sorry },
    have h2: Int.floor (- 7 / 3 : ℝ) = -3,
    { sorry },
    rw [h1, h2],
    norm_num
  }

end ceil_floor_sum_l86_86178


namespace porche_initial_time_l86_86783

theorem porche_initial_time (math_time english_time science_time history_time project_time : ℕ) 
    (h_math : math_time = 45) 
    (h_english : english_time = 30) 
    (h_science : science_time = 50) 
    (h_history : history_time = 25) 
    (h_project : project_time = 30) : 
    math_time + english_time + science_time + history_time + project_time = 180 := 
by
  rw [h_math, h_english, h_science, h_history, h_project]
  norm_num

end porche_initial_time_l86_86783


namespace expectation_and_variance_l86_86277

noncomputable theory
open_locale classical

variable {ξ : ℕ → ℕ}

def binomial_RNG (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
∑ x in finset.range (k+1), (nat.choose n x)*(p^x)*((1-p)^(n-x))

axiom ξ_distribution : ξ = λ k, classical.some (classical.indefinite_description _ (λ p, binomial_RNG 5 0.5 p = k))

noncomputable def η : ℕ → ℝ := λ k, 5 * ξ k

-- Theorem to prove
theorem expectation_and_variance (Eη : ℝ) (Dη : ℝ) : 
  Eη = 25 / 2 ∧ Dη = 125 / 4 :=
by
  sorry

end expectation_and_variance_l86_86277


namespace largest_sum_fraction_l86_86140

open Rat

theorem largest_sum_fraction :
  let a := (2:ℚ) / 5
  let c1 := (1:ℚ) / 6
  let c2 := (1:ℚ) / 3
  let c3 := (1:ℚ) / 7
  let c4 := (1:ℚ) / 8
  let c5 := (1:ℚ) / 9
  max (a + c1) (max (a + c2) (max (a + c3) (max (a + c4) (a + c5)))) = a + c2
  ∧ a + c2 = (11:ℚ) / 15 := by
  sorry

end largest_sum_fraction_l86_86140


namespace spherical_to_rectangular_and_distance_correct_l86_86952

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

def distance_from_origin (x y z : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 + z^2)

theorem spherical_to_rectangular_and_distance_correct :
  let ρ := 8
  let θ := 5 * Real.pi / 4
  let φ := Real.pi / 4
  let (x, y, z) := spherical_to_rectangular ρ θ φ
  x = -4 ∧ y = -4 ∧ z = 4 * Real.sqrt 2 ∧ distance_from_origin x y z = 8 := by
  sorry

end spherical_to_rectangular_and_distance_correct_l86_86952


namespace polynomial_roots_l86_86610

theorem polynomial_roots :
  (∀ x : ℝ, (x^3 - 2*x^2 - 5*x + 6 = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3)) :=
by
  sorry

end polynomial_roots_l86_86610


namespace count_positive_integers_N_number_of_N_l86_86292

theorem count_positive_integers_N : ∀ N : ℕ, N < 2000 → ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N :=
begin
  sorry
end

theorem number_of_N : {N : ℕ // N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N}.card = 412 :=
begin
  sorry
end

end count_positive_integers_N_number_of_N_l86_86292


namespace difficult_more_than_easy_by_20_l86_86841

/--
Three students solved 100 problems while preparing for an exam. 
Each student solved 60 problems, and every problem was solved by at least one student.
A problem is considered difficult if only one student solved it and easy if all three students solved it.
--/
theorem difficult_more_than_easy_by_20
  (problems : ℕ)
  (solved_by_student : ℕ)
  (x1 x2 x3 y12 y13 y23 z : ℕ) :
  -- Total problems solved
  problems = 100 →
  -- Each student solved 60 problems
  solved_by_student = 60 →
  -- System of equations as derived
  (x1 + x2 + x3 + y12 + y13 + y23 + z = problems) →
  (x1 + y12 + y13 + z = solved_by_student) →
  (x2 + y12 + y23 + z = solved_by_student) →
  (x3 + y13 + y23 + z = solved_by_student) →
  -- Prove the number of difficult problems (x1 + x2 + x3) is greater than easy problems (z) by 20
  (x1 + x2 + x3 - z = 20) :=
begin
  intros problems_eq solved_solved_by_student_eq eq1 eq2 eq3 eq4,
  -- completing the proof step as per the derived equations
  sorry
end

end difficult_more_than_easy_by_20_l86_86841


namespace false_disjunction_implies_both_false_l86_86328

theorem false_disjunction_implies_both_false (p q : Prop) (h : ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
sorry

end false_disjunction_implies_both_false_l86_86328


namespace total_messages_l86_86702

theorem total_messages (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  sorry

end total_messages_l86_86702


namespace games_played_l86_86567

theorem games_played : ∀ (t c g : ℕ), t = 45 → c = 9 → t / c = g → g = 5 :=
by
  intros t c g h_t h_c h_tc
  rw [h_t, h_c] at h_tc
  simp at h_tc
  exact h_tc

end games_played_l86_86567


namespace sum_of_cubes_of_roots_l86_86509

theorem sum_of_cubes_of_roots (x₁ x₂ : ℝ) (h₀ : 3 * x₁ ^ 2 - 5 * x₁ - 2 = 0)
  (h₁ : 3 * x₂ ^ 2 - 5 * x₂ - 2 = 0) :
  x₁^3 + x₂^3 = 215 / 27 :=
by sorry

end sum_of_cubes_of_roots_l86_86509


namespace sqrt_16_l86_86461

theorem sqrt_16 : {x : ℝ | x^2 = 16} = {4, -4} :=
by
  sorry

end sqrt_16_l86_86461


namespace exists_quadrilateral_with_obtuse_diagonal_division_l86_86597

theorem exists_quadrilateral_with_obtuse_diagonal_division :
  ∃ (Q : Type) [Quadrilateral Q], 
  (∀ (d : Diagonal Q), ∀ T₁ T₂, (divides_into_triangles d T₁ T₂) → (obtuse_triangulated T₁ ∧ obtuse_triangulated T₂)) :=
sorry

end exists_quadrilateral_with_obtuse_diagonal_division_l86_86597


namespace maximum_minimum_product_of_any_two_numbers_l86_86381

theorem maximum_minimum_product_of_any_two_numbers :
  ∃ a : Fin 2018 → ℝ, (∑ i, a i = 0) ∧ (∑ i, (a i)^2 = 2018) ∧ ∀ b : Fin 2018 → ℝ, (∑ i, b i = 0) ∧ (∑ i, (b i)^2 = 2018) → (∀ i j, b i * b j ≥ a i * a j) :=
sorry

end maximum_minimum_product_of_any_two_numbers_l86_86381


namespace A_beats_B_by_42_42_meters_l86_86698

-- Definitions based on conditions
def speed_A : ℝ := 200 / 33
def time_A : ℝ := 33
def time_difference : ℝ := 7
def time_B : ℝ := time_A + time_difference
def distance_B_in_time_A : ℝ := speed_A * time_A

-- Theorem to prove the given problem's solution
theorem A_beats_B_by_42_42_meters :
  200 - distance_B_in_time_A ≈ 42.42 :=
by
  sorry

end A_beats_B_by_42_42_meters_l86_86698


namespace first_term_of_new_ratio_l86_86555

-- Given conditions as definitions
def original_ratio : ℚ := 6 / 7
def x (n : ℕ) : Prop := n ≥ 3

-- Prove that the first term of the ratio that the new ratio should be less than is 4
theorem first_term_of_new_ratio (n : ℕ) (h1 : x n) : ∃ b, (6 - n) / (7 - n) < 4 / b :=
by
  exists 5
  sorry

end first_term_of_new_ratio_l86_86555


namespace eval_expr_eq_zero_l86_86169

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l86_86169


namespace xiao_hong_mistake_l86_86510

theorem xiao_hong_mistake (a : ℕ) (h : 31 - a = 12) : 31 + a = 50 :=
by
  sorry

end xiao_hong_mistake_l86_86510


namespace min_distance_between_curves_l86_86243

theorem min_distance_between_curves :
  let P := (xP : ℝ) → (xP, xP^2 + 2)
  let Q := (xQ : ℝ) → (xQ, Real.sqrt(xQ - 2))
  ∃ xq, ∃ xp, (xq >= 2) → (xp ≥ 0) → 
    (|xP - xQ| + |xP^2 + 2 - Real.sqrt(xQ - 2)|) / Real.sqrt(2) = (7 * Real.sqrt(2)) / 4 :=
sorry

end min_distance_between_curves_l86_86243


namespace find_ratio_AM_MB_l86_86780

-- Define given conditions
variables {A B C D A₁ B₁ C₁ D₁ M N L K M₁ N₁ L₁ K₁ : Type*}
variables (α : ℝ) 
variables [cube ABCD A₁B₁C₁D₁] [on_edge M AB] [inscribed_rectangle MNLK ABCD M]
variables [orthogonal_projection M₁N₁L₁K₁ MNLK A₁B₁C₁D₁]
variables [angle_cosine MK₁L₁N_with_base α (sqrt (2 / 11))]

-- The theorem statement to prove the ratio
theorem find_ratio_AM_MB : AM / MB = 1 / 2 :=
sorry

end find_ratio_AM_MB_l86_86780


namespace four_digit_numbers_count_thousands_greater_than_hundreds_l86_86285

def valid_four_digit_numbers_count : ℕ :=
  let valid_pairs := ∑ k in Finset.range 10 \{0}, k -- sum of numbers from 1 to 9 (since 0 can't be in the thousands place)
  valid_pairs * 10 * 10 -- multiply by 10 for each possibilities in d2 and d1

theorem four_digit_numbers_count_thousands_greater_than_hundreds :
  valid_four_digit_numbers_count = 4500 :=
by
  sorry

end four_digit_numbers_count_thousands_greater_than_hundreds_l86_86285


namespace coin_toss_probability_4_heads_l86_86875

/--
A fair coin is tossed 5 times. Prove that the probability of getting exactly 4 heads is 0.15625.
-/
theorem coin_toss_probability_4_heads :
  let n := 5
      k := 4
      p := 0.5
      q := 0.5 - p in
  (nat.choose n k) * (p^k) * (q^(n-k)) = 0.15625 :=
by
  sorry

end coin_toss_probability_4_heads_l86_86875


namespace total_number_of_eyes_l86_86772

theorem total_number_of_eyes (n_spiders n_ants eyes_per_spider eyes_per_ant : ℕ)
  (h1 : n_spiders = 3) (h2 : n_ants = 50) (h3 : eyes_per_spider = 8) (h4 : eyes_per_ant = 2) :
  (n_spiders * eyes_per_spider + n_ants * eyes_per_ant) = 124 :=
by
  sorry

end total_number_of_eyes_l86_86772


namespace tangent_line_circle_p_l86_86023

theorem tangent_line_circle_p (p : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 6 * x + 8 = 0 → (x = -p/2 ∨ y = 0)) → 
  (p = 4 ∨ p = 8) :=
by
  sorry

end tangent_line_circle_p_l86_86023


namespace english_teachers_count_l86_86446

theorem english_teachers_count (E : ℕ) 
    (h_prob : 6 / ((E + 6) * (E + 5) / 2) = 1 / 12) : 
    E = 3 :=
by
  sorry

end english_teachers_count_l86_86446


namespace mary_total_nickels_l86_86389

def mary_initial_nickels : ℕ := 7
def mary_dad_nickels : ℕ := 5

theorem mary_total_nickels : mary_initial_nickels + mary_dad_nickels = 12 := by
  sorry

end mary_total_nickels_l86_86389


namespace problem_inequality_l86_86997

theorem problem_inequality (a b : ℝ) (hab : 1 / a + 1 / b = 1) : 
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) := 
by
  sorry

end problem_inequality_l86_86997


namespace ceiling_plus_floor_eq_zero_l86_86193

theorem ceiling_plus_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (-7 / 3)) = 0 := by
  sorry

end ceiling_plus_floor_eq_zero_l86_86193


namespace angle_44_45_3_l86_86377

-- Let B₁B₂B₃ be a right isosceles triangle such that ∠B₁B₂B₃ = 90°.
-- Define B_{n+3} as the midpoint of the hypotenuse B_nB_{n+1} for all n.
-- B₁ and B₂ are the endpoints of the hypotenuse.
-- The task is to prove that ∠B₄₄B₄₅B₃ = 90°.

noncomputable def angle_n (n : ℕ) : ℕ := (if n > 2 then 90 else 0) -- Placeholder: Assumes Bₙ₊₃ follows some pattern leading to 90°.

theorem angle_44_45_3 (B : ℕ → ℝ × ℝ) (h₁ : ∀ n, B (n + 3) = midpoint (B n) (B (n + 1))) (h₂ : B 1 = (0, 0)) (h₃ : B 2 = (1, 1))
  : angle_n 44 = 90 :=
by
  sorry

end angle_44_45_3_l86_86377


namespace pencils_left_l86_86934

def ashton_boxes : Nat := 3
def pencils_per_box : Nat := 14
def pencils_given_to_brother : Nat := 6
def pencils_given_to_friends : Nat := 12

theorem pencils_left (h₁ : ashton_boxes = 3) 
                     (h₂ : pencils_per_box = 14)
                     (h₃ : pencils_given_to_brother = 6)
                     (h₄ : pencils_given_to_friends = 12) :
  (ashton_boxes * pencils_per_box - pencils_given_to_brother - pencils_given_to_friends) = 24 :=
by
  sorry

end pencils_left_l86_86934


namespace cans_of_soup_feeds_adults_l86_86898

theorem cans_of_soup_feeds_adults (A B : ℕ) (C D : ℕ) (initial_cans used_cans remaining_cans : ℕ) :
  A = 3 →
  B = 5 →
  C = 5 →
  initial_cans = 5 →
  used_cans = C * B / A →
  remaining_cans = initial_cans - used_cans →
  D = 3 →
  remaining_cans * D = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h7, h5] at h6
  rw [nat.sub_eq_iff_eq_add] at h6
  simp [← nat.div_eq, nat.add_sub_right] at h6
  exact h6

end cans_of_soup_feeds_adults_l86_86898


namespace solve_for_x_l86_86332

theorem solve_for_x (x : ℤ) (h : 3 * x + 7 = -2) : x = -3 :=
by
  sorry

end solve_for_x_l86_86332


namespace biking_distance_differences_l86_86006

/-- Conditions: Constant speeds of the bikers. Bjorn: 45 miles in 4 hours, Alberto: 60 miles in 4 hours, Carlos: 75 miles in 4 hours. -/
constants (hours : ℤ) (miles_B : ℕ) (miles_A : ℕ) (miles_C : ℕ)

-- Given constant speeds and distances for 4 hours
def speed_B := miles_B / hours
def speed_A := miles_A / hours
def speed_C := miles_C / hours

-- Time duration for comparison
def time_duration : ℤ := 6

-- Distances covered in 6 hours
def distance_B := speed_B * time_duration
def distance_A := speed_A * time_duration
def distance_C := speed_C * time_duration

-- Computation of the differences
def difference_AB := distance_A - distance_B
def difference_CB := distance_C - distance_B

/-- Proof problem: Alberto has biked 22.5 more miles than Bjorn, and Carlos has biked 45 more miles than Bjorn after six hours. -/
theorem biking_distance_differences
  (hours_nonzero : hours ≠ 0)
  (miles_B := 45)
  (miles_A := 60)
  (miles_C := 75) :
  difference_AB = 22.5 ∧ difference_CB = 45 := by
  sorry

end biking_distance_differences_l86_86006


namespace translation_of_graph_l86_86069

variable {k : ℝ}
variable {f : ℝ → ℝ}

theorem translation_of_graph (h : k > 0) :
  ∀ x, y = f(x + k) → translates_left f x k :=
begin
  sorry
end

end translation_of_graph_l86_86069


namespace smallest_square_factor_2016_l86_86859

theorem smallest_square_factor_2016 : ∃ n : ℕ, (168 = n) ∧ (∃ k : ℕ, k^2 = n) ∧ (2016 ∣ k^2) :=
by
  sorry

end smallest_square_factor_2016_l86_86859


namespace lindy_distance_travelled_l86_86360

noncomputable def distanceJackChristina := 240 -- feet
noncomputable def speedJack := 5 -- feet per second
noncomputable def speedChristina := 3 -- feet per second
noncomputable def speedLindy := 9 -- feet per second

theorem lindy_distance_travelled :
  let time_to_meet := distanceJackChristina / (speedJack + speedChristina) in
  let distance_lindy := speedLindy * time_to_meet in
  distance_lindy = 270 := by
  -- Using given conditions and the correct answer
  sorry

end lindy_distance_travelled_l86_86360


namespace factorize_polynomial_l86_86975

theorem factorize_polynomial (x y : ℝ) : 2 * x^3 - 8 * x^2 * y + 8 * x * y^2 = 2 * x * (x - 2 * y) ^ 2 := 
by sorry

end factorize_polynomial_l86_86975


namespace intersection_A_B_subsets_C_l86_86248

-- Definition of sets A and B
def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | 0 ≤ x}

-- Definition of intersection C
def C : Set ℤ := A ∩ B

-- The proof statements
theorem intersection_A_B : C = {1, 2} := 
by sorry

theorem subsets_C : {s | s ⊆ C} = {∅, {1}, {2}, {1, 2}} := 
by sorry

end intersection_A_B_subsets_C_l86_86248


namespace probability_target_hit_l86_86823

theorem probability_target_hit (P_A P_B : ℚ) (h1 : P_A = 1/2) (h2 : P_B = 1/3) : 
  (1 - (1 - P_A) * (1 - P_B)) = 2/3 :=
by
  sorry

end probability_target_hit_l86_86823


namespace trigonometric_expression_evaluation_l86_86651

variable (α : ℝ)
hypothesis (h : Real.tan α = 2)

theorem trigonometric_expression_evaluation :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := by
  sorry

end trigonometric_expression_evaluation_l86_86651


namespace equal_volume_division_of_tetrahedron_l86_86785

variables {Point : Type} [metric_space Point]
variables {Tetrahedron : Type} [has_coe (point*4) Tetrahedron]

def is_midpoint (M : Point) (A B : Point) : Prop :=
  dist A M = dist M B

theorem equal_volume_division_of_tetrahedron (A B C D M K : Point)
  (H_midpoint_M : is_midpoint M A B)
  (H_midpoint_K : is_midpoint K C D)
  (tetra : Tetrahedron := coe (A, B, C, D)) : 
  volume (tetra_division_by_plane_MK tetra M K) = volume tetra / 2 :=
by sorry

end equal_volume_division_of_tetrahedron_l86_86785


namespace Jessie_lost_7_kilograms_l86_86364

def Jessie_previous_weight : ℕ := 74
def Jessie_current_weight : ℕ := 67
def Jessie_weight_lost : ℕ := Jessie_previous_weight - Jessie_current_weight

theorem Jessie_lost_7_kilograms : Jessie_weight_lost = 7 :=
by
  sorry

end Jessie_lost_7_kilograms_l86_86364


namespace number_of_valid_Ns_l86_86308

noncomputable def count_valid_N : ℕ :=
  (finset.range 2000).filter (λ N, ∃ x : ℝ, x^floor x = N).card

theorem number_of_valid_Ns :
  count_valid_N = 1287 :=
sorry

end number_of_valid_Ns_l86_86308


namespace part_1_part_2_l86_86668

noncomputable theory

-- Definition of set A based on given condition
def setA : set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }

-- Definition of set B based on given condition
def setB : set ℝ := { x | x < 3 }

-- Definition of set C based on given equation
def setC (a : ℝ) : set ℝ := { x | x ≥ a + 1 }

-- First part: Prove A ∩ B = {x | 1 ≤ x < 3}
theorem part_1 : setA ∩ setB = { x | 1 ≤ x ∧ x < 3 } :=
sorry

-- Second part: Prove range of a
-- Given (A ∩ B) ⊆ C, find the range for a
theorem part_2 : (setA ∩ setB) ⊆ setC a → a ≤ 0 :=
sorry

end part_1_part_2_l86_86668


namespace num_valid_Ns_less_2000_l86_86312

theorem num_valid_Ns_less_2000 : 
  {N : ℕ | N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x^⟨floor x⟩ = N}.card = 412 := 
sorry

end num_valid_Ns_less_2000_l86_86312


namespace cube_root_of_neg_eight_l86_86003

theorem cube_root_of_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by
  sorry

end cube_root_of_neg_eight_l86_86003


namespace proportion_equation_l86_86686

theorem proportion_equation (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
by 
  sorry

end proportion_equation_l86_86686


namespace pentagon_ratio_l86_86244

noncomputable theory
open Classical

/--
Given a regular pentagon ABCDE with points K on side AE and L on side CD.
Let ∠LAE + ∠KCD = 108 degrees and the ratio AK / KE = 3 / 7.
Prove that the ratio CL / AB = 0.7.
-/
theorem pentagon_ratio (ABCDE : list (ℝ × ℝ))
  (reg_pentagon : ∀ (i j : ℕ), ABCDE.nth_le i h₁ = ABCDE.nth_le j h₂ -> dist (ABCDE.nth_le i h₃) (ABCDE.nth_le j h₄) = dist (ABCDE.nth_le (i + 1 % 5) h₅) (ABCDE.nth_le (j + 1 % 5) h₆))
  (hK_on_AE : ∃ K : ℝ × ℝ, on_segment (ABCDE.nth_le 0 h₇) (ABCDE.nth_le 4 h₈) K)
  (hL_on_CD : ∃ L : ℝ × ℝ, on_segment (ABCDE.nth_le 2 h₉) (ABCDE.nth_le 3 h₁₀) L)
  (h_angles : ∠ (ABCDE.nth_le 4 h₁₁) K (ABCDE.nth_le 0 h₁₂) + ∠ (ABCDE.nth_le 3 h₁₃) L (ABCDE.nth_le 2 h₁₄) = 108)
  (h_ratios : dist (ABCDE.nth_le 0 h₁₅) K / dist K (ABCDE.nth_le 4 h₁₆) = 3 / 7)
  : dist (ABCDE.nth_le 2 h₁₇) L / dist (ABCDE.nth_le 0 h₁₈) (ABCDE.nth_le 1 h₁₉) = 0.7 :=
sorry

end pentagon_ratio_l86_86244


namespace sqrt3_minus_2_pow_0_log2_sqrt2_l86_86582

theorem sqrt3_minus_2_pow_0_log2_sqrt2 :
  ( (real.sqrt 3 - 2)^0 - real.logb 2 (real.sqrt 2) ) = 1 / 2 := 
by
  sorry

end sqrt3_minus_2_pow_0_log2_sqrt2_l86_86582


namespace quadratic_coeff_is_five_l86_86146

theorem quadratic_coeff_is_five (x : ℝ) : 
  let eqn := (2 * x + 1) * (3 * x - 2) - x ^ 2 - 2 
  in (∀ a b c : ℝ, a * x^2 + b * x + c = 0 → a = 5) :=
by 
  sorry

end quadratic_coeff_is_five_l86_86146


namespace total_number_of_eyes_l86_86771

theorem total_number_of_eyes (n_spiders n_ants eyes_per_spider eyes_per_ant : ℕ)
  (h1 : n_spiders = 3) (h2 : n_ants = 50) (h3 : eyes_per_spider = 8) (h4 : eyes_per_ant = 2) :
  (n_spiders * eyes_per_spider + n_ants * eyes_per_ant) = 124 :=
by
  sorry

end total_number_of_eyes_l86_86771


namespace min_vans_proof_l86_86595

-- Define the capacity and availability of each type of van
def capacity_A : Nat := 7
def capacity_B : Nat := 9
def capacity_C : Nat := 12

def available_A : Nat := 3
def available_B : Nat := 4
def available_C : Nat := 2

-- Define the number of people going on the trip
def students : Nat := 40
def adults : Nat := 14

-- Define the total number of people
def total_people : Nat := students + adults

-- Define the minimum number of vans needed
def min_vans_needed : Nat := 6

-- Define the number of each type of van used
def vans_A_used : Nat := 0
def vans_B_used : Nat := 4
def vans_C_used : Nat := 2

-- Prove the minimum number of vans needed to accommodate everyone is 6
theorem min_vans_proof : min_vans_needed = 6 ∧ 
  (vans_A_used * capacity_A + vans_B_used * capacity_B + vans_C_used * capacity_C = total_people) ∧
  vans_A_used <= available_A ∧ vans_B_used <= available_B ∧ vans_C_used <= available_C :=
by 
  sorry

end min_vans_proof_l86_86595


namespace problem_statement_l86_86239

theorem problem_statement (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a + b + c + 2 = a * b * c) :
  (a+1) * (b+1) * (c+1) ≥ 27 ∧ ((a+1) * (b+1) * (c+1) = 27 → a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end problem_statement_l86_86239


namespace ratio_eq_275_l86_86794

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

noncomputable def choose (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

def number_of_cards := 60
def number_of_distinct_numbers := 12
def cards_per_number := 5
def number_of_drawn_cards := 5

def p := (number_of_distinct_numbers : ℝ) / choose number_of_cards number_of_drawn_cards

def q := ((number_of_distinct_numbers * (number_of_distinct_numbers - 1) * choose cards_per_number 4 * choose cards_per_number 1) : ℝ) /
          choose number_of_cards number_of_drawn_cards

def ratio := q / p

theorem ratio_eq_275 : ratio = 275 := by
  sorry

end ratio_eq_275_l86_86794


namespace larger_number_hcf_lcm_l86_86444

-- Definitions from conditions
def gcd (a b : ℕ) := -- Define gcd for clarity, although it's built into Lean
  if a = 0 then b else gcd (b % a) a

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b

variable (a b : ℕ)

theorem larger_number_hcf_lcm {hcf : ℕ} {factor1 : ℕ} {factor2 : ℕ}
  (hcf_eq : hcf = 23) (factor1_eq : factor1 = 13) (factor2_eq : factor2 = 18)
  (gcd_ab : gcd a b = hcf)
  (lcm_ab : lcm a b = hcf * factor1 * factor2) :
  max a b = 414 := sorry

end larger_number_hcf_lcm_l86_86444


namespace ceiling_floor_sum_l86_86157

theorem ceiling_floor_sum : (Int.ceil (7 / 3) + Int.floor (-(7 / 3)) = 0) := by
  sorry

end ceiling_floor_sum_l86_86157


namespace range_of_m_l86_86267

theorem range_of_m (m : ℝ) :
  (∃ x ∈ set.Icc (0 : ℝ) (Real.pi / 2), 
     ((Real.sin x + Real.cos x)^2 - 2 * (Real.cos x)^2 - m = 0)) →
  m ∈ set.Icc (-1 : ℝ) (Real.sqrt 2) := 
begin
  sorry
end

end range_of_m_l86_86267


namespace find_prime_c_l86_86526

open Nat

theorem find_prime_c (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) 
                      (hab : a + b = 49) (hbc : b + c = 60) : c = 13 := by
  sorry

end find_prime_c_l86_86526


namespace spot_accessible_area_l86_86799

-- Define the constants
def side_length := 1 -- yard
def rope_length := 4 -- yards

-- Define the result based on given conditions
def region_area : ℝ := (82 * Real.pi) / 3

-- Theorem statement
theorem spot_accessible_area (s : ℝ) (r : ℝ) (h1 : s = side_length) (h2 : r = rope_length) :
  let hex_base_area := (3 * sqrt 3 / 2) * s^2 in
  let sector_300_area := (300 / 360) * Real.pi * r^2 in
  let sector_60_area   := (1 / 6) * Real.pi * s^2 in
  sector_300_area + 4 * sector_60_area = region_area :=
by
  -- Begin block for the proof (included only to show the structure, filled with 'sorry')
  sorry

end spot_accessible_area_l86_86799


namespace slope_of_line_l86_86819

-- Definitions and conditions
def point_x := (-8 : ℝ, 0 : ℝ)
def area_triangle := 16

-- The proof problem statement
theorem slope_of_line {b : ℝ} (h_area : (1 / 2) * 8 * b = area_triangle) :
  let h := b in
  (h / 8) = (1 / 2) :=
sorry

end slope_of_line_l86_86819


namespace complex_real_axis_l86_86252

theorem complex_real_axis {a : ℝ} (h : ((1 + complex.i) * (a + complex.i)).im = 0) : a = -1 :=
sorry

end complex_real_axis_l86_86252


namespace proof_quotient_l86_86616

/-- Let x be in the form (a + b * sqrt c) / d -/
def x_form (a b c d : ℤ) (x : ℝ) : Prop := x = (a + b * Real.sqrt c) / d

/-- Main theorem -/
theorem proof_quotient (a b c d : ℤ) (x : ℝ) (h_eq : 4 * x / 5 + 2 = 5 / x) (h_form : x_form a b c d x) : (a * c * d) / b = -20 := by
  sorry

end proof_quotient_l86_86616


namespace organization_members_count_l86_86715

noncomputable theory
open_locale classical

/-- In an organization with five committees where each member belongs to exactly two 
different committees and each pair of committees has exactly one member in common, 
the total number of members is 10. -/
theorem organization_members_count (committees : Finset (Fin 5)) 
  (members : Type) (member_of : members → Finset (Fin 5)) :
  (∀ (m : members), (member_of m).card = 2) ∧
  (∀ (c1 c2 : Fin 5), c1 ≠ c2 → ∃! (m : members), c1 ∈ member_of m ∧ c2 ∈ member_of m) → 
  (∃ (n : ℕ), (card (members) = n) ∧ n = 10) :=
by
  intro h
  sorry

end organization_members_count_l86_86715


namespace joe_paint_left_after_third_week_l86_86366

def initial_paint : ℕ := 360

def paint_used_first_week (initial_paint : ℕ) : ℕ := initial_paint / 4

def paint_left_after_first_week (initial_paint : ℕ) : ℕ := initial_paint - paint_used_first_week initial_paint

def paint_used_second_week (paint_left_after_first_week : ℕ) : ℕ := paint_left_after_first_week / 2

def paint_left_after_second_week (paint_left_after_first_week : ℕ) : ℕ := paint_left_after_first_week - paint_used_second_week paint_left_after_first_week

def paint_used_third_week (paint_left_after_second_week : ℕ) : ℕ := paint_left_after_second_week * 2 / 3

def paint_left_after_third_week (paint_left_after_second_week : ℕ) : ℕ := paint_left_after_second_week - paint_used_third_week paint_left_after_second_week

theorem joe_paint_left_after_third_week : 
  paint_left_after_third_week (paint_left_after_second_week (paint_left_after_first_week initial_paint)) = 45 :=
by 
  sorry

end joe_paint_left_after_third_week_l86_86366


namespace total_amount_l86_86966

theorem total_amount (A B : ℕ) (h_ratio : A : B = 1 : 2) (h_A : A = 200) : A + B = 600 :=
by
  sorry

end total_amount_l86_86966


namespace sum_of_interior_angles_of_hexagon_l86_86464

theorem sum_of_interior_angles_of_hexagon : 
  let n := 6 in (n - 2) * 180 = 720 := 
by
  let n := 6
  show (n - 2) * 180 = 720
  sorry

end sum_of_interior_angles_of_hexagon_l86_86464


namespace speed_against_stream_l86_86109

theorem speed_against_stream (man_rate : ℝ) (with_stream_speed : ℝ) (S : ℝ)
  (h1 : man_rate = 6) 
  (h2 : with_stream_speed = 22) 
  (h3 : S = with_stream_speed - man_rate) :
  |(man_rate - S)| = 10 :=
by
  rw [h1, h2, h3]
  -- The steps would go here if we were to provide a complete proof
  sorry

end speed_against_stream_l86_86109


namespace length_of_AB_in_triangle_l86_86714

open Real

theorem length_of_AB_in_triangle
  (AC BC : ℝ)
  (area : ℝ) :
  AC = 4 →
  BC = 3 →
  area = 3 * sqrt 3 →
  ∃ AB : ℝ, AB = sqrt 13 :=
by
  sorry

end length_of_AB_in_triangle_l86_86714


namespace determine_N_l86_86958

theorem determine_N :
  ∃ N : ℕ, 0 < N ∧ 15^4 * 28^2 = 12^2 * N^2 :=
by {
  use 525,
  split,
  { exact nat.zero_lt_bit0 nat.one_ne_zero, }, -- 525 > 0
  { norm_num, }, -- 15^4 * 28^2 = 12^2 * 525^2
}

end determine_N_l86_86958


namespace right_triangle_integer_segments_l86_86414

theorem right_triangle_integer_segments (DE EF : ℕ) (hDE : DE = 24) (hEF : EF = 25) : 
  ∃ n : ℕ, ∀ P, 
  P ∈ segment E DF →
  (integer_length_segment E P) →
  n = 9 := 
sorry

end right_triangle_integer_segments_l86_86414


namespace johns_spent_amount_l86_86921

def original_price : ℝ := 2000
def increase_rate : ℝ := 0.02

theorem johns_spent_amount : 
  let increased_amount := original_price * increase_rate in
  let john_total := original_price + increased_amount in
  john_total = 2040 :=
by
  sorry

end johns_spent_amount_l86_86921


namespace opposite_side_of_3_on_die_is_4_l86_86432

theorem opposite_side_of_3_on_die_is_4 :
  ∀ (d : ℕ → ℕ), (d 1 = 6) ∧ (d 2 = 5) ∧ (d 3 = 4) ∧ (d 4 = 3) ∧ (d 5 = 2) ∧ (d 6 = 1) → d 3 = 4 :=
by
  intro d h
  cases h with h1 hrest
  cases hrest with h2 hrest'
  cases hrest' with h3 hrest''
  cases hrest'' with h4 hrest'''
  cases hrest''' with h5 h6
  exact h3

end opposite_side_of_3_on_die_is_4_l86_86432


namespace geometric_sequence_first_term_l86_86440

theorem geometric_sequence_first_term 
  (T : ℕ → ℝ) 
  (h1 : T 5 = 243) 
  (h2 : T 6 = 729) 
  (hr : ∃ r : ℝ, ∀ n : ℕ, T n = T 1 * r^(n - 1)) :
  T 1 = 3 :=
by
  sorry

end geometric_sequence_first_term_l86_86440


namespace dog_food_packages_l86_86128

theorem dog_food_packages
  (packages_cat_food : Nat := 9)
  (cans_per_package_cat_food : Nat := 10)
  (cans_per_package_dog_food : Nat := 5)
  (more_cans_cat_food : Nat := 55)
  (total_cans_cat_food : Nat := packages_cat_food * cans_per_package_cat_food)
  (total_cans_dog_food : Nat := d * cans_per_package_dog_food)
  (h : total_cans_cat_food = total_cans_dog_food + more_cans_cat_food) :
  d = 7 :=
by
  sorry

end dog_food_packages_l86_86128


namespace number_of_valid_Ns_l86_86309

noncomputable def count_valid_N : ℕ :=
  (finset.range 2000).filter (λ N, ∃ x : ℝ, x^floor x = N).card

theorem number_of_valid_Ns :
  count_valid_N = 1287 :=
sorry

end number_of_valid_Ns_l86_86309


namespace total_eyes_among_ninas_pet_insects_l86_86773

theorem total_eyes_among_ninas_pet_insects
  (num_spiders : ℕ) (num_ants : ℕ)
  (eyes_per_spider : ℕ) (eyes_per_ant : ℕ)
  (h_num_spiders : num_spiders = 3)
  (h_num_ants : num_ants = 50)
  (h_eyes_per_spider : eyes_per_spider = 8)
  (h_eyes_per_ant : eyes_per_ant = 2) :
  num_spiders * eyes_per_spider + num_ants * eyes_per_ant = 124 := 
by
  rw [h_num_spiders, h_num_ants, h_eyes_per_spider, h_eyes_per_ant]
  norm_num
  done

end total_eyes_among_ninas_pet_insects_l86_86773


namespace valid_grid_count_l86_86587

def is_adjacent (i j : ℕ) (n : ℕ) : Prop :=
  (i = j + 1 ∨ i + 1 = j ∨ (i = n - 1 ∧ j = 0) ∨ (i = 0 ∧ j = n - 1))

def valid_grid (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 4 ∧ 0 ≤ j ∧ j < 4 →
         (is_adjacent i (i+1) 4 → grid i (i+1) * grid i (i+1) = 0) ∧ 
         (is_adjacent j (j+1) 4 → grid (j+1) j * grid (j+1) j = 0)

theorem valid_grid_count : 
  ∃ s : ℕ, s = 1234 ∧
    (∃ grid : ℕ → ℕ → ℕ, valid_grid grid) :=
sorry

end valid_grid_count_l86_86587


namespace sufficient_but_not_necessary_l86_86764

theorem sufficient_but_not_necessary
  (m n : ℝ) 
  (f : ℝ → ℝ := λ x, |x^2 + m * x + n|) 
  (g : ℝ → ℝ := λ x, log (x^2 + m * x + n)) :
  (∃ a b : ℝ, a ≠ b ∧ (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b)) 
  → (∃ a b : ℝ, a ≠ b ∧ (∀ x : ℝ, g x ∈ set.univ)) 
  :=
begin
  -- Proof to be filled in (sufficient but not necessary)
  sorry
end

end sufficient_but_not_necessary_l86_86764


namespace inequality_1_inequality_2_l86_86088

variable {n : ℕ} (x : Fin n → ℝ) (m : Fin n → ℝ)

theorem inequality_1 (h : ∀ i, 0 < x i) (hne : ∃ i j, i ≠ j ∧ x i ≠ x j) :
  Real.sqrt (∑ i, (x i)^2 / n) > ∑ i, x i / n := by
  sorry

theorem inequality_2 (h : ∀ i, 0 < x i) (hm : ∀ i, 0 < m i) (hne : ∃ i j, i ≠ j ∧ x i ≠ x j) :
  let M := ∑ i, m i in
  Real.sqrt (∑ i, (m i) * (x i)^2 / M) > ∑ i, (m i) * (x i) / M := by
  sorry

end inequality_1_inequality_2_l86_86088


namespace parallel_lines_m_values_l86_86327

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 ↔ mx + 3 * y - 2 = 0) → (m = -3 ∨ m = 2) :=
by
  sorry

end parallel_lines_m_values_l86_86327


namespace tan_sum_simplification_l86_86420

theorem tan_sum_simplification :
  tan (Real.pi / 8) + tan (5 * Real.pi / 24) = 
  2 * sin (13 * Real.pi / 24) / sqrt ((2 + sqrt 2) * (2 + sqrt 3)) :=
sorry

end tan_sum_simplification_l86_86420


namespace num_odd_integers_between_3000_and_6000_l86_86680

-- Number of odd integers between 3000 and 6000 with four different digits
theorem num_odd_integers_between_3000_and_6000 : 
  (∃ (nums : Finset ℕ), 
    (∀ n ∈ nums, 3000 ≤ n ∧ n < 6000 ∧ (n % 2 = 1)) ∧ 
    (∀ n ∈ nums, (Nat.digits 10 n).Nodup) ∧ nums.card = 840) := 
begin
  -- at this point the proof steps would go here
  sorry
end

end num_odd_integers_between_3000_and_6000_l86_86680


namespace probability_units_digit_one_l86_86111

/-- 
Given:
1. m is randomly selected from the set {11, 13, 17, 19, 23}
2. n is randomly selected from the range {2001, 2002, ..., 2020}

Prove that the probability that m^n has a units digit of 1 is 7/20.
-/
theorem probability_units_digit_one :
  let m_set := {11, 13, 17, 19, 23}
  let n_range := finset.Icc 2001 2020
  let event := { x ∈ m_set | ∃ y ∈ n_range, (x^y) % 10 = 1 }
  (event.card : ℚ) / (m_set.card * n_range.card) = 7 / 20 :=
sorry

end probability_units_digit_one_l86_86111


namespace fan_airflow_weekly_l86_86905

def fan_airflow_per_second : ℕ := 10
def fan_work_minutes_per_day : ℕ := 10
def minutes_to_seconds (m : ℕ) : ℕ := m * 60
def days_per_week : ℕ := 7

theorem fan_airflow_weekly : 
  (fan_airflow_per_second * (minutes_to_seconds fan_work_minutes_per_day) * days_per_week) = 42000 := 
by
  sorry

end fan_airflow_weekly_l86_86905


namespace sin_half_angle_product_le_cos_half_angle_product_le_cos_sum_le_l86_86730

theorem sin_half_angle_product_le (A B C : ℝ) (h : A + B + C = π) : 
  sin (A / 2) * sin (B / 2) * sin (C / 2) ≤ 1 / 8 := 
sorry

theorem cos_half_angle_product_le (A B C : ℝ) (h : A + B + C = π) : 
  cos (A / 2) * cos (B / 2) * cos (C / 2) ≤ 3 * real.sqrt 3 / 8 := 
sorry

theorem cos_sum_le (A B C : ℝ) (h : A + B + C = π) : 
  cos A + cos B + cos C ≤ 3 / 2 := 
sorry

end sin_half_angle_product_le_cos_half_angle_product_le_cos_sum_le_l86_86730


namespace find_possible_values_l86_86607

theorem find_possible_values (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_m_pos : 0 < m) (h_n_pos : 0 < n):
  ∃ k : ℕ, k = 1 ∨ k = 5 ∧ k = (m^2 + 20 * m * n + n^2) / (m^3 + n^3) :=
by {
  -- Since the problem asks for the Lean statement only, the proof is not included.
  sorry
}

end find_possible_values_l86_86607


namespace percentage_to_pass_l86_86767

theorem percentage_to_pass (score shortfall max_marks : ℕ) (h_score : score = 212) (h_shortfall : shortfall = 13) (h_max_marks : max_marks = 750) :
  (score + shortfall) / max_marks * 100 = 30 :=
by
  sorry

end percentage_to_pass_l86_86767


namespace binom_square_mod_p_sum_binom_powers_mod_p_l86_86754

theorem binom_square_mod_p (p : ℕ) (prime_p : nat.prime p) (k : ℕ) (h : k < p) :
  (binom (p-1) k)^2 - 1 ≡ 0 [MOD p] := sorry

theorem sum_binom_powers_mod_p (p : ℕ) (prime_p : nat.prime p) (s : ℕ) :
  (∑ k in finset.range p, (binom (p-1) k)^s) ≡
    if even s then 0 else 1 [MOD p] := sorry

end binom_square_mod_p_sum_binom_powers_mod_p_l86_86754


namespace fraction_of_quarters_l86_86971

-- Conditions as definitions
def total_quarters : ℕ := 30
def states_between_1790_1809 : ℕ := 18

-- Goal theorem to prove 
theorem fraction_of_quarters : (states_between_1790_1809 : ℚ) / (total_quarters : ℚ) = 3 / 5 :=
by 
  sorry

end fraction_of_quarters_l86_86971


namespace ceiling_plus_floor_eq_zero_l86_86190

theorem ceiling_plus_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (-7 / 3)) = 0 := by
  sorry

end ceiling_plus_floor_eq_zero_l86_86190


namespace sum_of_other_digits_l86_86027

theorem sum_of_other_digits (h : ℕ) (n : ℕ) (h_val : h = 1) (div_by_9: ∑ d in [7, 6, h, 4], d % 9 = 0) : 
  (7 + 1 + 4) = 12 := by
sorry

end sum_of_other_digits_l86_86027


namespace polynomial_remainder_l86_86659

theorem polynomial_remainder (c a b : ℤ) 
  (h1 : (16 * c + 8 * a + 2 * b = -12)) 
  (h2 : (81 * c - 27 * a - 3 * b = -85)) : 
  (a, b, c) = (5, 7, 1) :=
sorry

end polynomial_remainder_l86_86659


namespace isolate_urea_decomposing_bacteria_valid_option_l86_86070

variable (KH2PO4 Na2HPO4 MgSO4_7H2O urea glucose agar water : Type)
variable (urea_decomposing_bacteria : Type)
variable (CarbonSource : Type → Prop)
variable (NitrogenSource : Type → Prop)
variable (InorganicSalt : Type → Prop)
variable (bacteria_can_synthesize_urease : urea_decomposing_bacteria → Prop)

axiom KH2PO4_is_inorganic_salt : InorganicSalt KH2PO4
axiom Na2HPO4_is_inorganic_salt : InorganicSalt Na2HPO4
axiom MgSO4_7H2O_is_inorganic_salt : InorganicSalt MgSO4_7H2O
axiom urea_is_nitrogen_source : NitrogenSource urea

theorem isolate_urea_decomposing_bacteria_valid_option :
  (InorganicSalt KH2PO4) ∧
  (InorganicSalt Na2HPO4) ∧
  (InorganicSalt MgSO4_7H2O) ∧
  (NitrogenSource urea) ∧
  (CarbonSource glucose) → (∃ bacteria : urea_decomposing_bacteria, bacteria_can_synthesize_urease bacteria) := sorry

end isolate_urea_decomposing_bacteria_valid_option_l86_86070


namespace count_N_less_than_2000_l86_86304

noncomputable def count_valid_N (N : ℕ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x < 2000 ∧ N < 2000 ∧ x ^ (⌊x⌋₊) = N

theorem count_N_less_than_2000 : 
  ∃ (count : ℕ), count = 412 ∧ 
  (∀ (N : ℕ), N < 2000 → (∃ x : ℝ, x > 0 ∧ x < 2000 ∧ x ^ (⌊x⌋₊) = N) ↔ N < count) :=
sorry

end count_N_less_than_2000_l86_86304


namespace common_chord_line_l86_86009

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 3 = 0

-- Definition of the line equation for the common chord
def line (x y : ℝ) : Prop := 2*x - 2*y + 7 = 0

theorem common_chord_line (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : line x y :=
by
  sorry

end common_chord_line_l86_86009


namespace max_area_polygon_is_equilateral_triangle_l86_86560

-- Definitions of the conditions
variable (P : Type) [polygon P]
variable (a : ℝ)
variable (is_edge_of_length_a : ∃ e, e ∈ edges P ∧ length e = a)
variable (exterior_angles_sum : exterior_angles_sum (vertices_not_adjacent_to_edge P a) = 120)

-- Statement to be proven
theorem max_area_polygon_is_equilateral_triangle:
  ∃ P' : Type, [polygon P'] ∧ is_equilateral_triangle P' a ∧ (∀ P'' : Type, [polygon P''],
  (is_polygon_with_conditions P'' a 120) → (area P' ≥ area P'')) :=
sorry

-- Here we define what it means for a polygon to satisfy the given conditions
def is_polygon_with_conditions (P'' : Type) [polygon P''] (a : ℝ) (angle_sum : ℝ) : Prop :=
  ∃ e, e ∈ edges P'' ∧ length e = a ∧ (exterior_angles_sum (vertices_not_adjacent_to_edge P'' a) = angle_sum)

-- And we define what it means for a polygon to be an equilateral triangle with a given side length
def is_equilateral_triangle (T : Type) [triangle T] (a : ℝ) : Prop :=
  ∀ (e : edges T) , length e = a

end max_area_polygon_is_equilateral_triangle_l86_86560


namespace constant_term_expansion_l86_86957

open Polynomial

-- Define the binomial expression and the constant term calculation
def binomial_expr (x : ℝ) : ℝ := (x - (1 / (2 * x)))^6
def general_term (r : ℕ) (x : ℝ) : ℝ := (-(1 / 2))^r * (nat.choose 6 r) * x^(6 - 2 * r)
def constant_term : ℝ := ∑ r in finset.range 7, if 6 - 2 * r = 0 then general_term r 1 else 0

theorem constant_term_expansion : 
  constant_term = -5 / 2 :=
by sorry

end constant_term_expansion_l86_86957


namespace maximize_rectangular_box_volume_l86_86478

theorem maximize_rectangular_box_volume :
  ∃ x : ℝ, 0 < x ∧ x = 1 ∧
  ∀ (h : 0 < x ∧ x < 3 / 2), 
    let length := 2 * x
    let height := (18 - 4 * length) / 4 
    in ((x * length * height = (x * 2 * x * ((9 / 2) - 3 * x))
    ∧ x = 1))
:=
by
  sorry

end maximize_rectangular_box_volume_l86_86478


namespace eval_ceil_floor_l86_86173

-- Definitions of the ceiling and floor functions and their relevant properties
def ceil (x : ℝ) : ℤ := ⌈x⌉
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Mathematical statement to prove
theorem eval_ceil_floor : ceil (7 / 3) + floor (- (7 / 3)) = 0 :=
by
  -- Proof is omitted
  sorry

end eval_ceil_floor_l86_86173


namespace eval_ceil_floor_l86_86175

-- Definitions of the ceiling and floor functions and their relevant properties
def ceil (x : ℝ) : ℤ := ⌈x⌉
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Mathematical statement to prove
theorem eval_ceil_floor : ceil (7 / 3) + floor (- (7 / 3)) = 0 :=
by
  -- Proof is omitted
  sorry

end eval_ceil_floor_l86_86175


namespace bk_distance_dk_distance_l86_86433

variables (A B C D K : Type) [square : is_square A B C D]
variables [hypotenuse : coincides_hypotenuse AC (triangle A C K)] 
variables [same_side : same_side_of_line B K AC]

theorem bk_distance : 
  BK = (|AK - CK|) / (sqrt 2) :=
sorry

theorem dk_distance : 
  DK = (AK + CK) / (sqrt 2) :=
sorry

end bk_distance_dk_distance_l86_86433


namespace fabric_woven_in_30_days_l86_86803

theorem fabric_woven_in_30_days :
  let a1 := 5
  let d := 16 / 29
  (30 * a1 + (30 * (30 - 1) / 2) * d) = 390 :=
by
  let a1 := 5
  let d := 16 / 29
  sorry

end fabric_woven_in_30_days_l86_86803


namespace determinant_nonnegative_of_skew_symmetric_matrix_l86_86417

theorem determinant_nonnegative_of_skew_symmetric_matrix
  (a b c d e f : ℝ)
  (A : Matrix (Fin 4) (Fin 4) ℝ)
  (hA : A = ![
    ![0, a, b, c],
    ![-a, 0, d, e],
    ![-b, -d, 0, f],
    ![-c, -e, -f, 0]]) :
  0 ≤ Matrix.det A := by
  sorry

end determinant_nonnegative_of_skew_symmetric_matrix_l86_86417


namespace dima_and_serezha_meet_time_l86_86963

-- Define the conditions and the main theorem to be proven.
theorem dima_and_serezha_meet_time :
  let dima_run_time := 15 / 60.0 -- Dima runs for 15 minutes
  let dima_run_speed := 6.0 -- Dima's running speed is 6 km/h
  let serezha_boat_speed := 20.0 -- Serezha's boat speed is 20 km/h
  let serezha_boat_time := 30 / 60.0 -- Serezha's boat time is 30 minutes
  let common_run_speed := 6.0 -- Both run at 6 km/h towards each other
  let distance_to_meet := dima_run_speed * dima_run_time -- Distance Dima runs along the shore
  let total_time := distance_to_meet / (common_run_speed + common_run_speed) -- Time until they meet after parting
  total_time = 7.5 / 60.0 := -- 7.5 minutes converted to hours
sorry

end dima_and_serezha_meet_time_l86_86963


namespace completion_time_l86_86541

theorem completion_time (total_work : ℕ) (initial_num_men : ℕ) (initial_efficiency : ℝ)
  (new_num_men : ℕ) (new_efficiency : ℝ) :
  total_work = 12 ∧ initial_num_men = 4 ∧ initial_efficiency = 1.5 ∧
  new_num_men = 6 ∧ new_efficiency = 2.0 →
  total_work / (new_num_men * new_efficiency) = 1 :=
by
  sorry

end completion_time_l86_86541


namespace alice_current_age_l86_86598

theorem alice_current_age (a b : ℕ) 
  (h1 : a + 8 = 2 * (b + 8)) 
  (h2 : (a - 10) + (b - 10) = 21) : 
  a = 30 := 
by 
  sorry

end alice_current_age_l86_86598


namespace housewife_spends_700_l86_86913

noncomputable def reduced_expenditure : ℝ :=
  let P := 35 / 0.75 in
  let X := 175 / (P - 35) in
  35 * (X + 5)
  
theorem housewife_spends_700 :
  reduced_expenditure = 700 := by
  sorry

end housewife_spends_700_l86_86913


namespace find_sphere_radius_l86_86355

noncomputable def sphere_radius_in_tetrahedron (a : ℝ) : ℝ :=
  a * Real.sqrt (2 / 3) / (4 + 2 * Real.sqrt (2 / 3))

theorem find_sphere_radius
  (a : ℝ)
  (h₁ : a > 0)
  (h₂ : let r := sphere_radius_in_tetrahedron a in
    ∀ (s₁ s₂ : ℝ), s₁ = r → s₂ = r → s₁ * s₂ = r * r) :
  let r := sphere_radius_in_tetrahedron a in r = a * Real.sqrt (2 / 3) / (4 + 2 * Real.sqrt (2 / 3)) :=
by
  sorry

end find_sphere_radius_l86_86355


namespace tan_six_minus_tan_two_eq_zero_l86_86688

noncomputable def cos (x : ℝ) : ℝ := sorry
noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cot (x : ℝ) : ℝ := cos x / sin x
noncomputable def tan (x : ℝ) : ℝ := sin x / cos x

theorem tan_six_minus_tan_two_eq_zero (x : ℝ) (h : cos x * cot x = sin^2 x):
  (tan x)^6 - (tan x)^2 = 0 :=
by sorry

end tan_six_minus_tan_two_eq_zero_l86_86688


namespace eval_expr_eq_zero_l86_86168

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l86_86168


namespace ceiling_floor_sum_l86_86155

theorem ceiling_floor_sum : (Int.ceil (7 / 3) + Int.floor (-(7 / 3)) = 0) := by
  sorry

end ceiling_floor_sum_l86_86155


namespace quadratic_trinomial_neg_values_l86_86229

theorem quadratic_trinomial_neg_values (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 7 * x + 4 * a < 0) ↔ a < -7/4 := by
sorry

end quadratic_trinomial_neg_values_l86_86229


namespace part1_part2_l86_86673

variables (x : ℝ)

-- Definitions of vectors a and b
def vector_a : ℝ × ℝ := (x, -2)
def vector_b : ℝ × ℝ := (2, 4)

-- Proving x = -1 when vectors a and b are parallel
theorem part1 (h : vector_a x = (x, -2) ∧ vector_b = (2, 4)):
  (x / 2 = -1 / 2) → x = -1 :=
by
  intro h1,
  sorry

-- Proving x = 1 or x = -5 when |a + b| = sqrt 13
theorem part2 (h : vector_a x = (x, -2) ∧ vector_b = (2, 4)):
  (sqrt ((x + 2)^2 + 2^2) = sqrt 13) → x = 1 ∨ x = -5 :=
by
  intro h2,
  sorry

end part1_part2_l86_86673


namespace sum_of_interior_angles_of_hexagon_l86_86462

theorem sum_of_interior_angles_of_hexagon : 
  let n := 6 in (n - 2) * 180 = 720 := 
by
  let n := 6
  show (n - 2) * 180 = 720
  sorry

end sum_of_interior_angles_of_hexagon_l86_86462


namespace proof_lean_problem_l86_86347

open Real

noncomputable def problem_conditions (a b c : ℝ) (A B C : ℝ) (BD : ℝ) : Prop :=
  let condition_1 := c * sin ((A + C) / 2) = b * sin C
  let condition_2 := BD = 1
  let condition_3 := b = sqrt 3
  let condition_4 := BD = b * sin A
  let condition_5 := A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0
  condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5

noncomputable def find_B (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (h : problem_conditions a b c A B C BD) : Prop :=
  B = π / 3

noncomputable def find_perimeter (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (h : problem_conditions a b c A B C BD) : ℝ := 
  a + b + c = 3 + sqrt 3

theorem proof_lean_problem (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (h : problem_conditions a b c A B C BD) : 
  find_B a b c A B C BD h ∧ find_perimeter a b c A B C BD h :=
sorry

end proof_lean_problem_l86_86347


namespace evaluate_expr_at_125_l86_86603

theorem evaluate_expr_at_125 : 
  let x := 1.25 
  in (3 * x^2 - 8 * x + 2) * (4 * x - 5) = 0 := 
by 
  sorry

end evaluate_expr_at_125_l86_86603


namespace probability_five_distinct_numbers_l86_86067

theorem probability_five_distinct_numbers (d : Fin 6 → Fin 8) :
  (∃ (a b : Fin 6), a ≠ b ∧ d a = d b) ∧
  (∃ s : Finset (Fin 8), s.card = 5 ∧ (∀ x : Fin 6, d x ∈ s)) →
  ∃ p : ℚ, p = 315 / 409 :=
by
  -- Noncomputable necessary for division in rational numbers
  noncomputable def probability (n k : ℕ) : ℚ :=
    (Nat.choose 8 5 * Nat.choose 6 2 * Finset.univ.card! * (2 / 1)) / 8^6

  have h_prob : probability 8 5 = 315 / 409 := sorry
  exact ⟨315 / 409, h_prob⟩

end probability_five_distinct_numbers_l86_86067


namespace probability_area_less_than_circumference_l86_86539

theorem probability_area_less_than_circumference : 
  let d_values := {d : ℕ | 2 ≤ d ∧ d ≤ 16 ∧ d < 4} in
  let p_two := if 2 ∈ d_values then (1 / 8) * (1 / 8) else 0 in
  let p_three := if 3 ∈ d_values then 2 * (1 / 8) * (1 / 8) else 0 in
  let total_probability := p_two + p_three in
  total_probability = 3 / 64 := 
by
  sorry

end probability_area_less_than_circumference_l86_86539


namespace area_of_triangle_BEF_l86_86022

-- Define the problem conditions
variables (length width diagonal : ℝ) (E F G : ℝ)
-- Define the rectangle's dimensions
def ABCD_length := 8
def ABCD_width := 6

-- Define the diagonal calculation using Pythagorean theorem
def diagonal_AC := Real.sqrt (ABCD_length^2 + ABCD_width^2)

-- Define the condition that diagonal AC is divided into four equal segments
def segment_length := diagonal_AC / 4

-- Define the height of the triangle using its area and base
def triangle_ABC_area := (1 / 2) * ABCD_length * ABCD_width
def height_BC := (2 * triangle_ABC_area) / diagonal_AC

-- Define the area of triangle BEF where base is one segment length and height is height_BC
def area_BEF := (1 / 2) * segment_length * height_BC

-- Problem statement to prove the area of triangle BEF is 6 square inches
theorem area_of_triangle_BEF :
  area_BEF = 6 := by
  sorry

end area_of_triangle_BEF_l86_86022


namespace sum_evaluation_l86_86973

noncomputable def sum_to_nth_term (n : ℕ) : ℕ :=
  (finset.range n.succ).sum (λ k, (finset.range k.succ).sum (λ r, int.floor (r / (k : ℚ))))

noncomputable def find_N (limit : ℕ) : ℕ :=
  (finset.range limit.succ).find (λ N, N * (N + 1) / 2 > 2013) - 1

theorem sum_evaluation :
  sum_to_nth_term 2013 = 62 := 
sorry

end sum_evaluation_l86_86973


namespace find_integer_n_l86_86605

open Int

theorem find_integer_n (n a b : ℤ) :
  (4 * n + 1 = a^2) ∧ (9 * n + 1 = b^2) → n = 0 := by
sorry

end find_integer_n_l86_86605


namespace sequence_value_addition_l86_86278

/--
Given a sequence defined by: 
1 * 9 + 2 = 11,
12 * 9 + 3 = 111,
123 * 9 + 4 = 1111,
...
let ∆ = 12345, and O = 6.
Prove that ∆ + O = 12351.
-/
theorem sequence_value_addition :
  let ∆ := 12345
  let O := 6
  ∆ + O = 12351 := 
by
  -- Direct computation based on given constants
  sorry

end sequence_value_addition_l86_86278


namespace trajectory_of_P_range_of_x_Q_l86_86670

-- Define points M, N, and moving point P
structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := { x := -2, y := 0 }
def N : Point := { x := 2, y := 0 }

-- Define the distance function
def distance (P1 P2 : Point) : ℝ :=
  real.sqrt ((P1.x - P2.x)^2 + (P1.y - P2.y)^2)

-- Define the equation of the trajectory C
def trajectory_condition (P : Point) : Prop :=
  (distance P M + distance P N = 0) → (P.y^2 = -8 * P.x)

-- Define the range condition for point Q
def range_condition (k : ℝ) : Prop :=
  ∀ (P1 P2 : Point), P1.y^2 = -8 * P1.x ∧ P2.y^2 = -8 * P2.x →
  (P1 ≠ P2) ∧ (P1.y > 0 ∧ P2.y > 0) →
  P1.y + P2.y = -8 / k →
  let B := Point.mk (-8 / k + 2) (-8 / k) in
  let Q : Point := { x := -2 - 8 / k, y := 0 } in
  Q.x < -6

-- The Lean 4 statement for the proof problems
theorem trajectory_of_P (P : Point) (h : distance P M + distance P N = 0) : P.y^2 = -8 * P.x :=
sorry

theorem range_of_x_Q (k : ℝ) (hk : -1 < k ∧ k < 0) : range_condition k :=
sorry

end trajectory_of_P_range_of_x_Q_l86_86670


namespace max_g_on_interval_l86_86220

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_on_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3 :=
by
  sorry

end max_g_on_interval_l86_86220


namespace evaluate_expression_l86_86194

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l86_86194


namespace sum_first_2018_terms_l86_86245

-- Definitions from conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

def collinear (A B C O : ℝ^3) : Prop := 
  ∃ k : ℝ, A = k • B + (1 - k) • C

-- Given conditions
variables (a : ℕ → ℝ) (A B C O : ℝ^3)
hypothesis (h1 : is_arithmetic_sequence a)
hypothesis (h2 : O ≠ 0)
hypothesis (h3 : collinear A B C O)
hypothesis (h4 : A = a 3 • B + a 2016 • C)

-- The sum of the first n terms of the arithmetic sequence
def S (n : ℕ) := (n : ℝ) / 2 * (a 0 + a (n-1))

-- Mathematical equivalent proof problem
theorem sum_first_2018_terms :
  (a 3 + a 2016 = 1) → S a 2018 = 1009 := 
begin
  sorry
end

end sum_first_2018_terms_l86_86245


namespace triangle_area_proof_l86_86862

noncomputable def area_of_triangle_ABC : ℝ :=
  let AL := 4
  let BL := 2 * Real.sqrt 15
  let CL := 5
  let A := (0, 0) -- Just placeholders for vertices
  let B := (4, 0)
  let C := (3, 1) in -- Placeholder
  -- Assuming we have a function to compute area from given sides' lengths
  triangle_area AL BL CL = ⅑ * Real.sqrt 231 
  
theorem triangle_area_proof:
  ∀ AL BL CL : ℝ,
    AL = 4 → BL = 2 * Real.sqrt 15 → CL = 5 → 
    triangle_area AL BL CL = 9 * Real.sqrt 231 / 4 := 
begin
  intros,
  sorry
end

end triangle_area_proof_l86_86862


namespace exists_infinite_subset_with_equal_gcd_l86_86370

noncomputable def infinite_set_of_positive_integers 
   (k : ℕ) (A : Set ℕ) : Prop :=
   (∀ n ∈ A, ∃ ps : Finset ℕ, (∀ p ∈ ps, Nat.Prime p) ∧ (∏ i in ps, i = n) ∧ ps.card ≤ k) ∧
   (A.Infinite)

theorem exists_infinite_subset_with_equal_gcd
  (A : Set ℕ)
  (hA : infinite_set_of_positive_integers 1987 A) :
  ∃ (B : Set ℕ) (p : ℕ), B ⊆ A ∧ B.Infinite ∧ (∀ (x y ∈ B), x ≠ y → Nat.gcd x y = p) :=
sorry

end exists_infinite_subset_with_equal_gcd_l86_86370


namespace initial_ducks_count_l86_86936

theorem initial_ducks_count (D : ℕ) 
  (h1 : ∃ (G : ℕ), G = 2 * D - 10) 
  (h2 : ∃ (D_new : ℕ), D_new = D + 4) 
  (h3 : ∃ (G_new : ℕ), G_new = 2 * D - 20) 
  (h4 : ∀ (D_new G_new : ℕ), G_new = D_new + 1) : 
  D = 25 := by
  sorry

end initial_ducks_count_l86_86936


namespace inequality_system_solution_exists_l86_86326

theorem inequality_system_solution_exists (a : ℝ) : (∃ x : ℝ, x ≥ -1 ∧ 2 * x < a) ↔ a > -2 := 
sorry

end inequality_system_solution_exists_l86_86326


namespace additional_pass_combinations_l86_86039

def original_combinations : ℕ := 4 * 2 * 3 * 3
def new_combinations : ℕ := 6 * 2 * 4 * 3
def additional_combinations : ℕ := new_combinations - original_combinations

theorem additional_pass_combinations : additional_combinations = 72 := by
  sorry

end additional_pass_combinations_l86_86039


namespace price_of_basic_computer_l86_86879

theorem price_of_basic_computer (C P : ℝ) 
    (h1 : C + P = 2500) 
    (h2 : P = (1/8) * (C + 500 + P)) :
    C = 2125 :=
by
  sorry

end price_of_basic_computer_l86_86879


namespace max_pieces_l86_86231

theorem max_pieces (plywood_width plywood_height piece_width piece_height : ℕ)
  (h_plywood : plywood_width = 22) (h_plywood_height : plywood_height = 15)
  (h_piece : piece_width = 3) (h_piece_height : piece_height = 5) :
  (plywood_width * plywood_height) / (piece_width * piece_height) = 22 := by
  sorry

end max_pieces_l86_86231


namespace distance_eq_expression_l86_86103

noncomputable def distance_expression (side_length : ℝ) (height1 height2 height3 : ℝ) : ℝ :=
  let a := (height1 - height3) / side_length
  let b := (height2 - height3) / side_length
  let c := (height1 + height2 - 2 * height3) / side_length
  let d := height3 - (side_length / 2)
  d

noncomputable def find_r_s_t (side_length height1 height2 height3 : ℝ) : ℕ × ℕ × ℕ :=
  let dist := distance_expression side_length height1 height2 height3
  let r := ⌊dist * 5⌋.natAbs
  let s := ⌊(dist * 5 - 70) ^ 2⌋.natAbs
  (r, s, 5)

theorem distance_eq_expression (r s t : ℕ) (hrst: r + s + t = 105) :
  ∃ (side_length height1 height2 height3 : ℝ),
    side_length = 12 ∧ height1 = 13 ∧ height2 = 14 ∧ height3 = 15 ∧
    distance_expression side_length height1 height2 height3 = (r - Real.sqrt s) / t :=
  by
    use [12, 13, 14, 15]
    unfold distance_expression
    sorry

end distance_eq_expression_l86_86103


namespace percentage_spent_on_food_l86_86775

-- Definitions based on conditions
variables {T : ℝ} -- Total amount spent
def spent_on_clothing := 0.50 * T
def spent_on_food (x : ℝ) := (x / 100) * T
def spent_on_other_items := 0.30 * T

def tax_on_clothing := 0.05 * spent_on_clothing
def tax_on_food (x : ℝ) := 0 * spent_on_food x
def tax_on_other_items := 0.10 * spent_on_other_items
def total_tax := 0.055 * T

-- Theorem stating that the percentage spent on food is 20%
theorem percentage_spent_on_food : 
  (total_tax = tax_on_clothing + tax_on_food 20 + tax_on_other_items) →
  50 + 30 + 20 = 100 :=
begin
  intros h,
  sorry 
end

end percentage_spent_on_food_l86_86775


namespace two_a_minus_b_l86_86281

-- Definitions of vector components and parallelism condition
def is_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 - a.2 * b.1 = 0
def vector_a : ℝ × ℝ := (1, -2)

-- Given assumptions
variable (m : ℝ)
def vector_b : ℝ × ℝ := (m, 4)

-- Theorem statement
theorem two_a_minus_b (h : is_parallel vector_a (vector_b m)) : 2 • vector_a - vector_b m = (4, -8) :=
sorry

end two_a_minus_b_l86_86281


namespace solve_inequality_l86_86276

variables (a b c x α β : ℝ)

theorem solve_inequality 
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β)
  (h2 : β > α)
  (ha : a < 0)
  (h3 : α + β = -b / a)
  (h4 : α * β = c / a) :
  ∀ x, (c * x^2 + b * x + a < 0 ↔ x < 1 / β ∨ x > 1 / α) := 
  by
    -- A detailed proof would follow here.
    sorry

end solve_inequality_l86_86276


namespace tangent_line_slope_l86_86331

theorem tangent_line_slope (h : ℝ → ℝ) (a : ℝ) (P : ℝ × ℝ) 
  (tangent_eq : ∀ x y, 2 * x + y + 1 = 0 ↔ (x, y) = (a, h a)) : 
  deriv h a < 0 :=
sorry

end tangent_line_slope_l86_86331


namespace polar_form_product_l86_86591

theorem polar_form_product:
  ∀ (r1 r2 : ℝ) (θ1 θ2 : ℝ), 
    r1 > 0 → r2 > 0 →
    (0 ≤ θ1 ∧ θ1 < 360) → (0 ≤ θ2 ∧ θ2 < 360) →
    ((r1 * real.cos (θ1 * real.pi / 180) + r1 * real.sin (θ1 * real.pi / 180) * complex.I) *
     (r2 * real.cos (θ2 * real.pi / 180) + r2 * real.sin (θ2 * real.pi / 180) * complex.I) = 
     20 * real.cos (73 * real.pi / 180) + 20 * real.sin (73 * real.pi / 180) * complex.I) →
    (r1 * r2 = 20 ∧ θ1 + θ2 = 73) :=
begin
  intros r1 r2 θ1 θ2 hr1 hr2 hθ1 hθ2 h_prod,
  split,
  { exact mul_self (sqrt (5 * 4)), },
  { exact add_self (25 + 48), },
  sorry
end

end polar_form_product_l86_86591


namespace relationship_between_M_N_l86_86235

variables {a1 a2 : ℝ}

-- Conditions
def M := a1 * a2
def N := a1 + a2 - 1

-- Statement
theorem relationship_between_M_N
  (h1 : 0 < a1) (h2 : a1 < 1)
  (h3 : 0 < a2) (h4 : a2 < 1) :
  M > N :=
by
  sorry

end relationship_between_M_N_l86_86235


namespace mary_total_nickels_l86_86390

def mary_initial_nickels : ℕ := 7
def mary_dad_nickels : ℕ := 5

theorem mary_total_nickels : mary_initial_nickels + mary_dad_nickels = 12 := by
  sorry

end mary_total_nickels_l86_86390


namespace eval_expr_eq_zero_l86_86162

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l86_86162


namespace cyclic_quadrilateral_BCNM_l86_86371

-- Define the triangle ABC
variables {A B C D E F K G M N : Type*} [metric_space A]

-- Conditions for the problem
variables [has_lt A] (H_AB : A < B) (H_AC : A < C) (H_BC : B < C)
variables (circumcircle_ABC : circle A B C)
variables (center_D : D)
variables (touch_E : touches center_D E B C)
variables (touch_F : touches center_D F (B.extended A))
variables (intersect_KG_1 : intersects circumcircle_ABC center_D K G)
variables (intersect_EF : intersects (line K G) (line E F) M)
variables (intersect_CD : Line_Intersects (line K G) (line C D) N)

-- Prove that quadrilateral BCNM is cyclic
theorem cyclic_quadrilateral_BCNM : is_cyclic_quadrilateral B C N M :=
sorry

end cyclic_quadrilateral_BCNM_l86_86371


namespace train_length_is_correct_l86_86550

-- Define the given conditions
def speed_km_per_hr := 72
def time_seconds := 13.598912087033037
def bridge_length_m := 132

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (s_km_per_hr : ℕ) : ℝ := s_km_per_hr * 1000 / 3600

-- Calculate the speed in m/s
def speed_m_per_s := km_per_hr_to_m_per_s speed_km_per_hr

-- Calculate the total distance the train travels
def total_distance := speed_m_per_s * time_seconds

-- Define the length of the train
def train_length := total_distance - bridge_length_m

-- The theorem we are to prove
theorem train_length_is_correct : train_length = 140 := by
  -- You do not need to include the steps for the proof, just end with sorry.
  sorry

end train_length_is_correct_l86_86550


namespace inlet_fill_rate_correct_l86_86106

def tank_capacity : ℝ := 5760.000000000001

def leak_empty_time : ℝ := 6

def net_empty_time_with_inlet : ℝ := 8

def leak_rate (capacity : ℝ) (time : ℝ) : ℝ := capacity / time

def net_empty_rate_with_inlet (capacity : ℝ) (time : ℝ) : ℝ := capacity / time

theorem inlet_fill_rate_correct :
  let L := leak_rate tank_capacity leak_empty_time in
  let net_empty_rate := net_empty_rate_with_inlet tank_capacity net_empty_time_with_inlet in
  let F := L - net_empty_rate in
  F = 240 := by
  sorry

end inlet_fill_rate_correct_l86_86106


namespace angle_x_value_l86_86434

theorem angle_x_value (O P Q R S : Point) (h1 : is_semicircle O) 
  (h2 : dist O P = dist O Q) (h3 : ∠QOP = 67) (h4 : ∠PQS = 90) 
  (h5 : is_isosceles_triangle O Q R) : ∠OQR -  ∠OQS = 9 :=
by sorry

end angle_x_value_l86_86434


namespace no_prime_roots_l86_86937

def is_prime (n : ℕ) : Prop := ∃ p, nat.prime p ∧ p = n

theorem no_prime_roots (k : ℕ) :
  (∀ p q : ℕ, is_prime p ∧ is_prime q ∧ (p + q = 64) → (k = p * q)) → false :=
by
  sorry

end no_prime_roots_l86_86937


namespace area_of_square_from_circle_area_l86_86519

theorem area_of_square_from_circle_area
  (circle_area : ℝ)
  (h_circle_area : circle_area = 39424) :
  ∃ (s : ℝ), (4 * s = sqrt (39424 / Real.pi)) ∧ (s^2 = 784) := by
  sorry

end area_of_square_from_circle_area_l86_86519


namespace slips_with_3_count_l86_86802

def number_of_slips_with_3 (x : ℕ) : Prop :=
  let total_slips := 15
  let expected_value := 4.6
  let prob_3 := (x : ℚ) / total_slips
  let prob_8 := (total_slips - x : ℚ) / total_slips
  let E := prob_3 * 3 + prob_8 * 8
  E = expected_value

theorem slips_with_3_count : ∃ x : ℕ, number_of_slips_with_3 x ∧ x = 10 :=
by
  sorry

end slips_with_3_count_l86_86802


namespace compute_52_times_48_l86_86585

theorem compute_52_times_48 : 52 * 48 = 2496 :=
by
  have h : (50 + 2) * (50 - 2) = 50^2 - 2^2,
  { ring },
  have h1 : 50^2 = 2500 := rfl,
  have h2 : 2^2 = 4 := rfl,
  calc (50 + 2) * (50 - 2) = 50^2 - 2^2 : by exact h
                      ... = 2500 - 4   : by rw [h1, h2]
                      ... = 2496       : by norm_num

end compute_52_times_48_l86_86585


namespace permutation_divisible_by_seven_l86_86451

theorem permutation_divisible_by_seven :
  ∃ n : ℕ, (∃ (d1 d2 d3 d4 : ℕ) (h : {1, 3, 7, 9} = {d1, d2, d3, d4}), 
    n = d1 * 10^3 + d2 * 10^2 + d3 * 10 + d4) ∧ n % 7 = 0 :=
by
  sorry

end permutation_divisible_by_seven_l86_86451


namespace eval_ceil_floor_l86_86170

-- Definitions of the ceiling and floor functions and their relevant properties
def ceil (x : ℝ) : ℤ := ⌈x⌉
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Mathematical statement to prove
theorem eval_ceil_floor : ceil (7 / 3) + floor (- (7 / 3)) = 0 :=
by
  -- Proof is omitted
  sorry

end eval_ceil_floor_l86_86170


namespace checkerboard_red_squares_l86_86942

/-- Define the properties of the checkerboard -/
structure Checkerboard :=
  (size : ℕ)
  (colors : ℕ → ℕ → String)
  (corner_color : String)

/-- Our checkerboard patterning function -/
def checkerboard_colors (i j : ℕ) : String :=
  match (i + j) % 3 with
  | 0 => "blue"
  | 1 => "yellow"
  | _ => "red"

/-- Our checkerboard of size 33x33 -/
def chubby_checkerboard : Checkerboard :=
  { size := 33,
    colors := checkerboard_colors,
    corner_color := "blue" }

/-- Proof that the number of red squares is 363 -/
theorem checkerboard_red_squares (b : Checkerboard) (h1 : b.size = 33) (h2 : b.colors = checkerboard_colors) : ∃ n, n = 363 :=
  by sorry

end checkerboard_red_squares_l86_86942


namespace min_odd_integers_l86_86052

theorem min_odd_integers 
  (a b c d e f : ℤ)
  (h1 : a + b = 30)
  (h2 : c + d = 15)
  (h3 : e + f = 17)
  (h4 : c + d + e + f = 32) :
  ∃ n : ℕ, (n = 2) ∧ (∃ odd_count, 
  odd_count = (if (a % 2 = 0) then 0 else 1) + 
                     (if (b % 2 = 0) then 0 else 1) + 
                     (if (c % 2 = 0) then 0 else 1) + 
                     (if (d % 2 = 0) then 0 else 1) + 
                     (if (e % 2 = 0) then 0 else 1) + 
                     (if (f % 2 = 0) then 0 else 1) ∧
  odd_count = 2) := sorry

end min_odd_integers_l86_86052


namespace optimal_strategy_catch_second_l86_86439

-- Define the parameters and conditions
variable (city_post_office village_council : Type) 
variable (o : Type) -- The origin point
variable (t1 t2 : ℝ) -- Time of departure (with t2 = t1 + 15 minutes)
variable (v_walking : ℝ) -- Walking speed for messengers
variable (v_cycling : ℝ) -- Cycling speed for cyclist

-- Define the positions of the messengers (considering one-dimensional movement)
def position_messenger1 (t : ℝ) := v_walking * t
def position_messenger2 (t : ℝ) := v_walking * (t - 0.25) -- Starts 15 minutes later

-- Define the task for the cyclist
def catch_messenger (target_position : ℝ) := sorry -- The cyclist catches up the messenger

-- Assumption: The cyclist can deliver the money no matter which strategy
axiom cyclist_can_complete (t : ℝ) : 
  (v_cycling > v_walking) -> catch_messenger (position_messenger1 t) ∧ catch_messenger (position_messenger2 t)

-- The proof goal
theorem optimal_strategy_catch_second (t : ℝ) (h_v_cycling : v_cycling > v_walking) : 
  catch_messenger (position_messenger2 t) := sorry

end optimal_strategy_catch_second_l86_86439


namespace second_player_wins_on_1x2000_l86_86805

-- Definition of the game
inductive Cell
| S : Cell
| O : Cell
| Blank : Cell

def Game := List Cell

def is_win (game: Game) : Bool :=
  let check_consecutive := fun c1 c2 c3 =>
    match (c1, c2, c3) with
    | (Cell.S, Cell.O, Cell.S) => true
    | _ => false
  game.windowed(3).any check_consecutive

noncomputable def second_player_winning_strategy (game : Game) : bool :=
  -- This function represents a theoretical winning strategy for the second player.
  sorry

theorem second_player_wins_on_1x2000 :
  ∃ strategy : Game → bool, ∀ game : Game, strategy game = true → 
  second_player_winning_strategy game = true :=
sorry

end second_player_wins_on_1x2000_l86_86805


namespace find_Natisfy_condition_l86_86980

-- Define the original number
def N : Nat := 2173913043478260869565

-- Define the function to move the first digit of a number to the end
def move_first_digit_to_end (n : Nat) : Nat := sorry

-- The proof statement
theorem find_Natisfy_condition : 
  let new_num1 := N * 4
  let new_num2 := new_num1 / 5
  move_first_digit_to_end N = new_num2 
:=
  sorry

end find_Natisfy_condition_l86_86980


namespace area_of_isosceles_triangle_l86_86216

theorem area_of_isosceles_triangle (a b c : ℝ) (h1 : a = 10) (h2 : b = 13) (h3 : c = 13) :
  let s := (a + b + c) / 2 in  -- semi-perimeter
  sqrt (s * (s - a) * (s - b) * (s - c)) = 60 := 
by
  sorry

end area_of_isosceles_triangle_l86_86216


namespace photo_album_slots_l86_86953

def photos_from_cristina : Nat := 7
def photos_from_john : Nat := 10
def photos_from_sarah : Nat := 9
def photos_from_clarissa : Nat := 14

theorem photo_album_slots :
  photos_from_cristina + photos_from_john + photos_from_sarah + photos_from_clarissa = 40 :=
by
  sorry

end photo_album_slots_l86_86953


namespace calculate_AX_l86_86401

noncomputable def angle_deg_to_rad (x : ℝ) : ℝ :=
x * (Real.pi / 180)

theorem calculate_AX :
  ∀ (A B C X : ℝ × ℝ),
  (dist A B = 1) ∧
  ((dist B C) ^ 2 + (dist A C) ^ 2 = 1) ∧
  (angle_deg_to_rad 90 = Real.arccos ((dist A C) ^ 2 + (dist B C) ^ 2 - (dist A B) ^ 2 / (2 * dist A C * dist B C))) ∧
  (dist B X = dist C X) ∧
  (angle_deg_to_rad 18 = Real.arccos ((dist A X) ^ 2 + (dist C A + dist C X)^ 2 - (dist A C) ^ 2 / (2 * dist A X * (dist C A + dist C X)))) ∧ 
  angle_deg_to_rad 90 = Real.pi / 2 →
  let AC := dist A C in
  let BC := dist B C in
  AX = AC * Real.sqrt (Real.sec (angle_deg_to_rad 18) ^ 2 - BC * AC - 1 / 4) :=
sorry

end calculate_AX_l86_86401


namespace remainder_of_division_l86_86007

theorem remainder_of_division (L S R : ℕ) (h1 : L - S = 1365) (h2 : L = 1637) (h3 : L = 6 * S + R) : R = 5 :=
by
  sorry

end remainder_of_division_l86_86007


namespace subset_condition_intersection_condition_l86_86647

-- Definitions of the sets A and B
def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, 3 * a}

-- Theorem statements
theorem subset_condition (a : ℝ) : A ⊆ B a → (4 / 3) ≤ a ∧ a ≤ 2 := 
by 
  sorry

theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → (2 / 3) < a ∧ a < 4 := 
by 
  sorry

end subset_condition_intersection_condition_l86_86647


namespace primes_between_8000_and_12000_l86_86317

theorem primes_between_8000_and_12000 : 
  {p : ℕ | p.prime ∧ 8000 < p^2 ∧ p^2 < 12000}.card = 5 :=
by
  sorry

end primes_between_8000_and_12000_l86_86317


namespace distance_C_to_D_l86_86829

-- Define the given conditions
def smaller_square_perimeter : ℝ := 8
def larger_square_area : ℝ := 36

-- Define the side lengths of the squares derived from the conditions
def s1 : ℝ := smaller_square_perimeter / 4 -- side of the smaller square
def s2 : ℝ := real.sqrt larger_square_area -- side of the larger square

-- Define the distances for the legs of the right triangle
def leg_distance : ℝ := s2 - s1

-- State the main theorem to be proved 
theorem distance_C_to_D : 
  real.sqrt (leg_distance ^ 2 + leg_distance ^ 2) = real.sqrt 32 := 
by 
  sorry

end distance_C_to_D_l86_86829


namespace find_roots_of_polynomial_l86_86608

def f (x : ℝ) := x^3 - 2*x^2 - 5*x + 6

theorem find_roots_of_polynomial :
  (f 1 = 0) ∧ (f (-2) = 0) ∧ (f 3 = 0) :=
by
  -- Proof will be written here
  sorry

end find_roots_of_polynomial_l86_86608


namespace number_of_one_dollar_coins_l86_86674

theorem number_of_one_dollar_coins (t : ℕ) :
  (∃ k : ℕ, 3 * k = t) → ∃ k : ℕ, k = t / 3 :=
by
  sorry

end number_of_one_dollar_coins_l86_86674


namespace sum_of_legs_equal_l86_86724

theorem sum_of_legs_equal
  (a b c d e f g h : ℝ)
  (x y : ℝ)
  (h_similar_shaded1 : a = a * x ∧ b = a * y)
  (h_similar_shaded2 : c = c * x ∧ d = c * y)
  (h_similar_shaded3 : e = e * x ∧ f = e * y)
  (h_similar_shaded4 : g = g * x ∧ h = g * y)
  (h_similar_unshaded1 : h = h * x ∧ a = h * y)
  (h_similar_unshaded2 : b = b * x ∧ c = b * y)
  (h_similar_unshaded3 : d = d * x ∧ e = d * y)
  (h_similar_unshaded4 : f = f * x ∧ g = f * y)
  (x_non_zero : x ≠ 0) (y_non_zero : y ≠ 0) : 
  (a * y + b + c * x) + (c * y + d + e * x) + (e * y + f + g * x) + (g * y + h + a * x) 
  = (h * x + a + b * y) + (b * x + c + d * y) + (d * x + e + f * y) + (f * x + g + h * y) :=
sorry

end sum_of_legs_equal_l86_86724


namespace number_of_newborn_members_l86_86705

theorem number_of_newborn_members (N : ℝ) (h : (9/10 : ℝ) ^ 3 * N = 291.6) : N = 400 :=
sorry

end number_of_newborn_members_l86_86705


namespace equal_diagonals_of_inscribed_isosceles_trapezoids_l86_86932

variables {ABC: Type*} [circle ABC] [isosceles_trapezoid ABCD] [isosceles_trapezoid A1B1C1D1]
  (h1 : inscribed ABCD)
  (h2 : inscribed A1B1C1D1)
  (h3 : parallel AB CD)
  (h4 : parallel A1B1 C1D1)

theorem equal_diagonals_of_inscribed_isosceles_trapezoids :
  AC = A1C1 :=
by
  sorry

end equal_diagonals_of_inscribed_isosceles_trapezoids_l86_86932


namespace four_digit_zero_erased_decreases_by_nine_l86_86405

theorem four_digit_zero_erased_decreases_by_nine (n : ℕ) (a b c : ℕ) : 
  n = 2025 ∨ n = 6075 :=
begin
  have h0 : 1000 ≤ n ∧ n < 10000, from sorry,
  have h1 : (∃ a b c, n = a * 1000 + b * 100 + 0 * 10 + c ∨ n = a * 1000 + b * 100 + c * 10 + 0) ∧ 
            n = 9 * (a * 100 + b * 10 + c), from sorry,
  sorry,
end

end four_digit_zero_erased_decreases_by_nine_l86_86405


namespace pentagon_hexagon_side_length_ratio_l86_86117

theorem pentagon_hexagon_side_length_ratio:
  let perimeter_pentagon := 60
  let perimeter_hexagon := 60
  let side_length_pentagon := perimeter_pentagon / 5
  let side_length_hexagon := perimeter_hexagon / 6
  in side_length_pentagon / side_length_hexagon = 6 / 5 :=
sorry

end pentagon_hexagon_side_length_ratio_l86_86117


namespace eight_mul_reciprocal_frac_sum_l86_86497

theorem eight_mul_reciprocal_frac_sum : 
    8 * ((1/3) + (1/4) + (1/12))⁻¹ = 12 := 
by {
    -- Proof steps would go here
    sorry
}

end eight_mul_reciprocal_frac_sum_l86_86497


namespace second_supplier_more_cars_l86_86559

-- Define the constants and conditions given in the problem
def total_production := 5650000
def first_supplier := 1000000
def fourth_fifth_supplier := 325000

-- Define the unknown variable for the second supplier
noncomputable def second_supplier : ℕ := sorry

-- Define the equation based on the conditions
def equation := first_supplier + second_supplier + (first_supplier + second_supplier) + (4 * fourth_fifth_supplier / 2) = total_production

-- Prove that the second supplier receives 500,000 more cars than the first supplier
theorem second_supplier_more_cars : 
  ∃ X : ℕ, equation → (X = first_supplier + 500000) :=
sorry

end second_supplier_more_cars_l86_86559


namespace angle_sum_of_circle_points_l86_86781

theorem angle_sum_of_circle_points 
  (A B C D E F : Point)
  (circle : ∀ P : Point, P ∈ {A, B, C, D, E, F} → on_circle P)
  (arc_AB : measure_arc A B = 24)
  (arc_BC : measure_arc B C = 36)
  (arc_CD : measure_arc C D = 40)
  (P : Point)
  (P_extended_line_CF : on_line P C F) :
  angle_sum (angle_at_point C B D) (angle_at_point P (line P C) (line P F)) = 76 :=
sorry

end angle_sum_of_circle_points_l86_86781


namespace gcd_63_84_l86_86219

theorem gcd_63_84 : Nat.gcd 63 84 = 21 := by
  -- The proof will go here.
  sorry

end gcd_63_84_l86_86219


namespace smallest_n_l86_86586

theorem smallest_n (n : ℕ) : (∀ m < n, ∑ k in Finset.range m, Real.logb 3 (1 + 1 / 3^(3^k)) < 1 + Real.logb 3 (2022 / 2023)) ∧ 
                             (∑ k in Finset.range n, Real.logb 3 (1 + 1 / 3^(3^k)) ≥ 1 + Real.logb 3 (2022 / 2023)) 
                             → n = 1 :=
by { sorry }

end smallest_n_l86_86586


namespace billionth_term_is_16_l86_86152

-- Define the sequence generation rule
def sequence_rule (a_n : ℕ) : ℕ :=
  a_n + 5 * (a_n % 10)

-- Initial term of the sequence
def a_1 : ℕ := 112002

-- Define the nth term of the sequence recursively
def a : ℕ → ℕ
| 1     := a_1
| (n+1) := sequence_rule (a n)

-- The goal is to prove the billionth term
theorem billionth_term_is_16 : a (10^9) = 16 :=
sorry

end billionth_term_is_16_l86_86152


namespace hyperbola_equation_l86_86242

theorem hyperbola_equation
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0)
  (H1 : focus_parabola = (2, 0)) 
  (H2 : distance_focus_asymptote = 4 * real.sqrt 5 / 5) 
  (H3 : min_distance_sum = 3):
  (a^2 = 4 ∧ b^2 = 1) := by
sorry

end hyperbola_equation_l86_86242


namespace jumping_contest_l86_86017

theorem jumping_contest (grasshopper_jump frog_jump : ℕ) (h_grasshopper : grasshopper_jump = 9) (h_frog : frog_jump = 12) : frog_jump - grasshopper_jump = 3 := by
  ----- h_grasshopper and h_frog are our conditions -----
  ----- The goal is to prove frog_jump - grasshopper_jump = 3 -----
  sorry

end jumping_contest_l86_86017


namespace mean_score_of_seniors_l86_86777

variable (s n : ℕ)  -- Number of seniors and non-seniors
variable (m_s m_n : ℝ)  -- Mean scores of seniors and non-seniors
variable (total_mean : ℝ) -- Mean score of all students
variable (total_students : ℕ) -- Total number of students

theorem mean_score_of_seniors :
  total_students = 100 → total_mean = 100 →
  n = 3 * s / 2 →
  s * m_s + n * m_n = total_students * total_mean →
  m_s = (3 * m_n / 2) →
  m_s = 125 :=
by
  intros
  sorry

end mean_score_of_seniors_l86_86777


namespace triangle_area_l86_86398

theorem triangle_area (base height : ℝ) (h_base : base = 15) (h_height : height = 20) : 
  (base * height) / 2 = 150 :=
by
  rw [h_base, h_height]
  norm_num
  sorry

end triangle_area_l86_86398


namespace maximum_achievable_score_l86_86837

def robot_initial_iq : Nat := 25
def problem_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem maximum_achievable_score 
  (initial_iq : Nat := robot_initial_iq) 
  (scores : List Nat := problem_scores) 
  : Nat :=
  31

end maximum_achievable_score_l86_86837


namespace sum_of_interior_angles_of_regular_polygon_l86_86711

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h : 60 = 360 / n) : (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_of_regular_polygon_l86_86711


namespace triangle_area_AC15_BC13_IG_parallel_AB_l86_86692

theorem triangle_area_AC15_BC13_IG_parallel_AB
  (ABC : Type)
  [triangle ABC]
  (A B C I G : ABC)
  (hAC : dist A C = 15)
  (hBC : dist B C = 13)
  (hI_incenter : is_incenter I)
  (hG_centroid : is_centroid G)
  (hIG_parallel_AB : IG ∥ AB) :
  ∃ (area : ℝ), area = 84 := 
sorry

end triangle_area_AC15_BC13_IG_parallel_AB_l86_86692


namespace largest_positive_multiple_of_15_l86_86817

noncomputable def m : ℕ := 8888880

theorem largest_positive_multiple_of_15 
  (h1 : (m % 15) = 0) 
  (h2 : (∀ d : ℕ, d ∈ digits 10 m → d = 8 ∨ d = 0)) 
  (h3 : (digits 10 m).count 8 = 6): 
  m / 15 = 592592 := 
by 
  sorry

end largest_positive_multiple_of_15_l86_86817


namespace incenter_common_external_tangents_perpendicular_l86_86903

noncomputable def is_incenter (I A B C : Point) : Prop :=
  sorry  -- Definition of incenter based on angle bisectors

noncomputable def has_inscribed_circle (quadrilateral : Type) (I : Point) : Prop :=
  sorry  -- Definition of when a quadrilateral has an inscribed circle

theorem incenter_common_external_tangents_perpendicular
  (A B C D I I_a I_b I_c I_d X Y : Point)
  (h_convex: ConvexQuadrilateral A B C D)
  (h_inscribed: has_inscribed_circle (ConvexQuadrilateral A B C D) I)
  (h_Ia: is_incenter I_a D A B)
  (h_Ib: is_incenter I_b A B C)
  (h_Ic: is_incenter I_c B C D)
  (h_Id: is_incenter I_d C D A)
  (h_X: meet_at_common_external_tangents (circle A I_b I_d) (circle C I_b I_d) X)
  (h_Y: meet_at_common_external_tangents (circle B I_a I_c) (circle D I_a I_c) Y) :
  ∠ XI Y = 90 := 
  sorry

end incenter_common_external_tangents_perpendicular_l86_86903


namespace basis_of_constructing_angle_equal_to_known_angle_is_SSS_l86_86808

-- Definitions based on conditions
def using_compass_and_straightedge := true
def constructing_angle_equal_to_known_angle := true

-- Proof problem statement
theorem basis_of_constructing_angle_equal_to_known_angle_is_SSS
  (h1 : using_compass_and_straightedge)
  (h2 : constructing_angle_equal_to_known_angle) :
  "SSS" := 
sorry

end basis_of_constructing_angle_equal_to_known_angle_is_SSS_l86_86808


namespace find_v_l86_86988

theorem find_v (v : ℝ × ℝ)
  (h₁ : ∃ x y : ℝ, v = (x, y) ∧ (proj (3, 1) v = (6, 2)))
  (h₂ : ∃ x y : ℝ, v = (x, y) ∧ (proj (1, 2) v = (2, 4))) :
  v = (6, 2) := sorry

end find_v_l86_86988


namespace vectors_not_form_triangle_l86_86669

open EuclideanGeometry

def vector_a : ℝ × ℝ × ℝ := (-1, 2, -1)
def vector_b : ℝ × ℝ × ℝ := (-3, 6, -3)
def vector_c : ℝ × ℝ × ℝ := (-2, 4, -2)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def is_degenerate_triangle 
  (a b c : ℝ × ℝ × ℝ) : Prop :=
  let ma := magnitude a
  let mb := magnitude b
  let mc := magnitude c
  (ma + mc = mb ∨ ma + mb = mc ∨ mb + mc = ma)

theorem vectors_not_form_triangle : 
  ¬ ∃ (a b c : ℝ × ℝ × ℝ), 
    a = vector_a ∧ b = vector_b ∧ c = vector_c ∧ ¬ is_degenerate_triangle a b c := 
by
  sorry

end vectors_not_form_triangle_l86_86669


namespace bounded_S_k_l86_86374

variables {n : ℕ} (G : Set (Equiv.Perm (Fin n))) (k : Fin n)

-- Condition: \( n \geq 3 \)
hypothesis h_n : 3 ≤ n

-- Condition: \( G \) is a subgroup of \( S_n \) generated by \( n-2 \) transpositions
def G_is_subgroup_of_Sn : Prop :=
  ∃ (gen_set : Set (Equiv.Perm (Fin n))), gen_set.card = n - 2 ∧ Subgroup.closure gen_set = G

-- Condition: For all \( k \in \{1, 2, \ldots, n\} \), \( S(k) = \{ σ(k) : σ ∈ G \} \)
def S (k : Fin n) : Set (Fin n) := {i | ∃ σ ∈ G, σ k = i}

theorem bounded_S_k (hG : G_is_subgroup_of_Sn G) : ∀ k : Fin n, (S G k).card ≤ n - 1 :=
by sorry

end bounded_S_k_l86_86374


namespace inequality_holds_iff_even_l86_86761

theorem inequality_holds_iff_even (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∀ x y z : ℝ, (x - y) ^ a * (x - z) ^ b * (y - z) ^ c ≥ 0) ↔ (Even a ∧ Even b ∧ Even c) :=
by
  sorry

end inequality_holds_iff_even_l86_86761


namespace num_valid_Ns_less_2000_l86_86313

theorem num_valid_Ns_less_2000 : 
  {N : ℕ | N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x^⟨floor x⟩ = N}.card = 412 := 
sorry

end num_valid_Ns_less_2000_l86_86313


namespace eval_expr_eq_zero_l86_86166

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l86_86166


namespace car_pass_time_l86_86080

theorem car_pass_time (length : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ) (time : ℝ) :
  length = 10 → 
  speed_kmph = 36 → 
  speed_mps = speed_kmph * (1000 / 3600) → 
  time = length / speed_mps → 
  time = 1 :=
by
  intros h_length h_speed_kmph h_speed_conversion h_time_calculation
  -- Here we would normally construct the proof
  sorry

end car_pass_time_l86_86080


namespace specific_five_card_order_probability_l86_86225

open Classical

noncomputable def prob_five_cards_specified_order : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49) * (9 / 48)

theorem specific_five_card_order_probability :
  prob_five_cards_specified_order = 2304 / 31187500 :=
by
  sorry

end specific_five_card_order_probability_l86_86225


namespace cookie_radius_proof_l86_86804

-- Define the given equation of the cookie
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 36 = 6 * x + 9 * y

-- Define the radius computation for the circle derived from the given equation
def cookie_radius (r : ℝ) : Prop :=
  r = 3 * Real.sqrt 5 / 2

-- The theorem to prove that the radius of the described cookie is as obtained
theorem cookie_radius_proof :
  ∀ x y : ℝ, cookie_equation x y → cookie_radius (Real.sqrt (45 / 4)) :=
by
  sorry

end cookie_radius_proof_l86_86804


namespace new_average_income_l86_86426

theorem new_average_income (old_avg_income : ℝ) (num_members : ℕ) (deceased_income : ℝ) 
  (old_avg_income_eq : old_avg_income = 735) (num_members_eq : num_members = 4) 
  (deceased_income_eq : deceased_income = 990) : 
  ((old_avg_income * num_members) - deceased_income) / (num_members - 1) = 650 := 
by sorry

end new_average_income_l86_86426


namespace ceiling_floor_sum_l86_86158

theorem ceiling_floor_sum : (Int.ceil (7 / 3) + Int.floor (-(7 / 3)) = 0) := by
  sorry

end ceiling_floor_sum_l86_86158


namespace remainder_when_divided_by_5_l86_86624

theorem remainder_when_divided_by_5 (n : ℕ) (h1 : n^2 % 5 = 1) (h2 : n^3 % 5 = 4) : n % 5 = 4 :=
sorry

end remainder_when_divided_by_5_l86_86624


namespace geometric_progression_fifth_term_l86_86010

theorem geometric_progression_fifth_term (a1 a2 a3 : Real) 
  (h1 : a1 = Real.sqrt 4) 
  (h2 : a2 = Real.sqrt 6 4) 
  (h3 : a3 = Real.sqrt 12 4) : 
  ∃ a5 : Real, a5 = 1 / Real.sqrt 12 4 :=
sorry

end geometric_progression_fifth_term_l86_86010


namespace altitude_pass_midpoint_of_median_l86_86728

-- Given conditions as definitions
variables {A B C H M N O K : Type} [Point ℝ] 
variables (mid_AC : Midpoint A C M) (mid_BM : Midpoint B M H) 
variables (altitude : Altitude H A)

-- Define points and relations in triangle BMC
variables (mid_MC : Midpoint M C N) (mid_BN : Midpoint B N O)
variables (projection : Projection M B C K) (altitude_BMC : Altitude K M)

-- Theorem statement given the conditions
theorem altitude_pass_midpoint_of_median 
  (H_condition : passes_through altitude mid_BM)
  (M_condition : passes_through_median B M mid_AC)
  (BMC_condition_1 : passes_through projection B (Altitude K M))
  (BMC_condition_2 : collinear M O K) :
  passes_through (Altitude M K) O :=
sorry

end altitude_pass_midpoint_of_median_l86_86728


namespace cube_edge_factor_l86_86810

theorem cube_edge_factor (e f : ℝ) (h₁ : e > 0) (h₂ : (f * e) ^ 3 = 8 * e ^ 3) : f = 2 :=
by
  sorry

end cube_edge_factor_l86_86810


namespace new_sailor_weight_l86_86876

-- Define the conditions
variables {average_weight : ℝ} (new_weight : ℝ)
variable (old_weight : ℝ := 56)

-- State the property we need to prove
theorem new_sailor_weight
  (h : (new_weight - old_weight) = 8) :
  new_weight = 64 :=
by
  sorry

end new_sailor_weight_l86_86876


namespace matrix_P_condition_l86_86214

theorem matrix_P_condition (Q : Matrix (Fin 3) (Fin 3) ℝ) :
  ∃ (P : Matrix (Fin 3) (Fin 3) ℝ),
    P ⬝ Q = 
      Matrix.of (Fin 3) (Fin 3) ℝ (λ i j, 
        if i = 0 then 3 * Q 0 j
        else if i = 1 then Q 2 j
        else Q 1 j
      ) ∧ 
    P = λ i j, 
            if i = 0 then if j = 0 then 3 else 0
            else if i = 1 then if j = 2 then 1 else 0
            else if i = 2 then if j = 1 then 1 else 0
            else 0 :=
begin
  use ![
    ![3, 0, 0],
    ![0, 0, 1],
    ![0, 1, 0]
  ],
  split,
  { sorry },
  { sorry }
end

end matrix_P_condition_l86_86214


namespace count_valid_N_l86_86299

theorem count_valid_N : 
  ∃ (N : ℕ), (N < 2000) ∧ (∃ (x : ℝ), x^⌊x⌋₊ = N) :=
begin
  sorry
end

end count_valid_N_l86_86299


namespace eval_ceil_floor_sum_l86_86203

theorem eval_ceil_floor_sum :
  (Int.ceil (7 / 3) + Int.floor (- (7 / 3))) = 0 :=
by
  sorry

end eval_ceil_floor_sum_l86_86203


namespace tangent_slope_l86_86663

noncomputable def f (x : ℝ) : ℝ := x - 1 + 1 / Real.exp x

noncomputable def f' (x : ℝ) : ℝ := 1 - 1 / Real.exp x

theorem tangent_slope (k : ℝ) (x₀ : ℝ) (y₀ : ℝ) 
  (h_tangent_point: (x₀ = -1) ∧ (y₀ = x₀ - 1 + 1 / Real.exp x₀))
  (h_tangent_line : ∀ x, y₀ = f x₀ + f' x₀ * (x - x₀)) :
  k = 1 - Real.exp 1 := 
sorry

end tangent_slope_l86_86663


namespace integral_correct_value_l86_86135

noncomputable def integral_expression : ℝ :=
  ∫ x in 0..(π / 2), (cos x / (5 + 4 * cos x))

theorem integral_correct_value :
  integral_expression = (π / 8) - (5 / 6) * arctan (1 / 3) :=
by
  sorry

end integral_correct_value_l86_86135


namespace optimal_worker_distribution_l86_86827
noncomputable def forming_time (n1 : ℕ) : ℕ := (nat.ceil (75.0 / n1) : ℕ) * 15
noncomputable def firing_time (n3 : ℕ) : ℕ := (nat.ceil (75.0 / n3) : ℕ) * 30

theorem optimal_worker_distribution:
  ∃ n1 n3 : ℕ, n1 + n3 = 13 ∧ (forming_time n1 = 225 ∨ firing_time n3 = 330) :=
sorry

end optimal_worker_distribution_l86_86827


namespace y_worked_days_l86_86085

-- Definitions based on conditions
def work_rate_x := 1 / 20 -- x's work rate (W per day)
def work_rate_y := 1 / 16 -- y's work rate (W per day)

def remaining_work_by_x := 5 * work_rate_x -- Work finished by x after y left
def total_work := 1 -- Assume the total work W is 1 unit for simplicity

def days_y_worked (d : ℝ) := d * work_rate_y + remaining_work_by_x = total_work

-- The statement we need to prove
theorem y_worked_days :
  (exists d : ℕ, days_y_worked d ∧ d = 15) :=
sorry

end y_worked_days_l86_86085


namespace perpendicular_planes_condition_l86_86753

variables {α : Type} {l m n : α}
variables [LinearOrderedField α] [Nat α]

-- l ⊥ α is sufficient but not necessary for l ⊥ m and l ⊥ n, m and n in α
def perpendicular_sufficient_condition (l m n : α) (α : Set α) : Prop :=
  (∀ (l : α) (α : Set α), (l ⊥ α) → ((l ⊥ m) ∧ (l ⊥ n))) ∧
  ¬(∀ (l : α) (m n : α) (α : Set α), ((l ⊥ m) ∧ (l ⊥ n)) → (l ⊥ α))

theorem perpendicular_planes_condition (l m n : α) (α : Set α):
  perpendicular_sufficient_condition l m n α := sorry

end perpendicular_planes_condition_l86_86753


namespace trigonometric_identity_l86_86757

theorem trigonometric_identity (a b : ℝ) (h1 : sin a / sin b = 4) (h2 : cos a / cos b = 1 / 3) :
  sin (2 * a) / sin (2 * b) + cos (2 * a) / cos (2 * b) = 1 := by
  sorry

end trigonometric_identity_l86_86757


namespace function_extremes_analysis_l86_86885

def f (x : ℝ) : ℝ := x^3 + 2 * x^2 - 4 * x + 2

-- Given that the function f has extreme values at x = -2 and x = 1,
-- Prove that the analytical expression is f(x) = x^3 + 2x^2 - 4x + 2
-- and find the monotonic intervals for f(x).

theorem function_extremes_analysis :
  (∀ x : ℝ, f'(x) = 3 * x^2 + 4 * x - 4 ∧
  (f (-2)).deriv = 0 ∧ (f 1).deriv = 0) →
  (f = λ x : ℝ, x^3 + 2 * x^2 - 4 * x + 2) ∧
  (∀ x, f'(x) > 0 ↔ x < -2 ∨ x > 1) ∧
  (∀ x, f'(x) < 0 ↔ -2 < x ∧ x < 1) :=
sorry

end function_extremes_analysis_l86_86885


namespace find_n_l86_86657

theorem find_n : ∃ n : ℕ, 
  (S : ℕ) (i : ℕ) (hS₀ : S = 0) (hi₀ : i = 1)
  (hloop : ∀ (S i : ℕ), S ≤ 200 → (S+ i, i + 2)) 
  (hfinal : i - 2 = n), n = 27 :=
by
  let S := 0
  let i := 1
  let S := Nat.iterate 14 (λ (p : ℕ × ℕ), (p.1 + p.2, p.2 + 2)) (S, i)
  let S := S.fst
  let i := S.snd
  let n := i - 2
  exact ⟨n, by simp [S, i, n]⟩

end find_n_l86_86657


namespace total_eyes_among_ninas_pet_insects_l86_86774

theorem total_eyes_among_ninas_pet_insects
  (num_spiders : ℕ) (num_ants : ℕ)
  (eyes_per_spider : ℕ) (eyes_per_ant : ℕ)
  (h_num_spiders : num_spiders = 3)
  (h_num_ants : num_ants = 50)
  (h_eyes_per_spider : eyes_per_spider = 8)
  (h_eyes_per_ant : eyes_per_ant = 2) :
  num_spiders * eyes_per_spider + num_ants * eyes_per_ant = 124 := 
by
  rw [h_num_spiders, h_num_ants, h_eyes_per_spider, h_eyes_per_ant]
  norm_num
  done

end total_eyes_among_ninas_pet_insects_l86_86774


namespace cone_lateral_surface_area_l86_86259

theorem cone_lateral_surface_area (r l : ℝ) (h1 : r = 3) (h2 : l = 5) :
  1/2 * (2 * π * r) * l = 15 * π :=
by 
  rw [h1, h2]
  simp
  linarith

end cone_lateral_surface_area_l86_86259


namespace length_of_floor_l86_86445

theorem length_of_floor (b l d : ℝ) (h1 : l = 3.5 * b)
  (h2 : l^2 + b^2 = d^2) (h3 : d = 13) : l ≈ 14.49 :=
by
  sorry

end length_of_floor_l86_86445


namespace at_most_two_even_l86_86850

-- Assuming the negation of the proposition
def negate_condition (a b c : ℕ) : Prop := a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0

-- Proposition to prove by contradiction
theorem at_most_two_even 
  (a b c : ℕ) 
  (h : negate_condition a b c) 
  : False :=
sorry

end at_most_two_even_l86_86850


namespace work_completion_days_l86_86511

theorem work_completion_days (x : ℕ) 
  (h1 : (1 : ℚ) / x + 1 / 9 = 1 / 6) :
  x = 18 := 
sorry

end work_completion_days_l86_86511


namespace largest_prime_factor_of_675_l86_86056

theorem largest_prime_factor_of_675 : ∃ p, nat.prime p ∧ p ∣ 675 ∧ ∀ q, nat.prime q ∧ q ∣ 675 → q ≤ p :=
by {
  sorry
}

end largest_prime_factor_of_675_l86_86056


namespace kona_additional_miles_l86_86630

def distance_apartment_to_bakery : ℕ := 9
def distance_bakery_to_grandmothers_house : ℕ := 24
def distance_grandmothers_house_to_apartment : ℕ := 27

theorem kona_additional_miles:
  let no_bakery_round_trip := 2 * distance_grandmothers_house_to_apartment in
  let with_bakery_round_trip := 
    distance_apartment_to_bakery +
    distance_bakery_to_grandmothers_house +
    distance_grandmothers_house_to_apartment in
  with_bakery_round_trip - no_bakery_round_trip = 33 :=
by
  let no_bakery_round_trip : ℕ := 54
  let with_bakery_round_trip : ℕ := 60
  calc
    no_bakery_round_trip = 27 + 27 := sorry
    with_bakery_round_trip = 9 + 24 + 27 := sorry
    with_bakery_round_trip - no_bakery_round_trip = 33 := sorry

end kona_additional_miles_l86_86630


namespace minimize_triangle_area_equation_l86_86241

noncomputable def line_equation (P : ℝ × ℝ) : ℝ × ℝ × ℝ := (2, 1, -4)

theorem minimize_triangle_area_equation (P : ℝ × ℝ) (hP : P = (1, 2)) :
  ∃ l : ℝ × ℝ × ℝ, l = line_equation P ∧ l = (2, 1, -4) :=
  by
    use (2, 1, -4)
    simp [line_equation, hP]
    exact ⟨rfl, rfl⟩

end minimize_triangle_area_equation_l86_86241


namespace cake_division_l86_86568

/-- Anton's cake is in the shape of the letter "Ш".
    Prove that it is possible to divide this cake into exactly 4 pieces using a single straight cut. -/
theorem cake_division (cake_shape : Type) (is_Ш_shaped : cake_shape = "Ш") :
  ∃ (cut : cake_shape → ℕ), (cut "Ш" = 4) :=
sorry

end cake_division_l86_86568


namespace find_x_value_l86_86621

theorem find_x_value (x : ℝ) :
  |x - 25| + |x - 21| = |3 * x - 75| → x = 71 / 3 :=
by
  sorry

end find_x_value_l86_86621


namespace book_price_l86_86100

theorem book_price (B P : ℝ) (h_unsold : B / 3 = 30) (h_total_received : (2 / 3) * B * P = 255) : P = 4.25 :=
by
  have h1 : B = 30 * 3, from (eq_div_iff (div_ne_zero one_ne_zero three_ne_zero)).mp h_unsold.symm,
  rw h1 at *,
  have h2 : (2 / 3) * (30 * 3) = (30 * 2), by ring,
  rw h2 at *,
  have h3 : (30 * 2) = 60, by norm_num,
  rw h3 at *,
  have h4 : 60 * P = 255, from h_total_received,
  have h5 : P = 255 / 60, by exact (eq_div_iff (div_ne_zero sixty_ne_zero)).mp h4.symm,
  norm_num at h5,
  exact h5

end book_price_l86_86100


namespace eval_ceil_floor_l86_86177

-- Definitions of the ceiling and floor functions and their relevant properties
def ceil (x : ℝ) : ℤ := ⌈x⌉
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Mathematical statement to prove
theorem eval_ceil_floor : ceil (7 / 3) + floor (- (7 / 3)) = 0 :=
by
  -- Proof is omitted
  sorry

end eval_ceil_floor_l86_86177


namespace find_seq_b_l86_86882

variables {α : Type*} [LinearOrderedField α]

def seq_b (a : ℕ → α) (q : α) (n i : ℕ) : α :=
  (finset.range n).sum (λ k, a k * q ^ abs (i - k))

theorem find_seq_b (a : ℕ → α) (q : α) (h1 : ∀ k < n, 0 < a k) (h2 : 0 < q ∧ q < 1) :
  ∀ k < n,
    a k < seq_b a q n k ∧
    q < (seq_b a q n (k+1) + 1) / seq_b a q n k ∧
    (seq_b a q n 0 + seq_b a q n 1 + ... + seq_b a q n (n-1)) < (1 + q) / (1 - q) * (a 0 + a 1 + ... + a (n-1)) :=
begin
  assume k hnk,
  sorry
end

end find_seq_b_l86_86882


namespace ceiling_floor_sum_l86_86161

theorem ceiling_floor_sum : (Int.ceil (7 / 3) + Int.floor (-(7 / 3)) = 0) := by
  sorry

end ceiling_floor_sum_l86_86161


namespace neg_prop_true_l86_86025

theorem neg_prop_true (a : ℝ) :
  ¬ (∀ a : ℝ, a ≤ 2 → a^2 < 4) → ∃ a : ℝ, a > 2 ∧ a^2 ≥ 4 :=
by
  intros h
  sorry

end neg_prop_true_l86_86025


namespace least_four_digit_multiple_3_5_7_l86_86064

theorem least_four_digit_multiple_3_5_7 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ n = 1050 :=
by
  use 1050
  repeat {sorry}

end least_four_digit_multiple_3_5_7_l86_86064


namespace dice_probability_odd_sum_div4_l86_86623

theorem dice_probability_odd_sum_div4 :
  (∃ dice : Fin 6 → Fin 6, (∏ i, dice i) % 4 = 0 → (∑ i, dice i) % 2 = 1) →
  (∑ i : Fin 6, ∏ j : Fin 5, choose (6 : ℕ) (i.dice j)) = 1 / 7 := 
sorry

end dice_probability_odd_sum_div4_l86_86623


namespace positive_integers_odd_n_ge_3_l86_86606

theorem positive_integers_odd_n_ge_3
  (n : ℕ)
  (h : 3 ≤ n)
  (odd_n : n % 2 = 1)
  (a b : fin n → ℝ)
  (habs : ∀ k, |a k| + |b k| = 1) :
  ∃ ε : fin n → ℤ, (∀ i, ε i = 1 ∨ ε i = -1) ∧
           |∑ i, (ε i) * a i| + |∑ i, (ε i) * b i| ≤ 1 := 
sorry

end positive_integers_odd_n_ge_3_l86_86606


namespace circles_inscribed_equal_radii_l86_86403

-- Setup the conditions from part a)
structure Segment :=
(A B : Point)

structure Semicircle (seg : Segment) :=
(radius : ℝ)
(diameter : Segment)

structure CircleInscribedInCurvedTriangle {seg : Segment} (s : Semicircle) (C : Point) :=
(r : ℝ)

-- Define the initial conditions
def AC : Segment := { A := P1, B := P2 }
def BC : Segment := { A := P2, B := P3 }
def AB : Segment := { A := P1, B := P3 }

def semicircle_AC : Semicircle AC := { radius := 0.5 * AC.length, diameter := AC }
def semicircle_BC : Semicircle BC := { radius := 0.5 * BC.length, diameter := BC }
def semicircle_AB : Semicircle AB := { radius := 0.5 * AB.length, diameter := AB }

-- Define the circles inscribed in the curved triangles
def C : Point := midpoint AC
def S1 : CircleInscribedInCurvedTriangle semicircle_AC C := { r := (0.5 * AB.length * 0.5 * BC.length) / (0.5 * AB.length + 0.5 * BC.length) }
def S2 : CircleInscribedInCurvedTriangle semicircle_BC C := { r := (0.5 * AB.length * 0.5 * AC.length) / (0.5 * AB.length + 0.5 * AC.length) }

-- The theorem statement
theorem circles_inscribed_equal_radii : S1.r = S2.r :=
sorry

end circles_inscribed_equal_radii_l86_86403


namespace derivative_at_pi_over_3_l86_86013

-- Define the function f
def f (x : ℝ) : ℝ := 1 + Real.sin x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := Real.cos x

-- State the theorem to find f' at π/3
theorem derivative_at_pi_over_3 : f' (Real.pi / 3) = 1 / 2 :=
by
  -- Proof skipped
  sorry

end derivative_at_pi_over_3_l86_86013


namespace apples_in_each_basket_l86_86738

-- Definitions based on the conditions
def total_apples : ℕ := 64
def baskets : ℕ := 4
def apples_taken_per_basket : ℕ := 3

-- Theorem statement based on the question and correct answer
theorem apples_in_each_basket (h1 : total_apples = 64) 
                              (h2 : baskets = 4) 
                              (h3 : apples_taken_per_basket = 3) : 
    (total_apples / baskets - apples_taken_per_basket) = 13 := 
by
  sorry

end apples_in_each_basket_l86_86738


namespace find_vertex_angle_l86_86612

noncomputable def vertex_angle_cone (α : ℝ) : ℝ :=
  2 * Real.arcsin (α / (2 * Real.pi))

theorem find_vertex_angle (α : ℝ) :
  ∃ β : ℝ, β = vertex_angle_cone α :=
by
  use 2 * Real.arcsin (α / (2 * Real.pi))
  refl

end find_vertex_angle_l86_86612


namespace train_cross_platform_time_l86_86094

noncomputable def timeToCrossPlatform (length_train length_platform speed_train : ℕ) : ℝ :=
  (length_train + length_platform) / speed_train

theorem train_cross_platform_time :
  ∀ (length_train time_signal length_platform : ℕ),
    time_signal ≠ 0 →
    let speed_train := (length_train : ℝ) / time_signal in
    length_train = 300 →
    time_signal = 18 →
    length_platform = 500 →
    timeToCrossPlatform length_train length_platform speed_train = 800 / (300 / 18) :=
by 
  intros length_train time_signal length_platform ht hs hlt htp
  rw [hs, hlt, htp]
  norm_num
  exact rfl

end train_cross_platform_time_l86_86094


namespace find_matrix_M_l86_86213

theorem find_matrix_M (M : Matrix (Fin 3) (Fin 3) ℝ)
  (h : ∀ (N : Matrix (Fin 3) (Fin 3) ℝ), 
         M ⬝ N = 
         ![N 2 0, N 2 1, N 2 2; 
           N 0 0 + N 1 0, N 0 1 + N 1 1, N 0 2 + N 1 2;
           N 0 0, N 0 1, N 0 2]) :
  M = ![![0, 0, 1], ![1, 1, 0], ![1, 0, 0]] :=
sorry

end find_matrix_M_l86_86213


namespace problem_statement_l86_86409

open Nat

noncomputable def no_rational_roots (n : ℕ) (hn : n > 1) : Prop :=
  ∀ x : ℚ, (bigOperators.sum (range (n + 1)) (λ (k : ℕ), x^(n - k) / (k! : ℚ)) = -1) → False

theorem problem_statement (n : ℕ) (hn : n > 1) : no_rational_roots n hn :=
sorry

end problem_statement_l86_86409


namespace solve_for_x_l86_86506

theorem solve_for_x (x : ℝ) : (2010 + 2 * x) ^ 2 = x ^ 2 → x = -2010 ∨ x = -670 := by
  sorry

end solve_for_x_l86_86506


namespace exists_polynomial_triangle_property_l86_86525

noncomputable def f (x y z : ℝ) : ℝ :=
  (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

theorem exists_polynomial_triangle_property :
  ∀ (x y z : ℝ), (f x y z > 0 ↔ (|x| + |y| > |z| ∧ |y| + |z| > |x| ∧ |z| + |x| > |y|)) :=
sorry

end exists_polynomial_triangle_property_l86_86525


namespace solve_for_y_l86_86796

theorem solve_for_y : ∀ (y : ℝ), (3 / 4 - 5 / 8 = 1 / y) → y = 8 :=
by
  intros y h
  sorry

end solve_for_y_l86_86796


namespace find_a_find_t_analyze_roots_l86_86269

-- Step 1: Prove that a = 0
theorem find_a (f : ℝ → ℝ) (a : ℝ) (h_odd : ∀ x, f (-x) = -f x) : a = 0 :=
sorry

-- Step 2: Prove the range of t given the provided inequalities and decreasing function
theorem find_t (λ : ℝ) (h_decreasing : ∀ x ∈ set.Icc (-1 : ℝ) 1, λ + real.cos x ≤ 0)
  (h_inequality : ∀ x ∈ set.Icc (-1 : ℝ) 1, λ * x + real.sin x ≤ x^2 + λ * x + 1) : ∀ t : ℝ, t ≤ -1 :=
sorry

-- Step 3: Analyze the number of roots of the given equation with respect to m
theorem analyze_roots (m : ℝ) :
  (m > float.pi^2 + float.exp 1⁻¹) →
  (m = float.pi^2 + float.exp 1⁻¹) →
  (m < float.pi^2 + float.exp 1⁻¹) →
  ∃ nRoots : ℕ, nRoots = 
    if m > float.pi^2 + float.exp 1⁻¹ then 0
    else if m = float.pi^2 + float.exp 1⁻¹ then 1
    else 2 :=
sorry

end find_a_find_t_analyze_roots_l86_86269


namespace smartphone_price_difference_l86_86848

variable (priceA : ℝ) (discountA : ℝ) (priceB : ℝ) (discountB : ℝ)
variable (finalPriceA : ℝ) (finalPriceB : ℝ)
variable (difference : ℝ)

def discount_amount (price : ℝ) (discount_rate : ℝ) : ℝ := price * discount_rate

theorem smartphone_price_difference:
  priceA = 125 ∧ discountA = 0.08 ∧ priceB = 130 ∧ discountB = 0.10 →
  finalPriceA = priceA - discount_amount priceA discountA →
  finalPriceB = priceB - discount_amount priceB discountB →
  difference = finalPriceB - finalPriceA →
  difference = 2 :=
by {
  intros,
  simp at *,
  sorry
}

end smartphone_price_difference_l86_86848


namespace money_left_after_expenses_l86_86110

theorem money_left_after_expenses :
  let salary := 8123.08
  let food_expense := (1:ℝ) / 3 * salary
  let rent_expense := (1:ℝ) / 4 * salary
  let clothes_expense := (1:ℝ) / 5 * salary
  let total_expense := food_expense + rent_expense + clothes_expense
  let money_left := salary - total_expense
  money_left = 1759.00 :=
sorry

end money_left_after_expenses_l86_86110


namespace victor_won_games_l86_86494

theorem victor_won_games (V : ℕ) (ratio_victor_friend : 9 * 20 = 5 * V) : V = 36 :=
sorry

end victor_won_games_l86_86494


namespace ratio_first_term_l86_86553

theorem ratio_first_term (x : ℕ) (r : ℕ × ℕ) (h₀ : r = (6 - x, 7 - x)) 
        (h₁ : x ≥ 3) (h₂ : r.1 < r.2) : r.1 < 4 :=
by
  sorry

end ratio_first_term_l86_86553


namespace exponent_division_l86_86635

variable (a : ℝ) (m n : ℝ)
-- Conditions
def condition1 : Prop := a^m = 2
def condition2 : Prop := a^n = 16

-- Theorem Statement
theorem exponent_division (h1 : condition1 a m) (h2 : condition2 a n) : a^(m - n) = 1 / 8 := by
  sorry

end exponent_division_l86_86635


namespace num_polynomials_l86_86640

theorem num_polynomials : 
  ∃ count : ℕ, count = 5 ∧ ∀ (P : ℕ → ℤ) (n : ℕ), (n + |P 0| + |P (1 : ℕ)| + ... + |P n| = 3) → count = 5 :=
sorry

end num_polynomials_l86_86640


namespace odd_integers_between_3000_and_6000_l86_86678

theorem odd_integers_between_3000_and_6000 :
  let is_odd (n : ℕ) := n % 2 = 1
  ∧ ∀ n, 3000 ≤ n ∧ n < 6000 → is_odd n ∧ ∀ i j k, (i ≠ j ∧ j ≠ k ∧ k ≠ i)
  → ∃ d1 d2 d3 d4, n = 1000 * d1 + 100 * d2 + 10 * d3 + d4 
  → n ∈ finset.range 3000 6000
  → finset.card n = 840 := 
begin
  sorry
end

end odd_integers_between_3000_and_6000_l86_86678


namespace eval_ceil_floor_sum_l86_86209

theorem eval_ceil_floor_sum :
  (Int.ceil (7 / 3) + Int.floor (- (7 / 3))) = 0 :=
by
  sorry

end eval_ceil_floor_sum_l86_86209


namespace angle_B_is_pi_over_3_perimeter_of_triangle_l86_86350

-- Define the triangle with sides and angles
variables {A B C a b c : ℝ}
variables {BD : ℝ} -- BD is the altitude from B to AC

-- Given conditions
axiom angle_B_condition : c * sin ((A + C) / 2) = b * sin C
axiom BD_condition : BD = 1
axiom side_b_condition : b = sqrt 3

-- The statements to prove
theorem angle_B_is_pi_over_3 (h₁ : angle_B_condition) : B = π / 3 := by
  sorry

theorem perimeter_of_triangle (h₁ : angle_B_condition) (h₂ : BD_condition) (h₃ : side_b_condition) : (a + b + c) = 3 + sqrt 3 := by
  sorry

end angle_B_is_pi_over_3_perimeter_of_triangle_l86_86350


namespace arithmetic_expression_l86_86496

theorem arithmetic_expression :
  7 / 2 - 3 - 5 + 3 * 4 = 7.5 :=
by {
  -- We state the main equivalence to be proven
  sorry
}

end arithmetic_expression_l86_86496


namespace all_Garbs_are_Morks_and_Plogs_l86_86709

variables (Mork Plog Snark Garb : Type) 
variable (M : Mork → Plog)
variable (S : Snark → Mork)
variable (G : Garb → Plog)
variable (G_S : Garb → Snark)

theorem all_Garbs_are_Morks_and_Plogs : (Garb → Mork) ∧ (Garb → Plog) :=
by
  split
  { intro g
    exact S (G_S g) }
  { exact G }

end all_Garbs_are_Morks_and_Plogs_l86_86709


namespace jori_water_left_l86_86740

theorem jori_water_left (initial_water : ℚ) (used_water : ℚ) : initial_water = 3 ∧ used_water = 5/4 → initial_water - used_water = 7/4 :=
by {
  intro h,
  cases h with h1 h2,
  rw [h1, h2],
  norm_num,
}

end jori_water_left_l86_86740


namespace rhombus_ratio_l86_86118

-- Definitions given in the problem
structure Rhombus where
  short_diagonal : ℝ
  long_diagonal : ℝ

def rotated_rhombus (r : Rhombus) (angle : ℝ) : Rhombus := r

def union (r1 r2 : Rhombus) : ℝ := sorry
def intersection (r1 r2 : Rhombus) : ℝ := sorry

-- Conditions for our specific problem
def R : Rhombus := ⟨1, 2023⟩
def R' : Rhombus := rotated_rhombus R (π / 2)

-- Lean statement to express the ratio proof
theorem rhombus_ratio :
  let U := union R R'
  let I := intersection R R'
  I / U = 1 / 2023 :=
sorry

end rhombus_ratio_l86_86118


namespace shoes_selection_l86_86230

theorem shoes_selection : 
  (∃ (n m : ℕ), n = 5 ∧ m = 4 ∧ 
    ∑ (x in finset.range (n - m + 1)), nat.choose (n - x) m * (nat.choose (4 - m) 2) * (nat.choose 2 1) * (nat.choose 2 1) = 120) :=
begin
  sorry
end

end shoes_selection_l86_86230


namespace possible_values_l86_86649

def seq_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = 2 * a (n + 2) * a (n + 3) + 2016

theorem possible_values (a : ℕ → ℤ) (h : seq_condition a) :
  (a 1, a 2) = (0, 2016) ∨
  (a 1, a 2) = (-14, 70) ∨
  (a 1, a 2) = (-69, 15) ∨
  (a 1, a 2) = (-2015, 1) ∨
  (a 1, a 2) = (2016, 0) ∨
  (a 1, a 2) = (70, -14) ∨
  (a 1, a 2) = (15, -69) ∨
  (a 1, a 2) = (1, -2015) :=
sorry

end possible_values_l86_86649


namespace count_positive_integers_N_number_of_N_l86_86291

theorem count_positive_integers_N : ∀ N : ℕ, N < 2000 → ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N :=
begin
  sorry
end

theorem number_of_N : {N : ℕ // N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x ^ (Real.floor x) = N}.card = 412 :=
begin
  sorry
end

end count_positive_integers_N_number_of_N_l86_86291


namespace count_valid_N_l86_86296

theorem count_valid_N : 
  ∃ (N : ℕ), (N < 2000) ∧ (∃ (x : ℝ), x^⌊x⌋₊ = N) :=
begin
  sorry
end

end count_valid_N_l86_86296


namespace angle_B_is_pi_over_3_perimeter_of_triangle_l86_86349

-- Define the triangle with sides and angles
variables {A B C a b c : ℝ}
variables {BD : ℝ} -- BD is the altitude from B to AC

-- Given conditions
axiom angle_B_condition : c * sin ((A + C) / 2) = b * sin C
axiom BD_condition : BD = 1
axiom side_b_condition : b = sqrt 3

-- The statements to prove
theorem angle_B_is_pi_over_3 (h₁ : angle_B_condition) : B = π / 3 := by
  sorry

theorem perimeter_of_triangle (h₁ : angle_B_condition) (h₂ : BD_condition) (h₃ : side_b_condition) : (a + b + c) = 3 + sqrt 3 := by
  sorry

end angle_B_is_pi_over_3_perimeter_of_triangle_l86_86349


namespace total_dogs_l86_86955

def number_of_boxes : ℕ := 15
def dogs_per_box : ℕ := 8

theorem total_dogs : number_of_boxes * dogs_per_box = 120 := by
  sorry

end total_dogs_l86_86955


namespace least_four_digit_multiple_3_5_7_l86_86065

theorem least_four_digit_multiple_3_5_7 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ n = 1050 :=
by
  use 1050
  repeat {sorry}

end least_four_digit_multiple_3_5_7_l86_86065


namespace quiz_answer_key_combinations_l86_86079

theorem quiz_answer_key_combinations :
  let tf_combinations := 2 ^ 4,
      valid_tf_combinations := tf_combinations - 2,
      mcq_combinations := 4 ^ 2,
      total_combinations := valid_tf_combinations * mcq_combinations
  in total_combinations = 224 := sorry

end quiz_answer_key_combinations_l86_86079


namespace length_of_PR_l86_86721

/-- 
Mathematically equivalent proof problem:
Prove that the length of diagonal \( PR \) is \( \sqrt{251} \) meters given:
1. The total area of the four right-angled triangles cut off is \( 250 \text{ m}^2 \).
2. The side length of square \(ABCD\) is 1 meter.
-/

theorem length_of_PR 
  (area_cut_off : ℝ := 250)
  (side_length : ℝ := 1)
  (hx_pos : 0 ≤ x)
  (hy_pos : 0 ≤ y) :
  (let x := (side_length * side_length - 2 * area_cut_off) / side_length,
       y := (side_length * side_length - 2 * area_cut_off) / side_length 
   in x + y) = real.sqrt 251 :=
by
  sorry

end length_of_PR_l86_86721


namespace symmetry_of_f_transforms_l86_86626

theorem symmetry_of_f_transforms (f : ℝ → ℝ) :
  ∀ x, f(x-1) = f(-(x-1)) ↔ f(x-1) = f(-x+1) :=
by
  sorry

end symmetry_of_f_transforms_l86_86626


namespace number_of_ways_to_choose_teams_l86_86226

theorem number_of_ways_to_choose_teams : 
  ∃ (n : ℕ), n = Nat.choose 5 2 ∧ n = 10 :=
by
  have h : Nat.choose 5 2 = 10 := by sorry
  use 10
  exact ⟨h, rfl⟩

end number_of_ways_to_choose_teams_l86_86226


namespace DanteSoldCoconuts_l86_86407

variable (Paolo_coconuts : ℕ) (Dante_coconuts : ℕ) (coconuts_left : ℕ)

def PaoloHasCoconuts := Paolo_coconuts = 14

def DanteHasThriceCoconuts := Dante_coconuts = 3 * Paolo_coconuts

def DanteLeftCoconuts := coconuts_left = 32

theorem DanteSoldCoconuts 
  (h1 : PaoloHasCoconuts Paolo_coconuts) 
  (h2 : DanteHasThriceCoconuts Paolo_coconuts Dante_coconuts) 
  (h3 : DanteLeftCoconuts coconuts_left) : 
  Dante_coconuts - coconuts_left = 10 := 
by
  rw [PaoloHasCoconuts, DanteHasThriceCoconuts, DanteLeftCoconuts] at *
  sorry

end DanteSoldCoconuts_l86_86407


namespace right_trapezoid_perimeter_l86_86712

def is_right_trapezoid_volume (a b h : ℝ) : Prop :=
  (a < b) ∧ 
  (π * h^2 * a + (1/3) * π * h^2 * (b - a) = 80 * π) ∧
  (π * h^2 * b + (1/3) * π * h^2 * (b - a) = 112 * π) ∧
  ((1/3) * π * (a^2 + a * b + b^2) * h = 156 * π)

noncomputable def perimeter_of_right_trapezoid (a b h : ℝ) : ℝ :=
  a + b + h + real.sqrt(h^2 + (b - a)^2)

theorem right_trapezoid_perimeter
  (a b h : ℝ) (h_cond : is_right_trapezoid_volume a b h) :
  perimeter_of_right_trapezoid a b h = 20 + 2 * real.sqrt 13 := by
  sorry

end right_trapezoid_perimeter_l86_86712


namespace digit_in_numeral_46_place_face_value_difference_is_36_l86_86499

def place_value (n : ℕ) (digit : ℕ) : ℕ := 
  if digit = 4 then 40 else if digit = 6 then 6 else 0

def face_value (digit : ℕ) : ℕ := digit

theorem digit_in_numeral_46_place_face_value_difference_is_36 :
  ∃ digit : ℕ, digit ∈ {4, 6} ∧ place_value 46 digit - face_value digit = 36 :=
by
  sorry

end digit_in_numeral_46_place_face_value_difference_is_36_l86_86499


namespace find_period_and_extrema_l86_86662

def f (x : ℝ) : ℝ := cos x ^ 2 - sin x ^ 2 + 2 * real.sqrt 3 * sin x * cos x

theorem find_period_and_extrema :
  (∀ x, f (x + π) = f x) ∧
  (∀ x, x ∈ Ico (0 : ℝ) (π / 4) → 1 ≤ f x ∧ f x ≤ 2) := by
sorry

end find_period_and_extrema_l86_86662


namespace max_angle_sum_l86_86801

theorem max_angle_sum (n : ℕ) (lines : Finset ℝ)
  (h1 : ∀ (a b : ℝ), a ∈ lines → b ∈ lines → a ≠ b)
  (h2: ∀ (a b c : ℝ), a ∈ lines → b ∈ lines → c ∈ lines → (a ≠ b → b ≠ c → a ≠ c))
  (h3: ∀ (a b : ℝ), a ∈ lines → b ∈ lines → 0 ≤ a.angle b ∧ a.angle b ≤ π / 2) :
  sum (λ (i : Finset.Choose 2 lines), angle i.1 i.2) ≤ 
  (π / 2) * ⌊ (n / 2 : ℝ) ⌋ * ( ⌊ (n / 2 : ℝ) ⌋ + 1  ) := 
sorry

end max_angle_sum_l86_86801


namespace linear_function_m_value_l86_86694

theorem linear_function_m_value (m : ℝ) :
  (∃ (f : ℝ → ℝ), f = (λ x, (m - 3) * x ^ (4 - |m|) + m + 7) ∧ 
   (∀ x, f x = a * x + b → (m - 3 ≠ 0) ∧ (4 - |m| = 1))) → 
  m = -3 :=
by
  intro h
  sorry

end linear_function_m_value_l86_86694


namespace optimal_worker_distribution_l86_86826
noncomputable def forming_time (n1 : ℕ) : ℕ := (nat.ceil (75.0 / n1) : ℕ) * 15
noncomputable def firing_time (n3 : ℕ) : ℕ := (nat.ceil (75.0 / n3) : ℕ) * 30

theorem optimal_worker_distribution:
  ∃ n1 n3 : ℕ, n1 + n3 = 13 ∧ (forming_time n1 = 225 ∨ firing_time n3 = 330) :=
sorry

end optimal_worker_distribution_l86_86826


namespace max_M_l86_86593

theorem max_M (S : ℝ) (a : ℝ) (ha : a > 0) (hS : S > 0) (hM : ∀ a > 0, 2*a + 2*(S/a) = p):
  (¬M ∃ M, M = S / (2*S + 2*a + 2*(S/a) + 2)) ≤ S / (2 * (Real.sqrt S + 1) ^ 2) := 
sorry

end max_M_l86_86593


namespace find_principal_amount_l86_86076

noncomputable def simple_interest_proof : Prop :=
  ∃ (P R : ℝ), 
    (let SI₁ := (P * R * 10) / 100 in
     let SI₂ := (P * (R + 5) * 10) / 100 in
     SI₂ = SI₁ + 400 ∧ P = 800)

theorem find_principal_amount : simple_interest_proof :=
by sorry

end find_principal_amount_l86_86076


namespace ellipse_properties_l86_86260

-- Define the ellipse E with its given properties
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define properties related to the intersection points and lines
def intersects (l : ℝ → ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  l (-1) = 0 ∧ 
  is_ellipse x₁ (l x₁) ∧ 
  is_ellipse x₂ (l x₂) ∧ 
  y₁ = l x₁ ∧ 
  y₂ = l x₂

def perpendicular_lines (l1 l2 : ℝ → ℝ) : Prop :=
  ∀ x, l1 x * l2 x = -1

-- Define the main theorem
theorem ellipse_properties :
  (∀ (x y : ℝ), is_ellipse x y) →
  (∀ (l1 l2 : ℝ → ℝ) 
     (A B C D : ℝ × ℝ),
      intersects l1 A.1 A.2 B.1 B.2 → 
      intersects l2 C.1 C.2 D.1 D.2 → 
      perpendicular_lines l1 l2 → 
      12 * (|A.1 - B.1| + |C.1 - D.1|) = 7 * |A.1 - B.1| * |C.1 - D.1|) :=
by 
  sorry

end ellipse_properties_l86_86260


namespace hexagon_interior_angles_sum_l86_86469

theorem hexagon_interior_angles_sum :
  let n := 6 in
  (n - 2) * 180 = 720 :=
by
  sorry

end hexagon_interior_angles_sum_l86_86469


namespace angle_bisectors_intersect_on_segment_l86_86050

open EuclideanGeometry

variables {P : Type*} [MetricSpace P] [normed_space ℝ P] [inner_product_space ℝ P] [finite_dimensional ℝ P] 
variables {ω₁ ω₂ : Circle P} 
variables {A B C D : P} 

theorem angle_bisectors_intersect_on_segment 
  (h_inter : ω₁.intersections ω₂ = {A, B}) 
  (h_line : line_through A C ∩ ω₁ = {C}) 
  (h_line2 : line_through A D ∩ ω₂ = {D})
  (h_between : between C A D)
  (h_proportional : ∃ k : ℝ, dist A C = k * ω₁.radius ∧ dist A D = k * ω₂.radius) :
  ∃ E : P, on_segment A B E ∧ angle_bisector A B E D = angle_bisector A B E C :=
begin
  sorry
end

end angle_bisectors_intersect_on_segment_l86_86050


namespace sqrt_sum_gt_l86_86833

theorem sqrt_sum_gt (a b : ℝ) (ha : a = 2) (hb : b = 3) : 
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by 
  sorry

end sqrt_sum_gt_l86_86833


namespace area_ratio_ABG_AGC_angle_GCA_if_AGC_90_l86_86732

variables {A B C E F G : Type}
variables [midpoint E A B] [midpoint F B C] [on_segment G E F] [ratio EG AE (1 : 2)] [distance FG BE]

-- (a) Ratios of the areas
theorem area_ratio_ABG_AGC (ratio_ABG_AGC : ℝ)
  (h1 : midpoint E A B)
  (h2 : midpoint F B C)
  (h3 : on_segment G E F)
  (h4 : ratio EG AE (1 : 2))
  (h5 : distance FG BE)
  (h6 : triangle A B G)
  (h7 : triangle A G C) :
  ratio_of_areas (triangle_area ABG) (triangle_area AGC) = 1 / 3 :=
sorry

-- (b) Angle GCA when AGC = 90 degrees
theorem angle_GCA_if_AGC_90 (angle_GCA : ℝ)
  (h1 : midpoint E A B)
  (h2 : midpoint F B C)
  (h3 : on_segment G E F)
  (h4 : ratio EG AE (1 : 2))
  (h5 : distance FG BE)
  (h6 : angle AGC = 90) :
  angle GCA = arcsin (1 / (2 * sqrt 2)) :=
sorry

end area_ratio_ABG_AGC_angle_GCA_if_AGC_90_l86_86732


namespace tortoise_speed_l86_86789

theorem tortoise_speed :
  ∀ (d t  : ℕ) (r_s rabbit_rest turtle_cross_time : ℚ),
    d = 2640 →
    r_s = 36 →
    rabbit_rest = ∑ k in (range 24).map (λ k, (k+1) • (1 / 2 : ℚ)) →
    turtle_cross_time = (d / r_s) + rabbit_rest - (10 / 3) →
    (d / turtle_cross_time) = 12 :=
by
  intros d t r_s rabbit_rest turtle_cross_time 
  assume h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end tortoise_speed_l86_86789


namespace Tim_total_payment_l86_86482

-- Define the context for the problem
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.3

-- Define the total amount paid as the sum of the manicure cost and the tip
def total_amount_paid (cost : ℝ) (tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- The theorem to be proven
theorem Tim_total_payment : total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end Tim_total_payment_l86_86482


namespace common_point_any_two_triangles_l86_86443

-- Definitions based on the conditions in the problem.
def nondegenerate_triangle (A B C : Point) : Prop := 
  -- Define the predicate for a non-degenerate triangle, could involve checking the collinearity of the points, etc.
  sorry

def general_position (A B C D : Point) : Prop := 
  -- Define the predicate for points in general position 
  sorry

-- The function assigning circles to non-degenerate triangles
noncomputable def f (T : Triangle) : Circle := 
  -- Define a function which we assume assigns a circle to a triangle
  sorry

-- Hypothesis provided in the conditions
axiom common_point_four_points_in_general_position 
  (A B C D : Point) 
  (hA : nondegenerate_triangle A B C) 
  (hB : nondegenerate_triangle B C D) 
  (hC : nondegenerate_triangle C D A) 
  (hD : nondegenerate_triangle D A B) 
  (h_gen : general_position A B C D) : 
  ∃ P : Point, P ∈ f(triangle A B C) ∧ P ∈ f(triangle B C D) ∧ P ∈ f(triangle C D A) ∧ P ∈ f(triangle D A B)

-- The theorem statement we aim to prove
theorem common_point_any_two_triangles 
  (T1 T2 : Triangle) 
  (hT1 : nondegenerate_triangle T1.A T1.B T1.C) 
  (hT2 : nondegenerate_triangle T2.A T2.B T2.C) : 
  ∃ P : Point, P ∈ f(T1) ∧ P ∈ f(T2) := 
  sorry

end common_point_any_two_triangles_l86_86443


namespace probability_of_cosine_range_l86_86538

noncomputable def probability_cos_in_intervals : ℝ :=
  let total_length := 2 * Real.pi in
  let interval_length := (5 * Real.pi / 6 - Real.pi / 6) + (Real.pi / 6 - -5 * Real.pi / 6) in
  interval_length / total_length

theorem probability_of_cosine_range :
  probability_cos_in_intervals = 2 / 3 :=
by
  sorry

end probability_of_cosine_range_l86_86538


namespace inequality_of_sum_l86_86523

theorem inequality_of_sum (n : ℕ) (hn : 2 ≤ n) 
  (x y : Fin n → ℝ) (hx_sorted : ∀ i j, i ≤ j → x i ≥ x j) (hy_sorted : ∀ i j, i ≤ j → y i ≥ y j)
  (hx_sum_zero : ∑ i, x i = 0) (hy_sum_zero : ∑ i, y i = 0)
  (hx_sum_squares_one : ∑ i, (x i)^2 = 1) (hy_sum_squares_one : ∑ i, (y i)^2 = 1) :
  ∑ i, (x i * y i - x i * y (n+1-i)) ≥ 2 / Real.sqrt (n - 1) :=
by
  -- proof omitted
  sorry

end inequality_of_sum_l86_86523


namespace sum_of_ages_is_correct_l86_86393

-- Definitions based on conditions
def Masc_age := 17
def Sam_age := 10
def age_difference := Masc_age - Sam_age

-- Assertion statement to prove the sum of their ages is 27
theorem sum_of_ages_is_correct : Masc_age + Sam_age = 27 :=
by
  -- Definitions must be used directly from the conditions given in a)
  let h1 := age_difference
  have h2 : Masc_age = Sam_age + 7 := by
    rw [Masc_age, Sam_age]
    exact rfl
  
  have h3 : Masc_age + Sam_age = 10 + 17 := by
    rw [Masc_age, Sam_age]

  exact h3

end sum_of_ages_is_correct_l86_86393


namespace mary_total_nickels_l86_86392

theorem mary_total_nickels (n1 n2 : ℕ) (h1 : n1 = 7) (h2 : n2 = 5) : n1 + n2 = 12 := by
  sorry

end mary_total_nickels_l86_86392


namespace smallest_angle_BFE_is_113_l86_86569

noncomputable def incenter (A B C : Type*) : Type := sorry

variables {A B C D E F : Type*}

-- Definitions of incenters
def is_incenter_D : incenter A B C := sorry
def is_incenter_E : incenter A B D := sorry
def is_incenter_F : incenter B D E := sorry

-- Main theorem to prove
theorem smallest_angle_BFE_is_113 
  (hD : is_incenter_D) 
  (hE : is_incenter_E) 
  (hF : is_incenter_F) : 
  ∃ n : ℤ, n = 113 ∧ ∃ θ_BFE : ℤ, θ_BFE = n := 
sorry

end smallest_angle_BFE_is_113_l86_86569


namespace pairings_correct_l86_86881

-- Define the participants
structure Person := (name : String) (is_boy : Bool)

-- Define the participants as constants
def Yura : Person := ⟨"Yura Vorobyev", true⟩
def Andrey : Person := ⟨"Andrey Egorov", true⟩
def Seryozha : Person := ⟨"Seryozha Petrov", true⟩
def Dima : Person := ⟨"Dima Krymov", true⟩
def Lyusya : Person := ⟨"Lyusya Egorova", false⟩
def Olya : Person := ⟨"Olya Petrova", false⟩
def Inna : Person := ⟨"Inna Krymova", false⟩
def Anya : Person := ⟨"Anya Vorobyeva", false⟩

-- Define an ordering over the participants by height
def height_order : List Person := [Yura, Andrey, Lyusya, Seryozha, Olya, Dima, Inna, Anya]

-- Define pairs
structure Pair := (gentleman : Person) (lady : Person)

-- Conditions
axiom taller (p1 p2 : Person) : p1 ∈ height_order → p2 ∈ height_order → p1.is_boy = true → p2.is_boy = false → height_order.indexOf p1 < height_order.indexOf p2

axiom not_siblings (p1 p2 : Person) : p1 ∈ height_order → p2 ∈ height_order → p1.name.split.last (λ a => True) != p2.name.split.last (λ a => True)

-- Define the specific pairings to be proven
def Lyusya_Yura_Pair : Pair := ⟨Yura, Lyusya⟩
def Olya_Andrey_Pair : Pair := ⟨Andrey, Olya⟩
def Inna_Seryozha_Pair : Pair := ⟨Seryozha, Inna⟩
def Anya_Dima_Pair : Pair := ⟨Dima, Anya⟩

-- The Lean statement to prove the pairs
theorem pairings_correct :
  ∃ pairs : List Pair,
    Lyusya_Yura_Pair ∈ pairs ∧
    Olya_Andrey_Pair ∈ pairs ∧
    Inna_Seryozha_Pair ∈ pairs ∧
    Anya_Dima_Pair ∈ pairs ∧
    ∀ (p : Pair), p ∈ pairs → taller p.gentleman p.lady ∧ not_siblings p.gentleman p.lady
:= sorry

end pairings_correct_l86_86881


namespace probability_club_then_spade_l86_86845

/--
   Two cards are dealt at random from a standard deck of 52 cards.
   Prove that the probability that the first card is a club (♣) and the second card is a spade (♠) is 13/204.
-/
theorem probability_club_then_spade :
  let total_cards := 52
  let clubs := 13
  let spades := 13
  let first_card_club_prob := (clubs : ℚ) / total_cards
  let second_card_spade_prob := (spades : ℚ) / (total_cards - 1)
  first_card_club_prob * second_card_spade_prob = 13 / 204 :=
by
  sorry

end probability_club_then_spade_l86_86845


namespace socks_cost_5_l86_86676

theorem socks_cost_5
  (jeans t_shirt socks : ℕ)
  (h1 : jeans = 2 * t_shirt)
  (h2 : t_shirt = socks + 10)
  (h3 : jeans = 30) :
  socks = 5 :=
by
  sorry

end socks_cost_5_l86_86676


namespace minimize_cost_l86_86098

-- define volume, depth, and costs
def volume : ℝ := 300
def depth : ℝ := 3
def cost_bottom : ℝ := 120
def cost_walls : ℝ := 100

-- define the objective function (total cost)
def total_cost (x y : ℝ) : ℝ := 
  cost_bottom * (volume / depth) + cost_walls * (2 * depth * x + 2 * depth * y)

-- define the volume condition
def volume_condition (x y : ℝ) : Prop := x * y = volume / depth

-- define the proof statement
theorem minimize_cost :
  ∃ x y, volume_condition x y ∧ total_cost x y = 24000 :=
sorry

end minimize_cost_l86_86098


namespace rectangle_side_equality_l86_86697

-- Define the sides of the rectangles
variables {a1 b1 a2 b2 : ℝ}

theorem rectangle_side_equality (h_perimeter : 2 * (a1 + b1) = 2 * (a2 + b2))
    (h_area : a1 * b1 = a2 * b2) : {a1, b1} = {a2, b2} :=
by
  sorry

end rectangle_side_equality_l86_86697


namespace loss_percentage_is_26_l86_86431

/--
Given the cost price of a radio is Rs. 1500 and the selling price is Rs. 1110, 
prove that the loss percentage is 26%
-/
theorem loss_percentage_is_26 (cost_price selling_price : ℝ)
  (h₀ : cost_price = 1500)
  (h₁ : selling_price = 1110) :
  ((cost_price - selling_price) / cost_price) * 100 = 26 := 
by 
  sorry

end loss_percentage_is_26_l86_86431


namespace circuit_disconnected_scenarios_l86_86132

def num_scenarios_solder_points_fall_off (n : Nat) : Nat :=
  2 ^ n - 1

theorem circuit_disconnected_scenarios : num_scenarios_solder_points_fall_off 6 = 63 :=
by
  sorry

end circuit_disconnected_scenarios_l86_86132


namespace number_of_people_with_cards_greater_than_0p3_l86_86367

theorem number_of_people_with_cards_greater_than_0p3 :
  (∃ (number_of_people : ℕ),
     number_of_people = (if 0.3 < 0.8 then 1 else 0) +
                        (if 0.3 < (1 / 2) then 1 else 0) +
                        (if 0.3 < 0.9 then 1 else 0) +
                        (if 0.3 < (1 / 3) then 1 else 0)) →
  number_of_people = 4 :=
by
  sorry

end number_of_people_with_cards_greater_than_0p3_l86_86367


namespace mike_earnings_l86_86397

def prices : List ℕ := [5, 7, 12, 9, 6, 15, 11, 10]

theorem mike_earnings :
  List.sum prices = 75 :=
by
  sorry

end mike_earnings_l86_86397


namespace find_length_of_side_a_l86_86334

-- Define the conditions in Lean
def angle_sum_property (A B C : ℝ) : Prop :=
  A + B + C = 180

def sin_15_degrees : Real :=
  (Real.sin (π / 12))

-- Lean function definition for our problem
theorem find_length_of_side_a (b : ℝ) (B C : ℝ) (hB : B = π / 6) (hC : C = 3 * π / 4) (h_b : b = 2) :
  ∃ a : ℝ, a = √6 - √2 :=
by
  let A := π - B - C
  have hA : A = π / 12 := by
    calc
      A = π - B - C     : rfl
      _ = π - (π / 6) - (3 * π / 4) : by rw [hB, hC]
      _ = π / 12 : by sorry -- This will calculate to π / 12 correctly
  
  have h_sin_A : Real.sin A = sin_15_degrees := by
    sorry -- This will use the known formula for sin 15 degrees

  have H_law_of_sines : a = (b * (Real.sin A)) / (Real.sin B) := by
    sorry -- Apply the law of sines

  have H_a : a = √6 - √2 := by
    sorry -- Derived from substitution and simplifying

  existsi √6 - √2
  exact H_a

end find_length_of_side_a_l86_86334


namespace symmetric_graph_ln_l86_86261

def f (x : ℝ) : ℝ := Real.exp x

theorem symmetric_graph_ln (c : ℝ) (h : ∀ x : ℝ, f x = Real.exp x) : f c = Real.exp c := by
  sorry

example : f 2 = Real.exp 2 :=
  symmetric_graph_ln 2 (by intros; refl)

end symmetric_graph_ln_l86_86261


namespace M_greater_than_N_l86_86995

variable (a : ℝ)

def M := 2 * a^2 - 4 * a
def N := a^2 - 2 * a - 3

theorem M_greater_than_N : M a > N a := by
  sorry

end M_greater_than_N_l86_86995


namespace number_of_possible_p_values_l86_86592

noncomputable def integer_zero (a : ℤ) (p : ℤ) : Prop :=
  ∃ b c : ℚ, (b ≠ c) ∧ (b ≠ a - b - c) ∧ (c ≠ a - b - c) ∧ 
  (a = b + c + (a - b - c)) ∧ 
  (a = 2510) ∧
  (p = a * b * c * (a - b - c))

theorem number_of_possible_p_values : 
  (∑ p in set_of integer_zero, 1).card = 6299599 := 
sorry

end number_of_possible_p_values_l86_86592


namespace boys_variance_greater_than_girls_l86_86901

/-- Define the given scores for boys and girls. --/
def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

/-- Function to calculate the mean of a list of real numbers. --/
def mean (scores : List ℝ) : ℝ := (scores.sum) / (scores.length)

/-- Function to calculate the variance of a list of real numbers. --/
def variance (scores : List ℝ) : ℝ :=
  let avg := mean scores
  (scores.map (λ x => (x - avg) ^ 2)).sum / (scores.length)

/-- Prove that the variance of the boys' scores is greater than the variance of the girls' scores. --/
theorem boys_variance_greater_than_girls :
  variance boys_scores > variance girls_scores :=
  sorry

end boys_variance_greater_than_girls_l86_86901


namespace solve_problem_l86_86798

open Real

noncomputable def problem (x : ℝ) : Prop :=
  (cos (2 * x / 5) - cos (2 * π / 15)) ^ 2 + (sin (2 * x / 3) - sin (4 * π / 9)) ^ 2 = 0

theorem solve_problem : ∀ t : ℤ, problem ((29 * π / 3) + 15 * π * t) :=
by
  intro t
  sorry

end solve_problem_l86_86798


namespace prob_point_closer_to_six_than_zero_l86_86114

theorem prob_point_closer_to_six_than_zero : 
  let interval_start := 0
  let interval_end := 7
  let closer_to_six := fun x => x > ((interval_start + 6) / 2)
  let total_length := interval_end - interval_start
  let length_closer_to_six := interval_end - (interval_start + 6) / 2
  total_length > 0 -> length_closer_to_six / total_length = 4 / 7 :=
by
  sorry

end prob_point_closer_to_six_than_zero_l86_86114


namespace johns_phone_price_l86_86918

-- Define Alan's phone price
def alans_price : ℝ := 2000

-- Define the percentage increase
def percentage_increase : ℝ := 2/100

-- Define John's phone price
def johns_price := alans_price * (1 + percentage_increase)

-- The main theorem
theorem johns_phone_price : johns_price = 2040 := by
  sorry

end johns_phone_price_l86_86918


namespace intersection_complement_l86_86232

open Set

variable (U : Type) [TopologicalSpace U]

def A : Set ℝ := { x | x ≥ 0 }

def B : Set ℝ := { y | y ≤ 0 }

theorem intersection_complement (U : Type) [TopologicalSpace U] : 
  A ∩ (compl B) = { x | x > 0 } :=
by
  sorry

end intersection_complement_l86_86232


namespace correct_operation_result_l86_86888

variable (x : ℕ)

theorem correct_operation_result 
  (h : x / 15 = 6) : 15 * x = 1350 :=
sorry

end correct_operation_result_l86_86888


namespace loss_per_metre_proof_l86_86915

-- Define the given conditions
def cost_price_per_metre : ℕ := 66
def quantity_sold : ℕ := 200
def total_selling_price : ℕ := 12000

-- Define total cost price based on cost price per metre and quantity sold
def total_cost_price : ℕ := cost_price_per_metre * quantity_sold

-- Define total loss based on total cost price and total selling price
def total_loss : ℕ := total_cost_price - total_selling_price

-- Define loss per metre
def loss_per_metre : ℕ := total_loss / quantity_sold

-- The theorem we need to prove:
theorem loss_per_metre_proof : loss_per_metre = 6 :=
  by
    sorry

end loss_per_metre_proof_l86_86915


namespace triangle_PQO_side_length_equilateral_l86_86782

noncomputable def side_length_triangle_PQO_on_parabola : ℝ :=
  2 * real.sqrt 3

theorem triangle_PQO_side_length_equilateral {P Q : ℝ × ℝ}
  (hP : P.2 = -(P.1 ^ 2)) 
  (hQ : Q.1 = -P.1 ∧ Q.2 = P.2) 
  (dist_eq : dist (0, 0) P = dist (0, 0) Q ∧ dist (0, 0) P = dist P Q) : 
  dist (0,0) P = side_length_triangle_PQO_on_parabola :=
sorry

end triangle_PQO_side_length_equilateral_l86_86782


namespace mary_average_speed_l86_86734

/-- Define the problem stating that Mary walks 1.5 km uphill in 45 minutes and 1.5 km downhill in 15 minutes.
    We need to prove that her average speed for the round trip is 3 km/hr. -/
theorem mary_average_speed :
  ∀ (distance_uphill distance_downhill : ℝ) (time_uphill time_downhill : ℝ),
    distance_uphill = 1.5 → 
    distance_downhill = 1.5 → 
    time_uphill = 45 / 60 →  -- converting minutes to hours
    time_downhill = 15 / 60 →  -- converting minutes to hours
    (2 * distance_uphill + 2 * distance_downhill) / (time_uphill + time_downhill) = 3 := 
by
  intros distance_uphill distance_downhill time_uphill time_downhill hup hill tup tdown
  have htotal_distance : 2 * distance_uphill + 2 * distance_downhill = 3 := by
    calc
      2 * distance_uphill + 2 * distance_downhill =
      2 * 1.5 + 2 * 1.5 : by rw [hup, hill]
  sorry

end mary_average_speed_l86_86734


namespace eval_ceil_floor_sum_l86_86204

theorem eval_ceil_floor_sum :
  (Int.ceil (7 / 3) + Int.floor (- (7 / 3))) = 0 :=
by
  sorry

end eval_ceil_floor_sum_l86_86204


namespace unique_zero_solution_l86_86951

variables {R : Type*} [LinearOrderedField R]

-- Assumptions
variables (a11 a22 a33 a12 a13 a21 a23 a31 a32 a1 a2 a3 x1 x2 x3 : R)
variable h₁ : 0 < a11
variable h₂ : 0 < a22
variable h₃ : 0 < a33
variable h₄ : a12 < 0
variable h₅ : a13 < 0
variable h₆ : a21 < 0
variable h₇ : a23 < 0
variable h₈ : a31 < 0
variable h₉ : a32 < 0
variable h₁₀ : a11 + a12 + a13 > 0
variable h₁₁ : a21 + a22 + a23 > 0
variable h₁₂ : a31 + a32 + a33 > 0
variable eq1 : a11 * x1 + a12 * x2 + a13 * x3 = 0
variable eq2 : a21 * x1 + a22 * x2 + a23 * x3 = 0
variable eq3 : a31 * x1 + a32 * x2 + a33 * x3 = 0

theorem unique_zero_solution :
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 :=
by {
  sorry
}

end unique_zero_solution_l86_86951


namespace angle_B_is_pi_over_3_perimeter_of_triangle_l86_86348

-- Define the triangle with sides and angles
variables {A B C a b c : ℝ}
variables {BD : ℝ} -- BD is the altitude from B to AC

-- Given conditions
axiom angle_B_condition : c * sin ((A + C) / 2) = b * sin C
axiom BD_condition : BD = 1
axiom side_b_condition : b = sqrt 3

-- The statements to prove
theorem angle_B_is_pi_over_3 (h₁ : angle_B_condition) : B = π / 3 := by
  sorry

theorem perimeter_of_triangle (h₁ : angle_B_condition) (h₂ : BD_condition) (h₃ : side_b_condition) : (a + b + c) = 3 + sqrt 3 := by
  sorry

end angle_B_is_pi_over_3_perimeter_of_triangle_l86_86348


namespace gym_distance_l86_86436

def distance_to_work : ℕ := 10
def distance_to_gym (dist : ℕ) : ℕ := (dist / 2) + 2

theorem gym_distance :
  distance_to_gym distance_to_work = 7 :=
sorry

end gym_distance_l86_86436


namespace position_of_seventeen_fifteen_in_sequence_l86_86666

theorem position_of_seventeen_fifteen_in_sequence :
  ∃ n : ℕ, (17 : ℚ) / 15 = (n + 3 : ℚ) / (n + 1) :=
sorry

end position_of_seventeen_fifteen_in_sequence_l86_86666


namespace equilateral_triangle_sum_of_products_l86_86943

theorem equilateral_triangle_sum_of_products (a b c : Complex) (h_eq_triangle : ∃ θ : ℝ, θ = 2 * π / 3 ∧ 
  ((a - b).abs = 24) ∧ ((b - c).abs = 24) ∧ ((c - a).abs = 24)) (h_sum : (a + b + c).abs = 48) : 
  (ab + ac + bc).abs = 768 :=
by
  sorry

end equilateral_triangle_sum_of_products_l86_86943


namespace f_2_2_eq_7_f_3_3_eq_61_f_4_4_can_be_evaluated_l86_86087

noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem f_2_2_eq_7 : f 2 2 = 7 :=
sorry

theorem f_3_3_eq_61 : f 3 3 = 61 :=
sorry

theorem f_4_4_can_be_evaluated : ∃ n, f 4 4 = n :=
sorry

end f_2_2_eq_7_f_3_3_eq_61_f_4_4_can_be_evaluated_l86_86087


namespace FG_length_of_trapezoid_l86_86806

-- Define the dimensions and properties of trapezoid EFGH.
def EFGH_trapezoid (area : ℝ) (altitude : ℝ) (EF : ℝ) (GH : ℝ) : Prop :=
  area = 180 ∧ altitude = 9 ∧ EF = 12 ∧ GH = 20

-- State the theorem to prove the length of FG.
theorem FG_length_of_trapezoid : 
  ∀ {E F G H : Type} (area EF GH fg : ℝ) (altitude : ℝ),
  EFGH_trapezoid area altitude EF GH → fg = 6.57 :=
by sorry

end FG_length_of_trapezoid_l86_86806


namespace max_coefficient_value_l86_86341

-- Define the conditions given in the problem
def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k
def general_term (n r : ℕ) (a x : ℝ) : ℝ := binomial_coefficient n r * ((-a)^r) * (x^(n - 2*r))
axiom coeff_x3_eq_neg5 (a : ℝ) : (binomial_coefficient 5 1 * (-a)) = -5

-- State the objective to prove
theorem max_coefficient_value (a : ℝ) (h : a = 1) : 
  binomial_coefficient 5 2 = 10 :=
by
  sorry

end max_coefficient_value_l86_86341


namespace apples_in_each_basket_l86_86737

-- Definitions based on the conditions
def total_apples : ℕ := 64
def baskets : ℕ := 4
def apples_taken_per_basket : ℕ := 3

-- Theorem statement based on the question and correct answer
theorem apples_in_each_basket (h1 : total_apples = 64) 
                              (h2 : baskets = 4) 
                              (h3 : apples_taken_per_basket = 3) : 
    (total_apples / baskets - apples_taken_per_basket) = 13 := 
by
  sorry

end apples_in_each_basket_l86_86737


namespace price_of_cork_l86_86096

theorem price_of_cork (C : ℝ) 
  (h₁ : ∃ (bottle_with_cork bottle_without_cork : ℝ), bottle_with_cork = 2.10 ∧ bottle_without_cork = C + 2.00 ∧ bottle_with_cork = C + bottle_without_cork) :
  C = 0.05 :=
by
  obtain ⟨bottle_with_cork, bottle_without_cork, hwc, hwoc, ht⟩ := h₁
  sorry

end price_of_cork_l86_86096


namespace num_sets_l86_86450

open Set

theorem num_sets : 
  let universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
  let proper_subset : Set ℕ := {1, 2, 3}
  let satisfies_conditions (M : Set ℕ) : Prop := proper_subset ⊂ M ∧ M ⊆ universal_set
  finset.card {M : Set ℕ // satisfies_conditions M} = 7 :=
by sorry

end num_sets_l86_86450


namespace values_of_k_real_equal_roots_l86_86960

theorem values_of_k_real_equal_roots (k : ℝ) : 
  (∃ k, (3 - 2 * k)^2 - 4 * 3 * 12 = 0 ∧ (k = -9 / 2 ∨ k = 15 / 2)) :=
by
  sorry

end values_of_k_real_equal_roots_l86_86960


namespace tan_sin_cos_proof_l86_86944

theorem tan_sin_cos_proof (h1 : Real.sin (Real.pi / 6) = 1 / 2)
    (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2)
    (h3 : Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6)) :
    ((Real.tan (Real.pi / 6))^2 - (Real.sin (Real.pi / 6))^2) / ((Real.tan (Real.pi / 6))^2 * (Real.cos (Real.pi / 6))^2) = 1 / 3 := by
  sorry

end tan_sin_cos_proof_l86_86944


namespace validity_thm_l86_86142

noncomputable section
open Complex Real

/-- Proof Problem: Validity of four mathematical statements where at least one of the variables is non-zero. -/
def validity_of_statements (a b : ℂ) : Prop :=
(√(a ^ 2 + b ^ 2) ≥ 0) ∧ 
(√(a ^ 2 + b ^ 2) ≥ complex.abs (a - b)) ∧ 
(¬(√(a ^ 2 + b ^ 2) = complex.abs a + complex.abs b)) ∧
(∃ a b : ℂ, (a ≠ 0 ∨ b ≠ 0) ∧ √(a ^ 2 + b ^ 2) = a * b + 1)

theorem validity_thm (a b : ℂ) : validity_of_statements a b :=
by
  sorry

end validity_thm_l86_86142


namespace polar_to_rectangular_l86_86453

theorem polar_to_rectangular (ρ θ x y : ℝ) (h1 : ρ = 4 * sin θ) 
    (h2 : ρ^2 = x^2 + y^2) (h3 : ρ * sin θ = y) :
  x^2 + y^2 - 4 * y = 0 :=
by
  sorry

end polar_to_rectangular_l86_86453


namespace complex_number_solution_l86_86604

open Complex

theorem complex_number_solution (z : ℂ) (h : z^2 = -99 - 40 * I) : z = 2 - 10 * I ∨ z = -2 + 10 * I :=
sorry

end complex_number_solution_l86_86604


namespace blocks_differ_in_two_ways_count_l86_86099

-- Definitions for conditions
def materials := ["plastic", "wood", "metal"]
def sizes := ["small", "medium", "large"]
def colors := ["blue", "green", "red", "yellow"]
def shapes := ["circle", "hexagon", "square", "triangle"]

-- Main proof statement
theorem blocks_differ_in_two_ways_count : 
  ∃ count : ℕ, 
    (count = 49) ∧
    (∀ block : (materials × sizes × colors × shapes), 
    ∃ (properties_diff : materials × sizes × colors × shapes), 
    (block ≠ properties_diff) ∧
    (properties_diff_differs_in_two_ways (properties_diff, "plastic", "medium", "red", "circle"))) :=
by
  -- Leaving the proof as a placeholder
  sorry

end blocks_differ_in_two_ways_count_l86_86099


namespace Palić_infinitely_differentiable_l86_86744

section PalićFunction

variables {a b c x y z : ℝ}

-- Conditions
def conditions (a b c : ℝ) : Prop :=
  (a + b + c = 1) ∧ (a^2 + b^2 + c^2 = 1) ∧ (a^3 + b^3 + c^3 ≠ 1)

-- Definition of a Palić function
def Palić (a b c : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f x + f y + f z = f (a*x + b*y + c*z) + f (b*x + c*y + a*z) + f (c*x + a*y + b*z)

-- Axioms for a given a, b, and c
axiom ab_valid : conditions a b c

-- Theorem to prove
theorem Palić_infinitely_differentiable (f : ℝ → ℝ) (h : Palić a b c f) : 
  ∀ x, ∃ A B C : ℝ, f x = A*x^2 + B*x + C := sorry

end PalićFunction

end Palić_infinitely_differentiable_l86_86744


namespace symmetric_point_in_third_quadrant_l86_86455

def point_symmetric (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem symmetric_point_in_third_quadrant (P : ℝ × ℝ) (h : P = (-2, 1)) : point_symmetric P = (-2, -1) ∧ point_symmetric P ∈ { p : ℝ × ℝ | p.1 < 0 ∧ p.2 < 0 } :=
by
  have hs : point_symmetric P = (-2, -1), from sorry
  split
  case inl =>
    exact hs
  case inr =>
    rw [hs]
    exact sorry

end symmetric_point_in_third_quadrant_l86_86455


namespace bricks_in_wall_l86_86572

theorem bricks_in_wall (x : ℕ) :
  (∀ x', (12 * x' = x) → (∃ t : ℝ, t = 12 ∧ t * (x / 12 - 15) = x)) ∧
  (∀ x', (15 * x' = x) → (∃ t : ℝ, t = 15 ∧ t * (x / 15 - 15) = x)) ∧
  (∃ t : ℝ, t = 6 ∧ t * ((3 * x) / 20 - 15) = x) ↔ x = 540 :=
begin
  sorry
end

end bricks_in_wall_l86_86572


namespace tunnel_length_l86_86144

noncomputable def diagonal_length := 2
noncomputable def side_length_of_square := diagonal_length / Real.sqrt 2
noncomputable def distance_center_to_corner := side_length_of_square * Real.sqrt 2 / 2
noncomputable def distance_midpoint_to_corner := side_length_of_square / 2

theorem tunnel_length : 
  ∃ (t : ℝ), t = Real.sqrt (distance_center_to_corner^2 + distance_midpoint_to_corner^2) ∧ t = (Real.sqrt 6) / 2 :=
by
  let side_length := side_length_of_square
  let center_to_corner := distance_center_to_corner
  let midpoint_to_corner := distance_midpoint_to_corner
  have h1 : side_length = Real.sqrt 2 := sorry
  have h2 : center_to_corner = 1 := sorry
  have h3 : midpoint_to_corner = Real.sqrt 2 / 2 := sorry
  use Real.sqrt (center_to_corner^2 + midpoint_to_corner^2)
  split
  case h_1 {
    rw [h2, h3]
    calc Real.sqrt ((1)^2 + (Real.sqrt 2 / 2)^2)
        = Real.sqrt (1 + (2 / 4)) : by sorry
    ... = Real.sqrt (3 / 2) : by sorry
    ... = (Real.sqrt 6) / 2 : by sorry
  }
  case h_2 {
    rw [h2, h3]
    exact sorry
  }

end tunnel_length_l86_86144


namespace compounded_ratio_l86_86877

theorem compounded_ratio (a b c d e f g h i j k l : ℤ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 6) (h4 : d = 11) (h5 : e = 11) (h6 : f = 2) (h7 : g = 132) (h8 : h = 66) (h9 : (a * c * e) = g) (h10 : (b * d * f) = h) (h11 : Int.gcd g h = 66) :
  (g / h) = 2 := by
  rw [h7, h8, Int.div_self],
  sorry

end compounded_ratio_l86_86877


namespace land_connects_through_center_l86_86404

-- Define the conditions and statement
variable (S : Type) [TopologicalSpace S] [Fintype S]
variable (L : Set S) [MeasurableSpace S] 
variable [MeasureSpace S] [IsSphere S]

-- Assume that the land area is more than half of the total surface area
variable (h : MeasureTheory.MeasureMeasure.to_fun L > MeasureTheory.MeasureMeasure.to_fun (Set.univ \ L) / 2)

-- Define a sphere and the property of the land area
theorem land_connects_through_center:
∃ (x y : L), ∥x - y∥ = 2 * radius (S)
:= sorry

end land_connects_through_center_l86_86404


namespace trajectory_M_line_l_and_area_triangle_smallest_circle_l86_86645

-- 1. Prove the trajectory equation of M
theorem trajectory_M (M : ℝ × ℝ) (h1 : ∃ x y, M = (x, y) ∧ (x - 1)^2 + (y - 3)^2 = 2) :
  (∃ x y, M = (x, y) ∧ (x - 1)^2 + (y - 3)^2 = 2) := by
  sorry

-- 2. Given |OP| = |OM|, prove the equation of line l, and the area of ∆POM
theorem line_l_and_area_triangle (l : ℝ × ℝ) (h2 : |OP| = |OM|) :
  (∃ b m, l = (b, m) ∧ y = -1/3 * x + 8/3) ∧
  ∃ (area : ℝ), area = 16/5 := by
  sorry

-- 3. Under conditions of (2), prove the equation of the smallest circle through intersection points
theorem smallest_circle (l_intersect : ℝ × ℝ) (h3 : ∃ p1 p2, l_intersect = (p1, p2)) :
  ∃ c r, (c = (-2/5, 14/5)) ∧ (r = √(72/5)) := by
  sorry

end trajectory_M_line_l_and_area_triangle_smallest_circle_l86_86645


namespace sum_of_odd_integers_excluding_multiples_of_5_l86_86856

theorem sum_of_odd_integers_excluding_multiples_of_5 :
  let S := ∑ k in Finset.filter (λ x, x % 2 = 1 ∧ x % 5 ≠ 0) (Finset.Ico 200 600),
     T := 80000,
     U := 16000
  in S = T - U :=
by
  sorry

end sum_of_odd_integers_excluding_multiples_of_5_l86_86856


namespace harriet_forward_speed_proof_l86_86860

def harriet_forward_time : ℝ := 3 -- forward time in hours
def harriet_return_speed : ℝ := 150 -- return speed in km/h
def harriet_total_time : ℝ := 5 -- total trip time in hours

noncomputable def harriet_forward_speed : ℝ :=
  let distance := harriet_return_speed * (harriet_total_time - harriet_forward_time)
  distance / harriet_forward_time

theorem harriet_forward_speed_proof : harriet_forward_speed = 100 := by
  sorry

end harriet_forward_speed_proof_l86_86860


namespace even_and_increasing_on_0_inf_l86_86131

noncomputable def fA (x : ℝ) : ℝ := x^(2/3)
noncomputable def fB (x : ℝ) : ℝ := (1/2)^x
noncomputable def fC (x : ℝ) : ℝ := Real.log x
noncomputable def fD (x : ℝ) : ℝ := -x^2 + 1

theorem even_and_increasing_on_0_inf (f : ℝ → ℝ) : 
  (∀ x, f x = f (-x)) ∧ (∀ a b, (0 < a ∧ a < b) → f a < f b) ↔ f = fA :=
sorry

end even_and_increasing_on_0_inf_l86_86131


namespace part1_partial_fractions_part2_series_value_l86_86412

/-
  Part 1: Expressing (2n+1)/(n^2+n) as partial fractions
-/
theorem part1_partial_fractions (n : ℤ) (h : n ≠ 0) : 
  (2 * n + 1) / (n^2 + n) = 1 / n + 1 / (n + 1) :=
by
  sorry

/-
  Part 2: Evaluating the series
-/
theorem part2_series_value : 
  ∑ k in finset.range 20, (-1 : ℤ) ^ k * (2 * k + 3) / ((k + 1) * (k + 2)) = 20 / 21 :=
by
  sorry

end part1_partial_fractions_part2_series_value_l86_86412


namespace intersection_A_B_l86_86648

-- Defining the sets A and B based on the given conditions
def A : set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : set ℝ := {x | 3^x < 1}

-- Stating the theorem to prove the intersection A ∩ B
theorem intersection_A_B : A ∩ B = {x | -1 ≤ x ∧ x < 0} :=
by
  sorry

end intersection_A_B_l86_86648


namespace perimeter_of_park_l86_86868

theorem perimeter_of_park (s : ℝ) (h1 : ∃(s : ℝ), (3*3 + 2*s*3 = 3*3 + 3)) :=
    (4 * s = 600) :=
sorry

end perimeter_of_park_l86_86868


namespace find_distance_P_to_plane_l86_86257

-- Conditions for the problem
variables (A B M P : Type) [Point A B M P]

-- Midpoint condition
def midpoint (A B M : Type) [LineSegment A B] : Prop := dist A M = 1 ∧ dist B M = 1

-- Given distance and angles
def given_conditions (A B M P : Type) [Plane alpha M] (alpha: Plane A) : Prop :=
  dist A B = 2 ∧ M ∈ alpha ∧
  ∠informal P A alpha = 30 ∧
  ∠informal P M alpha = 45 ∧
  ∠informal P B alpha = 60

-- Goal: Find the distance from P to plane alpha
def distance_P_to_plane (P : Point) (alpha : Plane) : ℝ := sorry

-- The main problem statement
theorem find_distance_P_to_plane (A B M P alpha : Type) [Point A B M P] [Plane alpha M A] [LineSegment A B]:
  midpoint A B M →
  given_conditions A B M P alpha →
  distance_P_to_plane P alpha = sqrt 6 / 2 := 
sorry

end find_distance_P_to_plane_l86_86257


namespace shanmukham_total_payment_l86_86415

noncomputable def total_price_shanmukham_pays : Real :=
  let itemA_price : Real := 6650
  let itemA_rebate : Real := 6 -- percentage
  let itemA_tax : Real := 10 -- percentage

  let itemB_price : Real := 8350
  let itemB_rebate : Real := 4 -- percentage
  let itemB_tax : Real := 12 -- percentage

  let itemC_price : Real := 9450
  let itemC_rebate : Real := 8 -- percentage
  let itemC_tax : Real := 15 -- percentage

  let final_price (price : Real) (rebate : Real) (tax : Real) : Real :=
    let rebate_amt := (rebate / 100) * price
    let price_after_rebate := price - rebate_amt
    let tax_amt := (tax / 100) * price_after_rebate
    price_after_rebate + tax_amt

  final_price itemA_price itemA_rebate itemA_tax +
  final_price itemB_price itemB_rebate itemB_tax +
  final_price itemC_price itemC_rebate itemC_tax

theorem shanmukham_total_payment :
  total_price_shanmukham_pays = 25852.12 := by
  sorry

end shanmukham_total_payment_l86_86415


namespace sum_of_first_ten_nice_numbers_is_182_l86_86573

def is_proper_divisor (n d : ℕ) : Prop :=
  d > 1 ∧ d < n ∧ n % d = 0

def is_nice (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, is_proper_divisor n m → ∃ p q, n = p * q ∧ p ≠ q

def first_ten_nice_numbers : List ℕ := [6, 8, 10, 14, 15, 21, 22, 26, 27, 33]

def sum_first_ten_nice_numbers : ℕ := first_ten_nice_numbers.sum

theorem sum_of_first_ten_nice_numbers_is_182 :
  sum_first_ten_nice_numbers = 182 :=
by
  sorry

end sum_of_first_ten_nice_numbers_is_182_l86_86573


namespace find_smaller_base_l86_86020

theorem find_smaller_base (p q a : ℝ) (h : p < q) (angle_ratio : ∃ α : ℝ, tan α = q / (2 * p)) :
  ∃ x : ℝ, x = (p^2 + a * p - q^2) / p :=
by sorry

end find_smaller_base_l86_86020


namespace sum_series_l86_86949

theorem sum_series : ∑' n, (2 * n + 1) / (n * (n + 1) * (n + 2)) = 1 := 
sorry

end sum_series_l86_86949


namespace a_2_is_minus_1_l86_86036
open Nat

variable (a S : ℕ → ℤ)

-- Conditions
axiom sum_first_n (n : ℕ) (hn : n > 0) : 2 * S n - n * a n = n
axiom S_20 : S 20 = -360

-- The problem statement to prove
theorem a_2_is_minus_1 : a 2 = -1 :=
by 
  sorry

end a_2_is_minus_1_l86_86036


namespace money_distribution_l86_86557

theorem money_distribution (A B C : ℝ) 
  (h1 : A + B + C = 500) 
  (h2 : A + C = 200) 
  (h3 : B + C = 340) : 
  C = 40 := 
sorry

end money_distribution_l86_86557


namespace tangent_circle_exists_l86_86989

theorem tangent_circle_exists 
  (A B C D E : Point) 
  (tau : Circle) 
  (order : cyclic_order [A, B, C, D, E] tau)
  (AB_parallel_CE : is_parallel (line_through A B) (line_through C E))
  (angle_ABC_gt_90 : angle A B C > (90 : ℝ))
  (k : Circle)
  (tangent_AD : is_tangent k (line_through A D))
  (tangent_CE : is_tangent k (line_through C E))
  (tangent_tau : is_tangent k tau)
  (k_tau_touch : touch arc_ED (ne A B C k tau))
  (F : Point)
  (intersection_F : F ≠ A ∧ F ∈ intersection_of tau (tangent_to k through A different_from AD)) :
  ∃ (circle : Circle), is_tangent circle (line_through B D) ∧ is_tangent circle (line_through B F) ∧ is_tangent circle (line_through C E) ∧ is_tangent circle tau :=
sorry

end tangent_circle_exists_l86_86989


namespace total_messages_equation_l86_86699

theorem total_messages_equation (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  exact h

end total_messages_equation_l86_86699


namespace mean_of_three_added_numbers_l86_86807

theorem mean_of_three_added_numbers (twelve_numbers : ℕ → ℝ) (x y z : ℝ) 
  (h_sum_twelve : (∑ i in finset.range 12, twelve_numbers i) = 384)
  (h_mean_new : (384 + x + y + z) / 15 = 48) :
  (x + y + z) / 3 = 112 :=
sorry

end mean_of_three_added_numbers_l86_86807


namespace circle_cover_l86_86633

structure Circle (Point : Type) [MetricSpace Point] :=
(center : Point)
(radius : ℝ)

variables {Point : Type} [MetricSpace Point]
variables {O1 O2 : Point}
variables {r1 r2 : ℝ}

theorem circle_cover (h : r2 > r1) :
  (∀ (x : Point), Dist x O1 ≤ r1 → Dist x O2 ≤ r2) ∧ ¬ (∀ (x : Point), Dist x O2 ≤ r2 → Dist x O1 ≤ r1) := by
  sorry

end circle_cover_l86_86633


namespace average_velocity_interval_l86_86021

-- Definitions based on the conditions
def law_of_motion (t : ℝ) : ℝ := t^2 + 3

-- Theorem statement
theorem average_velocity_interval (Δt : ℝ) : 
  let s := law_of_motion in
  (s (3 + Δt) - s 3) / Δt = 6 + Δt :=
sorry

end average_velocity_interval_l86_86021


namespace part1_part2_l86_86886

variable {f : ℝ → ℝ}
variable (a : ℝ)

-- The function f has domain ℝ and its derivative satisfies 0 < f'(x) < 1.
hypothesis h1 : ∀ x, 0 < deriv f x ∧ deriv f x < 1
-- The constant a is a real root of the equation f(x) = x.
hypothesis h2 : f a = a

-- 1. Prove that when x > a, it always holds that x > f(x).
theorem part1 (x : ℝ) (hx : x > a) : x > f x :=
sorry

-- 2. Prove that for any x1 and x2, if |x1 - a| < 1 and |x2 - a| < 1, then |f(x1) - f(x2)| < 2.
theorem part2 (x1 x2 : ℝ) (h1 : |x1 - a| < 1) (h2 : |x2 - a| < 1) : |f x1 - f x2| < 2 :=
sorry

end part1_part2_l86_86886


namespace circle_center_coordinates_l86_86809

theorem circle_center_coordinates :
  ∃ (h k : ℝ), (x^2 + y^2 - 2*x + 4*y = 0) → (x - h)^2 + (y + k)^2 = 5 :=
sorry

end circle_center_coordinates_l86_86809


namespace eval_expr_eq_zero_l86_86163

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l86_86163


namespace coefficient_of_x_equation_l86_86427

theorem coefficient_of_x_equation (n : ℕ) (h : (-2) ^ n + (-2) ^ (n - 1) * n = -128) : n = 6 :=
sorry

end coefficient_of_x_equation_l86_86427


namespace denominator_of_0_27_repeating_is_11_l86_86004

theorem denominator_of_0_27_repeating_is_11 :
  ∃ (denom : ℕ), (0.27272727...) = (3 : ℝ) / (denom : ℝ) ∧ denom = 11 :=
by
  sorry

end denominator_of_0_27_repeating_is_11_l86_86004


namespace area_of_fourth_rectangle_l86_86912

theorem area_of_fourth_rectangle (a b c d : ℕ) (h1 : a = 18) (h2 : b = 27) (h3 : c = 12) :
d = 93 :=
by
  -- Problem reduces to showing that d equals 93 using the given h1, h2, h3
  sorry

end area_of_fourth_rectangle_l86_86912


namespace positive_factors_of_450_are_perfect_squares_eq_8_l86_86288

theorem positive_factors_of_450_are_perfect_squares_eq_8 :
  let n := 450 in
  let is_factor (m k : ℕ) : Prop := k ≠ 0 ∧ k ≤ m ∧ m % k = 0 in
  let is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n in
  let factors (m : ℕ) : List ℕ := List.filter (λ k, is_factor m k) (List.range (m + 1)) in
  let perfect_square_factors := List.filter is_perfect_square (factors n) in
  List.length perfect_square_factors = 8 :=
by 
  have h1 : n = 2 * 3^2 * 5^2 := by sorry
  have h2 : ∀ m, m ∈ factors n ↔ is_factor n m := by sorry
  have h3 : ∀ m, is_perfect_square m ↔ (∃ k, k * k = m) := by sorry
  have h4 : List.length perfect_square_factors = 8 := by sorry
  exact h4

end positive_factors_of_450_are_perfect_squares_eq_8_l86_86288


namespace perpendicular_vectors_implies_k_l86_86996

variables (k : ℝ)

def a := (2, k)
def b := (-1, 3)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem perpendicular_vectors_implies_k :
  dot_product a b = 0 → k = 2 / 3 :=
by
  intro h
  sorry

end perpendicular_vectors_implies_k_l86_86996


namespace ceiling_floor_sum_l86_86159

theorem ceiling_floor_sum : (Int.ceil (7 / 3) + Int.floor (-(7 / 3)) = 0) := by
  sorry

end ceiling_floor_sum_l86_86159


namespace least_four_digit_with_factors_3_5_7_l86_86060

open Nat

-- Definitions for the conditions
def has_factors (n : ℕ) (factors : List ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Main theorem statement
theorem least_four_digit_with_factors_3_5_7
  (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000)
  (h3 : has_factors n [3, 5, 7]) :
  n = 1050 :=
sorry

end least_four_digit_with_factors_3_5_7_l86_86060


namespace time_to_meet_l86_86964

-- Definitions based on conditions
def motorboat_speed_Serezha : ℝ := 20 -- km/h
def crossing_time_Serezha : ℝ := 0.5 -- hours (30 minutes)
def running_speed_Dima : ℝ := 6 -- km/h
def running_time_Dima : ℝ := 0.25 -- hours (15 minutes)
def combined_speed : ℝ := running_speed_Dima + running_speed_Dima -- equal speeds running towards each other
def distance_meet : ℝ := (running_speed_Dima * running_time_Dima) -- The distance they need to cover towards each other

-- Prove the time for them to meet
theorem time_to_meet : (distance_meet / combined_speed) = (7.5 / 60) :=
by
  sorry

end time_to_meet_l86_86964


namespace edmonton_to_calgary_travel_time_l86_86600

theorem edmonton_to_calgary_travel_time :
  let distance_edmonton_red_deer := 220
  let distance_red_deer_calgary := 110
  let speed_to_red_deer := 100
  let detour_distance := 30
  let detour_time := (distance_edmonton_red_deer + detour_distance) / speed_to_red_deer
  let stop_time := 1
  let speed_to_calgary := 90
  let travel_time_to_calgary := distance_red_deer_calgary / speed_to_calgary
  detour_time + stop_time + travel_time_to_calgary = 4.72 := by
  sorry

end edmonton_to_calgary_travel_time_l86_86600


namespace ceil_floor_sum_l86_86181

theorem ceil_floor_sum : (Int.ceil (7 / 3 : ℝ) + Int.floor (- 7 / 3 : ℝ) = 0) := 
  by {
    -- referring to the proof steps
    have h1: Int.ceil (7 / 3 : ℝ) = 3,
    { sorry },
    have h2: Int.floor (- 7 / 3 : ℝ) = -3,
    { sorry },
    rw [h1, h2],
    norm_num
  }

end ceil_floor_sum_l86_86181


namespace find_omega_l86_86324

-- Define the function and conditions
def f (x : ℝ) (ω : ℝ) := 2 * Real.sin (ω * x + Real.pi / 3)

-- The main theorem statement
theorem find_omega (ω : ℝ) (hω : ω > 0)
  (h_intersection_distance : ∃ d > 0, d = 2) :
  ω = Real.pi / 2 :=
  sorry

end find_omega_l86_86324


namespace right_triangle_vertices_count_l86_86402

theorem right_triangle_vertices_count : ∃ n, (n = 321372) ∧
    let P_1 := set.range (λ i, (2, i)) in
    let P_2 := set.range (λ i, (9, i)) in
    set.card P_1 = 400 ∧
    set.card P_2 = 400 ∧
    set.card (P_1 ∪ P_2) = 800 ∧
    ∀ (p1 p2 p3 : ℝ × ℝ),
    p1 ∈ (P_1 ∪ P_2) → p2 ∈ (P_1 ∪ P_2) → p3 ∈ (P_1 ∪ P_2) →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    (p1.1 ≠ p2.1 ∨ p2.1 ≠ p3.1 ∨ p1.1 ≠ p3.1) →
    ((p1.2 - p2.2) * (p2.2 - p3.2) * (p1.2 - p3.2) = 0 → n = 321372 :=
sorry

end right_triangle_vertices_count_l86_86402


namespace factory_production_days_l86_86530

theorem factory_production_days 
  (computers_per_month : ℕ)
  (computers_per_30_minutes : ℕ) :
  computers_per_month = 4032 ∧ computers_per_30_minutes = 3 →
  (∃ days_in_month : ℕ, days_in_month = 28) :=
by
  intro h
  use 28
  sorry

end factory_production_days_l86_86530


namespace num_perfect_square_factors_of_450_l86_86289

theorem num_perfect_square_factors_of_450 :
  ∃ n : ℕ, n = 4 ∧ ∀ d : ℕ, d ∣ 450 → (∃ k : ℕ, d = k * k) → d = 1 ∨ d = 25 ∨ d = 9 ∨ d = 225 :=
by
  sorry

end num_perfect_square_factors_of_450_l86_86289


namespace eval_ceil_floor_l86_86176

-- Definitions of the ceiling and floor functions and their relevant properties
def ceil (x : ℝ) : ℤ := ⌈x⌉
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Mathematical statement to prove
theorem eval_ceil_floor : ceil (7 / 3) + floor (- (7 / 3)) = 0 :=
by
  -- Proof is omitted
  sorry

end eval_ceil_floor_l86_86176


namespace EQ_is_sqrt_c_plus_sqrt_d_l86_86408

open Real EuclideanGeometry

noncomputable def point := (ℝ × ℝ)

def Q_on_diagonal_AC (E F G H Q : point) :=
  ∃ k ∈ (0 : ℝ) .. 1, Q = (1 - k) • E + k • H -- property of being on diagonal

def EQ_gt_GQ (E G Q : point) : Prop :=
  (dist E Q) > (dist G Q)

def circumcenter (Δ : triplet point) (R : point) : Prop :=
  ∃ (O : point), R = circumarc (Δ, O)

def square (E F G H : point) : Prop :=
  dist E F = 8 ∧ dist F G = 8 ∧ dist G H = 8 ∧ dist H E = 8 ∧ 
  dist E G = √128 ∧ dist F H = √128 ∧
  angle (E, F, G) = π/2 ∧ angle (F, G, H) = π/2 ∧ angle (G, H, E) = π/2 ∧ angle (H, E, F) = π/2

def right_angle (R1 Q R2 : point) : Prop :=
  angle (R1, Q, R2) = π/2

theorem EQ_is_sqrt_c_plus_sqrt_d (E F G H Q R1 R2 : point) (c d : ℕ) :
  square E F G H →
  Q_on_diagonal_AC E F G H Q →
  EQ_gt_GQ E G Q →
  circumcenter (E, F, Q) R1 →
  circumcenter (G, H, Q) R2 →
  right_angle R1 Q R2 →
  (dist E Q) = (sqrt c) + (sqrt d) →
  c + d = 40 :=
sorry

end EQ_is_sqrt_c_plus_sqrt_d_l86_86408


namespace Tim_total_payment_l86_86480

-- Define the context for the problem
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.3

-- Define the total amount paid as the sum of the manicure cost and the tip
def total_amount_paid (cost : ℝ) (tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- The theorem to be proven
theorem Tim_total_payment : total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end Tim_total_payment_l86_86480


namespace num_valid_Ns_less_2000_l86_86315

theorem num_valid_Ns_less_2000 : 
  {N : ℕ | N < 2000 ∧ ∃ x : ℝ, x > 0 ∧ x^⟨floor x⟩ = N}.card = 412 := 
sorry

end num_valid_Ns_less_2000_l86_86315


namespace sum_of_interior_angles_of_hexagon_l86_86463

theorem sum_of_interior_angles_of_hexagon : 
  let n := 6 in (n - 2) * 180 = 720 := 
by
  let n := 6
  show (n - 2) * 180 = 720
  sorry

end sum_of_interior_angles_of_hexagon_l86_86463


namespace BC_line_eq_area_triangle_ABC_l86_86472

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (4, 0)
def B : (ℝ × ℝ) := (6, 7)
def C : (ℝ × ℝ) := (0, 3)

-- Statement 1: Prove the equation of the line on which side BC lies
theorem BC_line_eq : ∃ (a b c : ℝ), a * B.1 + b * B.2 + c = 0 ∧ a * C.1 + b * C.2 + c = 0 ∧ a * 2 - 3 * 3 + 9 = 0 :=
  sorry

-- Statement 2: Prove the area of the triangle ABC
theorem area_triangle_ABC : 
  let S := 1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) in
  S = 17 :=
  sorry

end BC_line_eq_area_triangle_ABC_l86_86472


namespace distance_between_vertices_l86_86141

-- Define the equation
def equation (x y : ℝ) : Prop := sqrt(x^2 + y^2) + abs(y + 2) = 4

-- Define the vertex for y >= -2
def vertex1 : ℝ × ℝ := (0, 1)

-- Define the vertex for y < -2
def vertex2 : ℝ × ℝ := (0, -3)

-- Proof statement
theorem distance_between_vertices : 
  (∃ x y : ℝ, equation x y) →
    abs ((vertex1.snd) - (vertex2.snd)) = 4 :=
by
  sorry

end distance_between_vertices_l86_86141


namespace ratio_area_quadrilateral_decagon_l86_86413

-- Definitions of the areas and the regular decagon
variables {A B C D E F G H I J : ℝ}
variables (m n : ℝ)

-- Conditions
def is_regular_decagon (A B C D E F G H I J : ℝ) : Prop :=
  -- (Assume appropriate properties that can define a regular decagon)
  sorry

def area_decagon (A B C D E F G H I J : ℝ) : ℝ :=
  n

def area_quadrilateral_aceg (A C E G : ℝ) : ℝ :=
  m

-- The proof problem.
theorem ratio_area_quadrilateral_decagon
  (h_reg_dec : is_regular_decagon A B C D E F G H I J)
  (h_area_dec : area_decagon A B C D E F G H I J = n)
  (h_area_quad : area_quadrilateral_aceg A C E G = m) :
  m / n = 1 / 4 :=
sorry

end ratio_area_quadrilateral_decagon_l86_86413


namespace remainder_when_x_squared_divided_by_24_l86_86687

theorem remainder_when_x_squared_divided_by_24 (x : ℤ) (h1 : 6 * x ≡ 12 [MOD 24]) (h2 : 4 * x ≡ 20 [MOD 24]) : (x^2 ≡ 12 [MOD 24]) := 
sorry

end remainder_when_x_squared_divided_by_24_l86_86687


namespace chess_game_coordinates_l86_86718

-- Define the initial conditions and the movement rules
def initial_position := (0, 0 : ℤ × ℤ)

def move : ℕ → ℤ × ℤ → ℤ × ℤ
| n, (x, y) := if n % 3 = 0 then (x, y + 1)
               else if n % 3 = 1 then (x + 1, y)
               else (x + 2, y)

def final_position_after_steps : ℕ → ℤ × ℤ
| 0     := initial_position
| (n+1) := move (n+1) (final_position_after_steps n)

-- Define the goal
theorem chess_game_coordinates :
  final_position_after_steps 100 = (100, 33) :=
sorry

end chess_game_coordinates_l86_86718


namespace eval_ceil_floor_sum_l86_86202

theorem eval_ceil_floor_sum :
  (Int.ceil (7 / 3) + Int.floor (- (7 / 3))) = 0 :=
by
  sorry

end eval_ceil_floor_sum_l86_86202


namespace balls_into_boxes_l86_86787

theorem balls_into_boxes : ∃ n : ℕ, n = 240 ∧ ∃ f : Fin 5 → Fin 4, ∀ i : Fin 4, ∃ j : Fin 5, f j = i := by
  sorry

end balls_into_boxes_l86_86787


namespace proportion_equation_l86_86685

theorem proportion_equation (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
by 
  sorry

end proportion_equation_l86_86685


namespace minimum_red_points_l86_86866

open Set

theorem minimum_red_points (n : ℕ) (h : n ≥ 2) (A : Fin n → ℝ × ℝ) (hA : Injective A) :
  ∃ R : Fin (2*n - 3) → ℝ × ℝ, ∀ i j : Fin n, i < j → ∃ k : Fin (2*n - 3), R k = ((A i.1 + A j.1) / 2, (A i.2 + A j.2) / 2) := sorry

end minimum_red_points_l86_86866


namespace number_of_notebooks_in_class2_l86_86043

theorem number_of_notebooks_in_class2 :
  ∀ (not_class1 : ℕ) (not_class2 : ℕ) (both_classes : ℕ),
  not_class1 = 162 →
  not_class2 = 143 →
  both_classes = 87 →
  (let neither_class := not_class1 - not_class2 in 
   let total := both_classes + neither_class in
   (total / 2) = 53) := 
by
  intros not_class1 not_class2 both_classes h1 h2 h3
  let neither_class := not_class1 - not_class2
  let total := both_classes + neither_class
  show (total / 2) = 53
  rw [h1, h2, h3]
  sorry

end number_of_notebooks_in_class2_l86_86043


namespace gym_distance_l86_86435

def distance_to_work : ℕ := 10
def distance_to_gym (dist : ℕ) : ℕ := (dist / 2) + 2

theorem gym_distance :
  distance_to_gym distance_to_work = 7 :=
sorry

end gym_distance_l86_86435


namespace smallest_number_among_l86_86562

noncomputable def smallest_real among (a b c d : Real) : Real :=
  if a <= b && a <= c && a <= d then a
  else if b <= a && b <= c && b <= d then b
  else if c <= a && c <= b && c <= d then c
  else d

theorem smallest_number_among :=
  let r1 := -1.0
  let r2 := -|(-2.0)|
  let r3 := 0.0
  let r4 := Real.pi
  smallest_real r1 r2 r3 r4 = -2.0 :=
by
  -- placeholder for proof
  sorry

end smallest_number_among_l86_86562


namespace angle_is_60_l86_86849

def angle_60 (square : ℝ) : Prop :=
  ∃ adjacent : ℝ, adjacent = 120 ∧ adjacent + square = 180 ∧ square = 60

theorem angle_is_60 : angle_60 60 :=
by
  unfold angle_60
  apply Exists.intro 120
  split
  { rfl }
  split
  { exact add_eq_right_iff.mpr rfl }
  { rfl }

end angle_is_60_l86_86849


namespace smallest_four_digit_product_digits_eq_512_l86_86222

theorem smallest_four_digit_product_digits_eq_512 :
  ∃ n : ℕ, (1000 ≤ n ∧ n < 10000) ∧ (n.digits 10).prod = 512 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m < 10000) ∧ (m.digits 10).prod = 512 → n ≤ m :=
begin
  sorry,
end

end smallest_four_digit_product_digits_eq_512_l86_86222


namespace find_k_l86_86264

noncomputable def vec (α : Type*) := α

variables {α : Type*} [vector_space ℝ α]

variables (a b m n : α) (k : ℝ)

-- Conditions
def not_collinear (a b : α) : Prop := ¬ collinear ℝ ![a, b]

def m_def := (m = (2:ℝ) • a - (3:ℝ) • b)
def n_def := (n = (3:ℝ) • a + k • b)

-- Parallelism condition
def parallel (m n : α) : Prop := ∃ λ : ℝ, n = λ • m

theorem find_k (a b : α) (k : ℝ)
  (h1 : not_collinear a b)
  (h2 : m_def (2 • a - 3 • b))
  (h3 : n_def (3 • a + k • b))
  (h4 : parallel (2 • a - 3 • b) (3 • a + k • b)) :
  k = -9 / 2 :=
sorry

end find_k_l86_86264


namespace boys_in_2nd_l86_86044

def students_in_3rd := 19
def students_in_4th := 2 * students_in_3rd
def girls_in_2nd := 19
def total_students := 86
def students_in_2nd := total_students - students_in_3rd - students_in_4th

theorem boys_in_2nd : students_in_2nd - girls_in_2nd = 10 := by
  sorry

end boys_in_2nd_l86_86044


namespace arithmetic_sequence_general_term_sum_of_b_n_l86_86643

noncomputable def a_n (n : ℕ) : ℕ := 2 * n

def b_n (n : ℕ) : ℝ := 4 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, b_n i

theorem arithmetic_sequence_general_term (n : ℕ) : a_n n = 2 * n := by
  sorry

theorem sum_of_b_n (n : ℕ) : T_n n = n / (n + 1) := by
  sorry

end arithmetic_sequence_general_term_sum_of_b_n_l86_86643


namespace coplanar_points_eqn_l86_86749

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (E F G H : V)

theorem coplanar_points_eqn (m : ℝ) 
  (h : 4 • E - 3 • F + 2 • G + m • H = 0) :
  ∀ (E F G H : V), coplanar {E, F, G, H} → m = -3 :=
by
  sorry

end coplanar_points_eqn_l86_86749


namespace max_radius_of_inscribed_sphere_l86_86396

theorem max_radius_of_inscribed_sphere 
  (edge1 edge2 edge3 : ℝ) 
  (h1 : edge1 = 13) 
  (h2 : edge2 = 14) 
  (h3 : edge3 = 15) : 
  ∃ r : ℝ, r = 3 * Real.sqrt 55 / 8 := 
by
  use 3 * Real.sqrt 55 / 8
  sorry

end max_radius_of_inscribed_sphere_l86_86396


namespace quadratic_coeffs_l86_86816

noncomputable def quadratic_highest_point (b c : ℝ) : Prop :=
  ∀ x : ℝ, -x^2 + b*x + c ≤ -((x + 1)^2) - 3

theorem quadratic_coeffs :
  ∃ (b c : ℝ), quadratic_highest_point b c ∧ b = -2 ∧ c = -4 :=
by
  use [-2, -4]
  split
  · intro x
    calc
      -x^2 + (-2)*x + (-4)
          = -(x^2) + (-2)*x + (-4) : by ring
      ... = -(x^2 + 2*x + 1 - 1) + (-4) : by ring
      ... = -((x + 1)^2) + 1 - 4 : by rw [neg_add_eq_sub, sub_add_sub_cancel]
      ... = -((x + 1)^2) - 3 : by ring
  · split
    · rfl
    · rfl

end quadratic_coeffs_l86_86816


namespace cookies_per_bag_l86_86074

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (H1 : total_cookies = 703) (H2 : num_bags = 37) : total_cookies / num_bags = 19 := by
  sorry

end cookies_per_bag_l86_86074


namespace problem_solution_l86_86266

def BinomialVariance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

def PropositionsCorrect : ℕ :=
  let X_var : ℝ := BinomialVariance 4 0.1
  let prop1 : Prop := X_var = 0.36
  let prop2 : Prop := ∀ (x : ℝ) (data : List ℝ), (data.mean - x) ≠ data.mean ∨ (data.variance - 0) = data.variance
  let prop3 : Prop := ∀ (students : List ℤ), students = [5, 16, 27, 38, 49] → list.prod (list.map (λ (x : ℤ), x - students.head!) students) = 55
  (if prop1 then 1 else 0) + (if prop2 then 1 else 0) + (if prop3 then 1 else 0)

theorem problem_solution : PropositionsCorrect = 1 := by
  sorry

end problem_solution_l86_86266


namespace volume_of_wedge_l86_86363

theorem volume_of_wedge : 
  let r : ℝ := 7
  let h : ℝ := 9
  let π_approx : ℝ := 3.14
  let cylinder_volume : ℝ := π * r * r * h
  let wedge_volume := (1 / 3) * cylinder_volume 
  let approximated_wedge_volume := wedge_volume * π_approx/π
  in
  approximated_wedge_volume = 461.58 :=
by
  unfold wedge_volume approximated_wedge_volume
  sorry

end volume_of_wedge_l86_86363


namespace right_triangle_acute_angle_l86_86914

theorem right_triangle_acute_angle (x : ℝ) 
  (h1 : 5 * x = 90) : x = 18 :=
by sorry

end right_triangle_acute_angle_l86_86914


namespace one_correct_proposition_l86_86656

-- Define the propositions as boolean values
def proposition1 (m n : Type) (α : Type) [Line m] [Line n] [Plane α] :
  Prop := (Plane.is_parallel_to m α) ∧ (Plane.is_parallel_to n α) → ¬Line.intersects m n

def proposition2 (m n : Type) (α : Type) [Line m] [Line n] [Plane α] :
  Prop := (Line.is_perpendicular_to m α) ∧ (Line.is_perpendicular_to n α) → Line.is_parallel_to m n

def proposition3 (m n : Type) (α β : Type) [Line m] [Line n] [Plane α] [Plane β] :
  Prop := (Plane.is_parallel_to α β) ∧ (Line.is_in m α) ∧ (Line.is_in n β) → Line.is_parallel_to m n

def proposition4 (m n : Type) (α β : Type) [Line m] [Line n] [Plane α] [Plane β] :
  Prop := (Plane.is_perpendicular_to α β) ∧ (Line.is_perpendicular_to m α) ∧ (Line.is_perpendicular_to n β) → (Line.is_perpendicular_to n β)

-- Prove the number of correct propositions is 1
theorem one_correct_proposition :
  (proposition1 m n α ∨ proposition2 m n α ∨ proposition3 m n α ∨ proposition4 m n α)
  ∧ ¬(proposition1 m n α ∧ proposition2 m n α)
  ∧ ¬(proposition1 m n α ∧ proposition3 m n α)
  ∧ ¬(proposition1 m n α ∧ proposition4 m n α)
  ∧ ¬(proposition2 m n α ∧ proposition3 m n α)
  ∧ ¬(proposition2 m n α ∧ proposition4 m n α)
  ∧ ¬(proposition3 m n α ∧ proposition4 m n α) :=
sorry

end one_correct_proposition_l86_86656


namespace max_min_values_l86_86329

theorem max_min_values (x y : ℝ) (h : x^2 + y^2 - 4 * x + 2 * y + 2 = 0) :
  (∃ θ : ℝ, x = 2 + sqrt 3 * cos θ ∧ y = -1 + sqrt 3 * sin θ) ∧
  (∀ θ, x + y ≤ 1 + sqrt 6) ∧
  (∀ θ, x + y ≥ 1 - sqrt 6) := sorry

end max_min_values_l86_86329


namespace least_k_l86_86828

noncomputable def sequence (b : ℕ → ℝ) : Prop :=
  b 1 = 2 ∧ ∀ n ≥ 1, 3^(b (n + 1) - b n) = 1 + 1/(n + 1/2)

theorem least_k (b : ℕ → ℝ) (k : ℕ) : sequence b → k > 1 → b k ∈ ℤ → k = 3 :=
by
  sorry

end least_k_l86_86828


namespace problem1_problem2_l86_86583

-- Problem 1: Evaluating an integer arithmetic expression
theorem problem1 : (1 * (-8)) - (-6) + (-3) = -5 := 
by
  sorry

-- Problem 2: Evaluating a mixed arithmetic expression with rational numbers and decimals
theorem problem2 : (5 / 13) - 3.7 + (8 / 13) - (-1.7) = -1 := 
by
  sorry

end problem1_problem2_l86_86583


namespace calculation_l86_86581

theorem calculation :
  7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end calculation_l86_86581


namespace largest_number_with_unique_digits_sum_23_is_986_l86_86502

theorem largest_number_with_unique_digits_sum_23_is_986 :
  ∃ (n : ℕ), (∀ (d : ℕ), d ∈ (nat.digits 10 n) → d < 10) ∧
             (list.nodup (nat.digits 10 n)) ∧
             (list.sum (nat.digits 10 n) = 23) ∧
             n = 986 :=
by
  sorry

end largest_number_with_unique_digits_sum_23_is_986_l86_86502


namespace eval_ceil_floor_l86_86174

-- Definitions of the ceiling and floor functions and their relevant properties
def ceil (x : ℝ) : ℤ := ⌈x⌉
def floor (x : ℝ) : ℤ := ⌊x⌋

-- Mathematical statement to prove
theorem eval_ceil_floor : ceil (7 / 3) + floor (- (7 / 3)) = 0 :=
by
  -- Proof is omitted
  sorry

end eval_ceil_floor_l86_86174


namespace stratified_sampling_third_year_students_l86_86105

theorem stratified_sampling_third_year_students
  (total_students : ℕ)
  (first_year_students : ℕ)
  (second_year_male_students : ℕ)
  (prob_female_second_year : ℝ)
  (total_sampling_students : ℕ)
  (h1 : total_students = 1000)
  (h2 : first_year_students = 380)
  (h3 : second_year_male_students = 180)
  (h4 : prob_female_second_year = 0.19)
  (h5 : total_sampling_students = 100) :
  let second_year_female_students := total_students * prob_female_second_year in
  let total_second_year_students := second_year_male_students + second_year_female_students in
  let third_year_students := total_students - total_second_year_students - first_year_students in
  let third_year_sampling_students := third_year_students * total_sampling_students / total_students in
  third_year_sampling_students = 25 := by
  sorry

end stratified_sampling_third_year_students_l86_86105


namespace change_in_average_l86_86765

def lisa_scores := (92 : ℝ, 89 : ℝ, 93 : ℝ, 95 : ℝ)
def first_three_average (a b c : ℝ) : ℝ := (a + b + c) / 3
def four_scores_average (a b c d : ℝ) : ℝ := (a + b + c + d) / 4

theorem change_in_average :
  let (s1, s2, s3, s4) := lisa_scores in
  four_scores_average s1 s2 s3 s4 - first_three_average s1 s2 s3 = 0.92 :=
by
  sorry

end change_in_average_l86_86765


namespace calculate_value_l86_86580

theorem calculate_value : ((3^2 * 20.4) - (5100 / 102) + (fact 4)) * 17 = 2679.2 := by
  sorry

end calculate_value_l86_86580


namespace expensive_stock_price_l86_86395

theorem expensive_stock_price (x : ℕ) 
  (price_relation : ∀ x, 14 * (2 * x) + 26 * x = 2106) : 
  2 * x = 78 :=
by {
  have h1 : 54 * x = 2106, from price_relation x,
  have h2 : x = 39,
  { exact (nat.eq_of_mul_eq_mul_right _ h1) },
  rw h2,
  exact calc 2 * 39 = 78 : by norm_num
}

end expensive_stock_price_l86_86395


namespace solve_for_y_l86_86873

theorem solve_for_y (x y : ℝ) (h1 : x * y = 1) (h2 : x / y = 36) (h3 : 0 < x) (h4 : 0 < y) : 
  y = 1 / 6 := 
sorry

end solve_for_y_l86_86873


namespace equidistant_point_l86_86910

theorem equidistant_point (x y : ℝ) :
  (abs x = abs y) → (abs x = abs (x + y - 3) / (Real.sqrt 2)) → x = 1.5 :=
by {
  -- proof omitted
  sorry
}

end equidistant_point_l86_86910


namespace solution_set_f_le_g_l86_86378

noncomputable def floor (x : ℝ) := (⌊x⌋ : ℝ)

def f (x : ℝ) := floor x * (x - floor x)
def g (x : ℝ) := x - 1

theorem solution_set_f_le_g :
  { x : ℝ | 0 ≤ x ∧ x ≤ 2012 ∧ f x ≤ g x } = { x : ℝ | 1 ≤ x ∧ x ≤ 2012 } :=
by
  sorry

end solution_set_f_le_g_l86_86378


namespace sin_double_angle_l86_86634

theorem sin_double_angle (α : ℝ) :
  cos (π / 4 - α) = 4 / 5 → sin (2 * α) = 7 / 25 :=
by
  sorry

end sin_double_angle_l86_86634


namespace option_B_is_monotonically_increasing_l86_86927

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

def f_A (x : ℝ) : ℝ := |x|
def f_B (x : ℝ) : ℝ := x^3
def f_C (x : ℝ) : ℝ := real.log x / real.log 2
def f_D (x : ℝ) : ℝ := (1/2)^x

theorem option_B_is_monotonically_increasing :
  is_monotonically_increasing f_B :=
sorry

end option_B_is_monotonically_increasing_l86_86927


namespace number_of_primes_squared_between_8000_and_12000_l86_86318

-- Define the conditions
def n_min := 90 -- the smallest integer such that n^2 > 8000
def n_max := 109 -- the largest integer such that n^2 < 12000

-- Check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Find the count of primes whose squares lie between 8000 and 12000
def primes_in_range_count : ℕ := 
  (List.filter (λ n, is_prime n) (List.range (n_max - n_min + 1)).map (λ x, x + n_min)).length

-- Define the theorem to match the problem statement
theorem number_of_primes_squared_between_8000_and_12000 : primes_in_range_count = 5 := 
  by sorry

end number_of_primes_squared_between_8000_and_12000_l86_86318


namespace smallest_positive_period_one_increasing_interval_l86_86236

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

def is_periodic_with_period (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x + T) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem smallest_positive_period :
  is_periodic_with_period f Real.pi :=
sorry

theorem one_increasing_interval :
  is_increasing_on f (-(Real.pi / 8)) (3 * Real.pi / 8) :=
sorry

end smallest_positive_period_one_increasing_interval_l86_86236


namespace proof_lean_problem_l86_86346

open Real

noncomputable def problem_conditions (a b c : ℝ) (A B C : ℝ) (BD : ℝ) : Prop :=
  let condition_1 := c * sin ((A + C) / 2) = b * sin C
  let condition_2 := BD = 1
  let condition_3 := b = sqrt 3
  let condition_4 := BD = b * sin A
  let condition_5 := A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0
  condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5

noncomputable def find_B (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (h : problem_conditions a b c A B C BD) : Prop :=
  B = π / 3

noncomputable def find_perimeter (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (h : problem_conditions a b c A B C BD) : ℝ := 
  a + b + c = 3 + sqrt 3

theorem proof_lean_problem (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (h : problem_conditions a b c A B C BD) : 
  find_B a b c A B C BD h ∧ find_perimeter a b c A B C BD h :=
sorry

end proof_lean_problem_l86_86346


namespace bricks_required_to_pave_courtyard_l86_86102

theorem bricks_required_to_pave_courtyard :
  let courtyard_length_m := 24
  let courtyard_width_m := 14
  let brick_length_cm := 25
  let brick_width_cm := 15
  let courtyard_area_m2 := courtyard_length_m * courtyard_width_m
  let courtyard_area_cm2 := courtyard_area_m2 * 10000
  let brick_area_cm2 := brick_length_cm * brick_width_cm
  let num_bricks := courtyard_area_cm2 / brick_area_cm2
  num_bricks = 8960 := by
  {
    -- Additional context not needed for theorem statement, mock proof omitted
    sorry
  }

end bricks_required_to_pave_courtyard_l86_86102


namespace number_difference_l86_86491

theorem number_difference (a b : ℕ) (h1 : a + b = 44) (h2 : 8 * a = 3 * b) : b - a = 20 := by
  sorry

end number_difference_l86_86491


namespace initial_population_proof_l86_86713

-- Definitions and conditions based on the problem
def initial_population (P : ℝ) : ℝ :=
  let deceased := 0.1 * P in
  let not_deceased := P - deceased in
  let left_village := 0.2 * not_deceased in
  let remained_population := not_deceased - left_village in
  let returned := 0.05 * left_village in
  let current_population := remained_population + returned in
  let hospitalized := 0.08 * current_population in
  let total_including_hospitalized := current_population + hospitalized in
  total_including_hospitalized

-- Total population condition
def total_population_condition : ℝ := 3240

-- The main theorem statement
theorem initial_population_proof : 
  ∃ P : ℝ, initial_population P = total_population_condition ∧ P ≈ 4115 :=
by
  -- We state the proof exists and leave the detailed steps as a sorry for now.
  sorry

end initial_population_proof_l86_86713


namespace ratio_of_boys_l86_86124

variables {b g o : ℝ}

theorem ratio_of_boys (h1 : b = (1/2) * o)
  (h2 : g = o - b)
  (h3 : b + g + o = 1) :
  b = 1 / 4 :=
by
  sorry

end ratio_of_boys_l86_86124


namespace area_ABC_length_AC_l86_86930

variables (A B C O P T K : Point)
variables (ω : Circle)
variables (h1 : Circle O A C = ω)
variables (h2 : Segment_inter B C P)
variables (h3 : Tangent_to ω A)
variables (h4 : Tangent_to ω C)
variables (h5 : Se(Tangent_to ω A T))
variables (h6 : Se(Tangent_to ω C T))
variables (TP : Segment T P)
variables (TP_IC AC : Point)
variables (area_APK : Area (AP K) = 12)
variables (area_CPK : Area (CP K) = 9)
variables (angle_ABC : ∡ABC = arctan (3 / 7))

theorem area_ABC : Area ABC = 49 :=
sorry

theorem length_AC : AC = 7 * sqrt (5) / sqrt (6) :=
sorry

end area_ABC_length_AC_l86_86930


namespace num_roots_sqrt_eq_l86_86820

theorem num_roots_sqrt_eq (x : ℂ) :
  (sqrt(9 - x) = x * sqrt(9 - x) → x = 1 ∨ x = 9) →
  ∃ n, n = 2 :=
by
  intro h
  use 2
  sorry

end num_roots_sqrt_eq_l86_86820


namespace derivative_at_pi_over_3_l86_86015

-- Define the function f
def f (x : ℝ) : ℝ := 1 + Real.sin x

-- State the theorem to be proven
theorem derivative_at_pi_over_3 : 
  (Real.deriv f) (Real.pi / 3) = 1 / 2 :=
by 
  -- Proof goes here
  sorry

end derivative_at_pi_over_3_l86_86015


namespace mary_total_nickels_l86_86391

theorem mary_total_nickels (n1 n2 : ℕ) (h1 : n1 = 7) (h2 : n2 = 5) : n1 + n2 = 12 := by
  sorry

end mary_total_nickels_l86_86391


namespace elmo_jam_cost_l86_86601

theorem elmo_jam_cost (N B J : ℕ) (hN : N > 1) (hTotalCost : N * (6 * B + 7 * J) = 306)
: (N * J * 7) / 100 = 2.38 :=
by
sorry

end elmo_jam_cost_l86_86601


namespace ceiling_plus_floor_eq_zero_l86_86191

theorem ceiling_plus_floor_eq_zero : (Int.ceil (7 / 3) + Int.floor (-7 / 3)) = 0 := by
  sorry

end ceiling_plus_floor_eq_zero_l86_86191


namespace percentage_increase_correct_l86_86516

variable (B : ℝ) (N : ℝ)

def original_biographies : ℝ := 0.20 * B 

def after_purchase_biographies : ℝ := 0.32 * (B + N)

def calculate_N (B N : ℝ) : ℝ :=
  N - 0.32 * N = 0.32 * B - 0.20 * B

def percentage_increase (B N : ℝ) : ℝ :=
  let N_val := 0.12 * B / 0.68
  (N_val / (0.20 * B)) * 100

theorem percentage_increase_correct (B N : ℝ) :
  percentage_increase B N = 88.24 := by
  sorry

end percentage_increase_correct_l86_86516


namespace inverse_square_variation_value_of_x_given_y_l86_86086

theorem inverse_square_variation (k : ℝ) (h1 : k = 0.5625 * 16) :
  (∀ y : ℝ, x = k / y^2) → x = 1 := by
  intros h
  have hk : k = 9 := by rw [h1]; norm_num
  have hy : y = 3 := by norm_num
  sorry

-- Given conditions
def y : ℝ := 3
def x (k : ℝ) (y : ℝ) : ℝ := k / y^2

-- What needs to be proved
theorem value_of_x_given_y : (x 9 3) = 1 := by
  simp [x]
  norm_num
  sorry

end inverse_square_variation_value_of_x_given_y_l86_86086


namespace perpendicular_condition_projection_condition_l86_86283

-- Definitions based on conditions
def vector_a (λ : ℝ) : (ℝ × ℝ) := (λ, 3)
def vector_b : (ℝ × ℝ) := (-2, 4)

-- Theorem Statement 1
theorem perpendicular_condition (λ : ℝ) (h : (2 * (vector_a λ).1 + vector_b.1, 2 * (vector_a λ).2 + vector_b.2) • vector_b = 0) :
  λ = 11 := sorry

-- Theorem Statement 2 with given λ = 4
theorem projection_condition (λ : ℝ) (h : λ = 4) :
  let a := vector_a λ in
  let b := vector_b in
  (a • b / (real.sqrt ((b.1 ^ 2 + b.2 ^ 2) * (a.1 ^ 2 + a.2 ^ 2)))) = 2 * real.sqrt 5 := sorry

end perpendicular_condition_projection_condition_l86_86283


namespace phi_range_proof_l86_86661

noncomputable def phi_range (phi : ℝ) :=
  abs phi < real.pi

theorem phi_range_proof : 
  (∀ (x : ℝ), ( real.pi / 5 < x ∧ x < 5 * real.pi / 8) →
    ∃ (y : ℝ), 
      y = -2 * real.sin (2 * x + real.pi / 4) ∧ 
      real.pi / 10 ≤ phi ∧ phi ≤ real.pi / 4) :=
by
  sorry

end phi_range_proof_l86_86661


namespace smallest_positive_period_pi_interval_monotonic_increase_l86_86272

-- Define the function f(x)
def f (x : ℝ) : ℝ := cos (2 * x - π / 3) + 2 * sin (x + π / 4) * sin (x - π / 4)

-- Prove that the smallest positive period of f(x) is π
theorem smallest_positive_period_pi : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = π :=
by
  sorry

-- Prove that the interval of monotonic increase of f(x) on [-π/4, π/4] is [-π/6, π/4]
theorem interval_monotonic_increase : Icc (-π / 4) (π / 4) ⊆ Icc (-π / 6) (π / 4) :=
by
  sorry

end smallest_positive_period_pi_interval_monotonic_increase_l86_86272


namespace percentage_of_number_l86_86089

noncomputable def percentage := (20 : ℝ) / 100 
noncomputable def number := 20

theorem percentage_of_number (percentage : ℝ) (number : ℝ) : percentage * number = 4 := by
  rw [percentage, number]
  norm_num
  sorry

end percentage_of_number_l86_86089


namespace sum_of_interior_angles_of_hexagon_l86_86466

theorem sum_of_interior_angles_of_hexagon
  (n : ℕ)
  (h : n = 6) :
  (n - 2) * 180 = 720 := by
  sorry

end sum_of_interior_angles_of_hexagon_l86_86466


namespace find_angle_B_find_perimeter_l86_86351

-- Definitions of the variables and conditions
variable {α : Type} [LinearOrder α] [Field α] [TrigonometricFunctions α] -- for trigonometric functions

-- Definitions for the angles and sides of triangle ABC
variables (A B C a b c : α)

-- Conditions provided in the problem
variable (h1 : c * sin ((A + C) / 2) = b * sin C)
variable (h2 : BD = 1)
variable (h3 : b = sqrt 3)

-- Proving the angle B
theorem find_angle_B : B = π / 3 :=
sorry

-- Proving the perimeter of triangle ABC
theorem find_perimeter (B_eq : B = π / 3) : a + b + c = 3 + sqrt 3 :=
sorry

end find_angle_B_find_perimeter_l86_86351


namespace at_least_3_defective_correct_l86_86704

/-- Number of products in batch -/
def total_products : ℕ := 50

/-- Number of defective products -/
def defective_products : ℕ := 4

/-- Number of products drawn -/
def drawn_products : ℕ := 5

/-- Number of ways to draw at least 3 defective products out of 5 -/
def num_ways_at_least_3_defective : ℕ :=
  (Nat.choose defective_products 4) * (Nat.choose (total_products - defective_products) 1) +
  (Nat.choose defective_products 3) * (Nat.choose (total_products - defective_products) 2)

theorem at_least_3_defective_correct : num_ways_at_least_3_defective = 4186 := by
  sorry

end at_least_3_defective_correct_l86_86704


namespace dima_and_serezha_meet_time_l86_86962

-- Define the conditions and the main theorem to be proven.
theorem dima_and_serezha_meet_time :
  let dima_run_time := 15 / 60.0 -- Dima runs for 15 minutes
  let dima_run_speed := 6.0 -- Dima's running speed is 6 km/h
  let serezha_boat_speed := 20.0 -- Serezha's boat speed is 20 km/h
  let serezha_boat_time := 30 / 60.0 -- Serezha's boat time is 30 minutes
  let common_run_speed := 6.0 -- Both run at 6 km/h towards each other
  let distance_to_meet := dima_run_speed * dima_run_time -- Distance Dima runs along the shore
  let total_time := distance_to_meet / (common_run_speed + common_run_speed) -- Time until they meet after parting
  total_time = 7.5 / 60.0 := -- 7.5 minutes converted to hours
sorry

end dima_and_serezha_meet_time_l86_86962


namespace james_worked_41_hours_l86_86599

theorem james_worked_41_hours (x : ℝ) :
  ∃ (J : ℕ), 
    (24 * x + 12 * 1.5 * x = 40 * x + (J - 40) * 2 * x) ∧ 
    J = 41 := 
by 
  sorry

end james_worked_41_hours_l86_86599


namespace fraction_equivalence_proof_l86_86005

def numerator_of_fraction : ℕ :=
  let x := 12 in x

theorem fraction_equivalence_proof :
  ∀ (x : ℕ),
  (∀ (d : ℕ), d = 2 * x + 4 → x = 12) ∧ (x / (2 * x + 4) = 3 / 7) :=
by
  assume x,
  cases x,
  { sorry },
  { sorry }

end fraction_equivalence_proof_l86_86005


namespace count_valid_N_l86_86298

theorem count_valid_N : 
  ∃ (N : ℕ), (N < 2000) ∧ (∃ (x : ℝ), x^⌊x⌋₊ = N) :=
begin
  sorry
end

end count_valid_N_l86_86298


namespace eval_expr_eq_zero_l86_86164

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l86_86164


namespace ab_bisects_cd_l86_86490

open Classical

noncomputable def bisect_point {α : Type*} [metric_space α] (A B C D : α) (Γ₁ Γ₂ : set α)
  (h₁ : is_circle_Γ₁) (h₂ : is_circle_Γ₂) (A ∈ Γ₁ ∩ Γ₂) (B ∈ Γ₁ ∩ Γ₂)
  (C ∈ tangent_point Γ₁) (D ∈ tangent_point Γ₂) : Prop :=
tangent C D → ∃ M : α, M ∈ line AB ∧ M ∈ line CD ∧ equidistant M C D

theorem ab_bisects_cd {α : Type*} [metric_space α] (A B C D : α) (Γ₁ Γ₂ : set α) :
  is_circle_Γ₁ → is_circle_Γ₂ → A ∈ Γ₁ ∩ Γ₂ → B ∈ Γ₁ ∩ Γ₂ → C ∈ tangent_point Γ₁ → D ∈ tangent_point Γ₂ →
  tangent C D → ∃ M : α, M ∈ line AB ∧ M ∈ line CD ∧ equidistant M C D :=
sorry

end ab_bisects_cd_l86_86490


namespace triangle_sides_arithmetic_progression_l86_86150

theorem triangle_sides_arithmetic_progression (a d : ℤ) (h : 3 * a = 15) (h1 : a > 0) (h2 : d ≥ 0) :
  (a - d = 5 ∨ a - d = 4 ∨ a - d = 3) ∧ 
  (a = 5) ∧ 
  (a + d = 5 ∨ a + d = 6 ∨ a + d = 7) := 
  sorry

end triangle_sides_arithmetic_progression_l86_86150


namespace arcsin_one_half_l86_86945

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  -- Conditions
  have h1 : -Real.pi / 2 ≤ Real.pi / 6 ∧ Real.pi / 6 ≤ Real.pi / 2 := by
    -- Proof the range of pi/6 is within [-pi/2, pi/2]
    sorry
  have h2 : ∀ x, Real.sin x = 1 / 2 → x = Real.pi / 6 := by
    -- Proof sin(pi/6) = 1 / 2
    sorry
  show Real.arcsin (1 / 2) = Real.pi / 6
  -- Proof arcsin(1/2) = pi/6 based on the above conditions
  sorry

end arcsin_one_half_l86_86945


namespace total_teaching_hours_l86_86969

-- Define the durations of the classes
def eduardo_math_classes : ℕ := 3
def eduardo_science_classes : ℕ := 4
def eduardo_history_classes : ℕ := 2

def math_class_duration : ℕ := 1
def science_class_duration : ℚ := 1.5
def history_class_duration : ℕ := 2

-- Define Eduardo's teaching time
def eduardo_total_time : ℚ :=
  eduardo_math_classes * math_class_duration +
  eduardo_science_classes * science_class_duration +
  eduardo_history_classes * history_class_duration

-- Define Frankie's teaching time (double the classes of Eduardo)
def frankie_total_time : ℚ :=
  2 * (eduardo_math_classes * math_class_duration) +
  2 * (eduardo_science_classes * science_class_duration) +
  2 * (eduardo_history_classes * history_class_duration)

-- Define the total teaching time for both Eduardo and Frankie
def total_teaching_time : ℚ :=
  eduardo_total_time + frankie_total_time

-- Theorem statement that both their total teaching time is 39 hours
theorem total_teaching_hours : total_teaching_time = 39 :=
by
  -- skipping the proof using sorry
  sorry

end total_teaching_hours_l86_86969


namespace general_term_a_seq_a_seq_inequality_l86_86762

noncomputable def a_seq (n : ℕ) : ℝ :=
  if n = 1 then 1 / 2
  else if n = 2 then 3 / 8
  else sorry -- Define the recurrence relation here

theorem general_term_a_seq (n : ℕ) (h : 1 ≤ n) : 
  a_seq n = (2 * n - 1)‼ / (2 * n)‼ :=
sorry

theorem a_seq_inequality (n : ℕ) (h : 1 ≤ n) : 
  0 < a_seq n ∧ a_seq n < 1 / Real.sqrt (2 * n + 1) :=
sorry

end general_term_a_seq_a_seq_inequality_l86_86762


namespace least_four_digit_with_factors_l86_86058

open Nat

theorem least_four_digit_with_factors (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000) 
  (h3 : 3 ∣ n) 
  (h4 : 5 ∣ n) 
  (h5 : 7 ∣ n) : n = 1050 :=
by
  sorry

end least_four_digit_with_factors_l86_86058


namespace base_6_to_10_conversion_l86_86501

theorem base_6_to_10_conversion : 
  ∀ (n : ℕ), n = 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0 → n = 1295 :=
by
  intro n h
  sorry

end base_6_to_10_conversion_l86_86501


namespace find_equation_of_ellipse_and_k_value_l86_86665

noncomputable def find_ellipse_equation : Prop :=
  let l : ℝ → ℝ := λ x, k * x + sqrt 3
  let C : ℝ → ℝ → Prop := λ x y, (x * x + (y * y) / 4 = 1)
  let F : ℝ × ℝ := (0, sqrt 3)
  
  (F = (0, sqrt 3)) ∧ ((F = (C: 0, sqrt 3)))

noncomputable def find_k_value : Prop :=
  let l : ℝ → ℝ := λ x, k * x + sqrt 3
  let C : ℝ → ℝ → Prop := λ x y, (x * x + (y * y) / 4 = 1)

  ∃ k, (k = sqrt 11 / 2 ∨ k = - sqrt 11 / 2) ∧
  let circle : (ℝ × ℝ) → Prop := λ p, let (x, y) := p in (x - y) * (x - y)

  circle = (0, k)

theorem find_equation_of_ellipse_and_k_value :
  find_ellipse_equation ∧ find_k_value :=
begin
  sorry
end

end find_equation_of_ellipse_and_k_value_l86_86665


namespace white_squares_in_10th_row_l86_86890

theorem white_squares_in_10th_row :
  ∀ n, (N: ℕ) -> (N = 2 * n - 1) -> (W: ℕ) -> (row_starts_and_ends_black : true) ->
  (row_alternates_colors : true) -> (n = 10) -> ⌊N / 2⌋ = 9 
:= 
begin
  intros,
  sorry
end

end white_squares_in_10th_row_l86_86890


namespace minimum_shirts_to_save_money_l86_86127

def acme_cost (x : ℕ) : ℕ := 60 + 8 * x
def delta_cost (x : ℕ) : ℕ := 12 * x

theorem minimum_shirts_to_save_money : ∃ (x : ℕ), x = 16 ∧ acme_cost x < delta_cost x :=
by
  existsi 16
  dsimp [acme_cost, delta_cost]
  linarith

end minimum_shirts_to_save_money_l86_86127


namespace cost_per_square_foot_l86_86362

theorem cost_per_square_foot (skirts : ℕ) (skirt_length skirt_width : ℕ) (bodice_shirt bodice_sleeves : ℕ) (total_cost : ℕ)
  (h_skirts : skirts = 3)
  (h_skirt_length : skirt_length = 12)
  (h_skirt_width : skirt_width = 4)
  (h_bodice_shirt : bodice_shirt = 2)
  (h_bodice_sleeves : bodice_sleeves = 5)
  (h_total_cost : total_cost = 468) : 
  let skirt_area := skirts * skirt_length * skirt_width
      bodice_area := bodice_shirt + 2 * bodice_sleeves
      total_area := skirt_area + bodice_area in
  total_cost / total_area = 3 := by
  sorry

end cost_per_square_foot_l86_86362


namespace limit_sequence_l86_86522

theorem limit_sequence :
  (Real.limit (λ n : ℕ, (↑(n:ℕ) + 7)^3 - (↑(n:ℕ) + 2)^3) /
   (Real.limit (λ n : ℕ, (3 * (↑(n:ℕ)) + 2)^2 + (4 * (↑(n:ℕ)) + 1)^2)) =
   3 / 5) :=
begin
  sorry
end

end limit_sequence_l86_86522


namespace chord_count_l86_86495

theorem chord_count {n : ℕ} (h : n = 2024) : 
  ∃ k : ℕ, k ≥ 1024732 ∧ ∀ (i j : ℕ), (i < n → j < n → i ≠ j → true) := sorry

end chord_count_l86_86495


namespace consumption_increase_percentage_l86_86038

theorem consumption_increase_percentage :
  ∀ (T C : ℝ) (P : ℝ),
  ((0.85 * T) * (C * (1 + P / 100)) = 1.065 * (T * C)) ->
  P ≈ 25.29 :=
by
  intros T C P h
  sorry

end consumption_increase_percentage_l86_86038


namespace gas_consumption_reduction_l86_86521

-- Define initial conditions
def P : ℝ := 1 -- Normalized original price
def C : ℝ := 1 -- Normalized original consumption

-- Price changes
def priceAfterFirstIncrease : ℝ := P * 1.15
def priceAfterSecondIncrease : ℝ := priceAfterFirstIncrease * 1.10

-- New consumption required to keep expenditure constant
def C_new : ℝ := C / priceAfterSecondIncrease

-- Percentage reduction in consumption
def percentageReduction : ℝ := (1 - C_new) * 100

-- Prove that the percentage reduction is approximately 20.95%
theorem gas_consumption_reduction :
  |percentageReduction - 20.95| < 0.01 :=
by
  sorry

end gas_consumption_reduction_l86_86521


namespace solve_quadratic_l86_86034

theorem solve_quadratic :
  (x = 0 ∨ x = 2/5) ↔ (5 * x^2 - 2 * x = 0) :=
by
  sorry

end solve_quadratic_l86_86034


namespace arcsin_one_half_l86_86947

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l86_86947


namespace probability_not_passing_third_quadrant_l86_86584

theorem probability_not_passing_third_quadrant : 
  let k_values := [-2, -1, 1, -3] in
  let suitable_k := k_values.filter (λ k, k < 0) in
  (suitable_k.length : ℚ) / (k_values.length : ℚ) = 3 / 4 :=
by
  sorry

end probability_not_passing_third_quadrant_l86_86584


namespace blue_jellybeans_l86_86907

theorem blue_jellybeans (total_jellybeans purple_jellybeans orange_jellybeans red_jellybeans : ℕ) 
  (h_purple: purple_jellybeans = 26) 
  (h_orange: orange_jellybeans = 40) 
  (h_red: red_jellybeans = 120)
  (h_total: total_jellybeans = 200) 
  : ∃ blue_jellybeans : ℕ, blue_jellybeans = total_jellybeans - (purple_jellybeans + orange_jellybeans + red_jellybeans) ∧ blue_jellybeans = 14 :=
begin
  use 14,
  split,
  { 
    rw [h_purple, h_orange, h_red, h_total],
    norm_num,
  },
  refl
end

end blue_jellybeans_l86_86907


namespace solute_volume_correct_nearest_hundredth_l86_86904

def cylinder_solute_volume 
  (height : ℝ) (diameter : ℝ) (full_fraction : ℝ) (ratio_solute_solvent : ℝ) : ℝ :=
  let radius := diameter / 2
  let filled_height := height * full_fraction
  let volume_solution := Real.pi * (radius ^ 2) * filled_height
  let solute_fraction := ratio_solute_solvent / (1 + ratio_solute_solvent)
  let volume_solute := volume_solution * solute_fraction
  realApprox (volume_solute) -- Using a hypothetical realApprox function representing the nearest hundredth approximation

theorem solute_volume_correct_nearest_hundredth :
  cylinder_solute_volume 8 3 (1 / 4) (1 / 9) = 1.41 := by 
    sorry

end solute_volume_correct_nearest_hundredth_l86_86904


namespace range_of_a_l86_86330

-- Define the inequality condition
def inequality (a x : ℝ) : Prop := (a-2)*x^2 + 2*(a-2)*x < 4

-- The main theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, inequality a x) ↔ (-2 : ℝ) < a ∧ a ≤ 2 := by
  sorry

end range_of_a_l86_86330


namespace correct_statements_about_opposite_numbers_l86_86442

/-- Definition of opposite numbers: two numbers are opposite if one is the negative of the other --/
def is_opposite (a b : ℝ) : Prop := a = -b

theorem correct_statements_about_opposite_numbers (a b : ℝ) :
  (is_opposite a b ↔ a + b = 0) ∧
  (a + b = 0 ↔ is_opposite a b) ∧
  ((is_opposite a b ∧ a ≠ 0 ∧ b ≠ 0) ↔ (a / b = -1)) ∧
  ((a / b = -1 ∧ b ≠ 0) ↔ is_opposite a b) :=
by {
  sorry -- Proof is omitted
}

end correct_statements_about_opposite_numbers_l86_86442


namespace evaluate_expression_l86_86201

theorem evaluate_expression : Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) = 0 := by
  -- The proof part is omitted as per instructions.
  sorry

end evaluate_expression_l86_86201


namespace find_m_n_l86_86981

noncomputable def solution_set : set (ℕ × ℕ) := { (4, 4), (5, 6), (6, 5) }

theorem find_m_n :
  ∀ m n : ℕ, 
  (∀ x : ℝ, (polynomial.eval x (polynomial.C n + polynomial.X * polynomial.negate (polynomial.C m) + polynomial.X^2) = 0 → x ∈ ℕ))
  ∧ 
  (∀ x : ℝ, (polynomial.eval x (polynomial.C m + polynomial.X * polynomial.negate (polynomial.C n) + polynomial.X^2) = 0 → x ∈ ℕ))
  ↔ (m, n) ∈ solution_set :=
by
  sorry

end find_m_n_l86_86981


namespace difference_of_squares_l86_86579

-- Definition of the constants a and b as given in the problem
def a := 502
def b := 498

theorem difference_of_squares : a^2 - b^2 = 4000 := by
  sorry

end difference_of_squares_l86_86579


namespace path_length_traversed_by_vertex_Q_l86_86565

theorem path_length_traversed_by_vertex_Q :
  let R := (3, 0)
  let hypotenuse := 3 * Real.sqrt 2
  let rotations := 6 -- Rectangle's perimeter movements that cause rotations
  let arc_length_per_rotation := hypotenuse * (Real.pi / 2)
  let total_path_length := rotations * arc_length_per_rotation
  total_path_length = 9 * Real.sqrt 2 * Real.pi := 
by
  let R := (3, 0)
  let hypotenuse := 3 * Real.sqrt 2
  let rotations := 6
  let arc_length_per_rotation := hypotenuse * (Real.pi / 2)
  let total_path_length := rotations * arc_length_per_rotation
  have h1 : hypotenuse = 3 * Real.sqrt 2, 
  have h2 : arc_length_per_rotation = (3 * Real.sqrt 2) * (Real.pi / 2),
  have h3 : total_path_length = 6 * ((3 * Real.sqrt 2) * (Real.pi / 2))
  have h4 : total_path_length = (9 * Real.sqrt 2 * Real.pi),
  exact h4
  sorry

end path_length_traversed_by_vertex_Q_l86_86565


namespace seating_arrangements_l86_86339

-- Definitions based on conditions
def window_seat_options : List String := ["front", "back_left", "back_right"]
def seats : List String := ["front", "back_left", "back_middle", "back_right"]

-- Function to calculate the number of seating arrangements
def number_of_arrangements (specific_passenger : String) : Nat :=
  let remaining_seats := seats.erase specific_passenger
  remaining_seats.length * (remaining_seats.length - 1) * (remaining_seats.length - 2)

theorem seating_arrangements (h : ∀ passenger, passenger ∈ window_seat_options) : 
  ∃ (n : Nat), n = 18 :=
by
  use 18
  sorry

end seating_arrangements_l86_86339


namespace value_of_A_l86_86125

theorem value_of_A
  (A B C D E F G H I J : ℕ)
  (h_diff : ∀ x y : ℕ, x ≠ y → x ≠ y)
  (h_decreasing_ABC : A > B ∧ B > C)
  (h_decreasing_DEF : D > E ∧ E > F)
  (h_decreasing_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_consecutive_odd_DEF : D % 2 = 1 ∧ E % 2 = 1 ∧ F % 2 = 1 ∧ E = D - 2 ∧ F = E - 2)
  (h_consecutive_even_GHIJ : G % 2 = 0 ∧ H % 2 = 0 ∧ I % 2 = 0 ∧ J % 2 = 0 ∧ H = G - 2 ∧ I = H - 2 ∧ J = I - 2)
  (h_sum : A + B + C = 9) : 
  A = 8 :=
sorry

end value_of_A_l86_86125


namespace perpendicular_lines_l86_86751

theorem perpendicular_lines (a b c A B C : ℝ)
  (hA: a = 2 * (real.pi / 180) * sin A)
  (hB: b = 2 * (real.pi / 180) * sin B)
  (hC: c = 2 * (real.pi / 180) * sin C) :
  (∃ x y : ℝ, sin A * x - a * y - c = 0) →
  (∃ x y : ℝ, b * x + sin B * y + sin C = 0) →
  (∃ x y : ℝ, (sin A / a) * x * (-b / sin B) * y = -1) :=
by
  sorry

end perpendicular_lines_l86_86751


namespace find_k_l86_86671

-- Definitions of the vectors
def a : Vector ℝ 3 := ![1, 1, 0]
def b : Vector ℝ 3 := ![-1, 0, 2]

-- The condition that k * a + b is perpendicular to 2 * a - b
def is_perpendicular (v1 v2 : Vector ℝ 3) : Prop := 
  v1 ⬝ v2 = 0 -- dot product equals zero implies perpendicular

-- The main theorem to prove
theorem find_k (k : ℝ) : 
  is_perpendicular (k • a + b) (2 • a - b) ↔ k = 1 :=
sorry

end find_k_l86_86671


namespace matrix_product_sequence_l86_86577

-- Defining the specific type of matrix used in the problem
def upper_triangular_matrix (x : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![1, x], ![0, 1]]

-- The main theorem statement
theorem matrix_product_sequence :
  (List.foldl Matrix.mul (1 : Matrix (Fin 2) (Fin 2) ℕ)
    (List.map upper_triangular_matrix (List.range' 2 50).map (λ n => 2 * n))) = ![![1, 2550], ![0, 1]] := 
by
  sorry

end matrix_product_sequence_l86_86577


namespace total_amount_paid_l86_86488

-- Definitions based on the conditions in part (a)
def cost_of_manicure : ℝ := 30
def tip_percentage : ℝ := 0.30

-- Proof statement based on conditions and answer in part (c)
theorem total_amount_paid : cost_of_manicure + (cost_of_manicure * tip_percentage) = 39 := by
  sorry

end total_amount_paid_l86_86488


namespace zero_in_interval_l86_86696

noncomputable def f (x : ℝ) : ℝ := log x + x - 3

theorem zero_in_interval {k : ℤ} (h₁ : ∃ x : ℝ, x ∈ (k : ℝ, (k + 1 : ℝ)) ∧ f x = 0) : k = 2 :=
  sorry

end zero_in_interval_l86_86696


namespace relationship_between_a_and_b_l86_86760

theorem relationship_between_a_and_b 
  (a b x : ℝ) 
  (hx : x > 0) 
  (h : 0 < b^x ∧ b^x < a^x ∧ a^x < 1) : 
  1 > a ∧ a > b :=
begin
  sorry
end

end relationship_between_a_and_b_l86_86760


namespace dennis_years_of_teaching_l86_86053

variable (V A D E N : ℕ)

def combined_years_taught : Prop :=
  V + A + D + E + N = 225

def virginia_adrienne_relation : Prop :=
  V = A + 9

def virginia_dennis_relation : Prop :=
  V = D - 15

def elijah_adrienne_relation : Prop :=
  E = A - 3

def elijah_nadine_relation : Prop :=
  E = N + 7

theorem dennis_years_of_teaching 
  (h1 : combined_years_taught V A D E N) 
  (h2 : virginia_adrienne_relation V A)
  (h3 : virginia_dennis_relation V D)
  (h4 : elijah_adrienne_relation E A) 
  (h5 : elijah_nadine_relation E N) : 
  D = 65 :=
  sorry

end dennis_years_of_teaching_l86_86053


namespace find_extreme_values_l86_86237

noncomputable def f (x : ℝ) : ℝ := 1/(4^x) - 1/(2^x) + 1

theorem find_extreme_values :
  let t (x: ℝ) := 1 / (2^x) in
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f x ∈ Set.Icc (3 / 4) 57) :=
begin
  intros x hx,
  sorry
end

end find_extreme_values_l86_86237


namespace max_students_can_be_equally_distributed_l86_86518

def num_pens : ℕ := 2730
def num_pencils : ℕ := 1890

theorem max_students_can_be_equally_distributed : Nat.gcd num_pens num_pencils = 210 := by
  sorry

end max_students_can_be_equally_distributed_l86_86518


namespace total_expenses_l86_86357

def tulips : ℕ := 250
def carnations : ℕ := 375
def roses : ℕ := 320
def cost_per_flower : ℕ := 2

theorem total_expenses :
  tulips + carnations + roses * cost_per_flower = 1890 := 
sorry

end total_expenses_l86_86357


namespace unique_solutions_l86_86212

noncomputable def is_solution (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ∣ (b^4 + 1) ∧ b ∣ (a^4 + 1) ∧ (Nat.floor (Real.sqrt a) = Nat.floor (Real.sqrt b))

theorem unique_solutions :
  ∀ (a b : ℕ), is_solution a b → (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by 
  sorry

end unique_solutions_l86_86212
