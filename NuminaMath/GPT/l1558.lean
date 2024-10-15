import Mathlib

namespace NUMINAMATH_GPT_mike_profit_l1558_155821

-- Define the conditions
def total_acres : ℕ := 200
def cost_per_acre : ℕ := 70
def sold_acres := total_acres / 2
def selling_price_per_acre : ℕ := 200

-- Statement to prove the profit Mike made is $6,000
theorem mike_profit :
  let total_cost := total_acres * cost_per_acre
  let total_revenue := sold_acres * selling_price_per_acre
  total_revenue - total_cost = 6000 := 
by
  sorry

end NUMINAMATH_GPT_mike_profit_l1558_155821


namespace NUMINAMATH_GPT_minimum_stamps_to_make_47_cents_l1558_155859

theorem minimum_stamps_to_make_47_cents (c f : ℕ) (h : 5 * c + 7 * f = 47) : c + f = 7 :=
sorry

end NUMINAMATH_GPT_minimum_stamps_to_make_47_cents_l1558_155859


namespace NUMINAMATH_GPT_tan_half_angle_is_two_l1558_155822

-- Define the setup
variables (α : ℝ) (H1 : α ∈ Icc (π/2) π) (H2 : 3 * Real.sin α + 4 * Real.cos α = 0)

-- Define the main theorem
theorem tan_half_angle_is_two : Real.tan (α / 2) = 2 :=
sorry

end NUMINAMATH_GPT_tan_half_angle_is_two_l1558_155822


namespace NUMINAMATH_GPT_n_power_four_plus_sixtyfour_power_n_composite_l1558_155875

theorem n_power_four_plus_sixtyfour_power_n_composite (n : ℕ) : ∃ m k, m * k = n^4 + 64^n ∧ m > 1 ∧ k > 1 :=
by
  sorry

end NUMINAMATH_GPT_n_power_four_plus_sixtyfour_power_n_composite_l1558_155875


namespace NUMINAMATH_GPT_students_with_both_uncool_parents_l1558_155807

theorem students_with_both_uncool_parents :
  let total_students := 35
  let cool_dads := 18
  let cool_moms := 22
  let both_cool := 11
  total_students - (cool_dads + cool_moms - both_cool) = 6 := by
sorry

end NUMINAMATH_GPT_students_with_both_uncool_parents_l1558_155807


namespace NUMINAMATH_GPT_task_pages_l1558_155899

theorem task_pages (A B T : ℕ) (hB : B = A + 5) (hTogether : (A + B) * 18 = T)
  (hAlone : A * 60 = T) : T = 225 :=
by
  sorry

end NUMINAMATH_GPT_task_pages_l1558_155899


namespace NUMINAMATH_GPT_taller_tree_height_l1558_155898

/-- The top of one tree is 20 feet higher than the top of another tree.
    The heights of the two trees are in the ratio 2:3.
    The shorter tree is 40 feet tall.
    Show that the height of the taller tree is 60 feet. -/
theorem taller_tree_height 
  (shorter_tree_height : ℕ) 
  (height_difference : ℕ)
  (height_ratio_num : ℕ)
  (height_ratio_denom : ℕ)
  (H1 : shorter_tree_height = 40)
  (H2 : height_difference = 20)
  (H3 : height_ratio_num = 2)
  (H4 : height_ratio_denom = 3)
  : ∃ taller_tree_height : ℕ, taller_tree_height = 60 :=
by
  sorry

end NUMINAMATH_GPT_taller_tree_height_l1558_155898


namespace NUMINAMATH_GPT_loan_payment_difference_l1558_155868

noncomputable def compounded_amount (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P + P * r * t

noncomputable def loan1_payment (P : ℝ) (r : ℝ) (n : ℝ) (t1 : ℝ) (t2 : ℝ) : ℝ :=
  let A1 := compounded_amount P r n t1
  let one_third_payment := A1 / 3
  let remaining := A1 - one_third_payment
  one_third_payment + compounded_amount remaining r n t2

noncomputable def loan2_payment (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  simple_interest_amount P r t

noncomputable def positive_difference (x y : ℝ) : ℝ :=
  if x > y then x - y else y - x

theorem loan_payment_difference: 
  ∀ P : ℝ, ∀ r1 r2 : ℝ, ∀ n : ℝ, ∀ t1 t2 : ℝ,
  P = 12000 → r1 = 0.08 → r2 = 0.09 → n = 12 → t1 = 7 → t2 = 8 →
  positive_difference 
    (loan2_payment P r2 (t1 + t2)) 
    (loan1_payment P r1 n t1 t2) = 2335 := 
by
  intros
  sorry

end NUMINAMATH_GPT_loan_payment_difference_l1558_155868


namespace NUMINAMATH_GPT_divisibility_criterion_l1558_155818

theorem divisibility_criterion (x y : ℕ) (h_two_digit : 10 ≤ x ∧ x < 100) :
  (1207 % x = 0) ↔ (x = 10 * (x / 10) + (x % 10) ∧ (x / 10)^3 + (x % 10)^3 = 344) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_criterion_l1558_155818


namespace NUMINAMATH_GPT_find_x_satisfying_inequality_l1558_155865

open Real

theorem find_x_satisfying_inequality :
  ∀ x : ℝ, 0 < x → (x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16 ↔ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_x_satisfying_inequality_l1558_155865


namespace NUMINAMATH_GPT_marble_ratio_l1558_155864

theorem marble_ratio 
  (K A M : ℕ) 
  (M_has_5_times_as_many_as_K : M = 5 * K)
  (M_has_85_marbles : M = 85)
  (M_has_63_more_than_A : M = A + 63)
  (A_needs_12_more : A + 12 = 34) :
  34 / 17 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_marble_ratio_l1558_155864


namespace NUMINAMATH_GPT_solve_inequalities_l1558_155885

theorem solve_inequalities :
  {x : ℝ // 3 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 9 * x - 8} = {x : ℝ // 3 < x ∧ x < 4} :=
sorry

end NUMINAMATH_GPT_solve_inequalities_l1558_155885


namespace NUMINAMATH_GPT_expected_value_of_unfair_die_l1558_155820

-- Define the probabilities for each face of the die.
def prob_face (n : ℕ) : ℚ :=
  if n = 8 then 5/14 else 1/14

-- Define the expected value of a roll of this die.
def expected_value : ℚ :=
  (1 / 14) * 1 + (1 / 14) * 2 + (1 / 14) * 3 + (1 / 14) * 4 + (1 / 14) * 5 + (1 / 14) * 6 + (1 / 14) * 7 + (5 / 14) * 8

-- The statement to prove: the expected value of a roll of this die is 4.857.
theorem expected_value_of_unfair_die : expected_value = 4.857 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_unfair_die_l1558_155820


namespace NUMINAMATH_GPT_total_runs_opponents_correct_l1558_155886

-- Define the scoring conditions
def team_scores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
def lost_games_scores : List ℕ := [3, 5, 7, 9, 11, 13]
def won_games_scores : List ℕ := [2, 4, 6, 8, 10, 12]

-- Define the total runs scored by opponents in lost games
def total_runs_lost_games : ℕ := (lost_games_scores.map (λ x => x + 1)).sum

-- Define the total runs scored by opponents in won games
def total_runs_won_games : ℕ := (won_games_scores.map (λ x => x / 2)).sum

-- Total runs scored by opponents (given)
def total_runs_opponents : ℕ := total_runs_lost_games + total_runs_won_games

-- The theorem to prove
theorem total_runs_opponents_correct : total_runs_opponents = 75 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_runs_opponents_correct_l1558_155886


namespace NUMINAMATH_GPT_sequence_area_formula_l1558_155842

open Real

noncomputable def S_n (n : ℕ) : ℝ := (8 / 5) - (3 / 5) * (4 / 9) ^ n

theorem sequence_area_formula (n : ℕ) :
  S_n n = (8 / 5) - (3 / 5) * (4 / 9) ^ n := sorry

end NUMINAMATH_GPT_sequence_area_formula_l1558_155842


namespace NUMINAMATH_GPT_domain_sqrt_function_l1558_155855

noncomputable def quadratic_nonneg_for_all_x (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 - a * x + 1 ≥ 0

theorem domain_sqrt_function (a : ℝ) :
  quadratic_nonneg_for_all_x a ↔ (0 ≤ a ∧ a ≤ 4) :=
by sorry

end NUMINAMATH_GPT_domain_sqrt_function_l1558_155855


namespace NUMINAMATH_GPT_sum_of_reflection_midpoint_coordinates_l1558_155811

theorem sum_of_reflection_midpoint_coordinates (P R : ℝ × ℝ) (M : ℝ × ℝ) (P' R' M' : ℝ × ℝ) :
  P = (2, 1) → R = (12, 15) → 
  M = ((P.fst + R.fst) / 2, (P.snd + R.snd) / 2) →
  P' = (-P.fst, P.snd) → R' = (-R.fst, R.snd) →
  M' = ((P'.fst + R'.fst) / 2, (P'.snd + R'.snd) / 2) →
  (M'.fst + M'.snd) = 1 := 
by 
  intros
  sorry

end NUMINAMATH_GPT_sum_of_reflection_midpoint_coordinates_l1558_155811


namespace NUMINAMATH_GPT_sum_of_cubes_eq_neg_27_l1558_155808

variable {a b c : ℝ}

-- Define the condition that k is the same for a, b, and c
def same_k (a b c k : ℝ) : Prop :=
  k = (a^3 + 9) / a ∧ k = (b^3 + 9) / b ∧ k = (c^3 + 9) / c

-- Theorem: Given the conditions, a^3 + b^3 + c^3 = -27
theorem sum_of_cubes_eq_neg_27 (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_same_k : ∃ k, same_k a b c k) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_eq_neg_27_l1558_155808


namespace NUMINAMATH_GPT_nurses_count_l1558_155893

theorem nurses_count (total : ℕ) (ratio_doc : ℕ) (ratio_nurse : ℕ) (nurses : ℕ) : 
  total = 200 → 
  ratio_doc = 4 → 
  ratio_nurse = 6 → 
  nurses = (ratio_nurse * total / (ratio_doc + ratio_nurse)) → 
  nurses = 120 := 
by 
  intros h_total h_ratio_doc h_ratio_nurse h_calc
  rw [h_total, h_ratio_doc, h_ratio_nurse] at h_calc
  simp at h_calc
  exact h_calc

end NUMINAMATH_GPT_nurses_count_l1558_155893


namespace NUMINAMATH_GPT_find_a_l1558_155824

theorem find_a (a : ℝ) : (4, -5).2 = (a - 2, a + 1).2 → a = -6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l1558_155824


namespace NUMINAMATH_GPT_rain_probability_at_most_3_days_l1558_155891

open BigOperators

def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def rain_probability := (1:ℝ)/5
noncomputable def no_rain_probability := (4:ℝ)/5

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p^k) * ((1-p)^(n-k))

theorem rain_probability_at_most_3_days :
  ∑ k in Finset.range 4, binomial_probability 31 k rain_probability = 0.544 :=
by
  sorry

end NUMINAMATH_GPT_rain_probability_at_most_3_days_l1558_155891


namespace NUMINAMATH_GPT_similar_triangles_proportionalities_l1558_155860

-- Definitions of the conditions as hypotheses
variables (A B C D E F : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
variables (AB_DE_ratio : AB / DE = 1 / 2)
variables (BC_length : BC = 2)

-- Defining the hypothesis of similarity
def SimilarTriangles (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
  ∀ (AB BC CA DE EF FD : ℝ), (AB / DE = BC / EF) ∧ (BC / EF = CA / FD) ∧ (CA / FD = AB / DE)

-- The proof statement
theorem similar_triangles_proportionalities (A B C D E F : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
  (AB_DE_ratio : AB / DE = 1 / 2)
  (BC_length : BC = 2) : 
  EF = 4 := 
by sorry

end NUMINAMATH_GPT_similar_triangles_proportionalities_l1558_155860


namespace NUMINAMATH_GPT_problem_statement_l1558_155871

theorem problem_statement (m n : ℤ) (h : |m - 2| + (n + 1)^2 = 0) : m + n = 1 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1558_155871


namespace NUMINAMATH_GPT_quadruple_nested_function_l1558_155858

def a (k : ℕ) : ℕ := (k + 1) ^ 2

theorem quadruple_nested_function (k : ℕ) (h : k = 1) : a (a (a (a (k)))) = 458329 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_quadruple_nested_function_l1558_155858


namespace NUMINAMATH_GPT_tagged_fish_ratio_l1558_155856

theorem tagged_fish_ratio (tagged_first_catch : ℕ) 
(tagged_second_catch : ℕ) (total_second_catch : ℕ) 
(h1 : tagged_first_catch = 30) (h2 : tagged_second_catch = 2) 
(h3 : total_second_catch = 50) : tagged_second_catch / total_second_catch = 1 / 25 :=
by
  sorry

end NUMINAMATH_GPT_tagged_fish_ratio_l1558_155856


namespace NUMINAMATH_GPT_percentage_of_fair_haired_employees_who_are_women_l1558_155861

variable (E : ℝ) -- Total number of employees
variable (h1 : 0.1 * E = women_with_fair_hair_E) -- 10% of employees are women with fair hair
variable (h2 : 0.25 * E = fair_haired_employees_E) -- 25% of employees have fair hair

theorem percentage_of_fair_haired_employees_who_are_women :
  (women_with_fair_hair_E / fair_haired_employees_E) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_fair_haired_employees_who_are_women_l1558_155861


namespace NUMINAMATH_GPT_find_b_l1558_155854

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 :=
by {
  -- Proof will be filled in here
  sorry
}

end NUMINAMATH_GPT_find_b_l1558_155854


namespace NUMINAMATH_GPT_kolya_made_mistake_l1558_155869

theorem kolya_made_mistake (ab cd effe : ℕ)
  (h_eq : ab * cd = effe)
  (h_eff_div_11 : effe % 11 = 0)
  (h_ab_cd_not_div_11 : ab % 11 ≠ 0 ∧ cd % 11 ≠ 0) :
  false :=
by
  -- Note: This is where the proof would go, but we are illustrating the statement only.
  sorry

end NUMINAMATH_GPT_kolya_made_mistake_l1558_155869


namespace NUMINAMATH_GPT_part_a_part_b_l1558_155897

namespace ShaltaevBoltaev

variables {s b : ℕ}

-- Condition: 175s > 125b
def condition1 (s b : ℕ) : Prop := 175 * s > 125 * b

-- Condition: 175s < 126b
def condition2 (s b : ℕ) : Prop := 175 * s < 126 * b

-- Prove that 3s + b > 80
theorem part_a (s b : ℕ) (h1 : condition1 s b) (h2 : condition2 s b) : 
  3 * s + b > 80 := sorry

-- Prove that 3s + b > 100
theorem part_b (s b : ℕ) (h1 : condition1 s b) (h2 : condition2 s b) : 
  3 * s + b > 100 := sorry

end ShaltaevBoltaev

end NUMINAMATH_GPT_part_a_part_b_l1558_155897


namespace NUMINAMATH_GPT_probability_exactly_one_instrument_l1558_155812

-- Definitions of the conditions
def total_people : ℕ := 800
def frac_one_instrument : ℚ := 1 / 5
def people_two_or_more_instruments : ℕ := 64

-- Statement of the problem
theorem probability_exactly_one_instrument :
  let people_at_least_one_instrument := frac_one_instrument * total_people
  let people_exactly_one_instrument := people_at_least_one_instrument - people_two_or_more_instruments
  let probability := people_exactly_one_instrument / total_people
  probability = 3 / 25 :=
by
  -- Definitions
  let people_at_least_one_instrument : ℚ := frac_one_instrument * total_people
  let people_exactly_one_instrument : ℚ := people_at_least_one_instrument - people_two_or_more_instruments
  let probability : ℚ := people_exactly_one_instrument / total_people
  
  -- Sorry statement to skip the proof
  exact sorry

end NUMINAMATH_GPT_probability_exactly_one_instrument_l1558_155812


namespace NUMINAMATH_GPT_eval_expr_l1558_155839

namespace ProofProblem

variables (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d = a + b + c)

theorem eval_expr :
  d = a + b + c →
  (a^3 + b^3 + c^3 - 3 * a * b * c) / (a * b * c) = (d * (a^2 + b^2 + c^2 - a * b - a * c - b * c)) / (a * b * c) :=
by
  intros hd
  sorry

end ProofProblem

end NUMINAMATH_GPT_eval_expr_l1558_155839


namespace NUMINAMATH_GPT_total_students_at_gathering_l1558_155814

theorem total_students_at_gathering (x : ℕ) 
  (h1 : ∃ x : ℕ, 0 < x)
  (h2 : (x + 6) / (2 * x + 6) = 2 / 3) : 
  (2 * x + 6) = 18 := 
  sorry

end NUMINAMATH_GPT_total_students_at_gathering_l1558_155814


namespace NUMINAMATH_GPT_ratio_platform_to_train_length_l1558_155828

variable (L P t : ℝ)

-- Definitions based on conditions
def train_has_length (L : ℝ) : Prop := true
def train_constant_velocity : Prop := true
def train_passes_pole_in_t_seconds (L t : ℝ) : Prop := L / t = L
def train_passes_platform_in_4t_seconds (L P t : ℝ) : Prop := L / t = (L + P) / (4 * t)

-- Theorem statement: ratio of the length of the platform to the length of the train is 3:1
theorem ratio_platform_to_train_length (h1 : train_has_length L) 
                                      (h2 : train_constant_velocity) 
                                      (h3 : train_passes_pole_in_t_seconds L t)
                                      (h4 : train_passes_platform_in_4t_seconds L P t) :
  P / L = 3 := 
by sorry

end NUMINAMATH_GPT_ratio_platform_to_train_length_l1558_155828


namespace NUMINAMATH_GPT_simplify_expression_l1558_155877

theorem simplify_expression : 
  ((Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2)) - 
  (Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt (2 / 3))) = -Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1558_155877


namespace NUMINAMATH_GPT_fraction_subtraction_l1558_155890

theorem fraction_subtraction (x : ℝ) : (8000 * x - (0.05 / 100 * 8000) = 796) → x = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l1558_155890


namespace NUMINAMATH_GPT_estimate_red_balls_l1558_155845

-- Define the conditions in Lean 4
def total_balls : ℕ := 15
def freq_red_ball : ℝ := 0.4

-- Define the proof statement without proving it
theorem estimate_red_balls (x : ℕ) 
  (h1 : x ≤ total_balls) 
  (h2 : ∃ (p : ℝ), p = x / total_balls ∧ p = freq_red_ball) :
  x = 6 :=
sorry

end NUMINAMATH_GPT_estimate_red_balls_l1558_155845


namespace NUMINAMATH_GPT_max_imag_part_of_roots_l1558_155804

noncomputable def polynomial (z : ℂ) : ℂ := z^12 - z^9 + z^6 - z^3 + 1

theorem max_imag_part_of_roots :
  ∃ (z : ℂ), polynomial z = 0 ∧ ∀ w, polynomial w = 0 → (z.im ≤ w.im) := sorry

end NUMINAMATH_GPT_max_imag_part_of_roots_l1558_155804


namespace NUMINAMATH_GPT_impossible_arrangement_of_numbers_l1558_155816

theorem impossible_arrangement_of_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) 
(hpos : ∀ i, 0 < a i)
(hdiff : ∃ i, ∀ j ≠ i, a j = a ((j + 1) % n) - a ((j - 1 + n) % n)):
  false :=
by
  sorry

end NUMINAMATH_GPT_impossible_arrangement_of_numbers_l1558_155816


namespace NUMINAMATH_GPT_economy_value_after_two_years_l1558_155817

/--
Given an initial amount A₀ = 3200,
that increases annually by 1/8th of itself,
with an inflation rate of 3% in the first year and 4% in the second year,
prove that the value of the amount after two years is 3771.36
-/
theorem economy_value_after_two_years :
  let A₀ := 3200 
  let increase_rate := 1 / 8
  let inflation_rate_year_1 := 0.03
  let inflation_rate_year_2 := 0.04
  let A₁ := A₀ * (1 + increase_rate)
  let V₁ := A₁ * (1 - inflation_rate_year_1)
  let A₂ := V₁ * (1 + increase_rate)
  let V₂ := A₂ * (1 - inflation_rate_year_2)
  V₂ = 3771.36 :=
by
  simp only []
  sorry

end NUMINAMATH_GPT_economy_value_after_two_years_l1558_155817


namespace NUMINAMATH_GPT_real_number_x_equal_2_l1558_155883

theorem real_number_x_equal_2 (x : ℝ) (i : ℂ) (h : i * i = -1) :
  (1 - 2 * i) * (x + i) = 4 - 3 * i → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_real_number_x_equal_2_l1558_155883


namespace NUMINAMATH_GPT_cube_painting_distinct_ways_l1558_155835

theorem cube_painting_distinct_ways : ∃ n : ℕ, n = 7 := sorry

end NUMINAMATH_GPT_cube_painting_distinct_ways_l1558_155835


namespace NUMINAMATH_GPT_brown_eyed_brunettes_count_l1558_155813

-- Definitions of conditions
variables (total_students blue_eyed_blondes brunettes brown_eyed_students : ℕ)
variable (brown_eyed_brunettes : ℕ)

-- Initial conditions
axiom h1 : total_students = 60
axiom h2 : blue_eyed_blondes = 18
axiom h3 : brunettes = 40
axiom h4 : brown_eyed_students = 24

-- Proof objective
theorem brown_eyed_brunettes_count :
  brown_eyed_brunettes = 24 - (24 - (20 - (20 - 18))) := sorry

end NUMINAMATH_GPT_brown_eyed_brunettes_count_l1558_155813


namespace NUMINAMATH_GPT_correct_student_mark_l1558_155844

theorem correct_student_mark (x : ℕ) : 
  (∀ (n : ℕ), n = 30) →
  (∀ (avg correct_avg wrong_mark correct_mark : ℕ), 
    avg = 100 ∧ 
    correct_avg = 98 ∧ 
    wrong_mark = 70 ∧ 
    (n * avg) - wrong_mark + correct_mark = n * correct_avg) →
  x = 10 := by
  intros
  sorry

end NUMINAMATH_GPT_correct_student_mark_l1558_155844


namespace NUMINAMATH_GPT_no_b_satisfies_l1558_155876

theorem no_b_satisfies (b : ℝ) : ¬ (2 * 1 - b * (-2) + 1 ≤ 0 ∧ 2 * (-1) - b * 2 + 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_no_b_satisfies_l1558_155876


namespace NUMINAMATH_GPT_functional_equation_implies_identity_l1558_155882

theorem functional_equation_implies_identity 
  (f : ℝ → ℝ) 
  (hf : ∀ x y : ℝ, 0 < x → 0 < y → 
    f ((x + y) / 2) + f ((2 * x * y) / (x + y)) = f x + f y) 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  : 2 * f (Real.sqrt (x * y)) = f x + f y := sorry

end NUMINAMATH_GPT_functional_equation_implies_identity_l1558_155882


namespace NUMINAMATH_GPT_negation_equivalence_l1558_155874

variable (U : Type) (S R : U → Prop)

-- Original statement: All students of this university are non-residents, i.e., ∀ x, S(x) → ¬ R(x)
def original_statement : Prop := ∀ x, S x → ¬ R x

-- Negation of the original statement: ∃ x, S(x) ∧ R(x)
def negated_statement : Prop := ∃ x, S x ∧ R x

-- Lean statement to prove that the negation of the original statement is equivalent to some students are residents
theorem negation_equivalence : ¬ original_statement U S R = negated_statement U S R :=
sorry

end NUMINAMATH_GPT_negation_equivalence_l1558_155874


namespace NUMINAMATH_GPT_Carmen_average_speed_l1558_155887

/-- Carmen participates in a two-part cycling race. In the first part, she covers 24 miles in 3 hours.
    In the second part, due to fatigue, her speed decreases, and she takes 4 hours to cover 16 miles.
    Calculate Carmen's average speed for the entire race. -/
theorem Carmen_average_speed :
  let distance1 := 24 -- miles in the first part
  let time1 := 3 -- hours in the first part
  let distance2 := 16 -- miles in the second part
  let time2 := 4 -- hours in the second part
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 40 / 7 :=
by
  sorry

end NUMINAMATH_GPT_Carmen_average_speed_l1558_155887


namespace NUMINAMATH_GPT_unique_root_iff_k_eq_4_l1558_155851

theorem unique_root_iff_k_eq_4 (k : ℝ) : 
  (∃! x : ℝ, x^2 - 4 * x + k = 0) ↔ k = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_unique_root_iff_k_eq_4_l1558_155851


namespace NUMINAMATH_GPT_ratio_problem_l1558_155847

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end NUMINAMATH_GPT_ratio_problem_l1558_155847


namespace NUMINAMATH_GPT_tyrone_money_l1558_155831

def bill_value (count : ℕ) (val : ℝ) : ℝ :=
  count * val

def total_value : ℝ :=
  bill_value 2 1 + bill_value 1 5 + bill_value 13 0.25 + bill_value 20 0.10 + bill_value 8 0.05 + bill_value 35 0.01

theorem tyrone_money : total_value = 13 := by 
  sorry

end NUMINAMATH_GPT_tyrone_money_l1558_155831


namespace NUMINAMATH_GPT_alpha_eq_beta_l1558_155884

variable {α β : ℝ}

theorem alpha_eq_beta
  (h_alpha : 0 < α ∧ α < (π / 2))
  (h_beta : 0 < β ∧ β < (π / 2))
  (h_sin : Real.sin (α + β) + Real.sin (α - β) = Real.sin (2 * β)) :
  α = β :=
by
  sorry

end NUMINAMATH_GPT_alpha_eq_beta_l1558_155884


namespace NUMINAMATH_GPT_amount_p_l1558_155809

variable (P : ℚ)

/-- p has $42 more than what q and r together would have had if both q and r had 1/8 of what p has.
    We need to prove that P = 56. -/
theorem amount_p (h : P = (1/8 : ℚ) * P + (1/8) * P + 42) : P = 56 :=
by
  sorry

end NUMINAMATH_GPT_amount_p_l1558_155809


namespace NUMINAMATH_GPT_maximize_net_income_l1558_155829

-- Define the conditions of the problem
def bicycles := 50
def management_cost := 115

def rental_income (x : ℕ) : ℕ :=
if x ≤ 6 then bicycles * x
else (bicycles - 3 * (x - 6)) * x

def net_income (x : ℕ) : ℤ :=
rental_income x - management_cost

-- Define the domain of the function
def domain (x : ℕ) : Prop := 3 ≤ x ∧ x ≤ 20

-- Define the piecewise function for y = f(x)
def f (x : ℕ) : ℤ :=
if 3 ≤ x ∧ x ≤ 6 then 50 * x - 115
else if 6 < x ∧ x ≤ 20 then -3 * x * x + 68 * x - 115
else 0  -- Out of domain

-- The theorem that we need to prove
theorem maximize_net_income :
  (∀ x, domain x → net_income x = f x) ∧
  (∃ x, domain x ∧ (∀ y, domain y → net_income y ≤ net_income x) ∧ x = 11) :=
by
  sorry

end NUMINAMATH_GPT_maximize_net_income_l1558_155829


namespace NUMINAMATH_GPT_no_exact_cover_l1558_155838

theorem no_exact_cover (large_w : ℕ) (large_h : ℕ) (small_w : ℕ) (small_h : ℕ) (n : ℕ) :
  large_w = 13 → large_h = 7 → small_w = 2 → small_h = 3 → n = 15 →
  ¬ (small_w * small_h * n = large_w * large_h) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_no_exact_cover_l1558_155838


namespace NUMINAMATH_GPT_equation_of_tangent_circle_l1558_155878

/-- Lean Statement for the circle problem -/
theorem equation_of_tangent_circle (center_C : ℝ × ℝ)
    (h1 : ∃ x, center_C = (x, 0) ∧ x - 0 + 1 = 0)
    (circle_tangent : ∃ r, ((2 - (center_C.1))^2 + (3 - (center_C.2))^2 = (2 * Real.sqrt 2) + r)) :
    ∃ r, (x + 1)^2 + y^2 = r^2 := 
sorry

end NUMINAMATH_GPT_equation_of_tangent_circle_l1558_155878


namespace NUMINAMATH_GPT_population_of_metropolitan_county_l1558_155843

theorem population_of_metropolitan_county : 
  let average_population := 5500
  let two_populous_cities_population := 2 * average_population
  let remaining_cities := 25 - 2
  let remaining_population := remaining_cities * average_population
  let total_population := (2 * two_populous_cities_population) + remaining_population
  total_population = 148500 := by
sorry

end NUMINAMATH_GPT_population_of_metropolitan_county_l1558_155843


namespace NUMINAMATH_GPT_solve_pow_problem_l1558_155840

theorem solve_pow_problem : (-2)^1999 + (-2)^2000 = 2^1999 := 
sorry

end NUMINAMATH_GPT_solve_pow_problem_l1558_155840


namespace NUMINAMATH_GPT_find_principal_sum_l1558_155879

theorem find_principal_sum (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) 
  (hSI : SI = 4016.25) (hR : R = 9) (hT : T = 5) : P = 8925 := 
by
  sorry

end NUMINAMATH_GPT_find_principal_sum_l1558_155879


namespace NUMINAMATH_GPT_number_of_classes_l1558_155852

theorem number_of_classes (x : ℕ) (total_games : ℕ) (h : total_games = 45) :
  (x * (x - 1)) / 2 = total_games → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_classes_l1558_155852


namespace NUMINAMATH_GPT_price_change_38_percent_l1558_155830

variables (P : ℝ) (x : ℝ)
noncomputable def final_price := P * (1 - (x / 100)^2) * 0.9
noncomputable def target_price := 0.77 * P

theorem price_change_38_percent (h : final_price P x = target_price P):
  x = 38 := sorry

end NUMINAMATH_GPT_price_change_38_percent_l1558_155830


namespace NUMINAMATH_GPT_average_ABC_is_three_l1558_155872
-- Import the entirety of the Mathlib library

-- Define the required conditions and the theorem to be proved
theorem average_ABC_is_three (A B C : ℝ) 
    (h1 : 2012 * C - 4024 * A = 8048) 
    (h2 : 2012 * B + 6036 * A = 10010) : 
    (A + B + C) / 3 = 3 := 
by
  sorry

end NUMINAMATH_GPT_average_ABC_is_three_l1558_155872


namespace NUMINAMATH_GPT_range_of_m_l1558_155819

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, 2^x - m + 1 > 0
def proposition_q (m : ℝ) : Prop := 5 - 2*m > 1

theorem range_of_m (m : ℝ) (hp : proposition_p m) (hq : proposition_q m) : m ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1558_155819


namespace NUMINAMATH_GPT_dan_remaining_money_l1558_155881

noncomputable def calculate_remaining_money (initial_amount : ℕ) : ℕ :=
  let candy_bars_qty := 5
  let candy_bar_price := 125
  let candy_bars_discount := 10
  let gum_qty := 3
  let gum_price := 80
  let soda_qty := 4
  let soda_price := 240
  let chips_qty := 2
  let chip_price := 350
  let chips_discount := 15
  let low_tax := 7
  let high_tax := 12

  let total_candy_bars_cost := candy_bars_qty * candy_bar_price
  let discounted_candy_bars_cost := total_candy_bars_cost * (100 - candy_bars_discount) / 100

  let total_gum_cost := gum_qty * gum_price

  let total_soda_cost := soda_qty * soda_price

  let total_chips_cost := chips_qty * chip_price
  let discounted_chips_cost := total_chips_cost * (100 - chips_discount) / 100

  let candy_bars_tax := discounted_candy_bars_cost * low_tax / 100
  let gum_tax := total_gum_cost * low_tax / 100

  let soda_tax := total_soda_cost * high_tax / 100
  let chips_tax := discounted_chips_cost * high_tax / 100

  let total_candy_bars_with_tax := discounted_candy_bars_cost + candy_bars_tax
  let total_gum_with_tax := total_gum_cost + gum_tax
  let total_soda_with_tax := total_soda_cost + soda_tax
  let total_chips_with_tax := discounted_chips_cost + chips_tax

  let total_cost := total_candy_bars_with_tax + total_gum_with_tax + total_soda_with_tax + total_chips_with_tax

  initial_amount - total_cost

theorem dan_remaining_money : 
  calculate_remaining_money 10000 = 7399 :=
sorry

end NUMINAMATH_GPT_dan_remaining_money_l1558_155881


namespace NUMINAMATH_GPT_geometric_seq_sum_l1558_155801

noncomputable def a_n (n : ℕ) : ℤ :=
  (-3)^(n-1)

theorem geometric_seq_sum :
  let a1 := a_n 1
  let a2 := a_n 2
  let a3 := a_n 3
  let a4 := a_n 4
  let a5 := a_n 5
  a1 + |a2| + a3 + |a4| + a5 = 121 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_sum_l1558_155801


namespace NUMINAMATH_GPT_total_animals_l1558_155870

theorem total_animals (H C2 C1 : ℕ) (humps_eq : 2 * C2 + C1 = 200) (horses_eq : H = C2) :
  H + C2 + C1 = 200 :=
by
  /- Proof steps are not required -/
  sorry

end NUMINAMATH_GPT_total_animals_l1558_155870


namespace NUMINAMATH_GPT_always_odd_l1558_155849

theorem always_odd (p m : ℕ) (hp : p % 2 = 1) : (p^3 + 3*p*m^2 + 2*m) % 2 = 1 := 
by sorry

end NUMINAMATH_GPT_always_odd_l1558_155849


namespace NUMINAMATH_GPT_Debby_spent_on_yoyo_l1558_155894

theorem Debby_spent_on_yoyo 
  (hat_tickets stuffed_animal_tickets total_tickets : ℕ) 
  (h1 : hat_tickets = 2) 
  (h2 : stuffed_animal_tickets = 10) 
  (h3 : total_tickets = 14) 
  : ∃ yoyo_tickets : ℕ, hat_tickets + stuffed_animal_tickets + yoyo_tickets = total_tickets ∧ yoyo_tickets = 2 := 
by 
  sorry

end NUMINAMATH_GPT_Debby_spent_on_yoyo_l1558_155894


namespace NUMINAMATH_GPT_price_of_other_frisbees_proof_l1558_155889

noncomputable def price_of_other_frisbees (P : ℝ) : Prop :=
  ∃ x : ℝ, x + (60 - x) = 60 ∧ x ≥ 0 ∧ P * x + 4 * (60 - x) = 204 ∧ (60 - x) ≥ 24

theorem price_of_other_frisbees_proof : price_of_other_frisbees 3 :=
by
  sorry

end NUMINAMATH_GPT_price_of_other_frisbees_proof_l1558_155889


namespace NUMINAMATH_GPT_john_walks_further_than_nina_l1558_155833

theorem john_walks_further_than_nina :
  let john_distance := 0.7
  let nina_distance := 0.4
  john_distance - nina_distance = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_john_walks_further_than_nina_l1558_155833


namespace NUMINAMATH_GPT_truck_dirt_road_time_l1558_155832

noncomputable def time_on_dirt_road (time_paved : ℝ) (speed_increment : ℝ) (total_distance : ℝ) (dirt_speed : ℝ) : ℝ :=
  let paved_speed := dirt_speed + speed_increment
  let distance_paved := paved_speed * time_paved
  let distance_dirt := total_distance - distance_paved
  distance_dirt / dirt_speed

theorem truck_dirt_road_time :
  time_on_dirt_road 2 20 200 32 = 3 :=
by
  sorry

end NUMINAMATH_GPT_truck_dirt_road_time_l1558_155832


namespace NUMINAMATH_GPT_imaginary_part_of_fraction_l1558_155834

open Complex

theorem imaginary_part_of_fraction :
  ∃ z : ℂ, z = ⟨0, 1⟩ / ⟨1, 1⟩ ∧ z.im = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_fraction_l1558_155834


namespace NUMINAMATH_GPT_ratio_y_to_x_l1558_155800

variable (x y z : ℝ)

-- Conditions
def condition1 (x y z : ℝ) := 0.6 * (x - y) = 0.4 * (x + y) + 0.3 * (x - 3 * z)
def condition2 (y z : ℝ) := ∃ k : ℝ, z = k * y
def condition3 (y z : ℝ) := z = 7 * y
def condition4 (x y : ℝ) := y = 5 * x / 7

theorem ratio_y_to_x (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 y z) (h3 : condition3 y z) (h4 : condition4 x y) : y / x = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_y_to_x_l1558_155800


namespace NUMINAMATH_GPT_inequality_solution_set_system_of_inequalities_solution_set_l1558_155848

theorem inequality_solution_set (x : ℝ) (h : 3 * x - 5 > 5 * x + 3) : x < -4 :=
by sorry

theorem system_of_inequalities_solution_set (x : ℤ) 
  (h₁ : x - 1 ≥ 1 - x) 
  (h₂ : x + 8 > 4 * x - 1) : x = 1 ∨ x = 2 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_system_of_inequalities_solution_set_l1558_155848


namespace NUMINAMATH_GPT_perimeter_of_triangle_LMN_l1558_155888

variable (K L M N : Type)
variables [MetricSpace K]
variables [MetricSpace L]
variables [MetricSpace M]
variables [MetricSpace N]
variables (KL LN MN : ℝ)
variables (perimeter_LMN : ℝ)

-- Given conditions
axiom KL_eq_24 : KL = 24
axiom LN_eq_24 : LN = 24
axiom MN_eq_9  : MN = 9

-- Prove the perimeter is 57
theorem perimeter_of_triangle_LMN : perimeter_LMN = KL + LN + MN → perimeter_LMN = 57 :=
by sorry

end NUMINAMATH_GPT_perimeter_of_triangle_LMN_l1558_155888


namespace NUMINAMATH_GPT_mutually_exclusive_but_not_opposite_l1558_155803

-- Define the cards and the people
inductive Card
| Red
| Black
| Blue
| White

inductive Person
| A
| B
| C
| D

-- Define the events
def eventA_gets_red (distribution : Person → Card) : Prop :=
distribution Person.A = Card.Red

def eventB_gets_red (distribution : Person → Card) : Prop :=
distribution Person.B = Card.Red

-- Define mutually exclusive events
def mutually_exclusive (P Q : Prop) : Prop :=
P → ¬ Q

-- Statement of the problem
theorem mutually_exclusive_but_not_opposite :
  ∀ (distribution : Person → Card), 
    mutually_exclusive (eventA_gets_red distribution) (eventB_gets_red distribution) ∧ 
    ¬ (eventA_gets_red distribution ↔ eventB_gets_red distribution) :=
by sorry

end NUMINAMATH_GPT_mutually_exclusive_but_not_opposite_l1558_155803


namespace NUMINAMATH_GPT_total_dogs_barking_l1558_155837

theorem total_dogs_barking 
  (initial_dogs : ℕ)
  (new_dogs : ℕ)
  (h1 : initial_dogs = 30)
  (h2 : new_dogs = 3 * initial_dogs) :
  initial_dogs + new_dogs = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_dogs_barking_l1558_155837


namespace NUMINAMATH_GPT_average_output_assembly_line_l1558_155815

theorem average_output_assembly_line
  (initial_rate : ℕ) (initial_cogs : ℕ) 
  (increased_rate : ℕ) (increased_cogs : ℕ)
  (h1 : initial_rate = 15)
  (h2 : initial_cogs = 60)
  (h3 : increased_rate = 60)
  (h4 : increased_cogs = 60) :
  (initial_cogs + increased_cogs) / (initial_cogs / initial_rate + increased_cogs / increased_rate) = 24 := 
by sorry

end NUMINAMATH_GPT_average_output_assembly_line_l1558_155815


namespace NUMINAMATH_GPT_limes_left_l1558_155826

-- Define constants
def num_limes_initial : ℕ := 9
def num_limes_given : ℕ := 4

-- Theorem to be proved
theorem limes_left : num_limes_initial - num_limes_given = 5 :=
by
  sorry

end NUMINAMATH_GPT_limes_left_l1558_155826


namespace NUMINAMATH_GPT_chocolate_candy_pieces_l1558_155825

-- Define the initial number of boxes and the boxes given away
def initial_boxes : Nat := 12
def boxes_given : Nat := 7

-- Define the number of remaining boxes
def remaining_boxes := initial_boxes - boxes_given

-- Define the number of pieces per box
def pieces_per_box : Nat := 6

-- Calculate the total pieces Tom still has
def total_pieces := remaining_boxes * pieces_per_box

-- State the theorem
theorem chocolate_candy_pieces : total_pieces = 30 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_chocolate_candy_pieces_l1558_155825


namespace NUMINAMATH_GPT_largest_x_by_equation_l1558_155853

theorem largest_x_by_equation : ∃ x : ℚ, 
  (∀ y : ℚ, 6 * (12 * y^2 + 12 * y + 11) = y * (12 * y - 44) → y ≤ x) 
  ∧ 6 * (12 * x^2 + 12 * x + 11) = x * (12 * x - 44) 
  ∧ x = -1 := 
sorry

end NUMINAMATH_GPT_largest_x_by_equation_l1558_155853


namespace NUMINAMATH_GPT_john_spent_at_candy_store_l1558_155850

noncomputable def johns_allowance : ℝ := 2.40
noncomputable def arcade_spending : ℝ := (3 / 5) * johns_allowance
noncomputable def remaining_after_arcade : ℝ := johns_allowance - arcade_spending
noncomputable def toy_store_spending : ℝ := (1 / 3) * remaining_after_arcade
noncomputable def remaining_after_toy_store : ℝ := remaining_after_arcade - toy_store_spending
noncomputable def candy_store_spending : ℝ := remaining_after_toy_store

theorem john_spent_at_candy_store : candy_store_spending = 0.64 := by sorry

end NUMINAMATH_GPT_john_spent_at_candy_store_l1558_155850


namespace NUMINAMATH_GPT_log_eqn_l1558_155873

theorem log_eqn (a b : ℝ) (h1 : a = (Real.log 400 / Real.log 16))
                          (h2 : b = Real.log 20 / Real.log 2) : a = (1/2) * b :=
sorry

end NUMINAMATH_GPT_log_eqn_l1558_155873


namespace NUMINAMATH_GPT_emma_harry_weight_l1558_155802

theorem emma_harry_weight (e f g h : ℕ) 
  (h1 : e + f = 280) 
  (h2 : f + g = 260) 
  (h3 : g + h = 290) : 
  e + h = 310 := 
sorry

end NUMINAMATH_GPT_emma_harry_weight_l1558_155802


namespace NUMINAMATH_GPT_sum_of_non_common_roots_zero_l1558_155806

theorem sum_of_non_common_roots_zero (m α β γ : ℝ) 
  (h1 : α + β = -(m + 1))
  (h2 : α * β = -3)
  (h3 : α + γ = 4)
  (h4 : α * γ = -m)
  (h_common : α^2 + (m + 1)*α - 3 = 0)
  (h_common2 : α^2 - 4*α - m = 0)
  : β + γ = 0 := sorry

end NUMINAMATH_GPT_sum_of_non_common_roots_zero_l1558_155806


namespace NUMINAMATH_GPT_inequality_equality_condition_l1558_155866

theorem inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2 :=
sorry

end NUMINAMATH_GPT_inequality_equality_condition_l1558_155866


namespace NUMINAMATH_GPT_count_two_digit_primes_with_units_digit_3_l1558_155827

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end NUMINAMATH_GPT_count_two_digit_primes_with_units_digit_3_l1558_155827


namespace NUMINAMATH_GPT_nylon_needed_for_one_dog_collor_l1558_155863

-- Define the conditions as given in the problem
def nylon_for_dog (x : ℝ) : ℝ := x
def nylon_for_cat : ℝ := 10
def total_nylon_used (x : ℝ) : ℝ := 9 * (nylon_for_dog x) + 3 * (nylon_for_cat)

-- Prove the required statement under the given conditions
theorem nylon_needed_for_one_dog_collor : total_nylon_used 18 = 192 :=
by
  -- adding the proof step using sorry as required
  sorry

end NUMINAMATH_GPT_nylon_needed_for_one_dog_collor_l1558_155863


namespace NUMINAMATH_GPT_calc_result_l1558_155895

theorem calc_result : 75 * 1313 - 25 * 1313 = 65750 := 
by 
  sorry

end NUMINAMATH_GPT_calc_result_l1558_155895


namespace NUMINAMATH_GPT_coordinates_of_point_l1558_155896

theorem coordinates_of_point (x y : ℝ) (h : (x, y) = (-2, 3)) : (x, y) = (-2, 3) :=
by
  exact h

end NUMINAMATH_GPT_coordinates_of_point_l1558_155896


namespace NUMINAMATH_GPT_find_m_l1558_155892

def vector (α : Type*) := α × α

def a : vector ℤ := (1, -2)
def b : vector ℤ := (3, 0)

def two_a_plus_b (a b : vector ℤ) : vector ℤ := (2 * a.1 + b.1, 2 * a.2 + b.2)
def m_a_minus_b (m : ℤ) (a b : vector ℤ) : vector ℤ := (m * a.1 - b.1, m * a.2 - b.2)

def parallel (v w : vector ℤ) : Prop := v.1 * w.2 = v.2 * w.1

theorem find_m : parallel (two_a_plus_b a b) (m_a_minus_b (-2) a b) :=
by
  sorry -- proof placeholder

end NUMINAMATH_GPT_find_m_l1558_155892


namespace NUMINAMATH_GPT_dodgeball_cost_l1558_155841

theorem dodgeball_cost (B : ℝ) 
  (hb1 : 1.20 * B = 90) 
  (hb2 : B / 15 = 5) :
  ∃ (cost_per_dodgeball : ℝ), cost_per_dodgeball = 5 := by
sorry

end NUMINAMATH_GPT_dodgeball_cost_l1558_155841


namespace NUMINAMATH_GPT_Amanda_money_left_l1558_155823

theorem Amanda_money_left (initial_amount cost_cassette tape_count cost_headphone : ℕ) 
  (h1 : initial_amount = 50) 
  (h2 : cost_cassette = 9) 
  (h3 : tape_count = 2) 
  (h4 : cost_headphone = 25) :
  initial_amount - (tape_count * cost_cassette + cost_headphone) = 7 :=
by
  sorry

end NUMINAMATH_GPT_Amanda_money_left_l1558_155823


namespace NUMINAMATH_GPT_abc_cubic_sum_identity_l1558_155846

theorem abc_cubic_sum_identity (a b c : ℂ) 
  (M : Matrix (Fin 3) (Fin 3) ℂ)
  (h1 : M = fun i j => if i = 0 then (if j = 0 then a else if j = 1 then b else c)
                      else if i = 1 then (if j = 0 then b else if j = 1 then c else a)
                      else (if j = 0 then c else if j = 1 then a else b))
  (h2 : M ^ 3 = 1)
  (h3 : a * b * c = -1) :
  a^3 + b^3 + c^3 = 4 := sorry

end NUMINAMATH_GPT_abc_cubic_sum_identity_l1558_155846


namespace NUMINAMATH_GPT_problem_1_l1558_155857

theorem problem_1 : -9 + 5 - (-12) + (-3) = 5 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_problem_1_l1558_155857


namespace NUMINAMATH_GPT_common_difference_arithmetic_seq_l1558_155810

theorem common_difference_arithmetic_seq (a1 d : ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n * a1 + n * (n - 1) / 2 * d) : 
  (S 5 / 5 - S 2 / 2 = 3) → d = 2 :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_seq_l1558_155810


namespace NUMINAMATH_GPT_last_passenger_seats_probability_l1558_155867

theorem last_passenger_seats_probability (n : ℕ) (hn : n > 0) :
  ∀ (P : ℝ), P = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_last_passenger_seats_probability_l1558_155867


namespace NUMINAMATH_GPT_numOxygenAtoms_l1558_155836

-- Define the conditions as hypothesis
def numCarbonAtoms : ℕ := 4
def numHydrogenAtoms : ℕ := 8
def molecularWeight : ℕ := 88
def atomicWeightCarbon : ℕ := 12
def atomicWeightHydrogen : ℕ := 1
def atomicWeightOxygen : ℕ := 16

-- The statement to be proved
theorem numOxygenAtoms :
  let totalWeightC := numCarbonAtoms * atomicWeightCarbon
  let totalWeightH := numHydrogenAtoms * atomicWeightHydrogen
  let totalWeightCH := totalWeightC + totalWeightH
  let weightOxygenAtoms := molecularWeight - totalWeightCH
  let numOxygenAtoms := weightOxygenAtoms / atomicWeightOxygen
  numOxygenAtoms = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_numOxygenAtoms_l1558_155836


namespace NUMINAMATH_GPT_regular_seven_gon_l1558_155862

theorem regular_seven_gon 
    (A : Fin 7 → ℝ × ℝ)
    (cong_diagonals_1 : ∀ (i : Fin 7), dist (A i) (A ((i + 2) % 7)) = dist (A 0) (A 2))
    (cong_diagonals_2 : ∀ (i : Fin 7), dist (A i) (A ((i + 3) % 7)) = dist (A 0) (A 3))
    : ∀ (i j : Fin 7), dist (A i) (A ((i + 1) % 7)) = dist (A j) (A ((j + 1) % 7)) :=
by sorry

end NUMINAMATH_GPT_regular_seven_gon_l1558_155862


namespace NUMINAMATH_GPT_password_probability_l1558_155880

def is_prime_single_digit : Fin 10 → Prop
| 2 | 3 | 5 | 7 => true
| _ => false

def is_vowel : Char → Prop
| 'A' | 'E' | 'I' | 'O' | 'U' => true
| _ => false

def is_positive_even_single_digit : Fin 9 → Prop
| 2 | 4 | 6 | 8 => true
| _ => false

def prime_probability : ℚ := 4 / 10
def vowel_probability : ℚ := 5 / 26
def even_pos_digit_probability : ℚ := 4 / 9

theorem password_probability :
  prime_probability * vowel_probability * even_pos_digit_probability = 8 / 117 := by
  sorry

end NUMINAMATH_GPT_password_probability_l1558_155880


namespace NUMINAMATH_GPT_remaining_pieces_l1558_155805

theorem remaining_pieces (initial_pieces : ℕ) (arianna_lost : ℕ) (samantha_lost : ℕ) (diego_lost : ℕ) (lucas_lost : ℕ) :
  initial_pieces = 128 → arianna_lost = 3 → samantha_lost = 9 → diego_lost = 5 → lucas_lost = 7 →
  initial_pieces - (arianna_lost + samantha_lost + diego_lost + lucas_lost) = 104 := by
  sorry

end NUMINAMATH_GPT_remaining_pieces_l1558_155805
