import Mathlib

namespace yoongi_has_fewer_apples_l6_6978

-- Define the number of apples Jungkook originally has and receives more.
def jungkook_original_apples := 6
def jungkook_received_apples := 3

-- Calculate the total number of apples Jungkook has.
def jungkook_total_apples := jungkook_original_apples + jungkook_received_apples

-- Define the number of apples Yoongi has.
def yoongi_apples := 4

-- State that Yoongi has fewer apples than Jungkook.
theorem yoongi_has_fewer_apples : yoongi_apples < jungkook_total_apples := by
  sorry

end yoongi_has_fewer_apples_l6_6978


namespace cube_side_length_eq_three_l6_6482

theorem cube_side_length_eq_three (n : ℕ) (h1 : 6 * n^2 = 6 * n^3 / 3) : n = 3 := by
  -- The proof is omitted as per instructions, we use sorry to skip it.
  sorry

end cube_side_length_eq_three_l6_6482


namespace g_frac_8_12_l6_6687

def g (q : ℚ) : ℤ := sorry  -- Defined as integer-valued function

axiom g_mul (a b : ℚ) (ha : 0 < a) (hb : 0 < b) : g (a * b) = g a + g b

axiom g_prime_p (p : ℚ) (hp : nat.prime p.nat_abs) : g p = p.nat_abs

axiom g_coprime (a b : ℚ) (ha : 0 < a) (hb : 0 < b) (hcop : nat.coprime a.nat_abs b.nat_abs) :
  g (a + b) = g a + g b - 1

theorem g_frac_8_12 : g (8 / 12) < 0 :=
sorry

end g_frac_8_12_l6_6687


namespace Carlson_max_jars_l6_6225

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l6_6225


namespace minimum_value_l6_6042

noncomputable def f : ℝ → ℝ
| x => if h : 0 < x ∧ x ≤ 1 then x^2 - x else
         if h : 1 < x ∧ x ≤ 2 then -2 * (x - 1)^2 + 6 * (x - 1) - 5
         else 0 -- extend as appropriate outside given ranges

noncomputable def g (x : ℝ) : ℝ := x - 1

theorem minimum_value (x_1 x_2 : ℝ) (h1 : 1 < x_1 ∧ x_1 ≤ 2) : 
  (x_1 - x_2)^2 + (f x_1 - g x_2)^2 = 49 / 128 :=
sorry

end minimum_value_l6_6042


namespace only_one_positive_integer_n_l6_6807

theorem only_one_positive_integer_n (k : ℕ) (hk : 0 < k) (m : ℕ) (hm : k + 2 ≤ m) :
  ∃! (n : ℕ), 0 < n ∧ n^m ∣ 5^(n^k) + 1 :=
sorry

end only_one_positive_integer_n_l6_6807


namespace period_of_time_l6_6109

-- We define the annual expense and total amount spent as constants
def annual_expense : ℝ := 2
def total_amount_spent : ℝ := 20

-- Theorem to prove the period of time (in years)
theorem period_of_time : total_amount_spent / annual_expense = 10 :=
by 
  -- Placeholder proof
  sorry

end period_of_time_l6_6109


namespace cosine_inequality_l6_6941

theorem cosine_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 0 < x^2 + y^2 ∧ x^2 + y^2 ≤ π) :
  1 + Real.cos (x * y) ≥ Real.cos x + Real.cos y :=
sorry

end cosine_inequality_l6_6941


namespace avg_growth_rate_proof_l6_6490

noncomputable def avg_growth_rate_correct_eqn (x : ℝ) : Prop :=
  40 * (1 + x)^2 = 48.4

theorem avg_growth_rate_proof (x : ℝ) 
  (h1 : 40 = avg_working_hours_first_week)
  (h2 : 48.4 = avg_working_hours_third_week) :
  avg_growth_rate_correct_eqn x :=
by 
  sorry

/- Defining the known conditions -/
def avg_working_hours_first_week : ℝ := 40
def avg_working_hours_third_week : ℝ := 48.4

end avg_growth_rate_proof_l6_6490


namespace fraction_replaced_l6_6611

theorem fraction_replaced (x : ℝ) (h₁ : 0.15 * (1 - x) + 0.19000000000000007 * x = 0.16) : x = 0.25 :=
by
  sorry

end fraction_replaced_l6_6611


namespace problem_proof_l6_6947

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + (1 / Real.sqrt (2 - x))
noncomputable def g (x : ℝ) : ℝ := x^2 + 1

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {y | y ≥ 1}
def CU_B : Set ℝ := {y | y < 1}
def U : Set ℝ := Set.univ

theorem problem_proof :
  (∀ x, x ∈ A ↔ -1 ≤ x ∧ x < 2) ∧
  (∀ y, y ∈ B ↔ y ≥ 1) ∧
  (A ∩ CU_B = {x | -1 ≤ x ∧ x < 1}) :=
by
  sorry

end problem_proof_l6_6947


namespace find_d_l6_6716

theorem find_d (d : ℝ) : (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  { sorry }

end find_d_l6_6716


namespace increasing_on_interval_l6_6771

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x
noncomputable def f2 (x : ℝ) : ℝ := x * Real.exp 2
noncomputable def f3 (x : ℝ) : ℝ := x^3 - x
noncomputable def f4 (x : ℝ) : ℝ := Real.log x - x

theorem increasing_on_interval (x : ℝ) (h : 0 < x) : 
  f2 (x) = x * Real.exp 2 ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f1 x < f1 y) ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f3 x < f3 y) ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f4 x < f4 y) :=
by sorry

end increasing_on_interval_l6_6771


namespace curves_intersection_probability_l6_6336

-- Definitions based on conditions
def curve1 (p q x : ℝ) := 2 * x^2 + p * x + q
def curve2 (r s x : ℝ) := -x^2 + r * x + s

def intersection_probability : ℝ :=
  let choices_p : Finset ℤ := {1, 2, 3}
  let choices_q : Finset ℤ := {0, 1}
  let combs := (choices_p ×ˢ choices_q).product (choices_p ×ˢ choices_q)
  let valid_combs := combs.filter (λ ⟨⟨p, q⟩, ⟨r, s⟩⟩, 
    let a := 3
    let b := p - r
    let c := q - s
    let Δ := b^2 - 4 * a * c
    Δ ≥ 0)
  (valid_combs.card.to_real / combs.card.to_real)

-- Lean statement
theorem curves_intersection_probability : intersection_probability = 13 / 36 := by
  sorry

end curves_intersection_probability_l6_6336


namespace spring_length_function_l6_6486

noncomputable def spring_length (x : ℝ) : ℝ :=
  12 + 3 * x

theorem spring_length_function :
  ∀ (x : ℝ), spring_length x = 12 + 3 * x :=
by
  intro x
  rfl

end spring_length_function_l6_6486


namespace find_counterfeit_10_l6_6753

theorem find_counterfeit_10 (coins : Fin 10 → ℕ) (h_counterfeit : ∃ k, ∀ i, i ≠ k → coins i < coins k) : 
  ∃ w : ℕ → ℕ → Prop, (∀ g1 g2, g1 ≠ g2 → w g1 g2 ∨ w g2 g1) → 
  ∃ k, ∀ i, i ≠ k → coins i < coins k :=
sorry

end find_counterfeit_10_l6_6753


namespace anne_cleaning_time_l6_6778

-- Define the conditions in the problem
variable (B A : ℝ) -- B and A are Bruce's and Anne's cleaning rates respectively

-- Conditions based on the given problem
axiom cond1 : (B + A) * 4 = 1 -- Together they can clean the house in 4 hours
axiom cond2 : (B + 2 * A) * 3 = 1 -- With Anne's speed doubled, they clean in 3 hours

-- The theorem statement asserting Anne’s time to clean the house alone is 12 hours
theorem anne_cleaning_time : (1 / A) = 12 :=
by 
  -- start by analyzing the first condition
  have h1 : 4 * B + 4 * A = 1, from cond1,
  -- next, process the second condition
  have h2 : 3 * B + 6 * A = 1, from cond2,
  -- combine and solve these conditions
  sorry

end anne_cleaning_time_l6_6778


namespace sufficient_condition_for_proposition_l6_6466

theorem sufficient_condition_for_proposition (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 5) :=
by 
  sorry

end sufficient_condition_for_proposition_l6_6466


namespace votes_cast_is_750_l6_6503

-- Define the conditions as Lean statements
def initial_score : ℤ := 0
def score_increase (likes : ℕ) : ℤ := likes
def score_decrease (dislikes : ℕ) : ℤ := -dislikes
def observed_score : ℤ := 150
def percent_likes : ℚ := 0.60

-- Express the proof
theorem votes_cast_is_750 (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ) 
  (h1 : total_votes = likes + dislikes) 
  (h2 : percent_likes * total_votes = likes) 
  (h3 : dislikes = (1 - percent_likes) * total_votes)
  (h4 : observed_score = score_increase likes + score_decrease dislikes) :
  total_votes = 750 := 
sorry

end votes_cast_is_750_l6_6503


namespace ordering_abc_l6_6076

noncomputable def a : ℝ := Real.sqrt 1.01
noncomputable def b : ℝ := Real.exp 0.01 / 1.01
noncomputable def c : ℝ := Real.log (1.01 * Real.exp 1)

theorem ordering_abc : b < a ∧ a < c := by
  -- Proof of the theorem goes here
  sorry

end ordering_abc_l6_6076


namespace total_amount_paid_l6_6492

theorem total_amount_paid :
  let chapati_cost := 6
  let rice_cost := 45
  let mixed_vegetable_cost := 70
  let ice_cream_cost := 40
  let chapati_quantity := 16
  let rice_quantity := 5
  let mixed_vegetable_quantity := 7
  let ice_cream_quantity := 6
  let total_cost := chapati_quantity * chapati_cost +
                    rice_quantity * rice_cost +
                    mixed_vegetable_quantity * mixed_vegetable_cost +
                    ice_cream_quantity * ice_cream_cost
  total_cost = 1051 := by
  sorry

end total_amount_paid_l6_6492


namespace min_value_of_fraction_l6_6444

theorem min_value_of_fraction (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (h_sum : a + 3 * b = 2) : 
  ∃ m, (∀ (a b : ℝ), a > 0 → b > 0 → a + 3 * b = 2 → 1 / a + 3 / b ≥ m) ∧ m = 8 := 
by
  sorry

end min_value_of_fraction_l6_6444


namespace investment_amount_l6_6775

theorem investment_amount (x y : ℝ) (hx : x ≤ 11000) (hy : 0.07 * x + 0.12 * y ≥ 2450) : x + y = 25000 := 
sorry

end investment_amount_l6_6775


namespace denomination_of_bill_l6_6698

def cost_berries : ℝ := 7.19
def cost_peaches : ℝ := 6.83
def change_received : ℝ := 5.98

theorem denomination_of_bill :
  (cost_berries + cost_peaches) + change_received = 20.0 := 
by 
  sorry

end denomination_of_bill_l6_6698


namespace largest_y_coordinate_l6_6517

theorem largest_y_coordinate (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 :=
sorry

end largest_y_coordinate_l6_6517


namespace original_fraction_is_one_third_l6_6712

theorem original_fraction_is_one_third
  (a b : ℕ) (h₁ : Nat.gcd a b = 1)
  (h₂ : (a + 2) * b = 3 * a * b^2) : 
  a = 1 ∧ b = 3 := by
  sorry

end original_fraction_is_one_third_l6_6712


namespace ordered_pair_count_l6_6400

theorem ordered_pair_count :
  (∃ (bc : ℕ × ℕ), bc.1 > 0 ∧ bc.2 > 0 ∧ bc.1 ^ 4 - 4 * bc.2 ≤ 0 ∧ bc.2 ^ 4 - 4 * bc.1 ≤ 0) ∧
  ∀ (bc1 bc2 : ℕ × ℕ),
    bc1 ≠ bc2 →
    bc1.1 > 0 ∧ bc1.2 > 0 ∧ bc1.1 ^ 4 - 4 * bc1.2 ≤ 0 ∧ bc1.2 ^ 4 - 4 * bc1.1 ≤ 0 →
    bc2.1 > 0 ∧ bc2.2 > 0 ∧ bc2.1 ^ 4 - 4 * bc2.2 ≤ 0 ∧ bc2.2 ^ 4 - 4 * bc2.1 ≤ 0 →
    false
:=
sorry

end ordered_pair_count_l6_6400


namespace f_properties_l6_6409

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ * f x₂) :=
by 
  sorry

end f_properties_l6_6409


namespace hyperbola_eccentricity_l6_6041

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ), (b = 5) → (c = 3) → (c^2 = a^2 + b) → (a > 0) →
  (a + c = 3) → (e = c / a) → (e = 3 / 2) :=
by
  intros a b c hb hc hc2 ha hac he
  sorry

end hyperbola_eccentricity_l6_6041


namespace avg_first_six_results_l6_6709

theorem avg_first_six_results (A : ℝ) :
  (∀ (results : Fin 12 → ℝ), 
    (results 0 + results 1 + results 2 + results 3 + results 4 + results 5 + 
     results 6 + results 7 + results 8 + results 9 + results 10 + results 11) / 11 = 60 → 
    (results 0 + results 1 + results 2 + results 3 + results 4 + results 5) / 6 = A → 
    (results 5 + results 6 + results 7 + results 8 + results 9 + results 10) / 6 = 63 → 
    results 5 = 66) → 
  A = 58 :=
by
  sorry

end avg_first_six_results_l6_6709


namespace original_fraction_is_one_third_l6_6713

theorem original_fraction_is_one_third
  (a b : ℕ) (h₁ : Nat.gcd a b = 1)
  (h₂ : (a + 2) * b = 3 * a * b^2) : 
  a = 1 ∧ b = 3 := by
  sorry

end original_fraction_is_one_third_l6_6713


namespace lcm_of_10_and_21_l6_6254

theorem lcm_of_10_and_21 : Nat.lcm 10 21 = 210 :=
by
  sorry

end lcm_of_10_and_21_l6_6254


namespace ball_in_78th_position_is_green_l6_6212

-- Definition of colors in the sequence
inductive Color
| red
| yellow
| green
| blue
| violet

open Color

-- Function to compute the color of a ball at a given position within a cycle
def ball_color (n : Nat) : Color :=
  match n % 5 with
  | 0 => red    -- 78 % 5 == 3, hence 3 + 1 == 4 ==> Using 0 for red to 4 for violet
  | 1 => yellow
  | 2 => green
  | 3 => blue
  | 4 => violet
  | _ => red  -- default case, should not be reached

-- Theorem stating the desired proof problem
theorem ball_in_78th_position_is_green : ball_color 78 = green :=
by
  sorry

end ball_in_78th_position_is_green_l6_6212


namespace determine_digits_l6_6638

theorem determine_digits (h t u : ℕ) (hu: h > u) (h_subtr: t = h - 5) (unit_result: u = 3) : (h = 9 ∧ t = 4 ∧ u = 3) := by
  sorry

end determine_digits_l6_6638


namespace Zack_traveled_18_countries_l6_6350

variables (countries_Alex countries_George countries_Joseph countries_Patrick countries_Zack : ℕ)
variables (h1 : countries_Alex = 24)
variables (h2 : countries_George = countries_Alex / 4)
variables (h3 : countries_Joseph = countries_George / 2)
variables (h4 : countries_Patrick = 3 * countries_Joseph)
variables (h5 : countries_Zack = 2 * countries_Patrick)

theorem Zack_traveled_18_countries :
  countries_Zack = 18 :=
by sorry

end Zack_traveled_18_countries_l6_6350


namespace JerryAge_l6_6083

-- Given definitions
def MickeysAge : ℕ := 20
def AgeRelationship (M J : ℕ) : Prop := M = 2 * J + 10

-- Proof statement
theorem JerryAge : ∃ J : ℕ, AgeRelationship MickeysAge J ∧ J = 5 :=
by
  sorry

end JerryAge_l6_6083


namespace rational_solution_exists_l6_6069

theorem rational_solution_exists :
  ∃ (a b : ℚ), (a + b) / a + a / (a + b) = b :=
by
  sorry

end rational_solution_exists_l6_6069


namespace new_salary_l6_6007

theorem new_salary (increase : ℝ) (percent_increase : ℝ) (S_new : ℝ) :
  increase = 25000 → percent_increase = 38.46153846153846 → S_new = 90000 :=
by
  sorry

end new_salary_l6_6007


namespace anne_cleaning_time_l6_6782

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l6_6782


namespace computation_distinct_collections_l6_6085

open Finset

def vowels := {'A', 'E', 'I', 'O', 'U'}  -- The set of all vowels in general.
def letters := "COMPUTATION".toList.toFinset  -- The letters in the word COMPUTATION (treated as a Finset of characters).

-- The specific numbers of each type of letter available:
def counts :=
  ('C', 1) :: ('O', 2) :: ('M', 1) :: ('P', 1) :: ('U', 1) :: ('T', 2) 
  :: ('A', 1) :: ('I', 1) :: ('N', 1) :: []

def distinct_collections_count : ℕ :=
  let vowels_count := ('A', 1) :: ('O', 2) :: ('U', 1) :: ('I', 1) :: []
  let consonants_count := ('C', 1) :: ('M', 1) :: ('P', 1) :: ('T', 2) :: ('N', 1) :: []
  let total_vowels_choices := (combinatorics.binomial 5 3).to_nat * 
    (let choices := (0 :: 1 :: []) in choices.sum (λ t_count, combinatorics.binomial 4 (4 - t_count).to_nat))
  let total_all_t_choices := (combinatorics.binomial 4 2).to_nat * 6
  in total_vowels_choices + total_all_t_choices

theorem computation_distinct_collections :
  distinct_collections_count = 110 :=
begin
  -- Omitted proof
  sorry
end

end computation_distinct_collections_l6_6085


namespace line_through_point_with_equal_intercepts_l6_6025

-- Definition of the conditions
def point := (1 : ℝ, 2 : ℝ)
def eq_intercepts (line : ℝ → ℝ) := ∃ a b : ℝ, a = b ∧ (∀ x, line x = b - x * (b/a))

-- The proof statement
theorem line_through_point_with_equal_intercepts (line : ℝ → ℝ) : 
  (line 1 = 2 ∧ eq_intercepts line) → (line = (λ x, 2 * x) ∨ line = (λ x, 3 - x)) :=
by
  sorry

end line_through_point_with_equal_intercepts_l6_6025


namespace sum_of_a_for_unique_solution_l6_6937

theorem sum_of_a_for_unique_solution (a : ℝ) (x : ℝ) :
  (∃ (a : ℝ), 3 * x ^ 2 + a * x + 6 * x + 7 = 0 ∧ (a + 6) ^ 2 - 4 * 3 * 7 = 0) →
  (-6 + 2 * Real.sqrt 21 + -6 - 2 * Real.sqrt 21 = -12) :=
by
  sorry

end sum_of_a_for_unique_solution_l6_6937


namespace probability_at_least_one_passes_l6_6738

open Probability

theorem probability_at_least_one_passes (A B C : Event) (P : Prob) :
  P(A) = 1/3 ∧ P(B) = 1/3 ∧ P(C) = 1/3 ∧ indep_indep A B ∧ indep_indep B C ∧ indep_indep A C ->
  P(A ∪ B ∪ C) = 19/27 := by
  intros h
  sorry

end probability_at_least_one_passes_l6_6738


namespace hexagon_area_l6_6565

theorem hexagon_area (A C : ℝ × ℝ) 
  (hA : A = (0, 0)) 
  (hC : C = (2 * Real.sqrt 3, 2)) : 
  6 * Real.sqrt 3 = 6 * Real.sqrt 3 := 
by sorry

end hexagon_area_l6_6565


namespace surfers_ratio_l6_6311

theorem surfers_ratio (S1 : ℕ) (S3 : ℕ) : S1 = 1500 → 
  (∀ S2 : ℕ, S2 = S1 + 600 → (1400 * 3 = S1 + S2 + S3) → 
  S3 = 600) → (S3 / S1 = 2 / 5) :=
sorry

end surfers_ratio_l6_6311


namespace correct_system_of_equations_l6_6116

-- Definitions based on the conditions
def rope_exceeds (x y : ℝ) : Prop := x - y = 4.5
def rope_half_falls_short (x y : ℝ) : Prop := (1/2) * x + 1 = y

-- Proof statement
theorem correct_system_of_equations (x y : ℝ) :
  rope_exceeds x y → rope_half_falls_short x y → 
  (x - y = 4.5 ∧ (1/2 * x + 1 = y)) := 
by 
  sorry

end correct_system_of_equations_l6_6116


namespace two_a_plus_two_d_eq_zero_l6_6445

theorem two_a_plus_two_d_eq_zero
  (a b c d : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h₃ : d ≠ 0)
  (h₄ : ∀ x : ℝ, (2 * a * ((2 * a * x + b) / (3 * c * x + 2 * d)) + b)
                 / (3 * c * ((2 * a * x + b) / (3 * c * x + 2 * d)) + 2 * d) = x) :
  2 * a + 2 * d = 0 :=
by sorry

end two_a_plus_two_d_eq_zero_l6_6445


namespace alex_and_zhu_probability_l6_6589

theorem alex_and_zhu_probability :
  let num_students := 100
  let num_selected := 60
  let num_sections := 3
  let section_size := 20
  let P_alex_selected := 3 / 5
  let P_zhu_selected_given_alex_selected := 59 / 99
  let P_same_section_given_both_selected := 19 / 59
  (P_alex_selected * P_zhu_selected_given_alex_selected * P_same_section_given_both_selected) = 19 / 165 := 
by {
  sorry
}

end alex_and_zhu_probability_l6_6589


namespace parallelogram_sides_l6_6468

theorem parallelogram_sides (x y : ℝ) (h1 : 12 * y - 2 = 10) (h2 : 5 * x + 15 = 20) : x + y = 2 :=
by
  sorry

end parallelogram_sides_l6_6468


namespace value_of_f_at_pi_over_12_l6_6474

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 12)

theorem value_of_f_at_pi_over_12 : f (Real.pi / 12) = Real.sqrt 2 / 2 :=
by
  sorry

end value_of_f_at_pi_over_12_l6_6474


namespace cakes_given_away_l6_6459

theorem cakes_given_away 
  (cakes_baked : ℕ) 
  (candles_per_cake : ℕ) 
  (total_candles : ℕ) 
  (cakes_given : ℕ) 
  (cakes_left : ℕ) 
  (h1 : cakes_baked = 8) 
  (h2 : candles_per_cake = 6) 
  (h3 : total_candles = 36) 
  (h4 : total_candles = candles_per_cake * cakes_left) 
  (h5 : cakes_given = cakes_baked - cakes_left) 
  : cakes_given = 2 :=
sorry

end cakes_given_away_l6_6459


namespace shenille_scores_points_l6_6539

theorem shenille_scores_points :
  ∀ (x y : ℕ), (x + y = 45) → (x = 2 * y) → 
  (25/100 * x + 40/100 * y) * 3 + (40/100 * y) * 2 = 33 :=
by 
  intros x y h1 h2
  sorry

end shenille_scores_points_l6_6539


namespace isosceles_triangle_height_l6_6913

theorem isosceles_triangle_height (l w h : ℝ) 
  (h1 : l * w = (1 / 2) * w * h) : h = 2 * l :=
by
  sorry

end isosceles_triangle_height_l6_6913


namespace regular_polygon_sides_l6_6187

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6187


namespace total_copper_mined_l6_6218

theorem total_copper_mined :
  let daily_production_A := 4500
  let daily_production_B := 6000
  let daily_production_C := 5000
  let daily_production_D := 3500
  let copper_percentage_A := 0.055
  let copper_percentage_B := 0.071
  let copper_percentage_C := 0.147
  let copper_percentage_D := 0.092
  (daily_production_A * copper_percentage_A +
   daily_production_B * copper_percentage_B +
   daily_production_C * copper_percentage_C +
   daily_production_D * copper_percentage_D) = 1730.5 :=
by
  sorry

end total_copper_mined_l6_6218


namespace rectangle_area_change_l6_6575

theorem rectangle_area_change (x : ℝ) :
  let L := 1 -- arbitrary non-zero value for length
  let W := 1 -- arbitrary non-zero value for width
  (1 + x / 100) * (1 - x / 100) = 1.01 -> x = 10 := 
by
  sorry

end rectangle_area_change_l6_6575


namespace digit_distribution_l6_6289

theorem digit_distribution (n : ℕ) (d1 d2 d5 do : ℚ) (h : d1 = 1 / 2 ∧ d2 = 1 / 5 ∧ d5 = 1 / 5 ∧ do = 1 / 10) :
  d1 + d2 + d5 + do = 1 → n = 10 :=
begin
  sorry
end

end digit_distribution_l6_6289


namespace no_three_positive_reals_l6_6256

noncomputable def S (a : ℝ) : Set ℕ := { n | ∃ (k : ℕ), n = ⌊(k : ℝ) * a⌋ }

theorem no_three_positive_reals (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (S a ∩ S b = ∅) ∧ (S b ∩ S c = ∅) ∧ (S c ∩ S a = ∅) ∧ (S a ∪ S b ∪ S c = Set.univ) → false :=
sorry

end no_three_positive_reals_l6_6256


namespace geometric_arithmetic_sequence_common_ratio_l6_6516

theorem geometric_arithmetic_sequence_common_ratio (a_1 a_2 a_3 q : ℝ) 
  (h1 : a_2 = a_1 * q) 
  (h2 : a_3 = a_1 * q^2)
  (h3 : 2 * a_3 = a_1 + a_2) : (q = 1) ∨ (q = -1) :=
by
  sorry

end geometric_arithmetic_sequence_common_ratio_l6_6516


namespace unbroken_seashells_l6_6739

theorem unbroken_seashells (total_seashells : ℕ) (broken_seashells : ℕ) (h1 : total_seashells = 23) (h2 : broken_seashells = 11) : total_seashells - broken_seashells = 12 := by
  sorry

end unbroken_seashells_l6_6739


namespace second_trial_temperatures_l6_6625

-- Definitions based on the conditions
def range_start : ℝ := 60
def range_end : ℝ := 70
def golden_ratio : ℝ := 0.618

-- Calculations for trial temperatures
def lower_trial_temp : ℝ := range_start + (range_end - range_start) * golden_ratio
def upper_trial_temp : ℝ := range_end - (range_end - range_start) * golden_ratio

-- Lean 4 statement to prove the trial temperatures
theorem second_trial_temperatures :
  lower_trial_temp = 66.18 ∧ upper_trial_temp = 63.82 :=
by
  sorry

end second_trial_temperatures_l6_6625


namespace count_valid_subsets_l6_6425

open Finset Nat

def set := {12, 18, 25, 33, 47, 52}

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def valid_subsets : Finset (Finset ℕ) :=
  (powerset set).filter (λ s, s.card = 3 ∧ is_divisible_by_3 (∑ x in s, x))

theorem count_valid_subsets : valid_subsets.card = 7 := by
  sorry

end count_valid_subsets_l6_6425


namespace orange_weight_l6_6126

variable (A O : ℕ)

theorem orange_weight (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 :=
  sorry

end orange_weight_l6_6126


namespace ball_distribution_l6_6954

theorem ball_distribution :
  (∃ f : Fin 6 → Fin 2, (∀ b : Fin 2, (∃ n : ℕ, (n ≤ 4 ∧ (f ⁻¹' {b}).card = n)) 
  ∧  (Finset.filter (λ i, f i = 0) (Finset.univ: Finset (Fin 6))).card ≤ 4 
  ∧  (Finset.filter (λ i, f i = 1) (Finset.univ: Finset (Fin 6))).card ≤ 4 
  ∧ Finset.card (Finset.range 2) = 2)) ∧ 
  (∑ i in (Finset.filter (λ i, i = 1) (Finset.Powerset (Finset.fin_range 6))), 
  2 * binomial 6 (i.card)) / 2 + 
  (∑ i in (Finset.filter (λ i, i = 0) (Finset.Powerset (Finset.fin_range 6))), 
  binomial 6 (i.card)) = 25 :=
begin
  sorry
end

end ball_distribution_l6_6954


namespace find_x_l6_6528

noncomputable def positive_real (a : ℝ) := 0 < a

theorem find_x (x y : ℝ) (h1 : positive_real x) (h2 : positive_real y)
  (h3 : 6 * x^3 + 12 * x^2 * y = 2 * x^4 + 3 * x^3 * y)
  (h4 : x + y = 3) : x = 2 :=
by
  sorry

end find_x_l6_6528


namespace min_sum_arth_seq_l6_6813

theorem min_sum_arth_seq (a : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1))
  (h2 : a 1 = -3)
  (h3 : 11 * a 5 = 5 * a 8) : n = 4 := by
  sorry

end min_sum_arth_seq_l6_6813


namespace regular_polygon_sides_l6_6145

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6145


namespace find_a_value_l6_6803

theorem find_a_value (a x y : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : a * x - 3 * y = 3) : a = 6 :=
by
  rw [h1, h2] at h3 -- Substitute x and y values into the equation
  sorry -- The proof is omitted as per instructions.

end find_a_value_l6_6803


namespace cheese_pizzas_l6_6849

theorem cheese_pizzas (p b c total : ℕ) (h1 : p = 2) (h2 : b = 6) (h3 : total = 14) (ht : p + b + c = total) : c = 6 := 
by
  sorry

end cheese_pizzas_l6_6849


namespace equip_20posts_with_5new_weapons_l6_6794

/-- 
Theorem: In a line of 20 defense posts, the number of ways to equip 5 different new weapons 
such that:
1. The first and last posts are not equipped with new weapons.
2. Each set of 5 consecutive posts has at least one post equipped with a new weapon.
3. No two adjacent posts are equipped with new weapons.
is 69600. 
-/
theorem equip_20posts_with_5new_weapons : ∃ ways : ℕ, ways = 69600 :=
by
  sorry

end equip_20posts_with_5new_weapons_l6_6794


namespace part_I_part_II_l6_6418

noncomputable def A : Set ℝ := {x | 2*x^2 - 5*x - 3 <= 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - (2*a + 1)) * (x - (a - 1)) < 0}

theorem part_I :
  (A ∪ B 0 = {x : ℝ | -1 < x ∧ x ≤ 3}) :=
by sorry

theorem part_II (a : ℝ) :
  (A ∩ B a = ∅) →
  (a ≤ -3/4 ∨ a ≥ 4) ∧ a ≠ -2 :=
by sorry


end part_I_part_II_l6_6418


namespace carlson_max_jars_l6_6224

theorem carlson_max_jars (n a k : ℕ) (h1 : a = 5 * k)
  (h2 : n = 9 * k)
  (total_weight_carlson : 13 * n)
  (total_weight_baby : n)
  (h3 : 13 * n - a = 8 * (n + a)) :
  ∃ (j : ℕ), j ≤ 23 :=
by sorry

end carlson_max_jars_l6_6224


namespace sum_of_x_y_l6_6864

theorem sum_of_x_y (x y : ℕ) (x_square_condition : ∃ x, ∃ n : ℕ, 450 * x = n^2)
                   (y_cube_condition : ∃ y, ∃ m : ℕ, 450 * y = m^3) :
                   x = 2 ∧ y = 4 → x + y = 6 := 
sorry

end sum_of_x_y_l6_6864


namespace anne_cleaning_time_l6_6780

-- Define the conditions in the problem
variable (B A : ℝ) -- B and A are Bruce's and Anne's cleaning rates respectively

-- Conditions based on the given problem
axiom cond1 : (B + A) * 4 = 1 -- Together they can clean the house in 4 hours
axiom cond2 : (B + 2 * A) * 3 = 1 -- With Anne's speed doubled, they clean in 3 hours

-- The theorem statement asserting Anne’s time to clean the house alone is 12 hours
theorem anne_cleaning_time : (1 / A) = 12 :=
by 
  -- start by analyzing the first condition
  have h1 : 4 * B + 4 * A = 1, from cond1,
  -- next, process the second condition
  have h2 : 3 * B + 6 * A = 1, from cond2,
  -- combine and solve these conditions
  sorry

end anne_cleaning_time_l6_6780


namespace shared_friends_l6_6449

theorem shared_friends (crackers total_friends : ℕ) (each_friend_crackers : ℕ) 
  (h1 : crackers = 22) 
  (h2 : each_friend_crackers = 2)
  (h3 : crackers = each_friend_crackers * total_friends) 
  : total_friends = 11 := by 
  sorry

end shared_friends_l6_6449


namespace tangent_normal_at_t1_l6_6253

noncomputable def curve_param_x (t: ℝ) : ℝ := Real.arcsin (t / Real.sqrt (1 + t^2))
noncomputable def curve_param_y (t: ℝ) : ℝ := Real.arccos (1 / Real.sqrt (1 + t^2))

theorem tangent_normal_at_t1 : 
  curve_param_x 1 = Real.pi / 4 ∧
  curve_param_y 1 = Real.pi / 4 ∧
  ∃ (x y : ℝ), (y = 2*x - Real.pi/4) ∧ (y = -x/2 + 3*Real.pi/8) :=
  sorry

end tangent_normal_at_t1_l6_6253


namespace mean_of_three_added_numbers_l6_6093

theorem mean_of_three_added_numbers (x y z : ℝ) :
  (∀ (s : ℝ), (s / 7 = 75) → (s + x + y + z) / 10 = 90) → (x + y + z) / 3 = 125 :=
by
  intro h
  sorry

end mean_of_three_added_numbers_l6_6093


namespace john_initial_pens_l6_6293

theorem john_initial_pens (P S C : ℝ) (n : ℕ) 
  (h1 : 20 * S = P) 
  (h2 : C = (2 / 3) * S) 
  (h3 : n * C = P)
  (h4 : P > 0) 
  (h5 : S > 0) 
  (h6 : C > 0)
  : n = 30 :=
by
  sorry

end john_initial_pens_l6_6293


namespace find_s_l6_6442

theorem find_s (c d n : ℝ) (h1 : c + d = n) (h2 : c * d = 3) :
  let s := (c + 1/d) * (d + 1/c) 
  in s = 16 / 3 := 
by
  let s := (c + 1 / d) * (d + 1 / c)
  have : s = 16 / 3 := sorry
  exact this

end find_s_l6_6442


namespace ellipse_standard_equation_l6_6097

theorem ellipse_standard_equation
  (a b c : ℝ)
  (h1 : (3 * a) / (-a) + 16 / b = 1)
  (h2 : (3 * a) / c + 16 / (-b) = 1)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : a > b)
  (h6 : a^2 = b^2 + c^2) : 
  (a = 5 ∧ b = 4 ∧ c = 3) ∧ (∀ x y, x^2 / 25 + y^2 / 16 = 1 ↔ (a = 5 ∧ b = 4)) := 
sorry

end ellipse_standard_equation_l6_6097


namespace polygon_with_150_degree_interior_angles_has_12_sides_l6_6169

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l6_6169


namespace stewart_farm_sheep_l6_6585

variable (S H : ℕ)

theorem stewart_farm_sheep
  (ratio_condition : 4 * H = 7 * S)
  (food_per_horse : 230)
  (total_food : 12880)
  (total_food_condition : H * food_per_horse = total_food) :
  S = 32 := 
by {
  have h1 : H = 56, 
  {  -- given H * 230 = 12880 show that H = 56
    sorry,
  },
  have s1 : 7 * S = 224, 
  {  -- given 4 * H = 7 * S and H = 56 show 7 * S = 224
    sorry,
  },
  -- Finally show S = 32 from 7 * S = 224
  sorry,
}

end stewart_farm_sheep_l6_6585


namespace Carlson_initial_jars_max_count_l6_6229

def initial_jar_weight_ratio (c_initial_weight b_initial_weight: ℕ) : Prop := 
  c_initial_weight = 13 * b_initial_weight

def new_jar_weight_ratio (c_new_weight b_new_weight: ℕ) : Prop := 
  c_new_weight = 8 * b_new_weight

theorem Carlson_initial_jars_max_count (c_initial_weight b_initial_weight c_new_weight b_new_weight: ℕ) 
  (h1 : initial_jar_weight_ratio c_initial_weight b_initial_weight) 
  (h2 : new_jar_weight_ratio c_new_weight b_new_weight)
  (h3 : ∀ a: ℕ, c_new_weight = c_initial_weight - a ∧ b_new_weight = b_initial_weight + a) :
  ∃ n: ℕ, n ≤ 23 :=
begin
  sorry,
end

end Carlson_initial_jars_max_count_l6_6229


namespace hiker_walked_distance_first_day_l6_6368

theorem hiker_walked_distance_first_day (h d_1 d_2 d_3 : ℕ) (H₁ : d_1 = 3 * h)
    (H₂ : d_2 = 4 * (h - 1)) (H₃ : d_3 = 30) (H₄ : d_1 + d_2 + d_3 = 68) :
    d_1 = 18 := 
by 
  sorry

end hiker_walked_distance_first_day_l6_6368


namespace regular_polygon_sides_l6_6148

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6148


namespace part_a_part_b_l6_6683

variables {A B C D E F H I J K L G M N P Q : Point}

-- given conditions
def condition1 : Prop := ¬ IsIsoscelesTriangle ABC ∧ IsAcuteTriangle ABC
def condition2 : Altitude AD ABC
def condition3 : Altitude BE ABC
def condition4 : Altitude CF ABC
def condition5 : Orthocenter H ABC
def condition6 : Circumcenter I (Triangle HEF)
def condition7 : Midpoint K (Segment BC)
def condition8 : Midpoint J (Segment EF)
def condition9 : IntersectsAt HJ (Circumcircle (Triangle HEF)) G
def condition10 : IntersectsAt GK (Circumcircle (Triangle HEF)) L ∧ L ≠ G

-- prove part (a)
theorem part_a (cond1 : condition1) (cond2 : condition2) (cond3 : condition3) 
  (cond4 : condition4) (cond5 : condition5) (cond6 : condition6)
  (cond7 : condition7) (cond8 : condition8) (cond9 : condition9) 
  (cond10 : condition10) : Perpendicular (Line AL) (Line EF) := 
sorry

-- prove part (b)
theorem part_b (cond1 : condition1) (cond2 : condition2) (cond3 : condition3) 
  (cond4 : condition4) (cond5 : condition5) (cond6 : condition6)
  (cond7 : condition7) (cond8 : condition8) (cond9 : condition9) 
  (cond10 : condition10) (intersection1 : IntersectsAt (line AL) (line EF) M) 
  (intersection2 : IntersectsCircumcircleAgain (line IM) (circumcirlce (triangle IEF)) N)
  (intersection3 : IntersectsAt (line DN) (line AB) P)
  (intersection4 : IntersectsAt (line DN) (line AC) Q) :
  Concurrent (line PE) (line QF) (line AK) :=
sorry

end part_a_part_b_l6_6683


namespace problem_1_problem_2_l6_6295

noncomputable def f (x p : ℝ) := p * x - p / x - 2 * Real.log x
noncomputable def g (x : ℝ) := 2 * Real.exp 1 / x

theorem problem_1 (p : ℝ) : 
  (∀ x : ℝ, 0 < x → p * x - p / x - 2 * Real.log x ≥ 0) ↔ p ≥ 1 := 
by sorry

theorem problem_2 (p : ℝ) : 
  (∃ x_0 : ℝ, 1 ≤ x_0 ∧ x_0 ≤ Real.exp 1 ∧ f x_0 p > g x_0) ↔ 
  p > 4 * Real.exp 1 / (Real.exp 2 - 1) :=
by sorry

end problem_1_problem_2_l6_6295


namespace value_of_expression_l6_6430

theorem value_of_expression (n : ℝ) (h : n + 1/n = 6) : n^2 + 1/n^2 + 9 = 43 :=
by
  sorry

end value_of_expression_l6_6430


namespace smallest_result_l6_6880

-- Define the given set of numbers
def given_set : Set Nat := {3, 4, 7, 11, 13, 14}

-- Define the condition for prime numbers greater than 10
def is_prime_gt_10 (n : Nat) : Prop :=
  Nat.Prime n ∧ n > 10

-- Define the property of choosing three different numbers and computing the result
def compute (a b c : Nat) : Nat :=
  (a + b) * c

-- The main theorem stating the problem and its solution
theorem smallest_result : ∃ (a b c : Nat), 
  a ∈ given_set ∧ b ∈ given_set ∧ c ∈ given_set ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (is_prime_gt_10 a ∨ is_prime_gt_10 b ∨ is_prime_gt_10 c) ∧
  compute a b c = 77 ∧
  ∀ (a' b' c' : Nat), 
    a' ∈ given_set ∧ b' ∈ given_set ∧ c' ∈ given_set ∧
    a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c' ∧
    (is_prime_gt_10 a' ∨ is_prime_gt_10 b' ∨ is_prime_gt_10 c') →
    compute a' b' c' ≥ 77 :=
by
  -- Proof is not required, hence sorry
  sorry

end smallest_result_l6_6880


namespace Carlson_max_jars_l6_6228

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l6_6228


namespace parabola_properties_l6_6812

theorem parabola_properties (a b c: ℝ) (ha : a ≠ 0) (hc : c > 1) (h1 : 4 * a + 2 * b + c = 0) (h2 : -b / (2 * a) = 1/2):
  a * b * c < 0 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = a ∧ a * x2^2 + b * x2 + c = a) ∧ a < -1/2 :=
by {
    sorry
}

end parabola_properties_l6_6812


namespace good_jars_l6_6696

def original_cartons : Nat := 50
def jars_per_carton : Nat := 20
def less_cartons_received : Nat := 20
def damaged_jars_per_5_cartons : Nat := 3
def total_damaged_cartons : Nat := 1
def total_good_jars : Nat := 565

theorem good_jars (original_cartons jars_per_carton less_cartons_received damaged_jars_per_5_cartons total_damaged_cartons : Nat) :
  (original_cartons - less_cartons_received) * jars_per_carton 
  - (5 * damaged_jars_per_5_cartons + total_damaged_cartons * jars_per_carton) = total_good_jars := 
by 
  sorry

end good_jars_l6_6696


namespace lines_intersect_l6_6014

-- Define the coefficients of the lines
def A1 : ℝ := 3
def B1 : ℝ := -2
def C1 : ℝ := 5

def A2 : ℝ := 1
def B2 : ℝ := 3
def C2 : ℝ := 10

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := A1 * x + B1 * y + C1 = 0
def line2 (x y : ℝ) : Prop := A2 * x + B2 * y + C2 = 0

-- Mathematical problem to prove
theorem lines_intersect : ∃ (x y : ℝ), line1 x y ∧ line2 x y :=
by
  sorry

end lines_intersect_l6_6014


namespace line_AB_equation_l6_6553

theorem line_AB_equation (m : ℝ) (A B : ℝ × ℝ)
  (hA : A = (0, 0)) (hA_line : ∀ (x y : ℝ), A = (x, y) → x + m * y = 0)
  (hB : B = (1, 3)) (hB_line : ∀ (x y : ℝ), B = (x, y) → m * x - y - m + 3 = 0) :
  ∃ (a b c : ℝ), a * 1 - b * 3 + c = 0 ∧ a * x + b * y + c * 0 = 0 ∧ 3 * x - y + 0 = 0 :=
by
  sorry

end line_AB_equation_l6_6553


namespace inscribed_rectangle_circumference_l6_6358

def rectangle : Type := {width : ℝ, height : ℝ}

def inscribed_circle (r : rectangle) : Type := {radius : ℝ}

theorem inscribed_rectangle_circumference:
  ∀ (r : rectangle) (c : inscribed_circle r), 
    r.width = 9 ∧ r.height = 12 → c.radius = 15 / 2 → 
    2 * Real.pi * c.radius = 15 * Real.pi :=
by
  intros
  sorry

end inscribed_rectangle_circumference_l6_6358


namespace friend_selling_price_l6_6901

-- Definitions and conditions
def original_cost_price : ℝ := 51724.14

def loss_percentage : ℝ := 0.13
def gain_percentage : ℝ := 0.20

def selling_price_man (CP : ℝ) : ℝ := (1 - loss_percentage) * CP
def selling_price_friend (SP1 : ℝ) : ℝ := (1 + gain_percentage) * SP1

-- Prove that the friend's selling price is 54,000 given the conditions
theorem friend_selling_price :
  selling_price_friend (selling_price_man original_cost_price) = 54000 :=
by
  sorry

end friend_selling_price_l6_6901


namespace find_number_of_breeding_rabbits_l6_6985

def breeding_rabbits_condition (B : ℕ) : Prop :=
  ∃ (kittens_first_spring remaining_kittens_first_spring kittens_second_spring remaining_kittens_second_spring : ℕ),
    kittens_first_spring = 10 * B ∧
    remaining_kittens_first_spring = 5 * B + 5 ∧
    kittens_second_spring = 60 ∧
    remaining_kittens_second_spring = kittens_second_spring - 4 ∧
    B + remaining_kittens_first_spring + remaining_kittens_second_spring = 121

theorem find_number_of_breeding_rabbits (B : ℕ) : breeding_rabbits_condition B → B = 10 :=
by
  sorry

end find_number_of_breeding_rabbits_l6_6985


namespace maximum_initial_jars_l6_6235

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l6_6235


namespace range_of_2a_plus_b_l6_6433

variable {a b c A B C : Real}
variable {sin cos : Real → Real}

theorem range_of_2a_plus_b (h1 : a^2 + b^2 + ab = 4) (h2 : c = 2) (h3 : a = c * sin A / sin C) (h4 : b = c * sin B / sin C) :
  2 < 2 * a + b ∧ 2 * a + b < 4 :=
by
  sorry

end range_of_2a_plus_b_l6_6433


namespace average_visitors_in_month_of_30_days_starting_with_sunday_l6_6898

def average_visitors_per_day (sundays_visitors : ℕ) (other_days_visitors : ℕ) (num_sundays : ℕ) (num_other_days : ℕ) : ℕ :=
  (sundays_visitors * num_sundays + other_days_visitors * num_other_days) / (num_sundays + num_other_days)

theorem average_visitors_in_month_of_30_days_starting_with_sunday :
  average_visitors_per_day 1000 700 5 25 = 750 := sorry

end average_visitors_in_month_of_30_days_starting_with_sunday_l6_6898


namespace find_a_l6_6951

noncomputable def M (a : ℤ) : Set ℤ := {a, 0}
noncomputable def N : Set ℤ := { x : ℤ | 2 * x^2 - 3 * x < 0 }

theorem find_a (a : ℤ) (h : (M a ∩ N).Nonempty) : a = 1 := sorry

end find_a_l6_6951


namespace functional_eq_solution_l6_6247

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x := 
sorry

end functional_eq_solution_l6_6247


namespace monotonic_intervals_extreme_values_in_interval_l6_6948

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 3 * x + 2 * Real.log x

theorem monotonic_intervals : (∀ x > 0, 0 < x ∧ x < 1 → f x > 0 ∨ 2 < x → f x > 0) ∧ 
  (∀ x > 0, 1 < x ∧ x < 2 → f x < 0) := 
  sorry

theorem extreme_values_in_interval : 
  f 1 = -(5 / 2) ∧ 
  f 2 = 2 * Real.log 2 - 4 ∧ 
  f 3 = 2 * Real.log 3 - 9 / 2 ∧ 
  ∀ x ∈ Set.Icc 1 3, ∃ (minval : ℝ) (maxval : ℝ), minval = f 2 ∧ maxval = f 3 :=
  sorry

end monotonic_intervals_extreme_values_in_interval_l6_6948


namespace line_through_point_with_equal_intercepts_l6_6026

-- Definition of the conditions
def point := (1 : ℝ, 2 : ℝ)
def eq_intercepts (line : ℝ → ℝ) := ∃ a b : ℝ, a = b ∧ (∀ x, line x = b - x * (b/a))

-- The proof statement
theorem line_through_point_with_equal_intercepts (line : ℝ → ℝ) : 
  (line 1 = 2 ∧ eq_intercepts line) → (line = (λ x, 2 * x) ∨ line = (λ x, 3 - x)) :=
by
  sorry

end line_through_point_with_equal_intercepts_l6_6026


namespace addition_results_in_perfect_square_l6_6893

theorem addition_results_in_perfect_square : ∃ n: ℕ, n * n = 4440 + 49 :=
by
  sorry

end addition_results_in_perfect_square_l6_6893


namespace digit_distribution_l6_6290

theorem digit_distribution (n : ℕ) (d1 d2 d5 do : ℚ) (h : d1 = 1 / 2 ∧ d2 = 1 / 5 ∧ d5 = 1 / 5 ∧ do = 1 / 10) :
  d1 + d2 + d5 + do = 1 → n = 10 :=
begin
  sorry
end

end digit_distribution_l6_6290


namespace cubic_polynomial_roots_l6_6554

noncomputable def cubic_polynomial (a_3 a_2 a_1 a_0 x : ℝ) : ℝ :=
  a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0

theorem cubic_polynomial_roots (a_3 a_2 a_1 a_0 : ℝ) 
    (h_nonzero_a3 : a_3 ≠ 0)
    (r1 r2 r3 : ℝ)
    (h_roots : cubic_polynomial a_3 a_2 a_1 a_0 r1 = 0 ∧
               cubic_polynomial a_3 a_2 a_1 a_0 r2 = 0 ∧
               cubic_polynomial a_3 a_2 a_1 a_0 r3 = 0)
    (h_condition : (cubic_polynomial a_3 a_2 a_1 a_0 (1/2) 
                    + cubic_polynomial a_3 a_2 a_1 a_0 (-1/2)) 
                    / (cubic_polynomial a_3 a_2 a_1 a_0 0) = 1003) :
  (1 / (r1 * r2) + 1 / (r2 * r3) + 1 / (r3 * r1)) = 2002 :=
sorry

end cubic_polynomial_roots_l6_6554


namespace regular_polygon_sides_l6_6204

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6204


namespace quadratic_has_two_distinct_real_roots_l6_6721

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + m * x₁ - 8 = 0) ∧ (x₂^2 + m * x₂ - 8 = 0) :=
by
  let Δ := m^2 + 32
  have hΔ : Δ > 0 := by
    simp [Δ]
    exact add_pos_of_nonneg_of_pos (sq_nonneg m) (by norm_num)
  sorry

end quadratic_has_two_distinct_real_roots_l6_6721


namespace regular_polygon_sides_l6_6184

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6184


namespace percent_increase_correct_l6_6761

-- Define the original and new visual ranges
def original_range : Float := 90
def new_range : Float := 150

-- Define the calculation for percent increase
def percent_increase : Float :=
  ((new_range - original_range) / original_range) * 100

-- Statement to prove
theorem percent_increase_correct : percent_increase = 66.67 :=
by
  -- To be proved
  sorry

end percent_increase_correct_l6_6761


namespace average_gas_mileage_round_trip_l6_6895

-- necessary definitions related to the problem conditions
def total_distance_one_way := 150
def fuel_efficiency_going := 35
def fuel_efficiency_return := 30
def round_trip_distance := total_distance_one_way + total_distance_one_way

-- calculation of gasoline used for each trip and total usage
def gasoline_used_going := total_distance_one_way / fuel_efficiency_going
def gasoline_used_return := total_distance_one_way / fuel_efficiency_return
def total_gasoline_used := gasoline_used_going + gasoline_used_return

-- calculation of average gas mileage
def average_gas_mileage := round_trip_distance / total_gasoline_used

-- the final theorem to prove the average gas mileage for the round trip 
theorem average_gas_mileage_round_trip : average_gas_mileage = 32 := 
by
  sorry

end average_gas_mileage_round_trip_l6_6895


namespace algebraic_expression_result_l6_6667

theorem algebraic_expression_result (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 - 12 = -11 :=
by
  sorry

end algebraic_expression_result_l6_6667


namespace find_father_age_l6_6750

variable (M F : ℕ)

noncomputable def age_relation_1 : Prop := M = (2 / 5) * F
noncomputable def age_relation_2 : Prop := M + 5 = (1 / 2) * (F + 5)

theorem find_father_age (h1 : age_relation_1 M F) (h2 : age_relation_2 M F) : F = 25 := by
  sorry

end find_father_age_l6_6750


namespace abs_value_expression_l6_6382

theorem abs_value_expression : abs (3 * Real.pi - abs (3 * Real.pi - 10)) = 6 * Real.pi - 10 :=
by sorry

end abs_value_expression_l6_6382


namespace smallest_four_digit_number_divisible_by_40_l6_6594

theorem smallest_four_digit_number_divisible_by_40 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 40 = 0 ∧ ∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 40 = 0 → n <= m :=
by
  use 1000
  sorry

end smallest_four_digit_number_divisible_by_40_l6_6594


namespace toothpick_problem_l6_6590

theorem toothpick_problem : 
  ∃ (N : ℕ), N > 5000 ∧ 
            N % 10 = 9 ∧ 
            N % 9 = 8 ∧ 
            N % 8 = 7 ∧ 
            N % 7 = 6 ∧ 
            N % 6 = 5 ∧ 
            N % 5 = 4 ∧ 
            N = 5039 :=
by
  sorry

end toothpick_problem_l6_6590


namespace problem_solution_l6_6631

theorem problem_solution (x : ℝ) (h : x ≠ 5) : (x ≥ 8) ↔ ((x + 1) / (x - 5) ≥ 3) :=
sorry

end problem_solution_l6_6631


namespace sufficient_not_necessary_condition_l6_6957

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x^2 > 1 → 1 / x < 1) ∧ (¬(1 / x < 1 → x^2 > 1)) :=
by sorry

end sufficient_not_necessary_condition_l6_6957


namespace point_in_second_quadrant_l6_6213

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

def problem_points : List (ℝ × ℝ) :=
  [(1, -2), (2, 1), (-2, -1), (-1, 2)]

theorem point_in_second_quadrant :
  ∃ (p : ℝ × ℝ), p ∈ problem_points ∧ is_in_second_quadrant p.1 p.2 := by
  use (-1, 2)
  sorry

end point_in_second_quadrant_l6_6213


namespace regular_polygon_sides_l6_6149

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6149


namespace temperature_decrease_l6_6971

theorem temperature_decrease (T : ℝ) 
    (h1 : T * (3 / 4) = T - 21)
    (h2 : T > 0) : 
    T = 84 := 
  sorry

end temperature_decrease_l6_6971


namespace polygon_with_150_degree_interior_angles_has_12_sides_l6_6171

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l6_6171


namespace regular_polygon_sides_l6_6198

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l6_6198


namespace teresa_jogged_distance_l6_6573

-- Define the conditions as Lean constants.
def teresa_speed : ℕ := 5 -- Speed in kilometers per hour
def teresa_time : ℕ := 5 -- Time in hours

-- Define the distance formula.
def teresa_distance (speed time : ℕ) : ℕ := speed * time

-- State the theorem.
theorem teresa_jogged_distance : teresa_distance teresa_speed teresa_time = 25 := by
  -- Proof is skipped using 'sorry'.
  sorry

end teresa_jogged_distance_l6_6573


namespace max_value_of_angle_B_l6_6644

theorem max_value_of_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1: a + c = 2 * b)
  (h2: a^2 + b^2 - 2*a*b <= c^2 - 2*b*c - 2*a*c)
  (h3: A + B + C = π)
  (h4: 0 < A ∧ A < π) :  
  B ≤ π / 3 :=
sorry

end max_value_of_angle_B_l6_6644


namespace no_common_points_line_circle_l6_6970

theorem no_common_points_line_circle (m : ℝ) :
  (∃ x y : ℝ, (3 * x + 4 * y + m = 0) ∧ ((x + 1) ^ 2 + (y - 2) ^ 2 = 1)) ↔
  m ∈ (-∞:ℝ, -10) ∪ (0:ℝ, ∞:ℝ) :=
begin
  sorry
end

end no_common_points_line_circle_l6_6970


namespace probability_all_truth_l6_6768

noncomputable def probability_A : ℝ := 0.55
noncomputable def probability_B : ℝ := 0.60
noncomputable def probability_C : ℝ := 0.45
noncomputable def probability_D : ℝ := 0.70

theorem probability_all_truth : 
  (probability_A * probability_B * probability_C * probability_D = 0.10395) := 
by 
  sorry

end probability_all_truth_l6_6768


namespace total_cost_john_paid_l6_6679

theorem total_cost_john_paid 
  (meters_of_cloth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ)
  (h1 : meters_of_cloth = 9.25)
  (h2 : cost_per_meter = 48)
  (h3 : total_cost = meters_of_cloth * cost_per_meter) :
  total_cost = 444 :=
sorry

end total_cost_john_paid_l6_6679


namespace Carlson_max_jars_l6_6226

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l6_6226


namespace part1_part2_l6_6818

open Set

/-- Define sets A and B as per given conditions --/
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

/-- Part 1: Prove the intersection and union with complements --/
theorem part1 :
  A ∩ B = {x | 3 ≤ x ∧ x < 6} ∧ (compl B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
by {
  sorry
}

/-- Part 2: Given C ⊆ B, prove the constraints on a --/
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem part2 (a : ℝ) (h : C a ⊆ B) : 2 ≤ a ∧ a ≤ 8 :=
by {
  sorry
}

end part1_part2_l6_6818


namespace solve_for_m_l6_6259

theorem solve_for_m : ∃ m : ℝ, ((∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → (m = 6)) :=
by
  sorry

end solve_for_m_l6_6259


namespace booknote_unique_letters_count_l6_6582

def booknote_set : Finset Char := {'b', 'o', 'k', 'n', 't', 'e'}

theorem booknote_unique_letters_count : booknote_set.card = 6 :=
by
  sorry

end booknote_unique_letters_count_l6_6582


namespace geometric_sequence_ratio_l6_6100

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (S3 : ℝ) 
  (h1 : a 1 = 1) (h2 : S3 = 3 / 4) 
  (h3 : S3 = a 1 + a 1 * q + a 1 * q^2) :
  q = -1 / 2 := 
by
  sorry

end geometric_sequence_ratio_l6_6100


namespace correct_operation_l6_6885

theorem correct_operation (a b : ℝ) : ((-3 * a^2 * b)^2 = 9 * a^4 * b^2) := sorry

end correct_operation_l6_6885


namespace avg_three_numbers_l6_6104

theorem avg_three_numbers (A B C : ℝ) 
  (h1 : A + B = 53)
  (h2 : B + C = 69)
  (h3 : A + C = 58) : 
  (A + B + C) / 3 = 30 := 
by
  sorry

end avg_three_numbers_l6_6104


namespace Carlson_initial_jars_max_count_l6_6232

def initial_jar_weight_ratio (c_initial_weight b_initial_weight: ℕ) : Prop := 
  c_initial_weight = 13 * b_initial_weight

def new_jar_weight_ratio (c_new_weight b_new_weight: ℕ) : Prop := 
  c_new_weight = 8 * b_new_weight

theorem Carlson_initial_jars_max_count (c_initial_weight b_initial_weight c_new_weight b_new_weight: ℕ) 
  (h1 : initial_jar_weight_ratio c_initial_weight b_initial_weight) 
  (h2 : new_jar_weight_ratio c_new_weight b_new_weight)
  (h3 : ∀ a: ℕ, c_new_weight = c_initial_weight - a ∧ b_new_weight = b_initial_weight + a) :
  ∃ n: ℕ, n ≤ 23 :=
begin
  sorry,
end

end Carlson_initial_jars_max_count_l6_6232


namespace carlson_max_jars_l6_6237

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l6_6237


namespace inscribed_rectangle_circumference_l6_6359

def rectangle : Type := {width : ℝ, height : ℝ}

def inscribed_circle (r : rectangle) : Type := {radius : ℝ}

theorem inscribed_rectangle_circumference:
  ∀ (r : rectangle) (c : inscribed_circle r), 
    r.width = 9 ∧ r.height = 12 → c.radius = 15 / 2 → 
    2 * Real.pi * c.radius = 15 * Real.pi :=
by
  intros
  sorry

end inscribed_rectangle_circumference_l6_6359


namespace proof1_proof2_l6_6081

-- Definitions based on the conditions given in the problem description.

def f1 (x : ℝ) (a : ℝ) : ℝ := abs (3 * x - 1) + a * x + 3

theorem proof1 (x : ℝ) : (f1 x 1 ≤ 4) ↔ 0 ≤ x ∧ x ≤ 1 / 2 := 
by
  sorry

theorem proof2 (a : ℝ) : (-3 ≤ a ∧ a ≤ 3) ↔ 
  ∃ (x : ℝ), ∀ y : ℝ, f1 x a ≤ f1 y a := 
by
  sorry

end proof1_proof2_l6_6081


namespace knights_probability_sum_l6_6736

theorem knights_probability_sum (P : ℚ) (num den : ℕ) 
  (hP : P = 53 / 85) 
  (h_frac : P = num / den) 
  (h_gcd : Nat.gcd num den = 1) : 
  num + den = 138 := 
by {
  -- The proof steps would determine the conditions were met but we use sorry here
  sorry
}

end knights_probability_sum_l6_6736


namespace distinct_digit_sum_l6_6054

theorem distinct_digit_sum (a b c d : ℕ) (h1 : a + c = 10) (h2 : b + c = 9) (h3 : a + d = 1)
  (h4 : a ≠ b) (h5 : a ≠ c) (h6 : a ≠ d) (h7 : b ≠ c) (h8 : b ≠ d) (h9 : c ≠ d)
  (h10 : a < 10) (h11 : b < 10) (h12 : c < 10) (h13 : d < 10)
  (h14 : 0 ≤ a) (h15 : 0 ≤ b) (h16 : 0 ≤ c) (h17 : 0 ≤ d) :
  a + b + c + d = 18 :=
sorry

end distinct_digit_sum_l6_6054


namespace larger_integer_is_72_l6_6337

theorem larger_integer_is_72 (x y : ℤ) (h1 : y = 4 * x) (h2 : (x + 6) * 3 = y) : y = 72 :=
sorry

end larger_integer_is_72_l6_6337


namespace closest_multiple_of_17_to_2502_is_2499_l6_6478

def isNearestMultipleOf17 (m n : ℤ) : Prop :=
  ∃ k : ℤ, 17 * k = n ∧ abs (m - n) ≤ abs (m - 17 * (k + 1)) ∧ abs (m - n) ≤ abs (m - 17 * (k - 1))

theorem closest_multiple_of_17_to_2502_is_2499 :
  isNearestMultipleOf17 2502 2499 :=
sorry

end closest_multiple_of_17_to_2502_is_2499_l6_6478


namespace temperature_difference_l6_6867

theorem temperature_difference (initial_temp rise fall : ℤ) (h1 : initial_temp = 25)
    (h2 : rise = 3) (h3 : fall = 15) : initial_temp + rise - fall = 13 := by
  rw [h1, h2, h3]
  norm_num

end temperature_difference_l6_6867


namespace walking_time_l6_6976

theorem walking_time (r s : ℕ) (h₁ : r + s = 50) (h₂ : 2 * s = 30) : 2 * r = 70 :=
by
  sorry

end walking_time_l6_6976


namespace tangent_line_at_origin_local_minimum_at_zero_number_of_zeros_l6_6649

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x * (x - 1) - 1 / 2 * Real.exp a * x^2

theorem tangent_line_at_origin (a : ℝ) (h : a < 0) : 
  let f₀ := f 0 a
  ∃ c : ℝ, (∀ x : ℝ,  f₀ + c * x = -1) := sorry

theorem local_minimum_at_zero (a : ℝ) (h : a < 0) :
  ∀ x : ℝ, f 0 a ≤ f x a := sorry

theorem number_of_zeros (a : ℝ) (h : a < 0) :
  ∃! x : ℝ, f x a = 0 := sorry

end tangent_line_at_origin_local_minimum_at_zero_number_of_zeros_l6_6649


namespace rectangle_inscribed_circle_circumference_l6_6362

-- Define the conditions
def rectangle_width : ℝ := 9
def rectangle_height : ℝ := 12

-- The Lean theorem statement
theorem rectangle_inscribed_circle_circumference (w h : ℝ) (hw : w = 9) (hh : h = 12) : 
    let d := Real.sqrt (w^2 + h^2) in
    let C := Real.pi * d in
    C = 15 * Real.pi :=
by
    rw [hw, hh]
    have h_diag : sqrt (rectangle_width^2 + rectangle_height^2) = 15 := by
        sorry
    rw h_diag
    rw [←mul_assoc, mul_one]

end rectangle_inscribed_circle_circumference_l6_6362


namespace add_fractions_l6_6791

theorem add_fractions : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by sorry

end add_fractions_l6_6791


namespace snail_kite_first_day_snails_l6_6767

theorem snail_kite_first_day_snails (x : ℕ) 
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 35) : 
  x = 3 :=
sorry

end snail_kite_first_day_snails_l6_6767


namespace cos_sum_proof_l6_6809

theorem cos_sum_proof (x : ℝ) (h : Real.cos (x - (Real.pi / 6)) = Real.sqrt 3 / 3) :
  Real.cos x + Real.cos (x - Real.pi / 3) = 1 := 
sorry

end cos_sum_proof_l6_6809


namespace carlson_max_jars_l6_6223

theorem carlson_max_jars (n a k : ℕ) (h1 : a = 5 * k)
  (h2 : n = 9 * k)
  (total_weight_carlson : 13 * n)
  (total_weight_baby : n)
  (h3 : 13 * n - a = 8 * (n + a)) :
  ∃ (j : ℕ), j ≤ 23 :=
by sorry

end carlson_max_jars_l6_6223


namespace regular_polygon_sides_l6_6168

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6168


namespace regular_polygon_sides_l6_6196

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l6_6196


namespace base7_to_base10_conversion_l6_6678

theorem base7_to_base10_conversion (n: ℕ) (H: n = 3652) : 
  (3 * 7^3 + 6 * 7^2 + 5 * 7^1 + 2 * 7^0 = 1360) := by
  sorry

end base7_to_base10_conversion_l6_6678


namespace triangle_side_lengths_inequality_iff_l6_6909

theorem triangle_side_lengths_inequality_iff :
  {x : ℕ | 7 < x^2 ∧ x^2 < 17} = {3, 4} :=
by
  sorry

end triangle_side_lengths_inequality_iff_l6_6909


namespace find_table_price_l6_6487

noncomputable def chair_price (C T : ℝ) : Prop := 2 * C + T = 0.6 * (C + 2 * T)
noncomputable def chair_table_sum (C T : ℝ) : Prop := C + T = 64

theorem find_table_price (C T : ℝ) (h1 : chair_price C T) (h2 : chair_table_sum C T) : T = 56 :=
by sorry

end find_table_price_l6_6487


namespace regular_polygon_sides_l6_6129

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l6_6129


namespace dorothy_will_be_twice_as_old_l6_6502

-- Define some variables
variables (D S Y : ℕ)

-- Hypothesis
def dorothy_age_condition (D S : ℕ) : Prop := D = 3 * S
def dorothy_current_age (D : ℕ) : Prop := D = 15

-- Theorems we want to prove
theorem dorothy_will_be_twice_as_old (D S Y : ℕ) 
  (h1 : dorothy_age_condition D S)
  (h2 : dorothy_current_age D)
  (h3 : D = 15)
  (h4 : S = 5)
  (h5 : D + Y = 2 * (S + Y)) : Y = 5 := 
sorry

end dorothy_will_be_twice_as_old_l6_6502


namespace find_joe_age_l6_6777

noncomputable def billy_age (joe_age : ℕ) : ℕ := 3 * joe_age
noncomputable def emily_age (billy_age joe_age : ℕ) : ℕ := (billy_age + joe_age) / 2

theorem find_joe_age (joe_age : ℕ) 
    (h1 : billy_age joe_age = 3 * joe_age)
    (h2 : emily_age (billy_age joe_age) joe_age = (billy_age joe_age + joe_age) / 2)
    (h3 : billy_age joe_age + joe_age + emily_age (billy_age joe_age) joe_age = 90) : 
    joe_age = 15 :=
by
  sorry

end find_joe_age_l6_6777


namespace at_least_one_divisible_by_5_l6_6702

theorem at_least_one_divisible_by_5 (k m n : ℕ) (hk : ¬ (5 ∣ k)) (hm : ¬ (5 ∣ m)) (hn : ¬ (5 ∣ n)) : 
  (5 ∣ (k^2 - m^2)) ∨ (5 ∣ (m^2 - n^2)) ∨ (5 ∣ (n^2 - k^2)) :=
by {
    sorry
}

end at_least_one_divisible_by_5_l6_6702


namespace all_statements_false_l6_6668

theorem all_statements_false (r1 r2 : ℝ) (h1 : r1 ≠ r2) (h2 : r1 + r2 = 5) (h3 : r1 * r2 = 6) :
  ¬(|r1 + r2| > 6) ∧ ¬(3 < |r1 * r2| ∧ |r1 * r2| < 8) ∧ ¬(r1 < 0 ∧ r2 < 0) :=
by
  sorry

end all_statements_false_l6_6668


namespace total_balls_without_holes_l6_6700

theorem total_balls_without_holes 
  (soccer_balls : ℕ) (soccer_balls_with_hole : ℕ)
  (basketballs : ℕ) (basketballs_with_hole : ℕ)
  (h1 : soccer_balls = 40)
  (h2 : soccer_balls_with_hole = 30)
  (h3 : basketballs = 15)
  (h4 : basketballs_with_hole = 7) :
  soccer_balls - soccer_balls_with_hole + (basketballs - basketballs_with_hole) = 18 := by
  sorry

end total_balls_without_holes_l6_6700


namespace simplified_expression_eq_l6_6999

noncomputable def simplify_expression : ℚ :=
  1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)

-- We need to prove that the simplifed expression is equal to 1 / 39
theorem simplified_expression_eq : simplify_expression = 1 / 39 :=
by sorry

end simplified_expression_eq_l6_6999


namespace find_upper_book_pages_l6_6612

noncomputable def pages_in_upper_book (total_digits : ℕ) (page_diff : ℕ) : ℕ :=
  -- Here we would include the logic to determine the number of pages, but we are only focusing on the statement.
  207

theorem find_upper_book_pages :
  ∀ (total_digits page_diff : ℕ), total_digits = 999 → page_diff = 9 → pages_in_upper_book total_digits page_diff = 207 :=
by
  intros total_digits page_diff h1 h2
  sorry

end find_upper_book_pages_l6_6612


namespace moles_of_NaHCO3_needed_l6_6931

theorem moles_of_NaHCO3_needed 
  (HC2H3O2_moles: ℕ)
  (H2O_moles: ℕ)
  (NaHCO3_HC2H3O2_molar_ratio: ℕ)
  (reaction: NaHCO3_HC2H3O2_molar_ratio = 1 ∧ H2O_moles = 3) :
  ∃ NaHCO3_moles : ℕ, NaHCO3_moles = 3 :=
by
  sorry

end moles_of_NaHCO3_needed_l6_6931


namespace edward_money_left_l6_6795

def earnings_from_lawns (lawns_mowed : Nat) (dollar_per_lawn : Nat) : Nat :=
  lawns_mowed * dollar_per_lawn

def earnings_from_gardens (gardens_cleaned : Nat) (dollar_per_garden : Nat) : Nat :=
  gardens_cleaned * dollar_per_garden

def total_earnings (earnings_lawns : Nat) (earnings_gardens : Nat) : Nat :=
  earnings_lawns + earnings_gardens

def total_expenses (fuel_expense : Nat) (equipment_expense : Nat) : Nat :=
  fuel_expense + equipment_expense

def total_earnings_with_savings (total_earnings : Nat) (savings : Nat) : Nat :=
  total_earnings + savings

def money_left (earnings_with_savings : Nat) (expenses : Nat) : Nat :=
  earnings_with_savings - expenses

theorem edward_money_left : 
  let lawns_mowed := 5
  let dollar_per_lawn := 8
  let gardens_cleaned := 3
  let dollar_per_garden := 12
  let fuel_expense := 10
  let equipment_expense := 15
  let savings := 7
  let earnings_lawns := earnings_from_lawns lawns_mowed dollar_per_lawn
  let earnings_gardens := earnings_from_gardens gardens_cleaned dollar_per_garden
  let total_earnings_work := total_earnings earnings_lawns earnings_gardens
  let expenses := total_expenses fuel_expense equipment_expense
  let earnings_with_savings := total_earnings_with_savings total_earnings_work savings
  money_left earnings_with_savings expenses = 58
:= by sorry

end edward_money_left_l6_6795


namespace equivalent_condition_for_continuity_l6_6662

theorem equivalent_condition_for_continuity {x c d : ℝ} (g : ℝ → ℝ) (h1 : g x = 5 * x - 3) (h2 : ∀ x, |g x - 1| < c → |x - 1| < d) (hc : c > 0) (hd : d > 0) : d ≤ c / 5 :=
sorry

end equivalent_condition_for_continuity_l6_6662


namespace domain_of_f_l6_6248

-- Define the function y = sqrt(x-1) + sqrt(x*(3-x))
noncomputable def f (x : ℝ) := Real.sqrt (x - 1) + Real.sqrt (x * (3 - x))

-- Proposition about the domain of the function
theorem domain_of_f (x : ℝ) : (∃ y : ℝ, y = f x) ↔ 1 ≤ x ∧ x ≤ 3 :=
by
  sorry

end domain_of_f_l6_6248


namespace ball_third_bounce_distance_is_correct_l6_6373

noncomputable def total_distance_third_bounce (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  initial_height + 2 * (initial_height * rebound_ratio) + 2 * (initial_height * rebound_ratio^2)

theorem ball_third_bounce_distance_is_correct : 
  total_distance_third_bounce 80 (2/3) = 257.78 := 
by
  sorry

end ball_third_bounce_distance_is_correct_l6_6373


namespace no_integer_solution_l6_6008

theorem no_integer_solution (a b : ℤ) : ¬(a^2 + b^2 = 10^100 + 3) :=
sorry

end no_integer_solution_l6_6008


namespace intersection_point_l6_6399

variable (x y : ℝ)

theorem intersection_point :
  (y = 9 / (x^2 + 3)) →
  (x + y = 3) →
  (x = 0) := by
  intros h1 h2
  sorry

end intersection_point_l6_6399


namespace empty_set_iff_k_single_element_set_iff_k_l6_6817

noncomputable def quadratic_set (k : ℝ) : Set ℝ := {x | k * x^2 - 3 * x + 2 = 0}

theorem empty_set_iff_k (k : ℝ) : 
  quadratic_set k = ∅ ↔ k > 9/8 := by
  sorry

theorem single_element_set_iff_k (k : ℝ) : 
  (∃ x : ℝ, quadratic_set k = {x}) ↔ (k = 0 ∧ quadratic_set k = {2 / 3}) ∨ (k = 9 / 8 ∧ quadratic_set k = {4 / 3}) := by
  sorry

end empty_set_iff_k_single_element_set_iff_k_l6_6817


namespace fraction_meaningful_condition_l6_6094

theorem fraction_meaningful_condition (x : ℝ) : (∃ y, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end fraction_meaningful_condition_l6_6094


namespace rachel_assembly_time_l6_6994

theorem rachel_assembly_time :
  let chairs := 20
  let tables := 8
  let bookshelves := 5
  let time_per_chair := 6
  let time_per_table := 8
  let time_per_bookshelf := 12
  let total_chairs_time := chairs * time_per_chair
  let total_tables_time := tables * time_per_table
  let total_bookshelves_time := bookshelves * time_per_bookshelf
  total_chairs_time + total_tables_time + total_bookshelves_time = 244 := by
  sorry

end rachel_assembly_time_l6_6994


namespace f_2_solutions_l6_6635

theorem f_2_solutions : 
  ∀ (x y : ℤ), 
    (1 ≤ x) ∧ (0 ≤ y) ∧ (y ≤ (-x + 2)) → 
    (∃ (a b c : Int), 
      (a = 1 ∧ (b = 0 ∨ b = 1) ∨ 
       a = 2 ∧ b = 0) ∧ 
      a = x ∧ b = y ∨ 
      c = 3 → false) ∧ 
    (∃ n : ℕ, n = 3) := by
  sorry

end f_2_solutions_l6_6635


namespace maximal_edges_in_graph_l6_6548

-- Define the problem conditions
variables {α : Type*} [Fintype α]

def max_edges_in_graph (G : SimpleGraph α) : ℕ :=
  if h : Fintype.card α = 100 then
    let vertices := Fintype.elems α in
    let neighbors_disjoint (u v : α) := (u ≠ v) → 
      (G.neighborSet u ∩ G.neighborSet v = ∅) in
    if ∀ u ∈ vertices, ∃ v ∈ G.neighborSet u, neighbors_disjoint u v then
      3822
    else
      0
  else
    0

-- The theorem statement
theorem maximal_edges_in_graph 
  (G : SimpleGraph α)
  (h_card : Fintype.card α = 100)
  (h_condition : ∀ u ∈ (Fintype.elems α), ∃ v ∈ G.neighborSet u, G.neighborSet u ∩ G.neighborSet v = ∅) :
  max_edges_in_graph G = 3822 :=
by 
  -- proof would go here
  sorry

end maximal_edges_in_graph_l6_6548


namespace regular_polygon_sides_l6_6142

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6142


namespace reema_loan_period_l6_6563

theorem reema_loan_period (P SI : ℕ) (R : ℚ) (h1 : P = 1200) (h2 : SI = 432) (h3 : R = 6) : 
  ∃ T : ℕ, SI = (P * R * T) / 100 ∧ T = 6 :=
by
  sorry

end reema_loan_period_l6_6563


namespace clare_money_left_l6_6494

noncomputable def cost_of_bread : ℝ := 4 * 2
noncomputable def cost_of_milk : ℝ := 2 * 2
noncomputable def cost_of_cereal : ℝ := 3 * 3
noncomputable def cost_of_apples : ℝ := 1 * 4

noncomputable def total_cost_before_discount : ℝ := cost_of_bread + cost_of_milk + cost_of_cereal + cost_of_apples
noncomputable def discount_amount : ℝ := total_cost_before_discount * 0.1
noncomputable def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount
noncomputable def sales_tax : ℝ := total_cost_after_discount * 0.05
noncomputable def total_cost_after_discount_and_tax : ℝ := total_cost_after_discount + sales_tax

noncomputable def initial_amount : ℝ := 47
noncomputable def money_left : ℝ := initial_amount - total_cost_after_discount_and_tax

theorem clare_money_left : money_left = 23.37 := by
  sorry

end clare_money_left_l6_6494


namespace smallest_b_l6_6551

noncomputable def geometric_sequence : Prop :=
∃ (a b c r : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ b = a * r ∧ c = a * r^2 ∧ a * b * c = 216

theorem smallest_b (a b c r: ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_geom: b = a * r ∧ c = a * r^2 ∧ a * b * c = 216) : b = 6 :=
sorry

end smallest_b_l6_6551


namespace oranges_taken_by_susan_l6_6733

-- Defining the conditions
def original_number_of_oranges_in_box : ℕ := 55
def oranges_left_in_box_after_susan_takes : ℕ := 20

-- Statement to prove:
theorem oranges_taken_by_susan :
  original_number_of_oranges_in_box - oranges_left_in_box_after_susan_takes = 35 :=
by
  sorry

end oranges_taken_by_susan_l6_6733


namespace regular_polygon_sides_l6_6195

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l6_6195


namespace divisor_of_difference_is_62_l6_6902

-- The problem conditions as definitions
def x : Int := 859622
def y : Int := 859560
def difference : Int := x - y

-- The proof statement
theorem divisor_of_difference_is_62 (d : Int) (h₁ : d ∣ y) (h₂ : d ∣ difference) : d = 62 := by
  sorry

end divisor_of_difference_is_62_l6_6902


namespace conditional_probability_l6_6759

-- Define the events and sample space
def sample_space : Finset (ℕ × ℕ) :=
  {(1,2), (1,3), (1,4), (2,1), (2,3), (2,4), (3,1), (3,2), (3,4), (4,1), (4,2), (4,3)}

def first_class_products : Set ℕ := {1, 2, 3}

-- Define events A and B
def event_A (x : ℕ × ℕ) : Prop := x.fst ∈ first_class_products
def event_B (x : ℕ × ℕ) : Prop := x.snd ∈ first_class_products

-- Definition of conditional probability P(B|A)
def P_conditional (B A : ℕ × ℕ → Prop) (Ω : Finset (ℕ × ℕ)) : ℚ :=
  (Ω.filter (λ x, B x ∧ A x)).card.to_rat / (Ω.filter A).card.to_rat

-- The conditional probability theorem
theorem conditional_probability : P_conditional event_B event_A sample_space = 2 / 3 :=
by  sorry

end conditional_probability_l6_6759


namespace sum_eq_2184_l6_6443

variable (p q r s : ℝ)

-- Conditions
axiom h1 : r + s = 12 * p
axiom h2 : r * s = 14 * q
axiom h3 : p + q = 12 * r
axiom h4 : p * q = 14 * s
axiom distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

-- Problem: Prove that p + q + r + s = 2184
theorem sum_eq_2184 : p + q + r + s = 2184 := 
by {
  sorry
}

end sum_eq_2184_l6_6443


namespace number_line_distance_l6_6583

theorem number_line_distance (x : ℝ) : (abs (-3 - x) = 2) ↔ (x = -5 ∨ x = -1) :=
by
  sorry

end number_line_distance_l6_6583


namespace inequality_solution_value_l6_6534

theorem inequality_solution_value 
  (a : ℝ)
  (h : ∀ x, (1 < x ∧ x < 2) ↔ (ax / (x - 1) > 1)) :
  a = 1 / 2 :=
sorry

end inequality_solution_value_l6_6534


namespace jane_purchased_pudding_l6_6292

theorem jane_purchased_pudding (p : ℕ) 
  (ice_cream_cost_per_cone : ℕ := 5)
  (num_ice_cream_cones : ℕ := 15)
  (pudding_cost_per_cup : ℕ := 2)
  (cost_difference : ℕ := 65)
  (total_ice_cream_cost : ℕ := num_ice_cream_cones * ice_cream_cost_per_cone) 
  (total_pudding_cost : ℕ := p * pudding_cost_per_cup) :
  total_ice_cream_cost = total_pudding_cost + cost_difference → p = 5 :=
by
  sorry

end jane_purchased_pudding_l6_6292


namespace min_value_of_expression_l6_6936

theorem min_value_of_expression (a b c : ℝ) (hb : b > a) (ha : a > c) (hc : b ≠ 0) :
  ∃ l : ℝ, l = 5.5 ∧ l ≤ (a + b)^2 / b^2 + (b + c)^2 / b^2 + (c + a)^2 / b^2 :=
by
  sorry

end min_value_of_expression_l6_6936


namespace circle_radius_l6_6858

theorem circle_radius (A : ℝ) (r : ℝ) (hA : A = 121 * Real.pi) (hArea : A = Real.pi * r^2) : r = 11 :=
by
  sorry

end circle_radius_l6_6858


namespace spring_length_relationship_l6_6944

def spring_length (x : ℝ) : ℝ := 6 + 0.3 * x

theorem spring_length_relationship (x : ℝ) : spring_length x = 0.3 * x + 6 :=
by sorry

end spring_length_relationship_l6_6944


namespace regular_polygon_sides_l6_6183

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6183


namespace men_sent_to_other_project_l6_6070

-- Let the initial number of men be 50
def initial_men : ℕ := 50
-- Let the time to complete the work initially be 10 days
def initial_days : ℕ := 10
-- Calculate the total work in man-days
def total_work : ℕ := initial_men * initial_days

-- Let the total time taken after sending some men to another project be 30 days
def new_days : ℕ := 30
-- Let the number of men sent to another project be x
variable (x : ℕ)
-- Let the new number of men be (initial_men - x)
def new_men : ℕ := initial_men - x

theorem men_sent_to_other_project (x : ℕ):
total_work = new_men x * new_days -> x = 33 :=
by
  sorry

end men_sent_to_other_project_l6_6070


namespace equal_pair_c_l6_6003

theorem equal_pair_c : (-4)^3 = -(4^3) := 
by {
  sorry
}

end equal_pair_c_l6_6003


namespace ratio_of_money_given_l6_6303

theorem ratio_of_money_given
  (T : ℕ) (W : ℕ) (Th : ℕ) (m : ℕ)
  (h1 : T = 8) 
  (h2 : W = m * T) 
  (h3 : Th = W + 9)
  (h4 : Th = T + 41) : 
  W / T = 5 := 
sorry

end ratio_of_money_given_l6_6303


namespace find_d_l6_6717

theorem find_d (d : ℝ) : (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  { sorry }

end find_d_l6_6717


namespace fraction_solution_l6_6519

theorem fraction_solution (x : ℝ) (h1 : (x - 4) / (x^2) = 0) (h2 : x ≠ 0) : x = 4 :=
sorry

end fraction_solution_l6_6519


namespace find_f3_value_l6_6406

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * Real.tan x - b * x^5 + c * x - 3

theorem find_f3_value (a b c : ℝ) (h : f (-3) a b c = 7) : f 3 a b c = -13 := 
by 
  sorry

end find_f3_value_l6_6406


namespace solve_for_x_l6_6458

theorem solve_for_x (x : ℝ) (h : (6 * x ^ 2 + 111 * x + 1) / (2 * x + 37) = 3 * x + 1) : x = -18 :=
sorry

end solve_for_x_l6_6458


namespace egg_whites_per_cake_l6_6380

-- Define the conversion ratio between tablespoons of aquafaba and egg whites
def tablespoons_per_egg_white : ℕ := 2

-- Define the total amount of aquafaba used for two cakes
def total_tablespoons_for_two_cakes : ℕ := 32

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Prove the number of egg whites needed per cake
theorem egg_whites_per_cake :
  (total_tablespoons_for_two_cakes / tablespoons_per_egg_white) / number_of_cakes = 8 := by
  sorry

end egg_whites_per_cake_l6_6380


namespace complement_A_eq_interval_l6_6690

open Set

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := {x : ℝ | True}

-- Define the set A according to the given conditions
def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x ≤ 0}

-- State the theorem that the complement of A with respect to U is (0, 1)
theorem complement_A_eq_interval : ∀ x : ℝ, x ∈ U \ A ↔ x ∈ Ioo 0 1 := by
  intros x
  -- Proof skipped
  sorry

end complement_A_eq_interval_l6_6690


namespace no_positive_integer_n_has_perfect_square_form_l6_6308

theorem no_positive_integer_n_has_perfect_square_form (n : ℕ) (h : 0 < n) : 
  ¬ ∃ k : ℕ, n^4 + 2 * n^3 + 2 * n^2 + 2 * n + 1 = k^2 := 
sorry

end no_positive_integer_n_has_perfect_square_form_l6_6308


namespace value_of_p_l6_6464

-- Let us assume the conditions given, and the existence of positive values p and q such that p + q = 1,
-- and the second term and fourth term of the polynomial expansion (x + y)^10 are equal when x = p and y = q.

theorem value_of_p (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (h_sum : p + q = 1) (h_eq_terms : 10 * p ^ 9 * q = 120 * p ^ 7 * q ^ 3) :
    p = Real.sqrt (12 / 13) :=
    by sorry

end value_of_p_l6_6464


namespace carlson_max_jars_l6_6221

theorem carlson_max_jars (n a k : ℕ) (h1 : a = 5 * k)
  (h2 : n = 9 * k)
  (total_weight_carlson : 13 * n)
  (total_weight_baby : n)
  (h3 : 13 * n - a = 8 * (n + a)) :
  ∃ (j : ℕ), j ≤ 23 :=
by sorry

end carlson_max_jars_l6_6221


namespace simplified_expression_eq_l6_6997

noncomputable def simplify_expression : ℚ :=
  1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)

-- We need to prove that the simplifed expression is equal to 1 / 39
theorem simplified_expression_eq : simplify_expression = 1 / 39 :=
by sorry

end simplified_expression_eq_l6_6997


namespace johns_salary_before_raise_l6_6834

variable (x : ℝ)

theorem johns_salary_before_raise (h : x + 0.3333 * x = 80) : x = 60 :=
by
  sorry

end johns_salary_before_raise_l6_6834


namespace probability_sum_of_two_draws_is_three_l6_6828

theorem probability_sum_of_two_draws_is_three :
  let outcomes := [(1, 1), (1, 2), (2, 1), (2, 2)]
  let favorable := [(1, 2), (2, 1)]
  (favorable.length : ℚ) / (outcomes.length : ℚ) = 1 / 2 :=
by
  sorry

end probability_sum_of_two_draws_is_three_l6_6828


namespace total_percentage_increase_l6_6214

def initial_time : ℝ := 45
def additive_A_increase : ℝ := 0.35
def additive_B_increase : ℝ := 0.20

theorem total_percentage_increase :
  let time_after_A := initial_time * (1 + additive_A_increase)
  let time_after_B := time_after_A * (1 + additive_B_increase)
  (time_after_B - initial_time) / initial_time * 100 = 62 :=
  sorry

end total_percentage_increase_l6_6214


namespace min_value_expression_l6_6744

theorem min_value_expression : ∀ x : ℝ, (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 ≥ 3625 :=
by
  sorry

end min_value_expression_l6_6744


namespace reflect_P_across_x_axis_l6_6066

def point_reflection_over_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem reflect_P_across_x_axis : 
  point_reflection_over_x_axis (-3, 1) = (-3, -1) :=
  by
    sorry

end reflect_P_across_x_axis_l6_6066


namespace pascal_no_divisible_by_prime_iff_form_l6_6982

theorem pascal_no_divisible_by_prime_iff_form (p : ℕ) (n : ℕ) 
  (hp : Nat.Prime p) :
  (∀ k ≤ n, Nat.choose n k % p ≠ 0) ↔ ∃ s q : ℕ, s ≥ 0 ∧ 0 < q ∧ q < p ∧ n = p^s * q - 1 :=
by
  sorry

end pascal_no_divisible_by_prime_iff_form_l6_6982


namespace shopping_people_count_l6_6870

theorem shopping_people_count :
  ∃ P : ℕ, P = 10 ∧
  ∃ (stores : ℕ) (total_visits : ℕ) (two_store_visitors : ℕ) 
    (at_least_one_store_visitors : ℕ) (max_stores_visited : ℕ),
    stores = 8 ∧
    total_visits = 22 ∧
    two_store_visitors = 8 ∧
    at_least_one_store_visitors = P ∧
    max_stores_visited = 3 ∧
    total_visits = (two_store_visitors * 2) + 6 ∧
    P = two_store_visitors + 2 :=
by {
    sorry
}

end shopping_people_count_l6_6870


namespace staircase_ways_four_steps_l6_6381

theorem staircase_ways_four_steps : 
  let one_step := 1
  let two_steps := 2
  let three_steps := 3
  let four_steps := 4
  1           -- one step at a time
  + 3         -- combination of one and two steps
  + 2         -- combination of one and three steps
  + 1         -- two steps at a time
  + 1 = 8     -- all four steps in one stride
:= by
  sorry

end staircase_ways_four_steps_l6_6381


namespace euler_no_k_divisible_l6_6374

theorem euler_no_k_divisible (n : ℕ) (k : ℕ) (h : k < 5^n - 5^(n-1)) : ¬ (5^n ∣ 2^k - 1) := 
sorry

end euler_no_k_divisible_l6_6374


namespace eval_expression_l6_6250

theorem eval_expression : ⌈- (7 / 3 : ℚ)⌉ + ⌊(7 / 3 : ℚ)⌋ = 0 := 
by 
  sorry

end eval_expression_l6_6250


namespace domain_of_function_l6_6545

noncomputable def function_defined (x : ℝ) : Prop :=
  (x > 1) ∧ (x ≠ 2)

theorem domain_of_function :
  ∀ x : ℝ, (∃ y : ℝ, y = (1 / (Real.sqrt (x - 1))) + (1 / (x - 2))) ↔ function_defined x :=
by sorry

end domain_of_function_l6_6545


namespace sum_of_three_consecutive_integers_product_504_l6_6321

theorem sum_of_three_consecutive_integers_product_504 : 
  ∃ n : ℤ, n * (n + 1) * (n + 2) = 504 ∧ n + (n + 1) + (n + 2) = 24 := 
by
  sorry

end sum_of_three_consecutive_integers_product_504_l6_6321


namespace find_d_l6_6822

theorem find_d (c : ℝ) (d : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ) (ω : ℝ)
  (h1 : α = c)
  (h2 : β = 43)
  (h3 : γ = 59)
  (h4 : ω = d)
  (h5 : c = 36) :
  d = 42 := 
sorry

end find_d_l6_6822


namespace starting_number_of_three_squares_less_than_2300_l6_6868

theorem starting_number_of_three_squares_less_than_2300 : 
  ∃ n1 n2 n3 : ℕ, n1 < n2 ∧ n2 < n3 ∧ n3^2 < 2300 ∧ n2^2 < 2300 ∧ n1^2 < 2300 ∧ n3^2 ≥ 2209 ∧ n2^2 ≥ 2116 ∧ n1^2 = 2025 :=
by {
  sorry
}

end starting_number_of_three_squares_less_than_2300_l6_6868


namespace p_implies_q_l6_6511

theorem p_implies_q (x : ℝ) :
  (|2*x - 3| < 1) → (x*(x - 3) < 0) :=
by
  intros hp
  sorry

end p_implies_q_l6_6511


namespace selling_price_eq_120_l6_6772

-- Definitions based on the conditions
def cost_price : ℝ := 96
def profit_percentage : ℝ := 0.25

-- The proof statement
theorem selling_price_eq_120 (cost_price : ℝ) (profit_percentage : ℝ) : cost_price = 96 → profit_percentage = 0.25 → (cost_price + cost_price * profit_percentage) = 120 :=
by
  intros hcost hprofit
  rw [hcost, hprofit]
  sorry

end selling_price_eq_120_l6_6772


namespace rope_total_length_is_54m_l6_6740

noncomputable def totalRopeLength : ℝ :=
  let horizontalDistance : ℝ := 16
  let heightAB : ℝ := 18
  let heightCD : ℝ := 30
  let ropeBC := Real.sqrt (horizontalDistance^2 + (heightCD - heightAB)^2)
  let ropeAC := Real.sqrt (horizontalDistance^2 + heightCD^2)
  ropeBC + ropeAC

theorem rope_total_length_is_54m : totalRopeLength = 54 := sorry

end rope_total_length_is_54m_l6_6740


namespace MeganSavingsExceed500_l6_6447

theorem MeganSavingsExceed500 :
  ∃ n : ℕ, n ≥ 7 ∧ ((3^n - 1) / 2 > 500) :=
sorry

end MeganSavingsExceed500_l6_6447


namespace trigonometric_values_l6_6943

theorem trigonometric_values (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 1 / 3) 
  (h2 : Real.cos x - Real.cos y = 1 / 5) : 
  Real.cos (x + y) = 208 / 225 ∧ Real.sin (x - y) = -15 / 17 := 
by 
  sorry

end trigonometric_values_l6_6943


namespace green_notebook_cost_l6_6851

-- Define the conditions
def num_notebooks : Nat := 4
def num_green_notebooks : Nat := 2
def num_black_notebooks : Nat := 1
def num_pink_notebooks : Nat := 1
def total_cost : ℕ := 45
def black_notebook_cost : ℕ := 15
def pink_notebook_cost : ℕ := 10

-- Define what we need to prove: The cost of each green notebook
def green_notebook_cost_each : ℕ := 10

-- The statement that combines the conditions with the goal to prove
theorem green_notebook_cost : 
  num_notebooks = 4 ∧ 
  num_green_notebooks = 2 ∧ 
  num_black_notebooks = 1 ∧ 
  num_pink_notebooks = 1 ∧ 
  total_cost = 45 ∧ 
  black_notebook_cost = 15 ∧ 
  pink_notebook_cost = 10 →
  2 * green_notebook_cost_each = total_cost - (black_notebook_cost + pink_notebook_cost) :=
by
  sorry

end green_notebook_cost_l6_6851


namespace cost_of_one_pack_l6_6531

-- Given condition
def total_cost (packs: ℕ) : ℕ := 110
def number_of_packs : ℕ := 10

-- Question: How much does one pack cost?
-- We need to prove that one pack costs 11 dollars
theorem cost_of_one_pack : (total_cost number_of_packs) / number_of_packs = 11 :=
by
  sorry

end cost_of_one_pack_l6_6531


namespace combined_proposition_range_l6_6639

def p (a : ℝ) : Prop := ∀ x ∈ ({1, 2} : Set ℝ), 3 * x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem combined_proposition_range (a : ℝ) : 
  (p a ∧ q a) ↔ (a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3)) := 
  sorry

end combined_proposition_range_l6_6639


namespace geometric_sum_first_six_terms_l6_6676

variable (a_n : ℕ → ℝ)

axiom geometric_seq (r a1 : ℝ) : ∀ n, a_n n = a1 * r ^ (n - 1)
axiom a2_val : a_n 2 = 2
axiom a5_val : a_n 5 = 16

theorem geometric_sum_first_six_terms (S6 : ℝ) : S6 = 1 * (1 - 2^6) / (1 - 2) := by
  sorry

end geometric_sum_first_six_terms_l6_6676


namespace regular_polygon_sides_l6_6200

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6200


namespace Sn_divisible_by_3_Sn_divisible_by_9_iff_even_l6_6079

-- Define \( S_n \) following the given conditions
def S (n : ℕ) : ℕ :=
  let a := 2^n + 1 -- first term
  let b := 2^(n+1) - 1 -- last term
  let m := b - a + 1 -- number of terms
  (m * (a + b)) / 2 -- sum of the arithmetic series

-- The first part: Prove that \( S_n \) is divisible by 3 for all positive integers \( n \)
theorem Sn_divisible_by_3 (n : ℕ) (hn : 0 < n) : 3 ∣ S n := sorry

-- The second part: Prove that \( S_n \) is divisible by 9 if and only if \( n \) is even
theorem Sn_divisible_by_9_iff_even (n : ℕ) (hn : 0 < n) : 9 ∣ S n ↔ Even n := sorry

end Sn_divisible_by_3_Sn_divisible_by_9_iff_even_l6_6079


namespace jar_filled_fraction_l6_6600

variable (S L : ℝ)

-- Conditions
axiom h1 : S * (1/3) = L * (1/2)

-- Statement of the problem
theorem jar_filled_fraction :
  (L * (1/2)) + (S * (1/3)) = L := by
sorry

end jar_filled_fraction_l6_6600


namespace edward_rides_eq_8_l6_6354

-- Define the initial conditions
def initial_tickets : ℕ := 79
def spent_tickets : ℕ := 23
def cost_per_ride : ℕ := 7

-- Define the remaining tickets after spending at the booth
def remaining_tickets : ℕ := initial_tickets - spent_tickets

-- Define the number of rides Edward could go on
def number_of_rides : ℕ := remaining_tickets / cost_per_ride

-- The goal is to prove that the number of rides is equal to 8.
theorem edward_rides_eq_8 : number_of_rides = 8 := by sorry

end edward_rides_eq_8_l6_6354


namespace common_points_line_circle_l6_6272

theorem common_points_line_circle (a b : ℝ) :
    (∃ x y : ℝ, x / a + y / b = 1 ∧ x^2 + y^2 = 1) →
    (1 / (a * a) + 1 / (b * b) ≥ 1) :=
by
  sorry

end common_points_line_circle_l6_6272


namespace max_value_of_expression_l6_6080

noncomputable def maximum_value (x y z : ℝ) := 8 * x + 3 * y + 10 * z

theorem max_value_of_expression :
  ∀ (x y z : ℝ), 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 → maximum_value x y z ≤ (Real.sqrt 481) / 6 :=
by
  sorry

end max_value_of_expression_l6_6080


namespace max_value_of_a_l6_6648

noncomputable def f (a x : ℝ) : ℝ := abs (8 * x^3 - 12 * x - a) + a

theorem max_value_of_a :
  (∀ x ∈ set.Icc 0 1, f (-2 * real.sqrt 2) x = 0) ∧
  (∀ a ≤ -2 * real.sqrt 2, ∃ x ∈ set.Icc 0 1, f a x ≠ 0) := 
sorry

end max_value_of_a_l6_6648


namespace revenue_from_full_price_tickets_l6_6906

theorem revenue_from_full_price_tickets (f h p : ℕ) 
    (h1 : f + h = 160) 
    (h2 : f * p + h * (p / 2) = 2400) 
    (h3 : h = 160 - f)
    (h4 : 2 * 2400 = 4800) :
  f * p = 800 := 
sorry

end revenue_from_full_price_tickets_l6_6906


namespace solve_for_x_l6_6090

theorem solve_for_x (x : ℝ) (h : x - 5.90 = 9.28) : x = 15.18 :=
by
  sorry

end solve_for_x_l6_6090


namespace A_n_eq_B_n_l6_6073

open Real

noncomputable def A_n (n : ℕ) : ℝ :=
  1408 * (1 - (1 / (2 : ℝ) ^ n))

noncomputable def B_n (n : ℕ) : ℝ :=
  (3968 / 3) * (1 - (1 / (-2 : ℝ) ^ n))

theorem A_n_eq_B_n : A_n 5 = B_n 5 := sorry

end A_n_eq_B_n_l6_6073


namespace determine_speed_A_l6_6850

theorem determine_speed_A (v1 v2 : ℝ) 
  (h1 : v1 > v2) 
  (h2 : 8 * (v1 + v2) = 6 * (v1 + v2 + 4)) 
  (h3 : 6 * (v1 + 2 - (v2 + 2)) = 6) 
  : v1 = 6.5 :=
by
  sorry

end determine_speed_A_l6_6850


namespace card_combinations_l6_6452

noncomputable def valid_card_combinations : List (ℕ × ℕ × ℕ × ℕ) :=
  [(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)]

theorem card_combinations (a b c d : ℕ) (h : a ≤ b ∧ b ≤ c ∧ c ≤ d) :
  (1, 2, 7, 8) ∈ valid_card_combinations ∨ 
  (1, 3, 6, 8) ∈ valid_card_combinations ∨ 
  (1, 4, 5, 8) ∈ valid_card_combinations ∨ 
  (2, 3, 6, 7) ∈ valid_card_combinations ∨ 
  (2, 4, 5, 7) ∈ valid_card_combinations ∨ 
  (3, 4, 5, 6) ∈ valid_card_combinations :=
sorry

end card_combinations_l6_6452


namespace binary_representation_of_14_binary_representation_of_14_l6_6926

-- Define the problem as a proof goal
theorem binary_representation_of_14 : (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by sorry

-- An alternative formula to exactly represent the binary string using a conversion function can be provided:
theorem binary_representation_of_14' : nat.to_digits 2 14 = [1, 1, 1, 0] :=
by sorry

end binary_representation_of_14_binary_representation_of_14_l6_6926


namespace cylinder_volume_ratio_l6_6896

noncomputable def ratio_of_volumes (r h V_small V_large : ℝ) : ℝ := V_large / V_small

theorem cylinder_volume_ratio (r : ℝ) (h : ℝ) 
  (original_height : ℝ := 3 * r)
  (height_small : ℝ := r / 4)
  (height_large : ℝ := 3 * r - height_small)
  (A_small : ℝ := 2 * π * r * (r + height_small))
  (A_large : ℝ := 2 * π * r * (r + height_large))
  (V_small : ℝ := π * r^2 * height_small) 
  (V_large : ℝ := π * r^2 * height_large) :
  A_large = 3 * A_small → 
  ratio_of_volumes r height_small V_small V_large = 11 := by 
  sorry

end cylinder_volume_ratio_l6_6896


namespace fraction_sum_identity_l6_6555

theorem fraction_sum_identity (p q r : ℝ) (h₀ : p ≠ q) (h₁ : p ≠ r) (h₂ : q ≠ r) 
(h : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 1 / (q - r) + 1 / (r - p) + 1 / (p - q) - 1 := 
sorry

end fraction_sum_identity_l6_6555


namespace closest_vector_l6_6255

open Real

def u (s : ℝ) : ℝ × ℝ × ℝ := (1 + 3 * s, -4 + 7 * s, 2 + 4 * s)
def b : ℝ × ℝ × ℝ := (5, 1, -3)
def direction : ℝ × ℝ × ℝ := (3, 7, 4)

theorem closest_vector (s : ℝ) :
  (u s - b) • direction = 0 ↔ s = 27 / 74 :=
sorry

end closest_vector_l6_6255


namespace all_weights_equal_l6_6330

theorem all_weights_equal (w : Fin 13 → ℤ) 
  (h : ∀ (i : Fin 13), ∃ (a b : Multiset (Fin 12)),
    a + b = (Finset.univ.erase i).val ∧ Multiset.card a = 6 ∧ 
    Multiset.card b = 6 ∧ Multiset.sum (a.map w) = Multiset.sum (b.map w)) :
  ∀ i j, w i = w j :=
by sorry

end all_weights_equal_l6_6330


namespace min_area_quadrilateral_l6_6860

theorem min_area_quadrilateral
  (S_AOB S_COD : ℝ) (h₁ : S_AOB = 4) (h₂ : S_COD = 9) :
  ∃ S_BOC S_AOD, S_AOB + S_COD + S_BOC + S_AOD = 25 :=
by
  sorry

end min_area_quadrilateral_l6_6860


namespace combined_transformation_matrix_l6_6396

-- Definitions for conditions
def dilation_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![s, 0], ![0, s]]

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

-- Theorem to be proven
theorem combined_transformation_matrix :
  (rotation_matrix_90_ccw * dilation_matrix 4) = ![![0, -4], ![4, 0]] :=
by
  sorry

end combined_transformation_matrix_l6_6396


namespace inequality_correct_l6_6032

theorem inequality_correct (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c :=
sorry

end inequality_correct_l6_6032


namespace total_cost_charlotte_spends_l6_6579

-- Definitions of conditions
def original_price : ℝ := 40.00
def discount_rate : ℝ := 0.25
def number_of_people : ℕ := 5

-- Prove the total cost Charlotte will spend given the conditions
theorem total_cost_charlotte_spends : 
  let discounted_price := original_price * (1 - discount_rate)
  in discounted_price * number_of_people = 150 := by
  sorry

end total_cost_charlotte_spends_l6_6579


namespace expression_value_l6_6620

theorem expression_value : 
  ∀ (x y z: ℤ), x = 2 ∧ y = -3 ∧ z = 1 → x^2 + y^2 - z^2 - 2*x*y = 24 := by
  sorry

end expression_value_l6_6620


namespace dima_story_telling_l6_6387

theorem dima_story_telling (initial_spoons final_spoons : ℕ) 
  (h1 : initial_spoons = 26) (h2 : final_spoons = 33696)
  (h3 : ∃ (n : ℕ), final_spoons = initial_spoons * (2^5 * 3^4) * 13) : 
  ∃ n : ℕ, n = 9 := 
sorry

end dima_story_telling_l6_6387


namespace intersection_M_N_l6_6521

def M : Set ℝ := { y | ∃ x : ℝ, y = x^2 }
def N : Set ℝ := { y | ∃ x : ℝ, y = x }

theorem intersection_M_N : (M ∩ N) = { y : ℝ | 0 ≤ y } :=
by
  sorry

end intersection_M_N_l6_6521


namespace regular_polygon_sides_l6_6130

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l6_6130


namespace regular_polygon_sides_l6_6177

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l6_6177


namespace hyperbola_equation_l6_6815

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote_eq : ∀ x y, 3*x + 4*y = 0 → y = (-3/4) * x)
  (focus_eq : (0, 5) = (0, 5)) :
  ∃ a b : ℝ, a = 3 ∧ b = 4 ∧ (∀ y x, (y^2 / 9 - x^2 / 16 = 1)) :=
sorry

end hyperbola_equation_l6_6815


namespace circle_circumference_l6_6360

theorem circle_circumference (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi :=
by
  -- Use the given dimensions to find the diagonal (which is the diameter).
  -- Calculate the circumference using the calculated diameter.
  sorry

end circle_circumference_l6_6360


namespace quadratic_has_two_distinct_real_roots_l6_6724

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  let a := 1
      c := -8
      b := m
      Δ := b^2 - 4 * a * c 
  in (Δ > 0) :=
by
  let a := 1
  let c := -8
  let b := m
  let Δ := b^2 - 4 * a * c
  show Δ > 0
  sorry

end quadratic_has_two_distinct_real_roots_l6_6724


namespace total_cost_charlotte_l6_6578

noncomputable def regular_rate : ℝ := 40.00
noncomputable def discount_rate : ℝ := 0.25
noncomputable def number_of_people : ℕ := 5

theorem total_cost_charlotte :
  number_of_people * (regular_rate * (1 - discount_rate)) = 150.00 := by
  sorry

end total_cost_charlotte_l6_6578


namespace division_example_l6_6617

theorem division_example : 72 / (6 / 3) = 36 :=
by sorry

end division_example_l6_6617


namespace total_balls_without_holes_correct_l6_6699

variable (soccerBalls basketballs soccerBallsWithHoles basketballsWithHoles : ℕ)

def totalBallsWithoutHoles (soccerBalls basketballs soccerBallsWithHoles basketballsWithHoles : ℕ) : ℕ :=
  (soccerBalls - soccerBallsWithHoles) + (basketballs - basketballsWithHoles)

theorem total_balls_without_holes_correct
  (h1 : soccerBalls = 40)
  (h2 : basketballs = 15)
  (h3 : soccerBallsWithHoles = 30)
  (h4 : basketballsWithHoles = 7) :
  totalBallsWithoutHoles 40 15 30 7 = 18 :=
by
  unfold totalBallsWithoutHoles
  rw [h1, h2, h3, h4]
  norm_num
  -- final result should yield 18
  sorry

end total_balls_without_holes_correct_l6_6699


namespace jake_car_washes_l6_6975

theorem jake_car_washes :
  ∀ (washes_per_bottle cost_per_bottle total_spent weekly_washes : ℕ),
  washes_per_bottle = 4 →
  cost_per_bottle = 4 →
  total_spent = 20 →
  weekly_washes = 1 →
  (total_spent / cost_per_bottle) * washes_per_bottle / weekly_washes = 20 :=
by
  intros washes_per_bottle cost_per_bottle total_spent weekly_washes
  sorry

end jake_car_washes_l6_6975


namespace mark_buttons_l6_6558

/-- Mark started the day with some buttons. His friend Shane gave him 3 times that amount of buttons.
    Then his other friend Sam asked if he could have half of Mark’s buttons. 
    Mark ended up with 28 buttons. How many buttons did Mark start the day with? --/
theorem mark_buttons (B : ℕ) (h1 : 2 * B = 28) : B = 14 := by
  sorry

end mark_buttons_l6_6558


namespace total_points_l6_6424

variable (FirstTry SecondTry ThirdTry : ℕ)

def HomerScoringConditions : Prop :=
  FirstTry = 400 ∧
  SecondTry = FirstTry - 70 ∧
  ThirdTry = 2 * SecondTry

theorem total_points (h : HomerScoringConditions FirstTry SecondTry ThirdTry) : 
  FirstTry + SecondTry + ThirdTry = 1390 := 
by
  cases h with
  | intro h1 h2 h3 =>
  sorry

end total_points_l6_6424


namespace digit_distribution_l6_6288

theorem digit_distribution (n: ℕ) : 
(1 / 2) * n + (1 / 5) * n + (1 / 5) * n + (1 / 10) * n = n → 
n = 10 :=
by
  sorry

end digit_distribution_l6_6288


namespace rectangle_inscribed_circle_circumference_l6_6363

-- Define the conditions
def rectangle_width : ℝ := 9
def rectangle_height : ℝ := 12

-- The Lean theorem statement
theorem rectangle_inscribed_circle_circumference (w h : ℝ) (hw : w = 9) (hh : h = 12) : 
    let d := Real.sqrt (w^2 + h^2) in
    let C := Real.pi * d in
    C = 15 * Real.pi :=
by
    rw [hw, hh]
    have h_diag : sqrt (rectangle_width^2 + rectangle_height^2) = 15 := by
        sorry
    rw h_diag
    rw [←mul_assoc, mul_one]

end rectangle_inscribed_circle_circumference_l6_6363


namespace value_of_f_at_3_l6_6495

def f (x : ℝ) := 2 * x - 1

theorem value_of_f_at_3 : f 3 = 5 := by
  sorry

end value_of_f_at_3_l6_6495


namespace equal_pair_c_l6_6004

theorem equal_pair_c : (-4)^3 = -(4^3) := 
by {
  sorry
}

end equal_pair_c_l6_6004


namespace actual_area_of_lawn_l6_6903

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

end actual_area_of_lawn_l6_6903


namespace regular_polygon_sides_l6_6140

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6140


namespace common_chord_length_l6_6952

noncomputable def dist_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / Real.sqrt (a^2 + b^2)

theorem common_chord_length
  (x y : ℝ)
  (h1 : (x-2)^2 + (y-1)^2 = 10)
  (h2 : (x+6)^2 + (y+3)^2 = 50) :
  (dist_to_line (2, 1) 2 1 0 = Real.sqrt 5) →
  2 * Real.sqrt 5 = 2 * Real.sqrt 5 :=
by
  sorry

end common_chord_length_l6_6952


namespace angle_quadrant_l6_6641

theorem angle_quadrant (α : ℝ) (h : 0 < α ∧ α < 90) : 90 < α + 180 ∧ α + 180 < 270 :=
by
  sorry

end angle_quadrant_l6_6641


namespace original_price_of_cycle_l6_6124

noncomputable def original_price_given_gain (SP : ℝ) (gain : ℝ) : ℝ :=
  SP / (1 + gain)

theorem original_price_of_cycle (SP : ℝ) (HSP : SP = 1350) (Hgain : gain = 0.5) : 
  original_price_given_gain SP gain = 900 := 
by
  sorry

end original_price_of_cycle_l6_6124


namespace max_value_f_zero_points_range_k_l6_6276

noncomputable def f (x k : ℝ) : ℝ := 3 * x^2 + 2 * (k - 1) * x + (k + 5)

theorem max_value_f (k : ℝ) (h : k < -7/2 ∨ k ≥ -7/2) :
  ∃ max_val : ℝ, max_val = if k < -7/2 then k + 5 else 7 * k + 26 :=
sorry

theorem zero_points_range_k :
  ∀ k : ℝ, (f 0 k) * (f 3 k) ≤ 0 ↔ (-5 ≤ k ∧ k ≤ -2) :=
sorry

end max_value_f_zero_points_range_k_l6_6276


namespace sum_of_decimals_l6_6011

theorem sum_of_decimals :
  let a := 0.3
  let b := 0.08
  let c := 0.007
  a + b + c = 0.387 :=
by
  sorry

end sum_of_decimals_l6_6011


namespace circle_circumference_l6_6361

theorem circle_circumference (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi :=
by
  -- Use the given dimensions to find the diagonal (which is the diameter).
  -- Calculate the circumference using the calculated diameter.
  sorry

end circle_circumference_l6_6361


namespace option_A_is_quadratic_l6_6884

def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0

-- Given options
def option_A_equation (x : ℝ) : Prop :=
  x^2 - 2 = 0

def option_B_equation (x y : ℝ) : Prop :=
  x + 2 * y = 3

def option_C_equation (x : ℝ) : Prop :=
  x - 1/x = 1

def option_D_equation (x y : ℝ) : Prop :=
  x^2 + x = y + 1

-- Prove that option A is a quadratic equation
theorem option_A_is_quadratic (x : ℝ) : is_quadratic_equation 1 0 (-2) :=
by
  sorry

end option_A_is_quadratic_l6_6884


namespace dima_story_retelling_count_l6_6385

theorem dima_story_retelling_count :
  ∃ n, (26 * (2 ^ 5) * (3 ^ 4)) = 33696 ∧ n = 9 :=
by
  sorry

end dima_story_retelling_count_l6_6385


namespace vinegar_ratio_to_total_capacity_l6_6448

theorem vinegar_ratio_to_total_capacity (bowl_capacity : ℝ) (oil_fraction : ℝ) 
  (oil_density : ℝ) (vinegar_density : ℝ) (total_weight : ℝ) :
  bowl_capacity = 150 ∧ oil_fraction = 2/3 ∧ oil_density = 5 ∧ vinegar_density = 4 ∧ total_weight = 700 →
  (total_weight - (bowl_capacity * oil_fraction * oil_density)) / vinegar_density / bowl_capacity = 1/3 :=
by
  sorry

end vinegar_ratio_to_total_capacity_l6_6448


namespace graph_passes_quadrants_l6_6526

theorem graph_passes_quadrants (a b : ℝ) (h_a : 1 < a) (h_b : -1 < b ∧ b < 0) : 
    ∀ x : ℝ, (0 < a^x + b ∧ x > 0) ∨ (a^x + b < 0 ∧ x < 0) ∨ (0 < x ∧ a^x + b = 0) → x ≠ 0 ∧ 0 < x :=
sorry

end graph_passes_quadrants_l6_6526


namespace least_n_questions_l6_6123

theorem least_n_questions {n : ℕ} : 
  (1/2 : ℝ)^n < 1/10 → n ≥ 4 :=
by
  sorry

end least_n_questions_l6_6123


namespace minimum_common_perimeter_l6_6339

namespace IsoscelesTriangles

def integer_sided_isosceles_triangles (a b x : ℕ) :=
  2 * a + 10 * x = 2 * b + 8 * x ∧
  5 * Real.sqrt (a^2 - 25 * x^2) = 4 * Real.sqrt (b^2 - 16 * x^2) ∧
  5 * b = 4 * (b + x)

theorem minimum_common_perimeter : ∃ (a b x : ℕ), 
  integer_sided_isosceles_triangles a b x ∧
  2 * a + 10 * x = 192 :=
by
  sorry

end IsoscelesTriangles

end minimum_common_perimeter_l6_6339


namespace quadratic_real_root_exists_l6_6886

theorem quadratic_real_root_exists :
  ¬ (∃ x : ℝ, x^2 + 1 = 0) ∧
  ¬ (∃ x : ℝ, x^2 + x + 1 = 0) ∧
  ¬ (∃ x : ℝ, x^2 - x + 1 = 0) ∧
  (∃ x : ℝ, x^2 - x - 1 = 0) :=
by
  sorry

end quadratic_real_root_exists_l6_6886


namespace total_students_correct_l6_6098

def students_in_school : ℕ :=
  let students_per_class := 23
  let classes_per_grade := 12
  let grades_per_school := 3
  students_per_class * classes_per_grade * grades_per_school

theorem total_students_correct :
  students_in_school = 828 :=
by
  sorry

end total_students_correct_l6_6098


namespace option_A_option_B_option_C_option_D_l6_6002

theorem option_A : (-(-1) : ℤ) ≠ -|(-1 : ℤ)| := by
  sorry

theorem option_B : ((-3)^2 : ℤ) ≠ -(3^2 : ℤ) := by
  sorry

theorem option_C : ((-4)^3 : ℤ) = -(4^3 : ℤ) := by
  sorry

theorem option_D : ((2^2 : ℚ)/3) ≠ ((2/3)^2 : ℚ) := by
  sorry

end option_A_option_B_option_C_option_D_l6_6002


namespace solution_set_of_inequality_l6_6469

theorem solution_set_of_inequality :
  {x : ℝ | x * (x - 1) * (x - 2) > 0} = {x | (0 < x ∧ x < 1) ∨ x > 2} :=
by sorry

end solution_set_of_inequality_l6_6469


namespace factorial_mod_13_l6_6030

open Nat

theorem factorial_mod_13 :
  let n := 10
  let p := 13
  n! % p = 6 := by
sorry

end factorial_mod_13_l6_6030


namespace regular_polygon_sides_l6_6135

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l6_6135


namespace longer_side_length_l6_6371

theorem longer_side_length (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x * y = 221) : max x y = 17 :=
by
  sorry

end longer_side_length_l6_6371


namespace smallest_prime_p_l6_6498

-- Definitions based on the problem's conditions
def legendre_formula_vp (p n : ℕ) : ℕ :=
  if p = 0 then 0
  else ∑ i in finset.range (n.log p + 1), n / p^i

-- The main statement we need to prove
theorem smallest_prime_p (p : ℕ) (hp : nat.prime p) :
  (legendre_formula_vp p 2018 = 3) ↔ p = 509 :=
begin
  sorry
end

end smallest_prime_p_l6_6498


namespace nina_money_l6_6701

theorem nina_money (W : ℝ) (h1 : W > 0) (h2 : 10 * W = 14 * (W - 1)) : 10 * W = 35 := by
  sorry

end nina_money_l6_6701


namespace triangle_obtuse_l6_6827

theorem triangle_obtuse (α β γ : ℝ) 
  (h1 : α ≤ β) (h2 : β < γ) 
  (h3 : α + β + γ = 180) 
  (h4 : α + β < γ) : 
  γ > 90 :=
  sorry

end triangle_obtuse_l6_6827


namespace correct_statements_count_l6_6596

theorem correct_statements_count :
  (¬(1 = 1) ∧ ¬(1 = 0)) ∧
  (¬(1 = 11)) ∧
  ((1 - 2 + 1 / 2) = 3) ∧
  (2 = 2) →
  2 = ([false, false, true, true].count true) := 
sorry

end correct_statements_count_l6_6596


namespace no_positive_integer_n_exists_l6_6291

theorem no_positive_integer_n_exists :
  ¬ ∃ (n : ℕ), (n > 0) ∧ (∀ (r : ℚ), ∃ (b : ℤ) (a : Fin n → ℤ), (∀ i, a i ≠ 0) ∧ (r = b + (∑ i, (1 : ℚ) / a i))) :=
by
  -- Proof omitted
  sorry

end no_positive_integer_n_exists_l6_6291


namespace pythagorean_triple_correct_l6_6113

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_correct : 
  is_pythagorean_triple 9 12 15 ∧ ¬ is_pythagorean_triple 3 4 6 ∧ ¬ is_pythagorean_triple 1 2 3 ∧ ¬ is_pythagorean_triple 6 12 13 :=
by
  sorry

end pythagorean_triple_correct_l6_6113


namespace regular_polygon_sides_l6_6188

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6188


namespace regular_polygon_sides_l6_6153

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6153


namespace dimes_total_l6_6996

def initial_dimes : ℕ := 9
def added_dimes : ℕ := 7

theorem dimes_total : initial_dimes + added_dimes = 16 := by
  sorry

end dimes_total_l6_6996


namespace ratio_of_N_to_R_l6_6593

variables (N T R k : ℝ)

theorem ratio_of_N_to_R (h1 : T = (1 / 4) * N)
                        (h2 : R = 40)
                        (h3 : N = k * R)
                        (h4 : T + R + N = 190) :
    N / R = 3 :=
by
  sorry

end ratio_of_N_to_R_l6_6593


namespace age_of_youngest_boy_l6_6754

theorem age_of_youngest_boy (average_age : ℕ) (age_proportion : ℕ → ℕ) 
  (h1 : average_age = 120) 
  (h2 : ∀ x, age_proportion x = 2 * x ∨ age_proportion x = 6 * x ∨ age_proportion x = 8 * x)
  (total_age : ℕ) 
  (h3 : total_age = 3 * average_age) :
  ∃ x, age_proportion x = 2 * x ∧ 2 * x * (3 * average_age / total_age) = 45 :=
by {
  sorry
}

end age_of_youngest_boy_l6_6754


namespace exists_nat_not_in_geom_progressions_l6_6871

theorem exists_nat_not_in_geom_progressions
  (progressions : Fin 5 → ℕ → ℕ)
  (is_geometric : ∀ i : Fin 5, ∃ a q : ℕ, ∀ n : ℕ, progressions i n = a * q^n) :
  ∃ n : ℕ, ∀ i : Fin 5, ∀ m : ℕ, progressions i m ≠ n :=
by
  sorry

end exists_nat_not_in_geom_progressions_l6_6871


namespace solve_for_sum_l6_6995

theorem solve_for_sum (x y : ℝ) (h : x^2 + y^2 = 18 * x - 10 * y + 22) : x + y = 4 + 2 * Real.sqrt 42 :=
sorry

end solve_for_sum_l6_6995


namespace correct_calculated_value_l6_6883

theorem correct_calculated_value (x : ℤ) 
  (h : x / 16 = 8 ∧ x % 16 = 4) : (x * 16 + 8 = 2120) := by
  sorry

end correct_calculated_value_l6_6883


namespace problem_21_divisor_l6_6964

theorem problem_21_divisor 
    (k : ℕ) 
    (h1 : ∃ k, 21^k ∣ 435961) 
    (h2 : 21^k ∣ 435961) 
    : 7^k - k^7 = 1 := 
sorry

end problem_21_divisor_l6_6964


namespace ball_bounce_count_l6_6427

theorem ball_bounce_count :
  let A := (0 : ℚ, 0 : ℚ)
  let Y := (7 / 2 : ℚ, (3 * Real.sqrt 3) / 2 : ℚ)
  -- Conditions defining the ball's path and the reflection pattern
  let reflection_scheme (A Y : ℚ × ℚ) := ... -- Define the scheme based on the reflections
  -- Conclusion: number of bounces
  7 = reflection_scheme A Y :=
sorry

end ball_bounce_count_l6_6427


namespace second_point_x_coord_l6_6831

open Function

variable (n : ℝ)

def line_eq (y : ℝ) : ℝ := 2 * y + 5

theorem second_point_x_coord (h₁ : ∀ (x y : ℝ), x = line_eq y → True) :
  ∃ m : ℝ, ∀ n : ℝ, m = 2 * n + 5 → (m + 1 = line_eq (n + 0.5)) :=
by
  sorry

end second_point_x_coord_l6_6831


namespace value_of_polynomial_l6_6805

theorem value_of_polynomial (a b : ℝ) (h : a^2 - 2 * b - 1 = 0) : -2 * a^2 + 4 * b + 2025 = 2023 :=
by
  sorry

end value_of_polynomial_l6_6805


namespace graveling_cost_l6_6889

def lawn_length : ℝ := 110
def lawn_breadth: ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_meter : ℝ := 3

def road_1_area : ℝ := lawn_length * road_width
def intersecting_length : ℝ := lawn_breadth - road_width
def road_2_area : ℝ := intersecting_length * road_width
def total_area : ℝ := road_1_area + road_2_area
def total_cost : ℝ := total_area * cost_per_sq_meter

theorem graveling_cost :
  total_cost = 4800 := 
  by
    sorry

end graveling_cost_l6_6889


namespace inequality_solution_l6_6525

/-- Define conditions and state the corresponding theorem -/
theorem inequality_solution (a x : ℝ) (h : a < 0) : ax - 1 > 0 ↔ x < 1 / a :=
by sorry

end inequality_solution_l6_6525


namespace boat_distance_downstream_l6_6674

theorem boat_distance_downstream (v_s : ℝ) (h : 8 - v_s = 5) :
  8 + v_s = 11 :=
by
  sorry

end boat_distance_downstream_l6_6674


namespace xiaoming_wait_probability_l6_6597

-- Conditions
def green_light_duration : ℕ := 40
def red_light_duration : ℕ := 50
def total_light_cycle : ℕ := green_light_duration + red_light_duration
def waiting_time_threshold : ℕ := 20
def long_wait_interval : ℕ := 30 -- from problem (20 seconds to wait corresponds to 30 seconds interval)

-- Probability calculation
theorem xiaoming_wait_probability :
  ∀ (arrival_time : ℕ), arrival_time < total_light_cycle →
    (30 : ℝ) / (total_light_cycle : ℝ) = 1 / 3 := by sorry

end xiaoming_wait_probability_l6_6597


namespace fence_cost_l6_6595

noncomputable def price_per_foot (total_cost : ℝ) (perimeter : ℝ) : ℝ :=
  total_cost / perimeter

theorem fence_cost (area : ℝ) (total_cost : ℝ) (price : ℝ) :
  area = 289 → total_cost = 4012 → price = price_per_foot 4012 (4 * (Real.sqrt 289)) → price = 59 :=
by
  intros h_area h_cost h_price
  sorry

end fence_cost_l6_6595


namespace regular_polygon_sides_l6_6165

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6165


namespace mean_of_remaining_two_numbers_l6_6633

/-- 
Given seven numbers:
a = 1870, b = 1995, c = 2020, d = 2026, e = 2110, f = 2124, g = 2500
and the condition that the mean of five of these numbers is 2100,
prove that the mean of the remaining two numbers is 2072.5.
-/
theorem mean_of_remaining_two_numbers :
  let a := 1870
  let b := 1995
  let c := 2020
  let d := 2026
  let e := 2110
  let f := 2124
  let g := 2500
  a + b + c + d + e + f + g = 14645 →
  (a + b + c + d + e + f + g) = 14645 →
  (a + b + c + d + e) / 5 = 2100 →
  (f + g) / 2 = 2072.5 :=
by
  let a := 1870
  let b := 1995
  let c := 2020
  let d := 2026
  let e := 2110
  let f := 2124
  let g := 2500
  sorry

end mean_of_remaining_two_numbers_l6_6633


namespace distance_between_A_and_B_l6_6060

noncomputable def distance_between_points (v_A v_B : ℝ) (t_meet t_A_to_B_after_meet : ℝ) : ℝ :=
  let t_total_A := t_meet + t_A_to_B_after_meet
  let t_total_B := t_meet + (t_meet - t_A_to_B_after_meet)
  let D := v_A * t_total_A + v_B * t_total_B
  D

-- Given conditions
def t_meet : ℝ := 4
def t_A_to_B_after_meet : ℝ := 3
def speed_difference : ℝ := 20

-- Function to calculate speeds based on given conditions
noncomputable def calculate_speeds (v_B : ℝ) : ℝ × ℝ :=
  let v_A := v_B + speed_difference
  (v_A, v_B)

-- Statement of the problem in Lean 4
theorem distance_between_A_and_B : ∃ (v_B v_A : ℝ), 
  v_A = v_B + speed_difference ∧
  distance_between_points v_A v_B t_meet t_A_to_B_after_meet = 240 :=
by 
  sorry

end distance_between_A_and_B_l6_6060


namespace S_n_min_at_5_min_nS_n_is_neg_49_l6_6470

variable {S_n : ℕ → ℝ}
variable {a_1 d : ℝ}

-- Conditions
axiom sum_first_n_terms (n : ℕ) : S_n n = n / 2 * (2 * a_1 + (n - 1) * d)

axiom S_10 : S_n 10 = 0
axiom S_15 : S_n 15 = 25

-- Proving the following statements
theorem S_n_min_at_5 :
  (∀ n, S_n n ≥ S_n 5) :=
sorry

theorem min_nS_n_is_neg_49 :
  (∀ n, n * S_n n ≥ -49) :=
sorry

end S_n_min_at_5_min_nS_n_is_neg_49_l6_6470


namespace find_num_trumpet_players_l6_6543

namespace OprahWinfreyHighSchoolMarchingBand

def num_trumpet_players (total_weight : ℕ) 
  (num_clarinet : ℕ) (num_trombone : ℕ) 
  (num_tuba : ℕ) (num_drum : ℕ) : ℕ :=
(total_weight - 
  ((num_clarinet * 5) + 
  (num_trombone * 10) + 
  (num_tuba * 20) + 
  (num_drum * 15)))
  / 5

theorem find_num_trumpet_players :
  num_trumpet_players 245 9 8 3 2 = 6 :=
by
  -- calculation and reasoning steps would go here
  sorry

end OprahWinfreyHighSchoolMarchingBand

end find_num_trumpet_players_l6_6543


namespace power_greater_than_any_l6_6742

theorem power_greater_than_any {p M : ℝ} (hp : p > 0) (hM : M > 0) : ∃ n : ℕ, (1 + p)^n > M :=
by
  sorry

end power_greater_than_any_l6_6742


namespace max_value_npk_l6_6421

theorem max_value_npk : 
  ∃ (M K : ℕ), 
    (M ≠ K) ∧ (1 ≤ M ∧ M ≤ 9) ∧ (1 ≤ K ∧ K ≤ 9) ∧ 
    (NPK = 11 * M * K ∧ 100 ≤ NPK ∧ NPK < 1000 ∧ NPK = 891) :=
sorry

end max_value_npk_l6_6421


namespace cost_of_sculpture_cny_l6_6306

def exchange_rate_usd_to_nad := 8 -- 1 USD = 8 NAD
def exchange_rate_usd_to_cny := 5  -- 1 USD = 5 CNY
def cost_of_sculpture_nad := 160  -- Cost of sculpture in NAD

theorem cost_of_sculpture_cny : (cost_of_sculpture_nad / exchange_rate_usd_to_nad) * exchange_rate_usd_to_cny = 100 := by
  sorry

end cost_of_sculpture_cny_l6_6306


namespace percent_flamingos_among_non_parrots_l6_6915

theorem percent_flamingos_among_non_parrots
  (total_birds : ℝ) (flamingos : ℝ) (parrots : ℝ) (eagles : ℝ) (owls : ℝ)
  (h_total : total_birds = 100)
  (h_flamingos : flamingos = 40)
  (h_parrots : parrots = 20)
  (h_eagles : eagles = 15)
  (h_owls : owls = 25) :
  ((flamingos / (total_birds - parrots)) * 100 = 50) :=
by sorry

end percent_flamingos_among_non_parrots_l6_6915


namespace total_selection_methods_l6_6872

theorem total_selection_methods (synthetic_students : ℕ) (analytical_students : ℕ)
  (h_synthetic : synthetic_students = 5) (h_analytical : analytical_students = 3) :
  synthetic_students + analytical_students = 8 :=
by
  -- Proof is omitted
  sorry

end total_selection_methods_l6_6872


namespace regular_polygon_sides_l6_6152

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6152


namespace value_of_2x_l6_6061

theorem value_of_2x (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_eq : 2 * x = 6 * z) (h_sum : x + y + z = 26) : 2 * x = 6 := 
by
  sorry

end value_of_2x_l6_6061


namespace unique_solution_for_value_of_m_l6_6258

theorem unique_solution_for_value_of_m :
  ∃ m : ℝ, (∀ x : ℝ, (x+5)*(x+2) = m + 3*x) → m = 6 ∧ 
  (∀ a b c: ℝ, a = 1 ∧ b = 4 ∧ c = (10 - m) → b^2 - 4 * a * c = 0) := 
begin
  sorry
end

end unique_solution_for_value_of_m_l6_6258


namespace correct_mark_proof_l6_6905

-- Define the conditions
def wrong_mark := 85
def increase_in_average : ℝ := 0.5
def number_of_pupils : ℕ := 104

-- Define the correct mark to be proven
noncomputable def correct_mark : ℕ := 33

-- Statement to be proven
theorem correct_mark_proof (x : ℝ) :
  (wrong_mark - x) / number_of_pupils = increase_in_average → x = correct_mark :=
by
  sorry

end correct_mark_proof_l6_6905


namespace mod_pow_sub_eq_l6_6790

theorem mod_pow_sub_eq : 
  (45^1537 - 25^1537) % 8 = 4 := 
by
  have h1 : 45 % 8 = 5 := by norm_num
  have h2 : 25 % 8 = 1 := by norm_num
  sorry

end mod_pow_sub_eq_l6_6790


namespace tina_spent_on_books_l6_6106

theorem tina_spent_on_books : 
  ∀ (saved_in_june saved_in_july saved_in_august spend_on_books spend_on_shoes money_left : ℤ),
  saved_in_june = 27 →
  saved_in_july = 14 →
  saved_in_august = 21 →
  spend_on_shoes = 17 →
  money_left = 40 →
  (saved_in_june + saved_in_july + saved_in_august) - spend_on_books - spend_on_shoes = money_left →
  spend_on_books = 5 :=
by
  intros saved_in_june saved_in_july saved_in_august spend_on_books spend_on_shoes money_left
  intros h_june h_july h_august h_shoes h_money_left h_eq
  sorry

end tina_spent_on_books_l6_6106


namespace mary_flour_l6_6082

-- Defining the conditions
def total_flour : ℕ := 11
def total_sugar : ℕ := 7
def flour_difference : ℕ := 2

-- The problem we want to prove
theorem mary_flour (F : ℕ) (C : ℕ) (S : ℕ)
  (h1 : C + 2 = S)
  (h2 : total_flour = F + C)
  (h3 : S = total_sugar) :
  F = 2 :=
by
  sorry

end mary_flour_l6_6082


namespace cubic_poly_l6_6505

noncomputable def q (x : ℝ) : ℝ := - (2 / 3) * x ^ 3 + 2 * x ^ 2 - (8 / 3) * x - (16 / 3)

theorem cubic_poly:
  ( ∃ (a b c d : ℝ), 
    (∀ x : ℝ, q x = a * x ^ 3 + b * x ^ 2 + c * x + d)
    ∧ q 1 = -6
    ∧ q 2 = -8
    ∧ q 3 = -14
    ∧ q 4 = -28
  ) → 
  q x = - (2 / 3) * x ^ 3 + 2 * x ^ 2 - (8 / 3) * x - (16 / 3) := 
sorry

end cubic_poly_l6_6505


namespace symmetric_circle_eq_l6_6412

open Real

-- Define the original circle equation and the line of symmetry
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def line_of_symmetry (x y : ℝ) : Prop := y = -x

-- Define the symmetry transformation with respect to the line y = -x
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

-- Define the new circle that is symmetric to the original circle with respect to y = -x
def new_circle (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

-- The theorem to be proven
theorem symmetric_circle_eq :
  ∀ x y : ℝ, original_circle (-y) (-x) ↔ new_circle x y := 
by
  sorry

end symmetric_circle_eq_l6_6412


namespace exists_hexagon_in_square_l6_6086

structure Point (α : Type*) :=
(x : α)
(y : α)

def is_in_square (p : Point ℕ) : Prop :=
p.x ≤ 4 ∧ p.y ≤ 4

def area_of_hexagon (vertices : List (Point ℕ)) : ℝ :=
-- placeholder for actual area calculation of a hexagon
sorry

theorem exists_hexagon_in_square : ∃ (p1 p2 : Point ℕ), 
  is_in_square p1 ∧ is_in_square p2 ∧ 
  area_of_hexagon [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 0⟩, ⟨4, 4⟩, p1, p2] = 6 :=
sorry

end exists_hexagon_in_square_l6_6086


namespace biscuits_per_guest_correct_l6_6979

def flour_per_batch : ℚ := 5 / 4
def biscuits_per_batch : ℕ := 9
def flour_needed : ℚ := 5
def guests : ℕ := 18

theorem biscuits_per_guest_correct :
  (flour_needed * biscuits_per_batch / flour_per_batch) / guests = 2 := by
  sorry

end biscuits_per_guest_correct_l6_6979


namespace trish_walks_l6_6873

variable (n : ℕ) (M D : ℝ)
variable (d : ℕ → ℝ)
variable (H1 : d 1 = 1)
variable (H2 : ∀ k : ℕ, d (k + 1) = 2 * d k)
variable (H3 : d n > M)

theorem trish_walks (n : ℕ) (M : ℝ) (H1 : d 1 = 1) (H2 : ∀ k : ℕ, d (k + 1) = 2 * d k) (H3 : d n > M) : 2^(n-1) > M := by
  sorry

end trish_walks_l6_6873


namespace product_of_a_values_has_three_solutions_eq_20_l6_6260

noncomputable def f (x : ℝ) : ℝ := abs ((x^2 - 10 * x + 25) / (x - 5) - (x^2 - 3 * x) / (3 - x))

def has_three_solutions (a : ℝ) : Prop :=
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ abs (abs (f x1) - 5) = a ∧ abs (abs (f x2) - 5) = a ∧ abs (abs (f x3) - 5) = a)

theorem product_of_a_values_has_three_solutions_eq_20 :
  ∃ a1 a2 : ℝ, has_three_solutions a1 ∧ has_three_solutions a2 ∧ a1 * a2 = 20 :=
sorry

end product_of_a_values_has_three_solutions_eq_20_l6_6260


namespace sculpture_cost_in_CNY_l6_6304

theorem sculpture_cost_in_CNY (USD_to_NAD USD_to_CNY cost_NAD : ℝ) :
  USD_to_NAD = 8 → USD_to_CNY = 5 → cost_NAD = 160 → (cost_NAD * (1 / USD_to_NAD) * USD_to_CNY) = 100 :=
by
  intros h1 h2 h3
  sorry

end sculpture_cost_in_CNY_l6_6304


namespace fg_of_3_eq_29_l6_6665

def g (x : ℝ) : ℝ := x^2
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_29 : f (g 3) = 29 := by
  sorry

end fg_of_3_eq_29_l6_6665


namespace stewart_farm_horse_food_l6_6378

def sheep_to_horse_ratio := 3 / 7
def horses_needed (sheep : ℕ) := (sheep * 7) / 3 
def daily_food_per_horse := 230
def sheep_count := 24
def total_horses := horses_needed sheep_count
def total_daily_horse_food := total_horses * daily_food_per_horse

theorem stewart_farm_horse_food : total_daily_horse_food = 12880 := by
  have num_horses : horses_needed 24 = 56 := by
    unfold horses_needed
    sorry -- Omitted for brevity, this would be solved

  have food_needed : 56 * 230 = 12880 := by
    sorry -- Omitted for brevity, this would be solved

  exact food_needed

end stewart_farm_horse_food_l6_6378


namespace polynomial_not_factorizable_l6_6980

theorem polynomial_not_factorizable
  (n m : ℕ)
  (hnm : n > m)
  (hm1 : m > 1)
  (hn_odd : n % 2 = 1)
  (hm_odd : m % 2 = 1) :
  ¬ ∃ (g h : Polynomial ℤ), g.degree > 0 ∧ h.degree > 0 ∧ (x^n + x^m + x + 1 = g * h) :=
by
  sorry

end polynomial_not_factorizable_l6_6980


namespace unique_solution_range_a_l6_6273

theorem unique_solution_range_a :
  (∀ (t : ℝ) (ht : 1 ≤ t ∧ t ≤ 3), ∃! (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1), x^2 * real.exp x + t - a = 0) →
  (∃ a : ℝ, ∀ t : ℝ, 1 ≤ t ∧ t ≤ 3 → ∃! x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ (a = x^2 * real.exp x + t)) →
  (a > 1 / real.exp 1 + 3 ∧ a ≤ real.exp 1 + 1) :=
sorry

end unique_solution_range_a_l6_6273


namespace original_fraction_is_one_third_l6_6711

theorem original_fraction_is_one_third (a b : ℕ) 
  (coprime_ab : Nat.gcd a b = 1) 
  (h : (a + 2) * b = 3 * a * b^2) : 
  (a = 1 ∧ b = 3) := 
by 
  sorry

end original_fraction_is_one_third_l6_6711


namespace quadratic_eq_roots_minus5_and_7_l6_6533

theorem quadratic_eq_roots_minus5_and_7 : ∀ x : ℝ, (x + 5) * (x - 7) = 0 ↔ x = -5 ∨ x = 7 := by
  sorry

end quadratic_eq_roots_minus5_and_7_l6_6533


namespace total_apples_collected_l6_6210

variable (dailyPicks : ℕ) (days : ℕ) (remainingPicks : ℕ)

theorem total_apples_collected (h1 : dailyPicks = 4) (h2 : days = 30) (h3 : remainingPicks = 230) :
  dailyPicks * days + remainingPicks = 350 :=
by
  sorry

end total_apples_collected_l6_6210


namespace fish_count_together_l6_6572

namespace FishProblem

def JerkTunaFish : ℕ := 144
def TallTunaFish : ℕ := 2 * JerkTunaFish
def SwellTunaFish : ℕ := TallTunaFish + (TallTunaFish / 2)
def totalFish : ℕ := JerkTunaFish + TallTunaFish + SwellTunaFish

theorem fish_count_together : totalFish = 864 := by
  sorry

end FishProblem

end fish_count_together_l6_6572


namespace cat_mouse_positions_after_247_moves_l6_6830

-- Definitions for Positions:
inductive Position
| TopLeft
| TopRight
| BottomRight
| BottomLeft
| TopMiddle
| RightMiddle
| BottomMiddle
| LeftMiddle

open Position

-- Function to calculate position of the cat
def cat_position (n : ℕ) : Position :=
  match n % 4 with
  | 0 => TopLeft
  | 1 => TopRight
  | 2 => BottomRight
  | 3 => BottomLeft
  | _ => TopLeft   -- This case is impossible since n % 4 is in {0, 1, 2, 3}

-- Function to calculate position of the mouse
def mouse_position (n : ℕ) : Position :=
  match n % 8 with
  | 0 => TopMiddle
  | 1 => TopRight
  | 2 => RightMiddle
  | 3 => BottomRight
  | 4 => BottomMiddle
  | 5 => BottomLeft
  | 6 => LeftMiddle
  | 7 => TopLeft
  | _ => TopMiddle -- This case is impossible since n % 8 is in {0, 1, .., 7}

-- Target theorem
theorem cat_mouse_positions_after_247_moves :
  cat_position 247 = BottomRight ∧ mouse_position 247 = LeftMiddle :=
by
  sorry

end cat_mouse_positions_after_247_moves_l6_6830


namespace increase_in_area_l6_6967

theorem increase_in_area (a : ℝ) : 
  let original_radius := 3
  let new_radius := original_radius + a
  let original_area := π * original_radius ^ 2
  let new_area := π * new_radius ^ 2
  new_area - original_area = π * (3 + a) ^ 2 - 9 * π := 
by
  sorry

end increase_in_area_l6_6967


namespace circumference_of_inscribed_circle_l6_6364

-- Define the dimensions of the rectangle
def width : ℝ := 9
def height : ℝ := 12

-- Define the function to compute the diagonal of the rectangle
def diagonal (w h : ℝ) : ℝ := Real.sqrt (w ^ 2 + h ^ 2)

-- Define the function to compute the circumference of the circle given its diameter
def circumference (d : ℝ) : ℝ := Real.pi * d

-- State the theorem
theorem circumference_of_inscribed_circle :
  circumference (diagonal width height) = 15 * Real.pi := by
  sorry

end circumference_of_inscribed_circle_l6_6364


namespace polynomials_symmetric_l6_6908

noncomputable def P : ℕ → (ℝ → ℝ → ℝ → ℝ)
  | 0       => λ x y z => 1
  | (m + 1) => λ x y z => (x + z) * (y + z) * (P m x y (z + 1)) - z^2 * (P m x y z)

theorem polynomials_symmetric (m : ℕ) (x y z : ℝ) : 
  P m x y z = P m y x z ∧ P m x y z = P m x z y := 
sorry

end polynomials_symmetric_l6_6908


namespace flower_bed_profit_l6_6121

theorem flower_bed_profit (x : ℤ) :
  (3 + x) * (10 - x) = 40 :=
sorry

end flower_bed_profit_l6_6121


namespace minimum_m_minus_n_l6_6446

theorem minimum_m_minus_n (m n : ℕ) (hm : m > n) (h : (9^m) % 100 = (9^n) % 100) : m - n = 10 := 
sorry

end minimum_m_minus_n_l6_6446


namespace Ryan_bike_time_l6_6107

-- Definitions of the conditions
variables (B : ℕ)

-- Conditions
def bike_time := B
def bus_time := B + 10
def friend_time := B / 3
def commuting_time := bike_time B + 3 * bus_time B + friend_time B = 160

-- Goal to prove
theorem Ryan_bike_time : commuting_time B → B = 30 :=
by
  intro h
  sorry

end Ryan_bike_time_l6_6107


namespace find_x_l6_6675

theorem find_x (x : ℝ) (h : 6 * x + 7 * x + 3 * x + 2 * x + 4 * x = 360) : 
  x = 180 / 11 := 
by
  sorry

end find_x_l6_6675


namespace find_number_l6_6755

noncomputable def number := 115.2 / 0.32

theorem find_number : number = 360 := 
by
  sorry

end find_number_l6_6755


namespace milk_production_l6_6310

-- Variables representing the problem parameters
variables {a b c f d e g : ℝ}

-- Preconditions
axiom pos_a : a > 0
axiom pos_c : c > 0
axiom pos_f : f > 0
axiom pos_d : d > 0
axiom pos_e : e > 0
axiom pos_g : g > 0

theorem milk_production (a b c f d e g : ℝ) (h_a : a > 0) (h_c : c > 0) (h_f : f > 0) (h_d : d > 0) (h_e : e > 0) (h_g : g > 0) :
  d * e * g * (b / (a * c * f)) = (b * d * e * g) / (a * c * f) := by
  sorry

end milk_production_l6_6310


namespace infinite_sum_fraction_equals_quarter_l6_6627

theorem infinite_sum_fraction_equals_quarter :
  (∑' n : ℕ, (3 ^ n) / (1 + 3 ^ n + 3 ^ (n + 1) + 3 ^ (2 * n + 1))) = 1 / 4 :=
by
  -- With the given conditions, we need to prove the above statement
  -- The conditions have been used to express the problem in Lean
  sorry

end infinite_sum_fraction_equals_quarter_l6_6627


namespace anne_cleans_in_12_hours_l6_6784

-- Define the rates of Bruce and Anne
variables (B A : ℝ)

-- Define the conditions of the problem
constants (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1)

theorem anne_cleans_in_12_hours (B A : ℝ) (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1) : 1 / A = 12 :=
  sorry

end anne_cleans_in_12_hours_l6_6784


namespace remainder_of_division_l6_6746

-- Define the dividend and divisor
def dividend : ℕ := 3^303 + 303
def divisor : ℕ := 3^101 + 3^51 + 1

-- State the theorem to be proven
theorem remainder_of_division:
  (dividend % divisor) = 303 := by
  sorry

end remainder_of_division_l6_6746


namespace eval_expression_l6_6923

theorem eval_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ x) (hz' : z ≠ -x) :
  ((x / (x + z) + z / (x - z)) / (z / (x + z) - x / (x - z)) = -1) :=
by
  sorry

end eval_expression_l6_6923


namespace probability_of_same_color_pairs_left_right_l6_6856

-- Define the counts of different pairs
def total_pairs := 15
def black_pairs := 8
def red_pairs := 4
def white_pairs := 3

-- Define the total number of shoes
def total_shoes := 30

-- Define the total ways to choose any 2 shoes out of total_shoes
def total_ways := Nat.choose total_shoes 2

-- Define the ways to choose one left and one right for each color
def black_ways := black_pairs * black_pairs
def red_ways := red_pairs * red_pairs
def white_ways := white_pairs * white_pairs

-- Define the total favorable outcomes for same color pairs
def total_favorable := black_ways + red_ways + white_ways

-- Define the probability
def probability := (total_favorable, total_ways)

-- Statement to prove
theorem probability_of_same_color_pairs_left_right :
  probability = (89, 435) :=
by
  sorry

end probability_of_same_color_pairs_left_right_l6_6856


namespace probability_point_in_sphere_l6_6765

noncomputable def volume_of_cube : ℝ := 8
noncomputable def volume_of_sphere : ℝ := (4 * Real.pi) / 3
noncomputable def probability : ℝ := volume_of_sphere / volume_of_cube

theorem probability_point_in_sphere (x y z : ℝ) (h₁ : -1 ≤ x ∧ x ≤ 1) (h₂ : -1 ≤ y ∧ y ≤ 1) (h₃ : -1 ≤ z ∧ z ≤ 1) :
    probability = Real.pi / 6 :=
by
  sorry

end probability_point_in_sphere_l6_6765


namespace average_weight_of_Arun_l6_6351

def Arun_weight_opinion (w : ℝ) : Prop :=
  (66 < w) ∧ (w < 72)

def Brother_weight_opinion (w : ℝ) : Prop :=
  (60 < w) ∧ (w < 70)

def Mother_weight_opinion (w : ℝ) : Prop :=
  w ≤ 69

def Father_weight_opinion (w : ℝ) : Prop :=
  (65 ≤ w) ∧ (w ≤ 71)

def Sister_weight_opinion (w : ℝ) : Prop :=
  (62 < w) ∧ (w ≤ 68)

def All_opinions (w : ℝ) : Prop :=
  Arun_weight_opinion w ∧
  Brother_weight_opinion w ∧
  Mother_weight_opinion w ∧
  Father_weight_opinion w ∧
  Sister_weight_opinion w

theorem average_weight_of_Arun : ∃ avg : ℝ, avg = 67.5 ∧ (∀ w, All_opinions w → (w = 67 ∨ w = 68)) :=
by
  sorry

end average_weight_of_Arun_l6_6351


namespace correct_sunset_time_proof_l6_6703

def Time := ℕ × ℕ  -- hours and minutes

def sunrise_time : Time := (7, 12)  -- 7:12 AM
def incorrect_daylight_duration : Time := (11, 15)  -- 11 hours 15 minutes as per newspaper

def add_time (t1 t2 : Time) : Time :=
  let (h1, m1) := t1
  let (h2, m2) := t2
  let minutes := m1 + m2
  let hours := h1 + h2 + minutes / 60
  (hours % 24, minutes % 60)

def correct_sunset_time : Time := (18, 27)  -- 18:27 in 24-hour format equivalent to 6:27 PM in 12-hour format

theorem correct_sunset_time_proof :
  add_time sunrise_time incorrect_daylight_duration = correct_sunset_time :=
by
  -- skipping the detailed proof for now
  sorry

end correct_sunset_time_proof_l6_6703


namespace log_exp_identity_l6_6661

theorem log_exp_identity (a : ℝ) (h : a = Real.log 5 / Real.log 4) : 
  (2^a + 2^(-a) = 6 * Real.sqrt 5 / 5) :=
by {
  -- a = log_4 (5) can be rewritten using change-of-base formula: log 5 / log 4
  -- so, it can be used directly in the theorem
  sorry
}

end log_exp_identity_l6_6661


namespace fraction_addition_l6_6876

theorem fraction_addition :
  (3 / 4) / (5 / 8) + (1 / 2) = 17 / 10 :=
by
  sorry

end fraction_addition_l6_6876


namespace regular_polygon_num_sides_l6_6158

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l6_6158


namespace fg_of_3_eq_29_l6_6664

def g (x : ℕ) : ℕ := x * x
def f (x : ℕ) : ℕ := 3 * x + 2

theorem fg_of_3_eq_29 : f (g 3) = 29 :=
by
  sorry

end fg_of_3_eq_29_l6_6664


namespace logan_gas_expense_l6_6692

-- Definitions based on conditions:
def annual_salary := 65000
def rent_expense := 20000
def grocery_expense := 5000
def desired_savings := 42000
def new_income_target := annual_salary + 10000

-- The property to be proved:
theorem logan_gas_expense : 
  ∀ (gas_expense : ℕ), 
  new_income_target - desired_savings = rent_expense + grocery_expense + gas_expense → 
  gas_expense = 8000 := 
by 
  sorry

end logan_gas_expense_l6_6692


namespace ellipse_foci_condition_l6_6410

theorem ellipse_foci_condition {m : ℝ} :
  (1 < m ∧ m < 2) ↔ (∃ (x y : ℝ), (x^2 / (m - 1) + y^2 / (3 - m) = 1) ∧ (3 - m > m - 1) ∧ (m - 1 > 0) ∧ (3 - m > 0)) :=
by
  sorry

end ellipse_foci_condition_l6_6410


namespace quadratic_roots_distinct_l6_6725

theorem quadratic_roots_distinct (m : ℝ) :
  let Δ := m^2 + 32 in
  Δ > 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 - 8 = 0 ∧ x2^2 + m * x2 - 8 = 0) :=
begin
  sorry
end

end quadratic_roots_distinct_l6_6725


namespace ratio_of_students_l6_6560

-- Define the total number of students
def total_students : ℕ := 24

-- Define the number of students in the chess program
def students_in_chess_program : ℕ := total_students / 3

-- Define the number of students going to the tournament
def students_going_to_tournament : ℕ := 4

-- State the proposition to be proved: The ratio of students going to the tournament to the chess program is 1:2
theorem ratio_of_students :
  (students_going_to_tournament : ℚ) / (students_in_chess_program : ℚ) = 1 / 2 :=
by
  sorry

end ratio_of_students_l6_6560


namespace milk_exchange_l6_6461

theorem milk_exchange (initial_empty_bottles : ℕ) (exchange_rate : ℕ) (start_full_bottles : ℕ) : initial_empty_bottles = 43 → exchange_rate = 4 → start_full_bottles = 0 → ∃ liters_of_milk : ℕ, liters_of_milk = 14 :=
by
  intro h1 h2 h3
  sorry

end milk_exchange_l6_6461


namespace people_per_apartment_l6_6118

/-- A 25 story building has 4 apartments on each floor. 
There are 200 people in the building. 
Prove that each apartment houses 2 people. -/
theorem people_per_apartment (stories : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ)
    (h_stories : stories = 25)
    (h_apartments_per_floor : apartments_per_floor = 4)
    (h_total_people : total_people = 200) :
  (total_people / (stories * apartments_per_floor)) = 2 :=
by
  sorry

end people_per_apartment_l6_6118


namespace probability_two_red_shoes_l6_6332

theorem probability_two_red_shoes :
  let total_shoes := 10 in
  let red_shoes := 4 in
  let total_drawings := nat.choose total_shoes 2 in
  let red_drawings := nat.choose red_shoes 2 in
  (red_drawings : ℚ) / total_drawings = 2 / 15 := 
by
  let total_shoes := 10
  let red_shoes := 4
  let total_drawings := nat.choose total_shoes 2
  let red_drawings := nat.choose red_shoes 2
  show (red_drawings : ℚ) / total_drawings = 2 / 15
  sorry

end probability_two_red_shoes_l6_6332


namespace cubic_polynomial_sum_l6_6685

noncomputable def Q (a b c m x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2 * m

theorem cubic_polynomial_sum (a b c m : ℝ) :
  Q a b c m 0 = 2 * m ∧ Q a b c m 1 = 3 * m ∧ Q a b c m (-1) = 5 * m →
  Q a b c m 2 + Q a b c m (-2) = 20 * m :=
by
  intro h
  sorry

end cubic_polynomial_sum_l6_6685


namespace measure_four_liters_impossible_l6_6974

theorem measure_four_liters_impossible (a b c : ℕ) (h1 : a = 12) (h2 : b = 9) (h3 : c = 4) :
  ¬ ∃ x y : ℕ, x * a + y * b = c := 
by
  sorry

end measure_four_liters_impossible_l6_6974


namespace equation_of_line_through_point_with_equal_intercepts_l6_6023

open LinearAlgebra

theorem equation_of_line_through_point_with_equal_intercepts :
  ∃ (a b c : ℝ), (a * 1 + b * 2 + c = 0) ∧ (a * b < 0) ∧ ∀ x y : ℝ, 
  (a * x + b * y + c = 0 ↔ (2 * x - y = 0 ∨ x + y - 3 = 0)) :=
sorry

end equation_of_line_through_point_with_equal_intercepts_l6_6023


namespace average_sales_l6_6897

/-- The sales for the first five months -/
def sales_first_five_months := [5435, 5927, 5855, 6230, 5562]

/-- The sale for the sixth month -/
def sale_sixth_month := 3991

/-- The correct average sale to be achieved -/
def correct_average_sale := 5500

theorem average_sales :
  (sales_first_five_months.sum + sale_sixth_month) / 6 = correct_average_sale :=
by
  sorry

end average_sales_l6_6897


namespace binary_representation_of_14_l6_6927

theorem binary_representation_of_14 : nat.binary_repr 14 = "1110" :=
sorry

end binary_representation_of_14_l6_6927


namespace regular_polygon_sides_l6_6181

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6181


namespace combined_gravitational_force_l6_6577

theorem combined_gravitational_force 
    (d_E_surface : ℝ) (f_E_surface : ℝ) (d_M_surface : ℝ) (f_M_surface : ℝ) 
    (d_E_new : ℝ) (d_M_new : ℝ) 
    (k_E : ℝ) (k_M : ℝ) 
    (h1 : k_E = f_E_surface * d_E_surface^2)
    (h2 : k_M = f_M_surface * d_M_surface^2)
    (h3 : f_E_new = k_E / d_E_new^2)
    (h4 : f_M_new = k_M / d_M_new^2) : 
  f_E_new + f_M_new = 755.7696 :=
by
  sorry

end combined_gravitational_force_l6_6577


namespace necessary_and_sufficient_condition_l6_6956

variable (m n : ℕ)
def positive_integers (m n : ℕ) := m > 0 ∧ n > 0
def at_least_one_is_1 (m n : ℕ) : Prop := m = 1 ∨ n = 1
def sum_gt_product (m n : ℕ) : Prop := m + n > m * n

theorem necessary_and_sufficient_condition (h : positive_integers m n) : 
  sum_gt_product m n ↔ at_least_one_is_1 m n :=
by sorry

end necessary_and_sufficient_condition_l6_6956


namespace chess_team_girls_count_l6_6907

theorem chess_team_girls_count (B G : ℕ) 
  (h1 : B + G = 26) 
  (h2 : (3 / 4 : ℝ) * B + (1 / 4 : ℝ) * G = 13) : G = 13 := 
sorry

end chess_team_girls_count_l6_6907


namespace RS_plus_ST_l6_6092

theorem RS_plus_ST {a b c d e : ℕ} 
  (h1 : a = 68) 
  (h2 : b = 10) 
  (h3 : c = 7) 
  (h4 : d = 6) 
  : e = 3 :=
sorry

end RS_plus_ST_l6_6092


namespace find_polynomial_coefficients_l6_6933

-- Define the quadratic polynomial q(x) = ax^2 + bx + c
def polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions for polynomial
axiom condition1 (a b c : ℝ) : polynomial a b c (-2) = 9
axiom condition2 (a b c : ℝ) : polynomial a b c 1 = 2
axiom condition3 (a b c : ℝ) : polynomial a b c 3 = 10

-- Conjecture for the polynomial q(x)
theorem find_polynomial_coefficients : 
  ∃ (a b c : ℝ), 
    polynomial a b c (-2) = 9 ∧
    polynomial a b c 1 = 2 ∧
    polynomial a b c 3 = 10 ∧
    a = 19 / 15 ∧
    b = -2 / 15 ∧
    c = 13 / 15 :=
by {
  -- Placeholder proof
  sorry
}

end find_polynomial_coefficients_l6_6933


namespace bees_lost_each_day_l6_6355

theorem bees_lost_each_day
    (initial_bees : ℕ)
    (daily_hatch : ℕ)
    (days : ℕ)
    (total_bees_after_days : ℕ)
    (bees_lost_each_day : ℕ) :
    initial_bees = 12500 →
    daily_hatch = 3000 →
    days = 7 →
    total_bees_after_days = 27201 →
    (initial_bees + days * (daily_hatch - bees_lost_each_day) = total_bees_after_days) →
    bees_lost_each_day = 899 :=
by
  intros h_initial h_hatch h_days h_total h_eq
  sorry

end bees_lost_each_day_l6_6355


namespace initial_height_after_10_seconds_l6_6215

open Nat

def distance_fallen_in_nth_second (n : ℕ) : ℕ := 10 * n - 5

def total_distance_fallen (n : ℕ) : ℕ :=
  (n * (distance_fallen_in_nth_second 1 + distance_fallen_in_nth_second n)) / 2

theorem initial_height_after_10_seconds : 
  total_distance_fallen 10 = 500 := 
by
  sorry

end initial_height_after_10_seconds_l6_6215


namespace binary_addition_l6_6570

theorem binary_addition (M : ℕ) (hM : M = 0b101110) :
  let M_plus_five := M + 5 
  let M_plus_five_binary := 0b110011
  let M_plus_five_predecessor := 0b110010
  M_plus_five = M_plus_five_binary ∧ M_plus_five - 1 = M_plus_five_predecessor :=
by
  sorry

end binary_addition_l6_6570


namespace remaining_soup_feeds_adults_l6_6367

theorem remaining_soup_feeds_adults :
  (∀ (cans : ℕ), cans ≥ 8 ∧ cans / 6 ≥ 24) → (∃ (adults : ℕ), adults = 16) :=
by
  sorry

end remaining_soup_feeds_adults_l6_6367


namespace time_spent_watching_movies_l6_6616

def total_flight_time_minutes : ℕ := 11 * 60 + 20
def time_reading_minutes : ℕ := 2 * 60
def time_eating_dinner_minutes : ℕ := 30
def time_listening_radio_minutes : ℕ := 40
def time_playing_games_minutes : ℕ := 1 * 60 + 10
def time_nap_minutes : ℕ := 3 * 60

theorem time_spent_watching_movies :
  total_flight_time_minutes
  - time_reading_minutes
  - time_eating_dinner_minutes
  - time_listening_radio_minutes
  - time_playing_games_minutes
  - time_nap_minutes = 4 * 60 := by
  sorry

end time_spent_watching_movies_l6_6616


namespace cosx_cos2x_not_rational_l6_6012

noncomputable def cosx_sqrt2_rational (x : ℝ) : Prop :=
  (∃ (a : ℚ), cos x + Real.sqrt 2 = a)

noncomputable def cos2x_sqrt2_rational (x : ℝ) : Prop :=
  (∃ (b : ℚ), cos (2*x) + Real.sqrt 2 = b)

theorem cosx_cos2x_not_rational : ∀ (x : ℝ), ¬(cosx_sqrt2_rational x ∧ cos2x_sqrt2_rational x) :=
by
  sorry

end cosx_cos2x_not_rational_l6_6012


namespace anne_cleaning_time_l6_6783

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l6_6783


namespace mean_total_sample_variance_total_sample_expected_final_score_l6_6327

section SeagrassStatistics

variables (m n : ℕ) (mean_x mean_y: ℝ) (var_x var_y: ℝ) (A_win_A B_win_A : ℝ)

-- Assumptions from the conditions
variable (hp1 : m = 12)
variable (hp2 : mean_x = 18)
variable (hp3 : var_x = 19)
variable (hp4 : n = 18)
variable (hp5 : mean_y = 36)
variable (hp6 : var_y = 70)
variable (hp7 : A_win_A = 3 / 5)
variable (hp8 : B_win_A = 1 / 2)

-- Statements to prove
theorem mean_total_sample (m n : ℕ) (mean_x mean_y : ℝ) : 
  m * mean_x + n * mean_y = (m + n) * 28.8 := sorry

theorem variance_total_sample (m n : ℕ) (mean_x mean_y var_x var_y : ℝ) :
  m * (var_x + (mean_x - 28.8)^2) + n * (var_y + (mean_y - 28.8)^2) = (m + n) * 127.36 := sorry

theorem expected_final_score (A_win_A B_win_A : ℝ) :
  2 * ((6/25) * 1 + (15/25) * 2 + (4/25) * 0) = 36 / 25 := sorry

end SeagrassStatistics

end mean_total_sample_variance_total_sample_expected_final_score_l6_6327


namespace anne_cleaning_time_l6_6787

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l6_6787


namespace proper_subset_A_B_l6_6440

theorem proper_subset_A_B (a : ℝ) : 
  (∀ x, 1 < x ∧ x < 2 → x < a) ∧ (∃ b, b < a ∧ ¬(1 < b ∧ b < 2)) ↔ 2 ≤ a :=
by
  sorry

end proper_subset_A_B_l6_6440


namespace increasing_intervals_f_value_g_pi_over_6_l6_6264

def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin (π - x) * sin x - (sin x - cos x) ^ 2

theorem increasing_intervals_f :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) →
  2 * sqrt 3 * cos x - 2 * cos (2 * x) > 0 := sorry

def g (x : ℝ) : ℝ := f (2 * (x + π / 3))

theorem value_g_pi_over_6 : g (π / 6) = 1 := sorry

end increasing_intervals_f_value_g_pi_over_6_l6_6264


namespace regular_polygon_sides_l6_6128

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l6_6128


namespace fg_of_3_eq_29_l6_6666

def g (x : ℝ) : ℝ := x^2
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_29 : f (g 3) = 29 := by
  sorry

end fg_of_3_eq_29_l6_6666


namespace stratified_sampling_B_l6_6532

-- Define the groups and their sizes
def num_people_A : ℕ := 18
def num_people_B : ℕ := 24
def num_people_C : ℕ := 30

-- Total number of people
def total_people : ℕ := num_people_A + num_people_B + num_people_C

-- Total sample size to be drawn
def sample_size : ℕ := 12

-- Proportion of group B
def proportion_B : ℚ := num_people_B / total_people

-- Number of people to be drawn from group B
def number_drawn_from_B : ℚ := sample_size * proportion_B

-- The theorem to be proved
theorem stratified_sampling_B : number_drawn_from_B = 4 := 
by
  -- This is where the proof would go
  sorry

end stratified_sampling_B_l6_6532


namespace ratio_of_areas_of_circles_l6_6281

theorem ratio_of_areas_of_circles (C_A C_B C_C : ℝ) (h1 : (60 / 360) * C_A = (40 / 360) * C_B) (h2 : (30 / 360) * C_B = (90 / 360) * C_C) : 
  (C_A / (2 * Real.pi))^2 / (C_C / (2 * Real.pi))^2 = 2 :=
by
  sorry

end ratio_of_areas_of_circles_l6_6281


namespace triangle_angles_l6_6467

theorem triangle_angles (r_a r_b r_c R : ℝ) (h1 : r_a + r_b = 3 * R) (h2 : r_b + r_c = 2 * R) :
  ∃ (α β γ : ℝ), α = 90 ∧ γ = 60 ∧ β = 30 :=
by
  sorry

end triangle_angles_l6_6467


namespace regular_polygon_sides_l6_6186

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6186


namespace regular_polygon_sides_l6_6139

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6139


namespace mila_father_total_pay_l6_6847

def first_job_pay : ℤ := 2125
def pay_difference : ℤ := 375
def second_job_pay : ℤ := first_job_pay - pay_difference
def total_pay : ℤ := first_job_pay + second_job_pay

theorem mila_father_total_pay :
  total_pay = 3875 := by
  sorry

end mila_father_total_pay_l6_6847


namespace sqrt_of_1024_is_32_l6_6530

theorem sqrt_of_1024_is_32 (y : ℕ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 :=
sorry

end sqrt_of_1024_is_32_l6_6530


namespace h_plus_k_l6_6769

theorem h_plus_k :
  ∀ h k : ℝ, (∀ x : ℝ, x^2 + 4 * x + 4 = (x + h) ^ 2 - k) → h + k = 2 :=
by
  intro h k H
  -- using sorry to indicate the proof is omitted
  sorry

end h_plus_k_l6_6769


namespace max_blue_cubes_visible_l6_6598

def max_visible_blue_cubes (board : ℕ × ℕ × ℕ → ℕ) : ℕ :=
  board (0, 0, 0)

theorem max_blue_cubes_visible (board : ℕ × ℕ × ℕ → ℕ) :
  max_visible_blue_cubes board = 12 :=
sorry

end max_blue_cubes_visible_l6_6598


namespace speed_of_first_plane_l6_6874

theorem speed_of_first_plane
  (v : ℕ)
  (travel_time : ℚ := 44 / 11)
  (relative_speed : ℚ := v + 90)
  (distance : ℚ := 800) :
  (relative_speed * travel_time = distance) → v = 110 :=
by
  sorry

end speed_of_first_plane_l6_6874


namespace largest_xy_l6_6296

-- Define the problem conditions
def conditions (x y : ℕ) : Prop := 27 * x + 35 * y ≤ 945 ∧ x > 0 ∧ y > 0

-- Define the largest value of xy
def largest_xy_value : ℕ := 234

-- Prove that the largest possible value of xy given conditions is 234
theorem largest_xy (x y : ℕ) (h : conditions x y) : x * y ≤ largest_xy_value :=
sorry

end largest_xy_l6_6296


namespace jack_bill_age_difference_l6_6546

theorem jack_bill_age_difference :
  ∃ (a b : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (7 * a - 29 * b = 14) ∧ ((10 * a + b) - (10 * b + a) = 36) :=
by
  sorry

end jack_bill_age_difference_l6_6546


namespace sufficient_not_necessary_condition_l6_6499

theorem sufficient_not_necessary_condition :
  ∀ x : ℝ, (x^2 - 3 * x < 0) → (0 < x ∧ x < 2) :=
by 
  sorry

end sufficient_not_necessary_condition_l6_6499


namespace odd_function_a_eq_minus_1_l6_6056

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x + a) / x

theorem odd_function_a_eq_minus_1 (a : ℝ) :
  (∀ x : ℝ, f (-x) a = -f x a) → a = -1 :=
by
  intros h
  sorry

end odd_function_a_eq_minus_1_l6_6056


namespace inequality_solution_l6_6309

theorem inequality_solution (x : ℝ) :
  27 ^ (Real.log x / Real.log 3) ^ 2 - 8 * x ^ (Real.log x / Real.log 3) ≥ 3 ↔
  x ∈ Set.Icc 0 (1 / 3) ∪ Set.Ici 3 :=
sorry

end inequality_solution_l6_6309


namespace gender_related_interest_expectation_value_l6_6707

/-
Given the contingency table:
Male Interested: 240, Male Less Interested: 160, Female Interested: 150, Female Less Interested: 50,
Total Interested: 390, Total Less Interested: 210, Total: 600.
-/

def contingency_table := {
  male_interested : ℕ := 240,
  male_less_interested : ℕ := 160,
  female_interested : ℕ := 150,
  female_less_interested : ℕ := 50,
  total_interested : ℕ := 390,
  total_less_interested : ℕ := 210,
  total : ℕ := 600
}

 /-
 Calculate the K^2 statistic.
 -/

def K_squared (n a b c d : ℕ) :=
  (n * (a * d - b * c)^2) / (a + b) / (c + d) / (a + c) / (b + d)

noncomputable def k_value :=
  K_squared 
    contingency_table.total 
    contingency_table.male_interested 
    contingency_table.male_less_interested 
    contingency_table.female_interested 
    contingency_table.female_less_interested

/-
 Prove that gender is related to interest in new energy vehicles.
 -/
theorem gender_related_interest : k_value > 6.635 := sorry

/-
 Given male-to-female ratio is 2:1, from 6 selected individuals (2 females, 4 males), let X be the number of females among 3 individuals selected.
 Find the distribution and expectation of X.
 -/
def male_female_ratio (total females males : ℕ) := 
  females = 1/3 * total ∧ 
  males = 2/3 * total

variable (total_individuals: ℕ := 6)
variable (selected_individuals: ℕ := 3)

def probs (females males selected : ℕ) :=
  (P (X = 0) := 1/5) ∧ 
  (P (X = 1) := 3/5) ∧ 
  (P (X = 2) := 1/5)

def expectation_X (p0 p1 p2 : ℕ) :=
  (0 * p0 + 1 * p1 + 2 * p2 : real)

noncomputable def E_X :=
  expectation_X 1/5 3/5 1/5

theorem expectation_value : E_X = 1 := sorry

end gender_related_interest_expectation_value_l6_6707


namespace set_intersection_eq_l6_6841

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

def B : Set ℝ := { x | x < -2 ∨ x > 5 }

def C_U (B : Set ℝ) : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }

theorem set_intersection_eq : A ∩ (C_U B) = { x | -2 ≤ x ∧ x ≤ 3 } :=
  sorry

end set_intersection_eq_l6_6841


namespace bridge_length_l6_6610

-- Defining the problem based on the given conditions and proof goal
theorem bridge_length (L : ℝ) 
  (h1 : L / 4 + L / 3 + 120 = L) :
  L = 288 :=
sorry

end bridge_length_l6_6610


namespace problem_statement_l6_6514

theorem problem_statement (x : ℕ) (h : 4 * (3^x) = 2187) : (x + 2) * (x - 2) = 21 := 
by
  sorry

end problem_statement_l6_6514


namespace max_parabola_ratio_l6_6652

noncomputable def parabola_max_ratio (x y : ℝ) : ℝ :=
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (x, y)
  
  let MO : ℝ := Real.sqrt (x^2 + y^2)
  let MF : ℝ := Real.sqrt ((x - 1)^2 + y^2)
  
  MO / MF

theorem max_parabola_ratio :
  ∃ x y : ℝ, y^2 = 4 * x ∧ parabola_max_ratio x y = 2 * Real.sqrt 3 / 3 :=
sorry

end max_parabola_ratio_l6_6652


namespace printer_time_l6_6114

theorem printer_time (Tx : ℝ) 
  (h1 : ∀ (Ty Tz : ℝ), Ty = 10 → Tz = 20 → 1 / Ty + 1 / Tz = 3 / 20) 
  (h2 : ∀ (T_combined : ℝ), T_combined = 20 / 3 → Tx / T_combined = 2.4) :
  Tx = 16 := 
by 
  sorry

end printer_time_l6_6114


namespace binomial_multiplication_subtraction_l6_6300

variable (x : ℤ)

theorem binomial_multiplication_subtraction :
  (4 * x - 3) * (x + 6) - ( (2 * x + 1) * (x - 4) ) = 2 * x^2 + 28 * x - 14 := by
  sorry

end binomial_multiplication_subtraction_l6_6300


namespace range_a_l6_6274

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2 * x else x - 1

theorem range_a (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 ≠ a^2 - 2 * a ∧ f x2 ≠ a^2 - 2 * a ∧ f x3 ≠ a^2 - 2 * a) ↔ (0 < a ∧ a < 1 ∨ 1 < a ∧ a < 2) :=
by
  sorry

end range_a_l6_6274


namespace intersection_complement_eq_l6_6277

/-- Define the sets U, A, and B -/
def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {3, 7, 9}
def B : Set ℕ := {1, 9}

/-- Define the complement of B with respect to U -/
def complement_U_B : Set ℕ := U \ B

/-- Theorem stating the intersection of A and the complement of B with respect to U -/
theorem intersection_complement_eq : A ∩ complement_U_B = {3, 7} :=
by
  sorry

end intersection_complement_eq_l6_6277


namespace other_leg_length_l6_6436

theorem other_leg_length (a b c : ℕ) (ha : a = 24) (hc : c = 25) 
  (h : a * a + b * b = c * c) : b = 7 := 
by 
  sorry

end other_leg_length_l6_6436


namespace number_of_white_tshirts_in_one_pack_l6_6843

namespace TShirts

variable (W : ℕ)

noncomputable def total_white_tshirts := 2 * W
noncomputable def total_blue_tshirts := 4 * 3
noncomputable def cost_per_tshirt := 3
noncomputable def total_cost := 66

theorem number_of_white_tshirts_in_one_pack :
  2 * W * cost_per_tshirt + total_blue_tshirts * cost_per_tshirt = total_cost → W = 5 :=
by
  sorry

end TShirts

end number_of_white_tshirts_in_one_pack_l6_6843


namespace john_bike_speed_l6_6835

noncomputable def average_speed_for_bike_ride (swim_distance swim_speed run_distance run_speed bike_distance total_time : ℕ) := 
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

theorem john_bike_speed : average_speed_for_bike_ride 1 5 8 12 (3 / 2) = 18 := by
  sorry

end john_bike_speed_l6_6835


namespace probability_of_graduate_degree_l6_6538

variables (G C N : ℕ)
axiom h1 : G / N = 1 / 8
axiom h2 : C / N = 2 / 3

noncomputable def total_college_graduates (G C : ℕ) : ℕ := G + C

noncomputable def probability_graduate_degree (G C : ℕ) : ℚ := G / (total_college_graduates G C)

theorem probability_of_graduate_degree :
  probability_graduate_degree 3 16 = 3 / 19 :=
by 
  -- Here, we need to prove that the probability of picking a college graduate with a graduate degree
  -- is 3 / 19 given the conditions.
  sorry

end probability_of_graduate_degree_l6_6538


namespace find_intersection_l6_6420

noncomputable def A : Set ℝ := { x | -4 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x | x ≤ 2 }

theorem find_intersection : A ∩ B = { x | -4 < x ∧ x ≤ 2 } := sorry

end find_intersection_l6_6420


namespace triangle_area_ratio_l6_6991

noncomputable def vector_sum_property (OA OB OC : ℝ × ℝ × ℝ) : Prop :=
  OA + (2 : ℝ) • OB + (3 : ℝ) • OC = (0 : ℝ × ℝ × ℝ)

noncomputable def area_ratio (S_ABC S_AOC : ℝ) : Prop :=
  S_ABC / S_AOC = 3

theorem triangle_area_ratio
    (OA OB OC : ℝ × ℝ × ℝ)
    (S_ABC S_AOC : ℝ)
    (h1 : vector_sum_property OA OB OC)
    (h2 : S_ABC = 3 * S_AOC) :
  area_ratio S_ABC S_AOC :=
by
  sorry

end triangle_area_ratio_l6_6991


namespace extreme_points_sum_gt_l6_6949

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * x^2 - Real.log x

theorem extreme_points_sum_gt (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 8)
    {x₁ x₂ : ℝ} (h₂ : f x₁ a = 0) (h₃ : f x₂ a = 0) (h₄ : x₁ < x₂)
    (h₅ : 0 < x₁) (h₆ : 0 < x₂) : f x₁ a + f x₂ a > 3 - 2 * Real.log 2 := sorry

end extreme_points_sum_gt_l6_6949


namespace charles_average_speed_l6_6379

theorem charles_average_speed
  (total_distance : ℕ)
  (half_distance : ℕ)
  (second_half_speed : ℕ)
  (total_time : ℕ)
  (first_half_distance second_half_distance : ℕ)
  (time_for_second_half : ℕ)
  (time_for_first_half : ℕ)
  (first_half_speed : ℕ)
  (h1 : total_distance = 3600)
  (h2 : half_distance = total_distance / 2)
  (h3 : first_half_distance = half_distance)
  (h4 : second_half_distance = half_distance)
  (h5 : second_half_speed = 180)
  (h6 : total_time = 30)
  (h7 : time_for_second_half = second_half_distance / second_half_speed)
  (h8 : time_for_first_half = total_time - time_for_second_half)
  (h9 : first_half_speed = first_half_distance / time_for_first_half) :
  first_half_speed = 90 := by
  sorry

end charles_average_speed_l6_6379


namespace suitable_for_sampling_l6_6348

-- Definitions based on conditions
def optionA_requires_comprehensive : Prop := true
def optionB_requires_comprehensive : Prop := true
def optionC_requires_comprehensive : Prop := true
def optionD_allows_sampling : Prop := true

-- Problem in Lean: Prove that option D is suitable for a sampling survey
theorem suitable_for_sampling : optionD_allows_sampling := by
  sorry

end suitable_for_sampling_l6_6348


namespace carter_has_255_cards_l6_6986

-- Definition of the number of baseball cards Marcus has.
def marcus_cards : ℕ := 350

-- Definition of the number of more cards Marcus has than Carter.
def difference : ℕ := 95

-- Definition of the number of baseball cards Carter has.
def carter_cards : ℕ := marcus_cards - difference

-- Theorem stating that Carter has 255 baseball cards.
theorem carter_has_255_cards : carter_cards = 255 :=
sorry

end carter_has_255_cards_l6_6986


namespace find_m_l6_6405

-- Definitions based on conditions in the problem
def f (x : ℝ) := 4 * x + 7

-- Theorem statement to prove m = 3/4 given the conditions
theorem find_m (m : ℝ) :
  (∀ x : ℝ, f (1/2 * x - 1) = 2 * x + 3) →
  f (m - 1) = 6 →
  m = 3 / 4 :=
by
  -- Proof should go here
  sorry

end find_m_l6_6405


namespace find_largest_number_l6_6031

theorem find_largest_number (w x y z : ℕ) 
  (h1 : w + x + y = 190) 
  (h2 : w + x + z = 210) 
  (h3 : w + y + z = 220) 
  (h4 : x + y + z = 235) : 
  max (max w x) (max y z) = 95 := 
sorry

end find_largest_number_l6_6031


namespace regular_polygon_sides_l6_6194

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l6_6194


namespace radon_nikodym_theorem_failure_l6_6564

open MeasureTheory

noncomputable def measurable_space' : MeasurableSpace ℝ := borel ℝ

noncomputable def lebesgue_measure (s: Set ℝ) : ℝ := by
  exact real.volume s

noncomputable def counting_measure (s: Set ℝ) : ℝ := by
  exact s.to_finset.card

theorem radon_nikodym_theorem_failure :
  ∀ B : Set ℝ, measurable_space'.measurable_set' B →
  (counting_measure B = 0 → lebesgue_measure B = 0) ∧ ¬(∃ (f : ℝ → ℝ), ∀ A : Set ℝ, measurable_space'.measurable_set' A →
  lebesgue_measure A = (∫ x in A, f x ∂counting_measure)) := sorry

end radon_nikodym_theorem_failure_l6_6564


namespace new_mix_concentration_l6_6890

theorem new_mix_concentration 
  (capacity1 capacity2 capacity_mix : ℝ)
  (alc_percent1 alc_percent2 : ℝ)
  (amount1 amount2 : capacity1 = 3 ∧ capacity2 = 5 ∧ capacity_mix = 10)
  (percent1: alc_percent1 = 0.25)
  (percent2: alc_percent2 = 0.40)
  (total_volume : ℝ)
  (eight_liters : total_volume = 8) :
  (alc_percent1 * capacity1 + alc_percent2 * capacity2) / total_volume * 100 = 34.375 :=
by
  sorry

end new_mix_concentration_l6_6890


namespace total_journey_distance_l6_6891

theorem total_journey_distance (D : ℝ)
  (h1 : (D / 2) / 21 + (D / 2) / 24 = 25) : D = 560 := by
  sorry

end total_journey_distance_l6_6891


namespace intersection_P_Q_l6_6684

def P := {x : ℤ | x^2 - 16 < 0}
def Q := {x : ℤ | ∃ n : ℤ, x = 2 * n}

theorem intersection_P_Q :
  P ∩ Q = {-2, 0, 2} :=
sorry

end intersection_P_Q_l6_6684


namespace no_nat_solutions_l6_6047

theorem no_nat_solutions (x y : ℕ) : (2 * x + y) * (2 * y + x) ≠ 2017 ^ 2017 := by sorry

end no_nat_solutions_l6_6047


namespace watermelon_vendor_profit_l6_6207

theorem watermelon_vendor_profit 
  (purchase_price : ℝ) (selling_price_initial : ℝ) (initial_quantity_sold : ℝ) 
  (decrease_factor : ℝ) (additional_quantity_per_decrease : ℝ) (fixed_cost : ℝ) 
  (desired_profit : ℝ) 
  (x : ℝ)
  (h_purchase : purchase_price = 2)
  (h_selling_initial : selling_price_initial = 3)
  (h_initial_quantity : initial_quantity_sold = 200)
  (h_decrease_factor : decrease_factor = 0.1)
  (h_additional_quantity : additional_quantity_per_decrease = 40)
  (h_fixed_cost : fixed_cost = 24)
  (h_desired_profit : desired_profit = 200) :
  (x = 2.8 ∨ x = 2.7) ↔ 
  ((x - purchase_price) * (initial_quantity_sold + additional_quantity_per_decrease / decrease_factor * (selling_price_initial - x)) - fixed_cost = desired_profit) :=
by sorry

end watermelon_vendor_profit_l6_6207


namespace rectangle_inscribed_circle_circumference_l6_6356

/-- A 9 cm by 12 cm rectangle is inscribed in a circle. The circumference of the circle is 15π cm. -/
theorem rectangle_inscribed_circle_circumference :
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  circumference = 15 * Real.pi :=
by
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  have h_diameter : diameter = 15 := by
    sorry
  have h_circumference : circumference = 15 * Real.pi := by
    sorry
  exact h_circumference

end rectangle_inscribed_circle_circumference_l6_6356


namespace maximum_capacity_of_smallest_barrel_l6_6606

theorem maximum_capacity_of_smallest_barrel : 
  ∃ (A B C D E F : ℕ), 
    8 ≤ A ∧ A ≤ 16 ∧
    8 ≤ B ∧ B ≤ 16 ∧
    8 ≤ C ∧ C ≤ 16 ∧
    8 ≤ D ∧ D ≤ 16 ∧
    8 ≤ E ∧ E ≤ 16 ∧
    8 ≤ F ∧ F ≤ 16 ∧
    (A + B + C + D + E + F = 72) ∧
    ((C + D) / 2 = 14) ∧ 
    (F = 11 ∨ F = 13) ∧
    (∀ (A' : ℕ), 8 ≤ A' ∧ A' ≤ 16 ∧
      ∃ (B' C' D' E' F' : ℕ), 
      8 ≤ B' ∧ B' ≤ 16 ∧
      8 ≤ C' ∧ C' ≤ 16 ∧
      8 ≤ D' ∧ D' ≤ 16 ∧
      8 ≤ E' ∧ E' ≤ 16 ∧
      8 ≤ F' ∧ F' ≤ 16 ∧
      (A' + B' + C' + D' + E' + F' = 72) ∧
      ((C' + D') / 2 = 14) ∧ 
      (F' = 11 ∨ F' = 13) → A' ≤ A ) :=
sorry

end maximum_capacity_of_smallest_barrel_l6_6606


namespace fraction_correct_l6_6875

-- Define the total number of coins.
def total_coins : ℕ := 30

-- Define the number of states that joined the union in the decade 1800 through 1809.
def states_1800_1809 : ℕ := 4

-- Define the fraction of coins representing states joining in the decade 1800 through 1809.
def fraction_coins_1800_1809 : ℚ := states_1800_1809 / total_coins

-- The theorem statement that needs to be proved.
theorem fraction_correct : fraction_coins_1800_1809 = (2 / 15) := 
by
  sorry

end fraction_correct_l6_6875


namespace age_difference_between_brother_and_cousin_l6_6691

-- Define the ages used in the problem 
def Lexie_age : ℕ := 8
def Grandma_age : ℕ := 68
def Brother_age : ℕ := Lexie_age - 6
def Sister_age : ℕ := 2 * Lexie_age
def Uncle_age : ℕ := Grandma_age - 12
def Cousin_age : ℕ := Brother_age + 5

-- The proof problem statement in Lean 4
theorem age_difference_between_brother_and_cousin : 
  Brother_age < Cousin_age ∧ Cousin_age - Brother_age = 5 :=
by
  -- Definitions and imports are done above. The statement below should prove the age difference.
  sorry

end age_difference_between_brother_and_cousin_l6_6691


namespace regular_polygon_sides_l6_6203

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6203


namespace k_gonal_number_proof_l6_6862

-- Definitions for specific k-gonal numbers based on given conditions.
def triangular_number (n : ℕ) := (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n
def square_number (n : ℕ) := n^2
def pentagonal_number (n : ℕ) := (3/2 : ℚ) * n^2 - (1/2 : ℚ) * n
def hexagonal_number (n : ℕ) := 2 * n^2 - n

-- General definition for the k-gonal number
def k_gonal_number (n k : ℕ) : ℚ := ((k - 2) / 2) * n^2 + ((4 - k) / 2) * n

-- Corresponding Lean statement for the proof problem
theorem k_gonal_number_proof (n k : ℕ) (hk : k ≥ 3) :
    (k = 3 -> triangular_number n = k_gonal_number n k) ∧
    (k = 4 -> square_number n = k_gonal_number n k) ∧
    (k = 5 -> pentagonal_number n = k_gonal_number n k) ∧
    (k = 6 -> hexagonal_number n = k_gonal_number n k) ∧
    (n = 10 ∧ k = 24 -> k_gonal_number n k = 1000) :=
by
  intros
  sorry

end k_gonal_number_proof_l6_6862


namespace sundae_cost_l6_6049

theorem sundae_cost (ice_cream_cost toppings_cost : ℕ) (num_toppings : ℕ) :
  ice_cream_cost = 200  →
  toppings_cost = 50 →
  num_toppings = 10 →
  ice_cream_cost + num_toppings * toppings_cost = 700 := by
  sorry

end sundae_cost_l6_6049


namespace minimum_rectangles_to_cover_cells_l6_6341

theorem minimum_rectangles_to_cover_cells (figure : Type) 
  (cells : set figure) 
  (corners_1 : fin 12 → figure)
  (corners_2 : fin 12 → figure)
  (grouped_corners_2 : fin 4 → fin 3 → figure)
  (h1 : ∀ i, corners_2 i ∈ grouped_corners_2 (i / 3) ((i % 3) + 1)) 
  (rectangles : set (set figure)) 
  (h2 : ∀ i j, j ≠ i → corners_1 i ∉ corners_1 j)
  (h3 : ∀ i j k, grouped_corners_2 i j ≠ grouped_corners_2 i k) :
  ∃ rectangles : set (set figure), rectangles.card = 12 ∧
  (∀ cell ∈ cells, ∃ rectangle ∈ rectangles, cell ∈ rectangle) :=
sorry

end minimum_rectangles_to_cover_cells_l6_6341


namespace find_y_given_conditions_l6_6960

theorem find_y_given_conditions (x : ℤ) (y : ℤ) (h1 : x^2 - 3 * x + 7 = y + 2) (h2 : x = -5) : y = 45 :=
by
  sorry

end find_y_given_conditions_l6_6960


namespace tan_theta_is_sqrt3_div_5_l6_6428

open Real

theorem tan_theta_is_sqrt3_div_5 (theta : ℝ) (h : 2 * sin (theta + π / 3) = 3 * sin (π / 3 - theta)) :
  tan theta = sqrt 3 / 5 :=
sorry

end tan_theta_is_sqrt3_div_5_l6_6428


namespace regular_polygon_sides_l6_6180

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l6_6180


namespace vectors_parallel_perpendicular_l6_6044

theorem vectors_parallel_perpendicular (t t1 t2 : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) 
    (h_a : a = (2, t)) (h_b : b = (1, 2)) :
    ((2 * 2 = t * 1) → t1 = 4) ∧ ((2 * 1 + 2 * t = 0) → t2 = -1) :=
by 
  sorry

end vectors_parallel_perpendicular_l6_6044


namespace least_number_to_subtract_l6_6477

theorem least_number_to_subtract (x : ℕ) :
  1439 - x ≡ 3 [MOD 5] ∧ 
  1439 - x ≡ 3 [MOD 11] ∧ 
  1439 - x ≡ 3 [MOD 13] ↔ 
  x = 9 :=
by sorry

end least_number_to_subtract_l6_6477


namespace max_triangle_area_l6_6111

theorem max_triangle_area :
  ∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 1 ≤ b ∧ b ≤ 2 ∧ 2 ≤ c ∧ c ≤ 3 ∧ 
  (a + b > c ∧ a + c > b ∧ b + c > a) ∧ (1 ≤ 0.5 * a * b) := sorry

end max_triangle_area_l6_6111


namespace polygon_with_150_degree_interior_angles_has_12_sides_l6_6172

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l6_6172


namespace age_difference_36_l6_6471

noncomputable def jack_age (a b : ℕ) : ℕ := 10 * a + b
noncomputable def bill_age (b a : ℕ) : ℕ := 10 * b + a

theorem age_difference_36 (a b : ℕ) (h : 10 * a + b + 3 = 3 * (10 * b + a + 3)) :
  jack_age a b - bill_age b a = 36 :=
by sorry

end age_difference_36_l6_6471


namespace polygon_with_150_degree_interior_angles_has_12_sides_l6_6173

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l6_6173


namespace abs_of_sub_sqrt_l6_6251

theorem abs_of_sub_sqrt (h : 2 > Real.sqrt 3) : |2 - Real.sqrt 3| = 2 - Real.sqrt 3 :=
sorry

end abs_of_sub_sqrt_l6_6251


namespace algebraic_expression_value_l6_6347

-- Given conditions as definitions and assumption
variables (a b : ℝ)
def expression1 (x : ℝ) := 2 * a * x^3 - 3 * b * x + 8
def expression2 := 9 * b - 6 * a + 2

theorem algebraic_expression_value
  (h1 : expression1 (-1) = 18) :
  expression2 = 32 :=
by
  sorry

end algebraic_expression_value_l6_6347


namespace range_of_a_l6_6811

theorem range_of_a 
{α : Type*} [LinearOrderedField α] (a : α) 
(h : ∃ x, x = 3 ∧ (x - a) * (x + 2 * a - 1) ^ 2 * (x - 3 * a) ≤ 0) :
a = -1 ∨ (1 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_l6_6811


namespace line_l_statements_correct_l6_6515

theorem line_l_statements_correct
  (A B C : ℝ)
  (hAB : ¬(A = 0 ∧ B = 0)) :
  ( (2 * A + B + C = 0 → ∀ x y, A * (x - 2) + B * (y - 1) = 0 ↔ A * x + B * y + C = 0 ) ∧
    ((A ≠ 0 ∧ B ≠ 0) → ∃ x, A * x + C = 0 ∧ ∃ y, B * y + C = 0) ∧
    (A = 0 ∧ B ≠ 0 ∧ C ≠ 0 → ∀ y, B * y + C = 0 ↔ y = -C / B) ∧
    (A ≠ 0 ∧ B^2 + C^2 = 0 → ∀ x, A * x = 0 ↔ x = 0) ) :=
by
  sorry

end line_l_statements_correct_l6_6515


namespace regular_polygon_sides_l6_6191

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6191


namespace card_swapping_cost_l6_6105

noncomputable def swap_cost (x y : ℕ) : ℕ := 2 * (x - y).natAbs

theorem card_swapping_cost (n : ℕ) (a : Fin n → Fin n) (perm : ∀ i, 1 ≤ a i.val.succ ∧ a i.val.succ <= n) :
  ∃ (swap_seq : List (Fin n × Fin n)), 
  (∀ (swap : Fin n × Fin n), swap ∈ swap_seq → 
      swap_cost swap.1.val swap.2.val <= |swap.1.val + 1 - swap.2.val + 1| ) ∧ 
  swap_seq.map (λ swap, swap_cost swap.1.val swap.2.val).sum ≤ ∑ i, (a i).val ∣ i := 
begin
  sorry,
end

end card_swapping_cost_l6_6105


namespace octal_addition_correct_l6_6211

def octal_to_decimal (n : ℕ) : ℕ := 
  /- function to convert an octal number to decimal goes here -/
  sorry

def decimal_to_octal (n : ℕ) : ℕ :=
  /- function to convert a decimal number to octal goes here -/
  sorry

theorem octal_addition_correct :
  let a := 236 
  let b := 521
  let c := 74
  let sum_decimal := octal_to_decimal a + octal_to_decimal b + octal_to_decimal c
  decimal_to_octal sum_decimal = 1063 :=
by
  sorry

end octal_addition_correct_l6_6211


namespace rectangle_inscribed_circle_circumference_l6_6357

/-- A 9 cm by 12 cm rectangle is inscribed in a circle. The circumference of the circle is 15π cm. -/
theorem rectangle_inscribed_circle_circumference :
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  circumference = 15 * Real.pi :=
by
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  have h_diameter : diameter = 15 := by
    sorry
  have h_circumference : circumference = 15 * Real.pi := by
    sorry
  exact h_circumference

end rectangle_inscribed_circle_circumference_l6_6357


namespace find_x_l6_6249

theorem find_x :
  (x : ℝ) →
  (0.40 * 2 = 0.25 * (0.30 * 15 + x)) →
  x = -1.3 :=
by
  intros x h
  sorry

end find_x_l6_6249


namespace option_D_is_greater_than_reciprocal_l6_6887

theorem option_D_is_greater_than_reciprocal:
  ∀ (x : ℚ), (x = 2) → x > 1/x := by
  intro x
  intro hx
  rw [hx]
  norm_num

end option_D_is_greater_than_reciprocal_l6_6887


namespace max_initial_jars_l6_6241

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l6_6241


namespace regular_polygon_sides_l6_6167

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6167


namespace regular_polygon_sides_l6_6166

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6166


namespace incorrect_statement_D_l6_6265

noncomputable def a := (-3.0) ^ (-4)
noncomputable def b := - (3.0 ^ 4)
noncomputable def c := - (3.0 ^ (-4))

theorem incorrect_statement_D : a * b ≠ 1 := by
  have ha : a = (1 / (3 ^ 4)) := by sorry
  have hb : b = -(3 ^ 4) := by sorry
  have hab : a * b = (1 / (3 ^ 4)) * -(3 ^ 4) := by 
    rw [ha, hb]
    sorry
  show (a * b ≠ 1) from 
    calc  (1 / (3 ^ 4)) * -(3 ^ 4) = -1 := by sorry
    -1 ≠ 1 := by linarith

end incorrect_statement_D_l6_6265


namespace regular_polygon_sides_l6_6137

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l6_6137


namespace xiaomings_mother_money_l6_6115

-- Definitions for the conditions
def price_A : ℕ := 6
def price_B : ℕ := 9
def units_more_A := 2

-- Main statement to prove
theorem xiaomings_mother_money (x : ℕ) (M : ℕ) :
  M = 6 * x ∧ M = 9 * (x - 2) → M = 36 :=
by
  -- Assuming the conditions are given
  rintro ⟨hA, hB⟩
  -- The proof is omitted
  sorry

end xiaomings_mother_money_l6_6115


namespace min_value_l6_6983

noncomputable def min_value_expression (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  ∑ i in Finset.range n, (x i) ^ (i + 1) / (i + 1)

theorem min_value (n : ℕ) (hpos : 0 < n)
  (x : Fin n → ℝ) (hx_pos : ∀ i, 0 < x i)
  (hx_sum : ∑ i in Finset.range n, 1 / (x i) = n) :
  min_value_expression n x = ∑ i in Finset.range n, 1 / (i + 1) :=
begin
  sorry
end

end min_value_l6_6983


namespace probability_of_earning_2400_l6_6087

noncomputable def spinner_labels := ["Bankrupt", "$700", "$900", "$200", "$3000", "$800"]
noncomputable def total_possibilities := (spinner_labels.length : ℕ) ^ 3
noncomputable def favorable_outcomes := 6

theorem probability_of_earning_2400 :
  (favorable_outcomes : ℚ) / total_possibilities = 1 / 36 := by
  sorry

end probability_of_earning_2400_l6_6087


namespace percentage_dogs_movies_l6_6922

-- Definitions from conditions
def total_students : ℕ := 30
def students_preferring_dogs_videogames : ℕ := total_students / 2
def students_preferring_dogs : ℕ := 18
def students_preferring_dogs_movies : ℕ := students_preferring_dogs - students_preferring_dogs_videogames

-- Theorem statement
theorem percentage_dogs_movies : (students_preferring_dogs_movies * 100 / total_students) = 10 := by
  sorry

end percentage_dogs_movies_l6_6922


namespace original_fraction_is_one_third_l6_6710

theorem original_fraction_is_one_third (a b : ℕ) 
  (coprime_ab : Nat.gcd a b = 1) 
  (h : (a + 2) * b = 3 * a * b^2) : 
  (a = 1 ∧ b = 3) := 
by 
  sorry

end original_fraction_is_one_third_l6_6710


namespace equation1_solution_equation2_solution_l6_6398

theorem equation1_solution (x : ℝ) : (x - 1) ^ 3 = 64 ↔ x = 5 := sorry

theorem equation2_solution (x : ℝ) : 25 * x ^ 2 + 3 = 12 ↔ x = 3 / 5 ∨ x = -3 / 5 := sorry

end equation1_solution_equation2_solution_l6_6398


namespace main_theorem_l6_6629

-- Define the interval (3π/4, π)
def theta_range (θ : ℝ) : Prop :=
  (3 * Real.pi / 4) < θ ∧ θ < Real.pi

-- Define the condition
def inequality_condition (θ x : ℝ) : Prop :=
  x^2 * Real.sin θ - x * (1 - x) + (1 - x)^2 * Real.cos θ + 2 * x * (1 - x) * Real.sqrt (Real.cos θ * Real.sin θ) > 0

-- The main theorem
theorem main_theorem (θ x : ℝ) (hθ : theta_range θ) (hx : 0 ≤ x ∧ x ≤ 1) : inequality_condition θ x :=
by
  sorry

end main_theorem_l6_6629


namespace homer_total_points_l6_6423

noncomputable def first_try_points : ℕ := 400
noncomputable def second_try_points : ℕ := first_try_points - 70
noncomputable def third_try_points : ℕ := 2 * second_try_points
noncomputable def total_points : ℕ := first_try_points + second_try_points + third_try_points

theorem homer_total_points : total_points = 1390 :=
by
  -- Using the definitions above, we need to show that total_points = 1390
  sorry

end homer_total_points_l6_6423


namespace unique_b_for_unique_solution_l6_6799

theorem unique_b_for_unique_solution (c : ℝ) (h₁ : c ≠ 0) :
  (∃ b : ℝ, b > 0 ∧ ∃! x : ℝ, x^2 + (b + (2 / b)) * x + c = 0) →
  c = 2 :=
by
  -- sorry will go here to indicate the proof is to be filled in
  sorry

end unique_b_for_unique_solution_l6_6799


namespace find_a_l6_6672

theorem find_a (x a : ℝ) : 
  (a + 2 = 0) ↔ (a = -2) :=
by
  sorry

end find_a_l6_6672


namespace smaller_number_4582_l6_6465

theorem smaller_number_4582 (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha_b : a < 100) (hb_b : b < 100) (h : a * b = 4582) :
  min a b = 21 :=
sorry

end smaller_number_4582_l6_6465


namespace no_a_for_x4_l6_6052

theorem no_a_for_x4 : ∃ a : ℝ, (1 / (4 + a) + 1 / (4 - a) = 1 / (4 - a)) → false :=
  by sorry

end no_a_for_x4_l6_6052


namespace fraction_decomposition_l6_6319
noncomputable def A := (48 : ℚ) / 17
noncomputable def B := (-(25 : ℚ) / 17)

theorem fraction_decomposition (A : ℚ) (B : ℚ) :
  ( ∀ x : ℚ, x ≠ -5 ∧ x ≠ 2/3 →
    (7 * x - 13) / (3 * x^2 + 13 * x - 10) = A / (x + 5) + B / (3 * x - 2) ) ↔ 
    (A = (48 : ℚ) / 17 ∧ B = (-(25 : ℚ) / 17)) :=
by
  sorry

end fraction_decomposition_l6_6319


namespace max_value_of_f_l6_6580

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + Real.sin (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, f x = 1 + Real.sqrt 2 := 
sorry

end max_value_of_f_l6_6580


namespace anne_cleaning_time_l6_6789

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l6_6789


namespace second_divisor_is_24_l6_6882

theorem second_divisor_is_24 (m n k l : ℤ) (hm : m = 288 * k + 47) (hn : m = n * l + 23) : n = 24 :=
by
  sorry

end second_divisor_is_24_l6_6882


namespace solve_f_435_l6_6275

variable (f : ℝ → ℝ)

-- Conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (3 - x) = f x

-- To Prove
theorem solve_f_435 : f 435 = 0 :=
by
  sorry

end solve_f_435_l6_6275


namespace find_integer_n_l6_6632

theorem find_integer_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] ∧ n = 0 :=
by
  sorry

end find_integer_n_l6_6632


namespace log_exp_sum_l6_6220

theorem log_exp_sum :
  2^(Real.log 3 / Real.log 2) + Real.log (Real.sqrt 5) / Real.log 10 + Real.log (Real.sqrt 20) / Real.log 10 = 4 :=
by
  sorry

end log_exp_sum_l6_6220


namespace triangle_angle_split_l6_6840

-- Conditions
variables (A B C C1 C2 : ℝ)
-- Axioms/Assumptions
axiom angle_order : A < B
axiom angle_partition : A + C1 = 90 ∧ B + C2 = 90

-- The theorem to prove
theorem triangle_angle_split : C1 - C2 = B - A :=
by {
  sorry
}

end triangle_angle_split_l6_6840


namespace rectangle_area_from_square_l6_6206

theorem rectangle_area_from_square 
  (square_area : ℕ) 
  (width_rect : ℕ) 
  (length_rect : ℕ) 
  (h_square_area : square_area = 36)
  (h_width_rect : width_rect * width_rect = square_area)
  (h_length_rect : length_rect = 3 * width_rect) :
  width_rect * length_rect = 108 :=
by
  sorry

end rectangle_area_from_square_l6_6206


namespace circumcircle_trilinear_eq_incircle_trilinear_eq_excircle_trilinear_eq_l6_6252

-- Define the variables
variables {a b c : ℝ} {x y z : ℝ}
variables {α β γ : ℝ}

-- Circumcircle equation
theorem circumcircle_trilinear_eq :
  a * y * z + b * x * z + c * x * y = 0 :=
sorry

-- Incircle equation
theorem incircle_trilinear_eq :
  (Real.cos (α / 2) * Real.sqrt x) + 
  (Real.cos (β / 2) * Real.sqrt y) + 
  (Real.cos (γ / 2) * Real.sqrt z) = 0 :=
sorry

-- Excircle equation
theorem excircle_trilinear_eq :
  (Real.cos (α / 2) * Real.sqrt (-x)) + 
  (Real.cos (β / 2) * Real.sqrt y) + 
  (Real.cos (γ / 2) * Real.sqrt z) = 0 :=
sorry

end circumcircle_trilinear_eq_incircle_trilinear_eq_excircle_trilinear_eq_l6_6252


namespace reaction_produces_correct_moles_l6_6506

-- Define the variables and constants
def moles_CO2 := 2
def moles_H2O := 2
def moles_H2CO3 := moles_CO2 -- based on the balanced reaction CO2 + H2O → H2CO3

-- The theorem we need to prove
theorem reaction_produces_correct_moles :
  moles_H2CO3 = 2 :=
by
  -- Mathematical reasoning goes here
  sorry

end reaction_produces_correct_moles_l6_6506


namespace num_ways_express_2009_as_diff_of_squares_l6_6018

theorem num_ways_express_2009_as_diff_of_squares : 
  ∃ (n : Nat), n = 12 ∧ 
  ∃ (a b : Int), ∀ c, 2009 = a^2 - b^2 ∧ 
  (c = 1 ∨ c = -1) ∧ (2009 = (c * a)^2 - (c * b)^2) :=
sorry

end num_ways_express_2009_as_diff_of_squares_l6_6018


namespace regular_polygon_sides_l6_6138

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l6_6138


namespace domain_of_fn_l6_6462

noncomputable def domain_fn (x : ℝ) : ℝ := (Real.sqrt (3 * x + 4)) / x

theorem domain_of_fn :
  { x : ℝ | x ≥ -4 / 3 ∧ x ≠ 0 } =
  { x : ℝ | 3 * x + 4 ≥ 0 ∧ x ≠ 0 } :=
by
  ext x
  simp
  exact sorry

end domain_of_fn_l6_6462


namespace exponential_inequality_l6_6640

theorem exponential_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) : 
  x₂ * Real.exp x₁ > x₁ * Real.exp x₂ :=
sorry

end exponential_inequality_l6_6640


namespace find_p_l6_6429

theorem find_p (p q : ℚ) (h1 : 5 * p + 3 * q = 10) (h2 : 3 * p + 5 * q = 20) : 
  p = -5 / 8 :=
by
  sorry

end find_p_l6_6429


namespace flower_profit_equation_l6_6120

theorem flower_profit_equation
  (initial_plants : ℕ := 3)
  (initial_profit_per_plant : ℕ := 10)
  (decrease_in_profit_per_additional_plant : ℕ := 1)
  (target_profit_per_pot : ℕ := 40)
  (x : ℕ) :
  (initial_plants + x) * (initial_profit_per_plant - x) = target_profit_per_pot :=
sorry

end flower_profit_equation_l6_6120


namespace exclude_domain_and_sum_l6_6793

noncomputable def g (x : ℝ) : ℝ :=
  1 / (2 + 1 / (2 + 1 / x))

theorem exclude_domain_and_sum :
  { x : ℝ | x = 0 ∨ x = -1/2 ∨ x = -1/4 } = { x : ℝ | ¬(x ≠ 0 ∧ (2 + 1 / x ≠ 0) ∧ (2 + 1 / (2 + 1 / x) ≠ 0)) } ∧
  (0 + (-1 / 2) + (-1 / 4) = -3 / 4) :=
by
  sorry

end exclude_domain_and_sum_l6_6793


namespace simplify_expression_l6_6569

theorem simplify_expression (α : ℝ) (h_sin_ne_zero : Real.sin α ≠ 0) :
    (1 / Real.sin α + 1 / Real.tan α) * (1 - Real.cos α) = Real.sin α := 
sorry

end simplify_expression_l6_6569


namespace geometric_arithmetic_sequence_relation_l6_6437

theorem geometric_arithmetic_sequence_relation 
    (a : ℕ → ℝ) (b : ℕ → ℝ) (q d a1 : ℝ)
    (h1 : a 1 = a1) (h2 : b 1 = a1) (h3 : a 3 = a1 * q^2)
    (h4 : b 3 = a1 + 2 * d) (h5 : a 3 = b 3) (h6 : a1 > 0) (h7 : q^2 ≠ 1) :
    a 5 > b 5 :=
by
  -- Proof goes here
  sorry

end geometric_arithmetic_sequence_relation_l6_6437


namespace oli_scoops_l6_6302

theorem oli_scoops : ∃ x : ℤ, ∀ y : ℤ, y = 2 * x ∧ y = x + 4 → x = 4 :=
by
  sorry

end oli_scoops_l6_6302


namespace picnic_problem_l6_6751

theorem picnic_problem
  (M W C A : ℕ)
  (h1 : M + W + C = 240)
  (h2 : M = W + 80)
  (h3 : A = C + 80)
  (h4 : A = M + W) :
  M = 120 :=
by
  sorry

end picnic_problem_l6_6751


namespace triangle_is_isosceles_l6_6038

theorem triangle_is_isosceles
  (α β γ x y z w : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α + β = x)
  (h3 : β + γ = y)
  (h4 : γ + α = z)
  (h5 : x + y + z + w = 360) : 
  (α = β ∧ β = γ) ∨ (α = γ ∧ γ = β) ∨ (β = α ∧ α = γ) := by
  sorry

end triangle_is_isosceles_l6_6038


namespace toppings_combination_l6_6084

-- Define the combination function
def combination (n k : ℕ) : ℕ := n.choose k

theorem toppings_combination :
  combination 9 3 = 84 := by
  sorry

end toppings_combination_l6_6084


namespace fraction_mango_sold_l6_6774

theorem fraction_mango_sold :
  ∀ (choco_total mango_total choco_sold unsold: ℕ) (x : ℚ),
    choco_total = 50 →
    mango_total = 54 →
    choco_sold = (3 * 50) / 5 →
    unsold = 38 →
    (choco_total + mango_total) - (choco_sold + x * mango_total) = unsold →
    x = 4 / 27 :=
by
  intros choco_total mango_total choco_sold unsold x
  sorry

end fraction_mango_sold_l6_6774


namespace anthony_pencils_l6_6377

def initial_pencils : ℝ := 56.0  -- Condition 1
def pencils_left : ℝ := 47.0     -- Condition 2
def pencils_given : ℝ := 9.0     -- Correct Answer

theorem anthony_pencils :
  initial_pencils - pencils_left = pencils_given :=
by
  sorry

end anthony_pencils_l6_6377


namespace quadratic_roots_distinct_l6_6726

theorem quadratic_roots_distinct (m : ℝ) :
  let Δ := m^2 + 32 in
  Δ > 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 - 8 = 0 ∧ x2^2 + m * x2 - 8 = 0) :=
begin
  sorry
end

end quadratic_roots_distinct_l6_6726


namespace solution_set_inequality_l6_6328

theorem solution_set_inequality (x : ℝ) : 
  ((x-2) * (3-x) > 0) ↔ (2 < x ∧ x < 3) :=
by sorry

end solution_set_inequality_l6_6328


namespace angle_equivalence_l6_6479

theorem angle_equivalence : (2023 % 360 = -137 % 360) := 
by 
  sorry

end angle_equivalence_l6_6479


namespace sum_inequality_l6_6839

open Real

theorem sum_inequality (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 11) + 1 / (5 * b^2 - 4 * b + 11) + 1 / (5 * c^2 - 4 * c + 11)) ≤ 1 / 4 :=
by
  sorry

end sum_inequality_l6_6839


namespace smallest_percent_both_l6_6848

theorem smallest_percent_both (S J : ℝ) (hS : S = 0.9) (hJ : J = 0.8) : 
  ∃ B, B = S + J - 1 ∧ B = 0.7 :=
by
  sorry

end smallest_percent_both_l6_6848


namespace tetrahedron_altitude_exsphere_eq_l6_6562

variable (h₁ h₂ h₃ h₄ r₁ r₂ r₃ r₄ : ℝ)

/-- The equality of the sum of the reciprocals of the heights and the radii of the exspheres of 
a tetrahedron -/
theorem tetrahedron_altitude_exsphere_eq :
  2 * (1 / h₁ + 1 / h₂ + 1 / h₃ + 1 / h₄) = (1 / r₁ + 1 / r₂ + 1 / r₃ + 1 / r₄) :=
sorry

end tetrahedron_altitude_exsphere_eq_l6_6562


namespace fraction_addition_l6_6879

theorem fraction_addition : (3/4) / (5/8) + (1/2) = 17/10 := by
  sorry

end fraction_addition_l6_6879


namespace total_birds_on_fence_l6_6735

variable (initial_birds : ℕ := 1)
variable (added_birds : ℕ := 4)

theorem total_birds_on_fence : initial_birds + added_birds = 5 := by
  sorry

end total_birds_on_fence_l6_6735


namespace dima_story_telling_l6_6386

theorem dima_story_telling (initial_spoons final_spoons : ℕ) 
  (h1 : initial_spoons = 26) (h2 : final_spoons = 33696)
  (h3 : ∃ (n : ℕ), final_spoons = initial_spoons * (2^5 * 3^4) * 13) : 
  ∃ n : ℕ, n = 9 := 
sorry

end dima_story_telling_l6_6386


namespace roots_polynomial_l6_6441

theorem roots_polynomial (n r s : ℚ) (c d : ℚ)
  (h1 : c * c - n * c + 3 = 0)
  (h2 : d * d - n * d + 3 = 0)
  (h3 : (c + 1/d) * (d + 1/c) = s)
  (h4 : c * d = 3) :
  s = 16/3 :=
by
  sorry

end roots_polynomial_l6_6441


namespace last_three_digits_of_5_power_15000_l6_6268

theorem last_three_digits_of_5_power_15000:
  (5^15000) % 1000 = 1 % 1000 :=
by
  have h : 5^500 % 1000 = 1 % 1000 := by sorry
  sorry

end last_three_digits_of_5_power_15000_l6_6268


namespace jake_has_one_more_balloon_than_allan_l6_6614

def balloons_allan : ℕ := 6
def balloons_jake_initial : ℕ := 3
def balloons_jake_additional : ℕ := 4

theorem jake_has_one_more_balloon_than_allan :
  (balloons_jake_initial + balloons_jake_additional - balloons_allan) = 1 :=
by
  sorry

end jake_has_one_more_balloon_than_allan_l6_6614


namespace find_two_digit_number_l6_6476

theorem find_two_digit_number :
  ∃ x y : ℕ, 10 * x + y = 78 ∧ 10 * x + y < 100 ∧ y ≠ 0 ∧ (10 * x + y) / y = 9 ∧ (10 * x + y) % y = 6 :=
by
  sorry

end find_two_digit_number_l6_6476


namespace smallest_a_for_x4_plus_a2_not_prime_l6_6800

theorem smallest_a_for_x4_plus_a2_not_prime :
  ∀ a : ℕ, (∀ x : ℤ, ¬Nat.Prime (x^4 + a^2)) → a = 9 :=
begin
  sorry
end

end smallest_a_for_x4_plus_a2_not_prime_l6_6800


namespace clock_correct_after_240_days_l6_6316

theorem clock_correct_after_240_days (days : ℕ) (minutes_fast_per_day : ℕ) (hours_to_be_correct : ℕ) 
  (h1 : minutes_fast_per_day = 3) (h2 : hours_to_be_correct = 12) : 
  (days * minutes_fast_per_day) % (hours_to_be_correct * 60) = 0 :=
by 
  -- Proof skipped
  sorry

end clock_correct_after_240_days_l6_6316


namespace digit_proportions_l6_6285

theorem digit_proportions (n : ℕ) :
  (∃ (n1 n2 n5 nother : ℕ),
    n1 = n / 2 ∧
    n2 = n / 5 ∧
    n5 = n / 5 ∧
    nother = n / 10 ∧
    n1 + n2 + n5 + nother = n) ↔ n = 10 :=
by
  sorry

end digit_proportions_l6_6285


namespace range_of_x_for_y1_gt_y2_l6_6266

noncomputable def y1 (x : ℝ) : ℝ := x - 3
noncomputable def y2 (x : ℝ) : ℝ := 4 / x

theorem range_of_x_for_y1_gt_y2 :
  ∀ x : ℝ, (y1 x > y2 x) ↔ ((-1 < x ∧ x < 0) ∨ (x > 4)) := by
  sorry

end range_of_x_for_y1_gt_y2_l6_6266


namespace system_solutions_are_equivalent_l6_6671

theorem system_solutions_are_equivalent :
  ∀ (a b x y : ℝ),
  (2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9) ∧
  (a = 8.3 ∧ b = 1.2) ∧
  (x + 2 = a ∧ y - 1 = b) →
  x = 6.3 ∧ y = 2.2 :=
by
  -- Sorry is added intentionally to skip the proof
  sorry

end system_solutions_are_equivalent_l6_6671


namespace cone_height_l6_6320

theorem cone_height 
  (sector_radius : ℝ) 
  (central_angle : ℝ) 
  (sector_radius_eq : sector_radius = 3) 
  (central_angle_eq : central_angle = 2 * π / 3) : 
  ∃ h : ℝ, h = 2 * Real.sqrt 2 :=
by
  -- Formalize conditions
  let r := 1
  let l := sector_radius
  let θ := central_angle

  -- Combine conditions
  have r_eq : r = 1 := by sorry

  -- Calculate height using Pythagorean theorem
  let h := (l^2 - r^2).sqrt

  use h
  have h_eq : h = 2 * Real.sqrt 2 := by sorry
  exact h_eq

end cone_height_l6_6320


namespace alyssa_total_games_l6_6770

def calc_total_games (games_this_year games_last_year games_next_year : ℕ) : ℕ :=
  games_this_year + games_last_year + games_next_year

theorem alyssa_total_games :
  calc_total_games 11 13 15 = 39 :=
by
  -- Proof goes here
  sorry

end alyssa_total_games_l6_6770


namespace find_common_ratio_l6_6550

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

axiom a2 : a 2 = 9
axiom a3_plus_a4 : a 3 + a 4 = 18
axiom q_not_one : q ≠ 1

-- Proof problem
theorem find_common_ratio
  (h : is_geometric_sequence a q)
  (ha2 : a 2 = 9)
  (ha3a4 : a 3 + a 4 = 18)
  (hq : q ≠ 1) :
  q = -2 :=
sorry

end find_common_ratio_l6_6550


namespace find_a_from_roots_l6_6270

theorem find_a_from_roots (θ : ℝ) (a : ℝ) (h1 : ∀ x : ℝ, 4 * x^2 + 2 * a * x + a = 0 → (x = Real.sin θ ∨ x = Real.cos θ)) :
  a = 1 - Real.sqrt 5 :=
by
  sorry

end find_a_from_roots_l6_6270


namespace remainder_of_3_pow_99_plus_5_mod_9_l6_6747

theorem remainder_of_3_pow_99_plus_5_mod_9 : (3 ^ 99 + 5) % 9 = 5 := by
  -- Here we state the main goal
  sorry -- Proof to be filled in

end remainder_of_3_pow_99_plus_5_mod_9_l6_6747


namespace sphere_volume_ratio_l6_6059

theorem sphere_volume_ratio (r1 r2 : ℝ) (S1 S2 V1 V2 : ℝ) 
(h1 : S1 = 4 * Real.pi * r1^2)
(h2 : S2 = 4 * Real.pi * r2^2)
(h3 : V1 = (4 / 3) * Real.pi * r1^3)
(h4 : V2 = (4 / 3) * Real.pi * r2^3)
(h_surface_ratio : S1 / S2 = 2 / 3) :
V1 / V2 = (2 * Real.sqrt 6) / 9 :=
by
  sorry

end sphere_volume_ratio_l6_6059


namespace seashells_total_l6_6706

theorem seashells_total {sally tom jessica : ℕ} (h₁ : sally = 9) (h₂ : tom = 7) (h₃ : jessica = 5) : sally + tom + jessica = 21 := by
  sorry

end seashells_total_l6_6706


namespace floor_of_neg_five_thirds_l6_6796

theorem floor_of_neg_five_thirds : Int.floor (-5/3 : ℝ) = -2 := 
by 
  sorry

end floor_of_neg_five_thirds_l6_6796


namespace find_fx_l6_6965

theorem find_fx (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f (-x) = -(2 * x - 3)) 
  (h2 : ∀ x < 0, -f x = f (-x)) :
  ∀ x < 0, f x = 2 * x + 3 :=
by
  sorry

end find_fx_l6_6965


namespace calculate_8b_l6_6647

-- Define the conditions \(6a + 3b = 0\), \(b - 3 = a\), and \(b + c = 5\)
variables (a b c : ℝ)

theorem calculate_8b :
  (6 * a + 3 * b = 0) → (b - 3 = a) → (b + c = 5) → (8 * b = 16) :=
by
  intros h1 h2 h3
  -- Proof goes here, but we will use sorry to skip the proof.
  sorry

end calculate_8b_l6_6647


namespace find_value_of_a_l6_6731

theorem find_value_of_a (a : ℝ) (h : ( (-2 - (2 * a - 1)) / (3 - (-2)) = -1 )) : a = 2 :=
sorry

end find_value_of_a_l6_6731


namespace worth_of_each_gift_l6_6673

def workers_per_block : Nat := 200
def total_amount_for_gifts : Nat := 6000
def number_of_blocks : Nat := 15

theorem worth_of_each_gift (workers_per_block : Nat) (total_amount_for_gifts : Nat) (number_of_blocks : Nat) : 
  (total_amount_for_gifts / (workers_per_block * number_of_blocks)) = 2 := 
by 
  sorry

end worth_of_each_gift_l6_6673


namespace algebraic_expression_evaluation_l6_6535

theorem algebraic_expression_evaluation (x y : ℝ) (h : 2 * x - y + 1 = 3) : 4 * x - 2 * y + 5 = 9 := 
by
  sorry

end algebraic_expression_evaluation_l6_6535


namespace y_in_terms_of_x_l6_6051

theorem y_in_terms_of_x (p x y : ℝ) (h1 : x = 2 + 2^p) (h2 : y = 1 + 2^(-p)) : 
  y = (x-1)/(x-2) :=
by
  sorry

end y_in_terms_of_x_l6_6051


namespace find_x_l6_6656

def vector := ℝ × ℝ

def a : vector := (1, 1)
def b (x : ℝ) : vector := (2, x)

def vector_add (u v : vector) : vector :=
(u.1 + v.1, u.2 + v.2)

def scalar_mul (k : ℝ) (v : vector) : vector :=
(k * v.1, k * v.2)

def vector_sub (u v : vector) : vector :=
(u.1 - v.1, u.2 - v.2)

def are_parallel (u v : vector) : Prop :=
∃ k : ℝ, u = scalar_mul k v

theorem find_x (x : ℝ) : are_parallel (vector_add a (b x)) (vector_sub (scalar_mul 4 (b x)) (scalar_mul 2 a)) → x = 2 :=
by
  sorry

end find_x_l6_6656


namespace largest_angle_of_triangle_l6_6581

theorem largest_angle_of_triangle (x : ℝ) 
  (h1 : 35 + 70 + x = 180) : 75 = max (max 35 70) x := 
sorry

end largest_angle_of_triangle_l6_6581


namespace exponent_calculation_l6_6919

theorem exponent_calculation :
  ((19 ^ 11) / (19 ^ 8) * (19 ^ 3) = 47015881) :=
by
  sorry

end exponent_calculation_l6_6919


namespace find_b_l6_6074

open_locale matrix

def a : ℝ^3 := ![5, -3, -6]
def c : ℝ^3 := ![-3, -2, 3]
def b : ℝ^3 := ![-1, -3/4, 3/4]

theorem find_b (h1 : ∃ k : ℝ, b = k • a ∨ b = k • c)
(h2: inner_product_space.angle a b = inner_product_space.angle b c):
  b = ![-1, -3/4, 3/4] :=
sorry

end find_b_l6_6074


namespace set_diff_example_l6_6623

-- Definitions of sets A and B
def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 3, 4}

-- Definition of set difference
def set_diff (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- The mathematically equivalent proof problem statement
theorem set_diff_example :
  set_diff A B = {2} :=
sorry

end set_diff_example_l6_6623


namespace max_problems_missed_to_pass_l6_6914

theorem max_problems_missed_to_pass (total_problems : ℕ) (min_percentage : ℚ) 
  (h_total_problems : total_problems = 40) 
  (h_min_percentage : min_percentage = 0.85) : 
  ∃ max_missed : ℕ, max_missed = total_problems - ⌈total_problems * min_percentage⌉₊ ∧ max_missed = 6 :=
by
  sorry

end max_problems_missed_to_pass_l6_6914


namespace max_dot_product_on_circle_l6_6942

theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ) (O : ℝ × ℝ) (A : ℝ × ℝ),
  O = (0, 0) →
  A = (-2, 0) →
  P.1 ^ 2 + P.2 ^ 2 = 1 →
  let AO := (2, 0)
  let AP := (P.1 + 2, P.2)
  ∃ α : ℝ, P = (Real.cos α, Real.sin α) ∧ 
  ∃ max_val : ℝ, max_val = 6 ∧ 
  (2 * (Real.cos α + 2) = max_val) :=
by
  intro P O A hO hA hP 
  let AO := (2, 0)
  let AP := (P.1 + 2, P.2)
  sorry

end max_dot_product_on_circle_l6_6942


namespace least_number_subtracted_to_divisible_by_10_l6_6352

def least_subtract_to_divisible_by_10 (n : ℕ) : ℕ :=
  let last_digit := n % 10
  10 - last_digit

theorem least_number_subtracted_to_divisible_by_10 (n : ℕ) : (n = 427751) → ((n - least_subtract_to_divisible_by_10 n) % 10 = 0) :=
by
  intros h
  sorry

end least_number_subtracted_to_divisible_by_10_l6_6352


namespace minimum_correct_answers_l6_6542

/-
There are a total of 20 questions. Answering correctly scores 10 points, while answering incorrectly or not answering deducts 5 points. 
To pass, one must score no less than 80 points. Xiao Ming passed the selection. Prove that the minimum number of questions Xiao Ming 
must have answered correctly is no less than 12.
-/

theorem minimum_correct_answers (total_questions correct_points incorrect_points pass_score : ℕ)
  (h1 : total_questions = 20)
  (h2 : correct_points = 10)
  (h3 : incorrect_points = 5)
  (h4 : pass_score = 80)
  (h_passed : ∃ x : ℕ, x ≤ total_questions ∧ (correct_points * x - incorrect_points * (total_questions - x)) ≥ pass_score) :
  ∃ x : ℕ, x ≥ 12 ∧ (correct_points * x - incorrect_points * (total_questions - x)) ≥ pass_score := 
sorry

end minimum_correct_answers_l6_6542


namespace largest_common_term_in_sequences_l6_6006

/-- An arithmetic sequence starts with 3 and has a common difference of 10. A second sequence starts
with 5 and has a common difference of 8. In the range of 1 to 150, the largest number common to 
both sequences is 133. -/
theorem largest_common_term_in_sequences : ∃ (b : ℕ), b < 150 ∧ (∃ (n m : ℤ), b = 3 + 10 * n ∧ b = 5 + 8 * m) ∧ (b = 133) := 
by
  sorry

end largest_common_term_in_sequences_l6_6006


namespace fraction_is_integer_l6_6529

theorem fraction_is_integer (b t : ℤ) (hb : b ≠ 1) :
  ∃ (k : ℤ), (t^5 - 5 * b + 4) = k * (b^2 - 2 * b + 1) :=
by 
  sorry

end fraction_is_integer_l6_6529


namespace regular_polygon_sides_l6_6150

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6150


namespace sundae_cost_l6_6048

def ice_cream_cost := 2.00
def topping_cost := 0.50
def number_of_toppings := 10

theorem sundae_cost : ice_cream_cost + topping_cost * number_of_toppings = 7.00 := 
by
  sorry

end sundae_cost_l6_6048


namespace flower_profit_equation_l6_6119

theorem flower_profit_equation
  (initial_plants : ℕ := 3)
  (initial_profit_per_plant : ℕ := 10)
  (decrease_in_profit_per_additional_plant : ℕ := 1)
  (target_profit_per_pot : ℕ := 40)
  (x : ℕ) :
  (initial_plants + x) * (initial_profit_per_plant - x) = target_profit_per_pot :=
sorry

end flower_profit_equation_l6_6119


namespace quadratic_has_two_distinct_real_roots_l6_6722

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + m * x₁ - 8 = 0) ∧ (x₂^2 + m * x₂ - 8 = 0) :=
by
  let Δ := m^2 + 32
  have hΔ : Δ > 0 := by
    simp [Δ]
    exact add_pos_of_nonneg_of_pos (sq_nonneg m) (by norm_num)
  sorry

end quadratic_has_two_distinct_real_roots_l6_6722


namespace exists_fixed_point_subset_l6_6981

-- Definitions of set and function f with the required properties
variable {α : Type} [DecidableEq α]
variable (H : Finset α)
variable (f : Finset α → Finset α)

-- Conditions
axiom increasing_mapping (X Y : Finset α) : X ⊆ Y → f X ⊆ f Y
axiom range_in_H (X : Finset α) : f X ⊆ H

-- Statement to prove
theorem exists_fixed_point_subset : ∃ H₀ ⊆ H, f H₀ = H₀ :=
sorry

end exists_fixed_point_subset_l6_6981


namespace orange_harvest_exists_l6_6016

theorem orange_harvest_exists :
  ∃ (A B C D : ℕ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ A + B + C + D = 56 :=
by
  use 10
  use 15
  use 16
  use 15
  repeat {split};
  sorry

end orange_harvest_exists_l6_6016


namespace tens_of_80_tens_of_190_l6_6756

def tens_place (n : Nat) : Nat :=
  (n / 10) % 10

theorem tens_of_80 : tens_place 80 = 8 := 
  by
  sorry

theorem tens_of_190 : tens_place 190 = 9 := 
  by
  sorry

end tens_of_80_tens_of_190_l6_6756


namespace pen_cost_l6_6697

theorem pen_cost (x : ℝ) (h1 : 5 * x + x = 24) : x = 4 :=
by
  sorry

end pen_cost_l6_6697


namespace fixed_fee_1430_l6_6776

def fixed_monthly_fee (f p : ℝ) : Prop :=
  f + p = 20.60 ∧ f + 3 * p = 33.20

theorem fixed_fee_1430 (f p: ℝ) (h : fixed_monthly_fee f p) : 
  f = 14.30 :=
by
  sorry

end fixed_fee_1430_l6_6776


namespace linda_original_savings_l6_6299

theorem linda_original_savings :
  ∃ S : ℝ, 
    (5 / 8) * S + (1 / 4) * S = 400 ∧
    (1 / 8) * S = 600 ∧
    S = 4800 :=
by
  sorry

end linda_original_savings_l6_6299


namespace angle_less_than_45_degree_among_30_vectors_l6_6826

theorem angle_less_than_45_degree_among_30_vectors :
  ∃ (u v : ℝ^3), u ≠ 0 ∧ v ≠ 0 ∧ ∠ u v < π / 4 :=
by
  -- We have 30 non-zero vectors
  assume vectors : Fin 30 → ℝ^3,
  -- Conditions stating all vectors are non-zero
  have non_zero_vectors : ∀ i, vectors i ≠ 0,
  sorry -- proof would go here

end angle_less_than_45_degree_among_30_vectors_l6_6826


namespace stewart_farm_sheep_count_l6_6584

theorem stewart_farm_sheep_count 
  (S H : ℕ) 
  (ratio : S * 7 = 4 * H)
  (food_per_horse : H * 230 = 12880) : 
  S = 32 := 
sorry

end stewart_farm_sheep_count_l6_6584


namespace intersection_M_N_l6_6655

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_M_N : M ∩ N = Set.Ico 1 3 := 
by
  sorry

end intersection_M_N_l6_6655


namespace solve_complex_eq_l6_6117

open Complex

theorem solve_complex_eq (z : ℂ) (h : (3 - 4 * I) * z = 5) : z = (3 / 5) + (4 / 5) * I :=
by
  sorry

end solve_complex_eq_l6_6117


namespace cakes_left_l6_6205

def cakes_yesterday : ℕ := 3
def baked_today : ℕ := 5
def sold_today : ℕ := 6

theorem cakes_left (cakes_yesterday baked_today sold_today : ℕ) : cakes_yesterday + baked_today - sold_today = 2 := by
  sorry

end cakes_left_l6_6205


namespace vector_subtraction_parallel_l6_6422

theorem vector_subtraction_parallel (t : ℝ) 
  (h_parallel : -1 / 2 = -3 / t) : 
  ( (-1 : ℝ), -3 ) - ( 2, t ) = (-3, -9) :=
by
  -- proof goes here
  sorry

end vector_subtraction_parallel_l6_6422


namespace relationship_between_A_and_B_l6_6552

noncomputable def f (x : ℝ) : ℝ := x^2

def A : Set ℝ := {x | f x = x}

def B : Set ℝ := {x | f (f x) = x}

theorem relationship_between_A_and_B : A ∩ B = A :=
by sorry

end relationship_between_A_and_B_l6_6552


namespace sum_of_tripled_numbers_l6_6866

theorem sum_of_tripled_numbers (a b S : ℤ) (h : a + b = S) : 3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_tripled_numbers_l6_6866


namespace regular_polygon_sides_l6_6190

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6190


namespace David_squats_l6_6246

theorem David_squats (h1: ∀ d z: ℕ, d = 3 * 58) : d = 174 :=
by
  sorry

end David_squats_l6_6246


namespace benny_birthday_money_l6_6216

def money_spent_on_gear : ℕ := 34
def money_left_over : ℕ := 33

theorem benny_birthday_money : money_spent_on_gear + money_left_over = 67 :=
by
  sorry

end benny_birthday_money_l6_6216


namespace cuboid_to_cube_surface_area_l6_6763

variable (h w l : ℝ)
variable (volume_decreases : 64 = w^3 - w^2 * h)

theorem cuboid_to_cube_surface_area 
  (h w l : ℝ) 
  (cube_condition : w = l ∧ h = w + 4)
  (volume_condition : w^2 * h - w^3 = 64) : 
  (6 * w^2 = 96) :=
by
  sorry

end cuboid_to_cube_surface_area_l6_6763


namespace problem_KMO_16_l6_6797

theorem problem_KMO_16
  (m : ℕ) (h_pos : m > 0) :
  (2^(m+1) + 1) ∣ (3^(2^m) + 1) ↔ Nat.Prime (2^(m+1) + 1) :=
by
  sorry

end problem_KMO_16_l6_6797


namespace value_of_c_l6_6298

noncomputable def f (x a b c : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

theorem value_of_c (a b c : ℤ) (ha: a ≠ 0) (hb: b ≠ 0) (hc: c ≠ 0)
  (hfa: f a a b c = a^3) (hfb: f b a b c = b^3) : c = 16 := by
    sorry

end value_of_c_l6_6298


namespace range_of_f_l6_6814

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f : Set.Icc (-(3 / 2)) 3 = Set.image f (Set.Icc 0 (Real.pi / 2)) :=
  sorry

end range_of_f_l6_6814


namespace calculate_expression_l6_6621

theorem calculate_expression :
  (1.99^2 - 1.98 * 1.99 + 0.99^2 = 1) :=
by
  sorry

end calculate_expression_l6_6621


namespace prob_only_one_passes_l6_6089

open Probability

axiom prob_A : ℝ
axiom prob_B : ℝ
axiom prob_A_val : prob_A = 1 / 2
axiom prob_B_val : prob_B = 1 / 3
axiom independent_events : is_independent prob_A prob_B

noncomputable def prob_C := 
  let prob_not_B := 1 - prob_B
  let prob_not_A := 1 - prob_A
  prob_A * prob_not_B + prob_not_A * prob_B

theorem prob_only_one_passes :
  prob_C = 1 / 2 :=
sorry

end prob_only_one_passes_l6_6089


namespace percentage_increase_twice_eq_16_64_l6_6863

theorem percentage_increase_twice_eq_16_64 (x : ℝ) (hx : (1 + x)^2 = 1 + 0.1664) : x = 0.08 :=
by
  sorry -- This is the placeholder for the proof.

end percentage_increase_twice_eq_16_64_l6_6863


namespace remainder_145_mul_155_div_12_l6_6343

theorem remainder_145_mul_155_div_12 : (145 * 155) % 12 = 11 := by
  sorry

end remainder_145_mul_155_div_12_l6_6343


namespace anne_cleans_in_12_hours_l6_6786

-- Define the rates of Bruce and Anne
variables (B A : ℝ)

-- Define the conditions of the problem
constants (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1)

theorem anne_cleans_in_12_hours (B A : ℝ) (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1) : 1 / A = 12 :=
  sorry

end anne_cleans_in_12_hours_l6_6786


namespace arithmetic_sqrt_of_4_eq_2_l6_6313

theorem arithmetic_sqrt_of_4_eq_2 (x : ℕ) (h : x^2 = 4) : x = 2 :=
sorry

end arithmetic_sqrt_of_4_eq_2_l6_6313


namespace coefficients_sum_l6_6660

theorem coefficients_sum (a0 a1 a2 a3 a4 : ℝ) (h : (1 - 2*x)^4 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) : 
  a0 + a4 = 17 :=
by
  sorry

end coefficients_sum_l6_6660


namespace parabola_focus_l6_6317

theorem parabola_focus (x y : ℝ) (h : y^2 = 8 * x) : (x, y) = (2, 0) :=
sorry

end parabola_focus_l6_6317


namespace teresa_jogged_distance_l6_6574

-- Define the conditions as Lean constants.
def teresa_speed : ℕ := 5 -- Speed in kilometers per hour
def teresa_time : ℕ := 5 -- Time in hours

-- Define the distance formula.
def teresa_distance (speed time : ℕ) : ℕ := speed * time

-- State the theorem.
theorem teresa_jogged_distance : teresa_distance teresa_speed teresa_time = 25 := by
  -- Proof is skipped using 'sorry'.
  sorry

end teresa_jogged_distance_l6_6574


namespace number_of_jars_good_for_sale_l6_6694

def numberOfGoodJars (initialCartons : Nat) (cartonsNotDelivered : Nat) (jarsPerCarton : Nat)
  (damagedJarsPerCarton : Nat) (numberOfDamagedCartons : Nat) (oneTotallyDamagedCarton : Nat) : Nat := 
  let deliveredCartons := initialCartons - cartonsNotDelivered
  let totalJars := deliveredCartons * jarsPerCarton
  let damagedJars := (damagedJarsPerCarton * numberOfDamagedCartons) + oneTotallyDamagedCarton
  totalJars - damagedJars

theorem number_of_jars_good_for_sale : 
  numberOfGoodJars 50 20 20 3 5 20 = 565 :=
by
  sorry

end number_of_jars_good_for_sale_l6_6694


namespace anne_cleans_in_12_hours_l6_6785

-- Define the rates of Bruce and Anne
variables (B A : ℝ)

-- Define the conditions of the problem
constants (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1)

theorem anne_cleans_in_12_hours (B A : ℝ) (cond1 : (B + A) * 4 = 1) (cond2 : (B + 2 * A) * 3 = 1) : 1 / A = 12 :=
  sorry

end anne_cleans_in_12_hours_l6_6785


namespace adam_students_in_10_years_l6_6375

-- Define the conditions
def teaches_per_year : Nat := 50
def first_year_students : Nat := 40
def years_teaching : Nat := 10

-- Define the total number of students Adam will teach in 10 years
def total_students (first_year: Nat) (rest_years: Nat) (students_per_year: Nat) : Nat :=
  first_year + (rest_years * students_per_year)

-- State the theorem
theorem adam_students_in_10_years :
  total_students first_year_students (years_teaching - 1) teaches_per_year = 490 :=
by
  sorry

end adam_students_in_10_years_l6_6375


namespace value_expression_l6_6601

theorem value_expression (x : ℝ) (h : x = 1 + Real.sqrt 2) : x^4 - 4 * x^3 + 4 * x^2 + 4 = 5 :=
by
  sorry

end value_expression_l6_6601


namespace regular_polygon_sides_l6_6147

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6147


namespace sum_of_absolute_values_of_coefficients_l6_6404

theorem sum_of_absolute_values_of_coefficients :
  ∀ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ),
  (∀ x : ℝ, (1 - 3 * x) ^ 9 = a + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 + a_6 * x ^ 6 + a_7 * x ^ 7 + a_8 * x ^ 8 + a_9 * x ^ 9) →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 4 ^ 9 :=
by
  intro a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 h
  sorry

end sum_of_absolute_values_of_coefficients_l6_6404


namespace new_boarders_day_scholars_ratio_l6_6326

theorem new_boarders_day_scholars_ratio
  (initial_boarders : ℕ)
  (initial_day_scholars : ℕ)
  (ratio_boarders_day_scholars : ℕ → ℕ → Prop)
  (additional_boarders : ℕ)
  (new_boarders : ℕ)
  (new_ratio : ℕ → ℕ → Prop)
  (r1 r2 : ℕ)
  (h1 : ratio_boarders_day_scholars 7 16)
  (h2 : initial_boarders = 560)
  (h3 : initial_day_scholars = 1280)
  (h4 : additional_boarders = 80)
  (h5 : new_boarders = initial_boarders + additional_boarders)
  (h6 : new_ratio new_boarders initial_day_scholars) :
  new_ratio r1 r2 → r1 = 1 ∧ r2 = 2 :=
by {
    sorry
}

end new_boarders_day_scholars_ratio_l6_6326


namespace arithmetic_mean_geometric_mean_l6_6993

theorem arithmetic_mean_geometric_mean (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end arithmetic_mean_geometric_mean_l6_6993


namespace doris_hourly_wage_l6_6501

-- Defining the conditions from the problem
def money_needed : ℕ := 1200
def weekday_hours_per_day : ℕ := 3
def saturday_hours_per_day : ℕ := 5
def weeks_needed : ℕ := 3
def weekdays_per_week : ℕ := 5
def saturdays_per_week : ℕ := 1

-- Calculating total hours worked by Doris in 3 weeks
def total_hours (w_hours: ℕ) (s_hours: ℕ) 
    (w_days : ℕ) (s_days : ℕ) (weeks : ℕ) : ℕ := 
    (w_days * w_hours + s_days * s_hours) * weeks

-- Defining the weekly work hours
def weekly_hours := total_hours weekday_hours_per_day saturday_hours_per_day weekdays_per_week saturdays_per_week 1

-- Result of hours worked in 3 weeks
def hours_worked_in_3_weeks := weekly_hours * weeks_needed

-- Define the proof task
theorem doris_hourly_wage : 
  (money_needed : ℕ) / (hours_worked_in_3_weeks : ℕ) = 20 := by 
  sorry

end doris_hourly_wage_l6_6501


namespace polyhedron_with_n_edges_l6_6508

noncomputable def construct_polyhedron_with_n_edges (n : ℤ) : Prop :=
  ∃ (k : ℤ) (m : ℤ), (k = 8 ∨ k = 9 ∨ k = 10) ∧ (n = k + 3 * m)

theorem polyhedron_with_n_edges (n : ℤ) (h : n ≥ 8) : 
  construct_polyhedron_with_n_edges n :=
sorry

end polyhedron_with_n_edges_l6_6508


namespace more_cats_than_dogs_l6_6917

theorem more_cats_than_dogs:
  ∃ (cats_before cats_after dogs: ℕ),
    cats_before = 28 ∧
    dogs = 18 ∧
    cats_after = cats_before - 3 ∧
    cats_after - dogs = 7 :=
by
  use 28, 25, 18
  split
  case left =>
    exact rfl
  case right =>
    split
    case left =>
      exact rfl
    case right =>
      split
      case left =>
        exact rfl
      case right =>
        exact rfl

end more_cats_than_dogs_l6_6917


namespace smallest_integer_y_l6_6344

theorem smallest_integer_y (y : ℤ) : (∃ y : ℤ, (y / 4 + 3 / 7 > 2 / 3)) ∧ ∀ z : ℤ, (z / 4 + 3 / 7 > 2 / 3) → (y ≤ z) :=
begin
  sorry
end

end smallest_integer_y_l6_6344


namespace quadratic_has_two_distinct_real_roots_l6_6729

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ a b : ℝ, a ≠ b ∧ (a ^ 2 + m * a - 8 = 0) ∧ (b ^ 2 + m * b - 8 = 0) :=
sorry

end quadratic_has_two_distinct_real_roots_l6_6729


namespace triangle_is_isosceles_l6_6037

theorem triangle_is_isosceles
  (α β γ x y z w : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α + β = x)
  (h3 : β + γ = y)
  (h4 : γ + α = z)
  (h5 : x + y + z + w = 360) : 
  (α = β ∧ β = γ) ∨ (α = γ ∧ γ = β) ∨ (β = α ∧ α = γ) := by
  sorry

end triangle_is_isosceles_l6_6037


namespace inequality_am_gm_l6_6407

theorem inequality_am_gm (a b : ℝ) (p q : ℝ) (h1: a > 0) (h2: b > 0) (h3: p > 1) (h4: q > 1) (h5 : 1/p + 1/q = 1) : 
  a^(1/p) * b^(1/q) ≤ a/p + b/q :=
by
  sorry

end inequality_am_gm_l6_6407


namespace find_magnitude_of_z_l6_6460

open Complex

theorem find_magnitude_of_z
    (z : ℂ)
    (h : z^4 = 80 - 96 * I) : abs z = 5^(3/4) :=
by sorry

end find_magnitude_of_z_l6_6460


namespace arithmetic_sequence_a9_l6_6067

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Assume arithmetic sequence: a(n) = a1 + (n-1)d
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (n : ℕ) : ℤ := a 1 + (n - 1) * d

-- Given conditions
axiom condition1 : arithmetic_sequence a d 5 + arithmetic_sequence a d 7 = 16
axiom condition2 : arithmetic_sequence a d 3 = 1

-- Prove that a₉ = 15
theorem arithmetic_sequence_a9 : arithmetic_sequence a d 9 = 15 := by
  sorry

end arithmetic_sequence_a9_l6_6067


namespace translate_parabola_l6_6335

noncomputable def f (x : ℝ) : ℝ := 3 * x^2

noncomputable def g (x : ℝ) : ℝ := 3 * (x - 1)^2 - 4

theorem translate_parabola (x : ℝ) : g x = 3 * (x - 1)^2 - 4 :=
by {
  -- proof would go here
  sorry
}

end translate_parabola_l6_6335


namespace weight_per_linear_foot_l6_6547

theorem weight_per_linear_foot 
  (length_of_log : ℕ) 
  (cut_length : ℕ) 
  (piece_weight : ℕ) 
  (h1 : length_of_log = 20) 
  (h2 : cut_length = length_of_log / 2) 
  (h3 : piece_weight = 1500) 
  (h4 : length_of_log / 2 = 10) 
  : piece_weight / cut_length = 150 := 
  by 
  sorry

end weight_per_linear_foot_l6_6547


namespace carlson_max_jars_l6_6222

theorem carlson_max_jars (n a k : ℕ) (h1 : a = 5 * k)
  (h2 : n = 9 * k)
  (total_weight_carlson : 13 * n)
  (total_weight_baby : n)
  (h3 : 13 * n - a = 8 * (n + a)) :
  ∃ (j : ℕ), j ≤ 23 :=
by sorry

end carlson_max_jars_l6_6222


namespace solve_expression_l6_6267

theorem solve_expression (x y z : ℚ)
  (h1 : 2 * x + 3 * y + z = 20)
  (h2 : x + 2 * y + 3 * z = 26)
  (h3 : 3 * x + y + 2 * z = 29) :
  12 * x^2 + 22 * x * y + 12 * y^2 + 12 * x * z + 12 * y * z + 12 * z^2 = (computed_value : ℚ) :=
by
  sorry

end solve_expression_l6_6267


namespace c_sq_minus_a_sq_divisible_by_48_l6_6961

theorem c_sq_minus_a_sq_divisible_by_48
  (a b c : ℤ) (h_ac : a < c) (h_eq : a^2 + c^2 = 2 * b^2) : 48 ∣ (c^2 - a^2) := 
  sorry

end c_sq_minus_a_sq_divisible_by_48_l6_6961


namespace inequality_xyz_l6_6853

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xyz / (x^3 + y^3 + xyz) + xyz / (y^3 + z^3 + xyz) + xyz / (z^3 + x^3 + xyz) ≤ 1) := by
  sorry

end inequality_xyz_l6_6853


namespace buffaloes_number_l6_6602

theorem buffaloes_number (B D : ℕ) 
  (h : 4 * B + 2 * D = 2 * (B + D) + 24) : 
  B = 12 :=
sorry

end buffaloes_number_l6_6602


namespace regular_polygon_sides_l6_6132

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l6_6132


namespace number_of_jars_good_for_sale_l6_6693

def numberOfGoodJars (initialCartons : Nat) (cartonsNotDelivered : Nat) (jarsPerCarton : Nat)
  (damagedJarsPerCarton : Nat) (numberOfDamagedCartons : Nat) (oneTotallyDamagedCarton : Nat) : Nat := 
  let deliveredCartons := initialCartons - cartonsNotDelivered
  let totalJars := deliveredCartons * jarsPerCarton
  let damagedJars := (damagedJarsPerCarton * numberOfDamagedCartons) + oneTotallyDamagedCarton
  totalJars - damagedJars

theorem number_of_jars_good_for_sale : 
  numberOfGoodJars 50 20 20 3 5 20 = 565 :=
by
  sorry

end number_of_jars_good_for_sale_l6_6693


namespace purely_imaginary_sufficient_but_not_necessary_l6_6669

theorem purely_imaginary_sufficient_but_not_necessary (a b : ℝ) (h : ¬(b = 0)) : 
  (a = 0 → p ∧ q) → (q ∧ ¬p) :=
by
  sorry

end purely_imaginary_sufficient_but_not_necessary_l6_6669


namespace rob_nickels_count_l6_6457

noncomputable def value_of_quarters (num_quarters : ℕ) : ℝ := num_quarters * 0.25
noncomputable def value_of_dimes (num_dimes : ℕ) : ℝ := num_dimes * 0.10
noncomputable def value_of_pennies (num_pennies : ℕ) : ℝ := num_pennies * 0.01
noncomputable def value_of_nickels (num_nickels : ℕ) : ℝ := num_nickels * 0.05

theorem rob_nickels_count :
  let quarters := 7
  let dimes := 3
  let pennies := 12
  let total := 2.42
  let nickels := 5
  value_of_quarters quarters + value_of_dimes dimes + value_of_pennies pennies + value_of_nickels nickels = total :=
by
  sorry

end rob_nickels_count_l6_6457


namespace fraction_non_throwers_left_handed_l6_6451

theorem fraction_non_throwers_left_handed (total_players : ℕ) (num_throwers : ℕ) (total_right_handed : ℕ) (all_throwers_right_handed : ∀ x, x < num_throwers → true) (num_right_handed := total_right_handed - num_throwers) (non_throwers := total_players - num_throwers) (num_left_handed := non_throwers - num_right_handed) : 
    total_players = 70 → 
    num_throwers = 40 → 
    total_right_handed = 60 → 
    (∃ f: ℚ, f = num_left_handed / non_throwers ∧ f = 1/3) := 
by {
  sorry
}

end fraction_non_throwers_left_handed_l6_6451


namespace dima_story_retelling_count_l6_6384

theorem dima_story_retelling_count :
  ∃ n, (26 * (2 ^ 5) * (3 ^ 4)) = 33696 ∧ n = 9 :=
by
  sorry

end dima_story_retelling_count_l6_6384


namespace factorize_expression_l6_6390

theorem factorize_expression (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) :=
by 
  sorry

end factorize_expression_l6_6390


namespace remainder_of_p_div_10_is_6_l6_6749

-- Define the problem
def a : ℕ := sorry -- a is a positive integer and a multiple of 2

-- Define p based on a
def p : ℕ := 4^a

-- The main goal is to prove the remainder when p is divided by 10 is 6
theorem remainder_of_p_div_10_is_6 (ha : a > 0 ∧ a % 2 = 0) : p % 10 = 6 := by
  sorry

end remainder_of_p_div_10_is_6_l6_6749


namespace total_number_of_coins_is_336_l6_6473

theorem total_number_of_coins_is_336 (N20 : ℕ) (N25 : ℕ) (total_value_rupees : ℚ)
    (h1 : N20 = 260) (h2 : total_value_rupees = 71) (h3 : 20 * N20 + 25 * N25 = 7100) :
    N20 + N25 = 336 :=
by
  sorry

end total_number_of_coins_is_336_l6_6473


namespace problem_sequence_sum_l6_6622

theorem problem_sequence_sum (a : ℤ) (h : 14 * a^2 + 7 * a = 135) : 7 * a + (a - 1) = 23 :=
by {
  sorry
}

end problem_sequence_sum_l6_6622


namespace find_m_n_l6_6801

theorem find_m_n (m n : ℕ) (hmn : m + 6 < n + 4)
  (median_cond : ((m + 2 + m + 6 + n + 4 + n + 5) / 7) = n + 2)
  (mean_cond : ((m + (m + 2) + (m + 6) + (n + 4) + (n + 5) + (2 * n - 1) + (2 * n + 2)) / 7) = n + 2) :
  m + n = 10 :=
sorry

end find_m_n_l6_6801


namespace palace_to_airport_distance_l6_6389

-- Let I be the distance from the palace to the airport
-- Let v be the speed of the Emir's car
-- Let t be the time taken to travel from the palace to the airport

theorem palace_to_airport_distance (v t I : ℝ) 
    (h1 : v = I / t) 
    (h2 : v + 20 = I / (t - 2 / 60)) 
    (h3 : v - 20 = I / (t + 3 / 60)) : 
    I = 20 := by
  sorry

end palace_to_airport_distance_l6_6389


namespace regular_polygon_sides_l6_6144

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6144


namespace total_marbles_left_is_correct_l6_6370

def marbles_left_after_removal : ℕ :=
  let red_initial := 80
  let blue_initial := 120
  let green_initial := 75
  let yellow_initial := 50
  let red_removed := red_initial / 4
  let blue_removed := 3 * (green_initial / 5)
  let green_removed := (green_initial * 3) / 10
  let yellow_removed := 25
  let red_left := red_initial - red_removed
  let blue_left := blue_initial - blue_removed
  let green_left := green_initial - green_removed
  let yellow_left := yellow_initial - yellow_removed
  red_left + blue_left + green_left + yellow_left

theorem total_marbles_left_is_correct :
  marbles_left_after_removal = 213 :=
  by
    sorry

end total_marbles_left_is_correct_l6_6370


namespace arithmetic_square_root_of_4_l6_6314

theorem arithmetic_square_root_of_4 : ∃ x : ℕ, x * x = 4 ∧ x = 2 := 
sorry

end arithmetic_square_root_of_4_l6_6314


namespace polygon_with_150_degree_interior_angles_has_12_sides_l6_6170

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l6_6170


namespace apples_eq_pears_l6_6833

-- Define the conditions
def apples_eq_oranges (a o : ℕ) : Prop := 4 * a = 6 * o
def oranges_eq_pears (o p : ℕ) : Prop := 5 * o = 3 * p

-- The main problem statement
theorem apples_eq_pears (a o p : ℕ) (h1 : apples_eq_oranges a o) (h2 : oranges_eq_pears o p) :
  24 * a = 21 * p :=
sorry

end apples_eq_pears_l6_6833


namespace digit_proportions_l6_6286

theorem digit_proportions (n : ℕ) :
  (∃ (n1 n2 n5 nother : ℕ),
    n1 = n / 2 ∧
    n2 = n / 5 ∧
    n5 = n / 5 ∧
    nother = n / 10 ∧
    n1 + n2 + n5 + nother = n) ↔ n = 10 :=
by
  sorry

end digit_proportions_l6_6286


namespace find_integers_a_b_c_l6_6496

theorem find_integers_a_b_c :
  ∃ (a b c : ℤ), (∀ (x : ℤ), (x - a) * (x - 8) + 4 = (x + b) * (x + c)) ∧ 
  (a = 20 ∨ a = 29) :=
 by {
      sorry 
}

end find_integers_a_b_c_l6_6496


namespace least_multiple_of_29_gt_500_l6_6743

theorem least_multiple_of_29_gt_500 : ∃ n : ℕ, n > 0 ∧ 29 * n > 500 ∧ 29 * n = 522 :=
by
  use 18
  sorry

end least_multiple_of_29_gt_500_l6_6743


namespace find_numbers_l6_6029

theorem find_numbers :
  ∃ a b : ℕ, a + b = 60 ∧ Nat.gcd a b + Nat.lcm a b = 84 :=
by
  sorry

end find_numbers_l6_6029


namespace trip_distance_1200_miles_l6_6845

theorem trip_distance_1200_miles
    (D : ℕ)
    (H : D / 50 - D / 60 = 4) :
    D = 1200 :=
by
    sorry

end trip_distance_1200_miles_l6_6845


namespace probability_blue_given_not_red_l6_6609

theorem probability_blue_given_not_red :
  let total_balls := 20
  let red_balls := 5
  let yellow_balls := 5
  let blue_balls := 10
  let non_red_balls := yellow_balls + blue_balls
  let blue_given_not_red := (blue_balls : ℚ) / non_red_balls
  blue_given_not_red = 2 / 3 := 
by
  sorry

end probability_blue_given_not_red_l6_6609


namespace infinite_series_sum_l6_6686

noncomputable def sum_geometric_series (a b : ℝ) (h : ∑' n : ℕ, a / b ^ (n + 1) = 3) : ℝ :=
  ∑' n : ℕ, a / b ^ (n + 1)

theorem infinite_series_sum (a b c : ℝ) (h : sum_geometric_series a b (by sorry) = 3) :
  ∑' n : ℕ, (c * a) / (a + b) ^ (n + 1) = 3 * c / 4 :=
sorry

end infinite_series_sum_l6_6686


namespace adam_age_is_8_l6_6910

variables (A : ℕ) -- Adam's current age
variable (tom_age : ℕ) -- Tom's current age
variable (combined_age : ℕ) -- Their combined age in 12 years

theorem adam_age_is_8 (h1 : tom_age = 12) -- Tom is currently 12 years old
                    (h2 : combined_age = 44) -- In 12 years, their combined age will be 44 years old
                    (h3 : A + 12 + (tom_age + 12) = combined_age) -- Equation representing the combined age in 12 years
                    : A = 8 :=
by
  sorry

end adam_age_is_8_l6_6910


namespace regular_polygon_sides_l6_6192

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6192


namespace yanna_afternoon_baking_l6_6349

noncomputable def butter_cookies_in_afternoon (B : ℕ) : Prop :=
  let biscuits_afternoon := 20
  let butter_cookies_morning := 20
  let biscuits_morning := 40
  (biscuits_afternoon = B + 30) → B = 20

theorem yanna_afternoon_baking (h : butter_cookies_in_afternoon 20) : 20 = 20 :=
by {
  sorry
}

end yanna_afternoon_baking_l6_6349


namespace anne_cleaning_time_l6_6781

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l6_6781


namespace third_trial_point_l6_6068

variable (a b : ℝ) (x₁ x₂ x₃ : ℝ)

axiom experimental_range : a = 2 ∧ b = 4
axiom method_0618 : ∀ x1 x2, (x1 = 2 + 0.618 * (4 - 2) ∧ x2 = 2 + (4 - x1)) ∨ 
                              (x1 = (2 + (4 - 3.236)) ∧ x2 = 3.236)
axiom better_result (x₁ x₂ : ℝ) : x₁ > x₂  -- Assuming better means strictly greater

axiom x1_value : x₁ = 3.236 ∨ x₁ = 2.764
axiom x2_value : x₂ = 2.764 ∨ x₂ = 3.236
axiom x3_cases : (x₃ = 4 - 0.618 * (4 - x₁)) ∨ (x₃ = 2 + (4 - x₂))

theorem third_trial_point : x₃ = 3.528 ∨ x₃ = 2.472 :=
by
  sorry

end third_trial_point_l6_6068


namespace travel_through_cities_l6_6760

theorem travel_through_cities (V : Type*) (E_highway E_railway E_rural : set (V × V))
  (G : SimpleGraph V) [DecidableRel G.Adj] :
  (∀ v : V, ∃ u w x : V, (v, u) ∈ E_highway ∧ (v, w) ∈ E_railway ∧ (v, x) ∈ E_rural) →
  (∀ v w : V, G.Adj v w → ∃ t : ℕ, (G.walk v w).steps.length = t) →
  (∀ v w : V, exists_walk v w) →
  ∃ (circuit : G.walk v v), circuit.is_eulerian :=
by
  intros h1 h2 h3
  sorry

end travel_through_cities_l6_6760


namespace count_homologous_functions_l6_6512

open Set Finset

def homologous_domain (s : Set ℤ) : Prop :=
  ∀ x, x^2 ∈ s ↔ x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2

theorem count_homologous_functions :
  let preimage := {x : ℤ | x^2 ∈ {1, 4}}
  let domains := {s : Finset ℤ | ∀ x, x ∈ s ↔ x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2}
  domains.card = 9 :=
by
  sorry

end count_homologous_functions_l6_6512


namespace sin_alpha_expression_l6_6636

theorem sin_alpha_expression (α : ℝ) 
  (h_tan : Real.tan α = -3 / 4) : 
  Real.sin α * (Real.sin α - Real.cos α) = 21 / 25 := 
sorry

end sin_alpha_expression_l6_6636


namespace angle_of_inclination_l6_6463

theorem angle_of_inclination (t : ℝ) (x y : ℝ) :
  (x = 1 + t * (Real.sin (Real.pi / 6))) ∧ 
  (y = 2 + t * (Real.cos (Real.pi / 6))) →
  ∃ α : ℝ, α = Real.arctan (Real.sqrt 3) ∧ (0 ≤ α ∧ α < Real.pi) := 
by 
  sorry

end angle_of_inclination_l6_6463


namespace regular_polygon_sides_l6_6175

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l6_6175


namespace regular_polygon_sides_l6_6141

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6141


namespace Carlson_initial_jars_max_count_l6_6231

def initial_jar_weight_ratio (c_initial_weight b_initial_weight: ℕ) : Prop := 
  c_initial_weight = 13 * b_initial_weight

def new_jar_weight_ratio (c_new_weight b_new_weight: ℕ) : Prop := 
  c_new_weight = 8 * b_new_weight

theorem Carlson_initial_jars_max_count (c_initial_weight b_initial_weight c_new_weight b_new_weight: ℕ) 
  (h1 : initial_jar_weight_ratio c_initial_weight b_initial_weight) 
  (h2 : new_jar_weight_ratio c_new_weight b_new_weight)
  (h3 : ∀ a: ℕ, c_new_weight = c_initial_weight - a ∧ b_new_weight = b_initial_weight + a) :
  ∃ n: ℕ, n ≤ 23 :=
begin
  sorry,
end

end Carlson_initial_jars_max_count_l6_6231


namespace pond_width_l6_6541

theorem pond_width
  (L : ℝ) (D : ℝ) (V : ℝ) (W : ℝ)
  (hL : L = 20)
  (hD : D = 5)
  (hV : V = 1000)
  (hVolume : V = L * W * D) :
  W = 10 :=
by {
  sorry
}

end pond_width_l6_6541


namespace cone_volume_l6_6920

theorem cone_volume (d h : ℝ) (V : ℝ) (hd : d = 12) (hh : h = 8) :
  V = (1 / 3) * Real.pi * (d / 2) ^ 2 * h → V = 96 * Real.pi :=
by
  rw [hd, hh]
  sorry

end cone_volume_l6_6920


namespace geometric_sequence_sum_of_first_five_l6_6043

theorem geometric_sequence_sum_of_first_five :
  (∃ (a : ℕ → ℝ) (r : ℝ),
    (∀ n, n > 0 → a n > 0) ∧
    a 2 = 2 ∧
    a 4 = 8 ∧
    r = 2 ∧
    a 1 = 1 ∧
    a 3 = a 1 * r^2 ∧
    a 5 = a 1 * r^4 ∧
    (a 1 + a 2 + a 3 + a 4 + a 5 = 31)
  ) :=
sorry

end geometric_sequence_sum_of_first_five_l6_6043


namespace dilation_and_rotation_l6_6397

-- Definitions translating the conditions
def dilation_matrix (s : ℝ) : matrix (fin 2) (fin 2) ℝ := ![![s, 0], ![0, s]]
def rotation_matrix_90_ccw : matrix (fin 2) (fin 2) ℝ := ![![0, -1], ![1, 0]]

-- Combined transformation matrix
def combined_transformation_matrix (s : ℝ) : matrix (fin 2) (fin 2) ℝ := 
  (rotation_matrix_90_ccw ⬝ dilation_matrix s : matrix (fin 2) (fin 2) ℝ)

-- Theorem statement
theorem dilation_and_rotation (s : ℝ) (h : s = 4) :
  combined_transformation_matrix s = ![![0, -4], ![4, 0]] :=
sorry

end dilation_and_rotation_l6_6397


namespace special_lines_count_l6_6899

noncomputable def count_special_lines : ℕ :=
  sorry

theorem special_lines_count :
  count_special_lines = 3 :=
by sorry

end special_lines_count_l6_6899


namespace regular_polygon_sides_l6_6199

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6199


namespace cos_third_quadrant_l6_6670

theorem cos_third_quadrant (B : ℝ) (hB : -π < B ∧ B < -π / 2) (sin_B : Real.sin B = 5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end cos_third_quadrant_l6_6670


namespace lowest_score_to_average_90_l6_6681

theorem lowest_score_to_average_90 {s1 s2 s3 max_score avg_score : ℕ} 
    (h1: s1 = 88) 
    (h2: s2 = 96) 
    (h3: s3 = 105) 
    (hmax: max_score = 120) 
    (havg: avg_score = 90) 
    : ∃ s4 s5, s4 ≤ max_score ∧ s5 ≤ max_score ∧ (s1 + s2 + s3 + s4 + s5) / 5 = avg_score ∧ (min s4 s5 = 41) :=
by {
    sorry
}

end lowest_score_to_average_90_l6_6681


namespace x_value_not_unique_l6_6493

theorem x_value_not_unique (x y : ℝ) (h1 : y = x) (h2 : y = (|x + y - 2|) / (Real.sqrt 2)) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
(∃ y1 y2 : ℝ, (y1 = x1 ∧ y2 = x2 ∧ y1 = (|x1 + y1 - 2|) / Real.sqrt 2 ∧ y2 = (|x2 + y2 - 2|) / Real.sqrt 2)) :=
by
  sorry

end x_value_not_unique_l6_6493


namespace radius_of_ball_is_13_l6_6366

-- Define the conditions
def hole_radius : ℝ := 12
def hole_depth : ℝ := 8

-- The statement to prove
theorem radius_of_ball_is_13 : (∃ x : ℝ, x^2 + hole_radius^2 = (x + hole_depth)^2) → x + hole_depth = 13 :=
by
  sorry

end radius_of_ball_is_13_l6_6366


namespace carlson_max_jars_l6_6238

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l6_6238


namespace BR_squared_is_160_17_final_result_l6_6072

noncomputable def square_side_length := 4
noncomputable def point_B := (4 : ℝ, 4 : ℝ)
noncomputable def point_A := (0 : ℝ, 4 : ℝ)
noncomputable def point_D := (0 : ℝ, 0 : ℝ)
noncomputable def point_C := (4 : ℝ, 0 : ℝ)
noncomputable def BP := 3
noncomputable def BQ := 1
noncomputable def point_P := (1 : ℝ, 4 : ℝ)
noncomputable def point_Q := (4 : ℝ, 3 : ℝ)

noncomputable def point_R : ℝ × ℝ := 
  let x := 16 / 17
  let y := 64 / 17
  (x, y)

noncomputable def BR_sq : ℝ := 
  let (bx, by) := point_B
  let (rx, ry) := point_R
  ((bx - rx)^2 + (by - ry)^2)

theorem BR_squared_is_160_17 : BR_sq = 160 / 17 := by
  -- calculation steps skipped
  sorry

theorem final_result : 160 + 17 = 177 := by
  -- calculation steps skipped
  sorry

end BR_squared_is_160_17_final_result_l6_6072


namespace regular_polygon_num_sides_l6_6157

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l6_6157


namespace intersection_M_N_l6_6984

-- Define the sets M and N
def M : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }
def N : Set ℝ := { x : ℝ | 1 < x }

-- State the problem in terms of Lean definitions and theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_M_N_l6_6984


namespace bertha_descendants_no_children_l6_6217

-- Definitions based on the conditions of the problem.
def bertha_daughters : ℕ := 10
def total_descendants : ℕ := 40
def granddaughters : ℕ := total_descendants - bertha_daughters
def daughters_with_children : ℕ := 8
def children_per_daughter_with_children : ℕ := 4
def number_of_granddaughters : ℕ := daughters_with_children * children_per_daughter_with_children
def total_daughters_and_granddaughters : ℕ := bertha_daughters + number_of_granddaughters
def without_children : ℕ := total_daughters_and_granddaughters - daughters_with_children

-- Lean statement to prove the main question given the definitions.
theorem bertha_descendants_no_children : without_children = 34 := by
  -- Placeholder for the proof
  sorry

end bertha_descendants_no_children_l6_6217


namespace division_remainder_l6_6745

theorem division_remainder :
  (1225 * 1227 * 1229) % 12 = 3 :=
by sorry

end division_remainder_l6_6745


namespace arc_length_l6_6034

theorem arc_length (C : ℝ) (theta : ℝ) (hC : C = 100) (htheta : theta = 30) :
  (theta / 360) * C = 25 / 3 :=
by sorry

end arc_length_l6_6034


namespace age_of_15th_student_l6_6603

theorem age_of_15th_student
  (avg_age_15_students : ℕ)
  (total_students : ℕ)
  (avg_age_5_students : ℕ)
  (students_5 : ℕ)
  (avg_age_9_students : ℕ)
  (students_9 : ℕ)
  (total_age_15_students_eq : avg_age_15_students * total_students = 225)
  (total_age_5_students_eq : avg_age_5_students * students_5 = 70)
  (total_age_9_students_eq : avg_age_9_students * students_9 = 144) :
  (avg_age_15_students * total_students - (avg_age_5_students * students_5 + avg_age_9_students * students_9) = 11) :=
by
  sorry

end age_of_15th_student_l6_6603


namespace pool_min_cost_l6_6624

noncomputable def CostMinimization (x : ℝ) : ℝ :=
  150 * 1600 + 720 * (x + 1600 / x)

theorem pool_min_cost :
  ∃ (x : ℝ), x = 40 ∧ CostMinimization x = 297600 :=
by
  sorry

end pool_min_cost_l6_6624


namespace exists_nat_lt_100_two_different_squares_l6_6015

theorem exists_nat_lt_100_two_different_squares :
  ∃ n : ℕ, n < 100 ∧ 
    ∃ a b c d : ℕ, a^2 + b^2 = n ∧ c^2 + d^2 = n ∧ (a ≠ c ∨ b ≠ d) ∧ a ≠ b ∧ c ≠ d :=
by
  sorry

end exists_nat_lt_100_two_different_squares_l6_6015


namespace midpoint_s2_l6_6567

structure Point where
  x : ℤ
  y : ℤ

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def translate (p : Point) (dx dy : ℤ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem midpoint_s2 :
  let s1_p1 := ⟨6, -2⟩
  let s1_p2 := ⟨-4, 6⟩
  let s1_mid := midpoint s1_p1 s1_p2
  let s2_mid_translated := translate s1_mid (-3) (-4)
  s2_mid_translated = ⟨-2, -2⟩ := 
by
  sorry

end midpoint_s2_l6_6567


namespace quadratic_roots_distinct_l6_6728

-- Define the quadratic equation condition
def quadratic_eq : (ℝ → ℝ) :=
  λ x m => x^2 + m * x - 8

-- State the problem
theorem quadratic_roots_distinct (m : ℝ) : 
  let Δ := m^2 + 32 in Δ > 0 :=
  by
   -- We need this theorem to state that the equation always has distinct real roots
  let Δ := m^2 + 32
  sorry

end quadratic_roots_distinct_l6_6728


namespace div_floor_factorial_l6_6297

theorem div_floor_factorial (n q : ℕ) (hn : n ≥ 5) (hq : 2 ≤ q ∧ q ≤ n) :
  q - 1 ∣ (Nat.floor ((Nat.factorial (n - 1)) / q : ℚ)) :=
by
  sorry

end div_floor_factorial_l6_6297


namespace graph_of_g_contains_1_0_and_sum_l6_6945

noncomputable def f : ℝ → ℝ := sorry

def g (x y : ℝ) : Prop := 3 * y = 2 * f (3 * x) + 4

theorem graph_of_g_contains_1_0_and_sum :
  f 3 = -2 → g 1 0 ∧ (1 + 0 = 1) :=
by
  intro h
  sorry

end graph_of_g_contains_1_0_and_sum_l6_6945


namespace find_n_l6_6798

def valid_n (n : ℕ) : Prop :=
  0 ≤ n ∧ n ≤ 14 ∧ n ≡ 10403 [MOD 15]

theorem find_n : ∃ n, valid_n n ∧ n = 8 :=
by
  sorry

end find_n_l6_6798


namespace isosceles_triangle_l6_6035

theorem isosceles_triangle
  (α β γ x y z w : ℝ)
  (h_triangle : α + β + γ = 180)
  (h_quad : x + y + z + w = 360)
  (h_conditions : (x = α + β) ∧ (y = β + γ) ∧ (z = γ + α) ∨ (w = α + β) ∧ (x = β + γ) ∧ (y = γ + α) ∨ (z = α + β) ∧ (w = β + γ) ∧ (x = γ + α) ∨ (y = α + β) ∧ (z = β + γ) ∧ (w = γ + α))
  : α = β ∨ β = γ ∨ γ = α := 
sorry

end isosceles_triangle_l6_6035


namespace select_16_genuine_coins_l6_6869

theorem select_16_genuine_coins (coins : Finset ℕ) (h_coins_count : coins.card = 40) 
  (counterfeit : Finset ℕ) (h_counterfeit_count : counterfeit.card = 3)
  (h_counterfeit_lighter : ∀ c ∈ counterfeit, ∀ g ∈ (coins \ counterfeit), c < g) :
  ∃ genuine : Finset ℕ, genuine.card = 16 ∧ 
    (∀ h1 h2 h3 : Finset ℕ, h1.card = 20 → h2.card = 10 → h3.card = 8 →
      ((h1 ⊆ coins ∧ h2 ⊆ h1 ∧ h3 ⊆ (h1 \ counterfeit)) ∨
       (h1 ⊆ coins ∧ h2 ⊆ (h1 \ counterfeit) ∧ h3 ⊆ (h2 \ counterfeit))) →
      genuine ⊆ coins \ counterfeit) :=
sorry

end select_16_genuine_coins_l6_6869


namespace solution_inequality_l6_6125

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function_at (f : ℝ → ℝ) (x : ℝ) : Prop := f (2 + x) = f (2 - x)
def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x < y → x ∈ s → y ∈ s → f x < f y

-- Main statement
theorem solution_inequality 
  (h1 : ∀ x, is_even_function_at f x)
  (h2 : is_increasing_on f {x : ℝ | x ≤ 2}) :
  (∀ a : ℝ, (a > -1) ∧ (a ≠ 0) ↔ f (a^2 + 3*a + 2) < f (a^2 - a + 2)) :=
by {
  sorry
}

end solution_inequality_l6_6125


namespace value_of_f_neg_4_l6_6642

noncomputable def f : ℝ → ℝ := λ x => if x ≥ 0 then Real.sqrt x else - (Real.sqrt (-x))

-- Definition that f is an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem value_of_f_neg_4 :
  isOddFunction f ∧ (∀ x, x ≥ 0 → f x = Real.sqrt x) → f (-4) = -2 := 
by
  sorry

end value_of_f_neg_4_l6_6642


namespace factorize_expression_l6_6391

variable (a b : ℝ) 

theorem factorize_expression : ab^2 - 9a = a * (b + 3) * (b - 3) := by
  sorry

end factorize_expression_l6_6391


namespace negation_of_forall_ge_implies_exists_lt_l6_6455

theorem negation_of_forall_ge_implies_exists_lt :
  ¬(∀ x : ℝ, x^2 + 1 ≥ 2 * x) ↔ ∃ x : ℝ, x^2 + 1 < 2 * x := by
  sorry

end negation_of_forall_ge_implies_exists_lt_l6_6455


namespace total_weight_of_2_meters_l6_6592

def tape_measure_length : ℚ := 5
def tape_measure_weight : ℚ := 29 / 8
def computer_length : ℚ := 4
def computer_weight : ℚ := 2.8

noncomputable def weight_per_meter_tape_measure : ℚ := tape_measure_weight / tape_measure_length
noncomputable def weight_per_meter_computer : ℚ := computer_weight / computer_length

noncomputable def total_weight : ℚ :=
  2 * weight_per_meter_tape_measure + 2 * weight_per_meter_computer

theorem total_weight_of_2_meters (h1 : tape_measure_length = 5)
    (h2 : tape_measure_weight = 29 / 8) 
    (h3 : computer_length = 4) 
    (h4 : computer_weight = 2.8): 
    total_weight = 57 / 20 := by 
  unfold total_weight
  sorry

end total_weight_of_2_meters_l6_6592


namespace regular_polygon_sides_l6_6151

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6151


namespace paula_shirts_count_l6_6453

variable {P : Type}

-- Given conditions as variable definitions
def initial_money : ℕ := 109
def shirt_cost : ℕ := 11
def pants_cost : ℕ := 13
def money_left : ℕ := 74
def money_spent : ℕ := initial_money - money_left
def shirts_count : ℕ → ℕ := λ S => shirt_cost * S

-- Main proposition to prove
theorem paula_shirts_count (S : ℕ) (h : money_spent = shirts_count S + pants_cost) : 
  S = 2 := by
  /- 
    Following the steps of the proof:
    1. Calculate money spent is $35.
    2. Set up the equation $11S + 13 = 35.
    3. Solve for S.
  -/
  sorry

end paula_shirts_count_l6_6453


namespace true_discount_correct_l6_6587

noncomputable def true_discount (FV BD : ℝ) : ℝ :=
  BD / (1 + (BD / FV))

theorem true_discount_correct
  (FV BD : ℝ)
  (hFV : FV = 2260)
  (hBD : BD = 428.21) :
  true_discount FV BD = 360.00 :=
by
  sorry

end true_discount_correct_l6_6587


namespace power_mod_equivalence_l6_6112

theorem power_mod_equivalence : (7^700) % 100 = 1 := 
by 
  -- Given that (7^4) % 100 = 1
  have h : 7^4 % 100 = 1 := by sorry
  -- Use this equivalence to prove the statement
  sorry

end power_mod_equivalence_l6_6112


namespace find_b_collinear_and_bisects_l6_6075

def a := (5 : ℤ, -3 : ℤ, -6 : ℤ)
def c := (-3 : ℤ, -2 : ℤ, 3 : ℤ)
def b := (1 : ℚ, -12/5 : ℚ, 3/5 : ℚ)

def collinear (a b c : α × α × α) [CommRing α] : Prop :=
  ∃ k : α, b = (a.1 + k * (c.1 - a.1), a.2 + k * (c.2 - a.2), a.3 + k * (c.3 - a.3))

def bisects_angle (a b c : ℚ × ℚ × ℚ) : Prop :=
  let dot_product (x y : ℚ × ℚ × ℚ) := x.1 * y.1 + x.2 * y.2 + x.3 * y.3
  let norm (x : ℚ × ℚ × ℚ) := real.sqrt (dot_product x x)
  (dot_product a b) / (norm a * norm b) = (dot_product b c) / (norm b * norm c)

theorem find_b_collinear_and_bisects :
  collinear a b c ∧ bisects_angle a b c :=
by
  sorry

end find_b_collinear_and_bisects_l6_6075


namespace exists_f_satisfying_iteration_l6_6626

-- Mathematically equivalent problem statement in Lean 4
theorem exists_f_satisfying_iteration :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, (f^[1995] n) = 2 * n :=
by
  -- Fill in proof here
  sorry

end exists_f_satisfying_iteration_l6_6626


namespace line_through_point_equal_intercepts_l6_6027

-- Definitions based on conditions
def passes_through (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l p.1 p.2

def equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a, a ≠ 0 ∧ (∀ x y, l x y ↔ x + y = a) ∨ (∀ x y, l x y ↔ y = 2 * x)

-- Theorem statement based on the problem
theorem line_through_point_equal_intercepts :
  ∃ l, passes_through (1, 2) l ∧ equal_intercepts l ∧
  (∀ x y, l x y ↔ 2 * x - y = 0) ∨ (∀ x y, l x y ↔ x + y - 3 = 0) :=
sorry

end line_through_point_equal_intercepts_l6_6027


namespace square_side_length_l6_6714

noncomputable def diagonal_in_inches : ℝ := 2 * Real.sqrt 2
noncomputable def inches_to_feet : ℝ := 1 / 12
noncomputable def diagonal_in_feet := diagonal_in_inches * inches_to_feet
noncomputable def factor_sqrt_2 : ℝ := 1 / Real.sqrt 2

theorem square_side_length :
  let diagonal_feet := diagonal_in_feet 
  let side_length_feet := diagonal_feet * factor_sqrt_2
  side_length_feet = 1 / 6 :=
sorry

end square_side_length_l6_6714


namespace where_to_place_minus_sign_l6_6544

theorem where_to_place_minus_sign :
  (6 + 9 + 12 + 15 + 18 + 21 - 2 * 18) = 45 :=
by
  sorry

end where_to_place_minus_sign_l6_6544


namespace regular_polygon_sides_l6_6134

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l6_6134


namespace volume_ratio_l6_6099

noncomputable def salinity_bay (salt_bay volume_bay : ℝ) : ℝ :=
  salt_bay / volume_bay

noncomputable def salinity_sea_excluding_bay (salt_sea_excluding_bay volume_sea_excluding_bay : ℝ) : ℝ :=
  salt_sea_excluding_bay / volume_sea_excluding_bay

noncomputable def salinity_whole_sea (salt_sea volume_sea : ℝ) : ℝ :=
  salt_sea / volume_sea

theorem volume_ratio (salt_bay volume_bay salt_sea_excluding_bay volume_sea_excluding_bay : ℝ) 
  (h_bay : salinity_bay salt_bay volume_bay = 240 / 1000)
  (h_sea_excluding_bay : salinity_sea_excluding_bay salt_sea_excluding_bay volume_sea_excluding_bay = 110 / 1000)
  (h_whole_sea : salinity_whole_sea (salt_bay + salt_sea_excluding_bay) (volume_bay + volume_sea_excluding_bay) = 120 / 1000) :
  (volume_bay + volume_sea_excluding_bay) / volume_bay = 13 := 
sorry

end volume_ratio_l6_6099


namespace hawks_points_l6_6062

theorem hawks_points (E H : ℕ) (h₁ : E + H = 82) (h₂ : E = H + 18) (h₃ : H ≥ 9) : H = 32 :=
sorry

end hawks_points_l6_6062


namespace largest_even_integer_sum_l6_6586

theorem largest_even_integer_sum (x : ℤ) (h : (20 * (x + x + 38) / 2) = 6400) : 
  x + 38 = 339 :=
sorry

end largest_even_integer_sum_l6_6586


namespace fraction_subtraction_l6_6219

theorem fraction_subtraction :
  (9 / 19) - (5 / 57) - (2 / 38) = 1 / 3 := by
sorry

end fraction_subtraction_l6_6219


namespace sphere_volume_from_area_l6_6646

/-- Given the surface area of a sphere is 24π, prove that the volume of the sphere is 8√6π. -/ 
theorem sphere_volume_from_area :
  ∀ {R : ℝ},
    4 * Real.pi * R^2 = 24 * Real.pi →
    (4 / 3) * Real.pi * R^3 = 8 * Real.sqrt 6 * Real.pi :=
by
  intro R h
  sorry

end sphere_volume_from_area_l6_6646


namespace division_example_l6_6618

theorem division_example : 72 / (6 / 3) = 36 :=
by sorry

end division_example_l6_6618


namespace regular_polygon_sides_l6_6154

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6154


namespace street_tree_fourth_point_l6_6261

theorem street_tree_fourth_point (a b : ℝ) (h_a : a = 0.35) (h_b : b = 0.37) :
  (a + 4 * ((b - a) / 4)) = b :=
by 
  rw [h_a, h_b]
  sorry

end street_tree_fourth_point_l6_6261


namespace digit_distribution_l6_6287

theorem digit_distribution (n: ℕ) : 
(1 / 2) * n + (1 / 5) * n + (1 / 5) * n + (1 / 10) * n = n → 
n = 10 :=
by
  sorry

end digit_distribution_l6_6287


namespace simplified_expression_eq_l6_6998

noncomputable def simplify_expression : ℚ :=
  1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)

-- We need to prove that the simplifed expression is equal to 1 / 39
theorem simplified_expression_eq : simplify_expression = 1 / 39 :=
by sorry

end simplified_expression_eq_l6_6998


namespace regular_polygon_sides_l6_6197

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l6_6197


namespace option_A_option_B_option_C_option_D_l6_6000

theorem option_A : (-(-1) : ℤ) ≠ -|(-1 : ℤ)| := by
  sorry

theorem option_B : ((-3)^2 : ℤ) ≠ -(3^2 : ℤ) := by
  sorry

theorem option_C : ((-4)^3 : ℤ) = -(4^3 : ℤ) := by
  sorry

theorem option_D : ((2^2 : ℚ)/3) ≠ ((2/3)^2 : ℚ) := by
  sorry

end option_A_option_B_option_C_option_D_l6_6000


namespace find_h_in_standard_form_l6_6432

-- The expression to be converted
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x - 24

-- The standard form with given h value
def standard_form (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

-- The theorem statement
theorem find_h_in_standard_form :
  ∃ k : ℝ, ∀ x : ℝ, quadratic_expr x = standard_form 3 (-1.5) k x :=
by
  let a := 3
  let h := -1.5
  existsi (-30.75)
  intro x
  sorry

end find_h_in_standard_form_l6_6432


namespace anne_cleaning_time_l6_6779

-- Define the conditions in the problem
variable (B A : ℝ) -- B and A are Bruce's and Anne's cleaning rates respectively

-- Conditions based on the given problem
axiom cond1 : (B + A) * 4 = 1 -- Together they can clean the house in 4 hours
axiom cond2 : (B + 2 * A) * 3 = 1 -- With Anne's speed doubled, they clean in 3 hours

-- The theorem statement asserting Anne’s time to clean the house alone is 12 hours
theorem anne_cleaning_time : (1 / A) = 12 :=
by 
  -- start by analyzing the first condition
  have h1 : 4 * B + 4 * A = 1, from cond1,
  -- next, process the second condition
  have h2 : 3 * B + 6 * A = 1, from cond2,
  -- combine and solve these conditions
  sorry

end anne_cleaning_time_l6_6779


namespace factorize_polynomial_l6_6019

theorem factorize_polynomial :
  ∀ (x : ℝ), x^4 + 2021 * x^2 + 2020 * x + 2021 = (x^2 + x + 1) * (x^2 - x + 2021) :=
by
  intros x
  sorry

end factorize_polynomial_l6_6019


namespace binary_representation_of_fourteen_l6_6925

theorem binary_representation_of_fourteen :
  (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by
  sorry

end binary_representation_of_fourteen_l6_6925


namespace four_clique_exists_in_tournament_l6_6301

open Finset

/-- Given a graph G with 9 vertices and 28 edges, prove that G contains a 4-clique. -/
theorem four_clique_exists_in_tournament 
  (V : Finset ℕ) (E : Finset (ℕ × ℕ)) 
  (hV : V.card = 9) 
  (hE : E.card = 28) :
  ∃ (S : Finset ℕ), S.card = 4 ∧ ∀ (v₁ v₂ : ℕ), v₁ ∈ S → v₂ ∈ S → v₁ ≠ v₂ → (v₁, v₂) ∈ E ∨ (v₂, v₁) ∈ E :=
sorry

end four_clique_exists_in_tournament_l6_6301


namespace pairings_without_alice_and_bob_l6_6973

theorem pairings_without_alice_and_bob (n : ℕ) (h : n = 12) : 
    ∃ k : ℕ, k = ((n * (n - 1)) / 2) - 1 ∧ k = 65 :=
by
  sorry

end pairings_without_alice_and_bob_l6_6973


namespace increasing_on_interval_l6_6415

open Real

noncomputable def f (x a b : ℝ) := abs (x^2 - 2*a*x + b)

theorem increasing_on_interval {a b : ℝ} (h : a^2 - b ≤ 0) :
  ∀ ⦃x1 x2⦄, a ≤ x1 → x1 ≤ x2 → f x1 a b ≤ f x2 a b := sorry

end increasing_on_interval_l6_6415


namespace least_number_to_subtract_l6_6346

theorem least_number_to_subtract (n : ℕ) (h1 : n = 157632)
  (h2 : ∃ k : ℕ, k = 12 * 18 * 24 / (gcd 12 (gcd 18 24)) ∧ k ∣ n - 24) :
  n - 24 = 24 := 
sorry

end least_number_to_subtract_l6_6346


namespace find_D_l6_6208

-- This representation assumes 'ABCD' represents digits A, B, C, and D forming a four-digit number.
def four_digit_number (A B C D : ℕ) : ℕ :=
  1000 * A + 100 * B + 10 * C + D

theorem find_D (A B C D : ℕ) (h1 : 1000 * A + 100 * B + 10 * C + D 
                            = 2736) (h2: A ≠ B) (h3: A ≠ C) 
  (h4: A ≠ D) (h5: B ≠ C) (h6: B ≠ D) (h7: C ≠ D) : D = 6 := 
sorry

end find_D_l6_6208


namespace sum_of_abs_values_l6_6402

-- Define the problem conditions
variable (a b c d m : ℤ)
variable (h1 : a + b + c + d = 1)
variable (h2 : a * b + a * c + a * d + b * c + b * d + c * d = 0)
variable (h3 : a * b * c + a * b * d + a * c * d + b * c * d = -4023)
variable (h4 : a * b * c * d = m)

-- Prove the required sum of absolute values
theorem sum_of_abs_values : |a| + |b| + |c| + |d| = 621 :=
by
  sorry

end sum_of_abs_values_l6_6402


namespace min_dist_of_PQ_l6_6039

open Real

theorem min_dist_of_PQ :
  ∀ (P Q : ℝ × ℝ),
    (P.fst - 3)^2 + (P.snd + 1)^2 = 4 →
    Q.fst = -3 →
    ∃ (min_dist : ℝ), min_dist = 4 :=
by
  sorry

end min_dist_of_PQ_l6_6039


namespace not_divisible_by_n_l6_6992

theorem not_divisible_by_n (n : ℕ) (h : n > 1) : ¬n ∣ 2^n - 1 :=
by
  -- proof to be filled in
  sorry

end not_divisible_by_n_l6_6992


namespace max_initial_jars_l6_6244

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l6_6244


namespace consecutive_integers_sum_l6_6323

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 504) : n + n+1 + n+2 = 24 :=
sorry

end consecutive_integers_sum_l6_6323


namespace tyler_bird_pairs_l6_6741

theorem tyler_bird_pairs (n_species : ℕ) (pairs_per_species : ℕ) (total_pairs : ℕ)
  (h1 : n_species = 29)
  (h2 : pairs_per_species = 7)
  (h3 : total_pairs = n_species * pairs_per_species) : total_pairs = 203 :=
by
  sorry

end tyler_bird_pairs_l6_6741


namespace isosceles_triangle_min_perimeter_l6_6338

theorem isosceles_triangle_min_perimeter 
  (a b c : ℕ) 
  (h_perimeter : 2 * a + 12 * c = 2 * b + 15 * c) 
  (h_area : 16 * (a^2 - 36 * c^2) = 25 * (b^2 - 56.25 * c^2))
  (h_ratio : 4 * b = 5 * 12 * c) : 
  2 * a + 12 * c ≥ 840 :=
by
  -- proof here
  sorry

end isosceles_triangle_min_perimeter_l6_6338


namespace cos_2alpha_plus_pi_div_2_eq_neg_24_div_25_l6_6271

theorem cos_2alpha_plus_pi_div_2_eq_neg_24_div_25
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (h_tanα : Real.tan α = 4 / 3) :
  Real.cos (2 * α + π / 2) = - 24 / 25 :=
by sorry

end cos_2alpha_plus_pi_div_2_eq_neg_24_div_25_l6_6271


namespace natural_number_pairs_sum_to_three_l6_6566

theorem natural_number_pairs_sum_to_three :
  {p : ℕ × ℕ | p.1 + p.2 = 3} = {(1, 2), (2, 1)} :=
by
  sorry

end natural_number_pairs_sum_to_three_l6_6566


namespace bicycle_wheels_l6_6331

theorem bicycle_wheels :
  ∃ b : ℕ, 
  (∃ (num_bicycles : ℕ) (num_tricycles : ℕ) (wheels_per_tricycle : ℕ) (total_wheels : ℕ),
    num_bicycles = 16 ∧ 
    num_tricycles = 7 ∧ 
    wheels_per_tricycle = 3 ∧ 
    total_wheels = 53 ∧ 
    16 * b + num_tricycles * wheels_per_tricycle = total_wheels) ∧ 
  b = 2 :=
by
  sorry

end bicycle_wheels_l6_6331


namespace regular_polygon_sides_l6_6176

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l6_6176


namespace no_possible_arrangement_l6_6091

theorem no_possible_arrangement :
  ¬ ∃ (a : Fin 9 → ℕ),
    (∀ i, 1 ≤ a i ∧ a i ≤ 9) ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) % 3 = 0) ∧
    (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) > 12) :=
  sorry

end no_possible_arrangement_l6_6091


namespace line_through_point_equal_intercepts_l6_6028

-- Definitions based on conditions
def passes_through (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l p.1 p.2

def equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a, a ≠ 0 ∧ (∀ x y, l x y ↔ x + y = a) ∨ (∀ x y, l x y ↔ y = 2 * x)

-- Theorem statement based on the problem
theorem line_through_point_equal_intercepts :
  ∃ l, passes_through (1, 2) l ∧ equal_intercepts l ∧
  (∀ x y, l x y ↔ 2 * x - y = 0) ∨ (∀ x y, l x y ↔ x + y - 3 = 0) :=
sorry

end line_through_point_equal_intercepts_l6_6028


namespace eval_expression_l6_6269

theorem eval_expression {p q r s : ℝ} 
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) + s / (40 - s) = 9) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) + 8 / (40 - s) = 7.6 := 
by 
  sorry

end eval_expression_l6_6269


namespace average_marks_increase_ratio_l6_6904

theorem average_marks_increase_ratio
  (T : ℕ)  -- The correct total marks of the class
  (n : ℕ)  -- The number of pupils in the class
  (h_n : n = 16) (wrong_mark : ℕ) (correct_mark : ℕ)  -- The wrong and correct marks
  (h_wrong : wrong_mark = 73) (h_correct : correct_mark = 65) :
  (8 : ℚ) / T = (wrong_mark - correct_mark : ℚ) / n * (n / T) :=
by
  sorry

end average_marks_increase_ratio_l6_6904


namespace find_constant_b_l6_6394

variable (x : ℝ)
variable (b d e : ℝ)

theorem find_constant_b   
  (h1 : (7 * x ^ 2 - 2 * x + 4 / 3) * (d * x ^ 2 + b * x + e) = 28 * x ^ 4 - 10 * x ^ 3 + 18 * x ^ 2 - 8 * x + 5 / 3)
  (h2 : d = 4) : 
  b = -2 / 7 := 
sorry

end find_constant_b_l6_6394


namespace employee_earnings_l6_6773

theorem employee_earnings (regular_rate overtime_rate first3_days_h second2_days_h total_hours overtime_hours : ℕ)
  (h1 : regular_rate = 30)
  (h2 : overtime_rate = 45)
  (h3 : first3_days_h = 6)
  (h4 : second2_days_h = 12)
  (h5 : total_hours = first3_days_h * 3 + second2_days_h * 2)
  (h6 : total_hours = 42)
  (h7 : overtime_hours = total_hours - 40)
  (h8 : overtime_hours = 2) :
  (40 * regular_rate + overtime_hours * overtime_rate) = 1290 := 
sorry

end employee_earnings_l6_6773


namespace inequality_solution_l6_6022

theorem inequality_solution :
  ∀ x : ℝ, (5 / 24 + |x - 11 / 48| < 5 / 16 ↔ (1 / 8 < x ∧ x < 1 / 3)) :=
by
  intro x
  sorry

end inequality_solution_l6_6022


namespace regular_polygon_sides_l6_6156

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6156


namespace max_initial_jars_l6_6243

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l6_6243


namespace tangent_neg_five_pi_six_eq_one_over_sqrt_three_l6_6628

noncomputable def tangent_neg_five_pi_six : Real :=
  Real.tan (-5 * Real.pi / 6)

theorem tangent_neg_five_pi_six_eq_one_over_sqrt_three :
  tangent_neg_five_pi_six = 1 / Real.sqrt 3 := by
  sorry

end tangent_neg_five_pi_six_eq_one_over_sqrt_three_l6_6628


namespace sequence_is_decreasing_l6_6278

-- Define the sequence {a_n} using a recursive function
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ (∀ n, a (n + 1) = a n / (3 * a n + 1))

-- Define a condition ensuring the sequence a_n is decreasing
theorem sequence_is_decreasing (a : ℕ → ℝ) (h : seq a) : ∀ n, a (n + 1) < a n :=
by
  intro n
  sorry

end sequence_is_decreasing_l6_6278


namespace at_least_one_not_land_designated_area_l6_6824

variable (p q : Prop)

theorem at_least_one_not_land_designated_area : ¬p ∨ ¬q ↔ ¬ (p ∧ q) :=
by sorry

end at_least_one_not_land_designated_area_l6_6824


namespace factor_3a3_minus_6a2_plus_3a_factor_a2_minus_b2_x_minus_y_factor_16a_plus_b_sq_minus_9a_minus_b_sq_l6_6928

-- First factorization problem
theorem factor_3a3_minus_6a2_plus_3a (a : ℝ) : 
  3 * a ^ 3 - 6 * a ^ 2 + 3 * a = 3 * a * (a - 1) ^ 2 :=
by sorry

-- Second factorization problem
theorem factor_a2_minus_b2_x_minus_y (a b x y : ℝ) : 
  a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a - b) * (a + b) :=
by sorry

-- Third factorization problem
theorem factor_16a_plus_b_sq_minus_9a_minus_b_sq (a b : ℝ) : 
  16 * (a + b) ^ 2 - 9 * (a - b) ^ 2 = (a + 7 * b) * (7 * a + b) :=
by sorry

end factor_3a3_minus_6a2_plus_3a_factor_a2_minus_b2_x_minus_y_factor_16a_plus_b_sq_minus_9a_minus_b_sq_l6_6928


namespace integer_solutions_l6_6020

theorem integer_solutions (x y z : ℤ) : 
  x + y + z = 3 ∧ x^3 + y^3 + z^3 = 3 ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 4 ∧ y = 4 ∧ z = -5) ∨
  (x = 4 ∧ y = -5 ∧ z = 4) ∨
  (x = -5 ∧ y = 4 ∧ z = 4) := 
sorry

end integer_solutions_l6_6020


namespace point_coordinates_l6_6040

theorem point_coordinates (M : ℝ × ℝ) 
  (hx : abs M.2 = 3) 
  (hy : abs M.1 = 2) 
  (h_first_quadrant : 0 < M.1 ∧ 0 < M.2) : 
  M = (2, 3) := 
sorry

end point_coordinates_l6_6040


namespace T7_value_l6_6643

-- Define the geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

-- Define the even function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (2 * a + 1) * x + 2 * a

-- The main theorem statement
theorem T7_value (a : ℕ → ℝ) (a2 a6 : ℝ) (a_val : ℝ) (q : ℝ) (T7 : ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : a 2 = a2)
  (h3 : a 6 = a6)
  (h4 : a2 - 2 = f a_val 0)
  (h5 : a6 - 3 = f a_val 0)
  (h6 : q > 1)
  (h7 : T7 = a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7) : 
  T7 = 128 :=
sorry

end T7_value_l6_6643


namespace directrix_parabola_y_eq_2x2_l6_6497

theorem directrix_parabola_y_eq_2x2 : (∃ y : ℝ, y = 2 * x^2) → (∃ y : ℝ, y = -1/8) :=
by
  sorry

end directrix_parabola_y_eq_2x2_l6_6497


namespace cos_double_angle_l6_6955

theorem cos_double_angle (θ : ℝ) (h : Real.tan θ = -1/3) : Real.cos (2 * θ) = 4/5 :=
sorry

end cos_double_angle_l6_6955


namespace ball_bounces_before_vertex_l6_6426

def bounces_to_vertex (v h : ℕ) (units_per_bounce_vert units_per_bounce_hor : ℕ) : ℕ :=
units_per_bounce_vert * v / units_per_bounce_hor * h

theorem ball_bounces_before_vertex (verts : ℕ) (h : ℕ) (units_per_bounce_vert units_per_bounce_hor : ℕ)
    (H_vert : verts = 10)
    (H_units_vert : units_per_bounce_vert = 2)
    (H_units_hor : units_per_bounce_hor = 7) :
    bounces_to_vertex verts h units_per_bounce_vert units_per_bounce_hor = 5 := 
by
  sorry

end ball_bounces_before_vertex_l6_6426


namespace quadratic_has_two_distinct_real_roots_l6_6730

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ a b : ℝ, a ≠ b ∧ (a ^ 2 + m * a - 8 = 0) ∧ (b ^ 2 + m * b - 8 = 0) :=
sorry

end quadratic_has_two_distinct_real_roots_l6_6730


namespace solve_for_a_l6_6810

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ∈ Ioc 0 2 then log x - a * x + 1 else 
if x ∈ Ioo (-2) 0 then -(log (-x) - a * (-x) + 1) else 0

theorem solve_for_a :
  (∃ a : ℝ, 
  (∀ x ∈ Icc 0 2, f a (-x) = -f a x) ∧ 
  ∀ x ∈ Ioo (-2) 0, (∃ y ∈ Icc 0 2, f a x = 1) ∧ 
    (∀ y ∈ Icc 0 2, f a y = -1) 
  → a = 2) :=
begin
  sorry
end

end solve_for_a_l6_6810


namespace compare_logs_l6_6838

theorem compare_logs (a b c : ℝ) (h1 : a = Real.log 6 / Real.log 3)
                              (h2 : b = Real.log 8 / Real.log 4)
                              (h3 : c = Real.log 10 / Real.log 5) : 
                              a > b ∧ b > c :=
by
  sorry

end compare_logs_l6_6838


namespace carlson_max_jars_l6_6240

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l6_6240


namespace original_number_is_17_l6_6333

-- Function to reverse the digits of a two-digit number
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  (ones * 10) + tens

-- Problem statement
theorem original_number_is_17 (x : ℕ) (h1 : reverse_digits (2 * x) + 2 = 45) : x = 17 :=
by
  sorry

end original_number_is_17_l6_6333


namespace total_students_in_school_l6_6485

theorem total_students_in_school : 
  ∀ (number_of_deaf_students number_of_blind_students : ℕ), 
  (number_of_deaf_students = 180) → 
  (number_of_deaf_students = 3 * number_of_blind_students) → 
  (number_of_deaf_students + number_of_blind_students = 240) :=
by 
  sorry

end total_students_in_school_l6_6485


namespace paula_paint_coverage_l6_6561

-- Define the initial conditions
def initial_capacity : ℕ := 36
def lost_cans : ℕ := 4
def reduced_capacity : ℕ := 28

-- Define the proof problem
theorem paula_paint_coverage :
  (initial_capacity - reduced_capacity = lost_cans * (initial_capacity / reduced_capacity)) →
  (reduced_capacity / (initial_capacity / reduced_capacity) = 14) :=
by
  sorry

end paula_paint_coverage_l6_6561


namespace smallest_integer_y_l6_6345

theorem smallest_integer_y (y : ℤ) :
  (∃ y : ℤ, ((y / 4 : ℚ) + (3 / 7 : ℚ) > 2 / 3) ∧ (∀ z : ℤ, (z > 20 / 21) → y ≤ z)) :=
sorry

end smallest_integer_y_l6_6345


namespace find_integer_x_l6_6989

theorem find_integer_x (x : ℤ) :
  (2 * x > 70 → x > 35 ∧ 4 * x > 25 → x > 6) ∧
  (x < 100 → x = 6) ∧ -- Given valid inequality relations and restrictions
  ((2 * x <= 70 ∨ x >= 100) ∨ (4 * x <= 25 ∨ x <= 5)) := -- Certain contradictions inherent

  by sorry

end find_integer_x_l6_6989


namespace Carlson_initial_jars_max_count_l6_6230

def initial_jar_weight_ratio (c_initial_weight b_initial_weight: ℕ) : Prop := 
  c_initial_weight = 13 * b_initial_weight

def new_jar_weight_ratio (c_new_weight b_new_weight: ℕ) : Prop := 
  c_new_weight = 8 * b_new_weight

theorem Carlson_initial_jars_max_count (c_initial_weight b_initial_weight c_new_weight b_new_weight: ℕ) 
  (h1 : initial_jar_weight_ratio c_initial_weight b_initial_weight) 
  (h2 : new_jar_weight_ratio c_new_weight b_new_weight)
  (h3 : ∀ a: ℕ, c_new_weight = c_initial_weight - a ∧ b_new_weight = b_initial_weight + a) :
  ∃ n: ℕ, n ≤ 23 :=
begin
  sorry,
end

end Carlson_initial_jars_max_count_l6_6230


namespace perfect_squares_difference_l6_6046

theorem perfect_squares_difference : 
  let N : ℕ := 20000;
  let diff_squared (b : ℤ) : ℤ := (b+2)^2 - b^2;
  ∃ k : ℕ, (1 ≤ k ∧ k ≤ 70) ∧ (∀ m : ℕ, (m < N) → (∃ b : ℤ, m = diff_squared b) → m = (2 * k)^2)
:= sorry

end perfect_squares_difference_l6_6046


namespace find_abscissa_of_P_l6_6413

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem find_abscissa_of_P (x_P : ℝ) :
  (x + 2*y - 1 = 0 -> 
  (f' x_P = 2 -> 
  (f x_P - 2) * (x_P^2 - 1) = 0)) := by
  sorry

end find_abscissa_of_P_l6_6413


namespace find_the_number_l6_6819

noncomputable def special_expression (x : ℝ) : ℝ :=
  9 - 8 / x * 5 + 10

theorem find_the_number (x : ℝ) (h : special_expression x = 13.285714285714286) : x = 7 := by
  sorry

end find_the_number_l6_6819


namespace first_number_percentage_of_second_l6_6057

theorem first_number_percentage_of_second (X : ℝ) (h1 : First = 0.06 * X) (h2 : Second = 0.18 * X) : 
  (First / Second) * 100 = 33.33 := 
by 
  sorry

end first_number_percentage_of_second_l6_6057


namespace correct_propositions_l6_6518

-- Define propositions
def proposition1 : Prop :=
  ∀ x, 2 * (Real.cos (1/3 * x + Real.pi / 4))^2 - 1 = -Real.sin (2 * x / 3)

def proposition2 : Prop :=
  ∃ α : ℝ, Real.sin α + Real.cos α = 3 / 2

def proposition3 : Prop :=
  ∀ α β : ℝ, (0 < α ∧ α < Real.pi / 2) → (0 < β ∧ β < Real.pi / 2) → α < β → Real.tan α < Real.tan β

def proposition4 : Prop :=
  ∀ x, x = Real.pi / 8 → Real.sin (2 * x + 5 * Real.pi / 4) = -1

def proposition5 : Prop :=
  Real.sin ( 2 * (Real.pi / 12) + Real.pi / 3 ) = 0

-- Define the main theorem combining correct propositions
theorem correct_propositions : 
  proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ proposition4 ∧ ¬proposition5 :=
  by
  -- Since we only need to state the theorem, we use sorry.
  sorry

end correct_propositions_l6_6518


namespace range_of_3a_minus_2b_l6_6509

theorem range_of_3a_minus_2b (a b : ℝ) (h1 : 1 ≤ a - b ∧ a - b ≤ 2) (h2 : 2 ≤ a + b ∧ a + b ≤ 4) :
  7 / 2 ≤ 3 * a - 2 * b ∧ 3 * a - 2 * b ≤ 7 :=
sorry

end range_of_3a_minus_2b_l6_6509


namespace number_of_chickens_l6_6472

-- Definitions based on conditions
def totalAnimals := 100
def legDifference := 26

-- The problem statement to be proved
theorem number_of_chickens (x : Nat) (r : Nat) (legs_chickens : Nat) (legs_rabbits : Nat) (total : Nat := totalAnimals) (diff : Nat := legDifference) :
  x + r = total ∧ 2 * x + 4 * r - 4 * r = 2 * x + diff → x = 71 :=
by
  intro h
  sorry

end number_of_chickens_l6_6472


namespace find_starting_number_l6_6708

theorem find_starting_number (x : ℕ) (h1 : (50 + 250) / 2 = 150)
  (h2 : (x + 400) / 2 = 150 + 100) : x = 100 := by
  sorry

end find_starting_number_l6_6708


namespace arithmetic_sqrt_of_4_eq_2_l6_6312

theorem arithmetic_sqrt_of_4_eq_2 (x : ℕ) (h : x^2 = 4) : x = 2 :=
sorry

end arithmetic_sqrt_of_4_eq_2_l6_6312


namespace height_of_parallelogram_l6_6484

def area_of_parallelogram (base height : ℝ) : ℝ := base * height

theorem height_of_parallelogram (A B H : ℝ) (hA : A = 33.3) (hB : B = 9) (hAparallelogram : A = area_of_parallelogram B H) :
  H = 3.7 :=
by 
  -- Proof would go here
  sorry

end height_of_parallelogram_l6_6484


namespace police_emergency_number_prime_factor_l6_6480

theorem police_emergency_number_prime_factor (N : ℕ) (h1 : N % 1000 = 133) : 
  ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ N :=
sorry

end police_emergency_number_prime_factor_l6_6480


namespace number_of_3digit_even_numbers_divisible_by_9_l6_6523

theorem number_of_3digit_even_numbers_divisible_by_9 : 
    ∃ n : ℕ, (n = 50) ∧
    (∀ k, (108 + (k - 1) * 18 = 990) ↔ (108 ≤ 108 + (k - 1) * 18 ∧ 108 + (k - 1) * 18 ≤ 999)) :=
by {
  sorry
}

end number_of_3digit_even_numbers_divisible_by_9_l6_6523


namespace cricket_team_players_l6_6734

-- Define conditions 
def non_throwers (T P : ℕ) : ℕ := P - T
def left_handers (N : ℕ) : ℕ := N / 3
def right_handers_non_thrower (N : ℕ) : ℕ := 2 * N / 3
def total_right_handers (T R : ℕ) : Prop := R = T + right_handers_non_thrower (non_throwers T R)

-- Assume conditions are given
variables (P N R T : ℕ)
axiom hT : T = 37
axiom hR : R = 49
axiom hNonThrower : N = non_throwers T P
axiom hRightHanders : right_handers_non_thrower N = R - T

-- Prove the total number of players is 55
theorem cricket_team_players : P = 55 :=
by
  sorry

end cricket_team_players_l6_6734


namespace regular_polygon_sides_l6_6185

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6185


namespace speed_of_current_l6_6758

-- Definitions of the given conditions
def downstream_time := 6 / 60 -- time in hours to travel 1 km downstream
def upstream_time := 10 / 60 -- time in hours to travel 1 km upstream

-- Definition of speeds
def downstream_speed := 1 / downstream_time -- speed in km/h downstream
def upstream_speed := 1 / upstream_time -- speed in km/h upstream

-- Theorem statement
theorem speed_of_current : 
  (downstream_speed - upstream_speed) / 2 = 2 := 
by 
  -- We skip the proof for now
  sorry

end speed_of_current_l6_6758


namespace simplify_polynomial_expression_l6_6888

noncomputable def polynomial_expression (x : ℝ) := 
  (3 * x^3 + x^2 - 5 * x + 9) * (x + 2) - (x + 2) * (2 * x^3 - 4 * x + 8) + (x^2 - 6 * x + 13) * (x + 2) * (x - 3)

theorem simplify_polynomial_expression (x : ℝ) :
  polynomial_expression x = 2 * x^4 + x^3 + 9 * x^2 + 23 * x + 2 :=
sorry

end simplify_polynomial_expression_l6_6888


namespace locus_of_points_l6_6383

theorem locus_of_points 
  (x y : ℝ) 
  (A : (ℝ × ℝ)) 
  (B : (ℝ × ℝ)) 
  (C : (ℝ × ℝ))
  (PA PB PC : ℝ)
  (PA := (x - A.1)^2 + (y - A.2)^2)
  (PB := (x - B.1)^2 + (y - B.2)^2)
  (PC := (x - C.1)^2 + (y - C.2)^2) 
  (area : ℝ)
  (area := 6) -- since the area of the triangle is given as 6
  (h : PA + PB + PC - 2 * (area)^2 = 50) :
  (x - 1)^2 + (y - 4 / 3)^2 = 116 / 3 :=
by
  sorry

end locus_of_points_l6_6383


namespace parts_processed_per_day_l6_6844

-- Given conditions
variable (a : ℕ)

-- Goal: Prove the daily productivity of Master Wang given the conditions
theorem parts_processed_per_day (h1 : ∀ n, n = 8) (h2 : ∃ m, m = a + 3):
  (a + 3) / 8 = (a + 3) / 8 :=
by
  sorry

end parts_processed_per_day_l6_6844


namespace maximize_S_l6_6653

noncomputable def a (n: ℕ) : ℝ := 24 - 2 * n

noncomputable def S (n: ℕ) : ℝ := -n^2 + 23 * n

theorem maximize_S (n : ℕ) : 
  (n = 11 ∨ n = 12) → ∀ m : ℕ, m ≠ 11 ∧ m ≠ 12 → S m ≤ S n :=
sorry

end maximize_S_l6_6653


namespace total_milk_bottles_l6_6557

theorem total_milk_bottles (marcus_bottles : ℕ) (john_bottles : ℕ) (h1 : marcus_bottles = 25) (h2 : john_bottles = 20) : marcus_bottles + john_bottles = 45 := by
  sorry

end total_milk_bottles_l6_6557


namespace inverse_proportion_quadrants_l6_6651

theorem inverse_proportion_quadrants (a k : ℝ) (ha : a ≠ 0) (h : (3 * a, a) ∈ {p : ℝ × ℝ | p.snd = k / p.fst}) :
  k = 3 * a^2 ∧ k > 0 ∧
  (
    (∀ x y : ℝ, x > 0 → y = k / x → y > 0) ∨
    (∀ x y : ℝ, x < 0 → y = k / x → y < 0)
  ) :=
by
  sorry

end inverse_proportion_quadrants_l6_6651


namespace arithmetic_square_root_of_4_l6_6315

theorem arithmetic_square_root_of_4 : ∃ x : ℕ, x * x = 4 ∧ x = 2 := 
sorry

end arithmetic_square_root_of_4_l6_6315


namespace reduced_price_is_16_l6_6103

noncomputable def reduced_price_per_kg (P : ℝ) (r : ℝ) : ℝ :=
  0.9 * (P * (1 + r))

theorem reduced_price_is_16 (P r : ℝ) (h₀ : (0.9 : ℝ) * (P * (1 + r)) = 16) : 
  reduced_price_per_kg P r = 16 :=
by
  -- We have the hypothesis and we need to prove the result
  exact h₀

end reduced_price_is_16_l6_6103


namespace determine_a_minus_b_l6_6946

theorem determine_a_minus_b (a b : ℤ) 
  (h1 : 2009 * a + 2013 * b = 2021) 
  (h2 : 2011 * a + 2015 * b = 2023) : 
  a - b = -5 :=
sorry

end determine_a_minus_b_l6_6946


namespace compute_five_fold_application_l6_6682

def f (x : ℤ) : ℤ :=
  if x >= 0 then -(x^3) else x + 10

theorem compute_five_fold_application : f (f (f (f (f 2)))) = -8 := by
  sorry

end compute_five_fold_application_l6_6682


namespace exists_triangle_with_side_lengths_l6_6630

theorem exists_triangle_with_side_lengths (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

end exists_triangle_with_side_lengths_l6_6630


namespace prob_black_yellow_l6_6282

theorem prob_black_yellow:
  ∃ (x y : ℚ), 12 > 0 ∧
  (∃ (r b y' : ℚ), r = 1/3 ∧ b - y' = 1/6 ∧ b + y' = 2/3 ∧ r + b + y' = 1) ∧
  x = 5/12 ∧ y = 1/4 :=
by
  sorry

end prob_black_yellow_l6_6282


namespace find_angle_EFC_l6_6808

-- Define the properties of the problem.
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist A C

def angle (A B C : ℝ × ℝ) : ℝ :=
  -- Compute the angle using the law of cosines or any other method
  sorry

def perpendicular_foot (P A B : ℝ × ℝ) : ℝ × ℝ :=
  -- Compute the foot of the perpendicular from point P to the line AB
  sorry

noncomputable def main_problem : Prop :=
  ∀ (A B C D E F : ℝ × ℝ),
    is_isosceles A B C →
    angle A B C = 22 →  -- Given angle BAC
    ∃ x : ℝ, dist B D = 2 * dist D C →  -- Point D such that BD = 2 * CD
    E = perpendicular_foot B A D →
    F = perpendicular_foot B A C →
    angle E F C = 33  -- required to prove

-- Statement of the main problem.
theorem find_angle_EFC : main_problem := sorry

end find_angle_EFC_l6_6808


namespace minimum_point_translation_l6_6571

theorem minimum_point_translation (x y : ℝ) : 
  (∀ (x : ℝ), y = 2 * |x| - 4) →
  x = 0 →
  y = -4 →
  (∀ (x y : ℝ), x_new = x + 3 ∧ y_new = y + 4) →
  (x_new, y_new) = (3, 0) :=
sorry

end minimum_point_translation_l6_6571


namespace wholesale_cost_calc_l6_6892

theorem wholesale_cost_calc (wholesale_cost : ℝ) 
  (h_profit : 0.15 * wholesale_cost = 28 - wholesale_cost) : 
  wholesale_cost = 28 / 1.15 :=
by
  sorry

end wholesale_cost_calc_l6_6892


namespace mia_min_stamps_l6_6846

theorem mia_min_stamps (x y : ℕ) (hx : 5 * x + 7 * y = 37) : x + y = 7 :=
sorry

end mia_min_stamps_l6_6846


namespace isosceles_triangle_l6_6036

theorem isosceles_triangle
  (α β γ x y z w : ℝ)
  (h_triangle : α + β + γ = 180)
  (h_quad : x + y + z + w = 360)
  (h_conditions : (x = α + β) ∧ (y = β + γ) ∧ (z = γ + α) ∨ (w = α + β) ∧ (x = β + γ) ∧ (y = γ + α) ∨ (z = α + β) ∧ (w = β + γ) ∧ (x = γ + α) ∨ (y = α + β) ∧ (z = β + γ) ∧ (w = γ + α))
  : α = β ∨ β = γ ∨ γ = α := 
sorry

end isosceles_triangle_l6_6036


namespace intersection_points_l6_6932

theorem intersection_points :
  {p : ℝ × ℝ |
    (∃ x : ℝ, p = (x, 3*x^2 - 4*x + 2) ∧ p = (x, x^3 - 2*x^2 + 5*x - 1))} =
  {(1, 1), (3, 17)} :=
  sorry

end intersection_points_l6_6932


namespace john_cuts_his_grass_to_l6_6977

theorem john_cuts_his_grass_to (growth_rate monthly_cost annual_cost cut_height : ℝ)
  (h : ℝ) : 
  growth_rate = 0.5 ∧ monthly_cost = 100 ∧ annual_cost = 300 ∧ cut_height = 4 →
  h = 2 := by
  intros conditions
  sorry

end john_cuts_his_grass_to_l6_6977


namespace fg_of_3_eq_29_l6_6663

def g (x : ℕ) : ℕ := x * x
def f (x : ℕ) : ℕ := 3 * x + 2

theorem fg_of_3_eq_29 : f (g 3) = 29 :=
by
  sorry

end fg_of_3_eq_29_l6_6663


namespace regular_polygon_sides_l6_6182

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → is_regular_polygon n) (h2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → angle.polygon_interior_angle n i = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6182


namespace depletion_rate_l6_6900

theorem depletion_rate (initial_value final_value : ℝ) (years: ℕ) (r : ℝ) 
  (h1 : initial_value = 2500)
  (h2 : final_value = 2256.25)
  (h3 : years = 2)
  (h4 : final_value = initial_value * (1 - r) ^ years) :
  r = 0.05 :=
by
  sorry

end depletion_rate_l6_6900


namespace families_with_neither_l6_6284

theorem families_with_neither (total_families : ℕ) (families_with_cats : ℕ) (families_with_dogs : ℕ) (families_with_both : ℕ) :
  total_families = 40 → families_with_cats = 18 → families_with_dogs = 24 → families_with_both = 10 → 
  total_families - (families_with_cats + families_with_dogs - families_with_both) = 8 :=
by
  intros h1 h2 h3 h4
  sorry

end families_with_neither_l6_6284


namespace joe_bought_books_l6_6438

theorem joe_bought_books (money_given : ℕ) (notebook_cost : ℕ) (num_notebooks : ℕ) (book_cost : ℕ) (leftover_money : ℕ) (total_spent := money_given - leftover_money) (spent_on_notebooks := num_notebooks * notebook_cost) (spent_on_books := total_spent - spent_on_notebooks) (num_books := spent_on_books / book_cost) : money_given = 56 → notebook_cost = 4 → num_notebooks = 7 → book_cost = 7 → leftover_money = 14 → num_books = 2 := by
  intros
  sorry

end joe_bought_books_l6_6438


namespace regular_polygon_num_sides_l6_6161

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l6_6161


namespace oliver_workout_hours_l6_6987

variable (x : ℕ)

theorem oliver_workout_hours :
  (x + (x - 2) + 2 * x + 2 * (x - 2) = 18) → x = 4 :=
by
  sorry

end oliver_workout_hours_l6_6987


namespace negative_10m_means_westward_l6_6659

-- Definitions to specify conditions
def is_eastward (m: Int) : Prop :=
  m > 0

def is_westward (m: Int) : Prop :=
  m < 0

-- Theorem to state the proof problem
theorem negative_10m_means_westward (m : Int) (h : m = -10) : 
  is_westward m :=
begin
  rw h,
  exact dec_trivial,
end

end negative_10m_means_westward_l6_6659


namespace sidney_cats_l6_6854

theorem sidney_cats (A : ℕ) :
  (4 * 7 * (3 / 4) + A * 7 = 42) →
  A = 3 :=
by
  intro h
  sorry

end sidney_cats_l6_6854


namespace integers_solution_l6_6720

theorem integers_solution (a b : ℤ) (S D : ℤ) 
  (h1 : S = a + b) (h2 : D = a - b) (h3 : S / D = 3) (h4 : S * D = 300) : 
  ((a = 20 ∧ b = 10) ∨ (a = -20 ∧ b = -10)) :=
by
  sorry

end integers_solution_l6_6720


namespace greatest_possible_mean_BC_l6_6605

theorem greatest_possible_mean_BC :
  ∀ (A_n B_n C_weight C_n : ℕ),
    (A_n > 0) ∧ (B_n > 0) ∧ (C_n > 0) ∧
    (40 * A_n + 50 * B_n) / (A_n + B_n) = 43 ∧
    (40 * A_n + C_weight) / (A_n + C_n) = 44 →
    ∃ k : ℕ, ∃ n : ℕ, 
      A_n = 7 * k ∧ B_n = 3 * k ∧ 
      C_weight = 28 * k + 44 * n ∧ 
      44 + 46 * k / (3 * k + n) ≤ 59 :=
sorry

end greatest_possible_mean_BC_l6_6605


namespace regular_polygon_sides_l6_6201

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6201


namespace sum_of_angles_is_correct_l6_6677

noncomputable def hexagon_interior_angle : ℝ := 180 * (6 - 2) / 6
noncomputable def pentagon_interior_angle : ℝ := 180 * (5 - 2) / 5
noncomputable def sum_of_hexagon_and_pentagon_angles (A B C D : Type) 
  (hexagon_interior_angle : ℝ) 
  (pentagon_interior_angle : ℝ) : ℝ := 
  hexagon_interior_angle + pentagon_interior_angle

theorem sum_of_angles_is_correct (A B C D : Type) : 
  sum_of_hexagon_and_pentagon_angles A B C D hexagon_interior_angle pentagon_interior_angle = 228 := 
by
  simp [hexagon_interior_angle, pentagon_interior_angle]
  sorry

end sum_of_angles_is_correct_l6_6677


namespace subtract_value_l6_6058

theorem subtract_value (N x : ℤ) (h1 : (N - x) / 7 = 7) (h2 : (N - 6) / 8 = 6) : x = 5 := 
by 
  sorry

end subtract_value_l6_6058


namespace arithmetic_sequence_30th_term_value_l6_6576

def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) : ℤ :=
  a_1 + (n - 1) * d

-- Given conditions
def a1 : ℤ := 3
def a2 : ℤ := 15
def a3 : ℤ := 27

-- Calculate the common difference d
def d : ℤ := a2 - a1

-- Define the 30th term
def a30 := arithmetic_sequence a1 d 30

theorem arithmetic_sequence_30th_term_value :
  a30 = 351 := by
  sorry

end arithmetic_sequence_30th_term_value_l6_6576


namespace regular_polygon_sides_l6_6143

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6143


namespace maximum_initial_jars_l6_6234

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l6_6234


namespace even_function_periodic_odd_function_period_generalized_period_l6_6078

-- Problem 1
theorem even_function_periodic (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (-x) = f x) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 2 * a) = f x :=
by sorry

-- Problem 2
theorem odd_function_period (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 4 * a) = f x :=
by sorry

-- Problem 3
theorem generalized_period (f : ℝ → ℝ) (a m n : ℝ) (h₁ : ∀ x : ℝ, 2 * n - f x = f (2 * m - x)) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 4 * (m - a)) = f x :=
by sorry

end even_function_periodic_odd_function_period_generalized_period_l6_6078


namespace regular_polygon_num_sides_l6_6162

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l6_6162


namespace perfect_square_A_plus_B_plus1_l6_6353

-- Definitions based on conditions
def A (m : ℕ) : ℕ := (10^2*m - 1) / 9
def B (m : ℕ) : ℕ := 4 * (10^m - 1) / 9

-- Proof statement
theorem perfect_square_A_plus_B_plus1 (m : ℕ) : A m + B m + 1 = ((10^m + 2) / 3)^2 :=
by
  sorry

end perfect_square_A_plus_B_plus1_l6_6353


namespace area_of_triangle_l6_6537

open Real

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : sin A = sqrt 3 * sin C)
                        (h2 : B = π / 6) (h3 : b = 2) :
    1 / 2 * a * c * sin B = sqrt 3 :=
by
  sorry

end area_of_triangle_l6_6537


namespace jack_runs_faster_than_paul_l6_6071

noncomputable def convert_km_hr_to_m_s (v : ℝ) : ℝ :=
  v * (1000 / 3600)

noncomputable def speed_difference : ℝ :=
  let v_J_km_hr := 20.62665  -- Jack's speed in km/hr
  let v_J_m_s := convert_km_hr_to_m_s v_J_km_hr  -- Jack's speed in m/s
  let distance := 1000  -- distance in meters
  let time_J := distance / v_J_m_s  -- Jack's time in seconds
  let time_P := time_J + 1.5  -- Paul's time in seconds
  let v_P_m_s := distance / time_P  -- Paul's speed in m/s
  let speed_diff_m_s := v_J_m_s - v_P_m_s  -- speed difference in m/s
  let speed_diff_km_hr := speed_diff_m_s * (3600 / 1000)  -- convert to km/hr
  speed_diff_km_hr

theorem jack_runs_faster_than_paul : speed_difference = 0.18225 :=
by
  -- Proof is omitted
  sorry

end jack_runs_faster_than_paul_l6_6071


namespace totalMarbles_l6_6680

def originalMarbles : ℕ := 22
def marblesGiven : ℕ := 20

theorem totalMarbles : originalMarbles + marblesGiven = 42 := by
  sorry

end totalMarbles_l6_6680


namespace equation_solution_l6_6820

theorem equation_solution (x : ℤ) (h : 3 * x - 2 * x + x = 3 - 2 + 1) : x = 2 :=
by
  sorry

end equation_solution_l6_6820


namespace smallest_positive_period_interval_monotonic_increase_range_of_f_l6_6520

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 12)

theorem smallest_positive_period :
  ∀ x, f (x + Real.pi) = f x := sorry

theorem interval_monotonic_increase (k : ℤ) :
  ∀ x, x ∈ Set.Ico (-7 * Real.pi / 24 + k * Real.pi) (5 * Real.pi / 24 + k * Real.pi) →
        f.deriv x > 0 := sorry

theorem range_of_f (x : ℝ) :
  x ∈ Set.Icc (Real.pi / 8) (7 * Real.pi / 12) →
  f x ∈ Set.Icc (-Real.sqrt 2 / 2) 1 := sorry

end smallest_positive_period_interval_monotonic_increase_range_of_f_l6_6520


namespace max_subjects_per_teacher_l6_6766

theorem max_subjects_per_teacher
  (math_teachers : ℕ := 7)
  (physics_teachers : ℕ := 6)
  (chemistry_teachers : ℕ := 5)
  (min_teachers_required : ℕ := 6)
  (total_subjects : ℕ := 18) :
  ∀ (x : ℕ), x ≥ 3 ↔ 6 * x ≥ total_subjects := by
  sorry

end max_subjects_per_teacher_l6_6766


namespace regular_polygon_sides_l6_6127

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l6_6127


namespace thomas_monthly_earnings_l6_6604

def weekly_earnings : ℕ := 4550
def weeks_in_month : ℕ := 4
def monthly_earnings : ℕ := weekly_earnings * weeks_in_month

theorem thomas_monthly_earnings : monthly_earnings = 18200 := by
  sorry

end thomas_monthly_earnings_l6_6604


namespace power_inequality_l6_6688

variable (a b c : ℝ)
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_ineq : a^19 / b^19 + b^19 / c^19 + c^19 / a^19 ≤ a^19 / c^19 + b^19 / a^19 + c^19 / b^19)

theorem power_inequality :
  a^20 / b^20 + b^20 / c^20 + c^20 / a^20 ≤ a^20 / c^20 + b^20 / a^20 + c^20 / b^20 :=
by
  sorry

end power_inequality_l6_6688


namespace quadratic_has_two_distinct_real_roots_l6_6723

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  let a := 1
      c := -8
      b := m
      Δ := b^2 - 4 * a * c 
  in (Δ > 0) :=
by
  let a := 1
  let c := -8
  let b := m
  let Δ := b^2 - 4 * a * c
  show Δ > 0
  sorry

end quadratic_has_two_distinct_real_roots_l6_6723


namespace find_integer_x_l6_6990

theorem find_integer_x (x : ℤ) :
  (2 * x > 70 → x > 35 ∧ 4 * x > 25 → x > 6) ∧
  (x < 100 → x = 6) ∧ -- Given valid inequality relations and restrictions
  ((2 * x <= 70 ∨ x >= 100) ∨ (4 * x <= 25 ∨ x <= 5)) := -- Certain contradictions inherent

  by sorry

end find_integer_x_l6_6990


namespace consecutive_integers_sum_l6_6324

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 504) : n + n+1 + n+2 = 24 :=
sorry

end consecutive_integers_sum_l6_6324


namespace max_diameter_min_diameter_l6_6431

-- Definitions based on problem conditions
def base_diameter : ℝ := 30
def positive_tolerance : ℝ := 0.03
def negative_tolerance : ℝ := 0.04

-- The corresponding proof problem statements in Lean 4
theorem max_diameter : base_diameter + positive_tolerance = 30.03 := sorry
theorem min_diameter : base_diameter - negative_tolerance = 29.96 := sorry

end max_diameter_min_diameter_l6_6431


namespace sector_properties_l6_6894

-- Definitions for the conditions
def central_angle (α : ℝ) : Prop := α = 2 * Real.pi / 3

def radius (r : ℝ) : Prop := r = 6

def sector_perimeter (l r : ℝ) : Prop := l + 2 * r = 20

-- The statement encapsulating the proof problem
theorem sector_properties :
  (central_angle (2 * Real.pi / 3) ∧ radius 6 →
    ∃ l S, l = 4 * Real.pi ∧ S = 12 * Real.pi) ∧
  (∃ l r, sector_perimeter l r ∧ 
    ∃ α S, α = 2 ∧ S = 25) := by
  sorry

end sector_properties_l6_6894


namespace sum_of_digits_of_greatest_prime_divisor_l6_6881

-- Define the number 32767
def number := 32767

-- Find the greatest prime divisor of 32767
def greatest_prime_divisor : ℕ :=
  127

-- Prove the sum of the digits of the greatest prime divisor is 10
theorem sum_of_digits_of_greatest_prime_divisor (h : greatest_prime_divisor = 127) : (1 + 2 + 7) = 10 :=
  sorry

end sum_of_digits_of_greatest_prime_divisor_l6_6881


namespace hydrochloric_acid_solution_l6_6279

variable (V : ℝ) (pure_acid_added : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ)

theorem hydrochloric_acid_solution :
  initial_concentration = 0.10 → 
  final_concentration = 0.15 → 
  pure_acid_added = 3.52941176471 → 
  0.10 * V + 3.52941176471 = 0.15 * (V + 3.52941176471) → 
  V = 60 :=
by
  intros h_initial h_final h_pure h_equation
  sorry

end hydrochloric_acid_solution_l6_6279


namespace range_of_t_l6_6650
noncomputable def f (x : ℝ) (t : ℝ) : ℝ := Real.exp (2 * x) - t
noncomputable def g (x : ℝ) (t : ℝ) : ℝ := t * Real.exp x - 1

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x t ≥ g x t) ↔ t ≤ 2 * Real.sqrt 2 - 2 :=
by sorry

end range_of_t_l6_6650


namespace problem_1110_1111_1112_1113_l6_6852

theorem problem_1110_1111_1112_1113 (r : ℕ) (hr : r > 5) : 
  (r^3 + r^2 + r) * (r^3 + r^2 + r + 1) * (r^3 + r^2 + r + 2) * (r^3 + r^2 + r + 3) = (r^6 + 2 * r^5 + 3 * r^4 + 5 * r^3 + 4 * r^2 + 3 * r + 1)^2 - 1 :=
by
  sorry

end problem_1110_1111_1112_1113_l6_6852


namespace negation_of_exists_eq_sin_l6_6719

theorem negation_of_exists_eq_sin : ¬ (∃ x : ℝ, x = Real.sin x) ↔ ∀ x : ℝ, x ≠ Real.sin x :=
by
  sorry

end negation_of_exists_eq_sin_l6_6719


namespace find_r_s_l6_6837

noncomputable def parabola_line_intersection (x y m : ℝ) : Prop :=
  y = x^2 + 5*x ∧ y + 6 = m*(x - 10)

theorem find_r_s (r s m : ℝ) (Q : ℝ × ℝ)
  (hq : Q = (10, -6))
  (h_parabola : ∀ x, ∃ y, y = x^2 + 5*x)
  (h_line : ∀ x, ∃ y, y + 6 = m*(x - 10)) :
  parabola_line_intersection x y m → (r < m ∧ m < s) ∧ (r + s = 50) :=
sorry

end find_r_s_l6_6837


namespace find_a_plus_b_minus_c_l6_6263

theorem find_a_plus_b_minus_c (a b c : ℤ) (h1 : 3 * b = 5 * a) (h2 : 7 * a = 3 * c) (h3 : 3 * a + 2 * b - 4 * c = -9) : a + b - c = 1 :=
by
  sorry

end find_a_plus_b_minus_c_l6_6263


namespace find_initial_money_l6_6450

-- Definitions of the conditions
def basketball_card_cost : ℕ := 3
def baseball_card_cost : ℕ := 4
def basketball_packs : ℕ := 2
def baseball_decks : ℕ := 5
def change_received : ℕ := 24

-- Total cost calculation
def total_cost : ℕ := (basketball_card_cost * basketball_packs) + (baseball_card_cost * baseball_decks)

-- Initial money calculation
def initial_money : ℕ := total_cost + change_received

-- Proof statement
theorem find_initial_money : initial_money = 50 := 
by
  -- Proof steps would go here
  sorry

end find_initial_money_l6_6450


namespace gcd_seven_factorial_ten_fact_div_5_fact_l6_6342

def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define 7!
def seven_factorial := factorial 7

-- Define 10! / 5!
def ten_fact_div_5_fact := factorial 10 / factorial 5

-- Prove that the GCD of 7! and (10! / 5!) is 2520
theorem gcd_seven_factorial_ten_fact_div_5_fact :
  Nat.gcd seven_factorial ten_fact_div_5_fact = 2520 := by
sorry

end gcd_seven_factorial_ten_fact_div_5_fact_l6_6342


namespace residue_class_equivalence_l6_6704

variable {a m : ℤ}
variable {b : ℤ}

def residue_class (a m b : ℤ) : Prop := ∃ t : ℤ, b = m * t + a

theorem residue_class_equivalence (m a b : ℤ) :
  (∃ t : ℤ, b = m * t + a) ↔ b % m = a % m :=
by sorry

end residue_class_equivalence_l6_6704


namespace minimum_rectangles_needed_l6_6340

/-- The theorem that defines the minimum number of rectangles needed to cover the specified figure -/
theorem minimum_rectangles_needed 
    (rectangles : ℕ) 
    (figure : Type)
    (covers : figure → Prop) :
  rectangles = 12 :=
sorry

end minimum_rectangles_needed_l6_6340


namespace total_distance_l6_6491

theorem total_distance (x : ℝ) (h : (1/2) * (x - 1) = (1/3) * x + 1) : x = 9 := 
by 
  sorry

end total_distance_l6_6491


namespace inverse_function_l6_6929

theorem inverse_function (x : ℝ) (hx : x > 1) : ∃ y : ℝ, x = 2^y + 1 ∧ y = Real.logb 2 (x - 1) :=
sorry

end inverse_function_l6_6929


namespace count_multiples_of_5_l6_6280

theorem count_multiples_of_5 (a b : ℕ) (h₁ : 50 ≤ a) (h₂ : a ≤ 300) (h₃ : 50 ≤ b) (h₄ : b ≤ 300) (h₅ : a % 5 = 0) (h₆ : b % 5 = 0) 
  (h₇ : ∀ n : ℕ, 50 ≤ n ∧ n ≤ 300 → n % 5 = 0 → a ≤ n ∧ n ≤ b) :
  b = a + 48 * 5 → (b - a) / 5 + 1 = 49 :=
by
  sorry

end count_multiples_of_5_l6_6280


namespace volume_of_sphere_from_cube_surface_area_l6_6968

theorem volume_of_sphere_from_cube_surface_area (S : ℝ) (h : S = 24) : 
  ∃ V : ℝ, V = 4 * Real.sqrt 3 * Real.pi := 
sorry

end volume_of_sphere_from_cube_surface_area_l6_6968


namespace even_sum_three_numbers_probability_l6_6507

theorem even_sum_three_numbers_probability :
  let S := {2, 3, 4, 5, 6}
  (finsub : ℕ → ℕ → ℕ) := @Nat.choose
  (total_combinations := finsub 5 3)
  (even_combinations := 1)
  (odd_even_combinations := 3)
  (probability : ℚ := (even_combinations + odd_even_combinations) / total_combinations)
  (probability = 2 / 5) :=
by
  sorry

end even_sum_three_numbers_probability_l6_6507


namespace parabola_opens_downward_iff_l6_6416

theorem parabola_opens_downward_iff (m : ℝ) : (m - 1 < 0) ↔ (m < 1) :=
by
  sorry

end parabola_opens_downward_iff_l6_6416


namespace even_times_odd_is_even_l6_6921

theorem even_times_odd_is_even {a b : ℤ} (h₁ : ∃ k, a = 2 * k) (h₂ : ∃ j, b = 2 * j + 1) : ∃ m, a * b = 2 * m :=
by
  sorry

end even_times_odd_is_even_l6_6921


namespace remainder_when_dividing_386_l6_6395

theorem remainder_when_dividing_386 :
  (386 % 35 = 1) ∧ (386 % 11 = 1) :=
by
  sorry

end remainder_when_dividing_386_l6_6395


namespace find_n_modulo_l6_6110

theorem find_n_modulo :
  ∀ n : ℤ, (0 ≤ n ∧ n < 25 ∧ -175 % 25 = n % 25) → n = 0 :=
by
  intros n h
  sorry

end find_n_modulo_l6_6110


namespace salary_increase_l6_6439

theorem salary_increase (S : ℝ) (P : ℝ) (H0 : P > 0 )  
  (saved_last_year : ℝ := 0.10 * S)
  (salary_this_year : ℝ := S * (1 + P / 100))
  (saved_this_year : ℝ := 0.15 * salary_this_year)
  (H1 : saved_this_year = 1.65 * saved_last_year) :
  P = 10 :=
by
  sorry

end salary_increase_l6_6439


namespace unit_price_of_each_chair_is_42_l6_6865

-- Definitions from conditions
def total_cost_desks (unit_price_desk : ℕ) (number_desks : ℕ) : ℕ := unit_price_desk * number_desks
def remaining_cost_chairs (total_cost : ℕ) (cost_desks : ℕ) : ℕ := total_cost - cost_desks
def unit_price_chairs (remaining_cost : ℕ) (number_chairs : ℕ) : ℕ := remaining_cost / number_chairs

-- Given conditions
def unit_price_desk := 180
def number_desks := 5
def total_cost := 1236
def number_chairs := 8

-- The question: determining the unit price of each chair
theorem unit_price_of_each_chair_is_42 : 
  unit_price_chairs (remaining_cost_chairs total_cost (total_cost_desks unit_price_desk number_desks)) number_chairs = 42 := sorry

end unit_price_of_each_chair_is_42_l6_6865


namespace cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths_l6_6500

theorem cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths 
  (b : ℝ)
  (h : ∀ x : ℝ, 4 * x^3 + 3 * x^2 + b * x + 27 = 0 → ∃! r : ℝ, r = x) :
  b = 3 / 4 := 
by
  sorry

end cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths_l6_6500


namespace right_triangle_third_angle_l6_6435

-- Define the problem
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Define the given angles
def is_right_angle (a : ℝ) : Prop := a = 90
def given_angle (b : ℝ) : Prop := b = 25

-- Define the third angle
def third_angle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove 
theorem right_triangle_third_angle : ∀ (a b c : ℝ), 
  is_right_angle a → given_angle b → third_angle a b c → c = 65 :=
by
  intros a b c ha hb h_triangle
  sorry

end right_triangle_third_angle_l6_6435


namespace tank_fraction_before_gas_added_l6_6488

theorem tank_fraction_before_gas_added (capacity : ℝ) (added_gasoline : ℝ) (fraction_after : ℝ) (initial_fraction : ℝ) :
  capacity = 42 → added_gasoline = 7 → fraction_after = 9 / 10 → (initial_fraction * capacity + added_gasoline = fraction_after * capacity) → initial_fraction = 733 / 1000 :=
by
  intros h_capacity h_added_gasoline h_fraction_after h_equation
  sorry

end tank_fraction_before_gas_added_l6_6488


namespace regular_polygon_sides_l6_6136

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l6_6136


namespace prod_eq_of_eqs_l6_6549

variable (a : ℝ) (m n p q : ℕ)
variable (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1)
variable (h4 : a^m + a^n = a^p + a^q) (h5 : a^{3*m} + a^{3*n} = a^{3*p} + a^{3*q})

theorem prod_eq_of_eqs : m * n = p * q := by
  sorry

end prod_eq_of_eqs_l6_6549


namespace quadrilateral_area_l6_6454

-- Define the number of interior and boundary points
def interior_points : ℕ := 5
def boundary_points : ℕ := 4

-- State the theorem to prove the area of the quadrilateral using Pick's Theorem
theorem quadrilateral_area : interior_points + (boundary_points / 2) - 1 = 6 := by sorry

end quadrilateral_area_l6_6454


namespace quadratic_roots_distinct_l6_6727

-- Define the quadratic equation condition
def quadratic_eq : (ℝ → ℝ) :=
  λ x m => x^2 + m * x - 8

-- State the problem
theorem quadratic_roots_distinct (m : ℝ) : 
  let Δ := m^2 + 32 in Δ > 0 :=
  by
   -- We need this theorem to state that the equation always has distinct real roots
  let Δ := m^2 + 32
  sorry

end quadratic_roots_distinct_l6_6727


namespace laser_beam_total_distance_l6_6483

theorem laser_beam_total_distance :
  let A := (4, 7)
  let B := (-4, 7)
  let C := (-4, -7)
  let D := (4, -7)
  let E := (9, 7)
  let dist (p1 p2 : (ℤ × ℤ)) : ℝ := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  dist A B + dist B C + dist C D + dist D E = 30 + Real.sqrt 221 :=
by
  sorry

end laser_beam_total_distance_l6_6483


namespace alice_unanswered_questions_l6_6376

theorem alice_unanswered_questions 
    (c w u : ℕ)
    (h1 : 6 * c - 2 * w + 3 * u = 120)
    (h2 : 3 * c - w = 70)
    (h3 : c + w + u = 40) :
    u = 10 :=
sorry

end alice_unanswered_questions_l6_6376


namespace regular_polygon_num_sides_l6_6160

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l6_6160


namespace find_b_l6_6262

noncomputable def triangle_b_value (a : ℝ) (C : ℝ) (area : ℝ) : ℝ :=
  let sin_C := Real.sin C
  let b := (2 * area) / (a * sin_C)
  b

theorem find_b (h₁ : a = 1)
              (h₂ : C = Real.pi / 4)
              (h₃ : area = 2 * a) :
              triangle_b_value a C area = 8 * Real.sqrt 2 :=
by
  -- Definitions imply what we need
  sorry

end find_b_l6_6262


namespace quadratic_inequality_l6_6456

theorem quadratic_inequality (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4 * a * c :=
sorry

end quadratic_inequality_l6_6456


namespace total_apples_collected_l6_6209

theorem total_apples_collected (daily_pick: ℕ) (days: ℕ) (remaining: ℕ) 
  (h_daily_pick: daily_pick = 4) 
  (h_days: days = 30) 
  (h_remaining: remaining = 230) : 
  daily_pick * days + remaining = 350 := 
by
  rw [h_daily_pick, h_days, h_remaining]
  norm_num
  sorry

end total_apples_collected_l6_6209


namespace geometric_progression_terms_l6_6475

theorem geometric_progression_terms (a b r : ℝ) (n : ℕ) (h1 : 0 < r) (h2: a ≠ 0) (h3 : b = a * r^(n-1)) :
  n = 1 + (Real.log (b / a)) / (Real.log r) :=
by sorry

end geometric_progression_terms_l6_6475


namespace fraction_addition_l6_6878

theorem fraction_addition : (3/4) / (5/8) + (1/2) = 17/10 := by
  sorry

end fraction_addition_l6_6878


namespace largest_angle_in_triangle_l6_6063

theorem largest_angle_in_triangle 
  (A B C : ℝ)
  (h_sum_angles: 2 * A + 20 = 105)
  (h_triangle_sum: A + (A + 20) + C = 180)
  (h_A_ge_0: A ≥ 0)
  (h_B_ge_0: B ≥ 0)
  (h_C_ge_0: C ≥ 0) : 
  max A (max (A + 20) C) = 75 := 
by
  -- Placeholder proof
  sorry

end largest_angle_in_triangle_l6_6063


namespace regular_polygon_sides_l6_6202

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6202


namespace solution_comparison_l6_6294

open Real

theorem solution_comparison (c d e f : ℝ) (hc : c ≠ 0) (he : e ≠ 0) :
  (-(d / c) > -(f / e)) ↔ ((f / e) > (d / c)) :=
by
  sorry

end solution_comparison_l6_6294


namespace circumference_of_inscribed_circle_l6_6365

-- Define the dimensions of the rectangle
def width : ℝ := 9
def height : ℝ := 12

-- Define the function to compute the diagonal of the rectangle
def diagonal (w h : ℝ) : ℝ := Real.sqrt (w ^ 2 + h ^ 2)

-- Define the function to compute the circumference of the circle given its diameter
def circumference (d : ℝ) : ℝ := Real.pi * d

-- State the theorem
theorem circumference_of_inscribed_circle :
  circumference (diagonal width height) = 15 * Real.pi := by
  sorry

end circumference_of_inscribed_circle_l6_6365


namespace find_roots_l6_6393

def polynomial (x: ℝ) := x^3 - 2*x^2 - x + 2

theorem find_roots : { x : ℝ // polynomial x = 0 } = ({1, -1, 2} : Set ℝ) :=
by
  sorry

end find_roots_l6_6393


namespace remainder_mod_1220_l6_6607

theorem remainder_mod_1220 (q : ℕ) (h_q : q = 1220) :
  ∃ m n : ℕ, (nat.coprime m n) ∧ (m = 2 * q * q) ∧ (n = (2 * q - 1) * (q - 1)) ∧ ((m + n) % q = 1) :=
by
  -- Define m and n
  let m := 2 * q * q
  let n := (2 * q - 1) * (q - 1)
  use [m, n]
  -- Split the goals
  split
  -- Show that m and n are coprime
  { sorry }
  split
  -- Show that m = 2 * q * q
  { simp [m] }
  split
  -- Show that n = (2 * q - 1) * (q - 1)
  { simp [n] }
  -- Show that (m + n) % q = 1
  { rw h_q, simp [m, n, nat.add_mod], sorry }

end remainder_mod_1220_l6_6607


namespace inequality_proof_l6_6408

variable (x y z : ℝ)

theorem inequality_proof (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) 
  ≥ Real.sqrt (3 / 2 * (x + y + z)) :=
sorry

end inequality_proof_l6_6408


namespace find_m_for_increasing_graph_l6_6966

theorem find_m_for_increasing_graph (m : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → (m + 1) * x ^ (3 - m^2) < (m + 1) * y ^ (3 - m^2) → x < y) ↔ m = -2 :=
by
  sorry

end find_m_for_increasing_graph_l6_6966


namespace parallel_resistor_problem_l6_6064

theorem parallel_resistor_problem
  (x : ℝ)
  (r : ℝ := 2.2222222222222223)
  (y : ℝ := 5) : 
  (1 / r = 1 / x + 1 / y) → x = 4 :=
by sorry

end parallel_resistor_problem_l6_6064


namespace number_of_children_l6_6017

-- Definitions of the conditions
def crayons_per_child : ℕ := 8
def total_crayons : ℕ := 56

-- Statement of the problem
theorem number_of_children : total_crayons / crayons_per_child = 7 := by
  sorry

end number_of_children_l6_6017


namespace find_certain_value_l6_6369

noncomputable def certain_value 
  (total_area : ℝ) (smaller_part : ℝ) (difference_fraction : ℝ) : ℝ :=
  (total_area - 2 * smaller_part) / difference_fraction

theorem find_certain_value (total_area : ℝ) (smaller_part : ℝ) (X : ℝ) : 
  total_area = 700 → 
  smaller_part = 315 → 
  (total_area - 2 * smaller_part) / (1/5) = X → 
  X = 350 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end find_certain_value_l6_6369


namespace prove_mutually_exclusive_and_exhaustive_events_l6_6764

-- Definitions of conditions
def number_of_boys : ℕ := 3
def number_of_girls : ℕ := 2

-- Definitions of options
def option_A : Prop := (∃ (b1 b2 b3 : Bool) (g1 g2 : Bool), b1 ∧ b2 ∧ ¬b3 ∧ ¬g1 ∧ g2)  -- Exactly 1 boy and exactly 2 girls
def option_B : Prop := (∃ (b1 b2 b3 : Bool), b1 ∧ b2 ∧ b3)  -- At least 1 boy and all boys
def option_C : Prop := (∃ (b1 b2 b3 : Bool) (g1 g2 : Bool), b1 ∧ b2 ∧ (b3 ∨ g1 ∨ g2))  -- At least 1 boy and at least 1 girl
def option_D : Prop := (∃ (b1 b2 : Bool) (g3 : Bool), b1 ∧ ¬b2 ∧ g3)  -- At least 1 boy and all girls

-- The proof statement showing that option_D == Mutually Exclusive and Exhaustive Events
theorem prove_mutually_exclusive_and_exhaustive_events : option_D :=
sorry

end prove_mutually_exclusive_and_exhaustive_events_l6_6764


namespace minimum_value_of_PQ_l6_6762

theorem minimum_value_of_PQ {x y : ℝ} (P : ℝ × ℝ) (h₁ : (P.1 - 3)^2 + (P.2 - 4)^2 > 4)
  (h₂ : ∀ Q : ℝ × ℝ, (Q.1 - 3)^2 + (Q.2 - 4)^2 = 4 → (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1)^2 + (P.2)^2) :
  ∃ PQ_min : ℝ, PQ_min = 17/2 := by
  sorry

end minimum_value_of_PQ_l6_6762


namespace partition_exists_l6_6556
open Set Real

theorem partition_exists (r : ℚ) (hr : r > 1) :
  ∃ (A B : ℕ → Prop), (∀ n, A n ∨ B n) ∧ (∀ n, ¬(A n ∧ B n)) ∧ 
  (∀ k l, A k → A l → (k : ℚ) / (l : ℚ) ≠ r) ∧ 
  (∀ k l, B k → B l → (k : ℚ) / (l : ℚ) ≠ r) :=
sorry

end partition_exists_l6_6556


namespace inequality_1_inequality_2_inequality_4_l6_6939

variable {a b : ℝ}

def condition (a b : ℝ) : Prop := (1/a < 1/b) ∧ (1/b < 0)

theorem inequality_1 (ha : a < 0) (hb : b < 0) (hc : condition a b) : a + b < a * b :=
sorry

theorem inequality_2 (hc : condition a b) : |a| < |b| :=
sorry

theorem inequality_4 (hc : condition a b) : (b / a) + (a / b) > 2 :=
sorry

end inequality_1_inequality_2_inequality_4_l6_6939


namespace units_digit_7_pow_103_l6_6748

theorem units_digit_7_pow_103 : Nat.mod (7 ^ 103) 10 = 3 := sorry

end units_digit_7_pow_103_l6_6748


namespace reduced_price_proof_l6_6752

noncomputable def reduced_price (P: ℝ) := 0.88 * P

theorem reduced_price_proof :
  ∃ R P : ℝ, R = reduced_price P ∧ 1200 / R = 1200 / P + 6 ∧ R = 24 :=
by
  sorry

end reduced_price_proof_l6_6752


namespace rational_terms_binomial_expansion_coefficient_x2_expansion_of_polynomials_l6_6645

theorem rational_terms_binomial_expansion
  (n : ℕ)
  (h1 : (∑ r in (range (n/2).succ).map (λ r, ↑(choose n (2*r+1) * ((sqrt x)^(n - (2*r + 1)) * (-(root 3 x))^(2*r+1)))) = (512 : ℕ)) :
  n = 10 ∧ (∃ a b : ℤ, a = 1 ∧ b = 210 ∧ 
  (expansion_term (range (n/2).succ) /\ 
  exp_term r a b = (finite_sum (range (n/2).succ).map 
  λ r, C(n, r) * (x ^ (integral_exponent r))))) := sorry

theorem coefficient_x2_expansion_of_polynomials
  (h2 : (n : ℕ) = 10) :
  (∑ k in range (10 - 3 + 1), (choose (3 + k) 2)) = 164 := sorry

end rational_terms_binomial_expansion_coefficient_x2_expansion_of_polynomials_l6_6645


namespace driving_time_l6_6108

-- Conditions from problem
variable (distance1 : ℕ) (time1 : ℕ) (distance2 : ℕ)
variable (same_speed : distance1 / time1 = distance2 / (5 : ℕ))

-- Statement to prove
theorem driving_time (h1 : distance1 = 120) (h2 : time1 = 3) (h3 : distance2 = 200)
  : distance2 / (40 : ℕ) = (5 : ℕ) := by
  sorry

end driving_time_l6_6108


namespace correct_option_l6_6615

def monomial_structure_same (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

def monomial1 : ℕ → ℕ
| 0 => 1 -- Exponent of a in 3ab^2
| 1 => 2 -- Exponent of b in 3ab^2
| _ => 0

def monomial2 : ℕ → ℕ
| 0 => 1 -- Exponent of a in 4ab^2
| 1 => 2 -- Exponent of b in 4ab^2
| _ => 0

theorem correct_option :
  monomial_structure_same monomial1 monomial2 := sorry

end correct_option_l6_6615


namespace equation_of_line_through_point_with_equal_intercepts_l6_6024

open LinearAlgebra

theorem equation_of_line_through_point_with_equal_intercepts :
  ∃ (a b c : ℝ), (a * 1 + b * 2 + c = 0) ∧ (a * b < 0) ∧ ∀ x y : ℝ, 
  (a * x + b * y + c = 0 ↔ (2 * x - y = 0 ∨ x + y - 3 = 0)) :=
sorry

end equation_of_line_through_point_with_equal_intercepts_l6_6024


namespace fraction_addition_l6_6877

theorem fraction_addition :
  (3 / 4) / (5 / 8) + (1 / 2) = 17 / 10 :=
by
  sorry

end fraction_addition_l6_6877


namespace notebook_distribution_l6_6859

theorem notebook_distribution (x : ℕ) : 
  (∃ k₁ : ℕ, x = 3 * k₁ + 1) ∧ (∃ k₂ : ℕ, x = 4 * k₂ - 2) → (x - 1) / 3 = (x + 2) / 4 :=
by
  sorry

end notebook_distribution_l6_6859


namespace license_plate_combinations_l6_6009

-- Definitions representing the conditions
def valid_license_plates_count : ℕ :=
  let letter_combinations := Nat.choose 26 2 -- Choose 2 unique letters
  let letter_arrangements := Nat.choose 4 2 * 2 -- Arrange the repeated letters
  let digit_combinations := 10 * 9 * 8 -- Choose different digits
  letter_combinations * letter_arrangements * digit_combinations

-- The theorem representing the problem statement
theorem license_plate_combinations :
  valid_license_plates_count = 2808000 := 
  sorry

end license_plate_combinations_l6_6009


namespace fewer_pushups_l6_6010

theorem fewer_pushups (sets: ℕ) (pushups_per_set : ℕ) (total_pushups : ℕ) 
  (h1 : sets = 3) (h2 : pushups_per_set = 15) (h3 : total_pushups = 40) :
  sets * pushups_per_set - total_pushups = 5 :=
by
  sorry

end fewer_pushups_l6_6010


namespace suitable_sampling_method_l6_6911

-- Conditions given
def num_products : ℕ := 40
def num_top_quality : ℕ := 10
def num_second_quality : ℕ := 25
def num_defective : ℕ := 5
def draw_count : ℕ := 8

-- Possible sampling methods
inductive SamplingMethod
| DrawingLots : SamplingMethod
| RandomNumberTable : SamplingMethod
| Systematic : SamplingMethod
| Stratified : SamplingMethod

-- Problem statement (to be proved)
theorem suitable_sampling_method : 
  (num_products = 40) ∧ 
  (num_top_quality = 10) ∧ 
  (num_second_quality = 25) ∧ 
  (num_defective = 5) ∧ 
  (draw_count = 8) → 
  SamplingMethod.Stratified = SamplingMethod.Stratified :=
by sorry

end suitable_sampling_method_l6_6911


namespace veggie_patty_percentage_l6_6489

-- Let's define the weights
def weight_total : ℕ := 150
def weight_additives : ℕ := 45

-- Let's express the proof statement as a theorem
theorem veggie_patty_percentage : (weight_total - weight_additives) * 100 / weight_total = 70 := by
  sorry

end veggie_patty_percentage_l6_6489


namespace proof_problem_l6_6958

variable (a b c d x : ℤ)

-- Conditions
axiom condition1 : a - b = c + d + x
axiom condition2 : a + b = c - d - 3
axiom condition3 : a - c = 3
axiom answer_eq : x = 9

-- Proof statement
theorem proof_problem : (a - b) = (c + d + 9) :=
by
  sorry

end proof_problem_l6_6958


namespace shooting_competition_probabilities_l6_6283

theorem shooting_competition_probabilities (p_A_not_losing p_B_losing : ℝ)
  (h₁ : p_A_not_losing = 0.59)
  (h₂ : p_B_losing = 0.44) :
  (1 - p_B_losing = 0.56) ∧ (p_A_not_losing - p_B_losing = 0.15) :=
by
  sorry

end shooting_competition_probabilities_l6_6283


namespace Carlson_max_jars_l6_6227

theorem Carlson_max_jars (n a : ℕ) (hn : 13 * n = 5 * (8 * n + 9 * a)) : ∃ k : ℕ, k ≤ 23 := 
sorry

end Carlson_max_jars_l6_6227


namespace num_integer_solutions_prime_l6_6938

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 ∧ m < n → n % m ≠ 0

def integer_solutions : List ℤ := [-1, 3]

theorem num_integer_solutions_prime :
  (∀ x ∈ integer_solutions, is_prime (|15 * x^2 - 32 * x - 28|)) ∧ (integer_solutions.length = 2) :=
by
  sorry

end num_integer_solutions_prime_l6_6938


namespace find_x_l6_6969

theorem find_x (x y z : ℕ) (h_pos : 0 < x) (h_pos : 0 < y) (h_pos : 0 < z) (h_eq1 : x + y + z = 37) (h_eq2 : 5 * y = 6 * z) : x = 21 :=
sorry

end find_x_l6_6969


namespace good_jars_l6_6695

def original_cartons : Nat := 50
def jars_per_carton : Nat := 20
def less_cartons_received : Nat := 20
def damaged_jars_per_5_cartons : Nat := 3
def total_damaged_cartons : Nat := 1
def total_good_jars : Nat := 565

theorem good_jars (original_cartons jars_per_carton less_cartons_received damaged_jars_per_5_cartons total_damaged_cartons : Nat) :
  (original_cartons - less_cartons_received) * jars_per_carton 
  - (5 * damaged_jars_per_5_cartons + total_damaged_cartons * jars_per_carton) = total_good_jars := 
by 
  sorry

end good_jars_l6_6695


namespace binary_10101000_is_1133_base_5_l6_6792

def binary_to_decimal (b : Nat) : Nat :=
  128 * (b / 128 % 2) + 64 * (b / 64 % 2) + 32 * (b / 32 % 2) + 16 * (b / 16 % 2) + 8 * (b / 8 % 2) + 4 * (b / 4 % 2) + 2 * (b / 2 % 2) + (b % 2)

def decimal_to_base_5 (d : Nat) : List Nat :=
  if d = 0 then [] else (d % 5) :: decimal_to_base_5 (d / 5)

def binary_to_base_5 (b : Nat) : List Nat :=
  decimal_to_base_5 (binary_to_decimal b)

theorem binary_10101000_is_1133_base_5 :
  binary_to_base_5 168 = [1, 1, 3, 3] := 
by 
  sorry

end binary_10101000_is_1133_base_5_l6_6792


namespace A_3_2_eq_29_l6_6013

def A : ℕ → ℕ → ℕ
| 0, n     => n + 1
| (m + 1), 0 => A m 1
| (m + 1), (n + 1) => A m (A (m + 1) n)

theorem A_3_2_eq_29 : A 3 2 = 29 := by
  sorry

end A_3_2_eq_29_l6_6013


namespace cuboid_ratio_l6_6715

theorem cuboid_ratio (length breadth height: ℕ) (h_length: length = 90) (h_breadth: breadth = 75) (h_height: height = 60) : 
(length / Nat.gcd length (Nat.gcd breadth height) = 6) ∧ 
(breadth / Nat.gcd length (Nat.gcd breadth height) = 5) ∧ 
(height / Nat.gcd length (Nat.gcd breadth height) = 4) := by 
  -- intentionally skipped proof 
  sorry

end cuboid_ratio_l6_6715


namespace arccos_cos_eight_l6_6245

theorem arccos_cos_eight : Real.arccos (Real.cos 8) = 8 - 2 * Real.pi :=
by sorry

end arccos_cos_eight_l6_6245


namespace cylinder_volume_triple_quadruple_l6_6325

theorem cylinder_volume_triple_quadruple (r h : ℝ) (V : ℝ) (π : ℝ) (original_volume : V = π * r^2 * h) 
                                         (original_volume_value : V = 8):
  ∃ V', V' = π * (3 * r)^2 * (4 * h) ∧ V' = 288 :=
by
  sorry

end cylinder_volume_triple_quadruple_l6_6325


namespace chocolate_bars_in_large_box_l6_6599

theorem chocolate_bars_in_large_box
  (small_box_count : ℕ) (chocolate_per_small_box : ℕ)
  (h1 : small_box_count = 20)
  (h2 : chocolate_per_small_box = 25) :
  (small_box_count * chocolate_per_small_box) = 500 :=
by
  sorry

end chocolate_bars_in_large_box_l6_6599


namespace find_parallel_lines_l6_6816

open Real

-- Definitions for the problem conditions
def line1 (a x y : ℝ) : Prop := x + 2 * a * y - 1 = 0
def line2 (a x y : ℝ) : Prop := (2 * a - 1) * x - a * y - 1 = 0

-- Definition of when two lines are parallel in ℝ²
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, (l1 x y → ∃ k, ∀ x' y', l2 x' y' → x = k * x' ∧ y = k * y')

-- Main theorem statement
theorem find_parallel_lines:
  ∀ a : ℝ, (parallel (line1 a) (line2 a)) → (a = 0 ∨ a = 1 / 4) :=
by sorry

end find_parallel_lines_l6_6816


namespace no_infinite_pos_sequence_l6_6388

theorem no_infinite_pos_sequence (α : ℝ) (hα : 0 < α ∧ α < 1) :
  ¬(∃ a : ℕ → ℝ, (∀ n : ℕ, a n > 0) ∧ (∀ n : ℕ, 1 + a (n + 1) ≤ a n + (α / n) * a n)) :=
sorry

end no_infinite_pos_sequence_l6_6388


namespace carlson_max_jars_l6_6239

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l6_6239


namespace cos_squared_identity_l6_6804

theorem cos_squared_identity (α : ℝ) (h : Real.tan (α + π / 4) = 3 / 4) :
    Real.cos (π / 4 - α) ^ 2 = 9 / 25 := by
  sorry

end cos_squared_identity_l6_6804


namespace problem_a_problem_c_l6_6033

variable {a b : ℝ}

theorem problem_a (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 1) : ab ≤ 1 / 8 :=
by
  sorry

theorem problem_c (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 1) : 1 / a + 2 / b ≥ 9 :=
by
  sorry

end problem_a_problem_c_l6_6033


namespace profit_percentage_with_discount_correct_l6_6372

variable (CP SP_without_discount Discounted_SP : ℝ)
variable (profit_without_discount profit_with_discount : ℝ)
variable (discount_percentage profit_percentage_without_discount profit_percentage_with_discount : ℝ)
variable (h1 : CP = 100)
variable (h2 : SP_without_discount = CP + profit_without_discount)
variable (h3 : profit_without_discount = 1.20 * CP)
variable (h4 : Discounted_SP = SP_without_discount - discount_percentage * SP_without_discount)
variable (h5 : discount_percentage = 0.05)
variable (h6 : profit_with_discount = Discounted_SP - CP)
variable (h7 : profit_percentage_with_discount = (profit_with_discount / CP) * 100)

theorem profit_percentage_with_discount_correct : profit_percentage_with_discount = 109 := by
  sorry

end profit_percentage_with_discount_correct_l6_6372


namespace church_distance_l6_6832

def distance_to_church (speed : ℕ) (hourly_rate : ℕ) (flat_fee : ℕ) (total_paid : ℕ) : ℕ :=
  let hours := (total_paid - flat_fee) / hourly_rate
  hours * speed

theorem church_distance :
  distance_to_church 10 30 20 80 = 20 :=
by
  sorry

end church_distance_l6_6832


namespace regular_polygon_sides_l6_6164

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6164


namespace correct_relationships_l6_6510

open Real

theorem correct_relationships (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a + c > b + c) ∧ (1/a < 1/b) := by
    sorry

end correct_relationships_l6_6510


namespace rachel_class_choices_l6_6065

theorem rachel_class_choices : (Nat.choose 8 3) = 56 :=
by
  sorry

end rachel_class_choices_l6_6065


namespace distance_gracie_joe_l6_6953

noncomputable def distance_between_points := Real.sqrt (5^2 + (-1)^2)
noncomputable def joe_point := Complex.mk 3 (-4)
noncomputable def gracie_point := Complex.mk (-2) (-3)

theorem distance_gracie_joe : Complex.abs (joe_point - gracie_point) = distance_between_points := by 
  sorry

end distance_gracie_joe_l6_6953


namespace largest_room_length_l6_6861

theorem largest_room_length (L : ℕ) (w_large w_small l_small diff_area : ℕ)
  (h1 : w_large = 45)
  (h2 : w_small = 15)
  (h3 : l_small = 8)
  (h4 : diff_area = 1230)
  (h5 : w_large * L - (w_small * l_small) = diff_area) :
  L = 30 :=
by sorry

end largest_room_length_l6_6861


namespace fraction_of_Bs_l6_6823

theorem fraction_of_Bs 
  (num_students : ℕ)
  (As_fraction : ℚ)
  (Cs_fraction : ℚ)
  (Ds_number : ℕ)
  (total_students : ℕ) 
  (h1 : As_fraction = 1 / 5) 
  (h2 : Cs_fraction = 1 / 2) 
  (h3 : Ds_number = 40) 
  (h4 : total_students = 800) : 
  num_students / total_students = 1 / 4 :=
by
sorry

end fraction_of_Bs_l6_6823


namespace radius_of_circle_nearest_integer_l6_6912

theorem radius_of_circle_nearest_integer (θ L : ℝ) (hθ : θ = 300) (hL : L = 2000) : 
  abs ((1200 / (Real.pi)) - 382) < 1 := 
by {
  sorry
}

end radius_of_circle_nearest_integer_l6_6912


namespace jonathan_needs_12_bottles_l6_6836

noncomputable def fl_oz_to_liters (fl_oz : ℝ) : ℝ :=
  fl_oz / 33.8

noncomputable def liters_to_ml (liters : ℝ) : ℝ :=
  liters * 1000

noncomputable def num_bottles_needed (ml : ℝ) : ℝ :=
  ml / 150

theorem jonathan_needs_12_bottles :
  num_bottles_needed (liters_to_ml (fl_oz_to_liters 60)) = 12 := 
by
  sorry

end jonathan_needs_12_bottles_l6_6836


namespace max_initial_jars_l6_6242

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l6_6242


namespace train_crosses_signal_post_time_l6_6608

theorem train_crosses_signal_post_time 
  (length_train : ℕ) 
  (length_bridge : ℕ) 
  (time_bridge_minutes : ℕ) 
  (time_signal_post_seconds : ℕ) 
  (h_length_train : length_train = 600) 
  (h_length_bridge : length_bridge = 1800) 
  (h_time_bridge_minutes : time_bridge_minutes = 2) 
  (h_time_signal_post : time_signal_post_seconds = 30) : 
  (length_train / ((length_train + length_bridge) / (time_bridge_minutes * 60))) = time_signal_post_seconds :=
by
  sorry

end train_crosses_signal_post_time_l6_6608


namespace coplanar_vectors_m_value_l6_6536

variable (m : ℝ)
variable (α β : ℝ)
def a := (5, 9, m)
def b := (1, -1, 2)
def c := (2, 5, 1)

theorem coplanar_vectors_m_value :
  ∃ (α β : ℝ), (5 = α + 2 * β) ∧ (9 = -α + 5 * β) ∧ (m = 2 * α + β) → m = 4 :=
by
  sorry

end coplanar_vectors_m_value_l6_6536


namespace black_eyes_ratio_l6_6588

-- Define the number of people in the theater
def total_people : ℕ := 100

-- Define the number of people with blue eyes
def blue_eyes : ℕ := 19

-- Define the number of people with brown eyes
def brown_eyes : ℕ := 50

-- Define the number of people with green eyes
def green_eyes : ℕ := 6

-- Define the number of people with black eyes
def black_eyes : ℕ := total_people - (blue_eyes + brown_eyes + green_eyes)

-- Prove that the ratio of the number of people with black eyes to the total number of people is 1:4
theorem black_eyes_ratio :
  black_eyes * 4 = total_people := by
  sorry

end black_eyes_ratio_l6_6588


namespace find_y_l6_6077

theorem find_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) 
  (h : (2 * a) ^ (2 * b ^ 2) = (a ^ b + y ^ b) ^ 2) : y = 4 * a ^ 2 - a := 
sorry

end find_y_l6_6077


namespace problem1_l6_6568

theorem problem1 (a b : ℝ) : (a - b)^3 + 3 * a * b * (a - b) + b^3 - a^3 = 0 :=
sorry

end problem1_l6_6568


namespace geometric_sequence_a6_l6_6637

theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 3 = 5 / 2) (h2 : a 2 + a 4 = 5 / 4) 
  (h3 : ∀ n, a (n + 1) = a n * q) : a 6 = 1 / 16 :=
by
  sorry

end geometric_sequence_a6_l6_6637


namespace number_of_free_ranging_chickens_l6_6732

-- Define the conditions as constants
def coop_chickens : ℕ := 14
def run_chickens : ℕ := 2 * coop_chickens
def barn_chickens : ℕ := coop_chickens / 2
def total_chickens_in_coop_and_run : ℕ := coop_chickens + run_chickens    
def free_ranging_chickens_condition : ℕ := 2 * run_chickens - 4
def ratio_condition : Prop := total_chickens_in_coop_and_run * 5 = 2 * (total_chickens_in_coop_and_run + free_ranging_chickens_condition)
def target_free_ranging_chickens : ℕ := 105

-- The proof statement
theorem number_of_free_ranging_chickens : 
  total_chickens_in_coop_and_run * 5 = 2 * (total_chickens_in_coop_and_run + target_free_ranging_chickens) →
  free_ranging_chickens_condition = target_free_ranging_chickens :=
by {
  sorry
}

end number_of_free_ranging_chickens_l6_6732


namespace sculpture_cost_in_CNY_l6_6305

theorem sculpture_cost_in_CNY (USD_to_NAD USD_to_CNY cost_NAD : ℝ) :
  USD_to_NAD = 8 → USD_to_CNY = 5 → cost_NAD = 160 → (cost_NAD * (1 / USD_to_NAD) * USD_to_CNY) = 100 :=
by
  intros h1 h2 h3
  sorry

end sculpture_cost_in_CNY_l6_6305


namespace determine_right_triangle_l6_6963

theorem determine_right_triangle (a b c : ℕ) :
  (∀ c b, (c + b) * (c - b) = a^2 → c^2 = a^2 + b^2) ∧
  (∀ A B C, A + B = C → C = 90) ∧
  (a = 3^2 ∧ b = 4^2 ∧ c = 5^2 → a^2 + b^2 ≠ c^2) ∧
  (a = 5 ∧ b = 12 ∧ c = 13 → a^2 + b^2 = c^2) → 
  ( ∃ x y z : ℕ, x = a ∧ y = b ∧ z = c ∧ x^2 + y^2 ≠ z^2 )
:= by
  sorry

end determine_right_triangle_l6_6963


namespace geom_seq_identity_l6_6972

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n, ∃ r, a (n+1) = r * a n

theorem geom_seq_identity (a : ℕ → ℝ) (r : ℝ) (h1 : geometric_sequence a) (h2 : a 2 + a 4 = 2) :
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 4 := 
  sorry

end geom_seq_identity_l6_6972


namespace brady_earns_181_l6_6916

def bradyEarnings (basic_count : ℕ) (gourmet_count : ℕ) (total_cards : ℕ) : ℕ :=
  let basic_earnings := basic_count * 70
  let gourmet_earnings := gourmet_count * 90
  let total_earnings := basic_earnings + gourmet_earnings
  let total_bonus := (total_cards / 100) * 10 + ((total_cards / 100) - 1) * 5
  total_earnings + total_bonus

theorem brady_earns_181 :
  bradyEarnings 120 80 200 = 181 :=
by 
  sorry

end brady_earns_181_l6_6916


namespace sum_of_three_consecutive_integers_product_504_l6_6322

theorem sum_of_three_consecutive_integers_product_504 : 
  ∃ n : ℤ, n * (n + 1) * (n + 2) = 504 ∧ n + (n + 1) + (n + 2) = 24 := 
by
  sorry

end sum_of_three_consecutive_integers_product_504_l6_6322


namespace find_angle_l6_6102

variable (x : ℝ)

theorem find_angle (h1 : x + (180 - x) = 180) (h2 : x + (90 - x) = 90) (h3 : 180 - x = 3 * (90 - x)) : x = 45 := 
by
  sorry

end find_angle_l6_6102


namespace regular_polygon_sides_l6_6155

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i ∈ finRange n, 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6155


namespace sin_beta_value_l6_6527

theorem sin_beta_value (a β : ℝ) (ha : 0 < a ∧ a < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (hcos_a : Real.cos a = 4 / 5)
  (hcos_a_plus_beta : Real.cos (a + β) = 5 / 13) :
  Real.sin β = 63 / 65 :=
sorry

end sin_beta_value_l6_6527


namespace simplify_expression_l6_6504

variable (z : ℝ)

theorem simplify_expression :
  (z - 2 * z + 4 * z - 6 + 3 + 7 - 2) = (3 * z + 2) := by
  sorry

end simplify_expression_l6_6504


namespace intersection_equiv_l6_6654

open Set

def A : Set ℝ := { x | 2 * x < 2 + x }
def B : Set ℝ := { x | 5 - x > 8 - 4 * x }

theorem intersection_equiv : A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } := 
by 
  sorry

end intersection_equiv_l6_6654


namespace infinite_68_in_cells_no_repeats_in_cells_l6_6829

-- Define the spiral placement function
def spiral (n : ℕ) : ℕ := sorry  -- This function should describe the placement of numbers in the spiral

-- Define a function to get the sum of the numbers in the nodes of a cell.
def cell_sum (cell : ℕ) : ℕ := sorry  -- This function should calculate the sum based on the spiral placement.

-- Proving that numbers divisible by 68 appear infinitely many times in cell centers
theorem infinite_68_in_cells : ∀ N : ℕ, ∃ n > N, 68 ∣ cell_sum n :=
by sorry

-- Proving that numbers in cell centers do not repeat
theorem no_repeats_in_cells : ∀ m n : ℕ, m ≠ n → cell_sum m ≠ cell_sum n :=
by sorry

end infinite_68_in_cells_no_repeats_in_cells_l6_6829


namespace translate_quadratic_l6_6334

-- Define the original quadratic function
def original_quadratic (x : ℝ) : ℝ := (x - 2)^2 - 4

-- Define the translation of the graph one unit to the left and two units up
def translated_quadratic (x : ℝ) : ℝ := (x - 1)^2 - 2

-- Statement to be proved
theorem translate_quadratic :
  ∀ x : ℝ, translated_quadratic x = original_quadratic (x-1) + 2 :=
by
  intro x
  unfold translated_quadratic original_quadratic
  sorry

end translate_quadratic_l6_6334


namespace subset_contains_square_l6_6657

theorem subset_contains_square {A : Finset ℕ} (hA₁ : A ⊆ Finset.range 101) (hA₂ : A.card = 50) (hA₃ : ∀ x ∈ A, ∀ y ∈ A, x + y ≠ 100) : 
  ∃ x ∈ A, ∃ k : ℕ, x = k^2 := 
sorry

end subset_contains_square_l6_6657


namespace factorization_identity_l6_6392

theorem factorization_identity (m : ℝ) : m^3 - m = m * (m + 1) * (m - 1) :=
by
  sorry

end factorization_identity_l6_6392


namespace probability_of_picking_letter_in_mathematics_l6_6959

-- Definitions and conditions
def total_letters : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Theorem to be proven
theorem probability_of_picking_letter_in_mathematics :
  probability unique_letters_in_mathematics total_letters = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l6_6959


namespace no_n_geq_2_makes_10101n_prime_l6_6401

theorem no_n_geq_2_makes_10101n_prime : ∀ n : ℕ, n ≥ 2 → ¬ Prime (n^4 + n^2 + 1) :=
by
  sorry

end no_n_geq_2_makes_10101n_prime_l6_6401


namespace max_min_magnitude_of_sum_l6_6050

open Real

-- Define the vectors a and b and their magnitudes
variables {a b : ℝ × ℝ}
variable (h_a : ‖a‖ = 5)
variable (h_b : ‖b‖ = 2)

-- Define the constant 7 and 3 for the max and min values
noncomputable def max_magnitude : ℝ := 7
noncomputable def min_magnitude : ℝ := 3

-- State the theorem
theorem max_min_magnitude_of_sum (h_a : ‖a‖ = 5) (h_b : ‖b‖ = 2) :
  ‖a + b‖ ≤ max_magnitude ∧ ‖a + b‖ ≥ min_magnitude :=
by {
  sorry -- Proof goes here
}

end max_min_magnitude_of_sum_l6_6050


namespace flower_bed_profit_l6_6122

theorem flower_bed_profit (x : ℤ) :
  (3 + x) * (10 - x) = 40 :=
sorry

end flower_bed_profit_l6_6122


namespace largest_root_vieta_l6_6930

theorem largest_root_vieta 
  (a b c : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a * b + a * c + b * c = 11)
  (h3 : a * b * c = -6) : 
  max a (max b c) = 3 :=
sorry

end largest_root_vieta_l6_6930


namespace find_ordered_pair_l6_6857

theorem find_ordered_pair (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (hroots : ∀ x, x^2 + c * x + d = (x - c) * (x - d)) : 
  (c, d) = (1, -2) :=
sorry

end find_ordered_pair_l6_6857


namespace inequality_l6_6522

-- Given three distinct positive real numbers a, b, c
variables {a b c : ℝ}

-- Assume a, b, and c are distinct and positive
axiom distinct_positive (h: a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha: a > 0) (hb: b > 0) (hc: c > 0) : 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0

-- The inequality to be proven
theorem inequality (h: a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha: a > 0) (hb: b > 0) (hc: c > 0) :
  (a / b) + (b / c) > (a / c) + (c / a) := 
sorry

end inequality_l6_6522


namespace maximal_n_for_sequence_l6_6825

theorem maximal_n_for_sequence
  (a : ℕ → ℤ)
  (n : ℕ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n - 2 → a i + a (i + 1) + a (i + 2) > 0)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n - 4 → a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) < 0)
  : n ≤ 9 :=
sorry

end maximal_n_for_sequence_l6_6825


namespace solution_interval_for_x_l6_6021

theorem solution_interval_for_x (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 48) ↔ (48 / 7 ≤ x ∧ x < 49 / 7) :=
by sorry

end solution_interval_for_x_l6_6021


namespace walking_representation_l6_6658

-- Definitions based on conditions
def represents_walking_eastward (m : ℤ) : Prop := m > 0

-- The theorem to prove based on the problem statement
theorem walking_representation :
  represents_walking_eastward 5 →
  ¬ represents_walking_eastward (-10) ∧ abs (-10) = 10 :=
by
  sorry

end walking_representation_l6_6658


namespace find_m_prove_inequality_l6_6414

-- Using noncomputable to handle real numbers where needed
noncomputable def f (x m : ℝ) := m - |x - 1|

-- First proof: Find m given conditions on f(x)
theorem find_m (m : ℝ) :
  (∀ x, f (x + 2) m + f (x - 2) m ≥ 0 ↔ -2 ≤ x ∧ x ≤ 4) → m = 3 :=
sorry

-- Second proof: Prove the inequality given m = 3
theorem prove_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / a + 1 / (2 * b) + 1 / (3 * c) = 3) → a + 2 * b + 3 * c ≥ 3 :=
sorry

end find_m_prove_inequality_l6_6414


namespace train_crossing_time_l6_6045

-- Definitions of the given conditions
def length_of_train : ℝ := 110
def speed_of_train_kmph : ℝ := 72
def length_of_bridge : ℝ := 175

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600

noncomputable def total_distance : ℝ := length_of_train + length_of_bridge

noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross_bridge = 14.25 := 
sorry

end train_crossing_time_l6_6045


namespace pants_price_l6_6329

theorem pants_price (P B : ℝ) 
  (condition1 : P + B = 70.93)
  (condition2 : P = B - 2.93) : 
  P = 34.00 :=
by
  sorry

end pants_price_l6_6329


namespace more_cats_than_dogs_l6_6918

-- Define the initial conditions
def initial_cats : ℕ := 28
def initial_dogs : ℕ := 18
def cats_adopted : ℕ := 3

-- Compute the number of cats after adoption
def cats_now : ℕ := initial_cats - cats_adopted

-- Define the target statement
theorem more_cats_than_dogs : cats_now - initial_dogs = 7 := by
  unfold cats_now
  unfold initial_cats
  unfold cats_adopted
  unfold initial_dogs
  sorry

end more_cats_than_dogs_l6_6918


namespace quotient_of_polynomial_l6_6934

theorem quotient_of_polynomial (x : ℤ) :
  (x^6 + 8) = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 9 :=
by { sorry }

end quotient_of_polynomial_l6_6934


namespace quadratic_equation_solutions_l6_6101

theorem quadratic_equation_solutions (x : ℝ) : x * (x - 7) = 0 ↔ x = 0 ∨ x = 7 :=
by
  sorry

end quadratic_equation_solutions_l6_6101


namespace smallest_constant_l6_6935

theorem smallest_constant (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (a^2 + b^2 + a * b) / c^2 ≥ 3 / 4 :=
sorry

end smallest_constant_l6_6935


namespace anne_cleaning_time_l6_6788

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l6_6788


namespace area_within_square_outside_semicircles_l6_6540

theorem area_within_square_outside_semicircles (side_length : ℝ) (r : ℝ) (area_square : ℝ) (area_semicircles : ℝ) (area_shaded : ℝ) 
  (h1 : side_length = 4)
  (h2 : r = side_length / 2)
  (h3 : area_square = side_length * side_length)
  (h4 : area_semicircles = 4 * (1 / 2 * π * r^2))
  (h5 : area_shaded = area_square - area_semicircles)
  : area_shaded = 16 - 8 * π :=
sorry

end area_within_square_outside_semicircles_l6_6540


namespace total_shoes_in_box_l6_6481

theorem total_shoes_in_box (pairs : ℕ) (prob_matching : ℚ) (h1 : pairs = 7) (h2 : prob_matching = 1 / 13) : 
  ∃ (n : ℕ), n = 2 * pairs ∧ n = 14 :=
by 
  sorry

end total_shoes_in_box_l6_6481


namespace range_of_k_l6_6411

theorem range_of_k (a b c d k : ℝ) (hA : b = k * a - 2 * a - 1) (hB : d = k * c - 2 * c - 1) (h_diff : a ≠ c) (h_lt : (c - a) * (d - b) < 0) : k < 2 := 
sorry

end range_of_k_l6_6411


namespace graph_must_pass_l6_6689

variable (f : ℝ → ℝ)
variable (finv : ℝ → ℝ)
variable (h_inv : ∀ y, f (finv y) = y ∧ finv (f y) = y)
variable (h_point : (2 - f 2) = 5)

theorem graph_must_pass : finv (-3) + 3 = 5 :=
by
  -- Proof to be filled in
  sorry

end graph_must_pass_l6_6689


namespace painted_by_all_three_l6_6737

/-
Statement: Given that 75% of the floor is painted red, 70% painted green, and 65% painted blue,
prove that at least 10% of the floor is painted with all three colors.
-/

def painted_by_red (floor : ℝ) : ℝ := 0.75 * floor
def painted_by_green (floor : ℝ) : ℝ := 0.70 * floor
def painted_by_blue (floor : ℝ) : ℝ := 0.65 * floor

theorem painted_by_all_three (floor : ℝ) :
  ∃ (x : ℝ), x = 0.10 * floor ∧
  (painted_by_red floor) + (painted_by_green floor) + (painted_by_blue floor) ≥ 2 * floor :=
sorry

end painted_by_all_three_l6_6737


namespace regular_polygon_num_sides_l6_6159

theorem regular_polygon_num_sides (n : ℕ) (h1 : (∀ a ∈ angles, a = 150)) (h2 : ∀ a, a = (n - 2) * 180 / n) : n = 12 :=
sorry

end regular_polygon_num_sides_l6_6159


namespace equal_pair_c_l6_6005

theorem equal_pair_c : (-4)^3 = -(4^3) := 
by {
  sorry
}

end equal_pair_c_l6_6005


namespace profit_at_end_of_first_year_l6_6705

theorem profit_at_end_of_first_year :
  let total_amount := 50000
  let part1 := 30000
  let interest_rate1 := 0.10
  let part2 := total_amount - part1
  let interest_rate2 := 0.20
  let time_period := 1
  let interest1 := part1 * interest_rate1 * time_period
  let interest2 := part2 * interest_rate2 * time_period
  let total_profit := interest1 + interest2
  total_profit = 7000 := 
by 
  sorry

end profit_at_end_of_first_year_l6_6705


namespace polygon_with_150_degree_interior_angles_has_12_sides_l6_6174

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l6_6174


namespace find_cost_of_article_l6_6962

-- Define the given conditions and the corresponding proof statement.
theorem find_cost_of_article
  (tax_rate : ℝ) (selling_price1 : ℝ)
  (selling_price2 : ℝ) (profit_increase_rate : ℝ)
  (cost : ℝ) : tax_rate = 0.05 →
              selling_price1 = 360 →
              selling_price2 = 340 →
              profit_increase_rate = 0.05 →
              (selling_price1 / (1 + tax_rate) - cost = 1.05 * (selling_price2 / (1 + tax_rate) - cost)) →
              cost = 57.13 :=
by sorry

end find_cost_of_article_l6_6962


namespace find_d_l6_6757

variable (d x : ℕ)
axiom balls_decomposition : d = x + (x + 1) + (x + 2)
axiom probability_condition : (x : ℚ) / (d : ℚ) < 1 / 6

theorem find_d : d = 3 := sorry

end find_d_l6_6757


namespace regular_polygon_sides_l6_6146

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6146


namespace option_A_option_B_option_C_option_D_l6_6001

theorem option_A : (-(-1) : ℤ) ≠ -|(-1 : ℤ)| := by
  sorry

theorem option_B : ((-3)^2 : ℤ) ≠ -(3^2 : ℤ) := by
  sorry

theorem option_C : ((-4)^3 : ℤ) = -(4^3 : ℤ) := by
  sorry

theorem option_D : ((2^2 : ℚ)/3) ≠ ((2/3)^2 : ℚ) := by
  sorry

end option_A_option_B_option_C_option_D_l6_6001


namespace regular_polygon_sides_l6_6131

theorem regular_polygon_sides (A : ℝ) (h : A = 150) : ∃ n : ℕ, (n > 2) ∧ (180 * (n - 2) / n = A) ∧ (n = 12) := by
  sorry

end regular_polygon_sides_l6_6131


namespace distance_between_islands_l6_6591

theorem distance_between_islands (AB : ℝ) (angle_BAC angle_ABC : ℝ) : 
  AB = 20 ∧ angle_BAC = 60 ∧ angle_ABC = 75 → 
  (∃ BC : ℝ, BC = 10 * Real.sqrt 6) := by
  intro h
  sorry

end distance_between_islands_l6_6591


namespace complement_U_A_l6_6950

-- Definitions of U and A based on problem conditions
def U : Set ℤ := {-1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 2}

-- Definition of the complement in Lean
def complement (A B : Set ℤ) : Set ℤ := {x | x ∈ A ∧ x ∉ B}

-- The main statement to be proved
theorem complement_U_A :
  complement U A = {1, 3} :=
sorry

end complement_U_A_l6_6950


namespace equation_equiv_product_zero_l6_6096

theorem equation_equiv_product_zero (a b x y : ℝ) :
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1) →
  ∃ (m n p : ℤ), (a^m * x - a^n) * (a^p * y - a^3) = a^5 * b^5 ∧ m * n * p = 0 :=
by
  intros h
  sorry

end equation_equiv_product_zero_l6_6096


namespace ten_times_product_is_2010_l6_6559

theorem ten_times_product_is_2010 (n : ℕ) (hn : 10 ≤ n ∧ n < 100) : 
  (∃ k : ℤ, 4.02 * (n : ℝ) = k) → (10 * k = 2010) :=
by
  sorry

end ten_times_product_is_2010_l6_6559


namespace find_original_number_l6_6434

def is_valid_digit (d : ℕ) : Prop := d < 10

def original_number (a b c : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
  222 * (a + b + c) - 5 * (100 * a + 10 * b + c) = 3194

theorem find_original_number (a b c : ℕ) (h_valid: is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c)
  (h_sum : 222 * (a + b + c) - 5 * (100 * a + 10 * b + c) = 3194) : 
  100 * a + 10 * b + c = 358 := 
sorry

end find_original_number_l6_6434


namespace percentage_of_mortality_l6_6053

theorem percentage_of_mortality
  (P : ℝ) -- The population size could be represented as a real number
  (affected_fraction : ℝ) (dead_fraction : ℝ)
  (h1 : affected_fraction = 0.15) -- 15% of the population is affected
  (h2 : dead_fraction = 0.08) -- 8% of the affected population died
: (affected_fraction * dead_fraction) * 100 = 1.2 :=
by
  sorry

end percentage_of_mortality_l6_6053


namespace reflected_coordinates_l6_6095

-- Define the coordinates of point P
def point_P : ℝ × ℝ := (-2, -3)

-- Define the function for reflection across the origin
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- State the theorem to prove
theorem reflected_coordinates :
  reflect_origin point_P = (2, 3) := by
  sorry

end reflected_coordinates_l6_6095


namespace regular_polygon_sides_l6_6163

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i ∈ {1, ..., n} → (interior_angle : ℕ) = 150) 
                               (h2 : (sum_of_interior_angles : ℕ) = 180 * (n - 2)) 
                               (h3 : (regular : Prop)) : 
  n = 12 :=
by
  sorry

end regular_polygon_sides_l6_6163


namespace cost_of_iPhone_l6_6842

theorem cost_of_iPhone (P : ℝ) 
  (phone_contract_cost : ℝ := 200)
  (case_percent_of_P : ℝ := 0.20)
  (headphones_percent_of_case : ℝ := 0.50)
  (total_yearly_cost : ℝ := 3700) :
  let year_phone_contract_cost := (phone_contract_cost * 12)
  let case_cost := (case_percent_of_P * P)
  let headphones_cost := (headphones_percent_of_case * case_cost)
  P + year_phone_contract_cost + case_cost + headphones_cost = total_yearly_cost → 
  P = 1000 :=
by
  sorry  -- proof not required

end cost_of_iPhone_l6_6842


namespace regular_polygon_sides_l6_6133

theorem regular_polygon_sides (n : ℕ) (h : ∀ i : fin n, (interior_angle n) = 150) : n = 12 :=
by
  sorry

def interior_angle (n : ℕ) := 180 * (n - 2) / n

end regular_polygon_sides_l6_6133


namespace crayons_divided_equally_l6_6802

theorem crayons_divided_equally (total_crayons : ℕ) (number_of_people : ℕ) (crayons_per_person : ℕ) 
  (h1 : total_crayons = 24) (h2 : number_of_people = 3) : 
  crayons_per_person = total_crayons / number_of_people → crayons_per_person = 8 :=
by
  intro h
  rw [h1, h2] at h
  have : 24 / 3 = 8 := by norm_num
  rw [this] at h
  exact h

end crayons_divided_equally_l6_6802


namespace problem_statement_l6_6940

open Real

noncomputable def to_prove (x : ℝ) : Prop :=
  tan x = -2 ∧ (π / 2 < x ∧ x < π) → cos x = - (sqrt 5) / 5

theorem problem_statement (x : ℝ) : to_prove x := by
  sorry

end problem_statement_l6_6940


namespace calculate_expression_l6_6619

-- Theorem statement for the provided problem
theorem calculate_expression :
  ((18 ^ 15 / 18 ^ 14)^3 * 8 ^ 3) / 4 ^ 5 = 2916 := by
  sorry

end calculate_expression_l6_6619


namespace min_m_n_l6_6318

def smallest_m_plus_n (m n : ℕ) : Prop :=
  1 < m ∧ 
  let interval_length := ((m : ℝ) / n) - (1 / (m * n)) in
  interval_length = 1 / 2013 ∧
  ∀ (m' n' : ℕ), 1 < m' ∧
    let interval_length' := ((m' : ℝ) / n') - (1 / (m' * n')) in
    interval_length' = 1 / 2013 → (m + n) ≤ (m' + n')

theorem min_m_n (m n : ℕ) (h : smallest_m_plus_n m n) :
  m + n = 5371 :=
sorry

end min_m_n_l6_6318


namespace regular_polygon_sides_l6_6189

theorem regular_polygon_sides (n : ℕ) (h₁ : ∀ i, interior_angle i n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l6_6189


namespace total_miles_l6_6403

theorem total_miles (miles_Katarina miles_Harriet miles_Tomas miles_Tyler : ℕ)
  (hK : miles_Katarina = 51)
  (hH : miles_Harriet = 48)
  (hT : miles_Tomas = 48)
  (hTy : miles_Tyler = 48) :
  miles_Katarina + miles_Harriet + miles_Tomas + miles_Tyler = 195 :=
  by
    sorry

end total_miles_l6_6403


namespace solve_for_a_l6_6821

theorem solve_for_a
  (h : ∀ x : ℝ, (1 < x ∧ x < 2) ↔ (x^2 - a * x + 2 < 0)) :
  a = 3 :=
sorry

end solve_for_a_l6_6821


namespace maximum_initial_jars_l6_6236

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l6_6236


namespace maximum_initial_jars_l6_6233

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l6_6233


namespace solve_inequality_l6_6855

noncomputable def rational_inequality_solution (x : ℝ) : Prop :=
  3 - (x^2 - 4 * x - 5) / (3 * x + 2) > 1

theorem solve_inequality (x : ℝ) :
  rational_inequality_solution x ↔ (x > -2 / 3 ∧ x < 9) :=
by
  sorry

end solve_inequality_l6_6855


namespace point_on_angle_bisector_l6_6055

theorem point_on_angle_bisector (a : ℝ) 
  (h : (2 : ℝ) * a + (3 : ℝ) = a) : a = -3 :=
sorry

end point_on_angle_bisector_l6_6055


namespace problem_sum_of_pairwise_prime_product_l6_6634

theorem problem_sum_of_pairwise_prime_product:
  ∃ a b c d: ℕ, a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧
  a * b * c * d = 288000 ∧
  gcd a b = 1 ∧ gcd a c = 1 ∧ gcd a d = 1 ∧
  gcd b c = 1 ∧ gcd b d = 1 ∧ gcd c d = 1 ∧
  a + b + c + d = 390 :=
sorry

end problem_sum_of_pairwise_prime_product_l6_6634


namespace cost_of_sculpture_cny_l6_6307

def exchange_rate_usd_to_nad := 8 -- 1 USD = 8 NAD
def exchange_rate_usd_to_cny := 5  -- 1 USD = 5 CNY
def cost_of_sculpture_nad := 160  -- Cost of sculpture in NAD

theorem cost_of_sculpture_cny : (cost_of_sculpture_nad / exchange_rate_usd_to_nad) * exchange_rate_usd_to_cny = 100 := by
  sorry

end cost_of_sculpture_cny_l6_6307


namespace share_apples_l6_6088

theorem share_apples (h : 9 / 3 = 3) : true :=
sorry

end share_apples_l6_6088


namespace optimal_selling_price_minimize_loss_l6_6988

theorem optimal_selling_price_minimize_loss 
  (C : ℝ) (h1 : 17 * C = 720 + 5 * C) 
  (h2 : ∀ x : ℝ, x * (1 - 0.1) = 720 * 0.9)
  (h3 : ∀ y : ℝ, y * (1 + 0.05) = 648 * 1.05)
  (selling_price : ℝ)
  (optimal_selling_price : selling_price = 60) :
  selling_price = C :=
by 
  sorry

end optimal_selling_price_minimize_loss_l6_6988


namespace min_value_x_plus_2y_l6_6513

open Real

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) :
  x + 2 * y ≥ 16 :=
sorry

end min_value_x_plus_2y_l6_6513


namespace differences_impossible_l6_6806

def sum_of_digits (n : ℕ) : ℕ :=
  -- A simple definition for the sum of digits function
  n.digits 10 |>.sum

theorem differences_impossible (a : Fin 100 → ℕ) :
    ¬∃ (perm : Fin 100 → Fin 100), 
      (∀ i, a i - sum_of_digits (a (perm (i : ℕ) % 100)) = i + 1) :=
by
  sorry

end differences_impossible_l6_6806


namespace inequality_proof_l6_6417

theorem inequality_proof
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (ha1 : 0 < a1) (hb1 : 0 < b1) (hc1 : 0 < c1)
  (ha2 : 0 < a2) (hb2 : 0 < b2) (hc2 : 0 < c2)
  (h1: b1^2 ≤ a1 * c1)
  (h2: b2^2 ≤ a2 * c2) :
  (a1 + a2 + 5) * (c1 + c2 + 2) > (b1 + b2 + 3)^2 :=
by
  sorry

end inequality_proof_l6_6417


namespace angle_ABC_is_50_l6_6524

theorem angle_ABC_is_50
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h1 : a = 90)
  (h2 : b = 60)
  (h3 : a + b + c = 200): c = 50 := by
  rw [h1, h2] at h3
  linarith

end angle_ABC_is_50_l6_6524


namespace evaluate_expression_l6_6924

theorem evaluate_expression : (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := 
by 
  sorry

end evaluate_expression_l6_6924


namespace negation_of_every_square_positive_l6_6718

theorem negation_of_every_square_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, n^2 ≤ 0 := sorry

end negation_of_every_square_positive_l6_6718


namespace sandwiches_per_person_l6_6257

-- Definitions derived from conditions
def cost_of_12_croissants := 8.0
def number_of_people := 24
def total_spending := 32.0
def croissants_per_set := 12

-- Statement to be proved
theorem sandwiches_per_person :
  ∀ (cost_of_12_croissants total_spending croissants_per_set number_of_people : ℕ),
  total_spending / cost_of_12_croissants * croissants_per_set / number_of_people = 2 :=
by
  sorry

end sandwiches_per_person_l6_6257


namespace regular_polygon_sides_l6_6179

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l6_6179


namespace regular_polygon_sides_l6_6193

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → interior_angle = 150) : n = 12 :=
by
  -- Define the formula for the sum of the interior angles of an n-sided polygon
  have H1 : 180 * (n - 2) = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Solving the equation to find n
  have H2 : 180 * n - 360 = 150 * n,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Further simplification
  have H3 : 30 * n = 360,
  {
    sorry -- Skipping the proof as per the instruction
  },
  -- Dividing both sides by 30
  have H4 : n = 12,
  {
    sorry -- Skipping the proof as per the instruction
  },
  exact H4

end regular_polygon_sides_l6_6193


namespace regular_polygon_sides_l6_6178

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l6_6178


namespace cost_per_square_meter_l6_6613

-- Definitions from conditions
def lawn_length : ℝ := 80
def lawn_breadth : ℝ := 50
def road_width : ℝ := 10
def total_cost : ℝ := 3600

-- Theorem to prove the cost per square meter of traveling the roads
theorem cost_per_square_meter :
  total_cost / 
  ((lawn_length * road_width) + (lawn_breadth * road_width) - (road_width * road_width)) = 3 := by
  sorry

end cost_per_square_meter_l6_6613


namespace complement_of_A_l6_6419

def A : Set ℝ := {y : ℝ | ∃ (x : ℝ), y = 2^x}

theorem complement_of_A : (Set.compl A) = {y : ℝ | y ≤ 0} :=
by
  sorry

end complement_of_A_l6_6419
