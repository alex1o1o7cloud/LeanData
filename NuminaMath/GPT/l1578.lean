import Mathlib

namespace distinct_socks_pairs_l1578_157802

theorem distinct_socks_pairs (n : ℕ) (h : n = 9) : (Nat.choose n 2) = 36 := by
  rw [h]
  norm_num
  sorry

end distinct_socks_pairs_l1578_157802


namespace additional_increment_charge_cents_l1578_157804

-- Conditions as definitions
def first_increment_charge_cents : ℝ := 3.10
def total_charge_8_minutes_cents : ℝ := 18.70
def total_minutes : ℝ := 8
def increments_per_minute : ℝ := 5
def total_increments : ℝ := total_minutes * increments_per_minute
def remaining_increments : ℝ := total_increments - 1
def remaining_charge_cents : ℝ := total_charge_8_minutes_cents - first_increment_charge_cents

-- Proof problem: What is the charge for each additional 1/5 of a minute?
theorem additional_increment_charge_cents : remaining_charge_cents / remaining_increments = 0.40 := by
  sorry

end additional_increment_charge_cents_l1578_157804


namespace find_n_l1578_157830

theorem find_n (a1 a2 : ℕ) (s2 s1 : ℕ) (n : ℕ) :
    a1 = 12 →
    a2 = 3 →
    s2 = 3 * s1 →
    ∃ n : ℕ, a1 / (1 - a2/a1) = 16 ∧
             a1 / (1 - (a2 + n) / a1) = s2 →
             n = 6 :=
by
  intros
  sorry

end find_n_l1578_157830


namespace odd_and_increasing_l1578_157886

-- Define the function f(x) = e^x - e^{-x}
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- We want to prove that this function is both odd and increasing.
theorem odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
sorry

end odd_and_increasing_l1578_157886


namespace triangle_ratio_l1578_157893

theorem triangle_ratio
  (D E F X : Type)
  [DecidableEq D] [DecidableEq E] [DecidableEq F] [DecidableEq X]
  (DE DF : ℝ)
  (hDE : DE = 36)
  (hDF : DF = 40)
  (DX_bisects_EDF : ∀ EX FX, (DE * FX = DF * EX)) :
  ∃ (EX FX : ℝ), EX / FX = 9 / 10 :=
sorry

end triangle_ratio_l1578_157893


namespace arithmetic_sequence_equality_l1578_157847

theorem arithmetic_sequence_equality {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (a20 : a ≠ c) (a2012 : b ≠ c) 
(h₄ : ∀ (i : ℕ), ∃ d : ℝ, a_n = a + i * d) : 
  1992 * a * c - 1811 * b * c - 181 * a * b = 0 := 
by {
  sorry
}

end arithmetic_sequence_equality_l1578_157847


namespace mod_equivalence_l1578_157854

theorem mod_equivalence (a b : ℤ) (d : ℕ) (hd : d ≠ 0) 
  (a' b' : ℕ) (ha' : a % d = a') (hb' : b % d = b') : (a ≡ b [ZMOD d]) ↔ a' = b' := 
sorry

end mod_equivalence_l1578_157854


namespace proof_problem_l1578_157857

-- Conditions
def a : ℤ := 1
def b : ℤ := 0
def c : ℤ := -1 + 3

-- Proof Statement
theorem proof_problem : (2 * a + 3 * c) * b = 0 := by
  sorry

end proof_problem_l1578_157857


namespace jack_pays_back_l1578_157809

-- Define the principal amount P and interest rate r
def principal_amount : ℝ := 1200
def interest_rate : ℝ := 0.1

-- Define the interest and the total amount Jack has to pay back
def interest : ℝ := interest_rate * principal_amount
def total_amount : ℝ := principal_amount + interest

-- State the theorem to prove that the total amount Jack pays back is 1320
theorem jack_pays_back : total_amount = 1320 := by
  sorry

end jack_pays_back_l1578_157809


namespace irreducible_fraction_for_any_n_l1578_157895

theorem irreducible_fraction_for_any_n (n : ℤ) : Int.gcd (14 * n + 3) (21 * n + 4) = 1 := 
by {
  sorry
}

end irreducible_fraction_for_any_n_l1578_157895


namespace hyperbola_eccentricity_cond_l1578_157890

def hyperbola_eccentricity_condition (m : ℝ) : Prop :=
  let a := Real.sqrt m
  let b := Real.sqrt 3
  let c := Real.sqrt (m + 3)
  let e := 2
  (e * e) = (c * c) / (a * a)

theorem hyperbola_eccentricity_cond (m : ℝ) :
  hyperbola_eccentricity_condition m ↔ m = 1 :=
by
  sorry

end hyperbola_eccentricity_cond_l1578_157890


namespace value_of_n_l1578_157861

-- Definitions of the question and conditions
def is_3_digit_integer (x : ℕ) : Prop := 100 ≤ x ∧ x < 1000
def not_divisible_by (x : ℕ) (d : ℕ) : Prop := ¬ (d ∣ x)

def problem (m n : ℕ) : Prop :=
  lcm m n = 690 ∧ is_3_digit_integer n ∧ not_divisible_by n 3 ∧ not_divisible_by m 2

-- The theorem to prove
theorem value_of_n {m n : ℕ} (h : problem m n) : n = 230 :=
sorry

end value_of_n_l1578_157861


namespace factorization_a_squared_minus_3a_l1578_157841

theorem factorization_a_squared_minus_3a (a : ℝ) : a^2 - 3 * a = a * (a - 3) := 
by 
  sorry

end factorization_a_squared_minus_3a_l1578_157841


namespace rectangle_area_l1578_157865

theorem rectangle_area (y : ℝ) (w : ℝ) : 
  (3 * w) ^ 2 + w ^ 2 = y ^ 2 → 
  3 * w * w = (3 / 10) * y ^ 2 :=
by
  intro h
  sorry

end rectangle_area_l1578_157865


namespace least_positive_integer_solution_l1578_157805

theorem least_positive_integer_solution :
  ∃ x : ℕ, (x + 7391) % 12 = 167 % 12 ∧ x = 8 :=
by 
  sorry

end least_positive_integer_solution_l1578_157805


namespace coloring_satisfies_conditions_l1578_157869

-- Definitions of point colors
inductive Color
| Red
| White
| Black

def color_point (x y : ℤ) : Color :=
  if (x + y) % 2 = 1 then Color.Red
  else if (x % 2 = 1 ∧ y % 2 = 0) then Color.White
  else Color.Black

-- Problem statement
theorem coloring_satisfies_conditions :
  (∀ y : ℤ, ∃ x1 x2 x3 : ℤ, 
    color_point x1 y = Color.Red ∧ 
    color_point x2 y = Color.White ∧
    color_point x3 y = Color.Black)
  ∧ 
  (∀ (x1 y1 x2 y2 x3 y3 : ℤ),
    color_point x1 y1 = Color.White →
    color_point x2 y2 = Color.Red →
    color_point x3 y3 = Color.Black →
    ∃ x4 y4, 
      color_point x4 y4 = Color.Red ∧ 
      x4 = x3 + (x1 - x2) ∧ 
      y4 = y3 + (y1 - y2)) :=
by
  sorry

end coloring_satisfies_conditions_l1578_157869


namespace find_angle_C_find_max_area_l1578_157897

variable {A B C a b c : ℝ}

-- Given Conditions
def condition1 (c B a b C : ℝ) := c * Real.cos B + (b - 2 * a) * Real.cos C = 0
def condition2 (c : ℝ) := c = 2 * Real.sqrt 3

-- Problem (1): Prove the size of angle C
theorem find_angle_C (h : condition1 c B a b C) (h2 : condition2 c) : C = Real.pi / 3 := 
  sorry

-- Problem (2): Prove the maximum area of ΔABC
theorem find_max_area (h : condition1 c B a b C) (h2 : condition2 c) :
  ∃ (A B : ℝ), B = 2 * Real.pi / 3 - A ∧ 
    (∀ (A B : ℝ), Real.sin (2 * A - Real.pi / 6) = 1 → 
    1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 ∧ 
    a = b ∧ b = c) := 
  sorry

end find_angle_C_find_max_area_l1578_157897


namespace find_k4_l1578_157815

theorem find_k4
  (a_n : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a_n (n + 1) = a_n n + d)
  (h2 : d ≠ 0)
  (h3 : ∃ r : ℝ, a_n 2^2 = a_n 1 * a_n 6)
  (h4 : a_n 1 = a_n k_1)
  (h5 : a_n 2 = a_n k_2)
  (h6 : a_n 6 = a_n k_3)
  (h_k1 : k_1 = 1)
  (h_k2 : k_2 = 2)
  (h_k3 : k_3 = 6) 
  : ∃ k_4 : ℕ, k_4 = 22 := sorry

end find_k4_l1578_157815


namespace perpendicular_lines_l1578_157806

theorem perpendicular_lines (a : ℝ) : 
  (∀ (x y : ℝ), (1 - 2 * a) * x - 2 * y + 3 = 0 → 3 * x + y + 2 * a = 0) → 
  a = 1 / 6 :=
by
  sorry

end perpendicular_lines_l1578_157806


namespace max_chords_intersecting_line_l1578_157842

theorem max_chords_intersecting_line (A : Fin 2017 → Type) :
  ∃ k : ℕ, (k ≤ 2016 ∧ ∃ m : ℕ, (m = k * (2016 - k) + 2016) ∧ m = 1018080) :=
sorry

end max_chords_intersecting_line_l1578_157842


namespace count_integers_divisible_by_2_3_5_7_l1578_157807

theorem count_integers_divisible_by_2_3_5_7 :
  ∃ n : ℕ, (∀ k : ℕ, k < 500 → (k % 2 = 0 ∧ k % 3 = 0 ∧ k % 5 = 0 ∧ k % 7 = 0) → k ≠ n → k < 500 ∧ k > 0) ∧
  (n = 2) :=
by
  sorry

end count_integers_divisible_by_2_3_5_7_l1578_157807


namespace capacity_of_each_type_l1578_157885

def total_capacity_barrels : ℕ := 7000

def increased_by_first_type : ℕ := 8000

def decreased_by_second_type : ℕ := 3000

theorem capacity_of_each_type 
  (x y : ℕ) 
  (n k : ℕ)
  (h1 : x + y = total_capacity_barrels)
  (h2 : x * (n + k) / n = increased_by_first_type)
  (h3 : y * (n + k) / k = decreased_by_second_type) :
  x = 6400 ∧ y = 600 := sorry

end capacity_of_each_type_l1578_157885


namespace horner_method_complexity_l1578_157811

variable {α : Type*} [Field α]

/-- Evaluating a polynomial of degree n using Horner's method requires exactly n multiplications
    and n additions, and 0 exponentiations.  -/
theorem horner_method_complexity (n : ℕ) (a : Fin (n + 1) → α) (x₀ : α) :
  ∃ (muls adds exps : ℕ), 
    (muls = n) ∧ (adds = n) ∧ (exps = 0) :=
by
  sorry

end horner_method_complexity_l1578_157811


namespace provisions_last_days_l1578_157860

def num_soldiers_initial : ℕ := 1200
def daily_consumption_initial : ℝ := 3
def initial_duration : ℝ := 30
def extra_soldiers : ℕ := 528
def daily_consumption_new : ℝ := 2.5

noncomputable def total_provisions : ℝ := num_soldiers_initial * daily_consumption_initial * initial_duration
noncomputable def total_soldiers_after_joining : ℕ := num_soldiers_initial + extra_soldiers
noncomputable def new_daily_consumption : ℝ := total_soldiers_after_joining * daily_consumption_new

theorem provisions_last_days : (total_provisions / new_daily_consumption) = 25 := by
  sorry

end provisions_last_days_l1578_157860


namespace line_intersects_ellipse_max_chord_length_l1578_157845

theorem line_intersects_ellipse (m : ℝ) : 
  (∃ x y : ℝ, (y = (3/2 : ℝ) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1)) ↔ 
  (-3 * Real.sqrt 2 ≤ m ∧ m ≤ 3 * Real.sqrt 2) := 
by sorry

theorem max_chord_length : 
  (∃ m : ℝ, (m = 0) ∧ 
    (∀ x y x1 y1 : ℝ, (y = (3/2 : ℝ) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1) ∧ 
     (y1 = (3/2 : ℝ) * x1 + m) ∧ (x1^2 / 4 + y1^2 / 9 = 1) ∧ 
     (x ≠ x1 ∨ y ≠ y1) → 
     (Real.sqrt (13 / 9) * Real.sqrt (18 - m^2) = Real.sqrt 26))) := 
by sorry

end line_intersects_ellipse_max_chord_length_l1578_157845


namespace matt_days_alone_l1578_157894

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem matt_days_alone (M P : ℝ) (h1 : work_rate M + work_rate P = work_rate 20) 
  (h2 : 1 - 12 * (work_rate M + work_rate P) = 2 / 5) 
  (h3 : 10 * work_rate M = 2 / 5) : M = 25 :=
by
  sorry

end matt_days_alone_l1578_157894


namespace range_of_m_l1578_157864

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, 0 < x ∧ mx^2 + 2 * x + m > 0) →
  m ≤ -1 := by
  sorry

end range_of_m_l1578_157864


namespace simplified_identity_l1578_157843

theorem simplified_identity :
  (12 : ℚ) * ( (1/3 : ℚ) + (1/4) + (1/6) + (1/12) )⁻¹ = 72 / 5 :=
  sorry

end simplified_identity_l1578_157843


namespace total_flowers_sold_l1578_157888

-- Definitions for conditions
def roses_per_bouquet : ℕ := 12
def daisies_per_bouquet : ℕ := 12  -- Assuming each daisy bouquet contains the same number of daisies as roses
def total_bouquets : ℕ := 20
def rose_bouquets_sold : ℕ := 10
def daisy_bouquets_sold : ℕ := 10

-- Statement of the equivalent Lean theorem
theorem total_flowers_sold :
  (rose_bouquets_sold * roses_per_bouquet) + (daisy_bouquets_sold * daisies_per_bouquet) = 240 :=
by
  sorry

end total_flowers_sold_l1578_157888


namespace new_average_is_ten_l1578_157899

-- Define the initial conditions
def initial_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) : Prop :=
  x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈ + x₉ = 9 * 7

-- Define the transformation on the nine numbers
def transformed_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) : ℝ :=
  (x₁ - 3) + (x₂ - 3) + (x₃ - 3) +
  (x₄ + 5) + (x₅ + 5) + (x₆ + 5) +
  (2 * x₇) + (2 * x₈) + (2 * x₉)

-- The theorem to prove the new average is 10
theorem new_average_is_ten (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) 
  (h : initial_sum x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉) :
  transformed_sum x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ / 9 = 10 :=
by 
  sorry

end new_average_is_ten_l1578_157899


namespace bert_kangaroos_equal_to_kameron_in_40_days_l1578_157878

theorem bert_kangaroos_equal_to_kameron_in_40_days
  (k_count : ℕ) (b_count : ℕ) (rate : ℕ) (days : ℕ)
  (h1 : k_count = 100)
  (h2 : b_count = 20)
  (h3 : rate = 2)
  (h4 : days = 40) :
  b_count + days * rate = k_count := 
by
  sorry

end bert_kangaroos_equal_to_kameron_in_40_days_l1578_157878


namespace fuel_oil_used_l1578_157853

theorem fuel_oil_used (V_initial : ℕ) (V_jan : ℕ) (V_may : ℕ) : 
  (V_initial - V_jan) + (V_initial - V_may) = 4582 :=
by
  let V_initial := 3000
  let V_jan := 180
  let V_may := 1238
  sorry

end fuel_oil_used_l1578_157853


namespace sum_S_15_22_31_l1578_157826

-- Define the sequence \{a_n\} with the sum of the first n terms S_n
def S : ℕ → ℤ
| 0 => 0
| n + 1 => S n + (-1: ℤ)^n * (4 * (n + 1) - 3)

-- The statement to prove: S_{15} + S_{22} - S_{31} = -76
theorem sum_S_15_22_31 : S 15 + S 22 - S 31 = -76 :=
sorry

end sum_S_15_22_31_l1578_157826


namespace eleven_billion_in_scientific_notation_l1578_157838

namespace ScientificNotation

def Yi : ℝ := 10 ^ 8

theorem eleven_billion_in_scientific_notation : (11 * (10 : ℝ) ^ 9) = (1.1 * (10 : ℝ) ^ 10) :=
by 
  sorry

end ScientificNotation

end eleven_billion_in_scientific_notation_l1578_157838


namespace express_y_in_terms_of_x_l1578_157825

variable (x y p : ℝ)

-- Conditions
def condition1 := x = 1 + 3^p
def condition2 := y = 1 + 3^(-p)

-- The theorem to be proven
theorem express_y_in_terms_of_x (h1 : condition1 x p) (h2 : condition2 y p) : y = x / (x - 1) :=
sorry

end express_y_in_terms_of_x_l1578_157825


namespace exists_x_gg_eq_3_l1578_157821

noncomputable def g (x : ℝ) : ℝ :=
if x < -3 then -0.5 * x^2 + 3
else if x < 2 then 1
else 0.5 * x^2 - 1.5 * x + 3

theorem exists_x_gg_eq_3 : ∃ x : ℝ, x = -5 ∨ x = 5 ∧ g (g x) = 3 :=
by
  sorry

end exists_x_gg_eq_3_l1578_157821


namespace bella_total_roses_l1578_157818

-- Define the constants and conditions
def dozen := 12
def roses_from_parents := 2 * dozen
def friends := 10
def roses_per_friend := 2
def total_roses := roses_from_parents + (roses_per_friend * friends)

-- Prove that the total number of roses Bella received is 44
theorem bella_total_roses : total_roses = 44 := 
by
  sorry

end bella_total_roses_l1578_157818


namespace ordered_triples_54000_l1578_157876

theorem ordered_triples_54000 : 
  ∃ (count : ℕ), 
  count = 16 ∧ 
  ∀ (a b c : ℕ), 
  0 < a → 0 < b → 0 < c → a^4 * b^2 * c = 54000 → 
  count = 16 := 
sorry

end ordered_triples_54000_l1578_157876


namespace identity_eq_a_minus_b_l1578_157877

theorem identity_eq_a_minus_b (a b : ℚ) (x : ℚ) (h : ∀ x, x > 0 → 
  (a / (2^x - 2) + b / (2^x + 3) = (5 * 2^x + 4) / ((2^x - 2) * (2^x + 3)))) : 
  a - b = 3 / 5 := 
by 
  sorry

end identity_eq_a_minus_b_l1578_157877


namespace probability_at_least_one_card_each_cousin_correct_l1578_157879

noncomputable def probability_at_least_one_card_each_cousin : ℚ :=
  let total_cards := 16
  let cards_per_cousin := 8
  let selections := 3
  let total_ways := Nat.choose total_cards selections
  let ways_all_from_one_cousin := Nat.choose cards_per_cousin selections * 2  -- twice: once for each cousin
  let prob_all_from_one_cousin := (ways_all_from_one_cousin : ℚ) / total_ways
  1 - prob_all_from_one_cousin

theorem probability_at_least_one_card_each_cousin_correct :
  probability_at_least_one_card_each_cousin = 4 / 5 :=
by
  -- Proof would go here
  sorry

end probability_at_least_one_card_each_cousin_correct_l1578_157879


namespace most_suitable_for_comprehensive_survey_l1578_157828

-- Definitions of the survey options
inductive SurveyOption
| A
| B
| C
| D

-- Condition definitions based on the problem statement
def comprehensive_survey (option : SurveyOption) : Prop :=
  option = SurveyOption.B

-- The theorem stating that the most suitable survey is option B
theorem most_suitable_for_comprehensive_survey : ∀ (option : SurveyOption), comprehensive_survey option ↔ option = SurveyOption.B :=
by
  intro option
  sorry

end most_suitable_for_comprehensive_survey_l1578_157828


namespace divide_by_10_result_l1578_157887

theorem divide_by_10_result (x : ℕ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end divide_by_10_result_l1578_157887


namespace plane_distance_l1578_157880

theorem plane_distance (n : ℕ) : n % 45 = 0 ∧ (n / 10) % 100 = 39 ∧ n <= 5000 → n = 1395 := 
by
  sorry

end plane_distance_l1578_157880


namespace sally_bread_consumption_l1578_157837

/-
Given:
    1) Sally eats 2 sandwiches on Saturday
    2) Sally eats 1 sandwich on Sunday
    3) Each sandwich uses 2 pieces of bread

Prove:
    Sally eats 6 pieces of bread across Saturday and Sunday
-/
theorem sally_bread_consumption (sandwiches_sat : Nat) (sandwiches_sun : Nat) (bread_per_sandwich : Nat)
    (H1 : sandwiches_sat = 2) (H2 : sandwiches_sun = 1) (H3 : bread_per_sandwich = 2) :
    2 * bread_per_sandwich + 1 * bread_per_sandwich = 6 := by
  sorry

end sally_bread_consumption_l1578_157837


namespace find_sum_12_terms_of_sequence_l1578_157836

variable {a : ℕ → ℕ}

def geometric_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

def is_periodic_sequence (a : ℕ → ℕ) (period : ℕ) : Prop :=
  ∀ n : ℕ, a n = a (n + period)

noncomputable def given_sequence : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => (given_sequence n * given_sequence (n + 1) / 4) -- This should ensure periodic sequence of period 3 given a common product of 8 and simplifying the product equation.

theorem find_sum_12_terms_of_sequence :
  geometric_sequence given_sequence 8 ∧ given_sequence 0 = 1 ∧ given_sequence 1 = 2 →
  (Finset.range 12).sum given_sequence = 28 :=
by
  sorry

end find_sum_12_terms_of_sequence_l1578_157836


namespace price_reduction_equation_l1578_157848

theorem price_reduction_equation (x : ℝ) : 25 * (1 - x)^2 = 16 :=
by
  sorry

end price_reduction_equation_l1578_157848


namespace total_workers_is_28_l1578_157840

noncomputable def avg_salary_total : ℝ := 750
noncomputable def num_type_a : ℕ := 5
noncomputable def avg_salary_type_a : ℝ := 900
noncomputable def num_type_b : ℕ := 4
noncomputable def avg_salary_type_b : ℝ := 800
noncomputable def avg_salary_type_c : ℝ := 700

theorem total_workers_is_28 :
  ∃ (W : ℕ) (C : ℕ),
  W * avg_salary_total = num_type_a * avg_salary_type_a + num_type_b * avg_salary_type_b + C * avg_salary_type_c ∧
  W = num_type_a + num_type_b + C ∧
  W = 28 :=
by
  sorry

end total_workers_is_28_l1578_157840


namespace condition_two_eqn_l1578_157883

def line_through_point_and_perpendicular (x1 y1 : ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, (y - y1) = -1/(x - x1) * (x - x1 + c) → x - y + c = 0

theorem condition_two_eqn :
  line_through_point_and_perpendicular 1 (-2) (-3) :=
sorry

end condition_two_eqn_l1578_157883


namespace scientific_notation_4947_66_billion_l1578_157856

theorem scientific_notation_4947_66_billion :
  4947.66 * 10^8 = 4.94766 * 10^11 :=
sorry

end scientific_notation_4947_66_billion_l1578_157856


namespace estimate_production_in_March_l1578_157884

theorem estimate_production_in_March 
  (monthly_production : ℕ → ℝ)
  (x y : ℝ)
  (hx : x = 3)
  (hy : y = x + 1) : y = 4 :=
by
  sorry

end estimate_production_in_March_l1578_157884


namespace merchant_profit_percentage_is_35_l1578_157868

noncomputable def cost_price : ℝ := 100
noncomputable def markup_percentage : ℝ := 0.80
noncomputable def discount_percentage : ℝ := 0.25

-- Marked price after 80% markup
noncomputable def marked_price (cp : ℝ) (markup_pct : ℝ) : ℝ :=
  cp + (markup_pct * cp)

-- Selling price after 25% discount on marked price
noncomputable def selling_price (mp : ℝ) (discount_pct : ℝ) : ℝ :=
  mp - (discount_pct * mp)

-- Profit as the difference between selling price and cost price
noncomputable def profit (sp cp : ℝ) : ℝ :=
  sp - cp

-- Profit percentage
noncomputable def profit_percentage (profit cp : ℝ) : ℝ :=
  (profit / cp) * 100

theorem merchant_profit_percentage_is_35 :
  let cp := cost_price
  let mp := marked_price cp markup_percentage
  let sp := selling_price mp discount_percentage
  let prof := profit sp cp
  profit_percentage prof cp = 35 :=
by
  let cp := cost_price
  let mp := marked_price cp markup_percentage
  let sp := selling_price mp discount_percentage
  let prof := profit sp cp
  show profit_percentage prof cp = 35
  sorry

end merchant_profit_percentage_is_35_l1578_157868


namespace compare_negative_fractions_l1578_157823

theorem compare_negative_fractions :
  (-5 : ℝ) / 6 < (-4 : ℝ) / 5 :=
sorry

end compare_negative_fractions_l1578_157823


namespace graph_paper_problem_l1578_157834

theorem graph_paper_problem :
  let line_eq := ∀ x y : ℝ, 7 * x + 268 * y = 1876
  ∃ (n : ℕ), 
  (∀ x y : ℕ, 0 < x ∧ x ≤ 268 ∧ 0 < y ∧ y ≤ 7 ∧ (7 * (x:ℝ) + 268 * (y:ℝ)) < 1876) →
  n = 801 :=
by
  sorry

end graph_paper_problem_l1578_157834


namespace greatest_k_for_200k_divides_100_factorial_l1578_157820

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_k_for_200k_divides_100_factorial :
  let x := factorial 100
  let k_max := 12
  ∃ k : ℕ, y = 200 ^ k ∧ y ∣ x ∧ k = k_max :=
sorry

end greatest_k_for_200k_divides_100_factorial_l1578_157820


namespace min_distance_between_parallel_lines_l1578_157803

theorem min_distance_between_parallel_lines
  (m c_1 c_2 : ℝ)
  (h_parallel : ∀ x : ℝ, m * x + c_1 = m * x + c_2 → false) :
  ∃ D : ℝ, D = (|c_2 - c_1|) / (Real.sqrt (1 + m^2)) :=
by
  sorry

end min_distance_between_parallel_lines_l1578_157803


namespace translation_equivalence_l1578_157839

def f₁ (x : ℝ) : ℝ := 4 * (x + 3)^2 - 4
def f₂ (x : ℝ) : ℝ := 4 * (x - 3)^2 + 4

theorem translation_equivalence :
  (∀ x : ℝ, f₁ (x + 6) = 4 * (x + 9)^2 + 4) ∧
  (∀ x : ℝ, f₁ x  - 8 = 4 * (x + 3)^2 - 4) :=
by sorry

end translation_equivalence_l1578_157839


namespace tank_capacity_l1578_157851

theorem tank_capacity
  (x : ℝ) -- define x as the full capacity of the tank in gallons
  (h1 : (5/6) * x - (2/3) * x = 15) -- first condition
  (h2 : (2/3) * x = y) -- second condition, though not actually needed
  : x = 90 := 
by sorry

end tank_capacity_l1578_157851


namespace product_of_two_numbers_ratio_l1578_157816

theorem product_of_two_numbers_ratio {x y : ℝ}
  (h1 : x + y = (5/3) * (x - y))
  (h2 : x * y = 5 * (x - y)) :
  x * y = 56.25 := sorry

end product_of_two_numbers_ratio_l1578_157816


namespace find_a_find_m_l1578_157873

noncomputable def f (x a : ℝ) : ℝ := Real.exp 1 * x - a * Real.log x

theorem find_a {a : ℝ} (h : ∀ x, f x a = Real.exp 1 - a / x)
  (hx : f (1 / Real.exp 1) a = 0) :
  a = 1 :=
by
  sorry

theorem find_m (a : ℝ) (h_a : a = 1)
  (h_exists : ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) 
    ∧ f x₀ a < x₀ + m) :
  1 + Real.log (Real.exp 1 - 1) < m :=
by
  sorry

end find_a_find_m_l1578_157873


namespace trajectory_of_midpoint_l1578_157881

noncomputable section

open Real

-- Define the points and lines
def C : ℝ × ℝ := (-2, -2)
def A (x : ℝ) : ℝ × ℝ := (x, 0)
def B (y : ℝ) : ℝ × ℝ := (0, y)
def M (x y : ℝ) : ℝ × ℝ := ((x + 0) / 2, (0 + y) / 2)

theorem trajectory_of_midpoint (CA_dot_CB : (C.1 * (A 0).1 + (C.2 - (A 0).2)) * (C.1 * (B 0).1 + (C.2 - (B 0).2)) = 0) :
  ∀ (M : ℝ × ℝ), (M.1 = (A 0).1 / 2) ∧ (M.2 = (B 0).2 / 2) → (M.1 + M.2 + 2 = 0) :=
by
  -- here's where the proof would go
  sorry

end trajectory_of_midpoint_l1578_157881


namespace circle_area_of_white_cube_l1578_157882

/-- 
Marla has a large white cube with an edge length of 12 feet and enough green paint to cover 432 square feet.
Marla paints a white circle centered on each face of the cube, surrounded by a green border.
Prove the area of one of the white circles is 72 square feet.
 -/
theorem circle_area_of_white_cube
  (edge_length : ℝ) (paint_area : ℝ) (faces : ℕ)
  (h_edge_length : edge_length = 12)
  (h_paint_area : paint_area = 432)
  (h_faces : faces = 6) :
  ∃ (circle_area : ℝ), circle_area = 72 :=
by
  sorry

end circle_area_of_white_cube_l1578_157882


namespace find_f_pi_over_4_l1578_157800

variable (f : ℝ → ℝ)
variable (h : ∀ x, f x = f (Real.pi / 4) * Real.cos x + Real.sin x)

theorem find_f_pi_over_4 : f (Real.pi / 4) = 1 := by
  sorry

end find_f_pi_over_4_l1578_157800


namespace max_volume_of_acetic_acid_solution_l1578_157874

theorem max_volume_of_acetic_acid_solution :
  (∀ (V : ℝ), 0 ≤ V ∧ (V * 0.09) = (25 * 0.7 + (V - 25) * 0.05)) →
  V = 406.25 :=
by
  sorry

end max_volume_of_acetic_acid_solution_l1578_157874


namespace no_quadruples_sum_2013_l1578_157850

theorem no_quadruples_sum_2013 :
  ¬ ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + b + c + d = 2013 ∧
  2013 % a = 0 ∧ 2013 % b = 0 ∧ 2013 % c = 0 ∧ 2013 % d = 0 :=
by
  sorry

end no_quadruples_sum_2013_l1578_157850


namespace total_children_in_circle_l1578_157891

theorem total_children_in_circle 
  (n : ℕ)  -- number of children
  (h_even : Even n)   -- condition: the circle is made up of an even number of children
  (h_pos : n > 0) -- condition: there are some children
  (h_opposite : (15 % n + 15 % n) % n = 0)  -- condition: the 15th child clockwise from Child A is facing Child A (implies opposite)
  : n = 30 := 
sorry

end total_children_in_circle_l1578_157891


namespace sin_half_angle_l1578_157817

theorem sin_half_angle
  (theta : ℝ)
  (h1 : Real.sin theta = 3 / 5)
  (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) :
  Real.sin (theta / 2) = - (3 * Real.sqrt 10 / 10) :=
by
  sorry

end sin_half_angle_l1578_157817


namespace greatest_int_less_than_neg_19_div_3_l1578_157822

theorem greatest_int_less_than_neg_19_div_3 : ∃ n : ℤ, n = -7 ∧ n < (-19 / 3 : ℚ) ∧ (-19 / 3 : ℚ) < n + 1 := 
by
  sorry

end greatest_int_less_than_neg_19_div_3_l1578_157822


namespace profit_percentage_example_l1578_157875

noncomputable def selling_price : ℝ := 100
noncomputable def cost_price (sp : ℝ) : ℝ := 0.75 * sp
noncomputable def profit (sp cp : ℝ) : ℝ := sp - cp
noncomputable def profit_percentage (profit cp : ℝ) : ℝ := (profit / cp) * 100

theorem profit_percentage_example :
  profit_percentage (profit selling_price (cost_price selling_price)) (cost_price selling_price) = 33.33 :=
by
  -- Proof will go here
  sorry

end profit_percentage_example_l1578_157875


namespace inequality_always_holds_l1578_157855

variable {a b : ℝ}

theorem inequality_always_holds (ha : a > 0) (hb : b < 0) : 1 / a > 1 / b :=
by
  sorry

end inequality_always_holds_l1578_157855


namespace condition_sufficient_not_necessary_l1578_157872

theorem condition_sufficient_not_necessary (x : ℝ) : (0 < x ∧ x < 5) → (|x - 2| < 3) ∧ (¬ ((|x - 2| < 3) → (0 < x ∧ x < 5))) :=
by
  sorry

end condition_sufficient_not_necessary_l1578_157872


namespace number_of_initials_is_10000_l1578_157813

-- Define the set of letters A through J as a finite set
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

open Letter

-- Define a function to count the number of different four-letter sets of initials
def count_initials : Nat :=
  10 ^ 4

-- The theorem to prove: the number of different four-letter sets of initials is 10000
theorem number_of_initials_is_10000 : count_initials = 10000 := by
  sorry

end number_of_initials_is_10000_l1578_157813


namespace reach_any_natural_number_l1578_157831

theorem reach_any_natural_number (n : ℕ) : ∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = 3 * f k + 1 ∨ f (k + 1) = f k / 2) ∧ (∃ m, f m = n) := by
  sorry

end reach_any_natural_number_l1578_157831


namespace only_zero_function_satisfies_inequality_l1578_157819

noncomputable def f (x : ℝ) : ℝ := sorry

theorem only_zero_function_satisfies_inequality (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  (∀ x y : ℝ, 0 < x → 0 < y →
    f x * f y ≥ (y^α / (x^α + x^β)) * (f x)^2 + (x^β / (y^α + y^β)) * (f y)^2) →
  ∀ x : ℝ, 0 < x → f x = 0 :=
sorry

end only_zero_function_satisfies_inequality_l1578_157819


namespace correct_card_ordering_l1578_157808

structure CardOrder where
  left : String
  middle : String
  right : String

def is_right_of (a b : String) : Prop := (a = "club" ∧ (b = "heart" ∨ b = "diamond")) ∨ (a = "8" ∧ b = "4")

def is_left_of (a b : String) : Prop := a = "5" ∧ b = "heart"

def correct_order : CardOrder :=
  { left := "5 of diamonds", middle := "4 of hearts", right := "8 of clubs" }

theorem correct_card_ordering : 
  ∀ order : CardOrder, 
  is_right_of order.right order.middle ∧ is_right_of order.right order.left ∧ is_left_of order.left order.middle 
  → order = correct_order := 
by
  intro order
  intro h
  sorry

end correct_card_ordering_l1578_157808


namespace ratio_of_good_states_l1578_157889

theorem ratio_of_good_states (n : ℕ) :
  let total_states := 2^(2*n)
  let good_states := Nat.choose (2 * n) n
  good_states / total_states = (List.range n).foldr (fun i acc => acc * (2*i+1)) 1 / (2^n * Nat.factorial n) := sorry

end ratio_of_good_states_l1578_157889


namespace ellipse_standard_equation_parabola_standard_equation_l1578_157849

-- Ellipse with major axis length 10 and eccentricity 4/5
theorem ellipse_standard_equation (a c b : ℝ) (h₀ : a = 5) (h₁ : c = 4) (h₂ : b = 3) :
  (x^2 / a^2) + (y^2 / b^2) = 1 := by sorry

-- Parabola with vertex at the origin and directrix y = 2
theorem parabola_standard_equation (p : ℝ) (h₀ : p = 4) :
  x^2 = -8 * y := by sorry

end ellipse_standard_equation_parabola_standard_equation_l1578_157849


namespace bren_age_indeterminate_l1578_157892

/-- The problem statement: The ratio of ages of Aman, Bren, and Charlie are in 
the ratio 5:8:7 respectively. A certain number of years ago, the sum of their ages was 76. 
We need to prove that without additional information, it is impossible to uniquely 
determine Bren's age 10 years from now. -/
theorem bren_age_indeterminate
  (x y : ℕ) 
  (h_ratio : true)
  (h_sum : 20 * x - 3 * y = 76) : 
  ∃ x y : ℕ, (20 * x - 3 * y = 76) ∧ ∀ bren_age_future : ℕ, ∃ x' y' : ℕ, (20 * x' - 3 * y' = 76) ∧ (8 * x' + 10) ≠ bren_age_future :=
sorry

end bren_age_indeterminate_l1578_157892


namespace group_size_of_bananas_l1578_157867

theorem group_size_of_bananas (totalBananas numberOfGroups : ℕ) (h1 : totalBananas = 203) (h2 : numberOfGroups = 7) :
  totalBananas / numberOfGroups = 29 :=
sorry

end group_size_of_bananas_l1578_157867


namespace max_value_ln_x_plus_x_l1578_157844

theorem max_value_ln_x_plus_x (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ Real.exp 1) : 
  ∃ y, y = Real.log x + x ∧ y ≤ Real.log (Real.exp 1) + Real.exp 1 :=
sorry

end max_value_ln_x_plus_x_l1578_157844


namespace simplify_polynomial_l1578_157801

theorem simplify_polynomial (x : ℝ) : 
  (3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2) = (-x^2 + 23 * x - 3) := 
by
  sorry

end simplify_polynomial_l1578_157801


namespace ratio_of_number_to_ten_l1578_157859

theorem ratio_of_number_to_ten (n : ℕ) (h : n = 200) : n / 10 = 20 :=
by
  sorry

end ratio_of_number_to_ten_l1578_157859


namespace registration_methods_for_5_students_l1578_157898

def number_of_registration_methods (students groups : ℕ) : ℕ :=
  groups ^ students

theorem registration_methods_for_5_students : number_of_registration_methods 5 2 = 32 := by
  sorry

end registration_methods_for_5_students_l1578_157898


namespace percentage_y_less_than_x_l1578_157858

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 4 * y) : (x - y) / x * 100 = 75 := by
  sorry

end percentage_y_less_than_x_l1578_157858


namespace rectangle_area_l1578_157896

noncomputable def circle_radius := 8
noncomputable def rect_ratio : ℕ × ℕ := (3, 1)
noncomputable def rect_area (width length : ℕ) : ℕ := width * length

theorem rectangle_area (width length : ℕ) 
  (h1 : 2 * circle_radius = width) 
  (h2 : rect_ratio.1 * width = length) : 
  rect_area width length = 768 := 
sorry

end rectangle_area_l1578_157896


namespace employed_population_percentage_l1578_157862

theorem employed_population_percentage
  (P : ℝ) -- Total population
  (E : ℝ) -- Fraction of population that is employed
  (employed_males : ℝ) -- Fraction of population that is employed males
  (employed_females_fraction : ℝ)
  (h1 : employed_males = 0.8 * P)
  (h2 : employed_females_fraction = 1 / 3) :
  E = 0.6 :=
by
  -- We don't need the proof here.
  sorry

end employed_population_percentage_l1578_157862


namespace invisible_dots_48_l1578_157871

theorem invisible_dots_48 (visible : Multiset ℕ) (hv : visible = [1, 2, 3, 3, 4, 5, 6, 6, 6]) :
  let total_dots := 4 * (1 + 2 + 3 + 4 + 5 + 6)
  let visible_sum := visible.sum
  total_dots - visible_sum = 48 :=
by
  sorry

end invisible_dots_48_l1578_157871


namespace min_teachers_required_l1578_157852

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

end min_teachers_required_l1578_157852


namespace factorization_identity_sum_l1578_157863

theorem factorization_identity_sum (a b c : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 15 * x + 36 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) :
  a + b + c = 20 :=
sorry

end factorization_identity_sum_l1578_157863


namespace Jason_total_money_l1578_157812

theorem Jason_total_money :
  let quarter_value := 0.25
  let dime_value := 0.10
  let nickel_value := 0.05
  let initial_quarters := 49
  let initial_dimes := 32
  let initial_nickels := 18
  let additional_quarters := 25
  let additional_dimes := 15
  let additional_nickels := 10
  let initial_money := initial_quarters * quarter_value + initial_dimes * dime_value + initial_nickels * nickel_value
  let additional_money := additional_quarters * quarter_value + additional_dimes * dime_value + additional_nickels * nickel_value
  initial_money + additional_money = 24.60 :=
by
  sorry

end Jason_total_money_l1578_157812


namespace store_profit_in_february_l1578_157835

variable (C : ℝ)

def initialSellingPrice := C * 1.20
def secondSellingPrice := initialSellingPrice C * 1.25
def finalSellingPrice := secondSellingPrice C * 0.88

theorem store_profit_in_february
  (initialSellingPrice_eq : initialSellingPrice C = C * 1.20)
  (secondSellingPrice_eq : secondSellingPrice C = initialSellingPrice C * 1.25)
  (finalSellingPrice_eq : finalSellingPrice C = secondSellingPrice C * 0.88)
  : finalSellingPrice C - C = 0.32 * C :=
sorry

end store_profit_in_february_l1578_157835


namespace total_coins_l1578_157870

-- Define the number of stacks and the number of coins per stack
def stacks : ℕ := 5
def coins_per_stack : ℕ := 3

-- State the theorem to prove the total number of coins
theorem total_coins (s c : ℕ) (hs : s = stacks) (hc : c = coins_per_stack) : s * c = 15 :=
by
  -- Proof is omitted
  sorry

end total_coins_l1578_157870


namespace find_Q_l1578_157829

theorem find_Q (m n Q p : ℝ) (h1 : m = 6 * n + 5)
    (h2 : p = 0.3333333333333333)
    (h3 : m + Q = 6 * (n + p) + 5) : Q = 2 := 
by
  sorry

end find_Q_l1578_157829


namespace seq_sum_terms_l1578_157824

def S (n : ℕ) : ℕ := 3^n - 2

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2 * 3^(n - 1)

theorem seq_sum_terms (n : ℕ) : 
  a n = if n = 1 then 1 else 2 * 3^(n-1) :=
sorry

end seq_sum_terms_l1578_157824


namespace least_multiple_of_13_gt_450_l1578_157833

theorem least_multiple_of_13_gt_450 : ∃ (n : ℕ), (455 = 13 * n) ∧ 455 > 450 ∧ ∀ m : ℕ, (13 * m > 450) → 455 ≤ 13 * m :=
by
  sorry

end least_multiple_of_13_gt_450_l1578_157833


namespace solve_f_lt_zero_solve_f_plus_4_abs_x_minus_4_gt_m_l1578_157866

def f (x : ℝ) : ℝ := |3 * x + 1| - |x - 4|

theorem solve_f_lt_zero :
  { x : ℝ | f x < 0 } = { x : ℝ | -5 / 2 < x ∧ x < 3 / 4 } := 
sorry

theorem solve_f_plus_4_abs_x_minus_4_gt_m (m : ℝ) :
  (∀ x : ℝ, f x + 4 * |x - 4| > m) → m < 15 :=
sorry

end solve_f_lt_zero_solve_f_plus_4_abs_x_minus_4_gt_m_l1578_157866


namespace haleys_current_height_l1578_157832

-- Define the conditions
def growth_rate : ℕ := 3
def years : ℕ := 10
def future_height : ℕ := 50

-- Define the proof problem
theorem haleys_current_height : (future_height - growth_rate * years) = 20 :=
by {
  -- This is where the actual proof would go
  sorry
}

end haleys_current_height_l1578_157832


namespace find_number_l1578_157846

variable (number : ℤ)

theorem find_number (h : number - 44 = 15) : number = 59 := 
sorry

end find_number_l1578_157846


namespace average_price_per_book_l1578_157810

theorem average_price_per_book
  (amount_spent_first_shop : ℕ)
  (amount_spent_second_shop : ℕ)
  (books_first_shop : ℕ)
  (books_second_shop : ℕ)
  (total_amount_spent : ℕ := amount_spent_first_shop + amount_spent_second_shop)
  (total_books_bought : ℕ := books_first_shop + books_second_shop)
  (average_price : ℕ := total_amount_spent / total_books_bought) :
  amount_spent_first_shop = 520 → amount_spent_second_shop = 248 →
  books_first_shop = 42 → books_second_shop = 22 →
  average_price = 12 :=
by
  intros
  sorry

end average_price_per_book_l1578_157810


namespace prob_at_least_one_palindrome_correct_l1578_157827

-- Define a function to represent the probability calculation.
def probability_at_least_one_palindrome : ℚ :=
  let prob_digit_palindrome : ℚ := 1 / 100
  let prob_letter_palindrome : ℚ := 1 / 676
  let prob_both_palindromes : ℚ := (1 / 100) * (1 / 676)
  (prob_digit_palindrome + prob_letter_palindrome - prob_both_palindromes)

-- The theorem we are stating based on the given problem and solution:
theorem prob_at_least_one_palindrome_correct : probability_at_least_one_palindrome = 427 / 2704 :=
by
  -- We assume this step for now as we are just stating the theorem
  sorry

end prob_at_least_one_palindrome_correct_l1578_157827


namespace problem_statement_l1578_157814

variable (g : ℝ)

-- Definition of the operation
def my_op (g y : ℝ) : ℝ := g^2 + 2 * y

-- The statement we want to prove
theorem problem_statement : my_op g (my_op g g) = g^4 + 4 * g^3 + 6 * g^2 + 4 * g :=
by
  sorry

end problem_statement_l1578_157814
