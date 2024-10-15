import Mathlib

namespace NUMINAMATH_GPT_imaginary_unit_root_l1468_146883

theorem imaginary_unit_root (a b : ℝ) (h : (Complex.I : ℂ) ^ 2 + a * Complex.I + b = 0) : a + b = 1 := by
  -- Since this is just the statement, we add a sorry to focus on the structure
  sorry

end NUMINAMATH_GPT_imaginary_unit_root_l1468_146883


namespace NUMINAMATH_GPT_exists_k_seq_zero_to_one_l1468_146891

noncomputable def seq (a : ℕ → ℝ) (h : ∀ n, a (n + 2) = |a (n + 1) - a n|) := a

theorem exists_k_seq_zero_to_one (a : ℕ → ℝ) (h : ∀ n, a (n + 2) = |a (n + 1) - a n|) :
  ∃ k : ℕ, 0 ≤ a k ∧ a k < 1 :=
sorry

end NUMINAMATH_GPT_exists_k_seq_zero_to_one_l1468_146891


namespace NUMINAMATH_GPT_famous_sentences_correct_l1468_146887

def blank_1 : String := "correct_answer_1"
def blank_2 : String := "correct_answer_2"
def blank_3 : String := "correct_answer_3"
def blank_4 : String := "correct_answer_4"
def blank_5 : String := "correct_answer_5"
def blank_6 : String := "correct_answer_6"
def blank_7 : String := "correct_answer_7"
def blank_8 : String := "correct_answer_8"

theorem famous_sentences_correct :
  blank_1 = "correct_answer_1" ∧
  blank_2 = "correct_answer_2" ∧
  blank_3 = "correct_answer_3" ∧
  blank_4 = "correct_answer_4" ∧
  blank_5 = "correct_answer_5" ∧
  blank_6 = "correct_answer_6" ∧
  blank_7 = "correct_answer_7" ∧
  blank_8 = "correct_answer_8" :=
by
  -- The proof details correspond to the part "refer to the correct solution for each blank"
  sorry

end NUMINAMATH_GPT_famous_sentences_correct_l1468_146887


namespace NUMINAMATH_GPT_project_total_hours_l1468_146853

def pat_time (k : ℕ) : ℕ := 2 * k
def mark_time (k : ℕ) : ℕ := k + 120

theorem project_total_hours (k : ℕ) (H1 : 3 * 2 * k = k + 120) :
  k + pat_time k + mark_time k = 216 :=
by
  sorry

end NUMINAMATH_GPT_project_total_hours_l1468_146853


namespace NUMINAMATH_GPT_find_value_of_fraction_of_x_six_l1468_146893

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := (Real.log x) / (Real.log b)

theorem find_value_of_fraction_of_x_six (x : ℝ) (h : log_base (10 * x) 10 + log_base (100 * x ^ 2) 10 = -1) : 
    1 / x ^ 6 = 31622.7766 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_fraction_of_x_six_l1468_146893


namespace NUMINAMATH_GPT_coordinate_identification_l1468_146882

noncomputable def x1 := (4 * Real.pi) / 5
noncomputable def y1 := -(Real.pi) / 5

noncomputable def x2 := (12 * Real.pi) / 5
noncomputable def y2 := -(3 * Real.pi) / 5

noncomputable def x3 := (4 * Real.pi) / 3
noncomputable def y3 := -(Real.pi) / 3

theorem coordinate_identification :
  (x1, y1) = (4 * Real.pi / 5, -(Real.pi) / 5) ∧
  (x2, y2) = (12 * Real.pi / 5, -(3 * Real.pi) / 5) ∧
  (x3, y3) = (4 * Real.pi / 3, -(Real.pi) / 3) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_coordinate_identification_l1468_146882


namespace NUMINAMATH_GPT_evaluate_expression_l1468_146884

theorem evaluate_expression (x : ℤ) (h1 : 0 ≤ x ∧ x ≤ 2) (h2 : x ≠ 1) (h3 : x ≠ 2) (h4 : x = 0) :
    ( ((4 - x) / (x - 1) - x) / ((x - 2) / (x - 1)) ) = -2 :=
by
    sorry

end NUMINAMATH_GPT_evaluate_expression_l1468_146884


namespace NUMINAMATH_GPT_ratio_third_second_l1468_146831

theorem ratio_third_second (k : ℝ) (x y z : ℝ) (h1 : y = 4 * x) (h2 : x = 18) (h3 : z = k * y) (h4 : (x + y + z) / 3 = 78) :
  z = 2 * y :=
by
  sorry

end NUMINAMATH_GPT_ratio_third_second_l1468_146831


namespace NUMINAMATH_GPT_exists_indices_l1468_146859

open Nat List

theorem exists_indices (m n : ℕ) (a : Fin m → ℕ) (b : Fin n → ℕ) 
  (h1 : ∀ i : Fin m, a i ≤ n) (h2 : ∀ i j : Fin m, i ≤ j → a i ≤ a j)
  (h3 : ∀ j : Fin n, b j ≤ m) (h4 : ∀ i j : Fin n, i ≤ j → b i ≤ b j) :
  ∃ i : Fin m, ∃ j : Fin n, a i + i.val + 1 = b j + j.val + 1 := by
  sorry

end NUMINAMATH_GPT_exists_indices_l1468_146859


namespace NUMINAMATH_GPT_selling_price_41_l1468_146875

-- Purchase price per item
def purchase_price : ℝ := 30

-- Government restriction on pice increase: selling price cannot be more than 40% increase of the purchase price
def price_increase_restriction (a : ℝ) : Prop :=
  a <= purchase_price * 1.4

-- Profit condition equation
def profit_condition (a : ℝ) : Prop :=
  (a - purchase_price) * (112 - 2 * a) = 330

-- The selling price of each item that satisfies all conditions is 41 yuan  
theorem selling_price_41 (a : ℝ) (h1 : profit_condition a) (h2 : price_increase_restriction a) :
  a = 41 := sorry

end NUMINAMATH_GPT_selling_price_41_l1468_146875


namespace NUMINAMATH_GPT_value_of_x_l1468_146897

theorem value_of_x (x : ℝ) :
  (x^2 - 1 + (x - 1) * I = 0 ∨ x^2 - 1 = 0 ∧ x - 1 ≠ 0) → x = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1468_146897


namespace NUMINAMATH_GPT_simple_interest_rate_problem_l1468_146827

noncomputable def simple_interest_rate (P : ℝ) (T : ℝ) (final_amount : ℝ) : ℝ :=
  (final_amount - P) * 100 / (P * T)

theorem simple_interest_rate_problem
  (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : T = 2)
  (h2 : final_amount = (7 / 6) * P)
  (h3 : simple_interest_rate P T final_amount = R) : 
  R = 100 / 12 := sorry

end NUMINAMATH_GPT_simple_interest_rate_problem_l1468_146827


namespace NUMINAMATH_GPT_kate_average_speed_correct_l1468_146860

noncomputable def kate_average_speed : ℝ :=
  let biking_time_hours := 20 / 60
  let walking_time_hours := 60 / 60
  let jogging_time_hours := 40 / 60
  let biking_distance := 20 * biking_time_hours
  let walking_distance := 4 * walking_time_hours
  let jogging_distance := 6 * jogging_time_hours
  let total_distance := biking_distance + walking_distance + jogging_distance
  let total_time_hours := biking_time_hours + walking_time_hours + jogging_time_hours
  total_distance / total_time_hours

theorem kate_average_speed_correct : kate_average_speed = 9 :=
by
  sorry

end NUMINAMATH_GPT_kate_average_speed_correct_l1468_146860


namespace NUMINAMATH_GPT_greatest_prime_factor_341_l1468_146899

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_341_l1468_146899


namespace NUMINAMATH_GPT_total_chickens_after_purchase_l1468_146851

def initial_chickens : ℕ := 400
def percentage_died : ℕ := 40
def times_to_buy : ℕ := 10

noncomputable def chickens_died : ℕ := (percentage_died * initial_chickens) / 100
noncomputable def chickens_remaining : ℕ := initial_chickens - chickens_died
noncomputable def chickens_bought : ℕ := times_to_buy * chickens_died
noncomputable def total_chickens : ℕ := chickens_remaining + chickens_bought

theorem total_chickens_after_purchase : total_chickens = 1840 :=
by
  sorry

end NUMINAMATH_GPT_total_chickens_after_purchase_l1468_146851


namespace NUMINAMATH_GPT_find_number_l1468_146852

theorem find_number (N Q : ℕ) (h1 : N = 11 * Q) (h2 : Q + N + 11 = 71) : N = 55 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l1468_146852


namespace NUMINAMATH_GPT_compute_b_l1468_146829

theorem compute_b (x y b : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 7 * y = 3 * b) (hx : x = 3) : b = 66 :=
sorry

end NUMINAMATH_GPT_compute_b_l1468_146829


namespace NUMINAMATH_GPT_selling_price_correct_l1468_146888

noncomputable def cost_price : ℝ := 90.91

noncomputable def profit_rate : ℝ := 0.10

noncomputable def profit : ℝ := profit_rate * cost_price

noncomputable def selling_price : ℝ := cost_price + profit

theorem selling_price_correct : selling_price = 100.00 := by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l1468_146888


namespace NUMINAMATH_GPT_toilet_paper_packs_needed_l1468_146879

-- Definitions based on conditions
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def weeks : ℕ := 4
def rolls_per_pack : ℕ := 12
def daily_stock : ℕ := 1

-- The main theorem statement
theorem toilet_paper_packs_needed : 
  (bathrooms * days_per_week * weeks) / rolls_per_pack = 14 := by
sorry

end NUMINAMATH_GPT_toilet_paper_packs_needed_l1468_146879


namespace NUMINAMATH_GPT_determine_k_a_l1468_146866

theorem determine_k_a (k a : ℝ) (h : k - a ≠ 0) : (k = 0 ∧ a = 1 / 2) ↔ 
  (∀ x : ℝ, (x + 2) / (kx - ax - 1) = x → x = -2) :=
by
  sorry

end NUMINAMATH_GPT_determine_k_a_l1468_146866


namespace NUMINAMATH_GPT_third_offense_percentage_increase_l1468_146854

theorem third_offense_percentage_increase 
    (base_per_5000 : ℕ)
    (goods_stolen : ℕ)
    (additional_years : ℕ)
    (total_sentence : ℕ) :
    base_per_5000 = 1 →
    goods_stolen = 40000 →
    additional_years = 2 →
    total_sentence = 12 →
    100 * (total_sentence - additional_years - goods_stolen / 5000) / (goods_stolen / 5000) = 25 :=
by
  intros h_base h_goods h_additional h_total
  sorry

end NUMINAMATH_GPT_third_offense_percentage_increase_l1468_146854


namespace NUMINAMATH_GPT_proof_equivalent_problem_l1468_146824

-- Definition of conditions
def cost_condition_1 (x y : ℚ) : Prop := 500 * x + 40 * y = 1250
def cost_condition_2 (x y : ℚ) : Prop := 1000 * x + 20 * y = 1000
def budget_condition (a b : ℕ) (total_masks : ℕ) (budget : ℕ) : Prop := 2 * a + (total_masks - a) / 2 + 25 * b = budget

-- Main theorem
theorem proof_equivalent_problem : 
  ∃ (x y : ℚ) (a b : ℕ), 
    cost_condition_1 x y ∧
    cost_condition_2 x y ∧
    (x = 1 / 2) ∧ 
    (y = 25) ∧
    (budget_condition a b 200 400) ∧
    ((a = 150 ∧ b = 3) ∨
     (a = 100 ∧ b = 6) ∨
     (a = 50 ∧ b = 9)) :=
by {
  sorry -- The proof steps are not required
}

end NUMINAMATH_GPT_proof_equivalent_problem_l1468_146824


namespace NUMINAMATH_GPT_tan_alpha_l1468_146819

theorem tan_alpha {α : ℝ} (h : 3 * Real.sin α + 4 * Real.cos α = 5) : Real.tan α = 3 / 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tan_alpha_l1468_146819


namespace NUMINAMATH_GPT_feet_to_inches_conversion_l1468_146885

-- Define the constant equivalence between feet and inches
def foot_to_inches := 12

-- Prove the conversion factor between feet and inches
theorem feet_to_inches_conversion:
  foot_to_inches = 12 :=
by
  sorry

end NUMINAMATH_GPT_feet_to_inches_conversion_l1468_146885


namespace NUMINAMATH_GPT_players_taking_all_three_subjects_l1468_146898

-- Define the variables for the number of players in each category
def num_players : ℕ := 18
def num_physics : ℕ := 10
def num_biology : ℕ := 7
def num_chemistry : ℕ := 5
def num_physics_biology : ℕ := 3
def num_biology_chemistry : ℕ := 2
def num_physics_chemistry : ℕ := 1

-- Define the proposition we want to prove
theorem players_taking_all_three_subjects :
  ∃ x : ℕ, x = 2 ∧
  num_players = num_physics + num_biology + num_chemistry
                - num_physics_chemistry
                - num_physics_biology
                - num_biology_chemistry
                + x :=
by {
  sorry -- Placeholder for the proof
}

end NUMINAMATH_GPT_players_taking_all_three_subjects_l1468_146898


namespace NUMINAMATH_GPT_a2017_value_l1468_146817

def seq (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) = a n / (a n + 1)

theorem a2017_value :
  ∃ (a : ℕ → ℝ),
  seq a ∧ a 1 = 1 / 2 ∧ a 2017 = 1 / 2018 :=
by
  sorry

end NUMINAMATH_GPT_a2017_value_l1468_146817


namespace NUMINAMATH_GPT_sequence_properties_l1468_146802

open BigOperators

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) := ∃ q > 0, ∀ n, a (n + 1) = a n * q
def sequence_a (n : ℕ) : ℝ := 2^(n - 1)

-- Definitions for b_n and S_n
def sequence_b (n : ℕ) : ℕ := n - 1
def sequence_c (n : ℕ) : ℝ := sequence_a n * (sequence_b n) -- c_n = a_n * b_n

-- Statement of the problem
theorem sequence_properties (a : ℕ → ℝ) (hgeo : is_geometric_sequence a) (h1 : a 1 = 1) (h2 : a 2 * a 4 = 16) : 
 (∀ n, sequence_b n = n - 1 ) ∧ S_n = ∑ i in Finset.range n, sequence_c (i + 1) := sorry

end NUMINAMATH_GPT_sequence_properties_l1468_146802


namespace NUMINAMATH_GPT_original_cube_volume_l1468_146865

theorem original_cube_volume (V₂ : ℝ) (s : ℝ) (h₀ : V₂ = 216) (h₁ : (2 * s) ^ 3 = V₂) : s ^ 3 = 27 := by
  sorry

end NUMINAMATH_GPT_original_cube_volume_l1468_146865


namespace NUMINAMATH_GPT_neg_09_not_in_integers_l1468_146876

def negative_numbers : Set ℝ := {x | x < 0}
def fractions : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}
def integers : Set ℝ := {x | ∃ (n : ℤ), x = n}
def rational_numbers : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

theorem neg_09_not_in_integers : -0.9 ∉ integers :=
by {
  sorry
}

end NUMINAMATH_GPT_neg_09_not_in_integers_l1468_146876


namespace NUMINAMATH_GPT_find_distance_l1468_146844

theorem find_distance (T D : ℝ) 
  (h1 : D = 5 * (T + 0.2)) 
  (h2 : D = 6 * (T - 0.25)) : 
  D = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_l1468_146844


namespace NUMINAMATH_GPT_printer_time_ratio_l1468_146810

theorem printer_time_ratio
  (X_time : ℝ) (Y_time : ℝ) (Z_time : ℝ)
  (hX : X_time = 15)
  (hY : Y_time = 10)
  (hZ : Z_time = 20) :
  (X_time / (Y_time * Z_time / (Y_time + Z_time))) = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_printer_time_ratio_l1468_146810


namespace NUMINAMATH_GPT_opposite_face_A_is_E_l1468_146870

-- Axiomatically defining the basic conditions from the problem statement.

-- We have six labels for the faces of a net
inductive Face : Type
| A | B | C | D | E | F

open Face

-- Define the adjacency relation
def adjacent (x y : Face) : Prop :=
  (x = A ∧ y = B) ∨ (x = A ∧ y = D) ∨ (x = B ∧ y = A) ∨ (x = D ∧ y = A)

-- Define the "not directly attached" relationship
def not_adjacent (x y : Face) : Prop :=
  ¬adjacent x y

-- Given the conditions in the problem statement
axiom condition1 : adjacent A B
axiom condition2 : adjacent A D
axiom condition3 : not_adjacent A E

-- The proof objective is to show that E is the face opposite to A
theorem opposite_face_A_is_E : ∃ (F : Face), 
  (∀ x : Face, adjacent A x ∨ not_adjacent A x) → (∀ y : Face, adjacent A y ↔ y ≠ E) → E = F :=
sorry

end NUMINAMATH_GPT_opposite_face_A_is_E_l1468_146870


namespace NUMINAMATH_GPT_pauls_score_is_91_l1468_146868

theorem pauls_score_is_91 (q s c w : ℕ) 
  (h1 : q = 35)
  (h2 : s = 35 + 5 * c - 2 * w)
  (h3 : s > 90)
  (h4 : c + w ≤ 35)
  (h5 : ∀ s', 90 < s' ∧ s' < s → ¬ (∃ c' w', s' = 35 + 5 * c' - 2 * w' ∧ c' + w' ≤ 35 ∧ c' ≠ c)) : 
  s = 91 := 
sorry

end NUMINAMATH_GPT_pauls_score_is_91_l1468_146868


namespace NUMINAMATH_GPT_power_function_monotonic_incr_l1468_146823

theorem power_function_monotonic_incr (m : ℝ) (h₁ : m^2 - 5 * m + 7 = 1) (h₂ : m^2 - 6 > 0) : m = 3 := 
by
  sorry

end NUMINAMATH_GPT_power_function_monotonic_incr_l1468_146823


namespace NUMINAMATH_GPT_common_tangents_count_l1468_146835

-- Define the first circle Q1
def Q1 (x y : ℝ) := x^2 + y^2 = 9

-- Define the second circle Q2
def Q2 (x y : ℝ) := (x - 3)^2 + (y - 4)^2 = 1

-- Prove the number of common tangents between Q1 and Q2
theorem common_tangents_count :
  ∃ n : ℕ, n = 4 ∧ ∀ x y : ℝ, Q1 x y ∧ Q2 x y -> n = 4 := sorry

end NUMINAMATH_GPT_common_tangents_count_l1468_146835


namespace NUMINAMATH_GPT_ratio_Pat_Mark_l1468_146807

-- Total hours charged by all three
def total_hours (P K M : ℕ) : Prop :=
  P + K + M = 144

-- Pat charged twice as much time as Kate
def pat_hours (P K : ℕ) : Prop :=
  P = 2 * K

-- Mark charged 80 hours more than Kate
def mark_hours (M K : ℕ) : Prop :=
  M = K + 80

-- The ratio of Pat's hours to Mark's hours
def ratio (P M : ℕ) : ℚ :=
  (P : ℚ) / (M : ℚ)

theorem ratio_Pat_Mark (P K M : ℕ)
  (h1 : total_hours P K M)
  (h2 : pat_hours P K)
  (h3 : mark_hours M K) :
  ratio P M = (1 : ℚ) / (3 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_ratio_Pat_Mark_l1468_146807


namespace NUMINAMATH_GPT_teamAPointDifferenceTeamB_l1468_146808

-- Definitions for players' scores and penalties
structure Player where
  name : String
  points : ℕ
  penalties : List ℕ

def TeamA : List Player := [
  { name := "Beth", points := 12, penalties := [1, 2] },
  { name := "Jan", points := 18, penalties := [1, 2, 3] },
  { name := "Mike", points := 5, penalties := [] },
  { name := "Kim", points := 7, penalties := [1, 2] },
  { name := "Chris", points := 6, penalties := [1] }
]

def TeamB : List Player := [
  { name := "Judy", points := 10, penalties := [1, 2] },
  { name := "Angel", points := 9, penalties := [1] },
  { name := "Nick", points := 12, penalties := [] },
  { name := "Steve", points := 8, penalties := [1, 2, 3] },
  { name := "Mary", points := 5, penalties := [1, 2] },
  { name := "Vera", points := 4, penalties := [1] }
]

-- Helper function to calculate total points for a player considering penalties
def Player.totalPoints (p : Player) : ℕ :=
  p.points - p.penalties.sum

-- Helper function to calculate total points for a team
def totalTeamPoints (team : List Player) : ℕ :=
  team.foldr (λ p acc => acc + p.totalPoints) 0

def teamAPoints : ℕ := totalTeamPoints TeamA
def teamBPoints : ℕ := totalTeamPoints TeamB

theorem teamAPointDifferenceTeamB :
  teamAPoints - teamBPoints = 1 :=
  sorry

end NUMINAMATH_GPT_teamAPointDifferenceTeamB_l1468_146808


namespace NUMINAMATH_GPT_total_amount_l1468_146895

theorem total_amount (x y z total : ℝ) (h1 : y = 0.45 * x) (h2 : z = 0.50 * x) (h3 : y = 27) : total = 117 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_total_amount_l1468_146895


namespace NUMINAMATH_GPT_tennis_racket_price_l1468_146856

theorem tennis_racket_price (P : ℝ) : 
    (0.8 * P + 515) * 1.10 + 20 = 800 → 
    P = 242.61 :=
by
  sorry

end NUMINAMATH_GPT_tennis_racket_price_l1468_146856


namespace NUMINAMATH_GPT_circle_not_pass_second_quadrant_l1468_146811

theorem circle_not_pass_second_quadrant (a : ℝ) : ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ (x - a)^2 + y^2 = 4) → a ≥ 2 :=
by
  intro h
  by_contra
  sorry

end NUMINAMATH_GPT_circle_not_pass_second_quadrant_l1468_146811


namespace NUMINAMATH_GPT_trig_problem_1_trig_problem_2_l1468_146805

-- Problem (1)
theorem trig_problem_1 (α : ℝ) (h1 : Real.tan (π + α) = -4 / 3) (h2 : 3 * Real.sin α / 4 = -Real.cos α)
  : Real.sin α = -4 / 5 ∧ Real.cos α = 3 / 5 := by
  sorry

-- Problem (2)
theorem trig_problem_2 : Real.sin (25 * π / 6) + Real.cos (26 * π / 3) + Real.tan (-25 * π / 4) = -1 := by
  sorry

end NUMINAMATH_GPT_trig_problem_1_trig_problem_2_l1468_146805


namespace NUMINAMATH_GPT_platform_length_correct_l1468_146816

noncomputable def platform_length (train_speed_kmph : ℝ) (crossing_time_s : ℝ) (train_length_m : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_covered := train_speed_mps * crossing_time_s
  distance_covered - train_length_m

theorem platform_length_correct :
  platform_length 72 26 260.0416 = 259.9584 :=
by
  sorry

end NUMINAMATH_GPT_platform_length_correct_l1468_146816


namespace NUMINAMATH_GPT_marginal_cost_proof_l1468_146820

theorem marginal_cost_proof (fixed_cost : ℕ) (total_cost : ℕ) (n : ℕ) (MC : ℕ)
  (h1 : fixed_cost = 12000)
  (h2 : total_cost = 16000)
  (h3 : n = 20)
  (h4 : total_cost = fixed_cost + MC * n) :
  MC = 200 :=
  sorry

end NUMINAMATH_GPT_marginal_cost_proof_l1468_146820


namespace NUMINAMATH_GPT_initial_concentration_l1468_146815

theorem initial_concentration (f : ℚ) (C : ℚ) (h₀ : f = 0.7142857142857143) (h₁ : (1 - f) * C + f * 0.25 = 0.35) : C = 0.6 :=
by
  rw [h₀] at h₁
  -- The proof will follow the steps to solve for C
  sorry

end NUMINAMATH_GPT_initial_concentration_l1468_146815


namespace NUMINAMATH_GPT_contractor_total_engaged_days_l1468_146864

-- Definitions based on conditions
def earnings_per_work_day : ℝ := 25
def fine_per_absent_day : ℝ := 7.5
def total_earnings : ℝ := 425
def days_absent : ℝ := 10

-- The proof problem statement
theorem contractor_total_engaged_days :
  ∃ (x y : ℝ), y = days_absent ∧ total_earnings = earnings_per_work_day * x - fine_per_absent_day * y ∧ x + y = 30 :=
by
  -- let x be the number of working days
  -- let y be the number of absent days
  -- y is given as 10
  -- total_earnings = 25 * x - 7.5 * 10
  -- solve for x and sum x and y to get 30
  sorry

end NUMINAMATH_GPT_contractor_total_engaged_days_l1468_146864


namespace NUMINAMATH_GPT_central_angle_of_sector_l1468_146877

theorem central_angle_of_sector (R r n : ℝ) (h_lateral_area : 2 * π * r^2 = π * r * R) 
  (h_arc_length : (n * π * R) / 180 = 2 * π * r) : n = 180 :=
by 
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l1468_146877


namespace NUMINAMATH_GPT_num_quadricycles_l1468_146834

theorem num_quadricycles (b t q : ℕ) (h1 : b + t + q = 10) (h2 : 2 * b + 3 * t + 4 * q = 30) : q = 2 :=
by sorry

end NUMINAMATH_GPT_num_quadricycles_l1468_146834


namespace NUMINAMATH_GPT_remainder_of_repeated_23_l1468_146825

theorem remainder_of_repeated_23 {n : ℤ} (n : ℤ) (hn : n = 23 * 10^(2*23)) : 
  (n % 32) = 19 :=
sorry

end NUMINAMATH_GPT_remainder_of_repeated_23_l1468_146825


namespace NUMINAMATH_GPT_simplify_expression_l1468_146894

theorem simplify_expression (a : Int) : 2 * a - a = a :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1468_146894


namespace NUMINAMATH_GPT_necessarily_negative_b_plus_3b_squared_l1468_146828

theorem necessarily_negative_b_plus_3b_squared
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 1) :
  b + 3 * b^2 < 0 :=
sorry

end NUMINAMATH_GPT_necessarily_negative_b_plus_3b_squared_l1468_146828


namespace NUMINAMATH_GPT_sum_of_coefficients_evaluated_l1468_146836

theorem sum_of_coefficients_evaluated 
  (x y : ℤ) (h1 : x = 2) (h2 : y = -1)
  : (3 * x + 4 * y)^9 + (2 * x - 5 * y)^9 = 387420501 := 
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_evaluated_l1468_146836


namespace NUMINAMATH_GPT_time_to_run_up_and_down_l1468_146867

/-- Problem statement: Prove that the time it takes Vasya to run up and down a moving escalator 
which moves upwards is 468 seconds, given these conditions:
1. Vasya runs down twice as fast as he runs up.
2. When the escalator is not working, it takes Vasya 6 minutes to run up and down.
3. When the escalator is moving down, it takes Vasya 13.5 minutes to run up and down.
--/
theorem time_to_run_up_and_down (up_speed down_speed : ℝ) (escalator_speed : ℝ) 
  (h1 : down_speed = 2 * up_speed) 
  (h2 : (1 / up_speed + 1 / down_speed) = 6) 
  (h3 : (1 / (up_speed + escalator_speed) + 1 / (down_speed - escalator_speed)) = 13.5) : 
  (1 / (up_speed - escalator_speed) + 1 / (down_speed + escalator_speed)) * 60 = 468 := 
sorry

end NUMINAMATH_GPT_time_to_run_up_and_down_l1468_146867


namespace NUMINAMATH_GPT_distance_to_base_is_42_l1468_146847

theorem distance_to_base_is_42 (x : ℕ) (hx : 4 * x + 3 * (x + 3) = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) :
  4 * x = 36 ∨ 4 * x + 6 = 42 := 
by
  sorry

end NUMINAMATH_GPT_distance_to_base_is_42_l1468_146847


namespace NUMINAMATH_GPT_triangle_side_identity_l1468_146803

theorem triangle_side_identity
  (a b c : ℝ)
  (alpha beta gamma : ℝ)
  (h1 : alpha = 60)
  (h2 : a^2 = b^2 + c^2 - b * c) :
  a^2 = (a^3 + b^3 + c^3) / (a + b + c) := 
by
  sorry

end NUMINAMATH_GPT_triangle_side_identity_l1468_146803


namespace NUMINAMATH_GPT_monotonic_increasing_interval_f_l1468_146832

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2 * x + 8)

theorem monotonic_increasing_interval_f :
  ∃ I : Set ℝ, (I = Set.Icc (-2) 1) ∧ (∀x1 ∈ I, ∀x2 ∈ I, x1 ≤ x2 → f x1 ≤ f x2) :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_f_l1468_146832


namespace NUMINAMATH_GPT_divisor_is_36_l1468_146881

theorem divisor_is_36
  (Dividend Quotient Remainder : ℕ)
  (h1 : Dividend = 690)
  (h2 : Quotient = 19)
  (h3 : Remainder = 6)
  (h4 : Dividend = (Divisor * Quotient) + Remainder) :
  Divisor = 36 :=
sorry

end NUMINAMATH_GPT_divisor_is_36_l1468_146881


namespace NUMINAMATH_GPT_max_sum_abs_values_l1468_146837

-- Define the main problem in Lean
theorem max_sum_abs_values (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  |a| + |b| + |c| ≤ 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_max_sum_abs_values_l1468_146837


namespace NUMINAMATH_GPT_number_of_red_candies_is_4_l1468_146814

-- Define the parameters as given in the conditions
def number_of_green_candies : ℕ := 5
def number_of_blue_candies : ℕ := 3
def likelihood_of_blue_candy : ℚ := 25 / 100

-- Define the total number of candies
def total_number_of_candies (number_of_red_candies : ℕ) : ℕ :=
  number_of_green_candies + number_of_blue_candies + number_of_red_candies

-- Define the proof statement
theorem number_of_red_candies_is_4 (R : ℕ) :
  (3 / total_number_of_candies R = 25 / 100) → R = 4 :=
sorry

end NUMINAMATH_GPT_number_of_red_candies_is_4_l1468_146814


namespace NUMINAMATH_GPT_quadratic_function_is_parabola_l1468_146830

theorem quadratic_function_is_parabola (a : ℝ) (b : ℝ) (c : ℝ) :
  ∃ k h, ∀ x, (y = a * (x - h)^2 + k) ∧ a ≠ 0 → (y = 3 * (x - 2)^2 + 6) → (a = 3 ∧ h = 2 ∧ k = 6) → ∀ x, (y = 3 * (x - 2)^2 + 6) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_function_is_parabola_l1468_146830


namespace NUMINAMATH_GPT_circular_patch_radius_l1468_146890

theorem circular_patch_radius : 
  let r_cylinder := 3  -- radius of the container in cm
  let h_cylinder := 6  -- height of the container in cm
  let t_patch := 0.2   -- thickness of each patch in cm
  let V := π * r_cylinder^2 * h_cylinder -- Volume of the liquid

  let V_patch := V / 2                  -- Volume of each patch
  let r := 3 * Real.sqrt 15              -- the radius we want to prove

  r^2 * π * t_patch = V_patch           -- the volume equation for one patch
  →

  r = 3 * Real.sqrt 15 := 
by
  sorry

end NUMINAMATH_GPT_circular_patch_radius_l1468_146890


namespace NUMINAMATH_GPT_simplify_expression_l1468_146841

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ( (x^6 - 1) / (3 * x^3) )^2) = Real.sqrt (x^12 + 7 * x^6 + 1) / (3 * x^3) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1468_146841


namespace NUMINAMATH_GPT_arrangement_count_l1468_146840

-- Definitions corresponding to the conditions in a)
def num_students : ℕ := 8
def max_per_activity : ℕ := 5

-- Lean statement reflecting the target theorem in c)
theorem arrangement_count (n : ℕ) (max : ℕ) 
  (h1 : n = num_students)
  (h2 : max = max_per_activity) :
  ∃ total : ℕ, total = 182 :=
sorry

end NUMINAMATH_GPT_arrangement_count_l1468_146840


namespace NUMINAMATH_GPT_largest_possible_value_of_m_l1468_146826

theorem largest_possible_value_of_m :
  ∃ (X Y Z : ℕ), 0 ≤ X ∧ X ≤ 7 ∧ 0 ≤ Y ∧ Y ≤ 7 ∧ 0 ≤ Z ∧ Z ≤ 7 ∧
                 (64 * X + 8 * Y + Z = 475) ∧ 
                 (144 * Z + 12 * Y + X = 475) := 
sorry

end NUMINAMATH_GPT_largest_possible_value_of_m_l1468_146826


namespace NUMINAMATH_GPT_distance_is_3_l1468_146822

-- define the distance between Masha's and Misha's homes
def distance_between_homes (d : ℝ) : Prop :=
  -- Masha and Misha meet 1 kilometer from Masha's home in the first occasion
  (∃ v_m v_i : ℝ, v_m > 0 ∧ v_i > 0 ∧
  1 / v_m = (d - 1) / v_i) ∧

  -- On the second occasion, Masha walked at twice her original speed,
  -- and Misha walked at half his original speed, and they met 1 kilometer away from Misha's home.
  (∃ v_m v_i : ℝ, v_m > 0 ∧ v_i > 0 ∧
  1 / (2 * v_m) = 2 * (d - 1) / (0.5 * v_i))

-- The theorem to prove the distance is 3
theorem distance_is_3 : distance_between_homes 3 :=
  sorry

end NUMINAMATH_GPT_distance_is_3_l1468_146822


namespace NUMINAMATH_GPT_find_other_person_money_l1468_146871

noncomputable def other_person_money (mias_money : ℕ) : ℕ :=
  let x := (mias_money - 20) / 2
  x

theorem find_other_person_money (mias_money : ℕ) (h_mias_money : mias_money = 110) : 
  other_person_money mias_money = 45 := by
  sorry

end NUMINAMATH_GPT_find_other_person_money_l1468_146871


namespace NUMINAMATH_GPT_karlson_word_count_l1468_146804

def single_word_count : Nat := 9
def ten_to_nineteen_count : Nat := 10
def two_word_count (num_tens_units : Nat) : Nat := 2 * num_tens_units

def count_words_1_to_99 : Nat :=
  let single_word := single_word_count + ten_to_nineteen_count
  let two_word := two_word_count (99 - (single_word_count + ten_to_nineteen_count))
  single_word + two_word

def prefix_hundred (count_1_to_99 : Nat) : Nat := 9 * count_1_to_99
def extra_prefix (num_two_word_transformed : Nat) : Nat := 9 * num_two_word_transformed

def total_words : Nat :=
  let first_99 := count_words_1_to_99
  let nine_hundreds := prefix_hundred count_words_1_to_99 + extra_prefix 72
  first_99 + nine_hundreds + 37

theorem karlson_word_count : total_words = 2611 :=
  by
    sorry

end NUMINAMATH_GPT_karlson_word_count_l1468_146804


namespace NUMINAMATH_GPT_texts_sent_total_l1468_146861

def texts_sent_on_monday_to_allison_and_brittney : Nat := 5 + 5
def texts_sent_on_tuesday_to_allison_and_brittney : Nat := 15 + 15

def total_texts_sent (texts_monday : Nat) (texts_tuesday : Nat) : Nat := texts_monday + texts_tuesday

theorem texts_sent_total :
  total_texts_sent texts_sent_on_monday_to_allison_and_brittney texts_sent_on_tuesday_to_allison_and_brittney = 40 :=
by
  sorry

end NUMINAMATH_GPT_texts_sent_total_l1468_146861


namespace NUMINAMATH_GPT_complex_fraction_value_l1468_146896

-- Define the imaginary unit
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_value : (3 : ℂ) / ((1 - i) ^ 2) = (3 / 2) * i := by
  sorry

end NUMINAMATH_GPT_complex_fraction_value_l1468_146896


namespace NUMINAMATH_GPT_area_of_rhombus_l1468_146850

/-- Given the radii of the circles circumscribed around triangles EFG and EGH
    are 10 and 20, respectively, then the area of rhombus EFGH is 30.72√3. -/
theorem area_of_rhombus (R1 R2 : ℝ) (A : ℝ) :
  R1 = 10 → R2 = 20 → A = 30.72 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_area_of_rhombus_l1468_146850


namespace NUMINAMATH_GPT_heartsuit_fraction_l1468_146869

def heartsuit (n m : ℕ) : ℕ := n ^ 4 * m ^ 3

theorem heartsuit_fraction :
  (heartsuit 3 5) / (heartsuit 5 3) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_heartsuit_fraction_l1468_146869


namespace NUMINAMATH_GPT_tree_growth_per_two_weeks_l1468_146874

-- Definitions based on conditions
def initial_height_meters : ℕ := 2
def initial_height_centimeters : ℕ := initial_height_meters * 100
def final_height_centimeters : ℕ := 600
def total_growth : ℕ := final_height_centimeters - initial_height_centimeters
def weeks_in_4_months : ℕ := 16
def number_of_two_week_periods : ℕ := weeks_in_4_months / 2

-- Objective: Prove that the growth every two weeks is 50 centimeters
theorem tree_growth_per_two_weeks :
  (total_growth / number_of_two_week_periods) = 50 :=
  by
  sorry

end NUMINAMATH_GPT_tree_growth_per_two_weeks_l1468_146874


namespace NUMINAMATH_GPT_triangles_in_figure_l1468_146842

-- Definitions for the figure
def number_of_triangles : ℕ :=
  -- The number of triangles in a figure composed of a rectangle with three vertical lines and two horizontal lines
  50

-- The theorem we want to prove
theorem triangles_in_figure : number_of_triangles = 50 :=
by
  sorry

end NUMINAMATH_GPT_triangles_in_figure_l1468_146842


namespace NUMINAMATH_GPT_find_other_root_l1468_146873

theorem find_other_root 
  (m : ℚ) 
  (h : 3 * 3^2 + m * 3 - 5 = 0) :
  (1 - 3) * (x : ℚ) = 0 :=
sorry

end NUMINAMATH_GPT_find_other_root_l1468_146873


namespace NUMINAMATH_GPT_total_books_l1468_146872

-- Definitions for the conditions
def SandyBooks : Nat := 10
def BennyBooks : Nat := 24
def TimBooks : Nat := 33

-- Stating the theorem we need to prove
theorem total_books : SandyBooks + BennyBooks + TimBooks = 67 := by
  sorry

end NUMINAMATH_GPT_total_books_l1468_146872


namespace NUMINAMATH_GPT_magnitude_difference_l1468_146801

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (norm_a : ‖a‖ = 2) (norm_b : ‖b‖ = 1) (norm_a_plus_b : ‖a + b‖ = Real.sqrt 3)

theorem magnitude_difference :
  ‖a - b‖ = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_difference_l1468_146801


namespace NUMINAMATH_GPT_sum_of_roots_l1468_146878

theorem sum_of_roots :
  let a := 1
  let b := 10
  let c := -25
  let sum_of_roots := -b / a
  (∀ x, 25 - 10 * x - x ^ 2 = 0 ↔ x ^ 2 + 10 * x - 25 = 0) →
  sum_of_roots = -10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1468_146878


namespace NUMINAMATH_GPT_denise_travel_l1468_146806

theorem denise_travel (a b c : ℕ) (h₀ : a ≥ 1) (h₁ : a + b + c = 8) (h₂ : 90 * (b - a) % 48 = 0) : a^2 + b^2 + c^2 = 26 :=
sorry

end NUMINAMATH_GPT_denise_travel_l1468_146806


namespace NUMINAMATH_GPT_tissue_pallets_ratio_l1468_146813

-- Define the total number of pallets received
def total_pallets : ℕ := 20

-- Define the number of pallets of each type
def paper_towels_pallets : ℕ := total_pallets / 2
def paper_plates_pallets : ℕ := total_pallets / 5
def paper_cups_pallets : ℕ := 1

-- Calculate the number of pallets of tissues
def tissues_pallets : ℕ := total_pallets - (paper_towels_pallets + paper_plates_pallets + paper_cups_pallets)

-- Prove the ratio of pallets of tissues to total pallets is 1/4
theorem tissue_pallets_ratio : (tissues_pallets : ℚ) / total_pallets = 1 / 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tissue_pallets_ratio_l1468_146813


namespace NUMINAMATH_GPT_total_amount_l1468_146818

noncomputable def x_share : ℝ := 60
noncomputable def y_share : ℝ := 27
noncomputable def z_share : ℝ := 0.30 * x_share

theorem total_amount (hx : y_share = 0.45 * x_share) : x_share + y_share + z_share = 105 :=
by
  have hx_val : x_share = 27 / 0.45 := by
  { -- Proof that x_share is indeed 60 as per the given problem
    sorry }
  sorry

end NUMINAMATH_GPT_total_amount_l1468_146818


namespace NUMINAMATH_GPT_smallest_possible_sum_l1468_146849

-- Defining the conditions for x and y.
variables (x y : ℕ)

-- We need a theorem to formalize our question with the given conditions.
theorem smallest_possible_sum (hx : x > 0) (hy : y > 0) (hne : x ≠ y) (hxy : 1/x + 1/y = 1/24) : x + y = 100 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_sum_l1468_146849


namespace NUMINAMATH_GPT_simple_interest_double_in_4_years_interest_25_percent_l1468_146863

theorem simple_interest_double_in_4_years_interest_25_percent :
  ∀ {P : ℕ} (h : P > 0), ∃ (R : ℕ), R = 25 ∧ P + P * R * 4 / 100 = 2 * P :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_double_in_4_years_interest_25_percent_l1468_146863


namespace NUMINAMATH_GPT_find_power_of_7_l1468_146839

theorem find_power_of_7 :
  (7^(1/4)) / (7^(1/6)) = 7^(1/12) :=
by
  sorry

end NUMINAMATH_GPT_find_power_of_7_l1468_146839


namespace NUMINAMATH_GPT_inequality_proof_l1468_146812

theorem inequality_proof (a b c : ℝ) : 
  1 < (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + 
  c / (Real.sqrt (c^2 + a^2))) ∧ 
  (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + 
  c / (Real.sqrt (c^2 + a^2))) ≤ (3 * Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1468_146812


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_conversion_l1468_146838

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular 6 (5 * Real.pi / 4) (-3) = (-3 * Real.sqrt 2, -3 * Real.sqrt 2, -3) :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_conversion_l1468_146838


namespace NUMINAMATH_GPT_kopecks_to_rubles_l1468_146889

noncomputable def exchangeable_using_coins (total : ℕ) (num_coins : ℕ) : Prop :=
  ∃ (x y z t u v w : ℕ), 
    total = x * 1 + y * 2 + z * 5 + t * 10 + u * 20 + v * 50 + w * 100 ∧ 
    num_coins = x + y + z + t + u + v + w

theorem kopecks_to_rubles (A B : ℕ)
  (h : exchangeable_using_coins A B) : exchangeable_using_coins (100 * B) A :=
sorry

end NUMINAMATH_GPT_kopecks_to_rubles_l1468_146889


namespace NUMINAMATH_GPT_intersection_proof_l1468_146892

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def N : Set ℕ := { x | Real.sqrt (2^x - 1) < 5 }
def expected_intersection : Set ℕ := {1, 2, 3, 4}

theorem intersection_proof : M ∩ N = expected_intersection := by
  sorry

end NUMINAMATH_GPT_intersection_proof_l1468_146892


namespace NUMINAMATH_GPT_inequality_proof_l1468_146857

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem inequality_proof :
  (1 + a) / (1 - a) + (1 + b) / (1 - a) + (1 + c) / (1 - c) ≤ 2 * ((b / a) + (c / b) + (a / c)) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l1468_146857


namespace NUMINAMATH_GPT_simplify_expression_l1468_146845

-- Define the given expression
def given_expr (x y : ℝ) := 3 * x + 4 * y + 5 * x^2 + 2 - (8 - 5 * x - 3 * y - 2 * x^2)

-- Define the expected simplified expression
def simplified_expr (x y : ℝ) := 7 * x^2 + 8 * x + 7 * y - 6

-- Theorem statement to prove the equivalence of the expressions
theorem simplify_expression (x y : ℝ) : 
  given_expr x y = simplified_expr x y := sorry

end NUMINAMATH_GPT_simplify_expression_l1468_146845


namespace NUMINAMATH_GPT_parabola_axis_of_symmetry_l1468_146833

theorem parabola_axis_of_symmetry (p : ℝ) :
  (∀ x : ℝ, x = 3 → -x^2 - p*x + 2 = -x^2 - (-6)*x + 2) → p = -6 :=
by sorry

end NUMINAMATH_GPT_parabola_axis_of_symmetry_l1468_146833


namespace NUMINAMATH_GPT_find_a_l1468_146809

/-- 
Given sets A and B defined by specific quadratic equations, 
if A ∪ B = A, then a ∈ (-∞, 0).
-/
theorem find_a :
  ∀ (a : ℝ),
    (A = {x : ℝ | x^2 - 3 * x + 2 = 0}) →
    (B = {x : ℝ | x^2 - 2 * a * x + a^2 - a = 0}) →
    (A ∪ B = A) →
    a < 0 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1468_146809


namespace NUMINAMATH_GPT_determine_a_b_l1468_146843

theorem determine_a_b (a b : ℤ) :
  (∀ x : ℤ, x^2 + a * x + b = (x - 1) * (x + 4)) → (a = 3 ∧ b = -4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_a_b_l1468_146843


namespace NUMINAMATH_GPT_find_a_l1468_146821

def new_operation (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a (a : ℝ) (b : ℝ) (h : b = 4) (h2 : new_operation a b = 10) : a = 14 := by
  have h' : new_operation a 4 = 10 := by rw [h] at h2; exact h2
  unfold new_operation at h'
  linarith

end NUMINAMATH_GPT_find_a_l1468_146821


namespace NUMINAMATH_GPT_peasant_initial_money_l1468_146858

theorem peasant_initial_money :
  ∃ (x1 x2 x3 : ℕ), 
    (x1 / 2 + 1 = x2) ∧ 
    (x2 / 2 + 2 = x3) ∧ 
    (x3 / 2 + 1 = 0) ∧ 
    x1 = 18 := 
by
  sorry

end NUMINAMATH_GPT_peasant_initial_money_l1468_146858


namespace NUMINAMATH_GPT_cos_three_theta_l1468_146880

open Complex

theorem cos_three_theta (θ : ℝ) (h : cos θ = 1 / 2) : cos (3 * θ) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_three_theta_l1468_146880


namespace NUMINAMATH_GPT_Irene_age_is_46_l1468_146846

-- Definitions based on the given conditions
def Eddie_age : ℕ := 92
def Becky_age : ℕ := Eddie_age / 4
def Irene_age : ℕ := 2 * Becky_age

-- Theorem we aim to prove that Irene's age is 46
theorem Irene_age_is_46 : Irene_age = 46 := by
  sorry

end NUMINAMATH_GPT_Irene_age_is_46_l1468_146846


namespace NUMINAMATH_GPT_min_value_x2_y2_l1468_146855

theorem min_value_x2_y2 (x y : ℝ) (h : 2 * x + y + 5 = 0) : x^2 + y^2 ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x2_y2_l1468_146855


namespace NUMINAMATH_GPT_simplify_expression_l1468_146862

theorem simplify_expression : 
  (1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3 + 1 / (1 / 3)^4)) = 1 / 120 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1468_146862


namespace NUMINAMATH_GPT_man_work_days_l1468_146848

theorem man_work_days (M : ℕ) (h1 : (1 : ℝ)/M + (1 : ℝ)/10 = 1/5) : M = 10 :=
sorry

end NUMINAMATH_GPT_man_work_days_l1468_146848


namespace NUMINAMATH_GPT_chrysler_floors_difference_l1468_146886

theorem chrysler_floors_difference (C L : ℕ) (h1 : C = 23) (h2 : C + L = 35) : C - L = 11 := by
  sorry

end NUMINAMATH_GPT_chrysler_floors_difference_l1468_146886


namespace NUMINAMATH_GPT_total_amount_in_account_after_two_years_l1468_146800

-- Initial definitions based on conditions in the problem
def initial_investment : ℝ := 76800
def annual_interest_rate : ℝ := 0.125
def annual_contribution : ℝ := 5000

-- Function to calculate amount after n years with annual contributions
def total_amount_after_years (P : ℝ) (r : ℝ) (A : ℝ) (n : ℕ) : ℝ :=
  let rec helper (P : ℝ) (n : ℕ) :=
    if n = 0 then P
    else 
      let previous_amount := helper P (n - 1)
      (previous_amount * (1 + r) + A)
  helper P n

-- Theorem to prove the final total amount after 2 years
theorem total_amount_in_account_after_two_years :
  total_amount_after_years initial_investment annual_interest_rate annual_contribution 2 = 107825 :=
  by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_amount_in_account_after_two_years_l1468_146800
