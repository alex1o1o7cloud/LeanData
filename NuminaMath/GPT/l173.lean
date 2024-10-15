import Mathlib

namespace NUMINAMATH_GPT_joe_paid_4_more_than_jenny_l173_17344

theorem joe_paid_4_more_than_jenny
  (total_plain_pizza_cost : ℕ := 12) 
  (total_slices : ℕ := 12)
  (additional_cost_per_mushroom_slice : ℕ := 1) -- 0.50 dollars represented in integer (value in cents or minimal currency unit)
  (mushroom_slices : ℕ := 4) 
  (plain_slices := total_slices - mushroom_slices) -- Calculate plain slices.
  (total_additional_cost := mushroom_slices * additional_cost_per_mushroom_slice)
  (total_pizza_cost := total_plain_pizza_cost + total_additional_cost)
  (plain_slice_cost := total_plain_pizza_cost / total_slices)
  (mushroom_slice_cost := plain_slice_cost + additional_cost_per_mushroom_slice) 
  (joe_mushroom_slices := mushroom_slices) 
  (joe_plain_slices := 3) 
  (jenny_plain_slices := plain_slices - joe_plain_slices) 
  (joe_paid := (joe_mushroom_slices * mushroom_slice_cost) + (joe_plain_slices * plain_slice_cost))
  (jenny_paid := jenny_plain_slices * plain_slice_cost) : 
  joe_paid - jenny_paid = 4 := 
by {
  -- Here, we define the steps we used to calculate the cost.
  sorry -- Proof skipped as per instructions.
}

end NUMINAMATH_GPT_joe_paid_4_more_than_jenny_l173_17344


namespace NUMINAMATH_GPT_sum_infinite_series_l173_17333

theorem sum_infinite_series : 
  ∑' n : ℕ, (3 * (n + 1) - 1) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3)) = 73 / 12 := 
by sorry

end NUMINAMATH_GPT_sum_infinite_series_l173_17333


namespace NUMINAMATH_GPT_solve_for_T_l173_17367

theorem solve_for_T : ∃ T : ℝ, (3 / 4) * (1 / 6) * T = (2 / 5) * (1 / 4) * 200 ∧ T = 80 :=
by
  use 80
  -- The proof part is omitted as instructed
  sorry

end NUMINAMATH_GPT_solve_for_T_l173_17367


namespace NUMINAMATH_GPT_smallest_possible_k_l173_17356

def infinite_increasing_seq (a : ℕ → ℕ) : Prop :=
∀ n, a n < a (n + 1)

def divisible_by_1005_or_1006 (a : ℕ) : Prop :=
a % 1005 = 0 ∨ a % 1006 = 0

def not_divisible_by_97 (a : ℕ) : Prop :=
a % 97 ≠ 0

def diff_less_than_k (a : ℕ → ℕ) (k : ℕ) : Prop :=
∀ n, (a (n + 1) - a n) ≤ k

theorem smallest_possible_k :
  ∀ (a : ℕ → ℕ), infinite_increasing_seq a →
  (∀ n, divisible_by_1005_or_1006 (a n)) →
  (∀ n, not_divisible_by_97 (a n)) →
  (∃ k, diff_less_than_k a k) →
  (∃ k, k = 2010 ∧ diff_less_than_k a k) :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_k_l173_17356


namespace NUMINAMATH_GPT_ratio_of_third_to_second_l173_17391

-- Assume we have three numbers (a, b, c) where
-- 1. b = 2 * a
-- 2. c = k * b
-- 3. (a + b + c) / 3 = 165
-- 4. a = 45

theorem ratio_of_third_to_second (a b c k : ℝ) (h1 : b = 2 * a) (h2 : c = k * b) 
  (h3 : (a + b + c) / 3 = 165) (h4 : a = 45) : k = 4 := by 
  sorry

end NUMINAMATH_GPT_ratio_of_third_to_second_l173_17391


namespace NUMINAMATH_GPT_min_distance_from_C_to_circle_l173_17317

theorem min_distance_from_C_to_circle
  (R : ℝ) (AC : ℝ) (CB : ℝ) (C M : ℝ)
  (hR : R = 6) (hAC : AC = 4) (hCB : CB = 5)
  (hCM_eq : C = 12 - M) :
  C * M = 20 → (M < 6) → M = 2 := 
sorry

end NUMINAMATH_GPT_min_distance_from_C_to_circle_l173_17317


namespace NUMINAMATH_GPT_dennis_initial_money_l173_17393

def initial_money (shirt_cost: ℕ) (ten_dollar_bills: ℕ) (loose_coins: ℕ) : ℕ :=
  shirt_cost + (10 * ten_dollar_bills) + loose_coins

theorem dennis_initial_money : initial_money 27 2 3 = 50 :=
by 
  -- Here would go the proof steps based on the solution steps identified before
  sorry

end NUMINAMATH_GPT_dennis_initial_money_l173_17393


namespace NUMINAMATH_GPT_line_through_points_eq_l173_17303

theorem line_through_points_eq
  (x1 y1 x2 y2 : ℝ)
  (h1 : 2 * x1 + 3 * y1 = 4)
  (h2 : 2 * x2 + 3 * y2 = 4) :
  ∃ m b : ℝ, (∀ x y : ℝ, (y = m * x + b) ↔ (2 * x + 3 * y = 4)) :=
by
  sorry

end NUMINAMATH_GPT_line_through_points_eq_l173_17303


namespace NUMINAMATH_GPT_escalator_rate_is_15_l173_17330

noncomputable def rate_escalator_moves (escalator_length : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_length / time) - person_speed

theorem escalator_rate_is_15 :
  rate_escalator_moves 200 5 10 = 15 := by
  sorry

end NUMINAMATH_GPT_escalator_rate_is_15_l173_17330


namespace NUMINAMATH_GPT_value_of_n_l173_17395

theorem value_of_n (n : ℕ) : (1 / 5 : ℝ) ^ n * (1 / 4 : ℝ) ^ 18 = 1 / (2 * (10 : ℝ) ^ 35) → n = 35 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_n_l173_17395


namespace NUMINAMATH_GPT_length_of_AD_l173_17345

theorem length_of_AD (AB BC CD DE : ℝ) (right_angle_B right_angle_C : Prop) :
  AB = 6 → BC = 7 → CD = 25 → DE = 15 → AD = Real.sqrt 274 :=
by
  intros
  sorry

end NUMINAMATH_GPT_length_of_AD_l173_17345


namespace NUMINAMATH_GPT_select_1996_sets_l173_17337

theorem select_1996_sets (k : ℕ) (sets : Finset (Finset ℕ)) (h : k > 1993006) (h_sets : sets.card = k) :
  ∃ (selected_sets : Finset (Finset ℕ)), selected_sets.card = 1996 ∧
  ∀ (x y z : Finset ℕ), x ∈ selected_sets → y ∈ selected_sets → z ∈ selected_sets → z = x ∪ y → false :=
sorry

end NUMINAMATH_GPT_select_1996_sets_l173_17337


namespace NUMINAMATH_GPT_henry_games_given_l173_17307

theorem henry_games_given (G : ℕ) (henry_initial : ℕ) (neil_initial : ℕ) (henry_now : ℕ) (neil_now : ℕ) :
  henry_initial = 58 →
  neil_initial = 7 →
  henry_now = henry_initial - G →
  neil_now = neil_initial + G →
  henry_now = 4 * neil_now →
  G = 6 :=
by
  intros h_initial n_initial h_now n_now eq_henry
  sorry

end NUMINAMATH_GPT_henry_games_given_l173_17307


namespace NUMINAMATH_GPT_year_2049_is_Jisi_l173_17392

-- Define Heavenly Stems
def HeavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]

-- Define Earthly Branches
def EarthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

-- Define the indices of Ding (丁) and You (酉) based on 2017
def Ding_index : Nat := 3
def You_index : Nat := 9

-- Define the year difference
def year_difference : Nat := 2049 - 2017

-- Calculate the indices for the Heavenly Stem and Earthly Branch in 2049
def HeavenlyStem_index_2049 : Nat := (Ding_index + year_difference) % 10
def EarthlyBranch_index_2049 : Nat := (You_index + year_difference) % 12

theorem year_2049_is_Jisi : 
  HeavenlyStems[HeavenlyStem_index_2049]? = some "Ji" ∧ EarthlyBranches[EarthlyBranch_index_2049]? = some "Si" :=
by
  sorry

end NUMINAMATH_GPT_year_2049_is_Jisi_l173_17392


namespace NUMINAMATH_GPT_find_number_l173_17327

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end NUMINAMATH_GPT_find_number_l173_17327


namespace NUMINAMATH_GPT_baseball_team_groups_l173_17376

theorem baseball_team_groups
  (new_players : ℕ) 
  (returning_players : ℕ)
  (players_per_group : ℕ)
  (total_players : ℕ := new_players + returning_players) :
  new_players = 48 → 
  returning_players = 6 → 
  players_per_group = 6 → 
  total_players / players_per_group = 9 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end NUMINAMATH_GPT_baseball_team_groups_l173_17376


namespace NUMINAMATH_GPT_divisibility_by_5_l173_17383

theorem divisibility_by_5 (x y : ℤ) : (x^2 - 2 * x * y + 2 * y^2) % 5 = 0 ∨ (x^2 + 2 * x * y + 2 * y^2) % 5 = 0 ↔ (x % 5 = 0 ∧ y % 5 = 0) ∨ (x % 5 ≠ 0 ∧ y % 5 ≠ 0) := 
by
  sorry

end NUMINAMATH_GPT_divisibility_by_5_l173_17383


namespace NUMINAMATH_GPT_sequence_evaluation_l173_17320

noncomputable def a : ℕ → ℤ → ℤ
| 0, x => 1
| 1, x => x^2 + x + 1
| (n + 2), x => (x^n + 1) * a (n + 1) x - a n x 

theorem sequence_evaluation : a 2010 1 = 4021 := by
  sorry

end NUMINAMATH_GPT_sequence_evaluation_l173_17320


namespace NUMINAMATH_GPT_roots_triple_relation_l173_17305

theorem roots_triple_relation (p q r α β : ℝ) (h1 : α + β = -q / p) (h2 : α * β = r / p) (h3 : β = 3 * α) :
  3 * q ^ 2 = 16 * p * r :=
sorry

end NUMINAMATH_GPT_roots_triple_relation_l173_17305


namespace NUMINAMATH_GPT_reciprocal_of_mixed_num_l173_17341

-- Define the fraction representation of the mixed number -1 1/2
def mixed_num_to_improper (a : ℚ) : ℚ := -3/2

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Prove the statement
theorem reciprocal_of_mixed_num : reciprocal (mixed_num_to_improper (-1.5)) = -2/3 :=
by
  -- skip proof
  sorry

end NUMINAMATH_GPT_reciprocal_of_mixed_num_l173_17341


namespace NUMINAMATH_GPT_cost_price_books_l173_17354

def cost_of_type_A (cost_A cost_B : ℝ) : Prop :=
  cost_A = cost_B + 15

def quantity_equal (cost_A cost_B : ℝ) : Prop :=
  675 / cost_A = 450 / cost_B

theorem cost_price_books (cost_A cost_B : ℝ) (h1 : cost_of_type_A cost_A cost_B) (h2 : quantity_equal cost_A cost_B) : 
  cost_A = 45 ∧ cost_B = 30 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cost_price_books_l173_17354


namespace NUMINAMATH_GPT_circle_tangent_line_l173_17315

theorem circle_tangent_line {m : ℝ} : 
  (3 * (0 : ℝ) - 4 * (1 : ℝ) - 6 = 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 2 * y + m = 0) → 
  m = -3 := by
  sorry

end NUMINAMATH_GPT_circle_tangent_line_l173_17315


namespace NUMINAMATH_GPT_find_x_l173_17397

-- Define the known values
def a := 6
def b := 16
def c := 8
def desired_average := 13

-- Define the target number we need to find
def target_x := 22

-- Prove that the number we need to add to get the desired average is 22
theorem find_x : (a + b + c + target_x) / 4 = desired_average :=
by
  -- The proof itself is omitted as per instructions
  sorry

end NUMINAMATH_GPT_find_x_l173_17397


namespace NUMINAMATH_GPT_subset_bound_l173_17364

theorem subset_bound {m n k : ℕ} (h1 : m ≥ n) (h2 : n > 1) 
  (F : Fin k → Finset (Fin m)) 
  (hF : ∀ i j, i < j → (F i ∩ F j).card ≤ 1) 
  (hcard : ∀ i, (F i).card = n) : 
  k ≤ (m * (m - 1)) / (n * (n - 1)) :=
sorry

end NUMINAMATH_GPT_subset_bound_l173_17364


namespace NUMINAMATH_GPT_identity_proof_l173_17368

theorem identity_proof (x y : ℝ) (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 55) : x^2 - y^2 = 1 / 121 :=
by 
  sorry

end NUMINAMATH_GPT_identity_proof_l173_17368


namespace NUMINAMATH_GPT_division_problem_l173_17365

theorem division_problem : (5 * 8) / 10 = 4 := by
  sorry

end NUMINAMATH_GPT_division_problem_l173_17365


namespace NUMINAMATH_GPT_smallest_lambda_inequality_l173_17347

theorem smallest_lambda_inequality 
  (a b c d : ℝ) (h_pos : ∀ x ∈ [a, b, c, d], 0 < x) (h_sum : a + b + c + d = 4) :
  5 * (a*b + a*c + a*d + b*c + b*d + c*d) ≤ 8 * (a*b*c*d) + 12 :=
sorry

end NUMINAMATH_GPT_smallest_lambda_inequality_l173_17347


namespace NUMINAMATH_GPT_nine_chapters_problem_l173_17387

theorem nine_chapters_problem (n x : ℤ) (h1 : 8 * n = x + 3) (h2 : 7 * n = x - 4) :
  (x + 3) / 8 = (x - 4) / 7 :=
  sorry

end NUMINAMATH_GPT_nine_chapters_problem_l173_17387


namespace NUMINAMATH_GPT_set_intersection_complement_l173_17332

-- Definitions of the sets A and B
def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

-- Statement of the problem for Lean 4
theorem set_intersection_complement :
  A ∩ (Set.compl B) = {1, 5, 7} := 
sorry

end NUMINAMATH_GPT_set_intersection_complement_l173_17332


namespace NUMINAMATH_GPT_total_birds_in_store_l173_17350

def num_bird_cages := 4
def parrots_per_cage := 8
def parakeets_per_cage := 2
def birds_per_cage := parrots_per_cage + parakeets_per_cage
def total_birds := birds_per_cage * num_bird_cages

theorem total_birds_in_store : total_birds = 40 :=
  by sorry

end NUMINAMATH_GPT_total_birds_in_store_l173_17350


namespace NUMINAMATH_GPT_probability_useful_parts_l173_17316

noncomputable def probability_three_parts_useful (pipe_length : ℝ) (min_length : ℝ) : ℝ :=
  let total_area := (pipe_length * pipe_length) / 2
  let feasible_area := ((pipe_length - min_length) * (pipe_length - min_length)) / 2
  feasible_area / total_area

theorem probability_useful_parts :
  probability_three_parts_useful 300 75 = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_useful_parts_l173_17316


namespace NUMINAMATH_GPT_inequality_solution_l173_17374

theorem inequality_solution (x : ℝ)
  (h : ∀ x, x^2 + 2 * x + 7 > 0) :
  (x - 3) / (x^2 + 2 * x + 7) ≥ 0 ↔ x ∈ Set.Ici 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l173_17374


namespace NUMINAMATH_GPT_divisor_of_polynomial_l173_17324

theorem divisor_of_polynomial (a : ℤ) (h : ∀ x : ℤ, (x^2 - x + a) ∣ (x^13 + x + 180)) : a = 1 :=
sorry

end NUMINAMATH_GPT_divisor_of_polynomial_l173_17324


namespace NUMINAMATH_GPT_max_ab_condition_l173_17302

theorem max_ab_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 4 = 0)
  (line_check : ∀ x y : ℝ, (x = 1 ∧ y = -2) → 2*a*x - b*y - 2 = 0) : ab ≤ 1/4 :=
by
  sorry

end NUMINAMATH_GPT_max_ab_condition_l173_17302


namespace NUMINAMATH_GPT_simplify_expression_l173_17369

theorem simplify_expression
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (cos_double_angle : ∀ x, cos (2 * x) = cos x * cos x - sin x * sin x)
  (sin_double_angle : ∀ x, sin (2 * x) = 2 * sin x * cos x)
  (sin_cofunction : ∀ x, sin (Real.pi / 2 - x) = cos x) :
  (cos 5 * cos 5 - sin 5 * sin 5) / (sin 40 * cos 40) = 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l173_17369


namespace NUMINAMATH_GPT_perimeter_of_triangle_ABC_l173_17328

noncomputable def triangle_perimeter (r1 r2 r3 : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ :=
  let x1 := r1 * Real.cos θ1
  let y1 := r1 * Real.sin θ1
  let x2 := r2 * Real.cos θ2
  let y2 := r2 * Real.sin θ2
  let x3 := r3 * Real.cos θ3
  let y3 := r3 * Real.sin θ3
  let d12 := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let d23 := Real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)
  let d31 := Real.sqrt ((x3 - x1)^2 + (y3 - y1)^2)
  d12 + d23 + d31

--prove

theorem perimeter_of_triangle_ABC (θ1 θ2 θ3: ℝ)
  (h1: θ1 - θ2 = Real.pi / 3)
  (h2: θ2 - θ3 = Real.pi / 3) :
  triangle_perimeter 4 5 7 θ1 θ2 θ3 = sorry := 
sorry

end NUMINAMATH_GPT_perimeter_of_triangle_ABC_l173_17328


namespace NUMINAMATH_GPT_trig_sum_identity_l173_17399

theorem trig_sum_identity :
  Real.sin (47 * Real.pi / 180) * Real.cos (43 * Real.pi / 180) 
  + Real.sin (137 * Real.pi / 180) * Real.sin (43 * Real.pi / 180) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trig_sum_identity_l173_17399


namespace NUMINAMATH_GPT_sofia_total_cost_l173_17372

def shirt_cost : ℕ := 7
def shoes_cost : ℕ := shirt_cost + 3
def two_shirts_cost : ℕ := 2 * shirt_cost
def total_clothes_cost : ℕ := two_shirts_cost + shoes_cost
def bag_cost : ℕ := total_clothes_cost / 2
def total_cost : ℕ := two_shirts_cost + shoes_cost + bag_cost

theorem sofia_total_cost : total_cost = 36 := by
  sorry

end NUMINAMATH_GPT_sofia_total_cost_l173_17372


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l173_17390

variable {a : ℕ → ℝ} 

-- Condition: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Condition: Given sum of specific terms in the sequence
def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 10 = 16

-- Problem: Proving the correct answer
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : given_condition a) :
  a 4 + a 6 + a 8 = 24 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l173_17390


namespace NUMINAMATH_GPT_price_of_silver_l173_17321

theorem price_of_silver
  (side : ℕ) (side_eq : side = 3)
  (weight_per_cubic_inch : ℕ) (weight_per_cubic_inch_eq : weight_per_cubic_inch = 6)
  (selling_price : ℝ) (selling_price_eq : selling_price = 4455)
  (markup_percentage : ℝ) (markup_percentage_eq : markup_percentage = 1.10)
  : 4050 / 162 = 25 :=
by
  -- Given conditions are side_eq, weight_per_cubic_inch_eq, selling_price_eq, and markup_percentage_eq
  -- The statement requiring proof, i.e., price per ounce calculation, is provided.
  sorry

end NUMINAMATH_GPT_price_of_silver_l173_17321


namespace NUMINAMATH_GPT_angle_C_is_3pi_over_4_l173_17318

theorem angle_C_is_3pi_over_4 (A B C : ℝ) (a b c : ℝ) (h_tri : 0 < B ∧ B < π ∧ 0 < C ∧ C < π) 
  (h_eq : b * Real.cos C + c * Real.sin B = 0) : C = 3 * π / 4 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_is_3pi_over_4_l173_17318


namespace NUMINAMATH_GPT_total_marks_l173_17360

variable (E S M : Nat)

-- Given conditions
def thrice_as_many_marks_in_English_as_in_Science := E = 3 * S
def ratio_of_marks_in_English_and_Maths            := M = 4 * E
def marks_in_Science                               := S = 17

-- Proof problem statement
theorem total_marks (h1 : E = 3 * S) (h2 : M = 4 * E) (h3 : S = 17) :
  E + S + M = 272 :=
by
  sorry

end NUMINAMATH_GPT_total_marks_l173_17360


namespace NUMINAMATH_GPT_initial_rope_length_l173_17309

theorem initial_rope_length : 
  ∀ (π : ℝ), 
  ∀ (additional_area : ℝ) (new_rope_length : ℝ), 
  additional_area = 933.4285714285714 →
  new_rope_length = 21 →
  ∃ (initial_rope_length : ℝ), 
  additional_area = π * (new_rope_length^2 - initial_rope_length^2) ∧
  initial_rope_length = 12 :=
by
  sorry

end NUMINAMATH_GPT_initial_rope_length_l173_17309


namespace NUMINAMATH_GPT_outstanding_consumer_installment_credit_l173_17388

-- Given conditions
def total_consumer_installment_credit (C : ℝ) : Prop :=
  let automobile_installment_credit := 0.36 * C
  let automobile_finance_credit := 75
  let total_automobile_credit := 2 * automobile_finance_credit
  automobile_installment_credit = total_automobile_credit

-- Theorem to prove
theorem outstanding_consumer_installment_credit : ∃ (C : ℝ), total_consumer_installment_credit C ∧ C = 416.67 := 
by
  sorry

end NUMINAMATH_GPT_outstanding_consumer_installment_credit_l173_17388


namespace NUMINAMATH_GPT_symmetric_point_l173_17334

theorem symmetric_point : ∃ (x0 y0 : ℝ), 
  (x0 = -6 ∧ y0 = -3) ∧ 
  (∃ (m1 m2 : ℝ), 
    m1 = -1 ∧ 
    m2 = (y0 - 2) / (x0 + 1) ∧ 
    m1 * m2 = -1) ∧ 
  (∃ (x_mid y_mid : ℝ), 
    x_mid = (x0 - 1) / 2 ∧ 
    y_mid = (y0 + 2) / 2 ∧ 
    x_mid + y_mid + 4 = 0) := 
sorry

end NUMINAMATH_GPT_symmetric_point_l173_17334


namespace NUMINAMATH_GPT_sum_of_three_numbers_is_71_point_5_l173_17355

noncomputable def sum_of_three_numbers (a b c : ℝ) : ℝ :=
a + b + c

theorem sum_of_three_numbers_is_71_point_5 (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 48) (h3 : c + a = 60) :
  sum_of_three_numbers a b c = 71.5 :=
by
  unfold sum_of_three_numbers
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_is_71_point_5_l173_17355


namespace NUMINAMATH_GPT_number_of_schools_l173_17319

theorem number_of_schools (cost_per_school : ℝ) (population : ℝ) (savings_per_day_per_person : ℝ) (days_in_year : ℕ) :
  cost_per_school = 5 * 10^5 →
  population = 1.3 * 10^9 →
  savings_per_day_per_person = 0.01 →
  days_in_year = 365 →
  (population * savings_per_day_per_person * days_in_year) / cost_per_school = 9.49 * 10^3 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_number_of_schools_l173_17319


namespace NUMINAMATH_GPT_kim_min_pours_l173_17362

-- Define the initial conditions
def initial_volume (V : ℝ) : ℝ := V
def pour (V : ℝ) : ℝ := 0.9 * V

-- Define the remaining volume after n pours
def remaining_volume (V : ℝ) (n : ℕ) : ℝ := V * (0.9)^n

-- State the problem: After 7 pours, the remaining volume is less than half the initial volume
theorem kim_min_pours (V : ℝ) (hV : V > 0) : remaining_volume V 7 < V / 2 :=
by
  -- Because the proof is not required, we use sorry
  sorry

end NUMINAMATH_GPT_kim_min_pours_l173_17362


namespace NUMINAMATH_GPT_choir_average_age_l173_17331

theorem choir_average_age
  (avg_females_age : ℕ)
  (num_females : ℕ)
  (avg_males_age : ℕ)
  (num_males : ℕ)
  (females_avg_condition : avg_females_age = 28)
  (females_num_condition : num_females = 8)
  (males_avg_condition : avg_males_age = 32)
  (males_num_condition : num_males = 17) :
  ((avg_females_age * num_females + avg_males_age * num_males) / (num_females + num_males) = 768 / 25) :=
by
  sorry

end NUMINAMATH_GPT_choir_average_age_l173_17331


namespace NUMINAMATH_GPT_m_value_l173_17335

theorem m_value (m : ℝ) (h : (243:ℝ) ^ (1/3) = 3 ^ m) : m = 5 / 3 :=
sorry

end NUMINAMATH_GPT_m_value_l173_17335


namespace NUMINAMATH_GPT_find_m_direct_proportion_l173_17378

theorem find_m_direct_proportion (m : ℝ) (h1 : m + 2 ≠ 0) (h2 : |m| - 1 = 1) : m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_direct_proportion_l173_17378


namespace NUMINAMATH_GPT_complete_the_square_l173_17396

theorem complete_the_square (x : ℝ) : 
  (x^2 - 8 * x + 10 = 0) → 
  ((x - 4)^2 = 6) :=
sorry

end NUMINAMATH_GPT_complete_the_square_l173_17396


namespace NUMINAMATH_GPT_simplify_abs_neg_pow_sub_l173_17373

theorem simplify_abs_neg_pow_sub (a b : ℤ) (h : a = 4) (h' : b = 6) : 
  (|-(a ^ 2) - b| = 22) := 
by
  sorry

end NUMINAMATH_GPT_simplify_abs_neg_pow_sub_l173_17373


namespace NUMINAMATH_GPT_part1a_part1b_part2_part3a_part3b_l173_17322

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
noncomputable def g (x : ℝ) : ℝ := x^2 + 2

-- Prove f(2) = 1/3
theorem part1a : f 2 = 1 / 3 := 
by sorry

-- Prove g(2) = 6
theorem part1b : g 2 = 6 :=
by sorry

-- Prove f[g(2)] = 1/7 
theorem part2 : f (g 2) = 1 / 7 :=
by sorry

-- Prove f[g(x)] = 1/(x^2 + 3) 
theorem part3a : ∀ x : ℝ, f (g x) = 1 / (x^2 + 3) :=
by sorry

-- Prove g[f(x)] = 1/((1 + x)^2) + 2 
theorem part3b : ∀ x : ℝ, g (f x) = 1 / (1 + x)^2 + 2 :=
by sorry

end NUMINAMATH_GPT_part1a_part1b_part2_part3a_part3b_l173_17322


namespace NUMINAMATH_GPT_reciprocal_of_neg_one_div_2023_l173_17343

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end NUMINAMATH_GPT_reciprocal_of_neg_one_div_2023_l173_17343


namespace NUMINAMATH_GPT_average_age_of_community_l173_17312

theorem average_age_of_community 
  (k : ℕ) 
  (hwomen : ℕ := 7 * k)
  (hmen : ℕ := 5 * k)
  (avg_age_women : ℝ := 30)
  (avg_age_men : ℝ := 35)
  (total_women_age : ℝ := avg_age_women * hwomen)
  (total_men_age : ℝ := avg_age_men * hmen)
  (total_population : ℕ := hwomen + hmen)
  (total_age : ℝ := total_women_age + total_men_age) : 
  total_age / total_population = 32 + 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_community_l173_17312


namespace NUMINAMATH_GPT_patio_length_l173_17326

theorem patio_length (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 := 
by 
  sorry

end NUMINAMATH_GPT_patio_length_l173_17326


namespace NUMINAMATH_GPT_odd_multiple_of_9_implies_multiple_of_3_l173_17325

-- Define an odd number that is a multiple of 9
def odd_multiple_of_nine (m : ℤ) : Prop := 9 * m % 2 = 1

-- Define multiples of 3 and 9
def multiple_of_three (n : ℤ) : Prop := ∃ k : ℤ, n = 3 * k
def multiple_of_nine (n : ℤ) : Prop := ∃ k : ℤ, n = 9 * k

-- The main statement
theorem odd_multiple_of_9_implies_multiple_of_3 (n : ℤ) 
  (h1 : ∀ n, multiple_of_nine n → multiple_of_three n)
  (h2 : odd_multiple_of_nine n ∧ multiple_of_nine n) : 
  multiple_of_three n :=
by sorry

end NUMINAMATH_GPT_odd_multiple_of_9_implies_multiple_of_3_l173_17325


namespace NUMINAMATH_GPT_find_speed_of_car_y_l173_17301

noncomputable def average_speed_of_car_y (sₓ : ℝ) (delay : ℝ) (d_afterₓ_started : ℝ) : ℝ :=
  let tₓ_before := delay
  let dₓ_before := sₓ * tₓ_before
  let total_dₓ := dₓ_before + d_afterₓ_started
  let tₓ_after := d_afterₓ_started / sₓ
  let total_time_y := tₓ_after
  d_afterₓ_started / total_time_y

theorem find_speed_of_car_y (h₁ : ∀ t, t = 1.2) (h₂ : ∀ sₓ, sₓ = 35) (h₃ : ∀ d_afterₓ_started, d_afterₓ_started = 42) : 
  average_speed_of_car_y 35 1.2 42 = 35 := by
  unfold average_speed_of_car_y
  simp
  sorry

end NUMINAMATH_GPT_find_speed_of_car_y_l173_17301


namespace NUMINAMATH_GPT_train_crossing_time_l173_17361

theorem train_crossing_time 
    (length : ℝ) (speed_kmph : ℝ) 
    (conversion_factor: ℝ) (speed_mps: ℝ) 
    (time : ℝ) :
  length = 400 ∧ speed_kmph = 144 ∧ conversion_factor = 1000 / 3600 ∧ speed_mps = speed_kmph * conversion_factor ∧ time = length / speed_mps → time = 10 := 
by 
  sorry

end NUMINAMATH_GPT_train_crossing_time_l173_17361


namespace NUMINAMATH_GPT_range_of_a_l173_17359

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 < x ∧ x < -1 → (a + x) * (1 + x) < 0) → a > 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l173_17359


namespace NUMINAMATH_GPT_ram_total_distance_l173_17329

noncomputable def total_distance 
  (speed1 speed2 time1 total_time : ℝ) 
  (h_speed1 : speed1 = 20) 
  (h_speed2 : speed2 = 70)
  (h_time1 : time1 = 3.2)
  (h_total_time : total_time = 8) 
  : ℝ := 
  speed1 * time1 + speed2 * (total_time - time1)

theorem ram_total_distance
  (speed1 speed2 time1 total_time : ℝ)
  (h_speed1 : speed1 = 20)
  (h_speed2 : speed2 = 70)
  (h_time1 : time1 = 3.2)
  (h_total_time : total_time = 8)
  : total_distance speed1 speed2 time1 total_time h_speed1 h_speed2 h_time1 h_total_time = 400 :=
  sorry

end NUMINAMATH_GPT_ram_total_distance_l173_17329


namespace NUMINAMATH_GPT_four_mutually_acquainted_l173_17363

theorem four_mutually_acquainted (G : SimpleGraph (Fin 9)) 
  (h : ∀ (s : Finset (Fin 9)), s.card = 3 → ∃ (u v : Fin 9), u ∈ s ∧ v ∈ s ∧ G.Adj u v) :
  ∃ (s : Finset (Fin 9)), s.card = 4 ∧ ∀ (u v : Fin 9), u ∈ s → v ∈ s → G.Adj u v :=
by
  sorry

end NUMINAMATH_GPT_four_mutually_acquainted_l173_17363


namespace NUMINAMATH_GPT_triangle_inequality_l173_17300

theorem triangle_inequality
  (R r p : ℝ) (a b c : ℝ)
  (h1 : a * b + b * c + c * a = r^2 + p^2 + 4 * R * r)
  (h2 : 16 * R * r - 5 * r^2 ≤ p^2)
  (h3 : p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2):
  20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 4 * (R + r)^2 := 
  by
    sorry

end NUMINAMATH_GPT_triangle_inequality_l173_17300


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l173_17398

variable (α : Real)

theorem trigonometric_identity_proof (h1 : Real.tan α = 4 / 3) (h2 : 0 < α ∧ α < Real.pi / 2) :
  Real.sin (Real.pi + α) + Real.cos (Real.pi - α) = -7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l173_17398


namespace NUMINAMATH_GPT_correct_exponent_calculation_l173_17339

theorem correct_exponent_calculation : 
(∀ (a b : ℝ), (a + b)^2 ≠ a^2 + b^2) ∧
(∀ (a : ℝ), a^9 / a^3 ≠ a^3) ∧
(∀ (a b : ℝ), (ab)^3 = a^3 * b^3) ∧
(∀ (a : ℝ), (a^5)^2 ≠ a^7) :=
by 
  sorry

end NUMINAMATH_GPT_correct_exponent_calculation_l173_17339


namespace NUMINAMATH_GPT_circles_intersect_condition_l173_17385

theorem circles_intersect_condition (a : ℝ) (ha : a > 0) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + y^2 = 16) ↔ 3 < a ∧ a < 5 :=
by sorry

end NUMINAMATH_GPT_circles_intersect_condition_l173_17385


namespace NUMINAMATH_GPT_diff_squares_of_roots_l173_17342

theorem diff_squares_of_roots : ∀ α β : ℝ, (α * β = 6) ∧ (α + β = 5) -> (α - β)^2 = 1 := by
  sorry

end NUMINAMATH_GPT_diff_squares_of_roots_l173_17342


namespace NUMINAMATH_GPT_inequality_solution_set_nonempty_range_l173_17349

theorem inequality_solution_set_nonempty_range (a : ℝ) :
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) ↔ (a ≤ -2 ∨ a ≥ 6 / 5) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_inequality_solution_set_nonempty_range_l173_17349


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l173_17370

-- Let {a_n} be an arithmetic sequence.
-- Define Sn as the sum of the first n terms.
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n-1))) / 2

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_condition : 2 * a 6 = a 7 + 5) :
  S a 11 = 55 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l173_17370


namespace NUMINAMATH_GPT_initial_nickels_l173_17381

theorem initial_nickels (quarters : ℕ) (initial_nickels : ℕ) (borrowed_nickels : ℕ) (current_nickels : ℕ) 
  (H1 : initial_nickels = 87) (H2 : borrowed_nickels = 75) (H3 : current_nickels = 12) : 
  initial_nickels = current_nickels + borrowed_nickels := 
by 
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_initial_nickels_l173_17381


namespace NUMINAMATH_GPT_isosceles_triangle_CBD_supplement_l173_17377

/-- Given an isosceles triangle ABC with AC = BC and angle C = 50 degrees,
    and point D such that angle CBD is supplementary to angle ABC,
    prove that angle CBD is 115 degrees. -/
theorem isosceles_triangle_CBD_supplement 
  (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (AC BC : ℝ) (angleBAC angleABC angleC angleCBD : ℝ)
  (isosceles : AC = BC)
  (angle_C_eq : angleC = 50)
  (supplement : angleCBD = 180 - angleABC) :
  angleCBD = 115 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_CBD_supplement_l173_17377


namespace NUMINAMATH_GPT_problem_proof_l173_17375

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end NUMINAMATH_GPT_problem_proof_l173_17375


namespace NUMINAMATH_GPT_star_4_3_l173_17353

def star (a b : ℤ) : ℤ := a^2 - a * b + b^2

theorem star_4_3 : star 4 3 = 13 :=
by
  sorry

end NUMINAMATH_GPT_star_4_3_l173_17353


namespace NUMINAMATH_GPT_profit_rate_is_five_percent_l173_17379

theorem profit_rate_is_five_percent (cost_price selling_price : ℝ) (hx : 1.1 * cost_price - 10 = 210) : 
  (selling_price = 1.1 * cost_price) → 
  (selling_price - cost_price) / cost_price * 100 = 5 :=
by
  sorry

end NUMINAMATH_GPT_profit_rate_is_five_percent_l173_17379


namespace NUMINAMATH_GPT_rectangle_perimeter_l173_17306

theorem rectangle_perimeter (a b c width : ℕ) (area : ℕ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) 
  (h5 : area = (a * b) / 2) 
  (h6 : width = 5) 
  (h7 : area = width * ((area * 2) / (a * b)))
  : 2 * (width + (area / width)) = 22 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l173_17306


namespace NUMINAMATH_GPT_y_paisa_for_each_rupee_x_l173_17338

theorem y_paisa_for_each_rupee_x (p : ℕ) (x : ℕ) (y_share total_amount : ℕ) 
  (h₁ : y_share = 2700) 
  (h₂ : total_amount = 10500) 
  (p_condition : (130 + p) * x = total_amount) 
  (y_condition : p * x = y_share) : 
  p = 45 := 
by
  sorry

end NUMINAMATH_GPT_y_paisa_for_each_rupee_x_l173_17338


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l173_17394

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 3) : 
  ((x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4))) = (5 : ℝ) / 3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l173_17394


namespace NUMINAMATH_GPT_sugar_amount_first_week_l173_17389

theorem sugar_amount_first_week (s : ℕ → ℕ) (h : s 4 = 3) (h_rec : ∀ n, s (n + 1) = s n / 2) : s 1 = 24 :=
by
  sorry

end NUMINAMATH_GPT_sugar_amount_first_week_l173_17389


namespace NUMINAMATH_GPT_even_numbers_average_19_l173_17351

theorem even_numbers_average_19 (n : ℕ) (h1 : (n / 2) * (2 + 2 * n) / n = 19) : n = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_even_numbers_average_19_l173_17351


namespace NUMINAMATH_GPT_sides_of_second_polygon_l173_17311

theorem sides_of_second_polygon (s : ℝ) (n : ℕ) 
  (perimeter1_is_perimeter2 : 38 * (2 * s) = n * s) : 
  n = 76 := by
  sorry

end NUMINAMATH_GPT_sides_of_second_polygon_l173_17311


namespace NUMINAMATH_GPT_pizza_slices_left_over_l173_17313

def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8
def small_pizzas_purchased : ℕ := 3
def large_pizzas_purchased : ℕ := 2
def george_slices : ℕ := 3
def bob_slices : ℕ := george_slices + 1
def susie_slices : ℕ := bob_slices / 2
def bill_slices : ℕ := 3
def fred_slices : ℕ := 3
def mark_slices : ℕ := 3

def total_pizza_slices : ℕ := (small_pizzas_purchased * small_pizza_slices) + (large_pizzas_purchased * large_pizza_slices)
def total_slices_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

theorem pizza_slices_left_over : total_pizza_slices - total_slices_eaten = 10 :=
by sorry

end NUMINAMATH_GPT_pizza_slices_left_over_l173_17313


namespace NUMINAMATH_GPT_calculation_correct_l173_17346

theorem calculation_correct : 2 * (3 ^ 2) ^ 4 = 13122 := by
  sorry

end NUMINAMATH_GPT_calculation_correct_l173_17346


namespace NUMINAMATH_GPT_correct_parameterizations_of_line_l173_17384

theorem correct_parameterizations_of_line :
  ∀ (t : ℝ),
    (∀ (x y : ℝ), ((x = 5/3) ∧ (y = 0) ∨ (x = 0) ∧ (y = -5) ∨ (x = -5/3) ∧ (y = 0) ∨ 
                   (x = 1) ∧ (y = -2) ∨ (x = -2) ∧ (y = -11)) → 
                   y = 3 * x - 5) ∧
    (∀ (a b : ℝ), ((a = 1) ∧ (b = 3) ∨ (a = 3) ∧ (b = 1) ∨ (a = -1) ∧ (b = -3) ∨
                   (a = 1/3) ∧ (b = 1)) → 
                   b = 3 * a) →
    -- Check only Options D and E
    ((x = 1) → (y = -2) → (a = 1/3) → (b = 1) → y = 3 * x - 5 ∧ b = 3 * a) ∨
    ((x = -2) → (y = -11) → (a = 1/3) → (b = 1) → y = 3 * x - 5 ∧ b = 3 * a) :=
by
  sorry

end NUMINAMATH_GPT_correct_parameterizations_of_line_l173_17384


namespace NUMINAMATH_GPT_a3_pm_2b3_not_div_by_37_l173_17358

theorem a3_pm_2b3_not_div_by_37 {a b : ℤ} (ha : ¬ (37 ∣ a)) (hb : ¬ (37 ∣ b)) :
  ¬ (37 ∣ (a^3 + 2 * b^3)) ∧ ¬ (37 ∣ (a^3 - 2 * b^3)) :=
  sorry

end NUMINAMATH_GPT_a3_pm_2b3_not_div_by_37_l173_17358


namespace NUMINAMATH_GPT_Felicity_family_store_visits_l173_17371

theorem Felicity_family_store_visits
  (lollipop_stick : ℕ := 1)
  (fort_total_sticks : ℕ := 400)
  (fort_completion_percent : ℕ := 60)
  (weeks_collected : ℕ := 80)
  (sticks_collected : ℕ := (fort_total_sticks * fort_completion_percent) / 100)
  (store_visits_per_week : ℕ := sticks_collected / weeks_collected) :
  store_visits_per_week = 3 := by
  sorry

end NUMINAMATH_GPT_Felicity_family_store_visits_l173_17371


namespace NUMINAMATH_GPT_collinear_condition_l173_17304

theorem collinear_condition {a b c d : ℝ} (h₁ : a < b) (h₂ : c < d) (h₃ : a < d) (h₄ : c < b) :
  (a / d) + (c / b) = 1 := 
sorry

end NUMINAMATH_GPT_collinear_condition_l173_17304


namespace NUMINAMATH_GPT_tory_sells_grandmother_l173_17380

theorem tory_sells_grandmother (G : ℕ)
    (total_goal : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ)
    (h_goal : total_goal = 50) (h_sold_to_uncle : sold_to_uncle = 7)
    (h_sold_to_neighbor : sold_to_neighbor = 5) (h_remaining_to_sell : remaining_to_sell = 26) :
    (G + sold_to_uncle + sold_to_neighbor + remaining_to_sell = total_goal) → G = 12 :=
by
    intros h
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_tory_sells_grandmother_l173_17380


namespace NUMINAMATH_GPT_x_squared_inverse_y_fourth_l173_17340

theorem x_squared_inverse_y_fourth (x y : ℝ) (k : ℝ) (h₁ : x = 8) (h₂ : y = 2) (h₃ : (x^2) * (y^4) = k) : x^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_x_squared_inverse_y_fourth_l173_17340


namespace NUMINAMATH_GPT_coordinate_equation_solution_l173_17382

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end NUMINAMATH_GPT_coordinate_equation_solution_l173_17382


namespace NUMINAMATH_GPT_card_probability_l173_17386

-- Definitions of the conditions
def is_multiple (n d : ℕ) : Prop := d ∣ n

def count_multiples (d m : ℕ) : ℕ := (m / d)

def multiples_in_range (n : ℕ) : ℕ := 
  count_multiples 2 n + count_multiples 3 n + count_multiples 5 n
  - count_multiples 6 n - count_multiples 10 n - count_multiples 15 n 
  + count_multiples 30 n

def probability_of_multiples_in_range (n : ℕ) : ℚ := 
  multiples_in_range n / n 

-- Proof statement
theorem card_probability (n : ℕ) (h : n = 120) : probability_of_multiples_in_range n = 11 / 15 :=
  sorry

end NUMINAMATH_GPT_card_probability_l173_17386


namespace NUMINAMATH_GPT_inequality_does_not_hold_l173_17308

noncomputable def f : ℝ → ℝ := sorry -- define f satisfying the conditions from a)

theorem inequality_does_not_hold :
  (∀ x, f (-x) = f x) ∧ -- f is even
  (∀ x, f x = f (x + 2)) ∧ -- f is periodic with period 2
  (∀ x, 3 ≤ x ∧ x ≤ 4 → f x = 2^x) → -- f(x) = 2^x when x is in [3, 4]
  ¬ (f (Real.sin 3) < f (Real.cos 3)) := by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_inequality_does_not_hold_l173_17308


namespace NUMINAMATH_GPT_totalPeoplePresent_is_630_l173_17310

def totalParents : ℕ := 105
def totalPupils : ℕ := 698

def groupA_fraction : ℚ := 30 / 100
def groupB_fraction : ℚ := 25 / 100
def groupC_fraction : ℚ := 20 / 100
def groupD_fraction : ℚ := 15 / 100
def groupE_fraction : ℚ := 10 / 100

def groupA_attendance : ℚ := 90 / 100
def groupB_attendance : ℚ := 80 / 100
def groupC_attendance : ℚ := 70 / 100
def groupD_attendance : ℚ := 60 / 100
def groupE_attendance : ℚ := 50 / 100

def junior_fraction : ℚ := 30 / 100
def intermediate_fraction : ℚ := 35 / 100
def senior_fraction : ℚ := 20 / 100
def advanced_fraction : ℚ := 15 / 100

def junior_attendance : ℚ := 85 / 100
def intermediate_attendance : ℚ := 80 / 100
def senior_attendance : ℚ := 75 / 100
def advanced_attendance : ℚ := 70 / 100

noncomputable def totalPeoplePresent : ℚ := 
  totalParents * groupA_fraction * groupA_attendance +
  totalParents * groupB_fraction * groupB_attendance +
  totalParents * groupC_fraction * groupC_attendance +
  totalParents * groupD_fraction * groupD_attendance +
  totalParents * groupE_fraction * groupE_attendance +
  totalPupils * junior_fraction * junior_attendance +
  totalPupils * intermediate_fraction * intermediate_attendance +
  totalPupils * senior_fraction * senior_attendance +
  totalPupils * advanced_fraction * advanced_attendance

theorem totalPeoplePresent_is_630 : totalPeoplePresent.floor = 630 := 
by 
  sorry -- no proof required as per the instructions

end NUMINAMATH_GPT_totalPeoplePresent_is_630_l173_17310


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l173_17336

theorem sum_of_roots_of_quadratic :
  ∀ x : ℝ, (x - 1) * (x + 4) = 18 -> (∃ a b c : ℝ, a = 1 ∧ b = 3 ∧ c = -22 ∧ ((a * x^2 + b * x + c = 0) ∧ (-b / a = -3))) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l173_17336


namespace NUMINAMATH_GPT_find_b_l173_17366

theorem find_b (b : ℚ) (h : ∃ c : ℚ, (3 * x + c)^2 = 9 * x^2 + 27 * x + b) : b = 81 / 4 := 
sorry

end NUMINAMATH_GPT_find_b_l173_17366


namespace NUMINAMATH_GPT_part1_part2_l173_17357

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

end NUMINAMATH_GPT_part1_part2_l173_17357


namespace NUMINAMATH_GPT_trapezoids_not_necessarily_congruent_l173_17352

-- Define trapezoid structure
structure Trapezoid (α : Type) [LinearOrderedField α] :=
(base1 base2 side1 side2 diag1 diag2 : α) -- sides and diagonals
(angle1 angle2 angle3 angle4 : α)        -- internal angles

-- Conditions about given trapezoids
variables {α : Type} [LinearOrderedField α]
variables (T1 T2 : Trapezoid α)

-- The condition that corresponding angles of the trapezoids are equal
def equal_angles := 
  T1.angle1 = T2.angle1 ∧ T1.angle2 = T2.angle2 ∧ 
  T1.angle3 = T2.angle3 ∧ T1.angle4 = T2.angle4

-- The condition that diagonals of the trapezoids are equal
def equal_diagonals := 
  T1.diag1 = T2.diag1 ∧ T1.diag2 = T2.diag2

-- The statement to prove
theorem trapezoids_not_necessarily_congruent :
  equal_angles T1 T2 ∧ equal_diagonals T1 T2 → ¬ (T1 = T2) := by
  sorry

end NUMINAMATH_GPT_trapezoids_not_necessarily_congruent_l173_17352


namespace NUMINAMATH_GPT_evaluate_expression_l173_17314

-- Define the mathematical expressions using Lean's constructs
def expr1 : ℕ := 201 * 5 + 1220 - 2 * 3 * 5 * 7

-- State the theorem we aim to prove
theorem evaluate_expression : expr1 = 2015 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l173_17314


namespace NUMINAMATH_GPT_complex_multiplication_l173_17348

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i := by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l173_17348


namespace NUMINAMATH_GPT_bags_total_weight_l173_17323

noncomputable def total_weight_of_bags (x y z : ℕ) : ℕ := x + y + z

theorem bags_total_weight (x y z : ℕ) (h1 : x + y = 90) (h2 : y + z = 100) (h3 : z + x = 110) :
  total_weight_of_bags x y z = 150 :=
by
  sorry

end NUMINAMATH_GPT_bags_total_weight_l173_17323
