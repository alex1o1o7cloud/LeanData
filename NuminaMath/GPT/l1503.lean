import Mathlib

namespace NUMINAMATH_GPT_solve_for_x_l1503_150370

theorem solve_for_x : ∃ x : ℝ, (x + 36) / 3 = (7 - 2 * x) / 6 ∧ x = -65 / 4 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1503_150370


namespace NUMINAMATH_GPT_min_value_of_polynomial_l1503_150344

theorem min_value_of_polynomial :
  ∃ x : ℝ, ∀ y, y = (x - 16) * (x - 14) * (x + 14) * (x + 16) → y ≥ -900 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_polynomial_l1503_150344


namespace NUMINAMATH_GPT_jose_investment_proof_l1503_150322

noncomputable def jose_investment (total_profit jose_share : ℕ) (tom_investment : ℕ) (months_tom months_jose : ℕ) : ℕ :=
  let tom_share := total_profit - jose_share
  let tom_investment_mr := tom_investment * months_tom
  let ratio := tom_share * months_jose
  tom_investment_mr * jose_share / ratio

theorem jose_investment_proof : 
  ∃ (jose_invested : ℕ), 
    let total_profit := 5400
    let jose_share := 3000
    let tom_invested := 3000
    let months_tom := 12
    let months_jose := 10
    jose_investment total_profit jose_share tom_invested months_tom months_jose = 4500 :=
by
  use 4500
  sorry

end NUMINAMATH_GPT_jose_investment_proof_l1503_150322


namespace NUMINAMATH_GPT_jacob_current_age_l1503_150306

theorem jacob_current_age 
  (M : ℕ) 
  (Drew_age : ℕ := M + 5) 
  (Peter_age : ℕ := Drew_age + 4) 
  (John_age : ℕ := 30) 
  (maya_age_eq : 2 * M = John_age) 
  (jacob_future_age : ℕ := Peter_age / 2) 
  (jacob_current_age_eq : ℕ := jacob_future_age - 2) : 
  jacob_current_age_eq = 11 := 
sorry

end NUMINAMATH_GPT_jacob_current_age_l1503_150306


namespace NUMINAMATH_GPT_magicStack_cardCount_l1503_150350

-- Define the conditions and question based on a)
def isMagicStack (n : ℕ) : Prop :=
  let totalCards := 2 * n
  ∃ (A B : Finset ℕ), (A ∪ B = Finset.range totalCards) ∧
    (∀ x ∈ A, x < n) ∧ (∀ x ∈ B, x ≥ n) ∧
    (∀ i ∈ A, i % 2 = 1) ∧ (∀ j ∈ B, j % 2 = 0) ∧
    (151 ∈ A) ∧
    ∃ (newStack : Finset ℕ), (newStack = A ∪ B) ∧
    (∀ k ∈ newStack, k ∈ A ∨ k ∈ B) ∧
    (151 = 151)

-- The theorem that states the number of cards, when card 151 retains its position, is 452.
theorem magicStack_cardCount :
  isMagicStack 226 → 2 * 226 = 452 :=
by
  sorry

end NUMINAMATH_GPT_magicStack_cardCount_l1503_150350


namespace NUMINAMATH_GPT_third_beats_seventh_l1503_150333

-- Definitions and conditions
variable (points : Fin 8 → ℕ)
variable (distinct_points : Function.Injective points)
variable (sum_last_four : points 1 = points 4 + points 5 + points 6 + points 7)

-- Proof statement
theorem third_beats_seventh 
  (h_distinct : ∀ i j, i ≠ j → points i ≠ points j)
  (h_sum : points 1 = points 4 + points 5 + points 6 + points 7) :
  points 2 > points 6 :=
sorry

end NUMINAMATH_GPT_third_beats_seventh_l1503_150333


namespace NUMINAMATH_GPT_value_of_b_l1503_150382

theorem value_of_b (f : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x ≠ 0, f x = -1 / x) (h2 : f a = -1 / 3) (h3 : f (a * b) = 1 / 6) : b = -2 :=
sorry

end NUMINAMATH_GPT_value_of_b_l1503_150382


namespace NUMINAMATH_GPT_leaves_fall_total_l1503_150351

theorem leaves_fall_total : 
  let planned_cherry_trees := 7 
  let planned_maple_trees := 5 
  let actual_cherry_trees := 2 * planned_cherry_trees
  let actual_maple_trees := 3 * planned_maple_trees
  let leaves_per_cherry_tree := 100
  let leaves_per_maple_tree := 150
  actual_cherry_trees * leaves_per_cherry_tree + actual_maple_trees * leaves_per_maple_tree = 3650 :=
by
  let planned_cherry_trees := 7 
  let planned_maple_trees := 5 
  let actual_cherry_trees := 2 * planned_cherry_trees
  let actual_maple_trees := 3 * planned_maple_trees
  let leaves_per_cherry_tree := 100
  let leaves_per_maple_tree := 150
  sorry

end NUMINAMATH_GPT_leaves_fall_total_l1503_150351


namespace NUMINAMATH_GPT_product_is_2008th_power_l1503_150369

theorem product_is_2008th_power (a b c : ℕ) (h1 : a = (b + c) / 2) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : a ≠ b) :
  ∃ k : ℕ, (a * b * c) = k^2008 :=
by
  sorry

end NUMINAMATH_GPT_product_is_2008th_power_l1503_150369


namespace NUMINAMATH_GPT_parallel_vectors_l1503_150332

noncomputable def vector_a : ℝ × ℝ := (2, 1)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors {m : ℝ} (h : (∃ k : ℝ, vector_a = k • vector_b m)) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l1503_150332


namespace NUMINAMATH_GPT_digit_relationship_l1503_150391

theorem digit_relationship (d1 d2 : ℕ) (h1 : d1 * 10 + d2 = 16) (h2 : d1 + d2 = 7) : d2 = 6 * d1 :=
by
  sorry

end NUMINAMATH_GPT_digit_relationship_l1503_150391


namespace NUMINAMATH_GPT_taxi_ride_cost_l1503_150377

theorem taxi_ride_cost :
  let base_fare : ℝ := 2.00
  let cost_per_mile_first_3 : ℝ := 0.30
  let cost_per_mile_additional : ℝ := 0.40
  let total_distance : ℕ := 8
  let first_3_miles_cost : ℝ := base_fare + 3 * cost_per_mile_first_3
  let additional_miles_cost : ℝ := (total_distance - 3) * cost_per_mile_additional
  let total_cost : ℝ := first_3_miles_cost + additional_miles_cost
  total_cost = 4.90 :=
by
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l1503_150377


namespace NUMINAMATH_GPT_painting_perimeter_l1503_150334

-- Definitions for the problem conditions
def frame_thickness : ℕ := 3
def frame_area : ℕ := 108

-- Declaration that expresses the given conditions and the problem's conclusion
theorem painting_perimeter {w h : ℕ} (h_frame : (w + 2 * frame_thickness) * (h + 2 * frame_thickness) - w * h = frame_area) :
  2 * (w + h) = 24 :=
by
  sorry

end NUMINAMATH_GPT_painting_perimeter_l1503_150334


namespace NUMINAMATH_GPT_product_of_first_four_consecutive_primes_l1503_150378

theorem product_of_first_four_consecutive_primes : 
  (2 * 3 * 5 * 7) = 210 :=
by
  sorry

end NUMINAMATH_GPT_product_of_first_four_consecutive_primes_l1503_150378


namespace NUMINAMATH_GPT_cubic_polynomial_solution_l1503_150386

theorem cubic_polynomial_solution (x : ℝ) :
  x^3 + 6*x^2 + 11*x + 6 = 12 ↔ x = -1 ∨ x = -2 ∨ x = -3 := by
  sorry

end NUMINAMATH_GPT_cubic_polynomial_solution_l1503_150386


namespace NUMINAMATH_GPT_solve_fraction_eqn_l1503_150363

def fraction_eqn_solution : Prop :=
  ∃ (x : ℝ), (x + 2) / (x - 1) = 0 ∧ x ≠ 1 ∧ x = -2

theorem solve_fraction_eqn : fraction_eqn_solution :=
sorry

end NUMINAMATH_GPT_solve_fraction_eqn_l1503_150363


namespace NUMINAMATH_GPT_gcd_three_numbers_l1503_150346

theorem gcd_three_numbers :
  gcd (gcd 324 243) 135 = 27 :=
by
  sorry

end NUMINAMATH_GPT_gcd_three_numbers_l1503_150346


namespace NUMINAMATH_GPT_moses_more_than_esther_l1503_150395

theorem moses_more_than_esther (total_amount: ℝ) (moses_share: ℝ) (tony_esther_share: ℝ) :
  total_amount = 50 → moses_share = 0.40 * total_amount → 
  tony_esther_share = (total_amount - moses_share) / 2 → 
  moses_share - tony_esther_share = 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_moses_more_than_esther_l1503_150395


namespace NUMINAMATH_GPT_intersection_M_N_is_valid_l1503_150367

-- Define the conditions given in the problem
def M := {x : ℝ |  3 / 4 < x ∧ x ≤ 1}
def N := {y : ℝ | 0 ≤ y}

-- State the theorem that needs to be proved
theorem intersection_M_N_is_valid : M ∩ N = {x : ℝ | 3 / 4 < x ∧ x ≤ 1} :=
by 
  sorry

end NUMINAMATH_GPT_intersection_M_N_is_valid_l1503_150367


namespace NUMINAMATH_GPT_exists_unique_pair_l1503_150355

theorem exists_unique_pair (X : Set ℤ) :
  (∀ n : ℤ, ∃! (a b : ℤ), a ∈ X ∧ b ∈ X ∧ a + 2 * b = n) :=
sorry

end NUMINAMATH_GPT_exists_unique_pair_l1503_150355


namespace NUMINAMATH_GPT_regular_polygon_sides_l1503_150387

theorem regular_polygon_sides (n : ℕ) (h : ∀ (polygon : ℕ), (polygon = 160) → 2 < polygon ∧ (180 * (polygon - 2) / polygon) = 160) : n = 18 := 
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1503_150387


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l1503_150398

theorem common_ratio_of_geometric_series (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) :
  S = a / (1 - r) → r = 21 / 25 :=
by
  intros h₃
  rw [h₁, h₂] at h₃
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l1503_150398


namespace NUMINAMATH_GPT_roots_quadratic_expression_l1503_150396

theorem roots_quadratic_expression :
  ∀ (a b : ℝ), (a^2 - 5 * a + 6 = 0) ∧ (b^2 - 5 * b + 6 = 0) → 
  a^3 + a^4 * b^2 + a^2 * b^4 + b^3 + a * b * (a + b) = 533 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_roots_quadratic_expression_l1503_150396


namespace NUMINAMATH_GPT_maximize_profit_l1503_150301

noncomputable def production_problem : Prop :=
  ∃ (x y : ℕ), (3 * x + 2 * y ≤ 1200) ∧ (x + 2 * y ≤ 800) ∧ 
               (30 * x + 40 * y) = 18000 ∧ 
               x = 200 ∧ 
               y = 300

theorem maximize_profit : production_problem :=
sorry

end NUMINAMATH_GPT_maximize_profit_l1503_150301


namespace NUMINAMATH_GPT_peach_pies_l1503_150317

theorem peach_pies (total_pies : ℕ) (apple_ratio blueberry_ratio peach_ratio : ℕ)
  (h_ratio : apple_ratio + blueberry_ratio + peach_ratio = 10)
  (h_total : total_pies = 30)
  (h_ratios : apple_ratio = 3 ∧ blueberry_ratio = 2 ∧ peach_ratio = 5) :
  total_pies / (apple_ratio + blueberry_ratio + peach_ratio) * peach_ratio = 15 :=
by
  sorry

end NUMINAMATH_GPT_peach_pies_l1503_150317


namespace NUMINAMATH_GPT_number_before_star_is_five_l1503_150376

theorem number_before_star_is_five (n : ℕ) (h1 : n % 72 = 0) (h2 : n % 10 = 0) (h3 : ∃ k, n = 400 + 10 * k) : (n / 10) % 10 = 5 :=
sorry

end NUMINAMATH_GPT_number_before_star_is_five_l1503_150376


namespace NUMINAMATH_GPT_range_of_a_plus_b_l1503_150328

theorem range_of_a_plus_b (a b : ℝ) (h : |a| + |b| + |a - 1| + |b - 1| ≤ 2) : 
  0 ≤ a + b ∧ a + b ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_plus_b_l1503_150328


namespace NUMINAMATH_GPT_find_a_l1503_150335

variable {x n : ℝ}

theorem find_a (hx : x > 0) (hn : n > 0) :
    (∀ n > 0, x + n^n / x^n ≥ n + 1) ↔ (∀ n > 0, a = n^n) :=
sorry

end NUMINAMATH_GPT_find_a_l1503_150335


namespace NUMINAMATH_GPT_problem_solution_l1503_150353

theorem problem_solution (m n : ℕ) (h1 : m + 7 < n + 3) 
  (h2 : (m + (m+3) + (m+7) + (n+3) + (n+6) + 2 * n) / 6 = n + 3) 
  (h3 : (m + 7 + n + 3) / 2 = n + 3) : m + n = 12 := 
  sorry

end NUMINAMATH_GPT_problem_solution_l1503_150353


namespace NUMINAMATH_GPT_common_ratio_l1503_150300

def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def arith_seq (a : ℕ → ℝ) (x y z : ℕ) := 2 * a z = a x + a y

theorem common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q) (h_arith : arith_seq a 0 1 2) (h_nonzero : a 0 ≠ 0) : q = 1 ∨ q = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_l1503_150300


namespace NUMINAMATH_GPT_number_of_terms_arithmetic_sequence_l1503_150365

-- Definitions for the arithmetic sequence conditions
open Nat

noncomputable def S4 := 26
noncomputable def Sn := 187
noncomputable def last4_sum (n : ℕ) (a d : ℕ) := 
  (n - 3) * a + 3 * (n - 2) * d + 3 * (n - 1) * d + n * d

-- Statement for the problem
theorem number_of_terms_arithmetic_sequence 
  (a d n : ℕ) (h1 : 4 * a + 6 * d = S4) (h2 : n * (2 * a + (n - 1) * d) / 2 = Sn) 
  (h3 : last4_sum n a d = 110) : 
  n = 11 :=
sorry

end NUMINAMATH_GPT_number_of_terms_arithmetic_sequence_l1503_150365


namespace NUMINAMATH_GPT_exists_polynomial_p_l1503_150345

theorem exists_polynomial_p (x : ℝ) (h : x ∈ Set.Icc (1 / 10 : ℝ) (9 / 10 : ℝ)) :
  ∃ (P : ℝ → ℝ), (∀ (k : ℤ), P k = P k) ∧ (∀ (x : ℝ), x ∈ Set.Icc (1 / 10 : ℝ) (9 / 10 : ℝ) → 
  abs (P x - 1 / 2) < 1 / 1000) :=
by
  sorry

end NUMINAMATH_GPT_exists_polynomial_p_l1503_150345


namespace NUMINAMATH_GPT_mat_weavers_equiv_l1503_150347

theorem mat_weavers_equiv {x : ℕ} 
  (h1 : 4 * 1 = 4) 
  (h2 : 16 * (64 / 16) = 64) 
  (h3 : 1 = 64 / (16 * x)) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_mat_weavers_equiv_l1503_150347


namespace NUMINAMATH_GPT_find_a_l1503_150394

theorem find_a (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) (h1 : ∀ n, S_n n = 3^(n+1) + a)
  (h2 : ∀ n, a_n (n+1) = S_n (n+1) - S_n n)
  (h3 : ∀ n m k, a_n m * a_n k = (a_n n)^2 → n = m + k) : 
  a = -3 := 
sorry

end NUMINAMATH_GPT_find_a_l1503_150394


namespace NUMINAMATH_GPT_slices_with_both_onions_and_olives_l1503_150381

noncomputable def slicesWithBothToppings (total_slices slices_with_onions slices_with_olives : Nat) : Nat :=
  slices_with_onions + slices_with_olives - total_slices

theorem slices_with_both_onions_and_olives 
  (total_slices : Nat) (slices_with_onions : Nat) (slices_with_olives : Nat) :
  total_slices = 18 ∧ slices_with_onions = 10 ∧ slices_with_olives = 10 →
  slicesWithBothToppings total_slices slices_with_onions slices_with_olives = 2 :=
by
  sorry

end NUMINAMATH_GPT_slices_with_both_onions_and_olives_l1503_150381


namespace NUMINAMATH_GPT_exists_X_Y_sum_not_in_third_subset_l1503_150379

open Nat Set

theorem exists_X_Y_sum_not_in_third_subset :
  ∀ (M_1 M_2 M_3 : Set ℕ), 
  Disjoint M_1 M_2 ∧ Disjoint M_2 M_3 ∧ Disjoint M_1 M_3 → 
  ∃ (X Y : ℕ), (X ∈ M_1 ∪ M_2 ∪ M_3) ∧ (Y ∈ M_1 ∪ M_2 ∪ M_3) ∧  
  (X ∈ M_1 → Y ∈ M_2 ∨ Y ∈ M_3) ∧
  (X ∈ M_2 → Y ∈ M_1 ∨ Y ∈ M_3) ∧
  (X ∈ M_3 → Y ∈ M_1 ∨ Y ∈ M_2) ∧
  (X + Y ∉ M_3) :=
by
  intros M_1 M_2 M_3 disj
  sorry

end NUMINAMATH_GPT_exists_X_Y_sum_not_in_third_subset_l1503_150379


namespace NUMINAMATH_GPT_polyhedron_value_calculation_l1503_150348

noncomputable def calculate_value (P T V : ℕ) : ℕ :=
  100 * P + 10 * T + V

theorem polyhedron_value_calculation :
  ∀ (P T V E F : ℕ),
    F = 36 ∧
    T + P = 36 ∧
    E = (3 * T + 5 * P) / 2 ∧
    V = E - F + 2 →
    calculate_value P T V = 2018 :=
by
  intros P T V E F h
  sorry

end NUMINAMATH_GPT_polyhedron_value_calculation_l1503_150348


namespace NUMINAMATH_GPT_unoccupied_seats_l1503_150364

theorem unoccupied_seats (rows chairs_per_row seats_taken : Nat) (h1 : rows = 40)
  (h2 : chairs_per_row = 20) (h3 : seats_taken = 790) :
  rows * chairs_per_row - seats_taken = 10 :=
by
  sorry

end NUMINAMATH_GPT_unoccupied_seats_l1503_150364


namespace NUMINAMATH_GPT_transylvanian_is_sane_human_l1503_150307

def Transylvanian : Type := sorry -- Placeholder type for Transylvanian
def Human : Transylvanian → Prop := sorry
def Sane : Transylvanian → Prop := sorry
def InsaneVampire : Transylvanian → Prop := sorry

/-- The Transylvanian stated: "Either I am a human, or I am sane." -/
axiom statement (T : Transylvanian) : Human T ∨ Sane T

/-- Insane vampires only make true statements. -/
axiom insane_vampire_truth (T : Transylvanian) : InsaneVampire T → (Human T ∨ Sane T)

/-- Insane vampires cannot be sane or human. -/
axiom insane_vampire_condition (T : Transylvanian) : InsaneVampire T → ¬ Human T ∧ ¬ Sane T

theorem transylvanian_is_sane_human (T : Transylvanian) :
  ¬ (InsaneVampire T) → (Human T ∧ Sane T) := sorry

end NUMINAMATH_GPT_transylvanian_is_sane_human_l1503_150307


namespace NUMINAMATH_GPT_money_constraints_l1503_150352

variable (a b : ℝ)

theorem money_constraints (h1 : 8 * a - b = 98) (h2 : 2 * a + b > 36) : a > 13.4 ∧ b > 9.2 :=
sorry

end NUMINAMATH_GPT_money_constraints_l1503_150352


namespace NUMINAMATH_GPT_B_work_rate_l1503_150340

theorem B_work_rate (B : ℕ) (A_rate C_rate : ℚ) 
  (A_work : A_rate = 1 / 6)
  (C_work : C_rate = 1 / 8 * (1 / 6 + 1 / B))
  (combined_work : 1 / 6 + 1 / B + C_rate = 1 / 3) : 
  B = 28 :=
by 
  sorry

end NUMINAMATH_GPT_B_work_rate_l1503_150340


namespace NUMINAMATH_GPT_find_point_A_l1503_150330

theorem find_point_A (x : ℝ) (h : x + 7 - 4 = 0) : x = -3 :=
sorry

end NUMINAMATH_GPT_find_point_A_l1503_150330


namespace NUMINAMATH_GPT_detail_understanding_word_meaning_guessing_logical_reasoning_l1503_150339

-- Detail Understanding Question
theorem detail_understanding (sentence: String) (s: ∀ x : String, x ∈ ["He hardly watered his new trees,..."] → x = sentence) :
  sentence = "He hardly watered his new trees,..." :=
sorry

-- Word Meaning Guessing Question
theorem word_meaning_guessing (adversity_meaning: String) (meanings: ∀ y : String, y ∈ ["adversity means misfortune or disaster", "lack of water", "sufficient care/attention", "bad weather"] → y = adversity_meaning) :
  adversity_meaning = "adversity means misfortune or disaster" :=
sorry

-- Logical Reasoning Question
theorem logical_reasoning (hope: String) (sentences: ∀ z : String, z ∈ ["The author hopes his sons can withstand the tests of wind and rain in their life journey"] → z = hope) :
  hope = "The author hopes his sons can withstand the tests of wind and rain in their life journey" :=
sorry

end NUMINAMATH_GPT_detail_understanding_word_meaning_guessing_logical_reasoning_l1503_150339


namespace NUMINAMATH_GPT_new_mean_after_adding_14_to_each_of_15_numbers_l1503_150310

theorem new_mean_after_adding_14_to_each_of_15_numbers (avg : ℕ) (n : ℕ) (n_sum : ℕ) (new_sum : ℕ) :
  avg = 40 →
  n = 15 →
  n_sum = n * avg →
  new_sum = n_sum + n * 14 →
  new_sum / n = 54 :=
by
  intros h_avg h_n h_n_sum h_new_sum
  sorry

end NUMINAMATH_GPT_new_mean_after_adding_14_to_each_of_15_numbers_l1503_150310


namespace NUMINAMATH_GPT_canteen_distance_l1503_150316

-- Given definitions
def G_to_road : ℝ := 450
def G_to_B : ℝ := 700

-- Proof statement
theorem canteen_distance :
  ∃ x : ℝ, (x ≠ 0) ∧ 
           (G_to_road^2 + (x - G_to_road)^2 = x^2) ∧ 
           (x = 538) := 
by {
  sorry
}

end NUMINAMATH_GPT_canteen_distance_l1503_150316


namespace NUMINAMATH_GPT_value_to_be_subtracted_l1503_150343

theorem value_to_be_subtracted (N x : ℕ) (h1 : (N - x) / 7 = 7) (h2 : (N - 24) / 10 = 3) : x = 5 := by
  sorry

end NUMINAMATH_GPT_value_to_be_subtracted_l1503_150343


namespace NUMINAMATH_GPT_max_black_cells_in_101x101_grid_l1503_150349

theorem max_black_cells_in_101x101_grid :
  ∀ (k : ℕ), k ≤ 101 → 2 * k * (101 - k) ≤ 5100 :=
by
  sorry

end NUMINAMATH_GPT_max_black_cells_in_101x101_grid_l1503_150349


namespace NUMINAMATH_GPT_sum_of_squares_first_20_l1503_150368

-- Define the sum of squares function
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Specific problem instance
theorem sum_of_squares_first_20 : sum_of_squares 20 = 5740 :=
  by
  -- Proof skipping placeholder
  sorry

end NUMINAMATH_GPT_sum_of_squares_first_20_l1503_150368


namespace NUMINAMATH_GPT_inequality_proof_l1503_150311

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l1503_150311


namespace NUMINAMATH_GPT_illuminated_area_correct_l1503_150360

noncomputable def cube_illuminated_area (a ρ : ℝ) (h₁ : a = 1 / Real.sqrt 2) (h₂ : ρ = Real.sqrt (2 - Real.sqrt 3)) : ℝ :=
  (Real.sqrt 3 - 3 / 2) * (Real.pi + 3)

theorem illuminated_area_correct :
  cube_illuminated_area (1 / Real.sqrt 2) (Real.sqrt (2 - Real.sqrt 3)) (by norm_num) (by norm_num) = (Real.sqrt 3 - 3 / 2) * (Real.pi + 3) :=
sorry

end NUMINAMATH_GPT_illuminated_area_correct_l1503_150360


namespace NUMINAMATH_GPT_sqrt_multiplication_division_l1503_150389

theorem sqrt_multiplication_division :
  Real.sqrt 27 * Real.sqrt (8 / 3) / Real.sqrt (1 / 2) = 18 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_multiplication_division_l1503_150389


namespace NUMINAMATH_GPT_ratio_of_rectangle_sides_l1503_150366

theorem ratio_of_rectangle_sides (x y : ℝ) (h : x < y) 
  (hs : x + y - Real.sqrt (x^2 + y^2) = (1 / 3) * y) : 
  x / y = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_rectangle_sides_l1503_150366


namespace NUMINAMATH_GPT_solve_equation_l1503_150384

theorem solve_equation (x : ℝ) (h : x ≠ 1) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) → x = -4 :=
by
  intro hyp
  sorry

end NUMINAMATH_GPT_solve_equation_l1503_150384


namespace NUMINAMATH_GPT_number_of_children_l1503_150354

-- Definitions for the conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 3
def total_amount : ℕ := 35

-- Theorem stating the proof problem
theorem number_of_children (A C T : ℕ) (hc: A = adult_ticket_cost) (ha: C = child_ticket_cost) (ht: T = total_amount) :
  (T - A) / C = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l1503_150354


namespace NUMINAMATH_GPT_reeya_fourth_subject_score_l1503_150331

theorem reeya_fourth_subject_score (s1 s2 s3 s4 : ℕ) (avg : ℕ) (n : ℕ)
  (h_avg : avg = 75) (h_n : n = 4) (h_s1 : s1 = 65) (h_s2 : s2 = 67) (h_s3 : s3 = 76)
  (h_total_sum : avg * n = s1 + s2 + s3 + s4) : s4 = 92 := by
  sorry

end NUMINAMATH_GPT_reeya_fourth_subject_score_l1503_150331


namespace NUMINAMATH_GPT_range_of_a_l1503_150380

theorem range_of_a (a : ℝ) : (∀ x : ℕ, 4 * x + a ≤ 5 → x ≥ 1 → x ≤ 3) ↔ (-11 < a ∧ a ≤ -7) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1503_150380


namespace NUMINAMATH_GPT_value_of_fraction_l1503_150308

theorem value_of_fraction (x y z w : ℝ) 
  (h1 : x = 4 * y) 
  (h2 : y = 3 * z) 
  (h3 : z = 5 * w) : 
  (x * z) / (y * w) = 20 := 
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l1503_150308


namespace NUMINAMATH_GPT_min_value_of_a_plus_2b_l1503_150338

theorem min_value_of_a_plus_2b (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (h_eq : 1 / a + 2 / b = 4) : a + 2 * b = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_plus_2b_l1503_150338


namespace NUMINAMATH_GPT_abs_difference_of_two_numbers_l1503_150324

theorem abs_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 105) :
  |x - y| = 6 * Real.sqrt 24.333 := sorry

end NUMINAMATH_GPT_abs_difference_of_two_numbers_l1503_150324


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1503_150326

theorem quadratic_inequality_solution (d : ℝ) 
  (h1 : 0 < d) 
  (h2 : d < 16) : 
  ∃ x : ℝ, (x^2 - 8*x + d < 0) :=
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1503_150326


namespace NUMINAMATH_GPT_production_steps_use_process_flowchart_l1503_150356

def describe_production_steps (task : String) : Prop :=
  task = "describe production steps of a certain product in a factory"

def correct_diagram (diagram : String) : Prop :=
  diagram = "Process Flowchart"

theorem production_steps_use_process_flowchart (task : String) (diagram : String) :
  describe_production_steps task → correct_diagram diagram :=
sorry

end NUMINAMATH_GPT_production_steps_use_process_flowchart_l1503_150356


namespace NUMINAMATH_GPT_remainder_9053_div_98_l1503_150373

theorem remainder_9053_div_98 : 9053 % 98 = 37 :=
by sorry

end NUMINAMATH_GPT_remainder_9053_div_98_l1503_150373


namespace NUMINAMATH_GPT_particle_speed_at_time_t_l1503_150329

noncomputable def position (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + t + 1, 6 * t + 2)

theorem particle_speed_at_time_t (t : ℝ) :
  let dx := (position t).1
  let dy := (position t).2
  let vx := 6 * t + 1
  let vy := 6
  let speed := Real.sqrt (vx^2 + vy^2)
  speed = Real.sqrt (36 * t^2 + 12 * t + 37) :=
by
  sorry

end NUMINAMATH_GPT_particle_speed_at_time_t_l1503_150329


namespace NUMINAMATH_GPT_savings_of_person_l1503_150372

theorem savings_of_person (income expenditure : ℕ) (h_ratio : 3 * expenditure = 2 * income) (h_income : income = 21000) :
  income - expenditure = 7000 :=
by
  sorry

end NUMINAMATH_GPT_savings_of_person_l1503_150372


namespace NUMINAMATH_GPT_sequence_becomes_negative_from_8th_term_l1503_150315

def seq (n : ℕ) : ℤ := 21 + 4 * n - n ^ 2

theorem sequence_becomes_negative_from_8th_term :
  ∀ n, n ≥ 8 ↔ seq n < 0 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_sequence_becomes_negative_from_8th_term_l1503_150315


namespace NUMINAMATH_GPT_find_lesser_fraction_l1503_150392

theorem find_lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_lesser_fraction_l1503_150392


namespace NUMINAMATH_GPT_inequality_range_l1503_150305

theorem inequality_range (a : ℝ) (h : ∀ x : ℝ, |x - 3| + |x + 1| > a) : a < 4 := by
  sorry

end NUMINAMATH_GPT_inequality_range_l1503_150305


namespace NUMINAMATH_GPT_num_distinct_convex_polygons_on_12_points_l1503_150336

theorem num_distinct_convex_polygons_on_12_points : 
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 :=
by
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  have h : num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 := by sorry
  exact h

end NUMINAMATH_GPT_num_distinct_convex_polygons_on_12_points_l1503_150336


namespace NUMINAMATH_GPT_possible_N_l1503_150390

/-- 
  Let N be an integer with N ≥ 3, and let a₀, a₁, ..., a_(N-1) be pairwise distinct reals such that 
  aᵢ ≥ a_(2i mod N) for all i. Prove that N must be a power of 2.
-/
theorem possible_N (N : ℕ) (hN : N ≥ 3) (a : Fin N → ℝ) (h_distinct: Function.Injective a) 
  (h_condition : ∀ i : Fin N, a i ≥ a (⟨(2 * i) % N, sorry⟩)) 
  : ∃ k : ℕ, N = 2^k := 
sorry

end NUMINAMATH_GPT_possible_N_l1503_150390


namespace NUMINAMATH_GPT_pumpkins_eaten_l1503_150393

-- Definitions for the conditions
def originalPumpkins : ℕ := 43
def leftPumpkins : ℕ := 20

-- Theorem statement
theorem pumpkins_eaten : originalPumpkins - leftPumpkins = 23 :=
  by
    -- Proof steps are omitted
    sorry

end NUMINAMATH_GPT_pumpkins_eaten_l1503_150393


namespace NUMINAMATH_GPT_percentage_difference_l1503_150399

theorem percentage_difference:
  let x1 := 0.4 * 60
  let x2 := 0.8 * 25
  x1 - x2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1503_150399


namespace NUMINAMATH_GPT_exponents_problem_l1503_150303

theorem exponents_problem :
  5000 * (5000^9) * 2^(1000) = 5000^(10) * 2^(1000) := by sorry

end NUMINAMATH_GPT_exponents_problem_l1503_150303


namespace NUMINAMATH_GPT_count_valid_three_digit_numbers_l1503_150318

theorem count_valid_three_digit_numbers : 
  let total_three_digit_numbers := 900 
  let invalid_AAB_or_ABA := 81 + 81
  total_three_digit_numbers - invalid_AAB_or_ABA = 738 := 
by 
  let total_three_digit_numbers := 900
  let invalid_AAB_or_ABA := 81 + 81
  show total_three_digit_numbers - invalid_AAB_or_ABA = 738 
  sorry

end NUMINAMATH_GPT_count_valid_three_digit_numbers_l1503_150318


namespace NUMINAMATH_GPT_floor_sqrt_77_l1503_150320

theorem floor_sqrt_77 : 8 < Real.sqrt 77 ∧ Real.sqrt 77 < 9 → Int.floor (Real.sqrt 77) = 8 :=
by
  sorry

end NUMINAMATH_GPT_floor_sqrt_77_l1503_150320


namespace NUMINAMATH_GPT_red_shells_correct_l1503_150375

-- Define the conditions
def total_shells : Nat := 291
def green_shells : Nat := 49
def non_red_green_shells : Nat := 166

-- Define the number of red shells as per the given conditions
def red_shells : Nat :=
  total_shells - green_shells - non_red_green_shells

-- State the theorem
theorem red_shells_correct : red_shells = 76 :=
by
  sorry

end NUMINAMATH_GPT_red_shells_correct_l1503_150375


namespace NUMINAMATH_GPT_largest_product_of_three_l1503_150314

-- Definitions of the numbers in the set
def numbers : List Int := [-5, 1, -3, 5, -2, 2]

-- Define a function to calculate the product of a list of three integers
def product_of_three (a b c : Int) : Int := a * b * c

-- Define a predicate to state that 75 is the largest product of any three numbers from the given list
theorem largest_product_of_three :
  ∃ (a b c : Int), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ product_of_three a b c = 75 :=
sorry

end NUMINAMATH_GPT_largest_product_of_three_l1503_150314


namespace NUMINAMATH_GPT_expectation_defective_items_variance_of_defective_items_l1503_150325
-- Importing the necessary library from Mathlib

-- Define the conditions
def total_products : ℕ := 100
def defective_products : ℕ := 10
def selected_products : ℕ := 3

-- Define the expected number of defective items
def expected_defective_items : ℝ := 0.3

-- Define the variance of defective items
def variance_defective_items : ℝ := 0.2645

-- Lean statements to verify the conditions and results
theorem expectation_defective_items :
  let p := (defective_products: ℝ) / (total_products: ℝ)
  p * (selected_products: ℝ) = expected_defective_items := by sorry

theorem variance_of_defective_items :
  let p := (defective_products: ℝ) / (total_products: ℝ)
  let n := (selected_products: ℝ)
  n * p * (1 - p) * (total_products - n) / (total_products - 1) = variance_defective_items := by sorry

end NUMINAMATH_GPT_expectation_defective_items_variance_of_defective_items_l1503_150325


namespace NUMINAMATH_GPT_student_selection_l1503_150327

theorem student_selection : 
  let first_year := 4
  let second_year := 5
  let third_year := 4
  (first_year * second_year) + (first_year * third_year) + (second_year * third_year) = 56 := by
  let first_year := 4
  let second_year := 5
  let third_year := 4
  sorry

end NUMINAMATH_GPT_student_selection_l1503_150327


namespace NUMINAMATH_GPT_earbuds_cost_before_tax_l1503_150362

-- Define the conditions
variable (C : ℝ) -- The cost before tax
variable (taxRate : ℝ := 0.15)
variable (totalPaid : ℝ := 230)

-- Define the main question in Lean
theorem earbuds_cost_before_tax : C + taxRate * C = totalPaid → C = 200 :=
by
  sorry

end NUMINAMATH_GPT_earbuds_cost_before_tax_l1503_150362


namespace NUMINAMATH_GPT_jade_savings_per_month_l1503_150359

def jade_monthly_income : ℝ := 1600
def jade_living_expense_rate : ℝ := 0.75
def jade_insurance_rate : ℝ := 0.2

theorem jade_savings_per_month : 
  jade_monthly_income * (1 - jade_living_expense_rate - jade_insurance_rate) = 80 := by
  sorry

end NUMINAMATH_GPT_jade_savings_per_month_l1503_150359


namespace NUMINAMATH_GPT_tetrahedron_circumsphere_radius_l1503_150397

theorem tetrahedron_circumsphere_radius :
  ∃ (r : ℝ), 
    (∀ (A B C P : ℝ × ℝ × ℝ),
      (dist A B = 5) ∧
      (dist A C = 5) ∧
      (dist A P = 5) ∧
      (dist B C = 5) ∧
      (dist B P = 5) ∧
      (dist C P = 6) →
      r = (20 * Real.sqrt 39) / 39) :=
sorry

end NUMINAMATH_GPT_tetrahedron_circumsphere_radius_l1503_150397


namespace NUMINAMATH_GPT_farmer_eggs_per_week_l1503_150383

theorem farmer_eggs_per_week (E : ℝ) (chickens : ℝ) (price_per_dozen : ℝ) (total_revenue : ℝ) (num_weeks : ℝ) (total_chickens : ℝ) (dozen : ℝ) 
    (H1 : total_chickens = 46)
    (H2 : price_per_dozen = 3)
    (H3 : total_revenue = 552)
    (H4 : num_weeks = 8)
    (H5 : dozen = 12)
    (H6 : chickens = 46)
    : E = 6 :=
by
  sorry

end NUMINAMATH_GPT_farmer_eggs_per_week_l1503_150383


namespace NUMINAMATH_GPT_maddie_weekend_watch_time_l1503_150357

-- Defining the conditions provided in the problem
def num_episodes : ℕ := 8
def duration_per_episode : ℕ := 44
def minutes_on_monday : ℕ := 138
def minutes_on_tuesday : ℕ := 0
def minutes_on_wednesday : ℕ := 0
def minutes_on_thursday : ℕ := 21
def episodes_on_friday : ℕ := 2

-- Define the total time watched from Monday to Friday
def total_minutes_week : ℕ := num_episodes * duration_per_episode
def total_minutes_mon_to_fri : ℕ := 
  minutes_on_monday + 
  minutes_on_tuesday + 
  minutes_on_wednesday + 
  minutes_on_thursday + 
  (episodes_on_friday * duration_per_episode)

-- Define the weekend watch time
def weekend_watch_time : ℕ := total_minutes_week - total_minutes_mon_to_fri

-- The theorem to prove the correct answer
theorem maddie_weekend_watch_time : weekend_watch_time = 105 := by
  sorry

end NUMINAMATH_GPT_maddie_weekend_watch_time_l1503_150357


namespace NUMINAMATH_GPT_remainder_98_pow_50_mod_100_l1503_150388

/-- 
Theorem: The remainder when \(98^{50}\) is divided by 100 is 24.
-/
theorem remainder_98_pow_50_mod_100 : (98^50 % 100) = 24 := by
  sorry

end NUMINAMATH_GPT_remainder_98_pow_50_mod_100_l1503_150388


namespace NUMINAMATH_GPT_S₉_eq_81_l1503_150358

variable (aₙ : ℕ → ℕ) (S : ℕ → ℕ)
variable (n : ℕ)
variable (a₁ d : ℕ)

-- Conditions
axiom S₃_eq_9 : S 3 = 9
axiom S₆_eq_36 : S 6 = 36
axiom S_n_def : ∀ n, S n = n * a₁ + n * (n - 1) / 2 * d

-- Proof obligation
theorem S₉_eq_81 : S 9 = 81 :=
by
  sorry

end NUMINAMATH_GPT_S₉_eq_81_l1503_150358


namespace NUMINAMATH_GPT_area_trapezoid_def_l1503_150361

noncomputable def area_trapezoid (a : ℝ) (h : a ≠ 0) : ℝ :=
  let b := 108 / a
  let DE := a / 2
  let FG := b / 3
  let height := b / 2
  (DE + FG) * height / 2

theorem area_trapezoid_def (a : ℝ) (h : a ≠ 0) :
  area_trapezoid a h = 18 + 18 / a :=
by
  sorry

end NUMINAMATH_GPT_area_trapezoid_def_l1503_150361


namespace NUMINAMATH_GPT_foma_wait_time_probability_l1503_150319

noncomputable def probability_no_more_than_four_minutes_wait (x y : ℝ) : ℝ :=
if h : 2 < x ∧ x < y ∧ y < 10 ∧ y - x ≤ 4 then
  (1 / 2)
else 0

theorem foma_wait_time_probability :
  ∀ (x y : ℝ), 2 < x → x < y → y < 10 → 
  (probability_no_more_than_four_minutes_wait x y) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_foma_wait_time_probability_l1503_150319


namespace NUMINAMATH_GPT_total_questions_correct_total_answers_correct_l1503_150371

namespace ForumCalculation

def members : ℕ := 200
def questions_per_hour_per_user : ℕ := 3
def hours_in_day : ℕ := 24
def answers_multiplier : ℕ := 3

def total_questions_per_user_per_day : ℕ :=
  questions_per_hour_per_user * hours_in_day

def total_questions_in_a_day : ℕ :=
  members * total_questions_per_user_per_day

def total_answers_per_user_per_day : ℕ :=
  answers_multiplier * total_questions_per_user_per_day

def total_answers_in_a_day : ℕ :=
  members * total_answers_per_user_per_day

theorem total_questions_correct :
  total_questions_in_a_day = 14400 :=
by
  sorry

theorem total_answers_correct :
  total_answers_in_a_day = 43200 :=
by
  sorry

end ForumCalculation

end NUMINAMATH_GPT_total_questions_correct_total_answers_correct_l1503_150371


namespace NUMINAMATH_GPT_arithmetic_progression_no_rth_power_l1503_150323

noncomputable def is_arith_sequence (a : ℕ → ℤ) : Prop := 
∀ n : ℕ, a n = 4 * (n : ℤ) - 2

theorem arithmetic_progression_no_rth_power (n : ℕ) :
  ∃ a : ℕ → ℤ, is_arith_sequence a ∧ 
  (∀ r : ℕ, 2 ≤ r ∧ r ≤ n → 
  ¬ (∃ k : ℤ, ∃ m : ℕ, m > 0 ∧ a m = k ^ r)) := 
sorry

end NUMINAMATH_GPT_arithmetic_progression_no_rth_power_l1503_150323


namespace NUMINAMATH_GPT_total_houses_in_lincoln_county_l1503_150302

theorem total_houses_in_lincoln_county 
  (original_houses : ℕ) 
  (houses_built : ℕ) 
  (h_original : original_houses = 20817) 
  (h_built : houses_built = 97741) : 
  original_houses + houses_built = 118558 := 
by 
  -- Proof steps or tactics would go here
  sorry

end NUMINAMATH_GPT_total_houses_in_lincoln_county_l1503_150302


namespace NUMINAMATH_GPT_triangle_side_b_l1503_150309

theorem triangle_side_b (A B C a b c : ℝ)
  (hA : A = 135)
  (hc : c = 1)
  (hSinB_SinC : Real.sin B * Real.sin C = Real.sqrt 2 / 10) :
  b = Real.sqrt 2 ∨ b = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_b_l1503_150309


namespace NUMINAMATH_GPT_nonnegative_diff_roots_eq_8sqrt2_l1503_150342

noncomputable def roots_diff (a b c : ℝ) : ℝ :=
  if h : b^2 - 4*a*c ≥ 0 then 
    let root1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
    let root2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
    abs (root1 - root2)
  else 
    0

theorem nonnegative_diff_roots_eq_8sqrt2 : 
  roots_diff 1 42 409 = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_nonnegative_diff_roots_eq_8sqrt2_l1503_150342


namespace NUMINAMATH_GPT_expenses_recorded_as_negative_l1503_150321

/-*
  Given:
  1. The income of 5 yuan is recorded as +5 yuan.
  Prove:
  2. The expenses of 5 yuan are recorded as -5 yuan.
*-/

theorem expenses_recorded_as_negative (income_expenses_opposite_sign : ∀ (a : ℤ), -a = -a)
    (income_five_recorded_as_positive : (5 : ℤ) = 5) :
    (-5 : ℤ) = -5 :=
by sorry

end NUMINAMATH_GPT_expenses_recorded_as_negative_l1503_150321


namespace NUMINAMATH_GPT_ivy_baked_55_cupcakes_l1503_150313

-- Definitions based on conditions
def cupcakes_morning : ℕ := 20
def cupcakes_afternoon : ℕ := cupcakes_morning + 15
def total_cupcakes : ℕ := cupcakes_morning + cupcakes_afternoon

-- Theorem statement that needs to be proved
theorem ivy_baked_55_cupcakes : total_cupcakes = 55 := by
    sorry

end NUMINAMATH_GPT_ivy_baked_55_cupcakes_l1503_150313


namespace NUMINAMATH_GPT_general_term_sequence_x_l1503_150312

-- Definitions used in Lean statement corresponding to the conditions.
noncomputable def sequence_a (n : ℕ) : ℝ := sorry

noncomputable def sequence_x (n : ℕ) : ℝ := sorry

axiom condition_1 : ∀ n : ℕ, 
  ((sequence_a (n + 2))⁻¹ = ((sequence_a n)⁻¹ + (sequence_a (n + 1))⁻¹) / 2)

axiom condition_2 {n : ℕ} : sequence_x n > 0

axiom condition_3 : sequence_x 1 = 3

axiom condition_4 : sequence_x 1 + sequence_x 2 + sequence_x 3 = 39

axiom condition_5 (n : ℕ) : (sequence_x n)^(sequence_a n) = 
  (sequence_x (n + 1))^(sequence_a (n + 1)) ∧ 
  (sequence_x (n + 1))^(sequence_a (n + 1)) = 
  (sequence_x (n + 2))^(sequence_a (n + 2))

-- Theorem stating that the general term of sequence {x_n} is 3^n.
theorem general_term_sequence_x : ∀ n : ℕ, sequence_x n = 3^n :=
by
  sorry

end NUMINAMATH_GPT_general_term_sequence_x_l1503_150312


namespace NUMINAMATH_GPT_johns_total_cost_after_discount_l1503_150341

/-- Price of different utensils for John's purchase --/
def forks_cost : ℕ := 25
def knives_cost : ℕ := 30
def spoons_cost : ℕ := 20
def dinner_plate_cost (silverware_cost : ℕ) : ℚ := 0.5 * silverware_cost

/-- Calculating the total cost of silverware --/
def total_silverware_cost : ℕ := forks_cost + knives_cost + spoons_cost

/-- Calculating the total cost before discount --/
def total_cost_before_discount : ℚ := total_silverware_cost + dinner_plate_cost total_silverware_cost

/-- Discount rate --/
def discount_rate : ℚ := 0.10

/-- Discount amount --/
def discount_amount (total_cost : ℚ) : ℚ := discount_rate * total_cost

/-- Total cost after applying discount --/
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount total_cost_before_discount

/-- John's total cost after the discount should be $101.25 --/
theorem johns_total_cost_after_discount : total_cost_after_discount = 101.25 := by
  sorry

end NUMINAMATH_GPT_johns_total_cost_after_discount_l1503_150341


namespace NUMINAMATH_GPT_frank_total_cans_l1503_150304

def cansCollectedSaturday : List Nat := [4, 6, 5, 7, 8]
def cansCollectedSunday : List Nat := [6, 5, 9]
def cansCollectedMonday : List Nat := [8, 8]

def totalCansCollected (lst1 lst2 lst3 : List Nat) : Nat :=
  lst1.sum + lst2.sum + lst3.sum

theorem frank_total_cans :
  totalCansCollected cansCollectedSaturday cansCollectedSunday cansCollectedMonday = 66 :=
by
  sorry

end NUMINAMATH_GPT_frank_total_cans_l1503_150304


namespace NUMINAMATH_GPT_find_x_l1503_150374

theorem find_x (x : ℚ) (h : (3 * x + 4) / 5 = 15) : x = 71 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1503_150374


namespace NUMINAMATH_GPT_situps_ratio_l1503_150385

theorem situps_ratio (ken_situps : ℕ) (nathan_situps : ℕ) (bob_situps : ℕ) :
  ken_situps = 20 →
  nathan_situps = 2 * ken_situps →
  bob_situps = ken_situps + 10 →
  (bob_situps : ℚ) / (ken_situps + nathan_situps : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_situps_ratio_l1503_150385


namespace NUMINAMATH_GPT_point_P_on_number_line_l1503_150337

variable (A : ℝ) (B : ℝ) (P : ℝ)

theorem point_P_on_number_line (hA : A = -1) (hB : B = 5) (hDist : abs (P - A) = abs (B - P)) : P = 2 := 
sorry

end NUMINAMATH_GPT_point_P_on_number_line_l1503_150337
