import Mathlib

namespace NUMINAMATH_GPT_find_constants_l1404_140468

theorem find_constants (A B C : ℤ) (h1 : 1 = A + B) (h2 : -2 = C) (h3 : 5 = -A) :
  A = -5 ∧ B = 6 ∧ C = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_constants_l1404_140468


namespace NUMINAMATH_GPT_probability_of_PAIR_letters_in_PROBABILITY_l1404_140458

theorem probability_of_PAIR_letters_in_PROBABILITY : 
  let total_letters := 11
  let favorable_letters := 4
  favorable_letters / total_letters = 4 / 11 :=
by
  let total_letters := 11
  let favorable_letters := 4
  show favorable_letters / total_letters = 4 / 11
  sorry

end NUMINAMATH_GPT_probability_of_PAIR_letters_in_PROBABILITY_l1404_140458


namespace NUMINAMATH_GPT_lcm_15_18_l1404_140403

theorem lcm_15_18 : Nat.lcm 15 18 = 90 := by
  sorry

end NUMINAMATH_GPT_lcm_15_18_l1404_140403


namespace NUMINAMATH_GPT_ratio_of_cream_l1404_140454

theorem ratio_of_cream
  (joes_initial_coffee : ℕ := 20)
  (joe_cream_added : ℕ := 3)
  (joe_amount_drank : ℕ := 4)
  (joanns_initial_coffee : ℕ := 20)
  (joann_amount_drank : ℕ := 4)
  (joann_cream_added : ℕ := 3) :
  let joe_final_cream := (joe_cream_added - joe_amount_drank * (joe_cream_added / (joe_cream_added + joes_initial_coffee)))
  let joann_final_cream := joann_cream_added
  (joe_final_cream / joanns_initial_coffee + joann_cream_added = 15 / 23) :=
sorry

end NUMINAMATH_GPT_ratio_of_cream_l1404_140454


namespace NUMINAMATH_GPT_negation_of_proposition_l1404_140457

noncomputable def P (x : ℝ) : Prop := x^2 + 1 ≥ 0

theorem negation_of_proposition :
  (¬ ∀ x, x > 1 → P x) ↔ (∃ x, x > 1 ∧ ¬ P x) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1404_140457


namespace NUMINAMATH_GPT_arctan_sum_l1404_140435

theorem arctan_sum {a b : ℝ} (h3 : a = 3) (h7 : b = 7) :
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_arctan_sum_l1404_140435


namespace NUMINAMATH_GPT_part_I_part_II_l1404_140434

def f (x : ℝ) : ℝ := abs (2 * x - 7) + 1

def g (x : ℝ) : ℝ := abs (2 * x - 7) - 2 * abs (x - 1) + 1

theorem part_I :
  {x : ℝ | f x ≤ x} = {x : ℝ | (8 / 3) ≤ x ∧ x ≤ 6} := sorry

theorem part_II (a : ℝ) :
  (∃ x : ℝ, g x ≤ a) → a ≥ -4 := sorry

end NUMINAMATH_GPT_part_I_part_II_l1404_140434


namespace NUMINAMATH_GPT_fraction_addition_l1404_140485

theorem fraction_addition : 
  (2 : ℚ) / 5 + (3 : ℚ) / 8 + 1 = 71 / 40 :=
by
  sorry

end NUMINAMATH_GPT_fraction_addition_l1404_140485


namespace NUMINAMATH_GPT_bob_needs_50_planks_l1404_140488

-- Define the raised bed dimensions and requirements
structure RaisedBedDimensions where
  height : ℕ -- in feet
  width : ℕ  -- in feet
  length : ℕ -- in feet

def plank_length : ℕ := 8  -- length of each plank in feet
def plank_width : ℕ := 1  -- width of each plank in feet
def num_beds : ℕ := 10

def planks_needed (bed : RaisedBedDimensions) : ℕ :=
  let long_sides := 2  -- 2 long sides per bed
  let short_sides := 2 * (bed.width / plank_length)  -- 1/4 plank per short side if width is 2 feet
  let total_sides := long_sides + short_sides
  let stacked_sides := total_sides * (bed.height / plank_width)  -- stacked to match height
  stacked_sides

def raised_bed : RaisedBedDimensions := {height := 2, width := 2, length := 8}

theorem bob_needs_50_planks : planks_needed raised_bed * num_beds = 50 := by
  sorry

end NUMINAMATH_GPT_bob_needs_50_planks_l1404_140488


namespace NUMINAMATH_GPT_green_competition_l1404_140474

theorem green_competition {x : ℕ} (h : 0 ≤ x ∧ x ≤ 25) : 
  5 * x - (25 - x) ≥ 85 :=
by
  sorry

end NUMINAMATH_GPT_green_competition_l1404_140474


namespace NUMINAMATH_GPT_adam_coins_value_l1404_140479

theorem adam_coins_value (num_coins : ℕ) (subset_value: ℕ) (subset_num: ℕ) (total_value: ℕ)
  (h1 : num_coins = 20)
  (h2 : subset_value = 16)
  (h3 : subset_num = 4)
  (h4 : total_value = num_coins * (subset_value / subset_num)) :
  total_value = 80 := 
by
  sorry

end NUMINAMATH_GPT_adam_coins_value_l1404_140479


namespace NUMINAMATH_GPT_nearest_integer_to_x_plus_2y_l1404_140496

theorem nearest_integer_to_x_plus_2y
  (x y : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : |x| + 2 * y = 6)
  (h2 : |x| * y + x^3 = 2) :
  Int.floor (x + 2 * y + 0.5) = 6 :=
by sorry

end NUMINAMATH_GPT_nearest_integer_to_x_plus_2y_l1404_140496


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1404_140428

theorem necessary_but_not_sufficient (x : ℝ) : ( (x + 1) * (x + 2) > 0 → (x + 1) * (x^2 + 2) > 0 ) :=
by
  intro h
  -- insert steps urther here, if proof was required
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1404_140428


namespace NUMINAMATH_GPT_inscribed_square_ratio_l1404_140442

theorem inscribed_square_ratio
  (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 13) (h₁ : a^2 + b^2 = c^2)
  (x y : ℝ) (hx : x = 60 / 17) (hy : y = 144 / 17) :
  (x / y) = 5 / 12 := sorry

end NUMINAMATH_GPT_inscribed_square_ratio_l1404_140442


namespace NUMINAMATH_GPT_sixth_graders_more_than_seventh_l1404_140469

theorem sixth_graders_more_than_seventh
  (bookstore_sells_pencils_in_whole_cents : True)
  (seventh_graders : ℕ)
  (sixth_graders : ℕ)
  (seventh_packs_payment : ℕ)
  (sixth_packs_payment : ℕ)
  (each_pack_contains_two_pencils : True)
  (seventh_graders_condition : seventh_graders = 25)
  (seventh_packs_payment_condition : seventh_packs_payment * seventh_graders = 275)
  (sixth_graders_condition : sixth_graders = 36 / 2)
  (sixth_packs_payment_condition : sixth_packs_payment * sixth_graders = 216) : 
  sixth_graders - seventh_graders = 7 := sorry

end NUMINAMATH_GPT_sixth_graders_more_than_seventh_l1404_140469


namespace NUMINAMATH_GPT_find_q_l1404_140472

noncomputable def has_two_distinct_negative_roots (q : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
  (x₁ ^ 4 + q * x₁ ^ 3 + 2 * x₁ ^ 2 + q * x₁ + 4 = 0) ∧ 
  (x₂ ^ 4 + q * x₂ ^ 3 + 2 * x₂ ^ 2 + q * x₂ + 4 = 0)

theorem find_q (q : ℝ) : 
  has_two_distinct_negative_roots q ↔ q ≤ 3 / Real.sqrt 2 := sorry

end NUMINAMATH_GPT_find_q_l1404_140472


namespace NUMINAMATH_GPT_minimum_floor_sum_l1404_140478

theorem minimum_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ⌊a^2 + b^2 / c⌋ + ⌊b^2 + c^2 / a⌋ + ⌊c^2 + a^2 / b⌋ = 34 :=
sorry

end NUMINAMATH_GPT_minimum_floor_sum_l1404_140478


namespace NUMINAMATH_GPT_selection_count_l1404_140481

theorem selection_count :
  (Nat.choose 6 3) * (Nat.choose 5 2) = 200 := 
sorry

end NUMINAMATH_GPT_selection_count_l1404_140481


namespace NUMINAMATH_GPT_sphere_volume_ratio_l1404_140456

theorem sphere_volume_ratio
  (r R : ℝ)
  (h : (4:ℝ) * π * r^2 / (4 * π * R^2) = (4:ℝ) / 9) : 
  (r^3 / R^3 = (8:ℝ) / 27) := by
  sorry

end NUMINAMATH_GPT_sphere_volume_ratio_l1404_140456


namespace NUMINAMATH_GPT_beads_cost_is_three_l1404_140444

-- Define the given conditions
def cost_of_string_per_bracelet : Nat := 1
def selling_price_per_bracelet : Nat := 6
def number_of_bracelets_sold : Nat := 25
def total_profit : Nat := 50

-- The amount spent on beads per bracelet
def amount_spent_on_beads_per_bracelet (B : Nat) : Prop :=
  B = (total_profit + number_of_bracelets_sold * (cost_of_string_per_bracelet + B) - number_of_bracelets_sold * selling_price_per_bracelet) / number_of_bracelets_sold 

-- The main goal is to prove that the amount spent on beads is 3
theorem beads_cost_is_three : amount_spent_on_beads_per_bracelet 3 :=
by sorry

end NUMINAMATH_GPT_beads_cost_is_three_l1404_140444


namespace NUMINAMATH_GPT_area_triangle_3_6_l1404_140482

/-
Problem: Prove that the area of a triangle with base 3 meters and height 6 meters is 9 square meters.
Definitions: 
- base: The base of the triangle is 3 meters.
- height: The height of the triangle is 6 meters.
Conditions: 
- The area of a triangle formula.
Correct Answer: 9 square meters.
-/

def area_of_triangle (base height : ℕ) : ℕ := (base * height) / 2

theorem area_triangle_3_6 : area_of_triangle 3 6 = 9 := by
  sorry

end NUMINAMATH_GPT_area_triangle_3_6_l1404_140482


namespace NUMINAMATH_GPT_number_of_smaller_pipes_l1404_140404

theorem number_of_smaller_pipes (D_L D_s : ℝ) (h1 : D_L = 8) (h2 : D_s = 2) (v: ℝ) :
  let A_L := (π * (D_L / 2)^2)
  let A_s := (π * (D_s / 2)^2)
  (A_L / A_s) = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_smaller_pipes_l1404_140404


namespace NUMINAMATH_GPT_target_heart_rate_of_30_year_old_l1404_140422

variable (age : ℕ) (T M : ℕ)

def maximum_heart_rate (age : ℕ) : ℕ :=
  210 - age

def target_heart_rate (M : ℕ) : ℕ :=
  (75 * M) / 100

theorem target_heart_rate_of_30_year_old :
  maximum_heart_rate 30 = 180 →
  target_heart_rate (maximum_heart_rate 30) = 135 :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_target_heart_rate_of_30_year_old_l1404_140422


namespace NUMINAMATH_GPT_min_prime_factor_sum_l1404_140489

theorem min_prime_factor_sum (x y a b c d : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : 5 * x^7 = 13 * y^11)
  (h4 : x = 13^6 * 5^7) (h5 : a = 13) (h6 : b = 5) (h7 : c = 6) (h8 : d = 7) : 
  a + b + c + d = 31 :=
by
  sorry

end NUMINAMATH_GPT_min_prime_factor_sum_l1404_140489


namespace NUMINAMATH_GPT_exists_group_of_three_friends_l1404_140461

-- Defining the context of the problem
def people := Fin 10 -- a finite set of 10 people
def quarrel (x y : people) : Prop := -- a predicate indicating a quarrel between two people
sorry

-- Given conditions
axiom quarreled_pairs : ∃ S : Finset (people × people), S.card = 14 ∧ 
  ∀ {x y : people}, (x, y) ∈ S → x ≠ y ∧ quarrel x y

-- Question: Prove there exists a set of 3 friends among these 10 people
theorem exists_group_of_three_friends (p : Finset people):
  ∃ (group : Finset people), group.card = 3 ∧ ∀ {x y : people}, 
  x ∈ group → y ∈ group → x ≠ y → ¬ quarrel x y :=
sorry

end NUMINAMATH_GPT_exists_group_of_three_friends_l1404_140461


namespace NUMINAMATH_GPT_circle_equation_l1404_140467

variable (x y : ℝ)

def center : ℝ × ℝ := (4, -6)
def radius : ℝ := 3

theorem circle_equation : (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ (x - 4)^2 + (y + 6)^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l1404_140467


namespace NUMINAMATH_GPT_reciprocal_of_2023_l1404_140499

theorem reciprocal_of_2023 : (2023 : ℝ)⁻¹ = 1 / 2023 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_2023_l1404_140499


namespace NUMINAMATH_GPT_points_after_perfect_games_l1404_140415

-- Given conditions
def perfect_score := 21
def num_games := 3

-- Theorem statement
theorem points_after_perfect_games : perfect_score * num_games = 63 := by
  sorry

end NUMINAMATH_GPT_points_after_perfect_games_l1404_140415


namespace NUMINAMATH_GPT_alice_bob_sum_proof_l1404_140476

noncomputable def alice_bob_sum_is_22 : Prop :=
  ∃ A B : ℕ, (1 ≤ A ∧ A ≤ 50) ∧ (1 ≤ B ∧ B ≤ 50) ∧ (B % 3 = 0) ∧ (∃ k : ℕ, 2 * B + A = k^2) ∧ (A + B = 22)

theorem alice_bob_sum_proof : alice_bob_sum_is_22 :=
sorry

end NUMINAMATH_GPT_alice_bob_sum_proof_l1404_140476


namespace NUMINAMATH_GPT_calculate_k_l1404_140495

theorem calculate_k (β : ℝ) (hβ : (Real.tan β + 1 / Real.tan β) ^ 2 = k + 1) : k = 1 := by
  sorry

end NUMINAMATH_GPT_calculate_k_l1404_140495


namespace NUMINAMATH_GPT_z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l1404_140409

-- Given conditions and definitions
variables {α : ℝ} {z : ℂ} 
  (hz : z + 1/z = 2 * Real.cos α)

-- The target statement
theorem z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha (n : ℕ) (hz : z + 1/z = 2 * Real.cos α) : 
  z ^ n + 1 / (z ^ n) = 2 * Real.cos (n * α) := 
  sorry

end NUMINAMATH_GPT_z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l1404_140409


namespace NUMINAMATH_GPT_triangle_min_value_l1404_140441

open Real

theorem triangle_min_value
  (A B C : ℝ)
  (h_triangle: A + B + C = π)
  (h_sin: sin (2 * A + B) = 2 * sin B) :
  tan A + tan C + 2 / tan B ≥ 2 :=
sorry

end NUMINAMATH_GPT_triangle_min_value_l1404_140441


namespace NUMINAMATH_GPT_train_speed_l1404_140418

theorem train_speed
  (num_carriages : ℕ)
  (length_carriage length_engine : ℕ)
  (bridge_length_km : ℝ)
  (crossing_time_min : ℝ)
  (h1 : num_carriages = 24)
  (h2 : length_carriage = 60)
  (h3 : length_engine = 60)
  (h4 : bridge_length_km = 4.5)
  (h5 : crossing_time_min = 6) :
  (num_carriages * length_carriage + length_engine) / 1000 + bridge_length_km / (crossing_time_min / 60) = 60 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1404_140418


namespace NUMINAMATH_GPT_S8_value_l1404_140425

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 0 + n * (a 1 - a 0)

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

def condition_a3_a6 (a : ℕ → ℝ) : Prop :=
  a 3 = 9 - a 6

theorem S8_value (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_formula : sum_of_first_n_terms S a)
  (h_condition : condition_a3_a6 a) :
  S 8 = 72 :=
by
  sorry

end NUMINAMATH_GPT_S8_value_l1404_140425


namespace NUMINAMATH_GPT_x_intercept_of_l1_is_2_l1404_140446

theorem x_intercept_of_l1_is_2 (a : ℝ) (l1_perpendicular_l2 : ∀ (x y : ℝ), 
  ((a+3)*x + y - 4 = 0) -> (x + (a-1)*y + 4 = 0) -> False) : 
  ∃ b : ℝ, (2*b + 0 - 4 = 0) ∧ b = 2 := 
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_l1_is_2_l1404_140446


namespace NUMINAMATH_GPT_compute_expression_l1404_140430

def sum_of_squares := 7^2 + 5^2
def square_of_sum := (7 + 5)^2
def sum_of_both := sum_of_squares + square_of_sum
def final_result := 2 * sum_of_both

theorem compute_expression : final_result = 436 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1404_140430


namespace NUMINAMATH_GPT_fourth_term_of_geometric_sequence_l1404_140443

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) :=
  a * r ^ (n - 1)

theorem fourth_term_of_geometric_sequence 
  (a : ℝ) (r : ℝ) (ar5_eq : a * r ^ 5 = 32) 
  (a_eq : a = 81) :
  geometric_sequence a r 4 = 24 := 
by 
  sorry

end NUMINAMATH_GPT_fourth_term_of_geometric_sequence_l1404_140443


namespace NUMINAMATH_GPT_correct_option_is_B_l1404_140448

theorem correct_option_is_B (a : ℝ) : 
  (¬ (-2 * a^2 * b)^3 = -6 * a^6 * b^3) ∧
  (a^7 / a = a^6) ∧
  (¬ (a + 1)^2 = a^2 + 1) ∧
  (¬ 2 * a + 3 * b = 5 * a * b) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l1404_140448


namespace NUMINAMATH_GPT_sum_q_evals_l1404_140431

noncomputable def q : ℕ → ℤ := sorry -- definition of q will be derived from conditions

theorem sum_q_evals :
  (q 0) + (q 1) + (q 2) + (q 3) + (q 4) + (q 5) + (q 6) + (q 7) + (q 8) + (q 9) +
  (q 10) + (q 11) + (q 12) + (q 13) + (q 14) + (q 15) + (q 16) + (q 17) + (q 18) = 456 :=
by
  -- Given conditions
  have h1 : q 1 = 3 := sorry
  have h6 : q 6 = 23 := sorry
  have h12 : q 12 = 17 := sorry
  have h17 : q 17 = 31 := sorry
  -- Proof outline (solved steps omitted for clarity)
  sorry

end NUMINAMATH_GPT_sum_q_evals_l1404_140431


namespace NUMINAMATH_GPT_value_of_a_27_l1404_140437

def a_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * n

theorem value_of_a_27 (a : ℕ → ℕ) (h : a_sequence a) : a 27 = 702 :=
sorry

end NUMINAMATH_GPT_value_of_a_27_l1404_140437


namespace NUMINAMATH_GPT_multiply_1546_by_100_l1404_140470

theorem multiply_1546_by_100 : 15.46 * 100 = 1546 :=
by
  sorry

end NUMINAMATH_GPT_multiply_1546_by_100_l1404_140470


namespace NUMINAMATH_GPT_half_angle_in_quadrant_l1404_140449

theorem half_angle_in_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 / 2) * Real.pi) :
  (π / 2 < α / 2 ∧ α / 2 < π) ∨ (3 * π / 2 < α / 2 ∧ α / 2 < 2 * π) :=
sorry

end NUMINAMATH_GPT_half_angle_in_quadrant_l1404_140449


namespace NUMINAMATH_GPT_problem_l1404_140465

theorem problem (a k : ℕ) (h_a_pos : 0 < a) (h_a_k_pos : 0 < k) (h_div : (a^2 + k) ∣ ((a - 1) * a * (a + 1))) : k ≥ a :=
sorry

end NUMINAMATH_GPT_problem_l1404_140465


namespace NUMINAMATH_GPT_probability_top_three_same_color_l1404_140475

/-- 
  A theorem stating the probability that the top three cards from a shuffled 
  standard deck of 52 cards are all of the same color is \(\frac{12}{51}\).
-/
theorem probability_top_three_same_color : 
  let deck := 52
  let colors := 2
  let cards_per_color := 26
  let favorable_outcomes := 2 * 26 * 25 * 24
  let total_outcomes := 52 * 51 * 50
  favorable_outcomes / total_outcomes = 12 / 51 :=
by
  sorry

end NUMINAMATH_GPT_probability_top_three_same_color_l1404_140475


namespace NUMINAMATH_GPT_problem_g_eq_l1404_140486

noncomputable def g : ℝ → ℝ := sorry

theorem problem_g_eq :
  (∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x + x) →
  g 3 = ( -31 - 3 * 3^(1/3)) / 8 :=
by
  intro h
  -- proof goes here
  sorry

end NUMINAMATH_GPT_problem_g_eq_l1404_140486


namespace NUMINAMATH_GPT_product_of_number_subtracting_7_equals_9_l1404_140417

theorem product_of_number_subtracting_7_equals_9 (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 := by
  sorry

end NUMINAMATH_GPT_product_of_number_subtracting_7_equals_9_l1404_140417


namespace NUMINAMATH_GPT_find_s_l1404_140464

def f (x s : ℝ) := 3 * x^5 + 2 * x^4 - x^3 + 4 * x^2 - 5 * x + s

theorem find_s (s : ℝ) (h : f 3 s = 0) : s = -885 :=
  by sorry

end NUMINAMATH_GPT_find_s_l1404_140464


namespace NUMINAMATH_GPT_distribute_5_cousins_in_4_rooms_l1404_140445

theorem distribute_5_cousins_in_4_rooms : 
  let rooms := 4
  let cousins := 5
  ∃ ways : ℕ, ways = 67 ∧ rooms = 4 ∧ cousins = 5 := sorry

end NUMINAMATH_GPT_distribute_5_cousins_in_4_rooms_l1404_140445


namespace NUMINAMATH_GPT_equation_equiv_product_zero_l1404_140477

theorem equation_equiv_product_zero (a b x y : ℝ) :
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1) →
  ∃ (m n p : ℤ), (a^m * x - a^n) * (a^p * y - a^3) = a^5 * b^5 ∧ m * n * p = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_equation_equiv_product_zero_l1404_140477


namespace NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l1404_140410

theorem condition_sufficient_but_not_necessary (x : ℝ) :
  (x^3 > 8 → |x| > 2) ∧ (|x| > 2 → ¬ (x^3 ≤ 8 ∨ x^3 ≥ 8)) := by
  sorry

end NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l1404_140410


namespace NUMINAMATH_GPT_different_product_l1404_140473

theorem different_product :
  let P1 := 190 * 80
  let P2 := 19 * 800
  let P3 := 19 * 8 * 10
  let P4 := 19 * 8 * 100
  P3 ≠ P1 ∧ P3 ≠ P2 ∧ P3 ≠ P4 :=
by
  sorry

end NUMINAMATH_GPT_different_product_l1404_140473


namespace NUMINAMATH_GPT_prime_squares_5000_9000_l1404_140402

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_squares_5000_9000 : 
  ∃ (l : List ℕ), 
  (∀ p ∈ l, is_prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧ 
  l.length = 6 := 
by
  sorry

end NUMINAMATH_GPT_prime_squares_5000_9000_l1404_140402


namespace NUMINAMATH_GPT_markup_percentage_l1404_140492

theorem markup_percentage {C : ℝ} (hC0: 0 < C) (h1: 0 < 1.125 * C) : 
  ∃ (x : ℝ), 0.75 * (1.20 * C * (1 + x / 100)) = 1.125 * C ∧ x = 25 := 
by
  have h2 : 1.20 = (6 / 5 : ℝ) := by norm_num
  have h3 : 0.75 = (3 / 4 : ℝ) := by norm_num
  sorry

end NUMINAMATH_GPT_markup_percentage_l1404_140492


namespace NUMINAMATH_GPT_equal_triples_l1404_140432

theorem equal_triples (a b c x : ℝ) (h_abc : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : (xb + (1 - x) * c) / a = (x * c + (1 - x) * a) / b ∧ 
          (x * c + (1 - x) * a) / b = (x * a + (1 - x) * b) / c) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_GPT_equal_triples_l1404_140432


namespace NUMINAMATH_GPT_maximum_cows_l1404_140463

theorem maximum_cows (s c : ℕ) (h1 : 30 * s + 33 * c = 1300) (h2 : c > 2 * s) : c ≤ 30 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_maximum_cows_l1404_140463


namespace NUMINAMATH_GPT_find_min_value_expression_l1404_140440

noncomputable def minValueExpression (θ : ℝ) : ℝ :=
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ

theorem find_min_value_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 ∧ minValueExpression θ = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_find_min_value_expression_l1404_140440


namespace NUMINAMATH_GPT_clock_first_ring_at_midnight_l1404_140427

theorem clock_first_ring_at_midnight (rings_every_n_hours : ℕ) (rings_per_day : ℕ) (hours_in_day : ℕ) :
  rings_every_n_hours = 3 ∧ rings_per_day = 8 ∧ hours_in_day = 24 →
  ∃ first_ring_time : Nat, first_ring_time = 0 :=
by
  sorry

end NUMINAMATH_GPT_clock_first_ring_at_midnight_l1404_140427


namespace NUMINAMATH_GPT_find_third_divisor_l1404_140480

theorem find_third_divisor (n : ℕ) (d : ℕ) 
  (h1 : (n - 4) % 12 = 0)
  (h2 : (n - 4) % 16 = 0)
  (h3 : (n - 4) % d = 0)
  (h4 : (n - 4) % 21 = 0)
  (h5 : (n - 4) % 28 = 0)
  (h6 : n = 1012) :
  d = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_third_divisor_l1404_140480


namespace NUMINAMATH_GPT_math_problem_l1404_140429

theorem math_problem
  (z : ℝ)
  (hz : z = 80)
  (y : ℝ)
  (hy : y = (1/4) * z)
  (x : ℝ)
  (hx : x = (1/3) * y)
  (w : ℝ)
  (hw : w = x + y + z) :
  x = 20 / 3 ∧ w = 320 / 3 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1404_140429


namespace NUMINAMATH_GPT_solve_sin_cos_eqn_l1404_140438

theorem solve_sin_cos_eqn (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x + Real.cos x = 1) :
  x = 0 ∨ x = Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_solve_sin_cos_eqn_l1404_140438


namespace NUMINAMATH_GPT_find_f3_value_l1404_140459

noncomputable def f (x : ℚ) : ℚ := (x^2 + 2*x + 1) / (4*x - 5)

theorem find_f3_value : f 3 = 16 / 7 :=
by sorry

end NUMINAMATH_GPT_find_f3_value_l1404_140459


namespace NUMINAMATH_GPT_new_average_score_l1404_140423

theorem new_average_score (n : ℕ) (initial_avg : ℕ) (grace_marks : ℕ) (h1 : n = 35) (h2 : initial_avg = 37) (h3 : grace_marks = 3) : initial_avg + grace_marks = 40 := by
  sorry

end NUMINAMATH_GPT_new_average_score_l1404_140423


namespace NUMINAMATH_GPT_inverse_function_correct_l1404_140498

noncomputable def f (x : ℝ) : ℝ :=
  (x - 1) ^ 2 + 1

noncomputable def f_inv (y : ℝ) : ℝ :=
  1 - Real.sqrt (y - 1)

theorem inverse_function_correct (x : ℝ) (hx : x ≥ 2) :
  f_inv x = 1 - Real.sqrt (x - 1) ∧ ∀ y : ℝ, (y ≤ 0) → f y = x → y = f_inv x :=
by {
  sorry
}

end NUMINAMATH_GPT_inverse_function_correct_l1404_140498


namespace NUMINAMATH_GPT_total_shaded_area_l1404_140401

theorem total_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) : 
  S^2 + 12 * T^2 = 15.75 :=
by 
  sorry

end NUMINAMATH_GPT_total_shaded_area_l1404_140401


namespace NUMINAMATH_GPT_eunsung_sungmin_menu_cases_l1404_140487

theorem eunsung_sungmin_menu_cases :
  let kinds_of_chicken := 4
  let kinds_of_pizza := 3
  let same_chicken_different_pizza :=
    kinds_of_chicken * (kinds_of_pizza * (kinds_of_pizza - 1))
  let same_pizza_different_chicken :=
    kinds_of_pizza * (kinds_of_chicken * (kinds_of_chicken - 1))
  same_chicken_different_pizza + same_pizza_different_chicken = 60 :=
by
  sorry

end NUMINAMATH_GPT_eunsung_sungmin_menu_cases_l1404_140487


namespace NUMINAMATH_GPT_Alice_favorite_number_l1404_140421

theorem Alice_favorite_number :
  ∃ n : ℕ, (30 ≤ n ∧ n ≤ 70) ∧ (7 ∣ n) ∧ ¬(3 ∣ n) ∧ (4 ∣ (n / 10 + n % 10)) ∧ n = 35 :=
by
  sorry

end NUMINAMATH_GPT_Alice_favorite_number_l1404_140421


namespace NUMINAMATH_GPT_horse_buying_problem_l1404_140490

variable (x y z : ℚ)

theorem horse_buying_problem :
  (x + 1/2 * y + 1/2 * z = 12) →
  (y + 1/3 * x + 1/3 * z = 12) →
  (z + 1/4 * x + 1/4 * y = 12) →
  x = 60/17 ∧ y = 136/17 ∧ z = 156/17 :=
by
  sorry

end NUMINAMATH_GPT_horse_buying_problem_l1404_140490


namespace NUMINAMATH_GPT_no_real_roots_abs_eq_l1404_140426

theorem no_real_roots_abs_eq (x : ℝ) : 
  |2*x - 5| + |3*x - 7| + |5*x - 11| = 2015/2016 → false :=
by sorry

end NUMINAMATH_GPT_no_real_roots_abs_eq_l1404_140426


namespace NUMINAMATH_GPT_perpendicular_condition_l1404_140439

def vector_a : ℝ × ℝ := (4, 3)
def vector_b : ℝ × ℝ := (-1, 2)

def add_vector_scaled (a b : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  (a.1 + k * b.1, a.2 + k * b.2)

def sub_vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perpendicular_condition (k : ℝ) :
  dot_product (add_vector_scaled vector_a vector_b k) (sub_vector vector_a vector_b) = 0 ↔ k = 23 / 3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_condition_l1404_140439


namespace NUMINAMATH_GPT_coffee_cost_l1404_140460

theorem coffee_cost :
  ∃ y : ℕ, 
  (∃ x : ℕ, 3 * x + 2 * y = 630 ∧ 2 * x + 3 * y = 690) → y = 162 :=
by
  sorry

end NUMINAMATH_GPT_coffee_cost_l1404_140460


namespace NUMINAMATH_GPT_temperature_on_Saturday_l1404_140419

theorem temperature_on_Saturday 
  (avg_temp : ℕ)
  (sun_temp : ℕ) 
  (mon_temp : ℕ) 
  (tue_temp : ℕ) 
  (wed_temp : ℕ) 
  (thu_temp : ℕ) 
  (fri_temp : ℕ)
  (saturday_temp : ℕ)
  (h_avg : avg_temp = 53)
  (h_sun : sun_temp = 40)
  (h_mon : mon_temp = 50) 
  (h_tue : tue_temp = 65) 
  (h_wed : wed_temp = 36) 
  (h_thu : thu_temp = 82) 
  (h_fri : fri_temp = 72) 
  (h_week : 7 * avg_temp = sun_temp + mon_temp + tue_temp + wed_temp + thu_temp + fri_temp + saturday_temp) :
  saturday_temp = 26 := 
by
  sorry

end NUMINAMATH_GPT_temperature_on_Saturday_l1404_140419


namespace NUMINAMATH_GPT_quadrilateral_area_inequality_l1404_140406

theorem quadrilateral_area_inequality
  (a b c d S : ℝ)
  (hS : 0 ≤ S)
  (h : S = (a + b) / 4 * (c + d) / 4)
  : S ≤ (a + b) / 4 * (c + d) / 4 := by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_inequality_l1404_140406


namespace NUMINAMATH_GPT_branches_on_one_stem_l1404_140405

theorem branches_on_one_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_branches_on_one_stem_l1404_140405


namespace NUMINAMATH_GPT_car_dealership_theorem_l1404_140416

def car_dealership_problem : Prop :=
  let initial_cars := 100
  let new_shipment := 150
  let initial_silver_percentage := 0.20
  let new_silver_percentage := 0.40
  let initial_silver := initial_silver_percentage * initial_cars
  let new_silver := new_silver_percentage * new_shipment
  let total_silver := initial_silver + new_silver
  let total_cars := initial_cars + new_shipment
  let silver_percentage := (total_silver / total_cars) * 100
  silver_percentage = 32

theorem car_dealership_theorem : car_dealership_problem :=
by {
  sorry
}

end NUMINAMATH_GPT_car_dealership_theorem_l1404_140416


namespace NUMINAMATH_GPT_x_finishes_remaining_work_in_14_days_l1404_140462

-- Define the work rates of X and Y
def work_rate_X : ℚ := 1 / 21
def work_rate_Y : ℚ := 1 / 15

-- Define the amount of work Y completed in 5 days
def work_done_by_Y_in_5_days : ℚ := 5 * work_rate_Y

-- Define the remaining work after Y left
def remaining_work : ℚ := 1 - work_done_by_Y_in_5_days

-- Define the number of days needed for X to finish the remaining work
def x_days_remaining : ℚ := remaining_work / work_rate_X

-- Statement to prove
theorem x_finishes_remaining_work_in_14_days : x_days_remaining = 14 := by
  sorry

end NUMINAMATH_GPT_x_finishes_remaining_work_in_14_days_l1404_140462


namespace NUMINAMATH_GPT_arithmetic_progression_num_terms_l1404_140466

theorem arithmetic_progression_num_terms (a d n : ℕ) (h_even : n % 2 = 0) 
    (h_sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 30)
    (h_sum_even : (n / 2) * (2 * a + 2 * d + (n - 2) * d) = 36)
    (h_diff_last_first : (n - 1) * d = 12) :
    n = 8 := 
sorry

end NUMINAMATH_GPT_arithmetic_progression_num_terms_l1404_140466


namespace NUMINAMATH_GPT_time_calculation_correct_l1404_140453

theorem time_calculation_correct :
  let start_hour := 3
  let start_minute := 0
  let start_second := 0
  let hours_to_add := 158
  let minutes_to_add := 55
  let seconds_to_add := 32
  let total_seconds := seconds_to_add + minutes_to_add * 60 + hours_to_add * 3600
  let new_hour := (start_hour + (total_seconds / 3600) % 12) % 12
  let new_minute := (start_minute + (total_seconds / 60) % 60) % 60
  let new_second := (start_second + total_seconds % 60) % 60
  let A := new_hour
  let B := new_minute
  let C := new_second
  A + B + C = 92 :=
by
  sorry

end NUMINAMATH_GPT_time_calculation_correct_l1404_140453


namespace NUMINAMATH_GPT_quadrant_of_angle_l1404_140471

variable (α : ℝ)

theorem quadrant_of_angle (h₁ : Real.sin α < 0) (h₂ : Real.tan α > 0) : 
  3 * (π / 2) < α ∧ α < 2 * π ∨ π < α ∧ α < 3 * (π / 2) :=
by
  sorry

end NUMINAMATH_GPT_quadrant_of_angle_l1404_140471


namespace NUMINAMATH_GPT_find_a_l1404_140450

-- Given function
def quadratic_func (a x : ℝ) := a * (x - 1)^2 - a

-- Conditions
def condition1 (a : ℝ) := a ≠ 0
def condition2 (x : ℝ) := -1 ≤ x ∧ x ≤ 4
def min_value (y : ℝ) := y = -4

theorem find_a (a : ℝ) (ha : condition1 a) :
  ∃ a, (∀ x, condition2 x → quadratic_func a x = -4) → (a = 4 ∨ a = -1 / 2) :=
sorry

end NUMINAMATH_GPT_find_a_l1404_140450


namespace NUMINAMATH_GPT_gcd_polynomials_l1404_140411

-- Given condition: a is an even multiple of 1009
def is_even_multiple_of_1009 (a : ℤ) : Prop :=
  ∃ k : ℤ, a = 2 * 1009 * k

-- Statement: gcd(2a^2 + 31a + 58, a + 15) = 1
theorem gcd_polynomials (a : ℤ) (ha : is_even_multiple_of_1009 a) :
  gcd (2 * a^2 + 31 * a + 58) (a + 15) = 1 := 
sorry

end NUMINAMATH_GPT_gcd_polynomials_l1404_140411


namespace NUMINAMATH_GPT_trig_expression_evaluation_l1404_140414

theorem trig_expression_evaluation (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := 
  sorry

end NUMINAMATH_GPT_trig_expression_evaluation_l1404_140414


namespace NUMINAMATH_GPT_acceptable_colorings_correct_l1404_140407

def acceptableColorings (n : ℕ) : ℕ :=
  (3^(n + 1) + (-1:ℤ)^(n + 1)).natAbs / 2

theorem acceptable_colorings_correct (n : ℕ) :
  acceptableColorings n = (3^(n + 1) + (-1:ℤ)^(n + 1)).natAbs / 2 :=
by
  sorry

end NUMINAMATH_GPT_acceptable_colorings_correct_l1404_140407


namespace NUMINAMATH_GPT_product_of_bc_l1404_140455

theorem product_of_bc
  (b c : Int)
  (h1 : ∀ r, r^2 - r - 1 = 0 → r^5 - b * r - c = 0) :
  b * c = 15 :=
by
  -- We start the proof assuming the conditions
  sorry

end NUMINAMATH_GPT_product_of_bc_l1404_140455


namespace NUMINAMATH_GPT_real_solutions_of_fraction_eqn_l1404_140493

theorem real_solutions_of_fraction_eqn (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 7) :
  ( x = 3 + Real.sqrt 3 ∨ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5 ) ↔
    ((x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)) / ((x - 3) * (x - 7) * (x - 3)) = 1 :=
sorry

end NUMINAMATH_GPT_real_solutions_of_fraction_eqn_l1404_140493


namespace NUMINAMATH_GPT_bag_cost_is_2_l1404_140408

-- Define the inputs and conditions
def carrots_per_day := 1
def days_per_year := 365
def carrots_per_bag := 5
def yearly_spending := 146

-- The final goal is to find the cost per bag
def cost_per_bag := yearly_spending / ((carrots_per_day * days_per_year) / carrots_per_bag)

-- Prove that the cost per bag is $2
theorem bag_cost_is_2 : cost_per_bag = 2 := by
  -- Using sorry to complete the proof
  sorry

end NUMINAMATH_GPT_bag_cost_is_2_l1404_140408


namespace NUMINAMATH_GPT_roger_earned_54_dollars_l1404_140436

-- Definitions based on problem conditions
def lawns_had : ℕ := 14
def lawns_forgot : ℕ := 8
def earn_per_lawn : ℕ := 9

-- The number of lawns actually mowed
def lawns_mowed : ℕ := lawns_had - lawns_forgot

-- The amount of money earned
def money_earned : ℕ := lawns_mowed * earn_per_lawn

-- Proof statement: Roger actually earned 54 dollars
theorem roger_earned_54_dollars : money_earned = 54 := sorry

end NUMINAMATH_GPT_roger_earned_54_dollars_l1404_140436


namespace NUMINAMATH_GPT_circle_center_radius_l1404_140494

theorem circle_center_radius : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  center = (2, 0) ∧ radius = 2 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4 * x = 0 ↔ (x - 2)^2 + y^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_l1404_140494


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1404_140491

theorem problem1 (x : ℝ) : x^2 - 2 * x + 1 = 0 ↔ x = 1 := 
by sorry

theorem problem2 (x : ℝ) : x^2 + 2 * x - 3 = 0 ↔ x = 1 ∨ x = -3 :=
by sorry

theorem problem3 (x : ℝ) : 2 * x^2 + 5 * x - 1 = 0 ↔ x = (-5 + Real.sqrt 33) / 4 ∨ x = (-5 - Real.sqrt 33) / 4 :=
by sorry

theorem problem4 (x : ℝ) : 2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1404_140491


namespace NUMINAMATH_GPT_epsilon_max_success_ratio_l1404_140447

theorem epsilon_max_success_ratio :
  ∃ (x y z w u v: ℕ), 
  (y ≠ 350) ∧
  0 < x ∧ 0 < z ∧ 0 < u ∧ 
  x < y ∧ z < w ∧ u < v ∧
  x + z + u < y + w + v ∧
  y + w + v = 800 ∧
  (x / y : ℚ) < (210 / 350 : ℚ) ∧ 
  (z / w : ℚ) < (delta_day_2_ratio) ∧ 
  (u / v : ℚ) < (delta_day_3_ratio) ∧ 
  (x + z + u) / 800 = (789 / 800 : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_epsilon_max_success_ratio_l1404_140447


namespace NUMINAMATH_GPT_trig_identity_l1404_140484

theorem trig_identity : 
  Real.sin (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) + 
  Real.cos (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l1404_140484


namespace NUMINAMATH_GPT_probability_two_or_fewer_distinct_digits_l1404_140400

def digits : Set ℕ := {1, 2, 3}

def total_3_digit_numbers : ℕ := 27

def distinct_3_digit_numbers : ℕ := 6

def at_most_two_distinct_numbers : ℕ := total_3_digit_numbers - distinct_3_digit_numbers

theorem probability_two_or_fewer_distinct_digits :
  (at_most_two_distinct_numbers : ℚ) / total_3_digit_numbers = 7 / 9 := by
  sorry

end NUMINAMATH_GPT_probability_two_or_fewer_distinct_digits_l1404_140400


namespace NUMINAMATH_GPT_monica_tiles_l1404_140452

theorem monica_tiles (room_length : ℕ) (room_width : ℕ) (border_tile_size : ℕ) (inner_tile_size : ℕ) 
  (border_tiles : ℕ) (inner_tiles : ℕ) (total_tiles : ℕ) :
  room_length = 24 ∧ room_width = 18 ∧ border_tile_size = 2 ∧ inner_tile_size = 3 ∧ 
  border_tiles = 38 ∧ inner_tiles = 32 → total_tiles = 70 :=
by {
  sorry
}

end NUMINAMATH_GPT_monica_tiles_l1404_140452


namespace NUMINAMATH_GPT_rectangle_length_width_l1404_140497

theorem rectangle_length_width 
  (x y : ℚ)
  (h1 : x - 5 = y + 2)
  (h2 : x * y = (x - 5) * (y + 2)) :
  x = 25 / 3 ∧ y = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_width_l1404_140497


namespace NUMINAMATH_GPT_lawn_mowing_rate_l1404_140413

-- Definitions based on conditions
def total_hours_mowed : ℕ := 2 * 7
def money_left_after_expenses (R : ℕ) : ℕ := (14 * R) / 4

-- The problem statement
theorem lawn_mowing_rate (h : money_left_after_expenses R = 49) : R = 14 := 
sorry

end NUMINAMATH_GPT_lawn_mowing_rate_l1404_140413


namespace NUMINAMATH_GPT_boxes_of_apples_l1404_140451

theorem boxes_of_apples (n_crates apples_per_crate rotten_apples apples_per_box : ℕ) 
  (h1 : n_crates = 12) 
  (h2 : apples_per_crate = 42)
  (h3: rotten_apples = 4) 
  (h4 : apples_per_box = 10) : 
  (n_crates * apples_per_crate - rotten_apples) / apples_per_box = 50 :=
by
  sorry

end NUMINAMATH_GPT_boxes_of_apples_l1404_140451


namespace NUMINAMATH_GPT_average_apples_per_hour_l1404_140433

theorem average_apples_per_hour (total_apples : ℝ) (total_hours : ℝ) (h1 : total_apples = 5.0) (h2 : total_hours = 3.0) : total_apples / total_hours = 1.67 :=
  sorry

end NUMINAMATH_GPT_average_apples_per_hour_l1404_140433


namespace NUMINAMATH_GPT_min_double_rooms_needed_min_triple_rooms_needed_with_discount_l1404_140424

-- Define the conditions 
def double_room_price : ℕ := 200
def triple_room_price : ℕ := 250
def total_students : ℕ := 50
def male_students : ℕ := 27
def female_students : ℕ := 23
def discount : ℚ := 0.2
def max_double_rooms : ℕ := 15

-- Define the property for part (1)
theorem min_double_rooms_needed (d : ℕ) (t : ℕ) : 
  2 * d + 3 * t = total_students ∧
  2 * (d - 1) + 3 * t ≠ total_students :=
sorry

-- Define the property for part (2)
theorem min_triple_rooms_needed_with_discount (d : ℕ) (t : ℕ) : 
  d + t = total_students ∧
  d ≤ max_double_rooms ∧
  2 * d + 3 * t = total_students ∧
  (1* (d - 1) + 3 * t ≠ total_students) :=
sorry

end NUMINAMATH_GPT_min_double_rooms_needed_min_triple_rooms_needed_with_discount_l1404_140424


namespace NUMINAMATH_GPT_maximize_profit_l1404_140483

/-- 
The total number of rooms in the hotel 
-/
def totalRooms := 80

/-- 
The initial rent when the hotel is fully booked 
-/
def initialRent := 160

/-- 
The loss in guests for each increase in rent by 20 yuan 
-/
def guestLossPerIncrease := 3

/-- 
The increase in rent 
-/
def increasePer20Yuan := 20

/-- 
The daily service and maintenance cost per occupied room
-/
def costPerOccupiedRoom := 40

/-- 
Maximize profit given the conditions
-/
theorem maximize_profit : 
  ∃ x : ℕ, x = 360 ∧ 
            ∀ y : ℕ,
              (initialRent - costPerOccupiedRoom) * (totalRooms - guestLossPerIncrease * (x - initialRent) / increasePer20Yuan)
              ≥ (y - costPerOccupiedRoom) * (totalRooms - guestLossPerIncrease * (y - initialRent) / increasePer20Yuan) := 
sorry

end NUMINAMATH_GPT_maximize_profit_l1404_140483


namespace NUMINAMATH_GPT_find_x_l1404_140412

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l1404_140412


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1404_140420

theorem solution_set_of_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1404_140420
