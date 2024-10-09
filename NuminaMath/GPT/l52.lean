import Mathlib

namespace expression_evaluation_l52_5293

theorem expression_evaluation :
  (3 : ℝ) + 3 * Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (3 - Real.sqrt 3)) = 4 + 3 * Real.sqrt 3 :=
sorry

end expression_evaluation_l52_5293


namespace martin_speed_l52_5223

theorem martin_speed (distance time : ℝ) (h_distance : distance = 12) (h_time : time = 6) :
  distance / time = 2 :=
by
  rw [h_distance, h_time]
  norm_num

end martin_speed_l52_5223


namespace number_of_sets_B_l52_5242

theorem number_of_sets_B (A : Set ℕ) (hA : A = {1, 2}) :
    ∃ (n : ℕ), n = 4 ∧ (∀ B : Set ℕ, A ∪ B = {1, 2} → B ⊆ A) := sorry

end number_of_sets_B_l52_5242


namespace evaluate_expression_l52_5201

theorem evaluate_expression : (1.2^3 - (0.9^3 / 1.2^2) + 1.08 + 0.9^2 = 3.11175) :=
by
  sorry -- Proof goes here

end evaluate_expression_l52_5201


namespace range_of_m_l52_5238

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (x + m) / (x - 2) + 2 * m / (2 - x) = 3) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_l52_5238


namespace max_perimeter_of_polygons_l52_5217

noncomputable def largest_possible_perimeter (sides1 sides2 sides3 : Nat) (len : Nat) : Nat :=
  (sides1 + sides2 + sides3) * len

theorem max_perimeter_of_polygons
  (a b c : ℕ)
  (h1 : a % 2 = 0)
  (h2 : b % 2 = 0)
  (h3 : c % 2 = 0)
  (h4 : 180 * (a - 2) / a + 180 * (b - 2) / b + 180 * (c - 2) / c = 360)
  (h5 : ∃ (p : ℕ), ∃ q : ℕ, (a = p ∧ c = p ∧ a = q ∨ a = q ∧ b = p ∨ b = q ∧ c = p))
  : largest_possible_perimeter a b c 2 = 24 := 
sorry

end max_perimeter_of_polygons_l52_5217


namespace pipes_fill_cistern_together_time_l52_5256

theorem pipes_fill_cistern_together_time
  (t : ℝ)
  (h1 : t * (1 / 12 + 1 / 15) + 6 * (1 / 15) = 1) : 
  t = 4 := 
by
  -- Proof is omitted here as instructed
  sorry

end pipes_fill_cistern_together_time_l52_5256


namespace birds_count_l52_5239

theorem birds_count (N B : ℕ) 
  (h1 : B = 5 * N)
  (h2 : B = N + 360) : 
  B = 450 := by
  sorry

end birds_count_l52_5239


namespace find_DP_l52_5237

theorem find_DP (AP BP CP DP : ℚ) (h1 : AP = 4) (h2 : BP = 6) (h3 : CP = 9) (h4 : AP * BP = CP * DP) :
  DP = 8 / 3 :=
by
  rw [h1, h2, h3] at h4
  sorry

end find_DP_l52_5237


namespace aluminum_carbonate_weight_l52_5208

-- Define the atomic weights
def Al : ℝ := 26.98
def C : ℝ := 12.01
def O : ℝ := 16.00

-- Define the molecular weight of aluminum carbonate
def molecularWeightAl2CO3 : ℝ := (2 * Al) + (3 * C) + (9 * O)

-- Define the number of moles
def moles : ℝ := 5

-- Calculate the total weight of 5 moles of aluminum carbonate
def totalWeight : ℝ := moles * molecularWeightAl2CO3

-- Statement to prove
theorem aluminum_carbonate_weight : totalWeight = 1169.95 :=
by {
  sorry
}

end aluminum_carbonate_weight_l52_5208


namespace intersection_line_canonical_equation_l52_5240

def plane1 (x y z : ℝ) : Prop := 6 * x - 7 * y - z - 2 = 0
def plane2 (x y z : ℝ) : Prop := x + 7 * y - 4 * z - 5 = 0
def canonical_equation (x y z : ℝ) : Prop := 
  (x - 1) / 35 = (y - 4 / 7) / 23 ∧ (y - 4 / 7) / 23 = z / 49

theorem intersection_line_canonical_equation (x y z : ℝ) :
  plane1 x y z → plane2 x y z → canonical_equation x y z :=
by
  intros h1 h2
  unfold plane1 at h1
  unfold plane2 at h2
  unfold canonical_equation
  sorry

end intersection_line_canonical_equation_l52_5240


namespace probability_same_group_l52_5231

noncomputable def num_students : ℕ := 800
noncomputable def num_groups : ℕ := 4
noncomputable def group_size : ℕ := num_students / num_groups
noncomputable def amy := 0
noncomputable def ben := 1
noncomputable def clara := 2

theorem probability_same_group : ∃ p : ℝ, p = 1 / 16 :=
by
  let P_ben_with_amy : ℝ := group_size / num_students
  let P_clara_with_amy : ℝ := group_size / num_students
  let P_all_same := P_ben_with_amy * P_clara_with_amy
  use P_all_same
  sorry

end probability_same_group_l52_5231


namespace no_roots_ge_two_l52_5232

theorem no_roots_ge_two (x : ℝ) (h : x ≥ 2) : 4 * x^3 - 5 * x^2 - 6 * x + 3 ≠ 0 := by
  sorry

end no_roots_ge_two_l52_5232


namespace relationship_a_e_l52_5233

theorem relationship_a_e (a : ℝ) (h : 0 < a ∧ a < 1) : a < Real.exp a - 1 ∧ Real.exp a - 1 < a ^ Real.exp 1 := by
  sorry

end relationship_a_e_l52_5233


namespace A_profit_share_l52_5218

theorem A_profit_share (A_shares : ℚ) (B_shares : ℚ) (C_shares : ℚ) (D_shares : ℚ) (total_profit : ℚ) (A_profit : ℚ) :
  A_shares = 1/3 → B_shares = 1/4 → C_shares = 1/5 → 
  D_shares = 1 - (A_shares + B_shares + C_shares) → total_profit = 2445 → A_profit = 815 →
  A_shares * total_profit = A_profit :=
by sorry

end A_profit_share_l52_5218


namespace expand_and_simplify_l52_5286

theorem expand_and_simplify (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := 
by
  sorry

end expand_and_simplify_l52_5286


namespace solution_set_inequality_l52_5271

theorem solution_set_inequality : 
  {x : ℝ | abs ((x - 3) / x) > ((x - 3) / x)} = {x : ℝ | 0 < x ∧ x < 3} :=
sorry

end solution_set_inequality_l52_5271


namespace contrapositive_of_happy_people_possess_it_l52_5288

variable (P Q : Prop)

theorem contrapositive_of_happy_people_possess_it
  (h : P → Q) : ¬ Q → ¬ P := by
  intro hq
  intro p
  apply hq
  apply h
  exact p

#check contrapositive_of_happy_people_possess_it

end contrapositive_of_happy_people_possess_it_l52_5288


namespace instantaneous_velocity_at_3_l52_5213

-- Define the position function s(t)
def s (t : ℝ) : ℝ := 1 - t + t^2

-- The main statement we need to prove
theorem instantaneous_velocity_at_3 : (deriv s 3) = 5 :=
by 
  -- The theorem requires a proof which we mark as sorry for now.
  sorry

end instantaneous_velocity_at_3_l52_5213


namespace total_students_suggestion_l52_5216

theorem total_students_suggestion :
  let m := 324
  let b := 374
  let t := 128
  m + b + t = 826 := by
  sorry

end total_students_suggestion_l52_5216


namespace seventh_observation_value_l52_5266

def average_initial_observations (S : ℝ) (n : ℕ) : Prop :=
  S / n = 13

def total_observations (n : ℕ) : Prop :=
  n + 1 = 7

def new_average (S : ℝ) (x : ℝ) (n : ℕ) : Prop :=
  (S + x) / (n + 1) = 12

theorem seventh_observation_value (S : ℝ) (n : ℕ) (x : ℝ) :
  average_initial_observations S n →
  total_observations n →
  new_average S x n →
  x = 6 :=
by
  intros h1 h2 h3
  sorry

end seventh_observation_value_l52_5266


namespace exists_n_not_perfect_square_l52_5200

theorem exists_n_not_perfect_square (a b : ℤ) (h1 : a > 1) (h2 : b > 1) (h3 : a ≠ b) : 
  ∃ (n : ℕ), (n > 0) ∧ ¬∃ (k : ℤ), (a^n - 1) * (b^n - 1) = k^2 :=
by sorry

end exists_n_not_perfect_square_l52_5200


namespace evaluate_expression_l52_5246

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7

theorem evaluate_expression : ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = (7 / 49) :=
by
  sorry

end evaluate_expression_l52_5246


namespace no_neighboring_beads_same_color_probability_l52_5262

theorem no_neighboring_beads_same_color_probability : 
  let total_beads := 9
  let count_red := 4
  let count_white := 3
  let count_blue := 2
  let total_permutations := Nat.factorial total_beads / (Nat.factorial count_red * Nat.factorial count_white * Nat.factorial count_blue)
  ∃ valid_permutations : ℕ,
  valid_permutations = 100 ∧
  valid_permutations / total_permutations = 5 / 63 := by
  sorry

end no_neighboring_beads_same_color_probability_l52_5262


namespace real_z9_count_l52_5249

theorem real_z9_count (z : ℂ) (hz : z^18 = 1) : 
  (∃! z : ℂ, z^18 = 1 ∧ (z^9).im = 0) :=
sorry

end real_z9_count_l52_5249


namespace find_e_l52_5290

-- Define the conditions and state the theorem.
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) 
  (h1: ∃ a b c : ℝ, (a + b + c)/3 = -3 ∧ a * b * c = -3 ∧ 3 + d + e + f = -3)
  (h2: Q 0 d e f = 9) : e = -42 :=
by
  sorry

end find_e_l52_5290


namespace expand_expression_l52_5207

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l52_5207


namespace day_of_week_2_2312_wednesday_l52_5277

def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ ((y % 4 = 0) ∧ (y % 100 ≠ 0))

theorem day_of_week_2_2312_wednesday (birth_year : ℕ) (birth_day : String) 
  (h1 : birth_year = 2312 - 300)
  (h2 : birth_day = "Wednesday") :
  "Monday" = "Monday" :=
sorry

end day_of_week_2_2312_wednesday_l52_5277


namespace find_ratio_of_geometric_sequence_l52_5270

open Real

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def arithmetic_sequence (a1 a2 a3 : ℝ) : Prop :=
  2 * a2 = a1 + a3

theorem find_ratio_of_geometric_sequence 
  {a : ℕ → ℝ} {q : ℝ}
  (h_pos : ∀ n, 0 < a n)
  (h_geo : geometric_sequence a q)
  (h_arith : arithmetic_sequence (a 1) ((1/2) * a 3) (2 * a 2)) :
  (a 10) / (a 8) = 3 + 2 * sqrt 2 :=
sorry

end find_ratio_of_geometric_sequence_l52_5270


namespace smallest_perfect_square_gt_100_has_odd_number_of_factors_l52_5279

theorem smallest_perfect_square_gt_100_has_odd_number_of_factors : 
  ∃ n : ℕ, (n > 100) ∧ (∃ k : ℕ, n = k * k) ∧ (∀ m > 100, ∃ t : ℕ, m = t * t → n ≤ m) := 
sorry

end smallest_perfect_square_gt_100_has_odd_number_of_factors_l52_5279


namespace distribute_candy_bars_l52_5212

theorem distribute_candy_bars (candies bags : ℕ) (h1 : candies = 15) (h2 : bags = 5) :
  candies / bags = 3 :=
by
  sorry

end distribute_candy_bars_l52_5212


namespace coefficient_of_term_x7_in_expansion_l52_5254

theorem coefficient_of_term_x7_in_expansion:
  let general_term (r : ℕ) := (Nat.choose 6 r) * (2 : ℤ)^(6 - r) * (-1 : ℤ)^r * (x : ℤ)^(12 - (5 * r) / 2)
  ∃ r : ℕ, 12 - (5 * r) / 2 = 7 ∧ (Nat.choose 6 r) * (2 : ℤ)^(6 - r) * (-1 : ℤ)^r = 240 := 
sorry

end coefficient_of_term_x7_in_expansion_l52_5254


namespace inscribed_square_sum_c_d_eq_200689_l52_5285

theorem inscribed_square_sum_c_d_eq_200689 :
  ∃ (c d : ℕ), Nat.gcd c d = 1 ∧ (∃ x : ℚ, x = (c : ℚ) / (d : ℚ) ∧ 
    let a := 48
    let b := 55
    let longest_side := 73
    let s := (a + b + longest_side) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - longest_side))
    area = 1320 ∧ x = 192720 / 7969 ∧ c + d = 200689) :=
sorry

end inscribed_square_sum_c_d_eq_200689_l52_5285


namespace value_of_expression_l52_5244

theorem value_of_expression (x : ℤ) (h : x^2 = 1369) : (x + 1) * (x - 1) = 1368 := 
by 
  sorry

end value_of_expression_l52_5244


namespace Vasya_numbers_l52_5253

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l52_5253


namespace cherry_trees_leaves_l52_5247

-- Define the original number of trees
def original_num_trees : ℕ := 7

-- Define the number of trees actually planted
def actual_num_trees : ℕ := 2 * original_num_trees

-- Define the number of leaves each tree drops
def leaves_per_tree : ℕ := 100

-- Define the total number of leaves that fall
def total_leaves : ℕ := actual_num_trees * leaves_per_tree

-- Theorem statement for the problem
theorem cherry_trees_leaves : total_leaves = 1400 := by
  sorry

end cherry_trees_leaves_l52_5247


namespace function_classification_l52_5269

theorem function_classification {f : ℝ → ℝ} 
    (h : ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) : 
    ∀ x : ℝ, f x = 0 ∨ f x = 1 :=
by
  sorry

end function_classification_l52_5269


namespace find_n_lcm_l52_5227

theorem find_n_lcm (m n : ℕ) (h1 : Nat.lcm m n = 690) (h2 : n ≥ 100) (h3 : n < 1000) (h4 : ¬ (3 ∣ n)) (h5 : ¬ (2 ∣ m)) : n = 230 :=
sorry

end find_n_lcm_l52_5227


namespace ivy_covering_the_tree_l52_5283

def ivy_stripped_per_day := 6
def ivy_grows_per_night := 2
def days_to_strip := 10
def net_ivy_stripped_per_day := ivy_stripped_per_day - ivy_grows_per_night

theorem ivy_covering_the_tree : net_ivy_stripped_per_day * days_to_strip = 40 := by
  have h1 : net_ivy_stripped_per_day = 4 := by
    unfold net_ivy_stripped_per_day
    rfl
  rw [h1]
  show 4 * 10 = 40
  rfl

end ivy_covering_the_tree_l52_5283


namespace translation_is_elevator_l52_5220

-- Definitions representing the conditions
def P_A : Prop := true  -- The movement of elevators constitutes translation.
def P_B : Prop := false -- Swinging on a swing does not constitute translation.
def P_C : Prop := false -- Closing an open textbook does not constitute translation.
def P_D : Prop := false -- The swinging of a pendulum does not constitute translation.

-- The goal is to prove that Option A is the phenomenon that belongs to translation
theorem translation_is_elevator : P_A ∧ ¬P_B ∧ ¬P_C ∧ ¬P_D :=
by
  sorry -- proof not required

end translation_is_elevator_l52_5220


namespace water_needed_l52_5211

-- Definitions as per conditions
def heavy_wash : ℕ := 20
def regular_wash : ℕ := 10
def light_wash : ℕ := 2
def extra_light_wash (bleach : ℕ) : ℕ := bleach * light_wash

def num_heavy_washes : ℕ := 2
def num_regular_washes : ℕ := 3
def num_light_washes : ℕ := 1
def num_bleached_loads : ℕ := 2

-- Function to calculate total water usage
def total_water_used : ℕ :=
  (num_heavy_washes * heavy_wash) +
  (num_regular_washes * regular_wash) +
  (num_light_washes * light_wash) + 
  (extra_light_wash num_bleached_loads)

-- Theorem to be proved
theorem water_needed : total_water_used = 76 := by
  sorry

end water_needed_l52_5211


namespace tetrahedron_dihedral_face_areas_l52_5225

variables {S₁ S₂ a b : ℝ} {α φ : ℝ}

theorem tetrahedron_dihedral_face_areas :
  S₁^2 + S₂^2 - 2 * S₁ * S₂ * Real.cos α = (a * b * Real.sin φ / 4)^2 :=
sorry

end tetrahedron_dihedral_face_areas_l52_5225


namespace fisherman_gets_14_tunas_every_day_l52_5230

-- Define the conditions
def red_snappers_per_day := 8
def cost_per_red_snapper := 3
def cost_per_tuna := 2
def total_earnings_per_day := 52

-- Define the hypothesis
def total_earnings_from_red_snappers := red_snappers_per_day * cost_per_red_snapper  -- $24
def total_earnings_from_tunas := total_earnings_per_day - total_earnings_from_red_snappers -- $28
def number_of_tunas := total_earnings_from_tunas / cost_per_tuna -- 14

-- Lean statement to verify
theorem fisherman_gets_14_tunas_every_day : number_of_tunas = 14 :=
by 
  sorry

end fisherman_gets_14_tunas_every_day_l52_5230


namespace max_value_frac_l52_5209

theorem max_value_frac (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    ∃ (c : ℝ), c = 1/4 ∧ (∀ (x y z : ℝ), (0 < x) → (0 < y) → (0 < z) → (xyz * (x + y + z)) / ((x + z)^2 * (z + y)^2) ≤ c) := 
by
  sorry

end max_value_frac_l52_5209


namespace exponent_equality_l52_5243

theorem exponent_equality (y : ℕ) (z : ℕ) (h1 : 16 ^ y = 4 ^ z) (h2 : y = 8) : z = 16 := by
  sorry

end exponent_equality_l52_5243


namespace x_intercepts_of_parabola_l52_5289

theorem x_intercepts_of_parabola : 
  (∃ y : ℝ, x = -3 * y^2 + 2 * y + 2) → ∃ y : ℝ, y = 0 ∧ x = 2 ∧ ∀ y' ≠ 0, x ≠ -3 * y'^2 + 2 * y' + 2 :=
by
  sorry

end x_intercepts_of_parabola_l52_5289


namespace men_in_first_group_l52_5226

theorem men_in_first_group (M : ℕ) (h1 : M * 18 * 6 = 15 * 12 * 6) : M = 10 :=
by
  sorry

end men_in_first_group_l52_5226


namespace sequence_formula_l52_5210

-- Define the properties of the sequence
axiom seq_prop_1 (a : ℕ → ℝ) (m n : ℕ) (h : m > n) : a (m - n) = a m - a n

axiom seq_increasing (a : ℕ → ℝ) : ∀ n m : ℕ, n < m → a n < a m

-- Formulate the theorem to prove the general sequence formula
theorem sequence_formula (a : ℕ → ℝ) (h1 : ∀ m n : ℕ, m > n → a (m - n) = a m - a n)
    (h2 : ∀ n m : ℕ, n < m → a n < a m) :
    ∃ k > 0, ∀ n, a n = k * n :=
sorry

end sequence_formula_l52_5210


namespace maximum_value_of_function_l52_5296

theorem maximum_value_of_function : ∃ x, x > (1 : ℝ) ∧ (∀ y, y > 1 → (x + 1 / (x - 1) ≥ y + 1 / (y - 1))) ∧ (x = 2 ∧ (x + 1 / (x - 1) = 3)) :=
sorry

end maximum_value_of_function_l52_5296


namespace johns_coin_collection_value_l52_5267

theorem johns_coin_collection_value :
  ∀ (n : ℕ) (value : ℕ), n = 24 → value = 20 → 
  ((n/3) * (value/8)) = 60 :=
by
  intro n value n_eq value_eq
  sorry

end johns_coin_collection_value_l52_5267


namespace range_g_l52_5295

variable (x : ℝ)
noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

theorem range_g (y : ℝ) : 
  (∃ x, g x = y) ↔ y > 0 :=
by
  sorry

end range_g_l52_5295


namespace downstream_distance_l52_5264

theorem downstream_distance (speed_boat : ℝ) (speed_current : ℝ) (time_minutes : ℝ) (distance : ℝ) :
  speed_boat = 20 ∧ speed_current = 5 ∧ time_minutes = 24 ∧ distance = 10 →
  (speed_boat + speed_current) * (time_minutes / 60) = distance :=
by
  sorry

end downstream_distance_l52_5264


namespace game_result_l52_5297

theorem game_result (a : ℤ) : ((2 * a + 6) / 2 - a = 3) :=
by
  sorry

end game_result_l52_5297


namespace percentage_of_students_owning_only_cats_l52_5214

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ) (students_owning_dogs : ℕ) (students_owning_cats : ℕ) (students_owning_both : ℕ)
  (h1 : total_students = 500) (h2 : students_owning_dogs = 200) (h3 : students_owning_cats = 100) (h4 : students_owning_both = 50) :
  ((students_owning_cats - students_owning_both) * 100 / total_students) = 10 :=
by
  -- Placeholder for proof
  sorry

end percentage_of_students_owning_only_cats_l52_5214


namespace amount_for_second_shop_l52_5222

-- Definitions based on conditions
def books_from_first_shop : Nat := 65
def amount_first_shop : Float := 1160.0
def books_from_second_shop : Nat := 50
def avg_price_per_book : Float := 18.08695652173913
def total_books : Nat := books_from_first_shop + books_from_second_shop
def total_amount_spent : Float := avg_price_per_book * (total_books.toFloat)

-- The Lean statement to prove
theorem amount_for_second_shop : total_amount_spent - amount_first_shop = 920.0 := by
  sorry

end amount_for_second_shop_l52_5222


namespace multiplication_scaling_l52_5280

theorem multiplication_scaling (h : 28 * 15 = 420) : 
  (28 / 10) * (15 / 10) = 2.8 * 1.5 ∧ 
  (28 / 100) * 1.5 = 0.28 * 1.5 ∧ 
  (28 / 1000) * (15 / 100) = 0.028 * 0.15 :=
by 
  sorry

end multiplication_scaling_l52_5280


namespace power_difference_mod_7_l52_5221

theorem power_difference_mod_7 :
  (45^2011 - 23^2011) % 7 = 5 := by
  have h45 : 45 % 7 = 3 := by norm_num
  have h23 : 23 % 7 = 2 := by norm_num
  sorry

end power_difference_mod_7_l52_5221


namespace store_loses_out_l52_5215

theorem store_loses_out (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (x y : ℝ)
    (h1 : a = b * x) (h2 : b = a * y) : x + y > 2 :=
by
  sorry

end store_loses_out_l52_5215


namespace temperature_difference_l52_5281

theorem temperature_difference (t_low t_high : ℝ) (h_low : t_low = -2) (h_high : t_high = 5) :
  t_high - t_low = 7 :=
by
  rw [h_low, h_high]
  norm_num

end temperature_difference_l52_5281


namespace joint_savings_account_total_l52_5292

theorem joint_savings_account_total :
  let kimmie_earnings : ℕ := 450
  let zahra_earnings : ℕ := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings : ℕ := kimmie_earnings / 2
  let zahra_savings : ℕ := zahra_earnings / 2
  kimmie_savings + zahra_savings = 375 :=
by
  let kimmie_earnings := 450
  let zahra_earnings := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings := kimmie_earnings / 2
  let zahra_savings := zahra_earnings / 2
  have h : kimmie_savings + zahra_savings = 375 := sorry
  exact h

end joint_savings_account_total_l52_5292


namespace one_equation_does_not_pass_origin_l52_5260

def passes_through_origin (eq : ℝ → ℝ) : Prop := eq 0 = 0

def equation1 (x : ℝ) : ℝ := x^4 + 1
def equation2 (x : ℝ) : ℝ := x^4 + x
def equation3 (x : ℝ) : ℝ := x^4 + x^2
def equation4 (x : ℝ) : ℝ := x^4 + x^3

theorem one_equation_does_not_pass_origin :
  (¬ passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  ¬ passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  ¬ passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  ¬ passes_through_origin equation4) :=
sorry

end one_equation_does_not_pass_origin_l52_5260


namespace smallest_n_l52_5275

theorem smallest_n (m l n : ℕ) :
  (∃ m : ℕ, 2 * n = m ^ 4) ∧ (∃ l : ℕ, 3 * n = l ^ 6) → n = 1944 :=
by
  sorry

end smallest_n_l52_5275


namespace simplify_evaluate_expression_l52_5282

theorem simplify_evaluate_expression (a b : ℤ) (h1 : a = -1) (h2 : b = 2) :
  3 * (a^2 * b + a * b^2) - 2 * (a^2 * b - 1) - 2 * (a * b^2) - 2 = -2 :=
by
  sorry

end simplify_evaluate_expression_l52_5282


namespace circle_equation_l52_5248

-- Define the given conditions
def point_P : ℝ × ℝ := (-1, 0)
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def center_C : ℝ × ℝ := (1, 2)

-- Define the required equation of the circle and the claim
def required_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

-- The Lean theorem statement
theorem circle_equation :
  ∃ (x y : ℝ), required_circle x y :=
sorry

end circle_equation_l52_5248


namespace inequality_solution_l52_5251

theorem inequality_solution (x : ℝ) (h : x ≠ 1) : (x + 1) * (x + 3) / (x - 1)^2 ≤ 0 ↔ (-3 ≤ x ∧ x ≤ -1) :=
by
  sorry

end inequality_solution_l52_5251


namespace machine_C_works_in_6_hours_l52_5241

theorem machine_C_works_in_6_hours :
  ∃ C : ℝ, (0 < C ∧ (1/4 + 1/12 + 1/C = 1/2)) → C = 6 :=
by
  sorry

end machine_C_works_in_6_hours_l52_5241


namespace negation_proof_l52_5255

def P (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

theorem negation_proof : (¬(∀ x : ℝ, P x)) ↔ (∃ x : ℝ, ¬(P x)) :=
by sorry

end negation_proof_l52_5255


namespace ratio_of_andy_age_in_5_years_to_rahim_age_l52_5229

def rahim_age_now : ℕ := 6
def andy_age_now : ℕ := rahim_age_now + 1
def andy_age_in_5_years : ℕ := andy_age_now + 5
def ratio (a b : ℕ) : ℕ := a / b

theorem ratio_of_andy_age_in_5_years_to_rahim_age : ratio andy_age_in_5_years rahim_age_now = 2 := by
  sorry

end ratio_of_andy_age_in_5_years_to_rahim_age_l52_5229


namespace celestia_badges_l52_5276

theorem celestia_badges (H L C : ℕ) (total_badges : ℕ) (h1 : H = 14) (h2 : L = 17) (h3 : total_badges = 83) (h4 : H + L + C = total_badges) : C = 52 :=
by
  sorry

end celestia_badges_l52_5276


namespace haman_dropped_trays_l52_5291

def initial_trays_to_collect : ℕ := 10
def additional_trays : ℕ := 7
def eggs_sold : ℕ := 540
def eggs_per_tray : ℕ := 30

theorem haman_dropped_trays :
  ∃ dropped_trays : ℕ,
  (initial_trays_to_collect + additional_trays - dropped_trays)*eggs_per_tray = eggs_sold → dropped_trays = 8 :=
sorry

end haman_dropped_trays_l52_5291


namespace function_solution_l52_5299

theorem function_solution (f : ℝ → ℝ) (α : ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) →
  (∀ x : ℝ, f x = x ^ 3 + α * x) :=
by
  sorry

end function_solution_l52_5299


namespace initial_water_amount_l52_5294

theorem initial_water_amount (x : ℝ) (h : x + 6.8 = 9.8) : x = 3 := 
by
  sorry

end initial_water_amount_l52_5294


namespace Sandy_marks_per_correct_sum_l52_5235

theorem Sandy_marks_per_correct_sum
  (x : ℝ)  -- number of marks Sandy gets for each correct sum
  (marks_lost_per_incorrect : ℝ := 2)  -- 2 marks lost for each incorrect sum, default value is 2
  (total_attempts : ℤ := 30)  -- Sandy attempts 30 sums, default value is 30
  (total_marks : ℝ := 60)  -- Sandy obtains 60 marks, default value is 60
  (correct_sums : ℤ := 24)  -- Sandy got 24 sums correct, default value is 24
  (incorrect_sums := total_attempts - correct_sums) -- incorrect sums are the remaining attempts
  (marks_from_correct := correct_sums * x) -- total marks from the correct sums
  (marks_lost_from_incorrect := incorrect_sums * marks_lost_per_incorrect) -- total marks lost from the incorrect sums
  (total_marks_obtained := marks_from_correct - marks_lost_from_incorrect) -- total marks obtained

  -- The theorem states that x must be 3 given the conditions above
  : total_marks_obtained = total_marks → x = 3 := by sorry

end Sandy_marks_per_correct_sum_l52_5235


namespace marble_prob_l52_5284

theorem marble_prob (T : ℕ) (hT1 : T > 12) 
  (hP : ((T - 12) / T : ℚ) * ((T - 12) / T) = 36 / 49) : T = 84 :=
sorry

end marble_prob_l52_5284


namespace range_of_a_l52_5261

noncomputable def f (a x : ℝ) := Real.logb (1 / 2) (x^2 - a * x - a)

theorem range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, f a x ∈ Set.univ) ∧ 
            (∀ x1 x2 : ℝ, -3 < x1 ∧ x1 < x2 ∧ x2 < 1 - Real.sqrt 3 → f a x1 < f a x2)) → 
  (0 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l52_5261


namespace find_n_l52_5273

noncomputable def imaginary_unit : ℂ := Complex.I

theorem find_n (n : ℝ) (h : (2 : ℂ) / (1 - imaginary_unit) = 1 + n * imaginary_unit) : n = 1 :=
sorry

end find_n_l52_5273


namespace length_of_cube_side_l52_5298

theorem length_of_cube_side (SA : ℝ) (h₀ : SA = 600) (h₁ : SA = 6 * a^2) : a = 10 := by
  sorry

end length_of_cube_side_l52_5298


namespace element_of_M_l52_5234

def M : Set (ℕ × ℕ) := { (2, 3) }

theorem element_of_M : (2, 3) ∈ M :=
by
  sorry

end element_of_M_l52_5234


namespace part_a_least_moves_part_b_least_moves_l52_5252

def initial_position : Nat := 0
def total_combinations : Nat := 10^6
def excluded_combinations : List Nat := [0, 10^5, 2 * 10^5, 3 * 10^5, 4 * 10^5, 5 * 10^5, 6 * 10^5, 7 * 10^5, 8 * 10^5, 9 * 10^5]

theorem part_a_least_moves : total_combinations - 1 = 10^6 - 1 := by
  simp [total_combinations, Nat.pow]

theorem part_b_least_moves : total_combinations - excluded_combinations.length = 10^6 - 10 := by
  simp [total_combinations, excluded_combinations, Nat.pow, List.length]

end part_a_least_moves_part_b_least_moves_l52_5252


namespace average_salary_of_all_workers_l52_5204

-- Definitions of conditions
def T : ℕ := 7
def total_workers : ℕ := 56
def W : ℕ := total_workers - T
def A_T : ℕ := 12000
def A_W : ℕ := 6000

-- Definition of total salary and average salary
def total_salary : ℕ := (T * A_T) + (W * A_W)

theorem average_salary_of_all_workers : total_salary / total_workers = 6750 := 
  by sorry

end average_salary_of_all_workers_l52_5204


namespace system_equations_solution_l52_5287

theorem system_equations_solution (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  (1 / x + 1 / y + 1 / z = 3) ∧ 
  (1 / (x * y) + 1 / (y * z) + 1 / (z * x) = 3) ∧ 
  (1 / (x * y * z) = 1) → 
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end system_equations_solution_l52_5287


namespace value_of_expression_l52_5265

theorem value_of_expression (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (3 * x - 4 * y) / z = 1 / 4 := 
by 
  sorry

end value_of_expression_l52_5265


namespace simplify_expression_l52_5203

theorem simplify_expression (x y z : ℝ) : (x + (y + z)) - ((x + z) + y) = 0 :=
by
  sorry

end simplify_expression_l52_5203


namespace percents_multiplication_l52_5272

theorem percents_multiplication :
  let p1 := 0.40
  let p2 := 0.35
  let p3 := 0.60
  let p4 := 0.70
  (p1 * p2 * p3 * p4) * 100 = 5.88 := 
by
  let p1 := 0.40
  let p2 := 0.35
  let p3 := 0.60
  let p4 := 0.70
  sorry

end percents_multiplication_l52_5272


namespace complement_of_A_l52_5219

def A : Set ℝ := {y : ℝ | ∃ (x : ℝ), y = 2^x}

theorem complement_of_A : (Set.compl A) = {y : ℝ | y ≤ 0} :=
by
  sorry

end complement_of_A_l52_5219


namespace curve_points_satisfy_equation_l52_5257

theorem curve_points_satisfy_equation (C : Set (ℝ × ℝ)) (f : ℝ × ℝ → ℝ) :
  (∀ p : ℝ × ℝ, p ∈ C → f p = 0) → (∀ q : ℝ × ℝ, f q ≠ 0 → q ∉ C) :=
by
  intro h₁
  intro q
  intro h₂
  sorry

end curve_points_satisfy_equation_l52_5257


namespace parallel_lines_same_slope_l52_5245

theorem parallel_lines_same_slope (k : ℝ) : 
  (2*x + y + 1 = 0) ∧ (y = k*x + 3) → (k = -2) := 
by
  sorry

end parallel_lines_same_slope_l52_5245


namespace M_inter_N_is_5_l52_5224

/-- Define the sets M and N. -/
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {2, 5, 8}

/-- Prove the intersection of M and N is {5}. -/
theorem M_inter_N_is_5 : M ∩ N = {5} :=
by
  sorry

end M_inter_N_is_5_l52_5224


namespace find_a_l52_5205

def possible_scores : List ℕ := [103, 104, 105, 106, 107, 108, 109, 110]

def is_possible_score (a : ℕ) (n : ℕ) : Prop :=
  ∃ (k8 k0 ka : ℕ), k8 * 8 + ka * a + k0 * 0 = n

def is_impossible_score (a : ℕ) (n : ℕ) : Prop :=
  ¬ is_possible_score a n

theorem find_a : ∀ (a : ℕ), a ≠ 0 → a ≠ 8 →
  (∀ n ∈ possible_scores, is_possible_score a n) →
  is_impossible_score a 83 →
  a = 13 := by
  intros a ha1 ha2 hpossible himpossible
  sorry

end find_a_l52_5205


namespace consecutive_integers_sum_and_difference_l52_5258

theorem consecutive_integers_sum_and_difference (x y : ℕ) 
(h1 : y = x + 1) 
(h2 : x * y = 552) 
: x + y = 47 ∧ y - x = 1 :=
by {
  sorry
}

end consecutive_integers_sum_and_difference_l52_5258


namespace problem_solution_1_problem_solution_2_l52_5278

def Sn (n : ℕ) := n * (n + 2)

def a_n (n : ℕ) := 2 * n + 1

def b_n (n : ℕ) := 2 ^ (n - 1)

def c_n (n : ℕ) := if n % 2 = 1 then 2 / Sn n else b_n n

def T_n (n : ℕ) : ℤ := (Finset.range n).sum (λ i => c_n (i + 1))

theorem problem_solution_1 : 
  ∀ (n : ℕ), a_n n = 2 * n + 1 ∧ b_n n = 2 ^ (n - 1) := 
  by sorry

theorem problem_solution_2 (n : ℕ) : 
  T_n (2 * n) = (2 * n) / (2 * n + 1) + (2 / 3) * (4 ^ n - 1) := 
  by sorry

end problem_solution_1_problem_solution_2_l52_5278


namespace comb_eq_comb_imp_n_eq_18_l52_5228

theorem comb_eq_comb_imp_n_eq_18 {n : ℕ} (h : Nat.choose n 14 = Nat.choose n 4) : n = 18 :=
sorry

end comb_eq_comb_imp_n_eq_18_l52_5228


namespace arithmetic_sequence_sum_l52_5268

variable {α : Type*} [LinearOrderedField α]

def sum_n_terms (a₁ d : α) (n : ℕ) : α :=
  n / 2 * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_sum 
  (a₁ : α) (h : sum_n_terms a₁ 1 4 = 1) :
  sum_n_terms a₁ 1 8 = 18 := by
  sorry

end arithmetic_sequence_sum_l52_5268


namespace max_integers_sum_power_of_two_l52_5236

open Set

/-- Given a finite set of positive integers such that the sum of any two distinct elements is a power of two,
    the cardinality of the set is at most 2. -/
theorem max_integers_sum_power_of_two (S : Finset ℕ) (h_pos : ∀ x ∈ S, 0 < x)
  (h_sum : ∀ {a b : ℕ}, a ∈ S → b ∈ S → a ≠ b → ∃ n : ℕ, a + b = 2^n) : S.card ≤ 2 :=
sorry

end max_integers_sum_power_of_two_l52_5236


namespace binom_12_3_equal_220_l52_5202

theorem binom_12_3_equal_220 : Nat.choose 12 3 = 220 := by sorry

end binom_12_3_equal_220_l52_5202


namespace find_four_digit_number_l52_5250

noncomputable def reverse_num (n : ℕ) : ℕ := -- assume definition to reverse digits
  sorry

theorem find_four_digit_number :
  ∃ (A : ℕ), 1000 ≤ A ∧ A ≤ 9999 ∧ reverse_num (9 * A) = A ∧ 9 * A = reverse_num A ∧ A = 1089 :=
sorry

end find_four_digit_number_l52_5250


namespace sum_of_a_for_one_solution_l52_5206

theorem sum_of_a_for_one_solution (a : ℝ) :
  (∀ x : ℝ, 3 * x^2 + (a + 15) * x + 18 = 0 ↔ (a + 15) ^ 2 - 4 * 3 * 18 = 0) →
  a = -15 + 6 * Real.sqrt 6 ∨ a = -15 - 6 * Real.sqrt 6 → a + (-15 + 6 * Real.sqrt 6) + (-15 - 6 * Real.sqrt 6) = -30 :=
by
  intros h1 h2
  have hsum : (-15 + 6 * Real.sqrt 6) + (-15 - 6 * Real.sqrt 6) = -30 := by linarith [Real.sqrt 6]
  sorry

end sum_of_a_for_one_solution_l52_5206


namespace angle_between_clock_hands_at_7_oclock_l52_5274

theorem angle_between_clock_hands_at_7_oclock
  (complete_circle : ℕ := 360)
  (hours_in_clock : ℕ := 12)
  (degrees_per_hour : ℕ := complete_circle / hours_in_clock)
  (position_hour_12 : ℕ := 12)
  (position_hour_7 : ℕ := 7)
  (hour_difference : ℕ := position_hour_12 - position_hour_7)
  : degrees_per_hour * hour_difference = 150 := by
  sorry

end angle_between_clock_hands_at_7_oclock_l52_5274


namespace transformation_correctness_l52_5259

variable (x x' y y' : ℝ)

-- Conditions
def original_curve : Prop := y^2 = 4
def transformed_curve : Prop := (x'^2)/1 + (y'^2)/4 = 1
def transformation_formula : Prop := (x = 2 * x') ∧ (y = y')

-- Proof Statement
theorem transformation_correctness (h1 : original_curve y) (h2 : transformed_curve x' y') :
  transformation_formula x x' y y' :=
  sorry

end transformation_correctness_l52_5259


namespace additional_boxes_needed_l52_5263

theorem additional_boxes_needed
  (total_chocolates : ℕ)
  (chocolates_not_in_box : ℕ)
  (boxes_filled : ℕ)
  (friend_brought_chocolates : ℕ)
  (chocolates_per_box : ℕ)
  (h1 : total_chocolates = 50)
  (h2 : chocolates_not_in_box = 5)
  (h3 : boxes_filled = 3)
  (h4 : friend_brought_chocolates = 25)
  (h5 : chocolates_per_box = 15) :
  (chocolates_not_in_box + friend_brought_chocolates) / chocolates_per_box = 2 :=
by
  sorry
  
end additional_boxes_needed_l52_5263
