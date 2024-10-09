import Mathlib

namespace lower_right_is_one_l152_15243

def initial_grid : Matrix (Fin 5) (Fin 5) (Option (Fin 5)) :=
![![some 0, none, some 1, none, none],
  ![some 1, some 3, none, none, none],
  ![none, none, none, some 4, none],
  ![none, some 4, none, none, none],
  ![none, none, none, none, none]]

theorem lower_right_is_one 
  (complete_grid : Matrix (Fin 5) (Fin 5) (Fin 5)) 
  (unique_row_col : ∀ i j k, 
      complete_grid i j = complete_grid i k ↔ j = k ∧ 
      complete_grid i j = complete_grid k j ↔ i = k)
  (matches_partial : ∀ i j, ∃ x, 
      initial_grid i j = some x → complete_grid i j = x) :
  complete_grid 4 4 = 0 := 
sorry

end lower_right_is_one_l152_15243


namespace spherical_coordinate_cone_l152_15282

-- Define spherical coordinates
structure SphericalCoordinate :=
  (ρ : ℝ)
  (θ : ℝ)
  (φ : ℝ)

-- Definition to describe the cone condition
def isCone (d : ℝ) (p : SphericalCoordinate) : Prop :=
  p.φ = d

-- The main theorem to state the problem
theorem spherical_coordinate_cone (d : ℝ) :
  ∀ (p : SphericalCoordinate), isCone d p → ∃ (ρ : ℝ), ∃ (θ : ℝ), (p = ⟨ρ, θ, d⟩) := sorry

end spherical_coordinate_cone_l152_15282


namespace find_number_l152_15276

theorem find_number (x : ℕ) (h : (x / 5) - 154 = 6) : x = 800 := by
  sorry

end find_number_l152_15276


namespace quadratic_rewrite_l152_15229

theorem quadratic_rewrite (d e f : ℤ) (h1 : d^2 = 25) (h2 : 2 * d * e = -40) (h3 : e^2 + f = -75) : d * e = -20 := 
by 
  sorry

end quadratic_rewrite_l152_15229


namespace janet_total_earnings_l152_15210

def hourly_wage_exterminator := 70
def hourly_work_exterminator := 20
def sculpture_price_per_pound := 20
def sculpture_1_weight := 5
def sculpture_2_weight := 7

theorem janet_total_earnings :
  (hourly_wage_exterminator * hourly_work_exterminator) +
  (sculpture_price_per_pound * sculpture_1_weight) +
  (sculpture_price_per_pound * sculpture_2_weight) = 1640 := by
  sorry

end janet_total_earnings_l152_15210


namespace smallest_cars_number_l152_15283

theorem smallest_cars_number :
  ∃ N : ℕ, N > 2 ∧ (N % 5 = 2) ∧ (N % 6 = 2) ∧ (N % 7 = 2) ∧ N = 212 := by
  sorry

end smallest_cars_number_l152_15283


namespace total_travel_time_l152_15252

theorem total_travel_time (x y : ℝ) : 
  (x / 50 + y / 70 + 0.5) = (7 * x + 5 * y) / 350 + 0.5 :=
by
  sorry

end total_travel_time_l152_15252


namespace find_sequence_l152_15249

def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → 
    a (n + 1) = (a n * a (n - 1)) / 
               Real.sqrt (a n^2 + a (n - 1)^2 + 1)

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 5

def sequence_property (F : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = Real.sqrt (1 / (Real.exp (F n * Real.log 10) - 1))

theorem find_sequence (a : ℕ → ℝ) (F : ℕ → ℝ) :
  initial_conditions a →
  recurrence_relation a →
  (∀ n : ℕ, n ≥ 2 →
    F (n + 1) = F n + F (n - 1)) →
  sequence_property F a :=
by
  intros h_initial h_recur h_F
  sorry

end find_sequence_l152_15249


namespace tom_chocolates_l152_15257

variable (n : ℕ)

-- Lisa's box holds 64 chocolates and has unit dimensions (1^3 = 1 cubic unit)
def lisa_chocolates := 64
def lisa_volume := 1

-- Tom's box has dimensions thrice Lisa's and hence its volume (3^3 = 27 cubic units)
def tom_volume := 27

-- Number of chocolates Tom's box holds
theorem tom_chocolates : lisa_chocolates * tom_volume = 1728 := by
  -- calculations with known values
  sorry

end tom_chocolates_l152_15257


namespace farmer_spending_l152_15246

theorem farmer_spending (X : ℝ) (hc : 0.80 * X + 0.60 * X = 49) : X = 35 := 
by
  sorry

end farmer_spending_l152_15246


namespace two_lines_in_3d_space_l152_15281

theorem two_lines_in_3d_space : 
  ∀ x y z : ℝ, x^2 + 2 * x * (y + z) + y^2 = z^2 + 2 * z * (y + x) + x^2 → 
  (∃ a : ℝ, y = -z ∧ x = 0) ∨ (∃ b : ℝ, z = - (2 / 3) * x) :=
  sorry

end two_lines_in_3d_space_l152_15281


namespace range_x_when_p_and_q_m_eq_1_range_m_for_not_p_necessary_not_sufficient_q_l152_15285

-- Define the propositions p and q in terms of x and m
def p (x m : ℝ) : Prop := |2 * x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3 * x) / (x + 2) > 0

-- The range of x for p ∧ q when m = 1
theorem range_x_when_p_and_q_m_eq_1 : {x : ℝ | p x 1 ∧ q x} = {x : ℝ | -2 < x ∧ x ≤ 0} :=
by sorry

-- The range of m where ¬p is a necessary but not sufficient condition for q
theorem range_m_for_not_p_necessary_not_sufficient_q : {m : ℝ | ∀ x, ¬p x m → q x} ∩ {m : ℝ | ∃ x, ¬p x m ∧ q x} = {m : ℝ | -3 ≤ m ∧ m ≤ -1/3} :=
by sorry

end range_x_when_p_and_q_m_eq_1_range_m_for_not_p_necessary_not_sufficient_q_l152_15285


namespace multiples_33_between_1_and_300_l152_15208

theorem multiples_33_between_1_and_300 : ∃ (x : ℕ), (∀ n : ℕ, n ≤ 300 → n % x = 0 → n / x ≤ 33) ∧ x = 9 :=
by
  sorry

end multiples_33_between_1_and_300_l152_15208


namespace intersection_of_M_and_N_l152_15236

def M : Set ℕ := {0, 2, 4}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N :
  {x | x ∈ M ∧ x ∈ N} = {0, 4} := by
  sorry

end intersection_of_M_and_N_l152_15236


namespace consecutive_sum_150_l152_15284

theorem consecutive_sum_150 : ∃ (n : ℕ), n ≥ 2 ∧ (∃ a : ℕ, (n * (2 * a + n - 1)) / 2 = 150) :=
sorry

end consecutive_sum_150_l152_15284


namespace value_of_x2_plus_inv_x2_l152_15232

theorem value_of_x2_plus_inv_x2 (x : ℝ) (hx : x ≠ 0) (h : x^4 + 1 / x^4 = 47) : x^2 + 1 / x^2 = 7 :=
sorry

end value_of_x2_plus_inv_x2_l152_15232


namespace max_voters_after_T_l152_15258

theorem max_voters_after_T (x : ℕ) (n : ℕ) (y : ℕ) (T : ℕ)  
  (h1 : x <= 10)
  (h2 : x > 0)
  (h3 : (nx + y) ≤ (n + 1) * (x - 1))
  (h4 : ∀ k, (x - k ≥ 0) ↔ (n ≤ T + 5)) :
  ∃ (m : ℕ), m = 5 := 
sorry

end max_voters_after_T_l152_15258


namespace dawson_marks_l152_15277

theorem dawson_marks :
  ∀ (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) (M : ℕ),
  max_marks = 220 →
  passing_percentage = 30 →
  failed_by = 36 →
  M = (passing_percentage * max_marks / 100) - failed_by →
  M = 30 := by
  intros max_marks passing_percentage failed_by M h_max h_percent h_failed h_M
  rw [h_max, h_percent, h_failed] at h_M
  norm_num at h_M
  exact h_M

end dawson_marks_l152_15277


namespace race_distance_difference_l152_15273

theorem race_distance_difference
  (d : ℕ) (tA tB : ℕ)
  (h_d: d = 80) 
  (h_tA: tA = 20) 
  (h_tB: tB = 25) :
  (d / tA) * tA = d ∧ (d - (d / tB) * tA) = 16 := 
by
  sorry

end race_distance_difference_l152_15273


namespace student_percentage_in_math_l152_15266

theorem student_percentage_in_math (M H T : ℝ) (H_his : H = 84) (H_third : T = 69) (H_avg : (M + H + T) / 3 = 75) : M = 72 :=
by
  sorry

end student_percentage_in_math_l152_15266


namespace untouched_shapes_after_moves_l152_15231

-- Definitions
def num_shapes : ℕ := 12
def num_triangles : ℕ := 3
def num_squares : ℕ := 4
def num_pentagons : ℕ := 5
def total_moves : ℕ := 10
def petya_moves_first : Prop := True
def vasya_strategy : Prop := True  -- Vasya's strategy to minimize untouched shapes
def petya_strategy : Prop := True  -- Petya's strategy to maximize untouched shapes

-- Theorem
theorem untouched_shapes_after_moves : num_shapes = 12 ∧ num_triangles = 3 ∧ num_squares = 4 ∧ num_pentagons = 5 ∧
                                        total_moves = 10 ∧ petya_moves_first ∧ vasya_strategy ∧ petya_strategy → 
                                        num_shapes - 5 = 6 :=
by
  sorry

end untouched_shapes_after_moves_l152_15231


namespace find_principal_l152_15271

theorem find_principal :
  ∃ P r : ℝ, (8820 = P * (1 + r) ^ 2) ∧ (9261 = P * (1 + r) ^ 3) → (P = 8000) :=
by
  sorry

end find_principal_l152_15271


namespace common_integer_root_l152_15253

theorem common_integer_root (a x : ℤ) : (a * x + a = 7) ∧ (3 * x - a = 17) → a = 1 :=
by
    sorry

end common_integer_root_l152_15253


namespace exponent_product_l152_15207

variables {a m n : ℝ}

theorem exponent_product (h1 : a^m = 2) (h2 : a^n = 8) : a^m * a^n = 16 :=
by
  sorry

end exponent_product_l152_15207


namespace abs_eq_zero_solve_l152_15230

theorem abs_eq_zero_solve (a b : ℚ) (h : |a - (1/2 : ℚ)| + |b + 5| = 0) : a + b = -9 / 2 := 
by
  sorry

end abs_eq_zero_solve_l152_15230


namespace decreasing_line_implies_m_half_l152_15296

theorem decreasing_line_implies_m_half (m b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * m - 1) * x₁ + b > (2 * m - 1) * x₂ + b) → m < 1 / 2 :=
by
  intro h
  sorry

end decreasing_line_implies_m_half_l152_15296


namespace carson_clouds_l152_15247

theorem carson_clouds (C D : ℕ) (h1 : D = 3 * C) (h2 : C + D = 24) : C = 6 :=
by
  sorry

end carson_clouds_l152_15247


namespace number_of_truthful_warriors_l152_15286

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l152_15286


namespace largest_integer_n_l152_15219

theorem largest_integer_n (n : ℤ) (h : n^2 - 13 * n + 40 < 0) : n = 7 :=
by
  sorry

end largest_integer_n_l152_15219


namespace puppy_price_l152_15203

theorem puppy_price (P : ℕ) (kittens_price : ℕ) (total_earnings : ℕ) :
  (kittens_price = 2 * 6) → (total_earnings = 17) → (kittens_price + P = total_earnings) → P = 5 :=
by
  intros h1 h2 h3
  sorry

end puppy_price_l152_15203


namespace solve_for_k_and_j_l152_15223

theorem solve_for_k_and_j (k j : ℕ) (h1 : 64 / k = 8) (h2 : k * j = 128) : k = 8 ∧ j = 16 := by
  sorry

end solve_for_k_and_j_l152_15223


namespace dave_paid_4_more_than_doug_l152_15239

theorem dave_paid_4_more_than_doug :
  let slices := 8
  let plain_cost := 8
  let anchovy_additional_cost := 2
  let total_cost := plain_cost + anchovy_additional_cost
  let cost_per_slice := total_cost / slices
  let dave_slices := 5
  let doug_slices := slices - dave_slices
  -- Calculate payments
  let dave_payment := dave_slices * cost_per_slice
  let doug_payment := doug_slices * cost_per_slice
  dave_payment - doug_payment = 4 :=
by
  sorry

end dave_paid_4_more_than_doug_l152_15239


namespace product_eq_5832_l152_15248

-- Define the integers A, B, C, D that satisfy the given conditions.
variables (A B C D : ℕ)

-- Define the conditions in the problem.
def conditions : Prop :=
  (A + B + C + D = 48) ∧
  (A + 3 = B - 3) ∧
  (A + 3 = C * 3) ∧
  (A + 3 = D / 3)

-- State the final theorem we want to prove.
theorem product_eq_5832 : conditions A B C D → A * B * C * D = 5832 :=
by 
  sorry

end product_eq_5832_l152_15248


namespace tyler_meal_combinations_is_720_l152_15299

-- Required imports for permutations and combinations
open Nat
open BigOperators

-- Assumptions based on the problem conditions
def meat_options  := 4
def veg_options := 4
def dessert_options := 5
def bread_options := 3

-- Using combinations and permutations for calculations
def comb(n k : ℕ) := Nat.choose n k
def perm(n k : ℕ) := n.factorial / (n - k).factorial

-- Number of ways to choose meals
def meal_combinations : ℕ :=
  meat_options * (comb veg_options 2) * dessert_options * (perm bread_options 2)

theorem tyler_meal_combinations_is_720 : meal_combinations = 720 := by
  -- We provide proof later; for now, put sorry to skip
  sorry

end tyler_meal_combinations_is_720_l152_15299


namespace total_legs_arms_tentacles_correct_l152_15270

-- Define the counts of different animals
def num_horses : Nat := 2
def num_dogs : Nat := 5
def num_cats : Nat := 7
def num_turtles : Nat := 3
def num_goat : Nat := 1
def num_snakes : Nat := 4
def num_spiders : Nat := 2
def num_birds : Nat := 3
def num_starfish : Nat := 1
def num_octopus : Nat := 1
def num_three_legged_dogs : Nat := 1

-- Define the legs, arms, and tentacles for each type of animal
def legs_per_horse : Nat := 4
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def legs_per_turtle : Nat := 4
def legs_per_goat : Nat := 4
def legs_per_snake : Nat := 0
def legs_per_spider : Nat := 8
def legs_per_bird : Nat := 2
def arms_per_starfish : Nat := 5
def tentacles_per_octopus : Nat := 6
def legs_per_three_legged_dog : Nat := 3

-- Define the total number of legs, arms, and tentacles
def total_legs_arms_tentacles : Nat := 
  (num_horses * legs_per_horse) + 
  (num_dogs * legs_per_dog) + 
  (num_cats * legs_per_cat) + 
  (num_turtles * legs_per_turtle) + 
  (num_goat * legs_per_goat) + 
  (num_snakes * legs_per_snake) + 
  (num_spiders * legs_per_spider) + 
  (num_birds * legs_per_bird) + 
  (num_starfish * arms_per_starfish) + 
  (num_octopus * tentacles_per_octopus) + 
  (num_three_legged_dogs * legs_per_three_legged_dog)

-- The theorem to prove
theorem total_legs_arms_tentacles_correct :
  total_legs_arms_tentacles = 108 := by
  -- Proof goes here
  sorry

end total_legs_arms_tentacles_correct_l152_15270


namespace arithmetic_sequence_common_difference_l152_15211
-- Lean 4 Proof Statement


theorem arithmetic_sequence_common_difference 
  (a : ℕ) (n : ℕ) (d : ℕ) (S_n : ℕ) (a_n : ℕ) 
  (h1 : a = 2) 
  (h2 : a_n = 29) 
  (h3 : S_n = 155) 
  (h4 : S_n = n * (a + a_n) / 2) 
  (h5 : a_n = a + (n - 1) * d) 
  : d = 3 := 
by 
  sorry

end arithmetic_sequence_common_difference_l152_15211


namespace compare_pi_314_compare_neg_sqrt3_neg_sqrt2_compare_2_sqrt5_l152_15228

theorem compare_pi_314 : Real.pi > 3.14 :=
by sorry

theorem compare_neg_sqrt3_neg_sqrt2 : -Real.sqrt 3 < -Real.sqrt 2 :=
by sorry

theorem compare_2_sqrt5 : 2 < Real.sqrt 5 :=
by sorry

end compare_pi_314_compare_neg_sqrt3_neg_sqrt2_compare_2_sqrt5_l152_15228


namespace determine_k_for_linear_dependence_l152_15287

theorem determine_k_for_linear_dependence :
  ∃ k : ℝ, (∀ (a1 a2 : ℝ), a1 ≠ 0 ∧ a2 ≠ 0 → 
  a1 • (⟨1, 2, 3⟩ : ℝ × ℝ × ℝ) + a2 • (⟨4, k, 6⟩ : ℝ × ℝ × ℝ) = (⟨0, 0, 0⟩ : ℝ × ℝ × ℝ)) → k = 8 :=
by
  sorry

end determine_k_for_linear_dependence_l152_15287


namespace train_passing_platform_time_l152_15242

theorem train_passing_platform_time
  (L_train : ℝ) (L_plat : ℝ) (time_to_cross_tree : ℝ) (time_to_pass_platform : ℝ)
  (H1 : L_train = 2400) 
  (H2 : L_plat = 800)
  (H3 : time_to_cross_tree = 60) :
  time_to_pass_platform = 80 :=
by
  -- add proof here
  sorry

end train_passing_platform_time_l152_15242


namespace part1_part2_l152_15238

variables (x y z : ℝ)

-- Conditions
def conditions := (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (x + y + z = 1)

-- Part 1: Prove 2(x^2 + y^2 + z^2) + 9xyz >= 1
theorem part1 (h : conditions x y z) : 2 * (x^2 + y^2 + z^2) + 9 * x * y * z ≥ 1 :=
sorry

-- Part 2: Prove xy + yz + zx - 3xyz ≤ 1/4
theorem part2 (h : conditions x y z) : x * y + y * z + z * x - 3 * x * y * z ≤ 1 / 4 :=
sorry

end part1_part2_l152_15238


namespace shaded_area_l152_15267

-- Definitions based on given conditions
def Rectangle (A B C D : ℝ) := True -- Placeholder for the geometric definition of a rectangle

-- Total area of the non-shaded part
def non_shaded_area : ℝ := 10

-- Problem statement in Lean
theorem shaded_area (A B C D : ℝ) :
  Rectangle A B C D →
  (exists shaded_area : ℝ, shaded_area = 14 ∧ non_shaded_area + shaded_area = A * B) :=
by
  sorry

end shaded_area_l152_15267


namespace geometric_sequence_common_ratio_l152_15218

theorem geometric_sequence_common_ratio (a1 a2 a3 : ℤ) (r : ℤ)
  (h1 : a1 = 9) (h2 : a2 = -18) (h3 : a3 = 36) (h4 : a2 / a1 = r) (h5 : a3 = a2 * r) :
  r = -2 := 
sorry

end geometric_sequence_common_ratio_l152_15218


namespace bags_of_cookies_l152_15263

theorem bags_of_cookies (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 33) (h2 : cookies_per_bag = 11) : total_cookies / cookies_per_bag = 3 :=
by
  sorry

end bags_of_cookies_l152_15263


namespace complement_of_union_l152_15255

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union (hU : U = {1, 2, 3, 4}) (hM : M = {1, 2}) (hN : N = {2, 3}) :
  (U \ (M ∪ N)) = {4} :=
by
  sorry

end complement_of_union_l152_15255


namespace consecutive_page_numbers_sum_l152_15237

theorem consecutive_page_numbers_sum (n : ℕ) (h : n * (n + 1) = 19881) : n + (n + 1) = 283 :=
sorry

end consecutive_page_numbers_sum_l152_15237


namespace black_ink_cost_l152_15226

theorem black_ink_cost (B : ℕ) 
  (h1 : 2 * B + 3 * 15 + 2 * 13 = 50 + 43) : B = 11 :=
by
  sorry

end black_ink_cost_l152_15226


namespace age_difference_l152_15244

theorem age_difference (A B C : ℕ) (h1 : B = 20) (h2 : C = B / 2) (h3 : A + B + C = 52) : A - B = 2 := by
  sorry

end age_difference_l152_15244


namespace period_of_trig_sum_l152_15260

theorem period_of_trig_sum : ∀ x : ℝ, 2 * Real.sin x + 3 * Real.cos x = 2 * Real.sin (x + 2 * Real.pi) + 3 * Real.cos (x + 2 * Real.pi) := 
sorry

end period_of_trig_sum_l152_15260


namespace nate_age_when_ember_is_14_l152_15279

theorem nate_age_when_ember_is_14
  (nate_age : ℕ)
  (ember_age : ℕ)
  (h_half_age : ember_age = nate_age / 2)
  (h_nate_current_age : nate_age = 14) :
  nate_age + (14 - ember_age) = 21 :=
by
  sorry

end nate_age_when_ember_is_14_l152_15279


namespace diamond_45_15_eq_3_l152_15235

noncomputable def diamond (x y : ℝ) : ℝ := x / y

theorem diamond_45_15_eq_3 :
  ∀ (x y : ℝ), 
    (∀ x y : ℝ, (x * y) / y = x * (x / y)) ∧
    (∀ x : ℝ, (x / 1) / x = x / 1) ∧
    (∀ x y : ℝ, x / y = x / y) ∧
    1 / 1 = 1
    → diamond 45 15 = 3 :=
by
  intros x y H
  sorry

end diamond_45_15_eq_3_l152_15235


namespace zack_traveled_countries_l152_15289

theorem zack_traveled_countries 
  (a : ℕ) (g : ℕ) (j : ℕ) (p : ℕ) (z : ℕ)
  (ha : a = 30)
  (hg : g = (3 / 5) * a)
  (hj : j = (1 / 3) * g)
  (hp : p = (4 / 3) * j)
  (hz : z = (5 / 2) * p) :
  z = 20 := 
sorry

end zack_traveled_countries_l152_15289


namespace choir_members_l152_15245

theorem choir_members (n : ℕ) :
  (150 < n) ∧ (n < 250) ∧ (n % 4 = 3) ∧ (n % 5 = 4) ∧ (n % 8 = 5) → n = 159 :=
by
  sorry

end choir_members_l152_15245


namespace tom_filled_balloons_l152_15297

theorem tom_filled_balloons :
  ∀ (Tom Luke Anthony : ℕ), 
    (Tom = 3 * Luke) →
    (Luke = Anthony / 4) →
    (Anthony = 44) →
    (Tom = 33) :=
by
  intros Tom Luke Anthony hTom hLuke hAnthony
  sorry

end tom_filled_balloons_l152_15297


namespace range_of_a_l152_15204

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ (a ≤ 0 ∨ a ≥ 4) :=
by {
  sorry
}

end range_of_a_l152_15204


namespace p_implies_q_l152_15209

theorem p_implies_q (x : ℝ) (h : |5 * x - 1| > 4) : x^2 - (3/2) * x + (1/2) > 0 := sorry

end p_implies_q_l152_15209


namespace original_rectangle_perimeter_l152_15254

theorem original_rectangle_perimeter (l w : ℝ) (h1 : w = l / 2)
  (h2 : 2 * (w + l / 3) = 40) : 2 * l + 2 * w = 72 :=
by
  sorry

end original_rectangle_perimeter_l152_15254


namespace total_votes_cast_l152_15268

theorem total_votes_cast (total_votes : ℕ) (brenda_votes : ℕ) (percentage_brenda : ℚ) 
  (h1 : brenda_votes = 40) (h2 : percentage_brenda = 0.25) 
  (h3 : brenda_votes = percentage_brenda * total_votes) : total_votes = 160 := 
by sorry

end total_votes_cast_l152_15268


namespace delivery_boxes_l152_15295

-- Define the conditions
def stops : ℕ := 3
def boxes_per_stop : ℕ := 9

-- Define the total number of boxes
def total_boxes : ℕ := stops * boxes_per_stop

-- State the theorem
theorem delivery_boxes : total_boxes = 27 := by
  sorry

end delivery_boxes_l152_15295


namespace number_of_citroens_submerged_is_zero_l152_15292

-- Definitions based on the conditions
variables (x y : ℕ) -- Define x as the number of Citroen and y as the number of Renault submerged
variables (r p c vr vp : ℕ) -- Define r as the number of Renault, p as the number of Peugeot, c as the number of Citroën

-- Given conditions translated
-- Condition 1: There were twice as many Renault cars as there were Peugeot cars
def condition1 (r p : ℕ) : Prop := r = 2 * p
-- Condition 2: There were twice as many Peugeot cars as there were Citroens
def condition2 (p c : ℕ) : Prop := p = 2 * c
-- Condition 3: As many Citroens as Renaults were submerged in the water
def condition3 (x y : ℕ) : Prop := y = x
-- Condition 4: Three times as many Renaults were in the water as there were Peugeots
def condition4 (r y : ℕ) : Prop := r = 3 * y
-- Condition 5: As many Peugeots visible in the water as there were Citroens
def condition5 (vp c : ℕ) : Prop := vp = c

-- The question to prove: The number of Citroen cars submerged is 0
theorem number_of_citroens_submerged_is_zero
  (h1 : condition1 r p) 
  (h2 : condition2 p c)
  (h3 : condition3 x y)
  (h4 : condition4 r y)
  (h5 : condition5 vp c) :
  x = 0 :=
sorry

end number_of_citroens_submerged_is_zero_l152_15292


namespace range_of_f_l152_15269

noncomputable def f (x : ℝ) : ℝ := 2^(x + 2) - 4^x

def domain_M (x : ℝ) : Prop := 1 < x ∧ x < 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, domain_M x ∧ f x = y) ↔ -32 < y ∧ y < 4 :=
sorry

end range_of_f_l152_15269


namespace min_value_frac_l152_15215

open Real

theorem min_value_frac (a b : ℝ) (h1 : a + b = 1/2) (h2 : a > 0) (h3 : b > 0) :
    (4 / a + 1 / b) = 18 :=
sorry

end min_value_frac_l152_15215


namespace forty_percent_of_jacquelines_candy_bars_is_120_l152_15205

-- Define the number of candy bars Fred has
def fred_candy_bars : ℕ := 12

-- Define the number of candy bars Uncle Bob has
def uncle_bob_candy_bars : ℕ := fred_candy_bars + 6

-- Define the total number of candy bars Fred and Uncle Bob have together
def total_candy_bars : ℕ := fred_candy_bars + uncle_bob_candy_bars

-- Define the number of candy bars Jacqueline has
def jacqueline_candy_bars : ℕ := 10 * total_candy_bars

-- Define the number of candy bars that is 40% of Jacqueline's total
def forty_percent_jacqueline_candy_bars : ℕ := (40 * jacqueline_candy_bars) / 100

-- The statement to prove
theorem forty_percent_of_jacquelines_candy_bars_is_120 :
  forty_percent_jacqueline_candy_bars = 120 :=
sorry

end forty_percent_of_jacquelines_candy_bars_is_120_l152_15205


namespace circle_radius_l152_15206

theorem circle_radius :
  ∃ r : ℝ, ∀ x y : ℝ, (x^2 - 8 * x + y^2 + 4 * y + 16 = 0) → r = 2 :=
sorry

end circle_radius_l152_15206


namespace original_fraction_l152_15250

theorem original_fraction (n d : ℝ) (h1 : n + d = 5.25) (h2 : (n + 3) / (2 * d) = 1 / 3) : n / d = 2 / 33 :=
by
  sorry

end original_fraction_l152_15250


namespace multiple_of_4_multiple_of_8_not_multiple_of_16_multiple_of_24_l152_15278

def y : ℕ := 48 + 72 + 144 + 192 + 336 + 384 + 3072

theorem multiple_of_4 : ∃ k : ℕ, y = 4 * k := by
  sorry

theorem multiple_of_8 : ∃ k : ℕ, y = 8 * k := by
  sorry

theorem not_multiple_of_16 : ¬ ∃ k : ℕ, y = 16 * k := by
  sorry

theorem multiple_of_24 : ∃ k : ℕ, y = 24 * k := by
  sorry

end multiple_of_4_multiple_of_8_not_multiple_of_16_multiple_of_24_l152_15278


namespace net_income_correct_l152_15274

-- Definition of income before tax
def total_income_before_tax : ℝ := 45000

-- Definition of tax rate
def tax_rate : ℝ := 0.13

-- Definition of tax amount
def tax_amount : ℝ := tax_rate * total_income_before_tax

-- Definition of net income after tax
def net_income_after_tax : ℝ := total_income_before_tax - tax_amount

-- Theorem statement
theorem net_income_correct : net_income_after_tax = 39150 := by
  sorry

end net_income_correct_l152_15274


namespace trigonometric_identities_l152_15275

theorem trigonometric_identities
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (sinα : Real.sin α = 4 / 5)
  (cosβ : Real.cos β = 12 / 13) :
  Real.sin (α + β) = 63 / 65 ∧ Real.tan (α - β) = 33 / 56 := by
  sorry

end trigonometric_identities_l152_15275


namespace polar_to_cartesian_l152_15262

theorem polar_to_cartesian (p θ : ℝ) (x y : ℝ) (hp : p = 8 * Real.cos θ)
  (hx : x = p * Real.cos θ) (hy : y = p * Real.sin θ) :
  x^2 + y^2 = 8 * x := 
sorry

end polar_to_cartesian_l152_15262


namespace find_b_l152_15256

def nabla (a b : ℤ) (h : a ≠ b) : ℤ := (a + b) / (a - b)

theorem find_b (b : ℤ) (h : 3 ≠ b) (h_eq : nabla 3 b h = -4) : b = 5 :=
sorry

end find_b_l152_15256


namespace can_vasya_obtain_400_mercedes_l152_15293

-- Define the types for the cars
inductive Car : Type
| Zh : Car
| V : Car
| M : Car

-- Define the initial conditions as exchange constraints
def exchange1 (Zh V M : ℕ) : Prop :=
  3 * Zh = V + M

def exchange2 (V Zh M : ℕ) : Prop :=
  3 * V = 2 * Zh + M

-- Define the initial number of Zhiguli cars Vasya has.
def initial_Zh : ℕ := 700

-- Define the target number of Mercedes cars Vasya wants.
def target_M : ℕ := 400

-- The proof goal: Vasya cannot exchange to get exactly 400 Mercedes cars.
theorem can_vasya_obtain_400_mercedes (Zh V M : ℕ) (h1 : exchange1 Zh V M) (h2 : exchange2 V Zh M) :
  initial_Zh = 700 → target_M = 400 → (Zh ≠ 0 ∨ V ≠ 0 ∨ M ≠ 400) := sorry

end can_vasya_obtain_400_mercedes_l152_15293


namespace sum_excluded_values_domain_l152_15265

theorem sum_excluded_values_domain (x : ℝ) :
  (3 * x^2 - 9 * x + 6 = 0) → (x = 1 ∨ x = 2) ∧ (1 + 2 = 3) :=
by {
  -- given that 3x² - 9x + 6 = 0, we need to show that x = 1 or x = 2, and that their sum is 3
  sorry
}

end sum_excluded_values_domain_l152_15265


namespace car_daily_rental_cost_l152_15251

theorem car_daily_rental_cost 
  (x : ℝ)
  (cost_per_mile : ℝ)
  (budget : ℝ)
  (miles : ℕ)
  (h1 : cost_per_mile = 0.18)
  (h2 : budget = 75)
  (h3 : miles = 250)
  (h4 : x + (miles * cost_per_mile) = budget) : 
  x = 30 := 
sorry

end car_daily_rental_cost_l152_15251


namespace possible_values_of_k_l152_15225

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l152_15225


namespace coconut_transport_l152_15272

theorem coconut_transport (coconuts total_coconuts barbie_capacity bruno_capacity combined_capacity trips : ℕ)
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : bruno_capacity = 8)
  (h4 : combined_capacity = barbie_capacity + bruno_capacity)
  (h5 : combined_capacity = 12)
  (h6 : trips = total_coconuts / combined_capacity) :
  trips = 12 :=
by
  sorry

end coconut_transport_l152_15272


namespace motorboat_max_distance_l152_15298

/-- Given a motorboat which, when fully fueled, can travel exactly 40 km against the current 
    or 60 km with the current, proves that the maximum distance it can travel up the river and 
    return to the starting point with the available fuel is 24 km. -/
theorem motorboat_max_distance (upstream_dist : ℕ) (downstream_dist : ℕ) : 
  upstream_dist = 40 → downstream_dist = 60 → 
  ∃ max_round_trip_dist : ℕ, max_round_trip_dist = 24 :=
by
  intros h1 h2
  -- The proof would go here
  sorry

end motorboat_max_distance_l152_15298


namespace find_n_from_lcm_gcf_l152_15202

open scoped Classical

noncomputable def LCM (a b : ℕ) : ℕ := sorry
noncomputable def GCF (a b : ℕ) : ℕ := sorry

theorem find_n_from_lcm_gcf (n m : ℕ) (h1 : LCM n m = 48) (h2 : GCF n m = 18) (h3 : m = 16) : n = 54 :=
by sorry

end find_n_from_lcm_gcf_l152_15202


namespace soccer_games_total_l152_15212

variable (wins losses ties total_games : ℕ)

theorem soccer_games_total
    (h1 : losses = 9)
    (h2 : 4 * wins + 3 * losses + ties = 8 * total_games) :
    total_games = 24 :=
by
  sorry

end soccer_games_total_l152_15212


namespace trivia_team_students_l152_15227

theorem trivia_team_students (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) (total_students : ℕ) :
  not_picked = 17 →
  groups = 8 →
  students_per_group = 6 →
  total_students = not_picked + groups * students_per_group →
  total_students = 65 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end trivia_team_students_l152_15227


namespace angle_bisector_segment_conditional_equality_l152_15241

theorem angle_bisector_segment_conditional_equality
  (a1 b1 a2 b2 : ℝ)
  (h1 : ∃ (P : ℝ), ∃ (e1 e2 : ℝ → ℝ), (e1 P = a1 ∧ e2 P = b1) ∧ (e1 P = a2 ∧ e2 P = b2)) :
  (1 / a1 + 1 / b1 = 1 / a2 + 1 / b2) :=
by 
  sorry

end angle_bisector_segment_conditional_equality_l152_15241


namespace range_k_domain_f_l152_15220

theorem range_k_domain_f :
  (∀ x : ℝ, x^2 - 6*k*x + k + 8 ≥ 0) ↔ (-8/9 ≤ k ∧ k ≤ 1) :=
sorry

end range_k_domain_f_l152_15220


namespace quadratic_solution_1_l152_15217

theorem quadratic_solution_1 :
  (∃ x, x^2 - 4 * x + 3 = 0 ∧ (x = 1 ∨ x = 3)) :=
sorry

end quadratic_solution_1_l152_15217


namespace problem_solution_l152_15214

variable (a b c d m : ℝ)

-- Conditions
def opposite_numbers (a b : ℝ) : Prop := a + b = 0
def reciprocals (c d : ℝ) : Prop := c * d = 1
def absolute_value_eq (m : ℝ) : Prop := |m| = 3

theorem problem_solution
  (h1 : opposite_numbers a b)
  (h2 : reciprocals c d)
  (h3 : absolute_value_eq m) :
  (a + b) / 2023 - 4 * (c * d) + m^2 = 5 :=
by
  sorry

end problem_solution_l152_15214


namespace smallest_n_l152_15201

theorem smallest_n (n : ℕ) (h1 : n % 6 = 5) (h2 : n % 7 = 4) (h3 : n > 20) : n = 53 :=
sorry

end smallest_n_l152_15201


namespace Iris_total_spent_l152_15200

theorem Iris_total_spent :
  let jackets := 3
  let cost_per_jacket := 10
  let shorts := 2
  let cost_per_short := 6
  let pants := 4
  let cost_per_pant := 12
  jackets * cost_per_jacket + shorts * cost_per_short + pants * cost_per_pant = 90 := by
  sorry

end Iris_total_spent_l152_15200


namespace find_common_difference_l152_15291

variable {a : ℕ → ℝ}
variable {p q : ℕ}
variable {d : ℝ}

-- Definitions based on the conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) (p : ℕ) := a p = 4
def condition2 (a : ℕ → ℝ) (q : ℕ) := a q = 2
def condition3 (p q : ℕ) := p = 4 + q

-- The goal statement
theorem find_common_difference
  (a_seq : arithmetic_sequence a d)
  (h1 : condition1 a p)
  (h2 : condition2 a q)
  (h3 : condition3 p q) :
  d = 1 / 2 :=
by
  sorry

end find_common_difference_l152_15291


namespace violet_balloons_count_l152_15288

-- Define the initial number of violet balloons
def initial_violet_balloons := 7

-- Define the number of violet balloons Jason lost
def lost_violet_balloons := 3

-- Define the remaining violet balloons after losing some
def remaining_violet_balloons := initial_violet_balloons - lost_violet_balloons

-- Prove that the remaining violet balloons is equal to 4
theorem violet_balloons_count : remaining_violet_balloons = 4 :=
by
  sorry

end violet_balloons_count_l152_15288


namespace simplify_fraction_l152_15240

theorem simplify_fraction :
  ∀ (x : ℝ),
    (18 * x^4 - 9 * x^3 - 86 * x^2 + 16 * x + 96) /
    (18 * x^4 - 63 * x^3 + 22 * x^2 + 112 * x - 96) =
    (2 * x + 3) / (2 * x - 3) :=
by sorry

end simplify_fraction_l152_15240


namespace smallest_area_right_triangle_l152_15216

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l152_15216


namespace sqrt_seven_lt_three_l152_15280

theorem sqrt_seven_lt_three : Real.sqrt 7 < 3 :=
by
  sorry

end sqrt_seven_lt_three_l152_15280


namespace tangent_line_sum_l152_15221

theorem tangent_line_sum {f : ℝ → ℝ} (h_tangent : ∀ x, f x = (1/2 * x) + 2) :
  (f 1) + (deriv f 1) = 3 :=
by
  -- derive the value at x=1 and the derivative manually based on h_tangent
  sorry

end tangent_line_sum_l152_15221


namespace green_eyes_students_l152_15259

def total_students := 45
def brown_hair_condition (green_eyes : ℕ) := 3 * green_eyes
def both_attributes := 9
def neither_attributes := 5

theorem green_eyes_students (green_eyes : ℕ) :
  (total_students = (green_eyes - both_attributes) + both_attributes
    + (brown_hair_condition green_eyes - both_attributes) + neither_attributes) →
  green_eyes = 10 :=
by
  sorry

end green_eyes_students_l152_15259


namespace zoo_feeding_ways_l152_15224

-- Noncomputable is used for definitions that are not algorithmically computable
noncomputable def numFeedingWays : Nat :=
  4 * 3 * 3 * 2 * 2

theorem zoo_feeding_ways :
  ∀ (pairs : Fin 4 → (String × String)), -- Representing pairs of animals
  numFeedingWays = 144 :=
by
  sorry

end zoo_feeding_ways_l152_15224


namespace train_speed_l152_15294

theorem train_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) (h_distance : distance = 7.5) (h_time : time_minutes = 5) :
  speed = 90 :=
by
  sorry

end train_speed_l152_15294


namespace johns_cocktail_not_stronger_l152_15233

-- Define the percentage of alcohol in each beverage
def beer_percent_alcohol : ℝ := 0.05
def liqueur_percent_alcohol : ℝ := 0.10
def vodka_percent_alcohol : ℝ := 0.40
def whiskey_percent_alcohol : ℝ := 0.50

-- Define the weights of the liquids used in the cocktails
def john_liqueur_weight : ℝ := 400
def john_whiskey_weight : ℝ := 100
def ivan_vodka_weight : ℝ := 400
def ivan_beer_weight : ℝ := 100

-- Calculate the alcohol contents in each cocktail
def johns_cocktail_alcohol : ℝ := john_liqueur_weight * liqueur_percent_alcohol + john_whiskey_weight * whiskey_percent_alcohol
def ivans_cocktail_alcohol : ℝ := ivan_vodka_weight * vodka_percent_alcohol + ivan_beer_weight * beer_percent_alcohol

-- The proof statement to prove that John's cocktail is not stronger than Ivan's cocktail
theorem johns_cocktail_not_stronger :
  johns_cocktail_alcohol ≤ ivans_cocktail_alcohol :=
by
  -- Add proof steps here
  sorry

end johns_cocktail_not_stronger_l152_15233


namespace jackson_running_l152_15222

variable (x : ℕ)

theorem jackson_running (h : x + 4 = 7) : x = 3 := by
  sorry

end jackson_running_l152_15222


namespace smallest_integer_solution_l152_15261

theorem smallest_integer_solution : ∀ x : ℤ, (x < 2 * x - 7) → (8 = x) :=
by
  sorry

end smallest_integer_solution_l152_15261


namespace count_routes_from_P_to_Q_l152_15234

variable (P Q R S T : Type)
variable (roadPQ roadPS roadPT roadQR roadQS roadRS roadST : Prop)

theorem count_routes_from_P_to_Q :
  ∃ (routes : ℕ), routes = 16 :=
by
  sorry

end count_routes_from_P_to_Q_l152_15234


namespace problem1_problem2_l152_15264

-- Problem (1) Lean Statement
theorem problem1 (c a b : ℝ) (hc : c > a) (ha : a > b) (hb : b > 0) : 
  a / (c - a) > b / (c - b) :=
sorry

-- Problem (2) Lean Statement
theorem problem2 (x : ℝ) (hx : x > 2) : 
  ∃ (xmin : ℝ), xmin = 6 ∧ (x = 6 → (x + 16 / (x - 2)) = 10) :=
sorry

end problem1_problem2_l152_15264


namespace proof_problem_l152_15213

open Real

noncomputable def p : Prop := ∃ x : ℝ, x - 2 > log x / log 10
noncomputable def q : Prop := ∀ x : ℝ, x^2 > 0

theorem proof_problem :
  (p ∧ ¬q) := by
  sorry

end proof_problem_l152_15213


namespace subtract_add_example_l152_15290

theorem subtract_add_example : (3005 - 3000) + 10 = 15 :=
by
  sorry

end subtract_add_example_l152_15290
