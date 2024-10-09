import Mathlib

namespace find_smallest_angle_l144_14494

theorem find_smallest_angle 
  (x y : ℝ)
  (hx : x + y = 45)
  (hy : y = x - 5)
  (hz : x > 0 ∧ y > 0 ∧ x + y < 180) :
  min x y = 20 := 
sorry

end find_smallest_angle_l144_14494


namespace find_number_l144_14433

theorem find_number (x : ℝ) (h : 0.62 * x - 50 = 43) : x = 150 :=
sorry

end find_number_l144_14433


namespace factorize_expression_l144_14451

theorem factorize_expression (x : ℝ) :
  9 * x^2 - 6 * x + 1 = (3 * x - 1)^2 := 
by sorry

end factorize_expression_l144_14451


namespace system1_solution_system2_solution_l144_14420

theorem system1_solution (x y : ℚ) :
  x + y = 4 ∧ 5 * (x - y) - 2 * (x + y) = -1 →
  x = 27 / 10 ∧ y = 13 / 10 := by
  sorry

theorem system2_solution (x y : ℚ) :
  (2 * (x - y) / 3) - ((x + y) / 4) = -1 / 12 ∧ 3 * (x + y) - 2 * (2 * x - y) = 3 →
  x = 2 ∧ y = 1 := by
  sorry

end system1_solution_system2_solution_l144_14420


namespace find_angle_D_l144_14476

theorem find_angle_D 
  (A B C D : ℝ) 
  (h1 : A + B = 180) 
  (h2 : C = D + 10) 
  (h3 : A = 50)
  : D = 20 := by
  sorry

end find_angle_D_l144_14476


namespace total_number_of_toy_cars_l144_14402

-- Definitions based on conditions
def numCarsBox1 : ℕ := 21
def numCarsBox2 : ℕ := 31
def numCarsBox3 : ℕ := 19

-- The proof statement
theorem total_number_of_toy_cars : numCarsBox1 + numCarsBox2 + numCarsBox3 = 71 := by
  sorry

end total_number_of_toy_cars_l144_14402


namespace sum_series_l144_14490

theorem sum_series : (List.sum [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56, -59]) = -30 :=
by
  sorry

end sum_series_l144_14490


namespace min_homework_assignments_l144_14474

variable (p1 p2 p3 : Nat)

-- Define the points and assignments
def points_first_10 : Nat := 10
def assignments_first_10 : Nat := 10 * 1

def points_second_10 : Nat := 10
def assignments_second_10 : Nat := 10 * 2

def points_third_10 : Nat := 10
def assignments_third_10 : Nat := 10 * 3

def total_points : Nat := points_first_10 + points_second_10 + points_third_10
def total_assignments : Nat := assignments_first_10 + assignments_second_10 + assignments_third_10

theorem min_homework_assignments (hp1 : points_first_10 = 10) (ha1 : assignments_first_10 = 10) 
  (hp2 : points_second_10 = 10) (ha2 : assignments_second_10 = 20)
  (hp3 : points_third_10 = 10) (ha3 : assignments_third_10 = 30)
  (tp : total_points = 30) : 
  total_assignments = 60 := 
by sorry

end min_homework_assignments_l144_14474


namespace dinner_cost_l144_14432

variable (total_cost : ℝ)
variable (tax_rate : ℝ)
variable (tip_rate : ℝ)
variable (pre_tax_cost : ℝ)
variable (tip : ℝ)
variable (tax : ℝ)
variable (final_cost : ℝ)

axiom h1 : total_cost = 27.50
axiom h2 : tax_rate = 0.10
axiom h3 : tip_rate = 0.15
axiom h4 : tax = tax_rate * pre_tax_cost
axiom h5 : tip = tip_rate * pre_tax_cost
axiom h6 : final_cost = pre_tax_cost + tax + tip

theorem dinner_cost : pre_tax_cost = 22 := by sorry

end dinner_cost_l144_14432


namespace find_distance_between_B_and_C_l144_14413

def problem_statement : Prop :=
  ∃ (x y : ℝ),
  (y / 75 + x / 145 = 4.8) ∧ 
  ((x + y) / 100 = 2 + y / 70) ∧ 
  x = 290

theorem find_distance_between_B_and_C : problem_statement :=
by
  sorry

end find_distance_between_B_and_C_l144_14413


namespace shaded_region_area_l144_14465

open Real

noncomputable def area_of_shaded_region (side : ℝ) (radius : ℝ) : ℝ :=
  let area_square := side ^ 2
  let area_sector := π * radius ^ 2 / 4
  let area_triangle := (1 / 2) * (side / 2) * sqrt ((side / 2) ^ 2 - radius ^ 2)
  area_square - 8 * area_triangle - 4 * area_sector

theorem shaded_region_area (h_side : ℝ) (h_radius : ℝ)
  (h1 : h_side = 8) (h2 : h_radius = 3) :
  area_of_shaded_region h_side h_radius = 64 - 16 * sqrt 7 - 3 * π :=
by
  rw [h1, h2]
  sorry

end shaded_region_area_l144_14465


namespace defeat_giant_enemy_crab_l144_14486

-- Definitions for the conditions of cutting legs and claws
def claws : ℕ := 2
def legs : ℕ := 6
def totalCuts : ℕ := claws + legs
def valid_sequences : ℕ :=
  (Nat.factorial legs) * (Nat.factorial claws) * Nat.choose (totalCuts - claws - 1) claws

-- Statement to prove the number of valid sequences of cuts given the conditions
theorem defeat_giant_enemy_crab : valid_sequences = 14400 := by
  sorry

end defeat_giant_enemy_crab_l144_14486


namespace find_f_2012_l144_14427

noncomputable def f : ℤ → ℤ := sorry

axiom even_function : ∀ x : ℤ, f (-x) = f x
axiom f_1 : f 1 = 1
axiom f_2011_ne_1 : f 2011 ≠ 1
axiom max_property : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)

theorem find_f_2012 : f 2012 = 1 := sorry

end find_f_2012_l144_14427


namespace B_cycling_speed_l144_14496

/--
A walks at 10 kmph. 10 hours after A starts, B cycles after him at a certain speed.
B catches up with A at a distance of 200 km from the start. Prove that B's cycling speed is 20 kmph.
-/
theorem B_cycling_speed (speed_A : ℝ) (time_A_to_start_B : ℝ) 
  (distance_at_catch : ℝ) (B_speed : ℝ)
  (h1 : speed_A = 10) 
  (h2 : time_A_to_start_B = 10)
  (h3 : distance_at_catch = 200)
  (h4 : distance_at_catch = speed_A * time_A_to_start_B + speed_A * (distance_at_catch / speed_B)) :
    B_speed = 20 := by
  sorry

end B_cycling_speed_l144_14496


namespace max_area_central_angle_l144_14441

theorem max_area_central_angle (r l : ℝ) (S α : ℝ) (h1 : 2 * r + l = 4)
  (h2 : S = (1 / 2) * l * r) : (∀ x y : ℝ, (1 / 2) * x * y ≤ (1 / 4) * ((x + y) / 2) ^ 2) → α = l / r → α = 2 :=
by
  sorry

end max_area_central_angle_l144_14441


namespace new_pressure_of_helium_l144_14497

noncomputable def helium_pressure (p V p' V' : ℝ) (k : ℝ) : Prop :=
  p * V = k ∧ p' * V' = k

theorem new_pressure_of_helium :
  ∀ (p V p' V' k : ℝ), 
  p = 8 ∧ V = 3.5 ∧ V' = 7 ∧ k = 28 →
  helium_pressure p V p' V' k →
  p' = 4 :=
by
  intros p V p' V' k h1 h2
  sorry

end new_pressure_of_helium_l144_14497


namespace weierstrass_limit_l144_14425

theorem weierstrass_limit (a_n : ℕ → ℝ) (M : ℝ) :
  (∀ n m, n ≤ m → a_n n ≤ a_n m) → 
  (∀ n, a_n n ≤ M ) → 
  ∃ c, ∀ ε > 0, ∃ N, ∀ n ≥ N, |a_n n - c| < ε :=
by
  sorry

end weierstrass_limit_l144_14425


namespace solve_for_k_l144_14435

theorem solve_for_k (k x : ℝ) (h₁ : 4 * k - 3 * x = 2) (h₂ : x = -1) : 
  k = -1 / 4 := 
by sorry

end solve_for_k_l144_14435


namespace find_vector_coordinates_l144_14473

structure Point3D :=
  (x y z : ℝ)

def vector_sub (a b : Point3D) : Point3D :=
  Point3D.mk (b.x - a.x) (b.y - a.y) (b.z - a.z)

theorem find_vector_coordinates (A B : Point3D)
  (hA : A = { x := 1, y := -3, z := 4 })
  (hB : B = { x := -3, y := 2, z := 1 }) :
  vector_sub A B = { x := -4, y := 5, z := -3 } :=
by
  -- Proof is omitted
  sorry

end find_vector_coordinates_l144_14473


namespace parallel_lines_slope_equal_l144_14484

theorem parallel_lines_slope_equal (k : ℝ) : (∀ x : ℝ, 2 * x = k * x + 3) → k = 2 :=
by
  intros
  sorry

end parallel_lines_slope_equal_l144_14484


namespace probability_yellow_chalk_is_three_fifths_l144_14442

open Nat

theorem probability_yellow_chalk_is_three_fifths
  (yellow_chalks : ℕ) (red_chalks : ℕ) (total_chalks : ℕ)
  (h_yellow : yellow_chalks = 3) (h_red : red_chalks = 2) (h_total : total_chalks = yellow_chalks + red_chalks) :
  (yellow_chalks : ℚ) / (total_chalks : ℚ) = 3 / 5 := by
  sorry

end probability_yellow_chalk_is_three_fifths_l144_14442


namespace process_cannot_continue_indefinitely_l144_14426

theorem process_cannot_continue_indefinitely (n : ℕ) (hn : 2018 ∣ n) :
  ¬(∀ m, ∃ k, (10*m + k) % 11 = 0 ∧ (10*m + k) / 11 ∣ n) :=
sorry

end process_cannot_continue_indefinitely_l144_14426


namespace find_angle_B_l144_14456

-- Definitions and conditions
variables (α β γ δ : ℝ) -- representing angles ∠A, ∠B, ∠C, and ∠D

-- Given Condition: it's a parallelogram and sum of angles A and C
def quadrilateral_parallelogram (A B C D : ℝ) : Prop :=
  A + C = 200 ∧ A = C ∧ A + B = 180

-- Theorem: Degree of angle B is 80°
theorem find_angle_B (A B C D : ℝ) (h : quadrilateral_parallelogram A B C D) : B = 80 := 
  by sorry

end find_angle_B_l144_14456


namespace prime_condition_composite_condition_l144_14480

theorem prime_condition (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.Injective a)
  (h_prime : Prime (2 * n - 1)) :
  ∃ i j : Fin n, i ≠ j ∧ ((a i + a j) / Nat.gcd (a i) (a j) ≥ 2 * n - 1) := 
sorry

theorem composite_condition (n : ℕ) (h_composite : ¬ Prime (2 * n - 1)) :
  ∃ a : Fin n → ℕ, Function.Injective a ∧ (∀ i j : Fin n, i ≠ j → ((a i + a j) / Nat.gcd (a i) (a j) < 2 * n - 1)) := 
sorry

end prime_condition_composite_condition_l144_14480


namespace malcolm_social_media_followers_l144_14439

theorem malcolm_social_media_followers :
  let instagram_initial := 240
  let facebook_initial := 500
  let twitter_initial := (instagram_initial + facebook_initial) / 2
  let tiktok_initial := 3 * twitter_initial
  let youtube_initial := tiktok_initial + 510
  let pinterest_initial := 120
  let snapchat_initial := pinterest_initial / 2

  let instagram_after := instagram_initial + (15 * instagram_initial / 100)
  let facebook_after := facebook_initial + (20 * facebook_initial / 100)
  let twitter_after := twitter_initial - 12
  let tiktok_after := tiktok_initial + (10 * tiktok_initial / 100)
  let youtube_after := youtube_initial + (8 * youtube_initial / 100)
  let pinterest_after := pinterest_initial + 20
  let snapchat_after := snapchat_initial - (5 * snapchat_initial / 100)

  instagram_after + facebook_after + twitter_after + tiktok_after + youtube_after + pinterest_after + snapchat_after = 4402 := sorry

end malcolm_social_media_followers_l144_14439


namespace intersection_A1_B1_complement_A1_B1_union_A2_B2_l144_14460

-- Problem 1: Intersection and Complement
def setA1 : Set ℕ := {x : ℕ | x > 0 ∧ x < 9}
def setB1 : Set ℕ := {1, 2, 3}

theorem intersection_A1_B1 : (setA1 ∩ setB1) = {1, 2, 3} := by
  sorry

theorem complement_A1_B1 : {x : ℕ | x ∈ setA1 ∧ x ∉ setB1} = {4, 5, 6, 7, 8} := by
  sorry

-- Problem 2: Union
def setA2 : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def setB2 : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem union_A2_B2 : (setA2 ∪ setB2) = {x : ℝ | (-3 < x ∧ x < 1) ∨ (2 < x ∧ x < 10)} := by
  sorry

end intersection_A1_B1_complement_A1_B1_union_A2_B2_l144_14460


namespace part1_max_price_part2_min_sales_volume_l144_14479

noncomputable def original_price : ℝ := 25
noncomputable def original_sales_volume : ℝ := 80000
noncomputable def original_revenue : ℝ := original_price * original_sales_volume
noncomputable def max_new_price (t : ℝ) : Prop := t * (130000 - 2000 * t) ≥ original_revenue

theorem part1_max_price (t : ℝ) (ht : max_new_price t) : t ≤ 40 :=
sorry

noncomputable def investment (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600) + 50 + (x / 5)
noncomputable def min_sales_volume (x : ℝ) (a : ℝ) : Prop := a * x ≥ original_revenue + investment x

theorem part2_min_sales_volume (a : ℝ) : min_sales_volume 30 a → a ≥ 10.2 :=
sorry

end part1_max_price_part2_min_sales_volume_l144_14479


namespace budget_percentage_for_genetically_modified_organisms_l144_14461

theorem budget_percentage_for_genetically_modified_organisms
  (microphotonics : ℝ)
  (home_electronics : ℝ)
  (food_additives : ℝ)
  (industrial_lubricants : ℝ)
  (astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 15 →
  industrial_lubricants = 8 →
  astrophysics_degrees = 72 →
  (72 / 360) * 100 = 20 →
  100 - (14 + 24 + 15 + 8 + 20) = 19 :=
  sorry

end budget_percentage_for_genetically_modified_organisms_l144_14461


namespace pow_mod_1110_l144_14463

theorem pow_mod_1110 (n : ℕ) (h₀ : 0 ≤ n ∧ n < 1111)
    (h₁ : 2^1110 % 11 = 1) (h₂ : 2^1110 % 101 = 14) : 
    n = 1024 := 
sorry

end pow_mod_1110_l144_14463


namespace factor_polynomials_l144_14487

theorem factor_polynomials (x : ℝ) :
  (x^2 + 4 * x + 3) * (x^2 + 9 * x + 20) + (x^2 + 6 * x - 9) = 
  (x^2 + 6 * x + 6) * (x^2 + 6 * x + 3) :=
sorry

end factor_polynomials_l144_14487


namespace find_y_for_orthogonality_l144_14499

theorem find_y_for_orthogonality (y : ℝ) : (3 * y + 7 * (-4) = 0) → y = 28 / 3 := by
  sorry

end find_y_for_orthogonality_l144_14499


namespace unvisited_planet_exists_l144_14423

theorem unvisited_planet_exists (n : ℕ) (h : 1 ≤ n)
  (planets : Fin (2 * n + 1) → ℝ) 
  (distinct_distances : ∀ i j : Fin (2 * n + 1), i ≠ j → planets i ≠ planets j) 
  (expeditions : Fin (2 * n + 1) → Fin (2 * n + 1))
  (closest : ∀ i : Fin (2 * n + 1), expeditions i = i ↔ False) :
  ∃ p : Fin (2 * n + 1), ∀ q : Fin (2 * n + 1), expeditions q ≠ p := sorry

end unvisited_planet_exists_l144_14423


namespace range_a_l144_14405

noncomputable def f (x : ℝ) : ℝ := -(1 / 3) * x^3 + x

theorem range_a (a : ℝ) (h1 : a < 1) (h2 : 1 < 10 - a^2) (h3 : f a ≤ f 1) :
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_a_l144_14405


namespace evaluate_polynomial_l144_14437

theorem evaluate_polynomial (x : ℝ) : x * (x * (x * (x - 3) - 5) + 9) + 2 = x^4 - 3 * x^3 - 5 * x^2 + 9 * x + 2 := by
  sorry

end evaluate_polynomial_l144_14437


namespace number_chosen_div_8_sub_100_eq_6_l144_14457

variable (n : ℤ)

theorem number_chosen_div_8_sub_100_eq_6 (h : (n / 8) - 100 = 6) : n = 848 := 
by
  sorry

end number_chosen_div_8_sub_100_eq_6_l144_14457


namespace sum_weights_greater_than_2p_l144_14468

variables (p x y l l' : ℝ)

-- Conditions
axiom balance1 : x * l = p * l'
axiom balance2 : y * l' = p * l

-- The statement to prove
theorem sum_weights_greater_than_2p : x + y > 2 * p :=
by
  sorry

end sum_weights_greater_than_2p_l144_14468


namespace distinct_natural_numbers_l144_14443

theorem distinct_natural_numbers (n : ℕ) (h : n = 100) : 
  ∃ (nums : Fin n → ℕ), 
    (∀ i j, i ≠ j → nums i ≠ nums j) ∧
    (∀ (a b c d e : Fin n), 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
     c ≠ d ∧ c ≠ e ∧ 
     d ≠ e →
      (nums a) * (nums b) * (nums c) * (nums d) * (nums e) % ((nums a) + (nums b) + (nums c) + (nums d) + (nums e)) = 0) :=
by
  sorry

end distinct_natural_numbers_l144_14443


namespace range_of_a_l144_14416

noncomputable def M : Set ℝ := {2, 0, -1}
noncomputable def N (a : ℝ) : Set ℝ := {x | abs (x - a) < 1}

theorem range_of_a (a : ℝ) : (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 3) ↔ M ∩ N a = {x} :=
by
  sorry

end range_of_a_l144_14416


namespace probability_at_least_one_defective_probability_at_most_one_defective_l144_14438

noncomputable def machine_a_defect_rate : ℝ := 0.05
noncomputable def machine_b_defect_rate : ℝ := 0.1

/-- 
Prove the probability that there is at least one defective part among the two parts
given the defect rates of machine A and machine B
--/
theorem probability_at_least_one_defective (pA pB : ℝ) (hA : pA = machine_a_defect_rate) (hB : pB = machine_b_defect_rate) : 
  (1 - (1 - pA) * (1 - pB)) = 0.145 :=
  sorry

/-- 
Prove the probability that there is at most one defective part among the two parts
given the defect rates of machine A and machine B
--/
theorem probability_at_most_one_defective (pA pB : ℝ) (hA : pA = machine_a_defect_rate) (hB : pB = machine_b_defect_rate) : 
  (1 - pA * pB) = 0.995 :=
  sorry

end probability_at_least_one_defective_probability_at_most_one_defective_l144_14438


namespace area_of_triangle_ACD_l144_14412

theorem area_of_triangle_ACD (p : ℝ) (y1 y2 x1 x2 : ℝ)
  (h1 : y1^2 = 2 * p * x1)
  (h2 : y2^2 = 2 * p * x2)
  (h3 : y1 + y2 = 4 * p)
  (h4 : y2 - y1 = p)
  (h5 : 2 * y1 + 2 * y2 = 8 * p^2 / (x2 - x1))
  (h6 : x2 - x1 = 2 * p)
  (h7 : 8 * p^2 = (y1 + y2) * 2 * p) :
  1 / 2 * (y1 * (x1 - (x2 + x1) / 2) + y2 * (x2 - (x2 + x1) / 2)) = 15 / 2 * p^2 :=
by
  sorry

end area_of_triangle_ACD_l144_14412


namespace baguettes_leftover_l144_14448

-- Definitions based on conditions
def batches_per_day := 3
def baguettes_per_batch := 48
def sold_after_first_batch := 37
def sold_after_second_batch := 52
def sold_after_third_batch := 49

-- Prove the question equals the answer
theorem baguettes_leftover : 
  (batches_per_day * baguettes_per_batch - (sold_after_first_batch + sold_after_second_batch + sold_after_third_batch)) = 6 := 
by 
  sorry

end baguettes_leftover_l144_14448


namespace percent_savings_12_roll_package_l144_14414

def percent_savings_per_roll (package_cost : ℕ) (individual_cost : ℕ) (num_rolls : ℕ) : ℚ :=
  let individual_total := num_rolls * individual_cost
  let package_total := package_cost
  let per_roll_package_cost := package_total / num_rolls
  let savings_per_roll := individual_cost - per_roll_package_cost
  (savings_per_roll / individual_cost) * 100

theorem percent_savings_12_roll_package :
  percent_savings_per_roll 9 1 12 = 25 := 
sorry

end percent_savings_12_roll_package_l144_14414


namespace proof_A_union_B_eq_R_l144_14485

def A : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - 5) < a }

theorem proof_A_union_B_eq_R (a : ℝ) (h : a > 6) : 
  A ∪ B a = Set.univ :=
by {
  sorry
}

end proof_A_union_B_eq_R_l144_14485


namespace race_distance_l144_14404

def race_distance_problem (V_A V_B T : ℝ) : Prop :=
  V_A * T = 218.75 ∧
  V_B * T = 193.75 ∧
  V_B * (T + 10) = 218.75 ∧
  T = 77.5

theorem race_distance (D : ℝ) (V_A V_B T : ℝ) 
  (h1 : V_A * T = D) 
  (h2 : V_B * T = D - 25) 
  (h3 : V_B * (T + 10) = D) 
  (h4 : V_A * T = 218.75) 
  (h5 : T = 77.5) 
  : D = 218.75 := 
by 
  sorry

end race_distance_l144_14404


namespace intersection_eq_l144_14430

def setM (x : ℝ) : Prop := x > -1
def setN (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem intersection_eq : {x : ℝ | setM x} ∩ {x | setN x} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end intersection_eq_l144_14430


namespace max_square_side_length_l144_14421

-- Given: distances between consecutive lines in L and P
def distances_L : List ℕ := [2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6, 2]
def distances_P : List ℕ := [3, 1, 2, 6, 3, 1, 2, 6, 3, 1, 2, 6, 3, 1]

-- Theorem: Maximum possible side length of a square with sides on lines L and P
theorem max_square_side_length : ∀ (L P : List ℕ), L = distances_L → P = distances_P → ∃ s, s = 40 :=
by
  intros L P hL hP
  sorry

end max_square_side_length_l144_14421


namespace cheaper_to_buy_more_cheaper_2_values_l144_14410

def cost_function (n : ℕ) : ℕ :=
  if (1 ≤ n ∧ n ≤ 30) then 15 * n - 20
  else if (31 ≤ n ∧ n ≤ 55) then 14 * n
  else if (56 ≤ n) then 13 * n + 10
  else 0  -- Assuming 0 for n < 1 as it shouldn't happen in this context

theorem cheaper_to_buy_more_cheaper_2_values : 
  ∃ n1 n2 : ℕ, n1 < n2 ∧ cost_function (n1 + 1) < cost_function n1 ∧ cost_function (n2 + 1) < cost_function n2 ∧
  ∀ n : ℕ, (cost_function (n + 1) < cost_function n → n = n1 ∨ n = n2) := 
sorry

end cheaper_to_buy_more_cheaper_2_values_l144_14410


namespace age_of_child_l144_14417

theorem age_of_child 
  (avg_age_3_years_ago : ℕ)
  (family_size_3_years_ago : ℕ)
  (current_family_size : ℕ)
  (current_avg_age : ℕ)
  (h1 : avg_age_3_years_ago = 17)
  (h2 : family_size_3_years_ago = 5)
  (h3 : current_family_size = 6)
  (h4 : current_avg_age = 17)
  : ∃ age_of_baby : ℕ, age_of_baby = 2 := 
by
  sorry

end age_of_child_l144_14417


namespace kristy_baked_cookies_l144_14428

theorem kristy_baked_cookies (C : ℕ) :
  (C - 3) - 8 - 12 - 16 - 6 - 14 = 10 ↔ C = 69 := by
  sorry

end kristy_baked_cookies_l144_14428


namespace product_of_consecutive_integers_l144_14472

theorem product_of_consecutive_integers
  (a b : ℕ) (n : ℕ)
  (h1 : a = 12)
  (h2 : b = 22)
  (mean_five_numbers : (a + b + n + (n + 1) + (n + 2)) / 5 = 17) :
  (n * (n + 1) * (n + 2)) = 4896 := by
  sorry

end product_of_consecutive_integers_l144_14472


namespace probability_not_all_same_l144_14495

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l144_14495


namespace planA_text_message_cost_l144_14483

def planA_cost (x : ℝ) : ℝ := 60 * x + 9
def planB_cost : ℝ := 60 * 0.40

theorem planA_text_message_cost (x : ℝ) (h : planA_cost x = planB_cost) : x = 0.25 :=
by
  -- h represents the condition that the costs are equal
  -- The proof is skipped with sorry
  sorry

end planA_text_message_cost_l144_14483


namespace triangle_inequality_necessary_conditions_triangle_inequality_sufficient_conditions_l144_14455

/-- Points \(P, Q, R, S\) are distinct, collinear, and ordered on a line with line segment lengths \( a, b, c \)
    such that \(a = PQ\), \(b = PR\), \(c = PS\). After rotating \(PQ\) and \(RS\) to make \( P \) and \( S \) coincide
    and form a triangle with a positive area, we must show:
    \(I. a < \frac{c}{3}\) must be satisfied in accordance to the triangle inequality revelations -/
theorem triangle_inequality_necessary_conditions (a b c : ℝ)
  (h_abc1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : b > c - b ∧ c > a ∧ c > b - a) :
  a < c / 3 :=
sorry

theorem triangle_inequality_sufficient_conditions (a b c : ℝ)
  (h_abc2 : b ≥ c / 3 ∧ a < c ∧ 2 * b ≤ c) :
  ¬ b < c / 3 :=
sorry

end triangle_inequality_necessary_conditions_triangle_inequality_sufficient_conditions_l144_14455


namespace square_root_calc_l144_14488

theorem square_root_calc (x : ℤ) (hx : x^2 = 1764) : (x + 2) * (x - 2) = 1760 := by
  sorry

end square_root_calc_l144_14488


namespace minimum_seats_occupied_l144_14481

-- Define the conditions
def initial_seat_count : Nat := 150
def people_initially_leaving_up_to_two_empty_seats := true
def eventually_rule_changes_to_one_empty_seat := true

-- Define the function which checks the minimum number of occupied seats needed
def fewest_occupied_seats (total_seats : Nat) (initial_rule : Bool) (final_rule : Bool) : Nat :=
  if initial_rule && final_rule && total_seats = 150 then 57 else 0

-- The main theorem we need to prove
theorem minimum_seats_occupied {total_seats : Nat} : 
  total_seats = initial_seat_count → 
  people_initially_leaving_up_to_two_empty_seats → 
  eventually_rule_changes_to_one_empty_seat → 
  fewest_occupied_seats total_seats people_initially_leaving_up_to_two_empty_seats eventually_rule_changes_to_one_empty_seat = 57 :=
by
  intro h1 h2 h3
  sorry

end minimum_seats_occupied_l144_14481


namespace price_of_coffee_increased_by_300_percent_l144_14458

theorem price_of_coffee_increased_by_300_percent
  (P : ℝ) -- cost per pound of milk powder and coffee in June
  (h1 : 0.20 * P = 0.20) -- price of a pound of milk powder in July
  (h2 : 1.5 * 0.20 = 0.30) -- cost of 1.5 lbs of milk powder in July
  (h3 : 6.30 - 0.30 = 6.00) -- cost of 1.5 lbs of coffee in July
  (h4 : 6.00 / 1.5 = 4.00) -- price per pound of coffee in July
  : ((4.00 - 1.00) / 1.00) * 100 = 300 := 
by 
  sorry

end price_of_coffee_increased_by_300_percent_l144_14458


namespace least_number_to_divisible_sum_l144_14444

-- Define the conditions and variables
def initial_number : ℕ := 1100
def divisor : ℕ := 23
def least_number_to_add : ℕ := 4

-- Statement to prove
theorem least_number_to_divisible_sum :
  ∃ least_n, least_n + initial_number % divisor = divisor ∧ least_n = least_number_to_add :=
  by
    sorry

end least_number_to_divisible_sum_l144_14444


namespace luke_bike_vs_bus_slowness_l144_14450

theorem luke_bike_vs_bus_slowness
  (luke_bus_time : ℕ)
  (paula_ratio : ℚ)
  (total_travel_time : ℕ)
  (paula_total_bus_time : ℕ)
  (luke_total_travel_time_lhs : ℕ)
  (luke_total_travel_time_rhs : ℕ)
  (bike_time : ℕ)
  (ratio : ℚ) :
  luke_bus_time = 70 ∧
  paula_ratio = 3 / 5 ∧
  total_travel_time = 504 ∧
  paula_total_bus_time = 2 * (paula_ratio * luke_bus_time) ∧
  luke_total_travel_time_lhs = luke_bus_time + bike_time ∧
  luke_total_travel_time_rhs + paula_total_bus_time = total_travel_time ∧
  bike_time = ratio * luke_bus_time ∧
  ratio = bike_time / luke_bus_time →
  ratio = 5 :=
sorry

end luke_bike_vs_bus_slowness_l144_14450


namespace father_children_age_l144_14454

theorem father_children_age (F C n : Nat) (h1 : F = C) (h2 : F = 75) (h3 : C + 5 * n = 2 * (F + n)) : 
  n = 25 :=
by
  sorry

end father_children_age_l144_14454


namespace count_positive_multiples_of_7_ending_in_5_below_1500_l144_14401

theorem count_positive_multiples_of_7_ending_in_5_below_1500 : 
  ∃ n : ℕ, n = 21 ∧ (∀ k : ℕ, (k < 1500) → ((k % 7 = 0) ∧ (k % 10 = 5) → (∃ m : ℕ, k = 35 + 70 * m) ∧ (0 ≤ m) ∧ (m < 21))) :=
sorry

end count_positive_multiples_of_7_ending_in_5_below_1500_l144_14401


namespace value_added_to_number_l144_14403

theorem value_added_to_number (x : ℤ) : 
  (150 - 109 = 109 + x) → (x = -68) :=
by
  sorry

end value_added_to_number_l144_14403


namespace distance_travelled_downstream_l144_14493

def speed_boat_still_water : ℕ := 24
def speed_stream : ℕ := 4
def time_downstream : ℕ := 6

def effective_speed_downstream : ℕ := speed_boat_still_water + speed_stream
def distance_downstream : ℕ := effective_speed_downstream * time_downstream

theorem distance_travelled_downstream : distance_downstream = 168 := by
  sorry

end distance_travelled_downstream_l144_14493


namespace exists_digit_sum_divisible_by_11_l144_14492

-- Define a function to compute the sum of the digits of a natural number
def digit_sum (n : ℕ) : ℕ := 
  Nat.digits 10 n |>.sum

-- The main theorem to be proven
theorem exists_digit_sum_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k < 39 ∧ (digit_sum (N + k) % 11 = 0) := 
sorry

end exists_digit_sum_divisible_by_11_l144_14492


namespace product_of_sequence_l144_14467

theorem product_of_sequence : 
  (1 / 2) * (4 / 1) * (1 / 8) * (16 / 1) * (1 / 32) * (64 / 1) * (1 / 128) * (256 / 1) * (1 / 512) * (1024 / 1) * (1 / 2048) * (4096 / 1) = 64 := 
by
  sorry

end product_of_sequence_l144_14467


namespace arithmetic_mean_q_r_l144_14470

theorem arithmetic_mean_q_r (p q r : ℝ) (h1 : (p + q) / 2 = 10) (h2 : (q + r) / 2 = 27) (h3 : r - p = 34) : (q + r) / 2 = 27 :=
sorry

end arithmetic_mean_q_r_l144_14470


namespace general_formula_sum_of_first_10_terms_l144_14445

variable (a : ℕ → ℝ) (d : ℝ) (S_10 : ℝ)
variable (h1 : a 5 = 11) (h2 : a 8 = 5)

theorem general_formula (n : ℕ) : a n = -2 * n + 21 :=
sorry

theorem sum_of_first_10_terms : S_10 = 100 :=
sorry

end general_formula_sum_of_first_10_terms_l144_14445


namespace heavier_boxes_weight_l144_14489

theorem heavier_boxes_weight
  (x y : ℤ)
  (h1 : x ≥ 0)
  (h2 : x ≤ 30)
  (h3 : 10 * x + (30 - x) * y = 540)
  (h4 : 10 * x + (15 - x) * y = 240) :
  y = 20 :=
by
  sorry

end heavier_boxes_weight_l144_14489


namespace polynomial_value_at_five_l144_14464

def f (x : ℤ) : ℤ := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x + 7

theorem polynomial_value_at_five : f 5 = 2677 := by
  -- The proof goes here.
  sorry

end polynomial_value_at_five_l144_14464


namespace mass_percentage_K_l144_14408

theorem mass_percentage_K (compound : Type) (m : ℝ) (mass_percentage : ℝ) (h : mass_percentage = 23.81) : mass_percentage = 23.81 :=
by
  sorry

end mass_percentage_K_l144_14408


namespace prove_ratio_l144_14447

variable (a b c d : ℚ)

-- Conditions
def cond1 : a / b = 5 := sorry
def cond2 : b / c = 1 / 4 := sorry
def cond3 : c / d = 7 := sorry

-- Theorem to prove the final result
theorem prove_ratio (h1 : a / b = 5) (h2 : b / c = 1 / 4) (h3 : c / d = 7) :
  d / a = 4 / 35 :=
by
  sorry

end prove_ratio_l144_14447


namespace plain_pancakes_l144_14491

/-- Define the given conditions -/
def total_pancakes : ℕ := 67
def blueberry_pancakes : ℕ := 20
def banana_pancakes : ℕ := 24

/-- Define a theorem stating the number of plain pancakes given the conditions -/
theorem plain_pancakes : total_pancakes - (blueberry_pancakes + banana_pancakes) = 23 := by
  -- Here we will provide a proof
  sorry

end plain_pancakes_l144_14491


namespace average_s_t_l144_14422

theorem average_s_t (s t : ℝ) 
  (h : (1 + 3 + 7 + s + t) / 5 = 12) : 
  (s + t) / 2 = 24.5 :=
by
  sorry

end average_s_t_l144_14422


namespace ab_value_l144_14406

variable (a b : ℝ)

theorem ab_value (h1 : a^5 * b^8 = 12) (h2 : a^8 * b^13 = 18) : a * b = 128 / 3 := 
by 
  sorry

end ab_value_l144_14406


namespace orange_slices_l144_14469

theorem orange_slices (x : ℕ) (hx1 : 5 * x = x + 8) : x + 2 * x + 5 * x = 16 :=
by {
  sorry
}

end orange_slices_l144_14469


namespace average_pastries_per_day_l144_14446

def monday_sales : ℕ := 2
def increment_weekday : ℕ := 2
def increment_weekend : ℕ := 3

def tuesday_sales : ℕ := monday_sales + increment_weekday
def wednesday_sales : ℕ := tuesday_sales + increment_weekday
def thursday_sales : ℕ := wednesday_sales + increment_weekday
def friday_sales : ℕ := thursday_sales + increment_weekday
def saturday_sales : ℕ := friday_sales + increment_weekend
def sunday_sales : ℕ := saturday_sales + increment_weekend

def total_sales_week : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales
def average_sales_per_day : ℚ := total_sales_week / 7

theorem average_pastries_per_day : average_sales_per_day = 59 / 7 := by
  sorry

end average_pastries_per_day_l144_14446


namespace correct_option_l144_14418

-- Conditions
def option_A (a : ℝ) : Prop := a^3 + a^3 = a^6
def option_B (a : ℝ) : Prop := (a^3)^2 = a^9
def option_C (a : ℝ) : Prop := a^6 / a^3 = a^2
def option_D (a b : ℝ) : Prop := (a * b)^2 = a^2 * b^2

-- Proof Problem Statement
theorem correct_option (a b : ℝ) : option_D a b ↔ ¬option_A a ∧ ¬option_B a ∧ ¬option_C a :=
by
  sorry

end correct_option_l144_14418


namespace perpendicular_lines_a_value_l144_14498

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∃ x y : ℝ, ax + y + 1 = 0) ∧ (∃ x y : ℝ, x + y + 2 = 0) ∧ (∃ x y : ℝ, (y = -ax)) → a = -1 := by
  sorry

end perpendicular_lines_a_value_l144_14498


namespace all_increased_quadratics_have_integer_roots_l144_14449

def original_quadratic (p q : ℤ) : Prop :=
  ∃ α β : ℤ, α + β = -p ∧ α * β = q

def increased_quadratic (p q n : ℤ) : Prop :=
  ∃ α β : ℤ, α + β = -(p + n) ∧ α * β = (q + n)

theorem all_increased_quadratics_have_integer_roots (p q : ℤ) :
  original_quadratic p q →
  (∀ n, 0 ≤ n ∧ n ≤ 9 → increased_quadratic p q n) :=
sorry

end all_increased_quadratics_have_integer_roots_l144_14449


namespace machine_does_not_require_repair_l144_14411

-- Define the conditions.

def max_deviation := 37

def nominal_portion_max_deviation_percentage := 0.10

def deviation_within_limit (M : ℝ) : Prop :=
  37 ≤ 0.10 * M

def unreadable_measurements_deviation (deviation : ℝ) : Prop :=
  deviation < 37

-- Define the theorem we want to prove

theorem machine_does_not_require_repair (M : ℝ)
  (h1 : deviation_within_limit M)
  (h2 : ∀ deviation, unreadable_measurements_deviation deviation) :
  true := 
sorry

end machine_does_not_require_repair_l144_14411


namespace total_worth_of_travelers_checks_l144_14452

variable (x y : ℕ)

theorem total_worth_of_travelers_checks
  (h1 : x + y = 30)
  (h2 : 50 * (x - 15) + 100 * y = 1050) :
  50 * x + 100 * y = 1800 :=
sorry

end total_worth_of_travelers_checks_l144_14452


namespace actual_discount_is_expected_discount_l144_14471

-- Define the conditions
def promotional_discount := 20 / 100  -- 20% discount
def vip_card_discount := 10 / 100  -- 10% additional discount

-- Define the combined discount calculation
def combined_discount := (1 - promotional_discount) * (1 - vip_card_discount)

-- Define the expected discount off the original price
def expected_discount := 28 / 100  -- 28% discount

-- Theorem statement proving the combined discount is equivalent to the expected discount
theorem actual_discount_is_expected_discount :
  combined_discount = 1 - expected_discount :=
by
  -- Proof omitted.
  sorry

end actual_discount_is_expected_discount_l144_14471


namespace directrix_of_parabola_l144_14462

theorem directrix_of_parabola (y x : ℝ) : 
  (∃ a h k : ℝ, y = a * (x - h)^2 + k ∧ a = 1/8 ∧ h = 4 ∧ k = 0) → 
  y = -1/2 :=
by
  intro h
  sorry

end directrix_of_parabola_l144_14462


namespace rounding_and_scientific_notation_l144_14482

-- Define the original number
def original_number : ℕ := 1694000

-- Define the function to round to the nearest hundred thousand
def round_to_nearest_hundred_thousand (n : ℕ) : ℕ :=
  ((n + 50000) / 100000) * 100000

-- Define the function to convert to scientific notation
def to_scientific_notation (n : ℕ) : String :=
  let base := n / 1000000
  let exponent := 6
  s!"{base}.0 × 10^{exponent}"

-- Assert the equivalence
theorem rounding_and_scientific_notation :
  to_scientific_notation (round_to_nearest_hundred_thousand original_number) = "1.7 × 10^{6}" :=
by
  sorry

end rounding_and_scientific_notation_l144_14482


namespace red_shirts_count_l144_14415

theorem red_shirts_count :
  ∀ (total blue_fraction green_fraction : ℕ),
    total = 60 →
    blue_fraction = total / 3 →
    green_fraction = total / 4 →
    (total - (blue_fraction + green_fraction)) = 25 :=
by
  intros total blue_fraction green_fraction h_total h_blue h_green
  rw [h_total, h_blue, h_green]
  norm_num
  sorry

end red_shirts_count_l144_14415


namespace total_distance_fourth_time_l144_14453

/-- 
A super ball is dropped from a height of 100 feet and rebounds half the distance it falls each time.
We need to prove that the total distance the ball travels when it hits the ground
the fourth time is 275 feet.
-/
noncomputable def total_distance : ℝ :=
  let first_descent := 100
  let second_descent := first_descent / 2
  let third_descent := second_descent / 2
  let fourth_descent := third_descent / 2
  let first_ascent := second_descent
  let second_ascent := third_descent
  let third_ascent := fourth_descent
  first_descent + second_descent + third_descent + fourth_descent +
  first_ascent + second_ascent + third_ascent

theorem total_distance_fourth_time : total_distance = 275 := 
  by
  sorry

end total_distance_fourth_time_l144_14453


namespace find_a_l144_14409

-- Define the sets A and B and the condition that A union B is a subset of A intersect B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

theorem find_a (a : ℝ) :
  A ∪ B a ⊆ A ∩ B a → a = 1 :=
sorry

end find_a_l144_14409


namespace students_in_second_class_l144_14431

variable (x : ℕ)

theorem students_in_second_class :
  (∃ x, 30 * 40 + 70 * x = (30 + x) * 58.75) → x = 50 :=
by
  sorry

end students_in_second_class_l144_14431


namespace fraction_of_shaded_hexagons_l144_14459

-- Definitions
def total_hexagons : ℕ := 9
def shaded_hexagons : ℕ := 5

-- Theorem statement
theorem fraction_of_shaded_hexagons : 
  (shaded_hexagons: ℚ) / (total_hexagons : ℚ) = 5 / 9 := by
sorry

end fraction_of_shaded_hexagons_l144_14459


namespace count_even_factors_is_correct_l144_14440

def prime_factors_444_533_72 := (2^8 * 5^3 * 7^2)

def range_a := {a : ℕ | 0 ≤ a ∧ a ≤ 8}
def range_b := {b : ℕ | 0 ≤ b ∧ b ≤ 3}
def range_c := {c : ℕ | 0 ≤ c ∧ c ≤ 2}

def even_factors_count : ℕ :=
  (8 - 1 + 1) * (3 - 0 + 1) * (2 - 0 + 1)

theorem count_even_factors_is_correct :
  even_factors_count = 96 := by
  sorry

end count_even_factors_is_correct_l144_14440


namespace find_x_l144_14429

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_x (x : ℝ) : 
  (sqrt x / sqrt 0.81 + sqrt 1.44 / sqrt 0.49 = 3.0751133491652576) → 
  x = 1.5 :=
by { sorry }

end find_x_l144_14429


namespace statue_original_cost_l144_14434

noncomputable def original_cost (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

theorem statue_original_cost :
  original_cost 660 0.20 = 550 := 
by
  sorry

end statue_original_cost_l144_14434


namespace number_of_non_officers_l144_14400

theorem number_of_non_officers 
  (avg_salary_employees: ℝ) (avg_salary_officers: ℝ) (avg_salary_nonofficers: ℝ) 
  (num_officers: ℕ) (num_nonofficers: ℕ):
  avg_salary_employees = 120 ∧ avg_salary_officers = 440 ∧ avg_salary_nonofficers = 110 ∧
  num_officers = 15 ∧ 
  (15 * 440 + num_nonofficers * 110 = (15 + num_nonofficers) * 120)  → 
  num_nonofficers = 480 := 
by 
sorry

end number_of_non_officers_l144_14400


namespace degree_reduction_l144_14424

theorem degree_reduction (x : ℝ) (h1 : x^2 = x + 1) (h2 : 0 < x) : x^4 - 2 * x^3 + 3 * x = 1 + Real.sqrt 5 :=
by
  sorry

end degree_reduction_l144_14424


namespace jake_comic_books_l144_14436

variables (J : ℕ)

def brother_comic_books := J + 15
def total_comic_books := J + brother_comic_books

theorem jake_comic_books : total_comic_books = 87 → J = 36 :=
by
  sorry

end jake_comic_books_l144_14436


namespace original_price_of_article_l144_14477

theorem original_price_of_article (SP : ℝ) (profit_rate : ℝ) (P : ℝ) (h1 : SP = 550) (h2 : profit_rate = 0.10) (h3 : SP = P * (1 + profit_rate)) : P = 500 :=
by
  sorry

end original_price_of_article_l144_14477


namespace find_constant_e_l144_14407

theorem find_constant_e {x y e : ℝ} : (x / (2 * y) = 3 / e) → ((7 * x + 4 * y) / (x - 2 * y) = 25) → (e = 2) :=
by
  intro h1 h2
  sorry

end find_constant_e_l144_14407


namespace tan_frac_eq_l144_14466

theorem tan_frac_eq (x : ℝ) (h : Real.tan (x + π / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := 
  sorry

end tan_frac_eq_l144_14466


namespace students_to_add_l144_14478

theorem students_to_add (students := 1049) (teachers := 9) : ∃ n, students + n ≡ 0 [MOD teachers] ∧ n = 4 :=
by
  use 4
  sorry

end students_to_add_l144_14478


namespace rhombus_area_l144_14475

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 5) (h2 : d2 = 6) : 
  1 / 2 * d1 * d2 = 15 :=
by
  rw [h1, h2]
  norm_num

end rhombus_area_l144_14475


namespace remove_one_to_get_average_of_75_l144_14419

theorem remove_one_to_get_average_of_75 : 
  ∃ l : List ℕ, l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] ∧ 
  (∃ m : ℕ, List.erase l m = ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] : List ℕ) ∧ 
  (12 : ℕ) = List.length (List.erase l m) ∧
  7.5 = ((List.sum (List.erase l m) : ℚ) / 12)) :=
sorry

end remove_one_to_get_average_of_75_l144_14419
