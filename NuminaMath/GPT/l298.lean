import Mathlib

namespace children_on_ferris_wheel_l298_29885

theorem children_on_ferris_wheel (x : ℕ) (h : 5 * x + 3 * 5 + 8 * 2 * 5 = 110) : x = 3 :=
sorry

end children_on_ferris_wheel_l298_29885


namespace smallest_whole_number_greater_than_triangle_perimeter_l298_29815

theorem smallest_whole_number_greater_than_triangle_perimeter 
  (a b : ℝ) (h_a : a = 7) (h_b : b = 23) :
  ∀ c : ℝ, 16 < c ∧ c < 30 → ⌈a + b + c⌉ = 60 :=
by
  intros c h
  rw [h_a, h_b]
  sorry

end smallest_whole_number_greater_than_triangle_perimeter_l298_29815


namespace tail_length_10_l298_29814

theorem tail_length_10 (length_body tail_length head_length width height overall_length: ℝ) 
  (h1 : tail_length = (1 / 2) * length_body)
  (h2 : head_length = (1 / 6) * length_body)
  (h3 : height = 1.5 * width)
  (h4 : overall_length = length_body + tail_length)
  (h5 : overall_length = 30)
  (h6 : width = 12) :
  tail_length = 10 :=
by
  sorry

end tail_length_10_l298_29814


namespace central_angle_of_regular_hexagon_l298_29887

theorem central_angle_of_regular_hexagon :
  ∀ (total_angle : ℝ) (sides : ℝ), total_angle = 360 → sides = 6 → total_angle / sides = 60 :=
by
  intros total_angle sides h_total_angle h_sides
  rw [h_total_angle, h_sides]
  norm_num

end central_angle_of_regular_hexagon_l298_29887


namespace taco_castle_parking_lot_l298_29800

variable (D F T V : ℕ)

theorem taco_castle_parking_lot (h1 : F = D / 3) (h2 : F = 2 * T) (h3 : V = T / 2) (h4 : V = 5) : D = 60 :=
by
  sorry

end taco_castle_parking_lot_l298_29800


namespace total_goals_proof_l298_29804

-- Definitions based on the conditions
def first_half_team_a := 8
def first_half_team_b := first_half_team_a / 2
def first_half_team_c := first_half_team_b * 2

def second_half_team_a := first_half_team_c
def second_half_team_b := first_half_team_a
def second_half_team_c := second_half_team_b + 3

-- Total scores for each team
def total_team_a := first_half_team_a + second_half_team_a
def total_team_b := first_half_team_b + second_half_team_b
def total_team_c := first_half_team_c + second_half_team_c

-- Total goals for all teams
def total_goals := total_team_a + total_team_b + total_team_c

-- The theorem to be proved
theorem total_goals_proof : total_goals = 47 := by
  sorry

end total_goals_proof_l298_29804


namespace blocks_total_l298_29806

theorem blocks_total (blocks_initial : ℕ) (blocks_added : ℕ) (total_blocks : ℕ) 
  (h1 : blocks_initial = 86) (h2 : blocks_added = 9) : total_blocks = 95 :=
by
  sorry

end blocks_total_l298_29806


namespace probability_of_shaded_section_l298_29892

theorem probability_of_shaded_section 
  (total_sections : ℕ)
  (shaded_sections : ℕ)
  (H1 : total_sections = 8)
  (H2 : shaded_sections = 4)
  : (shaded_sections / total_sections : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_shaded_section_l298_29892


namespace number_of_crayons_given_to_friends_l298_29828

def totalCrayonsLostOrGivenAway := 229
def crayonsLost := 16
def crayonsGivenToFriends := totalCrayonsLostOrGivenAway - crayonsLost

theorem number_of_crayons_given_to_friends :
  crayonsGivenToFriends = 213 :=
by
  sorry

end number_of_crayons_given_to_friends_l298_29828


namespace factorization_correctness_l298_29869

theorem factorization_correctness :
  ∀ x : ℝ, x^2 - 2*x + 1 = (x - 1)^2 :=
by
  -- Proof omitted
  sorry

end factorization_correctness_l298_29869


namespace find_angle_A_l298_29895

theorem find_angle_A (a b : ℝ) (B A : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) 
  (h3 : B = Real.pi / 4) : A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l298_29895


namespace find_f2_l298_29883

theorem find_f2 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 2 * x ^ 2) :
  f 2 = -1 / 4 :=
by
  sorry

end find_f2_l298_29883


namespace jacket_spending_l298_29817

def total_spent : ℝ := 14.28
def spent_on_shorts : ℝ := 9.54
def spent_on_jacket : ℝ := 4.74

theorem jacket_spending :
  spent_on_jacket = total_spent - spent_on_shorts :=
by sorry

end jacket_spending_l298_29817


namespace line_through_point_with_equal_intercepts_l298_29829

-- Define the point through which the line passes
def point : ℝ × ℝ := (3, -2)

-- Define the property of having equal absolute intercepts
def has_equal_absolute_intercepts (a b : ℝ) : Prop :=
  |a| = |b|

-- Define the general form of a line equation
def line_eq (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Main theorem: Any line passing through (3, -2) with equal absolute intercepts satisfies the given equations
theorem line_through_point_with_equal_intercepts (a b : ℝ) :
  has_equal_absolute_intercepts a b
  → line_eq 2 3 0 3 (-2)
  ∨ line_eq 1 1 (-1) 3 (-2)
  ∨ line_eq 1 (-1) (-5) 3 (-2) :=
by {
  sorry
}

end line_through_point_with_equal_intercepts_l298_29829


namespace birthday_paradox_l298_29813

-- Defining the problem conditions
def people (n : ℕ) := n ≥ 367

-- Using the Pigeonhole Principle as a condition
def pigeonhole_principle (pigeonholes pigeons : ℕ) := pigeonholes < pigeons

-- Stating the final proposition
theorem birthday_paradox (n : ℕ) (days_in_year : ℕ) (h1 : days_in_year = 366) (h2 : people n) : pigeonhole_principle days_in_year n :=
sorry

end birthday_paradox_l298_29813


namespace parallel_vectors_l298_29864

def a : (ℝ × ℝ) := (1, -2)
def b (x : ℝ) : (ℝ × ℝ) := (-2, x)

theorem parallel_vectors (x : ℝ) (h : 1 / -2 = -2 / x) : x = 4 := by
  sorry

end parallel_vectors_l298_29864


namespace problem_I_problem_II_l298_29824

-- Problem (I)
def A : Set ℝ := { x | x > 2 ∨ x < -1 }
def B : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }
def A_inter_B : Set ℝ := { x | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3) }

theorem problem_I : A ∩ B = A_inter_B :=
by
  sorry

-- Problem (II)
def C (m : ℝ) : Set ℝ := { x | 2 * m - 1 < x ∧ x < m + 1 }

theorem problem_II (m : ℝ) : (C m ⊆ B) → m ≥ -1 :=
by
  sorry

end problem_I_problem_II_l298_29824


namespace compute_expression_l298_29805

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l298_29805


namespace find_m_repeated_root_l298_29842

theorem find_m_repeated_root (m : ℝ) :
  (∃ x : ℝ, (x - 1) ≠ 0 ∧ (m - 1) - x = 0) → m = 2 :=
by
  sorry

end find_m_repeated_root_l298_29842


namespace vector_coordinates_l298_29871

-- Define the given vectors.
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

-- Define the proof goal.
theorem vector_coordinates :
  -2 • a - b = (-3, -1) :=
by
  sorry -- Proof not required.

end vector_coordinates_l298_29871


namespace calc_first_term_l298_29877

theorem calc_first_term (a d : ℚ)
    (h1 : 15 * (2 * a + 29 * d) = 300)
    (h2 : 20 * (2 * a + 99 * d) = 2200) :
    a = -121 / 14 :=
by
  -- We can add the sorry placeholder here as we are not providing the complete proof steps
  sorry

end calc_first_term_l298_29877


namespace pow_mod_eq_l298_29802

theorem pow_mod_eq (n : ℕ) : 
  (3^n % 5 = 3 % 5) → 
  (3^(n+1) % 5 = (3 * 3^n) % 5) → 
  (3^(n+2) % 5 = (3 * 3^(n+1)) % 5) → 
  (3^(n+3) % 5 = (3 * 3^(n+2)) % 5) → 
  (3^4 % 5 = 1 % 5) → 
  (2023 % 4 = 3) → 
  (3^2023 % 5 = 2 % 5) :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pow_mod_eq_l298_29802


namespace pentagon_angles_sum_l298_29801

theorem pentagon_angles_sum {α β γ δ ε : ℝ} (h1 : α + β + γ + δ + ε = 180) (h2 : α = 50) :
  β + ε = 230 := 
sorry

end pentagon_angles_sum_l298_29801


namespace ad_plus_bc_eq_pm_one_l298_29850

theorem ad_plus_bc_eq_pm_one
  (a b c d : ℤ)
  (h1 : ∃ n : ℤ, n = ad + bc ∧ n ∣ a ∧ n ∣ b ∧ n ∣ c ∧ n ∣ d) :
  ad + bc = 1 ∨ ad + bc = -1 := 
sorry

end ad_plus_bc_eq_pm_one_l298_29850


namespace find_recip_sum_of_shifted_roots_l298_29856

noncomputable def reciprocal_sum_of_shifted_roots (α β γ : ℝ) (hαβγ : Polynomial.roots (Polynomial.C α * Polynomial.C β * Polynomial.C γ + Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - Polynomial.X + Polynomial.C 2) = {α, β, γ}) : ℝ :=
  1 / (α + 2) + 1 / (β + 2) + 1 / (γ + 2)

theorem find_recip_sum_of_shifted_roots (α β γ : ℝ) (hαβγ : Polynomial.roots (Polynomial.C α * Polynomial.C β * Polynomial.C γ + Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - Polynomial.X + Polynomial.C 2) = {α, β, γ}) :
  reciprocal_sum_of_shifted_roots α β γ hαβγ = -19 / 14 :=
  sorry

end find_recip_sum_of_shifted_roots_l298_29856


namespace julie_initial_savings_l298_29893

def calculate_earnings (lawns newspapers dogs : ℕ) (price_lawn price_newspaper price_dog : ℝ) : ℝ :=
  (lawns * price_lawn) + (newspapers * price_newspaper) + (dogs * price_dog)

def calculate_total_spent_bike (earnings remaining_money : ℝ) : ℝ :=
  earnings + remaining_money

def calculate_initial_savings (cost_bike total_spent : ℝ) : ℝ :=
  cost_bike - total_spent

theorem julie_initial_savings :
  let cost_bike := 2345
  let lawns := 20
  let newspapers := 600
  let dogs := 24
  let price_lawn := 20
  let price_newspaper := 0.40
  let price_dog := 15
  let remaining_money := 155
  let earnings := calculate_earnings lawns newspapers dogs price_lawn price_newspaper price_dog
  let total_spent := calculate_total_spent_bike earnings remaining_money
  calculate_initial_savings cost_bike total_spent = 1190 :=
by
  -- Although the proof is not required, this setup assumes correctness.
  sorry

end julie_initial_savings_l298_29893


namespace oil_level_drop_l298_29818

noncomputable def stationary_tank_radius : ℝ := 100
noncomputable def stationary_tank_height : ℝ := 25
noncomputable def truck_tank_radius : ℝ := 7
noncomputable def truck_tank_height : ℝ := 10

noncomputable def π : ℝ := Real.pi
noncomputable def truck_tank_volume := π * truck_tank_radius^2 * truck_tank_height
noncomputable def stationary_tank_area := π * stationary_tank_radius^2

theorem oil_level_drop (volume_truck: ℝ) (area_stationary: ℝ) : volume_truck = 490 * π → area_stationary = π * 10000 → (volume_truck / area_stationary) = 0.049 :=
by
  intros h1 h2
  sorry

end oil_level_drop_l298_29818


namespace injectivity_of_composition_l298_29896

variable {R : Type*} [LinearOrderedField R]

def injective (f : R → R) := ∀ a b, f a = f b → a = b

theorem injectivity_of_composition {f g : R → R} (h : injective (g ∘ f)) : injective f :=
by
  sorry

end injectivity_of_composition_l298_29896


namespace team_points_difference_l298_29863

   -- Definitions for points of each member
   def Max_points : ℝ := 7
   def Dulce_points : ℝ := 5
   def Val_points : ℝ := 4 * (Max_points + Dulce_points)
   def Sarah_points : ℝ := 2 * Dulce_points
   def Steve_points : ℝ := 2.5 * (Max_points + Val_points)

   -- Definition for total points of their team
   def their_team_points : ℝ := Max_points + Dulce_points + Val_points + Sarah_points + Steve_points

   -- Definition for total points of the opponents' team
   def opponents_team_points : ℝ := 200

   -- The main theorem to prove
   theorem team_points_difference : their_team_points - opponents_team_points = 7.5 := by
     sorry
   
end team_points_difference_l298_29863


namespace all_numbers_appear_on_diagonal_l298_29884

theorem all_numbers_appear_on_diagonal 
  (n : ℕ) 
  (h_odd : n % 2 = 1)
  (A : Matrix (Fin n) (Fin n) (Fin n.succ))
  (h_elements : ∀ i j, 1 ≤ A i j ∧ A i j ≤ n) 
  (h_unique_row : ∀ i k, ∃! j, A i j = k)
  (h_unique_col : ∀ j k, ∃! i, A i j = k)
  (h_symmetric : ∀ i j, A i j = A j i)
  : ∀ k, 1 ≤ k ∧ k ≤ n → ∃ i, A i i = k := 
by {
  sorry
}

end all_numbers_appear_on_diagonal_l298_29884


namespace find_a_l298_29865

theorem find_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a * b - a - b = 4) : a = 6 :=
sorry

end find_a_l298_29865


namespace granger_total_payment_proof_l298_29836

-- Conditions
def cost_per_can_spam := 3
def cost_per_jar_peanut_butter := 5
def cost_per_loaf_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Calculation
def total_cost_spam := quantity_spam * cost_per_can_spam
def total_cost_peanut_butter := quantity_peanut_butter * cost_per_jar_peanut_butter
def total_cost_bread := quantity_bread * cost_per_loaf_bread

-- Total amount paid
def total_amount_paid := total_cost_spam + total_cost_peanut_butter + total_cost_bread

-- Theorem to be proven
theorem granger_total_payment_proof : total_amount_paid = 59 :=
by
  sorry

end granger_total_payment_proof_l298_29836


namespace common_root_value_l298_29878

theorem common_root_value (p : ℝ) (hp : p > 0) : 
  (∃ x : ℝ, 3 * x ^ 2 - 4 * p * x + 9 = 0 ∧ x ^ 2 - 2 * p * x + 5 = 0) ↔ p = 3 :=
by {
  sorry
}

end common_root_value_l298_29878


namespace quadratic_range_extrema_l298_29807

def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 2

theorem quadratic_range_extrema :
  let y := quadratic
  ∃ x_max x_min,
    (x_min = -2 ∧ y x_min = -2) ∧
    (x_max = -2 ∧ y x_max = 14 ∨ x_max = 5 ∧ y x_max = 7) := 
by
  sorry

end quadratic_range_extrema_l298_29807


namespace arithmetic_mean_12_24_36_48_l298_29857

theorem arithmetic_mean_12_24_36_48 : (12 + 24 + 36 + 48) / 4 = 30 :=
by
  sorry

end arithmetic_mean_12_24_36_48_l298_29857


namespace determine_color_sum_or_product_l298_29812

theorem determine_color_sum_or_product {x : ℕ → ℝ} (h_distinct: ∀ i j : ℕ, i < j → x i < x j) (x_pos : ∀ i : ℕ, x i > 0) :
  ∃ c : ℕ → ℝ, (∀ i : ℕ, c i > 0) ∧
  (∀ i j : ℕ, i < j → (∃ r1 r2 : ℕ, (r1 ≠ r2) ∧ (c r1 + c r2 = x₆₄ + x₆₃) ∧ (c r1 * c r2 = x₆₄ * x₆₃))) :=
sorry

end determine_color_sum_or_product_l298_29812


namespace equal_roots_polynomial_l298_29834

open ComplexConjugate

theorem equal_roots_polynomial (k : ℚ) :
  (3 : ℚ) * x^2 - k * x + 2 * x + (12 : ℚ) = 0 → 
  (b : ℚ) ^ 2 - 4 * (3 : ℚ) * (12 : ℚ) = 0 ↔ k = -10 ∨ k = 14 :=
by
  sorry

end equal_roots_polynomial_l298_29834


namespace animath_interns_pigeonhole_l298_29838

theorem animath_interns_pigeonhole (n : ℕ) (knows : Fin n → Finset (Fin n)) :
  ∃ (i j : Fin n), i ≠ j ∧ (knows i).card = (knows j).card :=
by
  sorry

end animath_interns_pigeonhole_l298_29838


namespace min_value_of_expression_l298_29810

theorem min_value_of_expression (a b c : ℝ) (hb : b > a) (ha : a > c) (hc : b ≠ 0) :
  ∃ l : ℝ, l = 5.5 ∧ l ≤ (a + b)^2 / b^2 + (b + c)^2 / b^2 + (c + a)^2 / b^2 :=
by
  sorry

end min_value_of_expression_l298_29810


namespace impossible_pawn_placement_l298_29844

theorem impossible_pawn_placement :
  ¬(∃ a b c : ℕ, a + b + c = 50 ∧ 
  ∀ (x y z : ℕ), 2 * a ≤ x ∧ x ≤ 2 * b ∧ 2 * b ≤ y ∧ y ≤ 2 * c ∧ 2 * c ≤ z ∧ z ≤ 2 * a) := sorry

end impossible_pawn_placement_l298_29844


namespace range_of_a_l298_29854

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2 * a)^x < (3 - 2 * a)^y

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : (1 ≤ a ∧ a < 2) ∨ (a ≤ -2) :=
sorry

end range_of_a_l298_29854


namespace find_number_l298_29879

theorem find_number (x : ℝ) (h : x / 0.07 = 700) : x = 49 :=
sorry

end find_number_l298_29879


namespace cartons_in_load_l298_29843

theorem cartons_in_load 
  (crate_weight : ℕ)
  (carton_weight : ℕ)
  (num_crates : ℕ)
  (total_load_weight : ℕ)
  (h1 : crate_weight = 4)
  (h2 : carton_weight = 3)
  (h3 : num_crates = 12)
  (h4 : total_load_weight = 96) :
  ∃ C : ℕ, num_crates * crate_weight + C * carton_weight = total_load_weight ∧ C = 16 := 
by 
  sorry

end cartons_in_load_l298_29843


namespace toothpick_pattern_15th_stage_l298_29897

theorem toothpick_pattern_15th_stage :
  let a₁ := 5
  let d := 3
  let n := 15
  a₁ + (n - 1) * d = 47 :=
by
  sorry

end toothpick_pattern_15th_stage_l298_29897


namespace toll_for_18_wheel_truck_l298_29826

-- Definitions based on conditions
def num_axles (total_wheels : ℕ) (wheels_front_axle : ℕ) (wheels_per_other_axle : ℕ) : ℕ :=
  1 + (total_wheels - wheels_front_axle) / wheels_per_other_axle

def toll (x : ℕ) : ℝ :=
  0.50 + 0.50 * (x - 2)

-- The problem statement to prove
theorem toll_for_18_wheel_truck : toll (num_axles 18 2 4) = 2.00 := by
  sorry

end toll_for_18_wheel_truck_l298_29826


namespace investment_ratio_l298_29876

theorem investment_ratio (P Q : ℝ) (h1 : (P * 5) / (Q * 9) = 7 / 9) : P / Q = 7 / 5 :=
by sorry

end investment_ratio_l298_29876


namespace sum_of_roots_quadratic_eq_l298_29841

theorem sum_of_roots_quadratic_eq (x₁ x₂ : ℝ) (h : x₁^2 + 2 * x₁ - 4 = 0 ∧ x₂^2 + 2 * x₂ - 4 = 0) : 
  x₁ + x₂ = -2 :=
sorry

end sum_of_roots_quadratic_eq_l298_29841


namespace mn_sum_value_l298_29890

-- Definition of the problem conditions
def faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_consecutive (a b : ℕ) : Prop :=
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨
  (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3) ∨
  (a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨
  (a = 5 ∧ b = 6) ∨ (a = 6 ∧ b = 5) ∨
  (a = 6 ∧ b = 7) ∨ (a = 7 ∧ b = 6) ∨
  (a = 7 ∧ b = 8) ∨ (a = 8 ∧ b = 7) ∨
  (a = 8 ∧ b = 9) ∨ (a = 9 ∧ b = 8) ∨
  (a = 9 ∧ b = 1) ∨ (a = 1 ∧ b = 9)

noncomputable def m_n_sum : ℕ :=
  let total_permutations := 5040
  let valid_permutations := 60
  let probability := valid_permutations / total_permutations
  let m := 1
  let n := total_permutations / valid_permutations
  m + n

theorem mn_sum_value : m_n_sum = 85 :=
  sorry

end mn_sum_value_l298_29890


namespace percentage_difference_is_20_l298_29835

/-
Barry can reach apples that are 5 feet high.
Larry is 5 feet tall.
When Barry stands on Larry's shoulders, they can reach 9 feet high.
-/
def Barry_height : ℝ := 5
def Larry_height : ℝ := 5
def Combined_height : ℝ := 9

/-
Prove the percentage difference between Larry's full height and his shoulder height is 20%.
-/
theorem percentage_difference_is_20 :
  ((Larry_height - (Combined_height - Barry_height)) / Larry_height) * 100 = 20 :=
by
  sorry

end percentage_difference_is_20_l298_29835


namespace container_could_be_emptied_l298_29880

theorem container_could_be_emptied (a b c : ℕ) (h : 0 ≤ a ∧ a ≤ b ∧ b ≤ c) :
  ∃ (a' b' c' : ℕ), (a' = 0 ∨ b' = 0 ∨ c' = 0) ∧
  (∀ x y z : ℕ, (a, b, c) = (x, y, z) → (a', b', c') = (y + y, z - y, x - y)) :=
sorry

end container_could_be_emptied_l298_29880


namespace value_of_f_neg6_l298_29853

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = -f x

theorem value_of_f_neg6 : f (-6) = 0 :=
by
  sorry

end value_of_f_neg6_l298_29853


namespace rationalize_denominator_l298_29882

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l298_29882


namespace domain_of_f_l298_29891

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2

theorem domain_of_f :
  {x : ℝ | x + 1 > 0} = {x : ℝ | x > -1} :=
by
  sorry

end domain_of_f_l298_29891


namespace ratio_larva_to_cocoon_l298_29894

theorem ratio_larva_to_cocoon (total_days : ℕ) (cocoon_days : ℕ)
  (h1 : total_days = 120) (h2 : cocoon_days = 30) :
  (total_days - cocoon_days) / cocoon_days = 3 := by
  sorry

end ratio_larva_to_cocoon_l298_29894


namespace total_volume_of_four_boxes_l298_29846

-- Define the edge length of the cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_one_cube := edge_length ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 4

-- The total volume of the four cubes
def total_volume := number_of_cubes * volume_of_one_cube

-- Statement to prove that the total volume equals 500 cubic feet
theorem total_volume_of_four_boxes :
  total_volume = 500 :=
sorry

end total_volume_of_four_boxes_l298_29846


namespace area_C_greater_than_sum_area_A_B_by_159_point_2_percent_l298_29849

-- Define the side lengths of squares A, B, and C
def side_length_A (s : ℝ) : ℝ := s
def side_length_B (s : ℝ) : ℝ := 2 * s
def side_length_C (s : ℝ) : ℝ := 3.6 * s

-- Define the areas of squares A, B, and C
def area_A (s : ℝ) : ℝ := (side_length_A s) ^ 2
def area_B (s : ℝ) : ℝ := (side_length_B s) ^ 2
def area_C (s : ℝ) : ℝ := (side_length_C s) ^ 2

-- Define the sum of areas of squares A and B
def sum_area_A_B (s : ℝ) : ℝ := area_A s + area_B s

-- Prove that the area of square C is 159.2% greater than the sum of areas of squares A and B
theorem area_C_greater_than_sum_area_A_B_by_159_point_2_percent (s : ℝ) : 
  ((area_C s - sum_area_A_B s) / (sum_area_A_B s)) * 100 = 159.2 := 
sorry

end area_C_greater_than_sum_area_A_B_by_159_point_2_percent_l298_29849


namespace work_completion_by_b_l298_29809

theorem work_completion_by_b (a_days : ℕ) (a_solo_days : ℕ) (a_b_combined_days : ℕ) (b_days : ℕ) :
  a_days = 12 ∧ a_solo_days = 3 ∧ a_b_combined_days = 5 → b_days = 15 :=
by
  sorry

end work_completion_by_b_l298_29809


namespace slow_car_speed_l298_29816

theorem slow_car_speed (x : ℝ) (hx : 0 < x) (distance : ℝ) (delay : ℝ) (fast_factor : ℝ) :
  distance = 60 ∧ delay = 0.5 ∧ fast_factor = 1.5 ∧ 
  (distance / x) - (distance / (fast_factor * x)) = delay → 
  x = 40 :=
by
  intros h
  sorry

end slow_car_speed_l298_29816


namespace abs_sum_eq_abs_add_iff_ab_gt_zero_l298_29860

theorem abs_sum_eq_abs_add_iff_ab_gt_zero (a b : ℝ) :
  (|a + b| = |a| + |b|) → (a = 0 ∧ b = 0 ∨ ab > 0) :=
sorry

end abs_sum_eq_abs_add_iff_ab_gt_zero_l298_29860


namespace min_value_expr_l298_29881

variable (a b : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = 1)

theorem min_value_expr : (1 / (2 * a)) + (1 / (2 * b)) + (8 / (a + b)) ≥ 4 :=
by
  sorry

end min_value_expr_l298_29881


namespace land_value_moon_l298_29819

-- Define the conditions
def surface_area_earth : ℕ := 200
def surface_area_ratio : ℕ := 5
def value_ratio : ℕ := 6
def total_value_earth : ℕ := 80

-- Define the question and the expected answer
noncomputable def total_value_moon : ℕ := 96

-- State the proof problem
theorem land_value_moon :
  (surface_area_earth / surface_area_ratio * value_ratio) * (surface_area_earth / surface_area_ratio) = total_value_moon := 
sorry

end land_value_moon_l298_29819


namespace trapezoid_height_l298_29825

theorem trapezoid_height (BC AD AB CD h : ℝ) (hBC : BC = 4) (hAD : AD = 25) (hAB : AB = 20) (hCD : CD = 13) :
  h = 12 :=
by
  sorry

end trapezoid_height_l298_29825


namespace remainder_when_divided_by_x_minus_2_l298_29858

def f (x : ℝ) : ℝ := x^5 + 2 * x^3 + x^2 + 4

theorem remainder_when_divided_by_x_minus_2 : f 2 = 56 :=
by
  -- Proof steps will go here.
  sorry

end remainder_when_divided_by_x_minus_2_l298_29858


namespace value_is_sqrt_5_over_3_l298_29862

noncomputable def findValue (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x / y + y / x = 8) : ℝ :=
  (x + y) / (x - y)

theorem value_is_sqrt_5_over_3 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x / y + y / x = 8) :
  findValue x y h1 h2 h3 = Real.sqrt (5 / 3) :=
sorry

end value_is_sqrt_5_over_3_l298_29862


namespace part1_part2_l298_29808

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l298_29808


namespace monotonicity_and_range_of_m_l298_29861

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (1 - a) / 2 * x ^ 2 + a * x - Real.log x

theorem monotonicity_and_range_of_m (a m : ℝ) (h₀ : 2 < a) (h₁ : a < 3)
  (h₂ : ∀ (x1 x2 : ℝ), 1 ≤ x1 ∧ x1 ≤ 2 → 1 ≤ x2 ∧ x2 ≤ 2 -> ma + Real.log 2 > |f x1 a - f x2 a|):
  m ≥ 0 :=
sorry

end monotonicity_and_range_of_m_l298_29861


namespace shelter_total_cats_l298_29848

theorem shelter_total_cats (total_adult_cats num_female_cats num_litters avg_kittens_per_litter : ℕ) 
  (h1 : total_adult_cats = 150) 
  (h2 : num_female_cats = 2 * total_adult_cats / 3)
  (h3 : num_litters = 2 * num_female_cats / 3)
  (h4 : avg_kittens_per_litter = 5):
  total_adult_cats + num_litters * avg_kittens_per_litter = 480 :=
by
  sorry

end shelter_total_cats_l298_29848


namespace find_making_lines_parallel_l298_29840

theorem find_making_lines_parallel (m : ℝ) : 
  let line1_slope := -1 / (1 + m)
  let line2_slope := -m / 2 
  (line1_slope = line2_slope) ↔ (m = 1) := 
by
  -- definitions
  intros
  let line1_slope := -1 / (1 + m)
  let line2_slope := -m / 2
  -- equation for slopes to be equal
  have slope_equation : line1_slope = line2_slope ↔ (m = 1)
  sorry

  exact slope_equation

end find_making_lines_parallel_l298_29840


namespace find_brown_mms_second_bag_l298_29889

variable (x : ℕ)

-- Definitions based on the conditions
def BrownMmsFirstBag := 9
def BrownMmsThirdBag := 8
def BrownMmsFourthBag := 8
def BrownMmsFifthBag := 3
def AveBrownMmsPerBag := 8
def NumBags := 5

-- Condition specifying the average brown M&Ms per bag
axiom average_condition : AveBrownMmsPerBag = (BrownMmsFirstBag + x + BrownMmsThirdBag + BrownMmsFourthBag + BrownMmsFifthBag) / NumBags

-- Prove the number of brown M&Ms in the second bag
theorem find_brown_mms_second_bag : x = 12 := by
  sorry

end find_brown_mms_second_bag_l298_29889


namespace range_of_m_range_of_x_l298_29859

variable {a b m : ℝ}

-- Given conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom sum_eq_one : a + b = 1

-- Problem (I): Prove range of m
theorem range_of_m (h : ab ≤ m) : m ≥ 1 / 4 := by
  sorry

variable {x : ℝ}

-- Problem (II): Prove range of x
theorem range_of_x (h : 4 / a + 1 / b ≥ |2 * x - 1| - |x + 2|) : -2 ≤ x ∧ x ≤ 6 := by
  sorry

end range_of_m_range_of_x_l298_29859


namespace book_has_50_pages_l298_29820

noncomputable def sentences_per_hour : ℕ := 200
noncomputable def hours_to_read : ℕ := 50
noncomputable def sentences_per_paragraph : ℕ := 10
noncomputable def paragraphs_per_page : ℕ := 20

theorem book_has_50_pages :
  (sentences_per_hour * hours_to_read) / sentences_per_paragraph / paragraphs_per_page = 50 :=
by
  sorry

end book_has_50_pages_l298_29820


namespace problem_inequality_l298_29898

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem problem_inequality (a x : ℝ) (h : a ∈ Set.Iic (-1/Real.exp 2)) :
  f a x ≥ 2 * a * x - x * Real.exp (a * x - 1) := 
sorry

end problem_inequality_l298_29898


namespace more_than_four_numbers_make_polynomial_prime_l298_29855

def polynomial (n : ℕ) : ℤ := n^3 - 10 * n^2 + 31 * n - 17

def is_prime (k : ℤ) : Prop :=
  k > 1 ∧ ∀ m : ℤ, m > 1 ∧ m < k → ¬ (m ∣ k)

theorem more_than_four_numbers_make_polynomial_prime :
  (∃ n1 n2 n3 n4 n5 : ℕ, 
    n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧ n4 > 0 ∧ n5 > 0 ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ n1 ≠ n5 ∧
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ n2 ≠ n5 ∧ 
    n3 ≠ n4 ∧ n3 ≠ n5 ∧ 
    n4 ≠ n5 ∧ 
    is_prime (polynomial n1) ∧
    is_prime (polynomial n2) ∧
    is_prime (polynomial n3) ∧
    is_prime (polynomial n4) ∧
    is_prime (polynomial n5)) :=
sorry

end more_than_four_numbers_make_polynomial_prime_l298_29855


namespace shortest_chord_length_l298_29866

theorem shortest_chord_length 
  (C : ℝ → ℝ → Prop) 
  (l : ℝ → ℝ → ℝ → Prop) 
  (radius : ℝ) 
  (center_x center_y : ℝ) 
  (cx cy : ℝ) 
  (m : ℝ) :
  (∀ x y, C x y ↔ (x - 1)^2 + (y - 2)^2 = 25) →
  (∀ x y m, l x y m ↔ (2*m+1)*x + (m+1)*y - 7*m - 4 = 0) →
  center_x = 1 →
  center_y = 2 →
  radius = 5 →
  cx = 3 →
  cy = 1 →
  ∃ shortest_chord_length : ℝ, shortest_chord_length = 4 * Real.sqrt 5 := sorry

end shortest_chord_length_l298_29866


namespace simplification_evaluation_l298_29839

theorem simplification_evaluation (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  ( (2 * x - 6) / (x - 2) ) / ( (5 / (x - 2)) - (x + 2) ) = Real.sqrt 2 - 2 :=
sorry

end simplification_evaluation_l298_29839


namespace find_n_value_l298_29851

theorem find_n_value (m n k : ℝ) (h1 : n = k / m) (h2 : m = k / 2) (h3 : k ≠ 0): n = 2 :=
sorry

end find_n_value_l298_29851


namespace red_to_blue_l298_29811

def is_red (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2020

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n ∧ ∃ m : ℕ, n = m ^ 2019

theorem red_to_blue (n : ℕ) (hn : n > 10^100000000) (hnred : is_red n) 
    (hn1red : is_red (n+1)) :
    ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 2019 ∧ is_blue (n + k) :=
sorry

end red_to_blue_l298_29811


namespace fraction_sum_condition_l298_29899

theorem fraction_sum_condition 
  (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0)
  (h : x + y = x * y): 
  (1/x + 1/y = 1) :=
by
  sorry

end fraction_sum_condition_l298_29899


namespace calculate_T6_l298_29822

noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + 1 / y^m

theorem calculate_T6 (y : ℝ) (h : y + 1 / y = 5) : T y 6 = 12098 := 
by
  sorry

end calculate_T6_l298_29822


namespace probability_two_tails_after_two_heads_l298_29873

noncomputable def fair_coin_probability : ℚ :=
  -- Given conditions:
  let p_head := (1 : ℚ) / 2
  let p_tail := (1 : ℚ) / 2

  -- Define the probability Q as stated in the problem
  let Q := ((1 : ℚ) / 4) / (1 - (1 : ℚ) / 4)

  -- Calculate the probability of starting with sequence "HTH"
  let p_HTH := p_head * p_tail * p_head

  -- Calculate the final probability
  p_HTH * Q

theorem probability_two_tails_after_two_heads :
  fair_coin_probability = (1 : ℚ) / 24 :=
by
  sorry

end probability_two_tails_after_two_heads_l298_29873


namespace at_least_50_singers_l298_29827

def youth_summer_village (total people_not_working people_with_families max_subset : ℕ) : Prop :=
  total = 100 ∧ 
  people_not_working = 50 ∧ 
  people_with_families = 25 ∧ 
  max_subset = 50

theorem at_least_50_singers (S : ℕ) (h : youth_summer_village 100 50 25 50) : S ≥ 50 :=
by
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end at_least_50_singers_l298_29827


namespace sin_2B_sin_A_sin_C_eq_neg_7_over_8_l298_29886

theorem sin_2B_sin_A_sin_C_eq_neg_7_over_8
    (A B C : ℝ)
    (a b c : ℝ)
    (h1 : (2 * a + c) * Real.cos B + b * Real.cos C = 0)
    (h2 : 1/2 * a * c * Real.sin B = 15 * Real.sqrt 3)
    (h3 : a + b + c = 30) :
    (2 * Real.sin B * Real.cos B) / (Real.sin A + Real.sin C) = -7/8 := 
sorry

end sin_2B_sin_A_sin_C_eq_neg_7_over_8_l298_29886


namespace find_xyz_l298_29852

open Complex

theorem find_xyz (a b c x y z : ℂ)
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0)
  (ha : a = (b + c) / (x + 1))
  (hb : b = (a + c) / (y + 1))
  (hc : c = (a + b) / (z + 1))
  (hxy_z_1 : x * y + x * z + y * z = 9)
  (hxy_z_2 : x + y + z = 5) :
  x * y * z = 13 := 
sorry

end find_xyz_l298_29852


namespace movie_marathon_duration_l298_29874

theorem movie_marathon_duration :
  let first_movie := 2
  let second_movie := first_movie + 0.5 * first_movie
  let combined_time := first_movie + second_movie
  let third_movie := combined_time - 1
  first_movie + second_movie + third_movie = 9 := by
  sorry

end movie_marathon_duration_l298_29874


namespace cindy_correct_answer_l298_29870

theorem cindy_correct_answer (x : ℝ) (h : (x - 10) / 5 = 50) : (x - 5) / 10 = 25.5 :=
sorry

end cindy_correct_answer_l298_29870


namespace FerrisWheelCostIsTwo_l298_29803

noncomputable def costFerrisWheel (rollerCoasterCost multipleRideDiscount coupon totalTicketsBought : ℝ) : ℝ :=
  totalTicketsBought + multipleRideDiscount + coupon - rollerCoasterCost

theorem FerrisWheelCostIsTwo :
  let rollerCoasterCost := 7.0
  let multipleRideDiscount := 1.0
  let coupon := 1.0
  let totalTicketsBought := 7.0
  costFerrisWheel rollerCoasterCost multipleRideDiscount coupon totalTicketsBought = 2.0 :=
by
  sorry

end FerrisWheelCostIsTwo_l298_29803


namespace least_possible_value_d_l298_29872

theorem least_possible_value_d 
  (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (hxy : x < y)
  (hyz : y < z)
  (hyx_gt_five : y - x > 5) : 
  z - x = 9 :=
sorry

end least_possible_value_d_l298_29872


namespace find_x_l298_29823

variable (x : ℝ)

theorem find_x (h : (15 - 2 + 4 / 1 / 2) * x = 77) : x = 77 / (15 - 2 + 4 / 1 / 2) :=
by sorry

end find_x_l298_29823


namespace negation_of_at_least_three_is_at_most_two_l298_29831

theorem negation_of_at_least_three_is_at_most_two :
  (¬ (∀ n : ℕ, n ≥ 3)) ↔ (∃ n : ℕ, n ≤ 2) :=
sorry

end negation_of_at_least_three_is_at_most_two_l298_29831


namespace inscribed_angle_sum_l298_29845

theorem inscribed_angle_sum : 
  let arcs := 24 
  let arc_to_angle (n : ℕ) := 360 / arcs * n / 2 
  (arc_to_angle 4 + arc_to_angle 6 = 75) :=
by
  sorry

end inscribed_angle_sum_l298_29845


namespace ways_to_turn_off_lights_l298_29837

-- Define the problem conditions
def streetlights := 12
def can_turn_off := 3
def not_turn_off_at_ends := true
def not_adjacent := true

-- The theorem to be proved
theorem ways_to_turn_off_lights : 
  ∃ n, 
  streetlights = 12 ∧ 
  can_turn_off = 3 ∧ 
  not_turn_off_at_ends ∧ 
  not_adjacent ∧ 
  n = 56 :=
by 
  sorry

end ways_to_turn_off_lights_l298_29837


namespace exists_positive_n_l298_29868

theorem exists_positive_n {k : ℕ} (h_k : 0 < k) {m : ℕ} (h_m : m % 2 = 1) :
  ∃ n : ℕ, 0 < n ∧ (n^n - m) % 2^k = 0 := 
sorry

end exists_positive_n_l298_29868


namespace real_solutions_l298_29888

noncomputable def solveEquation (x : ℝ) : Prop :=
  (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10

theorem real_solutions :
  {x : ℝ | solveEquation x} = {x : ℝ | x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15} :=
by
  sorry

end real_solutions_l298_29888


namespace determine_specialty_l298_29832

variables 
  (Peter_is_mathematician Sergey_is_physicist Roman_is_physicist : Prop)
  (Peter_is_chemist Sergey_is_mathematician Roman_is_chemist : Prop)

-- Conditions
axiom cond1 : Peter_is_mathematician → ¬ Sergey_is_physicist
axiom cond2 : ¬ Roman_is_physicist → Peter_is_mathematician
axiom cond3 : ¬ Sergey_is_mathematician → Roman_is_chemist

theorem determine_specialty 
  (h1 : ¬ Roman_is_physicist)
: Peter_is_chemist ∧ Sergey_is_mathematician ∧ Roman_is_physicist := 
by sorry

end determine_specialty_l298_29832


namespace nested_composition_l298_29867

def g (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem nested_composition : g (g (g (g (g (g 2))))) = 2 := by
  sorry

end nested_composition_l298_29867


namespace find_common_ratio_l298_29821

noncomputable def a_n (n : ℕ) (q : ℚ) : ℚ :=
  if n = 1 then 1 / 8 else (q^(n - 1)) * (1 / 8)

theorem find_common_ratio (q : ℚ) :
  (a_n 4 q = -1) ↔ (q = -2) :=
by
  sorry

end find_common_ratio_l298_29821


namespace bacteria_growth_time_l298_29875
-- Import necessary library

-- Define the conditions
def initial_bacteria_count : ℕ := 100
def final_bacteria_count : ℕ := 102400
def multiplication_factor : ℕ := 4
def multiplication_period_hours : ℕ := 6

-- Define the proof problem
theorem bacteria_growth_time :
  ∃ t : ℕ, t * multiplication_period_hours = 30 ∧ initial_bacteria_count * multiplication_factor^t = final_bacteria_count :=
by
  sorry

end bacteria_growth_time_l298_29875


namespace divide_5000_among_x_and_y_l298_29830

theorem divide_5000_among_x_and_y (total_amount : ℝ) (ratio_x : ℝ) (ratio_y : ℝ) (parts : ℝ) :
  total_amount = 5000 → ratio_x = 2 → ratio_y = 8 → parts = ratio_x + ratio_y → 
  (total_amount / parts) * ratio_x = 1000 := 
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end divide_5000_among_x_and_y_l298_29830


namespace total_gallons_l298_29833

-- Definitions from conditions
def num_vans : ℕ := 6
def standard_capacity : ℕ := 8000
def reduced_capacity : ℕ := standard_capacity - (30 * standard_capacity / 100)
def increased_capacity : ℕ := standard_capacity + (50 * standard_capacity / 100)

-- Total number of specific types of vans
def num_standard_vans : ℕ := 2
def num_reduced_vans : ℕ := 1
def num_increased_vans : ℕ := num_vans - num_standard_vans - num_reduced_vans

-- The proof goal
theorem total_gallons : 
  (num_standard_vans * standard_capacity) + 
  (num_reduced_vans * reduced_capacity) + 
  (num_increased_vans * increased_capacity) = 
  57600 := 
by
  -- The necessary proof can be filled here
  sorry

end total_gallons_l298_29833


namespace most_likely_units_digit_sum_is_zero_l298_29847

theorem most_likely_units_digit_sum_is_zero :
  ∃ (units_digit : ℕ), 
  (∀ m n : ℕ, (1 ≤ m ∧ m ≤ 9) ∧ (1 ≤ n ∧ n ≤ 9) → 
    units_digit = (m + n) % 10) ∧ 
  units_digit = 0 :=
sorry

end most_likely_units_digit_sum_is_zero_l298_29847
