import Mathlib

namespace NUMINAMATH_GPT_bankers_gain_is_60_l691_69135

def banker's_gain (BD F PV R T : ℝ) : ℝ :=
  let TD := F - PV
  BD - TD

theorem bankers_gain_is_60 (BD F PV R T BG : ℝ) (h₁ : BD = 260) (h₂ : R = 0.10) (h₃ : T = 3)
  (h₄ : F = 260 / 0.3) (h₅ : PV = F / (1 + (R * T))) :
  banker's_gain BD F PV R T = 60 :=
by
  rw [banker's_gain, h₄, h₅]
  -- Further simplifications and exact equality steps would be added here with actual proof steps
  sorry

end NUMINAMATH_GPT_bankers_gain_is_60_l691_69135


namespace NUMINAMATH_GPT_range_of_m_l691_69130

theorem range_of_m (k : ℝ) (m : ℝ) (y x : ℝ)
  (h1 : ∀ x, y = k * (x - 1) + m)
  (h2 : y = 3 ∧ x = -2)
  (h3 : (∃ x, x < 0 ∧ y > 0) ∧ (∃ x, x < 0 ∧ y < 0) ∧ (∃ x, x > 0 ∧ y < 0)) :
  m < - (3 / 2) :=
sorry

end NUMINAMATH_GPT_range_of_m_l691_69130


namespace NUMINAMATH_GPT_A_finishes_work_in_8_days_l691_69181

theorem A_finishes_work_in_8_days 
  (A_work B_work W : ℝ) 
  (h1 : 4 * A_work + 6 * B_work = W)
  (h2 : (A_work + B_work) * 4.8 = W) :
  A_work = W / 8 :=
by
  -- We should provide the proof here, but we will use "sorry" for now.
  sorry

end NUMINAMATH_GPT_A_finishes_work_in_8_days_l691_69181


namespace NUMINAMATH_GPT_count_possible_values_l691_69144

open Nat

def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

def is_valid_addition (A B C D : ℕ) : Prop :=
  ∀ x y z w v u : ℕ, 
  (x = A) ∧ (y = B) ∧ (z = C) ∧ (w = D) ∧ (v = B) ∧ (u = D) →
  (A + C = D) ∧ (A + D = B) ∧ (B + B = D) ∧ (D + D = C)

theorem count_possible_values : ∀ (A B C D : ℕ), 
  distinct_digits A B C D → is_valid_addition A B C D → num_of_possible_D = 4 :=
by
  intro A B C D hd hv
  sorry

end NUMINAMATH_GPT_count_possible_values_l691_69144


namespace NUMINAMATH_GPT_tea_bags_number_l691_69192

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_tea_bags_number_l691_69192


namespace NUMINAMATH_GPT_bat_wings_area_l691_69116

structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 0⟩
def Q : Point := ⟨5, 0⟩
def R : Point := ⟨5, 2⟩
def S : Point := ⟨0, 2⟩
def A : Point := ⟨5, 1⟩
def T : Point := ⟨3, 2⟩

def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

theorem bat_wings_area :
  area_triangle P A T = 5.5 :=
sorry

end NUMINAMATH_GPT_bat_wings_area_l691_69116


namespace NUMINAMATH_GPT_percentage_of_Luccas_balls_are_basketballs_l691_69109

-- Defining the variables and their conditions 
variables (P : ℝ) (Lucca_Balls : ℕ := 100) (Lucien_Balls : ℕ := 200)
variable (Total_Basketballs : ℕ := 50)

-- Condition that Lucien has 20% basketballs
def Lucien_Basketballs := (20 / 100) * Lucien_Balls

-- We need to prove that percentage of Lucca's balls that are basketballs is 10%
theorem percentage_of_Luccas_balls_are_basketballs :
  (P / 100) * Lucca_Balls + Lucien_Basketballs = Total_Basketballs → P = 10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_Luccas_balls_are_basketballs_l691_69109


namespace NUMINAMATH_GPT_ratio_sheep_to_horses_l691_69139

theorem ratio_sheep_to_horses (sheep horses : ℕ) (total_horse_food daily_food_per_horse : ℕ)
  (h1 : sheep = 16)
  (h2 : total_horse_food = 12880)
  (h3 : daily_food_per_horse = 230)
  (h4 : horses = total_horse_food / daily_food_per_horse) :
  (sheep / gcd sheep horses) / (horses / gcd sheep horses) = 2 / 7 := by
  sorry

end NUMINAMATH_GPT_ratio_sheep_to_horses_l691_69139


namespace NUMINAMATH_GPT_equidistant_point_l691_69186

theorem equidistant_point (x y : ℝ) :
  (abs x = abs y) → (abs x = abs (x + y - 3) / (Real.sqrt 2)) → x = 1.5 :=
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_equidistant_point_l691_69186


namespace NUMINAMATH_GPT_find_possible_values_l691_69182

noncomputable def complex_values (x y : ℂ) : Prop :=
  (x^2 + y^2) / (x + y) = 4 ∧ (x^4 + y^4) / (x^3 + y^3) = 2

theorem find_possible_values (x y : ℂ) (h : complex_values x y) :
  ∃ z : ℂ, z = (x^6 + y^6) / (x^5 + y^5) ∧ (z = 10 + 2 * Real.sqrt 17 ∨ z = 10 - 2 * Real.sqrt 17) :=
sorry

end NUMINAMATH_GPT_find_possible_values_l691_69182


namespace NUMINAMATH_GPT_largest_angle_of_pentagon_l691_69161

theorem largest_angle_of_pentagon (x : ℝ) : 
  (2*x + 2) + 3*x + 4*x + 5*x + (6*x - 2) = 540 → 
  6*x - 2 = 160 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_largest_angle_of_pentagon_l691_69161


namespace NUMINAMATH_GPT_cupcakes_left_l691_69127

def num_packages : ℝ := 3.5
def cupcakes_per_package : ℝ := 7
def cupcakes_eaten : ℝ := 5.75

theorem cupcakes_left :
  num_packages * cupcakes_per_package - cupcakes_eaten = 18.75 :=
by
  sorry

end NUMINAMATH_GPT_cupcakes_left_l691_69127


namespace NUMINAMATH_GPT_baseball_cards_per_friend_l691_69158

theorem baseball_cards_per_friend (total_cards friends : ℕ) (h_total : total_cards = 24) (h_friends : friends = 4) : total_cards / friends = 6 :=
by
  sorry

end NUMINAMATH_GPT_baseball_cards_per_friend_l691_69158


namespace NUMINAMATH_GPT_x_squared_eq_y_squared_iff_x_eq_y_or_neg_y_x_squared_eq_y_squared_necessary_but_not_sufficient_l691_69112

theorem x_squared_eq_y_squared_iff_x_eq_y_or_neg_y (x y : ℝ) : 
  (x^2 = y^2) ↔ (x = y ∨ x = -y) := by
  sorry

theorem x_squared_eq_y_squared_necessary_but_not_sufficient (x y : ℝ) :
  (x^2 = y^2 → x = y) ↔ false := by
  sorry

end NUMINAMATH_GPT_x_squared_eq_y_squared_iff_x_eq_y_or_neg_y_x_squared_eq_y_squared_necessary_but_not_sufficient_l691_69112


namespace NUMINAMATH_GPT_algebraic_expression_value_l691_69102

variables (x y : ℝ)

theorem algebraic_expression_value :
  x^2 - 4 * x - 1 = 0 →
  (2 * x - 3)^2 - (x + y) * (x - y) - y^2 = 12 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l691_69102


namespace NUMINAMATH_GPT_total_genuine_purses_and_handbags_l691_69115

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem total_genuine_purses_and_handbags : GenuinePurses + GenuineHandbags = 31 := by
  sorry

end NUMINAMATH_GPT_total_genuine_purses_and_handbags_l691_69115


namespace NUMINAMATH_GPT_restore_price_by_percentage_l691_69171

theorem restore_price_by_percentage 
  (p : ℝ) -- original price
  (h₀ : p > 0) -- condition that price is positive
  (r₁ : ℝ := 0.25) -- reduction of 25%
  (r₁_applied : ℝ := p * (1 - r₁)) -- first reduction
  (r₂ : ℝ := 0.20) -- additional reduction of 20%
  (r₂_applied : ℝ := r₁_applied * (1 - r₂)) -- second reduction
  (final_price : ℝ := r₂_applied) -- final price after two reductions
  (increase_needed : ℝ := p - final_price) -- amount to increase to restore the price
  (percent_increase : ℝ := (increase_needed / final_price) * 100) -- percentage increase needed
  : abs (percent_increase - 66.67) < 0.01 := -- proof that percentage increase is approximately 66.67%
sorry

end NUMINAMATH_GPT_restore_price_by_percentage_l691_69171


namespace NUMINAMATH_GPT_min_value_proof_l691_69198

noncomputable def min_value (x y : ℝ) : ℝ := 1 / x + 1 / (2 * y)

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 * y = 1) :
  min_value x y = 4 :=
sorry

end NUMINAMATH_GPT_min_value_proof_l691_69198


namespace NUMINAMATH_GPT_ratio_of_numbers_l691_69125

theorem ratio_of_numbers (A B : ℕ) (HCF_AB : Nat.gcd A B = 3) (LCM_AB : Nat.lcm A B = 36) : 
  A / B = 3 / 4 :=
sorry

end NUMINAMATH_GPT_ratio_of_numbers_l691_69125


namespace NUMINAMATH_GPT_tangent_parallel_to_line_l691_69150

theorem tangent_parallel_to_line (x y : ℝ) :
  (y = x^3 + x - 1) ∧ (3 * x^2 + 1 = 4) → (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -3) := by
  sorry

end NUMINAMATH_GPT_tangent_parallel_to_line_l691_69150


namespace NUMINAMATH_GPT_initial_bananas_tree_l691_69145

-- Definitions for the conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten_by_raj : ℕ := 70
def bananas_in_basket_of_raj := 2 * bananas_eaten_by_raj
def bananas_cut_from_tree := bananas_eaten_by_raj + bananas_in_basket_of_raj
def initial_bananas_on_tree := bananas_cut_from_tree + bananas_left_on_tree

-- The theorem to be proven
theorem initial_bananas_tree : initial_bananas_on_tree = 310 :=
by sorry

end NUMINAMATH_GPT_initial_bananas_tree_l691_69145


namespace NUMINAMATH_GPT_b_coordinates_bc_equation_l691_69154

section GeometryProof

-- Define point A
def A : ℝ × ℝ := (1, 1)

-- Altitude CD has the equation: 3x + y - 12 = 0
def altitude_CD (x y : ℝ) : Prop := 3 * x + y - 12 = 0

-- Angle bisector BE has the equation: x - 2y + 4 = 0
def angle_bisector_BE (x y : ℝ) : Prop := x - 2 * y + 4 = 0

-- Coordinates of point B
def B : ℝ × ℝ := (-8, -2)

-- Equation of line BC
def line_BC (x y : ℝ) : Prop := 9 * x - 13 * y + 46 = 0

-- Proof statement for the coordinates of point B
theorem b_coordinates : ∃ x y : ℝ, (x, y) = B :=
by sorry

-- Proof statement for the equation of line BC
theorem bc_equation : ∃ (f : ℝ → ℝ → Prop), f = line_BC :=
by sorry

end GeometryProof

end NUMINAMATH_GPT_b_coordinates_bc_equation_l691_69154


namespace NUMINAMATH_GPT_max_unsealed_windows_l691_69138

-- Definitions of conditions for the problem
def windows : Nat := 15
def panes : Nat := 15

-- Definition of the matching and selection process conditions
def matched_panes (window pane : Nat) : Prop :=
  pane >= window

-- Proof problem statement
theorem max_unsealed_windows 
  (glazier_approaches_window : ∀ (current_window : Nat), ∃ pane : Nat, pane >= current_window) :
  ∃ (max_unsealed : Nat), max_unsealed = 7 :=
by
  sorry

end NUMINAMATH_GPT_max_unsealed_windows_l691_69138


namespace NUMINAMATH_GPT_find_x_values_l691_69113

theorem find_x_values (x : ℝ) (h : x ≠ 5) : x + 36 / (x - 5) = -12 ↔ x = -8 ∨ x = 3 :=
by sorry

end NUMINAMATH_GPT_find_x_values_l691_69113


namespace NUMINAMATH_GPT_cupcakes_gluten_nut_nonvegan_l691_69174

-- Definitions based on conditions
def total_cupcakes := 120
def gluten_free_cupcakes := total_cupcakes / 3
def vegan_cupcakes := total_cupcakes / 4
def nut_free_cupcakes := total_cupcakes / 5
def gluten_and_vegan_cupcakes := 15
def vegan_and_nut_free_cupcakes := 10

-- Defining the theorem to prove the main question
theorem cupcakes_gluten_nut_nonvegan : 
  total_cupcakes - ((gluten_free_cupcakes + (vegan_cupcakes - gluten_and_vegan_cupcakes)) - vegan_and_nut_free_cupcakes) = 65 :=
by sorry

end NUMINAMATH_GPT_cupcakes_gluten_nut_nonvegan_l691_69174


namespace NUMINAMATH_GPT_total_clothes_donated_l691_69142

theorem total_clothes_donated
  (pants : ℕ) (jumpers : ℕ) (pajama_sets : ℕ) (tshirts : ℕ)
  (friends : ℕ)
  (adam_donation : ℕ)
  (half_adam_donated : ℕ)
  (friends_donation : ℕ)
  (total_donation : ℕ)
  (h1 : pants = 4) 
  (h2 : jumpers = 4) 
  (h3 : pajama_sets = 4 * 2) 
  (h4 : tshirts = 20) 
  (h5 : friends = 3)
  (h6 : adam_donation = pants + jumpers + pajama_sets + tshirts) 
  (h7 : half_adam_donated = adam_donation / 2) 
  (h8 : friends_donation = friends * adam_donation) 
  (h9 : total_donation = friends_donation + half_adam_donated) :
  total_donation = 126 :=
by
  sorry

end NUMINAMATH_GPT_total_clothes_donated_l691_69142


namespace NUMINAMATH_GPT_true_proposition_l691_69141

open Real

-- Proposition p
def p : Prop := ∀ x > 0, log x + 4 * x ≥ 3

-- Proposition q
def q : Prop := ∃ x > 0, 8 * x + 1 / (2 * x) ≤ 4

theorem true_proposition : ¬ p ∧ q := by
  sorry

end NUMINAMATH_GPT_true_proposition_l691_69141


namespace NUMINAMATH_GPT_part1_part2_l691_69100

variable (a : ℝ)
variable (x y : ℝ)
variable (P Q : ℝ × ℝ)

-- Part (1)
theorem part1 (hP : P = (2 * a - 2, a + 5)) (h_y : y = 0) : P = (-12, 0) :=
sorry

-- Part (2)
theorem part2 (hP : P = (2 * a - 2, a + 5)) (hQ : Q = (4, 5)) 
    (h_parallel : 2 * a - 2 = 4) : P = (4, 8) ∧ quadrant = "first" :=
sorry

end NUMINAMATH_GPT_part1_part2_l691_69100


namespace NUMINAMATH_GPT_steve_ate_bags_l691_69159

-- Given conditions
def total_macaroons : Nat := 12
def weight_per_macaroon : Nat := 5
def num_bags : Nat := 4
def total_weight_remaining : Nat := 45

-- Derived conditions
def total_weight_macaroons : Nat := total_macaroons * weight_per_macaroon
def macaroons_per_bag : Nat := total_macaroons / num_bags
def weight_per_bag : Nat := macaroons_per_bag * weight_per_macaroon
def bags_remaining : Nat := total_weight_remaining / weight_per_bag

-- Proof statement
theorem steve_ate_bags : num_bags - bags_remaining = 1 := by
  sorry

end NUMINAMATH_GPT_steve_ate_bags_l691_69159


namespace NUMINAMATH_GPT_part1_part2_l691_69176

-- Define set A and set B for m = 3
def setA : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def setB_m3 : Set ℝ := {x | x^2 - 2 * x - 3 < 0}

-- Define the complement of B in ℝ and the intersection of complements
def complB_m3 : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
def intersection_complB_A : Set ℝ := complB_m3 ∩ setA

-- Verify that the intersection of the complement of B and A equals the given set
theorem part1 : intersection_complB_A = {x | 3 ≤ x ∧ x ≤ 5} :=
by
  sorry

-- Define set A and the intersection of A and B
def setA' : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def setAB : Set ℝ := {x | -1 < x ∧ x < 4}

-- Given A ∩ B = {x | -1 < x < 4}, determine m such that B = {x | -1 < x < 4}
theorem part2 : ∃ m : ℝ, (setA' ∩ {x | x^2 - 2 * x - m < 0} = setAB) ∧ m = 8 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l691_69176


namespace NUMINAMATH_GPT_problem1_problem2_l691_69155

-- Problem 1
theorem problem1 : 
  (-2.8) - (-3.6) + (-1.5) - (3.6) = -4.3 := 
by 
  sorry

-- Problem 2
theorem problem2 :
  (- (5 / 6 : ℚ) + (1 / 3 : ℚ) - (3 / 4 : ℚ)) * (-24) = 30 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l691_69155


namespace NUMINAMATH_GPT_part_1_prob_excellent_part_2_rounds_pvalues_l691_69167

-- Definition of the probability of an excellent pair
def prob_excellent (p1 p2 : ℚ) : ℚ :=
  2 * p1 * (1 - p1) * p2 * p2 + p1 * p1 * 2 * p2 * (1 - p2) + p1 * p1 * p2 * p2

-- Part (1) statement: Prove the probability that they achieve "excellent pair" status in the first round
theorem part_1_prob_excellent (p1 p2 : ℚ) (hp1 : p1 = 3/4) (hp2 : p2 = 2/3) :
  prob_excellent p1 p2 = 2/3 := by
  rw [hp1, hp2]
  sorry

-- Part (2) statement: Prove the minimum number of rounds and values of p1 and p2
theorem part_2_rounds_pvalues (n : ℕ) (p1 p2 : ℚ) (h_sum : p1 + p2 = 4/3)
  (h_goal : n * prob_excellent p1 p2 ≥ 16) :
  (n = 27) ∧ (p1 = 2/3) ∧ (p2 = 2/3) := by
  sorry

end NUMINAMATH_GPT_part_1_prob_excellent_part_2_rounds_pvalues_l691_69167


namespace NUMINAMATH_GPT_inradius_of_triangle_l691_69170

theorem inradius_of_triangle (p A r : ℝ) (h1 : p = 20) (h2 : A = 25) : r = 2.5 :=
sorry

end NUMINAMATH_GPT_inradius_of_triangle_l691_69170


namespace NUMINAMATH_GPT_positive_integer_condition_l691_69129

theorem positive_integer_condition (n : ℕ) (h : 15 * n = n^2 + 56) : n = 8 :=
sorry

end NUMINAMATH_GPT_positive_integer_condition_l691_69129


namespace NUMINAMATH_GPT_express_set_l691_69160

open Set

/-- Define the set of natural numbers for which an expression is also a natural number. -/
theorem express_set : {x : ℕ | ∃ y : ℕ, 6 = y * (5 - x)} = {2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_express_set_l691_69160


namespace NUMINAMATH_GPT_fourth_power_square_prime_l691_69136

noncomputable def fourth_smallest_prime := 7

theorem fourth_power_square_prime :
  (fourth_smallest_prime ^ 2) ^ 4 = 5764801 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_fourth_power_square_prime_l691_69136


namespace NUMINAMATH_GPT_value_b15_l691_69114

def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d
def geometric_sequence (b : ℕ → ℤ) := ∃ q : ℤ, ∀ n : ℕ, b (n+1) = q * b n

theorem value_b15 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n : ℕ, S n = (n * (a 0 + a (n-1)) / 2))
  (h3 : S 9 = -18)
  (h4 : S 13 = -52)
  (h5 : geometric_sequence b)
  (h6 : b 5 = a 5)
  (h7 : b 7 = a 7) : 
  b 15 = -64 :=
sorry

end NUMINAMATH_GPT_value_b15_l691_69114


namespace NUMINAMATH_GPT_total_bike_clamps_given_away_l691_69117

-- Definitions for conditions
def bike_clamps_per_bike := 2
def bikes_sold_morning := 19
def bikes_sold_afternoon := 27

-- Theorem statement to be proven
theorem total_bike_clamps_given_away :
  bike_clamps_per_bike * bikes_sold_morning +
  bike_clamps_per_bike * bikes_sold_afternoon = 92 :=
by
  sorry -- Proof is to be filled in later

end NUMINAMATH_GPT_total_bike_clamps_given_away_l691_69117


namespace NUMINAMATH_GPT_find_divisor_l691_69188

theorem find_divisor : exists d : ℕ, 
  (∀ x : ℕ, x ≥ 10 ∧ x ≤ 1000000 → x % d = 0) ∧ 
  (10 + 999990 * d/111110 = 1000000) ∧
  d = 9 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l691_69188


namespace NUMINAMATH_GPT_domain_of_f_l691_69189

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_f :
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ -3) ↔ ((x < -3) ∨ (-3 < x ∧ x < 3) ∨ (x > 3)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l691_69189


namespace NUMINAMATH_GPT_total_birds_in_marsh_l691_69149

def number_of_geese : Nat := 58
def number_of_ducks : Nat := 37

theorem total_birds_in_marsh :
  number_of_geese + number_of_ducks = 95 :=
sorry

end NUMINAMATH_GPT_total_birds_in_marsh_l691_69149


namespace NUMINAMATH_GPT_dolls_total_l691_69193

theorem dolls_total (V S A : ℕ) 
  (hV : V = 20) 
  (hS : S = 2 * V)
  (hA : A = 2 * S) 
  : A + S + V = 140 := 
by 
  sorry

end NUMINAMATH_GPT_dolls_total_l691_69193


namespace NUMINAMATH_GPT_geometric_progression_common_ratio_l691_69184

/--
If \( a_1, a_2, a_3 \) are terms of an arithmetic progression with common difference \( d \neq 0 \),
and the products \( a_1 a_2, a_2 a_3, a_3 a_1 \) form a geometric progression,
then the common ratio of this geometric progression is \(-2\).
-/
theorem geometric_progression_common_ratio (a₁ a₂ a₃ d : ℝ) (h₀ : d ≠ 0) (h₁ : a₂ = a₁ + d)
  (h₂ : a₃ = a₁ + 2 * d) (h₃ : (a₂ * a₃) / (a₁ * a₂) = (a₃ * a₁) / (a₂ * a₃)) :
  (a₂ * a₃) / (a₁ * a₂) = -2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_common_ratio_l691_69184


namespace NUMINAMATH_GPT_count_integers_P_leq_0_l691_69195

def P(x : ℤ) : ℤ := 
  (x - 1^3) * (x - 2^3) * (x - 3^3) * (x - 4^3) * (x - 5^3) *
  (x - 6^3) * (x - 7^3) * (x - 8^3) * (x - 9^3) * (x - 10^3) *
  (x - 11^3) * (x - 12^3) * (x - 13^3) * (x - 14^3) * (x - 15^3) *
  (x - 16^3) * (x - 17^3) * (x - 18^3) * (x - 19^3) * (x - 20^3) *
  (x - 21^3) * (x - 22^3) * (x - 23^3) * (x - 24^3) * (x - 25^3) *
  (x - 26^3) * (x - 27^3) * (x - 28^3) * (x - 29^3) * (x - 30^3) *
  (x - 31^3) * (x - 32^3) * (x - 33^3) * (x - 34^3) * (x - 35^3) *
  (x - 36^3) * (x - 37^3) * (x - 38^3) * (x - 39^3) * (x - 40^3) *
  (x - 41^3) * (x - 42^3) * (x - 43^3) * (x - 44^3) * (x - 45^3) *
  (x - 46^3) * (x - 47^3) * (x - 48^3) * (x - 49^3) * (x - 50^3)

theorem count_integers_P_leq_0 : 
  ∃ n : ℕ, n = 15650 ∧ ∀ k : ℤ, (P k ≤ 0) → (n = 15650) :=
by sorry

end NUMINAMATH_GPT_count_integers_P_leq_0_l691_69195


namespace NUMINAMATH_GPT_sequence_formula_l691_69137

theorem sequence_formula (a : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a n ≠ 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (n + 1 + 1 / (a n))) :
  ∀ n : ℕ, n > 0 → a n = 2 / ((n : ℝ) ^ 2 - n + 2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l691_69137


namespace NUMINAMATH_GPT_value_of_1_plus_i_cubed_l691_69140

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- The main statement to verify
theorem value_of_1_plus_i_cubed : (1 + i ^ 3) = (1 - i) :=
by {  
  -- Use given conditions here if needed
  sorry
}

end NUMINAMATH_GPT_value_of_1_plus_i_cubed_l691_69140


namespace NUMINAMATH_GPT_diamonds_G20_l691_69151

def diamonds_in_figure (n : ℕ) : ℕ :=
if n = 1 then 1 else 4 * n^2 + 4 * n - 7

theorem diamonds_G20 : diamonds_in_figure 20 = 1673 :=
by sorry

end NUMINAMATH_GPT_diamonds_G20_l691_69151


namespace NUMINAMATH_GPT_problem_I_problem_II_l691_69190

-- Problem (I)
theorem problem_I (a b : ℝ) (h1 : a = 1) (h2 : b = 1) :
  { x : ℝ | |2*x + a| + |2*x - 2*b| + 3 > 8 } = 
  { x : ℝ | x < -1 ∨ x > 1.5 } := by
  sorry

-- Problem (II)
theorem problem_II (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ∀ x : ℝ, |2*x + a| + |2*x - 2*b| + 3 ≥ 5) :
  (1 / a + 1 / b) = (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l691_69190


namespace NUMINAMATH_GPT_solve_for_x_l691_69173

theorem solve_for_x (x : ℝ) (h_pos : 0 < x) (h : (x / 100) * (x ^ 2) = 9) : x = 10 * (3 ^ (1 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l691_69173


namespace NUMINAMATH_GPT_total_flowers_eaten_l691_69122

theorem total_flowers_eaten (bugs : ℕ) (flowers_per_bug : ℕ) (h_bugs : bugs = 3) (h_flowers_per_bug : flowers_per_bug = 2) :
  (bugs * flowers_per_bug) = 6 :=
by
  sorry

end NUMINAMATH_GPT_total_flowers_eaten_l691_69122


namespace NUMINAMATH_GPT_exists_zero_in_interval_l691_69164

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem exists_zero_in_interval : 
  (f 2) * (f 3) < 0 := by
  sorry

end NUMINAMATH_GPT_exists_zero_in_interval_l691_69164


namespace NUMINAMATH_GPT_min_additional_matchsticks_needed_l691_69191

-- Define the number of matchsticks in a 3x7 grid
def matchsticks_in_3x7_grid : Nat := 4 * 7 + 3 * 8

-- Define the number of matchsticks in a 5x5 grid
def matchsticks_in_5x5_grid : Nat := 6 * 5 + 6 * 5

-- Define the minimum number of additional matchsticks required
def additional_matchsticks (matchsticks_in_3x7_grid matchsticks_in_5x5_grid : Nat) : Nat :=
  matchsticks_in_5x5_grid - matchsticks_in_3x7_grid

theorem min_additional_matchsticks_needed :
  additional_matchsticks matchsticks_in_3x7_grid matchsticks_in_5x5_grid = 8 :=
by 
  unfold additional_matchsticks matchsticks_in_3x7_grid matchsticks_in_5x5_grid
  sorry

end NUMINAMATH_GPT_min_additional_matchsticks_needed_l691_69191


namespace NUMINAMATH_GPT_sacks_per_day_l691_69132

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (h1 : total_sacks = 56) (h2 : days = 4) : total_sacks / days = 14 := by
  sorry

end NUMINAMATH_GPT_sacks_per_day_l691_69132


namespace NUMINAMATH_GPT_janine_test_score_l691_69123

theorem janine_test_score :
  let num_mc := 10
  let p_mc := 0.80
  let num_sa := 30
  let p_sa := 0.70
  let total_questions := 40
  let correct_mc := p_mc * num_mc
  let correct_sa := p_sa * num_sa
  let total_correct := correct_mc + correct_sa
  (total_correct / total_questions) * 100 = 72.5 := 
by
  sorry

end NUMINAMATH_GPT_janine_test_score_l691_69123


namespace NUMINAMATH_GPT_cost_price_bicycle_A_l691_69148

variable {CP_A CP_B SP_C : ℝ}

theorem cost_price_bicycle_A (h1 : CP_B = 1.25 * CP_A) (h2 : SP_C = 1.25 * CP_B) (h3 : SP_C = 225) :
  CP_A = 144 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_bicycle_A_l691_69148


namespace NUMINAMATH_GPT_rectangle_area_with_circles_touching_l691_69194

theorem rectangle_area_with_circles_touching
  (r : ℝ)
  (radius_pos : r = 3)
  (short_side : ℝ)
  (long_side : ℝ)
  (dim_rect : short_side = 2 * r ∧ long_side = 4 * r) :
  short_side * long_side = 72 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_with_circles_touching_l691_69194


namespace NUMINAMATH_GPT_abs_difference_extrema_l691_69104

theorem abs_difference_extrema (x : ℝ) (h : 2 ≤ x ∧ x < 3) :
  max (|x-2| + |x-3| - |x-1|) = 0 ∧ min (|x-2| + |x-3| - |x-1|) = -1 :=
by
  sorry

end NUMINAMATH_GPT_abs_difference_extrema_l691_69104


namespace NUMINAMATH_GPT_arc_length_EF_l691_69134

-- Definitions based on the conditions
def angle_DEF_degrees : ℝ := 45
def circumference_D : ℝ := 80
def total_circle_degrees : ℝ := 360

-- Theorems/lemmata needed to prove the required statement
theorem arc_length_EF :
  let proportion := angle_DEF_degrees / total_circle_degrees
  let arc_length := proportion * circumference_D
  arc_length = 10 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_arc_length_EF_l691_69134


namespace NUMINAMATH_GPT_bricks_needed_for_room_floor_l691_69165

-- Conditions
def length : ℕ := 4
def breadth : ℕ := 5
def bricks_per_square_meter : ℕ := 17

-- Question and Answer (Proof Problem)
theorem bricks_needed_for_room_floor : 
  (length * breadth) * bricks_per_square_meter = 340 := by
  sorry

end NUMINAMATH_GPT_bricks_needed_for_room_floor_l691_69165


namespace NUMINAMATH_GPT_company_spends_less_l691_69196

noncomputable def total_spending_reduction_in_dollars : ℝ :=
  let magazine_initial_cost := 840.00
  let online_resources_initial_cost_gbp := 960.00
  let exchange_rate := 1.40
  let mag_cut_percentage := 0.30
  let online_cut_percentage := 0.20

  let magazine_cost_cut := magazine_initial_cost * mag_cut_percentage
  let online_resource_cost_cut_gbp := online_resources_initial_cost_gbp * online_cut_percentage
  
  let new_magazine_cost := magazine_initial_cost - magazine_cost_cut
  let new_online_resource_cost_gbp := online_resources_initial_cost_gbp - online_resource_cost_cut_gbp

  let online_resources_initial_cost := online_resources_initial_cost_gbp * exchange_rate
  let new_online_resource_cost := new_online_resource_cost_gbp * exchange_rate

  let mag_cut_amount := magazine_initial_cost - new_magazine_cost
  let online_cut_amount := online_resources_initial_cost - new_online_resource_cost
  
  mag_cut_amount + online_cut_amount

theorem company_spends_less : total_spending_reduction_in_dollars = 520.80 :=
by
  sorry

end NUMINAMATH_GPT_company_spends_less_l691_69196


namespace NUMINAMATH_GPT_book_total_pages_eq_90_l691_69106

theorem book_total_pages_eq_90 {P : ℕ} (h1 : (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 30) : P = 90 :=
sorry

end NUMINAMATH_GPT_book_total_pages_eq_90_l691_69106


namespace NUMINAMATH_GPT_necessity_of_A_for_B_l691_69199

variables {a b h : ℝ}

def PropA (a b h : ℝ) : Prop := |a - b| < 2 * h
def PropB (a b h : ℝ) : Prop := |a - 1| < h ∧ |b - 1| < h

theorem necessity_of_A_for_B (h_pos : 0 < h) : 
  (∀ a b, PropB a b h → PropA a b h) ∧ ¬ (∀ a b, PropA a b h → PropB a b h) :=
by sorry

end NUMINAMATH_GPT_necessity_of_A_for_B_l691_69199


namespace NUMINAMATH_GPT_flowers_per_bouquet_l691_69133

theorem flowers_per_bouquet (narcissus chrysanthemums bouquets : ℕ) 
  (h1: narcissus = 75) 
  (h2: chrysanthemums = 90) 
  (h3: bouquets = 33) 
  : (narcissus + chrysanthemums) / bouquets = 5 := 
by 
  sorry

end NUMINAMATH_GPT_flowers_per_bouquet_l691_69133


namespace NUMINAMATH_GPT_snowfall_rate_in_Hamilton_l691_69185

theorem snowfall_rate_in_Hamilton 
  (initial_depth_Kingston : ℝ := 12.1)
  (rate_Kingston : ℝ := 2.6)
  (initial_depth_Hamilton : ℝ := 18.6)
  (duration : ℕ := 13)
  (final_depth_equal : initial_depth_Kingston + rate_Kingston * duration = initial_depth_Hamilton + duration * x)
  (x : ℝ) :
  x = 2.1 :=
sorry

end NUMINAMATH_GPT_snowfall_rate_in_Hamilton_l691_69185


namespace NUMINAMATH_GPT_worm_in_apple_l691_69157

theorem worm_in_apple (radius : ℝ) (travel_distance : ℝ) (h_radius : radius = 31) (h_travel_distance : travel_distance = 61) :
  ∃ S : Set ℝ, ∀ point_on_path : ℝ, (point_on_path ∈ S) → false :=
by
  sorry

end NUMINAMATH_GPT_worm_in_apple_l691_69157


namespace NUMINAMATH_GPT_problem_solution_l691_69110

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define the set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the set N using the given condition
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- Define the complement of N in U
def complement_N : Set ℝ := U \ N

-- Define the intersection of M and the complement of N
def result_set : Set ℝ := M ∩ complement_N

-- Prove the desired result
theorem problem_solution : result_set = {x | -2 ≤ x ∧ x < 0} :=
sorry

end NUMINAMATH_GPT_problem_solution_l691_69110


namespace NUMINAMATH_GPT_number_of_cows_l691_69162

theorem number_of_cows (C H : ℕ) (hcnd : 4 * C + 2 * H = 2 * (C + H) + 18) :
  C = 9 :=
sorry

end NUMINAMATH_GPT_number_of_cows_l691_69162


namespace NUMINAMATH_GPT_min_weights_needed_l691_69183

theorem min_weights_needed :
  ∃ (weights : List ℕ), (∀ m : ℕ, 1 ≤ m ∧ m ≤ 100 → ∃ (left right : List ℕ), m = (left.sum - right.sum)) ∧ weights.length = 5 :=
sorry

end NUMINAMATH_GPT_min_weights_needed_l691_69183


namespace NUMINAMATH_GPT_problem_statement_l691_69180

structure Pricing :=
  (price_per_unit_1 : ℕ) (threshold_1 : ℕ)
  (price_per_unit_2 : ℕ) (threshold_2 : ℕ)
  (price_per_unit_3 : ℕ)

def cost (units : ℕ) (pricing : Pricing) : ℕ :=
  let t1 := pricing.threshold_1
  let t2 := pricing.threshold_2
  let p1 := pricing.price_per_unit_1
  let p2 := pricing.price_per_unit_2
  let p3 := pricing.price_per_unit_3
  if units ≤ t1 then units * p1
  else if units ≤ t2 then t1 * p1 + (units - t1) * p2
  else t1 * p1 + (t2 - t1) * p2 + (units - t2) * p3 

def units_given_cost (c : ℕ) (pricing : Pricing) : ℕ :=
  let t1 := pricing.threshold_1
  let t2 := pricing.threshold_2
  let p1 := pricing.price_per_unit_1
  let p2 := pricing.price_per_unit_2
  let p3 := pricing.price_per_unit_3
  if c ≤ t1 * p1 then c / p1
  else if c ≤ t1 * p1 + (t2 - t1) * p2 then t1 + (c - t1 * p1) / p2
  else t2 + (c - t1 * p1 - (t2 - t1) * p2) / p3

def double_eleven_case (total_units total_cost : ℕ) (x_units : ℕ) (pricing : Pricing) : ℕ :=
  let y_units := total_units - x_units
  let case1_cost := cost x_units pricing + cost y_units pricing
  if case1_cost = total_cost then (x_units, y_units).fst
  else sorry

theorem problem_statement (pricing : Pricing):
  (cost 120 pricing = 420) ∧ 
  (cost 260 pricing = 868) ∧
  (units_given_cost 740 pricing = 220) ∧
  (double_eleven_case 400 1349 290 pricing = 290)
  := sorry

end NUMINAMATH_GPT_problem_statement_l691_69180


namespace NUMINAMATH_GPT_parabola_x_intercepts_count_l691_69169

theorem parabola_x_intercepts_count :
  let a := -3
  let b := 4
  let c := -1
  let discriminant := b ^ 2 - 4 * a * c
  discriminant ≥ 0 →
  let num_roots := if discriminant > 0 then 2 else if discriminant = 0 then 1 else 0
  num_roots = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_parabola_x_intercepts_count_l691_69169


namespace NUMINAMATH_GPT_smallest_b_for_factors_l691_69124

theorem smallest_b_for_factors (b : ℕ) (h : ∃ r s : ℤ, (x : ℤ) → (x + r) * (x + s) = x^2 + ↑b * x + 2016 ∧ r * s = 2016) :
  b = 90 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_for_factors_l691_69124


namespace NUMINAMATH_GPT_arithmetic_sequence_length_l691_69107

theorem arithmetic_sequence_length 
  (a₁ : ℕ) (d : ℤ) (x : ℤ) (n : ℕ) 
  (h_start : a₁ = 20)
  (h_diff : d = -2)
  (h_eq : x = 10)
  (h_term : x = a₁ + (n - 1) * d) :
  n = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_length_l691_69107


namespace NUMINAMATH_GPT_honor_students_count_l691_69175

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end NUMINAMATH_GPT_honor_students_count_l691_69175


namespace NUMINAMATH_GPT_max_students_l691_69187

open BigOperators

def seats_in_row (i : ℕ) : ℕ := 8 + 2 * i

def max_students_in_row (i : ℕ) : ℕ := 4 + i

def total_max_students : ℕ := ∑ i in Finset.range 15, max_students_in_row (i + 1)

theorem max_students (condition1 : true) : total_max_students = 180 :=
by
  sorry

end NUMINAMATH_GPT_max_students_l691_69187


namespace NUMINAMATH_GPT_rearrange_expression_l691_69128

theorem rearrange_expression :
  1 - 2 - 3 - 4 - (5 - 6 - 7) = 0 :=
by
  sorry

end NUMINAMATH_GPT_rearrange_expression_l691_69128


namespace NUMINAMATH_GPT_xsq_plus_ysq_l691_69172

theorem xsq_plus_ysq (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : x^2 + y^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_xsq_plus_ysq_l691_69172


namespace NUMINAMATH_GPT_profit_without_discount_l691_69177

theorem profit_without_discount (CP SP_with_discount SP_without_discount : ℝ) (h1 : CP = 100) (h2 : SP_with_discount = CP + 0.235 * CP) (h3 : SP_with_discount = 0.95 * SP_without_discount) : (SP_without_discount - CP) / CP * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_profit_without_discount_l691_69177


namespace NUMINAMATH_GPT_find_sum_of_relatively_prime_integers_l691_69168

theorem find_sum_of_relatively_prime_integers :
  ∃ (x y : ℕ), x * y + x + y = 119 ∧ x < 25 ∧ y < 25 ∧ Nat.gcd x y = 1 ∧ x + y = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_relatively_prime_integers_l691_69168


namespace NUMINAMATH_GPT_juniors_in_program_l691_69103

theorem juniors_in_program (J S x y : ℕ) (h1 : J + S = 40) 
                           (h2 : x = y) 
                           (h3 : J / 5 = x) 
                           (h4 : S / 10 = y) : J = 12 :=
by
  sorry

end NUMINAMATH_GPT_juniors_in_program_l691_69103


namespace NUMINAMATH_GPT_max_score_exam_l691_69118

theorem max_score_exam (Gibi_percent Jigi_percent Mike_percent Lizzy_percent : ℝ)
  (avg_score total_score M : ℝ) :
  Gibi_percent = 0.59 →
  Jigi_percent = 0.55 →
  Mike_percent = 0.99 →
  Lizzy_percent = 0.67 →
  avg_score = 490 →
  total_score = avg_score * 4 →
  total_score = (Gibi_percent + Jigi_percent + Mike_percent + Lizzy_percent) * M →
  M = 700 :=
by
  intros hGibi hJigi hMike hLizzy hAvg hTotalScore hEq
  sorry

end NUMINAMATH_GPT_max_score_exam_l691_69118


namespace NUMINAMATH_GPT_x_coordinate_second_point_l691_69131

theorem x_coordinate_second_point (m n : ℝ) 
(h₁ : m = 2 * n + 5)
(h₂ : m + 2 = 2 * (n + 1) + 5) : 
  (m + 2) = 2 * n + 7 :=
by sorry

end NUMINAMATH_GPT_x_coordinate_second_point_l691_69131


namespace NUMINAMATH_GPT_min_n_for_circuit_l691_69121

theorem min_n_for_circuit
  (n : ℕ) 
  (p_success_component : ℝ)
  (p_work_circuit : ℝ) 
  (h1 : p_success_component = 0.5)
  (h2 : p_work_circuit = 1 - p_success_component ^ n) 
  (h3 : p_work_circuit ≥ 0.95) :
  n ≥ 5 := 
sorry

end NUMINAMATH_GPT_min_n_for_circuit_l691_69121


namespace NUMINAMATH_GPT_percent_calculation_l691_69119

theorem percent_calculation (x : ℝ) : 
  (∃ y : ℝ, y / 100 * x = 0.3 * 0.7 * x) → ∃ y : ℝ, y = 21 :=
by
  sorry

end NUMINAMATH_GPT_percent_calculation_l691_69119


namespace NUMINAMATH_GPT_func_passes_through_1_2_l691_69197

-- Given conditions
variable (a : ℝ) (x : ℝ) (y : ℝ)
variable (h1 : 0 < a) (h2 : a ≠ 1)

-- Definition of the function
noncomputable def func (x : ℝ) : ℝ := a^(x-1) + 1

-- Proof statement
theorem func_passes_through_1_2 : func a 1 = 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_func_passes_through_1_2_l691_69197


namespace NUMINAMATH_GPT_slope_of_line_l691_69156

theorem slope_of_line (x y : ℝ) (h : 4 * y = 5 * x + 20) : y = (5/4) * x + 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_slope_of_line_l691_69156


namespace NUMINAMATH_GPT_range_of_m_l691_69108

def p (m : ℝ) : Prop := m > 3
def q (m : ℝ) : Prop := m > (1 / 4)

theorem range_of_m (m : ℝ) (h1 : ¬p m) (h2 : p m ∨ q m) : (1 / 4) < m ∧ m ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l691_69108


namespace NUMINAMATH_GPT_ratio_bananas_dates_l691_69105

theorem ratio_bananas_dates (s c b d a : ℕ)
  (h1 : s = 780)
  (h2 : c = 60)
  (h3 : b = 3 * c)
  (h4 : a = 2 * d)
  (h5 : s = a + b + c + d) :
  b / d = 1 :=
by sorry

end NUMINAMATH_GPT_ratio_bananas_dates_l691_69105


namespace NUMINAMATH_GPT_find_start_number_l691_69166

def count_even_not_divisible_by_3 (start end_ : ℕ) : ℕ :=
  (end_ / 2 + 1) - (end_ / 6 + 1) - (if start = 0 then start / 2 else start / 2 + 1 - (start - 1) / 6 - 1)

theorem find_start_number (start end_ : ℕ) (h1 : end_ = 170) (h2 : count_even_not_divisible_by_3 start end_ = 54) : start = 8 :=
by 
  rw [h1] at h2
  sorry

end NUMINAMATH_GPT_find_start_number_l691_69166


namespace NUMINAMATH_GPT_find_c_l691_69163

theorem find_c 
  (a b c : ℝ) 
  (h_vertex : ∀ x y, y = a * x^2 + b * x + c → 
    (∃ k l, l = b / (2 * a) ∧ k = a * l^2 + b * l + c ∧ k = 3 ∧ l = -2))
  (h_pass : ∀ x y, y = a * x^2 + b * x + c → 
    (x = 2 ∧ y = 7)) : c = 4 :=
by sorry

end NUMINAMATH_GPT_find_c_l691_69163


namespace NUMINAMATH_GPT_solve_inequality_l691_69178

open Set

theorem solve_inequality (x : ℝ) (h : -3 * x^2 + 5 * x + 4 < 0 ∧ x > 0) : x ∈ Ioo 0 1 := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l691_69178


namespace NUMINAMATH_GPT_find_A_l691_69126

theorem find_A :
  ∃ A B C D : ℕ, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
               A * B = 72 ∧ C * D = 72 ∧
               A + B = C - D ∧ A = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l691_69126


namespace NUMINAMATH_GPT_net_effect_on_sale_l691_69120

theorem net_effect_on_sale (P Q : ℝ) :
  let new_price := 0.65 * P
  let new_quantity := 1.8 * Q
  let original_revenue := P * Q
  let new_revenue := new_price * new_quantity
  new_revenue - original_revenue = 0.17 * original_revenue :=
by
  sorry

end NUMINAMATH_GPT_net_effect_on_sale_l691_69120


namespace NUMINAMATH_GPT_profit_percentage_l691_69153

theorem profit_percentage (cost_price marked_price : ℝ) (discount_rate : ℝ) 
  (h1 : cost_price = 66.5) (h2 : marked_price = 87.5) (h3 : discount_rate = 0.05) : 
  (100 * ((marked_price * (1 - discount_rate) - cost_price) / cost_price)) = 25 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l691_69153


namespace NUMINAMATH_GPT_ratio_areas_l691_69111

theorem ratio_areas (H : ℝ) (L : ℝ) (r : ℝ) (A_rectangle : ℝ) (A_circle : ℝ) :
  H = 45 ∧ (L / H = 4 / 3) ∧ r = H / 2 ∧ A_rectangle = L * H ∧ A_circle = π * r^2 →
  (A_rectangle / A_circle = 17 / π) :=
by
  sorry

end NUMINAMATH_GPT_ratio_areas_l691_69111


namespace NUMINAMATH_GPT_divisibility_by_n_l691_69146

variable (a b c : ℤ) (n : ℕ)

theorem divisibility_by_n
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2 * n + 1) :
  ∃ k : ℤ, a^3 + b^2 - a^2 - b^3 = k * ↑n := 
sorry

end NUMINAMATH_GPT_divisibility_by_n_l691_69146


namespace NUMINAMATH_GPT_rope_segment_length_l691_69147

theorem rope_segment_length (L : ℕ) (half_fold_times : ℕ) (dm_to_cm : ℕ → ℕ) 
  (hL : L = 8) (h_half_fold_times : half_fold_times = 2) (h_dm_to_cm : dm_to_cm 1 = 10)
  : dm_to_cm (L / 2 ^ half_fold_times) = 20 := 
by 
  sorry

end NUMINAMATH_GPT_rope_segment_length_l691_69147


namespace NUMINAMATH_GPT_num_ways_award_medals_l691_69152

-- There are 8 sprinters in total
def num_sprinters : ℕ := 8

-- Three of the sprinters are Americans
def num_americans : ℕ := 3

-- The number of non-American sprinters
def num_non_americans : ℕ := num_sprinters - num_americans

-- The question to prove: the number of ways the medals can be awarded if at most one American gets a medal
theorem num_ways_award_medals 
  (n : ℕ) (m : ℕ) (k : ℕ) (h1 : n = num_sprinters) (h2 : m = num_americans) 
  (h3 : k = num_non_americans) 
  (no_american : ℕ := k * (k - 1) * (k - 2)) 
  (one_american : ℕ := m * 3 * k * (k - 1)) 
  : no_american + one_american = 240 :=
sorry

end NUMINAMATH_GPT_num_ways_award_medals_l691_69152


namespace NUMINAMATH_GPT_boiling_temperature_l691_69143

-- Definitions according to conditions
def initial_temperature : ℕ := 41

def temperature_increase_per_minute : ℕ := 3

def pasta_cooking_time : ℕ := 12

def mixing_and_salad_time : ℕ := pasta_cooking_time / 3

def total_evening_time : ℕ := 73

-- Conditions and the problem statement in Lean
theorem boiling_temperature :
  initial_temperature + (total_evening_time - (pasta_cooking_time + mixing_and_salad_time)) * temperature_increase_per_minute = 212 :=
by
  -- Here would be the proof, skipped with sorry
  sorry

end NUMINAMATH_GPT_boiling_temperature_l691_69143


namespace NUMINAMATH_GPT_prove_f_neg_a_l691_69101

noncomputable def f (x : ℝ) : ℝ := x + 1/x - 1

theorem prove_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -4 :=
by
  sorry

end NUMINAMATH_GPT_prove_f_neg_a_l691_69101


namespace NUMINAMATH_GPT_sequence_general_term_l691_69179

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 0 else 2 * n - 4

def S (n : ℕ) : ℤ :=
  n ^ 2 - 3 * n + 2

theorem sequence_general_term (n : ℕ) : a n = 
  if n = 1 then S n 
  else S n - S (n - 1) := by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l691_69179
