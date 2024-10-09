import Mathlib

namespace given_tan_alpha_eq_3_then_expression_eq_8_7_l1041_104156

theorem given_tan_alpha_eq_3_then_expression_eq_8_7 (α : ℝ) (h : Real.tan α = 3) :
  (6 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 8 / 7 := 
by
  sorry

end given_tan_alpha_eq_3_then_expression_eq_8_7_l1041_104156


namespace smallest_positive_integer_l1041_104108

theorem smallest_positive_integer (n : ℕ) (hn : 0 < n) (h : 19 * n ≡ 1456 [MOD 11]) : n = 6 :=
by
  sorry

end smallest_positive_integer_l1041_104108


namespace arcsin_eq_pi_div_two_solve_l1041_104177

theorem arcsin_eq_pi_div_two_solve :
  ∀ (x : ℝ), (Real.arcsin x + Real.arcsin (3 * x) = Real.pi / 2) → x = Real.sqrt 10 / 10 :=
by
  intro x h
  sorry -- Proof is omitted as per instructions

end arcsin_eq_pi_div_two_solve_l1041_104177


namespace inequality_generalization_l1041_104119

theorem inequality_generalization (x : ℝ) (n : ℕ) (hn : n > 0) (hx : x > 0) 
  (h1 : x + 1 / x ≥ 2) (h2 : x + 4 / (x ^ 2) = (x / 2) + (x / 2) + 4 / (x ^ 2) ∧ (x / 2) + (x / 2) + 4 / (x ^ 2) ≥ 3) : 
  x + n^n / x^n ≥ n + 1 := 
sorry

end inequality_generalization_l1041_104119


namespace roots_real_and_equal_l1041_104161

theorem roots_real_and_equal :
  ∀ x : ℝ,
  (x^2 - 4 * x * Real.sqrt 5 + 20 = 0) →
  (Real.sqrt ((-4 * Real.sqrt 5)^2 - 4 * 1 * 20) = 0) →
  (∃ r : ℝ, x = r ∧ x = r) :=
by
  intro x h_eq h_discriminant
  sorry

end roots_real_and_equal_l1041_104161


namespace solve_x_eq_10000_l1041_104158

theorem solve_x_eq_10000 (x : ℝ) (h : 5 * x^(1/4 : ℝ) - 3 * (x / x^(3/4 : ℝ)) = 10 + x^(1/4 : ℝ)) : x = 10000 :=
by
  sorry

end solve_x_eq_10000_l1041_104158


namespace triangle_ABC_is_right_triangle_l1041_104103

-- Define the triangle and the given conditions
variable (a b c : ℝ)
variable (h1 : a + c = 2*b)
variable (h2 : c - a = 1/2*b)

-- State the problem
theorem triangle_ABC_is_right_triangle : c^2 = a^2 + b^2 :=
by
  sorry

end triangle_ABC_is_right_triangle_l1041_104103


namespace simplify_expression_l1041_104190

theorem simplify_expression : 4 * (15 / 7) * (21 / -45) = -4 :=
by 
    -- Lean's type system will verify the correctness of arithmetic simplifications.
    sorry

end simplify_expression_l1041_104190


namespace apple_tree_total_production_l1041_104121

noncomputable def first_season_production : ℕ := 200
noncomputable def second_season_production : ℕ := 
  first_season_production - (first_season_production * 20 / 100)
noncomputable def third_season_production : ℕ := 
  second_season_production * 2
noncomputable def total_production : ℕ := 
  first_season_production + second_season_production + third_season_production

theorem apple_tree_total_production :
  total_production = 680 := by
  sorry

end apple_tree_total_production_l1041_104121


namespace purchasing_plan_exists_l1041_104174

-- Define the structure for our purchasing plan
structure PurchasingPlan where
  n3 : ℕ
  n6 : ℕ
  n9 : ℕ
  n12 : ℕ
  n15 : ℕ
  n19 : ℕ
  n21 : ℕ
  n30 : ℕ

-- Define the length function to sum up the total length of the purchasing plan
def length (p : PurchasingPlan) : ℕ :=
  3 * p.n3 + 6 * p.n6 + 9 * p.n9 + 12 * p.n12 + 15 * p.n15 + 19 * p.n19 + 21 * p.n21 + 30 * p.n30

-- Define the purchasing options
def options : List ℕ := [3, 6, 9, 12, 15, 19, 21, 30]

-- Define the requirement
def requiredLength : ℕ := 50

-- State the theorem that there exists a purchasing plan that sums up to the required length
theorem purchasing_plan_exists : ∃ p : PurchasingPlan, length p = requiredLength :=
  sorry

end purchasing_plan_exists_l1041_104174


namespace total_cats_l1041_104143

-- Define the conditions as constants
def asleep_cats : ℕ := 92
def awake_cats : ℕ := 6

-- State the theorem that proves the total number of cats
theorem total_cats : asleep_cats + awake_cats = 98 := 
by
  -- Proof omitted
  sorry

end total_cats_l1041_104143


namespace sqrt_123400_l1041_104118

theorem sqrt_123400 (h1: Real.sqrt 12.34 = 3.512) : Real.sqrt 123400 = 351.2 :=
by 
  sorry

end sqrt_123400_l1041_104118


namespace book_cost_l1041_104125

theorem book_cost (p : ℝ) (h1 : 14 * p < 25) (h2 : 16 * p > 28) : 1.75 < p ∧ p < 1.7857 :=
by
  -- This is where the proof would go
  sorry

end book_cost_l1041_104125


namespace ratio_of_areas_l1041_104120

theorem ratio_of_areas (r : ℝ) (A_triangle : ℝ) (A_circle : ℝ) 
  (h1 : ∀ r, A_triangle = (3 * r^2) / 4)
  (h2 : ∀ r, A_circle = π * r^2) 
  : (A_triangle / A_circle) = 3 / (4 * π) :=
sorry

end ratio_of_areas_l1041_104120


namespace determine_colors_l1041_104130

-- Define the colors
inductive Color
| white
| red
| blue

open Color

-- Define the friends
inductive Friend
| Tamara 
| Valya
| Lida

open Friend

-- Define a function from Friend to their dress color and shoes color
def Dress : Friend → Color := sorry
def Shoes : Friend → Color := sorry

-- The problem conditions
axiom cond1 : Dress Tamara = Shoes Tamara
axiom cond2 : Shoes Valya = white
axiom cond3 : Dress Lida ≠ red
axiom cond4 : Shoes Lida ≠ red

-- The proof goal
theorem determine_colors :
  Dress Tamara = red ∧ Shoes Tamara = red ∧
  Dress Valya = blue ∧ Shoes Valya = white ∧
  Dress Lida = white ∧ Shoes Lida = blue :=
sorry

end determine_colors_l1041_104130


namespace initial_men_count_l1041_104113

theorem initial_men_count
  (M : ℕ)
  (h1 : ∀ T : ℕ, (M * 8 * 10 = T) → (5 * 16 * 12 = T)) :
  M = 12 :=
by
  sorry

end initial_men_count_l1041_104113


namespace gcd_lcm_product_l1041_104144

theorem gcd_lcm_product (a b : ℕ) (h₀ : a = 15) (h₁ : b = 45) :
  Nat.gcd a b * Nat.lcm a b = 675 :=
by
  sorry

end gcd_lcm_product_l1041_104144


namespace find_constant_A_l1041_104196

theorem find_constant_A :
  ∀ (x : ℝ)
  (A B C D : ℝ),
      (
        (1 : ℝ) / (x^4 - 20 * x^3 + 147 * x^2 - 490 * x + 588) = 
        (A / (x + 3)) + (B / (x - 4)) + (C / ((x - 4)^2)) + (D / (x - 7))
      ) →
      A = - (1 / 490) := 
by 
  intro x A B C D h
  sorry

end find_constant_A_l1041_104196


namespace work_completion_time_l1041_104154

theorem work_completion_time :
  let work_rate_A := 1 / 8
  let work_rate_B := 1 / 6
  let work_rate_C := 1 / 4.8
  (work_rate_A + work_rate_B + work_rate_C) = 1 / 2 :=
by
  sorry

end work_completion_time_l1041_104154


namespace count_valid_n_l1041_104166

theorem count_valid_n : ∃ (count : ℕ), count = 6 ∧ ∀ n : ℕ,
  0 < n ∧ n < 42 → (∃ m : ℕ, m > 0 ∧ n = 42 * m / (m + 1)) :=
by
  sorry

end count_valid_n_l1041_104166


namespace infinite_grid_rectangles_l1041_104100

theorem infinite_grid_rectangles (m : ℕ) (hm : m > 12) : 
  ∃ (x y : ℕ), x * y > m ∧ x * (y - 1) < m := 
  sorry

end infinite_grid_rectangles_l1041_104100


namespace tan_theta_eq_sqrt3_div_3_l1041_104129

theorem tan_theta_eq_sqrt3_div_3
  (θ : ℝ)
  (h : (Real.cos θ * Real.sqrt 3 + Real.sin θ) = 2) :
  Real.tan θ = Real.sqrt 3 / 3 := by
  sorry

end tan_theta_eq_sqrt3_div_3_l1041_104129


namespace total_weight_of_8_moles_of_BaCl2_l1041_104109

-- Define atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_Cl : ℝ := 35.45

-- Define the molecular weight of BaCl2
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

-- Define the number of moles
def moles : ℝ := 8

-- Define the total weight calculation
def total_weight : ℝ := molecular_weight_BaCl2 * moles

-- The theorem to prove
theorem total_weight_of_8_moles_of_BaCl2 : total_weight = 1665.84 :=
by sorry

end total_weight_of_8_moles_of_BaCl2_l1041_104109


namespace kenny_cost_per_book_l1041_104189

theorem kenny_cost_per_book (B : ℕ) :
  let lawn_charge := 15
  let mowed_lawns := 35
  let video_game_cost := 45
  let video_games := 5
  let total_earnings := lawn_charge * mowed_lawns
  let spent_on_video_games := video_game_cost * video_games
  let remaining_money := total_earnings - spent_on_video_games
  remaining_money / B = 300 / B :=
by
  sorry

end kenny_cost_per_book_l1041_104189


namespace probability_of_both_qualified_bottles_expected_number_of_days_with_unqualified_milk_l1041_104131

noncomputable def qualification_rate : ℝ := 0.8
def probability_both_qualified (rate : ℝ) : ℝ := rate * rate
def unqualified_rate (rate : ℝ) : ℝ := 1 - rate
def expected_days (n : ℕ) (p : ℝ) : ℝ := n * p

theorem probability_of_both_qualified_bottles : 
  probability_both_qualified qualification_rate = 0.64 :=
by sorry

theorem expected_number_of_days_with_unqualified_milk :
  expected_days 3 (unqualified_rate qualification_rate) = 1.08 :=
by sorry

end probability_of_both_qualified_bottles_expected_number_of_days_with_unqualified_milk_l1041_104131


namespace math_problem_l1041_104102

theorem math_problem :
  (-1 : ℝ)^(53) + 3^(2^3 + 5^2 - 7^2) = -1 + (1 / 3^(16)) :=
by
  sorry

end math_problem_l1041_104102


namespace water_level_in_cubic_tank_is_one_l1041_104137

def cubic_tank : Type := {s : ℝ // s > 0}

def water_volume (s : cubic_tank) : ℝ := 
  let ⟨side, _⟩ := s 
  side^3

def water_level (s : cubic_tank) (volume : ℝ) (fill_ratio : ℝ) : ℝ := 
  let ⟨side, _⟩ := s 
  fill_ratio * side

theorem water_level_in_cubic_tank_is_one
  (s : cubic_tank)
  (h1 : water_volume s = 64)
  (h2 : water_volume s / 4 = 16)
  (h3 : 0 < 0.25 ∧ 0.25 ≤ 1) :
  water_level s 16 0.25 = 1 :=
by 
  sorry

end water_level_in_cubic_tank_is_one_l1041_104137


namespace find_p_l1041_104150

-- Assume the parametric equations and conditions specified in the problem.
noncomputable def parabola_eqns (p t : ℝ) (M E F : ℝ × ℝ) :=
  ∃ m : ℝ,
    (M = (6, m)) ∧
    (E = (-p / 2, m)) ∧
    (F = (p / 2, 0)) ∧
    (m^2 = 6 * p) ∧
    (|E.1 - F.1|^2 + |E.2 - F.2|^2 = |F.1 - M.1|^2 + |F.2 - M.2|^2) ∧
    (|F.1 - M.1|^2 + |F.2 - M.2|^2 = (F.1 + p / 2)^2 + (F.2 - m)^2)

theorem find_p {p t : ℝ} {M E F : ℝ × ℝ} (h : parabola_eqns p t M E F) : p = 4 :=
by
  sorry

end find_p_l1041_104150


namespace slope_of_parallel_line_l1041_104146

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l1041_104146


namespace division_of_powers_l1041_104195

variable {a : ℝ}

theorem division_of_powers (ha : a ≠ 0) : a^5 / a^3 = a^2 :=
by sorry

end division_of_powers_l1041_104195


namespace problem_l1041_104188

open Set

def M : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def N : Set ℝ := { x | x < 0 }
def complement_N : Set ℝ := { x | x ≥ 0 }

theorem problem : M ∩ complement_N = { x | 0 ≤ x ∧ x < 3 } :=
by
  sorry

end problem_l1041_104188


namespace sum_square_divisors_positive_l1041_104168

theorem sum_square_divisors_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b > 0 := 
by 
  sorry

end sum_square_divisors_positive_l1041_104168


namespace highland_baseball_club_members_l1041_104171

-- Define the given costs and expenditures.
def socks_cost : ℕ := 6
def tshirt_cost : ℕ := socks_cost + 7
def cap_cost : ℕ := socks_cost
def total_expenditure : ℕ := 5112
def home_game_cost : ℕ := socks_cost + tshirt_cost
def away_game_cost : ℕ := socks_cost + tshirt_cost + cap_cost
def cost_per_member : ℕ := home_game_cost + away_game_cost

theorem highland_baseball_club_members :
  total_expenditure / cost_per_member = 116 :=
by
  sorry

end highland_baseball_club_members_l1041_104171


namespace fewest_reciprocal_keypresses_l1041_104111

theorem fewest_reciprocal_keypresses (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0) 
  (h1 : f 50 = 1 / 50) (h2 : f (1 / 50) = 50) : 
  ∃ n : ℕ, n = 2 ∧ (∀ m : ℕ, (m < n) → (f^[m] 50 ≠ 50)) :=
by
  sorry

end fewest_reciprocal_keypresses_l1041_104111


namespace find_number_l1041_104104

def divisor : ℕ := 22
def quotient : ℕ := 12
def remainder : ℕ := 1
def number : ℕ := (divisor * quotient) + remainder

theorem find_number : number = 265 := by
  sorry

end find_number_l1041_104104


namespace num_solutions_gcd_lcm_l1041_104198

noncomputable def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

theorem num_solutions_gcd_lcm (x y : ℕ) :
  (Nat.gcd x y = factorial 20) ∧ (Nat.lcm x y = factorial 30) →
  2^10 = 1024 :=
  by
  intro h
  sorry

end num_solutions_gcd_lcm_l1041_104198


namespace production_statistics_relation_l1041_104159

noncomputable def a : ℚ := (10 + 12 + 14 + 14 + 15 + 15 + 16 + 17 + 17 + 17) / 10
noncomputable def b : ℚ := (15 + 15) / 2
noncomputable def c : ℤ := 17

theorem production_statistics_relation : c > a ∧ a > b :=
by
  sorry

end production_statistics_relation_l1041_104159


namespace intersection_complement_l1041_104148

open Set

variable (U A B : Set ℕ)

-- Given conditions:
def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def set_B : Set ℕ := {2, 3}

theorem intersection_complement (U A B : Set ℕ) : 
  U = universal_set → A = set_A → B = set_B → (A ∩ (U \ B)) = {1, 5} := by
  sorry

end intersection_complement_l1041_104148


namespace expression_value_l1041_104160

theorem expression_value (a b c d : ℤ) (h_a : a = 15) (h_b : b = 19) (h_c : c = 3) (h_d : d = 2) :
  (a - (b - c)) - ((a - b) - c + d) = 4 := 
by
  rw [h_a, h_b, h_c, h_d]
  sorry

end expression_value_l1041_104160


namespace inequality_solution_l1041_104149

theorem inequality_solution (x : ℝ) : (1 - 3 * (x - 1) < x) ↔ (x > 1) :=
by sorry

end inequality_solution_l1041_104149


namespace prime_related_divisors_circle_l1041_104163

variables (n : ℕ)

-- Definitions of prime-related and conditions for n
def is_prime (p: ℕ): Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p
def prime_related (a b : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ (a = p * b ∨ b = p * a)

-- The main statement to be proven
theorem prime_related_divisors_circle (n : ℕ) : 
  (n ≥ 3) ∧ (∀ a b, a ≠ b → (a ∣ n ∧ b ∣ n) → prime_related a b) ↔ ¬ (
    ∃ (p : ℕ) (k : ℕ), is_prime p ∧ (n = p ^ k) ∨ 
    ∃ (m : ℕ), n = m ^ 2 ) :=
sorry

end prime_related_divisors_circle_l1041_104163


namespace symmetric_line_eq_l1041_104179

-- Define points A and B
def A (a : ℝ) : ℝ × ℝ := (a-1, a+1)
def B (a : ℝ) : ℝ × ℝ := (a, a)

-- We want to prove the equation of the line L about which points A and B are symmetric is "x - y + 1 = 0".
theorem symmetric_line_eq (a : ℝ) : 
  ∃ m b, (m = 1) ∧ (b = 1) ∧ (∀ x y, (y = m * x + b) ↔ (x - y + 1 = 0)) :=
sorry

end symmetric_line_eq_l1041_104179


namespace rectangular_prism_pairs_l1041_104123

def total_pairs_of_edges_in_rect_prism_different_dimensions (length width height : ℝ) 
  (h1 : length ≠ width) 
  (h2 : width ≠ height) 
  (h3 : height ≠ length) 
  : ℕ :=
66

theorem rectangular_prism_pairs (length width height : ℝ) 
  (h1 : length ≠ width) 
  (h2 : width ≠ height) 
  (h3 : height ≠ length) 
  : total_pairs_of_edges_in_rect_prism_different_dimensions length width height h1 h2 h3 = 66 := 
sorry

end rectangular_prism_pairs_l1041_104123


namespace find_values_of_a_l1041_104192

def P : Set ℝ := { x | x^2 + x - 6 = 0 }
def S (a : ℝ) : Set ℝ := { x | a * x + 1 = 0 }

theorem find_values_of_a (a : ℝ) : (S a ⊆ P) ↔ (a = 0 ∨ a = 1/3 ∨ a = -1/2) := by
  sorry

end find_values_of_a_l1041_104192


namespace cake_heavier_than_bread_l1041_104194

-- Definitions
def weight_of_7_cakes_eq_1950_grams (C : ℝ) := 7 * C = 1950
def weight_of_5_cakes_12_breads_eq_2750_grams (C B : ℝ) := 5 * C + 12 * B = 2750

-- Statement
theorem cake_heavier_than_bread (C B : ℝ)
  (h1 : weight_of_7_cakes_eq_1950_grams C)
  (h2 : weight_of_5_cakes_12_breads_eq_2750_grams C B) :
  C - B = 165.47 :=
by {
  sorry
}

end cake_heavier_than_bread_l1041_104194


namespace no_integer_solutions_l1041_104124

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 19 * x^3 - 17 * y^3 = 50 := 
by 
  sorry

end no_integer_solutions_l1041_104124


namespace percent_value_in_quarters_l1041_104122

theorem percent_value_in_quarters (num_dimes num_quarters : ℕ) 
  (value_dime value_quarter total_value value_in_quarters : ℕ) 
  (h1 : num_dimes = 75)
  (h2 : num_quarters = 30)
  (h3 : value_dime = num_dimes * 10)
  (h4 : value_quarter = num_quarters * 25)
  (h5 : total_value = value_dime + value_quarter)
  (h6 : value_in_quarters = num_quarters * 25) :
  (value_in_quarters / total_value) * 100 = 50 :=
by
  sorry

end percent_value_in_quarters_l1041_104122


namespace complete_square_transform_l1041_104133

theorem complete_square_transform (x : ℝ) : 
  x^2 - 2 * x = 9 ↔ (x - 1)^2 = 10 :=
by
  sorry

end complete_square_transform_l1041_104133


namespace factorization_identity_l1041_104132

theorem factorization_identity (a b : ℝ) : (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2 :=
by
  sorry

end factorization_identity_l1041_104132


namespace inverse_of_f_at_10_l1041_104176

noncomputable def f (x : ℝ) : ℝ := 1 + 3^(-x)

theorem inverse_of_f_at_10 :
  f⁻¹ 10 = -2 :=
sorry

end inverse_of_f_at_10_l1041_104176


namespace exists_x_given_y_l1041_104191

theorem exists_x_given_y (y : ℝ) : ∃ x : ℝ, x^2 + y^2 = 10 ∧ x^2 - x * y - 3 * y + 12 = 0 := 
sorry

end exists_x_given_y_l1041_104191


namespace trapezium_other_side_length_l1041_104181

theorem trapezium_other_side_length (x : ℝ) : 
  (1 / 2) * (20 + x) * 13 = 247 → x = 18 :=
by
  sorry

end trapezium_other_side_length_l1041_104181


namespace total_slices_at_picnic_l1041_104153

def danny_watermelons : ℕ := 3
def danny_slices_per_watermelon : ℕ := 10
def sister_watermelons : ℕ := 1
def sister_slices_per_watermelon : ℕ := 15

def total_danny_slices : ℕ := danny_watermelons * danny_slices_per_watermelon
def total_sister_slices : ℕ := sister_watermelons * sister_slices_per_watermelon
def total_slices : ℕ := total_danny_slices + total_sister_slices

theorem total_slices_at_picnic : total_slices = 45 :=
by
  sorry

end total_slices_at_picnic_l1041_104153


namespace integer_solutions_count_l1041_104173

theorem integer_solutions_count :
  (∃ (n : ℕ), ∀ (x y : ℤ), x^2 + y^2 = 6 * x + 2 * y + 15 → n = 12) :=
by
  sorry

end integer_solutions_count_l1041_104173


namespace no_perfect_square_abc_sum_l1041_104165

theorem no_perfect_square_abc_sum (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
  ¬ ∃ m : ℕ, m * m = (100 * a + 10 * b + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) :=
by
  sorry

end no_perfect_square_abc_sum_l1041_104165


namespace tan_double_angle_l1041_104128

theorem tan_double_angle (α : ℝ) (h : 3 * Real.cos α + Real.sin α = 0) : 
    Real.tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_double_angle_l1041_104128


namespace largest_divisor_36_l1041_104140

theorem largest_divisor_36 (n : ℕ) (h : n > 0) (h_div : 36 ∣ n^3) : 6 ∣ n := 
sorry

end largest_divisor_36_l1041_104140


namespace ab_value_l1041_104116

theorem ab_value (a b : ℝ) :
  (A = { x : ℝ | x^2 - 8 * x + 15 = 0 }) ∧
  (B = { x : ℝ | x^2 - a * x + b = 0 }) ∧
  (A ∪ B = {2, 3, 5}) ∧
  (A ∩ B = {3}) →
  (a * b = 30) :=
by
  sorry

end ab_value_l1041_104116


namespace total_amount_divided_l1041_104139

theorem total_amount_divided (A B C : ℝ) (h1 : A / B = 3 / 4) (h2 : B / C = 5 / 6) (h3 : A = 29491.525423728814) :
  A + B + C = 116000 := 
sorry

end total_amount_divided_l1041_104139


namespace solve_inequality_l1041_104182

theorem solve_inequality (a x : ℝ) :
  (a > 0 → (a - 1) / a < x ∧ x < 1) ∧ 
  (a = 0 → x < 1) ∧ 
  (a < 0 → x > (a - 1) / a ∨ x < 1) ↔ 
  (ax / (x - 1) < (a - 1) / (x - 1)) :=
sorry

end solve_inequality_l1041_104182


namespace max_xy_l1041_104157

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 3 * y = 6) : xy ≤ 3 / 2 := sorry

end max_xy_l1041_104157


namespace sqrt_expression_eq_l1041_104151

theorem sqrt_expression_eq : 
  (Real.sqrt 18 / Real.sqrt 6 - Real.sqrt 12 + Real.sqrt 48 * Real.sqrt (1/3)) = -Real.sqrt 3 + 4 := 
by
  sorry

end sqrt_expression_eq_l1041_104151


namespace find_value_of_S_l1041_104197

theorem find_value_of_S (S : ℝ)
  (h1 : (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 180) :
  S = 180 :=
sorry

end find_value_of_S_l1041_104197


namespace value_of_expression_l1041_104152

theorem value_of_expression (x y : ℚ) (hx : x = 2/3) (hy : y = 5/8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 := 
by
  sorry

end value_of_expression_l1041_104152


namespace abs_neg_two_eq_two_neg_two_pow_zero_eq_one_l1041_104106

theorem abs_neg_two_eq_two : |(-2)| = 2 :=
sorry

theorem neg_two_pow_zero_eq_one : (-2)^0 = 1 :=
sorry

end abs_neg_two_eq_two_neg_two_pow_zero_eq_one_l1041_104106


namespace percentage_enclosed_by_pentagons_l1041_104134

-- Define the condition for the large square and smaller squares.
def large_square_area (b : ℝ) : ℝ := (4 * b) ^ 2

-- Define the condition for the number of smaller squares forming pentagons.
def pentagon_small_squares : ℝ := 10

-- Define the total number of smaller squares within a large square.
def total_small_squares : ℝ := 16

-- Prove that the percentage of the plane enclosed by pentagons is 62.5%.
theorem percentage_enclosed_by_pentagons :
  (pentagon_small_squares / total_small_squares) * 100 = 62.5 :=
by 
  -- The proof is left as an exercise.
  sorry

end percentage_enclosed_by_pentagons_l1041_104134


namespace ticket_sales_total_l1041_104178

variable (price_adult : ℕ) (price_child : ℕ) (total_tickets : ℕ) (child_tickets : ℕ)

def total_money_collected (price_adult : ℕ) (price_child : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - child_tickets
  let total_child := child_tickets * price_child
  let total_adult := adult_tickets * price_adult
  total_child + total_adult

theorem ticket_sales_total :
  price_adult = 6 →
  price_child = 4 →
  total_tickets = 21 →
  child_tickets = 11 →
  total_money_collected price_adult price_child total_tickets child_tickets = 104 :=
by
  intros
  unfold total_money_collected
  simp
  sorry

end ticket_sales_total_l1041_104178


namespace problem_1_problem_2_l1041_104110

noncomputable def f (x p : ℝ) := p * x - p / x - 2 * Real.log x
noncomputable def g (x : ℝ) := 2 * Real.exp 1 / x

theorem problem_1 (p : ℝ) : 
  (∀ x : ℝ, 0 < x → p * x - p / x - 2 * Real.log x ≥ 0) ↔ p ≥ 1 := 
by sorry

theorem problem_2 (p : ℝ) : 
  (∃ x_0 : ℝ, 1 ≤ x_0 ∧ x_0 ≤ Real.exp 1 ∧ f x_0 p > g x_0) ↔ 
  p > 4 * Real.exp 1 / (Real.exp 2 - 1) :=
by sorry

end problem_1_problem_2_l1041_104110


namespace max_lambda_leq_64_div_27_l1041_104172

theorem max_lambda_leq_64_div_27 (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (1:ℝ) + (64:ℝ) / (27:ℝ) * (1 - a) * (1 - b) * (1 - c) ≤ Real.sqrt 3 / Real.sqrt (a + b + c) := 
sorry

end max_lambda_leq_64_div_27_l1041_104172


namespace find_principal_l1041_104155

theorem find_principal
  (R : ℝ) (hR : R = 0.05)
  (I : ℝ) (hI : I = 0.02)
  (A : ℝ) (hA : A = 1120)
  (n : ℕ) (hn : n = 6)
  (R' : ℝ) (hR' : R' = ((1 + R) / (1 + I)) - 1) :
  P = 938.14 :=
by
  have compound_interest_formula := A / (1 + R')^n
  sorry

end find_principal_l1041_104155


namespace tight_sequence_from_sum_of_terms_range_of_q_for_tight_sequences_l1041_104136

def tight_sequence (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → (1/2 : ℚ) ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → S n = (1 / 4) * (n^2 + 3 * n)

noncomputable def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
∀ n : ℕ, n > 0 → a n = a 1 * q ^ (n - 1)

theorem tight_sequence_from_sum_of_terms (S : ℕ → ℚ) (a : ℕ → ℚ) : 
  (∀ n : ℕ, n > 0 → S n = (1 / 4) * (n^2 + 3 * n)) →
  (∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) →
  tight_sequence a :=
sorry

theorem range_of_q_for_tight_sequences (a : ℕ → ℚ) (S : ℕ → ℚ) (q : ℚ) :
  geometric_sequence a q →
  tight_sequence a →
  tight_sequence S →
  (1 / 2 : ℚ) ≤ q ∧ q < 1 :=
sorry

end tight_sequence_from_sum_of_terms_range_of_q_for_tight_sequences_l1041_104136


namespace car_sales_decrease_l1041_104135

theorem car_sales_decrease (P N : ℝ) (h1 : 1.30 * P / (N * (1 - D / 100)) = 1.8571 * (P / N)) : D = 30 :=
by
  sorry

end car_sales_decrease_l1041_104135


namespace find_other_number_l1041_104169

-- Defining the two numbers and their properties
def sum_is_84 (a b : ℕ) : Prop := a + b = 84
def one_is_36 (a b : ℕ) : Prop := a = 36 ∨ b = 36
def other_is_48 (a b : ℕ) : Prop := a = 48 ∨ b = 48

-- The theorem statement
theorem find_other_number (a b : ℕ) (h1 : sum_is_84 a b) (h2 : one_is_36 a b) : other_is_48 a b :=
by {
  sorry
}

end find_other_number_l1041_104169


namespace cantaloupe_total_l1041_104185

theorem cantaloupe_total (Fred Tim Alicia : ℝ) 
  (hFred : Fred = 38.5) 
  (hTim : Tim = 44.2)
  (hAlicia : Alicia = 29.7) : 
  Fred + Tim + Alicia = 112.4 :=
by
  sorry

end cantaloupe_total_l1041_104185


namespace jimmy_bought_3_pens_l1041_104184

def cost_of_notebooks (num_notebooks : ℕ) (price_per_notebook : ℕ) : ℕ := num_notebooks * price_per_notebook
def cost_of_folders (num_folders : ℕ) (price_per_folder : ℕ) : ℕ := num_folders * price_per_folder
def total_cost (cost_notebooks cost_folders : ℕ) : ℕ := cost_notebooks + cost_folders
def total_spent (initial_money change : ℕ) : ℕ := initial_money - change
def cost_of_pens (total_spent amount_for_items : ℕ) : ℕ := total_spent - amount_for_items
def num_pens (cost_pens price_per_pen : ℕ) : ℕ := cost_pens / price_per_pen

theorem jimmy_bought_3_pens :
  let pen_price := 1
  let notebook_price := 3
  let num_notebooks := 4
  let folder_price := 5
  let num_folders := 2
  let initial_money := 50
  let change := 25
  let cost_notebooks := cost_of_notebooks num_notebooks notebook_price
  let cost_folders := cost_of_folders num_folders folder_price
  let total_items_cost := total_cost cost_notebooks cost_folders
  let amount_spent := total_spent initial_money change
  let pen_cost := cost_of_pens amount_spent total_items_cost
  num_pens pen_cost pen_price = 3 :=
by
  sorry

end jimmy_bought_3_pens_l1041_104184


namespace sports_club_membership_l1041_104199

theorem sports_club_membership :
  ∀ (total T B_and_T neither : ℕ),
    total = 30 → 
    T = 19 →
    B_and_T = 9 →
    neither = 2 →
  ∃ (B : ℕ), B = 18 :=
by
  intros total T B_and_T neither ht hT hBandT hNeither
  let B := total - neither - T + B_and_T
  use B
  sorry

end sports_club_membership_l1041_104199


namespace negation_of_P_l1041_104170

open Real

theorem negation_of_P :
  (¬ (∀ x : ℝ, x > sin x)) ↔ (∃ x : ℝ, x ≤ sin x) :=
by
  sorry

end negation_of_P_l1041_104170


namespace incenter_divides_angle_bisector_2_1_l1041_104164

def is_incenter_divide_angle_bisector (AB BC AC : ℝ) (O : ℝ) : Prop :=
  AB = 15 ∧ BC = 12 ∧ AC = 18 → O = 2 / 1

theorem incenter_divides_angle_bisector_2_1 :
  is_incenter_divide_angle_bisector 15 12 18 (2 / 1) :=
by
  sorry

end incenter_divides_angle_bisector_2_1_l1041_104164


namespace multiplication_factor_l1041_104126

theorem multiplication_factor
  (n : ℕ) (avg_orig avg_new : ℝ) (F : ℝ)
  (H1 : n = 7)
  (H2 : avg_orig = 24)
  (H3 : avg_new = 120)
  (H4 : (n * avg_new) = F * (n * avg_orig)) :
  F = 5 :=
by {
  sorry
}

end multiplication_factor_l1041_104126


namespace range_of_k_l1041_104183

/-- If the function y = (k + 1) * x is decreasing on the entire real line, then k < -1. -/
theorem range_of_k (k : ℝ) (h : ∀ x y : ℝ, x < y → (k + 1) * x > (k + 1) * y) : k < -1 :=
sorry

end range_of_k_l1041_104183


namespace xyz_inequality_l1041_104105

theorem xyz_inequality : ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by
  intros
  sorry

end xyz_inequality_l1041_104105


namespace projectile_time_l1041_104115

theorem projectile_time : ∃ t : ℝ, (60 - 8 * t - 5 * t^2 = 30) ∧ t = 1.773 := by
  sorry

end projectile_time_l1041_104115


namespace usual_time_to_office_l1041_104167

theorem usual_time_to_office (P : ℝ) (T : ℝ) (h1 : T = (3 / 4) * (T + 20)) : T = 60 :=
by
  sorry

end usual_time_to_office_l1041_104167


namespace solve_inequalities_l1041_104141

theorem solve_inequalities (x : ℝ) (h1 : |4 - x| < 5) (h2 : x^2 < 36) : (-1 < x) ∧ (x < 6) :=
by
  sorry

end solve_inequalities_l1041_104141


namespace water_balloon_packs_l1041_104127

theorem water_balloon_packs (P : ℕ) : 
  (6 * P + 12 = 30) → P = 3 := by
  sorry

end water_balloon_packs_l1041_104127


namespace ab_bc_ca_leq_zero_l1041_104180

theorem ab_bc_ca_leq_zero (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end ab_bc_ca_leq_zero_l1041_104180


namespace net_profit_from_plant_sales_l1041_104193

noncomputable def calculate_net_profit : ℝ :=
  let cost_basil := 2.00
  let cost_mint := 3.00
  let cost_zinnia := 7.00
  let cost_soil := 15.00
  let total_cost := cost_basil + cost_mint + cost_zinnia + cost_soil
  let basil_germinated := 20 * 0.80
  let mint_germinated := 15 * 0.75
  let zinnia_germinated := 10 * 0.70
  let revenue_healthy_basil := 12 * 5.00
  let revenue_small_basil := 8 * 3.00
  let revenue_healthy_mint := 10 * 6.00
  let revenue_small_mint := 4 * 4.00
  let revenue_healthy_zinnia := 5 * 10.00
  let revenue_small_zinnia := 2 * 7.00
  let total_revenue := revenue_healthy_basil + revenue_small_basil + revenue_healthy_mint + revenue_small_mint + revenue_healthy_zinnia + revenue_small_zinnia
  total_revenue - total_cost

theorem net_profit_from_plant_sales : calculate_net_profit = 197.00 := by
  sorry

end net_profit_from_plant_sales_l1041_104193


namespace each_child_apples_l1041_104117

-- Define the given conditions
def total_apples : ℕ := 450
def num_adults : ℕ := 40
def num_adults_apples : ℕ := 3
def num_children : ℕ := 33

-- Define the theorem to prove
theorem each_child_apples : 
  let total_apples_eaten_by_adults := num_adults * num_adults_apples
  let total_apples_for_children := total_apples - total_apples_eaten_by_adults
  let apples_per_child := total_apples_for_children / num_children
  apples_per_child = 10 :=
by
  sorry

end each_child_apples_l1041_104117


namespace numbers_sum_prod_l1041_104147

theorem numbers_sum_prod (x y : ℝ) (h_sum : x + y = 10) (h_prod : x * y = 24) : (x = 4 ∧ y = 6) ∨ (x = 6 ∧ y = 4) :=
by
  sorry

end numbers_sum_prod_l1041_104147


namespace pumps_work_hours_l1041_104138

theorem pumps_work_hours (d : ℕ) (h_d_pos : d > 0) : 6 * (8 / d) * d = 48 :=
by
  -- The proof is omitted
  sorry

end pumps_work_hours_l1041_104138


namespace kernel_count_in_final_bag_l1041_104114

namespace PopcornKernelProblem

def percentage_popped (popped total : ℕ) : ℤ := ((popped : ℤ) * 100) / (total : ℤ)

def first_bag_percentage := percentage_popped 60 75
def second_bag_percentage := percentage_popped 42 50
def final_bag_percentage (x : ℕ) : ℤ := percentage_popped 82 x

theorem kernel_count_in_final_bag :
  (first_bag_percentage + second_bag_percentage + final_bag_percentage 100) / 3 = 82 := 
sorry

end PopcornKernelProblem

end kernel_count_in_final_bag_l1041_104114


namespace inequality_solution_empty_l1041_104175

theorem inequality_solution_empty {a : ℝ} :
  (∀ x : ℝ, ¬ (|x+2| + |x-1| < a)) ↔ a ≤ 3 :=
by
  sorry

end inequality_solution_empty_l1041_104175


namespace speed_of_water_l1041_104112

variable (v : ℝ) -- the speed of the water in km/h
variable (t : ℝ) -- time taken to swim back in hours
variable (d : ℝ) -- distance swum against the current in km
variable (s : ℝ) -- speed in still water

theorem speed_of_water :
  ∀ (v t d s : ℝ),
  s = 20 -> t = 5 -> d = 40 -> d = (s - v) * t -> v = 12 :=
by
  intros v t d s ht hs hd heq
  sorry

end speed_of_water_l1041_104112


namespace val_total_money_l1041_104186

theorem val_total_money : 
  ∀ (nickels_initial dimes_initial nickels_found : ℕ),
    nickels_initial = 20 →
    dimes_initial = 3 * nickels_initial →
    nickels_found = 2 * nickels_initial →
    (nickels_initial * 5 + dimes_initial * 10 + nickels_found * 5) / 100 = 9 :=
by
  intros nickels_initial dimes_initial nickels_found h1 h2 h3
  sorry

end val_total_money_l1041_104186


namespace total_balloons_l1041_104187

theorem total_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h₁ : joan_balloons = 40) (h₂ : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := 
by
  sorry

end total_balloons_l1041_104187


namespace find_a1_l1041_104142

-- Define the arithmetic sequence and the given conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_mean (x y z : ℝ) : Prop :=
  y^2 = x * z

def problem_statement (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (arithmetic_sequence a d) ∧ (geometric_mean (a 1) (a 2) (a 4))

theorem find_a1 (a : ℕ → ℝ) (d : ℝ) (h : problem_statement a d) : a 1 = 1 := by
  have h_seq : arithmetic_sequence a d := h.1
  have h_geom : geometric_mean (a 1) (a 2) (a 4) := h.2
  sorry

end find_a1_l1041_104142


namespace physics_marks_l1041_104162

theorem physics_marks (P C M : ℕ) (h1 : P + C + M = 180) (h2 : P + M = 180) (h3 : P + C = 140) : P = 140 :=
by
  sorry

end physics_marks_l1041_104162


namespace total_art_cost_l1041_104107

-- Definitions based on the conditions
def total_price_first_3_pieces (price_per_piece : ℤ) : ℤ :=
  price_per_piece * 3

def price_increase (price_per_piece : ℤ) : ℤ :=
  price_per_piece / 2

def total_price_all_arts (price_per_piece next_piece_price : ℤ) : ℤ :=
  (total_price_first_3_pieces price_per_piece) + next_piece_price

-- The proof problem statement
theorem total_art_cost : 
  ∀ (price_per_piece : ℤ),
  total_price_first_3_pieces price_per_piece = 45000 →
  next_piece_price = price_per_piece + price_increase price_per_piece →
  total_price_all_arts price_per_piece next_piece_price = 67500 :=
  by
    intros price_per_piece h1 h2
    sorry

end total_art_cost_l1041_104107


namespace number_of_child_workers_l1041_104101

-- Define the conditions
def number_of_male_workers : ℕ := 20
def number_of_female_workers : ℕ := 15
def wage_per_male : ℕ := 35
def wage_per_female : ℕ := 20
def wage_per_child : ℕ := 8
def average_wage : ℕ := 26

-- Define the proof goal
theorem number_of_child_workers (C : ℕ) : 
  ((number_of_male_workers * wage_per_male +
    number_of_female_workers * wage_per_female +
    C * wage_per_child) /
   (number_of_male_workers + number_of_female_workers + C) = average_wage) → 
  C = 5 :=
by 
  sorry

end number_of_child_workers_l1041_104101


namespace find_x_for_salt_solution_l1041_104145

theorem find_x_for_salt_solution : ∀ (x : ℝ),
  (1 + x) * 0.10 = (x * 0.50) →
  x = 0.25 :=
by
  intros x h
  sorry

end find_x_for_salt_solution_l1041_104145
