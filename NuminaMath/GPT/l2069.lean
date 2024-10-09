import Mathlib

namespace possible_integer_lengths_for_third_side_l2069_206976

theorem possible_integer_lengths_for_third_side (x : ℕ) : (8 < x ∧ x < 19) ↔ (4 ≤ x ∧ x ≤ 18) :=
sorry

end possible_integer_lengths_for_third_side_l2069_206976


namespace symmetric_line_equation_wrt_x_axis_l2069_206978

theorem symmetric_line_equation_wrt_x_axis :
  (∀ x y : ℝ, 3 * x + 4 * y + 5 = 0 ↔ 3 * x - 4 * (-y) + 5 = 0) :=
by
  sorry

end symmetric_line_equation_wrt_x_axis_l2069_206978


namespace incorrect_statement_l2069_206951

theorem incorrect_statement :
  ¬ (∀ (l1 l2 l3 : ℝ → ℝ → Prop), 
      (∀ (x y : ℝ), l3 x y → l1 x y) ∧ 
      (∀ (x y : ℝ), l3 x y → l2 x y) → 
      (∀ (x y : ℝ), l1 x y → l2 x y)) :=
by sorry

end incorrect_statement_l2069_206951


namespace students_neither_math_nor_physics_l2069_206946

theorem students_neither_math_nor_physics :
  let total_students := 150
  let students_math := 80
  let students_physics := 60
  let students_both := 20
  total_students - (students_math - students_both + students_physics - students_both + students_both) = 30 :=
by
  sorry

end students_neither_math_nor_physics_l2069_206946


namespace evaluate_expression_l2069_206949

theorem evaluate_expression : (3^3)^4 = 531441 :=
by sorry

end evaluate_expression_l2069_206949


namespace geometric_sequence_sum_5_l2069_206922

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ i j : ℕ, ∃ r : ℝ, a (i + 1) = a i * r ∧ a (j + 1) = a j * r

theorem geometric_sequence_sum_5
  (a : ℕ → ℝ)
  (h : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : a 2 * a 6 + 2 * a 4 * a 5 + (a 5) ^ 2 = 25) :
  a 4 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_5_l2069_206922


namespace number_divisible_by_5_l2069_206956

theorem number_divisible_by_5 (A B C : ℕ) :
  (∃ (k1 k2 k3 k4 k5 k6 : ℕ), 3*10^6 + 10^5 + 7*10^4 + A*10^3 + B*10^2 + 4*10 + C = k1 ∧ 5 * k1 = 0 ∧
                          5 * k2 + 10 = 5 * k2 ∧ 5 * k3 + 5 = 5 * k3 ∧ 
                          5 * k4 + 3 = 5 * k4 ∧ 5 * k5 + 1 = 5 * k5 ∧ 
                          5 * k6 + 7 = 5 * k6) → C = 5 :=
by
  sorry

end number_divisible_by_5_l2069_206956


namespace exists_sum_of_two_squares_l2069_206988

theorem exists_sum_of_two_squares (n : ℤ) (h : n > 10000) : ∃ m : ℤ, (∃ a b : ℤ, m = a^2 + b^2) ∧ 0 < m - n ∧ m - n < 3 * n^(1/4) :=
by
  sorry

end exists_sum_of_two_squares_l2069_206988


namespace units_digit_of_quotient_l2069_206995

theorem units_digit_of_quotient : 
  (7 ^ 2023 + 4 ^ 2023) % 9 = 2 → 
  (7 ^ 2023 + 4 ^ 2023) / 9 % 10 = 0 :=
by
  -- condition: calculation of modulo result
  have h1 : (7 ^ 2023 + 4 ^ 2023) % 9 = 2 := sorry

  -- we have the target statement here
  exact sorry

end units_digit_of_quotient_l2069_206995


namespace eq1_solution_eq2_no_solution_l2069_206944

theorem eq1_solution (x : ℝ) (h : x ≠ 0 ∧ x ≠ 2) :
  (2/x + 1/(x*(x-2)) = 5/(2*x)) ↔ x = 4 :=
by sorry

theorem eq2_no_solution (x : ℝ) (h : x ≠ 2) :
  (5*x - 4)/ (x - 2) = (4*x + 10) / (3*x - 6) - 1 ↔ false :=
by sorry

end eq1_solution_eq2_no_solution_l2069_206944


namespace minimize_expression_l2069_206967

theorem minimize_expression (a : ℝ) : ∃ c : ℝ, 0 ≤ c ∧ c ≤ a ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ a → (x^2 + 3 * (a-x)^2) ≥ ((3*a/4)^2 + 3 * (a-3*a/4)^2)) :=
by
  sorry

end minimize_expression_l2069_206967


namespace sum_distinct_x2_y2_z2_l2069_206901

/-
Given positive integers x, y, and z such that
x + y + z = 30 and gcd(x, y) + gcd(y, z) + gcd(z, x) = 10,
prove that the sum of all possible distinct values of x^2 + y^2 + z^2 is 404.
-/
theorem sum_distinct_x2_y2_z2 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 30) 
  (h_gcd : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10) : 
  x^2 + y^2 + z^2 = 404 :=
sorry

end sum_distinct_x2_y2_z2_l2069_206901


namespace function_C_is_even_l2069_206911

theorem function_C_is_even : ∀ x : ℝ, 2 * (-x)^2 - 1 = 2 * x^2 - 1 :=
by
  intro x
  sorry

end function_C_is_even_l2069_206911


namespace intersect_sets_l2069_206933

   variable (P : Set ℕ) (Q : Set ℕ)

   -- Definitions based on given conditions
   def P_def : Set ℕ := {1, 3, 5}
   def Q_def : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}

   -- Theorem statement in Lean 4
   theorem intersect_sets :
     P = P_def → Q = Q_def → P ∩ Q = {3, 5} :=
   by
     sorry
   
end intersect_sets_l2069_206933


namespace range_of_t_l2069_206921
noncomputable def f (x : ℝ) (t : ℝ) : ℝ := Real.exp (2 * x) - t
noncomputable def g (x : ℝ) (t : ℝ) : ℝ := t * Real.exp x - 1

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x t ≥ g x t) ↔ t ≤ 2 * Real.sqrt 2 - 2 :=
by sorry

end range_of_t_l2069_206921


namespace combined_mpg_rate_l2069_206947

-- Conditions of the problem
def ray_mpg : ℝ := 48
def tom_mpg : ℝ := 24
def ray_distance (s : ℝ) : ℝ := 2 * s
def tom_distance (s : ℝ) : ℝ := s

-- Theorem to prove the combined rate of miles per gallon
theorem combined_mpg_rate (s : ℝ) (h : s > 0) : 
  let total_distance := tom_distance s + ray_distance s
  let ray_gas_usage := ray_distance s / ray_mpg
  let tom_gas_usage := tom_distance s / tom_mpg
  let total_gas_usage := ray_gas_usage + tom_gas_usage
  total_distance / total_gas_usage = 36 := 
by
  sorry

end combined_mpg_rate_l2069_206947


namespace sum_of_roots_l2069_206998

theorem sum_of_roots (r p q : ℝ) 
  (h1 : (3 : ℝ) * r ^ 3 - (9 : ℝ) * r ^ 2 - (48 : ℝ) * r - (12 : ℝ) = 0)
  (h2 : (3 : ℝ) * p ^ 3 - (9 : ℝ) * p ^ 2 - (48 : ℝ) * p - (12 : ℝ) = 0)
  (h3 : (3 : ℝ) * q ^ 3 - (9 : ℝ) * q ^ 2 - (48 : ℝ) * q - (12 : ℝ) = 0)
  (roots_distinct : r ≠ p ∧ r ≠ q ∧ p ≠ q) :
  r + p + q = 3 := 
sorry

end sum_of_roots_l2069_206998


namespace integral_result_l2069_206950

theorem integral_result (b : ℝ) (h : ∫ x in e..b, (2 / x) = 6) : b = Real.exp 4 :=
sorry

end integral_result_l2069_206950


namespace Congcong_CO2_emissions_l2069_206964

-- Definitions based on conditions
def CO2_emissions (t: ℝ) : ℝ := t * 0.91 -- Condition 1: CO2 emissions calculation

def Congcong_water_usage : ℝ := 6 -- Condition 2: Congcong's water usage (6 tons)

-- Statement we want to prove
theorem Congcong_CO2_emissions : CO2_emissions Congcong_water_usage = 5.46 :=
by 
  sorry

end Congcong_CO2_emissions_l2069_206964


namespace larger_number_l2069_206903

theorem larger_number (x y : ℕ) (h1 : x + y = 47) (h2 : x - y = 3) : max x y = 25 :=
sorry

end larger_number_l2069_206903


namespace odd_powers_sum_divisible_by_p_l2069_206929

theorem odd_powers_sum_divisible_by_p
  (p : ℕ)
  (hp_prime : Prime p)
  (hp_gt_3 : 3 < p)
  (a b c d : ℕ)
  (h_sum : (a + b + c + d) % p = 0)
  (h_cube_sum : (a^3 + b^3 + c^3 + d^3) % p = 0)
  (n : ℕ)
  (hn_odd : n % 2 = 1 ) :
  (a^n + b^n + c^n + d^n) % p = 0 :=
sorry

end odd_powers_sum_divisible_by_p_l2069_206929


namespace avg_cards_removed_until_prime_l2069_206907

theorem avg_cards_removed_until_prime:
  let prime_count := 13
  let cards_count := 42
  let non_prime_count := cards_count - prime_count
  let groups_count := prime_count + 1
  let avg_non_prime_per_group := (non_prime_count: ℚ) / (groups_count: ℚ)
  (groups_count: ℚ) > 0 →
  avg_non_prime_per_group + 1 = (43: ℚ) / (14: ℚ) :=
by
  sorry

end avg_cards_removed_until_prime_l2069_206907


namespace new_class_mean_score_l2069_206983

theorem new_class_mean_score : 
  let s1 := 68
  let n1 := 50
  let s2 := 75
  let n2 := 8
  let s3 := 82
  let n3 := 2
  (n1 * s1 + n2 * s2 + n3 * s3) / (n1 + n2 + n3) = 69.4 := by
  sorry

end new_class_mean_score_l2069_206983


namespace roast_cost_l2069_206980

-- Given conditions as described in the problem.
def initial_money : ℝ := 100
def cost_vegetables : ℝ := 11
def money_left : ℝ := 72
def total_spent : ℝ := initial_money - money_left

-- The cost of the roast that we need to prove. We expect it to be €17.
def cost_roast : ℝ := total_spent - cost_vegetables

-- The theorem that states the cost of the roast given the conditions.
theorem roast_cost :
  cost_roast = 100 - 72 - 11 := by
  -- skipping the proof steps with sorry
  sorry

end roast_cost_l2069_206980


namespace distance_between_points_eq_l2069_206979

noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_between_points_eq :
  dist 1 5 7 2 = 3 * Real.sqrt 5 :=
by
  sorry

end distance_between_points_eq_l2069_206979


namespace find_other_number_l2069_206954

theorem find_other_number (a b : ℕ) (h₁ : Nat.lcm a b = 3780) (h₂ : Nat.gcd a b = 18) (h₃ : a = 180) : b = 378 := by
  sorry

end find_other_number_l2069_206954


namespace solve_for_x_l2069_206957

-- Step d: Lean 4 statement
theorem solve_for_x : 
  (∃ x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 2)) → (∃ x : ℚ, x = 1 / 3) :=
sorry

end solve_for_x_l2069_206957


namespace cannot_determine_number_of_pens_l2069_206943

theorem cannot_determine_number_of_pens 
  (P : ℚ) -- marked price of one pen
  (N : ℕ) -- number of pens = 46
  (discount : ℚ := 0.01) -- 1% discount
  (profit_percent : ℚ := 11.91304347826087) -- given profit percent
  : ¬ ∃ (N : ℕ), 
        profit_percent = ((N * P * (1 - discount) - N * P) / (N * P)) * 100 :=
by
  sorry

end cannot_determine_number_of_pens_l2069_206943


namespace value_of_m2_plus_3n2_l2069_206965

noncomputable def real_numbers_with_condition (m n : ℝ) : Prop :=
  (m^2 + 3*n^2)^2 - 4*(m^2 + 3*n^2) - 12 = 0

theorem value_of_m2_plus_3n2 (m n : ℝ) (h : real_numbers_with_condition m n) : m^2 + 3*n^2 = 6 :=
by
  sorry

end value_of_m2_plus_3n2_l2069_206965


namespace weight_of_new_student_l2069_206992

theorem weight_of_new_student (avg_decrease_per_student : ℝ) (num_students : ℕ) (weight_replaced_student : ℝ) (total_reduction : ℝ) 
    (h1 : avg_decrease_per_student = 5) (h2 : num_students = 8) (h3 : weight_replaced_student = 86) (h4 : total_reduction = num_students * avg_decrease_per_student) :
    ∃ (x : ℝ), x = weight_replaced_student - total_reduction ∧ x = 46 :=
by
  use 46
  simp [h1, h2, h3, h4]
  sorry

end weight_of_new_student_l2069_206992


namespace chameleons_color_change_l2069_206994

theorem chameleons_color_change (x : ℕ) 
    (h1 : 140 = 5 * x + (140 - 5 * x)) 
    (h2 : 140 = x + 3 * (140 - 5 * x)) :
    4 * x = 80 :=
by {
    sorry
}

end chameleons_color_change_l2069_206994


namespace greatest_possible_x_l2069_206910

-- Define the numbers and the lcm condition
def num1 := 12
def num2 := 18
def lcm_val := 108

-- Function to calculate the lcm of three numbers
def lcm3 (a b c : ℕ) := Nat.lcm (Nat.lcm a b) c

-- Proposition stating the problem condition
theorem greatest_possible_x (x : ℕ) (h : lcm3 x num1 num2 = lcm_val) : x ≤ lcm_val := sorry

end greatest_possible_x_l2069_206910


namespace charley_initial_pencils_l2069_206942

theorem charley_initial_pencils (P : ℕ) (lost_initially : P - 6 = (P - 1/3 * (P - 6) - 6)) (current_pencils : P - 1/3 * (P - 6) - 6 = 16) : P = 30 := 
sorry

end charley_initial_pencils_l2069_206942


namespace pieces_to_cut_l2069_206970

-- Define the conditions
def rodLength : ℝ := 42.5  -- Length of the rod
def pieceLength : ℝ := 0.85  -- Length of each piece

-- Define the theorem that needs to be proven
theorem pieces_to_cut (h1 : rodLength = 42.5) (h2 : pieceLength = 0.85) : 
  (rodLength / pieceLength) = 50 := 
  by sorry

end pieces_to_cut_l2069_206970


namespace possible_values_of_D_plus_E_l2069_206975

theorem possible_values_of_D_plus_E 
  (D E : ℕ) 
  (hD : 0 ≤ D ∧ D ≤ 9) 
  (hE : 0 ≤ E ∧ E ≤ 9) 
  (hdiv : (D + 8 + 6 + 4 + E + 7 + 2) % 9 = 0) : 
  D + E = 0 ∨ D + E = 9 ∨ D + E = 18 := 
sorry

end possible_values_of_D_plus_E_l2069_206975


namespace expected_value_of_white_balls_l2069_206935

-- Definitions for problem conditions
def totalBalls : ℕ := 6
def whiteBalls : ℕ := 2
def redBalls : ℕ := 4
def ballsDrawn : ℕ := 2

-- Probability calculations
def P_X_0 : ℚ := (Nat.choose 4 2) / (Nat.choose totalBalls ballsDrawn)
def P_X_1 : ℚ := ((Nat.choose whiteBalls 1) * (Nat.choose redBalls 1)) / (Nat.choose totalBalls ballsDrawn)
def P_X_2 : ℚ := (Nat.choose whiteBalls 2) / (Nat.choose totalBalls ballsDrawn)

-- Expected value calculation
def expectedValue : ℚ := (0 * P_X_0) + (1 * P_X_1) + (2 * P_X_2)

theorem expected_value_of_white_balls :
  expectedValue = 2 / 3 :=
by
  sorry

end expected_value_of_white_balls_l2069_206935


namespace physics_kit_prices_l2069_206913

theorem physics_kit_prices :
  ∃ (price_A price_B : ℝ), price_A = 180 ∧ price_B = 150 ∧
    price_A = 1.2 * price_B ∧
    9900 / price_A = 7500 / price_B + 5 :=
by
  use 180, 150
  sorry

end physics_kit_prices_l2069_206913


namespace combined_share_b_d_l2069_206919

-- Definitions for the amounts shared between the children
def total_amount : ℝ := 15800
def share_a_plus_c : ℝ := 7022.222222222222

-- The goal is to prove that the combined share of B and D is 8777.777777777778
theorem combined_share_b_d :
  ∃ B D : ℝ, (B + D = total_amount - share_a_plus_c) :=
by
  sorry

end combined_share_b_d_l2069_206919


namespace vasya_most_points_anya_least_possible_l2069_206932

theorem vasya_most_points_anya_least_possible :
  ∃ (A B V : ℕ) (A_score B_score V_score : ℕ),
  A > B ∧ B > V ∧
  A_score = 9 ∧ B_score = 10 ∧ V_score = 11 ∧
  (∃ (words_common_AB words_common_AV words_only_B words_only_V : ℕ),
  words_common_AB = 6 ∧ words_common_AV = 3 ∧ words_only_B = 2 ∧ words_only_V = 4 ∧
  A = words_common_AB + words_common_AV ∧
  B = words_only_B + words_common_AB ∧
  V = words_only_V + words_common_AV ∧
  A_score = words_common_AB + words_common_AV ∧
  B_score = 2 * words_only_B + words_common_AB ∧
  V_score = 2 * words_only_V + words_common_AV) :=
sorry

end vasya_most_points_anya_least_possible_l2069_206932


namespace towel_bleach_percentage_decrease_l2069_206938

-- Define the problem
theorem towel_bleach_percentage_decrease (L B : ℝ) (x : ℝ) (h_length : 0 < L) (h_breadth : 0 < B) 
  (h1 : 0.64 * L * B = 0.8 * L * (1 - x / 100) * B) :
  x = 20 :=
by
  -- The actual proof is not needed, providing "sorry" as a placeholder for the proof.
  sorry

end towel_bleach_percentage_decrease_l2069_206938


namespace John_spending_l2069_206959

theorem John_spending
  (X : ℝ)
  (h1 : (1/2) * X + (1/3) * X + (1/10) * X + 10 = X) :
  X = 150 :=
by
  sorry

end John_spending_l2069_206959


namespace find_triplets_l2069_206969

theorem find_triplets (a b p : ℕ) (ha : 0 < a) (hb : 0 < b) (hp : Nat.Prime p) (h_eq : (a + b)^p = p^a + p^b) : (a = 1 ∧ b = 1 ∧ p = 2) :=
by
  sorry

end find_triplets_l2069_206969


namespace rectangle_integer_sides_noncongruent_count_l2069_206939

theorem rectangle_integer_sides_noncongruent_count (h w : ℕ) :
  (2 * (w + h) = 72 ∧ w ≠ h) ∨ ((w = h) ∧ 2 * (w + h) = 72) →
  (∃ (count : ℕ), count = 18) :=
by
  sorry

end rectangle_integer_sides_noncongruent_count_l2069_206939


namespace largest_reservoir_is_D_l2069_206993

variables (a : ℝ) 
def final_amount_A : ℝ := a * (1 + 0.1) * (1 - 0.05)
def final_amount_B : ℝ := a * (1 + 0.09) * (1 - 0.04)
def final_amount_C : ℝ := a * (1 + 0.08) * (1 - 0.03)
def final_amount_D : ℝ := a * (1 + 0.07) * (1 - 0.02)

theorem largest_reservoir_is_D
  (hA : final_amount_A a = a * 1.045)
  (hB : final_amount_B a = a * 1.0464)
  (hC : final_amount_C a = a * 1.0476)
  (hD : final_amount_D a = a * 1.0486) :
  final_amount_D a > final_amount_A a ∧ 
  final_amount_D a > final_amount_B a ∧ 
  final_amount_D a > final_amount_C a :=
by sorry

end largest_reservoir_is_D_l2069_206993


namespace cost_of_tax_free_items_l2069_206918

-- Definitions based on the conditions.
def total_spending : ℝ := 20
def sales_tax_percentage : ℝ := 0.30
def tax_rate : ℝ := 0.06

-- Derived calculations for intermediate variables for clarity
def taxable_items_cost : ℝ := total_spending * (1 - sales_tax_percentage)
def sales_tax_paid : ℝ := taxable_items_cost * tax_rate
def tax_free_items_cost : ℝ := total_spending - taxable_items_cost

-- Lean 4 statement for the problem
theorem cost_of_tax_free_items :
  tax_free_items_cost = 6 := by
    -- The proof would go here, but we are skipping it.
    sorry

end cost_of_tax_free_items_l2069_206918


namespace proposition_B_l2069_206958

-- Definitions of the conditions
def line (α : Type) := α
def plane (α : Type) := α
def is_within {α : Type} (a : line α) (p : plane α) : Prop := sorry
def is_perpendicular {α : Type} (a : line α) (p : plane α) : Prop := sorry
def planes_are_perpendicular {α : Type} (p₁ p₂ : plane α) : Prop := sorry
def is_prism (poly : Type) : Prop := sorry

-- Propositions
def p {α : Type} (a : line α) (α₁ α₂ : plane α) : Prop :=
  is_within a α₁ ∧ is_perpendicular a α₂ → planes_are_perpendicular α₁ α₂

def q (poly : Type) : Prop := 
  (∃ (face1 face2 : poly), face1 ≠ face2 ∧ sorry) ∧ sorry

-- Proposition B
theorem proposition_B {α : Type} (a : line α) (α₁ α₂ : plane α) (poly : Type) :
  (p a α₁ α₂) ∧ ¬(q poly) :=
by {
  -- Skipping proof
  sorry
}

end proposition_B_l2069_206958


namespace general_term_formula_is_not_element_l2069_206926

theorem general_term_formula (a : ℕ → ℤ) (h1 : a 1 = 2) (h17 : a 17 = 66) :
  (∀ n, a n = 4 * n - 2) :=
by
  sorry

theorem is_not_element (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 2) :
  ¬ (∃ n : ℕ, a n = 88) :=
by
  sorry

end general_term_formula_is_not_element_l2069_206926


namespace a1964_eq_neg1_l2069_206948

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ ∀ n ≥ 4, a n = a (n-1) * a (n-3)

theorem a1964_eq_neg1 (a : ℕ → ℤ) (h : seq a) : a 1964 = -1 :=
  by sorry

end a1964_eq_neg1_l2069_206948


namespace number_of_cds_on_shelf_l2069_206977

-- Definitions and hypotheses
def cds_per_rack : ℕ := 8
def racks_per_shelf : ℕ := 4

-- Theorem statement
theorem number_of_cds_on_shelf :
  cds_per_rack * racks_per_shelf = 32 :=
by sorry

end number_of_cds_on_shelf_l2069_206977


namespace parabola_num_xintercepts_l2069_206908

-- Defining the equation of the parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- The main theorem to state: the number of x-intercepts for the parabola is 2.
theorem parabola_num_xintercepts : ∃ (a b : ℝ), parabola a = 0 ∧ parabola b = 0 ∧ a ≠ b :=
by
  sorry

end parabola_num_xintercepts_l2069_206908


namespace average_speed_round_trip_l2069_206962

theorem average_speed_round_trip (D : ℝ) (hD : D > 0) :
  let time_uphill := D / 5
  let time_downhill := D / 100
  let total_distance := 2 * D
  let total_time := time_uphill + time_downhill
  let average_speed := total_distance / total_time
  average_speed = 200 / 21 :=
by
  sorry

end average_speed_round_trip_l2069_206962


namespace min_value_of_sum_l2069_206920

noncomputable def min_value_x_3y (x y : ℝ) : ℝ :=
  x + 3 * y

theorem min_value_of_sum (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (cond : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  x + 3 * y ≥ 4 + 4 * Real.sqrt 3 :=
  sorry

end min_value_of_sum_l2069_206920


namespace total_legs_in_farm_l2069_206985

def num_animals : Nat := 13
def num_chickens : Nat := 4
def legs_per_chicken : Nat := 2
def legs_per_buffalo : Nat := 4

theorem total_legs_in_farm : 
  (num_chickens * legs_per_chicken) + ((num_animals - num_chickens) * legs_per_buffalo) = 44 :=
by
  sorry

end total_legs_in_farm_l2069_206985


namespace exists_infinite_repeated_sum_of_digits_l2069_206984

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence a_n which is the sum of digits of P(n)
def a (P : ℕ → ℤ) (n : ℕ) : ℕ :=
  sum_of_digits (P n).natAbs

theorem exists_infinite_repeated_sum_of_digits (P : ℕ → ℤ) (h_nat_coeffs : ∀ n, (P n) ≥ 0) :
  ∃ s : ℕ, ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ a P n = s :=
sorry

end exists_infinite_repeated_sum_of_digits_l2069_206984


namespace Jungkook_blue_balls_unchanged_l2069_206952

variable (initialRedBalls : ℕ) (initialBlueBalls : ℕ) (initialYellowBalls : ℕ)
variable (newYellowBallGifted: ℕ)

-- Define the initial conditions
def Jungkook_balls := initialRedBalls = 5 ∧ initialBlueBalls = 4 ∧ initialYellowBalls = 3 ∧ newYellowBallGifted = 1

-- State the theorem to prove
theorem Jungkook_blue_balls_unchanged (h : Jungkook_balls initRed initBlue initYellow newYellowGift): initialBlueBalls = 4 := 
by
sorry

end Jungkook_blue_balls_unchanged_l2069_206952


namespace molecular_weight_of_compound_l2069_206990

def atomic_weight_Al : ℕ := 27
def atomic_weight_I : ℕ := 127
def atomic_weight_O : ℕ := 16

def num_Al : ℕ := 1
def num_I : ℕ := 3
def num_O : ℕ := 2

def molecular_weight (n_Al n_I n_O w_Al w_I w_O : ℕ) : ℕ :=
  (n_Al * w_Al) + (n_I * w_I) + (n_O * w_O)

theorem molecular_weight_of_compound :
  molecular_weight num_Al num_I num_O atomic_weight_Al atomic_weight_I atomic_weight_O = 440 := 
sorry

end molecular_weight_of_compound_l2069_206990


namespace opinion_change_difference_l2069_206904

variables (initial_enjoy final_enjoy initial_not_enjoy final_not_enjoy : ℕ)
variables (n : ℕ) -- number of students in the class

-- Given conditions
def initial_conditions :=
  initial_enjoy = 40 * n / 100 ∧ initial_not_enjoy = 60 * n / 100

def final_conditions :=
  final_enjoy = 80 * n / 100 ∧ final_not_enjoy = 20 * n / 100

-- The theorem to prove
theorem opinion_change_difference :
  initial_conditions n initial_enjoy initial_not_enjoy →
  final_conditions n final_enjoy final_not_enjoy →
  (40 ≤ initial_enjoy + 20 ∧ 40 ≤ initial_not_enjoy + 20 ∧
  max_change = 60 ∧ min_change = 40 → max_change - min_change = 20) := 
  sorry

end opinion_change_difference_l2069_206904


namespace sin_diff_identity_l2069_206971

variable (α β : ℝ)

def condition1 := (Real.sin α - Real.cos β = 3 / 4)
def condition2 := (Real.cos α + Real.sin β = -2 / 5)

theorem sin_diff_identity : 
  condition1 α β → 
  condition2 α β → 
  Real.sin (α - β) = 511 / 800 :=
by
  intros h1 h2
  sorry

end sin_diff_identity_l2069_206971


namespace cost_of_tissues_l2069_206961
-- Import the entire Mathlib library

-- Define the context and the assertion without computing the proof details
theorem cost_of_tissues
  (n_tp : ℕ) -- Number of toilet paper rolls
  (c_tp : ℝ) -- Cost per toilet paper roll
  (n_pt : ℕ) -- Number of paper towels rolls
  (c_pt : ℝ) -- Cost per paper towel roll
  (n_t : ℕ) -- Number of tissue boxes
  (T : ℝ) -- Total cost of all items
  (H_tp : n_tp = 10) -- Given: 10 rolls of toilet paper
  (H_c_tp : c_tp = 1.5) -- Given: $1.50 per roll of toilet paper
  (H_pt : n_pt = 7) -- Given: 7 rolls of paper towels
  (H_c_pt : c_pt = 2) -- Given: $2 per roll of paper towel
  (H_t : n_t = 3) -- Given: 3 boxes of tissues
  (H_T : T = 35) -- Given: total cost is $35
  : (T - (n_tp * c_tp + n_pt * c_pt)) / n_t = 2 := -- Conclusion: the cost of one box of tissues is $2
by {
  sorry -- Proof details to be supplied here
}

end cost_of_tissues_l2069_206961


namespace find_common_difference_l2069_206989

theorem find_common_difference
  (a_1 : ℕ := 1)
  (S : ℕ → ℕ)
  (h1 : S 5 = 20)
  (h2 : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d))
  : d = 3 / 2 := 
by 
  sorry

end find_common_difference_l2069_206989


namespace current_price_after_adjustment_l2069_206909

variable (x : ℝ) -- Define x, the original price per unit

theorem current_price_after_adjustment (x : ℝ) : (x + 10) * 0.75 = ((x + 10) * 0.75) :=
by
  sorry

end current_price_after_adjustment_l2069_206909


namespace calc_nabla_l2069_206960

noncomputable def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem calc_nabla : (op_nabla (op_nabla 2 3) 4) = 11 / 9 :=
by
  unfold op_nabla
  sorry

end calc_nabla_l2069_206960


namespace linda_color_choices_l2069_206953

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def combination (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem linda_color_choices : combination 8 3 = 56 :=
  by sorry

end linda_color_choices_l2069_206953


namespace complement_U_A_l2069_206996

-- Definitions based on conditions
def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

-- Statement of the problem
theorem complement_U_A :
  (U \ A) = {4} :=
by
  sorry

end complement_U_A_l2069_206996


namespace value_of_g_at_3_l2069_206991

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := 5 * x^3 - 6 * x^2 - 3 * x + 5

-- The theorem statement
theorem value_of_g_at_3 : g 3 = 77 := by
  -- This would require a proof, but we put sorry as instructed
  sorry

end value_of_g_at_3_l2069_206991


namespace vertex_of_parabola_l2069_206966

theorem vertex_of_parabola :
  (∃ (h k : ℤ), ∀ (x : ℝ), y = (x - h)^2 + k) → (h = 2 ∧ k = -3) := by
  sorry

end vertex_of_parabola_l2069_206966


namespace find_letters_with_dot_but_no_straight_line_l2069_206928

-- Define the problem statement and conditions
def DL : ℕ := 16
def L : ℕ := 30
def Total_letters : ℕ := 50

-- Define the function that calculates the number of letters with a dot but no straight line
def letters_with_dot_but_no_straight_line (DL L Total_letters : ℕ) : ℕ := Total_letters - (L + DL)

-- State the theorem to be proved
theorem find_letters_with_dot_but_no_straight_line : letters_with_dot_but_no_straight_line DL L Total_letters = 4 :=
by
  sorry

end find_letters_with_dot_but_no_straight_line_l2069_206928


namespace cubed_expression_value_l2069_206955

open Real

theorem cubed_expression_value (a b c : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b + 2 * c = 0) :
  (a^3 + b^3 + 2 * c^3) / (a * b * c) = -3 * (a^2 - a * b + b^2) / (2 * a * b) :=
  sorry

end cubed_expression_value_l2069_206955


namespace five_digit_numbers_last_two_different_l2069_206917

def total_five_digit_numbers : ℕ := 90000

def five_digit_numbers_last_two_same : ℕ := 9000

theorem five_digit_numbers_last_two_different :
  (total_five_digit_numbers - five_digit_numbers_last_two_same) = 81000 := 
by 
  sorry

end five_digit_numbers_last_two_different_l2069_206917


namespace probability_of_fx_leq_zero_is_3_over_10_l2069_206902

noncomputable def fx (x : ℝ) : ℝ := -x + 2

def in_interval (x : ℝ) (a b : ℝ) : Prop := a ≤ x ∧ x ≤ b

def probability_fx_leq_zero : ℚ :=
  let interval_start := -5
  let interval_end := 5
  let fx_leq_zero_start := 2
  let fx_leq_zero_end := 5
  (fx_leq_zero_end - fx_leq_zero_start) / (interval_end - interval_start)

theorem probability_of_fx_leq_zero_is_3_over_10 :
  probability_fx_leq_zero = 3 / 10 :=
sorry

end probability_of_fx_leq_zero_is_3_over_10_l2069_206902


namespace simplify_expression_1_simplify_expression_2_l2069_206937

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) : 4 * (a + b) + 2 * (a + b) - (a + b) = 5 * a + 5 * b :=
  sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) : (3 * m / 2) - (5 * m / 2 - 1) + 3 * (4 - m) = -4 * m + 13 :=
  sorry

end simplify_expression_1_simplify_expression_2_l2069_206937


namespace smallest_n_for_divisibility_property_l2069_206936

theorem smallest_n_for_divisibility_property (k : ℕ) : ∃ n : ℕ, n = k + 2 ∧ ∀ (S : Finset ℤ), 
  S.card = n → 
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ (a ≠ b ∧ (a + b) % (2 * k + 1) = 0 ∨ (a - b) % (2 * k + 1) = 0) :=
by
sorry

end smallest_n_for_divisibility_property_l2069_206936


namespace denote_loss_of_300_dollars_l2069_206925

-- Define the concept of financial transactions
def denote_gain (amount : Int) : Int := amount
def denote_loss (amount : Int) : Int := -amount

-- The condition given in the problem
def earn_500_dollars_is_500 := denote_gain 500 = 500

-- The assertion we need to prove
theorem denote_loss_of_300_dollars : denote_loss 300 = -300 := 
by 
  sorry

end denote_loss_of_300_dollars_l2069_206925


namespace last_digit_x4_plus_inv_x4_l2069_206916

theorem last_digit_x4_plus_inv_x4 (x : ℝ) (h : x^2 - 13 * x + 1 = 0) : (x^4 + (1 / x)^4) % 10 = 7 := 
by
  sorry

end last_digit_x4_plus_inv_x4_l2069_206916


namespace max_sequence_value_l2069_206934

theorem max_sequence_value : 
  ∃ n ∈ (Set.univ : Set ℤ), (∀ m ∈ (Set.univ : Set ℤ), -m^2 + 15 * m + 3 ≤ -n^2 + 15 * n + 3) ∧ (-n^2 + 15 * n + 3 = 59) :=
by
  sorry

end max_sequence_value_l2069_206934


namespace find_CP_A_l2069_206940

noncomputable def CP_A : Float := 173.41
def SP_B (CP_A : Float) : Float := 1.20 * CP_A
def SP_C (SP_B : Float) : Float := 1.25 * SP_B
def TC_C (SP_C : Float) : Float := 1.15 * SP_C
def SP_D1 (TC_C : Float) : Float := 1.30 * TC_C
def SP_D2 (SP_D1 : Float) : Float := 0.90 * SP_D1
def SP_D2_actual : Float := 350

theorem find_CP_A : 
  (SP_D2 (SP_D1 (TC_C (SP_C (SP_B CP_A))))) = SP_D2_actual → 
  CP_A = 173.41 := sorry

end find_CP_A_l2069_206940


namespace min_games_to_achieve_98_percent_l2069_206999

-- Define initial conditions
def initial_games : ℕ := 5
def initial_sharks_wins : ℕ := 2
def initial_tigers_wins : ℕ := 3

-- Define the total number of games and the total number of wins by the Sharks after additional games
def total_games (N : ℕ) : ℕ := initial_games + N
def total_sharks_wins (N : ℕ) : ℕ := initial_sharks_wins + N

-- Define the Sharks' winning percentage
def sharks_winning_percentage (N : ℕ) : ℚ := total_sharks_wins N / total_games N

-- Define the minimum number of additional games needed
def minimum_N : ℕ := 145

-- Theorem: Prove that the Sharks' winning percentage is at least 98% when N = 145
theorem min_games_to_achieve_98_percent :
  sharks_winning_percentage minimum_N ≥ 49 / 50 :=
sorry

end min_games_to_achieve_98_percent_l2069_206999


namespace cars_count_l2069_206972

-- Define the number of cars as x
variable (x : ℕ)

-- The conditions for the problem
def condition1 := 3 * (x - 2)
def condition2 := 2 * x + 9

-- The main theorem stating that under the given conditions, x = 15
theorem cars_count : condition1 x = condition2 x → x = 15 := by
  sorry

end cars_count_l2069_206972


namespace remainder_when_divided_by_10_l2069_206912

theorem remainder_when_divided_by_10 :
  (4219 * 2675 * 394082 * 5001) % 10 = 0 :=
sorry

end remainder_when_divided_by_10_l2069_206912


namespace animal_counts_l2069_206982

-- Definitions based on given conditions
def ReptileHouse (R : ℕ) : ℕ := 3 * R - 5
def Aquarium (ReptileHouse : ℕ) : ℕ := 2 * ReptileHouse
def Aviary (Aquarium RainForest : ℕ) : ℕ := (Aquarium - RainForest) + 3

-- The main theorem statement
theorem animal_counts
  (R : ℕ)
  (ReptileHouse_eq : ReptileHouse R = 16)
  (A : ℕ := Aquarium 16)
  (V : ℕ := Aviary A R) :
  (R = 7) ∧ (A = 32) ∧ (V = 28) :=
by
  sorry

end animal_counts_l2069_206982


namespace probability_third_draw_first_class_expected_value_first_class_in_10_draws_l2069_206924

-- Define the problem with products
structure Products where
  total : ℕ
  first_class : ℕ
  second_class : ℕ

-- Given products configuration
def products : Products := { total := 5, first_class := 3, second_class := 2 }

-- Probability calculation without replacement
-- Define the event of drawing
def draw_without_replacement (p : Products) (draws : ℕ) (desired_event : ℕ -> Bool) : ℚ := 
  if draws = 3 ∧ desired_event 3 ∧ ¬ desired_event 1 ∧ ¬ desired_event 2 then
    (2 / 5) * ((1 : ℚ) / 4) * (3 / 3)
  else 
    0

-- Define desired_event for the specific problem
def desired_event (n : ℕ) : Bool := 
  match n with
  | 3 => true
  | _ => false

-- The first problem's proof statement
theorem probability_third_draw_first_class : draw_without_replacement products 3 desired_event = 1 / 10 := sorry

-- Expected value calculation with replacement
-- Binomial distribution to find expected value
def expected_value_with_replacement (p : Products) (draws : ℕ) : ℚ :=
  draws * (p.first_class / p.total)

-- The second problem's proof statement
theorem expected_value_first_class_in_10_draws : expected_value_with_replacement products 10 = 6 := sorry

end probability_third_draw_first_class_expected_value_first_class_in_10_draws_l2069_206924


namespace paint_fraction_second_week_l2069_206923

theorem paint_fraction_second_week
  (total_paint : ℕ)
  (first_week_fraction : ℚ)
  (total_used : ℕ)
  (paint_first_week : ℕ)
  (remaining_paint : ℕ)
  (paint_second_week : ℕ)
  (fraction_second_week : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1/4 →
  total_used = 225 →
  paint_first_week = first_week_fraction * total_paint →
  remaining_paint = total_paint - paint_first_week →
  paint_second_week = total_used - paint_first_week →
  fraction_second_week = paint_second_week / remaining_paint →
  fraction_second_week = 1/2 :=
by
  sorry

end paint_fraction_second_week_l2069_206923


namespace max_area_triangle_l2069_206930

theorem max_area_triangle (A B C : ℝ) (a b c : ℝ) (h1 : Real.sqrt 2 * Real.sin A = Real.sqrt 3 * Real.cos A) (h2 : a = Real.sqrt 3) :
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / (8 * Real.sqrt 5) := 
sorry

end max_area_triangle_l2069_206930


namespace senya_mistakes_in_OCTAHEDRON_l2069_206931

noncomputable def mistakes_in_word (word : String) : Nat :=
  if word = "TETRAHEDRON" then 5
  else if word = "DODECAHEDRON" then 6
  else if word = "ICOSAHEDRON" then 7
  else if word = "OCTAHEDRON" then 5 
  else 0

theorem senya_mistakes_in_OCTAHEDRON : mistakes_in_word "OCTAHEDRON" = 5 := by
  sorry

end senya_mistakes_in_OCTAHEDRON_l2069_206931


namespace tan_45_add_reciprocal_half_add_abs_neg_two_eq_five_l2069_206981

theorem tan_45_add_reciprocal_half_add_abs_neg_two_eq_five :
  (Real.tan (Real.pi / 4) + (1 / 2)⁻¹ + |(-2 : ℝ)|) = 5 :=
by
  -- Assuming the conditions provided in part a)
  have h1 : Real.tan (Real.pi / 4) = 1 := by sorry
  have h2 : (1 / 2 : ℝ)⁻¹ = 2 := by sorry
  have h3 : |(-2 : ℝ)| = 2 := by sorry

  -- Proof of the problem using the conditions
  rw [h1, h2, h3]
  norm_num

end tan_45_add_reciprocal_half_add_abs_neg_two_eq_five_l2069_206981


namespace find_f_2011_l2069_206974

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2011 :
  (∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3 * x + 2) = 9 * x^2 - 15 * x) →
  f 2011 = 6029 :=
by
  intros hf
  sorry

end find_f_2011_l2069_206974


namespace point_of_tangency_l2069_206927

noncomputable def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 10 * x + 14
noncomputable def parabola2 (y : ℝ) : ℝ := 4 * y^2 + 16 * y + 68

theorem point_of_tangency : 
  ∃ (x y : ℝ), parabola1 x = y ∧ parabola2 y = x ∧ x = -9/4 ∧ y = -15/8 :=
by
  -- The proof will show that the point of tangency is (-9/4, -15/8)
  sorry

end point_of_tangency_l2069_206927


namespace intersection_correct_l2069_206914

def setA : Set ℝ := { x | x - 1 ≤ 0 }
def setB : Set ℝ := { x | x^2 - 4 * x ≤ 0 }
def expected_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_correct : (setA ∩ setB) = expected_intersection :=
sorry

end intersection_correct_l2069_206914


namespace inequality_solution_set_range_of_a_l2069_206905

section
variable {x a : ℝ}

def f (x a : ℝ) := |2 * x - 5 * a| + |2 * x + 1|
def g (x : ℝ) := |x - 1| + 3

theorem inequality_solution_set :
  {x : ℝ | |g x| < 8} = {x : ℝ | -4 < x ∧ x < 6} :=
sorry

theorem range_of_a (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, f x₁ a = g x₂) :
  a ≥ 0.4 ∨ a ≤ -0.8 :=
sorry
end

end inequality_solution_set_range_of_a_l2069_206905


namespace minimum_value_am_bn_l2069_206945

-- Definitions and conditions
variables {a b m n : ℝ}
variables (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < m) (h₃ : 0 < n)
variables (h₄ : a + b = 1) (h₅ : m * n = 2)

-- Statement of the proof problem
theorem minimum_value_am_bn :
  ∃ c, (∀ a b m n : ℝ, 0 < a → 0 < b → 0 < m → 0 < n → a + b = 1 → m * n = 2 → (am * bn) * (bm * an) ≥ c) ∧ c = 2 :=
sorry

end minimum_value_am_bn_l2069_206945


namespace inequality_lemma_l2069_206963

theorem inequality_lemma (a b c d : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) (hd : 0 < d ∧ d < 1) :
  1 + a * b + b * c + c * d + d * a + a * c + b * d > a + b + c + d :=
by 
  sorry

end inequality_lemma_l2069_206963


namespace product_of_positive_solutions_l2069_206968

theorem product_of_positive_solutions :
  ∃ n : ℕ, ∃ p : ℕ, Prime p ∧ (n^2 - 41*n + 408 = p) ∧ (∀ m : ℕ, (Prime p ∧ (m^2 - 41*m + 408 = p)) → m = n) ∧ (n = 406) := 
sorry

end product_of_positive_solutions_l2069_206968


namespace factorize_a3_minus_ab2_l2069_206986

theorem factorize_a3_minus_ab2 (a b: ℝ) : 
  a^3 - a * b^2 = a * (a + b) * (a - b) :=
by
  sorry

end factorize_a3_minus_ab2_l2069_206986


namespace maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l2069_206900

-- Defining productivity functions for male and female kittens
def male_productivity (K : ℝ) : ℝ := 80 - 4 * K
def female_productivity (K : ℝ) : ℝ := 16 - 0.25 * K

-- Condition (a): Maximizing number of mice caught by 2 kittens
theorem maximize_mice_two_kittens : 
  ∃ (male1 male2 : ℝ) (K_m1 K_m2 : ℝ), 
    (male1 = male_productivity K_m1) ∧ 
    (male2 = male_productivity K_m2) ∧
    (K_m1 = 0) ∧ (K_m2 = 0) ∧
    (male1 + male2 = 160) := 
sorry

-- Condition (b): Different versions of JPPF
theorem different_versions_JPPF : 
  ∃ (v1 v2 v3 : Unit), 
    (v1 ≠ v2) ∧ (v2 ≠ v3) ∧ (v1 ≠ v3) :=
sorry

-- Condition (c): Analytical form of JPPF for each combination
theorem JPPF_combinations :
  ∃ (M K1 K2 : ℝ),
    (M = 160 - 4 * K1 ∧ K1 ≤ 40) ∨
    (M = 32 - 0.5 * K2 ∧ K2 ≤ 64) ∨
    (M = 96 - 0.25 * K2 ∧ K2 ≤ 64) ∨
    (M = 336 - 4 * K2 ∧ 64 < K2 ∧ K2 ≤ 84) :=
sorry

-- Condition (d): Analytical form for 2 males and 1 female
theorem JPPF_two_males_one_female :
  ∃ (M K : ℝ), 
    (0 < K ∧ K ≤ 64 ∧ M = 176 - 0.25 * K) ∨
    (64 < K ∧ K ≤ 164 ∧ M = 416 - 4 * K) :=
sorry

end maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l2069_206900


namespace find_primes_l2069_206906

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

/- Define the three conditions -/
def condition1 (p q r : ℕ) : Prop := divides p (1 + q ^ r)
def condition2 (p q r : ℕ) : Prop := divides q (1 + r ^ p)
def condition3 (p q r : ℕ) : Prop := divides r (1 + p ^ q)

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ condition1 p q r ∧ condition2 p q r ∧ condition3 p q r

theorem find_primes (p q r : ℕ) :
  satisfies_conditions p q r ↔ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) ∨ (p = 3 ∧ q = 2 ∧ r = 5) :=
by
  sorry

end find_primes_l2069_206906


namespace choir_meets_every_5_days_l2069_206987

theorem choir_meets_every_5_days (n : ℕ) (h1 : n = 15) (h2 : ∃ k : ℕ, 15 = 3 * k) : ∃ x : ℕ, 15 = x * 3 ∧ x = 5 := 
by
  sorry

end choir_meets_every_5_days_l2069_206987


namespace find_number_l2069_206915

theorem find_number (x : ℝ) (h : 0.5 * x = 0.1667 * x + 10) : x = 30 :=
sorry

end find_number_l2069_206915


namespace bee_paths_to_hive_6_correct_l2069_206973

noncomputable def num_paths_to_hive_6 : ℕ := 21

theorem bee_paths_to_hive_6_correct
  (start_pos : ℕ)
  (end_pos : ℕ)
  (bee_can_only_crawl : Prop)
  (bee_can_move_right : Prop)
  (bee_can_move_upper_right : Prop)
  (bee_can_move_lower_right : Prop)
  (total_hives : ℕ)
  (start_pos_is_initial : start_pos = 0)
  (end_pos_is_six : end_pos = 6) :
  num_paths_to_hive_6 = 21 :=
by
  sorry

end bee_paths_to_hive_6_correct_l2069_206973


namespace devin_teaching_years_l2069_206997

theorem devin_teaching_years :
  let calculus_years := 4
  let algebra_years := 2 * calculus_years
  let statistics_years := 5 * algebra_years
  calculus_years + algebra_years + statistics_years = 52 :=
by
  let calculus_years := 4
  let algebra_years := 2 * calculus_years
  let statistics_years := 5 * algebra_years
  show calculus_years + algebra_years + statistics_years = 52
  sorry

end devin_teaching_years_l2069_206997


namespace simplify_expression_l2069_206941

variable (x : ℝ)

theorem simplify_expression : 
  2 * x^3 - (7 * x^2 - 9 * x) - 2 * (x^3 - 3 * x^2 + 4 * x) = -x^2 + x := 
by
  sorry

end simplify_expression_l2069_206941
