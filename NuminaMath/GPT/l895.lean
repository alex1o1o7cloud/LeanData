import Mathlib

namespace minimize_distances_is_k5_l895_89579

-- Define the coordinates of points A, B, and D
def A : ℝ × ℝ := (4, 3)
def B : ℝ × ℝ := (1, 2)
def D : ℝ × ℝ := (0, 5)

-- Define C as a point vertically below D, implying the x-coordinate is the same as that of D and y = k
def C (k : ℝ) : ℝ × ℝ := (0, k)

-- Prove that the value of k that minimizes the distances over AC and BC is k = 5
theorem minimize_distances_is_k5 : ∃ k : ℝ, (C k = (0, 5)) ∧ k = 5 :=
by {
  sorry
}

end minimize_distances_is_k5_l895_89579


namespace remainder_of_M_l895_89516

def M : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_of_M : M % 32 = 31 := by
  -- Proof goes here
  sorry

end remainder_of_M_l895_89516


namespace rahim_pillows_l895_89586

theorem rahim_pillows (x T : ℕ) (h1 : T = 5 * x) (h2 : (T + 10) / (x + 1) = 6) : x = 4 :=
by
  sorry

end rahim_pillows_l895_89586


namespace description_of_T_l895_89503

def T : Set (ℝ × ℝ) := { p | ∃ c, (4 = p.1 + 3 ∨ 4 = p.2 - 2 ∨ p.1 + 3 = p.2 - 2) 
                           ∧ (p.1 + 3 ≤ c ∨ p.2 - 2 ≤ c ∨ 4 ≤ c) }

theorem description_of_T : 
  (∀ p ∈ T, (∃ x y : ℝ, p = (x, y) ∧ ((x = 1 ∧ y ≤ 6) ∨ (y = 6 ∧ x ≤ 1) ∨ (y = x + 5 ∧ x ≥ 1 ∧ y ≥ 6)))) :=
sorry

end description_of_T_l895_89503


namespace neg_number_is_A_l895_89557

def A : ℤ := -(3 ^ 2)
def B : ℤ := (-3) ^ 2
def C : ℤ := abs (-3)
def D : ℤ := -(-3)

theorem neg_number_is_A : A < 0 := 
by sorry

end neg_number_is_A_l895_89557


namespace prime_div_p_sq_minus_one_l895_89536

theorem prime_div_p_sq_minus_one {p : ℕ} (hp : p ≥ 7) (hp_prime : Nat.Prime p) : 
  (p % 10 = 1 ∨ p % 10 = 9) → 40 ∣ (p^2 - 1) :=
sorry

end prime_div_p_sq_minus_one_l895_89536


namespace fraction_meaningful_l895_89567

theorem fraction_meaningful (x : ℝ) : (x - 1 ≠ 0) ↔ (∃ (y : ℝ), y = 3 / (x - 1)) :=
by sorry

end fraction_meaningful_l895_89567


namespace river_depth_l895_89515

theorem river_depth (width depth : ℝ) (flow_rate_kmph : ℝ) (volume_m3_per_min : ℝ) 
  (h1 : width = 75) 
  (h2 : flow_rate_kmph = 4) 
  (h3 : volume_m3_per_min = 35000) : 
  depth = 7 := 
by
  sorry

end river_depth_l895_89515


namespace problem1_problem2_l895_89551

variable {x : ℝ} (hx : x > 0)

theorem problem1 : (2 / (3 * x)) * Real.sqrt (9 * x^3) + 6 * Real.sqrt (x / 4) - 2 * x * Real.sqrt (1 / x) = 3 * Real.sqrt x := 
by sorry

theorem problem2 : (Real.sqrt 24 + Real.sqrt 6) / Real.sqrt 3 + (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 3 * Real.sqrt 2 + 2 := 
by sorry

end problem1_problem2_l895_89551


namespace value_of_y_l895_89599

theorem value_of_y (y : ℤ) (h : (2010 + y)^2 = y^2) : y = -1005 :=
sorry

end value_of_y_l895_89599


namespace max_min_product_l895_89512

theorem max_min_product (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x + y + z = 15) (h2 : x * y + y * z + z * x = 45) :
    ∃ m : ℝ, m = min (x * y) (min (y * z) (z * x)) ∧ m ≤ 17.5 :=
by
  sorry

end max_min_product_l895_89512


namespace sequence_property_implies_geometric_progression_l895_89566

theorem sequence_property_implies_geometric_progression {p : ℝ} {a : ℕ → ℝ}
  (h_p : (2 / (Real.sqrt 5 + 1) ≤ p) ∧ (p < 1))
  (h_a : ∀ (e : ℕ → ℤ), (∀ n, (e n = 0) ∨ (e n = 1) ∨ (e n = -1)) →
    (∑' n, (e n) * (p ^ n)) = 0 → (∑' n, (e n) * (a n)) = 0) :
  ∃ c : ℝ, ∀ n, a n = c * (p ^ n) := by
  sorry

end sequence_property_implies_geometric_progression_l895_89566


namespace problem_statement_l895_89554

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := { x : ℝ | abs x < 2 }
noncomputable def N : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x - 1 }

theorem problem_statement :
  compl M ∪ compl N = Iic (-1) ∪ Ici 2 :=
by {
  sorry
}

end problem_statement_l895_89554


namespace total_cost_of_crayons_l895_89500

theorem total_cost_of_crayons (crayons_per_half_dozen : ℕ)
    (number_of_half_dozens : ℕ)
    (cost_per_crayon : ℕ)
    (total_cost : ℕ) :
  crayons_per_half_dozen = 6 →
  number_of_half_dozens = 4 →
  cost_per_crayon = 2 →
  total_cost = crayons_per_half_dozen * number_of_half_dozens * cost_per_crayon →
  total_cost = 48 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end total_cost_of_crayons_l895_89500


namespace cats_not_liking_catnip_or_tuna_l895_89501

theorem cats_not_liking_catnip_or_tuna :
  ∀ (total_cats catnip_lovers tuna_lovers both_lovers : ℕ),
  total_cats = 80 →
  catnip_lovers = 15 →
  tuna_lovers = 60 →
  both_lovers = 10 →
  (total_cats - (catnip_lovers - both_lovers + both_lovers + tuna_lovers - both_lovers)) = 15 :=
by
  intros total_cats catnip_lovers tuna_lovers both_lovers ht hc ht hboth
  sorry

end cats_not_liking_catnip_or_tuna_l895_89501


namespace line_equation_l895_89556

open Real

-- Define the points A, B, and C
def A : ℝ × ℝ := ⟨1, 4⟩
def B : ℝ × ℝ := ⟨3, 2⟩
def C : ℝ × ℝ := ⟨2, -1⟩

-- Definition for a line passing through point C
-- and having equal distance to points A and B
def is_line_equation (l : ℝ → ℝ → Prop) :=
  ∀ x y, (l x y ↔ (x + y - 1 = 0 ∨ x - 2 = 0))

-- Our main statement
theorem line_equation :
  ∃ l : ℝ → ℝ → Prop, is_line_equation l ∧ (l 2 (-1)) :=
by
  sorry  -- Proof goes here.

end line_equation_l895_89556


namespace calculate_opening_price_l895_89523

theorem calculate_opening_price (C : ℝ) (r : ℝ) (P : ℝ) 
  (h1 : C = 15)
  (h2 : r = 0.5)
  (h3 : C = P + r * P) :
  P = 10 :=
by sorry

end calculate_opening_price_l895_89523


namespace floor_cube_neg_seven_four_l895_89572

theorem floor_cube_neg_seven_four :
  (Int.floor ((-7 / 4 : ℚ) ^ 3) = -6) :=
by
  sorry

end floor_cube_neg_seven_four_l895_89572


namespace total_time_equiv_7_75_l895_89552

def acclimation_period : ℝ := 1
def learning_basics : ℝ := 2
def research_time_without_sabbatical : ℝ := learning_basics + 0.75 * learning_basics
def sabbatical : ℝ := 0.5
def research_time_with_sabbatical : ℝ := research_time_without_sabbatical + sabbatical
def dissertation_without_conference : ℝ := 0.5 * acclimation_period
def conference : ℝ := 0.25
def dissertation_with_conference : ℝ := dissertation_without_conference + conference
def total_time : ℝ := acclimation_period + learning_basics + research_time_with_sabbatical + dissertation_with_conference

theorem total_time_equiv_7_75 : total_time = 7.75 := by
  sorry

end total_time_equiv_7_75_l895_89552


namespace h_h_3_eq_3568_l895_89558

def h (x : ℤ) : ℤ := 3 * x * x + 3 * x - 2

theorem h_h_3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h_3_eq_3568_l895_89558


namespace max_value_5x_minus_25x_l895_89582

open Real

theorem max_value_5x_minus_25x : 
  ∃ x : ℝ, ∀ y : ℝ, (y = 5^x) → (y - y^2) ≤ 1 / 4 := 
by 
  sorry

end max_value_5x_minus_25x_l895_89582


namespace matrix_vector_addition_l895_89577

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -2], ![-5, 6]]
def v : Fin 2 → ℤ := ![5, -2]
def w : Fin 2 → ℤ := ![1, -1]

theorem matrix_vector_addition :
  (A.mulVec v + w) = ![25, -38] :=
by
  sorry

end matrix_vector_addition_l895_89577


namespace factor_expr_l895_89519

theorem factor_expr (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3 * x + x^2) := 
sorry

end factor_expr_l895_89519


namespace gross_profit_percentage_without_discount_l895_89541

theorem gross_profit_percentage_without_discount (C P : ℝ)
  (discount : P * 0.9 = C * 1.2)
  (discount_profit : C * 0.2 = P * 0.9 - C) :
  (P - C) / C * 100 = 33.3 :=
by
  sorry

end gross_profit_percentage_without_discount_l895_89541


namespace solve_quadratic_eq_l895_89590

theorem solve_quadratic_eq (x : ℝ) :
  (x^2 + (x - 1) * (x + 3) = 3 * x + 5) ↔ (x = -2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_eq_l895_89590


namespace three_a_in_S_implies_a_in_S_l895_89563

def S := {n | ∃ x y : ℤ, n = x^2 + 2 * y^2}

theorem three_a_in_S_implies_a_in_S (a : ℤ) (h : 3 * a ∈ S) : a ∈ S := 
sorry

end three_a_in_S_implies_a_in_S_l895_89563


namespace capacity_of_new_vessel_is_10_l895_89544

-- Define the conditions
def first_vessel_capacity : ℕ := 2
def first_vessel_concentration : ℚ := 0.25
def second_vessel_capacity : ℕ := 6
def second_vessel_concentration : ℚ := 0.40
def total_liquid_combined : ℕ := 8
def new_mixture_concentration : ℚ := 0.29
def total_alcohol_content : ℚ := (first_vessel_capacity * first_vessel_concentration) + (second_vessel_capacity * second_vessel_concentration)
def desired_vessel_capacity : ℚ := total_alcohol_content / new_mixture_concentration

-- The theorem we want to prove
theorem capacity_of_new_vessel_is_10 : desired_vessel_capacity = 10 := by
  sorry

end capacity_of_new_vessel_is_10_l895_89544


namespace sean_has_45_whistles_l895_89592

variable (Sean Charles : ℕ)

def sean_whistles (Charles : ℕ) : ℕ :=
  Charles + 32

theorem sean_has_45_whistles
    (Charles_whistles : Charles = 13) 
    (Sean_whistles_condition : Sean = sean_whistles Charles) :
    Sean = 45 := by
  sorry

end sean_has_45_whistles_l895_89592


namespace Jackie_hops_six_hops_distance_l895_89507

theorem Jackie_hops_six_hops_distance : 
  let a : ℝ := 1
  let r : ℝ := 1 / 2
  let S : ℝ := a * ((1 - r^6) / (1 - r))
  S = 63 / 32 :=
by 
  sorry

end Jackie_hops_six_hops_distance_l895_89507


namespace find_salary_J_l895_89591

variables {J F M A May : ℝ}
variables (h1 : (J + F + M + A) / 4 = 8000)
variables (h2 : (F + M + A + May) / 4 = 8200)
variables (h3 : May = 6500)

theorem find_salary_J : J = 5700 :=
by
  sorry

end find_salary_J_l895_89591


namespace gcd_18222_24546_66364_eq_2_l895_89534

/-- Definition of three integers a, b, c --/
def a : ℕ := 18222 
def b : ℕ := 24546
def c : ℕ := 66364

/-- Proof of the gcd of the three integers being 2 --/
theorem gcd_18222_24546_66364_eq_2 : Nat.gcd (Nat.gcd a b) c = 2 := by
  sorry

end gcd_18222_24546_66364_eq_2_l895_89534


namespace possible_values_of_ab_plus_ac_plus_bc_l895_89537

theorem possible_values_of_ab_plus_ac_plus_bc (a b c : ℝ) (h : a + b + 2 * c = 0) :
  ∃ x ∈ Set.Iic 0, ab + ac + bc = x :=
sorry

end possible_values_of_ab_plus_ac_plus_bc_l895_89537


namespace first_term_correct_l895_89528

noncomputable def first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - (r^3)) = 80) : ℝ :=
a

theorem first_term_correct (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - (r^3)) = 80) :
  first_term a r h1 h2 = 3.42 :=
sorry

end first_term_correct_l895_89528


namespace value_of_expression_l895_89505

theorem value_of_expression (a b : ℝ) (h : -3 * a - b = -1) : 3 - 6 * a - 2 * b = 1 :=
by
  sorry

end value_of_expression_l895_89505


namespace initial_blueberry_jelly_beans_l895_89502

-- Definitions for initial numbers of jelly beans and modified quantities after eating
variables (b c : ℕ)

-- Conditions stated as Lean hypothesis
axiom initial_relation : b = 2 * c
axiom new_relation : b - 5 = 4 * (c - 5)

-- Theorem statement to prove the initial number of blueberry jelly beans is 30
theorem initial_blueberry_jelly_beans : b = 30 :=
by
  sorry

end initial_blueberry_jelly_beans_l895_89502


namespace steps_to_Madison_eq_991_l895_89569

variable (steps_down steps_to_Madison : ℕ)

def total_steps (steps_down steps_to_Madison : ℕ) : ℕ :=
  steps_down + steps_to_Madison

theorem steps_to_Madison_eq_991 (h1 : steps_down = 676) (h2 : steps_to_Madison = 315) :
  total_steps steps_down steps_to_Madison = 991 :=
by
  sorry

end steps_to_Madison_eq_991_l895_89569


namespace divide_square_into_smaller_squares_l895_89538

-- Definition of the property P(n)
def P (n : ℕ) : Prop := ∃ (f : ℕ → ℕ), ∀ i, i < n → (f i > 0)

-- Proposition for the problem
theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
sorry

end divide_square_into_smaller_squares_l895_89538


namespace total_food_each_day_l895_89555

-- Conditions
def num_dogs : ℕ := 2
def food_per_dog : ℝ := 0.125
def total_food : ℝ := num_dogs * food_per_dog

-- Proof statement
theorem total_food_each_day : total_food = 0.25 :=
by
  sorry

end total_food_each_day_l895_89555


namespace total_profit_l895_89584

-- Define the variables for the subscriptions and profits
variables {A B C : ℕ} -- Subscription amounts
variables {profit : ℕ} -- Total profit

-- Given conditions
def conditions (A B C : ℕ) (profit : ℕ) :=
  50000 = A + B + C ∧
  A = B + 4000 ∧
  B = C + 5000 ∧
  A * profit = 29400 * 50000

-- Statement of the theorem
theorem total_profit (A B C : ℕ) (profit : ℕ) (h : conditions A B C profit) :
  profit = 70000 :=
sorry

end total_profit_l895_89584


namespace quadratic_solution_l895_89543

theorem quadratic_solution (a c: ℝ) (h1 : a + c = 7) (h2 : a < c) (h3 : 36 - 4 * a * c = 0) : 
  a = (7 - Real.sqrt 13) / 2 ∧ c = (7 + Real.sqrt 13) / 2 :=
by
  sorry

end quadratic_solution_l895_89543


namespace total_profit_correct_l895_89589

noncomputable def total_profit (a b c : ℕ) (c_share : ℕ) : ℕ :=
  let ratio := a + b + c
  let part_value := c_share / c
  ratio * part_value

theorem total_profit_correct (h_a : ℕ := 5000) (h_b : ℕ := 8000) (h_c : ℕ := 9000) (h_c_share : ℕ := 36000) :
  total_profit h_a h_b h_c h_c_share = 88000 :=
by
  sorry

end total_profit_correct_l895_89589


namespace option_D_not_equal_l895_89526

def frac1 := (-15 : ℚ) / 12
def fracA := (-30 : ℚ) / 24
def fracB := -1 - (3 : ℚ) / 12
def fracC := -1 - (9 : ℚ) / 36
def fracD := -1 - (5 : ℚ) / 15
def fracE := -1 - (25 : ℚ) / 100

theorem option_D_not_equal :
  fracD ≠ frac1 := 
sorry

end option_D_not_equal_l895_89526


namespace shorter_side_length_l895_89527

theorem shorter_side_length (a b : ℕ) (h1 : 2 * a + 2 * b = 42) (h2 : a * b = 108) : b = 9 :=
by
  sorry

end shorter_side_length_l895_89527


namespace middle_angle_of_triangle_l895_89575

theorem middle_angle_of_triangle (α β γ : ℝ) 
  (h1 : 0 < β) (h2 : β < 90) 
  (h3 : α ≤ β) (h4 : β ≤ γ) 
  (h5 : α + β + γ = 180) :
  True :=
by
  -- Proof would go here
  sorry

end middle_angle_of_triangle_l895_89575


namespace third_player_matches_l895_89587

theorem third_player_matches (first_player second_player third_player : ℕ) (h1 : first_player = 10) (h2 : second_player = 21) :
  third_player = 11 :=
by
  sorry

end third_player_matches_l895_89587


namespace geometric_series_sum_l895_89561

theorem geometric_series_sum :
  let a := (1 : ℝ) / 5
  let r := -(1 : ℝ) / 5
  let n := 5
  let S_n := (a * (1 - r ^ n)) / (1 - r)
  S_n = 521 / 3125 := by
  sorry

end geometric_series_sum_l895_89561


namespace cars_given_by_mum_and_dad_l895_89576

-- Define the conditions given in the problem
def initial_cars : ℕ := 150
def final_cars : ℕ := 196
def cars_by_auntie : ℕ := 6
def cars_more_than_uncle : ℕ := 1
def cars_given_by_family (uncle : ℕ) (grandpa : ℕ) (auntie : ℕ) : ℕ :=
  uncle + grandpa + auntie

-- Prove the required statement
theorem cars_given_by_mum_and_dad :
  ∃ (uncle grandpa : ℕ), grandpa = 2 * uncle ∧ auntie = uncle + cars_more_than_uncle ∧ 
    auntie = cars_by_auntie ∧
    final_cars - initial_cars - cars_given_by_family uncle grandpa auntie = 25 :=
by
  -- Placeholder for the actual proof
  sorry

end cars_given_by_mum_and_dad_l895_89576


namespace OReilly_triple_8_49_x_l895_89571

def is_OReilly_triple (a b x : ℕ) : Prop :=
  (a : ℝ)^(1/3) + (b : ℝ)^(1/2) = x

theorem OReilly_triple_8_49_x (x : ℕ) (h : is_OReilly_triple 8 49 x) : x = 9 := by
  sorry

end OReilly_triple_8_49_x_l895_89571


namespace part1_part2_l895_89518

variable (α : ℝ)

theorem part1 (h : Real.tan α = 2) : (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 := 
by
  sorry

theorem part2 (h : Real.tan α = 2) : Real.sin α ^ 2 + Real.sin (2 * α) = 8 / 5 :=
by
  sorry

end part1_part2_l895_89518


namespace john_new_salary_after_raise_l895_89553

theorem john_new_salary_after_raise (original_salary : ℝ) (percentage_increase : ℝ) (h1 : original_salary = 60) (h2 : percentage_increase = 0.8333333333333334) : 
  original_salary * (1 + percentage_increase) = 110 := 
sorry

end john_new_salary_after_raise_l895_89553


namespace clive_change_l895_89533

theorem clive_change (total_money : ℝ) (num_olives_needed : ℕ) (olives_per_jar : ℕ) (cost_per_jar : ℝ)
  (h1 : total_money = 10)
  (h2 : num_olives_needed = 80)
  (h3 : olives_per_jar = 20)
  (h4 : cost_per_jar = 1.5) : total_money - (num_olives_needed / olives_per_jar) * cost_per_jar = 4 := by
  sorry

end clive_change_l895_89533


namespace solve_equation_l895_89511

theorem solve_equation (x : ℝ) (h : x ≠ 1) (h_eq : x / (x - 1) = (x - 3) / (2 * x - 2)) : x = -3 :=
by
  sorry

end solve_equation_l895_89511


namespace slope_of_BC_l895_89539

theorem slope_of_BC
  (h₁ : ∀ x y : ℝ, (x^2 / 8) + (y^2 / 2) = 1)
  (h₂ : ∀ A : ℝ × ℝ, A = (2, 1))
  (h₃ : ∀ k₁ k₂ : ℝ, k₁ + k₂ = 0) :
  ∃ k : ℝ, k = 1 / 2 :=
by
  sorry

end slope_of_BC_l895_89539


namespace inequality_AM_GM_HM_l895_89588

variable {x y k : ℝ}

-- Define the problem conditions
def is_positive (a : ℝ) : Prop := a > 0
def is_unequal (a b : ℝ) : Prop := a ≠ b
def positive_constant_lessthan_two (c : ℝ) : Prop := c > 0 ∧ c < 2

-- State the theorem to be proven
theorem inequality_AM_GM_HM (h₁ : is_positive x) 
                             (h₂ : is_positive y) 
                             (h₃ : is_unequal x y) 
                             (h₄ : positive_constant_lessthan_two k) :
  ( ( ( (x + y) / 2 )^k > ( (x * y)^(1/2) )^k ) ∧ 
    ( ( (x * y)^(1/2) )^k > ( ( 2 * x * y ) / ( x + y ) )^k ) ) :=
by
  sorry

end inequality_AM_GM_HM_l895_89588


namespace sufficient_but_not_necessary_l895_89504

variable (x : ℝ)

theorem sufficient_but_not_necessary : (x = 1) → (x^3 = x) ∧ (∀ y, y^3 = y → y = 1 → x ≠ y) :=
by
  sorry

end sufficient_but_not_necessary_l895_89504


namespace weight_of_B_l895_89540

theorem weight_of_B (A B C : ℕ) (h1 : A + B + C = 90) (h2 : A + B = 50) (h3 : B + C = 56) : B = 16 := 
sorry

end weight_of_B_l895_89540


namespace field_area_is_13_point854_hectares_l895_89578

noncomputable def area_of_field_in_hectares (cost_fencing: ℝ) (rate_per_meter: ℝ): ℝ :=
  let length_of_fence := cost_fencing / rate_per_meter
  let radius := length_of_fence / (2 * Real.pi)
  let area_in_square_meters := Real.pi * (radius * radius)
  area_in_square_meters / 10000

theorem field_area_is_13_point854_hectares :
  area_of_field_in_hectares 6202.75 4.70 = 13.854 :=
by
  sorry

end field_area_is_13_point854_hectares_l895_89578


namespace dispatch_3_male_2_female_dispatch_at_least_2_male_l895_89562

-- Define the number of male and female drivers
def male_drivers : ℕ := 5
def female_drivers : ℕ := 4
def total_drivers_needed : ℕ := 5

-- Define the combination formula (binomial coefficient)
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- First part of the problem
theorem dispatch_3_male_2_female : 
  combination male_drivers 3 * combination female_drivers 2 = 60 :=
by sorry

-- Second part of the problem
theorem dispatch_at_least_2_male : 
  combination male_drivers 2 * combination female_drivers 3 + 
  combination male_drivers 3 * combination female_drivers 2 + 
  combination male_drivers 4 * combination female_drivers 1 + 
  combination male_drivers 5 * combination female_drivers 0 = 121 :=
by sorry

end dispatch_3_male_2_female_dispatch_at_least_2_male_l895_89562


namespace additional_toothpicks_needed_l895_89550

def three_step_toothpicks := 18
def four_step_toothpicks := 26

theorem additional_toothpicks_needed : 
  (∃ (f : ℕ → ℕ), f 3 = three_step_toothpicks ∧ f 4 = four_step_toothpicks ∧ (f 6 - f 4) = 22) :=
by {
  -- Assume f is a function that gives the number of toothpicks for a n-step staircase
  sorry
}

end additional_toothpicks_needed_l895_89550


namespace count_solutions_absolute_value_l895_89573

theorem count_solutions_absolute_value (x : ℤ) : 
  (|4 * x + 2| ≤ 10) ↔ (x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2) :=
by sorry

end count_solutions_absolute_value_l895_89573


namespace friend_cutoff_fraction_l895_89559

-- Definitions based on problem conditions
def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def days_biking : ℕ := 1
def days_bus : ℕ := 3
def days_friend : ℕ := 1
def total_weekly_commuting_time : ℕ := 160

-- Lean theorem statement
theorem friend_cutoff_fraction (F : ℕ) (hF : days_biking * biking_time + days_bus * bus_time + days_friend * F = total_weekly_commuting_time) :
  (biking_time - F) / biking_time = 2 / 3 :=
by
  sorry

end friend_cutoff_fraction_l895_89559


namespace sports_club_members_l895_89564

theorem sports_club_members (N B T : ℕ) (h_total : N = 30) (h_badminton : B = 18) (h_tennis : T = 19) (h_neither : N - (B + T - 9) = 2) : B + T - 9 = 28 :=
by
  sorry

end sports_club_members_l895_89564


namespace g_sum_zero_l895_89547

def g (x : ℝ) : ℝ := x^2 - 2013 * x

theorem g_sum_zero (a b : ℝ) (h₁ : g a = g b) (h₂ : a ≠ b) : g (a + b) = 0 :=
sorry

end g_sum_zero_l895_89547


namespace stratified_sampling_number_of_boys_stratified_sampling_probability_of_boy_l895_89560

theorem stratified_sampling_number_of_boys (total_students : Nat) (num_girls : Nat) (selected_students : Nat)
  (h1 : total_students = 125) (h2 : num_girls = 50) (h3 : selected_students = 25) :
  (total_students - num_girls) * selected_students / total_students = 15 :=
  sorry

theorem stratified_sampling_probability_of_boy (total_students : Nat) (selected_students : Nat)
  (h1 : total_students = 125) (h2 : selected_students = 25) :
  selected_students / total_students = 1 / 5 :=
  sorry

end stratified_sampling_number_of_boys_stratified_sampling_probability_of_boy_l895_89560


namespace painters_complete_three_rooms_in_three_hours_l895_89524

theorem painters_complete_three_rooms_in_three_hours :
  ∃ P, (∀ (P : ℕ), (P * 3) = 3) ∧ (9 * 9 = 27) → P = 3 := by
  sorry

end painters_complete_three_rooms_in_three_hours_l895_89524


namespace first_part_length_l895_89517

def total_length : ℝ := 74.5
def part_two : ℝ := 21.5
def part_three : ℝ := 21.5
def part_four : ℝ := 16

theorem first_part_length :
  total_length - (part_two + part_three + part_four) = 15.5 :=
by
  sorry

end first_part_length_l895_89517


namespace coin_toss_tails_count_l895_89574

theorem coin_toss_tails_count (flips : ℕ) (frequency_heads : ℝ) (h_flips : flips = 20) (h_frequency_heads : frequency_heads = 0.45) : 
  (20 : ℝ) * (1 - 0.45) = 11 := 
by
  sorry

end coin_toss_tails_count_l895_89574


namespace least_possible_z_minus_x_l895_89522

theorem least_possible_z_minus_x (x y z : ℤ) (h₁ : x < y) (h₂ : y < z) (h₃ : y - x > 11) 
  (h₄ : Even x) (h₅ : Odd y) (h₆ : Odd z) : z - x = 15 :=
sorry

end least_possible_z_minus_x_l895_89522


namespace mangoes_combined_l895_89594

variable (Alexis Dilan Ashley : ℕ)

theorem mangoes_combined :
  (Alexis = 60) → (Alexis = 4 * (Dilan + Ashley)) → (Alexis + Dilan + Ashley = 75) := 
by
  intros h₁ h₂
  sorry

end mangoes_combined_l895_89594


namespace clock_angle_at_7_oclock_l895_89510

theorem clock_angle_at_7_oclock : 
  ∀ (hour_angle minute_angle : ℝ), 
    (12 : ℝ) * (30 : ℝ) = 360 →
    (7 : ℝ) * (30 : ℝ) = 210 →
    (210 : ℝ) > 180 →
    (360 : ℝ) - (210 : ℝ) = 150 →
    hour_angle = 7 * 30 →
    minute_angle = 0 →
    min (abs (hour_angle - minute_angle)) (abs ((360 - hour_angle) - minute_angle)) = 150 := by
  sorry

end clock_angle_at_7_oclock_l895_89510


namespace students_selecting_water_l895_89581

-- Definitions of percentages and given values.
def p : ℝ := 0.7
def q : ℝ := 0.1
def n : ℕ := 140

-- The Lean statement to prove the number of students who selected water.
theorem students_selecting_water (p_eq : p = 0.7) (q_eq : q = 0.1) (n_eq : n = 140) :
  ∃ w : ℕ, w = (q / p) * n ∧ w = 20 :=
by sorry

end students_selecting_water_l895_89581


namespace team_t_speed_l895_89545

theorem team_t_speed (v t : ℝ) (h1 : 300 = v * t) (h2 : 300 = (v + 5) * (t - 3)) : v = 20 :=
by 
  sorry

end team_t_speed_l895_89545


namespace min_value_343_l895_89595

noncomputable def min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c)

theorem min_value_343 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ha hb hc = 343 :=
sorry

end min_value_343_l895_89595


namespace biggest_number_in_ratio_l895_89570

theorem biggest_number_in_ratio (x : ℕ) (h_sum : 2 * x + 3 * x + 4 * x + 5 * x = 1344) : 5 * x = 480 := 
by
  sorry

end biggest_number_in_ratio_l895_89570


namespace dorchester_puppies_washed_l895_89514

theorem dorchester_puppies_washed
  (total_earnings : ℝ)
  (daily_pay : ℝ)
  (earnings_per_puppy : ℝ)
  (p : ℝ)
  (h1 : total_earnings = 76)
  (h2 : daily_pay = 40)
  (h3 : earnings_per_puppy = 2.25)
  (hp : (total_earnings - daily_pay) / earnings_per_puppy = p) :
  p = 16 := sorry

end dorchester_puppies_washed_l895_89514


namespace find_principal_l895_89513

variable (P : ℝ) (r : ℝ) (t : ℕ) (CI : ℝ) (SI : ℝ)

-- Define simple and compound interest
def simple_interest (P r : ℝ) (t : ℕ) : ℝ := P * r * t
def compound_interest (P r : ℝ) (t : ℕ) : ℝ := P * (1 + r)^t - P

-- Given conditions
axiom H1 : r = 0.05
axiom H2 : t = 2
axiom H3 : compound_interest P r t - simple_interest P r t = 18

-- The principal sum is 7200
theorem find_principal : P = 7200 := 
by sorry

end find_principal_l895_89513


namespace fair_attendance_l895_89530

theorem fair_attendance (x y z : ℕ) 
    (h1 : y = 2 * x)
    (h2 : z = y - 200)
    (h3 : x + y + z = 2800) : x = 600 := by
  sorry

end fair_attendance_l895_89530


namespace alpha_plus_beta_l895_89529

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem alpha_plus_beta (α β : ℝ) (hα : 0 ≤ α) (hαβ : α < Real.pi) (hβ : 0 ≤ β) (hββ : β < Real.pi)
  (hα_neq_β : α ≠ β) (hf_α : f α = 1 / 2) (hf_β : f β = 1 / 2) : α + β = (7 * Real.pi) / 6 :=
by
  sorry

end alpha_plus_beta_l895_89529


namespace ordered_pairs_squares_diff_150_l895_89549

theorem ordered_pairs_squares_diff_150 (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn : m ≥ n) (h_diff : m^2 - n^2 = 150) : false :=
by {
    sorry
}

end ordered_pairs_squares_diff_150_l895_89549


namespace complex_quadrant_l895_89546

theorem complex_quadrant (θ : ℝ) (hθ : θ ∈ Set.Ioo (3/4 * Real.pi) (5/4 * Real.pi)) :
  let z := Complex.mk (Real.cos θ + Real.sin θ) (Real.sin θ - Real.cos θ)
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end complex_quadrant_l895_89546


namespace simplify_expression_l895_89597

variable (a : ℤ)

theorem simplify_expression : (-2 * a) ^ 3 * a ^ 3 + (-3 * a ^ 3) ^ 2 = a ^ 6 :=
by sorry

end simplify_expression_l895_89597


namespace ratio_problem_l895_89565

variable {a b c d : ℚ}

theorem ratio_problem (h₁ : a / b = 5) (h₂ : c / b = 3) (h₃ : c / d = 2) :
  d / a = 3 / 10 :=
sorry

end ratio_problem_l895_89565


namespace sapling_height_relationship_l895_89580

-- Definition to state the conditions
def initial_height : ℕ := 100
def growth_per_year : ℕ := 50
def height_after_years (years : ℕ) : ℕ := initial_height + growth_per_year * years

-- The theorem statement that should be proved
theorem sapling_height_relationship (x : ℕ) : height_after_years x = 50 * x + 100 := 
by
  sorry

end sapling_height_relationship_l895_89580


namespace polygon_angles_change_l895_89525

theorem polygon_angles_change (n : ℕ) :
  let initial_sum_interior := (n - 2) * 180
  let initial_sum_exterior := 360
  let new_sum_interior := (n + 2 - 2) * 180
  let new_sum_exterior := 360
  new_sum_exterior = initial_sum_exterior ∧ new_sum_interior - initial_sum_interior = 360 :=
by
  sorry

end polygon_angles_change_l895_89525


namespace find_value_l895_89521

theorem find_value (N : ℝ) (h : 1.20 * N = 6000) : 0.20 * N = 1000 :=
sorry

end find_value_l895_89521


namespace solve_2x2_minus1_eq_3x_l895_89585
noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let root1 := (-b + sqrt_discriminant) / (2 * a)
  let root2 := (-b - sqrt_discriminant) / (2 * a)
  (root1, root2)

theorem solve_2x2_minus1_eq_3x :
  solve_quadratic 2 (-3) (-1) = ( (3 + Real.sqrt 17) / 4, (3 - Real.sqrt 17) / 4 ) :=
by
  let roots := solve_quadratic 2 (-3) (-1)
  have : roots = ( (3 + Real.sqrt 17) / 4, (3 - Real.sqrt 17) / 4) := by sorry
  exact this

end solve_2x2_minus1_eq_3x_l895_89585


namespace growth_rate_correct_max_avg_visitors_correct_l895_89596

-- Define the conditions from part 1
def visitors_march : ℕ := 80000
def visitors_may : ℕ := 125000

-- Define the monthly average growth rate
def monthly_avg_growth_rate (x : ℝ) : Prop :=
(1 + x)^2 = (visitors_may / visitors_march : ℝ)

-- Define the condition for June
def visitors_june_1_10 : ℕ := 66250
def max_avg_visitors_per_day (y : ℝ) : Prop :=
6.625 + 20 * y ≤ 15.625

-- Prove the monthly growth rate
theorem growth_rate_correct : ∃ x : ℝ, monthly_avg_growth_rate x ∧ x = 0.25 := sorry

-- Prove the max average visitors per day in June
theorem max_avg_visitors_correct : ∃ y : ℝ, max_avg_visitors_per_day y ∧ y = 0.45 := sorry

end growth_rate_correct_max_avg_visitors_correct_l895_89596


namespace percentage_disliked_by_both_l895_89598

theorem percentage_disliked_by_both 
  (total_comic_books : ℕ) 
  (percentage_females_like : ℕ) 
  (comic_books_males_like : ℕ) :
  total_comic_books = 300 →
  percentage_females_like = 30 →
  comic_books_males_like = 120 →
  ((total_comic_books - (total_comic_books * percentage_females_like / 100) - comic_books_males_like) * 100 / total_comic_books) = 30 :=
by
  intros h1 h2 h3
  sorry

end percentage_disliked_by_both_l895_89598


namespace correct_factorization_l895_89532

theorem correct_factorization (a x m : ℝ) :
  (ax^2 - a = a * (x^2 - 1)) ∨
  (m^3 + m = m * (m^2 + 1)) ∨
  (x^2 + 2*x - 3 = x*(x+2) - 3) ∨
  (x^2 + 2*x - 3 = (x-3)*(x+1)) :=
by sorry

end correct_factorization_l895_89532


namespace total_rooms_l895_89520

-- Definitions for the problem conditions
variables (x y : ℕ)

-- Given conditions
def condition1 : Prop := x = 8
def condition2 : Prop := 2 * x + 3 * y = 31

-- The theorem to prove
theorem total_rooms (h1 : condition1 x) (h2 : condition2 x y) : x + y = 13 :=
by sorry

end total_rooms_l895_89520


namespace Dave_guitar_strings_replacement_l895_89506

theorem Dave_guitar_strings_replacement :
  (2 * 6 * 12) = 144 := by
  sorry

end Dave_guitar_strings_replacement_l895_89506


namespace a_eq_3_suff_not_nec_l895_89568

theorem a_eq_3_suff_not_nec (a : ℝ) : (a = 3 → a^2 = 9) ∧ (a^2 = 9 → ∃ b : ℝ, b = a ∧ (b = 3 ∨ b = -3)) :=
by
  sorry

end a_eq_3_suff_not_nec_l895_89568


namespace problem1_l895_89531

variable {x : ℝ} {b c : ℝ}

theorem problem1 (hb : b = 9) (hc : c = -11) :
  b + c = -2 := 
by
  simp [hb, hc]
  sorry

end problem1_l895_89531


namespace opposite_of_point_one_l895_89542

theorem opposite_of_point_one : ∃ x : ℝ, 0.1 + x = 0 ∧ x = -0.1 :=
by
  sorry

end opposite_of_point_one_l895_89542


namespace movie_watching_l895_89583

theorem movie_watching :
  let total_duration := 120 
  let watched1 := 35
  let watched2 := 20
  let watched3 := 15
  let total_watched := watched1 + watched2 + watched3
  total_duration - total_watched = 50 :=
by
  sorry

end movie_watching_l895_89583


namespace cos_function_max_value_l895_89535

theorem cos_function_max_value (k : ℤ) : (2 * Real.cos (2 * k * Real.pi) - 1) = 1 :=
by
  -- Proof not included
  sorry

end cos_function_max_value_l895_89535


namespace find_a12_a14_l895_89548

noncomputable def S (n : ℕ) (a_n : ℕ → ℝ) (b : ℝ) : ℝ := a_n n ^ 2 + b * n

noncomputable def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ (a1 : ℝ) (c : ℝ), ∀ n : ℕ, a_n n = a1 + (n - 1) * c

theorem find_a12_a14
  (a_n : ℕ → ℝ)
  (b : ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, S n = a_n n ^ 2 + b * n)
  (h2 : S 25 = 100)
  (h3 : is_arithmetic_sequence a_n) :
  a_n 12 + a_n 14 = 5 :=
sorry

end find_a12_a14_l895_89548


namespace halfway_between_fractions_l895_89593

theorem halfway_between_fractions : 
  (2:ℚ) / 9 + (5 / 12) / 2 = 23 / 72 := 
sorry

end halfway_between_fractions_l895_89593


namespace ratio_of_oranges_to_limes_l895_89508

-- Constants and Definitions
def initial_fruits : ℕ := 150
def half_fruits : ℕ := 75
def oranges : ℕ := 50
def limes : ℕ := half_fruits - oranges
def ratio_oranges_limes : ℕ × ℕ := (oranges / Nat.gcd oranges limes, limes / Nat.gcd oranges limes)

-- Theorem Statement
theorem ratio_of_oranges_to_limes : ratio_oranges_limes = (2, 1) := by
  sorry

end ratio_of_oranges_to_limes_l895_89508


namespace smallest_base10_integer_l895_89509

theorem smallest_base10_integer :
  ∃ (n : ℕ) (X : ℕ) (Y : ℕ), 
  (0 ≤ X ∧ X < 6) ∧ (0 ≤ Y ∧ Y < 8) ∧ 
  (n = 7 * X) ∧ (n = 9 * Y) ∧ n = 63 :=
by
  sorry

end smallest_base10_integer_l895_89509
