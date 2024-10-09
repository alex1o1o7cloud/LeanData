import Mathlib

namespace fraction_pow_four_result_l1104_110486

theorem fraction_pow_four_result (x : ℚ) (h : x = 1 / 4) : x ^ 4 = 390625 / 100000000 :=
by sorry

end fraction_pow_four_result_l1104_110486


namespace months_after_withdrawal_and_advance_eq_eight_l1104_110446

-- Define initial conditions
def initial_investment_A : ℝ := 3000
def initial_investment_B : ℝ := 4000
def withdrawal_A : ℝ := 1000
def advancement_B : ℝ := 1000
def total_profit : ℝ := 630
def share_A : ℝ := 240
def share_B : ℝ := total_profit - share_A

-- Define the main proof problem
theorem months_after_withdrawal_and_advance_eq_eight
  (initial_investment_A : ℝ) (initial_investment_B : ℝ)
  (withdrawal_A : ℝ) (advancement_B : ℝ)
  (total_profit : ℝ) (share_A : ℝ) (share_B : ℝ) : 
  ∃ x : ℝ, 
  (3000 * x + 2000 * (12 - x)) / (4000 * x + 5000 * (12 - x)) = 240 / 390 ∧
  x = 8 :=
sorry

end months_after_withdrawal_and_advance_eq_eight_l1104_110446


namespace peter_vacation_saving_l1104_110479

theorem peter_vacation_saving :
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  months_needed = 3 :=
by
  -- definitions
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  -- proof
  sorry

end peter_vacation_saving_l1104_110479


namespace matrix_solution_l1104_110444

variable {x : ℝ}

theorem matrix_solution (x: ℝ) :
  let M := (3*x) * (2*x + 1) - (1) * (2*x)
  M = 5 → (x = 5/6) ∨ (x = -1) :=
by
  sorry

end matrix_solution_l1104_110444


namespace area_of_square_with_diagonal_two_l1104_110419

theorem area_of_square_with_diagonal_two {a d : ℝ} (h : d = 2) (h' : d = a * Real.sqrt 2) : a^2 = 2 := 
by
  sorry

end area_of_square_with_diagonal_two_l1104_110419


namespace tap_C_fills_in_6_l1104_110459

-- Definitions for the rates at which taps fill the tank
def rate_A := 1/10
def rate_B := 1/15
def rate_combined := 1/3

-- Proof problem: Given the conditions, prove that the third tap fills the tank in 6 hours
theorem tap_C_fills_in_6 (rate_A rate_B rate_combined : ℚ) (h : rate_A + rate_B + 1/x = rate_combined) : x = 6 :=
sorry

end tap_C_fills_in_6_l1104_110459


namespace gasoline_reduction_l1104_110454

theorem gasoline_reduction (P Q : ℝ) :
  let new_price := 1.25 * P
  let new_budget := 1.10 * (P * Q)
  let new_quantity := new_budget / new_price
  let percent_reduction := 1 - (new_quantity / Q)
  percent_reduction = 0.12 :=
by
  sorry

end gasoline_reduction_l1104_110454


namespace carrie_jellybeans_l1104_110451

def volume (a : ℕ) : ℕ := a * a * a

def bert_box_volume : ℕ := 216

def carrie_factor : ℕ := 3

def count_error_factor : ℝ := 1.10

noncomputable def jellybeans_carrie (bert_box_volume carrie_factor count_error_factor : ℝ) : ℝ :=
  count_error_factor * (carrie_factor ^ 3 * bert_box_volume)

theorem carrie_jellybeans (bert_box_volume := 216) (carrie_factor := 3) (count_error_factor := 1.10) :
  jellybeans_carrie bert_box_volume carrie_factor count_error_factor = 6415 :=
sorry

end carrie_jellybeans_l1104_110451


namespace rational_inequality_solution_l1104_110424

theorem rational_inequality_solution {x : ℝ} : (4 / (x + 1) ≤ 1) → (x ∈ Set.Iic (-1) ∪ Set.Ici 3) :=
by 
  sorry

end rational_inequality_solution_l1104_110424


namespace jason_money_l1104_110432

theorem jason_money (fred_money_before : ℕ) (jason_money_before : ℕ)
  (fred_money_after : ℕ) (total_earned : ℕ) :
  fred_money_before = 111 →
  jason_money_before = 40 →
  fred_money_after = 115 →
  total_earned = 4 →
  jason_money_before = 40 := by
  intros h1 h2 h3 h4
  sorry

end jason_money_l1104_110432


namespace calculate_si_l1104_110487

section SimpleInterest

def Principal : ℝ := 10000
def Rate : ℝ := 0.04
def Time : ℝ := 1
def SimpleInterest : ℝ := Principal * Rate * Time

theorem calculate_si : SimpleInterest = 400 := by
  -- Proof goes here.
  sorry

end SimpleInterest

end calculate_si_l1104_110487


namespace ryan_recruit_people_l1104_110413

noncomputable def total_amount_needed : ℕ := 1000
noncomputable def amount_already_have : ℕ := 200
noncomputable def average_funding_per_person : ℕ := 10
noncomputable def additional_funding_needed : ℕ := total_amount_needed - amount_already_have
noncomputable def number_of_people_recruit : ℕ := additional_funding_needed / average_funding_per_person

theorem ryan_recruit_people : number_of_people_recruit = 80 := by
  sorry

end ryan_recruit_people_l1104_110413


namespace fraction_difference_l1104_110435

theorem fraction_difference : (18 / 42) - (3 / 8) = 3 / 56 := 
by
  sorry

end fraction_difference_l1104_110435


namespace rate_of_interest_l1104_110478

-- Given conditions
def P : ℝ := 1500
def SI : ℝ := 735
def r : ℝ := 7
def t := r  -- The time period in years is equal to the rate of interest

-- The formula for simple interest and the goal
theorem rate_of_interest : SI = P * r * t / 100 ↔ r = 7 := 
by
  -- We will use the given conditions and check if they support r = 7
  sorry

end rate_of_interest_l1104_110478


namespace find_p_plus_s_l1104_110442

noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem find_p_plus_s (p q r s : ℝ) (h : p * q * r * s ≠ 0) 
  (hg : ∀ x : ℝ, g p q r s (g p q r s x) = x) : p + s = 0 := 
by 
  sorry

end find_p_plus_s_l1104_110442


namespace trapezoid_height_l1104_110411

-- We are given the lengths of the sides of the trapezoid
def length_parallel1 : ℝ := 25
def length_parallel2 : ℝ := 4
def length_non_parallel1 : ℝ := 20
def length_non_parallel2 : ℝ := 13

-- We need to prove that the height of the trapezoid is 12 cm
theorem trapezoid_height (h : ℝ) :
  (h^2 + (20^2 - 16^2) = 144 ∧ h = 12) :=
sorry

end trapezoid_height_l1104_110411


namespace televisions_selection_ways_l1104_110422

noncomputable def combination (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

theorem televisions_selection_ways :
  let TypeA := 4
  let TypeB := 5
  let choosen := 3
  (∃ (n m : ℕ), n + m = choosen ∧ 1 ≤ n ∧ n ≤ TypeA ∧ 1 ≤ m ∧ m ≤ TypeB ∧
    combination TypeA n * combination TypeB m = 70) :=
by
  sorry

end televisions_selection_ways_l1104_110422


namespace integer_triplets_prime_l1104_110416

theorem integer_triplets_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ sol : ℕ, ((∃ (x y z : ℤ), (3 * x + y + z) * (x + 2 * y + z) * (x + y + z) = p) ∧
  if p = 2 then sol = 4 else sol = 12) :=
by
  sorry

end integer_triplets_prime_l1104_110416


namespace sarah_jamie_julien_ratio_l1104_110436

theorem sarah_jamie_julien_ratio (S J : ℕ) (R : ℝ) :
  -- Conditions
  (J = S + 20) ∧
  (S = R * 50) ∧
  (7 * (J + S + 50) = 1890) ∧
  -- Prove the ratio
  R = 2 := by
  sorry

end sarah_jamie_julien_ratio_l1104_110436


namespace olympiad_permutations_l1104_110475

theorem olympiad_permutations : 
  let total_permutations := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2) 
  let invalid_permutations := 5 * (Nat.factorial 4 / Nat.factorial 2)
  total_permutations - invalid_permutations = 90660 :=
by
  let total_permutations : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2)
  let invalid_permutations : ℕ := 5 * (Nat.factorial 4 / Nat.factorial 2)
  show total_permutations - invalid_permutations = 90660
  sorry

end olympiad_permutations_l1104_110475


namespace film_radius_l1104_110420

theorem film_radius 
  (thickness : ℝ)
  (container_volume : ℝ)
  (r : ℝ)
  (H1 : thickness = 0.25)
  (H2 : container_volume = 128) :
  r = Real.sqrt (512 / Real.pi) :=
by
  -- Placeholder for proof
  sorry

end film_radius_l1104_110420


namespace find_a_odd_function_l1104_110461

theorem find_a_odd_function (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, 0 < x → f x = 1 + a^x)
  (h3 : 0 < a)
  (h4 : a ≠ 1)
  (h5 : f (-1) = -3 / 2) :
  a = 1 / 2 :=
by
  sorry

end find_a_odd_function_l1104_110461


namespace factor_expression_l1104_110490

variable (x y : ℝ)

theorem factor_expression : 3 * x^3 - 6 * x^2 * y + 3 * x * y^2 = 3 * x * (x - y)^2 := 
by 
  sorry

end factor_expression_l1104_110490


namespace inequality_solution_set_l1104_110474

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (3 / 4 ≤ x ∧ x < 2) :=
by sorry

end inequality_solution_set_l1104_110474


namespace inequality_additive_l1104_110471

variable {a b c d : ℝ}

theorem inequality_additive (h1 : a > b) (h2 : c > d) : a + c > b + d :=
by
  sorry

end inequality_additive_l1104_110471


namespace sum_of_first_n_odd_integers_eq_169_l1104_110455

theorem sum_of_first_n_odd_integers_eq_169 (n : ℕ) 
  (h : n^2 = 169) : n = 13 :=
by sorry

end sum_of_first_n_odd_integers_eq_169_l1104_110455


namespace compare_probabilities_l1104_110428

-- Definitions
variables (M N: ℕ) (m n: ℕ)
  
-- Conditions
def condition_m_millionaire : Prop := m > 10^6
def condition_n_nonmillionaire : Prop := n ≤ 10^6

-- Probabilities
noncomputable def P_A : ℚ := (M:ℚ) / (M + (n:ℚ)/(m:ℚ) * N)
noncomputable def P_B : ℚ := (M:ℚ) / (M + N)

-- Theorem statement
theorem compare_probabilities
  (hM : M > 0) (hN : N > 0)
  (h_m_millionaire : condition_m_millionaire m)
  (h_n_nonmillionaire : condition_n_nonmillionaire n) :
  P_A M N m n > P_B M N := sorry

end compare_probabilities_l1104_110428


namespace truck_travel_due_east_distance_l1104_110427

theorem truck_travel_due_east_distance :
  ∀ (x : ℕ),
  (20 + 20)^2 + x^2 = 50^2 → x = 30 :=
by
  intro x
  sorry -- proof will be here

end truck_travel_due_east_distance_l1104_110427


namespace two_layers_area_zero_l1104_110457

theorem two_layers_area_zero (A X Y Z : ℕ)
  (h1 : A = 212)
  (h2 : X + Y + Z = 140)
  (h3 : Y + Z = 24)
  (h4 : Z = 24) : Y = 0 :=
by
  sorry

end two_layers_area_zero_l1104_110457


namespace gravitational_force_on_space_station_l1104_110408

-- Define the problem conditions and gravitational relationship
def gravitational_force_proportionality (f d : ℝ) : Prop :=
  ∃ k : ℝ, f * d^2 = k

-- Given conditions
def earth_surface_distance : ℝ := 6371
def space_station_distance : ℝ := 100000
def surface_gravitational_force : ℝ := 980
def proportionality_constant : ℝ := surface_gravitational_force * earth_surface_distance^2

-- Statement of the proof problem
theorem gravitational_force_on_space_station :
  gravitational_force_proportionality surface_gravitational_force earth_surface_distance →
  ∃ f2 : ℝ, f2 = 3.977 ∧ gravitational_force_proportionality f2 space_station_distance :=
sorry

end gravitational_force_on_space_station_l1104_110408


namespace max_value_y_l1104_110476

/-- Given x < 0, the maximum value of y = (1 + x^2) / x is -2 -/
theorem max_value_y {x : ℝ} (h : x < 0) : ∃ y, y = 1 + x^2 / x ∧ y ≤ -2 :=
sorry

end max_value_y_l1104_110476


namespace find_g_at_4_l1104_110441

theorem find_g_at_4 (g : ℝ → ℝ) (h : ∀ x, 2 * g x + 3 * g (1 - x) = 4 * x^3 - x) : g 4 = 193.2 :=
sorry

end find_g_at_4_l1104_110441


namespace parabola_vertex_at_origin_axis_x_passing_point_parabola_vertex_at_origin_axis_y_distance_focus_l1104_110439

-- Define the first parabola proof problem
theorem parabola_vertex_at_origin_axis_x_passing_point :
  (∃ (m : ℝ), ∀ (x y : ℝ), y^2 = m * x ↔ (y, x) = (0, 0) ∨ (x = 6 ∧ y = -3)) → 
  ∃ m : ℝ, m = 1.5 ∧ (y^2 = m * x) :=
sorry

-- Define the second parabola proof problem
theorem parabola_vertex_at_origin_axis_y_distance_focus :
  (∃ (p : ℝ), ∀ (x y : ℝ), x^2 = 4 * p * y ↔ (y, x) = (0, 0) ∨ (p = 3)) → 
  ∃ q : ℝ, q = 12 ∧ (x^2 = q * y ∨ x^2 = -q * y) :=
sorry

end parabola_vertex_at_origin_axis_x_passing_point_parabola_vertex_at_origin_axis_y_distance_focus_l1104_110439


namespace congruence_solution_count_l1104_110418

theorem congruence_solution_count :
  ∀ y : ℕ, y < 150 → (y ≡ 20 + 110 [MOD 46]) → y = 38 ∨ y = 84 ∨ y = 130 :=
by
  intro y
  intro hy
  intro hcong
  sorry

end congruence_solution_count_l1104_110418


namespace not_algebraic_expression_C_l1104_110412

-- Define what it means for something to be an algebraic expression, as per given problem's conditions
def is_algebraic_expression (expr : String) : Prop :=
  expr = "A" ∨ expr = "B" ∨ expr = "D"
  
theorem not_algebraic_expression_C : ¬ (is_algebraic_expression "C") :=
by
  -- This is a placeholder; proof steps are not required per instructions
  sorry

end not_algebraic_expression_C_l1104_110412


namespace probability_ratio_l1104_110415

theorem probability_ratio (bins balls n1 n2 n3 n4 : Nat)
  (h_balls : balls = 18)
  (h_bins : bins = 4)
  (scenarioA : n1 = 6 ∧ n2 = 2 ∧ n3 = 5 ∧ n4 = 5)
  (scenarioB : n1 = 5 ∧ n2 = 5 ∧ n3 = 4 ∧ n4 = 4) :
  ((Nat.choose bins 1) * (Nat.choose (bins - 1) 1) * Nat.factorial balls /
  (Nat.factorial n1 * Nat.factorial n2 * Nat.factorial n3 * Nat.factorial n4)) /
  ((Nat.choose bins 2) * Nat.factorial balls /
  (Nat.factorial n1 * Nat.factorial n2 * Nat.factorial n3 * Nat.factorial n4)) = 10 / 3 :=
by
  sorry

end probability_ratio_l1104_110415


namespace true_proposition_l1104_110438

-- Definitions of propositions
def p := ∃ (x : ℝ), x - x + 1 ≥ 0
def q := ∀ (a b : ℝ), a^2 < b^2 → a < b

-- Theorem statement
theorem true_proposition : p ∧ ¬q :=
by
  sorry

end true_proposition_l1104_110438


namespace new_sphere_radius_l1104_110421

noncomputable def calculateVolume (R r : ℝ) : ℝ :=
  let originalSphereVolume := (4 / 3) * Real.pi * R^3
  let cylinderHeight := 2 * Real.sqrt (R^2 - r^2)
  let cylinderVolume := Real.pi * r^2 * cylinderHeight
  let capHeight := R - Real.sqrt (R^2 - r^2)
  let capVolume := (Real.pi * capHeight^2 * (3 * R - capHeight)) / 3
  let totalCapVolume := 2 * capVolume
  originalSphereVolume - cylinderVolume - totalCapVolume

theorem new_sphere_radius
  (R : ℝ) (r : ℝ) (h : ℝ) (new_sphere_radius : ℝ)
  (h_eq: h = 2 * Real.sqrt (R^2 - r^2))
  (new_sphere_volume_eq: calculateVolume R r = (4 / 3) * Real.pi * new_sphere_radius^3)
  : new_sphere_radius = 16 :=
sorry

end new_sphere_radius_l1104_110421


namespace sum_of_factors_of_30_is_72_l1104_110463

-- Condition: given the number 30
def number := 30

-- Define the positive factors of 30
def factors : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- Statement to prove the sum of the positive factors
theorem sum_of_factors_of_30_is_72 : (factors.sum) = 72 := 
by
  sorry

end sum_of_factors_of_30_is_72_l1104_110463


namespace find_y_l1104_110465

theorem find_y (x y: ℝ) (h1: x = 680) (h2: 0.25 * x = 0.20 * y - 30) : y = 1000 :=
by 
  sorry

end find_y_l1104_110465


namespace infinite_quadruples_inequality_quadruple_l1104_110426

theorem infinite_quadruples 
  (a p q r : ℤ) 
  (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (hp_div : p ∣ (a * q * r + 1))
  (hq_div : q ∣ (a * p * r + 1))
  (hr_div : r ∣ (a * p * q + 1)) :
  ∃ (a p q r : ℕ), 
    1 < p ∧ 1 < q ∧ 1 < r ∧
    p ∣ (a * q * r + 1) ∧
    q ∣ (a * p * r + 1) ∧
    r ∣ (a * p * q + 1) :=
sorry

theorem inequality_quadruple
  (a p q r : ℤ) 
  (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (hp_div : p ∣ (a * q * r + 1))
  (hq_div : q ∣ (a * p * r + 1))
  (hr_div : r ∣ (a * p * q + 1)) :
  a ≥ (p * q * r - 1) / (p * q + q * r + r * p) :=
sorry

end infinite_quadruples_inequality_quadruple_l1104_110426


namespace expression_simplifies_to_49_l1104_110431

theorem expression_simplifies_to_49 (x : ℝ) : 
  (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end expression_simplifies_to_49_l1104_110431


namespace income_in_scientific_notation_l1104_110493

theorem income_in_scientific_notation :
  10870 = 1.087 * 10^4 := 
sorry

end income_in_scientific_notation_l1104_110493


namespace x_intercept_of_perpendicular_line_l1104_110423

theorem x_intercept_of_perpendicular_line 
  (a : ℝ)
  (l1 : ℝ → ℝ → Prop)
  (l1_eq : ∀ x y, l1 x y ↔ (a+3)*x + y - 4 = 0)
  (l2 : ℝ → ℝ → Prop)
  (l2_eq : ∀ x y, l2 x y ↔ x + (a-1)*y + 4 = 0)
  (perpendicular : ∀ x y, l1 x y → l2 x y → (a+3)*(a-1) = -1) :
  (∃ x : ℝ, l1 x 0 ∧ x = 2) :=
sorry

end x_intercept_of_perpendicular_line_l1104_110423


namespace gcd_min_b_c_l1104_110495

theorem gcd_min_b_c (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : Nat.gcd a b = 294) (h2 : Nat.gcd a c = 1155) :
  Nat.gcd b c = 21 :=
sorry

end gcd_min_b_c_l1104_110495


namespace fraction_identity_l1104_110452

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l1104_110452


namespace Jessie_weight_l1104_110445

theorem Jessie_weight (c l w : ℝ) (hc : c = 27) (hl : l = 101) : c + l = w ↔ w = 128 := by
  sorry

end Jessie_weight_l1104_110445


namespace right_angle_triangle_sets_l1104_110407

def is_right_angle_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angle_triangle_sets :
  ¬ is_right_angle_triangle (2 / 3) 2 (5 / 4) :=
by {
  sorry
}

end right_angle_triangle_sets_l1104_110407


namespace Mitch_saved_amount_l1104_110469

theorem Mitch_saved_amount :
  let boat_cost_per_foot := 1500
  let license_and_registration := 500
  let docking_fees := 3 * 500
  let longest_boat_length := 12
  let total_license_and_fees := license_and_registration + docking_fees
  let total_boat_cost := boat_cost_per_foot * longest_boat_length
  let total_saved := total_boat_cost + total_license_and_fees
  total_saved = 20000 :=
by
  sorry

end Mitch_saved_amount_l1104_110469


namespace expression_value_l1104_110499

theorem expression_value (x : ℝ) (h : x = 3 + 5 / (2 + 5 / x)) : x = 5 :=
sorry

end expression_value_l1104_110499


namespace negative_exp_eq_l1104_110429

theorem negative_exp_eq :
  (-2 : ℤ)^3 = (-2 : ℤ)^3 := by
  sorry

end negative_exp_eq_l1104_110429


namespace xy_sum_correct_l1104_110489

theorem xy_sum_correct (x y : ℝ) 
  (h : (4 + 10 + 16 + 24) / 4 = (14 + x + y) / 3) : 
  x + y = 26.5 :=
by
  sorry

end xy_sum_correct_l1104_110489


namespace baseball_league_games_l1104_110403

theorem baseball_league_games
  (N M : ℕ)
  (hN_gt_2M : N > 2 * M)
  (hM_gt_4 : M > 4)
  (h_total_games : 4 * N + 5 * M = 94) :
  4 * N = 64 :=
by
  sorry

end baseball_league_games_l1104_110403


namespace regular_polygon_interior_angle_integer_l1104_110414

theorem regular_polygon_interior_angle_integer :
  ∃ l : List ℕ, l.length = 9 ∧ ∀ n ∈ l, 3 ≤ n ∧ n ≤ 15 ∧ (180 * (n - 2)) % n = 0 :=
by
  sorry

end regular_polygon_interior_angle_integer_l1104_110414


namespace sum_of_xy_l1104_110417

theorem sum_of_xy (x y : ℝ) (h1 : x + 3 * y = 12) (h2 : 3 * x + y = 8) : x + y = 5 := 
by
  sorry

end sum_of_xy_l1104_110417


namespace area_increase_by_40_percent_l1104_110401

theorem area_increase_by_40_percent (s : ℝ) : 
  let A1 := s^2 
  let new_side := 1.40 * s 
  let A2 := new_side^2 
  (A2 - A1) / A1 * 100 = 96 := 
by 
  sorry

end area_increase_by_40_percent_l1104_110401


namespace problem_inequality_minimum_value_l1104_110468

noncomputable def f (x y z : ℝ) : ℝ := 
  (3 * x^2 - x) / (1 + x^2) + 
  (3 * y^2 - y) / (1 + y^2) + 
  (3 * z^2 - z) / (1 + z^2)

theorem problem_inequality (x y z : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h_sum : x + y + z = 1) :
  f x y z ≥ 0 :=
sorry

theorem minimum_value (x y z : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h_sum : x + y + z = 1) :
  f x y z = 0 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
sorry

end problem_inequality_minimum_value_l1104_110468


namespace Mary_regular_hourly_rate_l1104_110406

theorem Mary_regular_hourly_rate (R : ℝ) (h1 : ∃ max_hours : ℝ, max_hours = 70)
  (h2 : ∀ hours: ℝ, hours ≤ 70 → (hours ≤ 20 → earnings = hours * R) ∧ (hours > 20 → earnings = 20 * R + (hours - 20) * 1.25 * R))
  (h3 : ∀ max_earning: ℝ, max_earning = 660)
  : R = 8 := 
sorry

end Mary_regular_hourly_rate_l1104_110406


namespace root_properties_of_polynomial_l1104_110460

variables {r s t : ℝ}

def polynomial (x : ℝ) : ℝ := 6 * x^3 + 4 * x^2 + 1500 * x + 3000

theorem root_properties_of_polynomial :
  (∀ x : ℝ, polynomial x = 0 → (x = r ∨ x = s ∨ x = t)) →
  (r + s + t = -2 / 3) →
  (r * s + r * t + s * t = 250) →
  (r * s * t = -500) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = -5992 / 27 :=
by
  sorry

end root_properties_of_polynomial_l1104_110460


namespace find_f79_l1104_110404

noncomputable def f : ℝ → ℝ :=
  sorry

axiom condition1 : ∀ x y : ℝ, f (x * y) = x * f y
axiom condition2 : f 1 = 25

theorem find_f79 : f 79 = 1975 :=
by
  sorry

end find_f79_l1104_110404


namespace gallons_needed_to_grandmas_house_l1104_110443

def car_fuel_efficiency : ℝ := 20
def distance_to_grandmas_house : ℝ := 100

theorem gallons_needed_to_grandmas_house : (distance_to_grandmas_house / car_fuel_efficiency) = 5 :=
by
  sorry

end gallons_needed_to_grandmas_house_l1104_110443


namespace sum_of_coordinates_of_B_l1104_110466

theorem sum_of_coordinates_of_B (x y : ℕ) (hM : (2 * 6 = x + 10) ∧ (2 * 8 = y + 8)) :
    x + y = 10 :=
sorry

end sum_of_coordinates_of_B_l1104_110466


namespace Xiaofang_English_score_l1104_110425

/-- Given the conditions about the average scores of Xiaofang's subjects:
  1. The average score for 4 subjects is 88.
  2. The average score for the first 2 subjects is 93.
  3. The average score for the last 3 subjects is 87.
Prove that Xiaofang's English test score is 95. -/
theorem Xiaofang_English_score
    (L M E S : ℝ)
    (h1 : (L + M + E + S) / 4 = 88)
    (h2 : (L + M) / 2 = 93)
    (h3 : (M + E + S) / 3 = 87) :
    E = 95 :=
by
  sorry

end Xiaofang_English_score_l1104_110425


namespace cube_difference_divisibility_l1104_110477

-- Given conditions
variables {m n : ℤ} (h1 : m % 2 = 1) (h2 : n % 2 = 1) (k : ℕ)

-- The equivalent statement to be proven
theorem cube_difference_divisibility (h1 : m % 2 = 1) (h2 : n % 2 = 1) : 
  (2^k ∣ m^3 - n^3) ↔ (2^k ∣ m - n) :=
sorry

end cube_difference_divisibility_l1104_110477


namespace way_to_cut_grid_l1104_110491

def grid_ways : ℕ := 17

def rectangles (size : ℕ × ℕ) (count : ℕ) := 
  size = (1, 2) ∧ count = 8

def square (size : ℕ × ℕ) (count : ℕ) := 
  size = (1, 1) ∧ count = 1

theorem way_to_cut_grid :
  (∃ ways : ℕ, ways = 10) ↔ 
  ∀ g ways, g = grid_ways → 
  (rectangles (1, 2) 8 ∧ square (1, 1) 1 → ways = 10) :=
by 
  sorry

end way_to_cut_grid_l1104_110491


namespace total_tickets_sold_correct_l1104_110470

theorem total_tickets_sold_correct :
  ∀ (A : ℕ), (21 * A + 15 * 327 = 8748) → (A + 327 = 509) :=
by
  intros A h
  sorry

end total_tickets_sold_correct_l1104_110470


namespace area_of_EFGH_l1104_110437

def short_side_length : ℕ := 4
def long_side_length : ℕ := short_side_length * 2
def number_of_rectangles : ℕ := 4
def larger_rectangle_length : ℕ := short_side_length
def larger_rectangle_width : ℕ := number_of_rectangles * long_side_length

theorem area_of_EFGH :
  (larger_rectangle_length * larger_rectangle_width) = 128 := 
  by
    sorry

end area_of_EFGH_l1104_110437


namespace diamond_problem_l1104_110488

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_problem : (diamond (diamond 1 2) 3) - (diamond 1 (diamond 2 3)) = -7 / 30 := by
  sorry

end diamond_problem_l1104_110488


namespace candies_remaining_l1104_110434

theorem candies_remaining 
    (red_candies : ℕ)
    (yellow_candies : ℕ)
    (blue_candies : ℕ)
    (yellow_condition : yellow_candies = 3 * red_candies - 20)
    (blue_condition : blue_candies = yellow_candies / 2)
    (initial_red_candies : red_candies = 40) :
    (red_candies + yellow_candies + blue_candies - yellow_candies) = 90 := 
by
  sorry

end candies_remaining_l1104_110434


namespace range_of_ab_l1104_110494

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |2 - a^2| = |2 - b^2|) : 0 < a * b ∧ a * b < 2 := by
  sorry

end range_of_ab_l1104_110494


namespace find_quadruples_l1104_110453

open Nat

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

theorem find_quadruples (a b p n : ℕ) (hp : is_prime p) (h_ab : a + b ≠ 0) :
  a^3 + b^3 = p^n ↔ (a = 1 ∧ b = 1 ∧ p = 2 ∧ n = 1) ∨
               (a = 1 ∧ b = 2 ∧ p = 3 ∧ n = 2) ∨ 
               (a = 2 ∧ b = 1 ∧ p = 3 ∧ n = 2) ∨
               ∃ (k : ℕ), (a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨ 
                          (a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
                          (a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) := sorry

end find_quadruples_l1104_110453


namespace standard_robot_weight_l1104_110449

variable (S : ℕ) -- Define the variable for the standard robot's weight
variable (MaxWeight : ℕ := 210) -- Define the variable for the maximum weight of a robot, which is 210 pounds
variable (MinWeight : ℕ) -- Define the variable for the minimum weight of the robot

theorem standard_robot_weight (h1 : 2 * MinWeight ≥ MaxWeight) 
                             (h2 : MinWeight = S + 5) 
                             (h3 : MaxWeight = 210) :
  100 ≤ S ∧ S ≤ 105 := 
by
  sorry

end standard_robot_weight_l1104_110449


namespace final_match_l1104_110472

-- Definitions of players and conditions
inductive Player
| Antony | Bart | Carl | Damian | Ed | Fred | Glen | Harry

open Player

-- Condition definitions
def beat (p1 p2 : Player) : Prop := sorry

-- Given conditions
axiom Bart_beats_Antony : beat Bart Antony
axiom Carl_beats_Damian : beat Carl Damian
axiom Glen_beats_Harry : beat Glen Harry
axiom Glen_beats_Carl : beat Glen Carl
axiom Carl_beats_Bart : beat Carl Bart
axiom Ed_beats_Fred : beat Ed Fred
axiom Glen_beats_Ed : beat Glen Ed

-- The proof statement
theorem final_match : beat Glen Carl :=
by
  sorry

end final_match_l1104_110472


namespace lines_of_first_character_l1104_110448

-- Definitions for the number of lines each character has
def L3 : Nat := 2

def L2 : Nat := 3 * L3 + 6

def L1 : Nat := L2 + 8

-- The theorem we are proving
theorem lines_of_first_character : L1 = 20 :=
by
  -- The proof would go here
  sorry

end lines_of_first_character_l1104_110448


namespace set_condition_implies_union_l1104_110462

open Set

variable {α : Type*} {M P : Set α}

theorem set_condition_implies_union 
  (h : M ∩ P = P) : M ∪ P = M := 
sorry

end set_condition_implies_union_l1104_110462


namespace pirate_treasure_probability_l1104_110430

theorem pirate_treasure_probability :
  let p_treasure_no_traps := 1 / 3
  let p_traps_no_treasure := 1 / 6
  let p_neither := 1 / 2
  let choose_4_out_of_8 := 70
  let p_4_treasure_no_traps := (1 / 3) ^ 4
  let p_4_neither := (1 / 2) ^ 4
  choose_4_out_of_8 * p_4_treasure_no_traps * p_4_neither = 35 / 648 :=
by
  sorry

end pirate_treasure_probability_l1104_110430


namespace sum_of_xi_l1104_110484

theorem sum_of_xi {x1 x2 x3 x4 : ℝ} (h1: (x1 - 3) * Real.sin (π * x1) = 1)
  (h2: (x2 - 3) * Real.sin (π * x2) = 1)
  (h3: (x3 - 3) * Real.sin (π * x3) = 1)
  (h4: (x4 - 3) * Real.sin (π * x4) = 1)
  (hx1 : x1 > 0) (hx2: x2 > 0) (hx3 : x3 > 0) (hx4: x4 > 0) :
  x1 + x2 + x3 + x4 = 12 :=
by
  sorry

end sum_of_xi_l1104_110484


namespace value_of_a_l1104_110473

theorem value_of_a (a b : ℝ) (h1 : b = 2120) (h2 : a / b = 0.5) : a = 1060 := 
by
  sorry

end value_of_a_l1104_110473


namespace jenny_improvements_value_l1104_110480

-- Definitions based on the conditions provided
def property_tax_rate : ℝ := 0.02
def initial_house_value : ℝ := 400000
def rail_project_increase : ℝ := 0.25
def affordable_property_tax : ℝ := 15000

-- Statement of the theorem
theorem jenny_improvements_value :
  let new_house_value := initial_house_value * (1 + rail_project_increase)
  let max_affordable_house_value := affordable_property_tax / property_tax_rate
  let value_of_improvements := max_affordable_house_value - new_house_value
  value_of_improvements = 250000 := 
by
  sorry

end jenny_improvements_value_l1104_110480


namespace infinite_div_pairs_l1104_110458

theorem infinite_div_pairs {a : ℕ → ℕ} (h_seq : ∀ n, 0 < a (n + 1) - a n ∧ a (n + 1) - a n ≤ 2001) :
  ∃ (s : ℕ → (ℕ × ℕ)), (∀ n, (s n).2 < (s n).1) ∧ (a ((s n).2) ∣ a ((s n).1)) :=
sorry

end infinite_div_pairs_l1104_110458


namespace greatest_integer_b_l1104_110464

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem greatest_integer_b (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b:ℝ) * x + 12 ≠ 0) ↔ b = 6 := 
by
  sorry

end greatest_integer_b_l1104_110464


namespace factor_1000000000001_l1104_110402

theorem factor_1000000000001 : ∃ a b c : ℕ, 1000000000001 = a * b * c ∧ a = 73 ∧ b = 137 ∧ c = 99990001 :=
by {
  sorry
}

end factor_1000000000001_l1104_110402


namespace remainder_division_by_8_is_6_l1104_110497

theorem remainder_division_by_8_is_6 (N Q2 R1 : ℤ) (h1 : N = 64 + R1) (h2 : N % 5 = 4) : R1 = 6 :=
by
  sorry

end remainder_division_by_8_is_6_l1104_110497


namespace clock_strikes_twelve_l1104_110485

def clock_strike_interval (strikes : Nat) (time : Nat) : Nat :=
  if strikes > 1 then time / (strikes - 1) else 0

def total_time_for_strikes (strikes : Nat) (interval : Nat) : Nat :=
  if strikes > 1 then (strikes - 1) * interval else 0

theorem clock_strikes_twelve (interval_six : Nat) (time_six : Nat) (time_twelve : Nat) :
  interval_six = clock_strike_interval 6 time_six →
  time_twelve = total_time_for_strikes 12 interval_six →
  time_six = 30 →
  time_twelve = 66 :=
by
  -- The proof will go here
  sorry

end clock_strikes_twelve_l1104_110485


namespace positive_diff_of_supplementary_angles_l1104_110447

theorem positive_diff_of_supplementary_angles (x : ℝ) (h : 5 * x + 3 * x = 180) : 
  abs ((5 * x - 3 * x)) = 45 := by
  sorry

end positive_diff_of_supplementary_angles_l1104_110447


namespace part_a_part_b_l1104_110409

def fake_coin_min_weighings_9 (n : ℕ) : ℕ :=
  if n = 9 then 2 else 0

def fake_coin_min_weighings_27 (n : ℕ) : ℕ :=
  if n = 27 then 3 else 0

theorem part_a : fake_coin_min_weighings_9 9 = 2 := by
  sorry

theorem part_b : fake_coin_min_weighings_27 27 = 3 := by
  sorry

end part_a_part_b_l1104_110409


namespace find_m_plus_n_l1104_110483

theorem find_m_plus_n (a m n : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a^m = n) (h4 : a^0 = 1) : m + n = 1 :=
sorry

end find_m_plus_n_l1104_110483


namespace least_possible_perimeter_l1104_110467

/-- Proof that the least possible perimeter of a triangle with two sides of length 24 and 51 units,
    and the third side being an integer, is 103 units. -/
theorem least_possible_perimeter (a b : ℕ) (c : ℕ) (h1 : a = 24) (h2 : b = 51) (h3 : c > 27) (h4 : c < 75) :
    a + b + c = 103 :=
by
  sorry

end least_possible_perimeter_l1104_110467


namespace transformed_parabola_equation_l1104_110482

-- Conditions
def original_parabola (x : ℝ) : ℝ := 3 * x^2
def translate_downwards (y : ℝ) : ℝ := y - 3

-- Translations
def translate_to_right (x : ℝ) : ℝ := x - 2
def transformed_parabola (x : ℝ) : ℝ := 3 * (x - 2)^2 - 3

-- Assertion
theorem transformed_parabola_equation :
  (∀ x : ℝ, translate_downwards (original_parabola x) = 3 * (translate_to_right x)^2 - 3) := by
  sorry

end transformed_parabola_equation_l1104_110482


namespace total_reptiles_l1104_110400

theorem total_reptiles 
  (reptiles_in_s1 : ℕ := 523)
  (reptiles_in_s2 : ℕ := 689)
  (reptiles_in_s3 : ℕ := 784)
  (reptiles_in_s4 : ℕ := 392)
  (reptiles_in_s5 : ℕ := 563)
  (reptiles_in_s6 : ℕ := 842) :
  reptiles_in_s1 + reptiles_in_s2 + reptiles_in_s3 + reptiles_in_s4 + reptiles_in_s5 + reptiles_in_s6 = 3793 :=
by
  sorry

end total_reptiles_l1104_110400


namespace geometric_sequence_term_l1104_110450

theorem geometric_sequence_term
  (r a : ℝ)
  (h1 : 180 * r = a)
  (h2 : a * r = 81 / 32)
  (h3 : a > 0) :
  a = 135 / 19 :=
by sorry

end geometric_sequence_term_l1104_110450


namespace min_correct_answers_l1104_110492

theorem min_correct_answers (total_questions correct_points incorrect_points target_score : ℕ)
                            (h_total : total_questions = 22)
                            (h_correct_points : correct_points = 4)
                            (h_incorrect_points : incorrect_points = 2)
                            (h_target : target_score = 81) :
  ∃ x : ℕ, 4 * x - 2 * (22 - x) > 81 ∧ x ≥ 21 :=
by {
  sorry
}

end min_correct_answers_l1104_110492


namespace tan_double_angle_l1104_110456

theorem tan_double_angle (α : Real) (h1 : α > π ∧ α < 3 * π / 2) (h2 : Real.sin (π - α) = -3/5) :
  Real.tan (2 * α) = 24/7 := 
by
  sorry

end tan_double_angle_l1104_110456


namespace decomposition_of_x_l1104_110440

-- Definitions derived from the conditions
def x : ℝ × ℝ × ℝ := (11, 5, -3)
def p : ℝ × ℝ × ℝ := (1, 0, 2)
def q : ℝ × ℝ × ℝ := (-1, 0, 1)
def r : ℝ × ℝ × ℝ := (2, 5, -3)

-- Theorem statement proving the decomposition
theorem decomposition_of_x : x = (3 : ℝ) • p + (-6 : ℝ) • q + (1 : ℝ) • r := by
  sorry

end decomposition_of_x_l1104_110440


namespace rebecca_pies_l1104_110405

theorem rebecca_pies 
  (P : ℕ) 
  (slices_per_pie : ℕ := 8) 
  (rebecca_slices : ℕ := P) 
  (family_and_friends_slices : ℕ := (7 * P) / 2) 
  (additional_slices : ℕ := 2) 
  (remaining_slices : ℕ := 5) 
  (total_slices : ℕ := slices_per_pie * P) :
  rebecca_slices + family_and_friends_slices + additional_slices + remaining_slices = total_slices → 
  P = 2 := 
by { sorry }

end rebecca_pies_l1104_110405


namespace sum_of_remainders_is_six_l1104_110433

theorem sum_of_remainders_is_six (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
  (a + b + c) % 15 = 6 :=
by
  sorry

end sum_of_remainders_is_six_l1104_110433


namespace max_value_of_expr_l1104_110496

theorem max_value_of_expr : ∃ t : ℝ, (∀ u : ℝ, (3^u - 2*u) * u / 9^u ≤ (3^t - 2*t) * t / 9^t) ∧ (3^t - 2*t) * t / 9^t = 1/8 :=
by sorry

end max_value_of_expr_l1104_110496


namespace glove_selection_correct_l1104_110410

-- Define the total number of different pairs of gloves
def num_pairs : Nat := 6

-- Define the required number of gloves to select
def num_gloves_to_select : Nat := 4

-- Define the function to calculate the number of ways to select 4 gloves with exactly one matching pair
noncomputable def count_ways_to_select_gloves (num_pairs : Nat) : Nat :=
  let select_pair := Nat.choose num_pairs 1
  let remaining_gloves := 2 * (num_pairs - 1)
  let select_two_from_remaining := Nat.choose remaining_gloves 2
  let subtract_unwanted_pairs := num_pairs - 1
  select_pair * (select_two_from_remaining - subtract_unwanted_pairs)

-- The correct answer we need to prove
def expected_result : Nat := 240

-- The theorem to prove the number of ways to select the gloves
theorem glove_selection_correct : count_ways_to_select_gloves num_pairs = expected_result :=
  by
    sorry

end glove_selection_correct_l1104_110410


namespace find_x_l1104_110481

def seq : ℕ → ℕ
| 0 => 2
| 1 => 5
| 2 => 11
| 3 => 20
| 4 => 32
| 5 => 47
| (n+6) => seq (n+5) + 3 * (n + 1)

theorem find_x : seq 6 = 65 := by
  sorry

end find_x_l1104_110481


namespace union_A_B_l1104_110498

-- Definitions based on the conditions
def A := { x : ℝ | x < -1 ∨ (2 ≤ x ∧ x < 3) }
def B := { x : ℝ | -2 ≤ x ∧ x < 4 }

-- The proof goal
theorem union_A_B : A ∪ B = { x : ℝ | x < 4 } :=
by
  sorry -- Proof placeholder

end union_A_B_l1104_110498
