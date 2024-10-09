import Mathlib

namespace justin_home_time_l1515_151581

noncomputable def dinner_duration : ℕ := 45
noncomputable def homework_duration : ℕ := 30
noncomputable def cleaning_room_duration : ℕ := 30
noncomputable def taking_out_trash_duration : ℕ := 5
noncomputable def emptying_dishwasher_duration : ℕ := 10

noncomputable def total_time_required : ℕ :=
  dinner_duration + homework_duration + cleaning_room_duration + taking_out_trash_duration + emptying_dishwasher_duration

noncomputable def latest_start_time_hour : ℕ := 18 -- 6 pm in 24-hour format
noncomputable def total_time_required_hours : ℕ := 2
noncomputable def movie_time_hour : ℕ := 20 -- 8 pm in 24-hour format

theorem justin_home_time : latest_start_time_hour - total_time_required_hours = 16 := -- 4 pm in 24-hour format
by
  sorry

end justin_home_time_l1515_151581


namespace geometric_sequence_a6_l1515_151519

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_a6 
  (a_1 q : ℝ) 
  (a2_eq : a_1 + a_1 * q = -1)
  (a3_eq : a_1 - a_1 * q ^ 2 = -3) : 
  a_n a_1 q 6 = -32 :=
sorry

end geometric_sequence_a6_l1515_151519


namespace sandwich_cost_l1515_151528

theorem sandwich_cost 
  (loaf_sandwiches : ℕ) (target_sandwiches : ℕ) 
  (bread_cost : ℝ) (meat_cost : ℝ) (cheese_cost : ℝ) 
  (cheese_coupon : ℝ) (meat_coupon : ℝ) (total_threshold : ℝ) 
  (discount_rate : ℝ)
  (h1 : loaf_sandwiches = 10) 
  (h2 : target_sandwiches = 50) 
  (h3 : bread_cost = 4) 
  (h4 : meat_cost = 5) 
  (h5 : cheese_cost = 4) 
  (h6 : cheese_coupon = 1) 
  (h7 : meat_coupon = 1) 
  (h8 : total_threshold = 60) 
  (h9 : discount_rate = 0.1) :
  ( ∃ cost_per_sandwich : ℝ, 
      cost_per_sandwich = 1.944 ) :=
  sorry

end sandwich_cost_l1515_151528


namespace shirts_per_minute_l1515_151527

/--
An industrial machine made 8 shirts today and worked for 4 minutes today. 
Prove that the machine can make 2 shirts per minute.
-/
theorem shirts_per_minute (shirts_today : ℕ) (minutes_today : ℕ)
  (h1 : shirts_today = 8) (h2 : minutes_today = 4) :
  (shirts_today / minutes_today) = 2 :=
by sorry

end shirts_per_minute_l1515_151527


namespace minimum_omega_l1515_151559

noncomputable def f (omega phi x : ℝ) : ℝ := Real.sin (omega * x + phi)

theorem minimum_omega {omega : ℝ} (h_pos : omega > 0) (h_even : ∀ x : ℝ, f omega (Real.pi / 2) x = f omega (Real.pi / 2) (-x)) 
  (h_zero_point : ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ f omega (Real.pi / 2) x = 0) :
  omega ≥ 1 / 2 :=
sorry

end minimum_omega_l1515_151559


namespace train_speed_problem_l1515_151597

open Real

/-- Given specific conditions about the speeds and lengths of trains, prove the speed of the third train is 99 kmph. -/
theorem train_speed_problem
  (man_train_speed_kmph : ℝ)
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (goods_train_time : ℝ)
  (third_train_length : ℝ)
  (third_train_time : ℝ) :
  man_train_speed_kmph = 45 →
  man_train_speed = 45 * 1000 / 3600 →
  goods_train_length = 340 →
  goods_train_time = 8 →
  third_train_length = 480 →
  third_train_time = 12 →
  (third_train_length / third_train_time - man_train_speed) * 3600 / 1000 = 99 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end train_speed_problem_l1515_151597


namespace total_fireworks_correct_l1515_151555

variable (fireworks_num fireworks_reg)
variable (fireworks_H fireworks_E fireworks_L fireworks_O)
variable (fireworks_square fireworks_triangle fireworks_circle)
variable (boxes fireworks_per_box : ℕ)

-- Given Conditions
def fireworks_years_2021_2023 : ℕ := 6 * 4 * 3
def fireworks_HAPPY_NEW_YEAR : ℕ := 5 * 11 + 6
def fireworks_geometric_shapes : ℕ := 4 + 3 + 12
def fireworks_HELLO : ℕ := 8 + 7 + 6 * 2 + 9
def fireworks_additional_boxes : ℕ := 100 * 10

-- Total Fireworks
def total_fireworks : ℕ :=
  fireworks_years_2021_2023 + 
  fireworks_HAPPY_NEW_YEAR + 
  fireworks_geometric_shapes + 
  fireworks_HELLO + 
  fireworks_additional_boxes

theorem total_fireworks_correct : 
  total_fireworks = 1188 :=
  by
  -- The proof is omitted.
  sorry

end total_fireworks_correct_l1515_151555


namespace max_acute_triangles_l1515_151509

theorem max_acute_triangles (n : ℕ) (hn : n ≥ 3) :
  (∃ k, k = if n % 2 = 0 then (n * (n-2) * (n+2)) / 24 else (n * (n-1) * (n+1)) / 24) :=
by 
  sorry

end max_acute_triangles_l1515_151509


namespace exists_three_sticks_form_triangle_l1515_151524

theorem exists_three_sticks_form_triangle 
  (l : Fin 5 → ℝ) 
  (h1 : ∀ i, 2 < l i) 
  (h2 : ∀ i, l i < 8) : 
  ∃ (i j k : Fin 5), i < j ∧ j < k ∧ 
    (l i + l j > l k) ∧ 
    (l j + l k > l i) ∧ 
    (l k + l i > l j) :=
sorry

end exists_three_sticks_form_triangle_l1515_151524


namespace chess_player_total_games_l1515_151526

noncomputable def total_games_played (W L : ℕ) : ℕ :=
  W + L

theorem chess_player_total_games :
  ∃ (W L : ℕ), W = 16 ∧ (L : ℚ) / W = 7 / 4 ∧ total_games_played W L = 44 :=
by
  sorry

end chess_player_total_games_l1515_151526


namespace minimum_value_of_expression_l1515_151569

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  x^2 + x * y + y^2 + 7

theorem minimum_value_of_expression :
  ∃ x y : ℝ, min_value_expression x y = 7 :=
by
  use 0, 0
  sorry

end minimum_value_of_expression_l1515_151569


namespace gcd_75_225_l1515_151512

theorem gcd_75_225 : Int.gcd 75 225 = 75 :=
by
  sorry

end gcd_75_225_l1515_151512


namespace marbles_per_friend_l1515_151590

theorem marbles_per_friend (total_marbles friends : ℕ) (h1 : total_marbles = 5504) (h2 : friends = 64) :
  total_marbles / friends = 86 :=
by {
  -- Proof will be added here
  sorry
}

end marbles_per_friend_l1515_151590


namespace functional_eq_solve_l1515_151551

theorem functional_eq_solve (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (2*x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
sorry

end functional_eq_solve_l1515_151551


namespace prime_square_plus_eight_is_prime_l1515_151539

theorem prime_square_plus_eight_is_prime (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^2 + 8)) : p = 3 :=
sorry

end prime_square_plus_eight_is_prime_l1515_151539


namespace uncovered_side_length_l1515_151514

theorem uncovered_side_length :
  ∃ (L : ℝ) (W : ℝ), L * W = 680 ∧ 2 * W + L = 146 ∧ L = 136 := by
  sorry

end uncovered_side_length_l1515_151514


namespace sqrt_five_minus_one_range_l1515_151574

theorem sqrt_five_minus_one_range (h : 2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3) : 
  1 < Real.sqrt 5 - 1 ∧ Real.sqrt 5 - 1 < 2 := 
by 
  sorry

end sqrt_five_minus_one_range_l1515_151574


namespace number_of_children_l1515_151583

theorem number_of_children (n m : ℕ) (h1 : 11 * (m + 6) + n * m = n^2 + 3 * n - 2) : n = 9 :=
sorry

end number_of_children_l1515_151583


namespace proof_sin_sum_ineq_proof_sin_product_ineq_proof_cos_sum_double_ineq_proof_cos_square_sum_ineq_proof_cos_half_product_ineq_proof_cos_product_ineq_l1515_151558

noncomputable def sin_sum_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.sin A + Real.sin B + Real.sin C) ≤ (3 / 2) * Real.sqrt 3

noncomputable def sin_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.sin A * Real.sin B * Real.sin C) ≤ (3 / 8) * Real.sqrt 3

noncomputable def cos_sum_double_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos (2 * A) + Real.cos (2 * B) + Real.cos (2 * C)) ≥ (-3 / 2)

noncomputable def cos_square_sum_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2) ≥ (3 / 4)

noncomputable def cos_half_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) ≤ (3 / 8) * Real.sqrt 3

noncomputable def cos_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos A * Real.cos B * Real.cos C) ≤ (1 / 8)

theorem proof_sin_sum_ineq {A B C : ℝ} (hABC : A + B + C = π) : sin_sum_ineq A B C hABC := sorry

theorem proof_sin_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : sin_product_ineq A B C hABC := sorry

theorem proof_cos_sum_double_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_sum_double_ineq A B C hABC := sorry

theorem proof_cos_square_sum_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_square_sum_ineq A B C hABC := sorry

theorem proof_cos_half_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_half_product_ineq A B C hABC := sorry

theorem proof_cos_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_product_ineq A B C hABC := sorry

end proof_sin_sum_ineq_proof_sin_product_ineq_proof_cos_sum_double_ineq_proof_cos_square_sum_ineq_proof_cos_half_product_ineq_proof_cos_product_ineq_l1515_151558


namespace total_action_figures_l1515_151515

theorem total_action_figures (figures_per_shelf : ℕ) (number_of_shelves : ℕ) (h1 : figures_per_shelf = 10) (h2 : number_of_shelves = 8) : figures_per_shelf * number_of_shelves = 80 := by
  sorry

end total_action_figures_l1515_151515


namespace probability_opposite_vertex_l1515_151522

theorem probability_opposite_vertex (k : ℕ) (h : k > 0) : 
    P_k = (1 / 6 : ℝ) + (1 / (3 * (-2) ^ k) : ℝ) := 
sorry

end probability_opposite_vertex_l1515_151522


namespace find_five_digit_number_l1515_151596

theorem find_five_digit_number (a b c d e : ℕ) 
  (h : [ (10 * a + a), (10 * a + b), (10 * a + b), (10 * a + b), (10 * a + c), 
         (10 * b + c), (10 * b + b), (10 * b + c), (10 * c + b), (10 * c + b)] = 
         [33, 37, 37, 37, 38, 73, 77, 78, 83, 87]) :
  10000 * a + 1000 * b + 100 * c + 10 * d + e = 37837 :=
sorry

end find_five_digit_number_l1515_151596


namespace S_equals_2_l1515_151562

noncomputable def problem_S := 
  1 / (2 - Real.sqrt 3) - 1 / (Real.sqrt 3 - Real.sqrt 2) + 
  1 / (Real.sqrt 2 - 1) - 1 / (1 - Real.sqrt 3 + Real.sqrt 2)

theorem S_equals_2 : problem_S = 2 := by
  sorry

end S_equals_2_l1515_151562


namespace fraction_of_power_l1515_151542

noncomputable def m : ℕ := 32^500

theorem fraction_of_power (h : m = 2^2500) : m / 8 = 2^2497 :=
by
  have hm : m = 2^2500 := h
  sorry

end fraction_of_power_l1515_151542


namespace cannot_be_value_of_A_plus_P_l1515_151588

theorem cannot_be_value_of_A_plus_P (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (a_neq_b: a ≠ b) :
  let A : ℕ := a * b
  let P : ℕ := 2 * a + 2 * b
  A + P ≠ 102 :=
by
  sorry

end cannot_be_value_of_A_plus_P_l1515_151588


namespace circle_tangent_to_x_axis_at_origin_l1515_151505

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h1 : ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 → x = 0 ∧ y = 0 ∨ y = -D/E ∧ x = 0 ∧ F = 0):
  D = 0 ∧ E ≠ 0 ∧ F = 0 :=
sorry

end circle_tangent_to_x_axis_at_origin_l1515_151505


namespace time_jran_l1515_151580

variable (D : ℕ) (S : ℕ)

theorem time_jran (hD: D = 80) (hS : S = 10) : D / S = 8 := 
  sorry

end time_jran_l1515_151580


namespace maximal_p_sum_consecutive_l1515_151511

theorem maximal_p_sum_consecutive (k : ℕ) (h1 : k = 31250) : 
  ∃ p a : ℕ, p * (2 * a + p - 1) = k ∧ ∀ p' a', (p' * (2 * a' + p' - 1) = k) → p' ≤ p := by
  sorry

end maximal_p_sum_consecutive_l1515_151511


namespace correct_total_annual_cost_l1515_151570

def cost_after_coverage (cost: ℕ) (coverage: ℕ) : ℕ :=
  cost - (cost * coverage / 100)

def epiPen_costs : ℕ :=
  (cost_after_coverage 500 75) +
  (cost_after_coverage 550 60) +
  (cost_after_coverage 480 70) +
  (cost_after_coverage 520 65)

def monthly_medical_expenses : ℕ :=
  (cost_after_coverage 250 80) +
  (cost_after_coverage 180 70) +
  (cost_after_coverage 300 75) +
  (cost_after_coverage 350 60) +
  (cost_after_coverage 200 70) +
  (cost_after_coverage 400 80) +
  (cost_after_coverage 150 90) +
  (cost_after_coverage 100 100) +
  (cost_after_coverage 300 60) +
  (cost_after_coverage 350 90) +
  (cost_after_coverage 450 85) +
  (cost_after_coverage 500 65)

def total_annual_cost : ℕ :=
  epiPen_costs + monthly_medical_expenses

theorem correct_total_annual_cost :
  total_annual_cost = 1542 :=
  by sorry

end correct_total_annual_cost_l1515_151570


namespace length_of_FD_l1515_151572

/-- In a square of side length 8 cm, point E is located on side AD,
2 cm from A and 6 cm from D. Point F lies on side CD such that folding
the square so that C coincides with E creates a crease along GF. 
Prove that the length of segment FD is 7/4 cm. -/
theorem length_of_FD (x : ℝ) (h_square : ∀ (A B C D : ℝ), A = 8 ∧ B = 8 ∧ C = 8 ∧ D = 8)
    (h_AE : ∀ (A E : ℝ), A - E = 2) (h_ED : ∀ (E D : ℝ), E - D = 6)
    (h_pythagorean : ∀ (x : ℝ), (8 - x)^2 = x^2 + 6^2) : x = 7/4 :=
by
  sorry

end length_of_FD_l1515_151572


namespace maximum_value_expression_l1515_151533

theorem maximum_value_expression (a b c : ℕ) (ha : 0 < a ∧ a ≤ 9) (hb : 0 < b ∧ b ≤ 9) (hc : 0 < c ∧ c ≤ 9) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ∃ (v : ℚ), v = (1 / (a + 2010 / (b + 1 / c : ℚ))) ∧ v ≤ (1 / 203) :=
sorry

end maximum_value_expression_l1515_151533


namespace negation_proof_l1515_151595

theorem negation_proof :
  ¬ (∀ x : ℝ, 0 < x ∧ x < (π / 2) → x > Real.sin x) ↔ 
  ∃ x : ℝ, 0 < x ∧ x < (π / 2) ∧ x ≤ Real.sin x := 
sorry

end negation_proof_l1515_151595


namespace find_b_l1515_151578

theorem find_b (b : ℤ) (h₁ : b < 0) : (∃ n : ℤ, (x : ℤ) * x + b * x - 36 = (x + n) * (x + n) - 20) → b = -8 :=
by
  intro hX
  sorry

end find_b_l1515_151578


namespace six_digit_divisible_by_72_l1515_151550

theorem six_digit_divisible_by_72 (n m : ℕ) (h1 : n = 920160 ∨ n = 120168) :
  (∃ (x y : ℕ), 10 * x + y = 2016 ∧ (10^5 * x + n * 10 + m) % 72 = 0) :=
by
  sorry

end six_digit_divisible_by_72_l1515_151550


namespace volume_of_sphere_inscribed_in_cube_of_edge_8_l1515_151599

noncomputable def volume_of_inscribed_sphere (edge_length : ℝ) : ℝ := 
  (4 / 3) * Real.pi * (edge_length / 2) ^ 3

theorem volume_of_sphere_inscribed_in_cube_of_edge_8 :
  volume_of_inscribed_sphere 8 = (256 / 3) * Real.pi :=
by
  sorry

end volume_of_sphere_inscribed_in_cube_of_edge_8_l1515_151599


namespace max_area_triangle_max_area_quadrilateral_l1515_151584

-- Define the terms and conditions

variables {A O : Point}
variables {r d : ℝ}
variables {C D B : Point}

-- Problem (a)
theorem max_area_triangle (A O : Point) (d : ℝ) :
  (∃ x : ℝ, x = (3 / 4) * d) :=
sorry

-- Problem (b)
theorem max_area_quadrilateral (A O : Point) (d : ℝ) :
  (∃ x : ℝ, x = (1 / 2) * d) :=
sorry

end max_area_triangle_max_area_quadrilateral_l1515_151584


namespace gcd_multiple_less_than_120_l1515_151577

theorem gcd_multiple_less_than_120 (n : ℕ) (h1 : n < 120) (h2 : n % 10 = 0) (h3 : n % 15 = 0) : n ≤ 90 :=
by {
  sorry
}

end gcd_multiple_less_than_120_l1515_151577


namespace expression_equals_24_l1515_151552

noncomputable def f : ℕ → ℝ := sorry

axiom f_add (m n : ℕ) : f (m + n) = f m * f n
axiom f_one : f 1 = 3

theorem expression_equals_24 :
  (f 1^2 + f 2) / f 1 + (f 2^2 + f 4) / f 3 + (f 3^2 + f 6) / f 5 + (f 4^2 + f 8) / f 7 = 24 :=
by sorry

end expression_equals_24_l1515_151552


namespace general_equation_of_line_cartesian_equation_of_curve_minimum_AB_l1515_151592

open Real

-- Definitions for the problem
variable (t : ℝ) (φ θ : ℝ) (x y P : ℝ)

-- Conditions
def line_parametric := x = t * sin φ ∧ y = 1 + t * cos φ
def curve_polar := P * (cos θ)^2 = 4 * sin θ
def curve_cartesian := x^2 = 4 * y
def line_general := x * cos φ - y * sin φ + sin φ = 0

-- Proof problem statements

-- 1. Prove the general equation of line l
theorem general_equation_of_line (h : line_parametric t φ x y) : line_general φ x y :=
sorry

-- 2. Prove the cartesian coordinate equation of curve C
theorem cartesian_equation_of_curve (h : curve_polar P θ) : curve_cartesian x y :=
sorry

-- 3. Prove the minimum |AB| where line l intersects curve C
theorem minimum_AB (h_line : line_parametric t φ x y) (h_curve : curve_cartesian x y) : ∃ (min_ab : ℝ), min_ab = 4 :=
sorry

end general_equation_of_line_cartesian_equation_of_curve_minimum_AB_l1515_151592


namespace pasha_game_solvable_l1515_151506

def pasha_game : Prop :=
∃ (a : Fin 2017 → ℕ), 
  (∀ i, a i > 0) ∧
  (∃ (moves : ℕ), moves = 43 ∧
   (∀ (box_contents : Fin 2017 → ℕ), 
    (∀ j, box_contents j = 0) →
    (∃ (equal_count : ℕ),
      (∀ j, box_contents j = equal_count)
      ∧
      (∀ m < 43,
        ∃ j, box_contents j ≠ equal_count))))

theorem pasha_game_solvable : pasha_game :=
by
  sorry

end pasha_game_solvable_l1515_151506


namespace percentage_of_part_l1515_151545

theorem percentage_of_part (Part Whole : ℝ) (hPart : Part = 120) (hWhole : Whole = 50) : (Part / Whole) * 100 = 240 := 
by
  sorry

end percentage_of_part_l1515_151545


namespace expected_winnings_l1515_151586

theorem expected_winnings :
  let p_heads : ℚ := 1 / 4
  let p_tails : ℚ := 1 / 2
  let p_edge : ℚ := 1 / 4
  let win_heads : ℚ := 1
  let win_tails : ℚ := 3
  let loss_edge : ℚ := -8
  (p_heads * win_heads + p_tails * win_tails + p_edge * loss_edge) = -0.25 := 
by sorry

end expected_winnings_l1515_151586


namespace right_triangle_acute_angles_l1515_151517

variable (α β : ℝ)

noncomputable def prove_acute_angles (α β : ℝ) : Prop :=
  α + β = 90 ∧ 4 * α = 90

theorem right_triangle_acute_angles : 
  prove_acute_angles α β → α = 22.5 ∧ β = 67.5 := by
  sorry

end right_triangle_acute_angles_l1515_151517


namespace sum_modulo_seven_l1515_151529

theorem sum_modulo_seven (a b c : ℕ) (h1: a = 9^5) (h2: b = 8^6) (h3: c = 7^7) :
  (a + b + c) % 7 = 5 :=
by sorry

end sum_modulo_seven_l1515_151529


namespace child_grandmother_ratio_l1515_151516

variable (G D C : ℕ)

axiom cond1 : G + D + C = 120
axiom cond2 : D + C = 60
axiom cond3 : D = 48

theorem child_grandmother_ratio : (C : ℚ) / G = 1 / 5 :=
by
  sorry

end child_grandmother_ratio_l1515_151516


namespace variance_of_data_l1515_151598

theorem variance_of_data :
  let data := [3, 1, 0, -1, -3]
  let mean := (3 + 1 + 0 - 1 - 3) / (5:ℝ)
  let variance := (1 / 5:ℝ) * (3^2 + 1^2 + (-1)^2 + (-3)^2)
  variance = 4 := sorry

end variance_of_data_l1515_151598


namespace visitors_yesterday_l1515_151537

-- Definitions based on the given conditions
def visitors_today : ℕ := 583
def visitors_total : ℕ := 829

-- Theorem statement to prove the number of visitors the day before Rachel visited
theorem visitors_yesterday : ∃ v_yesterday: ℕ, v_yesterday = visitors_total - visitors_today ∧ v_yesterday = 246 :=
by
  sorry

end visitors_yesterday_l1515_151537


namespace total_cost_is_correct_l1515_151560

-- Define the number of total tickets and the number of children's tickets
def total_tickets : ℕ := 21
def children_tickets : ℕ := 16
def adult_tickets : ℕ := total_tickets - children_tickets

-- Define the cost of tickets for adults and children
def cost_per_adult_ticket : ℝ := 5.50
def cost_per_child_ticket : ℝ := 3.50

-- Define the total cost spent
def total_cost_spent : ℝ :=
  (adult_tickets * cost_per_adult_ticket) + (children_tickets * cost_per_child_ticket)

-- Prove that the total amount spent on tickets is $83.50
theorem total_cost_is_correct : total_cost_spent = 83.50 := by
  sorry

end total_cost_is_correct_l1515_151560


namespace xyz_divides_xyz_squared_l1515_151557

theorem xyz_divides_xyz_squared (x y z p : ℕ) (hxyz : x < y ∧ y < z ∧ z < p) (hp : Nat.Prime p) (hx3 : x^3 ≡ y^3 [MOD p])
    (hy3 : y^3 ≡ z^3 [MOD p]) (hz3 : z^3 ≡ x^3 [MOD p]) : (x + y + z) ∣ (x^2 + y^2 + z^2) :=
by
  sorry

end xyz_divides_xyz_squared_l1515_151557


namespace eval_at_neg_five_l1515_151568

def f (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem eval_at_neg_five : f (-5) = 12 :=
by
  sorry

end eval_at_neg_five_l1515_151568


namespace irrational_lattice_point_exists_l1515_151531

theorem irrational_lattice_point_exists (k : ℝ) (h_irrational : ¬ ∃ q r : ℚ, q / r = k)
  (ε : ℝ) (h_pos : ε > 0) : ∃ m n : ℤ, |m * k - n| < ε :=
by
  sorry

end irrational_lattice_point_exists_l1515_151531


namespace average_weight_of_Arun_l1515_151508

def Arun_weight_opinion (w : ℝ) : Prop :=
  (66 < w) ∧ (w < 72)

def Brother_weight_opinion (w : ℝ) : Prop :=
  (60 < w) ∧ (w < 70)

def Mother_weight_opinion (w : ℝ) : Prop :=
  w ≤ 69

def Father_weight_opinion (w : ℝ) : Prop :=
  (65 ≤ w) ∧ (w ≤ 71)

def Sister_weight_opinion (w : ℝ) : Prop :=
  (62 < w) ∧ (w ≤ 68)

def All_opinions (w : ℝ) : Prop :=
  Arun_weight_opinion w ∧
  Brother_weight_opinion w ∧
  Mother_weight_opinion w ∧
  Father_weight_opinion w ∧
  Sister_weight_opinion w

theorem average_weight_of_Arun : ∃ avg : ℝ, avg = 67.5 ∧ (∀ w, All_opinions w → (w = 67 ∨ w = 68)) :=
by
  sorry

end average_weight_of_Arun_l1515_151508


namespace inequality_k_m_l1515_151525

theorem inequality_k_m (k m : ℕ) (hk : 0 < k) (hm : 0 < m) (hkm : k > m) (hdiv : (k^3 - m^3) ∣ k * m * (k^2 - m^2)) :
  (k - m)^3 > 3 * k * m := 
by sorry

end inequality_k_m_l1515_151525


namespace find_xy_l1515_151500

theorem find_xy (x y : ℝ) (π_ne_zero : Real.pi ≠ 0) (h1 : 4 * (x + 2) = 6 * x) (h2 : 6 * x = 2 * Real.pi * y) : x = 4 ∧ y = 12 / Real.pi :=
by
  sorry

end find_xy_l1515_151500


namespace metro_earnings_in_6_minutes_l1515_151563

theorem metro_earnings_in_6_minutes 
  (ticket_cost : ℕ) 
  (tickets_per_minute : ℕ) 
  (duration_minutes : ℕ) 
  (earnings_in_one_minute : ℕ) 
  (earnings_in_six_minutes : ℕ) 
  (h1 : ticket_cost = 3) 
  (h2 : tickets_per_minute = 5) 
  (h3 : duration_minutes = 6) 
  (h4 : earnings_in_one_minute = tickets_per_minute * ticket_cost) 
  (h5 : earnings_in_six_minutes = earnings_in_one_minute * duration_minutes) 
  : earnings_in_six_minutes = 90 := 
by 
  -- Proof goes here
  sorry

end metro_earnings_in_6_minutes_l1515_151563


namespace sum_of_undefined_fractions_l1515_151503

theorem sum_of_undefined_fractions (x₁ x₂ : ℝ) (h₁ : x₁^2 - 7*x₁ + 12 = 0) (h₂ : x₂^2 - 7*x₂ + 12 = 0) :
  x₁ + x₂ = 7 :=
sorry

end sum_of_undefined_fractions_l1515_151503


namespace parabola_tangent_sequence_l1515_151561

noncomputable def geom_seq_sum (a2 : ℕ) : ℕ :=
  a2 + a2 / 4 + a2 / 16

theorem parabola_tangent_sequence (a2 : ℕ) (h : a2 = 32) : geom_seq_sum a2 = 42 :=
by
  rw [h]
  norm_num
  sorry

end parabola_tangent_sequence_l1515_151561


namespace hyperbola_eccentricity_l1515_151582

variables (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
variables (c e : ℝ)

-- Define the eccentricy condition for hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_eccentricity :
  -- Conditions regarding the hyperbola and the distances
  (∀ x y : ℝ, hyperbola a b x y → 
    (∃ x y : ℝ, y = (2 / 3) * c ∧ x = 2 * a + (2 / 3) * c ∧
    ((2 / 3) * c)^2 + (2 * a + (2 / 3) * c)^2 = 4 * c^2 ∧
    (7 * e^2 - 6 * e - 9 = 0))) →
  -- Proving that the eccentricity e is as given
  e = (3 + Real.sqrt 6) / 7 :=
sorry

end hyperbola_eccentricity_l1515_151582


namespace martins_travel_time_l1515_151520

-- Declare the necessary conditions from the problem
variables (speed : ℝ) (distance : ℝ)
-- Define the conditions
def martin_speed := speed = 12 -- Martin's speed is 12 miles per hour
def martin_distance := distance = 72 -- Martin drove 72 miles

-- State the theorem to prove the time taken is 6 hours
theorem martins_travel_time (h1 : martin_speed speed) (h2 : martin_distance distance) : distance / speed = 6 :=
by
  -- To complete the problem statement, insert sorry to skip the actual proof
  sorry

end martins_travel_time_l1515_151520


namespace factorial_inequality_l1515_151544

theorem factorial_inequality (n : ℕ) (h : n ≥ 1) : n! ≤ ((n+1)/2)^n := 
by {
  sorry
}

end factorial_inequality_l1515_151544


namespace min_m_value_inequality_x2y2z_l1515_151585

theorem min_m_value (a b : ℝ) (h1 : a * b > 0) (h2 : a^2 * b = 2) : 
  ∃ (m : ℝ), m = a * b + a^2 ∧ m = 3 :=
sorry

theorem inequality_x2y2z 
  (t : ℝ) (ht : t = 3) (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x^2 + y^2 + z^2 = t / 3) : 
  |x + 2 * y + 2 * z| ≤ 3 :=
sorry

end min_m_value_inequality_x2y2z_l1515_151585


namespace jar_water_transfer_l1515_151549

theorem jar_water_transfer
  (C_x : ℝ) (C_y : ℝ)
  (h1 : C_y = 1/2 * C_x)
  (WaterInX : ℝ)
  (WaterInY : ℝ)
  (h2 : WaterInX = 1/2 * C_x)
  (h3 : WaterInY = 1/2 * C_y) :
  WaterInX + WaterInY = 3/4 * C_x :=
by
  sorry

end jar_water_transfer_l1515_151549


namespace fraction_product_equals_l1515_151556

theorem fraction_product_equals :
  (7 / 4) * (14 / 49) * (10 / 15) * (12 / 36) * (21 / 14) * (40 / 80) * (33 / 22) * (16 / 64) = 1 / 12 := 
  sorry

end fraction_product_equals_l1515_151556


namespace river_bank_depth_l1515_151587

-- Definitions related to the problem
def is_trapezium (top_width bottom_width height area : ℝ) :=
  area = 1 / 2 * (top_width + bottom_width) * height

-- The theorem we want to prove
theorem river_bank_depth :
  ∀ (top_width bottom_width area : ℝ), 
    top_width = 12 → 
    bottom_width = 8 → 
    area = 500 → 
    ∃ h : ℝ, is_trapezium top_width bottom_width h area ∧ h = 50 :=
by
  intros top_width bottom_width area ht hb ha
  sorry

end river_bank_depth_l1515_151587


namespace find_fourth_number_in_sequence_l1515_151534

-- Define the conditions of the sequence
def first_number : ℤ := 1370
def second_number : ℤ := 1310
def third_number : ℤ := 1070
def fifth_number : ℤ := -6430

-- Define the differences
def difference1 : ℤ := second_number - first_number
def difference2 : ℤ := third_number - second_number

-- Define the ratio of differences
def ratio : ℤ := 4
def next_difference : ℤ := difference2 * ratio

-- Define the fourth number
def fourth_number : ℤ := third_number - (-next_difference)

-- Theorem stating the proof problem
theorem find_fourth_number_in_sequence : fourth_number = 2030 :=
by sorry

end find_fourth_number_in_sequence_l1515_151534


namespace proportion_margin_l1515_151576

theorem proportion_margin (S M C : ℝ) (n : ℝ) (hM : M = S / n) (hC : C = (1 - 1 / n) * S) :
  M / C = 1 / (n - 1) :=
by
  sorry

end proportion_margin_l1515_151576


namespace bert_average_words_in_crossword_l1515_151541

theorem bert_average_words_in_crossword :
  (10 * 35 + 4 * 65) / (10 + 4) = 43.57 :=
by
  sorry

end bert_average_words_in_crossword_l1515_151541


namespace marie_socks_problem_l1515_151579

theorem marie_socks_problem (x y z : ℕ) : 
  x + y + z = 15 → 
  2 * x + 3 * y + 5 * z = 36 → 
  1 ≤ x → 
  1 ≤ y → 
  1 ≤ z → 
  x = 11 :=
by
  sorry

end marie_socks_problem_l1515_151579


namespace negation_of_p_l1515_151518

-- Declare the proposition p as a condition
def p : Prop :=
  ∀ (x : ℝ), 0 ≤ x → x^2 + 4 * x + 3 > 0

-- State the problem
theorem negation_of_p : ¬ p ↔ ∃ (x : ℝ), 0 ≤ x ∧ x^2 + 4 * x + 3 ≤ 0 :=
by
  sorry

end negation_of_p_l1515_151518


namespace count_non_congruent_rectangles_l1515_151553

-- Definitions based on conditions given in the problem
def is_rectangle (w h : ℕ) : Prop := 2 * (w + h) = 40 ∧ w % 2 = 0

-- Theorem that we need to prove based on the problem statement
theorem count_non_congruent_rectangles : 
  ∃ n : ℕ, n = 9 ∧ 
  (∀ p : ℕ × ℕ, p ∈ { p | is_rectangle p.1 p.2 } → ∀ q : ℕ × ℕ, q ∈ { q | is_rectangle q.1 q.2 } → p = q ∨ p ≠ q) := 
sorry

end count_non_congruent_rectangles_l1515_151553


namespace find_value_of_g1_l1515_151589

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = g x

theorem find_value_of_g1 (h1 : is_odd f)
  (h2 : is_even g)
  (h3 : f (-1) + g 1 = 2)
  (h4 : f 1 + g (-1) = 4) : 
  g 1 = 3 :=
sorry

end find_value_of_g1_l1515_151589


namespace sandy_comic_books_l1515_151507

-- Problem definition
def initial_comic_books := 14
def sold_comic_books := initial_comic_books / 2
def remaining_comic_books := initial_comic_books - sold_comic_books
def bought_comic_books := 6
def final_comic_books := remaining_comic_books + bought_comic_books

-- Proof statement
theorem sandy_comic_books : final_comic_books = 13 := by
  sorry

end sandy_comic_books_l1515_151507


namespace find_x_l1515_151513

noncomputable def x_half_y (x y : ℚ) : Prop := x = (1 / 2) * y
noncomputable def y_third_z (y z : ℚ) : Prop := y = (1 / 3) * z

theorem find_x (x y z : ℚ) (h₁ : x_half_y x y) (h₂ : y_third_z y z) (h₃ : z = 100) :
  x = 16 + (2 / 3 : ℚ) :=
by
  sorry

end find_x_l1515_151513


namespace sum_of_5_and_8_l1515_151521

theorem sum_of_5_and_8 : 5 + 8 = 13 := by
  rfl

end sum_of_5_and_8_l1515_151521


namespace dan_took_pencils_l1515_151543

theorem dan_took_pencils (initial_pencils remaining_pencils : ℕ) (h_initial : initial_pencils = 34) (h_remaining : remaining_pencils = 12) : (initial_pencils - remaining_pencils) = 22 := 
by
  sorry

end dan_took_pencils_l1515_151543


namespace find_brick_width_l1515_151536

def SurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

theorem find_brick_width :
  ∃ width : ℝ, SurfaceArea 10 width 3 = 164 ∧ width = 4 :=
by
  sorry

end find_brick_width_l1515_151536


namespace middle_value_bounds_l1515_151532

theorem middle_value_bounds (a b c : ℝ) (h1 : a + b + c = 10)
  (h2 : a > b) (h3 : b > c) (h4 : a - c = 3) : 
  7 / 3 < b ∧ b < 13 / 3 :=
by
  sorry

end middle_value_bounds_l1515_151532


namespace distance_from_apex_to_larger_cross_section_l1515_151594

noncomputable def area1 : ℝ := 324 * Real.sqrt 2
noncomputable def area2 : ℝ := 648 * Real.sqrt 2
def distance_between_planes : ℝ := 12

theorem distance_from_apex_to_larger_cross_section
  (area1 area2 : ℝ)
  (distance_between_planes : ℝ)
  (h_area1 : area1 = 324 * Real.sqrt 2)
  (h_area2 : area2 = 648 * Real.sqrt 2)
  (h_distance : distance_between_planes = 12) :
  ∃ (H : ℝ), H = 24 + 12 * Real.sqrt 2 :=
by sorry

end distance_from_apex_to_larger_cross_section_l1515_151594


namespace one_interior_angle_of_polygon_with_five_diagonals_l1515_151510

theorem one_interior_angle_of_polygon_with_five_diagonals (n : ℕ) (h : n - 3 = 5) :
  let interior_angle := 180 * (n - 2) / n
  interior_angle = 135 :=
by
  sorry

end one_interior_angle_of_polygon_with_five_diagonals_l1515_151510


namespace find_d_in_triangle_ABC_l1515_151502

theorem find_d_in_triangle_ABC (AB BC AC : ℝ) (P : Type) (d : ℝ) 
  (h_AB : AB = 480) (h_BC : BC = 500) (h_AC : AC = 550)
  (h_segments_equal : ∀ (D D' E E' F F' : Type), true) : 
  d = 132000 / 654 :=
sorry

end find_d_in_triangle_ABC_l1515_151502


namespace train_stop_time_l1515_151523

theorem train_stop_time (speed_no_stops speed_with_stops : ℕ) (time_per_hour : ℕ) (stoppage_time_per_hour : ℕ) :
  speed_no_stops = 45 →
  speed_with_stops = 30 →
  time_per_hour = 60 →
  stoppage_time_per_hour = 20 :=
by
  intros h1 h2 h3
  sorry

end train_stop_time_l1515_151523


namespace profit_per_cake_l1515_151546

theorem profit_per_cake (ingredient_cost : ℝ) (packaging_cost : ℝ) (selling_price : ℝ) (cake_count : ℝ)
    (h1 : ingredient_cost = 12) (h2 : packaging_cost = 1) (h3 : selling_price = 15) (h4 : cake_count = 2) :
    selling_price - (ingredient_cost / cake_count + packaging_cost) = 8 := by
  sorry

end profit_per_cake_l1515_151546


namespace floor_identity_l1515_151540

theorem floor_identity (x : ℝ) : 
    (⌊(3 + x) / 6⌋ - ⌊(4 + x) / 6⌋ + ⌊(5 + x) / 6⌋ = ⌊(1 + x) / 2⌋ - ⌊(1 + x) / 3⌋) :=
by
  sorry

end floor_identity_l1515_151540


namespace sum_of_digits_b_n_l1515_151530

def a_n (n : ℕ) : ℕ := 10^(2^n) - 1

def b_n (n : ℕ) : ℕ :=
  List.prod (List.map a_n (List.range (n + 1)))

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_b_n (n : ℕ) : sum_of_digits (b_n n) = 9 * 2^n :=
  sorry

end sum_of_digits_b_n_l1515_151530


namespace distance_between_points_eq_l1515_151554

theorem distance_between_points_eq :
  let x1 := 2
  let y1 := -5
  let x2 := -8
  let y2 := 7
  let distance := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  distance = 2 * Real.sqrt 61 :=
by
  sorry

end distance_between_points_eq_l1515_151554


namespace first_number_is_210_l1515_151504

theorem first_number_is_210 (A B hcf lcm : ℕ) (h1 : lcm = 2310) (h2: hcf = 47) (h3 : B = 517) :
  A * B = lcm * hcf → A = 210 :=
by
  sorry

end first_number_is_210_l1515_151504


namespace card_dealing_probability_l1515_151548

noncomputable def probability_ace_then_ten_then_jack : ℚ :=
  let prob_ace := 4 / 52
  let prob_ten := 4 / 51
  let prob_jack := 4 / 50
  prob_ace * prob_ten * prob_jack

theorem card_dealing_probability :
  probability_ace_then_ten_then_jack = 16 / 33150 := by
  sorry

end card_dealing_probability_l1515_151548


namespace determine_marbles_l1515_151567

noncomputable def marbles_total (x : ℚ) := (4 * x + 2) + (2 * x) + (3 * x - 1)

theorem determine_marbles (x : ℚ) (h1 : marbles_total x = 47) :
  (4 * x + 2 = 202 / 9) ∧ (2 * x = 92 / 9) ∧ (3 * x - 1 = 129 / 9) :=
by
  sorry

end determine_marbles_l1515_151567


namespace candy_store_sampling_l1515_151573

theorem candy_store_sampling (total_customers sampling_customers caught_customers not_caught_customers : ℝ)
    (h1 : caught_customers = 0.22 * total_customers)
    (h2 : not_caught_customers = 0.15 * sampling_customers)
    (h3 : sampling_customers = caught_customers + not_caught_customers):
    sampling_customers = 0.2588 * total_customers := by
  sorry

end candy_store_sampling_l1515_151573


namespace number_of_boys_in_other_communities_l1515_151535

-- Definitions from conditions
def total_boys : ℕ := 700
def percentage_muslims : ℕ := 44
def percentage_hindus : ℕ := 28
def percentage_sikhs : ℕ := 10

-- Proof statement
theorem number_of_boys_in_other_communities : 
  (700 * (100 - (44 + 28 + 10)) / 100) = 126 := 
by
  sorry

end number_of_boys_in_other_communities_l1515_151535


namespace part1_solution_part2_no_solution_l1515_151501

theorem part1_solution (x y : ℚ) :
  x + y = 5 ∧ 3 * x + 10 * y = 30 ↔ x = 20 / 7 ∧ y = 15 / 7 :=
by
  sorry

theorem part2_no_solution (x : ℚ) :
  (x + 7) / 2 < 4 ∧ (3 * x - 1) / 2 ≤ 2 * x - 3 ↔ False :=
by
  sorry

end part1_solution_part2_no_solution_l1515_151501


namespace spatial_quadrilateral_angle_sum_l1515_151593

theorem spatial_quadrilateral_angle_sum (A B C D : ℝ) (ABD DBC ADB BDC : ℝ) :
  (A <= ABD + DBC) → (C <= ADB + BDC) → 
  (A + C + B + D <= 360) := 
by
  intros
  sorry

end spatial_quadrilateral_angle_sum_l1515_151593


namespace log_one_plus_two_x_lt_two_x_l1515_151575
open Real

theorem log_one_plus_two_x_lt_two_x {x : ℝ} (hx : x > 0) : log (1 + 2 * x) < 2 * x :=
sorry

end log_one_plus_two_x_lt_two_x_l1515_151575


namespace find_M_base7_l1515_151571

theorem find_M_base7 :
  ∃ M : ℕ, M = 48 ∧ (M^2).digits 7 = [6, 6] ∧ (∃ (m : ℕ), 49 ≤ m^2 ∧ m^2 < 343 ∧ M = m - 1) :=
sorry

end find_M_base7_l1515_151571


namespace parabola_translation_l1515_151566

-- Definitions based on the given conditions
def f (x : ℝ) : ℝ := (x - 1) ^ 2 + 5
def g (x : ℝ) : ℝ := x ^ 2 + 2 * x + 3

-- Statement of the translation problem in Lean 4
theorem parabola_translation :
  ∀ x : ℝ, g x = f (x + 2) - 3 := 
sorry

end parabola_translation_l1515_151566


namespace candy_distribution_l1515_151564

-- Define the problem conditions and theorem.

theorem candy_distribution (X : ℕ) (total_pieces : ℕ) (portions : ℕ) 
  (subsequent_more : ℕ) (h_total : total_pieces = 40) 
  (h_portions : portions = 4) 
  (h_subsequent : subsequent_more = 2) 
  (h_eq : X + (X + subsequent_more) + (X + subsequent_more * 2) + (X + subsequent_more * 3) = total_pieces) : 
  X = 7 := 
sorry

end candy_distribution_l1515_151564


namespace fraction_white_surface_area_l1515_151538

-- Definitions for conditions
def larger_cube_side : ℕ := 3
def smaller_cube_count : ℕ := 27
def white_cube_count : ℕ := 19
def black_cube_count : ℕ := 8
def black_corners : Nat := 8
def faces_per_cube : ℕ := 6
def exposed_faces_per_corner : ℕ := 3

-- Theorem statement for proving the fraction of the white surface area
theorem fraction_white_surface_area : (30 : ℚ) / 54 = 5 / 9 :=
by 
  -- Add the proof steps here if necessary
  sorry

end fraction_white_surface_area_l1515_151538


namespace remaining_average_l1515_151547

-- Definitions
def original_average (n : ℕ) (avg : ℝ) := n = 50 ∧ avg = 38
def discarded_numbers (a b : ℝ) := a = 45 ∧ b = 55

-- Proof Statement
theorem remaining_average (n : ℕ) (avg : ℝ) (a b : ℝ) (s : ℝ) :
  original_average n avg →
  discarded_numbers a b →
  s = (n * avg - (a + b)) / (n - 2) →
  s = 37.5 :=
by
  intros h_avg h_discard h_s
  sorry

end remaining_average_l1515_151547


namespace alcohol_percentage_l1515_151565

theorem alcohol_percentage (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 100) 
(h3 : (0.6 + (x / 100) * 6 = 2.4)) : x = 30 :=
by sorry

end alcohol_percentage_l1515_151565


namespace solve_equation_l1515_151591

theorem solve_equation (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^(2 * y - 1) + (x + 1)^(2 * y - 1) = (x + 2)^(2 * y - 1) ↔ (x = 1 ∧ y = 1) := by
  sorry

end solve_equation_l1515_151591
