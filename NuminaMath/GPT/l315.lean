import Mathlib

namespace sum_ratios_l315_31508

variable (a b d : ℕ)

def A_n (a b d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def arithmetic_sum (a n d : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem sum_ratios (k : ℕ) (h1 : 2 * (a + d) = 7 * k) (h2 : 4 * (a + 3 * d) = 6 * k) :
  arithmetic_sum a 7 d / arithmetic_sum a 3 d = 2 / 1 :=
by
  sorry

end sum_ratios_l315_31508


namespace original_number_l315_31597

theorem original_number (x : ℤ) (h : (x + 4) % 23 = 0) : x = 19 :=
sorry

end original_number_l315_31597


namespace moles_of_ca_oh_2_l315_31556

-- Define the chemical reaction
def ca_o := 1
def h_2_o := 1
def ca_oh_2 := ca_o + h_2_o

-- Prove the result of the reaction
theorem moles_of_ca_oh_2 :
  ca_oh_2 = 1 := by sorry

end moles_of_ca_oh_2_l315_31556


namespace compare_magnitudes_l315_31505

noncomputable def log_base_3_of_2 : ℝ := Real.log 2 / Real.log 3   -- def a
noncomputable def ln_2 : ℝ := Real.log 2                          -- def b
noncomputable def five_minus_pi : ℝ := 5 - Real.pi                -- def c

theorem compare_magnitudes :
  let a := log_base_3_of_2
  let b := ln_2
  let c := five_minus_pi
  c < a ∧ a < b :=
by
  sorry

end compare_magnitudes_l315_31505


namespace rice_bag_weight_l315_31576

theorem rice_bag_weight (r f : ℕ) (total_weight : ℕ) (h1 : 20 * r + 50 * f = 2250) (h2 : r = 2 * f) : r = 50 := 
by
  sorry

end rice_bag_weight_l315_31576


namespace acute_angle_probability_correct_l315_31559

noncomputable def acute_angle_probability (n : ℕ) (n_ge_4 : n ≥ 4) : ℝ :=
  (n * (n - 2)) / (2 ^ (n-1))

theorem acute_angle_probability_correct (n : ℕ) (h : n ≥ 4) (P : Fin n → ℝ) -- P represents points on the circle
    (uniformly_distributed : ∀ i, P i ∈ Set.Icc (0 : ℝ) 1) : 
    acute_angle_probability n h = (n * (n - 2)) / (2 ^ (n-1)) := 
  sorry

end acute_angle_probability_correct_l315_31559


namespace computation_l315_31577

theorem computation :
  52 * 46 + 104 * 52 = 7800 := by
  sorry

end computation_l315_31577


namespace area_of_triangle_is_27_over_5_l315_31509

def area_of_triangle_bounded_by_y_axis_and_lines : ℚ :=
  let y_intercept_1 := -2
  let y_intercept_2 := 4
  let base := y_intercept_2 - y_intercept_1
  let x_intersection : ℚ := 9 / 5   -- Calculated using the system of equations
  1 / 2 * base * x_intersection

theorem area_of_triangle_is_27_over_5 :
  area_of_triangle_bounded_by_y_axis_and_lines = 27 / 5 := by
  sorry

end area_of_triangle_is_27_over_5_l315_31509


namespace vector_problem_solution_l315_31536

variables (a b c : ℤ × ℤ) (m n : ℤ)

def parallel (v1 v2 : ℤ × ℤ) : Prop := v1.1 * v2.2 = v1.2 * v2.1
def perpendicular (v1 v2 : ℤ × ℤ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_problem_solution
  (a_eq : a = (1, -2))
  (b_eq : b = (2, m - 1))
  (c_eq : c = (4, n))
  (h1 : parallel a b)
  (h2 : perpendicular b c) :
  m + n = -1 := by
  sorry

end vector_problem_solution_l315_31536


namespace decagon_diagonals_l315_31526

-- The condition for the number of diagonals in a polygon
def number_of_diagonals (n : Nat) : Nat :=
  n * (n - 3) / 2

-- The specific proof statement for a decagon
theorem decagon_diagonals : number_of_diagonals 10 = 35 := by
  -- The proof would go here
  sorry

end decagon_diagonals_l315_31526


namespace al_sandwiches_count_l315_31503

noncomputable def total_sandwiches (bread meat cheese : ℕ) : ℕ :=
  bread * meat * cheese

noncomputable def prohibited_combinations (bread_forbidden_combination cheese_forbidden_combination : ℕ) : ℕ := 
  bread_forbidden_combination + cheese_forbidden_combination

theorem al_sandwiches_count (bread meat cheese : ℕ) 
  (bread_forbidden_combination cheese_forbidden_combination : ℕ) 
  (h1 : bread = 5) 
  (h2 : meat = 7) 
  (h3 : cheese = 6) 
  (h4 : bread_forbidden_combination = 5) 
  (h5 : cheese_forbidden_combination = 6) : 
  total_sandwiches bread meat cheese - prohibited_combinations bread_forbidden_combination cheese_forbidden_combination = 199 :=
by
  sorry

end al_sandwiches_count_l315_31503


namespace min_abs_sum_l315_31523

-- Definitions based on given conditions for the problem
variable (p q r s : ℤ)
variable (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
variable (h : (matrix 2 2 ℤ ![(p, q), (r, s)]) ^ 2 = matrix 2 2 ℤ ![(9, 0), (0, 9)])

-- Statement of the proof problem
theorem min_abs_sum :
  |p| + |q| + |r| + |s| = 8 :=
by
  sorry

end min_abs_sum_l315_31523


namespace percentage_flowering_plants_l315_31588

variable (P : ℝ)

theorem percentage_flowering_plants (h : 5 * (1 / 4) * (P / 100) * 80 = 40) : P = 40 :=
by
  -- This is where the proof would go, but we will use sorry to skip it for now
  sorry

end percentage_flowering_plants_l315_31588


namespace balls_in_boxes_l315_31513

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l315_31513


namespace line_tangent_to_circle_l315_31579

noncomputable def circle_diameter : ℝ := 13
noncomputable def distance_from_center_to_line : ℝ := 6.5

theorem line_tangent_to_circle :
  ∀ (d r : ℝ), d = 13 → r = 6.5 → r = d/2 → distance_from_center_to_line = r → 
  (distance_from_center_to_line = r) := 
by
  intros d r hdiam hdist hradius hdistance
  sorry

end line_tangent_to_circle_l315_31579


namespace solve_expression_l315_31529

noncomputable def given_expression : ℝ :=
  (10 * 1.8 - 2 * 1.5) / 0.3 + 3^(2 / 3) - Real.log 4 + Real.sin (Real.pi / 6) - Real.cos (Real.pi / 4) + Nat.factorial 4 / Nat.factorial 2

theorem solve_expression : given_expression = 59.6862 :=
by
  sorry

end solve_expression_l315_31529


namespace twice_perimeter_of_square_l315_31519

theorem twice_perimeter_of_square (s : ℝ) (h : s^2 = 625) : 2 * 4 * s = 200 :=
by sorry

end twice_perimeter_of_square_l315_31519


namespace parabola_max_value_l315_31557

theorem parabola_max_value 
  (y : ℝ → ℝ) 
  (h : ∀ x, y x = - (x + 1)^2 + 3) : 
  ∃ x, y x = 3 ∧ ∀ x', y x' ≤ 3 :=
by
  sorry

end parabola_max_value_l315_31557


namespace shortest_chord_value_of_m_l315_31514

theorem shortest_chord_value_of_m :
  (∃ m : ℝ,
      (∀ x y : ℝ, mx + y - 2 * m - 1 = 0) ∧
      (∀ x y : ℝ, x ^ 2 + y ^ 2 - 2 * x - 4 * y = 0) ∧
      (mx + y - 2 * m - 1 = 0 → ∃ x y : ℝ, (x, y) = (2, 1))
  ) → m = -1 :=
by
  sorry

end shortest_chord_value_of_m_l315_31514


namespace equation_proof_l315_31553

theorem equation_proof :
  (40 + 5 * 12) / (180 / 3^2) + Real.sqrt 49 = 12 := 
by 
  sorry

end equation_proof_l315_31553


namespace fraction_equivalence_l315_31501

theorem fraction_equivalence : (8 : ℝ) / (5 * 48) = 0.8 / (5 * 0.48) :=
  sorry

end fraction_equivalence_l315_31501


namespace cindy_added_pens_l315_31532

-- Define the initial number of pens
def initial_pens : ℕ := 5

-- Define the number of pens given by Mike
def pens_from_mike : ℕ := 20

-- Define the number of pens given to Sharon
def pens_given_to_sharon : ℕ := 10

-- Define the final number of pens
def final_pens : ℕ := 40

-- Formulate the theorem regarding the pens added by Cindy
theorem cindy_added_pens :
  final_pens = initial_pens + pens_from_mike - pens_given_to_sharon + 25 :=
by
  sorry

end cindy_added_pens_l315_31532


namespace tan_half_alpha_l315_31507

theorem tan_half_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 24 / 25) : Real.tan (α / 2) = 3 / 4 :=
by
  sorry

end tan_half_alpha_l315_31507


namespace trigonometric_identity_l315_31573

theorem trigonometric_identity : (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 1 - Real.sqrt 3 :=
by
  sorry

end trigonometric_identity_l315_31573


namespace hypotenuse_of_right_triangle_l315_31552

theorem hypotenuse_of_right_triangle (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ c^2 = a^2 + b^2 :=
by
  sorry

end hypotenuse_of_right_triangle_l315_31552


namespace necessary_condition_ac_eq_bc_l315_31563

theorem necessary_condition_ac_eq_bc {a b c : ℝ} (hc : c ≠ 0) : (ac = bc ↔ a = b) := by
  sorry

end necessary_condition_ac_eq_bc_l315_31563


namespace num_common_points_l315_31500

-- Definitions of the given conditions:
def line1 (x y : ℝ) := x + 2 * y - 3 = 0
def line2 (x y : ℝ) := 4 * x - y + 1 = 0
def line3 (x y : ℝ) := 2 * x - y - 5 = 0
def line4 (x y : ℝ) := 3 * x + 4 * y - 8 = 0

-- The proof goal:
theorem num_common_points : 
  ∃! p : ℝ × ℝ, (line1 p.1 p.2 ∨ line2 p.1 p.2) ∧ (line3 p.1 p.2 ∨ line4 p.1 p.2) :=
sorry

end num_common_points_l315_31500


namespace cost_of_jeans_l315_31595

    variable (J S : ℝ)

    def condition1 := 3 * J + 6 * S = 104.25
    def condition2 := 4 * J + 5 * S = 112.15

    theorem cost_of_jeans (h1 : condition1 J S) (h2 : condition2 J S) : J = 16.85 := by
      sorry
    
end cost_of_jeans_l315_31595


namespace intersection_A_B_l315_31599

-- Define sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

-- The theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 3} :=
by
  sorry -- proof is skipped as instructed

end intersection_A_B_l315_31599


namespace game_24_set1_game_24_set2_l315_31546

-- Equivalent proof problem for set {3, 2, 6, 7}
theorem game_24_set1 (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 2) (h₃ : c = 6) (h₄ : d = 7) :
  ((d / b) * c + a) = 24 := by
  subst_vars
  sorry

-- Equivalent proof problem for set {3, 4, -6, 10}
theorem game_24_set2 (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = -6) (h₄ : d = 10) :
  ((b + c + d) * a) = 24 := by
  subst_vars
  sorry

end game_24_set1_game_24_set2_l315_31546


namespace quadratic_root_expression_l315_31592

theorem quadratic_root_expression (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 + x - 2023 = 0 → (x = a ∨ x = b)) 
  (ha_neq_b : a ≠ b) :
  a^2 + 2*a + b = 2022 :=
sorry

end quadratic_root_expression_l315_31592


namespace geom_seq_property_l315_31542

noncomputable def a_n : ℕ → ℝ := sorry  -- The definition of the geometric sequence

theorem geom_seq_property (a_n : ℕ → ℝ) (h : a_n 6 + a_n 8 = 4) :
  a_n 8 * (a_n 4 + 2 * a_n 6 + a_n 8) = 16 := by
sorry

end geom_seq_property_l315_31542


namespace point_movement_l315_31516

theorem point_movement (P : ℤ) (hP : P = -5) (k : ℤ) (hk : (k = 3 ∨ k = -3)) :
  P + k = -8 ∨ P + k = -2 :=
by {
  sorry
}

end point_movement_l315_31516


namespace value_of_k_l315_31555

theorem value_of_k (x k : ℝ) (h : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) (hk : k ≠ 0) : k = 8 :=
sorry

end value_of_k_l315_31555


namespace how_many_buckets_did_Eden_carry_l315_31568

variable (E : ℕ) -- Natural number representing buckets Eden carried
variable (M : ℕ) -- Natural number representing buckets Mary carried
variable (I : ℕ) -- Natural number representing buckets Iris carried

-- Conditions based on the problem
axiom Mary_Carry_More : M = E + 3
axiom Iris_Carry_Less : I = M - 1
axiom Total_Buckets : E + M + I = 34

theorem how_many_buckets_did_Eden_carry (h1 : M = E + 3) (h2 : I = M - 1) (h3 : E + M + I = 34) :
  E = 29 / 3 := by
  sorry

end how_many_buckets_did_Eden_carry_l315_31568


namespace find_s_l315_31511

noncomputable def area_of_parallelogram (s : ℝ) : ℝ :=
  (3 * s) * (s * Real.sin (Real.pi / 3))

theorem find_s (s : ℝ) (h1 : area_of_parallelogram s = 27 * Real.sqrt 3) : s = 3 * Real.sqrt 2 := 
  sorry

end find_s_l315_31511


namespace sum_of_interior_angles_l315_31521

theorem sum_of_interior_angles (n : ℕ) 
  (h : 180 * (n - 2) = 3600) :
  180 * (n + 2 - 2) = 3960 ∧ 180 * (n - 2 - 2) = 3240 :=
by
  sorry

end sum_of_interior_angles_l315_31521


namespace village_population_rate_decrease_l315_31589

/--
Village X has a population of 78,000, which is decreasing at a certain rate \( R \) per year.
Village Y has a population of 42,000, which is increasing at the rate of 800 per year.
In 18 years, the population of the two villages will be equal.
We aim to prove that the rate of decrease in population per year for Village X is 1200.
-/
theorem village_population_rate_decrease (R : ℝ) 
  (hx : 78000 - 18 * R = 42000 + 18 * 800) : 
  R = 1200 :=
by
  sorry

end village_population_rate_decrease_l315_31589


namespace kiwi_lemon_relationship_l315_31524

open Nat

-- Define the conditions
def total_fruits : ℕ := 58
def mangoes : ℕ := 18
def pears : ℕ := 10
def pawpaws : ℕ := 12
def lemons_in_last_two_baskets : ℕ := 9

-- Define the question and the proof goal
theorem kiwi_lemon_relationship :
  ∃ (kiwis lemons : ℕ), 
  kiwis = lemons_in_last_two_baskets ∧ 
  lemons = lemons_in_last_two_baskets ∧ 
  kiwis + lemons = total_fruits - (mangoes + pears + pawpaws) :=
sorry

end kiwi_lemon_relationship_l315_31524


namespace kylie_beads_total_l315_31565

def number_necklaces_monday : Nat := 10
def number_necklaces_tuesday : Nat := 2
def number_bracelets_wednesday : Nat := 5
def number_earrings_wednesday : Nat := 7

def beads_per_necklace : Nat := 20
def beads_per_bracelet : Nat := 10
def beads_per_earring : Nat := 5

theorem kylie_beads_total :
  (number_necklaces_monday + number_necklaces_tuesday) * beads_per_necklace + 
  number_bracelets_wednesday * beads_per_bracelet + 
  number_earrings_wednesday * beads_per_earring = 325 := 
by
  sorry

end kylie_beads_total_l315_31565


namespace same_profit_and_loss_selling_price_l315_31541

theorem same_profit_and_loss_selling_price (CP SP : ℝ) (h₁ : CP = 49) (h₂ : (CP - 42) = (SP - CP)) : SP = 56 :=
by 
  sorry

end same_profit_and_loss_selling_price_l315_31541


namespace initial_short_bushes_l315_31548

theorem initial_short_bushes (B : ℕ) (H1 : B + 20 = 57) : B = 37 :=
by
  sorry

end initial_short_bushes_l315_31548


namespace sub_fraction_l315_31585

theorem sub_fraction (a b c d : ℚ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 1) (h4 : d = 6) : (a / b) - (c / d) = 7 / 18 := 
by
  sorry

end sub_fraction_l315_31585


namespace neg_mul_neg_pos_mul_neg_neg_l315_31554

theorem neg_mul_neg_pos (a b : Int) (ha : a < 0) (hb : b < 0) : a * b > 0 :=
sorry

theorem mul_neg_neg : (-1) * (-3) = 3 := 
by
  have h1 : -1 < 0 := by norm_num
  have h2 : -3 < 0 := by norm_num
  have h_pos := neg_mul_neg_pos (-1) (-3) h1 h2
  linarith

end neg_mul_neg_pos_mul_neg_neg_l315_31554


namespace actual_height_of_boy_is_236_l315_31534

-- Define the problem conditions
def average_height (n : ℕ) (avg : ℕ) := n * avg
def incorrect_total_height := average_height 35 180
def correct_total_height := average_height 35 178
def wrong_height := 166
def height_difference := incorrect_total_height - correct_total_height

-- Proving the actual height of the boy whose height was wrongly written
theorem actual_height_of_boy_is_236 : 
  wrong_height + height_difference = 236 := sorry

end actual_height_of_boy_is_236_l315_31534


namespace interest_problem_l315_31550

theorem interest_problem
  (P : ℝ)
  (h : P * 0.04 * 5 = P * 0.05 * 4) : 
  (P * 0.04 * 5) = 20 := 
by 
  sorry

end interest_problem_l315_31550


namespace total_clowns_l315_31539

def num_clown_mobiles : Nat := 5
def clowns_per_mobile : Nat := 28

theorem total_clowns : num_clown_mobiles * clowns_per_mobile = 140 := by
  sorry

end total_clowns_l315_31539


namespace six_nine_op_l315_31540

variable (m n : ℚ)

def op (x y : ℚ) : ℚ := m^2 * x + n * y - 1

theorem six_nine_op :
  (op m n 2 3 = 3) →
  (op m n 6 9 = 11) :=
by
  intro h
  sorry

end six_nine_op_l315_31540


namespace compute_xy_l315_31512

theorem compute_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end compute_xy_l315_31512


namespace minimum_value_proof_l315_31520

noncomputable def minimum_value (x : ℝ) (h : x > 1) : ℝ :=
  (x^2 + x + 1) / (x - 1)

theorem minimum_value_proof : ∃ x : ℝ, x > 1 ∧ minimum_value x (by sorry) = 3 + 2*Real.sqrt 3 :=
sorry

end minimum_value_proof_l315_31520


namespace solve_system_l315_31562

theorem solve_system (x y z a : ℝ) 
  (h1 : x + y + z = a) 
  (h2 : x^2 + y^2 + z^2 = a^2) 
  (h3 : x^3 + y^3 + z^3 = a^3) : 
  (x = 0 ∧ y = 0 ∧ z = a) ∨ 
  (x = 0 ∧ y = a ∧ z = 0) ∨ 
  (x = a ∧ y = 0 ∧ z = 0) := 
sorry

end solve_system_l315_31562


namespace lines_coplanar_iff_k_eq_neg2_l315_31590

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
(2 + s, 4 - k * s, 2 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
(t, 2 + 2 * t, 3 - t)

theorem lines_coplanar_iff_k_eq_neg2 :
  (∃ s t : ℝ, line1 s k = line2 t) → k = -2 :=
by
  sorry

end lines_coplanar_iff_k_eq_neg2_l315_31590


namespace original_price_l315_31593

-- Definitions based on the problem conditions
variables (P : ℝ)

def john_payment (P : ℝ) : ℝ :=
  0.9 * P + 0.15 * P

def jane_payment (P : ℝ) : ℝ :=
  0.9 * P + 0.15 * (0.9 * P)

def price_difference (P : ℝ) : ℝ :=
  john_payment P - jane_payment P

theorem original_price (h : price_difference P = 0.51) : P = 34 := 
by
  sorry

end original_price_l315_31593


namespace sum_of_variables_l315_31598

theorem sum_of_variables (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0) : 
  x + y + z = 2 :=
sorry

end sum_of_variables_l315_31598


namespace find_triples_l315_31510

-- Define the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def power_of_p (p n : ℕ) : Prop := ∃ (k : ℕ), n = p^k

-- Given the conditions
variable (p x y : ℕ)
variable (h_prime : is_prime p)
variable (h_pos_x : x > 0)
variable (h_pos_y : y > 0)

-- The problem statement
theorem find_triples (h1 : power_of_p p (x^(p-1) + y)) (h2 : power_of_p p (x + y^(p-1))) : 
  (p = 3 ∧ x = 2 ∧ y = 5) ∨
  (p = 3 ∧ x = 5 ∧ y = 2) ∨
  (p = 2 ∧ ∃ (n i : ℕ), n > 0 ∧ i > 0 ∧ x = n ∧ y = 2^i - n ∧ 0 < n ∧ n < 2^i) := 
sorry

end find_triples_l315_31510


namespace vertical_asymptote_sum_l315_31517

theorem vertical_asymptote_sum :
  ∀ x y : ℝ, (4 * x^2 + 8 * x + 3 = 0) → (4 * y^2 + 8 * y + 3 = 0) → x ≠ y → x + y = -2 :=
by
  sorry

end vertical_asymptote_sum_l315_31517


namespace sum_infinite_series_l315_31525

theorem sum_infinite_series :
  ∑' n : ℕ, (3 * (n+1) + 2) / ((n+1) * (n+2) * (n+4)) = 29 / 36 :=
by
  sorry

end sum_infinite_series_l315_31525


namespace greening_investment_growth_l315_31582

-- Define initial investment in 2020 and investment in 2022.
def investment_2020 : ℝ := 20000
def investment_2022 : ℝ := 25000

-- Define the average growth rate x
variable (x : ℝ)

-- The mathematically equivalent proof problem:
theorem greening_investment_growth : 
  20 * (1 + x) ^ 2 = 25 :=
sorry

end greening_investment_growth_l315_31582


namespace arithmetic_expression_evaluation_l315_31596

theorem arithmetic_expression_evaluation : (8 / 2 - 3 * 2 + 5^2 / 5) = 3 := by
  sorry

end arithmetic_expression_evaluation_l315_31596


namespace sequence_an_form_sum_cn_terms_l315_31581

theorem sequence_an_form (a_n S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (h : ∀ n : ℕ, a_n n = 3/4 * S_n n + 2) :
  ∀ n : ℕ, b_n n = 2 * n + 1 :=
sorry 

theorem sum_cn_terms (a_n S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (c_n : ℕ → ℕ) (T_n : ℕ → ℕ)
    (h : ∀ n : ℕ, a_n n = 3/4 * S_n n + 2)
    (hb : ∀ n : ℕ, b_n n = 2 * n + 1)
    (hc : ∀ n : ℕ, c_n n = 1 / (b_n n * b_n (n + 1))) :
  ∀ n : ℕ, T_n n = n / (3 * (2 * n + 3)) :=
sorry

end sequence_an_form_sum_cn_terms_l315_31581


namespace initial_fee_is_correct_l315_31504

noncomputable def initial_fee (total_charge : ℝ) (charge_per_segment : ℝ) (segment_length : ℝ) (distance : ℝ) : ℝ :=
  total_charge - (⌊distance / segment_length⌋ * charge_per_segment)

theorem initial_fee_is_correct :
  initial_fee 4.5 0.25 (2/5) 3.6 = 2.25 :=
by 
  sorry

end initial_fee_is_correct_l315_31504


namespace gran_age_indeterminate_l315_31591

theorem gran_age_indeterminate
(gran_age : ℤ) -- Let Gran's age be denoted by gran_age
(guess1 : ℤ := 75) -- The first grandchild guessed 75
(guess2 : ℤ := 78) -- The second grandchild guessed 78
(guess3 : ℤ := 81) -- The third grandchild guessed 81
-- One guess is mistaken by 1 year
(h1 : (abs (gran_age - guess1) = 1) ∨ (abs (gran_age - guess2) = 1) ∨ (abs (gran_age - guess3) = 1))
-- Another guess is mistaken by 2 years
(h2 : (abs (gran_age - guess1) = 2) ∨ (abs (gran_age - guess2) = 2) ∨ (abs (gran_age - guess3) = 2))
-- Another guess is mistaken by 4 years
(h3 : (abs (gran_age - guess1) = 4) ∨ (abs (gran_age - guess2) = 4) ∨ (abs (gran_age - guess3) = 4)) :
  False := sorry

end gran_age_indeterminate_l315_31591


namespace find_multiplier_l315_31545

theorem find_multiplier (N x : ℕ) (h₁ : N = 12) (h₂ : N * x - 3 = (N - 7) * 9) : x = 4 :=
by
  sorry

end find_multiplier_l315_31545


namespace second_number_division_l315_31583

theorem second_number_division (d x r : ℕ) (h1 : d = 16) (h2 : 25 % d = r) (h3 : 105 % d = r) (h4 : r = 9) : x % d = r → x = 41 :=
by 
  simp [h1, h2, h3, h4] 
  sorry

end second_number_division_l315_31583


namespace rectangular_floor_paint_l315_31527

theorem rectangular_floor_paint (a b : ℕ) (ha : a > 0) (hb : b > a) (h1 : a * b = 2 * (a - 4) * (b - 4) + 32) : 
  ∃ pairs : Finset (ℕ × ℕ), pairs.card = 3 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → b > a :=
by 
  sorry

end rectangular_floor_paint_l315_31527


namespace tan_sum_identity_l315_31531

theorem tan_sum_identity :
  Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180) + 
  Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180) = 1 :=
by sorry

end tan_sum_identity_l315_31531


namespace complex_root_seventh_power_l315_31561

theorem complex_root_seventh_power (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^5 - 1) * (r^6 - 1) = 2 := by
  sorry

end complex_root_seventh_power_l315_31561


namespace calculate_dividend_l315_31547

def divisor : ℕ := 21
def quotient : ℕ := 14
def remainder : ℕ := 7
def expected_dividend : ℕ := 301

theorem calculate_dividend : (divisor * quotient + remainder = expected_dividend) := 
by
  sorry

end calculate_dividend_l315_31547


namespace other_type_jelly_amount_l315_31558

-- Combined total amount of jelly
def total_jelly := 6310

-- Amount of one type of jelly
def type_one_jelly := 4518

-- Amount of the other type of jelly
def type_other_jelly := total_jelly - type_one_jelly

theorem other_type_jelly_amount :
  type_other_jelly = 1792 :=
by
  sorry

end other_type_jelly_amount_l315_31558


namespace tamika_greater_probability_l315_31564

-- Definitions for the conditions
def tamika_results : Set ℕ := {11 * 12, 11 * 13, 12 * 13}
def carlos_result : ℕ := 2 + 3 + 4

-- Theorem stating the problem
theorem tamika_greater_probability : 
  (∀ r ∈ tamika_results, r > carlos_result) → (1 : ℚ) = 1 := 
by
  intros h
  sorry

end tamika_greater_probability_l315_31564


namespace arithmetic_sequence_seventh_term_l315_31528

/-- In an arithmetic sequence, the sum of the first three terms is 9 and the third term is 8. 
    Prove that the seventh term is 28. -/
theorem arithmetic_sequence_seventh_term :
  ∃ (a d : ℤ), (a + (a + d) + (a + 2 * d) = 9) ∧ (a + 2 * d = 8) ∧ (a + 6 * d = 28) :=
by
  sorry

end arithmetic_sequence_seventh_term_l315_31528


namespace product_of_integers_l315_31578

theorem product_of_integers (X Y Z W : ℚ) (h_sum : X + Y + Z + W = 100)
  (h_relation : X + 5 = Y - 5 ∧ Y - 5 = 3 * Z ∧ 3 * Z = W / 3) :
  X * Y * Z * W = 29390625 / 256 := by
  sorry

end product_of_integers_l315_31578


namespace hawks_total_points_l315_31543

/-- 
  Define the number of points per touchdown 
  and the number of touchdowns scored by the Hawks. 
-/
def points_per_touchdown : ℕ := 7
def touchdowns : ℕ := 3

/-- 
  Prove that the total number of points the Hawks have is 21. 
-/
theorem hawks_total_points : touchdowns * points_per_touchdown = 21 :=
by
  sorry

end hawks_total_points_l315_31543


namespace reginald_apples_sold_l315_31522

theorem reginald_apples_sold 
  (apple_price : ℝ) 
  (bike_cost : ℝ)
  (repair_percentage : ℝ)
  (remaining_fraction : ℝ)
  (discount_apples : ℕ)
  (free_apples : ℕ)
  (total_apples_sold : ℕ) : 
  apple_price = 1.25 → 
  bike_cost = 80 → 
  repair_percentage = 0.25 → 
  remaining_fraction = 0.2 → 
  discount_apples = 5 → 
  free_apples = 1 → 
  (∃ (E : ℝ), (125 = E ∧ total_apples_sold = 120)) → 
  total_apples_sold = 120 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end reginald_apples_sold_l315_31522


namespace number_of_bowls_l315_31560

theorem number_of_bowls (n : ℕ) (h : 8 * 12 = 96) (avg_increase : 6 * n = 96) : n = 16 :=
by {
  sorry
}

end number_of_bowls_l315_31560


namespace avg_annual_growth_rate_l315_31586

variable (x : ℝ)

/-- Initial GDP in 2020 is 43903.89 billion yuan and GDP in 2022 is 53109.85 billion yuan. 
    Prove that the average annual growth rate x satisfies the equation 43903.89 * (1 + x)^2 = 53109.85 -/
theorem avg_annual_growth_rate (x : ℝ) :
  43903.89 * (1 + x)^2 = 53109.85 :=
sorry

end avg_annual_growth_rate_l315_31586


namespace extra_large_yellow_curlers_l315_31569

def total_curlers : ℕ := 120
def small_pink_curlers : ℕ := total_curlers / 5
def medium_blue_curlers : ℕ := 2 * small_pink_curlers
def large_green_curlers : ℕ := total_curlers / 4

theorem extra_large_yellow_curlers : 
  total_curlers - small_pink_curlers - medium_blue_curlers - large_green_curlers = 18 :=
by
  sorry

end extra_large_yellow_curlers_l315_31569


namespace even_func_min_value_l315_31594

theorem even_func_min_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_neq_a : a ≠ 1) (h_neq_b : b ≠ 1) (h_even : ∀ x : ℝ, a^x + b^x = a^(-x) + b^(-x)) :
  ab = 1 → (∃ y : ℝ, y = (1 / a + 4 / b) ∧ y = 4) :=
by
  sorry

end even_func_min_value_l315_31594


namespace real_solution_2015x_equation_l315_31575

theorem real_solution_2015x_equation (k : ℝ) :
  (∃ x : ℝ, (4 * 2015^x - 2015^(-x)) / (2015^x - 3 * 2015^(-x)) = k) ↔ (k < 1/3 ∨ k > 4) := 
by sorry

end real_solution_2015x_equation_l315_31575


namespace julian_owes_jenny_l315_31571

-- Define the initial debt and the additional borrowed amount
def initial_debt : ℕ := 20
def additional_borrowed : ℕ := 8

-- Define the total debt
def total_debt : ℕ := initial_debt + additional_borrowed

-- Statement of the problem: Prove that total_debt equals 28
theorem julian_owes_jenny : total_debt = 28 :=
by
  sorry

end julian_owes_jenny_l315_31571


namespace part1_part2_l315_31580

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x - (a - 1) / x

theorem part1 (a : ℝ) (x : ℝ) (h1 : a ≥ 1) (h2 : x > 0) : f a x ≤ -1 :=
sorry

theorem part2 (a : ℝ) (θ : ℝ) (h1 : a ≥ 1) (h2 : 0 ≤ θ) (h3 : θ ≤ Real.pi / 2) : 
  f a (1 - Real.sin θ) ≤ f a (1 + Real.sin θ) :=
sorry

end part1_part2_l315_31580


namespace find_m_l315_31538

theorem find_m (m : ℤ) (y : ℤ) : 
  (y^2 + m * y + 2) % (y - 1) = (m + 3) ∧ 
  (y^2 + m * y + 2) % (y + 1) = (3 - m) ∧
  (m + 3 = 3 - m) → m = 0 :=
sorry

end find_m_l315_31538


namespace circle_center_l315_31566

theorem circle_center : ∃ (a b : ℝ), (∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y - 4 = 0 ↔ (x - a)^2 + (y - b)^2 = 9) ∧ a = 1 ∧ b = 2 :=
sorry

end circle_center_l315_31566


namespace find_v5_l315_31570

noncomputable def sequence (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 3 * v (n + 1) + v n + 1

theorem find_v5 :
  ∃ (v : ℕ → ℝ), sequence v ∧ v 3 = 11 ∧ v 6 = 242 ∧ v 5 = 73.5 :=
by
  sorry

end find_v5_l315_31570


namespace abs_neg_four_squared_plus_six_l315_31574

theorem abs_neg_four_squared_plus_six : |(-4^2 + 6)| = 10 := by
  -- We skip the proof steps according to the instruction
  sorry

end abs_neg_four_squared_plus_six_l315_31574


namespace factorize_expression_l315_31518

theorem factorize_expression (x : ℝ) : x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 :=
by sorry

end factorize_expression_l315_31518


namespace coffee_merchant_mixture_price_l315_31535

theorem coffee_merchant_mixture_price
  (c1 c2 : ℝ) (w1 w2 total_cost mixture_price : ℝ)
  (h_c1 : c1 = 9)
  (h_c2 : c2 = 12)
  (h_w1w2 : w1 = 25 ∧ w2 = 25)
  (h_total_weight : w1 + w2 = 100)
  (h_total_cost : total_cost = w1 * c1 + w2 * c2)
  (h_mixture_price : mixture_price = total_cost / (w1 + w2)) :
  mixture_price = 5.25 :=
by sorry

end coffee_merchant_mixture_price_l315_31535


namespace cafe_location_l315_31502

-- Definition of points and conditions
structure Point where
  x : ℤ
  y : ℚ

def mark : Point := { x := 1, y := 8 }
def sandy : Point := { x := -5, y := 0 }

-- The problem statement
theorem cafe_location :
  ∃ cafe : Point, cafe.x = -3 ∧ cafe.y = 8/3 := by
  sorry

end cafe_location_l315_31502


namespace cost_price_decrease_proof_l315_31530

theorem cost_price_decrease_proof (x y : ℝ) (a : ℝ) (h1 : y - x = x * a / 100)
    (h2 : y = (1 + a / 100) * x)
    (h3 : y - 0.9 * x = (0.9 * x * a / 100) + 0.9 * x * 20 / 100) : a = 80 :=
  sorry

end cost_price_decrease_proof_l315_31530


namespace find_x_squared_plus_inverse_squared_l315_31533

theorem find_x_squared_plus_inverse_squared (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^2 + (1 / x)^2 = 7 :=
by
  sorry

end find_x_squared_plus_inverse_squared_l315_31533


namespace Mr_Kishore_saved_10_percent_l315_31551

-- Define the costs and savings
def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 6100
def savings : ℕ := 2400

-- Define the total expenses
def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

-- Define the total monthly salary
def total_monthly_salary : ℕ := total_expenses + savings

-- Define the percentage saved
def percentage_saved : ℕ := (savings * 100) / total_monthly_salary

-- The statement to prove
theorem Mr_Kishore_saved_10_percent : percentage_saved = 10 := by
  sorry

end Mr_Kishore_saved_10_percent_l315_31551


namespace fraction_evaluation_l315_31506

theorem fraction_evaluation : (1 / (2 + 1 / (3 + 1 / 4))) = (13 / 30) := by
  sorry

end fraction_evaluation_l315_31506


namespace first_number_in_a10_l315_31515

-- Define a function that captures the sequence of the first number in each sum 'a_n'.
def first_in_an (n : ℕ) : ℕ :=
  1 + 2 * (n * (n - 1)) / 2 

-- State the theorem we want to prove
theorem first_number_in_a10 : first_in_an 10 = 91 := 
  sorry

end first_number_in_a10_l315_31515


namespace non_deg_ellipse_condition_l315_31549

theorem non_deg_ellipse_condition (k : ℝ) : k > -19 ↔ 
  (∃ x y : ℝ, 3 * x^2 + 7 * y^2 - 12 * x + 14 * y = k) :=
sorry

end non_deg_ellipse_condition_l315_31549


namespace water_percentage_in_tomato_juice_l315_31572

-- Definitions from conditions
def tomato_juice_volume := 80 -- in liters
def tomato_puree_volume := 10 -- in liters
def tomato_puree_water_percentage := 20 -- in percent (20%)

-- Need to prove percentage of water in tomato juice is 20%
theorem water_percentage_in_tomato_juice : 
  (100 - tomato_puree_water_percentage) * tomato_puree_volume / tomato_juice_volume = 20 :=
by
  -- Skip the proof
  sorry

end water_percentage_in_tomato_juice_l315_31572


namespace best_fitting_regression_line_l315_31584

theorem best_fitting_regression_line
  (R2_A : ℝ) (R2_B : ℝ) (R2_C : ℝ) (R2_D : ℝ)
  (h_A : R2_A = 0.27)
  (h_B : R2_B = 0.85)
  (h_C : R2_C = 0.96)
  (h_D : R2_D = 0.5) :
  R2_C = 0.96 :=
by
  -- Proof goes here
  sorry

end best_fitting_regression_line_l315_31584


namespace solve_for_a_l315_31537

theorem solve_for_a (a x : ℝ) (h : x = 3) (eqn : a * x - 5 = x + 1) : a = 3 :=
by
  -- proof omitted
  sorry

end solve_for_a_l315_31537


namespace remainder_when_dividing_by_y_minus_4_l315_31544

def g (y : ℤ) : ℤ := y^5 - 8 * y^4 + 12 * y^3 + 25 * y^2 - 40 * y + 24

theorem remainder_when_dividing_by_y_minus_4 : g 4 = 8 :=
by
  sorry

end remainder_when_dividing_by_y_minus_4_l315_31544


namespace not_p_equiv_exists_leq_sin_l315_31587

-- Define the conditions as a Lean proposition
def p : Prop := ∀ x : ℝ, x > Real.sin x

-- State the problem as a theorem to be proved
theorem not_p_equiv_exists_leq_sin : ¬p = ∃ x : ℝ, x ≤ Real.sin x := 
by sorry

end not_p_equiv_exists_leq_sin_l315_31587


namespace remainder_1234567_div_256_l315_31567

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end remainder_1234567_div_256_l315_31567
