import Mathlib

namespace triangle_area_is_9sqrt2_l2026_202639

noncomputable def triangle_area_with_given_medians_and_angle (CM BN : ℝ) (angle_BKM : ℝ) : ℝ :=
  let centroid_division_ratio := (2.0 / 3.0)
  let BK := centroid_division_ratio * BN
  let MK := (1.0 / 3.0) * CM
  let area_BKM := (1.0 / 2.0) * BK * MK * Real.sin angle_BKM
  6.0 * area_BKM

theorem triangle_area_is_9sqrt2 :
  triangle_area_with_given_medians_and_angle 6 4.5 (Real.pi / 4) = 9 * Real.sqrt 2 :=
by
  sorry

end triangle_area_is_9sqrt2_l2026_202639


namespace maria_walk_to_school_l2026_202643

variable (w s : ℝ)

theorem maria_walk_to_school (h1 : 25 * w + 13 * s = 38) (h2 : 11 * w + 20 * s = 31) : 
  51 = 51 := by
  sorry

end maria_walk_to_school_l2026_202643


namespace distance_problem_l2026_202617

noncomputable def distance_point_to_plane 
  (x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ) : ℝ :=
  -- Equation of the plane passing through three points derived using determinants
  let a := x2 - x1
  let b := y2 - y1
  let c := z2 - z1
  let d := x3 - x1
  let e := y3 - y1
  let f := z3 - z1
  let A := b*f - c*e
  let B := c*d - a*f
  let C := a*e - b*d
  let D := -(A*x1 + B*y1 + C*z1)
  -- Distance from the given point to the above plane
  (|A*x0 + B*y0 + C*z0 + D|) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_problem :
  distance_point_to_plane 
  3 6 68 
  (-3) (-5) 6 
  2 1 (-4) 
  0 (-3) (-1) 
  = Real.sqrt 573 :=
by sorry

end distance_problem_l2026_202617


namespace CoinRun_ProcGen_ratio_l2026_202608

theorem CoinRun_ProcGen_ratio
  (greg_ppo_reward: ℝ)
  (maximum_procgen_reward: ℝ)
  (ppo_ratio: ℝ)
  (maximum_coinrun_reward: ℝ)
  (coinrun_to_procgen_ratio: ℝ)
  (greg_ppo_reward_eq: greg_ppo_reward = 108)
  (maximum_procgen_reward_eq: maximum_procgen_reward = 240)
  (ppo_ratio_eq: ppo_ratio = 0.90)
  (coinrun_equation: maximum_coinrun_reward = greg_ppo_reward / ppo_ratio)
  (ratio_definition: coinrun_to_procgen_ratio = maximum_coinrun_reward / maximum_procgen_reward) :
  coinrun_to_procgen_ratio = 0.5 :=
sorry

end CoinRun_ProcGen_ratio_l2026_202608


namespace latest_time_for_temperature_at_60_l2026_202697

theorem latest_time_for_temperature_at_60
  (t : ℝ) (h : -t^2 + 10 * t + 40 = 60) : t = 12 :=
sorry

end latest_time_for_temperature_at_60_l2026_202697


namespace fraction_of_x_by_110_l2026_202668

theorem fraction_of_x_by_110 (x : ℝ) (f : ℝ) (h1 : 0.6 * x = f * x + 110) (h2 : x = 412.5) : f = 1 / 3 :=
by 
  sorry

end fraction_of_x_by_110_l2026_202668


namespace max_non_managers_l2026_202689

theorem max_non_managers (n_mngrs n_non_mngrs : ℕ) (hmngrs : n_mngrs = 8) 
                (h_ratio : (5 : ℚ) / 24 < (n_mngrs : ℚ) / n_non_mngrs) :
                n_non_mngrs ≤ 38 :=
by {
  sorry
}

end max_non_managers_l2026_202689


namespace yard_fraction_occupied_by_flowerbeds_l2026_202630

noncomputable def rectangular_yard_area (length width : ℕ) : ℕ :=
  length * width

noncomputable def triangle_area (leg_length : ℕ) : ℕ :=
  2 * (1 / 2 * leg_length ^ 2)

theorem yard_fraction_occupied_by_flowerbeds :
  let length := 30
  let width := 7
  let parallel_side_short := 20
  let parallel_side_long := 30
  let flowerbed_leg := 7
  rectangular_yard_area length width ≠ 0 ∧
  triangle_area flowerbed_leg * 2 = 49 →
  (triangle_area flowerbed_leg * 2) / rectangular_yard_area length width = 7 / 30 :=
sorry

end yard_fraction_occupied_by_flowerbeds_l2026_202630


namespace complex_number_conditions_l2026_202695

open Complex Real

noncomputable def is_real (a : ℝ) : Prop :=
a ^ 2 - 2 * a - 15 = 0

noncomputable def is_imaginary (a : ℝ) : Prop :=
a ^ 2 - 2 * a - 15 ≠ 0

noncomputable def is_purely_imaginary (a : ℝ) : Prop :=
a ^ 2 - 9 = 0 ∧ a ^ 2 - 2 * a - 15 ≠ 0

theorem complex_number_conditions (a : ℝ) :
  (is_real a ↔ (a = 5 ∨ a = -3))
  ∧ (is_imaginary a ↔ (a ≠ 5 ∧ a ≠ -3))
  ∧ (¬(∃ a : ℝ, is_purely_imaginary a)) :=
by
  sorry

end complex_number_conditions_l2026_202695


namespace pies_not_eaten_with_forks_l2026_202656

variables (apple_pe_forked peach_pe_forked cherry_pe_forked chocolate_pe_forked lemon_pe_forked : ℤ)
variables (total_pies types_of_pies : ℤ)

def pies_per_type (total_pies types_of_pies : ℤ) : ℤ :=
  total_pies / types_of_pies

def not_eaten_with_forks (percentage_forked : ℤ) (pies : ℤ) : ℤ :=
  pies - (pies * percentage_forked) / 100

noncomputable def apple_not_forked  := not_eaten_with_forks apple_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def peach_not_forked  := not_eaten_with_forks peach_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def cherry_not_forked := not_eaten_with_forks cherry_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def chocolate_not_forked := not_eaten_with_forks chocolate_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def lemon_not_forked := not_eaten_with_forks lemon_pe_forked (pies_per_type total_pies types_of_pies)

theorem pies_not_eaten_with_forks :
  (apple_not_forked = 128) ∧
  (peach_not_forked = 112) ∧
  (cherry_not_forked = 84) ∧
  (chocolate_not_forked = 76) ∧
  (lemon_not_forked = 140) :=
by sorry

end pies_not_eaten_with_forks_l2026_202656


namespace number_of_draws_l2026_202622

-- Definition of the competition conditions
def competition_conditions (A B C D E : ℕ) : Prop :=
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧
  (A = B ∨ B = C ∨ C = D ∨ D = E) ∧
  15 ∣ (10000 * A + 1000 * B + 100 * C + 10 * D + E)

-- The main theorem stating the number of draws
theorem number_of_draws :
  ∃ (A B C D E : ℕ), competition_conditions A B C D E ∧ 
  (∃ (draws : ℕ), draws = 3) :=
by
  sorry

end number_of_draws_l2026_202622


namespace range_a_for_inequality_l2026_202624

theorem range_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, (a-2) * x^2 - 2 * (a-2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
by
  sorry

end range_a_for_inequality_l2026_202624


namespace x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842_l2026_202684

theorem x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842
  (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 :=
sorry

end x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842_l2026_202684


namespace x_intercept_correct_l2026_202688

noncomputable def x_intercept_of_line : ℝ × ℝ :=
if h : (-4 : ℝ) ≠ 0 then (24 / (-4), 0) else (0, 0)

theorem x_intercept_correct : x_intercept_of_line = (-6, 0) := by
  -- proof will be given here
  sorry

end x_intercept_correct_l2026_202688


namespace smaller_angle_at_seven_oclock_l2026_202699

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end smaller_angle_at_seven_oclock_l2026_202699


namespace surface_area_of_prism_l2026_202649

theorem surface_area_of_prism (l w h : ℕ)
  (h_internal_volume : l * w * h = 24)
  (h_external_volume : (l + 2) * (w + 2) * (h + 2) = 120) :
  2 * ((l + 2) * (w + 2) + (w + 2) * (h + 2) + (h + 2) * (l + 2)) = 148 :=
by
  sorry

end surface_area_of_prism_l2026_202649


namespace absolute_value_inequality_solution_set_l2026_202628

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 * x - 1| - |x - 2| < 0} = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end absolute_value_inequality_solution_set_l2026_202628


namespace union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_l2026_202675

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {4, 5, 6, 7, 8, 9}
def B : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem union_of_A_and_B : A ∪ B = U := by
  sorry

theorem intersection_of_A_and_B : A ∩ B = {4, 5, 6} := by
  sorry

theorem complement_of_intersection : U \ (A ∩ B) = {1, 2, 3, 7, 8, 9} := by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_l2026_202675


namespace amount_with_r_l2026_202611

theorem amount_with_r (p q r T : ℝ) 
  (h1 : p + q + r = 4000)
  (h2 : r = (2/3) * T)
  (h3 : T = p + q) : 
  r = 1600 := by
  sorry

end amount_with_r_l2026_202611


namespace largest_difference_l2026_202634

theorem largest_difference (P Q R S T U : ℕ) 
    (hP : P = 3 * 2003 ^ 2004)
    (hQ : Q = 2003 ^ 2004)
    (hR : R = 2002 * 2003 ^ 2003)
    (hS : S = 3 * 2003 ^ 2003)
    (hT : T = 2003 ^ 2003)
    (hU : U = 2003 ^ 2002) 
    : max (P - Q) (max (Q - R) (max (R - S) (max (S - T) (T - U)))) = P - Q :=
sorry

end largest_difference_l2026_202634


namespace beadshop_profit_on_wednesday_l2026_202681

theorem beadshop_profit_on_wednesday (total_profit profit_on_monday profit_on_tuesday profit_on_wednesday : ℝ)
  (h1 : total_profit = 1200)
  (h2 : profit_on_monday = total_profit / 3)
  (h3 : profit_on_tuesday = total_profit / 4)
  (h4 : profit_on_wednesday = total_profit - profit_on_monday - profit_on_tuesday) :
  profit_on_wednesday = 500 := 
sorry

end beadshop_profit_on_wednesday_l2026_202681


namespace james_marbles_left_l2026_202655

theorem james_marbles_left :
  ∀ (initial_marbles bags remaining_bags marbles_per_bag left_marbles : ℕ),
  initial_marbles = 28 →
  bags = 4 →
  marbles_per_bag = initial_marbles / bags →
  remaining_bags = bags - 1 →
  left_marbles = remaining_bags * marbles_per_bag →
  left_marbles = 21 :=
by
  intros initial_marbles bags remaining_bags marbles_per_bag left_marbles
  sorry

end james_marbles_left_l2026_202655


namespace gumball_difference_l2026_202677

theorem gumball_difference :
  ∀ C : ℕ, 19 ≤ (29 + C) / 3 ∧ (29 + C) / 3 ≤ 25 →
  (46 - 28) = 18 :=
by
  intros C h
  sorry

end gumball_difference_l2026_202677


namespace inscribed_square_proof_l2026_202654

theorem inscribed_square_proof :
  (∃ (r : ℝ), 2 * π * r = 72 * π ∧ r = 36) ∧ 
  (∃ (s : ℝ), (2 * (36:ℝ))^2 = 2 * s ^ 2 ∧ s = 36 * Real.sqrt 2) :=
by
  sorry

end inscribed_square_proof_l2026_202654


namespace ked_ben_eggs_ratio_l2026_202609

theorem ked_ben_eggs_ratio 
  (saly_needs_ben_weekly_ratio : ℕ)
  (weeks_in_month : ℕ := 4) 
  (total_production_month : ℕ := 124)
  (saly_needs_weekly : ℕ := 10) 
  (ben_needs_weekly : ℕ := 14)
  (ben_needs_monthly : ℕ := ben_needs_weekly * weeks_in_month)
  (saly_needs_monthly : ℕ := saly_needs_weekly * weeks_in_month)
  (total_saly_ben_monthly : ℕ := saly_needs_monthly + ben_needs_monthly)
  (ked_needs_monthly : ℕ := total_production_month - total_saly_ben_monthly)
  (ked_needs_weekly : ℕ := ked_needs_monthly / weeks_in_month) :
  ked_needs_weekly / ben_needs_weekly = 1 / 2 :=
sorry

end ked_ben_eggs_ratio_l2026_202609


namespace min_a2_plus_b2_l2026_202669

theorem min_a2_plus_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 :=
sorry

end min_a2_plus_b2_l2026_202669


namespace trig_identity_l2026_202607

theorem trig_identity (α : ℝ) (h : Real.sin (π + α) = 1 / 2) : Real.cos (α - 3 / 2 * π) = 1 / 2 :=
  sorry

end trig_identity_l2026_202607


namespace find_triples_l2026_202602

theorem find_triples (k : ℕ) (hk : 0 < k) :
  ∃ (a b c : ℕ), 
    (0 < a ∧ 0 < b ∧ 0 < c) ∧
    (a + b + c = 3 * k + 1) ∧ 
    (a * b + b * c + c * a = 3 * k^2 + 2 * k) ∧ 
    (a = k + 1 ∧ b = k ∧ c = k) :=
by
  sorry

end find_triples_l2026_202602


namespace eq_fraction_l2026_202659

def f(x : ℤ) : ℤ := 3 * x + 4
def g(x : ℤ) : ℤ := 2 * x - 1

theorem eq_fraction : (f (g (f 3))) / (g (f (g 3))) = 79 / 37 := by
  sorry

end eq_fraction_l2026_202659


namespace ratio_B_to_A_l2026_202661

def work_together_rate : Real := 0.75
def days_for_A : Real := 4

theorem ratio_B_to_A : 
  ∃ (days_for_B : Real), 
    (1/days_for_A + 1/days_for_B = work_together_rate) → 
    (days_for_B / days_for_A = 0.5) :=
by 
  sorry

end ratio_B_to_A_l2026_202661


namespace remainder_invariance_l2026_202692

theorem remainder_invariance (S A K : ℤ) (h : ∃ B r : ℤ, S = A * B + r ∧ 0 ≤ r ∧ r < |A|) :
  (∃ B' r' : ℤ, S + A * K = A * B' + r' ∧ r' = r) ∧ (∃ B'' r'' : ℤ, S - A * K = A * B'' + r'' ∧ r'' = r) :=
by
  sorry

end remainder_invariance_l2026_202692


namespace train_speed_in_kmph_l2026_202682

-- Definitions for the given problem conditions
def length_of_train : ℝ := 110
def length_of_bridge : ℝ := 240
def time_to_cross_bridge : ℝ := 20.99832013438925

-- Main theorem statement
theorem train_speed_in_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = 60.0084 := 
by
  sorry

end train_speed_in_kmph_l2026_202682


namespace distance_between_city_A_and_B_is_180_l2026_202606

theorem distance_between_city_A_and_B_is_180
  (D : ℝ)
  (h1 : ∀ T_C : ℝ, T_C = D / 30)
  (h2 : ∀ T_D : ℝ, T_D = T_C - 1)
  (h3 : ∀ V_D : ℝ, V_D > 36 → T_D = D / V_D) :
  D = 180 := 
by
  sorry

end distance_between_city_A_and_B_is_180_l2026_202606


namespace find_p_l2026_202627

theorem find_p :
  ∀ r s : ℝ, (3 * r^2 + 4 * r + 2 = 0) → (3 * s^2 + 4 * s + 2 = 0) →
  (∀ p q : ℝ, (p = - (1/(r^2)) - (1/(s^2))) → (p = -1)) :=
by 
  intros r s hr hs p q hp
  sorry

end find_p_l2026_202627


namespace complex_product_conjugate_l2026_202652

theorem complex_product_conjugate : (1 + Complex.I) * (1 - Complex.I) = 2 := 
by 
  -- Lean proof goes here
  sorry

end complex_product_conjugate_l2026_202652


namespace compute_expression_value_l2026_202632

theorem compute_expression_value (x y : ℝ) (hxy : x ≠ y) 
  (h : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (xy + 1)) :
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (xy + 1) = 2 :=
by
  sorry

end compute_expression_value_l2026_202632


namespace combined_transformation_matrix_l2026_202665

-- Definitions for conditions
def dilation_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![s, 0], ![0, s]]

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

-- Theorem to be proven
theorem combined_transformation_matrix :
  (rotation_matrix_90_ccw * dilation_matrix 4) = ![![0, -4], ![4, 0]] :=
by
  sorry

end combined_transformation_matrix_l2026_202665


namespace three_digit_factorions_l2026_202672

def is_factorion (n : ℕ) : Prop :=
  let digits := (n / 100, (n % 100) / 10, n % 10)
  let (a, b, c) := digits
  n = Nat.factorial a + Nat.factorial b + Nat.factorial c

theorem three_digit_factorions : ∀ n : ℕ, (100 ≤ n ∧ n < 1000) → is_factorion n → n = 145 :=
by
  sorry

end three_digit_factorions_l2026_202672


namespace intersection_point_l2026_202663

variable (x y : ℝ)

theorem intersection_point :
  (y = 9 / (x^2 + 3)) →
  (x + y = 3) →
  (x = 0) := by
  intros h1 h2
  sorry

end intersection_point_l2026_202663


namespace cookies_in_each_batch_l2026_202678

theorem cookies_in_each_batch (batches : ℕ) (people : ℕ) (consumption_per_person : ℕ) (cookies_per_dozen : ℕ) 
  (total_batches : batches = 4) 
  (total_people : people = 16) 
  (cookies_per_person : consumption_per_person = 6) 
  (dozen_size : cookies_per_dozen = 12) :
  (6 * 16) / 4 / 12 = 2 := 
by {
  sorry
}

end cookies_in_each_batch_l2026_202678


namespace vector_identity_l2026_202638

-- Definitions of the vectors
variables {V : Type*} [AddGroup V]

-- Conditions as Lean definitions
def cond1 (AB BO AO : V) : Prop := AB + BO = AO
def cond2 (AO OM AM : V) : Prop := AO + OM = AM
def cond3 (AM MB AB : V) : Prop := AM + MB = AB

-- The main statement to be proved
theorem vector_identity (AB MB BO BC OM AO AM AC : V) 
  (h1 : cond1 AB BO AO) 
  (h2 : cond2 AO OM AM) 
  (h3 : cond3 AM MB AB) 
  : (AB + MB) + (BO + BC) + OM = AC :=
sorry

end vector_identity_l2026_202638


namespace Mark_paid_total_cost_l2026_202686

def length_of_deck : ℝ := 30
def width_of_deck : ℝ := 40
def cost_per_sq_ft_without_sealant : ℝ := 3
def additional_cost_per_sq_ft_sealant : ℝ := 1

def area (length width : ℝ) : ℝ := length * width
def total_cost (area cost_without_sealant cost_sealant : ℝ) : ℝ := 
  area * cost_without_sealant + area * cost_sealant

theorem Mark_paid_total_cost :
  total_cost (area length_of_deck width_of_deck) cost_per_sq_ft_without_sealant additional_cost_per_sq_ft_sealant = 4800 := 
by
  -- Placeholder for proof
  sorry

end Mark_paid_total_cost_l2026_202686


namespace inequality_k_ge_2_l2026_202679

theorem inequality_k_ge_2 {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) (k : ℤ) (h_k : k ≥ 2) :
  (a^k / (a + b) + b^k / (b + c) + c^k / (c + a)) ≥ 3 / 2 :=
by
  sorry

end inequality_k_ge_2_l2026_202679


namespace xy_sum_l2026_202636

theorem xy_sum (x y : ℝ) (h1 : x^3 + 6 * x^2 + 16 * x = -15) (h2 : y^3 + 6 * y^2 + 16 * y = -17) : x + y = -4 :=
by
  -- The proof can be skipped with 'sorry'
  sorry

end xy_sum_l2026_202636


namespace trajectory_is_straight_line_l2026_202680

theorem trajectory_is_straight_line (x y : ℝ) (h : x + y = 0) : ∃ m b : ℝ, y = m * x + b :=
by
  use -1
  use 0
  sorry

end trajectory_is_straight_line_l2026_202680


namespace topsoil_cost_l2026_202615

theorem topsoil_cost
  (cost_per_cubic_foot : ℕ)
  (volume_cubic_yards : ℕ)
  (conversion_factor : ℕ)
  (volume_cubic_feet : ℕ := volume_cubic_yards * conversion_factor)
  (total_cost : ℕ := volume_cubic_feet * cost_per_cubic_foot)
  (cost_per_cubic_foot_def : cost_per_cubic_foot = 8)
  (volume_cubic_yards_def : volume_cubic_yards = 8)
  (conversion_factor_def : conversion_factor = 27) :
  total_cost = 1728 := by
  sorry

end topsoil_cost_l2026_202615


namespace solve_expression_l2026_202640

theorem solve_expression (x : ℝ) (h : 3 * x - 5 = 10 * x + 9) : 4 * (x + 7) = 20 :=
by
  sorry

end solve_expression_l2026_202640


namespace digimon_pack_price_l2026_202667

-- Defining the given conditions as Lean variables
variables (total_spent baseball_cost : ℝ)
variables (packs_of_digimon : ℕ)

-- Setting given values from the problem
def keith_total_spent : total_spent = 23.86 := sorry
def baseball_deck_cost : baseball_cost = 6.06 := sorry
def number_of_digimon_packs : packs_of_digimon = 4 := sorry

-- Stating the main theorem/problem to prove
theorem digimon_pack_price 
  (h1 : total_spent = 23.86)
  (h2 : baseball_cost = 6.06)
  (h3 : packs_of_digimon = 4) : 
  ∃ (price_per_pack : ℝ), price_per_pack = 4.45 :=
sorry

end digimon_pack_price_l2026_202667


namespace intersection_of_sets_eq_l2026_202610

noncomputable def set_intersection (M N : Set ℝ): Set ℝ :=
  {x | x ∈ M ∧ x ∈ N}

theorem intersection_of_sets_eq :
  let M := {x : ℝ | -2 < x ∧ x < 2}
  let N := {x : ℝ | x^2 - 2 * x - 3 < 0}
  set_intersection M N = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end intersection_of_sets_eq_l2026_202610


namespace shaded_area_correct_l2026_202635

-- Definitions of the given conditions
def first_rectangle_length : ℕ := 8
def first_rectangle_width : ℕ := 5
def second_rectangle_length : ℕ := 4
def second_rectangle_width : ℕ := 9
def overlapping_area : ℕ := 3

def first_rectangle_area := first_rectangle_length * first_rectangle_width
def second_rectangle_area := second_rectangle_length * second_rectangle_width

-- Problem statement in Lean 4
theorem shaded_area_correct :
  first_rectangle_area + second_rectangle_area - overlapping_area = 73 :=
by
  -- The proof is skipped
  sorry

end shaded_area_correct_l2026_202635


namespace apple_tree_total_production_l2026_202690

-- Definitions for conditions
def first_year_production : ℕ := 40
def second_year_production : ℕ := 2 * first_year_production + 8
def third_year_production : ℕ := second_year_production - second_year_production / 4

-- Theorem statement
theorem apple_tree_total_production :
  first_year_production + second_year_production + third_year_production = 194 :=
by
  sorry

end apple_tree_total_production_l2026_202690


namespace part_i_part_ii_l2026_202650

-- Define the variables and conditions
variable (a b : ℝ)
variable (h₁ : a > 0)
variable (h₂ : b > 0)
variable (h₃ : a + b = 1 / a + 1 / b)

-- Prove the first part: a + b ≥ 2
theorem part_i : a + b ≥ 2 := by
  sorry

-- Prove the second part: It is impossible for both a² + a < 2 and b² + b < 2 simultaneously
theorem part_ii : ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end part_i_part_ii_l2026_202650


namespace complement_of_65_degrees_l2026_202625

def angle_complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_65_degrees : angle_complement 65 = 25 := by
  -- Proof would follow here, but it's omitted since 'sorry' is added.
  sorry

end complement_of_65_degrees_l2026_202625


namespace find_integers_l2026_202604

-- Problem statement rewritten as a Lean 4 definition
theorem find_integers (a b c : ℤ) (H1 : a = 1) (H2 : b = 2) (H3 : c = 1) : 
  a^2 + b^2 + c^2 + 3 < a * b + 3 * b + 2 * c :=
by
  -- The proof will be presented here
  sorry

end find_integers_l2026_202604


namespace necessary_but_not_sufficient_l2026_202626

-- Define that for all x in ℝ, x^2 - 4x + 2m ≥ 0
def proposition_p (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 - 4 * x + 2 * m ≥ 0

-- Main theorem statement
theorem necessary_but_not_sufficient (m : ℝ) : 
  (proposition_p m → m ≥ 2) → (m ≥ 1 → m ≥ 2) :=
by
  intros h1 h2
  sorry

end necessary_but_not_sufficient_l2026_202626


namespace find_first_number_l2026_202637

theorem find_first_number 
  (second_number : ℕ)
  (increment : ℕ)
  (final_number : ℕ)
  (h1 : second_number = 45)
  (h2 : increment = 11)
  (h3 : final_number = 89)
  : ∃ first_number : ℕ, first_number + increment = second_number := 
by
  sorry

end find_first_number_l2026_202637


namespace largest_value_of_n_l2026_202621

theorem largest_value_of_n :
  ∃ (n : ℕ) (X Y Z : ℕ),
    n = 25 * X + 5 * Y + Z ∧
    n = 81 * Z + 9 * Y + X ∧
    X < 5 ∧ Y < 5 ∧ Z < 5 ∧
    n = 121 := by
  sorry

end largest_value_of_n_l2026_202621


namespace total_amount_in_wallet_l2026_202658

theorem total_amount_in_wallet
  (num_10_bills : ℕ)
  (num_20_bills : ℕ)
  (num_5_bills : ℕ)
  (amount_10_bills : ℕ)
  (num_20_bills_eq : num_20_bills = 4)
  (amount_10_bills_eq : amount_10_bills = 50)
  (total_num_bills : ℕ)
  (total_num_bills_eq : total_num_bills = 13)
  (num_10_bills_eq : num_10_bills = amount_10_bills / 10)
  (total_amount : ℕ)
  (total_amount_eq : total_amount = amount_10_bills + num_20_bills * 20 + num_5_bills * 5)
  (num_bills_accounted : ℕ)
  (num_bills_accounted_eq : num_bills_accounted = num_10_bills + num_20_bills)
  (num_5_bills_eq : num_5_bills = total_num_bills - num_bills_accounted)
  : total_amount = 150 :=
by
  sorry

end total_amount_in_wallet_l2026_202658


namespace a_8_is_256_l2026_202660

variable (a : ℕ → ℕ)

axiom a_1 : a 1 = 2

axiom a_pq : ∀ p q : ℕ, a (p + q) = a p * a q

theorem a_8_is_256 : a 8 = 256 := by
  sorry

end a_8_is_256_l2026_202660


namespace lcm_9_12_15_l2026_202623

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := sorry

end lcm_9_12_15_l2026_202623


namespace distance_between_circle_center_and_point_l2026_202600

theorem distance_between_circle_center_and_point (x y : ℝ) (h : x^2 + y^2 = 8*x - 12*y + 40) : 
  dist (4, -6) (4, -2) = 4 := 
by
  sorry

end distance_between_circle_center_and_point_l2026_202600


namespace turtles_remaining_on_log_l2026_202693

-- Define the initial conditions
def original_turtles := 9
def additional_turtles := (3 * original_turtles) - 2
def total_group := original_turtles + additional_turtles
def frightened_turtles := total_group / 2

-- Theorem statement
theorem turtles_remaining_on_log : total_group - frightened_turtles = 17 :=
by
  sorry

end turtles_remaining_on_log_l2026_202693


namespace cats_in_studio_count_l2026_202676

theorem cats_in_studio_count :
  (70 + 40 + 30 + 50
  - 25 - 15 - 20 - 28
  + 5 + 10 + 12
  - 8
  + 12) = 129 :=
by sorry

end cats_in_studio_count_l2026_202676


namespace ball_bounce_height_lt_one_l2026_202673

theorem ball_bounce_height_lt_one :
  ∃ (k : ℕ), 15 * (1/3:ℝ)^k < 1 ∧ k = 3 := 
sorry

end ball_bounce_height_lt_one_l2026_202673


namespace correct_option_D_l2026_202633

theorem correct_option_D : 
  (-3)^2 = 9 ∧ 
  - (x + y) = -x - y ∧ 
  ¬ (3 * a + 5 * b = 8 * a * b) ∧ 
  5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 :=
by { sorry }

end correct_option_D_l2026_202633


namespace jia_winning_strategy_l2026_202618

variables {p q : ℝ}
def is_quadratic_real_roots (a b c : ℝ) : Prop := b ^ 2 - 4 * a * c > 0

def quadratic_with_roots (x1 x2 : ℝ) :=
  x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ is_quadratic_real_roots 1 (- (x1 + x2)) (x1 * x2)

def modify_jia (p q x1 : ℝ) : (ℝ × ℝ) := (p + 1, q - x1)

def modify_yi1 (p q : ℝ) : (ℝ × ℝ) := (p - 1, q)

def modify_yi2 (p q x2 : ℝ) : (ℝ × ℝ) := (p - 1, q + x2)

def winning_strategy_jia (x1 x2 : ℝ) : Prop :=
  ∃ n : ℕ, ∀ m ≥ n, ∀ p q, quadratic_with_roots x1 x2 → 
  (¬ is_quadratic_real_roots 1 p q) ∨ (q ≤ 0)

theorem jia_winning_strategy (x1 x2 : ℝ)
  (h: quadratic_with_roots x1 x2) : 
  winning_strategy_jia x1 x2 :=
sorry

end jia_winning_strategy_l2026_202618


namespace value_of_c_l2026_202694

-- Define a structure representing conditions of the problem
structure ProblemConditions where
  c : Real

-- Define the problem in terms of given conditions and required proof
theorem value_of_c (conditions : ProblemConditions) : conditions.c = 5 / 2 := by
  sorry

end value_of_c_l2026_202694


namespace find_m_of_parallel_lines_l2026_202698

theorem find_m_of_parallel_lines (m : ℝ) 
  (H1 : ∃ x y : ℝ, m * x + 2 * y + 6 = 0) 
  (H2 : ∃ x y : ℝ, x + (m - 1) * y + m^2 - 1 = 0) : 
  m = -1 := 
by
  sorry

end find_m_of_parallel_lines_l2026_202698


namespace find_x_for_parallel_vectors_l2026_202648

theorem find_x_for_parallel_vectors :
  ∀ (x : ℚ), (∃ a b : ℚ × ℚ, a = (2 * x, 3) ∧ b = (1, 9) ∧ (∃ k : ℚ, (2 * x, 3) = (k * 1, k * 9))) ↔ x = 1 / 6 :=
by 
  sorry

end find_x_for_parallel_vectors_l2026_202648


namespace bertha_daughters_no_daughters_l2026_202613

theorem bertha_daughters_no_daughters (daughters granddaughters: ℕ) (no_great_granddaughters: granddaughters = 5 * daughters) (total_women: 8 + granddaughters = 48) :
  8 + granddaughters = 48 :=
by {
  sorry
}

end bertha_daughters_no_daughters_l2026_202613


namespace inequality_holds_for_all_real_l2026_202647

theorem inequality_holds_for_all_real (a : ℝ) : a + a^3 - a^4 - a^6 < 1 :=
by
  sorry

end inequality_holds_for_all_real_l2026_202647


namespace problem_statement_l2026_202620

-- Define the arithmetic sequence and required terms
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
variables (a : ℕ → ℝ) (d : ℝ)
axiom seq_is_arithmetic : arithmetic_seq a d
axiom sum_of_a2_a4_a6_is_3 : a 2 + a 4 + a 6 = 3

-- Goal: Prove a1 + a3 + a5 + a7 = 4
theorem problem_statement : a 1 + a 3 + a 5 + a 7 = 4 :=
by 
  sorry

end problem_statement_l2026_202620


namespace square_TU_squared_l2026_202685

theorem square_TU_squared (P Q R S T U : ℝ × ℝ)
  (side : ℝ) (RT SU PT QU : ℝ)
  (hpqrs : (P.1 - S.1)^2 + (P.2 - S.2)^2 = side^2 ∧ (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = side^2 ∧ 
            (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = side^2 ∧ (S.1 - R.1)^2 + (S.2 - R.2)^2 = side^2)
  (hRT : (R.1 - T.1)^2 + (R.2 - T.2)^2 = RT^2)
  (hSU : (S.1 - U.1)^2 + (S.2 - U.2)^2 = SU^2)
  (hPT : (P.1 - T.1)^2 + (P.2 - T.2)^2 = PT^2)
  (hQU : (Q.1 - U.1)^2 + (Q.2 - U.2)^2 = QU^2)
  (side_eq_17 : side = 17) (RT_SU_eq_8 : RT = 8) (PT_QU_eq_15 : PT = 15) :
  (T.1 - U.1)^2 + (T.2 - U.2)^2 = 979.5 :=
by
  -- proof to be filled in
  sorry

end square_TU_squared_l2026_202685


namespace distance_point_to_line_l2026_202642

theorem distance_point_to_line : 
  let x0 := 1
  let y0 := 0
  let A := 1
  let B := -2
  let C := 1 
  let dist := (A * x0 + B * y0 + C : ℝ) / Real.sqrt (A^2 + B^2)
  abs dist = 2 * Real.sqrt 5 / 5 :=
by
  -- Using basic principles of Lean and Mathlib to state the equality proof
  sorry

end distance_point_to_line_l2026_202642


namespace other_endpoint_diameter_l2026_202651

theorem other_endpoint_diameter (O : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hO : O = (2, 3)) (hA : A = (-1, -1)) 
  (h_midpoint : O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) : B = (5, 7) := by
  sorry

end other_endpoint_diameter_l2026_202651


namespace lcm_3_15_is_15_l2026_202641

theorem lcm_3_15_is_15 : Nat.lcm 3 15 = 15 :=
sorry

end lcm_3_15_is_15_l2026_202641


namespace perimeter_of_ABC_HI_IJK_l2026_202631

theorem perimeter_of_ABC_HI_IJK (AB AC AH HI AI AK KI IJ JK : ℝ) 
(H_midpoint : H = AC / 2) (K_midpoint : K = AI / 2) 
(equil_triangle_ABC : AB = AC) (equil_triangle_AHI : AH = HI ∧ HI = AI) 
(equil_triangle_IJK : IJ = JK ∧ JK = KI) 
(AB_eq : AB = 6) : 
  AB + AC + AH + HI + IJ + JK + KI = 22.5 :=
by
  sorry

end perimeter_of_ABC_HI_IJK_l2026_202631


namespace baylor_final_amount_l2026_202687

def CDA := 4000
def FCP := (1 / 2) * CDA
def SCP := FCP + (2 / 5) * FCP
def TCP := 2 * (FCP + SCP)
def FDA := CDA + FCP + SCP + TCP

theorem baylor_final_amount : FDA = 18400 := by
  sorry

end baylor_final_amount_l2026_202687


namespace cube_root_3375_l2026_202605

theorem cube_root_3375 (c d : ℕ) (h1 : c > 0 ∧ d > 0) (h2 : c * d^3 = 3375) (h3 : ∀ k : ℕ, k > 0 → c * (d / k)^3 ≠ 3375) : 
  c + d = 16 :=
sorry

end cube_root_3375_l2026_202605


namespace soda_amount_l2026_202683

theorem soda_amount (S : ℝ) (h1 : S / 2 + 2000 = (S - (S / 2 + 2000)) / 2 + 2000) : S = 12000 :=
by
  sorry

end soda_amount_l2026_202683


namespace problem_solution_l2026_202670

theorem problem_solution (x : ℝ) :
  (⌊|x^2 - 1|⌋ = 10) ↔ (x ∈ Set.Ioc (-2 * Real.sqrt 3) (-Real.sqrt 11) ∪ Set.Ico (Real.sqrt 11) (2 * Real.sqrt 3)) :=
by
  sorry

end problem_solution_l2026_202670


namespace prob_axisymmetric_and_centrally_symmetric_l2026_202671

theorem prob_axisymmetric_and_centrally_symmetric : 
  let card1 := "Line segment"
  let card2 := "Equilateral triangle"
  let card3 := "Parallelogram"
  let card4 := "Isosceles trapezoid"
  let card5 := "Circle"
  let cards := [card1, card2, card3, card4, card5]
  let symmetric_cards := [card1, card5]
  (symmetric_cards.length / cards.length : ℚ) = 2 / 5 :=
by sorry

end prob_axisymmetric_and_centrally_symmetric_l2026_202671


namespace smallest_fraction_greater_than_4_over_5_l2026_202657

theorem smallest_fraction_greater_than_4_over_5 :
  ∃ (b : ℕ), 10 ≤ b ∧ b < 100 ∧ 77 * 5 > 4 * b ∧ Int.gcd 77 b = 1 ∧
  ∀ (a : ℕ), 10 ≤ a ∧ a < 77 → ¬ ∃ (b' : ℕ), 10 ≤ b' ∧ b' < 100 ∧ a * 5 > 4 * b' ∧ Int.gcd a b' = 1 := by
  sorry

end smallest_fraction_greater_than_4_over_5_l2026_202657


namespace height_of_parallelogram_l2026_202674

noncomputable def parallelogram_height (base area : ℝ) : ℝ :=
  area / base

theorem height_of_parallelogram :
  parallelogram_height 8 78.88 = 9.86 :=
by
  -- This is where the proof would go, but it's being omitted as per instructions.
  sorry

end height_of_parallelogram_l2026_202674


namespace correct_calculation_l2026_202603

theorem correct_calculation : 
  ¬(2 * Real.sqrt 3 + 3 * Real.sqrt 2 = 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  ¬(5 * Real.sqrt 3 * 5 * Real.sqrt 2 = 5 * Real.sqrt 6) ∧
  ¬(Real.sqrt (4 + 1 / 2) = 2 * Real.sqrt (1 / 2)) :=
by {
  -- Using the conditions to prove the correct option B
  sorry
}

end correct_calculation_l2026_202603


namespace product_of_numbers_l2026_202666

theorem product_of_numbers (x y z n : ℕ) 
  (h1 : x + y + z = 150)
  (h2 : 7 * x = n)
  (h3 : y - 10 = n)
  (h4 : z + 10 = n) : x * y * z = 48000 := 
by 
  sorry

end product_of_numbers_l2026_202666


namespace calc_g_x_plus_3_l2026_202629

def g (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem calc_g_x_plus_3 (x : ℝ) : g (x + 3) = x^2 + 3*x + 2 :=
by
  sorry

end calc_g_x_plus_3_l2026_202629


namespace xy_range_l2026_202662

theorem xy_range (x y : ℝ)
  (h1 : x + y = 1)
  (h2 : 1 / 3 ≤ x ∧ x ≤ 2 / 3) :
  2 / 9 ≤ x * y ∧ x * y ≤ 1 / 4 :=
sorry

end xy_range_l2026_202662


namespace moles_of_magnesium_l2026_202614

-- Assuming the given conditions as hypotheses
variables (Mg CO₂ MgO C : ℕ)

-- Theorem statement
theorem moles_of_magnesium (h1 : 2 * Mg + CO₂ = 2 * MgO + C) 
                           (h2 : MgO = Mg) 
                           (h3 : CO₂ = 1) 
                           : Mg = 2 :=
by sorry  -- Proof to be provided

end moles_of_magnesium_l2026_202614


namespace final_mark_is_correct_l2026_202653

def term_mark : ℝ := 80
def term_weight : ℝ := 0.70
def exam_mark : ℝ := 90
def exam_weight : ℝ := 0.30

theorem final_mark_is_correct :
  (term_mark * term_weight + exam_mark * exam_weight) = 83 :=
by
  sorry

end final_mark_is_correct_l2026_202653


namespace simplify_trig_expression_l2026_202619

theorem simplify_trig_expression :
  (Real.cos (72 * Real.pi / 180) * Real.sin (78 * Real.pi / 180) +
   Real.sin (72 * Real.pi / 180) * Real.sin (12 * Real.pi / 180) = 1 / 2) :=
by sorry

end simplify_trig_expression_l2026_202619


namespace largest_angle_consecutive_even_pentagon_l2026_202645

theorem largest_angle_consecutive_even_pentagon :
  ∀ (n : ℕ), (2 * n + (2 * n + 2) + (2 * n + 4) + (2 * n + 6) + (2 * n + 8) = 540) →
  (2 * n + 8 = 112) :=
by
  intros n h
  sorry

end largest_angle_consecutive_even_pentagon_l2026_202645


namespace petya_payment_l2026_202664

theorem petya_payment (x y : ℤ) (h₁ : 14 * x + 3 * y = 107) (h₂ : |x - y| ≤ 5) : x + y = 10 :=
sorry

end petya_payment_l2026_202664


namespace min_area_and_line_eq_l2026_202691

theorem min_area_and_line_eq (a b : ℝ) (l : ℝ → ℝ → Prop)
    (h1 : l 3 2)
    (h2: ∀ x y: ℝ, l x y → (x/a + y/b = 1))
    (h3: a > 0)
    (h4: b > 0)
    : 
    a = 6 ∧ b = 4 ∧ 
    (∀ x y : ℝ, l x y ↔ (4 * x + 6 * y - 24 = 0)) ∧ 
    (∃ min_area : ℝ, min_area = 12) :=
by
  sorry

end min_area_and_line_eq_l2026_202691


namespace odd_prime_divides_seq_implies_power_of_two_divides_l2026_202644

theorem odd_prime_divides_seq_implies_power_of_two_divides (a : ℕ → ℤ) (p n : ℕ)
  (h0 : a 0 = 2)
  (hk : ∀ k, a (k + 1) = 2 * (a k) ^ 2 - 1)
  (h_odd_prime : Nat.Prime p)
  (h_odd : p % 2 = 1)
  (h_divides : ↑p ∣ a n) :
  2^(n + 3) ∣ p^2 - 1 :=
sorry

end odd_prime_divides_seq_implies_power_of_two_divides_l2026_202644


namespace maggie_earnings_proof_l2026_202616

def rate_per_subscription : ℕ := 5
def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_neighbor1 : ℕ := 2
def subscriptions_to_neighbor2 : ℕ := 2 * subscriptions_to_neighbor1
def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_neighbor1 + subscriptions_to_neighbor2
def total_earnings : ℕ := total_subscriptions * rate_per_subscription

theorem maggie_earnings_proof : total_earnings = 55 := by
  sorry

end maggie_earnings_proof_l2026_202616


namespace isosceles_triangle_base_length_l2026_202646

theorem isosceles_triangle_base_length
  (a : ℕ) (b : ℕ)
  (ha : a = 7) 
  (p : ℕ)
  (hp : p = a + a + b) 
  (hp_perimeter : p = 21) : b = 7 :=
by 
  -- The actual proof will go here, using the provided conditions
  sorry

end isosceles_triangle_base_length_l2026_202646


namespace simplified_expression_correct_l2026_202696

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) * (x^2 - 6*x + 8) / (x^2 - 8*x + 15)

theorem simplified_expression_correct (x : ℝ) :
  simplify_expression x = ((x - 1) * (x - 2) * (x - 4)) / ((x - 3) * (x - 5)) :=
  sorry

end simplified_expression_correct_l2026_202696


namespace evaluate_expression_l2026_202601

theorem evaluate_expression :
  (3 + 6 + 9 : ℚ) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  sorry

end evaluate_expression_l2026_202601


namespace gcd_91_49_l2026_202612

theorem gcd_91_49 : Int.gcd 91 49 = 7 := by
  sorry

end gcd_91_49_l2026_202612
