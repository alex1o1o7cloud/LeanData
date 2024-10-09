import Mathlib

namespace agatha_initial_money_l1818_181883

/-
Agatha has some money to spend on a new bike. She spends $15 on the frame, and $25 on the front wheel.
If she has $20 left to spend on a seat and handlebar tape, prove that she had $60 initially.
-/

theorem agatha_initial_money (frame_cost wheel_cost remaining_money initial_money: ℕ) 
  (h1 : frame_cost = 15) 
  (h2 : wheel_cost = 25) 
  (h3 : remaining_money = 20) 
  (h4 : initial_money = frame_cost + wheel_cost + remaining_money) : 
  initial_money = 60 :=
by {
  -- We state explicitly that initial_money should be 60
  sorry
}

end agatha_initial_money_l1818_181883


namespace card_distribution_count_l1818_181888

def card_distribution_ways : Nat := sorry

theorem card_distribution_count :
  card_distribution_ways = 9 := sorry

end card_distribution_count_l1818_181888


namespace find_s_range_l1818_181826

variables {a b c s t y1 y2 : ℝ}

-- Conditions
def is_vertex (a b c s t : ℝ) : Prop := ∀ x : ℝ, (a * x^2 + b * x + c = a * (x - s)^2 + t)

def passes_points (a b c y1 y2 : ℝ) : Prop := 
  (a * (-2)^2 + b * (-2) + c = y1) ∧ (a * 4^2 + b * 4 + c = y2)

def valid_constants (a y1 y2 t : ℝ) : Prop := 
  (a ≠ 0) ∧ (y1 > y2) ∧ (y2 > t)

-- Theorem
theorem find_s_range {a b c s t y1 y2 : ℝ}
  (hv : is_vertex a b c s t)
  (hp : passes_points a b c y1 y2)
  (vc : valid_constants a y1 y2 t) : 
  s > 1 ∧ s ≠ 4 :=
sorry -- Proof skipped

end find_s_range_l1818_181826


namespace common_chord_of_circles_l1818_181812

theorem common_chord_of_circles
  (x y : ℝ)
  (h1 : x^2 + y^2 + 2 * x = 0)
  (h2 : x^2 + y^2 - 4 * y = 0)
  : x + 2 * y = 0 := 
by
  -- Lean will check the logical consistency of the statement.
  sorry

end common_chord_of_circles_l1818_181812


namespace option_d_true_l1818_181825

theorem option_d_true (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpr_qr : p * r < q * r) : 1 > q / p :=
sorry

end option_d_true_l1818_181825


namespace equation1_equation2_equation3_equation4_l1818_181870

theorem equation1 (x : ℝ) : (x - 1) ^ 2 - 5 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

theorem equation2 (x : ℝ) : x * (x + 4) = -3 * (x + 4) ↔ x = -4 ∨ x = -3 := by
  sorry

theorem equation3 (y : ℝ) : 2 * y ^ 2 - 5 * y + 2 = 0 ↔ y = 1 / 2 ∨ y = 2 := by
  sorry

theorem equation4 (m : ℝ) : 2 * m ^ 2 - 7 * m - 3 = 0 ↔ m = (7 + Real.sqrt 73) / 4 ∨ m = (7 - Real.sqrt 73) / 4 := by
  sorry

end equation1_equation2_equation3_equation4_l1818_181870


namespace subset_condition_l1818_181865

theorem subset_condition (m : ℝ) (A : Set ℝ) (B : Set ℝ) :
  A = {1, 3} ∧ B = {1, 2, m} ∧ A ⊆ B → m = 3 :=
by
  sorry

end subset_condition_l1818_181865


namespace monotonicity_of_f_range_of_a_l1818_181861

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log x - a / x

theorem monotonicity_of_f (a : ℝ) (h : 0 < a) :
  ∀ x y : ℝ, (0 < x) → (0 < y) → (x < y) → (f x a < f y a) :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a < x ^ 2) ↔ (-1 ≤ a) :=
by
  sorry

end monotonicity_of_f_range_of_a_l1818_181861


namespace final_value_of_A_l1818_181862

-- Define the initial value of A
def initial_value (A : ℤ) : Prop := A = 15

-- Define the reassignment condition
def reassignment_cond (A : ℤ) : Prop := A = -A + 5

-- The theorem stating that given the initial value and reassignment condition, the final value of A is -10
theorem final_value_of_A (A : ℤ) (h1 : initial_value A) (h2 : reassignment_cond A) : A = -10 := by
  sorry

end final_value_of_A_l1818_181862


namespace second_offset_length_l1818_181877

noncomputable def quadrilateral_area (d o1 o2 : ℝ) : ℝ :=
  (1 / 2) * d * (o1 + o2)

theorem second_offset_length (d o1 A : ℝ) (h_d : d = 22) (h_o1 : o1 = 9) (h_A : A = 165) :
  ∃ o2, quadrilateral_area d o1 o2 = A ∧ o2 = 6 := by
  sorry

end second_offset_length_l1818_181877


namespace giraffes_count_l1818_181854

def numZebras : ℕ := 12

def numCamels : ℕ := numZebras / 2

def numMonkeys : ℕ := numCamels * 4

def numGiraffes : ℕ := numMonkeys - 22

theorem giraffes_count :
  numGiraffes = 2 :=
by 
  sorry

end giraffes_count_l1818_181854


namespace michelle_gas_left_l1818_181801

def gasLeft (initialGas: ℝ) (usedGas: ℝ) : ℝ :=
  initialGas - usedGas

theorem michelle_gas_left :
  gasLeft 0.5 0.3333333333333333 = 0.1666666666666667 :=
by
  -- proof goes here
  sorry

end michelle_gas_left_l1818_181801


namespace normal_mean_is_zero_if_symmetric_l1818_181824

-- Definition: A normal distribution with mean μ and standard deviation σ.
structure NormalDist where
  μ : ℝ
  σ : ℝ

-- Condition: The normal curve is symmetric about the y-axis.
def symmetric_about_y_axis (nd : NormalDist) : Prop :=
  nd.μ = 0

-- Theorem: If the normal curve is symmetric about the y-axis, then the mean μ of the corresponding normal distribution is 0.
theorem normal_mean_is_zero_if_symmetric (nd : NormalDist) (h : symmetric_about_y_axis nd) : nd.μ = 0 := 
by sorry

end normal_mean_is_zero_if_symmetric_l1818_181824


namespace expression_eval_l1818_181815

theorem expression_eval : 2 * 3 + 2 * 3 = 12 := by
  sorry

end expression_eval_l1818_181815


namespace total_ducks_and_ducklings_l1818_181856

theorem total_ducks_and_ducklings :
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6
  (ducks1 + ducks2 + ducks3) + (ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3) = 99 :=
by
  sorry

end total_ducks_and_ducklings_l1818_181856


namespace A_inter_B_eq_l1818_181810

-- Define set A based on the condition for different integer k.
def A (k : ℤ) : Set ℝ := {x | 2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi}

-- Define set B based on its condition.
def B : Set ℝ := {x | -5 ≤ x ∧ x < 4}

-- The final proof problem to show A ∩ B equals to the given set.
theorem A_inter_B_eq : 
  (⋃ k : ℤ, A k) ∩ B = {x | (-Real.pi < x ∧ x < 0) ∨ (Real.pi < x ∧ x < 4)} :=
by
  sorry

end A_inter_B_eq_l1818_181810


namespace optimal_order_l1818_181842

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end optimal_order_l1818_181842


namespace find_number_l1818_181843

theorem find_number (x : ℝ) (h : x / 5 + 23 = 42) : x = 95 :=
by
  -- Proof placeholder
  sorry

end find_number_l1818_181843


namespace least_not_lucky_multiple_of_6_l1818_181804

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_not_lucky_multiple_of_6 : ∃ k : ℕ, k > 0 ∧ k % 6 = 0 ∧ ¬ is_lucky k ∧ ∀ m : ℕ, m > 0 ∧ m % 6 = 0 ∧ ¬ is_lucky m → k ≤ m :=
  sorry

end least_not_lucky_multiple_of_6_l1818_181804


namespace pow_1999_mod_26_l1818_181849

theorem pow_1999_mod_26 (n : ℕ) (h1 : 17^1 % 26 = 17)
  (h2 : 17^2 % 26 = 17) (h3 : 17^3 % 26 = 17) : 17^1999 % 26 = 17 := by
  sorry

end pow_1999_mod_26_l1818_181849


namespace knight_king_moves_incompatible_l1818_181885

-- Definitions for moves and chessboards
structure Board :=
  (numbering : Fin 64 → Nat)
  (different_board : Prop)

def knights_move (x y : Fin 64) : Prop :=
  (abs (x / 8 - y / 8) = 2 ∧ abs (x % 8 - y % 8) = 1) ∨
  (abs (x / 8 - y / 8) = 1 ∧ abs (x % 8 - y % 8) = 2)

def kings_move (x y : Fin 64) : Prop :=
  abs (x / 8 - y / 8) ≤ 1 ∧ abs (x % 8 - y % 8) ≤ 1 ∧ (x ≠ y)

-- Theorem stating the proof problem
theorem knight_king_moves_incompatible (vlad_board gosha_board : Board) (h_board_diff: vlad_board.different_board):
  ¬ ∀ i j : Fin 64, (knights_move i j ↔ kings_move (vlad_board.numbering i) (vlad_board.numbering j)) :=
by {
  -- Skipping proofs with sorry
  sorry
}

end knight_king_moves_incompatible_l1818_181885


namespace students_in_class_C_l1818_181858

theorem students_in_class_C 
    (total_students : ℕ := 80) 
    (percent_class_A : ℕ := 40) 
    (class_B_difference : ℕ := 21) 
    (h_percent : percent_class_A = 40) 
    (h_class_B_diff : class_B_difference = 21) 
    (h_total_students : total_students = 80) : 
    total_students - ((percent_class_A * total_students) / 100 - class_B_difference + (percent_class_A * total_students) / 100) = 37 := by
    sorry

end students_in_class_C_l1818_181858


namespace repeating_decimal_as_fraction_l1818_181814

theorem repeating_decimal_as_fraction :
  (∃ y : ℚ, y = 737910 ∧ 0.73 + 864 / 999900 = y / 999900) :=
by
  -- proof omitted
  sorry

end repeating_decimal_as_fraction_l1818_181814


namespace find_missing_ratio_l1818_181892

theorem find_missing_ratio
  (x y : ℕ)
  (h : ((2 / 3 : ℚ) * (x / y : ℚ) * (11 / 2 : ℚ) = 2)) :
  x = 6 ∧ y = 11 :=
sorry

end find_missing_ratio_l1818_181892


namespace fruiting_plants_given_away_l1818_181895

noncomputable def roxy_fruiting_plants_given_away 
  (N_f : ℕ) -- initial flowering plants
  (N_ft : ℕ) -- initial fruiting plants
  (N_bsf : ℕ) -- flowering plants bought on Saturday
  (N_bst : ℕ) -- fruiting plants bought on Saturday
  (N_gsf : ℕ) -- flowering plant given away on Sunday
  (N_total_remaining : ℕ) -- total plants remaining 
  (H₁ : N_ft = 2 * N_f) -- twice as many fruiting plants
  (H₂ : N_total_remaining = (N_f + N_bsf - N_gsf) + (N_ft + N_bst - N_gst)) -- total plants equation
  : ℕ :=
  4

theorem fruiting_plants_given_away (N_f : ℕ) (N_ft : ℕ) (N_bsf : ℕ) (N_bst : ℕ) (N_gsf : ℕ) (N_total_remaining : ℕ)
  (H₁ : N_ft = 2 * N_f) (H₂ : N_total_remaining = (N_f + N_bsf - N_gsf) + (N_ft + N_bst - N_gst)) : N_ft - (N_total_remaining - (N_f + N_bsf - N_gsf)) = 4 := 
by
  sorry

end fruiting_plants_given_away_l1818_181895


namespace quadratic_m_value_l1818_181803

theorem quadratic_m_value (m : ℕ) :
  (∃ x : ℝ, x^(m + 1) - (m + 1) * x - 2 = 0) →
  m + 1 = 2 →
  m = 1 :=
by {
  sorry
}

end quadratic_m_value_l1818_181803


namespace no_representation_of_expr_l1818_181809

theorem no_representation_of_expr :
  ¬ ∃ f g : ℝ → ℝ, (∀ x y : ℝ, 1 + x ^ 2016 * y ^ 2016 = f x * g y) :=
by
  sorry

end no_representation_of_expr_l1818_181809


namespace intersection_M_N_l1818_181871

open Set Real

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - abs x)

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l1818_181871


namespace alix_more_chocolates_than_nick_l1818_181875

theorem alix_more_chocolates_than_nick :
  let nick_chocolates := 10
  let initial_alix_chocolates := 3 * nick_chocolates
  let after_mom_took_chocolates := initial_alix_chocolates - 5
  after_mom_took_chocolates - nick_chocolates = 15 := by
sorry

end alix_more_chocolates_than_nick_l1818_181875


namespace hiker_walking_speed_l1818_181852

theorem hiker_walking_speed (v : ℝ) :
  (∃ (hiker_shares_cyclist_distance : 20 / 60 * v = 25 * (5 / 60)), v = 6.25) :=
by
  sorry

end hiker_walking_speed_l1818_181852


namespace hyperbola_a_unique_l1818_181828

-- Definitions from the conditions
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1
def foci (c : ℝ) : Prop := c = 2 * Real.sqrt 3
def a_positive (a : ℝ) : Prop := a > 0

-- Statement to prove
theorem hyperbola_a_unique (a : ℝ) (h : hyperbola 0 0 a ∧ foci (2 * Real.sqrt 3) ∧ a_positive a) : a = 2 * Real.sqrt 2 := 
sorry

end hyperbola_a_unique_l1818_181828


namespace average_of_first_5_multiples_of_5_l1818_181868

theorem average_of_first_5_multiples_of_5 : 
  (5 + 10 + 15 + 20 + 25) / 5 = 15 :=
by
  sorry

end average_of_first_5_multiples_of_5_l1818_181868


namespace batsman_average_after_12th_innings_l1818_181840

theorem batsman_average_after_12th_innings 
  (A : ℕ) 
  (total_runs_11_innings : ℕ := 11 * A) 
  (new_average : ℕ := A + 2) 
  (total_runs_12_innings : ℕ := total_runs_11_innings + 92) 
  (increased_average_after_12 : 12 * new_average = total_runs_12_innings) 
  : new_average = 70 := 
by
  -- skipping proof
  sorry

end batsman_average_after_12th_innings_l1818_181840


namespace andrew_vacation_days_l1818_181830

theorem andrew_vacation_days (days_worked last_year vacation_per_10 worked_days in_march in_september : ℕ)
  (h1 : vacation_per_10 = 10)
  (h2 : days_worked_last_year = 300)
  (h3 : worked_days = days_worked_last_year / vacation_per_10)
  (h4 : in_march = 5)
  (h5 : in_september = 2 * in_march)
  (h6 : days_taken = in_march + in_september)
  (h7 : vacation_days_remaining = worked_days - days_taken) :
  vacation_days_remaining = 15 :=
by
  sorry

end andrew_vacation_days_l1818_181830


namespace parallel_vectors_k_eq_neg1_l1818_181866

theorem parallel_vectors_k_eq_neg1
  (k : ℤ)
  (a : ℤ × ℤ := (2 * k + 2, 4))
  (b : ℤ × ℤ := (k + 1, 8))
  (h : a.1 * b.2 = a.2 * b.1) :
  k = -1 :=
by
sorry

end parallel_vectors_k_eq_neg1_l1818_181866


namespace total_pens_l1818_181841

theorem total_pens (black_pens blue_pens : ℕ) (h1 : black_pens = 4) (h2 : blue_pens = 4) : black_pens + blue_pens = 8 :=
by
  sorry

end total_pens_l1818_181841


namespace least_number_of_groups_l1818_181821

def num_students : ℕ := 24
def max_students_per_group : ℕ := 10

theorem least_number_of_groups : ∃ x, ∀ y, y ≤ max_students_per_group ∧ num_students = x * y → x = 3 := by
  sorry

end least_number_of_groups_l1818_181821


namespace gamma_distribution_moments_l1818_181836

noncomputable def gamma_density (α β x : ℝ) : ℝ :=
  (1 / (β ^ (α + 1) * Real.Gamma (α + 1))) * x ^ α * Real.exp (-x / β)

open Real

theorem gamma_distribution_moments (α β : ℝ) (x_bar D_B : ℝ) (hα : α > -1) (hβ : β > 0) :
  α = x_bar ^ 2 / D_B - 1 ∧ β = D_B / x_bar :=
by
  sorry

end gamma_distribution_moments_l1818_181836


namespace unique_positive_solution_l1818_181835

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end unique_positive_solution_l1818_181835


namespace complement_union_eq_l1818_181884

def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}
def complement (A : Set ℝ) : Set ℝ := {x | x ∉ A}

theorem complement_union_eq :
  complement (M ∪ N) = {x | x ≥ 1} :=
sorry

end complement_union_eq_l1818_181884


namespace incenter_divides_segment_l1818_181860

variables (A B C I M : Type) (R r : ℝ)

-- Definitions based on conditions
def is_incenter (I : Type) (A B C : Type) : Prop := sorry
def is_circumcircle (C : Type) : Prop := sorry
def angle_bisector_intersects_at (A B C M : Type) : Prop := sorry
def divides_segment (I M : Type) (a b : ℝ) : Prop := sorry

-- Proof problem statement
theorem incenter_divides_segment (h1 : is_circumcircle C)
                                   (h2 : is_incenter I A B C)
                                   (h3 : angle_bisector_intersects_at A B C M)
                                   (h4 : divides_segment I M a b) :
  a * b = 2 * R * r :=
sorry

end incenter_divides_segment_l1818_181860


namespace sum_of_smallest_and_second_smallest_l1818_181822

-- Define the set of numbers
def numbers : Set ℕ := {10, 11, 12, 13}

-- Define the smallest and second smallest numbers
def smallest_number : ℕ := 10
def second_smallest_number : ℕ := 11

-- Prove the sum of the smallest and the second smallest numbers
theorem sum_of_smallest_and_second_smallest : smallest_number + second_smallest_number = 21 := by
  sorry

end sum_of_smallest_and_second_smallest_l1818_181822


namespace gdp_scientific_notation_l1818_181859

noncomputable def gdp_nanning_2007 : ℝ := 1060 * 10^8

theorem gdp_scientific_notation :
  gdp_nanning_2007 = 1.06 * 10^11 :=
by sorry

end gdp_scientific_notation_l1818_181859


namespace point_A_in_first_quadrant_l1818_181845

def point_in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem point_A_in_first_quadrant : point_in_first_quadrant 1 2 := by
  sorry

end point_A_in_first_quadrant_l1818_181845


namespace total_carrots_grown_l1818_181851

theorem total_carrots_grown :
  let Sandy := 6.5
  let Sam := 3.25
  let Sophie := 2.75 * Sam
  let Sara := (Sandy + Sam + Sophie) - 7.5
  Sandy + Sam + Sophie + Sara = 29.875 :=
by
  sorry

end total_carrots_grown_l1818_181851


namespace find_m_given_root_of_quadratic_l1818_181834

theorem find_m_given_root_of_quadratic (m : ℝ) : (∃ x : ℝ, x = 3 ∧ x^2 - m * x - 6 = 0) → m = 1 := 
by
  sorry

end find_m_given_root_of_quadratic_l1818_181834


namespace avg_height_is_28_l1818_181846

-- Define the height relationship between trees
def height_relation (a b : ℕ) := a = 2 * b ∨ a = b / 2

-- Given tree heights (partial information)
def height_tree_2 := 14
def height_tree_5 := 20

-- Define the tree heights variables
variables (height_tree_1 height_tree_3 height_tree_4 height_tree_6 : ℕ)

-- Conditions based on the given data and height relations
axiom h1 : height_relation height_tree_1 height_tree_2
axiom h2 : height_relation height_tree_2 height_tree_3
axiom h3 : height_relation height_tree_3 height_tree_4
axiom h4 : height_relation height_tree_4 height_tree_5
axiom h5 : height_relation height_tree_5 height_tree_6

-- Compute total and average height
def total_height := height_tree_1 + height_tree_2 + height_tree_3 + height_tree_4 + height_tree_5 + height_tree_6
def average_height := total_height / 6

-- Prove the average height is 28 meters
theorem avg_height_is_28 : average_height = 28 := by
  sorry

end avg_height_is_28_l1818_181846


namespace colombian_coffee_amount_l1818_181889

theorem colombian_coffee_amount 
  (C B : ℝ) 
  (h1 : C + B = 100)
  (h2 : 8.75 * C + 3.75 * B = 635) :
  C = 52 := 
sorry

end colombian_coffee_amount_l1818_181889


namespace corvette_trip_time_percentage_increase_l1818_181886

theorem corvette_trip_time_percentage_increase
  (total_distance : ℝ)
  (first_half_speed : ℝ)
  (average_speed : ℝ)
  (first_half_distance second_half_distance first_half_time second_half_time total_time : ℝ)
  (h1 : total_distance = 640)
  (h2 : first_half_speed = 80)
  (h3 : average_speed = 40)
  (h4 : first_half_distance = total_distance / 2)
  (h5 : second_half_distance = total_distance / 2)
  (h6 : first_half_time = first_half_distance / first_half_speed)
  (h7 : total_time = total_distance / average_speed)
  (h8 : second_half_time = total_time - first_half_time) :
  ((second_half_time - first_half_time) / first_half_time) * 100 = 200 := sorry

end corvette_trip_time_percentage_increase_l1818_181886


namespace no_roots_less_than_x0_l1818_181811

theorem no_roots_less_than_x0
  (x₀ a b c d : ℝ)
  (h₁ : ∀ x ≥ x₀, x^2 + a * x + b > 0)
  (h₂ : ∀ x ≥ x₀, x^2 + c * x + d > 0) :
  ∀ x ≥ x₀, x^2 + ((a + c) / 2) * x + ((b + d) / 2) > 0 := 
by
  sorry

end no_roots_less_than_x0_l1818_181811


namespace size_relationship_l1818_181802

noncomputable def a : ℝ := 1 + Real.sqrt 7
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 5
noncomputable def c : ℝ := 4

theorem size_relationship : a < b ∧ b < c := by
  sorry

end size_relationship_l1818_181802


namespace least_integer_condition_l1818_181831

theorem least_integer_condition : ∃ x : ℤ, (x^2 = 2 * x + 72) ∧ (x = -6) :=
sorry

end least_integer_condition_l1818_181831


namespace product_of_two_numbers_l1818_181853

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 8) (h2 : Nat.lcm a b = 48) : a * b = 384 :=
by
  sorry

end product_of_two_numbers_l1818_181853


namespace quadratic_completing_square_l1818_181817

theorem quadratic_completing_square (b c : ℝ) (h : ∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) :
    b + c = -106 :=
sorry

end quadratic_completing_square_l1818_181817


namespace determine_time_l1818_181805

variable (g a V_0 V S t : ℝ)

def velocity_eq : Prop := V = (g + a) * t + V_0
def displacement_eq : Prop := S = 1 / 2 * (g + a) * t^2 + V_0 * t

theorem determine_time (h1 : velocity_eq g a V_0 V t) (h2 : displacement_eq g a V_0 S t) :
  t = 2 * S / (V + V_0) := 
sorry

end determine_time_l1818_181805


namespace functions_increasing_in_interval_l1818_181863

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos x

theorem functions_increasing_in_interval :
  ∀ x, -Real.pi / 4 < x → x < Real.pi / 4 →
  (f x < f (x + 1e-6)) ∧ (g x < g (x + 1e-6)) :=
sorry

end functions_increasing_in_interval_l1818_181863


namespace find_k_l1818_181829

def otimes (a b : ℝ) := a * b + a + b^2

theorem find_k (k : ℝ) (h1 : otimes 1 k = 2) (h2 : 0 < k) :
  k = 1 :=
sorry

end find_k_l1818_181829


namespace cos_4theta_l1818_181898

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (4 * θ) = 17/81 :=
  sorry

end cos_4theta_l1818_181898


namespace favorite_numbers_parity_l1818_181873

variables (D J A H : ℤ)

def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem favorite_numbers_parity
  (h1 : odd (D + 3 * J))
  (h2 : odd ((A - H) * 5))
  (h3 : even (D * H + 17)) :
  odd D ∧ even J ∧ even A ∧ odd H := 
sorry

end favorite_numbers_parity_l1818_181873


namespace find_x_l1818_181876

theorem find_x (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 7 * x^2 + 14 * x * y = 2 * x^3 + 4 * x^2 * y + y^3) : 
  x = 7 :=
sorry

end find_x_l1818_181876


namespace no_solution_l1818_181878

theorem no_solution (m n : ℕ) : (5 + 3 * Real.sqrt 2) ^ m ≠ (3 + 5 * Real.sqrt 2) ^ n :=
sorry

end no_solution_l1818_181878


namespace unit_digit_7_pow_2023_l1818_181891

theorem unit_digit_7_pow_2023 : (7^2023) % 10 = 3 :=
by
  -- Provide proof here
  sorry

end unit_digit_7_pow_2023_l1818_181891


namespace number_of_legs_twice_heads_diff_eq_22_l1818_181857

theorem number_of_legs_twice_heads_diff_eq_22 (P H : ℕ) (L : ℤ) (Heads : ℕ) (X : ℤ) (h1 : P = 11)
  (h2 : L = 4 * P + 2 * H) (h3 : Heads = P + H) (h4 : L = 2 * Heads + X) : X = 22 :=
by
  sorry

end number_of_legs_twice_heads_diff_eq_22_l1818_181857


namespace scenario_1_scenario_2_scenario_3_scenario_4_l1818_181823

-- Definitions based on conditions
def prob_A_hit : ℚ := 2 / 3
def prob_B_hit : ℚ := 3 / 4

-- Scenario 1: Prove that the probability of A shooting 3 times and missing at least once is 19/27
theorem scenario_1 : 
  (1 - (prob_A_hit ^ 3)) = 19 / 27 :=
by sorry

-- Scenario 2: Prove that the probability of A hitting the target exactly 2 times and B hitting the target exactly 1 time after each shooting twice is 1/6
theorem scenario_2 : 
  (2 * ((prob_A_hit ^ 2) * (1 - prob_A_hit)) * (2 * (prob_B_hit * (1 - prob_B_hit)))) = 1 / 6 :=
by sorry

-- Scenario 3: Prove that the probability of A missing the target and B hitting the target 2 times after each shooting twice is 1/16
theorem scenario_3 :
  ((1 - prob_A_hit) ^ 2) * (prob_B_hit ^ 2) = 1 / 16 :=
by sorry

-- Scenario 4: Prove that the probability that both A and B hit the target once after each shooting twice is 1/6
theorem scenario_4 : 
  (2 * (prob_A_hit * (1 - prob_A_hit)) * 2 * (prob_B_hit * (1 - prob_B_hit))) = 1 / 6 :=
by sorry

end scenario_1_scenario_2_scenario_3_scenario_4_l1818_181823


namespace bush_height_at_2_years_l1818_181827

theorem bush_height_at_2_years (H: ℕ → ℕ) 
  (quadruple_height: ∀ (n: ℕ), H (n+1) = 4 * H n)
  (H_4: H 4 = 64) : H 2 = 4 :=
by
  sorry

end bush_height_at_2_years_l1818_181827


namespace find_k_and_b_l1818_181896

variables (k b : ℝ)

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (k * p.1, p.2 + b)

theorem find_k_and_b
  (h : f k b (6, 2) = (3, 1)) :
  k = 2 ∧ b = -1 :=
by {
  -- proof steps would go here
  sorry
}

end find_k_and_b_l1818_181896


namespace total_cost_shorts_tshirt_boots_shinguards_l1818_181800

variable (x : ℝ)

-- Definitions provided in the problem statement.
def cost_shorts : ℝ := x
def cost_shorts_and_tshirt : ℝ := 2 * x
def cost_shorts_and_boots : ℝ := 5 * x
def cost_shorts_and_shinguards : ℝ := 3 * x

-- The proof goal to verify:
theorem total_cost_shorts_tshirt_boots_shinguards : 
  (cost_shorts x + (cost_shorts_and_tshirt x - cost_shorts x) + 
   (cost_shorts_and_boots x - cost_shorts x) + 
   (cost_shorts_and_shinguards x - cost_shorts x)) = 8 * x := by 
  sorry

end total_cost_shorts_tshirt_boots_shinguards_l1818_181800


namespace percent_employed_females_l1818_181816

theorem percent_employed_females (h1 : 96 / 100 > 0) (h2 : 24 / 100 > 0) : 
  (96 - 24) / 96 * 100 = 75 := 
by 
  -- Proof to be filled out
  sorry

end percent_employed_females_l1818_181816


namespace line_circle_relationship_l1818_181848

theorem line_circle_relationship (m : ℝ) :
  (∃ x y : ℝ, (mx + y - m - 1 = 0) ∧ (x^2 + y^2 = 2)) ∨ 
  (∃ x : ℝ, (x - 1)^2 + (m*(x - 1) + (1 - 1))^2 = 2) :=
by
  sorry

end line_circle_relationship_l1818_181848


namespace marble_count_l1818_181847

variable (r b g : ℝ)

-- Conditions
def condition1 : b = r / 1.3 := sorry
def condition2 : g = 1.5 * r := sorry

-- Theorem statement
theorem marble_count (h1 : b = r / 1.3) (h2 : g = 1.5 * r) :
  r + b + g = 3.27 * r :=
by sorry

end marble_count_l1818_181847


namespace min_value_expression_l1818_181869

noncomputable def expression (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2)

theorem min_value_expression : ∃ x : ℝ, expression x = 2 * Real.sqrt 5 :=
by
  sorry

end min_value_expression_l1818_181869


namespace negation_even_l1818_181837

open Nat

theorem negation_even (x : ℕ) (h : 0 < x) :
  (∀ x : ℕ, 0 < x → Even x) ↔ ¬ (∃ x : ℕ, 0 < x ∧ Odd x) :=
by
  sorry

end negation_even_l1818_181837


namespace max_m_ratio_l1818_181894

theorem max_m_ratio (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : ∀ a b, (4 / a + 1 / b) ≥ m / (a + 4 * b)) :
  (m = 16) → (b / a = 1 / 4) :=
by sorry

end max_m_ratio_l1818_181894


namespace find_a_l1818_181806

noncomputable def quadratic_inequality_solution (a b : ℝ) : Prop :=
  a * ((-1/2) * (1/3)) * 20 = 20 ∧
  a < 0 ∧
  (-b / (2 * a)) = (-1 / 2 + 1 / 3)

theorem find_a (a b : ℝ) (h : quadratic_inequality_solution a b) : a = -12 :=
  sorry

end find_a_l1818_181806


namespace find_larger_number_l1818_181867

theorem find_larger_number (x y : ℝ) (h1 : x - y = 1860) (h2 : 0.075 * x = 0.125 * y) :
  x = 4650 :=
by
  sorry

end find_larger_number_l1818_181867


namespace find_asterisk_l1818_181887

theorem find_asterisk : ∃ (x : ℕ), (63 / 21) * (x / 189) = 1 ∧ x = 63 :=
by
  sorry

end find_asterisk_l1818_181887


namespace rowing_speed_in_still_water_l1818_181839

theorem rowing_speed_in_still_water (v c : ℝ) (h1 : c = 1.5) 
(h2 : ∀ t : ℝ, (v + c) * t = (v - c) * 2 * t) : 
  v = 4.5 :=
by
  sorry

end rowing_speed_in_still_water_l1818_181839


namespace min_value_l1818_181881

-- Define the conditions
variables (x y : ℝ)
-- Assume x and y are in the positive real numbers
axiom pos_x : 0 < x
axiom pos_y : 0 < y
-- Given equation
axiom eq1 : x + 2 * y = 2 * x * y

-- The goal is to prove that the minimum value of 3x + 4y is 5 + 2sqrt(6)
theorem min_value (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (eq1 : x + 2 * y = 2 * x * y) : 
  3 * x + 4 * y ≥ 5 + 2 * Real.sqrt 6 := 
sorry

end min_value_l1818_181881


namespace intersection_unique_point_l1818_181874

def f (x : ℝ) : ℝ := x^3 + 6 * x^2 + 16 * x + 28

theorem intersection_unique_point :
  ∃ a : ℝ, f a = a ∧ a = -4 := sorry

end intersection_unique_point_l1818_181874


namespace average_goals_increase_l1818_181819

theorem average_goals_increase (A : ℚ) (h1 : 4 * A + 2 = 4) : (4 / 5 - A) = 0.3 := by
  sorry

end average_goals_increase_l1818_181819


namespace find_c_l1818_181864

variable {r s b c : ℚ}

-- Conditions based on roots of the original quadratic equation
def roots_of_original_quadratic (r s : ℚ) := 
  (5 * r ^ 2 - 8 * r + 2 = 0) ∧ (5 * s ^ 2 - 8 * s + 2 = 0)

-- New quadratic equation with roots shifted by 3
def new_quadratic_roots (r s b c : ℚ) :=
  (r - 3) + (s - 3) = -b ∧ (r - 3) * (s - 3) = c 

theorem find_c (r s : ℚ) (hb : b = 22/5) : 
  (roots_of_original_quadratic r s) → 
  (new_quadratic_roots r s b c) → 
  c = 23/5 := 
by
  intros h1 h2
  sorry

end find_c_l1818_181864


namespace first_common_digit_three_digit_powers_l1818_181838

theorem first_common_digit_three_digit_powers (m n: ℕ) (hm: 100 ≤ 2^m ∧ 2^m < 1000) (hn: 100 ≤ 3^n ∧ 3^n < 1000) :
  (∃ d, (2^m).div 100 = d ∧ (3^n).div 100 = d ∧ d = 2) :=
sorry

end first_common_digit_three_digit_powers_l1818_181838


namespace Alpha_Beta_meet_at_Alpha_Beta_meet_again_l1818_181897

open Real

-- Definitions and conditions
def A : ℝ := -24
def B : ℝ := -10
def C : ℝ := 10
def Alpha_speed : ℝ := 4
def Beta_speed : ℝ := 6

-- Question 1: Prove that Alpha and Beta meet at -10.4
theorem Alpha_Beta_meet_at : 
  ∃ t : ℝ, (A + Alpha_speed * t = C - Beta_speed * t) ∧ (A + Alpha_speed * t = -10.4) :=
  sorry

-- Question 2: Prove that after reversing at t = 2, Alpha and Beta meet again at -44
theorem Alpha_Beta_meet_again :
  ∃ t z : ℝ, 
    ((t = 2) ∧ (4 * t + (14 - 4 * t) + (14 - 4 * t + 20) = 40) ∧ 
     (A + Alpha_speed * t - Alpha_speed * z = C - Beta_speed * t - Beta_speed * z) ∧ 
     (A + Alpha_speed * t - Alpha_speed * z = -44)) :=
  sorry  

end Alpha_Beta_meet_at_Alpha_Beta_meet_again_l1818_181897


namespace total_sums_attempted_l1818_181813

-- Define the necessary conditions
def num_sums_right : ℕ := 8
def num_sums_wrong : ℕ := 2 * num_sums_right

-- Define the theorem to prove
theorem total_sums_attempted : num_sums_right + num_sums_wrong = 24 := by
  sorry

end total_sums_attempted_l1818_181813


namespace product_of_two_numbers_l1818_181893

noncomputable def leastCommonMultiple (a b : ℕ) : ℕ :=
  Nat.lcm a b

noncomputable def greatestCommonDivisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem product_of_two_numbers (a b : ℕ) :
  leastCommonMultiple a b = 36 ∧ greatestCommonDivisor a b = 6 → a * b = 216 := by
  sorry

end product_of_two_numbers_l1818_181893


namespace part1_part2_l1818_181855

noncomputable def U : Set ℝ := Set.univ

noncomputable def A (a: ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
noncomputable def B (a: ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

theorem part1 (a : ℝ) (ha : a = 1/2) :
  (U \ (B a)) ∩ (A a) = {x | 9/4 ≤ x ∧ x < 5/2} :=
sorry

theorem part2 (p q : ℝ → Prop)
  (hp : ∀ x, p x → x ∈ A a) (hq : ∀ x, q x → x ∈ B a)
  (hq_necessary : ∀ x, p x → q x) :
  -1/2 ≤ a ∧ a ≤ (3 - Real.sqrt 5) / 2 :=
sorry

end part1_part2_l1818_181855


namespace difference_in_price_l1818_181882

-- Definitions based on the given conditions
def price_with_cork : ℝ := 2.10
def price_cork : ℝ := 0.05
def price_without_cork : ℝ := price_with_cork - price_cork

-- The theorem proving the given question and correct answer
theorem difference_in_price : price_with_cork - price_without_cork = price_cork :=
by
  -- Proof can be omitted
  sorry

end difference_in_price_l1818_181882


namespace intersection_point_k_value_l1818_181833

theorem intersection_point_k_value :
  (∃ (k : ℝ), (∀ (x y : ℝ),
    ((y = 2 * x + 3 ∧ y = k * x + 2) → (x = 1 ∧ y = 5))) → k = 3) :=
sorry

end intersection_point_k_value_l1818_181833


namespace guppies_total_l1818_181879

theorem guppies_total :
  let haylee := 3 * 12
  let jose := haylee / 2
  let charliz := jose / 3
  let nicolai := charliz * 4
  haylee + jose + charliz + nicolai = 84 :=
by
  sorry

end guppies_total_l1818_181879


namespace like_terms_implies_m_minus_n_l1818_181844

/-- If 4x^(2m+2)y^(n-1) and -3x^(3m+1)y^(3n-5) are like terms, then m - n = -1. -/
theorem like_terms_implies_m_minus_n
  (m n : ℤ)
  (h1 : 2 * m + 2 = 3 * m + 1)
  (h2 : n - 1 = 3 * n - 5) :
  m - n = -1 :=
by
  sorry

end like_terms_implies_m_minus_n_l1818_181844


namespace no_polynomial_transformation_l1818_181832

-- Define the problem conditions: initial and target sequences
def initial_seq : List ℤ := [-3, -1, 1, 3]
def target_seq : List ℤ := [-3, -1, -3, 3]

-- State the main theorem to be proved
theorem no_polynomial_transformation :
  ¬ (∃ (P : ℤ → ℤ), ∀ x ∈ initial_seq, P x ∈ target_seq) :=
  sorry

end no_polynomial_transformation_l1818_181832


namespace vector_arithmetic_l1818_181808

theorem vector_arithmetic (a b : ℝ × ℝ)
    (h₀ : a = (3, 5))
    (h₁ : b = (-2, 1)) :
    a - (2 : ℝ) • b = (7, 3) :=
sorry

end vector_arithmetic_l1818_181808


namespace rental_property_key_count_l1818_181807

def number_of_keys (complexes apartments_per_complex keys_per_lock locks_per_apartment : ℕ) : ℕ :=
  complexes * apartments_per_complex * keys_per_lock * locks_per_apartment

theorem rental_property_key_count : 
  number_of_keys 2 12 3 1 = 72 := by
  sorry

end rental_property_key_count_l1818_181807


namespace inequalities_proof_l1818_181820

theorem inequalities_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (a < (c / 2)) ∧ (b < a + c / 2) ∧ ¬(b < c / 2) :=
by
  constructor
  { sorry }
  { constructor
    { sorry }
    { sorry } }

end inequalities_proof_l1818_181820


namespace inequality_lemma_l1818_181818

theorem inequality_lemma (x y z : ℝ) (h1 : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z)
    (h2 : (1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1)) :
    (1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1) := 
by
  sorry

end inequality_lemma_l1818_181818


namespace simplify_polynomial_l1818_181850

theorem simplify_polynomial (s : ℝ) :
  (2*s^2 + 5*s - 3) - (2*s^2 + 9*s - 7) = -4*s + 4 :=
by
  sorry

end simplify_polynomial_l1818_181850


namespace final_ratio_of_milk_to_water_l1818_181899

-- Initial conditions definitions
def initial_milk_ratio : ℚ := 5 / 8
def initial_water_ratio : ℚ := 3 / 8
def additional_milk : ℚ := 8
def total_capacity : ℚ := 72

-- Final ratio statement
theorem final_ratio_of_milk_to_water :
  (initial_milk_ratio * (total_capacity - additional_milk) + additional_milk) / (initial_water_ratio * (total_capacity - additional_milk)) = 2 := by
  sorry

end final_ratio_of_milk_to_water_l1818_181899


namespace geometric_sequence_S5_l1818_181890

noncomputable def S5 (a₁ q : ℝ) : ℝ :=
  a₁ * (1 - q^5) / (1 - q)

theorem geometric_sequence_S5 
  (a₁ q : ℝ) 
  (h₁ : a₁ * (1 + q) = 3 / 4)
  (h₄ : a₁ * q^3 * (1 + q) = 6) :
  S5 a₁ q = 31 / 4 := 
sorry

end geometric_sequence_S5_l1818_181890


namespace value_of_a_plus_b_l1818_181880

noncomputable def f (a b x : ℝ) := x / (a * x + b)

theorem value_of_a_plus_b (a b : ℝ) (h₁: a ≠ 0) (h₂: f a b (-4) = 4)
    (h₃: ∀ x, f a b (f a b x) = x) : a + b = 3 / 2 :=
sorry

end value_of_a_plus_b_l1818_181880


namespace gain_percent_is_87_point_5_l1818_181872

noncomputable def gain_percent (C S : ℝ) : ℝ :=
  ((S - C) / C) * 100

theorem gain_percent_is_87_point_5 {C S : ℝ} (h : 75 * C = 40 * S) :
  gain_percent C S = 87.5 :=
by
  sorry

end gain_percent_is_87_point_5_l1818_181872
