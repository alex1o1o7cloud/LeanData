import Mathlib

namespace inverse_proportion_quadrants_l1327_132728

theorem inverse_proportion_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∃ x y : ℝ, x = -2 ∧ y = 3 ∧ y = k / x) →
  (∀ x : ℝ, (x < 0 → k / x > 0) ∧ (x > 0 → k / x < 0)) :=
sorry

end inverse_proportion_quadrants_l1327_132728


namespace real_part_of_product_l1327_132772

open Complex

theorem real_part_of_product (α β : ℝ) :
  let z1 := Complex.mk (Real.cos α) (Real.sin α)
  let z2 := Complex.mk (Real.cos β) (Real.sin β)
  Complex.re (z1 * z2) = Real.cos (α + β) :=
by
  let z1 := Complex.mk (Real.cos α) (Real.sin α)
  let z2 := Complex.mk (Real.cos β) (Real.sin β)
  sorry

end real_part_of_product_l1327_132772


namespace possible_measures_of_angle_X_l1327_132732

theorem possible_measures_of_angle_X :
  ∃ (n : ℕ), n = 17 ∧ ∀ (X Y : ℕ), 
    (X > 0) → 
    (Y > 0) → 
    (∃ k : ℕ, k ≥ 1 ∧ X = k * Y) → 
    X + Y = 180 → 
    ∃ d : ℕ, d ∈ {d | d ∣ 180 } ∧ d ≥ 2 :=
by
  sorry

end possible_measures_of_angle_X_l1327_132732


namespace cooler_capacity_l1327_132779

theorem cooler_capacity (linemen: ℕ) (linemen_drink: ℕ) 
                        (skill_position: ℕ) (skill_position_drink: ℕ) 
                        (linemen_count: ℕ) (skill_position_count: ℕ) 
                        (skill_wait: ℕ) 
                        (h1: linemen_count = 12) 
                        (h2: linemen_drink = 8) 
                        (h3: skill_position_count = 10) 
                        (h4: skill_position_drink = 6) 
                        (h5: skill_wait = 5):
 linemen_count * linemen_drink + skill_wait * skill_position_drink = 126 :=
by
  sorry

end cooler_capacity_l1327_132779


namespace number_of_flowers_alissa_picked_l1327_132798

-- Define the conditions
variable (A : ℕ) -- Number of flowers Alissa picked
variable (M : ℕ) -- Number of flowers Melissa picked
variable (flowers_gifted : ℕ := 18) -- Flowers given to mother
variable (flowers_left : ℕ := 14) -- Flowers left after gifting

-- Define that Melissa picked the same number of flowers as Alissa
axiom pick_equal : M = A

-- Define the total number of flowers they had initially
axiom total_flowers : 2 * A = flowers_gifted + flowers_left

-- Prove that Alissa picked 16 flowers
theorem number_of_flowers_alissa_picked : A = 16 := by
  -- Use placeholders for proof steps
  sorry

end number_of_flowers_alissa_picked_l1327_132798


namespace sedrach_divides_each_pie_l1327_132712

theorem sedrach_divides_each_pie (P : ℕ) :
  (13 * P * 5 = 130) → P = 2 :=
by
  sorry

end sedrach_divides_each_pie_l1327_132712


namespace total_number_of_cards_l1327_132770

theorem total_number_of_cards (groups : ℕ) (cards_per_group : ℕ) (h_groups : groups = 9) (h_cards_per_group : cards_per_group = 8) : groups * cards_per_group = 72 := by
  sorry

end total_number_of_cards_l1327_132770


namespace unique_positive_integer_satisfies_condition_l1327_132769

def is_positive_integer (n : ℕ) : Prop := n > 0

def condition (n : ℕ) : Prop := 20 - 5 * n ≥ 15

theorem unique_positive_integer_satisfies_condition :
  ∃! n : ℕ, is_positive_integer n ∧ condition n :=
by
  sorry

end unique_positive_integer_satisfies_condition_l1327_132769


namespace geometric_sequence_eleventh_term_l1327_132730

theorem geometric_sequence_eleventh_term (a₁ : ℚ) (r : ℚ) (n : ℕ) (hₐ : a₁ = 5) (hᵣ : r = 2 / 3) (hₙ : n = 11) :
  (a₁ * r^(n - 1) = 5120 / 59049) :=
by
  -- conditions of the problem
  rw [hₐ, hᵣ, hₙ]
  sorry

end geometric_sequence_eleventh_term_l1327_132730


namespace probability_of_spade_then_king_l1327_132746

theorem probability_of_spade_then_king :
  ( (24 / 104) * (8 / 103) + (2 / 104) * (7 / 103) ) = 103 / 5356 :=
sorry

end probability_of_spade_then_king_l1327_132746


namespace tangent_line_equation_l1327_132705

open Real

noncomputable def circle_center : ℝ × ℝ := (2, 1)
noncomputable def tangent_point : ℝ × ℝ := (4, 3)

def circle_equation (x y : ℝ) : Prop := (x - circle_center.1)^2 + (y - circle_center.2)^2 = 1

theorem tangent_line_equation :
  ∀ (x y : ℝ), ( (x = 4 ∧ y = 3) ∨ circle_equation x y ) → 2 * x + 2 * y - 7 = 0 :=
sorry

end tangent_line_equation_l1327_132705


namespace num_ordered_pairs_l1327_132799

theorem num_ordered_pairs (N : ℕ) :
  (N = 20) ↔ ∃ (a b : ℕ), 
  (a < b) ∧ (100 ≤ a ∧ a ≤ 1000)
  ∧ (100 ≤ b ∧ b ≤ 1000)
  ∧ (gcd a b * lcm a b = 495 * gcd a b)
  := 
sorry

end num_ordered_pairs_l1327_132799


namespace max_value_of_M_l1327_132729

theorem max_value_of_M (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (2 * x + y) + y / (2 * y + z) + z / (2 * z + x)) ≤ 1 :=
sorry -- Proof placeholder

end max_value_of_M_l1327_132729


namespace f_at_2_f_shifted_range_f_shifted_l1327_132758

def f (x : ℝ) := x^2 - 2*x + 7

-- 1) Prove that f(2) = 7
theorem f_at_2 : f 2 = 7 := sorry

-- 2) Prove the expressions for f(x-1) and f(x+1)
theorem f_shifted (x : ℝ) : f (x-1) = x^2 - 4*x + 10 ∧ f (x+1) = x^2 + 6 := sorry

-- 3) Prove the range of f(x+1) is [6, +∞)
theorem range_f_shifted : ∀ x, f (x+1) ≥ 6 := sorry

end f_at_2_f_shifted_range_f_shifted_l1327_132758


namespace value_of_x_plus_2y_l1327_132721

theorem value_of_x_plus_2y 
  (x y : ℝ) 
  (h : (x + 5)^2 = -(|y - 2|)) : 
  x + 2 * y = -1 :=
sorry

end value_of_x_plus_2y_l1327_132721


namespace absolute_value_half_l1327_132725

theorem absolute_value_half (a : ℝ) (h : |a| = 1/2) : a = 1/2 ∨ a = -1/2 :=
sorry

end absolute_value_half_l1327_132725


namespace no_such_divisor_l1327_132708

theorem no_such_divisor (n : ℕ) : 
  (n ∣ (823435 : ℕ)^15) ∧ (n^5 - n^n = 1) → false := 
by sorry

end no_such_divisor_l1327_132708


namespace greatest_difference_l1327_132707

-- Definitions: Number of marbles in each basket
def basketA_red : Nat := 4
def basketA_yellow : Nat := 2
def basketB_green : Nat := 6
def basketB_yellow : Nat := 1
def basketC_white : Nat := 3
def basketC_yellow : Nat := 9

-- Define the differences
def diff_basketA : Nat := basketA_red - basketA_yellow
def diff_basketB : Nat := basketB_green - basketB_yellow
def diff_basketC : Nat := basketC_yellow - basketC_white

-- The goal is to prove that 6 is the greatest difference
theorem greatest_difference : max (max diff_basketA diff_basketB) diff_basketC = 6 :=
by 
  -- The proof is not provided
  sorry

end greatest_difference_l1327_132707


namespace distribute_a_eq_l1327_132755

variable (a b c : ℝ)

theorem distribute_a_eq : a * (a + b - c) = a^2 + a * b - a * c := 
sorry

end distribute_a_eq_l1327_132755


namespace solution_set_l1327_132789

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x - 2) / (x - 4) ≥ 3

theorem solution_set :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | 4 < x ∧ x ≤ 5} :=
by
  sorry

end solution_set_l1327_132789


namespace balance_blue_balls_l1327_132790

variables (G Y W R B : ℕ)

axiom green_balance : 3 * G = 6 * B
axiom yellow_balance : 2 * Y = 5 * B
axiom white_balance : 6 * B = 4 * W
axiom red_balance : 4 * R = 10 * B

theorem balance_blue_balls : 5 * G + 3 * Y + 3 * W + 2 * R = 27 * B :=
  by
  sorry

end balance_blue_balls_l1327_132790


namespace intersection_of_A_and_B_l1327_132720

-- Definitions of the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

-- Statement to prove the intersection of sets A and B is {3}
theorem intersection_of_A_and_B : A ∩ B = {3} :=
sorry

end intersection_of_A_and_B_l1327_132720


namespace slope_intercept_product_l1327_132748

theorem slope_intercept_product (b m : ℤ) (h1 : b = -3) (h2 : m = 3) : m * b = -9 := by
  sorry

end slope_intercept_product_l1327_132748


namespace total_pennies_thrown_l1327_132724

theorem total_pennies_thrown (R G X M T : ℝ) (hR : R = 1500)
  (hG : G = (2 / 3) * R) (hX : X = (3 / 4) * G) 
  (hM : M = 3.5 * X) (hT : T = (4 / 5) * M) : 
  R + G + X + M + T = 7975 :=
by
  sorry

end total_pennies_thrown_l1327_132724


namespace fewest_handshakes_organizer_l1327_132756

theorem fewest_handshakes_organizer (n k : ℕ) (h : k < n) 
  (total_handshakes: n*(n-1)/2 + k = 406) :
  k = 0 :=
sorry

end fewest_handshakes_organizer_l1327_132756


namespace solve_inequalities_l1327_132726

/-- Solve the inequality system and find all non-negative integer solutions. -/
theorem solve_inequalities :
  { x : ℤ | 0 ≤ x ∧ 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {0, 1, 2} :=
by
  sorry

end solve_inequalities_l1327_132726


namespace man_rate_in_still_water_l1327_132776

theorem man_rate_in_still_water (V_m V_s: ℝ) 
(h1 : V_m + V_s = 19) 
(h2 : V_m - V_s = 11) : 
V_m = 15 := 
by
  sorry

end man_rate_in_still_water_l1327_132776


namespace distance_between_cities_l1327_132745

theorem distance_between_cities 
  (t : ℝ)
  (h1 : 60 * t = 70 * (t - 1 / 4)) 
  (d : ℝ) : 
  d = 105 := by
sorry

end distance_between_cities_l1327_132745


namespace coefficient_ratio_is_4_l1327_132706

noncomputable def coefficient_x3 := 
  let a := 60 -- Coefficient of x^3 in the expansion
  let b := Nat.choose 6 2 -- Binomial coefficient \binom{6}{2}
  a / b

theorem coefficient_ratio_is_4 : coefficient_x3 = 4 := by
  sorry

end coefficient_ratio_is_4_l1327_132706


namespace cos_alpha_minus_beta_cos_alpha_plus_beta_l1327_132786

variables (α β : Real) (h1 : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2)
           (h2 : Real.tan α * Real.tan β = 13/7)
           (h3 : Real.sin (α - β) = sqrt 5 / 3)

-- Part (1): Prove that cos (α - β) = 2/3
theorem cos_alpha_minus_beta : Real.cos (α - β) = 2 / 3 := by
  have h := h1
  have h := h2
  have h := h3
  sorry

-- Part (2): Prove that cos (α + β) = -1/5
theorem cos_alpha_plus_beta : Real.cos (α + β) = -1 / 5 := by
  have h := h1
  have h := h2
  have h := h3
  sorry

end cos_alpha_minus_beta_cos_alpha_plus_beta_l1327_132786


namespace calculate_unoccupied_volume_l1327_132781

def tank_length : ℕ := 12
def tank_width : ℕ := 10
def tank_height : ℕ := 8
def tank_volume : ℕ := tank_length * tank_width * tank_height

def water_volume : ℕ := tank_volume / 3
def ice_cube_volume : ℕ := 1
def ice_cubes_count : ℕ := 12
def total_ice_volume : ℕ := ice_cubes_count * ice_cube_volume
def occupied_volume : ℕ := water_volume + total_ice_volume

def unoccupied_volume : ℕ := tank_volume - occupied_volume

theorem calculate_unoccupied_volume : unoccupied_volume = 628 := by
  sorry

end calculate_unoccupied_volume_l1327_132781


namespace equal_volume_cubes_l1327_132766

noncomputable def volume_box : ℝ := 1 -- volume of the cubical box in cubic meters

noncomputable def edge_length_small_cube : ℝ := 0.04 -- edge length of small cubes in meters

noncomputable def number_of_cubes : ℝ := 15624.999999999998 -- number of small cubes

noncomputable def volume_small_cube : ℝ := edge_length_small_cube^3 -- volume of one small cube

theorem equal_volume_cubes : volume_box = volume_small_cube * number_of_cubes :=
  by
  -- Proof goes here
  sorry

end equal_volume_cubes_l1327_132766


namespace rectangle_area_l1327_132702

theorem rectangle_area (side_length width length : ℝ) (h_square_area : side_length^2 = 36)
  (h_width : width = side_length) (h_length : length = 2.5 * width) :
  width * length = 90 :=
by 
  sorry

end rectangle_area_l1327_132702


namespace solution_to_system_of_eqns_l1327_132754

theorem solution_to_system_of_eqns (x y z : ℝ) :
  (x = (2 * z ^ 2) / (1 + z ^ 2) ∧ y = (2 * x ^ 2) / (1 + x ^ 2) ∧ z = (2 * y ^ 2) / (1 + y ^ 2)) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end solution_to_system_of_eqns_l1327_132754


namespace find_S2_side_length_l1327_132750

theorem find_S2_side_length 
    (x r : ℝ)
    (h1 : 2 * r + x = 2100)
    (h2 : 3 * x + 300 = 3500)
    : x = 1066.67 := 
sorry

end find_S2_side_length_l1327_132750


namespace part_a_part_b_l1327_132785

-- Conditions
def has_three_classmates_in_any_group_of_ten (students : Fin 60 → Type) : Prop :=
  ∀ (g : Finset (Fin 60)), g.card = 10 → ∃ (a b c : Fin 60), a ∈ g ∧ b ∈ g ∧ c ∈ g ∧ students a = students b ∧ students b = students c

-- Part (a)
theorem part_a (students : Fin 60 → Type) (h : has_three_classmates_in_any_group_of_ten students) : ∃ g : Finset (Fin 60), g.card ≥ 15 ∧ ∀ a b : Fin 60, a ∈ g → b ∈ g → students a = students b :=
sorry

-- Part (b)
theorem part_b (students : Fin 60 → Type) (h : has_three_classmates_in_any_group_of_ten students) : ¬ ∃ g : Finset (Fin 60), g.card ≥ 16 ∧ ∀ a b : Fin 60, a ∈ g → b ∈ g → students a = students b :=
sorry

end part_a_part_b_l1327_132785


namespace parabola_and_line_sum_l1327_132713

theorem parabola_and_line_sum (A B F : ℝ × ℝ)
  (h_parabola : ∀ x y : ℝ, (y^2 = 4 * x) ↔ (x, y) = A ∨ (x, y) = B)
  (h_line : ∀ x y : ℝ, (2 * x + y - 4 = 0) ↔ (x, y) = A ∨ (x, y) = B)
  (h_focus : F = (1, 0))
  : |F - A| + |F - B| = 7 := 
sorry

end parabola_and_line_sum_l1327_132713


namespace henry_age_l1327_132717

theorem henry_age (H J : ℕ) 
  (sum_ages : H + J = 40) 
  (age_relation : H - 11 = 2 * (J - 11)) : 
  H = 23 := 
sorry

end henry_age_l1327_132717


namespace line_equation_through_origin_and_circle_chord_length_l1327_132737

theorem line_equation_through_origin_and_circle_chord_length 
  (x y : ℝ) 
  (h : x^2 + y^2 - 2 * x - 4 * y + 4 = 0) 
  (chord_length : ℝ) 
  (h_chord : chord_length = 2) 
  : 2 * x - y = 0 := 
sorry

end line_equation_through_origin_and_circle_chord_length_l1327_132737


namespace families_seating_arrangements_l1327_132764

theorem families_seating_arrangements : 
  let factorial := Nat.factorial
  let family_ways := factorial 3
  let bundles := family_ways * family_ways * family_ways
  let bundle_ways := factorial 3
  bundles * bundle_ways = (factorial 3) ^ 4 := by
  sorry

end families_seating_arrangements_l1327_132764


namespace value_of_a2_l1327_132701

variable {R : Type*} [Ring R] (x a_0 a_1 a_2 a_3 : R)

theorem value_of_a2 
  (h : ∀ x : R, x^3 = a_0 + a_1 * (x - 2) + a_2 * (x - 2)^2 + a_3 * (x - 2)^3) :
  a_2 = 6 :=
sorry

end value_of_a2_l1327_132701


namespace total_kids_played_l1327_132718

theorem total_kids_played (kids_monday : ℕ) (kids_tuesday : ℕ) (h_monday : kids_monday = 4) (h_tuesday : kids_tuesday = 14) : 
  kids_monday + kids_tuesday = 18 := 
by
  -- proof steps here (for now, use sorry to skip the proof)
  sorry

end total_kids_played_l1327_132718


namespace find_n_from_equation_l1327_132700

theorem find_n_from_equation (n m : ℕ) (h1 : (1^m / 5^m) * (1^n / 4^n) = 1 / (2 * 10^31)) (h2 : m = 31) : n = 16 := 
by
  sorry

end find_n_from_equation_l1327_132700


namespace gcd_2814_1806_l1327_132763

def a := 2814
def b := 1806

theorem gcd_2814_1806 : Nat.gcd a b = 42 :=
by
  sorry

end gcd_2814_1806_l1327_132763


namespace f_2015_eq_neg_2014_l1327_132704

variable {f : ℝ → ℝ}

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def isPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x
def f1_value : f 1 = 2014 := sorry

-- Theorem to prove
theorem f_2015_eq_neg_2014 :
  isOddFunction f → isPeriodic f 3 → (f 1 = 2014) → f 2015 = -2014 :=
by
  intros hOdd hPeriodic hF1
  sorry

end f_2015_eq_neg_2014_l1327_132704


namespace grace_crayon_selection_l1327_132714

def crayons := {i // 1 ≤ i ∧ i ≤ 15}
def red_crayons := {i // 1 ≤ i ∧ i ≤ 3}

def total_ways := Nat.choose 15 5
def non_favorable := Nat.choose 12 5

theorem grace_crayon_selection : total_ways - non_favorable = 2211 :=
by
  sorry

end grace_crayon_selection_l1327_132714


namespace sum_T_19_34_51_l1327_132715

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2 : ℕ) else (n + 1) / 2

def T (n : ℕ) : ℤ :=
  2 + S n

theorem sum_T_19_34_51 : T 19 + T 34 + T 51 = 25 := 
by
  -- Add the steps here
  sorry

end sum_T_19_34_51_l1327_132715


namespace smallest_n_divisible_l1327_132711

open Nat

theorem smallest_n_divisible (n : ℕ) : (∃ (n : ℕ), n > 0 ∧ 45 ∣ n^2 ∧ 720 ∣ n^3) → n = 60 :=
by
  sorry

end smallest_n_divisible_l1327_132711


namespace work_ratio_of_man_to_boy_l1327_132709

theorem work_ratio_of_man_to_boy 
  (M B : ℝ) 
  (work : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = work)
  (h2 : (13 * M + 24 * B) * 4 = work) :
  M / B = 2 :=
by 
  sorry

end work_ratio_of_man_to_boy_l1327_132709


namespace neither_coffee_tea_juice_l1327_132791

open Set

theorem neither_coffee_tea_juice (total : ℕ) (coffee : ℕ) (tea : ℕ) (both_coffee_tea : ℕ)
  (juice : ℕ) (juice_and_tea_not_coffee : ℕ) :
  total = 35 → 
  coffee = 18 → 
  tea = 15 → 
  both_coffee_tea = 7 → 
  juice = 6 → 
  juice_and_tea_not_coffee = 3 →
  (total - ((coffee + tea - both_coffee_tea) + (juice - juice_and_tea_not_coffee))) = 6 :=
sorry

end neither_coffee_tea_juice_l1327_132791


namespace cat_chase_rat_l1327_132793

/--
Given:
- The cat chases a rat 6 hours after the rat runs.
- The cat takes 4 hours to reach the rat.
- The average speed of the rat is 36 km/h.
Prove that the average speed of the cat is 90 km/h.
-/
theorem cat_chase_rat
  (t_rat_start : ℕ)
  (t_cat_chase : ℕ)
  (v_rat : ℕ)
  (h1 : t_rat_start = 6)
  (h2 : t_cat_chase = 4)
  (h3 : v_rat = 36)
  (v_cat : ℕ)
  (h4 : 4 * v_cat = t_rat_start * v_rat + t_cat_chase * v_rat) :
  v_cat = 90 :=
by
  sorry

end cat_chase_rat_l1327_132793


namespace range_of_f_l1327_132782

-- Define the function f
def f (x : ℕ) : ℤ := x^2 - 2 * x

-- Define the domain
def domain : Finset ℕ := {0, 1, 2, 3}

-- Define the expected range
def expected_range : Finset ℤ := {-1, 0, 3}

-- State the theorem
theorem range_of_f : (domain.image f) = expected_range := by
  sorry

end range_of_f_l1327_132782


namespace correct_comparison_l1327_132783

-- Definitions of conditions based on the problem 
def hormones_participate : Prop := false 
def enzymes_produced_by_living_cells : Prop := true 
def hormones_produced_by_endocrine : Prop := true 
def endocrine_can_produce_both : Prop := true 
def synthesize_enzymes_not_nec_hormones : Prop := true 
def not_all_proteins : Prop := true 

-- Statement of the equivalence between the correct answer and its proof
theorem correct_comparison :  (¬hormones_participate ∧ enzymes_produced_by_living_cells ∧ hormones_produced_by_endocrine ∧ endocrine_can_produce_both ∧ synthesize_enzymes_not_nec_hormones ∧ not_all_proteins) → (endocrine_can_produce_both) :=
by
  sorry

end correct_comparison_l1327_132783


namespace average_brown_MnMs_l1327_132780

theorem average_brown_MnMs 
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 9)
  (h2 : a2 = 12)
  (h3 : a3 = 8)
  (h4 : a4 = 8)
  (h5 : a5 = 3) : 
  (a1 + a2 + a3 + a4 + a5) / 5 = 8 :=
by
  sorry

end average_brown_MnMs_l1327_132780


namespace sum_of_inscribed_sphere_volumes_l1327_132703

theorem sum_of_inscribed_sphere_volumes :
  let height := 3
  let angle := Real.pi / 3
  let r₁ := height / 3 -- Radius of the first inscribed sphere
  let geometric_ratio := 1 / 3
  let volume (r : ℝ) := (4 / 3) * Real.pi * r^3
  let volumes : ℕ → ℝ := λ n => volume (r₁ * geometric_ratio^(n - 1))
  let total_volume := ∑' n, volumes n
  total_volume = (18 * Real.pi) / 13 :=
by
  sorry

end sum_of_inscribed_sphere_volumes_l1327_132703


namespace decreased_cost_proof_l1327_132731

def original_cost : ℝ := 200
def percentage_decrease : ℝ := 0.5
def decreased_cost (original_cost : ℝ) (percentage_decrease : ℝ) : ℝ := 
  original_cost - (percentage_decrease * original_cost)

theorem decreased_cost_proof : decreased_cost original_cost percentage_decrease = 100 := 
by { 
  sorry -- Proof is not required
}

end decreased_cost_proof_l1327_132731


namespace find_least_positive_x_l1327_132784

theorem find_least_positive_x :
  ∃ x : ℕ, 0 < x ∧ (x + 5713) % 15 = 1847 % 15 ∧ x = 4 :=
by
  sorry

end find_least_positive_x_l1327_132784


namespace find_a_l1327_132760

def M : Set ℝ := {-1, 0, 1}

def N (a : ℝ) : Set ℝ := {a, a^2}

theorem find_a (a : ℝ) : N a ⊆ M → a = -1 :=
by
  sorry

end find_a_l1327_132760


namespace farm_entrance_fee_for_students_is_five_l1327_132794

theorem farm_entrance_fee_for_students_is_five
  (students : ℕ) (adults : ℕ) (adult_fee : ℕ) (total_cost : ℕ) (student_fee : ℕ)
  (h_students : students = 35)
  (h_adults : adults = 4)
  (h_adult_fee : adult_fee = 6)
  (h_total_cost : total_cost = 199)
  (h_equation : students * student_fee + adults * adult_fee = total_cost) :
  student_fee = 5 :=
by
  sorry

end farm_entrance_fee_for_students_is_five_l1327_132794


namespace slope_of_line_eq_slope_of_line_l1327_132716

theorem slope_of_line_eq (x y : ℝ) (h : 4 * x + 6 * y = 24) : (6 * y = -4 * x + 24) → (y = - (2 : ℝ) / 3 * x + 4) :=
by
  intro h1
  sorry

theorem slope_of_line (x y m : ℝ) (h1 : 4 * x + 6 * y = 24) (h2 : y = - (2 : ℝ) / 3 * x + 4) : m = - (2 : ℝ) / 3 :=
by
  sorry

end slope_of_line_eq_slope_of_line_l1327_132716


namespace angle_y_value_l1327_132775

theorem angle_y_value (ABC ABD ABE BAE y : ℝ) (h1 : ABC = 180) (h2 : ABD = 66) 
  (h3 : ABE = 114) (h4 : BAE = 31) (h5 : 31 + 114 + y = 180) : y = 35 :=
  sorry

end angle_y_value_l1327_132775


namespace value_of_a_l1327_132773

theorem value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - x - 2 < 0 ↔ -2 < x ∧ x < a) → (a = 2 ∨ a = 3 ∨ a = 4) :=
by sorry

end value_of_a_l1327_132773


namespace inequality_for_any_x_l1327_132762

theorem inequality_for_any_x (a : ℝ) (h : ∀ x : ℝ, |3 * x + 2 * a| + |2 - 3 * x| - |a + 1| > 2) :
  a < -1/3 ∨ a > 5 := 
sorry

end inequality_for_any_x_l1327_132762


namespace greatest_integer_not_exceeding_a_l1327_132777

theorem greatest_integer_not_exceeding_a (a : ℝ) (h : 3^a + a^3 = 123) : ⌊a⌋ = 4 :=
sorry

end greatest_integer_not_exceeding_a_l1327_132777


namespace find_ordered_pair_l1327_132768

theorem find_ordered_pair (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^y + 1 = y^x) (h2 : 2 * x^y = y^x + 13) : (x = 2 ∧ y = 4) :=
by {
  sorry
}

end find_ordered_pair_l1327_132768


namespace half_angle_in_second_quadrant_l1327_132759

theorem half_angle_in_second_quadrant (α : ℝ) (h : 180 < α ∧ α < 270) : 90 < α / 2 ∧ α / 2 < 135 := 
by
  sorry

end half_angle_in_second_quadrant_l1327_132759


namespace rhombus_area_l1327_132742

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 12) : (d1 * d2) / 2 = 90 := by
  sorry

end rhombus_area_l1327_132742


namespace ellipse_equation_l1327_132744

theorem ellipse_equation
  (x y t : ℝ)
  (h1 : x = (3 * (Real.sin t - 2)) / (3 - Real.cos t))
  (h2 : y = (4 * (Real.cos t - 6)) / (3 - Real.cos t))
  (h3 : ∀ t : ℝ, (Real.cos t)^2 + (Real.sin t)^2 = 1) :
  ∃ (A B C D E F : ℤ), (9 * x^2 + 36 * x * y + 9 * y^2 + 216 * x + 432 * y + 1440 = 0) ∧ 
  (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd A B) C) D) E) F = 1) ∧
  (|A| + |B| + |C| + |D| + |E| + |F| = 2142) :=
sorry

end ellipse_equation_l1327_132744


namespace factorize_expr_l1327_132788

-- Define the expression
def expr (a : ℝ) := -3 * a + 12 * a^2 - 12 * a^3

-- State the theorem
theorem factorize_expr (a : ℝ) : expr a = -3 * a * (1 - 2 * a)^2 :=
by
  sorry

end factorize_expr_l1327_132788


namespace marley_total_fruits_l1327_132722

theorem marley_total_fruits (louis_oranges : ℕ) (louis_apples : ℕ) 
                            (samantha_oranges : ℕ) (samantha_apples : ℕ)
                            (marley_oranges : ℕ) (marley_apples : ℕ) : 
  (louis_oranges = 5) → (louis_apples = 3) → 
  (samantha_oranges = 8) → (samantha_apples = 7) → 
  (marley_oranges = 2 * louis_oranges) → (marley_apples = 3 * samantha_apples) → 
  (marley_oranges + marley_apples = 31) :=
by
  intros
  sorry

end marley_total_fruits_l1327_132722


namespace two_digit_number_is_42_l1327_132752

theorem two_digit_number_is_42 (a b : ℕ) (ha : a < 10) (hb : b < 10) (h : 10 * a + b = 42) :
  ((10 * a + b) : ℚ) / (10 * b + a) = 7 / 4 := by
  sorry

end two_digit_number_is_42_l1327_132752


namespace jane_played_8_rounds_l1327_132723

variable (points_per_round : ℕ)
variable (end_points : ℕ)
variable (lost_points : ℕ)
variable (total_points : ℕ)
variable (rounds_played : ℕ)

theorem jane_played_8_rounds 
  (h1 : points_per_round = 10) 
  (h2 : end_points = 60) 
  (h3 : lost_points = 20)
  (h4 : total_points = end_points + lost_points)
  (h5 : total_points = points_per_round * rounds_played) : 
  rounds_played = 8 := 
by 
  sorry

end jane_played_8_rounds_l1327_132723


namespace units_digit_of_153_base_3_l1327_132778

theorem units_digit_of_153_base_3 :
  (153 % 3 ^ 1) = 2 := by
sorry

end units_digit_of_153_base_3_l1327_132778


namespace solve_for_x_l1327_132774

theorem solve_for_x (x : ℚ) (h : (7 * x) / (x - 2) + 4 / (x - 2) = 6 / (x - 2)) : x = 2 / 7 :=
sorry

end solve_for_x_l1327_132774


namespace intersection_sets_l1327_132797

-- Define set A as all x such that x >= -2
def setA : Set ℝ := {x | x >= -2}

-- Define set B as all x such that x < 1
def setB : Set ℝ := {x | x < 1}

-- The statement to prove in Lean 4
theorem intersection_sets : (setA ∩ setB) = {x | -2 <= x ∧ x < 1} :=
by
  sorry

end intersection_sets_l1327_132797


namespace initial_percentage_rise_l1327_132757

-- Definition of the conditions
def final_price_gain (P : ℝ) (x : ℝ) : Prop :=
  P * (1 + x / 100) * 0.9 * 0.85 = P * 1.03275

-- The statement to be proven
theorem initial_percentage_rise (P : ℝ) (x : ℝ) : final_price_gain P x → x = 35.03 :=
by
  sorry -- Proof to be filled in

end initial_percentage_rise_l1327_132757


namespace circle_equation_l1327_132736

-- Define conditions
def on_parabola (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  x^2 = 4 * y

def tangent_to_y_axis (M : ℝ × ℝ) (r : ℝ) : Prop :=
  let (x, _) := M
  abs x = r

def tangent_to_axis_of_symmetry (M : ℝ × ℝ) (r : ℝ) : Prop :=
  let (_, y) := M
  abs (1 + y) = r

-- Main theorem statement
theorem circle_equation (M : ℝ × ℝ) (r : ℝ) (x y : ℝ)
  (h1 : on_parabola M)
  (h2 : tangent_to_y_axis M r)
  (h3 : tangent_to_axis_of_symmetry M r) :
  (x - M.1)^2 + (y - M.2)^2 = r^2 ↔
  x^2 + y^2 + 4 * M.1 * x - 2 * M.2 * y + 1 = 0 := 
sorry

end circle_equation_l1327_132736


namespace Roy_height_l1327_132792

theorem Roy_height (Sara_height Joe_height Roy_height : ℕ) 
  (h1 : Sara_height = 45)
  (h2 : Sara_height = Joe_height + 6)
  (h3 : Joe_height = Roy_height + 3) :
  Roy_height = 36 :=
by
  sorry

end Roy_height_l1327_132792


namespace lottery_profit_l1327_132751

-- Definitions

def Prob_A := (1:ℚ) / 5
def Prob_B := (4:ℚ) / 15
def Prob_C := (1:ℚ) / 5
def Prob_D := (2:ℚ) / 15
def Prob_E := (1:ℚ) / 5

def customers := 300

def first_prize_value := 9
def second_prize_value := 3
def third_prize_value := 1

-- Proof Problem Statement

theorem lottery_profit : 
  (first_prize_category == "D") ∧ 
  (second_prize_category == "B") ∧ 
  (300 * 3 - ((300 * Prob_D) * 9 + (300 * Prob_B) * 3 + (300 * (Prob_A + Prob_C + Prob_E)) * 1)) == 120 :=
by 
  -- Insert mathematical proof here using given probabilities and conditions
  sorry

end lottery_profit_l1327_132751


namespace sale_in_first_month_is_5000_l1327_132719

def sales : List ℕ := [6524, 5689, 7230, 6000, 12557]
def avg_sales : ℕ := 7000
def total_months : ℕ := 6

theorem sale_in_first_month_is_5000 :
  (avg_sales * total_months) - sales.sum = 5000 :=
by sorry

end sale_in_first_month_is_5000_l1327_132719


namespace cone_base_diameter_l1327_132795

theorem cone_base_diameter (l r : ℝ) 
  (h1 : (1/2) * π * l^2 + π * r^2 = 3 * π) 
  (h2 : π * l = 2 * π * r) : 2 * r = 2 :=
by
  sorry

end cone_base_diameter_l1327_132795


namespace length_of_segment_l1327_132739

theorem length_of_segment : ∃ (a b : ℝ), (|a - (16 : ℝ)^(1/5)| = 3) ∧ (|b - (16 : ℝ)^(1/5)| = 3) ∧ abs (a - b) = 6 :=
by
  sorry

end length_of_segment_l1327_132739


namespace discount_percentage_l1327_132735

theorem discount_percentage (original_price sale_price : ℝ) (h₁ : original_price = 128) (h₂ : sale_price = 83.2) :
  (original_price - sale_price) / original_price * 100 = 35 :=
by
  sorry

end discount_percentage_l1327_132735


namespace cooler1_water_left_l1327_132738

noncomputable def waterLeftInFirstCooler (gallons1 gallons2 : ℝ) (chairs rows : ℕ) (ozSmall ozLarge ozPerGallon : ℝ) : ℝ :=
  let totalChairs := chairs * rows
  let totalSmallOunces := totalChairs * ozSmall
  let initialOunces1 := gallons1 * ozPerGallon
  initialOunces1 - totalSmallOunces

theorem cooler1_water_left :
  waterLeftInFirstCooler 4.5 3.25 12 7 4 8 128 = 240 :=
by
  sorry

end cooler1_water_left_l1327_132738


namespace total_telephone_bill_second_month_l1327_132753

theorem total_telephone_bill_second_month
  (F C1 : ℝ) 
  (h1 : F + C1 = 46)
  (h2 : F + 2 * C1 = 76) :
  F + 2 * C1 = 76 :=
by
  sorry

end total_telephone_bill_second_month_l1327_132753


namespace ratio_of_q_to_r_l1327_132747

theorem ratio_of_q_to_r
  (P Q R : ℕ)
  (h1 : R = 400)
  (h2 : P + Q + R = 1210)
  (h3 : 5 * Q = 4 * P) :
  Q * 10 = R * 9 :=
by
  sorry

end ratio_of_q_to_r_l1327_132747


namespace c_geq_one_l1327_132749

open Real

theorem c_geq_one (a b : ℕ) (c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h_eqn : (a + 1) / (b + c) = b / a) : c ≥ 1 :=
by
  sorry

end c_geq_one_l1327_132749


namespace num_ways_to_select_officers_l1327_132787

def ways_to_select_five_officers (n : ℕ) (k : ℕ) : ℕ :=
  (List.range' (n - k + 1) k).foldl (λ acc x => acc * x) 1

theorem num_ways_to_select_officers :
  ways_to_select_five_officers 12 5 = 95040 :=
by
  -- By definition of ways_to_select_five_officers, this is equivalent to 12 * 11 * 10 * 9 * 8.
  sorry

end num_ways_to_select_officers_l1327_132787


namespace f_seven_point_five_l1327_132743

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom f_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem f_seven_point_five : f 7.5 = -0.5 := by
  sorry

end f_seven_point_five_l1327_132743


namespace beach_ball_properties_l1327_132741

theorem beach_ball_properties :
  let d : ℝ := 18
  let r : ℝ := d / 2
  let surface_area : ℝ := 4 * π * r^2
  let volume : ℝ := (4 / 3) * π * r^3
  surface_area = 324 * π ∧ volume = 972 * π :=
by
  sorry

end beach_ball_properties_l1327_132741


namespace smallest_positive_integer_a_l1327_132771

theorem smallest_positive_integer_a (a : ℕ) (hpos : a > 0) :
  (∃ k, 5880 * a = k ^ 2) → a = 15 := 
by
  sorry

end smallest_positive_integer_a_l1327_132771


namespace rebus_system_solution_l1327_132733

theorem rebus_system_solution :
  ∃ (M A H P h : ℕ), 
  (M > 0) ∧ (P > 0) ∧ 
  (M ≠ A) ∧ (M ≠ H) ∧ (M ≠ P) ∧ (M ≠ h) ∧
  (A ≠ H) ∧ (A ≠ P) ∧ (A ≠ h) ∧ 
  (H ≠ P) ∧ (H ≠ h) ∧ (P ≠ h) ∧
  ((M * 10 + A) * (M * 10 + A) = M * 100 + H * 10 + P) ∧ 
  ((A * 10 + M) * (A * 10 + M) = P * 100 + h * 10 + M) ∧ 
  (((M = 1) ∧ (A = 3) ∧ (H = 6) ∧ (P = 9) ∧ (h = 6)) ∨
   ((M = 3) ∧ (A = 1) ∧ (H = 9) ∧ (P = 6) ∧ (h = 9))) :=
by
  sorry

end rebus_system_solution_l1327_132733


namespace simplify_expression_l1327_132761

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := 4 * x^3 + 5 * x^2 + 2 * x + 8 - (3 * x^3 - 7 * x^2 + 4 * x - 6)

-- Define the simplified form
def simplified_expr (x : ℝ) : ℝ := x^3 + 12 * x^2 - 2 * x + 14

-- State the theorem
theorem simplify_expression (x : ℝ) : initial_expr x = simplified_expr x :=
by sorry

end simplify_expression_l1327_132761


namespace max_mondays_in_59_days_l1327_132767

theorem max_mondays_in_59_days (start_day : ℕ) : ∃ d : ℕ, d ≤ 6 ∧ 
  start_day = d → (d = 0 → ∃ m : ℕ, m = 9) :=
by 
  sorry

end max_mondays_in_59_days_l1327_132767


namespace dice_probabilities_relationship_l1327_132765

theorem dice_probabilities_relationship :
  let p1 := 5 / 18
  let p2 := 11 / 18
  let p3 := 1 / 2
  p1 < p3 ∧ p3 < p2
:= by
  sorry

end dice_probabilities_relationship_l1327_132765


namespace expression_at_x_equals_2_l1327_132727

theorem expression_at_x_equals_2 (a b : ℝ) (h : 2 * a - b = -1) : (2 * b - 4 * a) = 2 :=
by {
  sorry
}

end expression_at_x_equals_2_l1327_132727


namespace jane_vases_per_day_l1327_132796

theorem jane_vases_per_day : 
  ∀ (total_vases : ℝ) (days : ℝ), 
  total_vases = 248 → days = 16 → 
  (total_vases / days) = 15.5 :=
by
  intros total_vases days h_total_vases h_days
  rw [h_total_vases, h_days]
  norm_num

end jane_vases_per_day_l1327_132796


namespace least_months_exceed_tripled_borrowed_l1327_132734

theorem least_months_exceed_tripled_borrowed :
  ∃ t : ℕ, (1.03 : ℝ)^t > 3 ∧ ∀ n < t, (1.03 : ℝ)^n ≤ 3 :=
sorry

end least_months_exceed_tripled_borrowed_l1327_132734


namespace vehicle_wax_initial_amount_l1327_132740

theorem vehicle_wax_initial_amount
  (wax_car wax_suv wax_spilled wax_left original_amount : ℕ)
  (h_wax_car : wax_car = 3)
  (h_wax_suv : wax_suv = 4)
  (h_wax_spilled : wax_spilled = 2)
  (h_wax_left : wax_left = 2)
  (h_total_wax_used : wax_car + wax_suv = 7)
  (h_wax_before_waxing : wax_car + wax_suv + wax_spilled = 9) :
  original_amount = 11 := by
  sorry

end vehicle_wax_initial_amount_l1327_132740


namespace xyz_positive_and_distinct_l1327_132710

theorem xyz_positive_and_distinct (a b x y z : ℝ)
  (h₁ : x + y + z = a)
  (h₂ : x^2 + y^2 + z^2 = b^2)
  (h₃ : x * y = z^2)
  (ha_pos : a > 0)
  (hb_condition : b^2 < a^2 ∧ a^2 < 3*b^2) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by
  sorry

end xyz_positive_and_distinct_l1327_132710
