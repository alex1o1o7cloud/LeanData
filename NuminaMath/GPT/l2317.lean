import Mathlib

namespace amount_paid_l2317_231716

def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def change_received : ℕ := 11

theorem amount_paid (h_cost : ℕ := hamburger_cost) (o_cost : ℕ := onion_rings_cost) (s_cost : ℕ := smoothie_cost) (change : ℕ := change_received) :
  h_cost + o_cost + s_cost + change = 20 := by
  sorry

end amount_paid_l2317_231716


namespace upper_limit_of_arun_weight_l2317_231727

variable (w : ℝ)

noncomputable def arun_opinion (w : ℝ) := 62 < w ∧ w < 72
noncomputable def brother_opinion (w : ℝ) := 60 < w ∧ w < 70
noncomputable def average_weight := 64

theorem upper_limit_of_arun_weight 
  (h1 : ∀ w, arun_opinion w → brother_opinion w → 64 = (62 + w) / 2 ) 
  : ∀ w, arun_opinion w ∧ brother_opinion w → w ≤ 66 :=
sorry

end upper_limit_of_arun_weight_l2317_231727


namespace min_value_shift_l2317_231717

noncomputable def f (x : ℝ) (c : ℝ) := x^2 + 4 * x + 5 - c

theorem min_value_shift (c : ℝ) (h : ∀ x : ℝ, f x c ≥ 2) :
  ∀ x : ℝ, f (x - 2009) c ≥ 2 :=
sorry

end min_value_shift_l2317_231717


namespace smallest_solution_correct_l2317_231725

noncomputable def smallest_solution (x : ℝ) : ℝ :=
if (⌊ x^2 ⌋ - ⌊ x ⌋^2 = 17) then x else 0

theorem smallest_solution_correct :
  smallest_solution (7 * Real.sqrt 2) = 7 * Real.sqrt 2 :=
by sorry

end smallest_solution_correct_l2317_231725


namespace problem_a_l2317_231787

theorem problem_a (f : ℕ → ℕ) (h1 : f 1 = 2) (h2 : ∀ n, f (f n) = f n + 3 * n) : f 26 = 59 := 
sorry

end problem_a_l2317_231787


namespace inequality_for_positive_reals_l2317_231798

theorem inequality_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (b * (a + b)) + 1 / (c * (b + c)) + 1 / (a * (c + a)) ≥ 27 / (2 * (a + b + c)^2) :=
by
  sorry

end inequality_for_positive_reals_l2317_231798


namespace sum_of_cubes_of_nonneg_rationals_l2317_231704

theorem sum_of_cubes_of_nonneg_rationals (n : ℤ) (h1 : n > 1) (h2 : ∃ a b : ℚ, a^3 + b^3 = n) :
  ∃ c d : ℚ, c ≥ 0 ∧ d ≥ 0 ∧ c^3 + d^3 = n :=
sorry

end sum_of_cubes_of_nonneg_rationals_l2317_231704


namespace stock_price_end_of_second_year_l2317_231756

noncomputable def initial_price : ℝ := 120
noncomputable def price_after_first_year (initial_price : ℝ) : ℝ := initial_price * 2
noncomputable def price_after_second_year (price_after_first_year : ℝ) : ℝ := price_after_first_year * 0.7

theorem stock_price_end_of_second_year : 
  price_after_second_year (price_after_first_year initial_price) = 168 := 
by 
  sorry

end stock_price_end_of_second_year_l2317_231756


namespace smallest_positive_integer_n_l2317_231793

theorem smallest_positive_integer_n :
  ∃ (n : ℕ), 5 * n ≡ 1978 [MOD 26] ∧ n = 16 :=
by
  sorry

end smallest_positive_integer_n_l2317_231793


namespace total_stamps_l2317_231746

def num_foreign_stamps : ℕ := 90
def num_old_stamps : ℕ := 70
def num_both_foreign_old_stamps : ℕ := 20
def num_neither_stamps : ℕ := 60

theorem total_stamps :
  (num_foreign_stamps + num_old_stamps - num_both_foreign_old_stamps + num_neither_stamps) = 220 :=
  by
    sorry

end total_stamps_l2317_231746


namespace value_of_expression_l2317_231795

-- Define the hypothesis and the goal
theorem value_of_expression (x y : ℝ) (h : 3 * y - x^2 = -5) : 6 * y - 2 * x^2 - 6 = -16 := by
  sorry

end value_of_expression_l2317_231795


namespace find_speed_of_first_train_l2317_231766

variable (L1 L2 : ℝ) (V1 V2 : ℝ) (t : ℝ)

theorem find_speed_of_first_train (hL1 : L1 = 100) (hL2 : L2 = 200) (hV2 : V2 = 30) (ht: t = 14.998800095992321) :
  V1 = 42.005334224 := by
  -- Proof to be completed
  sorry

end find_speed_of_first_train_l2317_231766


namespace fraction_of_donations_l2317_231738

def max_donation_amount : ℝ := 1200
def total_money_raised : ℝ := 3750000
def donations_from_500_people : ℝ := 500 * max_donation_amount
def fraction_of_money_raised : ℝ := 0.4 * total_money_raised
def num_donors : ℝ := 1500

theorem fraction_of_donations (f : ℝ) :
  donations_from_500_people + num_donors * f * max_donation_amount = fraction_of_money_raised → f = 1 / 2 :=
by
  sorry

end fraction_of_donations_l2317_231738


namespace adam_has_23_tattoos_l2317_231731

-- Conditions as definitions
def tattoos_on_each_of_jason_arms := 2
def number_of_jason_arms := 2
def tattoos_on_each_of_jason_legs := 3
def number_of_jason_legs := 2

def jason_total_tattoos : Nat :=
  tattoos_on_each_of_jason_arms * number_of_jason_arms + tattoos_on_each_of_jason_legs * number_of_jason_legs

def adam_tattoos (jason_tattoos : Nat) : Nat :=
  2 * jason_tattoos + 3

-- The main theorem to be proved
theorem adam_has_23_tattoos : adam_tattoos jason_total_tattoos = 23 := by
  sorry

end adam_has_23_tattoos_l2317_231731


namespace cost_of_supplies_l2317_231757

theorem cost_of_supplies (x y z : ℝ) 
  (h1 : 3 * x + 7 * y + z = 3.15) 
  (h2 : 4 * x + 10 * y + z = 4.2) :
  (x + y + z = 1.05) :=
by 
  sorry

end cost_of_supplies_l2317_231757


namespace second_order_arithmetic_sequence_a30_l2317_231702

theorem second_order_arithmetic_sequence_a30 {a : ℕ → ℝ}
  (h₁ : ∀ n, a (n + 1) - a n - (a (n + 2) - a (n + 1)) = 20)
  (h₂ : a 10 = 23)
  (h₃ : a 20 = 23) :
  a 30 = 2023 := 
sorry

end second_order_arithmetic_sequence_a30_l2317_231702


namespace sum_x_coordinates_l2317_231723

-- Define the equations of the line segments
def segment1 (x : ℝ) := 2 * x + 6
def segment2 (x : ℝ) := -0.5 * x - 1.5
def segment3 (x : ℝ) := 2 * x + 1
def segment4 (x : ℝ) := -0.5 * x + 3.5
def segment5 (x : ℝ) := 2 * x - 4

-- Definition of the problem
theorem sum_x_coordinates (h1 : segment1 (-5) = -4 ∧ segment1 (-3) = 0)
    (h2 : segment2 (-3) = 0 ∧ segment2 (-1) = -1)
    (h3 : segment3 (-1) = -1 ∧ segment3 (1) = 3)
    (h4 : segment4 (1) = 3 ∧ segment4 (3) = 2)
    (h5 : segment5 (3) = 2 ∧ segment5 (5) = 6)
    (hx1 : ∃ x1, segment3 x1 = 2.4 ∧ -1 ≤ x1 ∧ x1 ≤ 1)
    (hx2 : ∃ x2, segment4 x2 = 2.4 ∧ 1 ≤ x2 ∧ x2 ≤ 3)
    (hx3 : ∃ x3, segment5 x3 = 2.4 ∧ 3 ≤ x3 ∧ x3 ≤ 5) :
    (∃ (x1 x2 x3 : ℝ), segment3 x1 = 2.4 ∧ segment4 x2 = 2.4 ∧ segment5 x3 = 2.4 ∧ x1 = 0.7 ∧ x2 = 2.2 ∧ x3 = 3.2 ∧ x1 + x2 + x3 = 6.1) :=
sorry

end sum_x_coordinates_l2317_231723


namespace prime_pairs_l2317_231724

open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, 2 ≤ m → m ≤ n / 2 → n % m ≠ 0

theorem prime_pairs :
  ∀ (p q : ℕ), is_prime p → is_prime q →
  1 < p → p < 100 →
  1 < q → q < 100 →
  is_prime (p + 6) →
  is_prime (p + 10) →
  is_prime (q + 4) →
  is_prime (q + 10) →
  is_prime (p + q + 1) →
  (p, q) = (7, 3) ∨ (p, q) = (13, 3) ∨ (p, q) = (37, 3) ∨ (p, q) = (97, 3) :=
by
  sorry

end prime_pairs_l2317_231724


namespace xiaoming_bus_time_l2317_231789

-- Definitions derived from the conditions:
def total_time : ℕ := 40
def transfer_time : ℕ := 6
def subway_time : ℕ := 30
def bus_time : ℕ := 50

-- Theorem statement to prove the bus travel time equals 10 minutes
theorem xiaoming_bus_time : (total_time - transfer_time = 34) ∧ (subway_time = 30 ∧ bus_time = 50) → 
  ∃ (T_bus : ℕ), T_bus = 10 := by
  sorry

end xiaoming_bus_time_l2317_231789


namespace centroid_of_triangle_l2317_231797

theorem centroid_of_triangle :
  let A := (2, 8)
  let B := (6, 2)
  let C := (0, 4)
  let centroid := ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 )
  centroid = (8 / 3, 14 / 3) := 
by
  sorry

end centroid_of_triangle_l2317_231797


namespace ratio_M_N_l2317_231722

theorem ratio_M_N (P Q M N : ℝ) (h1 : M = 0.30 * Q) (h2 : Q = 0.20 * P) (h3 : N = 0.50 * P) (hP_nonzero : P ≠ 0) :
  M / N = 3 / 25 := 
by 
  sorry

end ratio_M_N_l2317_231722


namespace distinct_ordered_pair_count_l2317_231720

theorem distinct_ordered_pair_count (x y : ℕ) (h1 : x + y = 50) (h2 : 1 ≤ x) (h3 : 1 ≤ y) : 
  ∃! (x y : ℕ), x + y = 50 ∧ 1 ≤ x ∧ 1 ≤ y :=
by
  sorry

end distinct_ordered_pair_count_l2317_231720


namespace puzzle_pieces_missing_l2317_231703

/-- Trevor and Joe were working together to finish a 500 piece puzzle. 
They put the border together first and that was 75 pieces. 
Trevor was able to place 105 pieces of the puzzle.
Joe was able to place three times the number of puzzle pieces as Trevor. 
Prove that the number of puzzle pieces missing is 5. -/
theorem puzzle_pieces_missing :
  let total_pieces := 500
  let border_pieces := 75
  let trevor_pieces := 105
  let joe_pieces := 3 * trevor_pieces
  let placed_pieces := trevor_pieces + joe_pieces
  let remaining_pieces := total_pieces - border_pieces
  remaining_pieces - placed_pieces = 5 :=
by
  sorry

end puzzle_pieces_missing_l2317_231703


namespace tangent_y_axis_circle_eq_l2317_231713

theorem tangent_y_axis_circle_eq (h k r : ℝ) (hc : h = -2) (kc : k = 3) (rc : r = abs h) :
  (x + h)^2 + (y - k)^2 = r^2 ↔ (x + 2)^2 + (y - 3)^2 = 4 := by
  sorry

end tangent_y_axis_circle_eq_l2317_231713


namespace gomoku_black_pieces_l2317_231734

/--
Two students, A and B, are preparing to play a game of Gomoku but find that 
the box only contains a certain number of black and white pieces, each of the
same quantity, and the total does not exceed 10. Then, they find 20 more pieces 
(only black and white) and add them to the box. At this point, the ratio of 
the total number of white to black pieces is 7:8. We want to prove that the total number
of black pieces in the box after adding is 16.
-/
theorem gomoku_black_pieces (x y : ℕ) (hx : x = 15 * y - 160) (h_total : x + y ≤ 5)
  (h_ratio : 7 * (x + y) = 8 * (x + (20 - y))) : (x + y = 16) :=
by
  sorry

end gomoku_black_pieces_l2317_231734


namespace willowbrook_team_combinations_l2317_231779

theorem willowbrook_team_combinations :
  let girls := 5
  let boys := 5
  let choose_three (n : ℕ) := n.choose 3
  let team_count := choose_three girls * choose_three boys
  team_count = 100 :=
by
  let girls := 5
  let boys := 5
  let choose_three (n : ℕ) := n.choose 3
  let team_count := choose_three girls * choose_three boys
  have h1 : choose_three girls = 10 := by sorry
  have h2 : choose_three boys = 10 := by sorry
  have h3 : team_count = 10 * 10 := by sorry
  exact h3

end willowbrook_team_combinations_l2317_231779


namespace ratio_of_first_to_fourth_term_l2317_231730

theorem ratio_of_first_to_fourth_term (a d : ℝ) (h1 : (a + d) + (a + 3 * d) = 6 * a) (h2 : a + 2 * d = 10) :
  a / (a + 3 * d) = 1 / 4 :=
by
  sorry

end ratio_of_first_to_fourth_term_l2317_231730


namespace integer_values_of_x_for_equation_l2317_231747

theorem integer_values_of_x_for_equation 
  (a b c : ℤ) (h1 : a ≠ 0) (h2 : a = b + c ∨ b = c + a ∨ c = b + a) : 
  ∃ x : ℤ, a * x + b = c :=
sorry

end integer_values_of_x_for_equation_l2317_231747


namespace consecutive_natural_numbers_sum_l2317_231770

theorem consecutive_natural_numbers_sum :
  (∃ (n : ℕ), 0 < n → n ≤ 4 ∧ (n-1) + n + (n+1) ≤ 12) → 
  (∃ n_sets : ℕ, n_sets = 4) :=
by
  sorry

end consecutive_natural_numbers_sum_l2317_231770


namespace number_of_boys_is_810_l2317_231754

theorem number_of_boys_is_810 (B G : ℕ) (h1 : B + G = 900) (h2 : G = B / 900 * 100) : B = 810 :=
by
  sorry

end number_of_boys_is_810_l2317_231754


namespace mixture_weight_l2317_231719

theorem mixture_weight :
  let weight_a_per_liter := 900 -- in gm
  let weight_b_per_liter := 750 -- in gm
  let ratio_a := 3
  let ratio_b := 2
  let total_volume := 4 -- in liters
  let volume_a := (ratio_a / (ratio_a + ratio_b)) * total_volume
  let volume_b := (ratio_b / (ratio_a + ratio_b)) * total_volume
  let weight_a := volume_a * weight_a_per_liter
  let weight_b := volume_b * weight_b_per_liter
  let total_weight_gm := weight_a + weight_b
  let total_weight_kg := total_weight_gm / 1000 
  total_weight_kg = 3.36 :=
by
  sorry

end mixture_weight_l2317_231719


namespace transform_f_to_g_l2317_231776

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem transform_f_to_g :
  ∀ x : ℝ, g x = f (x + (π / 8)) :=
by
  sorry

end transform_f_to_g_l2317_231776


namespace no_common_points_eq_l2317_231749

theorem no_common_points_eq (a : ℝ) : 
  ((∀ x y : ℝ, y = (a^2 - a) * x + 1 - a → y ≠ 2 * x - 1) ↔ (a = -1)) :=
by
  sorry

end no_common_points_eq_l2317_231749


namespace geometric_sequence_third_term_l2317_231796

theorem geometric_sequence_third_term 
  (a r : ℝ)
  (h1 : a = 3)
  (h2 : a * r^4 = 243) : 
  a * r^2 = 27 :=
by
  sorry

end geometric_sequence_third_term_l2317_231796


namespace algebraic_expression_value_l2317_231781

theorem algebraic_expression_value (m : ℝ) (h : m^2 + 2*m - 1 = 0) : 2*m^2 + 4*m + 2021 = 2023 := 
sorry

end algebraic_expression_value_l2317_231781


namespace inclination_angle_tan_60_perpendicular_l2317_231737

/-
The inclination angle of the line given by x = tan(60 degrees) is 90 degrees.
-/
theorem inclination_angle_tan_60_perpendicular : 
  ∀ (x : ℝ), x = Real.tan (60 *Real.pi / 180) → 
  ∃ θ : ℝ, θ = 90 :=
sorry

end inclination_angle_tan_60_perpendicular_l2317_231737


namespace john_monthly_income_l2317_231755

theorem john_monthly_income (I : ℝ) (h : I - 0.05 * I = 1900) : I = 2000 :=
by
  sorry

end john_monthly_income_l2317_231755


namespace tan_ratio_l2317_231788

theorem tan_ratio (α β : ℝ) 
  (h1 : Real.sin (α + β) = (Real.sqrt 3) / 2) 
  (h2 : Real.sin (α - β) = (Real.sqrt 2) / 2) : 
  (Real.tan α) / (Real.tan β) = (5 + 2 * Real.sqrt 6) / (5 - 2 * Real.sqrt 6) :=
by
  sorry

end tan_ratio_l2317_231788


namespace find_point_P_l2317_231785

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨2, 2⟩
def N : Point := ⟨5, -2⟩

def is_on_x_axis (P : Point) : Prop :=
  P.y = 0

def is_right_angle (M N P : Point) : Prop :=
  (M.x - P.x)*(N.x - P.x) + (M.y - P.y)*(N.y - P.y) = 0

noncomputable def P1 : Point := ⟨1, 0⟩
noncomputable def P2 : Point := ⟨6, 0⟩

theorem find_point_P :
  ∃ P : Point, is_on_x_axis P ∧ is_right_angle M N P ∧ (P = P1 ∨ P = P2) :=
by
  sorry

end find_point_P_l2317_231785


namespace restaurant_total_spent_l2317_231771

theorem restaurant_total_spent (appetizer_cost : ℕ) (entree_cost : ℕ) (num_entrees : ℕ) (tip_rate : ℚ) 
  (H1 : appetizer_cost = 10) (H2 : entree_cost = 20) (H3 : num_entrees = 4) (H4 : tip_rate = 0.20) :
  appetizer_cost + num_entrees * entree_cost + (appetizer_cost + num_entrees * entree_cost) * tip_rate = 108 :=
by
  sorry

end restaurant_total_spent_l2317_231771


namespace find_unit_prices_l2317_231711

variable (x : ℝ)

def typeB_unit_price (priceB : ℝ) : Prop :=
  priceB = 15

def typeA_unit_price (priceA : ℝ) : Prop :=
  priceA = 40

def budget_condition : Prop :=
  900 / x = 3 * (800 / (x + 25))

theorem find_unit_prices (h : budget_condition x) :
  typeB_unit_price x ∧ typeA_unit_price (x + 25) :=
sorry

end find_unit_prices_l2317_231711


namespace sequence_recurrence_l2317_231765

theorem sequence_recurrence (v : ℕ → ℝ) (h_rec : ∀ n, v (n + 2) = 3 * v (n + 1) + 2 * v n) 
    (h_v3 : v 3 = 8) (h_v6 : v 6 = 245) : v 5 = 70 :=
sorry

end sequence_recurrence_l2317_231765


namespace part1_part2_l2317_231700

variable (m : ℝ)

def p (m : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x - 2 ≥ m^2 - 3 * m
def q (m : ℝ) : Prop := ∃ x0 : ℝ, -1 ≤ x0 ∧ x0 ≤ 1 ∧ m ≤ x0

theorem part1 (h : p m) : 1 ≤ m ∧ m ≤ 2 := sorry

theorem part2 (h : ¬(p m ∧ q m) ∧ (p m ∨ q m)) : (m < 1) ∨ (1 < m ∧ m ≤ 2) := sorry

end part1_part2_l2317_231700


namespace carla_initial_marbles_l2317_231748

theorem carla_initial_marbles
  (marbles_bought : ℕ)
  (total_marbles_now : ℕ)
  (h1 : marbles_bought = 134)
  (h2 : total_marbles_now = 187) :
  total_marbles_now - marbles_bought = 53 :=
by
  sorry

end carla_initial_marbles_l2317_231748


namespace product_of_prs_l2317_231760

theorem product_of_prs 
  (p r s : Nat) 
  (h1 : 3^p + 3^5 = 270) 
  (h2 : 2^r + 58 = 122) 
  (h3 : 7^2 + 5^s = 2504) : 
  p * r * s = 54 := 
sorry

end product_of_prs_l2317_231760


namespace non_integer_interior_angle_count_l2317_231715

theorem non_integer_interior_angle_count :
  ∃! (n : ℕ), 3 ≤ n ∧ n < 10 ∧ ¬(∃ k : ℕ, 180 * (n - 2) = n * k) :=
by sorry

end non_integer_interior_angle_count_l2317_231715


namespace value_of_p_l2317_231794

theorem value_of_p (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 = 3 * x2 ∧ x^2 - (3 * p - 2) * x + p^2 - 1 = 0) →
  (p = 2 ∨ p = 14 / 11) :=
by
  sorry

end value_of_p_l2317_231794


namespace convex_polygon_sides_ne_14_l2317_231762

noncomputable def side_length : ℝ := 1

def is_triangle (s : ℝ) : Prop :=
  s = side_length

def is_dodecagon (s : ℝ) : Prop :=
  s = side_length

def side_coincide (t : ℝ) (d : ℝ) : Prop :=
  is_triangle t ∧ is_dodecagon d ∧ t = d

def valid_resulting_sides (s : ℤ) : Prop :=
  s = 11 ∨ s = 12 ∨ s = 13

theorem convex_polygon_sides_ne_14 : ∀ t d, side_coincide t d → ¬ valid_resulting_sides 14 := 
by
  intro t d h
  sorry

end convex_polygon_sides_ne_14_l2317_231762


namespace ratio_of_linear_combination_l2317_231742

theorem ratio_of_linear_combination (a b x y : ℝ) (hb : b ≠ 0) 
  (h1 : 4 * x - 2 * y = a) (h2 : 5 * y - 10 * x = b) :
  a / b = -2 / 5 :=
by {
  sorry
}

end ratio_of_linear_combination_l2317_231742


namespace average_rate_of_change_l2317_231701

noncomputable def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change :
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_l2317_231701


namespace range_of_b_if_solution_set_contains_1_2_3_l2317_231706

theorem range_of_b_if_solution_set_contains_1_2_3 
  (b : ℝ)
  (h : ∀ x : ℝ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) :
  5 < b ∧ b < 7 :=
sorry

end range_of_b_if_solution_set_contains_1_2_3_l2317_231706


namespace car_mileage_l2317_231718

/-- If a car needs 3.5 gallons of gasoline to travel 140 kilometers, it gets 40 kilometers per gallon. -/
theorem car_mileage (gallons_used : ℝ) (distance_traveled : ℝ) 
  (h : gallons_used = 3.5 ∧ distance_traveled = 140) : 
  distance_traveled / gallons_used = 40 :=
by
  sorry

end car_mileage_l2317_231718


namespace minimum_k_exists_l2317_231733

theorem minimum_k_exists (k : ℕ) (h : k > 0) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ a + c > b ∧ b + c > a) ↔ k = 6 :=
sorry

end minimum_k_exists_l2317_231733


namespace f_val_at_100_l2317_231782

theorem f_val_at_100 (f : ℝ → ℝ) (h₀ : ∀ x, f x * f (x + 3) = 12) (h₁ : f 1 = 4) : f 100 = 3 :=
sorry

end f_val_at_100_l2317_231782


namespace coordinates_of_B_l2317_231759

theorem coordinates_of_B (A B : ℝ × ℝ) (h1 : A = (-2, 3)) (h2 : (A.1 = B.1 ∨ A.1 + 1 = B.1 ∨ A.1 - 1 = B.1)) (h3 : A.2 = B.2) : 
  B = (-1, 3) ∨ B = (-3, 3) := 
sorry

end coordinates_of_B_l2317_231759


namespace unique_zero_a_neg_l2317_231769

noncomputable def f (a x : ℝ) : ℝ := 3 * Real.exp (abs (x - 1)) - a * (2^(x - 1) + 2^(1 - x)) - a^2

theorem unique_zero_a_neg (a : ℝ) (h_unique : ∃! x : ℝ, f a x = 0) (h_neg : a < 0) : a = -3 := 
sorry

end unique_zero_a_neg_l2317_231769


namespace det_E_eq_25_l2317_231739

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 0], ![0, 5]]

theorem det_E_eq_25 : E.det = 25 := by
  sorry

end det_E_eq_25_l2317_231739


namespace inequality_proof_l2317_231740

open Real

noncomputable def f (t x : ℝ) : ℝ := t * x - (t - 1) * log x - t

theorem inequality_proof (t x : ℝ) (h_t : t ≤ 0) (h_x : x > 1) : 
  f t x < exp (x - 1) - 1 :=
sorry

end inequality_proof_l2317_231740


namespace value_of_box_l2317_231761

theorem value_of_box (a b c : ℕ) (h1 : a + b = c) (h2 : a + b + c = 100) : c = 50 :=
sorry

end value_of_box_l2317_231761


namespace combined_distance_l2317_231792

noncomputable def radius_wheel1 : ℝ := 22.4
noncomputable def revolutions_wheel1 : ℕ := 750

noncomputable def radius_wheel2 : ℝ := 15.8
noncomputable def revolutions_wheel2 : ℕ := 950

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def distance_covered (r : ℝ) (rev : ℕ) : ℝ := circumference r * rev

theorem combined_distance :
  distance_covered radius_wheel1 revolutions_wheel1 + distance_covered radius_wheel2 revolutions_wheel2 = 199896.96 := by
  sorry

end combined_distance_l2317_231792


namespace average_net_sales_per_month_l2317_231777

def sales_jan : ℕ := 120
def sales_feb : ℕ := 80
def sales_mar : ℕ := 50
def sales_apr : ℕ := 130
def sales_may : ℕ := 90
def sales_jun : ℕ := 160

def monthly_expense : ℕ := 30
def num_months : ℕ := 6

def total_sales := sales_jan + sales_feb + sales_mar + sales_apr + sales_may + sales_jun
def total_expenses := monthly_expense * num_months
def net_total_sales := total_sales - total_expenses

theorem average_net_sales_per_month : net_total_sales / num_months = 75 :=
by {
  -- Lean code for proof here
  sorry
}

end average_net_sales_per_month_l2317_231777


namespace weight_of_daughter_l2317_231768

theorem weight_of_daughter 
  (M D C : ℝ)
  (h1 : M + D + C = 120)
  (h2 : D + C = 60)
  (h3 : C = (1 / 5) * M)
  : D = 48 :=
by
  sorry

end weight_of_daughter_l2317_231768


namespace hyperbola_eccentricity_is_2_l2317_231741

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let c := 4 * a
  let e := c / a
  e

theorem hyperbola_eccentricity_is_2
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity a b ha hb = 2 := 
sorry

end hyperbola_eccentricity_is_2_l2317_231741


namespace expected_value_is_10_l2317_231791

noncomputable def expected_value_adjacent_pairs (boys girls : ℕ) (total_people : ℕ) : ℕ :=
  if total_people = 20 ∧ boys = 8 ∧ girls = 12 then 10 else sorry

theorem expected_value_is_10 : expected_value_adjacent_pairs 8 12 20 = 10 :=
by
  -- Intuition and all necessary calculations (proof steps) have already been explained.
  -- Here we are directly stating the conclusion based on given problem conditions.
  trivial

end expected_value_is_10_l2317_231791


namespace similar_triangle_perimeter_l2317_231778

theorem similar_triangle_perimeter 
  (a b c : ℝ) (ha : a = 12) (hb : b = 12) (hc : c = 24) 
  (k : ℝ) (hk : k = 1.5) : 
  (1.5 * a) + (1.5 * b) + (1.5 * c) = 72 :=
by
  sorry

end similar_triangle_perimeter_l2317_231778


namespace no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime_l2317_231714

theorem no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 10000 ∧ (n % 10 = 0) ∧ (Prime n) → False :=
by
  sorry

end no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime_l2317_231714


namespace find_a_l2317_231732
open Real

theorem find_a (a : ℝ) (k : ℤ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (x1^2 + y1^2 = 10 * (x1 * cos a + y1 * sin a) ∧
     x2^2 + y2^2 = 10 * (x2 * sin (3 * a) + y2 * cos (3 * a)) ∧
     (x2 - x1)^2 + (y2 - y1)^2 = 64)) ↔
  (∃ k : ℤ, a = π / 8 + k * π / 2) :=
sorry

end find_a_l2317_231732


namespace leap_years_count_l2317_231743

def is_leap_year (y : ℕ) : Bool :=
  if y % 800 = 300 ∨ y % 800 = 600 then true else false

theorem leap_years_count : 
  { y : ℕ // 1500 ≤ y ∧ y ≤ 3500 ∧ y % 100 = 0 ∧ is_leap_year y } = {y | y = 1900 ∨ y = 2200 ∨ y = 2700 ∨ y = 3000 ∨ y = 3500} :=
by
  sorry

end leap_years_count_l2317_231743


namespace term_omit_perfect_squares_300_l2317_231764

theorem term_omit_perfect_squares_300 (n : ℕ) (hn : n = 300) : 
  ∃ k : ℕ, k = 317 ∧ (∀ m : ℕ, (m < k → m * m ≠ k)) := 
sorry

end term_omit_perfect_squares_300_l2317_231764


namespace bus_speed_including_stoppages_l2317_231729

theorem bus_speed_including_stoppages
  (speed_excluding_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ) :
  speed_excluding_stoppages = 64 ∧ stoppage_time_per_hour = 15 / 60 →
  (44 / 60) * speed_excluding_stoppages = 48 :=
by
  sorry

end bus_speed_including_stoppages_l2317_231729


namespace fraction_addition_simplification_l2317_231735

theorem fraction_addition_simplification :
  (2 / 5 : ℚ) + (3 / 15) = 3 / 5 :=
by
  sorry

end fraction_addition_simplification_l2317_231735


namespace initial_riding_time_l2317_231784

theorem initial_riding_time (t : ℝ) (h1 : t * 60 + 90 + 30 + 120 = 270) : t * 60 = 30 :=
by sorry

end initial_riding_time_l2317_231784


namespace sqrt_1_0201_eq_1_01_l2317_231710

theorem sqrt_1_0201_eq_1_01 (h : Real.sqrt 102.01 = 10.1) : Real.sqrt 1.0201 = 1.01 :=
by 
  sorry

end sqrt_1_0201_eq_1_01_l2317_231710


namespace Travis_annual_cereal_cost_l2317_231705

def cost_of_box_A : ℚ := 2.50
def cost_of_box_B : ℚ := 3.50
def cost_of_box_C : ℚ := 4.00
def cost_of_box_D : ℚ := 5.25
def cost_of_box_E : ℚ := 6.00

def quantity_of_box_A : ℚ := 1
def quantity_of_box_B : ℚ := 0.5
def quantity_of_box_C : ℚ := 0.25
def quantity_of_box_D : ℚ := 0.75
def quantity_of_box_E : ℚ := 1.5

def cost_week1 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  cost_of_box_B * quantity_of_box_B +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  cost_of_box_E * quantity_of_box_E

def cost_week2 : ℚ :=
  let subtotal := 
    cost_of_box_A * quantity_of_box_A +
    cost_of_box_B * quantity_of_box_B +
    cost_of_box_C * quantity_of_box_C +
    cost_of_box_D * quantity_of_box_D +
    cost_of_box_E * quantity_of_box_E
  subtotal * 0.8

def cost_week3 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  0 +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  cost_of_box_E * quantity_of_box_E

def cost_week4 : ℚ :=
  cost_of_box_A * quantity_of_box_A +
  cost_of_box_B * quantity_of_box_B +
  cost_of_box_C * quantity_of_box_C +
  cost_of_box_D * quantity_of_box_D +
  let discounted_box_E := cost_of_box_E * quantity_of_box_E * 0.85
  cost_of_box_A * quantity_of_box_A +
  discounted_box_E
  
def monthly_cost : ℚ :=
  cost_week1 + cost_week2 + cost_week3 + cost_week4

def annual_cost : ℚ :=
  monthly_cost * 12

theorem Travis_annual_cereal_cost :
  annual_cost = 792.24 := by
  sorry

end Travis_annual_cereal_cost_l2317_231705


namespace false_statement_d_l2317_231799

-- Define lines and planes
variables (l m : Type*) (α β : Type*)

-- Define parallel relation
def parallel (l m : Type*) : Prop := sorry

-- Define subset relation
def in_plane (l : Type*) (α : Type*) : Prop := sorry

-- Define the given conditions
axiom l_parallel_alpha : parallel l α
axiom m_in_alpha : in_plane m α

-- Main theorem statement: prove \( l \parallel m \) is false given the conditions.
theorem false_statement_d : ¬ parallel l m :=
sorry

end false_statement_d_l2317_231799


namespace solve_eq1_solve_eq2_l2317_231750

theorem solve_eq1 (x : ℝ) : 3 * (x - 2) ^ 2 = 27 ↔ (x = 5 ∨ x = -1) :=
by
  sorry

theorem solve_eq2 (x : ℝ) : (x + 5) ^ 3 + 27 = 0 ↔ x = -8 :=
by
  sorry

end solve_eq1_solve_eq2_l2317_231750


namespace car_passing_time_l2317_231752

open Real

theorem car_passing_time
  (vX : ℝ) (lX : ℝ)
  (vY : ℝ) (lY : ℝ)
  (t : ℝ)
  (h_vX : vX = 90)
  (h_lX : lX = 5)
  (h_vY : vY = 91)
  (h_lY : lY = 6)
  :
  (t * (vY - vX) / 3600) = 0.011 → t = 39.6 := 
by
  sorry

end car_passing_time_l2317_231752


namespace number_of_tetrises_l2317_231745

theorem number_of_tetrises 
  (points_per_single : ℕ := 1000)
  (points_per_tetris : ℕ := 8 * points_per_single)
  (singles_scored : ℕ := 6)
  (total_score : ℕ := 38000) :
  (total_score - (singles_scored * points_per_single)) / points_per_tetris = 4 := 
by 
  sorry

end number_of_tetrises_l2317_231745


namespace sqrt_7_irrational_l2317_231726

theorem sqrt_7_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a: ℝ) / b = Real.sqrt 7 := by
  sorry

end sqrt_7_irrational_l2317_231726


namespace large_beaker_multiple_small_beaker_l2317_231786

variables (S L : ℝ) (k : ℝ)

theorem large_beaker_multiple_small_beaker 
  (h1 : Small_beaker = S)
  (h2 : Large_beaker = k * S)
  (h3 : Salt_water_in_small = S/2)
  (h4 : Fresh_water_in_large = (Large_beaker) / 5)
  (h5 : (Salt_water_in_small + Fresh_water_in_large = 0.3 * (Large_beaker))) :
  k = 5 :=
sorry

end large_beaker_multiple_small_beaker_l2317_231786


namespace halfway_between_one_eighth_and_one_third_is_correct_l2317_231736

-- Define the fractions
def one_eighth : ℚ := 1 / 8
def one_third : ℚ := 1 / 3

-- Define the correct answer
def correct_answer : ℚ := 11 / 48

-- State the theorem to prove the halfway number is correct_answer
theorem halfway_between_one_eighth_and_one_third_is_correct : 
  (one_eighth + one_third) / 2 = correct_answer :=
sorry

end halfway_between_one_eighth_and_one_third_is_correct_l2317_231736


namespace sequence_properties_l2317_231774

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_seq (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given conditions
variables (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ)
  (h_arith : arithmetic_seq a 2)
  (h_sum_prop : sum_seq a S)
  (h_ratio : ∀ n, S (2 * n) / S n = 4)
  (b : ℕ → ℤ) (T : ℕ → ℤ)
  (h_b : ∀ n, b n = a n * 2 ^ (n - 1))

-- Prove the sequences
theorem sequence_properties :
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, S n = n^2) ∧
  (∀ n, T n = (2 * n - 3) * 2^n + 3) :=
by
  sorry

end sequence_properties_l2317_231774


namespace hexadecagon_area_l2317_231728

theorem hexadecagon_area (r : ℝ) : 
  let θ := (360 / 16 : ℝ)
  let A_triangle := (1 / 2) * r^2 * Real.sin (θ * Real.pi / 180)
  let total_area := 16 * A_triangle
  3 * r^2 = total_area :=
by
  sorry

end hexadecagon_area_l2317_231728


namespace probability_of_at_most_3_heads_l2317_231709

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l2317_231709


namespace no_integer_solutions_l2317_231783

theorem no_integer_solutions (a : ℕ) (h : a % 4 = 3) : ¬∃ (x y : ℤ), x^2 + y^2 = a := by
  sorry

end no_integer_solutions_l2317_231783


namespace distance_between_foci_l2317_231744

-- Given problem
def hyperbola_eq (x y : ℝ) : Prop := 9 * x^2 - 18 * x - 16 * y^2 + 32 * y = 144

theorem distance_between_foci :
  ∀ (x y : ℝ),
    hyperbola_eq x y →
    2 * Real.sqrt ((137 / 9) + (137 / 16)) / 72 = 38 * Real.sqrt 7 / 72 :=
by
  intros x y h
  sorry

end distance_between_foci_l2317_231744


namespace cos2_add_2sin2_eq_64_over_25_l2317_231707

theorem cos2_add_2sin2_eq_64_over_25 (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
sorry

end cos2_add_2sin2_eq_64_over_25_l2317_231707


namespace min_value_quadratic_l2317_231753

noncomputable def quadratic_expr (x : ℝ) : ℝ :=
  x^2 - 4 * x - 2019

theorem min_value_quadratic :
  ∀ x : ℝ, quadratic_expr x ≥ -2023 :=
by
  sorry

end min_value_quadratic_l2317_231753


namespace max_value_log_function_l2317_231790

theorem max_value_log_function (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1/2) :
  ∃ u : ℝ, (u = Real.logb (1/2) (8*x*y + 4*y^2 + 1)) ∧ (u ≤ 0) :=
sorry

end max_value_log_function_l2317_231790


namespace current_speed_l2317_231767

-- Define the constants based on conditions
def rowing_speed_kmph : Float := 24
def distance_meters : Float := 40
def time_seconds : Float := 4.499640028797696

-- Intermediate calculation: Convert rowing speed from km/h to m/s
def rowing_speed_mps : Float := rowing_speed_kmph * 1000 / 3600

-- Calculate downstream speed
def downstream_speed_mps : Float := distance_meters / time_seconds

-- Define the expected speed of the current
def expected_current_speed : Float := 2.22311111

-- The theorem to prove
theorem current_speed : 
  (downstream_speed_mps - rowing_speed_mps) = expected_current_speed :=
by 
  -- skipping the proof steps, as instructed
  sorry

end current_speed_l2317_231767


namespace shorter_piece_length_l2317_231772

theorem shorter_piece_length (P : ℝ) (Q : ℝ) (h1 : P + Q = 68) (h2 : Q = P + 12) : P = 28 := 
by
  sorry

end shorter_piece_length_l2317_231772


namespace geometric_sequence_value_of_b_l2317_231773

theorem geometric_sequence_value_of_b :
  ∀ (a b c : ℝ), 
  (∃ q : ℝ, q ≠ 0 ∧ a = 1 * q ∧ b = 1 * q^2 ∧ c = 1 * q^3 ∧ 4 = 1 * q^4) → 
  b = 2 :=
by
  intro a b c
  intro h
  obtain ⟨q, hq0, ha, hb, hc, hd⟩ := h
  sorry

end geometric_sequence_value_of_b_l2317_231773


namespace total_distance_apart_l2317_231712

def Jay_rate : ℕ := 1 / 15 -- Jay walks 1 mile every 15 minutes
def Paul_rate : ℕ := 3 / 30 -- Paul walks 3 miles every 30 minutes
def time_in_minutes : ℕ := 120 -- 2 hours converted to minutes

def Jay_distance (rate time : ℕ) : ℕ := rate * time / 15
def Paul_distance (rate time : ℕ) : ℕ := rate * time / 30

theorem total_distance_apart : 
  Jay_distance Jay_rate time_in_minutes + Paul_distance Paul_rate time_in_minutes = 20 :=
  by
  -- Proof here
  sorry

end total_distance_apart_l2317_231712


namespace bowling_tournament_l2317_231775

def num_possible_orders : ℕ := 32

theorem bowling_tournament : num_possible_orders = 2 * 2 * 2 * 2 * 2 := by
  -- The structure of the playoff with 2 choices per match until all matches are played,
  -- leading to a total of 5 rounds and 2 choices per round, hence 2^5 = 32.
  sorry

end bowling_tournament_l2317_231775


namespace find_x_l2317_231758

theorem find_x
  (x : ℕ)
  (h1 : x % 7 = 0)
  (h2 : x > 0)
  (h3 : x^2 > 144)
  (h4 : x < 25) : x = 14 := 
  sorry

end find_x_l2317_231758


namespace alternate_seating_boys_l2317_231763

theorem alternate_seating_boys (B : ℕ) (girl : ℕ) (ways : ℕ)
  (h1 : girl = 1)
  (h2 : ways = 24)
  (h3 : ways = B - 1) :
  B = 25 :=
sorry

end alternate_seating_boys_l2317_231763


namespace intersection_of_A_and_B_l2317_231751

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := sorry

end intersection_of_A_and_B_l2317_231751


namespace monotonic_increasing_quadratic_l2317_231721

theorem monotonic_increasing_quadratic (b : ℝ) (c : ℝ) :
  (∀ x y : ℝ, (0 ≤ x → x ≤ y → (x^2 + b*x + c) ≤ (y^2 + b*y + c))) ↔ (b ≥ 0) :=
sorry  -- Proof is omitted

end monotonic_increasing_quadratic_l2317_231721


namespace ethanol_percentage_in_fuel_A_l2317_231780

noncomputable def percent_ethanol_in_fuel_A : ℝ := 0.12

theorem ethanol_percentage_in_fuel_A
  (fuel_tank_capacity : ℝ)
  (fuel_A_volume : ℝ)
  (fuel_B_volume : ℝ)
  (fuel_B_ethanol_percent : ℝ)
  (total_ethanol : ℝ) :
  fuel_tank_capacity = 218 → 
  fuel_A_volume = 122 → 
  fuel_B_volume = 96 → 
  fuel_B_ethanol_percent = 0.16 → 
  total_ethanol = 30 → 
  (fuel_A_volume * percent_ethanol_in_fuel_A) + (fuel_B_volume * fuel_B_ethanol_percent) = total_ethanol :=
by
  sorry

end ethanol_percentage_in_fuel_A_l2317_231780


namespace min_value_geometric_sequence_l2317_231708

theorem min_value_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h : 0 < q ∧ 0 < a 0) 
  (H : 2 * a 3 + a 2 - 2 * a 1 - a 0 = 8) 
  (h_geom : ∀ n, a (n+1) = a n * q) : 
  2 * a 4 + a 3 = 12 * Real.sqrt 3 :=
sorry

end min_value_geometric_sequence_l2317_231708
