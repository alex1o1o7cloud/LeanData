import Mathlib

namespace solve_y_l1955_195523

theorem solve_y :
  ∀ y : ℚ, 6 * (4 * y - 1) - 3 = 3 * (2 - 5 * y) ↔ y = 5 / 13 :=
by
  sorry

end solve_y_l1955_195523


namespace rest_duration_per_kilometer_l1955_195517

theorem rest_duration_per_kilometer
  (speed : ℕ)
  (total_distance : ℕ)
  (total_time : ℕ)
  (walking_time : ℕ := total_distance / speed * 60)  -- walking_time in minutes
  (rest_time : ℕ := total_time - walking_time)  -- total resting time in minutes
  (number_of_rests : ℕ := total_distance - 1)  -- number of rests after each kilometer
  (duration_per_rest : ℕ := rest_time / number_of_rests)
  (h1 : speed = 10)
  (h2 : total_distance = 5)
  (h3 : total_time = 50) : 
  (duration_per_rest = 5) := 
sorry

end rest_duration_per_kilometer_l1955_195517


namespace find_larger_number_l1955_195532

variable (x y : ℕ)

theorem find_larger_number (h1 : x = 7) (h2 : x + y = 15) : y = 8 := by
  sorry

end find_larger_number_l1955_195532


namespace function_satisfy_f1_function_satisfy_f2_l1955_195560

noncomputable def f1 (x : ℝ) : ℝ := 2
noncomputable def f2 (x : ℝ) : ℝ := x

theorem function_satisfy_f1 : 
  ∀ x y : ℝ, x > 0 → y > 0 → f1 (x + y) + f1 x * f1 y = f1 (x * y) + f1 x + f1 y :=
by 
  intros x y hx hy
  unfold f1
  sorry

theorem function_satisfy_f2 :
  ∀ x y : ℝ, x > 0 → y > 0 → f2 (x + y) + f2 x * f2 y = f2 (x * y) + f2 x + f2 y :=
by 
  intros x y hx hy
  unfold f2
  sorry

end function_satisfy_f1_function_satisfy_f2_l1955_195560


namespace find_AX_length_l1955_195596

theorem find_AX_length (t BC AC BX : ℝ) (AX AB : ℝ)
  (h1 : t = 0.75)
  (h2 : AX = t * AB)
  (h3 : BC = 40)
  (h4 : AC = 35)
  (h5 : BX = 15) :
  AX = 105 / 8 := 
  sorry

end find_AX_length_l1955_195596


namespace extremum_f_range_a_for_no_zeros_l1955_195501

noncomputable def f (a b x : ℝ) : ℝ :=
  (a * (x - 1) + b * Real.exp x) / Real.exp x

theorem extremum_f (a b : ℝ) (h_a_ne_zero : a ≠ 0) :
  (∃ (x : ℝ), a = -1 ∧ b = 0 ∧ f a b x = -1 / Real.exp 2) := sorry

theorem range_a_for_no_zeros (a : ℝ) :
  (∀ x : ℝ, a * x - a + Real.exp x ≠ 0) ↔ (-Real.exp 2 < a ∧ a < 0) := sorry

end extremum_f_range_a_for_no_zeros_l1955_195501


namespace higher_amount_is_sixty_l1955_195553

theorem higher_amount_is_sixty (R : ℕ) (n : ℕ) (H : ℝ) 
  (h1 : 2000 = 40 * n + H * R)
  (h2 : 1800 = 40 * (n + 10) + H * (R - 10)) :
  H = 60 :=
by
  sorry

end higher_amount_is_sixty_l1955_195553


namespace set_range_of_three_numbers_l1955_195529

theorem set_range_of_three_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : (a + b + c) / 3 = 6) 
(h4 : b = 6) (h5 : c = 10) : c - a = 8 := by
  sorry

end set_range_of_three_numbers_l1955_195529


namespace find_x_l1955_195564

theorem find_x
  (a b c d k : ℝ)
  (h1 : a ≠ b)
  (h2 : b ≠ 0)
  (h3 : d ≠ 0)
  (h4 : k ≠ 0)
  (h5 : k ≠ 1)
  (h_frac_change : (a + k * x) / (b + x) = c / d) :
  x = (b * c - a * d) / (k * d - c) := by
  sorry

end find_x_l1955_195564


namespace max_ratio_of_right_triangle_l1955_195557

theorem max_ratio_of_right_triangle (a b c: ℝ) (h1: (1/2) * a * b = 30) (h2: a^2 + b^2 = c^2) : 
  (∀ x y z, (1/2 * x * y = 30) → (x^2 + y^2 = z^2) → 
  (x + y + z) / 30 ≤ (7.75 + 7.75 + 10.95) / 30) :=
by 
  sorry  -- The proof will show the maximum value is approximately 0.8817.

noncomputable def max_value := (7.75 + 7.75 + 10.95) / 30

end max_ratio_of_right_triangle_l1955_195557


namespace gas_fee_calculation_l1955_195536

theorem gas_fee_calculation (x : ℚ) (h_usage : x > 60) :
  60 * 0.8 + (x - 60) * 1.2 = 0.88 * x → x * 0.88 = 66 := by
  sorry

end gas_fee_calculation_l1955_195536


namespace working_mom_work_percentage_l1955_195599

theorem working_mom_work_percentage :
  let total_hours_in_day := 24
  let work_hours := 8
  let gym_hours := 2
  let cooking_hours := 1.5
  let bath_hours := 0.5
  let homework_hours := 1
  let packing_hours := 0.5
  let cleaning_hours := 0.5
  let leisure_hours := 2
  let total_activity_hours := work_hours + gym_hours + cooking_hours + bath_hours + homework_hours + packing_hours + cleaning_hours + leisure_hours
  16 = total_activity_hours →
  (work_hours / total_hours_in_day) * 100 = 33.33 :=
by
  sorry

end working_mom_work_percentage_l1955_195599


namespace distance_rowed_upstream_l1955_195544

noncomputable def speed_of_boat_in_still_water := 18 -- from solution step; b = 18 km/h
def speed_of_stream := 3 -- given
def time := 4 -- given
def distance_downstream := 84 -- given

theorem distance_rowed_upstream 
  (b : ℕ) (s : ℕ) (t : ℕ) (d_down : ℕ) (d_up : ℕ)
  (h_stream : s = 3) 
  (h_time : t = 4)
  (h_distance_downstream : d_down = 84) 
  (h_speed_boat : b = 18) 
  (h_effective_downstream_speed : b + s = d_down / t) :
  d_up = 60 := by
  sorry

end distance_rowed_upstream_l1955_195544


namespace other_acute_angle_in_right_triangle_l1955_195526

theorem other_acute_angle_in_right_triangle (α : ℝ) (β : ℝ) (γ : ℝ) 
  (h1 : α + β + γ = 180) (h2 : γ = 90) (h3 : α = 30) : β = 60 := 
sorry

end other_acute_angle_in_right_triangle_l1955_195526


namespace rod_total_length_l1955_195576

theorem rod_total_length
  (n : ℕ) (l : ℝ)
  (h₁ : n = 50)
  (h₂ : l = 0.85) :
  n * l = 42.5 := by
  sorry

end rod_total_length_l1955_195576


namespace initial_liquid_A_quantity_l1955_195506

theorem initial_liquid_A_quantity
  (x : ℝ)
  (init_A init_B init_C : ℝ)
  (removed_A removed_B removed_C : ℝ)
  (added_B added_C : ℝ)
  (new_A new_B new_C : ℝ)
  (h1 : init_A / init_B = 7 / 5)
  (h2 : init_A / init_C = 7 / 3)
  (h3 : init_A + init_B + init_C = 15 * x)
  (h4 : removed_A = 7 / 15 * 9)
  (h5 : removed_B = 5 / 15 * 9)
  (h6 : removed_C = 3 / 15 * 9)
  (h7 : new_A = init_A - removed_A)
  (h8 : new_B = init_B - removed_B + added_B)
  (h9 : new_C = init_C - removed_C + added_C)
  (h10 : new_A / (new_B + new_C) = 7 / 10)
  (h11 : added_B = 6)
  (h12 : added_C = 3) : 
  init_A = 35.7 :=
sorry

end initial_liquid_A_quantity_l1955_195506


namespace A_and_B_together_complete_work_in_24_days_l1955_195540

-- Define the variables
variables {W_A W_B : ℝ} (completeTime : ℝ → ℝ → ℝ)

-- Define conditions
def A_better_than_B (W_A W_B : ℝ) := W_A = 2 * W_B
def A_takes_36_days (W_A : ℝ) := W_A = 1 / 36

-- The proposition to prove
theorem A_and_B_together_complete_work_in_24_days 
  (h1 : A_better_than_B W_A W_B)
  (h2 : A_takes_36_days W_A) :
  completeTime W_A W_B = 24 :=
sorry

end A_and_B_together_complete_work_in_24_days_l1955_195540


namespace innokentiy_games_l1955_195570

def games_played_egor := 13
def games_played_nikita := 27
def games_played_innokentiy (N : ℕ) := N - games_played_egor

theorem innokentiy_games (N : ℕ) (h : N = games_played_nikita) : games_played_innokentiy N = 14 :=
by {
  sorry
}

end innokentiy_games_l1955_195570


namespace probability_of_negative_cosine_value_l1955_195541

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

def sum_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem probability_of_negative_cosine_value (a : ℕ → ℝ) (S : ℕ → ℝ) 
(h_arith_seq : arithmetic_sequence a)
(h_sum_seq : sum_arithmetic_sequence a S)
(h_S4 : S 4 = Real.pi)
(h_a4_eq_2a2 : a 4 = 2 * a 2) :
∃ p : ℝ, p = 7 / 15 ∧
  ∀ n, 1 ≤ n ∧ n ≤ 30 → 
  ((Real.cos (a n) < 0) → p = 7 / 15) :=
by sorry

end probability_of_negative_cosine_value_l1955_195541


namespace additional_distance_to_achieve_target_average_speed_l1955_195546

-- Given conditions
def initial_distance : ℕ := 20
def initial_speed : ℕ := 40
def target_average_speed : ℕ := 55

-- Prove that the additional distance required to average target speed is 90 miles
theorem additional_distance_to_achieve_target_average_speed 
  (total_distance : ℕ) 
  (total_time : ℚ) 
  (additional_distance : ℕ) 
  (additional_speed : ℕ) :
  total_distance = initial_distance + additional_distance →
  total_time = (initial_distance / initial_speed) + (additional_distance / additional_speed) →
  additional_speed = 60 →
  total_distance / total_time = target_average_speed →
  additional_distance = 90 :=
by 
  sorry

end additional_distance_to_achieve_target_average_speed_l1955_195546


namespace mean_value_of_quadrilateral_angles_l1955_195556

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l1955_195556


namespace no_two_right_angles_in_triangle_l1955_195503

theorem no_two_right_angles_in_triangle 
  (α β γ : ℝ)
  (h1 : α + β + γ = 180) :
  ¬ (α = 90 ∧ β = 90) :=
by
  sorry

end no_two_right_angles_in_triangle_l1955_195503


namespace quadratic_roots_equation_l1955_195554

theorem quadratic_roots_equation (a b c r s : ℝ)
    (h1 : a ≠ 0)
    (h2 : a * r^2 + b * r + c = 0)
    (h3 : a * s^2 + b * s + c = 0) :
    ∃ p q : ℝ, (x^2 - b * x + a * c = 0) ∧ (ar + b, as + b) = (p, q) :=
by
  sorry

end quadratic_roots_equation_l1955_195554


namespace integer_satisfies_mod_l1955_195567

theorem integer_satisfies_mod (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 23) (h3 : 38635 % 23 = n % 23) :
  n = 18 := 
sorry

end integer_satisfies_mod_l1955_195567


namespace max_value_of_g_l1955_195537

def g (n : ℕ) : ℕ :=
  if n < 20 then n + 20 else g (n - 7)

theorem max_value_of_g : ∀ n : ℕ, g n ≤ 39 ∧ (∃ m : ℕ, g m = 39) := by
  sorry

end max_value_of_g_l1955_195537


namespace jersey_sum_adjacent_gt_17_l1955_195521

theorem jersey_sum_adjacent_gt_17 (a : ℕ → ℕ) (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_range : ∀ n, 0 < a n ∧ a n ≤ 10) (h_circle : ∀ n, a n = a (n % 10)) :
  ∃ n, a n + a (n+1) + a (n+2) > 17 :=
by
  sorry

end jersey_sum_adjacent_gt_17_l1955_195521


namespace solve_for_2a_plus_b_l1955_195508

variable (a b : ℝ)

theorem solve_for_2a_plus_b (h1 : 4 * a ^ 2 - b ^ 2 = 12) (h2 : 2 * a - b = 4) : 2 * a + b = 3 := 
by
  sorry

end solve_for_2a_plus_b_l1955_195508


namespace prove_AB_and_circle_symmetry_l1955_195590

-- Definition of point A
def pointA : ℝ × ℝ := (4, -3)

-- Lengths relation |AB| = 2|OA|
def lengths_relation(u v : ℝ) : Prop :=
  u^2 + v^2 = 100

-- Orthogonality condition for AB and OA
def orthogonality_condition(u v : ℝ) : Prop :=
  4 * u - 3 * v = 0

-- Condition that ordinate of B is greater than 0
def ordinate_condition(v : ℝ) : Prop :=
  v - 3 > 0

-- Equation of the circle given in the problem
def given_circle_eqn(x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 10

-- Symmetric circle equation to be proved
def symmetric_circle_eqn(x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 10

theorem prove_AB_and_circle_symmetry :
  (∃ u v : ℝ, lengths_relation u v ∧ orthogonality_condition u v ∧ ordinate_condition v ∧ u = 6 ∧ v = 8) ∧
  (∃ x y : ℝ, given_circle_eqn x y → symmetric_circle_eqn x y) :=
by
  sorry

end prove_AB_and_circle_symmetry_l1955_195590


namespace combined_value_of_a_and_b_l1955_195549

theorem combined_value_of_a_and_b :
  (∃ a b : ℝ,
    0.005 * a = 95 / 100 ∧
    b = 3 * a - 50 ∧
    a + b = 710) :=
sorry

end combined_value_of_a_and_b_l1955_195549


namespace sum_coords_A_eq_neg9_l1955_195583

variable (A B C : ℝ × ℝ)
variable (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 3)
variable (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 / 3)
variable (hB : B = (2, 5))
variable (hC : C = (4, 11))

theorem sum_coords_A_eq_neg9 
  (A B C : ℝ × ℝ)
  (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 3)
  (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 / 3)
  (hB : B = (2, 5))
  (hC : C = (4, 11)) : 
  A.1 + A.2 = -9 :=
  sorry

end sum_coords_A_eq_neg9_l1955_195583


namespace fred_seashells_now_l1955_195594

def seashells_initial := 47
def seashells_given := 25

theorem fred_seashells_now : seashells_initial - seashells_given = 22 := 
by 
  sorry

end fred_seashells_now_l1955_195594


namespace calculate_expression_l1955_195586

theorem calculate_expression :
  ( (5^1010)^2 - (5^1008)^2) / ( (5^1009)^2 - (5^1007)^2) = 25 := 
by
  sorry

end calculate_expression_l1955_195586


namespace perpendicular_condition_l1955_195543

-- Definitions based on the conditions
def line_l1 (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x + (1 - m) * y - 1 = 0
def line_l2 (m : ℝ) (x y : ℝ) : Prop := (m - 1) * x + (2 * m + 1) * y + 4 = 0

-- Perpendicularity condition based on the definition in conditions
def perpendicular (m : ℝ) : Prop :=
  (m + 1) * (m - 1) + (1 - m) * (2 * m + 1) = 0

-- Sufficient but not necessary condition
def sufficient_but_not_necessary (m : ℝ) : Prop :=
  m = 0

-- Final statement to prove
theorem perpendicular_condition :
  sufficient_but_not_necessary 0 -> perpendicular 0 :=
by
  sorry

end perpendicular_condition_l1955_195543


namespace quadratic_rewrite_sum_l1955_195502

theorem quadratic_rewrite_sum (a b c : ℝ) (x : ℝ) :
  -3 * x^2 + 15 * x + 75 = a * (x + b)^2 + c → (a + b + c) = 88.25 :=
sorry

end quadratic_rewrite_sum_l1955_195502


namespace total_fruits_is_43_apple_to_pear_ratio_is_24_to_19_l1955_195577

def keith_pears : ℕ := 6
def keith_apples : ℕ := 4
def jason_pears : ℕ := 9
def jason_apples : ℕ := 8
def joan_pears : ℕ := 4
def joan_apples : ℕ := 12

def total_pears : ℕ := keith_pears + jason_pears + joan_pears
def total_apples : ℕ := keith_apples + jason_apples + joan_apples
def total_fruits : ℕ := total_pears + total_apples
def apple_to_pear_ratio : ℚ := total_apples / total_pears

theorem total_fruits_is_43 : total_fruits = 43 := by
  sorry

theorem apple_to_pear_ratio_is_24_to_19 : apple_to_pear_ratio = 24/19 := by
  sorry

end total_fruits_is_43_apple_to_pear_ratio_is_24_to_19_l1955_195577


namespace difference_sixth_seventh_l1955_195520

theorem difference_sixth_seventh
  (A1 A2 A3 A4 A5 A6 A7 A8 : ℕ)
  (h_avg_8 : (A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8) / 8 = 25)
  (h_avg_2 : (A1 + A2) / 2 = 20)
  (h_avg_3 : (A3 + A4 + A5) / 3 = 26)
  (h_A8 : A8 = 30)
  (h_A6_A8 : A6 = A8 - 6) :
  A7 - A6 = 4 :=
by
  sorry

end difference_sixth_seventh_l1955_195520


namespace where_to_place_minus_sign_l1955_195558

theorem where_to_place_minus_sign :
  (6 + 9 + 12 + 15 + 18 + 21 - 2 * 18) = 45 :=
by
  sorry

end where_to_place_minus_sign_l1955_195558


namespace longest_collection_has_more_pages_l1955_195519

noncomputable def miles_pages_per_inch := 5
noncomputable def daphne_pages_per_inch := 50
noncomputable def miles_height_inches := 240
noncomputable def daphne_height_inches := 25

noncomputable def miles_total_pages := miles_height_inches * miles_pages_per_inch
noncomputable def daphne_total_pages := daphne_height_inches * daphne_pages_per_inch

theorem longest_collection_has_more_pages :
  max miles_total_pages daphne_total_pages = 1250 := by
  -- Skip the proof
  sorry

end longest_collection_has_more_pages_l1955_195519


namespace units_digit_of_expression_l1955_195531

theorem units_digit_of_expression :
  (6 * 16 * 1986 - 6 ^ 4) % 10 = 0 := 
sorry

end units_digit_of_expression_l1955_195531


namespace domain_of_function_l1955_195568

theorem domain_of_function :
  {x : ℝ | x^3 + 5*x^2 + 6*x ≠ 0} =
  {x : ℝ | x < -3} ∪ {x : ℝ | -3 < x ∧ x < -2} ∪ {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x} :=
by
  sorry

end domain_of_function_l1955_195568


namespace min_value_x_plus_2y_l1955_195575

variable (x y : ℝ) (hx : x > 0) (hy : y > 0)

theorem min_value_x_plus_2y (h : (2 / x) + (1 / y) = 1) : x + 2 * y ≥ 8 := 
  sorry

end min_value_x_plus_2y_l1955_195575


namespace dad_strawberries_weight_proof_l1955_195515

/-
Conditions:
1. total_weight (the combined weight of Marco's and his dad's strawberries) is 23 pounds.
2. marco_weight (the weight of Marco's strawberries) is 14 pounds.
We need to prove that dad_weight (the weight of dad's strawberries) is 9 pounds.
-/

def total_weight : ℕ := 23
def marco_weight : ℕ := 14

def dad_weight : ℕ := total_weight - marco_weight

theorem dad_strawberries_weight_proof : dad_weight = 9 := by
  sorry

end dad_strawberries_weight_proof_l1955_195515


namespace machine_does_not_require_repair_l1955_195592

variable (nominal_mass max_deviation standard_deviation : ℝ)
variable (nominal_mass_ge : nominal_mass ≥ 370)
variable (max_deviation_le : max_deviation ≤ 0.1 * nominal_mass)
variable (all_deviations_le_max : ∀ d, d < max_deviation → d < 37)
variable (std_dev_le_max_dev : standard_deviation ≤ max_deviation)

theorem machine_does_not_require_repair :
  ¬ (standard_deviation > 37) :=
by 
  -- sorry annotation indicates the proof goes here
  sorry

end machine_does_not_require_repair_l1955_195592


namespace alice_bob_not_next_to_each_other_l1955_195505

open Nat

theorem alice_bob_not_next_to_each_other (A B C D E : Type) :
  let arrangements := 5!
  let together := 4! * 2
  arrangements - together = 72 :=
by
  let arrangements := 5!
  let together := 4! * 2
  sorry

end alice_bob_not_next_to_each_other_l1955_195505


namespace total_price_of_order_l1955_195555

theorem total_price_of_order :
  let num_ice_cream_bars := 225
  let price_per_ice_cream_bar := 0.60
  let num_sundaes := 125
  let price_per_sundae := 0.52
  (num_ice_cream_bars * price_per_ice_cream_bar + num_sundaes * price_per_sundae) = 200 := 
by
  -- The proof steps go here
  sorry

end total_price_of_order_l1955_195555


namespace fraction_exponentiation_l1955_195582

theorem fraction_exponentiation :
  (1 / 3) ^ 5 = 1 / 243 :=
sorry

end fraction_exponentiation_l1955_195582


namespace find_n_l1955_195500

theorem find_n :
  ∃ n : ℕ, 120 ^ 5 + 105 ^ 5 + 78 ^ 5 + 33 ^ 5 = n ^ 5 ∧ 
  (∀ m : ℕ, 120 ^ 5 + 105 ^ 5 + 78 ^ 5 + 33 ^ 5 = m ^ 5 → m = 144) :=
by
  sorry

end find_n_l1955_195500


namespace sum_of_integers_l1955_195538

theorem sum_of_integers (a b c : ℕ) :
  a > 1 → b > 1 → c > 1 →
  a * b * c = 1728 →
  gcd a b = 1 → gcd b c = 1 → gcd a c = 1 →
  a + b + c = 43 :=
by
  intro ha
  intro hb
  intro hc
  intro hproduct
  intro hgcd_ab
  intro hgcd_bc
  intro hgcd_ac
  sorry

end sum_of_integers_l1955_195538


namespace greatest_possible_value_of_a_l1955_195566

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ (x : ℤ), x * (x + a) = -21 → x^2 + a * x + 21 = 0) ∧
  (∀ (a' : ℕ), (∀ (x : ℤ), x * (x + a') = -21 → x^2 + a' * x + 21 = 0) → a' ≤ a) ∧
  a = 22 :=
sorry

end greatest_possible_value_of_a_l1955_195566


namespace rectangular_prism_diagonals_l1955_195522

structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)
  (length : ℝ)
  (height : ℝ)
  (width : ℝ)
  (length_ne_height : length ≠ height)
  (height_ne_width : height ≠ width)
  (width_ne_length : width ≠ length)

def diagonals (rp : RectangularPrism) : ℕ :=
  let face_diagonals := 12
  let space_diagonals := 4
  face_diagonals + space_diagonals

theorem rectangular_prism_diagonals (rp : RectangularPrism) :
  rp.faces = 6 →
  rp.edges = 12 →
  rp.vertices = 8 →
  diagonals rp = 16 ∧ 4 = 4 :=
by
  intros
  sorry

end rectangular_prism_diagonals_l1955_195522


namespace power_sum_divisible_by_five_l1955_195513

theorem power_sum_divisible_by_five : 
  (3^444 + 4^333) % 5 = 0 := 
by 
  sorry

end power_sum_divisible_by_five_l1955_195513


namespace largest_proper_divisor_condition_l1955_195518

def is_proper_divisor (n k : ℕ) : Prop :=
  k > 1 ∧ k < n ∧ n % k = 0

theorem largest_proper_divisor_condition (n p : ℕ) (hp : is_proper_divisor n p) (hl : ∀ k, is_proper_divisor n k → k ≤ n / p):
  n = 12 ∨ n = 33 :=
by
  -- Placeholder for proof
  sorry

end largest_proper_divisor_condition_l1955_195518


namespace shaded_region_area_eq_108_l1955_195551

/-- There are two concentric circles, where the outer circle has twice the radius of the inner circle,
and the total boundary length of the shaded region is 36π. Prove that the area of the shaded region
is nπ, where n = 108. -/
theorem shaded_region_area_eq_108 (r : ℝ) (h_outer : ∀ (c₁ c₂ : ℝ), c₁ = 2 * c₂) 
  (h_boundary : 2 * Real.pi * r + 2 * Real.pi * (2 * r) = 36 * Real.pi) : 
  ∃ (n : ℕ), n = 108 ∧ (Real.pi * (2 * r)^2 - Real.pi * r^2) = n * Real.pi := 
sorry

end shaded_region_area_eq_108_l1955_195551


namespace max_grandchildren_l1955_195569

theorem max_grandchildren (children_count : ℕ) (common_gc : ℕ) (special_gc_count : ℕ) : 
  children_count = 8 ∧ common_gc = 8 ∧ special_gc_count = 5 →
  (6 * common_gc + 2 * special_gc_count) = 58 := by
  sorry

end max_grandchildren_l1955_195569


namespace inequality_transitive_l1955_195530

theorem inequality_transitive (a b c : ℝ) (h : a < b) (h' : b < c) : a - c < b - c :=
by
  sorry

end inequality_transitive_l1955_195530


namespace number_of_positive_integer_solutions_l1955_195593

theorem number_of_positive_integer_solutions :
  ∃ n : ℕ, n = 84 ∧ (∀ x y z t : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < t ∧ x + y + z + t = 10 → true) :=
sorry

end number_of_positive_integer_solutions_l1955_195593


namespace fraction_of_menu_l1955_195565

def total_dishes (total : ℕ) : Prop := 
  6 = (1/4:ℚ) * total

def vegan_dishes (vegan : ℕ) (soy_free : ℕ) : Prop :=
  vegan = 6 ∧ soy_free = vegan - 5

theorem fraction_of_menu (total vegan soy_free : ℕ) (h1 : total_dishes total)
  (h2 : vegan_dishes vegan soy_free) : (soy_free:ℚ) / total = 1 / 24 := 
by sorry

end fraction_of_menu_l1955_195565


namespace line_sum_slope_intercept_l1955_195512

theorem line_sum_slope_intercept (m b : ℝ) (x y : ℝ)
  (hm : m = 3)
  (hpoint : (x, y) = (-2, 4))
  (heq : y = m * x + b) :
  m + b = 13 :=
by
  sorry

end line_sum_slope_intercept_l1955_195512


namespace axis_of_symmetry_parabola_l1955_195598

theorem axis_of_symmetry_parabola : 
  (∃ a b c : ℝ, ∀ x : ℝ, (y = x^2 + 4 * x - 5) ∧ (a = 1) ∧ (b = 4) → ( x = -b / (2 * a) ) → ( x = -2 ) ) :=
by
  sorry

end axis_of_symmetry_parabola_l1955_195598


namespace circle_symmetric_about_line_l1955_195597

-- The main proof statement
theorem circle_symmetric_about_line (x y : ℝ) (k : ℝ) :
  (x - 1)^2 + (y - 1)^2 = 2 ∧ y = k * x + 3 → k = -2 :=
by
  sorry

end circle_symmetric_about_line_l1955_195597


namespace area_of_triangle_ABC_l1955_195574

variable {α : Type} [LinearOrder α] [Field α]

-- Given: 
variables (A B C D E F : α) (area_ABC area_BDA area_DCA : α)

-- Conditions:
variable (midpoint_D : 2 * D = B + C)
variable (ratio_AE_EC : 3 * E = A + C)
variable (ratio_AF_FD : 2 * F = A + D)
variable (area_DEF : area_ABC / 6 = 12)

-- To Show:
theorem area_of_triangle_ABC :
  area_ABC = 96 :=
by
  sorry

end area_of_triangle_ABC_l1955_195574


namespace total_photos_l1955_195585

-- Define the number of photos Claire has taken
def photos_by_Claire : ℕ := 8

-- Define the number of photos Lisa has taken
def photos_by_Lisa : ℕ := 3 * photos_by_Claire

-- Define the number of photos Robert has taken
def photos_by_Robert : ℕ := photos_by_Claire + 16

-- State the theorem we want to prove
theorem total_photos : photos_by_Lisa + photos_by_Robert = 48 :=
by
  sorry

end total_photos_l1955_195585


namespace triangle_side_height_inequality_l1955_195581

theorem triangle_side_height_inequality (a b h_a h_b S : ℝ) (h1 : a > b) 
  (h2: h_a = 2 * S / a) (h3: h_b = 2 * S / b) :
  a + h_a ≥ b + h_b :=
by sorry

end triangle_side_height_inequality_l1955_195581


namespace min_value_of_expression_l1955_195572

theorem min_value_of_expression (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (habc : a + b + c = 1) (expected_value : 3 * a + 2 * b = 2) :
  ∃ a b, (a + b + (1 - a - b) = 1) ∧ (3 * a + 2 * b = 2) ∧ (∀ a b, ∃ m, m = (2/a + 1/(3*b)) ∧ m = 16/3) :=
sorry

end min_value_of_expression_l1955_195572


namespace find_13th_result_l1955_195595

theorem find_13th_result 
  (average_25 : ℕ) (average_12_first : ℕ) (average_12_last : ℕ) 
  (total_25 : average_25 * 25 = 600) 
  (total_12_first : average_12_first * 12 = 168) 
  (total_12_last : average_12_last * 12 = 204) 
: average_25 - average_12_first - average_12_last = 228 :=
by
  sorry

end find_13th_result_l1955_195595


namespace golf_balls_dozen_count_l1955_195516

theorem golf_balls_dozen_count (n d : Nat) (h1 : n = 108) (h2 : d = 12) : n / d = 9 :=
by
  sorry

end golf_balls_dozen_count_l1955_195516


namespace value_of_a_l1955_195514

theorem value_of_a (a : ℝ) :
  (∃ (l1 l2 : (ℝ × ℝ × ℝ)),
   l1 = (1, -a, a) ∧ l2 = (3, 1, 2) ∧
   (∃ (m1 m2 : ℝ), 
    (m1 = (1 : ℝ) / a ∧ m2 = -3) ∧ 
    (m1 * m2 = -1))) → a = 3 :=
by sorry

end value_of_a_l1955_195514


namespace cost_price_250_l1955_195527

theorem cost_price_250 (C : ℝ) (h1 : 0.90 * C = C - 0.10 * C) (h2 : 1.10 * C = C + 0.10 * C) (h3 : 1.10 * C - 0.90 * C = 50) : C = 250 := 
by
  sorry

end cost_price_250_l1955_195527


namespace solution_set_of_inequality_l1955_195589

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_of_inequality 
  (hf_even : ∀ x : ℝ, f x = f (|x|))
  (hf_increasing : ∀ x y : ℝ, x < y → x < 0 → y < 0 → f x < f y)
  (hf_value : f 3 = 1) :
  {x : ℝ | f (x - 1) < 1} = {x : ℝ | x > 4 ∨ x < -2} := 
sorry

end solution_set_of_inequality_l1955_195589


namespace delaney_bus_miss_theorem_l1955_195511

def delaneyMissesBus : Prop :=
  let busDeparture := 8 * 60               -- bus departure time in minutes (8:00 a.m.)
  let travelTime := 30                     -- travel time in minutes
  let departureTime := 7 * 60 + 50         -- departure time from home in minutes (7:50 a.m.)
  let arrivalTime := departureTime + travelTime -- arrival time at the pick-up point
  arrivalTime - busDeparture = 20 -- he misses the bus by 20 minutes

theorem delaney_bus_miss_theorem : delaneyMissesBus := sorry

end delaney_bus_miss_theorem_l1955_195511


namespace intersection_x_value_l1955_195524

theorem intersection_x_value :
  ∀ x y: ℝ,
    (y = 3 * x - 15) ∧ (3 * x + y = 120) → x = 22.5 := by
  sorry

end intersection_x_value_l1955_195524


namespace impossible_fractions_l1955_195510

theorem impossible_fractions (a b c r s t : ℕ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_pos_t : 0 < t)
  (h1 : a * b + 1 = r ^ 2) (h2 : a * c + 1 = s ^ 2) (h3 : b * c + 1 = t ^ 2) :
  ¬ (∃ (k1 k2 k3 : ℕ), rt / s = k1 ∧ rs / t = k2 ∧ st / r = k3) :=
by
  sorry

end impossible_fractions_l1955_195510


namespace hypotenuse_length_l1955_195580

theorem hypotenuse_length (a b c : ℝ) (h1 : a + b + c = 36) (h2 : 0.5 * a * b = 24) (h3 : a^2 + b^2 = c^2) :
  c = 50 / 3 :=
sorry

end hypotenuse_length_l1955_195580


namespace tax_amount_self_employed_l1955_195562

noncomputable def gross_income : ℝ := 350000.00
noncomputable def tax_rate : ℝ := 0.06

theorem tax_amount_self_employed :
  gross_income * tax_rate = 21000.00 :=
by
  sorry

end tax_amount_self_employed_l1955_195562


namespace coordinate_plane_condition_l1955_195588

theorem coordinate_plane_condition (a : ℝ) :
  a - 1 < 0 ∧ (3 * a + 1) / (a - 1) < 0 ↔ - (1 : ℝ)/3 < a ∧ a < 1 :=
by
  sorry

end coordinate_plane_condition_l1955_195588


namespace arithmetic_sequence_sum_ratio_l1955_195509

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 7 / 3) :
  S 5 / S 3 = 5 := 
by
  sorry

end arithmetic_sequence_sum_ratio_l1955_195509


namespace greatest_common_ratio_l1955_195534

theorem greatest_common_ratio {a b c : ℝ} (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : (b = (a + c) / 2 → b^2 = a * c) ∨ (c = (a + b) / 2 ∧ b = -a / 2)) :
  ∃ r : ℝ, r = -2 :=
by
  sorry

end greatest_common_ratio_l1955_195534


namespace speed_ratio_l1955_195539

variable (vA vB : ℝ)
variable (H1 : 3 * vA = abs (-400 + 3 * vB))
variable (H2 : 10 * vA = abs (-400 + 10 * vB))

theorem speed_ratio (vA vB : ℝ) (H1 : 3 * vA = abs (-400 + 3 * vB)) (H2 : 10 * vA = abs (-400 + 10 * vB)) : 
  vA / vB = 5 / 6 :=
  sorry

end speed_ratio_l1955_195539


namespace iris_total_spending_l1955_195578

theorem iris_total_spending :
  ∀ (price_jacket price_shorts price_pants : ℕ), 
  price_jacket = 10 → 
  price_shorts = 6 → 
  price_pants = 12 → 
  (3 * price_jacket + 2 * price_shorts + 4 * price_pants) = 90 :=
by
  intros price_jacket price_shorts price_pants
  sorry

end iris_total_spending_l1955_195578


namespace farmer_plough_rate_l1955_195525

theorem farmer_plough_rate (x : ℝ) (h1 : 85 * ((1400 / x) + 2) + 40 = 1400) : x = 100 :=
by
  sorry

end farmer_plough_rate_l1955_195525


namespace reading_enhusiasts_not_related_to_gender_l1955_195545

noncomputable def contingency_table (boys_scores : List Nat) (girls_scores : List Nat) :
  (Nat × Nat × Nat × Nat × Nat × Nat) × (Nat × Nat × Nat × Nat × Nat × Nat) :=
  let boys_range := (2, 3, 5, 15, 18, 12)
  let girls_range := (0, 5, 10, 10, 7, 13)
  ((2, 3, 5, 15, 18, 12), (0, 5, 10, 10, 7, 13))

theorem reading_enhusiasts_not_related_to_gender (boys_scores : List Nat) (girls_scores : List Nat) :
  let table := contingency_table boys_scores girls_scores
  let (boys_range, girls_range) := table
  let a := 45 -- Boys who are reading enthusiasts
  let b := 10 -- Boys who are non-reading enthusiasts
  let c := 30 -- Girls who are reading enthusiasts
  let d := 15 -- Girls who are non-reading enthusiasts
  let n := a + b + c + d
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  k_squared < 3.841 := 
sorry

end reading_enhusiasts_not_related_to_gender_l1955_195545


namespace total_weight_is_correct_l1955_195507

def siblings_suitcases : Nat := 1 + 2 + 3 + 4 + 5 + 6
def weight_per_sibling_suitcase : Nat := 10
def total_weight_siblings : Nat := siblings_suitcases * weight_per_sibling_suitcase

def parents : Nat := 2
def suitcases_per_parent : Nat := 3
def weight_per_parent_suitcase : Nat := 12
def total_weight_parents : Nat := parents * suitcases_per_parent * weight_per_parent_suitcase

def grandparents : Nat := 2
def suitcases_per_grandparent : Nat := 2
def weight_per_grandparent_suitcase : Nat := 8
def total_weight_grandparents : Nat := grandparents * suitcases_per_grandparent * weight_per_grandparent_suitcase

def other_relatives_suitcases : Nat := 8
def weight_per_other_relatives_suitcase : Nat := 15
def total_weight_other_relatives : Nat := other_relatives_suitcases * weight_per_other_relatives_suitcase

def total_weight_all_suitcases : Nat := total_weight_siblings + total_weight_parents + total_weight_grandparents + total_weight_other_relatives

theorem total_weight_is_correct : total_weight_all_suitcases = 434 := by {
  sorry
}

end total_weight_is_correct_l1955_195507


namespace bird_families_difference_l1955_195504

-- Define the conditions
def bird_families_to_africa : ℕ := 47
def bird_families_to_asia : ℕ := 94

-- The proof statement
theorem bird_families_difference : (bird_families_to_asia - bird_families_to_africa = 47) :=
by
  sorry

end bird_families_difference_l1955_195504


namespace total_distance_100_l1955_195548

-- Definitions for the problem conditions:
def initial_velocity : ℕ := 40
def common_difference : ℕ := 10
def total_time (v₀ : ℕ) (d : ℕ) : ℕ := (v₀ / d) + 1  -- The total time until the car stops
def distance_traveled (v₀ : ℕ) (d : ℕ) : ℕ :=
  (v₀ * total_time v₀ d) - (d * total_time v₀ d * (total_time v₀ d - 1)) / 2

-- Statement to prove:
theorem total_distance_100 : distance_traveled initial_velocity common_difference = 100 := by
  sorry

end total_distance_100_l1955_195548


namespace race_distance_l1955_195571

theorem race_distance
  (A B : Type)
  (D : ℕ) -- D is the total distance of the race
  (Va Vb : ℕ) -- A's speed and B's speed
  (H1 : D / 28 = Va) -- A's speed calculated from D and time
  (H2 : (D - 56) / 28 = Vb) -- B's speed calculated from distance and time
  (H3 : 56 / 7 = Vb) -- B's speed can also be calculated directly
  (H4 : Va = D / 28)
  (H5 : Vb = (D - 56) / 28) :
  D = 280 := sorry

end race_distance_l1955_195571


namespace michael_class_choosing_l1955_195587

open Nat

theorem michael_class_choosing :
  (choose 6 3) * (choose 4 2) + (choose 6 4) * (choose 4 1) + (choose 6 5) = 186 := 
by
  sorry

end michael_class_choosing_l1955_195587


namespace max_perimeter_isosceles_triangle_l1955_195550

/-- Out of all triangles with the same base and the same angle at the vertex, 
    the triangle with the largest perimeter is isosceles -/
theorem max_perimeter_isosceles_triangle {α β γ : ℝ} (b : ℝ) (B : ℝ) (A C : ℝ) 
  (hB : 0 < B ∧ B < π) (hβ : α + C = B) (h1 : A = β) (h2 : γ = β) :
  α = γ := sorry

end max_perimeter_isosceles_triangle_l1955_195550


namespace find_n_l1955_195573

theorem find_n (n : ℤ) : 43^2 = 1849 ∧ 44^2 = 1936 ∧ 45^2 = 2025 ∧ 46^2 = 2116 ∧ n < Real.sqrt 2023 ∧ Real.sqrt 2023 < n + 1 → n = 44 :=
by
  sorry

end find_n_l1955_195573


namespace evaluate_expression_l1955_195579

def S (a b c : ℤ) := a + b + c

theorem evaluate_expression (a b c : ℤ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 18) :
  (144 * ((1 : ℚ) / b - (1 : ℚ) / c) + 196 * ((1 : ℚ) / c - (1 : ℚ) / a) + 324 * ((1 : ℚ) / a - (1 : ℚ) / b)) /
  (12 * ((1 : ℚ) / b - (1 : ℚ) / c) + 14 * ((1 : ℚ) / c - (1 : ℚ) / a) + 18 * ((1 : ℚ) / a - (1 : ℚ) / b)) = 44 := 
sorry

end evaluate_expression_l1955_195579


namespace find_a_l1955_195563

noncomputable def circle1 (x y : ℝ) := x^2 + y^2 + 4 * y = 0

noncomputable def circle2 (x y a : ℝ) := x^2 + y^2 + 2 * (a - 1) * x + 2 * y + a^2 = 0

theorem find_a (a : ℝ) :
  (∀ x y, circle1 x y → circle2 x y a → false) → a = -2 :=
by sorry

end find_a_l1955_195563


namespace problem_inequality_l1955_195547

theorem problem_inequality {a : ℝ} (h : ∀ x : ℝ, (x - a) * (1 - x - a) < 1) : 
  -1/2 < a ∧ a < 3/2 := by
  sorry

end problem_inequality_l1955_195547


namespace fencing_required_l1955_195535

theorem fencing_required (L W : ℕ) (hL : L = 30) (hArea : L * W = 720) : L + 2 * W = 78 :=
by
  sorry

end fencing_required_l1955_195535


namespace area_of_triangle_with_rational_vertices_on_unit_circle_is_rational_l1955_195561

def rational_coords_on_unit_circle (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  x₁^2 + y₁^2 = 1 ∧ x₂^2 + y₂^2 = 1 ∧ x₃^2 + y₃^2 = 1

theorem area_of_triangle_with_rational_vertices_on_unit_circle_is_rational
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ)
  (h : rational_coords_on_unit_circle x₁ y₁ x₂ y₂ x₃ y₃) :
  ∃ (A : ℚ), A = 1 / 2 * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)) :=
sorry

end area_of_triangle_with_rational_vertices_on_unit_circle_is_rational_l1955_195561


namespace company_food_purchase_1_l1955_195584

theorem company_food_purchase_1 (x y : ℕ) (h1: x + y = 170) (h2: 15 * x + 20 * y = 3000) : 
  x = 80 ∧ y = 90 := by
  sorry

end company_food_purchase_1_l1955_195584


namespace find_positive_integer_x_l1955_195542

def positive_integer (x : ℕ) : Prop :=
  x > 0

def n (x : ℕ) : ℕ :=
  x^2 + 3 * x + 20

def d (x : ℕ) : ℕ :=
  3 * x + 4

def division_property (x : ℕ) : Prop :=
  ∃ q r : ℕ, q = x ∧ r = 8 ∧ n x = q * d x + r

theorem find_positive_integer_x :
  ∃ x : ℕ, positive_integer x ∧ n x = x * d x + 8 :=
sorry

end find_positive_integer_x_l1955_195542


namespace distinct_values_l1955_195559

-- Define the expressions as terms in Lean
def expr1 : ℕ := 3 ^ (3 ^ 3)
def expr2 : ℕ := (3 ^ 3) ^ 3

-- State the theorem that these terms yield exactly two distinct values
theorem distinct_values : (expr1 ≠ expr2) ∧ ((expr1 = 3^27) ∨ (expr1 = 19683)) ∧ ((expr2 = 3^27) ∨ (expr2 = 19683)) := 
  sorry

end distinct_values_l1955_195559


namespace monroe_legs_total_l1955_195552

def num_spiders : ℕ := 8
def num_ants : ℕ := 12
def legs_per_spider : ℕ := 8
def legs_per_ant : ℕ := 6

theorem monroe_legs_total :
  num_spiders * legs_per_spider + num_ants * legs_per_ant = 136 :=
by
  sorry

end monroe_legs_total_l1955_195552


namespace fourth_person_height_l1955_195528

variable (H : ℝ)
variable (height1 height2 height3 height4 : ℝ)

theorem fourth_person_height
  (h1 : height1 = H)
  (h2 : height2 = H + 2)
  (h3 : height3 = H + 4)
  (h4 : height4 = H + 10)
  (avg_height : (height1 + height2 + height3 + height4) / 4 = 78) :
  height4 = 84 :=
by
  sorry

end fourth_person_height_l1955_195528


namespace simplify_fraction_l1955_195533

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l1955_195533


namespace min_policemen_needed_l1955_195591

-- Definitions of the problem parameters
def city_layout (n m : ℕ) := n > 0 ∧ m > 0

-- Function to calculate the minimum number of policemen
def min_policemen (n m : ℕ) : ℕ := (m - 1) * (n - 1)

-- The theorem to prove
theorem min_policemen_needed (n m : ℕ) (h : city_layout n m) : min_policemen n m = (m - 1) * (n - 1) :=
by
  unfold city_layout at h
  unfold min_policemen
  sorry

end min_policemen_needed_l1955_195591
