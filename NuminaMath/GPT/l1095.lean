import Mathlib

namespace sum_of_two_numbers_eq_l1095_109545

theorem sum_of_two_numbers_eq (x y : ℝ) (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) : x + y = (16 * Real.sqrt 3) / 3 :=
by sorry

end sum_of_two_numbers_eq_l1095_109545


namespace chord_length_perpendicular_bisector_of_radius_l1095_109583

theorem chord_length_perpendicular_bisector_of_radius (r : ℝ) (h : r = 15) :
  ∃ (CD : ℝ), CD = 15 * Real.sqrt 3 :=
by
  sorry

end chord_length_perpendicular_bisector_of_radius_l1095_109583


namespace y1_gt_y2_l1095_109556

theorem y1_gt_y2 (y : ℤ → ℤ) (h_eq : ∀ x, y x = 8 * x - 1)
  (y1 y2 : ℤ) (h_y1 : y 3 = y1) (h_y2 : y 2 = y2) : y1 > y2 :=
by
  -- proof
  sorry

end y1_gt_y2_l1095_109556


namespace part_a_part_b_l1095_109586

variable {A : Type*} [Ring A]

def B (A : Type*) [Ring A] : Set A :=
  {a | a^2 = 1}

variable (a : A) (b : B A)

theorem part_a (a : A) (b : A) (h : b ∈ B A) : a * b - b * a = b * a * b - a := by
  sorry

theorem part_b (A : Type*) [Ring A] (h : ∀ x : A, x^2 = 0 -> x = 0) : Group (B A) := by
  sorry

end part_a_part_b_l1095_109586


namespace profit_function_is_correct_marginal_profit_function_is_correct_profit_function_max_value_marginal_profit_function_max_value_profit_and_marginal_profit_max_not_equal_l1095_109538

noncomputable def R (x : ℕ) : ℝ := 3000 * x - 20 * x^2
noncomputable def C (x : ℕ) : ℝ := 500 * x + 4000
noncomputable def p (x : ℕ) : ℝ := R x - C x
noncomputable def Mp (x : ℕ) : ℝ := p (x + 1) - p x

theorem profit_function_is_correct : ∀ x, p x = -20 * x^2 + 2500 * x - 4000 := 
by 
  intro x
  sorry

theorem marginal_profit_function_is_correct : ∀ x, 0 < x ∧ x ≤ 100 → Mp x = -40 * x + 2480 := 
by 
  intro x
  sorry

theorem profit_function_max_value : ∃ x, (x = 62 ∨ x = 63) ∧ p x = 74120 :=
by 
  sorry

theorem marginal_profit_function_max_value : ∃ x, x = 1 ∧ Mp x = 2440 :=
by 
  sorry

theorem profit_and_marginal_profit_max_not_equal : ¬ (∃ x y, (x = 62 ∨ x = 63) ∧ y = 1 ∧ p x = Mp y) :=
by 
  sorry

end profit_function_is_correct_marginal_profit_function_is_correct_profit_function_max_value_marginal_profit_function_max_value_profit_and_marginal_profit_max_not_equal_l1095_109538


namespace percent_of_x_is_y_l1095_109581

variable {x y : ℝ}

theorem percent_of_x_is_y
  (h : 0.5 * (x - y) = 0.4 * (x + y)) :
  y = (1 / 9) * x :=
sorry

end percent_of_x_is_y_l1095_109581


namespace consecutive_integers_sum_l1095_109575

theorem consecutive_integers_sum (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end consecutive_integers_sum_l1095_109575


namespace complex_root_condition_l1095_109591

open Complex

theorem complex_root_condition (u v : ℂ) 
    (h1 : 3 * abs (u + 1) * abs (v + 1) ≥ abs (u * v + 5 * u + 5 * v + 1))
    (h2 : abs (u + v) = abs (u * v + 1)) :
    u = 1 ∨ v = 1 :=
sorry

end complex_root_condition_l1095_109591


namespace rational_square_l1095_109589

theorem rational_square (a b c : ℚ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) : ∃ r : ℚ, (1 / (a - b)^2) + (1 / (b - c)^2) + (1 / (c - a)^2) = r^2 := 
by 
  sorry

end rational_square_l1095_109589


namespace greatest_integer_is_8_l1095_109500

theorem greatest_integer_is_8 {a b : ℤ} (h_sum : a + b + 8 = 21) : max a (max b 8) = 8 :=
by
  sorry

end greatest_integer_is_8_l1095_109500


namespace inequality_solution_inequality_proof_l1095_109508

def f (x: ℝ) := |x - 5|

theorem inequality_solution : {x : ℝ | f x + f (x + 2) ≤ 3} = {x | 5 / 2 ≤ x ∧ x ≤ 11 / 2} :=
sorry

theorem inequality_proof (a x : ℝ) (h : a < 0) : f (a * x) - f (5 * a) ≥ a * f x :=
sorry

end inequality_solution_inequality_proof_l1095_109508


namespace total_oranges_l1095_109563

def oranges_from_first_tree : Nat := 80
def oranges_from_second_tree : Nat := 60
def oranges_from_third_tree : Nat := 120

theorem total_oranges : oranges_from_first_tree + oranges_from_second_tree + oranges_from_third_tree = 260 :=
by
  sorry

end total_oranges_l1095_109563


namespace value_of_x_plus_inv_x_l1095_109504

theorem value_of_x_plus_inv_x (x : ℝ) (hx : x ≠ 0) (t : ℝ) (ht : t = x^2 + (1 / x)^2) : x + (1 / x) = 5 :=
by
  have ht_val : t = 23 := by
    rw [ht] -- assuming t = 23 by condition
    sorry -- proof continuation placeholder

  -- introduce y and relate it to t
  let y := x + (1 / x)

  -- express t in terms of y and handle the algebra:
  have t_expr : t = y^2 - 2 := by
    sorry -- proof continuation placeholder

  -- show that y^2 = 25 and therefore y = 5 as the only valid solution:
  have y_val : y = 5 := by
    sorry -- proof continuation placeholder

  -- hence, the required value is found:
  exact y_val

end value_of_x_plus_inv_x_l1095_109504


namespace probability_X1_lt_X2_lt_X3_is_1_6_l1095_109553

noncomputable def probability_X1_lt_X2_lt_X3 (n : ℕ) (h : n ≥ 3) : ℚ :=
if h : n ≥ 3 then
  1/6
else
  0

theorem probability_X1_lt_X2_lt_X3_is_1_6 (n : ℕ) (h : n ≥ 3) :
  probability_X1_lt_X2_lt_X3 n h = 1/6 :=
sorry

end probability_X1_lt_X2_lt_X3_is_1_6_l1095_109553


namespace cone_cylinder_volume_ratio_l1095_109503

theorem cone_cylinder_volume_ratio :
  let π := Real.pi
  let Vcylinder := π * (3:ℝ)^2 * (15:ℝ)
  let Vcone := (1/3:ℝ) * π * (2:ℝ)^2 * (5:ℝ)
  (Vcone / Vcylinder) = (4 / 81) :=
by
  let π := Real.pi
  let r_cylinder := (3:ℝ)
  let h_cylinder := (15:ℝ)
  let r_cone := (2:ℝ)
  let h_cone := (5:ℝ)
  let Vcylinder := π * r_cylinder^2 * h_cylinder
  let Vcone := (1/3:ℝ) * π * r_cone^2 * h_cone
  have h1 : Vcylinder = 135 * π := by sorry
  have h2 : Vcone = (20 / 3) * π := by sorry
  have h3 : (Vcone / Vcylinder) = (4 / 81) := by sorry
  exact h3

end cone_cylinder_volume_ratio_l1095_109503


namespace find_n_l1095_109552

theorem find_n (x n : ℝ) (h₁ : x = 1) (h₂ : 5 / (n + 1 / x) = 1) : n = 4 :=
sorry

end find_n_l1095_109552


namespace all_statements_imply_implication_l1095_109509

variables (p q r : Prop)

theorem all_statements_imply_implication :
  (p ∧ ¬ q ∧ r → ((p → q) → r)) ∧
  (¬ p ∧ ¬ q ∧ r → ((p → q) → r)) ∧
  (p ∧ ¬ q ∧ ¬ r → ((p → q) → r)) ∧
  (¬ p ∧ q ∧ r → ((p → q) → r)) :=
by { sorry }

end all_statements_imply_implication_l1095_109509


namespace price_difference_l1095_109566

theorem price_difference (total_cost shirt_price : ℝ) (h1 : total_cost = 80.34) (h2 : shirt_price = 36.46) :
  (total_cost - shirt_price) - shirt_price = 7.42 :=
by
  sorry

end price_difference_l1095_109566


namespace all_of_the_above_were_used_as_money_l1095_109516

-- Defining the conditions that each type was used as money
def gold_used_as_money : Prop := True
def stones_used_as_money : Prop := True
def horses_used_as_money : Prop := True
def dried_fish_used_as_money : Prop := True
def mollusk_shells_used_as_money : Prop := True

-- The statement to prove
theorem all_of_the_above_were_used_as_money :
  gold_used_as_money ∧
  stones_used_as_money ∧
  horses_used_as_money ∧
  dried_fish_used_as_money ∧
  mollusk_shells_used_as_money ↔
  (∀ x, (x = "gold" ∨ x = "stones" ∨ x = "horses" ∨ x = "dried fish" ∨ x = "mollusk shells") → 
  (x = "gold" ∧ gold_used_as_money) ∨ 
  (x = "stones" ∧ stones_used_as_money) ∨ 
  (x = "horses" ∧ horses_used_as_money) ∨ 
  (x = "dried fish" ∧ dried_fish_used_as_money) ∨ 
  (x = "mollusk shells" ∧ mollusk_shells_used_as_money)) :=
by
  sorry

end all_of_the_above_were_used_as_money_l1095_109516


namespace interior_diagonal_length_l1095_109570

variables (a b c : ℝ)

-- Conditions
def surface_area_eq : Prop := 2 * (a * b + b * c + c * a) = 22
def edge_length_eq : Prop := 4 * (a + b + c) = 24

-- Question to be proved
theorem interior_diagonal_length :
  surface_area_eq a b c → edge_length_eq a b c → (Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 14) :=
by
  intros h1 h2
  sorry

end interior_diagonal_length_l1095_109570


namespace households_with_only_bike_l1095_109559

theorem households_with_only_bike
  (N : ℕ) (H_no_car_or_bike : ℕ) (H_car_bike : ℕ) (H_car : ℕ)
  (hN : N = 90)
  (h_no_car_or_bike : H_no_car_or_bike = 11)
  (h_car_bike : H_car_bike = 16)
  (h_car : H_car = 44) :
  ∃ (H_bike_only : ℕ), H_bike_only = 35 :=
by {
  sorry
}

end households_with_only_bike_l1095_109559


namespace find_fraction_l1095_109544

theorem find_fraction (F : ℝ) (N : ℝ) (X : ℝ)
  (h1 : 0.85 * F = 36)
  (h2 : N = 70.58823529411765)
  (h3 : F = 42.35294117647059) :
  X * N = 42.35294117647059 → X = 0.6 :=
by
  sorry

end find_fraction_l1095_109544


namespace abs_inequality_solution_l1095_109569

theorem abs_inequality_solution (x : ℝ) : 
  (|5 - 2*x| >= 3) ↔ (x ≤ 1 ∨ x ≥ 4) := sorry

end abs_inequality_solution_l1095_109569


namespace tutors_meeting_schedule_l1095_109594

/-- In a school, five tutors, Jaclyn, Marcelle, Susanna, Wanda, and Thomas, 
are scheduled to work in the library. Their schedules are as follows: 
Jaclyn works every fifth school day, Marcelle works every sixth school day, 
Susanna works every seventh school day, Wanda works every eighth school day, 
and Thomas works every ninth school day. Today, all five tutors are working 
in the library. Prove that the least common multiple of 5, 6, 7, 8, and 9 is 2520 days. 
-/
theorem tutors_meeting_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))) = 2520 := 
by
  sorry

end tutors_meeting_schedule_l1095_109594


namespace max_height_reached_by_rocket_l1095_109579

def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 36

theorem max_height_reached_by_rocket : ∃ t : ℝ, h t = 144 ∧ ∀ t' : ℝ, h t' ≤ 144 := sorry

end max_height_reached_by_rocket_l1095_109579


namespace trapezium_parallel_side_length_l1095_109557

theorem trapezium_parallel_side_length (a h area x : ℝ) (h1 : a = 20) (h2 : h = 15) (h3 : area = 285) :
  area = 1/2 * (a + x) * h → x = 18 :=
by
  -- placeholder for the proof
  sorry

end trapezium_parallel_side_length_l1095_109557


namespace perp_lines_iff_m_values_l1095_109598

section
variables (m x y : ℝ)

def l1 := (m * x + y - 2 = 0)
def l2 := ((m + 1) * x - 2 * m * y + 1 = 0)

theorem perp_lines_iff_m_values (h1 : l1 m x y) (h2 : l2 m x y) (h_perp : (m * (m + 1) + (-2 * m) = 0)) : m = 0 ∨ m = 1 :=
by {
  sorry
}
end

end perp_lines_iff_m_values_l1095_109598


namespace polynomial_q_value_l1095_109527

theorem polynomial_q_value :
  ∀ (p q d : ℝ),
    (d = 6) →
    (-p / 3 = -d) →
    (1 + p + q + d = - d) →
    q = -31 :=
by sorry

end polynomial_q_value_l1095_109527


namespace money_left_after_purchase_l1095_109543

-- The costs and amounts for each item
def bread_cost : ℝ := 2.35
def num_bread : ℝ := 4
def peanut_butter_cost : ℝ := 3.10
def num_peanut_butter : ℝ := 2
def honey_cost : ℝ := 4.50
def num_honey : ℝ := 1

-- The coupon discount and budget
def coupon_discount : ℝ := 2
def budget : ℝ := 20

-- Calculate the total cost before applying the coupon
def total_before_coupon : ℝ := num_bread * bread_cost + num_peanut_butter * peanut_butter_cost + num_honey * honey_cost

-- Calculate the total cost after applying the coupon
def total_after_coupon : ℝ := total_before_coupon - coupon_discount

-- Calculate the money left over after the purchase
def money_left_over : ℝ := budget - total_after_coupon

-- The theorem to be proven
theorem money_left_after_purchase : money_left_over = 1.90 :=
by
  -- The proof of this theorem will involve the specific calculations and will be filled in later
  sorry

end money_left_after_purchase_l1095_109543


namespace largest_inscribed_equilateral_triangle_area_l1095_109560

noncomputable def inscribed_triangle_area (r : ℝ) : ℝ :=
  let s := r * (3 / Real.sqrt 3)
  let h := (Real.sqrt 3 / 2) * s
  (1 / 2) * s * h

theorem largest_inscribed_equilateral_triangle_area :
  inscribed_triangle_area 10 = 75 * Real.sqrt 3 :=
by
  simp [inscribed_triangle_area]
  sorry

end largest_inscribed_equilateral_triangle_area_l1095_109560


namespace train_crossing_time_l1095_109518

open Real

noncomputable def time_to_cross_bridge 
  (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_train_ms := speed_train_kmh * (1000/3600)
  total_distance / speed_train_ms

theorem train_crossing_time
  (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ)
  (h_length_train : length_train = 160)
  (h_speed_train_kmh : speed_train_kmh = 45)
  (h_length_bridge : length_bridge = 215) :
  time_to_cross_bridge length_train speed_train_kmh length_bridge = 30 :=
sorry

end train_crossing_time_l1095_109518


namespace number_of_multiples_of_3003_l1095_109530

theorem number_of_multiples_of_3003 (i j : ℕ) (h : 0 ≤ i ∧ i < j ∧ j ≤ 199): 
  (∃ n : ℕ, n = 3003 * k ∧ n = 10^j - 10^i) → 
  (number_of_solutions = 1568) :=
sorry

end number_of_multiples_of_3003_l1095_109530


namespace lines_intersection_l1095_109514

def intersection_point_of_lines
  (t u : ℚ)
  (x₁ y₁ x₂ y₂ : ℚ)
  (x y : ℚ) : Prop := 
  ∃ (t u : ℚ),
    (x₁ + 3*t = 7 + 6*u) ∧
    (y₁ - 4*t = -5 + 3*u) ∧
    (x = x₁ + 3 * t) ∧ 
    (y = y₁ - 4 * t)

theorem lines_intersection :
  ∀ (t u : ℚ),
    intersection_point_of_lines t u 3 2 7 (-5) (87/11) (-50/11) :=
by
  sorry

end lines_intersection_l1095_109514


namespace circle_area_irrational_if_rational_diameter_l1095_109588

noncomputable def pi : ℝ := Real.pi

theorem circle_area_irrational_if_rational_diameter (d : ℚ) :
  ¬ ∃ (A : ℝ), A = pi * (d / 2)^2 ∧ (∃ (q : ℚ), A = q) :=
by
  sorry

end circle_area_irrational_if_rational_diameter_l1095_109588


namespace minimum_value_l1095_109532

theorem minimum_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  (∃ (m : ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → m ≤ (y / x + 1 / y)) ∧
   m = 3 ∧ (∀ (x : ℝ), 0 < x → 0 < (1 - x) → (1 - x) + x = 1 → (y / x + 1 / y = m) ↔ x = 1 / 2)) :=
by
  sorry

end minimum_value_l1095_109532


namespace complete_residue_system_infinitely_many_positive_integers_l1095_109561

def is_complete_residue_system (n m : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → i ≠ j → (i^n % m ≠ j^n % m)

theorem complete_residue_system_infinitely_many_positive_integers (m : ℕ) (h_pos : 0 < m) :
  ∃ᶠ n in at_top, is_complete_residue_system n m :=
sorry

end complete_residue_system_infinitely_many_positive_integers_l1095_109561


namespace fuel_tank_capacity_l1095_109512

theorem fuel_tank_capacity (x : ℝ) 
  (h1 : (5 / 6) * x - (2 / 3) * x = 15) : x = 90 :=
sorry

end fuel_tank_capacity_l1095_109512


namespace valid_license_plates_count_l1095_109562

-- Defining the total number of choices for letters and digits
def num_letter_choices := 26
def num_digit_choices := 10

-- Function to calculate the total number of valid license plates
def total_license_plates := num_letter_choices ^ 3 * num_digit_choices ^ 4

-- The proof statement
theorem valid_license_plates_count : total_license_plates = 175760000 := 
by 
  -- The placeholder for the proof
  sorry

end valid_license_plates_count_l1095_109562


namespace part_a_part_b_part_c_part_d_l1095_109507

theorem part_a : (4237 * 27925 ≠ 118275855) :=
by sorry

theorem part_b : (42971064 / 8264 ≠ 5201) :=
by sorry

theorem part_c : (1965^2 ≠ 3761225) :=
by sorry

theorem part_d : (23 ^ 5 ≠ 371293) :=
by sorry

end part_a_part_b_part_c_part_d_l1095_109507


namespace remainder_when_four_times_number_minus_nine_divided_by_eight_l1095_109572

theorem remainder_when_four_times_number_minus_nine_divided_by_eight
  (n : ℤ) (h : n % 8 = 3) : (4 * n - 9) % 8 = 3 := by
  sorry

end remainder_when_four_times_number_minus_nine_divided_by_eight_l1095_109572


namespace solve_for_x_l1095_109542

theorem solve_for_x (x : ℝ) (h : (x * (x ^ (5 / 2))) ^ (1 / 4) = 4) : 
  x = 4 ^ (8 / 7) :=
sorry

end solve_for_x_l1095_109542


namespace translation_graph_pass_through_point_l1095_109597

theorem translation_graph_pass_through_point :
  (∃ a : ℝ, (∀ x y : ℝ, y = -2 * x + 1 - 3 → y = 3 → x = a) → a = -5/2) :=
sorry

end translation_graph_pass_through_point_l1095_109597


namespace unique_solution_3x_4y_5z_l1095_109502

theorem unique_solution_3x_4y_5z (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 ^ x + 4 ^ y = 5 ^ z → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  intro h
  sorry

end unique_solution_3x_4y_5z_l1095_109502


namespace escher_consecutive_probability_l1095_109577

open Classical

noncomputable def probability_Escher_consecutive (total_pieces escher_pieces: ℕ): ℚ :=
  if total_pieces < escher_pieces then 0 else (Nat.factorial (total_pieces - escher_pieces) * Nat.factorial escher_pieces) / Nat.factorial (total_pieces - 1)

theorem escher_consecutive_probability :
  probability_Escher_consecutive 12 4 = 1 / 41 :=
by
  sorry

end escher_consecutive_probability_l1095_109577


namespace uv_square_l1095_109505

theorem uv_square (u v : ℝ) (h1 : u * (u + v) = 50) (h2 : v * (u + v) = 100) : (u + v)^2 = 150 := by
  sorry

end uv_square_l1095_109505


namespace peter_has_142_nickels_l1095_109585

-- Define the conditions
def nickels (n : ℕ) : Prop :=
  40 < n ∧ n < 400 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 7 = 2

-- The theorem to prove the number of nickels
theorem peter_has_142_nickels : ∃ (n : ℕ), nickels n ∧ n = 142 :=
by {
  sorry
}

end peter_has_142_nickels_l1095_109585


namespace equal_work_women_l1095_109501

-- Let W be the amount of work one woman can do in a day.
-- Let M be the amount of work one man can do in a day.
-- Let x be the number of women who do the same amount of work as 5 men.

def numWomenEqualWork (W : ℝ) (M : ℝ) (x : ℝ) : Prop :=
  5 * M = x * W

theorem equal_work_women (W M x : ℝ) 
  (h1 : numWomenEqualWork W M x)
  (h2 : (3 * M + 5 * W) * 10 = (7 * W) * 14) :
  x = 8 :=
sorry

end equal_work_women_l1095_109501


namespace hotel_charge_percentage_l1095_109537

theorem hotel_charge_percentage (G R P : ℝ) 
  (hR : R = 1.60 * G) 
  (hP : P = 0.80 * G) : 
  ((R - P) / R) * 100 = 50 := by
  sorry

end hotel_charge_percentage_l1095_109537


namespace circle_touching_y_axis_radius_5_k_value_l1095_109533

theorem circle_touching_y_axis_radius_5_k_value :
  ∃ k : ℝ, ∀ x y : ℝ, (x^2 + 8 * x + y^2 + 4 * y - k = 0) →
    (∃ r : ℝ, r = 5 ∧ (∀ c : ℝ × ℝ, (c.1 + 4)^2 + (c.2 + 2)^2 = r^2) ∧
      (∃ x : ℝ, x + 4 = 0)) :=
by
  sorry

end circle_touching_y_axis_radius_5_k_value_l1095_109533


namespace ratio_cookies_to_pie_l1095_109554

def num_surveyed_students : ℕ := 800
def num_students_preferred_cookies : ℕ := 280
def num_students_preferred_pie : ℕ := 160

theorem ratio_cookies_to_pie : num_students_preferred_cookies / num_students_preferred_pie = 7 / 4 := by
  sorry

end ratio_cookies_to_pie_l1095_109554


namespace has_zero_in_intervals_l1095_109590

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x
noncomputable def f' (x : ℝ) : ℝ := (1 / 3) - (1 / x)

theorem has_zero_in_intervals : 
  (∃ x : ℝ, 0 < x ∧ x < 3 ∧ f x = 0) ∧ (∃ x : ℝ, 3 < x ∧ f x = 0) :=
sorry

end has_zero_in_intervals_l1095_109590


namespace garden_area_difference_l1095_109534

theorem garden_area_difference:
  (let length_rect := 60
   let width_rect := 20
   let perimeter_rect := 2 * (length_rect + width_rect)
   let side_square := perimeter_rect / 4
   let area_rect := length_rect * width_rect
   let area_square := side_square * side_square
   area_square - area_rect = 400) := 
by
  sorry

end garden_area_difference_l1095_109534


namespace lock_combination_l1095_109535

-- Define the digits as distinct
def distinct_digits (V E N U S I A R : ℕ) : Prop :=
  V ≠ E ∧ V ≠ N ∧ V ≠ U ∧ V ≠ S ∧ V ≠ I ∧ V ≠ A ∧ V ≠ R ∧
  E ≠ N ∧ E ≠ U ∧ E ≠ S ∧ E ≠ I ∧ E ≠ A ∧ E ≠ R ∧
  N ≠ U ∧ N ≠ S ∧ N ≠ I ∧ N ≠ A ∧ N ≠ R ∧
  U ≠ S ∧ U ≠ I ∧ U ≠ A ∧ U ≠ R ∧
  S ≠ I ∧ S ≠ A ∧ S ≠ R ∧
  I ≠ A ∧ I ≠ R ∧
  A ≠ R

-- Define the base 12 addition for the equation
def base12_addition (V E N U S I A R : ℕ) : Prop :=
  let VENUS := V * 12^4 + E * 12^3 + N * 12^2 + U * 12^1 + S
  let IS := I * 12^1 + S
  let NEAR := N * 12^3 + E * 12^2 + A * 12^1 + R
  let SUN := S * 12^2 + U * 12^1 + N
  VENUS + IS + NEAR = SUN

-- The theorem statement
theorem lock_combination :
  ∃ (V E N U S I A R : ℕ),
    distinct_digits V E N U S I A R ∧
    base12_addition V E N U S I A R ∧
    (S * 12^2 + U * 12^1 + N) = 655 := 
sorry

end lock_combination_l1095_109535


namespace problem_l1095_109587

-- Helper definition for point on a line
def point_on_line (x y : ℝ) (a b : ℝ) : Prop := y = a * x + b

-- Given condition: Point P(1, 3) lies on the line y = 2x + b
def P_on_l (b : ℝ) : Prop := point_on_line 1 3 2 b

-- The proof problem: Proving (2, 5) also lies on the line y = 2x + b where b is the constant found using P
theorem problem (b : ℝ) (h: P_on_l b) : point_on_line 2 5 2 b :=
by
  sorry

end problem_l1095_109587


namespace total_candies_is_90_l1095_109523

-- Defining the conditions
def boxes_chocolate := 6
def boxes_caramel := 4
def pieces_per_box := 9

-- Defining the total number of boxes
def total_boxes := boxes_chocolate + boxes_caramel

-- Defining the total number of candies
def total_candies := total_boxes * pieces_per_box

-- Theorem stating the proof problem
theorem total_candies_is_90 : total_candies = 90 := by
  -- Provide a placeholder for the proof
  sorry

end total_candies_is_90_l1095_109523


namespace range_of_b_no_common_points_l1095_109522

theorem range_of_b_no_common_points (b : ℝ) :
  ¬ (∃ x : ℝ, 2 ^ |x| - 1 = b) ↔ b < 0 :=
by
  sorry

end range_of_b_no_common_points_l1095_109522


namespace colored_ints_square_diff_l1095_109596

-- Define a coloring function c as a total function from ℤ to a finite set {0, 1, 2}
def c : ℤ → Fin 3 := sorry

-- Lean 4 statement for the problem
theorem colored_ints_square_diff : 
  ∃ a b : ℤ, a ≠ b ∧ c a = c b ∧ ∃ k : ℤ, a - b = k ^ 2 :=
sorry

end colored_ints_square_diff_l1095_109596


namespace probability_of_same_type_l1095_109555

-- Definitions for the given conditions
def total_books : ℕ := 12 + 9
def novels : ℕ := 12
def biographies : ℕ := 9

-- Define the number of ways to pick any two books
def total_ways_to_pick_two_books : ℕ := Nat.choose total_books 2

-- Define the number of ways to pick two novels
def ways_to_pick_two_novels : ℕ := Nat.choose novels 2

-- Define the number of ways to pick two biographies
def ways_to_pick_two_biographies : ℕ := Nat.choose biographies 2

-- Define the number of ways to pick two books of the same type
def ways_to_pick_two_books_of_same_type : ℕ := ways_to_pick_two_novels + ways_to_pick_two_biographies

-- Calculate the probability
noncomputable def probability_same_type (total_ways ways_same_type : ℕ) : ℚ :=
  ways_same_type / total_ways

theorem probability_of_same_type :
  probability_same_type total_ways_to_pick_two_books ways_to_pick_two_books_of_same_type = 17 / 35 := by
  sorry

end probability_of_same_type_l1095_109555


namespace greatest_value_x_l1095_109592

theorem greatest_value_x (x : ℝ) : 
  (x ≠ 9) → 
  (x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 6) →
  x ≤ -2 :=
by
  sorry

end greatest_value_x_l1095_109592


namespace function_decreasing_range_k_l1095_109564

theorem function_decreasing_range_k : 
  ∀ k : ℝ, (∀ x : ℝ, 1 ≤ x → ∀ y : ℝ, 1 ≤ y → x ≤ y → (k * x ^ 2 + (3 * k - 2) * x - 5) ≥ (k * y ^ 2 + (3 * k - 2) * y - 5)) ↔ (k ∈ Set.Iic 0) :=
by sorry

end function_decreasing_range_k_l1095_109564


namespace product_of_ratios_l1095_109593

theorem product_of_ratios 
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (hx1 : x1^3 - 3 * x1 * y1^2 = 2005)
  (hy1 : y1^3 - 3 * x1^2 * y1 = 2004)
  (hx2 : x2^3 - 3 * x2 * y2^2 = 2005)
  (hy2 : y2^3 - 3 * x2^2 * y2 = 2004)
  (hx3 : x3^3 - 3 * x3 * y3^2 = 2005)
  (hy3 : y3^3 - 3 * x3^2 * y3 = 2004) :
  (1 - x1/y1) * (1 - x2/y2) * (1 - x3/y3) = 1/1002 := 
sorry

end product_of_ratios_l1095_109593


namespace factor_quadratic_l1095_109568

theorem factor_quadratic (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := 
by 
  sorry

end factor_quadratic_l1095_109568


namespace number_of_teams_l1095_109513

-- Define the problem context
variables (n : ℕ)

-- Define the conditions
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The theorem we want to prove
theorem number_of_teams (h : total_games n = 55) : n = 11 :=
sorry

end number_of_teams_l1095_109513


namespace find_c_l1095_109580

-- Let a, b, c, d, and e be positive consecutive integers.
variables {a b c d e : ℕ}

-- Conditions: 
def conditions (a b c d e : ℕ) : Prop :=
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a + b = e - 1 ∧
  a * b = d + 1

-- Proof statement
theorem find_c (h : conditions a b c d e) : c = 4 :=
by sorry

end find_c_l1095_109580


namespace pentagon_perimeter_l1095_109525

-- Define the side length and number of sides for a regular pentagon
def side_length : ℝ := 5
def num_sides : ℕ := 5

-- Define the perimeter calculation as a constant
def perimeter (side_length : ℝ) (num_sides : ℕ) : ℝ := side_length * num_sides

theorem pentagon_perimeter : perimeter side_length num_sides = 25 := by
  sorry

end pentagon_perimeter_l1095_109525


namespace solve_for_q_l1095_109582

variable (k h q : ℝ)

-- Conditions given in the problem
axiom cond1 : (3 / 4) = (k / 48)
axiom cond2 : (3 / 4) = ((h + 36) / 60)
axiom cond3 : (3 / 4) = ((q - 9) / 80)

-- Our goal is to state that q = 69
theorem solve_for_q : q = 69 :=
by
  -- the proof goes here
  sorry

end solve_for_q_l1095_109582


namespace task_completion_time_l1095_109529

noncomputable def john_work_rate := (1: ℚ) / 20
noncomputable def jane_work_rate := (1: ℚ) / 12
noncomputable def combined_work_rate := john_work_rate + jane_work_rate
noncomputable def time_jane_disposed := 4

theorem task_completion_time :
  (∃ x : ℚ, (combined_work_rate * x + john_work_rate * time_jane_disposed = 1) ∧ (x + time_jane_disposed = 10)) :=
by
  use 6  
  sorry

end task_completion_time_l1095_109529


namespace youngest_brother_age_l1095_109539

theorem youngest_brother_age 
  (x : ℤ) 
  (h1 : ∃ (a b c : ℤ), a = x ∧ b = x + 1 ∧ c = x + 2 ∧ a + b + c = 96) : 
  x = 31 :=
by sorry

end youngest_brother_age_l1095_109539


namespace sum_of_three_fractions_is_one_l1095_109599

theorem sum_of_three_fractions_is_one (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ) = 1 ↔ 
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 6) ∨ 
  (a = 3 ∧ b = 6 ∧ c = 2) :=
by sorry

end sum_of_three_fractions_is_one_l1095_109599


namespace negation_of_p_l1095_109536

def proposition_p := ∃ x : ℝ, x ≥ 1 ∧ x^2 - x < 0

theorem negation_of_p : (∀ x : ℝ, x ≥ 1 → x^2 - x ≥ 0) :=
by
  sorry

end negation_of_p_l1095_109536


namespace tournament_games_count_l1095_109531

-- We define the conditions
def number_of_players : ℕ := 6

-- Function to calculate the number of games played in a tournament where each player plays twice with each opponent
def total_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- Now we state the theorem
theorem tournament_games_count : total_games number_of_players = 60 := by
  -- Proof goes here
  sorry

end tournament_games_count_l1095_109531


namespace f_45_g_10_l1095_109576

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_condition1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom g_condition2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x + y) = g x + g y
axiom f_15 : f 15 = 10
axiom g_5 : g 5 = 3

theorem f_45 : f 45 = 10 / 3 := sorry
theorem g_10 : g 10 = 6 := sorry

end f_45_g_10_l1095_109576


namespace number_of_triplets_l1095_109558

theorem number_of_triplets (N : ℕ) (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 2017 ≥ 10 * a) (h5 : 10 * a ≥ 100 * b) (h6 : 100 * b ≥ 1000 * c) : 
  N = 574 := 
sorry

end number_of_triplets_l1095_109558


namespace number_divisibility_l1095_109574

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem number_divisibility (n : ℕ) :
  (3^n ∣ A_n n) ∧ ¬ (3^(n + 1) ∣ A_n n) := by
  sorry

end number_divisibility_l1095_109574


namespace odd_function_def_l1095_109595

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x * (x - 1)
else -x * (x + 1)

theorem odd_function_def {x : ℝ} (h : x > 0) :
  f x = -x * (x + 1) :=
by
  sorry

end odd_function_def_l1095_109595


namespace photographer_choice_l1095_109546

theorem photographer_choice : 
  (Nat.choose 7 4) + (Nat.choose 7 5) = 56 := 
by 
  sorry

end photographer_choice_l1095_109546


namespace sum_of_ages_five_years_from_now_l1095_109573

noncomputable def viggo_age_when_brother_was_2 (brother_age: ℕ) : ℕ :=
  10 + 2 * brother_age

noncomputable def current_viggo_age (viggo_age_at_2: ℕ) (current_brother_age: ℕ) : ℕ :=
  viggo_age_at_2 + (current_brother_age - 2)

def sister_age (viggo_age: ℕ) : ℕ :=
  viggo_age + 5

noncomputable def cousin_age (viggo_age: ℕ) (brother_age: ℕ) (sister_age: ℕ) : ℕ :=
  ((viggo_age + brother_age + sister_age) / 3)

noncomputable def future_ages_sum (viggo_age: ℕ) (brother_age: ℕ) (sister_age: ℕ) (cousin_age: ℕ) : ℕ :=
  viggo_age + 5 + brother_age + 5 + sister_age + 5 + cousin_age + 5

theorem sum_of_ages_five_years_from_now :
  let current_brother_age := 10
  let viggo_age_at_2 := viggo_age_when_brother_was_2 2
  let current_viggo_age := current_viggo_age viggo_age_at_2 current_brother_age
  let current_sister_age := sister_age current_viggo_age
  let current_cousin_age := cousin_age current_viggo_age current_brother_age current_sister_age
  future_ages_sum current_viggo_age current_brother_age current_sister_age current_cousin_age = 99 := sorry

end sum_of_ages_five_years_from_now_l1095_109573


namespace remainder_of_7_pow_51_mod_8_l1095_109548

theorem remainder_of_7_pow_51_mod_8 : (7^51 % 8) = 7 := sorry

end remainder_of_7_pow_51_mod_8_l1095_109548


namespace frac_mul_square_l1095_109541

theorem frac_mul_square 
  : (8/9)^2 * (1/3)^2 = 64/729 := 
by 
  sorry

end frac_mul_square_l1095_109541


namespace children_ticket_price_difference_l1095_109517

noncomputable def regular_ticket_price : ℝ := 9
noncomputable def total_amount_given : ℝ := 2 * 20
noncomputable def total_change_received : ℝ := 1
noncomputable def num_adults : ℕ := 2
noncomputable def num_children : ℕ := 3
noncomputable def total_cost_of_tickets : ℝ := total_amount_given - total_change_received
noncomputable def children_ticket_cost := (total_cost_of_tickets - num_adults * regular_ticket_price) / num_children

theorem children_ticket_price_difference :
  (regular_ticket_price - children_ticket_cost) = 2 := by
  sorry

end children_ticket_price_difference_l1095_109517


namespace man_age_twice_son_age_in_2_years_l1095_109506

variable (currentAgeSon : ℕ)
variable (currentAgeMan : ℕ)
variable (Y : ℕ)

-- Given conditions
def sonCurrentAge : Prop := currentAgeSon = 23
def manCurrentAge : Prop := currentAgeMan = currentAgeSon + 25
def manAgeTwiceSonAgeInYYears : Prop := currentAgeMan + Y = 2 * (currentAgeSon + Y)

-- Theorem to prove
theorem man_age_twice_son_age_in_2_years :
  sonCurrentAge currentAgeSon →
  manCurrentAge currentAgeSon currentAgeMan →
  manAgeTwiceSonAgeInYYears currentAgeSon currentAgeMan Y →
  Y = 2 :=
by
  intros h_son_age h_man_age h_age_relation
  sorry

end man_age_twice_son_age_in_2_years_l1095_109506


namespace shaded_area_l1095_109526

/-- Prove that the shaded area of a shape formed by removing four right triangles of legs 2 from each corner of a 6 × 6 square is equal to 28 square units -/
theorem shaded_area (a b c d : ℕ) (square_side_length : ℕ) (triangle_leg_length : ℕ)
  (h1 : square_side_length = 6)
  (h2 : triangle_leg_length = 2)
  (h3 : a = 1)
  (h4 : b = 2)
  (h5 : c = b)
  (h6 : d = 4*a) : 
  a * square_side_length * square_side_length - d * (b * b / 2) = 28 := 
sorry

end shaded_area_l1095_109526


namespace daughter_age_in_3_years_l1095_109549

theorem daughter_age_in_3_years (mother_age_now : ℕ) (h1 : mother_age_now = 41)
  (h2 : ∃ daughter_age_5_years_ago : ℕ, mother_age_now - 5 = 2 * daughter_age_5_years_ago) :
  ∃ daughter_age_in_3_years : ℕ, daughter_age_in_3_years = 26 :=
by {
  sorry
}

end daughter_age_in_3_years_l1095_109549


namespace Annika_hiking_rate_is_correct_l1095_109565

def AnnikaHikingRate
  (distance_partial_east distance_total_east : ℕ)
  (time_back_to_start : ℕ)
  (equality_rate : Nat) : Prop :=
  distance_partial_east = 2750 / 1000 ∧
  distance_total_east = 3500 / 1000 ∧
  time_back_to_start = 51 ∧
  equality_rate = 34

theorem Annika_hiking_rate_is_correct :
  ∃ R : ℕ, ∀ d1 d2 t,
  AnnikaHikingRate d1 d2 t R → R = 34 :=
by
  sorry

end Annika_hiking_rate_is_correct_l1095_109565


namespace find_a_b_minimum_value_l1095_109511

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x^2

/-- Given the function y = f(x) = ax^3 + bx^2, when x = 1, it has a maximum value of 3 -/
def condition1 (a b : ℝ) : Prop :=
  f 1 a b = 3 ∧ (3 * a + 2 * b = 0)

/-- Find the values of the real numbers a and b -/
theorem find_a_b : ∃ (a b : ℝ), condition1 a b :=
sorry

/-- Find the minimum value of the function -/
theorem minimum_value : ∀ (a b : ℝ), condition1 a b → (∃ x_min, ∀ x, f x a b ≥ f x_min a b) :=
sorry

end find_a_b_minimum_value_l1095_109511


namespace loss_percentage_remaining_stock_l1095_109519

noncomputable def total_worth : ℝ := 9999.999999999998
def overall_loss : ℝ := 200
def profit_percentage_20 : ℝ := 0.1
def sold_20_percentage : ℝ := 0.2
def remaining_percentage : ℝ := 0.8

theorem loss_percentage_remaining_stock :
  ∃ L : ℝ, 0.8 * total_worth * (L / 100) - 0.02 * total_worth = overall_loss ∧ L = 5 :=
by sorry

end loss_percentage_remaining_stock_l1095_109519


namespace angle_sum_property_l1095_109540

theorem angle_sum_property 
  (P Q R S : Type) 
  (alpha beta : ℝ)
  (h1 : alpha = 3 * x)
  (h2 : beta = 2 * x)
  (h3 : alpha + beta = 90) :
  x = 18 :=
by
  sorry

end angle_sum_property_l1095_109540


namespace f_inequality_l1095_109567

-- Define the function f.
def f (x : ℝ) : ℝ := x^2 - x + 13

-- The main theorem to prove the given inequality.
theorem f_inequality (x m : ℝ) (h : |x - m| < 1) : |f x - f m| < 2*(|m| + 1) :=
by
  sorry

end f_inequality_l1095_109567


namespace range_of_t_l1095_109524

variable (f : ℝ → ℝ) (t : ℝ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_t {f : ℝ → ℝ} {t : ℝ} 
  (Hodd : is_odd f) 
  (Hperiodic : ∀ x, f (x + 5 / 2) = -1 / f x) 
  (Hf1 : f 1 ≥ 1) 
  (Hf2014 : f 2014 = (t + 3) / (t - 3)) : 
  0 ≤ t ∧ t < 3 := by
  sorry

end range_of_t_l1095_109524


namespace fraction_subtraction_identity_l1095_109551

theorem fraction_subtraction_identity (x y : ℕ) (hx : x = 3) (hy : y = 4) : (1 / (x : ℚ) - 1 / (y : ℚ) = 1 / 12) :=
by
  sorry

end fraction_subtraction_identity_l1095_109551


namespace least_possible_k_l1095_109571

-- Define the conditions
def prime_factor_form (k : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ k = 2^a * 3^b * 5^c

def divisible_by_1680 (k : ℕ) : Prop :=
  (k ^ 4) % 1680 = 0

-- Define the proof problem
theorem least_possible_k (k : ℕ) (h_div : divisible_by_1680 k) (h_prime : prime_factor_form k) : k = 210 :=
by
  -- Statement of the problem, proof to be filled
  sorry

end least_possible_k_l1095_109571


namespace arlene_hike_distance_l1095_109520

-- Define the conditions: Arlene's pace and the time she spent hiking
def arlene_pace : ℝ := 4 -- miles per hour
def arlene_time_hiking : ℝ := 6 -- hours

-- Define the problem statement and provide the mathematical proof
theorem arlene_hike_distance :
  arlene_pace * arlene_time_hiking = 24 :=
by
  -- This is where the proof would go
  sorry

end arlene_hike_distance_l1095_109520


namespace find_value_perpendicular_distances_l1095_109550

variable {R a b c D E F : ℝ}
variable {ABC : Triangle}

-- Assume the distances from point P on the circumcircle of triangle ABC
-- to the sides BC, CA, and AB respectively.
axiom D_def : D = R * a / (2 * R)
axiom E_def : E = R * b / (2 * R)
axiom F_def : F = R * c / (2 * R)

theorem find_value_perpendicular_distances
    (a b c R : ℝ) (D E F : ℝ) 
    (hD : D = R * a / (2 * R)) 
    (hE : E = R * b / (2 * R)) 
    (hF : F = R * c / (2 * R)) : 
    a^2 * D^2 + b^2 * E^2 + c^2 * F^2 = (a^4 + b^4 + c^4) / (4 * R^2) :=
by
  sorry

end find_value_perpendicular_distances_l1095_109550


namespace solve_for_q_l1095_109578

-- Define the conditions
variables (p q : ℝ)
axiom condition1 : 3 * p + 4 * q = 8
axiom condition2 : 4 * p + 3 * q = 13

-- State the goal to prove q = -1
theorem solve_for_q : q = -1 :=
by
  sorry

end solve_for_q_l1095_109578


namespace smallest_M_bound_l1095_109584

theorem smallest_M_bound {f : ℕ → ℝ} (hf1 : f 1 = 2) 
  (hf2 : ∀ n : ℕ, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1)) * f (2 * n)) : 
  ∃ M : ℕ, (∀ n : ℕ, f n < M) ∧ M = 10 :=
by
  sorry

end smallest_M_bound_l1095_109584


namespace change_positions_of_three_out_of_eight_l1095_109515

theorem change_positions_of_three_out_of_eight :
  (Nat.choose 8 3) * (Nat.factorial 3) = (Nat.choose 8 3) * 6 :=
by
  sorry

end change_positions_of_three_out_of_eight_l1095_109515


namespace parabola_coefficients_l1095_109510

theorem parabola_coefficients (a b c : ℝ) 
  (h_vertex : ∀ x, a * (x - 4) * (x - 4) + 3 = a * x * x + b * x + c) 
  (h_pass_point : 1 = a * (2 - 4) * (2 - 4) + 3) :
  (a = -1/2) ∧ (b = 4) ∧ (c = -5) :=
by
  sorry

end parabola_coefficients_l1095_109510


namespace carla_marbles_start_l1095_109547

-- Conditions defined as constants
def marblesBought : ℝ := 489.0
def marblesTotalNow : ℝ := 2778.0

-- Theorem statement
theorem carla_marbles_start (marblesBought marblesTotalNow: ℝ) :
  marblesTotalNow - marblesBought = 2289.0 := by
  sorry

end carla_marbles_start_l1095_109547


namespace apples_per_sandwich_l1095_109528

-- Define the conditions
def sam_sandwiches_per_day : Nat := 10
def days_in_week : Nat := 7
def total_apples_in_week : Nat := 280

-- Calculate total sandwiches in a week
def total_sandwiches_in_week := sam_sandwiches_per_day * days_in_week

-- Prove that Sam eats 4 apples for each sandwich
theorem apples_per_sandwich : total_apples_in_week / total_sandwiches_in_week = 4 :=
  by
    sorry

end apples_per_sandwich_l1095_109528


namespace calc_hash_2_5_3_l1095_109521

def operation_hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem calc_hash_2_5_3 : operation_hash 2 5 3 = 1 := by
  sorry

end calc_hash_2_5_3_l1095_109521
