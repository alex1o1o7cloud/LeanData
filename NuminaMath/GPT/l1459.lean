import Mathlib

namespace NUMINAMATH_GPT_cricket_bat_selling_price_l1459_145959

theorem cricket_bat_selling_price
    (profit : ℝ)
    (profit_percentage : ℝ)
    (CP : ℝ)
    (SP : ℝ)
    (h_profit : profit = 255)
    (h_profit_percentage : profit_percentage = 42.857142857142854)
    (h_CP : CP = 255 * 100 / 42.857142857142854)
    (h_SP : SP = CP + profit) :
    SP = 850 :=
by
  skip -- This is where the proof would go
  sorry -- Placeholder for the required proof

end NUMINAMATH_GPT_cricket_bat_selling_price_l1459_145959


namespace NUMINAMATH_GPT_essay_count_problem_l1459_145975

noncomputable def eighth_essays : ℕ := sorry
noncomputable def seventh_essays : ℕ := sorry

theorem essay_count_problem (x : ℕ) (h1 : eighth_essays = x) (h2 : seventh_essays = (1/2 : ℚ) * x - 2) (h3 : eighth_essays + seventh_essays = 118) : 
  seventh_essays = 38 :=
sorry

end NUMINAMATH_GPT_essay_count_problem_l1459_145975


namespace NUMINAMATH_GPT_translate_graph_upwards_l1459_145993

theorem translate_graph_upwards (x : ℝ) :
  (∀ x, (3*x - 1) + 3 = 3*x + 2) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_translate_graph_upwards_l1459_145993


namespace NUMINAMATH_GPT_increasing_power_function_l1459_145915

theorem increasing_power_function (m : ℝ) (h_power : m^2 - 1 = 1)
    (h_increasing : ∀ x : ℝ, x > 0 → (m^2 - 1) * m * x^(m-1) > 0) : m = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_increasing_power_function_l1459_145915


namespace NUMINAMATH_GPT_intersection_is_correct_l1459_145974

def setA : Set ℕ := {0, 1, 2}
def setB : Set ℕ := {1, 2, 3}

theorem intersection_is_correct : setA ∩ setB = {1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l1459_145974


namespace NUMINAMATH_GPT_trapezoid_shorter_base_l1459_145965

theorem trapezoid_shorter_base (L : ℝ) (S : ℝ) (m : ℝ)
  (hL : L = 100)
  (hm : m = 4)
  (h : m = (L - S) / 2) :
  S = 92 :=
by {
  sorry -- Proof is not required
}

end NUMINAMATH_GPT_trapezoid_shorter_base_l1459_145965


namespace NUMINAMATH_GPT_two_f_of_x_l1459_145917

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + x)

theorem two_f_of_x (x : ℝ) (h : x > 0) : 2 * f x = 18 / (9 + x) :=
  sorry

end NUMINAMATH_GPT_two_f_of_x_l1459_145917


namespace NUMINAMATH_GPT_problem_1_problem_2_l1459_145953

noncomputable def f (ω x : ℝ) : ℝ :=
  Real.sin (ω * x + (Real.pi / 4))

theorem problem_1 (ω : ℝ) (hω : ω > 0) : f ω 0 = Real.sqrt 2 / 2 :=
by
  unfold f
  simp [Real.sin_pi_div_four]

theorem problem_2 : 
  ∃ x : ℝ, 
  0 ≤ x ∧ x ≤ Real.pi / 2 ∧ 
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.pi / 2 → f 2 y ≤ f 2 x) ∧ 
  f 2 x = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1459_145953


namespace NUMINAMATH_GPT_intersection_of_sets_l1459_145986

theorem intersection_of_sets (M : Set ℤ) (N : Set ℤ) (H_M : M = {0, 1, 2, 3, 4}) (H_N : N = {-2, 0, 2}) :
  M ∩ N = {0, 2} :=
by
  rw [H_M, H_N]
  ext
  simp
  sorry  -- Proof to be filled in

end NUMINAMATH_GPT_intersection_of_sets_l1459_145986


namespace NUMINAMATH_GPT_y_intercept_l1459_145922

theorem y_intercept (x y : ℝ) (h : 2 * x - 3 * y = 6) : x = 0 → y = -2 :=
by
  intro h₁
  sorry

end NUMINAMATH_GPT_y_intercept_l1459_145922


namespace NUMINAMATH_GPT_curve_to_polar_l1459_145926

noncomputable def polar_eq_of_curve (x y : ℝ) (ρ θ : ℝ) : Prop :=
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ (x ^ 2 + y ^ 2 - 2 * x = 0) → (ρ = 2 * Real.cos θ)

theorem curve_to_polar (x y ρ θ : ℝ) :
  polar_eq_of_curve x y ρ θ :=
sorry

end NUMINAMATH_GPT_curve_to_polar_l1459_145926


namespace NUMINAMATH_GPT_eval_fraction_l1459_145908

theorem eval_fraction : (144 : ℕ) = 12 * 12 → (12 ^ 10 / (144 ^ 4) : ℝ) = 144 := by
  intro h
  have h1 : (144 : ℕ) = 12 ^ 2 := by
    exact h
  sorry

end NUMINAMATH_GPT_eval_fraction_l1459_145908


namespace NUMINAMATH_GPT_relay_race_time_reduction_l1459_145998

theorem relay_race_time_reduction
    (T T1 T2 T3 T4 T5 : ℝ)
    (h1 : T1 = 0.1 * T)
    (h2 : T2 = 0.2 * T)
    (h3 : T3 = 0.24 * T)
    (h4 : T4 = 0.3 * T)
    (h5 : T5 = 0.16 * T) :
    ((T1 + T2 + T3 + T4 + T5) - (T1 + T2 + T3 + T4 + T5 / 2)) / (T1 + T2 + T3 + T4 + T5) = 0.08 :=
by
  sorry

end NUMINAMATH_GPT_relay_race_time_reduction_l1459_145998


namespace NUMINAMATH_GPT_line_parallel_l1459_145906

theorem line_parallel (a : ℝ) : (∀ x y : ℝ, ax + y = 0) ↔ (x + ay + 1 = 0) → a = 1 ∨ a = -1 := 
sorry

end NUMINAMATH_GPT_line_parallel_l1459_145906


namespace NUMINAMATH_GPT_cost_of_flute_l1459_145927

def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := 8.89
def song_book_cost : ℝ := 7
def flute_cost : ℝ := 142.46

theorem cost_of_flute :
  total_spent - (music_stand_cost + song_book_cost) = flute_cost :=
by
  sorry

end NUMINAMATH_GPT_cost_of_flute_l1459_145927


namespace NUMINAMATH_GPT_largest_even_integer_of_product_2880_l1459_145909

theorem largest_even_integer_of_product_2880 :
  ∃ n : ℤ, (n-2) * n * (n+2) = 2880 ∧ n + 2 = 22 := 
by {
  sorry
}

end NUMINAMATH_GPT_largest_even_integer_of_product_2880_l1459_145909


namespace NUMINAMATH_GPT_range_is_fixed_points_l1459_145932

variable (f : ℕ → ℕ)

axiom functional_eq : ∀ m n, f (m + f n) = f (f m) + f n

theorem range_is_fixed_points :
  {n : ℕ | ∃ m : ℕ, f m = n} = {n : ℕ | f n = n} :=
sorry

end NUMINAMATH_GPT_range_is_fixed_points_l1459_145932


namespace NUMINAMATH_GPT_set_operation_correct_l1459_145911

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

-- Define the operation A * B
def set_operation (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem to be proved
theorem set_operation_correct : set_operation A B = {1, 3} :=
sorry

end NUMINAMATH_GPT_set_operation_correct_l1459_145911


namespace NUMINAMATH_GPT_charging_time_is_correct_l1459_145921

-- Lean definitions for the given conditions
def smartphone_charge_time : ℕ := 26
def tablet_charge_time : ℕ := 53
def phone_half_charge_time : ℕ := smartphone_charge_time / 2

-- Definition for the total charging time based on conditions
def total_charging_time : ℕ :=
  tablet_charge_time + phone_half_charge_time

-- Proof problem statement
theorem charging_time_is_correct : total_charging_time = 66 := by
  sorry

end NUMINAMATH_GPT_charging_time_is_correct_l1459_145921


namespace NUMINAMATH_GPT_white_washing_cost_l1459_145964

theorem white_washing_cost
    (length width height : ℝ)
    (door_width door_height window_width window_height : ℝ)
    (num_doors num_windows : ℝ)
    (paint_cost : ℝ)
    (extra_paint_fraction : ℝ)
    (perimeter := 2 * (length + width))
    (door_area := num_doors * (door_width * door_height))
    (window_area := num_windows * (window_width * window_height))
    (wall_area := perimeter * height)
    (paint_area := wall_area - door_area - window_area)
    (total_area := paint_area * (1 + extra_paint_fraction))
    : total_area * paint_cost = 6652.8 :=
by sorry

end NUMINAMATH_GPT_white_washing_cost_l1459_145964


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1459_145924

theorem solution_set_of_inequality :
  {x : ℝ | 2 * x^2 - 3 * x - 2 > 0} = {x : ℝ | x < -0.5 ∨ x > 2} := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1459_145924


namespace NUMINAMATH_GPT_vehicle_combinations_count_l1459_145920

theorem vehicle_combinations_count :
  ∃ (x y : ℕ), (4 * x + y = 79) ∧ (∃ (n : ℕ), n = 19) :=
sorry

end NUMINAMATH_GPT_vehicle_combinations_count_l1459_145920


namespace NUMINAMATH_GPT_basketball_player_ft_rate_l1459_145916

theorem basketball_player_ft_rate :
  ∃ P : ℝ, 1 - P^2 = 16 / 25 ∧ P = 3 / 5 := sorry

end NUMINAMATH_GPT_basketball_player_ft_rate_l1459_145916


namespace NUMINAMATH_GPT_choose_stick_l1459_145901

-- Define the lengths of the sticks Xiaoming has
def xm_stick1 : ℝ := 4
def xm_stick2 : ℝ := 7

-- Define the lengths of the sticks Xiaohong has
def stick2 : ℝ := 2
def stick3 : ℝ := 3
def stick8 : ℝ := 8
def stick12 : ℝ := 12

-- Define the condition for a valid stick choice from Xiaohong's sticks
def valid_stick (x : ℝ) : Prop := 3 < x ∧ x < 11

-- State the problem as a theorem to be proved
theorem choose_stick : valid_stick stick8 := by
  sorry

end NUMINAMATH_GPT_choose_stick_l1459_145901


namespace NUMINAMATH_GPT_inequality_sqrt_ab_l1459_145982

theorem inequality_sqrt_ab {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧ (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := 
sorry

end NUMINAMATH_GPT_inequality_sqrt_ab_l1459_145982


namespace NUMINAMATH_GPT_quadratic_relationship_l1459_145996

variable (y_1 y_2 y_3 : ℝ)

-- Conditions
def vertex := (-2, 1)
def opens_downwards := true
def intersects_x_axis_at_two_points := true
def passes_through_points := [(1, y_1), (-1, y_2), (-4, y_3)]

-- Proof statement
theorem quadratic_relationship : y_1 < y_3 ∧ y_3 < y_2 :=
  sorry

end NUMINAMATH_GPT_quadratic_relationship_l1459_145996


namespace NUMINAMATH_GPT_determine_angle_A_l1459_145937

noncomputable section

open Real

-- Definition of an acute triangle and its sides
variables {A B : ℝ} {a b : ℝ}

-- Additional conditions that are given before providing the theorem
variables (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
          (h5 : 2 * a * sin B = sqrt 3 * b)

-- Theorem statement
theorem determine_angle_A (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
  (h5 : 2 * a * sin B = sqrt 3 * b) : A = π / 3 :=
sorry

end NUMINAMATH_GPT_determine_angle_A_l1459_145937


namespace NUMINAMATH_GPT_largest_number_among_selected_students_l1459_145951

def total_students := 80

def smallest_numbers (x y : ℕ) : Prop :=
  x = 6 ∧ y = 14

noncomputable def selected_students (n : ℕ) : ℕ :=
  6 + (n - 1) * 8

theorem largest_number_among_selected_students :
  ∀ (x y : ℕ), smallest_numbers x y → (selected_students 10 = 78) :=
by
  intros x y h
  rw [smallest_numbers] at h
  have h1 : x = 6 := h.1
  have h2 : y = 14 := h.2
  exact rfl

#check largest_number_among_selected_students

end NUMINAMATH_GPT_largest_number_among_selected_students_l1459_145951


namespace NUMINAMATH_GPT_fraction_absent_l1459_145904

theorem fraction_absent (p : ℕ) (x : ℚ) (h : (W / p) * 1.2 = W / (p * (1 - x))) : x = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_absent_l1459_145904


namespace NUMINAMATH_GPT_coin_difference_l1459_145945

/-- 
  Given that Paul has 5-cent, 20-cent, and 15-cent coins, 
  prove that the difference between the maximum and minimum number of coins
  needed to make exactly 50 cents is 6.
-/
theorem coin_difference :
  ∃ (coins : Nat → Nat),
    (coins 5 + coins 20 + coins 15) = 6 ∧
    (5 * coins 5 + 20 * coins 20 + 15 * coins 15 = 50) :=
sorry

end NUMINAMATH_GPT_coin_difference_l1459_145945


namespace NUMINAMATH_GPT_least_amount_of_money_l1459_145942

variable (money : String → ℝ)
variable (Bo Coe Flo Jo Moe Zoe : String)

theorem least_amount_of_money :
  (money Bo ≠ money Coe) ∧ (money Bo ≠ money Flo) ∧ (money Bo ≠ money Jo) ∧ (money Bo ≠ money Moe) ∧ (money Bo ≠ money Zoe) ∧ 
  (money Coe ≠ money Flo) ∧ (money Coe ≠ money Jo) ∧ (money Coe ≠ money Moe) ∧ (money Coe ≠ money Zoe) ∧ 
  (money Flo ≠ money Jo) ∧ (money Flo ≠ money Moe) ∧ (money Flo ≠ money Zoe) ∧ 
  (money Jo ≠ money Moe) ∧ (money Jo ≠ money Zoe) ∧ 
  (money Moe ≠ money Zoe) ∧ 
  (money Flo > money Jo) ∧ (money Flo > money Bo) ∧
  (money Bo > money Moe) ∧ (money Coe > money Moe) ∧ 
  (money Jo > money Moe) ∧ (money Jo < money Bo) ∧ 
  (money Zoe > money Jo) ∧ (money Zoe < money Coe) →
  money Moe < money Bo ∧ money Moe < money Coe ∧ money Moe < money Flo ∧ money Moe < money Jo ∧ money Moe < money Zoe := 
sorry

end NUMINAMATH_GPT_least_amount_of_money_l1459_145942


namespace NUMINAMATH_GPT_decagon_diagonal_intersection_probability_l1459_145999

def probability_intersect_within_decagon : ℚ :=
  let total_vertices := 10
  let total_pairs_points := Nat.choose total_vertices 2
  let total_diagonals := total_pairs_points - total_vertices
  let ways_to_pick_2_diagonals := Nat.choose total_diagonals 2
  let combinations_4_vertices := Nat.choose total_vertices 4
  (combinations_4_vertices : ℚ) / (ways_to_pick_2_diagonals : ℚ)

theorem decagon_diagonal_intersection_probability :
  probability_intersect_within_decagon = 42 / 119 :=
sorry

end NUMINAMATH_GPT_decagon_diagonal_intersection_probability_l1459_145999


namespace NUMINAMATH_GPT_average_of_distinct_u_l1459_145980

theorem average_of_distinct_u :
  let u_values := { u : ℕ | ∃ (r_1 r_2 : ℕ), r_1 + r_2 = 6 ∧ r_1 * r_2 = u }
  u_values = {5, 8, 9} ∧ (5 + 8 + 9) / 3 = 22 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_of_distinct_u_l1459_145980


namespace NUMINAMATH_GPT_extreme_values_of_f_a_zero_on_interval_monotonicity_of_f_l1459_145936

-- Define the function f(x) = 2*x^3 + 3*(a-2)*x^2 - 12*a*x
def f (x : ℝ) (a : ℝ) := 2*x^3 + 3*(a-2)*x^2 - 12*a*x

-- Define the function f(x) when a = 0
def f_a_zero (x : ℝ) := f x 0

-- Define the intervals and extreme values problem
theorem extreme_values_of_f_a_zero_on_interval :
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 4, f_a_zero x ≤ max ∧ f_a_zero x ≥ min) ∧ max = 32 ∧ min = -40 :=
sorry

-- Define the function for the derivative of f(x)
def f_derivative (x : ℝ) (a : ℝ) := 6*x^2 + 6*(a-2)*x - 12*a

-- Prove the monotonicity based on the value of a
theorem monotonicity_of_f (a : ℝ) :
  (a > -2 → (∀ x, x < -a → f_derivative x a > 0) ∧ (∀ x, -a < x ∧ x < 2 → f_derivative x a < 0) ∧ (∀ x, x > 2 → f_derivative x a > 0)) ∧
  (a = -2 → ∀ x, f_derivative x a ≥ 0) ∧
  (a < -2 → (∀ x, x < 2 → f_derivative x a > 0) ∧ (∀ x, 2 < x ∧ x < -a → f_derivative x a < 0) ∧ (∀ x, x > -a → f_derivative x a > 0)) :=
sorry

end NUMINAMATH_GPT_extreme_values_of_f_a_zero_on_interval_monotonicity_of_f_l1459_145936


namespace NUMINAMATH_GPT_area_of_triangle_DEF_l1459_145938

-- Define point D
def pointD : ℝ × ℝ := (2, 5)

-- Reflect D over the y-axis to get E
def reflectY (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, P.2)
def pointE : ℝ × ℝ := reflectY pointD

-- Reflect E over the line y = -x to get F
def reflectYX (P : ℝ × ℝ) : ℝ × ℝ := (-P.2, -P.1)
def pointF : ℝ × ℝ := reflectYX pointE

-- Define function to calculate the area of the triangle given three points
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

-- Define the Lean 4 statement
theorem area_of_triangle_DEF : triangle_area pointD pointE pointF = 6 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_DEF_l1459_145938


namespace NUMINAMATH_GPT_solve_equation_l1459_145978

-- Given conditions and auxiliary definitions
def is_solution (x y z : ℕ) : Prop := 2 ^ x + 3 ^ y - 7 = Nat.factorial z

-- Primary theorem: the equivalent proof problem
theorem solve_equation (x y z : ℕ) :
  (is_solution x y 3 → (x = 2 ∧ y = 2)) ∧
  (∀ z, (z ≤ 3 → z ≠ 3) → ¬is_solution x y z) ∧
  (z ≥ 4 → ¬is_solution x y z) :=
  sorry

end NUMINAMATH_GPT_solve_equation_l1459_145978


namespace NUMINAMATH_GPT_fred_red_marbles_l1459_145961

theorem fred_red_marbles (total_marbles : ℕ) (dark_blue_marbles : ℕ) (green_marbles : ℕ) (red_marbles : ℕ) 
  (h1 : total_marbles = 63) 
  (h2 : dark_blue_marbles ≥ total_marbles / 3)
  (h3 : green_marbles = 4)
  (h4 : red_marbles = total_marbles - dark_blue_marbles - green_marbles) : 
  red_marbles = 38 := 
sorry

end NUMINAMATH_GPT_fred_red_marbles_l1459_145961


namespace NUMINAMATH_GPT_prove_0_in_A_prove_13_in_A_prove_74_in_A_prove_A_is_Z_l1459_145948

variable {A : Set Int}

-- Assuming set A is closed under subtraction
axiom A_closed_under_subtraction : ∀ x y, x ∈ A → y ∈ A → x - y ∈ A
axiom A_contains_4 : 4 ∈ A
axiom A_contains_9 : 9 ∈ A

theorem prove_0_in_A : 0 ∈ A :=
sorry

theorem prove_13_in_A : 13 ∈ A :=
sorry

theorem prove_74_in_A : 74 ∈ A :=
sorry

theorem prove_A_is_Z : A = Set.univ :=
sorry

end NUMINAMATH_GPT_prove_0_in_A_prove_13_in_A_prove_74_in_A_prove_A_is_Z_l1459_145948


namespace NUMINAMATH_GPT_trick_deck_cost_l1459_145957

theorem trick_deck_cost (x : ℝ) (h1 : 6 * x + 2 * x = 64) : x = 8 :=
  sorry

end NUMINAMATH_GPT_trick_deck_cost_l1459_145957


namespace NUMINAMATH_GPT_parallelogram_area_72_l1459_145958

def parallelogram_area (base height : ℕ) : ℕ :=
  base * height

theorem parallelogram_area_72 :
  parallelogram_area 12 6 = 72 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_72_l1459_145958


namespace NUMINAMATH_GPT_calculate_expression_l1459_145992

theorem calculate_expression : (-1:ℝ)^2 + (1/3:ℝ)^0 = 2 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1459_145992


namespace NUMINAMATH_GPT_simplify_expression_l1459_145983

variable {a b : ℝ}

theorem simplify_expression : (4 * a^3 * b - 2 * a * b) / (2 * a * b) = 2 * a^2 - 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1459_145983


namespace NUMINAMATH_GPT_find_y_l1459_145968

theorem find_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hr : x % y = 9) (hxy : (x : ℝ) / y = 96.45) : y = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1459_145968


namespace NUMINAMATH_GPT_possible_value_is_121_l1459_145902

theorem possible_value_is_121
  (x a y z b : ℕ) 
  (hx : x = 1 / 6 * a) 
  (hz : z = 1 / 6 * b) 
  (hy : y = (a + b) % 5) 
  (h_single_digit : ∀ n, n ∈ [x, a, y, z, b] → n < 10 ∧ 0 < n) : 
  100 * x + 10 * y + z = 121 :=
by
  sorry

end NUMINAMATH_GPT_possible_value_is_121_l1459_145902


namespace NUMINAMATH_GPT_ones_digit_of_34_34_times_17_17_is_6_l1459_145973

def cyclical_pattern_4 (n : ℕ) : ℕ :=
if n % 2 = 0 then 6 else 4

theorem ones_digit_of_34_34_times_17_17_is_6
  (h1 : 34 % 10 = 4)
  (h2 : ∀ n : ℕ, cyclical_pattern_4 n = if n % 2 = 0 then 6 else 4)
  (h3 : 17 % 2 = 1)
  (h4 : (34 * 17^17) % 2 = 0)
  (h5 : ∀ n : ℕ, cyclical_pattern_4 n = if n % 2 = 0 then 6 else 4) :
  (34^(34 * 17^17)) % 10 = 6 := 
by  
  sorry

end NUMINAMATH_GPT_ones_digit_of_34_34_times_17_17_is_6_l1459_145973


namespace NUMINAMATH_GPT_plane_equation_l1459_145997

theorem plane_equation (x y z : ℝ) (A B C D : ℤ) (h1 : A = 9) (h2 : B = -6) (h3 : C = 4) (h4 : D = -133) (A_pos : A > 0) (gcd_condition : Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) : 
  A * x + B * y + C * z + D = 0 :=
sorry

end NUMINAMATH_GPT_plane_equation_l1459_145997


namespace NUMINAMATH_GPT_triangle_side_ratio_sqrt2_l1459_145905

variables (A B C A1 B1 C1 X Y : Point)
variable (triangle : IsAcuteAngledTriangle A B C)
variable (altitudes : AreAltitudes A B C A1 B1 C1)
variable (midpoints : X = Midpoint A C1 ∧ Y = Midpoint A1 C)
variable (equality : Distance X Y = Distance B B1)

theorem triangle_side_ratio_sqrt2 :
  ∃ (AC AB : ℝ), (AC / AB = Real.sqrt 2) := sorry

end NUMINAMATH_GPT_triangle_side_ratio_sqrt2_l1459_145905


namespace NUMINAMATH_GPT_student_walks_fifth_to_first_l1459_145972

theorem student_walks_fifth_to_first :
  let floors := 4
  let staircases := 2
  (staircases ^ floors) = 16 := by
  sorry

end NUMINAMATH_GPT_student_walks_fifth_to_first_l1459_145972


namespace NUMINAMATH_GPT_trivia_team_total_members_l1459_145934

theorem trivia_team_total_members (x : ℕ) (h1 : 4 ≤ x) (h2 : (x - 4) * 8 = 64) : x = 12 :=
sorry

end NUMINAMATH_GPT_trivia_team_total_members_l1459_145934


namespace NUMINAMATH_GPT_Oates_reunion_l1459_145947

-- Declare the conditions as variables
variables (total_guests both_reunions yellow_reunion : ℕ)
variables (H1 : total_guests = 100)
variables (H2 : both_reunions = 7)
variables (H3 : yellow_reunion = 65)

-- The proof problem statement
theorem Oates_reunion (O : ℕ) (H4 : total_guests = O + yellow_reunion - both_reunions) : O = 42 :=
sorry

end NUMINAMATH_GPT_Oates_reunion_l1459_145947


namespace NUMINAMATH_GPT_divisor_of_z_in_form_4n_minus_1_l1459_145977

theorem divisor_of_z_in_form_4n_minus_1
  (x y : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (z : ℕ) 
  (hz : z = 4 * x * y / (x + y)) 
  (hz_odd : z % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ ∃ d : ℕ, d ∣ z ∧ d = 4 * n - 1 :=
sorry

end NUMINAMATH_GPT_divisor_of_z_in_form_4n_minus_1_l1459_145977


namespace NUMINAMATH_GPT_time_taken_by_A_l1459_145969

theorem time_taken_by_A (t : ℚ) (h1 : 3 * (t + 1 / 2) = 4 * t) : t = 3 / 2 ∧ (t + 1 / 2) = 2 := 
  by
  intros
  sorry

end NUMINAMATH_GPT_time_taken_by_A_l1459_145969


namespace NUMINAMATH_GPT_ab5_a2_c5_a2_inequality_l1459_145963

theorem ab5_a2_c5_a2_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a ^ 5 - a ^ 2 + 3) * (b ^ 5 - b ^ 2 + 3) * (c ^ 5 - c ^ 2 + 3) ≥ (a + b + c) ^ 3 := 
by
  sorry

end NUMINAMATH_GPT_ab5_a2_c5_a2_inequality_l1459_145963


namespace NUMINAMATH_GPT_applicants_less_4_years_no_degree_l1459_145962

theorem applicants_less_4_years_no_degree
    (total_applicants : ℕ)
    (A : ℕ) 
    (B : ℕ)
    (C : ℕ)
    (D : ℕ)
    (h_total : total_applicants = 30)
    (h_A : A = 10)
    (h_B : B = 18)
    (h_C : C = 9)
    (h_D : total_applicants - (A - C + B - C + C) = D) :
  D = 11 :=
by
  sorry

end NUMINAMATH_GPT_applicants_less_4_years_no_degree_l1459_145962


namespace NUMINAMATH_GPT_pickles_per_cucumber_l1459_145910

theorem pickles_per_cucumber (jars cucumbers vinegar_initial vinegar_left pickles_per_jar vinegar_per_jar total_pickles_per_cucumber : ℕ) 
    (h1 : jars = 4) 
    (h2 : cucumbers = 10) 
    (h3 : vinegar_initial = 100) 
    (h4 : vinegar_left = 60) 
    (h5 : pickles_per_jar = 12) 
    (h6 : vinegar_per_jar = 10) 
    (h7 : total_pickles_per_cucumber = 4): 
    total_pickles_per_cucumber = (vinegar_initial - vinegar_left) / vinegar_per_jar * pickles_per_jar / cucumbers := 
by 
  sorry

end NUMINAMATH_GPT_pickles_per_cucumber_l1459_145910


namespace NUMINAMATH_GPT_correct_statement_is_A_l1459_145913

theorem correct_statement_is_A : 
  (∀ x : ℝ, 0 ≤ x → abs x = x) ∧
  ¬ (∀ x : ℝ, x ≤ 0 → -x = x) ∧
  ¬ (∀ x : ℝ, (x ≠ 0 ∧ x⁻¹ = x) → (x = 1 ∨ x = -1 ∨ x = 0)) ∧
  ¬ (∀ x y : ℝ, x < 0 ∧ y < 0 → abs x < abs y → x < y) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_is_A_l1459_145913


namespace NUMINAMATH_GPT_other_point_on_circle_l1459_145940

noncomputable def circle_center_radius (p : ℝ × ℝ) (r : ℝ) : Prop :=
  dist p (0, 0) = r

theorem other_point_on_circle (r : ℝ) (h : r = 16) (point_on_circle : circle_center_radius (16, 0) r) :
  circle_center_radius (-16, 0) r :=
by
  sorry

end NUMINAMATH_GPT_other_point_on_circle_l1459_145940


namespace NUMINAMATH_GPT_symmetry_condition_l1459_145950

theorem symmetry_condition (p q r s t u : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (yx_eq : ∀ x y, y = (p * x ^ 2 + q * x + r) / (s * x ^ 2 + t * x + u) ↔ x = (p * y ^ 2 + q * y + r) / (s * y ^ 2 + t * y + u)) :
  p = s ∧ q = t ∧ r = u :=
sorry

end NUMINAMATH_GPT_symmetry_condition_l1459_145950


namespace NUMINAMATH_GPT_min_unplowed_cells_l1459_145944

theorem min_unplowed_cells (n k : ℕ) (hn : n > 0) (hk : k > 0) (hnk : n > k) :
  ∃ M : ℕ, M = (n - k)^2 := by
  sorry

end NUMINAMATH_GPT_min_unplowed_cells_l1459_145944


namespace NUMINAMATH_GPT_student_marks_l1459_145967

variable (max_marks : ℕ) (pass_percent : ℕ) (fail_by : ℕ)

theorem student_marks
  (h_max : max_marks = 400)
  (h_pass : pass_percent = 35)
  (h_fail : fail_by = 40)
  : max_marks * pass_percent / 100 - fail_by = 100 :=
by
  sorry

end NUMINAMATH_GPT_student_marks_l1459_145967


namespace NUMINAMATH_GPT_problem_statement_l1459_145941

variables {c c' d d' : ℝ}

theorem problem_statement (hc : c ≠ 0) (hc' : c' ≠ 0)
  (h : (-d) / (2 * c) = 2 * ((-d') / (3 * c'))) :
  (d / (2 * c)) = 2 * (d' / (3 * c')) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1459_145941


namespace NUMINAMATH_GPT_total_quarters_l1459_145979

def Sara_initial_quarters : Nat := 21
def quarters_given_by_dad : Nat := 49

theorem total_quarters : Sara_initial_quarters + quarters_given_by_dad = 70 := 
by
  sorry

end NUMINAMATH_GPT_total_quarters_l1459_145979


namespace NUMINAMATH_GPT_oil_leak_l1459_145900

theorem oil_leak (a b c : ℕ) (h₁ : a = 6522) (h₂ : b = 11687) (h₃ : c = b - a) : c = 5165 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end NUMINAMATH_GPT_oil_leak_l1459_145900


namespace NUMINAMATH_GPT_team_B_score_third_game_l1459_145955

theorem team_B_score_third_game (avg_points : ℝ) (additional_needed : ℝ) (total_target : ℝ) (P : ℝ) :
  avg_points = 61.5 → additional_needed = 330 → total_target = 500 →
  2 * avg_points + P + additional_needed = total_target → P = 47 :=
by
  intros avg_points_eq additional_needed_eq total_target_eq total_eq
  rw [avg_points_eq, additional_needed_eq, total_target_eq] at total_eq
  sorry

end NUMINAMATH_GPT_team_B_score_third_game_l1459_145955


namespace NUMINAMATH_GPT_point_symmetric_y_axis_l1459_145971

theorem point_symmetric_y_axis (a b : ℤ) (h₁ : a = -(-2)) (h₂ : b = 3) : a + b = 5 := by
  sorry

end NUMINAMATH_GPT_point_symmetric_y_axis_l1459_145971


namespace NUMINAMATH_GPT_part_a_l1459_145952

def f_X (X : Set (ℝ × ℝ)) (n : ℕ) : ℝ :=
  sorry  -- Placeholder for the largest possible area function

theorem part_a (X : Set (ℝ × ℝ)) (m n : ℕ) (h1 : m ≥ n) (h2 : n > 2) :
  f_X X m + f_X X n ≥ f_X X (m + 1) + f_X X (n - 1) :=
sorry

end NUMINAMATH_GPT_part_a_l1459_145952


namespace NUMINAMATH_GPT_recurring_decimal_sum_l1459_145960

theorem recurring_decimal_sum :
  let x := (4 / 33)
  let y := (34 / 99)
  x + y = (46 / 99) := by
  sorry

end NUMINAMATH_GPT_recurring_decimal_sum_l1459_145960


namespace NUMINAMATH_GPT_fraction_books_sold_l1459_145966

theorem fraction_books_sold (B : ℕ) (F : ℚ) (h1 : 36 = B - F * B) (h2 : 252 = 3.50 * F * B) : F = 2 / 3 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_fraction_books_sold_l1459_145966


namespace NUMINAMATH_GPT_golden_section_PB_l1459_145943

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem golden_section_PB {A B P : ℝ} (h1 : P = (1 - 1/(golden_ratio)) * A + (1/(golden_ratio)) * B)
  (h2 : AB = 2)
  (h3 : A ≠ B) : PB = 3 - Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_golden_section_PB_l1459_145943


namespace NUMINAMATH_GPT_point_reflection_xOy_l1459_145956

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def reflection_over_xOy (P : Point3D) : Point3D := 
  {x := P.x, y := P.y, z := -P.z}

theorem point_reflection_xOy :
  reflection_over_xOy {x := 1, y := 2, z := 3} = {x := 1, y := 2, z := -3} := by
  sorry

end NUMINAMATH_GPT_point_reflection_xOy_l1459_145956


namespace NUMINAMATH_GPT_vendor_second_day_sale_l1459_145949

theorem vendor_second_day_sale (n : ℕ) :
  let sold_first_day := (50 * n) / 100
  let remaining_after_first_sale := n - sold_first_day
  let thrown_away_first_day := (20 * remaining_after_first_sale) / 100
  let remaining_after_first_day := remaining_after_first_sale - thrown_away_first_day
  let total_thrown_away := (30 * n) / 100
  let thrown_away_second_day := total_thrown_away - thrown_away_first_day
  let sold_second_day := remaining_after_first_day - thrown_away_second_day
  let percent_sold_second_day := (sold_second_day * 100) / remaining_after_first_day
  percent_sold_second_day = 50 :=
sorry

end NUMINAMATH_GPT_vendor_second_day_sale_l1459_145949


namespace NUMINAMATH_GPT_initial_total_fish_l1459_145929

def total_days (weeks : ℕ) : ℕ := weeks * 7
def fish_added (rate : ℕ) (days : ℕ) : ℕ := rate * days
def initial_fish (final_count : ℕ) (added : ℕ) : ℕ := final_count - added

theorem initial_total_fish {final_goldfish final_koi rate_goldfish rate_koi days init_goldfish init_koi : ℕ}
    (h_final_goldfish : final_goldfish = 200)
    (h_final_koi : final_koi = 227)
    (h_rate_goldfish : rate_goldfish = 5)
    (h_rate_koi : rate_koi = 2)
    (h_days : days = total_days 3)
    (h_init_goldfish : init_goldfish = initial_fish final_goldfish (fish_added rate_goldfish days))
    (h_init_koi : init_koi = initial_fish final_koi (fish_added rate_koi days)) :
    init_goldfish + init_koi = 280 :=
by
    sorry -- skipping the proof

end NUMINAMATH_GPT_initial_total_fish_l1459_145929


namespace NUMINAMATH_GPT_teddy_bears_per_shelf_l1459_145933

theorem teddy_bears_per_shelf :
  (98 / 14 = 7) := 
by
  sorry

end NUMINAMATH_GPT_teddy_bears_per_shelf_l1459_145933


namespace NUMINAMATH_GPT_lunch_cost_calc_l1459_145970

-- Define the given conditions
def gasoline_cost : ℝ := 8
def gift_cost : ℝ := 5
def grandma_gift : ℝ := 10
def initial_money : ℝ := 50
def return_trip_money : ℝ := 36.35

-- Calculate the total expenses and determine the money spent on lunch
def total_gifts_cost : ℝ := 2 * gift_cost
def total_money_received : ℝ := initial_money + 2 * grandma_gift
def total_gas_gift_cost : ℝ := gasoline_cost + total_gifts_cost
def expected_remaining_money : ℝ := total_money_received - total_gas_gift_cost
def lunch_cost : ℝ := expected_remaining_money - return_trip_money

-- State theorem
theorem lunch_cost_calc : lunch_cost = 15.65 := by
  sorry

end NUMINAMATH_GPT_lunch_cost_calc_l1459_145970


namespace NUMINAMATH_GPT_discount_of_bag_l1459_145989

def discounted_price (marked_price discount_rate : ℕ) : ℕ :=
  marked_price - ((discount_rate * marked_price) / 100)

theorem discount_of_bag : discounted_price 200 40 = 120 :=
by
  unfold discounted_price
  norm_num

end NUMINAMATH_GPT_discount_of_bag_l1459_145989


namespace NUMINAMATH_GPT_range_of_m_l1459_145981

noncomputable def function_even_and_monotonic (f : ℝ → ℝ) := 
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x > f y)

variable (f : ℝ → ℝ)
variable (m : ℝ)

theorem range_of_m (h₁ : function_even_and_monotonic f) 
  (h₂ : f m > f (1 - m)) : m < 1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1459_145981


namespace NUMINAMATH_GPT_yards_mowed_by_christian_l1459_145984

-- Definitions based on the provided conditions
def initial_savings := 5 + 7
def sue_earnings := 6 * 2
def total_savings := initial_savings + sue_earnings
def additional_needed := 50 - total_savings
def short_amount := 6
def christian_earnings := additional_needed - short_amount
def charge_per_yard := 5

theorem yards_mowed_by_christian : 
  (christian_earnings / charge_per_yard) = 4 :=
by
  sorry

end NUMINAMATH_GPT_yards_mowed_by_christian_l1459_145984


namespace NUMINAMATH_GPT_find_Minchos_chocolate_l1459_145987

variable (M : ℕ)  -- Define M as a natural number

-- Define the conditions as Lean hypotheses
def TaeminChocolate := 5 * M
def KibumChocolate := 3 * M
def TotalChocolate := TaeminChocolate M + KibumChocolate M

theorem find_Minchos_chocolate (h : TotalChocolate M = 160) : M = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_Minchos_chocolate_l1459_145987


namespace NUMINAMATH_GPT_cost_of_45_daffodils_equals_75_l1459_145928

-- Conditions
def cost_of_15_daffodils : ℝ := 25
def number_of_daffodils_in_bouquet_15 : ℕ := 15
def number_of_daffodils_in_bouquet_45 : ℕ := 45
def directly_proportional (n m : ℕ) (c_n c_m : ℝ) : Prop := c_n / n = c_m / m

-- Statement to prove
theorem cost_of_45_daffodils_equals_75 :
  ∀ (c : ℝ), directly_proportional number_of_daffodils_in_bouquet_45 number_of_daffodils_in_bouquet_15 c cost_of_15_daffodils → c = 75 :=
by
  intro c hypothesis
  -- Proof would go here.
  sorry

end NUMINAMATH_GPT_cost_of_45_daffodils_equals_75_l1459_145928


namespace NUMINAMATH_GPT_student_question_choice_l1459_145925

/-- A student needs to choose 8 questions from part A and 5 questions from part B. Both parts contain 10 questions each.
   This Lean statement proves that the student can choose the questions in 11340 different ways. -/
theorem student_question_choice : (Nat.choose 10 8) * (Nat.choose 10 5) = 11340 := by
  sorry

end NUMINAMATH_GPT_student_question_choice_l1459_145925


namespace NUMINAMATH_GPT_ratio_of_fractions_proof_l1459_145935

noncomputable def ratio_of_fractions (x y : ℝ) : Prop :=
  (5 * x = 6 * y) → (x ≠ 0 ∧ y ≠ 0) → ((1/3) * x / ((1/5) * y) = 2)

theorem ratio_of_fractions_proof (x y : ℝ) (hx: 5 * x = 6 * y) (hnz: x ≠ 0 ∧ y ≠ 0) : ((1/3) * x / ((1/5) * y) = 2) :=
  by 
  sorry

end NUMINAMATH_GPT_ratio_of_fractions_proof_l1459_145935


namespace NUMINAMATH_GPT_sum_of_tens_and_units_digit_of_7_pow_2023_l1459_145946

theorem sum_of_tens_and_units_digit_of_7_pow_2023 :
  let n := 7 ^ 2023
  (n % 100).div 10 + (n % 10) = 16 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_tens_and_units_digit_of_7_pow_2023_l1459_145946


namespace NUMINAMATH_GPT_seed_selection_valid_l1459_145923

def seeds : List Nat := [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07]

def extractValidSeeds (lst : List Nat) (startIndex : Nat) (maxValue : Nat) (count : Nat) : List Nat :=
  lst.drop startIndex
  |>.filter (fun n => n < maxValue)
  |>.take count

theorem seed_selection_valid :
  extractValidSeeds seeds 10 850 4 = [169, 555, 671, 105] :=
by
  sorry

end NUMINAMATH_GPT_seed_selection_valid_l1459_145923


namespace NUMINAMATH_GPT_son_age_l1459_145907

-- Defining the variables
variables (S F : ℕ)

-- The conditions
def condition1 : Prop := F = S + 25
def condition2 : Prop := F + 2 = 2 * (S + 2)

-- The statement to be proved
theorem son_age (h1 : condition1 S F) (h2 : condition2 S F) : S = 23 :=
sorry

end NUMINAMATH_GPT_son_age_l1459_145907


namespace NUMINAMATH_GPT_cos2alpha_minus_sin2alpha_l1459_145954

theorem cos2alpha_minus_sin2alpha (α : ℝ) (h1 : α ∈ Set.Icc (-π/2) 0) 
  (h2 : (Real.sin (3 * α)) / (Real.sin α) = 13 / 5) :
  Real.cos (2 * α) - Real.sin (2 * α) = (3 + Real.sqrt 91) / 10 :=
sorry

end NUMINAMATH_GPT_cos2alpha_minus_sin2alpha_l1459_145954


namespace NUMINAMATH_GPT_find_value_l1459_145903

theorem find_value (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^2 + b^2 + a * b = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l1459_145903


namespace NUMINAMATH_GPT_product_increased_l1459_145912

theorem product_increased (a b c : ℕ) (h1 : a = 1) (h2: b = 1) (h3: c = 676) :
  ((a - 3) * (b - 3) * (c - 3) = a * b * c + 2016) :=
by
  simp [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_product_increased_l1459_145912


namespace NUMINAMATH_GPT_hyperbola_equation_l1459_145994

-- Define the hyperbola with vertices and other conditions
def Hyperbola (a b : ℝ) (h : a > 0 ∧ b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1)

-- Given conditions and the proof goal
theorem hyperbola_equation
  (a b : ℝ) (h : a > 0 ∧ b > 0)
  (P : ℝ × ℝ) (M N : ℝ × ℝ)
  (k_PA k_PB : ℝ)
  (PA_PB_condition : k_PA * k_PB = 3)
  (MN_min_value : |(M.1 - N.1) + (M.2 - N.2)| = 4) :
  Hyperbola a b h →
  (a = 2 ∧ b = 2 * Real.sqrt 3 ∧ (∀ (x y : ℝ), (x^2 / 4 - y^2 / 12 = 1)) ∨ 
   a = 2 / 3 ∧ b = 2 * Real.sqrt 3 / 3 ∧ (∀ (x y : ℝ), (9 * x^2 / 4 - 3 * y^2 / 4 = 1)))
:=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l1459_145994


namespace NUMINAMATH_GPT_geometric_segment_l1459_145976

theorem geometric_segment (AB A'B' : ℝ) (P D A B P' D' A' B' : ℝ) (x y a : ℝ) :
  AB = 3 ∧ A'B' = 6 ∧ (∀ P, dist P D = x) ∧ (∀ P', dist P' D' = 2 * x) ∧ x = a → x + y = 3 * a :=
by
  sorry

end NUMINAMATH_GPT_geometric_segment_l1459_145976


namespace NUMINAMATH_GPT_solve_inequalities_l1459_145919

theorem solve_inequalities (x : ℝ) :
  ( (-x + 3)/2 < x ∧ 2*(x + 6) ≥ 5*x ) ↔ (1 < x ∧ x ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l1459_145919


namespace NUMINAMATH_GPT_compare_neg_fractions_l1459_145995

theorem compare_neg_fractions : - (2 / 3 : ℝ) > - (3 / 4 : ℝ) :=
sorry

end NUMINAMATH_GPT_compare_neg_fractions_l1459_145995


namespace NUMINAMATH_GPT_proof_of_inequality_l1459_145930

theorem proof_of_inequality (a b : ℝ) (h : a > b) : a - 3 > b - 3 :=
sorry

end NUMINAMATH_GPT_proof_of_inequality_l1459_145930


namespace NUMINAMATH_GPT_total_flight_time_l1459_145991

theorem total_flight_time
  (distance : ℕ)
  (speed_out : ℕ)
  (speed_return : ℕ)
  (time_out : ℕ)
  (time_return : ℕ)
  (total_time : ℕ)
  (h1 : distance = 1500)
  (h2 : speed_out = 300)
  (h3 : speed_return = 500)
  (h4 : time_out = distance / speed_out)
  (h5 : time_return = distance / speed_return)
  (h6 : total_time = time_out + time_return) :
  total_time = 8 := 
  by {
    sorry
  }

end NUMINAMATH_GPT_total_flight_time_l1459_145991


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1459_145939

theorem arithmetic_sequence_common_difference :
  let a := 5
  let a_n := 50
  let S_n := 330
  exists (d n : ℤ), (a + (n - 1) * d = a_n) ∧ (n * (a + a_n) / 2 = S_n) ∧ (d = 45 / 11) :=
by
  let a := 5
  let a_n := 50
  let S_n := 330
  use 45 / 11, 12
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1459_145939


namespace NUMINAMATH_GPT_smallest_perfect_square_divisible_by_2_and_5_l1459_145990

theorem smallest_perfect_square_divisible_by_2_and_5 :
  ∃ n : ℕ, 0 < n ∧ (∃ k : ℕ, n = k ^ 2) ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ 
  (∀ m : ℕ, 0 < m ∧ (∃ k : ℕ, m = k ^ 2) ∧ (m % 2 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_perfect_square_divisible_by_2_and_5_l1459_145990


namespace NUMINAMATH_GPT_intersection_count_l1459_145931

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem intersection_count (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < Real.pi / 2) 
  (h_max : ∀ x, f x ω φ ≤ f (Real.pi / 6) ω φ)
  (h_period : ∀ x, f x ω φ = f (x + 2 * Real.pi / ω) ω φ) :
  ∃! x : ℝ, f x ω φ = -x + 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_intersection_count_l1459_145931


namespace NUMINAMATH_GPT_angle_between_vectors_l1459_145988

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem angle_between_vectors
  (a b : ℝ × ℝ)
  (h₁ : vector_norm a ≠ 0)
  (h₂ : vector_norm b ≠ 0)
  (h₃ : vector_norm a = vector_norm b)
  (h₄ : vector_norm a = vector_norm (a.1 + 2 * b.1, a.2 + 2 * b.2)) :
  ∃ θ : ℝ, θ = 180 ∧ cos θ = -1 := 
sorry

end NUMINAMATH_GPT_angle_between_vectors_l1459_145988


namespace NUMINAMATH_GPT_probability_of_four_odd_slips_l1459_145985

-- Define the conditions
def number_of_slips : ℕ := 10
def odd_slips : ℕ := 5
def even_slips : ℕ := 5
def slips_drawn : ℕ := 4

-- Define the required probability calculation
def probability_four_odd_slips : ℚ := (5 / 10) * (4 / 9) * (3 / 8) * (2 / 7)

-- State the theorem we want to prove
theorem probability_of_four_odd_slips :
  probability_four_odd_slips = 1 / 42 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_four_odd_slips_l1459_145985


namespace NUMINAMATH_GPT_boxes_needed_l1459_145918

theorem boxes_needed (balls : ℕ) (balls_per_box : ℕ) (h1 : balls = 10) (h2 : balls_per_box = 5) : balls / balls_per_box = 2 := by
  sorry

end NUMINAMATH_GPT_boxes_needed_l1459_145918


namespace NUMINAMATH_GPT_no_absolute_winner_l1459_145914

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end NUMINAMATH_GPT_no_absolute_winner_l1459_145914
