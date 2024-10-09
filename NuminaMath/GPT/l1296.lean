import Mathlib

namespace find_value_of_a_l1296_129621

variable (a : ℝ)

def f (x : ℝ) := x^2 + 4
def g (x : ℝ) := x^2 - 2

theorem find_value_of_a (h_pos : a > 0) (h_eq : f (g a) = 12) : a = Real.sqrt (2 * (Real.sqrt 2 + 1)) := 
by
  sorry

end find_value_of_a_l1296_129621


namespace minimum_value_of_k_l1296_129680

theorem minimum_value_of_k (m n a k : ℕ) (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) (hk : 1 < k) (h : 5^m + 63 * n + 49 = a^k) : k = 5 :=
sorry

end minimum_value_of_k_l1296_129680


namespace anthony_lunch_money_l1296_129614

-- Define the costs as given in the conditions
def juice_box_cost : ℕ := 27
def cupcake_cost : ℕ := 40
def amount_left : ℕ := 8

-- Define the total amount needed for lunch every day
def total_amount_for_lunch : ℕ := juice_box_cost + cupcake_cost + amount_left

theorem anthony_lunch_money : total_amount_for_lunch = 75 := by
  -- This is where the proof would go.
  sorry

end anthony_lunch_money_l1296_129614


namespace problem_b_correct_l1296_129619

theorem problem_b_correct (a b : ℝ) (h₁ : a < 0) (h₂ : 0 < b) (h₃ : b < 1) : (ab^2 > ab ∧ ab > a) :=
by
  sorry

end problem_b_correct_l1296_129619


namespace ahmed_goats_correct_l1296_129617

-- Definitions based on the conditions given in the problem.
def adam_goats : ℕ := 7
def andrew_goats : ℕ := 5 + 2 * adam_goats
def ahmed_goats : ℕ := andrew_goats - 6

-- The theorem statement that needs to be proven.
theorem ahmed_goats_correct : ahmed_goats = 13 := by
    sorry

end ahmed_goats_correct_l1296_129617


namespace time_addition_correct_l1296_129620

theorem time_addition_correct :
  let current_time := (3, 0, 0)  -- Representing 3:00:00 PM as a tuple (hours, minutes, seconds)
  let duration := (313, 45, 56)  -- Duration to be added: 313 hours, 45 minutes, and 56 seconds
  let new_time := ((3 + (313 % 12) + 45 / 60 + (56 / 3600)), (0 + 45 % 60), (0 + 56 % 60))
  let A := (4 : ℕ)  -- Extracted hour part of new_time
  let B := (45 : ℕ)  -- Extracted minute part of new_time
  let C := (56 : ℕ)  -- Extracted second part of new_time
  A + B + C = 105 := 
by
  -- Placeholder for the actual proof.
  sorry

end time_addition_correct_l1296_129620


namespace opposite_of_negative_one_fifth_l1296_129678

theorem opposite_of_negative_one_fifth : -(-1 / 5) = (1 / 5) :=
by
  sorry

end opposite_of_negative_one_fifth_l1296_129678


namespace vector_operation_l1296_129649

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (a b : α)

theorem vector_operation :
  (1 / 2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by sorry

end vector_operation_l1296_129649


namespace commission_percentage_l1296_129606

theorem commission_percentage 
  (cost_price : ℝ) (profit_percentage : ℝ) (observed_price : ℝ) (C : ℝ) 
  (h1 : cost_price = 15)
  (h2 : profit_percentage = 0.10)
  (h3 : observed_price = 19.8) 
  (h4 : 1 + C / 100 = 19.8 / (cost_price * (1 + profit_percentage)))
  : C = 20 := 
by
  sorry

end commission_percentage_l1296_129606


namespace sum_of_base_areas_eq_5_l1296_129658

-- Define the surface area, lateral area, and the sum of the areas of the two base faces.
def surface_area : ℝ := 30
def lateral_area : ℝ := 25
def sum_base_areas : ℝ := surface_area - lateral_area

-- The theorem statement.
theorem sum_of_base_areas_eq_5 : sum_base_areas = 5 := 
by 
  sorry

end sum_of_base_areas_eq_5_l1296_129658


namespace black_squares_31x31_l1296_129669

-- Definitions to express the checkerboard problem conditions
def isCheckerboard (n : ℕ) : Prop := 
  ∀ i j : ℕ,
    i < n → j < n → 
    ((i + j) % 2 = 0 → (i % 2 = 0 ∧ j % 2 = 0) ∨ (i % 2 = 1 ∧ j % 2 = 1))

def blackCornerSquares (n : ℕ) : Prop :=
  ∀ i j : ℕ,
    (i = 0 ∨ i = n - 1) ∧ (j = 0 ∨ j = n - 1) → (i + j) % 2 = 0

-- The main statement to prove
theorem black_squares_31x31 :
  ∃ (n : ℕ) (count : ℕ), n = 31 ∧ isCheckerboard n ∧ blackCornerSquares n ∧ count = 481 := 
by 
  sorry -- Proof to be provided

end black_squares_31x31_l1296_129669


namespace trigonometric_inequality_l1296_129630

theorem trigonometric_inequality (x : ℝ) (n m : ℕ) 
  (hx : 0 < x ∧ x < (Real.pi / 2))
  (hnm : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤
  3 * |Real.sin x ^ m - Real.cos x ^ m| := 
by 
  sorry

end trigonometric_inequality_l1296_129630


namespace marbles_in_jar_l1296_129672

theorem marbles_in_jar (T : ℕ) (T_half : T / 2 = 12) (red_marbles : ℕ) (orange_marbles : ℕ) (total_non_blue : red_marbles + orange_marbles = 12) (red_count : red_marbles = 6) (orange_count : orange_marbles = 6) : T = 24 :=
by
  sorry

end marbles_in_jar_l1296_129672


namespace measure_15_minutes_with_hourglasses_l1296_129618

theorem measure_15_minutes_with_hourglasses (h7 h11 : ℕ) (h7_eq : h7 = 7) (h11_eq : h11 = 11) : ∃ t : ℕ, t = 15 :=
by
  let t := 15
  have h7 : ℕ := 7
  have h11 : ℕ := 11
  exact ⟨t, by norm_num⟩

end measure_15_minutes_with_hourglasses_l1296_129618


namespace focus_of_hyperbola_l1296_129657

-- Define the given hyperbola equation and its conversion to standard form
def hyperbola_eq (x y : ℝ) : Prop := -2 * (x - 2)^2 + 3 * (y + 3)^2 - 28 = 0

-- Define the standard form equation of the hyperbola
def standard_form (x y : ℝ) : Prop :=
  ((y + 3)^2 / (28 / 3)) - ((x - 2)^2 / 14) = 1

-- Define the coordinates of one of the foci of the hyperbola
def focus (x y : ℝ) : Prop :=
  x = 2 ∧ y = -3 + Real.sqrt (70 / 3)

-- The theorem statement proving the given coordinates is a focus of the hyperbola
theorem focus_of_hyperbola :
  ∃ x y, hyperbola_eq x y ∧ standard_form x y → focus x y :=
by
  existsi 2, (-3 + Real.sqrt (70 / 3))
  sorry -- Proof is required to substantiate it, placeholder here.

end focus_of_hyperbola_l1296_129657


namespace no_hamiltonian_cycle_l1296_129654

-- Define the problem constants
def n : ℕ := 2016
def a : ℕ := 2
def b : ℕ := 3

-- Define the circulant graph and the conditions of the Hamiltonian cycle theorem
theorem no_hamiltonian_cycle (s t : ℕ) (h1 : s + t = Int.gcd n (a - b)) :
  ¬ (Int.gcd n (s * a + t * b) = 1) :=
by
  sorry  -- Proof not required as per instructions

end no_hamiltonian_cycle_l1296_129654


namespace students_neither_football_nor_cricket_l1296_129644

theorem students_neither_football_nor_cricket 
  (total_students : ℕ) 
  (football_players : ℕ) 
  (cricket_players : ℕ) 
  (both_players : ℕ) 
  (H1 : total_students = 410) 
  (H2 : football_players = 325) 
  (H3 : cricket_players = 175) 
  (H4 : both_players = 140) :
  total_students - (football_players + cricket_players - both_players) = 50 :=
by
  sorry

end students_neither_football_nor_cricket_l1296_129644


namespace find_k_l1296_129624

noncomputable def distance_x (x : ℝ) := 5
noncomputable def distance_y (x k : ℝ) := |x^2 - k|
noncomputable def total_distance (x k : ℝ) := distance_x x + distance_y x k

theorem find_k (x k : ℝ) (hk : distance_y x k = 2 * distance_x x) (htot : total_distance x k = 30) :
  k = x^2 - 10 :=
sorry

end find_k_l1296_129624


namespace opposite_of_neg_three_l1296_129600

theorem opposite_of_neg_three : -(-3) = 3 :=
by 
  sorry

end opposite_of_neg_three_l1296_129600


namespace john_total_distance_l1296_129602

-- Define the given conditions
def initial_speed : ℝ := 45 -- mph
def first_leg_time : ℝ := 2 -- hours
def second_leg_time : ℝ := 3 -- hours
def distance_before_lunch : ℝ := initial_speed * first_leg_time
def distance_after_lunch : ℝ := initial_speed * second_leg_time

-- Define the total distance
def total_distance : ℝ := distance_before_lunch + distance_after_lunch

-- Prove the total distance is 225 miles
theorem john_total_distance : total_distance = 225 := by
  sorry

end john_total_distance_l1296_129602


namespace rhombus_perimeter_area_l1296_129653

theorem rhombus_perimeter_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (right_angle : ∀ (x : ℝ), x = d1 / 2 ∧ x = d2 / 2 → x * x + x * x = (d1 / 2)^2 + (d2 / 2)^2) : 
  ∃ (P A : ℝ), P = 52 ∧ A = 120 :=
by
  sorry

end rhombus_perimeter_area_l1296_129653


namespace range_of_a_l1296_129694

-- Define the condition p
def p (x : ℝ) : Prop := (2 * x^2 - 3 * x + 1) ≤ 0

-- Define the condition q
def q (x a : ℝ) : Prop := (x^2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0

-- Lean statement for the problem
theorem range_of_a (a : ℝ) : (¬ (∃ x, p x) → ¬ (∃ x, q x a)) → ((0 : ℝ) ≤ a ∧ a ≤ (1 / 2 : ℝ)) :=
by 
  sorry

end range_of_a_l1296_129694


namespace car_speed_5_hours_l1296_129640

variable (T : ℝ)
variable (S : ℝ)

theorem car_speed_5_hours (h1 : T > 0) (h2 : 2 * T = S * 5.0) : S = 2 * T / 5.0 :=
sorry

end car_speed_5_hours_l1296_129640


namespace sequence_arith_or_geom_l1296_129616

def sequence_nature (a S : ℕ → ℝ) : Prop :=
  ∀ n, 4 * S n = (a n + 1) ^ 2

theorem sequence_arith_or_geom {a : ℕ → ℝ} {S : ℕ → ℝ} (h : sequence_nature a S) (h₁ : a 1 = 1) :
  (∃ d, ∀ n, a (n + 1) = a n + d) ∨ (∃ r, ∀ n, a (n + 1) = a n * r) :=
sorry

end sequence_arith_or_geom_l1296_129616


namespace parametric_two_rays_l1296_129693

theorem parametric_two_rays (t : ℝ) : (x = t + 1 / t ∧ y = 2) → (x ≤ -2 ∨ x ≥ 2) := by
  sorry

end parametric_two_rays_l1296_129693


namespace rhombus_area_l1296_129671

-- Define the given conditions as parameters
variables (EF GH : ℝ) -- Sides of the rhombus
variables (d1 d2 : ℝ) -- Diagonals of the rhombus

-- Statement of the theorem
theorem rhombus_area
  (rhombus_EFGH : ∀ (EF GH : ℝ), EF = GH)
  (perimeter_EFGH : 4 * EF = 40)
  (diagonal_EG_length : d1 = 16)
  (d1_half : d1 / 2 = 8)
  (side_length : EF = 10)
  (pythagorean_theorem : EF^2 = (d1 / 2)^2 + (d2 / 2)^2)
  (calculate_FI : d2 / 2 = 6)
  (diagonal_FG_length : d2 = 12) :
  (1 / 2) * d1 * d2 = 96 :=
sorry

end rhombus_area_l1296_129671


namespace circumradius_of_regular_tetrahedron_l1296_129660

theorem circumradius_of_regular_tetrahedron (a : ℝ) (h : a > 0) :
    ∃ R : ℝ, R = a * (Real.sqrt 6) / 4 :=
by
  sorry

end circumradius_of_regular_tetrahedron_l1296_129660


namespace jennifer_tanks_l1296_129664

theorem jennifer_tanks (initial_tanks : ℕ) (fish_per_initial_tank : ℕ) (total_fish_needed : ℕ) 
  (additional_tanks : ℕ) (fish_per_additional_tank : ℕ) 
  (initial_calculation : initial_tanks = 3) (fish_per_initial_calculation : fish_per_initial_tank = 15)
  (total_fish_calculation : total_fish_needed = 75) (additional_tanks_calculation : additional_tanks = 3) :
  initial_tanks * fish_per_initial_tank + additional_tanks * fish_per_additional_tank = total_fish_needed 
  → fish_per_additional_tank = 10 := 
by sorry

end jennifer_tanks_l1296_129664


namespace An_is_integer_l1296_129689

theorem An_is_integer 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_gt : a > b)
  (θ : ℝ) (h_theta : θ > 0 ∧ θ < Real.pi / 2)
  (h_sin : Real.sin θ = 2 * (a * b) / (a^2 + b^2)) :
  ∀ n : ℕ, ∃ k : ℤ, ((a^2 + b^2)^n * Real.sin (n * θ) : ℝ) = k :=
by sorry

end An_is_integer_l1296_129689


namespace greatest_power_sum_l1296_129626

theorem greatest_power_sum (a b : ℕ) (h1 : 0 < a) (h2 : 2 < b) (h3 : a^b < 500) (h4 : ∀ m n : ℕ, 0 < m → 2 < n → m^n < 500 → a^b ≥ m^n) : a + b = 10 :=
by
  -- Sorry is used to skip the proof steps
  sorry

end greatest_power_sum_l1296_129626


namespace billy_laundry_loads_l1296_129623

-- Define constants based on problem conditions
def sweeping_minutes_per_room := 3
def washing_minutes_per_dish := 2
def laundry_minutes_per_load := 9

def anna_rooms := 10
def billy_dishes := 6

-- Calculate total time spent by Anna and the time Billy spends washing dishes
def anna_total_time := sweeping_minutes_per_room * anna_rooms
def billy_dishwashing_time := washing_minutes_per_dish * billy_dishes

-- Define the time difference Billy needs to make up with laundry
def time_difference := anna_total_time - billy_dishwashing_time
def billy_required_laundry_loads := time_difference / laundry_minutes_per_load

-- The theorem to prove
theorem billy_laundry_loads : billy_required_laundry_loads = 2 := by 
  sorry

end billy_laundry_loads_l1296_129623


namespace digit_sum_subtraction_l1296_129632

theorem digit_sum_subtraction (P Q R S : ℕ) (hQ : Q + P = P) (hP : Q - P = 0) (h1 : P < 10) (h2 : Q < 10) (h3 : R < 10) (h4 : S < 10) : S = 0 := by
  sorry

end digit_sum_subtraction_l1296_129632


namespace arrangements_count_correct_l1296_129670

noncomputable def count_arrangements : ℕ :=
  -- The total number of different arrangements of students A, B, C, D in 3 communities
  -- such that each community has at least one student, and A and B are not in the same community.
  sorry

theorem arrangements_count_correct : count_arrangements = 30 := by
  sorry

end arrangements_count_correct_l1296_129670


namespace chef_cooked_potatoes_l1296_129615

theorem chef_cooked_potatoes
  (total_potatoes : ℕ)
  (cooking_time_per_potato : ℕ)
  (remaining_cooking_time : ℕ)
  (left_potatoes : ℕ)
  (cooked_potatoes : ℕ) :
  total_potatoes = 16 →
  cooking_time_per_potato = 5 →
  remaining_cooking_time = 45 →
  remaining_cooking_time / cooking_time_per_potato = left_potatoes →
  total_potatoes - left_potatoes = cooked_potatoes →
  cooked_potatoes = 7 :=
by
  intros h_total h_cooking_time h_remaining_time h_left_potatoes h_cooked_potatoes
  sorry

end chef_cooked_potatoes_l1296_129615


namespace cut_grid_into_six_polygons_with_identical_pair_l1296_129659

noncomputable def totalCells : Nat := 24
def polygonArea : Nat := 4

theorem cut_grid_into_six_polygons_with_identical_pair :
  ∃ (polygons : Fin 6 → Nat → Prop),
  (∀ i, (∃ (cells : Finset (Fin totalCells)), (cells.card = polygonArea ∧ ∀ c ∈ cells, polygons i c))) ∧
  (∃ i j, i ≠ j ∧ ∀ c, polygons i c ↔ polygons j c) :=
sorry

end cut_grid_into_six_polygons_with_identical_pair_l1296_129659


namespace union_of_A_and_B_l1296_129683

open Set

def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3, 4} :=
  sorry

end union_of_A_and_B_l1296_129683


namespace derivative_at_0_5_l1296_129651

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := -2

-- State the theorem
theorem derivative_at_0_5 : f' 0.5 = -2 :=
by {
  -- Proof placeholder
  sorry
}

end derivative_at_0_5_l1296_129651


namespace symmetric_points_x_axis_l1296_129647

theorem symmetric_points_x_axis (a b : ℝ) (h_a : a = -2) (h_b : b = -1) : a + b = -3 :=
by
  -- Skipping the proof steps and adding sorry
  sorry

end symmetric_points_x_axis_l1296_129647


namespace max_handshakes_l1296_129699

theorem max_handshakes (n : ℕ) (m : ℕ)
  (h_n : n = 25)
  (h_m : m = 20)
  (h_mem : n - m = 5)
  : ∃ (max_handshakes : ℕ), max_handshakes = 250 :=
by
  sorry

end max_handshakes_l1296_129699


namespace range_of_n_l1296_129682

theorem range_of_n (x : ℕ) (n : ℝ) : 
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 5 → x - 2 < n + 3) → ∃ n, 0 < n ∧ n ≤ 1 :=
by
  sorry

end range_of_n_l1296_129682


namespace smallest_munificence_monic_cubic_polynomial_l1296_129636

theorem smallest_munificence_monic_cubic_polynomial :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = x^3 + a * x^2 + b * x + c) ∧
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) ∧
  (∀ (M : ℝ), (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |f x| ≤ M) → M ≥ 1) :=
by
  sorry

end smallest_munificence_monic_cubic_polynomial_l1296_129636


namespace range_of_m_l1296_129661

theorem range_of_m (m : ℝ) :
  (∃! (x : ℤ), (x < 1 ∧ x > m - 1)) →
  (-1 ≤ m ∧ m < 0) :=
by
  sorry

end range_of_m_l1296_129661


namespace train_speed_l1296_129692

theorem train_speed (distance time : ℝ) (h1 : distance = 400) (h2 : time = 10) : 
  distance / time = 40 := 
sorry

end train_speed_l1296_129692


namespace inequality_solution_set_l1296_129698

theorem inequality_solution_set (x : ℝ) :
  ((1 - x) * (x - 3) < 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end inequality_solution_set_l1296_129698


namespace area_of_black_region_l1296_129629

theorem area_of_black_region :
  let side_large := 12
  let side_small := 5
  let area_large := side_large * side_large
  let area_small := side_small * side_small
  let num_smaller_squares := 2
  let total_area_small := num_smaller_squares * area_small
  area_large - total_area_small = 94 :=
by
  let side_large := 12
  let side_small := 5
  let area_large := side_large * side_large
  let area_small := side_small * side_small
  let num_smaller_squares := 2
  let total_area_small := num_smaller_squares * area_small
  sorry

end area_of_black_region_l1296_129629


namespace find_original_revenue_l1296_129601

variable (currentRevenue : ℝ) (percentageDecrease : ℝ)
noncomputable def originalRevenue (currentRevenue : ℝ) (percentageDecrease : ℝ) : ℝ :=
  currentRevenue / (1 - percentageDecrease)

theorem find_original_revenue (h1 : currentRevenue = 48.0) (h2 : percentageDecrease = 0.3333333333333333) :
  originalRevenue currentRevenue percentageDecrease = 72.0 := by
  rw [h1, h2]
  unfold originalRevenue
  norm_num
  sorry

end find_original_revenue_l1296_129601


namespace units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared_l1296_129686

-- Define the cycle of the units digits of powers of 7
def units_digit_of_7_power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1  -- 7^4, 7^8, ...
  | 1 => 7  -- 7^1, 7^5, ...
  | 2 => 9  -- 7^2, 7^6, ...
  | 3 => 3  -- 7^3, 7^7, ...
  | _ => 0  -- unreachable

-- The main theorem to prove
theorem units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared : 
  units_digit_of_7_power (3 ^ (5 ^ 2)) = 3 :=
by
  sorry

end units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared_l1296_129686


namespace fraction_problem_l1296_129681

theorem fraction_problem 
  (x : ℚ)
  (h : x = 45 / (8 - (3 / 7))) : 
  x = 315 / 53 := 
sorry

end fraction_problem_l1296_129681


namespace solution_set_of_inequality_l1296_129668

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + 2 * x - 3 > 0 } = { x : ℝ | x < -3 ∨ x > 1 } :=
sorry

end solution_set_of_inequality_l1296_129668


namespace unique_f_satisfies_eq_l1296_129677

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * (x^2 + 2 * x - 1)

theorem unique_f_satisfies_eq (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, 2 * f x + f (1 - x) = x^2) : 
  ∀ x : ℝ, f x = (1 / 3) * (x^2 + 2 * x - 1) :=
sorry

end unique_f_satisfies_eq_l1296_129677


namespace find_multiple_l1296_129697

-- Definitions based on the conditions provided
def mike_chocolate_squares : ℕ := 20
def jenny_chocolate_squares : ℕ := 65
def extra_squares : ℕ := 5

-- The theorem to prove the multiple
theorem find_multiple : ∃ (multiple : ℕ), jenny_chocolate_squares = mike_chocolate_squares * multiple + extra_squares ∧ multiple = 3 := by
  sorry

end find_multiple_l1296_129697


namespace Mary_takes_3_children_l1296_129695

def num_children (C : ℕ) : Prop :=
  ∃ (C : ℕ), 2 + C = 5

theorem Mary_takes_3_children (C : ℕ) : num_children C → C = 3 :=
by
  intro h
  sorry

end Mary_takes_3_children_l1296_129695


namespace who_is_next_to_boris_l1296_129648

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l1296_129648


namespace area_of_rhombus_l1296_129675

theorem area_of_rhombus (x : ℝ) :
  let d1 := 3 * x + 5
  let d2 := 2 * x + 4
  (d1 * d2) / 2 = 3 * x^2 + 11 * x + 10 :=
by
  let d1 := 3 * x + 5
  let d2 := 2 * x + 4
  have h1 : d1 = 3 * x + 5 := rfl
  have h2 : d2 = 2 * x + 4 := rfl
  simp [h1, h2]
  sorry

end area_of_rhombus_l1296_129675


namespace evaluate_powers_l1296_129674

theorem evaluate_powers : (81^(1/2:ℝ) * 64^(-1/3:ℝ) * 49^(1/4:ℝ) = 9 * (1/4) * Real.sqrt 7) :=
by
  sorry

end evaluate_powers_l1296_129674


namespace first_rocket_height_l1296_129604

theorem first_rocket_height (h : ℝ) (combined_height : ℝ) (second_rocket_height : ℝ) 
  (H1 : second_rocket_height = 2 * h) 
  (H2 : combined_height = h + second_rocket_height) 
  (H3 : combined_height = 1500) : h = 500 := 
by 
  -- The proof would go here but is not required as per the instruction.
  sorry

end first_rocket_height_l1296_129604


namespace area_of_inscribed_triangle_l1296_129603

noncomputable def calculate_triangle_area_inscribed_in_circle 
  (arc1 : ℝ) (arc2 : ℝ) (arc3 : ℝ) (total_circumference := arc1 + arc2 + arc3)
  (radius := total_circumference / (2 * Real.pi))
  (theta := (2 * Real.pi) / total_circumference)
  (angle1 := 5 * theta) (angle2 := 7 * theta) (angle3 := 8 * theta) : ℝ :=
  0.5 * (radius ^ 2) * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem area_of_inscribed_triangle : 
  calculate_triangle_area_inscribed_in_circle 5 7 8 = 119.85 / (Real.pi ^ 2) :=
by
  sorry

end area_of_inscribed_triangle_l1296_129603


namespace problem_l1296_129641

noncomputable def investment : ℝ := 13500
noncomputable def total_yield : ℝ := 19000
noncomputable def orchard_price_per_kg : ℝ := 4
noncomputable def market_price_per_kg (x : ℝ) : ℝ := x
noncomputable def daily_sales_rate_market : ℝ := 1000
noncomputable def days_to_sell_all (yield : ℝ) (rate : ℝ) : ℝ := yield / rate

-- Condition that x > 4
axiom x_gt_4 : ∀ (x : ℝ), x > 4

theorem problem (
  x : ℝ
) (hx : x > 4) : 
  -- Part 1
  days_to_sell_all total_yield daily_sales_rate_market = 19 ∧
  -- Part 2
  (total_yield * market_price_per_kg x - total_yield * orchard_price_per_kg) = 19000 * x - 76000 ∧
  -- Part 3
  (6000 * orchard_price_per_kg + (total_yield - 6000) * x - investment) = 13000 * x + 10500 :=
by sorry

end problem_l1296_129641


namespace geometric_sequence_a7_l1296_129665

noncomputable def a (n : ℕ) : ℝ := sorry -- Definition of the sequence

theorem geometric_sequence_a7 :
  a 3 = 1 → a 11 = 25 → a 7 = 5 := 
by
  intros h3 h11
  sorry

end geometric_sequence_a7_l1296_129665


namespace journey_speed_l1296_129652

theorem journey_speed (t_total : ℝ) (d_total : ℝ) (d_half : ℝ) (v_half2 : ℝ) (time_half2 : ℝ) (time_total : ℝ) (v_half1 : ℝ) :
  t_total = 5 ∧ d_total = 112 ∧ d_half = d_total / 2 ∧ v_half2 = 24 ∧ time_half2 = d_half / v_half2 ∧ time_total = t_total - time_half2 ∧ v_half1 = d_half / time_total → v_half1 = 21 :=
by
  intros h
  sorry

end journey_speed_l1296_129652


namespace num_real_x_l1296_129637

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end num_real_x_l1296_129637


namespace expenditure_on_house_rent_l1296_129639

theorem expenditure_on_house_rent
  (income petrol house_rent remaining_income : ℝ)
  (h1 : petrol = 0.30 * income)
  (h2 : petrol = 300)
  (h3 : remaining_income = income - petrol)
  (h4 : house_rent = 0.30 * remaining_income) :
  house_rent = 210 :=
by
  sorry

end expenditure_on_house_rent_l1296_129639


namespace line_equation_is_correct_l1296_129673

noncomputable def line_has_equal_intercepts_and_passes_through_A (p q : ℝ) : Prop :=
(p, q) = (3, 2) ∧ q ≠ 0 ∧ (∃ c : ℝ, p + q = c ∨ 2 * p - 3 * q = 0)

theorem line_equation_is_correct :
  line_has_equal_intercepts_and_passes_through_A 3 2 → 
  (∃ f g : ℝ, 2 * f - 3 * g = 0 ∨ f + g = 5) :=
by
  sorry

end line_equation_is_correct_l1296_129673


namespace min_value_f_min_value_achieved_l1296_129638

noncomputable def f (x y : ℝ) : ℝ :=
  (x^4 / y^4) + (y^4 / x^4) - (x^2 / y^2) - (y^2 / x^2) + (x / y) + (y / x)

theorem min_value_f :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) → f x y ≥ 2 :=
sorry

theorem min_value_achieved :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) → (f x y = 2) ↔ (x = y) :=
sorry

end min_value_f_min_value_achieved_l1296_129638


namespace smaller_side_of_new_rectangle_is_10_l1296_129610

/-- We have a 10x25 rectangle that is divided into two congruent polygons and rearranged 
to form another rectangle. We need to prove that the length of the smaller side of the 
resulting rectangle is 10. -/
theorem smaller_side_of_new_rectangle_is_10 :
  ∃ (y x : ℕ), (y * x = 10 * 25) ∧ (y ≤ x) ∧ y = 10 := 
sorry

end smaller_side_of_new_rectangle_is_10_l1296_129610


namespace range_of_m_for_circle_l1296_129645

theorem range_of_m_for_circle (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + m*x - 2*y + 4 = 0)  ↔ m < -2*Real.sqrt 3 ∨ m > 2*Real.sqrt 3 :=
by 
  sorry

end range_of_m_for_circle_l1296_129645


namespace weight_of_e_l1296_129622

variables (d e f : ℝ)

theorem weight_of_e
  (h_de_f : (d + e + f) / 3 = 42)
  (h_de : (d + e) / 2 = 35)
  (h_ef : (e + f) / 2 = 41) :
  e = 26 :=
by
  sorry

end weight_of_e_l1296_129622


namespace find_value_of_M_l1296_129676

variable {C y M A : ℕ}

theorem find_value_of_M (h1 : C + y + 2 * M + A = 11)
                        (h2 : C ≠ y)
                        (h3 : C ≠ M)
                        (h4 : C ≠ A)
                        (h5 : y ≠ M)
                        (h6 : y ≠ A)
                        (h7 : M ≠ A)
                        (h8 : 0 < C)
                        (h9 : 0 < y)
                        (h10 : 0 < M)
                        (h11 : 0 < A) : M = 1 :=
by
  sorry

end find_value_of_M_l1296_129676


namespace one_fourth_of_eight_times_x_plus_two_l1296_129605

theorem one_fourth_of_eight_times_x_plus_two (x : ℝ) : 
  (1 / 4) * (8 * x + 2) = 2 * x + 1 / 2 :=
by
  sorry

end one_fourth_of_eight_times_x_plus_two_l1296_129605


namespace worker_b_alone_time_l1296_129627

theorem worker_b_alone_time (A B C : ℝ) (h1 : A + B = 1 / 8)
  (h2 : A = 1 / 12) (h3 : C = 1 / 18) :
  1 / B = 24 :=
sorry

end worker_b_alone_time_l1296_129627


namespace find_x_l1296_129662

variable (x : ℝ)
variable (h : 0.3 * 100 = 0.5 * x + 10)

theorem find_x : x = 40 :=
by
  sorry

end find_x_l1296_129662


namespace expression_equals_4008_l1296_129625

def calculate_expression : ℤ :=
  let expr := (2004 - (2011 - 196)) + (2011 - (196 - 2004))
  expr

theorem expression_equals_4008 : calculate_expression = 4008 := 
by
  sorry

end expression_equals_4008_l1296_129625


namespace solve_for_x_l1296_129663

theorem solve_for_x :
  ∀ x : ℕ, 100^4 = 5^x → x = 8 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l1296_129663


namespace Jeffrey_steps_l1296_129631

theorem Jeffrey_steps
  (Andrew_steps : ℕ) (Jeffrey_steps : ℕ) (h_ratio : Andrew_steps / Jeffrey_steps = 3 / 4)
  (h_Andrew : Andrew_steps = 150) :
  Jeffrey_steps = 200 :=
by
  sorry

end Jeffrey_steps_l1296_129631


namespace range_fraction_l1296_129642

theorem range_fraction {x y : ℝ} (h : x^2 + y^2 + 2 * x = 0) :
  ∃ a b : ℝ, a = -1 ∧ b = 1 / 3 ∧ ∀ z, z = (y - x) / (x - 1) → a ≤ z ∧ z ≤ b :=
by 
  sorry

end range_fraction_l1296_129642


namespace deposit_percentage_l1296_129667

-- Define the conditions of the problem
def amount_deposited : ℕ := 5000
def monthly_income : ℕ := 25000

-- Define the percentage deposited formula
def percentage_deposited (amount_deposited monthly_income : ℕ) : ℚ :=
  (amount_deposited / monthly_income) * 100

-- State the theorem to be proved
theorem deposit_percentage :
  percentage_deposited amount_deposited monthly_income = 20 := by
  sorry

end deposit_percentage_l1296_129667


namespace largest_unshaded_area_l1296_129690

theorem largest_unshaded_area (s : ℝ) (π_approx : ℝ) :
    (let r := s / 2
     let area_square := s^2
     let area_circle := π_approx * r^2
     let area_triangle := (1 / 2) * (s / 2) * (s / 2)
     let unshaded_square := area_square - area_circle
     let unshaded_circle := area_circle - area_triangle
     unshaded_circle) > (unshaded_square) := by
        sorry

end largest_unshaded_area_l1296_129690


namespace polynomial_coefficient_B_l1296_129687

theorem polynomial_coefficient_B : 
  ∃ (A C D : ℤ), 
    (∀ z : ℤ, (z > 0) → (z^6 - 15 * z^5 + A * z^4 + B * z^3 + C * z^2 + D * z + 64 = 0)) ∧ 
    (B = -244) := 
by
  sorry

end polynomial_coefficient_B_l1296_129687


namespace max_profit_at_grade_9_l1296_129679

def profit (k : ℕ) : ℕ :=
  (8 + 2 * (k - 1)) * (60 - 3 * (k - 1))

theorem max_profit_at_grade_9 : ∀ k, 1 ≤ k ∧ k ≤ 10 → profit k ≤ profit 9 := 
by
  sorry

end max_profit_at_grade_9_l1296_129679


namespace find_number_of_students_l1296_129650

-- Definitions for the conditions
def avg_age_students := 14
def teacher_age := 65
def new_avg_age := 15

-- The total age of students is n multiplied by their average age
def total_age_students (n : ℕ) := n * avg_age_students

-- The total age including teacher
def total_age_incl_teacher (n : ℕ) := total_age_students n + teacher_age

-- The new average age when teacher is included
def new_avg_age_incl_teacher (n : ℕ) := total_age_incl_teacher n / (n + 1)

theorem find_number_of_students (n : ℕ) (h₁ : avg_age_students = 14) (h₂ : teacher_age = 65) (h₃ : new_avg_age = 15) 
  (h_averages_eq : new_avg_age_incl_teacher n = new_avg_age) : n = 50 :=
  sorry

end find_number_of_students_l1296_129650


namespace triangle_obtuse_of_inequality_l1296_129646

theorem triangle_obtuse_of_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (ineq : a^2 < (b + c) * (c - b)) :
  ∃ (A B C : ℝ), (A + B + C = π) ∧ (C > π / 2) :=
by
  sorry

end triangle_obtuse_of_inequality_l1296_129646


namespace finite_parabolas_do_not_cover_plane_l1296_129608

theorem finite_parabolas_do_not_cover_plane (parabolas : Finset (ℝ → ℝ)) :
  ¬ (∀ x y : ℝ, ∃ p ∈ parabolas, y < p x) :=
by sorry

end finite_parabolas_do_not_cover_plane_l1296_129608


namespace circumcircle_diameter_triangle_ABC_l1296_129613

theorem circumcircle_diameter_triangle_ABC
  (A : ℝ) (BC : ℝ) (R : ℝ)
  (hA : A = 60) (hBC : BC = 4)
  (hR_formula : 2 * R = BC / Real.sin (A * Real.pi / 180)) :
  2 * R = 8 * Real.sqrt 3 / 3 :=
by
  sorry

end circumcircle_diameter_triangle_ABC_l1296_129613


namespace P_is_necessary_but_not_sufficient_for_Q_l1296_129691

def P (x : ℝ) : Prop := |x - 1| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

theorem P_is_necessary_but_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) :=
by
  sorry

end P_is_necessary_but_not_sufficient_for_Q_l1296_129691


namespace find_n_l1296_129656

-- Define the first term a₁, the common ratio q, and the sum Sₙ
def a₁ : ℕ := 2
def q : ℕ := 2
def Sₙ (n : ℕ) : ℕ := 2^(n + 1) - 2

-- The sum of the first n terms is given as 126
def given_sum : ℕ := 126

-- The theorem to be proven
theorem find_n (n : ℕ) (h : Sₙ n = given_sum) : n = 6 :=
by
  sorry

end find_n_l1296_129656


namespace harkamal_total_amount_l1296_129609

-- Define the conditions as constants
def quantity_grapes : ℕ := 10
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost of grapes and mangoes based on the given conditions
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Define the total amount paid
def total_amount_paid : ℕ := cost_grapes + cost_mangoes

-- The theorem stating the problem and the solution
theorem harkamal_total_amount : total_amount_paid = 1195 := by
  -- Proof goes here (omitted)
  sorry

end harkamal_total_amount_l1296_129609


namespace compute_xy_l1296_129635

theorem compute_xy (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 62) : 
  xy = -126 / 25 ∨ xy = -6 := 
sorry

end compute_xy_l1296_129635


namespace number_of_digits_in_x20_l1296_129633

theorem number_of_digits_in_x20 (x : ℝ) (hx1 : 10^(7/4) ≤ x) (hx2 : x < 10^2) :
  10^35 ≤ x^20 ∧ x^20 < 10^36 :=
by
  -- Proof goes here
  sorry

end number_of_digits_in_x20_l1296_129633


namespace Joan_spent_on_shirt_l1296_129643

/-- Joan spent $15 on shorts, $14.82 on a jacket, and a total of $42.33 on clothing.
    Prove that Joan spent $12.51 on the shirt. -/
theorem Joan_spent_on_shirt (shorts jacket total: ℝ) 
                            (h1: shorts = 15)
                            (h2: jacket = 14.82)
                            (h3: total = 42.33) :
  total - (shorts + jacket) = 12.51 :=
by
  sorry

end Joan_spent_on_shirt_l1296_129643


namespace complex_transformation_result_l1296_129612

theorem complex_transformation_result :
  let z := -1 - 2 * Complex.I 
  let rotation := (1 / 2 : ℂ) + (Complex.I * (Real.sqrt 3) / 2)
  let dilation := 2
  (z * (rotation * dilation)) = (2 * Real.sqrt 3 - 1 - (2 + Real.sqrt 3) * Complex.I) :=
by
  sorry

end complex_transformation_result_l1296_129612


namespace smallest_n_for_candy_l1296_129696

theorem smallest_n_for_candy (r g b n : ℕ) (h1 : 10 * r = 18 * g) (h2 : 18 * g = 20 * b) (h3 : 20 * b = 24 * n) : n = 15 :=
by
  sorry

end smallest_n_for_candy_l1296_129696


namespace isosceles_triangle_perimeter_l1296_129655

-- Define the lengths of the sides
def side1 : ℕ := 4
def side2 : ℕ := 7

-- Condition: The given sides form an isosceles triangle
def is_isosceles_triangle (a b : ℕ) : Prop := a = b ∨ a = 4 ∧ b = 7 ∨ a = 7 ∧ b = 4

-- Condition: The triangle inequality theorem must be satisfied
def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem we want to prove
theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : is_isosceles_triangle a b) (h2 : triangle_inequality a a b ∨ triangle_inequality b b a) :
  a + a + b = 15 ∨ b + b + a = 18 := 
sorry

end isosceles_triangle_perimeter_l1296_129655


namespace abs_inequality_solution_l1296_129611

theorem abs_inequality_solution (x : ℝ) :
  abs (2 * x - 5) ≤ 7 ↔ -1 ≤ x ∧ x ≤ 6 :=
sorry

end abs_inequality_solution_l1296_129611


namespace ratio_of_55_to_11_l1296_129607

theorem ratio_of_55_to_11 : (55 / 11) = 5 := 
by
  sorry

end ratio_of_55_to_11_l1296_129607


namespace cylinder_volume_l1296_129634

theorem cylinder_volume (V1 V2 : ℝ) (π : ℝ) (r1 r3 h2 h5 : ℝ)
  (h_radii_ratio : r3 = 3 * r1)
  (h_heights_ratio : h5 = 5 / 2 * h2)
  (h_first_volume : V1 = π * r1^2 * h2)
  (h_V1_value : V1 = 40) :
  V2 = 900 :=
by sorry

end cylinder_volume_l1296_129634


namespace no_such_functions_exist_l1296_129666

theorem no_such_functions_exist :
  ¬ ∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y + 1 :=
by
  sorry

end no_such_functions_exist_l1296_129666


namespace percentage_of_failed_candidates_l1296_129628

theorem percentage_of_failed_candidates
(total_candidates : ℕ)
(girls : ℕ)
(passed_boys_percentage : ℝ)
(passed_girls_percentage : ℝ)
(h1 : total_candidates = 2000)
(h2 : girls = 900)
(h3 : passed_boys_percentage = 0.28)
(h4 : passed_girls_percentage = 0.32)
: (total_candidates - (passed_boys_percentage * (total_candidates - girls) + passed_girls_percentage * girls)) / total_candidates * 100 = 70.2 :=
by
  sorry

end percentage_of_failed_candidates_l1296_129628


namespace min_value_expression_l1296_129688

theorem min_value_expression (x : ℝ) (h : x > 1) : 
  ∃ min_val, min_val = 6 ∧ ∀ y > 1, 2 * y + 2 / (y - 1) ≥ min_val :=
by  
  use 6
  sorry

end min_value_expression_l1296_129688


namespace least_common_multiple_of_5_to_10_is_2520_l1296_129684

-- Definitions of the numbers
def numbers : List ℤ := [5, 6, 7, 8, 9, 10]

-- Definition of prime factorization for verification (optional, keeping it simple)
def prime_factors (n : ℤ) : List ℤ :=
  if n = 5 then [5]
  else if n = 6 then [2, 3]
  else if n = 7 then [7]
  else if n = 8 then [2, 2, 2]
  else if n = 9 then [3, 3]
  else if n = 10 then [2, 5]
  else []

-- The property to be proved: The least common multiple of numbers is 2520
theorem least_common_multiple_of_5_to_10_is_2520 : ∃ n : ℕ, (∀ m ∈ numbers, m ∣ n) ∧ n = 2520 := by
  use 2520
  sorry

end least_common_multiple_of_5_to_10_is_2520_l1296_129684


namespace cos_squared_sum_sin_squared_sum_l1296_129685

theorem cos_squared_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.cos (A / 2) ^ 2 + Real.cos (B / 2) ^ 2 + Real.cos (C / 2) ^ 2 =
  2 * (1 + Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2)) :=
sorry

theorem sin_squared_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (A / 2) ^ 2 + Real.sin (B / 2) ^ 2 + Real.sin (C / 2) ^ 2 =
  1 - 2 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) :=
sorry

end cos_squared_sum_sin_squared_sum_l1296_129685
