import Mathlib

namespace square_of_negative_eq_square_l148_148219

theorem square_of_negative_eq_square (a : ℝ) : (-a)^2 = a^2 :=
sorry

end square_of_negative_eq_square_l148_148219


namespace factorize_x_squared_minus_one_l148_148051

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l148_148051


namespace compute_value_l148_148021

theorem compute_value : (7^2 - 6^2)^3 = 2197 := by
  sorry

end compute_value_l148_148021


namespace missing_files_correct_l148_148501

def total_files : ℕ := 60
def files_in_morning : ℕ := total_files / 2
def files_in_afternoon : ℕ := 15
def missing_files : ℕ := total_files - (files_in_morning + files_in_afternoon)

theorem missing_files_correct : missing_files = 15 := by
  sorry

end missing_files_correct_l148_148501


namespace fair_hair_percentage_l148_148613

-- Define the main entities
variables (E F W : ℝ)

-- Define the conditions given in the problem
def women_with_fair_hair : Prop := W = 0.32 * E
def fair_hair_women_ratio : Prop := W = 0.40 * F

-- Define the theorem to prove
theorem fair_hair_percentage
  (hwf: women_with_fair_hair E W)
  (fhr: fair_hair_women_ratio W F) :
  (F / E) * 100 = 80 :=
by
  sorry

end fair_hair_percentage_l148_148613


namespace factorize_difference_of_squares_l148_148062

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l148_148062


namespace tangent_line_eqn_max_interval_monotonic_increase_l148_148092

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x - x^2

theorem tangent_line_eqn (a : ℝ) (h_a : 0 < a ∧ a ≤ 1) :
  tangent_line (f a) 1 = -1 / 2 * x := sorry

theorem max_interval_monotonic_increase (a : ℝ) (h_a : 0 < a ∧ a ≤ 1) :
  let t := (a + sqrt (a^2 + 8)) / 4
  in 0 < t ∧ t = 1 ↔ a = 1 := sorry

end tangent_line_eqn_max_interval_monotonic_increase_l148_148092


namespace pigeonhole_principle_f_m_l148_148553

theorem pigeonhole_principle_f_m :
  ∀ (n : ℕ) (f : ℕ × ℕ → Fin (n + 1)), n ≤ 44 →
    ∃ (i j l k p m : ℕ),
      1989 * m ≤ i ∧ i < l ∧ l < 1989 + 1989 * m ∧
      1989 * p ≤ j ∧ j < k ∧ k < 1989 + 1989 * p ∧
      f (i, j) = f (i, k) ∧ f (i, k) = f (l, j) ∧ f (l, j) = f (l, k) :=
by {
  sorry
}

end pigeonhole_principle_f_m_l148_148553


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l148_148720

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l148_148720


namespace six_digit_numbers_with_zero_l148_148743

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148743


namespace hog_cat_problem_l148_148798

theorem hog_cat_problem (hogs cats : ℕ)
  (hogs_eq : hogs = 75)
  (hogs_cats_relation : hogs = 3 * cats)
  : 5 < (6 / 10) * cats - 5 := 
by
  sorry

end hog_cat_problem_l148_148798


namespace no_common_points_iff_parallel_l148_148012

-- Definitions based on conditions:
def line (a : Type) : Prop := sorry
def plane (M : Type) : Prop := sorry
def no_common_points (a : Type) (M : Type) : Prop := sorry
def parallel (a : Type) (M : Type) : Prop := sorry

-- Theorem stating the relationship is necessary and sufficient
theorem no_common_points_iff_parallel (a M : Type) :
  no_common_points a M ↔ parallel a M := sorry

end no_common_points_iff_parallel_l148_148012


namespace percentage_of_loss_l148_148345

theorem percentage_of_loss
    (CP SP : ℝ)
    (h1 : CP = 1200)
    (h2 : SP = 1020)
    (Loss : ℝ)
    (h3 : Loss = CP - SP)
    (Percentage_of_Loss : ℝ)
    (h4 : Percentage_of_Loss = (Loss / CP) * 100) :
  Percentage_of_Loss = 15 := by
  sorry

end percentage_of_loss_l148_148345


namespace carpets_triple_overlap_area_l148_148449

theorem carpets_triple_overlap_area {W H : ℕ} (hW : W = 10) (hH : H = 10) 
    {w1 h1 w2 h2 w3 h3 : ℕ} 
    (h1_w1 : w1 = 6) (h1_h1 : h1 = 8)
    (h2_w2 : w2 = 6) (h2_h2 : h2 = 6)
    (h3_w3 : w3 = 5) (h3_h3 : h3 = 7) :
    ∃ (area : ℕ), area = 6 := by
  sorry

end carpets_triple_overlap_area_l148_148449


namespace max_colors_4x4_grid_l148_148178

def cell := (Fin 4) × (Fin 4)
def color := Fin 8

def valid_coloring (f : cell → color) : Prop :=
∀ c1 c2 : color, (c1 ≠ c2) →
(∃ i : Fin 4, ∃ j1 j2 : Fin 4, j1 ≠ j2 ∧ f (i, j1) = c1 ∧ f (i, j2) = c2) ∨ 
(∃ j : Fin 4, ∃ i1 i2 : Fin 4, i1 ≠ i2 ∧ f (i1, j) = c1 ∧ f (i2, j) = c2)

theorem max_colors_4x4_grid : ∃ (f : cell → color), valid_coloring f :=
sorry

end max_colors_4x4_grid_l148_148178


namespace moose_population_l148_148892

theorem moose_population (B M H : ℕ) (h1 : B = 2 * M) (h2 : H = 19 * B) (h3 : H = 38_000_000) : M = 1_000_000 :=
by sorry

end moose_population_l148_148892


namespace fire_fighting_max_saved_houses_l148_148897

noncomputable def max_houses_saved (n c : ℕ) : ℕ :=
  n^2 + c^2 - n * c - c

theorem fire_fighting_max_saved_houses (n c : ℕ) (h : c ≤ n / 2) :
    ∃ k, k = max_houses_saved n c :=
    sorry

end fire_fighting_max_saved_houses_l148_148897


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l148_148721

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l148_148721


namespace critical_temperature_of_water_l148_148921

/--
Given the following conditions:
1. The temperature at which solid, liquid, and gaseous water coexist is the triple point.
2. The temperature at which water vapor condenses is the condensation point.
3. The maximum temperature at which liquid water can exist.
4. The minimum temperature at which water vapor can exist.

Prove that the critical temperature of water is the maximum temperature at which liquid water can exist.
-/
theorem critical_temperature_of_water :
    ∀ (triple_point condensation_point maximum_liquid_temp minimum_vapor_temp critical_temp : ℝ), 
    (critical_temp = maximum_liquid_temp) ↔
    ((critical_temp ≠ triple_point) ∧ (critical_temp ≠ condensation_point) ∧ (critical_temp ≠ minimum_vapor_temp)) := 
  sorry

end critical_temperature_of_water_l148_148921


namespace trains_at_initial_stations_l148_148792

-- Define the durations of round trips for each line.
def red_round_trip : ℕ := 14
def blue_round_trip : ℕ := 16
def green_round_trip : ℕ := 18

-- Define the total time we are analyzing.
def total_time : ℕ := 2016

-- Define the statement that needs to be proved.
theorem trains_at_initial_stations : 
  (total_time % red_round_trip = 0) ∧ 
  (total_time % blue_round_trip = 0) ∧ 
  (total_time % green_round_trip = 0) := 
by
  -- The proof can be added here.
  sorry

end trains_at_initial_stations_l148_148792


namespace smallest_b_greater_than_5_perfect_cube_l148_148327

theorem smallest_b_greater_than_5_perfect_cube : ∃ b : ℕ, b > 5 ∧ ∃ n : ℕ, 4 * b + 3 = n ^ 3 ∧ b = 6 := 
by 
  sorry

end smallest_b_greater_than_5_perfect_cube_l148_148327


namespace factorize_difference_of_squares_l148_148047

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l148_148047


namespace smallest_k_for_sum_of_squares_multiple_of_360_l148_148859

theorem smallest_k_for_sum_of_squares_multiple_of_360 :
  ∃ k : ℕ, k > 0 ∧ (k * (k + 1) * (2 * k + 1)) / 6 % 360 = 0 ∧ ∀ n : ℕ, n > 0 → (n * (n + 1) * (2 * n + 1)) / 6 % 360 = 0 → k ≤ n :=
by sorry

end smallest_k_for_sum_of_squares_multiple_of_360_l148_148859


namespace missing_files_correct_l148_148500

def total_files : ℕ := 60
def files_in_morning : ℕ := total_files / 2
def files_in_afternoon : ℕ := 15
def missing_files : ℕ := total_files - (files_in_morning + files_in_afternoon)

theorem missing_files_correct : missing_files = 15 := by
  sorry

end missing_files_correct_l148_148500


namespace six_digit_numbers_with_zero_l148_148713

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l148_148713


namespace sum_of_all_possible_values_of_f_l148_148861

-- Define f as counting the valid (x, y) pairs
def f (a b c d : ℤ) : ℕ :=
  let candidates := finset.univ.product finset.univ in
  (candidates.filter (λ (xy : fin (5) × fin (5)),
    let x := xy.1.1 + 1 in
    let y := xy.2.1 + 1 in
    (a * x + b * y) % 5 = 0 ∧ (c * x + d * y) % 5 = 0)).card

-- Prove that the sum of all possible values of f(a, b, c, d) is 31
theorem sum_of_all_possible_values_of_f : 
  ∀ (a b c d : ℤ), f a b c d = 31 :=
by
  sorry

end sum_of_all_possible_values_of_f_l148_148861


namespace harriet_return_speed_l148_148809

noncomputable def harriet_speed_back_to_aville (speed_to_bt : ℝ) (time_to_bt_aville_hours : ℝ) (total_trip_hours : ℝ) :=
  let distance := speed_to_bt * time_to_bt_aville_hours
  let time_back_hours := total_trip_hours - time_to_bt_aville_hours
  let speed_back := distance / time_back_hours
  speed_back

theorem harriet_return_speed :
  harriet_speed_back_to_aville 90 3.2 5 = 160 :=
begin
  -- Given:
  -- speed_to_bt = 90 km/hr
  -- time_to_bt_aville_hours = 3.2 hours
  -- total_trip_hours = 5 hours

  -- The function will calculate the distance as:
  -- distance = 90 * 3.2 = 288 km

  -- The return time:
  -- time_back_hours = 5 - 3.2 = 1.8 hours

  -- The speed back:
  -- speed_back = 288 / 1.8 = 160 km/hr
  
  -- Therefore, the result should be:
  refl,
end

end harriet_return_speed_l148_148809


namespace eugene_payment_correct_l148_148270

theorem eugene_payment_correct :
  let t_price := 20
  let p_price := 80
  let s_price := 150
  let discount_rate := 0.1
  let t_quantity := 4
  let p_quantity := 3
  let s_quantity := 2
  let t_cost := t_quantity * t_price
  let p_cost := p_quantity * p_price
  let s_cost := s_quantity * s_price
  let total_cost := t_cost + p_cost + s_cost
  let discount := discount_rate * total_cost
  let final_cost := total_cost - discount
  final_cost = 558 :=
by
  sorry

end eugene_payment_correct_l148_148270


namespace find_f_x_minus_1_l148_148074

theorem find_f_x_minus_1 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x ^ 2 + 2 * x) :
  ∀ x : ℤ, f (x - 1) = x ^ 2 - 2 * x :=
by
  sorry

end find_f_x_minus_1_l148_148074


namespace find_length_of_c_find_measure_of_B_l148_148756

-- Definition of the conditions
def triangle (A B C a b c : ℝ) : Prop :=
  c - b = 2 * b * Real.cos A

noncomputable def value_c (a b : ℝ) : ℝ := sorry

noncomputable def value_B (A B : ℝ) : ℝ := sorry

-- Statement for problem (I)
theorem find_length_of_c (a b : ℝ) (h1 : a = 2 * Real.sqrt 6) (h2 : b = 3) (h3 : ∀ A B C, triangle A B C a b (value_c a b)) : 
  value_c a b = 5 :=
by 
  sorry

-- Statement for problem (II)
theorem find_measure_of_B (B : ℝ) (h1 : ∀ A, A + B = Real.pi / 2) (h2 : B = value_B A B) : 
  value_B A B = Real.pi / 6 :=
by 
  sorry

end find_length_of_c_find_measure_of_B_l148_148756


namespace problem_l148_148165

theorem problem 
  (k a b c : ℝ)
  (h1 : (3 : ℝ)^2 - 7 * 3 + k = 0)
  (h2 : (a : ℝ)^2 - 7 * a + k = 0)
  (h3 : (b : ℝ)^2 - 8 * b + (k + 1) = 0)
  (h4 : (c : ℝ)^2 - 8 * c + (k + 1) = 0) :
  a + b * c = 17 := sorry

end problem_l148_148165


namespace angle_covered_in_three_layers_l148_148336

theorem angle_covered_in_three_layers 
  (total_coverage : ℝ) (sum_of_angles : ℝ) 
  (h1 : total_coverage = 90) (h2 : sum_of_angles = 290) : 
  ∃ x : ℝ, 3 * x + 2 * (90 - x) = 290 ∧ x = 20 :=
by
  sorry

end angle_covered_in_three_layers_l148_148336


namespace solve_for_x_l148_148010

theorem solve_for_x
  (x y : ℝ)
  (h1 : x + 2 * y = 100)
  (h2 : y = 25) :
  x = 50 :=
by
  sorry

end solve_for_x_l148_148010


namespace bus_speed_l148_148351

theorem bus_speed (x y z : ℕ) (hx : x < 10) (hy : y < 10) (hz : z < 10)
    (h1 : 9 * (11 * y - x) = 5 * z)
    (h2 : z = 9) :
    ∀ speed, speed = 45 :=
by
  sorry

end bus_speed_l148_148351


namespace plastering_cost_correct_l148_148624

def length : ℕ := 40
def width : ℕ := 18
def depth : ℕ := 10
def cost_per_sq_meter : ℚ := 1.25

def area_bottom (L W : ℕ) : ℕ := L * W
def perimeter_bottom (L W : ℕ) : ℕ := 2 * (L + W)
def area_walls (P D : ℕ) : ℕ := P * D
def total_area (A_bottom A_walls : ℕ) : ℕ := A_bottom + A_walls
def total_cost (A_total : ℕ) (cost_per_sq_meter : ℚ) : ℚ := A_total * cost_per_sq_meter

theorem plastering_cost_correct :
  total_cost (total_area (area_bottom length width)
                        (area_walls (perimeter_bottom length width) depth))
             cost_per_sq_meter = 2350 :=
by 
  sorry

end plastering_cost_correct_l148_148624


namespace biased_coin_probability_l148_148598

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability mass function for a binomial distribution
def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ :=
  binomial n k * p^k * (1 - p)^(n - k)

-- Define the problem conditions
def problem_conditions : Prop :=
  let p := 1 / 3
  binomial_pmf 5 1 p = binomial_pmf 5 2 p ∧ p ≠ 0 ∧ (1 - p) ≠ 0

-- The target probability to prove
def target_probability := 40 / 243

-- The theorem statement
theorem biased_coin_probability : problem_conditions → binomial_pmf 5 3 (1 / 3) = target_probability :=
by
  intro h
  sorry

end biased_coin_probability_l148_148598


namespace six_digit_numbers_with_at_least_one_zero_l148_148661

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l148_148661


namespace tangent_subtraction_identity_l148_148995

theorem tangent_subtraction_identity (α β : ℝ) 
  (h1 : Real.tan α = -3/4) 
  (h2 : Real.tan (Real.pi - β) = 1/2) : 
  Real.tan (α - β) = -2/11 := 
sorry

end tangent_subtraction_identity_l148_148995


namespace evaluate_expression_l148_148853

theorem evaluate_expression : (-(18 / 3 * 12 - 80 + 4 * 12)) ^ 2 = 1600 := by
  sorry

end evaluate_expression_l148_148853


namespace real_b_values_for_non_real_roots_l148_148879

theorem real_b_values_for_non_real_roots (b : ℝ) :
  let discriminant := b^2 - 4 * 1 * 16
  discriminant < 0 ↔ -8 < b ∧ b < 8 := 
sorry

end real_b_values_for_non_real_roots_l148_148879


namespace trigonometric_identity_l148_148880

open Real

noncomputable def acute (x : ℝ) := 0 < x ∧ x < π / 2

theorem trigonometric_identity 
  {α β : ℝ} (hα : acute α) (hβ : acute β) (h : cos α > sin β) :
  α + β < π / 2 :=
sorry

end trigonometric_identity_l148_148880


namespace min_value_of_sum_of_squares_l148_148137

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4.8 :=
sorry

end min_value_of_sum_of_squares_l148_148137


namespace find_angle_A_find_area_of_ABC_l148_148535

variables {a b c : ℝ} {A B C : ℝ} {AB AC : ℝ}

-- Condition: b * cos C + c * cos B = 2 * a * cos A
axiom cond1 : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos A

-- Condition: The dot product \overrightarrow{AB} \cdot \overrightarrow{AC} = sqrt(3)
axiom cond2 : AB * AC * Real.cos A = sqrt 3

-- Proof requirement: ∠A = π/3
theorem find_angle_A (h : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos A) : A = π / 3 := 
sorry

-- Proof requirement: Area of ΔABC = 3/2
theorem find_area_of_ABC (h : AB * AC * Real.cos A = sqrt 3) : 
    let angle_A := π / 3 in -- Using the previously found angle A
    1/2 * AB * AC * abs (Real.sin angle_A) = 3 / 2 := 
sorry

end find_angle_A_find_area_of_ABC_l148_148535


namespace sasha_tree_planting_cost_l148_148300

theorem sasha_tree_planting_cost :
  ∀ (initial_temperature final_temperature : ℝ)
    (temp_drop_per_tree : ℝ) (cost_per_tree : ℝ)
    (temperature_drop : ℝ) (num_trees : ℕ)
    (total_cost : ℝ),
    initial_temperature = 80 →
    final_temperature = 78.2 →
    temp_drop_per_tree = 0.1 →
    cost_per_tree = 6 →
    temperature_drop = initial_temperature - final_temperature →
    num_trees = temperature_drop / temp_drop_per_tree →
    total_cost = num_trees * cost_per_tree →
    total_cost = 108 :=
by
  intros initial_temperature final_temperature temp_drop_per_tree
    cost_per_tree temperature_drop num_trees total_cost
    h_initial h_final h_drop_tree h_cost_tree
    h_temp_drop h_num_trees h_total_cost
  rw [h_initial, h_final] at h_temp_drop
  rw [h_temp_drop] at h_num_trees
  rw [h_num_trees] at h_total_cost
  rw [h_drop_tree] at h_total_cost
  rw [h_cost_tree] at h_total_cost
  norm_num at h_total_cost
  exact h_total_cost

end sasha_tree_planting_cost_l148_148300


namespace factorize_x_squared_minus_one_l148_148053

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l148_148053


namespace solve_inequality_l148_148381

theorem solve_inequality : {x : ℝ | (2 * x - 7) * (x - 3) / x ≥ 0} = {x | (0 < x ∧ x ≤ 3) ∨ (x ≥ 7 / 2)} :=
by
  sorry

end solve_inequality_l148_148381


namespace roots_sum_l148_148087

theorem roots_sum (a b : ℝ) 
  (h₁ : 3^(a-1) = 6 - a)
  (h₂ : 3^(6-b) = b - 1) : 
  a + b = 7 := 
by sorry

end roots_sum_l148_148087


namespace algebraic_expression_1_algebraic_expression_2_l148_148070

-- Problem 1
theorem algebraic_expression_1 (a : ℚ) (h : a = 4 / 5) : -24.7 * a + 1.3 * a - (33 / 5) * a = -24 := 
by 
  sorry

-- Problem 2
theorem algebraic_expression_2 (a b : ℕ) (ha : a = 899) (hb : b = 101) : a^2 + 2 * a * b + b^2 = 1000000 := 
by 
  sorry

end algebraic_expression_1_algebraic_expression_2_l148_148070


namespace cards_drawn_to_product_even_l148_148037

theorem cards_drawn_to_product_even :
  ∃ n, (∀ (cards_drawn : Finset ℕ), 
    (cards_drawn ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}) ∧
    (cards_drawn.card = n) → 
    ¬ (∀ c ∈ cards_drawn, c % 2 = 1)
  ) ∧ n = 8 :=
by
  sorry

end cards_drawn_to_product_even_l148_148037


namespace quadratic_condition_l148_148746

theorem quadratic_condition (m : ℤ) (x : ℝ) :
  (m + 1) * x^(m^2 + 1) - 2 * x - 5 = 0 ∧ m^2 + 1 = 2 ∧ m + 1 ≠ 0 ↔ m = 1 := 
by
  sorry

end quadratic_condition_l148_148746


namespace wall_width_l148_148837

theorem wall_width (area height : ℕ) (h1 : area = 16) (h2 : height = 4) : area / height = 4 :=
by
  sorry

end wall_width_l148_148837


namespace six_digit_numbers_with_at_least_one_zero_correct_l148_148727

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l148_148727


namespace smallest_integer_proof_l148_148457

theorem smallest_integer_proof :
  ∃ (x : ℤ), x^2 = 3 * x + 75 ∧ ∀ (y : ℤ), y^2 = 3 * y + 75 → x ≤ y := 
  sorry

end smallest_integer_proof_l148_148457


namespace solve_fractions_in_integers_l148_148439

theorem solve_fractions_in_integers :
  ∀ (a b c : ℤ), (1 / a + 1 / b + 1 / c = 1) ↔
  (a = 3 ∧ b = 3 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 6) ∨
  (a = 2 ∧ b = 4 ∧ c = 4) ∨
  (a = 1 ∧ ∃ t : ℤ, b = t ∧ c = -t) :=
by {
  sorry
}

end solve_fractions_in_integers_l148_148439


namespace solve_equation_in_integers_l148_148437

theorem solve_equation_in_integers (a b c : ℤ) (h : 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) :
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 1 ∧ ∃ t : ℤ, b = t ∧ c = -t) :=
sorry

end solve_equation_in_integers_l148_148437


namespace six_digit_numbers_with_zero_l148_148710

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l148_148710


namespace six_digit_numbers_with_zero_l148_148695

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148695


namespace sum_first_10_terms_arithmetic_seq_l148_148123

theorem sum_first_10_terms_arithmetic_seq :
  ∀ (a : ℕ → ℤ), (a 1 = -2) → (a 2 + a 6 = 2) →
  (∃ S₁₀, S₁₀ = 10 * a 1 + (10 * (10 - 1)) / 2 * (a 2 - a 1) / (2 - 1) ∧ S₁₀ = 25) :=
begin
  assume a,
  assume h1 : a 1 = -2,
  assume h2 : a 2 + a 6 = 2,
  sorry
end

end sum_first_10_terms_arithmetic_seq_l148_148123


namespace num_ways_to_write_3070_l148_148550

theorem num_ways_to_write_3070 :
  let valid_digits := {d : ℕ | d ≤ 99}
  ∃ (M : ℕ), 
  M = 6500 ∧
  ∃ (a3 a2 a1 a0 : ℕ) (H : a3 ∈ valid_digits) (H : a2 ∈ valid_digits) (H : a1 ∈ valid_digits) (H : a0 ∈ valid_digits),
  3070 = a3 * 10^3 + a2 * 10^2 + a1 * 10 + a0 := sorry

end num_ways_to_write_3070_l148_148550


namespace largest_of_A_B_C_l148_148497

noncomputable def A : ℝ := (3003 / 3002) + (3003 / 3004)
noncomputable def B : ℝ := (3003 / 3004) + (3005 / 3004)
noncomputable def C : ℝ := (3004 / 3003) + (3004 / 3005)

theorem largest_of_A_B_C : A > B ∧ A ≥ C := by
  sorry

end largest_of_A_B_C_l148_148497


namespace string_length_l148_148952

def cylindrical_post_circumference : ℝ := 6
def cylindrical_post_height : ℝ := 15
def loops : ℝ := 3

theorem string_length :
  (cylindrical_post_height / loops)^2 + cylindrical_post_circumference^2 = 61 → 
  loops * Real.sqrt 61 = 3 * Real.sqrt 61 :=
by
  sorry

end string_length_l148_148952


namespace hyperbola_eccentricity_l148_148251

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    let C := (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1,
        P := (x y : ℝ) => (x - b)^2 + y^2 = a^2,
        asymptote := (x y : ℝ) => b * x - a * y = 0,
        M N : ℝ × ℝ := sorry--intersection points of P and asymptote,
        angle_MPN := sorry-- ∠MPN = 90°
    in eccentricity(C) = √2 := sorry


end hyperbola_eccentricity_l148_148251


namespace stratified_sampling_example_l148_148475

theorem stratified_sampling_example 
    (high_school_students : ℕ)
    (junior_high_students : ℕ) 
    (sampled_high_school_students : ℕ)
    (sampling_ratio : ℚ)
    (total_students : ℕ)
    (n : ℕ)
    (h1 : high_school_students = 3500)
    (h2 : junior_high_students = 1500)
    (h3 : sampled_high_school_students = 70)
    (h4 : sampling_ratio = sampled_high_school_students / high_school_students)
    (h5 : total_students = high_school_students + junior_high_students) :
    n = total_students * sampling_ratio → 
    n = 100 :=
by
  sorry

end stratified_sampling_example_l148_148475


namespace six_digit_numbers_with_at_least_one_zero_l148_148707

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l148_148707


namespace find_value_of_expression_l148_148749

theorem find_value_of_expression 
(h : ∀ (a b : ℝ), a * (3:ℝ)^2 - b * (3:ℝ) = 6) : 
  ∀ (a b : ℝ), 2023 - 6 * a + 2 * b = 2019 := 
by
  intro a b
  have h1 : 9 * a - 3 * b = 6 := by sorry
  have h2 : 3 * a - b = 2 := by sorry
  have result := 2023 - 2 * (3 * a - b)
  rw h2 at result
  exact result

end find_value_of_expression_l148_148749


namespace gold_coin_multiple_l148_148215

theorem gold_coin_multiple (x y k : ℕ) (h₁ : x + y = 16) (h₂ : x ≠ y) (h₃ : x^2 - y^2 = k * (x - y)) : k = 16 :=
sorry

end gold_coin_multiple_l148_148215


namespace find_numbers_l148_148572

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l148_148572


namespace cone_lateral_surface_area_l148_148754

theorem cone_lateral_surface_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : 
  (r * l * Real.pi = 6 * Real.pi) := by
  sorry

end cone_lateral_surface_area_l148_148754


namespace simplify_fraction_l148_148516

variable {a b c k : ℝ}
variable (h : a * b = c * k ∧ a * b ≠ 0)

theorem simplify_fraction (h : a * b = c * k ∧ a * b ≠ 0) : 
  (a - b - c + k) / (a + b + c + k) = (a - c) / (a + c) :=
by
  sorry

end simplify_fraction_l148_148516


namespace num_ordered_pairs_of_squares_diff_by_144_l148_148874

theorem num_ordered_pairs_of_squares_diff_by_144 :
  ∃ (p : Finset (ℕ × ℕ)), p.card = 4 ∧ ∀ (a b : ℕ), (a, b) ∈ p → a ≥ b ∧ a^2 - b^2 = 144 := by
  sorry

end num_ordered_pairs_of_squares_diff_by_144_l148_148874


namespace quadratic_always_positive_l148_148392

theorem quadratic_always_positive (a b c : ℝ) (ha : a ≠ 0) (hpos : a > 0) (hdisc : b^2 - 4 * a * c < 0) :
  ∀ x : ℝ, a * x^2 + b * x + c > 0 := 
by
  sorry

end quadratic_always_positive_l148_148392


namespace ratio_of_x_y_l148_148885

theorem ratio_of_x_y (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + 3 * y + 1) = 4 / 5) : x / y = 22 / 7 :=
sorry

end ratio_of_x_y_l148_148885


namespace birds_more_than_half_sunflower_seeds_l148_148143

theorem birds_more_than_half_sunflower_seeds :
  ∃ (n : ℕ), n = 3 ∧ ((4 / 5)^n * (2 / 5) + (2 / 5) > 1 / 2) :=
by
  sorry

end birds_more_than_half_sunflower_seeds_l148_148143


namespace six_digit_numbers_with_zero_l148_148691

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l148_148691


namespace hoseok_result_l148_148007

theorem hoseok_result :
  ∃ X : ℤ, (X - 46 = 15) ∧ (X - 29 = 32) :=
by
  sorry

end hoseok_result_l148_148007


namespace solve_for_x_l148_148917

-- Definitions and conditions from a) directly 
def f (x : ℝ) : ℝ := 64 * (2 * x - 1) ^ 3

-- Lean 4 statement to prove the problem
theorem solve_for_x (x : ℝ) : f x = 27 → x = 7 / 8 :=
by
  intro h
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l148_148917


namespace modular_inverse_28_mod_29_l148_148857

theorem modular_inverse_28_mod_29 :
  28 * 28 ≡ 1 [MOD 29] :=
by
  sorry

end modular_inverse_28_mod_29_l148_148857


namespace dolls_completion_time_l148_148789

def time_to_complete_dolls (craft_time_per_doll break_time_per_three_dolls total_dolls start_time : Nat) : Nat :=
  let total_craft_time := craft_time_per_doll * total_dolls
  let total_breaks := (total_dolls / 3) * break_time_per_three_dolls
  let total_time := total_craft_time + total_breaks
  (start_time + total_time) % 1440 -- 1440 is the number of minutes in a day

theorem dolls_completion_time :
  time_to_complete_dolls 105 30 10 600 = 300 := -- 600 is 10:00 AM in minutes, 300 is 5:00 AM in minutes
sorry

end dolls_completion_time_l148_148789


namespace average_of_four_digits_l148_148919

theorem average_of_four_digits (sum9 : ℤ) (avg9 : ℤ) (avg5 : ℤ) (sum4 : ℤ) (n : ℤ) :
  avg9 = 18 →
  n = 9 →
  sum9 = avg9 * n →
  avg5 = 26 →
  sum4 = sum9 - (avg5 * 5) →
  avg4 = sum4 / 4 →
  avg4 = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_of_four_digits_l148_148919


namespace correct_operation_l148_148184

variable (a b m : ℕ)

theorem correct_operation :
  (3 * a^2 * 2 * a^2 ≠ 5 * a^2) ∧
  ((2 * a^2)^3 = 8 * a^6) ∧
  (m^6 / m^3 ≠ m^2) ∧
  ((a + b)^2 ≠ a^2 + b^2) →
  ((2 * a^2)^3 = 8 * a^6) :=
by
  intros
  sorry

end correct_operation_l148_148184


namespace function_is_odd_and_increasing_l148_148461

-- Define the function y = x^(3/5)
def f (x : ℝ) : ℝ := x ^ (3 / 5)

-- Define what it means for the function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define what it means for the function to be increasing in its domain
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- The proposition to prove
theorem function_is_odd_and_increasing :
  is_odd f ∧ is_increasing f :=
by
  sorry

end function_is_odd_and_increasing_l148_148461


namespace apples_from_C_to_D_l148_148470

theorem apples_from_C_to_D (n m : ℕ)
  (h_tree_ratio : ∀ (P V : ℕ), P = 2 * V)
  (h_apple_ratio : ∀ (P V : ℕ), P = 7 * V)
  (trees_CD_Petya trees_CD_Vasya : ℕ)
  (h_trees_CD : trees_CD_Petya = 2 * trees_CD_Vasya)
  (apples_CD_Petya apples_CD_Vasya: ℕ)
  (h_apples_CD : apples_CD_Petya = (m / 4) ∧ apples_CD_Vasya = (3 * m / 4)) : 
  apples_CD_Vasya = 3 * apples_CD_Petya := by
  sorry

end apples_from_C_to_D_l148_148470


namespace historical_fiction_new_releases_fraction_l148_148842

noncomputable def HF_fraction_total_inventory : ℝ := 0.4
noncomputable def Mystery_fraction_total_inventory : ℝ := 0.3
noncomputable def SF_fraction_total_inventory : ℝ := 0.2
noncomputable def Romance_fraction_total_inventory : ℝ := 0.1

noncomputable def HF_new_release_percentage : ℝ := 0.35
noncomputable def Mystery_new_release_percentage : ℝ := 0.60
noncomputable def SF_new_release_percentage : ℝ := 0.45
noncomputable def Romance_new_release_percentage : ℝ := 0.80

noncomputable def historical_fiction_new_releases : ℝ := HF_fraction_total_inventory * HF_new_release_percentage
noncomputable def mystery_new_releases : ℝ := Mystery_fraction_total_inventory * Mystery_new_release_percentage
noncomputable def sf_new_releases : ℝ := SF_fraction_total_inventory * SF_new_release_percentage
noncomputable def romance_new_releases : ℝ := Romance_fraction_total_inventory * Romance_new_release_percentage

noncomputable def total_new_releases : ℝ :=
  historical_fiction_new_releases + mystery_new_releases + sf_new_releases + romance_new_releases

theorem historical_fiction_new_releases_fraction :
  (historical_fiction_new_releases / total_new_releases) = (2 / 7) :=
by
  sorry

end historical_fiction_new_releases_fraction_l148_148842


namespace product_of_repeating_decimal_and_integer_l148_148492

noncomputable def repeating_decimal_to_fraction (s : ℝ) : ℚ := 
  456 / 999

noncomputable def multiply_and_simplify (s : ℝ) (n : ℤ) : ℚ := 
  (repeating_decimal_to_fraction s) * (n : ℚ)

theorem product_of_repeating_decimal_and_integer 
(s : ℝ) (h : s = 0.456456456456456456456456456456456456456456) :
  multiply_and_simplify s 8 = 1216 / 333 :=
by sorry

end product_of_repeating_decimal_and_integer_l148_148492


namespace isabella_more_than_giselle_l148_148542

variables (I S G : ℕ)

def isabella_has_more_than_sam : Prop := I = S + 45
def giselle_amount : Prop := G = 120
def total_amount : Prop := I + S + G = 345

theorem isabella_more_than_giselle
  (h1 : isabella_has_more_than_sam I S)
  (h2 : giselle_amount G)
  (h3 : total_amount I S G) :
  I - G = 15 :=
by
  sorry

end isabella_more_than_giselle_l148_148542


namespace assume_dead_heat_race_l148_148812

variable {Va Vb L H : ℝ}

theorem assume_dead_heat_race (h1 : Va = (51 / 44) * Vb) :
  H = (7 / 51) * L :=
sorry

end assume_dead_heat_race_l148_148812


namespace NoahMealsCount_l148_148357

-- Definition of all the choices available to Noah
def MainCourses := ["Pizza", "Burger", "Pasta"]
def Beverages := ["Soda", "Juice"]
def Snacks := ["Apple", "Banana", "Cookie"]

-- Condition that Noah avoids soda with pizza
def isValidMeal (main : String) (beverage : String) : Bool :=
  not (main = "Pizza" ∧ beverage = "Soda")

-- Total number of valid meal combinations
def totalValidMeals : Nat :=
  (if isValidMeal "Pizza" "Juice" then 1 else 0) * Snacks.length +
  (Beverages.length - 1) * Snacks.length * (MainCourses.length - 1) + -- for Pizza
  Beverages.length * Snacks.length * 2 -- for Burger and Pasta

-- The theorem that Noah can buy 15 distinct meals
theorem NoahMealsCount : totalValidMeals = 15 := by
  sorry

end NoahMealsCount_l148_148357


namespace sum_of_fourth_powers_of_solutions_l148_148649

theorem sum_of_fourth_powers_of_solutions (x y : ℝ)
  (h : |x^2 - 2 * x + 1/1004| = 1/1004 ∨ |y^2 - 2 * y + 1/1004| = 1/1004) :
  x^4 + y^4 = 20160427280144 / 12600263001 :=
sorry

end sum_of_fourth_powers_of_solutions_l148_148649


namespace factor_expression_l148_148855

theorem factor_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) + 2 * (y + 2) = (y + 2) * (5 * y + 11) := by
  sorry

end factor_expression_l148_148855


namespace sasha_tree_planting_cost_l148_148301

theorem sasha_tree_planting_cost :
  ∀ (initial_temperature final_temperature : ℝ)
    (temp_drop_per_tree : ℝ) (cost_per_tree : ℝ)
    (temperature_drop : ℝ) (num_trees : ℕ)
    (total_cost : ℝ),
    initial_temperature = 80 →
    final_temperature = 78.2 →
    temp_drop_per_tree = 0.1 →
    cost_per_tree = 6 →
    temperature_drop = initial_temperature - final_temperature →
    num_trees = temperature_drop / temp_drop_per_tree →
    total_cost = num_trees * cost_per_tree →
    total_cost = 108 :=
by
  intros initial_temperature final_temperature temp_drop_per_tree
    cost_per_tree temperature_drop num_trees total_cost
    h_initial h_final h_drop_tree h_cost_tree
    h_temp_drop h_num_trees h_total_cost
  rw [h_initial, h_final] at h_temp_drop
  rw [h_temp_drop] at h_num_trees
  rw [h_num_trees] at h_total_cost
  rw [h_drop_tree] at h_total_cost
  rw [h_cost_tree] at h_total_cost
  norm_num at h_total_cost
  exact h_total_cost

end sasha_tree_planting_cost_l148_148301


namespace june_initial_stickers_l148_148423

theorem june_initial_stickers (J b g t : ℕ) (h_b : b = 63) (h_g : g = 25) (h_t : t = 189) : 
  (J + g) + (b + g) = t → J = 76 :=
by
  sorry

end june_initial_stickers_l148_148423


namespace quotient_sum_40_5_l148_148936

theorem quotient_sum_40_5 : (40 + 5) / 5 = 9 := by
  sorry

end quotient_sum_40_5_l148_148936


namespace solid_is_triangular_prism_l148_148258

-- Given conditions as definitions
def front_view_is_isosceles_triangle (solid : Type) : Prop := 
  -- Define the property that the front view is an isosceles triangle
  sorry

def left_view_is_isosceles_triangle (solid : Type) : Prop := 
  -- Define the property that the left view is an isosceles triangle
  sorry

def top_view_is_circle (solid : Type) : Prop := 
   -- Define the property that the top view is a circle
  sorry

-- Define the property of being a triangular prism
def is_triangular_prism (solid : Type) : Prop :=
  -- Define the property that the solid is a triangular prism
  sorry

-- The main theorem: proving that given the conditions, the solid could be a triangular prism
theorem solid_is_triangular_prism (solid : Type) :
  front_view_is_isosceles_triangle solid ∧ 
  left_view_is_isosceles_triangle solid ∧ 
  top_view_is_circle solid →
  is_triangular_prism solid :=
sorry

end solid_is_triangular_prism_l148_148258


namespace worker_times_l148_148000

-- Define the problem
theorem worker_times (x y : ℝ) (h1 : (1 / x + 1 / y = 1 / 8)) (h2 : x = y - 12) :
    x = 24 ∧ y = 12 :=
by
  sorry

end worker_times_l148_148000


namespace intervals_of_monotonicity_range_of_a_for_fx_le_x_l148_148408

-- Define the function f
def f (x a : ℝ) : ℝ := x - a / x - (a + 1) * Real.log x

-- Define the derivative of f
def f' (x a : ℝ) : ℝ := (x - a) * (x - 1) / (x * x)

-- Prove the intervals of monotonicity for f when a = 1/2
theorem intervals_of_monotonicity (x : ℝ) (hx : 0 < x) (a := (1 : ℝ) / 2) :
  (f' x a > 0 → x ∈ (Set.Ioo 0 (1 / 2) ∪ Set.Ioi 1)) ∧
  (f' x a < 0 → x ∈ Set.Ioo (1 / 2) 1) :=
sorry

-- Define the function phi
def phi (x a : ℝ) : ℝ := a + (a + 1) * x * Real.log x

-- Define the derivative of phi
def phi' (x a : ℝ) : ℝ := (a + 1) * (1 + Real.log x)

-- Prove the range of a such that f(x) ≤ x for all x in (0, ∞)
theorem range_of_a_for_fx_le_x (a : ℝ) :
  (∀ x, 0 < x → f x a ≤ x) ↔ a ≥ 1 / (Real.exp 1 - 1) :=
sorry

end intervals_of_monotonicity_range_of_a_for_fx_le_x_l148_148408


namespace average_of_solutions_l148_148851

-- Define the quadratic equation condition
def quadratic_eq : Prop := ∃ x : ℂ, 3*x^2 - 4*x + 1 = 0

-- State the theorem
theorem average_of_solutions : quadratic_eq → (∃ avg : ℂ, avg = 2 / 3) :=
by
  sorry

end average_of_solutions_l148_148851


namespace percentage_increase_visitors_l148_148164

theorem percentage_increase_visitors
  (original_visitors : ℕ)
  (original_fee : ℝ := 1)
  (fee_reduction : ℝ := 0.25)
  (visitors_increase : ℝ := 0.20) :
  ((original_visitors + (visitors_increase * original_visitors)) / original_visitors - 1) * 100 = 20 := by
  sorry

end percentage_increase_visitors_l148_148164


namespace repeating_decimal_to_fraction_l148_148375

theorem repeating_decimal_to_fraction : 
  ∀ x : ℝ, x = (47 / 99 : ℝ) ↔ (repeating_decimal (47 : ℤ) (100 : ℤ) = x) := 
by
  sorry

end repeating_decimal_to_fraction_l148_148375


namespace line_through_point_outside_plane_l148_148333

-- Definitions based on conditions
variable {Point Line Plane : Type}
variable (P : Point) (a : Line) (α : Plane)

-- Define the conditions
variable (passes_through : Point → Line → Prop)
variable (outside_of : Point → Plane → Prop)

-- State the theorem
theorem line_through_point_outside_plane :
  (passes_through P a) ∧ (¬ outside_of P α) :=
sorry

end line_through_point_outside_plane_l148_148333


namespace quadratic_root_condition_l148_148875

theorem quadratic_root_condition (a : ℝ) :
  (4 * Real.sqrt 2) = 3 * Real.sqrt (3 - 2 * a) → a = 1 / 2 :=
by
  sorry

end quadratic_root_condition_l148_148875


namespace semi_minor_axis_l148_148264

theorem semi_minor_axis (a c : ℝ) (h_a : a = 5) (h_c : c = 2) : 
  ∃ b : ℝ, b = Real.sqrt (a^2 - c^2) ∧ b = Real.sqrt 21 :=
by
  use Real.sqrt 21
  sorry

end semi_minor_axis_l148_148264


namespace master_li_speeding_l148_148433

theorem master_li_speeding (distance : ℝ) (time : ℝ) (speed_limit : ℝ) (average_speed : ℝ)
  (h_distance : distance = 165)
  (h_time : time = 2)
  (h_speed_limit : speed_limit = 80)
  (h_average_speed : average_speed = distance / time)
  (h_speeding : average_speed > speed_limit) :
  True :=
sorry

end master_li_speeding_l148_148433


namespace range_of_a_l148_148242

-- Definitions of conditions
def is_odd_function {A : Type} [AddGroup A] (f : A → A) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing {A : Type} [LinearOrderedAddCommGroup A] (f : A → A) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Main statement
theorem range_of_a 
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_monotone_dec : is_monotonically_decreasing f)
  (h_domain : ∀ x, -7 < x ∧ x < 7 → -7 < f x ∧ f x < 7)
  (h_cond : ∀ a, f (1 - a) + f (2 * a - 5) < 0): 
  ∀ a, 4 < a → a < 6 :=
sorry

end range_of_a_l148_148242


namespace unique_sequence_count_l148_148405

def is_valid_sequence (a : Fin 5 → ℕ) :=
  a 0 = 1 ∧
  a 1 > a 0 ∧
  a 2 > a 1 ∧
  a 3 > a 2 ∧
  a 4 = 15 ∧
  (a 1) ^ 2 ≤ a 0 * a 2 + 1 ∧
  (a 2) ^ 2 ≤ a 1 * a 3 + 1 ∧
  (a 3) ^ 2 ≤ a 2 * a 4 + 1

theorem unique_sequence_count : 
  ∃! (a : Fin 5 → ℕ), is_valid_sequence a :=
sorry

end unique_sequence_count_l148_148405


namespace volume_calculation_l148_148983

noncomputable def volume_of_regular_tetrahedron 
  (a : ℝ) 
  (d_face : ℝ) 
  (d_edge : ℝ) : ℝ :=
  let h_base : ℝ := (real.sqrt 3 / 2) * a in
  let h_pyramid_midpoint : ℝ := real.sqrt (d_face^2 - d_edge^2) in
  let h_pyramid : ℝ := 2 * h_pyramid_midpoint in
  let base_area : ℝ := (real.sqrt 3 / 4) * a^2 in
  (1 / 3) * base_area * h_pyramid

theorem volume_calculation
  (a : ℝ) 
  (d_face : ℝ) 
  (d_edge : ℝ)
  (h_midpoint_eq_two : d_face = 2)
  (h_edge_eq_sqrt_six : d_edge = real.sqrt 6) : 
  volume_of_regular_tetrahedron a d_face d_edge = 
  (1 / 3) * ((real.sqrt 3 / 4) * a^2) * (2 * real.sqrt (d_face^2 - d_edge^2)) :=
by sorry

end volume_calculation_l148_148983


namespace find_b_value_l148_148924

-- Let's define the given conditions as hypotheses in Lean

theorem find_b_value 
  (x1 y1 x2 y2 : ℤ) 
  (h1 : (x1, y1) = (2, 2)) 
  (h2 : (x2, y2) = (8, 14)) 
  (midpoint : ∃ (m1 m2 : ℤ), m1 = (x1 + x2) / 2 ∧ m2 = (y1 + y2) / 2 ∧ (m1, m2) = (5, 8))
  (perpendicular_bisector : ∀ (x y : ℤ), x + y = b → (x, y) = (5, 8)) :
  b = 13 := 
by {
  sorry
}

end find_b_value_l148_148924


namespace polynomial_value_l148_148262
variable {x y : ℝ}
theorem polynomial_value (h : 3 * x^2 + 4 * y + 9 = 8) : 9 * x^2 + 12 * y + 8 = 5 :=
by
   sorry

end polynomial_value_l148_148262


namespace chores_per_week_l148_148147

theorem chores_per_week :
  ∀ (cookie_per_chore : ℕ) 
    (total_money : ℕ) 
    (cost_per_pack : ℕ) 
    (cookies_per_pack : ℕ) 
    (weeks : ℕ)
    (chores_per_week : ℕ),
  cookie_per_chore = 3 →
  total_money = 15 →
  cost_per_pack = 3 →
  cookies_per_pack = 24 →
  weeks = 10 →
  chores_per_week = (total_money / cost_per_pack * cookies_per_pack / weeks) / cookie_per_chore →
  chores_per_week = 4 :=
by
  intros cookie_per_chore total_money cost_per_pack cookies_per_pack weeks chores_per_week
  intros h1 h2 h3 h4 h5 h6
  sorry

end chores_per_week_l148_148147


namespace prove_final_value_is_111_l148_148959

theorem prove_final_value_is_111 :
  let initial_num := 16
  let doubled_num := initial_num * 2
  let added_five := doubled_num + 5
  let trebled_result := added_five * 3
  trebled_result = 111 :=
by
  sorry

end prove_final_value_is_111_l148_148959


namespace option_C_correct_l148_148462

theorem option_C_correct : ∀ x : ℝ, x^2 + 1 ≥ 2 * |x| :=
by
  intro x
  sorry

end option_C_correct_l148_148462


namespace math_competition_correct_answers_l148_148479

theorem math_competition_correct_answers (qA qB cA cB : ℕ) 
  (h_total_questions : qA + qB = 10)
  (h_score_A : cA * 5 - (qA - cA) * 2 = 36)
  (h_score_B : cB * 5 - (qB - cB) * 2 = 22) 
  (h_combined_score : cA * 5 - (qA - cA) * 2 + cB * 5 - (qB - cB) * 2 = 58)
  (h_score_difference : cA * 5 - (qA - cA) * 2 - (cB * 5 - (qB - cB) * 2) = 14) : 
  cA = 8 :=
by {
  sorry
}

end math_competition_correct_answers_l148_148479


namespace donovan_lap_time_is_45_l148_148036

-- Definitions based on the conditions
def circular_track_length : ℕ := 600
def michael_lap_time : ℕ := 40
def michael_laps_to_pass_donovan : ℕ := 9

-- The theorem to prove
theorem donovan_lap_time_is_45 : ∃ D : ℕ, 8 * D = michael_laps_to_pass_donovan * michael_lap_time ∧ D = 45 := by
  sorry

end donovan_lap_time_is_45_l148_148036


namespace dictionary_cost_l148_148547

def dinosaur_book_cost : ℕ := 19
def children_cookbook_cost : ℕ := 7
def saved_amount : ℕ := 8
def needed_amount : ℕ := 29

def total_amount_needed := saved_amount + needed_amount
def combined_books_cost := dinosaur_book_cost + children_cookbook_cost

theorem dictionary_cost : total_amount_needed - combined_books_cost = 11 :=
by
  -- proof omitted
  sorry

end dictionary_cost_l148_148547


namespace max_5x_min_25x_l148_148389

theorem max_5x_min_25x : ∃ x : ℝ, 5^x - 25^x = 1/4 :=
by
  sorry

end max_5x_min_25x_l148_148389


namespace max_value_of_f_l148_148387

noncomputable def f : ℝ → ℝ := λ x, 5^x - 25^x

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 1/4 ∧ ∃ y : ℝ, f y = 1/4 := by
  sorry

end max_value_of_f_l148_148387


namespace Leela_Hotel_all_three_reunions_l148_148967

theorem Leela_Hotel_all_three_reunions
  (A B C : Finset ℕ)
  (hA : A.card = 80)
  (hB : B.card = 90)
  (hC : C.card = 70)
  (hAB : (A ∩ B).card = 30)
  (hAC : (A ∩ C).card = 25)
  (hBC : (B ∩ C).card = 20)
  (hABC : ((A ∪ B ∪ C)).card = 150) : 
  (A ∩ B ∩ C).card = 15 :=
by
  sorry

end Leela_Hotel_all_three_reunions_l148_148967


namespace tray_height_l148_148204

noncomputable def height_of_tray : ℝ :=
  let side_length := 120
  let cut_distance := 4 * Real.sqrt 2
  let angle := 45 * (Real.pi / 180)
  -- Define the function that calculates height based on given conditions
  
  sorry

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  side_length = 120 ∧ cut_distance = 4 * Real.sqrt 2 ∧ angle = 45 * (Real.pi / 180) →
  height_of_tray = 4 * Real.sqrt 2 :=
by
  intros
  unfold height_of_tray
  sorry

end tray_height_l148_148204


namespace range_of_x_plus_2y_l148_148864

theorem range_of_x_plus_2y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) : x + 2 * y ≥ 9 :=
sorry

end range_of_x_plus_2y_l148_148864


namespace sum_first_10_terms_arithmetic_sequence_l148_148129

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l148_148129


namespace matrix_exponent_b_m_l148_148527

theorem matrix_exponent_b_m (b m : ℕ) :
  let C := Matrix.of 1 3 b 0 1 5 0 0 1 in
  C ^ m = Matrix.of 1 27 3005 0 1 45 0 0 1 →
  b + m = 283 := 
by
  sorry

end matrix_exponent_b_m_l148_148527


namespace factorize_x_squared_minus_1_l148_148039

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l148_148039


namespace a_plus_2b_eq_21_l148_148256

-- Definitions and conditions based on the problem statement
def a_log_250_2_plus_b_log_250_5_eq_3 (a b : ℤ) : Prop :=
  a * Real.log 2 / Real.log 250 + b * Real.log 5 / Real.log 250 = 3

-- The theorem that needs to be proved
theorem a_plus_2b_eq_21 (a b : ℤ) (h : a_log_250_2_plus_b_log_250_5_eq_3 a b) : a + 2 * b = 21 := 
  sorry

end a_plus_2b_eq_21_l148_148256


namespace cards_from_around_country_l148_148940

-- Define the total number of cards and the number from home
def total_cards : ℝ := 403.0
def home_cards : ℝ := 287.0

-- Define the expected number of cards from around the country
def expected_country_cards : ℝ := 116.0

-- Theorem statement
theorem cards_from_around_country :
  total_cards - home_cards = expected_country_cards :=
by
  -- Since this only requires the statement, the proof is omitted
  sorry

end cards_from_around_country_l148_148940


namespace find_n_l148_148469

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 52) (h2 : Nat.gcd n 16 = 8) : n = 26 := by
  sorry

end find_n_l148_148469


namespace RU_eq_825_l148_148323

variables (P Q R S T U : Type)
variables (PQ QR RP QS SR : ℝ)
variables (RU : ℝ)
variables (hPQ : PQ = 13)
variables (hQR : QR = 30)
variables (hRP : RP = 26)
variables (hQS : QS = 10)
variables (hSR : SR = 20)

theorem RU_eq_825 :
  RU = 8.25 :=
sorry

end RU_eq_825_l148_148323


namespace factorize_x_squared_minus_one_l148_148052

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l148_148052


namespace sum_of_first_ten_terms_seq_l148_148970

def a₁ : ℤ := -5
def d : ℤ := 6
def n : ℕ := 10

theorem sum_of_first_ten_terms_seq : (n * (a₁ + a₁ + (n - 1) * d)) / 2 = 220 :=
by
  sorry

end sum_of_first_ten_terms_seq_l148_148970


namespace horses_tiles_equation_l148_148442

-- Conditions from the problem
def total_horses (x y : ℕ) : Prop := x + y = 100
def total_tiles (x y : ℕ) : Prop := 3 * x + (1 / 3 : ℚ) * y = 100

-- The statement to prove
theorem horses_tiles_equation (x y : ℕ) :
  total_horses x y ∧ total_tiles x y ↔ 
  (x + y = 100 ∧ (3 * x + (1 / 3 : ℚ) * y = 100)) :=
by
  sorry

end horses_tiles_equation_l148_148442


namespace angle_PQC_in_triangle_l148_148901

theorem angle_PQC_in_triangle 
  (A B C P Q: ℝ)
  (h_in_triangle: A + B + C = 180)
  (angle_B_exterior_bisector: ∀ B_ext, B_ext = 180 - B →  angle_B = 90 - B / 2)
  (angle_C_exterior_bisector: ∀ C_ext, C_ext = 180 - C →  angle_C = 90 - C / 2)
  (h_PQ_BC_angle: ∀ PQ_angle BC_angle, PQ_angle = 30 → BC_angle = 30) :
  ∃ PQC_angle, PQC_angle = (180 - A) / 2 :=
by
  sorry

end angle_PQC_in_triangle_l148_148901


namespace A_share_of_gain_l148_148811

-- Given problem conditions
def investment_A (x : ℝ) : ℝ := x * 12
def investment_B (x : ℝ) : ℝ := 2 * x * 6
def investment_C (x : ℝ) : ℝ := 3 * x * 4
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def total_gain : ℝ := 21000

-- Mathematically equivalent proof problem statement
theorem A_share_of_gain (x : ℝ) : (investment_A x) / (total_investment x) * total_gain = 7000 :=
by
  sorry

end A_share_of_gain_l148_148811


namespace sum_divisible_by_5_and_7_l148_148447

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_divisible_by_5_and_7 (A B : ℕ) (hA_prime : is_prime A) 
  (hB_prime : is_prime B) (hA_minus_3_prime : is_prime (A - 3)) 
  (hA_plus_3_prime : is_prime (A + 3)) (hB_eq_2 : B = 2) : 
  5 ∣ (A + B + (A - 3) + (A + 3)) ∧ 7 ∣ (A + B + (A - 3) + (A + 3)) := by 
  sorry

end sum_divisible_by_5_and_7_l148_148447


namespace max_value_5x_minus_25x_l148_148382

open Real

theorem max_value_5x_minus_25x : 
  ∃ x : ℝ, ∀ y : ℝ, (y = 5^x) → (y - y^2) ≤ 1 / 4 := 
by 
  sorry

end max_value_5x_minus_25x_l148_148382


namespace train_b_speed_l148_148176

/-- Given:
    1. Length of train A: 150 m
    2. Length of train B: 150 m
    3. Speed of train A: 54 km/hr
    4. Time taken to cross train B: 12 seconds
    Prove: The speed of train B is 36 km/hr
-/
theorem train_b_speed (l_A l_B : ℕ) (V_A : ℕ) (t : ℕ) (h1 : l_A = 150) (h2 : l_B = 150) (h3 : V_A = 54) (h4 : t = 12) :
  ∃ V_B : ℕ, V_B = 36 := sorry

end train_b_speed_l148_148176


namespace inequality_proof_l148_148766

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  1 / (b * c + a + 1 / a) + 1 / (a * c + b + 1 / b) + 1 / (a * b + c + 1 / c) ≤ 27 / 31 :=
sorry

end inequality_proof_l148_148766


namespace mangoes_in_shop_l148_148291

-- Define the conditions
def ratio_mango_to_apple := 10 / 3
def apples := 36

-- Problem statement to prove
theorem mangoes_in_shop : ∃ (m : ℕ), m = 120 ∧ m = apples * ratio_mango_to_apple :=
by
  sorry

end mangoes_in_shop_l148_148291


namespace twenty_five_percent_of_x_l148_148331

-- Define the number x and the conditions
variable (x : ℝ)
variable (h : x - (3/4) * x = 100)

-- The theorem statement
theorem twenty_five_percent_of_x : (1/4) * x = 100 :=
by 
  -- Assume x satisfies the given condition
  sorry

end twenty_five_percent_of_x_l148_148331


namespace P2011_1_neg1_is_0_2_pow_1006_l148_148793

def P1 (x y : ℤ) : ℤ × ℤ := (x + y, x - y)

def Pn : ℕ → ℤ → ℤ → ℤ × ℤ 
| 0, x, y => (x, y)
| (n + 1), x, y => P1 (Pn n x y).1 (Pn n x y).2

theorem P2011_1_neg1_is_0_2_pow_1006 : Pn 2011 1 (-1) = (0, 2^1006) := by
  sorry

end P2011_1_neg1_is_0_2_pow_1006_l148_148793


namespace total_hair_cut_l148_148646

-- Define the amounts cut on two consecutive days
def first_cut : ℝ := 0.375
def second_cut : ℝ := 0.5

-- Statement: Prove that the total amount cut off is 0.875 inches
theorem total_hair_cut : first_cut + second_cut = 0.875 :=
by {
  -- The exact proof would go here
  sorry
}

end total_hair_cut_l148_148646


namespace cylinder_sphere_ratio_is_3_2_l148_148247

noncomputable def cylinder_sphere_surface_ratio (r : ℝ) : ℝ :=
  let cylinder_surface_area := 2 * Real.pi * r^2 + 2 * r * Real.pi * (2 * r)
  let sphere_surface_area := 4 * Real.pi * r^2
  cylinder_surface_area / sphere_surface_area

theorem cylinder_sphere_ratio_is_3_2 (r : ℝ) (h : r > 0) :
  cylinder_sphere_surface_ratio r = 3 / 2 :=
by
  sorry

end cylinder_sphere_ratio_is_3_2_l148_148247


namespace two_connected_iff_constructible_with_H_paths_l148_148953

-- A graph is represented as a structure with vertices and edges
structure Graph where
  vertices : Type
  edges : vertices → vertices → Prop

-- Function to check if a graph is 2-connected
noncomputable def isTwoConnected (G : Graph) : Prop := sorry

-- Function to check if a graph can be constructed by adding H-paths
noncomputable def constructibleWithHPaths (G H : Graph) : Prop := sorry

-- Given a graph G and subgraph H, we need to prove the equivalence
theorem two_connected_iff_constructible_with_H_paths (G H : Graph) :
  (isTwoConnected G) ↔ (constructibleWithHPaths G H) := sorry

end two_connected_iff_constructible_with_H_paths_l148_148953


namespace sin_inv_tan_eq_l148_148979

open Real

theorem sin_inv_tan_eq :
  let a := arcsin (4/5)
  let b := arctan 3
  sin (a + b) = (13 * sqrt 10) / 50 := 
by
  let a := arcsin (4/5)
  let b := arctan 3
  sorry

end sin_inv_tan_eq_l148_148979


namespace count_house_numbers_l148_148976

def isPrime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def twoDigitPrimesBetween40And60 : List ℕ :=
  [41, 43, 47, 53, 59]

theorem count_house_numbers : 
  ∃ n : ℕ, n = 20 ∧ 
  ∀ (AB CD : ℕ), 
  AB ∈ twoDigitPrimesBetween40And60 → 
  CD ∈ twoDigitPrimesBetween40And60 → 
  AB ≠ CD → 
  true :=
by
  sorry

end count_house_numbers_l148_148976


namespace ratio_of_running_speed_l148_148329

theorem ratio_of_running_speed (distance : ℝ) (time_jack : ℝ) (time_jill : ℝ) 
  (h_distance_eq : distance = 42) (h_time_jack_eq : time_jack = 6) 
  (h_time_jill_eq : time_jill = 4.2) :
  (distance / time_jack) / (distance / time_jill) = 7 / 10 := by 
  sorry

end ratio_of_running_speed_l148_148329


namespace f_increasing_maximum_b_condition_approximate_ln2_l148_148406

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem f_increasing : ∀ x y : ℝ, x < y → f x ≤ f y := 
sorry

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f (2 * x) - 4 * b * f x

theorem maximum_b_condition (x : ℝ) (H : 0 < x): ∃ b, g x b > 0 ∧ b ≤ 2 := 
sorry

theorem approximate_ln2 :
  0.692 ≤ Real.log 2 ∧ Real.log 2 ≤ 0.694 :=
sorry

end f_increasing_maximum_b_condition_approximate_ln2_l148_148406


namespace roots_cubic_sum_of_cubes_l148_148768

theorem roots_cubic_sum_of_cubes (a b c : ℝ)
  (h1 : Polynomial.eval a (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h4 : a + b + c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 753 :=
by
  sorry

end roots_cubic_sum_of_cubes_l148_148768


namespace original_number_l148_148958

theorem original_number (x y : ℝ) (h1 : 10 * x + 22 * y = 780) (h2 : y = 37.66666666666667) : 
  x + y = 32.7 := 
sorry

end original_number_l148_148958


namespace proper_subsets_count_l148_148907

theorem proper_subsets_count (A : Set (Fin 4)) (h : A = {1, 2, 3}) : 
  ∃ n : ℕ, n = 7 ∧ ∃ (S : Finset (Set (Fin 4))), S.card = n ∧ (∀ B, B ∈ S → B ⊂ A) := 
by {
  sorry
}

end proper_subsets_count_l148_148907


namespace evaluate_sqrt_sum_l148_148310

theorem evaluate_sqrt_sum : (Real.sqrt 1 + Real.sqrt 9) = 4 := by
  sorry

end evaluate_sqrt_sum_l148_148310


namespace possible_values_of_sum_of_reciprocals_l148_148877

theorem possible_values_of_sum_of_reciprocals {a b : ℝ} (h1 : a * b > 0) (h2 : a + b = 1) : 
  1 / a + 1 / b = 4 := 
by 
  sorry

end possible_values_of_sum_of_reciprocals_l148_148877


namespace six_digit_numbers_with_zero_count_l148_148676

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l148_148676


namespace area_of_region_S_is_correct_l148_148623

noncomputable def area_of_inverted_region (d : ℝ) : ℝ :=
  if h : d = 1.5 then 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi else 0

theorem area_of_region_S_is_correct :
  area_of_inverted_region 1.5 = 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi := 
by 
  sorry

end area_of_region_S_is_correct_l148_148623


namespace second_trial_amount_691g_l148_148370

theorem second_trial_amount_691g (low high : ℝ) (h_range : low = 500) (h_high : high = 1000) (h_method : ∃ x, x = 0.618) : 
  high - 0.618 * (high - low) = 691 :=
by
  sorry

end second_trial_amount_691g_l148_148370


namespace files_missing_is_15_l148_148498

def total_files : ℕ := 60
def morning_files : ℕ := total_files / 2
def afternoon_files : ℕ := 15
def organized_files : ℕ := morning_files + afternoon_files
def missing_files : ℕ := total_files - organized_files

theorem files_missing_is_15 : missing_files = 15 :=
  sorry

end files_missing_is_15_l148_148498


namespace domain_of_rational_func_l148_148364

noncomputable def rational_func (x : ℝ) : ℝ := (2 * x ^ 3 - 3 * x ^ 2 + 5 * x - 1) / (x ^ 2 - 5 * x + 6)

theorem domain_of_rational_func : 
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ↔ (∃ y : ℝ, rational_func y = x) :=
by
  sorry

end domain_of_rational_func_l148_148364


namespace triangle_height_l148_148348

theorem triangle_height (s h : ℝ) 
  (area_square : s^2 = s * s) 
  (area_triangle : 1/2 * s * h = s^2) 
  (areas_equal : s^2 = s^2) : 
  h = 2 * s := 
sorry

end triangle_height_l148_148348


namespace circle_to_ellipse_scaling_l148_148971

theorem circle_to_ellipse_scaling :
  ∀ (x' y' : ℝ), (4 * x')^2 + y'^2 = 16 → x'^2 / 16 + y'^2 / 4 = 1 :=
by
  intro x' y'
  intro h
  sorry

end circle_to_ellipse_scaling_l148_148971


namespace total_cats_l148_148621

def initial_siamese_cats : Float := 13.0
def initial_house_cats : Float := 5.0
def added_cats : Float := 10.0

theorem total_cats : initial_siamese_cats + initial_house_cats + added_cats = 28.0 := by
  sorry

end total_cats_l148_148621


namespace six_digit_numbers_with_at_least_one_zero_l148_148666

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l148_148666


namespace midpoint_set_of_segments_eq_circle_l148_148761

-- Define the existence of skew perpendicular lines with given properties
variable (a d : ℝ)

-- Conditions: Distance between lines is a, segment length is d
-- The coordinates system configuration
-- Point on the first line: (x, 0, 0)
-- Point on the second line: (0, y, a)
def are_midpoints_of_segments_of_given_length
  (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), 
    p = (x / 2, y / 2, a / 2) ∧ 
    x^2 + y^2 = d^2 - a^2

-- Proof statement
theorem midpoint_set_of_segments_eq_circle :
  { p : ℝ × ℝ × ℝ | are_midpoints_of_segments_of_given_length a d p } =
  { p : ℝ × ℝ × ℝ | ∃ (r : ℝ), p = (r * (d^2 - a^2) / (2*d), r * (d^2 - a^2) / (2*d), a / 2)
    ∧ r^2 * (d^2 - a^2) = (d^2 - a^2) } :=
sorry

end midpoint_set_of_segments_eq_circle_l148_148761


namespace prime_only_one_solution_l148_148065

theorem prime_only_one_solution (p : ℕ) (hp : Nat.Prime p) : 
  (∃ k : ℕ, 2 * p^4 - p^2 + 16 = k^2) → p = 3 := 
by 
  sorry

end prime_only_one_solution_l148_148065


namespace discount_calc_l148_148555

noncomputable def discount_percentage 
    (cost_price : ℝ) (markup_percentage : ℝ) (selling_price : ℝ) : ℝ :=
  let marked_price := cost_price + (markup_percentage / 100 * cost_price)
  let discount := marked_price - selling_price
  (discount / marked_price) * 100

theorem discount_calc :
  discount_percentage 540 15 460 = 25.92 :=
by
  sorry

end discount_calc_l148_148555


namespace five_less_than_sixty_percent_of_cats_l148_148796

theorem five_less_than_sixty_percent_of_cats (hogs cats : ℕ) 
  (hogs_eq : hogs = 3 * cats)
  (hogs_value : hogs = 75) : 
  5 < 60 * cats / 100 :=
by {
  sorry
}

end five_less_than_sixty_percent_of_cats_l148_148796


namespace alcohol_water_ratio_l148_148599

variable {r s V1 : ℝ}

theorem alcohol_water_ratio 
  (h1 : r > 0) 
  (h2 : s > 0) 
  (h3 : V1 > 0) :
  let alcohol_in_JarA := 2 * r * V1 / (r + 1) + V1
  let water_in_JarA := 2 * V1 / (r + 1)
  let alcohol_in_JarB := 3 * s * V1 / (s + 1)
  let water_in_JarB := 3 * V1 / (s + 1)
  let total_alcohol := alcohol_in_JarA + alcohol_in_JarB
  let total_water := water_in_JarA + water_in_JarB
  (total_alcohol / total_water) = 
  ((2 * r / (r + 1) + 1 + 3 * s / (s + 1)) / (2 / (r + 1) + 3 / (s + 1))) :=
by
  sorry

end alcohol_water_ratio_l148_148599


namespace arithmetic_sequence_common_difference_l148_148240

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 = 13) 
  (h2 : (5 * (a 1 + a 5)) / 2 = 35) 
  (h_arithmetic_sequence : ∀ n, a (n+1) = a n + d) : 
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l148_148240


namespace probability_not_in_square_b_l148_148608

theorem probability_not_in_square_b (area_A : ℝ) (perimeter_B : ℝ) 
  (area_A_eq : area_A = 30) (perimeter_B_eq : perimeter_B = 16) : 
  (14 / 30 : ℝ) = (7 / 15 : ℝ) :=
by
  sorry

end probability_not_in_square_b_l148_148608


namespace pennies_to_quarters_ratio_l148_148321

-- Define the given conditions as assumptions
variables (pennies dimes nickels quarters: ℕ)

-- Given conditions
axiom cond1 : dimes = pennies + 10
axiom cond2 : nickels = 2 * dimes
axiom cond3 : quarters = 4
axiom cond4 : nickels = 100

-- Theorem stating the final result should be a certain ratio
theorem pennies_to_quarters_ratio (hpn : pennies = 40) : pennies / quarters = 10 := 
by sorry

end pennies_to_quarters_ratio_l148_148321


namespace unit_fraction_representation_l148_148477

theorem unit_fraction_representation :
  ∃ (a b : ℕ), a > 8 ∧ b > 8 ∧ a ≠ b ∧ 1 / 8 = 1 / a + 1 / b →
  -- Count the number of such pairs
  (finset.card ((finset.filter (λ (p : ℕ × ℕ), 
     p.1 > 8 ∧ p.2 > 8 ∧ p.1 ≠ p.2 ∧ (1 / (p.1 + 0:ℚ) + 1 / (p.2 + 0:ℚ) = 1 / 8))
     (finset.Icc (8 + 1) 72).product (finset.Icc (8 + 1) 72))) = 3) :=
begin
  sorry
end

end unit_fraction_representation_l148_148477


namespace heather_distance_l148_148409

-- Definitions based on conditions
def distance_from_car_to_entrance (x : ℝ) : ℝ := x
def distance_from_entrance_to_rides (x : ℝ) : ℝ := x
def distance_from_rides_to_car : ℝ := 0.08333333333333333
def total_distance_walked : ℝ := 0.75

-- Lean statement to prove
theorem heather_distance (x : ℝ) (h : distance_from_car_to_entrance x + distance_from_entrance_to_rides x + distance_from_rides_to_car = total_distance_walked) :
  x = 0.33333333333333335 :=
by
  sorry

end heather_distance_l148_148409


namespace total_triangles_l148_148253

theorem total_triangles (small_triangles : ℕ)
    (triangles_4_small : ℕ)
    (triangles_9_small : ℕ)
    (triangles_16_small : ℕ)
    (number_small_triangles : small_triangles = 20)
    (number_triangles_4_small : triangles_4_small = 5)
    (number_triangles_9_small : triangles_9_small = 1)
    (number_triangles_16_small : triangles_16_small = 1) :
    small_triangles + triangles_4_small + triangles_9_small + triangles_16_small = 27 := 
by 
    -- proof omitted
    sorry

end total_triangles_l148_148253


namespace rectangle_same_color_exists_l148_148002

theorem rectangle_same_color_exists (grid : Fin 3 → Fin 7 → Bool) : 
  ∃ (r1 r2 c1 c2 : Fin 3), r1 ≠ r2 ∧ c1 ≠ c2 ∧ grid r1 c1 = grid r1 c2 ∧ grid r1 c1 = grid r2 c1 ∧ grid r1 c1 = grid r2 c2 :=
by
  sorry

end rectangle_same_color_exists_l148_148002


namespace find_x_given_sin_interval_l148_148098

open Real

theorem find_x_given_sin_interval (x : ℝ) (h1 : sin x = -3 / 5) (h2 : π < x ∧ x < 3 / 2 * π) :
  x = π + arcsin (3 / 5) :=
sorry

end find_x_given_sin_interval_l148_148098


namespace age_of_b_l148_148465

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 27) : b = 10 := by
  sorry

end age_of_b_l148_148465


namespace triangle_angles_and_area_l148_148117

variables (a b c : ℝ) (A B C : ℝ)

-- Conditions
def condition1 := (b - a) * (Real.sin B + Real.sin A) = c * (Real.sqrt 3 * Real.sin B - Real.sin C)
def law_of_sines := a / Real.sin A = b / Real.sin B
def law_of_sines2 := b / Real.sin B = c / Real.sin C
def law_of_cosines := b^2 + c^2 - a^2 = Real.sqrt 3 * b * c
def cosA := (b^2 + c^2 - a^2) / (2 * b * c) = Real.sqrt 3 / 2
def A_value := A = Real.pi / 6

noncomputable def area (a c B : ℝ) : ℝ := 1/2 * a * c * Real.sin B

-- Primary statement that proves the solutions
theorem triangle_angles_and_area 
  (h1 : condition1)
  (h2 : law_of_sines)
  (h3 : law_of_sines2) 
  (h4 : law_of_cosines) :
    A = Real.pi / 6 
    ∧ (A = Real.pi / 6 → (a = 2 → (B = Real.pi / 4 → area a (Real.sqrt 2 + Real.sqrt 6) B = Real.sqrt 3 + 1)))
    ∧ (A = Real.pi / 6 → (a = 2 → (c = Real.sqrt 3 * b → area 2 (2 * Real.sqrt 3) (Real.pi / 6) = Real.sqrt 3))) :=
by {
  sorry
}

end triangle_angles_and_area_l148_148117


namespace largest_digit_B_divisible_by_3_l148_148275

-- Define the six-digit number form and the known digits sum.
def isIntegerDivisibleBy3 (b : ℕ) : Prop :=
  b < 10 ∧ (b + 30) % 3 = 0

-- The main theorem: Find the largest digit B such that the number 4B5,894 is divisible by 3.
theorem largest_digit_B_divisible_by_3 : ∃ (B : ℕ), isIntegerDivisibleBy3 B ∧ ∀ (b' : ℕ), isIntegerDivisibleBy3 b' → b' ≤ B := by
  -- Notice the existential and universal quantifiers involved in finding the largest B.
  sorry

end largest_digit_B_divisible_by_3_l148_148275


namespace bus_travel_fraction_l148_148419

theorem bus_travel_fraction :
  ∃ D : ℝ, D = 30.000000000000007 ∧
            (1 / 3) * D + 2 + (18 / 30) * D = D ∧
            (18 / 30) = (3 / 5) :=
by
  sorry

end bus_travel_fraction_l148_148419


namespace speed_of_river_l148_148019

-- Definitions of the conditions
def rowing_speed_still_water := 9 -- kmph in still water
def total_time := 1 -- hour for a round trip
def total_distance := 8.84 -- km

-- Distance to the place the man rows to
def d := total_distance / 2

-- Problem statement in Lean 4
theorem speed_of_river (v : ℝ) : 
  rowing_speed_still_water = 9 ∧
  total_time = 1 ∧
  total_distance = 8.84 →
  (4.42 / (rowing_speed_still_water + v) + 4.42 / (rowing_speed_still_water - v) = 1) →
  v = 1.2 := 
by
  sorry

end speed_of_river_l148_148019


namespace sin_sum_to_product_l148_148647

-- Define the problem conditions
variable (x : ℝ)

-- State the problem and answer in Lean 4
theorem sin_sum_to_product :
  sin (5 * x) + sin (7 * x) = 2 * sin (6 * x) * cos (x) :=
sorry

end sin_sum_to_product_l148_148647


namespace tan_alpha_minus_pi_over_4_l148_148073

variable (α β : ℝ)

-- Given conditions
axiom h1 : Real.tan (α + β) = 2 / 5
axiom h2 : Real.tan β = 1 / 3

-- The goal to prove
theorem tan_alpha_minus_pi_over_4: 
  Real.tan (α - π / 4) = -8 / 9 := by
  sorry

end tan_alpha_minus_pi_over_4_l148_148073


namespace oranges_in_first_bucket_l148_148453

theorem oranges_in_first_bucket
  (x : ℕ) -- number of oranges in the first bucket
  (h1 : ∃ n, n = x) -- condition: There are some oranges in the first bucket
  (h2 : ∃ y, y = x + 17) -- condition: The second bucket has 17 more oranges than the first bucket
  (h3 : ∃ z, z = x + 6) -- condition: The third bucket has 11 fewer oranges than the second bucket
  (h4 : x + (x + 17) + (x + 6) = 89) -- condition: There are 89 oranges in all the buckets
  : x = 22 := -- conclusion: number of oranges in the first bucket is 22
sorry

end oranges_in_first_bucket_l148_148453


namespace cone_height_l148_148198

theorem cone_height (h : ℝ) (r : ℝ) 
  (volume_eq : (1/3) * π * r^2 * h = 19683 * π) 
  (isosceles_right_triangle : h = r) : 
  h = 39.0 :=
by
  -- The proof will go here
  sorry

end cone_height_l148_148198


namespace PartI_PartII_l148_148094

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Problem statement for (Ⅰ)
theorem PartI (x : ℝ) : (f x < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by sorry

-- Define conditions for PartII
variables (x y : ℝ)
def condition1 : Prop := |x - y - 1| ≤ 1 / 3
def condition2 : Prop := |2 * y + 1| ≤ 1 / 6

-- Problem statement for (Ⅱ)
theorem PartII (h1 : condition1 x y) (h2 : condition2 y) : f x < 1 :=
by sorry

end PartI_PartII_l148_148094


namespace minimum_rectangle_area_l148_148960

theorem minimum_rectangle_area (l w : ℕ) (h : 2 * (l + w) = 84) : 
  (l * w) = 41 :=
by sorry

end minimum_rectangle_area_l148_148960


namespace prove_solutions_l148_148232

noncomputable def solution1 (x : ℝ) : Prop :=
  3 * x^2 + 6 = abs (-25 + x)

theorem prove_solutions :
  solution1 ( (-1 + Real.sqrt 229) / 6 ) ∧ solution1 ( (-1 - Real.sqrt 229) / 6 ) :=
by
  sorry

end prove_solutions_l148_148232


namespace six_digit_numbers_with_at_least_one_zero_l148_148704

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l148_148704


namespace Sarah_brother_apples_l148_148915

theorem Sarah_brother_apples (n : Nat) (h1 : 45 = 5 * n) : n = 9 := 
  sorry

end Sarah_brother_apples_l148_148915


namespace smallest_n_for_divisibility_l148_148284

noncomputable def geometric_sequence (a₁ r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n-1)

theorem smallest_n_for_divisibility (h₁ : ∀ n : ℕ, geometric_sequence (1/2 : ℚ) 60 n = (1/2 : ℚ) * 60^(n-1))
    (h₂ : (60 : ℚ) * (1 / 2) = 30)
    (n : ℕ) :
  (∃ n : ℕ, n ≥ 1 ∧ (geometric_sequence (1/2 : ℚ) 60 n) ≥ 10^6) ↔ n = 7 :=
by
  sorry

end smallest_n_for_divisibility_l148_148284


namespace mod_inverse_non_existence_mod_inverse_existence_l148_148848

theorem mod_inverse_non_existence (a b c d : ℕ) (h1 : 1105 = a * b * c) (h2 : 15 = d * a) :
    ¬ ∃ x : ℕ, (15 * x) % 1105 = 1 := by sorry

theorem mod_inverse_existence (a b : ℕ) (h1 : 221 = a * b) :
    ∃ x : ℕ, (15 * x) % 221 = 59 := by sorry

end mod_inverse_non_existence_mod_inverse_existence_l148_148848


namespace center_of_tangent_circle_l148_148951

theorem center_of_tangent_circle (x y : ℝ) 
  (h1 : 3*x - 4*y = 12) 
  (h2 : 3*x - 4*y = -24)
  (h3 : x - 2*y = 0) : 
  (x, y) = (-6, -3) :=
by
  sorry

end center_of_tangent_circle_l148_148951


namespace roberta_listen_days_l148_148436

-- Define the initial number of records
def initial_records : ℕ := 8

-- Define the number of records received as gifts
def gift_records : ℕ := 12

-- Define the number of records bought
def bought_records : ℕ := 30

-- Define the number of days to listen to 1 record
def days_per_record : ℕ := 2

-- Define the total number of records
def total_records : ℕ := initial_records + gift_records + bought_records

-- Define the total number of days required to listen to all records
def total_days : ℕ := total_records * days_per_record

-- Theorem to prove the total days needed to listen to all records is 100
theorem roberta_listen_days : total_days = 100 := by
  sorry

end roberta_listen_days_l148_148436


namespace distribution_of_tickets_l148_148644

-- Define the number of total people and the number of tickets
def n : ℕ := 10
def k : ℕ := 3

-- Define the permutation function P(n, k)
def P (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Main theorem statement
theorem distribution_of_tickets : P n k = 720 := by
  unfold P
  sorry

end distribution_of_tickets_l148_148644


namespace crayons_given_correct_l148_148295

def crayons_lost : ℕ := 161
def additional_crayons : ℕ := 410
def crayons_given (lost : ℕ) (additional : ℕ) : ℕ := lost + additional

theorem crayons_given_correct : crayons_given crayons_lost additional_crayons = 571 :=
by
  sorry

end crayons_given_correct_l148_148295


namespace factorize_x_squared_minus_one_l148_148049

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l148_148049


namespace possible_sets_l148_148525

theorem possible_sets 
  (A B C : Set ℕ) 
  (U : Set ℕ := {a, b, c, d, e, f}) 
  (H1 : A ∪ B ∪ C = U) 
  (H2 : A ∩ B = {a, b, c, d}) 
  (H3 : c ∈ A ∩ B ∩ C) : 
  ∃ (n : ℕ), n = 200 :=
sorry

end possible_sets_l148_148525


namespace framing_required_l148_148615

/- 
  Problem: A 5-inch by 7-inch picture is enlarged by quadrupling its dimensions.
  A 3-inch-wide border is then placed around each side of the enlarged picture.
  What is the minimum number of linear feet of framing that must be purchased
  to go around the perimeter of the border?
-/
def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3

theorem framing_required : (2 * ((original_width * enlargement_factor + 2 * border_width) + (original_height * enlargement_factor + 2 * border_width))) / 12 = 10 :=
by
  sorry

end framing_required_l148_148615


namespace addition_example_l148_148815

theorem addition_example : 36 + 15 = 51 := 
by
  sorry

end addition_example_l148_148815


namespace value_of_m_l148_148925
noncomputable def y (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(m^2 - 3)

theorem value_of_m (m : ℝ) (x : ℝ) (h1 : x > 0) (h2 : ∀ x1 x2 : ℝ, x1 > x2 → y m x1 < y m x2) :
  m = 2 :=
sorry

end value_of_m_l148_148925


namespace wealth_ratio_l148_148361

theorem wealth_ratio (W P : ℝ) (hW_pos : 0 < W) (hP_pos : 0 < P) :
  let wX := 0.54 * W / (0.40 * P)
  let wY := 0.30 * W / (0.20 * P)
  wX / wY = 0.9 := 
by
  sorry

end wealth_ratio_l148_148361


namespace greatest_possible_x_exists_greatest_x_l148_148466

theorem greatest_possible_x (x : ℤ) (h1 : 6.1 * (10 : ℝ) ^ x < 620) : x ≤ 2 :=
sorry

theorem exists_greatest_x : ∃ x : ℤ, 6.1 * (10 : ℝ) ^ x < 620 ∧ x = 2 :=
sorry

end greatest_possible_x_exists_greatest_x_l148_148466


namespace cone_volume_is_correct_l148_148161

theorem cone_volume_is_correct (r l h : ℝ) 
  (h1 : 2 * r = Real.sqrt 2 * l)
  (h2 : π * r * l = 16 * Real.sqrt 2 * π)
  (h3 : h = r) : 
  (1 / 3) * π * r ^ 2 * h = (64 / 3) * π :=
by sorry

end cone_volume_is_correct_l148_148161


namespace primes_satisfying_condition_l148_148981

theorem primes_satisfying_condition :
    {p : ℕ | p.Prime ∧ ∀ q : ℕ, q.Prime ∧ q < p → ¬ ∃ n : ℕ, n^2 ∣ (p - (p / q) * q)} =
    {2, 3, 5, 7, 13} :=
by sorry

end primes_satisfying_condition_l148_148981


namespace x_values_l148_148654

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 1 / 3) :
  x = 3 ∨ x = 4 :=
by
  sorry

end x_values_l148_148654


namespace distinguishes_conditional_from_sequential_l148_148788

variable (C P S I D : Prop)

-- Conditions
def conditional_structure_includes_processing_box  : Prop := C = P
def conditional_structure_includes_start_end_box   : Prop := C = S
def conditional_structure_includes_io_box          : Prop := C = I
def conditional_structure_includes_decision_box    : Prop := C = D
def sequential_structure_excludes_decision_box     : Prop := ¬S = D

-- Proof problem statement
theorem distinguishes_conditional_from_sequential : C → S → I → D → P → 
    (conditional_structure_includes_processing_box C P) ∧ 
    (conditional_structure_includes_start_end_box C S) ∧ 
    (conditional_structure_includes_io_box C I) ∧ 
    (conditional_structure_includes_decision_box C D) ∧ 
    sequential_structure_excludes_decision_box S D → 
    (D = true) :=
by sorry

end distinguishes_conditional_from_sequential_l148_148788


namespace six_digit_numbers_with_zero_l148_148712

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l148_148712


namespace inheritance_shares_l148_148356

theorem inheritance_shares (A B : ℝ) (h1: A + B = 100) (h2: (1/4) * B - (1/3) * A = 11) : 
  A = 24 ∧ B = 76 := 
by 
  sorry

end inheritance_shares_l148_148356


namespace percentage_increase_l148_148642

theorem percentage_increase (employees_dec : ℝ) (employees_jan : ℝ) (inc : ℝ) (percentage : ℝ) :
  employees_dec = 470 →
  employees_jan = 408.7 →
  inc = employees_dec - employees_jan →
  percentage = (inc / employees_jan) * 100 →
  percentage = 15 := 
sorry

end percentage_increase_l148_148642


namespace dice_sum_probability_l148_148235

def four_dice_probability_sum_to_remain_die : ℚ :=
  let total_outcomes : ℚ := 6^4
  let favorable_outcomes : ℚ := 4 * 120
  favorable_outcomes / total_outcomes

theorem dice_sum_probability : four_dice_probability_sum_to_remain_die = 10 / 27 :=
  sorry

end dice_sum_probability_l148_148235


namespace unique_n_value_l148_148548

def is_n_table (n : ℕ) (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∃ i j, 
    (∀ k : Fin n, A i j ≥ A i k) ∧   -- Max in its row
    (∀ k : Fin n, A i j ≤ A k j)     -- Min in its column

theorem unique_n_value 
  {n : ℕ} (h : 2 ≤ n) 
  (A : Matrix (Fin n) (Fin n) ℕ) 
  (hA : ∀ i j, A i j ∈ Finset.range (n^2)) -- Each number appears exactly once
  (hn : is_n_table n A) : 
  ∃! a, ∃ i j, A i j = a ∧ 
           (∀ k : Fin n, a ≥ A i k) ∧ 
           (∀ k : Fin n, a ≤ A k j) := 
sorry

end unique_n_value_l148_148548


namespace failed_students_calculation_l148_148538

theorem failed_students_calculation (total_students : ℕ) (percentage_passed : ℕ)
  (h_total : total_students = 840) (h_passed : percentage_passed = 35) :
  (total_students * (100 - percentage_passed) / 100) = 546 :=
by
  sorry

end failed_students_calculation_l148_148538


namespace ratio_of_boys_l148_148893

theorem ratio_of_boys (p : ℚ) (hp : p = (3 / 4) * (1 - p)) : p = 3 / 7 :=
by
  -- Proof would be provided here
  sorry

end ratio_of_boys_l148_148893


namespace polynomial_j_value_l148_148311

noncomputable def polynomial_roots_in_ap (a d : ℝ) : Prop :=
  let r1 := a
  let r2 := a + d
  let r3 := a + 2 * d
  let r4 := a + 3 * d
  ∀ (r : ℝ), r = r1 ∨ r = r2 ∨ r = r3 ∨ r = r4

theorem polynomial_j_value (a d : ℝ) (h_ap : polynomial_roots_in_ap a d)
  (h_poly : ∀ (x : ℝ), (x - (a)) * (x - (a + d)) * (x - (a + 2 * d)) * (x - (a + 3 * d)) = x^4 + j * x^2 + k * x + 256) :
  j = -80 :=
by
  sorry

end polynomial_j_value_l148_148311


namespace ratio_condition_l148_148755

theorem ratio_condition (x y a b : ℝ) (h1 : 8 * x - 6 * y = a) 
  (h2 : 9 * y - 12 * x = b) (hx : x ≠ 0) (hy : y ≠ 0) (hb : b ≠ 0) : 
  a / b = -2 / 3 := 
by
  sorry

end ratio_condition_l148_148755


namespace Q_time_to_finish_job_l148_148942

theorem Q_time_to_finish_job :
  ∃ T_Q : ℚ, (1 / 4 + 3 / T_Q) * 3 = 19 / 20 ∧ T_Q = 15 := by
  existsi (15 : ℚ)
  split
  {
    field_simp
    norm_num
  }
  {
    norm_num
  }

end Q_time_to_finish_job_l148_148942


namespace carli_charlie_flute_ratio_l148_148031

theorem carli_charlie_flute_ratio :
  let charlie_flutes := 1
  let charlie_horns := 2
  let charlie_harps := 1
  let carli_horns := charlie_horns / 2
  let total_instruments := 7
  ∃ (carli_flutes : ℕ), 
    (charlie_flutes + charlie_horns + charlie_harps + carli_flutes + carli_horns = total_instruments) ∧ 
    (carli_flutes / charlie_flutes = 2) :=
by
  sorry

end carli_charlie_flute_ratio_l148_148031


namespace repeating_decimal_as_fraction_l148_148376

theorem repeating_decimal_as_fraction : 
  ∃ (x : ℚ), (x = 47 / 99) ∧ (x = (47 / 100 + 47 / 10000 + 47 / 1000000 + ...)) :=
sorry

end repeating_decimal_as_fraction_l148_148376


namespace six_digit_numbers_with_zero_l148_148689

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l148_148689


namespace roots_bounds_if_and_only_if_conditions_l148_148224

theorem roots_bounds_if_and_only_if_conditions (a b c : ℝ) (h : a > 0) (x1 x2 : ℝ) (hr : ∀ {x : ℝ}, a * x^2 + b * x + c = 0 → x = x1 ∨ x = x2) :
  (|x1| ≤ 1 ∧ |x2| ≤ 1) ↔ (a + b + c ≥ 0 ∧ a - b + c ≥ 0 ∧ a - c ≥ 0) :=
sorry

end roots_bounds_if_and_only_if_conditions_l148_148224


namespace cos_double_angle_l148_148398

open Real

theorem cos_double_angle (α : ℝ) (h : tan α = 3) : cos (2 * α) = -4 / 5 :=
sorry

end cos_double_angle_l148_148398


namespace six_digit_numbers_with_at_least_one_zero_correct_l148_148729

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l148_148729


namespace max_product_xy_l148_148080

theorem max_product_xy (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 2*x + y = 1) : 
  xy ≤ 1/8 :=
by sorry

end max_product_xy_l148_148080


namespace odd_number_divisibility_l148_148149

theorem odd_number_divisibility (a : ℤ) (h : a % 2 = 1) : ∃ (k : ℤ), a^4 + 9 * (9 - 2 * a^2) = 16 * k :=
by
  sorry

end odd_number_divisibility_l148_148149


namespace six_digit_numbers_with_zero_l148_148696

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148696


namespace find_200_digit_number_l148_148194

theorem find_200_digit_number : ∃ c ∈ {1, 2, 3}, ∃ (N : ℕ), N = 132 * c * 10^197 ∧ (N < 10^200 ∧ ∀ (N' : ℕ), remove_leading_and_third_digit N = N' → N = 44 * N') :=
by sorry

noncomputable def remove_leading_and_third_digit (N : ℕ) : ℕ := sorry

end find_200_digit_number_l148_148194


namespace orvin_balloons_l148_148146

def regular_price : ℕ := 2
def total_money_initial := 42 * regular_price
def pair_cost := regular_price + (regular_price / 2)
def pairs := total_money_initial / pair_cost
def balloons_from_sale := pairs * 2

def extra_money : ℕ := 18
def price_per_additional_balloon := 2 * regular_price
def additional_balloons := extra_money / price_per_additional_balloon
def greatest_number_of_balloons := balloons_from_sale + additional_balloons

theorem orvin_balloons (pairs balloons_from_sale additional_balloons greatest_number_of_balloons : ℕ) :
  pairs * 2 = 56 →
  additional_balloons = 4 →
  greatest_number_of_balloons = 60 :=
by
  sorry

end orvin_balloons_l148_148146


namespace project_completion_time_l148_148984

theorem project_completion_time (m n : ℝ) (hm : m > 0) (hn : n > 0):
  (1 / (1 / m + 1 / n)) = (m * n) / (m + n) :=
by
  sorry

end project_completion_time_l148_148984


namespace triangle_foci_angle_90_l148_148140

noncomputable def ellipse_point (x y : ℝ) : Prop :=
  (x^2 / 100 + y^2 / 36 = 1)

def foci1 : ℝ × ℝ := (-8, 0)

def foci2 : ℝ × ℝ := (8, 0)

def triangle_area (P : ℝ × ℝ) : Prop :=
  let ⟨x, y⟩ := P in
  36 = 0.5 * abs ((x * (0 - 0)) + (-8 * (0 - y)) + (8 * (y - 0)))

def angle_condition (P : ℝ × ℝ) : Prop :=
  let ⟨x, y⟩ := P in
  (foci1.1 - x) * (foci2.1 - x) + (foci1.2 - y) * (foci2.2 - y) = 90

theorem triangle_foci_angle_90 (P : ℝ × ℝ) (h1 : ellipse_point P.1 P.2)
  (h2 : triangle_area P) : angle_condition P :=
sorry

end triangle_foci_angle_90_l148_148140


namespace necessary_but_not_sufficient_cond_l148_148280

noncomputable
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_cond (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (hseq : geometric_sequence a a1 q)
  (hpos : a1 > 0) :
  (q < 0 ↔ (∀ n : ℕ, a (2 * n + 1) + a (2 * n + 2) < 0)) :=
sorry

end necessary_but_not_sufficient_cond_l148_148280


namespace option_A_option_C_option_D_l148_148182

noncomputable def ratio_12_11 := (12 : ℝ) / 11
noncomputable def ratio_11_10 := (11 : ℝ) / 10

theorem option_A : ratio_12_11^11 > ratio_11_10^10 := sorry

theorem option_C : ratio_12_11^10 > ratio_11_10^9 := sorry

theorem option_D : ratio_11_10^12 > ratio_12_11^13 := sorry

end option_A_option_C_option_D_l148_148182


namespace six_digit_numbers_with_zero_l148_148730

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148730


namespace six_digit_numbers_with_at_least_one_zero_l148_148705

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l148_148705


namespace car_return_speed_l148_148819

noncomputable def round_trip_speed (d : ℝ) (r : ℝ) : ℝ :=
  let travel_time_to_B := d / 75
  let break_time := 1 / 2
  let travel_time_to_A := d / r
  let total_time := travel_time_to_B + travel_time_to_A + break_time
  let total_distance := 2 * d
  total_distance / total_time

theorem car_return_speed :
  let d := 150
  let avg_speed := 50
  round_trip_speed d 42.857 = avg_speed :=
by
  sorry

end car_return_speed_l148_148819


namespace find_x_l148_148652

theorem find_x (x y : ℕ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end find_x_l148_148652


namespace d_is_greatest_l148_148282

variable (p : ℝ)

def a := p - 1
def b := p + 2
def c := p - 3
def d := p + 4

theorem d_is_greatest : d > b ∧ d > a ∧ d > c := 
by sorry

end d_is_greatest_l148_148282


namespace pears_value_l148_148159

-- Condition: 3/4 of 12 apples is equivalent to 6 pears
def apples_to_pears (a p : ℕ) : Prop := (3 / 4) * a = 6 * p

-- Target: 1/3 of 9 apples is equivalent to 2 pears
def target_equiv : Prop := (1 / 3) * 9 = 2

theorem pears_value (a p : ℕ) (h : apples_to_pears 12 6) : target_equiv := by
  sorry

end pears_value_l148_148159


namespace available_spaces_l148_148202

noncomputable def numberOfBenches : ℕ := 50
noncomputable def capacityPerBench : ℕ := 4
noncomputable def peopleSeated : ℕ := 80

theorem available_spaces :
  let totalCapacity := numberOfBenches * capacityPerBench;
  let availableSpaces := totalCapacity - peopleSeated;
  availableSpaces = 120 := by
    sorry

end available_spaces_l148_148202


namespace min_value_n_l148_148103

theorem min_value_n (n : ℕ) (h1 : 4 ∣ 60 * n) (h2 : 8 ∣ 60 * n) : n = 1 := 
  sorry

end min_value_n_l148_148103


namespace factorize_x_squared_minus_1_l148_148043

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l148_148043


namespace one_course_common_l148_148354

theorem one_course_common (A_can_choose B_can_choose : Finset ℕ) (n : ℕ) (hn : n = 4) 
  (hA : A_can_choose.card = 2) (hB : B_can_choose.card = 2) (hAB : ∃ x ∈ A_can_choose, x ∈ B_can_choose) :
  ∃! x, x = 24 := by
  sorry

end one_course_common_l148_148354


namespace find_numbers_with_sum_and_product_l148_148575

theorem find_numbers_with_sum_and_product (S P : ℝ) :
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  (x1 + y1 = S ∧ x1 * y1 = P) ∨ (x2 + y2 = S ∧ x2 * y2 = P) :=
sorry

end find_numbers_with_sum_and_product_l148_148575


namespace sum_of_tens_and_ones_digit_of_7_pow_25_l148_148459

theorem sum_of_tens_and_ones_digit_of_7_pow_25 : 
  let n := 7 ^ 25 
  let ones_digit := n % 10 
  let tens_digit := (n / 10) % 10 
  ones_digit + tens_digit = 11 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_25_l148_148459


namespace unique_positive_b_discriminant_zero_l148_148367

theorem unique_positive_b_discriminant_zero (c : ℚ) : 
  (∃! b : ℚ, b > 0 ∧ (b^2 + 3*b + 1/b)^2 - 4*c = 0) ↔ c = -1/2 :=
sorry

end unique_positive_b_discriminant_zero_l148_148367


namespace min_value_of_sum_of_squares_l148_148138

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4.8 :=
sorry

end min_value_of_sum_of_squares_l148_148138


namespace mean_second_set_l148_148109

theorem mean_second_set (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 62) :
  (48 + 62 + 98 + 124 + x) / 5 = 78 :=
sorry

end mean_second_set_l148_148109


namespace seats_per_bus_l148_148171

theorem seats_per_bus (students buses : ℕ) (h1 : students = 14) (h2 : buses = 7) : students / buses = 2 := by
  sorry

end seats_per_bus_l148_148171


namespace contrapositive_proof_l148_148163

theorem contrapositive_proof (a b : ℝ) : 
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) :=
sorry

end contrapositive_proof_l148_148163


namespace part_II_l148_148657

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 + (a - 1) * x - Real.log x

theorem part_II (a : ℝ) (h : a > 0) :
  ∀ x > 0, f a x ≥ 2 - (3 / (2 * a)) :=
sorry

end part_II_l148_148657


namespace constant_term_expansion_l148_148162

noncomputable def binomial_constant_term : ℤ :=
let x := 1 in
let expr := x^2 + (1 / x^2 : ℚ) - (2 : ℚ) in
∑ i in range 7, (binomial 6 i) * (-1)^i * x^(6 - 2 * i)

theorem constant_term_expansion : binomial_constant_term = -20 := by
  have h1 : binomial_constant_term = 6.choose 3 * (-1)^3 := sorry,
  have h2 : 6.choose 3 = 20 := by norm_num,
  calc
    binomial_constant_term
        = 6.choose 3 * (-1)^3 : by rw [h1]
    ... = 20 * (-1) : by rw [h2]
    ... = -20 : by norm_num

end constant_term_expansion_l148_148162


namespace min_ab_value_l148_148867

variable (a b : ℝ)

theorem min_ab_value (h1 : a > -1) (h2 : b > -2) (h3 : (a+1) * (b+2) = 16) : a + b ≥ 5 :=
by
  sorry

end min_ab_value_l148_148867


namespace smallest_number_divisible_by_6_is_123456_l148_148771

open Finset

def is_six_digit_number_with_1_to_6 := {n : ℕ | (multiset.of_nat_digits (Finset.univ.map coe )).to_finset = ({1, 2, 3, 4, 5, 6} : Finset ℕ) }

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

noncomputable def smallest_divisible_by_6 : ℕ :=
  Finset.min' (filter is_divisible_by_6 (is_six_digit_number_with_1_to_6)) sorry

theorem smallest_number_divisible_by_6_is_123456 : smallest_divisible_by_6 = 123456 :=
sorry

end smallest_number_divisible_by_6_is_123456_l148_148771


namespace select_team_with_smaller_variance_l148_148616

theorem select_team_with_smaller_variance 
    (variance_A variance_B : ℝ)
    (hA : variance_A = 1.5)
    (hB : variance_B = 2.8)
    : variance_A < variance_B → "Team A" = "Team A" :=
by
  intros h
  sorry

end select_team_with_smaller_variance_l148_148616


namespace intersection_of_sets_l148_148523

noncomputable def A : Set ℝ := {x | -1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 5}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_of_sets : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := 
by
  sorry

end intersection_of_sets_l148_148523


namespace range_of_m_l148_148285

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f m x < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end range_of_m_l148_148285


namespace Lauryn_employees_l148_148424

variables (M W : ℕ)

theorem Lauryn_employees (h1 : M = W - 20) (h2 : M + W = 180) : M = 80 :=
by {
    sorry
}

end Lauryn_employees_l148_148424


namespace fair_attendance_l148_148431

-- Define the variables x, y, and z
variables (x y z : ℕ)

-- Define the conditions given in the problem
def condition1 := z = 2 * y
def condition2 := x = z - 200
def condition3 := y = 600

-- State the main theorem proving the values of x, y, and z
theorem fair_attendance : condition1 y z → condition2 x z → condition3 y → (x = 1000 ∧ y = 600 ∧ z = 1200) := by
  intros h1 h2 h3
  sorry

end fair_attendance_l148_148431


namespace six_digit_numbers_with_zero_count_l148_148680

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l148_148680


namespace andy_demerits_for_joke_l148_148630

def max_demerits := 50
def demerits_late_per_instance := 2
def instances_late := 6
def remaining_demerits := 23
def total_demerits := max_demerits - remaining_demerits
def demerits_late := demerits_late_per_instance * instances_late
def demerits_joke := total_demerits - demerits_late

theorem andy_demerits_for_joke : demerits_joke = 15 := by
  sorry

end andy_demerits_for_joke_l148_148630


namespace digits_sum_is_15_l148_148946

theorem digits_sum_is_15 (f o g : ℕ) (h1 : f * 100 + o * 10 + g = 366) (h2 : 4 * (f * 100 + o * 10 + g) = 1464) (h3 : f < 10 ∧ o < 10 ∧ g < 10) :
  f + o + g = 15 :=
sorry

end digits_sum_is_15_l148_148946


namespace sum_of_tens_and_ones_digit_3_add_4_pow_25_l148_148458

theorem sum_of_tens_and_ones_digit_3_add_4_pow_25 : 
    let n := (3 + 4) ^ 25
    in (n / 10 % 10) + (n % 10) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_3_add_4_pow_25_l148_148458


namespace largest_sum_of_digits_in_display_l148_148476

-- Define the conditions
def is_valid_hour (h : Nat) : Prop := 0 <= h ∧ h < 24
def is_valid_minute (m : Nat) : Prop := 0 <= m ∧ m < 60

-- Define helper functions to convert numbers to their digit sums
def digit_sum (n : Nat) : Nat :=
  n.digits 10 |>.sum

-- Define the largest possible sum of the digits condition
def largest_possible_digit_sum : Prop :=
  ∀ (h m : Nat), is_valid_hour h → is_valid_minute m → 
    digit_sum (h.div 10 + h % 10) + digit_sum (m.div 10 + m % 10) ≤ 24 ∧
    ∃ (h m : Nat), is_valid_hour h ∧ is_valid_minute m ∧ digit_sum (h.div 10 + h % 10) + digit_sum (m.div 10 + m % 10) = 24

-- The statement to prove
theorem largest_sum_of_digits_in_display : largest_possible_digit_sum :=
by
  sorry

end largest_sum_of_digits_in_display_l148_148476


namespace six_digit_numbers_with_at_least_one_zero_l148_148703

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l148_148703


namespace angle_covered_in_three_layers_l148_148338

/-- Define the conditions: A 90-degree angle, sum of angles is 290 degrees,
    and prove the angle covered in three layers is 110 degrees. -/
theorem angle_covered_in_three_layers {α β : ℝ}
  (h1 : α + β = 90)
  (h2 : 2*α + 3*β = 290) :
  β = 110 := 
sorry

end angle_covered_in_three_layers_l148_148338


namespace count_big_boxes_l148_148217

theorem count_big_boxes (B : ℕ) (h : 7 * B + 4 * 9 = 71) : B = 5 :=
sorry

end count_big_boxes_l148_148217


namespace arithmetic_sequence_third_term_l148_148112

theorem arithmetic_sequence_third_term {a d : ℝ} (h : 2 * a + 4 * d = 10) : a + 2 * d = 5 :=
sorry

end arithmetic_sequence_third_term_l148_148112


namespace solve_fractions_in_integers_l148_148440

theorem solve_fractions_in_integers :
  ∀ (a b c : ℤ), (1 / a + 1 / b + 1 / c = 1) ↔
  (a = 3 ∧ b = 3 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 6) ∨
  (a = 2 ∧ b = 4 ∧ c = 4) ∨
  (a = 1 ∧ ∃ t : ℤ, b = t ∧ c = -t) :=
by {
  sorry
}

end solve_fractions_in_integers_l148_148440


namespace six_digit_numbers_with_zero_count_l148_148678

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l148_148678


namespace factorize_difference_of_squares_l148_148048

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l148_148048


namespace least_number_of_pennies_l148_148939

theorem least_number_of_pennies (a : ℕ) :
  (a ≡ 1 [MOD 7]) ∧ (a ≡ 0 [MOD 3]) → a = 15 := by
  sorry

end least_number_of_pennies_l148_148939


namespace six_digit_numbers_with_zero_l148_148694

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l148_148694


namespace negate_proposition_l148_148446

theorem negate_proposition :
    (¬ ∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by
  sorry

end negate_proposition_l148_148446


namespace population_total_l148_148346

variable (x y : ℕ)

theorem population_total (h1 : 20 * y = 12 * y * (x + y)) : x + y = 240 :=
  by
  -- Proceed with solving the provided conditions.
  sorry

end population_total_l148_148346


namespace quadratic_no_ten_powers_of_2_values_l148_148914

theorem quadratic_no_ten_powers_of_2_values 
  (a b : ℝ) :
  ¬ ∃ (j : ℤ), ∀ k : ℤ, j ≤ k ∧ k < j + 10 → ∃ n : ℕ, (k^2 + a * k + b) = 2 ^ n :=
by sorry

end quadratic_no_ten_powers_of_2_values_l148_148914


namespace enclosed_area_is_43pi_l148_148363

noncomputable def enclosed_area (x y : ℝ) : Prop :=
  (x^2 - 6*x + y^2 + 10*y = 9)

theorem enclosed_area_is_43pi :
  (∃ x y : ℝ, enclosed_area x y) → 
  ∃ A : ℝ, A = 43 * Real.pi :=
by
  sorry

end enclosed_area_is_43pi_l148_148363


namespace quadratic_binomial_square_l148_148974

theorem quadratic_binomial_square (a : ℚ) :
  (∃ r s : ℚ, (ax^2 + 22*x + 9 = (r*x + s)^2) ∧ s = 3 ∧ r = 11 / 3) → a = 121 / 9 := 
by 
  sorry

end quadratic_binomial_square_l148_148974


namespace trigonometric_identity_tan_two_l148_148866

theorem trigonometric_identity_tan_two (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 11 / 5 :=
by
  sorry

end trigonometric_identity_tan_two_l148_148866


namespace rope_segments_after_folding_l148_148557

theorem rope_segments_after_folding (n : ℕ) (h : n = 6) : 2^n + 1 = 65 :=
by
  rw [h]
  norm_num

end rope_segments_after_folding_l148_148557


namespace greatest_second_term_arithmetic_sequence_l148_148930

theorem greatest_second_term_arithmetic_sequence:
  ∃ a d : ℕ, (a > 0) ∧ (d > 0) ∧ (2 * a + 3 * d = 29) ∧ (4 * a + 6 * d = 58) ∧ (((a + d : ℤ) / 3 : ℤ) = 10) :=
sorry

end greatest_second_term_arithmetic_sequence_l148_148930


namespace geometric_sequence_ratio_l148_148996

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n+1) = q * a n)
  (h_a1 : a 1 = 4)
  (h_a4 : a 4 = 1/2) :
  q = 1/2 :=
sorry

end geometric_sequence_ratio_l148_148996


namespace train_bus_difference_l148_148350

variable (T : ℝ)  -- T is the cost of a train ride

-- conditions
def cond1 := T + 1.50 = 9.85
def cond2 := 1.50 = 1.50

theorem train_bus_difference (h1 : cond1 T) (h2 : cond2) : T - 1.50 = 6.85 := 
sorry

end train_bus_difference_l148_148350


namespace smallest_k_multiple_of_360_l148_148860

theorem smallest_k_multiple_of_360 :
  ∃ (k : ℕ), (k > 0 ∧ (k = 432) ∧ (2160 ∣ k * (k + 1) * (2 * k + 1))) :=
by
  complication_sorry_proved

end smallest_k_multiple_of_360_l148_148860


namespace six_digit_numbers_with_zero_l148_148673

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l148_148673


namespace not_integer_20_diff_l148_148035

theorem not_integer_20_diff (a b : ℝ) (hne : a ≠ b) 
  (no_roots1 : ∀ x, x^2 + 20 * a * x + 10 * b ≠ 0) 
  (no_roots2 : ∀ x, x^2 + 20 * b * x + 10 * a ≠ 0) : 
  ¬ (∃ k : ℤ, 20 * (b - a) = k) :=
by
  sorry

end not_integer_20_diff_l148_148035


namespace total_questions_in_two_hours_l148_148391

theorem total_questions_in_two_hours (r : ℝ) : 
  let Fiona_questions := 36 
  let Shirley_questions := Fiona_questions * r
  let Kiana_questions := (Fiona_questions + Shirley_questions) / 2
  let one_hour_total := Fiona_questions + Shirley_questions + Kiana_questions
  let two_hour_total := 2 * one_hour_total
  two_hour_total = 108 + 108 * r :=
by
  sorry

end total_questions_in_two_hours_l148_148391


namespace sum_modulo_seven_l148_148487

theorem sum_modulo_seven (a b c : ℕ) (h1: a = 9^5) (h2: b = 8^6) (h3: c = 7^7) :
  (a + b + c) % 7 = 5 :=
by sorry

end sum_modulo_seven_l148_148487


namespace negation_of_P_is_there_exists_x_ge_0_l148_148865

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

-- State the theorem of the negation of P
theorem negation_of_P_is_there_exists_x_ge_0 : ¬P ↔ ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
by sorry

end negation_of_P_is_there_exists_x_ge_0_l148_148865


namespace good_numbers_product_sum_digits_not_equal_l148_148286

def is_good_number (n : ℕ) : Prop :=
  n.digits 10 ⊆ [0, 1]

theorem good_numbers_product_sum_digits_not_equal (A B : ℕ) (hA : is_good_number A) (hB : is_good_number B) (hAB : is_good_number (A * B)) :
  ¬ ( (A.digits 10).sum * (B.digits 10).sum = ((A * B).digits 10).sum ) :=
sorry

end good_numbers_product_sum_digits_not_equal_l148_148286


namespace quadratic_roots_sum_squares_l148_148088

theorem quadratic_roots_sum_squares {a b : ℝ} 
  (h₁ : a + b = -1) 
  (h₂ : a * b = -5) : 
  2 * a^2 + a + b^2 = 16 :=
by sorry

end quadratic_roots_sum_squares_l148_148088


namespace six_digit_numbers_with_zero_l148_148681

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148681


namespace find_a_of_extremum_l148_148883

theorem find_a_of_extremum (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h1 : f x = x^3 + a*x^2 + b*x + a^2)
  (h2 : f' x = 3*x^2 + 2*a*x + b)
  (h3 : f' 1 = 0)
  (h4 : f 1 = 10) : a = 4 := by
  sorry

end find_a_of_extremum_l148_148883


namespace maynard_dog_holes_l148_148909

open Real

theorem maynard_dog_holes (h_filled : ℝ) (h_unfilled : ℝ) (percent_filled : ℝ) 
  (percent_unfilled : ℝ) (total_holes : ℝ) :
  percent_filled = 0.75 →
  percent_unfilled = 0.25 →
  h_unfilled = 2 →
  h_filled = total_holes * percent_filled →
  total_holes = 8 :=
by
  intros hf pu hu hf_total
  sorry

end maynard_dog_holes_l148_148909


namespace population_of_males_l148_148896

theorem population_of_males (total_population : ℕ) (num_parts : ℕ) (part_population : ℕ) 
  (male_population : ℕ) (female_population : ℕ) (children_population : ℕ) :
  total_population = 600 →
  num_parts = 4 →
  part_population = total_population / num_parts →
  children_population = 2 * male_population →
  male_population = part_population →
  male_population = 150 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end population_of_males_l148_148896


namespace total_metal_rods_needed_l148_148825

-- Definitions extracted from the conditions
def metal_sheets_per_panel := 3
def metal_beams_per_panel := 2
def panels := 10
def rods_per_sheet := 10
def rods_per_beam := 4

-- Problem statement: Prove the total number of metal rods required is 380
theorem total_metal_rods_needed :
  (panels * ((metal_sheets_per_panel * rods_per_sheet) + (metal_beams_per_panel * rods_per_beam))) = 380 :=
by
  sorry

end total_metal_rods_needed_l148_148825


namespace find_other_number_l148_148189

theorem find_other_number (B : ℕ) (HCF : ℕ) (LCM : ℕ) (A : ℕ) 
  (h1 : A = 24) 
  (h2 : HCF = 16) 
  (h3 : LCM = 312) 
  (h4 : HCF * LCM = A * B) :
  B = 208 :=
by
  sorry

end find_other_number_l148_148189


namespace cube_painting_problem_l148_148207

theorem cube_painting_problem (n : ℕ) (hn : n > 0) :
  (6 * n^2 = (6 * n^3) / 3) ↔ n = 3 :=
by sorry

end cube_painting_problem_l148_148207


namespace fraction_value_l148_148488

theorem fraction_value :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1 : ℚ) / (2 - 4 + 6 - 8 + 10 - 12 + 14) = 3/4 :=
sorry

end fraction_value_l148_148488


namespace number_of_buses_l148_148450

theorem number_of_buses (x y : ℕ) (h1 : x + y = 40) (h2 : 6 * x + 4 * y = 210) : x = 25 :=
by
  sorry

end number_of_buses_l148_148450


namespace total_cost_of_video_games_l148_148593

theorem total_cost_of_video_games :
  let cost_football_game := 14.02
  let cost_strategy_game := 9.46
  let cost_batman_game := 12.04
  let total_cost := cost_football_game + cost_strategy_game + cost_batman_game
  total_cost = 35.52 :=
by
  -- Proof goes here
  sorry

end total_cost_of_video_games_l148_148593


namespace find_200_digit_number_l148_148195

noncomputable def original_number_condition (N : ℕ) (c : ℕ) (k : ℕ) : Prop :=
  let m := 0
  let a := 2 * c
  let b := 3 * c
  k = 197 ∧ (c = 1 ∨ c = 2 ∨ c = 3) ∧ N = 132 * c * 10^197

theorem find_200_digit_number :
  ∃ N c, original_number_condition N c 197 :=
by
  sorry

end find_200_digit_number_l148_148195


namespace power_function_is_odd_l148_148869

open Function

noncomputable def power_function (a : ℝ) (b : ℝ) : ℝ → ℝ := λ x => (a - 1) * x^b

theorem power_function_is_odd (a b : ℝ) (h : power_function a b a = 1 / 8)
  :  a = 2 ∧ b = -3 → (∀ x : ℝ, power_function a b (-x) = -power_function a b x) :=
by
  intro ha hb
  -- proofs can be filled later with details
  sorry

end power_function_is_odd_l148_148869


namespace hyperbola_center_l148_148344

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h₁ : x1 = 3) (h₂ : y1 = 2) (h₃ : x2 = 11) (h₄ : y2 = 6) :
  (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 4 :=
by
  -- Use the conditions h₁, h₂, h₃, and h₄ to substitute values and prove the statement
  sorry

end hyperbola_center_l148_148344


namespace work_problem_l148_148472

theorem work_problem (x : ℝ) (hx : x > 0)
    (hB : B_work_rate = 1 / 18)
    (hTogether : together_work_rate = 1 / 7.2)
    (hCombined : together_work_rate = 1 / x + B_work_rate) :
    x = 2 := by
    sorry

end work_problem_l148_148472


namespace time_to_meet_l148_148912

variables {v_g : ℝ} -- speed of Petya and Vasya on the dirt road
variable {v_a : ℝ} -- speed of Petya on the paved road
variable {t : ℝ} -- time from start until Petya and Vasya meet
variable {x : ℝ} -- distance from the starting point to the bridge

-- Conditions
axiom Petya_speed : v_a = 3 * v_g
axiom Petya_bridge_time : x / v_a = 1
axiom Vasya_speed : v_a ≠ 0 ∧ v_g ≠ 0

-- Statement
theorem time_to_meet (h1 : v_a = 3 * v_g) (h2 : x / v_a = 1) (h3 : v_a ≠ 0 ∧ v_g ≠ 0) : t = 2 :=
by
  have h4 : x = 3 * v_g := sorry
  have h5 : (2 * x - 2 * v_g) / (2 * v_g) = 1 := sorry
  exact sorry

end time_to_meet_l148_148912


namespace necessary_and_sufficient_condition_l148_148770

def f (a x : ℝ) : ℝ := x^2 - a * x + 1

theorem necessary_and_sufficient_condition (a : ℝ) : 
  (∃ x : ℝ, f a x < 0) ↔ |a| > 2 :=
by
  sorry

end necessary_and_sufficient_condition_l148_148770


namespace arithmetic_sequence_sum_l148_148404

noncomputable def first_21_sum (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ) : ℝ :=
  let a1 := a 1
  let a21 := a 21
  21 * (a1 + a21) / 2

theorem arithmetic_sequence_sum
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_symmetry : ∀ x, f (x + 1) = f (-(x + 1)))
  (h_monotonic : ∀ x y, 1 < x → x < y → f x < f y)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_f_eq : f (a 4) = f (a 18))
  (h_non_zero_diff : d ≠ 0) :
  first_21_sum f a d = 21 := by
  sorry

end arithmetic_sequence_sum_l148_148404


namespace Jasmine_shoe_size_l148_148543

theorem Jasmine_shoe_size (J A : ℕ) (h1 : A = 2 * J) (h2 : J + A = 21) : J = 7 :=
by 
  sorry

end Jasmine_shoe_size_l148_148543


namespace bird_families_migration_l148_148185

theorem bird_families_migration 
  (total_families : ℕ)
  (africa_families : ℕ)
  (asia_families : ℕ)
  (south_america_families : ℕ)
  (africa_days : ℕ)
  (asia_days : ℕ)
  (south_america_days : ℕ)
  (migrated_families : ℕ)
  (remaining_families : ℕ)
  (total_migration_time : ℕ)
  (H1 : total_families = 200)
  (H2 : africa_families = 60)
  (H3 : asia_families = 95)
  (H4 : south_america_families = 30)
  (H5 : africa_days = 7)
  (H6 : asia_days = 14)
  (H7 : south_america_days = 10)
  (H8 : migrated_families = africa_families + asia_families + south_america_families)
  (H9 : remaining_families = total_families - migrated_families)
  (H10 : total_migration_time = 
          africa_families * africa_days + 
          asia_families * asia_days + 
          south_america_families * south_america_days) :
  remaining_families = 15 ∧ total_migration_time = 2050 :=
by
  sorry

end bird_families_migration_l148_148185


namespace integral_3x_plus_sin_x_l148_148218

theorem integral_3x_plus_sin_x :
  ∫ x in (0 : ℝ)..(π / 2), (3 * x + Real.sin x) = (3 / 8) * π^2 + 1 :=
by
  sorry

end integral_3x_plus_sin_x_l148_148218


namespace evaluate_custom_operation_l148_148100

def custom_operation (x y : ℕ) : ℕ := 2 * x - 4 * y

theorem evaluate_custom_operation :
  custom_operation 7 3 = 2 :=
by
  sorry

end evaluate_custom_operation_l148_148100


namespace intersection_of_lines_l148_148805

theorem intersection_of_lines : ∃ (x y : ℚ), y = -3 * x + 1 ∧ y + 5 = 15 * x - 2 ∧ x = 1 / 3 ∧ y = 0 :=
by
  sorry

end intersection_of_lines_l148_148805


namespace n_value_condition_l148_148395

theorem n_value_condition (n : ℤ) : 
  (3 * (n ^ 2 + n) + 7) % 5 = 0 ↔ n % 5 = 2 := sorry

end n_value_condition_l148_148395


namespace min_value_reciprocal_l148_148086

theorem min_value_reciprocal (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
by
  sorry

end min_value_reciprocal_l148_148086


namespace max_trading_cards_l148_148276

theorem max_trading_cards (h : 10 ≥ 1.25 * nat):
  nat ≤ 8 :=
sorry

end max_trading_cards_l148_148276


namespace european_savings_correct_l148_148956

noncomputable def movie_ticket_price : ℝ := 8
noncomputable def popcorn_price : ℝ := 8 - 3
noncomputable def drink_price : ℝ := popcorn_price + 1
noncomputable def candy_price : ℝ := drink_price / 2
noncomputable def hotdog_price : ℝ := 5

noncomputable def monday_discount_popcorn : ℝ := 0.15 * popcorn_price
noncomputable def wednesday_discount_candy : ℝ := 0.10 * candy_price
noncomputable def friday_discount_drink : ℝ := 0.05 * drink_price

noncomputable def monday_price : ℝ := 22
noncomputable def wednesday_price : ℝ := 20
noncomputable def friday_price : ℝ := 25
noncomputable def weekend_price : ℝ := 25
noncomputable def monday_exchange_rate : ℝ := 0.85
noncomputable def wednesday_exchange_rate : ℝ := 0.85
noncomputable def friday_exchange_rate : ℝ := 0.83
noncomputable def weekend_exchange_rate : ℝ := 0.81

noncomputable def total_cost_monday : ℝ := movie_ticket_price + (popcorn_price - monday_discount_popcorn) + drink_price + candy_price + hotdog_price
noncomputable def savings_monday_usd : ℝ := total_cost_monday - monday_price
noncomputable def savings_monday_eur : ℝ := savings_monday_usd * monday_exchange_rate

noncomputable def total_cost_wednesday : ℝ := movie_ticket_price + popcorn_price + drink_price + (candy_price - wednesday_discount_candy) + hotdog_price
noncomputable def savings_wednesday_usd : ℝ := total_cost_wednesday - wednesday_price
noncomputable def savings_wednesday_eur : ℝ := savings_wednesday_usd * wednesday_exchange_rate

noncomputable def total_cost_friday : ℝ := movie_ticket_price + popcorn_price + (drink_price - friday_discount_drink) + candy_price + hotdog_price
noncomputable def savings_friday_usd : ℝ := total_cost_friday - friday_price
noncomputable def savings_friday_eur : ℝ := savings_friday_usd * friday_exchange_rate

noncomputable def total_cost_weekend : ℝ := movie_ticket_price + popcorn_price + drink_price + candy_price + hotdog_price
noncomputable def savings_weekend_usd : ℝ := total_cost_weekend - weekend_price
noncomputable def savings_weekend_eur : ℝ := savings_weekend_usd * weekend_exchange_rate

theorem european_savings_correct :
  savings_monday_eur = 3.61 ∧ 
  savings_wednesday_eur = 5.70 ∧ 
  savings_friday_eur = 1.41 ∧ 
  savings_weekend_eur = 1.62 :=
by
  sorry

end european_savings_correct_l148_148956


namespace delegates_probability_mn_equal_47_l148_148435

theorem delegates_probability_mn_equal_47 :
  ∃ (m n : ℕ), Nat.coprime m n ∧ (∀ (arrangements : ℕ) (invalid_arrangements : ℕ),
  arrangements = Nat.factorial 8 ∧
  invalid_arrangements = 3 * Nat.factorial 6 * 6 + 3 * 6 * 6 * 4 - 2 * 216 ∧
  (arrangements - invalid_arrangements) / arrangements = m / n) ∧ (m + n = 47) :=
sorry

end delegates_probability_mn_equal_47_l148_148435


namespace number_of_shoes_lost_l148_148141

-- Definitions for the problem conditions
def original_pairs : ℕ := 20
def pairs_left : ℕ := 15
def shoes_per_pair : ℕ := 2

-- Translating the conditions to individual shoe counts
def original_shoes : ℕ := original_pairs * shoes_per_pair
def remaining_shoes : ℕ := pairs_left * shoes_per_pair

-- Statement of the proof problem
theorem number_of_shoes_lost : original_shoes - remaining_shoes = 10 := by
  sorry

end number_of_shoes_lost_l148_148141


namespace mark_parking_tickets_eq_l148_148632

def total_tickets : ℕ := 24
def sarah_speeding_tickets : ℕ := 6
def mark_speeding_tickets : ℕ := 6
def sarah_parking_tickets (S : ℕ) := S
def mark_parking_tickets (S : ℕ) := 2 * S
def total_traffic_tickets (S : ℕ) := S + 2 * S + sarah_speeding_tickets + mark_speeding_tickets

theorem mark_parking_tickets_eq (S : ℕ) (h1 : total_traffic_tickets S = total_tickets)
  (h2 : sarah_speeding_tickets = 6) (h3 : mark_speeding_tickets = 6) :
  mark_parking_tickets S = 8 :=
sorry

end mark_parking_tickets_eq_l148_148632


namespace find_fx2_l148_148245

theorem find_fx2 (f : ℝ → ℝ) (x : ℝ) (h : f (x - 1) = x ^ 2) : f (x ^ 2) = (x ^ 2 + 1) ^ 2 := by
  sorry

end find_fx2_l148_148245


namespace max_value_5x_minus_25x_l148_148383

open Real

theorem max_value_5x_minus_25x : 
  ∃ x : ℝ, ∀ y : ℝ, (y = 5^x) → (y - y^2) ≤ 1 / 4 := 
by 
  sorry

end max_value_5x_minus_25x_l148_148383


namespace primes_infinite_l148_148151

theorem primes_infinite : ∀ (S : Set ℕ), (∀ p, p ∈ S → Nat.Prime p) → (∃ a, a ∉ S ∧ Nat.Prime a) :=
by
  sorry

end primes_infinite_l148_148151


namespace eugene_total_payment_l148_148271

-- Define the initial costs of items
def cost_tshirt := 20
def cost_pants := 80
def cost_shoes := 150

-- Define the quantities
def quantity_tshirt := 4
def quantity_pants := 3
def quantity_shoes := 2

-- Define the discount rate
def discount_rate := 0.10

-- Define the total pre-discount cost
def pre_discount_cost :=
  (cost_tshirt * quantity_tshirt) +
  (cost_pants * quantity_pants) +
  (cost_shoes * quantity_shoes)

-- Define the discount amount
def discount_amount := discount_rate * pre_discount_cost

-- Define the post-discount cost
def post_discount_cost := pre_discount_cost - discount_amount

-- Theorem statement
theorem eugene_total_payment : post_discount_cost = 558 := by
  sorry

end eugene_total_payment_l148_148271


namespace distance_to_lake_l148_148212

theorem distance_to_lake (d : ℝ) :
  ¬ (d ≥ 10) → ¬ (d ≤ 9) → d ≠ 7 → d ∈ Set.Ioo 9 10 :=
by
  intros h1 h2 h3
  sorry

end distance_to_lake_l148_148212


namespace pebble_difference_l148_148635

-- Definitions and conditions
variables (x : ℚ) -- we use rational numbers for exact division
def Candy := 2 * x
def Lance := 5 * x
def Sandy := 4 * x
def condition1 := Lance = Candy + 10

-- Theorem statement
theorem pebble_difference (h : condition1) : Lance + Sandy - Candy = 30 :=
sorry

end pebble_difference_l148_148635


namespace Don_poured_milk_correct_amount_l148_148369

theorem Don_poured_milk_correct_amount :
  let original_milk : ℚ := 3 / 8
  let portion_poured : ℚ := 5 / 6
  let poured_milk : ℚ := portion_poured * original_milk
  poured_milk = 5 / 16 :=
by
  let original_milk : ℚ := 3 / 8
  let portion_poured : ℚ := 5 / 6
  let poured_milk : ℚ := portion_poured * original_milk
  show poured_milk = 5 / 16
  sorry

end Don_poured_milk_correct_amount_l148_148369


namespace no_solutions_eqn_in_interval_l148_148588

theorem no_solutions_eqn_in_interval :
  ∀ (x : ℝ), (π/4 ≤ x ∧ x ≤ π/2) →
  ¬ (sin (x ^ (Real.sin x)) = cos (x ^ (Real.cos x))) :=
by
  intros x hx
  sorry

end no_solutions_eqn_in_interval_l148_148588


namespace find_total_buffaloes_l148_148758

-- Define the problem parameters.
def number_of_ducks : ℕ := sorry
def number_of_cows : ℕ := 8

-- Define the conditions.
def duck_legs : ℕ := 2 * number_of_ducks
def cow_legs : ℕ := 4 * number_of_cows
def total_heads : ℕ := number_of_ducks + number_of_cows

-- The given equation as a condition.
def total_legs : ℕ := duck_legs + cow_legs

-- Translate condition from the problem:
def condition : Prop := total_legs = 2 * total_heads + 16

-- The proof statement.
theorem find_total_buffaloes : number_of_cows = 8 :=
by
  -- Place the placeholder proof here.
  sorry

end find_total_buffaloes_l148_148758


namespace easter_eggs_total_l148_148800

theorem easter_eggs_total (h he total : ℕ)
 (hannah_eggs : h = 42) 
 (twice_he : h = 2 * he) 
 (total_eggs : total = h + he) : 
 total = 63 := 
sorry

end easter_eggs_total_l148_148800


namespace speed_of_current_l148_148196

theorem speed_of_current (v_b v_c v_d : ℝ) (hd : v_d = 15) 
  (hvd1 : v_b + v_c = v_d) (hvd2 : v_b - v_c = 12) :
  v_c = 1.5 :=
by sorry

end speed_of_current_l148_148196


namespace correct_transformation_l148_148101

theorem correct_transformation (x y : ℤ) (h : x = y) : x - 2 = y - 2 :=
by
  sorry

end correct_transformation_l148_148101


namespace composite_function_properties_l148_148091

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem composite_function_properties
  (a b c : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_no_real_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) :=
by sorry

end composite_function_properties_l148_148091


namespace six_digit_numbers_with_zero_count_l148_148677

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l148_148677


namespace spinners_even_product_probability_l148_148158

def even_product_probability {A B : Type} [Fintype A] [Fintype B] (pA : A → ℕ) (pB : B → ℕ) (P : Finset A) (Q : Finset B) : ℚ :=
  (P.card * Q.card) / ((Fintype.card A) * (Fintype.card B))

theorem spinners_even_product_probability :
  let A := {1, 2, 3, 4, 5, 6}
  let B := {1, 2, 3, 4}
  let pA : Finset ℕ := {n : ℕ | n ∈ A ∧ n % 2 = 0}
  let pB : Finset ℕ := {n : ℕ | n ∈ B ∧ n % 2 = 0}
  let oddsA : Finset ℕ := {n : ℕ | n ∈ A ∧ ¬(n % 2 = 0)}
  let oddsB : Finset ℕ := {n : ℕ | n ∈ B ∧ ¬(n % 2 = 0)}
  even_product_probability (λ x => x) (λ y => y) pA B + even_product_probability (λ x => x) (λ y => y) A pB - even_product_probability (λ x => x) (λ y => y) pA pB = 1 / 2 := by
  sorry
 
end spinners_even_product_probability_l148_148158


namespace trapezoid_perimeter_l148_148474

theorem trapezoid_perimeter (height : ℝ) (radius : ℝ) (LM KN : ℝ) (LM_eq : LM = 16.5) (KN_eq : KN = 37.5)
  (LK MN : ℝ) (LK_eq : LK = 37.5) (MN_eq : MN = 37.5) (H : height = 36) (R : radius = 11) : 
  (LM + KN + LK + MN) = 129 :=
by
  -- The proof is omitted; only the statement is provided as specified.
  sorry

end trapezoid_perimeter_l148_148474


namespace prob_triangle_inequality_l148_148801

theorem prob_triangle_inequality (x y z : ℕ) (h1 : 1 ≤ x ∧ x ≤ 6) (h2 : 1 ≤ y ∧ y ≤ 6) (h3 : 1 ≤ z ∧ z ≤ 6) : 
  (∃ (p : ℚ), p = 37 / 72) := 
sorry

end prob_triangle_inequality_l148_148801


namespace find_a21_l148_148248

def seq_a (n : ℕ) : ℝ := sorry  -- This should define the sequence a_n
def seq_b (n : ℕ) : ℝ := sorry  -- This should define the sequence b_n

theorem find_a21 (h1 : seq_a 1 = 2)
  (h2 : ∀ n, seq_b n = seq_a (n + 1) / seq_a n)
  (h3 : ∀ n m, seq_b n = seq_b m * r^(n - m)) 
  (h4 : seq_b 10 * seq_b 11 = 2) :
  seq_a 21 = 2 ^ 11 :=
sorry

end find_a21_l148_148248


namespace unique_real_solution_l148_148933

theorem unique_real_solution : ∃ x : ℝ, (∀ t : ℝ, x^2 - t * x + 36 = 0 ∧ x^2 - 8 * x + t = 0) ∧ x = 3 :=
by
  sorry

end unique_real_solution_l148_148933


namespace neg_p_eq_exist_l148_148148

theorem neg_p_eq_exist:
  (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2 * a * b) ↔ ∃ a b : ℝ, a^2 + b^2 < 2 * a * b := by
  sorry

end neg_p_eq_exist_l148_148148


namespace men_employed_l148_148425

/- 
Lauryn owns a computer company that employs men and women in different positions in the company. 
How many men does he employ if there are 20 fewer men than women and 180 people working for Lauryn?
-/

/-- Proof that the number of men employed at Lauryn's company is 80 given the conditions -/
theorem men_employed (x : ℕ) : 
  (total_people : ℕ := 180) 
  (fewer_men : ℕ := 20) 
  (number_of_women := x + 20) 
  (number_of_men := x)
  (total_people = number_of_men + number_of_women) :=
begin
  sorry
end

end men_employed_l148_148425


namespace exists_same_color_rectangle_l148_148372

variable (coloring : ℕ × ℕ → Fin 3)

theorem exists_same_color_rectangle :
  (∃ (r1 r2 r3 r4 c1 c2 c3 c4 : ℕ), 
    r1 ≠ r2 ∧ r2 ≠ r3 ∧ r3 ≠ r4 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r4 ∧ 
    c1 ≠ c2 ∧ 
    coloring (4, 82) = 4 ∧ 
    coloring (r1, c1) = coloring (r1, c2) ∧ coloring (r1, c2) = coloring (r2, c1) ∧ 
    coloring (r2, c1) = coloring (r2, c2)) :=
sorry

end exists_same_color_rectangle_l148_148372


namespace find_f_at_3_l148_148445

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq : ∀ (x : ℝ), x ≠ 2 / 3 → f x + f ((x + 2) / (2 - 3 * x)) = 2 * x

theorem find_f_at_3 : f 3 = 3 :=
by {
  sorry
}

end find_f_at_3_l148_148445


namespace actual_distance_between_mountains_l148_148775

theorem actual_distance_between_mountains (D_map : ℝ) (d_map_ram : ℝ) (d_real_ram : ℝ)
  (hD_map : D_map = 312) (hd_map_ram : d_map_ram = 25) (hd_real_ram : d_real_ram = 10.897435897435898) :
  D_map / d_map_ram * d_real_ram = 136 :=
by
  -- Theorem statement is proven based on the given conditions.
  sorry

end actual_distance_between_mountains_l148_148775


namespace university_diploma_percentage_l148_148008

theorem university_diploma_percentage
  (A : ℝ) (B : ℝ) (C : ℝ)
  (hA : A = 0.40)
  (hB : B = 0.10)
  (hC : C = 0.15) :
  A - B + C * (1 - A) = 0.39 := 
sorry

end university_diploma_percentage_l148_148008


namespace distance_after_one_hour_l148_148309

-- Definitions representing the problem's conditions
def initial_distance : ℕ := 20
def speed_athos : ℕ := 4
def speed_aramis : ℕ := 5

-- The goal is to prove that the possible distances after one hour are among the specified values
theorem distance_after_one_hour :
  ∃ d : ℕ, d = 11 ∨ d = 29 ∨ d = 21 ∨ d = 19 :=
sorry -- proof not required as per the instructions

end distance_after_one_hour_l148_148309


namespace cost_per_mile_eq_l148_148097

theorem cost_per_mile_eq :
  ( ∀ x : ℝ, (65 + 0.40 * 325 = x * 325) → x = 0.60 ) :=
by
  intros x h
  have eq1 : 65 + 0.40 * 325 = 195 := by sorry
  rw [eq1] at h
  have eq2 : 195 = 325 * x := h
  field_simp at eq2
  exact eq2

end cost_per_mile_eq_l148_148097


namespace find_value_2_plus_a4_plus_9_l148_148512

def arithmetic_sequence_sum (a1 an : ℚ) (n : ℕ) : ℚ :=
  (a1 + an) * n / 2

noncomputable def arithmetic_sum_nineth (a1 an : ℚ) : Prop :=
  arithmetic_sequence_sum a1 an 9 = 54

def arithmetic_sequence_nth_term (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a1 + d * (n - 1)

noncomputable def arithmetic_sequence_fourth_term (a1 d : ℚ) : ℚ :=
  arithmetic_sequence_nth_term a1 d 4

theorem find_value_2_plus_a4_plus_9 :
  (∃ a1 an d : ℚ, 
     arithmetic_sum_nineth a1 an ∧ 
     an = a1 + 8 * d ∧
     ∀ (n : ℕ), (n = 4 → (a1, d)) = (a1 + 3 * d)) → (2 + arithmetic_sequence_fourth_term a1 d + 9 = 307 / 27)
  := sorry

end find_value_2_plus_a4_plus_9_l148_148512


namespace find_x_l148_148530

theorem find_x (x : ℝ) (h : x - 2 * x + 3 * x = 100) : x = 50 := by
  sorry

end find_x_l148_148530


namespace find_matrix_N_l148_148506

-- Define the given matrix equation
def condition (N : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  N ^ 3 - 3 * N ^ 2 + 4 * N = ![![8, 16], ![4, 8]]

-- State the theorem
theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℝ) (h : condition N) :
  N = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_N_l148_148506


namespace sum_first_10_terms_arithmetic_seq_l148_148125

theorem sum_first_10_terms_arithmetic_seq :
  ∀ (a : ℕ → ℤ), (a 1 = -2) → (a 2 + a 6 = 2) →
  (∃ S₁₀, S₁₀ = 10 * a 1 + (10 * (10 - 1)) / 2 * (a 2 - a 1) / (2 - 1) ∧ S₁₀ = 25) :=
begin
  assume a,
  assume h1 : a 1 = -2,
  assume h2 : a 2 + a 6 = 2,
  sorry
end

end sum_first_10_terms_arithmetic_seq_l148_148125


namespace equivalent_expression_l148_148904

theorem equivalent_expression (x : ℝ) : 
  (x-1)^4 + 4*(x-1)^3 + 6*(x-1)^2 + 4*(x-1) + 1 = x^4 := 
by
  sorry

end equivalent_expression_l148_148904


namespace minimum_value_of_f_l148_148069

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |3 - x|

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 1 ∧ (∃ x₀ : ℝ, f x₀ = 1) := by
  sorry

end minimum_value_of_f_l148_148069


namespace decreasing_interval_b_l148_148871

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := - (1 / 2) * x ^ 2 + b * Real.log x

theorem decreasing_interval_b (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici (Real.sqrt 2) → ∀ x1 x2 : ℝ, x1 ∈ Set.Ici (Real.sqrt 2) → x2 ∈ Set.Ici (Real.sqrt 2) → 
   x1 ≤ x2 → f x1 b ≥ f x2 b) ↔ b ≤ 2 :=
by
  sorry

end decreasing_interval_b_l148_148871


namespace original_number_q_l148_148947

variables (q : ℝ) (a b c : ℝ)
 
theorem original_number_q : 
  (a = 1.125 * q) → (b = 0.75 * q) → (c = 30) → (a - b = c) → q = 80 :=
by
  sorry

end original_number_q_l148_148947


namespace project_completion_time_l148_148818

theorem project_completion_time (x : ℕ) :
  (∀ (B_days : ℕ), B_days = 40 →
  (∀ (combined_work_days : ℕ), combined_work_days = 10 →
  (∀ (total_days : ℕ), total_days = 20 →
  10 * (1 / (x : ℚ) + 1 / 40) + 10 * (1 / 40) = 1))) →
  x = 20 :=
by
  sorry

end project_completion_time_l148_148818


namespace range_of_2a_plus_3b_l148_148528

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3) (h3 : 2 < a - b) (h4 : a - b < 4) :
  -9 / 2 < 2 * a + 3 * b ∧ 2 * a + 3 * b < 13 / 2 :=
sorry

end range_of_2a_plus_3b_l148_148528


namespace stepa_multiplied_numbers_l148_148559

theorem stepa_multiplied_numbers (x : ℤ) (hx : (81 * x) % 16 = 0) :
  ∃ (a b : ℕ), a * b = 54 ∧ a < 10 ∧ b < 10 :=
by {
  sorry
}

end stepa_multiplied_numbers_l148_148559


namespace find_numbers_with_sum_and_product_l148_148576

theorem find_numbers_with_sum_and_product (S P : ℝ) :
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  (x1 + y1 = S ∧ x1 * y1 = P) ∨ (x2 + y2 = S ∧ x2 * y2 = P) :=
sorry

end find_numbers_with_sum_and_product_l148_148576


namespace number_of_fours_is_even_l148_148358

theorem number_of_fours_is_even (n3 n4 n5 : ℕ) 
  (h1 : n3 + n4 + n5 = 80)
  (h2 : 3 * n3 + 4 * n4 + 5 * n5 = 276) : Even n4 := 
sorry

end number_of_fours_is_even_l148_148358


namespace fraction_identity_l148_148510

theorem fraction_identity (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1009) : (a + b) / (a - b) = -1009 :=
by
  sorry

end fraction_identity_l148_148510


namespace factorize_difference_of_squares_l148_148044

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l148_148044


namespace six_digit_numbers_with_zero_l148_148671

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l148_148671


namespace herd_total_cows_l148_148827

theorem herd_total_cows (n : ℕ) : 
  let first_son := 1 / 3 * n
  let second_son := 1 / 6 * n
  let third_son := 1 / 8 * n
  let remaining := n - (first_son + second_son + third_son)
  remaining = 9 ↔ n = 24 := 
by
  -- Skipping proof, placeholder
  sorry

end herd_total_cows_l148_148827


namespace files_missing_is_15_l148_148499

def total_files : ℕ := 60
def morning_files : ℕ := total_files / 2
def afternoon_files : ℕ := 15
def organized_files : ℕ := morning_files + afternoon_files
def missing_files : ℕ := total_files - organized_files

theorem files_missing_is_15 : missing_files = 15 :=
  sorry

end files_missing_is_15_l148_148499


namespace sum_due_is_correct_l148_148813

-- Define constants for Banker's Discount and True Discount
def BD : ℝ := 288
def TD : ℝ := 240

-- Define Banker's Gain as the difference between BD and TD
def BG : ℝ := BD - TD

-- Define the sum due (S.D.) as the face value including True Discount and Banker's Gain
def SD : ℝ := TD + BG

-- Create a theorem to prove the sum due is Rs. 288
theorem sum_due_is_correct : SD = 288 :=
by
  -- Skipping proof with sorry; expect this statement to be true based on given conditions 
  sorry

end sum_due_is_correct_l148_148813


namespace no_four_distinct_nat_dividing_pairs_l148_148764

theorem no_four_distinct_nat_dividing_pairs (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : a ∣ (b - c)) (h8 : a ∣ (b - d))
  (h9 : a ∣ (c - d)) (h10 : b ∣ (a - c)) (h11 : b ∣ (a - d)) (h12 : b ∣ (c - d))
  (h13 : c ∣ (a - b)) (h14 : c ∣ (a - d)) (h15 : c ∣ (b - d)) (h16 : d ∣ (a - b))
  (h17 : d ∣ (a - c)) (h18 : d ∣ (b - c)) : False := 
sorry

end no_four_distinct_nat_dividing_pairs_l148_148764


namespace volume_range_l148_148653

theorem volume_range (a b c : ℝ) (h1 : a + b + c = 9)
  (h2 : a * b + b * c + a * c = 24) : 16 ≤ a * b * c ∧ a * b * c ≤ 20 :=
by {
  -- Proof would go here
  sorry
}

end volume_range_l148_148653


namespace clock_chime_time_l148_148444

theorem clock_chime_time (t_5oclock : ℕ) (n_5chimes : ℕ) (t_10oclock : ℕ) (n_10chimes : ℕ)
  (h1: t_5oclock = 8) (h2: n_5chimes = 5) (h3: n_10chimes = 10) : 
  t_10oclock = 18 :=
by
  sorry

end clock_chime_time_l148_148444


namespace greatest_percentage_increase_l148_148265

def pop1970_F := 30000
def pop1980_F := 45000
def pop1970_G := 60000
def pop1980_G := 75000
def pop1970_H := 40000
def pop1970_I := 20000
def pop1980_combined_H := 70000
def pop1970_J := 90000
def pop1980_J := 120000

def percentage_increase (pop1970 pop1980 : ℕ) : ℚ :=
  ((pop1980 - pop1970 : ℚ) / pop1970) * 100

theorem greatest_percentage_increase :
  ∀ (city : ℕ), (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase pop1970_G pop1980_G) ∧
                (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase (pop1970_H + pop1970_I) pop1980_combined_H) ∧
                (city = pop1970_F -> percentage_increase pop1970_F pop1980_F >= percentage_increase pop1970_J pop1980_J) := by 
  sorry

end greatest_percentage_increase_l148_148265


namespace molecular_weight_of_3_moles_of_Fe2_SO4_3_l148_148845

noncomputable def mol_weight_fe : ℝ := 55.845
noncomputable def mol_weight_s : ℝ := 32.065
noncomputable def mol_weight_o : ℝ := 15.999

noncomputable def mol_weight_fe2_so4_3 : ℝ :=
  (2 * mol_weight_fe) + (3 * (mol_weight_s + (4 * mol_weight_o)))

theorem molecular_weight_of_3_moles_of_Fe2_SO4_3 :
  3 * mol_weight_fe2_so4_3 = 1199.619 := by
  sorry

end molecular_weight_of_3_moles_of_Fe2_SO4_3_l148_148845


namespace spent_on_books_l148_148254

theorem spent_on_books (allowance games_fraction snacks_fraction toys_fraction : ℝ)
  (h_allowance : allowance = 50)
  (h_games : games_fraction = 1/4)
  (h_snacks : snacks_fraction = 1/5)
  (h_toys : toys_fraction = 2/5) :
  allowance - (allowance * games_fraction + allowance * snacks_fraction + allowance * toys_fraction) = 7.5 :=
by
  sorry

end spent_on_books_l148_148254


namespace smallest_circle_equation_l148_148922

-- Definitions of the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- The statement of the problem
theorem smallest_circle_equation : ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ 
  A.1 = -3 ∧ A.2 = 0 ∧ B.1 = 3 ∧ B.2 = 0 ∧ ((x - 0)^2 + (y - 0)^2 = 9) :=
by
  sorry

end smallest_circle_equation_l148_148922


namespace negation_equiv_l148_148260

theorem negation_equiv (p : Prop) : 
  (p = (∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0)) → 
  (¬ p = (∀ x : ℝ, x^2 + 2*x + 2 > 0)) :=
by
  sorry

end negation_equiv_l148_148260


namespace square_grid_21_max_moves_rectangle_grid_20_21_max_moves_l148_148191

-- Define the problem conditions.
def square_grid (n : Nat) : Prop := true
def rectangle_grid (m n : Nat) : Prop := true

-- Define the grid size for square and rectangle.
def square_grid_21 := square_grid 21
def rectangle_grid_20_21 := rectangle_grid 20 21

-- Define the proof problem to find maximum moves.
theorem square_grid_21_max_moves : ∃ m : Nat, m = 3 :=
  sorry

theorem rectangle_grid_20_21_max_moves : ∃ m : Nat, m = 4 :=
  sorry

end square_grid_21_max_moves_rectangle_grid_20_21_max_moves_l148_148191


namespace sanghyeon_questions_l148_148168

variable (S : ℕ)

theorem sanghyeon_questions (h1 : S + (S + 5) = 43) : S = 19 :=
by
    sorry

end sanghyeon_questions_l148_148168


namespace factorize_difference_of_squares_l148_148061

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l148_148061


namespace yellow_balls_in_bag_l148_148949

theorem yellow_balls_in_bag (y : ℕ) (r : ℕ) (P_red : ℚ) (h_r : r = 8) (h_P_red : P_red = 1 / 3) 
  (h_prob : P_red = r / (r + y)) : y = 16 :=
by
  sorry

end yellow_balls_in_bag_l148_148949


namespace max_5x_min_25x_l148_148388

theorem max_5x_min_25x : ∃ x : ℝ, 5^x - 25^x = 1/4 :=
by
  sorry

end max_5x_min_25x_l148_148388


namespace zack_traveled_countries_l148_148464

theorem zack_traveled_countries 
  (a : ℕ) (g : ℕ) (j : ℕ) (p : ℕ) (z : ℕ)
  (ha : a = 30)
  (hg : g = (3 / 5) * a)
  (hj : j = (1 / 3) * g)
  (hp : p = (4 / 3) * j)
  (hz : z = (5 / 2) * p) :
  z = 20 := 
sorry

end zack_traveled_countries_l148_148464


namespace danny_more_caps_l148_148362

variable (found thrown_away : ℕ)

def bottle_caps_difference (found thrown_away : ℕ) : ℕ :=
  found - thrown_away

theorem danny_more_caps
  (h_found : found = 36)
  (h_thrown_away : thrown_away = 35) :
  bottle_caps_difference found thrown_away = 1 :=
by
  -- Proof is omitted with sorry
  sorry

end danny_more_caps_l148_148362


namespace number_of_boys_is_320_l148_148591

-- Definition of the problem's conditions
variable (B G : ℕ)
axiom condition1 : B + G = 400
axiom condition2 : G = (B / 400) * 100

-- Stating the theorem to prove number of boys is 320
theorem number_of_boys_is_320 : B = 320 :=
by
  sorry

end number_of_boys_is_320_l148_148591


namespace wendy_boxes_l148_148602

theorem wendy_boxes (x : ℕ) (w_brother : ℕ) (total : ℕ) (candy_per_box : ℕ) 
    (h_w_brother : w_brother = 6) 
    (h_candy_per_box : candy_per_box = 3) 
    (h_total : total = 12) 
    (h_equation : 3 * x + w_brother = total) : 
    x = 2 :=
by
  -- Proof would go here
  sorry

end wendy_boxes_l148_148602


namespace remainder_8_pow_1996_mod_5_l148_148806

theorem remainder_8_pow_1996_mod_5 :
  (8: ℕ) ≡ 3 [MOD 5] →
  3^4 ≡ 1 [MOD 5] →
  8^1996 ≡ 1 [MOD 5] :=
by
  sorry

end remainder_8_pow_1996_mod_5_l148_148806


namespace alexa_weight_proof_l148_148211

variable (totalWeight katerinaWeight alexaWeight : ℕ)

def weight_relation (totalWeight katerinaWeight alexaWeight : ℕ) : Prop :=
  totalWeight = katerinaWeight + alexaWeight

theorem alexa_weight_proof (h1 : totalWeight = 95) (h2 : katerinaWeight = 49) : alexaWeight = 46 :=
by
  have h : alexaWeight = totalWeight - katerinaWeight := by
    sorry
  rw [h1, h2] at h
  exact h

end alexa_weight_proof_l148_148211


namespace both_teams_joint_renovation_team_renovation_split_l148_148814

-- Problem setup for part 1
def renovation_total_length : ℕ := 2400
def teamA_daily_progress : ℕ := 30
def teamB_daily_progress : ℕ := 50
def combined_days_to_complete_renovation : ℕ := 30

theorem both_teams_joint_renovation (x : ℕ) :
  (teamA_daily_progress + teamB_daily_progress) * x = renovation_total_length → 
  x = combined_days_to_complete_renovation :=
by
  sorry

-- Problem setup for part 2
def total_renovation_days : ℕ := 60
def length_renovated_by_teamA : ℕ := 900
def length_renovated_by_teamB : ℕ := 1500

theorem team_renovation_split (a b : ℕ) :
  a / teamA_daily_progress + b / teamB_daily_progress = total_renovation_days ∧ 
  a + b = renovation_total_length → 
  a = length_renovated_by_teamA ∧ b = length_renovated_by_teamB :=
by
  sorry

end both_teams_joint_renovation_team_renovation_split_l148_148814


namespace eugene_payment_correct_l148_148269

theorem eugene_payment_correct :
  let t_price := 20
  let p_price := 80
  let s_price := 150
  let discount_rate := 0.1
  let t_quantity := 4
  let p_quantity := 3
  let s_quantity := 2
  let t_cost := t_quantity * t_price
  let p_cost := p_quantity * p_price
  let s_cost := s_quantity * s_price
  let total_cost := t_cost + p_cost + s_cost
  let discount := discount_rate * total_cost
  let final_cost := total_cost - discount
  final_cost = 558 :=
by
  sorry

end eugene_payment_correct_l148_148269


namespace magnitude_sum_unit_vectors_l148_148514

open Real

variables {V : Type*} [inner_product_space ℝ V]

theorem magnitude_sum_unit_vectors {a b : V} (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) 
(h_angle : real_angle a b = real.pi / 3) :
  ∥a + 3 • b∥ = sqrt 13 := 
by 
  sorry

end magnitude_sum_unit_vectors_l148_148514


namespace sum_of_squares_of_b_l148_148318

-- Define the constants
def b1 := 35 / 64
def b2 := 0
def b3 := 21 / 64
def b4 := 0
def b5 := 7 / 64
def b6 := 0
def b7 := 1 / 64

-- The goal is to prove the sum of squares of these constants
theorem sum_of_squares_of_b : 
  (b1 ^ 2 + b2 ^ 2 + b3 ^ 2 + b4 ^ 2 + b5 ^ 2 + b6 ^ 2 + b7 ^ 2) = 429 / 1024 :=
  by
    -- defer the proof
    sorry

end sum_of_squares_of_b_l148_148318


namespace arithmetic_sequence_lemma_l148_148241

theorem arithmetic_sequence_lemma (a : ℕ → ℝ) (h_arith_seq : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0)
  (h_condition : a 3 + a 11 = 22) : a 7 = 11 :=
sorry

end arithmetic_sequence_lemma_l148_148241


namespace measure_diagonal_of_brick_l148_148292

def RectangularParallelepiped (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def DiagonalMeasurementPossible (a b c : ℝ) : Prop :=
  ∃ d : ℝ, d = (a^2 + b^2 + c^2)^(1/2)

theorem measure_diagonal_of_brick (a b c : ℝ) 
  (h : RectangularParallelepiped a b c) : DiagonalMeasurementPossible a b c :=
by
  sorry

end measure_diagonal_of_brick_l148_148292


namespace time_spent_giving_bath_l148_148773

theorem time_spent_giving_bath
  (total_time : ℕ)
  (walk_time : ℕ)
  (bath_time blowdry_time : ℕ)
  (walk_distance walk_speed : ℤ)
  (walk_distance_eq : walk_distance = 3)
  (walk_speed_eq : walk_speed = 6)
  (total_time_eq : total_time = 60)
  (walk_time_eq : walk_time = (walk_distance * 60 / walk_speed))
  (half_blowdry_time : blowdry_time = bath_time / 2)
  (time_eq : bath_time + blowdry_time = total_time - walk_time)
  : bath_time = 20 := by
  sorry

end time_spent_giving_bath_l148_148773


namespace original_number_is_400_l148_148957

theorem original_number_is_400 (x : ℝ) (h : 1.20 * x = 480) : x = 400 :=
sorry

end original_number_is_400_l148_148957


namespace age_difference_is_20_l148_148785

-- Definitions for the ages of the two persons
def elder_age := 35
def younger_age := 15

-- Condition: Difference in ages
def age_difference := elder_age - younger_age

-- Theorem to prove the difference in ages is 20 years
theorem age_difference_is_20 : age_difference = 20 := by
  sorry

end age_difference_is_20_l148_148785


namespace minimum_button_presses_l148_148328

theorem minimum_button_presses :
  ∃ (r y g : ℕ), 
    2 * y - r = 3 ∧ 2 * g - y = 3 ∧ r + y + g = 9 :=
by sorry

end minimum_button_presses_l148_148328


namespace isosceles_triangle_base_length_l148_148611

theorem isosceles_triangle_base_length :
  ∀ (p_equilateral p_isosceles side_equilateral : ℕ), 
  p_equilateral = 60 → 
  side_equilateral = p_equilateral / 3 →
  p_isosceles = 55 →
  ∀ (base_isosceles : ℕ),
  side_equilateral + side_equilateral + base_isosceles = p_isosceles →
  base_isosceles = 15 :=
by
  intros p_equilateral p_isosceles side_equilateral h1 h2 h3 base_isosceles h4
  sorry

end isosceles_triangle_base_length_l148_148611


namespace trapezoid_smallest_angle_l148_148205

theorem trapezoid_smallest_angle (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : 2 * a + 3 * d = 180) : 
  a = 20 :=
by
  sorry

end trapezoid_smallest_angle_l148_148205


namespace vasya_max_earning_l148_148173

theorem vasya_max_earning (k : ℕ) (h₀: k ≤ 2013) (h₁: 2013 - 2*k % 11 = 0) : k % 11 = 0 → (k ≤ 5) := 
by
  sorry

end vasya_max_earning_l148_148173


namespace six_digit_numbers_with_zero_l148_148684

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148684


namespace athletes_camp_duration_l148_148954

theorem athletes_camp_duration
  (h : ℕ)
  (initial_athletes : ℕ := 300)
  (rate_leaving : ℕ := 28)
  (rate_entering : ℕ := 15)
  (hours_entering : ℕ := 7)
  (difference : ℕ := 7) :
  300 - 28 * h + 15 * 7 = 300 + 7 → h = 4 :=
by
  sorry

end athletes_camp_duration_l148_148954


namespace sum_infinite_series_l148_148849

theorem sum_infinite_series :
  (∑' n : ℕ, (4 * (n + 1) + 1) / (3^(n + 1))) = 7 / 2 :=
sorry

end sum_infinite_series_l148_148849


namespace min_value_ab_min_value_a_plus_2b_l148_148401
open Nat

theorem min_value_ab (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_cond : a * b = 2 * a + b) : 8 ≤ a * b :=
by
  sorry

theorem min_value_a_plus_2b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_cond : a * b = 2 * a + b) : 9 ≤ a + 2 * b :=
by
  sorry

end min_value_ab_min_value_a_plus_2b_l148_148401


namespace solve_triangle_l148_148900

theorem solve_triangle :
  (a = 6 ∧ b = 6 * Real.sqrt 3 ∧ A = 30) →
  ((B = 60 ∧ C = 90 ∧ c = 12) ∨ (B = 120 ∧ C = 30 ∧ c = 6)) :=
by
  intros h
  sorry

end solve_triangle_l148_148900


namespace cosh_le_exp_sqr_l148_148072

open Real

theorem cosh_le_exp_sqr {x k : ℝ} : (∀ x : ℝ, cosh x ≤ exp (k * x^2)) ↔ k ≥ 1/2 :=
sorry

end cosh_le_exp_sqr_l148_148072


namespace girls_in_school_l148_148314

theorem girls_in_school (boys girls : ℕ) (ratio : ℕ → ℕ → Prop) (h1 : ratio 5 4) (h2 : boys = 1500) :
    girls = 1200 :=
by
  sorry

end girls_in_school_l148_148314


namespace sum_of_a_and_b_l148_148998

theorem sum_of_a_and_b (a b : ℝ) (h_neq : a ≠ b) (h_a : a * (a - 4) = 21) (h_b : b * (b - 4) = 21) :
  a + b = 4 :=
by
  sorry

end sum_of_a_and_b_l148_148998


namespace div_operation_example_l148_148177

theorem div_operation_example : ((180 / 6) / 3) = 10 := by
  sorry

end div_operation_example_l148_148177


namespace six_digit_numbers_with_zero_l148_148732

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148732


namespace logical_equivalence_l148_148006

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ ¬R) → ¬Q) ↔ (Q → (¬P ∨ R)) :=
by
  sorry

end logical_equivalence_l148_148006


namespace locus_of_projection_l148_148908

theorem locus_of_projection {a b c : ℝ} (h : (1 / a ^ 2) + (1 / b ^ 2) = 1 / c ^ 2) :
  ∀ (x y : ℝ), (x, y) ∈ ({P : ℝ × ℝ | ∃ a b : ℝ, P = ((a * b^2) / (a^2 + b^2), (a^2 * b) / (a^2 + b^2)) ∧ (1 / a ^ 2) + (1 / b ^ 2) = 1 / c ^ 2}) → 
    x^2 + y^2 = c^2 := 
sorry

end locus_of_projection_l148_148908


namespace circle_center_radius_l148_148655

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 4 * x + 2 * y - 4 = 0 ↔ (x - 2)^2 + (y + 1)^2 = 3 :=
by
  sorry

end circle_center_radius_l148_148655


namespace factorize_x_squared_minus_one_l148_148058

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l148_148058


namespace sin_arcsin_plus_arctan_l148_148980

theorem sin_arcsin_plus_arctan :
  sin (Real.arcsin (4 / 5) + Real.arctan 3) = (13 * Real.sqrt 10) / 50 :=
by
  sorry

end sin_arcsin_plus_arctan_l148_148980


namespace number_of_auspicious_three_digit_numbers_l148_148926

-- Definition: sum of the digits of a number n
def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- Definition: "auspicious number"
def is_auspicious (n : ℕ) : Prop :=
  n % 6 = 0 ∧ sum_of_digits n = 6 ∧ 100 ≤ n ∧ n < 1000

-- Proof statement: there are exactly 12 auspicious numbers between 100 and 999
theorem number_of_auspicious_three_digit_numbers : 
  Finset.card (Finset.filter is_auspicious (Finset.range 1000 \ Finset.range 100)) = 12 := 
by
  sorry

end number_of_auspicious_three_digit_numbers_l148_148926


namespace find_B_squared_l148_148064

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 31 + 85 / x

theorem find_B_squared :
  let x1 := (Real.sqrt 31 + Real.sqrt 371) / 2
  let x2 := (Real.sqrt 31 - Real.sqrt 371) / 2
  let B := |x1| + |x2|
  B^2 = 371 :=
by
  sorry

end find_B_squared_l148_148064


namespace six_digit_numbers_with_zero_l148_148687

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148687


namespace parabola_focus_l148_148365

theorem parabola_focus (a b c : ℝ) (h k : ℝ) (p : ℝ) :
  (a = 4) →
  (b = -4) →
  (c = -3) →
  (h = -b / (2 * a)) →
  (k = a * h ^ 2 + b * h + c) →
  (p = 1 / (4 * a)) →
  (k + p = -4 + 1 / 16) →
  (h, k + p) = (1 / 2, -63 / 16) :=
by
  intros a_eq b_eq c_eq h_eq k_eq p_eq focus_eq
  rw [a_eq, b_eq, c_eq] at *
  sorry

end parabola_focus_l148_148365


namespace range_of_a_l148_148252

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a * x + 1 > 0) ↔ (-2 < a ∧ a < 2) :=
sorry

end range_of_a_l148_148252


namespace part1_part2_part3_l148_148175

variable (prob_A : ℚ) (prob_B : ℚ)
variables (prob_miss_A : ℚ) (prob_miss_B : ℚ)

theorem part1 :
  prob_A = 2 / 3 →
  prob_B = 3 / 4 →
  prob_A * prob_B = 1 / 2 :=
by
  intros h1 h2
  sorry

theorem part2 :
  prob_A = 2 / 3 →
  prob_miss_A = 1 / 3 →
  (prob_A ^ 3 * prob_miss_A + prob_miss_A * prob_A ^ 3) = 16 / 81 :=
by
  intros h1 h2
  sorry

theorem part3 :
  prob_B = 3 / 4 →
  prob_miss_B = 1 / 4 →
  (prob_B ^ 2 * prob_miss_B ^ 2 + prob_miss_B * prob_B * prob_miss_B ^ 2) = 3 / 64 :=
by
  intros h1 h2
  sorry

end part1_part2_part3_l148_148175


namespace find_y_l148_148752

theorem find_y (x y : ℝ) (h : x = 180) (h1 : 0.25 * x = 0.10 * y - 5) : y = 500 :=
by sorry

end find_y_l148_148752


namespace part1_part2_l148_148220

variable (a b : ℝ)

theorem part1 : ((-a)^2 * (a^2)^2 / a^3) = a^3 := sorry

theorem part2 : (a + b) * (a - b) - (a - b)^2 = 2 * a * b - 2 * b^2 := sorry

end part1_part2_l148_148220


namespace check_conditions_l148_148095

noncomputable def f (x a b : ℝ) : ℝ := |x^2 - 2 * a * x + b|

theorem check_conditions (a b : ℝ) :
  ¬ (∀ x : ℝ, f x a b = f (-x) a b) ∧         -- f(x) is not necessarily an even function
  ¬ (∀ x : ℝ, (f 0 a b = f 2 a b → (f x a b = f (2 - x) a b))) ∧ -- No guaranteed symmetry about x=1
  (a^2 - b^2 ≤ 0 → ∀ x : ℝ, x ≥ a → ∀ y : ℝ, y ≥ x → f y a b ≥ f x a b) ∧ -- f(x) is increasing on [a, +∞) if a^2 - b^2 ≤ 0
  ¬ (∀ x : ℝ, f x a b ≤ |a^2 - b|)         -- f(x) does not necessarily have a max value of |a^2 - b|
:= sorry

end check_conditions_l148_148095


namespace dress_hem_length_in_feet_l148_148277

def stitch_length_in_inches : ℚ := 1 / 4
def stitches_per_minute : ℕ := 24
def time_in_minutes : ℕ := 6

theorem dress_hem_length_in_feet :
  (stitch_length_in_inches * (stitches_per_minute * time_in_minutes)) / 12 = 3 :=
by
  sorry

end dress_hem_length_in_feet_l148_148277


namespace arithmetic_sequence_common_difference_l148_148759

theorem arithmetic_sequence_common_difference 
    (a : ℤ) (last_term : ℤ) (sum_terms : ℤ) (n : ℕ)
    (h1 : a = 3) 
    (h2 : last_term = 58) 
    (h3 : sum_terms = 488)
    (h4 : sum_terms = n * (a + last_term) / 2)
    (h5 : last_term = a + (n - 1) * d) :
    d = 11 / 3 := by
  sorry

end arithmetic_sequence_common_difference_l148_148759


namespace fractions_equal_l148_148745

theorem fractions_equal (x y z : ℝ) (hx1 : x ≠ 1) (hy1 : y ≠ 1) (hxy : x ≠ y)
  (h : (yz - x^2) / (1 - x) = (xz - y^2) / (1 - y)) : (yz - x^2) / (1 - x) = x + y + z ∧ (xz - y^2) / (1 - y) = x + y + z :=
sorry

end fractions_equal_l148_148745


namespace good_numbers_product_sum_digits_not_equal_l148_148287

def is_good_number (n : ℕ) : Prop :=
  n.digits 10 ⊆ [0, 1]

theorem good_numbers_product_sum_digits_not_equal (A B : ℕ) (hA : is_good_number A) (hB : is_good_number B) (hAB : is_good_number (A * B)) :
  ¬ ( (A.digits 10).sum * (B.digits 10).sum = ((A * B).digits 10).sum ) :=
sorry

end good_numbers_product_sum_digits_not_equal_l148_148287


namespace Cathy_and_Chris_worked_months_l148_148636

theorem Cathy_and_Chris_worked_months (Cathy_hours : ℕ) (weekly_hours : ℕ) (weeks_in_month : ℕ) (extra_weekly_hours : ℕ) (weeks_for_Chris_sick : ℕ) : 
  Cathy_hours = 180 →
  weekly_hours = 20 →
  weeks_in_month = 4 →
  extra_weekly_hours = weekly_hours →
  weeks_for_Chris_sick = 1 →
  (Cathy_hours - extra_weekly_hours * weeks_for_Chris_sick) / weekly_hours / weeks_in_month = (2 : ℕ) :=
by
  intros hCathy_hours hweekly_hours hweeks_in_month hextra_weekly_hours hweeks_for_Chris_sick
  rw [hCathy_hours, hweekly_hours, hweeks_in_month, hextra_weekly_hours, hweeks_for_Chris_sick]
  norm_num
  sorry

end Cathy_and_Chris_worked_months_l148_148636


namespace correct_sqrt_evaluation_l148_148183

theorem correct_sqrt_evaluation:
  2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2 :=
by 
  sorry

end correct_sqrt_evaluation_l148_148183


namespace multiple_of_large_block_length_l148_148473

-- Define the dimensions and volumes
variables (w d l : ℝ) -- Normal block dimensions
variables (V_normal V_large : ℝ) -- Volumes
variables (m : ℝ) -- Multiple for the length of the large block

-- Volume conditions for normal and large blocks
def normal_volume_condition (w d l : ℝ) (V_normal : ℝ) : Prop :=
  V_normal = w * d * l

def large_volume_condition (w d l m V_large : ℝ) : Prop :=
  V_large = (2 * w) * (2 * d) * (m * l)

-- Given problem conditions
axiom V_normal_eq_3 : normal_volume_condition w d l 3
axiom V_large_eq_36 : large_volume_condition w d l m 36

-- Statement we want to prove
theorem multiple_of_large_block_length : m = 3 :=
by
  -- Proof steps would go here
  sorry

end multiple_of_large_block_length_l148_148473


namespace room_breadth_l148_148009

theorem room_breadth :
  ∀ (length breadth carpet_width cost_per_meter total_cost : ℝ),
  length = 15 →
  carpet_width = 75 / 100 →
  cost_per_meter = 30 / 100 →
  total_cost = 36 →
  total_cost = cost_per_meter * (total_cost / cost_per_meter) →
  length * breadth = (total_cost / cost_per_meter) * carpet_width →
  breadth = 6 :=
by
  intros length breadth carpet_width cost_per_meter total_cost
  intros h_length h_carpet_width h_cost_per_meter h_total_cost h_total_cost_eq h_area_eq
  sorry

end room_breadth_l148_148009


namespace difference_two_smallest_integers_l148_148319

/--
There is more than one integer greater than 1 which, when divided by any integer k such that 2 ≤ k ≤ 11, has a remainder of 1.
Prove that the difference between the two smallest such integers is 27720.
-/
theorem difference_two_smallest_integers :
  ∃ n₁ n₂ : ℤ, 
  (∀ k : ℤ, 2 ≤ k ∧ k ≤ 11 → (n₁ % k = 1 ∧ n₂ % k = 1)) ∧ 
  n₁ > 1 ∧ n₂ > 1 ∧ 
  ∀ m : ℤ, (∀ k : ℤ, 2 ≤ k ∧ k ≤ 11 → (m % k =  1)) ∧ m > 1 → m = n₁ ∨ m = n₂ → 
  (n₂ - n₁ = 27720) := 
sorry

end difference_two_smallest_integers_l148_148319


namespace vicente_meat_purchase_l148_148601

theorem vicente_meat_purchase :
  ∃ (meat_lbs : ℕ),
  (∃ (rice_kgs cost_rice_per_kg cost_meat_per_lb total_spent : ℕ),
    rice_kgs = 5 ∧
    cost_rice_per_kg = 2 ∧
    cost_meat_per_lb = 5 ∧
    total_spent = 25 ∧
    total_spent - (rice_kgs * cost_rice_per_kg) = meat_lbs * cost_meat_per_lb) ∧
  meat_lbs = 3 :=
by {
  sorry
}

end vicente_meat_purchase_l148_148601


namespace solve_inequality_x_squared_minus_6x_gt_15_l148_148368

theorem solve_inequality_x_squared_minus_6x_gt_15 :
  { x : ℝ | x^2 - 6 * x > 15 } = { x : ℝ | x < -1.5 } ∪ { x : ℝ | x > 7.5 } :=
by
  sorry

end solve_inequality_x_squared_minus_6x_gt_15_l148_148368


namespace repeating_decimal_product_l148_148495

theorem repeating_decimal_product :
  let s := 0.\overline{456} in 
  s * 8 = 1216 / 333 :=
by
  sorry

end repeating_decimal_product_l148_148495


namespace tammy_investment_change_l148_148537

-- Defining initial investment, losses, and gains
def initial_investment : ℝ := 100
def first_year_loss : ℝ := 0.10
def second_year_gain : ℝ := 0.25

-- Defining the final amount after two years
def final_amount (initial_investment : ℝ) (first_year_loss : ℝ) (second_year_gain : ℝ) : ℝ :=
  let remaining_after_first_year := initial_investment * (1 - first_year_loss)
  remaining_after_first_year * (1 + second_year_gain)

-- Statement to prove
theorem tammy_investment_change :
  let percentage_change := ((final_amount initial_investment first_year_loss second_year_gain - initial_investment) / initial_investment) * 100
  percentage_change = 12.5 :=
by
  sorry

end tammy_investment_change_l148_148537


namespace final_score_is_80_l148_148471

def adam_final_score : ℕ :=
  let first_half := 8
  let second_half := 2
  let points_per_question := 8
  (first_half + second_half) * points_per_question

theorem final_score_is_80 : adam_final_score = 80 := by
  sorry

end final_score_is_80_l148_148471


namespace solve_inequality_l148_148656

def f (x : ℝ) : ℝ := |x + 1| - |x - 3|

theorem solve_inequality : ∀ x : ℝ, |f x| ≤ 4 :=
by
  intro x
  sorry

end solve_inequality_l148_148656


namespace correct_answers_l148_148578

noncomputable def find_numbers (S P : ℝ) : ℝ × ℝ :=
  let discriminant := S^2 - 4 * P
  if h : discriminant ≥ 0 then
    let sqrt_disc := real.sqrt discriminant
    ((S + sqrt_disc) / 2, (S - sqrt_disc) / 2)
  else
    sorry -- This would need further handling of complex numbers if discriminant < 0

theorem correct_answers (S P : ℝ) (x y : ℝ) :
  x + y = S ∧ x * y = P →
  (x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2)
  ∨ (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2) := 
begin
  assume h,
  sorry -- proof steps go here
end

end correct_answers_l148_148578


namespace chess_tournament_l148_148820

theorem chess_tournament :
  ∀ (participants : Finset ℕ) (pairing : participants → participants → Prop),
  participants.card = 10 →
  (∀ x y ∈ participants, x ≠ y → pairing x y → pairing y x) →
  (∀ x ∈ participants, (∑ y in participants, if pairing x y then 1 else 0).nat_abs = 9) →
  (∀ ⦃x y⦄, x ∈ participants → y ∈ participants → pairing x y ∨ pairing y x) →
  (∃ t : Finset (Finset ℕ), t.card = 2 ∧ ∀ (x ∈ t) (y ∈ t), x ≠ y) →
  (∃ round : finset (participants × participants), round.card = 5 ∧
    ∀ (game ∈ round), ∃ town t, ∀ (game = (x, y)), x ∈ t ∧ y ∈ t) →
  (∃ (x y : participants), x ≠ y ∧ x ∈ participants ∧ y ∈ participants ∧ (pairing x y ∨ pairing y x) ∧
    (∃ town, x ∈ town ∧ y ∈ town)) :=
by sorry

end chess_tournament_l148_148820


namespace sin_squared_value_l148_148989

theorem sin_squared_value (x : ℝ) (h : Real.tan x = 1 / 2) : 
  Real.sin (π / 4 + x) ^ 2 = 9 / 10 :=
by
  -- Proof part, skipped.
  sorry

end sin_squared_value_l148_148989


namespace find_smallest_r_disjoint_set_l148_148139

theorem find_smallest_r_disjoint_set 
  (A : Set ℕ := {a | ∃ k : ℕ, a = 3 + 10 * k ∨ a = 6 + 26 * k ∨ a = 5 + 29 * k})
  : ∃ (r b : ℕ), (∀ k : ℕ, b + r * k ∉ A) ∧ r = 290 :=
by
  sorry

end find_smallest_r_disjoint_set_l148_148139


namespace area_of_shaded_region_l148_148846

-- Define the vertices of the larger square
def large_square_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 40), (0, 40)]

-- Define the vertices of the polygon forming the shaded area
def shaded_polygon_vertices : List (ℝ × ℝ) := [(0, 0), (20, 0), (40, 30), (40, 40), (10, 40), (0, 10)]

-- Provide the area of the larger square for reference
def large_square_area : ℝ := 1600

-- Provide the area of the triangles subtracted
def triangles_area : ℝ := 450

-- The main theorem stating the problem:
theorem area_of_shaded_region :
  let shaded_area := large_square_area - triangles_area
  shaded_area = 1150 :=
by
  sorry

end area_of_shaded_region_l148_148846


namespace candy_cost_l148_148822

theorem candy_cost (C : ℝ) 
  (h1 : 20 + 40 = 60) 
  (h2 : 5 * 40 + 20 * C = 60 * 6) : 
  C = 8 :=
by
  sorry

end candy_cost_l148_148822


namespace solve_for_x_l148_148033

theorem solve_for_x (x y : ℝ) (h₁ : y = (x^2 - 9) / (x - 3)) (h₂ : y = 3 * x - 4) : x = 7 / 2 :=
by sorry

end solve_for_x_l148_148033


namespace cone_altitude_ratio_l148_148347

variable (r h : ℝ)
variable (radius_condition : r > 0)
variable (volume_condition : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3)

theorem cone_altitude_ratio {r h : ℝ}
  (radius_condition : r > 0) 
  (volume_condition : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 := by
  sorry

end cone_altitude_ratio_l148_148347


namespace six_digit_numbers_with_zero_l148_148686

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148686


namespace solve_for_x_l148_148157

theorem solve_for_x (x : ℝ) : 5 * x - 3 * x = 405 - 9 * (x + 4) → x = 369 / 11 := by
  sorry

end solve_for_x_l148_148157


namespace beads_per_necklace_l148_148852

theorem beads_per_necklace (n : ℕ) (b : ℕ) (total_beads : ℕ) (total_necklaces : ℕ)
  (h1 : total_necklaces = 6) (h2 : total_beads = 18) (h3 : b * total_necklaces = total_beads) :
  b = 3 :=
by {
  sorry
}

end beads_per_necklace_l148_148852


namespace six_digit_numbers_with_at_least_one_zero_l148_148660

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l148_148660


namespace probability_all_white_balls_l148_148016

-- Definitions
def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 7

-- Lean theorem statement
theorem probability_all_white_balls :
  (Nat.choose white_balls balls_drawn : ℚ) / (Nat.choose total_balls balls_drawn) = 8 / 6435 :=
sorry

end probability_all_white_balls_l148_148016


namespace value_of_m_over_q_l148_148650

-- Definitions for the given conditions
variables (n m p q : ℤ) 

-- Main theorem statement
theorem value_of_m_over_q (h1 : m = 10 * n) (h2 : p = 2 * n) (h3 : p = q / 5) :
  m / q = 1 :=
sorry

end value_of_m_over_q_l148_148650


namespace monthly_growth_rate_l148_148017

theorem monthly_growth_rate (x : ℝ)
  (turnover_may : ℝ := 1)
  (turnover_july : ℝ := 1.21)
  (growth_rate_condition : (1 + x) ^ 2 = 1.21) :
  x = 0.1 :=
sorry

end monthly_growth_rate_l148_148017


namespace ratio_of_A_to_B_is_4_l148_148297

noncomputable def A_share : ℝ := 360
noncomputable def B_share : ℝ := 90
noncomputable def ratio_A_B : ℝ := A_share / B_share

theorem ratio_of_A_to_B_is_4 : ratio_A_B = 4 :=
by
  -- This is the proof that we are skipping
  sorry

end ratio_of_A_to_B_is_4_l148_148297


namespace negation_of_proposition_l148_148554

open Nat 

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n > 0 ∧ n^2 > 2^n) ↔ ∀ n : ℕ, n > 0 → n^2 ≤ 2^n :=
by
  sorry

end negation_of_proposition_l148_148554


namespace quadratic_inequality_has_real_solutions_l148_148380

theorem quadratic_inequality_has_real_solutions (c : ℝ) (h : 0 < c) : 
  (∃ x : ℝ, x^2 - 6 * x + c < 0) ↔ (0 < c ∧ c < 9) :=
sorry

end quadratic_inequality_has_real_solutions_l148_148380


namespace b_is_nth_power_l148_148427

theorem b_is_nth_power (b n : ℕ) (h1 : b > 1) (h2 : n > 1) 
    (h3 : ∀ k > 1, ∃ a_k : ℕ, k ∣ (b - a_k^n)) : 
    ∃ A : ℕ, b = A^n :=
sorry

end b_is_nth_power_l148_148427


namespace five_less_than_sixty_percent_of_cats_l148_148797

theorem five_less_than_sixty_percent_of_cats (hogs cats : ℕ) 
  (hogs_eq : hogs = 3 * cats)
  (hogs_value : hogs = 75) : 
  5 < 60 * cats / 100 :=
by {
  sorry
}

end five_less_than_sixty_percent_of_cats_l148_148797


namespace six_digit_numbers_with_zero_l148_148738

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148738


namespace opposite_of_neg_one_third_l148_148794

theorem opposite_of_neg_one_third : -(-1/3) = 1/3 := 
sorry

end opposite_of_neg_one_third_l148_148794


namespace minimum_additional_small_bottles_needed_l148_148349

-- Definitions from the problem conditions
def small_bottle_volume : ℕ := 45
def large_bottle_total_volume : ℕ := 600
def initial_volume_in_large_bottle : ℕ := 90

-- The proof problem: How many more small bottles does Jasmine need to fill the large bottle?
theorem minimum_additional_small_bottles_needed : 
  (large_bottle_total_volume - initial_volume_in_large_bottle + small_bottle_volume - 1) / small_bottle_volume = 12 := 
by 
  sorry

end minimum_additional_small_bottles_needed_l148_148349


namespace probability_not_grade_5_l148_148234

theorem probability_not_grade_5 :
  let A1 := 0.3
  let A2 := 0.4
  let A3 := 0.2
  let A4 := 0.1
  (A1 + A2 + A3 + A4 = 1) → (1 - A1 = 0.7) := by
  intros A1_def A2_def A3_def A4_def h
  sorry

end probability_not_grade_5_l148_148234


namespace Carmen_candle_burn_time_l148_148489

theorem Carmen_candle_burn_time
  (night_to_last_candle_first_scenario : ℕ := 8)
  (hours_per_night_second_scenario : ℕ := 2)
  (nights_second_scenario : ℕ := 24)
  (candles_second_scenario : ℕ := 6) :
  ∃ T : ℕ, (night_to_last_candle_first_scenario * T = hours_per_night_second_scenario * (nights_second_scenario / candles_second_scenario)) ∧ T = 1 :=
by
  let T := (hours_per_night_second_scenario * (nights_second_scenario / candles_second_scenario)) / night_to_last_candle_first_scenario
  have : T = 1 := by sorry
  use T
  exact ⟨ by sorry, this⟩

end Carmen_candle_burn_time_l148_148489


namespace exists_rectangle_of_same_color_in_colored_plane_l148_148373

theorem exists_rectangle_of_same_color_in_colored_plane :
  ∀ (color : ℕ × ℕ → fin 3), ∃ (a b c d : ℕ × ℕ),
  a.1 = c.1 ∧ b.1 = d.1 ∧ a.2 = b.2 ∧ c.2 = d.2 ∧ color a = color b ∧ color b = color c ∧ color c = color d := by
sorry

end exists_rectangle_of_same_color_in_colored_plane_l148_148373


namespace perpendicular_lines_l148_148534

theorem perpendicular_lines (a : ℝ)
  (line1 : (a^2 + a - 6) * x + 12 * y - 3 = 0)
  (line2 : (a - 1) * x - (a - 2) * y + 4 - a = 0) :
  (a - 2) * (a - 3) * (a + 5) = 0 := sorry

end perpendicular_lines_l148_148534


namespace working_mom_hours_at_work_l148_148483

-- Definitions corresponding to the conditions
def hours_awake : ℕ := 16
def work_percentage : ℝ := 0.50

-- The theorem to be proved
theorem working_mom_hours_at_work : work_percentage * hours_awake = 8 :=
by sorry

end working_mom_hours_at_work_l148_148483


namespace final_coordinates_l148_148222

-- Definitions for the given conditions
def initial_point : ℝ × ℝ := (-2, 6)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

-- The final proof statement
theorem final_coordinates :
  let S_reflected := reflect_x_axis initial_point
  let S_translated := translate_up S_reflected 10
  S_translated = (-2, 4) :=
by
  sorry

end final_coordinates_l148_148222


namespace parabola_constant_unique_l148_148227

theorem parabola_constant_unique (b c : ℝ) :
  (∀ x y : ℝ, (x = 2 ∧ y = 20) → y = x^2 + b * x + c) →
  (∀ x y : ℝ, (x = -2 ∧ y = -4) → y = x^2 + b * x + c) →
  c = 4 :=
by
    sorry

end parabola_constant_unique_l148_148227


namespace mail_per_house_l148_148201

theorem mail_per_house (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : 
  total_mail / total_houses = 6 := 
by 
  sorry

end mail_per_house_l148_148201


namespace min_value_expression_l148_148552

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) ≥ 3 :=
by
  sorry

end min_value_expression_l148_148552


namespace arc_length_parametric_l148_148190

open Real Interval

noncomputable def arc_length (f_x f_y : ℝ → ℝ) (t1 t2 : ℝ) :=
  ∫ t in Set.Icc t1 t2, sqrt ((deriv f_x t)^2 + (deriv f_y t)^2)

theorem arc_length_parametric :
  arc_length
    (λ t => 2.5 * (t - sin t))
    (λ t => 2.5 * (1 - cos t))
    (π / 2) π = 5 * sqrt 2 :=
by
  sorry

end arc_length_parametric_l148_148190


namespace max_value_of_f_l148_148386

noncomputable def f : ℝ → ℝ := λ x, 5^x - 25^x

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 1/4 ∧ ∃ y : ℝ, f y = 1/4 := by
  sorry

end max_value_of_f_l148_148386


namespace determine_n_l148_148985

variable (x a n : ℕ)

def binomial_term (n k : ℕ) (x a : ℤ) : ℤ :=
  Nat.choose n k * x ^ (n - k) * a ^ k

theorem determine_n (hx : 0 < x) (ha : 0 < a)
  (h4 : binomial_term n 3 x a = 330)
  (h5 : binomial_term n 4 x a = 792)
  (h6 : binomial_term n 5 x a = 1716) :
  n = 7 :=
sorry

end determine_n_l148_148985


namespace cuboid_third_face_area_l148_148582

-- Problem statement in Lean
theorem cuboid_third_face_area (l w h : ℝ) (A₁ A₂ V : ℝ) 
  (hw1 : l * w = 120)
  (hw2 : w * h = 60)
  (hw3 : l * w * h = 720) : 
  l * h = 72 :=
sorry

end cuboid_third_face_area_l148_148582


namespace arccos_sin_eq_l148_148968

open Real

-- Definitions from the problem conditions
noncomputable def radians := π / 180

-- The theorem we need to prove
theorem arccos_sin_eq : arccos (sin 3) = 3 - (π / 2) :=
by
  sorry

end arccos_sin_eq_l148_148968


namespace six_digit_numbers_with_zero_l148_148668

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l148_148668


namespace triangle_medians_and_area_l148_148003

/-- Given a triangle with side lengths 13, 14, and 15,
    prove that the sum of the squares of the lengths of the medians is 385
    and the area of the triangle is 84. -/
theorem triangle_medians_and_area :
  let a := 13
  let b := 14
  let c := 15
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let m_a := Real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2
  let m_b := Real.sqrt (2 * c^2 + 2 * a^2 - b^2) / 2
  let m_c := Real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2
  m_a^2 + m_b^2 + m_c^2 = 385 ∧ area = 84 := sorry

end triangle_medians_and_area_l148_148003


namespace coefficient_x_squared_l148_148332

/-- Prove that the coefficient of x^2 in the expansion of (1 + 1/x^2)(1 + x)^6 is 30 -/
theorem coefficient_x_squared (x : ℝ) : 
  (polynomial.coeff (((1 + polynomial.C (1/x^2 : ℝ)) * (1 + x)^6) : polynomial ℝ) 2) = 30 := 
by {
  sorry
}

end coefficient_x_squared_l148_148332


namespace new_students_count_l148_148306

theorem new_students_count (O N : ℕ) (avg_class_age avg_new_students_age avg_decrease original_strength : ℕ)
  (h1 : avg_class_age = 40)
  (h2 : avg_new_students_age = 32)
  (h3 : avg_decrease = 4)
  (h4 : original_strength = 8)
  (total_age_class : ℕ := avg_class_age * original_strength)
  (new_avg_age : ℕ := avg_class_age - avg_decrease)
  (total_age_new_students : ℕ := avg_new_students_age * N)
  (total_students : ℕ := original_strength + N)
  (new_total_age : ℕ := total_age_class + total_age_new_students)
  (new_avg_class_age : ℕ := new_total_age / total_students)
  (h5 : new_avg_class_age = new_avg_age) : N = 8 :=
by
  sorry

end new_students_count_l148_148306


namespace tangent_of_alpha_solution_l148_148486

variable {α : ℝ}

theorem tangent_of_alpha_solution
  (h : 3 * Real.tan α - Real.sin α + 4 * Real.cos α = 12) :
  Real.tan α = 4 :=
sorry

end tangent_of_alpha_solution_l148_148486


namespace water_tank_capacity_l148_148352

theorem water_tank_capacity (rate : ℝ) (time : ℝ) (fraction : ℝ) (capacity : ℝ) : 
(rate = 10) → (time = 300) → (fraction = 3/4) → 
(rate * time = fraction * capacity) → 
capacity = 4000 := 
by
  intros h_rate h_time h_fraction h_equation
  rw [h_rate, h_time, h_fraction] at h_equation
  linarith

end water_tank_capacity_l148_148352


namespace question1_question2_l148_148782

theorem question1 (x : ℝ) : (1 < x^2 - 3*x + 1 ∧ x^2 - 3*x + 1 < 9 - x) ↔ (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 4) :=
by sorry

theorem question2 (x a : ℝ) : ((x - a)/(x - a^2) < 0)
  ↔ (a = 0 ∨ a = 1 → false)
  ∨ (0 < a ∧ a < 1 ∧ a^2 < x ∧ x < a)
  ∨ ((a < 0 ∨ a > 1) ∧ a < x ∧ x < a^2) :=
by sorry

end question1_question2_l148_148782


namespace hyperbola_focal_product_l148_148083

-- Define the hyperbola with given equation and point P conditions
def Hyperbola (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1 }

-- Define properties of vectors related to foci
def perpendicular (v1 v2 : ℝ × ℝ) := (v1.1 * v2.1 + v1.2 * v2.2 = 0)

-- Define the point-focus distance product condition
noncomputable def focalProduct (P F1 F2 : ℝ × ℝ) := (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) * (Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))

theorem hyperbola_focal_product :
  ∀ (a b : ℝ) (F1 F2 P : ℝ × ℝ),
  Hyperbola a b P ∧ perpendicular (P - F1) (P - F2) ∧
  -- Assuming a parabola property ties F1 with a specific value
  ((P.1 - F1.1)^2 + (P.2 - F1.2)^2 = 4 * (Real.sqrt  ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))) →
  focalProduct P F1 F2 = 14 := by
  sorry

end hyperbola_focal_product_l148_148083


namespace find_g_1_l148_148828

theorem find_g_1 (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (2*x - 3) = 2*x^2 - x + 4) : 
  g 1 = 11.5 :=
sorry

end find_g_1_l148_148828


namespace cos_value_of_angle_l148_148863

theorem cos_value_of_angle (α : ℝ) (h : Real.sin (α + Real.pi / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * Real.pi / 3) = -7 / 9 :=
by
  sorry

end cos_value_of_angle_l148_148863


namespace new_area_of_card_l148_148153

-- Conditions from the problem
def original_length : ℕ := 5
def original_width : ℕ := 7
def shortened_length := original_length - 2
def shortened_width := original_width - 1

-- Statement of the proof problem
theorem new_area_of_card : shortened_length * shortened_width = 18 :=
by
  sorry

end new_area_of_card_l148_148153


namespace six_digit_numbers_with_zero_l148_148741

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148741


namespace total_payment_is_correct_l148_148556

def daily_rental_cost : ℝ := 30
def per_mile_cost : ℝ := 0.25
def one_time_service_charge : ℝ := 15
def rent_duration : ℝ := 4
def distance_driven : ℝ := 500

theorem total_payment_is_correct :
  (daily_rental_cost * rent_duration + per_mile_cost * distance_driven + one_time_service_charge) = 260 := 
by
  sorry

end total_payment_is_correct_l148_148556


namespace cost_of_tree_planting_l148_148298

theorem cost_of_tree_planting 
  (initial_temp final_temp : ℝ) (temp_drop_per_tree cost_per_tree : ℝ) 
  (h_initial: initial_temp = 80) (h_final: final_temp = 78.2) 
  (h_temp_drop_per_tree: temp_drop_per_tree = 0.1) 
  (h_cost_per_tree: cost_per_tree = 6) : 
  (final_temp - initial_temp) / temp_drop_per_tree * cost_per_tree = 108 := 
by
  sorry

end cost_of_tree_planting_l148_148298


namespace min_value_reciprocals_l148_148521

theorem min_value_reciprocals (a b : ℝ) 
  (h1 : 2 * a + 2 * b = 2) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  (1 / a + 1 / b) ≥ 4 :=
sorry

end min_value_reciprocals_l148_148521


namespace compare_numbers_l148_148841

theorem compare_numbers : 222^2 < 22^22 ∧ 22^22 < 2^222 :=
by {
  sorry
}

end compare_numbers_l148_148841


namespace raine_change_l148_148829

noncomputable def price_bracelet : ℝ := 15
noncomputable def price_necklace : ℝ := 10
noncomputable def price_mug : ℝ := 20
noncomputable def price_keychain : ℝ := 5

noncomputable def quantity_bracelet : ℕ := 3
noncomputable def quantity_necklace : ℕ := 2
noncomputable def quantity_mug : ℕ := 1
noncomputable def quantity_keychain : ℕ := 4

noncomputable def discount_rate : ℝ := 0.12

noncomputable def amount_given : ℝ := 100

-- The total cost before discount
noncomputable def total_before_discount : ℝ := 
  quantity_bracelet * price_bracelet + 
  quantity_necklace * price_necklace + 
  quantity_mug * price_mug + 
  quantity_keychain * price_keychain

-- The discount amount
noncomputable def discount_amount : ℝ := total_before_discount * discount_rate

-- The final amount Raine has to pay after discount
noncomputable def final_amount : ℝ := total_before_discount - discount_amount

-- The change Raine gets back
noncomputable def change : ℝ := amount_given - final_amount

theorem raine_change : change = 7.60 := 
by sorry

end raine_change_l148_148829


namespace general_formula_for_sequence_l148_148303

noncomputable def a_n (n : ℕ) : ℕ := sorry
noncomputable def S_n (n : ℕ) : ℕ := sorry

theorem general_formula_for_sequence {n : ℕ} (hn: n > 0)
  (h1: ∀ n, a_n n > 0)
  (h2: ∀ n, 4 * S_n n = (a_n n)^2 + 2 * (a_n n))
  : a_n n = 2 * n := sorry

end general_formula_for_sequence_l148_148303


namespace six_digit_numbers_with_zero_l148_148690

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l148_148690


namespace find_polynomial_l148_148504

noncomputable def polynomial_satisfies_conditions (P : Polynomial ℝ) : Prop :=
  P.eval 0 = 0 ∧ ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1

theorem find_polynomial (P : Polynomial ℝ) (h : polynomial_satisfies_conditions P) : P = Polynomial.X :=
  sorry

end find_polynomial_l148_148504


namespace six_digit_numbers_with_zero_l148_148715

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l148_148715


namespace average_weight_increase_l148_148307

theorem average_weight_increase 
  (w_old : ℝ) (w_new : ℝ) (n : ℕ) 
  (h1 : w_old = 65) 
  (h2 : w_new = 93) 
  (h3 : n = 8) : 
  (w_new - w_old) / n = 3.5 := 
by 
  sorry

end average_weight_increase_l148_148307


namespace concentric_spheres_volume_l148_148454

theorem concentric_spheres_volume :
  let r1 := 4
  let r2 := 7
  let r3 := 10
  let volume (r : ℝ) := (4/3) * Real.pi * r^3
  volume r3 - volume r2 = 876 * Real.pi := 
by
  let r1 := 4
  let r2 := 7
  let r3 := 10
  let volume (r : ℝ) := (4/3) * Real.pi * r^3
  show volume r3 - volume r2 = 876 * Real.pi
  sorry

end concentric_spheres_volume_l148_148454


namespace jimmy_more_sheets_than_tommy_l148_148597

theorem jimmy_more_sheets_than_tommy 
  (jimmy_initial_sheets : ℕ)
  (tommy_initial_sheets : ℕ)
  (additional_sheets : ℕ)
  (h1 : tommy_initial_sheets = jimmy_initial_sheets + 25)
  (h2 : jimmy_initial_sheets = 58)
  (h3 : additional_sheets = 85) :
  (jimmy_initial_sheets + additional_sheets) - tommy_initial_sheets = 60 := 
by
  sorry

end jimmy_more_sheets_than_tommy_l148_148597


namespace average_score_girls_cedar_drake_l148_148840

theorem average_score_girls_cedar_drake
  (C c D d : ℕ)
  (cedar_boys_score cedar_girls_score cedar_combined_score
   drake_boys_score drake_girls_score drake_combined_score combined_boys_score : ℝ)
  (h1 : cedar_boys_score = 68)
  (h2 : cedar_girls_score = 80)
  (h3 : cedar_combined_score = 73)
  (h4 : drake_boys_score = 75)
  (h5 : drake_girls_score = 88)
  (h6 : drake_combined_score = 83)
  (h7 : combined_boys_score = 74)
  (h8 : (68 * C + 80 * c) / (C + c) = 73)
  (h9 : (75 * D + 88 * d) / (D + d) = 83)
  (h10 : (68 * C + 75 * D) / (C + D) = 74) :
  (80 * c + 88 * d) / (c + d) = 87 :=
by
  -- proof is omitted
  sorry

end average_score_girls_cedar_drake_l148_148840


namespace kay_exercise_time_l148_148279

variable (A W : ℕ)
variable (exercise_total : A + W = 250) 
variable (ratio_condition : A * 2 = 3 * W)

theorem kay_exercise_time :
  A = 150 ∧ W = 100 :=
by
  sorry

end kay_exercise_time_l148_148279


namespace x_100_equals_2_power_397_l148_148084

-- Define the sequences
noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b_n (n : ℕ) : ℕ := 5*n - 3

-- Define the merged sequence x_n
noncomputable def x_n (k : ℕ) : ℕ := 2^(4*k - 3)

-- Prove x_100 is 2^397
theorem x_100_equals_2_power_397 : x_n 100 = 2^397 := by
  unfold x_n
  show 2^(4*100 - 3) = 2^397
  rfl

end x_100_equals_2_power_397_l148_148084


namespace triangle_max_area_proof_l148_148889

open Real

noncomputable def triangle_max_area (A B C : ℝ) (AB : ℝ) (tanA tanB : ℝ) : Prop :=
  AB = 4 ∧ tanA * tanB = 3 / 4 → ∃ S : ℝ, S = 2 * sqrt 3

theorem triangle_max_area_proof (A B C : ℝ) (tanA tanB : ℝ) (AB : ℝ) : 
  triangle_max_area A B C AB tanA tanB :=
by
  sorry

end triangle_max_area_proof_l148_148889


namespace sin_B_value_cos_A_minus_cos_C_value_l148_148536

variables {A B C : ℝ} {a b c : ℝ}

theorem sin_B_value (h₁ : 4 * b * (Real.sin A) = Real.sqrt 7 * a) : Real.sin B = Real.sqrt 7 / 4 := 
sorry

theorem cos_A_minus_cos_C_value (h₁ : 4 * b * (Real.sin A) = Real.sqrt 7 * a) (h₂ : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := 
sorry

end sin_B_value_cos_A_minus_cos_C_value_l148_148536


namespace more_sons_than_daughters_prob_l148_148293

noncomputable def binom (n k : ℕ) : ℚ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

def prob_sons (k : ℕ) : ℚ := binom 8 k * (0.4^k) * (0.6^(8-k))

def prob_more_sons_than_daughters : ℚ :=
  prob_sons 5 + prob_sons 6 + prob_sons 7 + prob_sons 8

theorem more_sons_than_daughters_prob : prob_more_sons_than_daughters = 0.1752 := by
  sorry

end more_sons_than_daughters_prob_l148_148293


namespace simplify_T_l148_148905

noncomputable def T (x : ℝ) : ℝ :=
  (x+1)^4 - 4*(x+1)^3 + 6*(x+1)^2 - 4*(x+1) + 1

theorem simplify_T (x : ℝ) : T x = x^4 :=
  sorry

end simplify_T_l148_148905


namespace arithmetic_sequence_a1_a9_l148_148899

theorem arithmetic_sequence_a1_a9 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum_456 : a 4 + a 5 + a 6 = 36) : 
  a 1 + a 9 = 24 := 
sorry

end arithmetic_sequence_a1_a9_l148_148899


namespace bottle_caps_total_l148_148854

-- Mathematical conditions
def x : ℕ := 18
def y : ℕ := 63

-- Statement to prove
theorem bottle_caps_total : x + y = 81 :=
by
  -- The proof is skipped as indicated by 'sorry'
  sorry

end bottle_caps_total_l148_148854


namespace jack_more_emails_morning_than_afternoon_l148_148118

def emails_afternoon := 3
def emails_morning := 5

theorem jack_more_emails_morning_than_afternoon :
  emails_morning - emails_afternoon = 2 :=
by
  sorry

end jack_more_emails_morning_than_afternoon_l148_148118


namespace six_digit_numbers_with_zero_l148_148685

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148685


namespace find_width_of_rect_box_l148_148964

-- Define the dimensions of the wooden box in meters
def wooden_box_length_m : ℕ := 8
def wooden_box_width_m : ℕ := 7
def wooden_box_height_m : ℕ := 6

-- Define the dimensions of the rectangular boxes in centimeters (with unknown width W)
def rect_box_length_cm : ℕ := 8
def rect_box_height_cm : ℕ := 6

-- Define the maximum number of rectangular boxes
def max_boxes : ℕ := 1000000

-- Define the constraint that the total volume of the boxes should not exceed the volume of the wooden box
theorem find_width_of_rect_box (W : ℕ) (wooden_box_volume : ℕ := (wooden_box_length_m * 100) * (wooden_box_width_m * 100) * (wooden_box_height_m * 100)) : 
  (rect_box_length_cm * W * rect_box_height_cm) * max_boxes = wooden_box_volume → W = 7 :=
by
  sorry

end find_width_of_rect_box_l148_148964


namespace increase_average_by_runs_l148_148787

theorem increase_average_by_runs :
  let total_runs_10_matches : ℕ := 10 * 32
  let runs_scored_next_match : ℕ := 87
  let total_runs_11_matches : ℕ := total_runs_10_matches + runs_scored_next_match
  let new_average_11_matches : ℚ := total_runs_11_matches / 11
  let increased_average : ℚ := 32 + 5
  new_average_11_matches = increased_average :=
by
  sorry

end increase_average_by_runs_l148_148787


namespace value_of_b_l148_148937

theorem value_of_b (b : ℝ) :
  (∀ x : ℝ, (-x^2 + b * x - 7 < 0) ↔ (x < 2 ∨ x > 6)) → b = 8 :=
by
  sorry

end value_of_b_l148_148937


namespace rectangle_same_color_exists_l148_148371

theorem rectangle_same_color_exists :
  ∀ (coloring : ℕ × ℕ → ℕ), 
  (∀ (x y : ℕ × ℕ), coloring x ∈ {0, 1, 2}) → 
  ∃ (a b c d : ℕ × ℕ), 
    a.1 = b.1 ∧ b.2 = c.2 ∧ c.1 = d.1 ∧ d.2 = a.2 ∧ 
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    coloring a = coloring b ∧ coloring b = coloring c ∧ coloring c = coloring d :=
sorry

end rectangle_same_color_exists_l148_148371


namespace no_valid_height_configuration_l148_148451

-- Define the heights and properties
variables {a : Fin 7 → ℝ}
variables {p : ℝ}

-- Define the condition as a theorem
theorem no_valid_height_configuration (h : ∀ n : Fin 7, p * a n + (1 - p) * a (n + 2) % 7 > 
                                         p * a (n + 3) % 7 + (1 - p) * a (n + 1) % 7) :
  ¬ (∃ (a : Fin 7 → ℝ), 
    (∀ n : Fin 7, p * a n + (1 - p) * a (n + 2) % 7 > 
                  p * a (n + 3) % 7 + (1 - p) * a (n + 1) % 7) ∧
    true) :=
sorry

end no_valid_height_configuration_l148_148451


namespace sum_first_ten_terms_arithmetic_l148_148127

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end sum_first_ten_terms_arithmetic_l148_148127


namespace train_crosses_platform_in_20s_l148_148834

noncomputable def timeToCrossPlatform (train_length : ℝ) (platform_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_length + platform_length
  total_distance / train_speed_mps

theorem train_crosses_platform_in_20s :
  timeToCrossPlatform 120 213.36 60 = 20 :=
by
  sorry

end train_crosses_platform_in_20s_l148_148834


namespace problem1_problem2_l148_148520

-- Define the function f
def f (x b : ℝ) := |2 * x + b|

-- First problem: prove if the solution set of |2x + b| <= 3 is {x | -1 ≤ x ≤ 2}, then b = -1.
theorem problem1 (b : ℝ) : (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 2 → |2 * x + b| ≤ 3)) → b = -1 :=
sorry

-- Second problem: given b = -1, prove that for all x ∈ ℝ, |2(x+3)-1| + |2(x+1)-1| ≥ -4.
theorem problem2 : (∀ x : ℝ, f (x + 3) (-1) + f (x + 1) (-1) ≥ -4) :=
sorry

end problem1_problem2_l148_148520


namespace tony_rollercoasters_l148_148934

theorem tony_rollercoasters :
  let s1 := 50 -- speed of the first rollercoaster
  let s2 := 62 -- speed of the second rollercoaster
  let s3 := 73 -- speed of the third rollercoaster
  let s4 := 70 -- speed of the fourth rollercoaster
  let s5 := 40 -- speed of the fifth rollercoaster
  let avg_speed := 59 -- Tony's average speed during the day
  let total_speed := s1 + s2 + s3 + s4 + s5
  total_speed / avg_speed = 5 := sorry

end tony_rollercoasters_l148_148934


namespace expression_divisible_by_84_l148_148558

theorem expression_divisible_by_84 (p : ℕ) (hp : p > 0) : (4 ^ (2 * p) - 3 ^ (2 * p) - 7) % 84 = 0 :=
by
  sorry

end expression_divisible_by_84_l148_148558


namespace photograph_perimeter_l148_148203

-- Definitions of the conditions
def photograph_is_rectangular : Prop := True
def one_inch_border_area (w l m : ℕ) : Prop := (w + 2) * (l + 2) = m
def three_inch_border_area (w l m : ℕ) : Prop := (w + 6) * (l + 6) = m + 52

-- Lean statement of the problem
theorem photograph_perimeter (w l m : ℕ) 
  (h1 : photograph_is_rectangular)
  (h2 : one_inch_border_area w l m)
  (h3 : three_inch_border_area w l m) : 
  2 * (w + l) = 10 := 
by 
  sorry

end photograph_perimeter_l148_148203


namespace arithmetic_seq_a10_l148_148244

variable (a : ℕ → ℚ)
variable (S : ℕ → ℚ)
variable (d : ℚ := 1)

def is_arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

def sum_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem arithmetic_seq_a10 (h_seq : is_arithmetic_seq a d)
                          (h_sum : sum_first_n_terms a S)
                          (h_condition : S 8 = 4 * S 4) :
  a 10 = 19/2 := 
sorry

end arithmetic_seq_a10_l148_148244


namespace trigonometric_identity_l148_148085

theorem trigonometric_identity (α : ℝ)
  (h1 : Real.sin (π + α) = 3 / 5)
  (h2 : π < α ∧ α < 3 * π / 2) :
  (Real.sin ((π + α) / 2) - Real.cos ((π + α) / 2)) / 
  (Real.sin ((π - α) / 2) - Real.cos ((π - α) / 2)) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l148_148085


namespace reflections_composition_rotation_l148_148911

variable {α : ℝ} -- defining the angle α
variable {O : ℝ × ℝ} -- defining the point O, assuming the plane is represented as ℝ × ℝ

-- Define the lines that form the sides of the angle
variable (L1 L2 : ℝ × ℝ → Prop)

-- Assume α is the angle between L1 and L2 with O as the vertex
variable (hL1 : (L1 O))
variable (hL2 : (L2 O))

-- Assume reflections across L1 and L2
def reflect (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

theorem reflections_composition_rotation :
  ∀ A : ℝ × ℝ, (reflect (reflect A L1) L2) = sorry := 
sorry

end reflections_composition_rotation_l148_148911


namespace domain_shift_l148_148108

theorem domain_shift (f : ℝ → ℝ) (dom_f : ∀ x, 1 ≤ x ∧ x ≤ 4 → f x = f x) :
  ∀ x, -1 ≤ x ∧ x ≤ 2 ↔ (1 ≤ x + 2 ∧ x + 2 ≤ 4) :=
by
  sorry

end domain_shift_l148_148108


namespace framing_required_l148_148614

/- 
  Problem: A 5-inch by 7-inch picture is enlarged by quadrupling its dimensions.
  A 3-inch-wide border is then placed around each side of the enlarged picture.
  What is the minimum number of linear feet of framing that must be purchased
  to go around the perimeter of the border?
-/
def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3

theorem framing_required : (2 * ((original_width * enlargement_factor + 2 * border_width) + (original_height * enlargement_factor + 2 * border_width))) / 12 = 10 :=
by
  sorry

end framing_required_l148_148614


namespace six_digit_numbers_with_zero_l148_148742

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148742


namespace mean_inequalities_l148_148133

noncomputable def arith_mean (a : List ℝ) : ℝ := 
  (a.foldr (· + ·) 0) / a.length

noncomputable def geom_mean (a : List ℝ) : ℝ := 
  Real.exp ((a.foldr (λ x y => Real.log x + y) 0) / a.length)

noncomputable def harm_mean (a : List ℝ) : ℝ := 
  a.length / (a.foldr (λ x y => 1 / x + y) 0)

def is_positive (a : List ℝ) : Prop := 
  ∀ x ∈ a, x > 0

def bounds (a : List ℝ) (m g h : ℝ) : Prop := 
  let α := List.minimum a
  let β := List.maximum a
  α ≤ h ∧ h ≤ g ∧ g ≤ m ∧ m ≤ β

theorem mean_inequalities (a : List ℝ) (h g m : ℝ) (h_assoc: h = harm_mean a) (g_assoc: g = geom_mean a) (m_assoc: m = arith_mean a) :
  is_positive a → bounds a m g h :=
  
sorry

end mean_inequalities_l148_148133


namespace students_not_in_either_l148_148416

theorem students_not_in_either (total_students chemistry_students biology_students both_subjects neither_subjects : ℕ) 
  (h1 : total_students = 120) 
  (h2 : chemistry_students = 75) 
  (h3 : biology_students = 50) 
  (h4 : both_subjects = 15) 
  (h5 : neither_subjects = total_students - (chemistry_students - both_subjects + biology_students - both_subjects + both_subjects)) : 
  neither_subjects = 10 := 
by 
  sorry

end students_not_in_either_l148_148416


namespace mark_parking_tickets_l148_148633

theorem mark_parking_tickets (total_tickets : ℕ) (same_speeding_tickets : ℕ) (mark_parking_mult_sarah : ℕ) (sarah_speeding_tickets : ℕ) (mark_speeding_tickets : ℕ) (sarah_parking_tickets : ℕ) :
  total_tickets = 24 →
  mark_parking_mult_sarah = 2 →
  mark_speeding_tickets = same_speeding_tickets →
  sarah_speeding_tickets = same_speeding_tickets →
  same_speeding_tickets = 6 →
  sarah_parking_tickets = (total_tickets - 2 * same_speeding_tickets) / 3 →
  2 * sarah_parking_tickets = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw h1 at h6
  rw h5 at h6
  rw h2 at h6
  sorry

end mark_parking_tickets_l148_148633


namespace range_of_m_l148_148994

def proposition_p (m : ℝ) : Prop :=
  ∀ (x : ℝ), (1 ≤ x) → (x^2 - 2*m*x + 1/2 > 0)

def proposition_q (m : ℝ) : Prop :=
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (x^2 - m*x - 2 = 0)

theorem range_of_m (m : ℝ) (h1 : ¬ proposition_q m) (h2 : proposition_p m ∨ proposition_q m) :
  -1 < m ∧ m < 3/4 :=
  sorry

end range_of_m_l148_148994


namespace solution_set_of_inequality_l148_148868

theorem solution_set_of_inequality 
  (f : ℝ → ℝ)
  (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_at_2 : f 2 = 0)
  (condition : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x > 0) :
  {x : ℝ | x * f x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 2 < x} :=
sorry

end solution_set_of_inequality_l148_148868


namespace find_tents_l148_148249

theorem find_tents (x y : ℕ) (hx : x + y = 600) (hy : 1700 * x + 1300 * y = 940000) : x = 400 ∧ y = 200 :=
by
  sorry

end find_tents_l148_148249


namespace min_expression_value_l148_148326

theorem min_expression_value (x y z : ℝ) : ∃ x y z : ℝ, (xy - z)^2 + (x + y + z)^2 = 0 :=
by
  sorry

end min_expression_value_l148_148326


namespace find_fraction_l148_148410

theorem find_fraction (F N : ℝ) 
  (h1 : F * (1 / 4 * N) = 15)
  (h2 : (3 / 10) * N = 54) : 
  F = 1 / 3 := 
by
  sorry

end find_fraction_l148_148410


namespace repeating_decimal_eq_l148_148374

-- Define the repeating decimal as a constant
def repeating_decimal : ℚ := 47 / 99

-- Define what it means for a number to be the repeating decimal .474747...
def is_repeating_47 (x : ℚ) : Prop := x = repeating_decimal

-- The theorem to be proved
theorem repeating_decimal_eq : ∀ x : ℚ, is_repeating_47 x → x = 47 / 99 := by
  intros
  unfold is_repeating_47
  rw [H]
  rfl

end repeating_decimal_eq_l148_148374


namespace form_regular_octagon_l148_148612

def concentric_squares_form_regular_octagon (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^2 / b^2 = 2) : Prop :=
  ∀ (p : ℂ), ∃ (h₃ : ∀ (pvertices : ℤ → ℂ), -- vertices of the smaller square
                ∀ (lperpendiculars : ℤ → ℂ), -- perpendicular line segments
                true), -- additional conditions representing the perpendicular lines construction
    -- proving that the formed shape is a regular octagon:
    true -- Placeholder for actual condition/check for regular octagon

theorem form_regular_octagon (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^2 / b^2 = 2) :
  concentric_squares_form_regular_octagon a b h₀ h₁ h₂ :=
by sorry

end form_regular_octagon_l148_148612


namespace six_digit_numbers_with_zero_l148_148693

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l148_148693


namespace nonagon_diagonals_l148_148620

-- Define nonagon and its properties
def is_nonagon (n : ℕ) : Prop := n = 9
def has_parallel_sides (n : ℕ) : Prop := n = 9 ∧ true

-- Define the formula for calculating diagonals in a convex polygon
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The main theorem statement
theorem nonagon_diagonals :
  ∀ (n : ℕ), is_nonagon n → has_parallel_sides n → diagonals n = 27 :=  by 
  intros n hn _ 
  rw [is_nonagon] at hn
  rw [hn]
  sorry

end nonagon_diagonals_l148_148620


namespace walls_per_person_l148_148626

theorem walls_per_person (people : ℕ) (rooms : ℕ) (r4_walls r5_walls : ℕ) (total_walls : ℕ) (walls_each_person : ℕ)
  (h1 : people = 5)
  (h2 : rooms = 9)
  (h3 : r4_walls = 5 * 4)
  (h4 : r5_walls = 4 * 5)
  (h5 : total_walls = r4_walls + r5_walls)
  (h6 : walls_each_person = total_walls / people) :
  walls_each_person = 8 := by
  sorry

end walls_per_person_l148_148626


namespace algebra_expression_solution_l148_148076

theorem algebra_expression_solution
  (m : ℝ)
  (h : m^2 + m - 1 = 0) :
  m^3 + 2 * m^2 - 2001 = -2000 := by
  sorry

end algebra_expression_solution_l148_148076


namespace rectangular_prism_diagonal_l148_148090

theorem rectangular_prism_diagonal 
  (a b c : ℝ)
  (h1 : 2 * a * b + 2 * b * c + 2 * c * a = 11)
  (h2 : 4 * (a + b + c) = 24) :
  (a^2 + b^2 + c^2 = 25) :=
by {
  -- Sorry to skip the proof steps
  sorry
}

end rectangular_prism_diagonal_l148_148090


namespace max_k_solution_l148_148973

theorem max_k_solution
  (k x y : ℝ)
  (h_pos: 0 < k ∧ 0 < x ∧ 0 < y)
  (h_eq: 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  ∃ k, 8*k^3 - 8*k^2 - 7*k = 0 := 
sorry

end max_k_solution_l148_148973


namespace quotient_ab_solution_l148_148426

noncomputable def a : Real := sorry
noncomputable def b : Real := sorry

def condition1 (a b : Real) : Prop :=
  (1/(3 * a) + 1/b = 2011)

def condition2 (a b : Real) : Prop :=
  (1/a + 1/(3 * b) = 1)

theorem quotient_ab_solution (a b : Real) 
  (h1 : condition1 a b) 
  (h2 : condition2 a b) : 
  (a + b) / (a * b) = 1509 :=
sorry

end quotient_ab_solution_l148_148426


namespace six_digit_numbers_with_at_least_one_zero_l148_148702

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l148_148702


namespace correct_answers_l148_148579

noncomputable def find_numbers (S P : ℝ) : ℝ × ℝ :=
  let discriminant := S^2 - 4 * P
  if h : discriminant ≥ 0 then
    let sqrt_disc := real.sqrt discriminant
    ((S + sqrt_disc) / 2, (S - sqrt_disc) / 2)
  else
    sorry -- This would need further handling of complex numbers if discriminant < 0

theorem correct_answers (S P : ℝ) (x y : ℝ) :
  x + y = S ∧ x * y = P →
  (x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2)
  ∨ (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2) := 
begin
  assume h,
  sorry -- proof steps go here
end

end correct_answers_l148_148579


namespace compute_n_l148_148948

theorem compute_n (n : ℕ) : 5^n = 5 * 25^(3/2) * 125^(5/3) → n = 9 :=
by
  sorry

end compute_n_l148_148948


namespace six_digit_numbers_with_zero_l148_148714

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l148_148714


namespace real_roots_of_x_squared_minus_four_factorization_of_x_squared_minus_four_l148_148929

theorem real_roots_of_x_squared_minus_four :
  ∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
begin
  sorry
end

theorem factorization_of_x_squared_minus_four :
  ∀ x : ℝ, x^2 - 4 = (x + 2) * (x - 2) :=
begin
  sorry
end

end real_roots_of_x_squared_minus_four_factorization_of_x_squared_minus_four_l148_148929


namespace empty_plane_speed_l148_148802

variable (V : ℝ)

def speed_first_plane (V : ℝ) : ℝ := V - 2 * 50
def speed_second_plane (V : ℝ) : ℝ := V - 2 * 60
def speed_third_plane (V : ℝ) : ℝ := V - 2 * 40

theorem empty_plane_speed (V : ℝ) (h : (speed_first_plane V + speed_second_plane V + speed_third_plane V) / 3 = 500) : V = 600 :=
by 
  sorry

end empty_plane_speed_l148_148802


namespace correct_answers_l148_148580

noncomputable def find_numbers (S P : ℝ) : ℝ × ℝ :=
  let discriminant := S^2 - 4 * P
  if h : discriminant ≥ 0 then
    let sqrt_disc := real.sqrt discriminant
    ((S + sqrt_disc) / 2, (S - sqrt_disc) / 2)
  else
    sorry -- This would need further handling of complex numbers if discriminant < 0

theorem correct_answers (S P : ℝ) (x y : ℝ) :
  x + y = S ∧ x * y = P →
  (x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2)
  ∨ (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2) := 
begin
  assume h,
  sorry -- proof steps go here
end

end correct_answers_l148_148580


namespace arithmetic_progression_sum_l148_148261

-- Define the sum of the first 15 terms of the arithmetic progression
theorem arithmetic_progression_sum (a d : ℝ) 
  (h : (a + 3 * d) + (a + 11 * d) = 16) :
  (15 / 2) * (2 * a + 14 * d) = 120 := by
  sorry

end arithmetic_progression_sum_l148_148261


namespace find_numbers_l148_148573

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l148_148573


namespace find_a_value_l148_148441

noncomputable def a : ℝ := (384:ℝ)^(1/7)

variables (a b c : ℝ)
variables (h1 : a^2 / b = 2) (h2 : b^2 / c = 4) (h3 : c^2 / a = 6)

theorem find_a_value : a = 384^(1/7) :=
by
  sorry

end find_a_value_l148_148441


namespace gym_membership_cost_l148_148546

theorem gym_membership_cost 
    (cheap_monthly_fee : ℕ := 10)
    (cheap_signup_fee : ℕ := 50)
    (expensive_monthly_multiplier : ℕ := 3)
    (months_in_year : ℕ := 12)
    (expensive_signup_multiplier : ℕ := 4) :
    let cheap_gym_cost := cheap_monthly_fee * months_in_year + cheap_signup_fee
    let expensive_monthly_fee := cheap_monthly_fee * expensive_monthly_multiplier
    let expensive_gym_cost := expensive_monthly_fee * months_in_year + expensive_monthly_fee * expensive_signup_multiplier
    let total_cost := cheap_gym_cost + expensive_gym_cost
    total_cost = 650 :=
by
  sorry -- Proof is omitted because the focus is on the statement equivalency.

end gym_membership_cost_l148_148546


namespace Petya_meets_Vasya_l148_148913

def Petya_speed_on_paved (v_g : ℝ) : ℝ := 3 * v_g

def Distance_to_bridge (v_g : ℝ) : ℝ := 3 * v_g

def Vasya_travel_time (v_g t : ℝ) : ℝ := v_g * t

def Total_distance (v_g : ℝ) : ℝ := 2 * Distance_to_bridge v_g

def New_distance (v_g t : ℝ) : ℝ := (Total_distance v_g) - 2 * Vasya_travel_time v_g t

def Relative_speed (v_g : ℝ) : ℝ := v_g + v_g

def Time_to_meet (v_g : ℝ) : ℝ := (New_distance v_g 1) / Relative_speed v_g

theorem Petya_meets_Vasya (v_g : ℝ) : Time_to_meet v_g + 1 = 2 := by
  sorry

end Petya_meets_Vasya_l148_148913


namespace six_digit_numbers_with_at_least_one_zero_l148_148665

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l148_148665


namespace large_seat_capacity_l148_148160

-- Definition of conditions
def num_large_seats : ℕ := 7
def total_capacity_large_seats : ℕ := 84

-- Theorem to prove
theorem large_seat_capacity : total_capacity_large_seats / num_large_seats = 12 :=
by
  sorry

end large_seat_capacity_l148_148160


namespace simplify_expression_l148_148978

theorem simplify_expression (x y z : ℝ) : ((x + y) - (z - y)) - ((x + z) - (y + z)) = 3 * y - z := by
  sorry

end simplify_expression_l148_148978


namespace loan_amount_calculation_l148_148210

theorem loan_amount_calculation
  (annual_interest : ℝ) (interest_rate : ℝ) (time : ℝ) (loan_amount : ℝ)
  (h1 : annual_interest = 810)
  (h2 : interest_rate = 0.09)
  (h3 : time = 1)
  (h4 : loan_amount = annual_interest / (interest_rate * time)) :
  loan_amount = 9000 := by
sorry

end loan_amount_calculation_l148_148210


namespace min_value_of_expression_l148_148986

noncomputable def min_expression_value (y : ℝ) (hy : y > 2) : ℝ :=
  (y^2 + y + 1) / Real.sqrt (y - 2)

theorem min_value_of_expression (y : ℝ) (hy : y > 2) :
  min_expression_value y hy = 3 * Real.sqrt 35 :=
sorry

end min_value_of_expression_l148_148986


namespace solve_equation_theorem_l148_148567

noncomputable def solve_equations (S P : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  ((x1, y1), (x2, y2))

theorem solve_equation_theorem (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ x * y = P) ↔ (∃ (x1 y1 x2 y2 : ℝ), 
    ((x, y) = (x1, y1) ∨ (x, y) = (x2, y2)) ∧
    solve_equations S P = ((x1, y1), (x2, y2))) :=
by
  sorry

end solve_equation_theorem_l148_148567


namespace grasshopper_jump_distance_l148_148586

theorem grasshopper_jump_distance (frog_jump grasshopper_jump : ℝ) (h_frog : frog_jump = 40) (h_difference : frog_jump = grasshopper_jump + 15) : grasshopper_jump = 25 :=
by sorry

end grasshopper_jump_distance_l148_148586


namespace number_of_donuts_correct_l148_148561

noncomputable def number_of_donuts_in_each_box :=
  let x : ℕ := 12
  let total_boxes : ℕ := 4
  let donuts_given_to_mom : ℕ := x
  let donuts_given_to_sister : ℕ := 6
  let donuts_left : ℕ := 30
  x

theorem number_of_donuts_correct :
  ∀ (x : ℕ),
  (total_boxes * x - donuts_given_to_mom - donuts_given_to_sister = donuts_left) → x = 12 :=
by
  sorry

end number_of_donuts_correct_l148_148561


namespace find_number_of_Persians_l148_148765

variable (P : ℕ)  -- Number of Persian cats Jamie owns
variable (M : ℕ := 2)  -- Number of Maine Coons Jamie owns (given by conditions)
variable (G_P : ℕ := P / 2)  -- Number of Persian cats Gordon owns, which is half of Jamie's
variable (G_M : ℕ := M + 1)  -- Number of Maine Coons Gordon owns, one more than Jamie's
variable (H_P : ℕ := 0)  -- Number of Persian cats Hawkeye owns, which is 0
variable (H_M : ℕ := G_M - 1)  -- Number of Maine Coons Hawkeye owns, one less than Gordon's

theorem find_number_of_Persians (sum_cats : P + M + G_P + G_M + H_P + H_M = 13) : 
  P = 4 :=
by
  -- Proof can be filled in here
  sorry

end find_number_of_Persians_l148_148765


namespace most_stable_performance_l148_148988

-- Given variances for the four people
def S_A_var : ℝ := 0.56
def S_B_var : ℝ := 0.60
def S_C_var : ℝ := 0.50
def S_D_var : ℝ := 0.45

-- We need to prove that the variance for D is the smallest
theorem most_stable_performance :
  S_D_var < S_C_var ∧ S_D_var < S_A_var ∧ S_D_var < S_B_var :=
by
  sorry

end most_stable_performance_l148_148988


namespace box_weight_in_kg_l148_148342

def weight_of_one_bar : ℕ := 125 -- Weight of one chocolate bar in grams
def number_of_bars : ℕ := 16 -- Number of chocolate bars in the box
def grams_to_kg (g : ℕ) : ℕ := g / 1000 -- Function to convert grams to kilograms

theorem box_weight_in_kg : grams_to_kg (weight_of_one_bar * number_of_bars) = 2 :=
by
  sorry -- Proof is omitted

end box_weight_in_kg_l148_148342


namespace hog_cat_problem_l148_148799

theorem hog_cat_problem (hogs cats : ℕ)
  (hogs_eq : hogs = 75)
  (hogs_cats_relation : hogs = 3 * cats)
  : 5 < (6 / 10) * cats - 5 := 
by
  sorry

end hog_cat_problem_l148_148799


namespace ratio_of_men_to_women_l148_148610

-- Define conditions
def avg_height_students := 180
def avg_height_female := 170
def avg_height_male := 185

-- This is the math proof problem statement
theorem ratio_of_men_to_women (M W : ℕ) (h1 : (M * avg_height_male + W * avg_height_female) = (M + W) * avg_height_students) : 
  M / W = 2 :=
sorry

end ratio_of_men_to_women_l148_148610


namespace complex_number_solution_l148_148107

theorem complex_number_solution
  (z : ℂ)
  (h : i * (z - 1) = 1 + i) :
  z = 2 - i :=
sorry

end complex_number_solution_l148_148107


namespace angle_covered_in_three_layers_l148_148335

theorem angle_covered_in_three_layers 
  (total_coverage : ℝ) (sum_of_angles : ℝ) 
  (h1 : total_coverage = 90) (h2 : sum_of_angles = 290) : 
  ∃ x : ℝ, 3 * x + 2 * (90 - x) = 290 ∧ x = 20 :=
by
  sorry

end angle_covered_in_three_layers_l148_148335


namespace incircle_center_line_proof_l148_148850
  
open Real EuclideanGeometry

variables {A B C D M N : Point ℝ} 

-- Define cyclic trapezoid and its properties
def is_cyclic_trapezoid (A B C D : Point ℝ) : Prop :=
  cyclic A B C D ∧ parallel (Line.mk A B) (Line.mk C D) ∧ dist A B > dist C D

-- Define incircle tangency points
def tangent_points (A B C : Point ℝ) (M : Point ℝ) (N : Point ℝ) : Prop :=
  tangent_circle (incircle_triangle A B C) (Line.mk A B) M ∧
  tangent_circle (incircle_triangle A B C) (Line.mk A C) N

-- Define the center of the incircle lying on the line condition
def incircle_center_on_line (A B C D M N : Point ℝ) : Prop :=
  ∃ O : Point ℝ, center (incircle_trapezoid A B C D) = O ∧ 
                 collinear {M, N, O}

-- The main theorem
theorem incircle_center_line_proof 
  (A B C D M N : Point ℝ)
  (h1 : is_cyclic_trapezoid A B C D)
  (h2 : tangent_points A B C M N) :
  incircle_center_on_line A B C D M N :=
sorry

end incircle_center_line_proof_l148_148850


namespace min_value_of_sum_squares_l148_148136

theorem min_value_of_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
    x^2 + y^2 + z^2 ≥ 10 :=
sorry

end min_value_of_sum_squares_l148_148136


namespace min_value_of_w_l148_148366

noncomputable def w (x y : ℝ) : ℝ := 2 * x^2 + 3 * y^2 + 8 * x - 6 * y + 30

theorem min_value_of_w : ∃ x y : ℝ, ∀ (a b : ℝ), w x y ≤ w a b ∧ w x y = 19 :=
by
  sorry

end min_value_of_w_l148_148366


namespace smallest_discount_n_l148_148509

noncomputable def effective_discount_1 (x : ℝ) : ℝ := 0.64 * x
noncomputable def effective_discount_2 (x : ℝ) : ℝ := 0.614125 * x
noncomputable def effective_discount_3 (x : ℝ) : ℝ := 0.63 * x 

theorem smallest_discount_n (x : ℝ) (n : ℕ) (hx : x > 0) :
  (1 - n / 100 : ℝ) * x < effective_discount_1 x ∧ 
  (1 - n / 100 : ℝ) * x < effective_discount_2 x ∧ 
  (1 - n / 100 : ℝ) * x < effective_discount_3 x ↔ n = 39 := 
sorry

end smallest_discount_n_l148_148509


namespace six_digit_numbers_with_at_least_one_zero_correct_l148_148725

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l148_148725


namespace problem_f_2016_eq_l148_148093

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem problem_f_2016_eq :
  ∀ (a b : ℝ),
  f a b 2016 + f a b (-2016) + f' a b 2017 - f' a b (-2017) = 8 + 2 * b * 2016^3 :=
by
  intro a b
  sorry

end problem_f_2016_eq_l148_148093


namespace neg_parallelogram_is_rhombus_l148_148748

def parallelogram_is_rhombus := true

theorem neg_parallelogram_is_rhombus : ¬ parallelogram_is_rhombus := by
  sorry

end neg_parallelogram_is_rhombus_l148_148748


namespace six_digit_numbers_with_at_least_one_zero_correct_l148_148728

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l148_148728


namespace find_numbers_l148_148571

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l148_148571


namespace tom_spent_video_games_l148_148596

def cost_football := 14.02
def cost_strategy := 9.46
def cost_batman := 12.04
def total_spent := cost_football + cost_strategy + cost_batman

theorem tom_spent_video_games : total_spent = 35.52 :=
by
  sorry

end tom_spent_video_games_l148_148596


namespace cone_lateral_surface_area_l148_148753

-- Definitions based on conditions
def base_radius : ℝ := 2
def slant_height : ℝ := 3

-- Definition of the lateral surface area
def lateral_surface_area (r l : ℝ) : ℝ := π * r * l

-- Theorem stating the problem
theorem cone_lateral_surface_area : lateral_surface_area base_radius slant_height = 6 * π :=
by
  sorry

end cone_lateral_surface_area_l148_148753


namespace amoeba_population_after_ten_days_l148_148541

-- Definitions based on the conditions
def initial_population : ℕ := 3
def amoeba_growth (n : ℕ) : ℕ := initial_population * 2^n

-- Lean statement for the proof problem
theorem amoeba_population_after_ten_days : amoeba_growth 10 = 3072 :=
by 
  sorry

end amoeba_population_after_ten_days_l148_148541


namespace simplify_inv_sum_l148_148004

variables {x y z : ℝ}

theorem simplify_inv_sum (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = xyz / (yz + xz + xy) :=
by
  sorry

end simplify_inv_sum_l148_148004


namespace scientific_notation_of_1300000_l148_148192

theorem scientific_notation_of_1300000 : 1300000 = 1.3 * 10^6 :=
by
  sorry

end scientific_notation_of_1300000_l148_148192


namespace six_digit_numbers_with_zero_l148_148740

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148740


namespace scientific_notation_932700_l148_148625

theorem scientific_notation_932700 : 932700 = 9.327 * 10^5 :=
sorry

end scientific_notation_932700_l148_148625


namespace total_packets_needed_l148_148639

theorem total_packets_needed :
  let oak_seedlings := 420
  let oak_per_packet := 7
  let maple_seedlings := 825
  let maple_per_packet := 5
  let pine_seedlings := 2040
  let pine_per_packet := 12
  let oak_packets := oak_seedlings / oak_per_packet
  let maple_packets := maple_seedlings / maple_per_packet
  let pine_packets := pine_seedlings / pine_per_packet
  let total_packets := oak_packets + maple_packets + pine_packets
  total_packets = 395 := 
by {
  sorry
}

end total_packets_needed_l148_148639


namespace solve_for_x_l148_148916

-- Define the variables and conditions based on the problem statement
def equation (x : ℚ) := 5 * x - 3 * (x + 2) = 450 - 9 * (x - 4)

-- State the theorem to be proved, including the condition and the result
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = 44.72727272727273 := by
  sorry  -- The proof is omitted

end solve_for_x_l148_148916


namespace second_term_of_geo_series_l148_148026

theorem second_term_of_geo_series
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h_r : r = -1 / 3)
  (h_S : S = 25)
  (h_sum : S = a / (1 - r)) :
  (a * r) = -100 / 9 :=
by
  -- Definitions and conditions here are provided
  have hr : r = -1 / 3 := by exact h_r
  have hS : S = 25 := by exact h_S
  have hsum : S = a / (1 - r) := by exact h_sum
  -- The proof of (a * r) = -100 / 9 goes here
  sorry

end second_term_of_geo_series_l148_148026


namespace line_slope_intercept_product_l148_148923

theorem line_slope_intercept_product :
  ∃ (m b : ℝ), (b = -1) ∧ ((1 - (m * -1 + b) = 0) ∧ (mb = m * b)) ∧ (mb = 2) :=
by sorry

end line_slope_intercept_product_l148_148923


namespace unique_triple_satisfying_conditions_l148_148390

theorem unique_triple_satisfying_conditions :
  ∃! (x y z : ℝ), x + y = 4 ∧ xy - z^2 = 4 :=
sorry

end unique_triple_satisfying_conditions_l148_148390


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l148_148719

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l148_148719


namespace find_speed_A_l148_148600

-- Defining the distance between the two stations as 155 km.
def distance := 155

-- Train A starts from station A at 7 a.m. and meets Train B at 11 a.m.
-- Therefore, Train A travels for 4 hours.
def time_A := 4

-- Train B starts from station B at 8 a.m. and meets Train A at 11 a.m.
-- Therefore, Train B travels for 3 hours.
def time_B := 3

-- Speed of Train B is given as 25 km/h.
def speed_B := 25

-- Condition that the total distance covered by both trains equals the distance between the two stations.
def meet_condition (v_A : ℕ) := (time_A * v_A) + (time_B * speed_B) = distance

-- The Lean theorem statement to be proved
theorem find_speed_A (v_A := 20) : meet_condition v_A :=
by
  -- Using 'sorrry' to skip the proof
  sorry

end find_speed_A_l148_148600


namespace fixed_point_exists_l148_148585

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(a * (x + 1)) - 3

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = -1 :=
by
  -- Sorry for skipping the proof
  sorry

end fixed_point_exists_l148_148585


namespace correct_conclusions_l148_148460

/--
Given the following conditions:
1. For rolling two fair dice:
   A = "Odd number on the first roll".
   B = "Even number on the second roll".
   A and B are independent.
2. Event A and Event B have positive probabilities:
   P(A) > 0,
   P(B) > 0,

Prove that:
1. Events A and B are independent.
2. Events A and B cannot be both independent and mutually exclusive if their probabilities are positive.
-/
theorem correct_conclusions (A B : Event) (P : Event → ℝ) [IsProbabilityMeasure P] : 
    (independent P A B) ∧ 
    (0 < P A) ∧ (0 < P B) → 
    (¬mutually_exclusive A B) := sorry

end correct_conclusions_l148_148460


namespace percentage_half_day_students_l148_148216

theorem percentage_half_day_students
  (total_students : ℕ)
  (full_day_students : ℕ)
  (h_total : total_students = 80)
  (h_full_day : full_day_students = 60) :
  ((total_students - full_day_students) / total_students : ℚ) * 100 = 25 := 
by
  sorry

end percentage_half_day_students_l148_148216


namespace six_digit_numbers_with_zero_l148_148688

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l148_148688


namespace chadSavingsIsCorrect_l148_148638

noncomputable def chadSavingsAfterTaxAndConversion : ℝ :=
  let euroToUsd := 1.20
  let poundToUsd := 1.40
  let euroIncome := 600 * euroToUsd
  let poundIncome := 250 * poundToUsd
  let dollarIncome := 150 + 150
  let totalIncome := euroIncome + poundIncome + dollarIncome
  let taxRate := 0.10
  let taxedIncome := totalIncome * (1 - taxRate)
  let savingsRate := if taxedIncome ≤ 1000 then 0.20
                     else if taxedIncome ≤ 2000 then 0.30
                     else if taxedIncome ≤ 3000 then 0.40
                     else 0.50
  let savings := taxedIncome * savingsRate
  savings

theorem chadSavingsIsCorrect : chadSavingsAfterTaxAndConversion = 369.90 := by
  sorry

end chadSavingsIsCorrect_l148_148638


namespace value_of_other_bills_l148_148434

theorem value_of_other_bills (total_payment : ℕ) (num_fifty_dollar_bills : ℕ) (value_fifty_dollar_bill : ℕ) (num_other_bills : ℕ) 
  (total_fifty_dollars : ℕ) (remaining_payment : ℕ) (value_of_each_other_bill : ℕ) :
  total_payment = 170 →
  num_fifty_dollar_bills = 3 →
  value_fifty_dollar_bill = 50 →
  num_other_bills = 2 →
  total_fifty_dollars = num_fifty_dollar_bills * value_fifty_dollar_bill →
  remaining_payment = total_payment - total_fifty_dollars →
  value_of_each_other_bill = remaining_payment / num_other_bills →
  value_of_each_other_bill = 10 :=
by
  intros t_total_payment t_num_fifty_dollar_bills t_value_fifty_dollar_bill t_num_other_bills t_total_fifty_dollars t_remaining_payment t_value_of_each_other_bill
  sorry

end value_of_other_bills_l148_148434


namespace largest_of_four_consecutive_even_numbers_l148_148590

-- Conditions
def sum_of_four_consecutive_even_numbers (x : ℤ) : Prop :=
  x + (x + 2) + (x + 4) + (x + 6) = 92

-- Proof statement
theorem largest_of_four_consecutive_even_numbers (x : ℤ) 
  (h : sum_of_four_consecutive_even_numbers x) : x + 6 = 26 :=
by
  sorry

end largest_of_four_consecutive_even_numbers_l148_148590


namespace repeating_decimal_product_l148_148494

theorem repeating_decimal_product :
  let s := 0.\overline{456} in 
  s * 8 = 1216 / 333 :=
by
  sorry

end repeating_decimal_product_l148_148494


namespace six_digit_numbers_with_zero_l148_148711

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l148_148711


namespace new_solution_is_45_percent_liquid_x_l148_148781

-- Define initial conditions
def solution_y_initial_weight := 8.0 -- kilograms
def percent_liquid_x := 0.30
def percent_water := 0.70
def evaporated_water_weight := 4.0 -- kilograms
def added_solution_y_weight := 4.0 -- kilograms

-- Define the relevant quantities
def liquid_x_initial := solution_y_initial_weight * percent_liquid_x
def water_initial := solution_y_initial_weight * percent_water
def remaining_water_after_evaporation := water_initial - evaporated_water_weight

def liquid_x_after_evaporation := liquid_x_initial 
def water_after_evaporation := remaining_water_after_evaporation

def added_liquid_x := added_solution_y_weight * percent_liquid_x
def added_water := added_solution_y_weight * percent_water

def total_liquid_x := liquid_x_after_evaporation + added_liquid_x
def total_water := water_after_evaporation + added_water

def total_new_solution_weight := total_liquid_x + total_water

def new_solution_percent_liquid_x := (total_liquid_x / total_new_solution_weight) * 100

-- The theorem we want to prove
theorem new_solution_is_45_percent_liquid_x : new_solution_percent_liquid_x = 45 := by
  sorry

end new_solution_is_45_percent_liquid_x_l148_148781


namespace coffee_ratio_correct_l148_148784

noncomputable def ratio_of_guests (cups_weak : ℕ) (cups_strong : ℕ) (tablespoons_weak : ℕ) (tablespoons_strong : ℕ) (total_tablespoons : ℕ) : ℤ :=
  if (cups_weak * tablespoons_weak + cups_strong * tablespoons_strong = total_tablespoons) then
    (cups_weak * tablespoons_weak / gcd (cups_weak * tablespoons_weak) (cups_strong * tablespoons_strong)) /
    (cups_strong * tablespoons_strong / gcd (cups_weak * tablespoons_weak) (cups_strong * tablespoons_strong))
  else 0

theorem coffee_ratio_correct :
  ratio_of_guests 12 12 1 2 36 = 1 / 2 :=
by
  sorry

end coffee_ratio_correct_l148_148784


namespace Ming_initial_ladybugs_l148_148154

-- Define the conditions
def Sami_spiders : Nat := 3
def Hunter_ants : Nat := 12
def insects_remaining : Nat := 21
def ladybugs_flew_away : Nat := 2

-- Formalize the proof problem
theorem Ming_initial_ladybugs : Sami_spiders + Hunter_ants + (insects_remaining + ladybugs_flew_away) - (Sami_spiders + Hunter_ants) = 8 := by
  sorry

end Ming_initial_ladybugs_l148_148154


namespace product_of_two_numbers_l148_148167

noncomputable def leastCommonMultiple (a b : ℕ) : ℕ :=
  Nat.lcm a b

noncomputable def greatestCommonDivisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem product_of_two_numbers (a b : ℕ) :
  leastCommonMultiple a b = 36 ∧ greatestCommonDivisor a b = 6 → a * b = 216 := by
  sorry

end product_of_two_numbers_l148_148167


namespace find_incorrect_statement_l148_148627

variable (q n x y : ℚ)

theorem find_incorrect_statement :
  (∀ q, q < -1 → q < 1/q) ∧
  (∀ n, n ≥ 0 → -n ≥ n) ∧
  (∀ x, x < 0 → x^3 < x) ∧
  (∀ y, y < 0 → y^2 > y) →
  (∃ x, x < 0 ∧ ¬ (x^3 < x)) :=
by
  sorry

end find_incorrect_statement_l148_148627


namespace find_numbers_with_sum_and_product_l148_148577

theorem find_numbers_with_sum_and_product (S P : ℝ) :
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  (x1 + y1 = S ∧ x1 * y1 = P) ∨ (x2 + y2 = S ∧ x2 * y2 = P) :=
sorry

end find_numbers_with_sum_and_product_l148_148577


namespace division_problem_l148_148197

theorem division_problem (n : ℕ) (h : n / 6 = 209) : n = 1254 := 
sorry

end division_problem_l148_148197


namespace tan_3theta_eq_2_11_sin_3theta_eq_22_125_l148_148876

variable {θ : ℝ}

-- First, stating the condition \(\tan \theta = 2\)
axiom tan_theta_eq_2 : Real.tan θ = 2

-- Stating the proof problem for \(\tan 3\theta = \frac{2}{11}\)
theorem tan_3theta_eq_2_11 : Real.tan (3 * θ) = 2 / 11 :=
by 
  sorry

-- Stating the proof problem for \(\sin 3\theta = \frac{22}{125}\)
theorem sin_3theta_eq_22_125 : Real.sin (3 * θ) = 22 / 125 :=
by 
  sorry

end tan_3theta_eq_2_11_sin_3theta_eq_22_125_l148_148876


namespace area_PST_correct_l148_148539

noncomputable def area_of_triangle_PST : ℚ :=
  let P : ℚ × ℚ := (0, 0)
  let Q : ℚ × ℚ := (4, 0)
  let R : ℚ × ℚ := (0, 4)
  let S : ℚ × ℚ := (0, 2)
  let T : ℚ × ℚ := (8 / 3, 4 / 3)
  1 / 2 * (|P.1 * (S.2 - T.2) + S.1 * (T.2 - P.2) + T.1 * (P.2 - S.2)|)

theorem area_PST_correct : area_of_triangle_PST = 8 / 3 := sorry

end area_PST_correct_l148_148539


namespace num_second_grade_students_is_80_l148_148894

def ratio_fst : ℕ := 5
def ratio_snd : ℕ := 4
def ratio_trd : ℕ := 3
def total_students : ℕ := 240

def second_grade : ℕ := (ratio_snd * total_students) / (ratio_fst + ratio_snd + ratio_trd)

theorem num_second_grade_students_is_80 :
  second_grade = 80 := 
sorry

end num_second_grade_students_is_80_l148_148894


namespace tangent_line_intersecting_lines_l148_148243

variable (x y : ℝ)

-- Definition of the circle
def circle_C : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Definition of the point
def point_A : Prop := x = 1 ∧ y = 0

-- (I) Prove that if l is tangent to circle C and passes through A, l is 3x - 4y - 3 = 0
theorem tangent_line (l : ℝ → ℝ) (h : ∀ x, l x = k * (x - 1)) :
  (∀ {x y}, circle_C x y → 3 * x - 4 * y - 3 = 0) :=
by
  sorry

-- (II) Prove that the maximum area of triangle CPQ intersecting circle C is 2, and l's equations are y = 7x - 7 or y = x - 1
theorem intersecting_lines (k : ℝ) :
  (∃ x y, circle_C x y ∧ point_A x y) →
  (∃ k : ℝ, k = 7 ∨ k = 1) :=
by
  sorry

end tangent_line_intersecting_lines_l148_148243


namespace y_share_per_rupee_of_x_l148_148963

theorem y_share_per_rupee_of_x (share_y : ℝ) (total_amount : ℝ) (z_per_x : ℝ) (y_per_x : ℝ) 
  (h1 : share_y = 54) 
  (h2 : total_amount = 210) 
  (h3 : z_per_x = 0.30) 
  (h4 : share_y = y_per_x * (total_amount / (1 + y_per_x + z_per_x))) : 
  y_per_x = 0.45 :=
sorry

end y_share_per_rupee_of_x_l148_148963


namespace remainder_when_divided_by_9_l148_148071

open Nat

theorem remainder_when_divided_by_9 (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 :=
by
  sorry

end remainder_when_divided_by_9_l148_148071


namespace years_required_l148_148795

def num_stadiums := 30
def avg_cost_per_stadium := 900
def annual_savings := 1500
def total_cost := num_stadiums * avg_cost_per_stadium

theorem years_required : total_cost / annual_savings = 18 :=
by
  sorry

end years_required_l148_148795


namespace six_digit_numbers_with_zero_l148_148669

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l148_148669


namespace probability_of_collecting_both_types_correct_l148_148950

-- Defining the problem space
def toy_box := {1, 2} -- Representing two types of toys as 1 and 2

noncomputable def probability_of_collecting_both_types : ℚ :=
  let outcomes : list (list ℕ) := list.replicate 3 toy_box.to_list.choice in -- All possible outcomes of buying 3 blind boxes
  let favorable : list (list ℕ) := outcomes.filter (λ x, (1 ∈ x) ∧ (2 ∈ x)) in -- Outcomes where both types are collected
  (favorable.length.to_rat / outcomes.length.to_rat)

-- Statement of the theorem
theorem probability_of_collecting_both_types_correct :
  probability_of_collecting_both_types = 3 / 4 :=
by sorry

-- This is to ensure the program will compile.
def main : IO Unit :=
  IO.println s!"Theorem: {probability_of_collecting_both_types_correct}"

end probability_of_collecting_both_types_correct_l148_148950


namespace find_c_in_triangle_l148_148888

theorem find_c_in_triangle
  (A : Real) (a b S : Real) (c : Real)
  (hA : A = 60) 
  (ha : a = 6 * Real.sqrt 3)
  (hb : b = 12)
  (hS : S = 18 * Real.sqrt 3) :
  c = 6 := by
  sorry

end find_c_in_triangle_l148_148888


namespace fish_remaining_correct_l148_148643

def guppies := 225
def angelfish := 175
def tiger_sharks := 200
def oscar_fish := 140
def discus_fish := 120

def guppies_sold := 3/5 * guppies
def angelfish_sold := 3/7 * angelfish
def tiger_sharks_sold := 1/4 * tiger_sharks
def oscar_fish_sold := 1/2 * oscar_fish
def discus_fish_sold := 2/3 * discus_fish

def guppies_remaining := guppies - guppies_sold
def angelfish_remaining := angelfish - angelfish_sold
def tiger_sharks_remaining := tiger_sharks - tiger_sharks_sold
def oscar_fish_remaining := oscar_fish - oscar_fish_sold
def discus_fish_remaining := discus_fish - discus_fish_sold

def total_remaining_fish := guppies_remaining + angelfish_remaining + tiger_sharks_remaining + oscar_fish_remaining + discus_fish_remaining

theorem fish_remaining_correct : total_remaining_fish = 450 := 
by 
  -- insert the necessary steps of the proof here
  sorry

end fish_remaining_correct_l148_148643


namespace find_wheel_diameter_l148_148835

noncomputable def wheel_diameter (revolutions distance : ℝ) (π_approx : ℝ) : ℝ := 
  distance / (π_approx * revolutions)

theorem find_wheel_diameter : wheel_diameter 47.04276615104641 4136 3.14159 = 27.99 :=
by
  sorry

end find_wheel_diameter_l148_148835


namespace trevor_brother_age_l148_148322

theorem trevor_brother_age :
  ∃ B : ℕ, Trevor_current_age = 11 ∧
           Trevor_future_age = 24 ∧
           Brother_future_age = 3 * Trevor_current_age ∧
           B = Brother_future_age - (Trevor_future_age - Trevor_current_age) :=
sorry

end trevor_brother_age_l148_148322


namespace six_digit_numbers_with_zero_l148_148709

theorem six_digit_numbers_with_zero :
  let total := 9 * 10^5 in
  let no_zero := 9^6 in
  let with_zero := total - no_zero in
  with_zero = 368559 :=
by
  let total := 9 * 10^5
  let no_zero := 9^6
  let with_zero := total - no_zero
  have h1 : total = 900000 := by sorry
  have h2 : no_zero = 531441 := by sorry
  have h3 : with_zero = 368559 := by sorry
  exact h3

end six_digit_numbers_with_zero_l148_148709


namespace six_digit_numbers_with_zero_l148_148701

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148701


namespace factorize_x_squared_minus_one_l148_148054

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l148_148054


namespace six_digit_numbers_with_zero_l148_148736

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148736


namespace smallest_prime_less_than_square_l148_148180

theorem smallest_prime_less_than_square : ∃ p n : ℕ, Prime p ∧ p = n^2 - 20 ∧ p = 5 :=
by 
  sorry

end smallest_prime_less_than_square_l148_148180


namespace unknown_number_is_three_or_twenty_seven_l148_148763

theorem unknown_number_is_three_or_twenty_seven
    (x y : ℝ)
    (h1 : y - 3 = x - y)
    (h2 : (y - 6) / 3 = x / (y - 6)) :
    x = 3 ∨ x = 27 :=
by
  sorry

end unknown_number_is_three_or_twenty_seven_l148_148763


namespace new_average_age_l148_148443

theorem new_average_age (n_students : ℕ) (average_student_age : ℕ) (teacher_age : ℕ)
  (h_students : n_students = 50)
  (h_average_student_age : average_student_age = 14)
  (h_teacher_age : teacher_age = 65) :
  (n_students * average_student_age + teacher_age) / (n_students + 1) = 15 :=
by
  sorry

end new_average_age_l148_148443


namespace jims_investment_l148_148187

theorem jims_investment (total_investment : ℝ) (john_ratio : ℝ) (james_ratio : ℝ) (jim_ratio : ℝ) 
                        (h_total_investment : total_investment = 80000)
                        (h_ratio_john : john_ratio = 4)
                        (h_ratio_james : james_ratio = 7)
                        (h_ratio_jim : jim_ratio = 9) : 
    jim_ratio * (total_investment / (john_ratio + james_ratio + jim_ratio)) = 36000 :=
by 
  sorry

end jims_investment_l148_148187


namespace eugene_total_payment_l148_148272

-- Define the initial costs of items
def cost_tshirt := 20
def cost_pants := 80
def cost_shoes := 150

-- Define the quantities
def quantity_tshirt := 4
def quantity_pants := 3
def quantity_shoes := 2

-- Define the discount rate
def discount_rate := 0.10

-- Define the total pre-discount cost
def pre_discount_cost :=
  (cost_tshirt * quantity_tshirt) +
  (cost_pants * quantity_pants) +
  (cost_shoes * quantity_shoes)

-- Define the discount amount
def discount_amount := discount_rate * pre_discount_cost

-- Define the post-discount cost
def post_discount_cost := pre_discount_cost - discount_amount

-- Theorem statement
theorem eugene_total_payment : post_discount_cost = 558 := by
  sorry

end eugene_total_payment_l148_148272


namespace correct_statements_B_and_C_l148_148402

variable {a b c : ℝ}

-- Definitions from the conditions
def conditionB (a b c : ℝ) : Prop := a > b ∧ b > 0 ∧ c < 0
def conclusionB (a b c : ℝ) : Prop := c / a^2 > c / b^2

def conditionC (a b c : ℝ) : Prop := c > a ∧ a > b ∧ b > 0
def conclusionC (a b c : ℝ) : Prop := a / (c - a) > b / (c - b)

theorem correct_statements_B_and_C (a b c : ℝ) : 
  (conditionB a b c → conclusionB a b c) ∧ 
  (conditionC a b c → conclusionC a b c) :=
by
  sorry

end correct_statements_B_and_C_l148_148402


namespace ice_cubes_per_tray_l148_148359

theorem ice_cubes_per_tray (total_ice_cubes : ℕ) (number_of_trays : ℕ) (h1 : total_ice_cubes = 72) (h2 : number_of_trays = 8) : 
  total_ice_cubes / number_of_trays = 9 :=
by
  sorry

end ice_cubes_per_tray_l148_148359


namespace value_of_a_plus_one_l148_148878

theorem value_of_a_plus_one (a : ℤ) (h : |a| = 3) : a + 1 = 4 ∨ a + 1 = -2 :=
by
  sorry

end value_of_a_plus_one_l148_148878


namespace six_digit_numbers_with_at_least_one_zero_l148_148663

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l148_148663


namespace tan_sub_theta_cos_double_theta_l148_148511

variables (θ : ℝ)

-- Condition: given tan θ = 2
axiom tan_theta_eq_two : Real.tan θ = 2

-- Proof problem 1: Prove tan (π/4 - θ) = -1/3
theorem tan_sub_theta (h : Real.tan θ = 2) : Real.tan (Real.pi / 4 - θ) = -1/3 :=
by sorry

-- Proof problem 2: Prove cos 2θ = -3/5
theorem cos_double_theta (h : Real.tan θ = 2) : Real.cos (2 * θ) = -3/5 :=
by sorry

end tan_sub_theta_cos_double_theta_l148_148511


namespace ajax_weight_after_two_weeks_l148_148838

/-- Initial weight of Ajax in kilograms. -/
def initial_weight_kg : ℝ := 80

/-- Conversion factor from kilograms to pounds. -/
def kg_to_pounds : ℝ := 2.2

/-- Weight lost per hour of each exercise type. -/
def high_intensity_loss_per_hour : ℝ := 4
def moderate_intensity_loss_per_hour : ℝ := 2.5
def low_intensity_loss_per_hour : ℝ := 1.5

/-- Ajax's weekly exercise routine. -/
def weekly_high_intensity_hours : ℝ := 1 * 3 + 1.5 * 1
def weekly_moderate_intensity_hours : ℝ := 0.5 * 5
def weekly_low_intensity_hours : ℝ := 1 * 2 + 0.5 * 1

/-- Calculate the total weight loss in pounds per week. -/
def total_weekly_weight_loss_pounds : ℝ :=
  weekly_high_intensity_hours * high_intensity_loss_per_hour +
  weekly_moderate_intensity_hours * moderate_intensity_loss_per_hour +
  weekly_low_intensity_hours * low_intensity_loss_per_hour

/-- Calculate the total weight loss in pounds for two weeks. -/
def total_weight_loss_pounds_for_two_weeks : ℝ :=
  total_weekly_weight_loss_pounds * 2

/-- Calculate Ajax's initial weight in pounds. -/
def initial_weight_pounds : ℝ :=
  initial_weight_kg * kg_to_pounds

/-- Calculate Ajax's new weight after two weeks. -/
def new_weight_pounds : ℝ :=
  initial_weight_pounds - total_weight_loss_pounds_for_two_weeks

/-- Prove that Ajax's new weight in pounds is 120 after following the workout schedule for two weeks. -/
theorem ajax_weight_after_two_weeks :
  new_weight_pounds = 120 :=
by
  sorry

end ajax_weight_after_two_weeks_l148_148838


namespace geometric_sequence_sum_is_five_eighths_l148_148239

noncomputable def geometric_sequence_sum (a₁ : ℝ) (q : ℝ) : ℝ :=
  if q = 1 then 4 * a₁ else a₁ * (1 - q^4) / (1 - q)

theorem geometric_sequence_sum_is_five_eighths
  (a₁ q : ℝ)
  (h₀ : q ≠ 1)
  (h₁ : a₁ * (a₁ * q) * (a₁ * q^2) = -1 / 8)
  (h₂ : 2 * (a₁ * q^2) = a₁ * q + a₁ * q^2) :
  geometric_sequence_sum a₁ q = 5 / 8 := by
sorry

end geometric_sequence_sum_is_five_eighths_l148_148239


namespace length_of_la_l148_148001

variables {A b c l_a: ℝ}
variables (S_ABC S_ACA' S_ABA': ℝ)

axiom area_of_ABC: S_ABC = (1 / 2) * b * c * Real.sin A
axiom area_of_ACA: S_ACA' = (1 / 2) * b * l_a * Real.sin (A / 2)
axiom area_of_ABA: S_ABA' = (1 / 2) * c * l_a * Real.sin (A / 2)
axiom sin_double_angle: Real.sin A = 2 * Real.sin (A / 2) * Real.cos (A / 2)

theorem length_of_la :
  l_a = (2 * b * c * Real.cos (A / 2)) / (b + c) :=
sorry

end length_of_la_l148_148001


namespace spherical_cap_surface_area_l148_148932

theorem spherical_cap_surface_area (V : ℝ) (h : ℝ) (A : ℝ) (r : ℝ) 
  (volume_eq : V = (4 / 3) * π * r^3) 
  (cap_height : h = 2) 
  (sphere_volume : V = 288 * π) 
  (cap_surface_area : A = 2 * π * r * h) : 
  A = 24 * π := 
sorry

end spherical_cap_surface_area_l148_148932


namespace S_range_l148_148077

theorem S_range (x : ℝ) (y : ℝ) (S : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : 0 ≤ x) 
  (h3 : x ≤ 1 / 2) 
  (h4 : S = x * y) : 
  -1 / 8 ≤ S ∧ S ≤ 0 := 
sorry

end S_range_l148_148077


namespace part_one_part_two_l148_148651

noncomputable def f (x a: ℝ) : ℝ := abs (x - 1) + abs (x + a)
noncomputable def g (a : ℝ) : ℝ := a^2 - a - 2

theorem part_one (x : ℝ) : f x 3 > g 3 + 2 ↔ x < -4 ∨ x > 2 := by
  sorry

theorem part_two (a : ℝ) :
  (∀ x : ℝ, -a ≤ x ∧ x ≤ 1 → f x a ≤ g a) ↔ a ≥ 3 := by
  sorry

end part_one_part_two_l148_148651


namespace cubic_polynomial_greater_than_zero_l148_148862

theorem cubic_polynomial_greater_than_zero (x : ℝ) : x^3 + 3*x^2 + x - 5 > 0 → x > 1 :=
sorry

end cubic_polynomial_greater_than_zero_l148_148862


namespace book_price_increase_l148_148169

theorem book_price_increase (P : ℝ) (x : ℝ) :
  (P * (1 + x / 100)^2 = P * 1.3225) → x = 15 :=
by
  sorry

end book_price_increase_l148_148169


namespace product_of_repeating_decimal_l148_148490

noncomputable def t : ℚ := 152 / 333

theorem product_of_repeating_decimal :
  8 * t = 1216 / 333 :=
by {
  -- Placeholder for proof.
  sorry
}

end product_of_repeating_decimal_l148_148490


namespace Jennifer_apples_l148_148902

-- Define the conditions
def initial_apples : ℕ := 7
def found_apples : ℕ := 74

-- The theorem to prove
theorem Jennifer_apples : initial_apples + found_apples = 81 :=
by
  -- proof goes here, but we use sorry to skip the proof step
  sorry

end Jennifer_apples_l148_148902


namespace quadratic_has_real_solutions_iff_l148_148034

theorem quadratic_has_real_solutions_iff (m : ℝ) :
  ∃ x y : ℝ, (y = m * x + 3) ∧ (y = (3 * m - 2) * x ^ 2 + 5) ↔ 
  (m ≤ 12 - 8 * Real.sqrt 2) ∨ (m ≥ 12 + 8 * Real.sqrt 2) :=
by
  sorry

end quadratic_has_real_solutions_iff_l148_148034


namespace brick_length_proof_l148_148343

-- Definitions based on conditions
def courtyard_length_m : ℝ := 18
def courtyard_width_m : ℝ := 16
def brick_width_cm : ℝ := 10
def total_bricks : ℝ := 14400

-- Conversion factors
def sqm_to_sqcm (area_sqm : ℝ) : ℝ := area_sqm * 10000
def courtyard_area_cm2 : ℝ := sqm_to_sqcm (courtyard_length_m * courtyard_width_m)

-- The proof statement
theorem brick_length_proof :
  (∀ (L : ℝ), courtyard_area_cm2 = total_bricks * (L * brick_width_cm)) → 
  (∃ (L : ℝ), L = 20) :=
by
  intro h
  sorry

end brick_length_proof_l148_148343


namespace mixed_oil_rate_l148_148255

/-- Given quantities and prices of three types of oils, any combination
that satisfies the volume and price conditions will achieve a final mixture rate of Rs. 65 per litre. -/
theorem mixed_oil_rate (x y z : ℝ) : 
  12.5 * 55 + 7.75 * 70 + 3.25 * 82 = 1496.5 ∧ 12.5 + 7.75 + 3.25 = 23.5 →
  x + y + z = 23.5 ∧ 55 * x + 70 * y + 82 * z = 65 * 23.5 →
  true :=
by
  intros h1 h2
  sorry

end mixed_oil_rate_l148_148255


namespace find_volume_from_vessel_c_l148_148308

noncomputable def concentration_vessel_a : ℝ := 0.45
noncomputable def concentration_vessel_b : ℝ := 0.30
noncomputable def concentration_vessel_c : ℝ := 0.10
noncomputable def volume_vessel_a : ℝ := 4
noncomputable def volume_vessel_b : ℝ := 5
noncomputable def resultant_concentration : ℝ := 0.26

theorem find_volume_from_vessel_c (x : ℝ) : 
    concentration_vessel_a * volume_vessel_a + concentration_vessel_b * volume_vessel_b + concentration_vessel_c * x = 
    resultant_concentration * (volume_vessel_a + volume_vessel_b + x) → 
    x = 6 :=
by
  sorry

end find_volume_from_vessel_c_l148_148308


namespace factorize_x_squared_minus_1_l148_148040

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l148_148040


namespace fraction_of_25_exists_l148_148334

theorem fraction_of_25_exists :
  ∃ x : ℚ, 0.60 * 40 = x * 25 + 4 ∧ x = 4 / 5 :=
by
  simp
  sorry

end fraction_of_25_exists_l148_148334


namespace grinder_price_l148_148121

variable (G : ℝ) (PurchasedMobile : ℝ) (SoldMobile : ℝ) (overallProfit : ℝ)

theorem grinder_price (h1 : PurchasedMobile = 10000)
                      (h2 : SoldMobile = 11000)
                      (h3 : overallProfit = 400)
                      (h4 : 0.96 * G + SoldMobile = G + PurchasedMobile + overallProfit) :
                      G = 15000 := by
  sorry

end grinder_price_l148_148121


namespace square_of_hypotenuse_product_eq_160_l148_148803

noncomputable def square_of_product_of_hypotenuses (x y : ℝ) (h1 h2 : ℝ) : ℝ :=
  (h1 * h2) ^ 2

theorem square_of_hypotenuse_product_eq_160 :
  ∀ (x y h1 h2 : ℝ),
    (1 / 2) * x * (2 * y) = 4 →
    (1 / 2) * x * y = 8 →
    x^2 + (2 * y)^2 = h1^2 →
    x^2 + y^2 = h2^2 →
    square_of_product_of_hypotenuses x y h1 h2 = 160 :=
by
  intros x y h1 h2 area1 area2 pythagorean1 pythagorean2
  -- The detailed proof steps would go here
  sorry

end square_of_hypotenuse_product_eq_160_l148_148803


namespace cubic_conversion_l148_148744

theorem cubic_conversion (h : 1 = 100) : 1 = 1000000 :=
by
  sorry

end cubic_conversion_l148_148744


namespace factorize_x_squared_minus_1_l148_148042

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l148_148042


namespace shaded_region_area_is_correct_l148_148274

noncomputable def area_of_shaded_region : ℝ :=
  let R := 6 -- radius of the larger circle
  let r := R / 2 -- radius of each smaller circle
  let area_large_circle := Real.pi * R^2
  let area_two_small_circles := 2 * Real.pi * r^2
  area_large_circle - area_two_small_circles

theorem shaded_region_area_is_correct :
  area_of_shaded_region = 18 * Real.pi :=
sorry

end shaded_region_area_is_correct_l148_148274


namespace reduced_price_of_oil_l148_148943

theorem reduced_price_of_oil (P R : ℝ) (h1: R = 0.75 * P) (h2: 600 / (0.75 * P) = 600 / P + 5) :
  R = 30 :=
by
  sorry

end reduced_price_of_oil_l148_148943


namespace length_of_BC_is_7_l148_148884

noncomputable def triangle_length_BC (a b c : ℝ) (A : ℝ) (S : ℝ) (P : ℝ) : Prop :=
  (P = a + b + c) ∧ (P = 20) ∧ (S = 1 / 2 * b * c * Real.sin A) ∧ (S = 10 * Real.sqrt 3) ∧ (A = Real.pi / 3) ∧ (b * c = 20)

theorem length_of_BC_is_7 : ∃ a b c, triangle_length_BC a b c (Real.pi / 3) (10 * Real.sqrt 3) 20 ∧ a = 7 := 
by
  -- proof omitted
  sorry

end length_of_BC_is_7_l148_148884


namespace calculate_expression_l148_148872

theorem calculate_expression (x : ℤ) (h : x^2 = 1681) : (x + 3) * (x - 3) = 1672 :=
by
  sorry

end calculate_expression_l148_148872


namespace cost_price_l148_148772

theorem cost_price (SP MP CP : ℝ) (discount_rate : ℝ) 
  (h1 : MP = CP * 1.15)
  (h2 : SP = MP * (1 - discount_rate))
  (h3 : SP = 459)
  (h4 : discount_rate = 0.2608695652173913) : CP = 540 :=
by
  -- We use the hints given as conditions to derive the statement
  sorry

end cost_price_l148_148772


namespace sum_fractions_lt_one_l148_148997

theorem sum_fractions_lt_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  0 < (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)) ∧
  (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)) < 1 :=
by
  sorry

end sum_fractions_lt_one_l148_148997


namespace bike_price_l148_148120

theorem bike_price (x : ℝ) (h : 0.20 * x = 240) : x = 1200 :=
by
  sorry

end bike_price_l148_148120


namespace adela_numbers_l148_148965

theorem adela_numbers (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a - b)^2 = a^2 - b^2 - 4038) :
  (a = 2020 ∧ b = 1) ∨ (a = 2020 ∧ b = 2019) ∨ (a = 676 ∧ b = 3) ∨ (a = 676 ∧ b = 673) :=
sorry

end adela_numbers_l148_148965


namespace missing_number_l148_148532

theorem missing_number (m x : ℕ) (h : 744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + m + x = 750 * 10)  
  (hx : x = 755) : m = 805 := by 
  sorry

end missing_number_l148_148532


namespace equalities_imply_forth_l148_148430

variables {a b c d e f g h S1 S2 S3 O2 O3 : ℕ}

def S1_def := S1 = a + b + c
def S2_def := S2 = d + e + f
def S3_def := S3 = b + c + g + h - d
def O2_def := O2 = b + e + g
def O3_def := O3 = c + f + h

theorem equalities_imply_forth (h1 : S1 = S2) (h2 : S1 = S3) (h3 : S1 = O2) : S1 = O3 :=
  by sorry

end equalities_imply_forth_l148_148430


namespace number_of_buses_l148_148294

theorem number_of_buses (total_students : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) (buses : ℕ)
  (h1 : total_students = 375)
  (h2 : students_per_bus = 53)
  (h3 : students_in_cars = 4)
  (h4 : buses = (total_students - students_in_cars + students_per_bus - 1) / students_per_bus) :
  buses = 8 := by
  -- We will demonstrate that the number of buses indeed equals 8 under the given conditions.
  sorry

end number_of_buses_l148_148294


namespace sum_of_cubes_is_zero_l148_148412

theorem sum_of_cubes_is_zero 
  (a b : ℝ) 
  (h1 : a + b = 0) 
  (h2 : a * b = -1) : 
  a^3 + b^3 = 0 := by
  sorry

end sum_of_cubes_is_zero_l148_148412


namespace volume_Q3_l148_148400

def Q0 : ℚ := 8
def delta : ℚ := (1 / 3) ^ 3
def ratio : ℚ := 6 / 27

def Q (i : ℕ) : ℚ :=
  match i with
  | 0 => Q0
  | 1 => Q0 + 4 * delta
  | n + 1 => Q n + delta * (ratio ^ n)

theorem volume_Q3 : Q 3 = 5972 / 729 := 
by
  sorry

end volume_Q3_l148_148400


namespace square_area_from_triangle_perimeter_l148_148482

noncomputable def perimeter_triangle (a b c : ℝ) : ℝ := a + b + c

noncomputable def side_length_square (perimeter : ℝ) : ℝ := perimeter / 4

noncomputable def area_square (side_length : ℝ) : ℝ := side_length * side_length

theorem square_area_from_triangle_perimeter 
  (a b c : ℝ) 
  (h₁ : a = 5.5) 
  (h₂ : b = 7.5) 
  (h₃ : c = 11) 
  (h₄ : perimeter_triangle a b c = 24) 
  : area_square (side_length_square (perimeter_triangle a b c)) = 36 := 
by 
  simp [perimeter_triangle, side_length_square, area_square, h₁, h₂, h₃, h₄]
  sorry

end square_area_from_triangle_perimeter_l148_148482


namespace find_numbers_l148_148570

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l148_148570


namespace largest_divisor_of_n4_minus_n2_l148_148790

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 12 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_n4_minus_n2_l148_148790


namespace smallest_m_l148_148214

-- Definitions of lengths and properties of the pieces
variable {lengths : Fin 21 → ℝ} 
variable (h_all_pos : ∀ i, lengths i > 0)
variable (h_total_length : (Finset.univ : Finset (Fin 21)).sum lengths = 21)
variable (h_max_factor : ∀ i j, max (lengths i) (lengths j) ≤ 3 * min (lengths i) (lengths j))

-- Proof statement
theorem smallest_m (m : ℝ) (hm : ∀ i j, max (lengths i) (lengths j) ≤ m * min (lengths i) (lengths j)) : 
  m ≥ 1 := 
sorry

end smallest_m_l148_148214


namespace johns_average_speed_l148_148122

-- Conditions
def biking_time_minutes : ℝ := 45
def biking_speed_mph : ℝ := 20
def walking_time_minutes : ℝ := 120
def walking_speed_mph : ℝ := 3

-- Proof statement
theorem johns_average_speed :
  let biking_time_hours := biking_time_minutes / 60
  let biking_distance := biking_speed_mph * biking_time_hours
  let walking_time_hours := walking_time_minutes / 60
  let walking_distance := walking_speed_mph * walking_time_hours
  let total_distance := biking_distance + walking_distance
  let total_time := biking_time_hours + walking_time_hours
  let average_speed := total_distance / total_time
  average_speed = 7.64 :=
by
  sorry

end johns_average_speed_l148_148122


namespace six_digit_numbers_with_zero_l148_148699

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148699


namespace determine_position_correct_l148_148941

def determine_position (option : String) : Prop :=
  option = "East longitude 120°, North latitude 30°"

theorem determine_position_correct :
  determine_position "East longitude 120°, North latitude 30°" :=
by
  sorry

end determine_position_correct_l148_148941


namespace Fred_hourly_rate_l148_148396

-- Define the conditions
def hours_worked : ℝ := 8
def total_earned : ℝ := 100

-- Assert the proof goal
theorem Fred_hourly_rate : total_earned / hours_worked = 12.5 :=
by
  sorry

end Fred_hourly_rate_l148_148396


namespace concurrency_of_AP_BQ_CR_l148_148023

theorem concurrency_of_AP_BQ_CR (
  {A B C D E F P Q R G: Type*}
  [triangle ABC]
  [acute_angled_triangle ABC]
  [altitudes A D, B E, C F]
  [perpendicular A P E F, B Q F D, C R D E]
  [feet A P E F, B Q F D, C R D E]):
  concurrent AP BQ CR :=
sorry

end concurrency_of_AP_BQ_CR_l148_148023


namespace balance_balls_l148_148144

theorem balance_balls (R O G B : ℝ) (h₁ : 4 * R = 8 * G) (h₂ : 3 * O = 6 * G) (h₃ : 8 * G = 6 * B) :
  3 * R + 2 * O + 4 * B = (46 / 3) * G :=
by
  -- Using the given conditions to derive intermediate results (included in the detailed proof, not part of the statement)
  sorry

end balance_balls_l148_148144


namespace factorize_x_squared_minus_1_l148_148041

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l148_148041


namespace unique_solution_l148_148379

theorem unique_solution (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a * b - a - b = 1) : (a, b) = (3, 2) :=
by
  sorry

end unique_solution_l148_148379


namespace max_popsicles_with_10_dollars_l148_148776

theorem max_popsicles_with_10_dollars :
  (∃ (single_popsicle_cost : ℕ) (four_popsicle_box_cost : ℕ) (six_popsicle_box_cost : ℕ) (budget : ℕ),
    single_popsicle_cost = 1 ∧
    four_popsicle_box_cost = 3 ∧
    six_popsicle_box_cost = 4 ∧
    budget = 10 ∧
    ∃ (max_popsicles : ℕ),
      max_popsicles = 14 ∧
      ∀ (popsicles : ℕ),
        popsicles ≤ 14 →
        ∃ (x y z : ℕ),
          popsicles = x + 4*y + 6*z ∧
          x * single_popsicle_cost + y * four_popsicle_box_cost + z * six_popsicle_box_cost ≤ budget
  ) :=
sorry

end max_popsicles_with_10_dollars_l148_148776


namespace find_m_if_even_l148_148407

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def my_function (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem find_m_if_even (m : ℝ) :
  is_even_function (my_function m) → m = 2 := 
by
  sorry

end find_m_if_even_l148_148407


namespace product_eq_1519000000_div_6561_l148_148236

-- Given conditions
def P (X : ℚ) : ℚ := X - 5
def Q (X : ℚ) : ℚ := X + 5
def R (X : ℚ) : ℚ := X / 2
def S (X : ℚ) : ℚ := 2 * X

theorem product_eq_1519000000_div_6561 
  (X : ℚ) 
  (h : (P X) + (Q X) + (R X) + (S X) = 100) :
  (P X) * (Q X) * (R X) * (S X) = 1519000000 / 6561 := 
by sorry

end product_eq_1519000000_div_6561_l148_148236


namespace smallest_prime_20_less_than_square_l148_148179

open Nat

theorem smallest_prime_20_less_than_square : ∃ (p : ℕ), Prime p ∧ (∃ (n : ℕ), p = n^2 - 20) ∧ p = 5 := by
  sorry

end smallest_prime_20_less_than_square_l148_148179


namespace total_notebooks_l148_148605

-- Define the problem conditions
theorem total_notebooks (x : ℕ) (hx : x*x + 20 = (x+1)*(x+1) - 9) : x*x + 20 = 216 :=
by
  have h1 : x*x + 20 = 216 := sorry
  exact h1

end total_notebooks_l148_148605


namespace possible_values_y_l148_148428

theorem possible_values_y (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y : ℝ, (y = 0 ∨ y = 41 ∨ y = 144) ∧ y = (x - 3)^2 * (x + 4) / (2 * x - 5) :=
by sorry

end possible_values_y_l148_148428


namespace choir_members_max_l148_148961

-- Define the conditions and the proof for the equivalent problem.
theorem choir_members_max (c s y : ℕ) (h1 : c < 120) (h2 : s * y + 3 = c) (h3 : (s - 1) * (y + 2) = c) : c = 120 := by
  sorry

end choir_members_max_l148_148961


namespace six_digit_numbers_with_zero_count_l148_148675

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l148_148675


namespace small_denominator_difference_l148_148551

theorem small_denominator_difference :
  ∃ (p q : ℕ), 0 < p ∧ 0 < q ∧
               (5 : ℚ) / 9 < (p : ℚ) / q ∧
               (p : ℚ) / q < 4 / 7 ∧
               (∀ r, 0 < r → (5 : ℚ) / 9 < (p : ℚ) / r → (p : ℚ) / r < 4 / 7 → q ≤ r) ∧
               q - p = 7 := 
  by
  sorry

end small_denominator_difference_l148_148551


namespace palindromic_condition_l148_148832

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem palindromic_condition (m n : ℕ) :
  is_palindrome (2^n + 2^m + 1) ↔ (m ≤ 9 ∨ n ≤ 9) :=
sorry

end palindromic_condition_l148_148832


namespace simplify_and_evaluate_l148_148780

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) : 
  (1 - 1 / (a + 1)) * ((a^2 + 2 * a + 1) / a) = Real.sqrt 2 := 
by {
  sorry
}

end simplify_and_evaluate_l148_148780


namespace product_of_good_numbers_does_not_imply_sum_digits_property_l148_148288

-- Define what it means for a number to be "good".
def is_good (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

-- Define the sum of the digits of a number
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem statement
theorem product_of_good_numbers_does_not_imply_sum_digits_property :
  ∀ (A B : ℕ), is_good A → is_good B → is_good (A * B) →
  ¬ (sum_digits (A * B) = sum_digits A * sum_digits B) :=
by
  intros A B hA hB hAB
  -- The detailed proof is not provided here, hence we use sorry to skip it.
  sorry

end product_of_good_numbers_does_not_imply_sum_digits_property_l148_148288


namespace fraction_simplification_l148_148496

theorem fraction_simplification : (8 : ℝ) / (4 * 25) = 0.08 :=
by
  sorry

end fraction_simplification_l148_148496


namespace combine_like_terms_l148_148641

theorem combine_like_terms (x y : ℝ) : 3 * x * y - 6 * x * y + (-2) * x * y = -5 * x * y := by
  sorry

end combine_like_terms_l148_148641


namespace tan_frac_a_pi_six_eq_sqrt_three_l148_148533

theorem tan_frac_a_pi_six_eq_sqrt_three (a : ℝ) (h : (a, 9) ∈ { p : ℝ × ℝ | p.2 = 3 ^ p.1 }) : 
  Real.tan (a * Real.pi / 6) = Real.sqrt 3 := 
by
  sorry

end tan_frac_a_pi_six_eq_sqrt_three_l148_148533


namespace find_a_l148_148935

theorem find_a (a b c d : ℕ) (h1 : 2 * a + 2 = b) (h2 : 2 * b + 2 = c) (h3 : 2 * c + 2 = d) (h4 : 2 * d + 2 = 62) : a = 2 :=
by
  sorry

end find_a_l148_148935


namespace count_valid_abcd_is_zero_l148_148526

def valid_digits := {a // 1 ≤ a ∧ a ≤ 9} 
def zero_to_nine := {n // 0 ≤ n ∧ n ≤ 9}

noncomputable def increasing_arithmetic_sequence_with_difference_5 (a b c d : ℕ) : Prop := 
  10 * a + b + 5 = 10 * b + c ∧ 
  10 * b + c + 5 = 10 * c + d

theorem count_valid_abcd_is_zero :
  ∀ (a : valid_digits) (b c d : zero_to_nine),
    ¬ increasing_arithmetic_sequence_with_difference_5 a.val b.val c.val d.val := 
sorry

end count_valid_abcd_is_zero_l148_148526


namespace black_ball_on_second_draw_given_white_ball_on_first_draw_l148_148757

def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 5
def total_balls : ℕ := num_white_balls + num_black_balls

def P_A : ℚ := num_white_balls / total_balls
def P_AB : ℚ := (num_white_balls * num_black_balls) / (total_balls * (total_balls - 1))
def P_B_given_A : ℚ := P_AB / P_A

theorem black_ball_on_second_draw_given_white_ball_on_first_draw : P_B_given_A = 5 / 8 :=
by
  sorry

end black_ball_on_second_draw_given_white_ball_on_first_draw_l148_148757


namespace thirteen_members_divisible_by_13_l148_148174

theorem thirteen_members_divisible_by_13 (B : ℕ) (hB : B < 10) : 
  (∃ B, (2000 + B * 100 + 34) % 13 = 0) ↔ B = 6 :=
by
  sorry

end thirteen_members_divisible_by_13_l148_148174


namespace cheetah_catches_deer_in_10_minutes_l148_148607

noncomputable def deer_speed : ℝ := 50 -- miles per hour
noncomputable def cheetah_speed : ℝ := 60 -- miles per hour
noncomputable def time_difference : ℝ := 2 / 60 -- 2 minutes converted to hours
noncomputable def distance_deer : ℝ := deer_speed * time_difference
noncomputable def speed_difference : ℝ := cheetah_speed - deer_speed
noncomputable def catch_up_time : ℝ := distance_deer / speed_difference

theorem cheetah_catches_deer_in_10_minutes :
  catch_up_time * 60 = 10 :=
by
  sorry

end cheetah_catches_deer_in_10_minutes_l148_148607


namespace reggie_marbles_bet_l148_148152

theorem reggie_marbles_bet 
  (initial_marbles : ℕ) (final_marbles : ℕ) (games_played : ℕ) (games_lost : ℕ) (bet_per_game : ℕ)
  (h_initial : initial_marbles = 100) 
  (h_final : final_marbles = 90) 
  (h_games : games_played = 9) 
  (h_losses : games_lost = 1) : 
  bet_per_game = 13 :=
by
  sorry

end reggie_marbles_bet_l148_148152


namespace find_decimal_decrease_l148_148931

noncomputable def tax_diminished_percentage (T C : ℝ) (X : ℝ) : Prop :=
  let new_tax := T * (1 - X / 100)
  let new_consumption := C * 1.15
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  new_revenue = original_revenue * 0.943

theorem find_decimal_decrease (T C : ℝ) (X : ℝ) :
  tax_diminished_percentage T C X → X = 18 := sorry

end find_decimal_decrease_l148_148931


namespace find_three_digit_number_l148_148562

theorem find_three_digit_number (P Q R : ℕ) 
  (h1 : P ≠ Q) 
  (h2 : P ≠ R) 
  (h3 : Q ≠ R) 
  (h4 : P < 7) 
  (h5 : Q < 7) 
  (h6 : R < 7)
  (h7 : P ≠ 0) 
  (h8 : Q ≠ 0) 
  (h9 : R ≠ 0) 
  (h10 : 7 * P + Q + R = 7 * R) 
  (h11 : (7 * P + Q) + (7 * Q + P) = 49 + 7 * R + R)
  : P * 100 + Q * 10 + R = 434 :=
sorry

end find_three_digit_number_l148_148562


namespace product_divisible_by_sum_l148_148928

theorem product_divisible_by_sum (m n : ℕ) (h : ∃ k : ℕ, m * n = k * (m + n)) : m + n ≤ Nat.gcd m n * Nat.gcd m n := by
  sorry

end product_divisible_by_sum_l148_148928


namespace electricity_price_increase_percentage_l148_148313

noncomputable def old_power_kW : ℝ := 0.8
noncomputable def additional_power_percent : ℝ := 50 / 100
noncomputable def old_price_per_kWh : ℝ := 0.12
noncomputable def cost_for_50_hours : ℝ := 9
noncomputable def total_hours : ℝ := 50
noncomputable def energy_consumed := old_power_kW * total_hours

theorem electricity_price_increase_percentage :
  ∃ P : ℝ, 
    (energy_consumed * P = cost_for_50_hours) ∧
    ((P - old_price_per_kWh) / old_price_per_kWh) * 100 = 87.5 :=
by
  sorry

end electricity_price_increase_percentage_l148_148313


namespace factorize_x_squared_minus_one_l148_148055

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l148_148055


namespace ratio_cookies_to_pie_l148_148193

def num_surveyed_students : ℕ := 800
def num_students_preferred_cookies : ℕ := 280
def num_students_preferred_pie : ℕ := 160

theorem ratio_cookies_to_pie : num_students_preferred_cookies / num_students_preferred_pie = 7 / 4 := by
  sorry

end ratio_cookies_to_pie_l148_148193


namespace find_t_l148_148873

def vector := (ℝ × ℝ)

def a : vector := (-3, 4)
def b : vector := (-1, 5)
def c : vector := (2, 3)

def parallel (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem find_t (t : ℝ) : 
  parallel (a.1 - c.1, a.2 - c.2) ((2 * t) + b.1, (3 * t) + b.2) ↔ t = -24 / 17 :=
by
  sorry

end find_t_l148_148873


namespace avg_children_in_families_with_children_l148_148503

noncomputable def avg_children_with_children (total_families : ℕ) (avg_children : ℝ) (childless_families : ℕ) : ℝ :=
  let total_children := total_families * avg_children
  let families_with_children := total_families - childless_families
  total_children / families_with_children

theorem avg_children_in_families_with_children :
  avg_children_with_children 15 3 3 = 3.8 := by
  sorry

end avg_children_in_families_with_children_l148_148503


namespace arithmetic_sequence_S9_l148_148992

noncomputable def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_S9 (a : ℕ → ℕ)
    (h1 : 2 * a 6 = 6 + a 7) :
    Sn a 9 = 54 := 
sorry

end arithmetic_sequence_S9_l148_148992


namespace find_base_l148_148115

def distinct_three_digit_numbers (b : ℕ) : ℕ :=
    (b - 2) * (b - 3) + (b - 1) * (b - 3) + (b - 1) * (b - 2)

theorem find_base (b : ℕ) (h : distinct_three_digit_numbers b = 144) : b = 9 :=
by 
  sorry

end find_base_l148_148115


namespace original_sequence_polynomial_of_degree_3_l148_148078

def is_polynomial_of_degree (u : ℕ → ℤ) (n : ℕ) :=
  ∃ a b c d : ℤ, u n = a * n^3 + b * n^2 + c * n + d

def fourth_difference_is_zero (u : ℕ → ℤ) :=
  ∀ n : ℕ, (u (n + 4) - 4 * u (n + 3) + 6 * u (n + 2) - 4 * u (n + 1) + u n) = 0

theorem original_sequence_polynomial_of_degree_3 (u : ℕ → ℤ)
  (h : fourth_difference_is_zero u) : 
  ∃ (a b c d : ℤ), ∀ n : ℕ, u n = a * n^3 + b * n^2 + c * n + d := sorry

end original_sequence_polynomial_of_degree_3_l148_148078


namespace solve_equation_in_integers_l148_148438

theorem solve_equation_in_integers (a b c : ℤ) (h : 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) :
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 1 ∧ ∃ t : ℤ, b = t ∧ c = -t) :=
sorry

end solve_equation_in_integers_l148_148438


namespace proof_f_value_l148_148991

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then 1 - x^2 else 2^x

theorem proof_f_value : f (1 / f (Real.log 6 / Real.log 2)) = 35 / 36 := by
  sorry

end proof_f_value_l148_148991


namespace simplify_expression_l148_148519

theorem simplify_expression (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c = 0) :
  |a - c| + |c - b| - |a - b| = 0 :=
by
  sorry

end simplify_expression_l148_148519


namespace plane_through_A_perpendicular_to_BC_l148_148810

theorem plane_through_A_perpendicular_to_BC :
  ∃ (a b c d : ℝ), a = 5 ∧ b = -1 ∧ c = 3 ∧ d = -19 ∧
  (∀ (x y z : ℝ), a * (x - 5) + b * (y - 3) + c * (z + 1) = 0 ↔ 5 * x - y + 3 * z - 19 = 0) :=
begin
  use [5, -1, 3, -19],
  split, { refl },
  split, { refl },
  split, { refl },
  split, { refl },
  intros x y z,
  split,
  { intro h,
    calc 5 * x - y + 3 * z - 19 = 5 * x + (-1) * y + 3 * z + -19 : by ring
    ... = 0 : by { rw ← h,
                   ring } },
  { intro h,
    calc 5 * (x - 5) + (-1) * (y - 3) + 3 * (z + 1) = 5 * x - 25 + (- y + 3) + 3 * z + 3 : by ring
    ... = 5 * x - y + 3 * z - 19 : by ring
    ... = 0 : by rw h }
end

end plane_through_A_perpendicular_to_BC_l148_148810


namespace six_digit_numbers_with_zero_l148_148682

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148682


namespace factorize_difference_of_squares_l148_148059

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l148_148059


namespace fermats_little_theorem_l148_148945

theorem fermats_little_theorem (n p : ℕ) [hp : Fact p.Prime] : p ∣ (n^p - n) :=
sorry

end fermats_little_theorem_l148_148945


namespace proof_for_y_l148_148975

theorem proof_for_y (x y : ℝ) (h1 : 3 * x^2 + 5 * x + 4 * y + 2 = 0) (h2 : 3 * x + y + 4 = 0) : 
  y^2 + 15 * y + 2 = 0 :=
sorry

end proof_for_y_l148_148975


namespace length_of_AG_l148_148273

/-- Given a right-angled triangle ABC with ∠A = 90°, AB = 3 cm, AC = 3√5 cm. 
    E is the midpoint of BC, AD is the altitude from A to BC, and G is the point
    where AD intersects the median from B to E, then the length of AG is 3√10 / 2 cm. -/
theorem length_of_AG {A B C D E G : Point}
  (h_triangle : IsRightTriangle A B C)
  (h_angle : ∠A = 90°)
  (h_AB : dist A B = 3)
  (h_AC : dist A C = 3 * Real.sqrt 5)
  (h_E : Midpoint B C E)
  (h_AD_perp_BC : Perpendicular A D B C)
  (h_G : MedianIntersection A D B E G) :
  dist A G = 3 * (Real.sqrt 10) / 2 :=
  sorry

end length_of_AG_l148_148273


namespace pairs_satisfy_ineq_l148_148013

theorem pairs_satisfy_ineq (x y : ℝ) :
  |Real.sin x - Real.sin y| + Real.sin x * Real.sin y ≤ 0 ↔
  ∃ n m : ℤ, x = n * Real.pi ∧ y = m * Real.pi := 
sorry

end pairs_satisfy_ineq_l148_148013


namespace sum_first_10_terms_arithmetic_sequence_l148_148131

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l148_148131


namespace total_cost_of_video_games_l148_148594

theorem total_cost_of_video_games :
  let cost_football_game := 14.02
  let cost_strategy_game := 9.46
  let cost_batman_game := 12.04
  let total_cost := cost_football_game + cost_strategy_game + cost_batman_game
  total_cost = 35.52 :=
by
  -- Proof goes here
  sorry

end total_cost_of_video_games_l148_148594


namespace plane_eq_of_point_and_parallel_l148_148856

theorem plane_eq_of_point_and_parallel
    (A B C D : ℤ)
    (P : ℤ × ℤ × ℤ)
    (x y z : ℤ)
    (hx : A = 3) (hy : B = -2) (hz : C = 4) (hP : P = (2, -3, 1))
    (h_parallel : ∀ x y z, A * x + B * y + C * z = 5):
    A * 2 + B * (-3) + C * 1 + D = 0 ∧ A > 0 ∧ Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 :=
by
  sorry

end plane_eq_of_point_and_parallel_l148_148856


namespace find_three_xsq_ysq_l148_148230

theorem find_three_xsq_ysq (x y : ℤ) (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 3*x^2*y^2 = 588 :=
sorry

end find_three_xsq_ysq_l148_148230


namespace shares_difference_l148_148628

theorem shares_difference (x : ℝ) (hp : ℝ) (hq : ℝ) (hr : ℝ)
  (hx : hp = 3 * x) (hqx : hq = 7 * x) (hrx : hr = 12 * x) 
  (hqr_diff : hr - hq = 3500) : (hq - hp = 2800) :=
by
  -- The proof would be done here, but the problem statement requires only the theorem statement
  sorry

end shares_difference_l148_148628


namespace bill_has_6_less_pieces_than_mary_l148_148843

-- Definitions based on the conditions
def total_candy : ℕ := 20
def candy_kate : ℕ := 4
def candy_robert : ℕ := candy_kate + 2
def candy_mary : ℕ := candy_robert + 2
def candy_bill : ℕ := candy_kate - 2

-- Statement of the theorem
theorem bill_has_6_less_pieces_than_mary :
  candy_mary - candy_bill = 6 :=
sorry

end bill_has_6_less_pieces_than_mary_l148_148843


namespace shoe_pairs_l148_148339

theorem shoe_pairs (shoes_total : ℕ) (prob_matching : ℝ) (n : ℕ) :
  shoes_total = 14 →
  prob_matching = 0.07692307692307693 →
  (2 * n = shoes_total) →
  n = 7 :=
by
  intros h1 h2 h3
  have h4 : shoes_total = 2 * n := by rw [h3]
  have h5 : nat.choose 14 2 = 14 * 13 / 2 := by norm_num
  have h6 : prob_matching = n / ((14 * 13) / 2) :=
    by rw [nat.choice, h5]
  done
  sorry

end shoe_pairs_l148_148339


namespace factorization_proof_l148_148038

def factorization_problem (x : ℝ) : Prop := (x^2 - 1)^2 - 6 * (x^2 - 1) + 9 = (x - 2)^2 * (x + 2)^2

theorem factorization_proof (x : ℝ) : factorization_problem x :=
by
  -- The proof is omitted.
  sorry

end factorization_proof_l148_148038


namespace prob_A_and_B_truth_is_0_48_l148_148531

-- Conditions: Define the probabilities
def prob_A_truth : ℝ := 0.8
def prob_B_truth : ℝ := 0.6

-- Target: Define the probability that both A and B tell the truth at the same time.
def prob_A_and_B_truth : ℝ := prob_A_truth * prob_B_truth

-- Statement: Prove that the probability that both A and B tell the truth at the same time is 0.48.
theorem prob_A_and_B_truth_is_0_48 : prob_A_and_B_truth = 0.48 := by
  sorry

end prob_A_and_B_truth_is_0_48_l148_148531


namespace find_a_l148_148777

theorem find_a (a : ℤ) (h : |a + 1| = 3) : a = 2 ∨ a = -4 :=
sorry

end find_a_l148_148777


namespace solve_equation_theorem_l148_148566

noncomputable def solve_equations (S P : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  ((x1, y1), (x2, y2))

theorem solve_equation_theorem (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ x * y = P) ↔ (∃ (x1 y1 x2 y2 : ℝ), 
    ((x, y) = (x1, y1) ∨ (x, y) = (x2, y2)) ∧
    solve_equations S P = ((x1, y1), (x2, y2))) :=
by
  sorry

end solve_equation_theorem_l148_148566


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l148_148717

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l148_148717


namespace range_of_prime_set_l148_148468

theorem range_of_prime_set (a : ℕ) (ha : Nat.Prime a) 
  (x : Set ℕ) (hx : x = {3, 11, 7, a, 17, 19})
  (y : ℕ) (hy : y = 3 * 11 * 7 * a * 17 * 19)
  (h_even : 11 * y % 2 = 0) : 
  (19 - 2 = 17) :=
by
  have h_prime: ∀ n ∈ x, Nat.Prime n := by
    intro n hn
    rw [hx] at hn
    simp at hn
    cases hn <;> norm_num at hn
  obtain ⟨p, hp⟩ := h_even
  sorry

end range_of_prime_set_l148_148468


namespace six_digit_numbers_with_at_least_one_zero_l148_148662

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l148_148662


namespace div_gcd_iff_div_ab_gcd_mul_l148_148281

variable (a b n c : ℕ)
variables (h₀ : a ≠ 0) (d : ℕ)
variable (hd : d = Nat.gcd a b)

theorem div_gcd_iff_div_ab : (n ∣ a ∧ n ∣ b) ↔ n ∣ d :=
by
  sorry

theorem gcd_mul (h₁ : c > 0) : Nat.gcd (a * c) (b * c) = c * Nat.gcd a b :=
by
  sorry

end div_gcd_iff_div_ab_gcd_mul_l148_148281


namespace car_rental_cost_l148_148096

variable (x : ℝ)

theorem car_rental_cost (h : 65 + 0.40 * 325 = x * 325) : x = 0.60 :=
by 
  sorry

end car_rental_cost_l148_148096


namespace common_difference_arith_seq_l148_148762

theorem common_difference_arith_seq (a : ℕ → ℝ) (d : ℝ)
    (h₀ : a 1 + a 5 = 10)
    (h₁ : a 4 = 7)
    (h₂ : ∀ n, a (n + 1) = a n + d) : 
    d = 2 := by
  sorry

end common_difference_arith_seq_l148_148762


namespace prod_ge_27_eq_iff_equality_l148_148906

variable (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
          (h4 : a + b + c + 2 = a * b * c)

theorem prod_ge_27 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + c + 2 = a * b * c) : (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
by sorry

theorem eq_iff_equality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + c + 2 = a * b * c) : 
  ((a + 1) * (b + 1) * (c + 1) = 27) ↔ (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end prod_ge_27_eq_iff_equality_l148_148906


namespace complex_number_location_in_plane_l148_148927

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem complex_number_location_in_plane :
  is_in_second_quadrant (-2) 5 :=
by
  sorry

end complex_number_location_in_plane_l148_148927


namespace sum_first_ten_terms_arithmetic_l148_148126

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end sum_first_ten_terms_arithmetic_l148_148126


namespace ratio_fourth_to_third_l148_148142

theorem ratio_fourth_to_third (third_graders fifth_graders fourth_graders : ℕ) (H1 : third_graders = 20) (H2 : fifth_graders = third_graders / 2) (H3 : third_graders + fifth_graders + fourth_graders = 70) : fourth_graders / third_graders = 2 := by
  sorry

end ratio_fourth_to_third_l148_148142


namespace sams_speed_l148_148455

theorem sams_speed (lucas_speed : ℝ) (maya_factor : ℝ) (relationship_factor : ℝ) 
  (h_lucas : lucas_speed = 5)
  (h_maya : maya_factor = 4 / 5)
  (h_relationship : relationship_factor = 9 / 8) :
  (5 / relationship_factor) = 40 / 9 :=
by
  sorry

end sams_speed_l148_148455


namespace arithmetic_sequence_third_term_l148_148113

theorem arithmetic_sequence_third_term {a d : ℝ} (h : 2 * a + 4 * d = 10) : a + 2 * d = 5 :=
sorry

end arithmetic_sequence_third_term_l148_148113


namespace grasshoppers_cannot_return_to_initial_positions_l148_148317

theorem grasshoppers_cannot_return_to_initial_positions :
  (∀ (a b c : ℕ), a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 → a + b + c ≠ 1985) :=
by
  sorry

end grasshoppers_cannot_return_to_initial_positions_l148_148317


namespace six_digit_numbers_with_zero_l148_148670

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l148_148670


namespace quadratic_inequality_solution_set_l148_148583

theorem quadratic_inequality_solution_set (a b c : ℝ) (Δ : ℝ) (hΔ : Δ = b^2 - 4*a*c) :
  (∀ x : ℝ, a*x^2 + b*x + c > 0) ↔ (a > 0 ∧ Δ < 0) := by
  sorry

end quadratic_inequality_solution_set_l148_148583


namespace project_completion_time_l148_148340

theorem project_completion_time
  (A_time B_time : ℕ) 
  (hA : A_time = 20)
  (hB : B_time = 20)
  (A_quit_days : ℕ) 
  (hA_quit : A_quit_days = 10) :
  ∃ x : ℕ, (x - A_quit_days) * (1 / A_time : ℚ) + (x * (1 / B_time : ℚ)) = 1 ∧ x = 15 := by
  sorry

end project_completion_time_l148_148340


namespace combined_difference_is_correct_l148_148966

-- Define the number of cookies each person has
def alyssa_cookies : Nat := 129
def aiyanna_cookies : Nat := 140
def carl_cookies : Nat := 167

-- Define the differences between each pair of people's cookies
def diff_alyssa_aiyanna : Nat := aiyanna_cookies - alyssa_cookies
def diff_alyssa_carl : Nat := carl_cookies - alyssa_cookies
def diff_aiyanna_carl : Nat := carl_cookies - aiyanna_cookies

-- Define the combined difference
def combined_difference : Nat := diff_alyssa_aiyanna + diff_alyssa_carl + diff_aiyanna_carl

-- State the theorem to be proved
theorem combined_difference_is_correct : combined_difference = 76 := by
  sorry

end combined_difference_is_correct_l148_148966


namespace repeating_decimal_to_fraction_l148_148378

theorem repeating_decimal_to_fraction :
  let x := 0.47474747474747 in x = (47 / 99 : ℚ) :=
by
  sorry

end repeating_decimal_to_fraction_l148_148378


namespace expand_expression_l148_148229

theorem expand_expression (x : ℝ) :
  (2 * x + 3) * (4 * x - 5) = 8 * x^2 + 2 * x - 15 :=
by
  sorry

end expand_expression_l148_148229


namespace find_m_l148_148518

theorem find_m (x m : ℝ) (h1 : 4 * x + 2 * m = 5 * x + 1) (h2 : 3 * x = 6 * x - 1) : m = 2 / 3 :=
by
  sorry

end find_m_l148_148518


namespace solve_quadratic_l148_148565

theorem solve_quadratic (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ (x y : ℝ),
    (x + y = S) ∧
    (x * y = P) ∧
    ((x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2) ∨ 
     (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2)) := 
sorry

end solve_quadratic_l148_148565


namespace find_inscribed_circle_area_l148_148589

noncomputable def inscribed_circle_area (length : ℝ) (breadth : ℝ) : ℝ :=
  let perimeter_rectangle := 2 * (length + breadth)
  let side_square := perimeter_rectangle / 4
  let radius_circle := side_square / 2
  Real.pi * radius_circle^2

theorem find_inscribed_circle_area :
  inscribed_circle_area 36 28 = 804.25 := by
  sorry

end find_inscribed_circle_area_l148_148589


namespace factorize_x_squared_minus_one_l148_148057

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l148_148057


namespace combined_length_of_all_CDs_l148_148420

-- Define the lengths of each CD based on the conditions
def length_cd1 := 1.5
def length_cd2 := 1.5
def length_cd3 := 2 * length_cd1
def length_cd4 := length_cd2 / 2
def length_cd5 := length_cd1 + length_cd2

-- Define the combined length of all CDs
def combined_length := length_cd1 + length_cd2 + length_cd3 + length_cd4 + length_cd5

-- State the theorem
theorem combined_length_of_all_CDs : combined_length = 9.75 := by
  sorry

end combined_length_of_all_CDs_l148_148420


namespace union_sets_l148_148106

def setA : Set ℝ := { x | abs (x - 1) < 3 }
def setB : Set ℝ := { x | x^2 - 4 * x < 0 }

theorem union_sets :
  setA ∪ setB = { x : ℝ | -2 < x ∧ x < 4 } :=
sorry

end union_sets_l148_148106


namespace tom_spent_video_games_l148_148595

def cost_football := 14.02
def cost_strategy := 9.46
def cost_batman := 12.04
def total_spent := cost_football + cost_strategy + cost_batman

theorem tom_spent_video_games : total_spent = 35.52 :=
by
  sorry

end tom_spent_video_games_l148_148595


namespace six_digit_numbers_with_at_least_one_zero_correct_l148_148723

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l148_148723


namespace students_in_each_grade_l148_148448

theorem students_in_each_grade (total_students : ℕ) (total_grades : ℕ) (students_per_grade : ℕ) :
  total_students = 22800 → total_grades = 304 → students_per_grade = total_students / total_grades → students_per_grade = 75 :=
by
  intros h1 h2 h3
  sorry

end students_in_each_grade_l148_148448


namespace value_of_p_l148_148609

variable (m n p : ℝ)

-- The conditions from the problem
def first_point_on_line := m = (n / 6) - (2 / 5)
def second_point_on_line := m + p = ((n + 18) / 6) - (2 / 5)

-- The theorem to prove
theorem value_of_p (h1 : first_point_on_line m n) (h2 : second_point_on_line m n p) : p = 3 :=
  sorry

end value_of_p_l148_148609


namespace parabola_equation_line_AB_fixed_point_min_area_AMBN_l148_148089

-- Prove that the equation of the parabola is y^2 = 4x given the focus (1,0) for y^2 = 2px
theorem parabola_equation (p : ℝ) (h : p > 0) (foc : (1, 0) = (1, 2*p*1/4)):
  (∀ x y: ℝ, y^2 = 4*x ↔ y^2 = 2*p*x) := sorry

-- Prove that line AB passes through fixed point T(2,0) given conditions
theorem line_AB_fixed_point (A B : ℝ × ℝ) (hA : A.2^2 = 4*A.1) 
    (hB : B.2^2 = 4*B.1) (h : A.1*B.1 + A.2*B.2 = -4) :
  ∃ T : ℝ × ℝ, T = (2, 0) := sorry

-- Prove that minimum value of area Quadrilateral AMBN is 48
theorem min_area_AMBN (T : ℝ × ℝ) (A B M N : ℝ × ℝ)
    (hT : T = (2, 0)) (hA : A.2^2 = 4*A.1) (hB : B.2^2 = 4*B.1)
    (hM : M.2^2 = 4*M.1) (hN : N.2^2 = 4*N.1)
    (line_AB : A.1 * B.1 + A.2 * B.2 = -4) :
  ∀ (m : ℝ), T.2 = -(1/m)*T.1 + 2 → 
  ((1+m^2) * (1+1/m^2)) * ((m^2 + 2) * (1/m^2 + 2)) = 256 → 
  8 * 48 = 48 := sorry

end parabola_equation_line_AB_fixed_point_min_area_AMBN_l148_148089


namespace corrected_mean_l148_148467

theorem corrected_mean (n : ℕ) (obs_mean : ℝ) (obs_count : ℕ) (wrong_val correct_val : ℝ) :
  obs_count = 40 →
  obs_mean = 100 →
  wrong_val = 75 →
  correct_val = 50 →
  (obs_count * obs_mean - (wrong_val - correct_val)) / obs_count = 3975 / 40 :=
by
  sorry

end corrected_mean_l148_148467


namespace six_digit_numbers_with_zero_l148_148735

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148735


namespace find_a_l148_148751

theorem find_a (a : ℝ) (h : 0.005 * a = 65) : a = 13000 / 100 :=
by
  sorry

end find_a_l148_148751


namespace max_value_expression_l148_148385

noncomputable def expression (x : ℝ) : ℝ := 5^x - 25^x

theorem max_value_expression : 
  (∀ x : ℝ, expression x ≤ 1/4) ∧ (∃ x : ℝ, expression x = 1/4) := 
by 
  sorry

end max_value_expression_l148_148385


namespace total_movies_correct_l148_148831

def num_movies_Screen1 : Nat := 3
def num_movies_Screen2 : Nat := 4
def num_movies_Screen3 : Nat := 2
def num_movies_Screen4 : Nat := 3
def num_movies_Screen5 : Nat := 5
def num_movies_Screen6 : Nat := 2

def total_movies : Nat :=
  num_movies_Screen1 + num_movies_Screen2 + num_movies_Screen3 + num_movies_Screen4 + num_movies_Screen5 + num_movies_Screen6

theorem total_movies_correct :
  total_movies = 19 :=
by 
  sorry

end total_movies_correct_l148_148831


namespace paving_stone_length_l148_148320

theorem paving_stone_length 
  (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (num_stones : ℕ) (stone_width : ℝ) 
  (courtyard_area : ℝ) 
  (total_stones_area : ℝ) 
  (L : ℝ) :
  courtyard_length = 50 →
  courtyard_width = 16.5 →
  num_stones = 165 →
  stone_width = 2 →
  courtyard_area = courtyard_length * courtyard_width →
  total_stones_area = num_stones * stone_width * L →
  courtyard_area = total_stones_area →
  L = 2.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end paving_stone_length_l148_148320


namespace train_seat_count_l148_148022

theorem train_seat_count (t : ℝ)
  (h1 : ∃ (t : ℝ), t = 36 + 0.2 * t + 0.5 * t) :
  t = 120 :=
by
  sorry

end train_seat_count_l148_148022


namespace floor_S_proof_l148_148132

noncomputable def floor_S (a b c d: ℝ) : ℝ :=
⌊a + b + c + d⌋

theorem floor_S_proof (a b c d : ℝ)
  (h1 : a ^ 2 + 2 * b ^ 2 = 2016)
  (h2 : c ^ 2 + 2 * d ^ 2 = 2016)
  (h3 : a * c = 1024)
  (h4 : b * d = 1024)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : floor_S a b c d = 129 := 
sorry

end floor_S_proof_l148_148132


namespace students_drawn_in_sample_l148_148481

def total_people : ℕ := 1600
def number_of_teachers : ℕ := 100
def sample_size : ℕ := 80
def number_of_students : ℕ := total_people - number_of_teachers
def expected_students_sample : ℕ := 75

theorem students_drawn_in_sample : (sample_size * number_of_students) / total_people = expected_students_sample :=
by
  -- The proof steps would go here
  sorry

end students_drawn_in_sample_l148_148481


namespace gym_monthly_revenue_l148_148619

theorem gym_monthly_revenue (members_per_month_fee : ℕ) (num_members : ℕ) 
  (h1 : members_per_month_fee = 18 * 2) 
  (h2 : num_members = 300) : 
  num_members * members_per_month_fee = 10800 := 
by 
  -- calculation rationale goes here
  sorry

end gym_monthly_revenue_l148_148619


namespace winning_candidate_percentage_l148_148898

theorem winning_candidate_percentage
  (total_votes : ℕ)
  (vote_majority : ℕ)
  (winning_candidate_votes : ℕ)
  (losing_candidate_votes : ℕ) :
  total_votes = 400 →
  vote_majority = 160 →
  winning_candidate_votes = total_votes * 70 / 100 →
  losing_candidate_votes = total_votes - winning_candidate_votes →
  winning_candidate_votes - losing_candidate_votes = vote_majority →
  winning_candidate_votes = 280 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end winning_candidate_percentage_l148_148898


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l148_148722

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l148_148722


namespace k_range_l148_148081

def y_increasing (k : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ + 1 < k * x₂ + 1
def y_max_min (k : ℝ) : Prop := (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ (x^2 - 2 * x + 3 = 2)) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ (x^2 - 2 * x + 3 = 3))

theorem k_range (k : ℝ) (hk : (¬ (0 < k ∧ y_max_min k) ∧ (0 < k ∨ y_max_min k))) : 
  (0 < k ∧ k < 1) ∨ (k > 2) :=
sorry

end k_range_l148_148081


namespace width_minimizes_fencing_l148_148422

-- Define the conditions for the problem
def garden_area_cond (w : ℝ) : Prop :=
  w * (w + 10) ≥ 150

-- Define the main statement to prove
theorem width_minimizes_fencing (w : ℝ) (h : w ≥ 0) : garden_area_cond w → w = 10 :=
  by
  sorry

end width_minimizes_fencing_l148_148422


namespace problem_statement_l148_148134

variable (p q r s : ℝ) (ω : ℂ)

theorem problem_statement (hp : p ≠ -1) (hq : q ≠ -1) (hr : r ≠ -1) (hs : s ≠ -1) 
  (hω : ω ^ 4 = 1) (hω_ne : ω ≠ 1)
  (h_eq : (1 / (p + ω) + 1 / (q + ω) + 1 / (r + ω) + 1 / (s + ω)) = 3 / ω^2) :
  1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1) + 1 / (s + 1) = 3 := 
by sorry

end problem_statement_l148_148134


namespace max_sum_pyramid_l148_148480

theorem max_sum_pyramid (F_pentagonal : ℕ) (F_rectangular : ℕ) (E_pentagonal : ℕ) (E_rectangular : ℕ) (V_pentagonal : ℕ) (V_rectangular : ℕ)
  (original_faces : ℕ) (original_edges : ℕ) (original_vertices : ℕ)
  (H1 : original_faces = 7)
  (H2 : original_edges = 15)
  (H3 : original_vertices = 10)
  (H4 : F_pentagonal = 11)
  (H5 : E_pentagonal = 20)
  (H6 : V_pentagonal = 11)
  (H7 : F_rectangular = 10)
  (H8 : E_rectangular = 19)
  (H9 : V_rectangular = 11) :
  max (F_pentagonal + E_pentagonal + V_pentagonal) (F_rectangular + E_rectangular + V_rectangular) = 42 :=
by
  sorry

end max_sum_pyramid_l148_148480


namespace batsman_average_after_12th_inning_l148_148816

variable (A : ℕ) (total_balls_faced : ℕ)

theorem batsman_average_after_12th_inning 
  (h1 : ∃ A, ∀ total_runs, total_runs = 11 * A)
  (h2 : ∃ A, ∀ total_runs_new, total_runs_new = 12 * (A + 4) ∧ total_runs_new - 60 = 11 * A)
  (h3 : 8 * 4 ≤ 60)
  (h4 : 6000 / total_balls_faced ≥ 130) 
  : (A + 4 = 16) :=
by
  sorry

end batsman_average_after_12th_inning_l148_148816


namespace pens_bought_l148_148833

theorem pens_bought
  (P : ℝ)
  (cost := 36 * P)
  (discount := 0.99 * P)
  (profit_percent := 0.1)
  (profit := (40 * discount) - cost)
  (profit_eq : profit = profit_percent * cost) :
  40 = 40 := 
by
  sorry

end pens_bought_l148_148833


namespace four_digit_numbers_with_three_identical_digits_l148_148312

theorem four_digit_numbers_with_three_identical_digits :
  ∃ n : ℕ, (n = 18) ∧ (∀ x, 1000 ≤ x ∧ x < 10000 → 
  (x / 1000 = 1) ∧ (
    (x % 1000 / 100 = x % 100 / 10) ∧ (x % 1000 / 100 = x % 10))) :=
by
  sorry

end four_digit_numbers_with_three_identical_digits_l148_148312


namespace cubic_polynomial_value_at_3_and_neg3_l148_148283

variable (Q : ℝ → ℝ)
variable (a b c d m : ℝ)
variable (h1 : Q 1 = 5 * m)
variable (h0 : Q 0 = 2 * m)
variable (h_1 : Q (-1) = 6 * m)
variable (hQ : ∀ x, Q x = a * x^3 + b * x^2 + c * x + d)

theorem cubic_polynomial_value_at_3_and_neg3 :
  Q 3 + Q (-3) = 67 * m := by
  -- sorry is used to skip the proof
  sorry

end cubic_polynomial_value_at_3_and_neg3_l148_148283


namespace find_z_l148_148403

open Complex

theorem find_z (z : ℂ) (h : (1 - I) * z = 2 * I) : z = -1 + I := by
  sorry

end find_z_l148_148403


namespace six_digit_numbers_with_zero_l148_148734

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148734


namespace original_cost_of_each_magazine_l148_148830

-- Definitions and conditions
def magazine_cost (C : ℝ) : Prop :=
  let total_magazines := 10
  let sell_price := 3.50
  let gain := 5
  let total_revenue := total_magazines * sell_price
  let total_cost := total_revenue - gain
  C = total_cost / total_magazines

-- Goal to prove
theorem original_cost_of_each_magazine : ∃ C : ℝ, magazine_cost C ∧ C = 3 :=
by
  sorry

end original_cost_of_each_magazine_l148_148830


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l148_148716

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l148_148716


namespace circle_center_and_radius_l148_148920

noncomputable def circle_eq : Prop :=
  ∀ x y : ℝ, (x^2 + y^2 + 2 * x - 2 * y - 2 = 0) ↔ (x + 1)^2 + (y - 1)^2 = 4

theorem circle_center_and_radius :
  ∃ center : ℝ × ℝ, ∃ r : ℝ, 
  center = (-1, 1) ∧ r = 2 ∧ circle_eq :=
by
  sorry

end circle_center_and_radius_l148_148920


namespace six_digit_numbers_with_zero_l148_148672

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l148_148672


namespace min_value_of_trig_expression_l148_148587

open Real

theorem min_value_of_trig_expression (α : ℝ) (h₁ : sin α ≠ 0) (h₂ : cos α ≠ 0) : 
  (9 / (sin α)^2 + 1 / (cos α)^2) ≥ 16 :=
  sorry

end min_value_of_trig_expression_l148_148587


namespace six_digit_numbers_with_zero_count_l148_148674

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l148_148674


namespace miles_traveled_total_l148_148910

-- Define the initial distance and the additional distance
def initial_distance : ℝ := 212.3
def additional_distance : ℝ := 372.0

-- Define the total distance as the sum of the initial and additional distances
def total_distance : ℝ := initial_distance + additional_distance

-- Prove that the total distance is 584.3 miles
theorem miles_traveled_total : total_distance = 584.3 := by
  sorry

end miles_traveled_total_l148_148910


namespace product_of_repeating_decimal_l148_148491

noncomputable def t : ℚ := 152 / 333

theorem product_of_repeating_decimal :
  8 * t = 1216 / 333 :=
by {
  -- Placeholder for proof.
  sorry
}

end product_of_repeating_decimal_l148_148491


namespace problem_correct_statements_l148_148808

theorem problem_correct_statements : 
  (Nat.choose 7 2 = Nat.choose 7 5) ∧
  (Nat.choose 5 3 = Nat.choose 4 2 + Nat.choose 4 3) ∧
  (5 * Nat.factorial 5 = Nat.factorial 6 - Nat.factorial 5) :=
by {
  -- prove each condition step by step using required properties
  sorry
}

end problem_correct_statements_l148_148808


namespace quadrant_of_angle_l148_148529

variable (α : ℝ)

theorem quadrant_of_angle (h₁ : Real.sin α < 0) (h₂ : Real.tan α > 0) : 
  3 * (π / 2) < α ∧ α < 2 * π ∨ π < α ∧ α < 3 * (π / 2) :=
by
  sorry

end quadrant_of_angle_l148_148529


namespace S7_eq_14_l148_148032

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith_seq : arithmetic_sequence a)
variables (h_a3 : a 3 = 0) (h_a6_plus_a7 : a 6 + a 7 = 14)

theorem S7_eq_14 : S 7 = 14 := sorry

end S7_eq_14_l148_148032


namespace inequality_holds_l148_148987

variable {x y : ℝ}

theorem inequality_holds (h₀ : 0 < x) (h₁ : x < 1) (h₂ : 0 < y) (h₃ : y < 1) :
  (x^2 / (x + y)) + (y^2 / (1 - x)) + ((1 - x - y)^2 / (1 - y)) ≥ 1 / 2 := by
  sorry

end inequality_holds_l148_148987


namespace sampling_method_is_systematic_sampling_l148_148199

-- Definitions based on the problem's conditions
def produces_products (factory : Type) : Prop := sorry
def uses_conveyor_belt (factory : Type) : Prop := sorry
def takes_item_every_5_minutes (inspector : Type) : Prop := sorry

-- Lean 4 statement to prove the question equals the answer given the conditions
theorem sampling_method_is_systematic_sampling
  (factory : Type)
  (inspector : Type)
  (h1 : produces_products factory)
  (h2 : uses_conveyor_belt factory)
  (h3 : takes_item_every_5_minutes inspector) :
  systematic_sampling_method := 
sorry

end sampling_method_is_systematic_sampling_l148_148199


namespace angle_perpendicular_coterminal_l148_148418

theorem angle_perpendicular_coterminal (α β : ℝ) (k : ℤ) 
  (h_perpendicular : ∃ k, β = α + 90 + k * 360 ∨ β = α - 90 + k * 360) : 
  β = α + 90 + k * 360 ∨ β = α - 90 + k * 360 :=
sorry

end angle_perpendicular_coterminal_l148_148418


namespace total_metal_rods_needed_l148_148826

-- Definitions extracted from the conditions
def metal_sheets_per_panel := 3
def metal_beams_per_panel := 2
def panels := 10
def rods_per_sheet := 10
def rods_per_beam := 4

-- Problem statement: Prove the total number of metal rods required is 380
theorem total_metal_rods_needed :
  (panels * ((metal_sheets_per_panel * rods_per_sheet) + (metal_beams_per_panel * rods_per_beam))) = 380 :=
by
  sorry

end total_metal_rods_needed_l148_148826


namespace sequence_n_l148_148066

theorem sequence_n (a : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → (n^2 + 1) * a n = n * (a (n^2) + 1)) :
  ∀ n : ℕ, 0 < n → a n = n := 
by
  sorry

end sequence_n_l148_148066


namespace percentage_broken_in_second_set_l148_148397

-- Define the given conditions
def first_set_total : ℕ := 50
def first_set_broken_percent : ℚ := 0.10
def second_set_total : ℕ := 60
def total_broken : ℕ := 17

-- The proof problem statement
theorem percentage_broken_in_second_set :
  let first_set_broken := first_set_broken_percent * first_set_total
  let second_set_broken := total_broken - first_set_broken
  (second_set_broken / second_set_total) * 100 = 20 := 
sorry

end percentage_broken_in_second_set_l148_148397


namespace carly_practice_backstroke_days_per_week_l148_148221

theorem carly_practice_backstroke_days_per_week 
  (butterfly_hours_per_day : ℕ) 
  (butterfly_days_per_week : ℕ) 
  (backstroke_hours_per_day : ℕ) 
  (total_hours_per_month : ℕ)
  (weeks_per_month : ℕ)
  (d : ℕ)
  (h1 : butterfly_hours_per_day = 3)
  (h2 : butterfly_days_per_week = 4)
  (h3 : backstroke_hours_per_day = 2)
  (h4 : total_hours_per_month = 96)
  (h5 : weeks_per_month = 4)
  (h6 : total_hours_per_month - (butterfly_hours_per_day * butterfly_days_per_week * weeks_per_month) = backstroke_hours_per_day * d * weeks_per_month) :
  d = 6 := by
  sorry

end carly_practice_backstroke_days_per_week_l148_148221


namespace car_trip_problem_l148_148290

theorem car_trip_problem (a b c : ℕ) (x : ℕ) 
(h1 : 1 ≤ a) 
(h2 : a + b + c ≤ 9)
(h3 : 100 * b + 10 * c + a - 100 * a - 10 * b - c = 60 * x) 
: a^2 + b^2 + c^2 = 14 := 
by
  sorry

end car_trip_problem_l148_148290


namespace jill_food_spending_l148_148145

theorem jill_food_spending :
  ∀ (T : ℝ) (c f o : ℝ),
    c = 0.5 * T →
    o = 0.3 * T →
    (0.04 * c + 0 + 0.1 * o) = 0.05 * T →
    f = 0.2 * T :=
by
  intros T c f o h_c h_o h_tax
  sorry

end jill_food_spending_l148_148145


namespace milkshakes_per_hour_l148_148631

variable (L : ℕ) -- number of milkshakes Luna can make per hour

theorem milkshakes_per_hour
  (h1 : ∀ (A : ℕ), A = 3) -- Augustus makes 3 milkshakes per hour
  (h2 : ∀ (H : ℕ), H = 8) -- they have been making milkshakes for 8 hours
  (h3 : ∀ (Total : ℕ), Total = 80) -- together they made 80 milkshakes
  (h4 : ∀ (Augustus_milkshakes : ℕ), Augustus_milkshakes = 3 * 8) -- Augustus made 24 milkshakes in 8 hours
 : L = 7 := sorry

end milkshakes_per_hour_l148_148631


namespace ram_first_year_balance_l148_148804

-- Given conditions
def initial_deposit : ℝ := 1000
def interest_first_year : ℝ := 100

-- Calculate end of the first year balance
def balance_first_year := initial_deposit + interest_first_year

-- Prove that balance_first_year is $1100
theorem ram_first_year_balance :
  balance_first_year = 1100 :=
by 
  sorry

end ram_first_year_balance_l148_148804


namespace total_flowers_correct_l148_148778

def rosa_original_flowers : ℝ := 67.5
def andre_gifted_flowers : ℝ := 90.75
def total_flowers (rosa : ℝ) (andre : ℝ) : ℝ := rosa + andre

theorem total_flowers_correct : total_flowers rosa_original_flowers andre_gifted_flowers = 158.25 :=
by 
  rw [total_flowers]
  sorry

end total_flowers_correct_l148_148778


namespace value_of_a_2015_l148_148540

def a : ℕ → Int
| 0 => 1
| 1 => 5
| n+2 => a (n+1) - a n

theorem value_of_a_2015 : a 2014 = -5 := by
  sorry

end value_of_a_2015_l148_148540


namespace cube_sum_l148_148257

theorem cube_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 15) : x^3 + y^3 = 550 := by
  sorry

end cube_sum_l148_148257


namespace isosceles_triangle_base_angle_l148_148887

theorem isosceles_triangle_base_angle (vertex_angle : ℝ) (base_angle : ℝ) 
  (h1 : vertex_angle = 60) 
  (h2 : 2 * base_angle + vertex_angle = 180) : 
  base_angle = 60 := 
by 
  sorry

end isosceles_triangle_base_angle_l148_148887


namespace factorize_difference_of_squares_l148_148045

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l148_148045


namespace largest_square_area_l148_148027

theorem largest_square_area (a b c : ℝ)
  (h1 : a^2 + b^2 = c^2)
  (h2 : a^2 + b^2 + c^2 = 450) :
  c^2 = 225 :=
by
  sorry

end largest_square_area_l148_148027


namespace cube_painting_problem_l148_148206

theorem cube_painting_problem (n : ℕ) (hn : n > 0) :
  (6 * n^2 = (6 * n^3) / 3) ↔ n = 3 :=
by sorry

end cube_painting_problem_l148_148206


namespace cube_problem_l148_148209

theorem cube_problem (n : ℕ) (H1 : 6 * n^2 = 1 / 3 * 6 * n^3) : n = 3 :=
sorry

end cube_problem_l148_148209


namespace arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two_l148_148858

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two (x : ℝ) (hx1 : -1 ≤ x) (hx2 : x ≤ 1) :
  (Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = Real.pi / 2) :=
sorry

end arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two_l148_148858


namespace find_p_and_q_solution_set_l148_148172

theorem find_p_and_q (p q : ℝ) (h : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - p * x - q < 0) : 
  p = 5 ∧ q = -6 :=
sorry

theorem solution_set (p q : ℝ) (h_p : p = 5) (h_q : q = -6) : 
  ∀ x : ℝ, q * x^2 - p * x - 1 > 0 ↔ - (1 / 2) < x ∧ x < - (1 / 3) :=
sorry

end find_p_and_q_solution_set_l148_148172


namespace factorize_x_squared_minus_one_l148_148050

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_x_squared_minus_one_l148_148050


namespace return_trip_amount_l148_148296

noncomputable def gasoline_expense : ℝ := 8
noncomputable def lunch_expense : ℝ := 15.65
noncomputable def gift_expense_per_person : ℝ := 5
noncomputable def grandma_gift_per_person : ℝ := 10
noncomputable def initial_amount : ℝ := 50

theorem return_trip_amount : 
  let total_expense := gasoline_expense + lunch_expense + (gift_expense_per_person * 2)
  let total_money_gifted := grandma_gift_per_person * 2
  initial_amount - total_expense + total_money_gifted = 36.35 :=
by 
  let total_expense := gasoline_expense + lunch_expense + (gift_expense_per_person * 2)
  let total_money_gifted := grandma_gift_per_person * 2
  sorry

end return_trip_amount_l148_148296


namespace solve_for_k_l148_148750

theorem solve_for_k (k : ℤ) : (∃ x : ℤ, x = 5 ∧ 4 * x + 2 * k = 8) → k = -6 :=
by
  sorry

end solve_for_k_l148_148750


namespace codys_grandmother_age_l148_148640

theorem codys_grandmother_age (cody_age : ℕ) (grandmother_factor : ℕ) (h1 : cody_age = 14) (h2 : grandmother_factor = 6) :
  grandmother_factor * cody_age = 84 :=
by
  sorry

end codys_grandmother_age_l148_148640


namespace at_least_one_nonnegative_l148_148079

theorem at_least_one_nonnegative (a b c d e f g h : ℝ) :
  ac + bd ≥ 0 ∨ ae + bf ≥ 0 ∨ ag + bh ≥ 0 ∨ ce + df ≥ 0 ∨ cg + dh ≥ 0 ∨ eg + fh ≥ 0 :=
sorry

end at_least_one_nonnegative_l148_148079


namespace largest_interior_angle_of_triangle_l148_148886

theorem largest_interior_angle_of_triangle (exterior_ratio_2k : ℝ) (exterior_ratio_3k : ℝ) (exterior_ratio_4k : ℝ) (sum_exterior_angles : exterior_ratio_2k + exterior_ratio_3k + exterior_ratio_4k = 360) :
  180 - exterior_ratio_2k = 100 :=
by
  sorry

end largest_interior_angle_of_triangle_l148_148886


namespace Abby_in_seat_3_l148_148836

variables (P : Type) [Inhabited P]
variables (Abby Bret Carl Dana : P)
variables (seat : P → ℕ)

-- Conditions from the problem:
-- Bret is actually sitting in seat #2.
axiom Bret_in_seat_2 : seat Bret = 2

-- False statement 1: Dana is next to Bret.
axiom false_statement_1 : ¬ (seat Dana = 1 ∨ seat Dana = 3)

-- False statement 2: Carl is sitting between Dana and Bret.
axiom false_statement_2 : ¬ (seat Carl = 1)

-- The final translated proof problem:
theorem Abby_in_seat_3 : seat Abby = 3 :=
sorry

end Abby_in_seat_3_l148_148836


namespace six_digit_numbers_with_zero_l148_148667

theorem six_digit_numbers_with_zero : 
  let total_six_digit_numbers := 9 * 10^5 in
  let six_digit_numbers_without_zero := 9^6 in
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := 
by 
  sorry

end six_digit_numbers_with_zero_l148_148667


namespace solve_quadratic_l148_148564

theorem solve_quadratic (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ (x y : ℝ),
    (x + y = S) ∧
    (x * y = P) ∧
    ((x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2) ∨ 
     (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2)) := 
sorry

end solve_quadratic_l148_148564


namespace maximum_value_l148_148962

variable {a b c : ℝ}

-- Conditions
variable (h : a^2 + b^2 = c^2)

theorem maximum_value (h : a^2 + b^2 = c^2) : 
  (∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ 
   (∀ x y z : ℝ, x^2 + y^2 = z^2 → (x^2 + y^2 + x*y) / z^2 ≤ 1.5)) := 
sorry

end maximum_value_l148_148962


namespace ajay_total_gain_l148_148839

noncomputable def ajay_gain : ℝ :=
  let cost1 := 15 * 14.50
  let cost2 := 10 * 13
  let total_cost := cost1 + cost2
  let total_weight := 15 + 10
  let selling_price := total_weight * 15
  selling_price - total_cost

theorem ajay_total_gain :
  ajay_gain = 27.50 := by
  sorry

end ajay_total_gain_l148_148839


namespace six_digit_numbers_with_at_least_one_zero_correct_l148_148724

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l148_148724


namespace line_equation_through_P_and_equidistant_from_A_B_l148_148399

theorem line_equation_through_P_and_equidistant_from_A_B (P A B : ℝ × ℝ) (hP : P = (1, 2)) (hA : A = (2, 3)) (hB : B = (4, -5)) :
  (∃ l : ℝ × ℝ → Prop, ∀ x y, l (x, y) ↔ 4 * x + y - 6 = 0 ∨ 3 * x + 2 * y - 7 = 0) :=
sorry

end line_equation_through_P_and_equidistant_from_A_B_l148_148399


namespace age_is_50_l148_148478

-- Definitions only based on the conditions provided
def future_age (A: ℕ) := A + 5
def past_age (A: ℕ) := A - 5

theorem age_is_50 (A : ℕ) (h : 5 * future_age A - 5 * past_age A = A) : A = 50 := 
by 
  sorry  -- proof should be provided here

end age_is_50_l148_148478


namespace six_digit_numbers_with_zero_l148_148698

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148698


namespace six_digit_numbers_with_zero_l148_148692

theorem six_digit_numbers_with_zero :
  let total_six_digit := 9 * 10^5 in
  let six_digit_no_zeros := 9^6 in
  total_six_digit - six_digit_no_zeros = 368559 :=
by 
  let total_six_digit := 9 * 10^5;
  let six_digit_no_zeros := 9^6;
  exact Eq.refl (total_six_digit - six_digit_no_zeros)

-- placeholder for the detailed proof
sorry

end six_digit_numbers_with_zero_l148_148692


namespace num_undef_values_l148_148393

theorem num_undef_values : 
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℝ, (x^2 + 4 * x - 5) * (x - 4) = 0 → x = -5 ∨ x = 1 ∨ x = 4 :=
by
  -- We are stating that there exists a natural number n such that n = 3
  -- and for all real numbers x, if (x^2 + 4*x - 5)*(x - 4) = 0,
  -- then x must be one of -5, 1, or 4.
  sorry

end num_undef_values_l148_148393


namespace amusement_park_ticket_price_l148_148355

theorem amusement_park_ticket_price
  (num_people_weekday : ℕ)
  (num_people_saturday : ℕ)
  (num_people_sunday : ℕ)
  (total_people_week : ℕ)
  (total_revenue_week : ℕ)
  (people_per_day_weekday : num_people_weekday = 100)
  (people_saturday : num_people_saturday = 200)
  (people_sunday : num_people_sunday = 300)
  (total_people : total_people_week = 1000)
  (total_revenue : total_revenue_week = 3000)
  (total_people_calc : 5 * num_people_weekday + num_people_saturday + num_people_sunday = total_people_week)
  (revenue_eq : total_people_week * 3 = total_revenue_week) :
  3 = 3 :=
by
  sorry

end amusement_park_ticket_price_l148_148355


namespace seven_digit_divisible_by_11_l148_148411

theorem seven_digit_divisible_by_11 (m n : ℕ) (h1: 0 ≤ m ∧ m ≤ 9) (h2: 0 ≤ n ∧ n ≤ 9) (h3 : 10 + n - m ≡ 0 [MOD 11])  : m + n = 1 :=
by
  sorry

end seven_digit_divisible_by_11_l148_148411


namespace solution1_solution2_l148_148560

namespace MathProofProblem

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  4 * x - 2 * y = 14 ∧ 3 * x + 2 * y = 7

-- Prove the solution for the first system
theorem solution1 : ∃ (x y : ℝ), system1 x y ∧ x = 3 ∧ y = -1 := by
  sorry

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  y = x + 1 ∧ 2 * x + y = 10

-- Prove the solution for the second system
theorem solution2 : ∃ (x y : ℝ), system2 x y ∧ x = 3 ∧ y = 4 := by
  sorry

end MathProofProblem

end solution1_solution2_l148_148560


namespace fraction_of_draws_is_two_ninths_l148_148266

-- Define the fraction of games that Ben wins and Tom wins
def BenWins : ℚ := 4 / 9
def TomWins : ℚ := 1 / 3

-- Definition of the fraction of games ending in a draw
def fraction_of_draws (BenWins TomWins : ℚ) : ℚ :=
  1 - (BenWins + TomWins)

-- The theorem to be proved
theorem fraction_of_draws_is_two_ninths : fraction_of_draws BenWins TomWins = 2 / 9 :=
by
  sorry

end fraction_of_draws_is_two_ninths_l148_148266


namespace area_enclosed_by_cosine_l148_148581

theorem area_enclosed_by_cosine :
  ∫ x in -Real.pi..Real.pi, (1 + Real.cos x) = 2 * Real.pi := by
  sorry

end area_enclosed_by_cosine_l148_148581


namespace quadratic_no_real_roots_range_l148_148259

theorem quadratic_no_real_roots_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) → k < -1 :=
by
  sorry

end quadratic_no_real_roots_range_l148_148259


namespace product_of_repeating_decimal_and_integer_l148_148493

noncomputable def repeating_decimal_to_fraction (s : ℝ) : ℚ := 
  456 / 999

noncomputable def multiply_and_simplify (s : ℝ) (n : ℤ) : ℚ := 
  (repeating_decimal_to_fraction s) * (n : ℚ)

theorem product_of_repeating_decimal_and_integer 
(s : ℝ) (h : s = 0.456456456456456456456456456456456456456456) :
  multiply_and_simplify s 8 = 1216 / 333 :=
by sorry

end product_of_repeating_decimal_and_integer_l148_148493


namespace car_rental_cost_l148_148186

def day1_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def day2_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def day3_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def total_cost (day1 : ℝ) (day2 : ℝ) (day3 : ℝ) : ℝ :=
  day1 + day2 + day3

theorem car_rental_cost :
  let day1_base_rate := 150
  let day2_base_rate := 100
  let day3_base_rate := 75
  let day1_miles_driven := 620
  let day2_miles_driven := 744
  let day3_miles_driven := 510
  let day1_cost_per_mile := 0.50
  let day2_cost_per_mile := 0.40
  let day3_cost_per_mile := 0.30
  day1_cost day1_base_rate day1_miles_driven day1_cost_per_mile +
  day2_cost day2_base_rate day2_miles_driven day2_cost_per_mile +
  day3_cost day3_base_rate day3_miles_driven day3_cost_per_mile = 1085.60 :=
by
  let day1 := day1_cost 150 620 0.50
  let day2 := day2_cost 100 744 0.40
  let day3 := day3_cost 75 510 0.30
  let total := total_cost day1 day2 day3
  show total = 1085.60
  sorry

end car_rental_cost_l148_148186


namespace find_f_lg_lg2_l148_148870

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x ^ 2) - x) + 4

theorem find_f_lg_lg2 :
  f (Real.logb 10 (2)) = 3 :=
sorry

end find_f_lg_lg2_l148_148870


namespace marked_price_l148_148341

theorem marked_price (P : ℝ)
  (h₁ : 20 / 100 = 0.20)
  (h₂ : 15 / 100 = 0.15)
  (h₃ : 5 / 100 = 0.05)
  (h₄ : 7752 = 0.80 * 0.85 * 0.95 * P)
  : P = 11998.76 := by
  sorry

end marked_price_l148_148341


namespace domain_of_function_l148_148584

def domain_of_f (x : ℝ) : Prop :=
  (x ≤ 2) ∧ (x ≠ 1)

theorem domain_of_function :
  ∀ x : ℝ, x ∈ { x | (x ≤ 2) ∧ (x ≠ 1) } ↔ domain_of_f x :=
by
  sorry

end domain_of_function_l148_148584


namespace max_value_3x_plus_4y_l148_148082

theorem max_value_3x_plus_4y (x y : ℝ) : x^2 + y^2 = 14 * x + 6 * y + 6 → 3 * x + 4 * y ≤ 73 :=
sorry

end max_value_3x_plus_4y_l148_148082


namespace isosceles_triangle_y_value_l148_148018

theorem isosceles_triangle_y_value :
  ∃ y : ℝ, (y = 1 + Real.sqrt 51 ∨ y = 1 - Real.sqrt 51) ∧ 
  (Real.sqrt ((y - 1)^2 + (4 - (-3))^2) = 10) :=
by sorry

end isosceles_triangle_y_value_l148_148018


namespace sum_first_10_terms_arithmetic_sequence_l148_148130

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end sum_first_10_terms_arithmetic_sequence_l148_148130


namespace six_digit_numbers_with_zero_l148_148683

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148683


namespace sum_first_10_terms_arithmetic_seq_l148_148124

theorem sum_first_10_terms_arithmetic_seq :
  ∀ (a : ℕ → ℤ), (a 1 = -2) → (a 2 + a 6 = 2) →
  (∃ S₁₀, S₁₀ = 10 * a 1 + (10 * (10 - 1)) / 2 * (a 2 - a 1) / (2 - 1) ∧ S₁₀ = 25) :=
begin
  assume a,
  assume h1 : a 1 = -2,
  assume h2 : a 2 + a 6 = 2,
  sorry
end

end sum_first_10_terms_arithmetic_seq_l148_148124


namespace stratified_sampling_second_class_l148_148025

theorem stratified_sampling_second_class (total_products : ℕ) (first_class : ℕ) (second_class : ℕ) (third_class : ℕ) (sample_size : ℕ) (h_total : total_products = 200) (h_first : first_class = 40) (h_second : second_class = 60) (h_third : third_class = 100) (h_sample : sample_size = 40) :
  (second_class * sample_size) / total_products = 12 :=
by
  sorry

end stratified_sampling_second_class_l148_148025


namespace six_digit_numbers_with_at_least_one_zero_l148_148708

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l148_148708


namespace spider_to_fly_routes_l148_148618

-- Given conditions
def start_pos : ℕ × ℕ := (0, 0)
def end_pos : ℕ × ℕ := (5, 3)
def possible_moves : ℕ := end_pos.fst + end_pos.snd  -- Total moves: 5 right and 3 upward

-- Theorem to be proven
theorem spider_to_fly_routes : 
  nat.choose (start_pos.fst + start_pos.snd + possible_moves) end_pos.snd = 56 :=
sorry

end spider_to_fly_routes_l148_148618


namespace profit_calculation_l148_148200

open Nat

-- Define the conditions 
def cost_of_actors : Nat := 1200 
def number_of_people : Nat := 50
def cost_per_person_food : Nat := 3
def sale_price : Nat := 10000

-- Define the derived costs
def total_food_cost : Nat := number_of_people * cost_per_person_food
def total_combined_cost : Nat := cost_of_actors + total_food_cost
def equipment_rental_cost : Nat := 2 * total_combined_cost
def total_cost : Nat := cost_of_actors + total_food_cost + equipment_rental_cost
def expected_profit : Nat := 5950 

-- Define the profit calculation
def profit : Nat := sale_price - total_cost 

-- The theorem to be proved
theorem profit_calculation : profit = expected_profit := by
  -- Proof is omitted
  sorry

end profit_calculation_l148_148200


namespace find_constants_l148_148505

variable (x : ℝ)

def A := 3
def B := -3
def C := 11

theorem find_constants (h₁ : x ≠ 2) (h₂ : x ≠ 4) :
  (5 * x + 2) / ((x - 2) * (x - 4)^2) = A / (x - 2) + B / (x - 4) + C / (x - 4)^2 :=
by
  unfold A B C
  sorry

end find_constants_l148_148505


namespace factorize_x_squared_minus_one_l148_148056

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l148_148056


namespace find_counterfeit_coins_l148_148629

structure Coins :=
  (a a₁ b b₁ c c₁ : ℝ)
  (genuine_weight : ℝ)
  (counterfeit_weight : ℝ)
  (a_is_genuine_or_counterfeit : a = genuine_weight ∨ a = counterfeit_weight)
  (a₁_is_genuine_or_counterfeit : a₁ = genuine_weight ∨ a₁ = counterfeit_weight)
  (b_is_genuine_or_counterfeit : b = genuine_weight ∨ b = counterfeit_weight)
  (b₁_is_genuine_or_counterfeit : b₁ = genuine_weight ∨ b₁ = counterfeit_weight)
  (c_is_genuine_or_counterfeit : c = genuine_weight ∨ c = counterfeit_weight)
  (c₁_is_genuine_or_counterfeit : c₁ = genuine_weight ∨ c₁ = counterfeit_weight)
  (counterfeit_pair_ends_unit_segment : (a = counterfeit_weight ∧ a₁ = counterfeit_weight) 
                                        ∨ (b = counterfeit_weight ∧ b₁ = counterfeit_weight)
                                        ∨ (c = counterfeit_weight ∧ c₁ = counterfeit_weight))

theorem find_counterfeit_coins (coins : Coins) : 
  (coins.a = coins.genuine_weight ∧ coins.b = coins.genuine_weight → coins.a₁ = coins.counterfeit_weight ∧ coins.b₁ = coins.counterfeit_weight) 
  ∧ (coins.a < coins.b → coins.a = coins.counterfeit_weight ∧ coins.b₁ = coins.counterfeit_weight) 
  ∧ (coins.b < coins.a → coins.b = coins.counterfeit_weight ∧ coins.a₁ = coins.counterfeit_weight) := 
by
  sorry

end find_counterfeit_coins_l148_148629


namespace circle_hyperbola_intersection_l148_148011

def hyperbola_equation (x y a : ℝ) : Prop := x^2 - y^2 = a^2
def circle_equation (x y c d r : ℝ) : Prop := (x - c)^2 + (y - d)^2 = r^2

theorem circle_hyperbola_intersection (a r : ℝ) (P Q R S : ℝ × ℝ):
  (∃ c d: ℝ, 
    circle_equation P.1 P.2 c d r ∧ 
    circle_equation Q.1 Q.2 c d r ∧ 
    circle_equation R.1 R.2 c d r ∧ 
    circle_equation S.1 S.2 c d r ∧ 
    hyperbola_equation P.1 P.2 a ∧ 
    hyperbola_equation Q.1 Q.2 a ∧ 
    hyperbola_equation R.1 R.2 a ∧ 
    hyperbola_equation S.1 S.2 a
  ) →
  (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) + (R.1^2 + R.2^2) + (S.1^2 + S.2^2) = 4 * r^2 :=
by
  sorry

end circle_hyperbola_intersection_l148_148011


namespace mean_steps_per_day_l148_148484

theorem mean_steps_per_day (total_steps : ℕ) (days_in_april : ℕ) (h_total : total_steps = 243000) (h_days : days_in_april = 30) :
  (total_steps / days_in_april) = 8100 :=
by
  sorry

end mean_steps_per_day_l148_148484


namespace max_value_expression_l148_148384

noncomputable def expression (x : ℝ) : ℝ := 5^x - 25^x

theorem max_value_expression : 
  (∀ x : ℝ, expression x ≤ 1/4) ∧ (∃ x : ℝ, expression x = 1/4) := 
by 
  sorry

end max_value_expression_l148_148384


namespace total_cost_for_gym_memberships_l148_148545

def cheap_gym_monthly_fee : ℕ := 10
def cheap_gym_signup_fee : ℕ := 50
def expensive_gym_factor : ℕ := 3
def expensive_gym_signup_factor : ℕ := 4
def months_in_year : ℕ := 12

theorem total_cost_for_gym_memberships :
  let cheap_gym_annual_cost := months_in_year * cheap_gym_monthly_fee + cheap_gym_signup_fee in
  let expensive_gym_monthly_fee := expensive_gym_factor * cheap_gym_monthly_fee in
  let expensive_gym_annual_cost := months_in_year * expensive_gym_monthly_fee + expensive_gym_signup_factor * expensive_gym_monthly_fee in
  cheap_gym_annual_cost + expensive_gym_annual_cost = 650 :=
by
  sorry

end total_cost_for_gym_memberships_l148_148545


namespace gcd_9157_2695_eq_1_l148_148325

theorem gcd_9157_2695_eq_1 : Int.gcd 9157 2695 = 1 := 
by
  sorry

end gcd_9157_2695_eq_1_l148_148325


namespace probability_even_product_l148_148302

-- Define spinner A and spinner C
def SpinnerA : List ℕ := [1, 2, 3, 4]
def SpinnerC : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define even and odd number sets for Spinner A and Spinner C
def evenNumbersA : List ℕ := [2, 4]
def oddNumbersA : List ℕ := [1, 3]

def evenNumbersC : List ℕ := [2, 4, 6]
def oddNumbersC : List ℕ := [1, 3, 5]

-- Define a function to check if a product is even
def isEven (n : ℕ) : Bool := n % 2 == 0

-- Probability calculation
def evenProductProbability : ℚ :=
  let totalOutcomes := (SpinnerA.length * SpinnerC.length)
  let evenA_outcomes := (evenNumbersA.length * SpinnerC.length)
  let oddA_evenC_outcomes := (oddNumbersA.length * evenNumbersC.length)
  (evenA_outcomes + oddA_evenC_outcomes) / totalOutcomes

theorem probability_even_product :
  evenProductProbability = 3 / 4 :=
by
  sorry

end probability_even_product_l148_148302


namespace six_digit_numbers_with_at_least_one_zero_is_368559_l148_148718

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_is_368559_l148_148718


namespace reduced_price_l148_148944

theorem reduced_price (P : ℝ) (hP : P = 56)
    (original_qty : ℝ := 800 / P)
    (reduced_qty : ℝ := 800 / (0.65 * P))
    (diff_qty : ℝ := reduced_qty - original_qty)
    (difference_condition : diff_qty = 5) :
  0.65 * P = 36.4 :=
by
  rw [hP]
  sorry

end reduced_price_l148_148944


namespace zoe_total_cost_l148_148606

theorem zoe_total_cost 
  (app_cost : ℕ)
  (monthly_cost : ℕ)
  (item_cost : ℕ)
  (feature_cost : ℕ)
  (months_played : ℕ)
  (h1 : app_cost = 5)
  (h2 : monthly_cost = 8)
  (h3 : item_cost = 10)
  (h4 : feature_cost = 12)
  (h5 : months_played = 2) :
  app_cost + (months_played * monthly_cost) + item_cost + feature_cost = 43 := 
by 
  sorry

end zoe_total_cost_l148_148606


namespace equal_sundays_tuesdays_days_l148_148955

-- Define the problem in Lean
def num_equal_sundays_and_tuesdays_starts : ℕ :=
  3

-- Define a function that calculates the number of starting days that result in equal Sundays and Tuesdays
def calculate_sundays_tuesdays_starts (days_in_month : ℕ) : ℕ :=
  if days_in_month = 30 then 3 else 0

-- Prove that for a month of 30 days, there are 3 valid starting days for equal Sundays and Tuesdays
theorem equal_sundays_tuesdays_days :
  calculate_sundays_tuesdays_starts 30 = num_equal_sundays_and_tuesdays_starts :=
by 
  -- Proof outline here
  sorry

end equal_sundays_tuesdays_days_l148_148955


namespace expression_result_l148_148783

-- We define the mixed number fractions as conditions
def mixed_num_1 := 2 + 1 / 2         -- 2 1/2
def mixed_num_2 := 3 + 1 / 3         -- 3 1/3
def mixed_num_3 := 4 + 1 / 4         -- 4 1/4
def mixed_num_4 := 1 + 1 / 6         -- 1 1/6

-- Here are their improper fractions
def improper_fraction_1 := 5 / 2     -- (2 + 1/2) converted to improper fraction
def improper_fraction_2 := 10 / 3    -- (3 + 1/3) converted to improper fraction
def improper_fraction_3 := 17 / 4    -- (4 + 1/4) converted to improper fraction
def improper_fraction_4 := 7 / 6     -- (1 + 1/6) converted to improper fraction

-- Define the problematic expression
def expression := (improper_fraction_1 - improper_fraction_2)^2 / (improper_fraction_3 + improper_fraction_4)

-- Statement of the simplified result
theorem expression_result : expression = 5 / 39 :=
by
  sorry

end expression_result_l148_148783


namespace describe_random_event_l148_148005

def idiom_A : Prop := "海枯石烂" = "extremely improbable or far into the future, not random"
def idiom_B : Prop := "守株待兔" = "represents a random event"
def idiom_C : Prop := "画饼充饥" = "unreal hopes, not random"
def idiom_D : Prop := "瓜熟蒂落" = "natural or expected outcome, not random"

theorem describe_random_event : idiom_B := 
by
  -- Proof omitted; conclusion follows from the given definitions
  sorry

end describe_random_event_l148_148005


namespace unique_solution_set_l148_148524

theorem unique_solution_set :
  {a : ℝ | ∃ x : ℝ, (x+a)/(x^2-1) = 1 ∧ 
                    (∀ y : ℝ, (y+a)/(y^2-1) = 1 → y = x)} 
  = {-1, 1, -5/4} :=
sorry

end unique_solution_set_l148_148524


namespace remaining_budget_after_purchases_l148_148432

theorem remaining_budget_after_purchases :
  let budget := 80
  let fried_chicken_cost := 12
  let beef_cost_per_pound := 3
  let beef_quantity := 4.5
  let soup_cost_per_can := 2
  let soup_quantity := 3
  let milk_original_price := 4
  let milk_discount := 0.10
  let beef_cost := beef_quantity * beef_cost_per_pound
  let paid_soup_quantity := soup_quantity / 2
  let milk_discounted_price := milk_original_price * (1 - milk_discount)
  let total_cost := fried_chicken_cost + beef_cost + (paid_soup_quantity * soup_cost_per_can) + milk_discounted_price
  let remaining_budget := budget - total_cost
  remaining_budget = 47.90 :=
by
  sorry

end remaining_budget_after_purchases_l148_148432


namespace mean_and_variance_of_y_l148_148522

noncomputable def mean (xs : List ℝ) : ℝ :=
  if h : xs.length > 0 then xs.sum / xs.length else 0

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  if h : xs.length > 0 then (xs.map (λ x => (x - m)^2)).sum / xs.length else 0

theorem mean_and_variance_of_y
  (x : List ℝ)
  (hx_len : x.length = 20)
  (hx_mean : mean x = 1)
  (hx_var : variance x = 8) :
  let y := x.map (λ xi => 2 * xi + 3)
  mean y = 5 ∧ variance y = 32 :=
by
  let y := x.map (λ xi => 2 * xi + 3)
  sorry

end mean_and_variance_of_y_l148_148522


namespace quadratic_eqn_a_range_l148_148882

variable {a : ℝ}

theorem quadratic_eqn_a_range (a : ℝ) : (∃ x : ℝ, (a - 3) * x^2 - 4 * x + 1 = 0) ↔ a ≠ 3 :=
by sorry

end quadratic_eqn_a_range_l148_148882


namespace max_sum_of_four_distinct_with_lcm_165_l148_148791

theorem max_sum_of_four_distinct_with_lcm_165 (a b c d : ℕ)
  (h1 : Nat.lcm a b = 165)
  (h2 : Nat.lcm a c = 165)
  (h3 : Nat.lcm a d = 165)
  (h4 : Nat.lcm b c = 165)
  (h5 : Nat.lcm b d = 165)
  (h6 : Nat.lcm c d = 165)
  (h7 : a ≠ b) (h8 : a ≠ c) (h9 : a ≠ d)
  (h10 : b ≠ c) (h11 : b ≠ d) (h12 : c ≠ d) :
  a + b + c + d ≤ 268 := sorry

end max_sum_of_four_distinct_with_lcm_165_l148_148791


namespace polynomial_divisibility_condition_l148_148507

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^5 - x^4 + x^3 - p * x^2 + q * x - 6

theorem polynomial_divisibility_condition (p q : ℝ) :
  (f (-1) p q = 0) ∧ (f 2 p q = 0) → 
  (p = 0) ∧ (q = -9) := by
  sorry

end polynomial_divisibility_condition_l148_148507


namespace triangular_number_30_eq_465_perimeter_dots_30_eq_88_l148_148029

-- Definition of the 30th triangular number
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition of the perimeter dots for the triangular number
def perimeter_dots (n : ℕ) : ℕ := n + 2 * (n - 1)

-- Theorem to prove the 30th triangular number is 465
theorem triangular_number_30_eq_465 : triangular_number 30 = 465 := by
  sorry

-- Theorem to prove the perimeter dots for the 30th triangular number is 88
theorem perimeter_dots_30_eq_88 : perimeter_dots 30 = 88 := by
  sorry

end triangular_number_30_eq_465_perimeter_dots_30_eq_88_l148_148029


namespace angle_covered_in_three_layers_l148_148337

/-- Define the conditions: A 90-degree angle, sum of angles is 290 degrees,
    and prove the angle covered in three layers is 110 degrees. -/
theorem angle_covered_in_three_layers {α β : ℝ}
  (h1 : α + β = 90)
  (h2 : 2*α + 3*β = 290) :
  β = 110 := 
sorry

end angle_covered_in_three_layers_l148_148337


namespace evaluate_g_at_neg3_l148_148166

def g (x : ℤ) : ℤ := x^2 - x + 2 * x^3

theorem evaluate_g_at_neg3 : g (-3) = -42 := by
  sorry

end evaluate_g_at_neg3_l148_148166


namespace average_speed_palindrome_trip_l148_148544

theorem average_speed_palindrome_trip :
  ∀ (initial final : ℕ) (time : ℝ),
    initial = 13431 → final = 13531 → time = 3 →
    (final - initial) / time = 33 :=
by
  intros initial final time h_initial h_final h_time
  rw [h_initial, h_final, h_time]
  norm_num
  sorry

end average_speed_palindrome_trip_l148_148544


namespace cost_of_tree_planting_l148_148299

theorem cost_of_tree_planting 
  (initial_temp final_temp : ℝ) (temp_drop_per_tree cost_per_tree : ℝ) 
  (h_initial: initial_temp = 80) (h_final: final_temp = 78.2) 
  (h_temp_drop_per_tree: temp_drop_per_tree = 0.1) 
  (h_cost_per_tree: cost_per_tree = 6) : 
  (final_temp - initial_temp) / temp_drop_per_tree * cost_per_tree = 108 := 
by
  sorry

end cost_of_tree_planting_l148_148299


namespace dodecahedron_diagonals_l148_148030

-- Define a structure representing a dodecahedron with its properties
structure Dodecahedron where
  faces : Nat
  vertices : Nat
  faces_meeting_at_each_vertex : Nat

-- Concretely define a dodecahedron based on the given problem properties
def dodecahedron_example : Dodecahedron :=
  { faces := 12,
    vertices := 20,
    faces_meeting_at_each_vertex := 3 }

-- Lean statement to prove the number of interior diagonals in a dodecahedron
theorem dodecahedron_diagonals (d : Dodecahedron) (h : d = dodecahedron_example) : 
  (d.vertices * (d.vertices - d.faces_meeting_at_each_vertex) / 2) = 160 := by
  rw [h]
  -- Even though we skip the proof, Lean should recognize the transformation
  sorry

end dodecahedron_diagonals_l148_148030


namespace find_books_second_shop_l148_148155

def total_books (books_first_shop books_second_shop : ℕ) : ℕ :=
  books_first_shop + books_second_shop

def total_cost (cost_first_shop cost_second_shop : ℕ) : ℕ :=
  cost_first_shop + cost_second_shop

def average_price (total_cost total_books : ℕ) : ℕ :=
  total_cost / total_books

theorem find_books_second_shop : 
  ∀ (books_first_shop cost_first_shop cost_second_shop : ℕ),
    books_first_shop = 65 →
    cost_first_shop = 1480 →
    cost_second_shop = 920 →
    average_price (total_cost cost_first_shop cost_second_shop) (total_books books_first_shop (2400 / 20 - 65)) = 20 →
    2400 / 20 - 65 = 55 := 
by sorry

end find_books_second_shop_l148_148155


namespace probability_eight_or_more_stay_l148_148417

noncomputable def probability_at_least_8_stay : ℚ :=
  let n := 10 in
  let certain := 5 in
  let uncertain := 5 in
  let p_stay := 3 / 7 in
  let combinations := 10 * (p_stay ^ 3) * ((1 - p_stay) ^ 2) + (p_stay ^ 5) in
  combinations

theorem probability_eight_or_more_stay :
  probability_at_least_8_stay = 4563 / 16807 :=
by
  unfold probability_at_least_8_stay
  norm_num
  sorry

end probability_eight_or_more_stay_l148_148417


namespace shaded_area_percentage_l148_148938

def area_square (side : ℕ) : ℕ := side * side

def shaded_percentage (total_area shaded_area : ℕ) : ℚ :=
  ((shaded_area : ℚ) / total_area) * 100 

theorem shaded_area_percentage (side : ℕ) (total_area : ℕ) (shaded_area : ℕ) 
  (h_side : side = 7) (h_total_area : total_area = area_square side) 
  (h_shaded_area : shaded_area = 4 + 16 + 13) : 
  shaded_percentage total_area shaded_area = 3300 / 49 :=
by
  -- The proof will go here
  sorry

end shaded_area_percentage_l148_148938


namespace correct_calculation_l148_148181

theorem correct_calculation :
  (-2 * a * b^2)^3 = -8 * a^3 * b^6 :=
by sorry

end correct_calculation_l148_148181


namespace sin_alpha_value_l148_148238

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α - Real.pi / 4) = (7 * Real.sqrt 2) / 10)
  (h2 : Real.cos (2 * α) = 7 / 25) : 
  Real.sin α = 3 / 5 :=
sorry

end sin_alpha_value_l148_148238


namespace value_of_a_plus_d_l148_148102

variable (a b c d : ℝ)

theorem value_of_a_plus_d (h1 : a + b = 4) (h2 : b + c = 7) (h3 : c + d = 5) : a + d = 4 :=
sorry

end value_of_a_plus_d_l148_148102


namespace computation_l148_148360

theorem computation :
  52 * 46 + 104 * 52 = 7800 := by
  sorry

end computation_l148_148360


namespace find_numbers_l148_148569

theorem find_numbers (S P : ℝ) (h : S^2 - 4 * P ≥ 0) :
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧
             ((x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨
              (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2)) :=
by
  sorry

end find_numbers_l148_148569


namespace relationship_among_a_b_c_l148_148769

noncomputable def a := Real.sqrt 0.5
noncomputable def b := Real.sqrt 0.3
noncomputable def c := Real.log 0.2 / Real.log 0.3

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l148_148769


namespace michael_lost_at_least_800_l148_148330

theorem michael_lost_at_least_800 
  (T F : ℕ) 
  (h1 : T + F = 15) 
  (h2 : T = F + 1 ∨ T = F - 1) 
  (h3 : 10 * T + 50 * F = 1270) : 
  1270 - (10 * T + 50 * F) = 800 :=
by
  sorry

end michael_lost_at_least_800_l148_148330


namespace proof_of_area_weighted_sum_of_distances_l148_148622

def area_weighted_sum_of_distances
  (a b a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) 
  (t1 t2 t3 t4 t : ℝ)
  (z1 z2 z3 z4 z : ℝ) 
  (h1 : t1 = a1 * b1)
  (h2 : t2 = (a - a1) * b1)
  (h3 : t3 = a3 * b3)
  (h4 : t4 = (a - a3) * b3)
  (rect_area : t = a * b)
  : Prop :=
  t1 * z1 + t2 * z2 + t3 * z3 + t4 * z4 = t * z

theorem proof_of_area_weighted_sum_of_distances
  (a b a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (t1 t2 t3 t4 t : ℝ)
  (z1 z2 z3 z4 z : ℝ)
  (h1 : t1 = a1 * b1)
  (h2 : t2 = (a - a1) * b1)
  (h3 : t3 = a3 * b3)
  (h4 : t4 = (a - a3) * b3)
  (rect_area : t = a * b)
  : area_weighted_sum_of_distances a b a1 a2 a3 a4 b1 b2 b3 b4 t1 t2 t3 t4 t z1 z2 z3 z4 z h1 h2 h3 h4 rect_area :=
  sorry

end proof_of_area_weighted_sum_of_distances_l148_148622


namespace find_numbers_l148_148574

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l148_148574


namespace shaded_ratio_l148_148304

theorem shaded_ratio (full_rectangles half_rectangles : ℕ) (n m : ℕ) (rectangle_area shaded_area total_area : ℝ)
  (h1 : n = 4) (h2 : m = 5) (h3 : rectangle_area = n * m) 
  (h4 : full_rectangles = 3) (h5 : half_rectangles = 4)
  (h6 : shaded_area = full_rectangles * 1 + 0.5 * half_rectangles * 1)
  (h7 : total_area = rectangle_area) :
  shaded_area / total_area = 1 / 4 := by
  sorry

end shaded_ratio_l148_148304


namespace product_of_good_numbers_does_not_imply_sum_digits_property_l148_148289

-- Define what it means for a number to be "good".
def is_good (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

-- Define the sum of the digits of a number
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem statement
theorem product_of_good_numbers_does_not_imply_sum_digits_property :
  ∀ (A B : ℕ), is_good A → is_good B → is_good (A * B) →
  ¬ (sum_digits (A * B) = sum_digits A * sum_digits B) :=
by
  intros A B hA hB hAB
  -- The detailed proof is not provided here, hence we use sorry to skip it.
  sorry

end product_of_good_numbers_does_not_imply_sum_digits_property_l148_148289


namespace repeating_decimal_as_fraction_l148_148377

noncomputable def repeating_decimal_to_fraction : ℚ :=
  let x := (47 : ℕ) * (1 / (10 ^ 2 - 1)) in
  x

theorem repeating_decimal_as_fraction (x : ℚ) (hx : x = 0.474747474747...) : 
x = 47/99 :=
begin
  sorry
end

end repeating_decimal_as_fraction_l148_148377


namespace eugene_total_cost_l148_148267

variable (TshirtCost PantCost ShoeCost : ℕ)
variable (NumTshirts NumPants NumShoes Discount : ℕ)

theorem eugene_total_cost
  (hTshirtCost : TshirtCost = 20)
  (hPantCost : PantCost = 80)
  (hShoeCost : ShoeCost = 150)
  (hNumTshirts : NumTshirts = 4)
  (hNumPants : NumPants = 3)
  (hNumShoes : NumShoes = 2)
  (hDiscount : Discount = 10) :
  TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes - (TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes) * Discount / 100 = 558 := by
  sorry

end eugene_total_cost_l148_148267


namespace y_intercept_of_tangent_line_l148_148105

def point (x y : ℝ) : Prop := true

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 + 4*x - 2*y + 3

theorem y_intercept_of_tangent_line :
  ∃ m b : ℝ,
  (∀ x : ℝ, circle_eq x (m*x + b) = 0 → m * m = 1) ∧
  (∃ P: ℝ × ℝ, P = (-1, 0)) ∧
  ∀ b : ℝ, (∃ m : ℝ, m = 1 ∧ (∃ P: ℝ × ℝ, P = (-1, 0)) ∧ b = 1) := 
sorry

end y_intercept_of_tangent_line_l148_148105


namespace total_pears_l148_148213

theorem total_pears (Alyssa_picked Nancy_picked : ℕ) (h₁ : Alyssa_picked = 42) (h₂ : Nancy_picked = 17) : Alyssa_picked + Nancy_picked = 59 :=
by
  sorry

end total_pears_l148_148213


namespace integral_converges_l148_148150

open Real

theorem integral_converges :
  ∃ (I : ℝ), tendsto (λ (b : ℝ), ∫ x in 1..b, (x * cos x) / ((1 + x^2) * sqrt (4 + x^2))) at_top (nhds I) :=
  sorry

end integral_converges_l148_148150


namespace total_items_and_cost_per_pet_l148_148421

theorem total_items_and_cost_per_pet
  (treats_Jane : ℕ)
  (treats_Wanda : ℕ := treats_Jane / 2)
  (bread_Jane : ℕ := (3 * treats_Jane) / 4)
  (bread_Wanda : ℕ := 90)
  (bread_Carla : ℕ := 40)
  (treats_Carla : ℕ := 5 * bread_Carla / 2)
  (items_Peter : ℕ := 140)
  (treats_Peter : ℕ := items_Peter / 3)
  (bread_Peter : ℕ := 2 * treats_Peter)
  (x y z : ℕ) :
  (∀ B : ℕ, B = bread_Jane + bread_Wanda + bread_Carla + bread_Peter) ∧
  (∀ T : ℕ, T = treats_Jane + treats_Wanda + treats_Carla + treats_Peter) ∧
  (∀ Total : ℕ, Total = (bread_Jane + bread_Wanda + bread_Carla + bread_Peter) + (treats_Jane + treats_Wanda + treats_Carla + treats_Peter)) ∧
  (∀ ExpectedTotal : ℕ, ExpectedTotal = 427) ∧
  (∀ Cost : ℕ, Cost = (bread_Jane + bread_Wanda + bread_Carla + bread_Peter) * x + (treats_Jane + treats_Wanda + treats_Carla + treats_Peter) * y) ∧
  (∀ CostPerPet : ℕ, CostPerPet = Cost / z) ∧
  (B + T = 427) ∧
  ((Cost / z) = (235 * x + 192 * y) / z)
:=
  by
  sorry

end total_items_and_cost_per_pet_l148_148421


namespace extra_fruits_l148_148170

theorem extra_fruits (r g s : Nat) (hr : r = 42) (hg : g = 7) (hs : s = 9) : r + g - s = 40 :=
by
  sorry

end extra_fruits_l148_148170


namespace six_digit_numbers_with_zero_l148_148739

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148739


namespace friends_total_candies_l148_148305

noncomputable def total_candies (T S J C V B : ℕ) : ℕ :=
  T + S + J + C + V + B

theorem friends_total_candies :
  let T := 22
  let S := 16
  let J := T / 2
  let C := 2 * S
  let V := J + S
  let B := (T + C) / 2 + 9
  total_candies T S J C V B = 144 := by
  sorry

end friends_total_candies_l148_148305


namespace max_min_value_of_a_l148_148015

theorem max_min_value_of_a 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 3) 
  (h2 : a^2 + 2 * b^2 + 3 * c^2 + 6 * d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := 
sorry

end max_min_value_of_a_l148_148015


namespace average_weight_increase_l148_148263

theorem average_weight_increase (n : ℕ) (w_old w_new : ℝ) (h1 : n = 9) (h2 : w_old = 65) (h3 : w_new = 87.5) :
  (w_new - w_old) / n = 2.5 :=
by
  rw [h1, h2, h3]
  norm_num

end average_weight_increase_l148_148263


namespace age_difference_l148_148592

variable (A B C : ℕ)

theorem age_difference (h₁ : C = A - 20) : (A + B) = (B + C) + 20 := 
sorry

end age_difference_l148_148592


namespace circle_radius_l148_148821

theorem circle_radius (M N : ℝ) (hM : M = Real.pi * r ^ 2) (hN : N = 2 * Real.pi * r) (h : M / N = 15) : r = 30 := by
  sorry

end circle_radius_l148_148821


namespace exists_prime_and_positive_integer_l148_148226

theorem exists_prime_and_positive_integer (a : ℕ) (h : a = 9) : 
  ∃ (p : ℕ) (hp : Nat.Prime p) (b : ℕ) (hb : b ≥ 2), (a^p - a) / p = b^2 := 
  by
  sorry

end exists_prime_and_positive_integer_l148_148226


namespace total_production_l148_148463

variable (x : ℕ) -- total units produced by 4 machines in 6 days
variable (R : ℕ) -- rate of production per machine per day

-- Condition 1: 4 machines can produce x units in 6 days
axiom rate_definition : 4 * R * 6 = x

-- Question: Prove the total amount of product produced by 16 machines in 3 days is 2x
theorem total_production : 16 * R * 3 = 2 * x :=
by 
  sorry

end total_production_l148_148463


namespace remainder_when_divided_by_x_plus_2_l148_148604

-- Define the polynomial q(x)
def q (M N D x : ℝ) : ℝ := M * x^4 + N * x^2 + D * x - 5

-- Define the given conditions
def cond1 (M N D : ℝ) : Prop := q M N D 2 = 15

-- The theorem statement we want to prove
theorem remainder_when_divided_by_x_plus_2 (M N D : ℝ) (h1 : cond1 M N D) : q M N D (-2) = 15 :=
sorry

end remainder_when_divided_by_x_plus_2_l148_148604


namespace Shawna_situps_l148_148156

theorem Shawna_situps :
  ∀ (goal_per_day : ℕ) (total_days : ℕ) (tuesday_situps : ℕ) (wednesday_situps : ℕ),
  goal_per_day = 30 →
  total_days = 3 →
  tuesday_situps = 19 →
  wednesday_situps = 59 →
  (goal_per_day * total_days) - (tuesday_situps + wednesday_situps) = 12 :=
by
  intros goal_per_day total_days tuesday_situps wednesday_situps
  sorry

end Shawna_situps_l148_148156


namespace factorize_difference_of_squares_l148_148046

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l148_148046


namespace solve_system_l148_148918

def inequality1 (x : ℝ) : Prop := 5 / (x + 3) ≥ 1

def inequality2 (x : ℝ) : Prop := x^2 + x - 2 ≥ 0

def solution (x : ℝ) : Prop := (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)

theorem solve_system (x : ℝ) : inequality1 x ∧ inequality2 x → solution x := by
  sorry

end solve_system_l148_148918


namespace find_mistaken_divisor_l148_148114

-- Define the conditions
def remainder : ℕ := 0
def quotient_correct : ℕ := 32
def divisor_correct : ℕ := 21
def quotient_mistaken : ℕ := 56
def dividend : ℕ := quotient_correct * divisor_correct + remainder

-- Prove the mistaken divisor
theorem find_mistaken_divisor : ∃ x : ℕ, dividend = quotient_mistaken * x + remainder ∧ x = 12 :=
by
  -- We leave this as an exercise to the prover
  sorry

end find_mistaken_divisor_l148_148114


namespace vector_problem_l148_148659

open Real

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  (v.1 ^ 2 + v.2 ^ 2) ^ (1 / 2)

variables (a b : ℝ × ℝ)
variables (h1 : a ≠ (0, 0)) (h2 : b ≠ (0, 0))
variables (h3 : a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0)
variables (h4 : 2 * magnitude a = magnitude b) (h5 : magnitude b =2)

theorem vector_problem : magnitude (2 * a.1 - b.1, 2 * a.2 - b.2) = 2 :=
sorry

end vector_problem_l148_148659


namespace solve_polynomial_l148_148982

theorem solve_polynomial (z : ℂ) :
    z^5 - 5 * z^3 + 6 * z = 0 ↔ 
    z = 0 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = -Real.sqrt 3 ∨ z = Real.sqrt 3 := 
by 
  sorry

end solve_polynomial_l148_148982


namespace ball_radius_and_surface_area_l148_148014

theorem ball_radius_and_surface_area (d h : ℝ) (r : ℝ) :
  d = 12 ∧ h = 2 ∧ (6^2 + (r - h)^2 = r^2) → (r = 10 ∧ 4 * Real.pi * r^2 = 400 * Real.pi) := by
  sorry

end ball_radius_and_surface_area_l148_148014


namespace interest_rate_condition_l148_148413

theorem interest_rate_condition 
    (P1 P2 : ℝ) 
    (R2 : ℝ) 
    (T1 T2 : ℝ) 
    (SI500 SI160 : ℝ) 
    (H1: SI500 = (P1 * R2 * T1) / 100) 
    (H2: SI160 = (P2 * (25 / 100))):
  25 * (160 / 100) / 12.5  = 6.4 :=
by
  sorry

end interest_rate_condition_l148_148413


namespace compute_expression_l148_148969
-- Import the standard math library to avoid import errors.

-- Define the theorem statement based on the given conditions and the correct answer.
theorem compute_expression :
  (75 * 2424 + 25 * 2424) / 2 = 121200 :=
by
  sorry

end compute_expression_l148_148969


namespace sufficient_not_necessary_condition_l148_148225

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, (x^2 - 2 * x < 0 → 0 < x ∧ x < 4)) ∧ (∃ x : ℝ, (0 < x ∧ x < 4) ∧ ¬ (x^2 - 2 * x < 0)) :=
by
  sorry

end sufficient_not_necessary_condition_l148_148225


namespace min_value_of_sum_squares_l148_148135

theorem min_value_of_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
    x^2 + y^2 + z^2 ≥ 10 :=
sorry

end min_value_of_sum_squares_l148_148135


namespace factorize_difference_of_squares_l148_148063

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l148_148063


namespace seventh_graders_count_l148_148316

-- Define the problem conditions
def total_students (T : ℝ) : Prop := 0.38 * T = 76
def seventh_grade_ratio : ℝ := 0.32
def seventh_graders (S : ℝ) (T : ℝ) : Prop := S = seventh_grade_ratio * T

-- The goal statement
theorem seventh_graders_count {T S : ℝ} (h : total_students T) : seventh_graders S T → S = 64 :=
by
  sorry

end seventh_graders_count_l148_148316


namespace james_vacuuming_hours_l148_148119

/-- James spends some hours vacuuming and 3 times as long on the rest of his chores. 
    He spends 12 hours on his chores in total. -/
theorem james_vacuuming_hours (V : ℝ) (h : V + 3 * V = 12) : V = 3 := 
sorry

end james_vacuuming_hours_l148_148119


namespace white_black_arrangements_l148_148233

theorem white_black_arrangements (W B : ℕ) (hW : W = 5) (hB : B = 10) :
  ∃ n, n = (nat.choose (B) (W)) ∧ n = 252 :=
by
  sorry

end white_black_arrangements_l148_148233


namespace eugene_total_cost_l148_148268

variable (TshirtCost PantCost ShoeCost : ℕ)
variable (NumTshirts NumPants NumShoes Discount : ℕ)

theorem eugene_total_cost
  (hTshirtCost : TshirtCost = 20)
  (hPantCost : PantCost = 80)
  (hShoeCost : ShoeCost = 150)
  (hNumTshirts : NumTshirts = 4)
  (hNumPants : NumPants = 3)
  (hNumShoes : NumShoes = 2)
  (hDiscount : Discount = 10) :
  TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes - (TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes) * Discount / 100 = 558 := by
  sorry

end eugene_total_cost_l148_148268


namespace average_children_with_children_l148_148502

theorem average_children_with_children (total_families : ℕ) (avg_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 → avg_children_per_family = 3 → childless_families = 3 →
  (45 / (total_families - childless_families) : ℚ) = 3.75 :=
by
  intros h1 h2 h3
  have total_children : ℕ := 45
  have families_with_children : ℕ := total_families - childless_families
  have avg_children : ℚ := (total_children : ℚ) / families_with_children
  exact eq_of_sub_eq_zero (by norm_num : avg_children - 3.75 = 0)

end average_children_with_children_l148_148502


namespace distance_from_point_to_focus_l148_148658

theorem distance_from_point_to_focus (x0 : ℝ) (h1 : (2 * Real.sqrt 3)^2 = 4 * x0) :
    x0 + 1 = 4 := by
  sorry

end distance_from_point_to_focus_l148_148658


namespace six_digit_numbers_with_zero_l148_148733

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148733


namespace cube_problem_l148_148208

theorem cube_problem (n : ℕ) (H1 : 6 * n^2 = 1 / 3 * 6 * n^3) : n = 3 :=
sorry

end cube_problem_l148_148208


namespace total_first_year_students_400_l148_148353

theorem total_first_year_students_400 (N : ℕ) (A B C : ℕ) 
  (h1 : A = 80) 
  (h2 : B = 100) 
  (h3 : C = 20) 
  (h4 : A * B = C * N) : 
  N = 400 :=
sorry

end total_first_year_students_400_l148_148353


namespace other_root_of_quadratic_l148_148999

theorem other_root_of_quadratic (m : ℝ) :
  (∃ t : ℝ, (x^2 + m * x - 20 = 0) ∧ (x = -4 ∨ x = t)) → (t = 5) :=
by
  sorry

end other_root_of_quadratic_l148_148999


namespace value_of_k_odd_function_range_of_m_value_of_k_even_function_value_of_n_l148_148075

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := exp x + (k - 2) * exp (-x)

-- Part (1) Condition (1.1)
theorem value_of_k_odd_function :
  (∀ x : ℝ, f (-x) 1 = -f x 1) → (k = 1)


-- Part (1) Condition (1.2)
theorem range_of_m (m : ℝ) :
  ( ∀ x > 1, m * f x 1 - f (2 * x) 1 - 2 * exp (-2 * x) - 10 < 0) → m < 4 * real.sqrt 3


-- Part (2) Condition (2.1)
theorem value_of_k_even_function :
  (∀ x : ℝ, f (-x) 3 = f x 3) → (k = 3)


-- Part (2) Condition (2.2)
noncomputable def g (x : ℝ) : ℝ := real.log (f x 3) / real.log 2 

noncomputable def h (x n : ℝ) : ℝ := (g x - 1 + n) * (2 * n + 1 - g x) + n ^ 2 - n

theorem value_of_n (n : ℝ) :
  (∀ x : ℝ, h x n = 0 → (n ≤ 0 ∨ n ≥ 4 / 13)) 

end value_of_k_odd_function_range_of_m_value_of_k_even_function_value_of_n_l148_148075


namespace intersection_of_lines_l148_148068

theorem intersection_of_lines :
  ∃ (x y : ℝ), (8 * x + 5 * y = 40) ∧ (3 * x - 10 * y = 15) ∧ (x = 5) ∧ (y = 0) := 
by 
  sorry

end intersection_of_lines_l148_148068


namespace calculate_expression_l148_148634

theorem calculate_expression :
  2⁻¹ + (3 - Real.pi)^0 + abs (2 * Real.sqrt 3 - Real.sqrt 2) + 2 * Real.cos (Real.pi / 4) - Real.sqrt 12 = 3 / 2 :=
sorry

end calculate_expression_l148_148634


namespace six_digit_numbers_with_zero_count_l148_148679

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l148_148679


namespace six_digit_numbers_with_zero_l148_148700

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148700


namespace binomial_expansion_coefficient_l148_148881

theorem binomial_expansion_coefficient (a : ℝ)
  (h : ∃ r, 9 - 3 * r = 6 ∧ (-a)^r * (Nat.choose 9 r) = 36) :
  a = -4 :=
  sorry

end binomial_expansion_coefficient_l148_148881


namespace symmetry_of_transformed_graphs_l148_148250

variable (f : ℝ → ℝ)

theorem symmetry_of_transformed_graphs :
  (∀ x, f x = f (-x)) → (∀ x, f (1 + x) = f (1 - x)) :=
by
  intro h_symmetry
  intro x
  sorry

end symmetry_of_transformed_graphs_l148_148250


namespace rectangular_field_perimeter_l148_148786

theorem rectangular_field_perimeter (A L : ℝ) (h1 : A = 300) (h2 : L = 15) : 
  let W := A / L 
  let P := 2 * (L + W)
  P = 70 := by
  sorry

end rectangular_field_perimeter_l148_148786


namespace faster_speed_l148_148020

variable (v : ℝ)
variable (distance fasterDistance speed time : ℝ)
variable (h_distance : distance = 24)
variable (h_speed : speed = 4)
variable (h_fasterDistance : fasterDistance = distance + 6)
variable (h_time : time = distance / speed)

theorem faster_speed (h : 6 = fasterDistance / v) : v = 5 :=
by
  sorry

end faster_speed_l148_148020


namespace fractional_part_zero_l148_148394

noncomputable def fractional_part (z : ℝ) : ℝ := z - (⌊z⌋ : ℝ)

theorem fractional_part_zero (x : ℝ) :
  fractional_part (1 / 3 * (1 / 3 * (1 / 3 * x - 3) - 3) - 3) = 0 ↔ 
  ∃ k : ℤ, 27 * k + 9 ≤ x ∧ x < 27 * k + 18 :=
by
  sorry

end fractional_part_zero_l148_148394


namespace n_fifth_minus_n_divisible_by_30_l148_148315

theorem n_fifth_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end n_fifth_minus_n_divisible_by_30_l148_148315


namespace largest_two_digit_number_with_remainder_2_div_13_l148_148807

theorem largest_two_digit_number_with_remainder_2_div_13 : 
  ∃ (N : ℕ), (10 ≤ N ∧ N ≤ 99) ∧ N % 13 = 2 ∧ ∀ (M : ℕ), (10 ≤ M ∧ M ≤ 99) ∧ M % 13 = 2 → M ≤ N :=
  sorry

end largest_two_digit_number_with_remainder_2_div_13_l148_148807


namespace sum_first_ten_terms_arithmetic_l148_148128

def arithmetic_sequence_sum (a₁ : ℤ) (n : ℕ) (d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_ten_terms_arithmetic (a₁ a₂ a₆ d : ℤ) 
  (h1 : a₁ = -2) 
  (h2 : a₂ + a₆ = 2) 
  (common_diff : d = 1) :
  arithmetic_sequence_sum a₁ 10 d = 25 :=
by
  rw [h1, common_diff]
  sorry

end sum_first_ten_terms_arithmetic_l148_148128


namespace total_metal_rods_needed_l148_148824

def metal_rods_per_sheet : ℕ := 10
def sheets_per_panel : ℕ := 3
def metal_rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def panels : ℕ := 10

theorem total_metal_rods_needed : 
  (sheets_per_panel * metal_rods_per_sheet + beams_per_panel * metal_rods_per_beam) * panels = 380 :=
by
  exact rfl

end total_metal_rods_needed_l148_148824


namespace total_pokemon_cards_l148_148779

def initial_cards : Nat := 27
def received_cards : Nat := 41
def lost_cards : Nat := 20

theorem total_pokemon_cards : initial_cards + received_cards - lost_cards = 48 := by
  sorry

end total_pokemon_cards_l148_148779


namespace max_sum_of_ABC_l148_148890

/-- Theorem: The maximum value of A + B + C for distinct positive integers A, B, and C such that A * B * C = 2023 is 297. -/
theorem max_sum_of_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : A * B * C = 2023) :
  A + B + C ≤ 297 :=
sorry

end max_sum_of_ABC_l148_148890


namespace no_solutions_abs_eq_quadratic_l148_148067

theorem no_solutions_abs_eq_quadratic (x : ℝ) : ¬ (|x - 3| = x^2 + 2 * x + 4) := 
by
  sorry

end no_solutions_abs_eq_quadratic_l148_148067


namespace bumper_car_rides_l148_148028

-- Define the conditions
def rides_on_ferris_wheel : ℕ := 7
def cost_per_ride : ℕ := 5
def total_tickets : ℕ := 50

-- Formulate the statement to be proved
theorem bumper_car_rides : ∃ n : ℕ, 
  total_tickets = (rides_on_ferris_wheel * cost_per_ride) + (n * cost_per_ride) ∧ n = 3 :=
sorry

end bumper_car_rides_l148_148028


namespace solve_quadratic_l148_148563

theorem solve_quadratic (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ (x y : ℝ),
    (x + y = S) ∧
    (x * y = P) ∧
    ((x = (S + real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - real.sqrt (S^2 - 4 * P)) / 2) ∨ 
     (x = (S - real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + real.sqrt (S^2 - 4 * P)) / 2)) := 
sorry

end solve_quadratic_l148_148563


namespace BC_equals_expected_BC_l148_148993

def point := ℝ × ℝ -- Define a point as a pair of real numbers (coordinates).

def vector_sub (v1 v2 : point) : point := (v1.1 - v2.1, v1.2 - v2.2) -- Define vector subtraction.

-- Definitions of points A and B and vector AC
def A : point := (-1, 1)
def B : point := (0, 2)
def AC : point := (-2, 3)

-- Calculate vector AB
def AB : point := vector_sub B A

-- Calculate vector BC
def BC : point := vector_sub AC AB

-- Expected result
def expected_BC : point := (-3, 2)

-- Proof statement
theorem BC_equals_expected_BC : BC = expected_BC := by
  unfold BC AB AC A B vector_sub
  simp
  sorry

end BC_equals_expected_BC_l148_148993


namespace evaluate_expression_l148_148977

theorem evaluate_expression : (-3)^4 / 3^2 - 2^5 + 7^2 = 26 := by
  sorry

end evaluate_expression_l148_148977


namespace batsman_average_increase_l148_148817

def average_increase (avg_before : ℕ) (runs_12th_inning : ℕ) (avg_after : ℕ) : ℕ :=
  avg_after - avg_before

theorem batsman_average_increase :
  ∀ (avg_before runs_12th_inning avg_after : ℕ),
    (runs_12th_inning = 70) →
    (avg_after = 37) →
    (11 * avg_before + runs_12th_inning = 12 * avg_after) →
    average_increase avg_before runs_12th_inning avg_after = 3 :=
by
  intros avg_before runs_12th_inning avg_after h_runs h_avg_after h_total
  sorry

end batsman_average_increase_l148_148817


namespace tan_alpha_l148_148515

theorem tan_alpha (α : ℝ) (h1 : Real.sin (Real.pi - α) = 1 / 3) (h2 : Real.sin (2 * α) > 0) : 
  Real.tan α = Real.sqrt 2 / 4 :=
by 
  sorry

end tan_alpha_l148_148515


namespace vegetarian_family_member_count_l148_148485

variable (total_family : ℕ) (vegetarian_only : ℕ) (non_vegetarian_only : ℕ)
variable (both_vegetarian_nonvegetarian : ℕ) (vegan_only : ℕ)
variable (pescatarian : ℕ) (specific_vegetarian : ℕ)

theorem vegetarian_family_member_count :
  total_family = 35 →
  vegetarian_only = 11 →
  non_vegetarian_only = 6 →
  both_vegetarian_nonvegetarian = 9 →
  vegan_only = 3 →
  pescatarian = 4 →
  specific_vegetarian = 2 →
  vegetarian_only + both_vegetarian_nonvegetarian + vegan_only + pescatarian + specific_vegetarian = 29 :=
by
  intros
  sorry

end vegetarian_family_member_count_l148_148485


namespace six_digit_numbers_with_zero_l148_148697

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148697


namespace sufficient_but_not_necessary_l148_148990

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : (a^2 + b^2 < 1) → (ab + 1 > a + b) ∧ ¬(ab + 1 > a + b ↔ a^2 + b^2 < 1) := 
sorry

end sufficient_but_not_necessary_l148_148990


namespace euler_polyhedron_problem_l148_148228

theorem euler_polyhedron_problem
  (V E F : ℕ)
  (t h T H : ℕ)
  (euler_formula : V - E + F = 2)
  (faces_count : F = 30)
  (tri_hex_faces : t + h = 30)
  (edges_equation : E = (3 * t + 6 * h) / 2)
  (vertices_equation1 : V = (3 * t) / T)
  (vertices_equation2 : V = (6 * h) / H)
  (T_val : T = 1)
  (H_val : H = 2)
  (t_val : t = 10)
  (h_val : h = 20)
  (edges_val : E = 75)
  (vertices_val : V = 60) :
  100 * H + 10 * T + V = 270 :=
by
  sorry

end euler_polyhedron_problem_l148_148228


namespace j_h_five_l148_148747

-- Define the functions h and j
def h (x : ℤ) : ℤ := 4 * x + 5
def j (x : ℤ) : ℤ := 6 * x - 11

-- State the theorem to prove j(h(5)) = 139
theorem j_h_five : j (h 5) = 139 := by
  sorry

end j_h_five_l148_148747


namespace remainder_when_b_divided_by_23_l148_148429

theorem remainder_when_b_divided_by_23 :
  let b := (((13⁻¹ : ZMod 23) + (17⁻¹ : ZMod 23) + (19⁻¹ : ZMod 23))⁻¹ : ZMod 23)
  in b = 8 := by
{
  -- Proof omitted
  sorry
}

end remainder_when_b_divided_by_23_l148_148429


namespace corrected_mean_is_40_point_6_l148_148188

theorem corrected_mean_is_40_point_6 
  (mean_original : ℚ) (num_observations : ℕ) (wrong_observation : ℚ) (correct_observation : ℚ) :
  num_observations = 50 → mean_original = 40 → wrong_observation = 15 → correct_observation = 45 →
  ((mean_original * num_observations + (correct_observation - wrong_observation)) / num_observations = 40.6 : Prop) :=
by intros; sorry

end corrected_mean_is_40_point_6_l148_148188


namespace total_sand_l148_148452

variable (capacity_per_bag : ℕ) (number_of_bags : ℕ)

theorem total_sand (h1 : capacity_per_bag = 65) (h2 : number_of_bags = 12) : capacity_per_bag * number_of_bags = 780 := by
  sorry

end total_sand_l148_148452


namespace number_of_valid_selections_l148_148847

theorem number_of_valid_selections : 
  ∃ combinations : Finset (Finset ℕ), 
    combinations = {
      {2, 6, 3, 5}, 
      {2, 6, 1, 7}, 
      {2, 4, 1, 5}, 
      {4, 1, 3}, 
      {6, 1, 5}, 
      {4, 6, 3, 7}, 
      {2, 4, 6, 5, 7}
    } ∧ combinations.card = 7 :=
by sorry

end number_of_valid_selections_l148_148847


namespace video_streaming_budget_l148_148637

theorem video_streaming_budget 
  (weekly_food_budget : ℕ) 
  (weeks : ℕ) 
  (total_food_budget : ℕ) 
  (rent : ℕ) 
  (phone : ℕ) 
  (savings_rate : ℝ)
  (total_savings : ℕ) 
  (total_expenses : ℕ) 
  (known_expenses: ℕ) 
  (total_spending : ℕ):
  weekly_food_budget = 100 →
  weeks = 4 →
  total_food_budget = weekly_food_budget * weeks →
  rent = 1500 →
  phone = 50 →
  savings_rate = 0.10 →
  total_savings = 198 →
  total_expenses = total_food_budget + rent + phone →
  total_spending = (total_savings : ℝ) / savings_rate →
  known_expenses = total_expenses →
  total_spending - known_expenses = 30 :=
by sorry

end video_streaming_budget_l148_148637


namespace Jimmy_earns_229_l148_148278

-- Definitions based on conditions from the problem
def number_of_type_A : ℕ := 5
def number_of_type_B : ℕ := 4
def number_of_type_C : ℕ := 3

def value_of_type_A : ℕ := 20
def value_of_type_B : ℕ := 30
def value_of_type_C : ℕ := 40

def discount_type_A : ℕ := 7
def discount_type_B : ℕ := 10
def discount_type_C : ℕ := 12

-- Calculation of the total amount Jimmy will earn
def total_earnings : ℕ :=
  let price_A := value_of_type_A - discount_type_A
  let price_B := value_of_type_B - discount_type_B
  let price_C := value_of_type_C - discount_type_C
  (number_of_type_A * price_A) +
  (number_of_type_B * price_B) +
  (number_of_type_C * price_C)

-- The statement to be proved
theorem Jimmy_earns_229 : total_earnings = 229 :=
by
  -- Proof omitted
  sorry

end Jimmy_earns_229_l148_148278


namespace no_base_makes_131b_square_l148_148972

theorem no_base_makes_131b_square : ∀ (b : ℤ), b > 3 → ∀ (n : ℤ), n * n ≠ b^2 + 3 * b + 1 :=
by
  intros b h_gt_3 n
  sorry

end no_base_makes_131b_square_l148_148972


namespace usual_travel_time_l148_148324

theorem usual_travel_time
  (S : ℝ) (T : ℝ) 
  (h0 : S > 0)
  (h1 : (S / T) = (4 / 5 * S / (T + 6))) : 
  T = 30 :=
by sorry

end usual_travel_time_l148_148324


namespace quadratic_inequalities_solution_l148_148517

noncomputable def a : Type := sorry
noncomputable def b : Type := sorry
noncomputable def c : Type := sorry

theorem quadratic_inequalities_solution (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + bx + c > 0 ↔ -1/3 < x ∧ x < 2) :
  ∀ y, cx^2 + bx + a < 0 ↔ -3 < y ∧ y < 1/2 :=
sorry

end quadratic_inequalities_solution_l148_148517


namespace remainder_prod_mod_10_l148_148603

theorem remainder_prod_mod_10 :
  (2457 * 7963 * 92324) % 10 = 4 :=
  sorry

end remainder_prod_mod_10_l148_148603


namespace journey_time_l148_148456

variables (d1 d2 : ℝ) (T : ℝ)

theorem journey_time :
  (d1 / 30 + (150 - d1) / 4 = T) ∧
  (d1 / 30 + d2 / 30 + (150 - (d1 + d2)) / 4 = T) ∧
  (d2 / 4 + (150 - (d1 + d2)) / 4 = T) ∧
  (d1 = 3 / 2 * d2) 
  → T = 18 :=
by
  sorry

end journey_time_l148_148456


namespace six_digit_numbers_with_at_least_one_zero_l148_148664

def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

def has_at_least_one_zero (n : ℕ) : Prop :=
  ∃ i : Fin 6, (Nat.digits 10 n)!!i.val = 0

theorem six_digit_numbers_with_at_least_one_zero : 
  ∃ c : ℕ, (∀ n : ℕ, n ∈ set_of is_six_digit → has_at_least_one_zero n → true) ∧ 
  c = 368559 :=
sorry

end six_digit_numbers_with_at_least_one_zero_l148_148664


namespace binary_arithmetic_correct_l148_148223

theorem binary_arithmetic_correct :
  (2^3 + 2^2 + 2^0) + (2^2 + 2^1 + 2^0) - (2^3 + 2^2 + 2^1) + (2^3 + 2^0) + (2^3 + 2^1) = 2^4 + 2^3 + 2^0 :=
by sorry

end binary_arithmetic_correct_l148_148223


namespace egg_weight_probability_l148_148237

theorem egg_weight_probability : 
  let P_lt_30 := 0.3
  let P_30_40 := 0.5
  P_lt_30 + P_30_40 ≤ 1 → (1 - (P_lt_30 + P_30_40) = 0.2) := by
  intro h
  sorry

end egg_weight_probability_l148_148237


namespace sequence_general_term_l148_148116

theorem sequence_general_term (n : ℕ) (hn : 0 < n) : 
  ∃ (a_n : ℕ), a_n = 2 * Int.floor (Real.sqrt (n - 1)) + 1 :=
by
  sorry

end sequence_general_term_l148_148116


namespace min_value_expression_l148_148903

theorem min_value_expression (a b : ℝ) : ∃ v : ℝ, ∀ (a b : ℝ), (a^2 + a * b + b^2 - a - 2 * b) ≥ v ∧ v = -1 :=
by
  sorry

end min_value_expression_l148_148903


namespace fish_to_apples_l148_148895

variable {Fish Loaf Rice Apple : Type}
variable (f : Fish → ℝ) (l : Loaf → ℝ) (r : Rice → ℝ) (a : Apple → ℝ)
variable (F : Fish) (L : Loaf) (A : Apple) (R : Rice)

-- Conditions
axiom cond1 : 4 * f F = 3 * l L
axiom cond2 : l L = 5 * r R
axiom cond3 : r R = 2 * a A

-- Proof statement
theorem fish_to_apples : f F = 7.5 * a A :=
by
  sorry

end fish_to_apples_l148_148895


namespace alexei_loss_per_week_l148_148024

-- Definitions
def aleesia_loss_per_week : ℝ := 1.5
def aleesia_total_weeks : ℕ := 10
def total_loss : ℝ := 35
def alexei_total_weeks : ℕ := 8

-- The statement to prove
theorem alexei_loss_per_week :
  (total_loss - aleesia_loss_per_week * aleesia_total_weeks) / alexei_total_weeks = 2.5 := 
by sorry

end alexei_loss_per_week_l148_148024


namespace inequality_true_l148_148099

theorem inequality_true (a b : ℝ) (h : a^2 + b^2 > 1) : |a| + |b| > 1 :=
sorry

end inequality_true_l148_148099


namespace measure_of_angle_D_l148_148760

-- Definitions of angles in pentagon ABCDE
variables (A B C D E : ℝ)

-- Conditions
def condition1 := D = A + 30
def condition2 := E = A + 50
def condition3 := B = C
def condition4 := A = B - 45
def condition5 := A + B + C + D + E = 540

-- Theorem to prove
theorem measure_of_angle_D (h1 : condition1 A D)
                           (h2 : condition2 A E)
                           (h3 : condition3 B C)
                           (h4 : condition4 A B)
                           (h5 : condition5 A B C D E) :
  D = 104 :=
sorry

end measure_of_angle_D_l148_148760


namespace part1_solution_set_part2_range_of_m_l148_148645

def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

theorem part1_solution_set (x : ℝ) : (f x 3 >= 6) ↔ (x ≤ -2 ∨ x ≥ 4) :=
by sorry

theorem part2_range_of_m (m : ℝ) (x : ℝ) : 
 (∀ x : ℝ, f x m ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7) :=
by sorry

end part1_solution_set_part2_range_of_m_l148_148645


namespace factorize_difference_of_squares_l148_148060

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l148_148060


namespace four_painters_small_room_days_l148_148508

-- Define the constants and conditions
def large_room_days : ℕ := 2
def small_room_factor : ℝ := 0.5
def total_painters : ℕ := 5
def painters_available : ℕ := 4

-- Define the total painter-days needed for the small room
def small_room_painter_days : ℝ := total_painters * (small_room_factor * large_room_days)

-- Define the proof problem statement
theorem four_painters_small_room_days : (small_room_painter_days / painters_available) = 5 / 4 :=
by
  -- Placeholder for the proof: we assume the goal is true for now
  sorry

end four_painters_small_room_days_l148_148508


namespace six_digit_numbers_with_at_least_one_zero_l148_148706

theorem six_digit_numbers_with_at_least_one_zero :
  let total := 9 * 10^5,
      without_zero := 9^6
  in total - without_zero = 368559 :=
by
  sorry

end six_digit_numbers_with_at_least_one_zero_l148_148706


namespace vehicles_with_at_least_80_kmh_equal_50_l148_148844

variable (num_vehicles_80_to_89 : ℕ := 15)
variable (num_vehicles_90_to_99 : ℕ := 30)
variable (num_vehicles_100_to_109 : ℕ := 5)

theorem vehicles_with_at_least_80_kmh_equal_50 :
  num_vehicles_80_to_89 + num_vehicles_90_to_99 + num_vehicles_100_to_109 = 50 := by
  sorry

end vehicles_with_at_least_80_kmh_equal_50_l148_148844


namespace find_triplets_l148_148231

theorem find_triplets (m n k : ℕ) (pos_m : 0 < m) (pos_n : 0 < n) (pos_k : 0 < k) : 
  (k^m ∣ m^n - 1) ∧ (k^n ∣ n^m - 1) ↔ (k = 1) ∨ (m = 1 ∧ n = 1) :=
by
  sorry

end find_triplets_l148_148231


namespace third_term_of_arithmetic_sequence_l148_148111

theorem third_term_of_arithmetic_sequence (a d : ℝ) (h : a + (a + 4 * d) = 10) : a + 2 * d = 5 :=
by {
  sorry
}

end third_term_of_arithmetic_sequence_l148_148111


namespace total_metal_rods_needed_l148_148823

def metal_rods_per_sheet : ℕ := 10
def sheets_per_panel : ℕ := 3
def metal_rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def panels : ℕ := 10

theorem total_metal_rods_needed : 
  (sheets_per_panel * metal_rods_per_sheet + beams_per_panel * metal_rods_per_beam) * panels = 380 :=
by
  exact rfl

end total_metal_rods_needed_l148_148823


namespace third_term_of_arithmetic_sequence_l148_148110

theorem third_term_of_arithmetic_sequence (a d : ℝ) (h : a + (a + 4 * d) = 10) : a + 2 * d = 5 :=
by {
  sorry
}

end third_term_of_arithmetic_sequence_l148_148110


namespace daily_salary_of_manager_l148_148415

theorem daily_salary_of_manager
  (M : ℕ)
  (salary_clerk : ℕ)
  (num_managers : ℕ)
  (num_clerks : ℕ)
  (total_salary : ℕ)
  (h1 : salary_clerk = 2)
  (h2 : num_managers = 2)
  (h3 : num_clerks = 3)
  (h4 : total_salary = 16)
  (h5 : 2 * M + 3 * salary_clerk = total_salary) :
  M = 5 := 
  sorry

end daily_salary_of_manager_l148_148415


namespace triangle_inequality_l148_148767

variable (a b c R : ℝ)

-- Assuming a, b, c as the sides of a triangle
-- and R as the circumradius.

theorem triangle_inequality:
  (1 / (a * b)) + (1 / (b * c)) + (1 / (c * a)) ≥ (1 / (R * R)) :=
by
  sorry

end triangle_inequality_l148_148767


namespace selling_price_range_l148_148617

theorem selling_price_range
  (unit_purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_sales_volume : ℝ)
  (price_increase_effect : ℝ)
  (daily_profit_threshold : ℝ)
  (x : ℝ) :
  unit_purchase_price = 8 →
  initial_selling_price = 10 →
  initial_sales_volume = 100 →
  price_increase_effect = 10 →
  daily_profit_threshold = 320 →
  (initial_selling_price - unit_purchase_price) * initial_sales_volume > daily_profit_threshold →
  12 < x → x < 16 →
  (x - unit_purchase_price) * (initial_sales_volume - price_increase_effect * (x - initial_selling_price)) > daily_profit_threshold :=
sorry

end selling_price_range_l148_148617


namespace solve_equation_theorem_l148_148568

noncomputable def solve_equations (S P : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let x1 := (S + real.sqrt (S^2 - 4 * P)) / 2
  let y1 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let x2 := (S - real.sqrt (S^2 - 4 * P)) / 2
  let y2 := (S + real.sqrt (S^2 - 4 * P)) / 2
  ((x1, y1), (x2, y2))

theorem solve_equation_theorem (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ x * y = P) ↔ (∃ (x1 y1 x2 y2 : ℝ), 
    ((x, y) = (x1, y1) ∨ (x, y) = (x2, y2)) ∧
    solve_equations S P = ((x1, y1), (x2, y2))) :=
by
  sorry

end solve_equation_theorem_l148_148568


namespace six_digit_numbers_with_zero_l148_148737

theorem six_digit_numbers_with_zero :
  let total_digits := (9 * 10 * 10 * 10 * 10 * 10 : ℕ)
  let non_zero_digits := (9 * 9 * 9 * 9 * 9 * 9 : ℕ)
  total_digits - non_zero_digits = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l148_148737


namespace inequality_xy_gt_xz_l148_148246

theorem inequality_xy_gt_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 1) : 
  x * y > x * z := 
by
  sorry  -- Proof is not required as per the instructions

end inequality_xy_gt_xz_l148_148246


namespace intersection_of_A_and_B_eq_C_l148_148549

noncomputable def A (x : ℝ) : Prop := x^2 - 4*x + 3 < 0
noncomputable def B (x : ℝ) : Prop := 2 - x > 0
noncomputable def A_inter_B (x : ℝ) : Prop := A x ∧ B x

theorem intersection_of_A_and_B_eq_C :
  {x : ℝ | A_inter_B x} = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

end intersection_of_A_and_B_eq_C_l148_148549


namespace greatest_c_for_expression_domain_all_real_l148_148648

theorem greatest_c_for_expression_domain_all_real :
  ∃ c : ℤ, c ≤ 7 ∧ c ^ 2 < 60 ∧ ∀ d : ℤ, d > 7 → ¬ (d ^ 2 < 60) := sorry

end greatest_c_for_expression_domain_all_real_l148_148648


namespace part1_part2_l148_148513

-- Definitions of sets A and B
def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def B (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ 3 - 2 * a }

-- Part 1: Prove that (complement of A union B = Universal Set) implies a in (-∞, 0]
theorem part1 (U : Set ℝ) (hU : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part 2: Prove that (A intersection B = B) implies a in [1/2, ∞)
theorem part2 (h : (A ∩ B a) = B a) : 1/2 ≤ a := sorry

end part1_part2_l148_148513


namespace green_fraction_is_three_fifths_l148_148414

noncomputable def fraction_green_after_tripling (total_balloons : ℕ) : ℚ :=
  let green_balloons := total_balloons / 3
  let new_green_balloons := green_balloons * 3
  let new_total_balloons := total_balloons * (5 / 3)
  new_green_balloons / new_total_balloons

theorem green_fraction_is_three_fifths (total_balloons : ℕ) (h : total_balloons > 0) : 
  fraction_green_after_tripling total_balloons = 3 / 5 := 
by 
  sorry

end green_fraction_is_three_fifths_l148_148414


namespace Canada_moose_population_l148_148891

theorem Canada_moose_population (moose beavers humans : ℕ) (h1 : beavers = 2 * moose) 
                              (h2 : humans = 19 * beavers) (h3 : humans = 38 * 10^6) : 
                              moose = 1 * 10^6 :=
by
  sorry

end Canada_moose_population_l148_148891


namespace six_digit_numbers_with_at_least_one_zero_correct_l148_148726

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l148_148726


namespace six_digit_numbers_with_zero_l148_148731

theorem six_digit_numbers_with_zero :
  let total_six_digit_numbers := 9 * 10 ^ 5 in
  let total_numbers_no_zero := 9 ^ 6 in
  total_six_digit_numbers - total_numbers_no_zero = 368559 :=
by
  let total_six_digit_numbers := 9 * 10 ^ 5
  let total_numbers_no_zero := 9 ^ 6
  show total_six_digit_numbers - total_numbers_no_zero = 368559
  sorry

end six_digit_numbers_with_zero_l148_148731


namespace gold_cube_profit_multiple_l148_148774

theorem gold_cube_profit_multiple :
  let side_length : ℝ := 6
  let density : ℝ := 19
  let cost_per_gram : ℝ := 60
  let profit : ℝ := 123120
  let volume := side_length ^ 3
  let mass := density * volume
  let cost := mass * cost_per_gram
  let selling_price := cost + profit
  let multiple := selling_price / cost
  multiple = 1.5 := by
  sorry

end gold_cube_profit_multiple_l148_148774


namespace value_of_y_l148_148104

theorem value_of_y (x y : ℝ) (h1 : x^(2 * y) = 9) (h2 : x = 3) : y = 1 := by
  sorry

end value_of_y_l148_148104
