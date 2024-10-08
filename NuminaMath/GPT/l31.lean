import Mathlib

namespace minimum_value_on_line_l31_31065

theorem minimum_value_on_line : ∃ (x y : ℝ), (x + y = 4) ∧ (∀ x' y', (x' + y' = 4) → (x^2 + y^2 ≤ x'^2 + y'^2)) ∧ (x^2 + y^2 = 8) :=
sorry

end minimum_value_on_line_l31_31065


namespace probability_all_balls_same_color_probability_4_white_balls_l31_31499

-- Define initial conditions
def initial_white_balls : ℕ := 6
def initial_yellow_balls : ℕ := 4
def total_initial_balls : ℕ := initial_white_balls + initial_yellow_balls

-- Define the probability calculation for drawing balls as described
noncomputable def draw_probability_same_color_after_4_draws : ℚ :=
  (6 / 10) * (7 / 10) * (8 / 10) * (9 / 10)

noncomputable def draw_probability_4_white_balls_after_4_draws : ℚ :=
  (6 / 10) * (3 / 10) * (4 / 10) * (5 / 10) + 
  3 * ((4 / 10) * (5 / 10) * (4 / 10) * (5 / 10))

-- The theorem we want to prove about the probabilities
theorem probability_all_balls_same_color :
  draw_probability_same_color_after_4_draws = 189 / 625 := by
  sorry

theorem probability_4_white_balls :
  draw_probability_4_white_balls_after_4_draws = 19 / 125 := by
  sorry

end probability_all_balls_same_color_probability_4_white_balls_l31_31499


namespace cody_final_money_l31_31431

-- Definitions for the initial conditions
def original_money : ℝ := 45
def birthday_money : ℝ := 9
def game_price : ℝ := 19
def discount_rate : ℝ := 0.10
def friend_owes : ℝ := 12

-- Calculate the final amount Cody has
def final_amount : ℝ := original_money + birthday_money - (game_price * (1 - discount_rate)) + friend_owes

-- The theorem to prove the amount of money Cody has now
theorem cody_final_money :
  final_amount = 48.90 :=
by sorry

end cody_final_money_l31_31431


namespace h_at_4_l31_31914

noncomputable def f (x : ℝ) := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) := 3 - (4 / x)

noncomputable def h (x : ℝ) := (1 / f_inv x) + 10

theorem h_at_4 : h 4 = 10.5 :=
by
  sorry

end h_at_4_l31_31914


namespace number_of_eggs_left_l31_31704

theorem number_of_eggs_left (initial_eggs : ℕ) (eggs_eaten_morning : ℕ) (eggs_eaten_afternoon : ℕ) (eggs_left : ℕ) :
    initial_eggs = 20 → eggs_eaten_morning = 4 → eggs_eaten_afternoon = 3 → eggs_left = initial_eggs - (eggs_eaten_morning + eggs_eaten_afternoon) → eggs_left = 13 :=
by
  intros h_initial h_morning h_afternoon h_calc
  rw [h_initial, h_morning, h_afternoon] at h_calc
  norm_num at h_calc
  exact h_calc

end number_of_eggs_left_l31_31704


namespace parabola_sum_l31_31986

def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - b * x + c

def f (a b c : ℝ) (x : ℝ) : ℝ := a * (x + 7) ^ 2 - b * (x + 7) + c

def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x - 3) ^ 2 - b * (x - 3) + c

def fg (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_sum (a b c x : ℝ) : fg a b c x = 2 * a * x ^ 2 + (8 * a - 2 * b) * x + (58 * a - 4 * b + 2 * c) := by
  sorry

end parabola_sum_l31_31986


namespace product_of_numbers_l31_31836

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 26) (h2 : x - y = 8) : x * y = 153 :=
sorry

end product_of_numbers_l31_31836


namespace susan_age_l31_31559

theorem susan_age (S J B : ℝ) 
  (h1 : S = 2 * J)
  (h2 : S + J + B = 60) 
  (h3 : B = J + 10) : 
  S = 25 := sorry

end susan_age_l31_31559


namespace polynomial_root_abs_sum_eq_80_l31_31680

theorem polynomial_root_abs_sum_eq_80 (a b c : ℤ) (m : ℤ) 
  (h1 : a + b + c = 0) 
  (h2 : ab + bc + ac = -2023) 
  (h3 : ∃ m, ∀ x : ℤ, x^3 - 2023 * x + m = (x - a) * (x - b) * (x - c)) : 
  |a| + |b| + |c| = 80 := 
by {
  sorry
}

end polynomial_root_abs_sum_eq_80_l31_31680


namespace dealership_sales_l31_31700

theorem dealership_sales (sports_cars sedans suvs : ℕ) (h_sc : sports_cars = 35)
  (h_ratio_sedans : 5 * sedans = 8 * sports_cars) 
  (h_ratio_suvs : 5 * suvs = 3 * sports_cars) : 
  sedans = 56 ∧ suvs = 21 := by
  sorry

#print dealership_sales

end dealership_sales_l31_31700


namespace boat_speed_still_water_l31_31462

theorem boat_speed_still_water (V_s : ℝ) (T_u T_d : ℝ) 
  (h1 : V_s = 24) 
  (h2 : T_u = 2 * T_d) 
  (h3 : (V_b - V_s) * T_u = (V_b + V_s) * T_d) : 
  V_b = 72 := 
sorry

end boat_speed_still_water_l31_31462


namespace intersecting_circles_range_of_m_l31_31500

theorem intersecting_circles_range_of_m
  (x y m : ℝ)
  (C₁_eq : x^2 + y^2 - 2 * m * x + m^2 - 4 = 0)
  (C₂_eq : x^2 + y^2 + 2 * x - 4 * m * y + 4 * m^2 - 8 = 0)
  (intersect : ∃ x y : ℝ, (x^2 + y^2 - 2 * m * x + m^2 - 4 = 0) ∧ (x^2 + y^2 + 2 * x - 4 * m * y + 4 * m^2 - 8 = 0))
  : m ∈ Set.Ioo (-12/5) (-2/5) ∪ Set.Ioo (3/5) 2 := 
sorry

end intersecting_circles_range_of_m_l31_31500


namespace reflection_slope_intercept_l31_31651

noncomputable def reflect_line_slope_intercept (k : ℝ) (hk1 : k ≠ 0) (hk2 : k ≠ -1) : ℝ × ℝ :=
  let slope := (1 : ℝ) / k
  let intercept := (k - 1) / k
  (slope, intercept)

theorem reflection_slope_intercept {k : ℝ} (hk1 : k ≠ 0) (hk2 : k ≠ -1) :
  reflect_line_slope_intercept k hk1 hk2 = (1/k, (k-1)/k) := by
  sorry

end reflection_slope_intercept_l31_31651


namespace rows_seat_7_students_are_5_l31_31659

-- Definitions based on provided conditions
def total_students : Nat := 53
def total_rows (six_seat_rows seven_seat_rows : Nat) : Prop := 
  total_students = 6 * six_seat_rows + 7 * seven_seat_rows

-- To prove the number of rows seating exactly 7 students is 5
def number_of_7_seat_rows (six_seat_rows seven_seat_rows : Nat) : Prop := 
  total_rows six_seat_rows seven_seat_rows ∧ seven_seat_rows = 5

-- Statement to be proved
theorem rows_seat_7_students_are_5 : ∃ (six_seat_rows seven_seat_rows : Nat), number_of_7_seat_rows six_seat_rows seven_seat_rows := 
by
  -- Skipping the proof
  sorry

end rows_seat_7_students_are_5_l31_31659


namespace sum_of_squares_l31_31645

theorem sum_of_squares (x y : ℝ) (h1 : (x + y) ^ 2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
by
  sorry

end sum_of_squares_l31_31645


namespace curve_statements_incorrect_l31_31325

theorem curve_statements_incorrect (t : ℝ) :
  (1 < t ∧ t < 3 → ¬ ∀ x y : ℝ, (x^2 / (3 - t) + y^2 / (t - 1) = 1 → x^2 + y^2 ≠ 1)) ∧
  ((3 - t) * (t - 1) < 0 → ¬ t < 1) :=
by
  sorry

end curve_statements_incorrect_l31_31325


namespace xy_value_l31_31716

theorem xy_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 3 / x = y + 3 / y) (hxy : x ≠ y) : x * y = 3 :=
sorry

end xy_value_l31_31716


namespace determinant_zero_implies_sum_neg_nine_l31_31806

theorem determinant_zero_implies_sum_neg_nine
  (x y : ℝ)
  (h1 : x ≠ y)
  (h2 : x * y = 1)
  (h3 : (Matrix.det ![
    ![1, 5, 8], 
    ![3, x, y], 
    ![3, y, x]
  ]) = 0) : 
  x + y = -9 := 
sorry

end determinant_zero_implies_sum_neg_nine_l31_31806


namespace space_between_trees_l31_31359

theorem space_between_trees (tree_count : ℕ) (tree_space : ℕ) (road_length : ℕ)
  (h1 : tree_space = 1) (h2 : tree_count = 13) (h3 : road_length = 157) :
  (road_length - tree_count * tree_space) / (tree_count - 1) = 12 := by
  sorry

end space_between_trees_l31_31359


namespace pentagon_probability_l31_31363

/-- Ten points are equally spaced around the circumference of a regular pentagon,
with each side being divided into two equal segments.

We need to prove that the probability of choosing two points randomly and
having them be exactly one side of the pentagon apart is 2/9.
-/
theorem pentagon_probability : 
  let total_points := 10
  let favorable_pairs := 10
  let total_pairs := total_points * (total_points - 1) / 2
  (favorable_pairs / total_pairs : ℚ) = 2 / 9 :=
by
  sorry

end pentagon_probability_l31_31363


namespace fractions_sum_correct_l31_31183

noncomputable def fractions_sum : ℝ := (3 / 20) + (5 / 200) + (7 / 2000) + 5

theorem fractions_sum_correct : fractions_sum = 5.1785 :=
by
  sorry

end fractions_sum_correct_l31_31183


namespace max_area_of_triangle_l31_31633

theorem max_area_of_triangle (AB AC BC : ℝ) : 
  AB = 4 → AC = 2 * BC → 
  ∃ (S : ℝ), (∀ (S' : ℝ), S' ≤ S) ∧ S = 16 / 3 :=
by
  sorry

end max_area_of_triangle_l31_31633


namespace main_theorem_l31_31149

-- Let x be a real number
variable {x : ℝ}

-- Define the given identity
def identity (M₁ M₂ : ℝ) : Prop :=
  ∀ x, (50 * x - 42) / (x^2 - 5 * x + 6) = M₁ / (x - 2) + M₂ / (x - 3)

-- The proposition to prove the numerical value of M₁M₂
def prove_M1M2_value : Prop :=
  ∀ (M₁ M₂ : ℝ), identity M₁ M₂ → M₁ * M₂ = -6264

theorem main_theorem : prove_M1M2_value :=
  sorry

end main_theorem_l31_31149


namespace fitness_club_alpha_is_more_advantageous_l31_31152

-- Define the costs and attendance pattern constants
def yearly_cost_alpha : ℕ := 11988
def monthly_cost_beta : ℕ := 1299
def weeks_per_month : ℕ := 4

-- Define the attendance pattern
def attendance_pattern : List ℕ := [3 * weeks_per_month, 2 * weeks_per_month, 1 * weeks_per_month, 0 * weeks_per_month]

-- Compute the total visits in a year for regular attendance
def total_visits (patterns : List ℕ) : ℕ :=
  patterns.sum * 3

-- Compute the total yearly cost for Beta when considering regular attendance
def yearly_cost_beta (monthly_cost : ℕ) : ℕ :=
  monthly_cost * 12

-- Calculate cost per visit for each club with given attendance
def cost_per_visit (total_cost : ℕ) (total_visits : ℕ) : ℚ :=
  total_cost / total_visits

theorem fitness_club_alpha_is_more_advantageous :
  cost_per_visit yearly_cost_alpha (total_visits attendance_pattern) <
  cost_per_visit (yearly_cost_beta monthly_cost_beta) (total_visits attendance_pattern) :=
by
  sorry

end fitness_club_alpha_is_more_advantageous_l31_31152


namespace dot_product_solution_1_l31_31591

variable (a b : ℝ × ℝ)
variable (k : ℝ)

def two_a_add_b (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 + b.1, 2 * a.2 + b.2)

def dot_product (x y : ℝ × ℝ) : ℝ :=
x.1 * y.1 + x.2 * y.2

theorem dot_product_solution_1 :
  let a := (1, -1)
  let b := (-1, 2)
  dot_product (two_a_add_b a b) a = 1 := by
sorry

end dot_product_solution_1_l31_31591


namespace sled_total_distance_l31_31864

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

theorem sled_total_distance (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 6 → d = 8 → n = 20 → arithmetic_sequence_sum a₁ d n = 1640 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sled_total_distance_l31_31864


namespace lcm_of_two_numbers_l31_31205

theorem lcm_of_two_numbers (A B : ℕ) 
  (h_prod : A * B = 987153000) 
  (h_hcf : Int.gcd A B = 440) : 
  Nat.lcm A B = 2243525 :=
by
  sorry

end lcm_of_two_numbers_l31_31205


namespace line_equation_through_two_points_l31_31284

noncomputable def LineEquation (x0 y0 x1 y1 x y : ℝ) : Prop :=
  (x1 ≠ x0) → (y1 ≠ y0) → 
  (y - y0) / (y1 - y0) = (x - x0) / (x1 - x0)

theorem line_equation_through_two_points 
  (x0 y0 x1 y1 : ℝ) 
  (h₁ : x1 ≠ x0) 
  (h₂ : y1 ≠ y0) : 
  ∀ (x y : ℝ), LineEquation x0 y0 x1 y1 x y :=  
by
  sorry

end line_equation_through_two_points_l31_31284


namespace solve_system_eqns_l31_31536

theorem solve_system_eqns 
  {a b c : ℝ} (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
  {x y z : ℝ} 
  (h4 : a^3 + a^2 * x + a * y + z = 0)
  (h5 : b^3 + b^2 * x + b * y + z = 0)
  (h6 : c^3 + c^2 * x + c * y + z = 0) :
  x = -(a + b + c) ∧ y = ab + bc + ca ∧ z = -abc :=
by {
  sorry
}

end solve_system_eqns_l31_31536


namespace volume_of_box_ground_area_of_box_l31_31241

-- Given conditions
variable (l w h : ℕ)
variable (hl : l = 20)
variable (hw : w = 15)
variable (hh : h = 5)

-- Define volume and ground area
def volume (l w h : ℕ) : ℕ := l * w * h
def ground_area (l w : ℕ) : ℕ := l * w

-- Theorem to prove the correct volume
theorem volume_of_box : volume l w h = 1500 := by
  rw [hl, hw, hh]
  sorry

-- Theorem to prove the correct ground area
theorem ground_area_of_box : ground_area l w = 300 := by
  rw [hl, hw]
  sorry

end volume_of_box_ground_area_of_box_l31_31241


namespace sector_radius_l31_31292

theorem sector_radius (α S r : ℝ) (h1 : α = 3/4 * Real.pi) (h2 : S = 3/2 * Real.pi) :
  S = 1/2 * r^2 * α → r = 2 :=
by
  sorry

end sector_radius_l31_31292


namespace sqrt_product_eq_l31_31596

theorem sqrt_product_eq :
  (16 ^ (1 / 4) : ℝ) * (64 ^ (1 / 2)) = 16 := by
  sorry

end sqrt_product_eq_l31_31596


namespace hyperbola_center_l31_31941

theorem hyperbola_center : 
  (∃ x y : ℝ, (4 * y + 6)^2 / 16 - (5 * x - 3)^2 / 9 = 1) →
  (∃ h k : ℝ, h = 3 / 5 ∧ k = -3 / 2 ∧ 
    (∀ x' y', (4 * y' + 6)^2 / 16 - (5 * x' - 3)^2 / 9 = 1 → x' = h ∧ y' = k)) :=
sorry

end hyperbola_center_l31_31941


namespace days_to_complete_l31_31695

variable {m n : ℕ}

theorem days_to_complete (h : ∀ (m n : ℕ), (m + n) * m = 1) : 
  ∀ (n m : ℕ), (m * (m + n)) / n = m * (m + n) / n :=
by
  sorry

end days_to_complete_l31_31695


namespace isosceles_triangle_time_between_9_30_and_10_l31_31981

theorem isosceles_triangle_time_between_9_30_and_10 (time : ℕ) (h_time_range : 30 ≤ time ∧ time < 60)
  (h_isosceles : ∃ x : ℝ, 0 ≤ x ∧ x + 2 * x + 2 * x = 180) :
  time = 36 :=
  sorry

end isosceles_triangle_time_between_9_30_and_10_l31_31981


namespace trapezoid_division_areas_l31_31844

open Classical

variable (area_trapezoid : ℝ) (base1 base2 : ℝ)
variable (triangle1 triangle2 triangle3 triangle4 : ℝ)

theorem trapezoid_division_areas 
  (h1 : area_trapezoid = 3) 
  (h2 : base1 = 1) 
  (h3 : base2 = 2) 
  (h4 : triangle1 = 1 / 3)
  (h5 : triangle2 = 2 / 3)
  (h6 : triangle3 = 2 / 3)
  (h7 : triangle4 = 4 / 3) :
  triangle1 + triangle2 + triangle3 + triangle4 = area_trapezoid :=
by
  sorry

end trapezoid_division_areas_l31_31844


namespace exponential_function_fixed_point_l31_31588

theorem exponential_function_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1))} :=
by
  sorry

end exponential_function_fixed_point_l31_31588


namespace greatest_multiple_of_5_and_6_lt_1000_l31_31275

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end greatest_multiple_of_5_and_6_lt_1000_l31_31275


namespace particle_paths_count_l31_31029

-- Definitions for the movement in the Cartesian plane
def valid_moves (a b : ℕ) : List (ℕ × ℕ) := [(a + 2, b), (a, b + 2), (a + 1, b + 1)]

-- The condition to count unique paths from (0,0) to (6,6)
def count_paths (start target : ℕ × ℕ) : ℕ :=
  sorry -- The exact implementation to count paths is omitted here

theorem particle_paths_count :
  count_paths (0, 0) (6, 6) = 58 :=
sorry

end particle_paths_count_l31_31029


namespace first_plot_germination_rate_l31_31109

-- Define the known quantities and conditions
def plot1_seeds : ℕ := 300
def plot2_seeds : ℕ := 200
def plot2_germination_rate : ℚ := 35 / 100
def total_germination_percentage : ℚ := 26 / 100

-- Define a statement to prove the percentage of seeds that germinated in the first plot
theorem first_plot_germination_rate : 
  ∃ (x : ℚ), (x / 100) * plot1_seeds + (plot2_germination_rate * plot2_seeds) = total_germination_percentage * (plot1_seeds + plot2_seeds) ∧ x = 20 :=
by
  sorry

end first_plot_germination_rate_l31_31109


namespace slope_range_l31_31330

theorem slope_range (a : ℝ) (ha : a ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) :
  ∃ k : ℝ, k = Real.tan a ∧ k ∈ Set.Ici 1 :=
by {
  sorry
}

end slope_range_l31_31330


namespace channel_width_at_top_l31_31095

theorem channel_width_at_top 
  (area : ℝ) (bottom_width : ℝ) (depth : ℝ) 
  (H1 : bottom_width = 6) 
  (H2 : area = 630) 
  (H3 : depth = 70) : 
  ∃ w : ℝ, (∃ H : w + 6 > 0, area = 1 / 2 * (w + bottom_width) * depth) ∧ w = 12 :=
by
  sorry

end channel_width_at_top_l31_31095


namespace largest_rectangle_area_l31_31382

theorem largest_rectangle_area (x y : ℝ) (h1 : 2*x + 2*y = 60) (h2 : x ≥ 2*y) : ∃ A, A = x*y ∧ A ≤ 200 := by
  sorry

end largest_rectangle_area_l31_31382


namespace order_of_f0_f1_f_2_l31_31992

noncomputable def f (m x : ℝ) := (m-1) * x^2 + 6 * m * x + 2

theorem order_of_f0_f1_f_2 (m : ℝ) (h_even : ∀ x : ℝ, f m x = f m (-x)) :
  m = 0 → f m (-2) < f m 1 ∧ f m 1 < f m 0 :=
by 
  sorry

end order_of_f0_f1_f_2_l31_31992


namespace trains_cross_time_l31_31296

noncomputable def timeToCrossEachOther (L : ℝ) (T1 : ℝ) (T2 : ℝ) : ℝ :=
  let V1 := L / T1
  let V2 := L / T2
  let Vr := V1 + V2
  let totalDistance := L + L
  totalDistance / Vr

theorem trains_cross_time (L T1 T2 : ℝ) (hL : L = 120) (hT1 : T1 = 10) (hT2 : T2 = 15) :
  timeToCrossEachOther L T1 T2 = 12 :=
by
  simp [timeToCrossEachOther, hL, hT1, hT2]
  sorry

end trains_cross_time_l31_31296


namespace find_integers_a_l31_31143

theorem find_integers_a (a : ℤ) : 
  (∃ n : ℤ, (a^3 + 1 = (a - 1) * n)) ↔ a = -1 ∨ a = 0 ∨ a = 2 ∨ a = 3 := 
sorry

end find_integers_a_l31_31143


namespace ratio_of_work_completed_by_a_l31_31432

theorem ratio_of_work_completed_by_a (A B W : ℝ) (ha : (A + B) * 6 = W) :
  (A * 3) / W = 1 / 2 :=
by 
  sorry

end ratio_of_work_completed_by_a_l31_31432


namespace range_of_m_for_ellipse_l31_31573

-- Define the equation of the ellipse
def ellipse_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

-- The theorem to prove
theorem range_of_m_for_ellipse (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) →
  5 < m :=
sorry

end range_of_m_for_ellipse_l31_31573


namespace rectangular_prism_edge_sum_l31_31551

theorem rectangular_prism_edge_sum
  (V A : ℝ)
  (hV : V = 8)
  (hA : A = 32)
  (l w h : ℝ)
  (geom_prog : l = w / h ∧ w = l * h ∧ h = l * (w / l)) :
  4 * (l + w + h) = 28 :=
by 
  sorry

end rectangular_prism_edge_sum_l31_31551


namespace f_1_eq_zero_l31_31789

-- Given a function f with the specified properties
variable {f : ℝ → ℝ}

-- Given 1) the domain of the function
axiom domain_f : ∀ x, (x < 0 ∨ x > 0) → true 

-- Given 2) the functional equation
axiom functional_eq_f : ∀ x₁ x₂, (x₁ < 0 ∨ x₁ > 0) ∧ (x₂ < 0 ∨ x₂ > 0) → f (x₁ * x₂) = f x₁ + f x₂

-- Prove that f(1) = 0
theorem f_1_eq_zero : f 1 = 0 := 
  sorry

end f_1_eq_zero_l31_31789


namespace two_cos_45_eq_sqrt_2_l31_31427

theorem two_cos_45_eq_sqrt_2 : 2 * Real.cos (pi / 4) = Real.sqrt 2 := by
  sorry

end two_cos_45_eq_sqrt_2_l31_31427


namespace order_of_real_numbers_l31_31477

noncomputable def a : ℝ := Real.arcsin (3 / 4)
noncomputable def b : ℝ := Real.arccos (1 / 5)
noncomputable def c : ℝ := 1 + Real.arctan (2 / 3)

theorem order_of_real_numbers : a < b ∧ b < c :=
by sorry

end order_of_real_numbers_l31_31477


namespace range_a_sufficient_not_necessary_l31_31003

theorem range_a_sufficient_not_necessary (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, (x ^ 2 - 4 * a * x + 3 * a ^ 2 < 0) → (|x - 3| > 1)) →
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2 / 3) :=
sorry

end range_a_sufficient_not_necessary_l31_31003


namespace conner_collected_on_day_two_l31_31320

variable (s0 : ℕ) (c0 : ℕ) (s1 : ℕ) (c1 : ℕ) (c2 : ℕ) (s3 : ℕ) (c3 : ℕ) (total_sydney : ℕ) (total_conner : ℕ)

theorem conner_collected_on_day_two :
  s0 = 837 ∧ c0 = 723 ∧ 
  s1 = 4 ∧ c1 = 8 * s1 ∧
  s3 = 2 * c1 ∧ c3 = 27 ∧
  total_sydney = s0 + s1 + s3 ∧
  total_conner = c0 + c1 + c2 + c3 ∧
  total_conner >= total_sydney
  → c2 = 123 :=
by
  sorry

end conner_collected_on_day_two_l31_31320


namespace convert_300_degree_to_radian_l31_31125

theorem convert_300_degree_to_radian : (300 : ℝ) * π / 180 = 5 * π / 3 :=
by
  sorry

end convert_300_degree_to_radian_l31_31125


namespace probability_neither_red_nor_purple_l31_31714

section Probability

def total_balls : ℕ := 60
def red_balls : ℕ := 15
def purple_balls : ℕ := 3
def total_red_or_purple_balls : ℕ := red_balls + purple_balls
def non_red_or_purple_balls : ℕ := total_balls - total_red_or_purple_balls

theorem probability_neither_red_nor_purple :
  (non_red_or_purple_balls : ℚ) / (total_balls : ℚ) = 7 / 10 :=
by
  sorry

end Probability

end probability_neither_red_nor_purple_l31_31714


namespace intersection_A_B_l31_31404

def A : Set ℝ := { x | x^2 - 2*x < 0 }
def B : Set ℝ := { x | |x| > 1 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end intersection_A_B_l31_31404


namespace no_such_function_exists_l31_31611

open Classical

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (f 0 > 0) ∧ (∀ (x y : ℝ), f (x + y) ≥ f x + y * f (f x)) :=
sorry

end no_such_function_exists_l31_31611


namespace work_time_l31_31223

-- Definitions and conditions
variables (A B C D h : ℝ)
variable (h_def : ℝ := 1 / (1 / A + 1 / B + 1 / D))

-- Conditions
axiom cond1 : 1 / A + 1 / B + 1 / C + 1 / D = 1 / (A - 8)
axiom cond2 : 1 / A + 1 / B + 1 / C + 1 / D = 1 / (B - 2)
axiom cond3 : 1 / A + 1 / B + 1 / C + 1 / D = 3 / C
axiom cond4 : 1 / A + 1 / B + 1 / D = 2 / C

-- The statement to prove
theorem work_time : h_def = 16 / 11 := by
  sorry

end work_time_l31_31223


namespace laptop_cost_l31_31639

theorem laptop_cost (L : ℝ) (smartphone_cost : ℝ) (total_cost : ℝ) (change : ℝ) (n_laptops n_smartphones : ℕ) 
  (hl_smartphone : smartphone_cost = 400) 
  (hl_laptops : n_laptops = 2) 
  (hl_smartphones : n_smartphones = 4) 
  (hl_total : total_cost = 3000)
  (hl_change : change = 200) 
  (hl_total_spent : total_cost - change = 2 * L + 4 * smartphone_cost) : 
  L = 600 :=
by 
  sorry

end laptop_cost_l31_31639


namespace lindas_nickels_l31_31291

theorem lindas_nickels
  (N : ℕ)
  (initial_dimes : ℕ := 2)
  (initial_quarters : ℕ := 6)
  (initial_nickels : ℕ := N)
  (additional_dimes : ℕ := 2)
  (additional_quarters : ℕ := 10)
  (additional_nickels : ℕ := 2 * N)
  (total_coins : ℕ := 35)
  (h : initial_dimes + initial_quarters + initial_nickels + additional_dimes + additional_quarters + additional_nickels = total_coins) :
  N = 5 := by
  sorry

end lindas_nickels_l31_31291


namespace geometric_sequence_theorem_l31_31768

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = a n * r

def holds_condition (a : ℕ → ℝ) : Prop := 
  a 1 * a 10 = -2

theorem geometric_sequence_theorem (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_cond : holds_condition a) : a 4 * a 7 = -2 :=
by
  sorry

end geometric_sequence_theorem_l31_31768


namespace cube_vertex_plane_distance_l31_31189

theorem cube_vertex_plane_distance
  (d : ℝ)
  (h_dist : d = 9 - Real.sqrt 186)
  (h7 : ∀ (a b c  : ℝ), a^2 + b^2 + c^2 = 1 → 64 * (a^2 + b^2 + c^2) = 64)
  (h8 : ∀ (d : ℝ), 3 * d^2 - 54 * d + 181 = 0) :
  ∃ (p q r : ℕ), 
    p = 27 ∧ q = 186 ∧ r = 3 ∧ (p + q + r < 1000) ∧ (d = (p - Real.sqrt q) / r) := 
  by
    sorry

end cube_vertex_plane_distance_l31_31189


namespace cab_driver_income_l31_31056

theorem cab_driver_income (x : ℕ) 
  (h₁ : (45 + x + 60 + 65 + 70) / 5 = 58) : x = 50 := 
by
  -- Insert the proof here
  sorry

end cab_driver_income_l31_31056


namespace samson_mother_age_l31_31506

variable (S M : ℕ)
variable (x : ℕ)

def problem_statement : Prop :=
  S = 6 ∧
  S - x = 2 ∧
  M - x = 4 * 2 →
  M = 16

theorem samson_mother_age (S M x : ℕ) (h : problem_statement S M x) : M = 16 :=
by
  sorry

end samson_mother_age_l31_31506


namespace annual_increase_rate_l31_31969

theorem annual_increase_rate (r : ℝ) (h : 70400 * (1 + r)^2 = 89100) : r = 0.125 :=
sorry

end annual_increase_rate_l31_31969


namespace total_handshakes_eq_900_l31_31961

def num_boys : ℕ := 25
def handshakes_per_pair : ℕ := 3

theorem total_handshakes_eq_900 : (num_boys * (num_boys - 1) / 2) * handshakes_per_pair = 900 := by
  sorry

end total_handshakes_eq_900_l31_31961


namespace max_S_at_n_four_l31_31192

-- Define the sequence sum S_n
def S (n : ℕ) : ℤ := -(n^2 : ℤ) + (8 * n : ℤ)

-- Prove that S_n attains its maximum value at n = 4
theorem max_S_at_n_four : ∀ n : ℕ, S n ≤ S 4 :=
by
  sorry

end max_S_at_n_four_l31_31192


namespace max_notebooks_with_budget_l31_31887

/-- Define the prices and quantities of notebooks -/
def notebook_price : ℕ := 2
def four_pack_price : ℕ := 6
def seven_pack_price : ℕ := 9
def max_budget : ℕ := 15

def total_notebooks (single_packs four_packs seven_packs : ℕ) : ℕ :=
  single_packs + 4 * four_packs + 7 * seven_packs

theorem max_notebooks_with_budget : 
  ∃ (single_packs four_packs seven_packs : ℕ), 
    notebook_price * single_packs + 
    four_pack_price * four_packs + 
    seven_pack_price * seven_packs ≤ max_budget ∧ 
    booklet_price * single_packs + 
    four_pack_price * four_packs + 
    seven_pack_price * seven_packs + total_notebooks single_packs four_packs seven_packs = 11 := 
by
  sorry

end max_notebooks_with_budget_l31_31887


namespace distribute_stickers_l31_31294

-- Definitions based on conditions
def stickers : ℕ := 10
def sheets : ℕ := 5

-- Theorem stating the equivalence of distributing the stickers onto sheets
theorem distribute_stickers :
  (Nat.choose (stickers + sheets - 1) (sheets - 1)) = 1001 :=
by 
  -- Here is where the proof would go, but we skip it with sorry for the purpose of this task
  sorry

end distribute_stickers_l31_31294


namespace abc_inequality_l31_31364

variable {a b c : ℝ}

theorem abc_inequality (h₀ : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h₁ : a + b + c = 6)
  (h₂ : a * b + b * c + c * a = 9) :
  0 < a * b * c ∧ a * b * c < 4 := by
  sorry

end abc_inequality_l31_31364


namespace geometric_sequence_first_term_l31_31818

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^2 = 18) (h2 : a * r^4 = 72) : a = 4.5 := by
  sorry

end geometric_sequence_first_term_l31_31818


namespace intersect_point_l31_31679

noncomputable def f (x : ℤ) (b : ℤ) : ℤ := 5 * x + b
noncomputable def f_inv (x : ℤ) (b : ℤ) : ℤ := (x - b) / 5

theorem intersect_point (a b : ℤ) (h_intersections : (f (-3) b = a ∧ f a b = -3)) : a = -3 :=
by
  sorry

end intersect_point_l31_31679


namespace car_speed_l31_31774

theorem car_speed (v : ℝ) (hv : 2 + (1 / v) * 3600 = (1 / 90) * 3600) :
  v = 600 / 7 :=
sorry

end car_speed_l31_31774


namespace machines_working_together_l31_31531

theorem machines_working_together (x : ℝ) :
  (∀ P Q R : ℝ, P = x + 4 ∧ Q = x + 2 ∧ R = 2 * x + 2 ∧ (1 / P + 1 / Q + 1 / R = 1 / x)) ↔ (x = 2 / 3) :=
by
  sorry

end machines_working_together_l31_31531


namespace calc_expr_eq_l31_31601

theorem calc_expr_eq : 2 + 3 / (4 + 5 / 6) = 76 / 29 := 
by 
  sorry

end calc_expr_eq_l31_31601


namespace equation_of_circle_center_0_4_passing_through_3_0_l31_31252

noncomputable def circle_radius (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem equation_of_circle_center_0_4_passing_through_3_0 :
  ∃ (r : ℝ), (r = circle_radius 0 4 3 0) ∧ (r = 5) ∧ ((x y : ℝ) → ((x - 0) ^ 2 + (y - 4) ^ 2 = r ^ 2) ↔ (x ^ 2 + (y - 4) ^ 2 = 25)) :=
by
  sorry

end equation_of_circle_center_0_4_passing_through_3_0_l31_31252


namespace find_c_of_parabola_l31_31456

theorem find_c_of_parabola (a b c : ℝ) (h_vertex : ∀ x, y = a * (x - 3)^2 - 5)
                           (h_point : ∀ x y, (x = 1) → (y = -3) → y = a * (x - 3)^2 - 5)
                           (h_standard_form : ∀ x, y = a * x^2 + b * x + c) :
  c = -0.5 :=
sorry

end find_c_of_parabola_l31_31456


namespace symmetry_center_of_g_l31_31411

open Real

noncomputable def g (x : ℝ) : ℝ := cos ((1 / 2) * x - π / 6)

def center_of_symmetry : Set (ℝ × ℝ) := { p | ∃ k : ℤ, p = (2 * k * π + 4 * π / 3, 0) }

theorem symmetry_center_of_g :
  (∃ p : ℝ × ℝ, p ∈ center_of_symmetry) :=
sorry

end symmetry_center_of_g_l31_31411


namespace hypotenuse_length_l31_31585

theorem hypotenuse_length
    (a b c : ℝ)
    (h1: a^2 + b^2 + c^2 = 2450)
    (h2: b = a + 7)
    (h3: c^2 = a^2 + b^2) :
    c = 35 := sorry

end hypotenuse_length_l31_31585


namespace find_min_positive_n_l31_31758

-- Assume the sequence {a_n} is given
variables {a : ℕ → ℤ}

-- Given conditions
-- a4 < 0 and a5 > |a4|
def condition1 (a : ℕ → ℤ) : Prop := a 4 < 0
def condition2 (a : ℕ → ℤ) : Prop := a 5 > abs (a 4)

-- Sum of the first n terms of the arithmetic sequence
def S (n : ℕ) (a : ℕ → ℤ) : ℤ := n * (a 1 + a n) / 2

-- The main theorem we need to prove
theorem find_min_positive_n (a : ℕ → ℤ) (h1 : condition1 a) (h2 : condition2 a) : ∃ n : ℕ, n = 8 ∧ S n a > 0 :=
by
  sorry

end find_min_positive_n_l31_31758


namespace cos_alpha_value_l31_31469

open Real

theorem cos_alpha_value (α : ℝ) (h_cos : cos (α - π/6) = 15/17) (h_range : π/6 < α ∧ α < π/2) : 
  cos α = (15 * Real.sqrt 3 - 8) / 34 :=
by
  sorry

end cos_alpha_value_l31_31469


namespace two_digit_numbers_tens_greater_ones_l31_31737

theorem two_digit_numbers_tens_greater_ones : 
  ∃ (count : ℕ), count = 45 ∧ ∀ (n : ℕ), 10 ≤ n ∧ n < 100 → 
    let tens := n / 10;
    let ones := n % 10;
    tens > ones → count = 45 :=
by {
  sorry
}

end two_digit_numbers_tens_greater_ones_l31_31737


namespace ratio_of_areas_l31_31728

-- Definitions of the perimeters for each region
def perimeter_I : ℕ := 16
def perimeter_II : ℕ := 36
def perimeter_IV : ℕ := 48

-- Define the side lengths based on the given perimeters
def side_length (P : ℕ) : ℕ := P / 4

-- Calculate the areas from the side lengths
def area (s : ℕ) : ℕ := s * s

-- Now we state the theorem
theorem ratio_of_areas : 
  (area (side_length perimeter_II)) / (area (side_length perimeter_IV)) = 9 / 16 := 
by sorry

end ratio_of_areas_l31_31728


namespace abs_difference_of_numbers_l31_31512

theorem abs_difference_of_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 391) :
  |x - y| = 6 :=
sorry

end abs_difference_of_numbers_l31_31512


namespace popsicle_sticks_left_l31_31701

/-- Danielle has $10 for supplies. She buys one set of molds for $3, 
a pack of 100 popsicle sticks for $1. Each bottle of juice makes 20 popsicles and costs $2.
Prove that the number of popsicle sticks Danielle will be left with after making as many popsicles as she can is 40. -/
theorem popsicle_sticks_left (initial_money : ℕ)
    (mold_cost : ℕ) (sticks_cost : ℕ) (initial_sticks : ℕ)
    (juice_cost : ℕ) (popsicles_per_bottle : ℕ)
    (final_sticks : ℕ) :
    initial_money = 10 →
    mold_cost = 3 → 
    sticks_cost = 1 → 
    initial_sticks = 100 →
    juice_cost = 2 →
    popsicles_per_bottle = 20 →
    final_sticks = initial_sticks - (popsicles_per_bottle * (initial_money - mold_cost - sticks_cost) / juice_cost) →
    final_sticks = 40 :=
by
  intros h_initial_money h_mold_cost h_sticks_cost h_initial_sticks h_juice_cost h_popsicles_per_bottle h_final_sticks
  rw [h_initial_money, h_mold_cost, h_sticks_cost, h_initial_sticks, h_juice_cost, h_popsicles_per_bottle] at h_final_sticks
  norm_num at h_final_sticks
  exact h_final_sticks

end popsicle_sticks_left_l31_31701


namespace min_value_quadratic_l31_31068

theorem min_value_quadratic (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a * b = 1) :
  (∀ x, (a * x^2 + 2 * x + b = 0) → x = -1 / a) →
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 ∧ (∀ a b, a > b → b > 0 → a * b = 1 →
     c ≤ (a^2 + b^2) / (a - b)) :=
by
  sorry

end min_value_quadratic_l31_31068


namespace find_a_n_l31_31814

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : ∀ n, a n > 0)
  (h₂ : ∀ n, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
by
  sorry

end find_a_n_l31_31814


namespace log_expression_simplify_l31_31235

variable (x y : ℝ)

theorem log_expression_simplify (hx : 0 < x) (hx' : x ≠ 1) (hy : 0 < y) (hy' : y ≠ 1) :
  (Real.log x^2 / Real.log y^4) * 
  (Real.log y^3 / Real.log x^3) * 
  (Real.log x^4 / Real.log y^5) * 
  (Real.log y^5 / Real.log x^2) * 
  (Real.log x^3 / Real.log y^3) = (1 / 3) * Real.log x / Real.log y := 
sorry

end log_expression_simplify_l31_31235


namespace part1_l31_31081

variable (α : ℝ)

theorem part1 (h : Real.tan α = 2) : (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 :=
by
  sorry

end part1_l31_31081


namespace total_pieces_of_junk_mail_l31_31684

-- Definition of the problem based on given conditions
def pieces_per_house : ℕ := 4
def number_of_blocks : ℕ := 16
def houses_per_block : ℕ := 17

-- Statement of the theorem to prove the total number of pieces of junk mail
theorem total_pieces_of_junk_mail :
  (houses_per_block * pieces_per_house * number_of_blocks) = 1088 :=
by
  sorry

end total_pieces_of_junk_mail_l31_31684


namespace sum_of_repeating_decimals_l31_31888

-- Declare the repeating decimals as constants
def x : ℚ := 2/3
def y : ℚ := 7/9

-- The problem statement
theorem sum_of_repeating_decimals : x + y = 13 / 9 := by
  sorry

end sum_of_repeating_decimals_l31_31888


namespace find_positive_number_l31_31606

theorem find_positive_number (m : ℝ) 
  (h : (m - 1)^2 = (3 * m - 5)^2) : 
  (m - 1)^2 = 1 ∨ (m - 1)^2 = 1 / 4 :=
by sorry

end find_positive_number_l31_31606


namespace equation1_equation2_equation3_equation4_l31_31660

-- 1. Solve: 2(2x-1)^2 = 8
theorem equation1 (x : ℝ) : 2 * (2 * x - 1)^2 = 8 ↔ (x = 3/2) ∨ (x = -1/2) :=
sorry

-- 2. Solve: 2x^2 + 3x - 2 = 0
theorem equation2 (x : ℝ) : 2 * x^2 + 3 * x - 2 = 0 ↔ (x = 1/2) ∨ (x = -2) :=
sorry

-- 3. Solve: x(2x-7) = 3(2x-7)
theorem equation3 (x : ℝ) : x * (2 * x - 7) = 3 * (2 * x - 7) ↔ (x = 7/2) ∨ (x = 3) :=
sorry

-- 4. Solve: 2y^2 + 8y - 1 = 0
theorem equation4 (y : ℝ) : 2 * y^2 + 8 * y - 1 = 0 ↔ (y = (-4 + 3 * Real.sqrt 2) / 2) ∨ (y = (-4 - 3 * Real.sqrt 2) / 2) :=
sorry

end equation1_equation2_equation3_equation4_l31_31660


namespace sin_cos_relation_l31_31826

theorem sin_cos_relation 
  (α β : Real) 
  (h : 2 * Real.sin α - Real.cos β = 2) 
  : Real.sin α + 2 * Real.cos β = 1 ∨ Real.sin α + 2 * Real.cos β = -1 := 
sorry

end sin_cos_relation_l31_31826


namespace no_int_solutions_l31_31339

open Nat

theorem no_int_solutions (p1 p2 α n : ℕ)
  (hp1_prime : p1.Prime)
  (hp2_prime : p2.Prime)
  (hp1_odd : p1 % 2 = 1)
  (hp2_odd : p2 % 2 = 1)
  (hα_pos : 0 < α)
  (hn_pos : 0 < n)
  (hα_gt1 : 1 < α)
  (hn_gt1 : 1 < n) :
  ¬(let lhs := ((p2 - 1) / 2) ^ p1 + ((p2 + 1) / 2) ^ p1
    lhs = α ^ n) :=
sorry

end no_int_solutions_l31_31339


namespace segment_ratios_correct_l31_31843

noncomputable def compute_segment_ratios : (ℕ × ℕ) :=
  let ratio := 20 / 340;
  let gcd := Nat.gcd 1 17;
  if (ratio = 1 / 17) ∧ (gcd = 1) then (1, 17) else (0, 0) 

theorem segment_ratios_correct : 
  compute_segment_ratios = (1, 17) := 
by
  sorry

end segment_ratios_correct_l31_31843


namespace determinant_property_l31_31136

variable {R : Type} [CommRing R]
variable (x y z w : R)

theorem determinant_property 
  (h : x * w - y * z = 7) :
  (x + 2 * z) * w - (y + 2 * w) * z = 7 :=
by sorry

end determinant_property_l31_31136


namespace club_members_l31_31231

variable (x : ℕ)

theorem club_members (h1 : 2 * x + 5 = x + 15) : x = 10 := by
  sorry

end club_members_l31_31231


namespace milk_total_correct_l31_31824

def chocolate_milk : Nat := 2
def strawberry_milk : Nat := 15
def regular_milk : Nat := 3
def total_milk : Nat := chocolate_milk + strawberry_milk + regular_milk

theorem milk_total_correct : total_milk = 20 := by
  sorry

end milk_total_correct_l31_31824


namespace complex_plane_squares_areas_l31_31113

theorem complex_plane_squares_areas (z : ℂ) 
  (h1 : z^3 - z = i * (z^2 - z) ∨ z^3 - z = -i * (z^2 - z))
  (h2 : z^4 - z = i * (z^3 - z) ∨ z^4 - z = -i * (z^3 - z)) :
  ( ∃ A₁ A₂ : ℝ, (A₁ = 10 ∨ A₁ = 18) ∧ (A₂ = 10 ∨ A₂ = 18) ) := 
sorry

end complex_plane_squares_areas_l31_31113


namespace length_of_CD_l31_31377

theorem length_of_CD (L : ℝ) (r : ℝ) (V_total : ℝ) (cylinder_vol : ℝ) (hemisphere_vol : ℝ) : 
  r = 5 ∧ V_total = 900 * Real.pi ∧ cylinder_vol = Real.pi * r^2 * L ∧ hemisphere_vol = (2/3) *Real.pi * r^3 → 
  V_total = cylinder_vol + 2 * hemisphere_vol → 
  L = 88 / 3 := 
by
  sorry

end length_of_CD_l31_31377


namespace flowchart_output_correct_l31_31682

-- Define the conditions of the problem
def program_flowchart (initial : ℕ) : ℕ :=
  let step1 := initial * 2
  let step2 := step1 * 2
  let step3 := step2 * 2
  step3

-- State the proof problem
theorem flowchart_output_correct : program_flowchart 1 = 8 :=
by
  -- Sorry to skip the proof
  sorry

end flowchart_output_correct_l31_31682


namespace map_distance_to_actual_distance_l31_31892

theorem map_distance_to_actual_distance
  (map_distance : ℝ)
  (scale_map_to_real : ℝ)
  (scale_real_distance : ℝ)
  (H_map_distance : map_distance = 18)
  (H_scale_map : scale_map_to_real = 0.5)
  (H_scale_real : scale_real_distance = 6) :
  (map_distance / scale_map_to_real) * scale_real_distance = 216 :=
by
  sorry

end map_distance_to_actual_distance_l31_31892


namespace remainder_when_3n_plus_2_squared_divided_by_11_l31_31352

theorem remainder_when_3n_plus_2_squared_divided_by_11 (n : ℕ) (h : n % 7 = 5) : ((3 * n + 2)^2) % 11 = 3 :=
  sorry

end remainder_when_3n_plus_2_squared_divided_by_11_l31_31352


namespace part_a_part_b_l31_31463

-- Define what it means for a number to be "surtido"
def is_surtido (A : ℕ) : Prop :=
  ∀ n, (1 ≤ n → n ≤ (A.digits 10).sum → ∃ B : ℕ, n = (B.digits 10).sum) 

-- Part (a): Prove that if 1, 2, 3, 4, 5, 6, 7, and 8 can be expressed as sums of digits in A, then A is "surtido".
theorem part_a (A : ℕ)
  (h1 : ∃ B1 : ℕ, 1 = (B1.digits 10).sum)
  (h2 : ∃ B2 : ℕ, 2 = (B2.digits 10).sum)
  (h3 : ∃ B3 : ℕ, 3 = (B3.digits 10).sum)
  (h4 : ∃ B4 : ℕ, 4 = (B4.digits 10).sum)
  (h5 : ∃ B5 : ℕ, 5 = (B5.digits 10).sum)
  (h6 : ∃ B6 : ℕ, 6 = (B6.digits 10).sum)
  (h7 : ∃ B7 : ℕ, 7 = (B7.digits 10).sum)
  (h8 : ∃ B8 : ℕ, 8 = (B8.digits 10).sum) : is_surtido A :=
sorry

-- Part (b): Determine if having the sums 1, 2, 3, 4, 5, 6, and 7 as sums of digits in A implies that A is "surtido".
theorem part_b (A : ℕ)
  (h1 : ∃ B1 : ℕ, 1 = (B1.digits 10).sum)
  (h2 : ∃ B2 : ℕ, 2 = (B2.digits 10).sum)
  (h3 : ∃ B3 : ℕ, 3 = (B3.digits 10).sum)
  (h4 : ∃ B4 : ℕ, 4 = (B4.digits 10).sum)
  (h5 : ∃ B5 : ℕ, 5 = (B5.digits 10).sum)
  (h6 : ∃ B6 : ℕ, 6 = (B6.digits 10).sum)
  (h7 : ∃ B7 : ℕ, 7 = (B7.digits 10).sum) : ¬is_surtido A :=
sorry

end part_a_part_b_l31_31463


namespace train_speed_conversion_l31_31654

-- Define the speed of the train in meters per second.
def speed_mps : ℝ := 37.503

-- Definition of the conversion factor between m/s and km/h.
def conversion_factor : ℝ := 3.6

-- Define the expected speed of the train in kilometers per hour.
def expected_speed_kmph : ℝ := 135.0108

-- Prove that the speed in km/h is the expected value.
theorem train_speed_conversion :
  (speed_mps * conversion_factor = expected_speed_kmph) :=
by
  sorry

end train_speed_conversion_l31_31654


namespace work_completed_together_l31_31054

theorem work_completed_together (A_days B_days : ℕ) (hA : A_days = 40) (hB : B_days = 60) : 
  1 / (1 / (A_days: ℝ) + 1 / (B_days: ℝ)) = 24 :=
by
  sorry

end work_completed_together_l31_31054


namespace sequence_value_l31_31150

theorem sequence_value (a b c d x : ℕ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 17) (h4 : d = 33)
  (h5 : b - a = 4) (h6 : c - b = 8) (h7 : d - c = 16) (h8 : x - d = 32) : x = 65 := by
  sorry

end sequence_value_l31_31150


namespace x_y_quartic_l31_31903

theorem x_y_quartic (x y : ℝ) (h₁ : x - y = 2) (h₂ : x * y = 48) : x^4 + y^4 = 5392 := by
  sorry

end x_y_quartic_l31_31903


namespace circle_represents_range_l31_31982

theorem circle_represents_range (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x - 2 * y + 3 = 0 → (m > 2 * Real.sqrt 2 ∨ m < -2 * Real.sqrt 2)) :=
by
  sorry

end circle_represents_range_l31_31982


namespace solve_for_2023_minus_a_minus_2b_l31_31539

theorem solve_for_2023_minus_a_minus_2b (a b : ℝ) (h : 1^2 + a*1 + 2*b = 0) : 2023 - a - 2*b = 2024 := 
by sorry

end solve_for_2023_minus_a_minus_2b_l31_31539


namespace allison_greater_probability_l31_31353

-- Definitions and conditions for the problem
def faceRollAllison : Nat := 6
def facesBrian : List Nat := [1, 3, 3, 5, 5, 6]
def facesNoah : List Nat := [4, 4, 4, 4, 5, 5]

-- Function to calculate probability
def probability_less_than (faces : List Nat) (value : Nat) : ℚ :=
  (faces.filter (fun x => x < value)).length / faces.length

-- Main theorem statement
theorem allison_greater_probability :
  probability_less_than facesBrian 6 * probability_less_than facesNoah 6 = 5 / 6 := by
  sorry

end allison_greater_probability_l31_31353


namespace snack_eaters_initial_count_l31_31433

-- Define all variables and conditions used in the problem
variables (S : ℕ) (initial_people : ℕ) (new_outsiders_1 : ℕ) (new_outsiders_2 : ℕ) (left_after_first_half : ℕ) (left_after_second_half : ℕ) (remaining_snack_eaters : ℕ)

-- Assign the specific values according to conditions
def conditions := 
  initial_people = 200 ∧
  new_outsiders_1 = 20 ∧
  new_outsiders_2 = 10 ∧
  left_after_first_half = (S + new_outsiders_1) / 2 ∧
  left_after_second_half = left_after_first_half + new_outsiders_2 - 30 ∧
  remaining_snack_eaters = left_after_second_half / 2 ∧
  remaining_snack_eaters = 20

-- State the theorem to prove
theorem snack_eaters_initial_count (S : ℕ) (initial_people new_outsiders_1 new_outsiders_2 left_after_first_half left_after_second_half remaining_snack_eaters : ℕ) :
  conditions S initial_people new_outsiders_1 new_outsiders_2 left_after_first_half left_after_second_half remaining_snack_eaters → S = 100 :=
by sorry

end snack_eaters_initial_count_l31_31433


namespace jeans_price_difference_l31_31474

variable (x : Real)

theorem jeans_price_difference
  (hx : 0 < x) -- Assuming x > 0 for a positive cost
  (r := 1.40 * x)
  (c := 1.30 * r) :
  c = 1.82 * x :=
by
  sorry

end jeans_price_difference_l31_31474


namespace parabola_slopes_l31_31815

theorem parabola_slopes (k : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) 
    (hC : C = (0, -2)) (hA : A.1^2 = 2 * A.2) (hB : B.1^2 = 2 * B.2) 
    (hA_eq : A.2 = k * A.1 + 2) (hB_eq : B.2 = k * B.1 + 2) :
  ((C.2 - A.2) / (C.1 - A.1))^2 + ((C.2 - B.2) / (C.1 - B.1))^2 - 2 * k^2 = 8 := 
sorry

end parabola_slopes_l31_31815


namespace distance_from_A_to_B_l31_31110

-- Definitions of the conditions
def avg_speed : ℝ := 25
def distance_AB (D : ℝ) : Prop := ∃ T : ℝ, D / (4 * T) = avg_speed ∧ D = 3 * (T * avg_speed)∧ (D / 2) = (T * avg_speed)

theorem distance_from_A_to_B : ∃ D : ℝ, distance_AB D ∧ D = 100 / 3 :=
by
  sorry

end distance_from_A_to_B_l31_31110


namespace wilson_pays_total_l31_31907

def hamburger_price : ℝ := 5
def cola_price : ℝ := 2
def fries_price : ℝ := 3
def sundae_price : ℝ := 4
def discount_coupon : ℝ := 4
def loyalty_discount : ℝ := 0.10

def total_cost_before_discounts : ℝ :=
  2 * hamburger_price + 3 * cola_price + fries_price + sundae_price

def total_cost_after_coupon : ℝ :=
  total_cost_before_discounts - discount_coupon

def loyalty_discount_amount : ℝ :=
  loyalty_discount * total_cost_after_coupon

def total_cost_after_all_discounts : ℝ :=
  total_cost_after_coupon - loyalty_discount_amount

theorem wilson_pays_total : total_cost_after_all_discounts = 17.10 :=
  sorry

end wilson_pays_total_l31_31907


namespace star_evaluation_l31_31786

def star (X Y : ℚ) := (X + Y) / 4

theorem star_evaluation : star (star 3 8) 6 = 35 / 16 := by
  sorry

end star_evaluation_l31_31786


namespace geom_seq_sum_eq_six_l31_31049

theorem geom_seq_sum_eq_six 
    (a : ℕ → ℝ) 
    (r : ℝ) 
    (h_geom : ∀ n, a (n + 1) = a n * r) 
    (h_pos : ∀ n, a n > 0)
    (h_eq : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) 
    : a 5 + a 7 = 6 :=
sorry

end geom_seq_sum_eq_six_l31_31049


namespace percentage_of_orange_and_watermelon_juice_l31_31097

-- Define the total volume of the drink
def total_volume := 150

-- Define the volume of grape juice in the drink
def grape_juice_volume := 45

-- Define the percentage calculation for grape juice
def grape_juice_percentage := (grape_juice_volume / total_volume) * 100

-- Define the remaining percentage that is made of orange and watermelon juices
def remaining_percentage := 100 - grape_juice_percentage

-- Define the percentage of orange and watermelon juice being the same
def orange_and_watermelon_percentage := remaining_percentage / 2

theorem percentage_of_orange_and_watermelon_juice : 
  orange_and_watermelon_percentage = 35 :=
by
  -- The proof steps would go here
  sorry

end percentage_of_orange_and_watermelon_juice_l31_31097


namespace fib_100_mod_5_l31_31079

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end fib_100_mod_5_l31_31079


namespace square_completing_l31_31510

theorem square_completing (b c : ℤ) (h : (x^2 - 10 * x + 15 = 0) → ((x + b)^2 = c)) : 
  b + c = 5 :=
sorry

end square_completing_l31_31510


namespace total_yen_l31_31368

/-- 
Abe's family has a checking account with 6359 yen
and a savings account with 3485 yen.
-/
def checking_account : ℕ := 6359
def savings_account : ℕ := 3485

/-- 
Prove that the total amount of yen Abe's family has
is equal to 9844 yen.
-/
theorem total_yen : checking_account + savings_account = 9844 :=
by
  sorry

end total_yen_l31_31368


namespace seven_a_plus_seven_b_l31_31196

noncomputable def g (x : ℝ) : ℝ := 7 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 4
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem seven_a_plus_seven_b (a b : ℝ) (h₁ : ∀ x, g x = f_inv x - 2) (h₂ : ∀ x, f_inv (f x a b) = x) :
  7 * a + 7 * b = 5 :=
by
  sorry

end seven_a_plus_seven_b_l31_31196


namespace total_games_l31_31650

variable (L : ℕ) -- Number of games the team lost

-- Define the number of wins
def Wins := 3 * L + 14

theorem total_games (h_wins : Wins = 101) : (Wins + L = 130) :=
by
  sorry

end total_games_l31_31650


namespace part1_part2_l31_31568

open Real

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| - 1

-- Define the function g for the second part
def g (x : ℝ) : ℝ := |x - 2| + |x + 3|

theorem part1 (m : ℝ) : (∀ x, f x m ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) → m = 2 :=
  by sorry

theorem part2 (t x: ℝ) (h: ∀ x: ℝ, f x 2 + f (x + 5) 2 ≥ t - 2) : t ≤ 5 :=
  by sorry

end part1_part2_l31_31568


namespace polygon_sides_l31_31636

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
by sorry

end polygon_sides_l31_31636


namespace number_of_cats_l31_31309

variable (C D : ℕ)

-- Conditions
def condition1 : Prop := C = 15 * D / 7
def condition2 : Prop := C = 15 * (D + 12) / 11

-- Proof problem
theorem number_of_cats (h1 : condition1 C D) (h2 : condition2 C D) : C = 45 := sorry

end number_of_cats_l31_31309


namespace Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction_l31_31396

noncomputable def Thabo_book_count_problem : Prop :=
  let P := Nat
  let F := Nat
  ∃ (P F : Nat), 
    -- Conditions
    (P > 40) ∧ 
    (F = 2 * P) ∧ 
    (F + P + 40 = 220) ∧ 
    -- Conclusion
    (P - 40 = 20)

theorem Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction : Thabo_book_count_problem :=
  sorry

end Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction_l31_31396


namespace road_length_in_km_l31_31929

/-- The actual length of the road in kilometers is 7.5, given the scale of 1:50000 
    and the map length of 15 cm. -/

theorem road_length_in_km (s : ℕ) (map_length_cm : ℕ) (actual_length_cm : ℕ) (actual_length_km : ℝ) 
  (h_scale : s = 50000) (h_map_length : map_length_cm = 15) (h_conversion : actual_length_km = actual_length_cm / 100000) :
  actual_length_km = 7.5 :=
  sorry

end road_length_in_km_l31_31929


namespace probability_red_given_spade_or_king_l31_31488

def num_cards := 52
def num_spades := 13
def num_kings := 4
def num_red_kings := 2

def num_non_spade_kings := num_kings - 1
def num_spades_or_kings := num_spades + num_non_spade_kings

theorem probability_red_given_spade_or_king :
  (num_red_kings : ℚ) / num_spades_or_kings = 1 / 8 :=
sorry

end probability_red_given_spade_or_king_l31_31488


namespace x_in_interval_l31_31797

theorem x_in_interval (x : ℝ) (h : x = (1 / x) * (-x) + 2) : 0 < x ∧ x ≤ 2 :=
by
  -- Place the proof here
  sorry

end x_in_interval_l31_31797


namespace quadratic_distinct_real_roots_l31_31083

theorem quadratic_distinct_real_roots (k : ℝ) : 
  (∀ (x : ℝ), (k - 1) * x^2 + 4 * x + 1 = 0 → False) ↔ (k < 5 ∧ k ≠ 1) :=
by
  sorry

end quadratic_distinct_real_roots_l31_31083


namespace apple_bags_l31_31514

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l31_31514


namespace equalities_implied_by_sum_of_squares_l31_31712

variable {a b c d : ℝ}

theorem equalities_implied_by_sum_of_squares (h1 : a = b) (h2 : c = d) : 
  (a - b) ^ 2 + (c - d) ^ 2 = 0 :=
sorry

end equalities_implied_by_sum_of_squares_l31_31712


namespace prob_score_5_points_is_three_over_eight_l31_31253

noncomputable def probability_of_scoring_5_points : ℚ :=
  let total_events := 2^3
  let favorable_events := 3 -- Calculated from combinatorial logic.
  favorable_events / total_events

theorem prob_score_5_points_is_three_over_eight :
  probability_of_scoring_5_points = 3 / 8 :=
by
  sorry

end prob_score_5_points_is_three_over_eight_l31_31253


namespace three_digit_number_is_275_l31_31403

noncomputable def digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100 % 10, n / 10 % 10, n % 10)

theorem three_digit_number_is_275 :
  ∃ (n : ℕ), n / 100 % 10 + n % 10 = n / 10 % 10 ∧
              7 * (n / 100 % 10) = n % 10 + n / 10 % 10 + 2 ∧
              n / 100 % 10 + n / 10 % 10 + n % 10 = 14 ∧
              n = 275 :=
by
  sorry

end three_digit_number_is_275_l31_31403


namespace height_of_triangle_on_parabola_l31_31444

open Real

theorem height_of_triangle_on_parabola
  (x0 x1 : ℝ)
  (y0 y1 : ℝ)
  (hA : y0 = x0^2)
  (hB : y0 = (-x0)^2)
  (hC : y1 = x1^2)
  (hypotenuse_parallel : y0 = y1 + 1):
  y0 - y1 = 1 := 
by
  sorry

end height_of_triangle_on_parabola_l31_31444


namespace eval_expr_l31_31816

theorem eval_expr : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end eval_expr_l31_31816


namespace rowing_distance_l31_31331

theorem rowing_distance
  (rowing_speed_in_still_water : ℝ)
  (velocity_of_current : ℝ)
  (total_time : ℝ)
  (H1 : rowing_speed_in_still_water = 5)
  (H2 : velocity_of_current = 1)
  (H3 : total_time = 1) :
  ∃ (D : ℝ), D = 2.4 := 
sorry

end rowing_distance_l31_31331


namespace airplane_fraction_l31_31157

noncomputable def driving_time : ℕ := 195

noncomputable def airport_drive_time : ℕ := 10

noncomputable def waiting_time : ℕ := 20

noncomputable def get_off_time : ℕ := 10

noncomputable def faster_by : ℕ := 90

theorem airplane_fraction :
  ∃ x : ℕ, 195 = 40 + x + 90 ∧ x = 65 ∧ x = driving_time / 3 := sorry

end airplane_fraction_l31_31157


namespace part1_part2_l31_31581

variable (a b : ℝ)

-- Part (1)
theorem part1 (hA : a^2 - 2 * a * b + b^2 = A) (hB: a^2 + 2 * a * b + b^2 = B) (h : a ≠ b) :
  A + B > 0 := sorry

-- Part (2)
theorem part2 (hA : a^2 - 2 * a * b + b^2 = A) (hB: a^2 + 2 * a * b + b^2 = B) (h: a * b = 1) : 
  A - B = -4 := sorry

end part1_part2_l31_31581


namespace solve_for_x_l31_31967

-- Definition of the operation
def otimes (a b : ℝ) : ℝ := a^2 + b^2 - a * b

-- The mathematical statement to be proved
theorem solve_for_x (x : ℝ) (h : otimes x (x - 1) = 3) : x = 2 ∨ x = -1 := 
by 
  sorry

end solve_for_x_l31_31967


namespace value_of_a4_l31_31424

open Nat

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n + 1

theorem value_of_a4 (a : ℕ → ℕ) (h : sequence a) : a 4 = 23 :=
by
  -- Proof to be provided or implemented
  sorry

end value_of_a4_l31_31424


namespace smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450_l31_31316

theorem smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450 :
  ∃ n : ℕ, (n - 10) % 12 = 0 ∧
           (n - 10) % 16 = 0 ∧
           (n - 10) % 18 = 0 ∧
           (n - 10) % 21 = 0 ∧
           (n - 10) % 28 = 0 ∧
           (n - 10) % 35 = 0 ∧
           (n - 10) % 40 = 0 ∧
           (n - 10) % 45 = 0 ∧
           (n - 10) % 55 = 0 ∧
           n = 55450 :=
by
  sorry

end smallest_number_divisible_by_given_numbers_diminished_by_10_is_55450_l31_31316


namespace sum_of_roots_l31_31880

variable {p m n : ℝ}

axiom roots_condition (h : m * n = 4) : m + n = -4

theorem sum_of_roots (h : m * n = 4) : m + n = -4 := 
  roots_condition h

end sum_of_roots_l31_31880


namespace tetrahedron_solution_l31_31154

noncomputable def num_triangles (a : ℝ) (E F G : ℝ → ℝ → ℝ) : ℝ :=
  if a > 3 then 3 else 0

theorem tetrahedron_solution (a : ℝ) (E F G : ℝ → ℝ → ℝ) :
  a > 3 → num_triangles a E F G = 3 := by
  sorry

end tetrahedron_solution_l31_31154


namespace person_B_work_days_l31_31332

-- Let a be the work rate for person A, and b be the work rate for person B.
-- a completes the work in 20 days
-- b completes the work in x days
-- When working together, a and b complete 0.375 of the work in 5 days


theorem person_B_work_days (x : ℝ) :
  ((5 : ℝ) * ((1 / 20) + 1 / x) = 0.375) -> x = 40 := 
by 
  sorry

end person_B_work_days_l31_31332


namespace part1_l31_31644

noncomputable def f (a x : ℝ) : ℝ := a * x - 2 * Real.log x + 2 * (1 + a) + (a - 2) / x

theorem part1 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 1 ≤ x → f a x ≥ 0) ↔ 1 ≤ a :=
sorry

end part1_l31_31644


namespace no_discount_profit_percentage_l31_31333

noncomputable def cost_price : ℝ := 100
noncomputable def discount_percentage : ℝ := 4 / 100  -- 4%
noncomputable def profit_percentage_with_discount : ℝ := 20 / 100  -- 20%

theorem no_discount_profit_percentage : 
  (1 + profit_percentage_with_discount) * cost_price / (1 - discount_percentage) / cost_price - 1 = 0.25 := by
  sorry

end no_discount_profit_percentage_l31_31333


namespace money_has_48_l31_31979

-- Definitions derived from conditions:
def money (p : ℝ) := 
  p = (1/3 * p) + 32

-- The main theorem statement
theorem money_has_48 (p : ℝ) : money p → p = 48 := by
  intro h
  -- Skipping the proof
  sorry

end money_has_48_l31_31979


namespace minimum_guests_l31_31669

theorem minimum_guests (x : ℕ) : (120 + 18 * x > 250 + 15 * x) → (x ≥ 44) := by
  intro h
  sorry

end minimum_guests_l31_31669


namespace sufficient_but_not_necessary_condition_l31_31780

theorem sufficient_but_not_necessary_condition (x : ℝ) : 
  (x > 2 → (x-1)^2 > 1) ∧ (∃ (y : ℝ), y ≤ 2 ∧ (y-1)^2 > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l31_31780


namespace probability_third_white_no_replacement_probability_red_no_more_than_4_in_6_draws_with_replacement_l31_31175

noncomputable section

-- Problem 1: Probability of drawing a white ball on the third draw without replacement is 1/3.
theorem probability_third_white_no_replacement :
  let red := 4
  let white := 2
  let totalBalls := red + white
  let totalWaysToDraw3 := Nat.choose totalBalls 3
  let favorableWays := Nat.choose (totalBalls - 1) 2 * Nat.choose white 1
  let probability := favorableWays / totalWaysToDraw3
  probability = 1 / 3 :=
by
  sorry

-- Problem 2: Probability of drawing red balls no more than 4 times in 6 draws with replacement is 441/729.
theorem probability_red_no_more_than_4_in_6_draws_with_replacement :
  let red := 4
  let white := 2
  let totalBalls := red + white
  let p_red := red / totalBalls
  let p_X5 := Nat.choose 6 5 * p_red^5 * (1 - p_red)
  let p_X6 := Nat.choose 6 6 * p_red^6
  let probability := 1 - p_X5 - p_X6
  probability = 441 / 729 :=
by
  sorry

end probability_third_white_no_replacement_probability_red_no_more_than_4_in_6_draws_with_replacement_l31_31175


namespace problem1_problem2_problem3_l31_31658

-- Problem 1
theorem problem1 (x : ℝ) (h : x^2 + x - 2 = 0) : x^2 + x + 2023 = 2025 := 
  sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h : a + b = 5) : 2 * (a + b) - 4 * a - 4 * b + 21 = 11 := 
  sorry

-- Problem 3
theorem problem3 (a b : ℝ) (h1 : a^2 + 3 * a * b = 20) (h2 : b^2 + 5 * a * b = 8) : 2 * a^2 - b^2 + a * b = 32 := 
  sorry

end problem1_problem2_problem3_l31_31658


namespace price_for_3years_service_l31_31335

def full_price : ℝ := 85
def discount_price_1year (price : ℝ) : ℝ := price - (0.20 * price)
def discount_price_3years (price : ℝ) : ℝ := price - (0.25 * price)

theorem price_for_3years_service : discount_price_3years (discount_price_1year full_price) = 51 := 
by 
  sorry

end price_for_3years_service_l31_31335


namespace value_of_expression_l31_31839

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 3*x^2 + 9*x - 2 = 4 :=
by
  -- The proof will be filled here; it's currently skipped using 'sorry'
  sorry

end value_of_expression_l31_31839


namespace number_of_wheels_l31_31393

theorem number_of_wheels (V : ℕ) (W_2 : ℕ) (n : ℕ) 
  (hV : V = 16) 
  (h_eq : 2 * W_2 + 16 * n = 66) : 
  n = 4 := 
by 
  sorry

end number_of_wheels_l31_31393


namespace eval_expr_at_neg3_l31_31860

theorem eval_expr_at_neg3 : 
  (5 + 2 * (-3) * ((-3) + 5) - 5^2) / (2 * (-3) - 5 + 2 * (-3)^3) = 32 / 65 := 
by 
  sorry

end eval_expr_at_neg3_l31_31860


namespace identity_holds_l31_31202

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l31_31202


namespace average_grade_of_male_students_l31_31721

theorem average_grade_of_male_students (M : ℝ) (H1 : (90 : ℝ) = (8 + 32 : ℝ) / 40) 
(H2 : (92 : ℝ) = 32 / 40) :
  M = 82 := 
sorry

end average_grade_of_male_students_l31_31721


namespace value_of_expression_l31_31902

theorem value_of_expression (m n : ℝ) (h : m + n = 3) :
  2 * m^2 + 4 * m * n + 2 * n^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l31_31902


namespace dog_ate_cost_6_l31_31872

noncomputable def totalCost : ℝ := 4 + 2 + 0.5 + 2.5
noncomputable def costPerSlice : ℝ := totalCost / 6
noncomputable def slicesEatenByDog : ℕ := 6 - 2
noncomputable def costEatenByDog : ℝ := slicesEatenByDog * costPerSlice

theorem dog_ate_cost_6 : costEatenByDog = 6 := by
  sorry

end dog_ate_cost_6_l31_31872


namespace maximum_value_of_f_on_interval_l31_31532

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x + Real.sin x

theorem maximum_value_of_f_on_interval :
  ∃ M, M = Real.pi ∧ ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ M :=
by
  sorry

end maximum_value_of_f_on_interval_l31_31532


namespace high_probability_event_is_C_l31_31703

-- Define the probabilities of events A, B, and C
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.1
def prob_C : ℝ := 0.9

-- Statement asserting Event C has the high possibility of occurring
theorem high_probability_event_is_C : prob_C > prob_A ∧ prob_C > prob_B :=
by
  sorry

end high_probability_event_is_C_l31_31703


namespace first_floor_cost_l31_31492

-- Definitions and assumptions
variables (F : ℝ)
variables (earnings_first_floor earnings_second_floor earnings_third_floor : ℝ)
variables (total_monthly_earnings : ℝ)

-- Conditions from the problem
def costs := F
def second_floor_costs := F + 20
def third_floor_costs := 2 * F
def first_floor_rooms := 3 * costs
def second_floor_rooms := 3 * second_floor_costs
def third_floor_rooms := 3 * third_floor_costs

-- Total monthly earnings
def total_earnings := first_floor_rooms + second_floor_rooms + third_floor_rooms

-- Equality condition
axiom total_earnings_is_correct : total_earnings = 165

-- Theorem to be proved
theorem first_floor_cost :
  (F = 8.75) :=
by
  have earnings_first_floor_eq := first_floor_rooms
  have earnings_second_floor_eq := second_floor_rooms
  have earnings_third_floor_eq := third_floor_rooms
  have total_earning_eq := total_earnings_is_correct
  sorry

end first_floor_cost_l31_31492


namespace evaluate_expression_l31_31139

-- Define the conditions
def num : ℤ := 900^2
def a : ℤ := 306
def b : ℤ := 294
def denom : ℤ := a^2 - b^2

-- State the theorem to be proven
theorem evaluate_expression : (num : ℚ) / denom = 112.5 :=
by
  -- proof is skipped
  sorry

end evaluate_expression_l31_31139


namespace Petya_receives_last_wrapper_l31_31187

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem Petya_receives_last_wrapper
  (h1 : discriminant a b c ≥ 0)
  (h2 : discriminant a c b ≥ 0)
  (h3 : discriminant b a c ≥ 0)
  (h4 : discriminant c a b < 0)
  (h5 : discriminant b c a < 0) :
  discriminant c b a ≥ 0 :=
sorry

end Petya_receives_last_wrapper_l31_31187


namespace sum_units_tens_not_divisible_by_4_l31_31522

theorem sum_units_tens_not_divisible_by_4 :
  ∃ (n : ℕ), (n = 3674 ∨ n = 3684 ∨ n = 3694 ∨ n = 3704 ∨ n = 3714 ∨ n = 3722) ∧
  (¬ (∃ k, (n % 100) = 4 * k)) ∧
  ((n % 10) + (n / 10 % 10) = 11) :=
sorry

end sum_units_tens_not_divisible_by_4_l31_31522


namespace initial_number_of_children_l31_31847

-- Define the initial conditions
variables {X : ℕ} -- Initial number of children on the bus
variables (got_off got_on children_after : ℕ)
variables (H1 : got_off = 10)
variables (H2 : got_on = 5)
variables (H3 : children_after = 16)

-- Define the theorem to be proved
theorem initial_number_of_children (H : X - got_off + got_on = children_after) : X = 21 :=
by sorry

end initial_number_of_children_l31_31847


namespace marian_baked_cookies_l31_31733

theorem marian_baked_cookies :
  let cookies_per_tray := 12
  let trays_used := 23
  trays_used * cookies_per_tray = 276 :=
by
  sorry

end marian_baked_cookies_l31_31733


namespace binomial_expansion_fraction_l31_31505

theorem binomial_expansion_fraction :
  let a0 := 32
  let a1 := -80
  let a2 := 80
  let a3 := -40
  let a4 := 10
  let a5 := -1
  (2 - x)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 →
  (a0 + a2 + a4) / (a1 + a3) = -61 / 60 :=
by
  sorry

end binomial_expansion_fraction_l31_31505


namespace complete_square_l31_31310

theorem complete_square (x : ℝ) (h : x^2 + 8 * x + 9 = 0) : (x + 4)^2 = 7 := by
  sorry

end complete_square_l31_31310


namespace advantageous_bank_l31_31321

variable (C : ℝ) (p n : ℝ)

noncomputable def semiAnnualCompounding (p : ℝ) (n : ℝ) : ℝ :=
  (1 + p / (2 * 100)) ^ n

noncomputable def monthlyCompounding (p : ℝ) (n : ℝ) : ℝ :=
  (1 + p / (12 * 100)) ^ (6 * n)

theorem advantageous_bank (p n : ℝ) :
  monthlyCompounding p n - semiAnnualCompounding p n > 0 := sorry

#check advantageous_bank

end advantageous_bank_l31_31321


namespace find_number_l31_31760

variable (x : ℝ)

theorem find_number (h : 0.46 * x = 165.6) : x = 360 :=
sorry

end find_number_l31_31760


namespace basketball_team_avg_weight_l31_31785

theorem basketball_team_avg_weight :
  let n_tallest := 5
  let w_tallest := 90
  let n_shortest := 4
  let w_shortest := 75
  let n_remaining := 3
  let w_remaining := 80
  let total_weight := (n_tallest * w_tallest) + (n_shortest * w_shortest) + (n_remaining * w_remaining)
  let total_players := n_tallest + n_shortest + n_remaining
  (total_weight / total_players) = 82.5 :=
by
  sorry

end basketball_team_avg_weight_l31_31785


namespace smallest_d_l31_31876

theorem smallest_d (d t s : ℕ) (h1 : 3 * t - 4 * s = 2023)
                   (h2 : t = s + d) 
                   (h3 : 4 * s > 0)
                   (h4 : d % 3 = 0) :
                   d = 675 := sorry

end smallest_d_l31_31876


namespace inscribed_square_area_l31_31875

theorem inscribed_square_area (R : ℝ) (h : (R^2 * (π - 2) / 4) = (2 * π - 4)) : 
  ∃ (a : ℝ), a^2 = 16 := by
  sorry

end inscribed_square_area_l31_31875


namespace factor_theorem_for_Q_l31_31261

variable (d : ℝ) -- d is a real number

def Q (x : ℝ) : ℝ := x^3 + 3 * x^2 + d * x + 20

theorem factor_theorem_for_Q :
  (x : ℝ) → (Q x = 0) → (x = 4) → d = -33 :=
by
  intro x Q4 hx
  sorry

end factor_theorem_for_Q_l31_31261


namespace heart_then_club_probability_l31_31840

theorem heart_then_club_probability :
  (13 / 52) * (13 / 51) = 13 / 204 := by
  sorry

end heart_then_club_probability_l31_31840


namespace multiplication_subtraction_difference_l31_31556

theorem multiplication_subtraction_difference (x n : ℕ) (h₁ : x = 5) (h₂ : 3 * x = (16 - x) + n) : n = 4 :=
by
  -- Proof will go here
  sorry

end multiplication_subtraction_difference_l31_31556


namespace concentric_circles_radius_difference_l31_31744

theorem concentric_circles_radius_difference (r R : ℝ)
  (h : R^2 = 4 * r^2) :
  R - r = r :=
by
  sorry

end concentric_circles_radius_difference_l31_31744


namespace total_distance_l31_31913

/--
John's journey is from point (-3, 4) to (2, 2) to (6, -3).
Prove that the total distance John travels is the sum of distances
from (-3, 4) to (2, 2) and from (2, 2) to (6, -3).
-/
theorem total_distance : 
  let d1 := Real.sqrt ((-3 - 2)^2 + (4 - 2)^2)
  let d2 := Real.sqrt ((6 - 2)^2 + (-3 - 2)^2)
  d1 + d2 = Real.sqrt 29 + Real.sqrt 41 :=
by
  sorry

end total_distance_l31_31913


namespace spring_extension_l31_31706

theorem spring_extension (A1 A2 : ℝ) (x1 x2 : ℝ) (hA1 : A1 = 29.43) (hx1 : x1 = 0.05) (hA2 : A2 = 9.81) : x2 = 0.029 :=
by 
  sorry

end spring_extension_l31_31706


namespace relationship_among_y_values_l31_31169

theorem relationship_among_y_values (c y1 y2 y3 : ℝ) :
  (-1)^2 - 2 * (-1) + c = y1 →
  (3)^2 - 2 * 3 + c = y2 →
  (5)^2 - 2 * 5 + c = y3 →
  y1 = y2 ∧ y2 > y3 :=
by
  intros h1 h2 h3
  sorry

end relationship_among_y_values_l31_31169


namespace angle_C_is_80_l31_31114

-- Define the angles A, B, and C
def isoscelesTriangle (A B C : ℕ) : Prop :=
  -- Triangle ABC is isosceles with A = B, and C is 30 degrees more than A
  A = B ∧ C = A + 30 ∧ A + B + C = 180

-- Problem: Prove that angle C is 80 degrees given the conditions
theorem angle_C_is_80 (A B C : ℕ) (h : isoscelesTriangle A B C) : C = 80 :=
by sorry

end angle_C_is_80_l31_31114


namespace find_n_l31_31135

variable (n : ℚ)

theorem find_n (h : (2 / (n + 2) + 3 / (n + 2) + n / (n + 2) + 1 / (n + 2) = 4)) : 
  n = -2 / 3 :=
by
  sorry

end find_n_l31_31135


namespace problem_l31_31378

theorem problem (x : ℝ) (h : 8 * x = 3) : 200 * (1 / x) = 533.33 := by
  sorry

end problem_l31_31378


namespace binomial_12_10_l31_31763

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end binomial_12_10_l31_31763


namespace find_m_l31_31498

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {m : ℕ}

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def initial_condition (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def q_condition (q : ℝ) : Prop :=
  abs q ≠ 1

def a_m_condition (a : ℕ → ℝ) (m : ℕ) : Prop :=
  a m = a 1 * a 2 * a 3 * a 4 * a 5

-- Theorem to prove
theorem find_m (h1 : geometric_sequence a q) (h2 : initial_condition a) (h3 : q_condition q) (h4 : a_m_condition a m) : m = 11 :=
  sorry

end find_m_l31_31498


namespace percentage_of_adults_is_40_l31_31946

variables (A C : ℕ)

-- Given conditions as definitions
def total_members := 120
def more_children_than_adults := 24
def percentage_of_adults (A : ℕ) := (A.toFloat / total_members.toFloat) * 100

-- Lean 4 statement to prove the percentage of adults
theorem percentage_of_adults_is_40 (h1 : A + C = 120)
                                   (h2 : C = A + 24) :
  percentage_of_adults A = 40 :=
by
  sorry

end percentage_of_adults_is_40_l31_31946


namespace system_solution_and_range_l31_31770

theorem system_solution_and_range (a x y : ℝ) (h1 : 2 * x + y = 5 * a) (h2 : x - 3 * y = -a + 7) :
  (x = 2 * a + 1 ∧ y = a - 2) ∧ (-1/2 ≤ a ∧ a < 2 → 2 * a + 1 ≥ 0 ∧ a - 2 < 0) :=
by
  sorry

end system_solution_and_range_l31_31770


namespace three_digit_multiples_of_3_and_11_l31_31172

theorem three_digit_multiples_of_3_and_11 : 
  ∃ n, n = 27 ∧ ∀ x, 100 ≤ x ∧ x ≤ 999 ∧ x % 33 = 0 ↔ ∃ k, x = 33 * k ∧ 4 ≤ k ∧ k ≤ 30 :=
by
  sorry

end three_digit_multiples_of_3_and_11_l31_31172


namespace parabola_intersections_l31_31787

theorem parabola_intersections :
  ∃ y1 y2, (∀ x y, (y = 2 * x^2 + 5 * x + 1 ∧ y = - x^2 + 4 * x + 6) → 
     (x = ( -1 + Real.sqrt 61) / 6 ∧ y = y1) ∨ (x = ( -1 - Real.sqrt 61) / 6 ∧ y = y2)) := 
by
  sorry

end parabola_intersections_l31_31787


namespace stan_average_speed_l31_31754

/-- Given two trips with specified distances and times, prove that the overall average speed is 55 mph. -/
theorem stan_average_speed :
  let distance1 := 300
  let hours1 := 5
  let minutes1 := 20
  let distance2 := 360
  let hours2 := 6
  let minutes2 := 40
  let total_distance := distance1 + distance2
  let total_time := (hours1 + minutes1 / 60) + (hours2 + minutes2 / 60)
  total_distance / total_time = 55 := 
sorry

end stan_average_speed_l31_31754


namespace find_b_l31_31756

theorem find_b (b : ℤ) (h : ∃ x : ℝ, x^2 + b * x - 35 = 0 ∧ x = 5) : b = 2 :=
sorry

end find_b_l31_31756


namespace sin_300_eq_neg_sqrt_3_div_2_l31_31111

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l31_31111


namespace probability_not_within_square_b_l31_31865

noncomputable def prob_not_within_square_b : Prop :=
  let area_A := 121
  let side_length_B := 16 / 4
  let area_B := side_length_B * side_length_B
  let area_not_covered := area_A - area_B
  let prob := area_not_covered / area_A
  prob = (105 / 121)

theorem probability_not_within_square_b : prob_not_within_square_b :=
by
  sorry

end probability_not_within_square_b_l31_31865


namespace find_x_l31_31790

theorem find_x (x : ℕ) (h1 : 8 = 2 ^ 3) (h2 : 32 = 2 ^ 5) :
  (2^(x+2) * 8^(x-1) = 32^3) ↔ (x = 4) :=
by
  sorry

end find_x_l31_31790


namespace price_of_A_correct_l31_31810

noncomputable def A_price : ℝ := 25

theorem price_of_A_correct (H1 : 6000 / A_price - 4800 / (1.2 * A_price) = 80) 
                           (H2 : ∀ B_price : ℝ, B_price = 1.2 * A_price) : A_price = 25 := 
by
  sorry

end price_of_A_correct_l31_31810


namespace train_speed_is_64_kmh_l31_31920

noncomputable def train_speed_kmh (train_length platform_length time_seconds : ℕ) : ℕ :=
  let total_distance := train_length + platform_length
  let speed_mps := total_distance / time_seconds
  let speed_kmh := speed_mps * 36 / 10
  speed_kmh

theorem train_speed_is_64_kmh
  (train_length : ℕ)
  (platform_length : ℕ)
  (time_seconds : ℕ)
  (h_train_length : train_length = 240)
  (h_platform_length : platform_length = 240)
  (h_time_seconds : time_seconds = 27) :
  train_speed_kmh train_length platform_length time_seconds = 64 := by
  sorry

end train_speed_is_64_kmh_l31_31920


namespace parents_can_catch_ka_liang_l31_31417

-- Definitions according to the problem statement.
-- Define the condition of the roads and the speed of the participants.
def grid_with_roads : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  -- 4 roads forming the sides of a square with side length a
  True ∧
  -- 2 roads connecting the midpoints of opposite sides of the square
  True

def ka_liang_speed : ℝ := 2

def parent_speed : ℝ := 1

-- Condition that Ka Liang, father, and mother can see each other
def mutual_visibility (a b : ℝ) : Prop := True

-- The main proposition
theorem parents_can_catch_ka_liang (a b : ℝ) (hgrid : grid_with_roads)
    (hspeed : ka_liang_speed = 2 * parent_speed) (hvis : mutual_visibility a b) :
  True := 
sorry

end parents_can_catch_ka_liang_l31_31417


namespace necessary_but_not_sufficient_l31_31747

def quadratic_inequality (x : ℝ) : Prop :=
  x^2 - 3 * x + 2 < 0

def necessary_condition_A (x : ℝ) : Prop :=
  -1 < x ∧ x < 2

def necessary_condition_D (x : ℝ) : Prop :=
  -2 < x ∧ x < 2

theorem necessary_but_not_sufficient :
  (∀ x, quadratic_inequality x → ∃ x, necessary_condition_A x ∧ ¬(quadratic_inequality x ∧ necessary_condition_A x)) ∧ 
  (∀ x, quadratic_inequality x → ∃ x, necessary_condition_D x ∧ ¬(quadratic_inequality x ∧ necessary_condition_D x)) :=
sorry

end necessary_but_not_sufficient_l31_31747


namespace percentage_error_l31_31043

-- Define the conditions
def actual_side (a : ℝ) := a
def measured_side (a : ℝ) := 1.05 * a
def actual_area (a : ℝ) := a^2
def calculated_area (a : ℝ) := (1.05 * a)^2

-- Define the statement that we need to prove
theorem percentage_error (a : ℝ) (h : a > 0) :
  (calculated_area a - actual_area a) / actual_area a * 100 = 10.25 :=
by
  -- Proof goes here
  sorry

end percentage_error_l31_31043


namespace spend_on_rent_and_utilities_l31_31495

variable (P : ℝ) -- The percentage of her income she used to spend on rent and utilities
variable (I : ℝ) -- Her previous monthly income
variable (increase : ℝ) -- Her salary increase
variable (new_percentage : ℝ) -- The new percentage her rent and utilities amount to

noncomputable def initial_conditions : Prop :=
I = 1000 ∧ increase = 600 ∧ new_percentage = 0.25

theorem spend_on_rent_and_utilities (h : initial_conditions I increase new_percentage) :
    (P / 100) * I = 0.25 * (I + increase) → 
    P = 40 :=
by
  sorry

end spend_on_rent_and_utilities_l31_31495


namespace problem_travel_time_with_current_l31_31069

theorem problem_travel_time_with_current
  (D r c : ℝ) (t : ℝ)
  (h1 : (r - c) ≠ 0)
  (h2 : D / (r - c) = 60 / 7)
  (h3 : D / r = t - 7)
  (h4 : D / (r + c) = t)
  : t = 3 + 9 / 17 := 
sorry

end problem_travel_time_with_current_l31_31069


namespace product_of_solutions_of_quadratic_l31_31170

theorem product_of_solutions_of_quadratic :
  ∀ (x p q : ℝ), 36 - 9 * x - x^2 = 0 ∧ (x = p ∨ x = q) → p * q = -36 :=
by sorry

end product_of_solutions_of_quadratic_l31_31170


namespace lisa_investment_in_stocks_l31_31047

-- Definitions for the conditions
def total_investment (r : ℝ) : Prop := r + 7 * r = 200000
def stock_investment (r : ℝ) : ℝ := 7 * r

-- Given the conditions, we need to prove the amount invested in stocks
theorem lisa_investment_in_stocks (r : ℝ) (h : total_investment r) : stock_investment r = 175000 :=
by
  -- proof goes here
  sorry

end lisa_investment_in_stocks_l31_31047


namespace fraction_simplification_l31_31264

theorem fraction_simplification : (145^2 - 121^2) / 24 = 266 := by
  sorry

end fraction_simplification_l31_31264


namespace james_total_cost_is_100_l31_31622

def cost_of_shirts (number_of_shirts : Nat) (cost_per_shirt : Nat) : Nat :=
  number_of_shirts * cost_per_shirt

def cost_of_pants (number_of_pants : Nat) (cost_per_pants : Nat) : Nat :=
  number_of_pants * cost_per_pants

def total_cost (number_of_shirts : Nat) (number_of_pants : Nat) (cost_per_shirt : Nat) (cost_per_pants : Nat) : Nat :=
  cost_of_shirts number_of_shirts cost_per_shirt + cost_of_pants number_of_pants cost_per_pants

theorem james_total_cost_is_100 : 
  total_cost 10 (10 / 2) 6 8 = 100 :=
by
  sorry

end james_total_cost_is_100_l31_31622


namespace xyz_eq_7cubed_l31_31741

theorem xyz_eq_7cubed (x y z : ℤ) (h1 : x^2 * y * z^3 = 7^4) (h2 : x * y^2 = 7^5) : x * y * z = 7^3 := 
by 
  sorry

end xyz_eq_7cubed_l31_31741


namespace determine_a_l31_31966

def A := {x : ℝ | x < 6}
def B (a : ℝ) := {x : ℝ | x - a < 0}

theorem determine_a (a : ℝ) (h : A ⊆ B a) : 6 ≤ a := 
sorry

end determine_a_l31_31966


namespace find_fraction_value_l31_31416

theorem find_fraction_value {m n r t : ℚ}
  (h1 : m / n = 5 / 2)
  (h2 : r / t = 7 / 5) :
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -4 / 9 :=
by
  sorry

end find_fraction_value_l31_31416


namespace find_original_speed_l31_31520

theorem find_original_speed (r : ℝ) (t : ℝ)
  (h_circumference : r * t = 15 / 5280)
  (h_increase : (r + 8) * (t - 1/10800) = 15 / 5280) :
  r = 7.5 :=
sorry

end find_original_speed_l31_31520


namespace find_n_l31_31534

-- Definitions of the problem conditions
def sum_coefficients (n : ℕ) : ℕ := 4^n
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

-- The main theorem to be proved
theorem find_n (n : ℕ) (P S : ℕ) (hP : P = sum_coefficients n) (hS : S = sum_binomial_coefficients n) (h : P + S = 272) : n = 4 :=
by
  sorry

end find_n_l31_31534


namespace find_k_l31_31102

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

theorem find_k (k : ℝ) (h : dot_product (k * a.1, k * a.2 + b.2) (3 * a.1, 3 * a.2 - b.2) = 0) :
  k = 1 / 3 :=
by
  sorry

end find_k_l31_31102


namespace largest_side_of_enclosure_l31_31835

theorem largest_side_of_enclosure (l w : ℕ) (h1 : 2 * l + 2 * w = 180) (h2 : l * w = 1800) : max l w = 60 := 
by 
  sorry

end largest_side_of_enclosure_l31_31835


namespace find_sides_of_rectangle_l31_31718

-- Define the conditions
def isRectangle (w l : ℝ) : Prop :=
  l = 3 * w ∧ 2 * l + 2 * w = l * w

-- Main theorem statement
theorem find_sides_of_rectangle (w l : ℝ) :
  isRectangle w l → w = 8 / 3 ∧ l = 8 :=
by
  sorry

end find_sides_of_rectangle_l31_31718


namespace inequality_l31_31678

theorem inequality (a b c d e p q : ℝ) 
  (h0 : 0 < p ∧ p ≤ a ∧ p ≤ b ∧ p ≤ c ∧ p ≤ d ∧ p ≤ e)
  (h1 : a ≤ q ∧ b ≤ q ∧ c ≤ q ∧ d ≤ q ∧ e ≤ q) :
  (a + b + c + d + e) * ((1 / a) + (1 / b) + (1 / c) + (1 / d) + (1 / e)) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 :=
by
  sorry

end inequality_l31_31678


namespace greatest_integer_value_of_x_l31_31277

theorem greatest_integer_value_of_x :
  ∃ x : ℤ, (3 * |2 * x + 1| + 10 > 28) ∧ (∀ y : ℤ, 3 * |2 * y + 1| + 10 > 28 → y ≤ x) :=
sorry

end greatest_integer_value_of_x_l31_31277


namespace min_value_of_expression_l31_31436

theorem min_value_of_expression (x y : ℝ) (h : 2 * x - y = 4) : ∃ z : ℝ, (z = 4^x + (1/2)^y) ∧ z = 8 :=
by 
  sorry

end min_value_of_expression_l31_31436


namespace find_f_neg_one_l31_31481

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, f (x^2 + y) = f x + f (y^2)

theorem find_f_neg_one : f (-1) = 0 := sorry

end find_f_neg_one_l31_31481


namespace cone_lateral_surface_area_l31_31628

theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 3) (hh : h = 4) : 15 * Real.pi = Real.pi * r * (Real.sqrt (r^2 + h^2)) :=
by
  -- Prove that 15π = π * r * sqrt(r^2 + h^2) for r = 3 and h = 4
  sorry

end cone_lateral_surface_area_l31_31628


namespace num_two_digit_numbers_l31_31256

-- Define the set of given digits
def digits : Finset ℕ := {0, 2, 5}

-- Define the function that counts the number of valid two-digit numbers
def count_two_digit_numbers (d : Finset ℕ) : ℕ :=
  (d.erase 0).card * (d.card - 1)

theorem num_two_digit_numbers : count_two_digit_numbers digits = 4 :=
by {
  -- sorry placeholder for the proof
  sorry
}

end num_two_digit_numbers_l31_31256


namespace inequality_x4_y4_z2_l31_31685

theorem inequality_x4_y4_z2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^4 + y^4 + z^2 ≥  xyz * 8^(1/2) :=
  sorry

end inequality_x4_y4_z2_l31_31685


namespace x1_sufficient_not_necessary_l31_31615

theorem x1_sufficient_not_necessary : (x : ℝ) → (x = 1 ↔ (x - 1) * (x + 2) = 0) ∧ ∀ x, (x = 1 ∨ x = -2) → (x - 1) * (x + 2) = 0 ∧ (∀ y, (y - 1) * (y + 2) = 0 → (y = 1 ∨ y = -2)) :=
by
  sorry

end x1_sufficient_not_necessary_l31_31615


namespace opposite_of_neg_eight_l31_31965

theorem opposite_of_neg_eight : (-(-8)) = 8 :=
by
  sorry

end opposite_of_neg_eight_l31_31965


namespace binary_mul_correct_l31_31453

def binary_mul (a b : ℕ) : ℕ :=
  a * b

def binary_to_nat (s : String) : ℕ :=
  String.foldr (λ c acc => if c = '1' then acc * 2 + 1 else acc * 2) 0 s

def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0"
  else let rec aux (n : ℕ) (acc : String) :=
         if n = 0 then acc
         else aux (n / 2) (if n % 2 = 0 then "0" ++ acc else "1" ++ acc)
       aux n ""

theorem binary_mul_correct :
  nat_to_binary (binary_mul (binary_to_nat "1101") (binary_to_nat "111")) = "10001111" :=
by
  sorry

end binary_mul_correct_l31_31453


namespace faster_runner_l31_31693

-- Define the speeds of A and B
variables (v_A v_B : ℝ)
-- A's speed as a multiple of B's speed
variables (k : ℝ)

-- A's and B's distances in the race
variables (d_A d_B : ℝ)
-- Distance of the race
variables (distance : ℝ)
-- Head start given to B
variables (head_start : ℝ)

-- The theorem to prove that the factor k is 4 given the conditions
theorem faster_runner (k : ℝ) (v_A v_B : ℝ) (d_A d_B distance head_start : ℝ) :
  v_A = k * v_B ∧ d_B = distance - head_start ∧ d_A = distance ∧ (d_A / v_A) = (d_B / v_B) → k = 4 :=
by
  sorry

end faster_runner_l31_31693


namespace totalTilesUsed_l31_31228

-- Define the dining room dimensions
def diningRoomLength : ℕ := 18
def diningRoomWidth : ℕ := 15

-- Define the border width
def borderWidth : ℕ := 2

-- Define tile dimensions
def tile1x1 : ℕ := 1
def tile2x2 : ℕ := 2

-- Calculate the number of tiles used along the length and width for the border
def borderTileCountLength : ℕ := 2 * 2 * (diningRoomLength - 2 * borderWidth)
def borderTileCountWidth : ℕ := 2 * 2 * (diningRoomWidth - 2 * borderWidth)

-- Total number of one-foot by one-foot tiles for the border
def totalBorderTileCount : ℕ := borderTileCountLength + borderTileCountWidth

-- Calculate the inner area dimensions
def innerLength : ℕ := diningRoomLength - 2 * borderWidth
def innerWidth : ℕ := diningRoomWidth - 2 * borderWidth
def innerArea : ℕ := innerLength * innerWidth

-- Number of two-foot by two-foot tiles needed
def tile2x2Count : ℕ := (innerArea + tile2x2 * tile2x2 - 1) / (tile2x2 * tile2x2) -- Ensures rounding up without floating point arithmetic

-- Prove that the total number of tiles used is 139
theorem totalTilesUsed : totalBorderTileCount + tile2x2Count = 139 := by
  sorry

end totalTilesUsed_l31_31228


namespace product_of_base8_digits_of_5432_l31_31210

open Nat

def base8_digits (n : ℕ) : List ℕ :=
  let rec digits_helper (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc
    else digits_helper (n / 8) ((n % 8) :: acc)
  digits_helper n []

def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· * ·) 1

theorem product_of_base8_digits_of_5432 : 
    product_of_digits (base8_digits 5432) = 0 :=
by
  sorry

end product_of_base8_digits_of_5432_l31_31210


namespace point_C_correct_l31_31354

-- Definitions of point A and B
def A : ℝ × ℝ := (4, -4)
def B : ℝ × ℝ := (18, 6)

-- Coordinate of C obtained from the conditions of the problem
def C : ℝ × ℝ := (25, 11)

-- Proof statement
theorem point_C_correct :
  ∃ C : ℝ × ℝ, (∃ (BC : ℝ × ℝ), BC = (1/2) • (B.1 - A.1, B.2 - A.2) ∧ C = (B.1 + BC.1, B.2 + BC.2)) ∧ C = (25, 11) :=
by
  sorry

end point_C_correct_l31_31354


namespace cell_phone_total_cost_l31_31430

def base_cost : ℕ := 25
def text_cost_per_message : ℕ := 3
def extra_minute_cost_per_minute : ℕ := 15
def included_hours : ℕ := 40
def messages_sent_in_february : ℕ := 200
def hours_talked_in_february : ℕ := 41

theorem cell_phone_total_cost :
  base_cost + (messages_sent_in_february * text_cost_per_message) / 100 + 
  ((hours_talked_in_february - included_hours) * 60 * extra_minute_cost_per_minute) / 100 = 40 :=
by
  sorry

end cell_phone_total_cost_l31_31430


namespace sum_of_remainders_l31_31265

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) :
  (n % 2 + n % 9) = 3 :=
sorry

end sum_of_remainders_l31_31265


namespace total_pages_in_book_l31_31901

theorem total_pages_in_book (P : ℕ)
  (first_day : P - (P / 5) - 12 = remaining_1)
  (second_day : remaining_1 - (remaining_1 / 4) - 15 = remaining_2)
  (third_day : remaining_2 - (remaining_2 / 3) - 18 = 42) :
  P = 190 := 
sorry

end total_pages_in_book_l31_31901


namespace gcd_lcm_product_l31_31071

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 30) (h2 : b = 45) :
  Nat.gcd a b * Nat.lcm a b = 1350 :=
by
  rw [h1, h2]
  sorry

end gcd_lcm_product_l31_31071


namespace remainder_divisor_l31_31900

theorem remainder_divisor (d r : ℤ) (h1 : d > 1) 
  (h2 : 2024 % d = r) (h3 : 3250 % d = r) (h4 : 4330 % d = r) : d - r = 2 := 
by
  sorry

end remainder_divisor_l31_31900


namespace HCl_yield_l31_31441

noncomputable def total_moles_HCl (moles_C2H6 moles_Cl2 yield1 yield2 : ℝ) : ℝ :=
  let theoretical_yield1 := if moles_C2H6 ≤ moles_Cl2 then moles_C2H6 else moles_Cl2
  let actual_yield1 := theoretical_yield1 * yield1
  let theoretical_yield2 := actual_yield1
  let actual_yield2 := theoretical_yield2 * yield2
  actual_yield1 + actual_yield2

theorem HCl_yield (moles_C2H6 moles_Cl2 : ℝ) (yield1 yield2 : ℝ) :
  moles_C2H6 = 3 → moles_Cl2 = 3 → yield1 = 0.85 → yield2 = 0.70 →
  total_moles_HCl moles_C2H6 moles_Cl2 yield1 yield2 = 4.335 :=
by
  intros h1 h2 h3 h4
  simp [total_moles_HCl, h1, h2, h3, h4]
  sorry

end HCl_yield_l31_31441


namespace traceable_edges_l31_31449

-- Define the vertices of the rectangle
def vertex (x y : ℕ) : ℕ × ℕ := (x, y)

-- Define the edges of the rectangle
def edges : List (ℕ × ℕ) :=
  [vertex 0 0, vertex 0 1,    -- vertical edges
   vertex 1 0, vertex 1 1,
   vertex 2 0, vertex 2 1,
   vertex 0 0, vertex 1 0,    -- horizontal edges
   vertex 1 0, vertex 2 0,
   vertex 0 1, vertex 1 1,
   vertex 1 1, vertex 2 1]

-- Define the theorem to be proved
theorem traceable_edges :
  ∃ (count : ℕ), count = 61 :=
by
  sorry

end traceable_edges_l31_31449


namespace factorization_mn_l31_31051

variable (m n : ℝ) -- Declare m and n as arbitrary real numbers.

theorem factorization_mn (m n : ℝ) : m^2 - m * n = m * (m - n) := by
  sorry

end factorization_mn_l31_31051


namespace first_meet_at_starting_point_l31_31115

-- Definitions
def track_length := 300
def speed_A := 2
def speed_B := 4

-- Theorem: A and B will meet at the starting point for the first time after 400 seconds.
theorem first_meet_at_starting_point : 
  (∃ (t : ℕ), t = 400 ∧ (
    (∃ (n : ℕ), n * (track_length * (speed_B - speed_A)) = t * (speed_A + speed_B) * track_length) ∨
    (∃ (m : ℕ), m * (track_length * (speed_B + speed_A)) = t * (speed_A - speed_B) * track_length))) := 
    sorry

end first_meet_at_starting_point_l31_31115


namespace range_of_a_l31_31397

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_l31_31397


namespace grandson_age_l31_31833

variable (G F : ℕ)

-- Define the conditions given in the problem
def condition1 := F = 6 * G
def condition2 := (F + 4) + (G + 4) = 78

-- The theorem to prove
theorem grandson_age : condition1 G F → condition2 G F → G = 10 :=
by
  intros h1 h2
  sorry

end grandson_age_l31_31833


namespace f_inequality_l31_31346

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) : ∀ x : ℝ, f x a > 2 * Real.log a + 3 / 2 :=
sorry

end f_inequality_l31_31346


namespace line_equation_l31_31128

theorem line_equation (a T : ℝ) (h : 0 < a ∧ 0 < T) :
  ∃ (x y : ℝ), (2 * T * x - a^2 * y + 2 * a * T = 0) :=
by
  sorry

end line_equation_l31_31128


namespace rectangular_box_diagonals_l31_31038

noncomputable def interior_diagonals_sum (a b c : ℝ) : ℝ := 4 * Real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + c * a) = 150)
  (h2 : 4 * (a + b + c) = 60)
  (h3 : a * b * c = 216) :
  interior_diagonals_sum a b c = 20 * Real.sqrt 3 :=
by
  sorry

end rectangular_box_diagonals_l31_31038


namespace max_trading_cards_l31_31072

variable (money : ℝ) (cost_per_card : ℝ) (max_cards : ℕ)

theorem max_trading_cards (h_money : money = 9) (h_cost : cost_per_card = 1) : max_cards ≤ 9 :=
sorry

end max_trading_cards_l31_31072


namespace initial_amount_l31_31061

variable (X : ℝ)

/--
An individual deposited 20% of 25% of 30% of their initial amount into their bank account.
If the deposited amount is Rs. 750, prove that their initial amount was Rs. 50000.
-/
theorem initial_amount (h : (0.2 * 0.25 * 0.3 * X) = 750) : X = 50000 :=
by
  sorry

end initial_amount_l31_31061


namespace celebrity_baby_photo_probability_l31_31955

theorem celebrity_baby_photo_probability : 
  let total_arrangements := Nat.factorial 4
  let correct_arrangements := 1
  let probability := correct_arrangements / total_arrangements
  probability = 1/24 :=
by
  sorry

end celebrity_baby_photo_probability_l31_31955


namespace distance_eq_3_implies_points_l31_31576

-- Definition of the distance of point A to the origin
def distance_to_origin (x : ℝ) : ℝ := |x|

-- Theorem statement translating the problem
theorem distance_eq_3_implies_points (x : ℝ) (h : distance_to_origin x = 3) :
  x = 3 ∨ x = -3 :=
sorry

end distance_eq_3_implies_points_l31_31576


namespace fred_current_dimes_l31_31062

-- Definitions based on the conditions
def original_dimes : ℕ := 7
def borrowed_dimes : ℕ := 3

-- The theorem to prove
theorem fred_current_dimes : original_dimes - borrowed_dimes = 4 := by
  sorry

end fred_current_dimes_l31_31062


namespace rate_of_grapes_l31_31825

theorem rate_of_grapes (G : ℝ) (H : 8 * G + 9 * 50 = 1010) : G = 70 := by
  sorry

end rate_of_grapes_l31_31825


namespace can_capacity_l31_31455

-- Definitions of the conditions
variable (M W : ℕ) -- initial amounts of milk and water
variable (M' : ℕ := M + 2) -- new amount of milk after adding 2 liters
variable (ratio_initial : M / W = 1 / 5)
variable (ratio_new : M' / W = 3 / 5)

theorem can_capacity (M W : ℕ) (h_ratio_initial : M / W = 1 / 5) (h_ratio_new : (M + 2) / W = 3 / 5) : (M + W + 2) = 8 := 
by
  sorry

end can_capacity_l31_31455


namespace bags_of_chips_count_l31_31812

theorem bags_of_chips_count :
  ∃ n : ℕ, n * 400 + 4 * 50 = 2200 ∧ n = 5 :=
by {
  sorry
}

end bags_of_chips_count_l31_31812


namespace chess_tournament_participants_l31_31742

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 136) : n = 17 :=
by {
  sorry -- Proof will be here.
}

end chess_tournament_participants_l31_31742


namespace sum_of_coordinates_x_l31_31557

-- Given points Y and Z
def Y : ℝ × ℝ := (2, 8)
def Z : ℝ × ℝ := (0, -4)

-- Given ratio conditions
def ratio_condition (X Y Z : ℝ × ℝ) : Prop :=
  dist X Z / dist X Y = 1/3 ∧ dist Z Y / dist X Y = 1/3

-- Define X, ensuring Z is the midpoint of XY
def X : ℝ × ℝ := (4, 20)

-- Prove that sum of coordinates of X is 10
theorem sum_of_coordinates_x (h : ratio_condition X Y Z) : (X.1 + X.2) = 10 := 
  sorry

end sum_of_coordinates_x_l31_31557


namespace dirocks_rectangular_fence_count_l31_31801

/-- Dirock's backyard problem -/
def grid_side : ℕ := 32

def rock_placement (i j : ℕ) : Prop := (i % 3 = 0) ∧ (j % 3 = 0)

noncomputable def dirocks_rectangular_fence_ways : ℕ :=
  sorry

theorem dirocks_rectangular_fence_count : dirocks_rectangular_fence_ways = 1920 :=
sorry

end dirocks_rectangular_fence_count_l31_31801


namespace train_length_l31_31922

noncomputable def L_train : ℝ :=
  let speed_kmph : ℝ := 60
  let speed_mps : ℝ := (speed_kmph * 1000 / 3600)
  let time : ℝ := 30
  let length_bridge : ℝ := 140
  let total_distance : ℝ := speed_mps * time
  total_distance - length_bridge

theorem train_length : L_train = 360.1 :=
by
  -- Sorry statement to skip the proof
  sorry

end train_length_l31_31922


namespace probability_sum_9_is_correct_l31_31726

def num_faces : ℕ := 6

def possible_outcomes : ℕ := num_faces * num_faces

def favorable_outcomes : ℕ := 4  -- (3,6), (6,3), (4,5), (5,4)

def probability_sum_9 : ℚ := favorable_outcomes / possible_outcomes

theorem probability_sum_9_is_correct :
  probability_sum_9 = 1/9 :=
sorry

end probability_sum_9_is_correct_l31_31726


namespace max_pieces_four_cuts_l31_31023

theorem max_pieces_four_cuts (n : ℕ) (h : n = 4) : (by sorry : ℕ) = 14 := 
by sorry

end max_pieces_four_cuts_l31_31023


namespace min_value_two_x_plus_y_l31_31649

theorem min_value_two_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y + 2 * x * y = 5 / 4) : 2 * x + y ≥ 1 :=
by
  sorry

end min_value_two_x_plus_y_l31_31649


namespace wire_length_after_two_bends_is_three_l31_31732

-- Let's define the initial length and the property of bending the wire.
def initial_length : ℕ := 12

def half_length (length : ℕ) : ℕ :=
  length / 2

-- Define the final length after two bends.
def final_length_after_two_bends : ℕ :=
  half_length (half_length initial_length)

-- The theorem stating that the final length is 3 cm after two bends.
theorem wire_length_after_two_bends_is_three :
  final_length_after_two_bends = 3 :=
by
  -- The proof can be added later.
  sorry

end wire_length_after_two_bends_is_three_l31_31732


namespace num_distinct_intersections_l31_31800

def linear_eq1 (x y : ℝ) := x + 2 * y - 10
def linear_eq2 (x y : ℝ) := x - 4 * y + 8
def linear_eq3 (x y : ℝ) := 2 * x - y - 1
def linear_eq4 (x y : ℝ) := 5 * x + 3 * y - 15

theorem num_distinct_intersections (n : ℕ) :
  (∀ x y : ℝ, linear_eq1 x y = 0 ∨ linear_eq2 x y = 0) ∧ 
  (∀ x y : ℝ, linear_eq3 x y = 0 ∨ linear_eq4 x y = 0) →
  n = 3 :=
  sorry

end num_distinct_intersections_l31_31800


namespace remainder_when_13_plus_x_divided_by_26_l31_31262

theorem remainder_when_13_plus_x_divided_by_26 (x : ℕ) (h1 : 9 * x % 26 = 1) : (13 + x) % 26 = 16 := 
by sorry

end remainder_when_13_plus_x_divided_by_26_l31_31262


namespace domain_of_f_equals_l31_31190

noncomputable def domain_of_function := {x : ℝ | x > -1 ∧ -(x+4) * (x-1) > 0}

theorem domain_of_f_equals : domain_of_function = { x : ℝ | -1 < x ∧ x < 1 } :=
by
  sorry

end domain_of_f_equals_l31_31190


namespace initial_men_count_l31_31543

theorem initial_men_count (M : ℕ) (h1 : ∃ F : ℕ, F = M * 22) (h2 : ∃ F_remaining : ℕ, F_remaining = M * 20) (h3 : ∃ F_remaining_2 : ℕ, F_remaining_2 = (M + 1140) * 8) : 
  M = 760 := 
by
  -- Code to prove the theorem goes here.
  sorry

end initial_men_count_l31_31543


namespace solve_inequality_when_a_lt_2_find_a_range_when_x_in_2_3_l31_31620

variable (a : ℝ) (x : ℝ)

def inequality (a x : ℝ) : Prop :=
  a * x^2 - (a + 2) * x + 2 < 0

theorem solve_inequality_when_a_lt_2 (h : a < 2) :
  (a = 0 → ∀ x, x > 1 → inequality a x) ∧
  (a < 0 → ∀ x, x < 2 / a ∨ x > 1 → inequality a x) ∧
  (0 < a ∧ a < 2 → ∀ x, 1 < x ∧ x < 2 / a → inequality a x) := 
sorry

theorem find_a_range_when_x_in_2_3 :
  (∀ x, 2 ≤ x ∧ x ≤ 3 → inequality a x) → a < 2 / 3 :=
sorry

end solve_inequality_when_a_lt_2_find_a_range_when_x_in_2_3_l31_31620


namespace Chang_solution_A_amount_l31_31807

def solution_alcohol_content (A B : ℝ) (x : ℝ) : ℝ :=
  0.16 * x + 0.10 * (x + 500)

theorem Chang_solution_A_amount (x : ℝ) :
  solution_alcohol_content 0.16 0.10 x = 76 → x = 100 :=
by
  intro h
  sorry

end Chang_solution_A_amount_l31_31807


namespace largest_multiple_of_18_with_digits_9_or_0_l31_31230

theorem largest_multiple_of_18_with_digits_9_or_0 :
  ∃ (n : ℕ), (n = 9990) ∧ (n % 18 = 0) ∧ (∀ d ∈ (n.digits 10), d = 9 ∨ d = 0) ∧ (n / 18 = 555) :=
by
  sorry

end largest_multiple_of_18_with_digits_9_or_0_l31_31230


namespace find_k_l31_31769

-- Define the function f as described in the problem statement
def f (n : ℕ) : ℕ := 
  if n % 2 = 1 then 
    n + 3 
  else 
    n / 2

theorem find_k (k : ℕ) (h_odd : k % 2 = 1) : f (f (f k)) = k → k = 1 :=
by {
  sorry
}

end find_k_l31_31769


namespace dinosaur_count_l31_31817

theorem dinosaur_count (h : ℕ) (l : ℕ) (H1 : h = 1) (H2 : l = 3) (total_hl : ℕ) (H3 : total_hl = 20) :
  ∃ D : ℕ, 4 * D = total_hl := 
by
  use 5
  sorry

end dinosaur_count_l31_31817


namespace cuboid_height_l31_31722

-- Define the base area and volume of the cuboid
def base_area : ℝ := 50
def volume : ℝ := 2000

-- Prove that the height is 40 cm given the base area and volume
theorem cuboid_height : volume / base_area = 40 := by
  sorry

end cuboid_height_l31_31722


namespace man_wage_l31_31422

variable (m w b : ℝ) -- wages of man, woman, boy respectively
variable (W : ℝ) -- number of women equivalent to 5 men and 8 boys

-- Conditions given in the problem
axiom condition1 : 5 * m = W * w
axiom condition2 : W * w = 8 * b
axiom condition3 : 5 * m + 8 * b + 8 * b = 90

-- Prove the wage of one man
theorem man_wage : m = 6 := 
by
  -- proof steps would be here, but skipped as per instructions
  sorry

end man_wage_l31_31422


namespace problem_a_l31_31040

theorem problem_a (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  Int.floor (5 * x) + Int.floor (5 * y) ≥ Int.floor (3 * x + y) + Int.floor (3 * y + x) :=
sorry

end problem_a_l31_31040


namespace range_of_a_l31_31479

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def increasing_on_negative (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → y ≤ 0 → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (ha : even_function f) (hb : increasing_on_negative f) 
  (hc : ∀ a : ℝ, f a ≤ f (2 - a)) : ∀ a : ℝ, a < 1 → false :=
by
  sorry

end range_of_a_l31_31479


namespace proof_problem_l31_31549

variable {R : Type} [OrderedRing R]

-- Definitions and conditions
variable (g : R → R) (f : R → R) (k a m : R)
variable (h_odd : ∀ x : R, g (-x) = -g x)
variable (h_f_def : ∀ x : R, f x = g x + k)
variable (h_f_neg_a : f (-a) = m)

-- Theorem statement
theorem proof_problem : f a = 2 * k - m :=
by
  -- Here is where the proof would go.
  sorry

end proof_problem_l31_31549


namespace problem_statement_l31_31748

theorem problem_statement :
  ((8^5 / 8^2) * 2^10 - 2^2) = 2^19 - 4 := 
by 
  sorry

end problem_statement_l31_31748


namespace family_members_l31_31280

theorem family_members (N : ℕ) (income : ℕ → ℕ) (average_income : ℕ) :
  average_income = 10000 ∧
  income 0 = 8000 ∧
  income 1 = 15000 ∧
  income 2 = 6000 ∧
  income 3 = 11000 ∧
  (income 0 + income 1 + income 2 + income 3) = 4 * average_income →
  N = 4 :=
by {
  sorry
}

end family_members_l31_31280


namespace sum_of_bases_is_20_l31_31750

theorem sum_of_bases_is_20
  (B1 B2 : ℕ)
  (G1 : ℚ)
  (G2 : ℚ)
  (hG1_B1 : G1 = (4 * B1 + 5) / (B1^2 - 1))
  (hG2_B1 : G2 = (5 * B1 + 4) / (B1^2 - 1))
  (hG1_B2 : G1 = (3 * B2) / (B2^2 - 1))
  (hG2_B2 : G2 = (6 * B2) / (B2^2 - 1)) :
  B1 + B2 = 20 :=
sorry

end sum_of_bases_is_20_l31_31750


namespace simplify_and_evaluate_l31_31421

-- Define the constants
def a : ℤ := -1
def b : ℤ := 2

-- Declare the expression
def expr : ℤ := 7 * a ^ 2 * b + (-4 * a ^ 2 * b + 5 * a * b ^ 2) - (2 * a ^ 2 * b - 3 * a * b ^ 2)

-- Declare the final evaluated result
def result : ℤ := 2 * ((-1 : ℤ) ^ 2) + 8 * (-1) * (2 : ℤ) ^ 2 

-- The theorem we want to prove
theorem simplify_and_evaluate : expr = result :=
by
  sorry

end simplify_and_evaluate_l31_31421


namespace possible_values_of_a_plus_b_l31_31642

variable (a b : ℤ)

theorem possible_values_of_a_plus_b (h1 : |a| = 2) (h2 : |b| = a) :
  (a + b = 0 ∨ a + b = 4 ∨ a + b = -4) :=
sorry

end possible_values_of_a_plus_b_l31_31642


namespace fg_of_2_eq_81_l31_31140

def f (x : ℝ) : ℝ := x ^ 2
def g (x : ℝ) : ℝ := x ^ 2 + 2 * x + 1

theorem fg_of_2_eq_81 : f (g 2) = 81 := by
  sorry

end fg_of_2_eq_81_l31_31140


namespace divisor_of_425904_l31_31473

theorem divisor_of_425904 :
  ∃ (d : ℕ), d = 7 ∧ ∃ (n : ℕ), n = 425897 + 7 ∧ 425904 % d = 0 :=
by
  sorry

end divisor_of_425904_l31_31473


namespace max_true_statements_l31_31247

theorem max_true_statements 
  (a b : ℝ) 
  (cond1 : a > 0) 
  (cond2 : b > 0) : 
  ( 
    ( (1 / a > 1 / b) ∧ (a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( (1 / a > 1 / b) ∧ ¬(a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( ¬(1 / a > 1 / b) ∧ (a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( ¬(1 / a > 1 / b) ∧ ¬(a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
  ) 
→ 
  (true ∧ true ∧ true ∧ true → 4 = 4) :=
sorry

end max_true_statements_l31_31247


namespace units_digit_sum_base8_l31_31690

theorem units_digit_sum_base8 : 
  ∀ (x y : ℕ), (x = 64 ∧ y = 34 ∧ (x % 8 = 4) ∧ (y % 8 = 4) → (x + y) % 8 = 0) :=
by
  sorry

end units_digit_sum_base8_l31_31690


namespace john_marble_choices_l31_31075

open Nat

theorem john_marble_choices :
  (choose 4 2) * (choose 12 3) = 1320 :=
by
  sorry

end john_marble_choices_l31_31075


namespace other_root_is_minus_two_l31_31535

theorem other_root_is_minus_two (b : ℝ) (h : 1^2 + b * 1 - 2 = 0) : 
  ∃ (x : ℝ), x = -2 ∧ x^2 + b * x - 2 = 0 :=
by
  sorry

end other_root_is_minus_two_l31_31535


namespace intersection_point_and_distance_l31_31971

/-- Define the points A, B, C, D, and M based on the specified conditions. --/
def A := (0, 3)
def B := (6, 3)
def C := (6, 0)
def D := (0, 0)
def M := (3, 0)

/-- Define the equations of the circles. --/
def circle1 (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 2.25
def circle2 (x y : ℝ) : Prop := x ^ 2 + (y - 3) ^ 2 = 25

/-- The point P that is one of the intersection points of the two circles. --/
def P := (2, 1.5)

/-- Define the line AD as the y-axis. --/
def AD := 0

/-- Calculate the distance from point P to the y-axis (AD). --/
def distance_to_ad (x : ℝ) := |x|

theorem intersection_point_and_distance :
  circle1 (2 : ℝ) (1.5 : ℝ) ∧ circle2 (2 : ℝ) (1.5 : ℝ) ∧ distance_to_ad 2 = 2 :=
by
  unfold circle1 circle2 distance_to_ad
  norm_num
  sorry

end intersection_point_and_distance_l31_31971


namespace monomials_like_terms_l31_31133

theorem monomials_like_terms (a b : ℤ) (h1 : a + 1 = 2) (h2 : b - 2 = 3) : a + b = 6 :=
sorry

end monomials_like_terms_l31_31133


namespace oil_ratio_l31_31312

theorem oil_ratio (x : ℝ) (initial_small_tank : ℝ) (initial_large_tank : ℝ) (total_capacity_large : ℝ)
  (half_capacity_large : ℝ) (additional_needed : ℝ) :
  initial_small_tank = 4000 ∧ initial_large_tank = 3000 ∧ total_capacity_large = 20000 ∧
  half_capacity_large = total_capacity_large / 2 ∧ additional_needed = 4000 ∧
  (initial_large_tank + x + additional_needed = half_capacity_large) →
  x / initial_small_tank = 3 / 4 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  sorry

end oil_ratio_l31_31312


namespace quadratic_root_value_l31_31598
-- Import the entirety of the necessary library

-- Define the quadratic equation with one root being -1
theorem quadratic_root_value 
    (m : ℝ)
    (h1 : ∀ x : ℝ, x^2 + m * x + 3 = 0)
    (root1 : -1 ∈ {x : ℝ | x^2 + m * x + 3 = 0}) :
    m = 4 ∧ ∃ root2 : ℝ, root2 = -3 ∧ root2 ∈ {x : ℝ | x^2 + m * x + 3 = 0} :=
by
  sorry

end quadratic_root_value_l31_31598


namespace calculate_salary_l31_31657

-- Define the constants and variables
def food_percentage : ℝ := 0.35
def rent_percentage : ℝ := 0.25
def clothes_percentage : ℝ := 0.20
def transportation_percentage : ℝ := 0.10
def recreational_percentage : ℝ := 0.15
def emergency_fund : ℝ := 3000
def total_percentage : ℝ := food_percentage + rent_percentage + clothes_percentage + transportation_percentage + recreational_percentage

-- Define the salary
def salary (S : ℝ) : Prop :=
  (total_percentage - 1) * S = emergency_fund

-- The theorem stating the salary is 60000
theorem calculate_salary : ∃ S : ℝ, salary S ∧ S = 60000 :=
by
  use 60000
  unfold salary total_percentage
  sorry

end calculate_salary_l31_31657


namespace express_f12_in_terms_of_a_l31_31250

variable {f : ℝ → ℝ}
variable {a : ℝ}
variable (f_add : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (f_neg_three : f (-3) = a)

theorem express_f12_in_terms_of_a : f 12 = -4 * a := sorry

end express_f12_in_terms_of_a_l31_31250


namespace mike_taller_than_mark_l31_31676

def feet_to_inches (feet : ℕ) : ℕ := 12 * feet

def mark_height_feet := 5
def mark_height_inches := 3
def mike_height_feet := 6
def mike_height_inches := 1

def mark_total_height := feet_to_inches mark_height_feet + mark_height_inches
def mike_total_height := feet_to_inches mike_height_feet + mike_height_inches

theorem mike_taller_than_mark : mike_total_height - mark_total_height = 10 :=
by
  sorry

end mike_taller_than_mark_l31_31676


namespace first_set_cost_l31_31448

theorem first_set_cost (F S : ℕ) (hS : S = 50) (h_equation : 2 * F + 3 * S = 220) 
: 3 * F + S = 155 := 
sorry

end first_set_cost_l31_31448


namespace trapezoid_median_l31_31829

theorem trapezoid_median {BC AD : ℝ} (h AC CD : ℝ) (h_nonneg : h = 2) (AC_eq_CD : AC = 4) (BC_eq_0 : BC = 0) 
: (AD = 4 * Real.sqrt 3) → (median = 3 * Real.sqrt 3) := by
  sorry

end trapezoid_median_l31_31829


namespace unique_solution_single_element_l31_31117

theorem unique_solution_single_element (a : ℝ) 
  (h : ∀ x y : ℝ, (a * x^2 + a * x + 1 = 0) → (a * y^2 + a * y + 1 = 0) → x = y) : a = 4 := 
by
  sorry

end unique_solution_single_element_l31_31117


namespace base_conversion_subtraction_l31_31890

/-- Definition of base conversion from base 7 and base 5 to base 10. -/
def convert_base_7_to_10 (n : Nat) : Nat :=
  match n with
  | 52103 => 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 0 * 7^1 + 3 * 7^0
  | _ => 0

def convert_base_5_to_10 (n : Nat) : Nat :=
  match n with
  | 43120 => 4 * 5^4 + 3 * 5^3 + 1 * 5^2 + 2 * 5^1 + 0 * 5^0
  | _ => 0

theorem base_conversion_subtraction : 
  convert_base_7_to_10 52103 - convert_base_5_to_10 43120 = 9833 :=
by
  -- The proof goes here
  sorry

end base_conversion_subtraction_l31_31890


namespace compute_expression_l31_31101

-- Define the operation a Δ b
def Delta (a b : ℝ) : ℝ := a^2 - 2 * b

theorem compute_expression :
  let x := 3 ^ (Delta 4 10)
  let y := 4 ^ (Delta 2 3)
  Delta x y = ( -819.125 / 6561) :=
by 
  sorry

end compute_expression_l31_31101


namespace conversion_rates_l31_31055

noncomputable def teamADailyConversionRate (a b : ℝ) := 1.2 * b
noncomputable def teamBDailyConversionRate (a b : ℝ) := b

theorem conversion_rates (total_area : ℝ) (b : ℝ) (h1 : total_area = 1500) (h2 : b = 50) 
    (h3 : teamADailyConversionRate 1500 b * b = 1.2) 
    (h4 : teamBDailyConversionRate 1500 b = b) 
    (h5 : (1500 / teamBDailyConversionRate 1500 b) - 5 = 1500 / teamADailyConversionRate 1500 b) :
  teamADailyConversionRate 1500 b = 60 ∧ teamBDailyConversionRate 1500 b = 50 := 
by
  sorry

end conversion_rates_l31_31055


namespace A_share_value_l31_31198

-- Define the shares using the common multiplier x
variable (x : ℝ)

-- Define the shares in terms of x
def A_share := 5 * x
def B_share := 2 * x
def C_share := 4 * x
def D_share := 3 * x

-- Given condition that C gets Rs. 500 more than D
def condition := C_share - D_share = 500

-- State the theorem to determine A's share given the conditions
theorem A_share_value (h : condition) : A_share = 2500 := by 
  sorry

end A_share_value_l31_31198


namespace eggs_per_chicken_per_day_l31_31796

theorem eggs_per_chicken_per_day (E c d : ℕ) (hE : E = 36) (hc : c = 4) (hd : d = 3) :
  (E / d) / c = 3 := by
  sorry

end eggs_per_chicken_per_day_l31_31796


namespace at_most_n_zeros_l31_31337

-- Definitions of conditions
variables {α : Type*} [Inhabited α]

/-- Define the structure of the sheet of numbers with the given properties -/
structure sheet :=
(n : ℕ)
(val : ℕ → ℤ)

-- Assuming infinite sheet and the properties
variable (s : sheet)

-- Predicate for a row having only positive integers
def all_positive (r : ℕ → ℤ) : Prop := ∀ i, r i > 0

-- Define the initial row R which has all positive integers
variable {R : ℕ → ℤ}

-- Statement that each element in the row below is sum of element above and to the left
def below_sum (r R : ℕ → ℤ) (n : ℕ) : Prop := ∀ i, r i = R i + (if i = 0 then 0 else R (i - 1))

-- Variable for the row n below R
variable {Rn : ℕ → ℤ}

-- Main theorem statement
theorem at_most_n_zeros (n : ℕ) (hr : all_positive R) (hs : below_sum R Rn n) : 
  ∃ k ≤ n, Rn k = 0 ∨ Rn k > 0 := sorry

end at_most_n_zeros_l31_31337


namespace f_properties_l31_31046

open Real

-- Define the function f(x) = x^2
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the statement to be proved
theorem f_properties (x₁ x₂ : ℝ) (x : ℝ) (h : 0 < x) :
  (f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end f_properties_l31_31046


namespace problem1_solution_problem2_solution_l31_31428

theorem problem1_solution (x : ℝ) (h : 5 / (x - 1) = 1 / (2 * x + 1)) : x = -2 / 3 := sorry

theorem problem2_solution (x : ℝ) (h : 1 / (x - 2) + 2 = (1 - x) / (2 - x)) : false := sorry

end problem1_solution_problem2_solution_l31_31428


namespace line_circle_no_intersection_l31_31012

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l31_31012


namespace simplify_expression_l31_31345

variable {a : ℝ} (h1 : a ≠ -3) (h2 : a ≠ 3) (h3 : a ≠ 2) (h4 : 2 * a + 6 ≠ 0)

theorem simplify_expression : (1 / (a + 3) + 1 / (a ^ 2 - 9)) / ((a - 2) / (2 * a + 6)) = 2 / (a - 3) :=
by
  sorry

end simplify_expression_l31_31345


namespace painted_cubes_even_faces_l31_31298

theorem painted_cubes_even_faces :
  let L := 6 -- length of the block
  let W := 2 -- width of the block
  let H := 2 -- height of the block
  let total_cubes := 24 -- the block is cut into 24 1-inch cubes
  let cubes_even_faces := 12 -- the number of 1-inch cubes with even number of blue faces
  -- each cube has a total of 6 faces,
  -- we need to count how many cubes have an even number of painted faces.
  L * W * H = total_cubes → 
  cubes_even_faces = 12 := sorry

end painted_cubes_even_faces_l31_31298


namespace programmer_debugging_hours_l31_31093

theorem programmer_debugging_hours
    (total_hours : ℕ)
    (flow_chart_fraction : ℚ)
    (coding_fraction : ℚ)
    (meeting_fraction : ℚ)
    (flow_chart_hours : ℚ)
    (coding_hours : ℚ)
    (meeting_hours : ℚ)
    (debugging_hours : ℚ)
    (H1 : total_hours = 192)
    (H2 : flow_chart_fraction = 3 / 10)
    (H3 : coding_fraction = 3 / 8)
    (H4 : meeting_fraction = 1 / 5)
    (H5 : flow_chart_hours = flow_chart_fraction * total_hours)
    (H6 : coding_hours = coding_fraction * total_hours)
    (H7 : meeting_hours = meeting_fraction * total_hours)
    (H8 : debugging_hours = total_hours - (flow_chart_hours + coding_hours + meeting_hours))
    :
    debugging_hours = 24 :=
by 
  sorry

end programmer_debugging_hours_l31_31093


namespace angle_measure_l31_31621

variable (x : ℝ)

def complement (x : ℝ) : ℝ := 90 - x

def supplement (x : ℝ) : ℝ := 180 - x

theorem angle_measure (h : supplement x = 8 * complement x) : x = 540 / 7 := by
  sorry

end angle_measure_l31_31621


namespace johns_gas_usage_per_week_l31_31357

def mpg : ℕ := 30
def miles_to_work_each_way : ℕ := 20
def days_per_week_to_work : ℕ := 5
def leisure_miles_per_week : ℕ := 40

theorem johns_gas_usage_per_week : 
  (2 * miles_to_work_each_way * days_per_week_to_work + leisure_miles_per_week) / mpg = 8 :=
by
  sorry

end johns_gas_usage_per_week_l31_31357


namespace total_maggots_served_l31_31458

-- Define the conditions in Lean
def maggots_first_attempt : ℕ := 10
def maggots_second_attempt : ℕ := 10

-- Define the statement to prove
theorem total_maggots_served : maggots_first_attempt + maggots_second_attempt = 20 :=
by 
  sorry

end total_maggots_served_l31_31458


namespace determine_numbers_l31_31957

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l31_31957


namespace prime_sum_value_l31_31037

theorem prime_sum_value (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 2019) : 
  (p - 1)^(q - 1) = 1 ∨ (p - 1)^(q - 1) = 2016 :=
by
  sorry

end prime_sum_value_l31_31037


namespace f_5_eq_2_l31_31413

def f : ℕ → ℤ :=
sorry

axiom f_initial_condition : f 1 = 2

axiom f_functional_eq (a b : ℕ) : f (a + b) = 2 * f a + 2 * f b - 3 * f (a * b)

theorem f_5_eq_2 : f 5 = 2 :=
sorry

end f_5_eq_2_l31_31413


namespace problem_statement_period_property_symmetry_property_zero_property_l31_31991

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 3)

theorem problem_statement : ¬(∀ x : ℝ, (Real.pi / 2 < x ∧ x < Real.pi) → f x > f (x + ε))
  → ∃ x : ℝ, f (x + Real.pi) = 0 :=
by
  intro h
  use Real.pi / 6
  sorry

theorem period_property : ∀ k : ℤ, f (x + 2 * k * Real.pi) = f x :=
by
  intro k
  sorry

theorem symmetry_property : ∀ y : ℝ, f (8 * Real.pi / 3 - y) = f (8 * Real.pi / 3 + y) :=
by
  intro y
  sorry

theorem zero_property : f (Real.pi / 6 + Real.pi) = 0 :=
by
  sorry

end problem_statement_period_property_symmetry_property_zero_property_l31_31991


namespace sin_3x_over_4_period_l31_31798

noncomputable def sine_period (b : ℝ) : ℝ :=
  (2 * Real.pi) / b

theorem sin_3x_over_4_period :
  sine_period (3/4) = (8 * Real.pi) / 3 :=
by
  sorry

end sin_3x_over_4_period_l31_31798


namespace problem_1_l31_31394

theorem problem_1 : (-(5 / 8) / (14 / 3) * (-(16 / 5)) / (-(6 / 7))) = -1 / 2 :=
  sorry

end problem_1_l31_31394


namespace A_B_finish_l31_31446

theorem A_B_finish (A B C : ℕ → ℝ) (h1 : A + B + C = 1 / 6) (h2 : C = 1 / 10) :
  1 / (A + B) = 15 :=
by
  sorry

end A_B_finish_l31_31446


namespace cost_of_each_notebook_is_3_l31_31371

noncomputable def notebooks_cost (total_spent : ℕ) (backpack_cost : ℕ) (pens_cost : ℕ) (pencils_cost : ℕ) (num_notebooks : ℕ) : ℕ :=
  (total_spent - (backpack_cost + pens_cost + pencils_cost)) / num_notebooks

theorem cost_of_each_notebook_is_3 :
  notebooks_cost 32 15 1 1 5 = 3 :=
by
  sorry

end cost_of_each_notebook_is_3_l31_31371


namespace gcd_pow_minus_one_l31_31515

theorem gcd_pow_minus_one {m n a : ℕ} (hm : 0 < m) (hn : 0 < n) (ha : 2 ≤ a) : 
  Nat.gcd (a^n - 1) (a^m - 1) = a^(Nat.gcd m n) - 1 := 
sorry

end gcd_pow_minus_one_l31_31515


namespace mark_peters_pond_depth_l31_31286

theorem mark_peters_pond_depth :
  let mark_depth := 19
  let peter_depth := 5
  let three_times_peter_depth := 3 * peter_depth
  mark_depth - three_times_peter_depth = 4 :=
by
  sorry

end mark_peters_pond_depth_l31_31286


namespace seven_k_plus_four_l31_31434

theorem seven_k_plus_four (k m n : ℕ) (h1 : 4 * k + 5 = m^2) (h2 : 9 * k + 4 = n^2) (hk : k = 5) : 
  7 * k + 4 = 39 :=
by 
  -- assume conditions
  have h1' := h1
  have h2' := h2
  have hk' := hk
  sorry

end seven_k_plus_four_l31_31434


namespace complex_div_eq_l31_31823

theorem complex_div_eq (z1 z2 : ℂ) (h1 : z1 = 3 - i) (h2 : z2 = 2 + i) :
  z1 / z2 = 1 - i := by
  sorry

end complex_div_eq_l31_31823


namespace matrix_arithmetic_sequence_sum_l31_31028

theorem matrix_arithmetic_sequence_sum (a : ℕ → ℕ → ℕ)
  (h_row1 : ∀ i, 2 * a 4 2 = a 4 (i - 1) + a 4 (i + 1))
  (h_row2 : ∀ i, 2 * a 5 2 = a 5 (i - 1) + a 5 (i + 1))
  (h_row3 : ∀ i, 2 * a 6 2 = a 6 (i - 1) + a 6 (i + 1))
  (h_col1 : ∀ i, 2 * a 5 2 = a (i - 1) 2 + a (i + 1) 2)
  (h_sum : a 4 1 + a 4 2 + a 4 3 + a 5 1 + a 5 2 + a 5 3 + a 6 1 + a 6 2 + a 6 3 = 63)
  : a 5 2 = 7 := sorry

end matrix_arithmetic_sequence_sum_l31_31028


namespace mrs_hilt_rocks_proof_l31_31956

def num_rocks_already_placed : ℝ := 125.0
def total_num_rocks_planned : ℝ := 189
def num_more_rocks_needed : ℝ := 64

theorem mrs_hilt_rocks_proof : total_num_rocks_planned - num_rocks_already_placed = num_more_rocks_needed :=
by
  sorry

end mrs_hilt_rocks_proof_l31_31956


namespace jane_wins_game_l31_31677

noncomputable def jane_win_probability : ℚ :=
  1/3 / (1 - (2/3 * 1/3 * 2/3))

theorem jane_wins_game :
  jane_win_probability = 9/23 :=
by
  -- detailed proof steps would be filled in here
  sorry

end jane_wins_game_l31_31677


namespace locus_of_intersection_l31_31871

theorem locus_of_intersection (m : ℝ) :
  (∃ x y : ℝ, (m * x - y + 1 = 0) ∧ (x - m * y - 1 = 0)) ↔ (∃ x y : ℝ, (x - y = 0) ∨ (x - y + 1 = 0)) :=
by
  sorry

end locus_of_intersection_l31_31871


namespace car_initial_time_l31_31460

variable (t : ℝ)

theorem car_initial_time (h : 80 = 720 / (3/2 * t)) : t = 6 :=
sorry

end car_initial_time_l31_31460


namespace good_numbers_10_70_l31_31740

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def no_repeating_digits (n : ℕ) : Prop :=
  (n / 10 ≠ n % 10)

def is_good_number (n : ℕ) : Prop :=
  no_repeating_digits n ∧ (n % sum_of_digits n = 0)

theorem good_numbers_10_70 :
  is_good_number 10 ∧ is_good_number (10 + 11) ∧
  is_good_number 70 ∧ is_good_number (70 + 11) :=
by {
  -- Check that 10 is a good number
  -- Check that 21 is a good number
  -- Check that 70 is a good number
  -- Check that 81 is a good number
  sorry
}

end good_numbers_10_70_l31_31740


namespace shifted_parabola_equation_l31_31285

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the shifted parabola function
def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

-- Proposition to prove that the given parabola equation is correct after transformations
theorem shifted_parabola_equation : 
  ∀ x : ℝ, shifted_parabola x = -2 * (x + 1)^2 + 3 :=
by
  sorry

end shifted_parabola_equation_l31_31285


namespace number_of_ways_to_choose_one_book_is_correct_l31_31193

-- Definitions of the given problem conditions
def number_of_chinese_books : Nat := 10
def number_of_english_books : Nat := 7
def number_of_math_books : Nat := 5

-- Theorem stating the proof problem
theorem number_of_ways_to_choose_one_book_is_correct : 
  number_of_chinese_books + number_of_english_books + number_of_math_books = 22 := by
  -- This proof is left as an exercise.
  sorry

end number_of_ways_to_choose_one_book_is_correct_l31_31193


namespace percent_neither_filler_nor_cheese_l31_31295

-- Define the given conditions as constants
def total_weight : ℕ := 200
def filler_weight : ℕ := 40
def cheese_weight : ℕ := 30

-- Definition of the remaining weight that is neither filler nor cheese
def neither_weight : ℕ := total_weight - filler_weight - cheese_weight

-- Calculation of the percentage of the burger that is neither filler nor cheese
def percentage_neither : ℚ := (neither_weight : ℚ) / (total_weight : ℚ) * 100

-- The theorem to prove
theorem percent_neither_filler_nor_cheese :
  percentage_neither = 65 := by
  sorry

end percent_neither_filler_nor_cheese_l31_31295


namespace diana_age_l31_31086

open Classical

theorem diana_age :
  ∃ (D : ℚ), (∃ (C E : ℚ), C = 4 * D ∧ E = D + 5 ∧ C = E) ∧ D = 5/3 :=
by
  -- Definitions and conditions are encapsulated in the existential quantifiers and the proof concludes with D = 5/3.
  sorry

end diana_age_l31_31086


namespace valentino_farm_total_birds_l31_31689

-- The definitions/conditions from the problem statement
def chickens := 200
def ducks := 2 * chickens
def turkeys := 3 * ducks

-- The theorem to prove the total number of birds
theorem valentino_farm_total_birds : 
  chickens + ducks + turkeys = 1800 :=
by
  -- Proof is not required, so we use 'sorry'
  sorry

end valentino_farm_total_birds_l31_31689


namespace find_x_from_ratio_l31_31138

theorem find_x_from_ratio (x y k: ℚ) 
  (h1 : ∀ x y, (5 * x - 3) / (y + 20) = k) 
  (h2 : 5 * 1 - 3 = 2 * 22) (hy : y = 5) : 
  x = 58 / 55 := 
by 
  sorry

end find_x_from_ratio_l31_31138


namespace all_three_selected_l31_31112

-- Define the probabilities
def P_R : ℚ := 6 / 7
def P_Rv : ℚ := 1 / 5
def P_Rs : ℚ := 2 / 3
def P_Rv_given_R : ℚ := 2 / 5
def P_Rs_given_Rv : ℚ := 1 / 2

-- The probability that all three are selected
def P_all : ℚ := P_R * P_Rv_given_R * P_Rs_given_Rv

-- Prove that the calculated probability is equal to the given answer
theorem all_three_selected : P_all = 6 / 35 :=
by
  sorry

end all_three_selected_l31_31112


namespace odd_function_f_2_eq_2_l31_31586

noncomputable def f (x : ℝ) : ℝ := 
if x < 0 then x^2 + 3 * x else -(if -x < 0 then (-x)^2 + 3 * (-x) else x^2 + 3 * x)

theorem odd_function_f_2_eq_2 : f 2 = 2 :=
by
  -- sorry will be used to skip the actual proof
  sorry

end odd_function_f_2_eq_2_l31_31586


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l31_31509

variable {a b m : ℝ}

theorem sufficient_but_not_necessary_condition (h : a * m^2 < b * m^2) : a < b := by
  sorry

-- Additional statements to express the sufficiency and not necessity nature:
theorem not_necessary_condition (h : a < b) (hm : m = 0) : ¬ (a * m^2 < b * m^2) := by
  sorry

end sufficient_but_not_necessary_condition_not_necessary_condition_l31_31509


namespace prob_at_least_two_correct_l31_31414

-- Probability of guessing a question correctly
def prob_correct := 1 / 6

-- Probability of guessing a question incorrectly
def prob_incorrect := 5 / 6

-- Binomial probability mass function for k successes out of n trials
def binom_pmf (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ k) * ((1 - p) ^ (n - k))

-- Calculate probability P(X = 0)
def prob_X0 := binom_pmf 6 0 prob_correct

-- Calculate probability P(X = 1)
def prob_X1 := binom_pmf 6 1 prob_correct

-- Theorem for the desired probability
theorem prob_at_least_two_correct : 
  1 - (prob_X0 + prob_X1) = 34369 / 58420 := by
  sorry

end prob_at_least_two_correct_l31_31414


namespace diamond_19_98_l31_31131

variable {R : Type} [LinearOrderedField R]

noncomputable def diamond (x y : R) : R := sorry

axiom diamond_axiom1 : ∀ (x y : R) (hx : 0 < x) (hy : 0 < y), diamond (x * y) y = x * (diamond y y)

axiom diamond_axiom2 : ∀ (x : R) (hx : 0 < x), diamond (diamond x 1) x = diamond x 1

axiom diamond_axiom3 : diamond 1 1 = 1

theorem diamond_19_98 : diamond (19 : R) (98 : R) = 19 := 
sorry

end diamond_19_98_l31_31131


namespace simultaneous_equations_solution_l31_31319

-- Definition of the two equations
def eq1 (m x y : ℝ) : Prop := y = m * x + 5
def eq2 (m x y : ℝ) : Prop := y = (3 * m - 2) * x + 6

-- Lean theorem statement to check if the equations have a solution
theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 1) ↔ ∃ x y : ℝ, eq1 m x y ∧ eq2 m x y := 
sorry

end simultaneous_equations_solution_l31_31319


namespace reflect_A_across_x_axis_l31_31802

-- Define the point A
def A : ℝ × ℝ := (-3, 2)

-- Define the reflection function across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Theorem statement: The reflection of point A across the x-axis should be (-3, -2)
theorem reflect_A_across_x_axis : reflect_x A = (-3, -2) := by
  sorry

end reflect_A_across_x_axis_l31_31802


namespace isosceles_right_triangle_hypotenuse_length_l31_31258

theorem isosceles_right_triangle_hypotenuse_length (A B C : ℝ) (h1 : (A = 0) ∧ (B = 0) ∧ (C = 1)) (h2 : AC = 5) (h3 : BC = 5) : 
  AB = 5 * Real.sqrt 2 := 
sorry

end isosceles_right_triangle_hypotenuse_length_l31_31258


namespace five_student_committee_l31_31166

theorem five_student_committee : ∀ (students : Finset ℕ) (alice bob : ℕ), 
  alice ∈ students → bob ∈ students → students.card = 8 → ∃ (committees : Finset (Finset ℕ)),
  (∀ committee ∈ committees, alice ∈ committee ∧ bob ∈ committee) ∧
  ∀ committee ∈ committees, committee.card = 5 ∧ committees.card = 20 :=
by
  sorry

end five_student_committee_l31_31166


namespace find_x_from_arithmetic_mean_l31_31179

theorem find_x_from_arithmetic_mean (x : ℝ) 
  (h : (x + 10 + 18 + 3 * x + 16 + (x + 5) + (3 * x + 6)) / 6 = 25) : 
  x = 95 / 8 := by
  sorry

end find_x_from_arithmetic_mean_l31_31179


namespace football_team_practice_missed_days_l31_31121

theorem football_team_practice_missed_days 
(daily_practice_hours : ℕ) 
(total_practice_hours : ℕ) 
(days_in_week : ℕ) 
(h1 : daily_practice_hours = 5) 
(h2 : total_practice_hours = 30) 
(h3 : days_in_week = 7) : 
days_in_week - (total_practice_hours / daily_practice_hours) = 1 := 
by 
  sorry

end football_team_practice_missed_days_l31_31121


namespace Ravi_probability_l31_31705

-- Conditions from the problem
def P_Ram : ℚ := 4 / 7
def P_BothSelected : ℚ := 0.11428571428571428

-- Statement to prove
theorem Ravi_probability :
  ∃ P_Ravi : ℚ, P_Rami = 0.2 ∧ P_Ram * P_Ravi = P_BothSelected := by
  sorry

end Ravi_probability_l31_31705


namespace range_of_a_l31_31977

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(1 + a * x) - x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a (f a x) - x

theorem range_of_a (a : ℝ) : (F a 0 = 0 → F a e = 0) → 
  (0 < a ∧ a < (1 / (Real.exp 1 * Real.log 2))) :=
by
  sorry

end range_of_a_l31_31977


namespace minimum_value_y_l31_31372

noncomputable def y (x : ℚ) : ℚ := |3 - x| + |x - 2| + |-1 + x|

theorem minimum_value_y : ∃ x : ℚ, y x = 2 :=
by
  sorry

end minimum_value_y_l31_31372


namespace sequence_expression_l31_31439

noncomputable def a_n (n : ℕ) : ℤ :=
if n = 1 then -1 else 1 - 2^n

def S_n (a_n : ℕ → ℤ) (n : ℕ) : ℤ :=
2 * a_n n + n

theorem sequence_expression :
  ∀ n : ℕ, n > 0 → (a_n n = 1 - 2^n) :=
by
  intro n hn
  sorry

end sequence_expression_l31_31439


namespace solution_set_of_inequality_l31_31764

theorem solution_set_of_inequality (x : ℝ) : (|2 * x - 1| < 1) ↔ (0 < x ∧ x < 1) :=
sorry

end solution_set_of_inequality_l31_31764


namespace pepperoni_ratio_l31_31501

-- Definition of the problem's conditions
def total_pepperoni_slices : ℕ := 40
def slice_given_to_jelly_original : ℕ := 10
def slice_fallen_off : ℕ := 1

-- Our goal is to prove that the ratio is 3:10
theorem pepperoni_ratio (total_pepperoni_slices : ℕ) (slice_given_to_jelly_original : ℕ) (slice_fallen_off : ℕ) :
  (slice_given_to_jelly_original - slice_fallen_off) / (total_pepperoni_slices - slice_given_to_jelly_original) = 3 / 10 :=
by
  sorry

end pepperoni_ratio_l31_31501


namespace max_value_of_f_l31_31848

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + Real.sin (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, f x = 1 + Real.sqrt 2 := 
sorry

end max_value_of_f_l31_31848


namespace smallest_number_is_51_l31_31699

-- Definitions based on conditions
def conditions (x y : ℕ) : Prop :=
  (x + y = 2014) ∧ (∃ n a : ℕ, (x = 100 * n + a) ∧ (a < 100) ∧ (3 * n = y + 6))

-- The proof problem statement that needs to be proven
theorem smallest_number_is_51 :
  ∃ x y : ℕ, conditions x y ∧ min x y = 51 := 
sorry

end smallest_number_is_51_l31_31699


namespace distance_from_home_to_school_l31_31672

variable (t : ℕ) (D : ℕ)

-- conditions
def condition1 := 60 * (t - 10) = D
def condition2 := 50 * (t + 4) = D

-- the mathematical equivalent proof problem: proving the distance is 4200 given conditions
theorem distance_from_home_to_school :
  (∃ t, condition1 t 4200 ∧ condition2 t 4200) :=
  sorry

end distance_from_home_to_school_l31_31672


namespace sum_of_c_and_d_l31_31030

def digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

theorem sum_of_c_and_d 
  (c d : ℕ)
  (hcd : digit c)
  (hdd : digit d)
  (h1: (4*c) * 5 % 10 = 5)
  (h2: 215 = (10 * (4*(d*5) + c*5)) + d*10 + 5) :
  c + d = 5 := 
  sorry

end sum_of_c_and_d_l31_31030


namespace fraction_identity_l31_31274

theorem fraction_identity (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1009) : (a + b) / (a - b) = -1009 :=
by
  sorry

end fraction_identity_l31_31274


namespace subtraction_of_7_305_from_neg_3_219_l31_31507

theorem subtraction_of_7_305_from_neg_3_219 :
  -3.219 - 7.305 = -10.524 :=
by
  -- The proof would go here
  sorry

end subtraction_of_7_305_from_neg_3_219_l31_31507


namespace intersection_complement_eq_l31_31855

open Set

variable (U A B : Set ℕ)
  
theorem intersection_complement_eq : 
  U = {0, 1, 2, 3, 4} → 
  A = {0, 1, 3} → 
  B = {2, 3} → 
  A ∩ (U \ B) = {0, 1} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end intersection_complement_eq_l31_31855


namespace striped_octopus_has_8_legs_l31_31197

-- Definitions for Octopus and Statements
structure Octopus :=
  (legs : ℕ)
  (tellsTruth : Prop)

-- Given conditions translations
def tellsTruthCondition (o : Octopus) : Prop :=
  if o.legs % 2 = 0 then o.tellsTruth else ¬o.tellsTruth

def green_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def dark_blue_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def violet_octopus : Octopus :=
  { legs := 9, tellsTruth := sorry }  -- Placeholder truth value

def striped_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

-- Octopus statements (simplified for output purposes)
def green_statement := (green_octopus.legs = 8) ∧ (dark_blue_octopus.legs = 6)
def dark_blue_statement := (dark_blue_octopus.legs = 8) ∧ (green_octopus.legs = 7)
def violet_statement := (dark_blue_octopus.legs = 8) ∧ (violet_octopus.legs = 9)
def striped_statement := ¬(green_octopus.legs = 8 ∨ dark_blue_octopus.legs = 8 ∨ violet_octopus.legs = 8) ∧ (striped_octopus.legs = 8)

-- The goal to prove that the striped octopus has exactly 8 legs
theorem striped_octopus_has_8_legs : striped_octopus.legs = 8 :=
sorry

end striped_octopus_has_8_legs_l31_31197


namespace conditional_probability_l31_31804

/-
We define the probabilities of events A and B.
-/
variables (P : Set (Set α) → ℝ)
variable {α : Type*}

-- Event A: the animal lives up to 20 years old
def A : Set α := {x | true}   -- placeholder definition

-- Event B: the animal lives up to 25 years old
def B : Set α := {x | true}   -- placeholder definition

/-
Given conditions
-/
axiom P_A : P A = 0.8
axiom P_B : P B = 0.4

/-
Proof problem to show P(B | A) = 0.5
-/
theorem conditional_probability : P (B ∩ A) / P A = 0.5 :=
by
  sorry

end conditional_probability_l31_31804


namespace candies_per_packet_l31_31318

-- Define the given conditions
def monday_to_friday_candies_per_day := 2
def weekend_candies_per_day := 1
def weekdays := 5
def weekends := 2
def weeks := 3
def packets := 2

-- Calculate the number of candies Bobby eats in a week
def candies_per_week := (monday_to_friday_candies_per_day * weekdays) + (weekend_candies_per_day * weekends)

-- Calculate the total number of candies Bobby eats in the given 3 weeks
def total_candies_in_3_weeks := candies_per_week * weeks

-- Divide the total number of candies by the number of packets to find the candies per packet
theorem candies_per_packet : total_candies_in_3_weeks / packets = 18 := 
by
  -- Adding the proof placeholder
  sorry

end candies_per_packet_l31_31318


namespace average_speed_correct_l31_31546

noncomputable def total_distance := 120 + 70
noncomputable def total_time := 2
noncomputable def average_speed := total_distance / total_time

theorem average_speed_correct :
  average_speed = 95 := by
  sorry

end average_speed_correct_l31_31546


namespace problem_1_problem_2a_problem_2b_l31_31451

noncomputable def v_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def v_b : ℝ × ℝ := (3, -Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (v_a x).1 * (v_b).1 + (v_a x).2 * (v_b).2

theorem problem_1 (x : ℝ) (h : x ∈ Set.Icc 0 Real.pi) : 
  (v_a x).1 * (v_b).2 = (v_a x).2 * (v_b).1 → x = (5 * Real.pi / 6) :=
by
  sorry

theorem problem_2a : 
  ∃ x ∈ Set.Icc 0 Real.pi, f x = 3 ∧ ∀ y ∈ Set.Icc 0 Real.pi, f y ≤ 3 :=
by
  sorry

theorem problem_2b :
  ∃ x ∈ Set.Icc 0 Real.pi, f x = -2 * Real.sqrt 3 ∧ ∀ y ∈ Set.Icc 0 Real.pi, f y ≥ -2 * Real.sqrt 3 :=
by
  sorry

end problem_1_problem_2a_problem_2b_l31_31451


namespace area_and_cost_of_path_l31_31527

variables (length_field width_field path_width : ℝ) (cost_per_sq_m : ℝ)

noncomputable def area_of_path (length_field width_field path_width : ℝ) : ℝ :=
  let total_length := length_field + 2 * path_width
  let total_width := width_field + 2 * path_width
  let area_with_path := total_length * total_width
  let area_grass_field := length_field * width_field
  area_with_path - area_grass_field

noncomputable def cost_of_path (area_of_path cost_per_sq_m : ℝ) : ℝ :=
  area_of_path * cost_per_sq_m

theorem area_and_cost_of_path
  (length_field width_field path_width : ℝ)
  (cost_per_sq_m : ℝ)
  (h_length_field : length_field = 75)
  (h_width_field : width_field = 55)
  (h_path_width : path_width = 2.5)
  (h_cost_per_sq_m : cost_per_sq_m = 10) :
  area_of_path length_field width_field path_width = 675 ∧
  cost_of_path (area_of_path length_field width_field path_width) cost_per_sq_m = 6750 :=
by
  rw [h_length_field, h_width_field, h_path_width, h_cost_per_sq_m]
  simp [area_of_path, cost_of_path]
  sorry

end area_and_cost_of_path_l31_31527


namespace rectangular_solid_volume_l31_31909

variables {x y z : ℝ}

theorem rectangular_solid_volume :
  x * y = 15 ∧ y * z = 10 ∧ x * z = 6 ∧ x = 3 * y →
  x * y * z = 6 * Real.sqrt 5 :=
by
  intros h
  sorry

end rectangular_solid_volume_l31_31909


namespace area_of_shaded_triangle_l31_31720

-- Definitions of the conditions
def AC := 4
def BC := 3
def BD := 10
def CD := BD - BC

-- Statement of the proof problem
theorem area_of_shaded_triangle :
  (1 / 2 * CD * AC = 14) := by
  sorry

end area_of_shaded_triangle_l31_31720


namespace michael_exceeds_suresh_l31_31254

theorem michael_exceeds_suresh (P M S : ℝ) 
  (h_total : P + M + S = 2400)
  (h_p_m_ratio : P / 5 = M / 7)
  (h_m_s_ratio : M / 3 = S / 2) : M - S = 336 :=
by
  sorry

end michael_exceeds_suresh_l31_31254


namespace usual_time_to_cover_distance_l31_31153

variable (S T : ℝ)

-- Conditions:
-- 1. The man walks at 40% of his usual speed.
-- 2. He takes 24 minutes more to cover the same distance at this reduced speed.
-- 3. Usual speed is S.
-- 4. Usual time to cover the distance is T.

def usual_speed := S
def usual_time := T
def reduced_speed := 0.4 * S
def extra_time := 24

-- Question: Prove the man's usual time to cover the distance is 16 minutes.
theorem usual_time_to_cover_distance : T = 16 := 
by
  have speed_relation : S / (0.4 * S) = (T + 24) / T :=
    sorry
  have simplified_speed_relation : 2.5 = (T + 24) / T :=
    sorry
  have cross_multiplication_step : 2.5 * T = T + 24 :=
    sorry
  have solve_for_T_step : 1.5 * T = 24 :=
    sorry
  have final_step : T = 16 :=
    sorry
  exact final_step

end usual_time_to_cover_distance_l31_31153


namespace inequality_no_solution_l31_31425

theorem inequality_no_solution : 
  ∀ x : ℝ, -2 < (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) ∧ (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) < 2 → false :=
by sorry

end inequality_no_solution_l31_31425


namespace division_by_fraction_equiv_neg_multiplication_l31_31129

theorem division_by_fraction_equiv_neg_multiplication (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 :=
by
  -- Proof would go here, but we use sorry to skip the proof for now.
  sorry

end division_by_fraction_equiv_neg_multiplication_l31_31129


namespace john_initial_money_l31_31302

variable (X S : ℕ)
variable (L : ℕ := 500)
variable (cond1 : L = S - 600)
variable (cond2 : X = S + L)

theorem john_initial_money : X = 1600 :=
by
  sorry

end john_initial_money_l31_31302


namespace inequality_of_fractions_l31_31866

theorem inequality_of_fractions
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (c d : ℝ) (h3 : c < d) (h4 : d < 0)
  (e : ℝ) (h5 : e < 0) :
  (e / ((a - c)^2)) > (e / ((b - d)^2)) :=
by
  sorry

end inequality_of_fractions_l31_31866


namespace find_side_a_l31_31141

noncomputable def side_a (b : ℝ) (A : ℝ) (S : ℝ) : ℝ :=
  2 * S / (b * Real.sin A)

theorem find_side_a :
  let b := 2
  let A := Real.pi * 2 / 3 -- 120 degrees in radians
  let S := 2 * Real.sqrt 3
  side_a b A S = 4 :=
by
  let b := 2
  let A := Real.pi * 2 / 3
  let S := 2 * Real.sqrt 3
  show side_a b A S = 4
  sorry

end find_side_a_l31_31141


namespace range_of_expression_l31_31664

noncomputable def f (x : ℝ) := |Real.log x / Real.log 2|

theorem range_of_expression (a b : ℝ) (h_f_eq : f a = f b) (h_a_lt_b : a < b) :
  f a = f b → a < b → (∃ c > 3, c = (2 / a) + (1 / b)) := by
  sorry

end range_of_expression_l31_31664


namespace remaining_sessions_l31_31846

theorem remaining_sessions (total_sessions : ℕ) (p1_sessions : ℕ) (p2_sessions_more : ℕ) (remaining_sessions : ℕ) :
  total_sessions = 25 →
  p1_sessions = 6 →
  p2_sessions_more = 5 →
  remaining_sessions = total_sessions - (p1_sessions + (p1_sessions + p2_sessions_more)) →
  remaining_sessions = 8 :=
by
  intros
  sorry

end remaining_sessions_l31_31846


namespace rods_in_one_mile_l31_31094

-- Define the conditions as assumptions in Lean

-- 1. 1 mile = 8 furlongs
def mile_to_furlong : ℕ := 8

-- 2. 1 furlong = 220 paces
def furlong_to_pace : ℕ := 220

-- 3. 1 pace = 0.2 rods
def pace_to_rod : ℝ := 0.2

-- Define the statement to be proven
theorem rods_in_one_mile : (mile_to_furlong * furlong_to_pace * pace_to_rod) = 352 := by
  sorry

end rods_in_one_mile_l31_31094


namespace true_discount_correct_l31_31590

noncomputable def true_discount (FV BD : ℝ) : ℝ :=
  BD / (1 + (BD / FV))

theorem true_discount_correct
  (FV BD : ℝ)
  (hFV : FV = 2260)
  (hBD : BD = 428.21) :
  true_discount FV BD = 360.00 :=
by
  sorry

end true_discount_correct_l31_31590


namespace ganesh_ram_sohan_work_time_l31_31958

theorem ganesh_ram_sohan_work_time (G R S : ℝ)
  (H1 : G + R = 1 / 24)
  (H2 : S = 1 / 48) : (G + R + S = 1 / 16) ∧ (1 / (G + R + S) = 16) :=
by
  sorry

end ganesh_ram_sohan_work_time_l31_31958


namespace side_length_of_square_l31_31239

theorem side_length_of_square (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (a b c : ℝ) (h_leg1 : a = 12) (h_leg2 : b = 9) (h_right : c^2 = a^2 + b^2) :
  ∃ s : ℝ, s = 45/8 :=
by 
  -- Given the right triangle with legs 12 cm and 9 cm, the length of the side of the square is 45/8 cm
  let s := 45/8
  use s
  sorry

end side_length_of_square_l31_31239


namespace g_expression_f_expression_l31_31755

-- Given functions f and g that satisfy the conditions
variable {f g : ℝ → ℝ}

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom sum_eq : ∀ x, f x + g x = 2^x + 2 * x

-- Theorem statements to prove
theorem g_expression : g = fun x => 2^x := by sorry
theorem f_expression : f = fun x => 2 * x := by sorry

end g_expression_f_expression_l31_31755


namespace sum_of_k_values_l31_31788

theorem sum_of_k_values (k : ℤ) :
  (∃ (r s : ℤ), (r ≠ s) ∧ (3 * r * s = 9) ∧ (r + s = k / 3)) → k = 0 :=
by sorry

end sum_of_k_values_l31_31788


namespace perimeter_reduction_percentage_l31_31220

-- Given initial dimensions x and y
-- Initial Perimeter
def initial_perimeter (x y : ℝ) : ℝ := 2 * (x + y)

-- First reduction
def first_reduction_length (x : ℝ) : ℝ := 0.9 * x
def first_reduction_width (y : ℝ) : ℝ := 0.8 * y

-- New perimeter after first reduction
def new_perimeter_first (x y : ℝ) : ℝ := 2 * (first_reduction_length x + first_reduction_width y)

-- Condition: new perimeter is 88% of the initial perimeter
def perimeter_condition (x y : ℝ) : Prop := new_perimeter_first x y = 0.88 * initial_perimeter x y

-- Solve for x in terms of y
def solve_for_x (y : ℝ) : ℝ := 4 * y

-- Second reduction
def second_reduction_length (x : ℝ) : ℝ := 0.8 * x
def second_reduction_width (y : ℝ) : ℝ := 0.9 * y

-- New perimeter after second reduction
def new_perimeter_second (x y : ℝ) : ℝ := 2 * (second_reduction_length x + second_reduction_width y)

-- Proof statement
theorem perimeter_reduction_percentage (x y : ℝ) (h : perimeter_condition x y) : 
  new_perimeter_second x y = 0.82 * initial_perimeter x y :=
by
  sorry

end perimeter_reduction_percentage_l31_31220


namespace largest_real_number_l31_31597

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l31_31597


namespace minimum_value_of_f_minimum_value_achieved_sum_of_squares_ge_three_l31_31225

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 3 := by
  sorry

theorem minimum_value_achieved : ∃ x : ℝ, f x = 3 := by
  sorry

theorem sum_of_squares_ge_three (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := by
  sorry

end minimum_value_of_f_minimum_value_achieved_sum_of_squares_ge_three_l31_31225


namespace mary_candies_l31_31323

-- The conditions
def bob_candies : Nat := 10
def sue_candies : Nat := 20
def john_candies : Nat := 5
def sam_candies : Nat := 10
def total_candies : Nat := 50

-- The theorem to prove
theorem mary_candies :
  total_candies - (bob_candies + sue_candies + john_candies + sam_candies) = 5 := by
  -- Here is where the proof would go; currently using sorry to skip the proof
  sorry

end mary_candies_l31_31323


namespace evaluates_to_m_times_10_pow_1012_l31_31524

theorem evaluates_to_m_times_10_pow_1012 :
  let a := (3:ℤ) ^ 1010
  let b := (4:ℤ) ^ 1012
  (a + b) ^ 2 - (a - b) ^ 2 = 10 ^ 3642 := by
  sorry

end evaluates_to_m_times_10_pow_1012_l31_31524


namespace scrabble_champions_l31_31305

theorem scrabble_champions :
  let total_champions := 10
  let men_percentage := 0.40
  let men_champions := total_champions * men_percentage
  let bearded_percentage := 0.40
  let non_bearded_percentage := 0.60

  let bearded_men_champions := men_champions * bearded_percentage
  let non_bearded_men_champions := men_champions * non_bearded_percentage

  let bearded_bald_percentage := 0.60
  let bearded_with_hair_percentage := 0.40
  let non_bearded_bald_percentage := 0.30
  let non_bearded_with_hair_percentage := 0.70

  (bearded_men_champions * bearded_bald_percentage).round = 2 ∧
  (bearded_men_champions * bearded_with_hair_percentage).round = 1 ∧
  (non_bearded_men_champions * non_bearded_bald_percentage).round = 2 ∧
  (non_bearded_men_champions * non_bearded_with_hair_percentage).round = 4 :=
by 
sorry

end scrabble_champions_l31_31305


namespace distinct_values_least_count_l31_31652

theorem distinct_values_least_count (total_integers : ℕ) (mode_count : ℕ) (unique_mode : Prop) 
  (h1 : total_integers = 3200)
  (h2 : mode_count = 17)
  (h3 : unique_mode):
  ∃ (least_count : ℕ), least_count = 200 := by
  sorry

end distinct_values_least_count_l31_31652


namespace restore_salary_l31_31771

variable (W : ℝ) -- Define the initial wage as a real number
variable (newWage : ℝ := 0.7 * W) -- New wage after a 30% reduction

-- Define the hypothesis for the initial wage reduction
theorem restore_salary : (100 * (W / (0.7 * W) - 1)) = 42.86 :=
by
  sorry

end restore_salary_l31_31771


namespace calc_fraction_l31_31924

theorem calc_fraction : (36 + 12) / (6 - 3) = 16 :=
by
  sorry

end calc_fraction_l31_31924


namespace chickens_in_farm_l31_31761

theorem chickens_in_farm (c b : ℕ) (h1 : c + b = 9) (h2 : 2 * c + 4 * b = 26) : c = 5 := by sorry

end chickens_in_farm_l31_31761


namespace perfect_square_mod_3_l31_31314

theorem perfect_square_mod_3 (n : ℤ) : n^2 % 3 = 0 ∨ n^2 % 3 = 1 :=
sorry

end perfect_square_mod_3_l31_31314


namespace sqrt_two_squared_l31_31519

noncomputable def sqrt_two : Real := Real.sqrt 2

theorem sqrt_two_squared : (sqrt_two) ^ 2 = 2 :=
by
  sorry

end sqrt_two_squared_l31_31519


namespace percentage_water_in_fresh_grapes_is_65_l31_31016

noncomputable def percentage_water_in_fresh_grapes 
  (weight_fresh : ℝ) (weight_dried : ℝ) (percentage_water_dried : ℝ) : ℝ :=
  100 - ((weight_dried / weight_fresh) - percentage_water_dried / 100 * weight_dried / weight_fresh) * 100

theorem percentage_water_in_fresh_grapes_is_65 :
  percentage_water_in_fresh_grapes 400 155.56 10 = 65 := 
by
  sorry

end percentage_water_in_fresh_grapes_is_65_l31_31016


namespace shop_conditions_l31_31005

theorem shop_conditions (x y : ℕ) :
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) ↔ 
  ∃ x y : ℕ, 7 * x + 7 = y ∧ 9 * (x - 1) = y :=
sorry

end shop_conditions_l31_31005


namespace find_value_of_expression_l31_31579

theorem find_value_of_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) : x^2 + 2*x + 3 = 4 := by
  sorry

end find_value_of_expression_l31_31579


namespace farmer_land_owned_l31_31007

def total_land (farmer_land : ℝ) (cleared_land : ℝ) : Prop :=
  cleared_land = 0.9 * farmer_land

def cleared_with_tomato (cleared_land : ℝ) (tomato_land : ℝ) : Prop :=
  tomato_land = 0.1 * cleared_land
  
def tomato_land_given (tomato_land : ℝ) : Prop :=
  tomato_land = 90

theorem farmer_land_owned (T : ℝ) :
  (∃ cleared : ℝ, total_land T cleared ∧ cleared_with_tomato cleared 90) → T = 1000 :=
by
  sorry

end farmer_land_owned_l31_31007


namespace find_y_l31_31033

theorem find_y (x y : ℝ) (h₁ : x^2 - 2 * x + 5 = y + 3) (h₂ : x = 5) : y = 17 :=
by
  sorry

end find_y_l31_31033


namespace systematic_sampling_works_l31_31260

def missiles : List ℕ := List.range' 1 60 

-- Define the systematic sampling function
def systematic_sampling (start interval n : ℕ) : List ℕ :=
  List.range' 0 n |>.map (λ i => start + i * interval)

-- Stating the proof problem.
theorem systematic_sampling_works :
  systematic_sampling 5 12 5 = [5, 17, 29, 41, 53] :=
sorry

end systematic_sampling_works_l31_31260


namespace largest_c_value_l31_31237

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, x^2 + 5 * x + c = -3) → c ≤ 13 / 4 :=
sorry

end largest_c_value_l31_31237


namespace cannot_cover_chessboard_with_one_corner_removed_l31_31098

theorem cannot_cover_chessboard_with_one_corner_removed :
  ¬ (∃ (f : Fin (8*8 - 1) → Fin (64-1) × Fin (64-1)), 
        (∀ (i j : Fin (64-1)), 
          i ≠ j → f i ≠ f j) ∧ 
        (∀ (i : Fin (8 * 8 - 1)), 
          (f i).fst + (f i).snd = 2)) :=
by
  sorry

end cannot_cover_chessboard_with_one_corner_removed_l31_31098


namespace correct_product_l31_31610

-- We define the conditions
def number1 : ℝ := 0.85
def number2 : ℝ := 3.25
def without_decimal_points_prod : ℕ := 27625

-- We state the problem
theorem correct_product (h1 : (85 : ℕ) * (325 : ℕ) = without_decimal_points_prod)
                        (h2 : number1 * number2 * 10000 = (without_decimal_points_prod : ℝ)) :
  number1 * number2 = 2.7625 :=
by sorry

end correct_product_l31_31610


namespace proportion_of_white_pieces_l31_31525

theorem proportion_of_white_pieces (x : ℕ) (h1 : 0 < x) :
  let total_pieces := 3 * x
  let white_pieces := x + (1 - (5 / 9)) * x
  (white_pieces / total_pieces) = (13 / 27) :=
by
  sorry

end proportion_of_white_pieces_l31_31525


namespace negation_of_exists_gt_implies_forall_leq_l31_31952

theorem negation_of_exists_gt_implies_forall_leq (x : ℝ) (h : 0 < x) :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end negation_of_exists_gt_implies_forall_leq_l31_31952


namespace cab_driver_income_l31_31490

theorem cab_driver_income (x2 : ℕ) :
  (600 + x2 + 450 + 400 + 800) / 5 = 500 → x2 = 250 :=
by
  sorry

end cab_driver_income_l31_31490


namespace volume_region_between_spheres_l31_31521

theorem volume_region_between_spheres 
    (r1 r2 : ℝ) 
    (h1 : r1 = 4) 
    (h2 : r2 = 7) 
    : 
    ( (4/3) * π * r2^3 - (4/3) * π * r1^3 ) = 372 * π := 
    sorry

end volume_region_between_spheres_l31_31521


namespace sum_of_coefficients_eq_one_l31_31401

theorem sum_of_coefficients_eq_one (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x - 3) ^ 4 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  a + a₁ + a₂ + a₃ + a₄ = 1 :=
by
  intros h
  specialize h 1
  -- Specific calculation steps would go here
  sorry

end sum_of_coefficients_eq_one_l31_31401


namespace rectangle_diagonal_length_l31_31485

theorem rectangle_diagonal_length (P : ℝ) (L W D : ℝ) 
  (hP : P = 72) 
  (h_ratio : 3 * W = 2 * L) 
  (h_perimeter : 2 * (L + W) = P) :
  D = Real.sqrt (L * L + W * W) :=
sorry

end rectangle_diagonal_length_l31_31485


namespace inequality_proof_l31_31171

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a^3 + b^3 + c^3 = 3) :
  (1 / (a^4 + 3) + 1 / (b^4 + 3) + 1 / (c^4 + 3) >= 3 / 4) :=
by
  sorry

end inequality_proof_l31_31171


namespace minyoung_gave_nine_notebooks_l31_31381

theorem minyoung_gave_nine_notebooks (original left given : ℕ) (h1 : original = 17) (h2 : left = 8) (h3 : given = original - left) : given = 9 :=
by
  rw [h1, h2] at h3
  exact h3

end minyoung_gave_nine_notebooks_l31_31381


namespace simplify_expression_l31_31713

theorem simplify_expression (m : ℝ) (h : m ≠ 0) : (4 * m^2 - 2 * m) / (2 * m) = 2 * m - 1 := by
  sorry

end simplify_expression_l31_31713


namespace monomial_same_type_m_n_sum_l31_31934

theorem monomial_same_type_m_n_sum (m n : ℕ) (x y : ℤ) 
  (h1 : 2 * x ^ (m - 1) * y ^ 2 = 1/3 * x ^ 2 * y ^ (n + 1)) : 
  m + n = 4 := 
sorry

end monomial_same_type_m_n_sum_l31_31934


namespace johns_personal_payment_l31_31127

theorem johns_personal_payment 
  (cost_per_hearing_aid : ℕ)
  (num_hearing_aids : ℕ)
  (deductible : ℕ)
  (coverage_percent : ℕ)
  (coverage_limit : ℕ) 
  (total_payment : ℕ)
  (insurance_payment_over_limit : ℕ) : 
  cost_per_hearing_aid = 2500 ∧ 
  num_hearing_aids = 2 ∧ 
  deductible = 500 ∧ 
  coverage_percent = 80 ∧ 
  coverage_limit = 3500 →
  total_payment = cost_per_hearing_aid * num_hearing_aids - deductible →
  insurance_payment_over_limit = max 0 (coverage_percent * total_payment / 100 - coverage_limit) →
  (total_payment - min (coverage_percent * total_payment / 100) coverage_limit + deductible = 1500) :=
by
  intros
  sorry

end johns_personal_payment_l31_31127


namespace darnell_avg_yards_eq_11_l31_31905

-- Defining the given conditions
def malikYardsPerGame := 18
def josiahYardsPerGame := 22
def numberOfGames := 4
def totalYardsRun := 204

-- Defining the corresponding total yards for Malik and Josiah
def malikTotalYards := malikYardsPerGame * numberOfGames
def josiahTotalYards := josiahYardsPerGame * numberOfGames

-- The combined total yards for Malik and Josiah
def combinedTotal := malikTotalYards + josiahTotalYards

-- Calculate Darnell's total yards and average per game
def darnellTotalYards := totalYardsRun - combinedTotal
def darnellAverageYardsPerGame := darnellTotalYards / numberOfGames

-- Now, we write the theorem to prove darnell's average yards per game
theorem darnell_avg_yards_eq_11 : darnellAverageYardsPerGame = 11 := by
  sorry

end darnell_avg_yards_eq_11_l31_31905


namespace sum_of_first_eight_terms_l31_31443

-- Define the first term, common ratio, and the number of terms
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 8

-- Sum of the first n terms of a geometric sequence
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Proof statement
theorem sum_of_first_eight_terms : geometric_sum a r n = 3280 / 6561 :=
by
  sorry

end sum_of_first_eight_terms_l31_31443


namespace calculate_total_interest_l31_31336

theorem calculate_total_interest :
  let total_money := 9000
  let invested_at_8_percent := 4000
  let invested_at_9_percent := total_money - invested_at_8_percent
  let interest_rate_8 := 0.08
  let interest_rate_9 := 0.09
  let interest_from_8_percent := invested_at_8_percent * interest_rate_8
  let interest_from_9_percent := invested_at_9_percent * interest_rate_9
  let total_interest := interest_from_8_percent + interest_from_9_percent
  total_interest = 770 :=
by
  sorry

end calculate_total_interest_l31_31336


namespace intersection_A_B_l31_31538

def A := { x : ℝ | x / (x - 1) ≥ 0 }
def B := { y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B : A ∩ B = { y : ℝ | y > 1 } :=
by sorry

end intersection_A_B_l31_31538


namespace racing_magic_circle_time_l31_31948

theorem racing_magic_circle_time
  (T : ℕ) -- Time taken by the racing magic to circle the track once
  (bull_rounds_per_hour : ℕ := 40) -- Rounds the Charging Bull makes in an hour
  (meet_time_minutes : ℕ := 6) -- Time in minutes to meet at starting point
  (charging_bull_seconds_per_round : ℕ := 3600 / bull_rounds_per_hour) -- Time in seconds per Charging Bull round
  (meet_time_seconds : ℕ := meet_time_minutes * 60) -- Time in seconds to meet at starting point
  (rounds_by_bull : ℕ := meet_time_seconds / charging_bull_seconds_per_round) -- Rounds completed by the Charging Bull to meet again
  (rounds_by_magic : ℕ := meet_time_seconds / T) -- Rounds completed by the Racing Magic to meet again
  (h1 : rounds_by_magic = 1) -- Racing Magic completes 1 round in the meet time
  : T = 360 := -- Racing Magic takes 360 seconds to circle the track once
  sorry

end racing_magic_circle_time_l31_31948


namespace height_of_model_l31_31338

noncomputable def original_monument_height : ℝ := 100
noncomputable def original_monument_radius : ℝ := 20
noncomputable def original_monument_volume : ℝ := 125600
noncomputable def model_volume : ℝ := 1.256

theorem height_of_model : original_monument_height / (original_monument_volume / model_volume)^(1/3) = 1 :=
by
  sorry

end height_of_model_l31_31338


namespace squares_count_correct_l31_31057

-- Assuming basic setup and coordinate system.
def is_valid_point (x y : ℕ) : Prop :=
  x ≤ 8 ∧ y ≤ 8

-- Checking if a point (a, b) in the triangle as described.
def is_in_triangle (a b : ℕ) : Prop :=
  0 ≤ b ∧ b ≤ a ∧ a ≤ 4

-- Function derived from the solution detailing the number of such squares.
def count_squares (a b : ℕ) : ℕ :=
  -- Placeholder to represent the derived formula - to be replaced with actual derivation function
  (9 - a + b) * (a + b + 1) - 1

-- Statement to prove
theorem squares_count_correct (a b : ℕ) (h : is_in_triangle a b) :
  ∃ n, n = count_squares a b := 
sorry

end squares_count_correct_l31_31057


namespace discriminant_square_eq_l31_31025

variable {a b c x : ℝ}

-- Condition: a ≠ 0
axiom h_a : a ≠ 0

-- Condition: x is a root of the quadratic equation ax^2 + bx + c = 0
axiom h_root : a * x^2 + b * x + c = 0

theorem discriminant_square_eq (h_a : a ≠ 0) (h_root : a * x^2 + b * x + c = 0) :
  (2 * a * x + b)^2 = b^2 - 4 * a * c :=
by 
  sorry

end discriminant_square_eq_l31_31025


namespace center_of_circle_l31_31739

theorem center_of_circle (x y : ℝ) : (x^2 + y^2 - 10 * x + 4 * y + 13 = 0) → (x - y = 7) :=
by
  -- Statement, proof omitted
  sorry

end center_of_circle_l31_31739


namespace manufacturing_percentage_l31_31301

theorem manufacturing_percentage (a b : ℕ) (h1 : a = 108) (h2 : b = 360) : (a / b : ℚ) * 100 = 30 :=
by
  sorry

end manufacturing_percentage_l31_31301


namespace fifth_inequality_l31_31031

theorem fifth_inequality (h1: 1 / Real.sqrt 2 < 1)
                         (h2: 1 / Real.sqrt 2 + 1 / Real.sqrt 6 < Real.sqrt 2)
                         (h3: 1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 < Real.sqrt 3) :
                         1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 + 1 / Real.sqrt 20 + 1 / Real.sqrt 30 < Real.sqrt 5 := 
sorry

end fifth_inequality_l31_31031


namespace gain_percent_calculation_l31_31006

def gain : ℝ := 0.70
def cost_price : ℝ := 70.0

theorem gain_percent_calculation : (gain / cost_price) * 100 = 1 := by
  sorry

end gain_percent_calculation_l31_31006


namespace problem_l31_31856

variables (x y z : ℝ)

theorem problem :
  x - y - z = 3 ∧ yz - xy - xz = 3 → x^2 + y^2 + z^2 = 3 :=
by
  sorry

end problem_l31_31856


namespace five_digit_numbers_greater_than_20314_and_formable_with_0_to_5_l31_31994

theorem five_digit_numbers_greater_than_20314_and_formable_with_0_to_5 :
  (∃ (f : Fin 6 → Fin 5) (n : ℕ), 
    (n = 120 * 3 + 24 * 4 + 6 * 3 - 1) ∧
    (n = 473) ∧ 
    (∀ (x : Fin 6), f x = 0 ∨ f x = 1 ∨ f x = 2 ∨ f x = 3 ∨ f x = 4 ∨ f x = 5) ∧
    (∀ (i j : Fin 5), i ≠ j → f i ≠ f j)) :=
sorry

end five_digit_numbers_greater_than_20314_and_formable_with_0_to_5_l31_31994


namespace first_year_with_sum_of_digits_10_after_2200_l31_31698

/-- Prove that the first year after 2200 in which the sum of the digits equals 10 is 2224. -/
theorem first_year_with_sum_of_digits_10_after_2200 :
  ∃ y, y > 2200 ∧ (List.sum (y.digits 10) = 10) ∧ 
       ∀ z, (2200 < z ∧ z < y) → (List.sum (z.digits 10) ≠ 10) :=
sorry

end first_year_with_sum_of_digits_10_after_2200_l31_31698


namespace smallest_angle_l31_31282

noncomputable def smallest_angle_in_triangle (a b c : ℝ) : ℝ :=
  if h : 0 <= a ∧ 0 <= b ∧ 0 <= c ∧ a + b > c ∧ a + c > b ∧ b + c > a then
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  else
    0

theorem smallest_angle (a b c : ℝ) (h₁ : a = 4) (h₂ : b = 3) (h₃ : c = 2) :
  smallest_angle_in_triangle a b c = Real.arccos (7 / 8) :=
sorry

end smallest_angle_l31_31282


namespace paper_area_difference_l31_31206

def area (length width : ℕ) : ℕ := length * width

def combined_area (length width : ℕ) : ℕ := 2 * (area length width)

def sq_inch_to_sq_ft (sq_inch : ℕ) : ℕ := sq_inch / 144

theorem paper_area_difference :
  sq_inch_to_sq_ft (combined_area 15 24 - combined_area 12 18) = 2 :=
by
  sorry

end paper_area_difference_l31_31206


namespace ice_cream_cost_proof_l31_31461

-- Assume the cost of the ice cream and toppings
def cost_of_ice_cream : ℝ := 2 -- Ice cream cost in dollars
def cost_per_topping : ℝ := 0.5 -- Cost per topping in dollars
def total_cost_of_sundae_with_10_toppings : ℝ := 7 -- Total cost in dollars

theorem ice_cream_cost_proof :
  (∀ (cost_of_ice_cream : ℝ), 
    total_cost_of_sundae_with_10_toppings = cost_of_ice_cream + 10 * cost_per_topping) →
  cost_of_ice_cream = 2 :=
by
  sorry

end ice_cream_cost_proof_l31_31461


namespace solution_set_inequality_l31_31878

theorem solution_set_inequality (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x > 3 / 2 ∨ x < -1 := 
by sorry

end solution_set_inequality_l31_31878


namespace perimeter_of_triangle_l31_31580

-- Define the side lengths of the triangle
def side1 : ℕ := 2
def side2 : ℕ := 7

-- Define the third side of the triangle, which is an even number and satisfies the triangle inequality conditions
def side3 : ℕ := 6

-- Define the theorem to prove the perimeter of the triangle
theorem perimeter_of_triangle : side1 + side2 + side3 = 15 := by
  -- The proof is omitted for brevity
  sorry

end perimeter_of_triangle_l31_31580


namespace option_C_is_proposition_l31_31091

def is_proposition (s : Prop) : Prop := ∃ p : Prop, s = p

theorem option_C_is_proposition : is_proposition (4 + 3 = 8) := sorry

end option_C_is_proposition_l31_31091


namespace negation_equivalence_l31_31637

theorem negation_equivalence (x : ℝ) :
  (¬ (x ≥ 1 → x^2 - 4*x + 2 ≥ -1)) ↔ (x < 1 → x^2 - 4*x + 2 < -1) :=
by
  sorry

end negation_equivalence_l31_31637


namespace correct_operation_l31_31884

theorem correct_operation (x y a b : ℝ) :
  (-2 * x) * (3 * y) = -6 * x * y :=
by
  sorry

end correct_operation_l31_31884


namespace average_pages_per_day_is_correct_l31_31558

-- Definitions based on the given conditions
def first_book_pages := 249
def first_book_days := 3

def second_book_pages := 379
def second_book_days := 5

def third_book_pages := 480
def third_book_days := 6

-- Definition of total pages read
def total_pages := first_book_pages + second_book_pages + third_book_pages

-- Definition of total days spent reading
def total_days := first_book_days + second_book_days + third_book_days

-- Definition of expected average pages per day
def expected_average_pages_per_day := 79.14

-- The theorem to prove
theorem average_pages_per_day_is_correct : (total_pages.toFloat / total_days.toFloat) = expected_average_pages_per_day :=
by
  sorry

end average_pages_per_day_is_correct_l31_31558


namespace arithmetic_sequence_a7_value_l31_31173

variable (a : ℕ → ℝ) (a1 a13 a7 : ℝ)

theorem arithmetic_sequence_a7_value
  (h1 : a 1 = a1)
  (h13 : a 13 = a13)
  (h_sum : a1 + a13 = 12)
  (h_arith : 2 * a7 = a1 + a13) :
  a7 = 6 :=
by
  sorry

end arithmetic_sequence_a7_value_l31_31173


namespace frank_handed_cashier_amount_l31_31881

-- Place conditions as definitions
def cost_chocolate_bar : ℕ := 2
def cost_bag_chip : ℕ := 3
def num_chocolate_bars : ℕ := 5
def num_bag_chips : ℕ := 2
def change_received : ℕ := 4

-- Define the target theorem (Lean 4 statement)
theorem frank_handed_cashier_amount :
  (num_chocolate_bars * cost_chocolate_bar + num_bag_chips * cost_bag_chip + change_received = 20) := 
sorry

end frank_handed_cashier_amount_l31_31881


namespace circle_Q_radius_l31_31290

theorem circle_Q_radius
  (radius_P : ℝ := 2)
  (radius_S : ℝ := 4)
  (u v : ℝ)
  (h1: (2 + v)^2 = (2 + u)^2 + v^2)
  (h2: (4 - v)^2 = u^2 + v^2)
  (h3: v = u + u^2 / 2)
  (h4: v = 2 - u^2 / 4) :
  v = 16 / 9 :=
by
  /- Proof goes here. -/
  sorry

end circle_Q_radius_l31_31290


namespace volleyball_height_30_l31_31882

theorem volleyball_height_30 (t : ℝ) : (60 - 9 * t - 4.5 * t^2 = 30) → t = 1.77 :=
by
  intro h_eq
  sorry

end volleyball_height_30_l31_31882


namespace zero_of_function_l31_31589

theorem zero_of_function : ∃ x : Real, 4 * x - 2 = 0 ∧ x = 1 / 2 :=
by
  sorry

end zero_of_function_l31_31589


namespace smartphone_demand_inverse_proportional_l31_31365

theorem smartphone_demand_inverse_proportional (k : ℝ) (d d' p p' : ℝ) 
  (h1 : d = 30)
  (h2 : p = 600)
  (h3 : p' = 900)
  (h4 : d * p = k) :
  d' * p' = k → d' = 20 := 
by 
  sorry

end smartphone_demand_inverse_proportional_l31_31365


namespace find_number_l31_31587

-- Let's define the condition
def condition (x : ℝ) : Prop := x * 99999 = 58293485180

-- Statement to be proved
theorem find_number : ∃ x : ℝ, condition x ∧ x = 582.935 := 
by
  sorry

end find_number_l31_31587


namespace bicycles_wheels_l31_31670

theorem bicycles_wheels (b : ℕ) (h1 : 3 * b + 4 * 3 + 7 * 1 = 25) : b = 2 :=
sorry

end bicycles_wheels_l31_31670


namespace find_a4_l31_31827

noncomputable def S : ℕ → ℤ
| 0 => 0
| 1 => -1
| n+1 => 3 * S n + 2^(n+1) - 3

def a : ℕ → ℤ
| 0 => 0
| 1 => -1
| n+1 => 3 * a n + 2^n

theorem find_a4 (h1 : ∀ n ≥ 2, S n = 3 * S (n - 1) + 2^n - 3) (h2 : a 1 = -1) : a 4 = 11 :=
by
  sorry

end find_a4_l31_31827


namespace cute_angle_of_isosceles_cute_triangle_l31_31063

theorem cute_angle_of_isosceles_cute_triangle (A B C : ℝ) 
    (h1 : B = 2 * C)
    (h2 : A + B + C = 180)
    (h3 : A = B ∨ A = C) :
    A = 45 ∨ A = 72 :=
sorry

end cute_angle_of_isosceles_cute_triangle_l31_31063


namespace total_cost_of_cloth_l31_31013

/-- Define the length of the cloth in meters --/
def length_of_cloth : ℝ := 9.25

/-- Define the cost per meter in dollars --/
def cost_per_meter : ℝ := 46

/-- Theorem stating that the total cost is $425.50 given the length and cost per meter --/
theorem total_cost_of_cloth : length_of_cloth * cost_per_meter = 425.50 := by
  sorry

end total_cost_of_cloth_l31_31013


namespace total_paintable_area_l31_31041

-- Define the dimensions of a bedroom
def bedroom_length : ℕ := 10
def bedroom_width : ℕ := 12
def bedroom_height : ℕ := 9

-- Define the non-paintable area per bedroom
def non_paintable_area_per_bedroom : ℕ := 74

-- Number of bedrooms
def number_of_bedrooms : ℕ := 4

-- The total paintable area that we need to prove
theorem total_paintable_area : 
  4 * (2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height) - non_paintable_area_per_bedroom) = 1288 := 
by
  sorry

end total_paintable_area_l31_31041


namespace solve_cubic_equation_l31_31255

theorem solve_cubic_equation :
  ∀ x : ℝ, (x^3 + 2 * (x + 1)^3 + (x + 2)^3 = (x + 4)^3) → x = 3 :=
by
  intro x
  sorry

end solve_cubic_equation_l31_31255


namespace evaluate_expression_l31_31178

theorem evaluate_expression : 
  let a := 2
  let b := 1 / 2
  2 * (a^2 - 2 * a * b) - 3 * (a^2 - a * b - 4 * b^2) = -2 :=
by
  let a := 2
  let b := 1 / 2
  sorry

end evaluate_expression_l31_31178


namespace opposite_of_negative_2023_l31_31575

-- Define the opposite condition
def is_opposite (y x : Int) : Prop := y + x = 0

theorem opposite_of_negative_2023 : ∃ x : Int, is_opposite (-2023) x ∧ x = 2023 :=
by 
  use 2023
  sorry

end opposite_of_negative_2023_l31_31575


namespace time_worked_together_l31_31483

noncomputable def combined_rate (P_rate Q_rate : ℝ) : ℝ :=
  P_rate + Q_rate

theorem time_worked_together (P_rate Q_rate : ℝ) (t additional_time job_completed : ℝ) :
  P_rate = 1 / 4 ∧ Q_rate = 1 / 15 ∧ additional_time = 1 / 5 ∧ job_completed = (additional_time * P_rate) →
  (t * combined_rate P_rate Q_rate + job_completed = 1) → 
  t = 3 :=
sorry

end time_worked_together_l31_31483


namespace max_value_of_f_l31_31374

noncomputable def f (t : ℝ) : ℝ := ((2^(t+1) - 4*t) * t) / (16^t)

theorem max_value_of_f : ∃ t : ℝ, ∀ u : ℝ, f u ≤ f t ∧ f t = 1 / 16 := by
  sorry

end max_value_of_f_l31_31374


namespace sheena_weeks_to_complete_dresses_l31_31388

/- Sheena is sewing the bridesmaid's dresses for her sister's wedding.
There are 7 bridesmaids in the wedding.
Each bridesmaid's dress takes a different number of hours to sew due to different styles and sizes.
The hours needed to sew the bridesmaid's dresses are as follows: 15 hours, 18 hours, 20 hours, 22 hours, 24 hours, 26 hours, and 28 hours.
If Sheena sews the dresses 5 hours each week, prove that it will take her 31 weeks to complete all the dresses. -/

def bridesmaid_hours : List ℕ := [15, 18, 20, 22, 24, 26, 28]

def total_hours_needed (hours : List ℕ) : ℕ :=
  hours.sum

def weeks_needed (total_hours : ℕ) (hours_per_week : ℕ) : ℕ :=
  (total_hours + hours_per_week - 1) / hours_per_week

theorem sheena_weeks_to_complete_dresses :
  weeks_needed (total_hours_needed bridesmaid_hours) 5 = 31 := by
  sorry

end sheena_weeks_to_complete_dresses_l31_31388


namespace find_2a_plus_b_l31_31545

noncomputable def f (a b x : ℝ) : ℝ := a * x - b
noncomputable def g (x : ℝ) : ℝ := -4 * x + 6
noncomputable def h (a b x : ℝ) : ℝ := f a b (g x)
noncomputable def h_inv (x : ℝ) : ℝ := x + 9

theorem find_2a_plus_b (a b : ℝ) (h_inv_eq: ∀ x : ℝ, h a b (h_inv x) = x) : 2 * a + b = 7 :=
sorry

end find_2a_plus_b_l31_31545


namespace necessary_and_sufficient_condition_l31_31665

open Set

noncomputable def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1 ^ 2}

noncomputable def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 ^ 2 + (p.2 - a) ^ 2 ≤ 1}

theorem necessary_and_sufficient_condition (a : ℝ) :
  N a ⊆ M ↔ a ≥ 5 / 4 := sorry

end necessary_and_sufficient_condition_l31_31665


namespace wheat_acres_l31_31429

def cultivate_crops (x y : ℕ) : Prop :=
  (42 * x + 30 * y = 18600) ∧ (x + y = 500) 

theorem wheat_acres : ∃ y, ∃ x, 
  cultivate_crops x y ∧ y = 200 :=
by {sorry}

end wheat_acres_l31_31429


namespace num_ordered_pairs_l31_31974

theorem num_ordered_pairs (M N : ℕ) (hM : M > 0) (hN : N > 0) :
  (M * N = 32) → ∃ (k : ℕ), k = 6 :=
by
  sorry

end num_ordered_pairs_l31_31974


namespace concentration_replacement_l31_31891

theorem concentration_replacement 
  (initial_concentration : ℝ)
  (new_concentration : ℝ)
  (fraction_replaced : ℝ)
  (replacing_concentration : ℝ)
  (h1 : initial_concentration = 0.45)
  (h2 : new_concentration = 0.35)
  (h3 : fraction_replaced = 0.5) :
  replacing_concentration = 0.25 := by
  sorry

end concentration_replacement_l31_31891


namespace estimate_probability_concave_l31_31099

noncomputable def times_thrown : ℕ := 1000
noncomputable def frequency_convex : ℝ := 0.44

theorem estimate_probability_concave :
  (1 - frequency_convex) = 0.56 := by
  sorry

end estimate_probability_concave_l31_31099


namespace probability_four_squares_form_square_l31_31932

noncomputable def probability_form_square (n k : ℕ) :=
  if (k = 4) ∧ (n = 6) then (1 / 561 : ℚ) else 0

theorem probability_four_squares_form_square :
  probability_form_square 6 4 = (1 / 561 : ℚ) :=
by
  -- Here we would usually include the detailed proof
  -- corresponding to the solution steps from the problem,
  -- but we leave it as sorry for now.
  sorry

end probability_four_squares_form_square_l31_31932


namespace find_ages_l31_31019

theorem find_ages (P J G : ℕ)
  (h1 : P - 10 = 1 / 3 * (J - 10))
  (h2 : J = P + 12)
  (h3 : G = 1 / 2 * (P + J)) :
  P = 16 ∧ G = 22 :=
by
  sorry

end find_ages_l31_31019


namespace final_solution_sugar_percentage_l31_31608

-- Define the conditions of the problem
def initial_solution_sugar_percentage : ℝ := 0.10
def replacement_fraction : ℝ := 0.25
def second_solution_sugar_percentage : ℝ := 0.26

-- Define the Lean statement that proves the final sugar percentage
theorem final_solution_sugar_percentage:
  (0.10 * (1 - 0.25) + 0.26 * 0.25) * 100 = 14 :=
by
  sorry

end final_solution_sugar_percentage_l31_31608


namespace trenton_earning_goal_l31_31466

-- Parameters
def fixed_weekly_earnings : ℝ := 190
def commission_rate : ℝ := 0.04
def sales_amount : ℝ := 7750
def goal : ℝ := 500

-- Proof statement
theorem trenton_earning_goal :
  fixed_weekly_earnings + (commission_rate * sales_amount) = goal :=
by
  sorry

end trenton_earning_goal_l31_31466


namespace binomial_expansion_value_l31_31819

theorem binomial_expansion_value : 
  105^3 - 3 * 105^2 + 3 * 105 - 1 = 1124864 := by
  sorry

end binomial_expansion_value_l31_31819


namespace sequence_general_formula_l31_31711

theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (hS : ∀ n, S n = 3 / 2 * a n - 3) : 
  (∀ n, a n = 2 * 3 ^ n) :=
by 
  sorry

end sequence_general_formula_l31_31711


namespace find_actual_marks_l31_31910

theorem find_actual_marks (wrong_marks : ℕ) (avg_increase : ℕ) (num_pupils : ℕ) (h_wrong_marks: wrong_marks = 73) (h_avg_increase : avg_increase = 1/2) (h_num_pupils : num_pupils = 16) : 
  ∃ (actual_marks : ℕ), actual_marks = 65 :=
by
  have total_increase := num_pupils * avg_increase
  have eqn := wrong_marks - total_increase
  use eqn
  sorry

end find_actual_marks_l31_31910


namespace gcd_of_product_diff_is_12_l31_31185

theorem gcd_of_product_diff_is_12
  (a b c d : ℤ) : ∃ (D : ℤ), D = 12 ∧
  ∀ (a b c d : ℤ), D ∣ (b - a) * (c - b) * (d - c) * (d - a) * (c - a) * (d - b) :=
by
  use 12
  sorry

end gcd_of_product_diff_is_12_l31_31185


namespace solution_for_equation_l31_31593

theorem solution_for_equation (m n : ℕ) (h : 0 < m ∧ 0 < n ∧ 2 * m^2 = 3 * n^3) :
  ∃ k : ℕ, 0 < k ∧ m = 18 * k^3 ∧ n = 6 * k^2 :=
by sorry

end solution_for_equation_l31_31593


namespace smallest_sum_l31_31834

theorem smallest_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_neq : x ≠ y) 
  (h_fraction : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) : x + y = 64 :=
sorry

end smallest_sum_l31_31834


namespace point_in_third_quadrant_coordinates_l31_31548

theorem point_in_third_quadrant_coordinates :
  ∀ (P : ℝ × ℝ), (P.1 < 0) ∧ (P.2 < 0) ∧ (|P.2| = 2) ∧ (|P.1| = 3) -> P = (-3, -2) :=
by
  intros P h
  sorry

end point_in_third_quadrant_coordinates_l31_31548


namespace f_value_at_2_9_l31_31329

-- Define the function f with its properties as conditions
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the domain of f
axiom f_domain : ∀ x, 0 ≤ x ∧ x ≤ 1

-- Condition (i)
axiom f_0_eq : f 0 = 0

-- Condition (ii)
axiom f_monotone : ∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x ≤ f y

-- Condition (iii)
axiom f_symmetry : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (1 - x) = 3/4 - f x / 2

-- Condition (iv)
axiom f_scale : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x / 3) = f x / 3

-- Proof goal
theorem f_value_at_2_9 : f (2/9) = 5/24 := by
  sorry

end f_value_at_2_9_l31_31329


namespace union_of_sets_l31_31494

def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B : Set ℝ := {x | x > 0}
def C : Set ℝ := {x | x > -2}

theorem union_of_sets (A B : Set ℝ) : (A ∪ B) = C :=
  sorry

end union_of_sets_l31_31494


namespace lines_perpendicular_l31_31188

theorem lines_perpendicular
  (k₁ k₂ : ℝ)
  (h₁ : k₁^2 - 3*k₁ - 1 = 0)
  (h₂ : k₂^2 - 3*k₂ - 1 = 0) :
  k₁ * k₂ = -1 → 
  (∃ l₁ l₂: ℝ → ℝ, 
    ∀ x, l₁ x = k₁ * x ∧ l₂ x = k₂ * x → 
    ∃ m, m = -1) := 
sorry

end lines_perpendicular_l31_31188


namespace find_m_l31_31791

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 9

theorem find_m (m : ℝ) : f 5 - g 5 m = 20 → m = -16.8 :=
by
  -- Given f(x) and g(x, m) definitions, we want to prove m = -16.8 given f 5 - g 5 m = 20.
  sorry

end find_m_l31_31791


namespace equal_frac_implies_x_zero_l31_31627

theorem equal_frac_implies_x_zero (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
sorry

end equal_frac_implies_x_zero_l31_31627


namespace min_value_reciprocals_l31_31895

variable {a b : ℝ}

theorem min_value_reciprocals (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) :
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 2 → 
  (1/a + 1/b) ≥ 2) :=
sorry

end min_value_reciprocals_l31_31895


namespace real_y_values_for_given_x_l31_31209

theorem real_y_values_for_given_x (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 4 = 0) ↔ (x ≤ -2 / 3 ∨ x ≥ 4) :=
by
  sorry

end real_y_values_for_given_x_l31_31209


namespace total_sections_l31_31226

theorem total_sections (boys girls max_students per_section boys_ratio girls_ratio : ℕ)
  (hb : boys = 408) (hg : girls = 240) (hm : max_students = 24) 
  (br : boys_ratio = 3) (gr : girls_ratio = 2)
  (hboy_sec : (boys + max_students - 1) / max_students = 17)
  (hgirl_sec : (girls + max_students - 1) / max_students = 10) 
  : (3 * (((boys + max_students - 1) / max_students) + 2 * ((girls + max_students - 1) / max_students))) / 5 = 30 :=
by
  sorry

end total_sections_l31_31226


namespace inequality_proof_l31_31951

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b < 2) : 
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧ 
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ 0 < a ∧ a = b ∧ a < 1) := 
by 
  sorry

end inequality_proof_l31_31951


namespace Jasmine_initial_percentage_is_5_l31_31930

noncomputable def initial_percentage_of_jasmine 
  (V_initial : ℕ := 90) 
  (V_added_jasmine : ℕ := 8) 
  (V_added_water : ℕ := 2) 
  (V_final : ℕ := 100) 
  (P_final : ℚ := 12.5 / 100) : ℚ := 
  (P_final * V_final - V_added_jasmine) / V_initial * 100

theorem Jasmine_initial_percentage_is_5 :
  initial_percentage_of_jasmine = 5 := 
by 
  sorry

end Jasmine_initial_percentage_is_5_l31_31930


namespace weight_of_one_baseball_l31_31885

structure Context :=
  (numberBaseballs : ℕ)
  (numberBicycles : ℕ)
  (weightBicycles : ℕ)
  (weightTotalBicycles : ℕ)

def problem (ctx : Context) :=
  ctx.weightTotalBicycles = ctx.numberBicycles * ctx.weightBicycles ∧
  ctx.numberBaseballs * ctx.weightBicycles = ctx.weightTotalBicycles →
  (ctx.weightTotalBicycles / ctx.numberBaseballs) = 8

theorem weight_of_one_baseball (ctx : Context) : problem ctx :=
sorry

end weight_of_one_baseball_l31_31885


namespace total_pages_read_l31_31560

-- Definitions of the conditions
def pages_read_by_jairus : ℕ := 20

def pages_read_by_arniel : ℕ := 2 + 2 * pages_read_by_jairus

-- The statement to prove the total number of pages read by both is 62
theorem total_pages_read : pages_read_by_jairus + pages_read_by_arniel = 62 := by
  sorry

end total_pages_read_l31_31560


namespace find_V_y_l31_31124

-- Define the volumes and percentages given in the problem
def V_x : ℕ := 300
def percent_x : ℝ := 0.10
def percent_y : ℝ := 0.30
def desired_percent : ℝ := 0.22

-- Define the alcohol volumes in the respective solutions
def alcohol_x := percent_x * V_x
def total_volume (V_y : ℕ) := V_x + V_y
def desired_alcohol (V_y : ℕ) := desired_percent * (total_volume V_y)

-- Define our main statement
theorem find_V_y : ∃ (V_y : ℕ), alcohol_x + (percent_y * V_y) = desired_alcohol V_y ∧ V_y = 450 :=
by
  sorry

end find_V_y_l31_31124


namespace length_imaginary_axis_hyperbola_l31_31080

theorem length_imaginary_axis_hyperbola : 
  ∀ (a b : ℝ), (a = 2) → (b = 1) → 
  (∀ x y : ℝ, (y^2 / a^2 - x^2 = 1) → 2 * b = 2) :=
by intros a b ha hb x y h; sorry

end length_imaginary_axis_hyperbola_l31_31080


namespace base_conversion_l31_31838

theorem base_conversion (k : ℕ) (h : 26 = 3*k + 2) : k = 8 := 
by 
  sorry

end base_conversion_l31_31838


namespace square_perimeter_equals_66_88_l31_31638

noncomputable def circle_perimeter : ℝ := 52.5

noncomputable def circle_radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def circle_diameter (r : ℝ) : ℝ := 2 * r

noncomputable def square_side_length (d : ℝ) : ℝ := d

noncomputable def square_perimeter (s : ℝ) : ℝ := 4 * s

theorem square_perimeter_equals_66_88 :
  square_perimeter (square_side_length (circle_diameter (circle_radius circle_perimeter))) = 66.88 := 
by
  -- Placeholder for the proof
  sorry

end square_perimeter_equals_66_88_l31_31638


namespace samantha_trip_l31_31783

theorem samantha_trip (a b c d x : ℕ)
  (h1 : 1 ≤ a) (h2 : a + b + c + d ≤ 10) 
  (h3 : 1000 * d + 100 * c + 10 * b + a - (1000 * a + 100 * b + 10 * c + d) = 60 * x)
  : a^2 + b^2 + c^2 + d^2 = 83 :=
sorry

end samantha_trip_l31_31783


namespace decreasing_interval_f_l31_31710

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (4*x - x^2)

theorem decreasing_interval_f : ∀ x, (2 < x) ∧ (x < 4) → f x < f (2 : ℝ) :=
by
sorry

end decreasing_interval_f_l31_31710


namespace smallest_n_7n_eq_n7_mod_3_l31_31554

theorem smallest_n_7n_eq_n7_mod_3 : ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 3]) ∧ ∀ m : ℕ, m > 0 → (7^m ≡ m^7 [MOD 3] → m ≥ n) :=
by
  sorry

end smallest_n_7n_eq_n7_mod_3_l31_31554


namespace find_side_b_l31_31343

variable {a b c : ℝ} -- sides of the triangle
variable {A B C : ℝ} -- angles of the triangle
variable {area : ℝ}

axiom sides_form_arithmetic_sequence : 2 * b = a + c
axiom angle_B_is_60_degrees : B = Real.pi / 3
axiom area_is_3sqrt3 : area = 3 * Real.sqrt 3
axiom area_formula : area = 1 / 2 * a * c * Real.sin (B)

theorem find_side_b : b = 2 * Real.sqrt 3 := by
  sorry

end find_side_b_l31_31343


namespace Martha_blocks_end_l31_31407

variable (Ronald_blocks : ℕ) (Martha_start_blocks : ℕ) (Martha_found_blocks : ℕ)
variable (Ronald_has_blocks : Ronald_blocks = 13)
variable (Martha_has_start_blocks : Martha_start_blocks = 4)
variable (Martha_finds_more_blocks : Martha_found_blocks = 80)

theorem Martha_blocks_end : Martha_start_blocks + Martha_found_blocks = 84 :=
by
  have Martha_start_blocks := Martha_has_start_blocks
  have Martha_found_blocks := Martha_finds_more_blocks
  sorry

end Martha_blocks_end_l31_31407


namespace isosceles_triangle_angle_measure_l31_31199

theorem isosceles_triangle_angle_measure:
  ∀ (α β : ℝ), (α = 112.5) → (2 * β + α = 180) → β = 33.75 :=
by
  intros α β hα h_sum
  sorry

end isosceles_triangle_angle_measure_l31_31199


namespace minimum_value_of_y_l31_31078

theorem minimum_value_of_y : ∀ x : ℝ, ∃ y : ℝ, (y = 3 * x^2 + 6 * x + 9) → y ≥ 6 :=
by
  intro x
  use (3 * (x + 1)^2 + 6)
  intro h
  sorry

end minimum_value_of_y_l31_31078


namespace domain_of_f_l31_31707

noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 6 * x + 10⌋

theorem domain_of_f : {x : ℝ | ∀ y, f y ≠ 0 → x ≠ 3} = {x : ℝ | x < 3 ∨ x > 3} :=
by
  sorry

end domain_of_f_l31_31707


namespace find_number_l31_31161

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 10) : x = 5 :=
by
  sorry

end find_number_l31_31161


namespace Tom_earns_per_week_l31_31375

-- Definitions based on conditions
def crab_buckets_per_day := 8
def crabs_per_bucket := 12
def price_per_crab := 5
def days_per_week := 7

-- The proof goal
theorem Tom_earns_per_week :
  (crab_buckets_per_day * crabs_per_bucket * price_per_crab * days_per_week) = 3360 := by
  sorry

end Tom_earns_per_week_l31_31375


namespace ceil_sqrt_200_eq_15_l31_31917

theorem ceil_sqrt_200_eq_15 : Int.ceil (Real.sqrt 200) = 15 := by
  sorry

end ceil_sqrt_200_eq_15_l31_31917


namespace smallest_fraction_l31_31273

theorem smallest_fraction {a b c d e : ℚ}
  (ha : a = 7/15)
  (hb : b = 5/11)
  (hc : c = 16/33)
  (hd : d = 49/101)
  (he : e = 89/183) :
  (b < a) ∧ (b < c) ∧ (b < d) ∧ (b < e) := 
sorry

end smallest_fraction_l31_31273


namespace rationalize_denominator_l31_31504

theorem rationalize_denominator :
  (7 / (Real.sqrt 175 - Real.sqrt 75)) = (7 * (Real.sqrt 7 + Real.sqrt 3) / 20) :=
by
  have h1 : Real.sqrt 175 = 5 * Real.sqrt 7 := sorry
  have h2 : Real.sqrt 75 = 5 * Real.sqrt 3 := sorry
  sorry

end rationalize_denominator_l31_31504


namespace find_m_for_line_passing_through_circle_center_l31_31073

theorem find_m_for_line_passing_through_circle_center :
  ∀ (m : ℝ), (∀ (x y : ℝ), 2 * x + y + m = 0 ↔ (x - 1)^2 + (y + 2)^2 = 5) → m = 0 :=
by
  intro m
  intro h
  -- Here we construct that the center (1, -2) must lie on the line 2x + y + m = 0
  -- using the given condition of the circle center.
  have center := h 1 (-2)
  -- solving for the equation at the point (1, -2) must yield m = 0
  sorry

end find_m_for_line_passing_through_circle_center_l31_31073


namespace total_right_handed_players_l31_31418

-- Defining the conditions and the given values
def total_players : ℕ := 61
def throwers : ℕ := 37
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers

-- The proof goal
theorem total_right_handed_players 
  (h1 : total_players = 61)
  (h2 : throwers = 37)
  (h3 : non_throwers = total_players - throwers)
  (h4 : left_handed_non_throwers = non_throwers / 3)
  (h5 : right_handed_non_throwers = non_throwers - left_handed_non_throwers)
  (h6 : left_handed_non_throwers * 3 = non_throwers)
  : throwers + right_handed_non_throwers = 53 :=
sorry

end total_right_handed_players_l31_31418


namespace malcolm_followers_l31_31389

theorem malcolm_followers :
  let instagram_followers := 240
  let facebook_followers := 500
  let twitter_followers := (instagram_followers + facebook_followers) / 2
  let tiktok_followers := 3 * twitter_followers
  let youtube_followers := tiktok_followers + 510
  instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers = 3840 :=
by {
  sorry
}

end malcolm_followers_l31_31389


namespace unique_fish_total_l31_31240

-- Define the conditions as stated in the problem
def Micah_fish : ℕ := 7
def Kenneth_fish : ℕ := 3 * Micah_fish
def Matthias_fish : ℕ := Kenneth_fish - 15
def combined_fish : ℕ := Micah_fish + Kenneth_fish + Matthias_fish
def Gabrielle_fish : ℕ := 2 * combined_fish

def shared_fish_Micah_Matthias : ℕ := 4
def shared_fish_Kenneth_Gabrielle : ℕ := 6

-- Define the total unique fish computation
def total_unique_fish : ℕ := (Micah_fish + Kenneth_fish + Matthias_fish + Gabrielle_fish) - (shared_fish_Micah_Matthias + shared_fish_Kenneth_Gabrielle)

-- State the theorem
theorem unique_fish_total : total_unique_fish = 92 := by
  -- Proof omitted
  sorry

end unique_fish_total_l31_31240


namespace neg_p_implies_neg_q_l31_31459

variable {x : ℝ}

def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

theorem neg_p_implies_neg_q (h : ¬ p x) : ¬ q x :=
sorry

end neg_p_implies_neg_q_l31_31459


namespace find_sin_θ_find_cos_2θ_find_cos_φ_l31_31215

noncomputable def θ : ℝ := sorry
noncomputable def φ : ℝ := sorry

-- Conditions
axiom cos_eq : Real.cos θ = Real.sqrt 5 / 5
axiom θ_in_quadrant_I : 0 < θ ∧ θ < Real.pi / 2
axiom sin_diff_eq : Real.sin (θ - φ) = Real.sqrt 10 / 10
axiom φ_in_quadrant_I : 0 < φ ∧ φ < Real.pi / 2

-- Goals
-- Part (I) Prove the value of sin θ
theorem find_sin_θ : Real.sin θ = 2 * Real.sqrt 5 / 5 :=
by
  sorry

-- Part (II) Prove the value of cos 2θ
theorem find_cos_2θ : Real.cos (2 * θ) = -3 / 5 :=
by
  sorry

-- Part (III) Prove the value of cos φ
theorem find_cos_φ : Real.cos φ = Real.sqrt 2 / 2 :=
by
  sorry

end find_sin_θ_find_cos_2θ_find_cos_φ_l31_31215


namespace abs_diff_mn_sqrt_eight_l31_31894

theorem abs_diff_mn_sqrt_eight {m n p : ℝ} (h1 : m * n = 6) (h2 : m + n + p = 7) (h3 : p = 1) :
  |m - n| = 2 * Real.sqrt 3 :=
by
  sorry

end abs_diff_mn_sqrt_eight_l31_31894


namespace arithmetic_proof_l31_31008

theorem arithmetic_proof : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end arithmetic_proof_l31_31008


namespace fraction_of_total_cost_for_raisins_l31_31996

-- Define variables and constants
variable (R : ℝ) -- cost of a pound of raisins

-- Define the conditions as assumptions
variable (cost_of_nuts : ℝ := 4 * R)
variable (cost_of_dried_berries : ℝ := 2 * R)

variable (total_cost : ℝ := 3 * R + 4 * cost_of_nuts + 2 * cost_of_dried_berries)
variable (cost_of_raisins : ℝ := 3 * R)

-- Main statement that we want to prove
theorem fraction_of_total_cost_for_raisins :
  cost_of_raisins / total_cost = 3 / 23 := by
  sorry

end fraction_of_total_cost_for_raisins_l31_31996


namespace second_remainder_l31_31106

theorem second_remainder (n : ℕ) : n = 210 ∧ n % 13 = 3 → n % 17 = 6 :=
by
  sorry

end second_remainder_l31_31106


namespace quadratic_roots_condition_l31_31059

theorem quadratic_roots_condition (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 0) :
  ¬ ((∃ x y : ℝ, ax^2 + 2*x + 1 = 0 ∧ ax^2 + 2*y + 1 = 0 ∧ x*y < 0) ↔
     (a > 0 ∧ a ≠ 0)) :=
by
  sorry

end quadratic_roots_condition_l31_31059


namespace als_initial_portion_l31_31122

theorem als_initial_portion (a b c : ℝ)
  (h1 : a + b + c = 1200)
  (h2 : a - 150 + 3 * b + 3 * c = 1800) :
  a = 825 :=
sorry

end als_initial_portion_l31_31122


namespace halfway_fraction_eq_l31_31344

-- Define the fractions
def one_seventh := 1 / 7
def one_fourth := 1 / 4

-- Define the common denominators
def common_denom_1 := 4 / 28
def common_denom_2 := 7 / 28

-- Define the addition of the common denominators
def addition := common_denom_1 + common_denom_2

-- Define the average of the fractions
noncomputable def average := addition / 2

-- State the theorem
theorem halfway_fraction_eq : average = 11 / 56 :=
by
  -- Provide the steps which will be skipped here
  sorry

end halfway_fraction_eq_l31_31344


namespace candy_difference_l31_31933

theorem candy_difference (Frankie_candies Max_candies : ℕ) (hF : Frankie_candies = 74) (hM : Max_candies = 92) :
  Max_candies - Frankie_candies = 18 :=
by
  sorry

end candy_difference_l31_31933


namespace problem_solution_l31_31475

theorem problem_solution (a b c d e f g : ℝ) 
  (h1 : a + b + e = 7)
  (h2 : b + c + f = 10)
  (h3 : c + d + g = 6)
  (h4 : e + f + g = 9) : 
  a + d + g = 6 := 
sorry

end problem_solution_l31_31475


namespace no_arith_geo_progression_S1_S2_S3_l31_31603

noncomputable def S_1 (A B C : Point) : ℝ := sorry -- area of triangle ABC
noncomputable def S_2 (A B E : Point) : ℝ := sorry -- area of triangle ABE
noncomputable def S_3 (A B D : Point) : ℝ := sorry -- area of triangle ABD

def bisecting_plane (A B D C E : Point) : Prop := sorry -- plane bisects dihedral angle at AB

theorem no_arith_geo_progression_S1_S2_S3 (A B C D E : Point) 
(h_bisect : bisecting_plane A B D C E) :
¬ (∃ (S1 S2 S3 : ℝ), S1 = S_1 A B C ∧ S2 = S_2 A B E ∧ S3 = S_3 A B D ∧ 
  (S2 = (S1 + S3) / 2 ∨ S2^2 = S1 * S3 )) :=
sorry

end no_arith_geo_progression_S1_S2_S3_l31_31603


namespace barbara_candies_l31_31632

theorem barbara_candies : (9 + 18) = 27 :=
by
  sorry

end barbara_candies_l31_31632


namespace kitchen_supplies_sharon_wants_l31_31328

theorem kitchen_supplies_sharon_wants (P : ℕ) (plates_angela cutlery_angela pots_sharon plates_sharon cutlery_sharon : ℕ) 
  (h1 : plates_angela = 3 * P + 6) 
  (h2 : cutlery_angela = (3 * P + 6) / 2) 
  (h3 : pots_sharon = P / 2) 
  (h4 : plates_sharon = 3 * (3 * P + 6) - 20) 
  (h5 : cutlery_sharon = 2 * (3 * P + 6) / 2) 
  (h_total : pots_sharon + plates_sharon + cutlery_sharon = 254) : 
  P = 20 :=
sorry

end kitchen_supplies_sharon_wants_l31_31328


namespace tan_105_eq_neg2_sub_sqrt3_l31_31674

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l31_31674


namespace expression_equals_two_l31_31898

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem expression_equals_two : simplify_expression = 2 :=
by
  sorry

end expression_equals_two_l31_31898


namespace range_of_m_l31_31566

theorem range_of_m 
    (m : ℝ) (x : ℝ)
    (p : x^2 - 8 * x - 20 > 0)
    (q : (x - (1 - m)) * (x - (1 + m)) > 0)
    (h : ∀ x, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) :
    0 < m ∧ m ≤ 3 := by
  sorry

end range_of_m_l31_31566


namespace min_value_S_max_value_m_l31_31438

noncomputable def S (x : ℝ) : ℝ := abs (x - 2) + abs (x - 4)

theorem min_value_S : ∃ x, S x = 2 ∧ ∀ x, S x ≥ 2 := by
  sorry

theorem max_value_m : ∀ x y, S x ≥ m * (-y^2 + 2*y) → 0 ≤ m ∧ m ≤ 2 := by
  sorry

end min_value_S_max_value_m_l31_31438


namespace number_that_multiplies_x_l31_31268

variables (n x y : ℝ)

theorem number_that_multiplies_x :
  n * x = 3 * y → 
  x * y ≠ 0 → 
  (1 / 5 * x) / (1 / 6 * y) = 0.72 →
  n = 5 :=
by
  intros h1 h2 h3
  sorry

end number_that_multiplies_x_l31_31268


namespace fourth_root_cubed_eq_729_l31_31134

theorem fourth_root_cubed_eq_729 (x : ℝ) (hx : (x^(1/4))^3 = 729) : x = 6561 :=
  sorry

end fourth_root_cubed_eq_729_l31_31134


namespace least_number_of_equal_cubes_l31_31592

def cuboid_dimensions := (18, 27, 36)
def ratio := (1, 2, 3)

theorem least_number_of_equal_cubes :
  ∃ n, n = 648 ∧
  ∃ a b c : ℕ,
    (a, b, c) = (3, 6, 9) ∧
    (18 % a = 0 ∧ 27 % b = 0 ∧ 36 % c = 0) ∧
    18 * 27 * 36 = n * (a * b * c) :=
sorry

end least_number_of_equal_cubes_l31_31592


namespace train_pass_time_l31_31729

-- Definitions based on conditions
def train_length : Float := 250
def pole_time : Float := 10
def platform_length : Float := 1250
def incline_angle : Float := 5 -- degrees
def speed_reduction_factor : Float := 0.75

-- The statement to be proved
theorem train_pass_time :
  let original_speed := train_length / pole_time
  let incline_speed := original_speed * speed_reduction_factor
  let total_distance := train_length + platform_length
  let time_to_pass_platform := total_distance / incline_speed
  time_to_pass_platform = 80 := by
  simp [train_length, pole_time, platform_length, incline_angle, speed_reduction_factor]
  sorry

end train_pass_time_l31_31729


namespace correct_number_of_sequences_l31_31968

noncomputable def athlete_sequences : Nat :=
  let total_permutations := 24
  let A_first_leg := 6
  let B_fourth_leg := 6
  let A_first_and_B_fourth := 2
  total_permutations - (A_first_leg + B_fourth_leg - A_first_and_B_fourth)

theorem correct_number_of_sequences : athlete_sequences = 14 := by
  sorry

end correct_number_of_sequences_l31_31968


namespace pages_left_to_read_correct_l31_31137

def total_pages : Nat := 563
def pages_read : Nat := 147
def pages_left_to_read : Nat := 416

theorem pages_left_to_read_correct : total_pages - pages_read = pages_left_to_read := by
  sorry

end pages_left_to_read_correct_l31_31137


namespace triangle_min_area_l31_31734

theorem triangle_min_area :
  ∃ (p q : ℤ), (p, q).fst = 3 ∧ (p, q).snd = 3 ∧ 1/2 * |18 * p - 30 * q| = 3 := 
sorry

end triangle_min_area_l31_31734


namespace extinction_probability_l31_31158

-- Definitions from conditions
def prob_divide : ℝ := 0.6
def prob_die : ℝ := 0.4

-- Statement of the theorem
theorem extinction_probability :
  ∃ (v : ℝ), v = 2 / 3 :=
by
  sorry

end extinction_probability_l31_31158


namespace hotel_charge_difference_l31_31923

variables (G P R : ℝ)

-- Assumptions based on the problem conditions
variables
  (hR : R = 2 * G) -- Charge for a single room at hotel R is 100% greater than at hotel G
  (hP : P = 0.9 * G) -- Charge for a single room at hotel P is 10% less than at hotel G

theorem hotel_charge_difference :
  ((R - P) / R) * 100 = 55 :=
by
  -- Calculation
  sorry

end hotel_charge_difference_l31_31923


namespace incorrect_statement_A_l31_31259

-- Definitions based on conditions
def equilibrium_shifts (condition: Type) : Prop := sorry
def value_K_changes (condition: Type) : Prop := sorry

-- The incorrect statement definition
def statement_A (condition: Type) : Prop := equilibrium_shifts condition → value_K_changes condition

-- The final theorem stating that 'statement_A' is incorrect
theorem incorrect_statement_A (condition: Type) : ¬ statement_A condition :=
sorry

end incorrect_statement_A_l31_31259


namespace Jiyeol_average_score_l31_31927

theorem Jiyeol_average_score (K M E : ℝ)
  (h1 : (K + M) / 2 = 26.5)
  (h2 : (M + E) / 2 = 34.5)
  (h3 : (K + E) / 2 = 29) :
  (K + M + E) / 3 = 30 := 
sorry

end Jiyeol_average_score_l31_31927


namespace no_real_roots_range_l31_31612

theorem no_real_roots_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_range_l31_31612


namespace quadratic_solution_l31_31623

theorem quadratic_solution (a : ℝ) (h : 2^2 - 3 * 2 + a = 0) : 2 * a - 1 = 3 :=
by {
  sorry
}

end quadratic_solution_l31_31623


namespace infinitely_many_n_divisible_by_n_squared_l31_31390

theorem infinitely_many_n_divisible_by_n_squared :
  ∃ (n : ℕ → ℕ), (∀ k : ℕ, 0 < n k) ∧ (∀ k : ℕ, n k^2 ∣ 2^(n k) + 3^(n k)) :=
sorry

end infinitely_many_n_divisible_by_n_squared_l31_31390


namespace commensurable_iff_rat_l31_31177

def commensurable (A B : ℝ) : Prop :=
  ∃ d : ℝ, ∃ m n : ℤ, A = m * d ∧ B = n * d

theorem commensurable_iff_rat (A B : ℝ) :
  commensurable A B ↔ ∃ (m n : ℤ) (h : n ≠ 0), A / B = m / n :=
by
  sorry

end commensurable_iff_rat_l31_31177


namespace player_A_wins_iff_n_is_odd_l31_31090

-- Definitions of the problem conditions
structure ChessboardGame (n : ℕ) :=
  (stones : ℕ := 99)
  (playerA_first : Prop := true)
  (turns : ℕ := n * 99)

-- Statement of the problem
theorem player_A_wins_iff_n_is_odd (n : ℕ) (g : ChessboardGame n) : 
  PlayerA_has_winning_strategy ↔ n % 2 = 1 := 
sorry

end player_A_wins_iff_n_is_odd_l31_31090


namespace symmetric_probability_l31_31719

-- Definitions based on the problem conditions
def total_points : ℕ := 121
def central_point : ℕ × ℕ := (6, 6)
def remaining_points : ℕ := total_points - 1
def symmetric_points : ℕ := 40

-- Predicate for the probability that line PQ is a line of symmetry
def is_symmetrical_line (p q : (ℕ × ℕ)) : Prop := 
  (q.fst = 11 - p.fst ∧ q.snd = p.snd) ∨
  (q.fst = p.fst ∧ q.snd = 11 - p.snd) ∨
  (q.fst + q.snd = 12) ∨ 
  (q.fst - q.snd = 0)

-- The theorem stating the probability is 1/3
theorem symmetric_probability :
  ∃ (total_points : ℕ) (remaining_points : ℕ) (symmetric_points : ℕ),
    total_points = 121 ∧
    remaining_points = total_points - 1 ∧
    symmetric_points = 40 ∧
    (symmetric_points : ℚ) / (remaining_points : ℚ) = 1 / 3 :=
by
  sorry

end symmetric_probability_l31_31719


namespace birds_never_gather_44_l31_31766

theorem birds_never_gather_44 :
    ∀ (position : Fin 44 → Nat), 
    (∀ (i : Fin 44), position i ≤ 44) →
    (∀ (i j : Fin 44), position i ≠ position j) →
    ∃ (S : Nat), S % 4 = 2 →
    ∀ (moves : (Fin 44 → Fin 44) → (Fin 44 → Fin 44)),
    ¬(∃ (tree : Nat), ∀ (i : Fin 44), position i = tree) := 
sorry

end birds_never_gather_44_l31_31766


namespace lucky_sum_probability_eq_l31_31009

/--
Given that there are N balls numbered from 1 to N,
where 10 balls are selected in the main draw with their sum being 63,
and 8 balls are selected in the additional draw with their sum being 44,
we need to prove that N = 18 such that the events are equally likely.
-/
theorem lucky_sum_probability_eq (N : ℕ) (h1 : ∃ (S : Finset ℕ), S.card = 10 ∧ S.sum id = 63) 
    (h2 : ∃ (T : Finset ℕ), T.card = 8 ∧ T.sum id = 44) : N = 18 :=
sorry

end lucky_sum_probability_eq_l31_31009


namespace probability_of_region_F_l31_31853

theorem probability_of_region_F
  (pD pE pG pF : ℚ)
  (hD : pD = 3/8)
  (hE : pE = 1/4)
  (hG : pG = 1/8)
  (hSum : pD + pE + pF + pG = 1) : pF = 1/4 :=
by
  -- we can perform the steps as mentioned in the solution without actually executing them
  sorry

end probability_of_region_F_l31_31853


namespace twice_as_many_juniors_as_seniors_l31_31415

theorem twice_as_many_juniors_as_seniors (j s : ℕ) (h : (1/3 : ℝ) * j = (2/3 : ℝ) * s) : j = 2 * s :=
by
  --proof steps here
  sorry

end twice_as_many_juniors_as_seniors_l31_31415


namespace fraction_red_knights_magical_l31_31572

theorem fraction_red_knights_magical (total_knights : ℕ) (fraction_red fraction_magical : ℚ)
  (fraction_red_twice_fraction_blue : ℚ) 
  (h_total_knights : total_knights > 0)
  (h_fraction_red : fraction_red = 2 / 7)
  (h_fraction_magical : fraction_magical = 1 / 6)
  (h_relation : fraction_red_twice_fraction_blue = 2)
  (h_magic_eq : (total_knights : ℚ) * fraction_magical = 
    total_knights * fraction_red * fraction_red_twice_fraction_blue * fraction_magical / 2 + 
    total_knights * (1 - fraction_red) * fraction_magical / 2) :
  total_knights * (fraction_red * fraction_red_twice_fraction_blue / (fraction_red * fraction_red_twice_fraction_blue + (1 - fraction_red) / 2)) = 
  total_knights * 7 / 27 := 
sorry

end fraction_red_knights_magical_l31_31572


namespace total_cups_sold_l31_31842

theorem total_cups_sold (plastic_cups : ℕ) (ceramic_cups : ℕ) (total_sold : ℕ) :
  plastic_cups = 284 ∧ ceramic_cups = 284 → total_sold = 568 :=
by
  intros h
  cases h
  sorry

end total_cups_sold_l31_31842


namespace clare_money_left_l31_31985

noncomputable def cost_of_bread : ℝ := 4 * 2
noncomputable def cost_of_milk : ℝ := 2 * 2
noncomputable def cost_of_cereal : ℝ := 3 * 3
noncomputable def cost_of_apples : ℝ := 1 * 4

noncomputable def total_cost_before_discount : ℝ := cost_of_bread + cost_of_milk + cost_of_cereal + cost_of_apples
noncomputable def discount_amount : ℝ := total_cost_before_discount * 0.1
noncomputable def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount
noncomputable def sales_tax : ℝ := total_cost_after_discount * 0.05
noncomputable def total_cost_after_discount_and_tax : ℝ := total_cost_after_discount + sales_tax

noncomputable def initial_amount : ℝ := 47
noncomputable def money_left : ℝ := initial_amount - total_cost_after_discount_and_tax

theorem clare_money_left : money_left = 23.37 := by
  sorry

end clare_money_left_l31_31985


namespace remainder_proof_l31_31772

theorem remainder_proof (n : ℤ) (h : n % 6 = 1) : (3 * (n + 1812)) % 6 = 3 := 
by 
  sorry

end remainder_proof_l31_31772


namespace sum_red_equals_sum_blue_l31_31987

variable (r1 r2 r3 r4 b1 b2 b3 b4 w1 w2 w3 w4 : ℝ)

theorem sum_red_equals_sum_blue (h : (r1 + w1 / 2) + (r2 + w2 / 2) + (r3 + w3 / 2) + (r4 + w4 / 2) 
                                 = (b1 + w1 / 2) + (b2 + w2 / 2) + (b3 + w3 / 2) + (b4 + w4 / 2)) : 
  r1 + r2 + r3 + r4 = b1 + b2 + b3 + b4 :=
by sorry

end sum_red_equals_sum_blue_l31_31987


namespace area_of_triangle_ABC_l31_31542

theorem area_of_triangle_ABC :
  let A'B' := 4
  let B'C' := 3
  let angle_A'B'C' := 60
  let area_A'B'C' := (1 / 2) * A'B' * B'C' * Real.sin (angle_A'B'C' * Real.pi / 180)
  let ratio := 2 * Real.sqrt 2
  let area_ABC := ratio * area_A'B'C'
  area_ABC = 6 * Real.sqrt 6 := 
by
  sorry

end area_of_triangle_ABC_l31_31542


namespace equation_has_solution_implies_a_ge_2_l31_31921

theorem equation_has_solution_implies_a_ge_2 (a : ℝ) :
  (∃ x : ℝ, 4^x - a * 2^x - a + 3 = 0) → a ≥ 2 :=
by
  sorry

end equation_has_solution_implies_a_ge_2_l31_31921


namespace probability_colors_match_l31_31293

noncomputable def prob_abe_shows_blue : ℚ := 2 / 4
noncomputable def prob_bob_shows_blue : ℚ := 3 / 6
noncomputable def prob_abe_shows_green : ℚ := 2 / 4
noncomputable def prob_bob_shows_green : ℚ := 1 / 6

noncomputable def prob_same_color : ℚ :=
  (prob_abe_shows_blue * prob_bob_shows_blue) + (prob_abe_shows_green * prob_bob_shows_green)

theorem probability_colors_match : prob_same_color = 1 / 3 :=
by
  sorry

end probability_colors_match_l31_31293


namespace razorback_shop_tshirts_l31_31867

theorem razorback_shop_tshirts (T : ℕ) (h : 215 * T = 4300) : T = 20 :=
by sorry

end razorback_shop_tshirts_l31_31867


namespace smallest_x_no_triangle_l31_31877

def triangle_inequality_violated (a b c : ℝ) : Prop :=
a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem smallest_x_no_triangle (x : ℕ) (h : ∀ x, triangle_inequality_violated (7 - x : ℝ) (24 - x : ℝ) (26 - x : ℝ)) : x = 5 :=
sorry

end smallest_x_no_triangle_l31_31877


namespace optimal_play_winner_l31_31087

-- Definitions for the conditions
def chessboard_size (K N : ℕ) : Prop := True
def rook_initial_position (K N : ℕ) : (ℕ × ℕ) :=
  (K, N)
def move (r : ℕ × ℕ) (direction : ℕ) : (ℕ × ℕ) :=
  if direction = 0 then (r.1 - 1, r.2)
  else (r.1, r.2 - 1)
def rook_cannot_move (r : ℕ × ℕ) : Prop :=
  r.1 = 0 ∨ r.2 = 0

-- Theorem to prove the winner given the conditions
theorem optimal_play_winner (K N : ℕ) :
  (K = N → ∃ player : ℕ, player = 2) ∧ (K ≠ N → ∃ player : ℕ, player = 1) :=
by
  sorry

end optimal_play_winner_l31_31087


namespace area_of_formed_triangle_l31_31861

def triangle_area (S R d : ℝ) (S₁ : ℝ) : Prop :=
  S₁ = (S / 4) * |1 - (d^2 / R^2)|

variable (S R d : ℝ)

theorem area_of_formed_triangle (h : S₁ = (S / 4) * |1 - (d^2 / R^2)|) : triangle_area S R d S₁ :=
by
  sorry

end area_of_formed_triangle_l31_31861


namespace exists_root_in_interval_l31_31194

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x - 3

theorem exists_root_in_interval : ∃ c ∈ Set.Ioo (2 : ℝ) (3 : ℝ), f c = 0 :=
by
  sorry

end exists_root_in_interval_l31_31194


namespace distinct_factorizations_72_l31_31869

-- Define the function D that calculates the number of distinct factorizations.
noncomputable def D (n : Nat) : Nat := 
  -- Placeholder function to represent D, the actual implementation is skipped.
  sorry

-- Theorem stating the number of distinct factorizations of 72 considering the order of factors.
theorem distinct_factorizations_72 : D 72 = 119 :=
  sorry

end distinct_factorizations_72_l31_31869


namespace ages_total_l31_31724

variable (A B C : ℕ)

theorem ages_total (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : B = 10) : A + B + C = 27 :=
by
  sorry

end ages_total_l31_31724


namespace relationship_y1_y2_l31_31279

theorem relationship_y1_y2 (y1 y2 : ℝ) 
  (h1 : y1 = 3 / -1) 
  (h2 : y2 = 3 / -3) : 
  y1 < y2 :=
by
  sorry

end relationship_y1_y2_l31_31279


namespace mr_bird_on_time_58_mph_l31_31609

def mr_bird_travel_speed_exactly_on_time (d t: ℝ) (h₁ : d = 50 * (t + 1 / 15)) (h₂ : d = 70 * (t - 1 / 15)) : ℝ :=
  58

theorem mr_bird_on_time_58_mph (d t: ℝ) (h₁ : d = 50 * (t + 1 / 15)) (h₂ : d = 70 * (t - 1 / 15)) :
  mr_bird_travel_speed_exactly_on_time d t h₁ h₂ = 58 := 
  by
  sorry

end mr_bird_on_time_58_mph_l31_31609


namespace bookshelf_arrangements_l31_31854

theorem bookshelf_arrangements :
  let math_books := 6
  let english_books := 5
  let valid_arrangements := 2400
  (∃ (math_books : Nat) (english_books : Nat) (valid_arrangements : Nat), 
    math_books = 6 ∧ english_books = 5 ∧ valid_arrangements = 2400) :=
by
  sorry

end bookshelf_arrangements_l31_31854


namespace length_of_CD_l31_31020

theorem length_of_CD (x y : ℝ) (h1 : x / (3 + y) = 3 / 5) (h2 : (x + 3) / y = 4 / 7) (h3 : x + 3 + y = 273.6) : 3 + y = 273.6 :=
by
  sorry

end length_of_CD_l31_31020


namespace intersection_P_Q_l31_31162

-- Defining the two sets P and Q
def P := { x : ℤ | abs x ≤ 2 }
def Q := { x : ℝ | -1 < x ∧ x < 5/2 }

-- Statement to prove
theorem intersection_P_Q : 
  { x : ℤ | abs x ≤ 2 } ∩ { x : ℤ | -1 < ((x : ℝ)) ∧ ((x : ℝ)) < 5/2 } = {0, 1, 2} := sorry

end intersection_P_Q_l31_31162


namespace units_digit_of_sum_of_squares_2010_odds_l31_31841

noncomputable def sum_units_digit_of_squares (n : ℕ) : ℕ :=
  let units_digits := [1, 9, 5, 9, 1]
  List.foldl (λ acc x => (acc + x) % 10) 0 (List.map (λ i => units_digits.get! (i % 5)) (List.range (2 * n)))

theorem units_digit_of_sum_of_squares_2010_odds : sum_units_digit_of_squares 2010 = 0 := sorry

end units_digit_of_sum_of_squares_2010_odds_l31_31841


namespace johns_overall_profit_l31_31552

def cost_price_grinder : ℕ := 15000
def cost_price_mobile : ℕ := 8000
def loss_percent_grinder : ℝ := 0.05
def profit_percent_mobile : ℝ := 0.10

noncomputable def loss_amount_grinder : ℝ := loss_percent_grinder * cost_price_grinder
noncomputable def selling_price_grinder : ℝ := cost_price_grinder - loss_amount_grinder

noncomputable def profit_amount_mobile : ℝ := profit_percent_mobile * cost_price_mobile
noncomputable def selling_price_mobile : ℝ := cost_price_mobile + profit_amount_mobile

noncomputable def total_cost_price : ℝ := cost_price_grinder + cost_price_mobile
noncomputable def total_selling_price : ℝ := selling_price_grinder + selling_price_mobile
noncomputable def overall_profit : ℝ := total_selling_price - total_cost_price

theorem johns_overall_profit :
  overall_profit = 50 := 
by
  sorry

end johns_overall_profit_l31_31552


namespace cos_sum_seventh_roots_of_unity_l31_31972

noncomputable def cos_sum (α : ℝ) : ℝ := 
  Real.cos α + Real.cos (2 * α) + Real.cos (4 * α)

theorem cos_sum_seventh_roots_of_unity (z : ℂ) (α : ℝ)
  (hz : z^7 = 1) (hz_ne_one : z ≠ 1) (hα : z = Complex.exp (Complex.I * α)) :
  cos_sum α = -1/2 :=
by
  sorry

end cos_sum_seventh_roots_of_unity_l31_31972


namespace find_chemistry_marks_l31_31906

theorem find_chemistry_marks
  (english_marks : ℕ) (math_marks : ℕ) (physics_marks : ℕ) (biology_marks : ℕ) (average_marks : ℕ) (chemistry_marks : ℕ) :
  english_marks = 86 → math_marks = 89 → physics_marks = 82 → biology_marks = 81 → average_marks = 85 →
  chemistry_marks = 425 - (english_marks + math_marks + physics_marks + biology_marks) →
  chemistry_marks = 87 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  have total_marks := 425 - (86 + 89 + 82 + 81)
  norm_num at total_marks
  exact h6

end find_chemistry_marks_l31_31906


namespace train_passing_time_l31_31936

theorem train_passing_time :
  ∀ (length : ℕ) (speed_km_hr : ℕ), length = 300 ∧ speed_km_hr = 90 →
  (length / (speed_km_hr * (1000 / 3600)) = 12) := 
by
  intros length speed_km_hr h
  have h_length : length = 300 := h.1
  have h_speed : speed_km_hr = 90 := h.2
  sorry

end train_passing_time_l31_31936


namespace remainder_of_sum_of_consecutive_days_l31_31831

theorem remainder_of_sum_of_consecutive_days :
  (100045 + 100046 + 100047 + 100048 + 100049 + 100050 + 100051 + 100052) % 5 = 3 :=
by
  sorry

end remainder_of_sum_of_consecutive_days_l31_31831


namespace not_all_zero_iff_at_least_one_nonzero_l31_31564

theorem not_all_zero_iff_at_least_one_nonzero (a b c : ℝ) :
  ¬ (a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
by 
  sorry

end not_all_zero_iff_at_least_one_nonzero_l31_31564


namespace range_of_a_l31_31058

noncomputable def curve_y (a : ℝ) (x : ℝ) : ℝ := (a - 3) * x^3 + Real.log x
noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ := x^3 - a * x^2 - 3 * x + 1

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ deriv (curve_y a) x = 0) ∧
  (∀ x ∈ Set.Icc (1 : ℝ) 2, 0 ≤ deriv (function_f a) x) → a ≤ 0 :=
by sorry

end range_of_a_l31_31058


namespace range_of_a_in_third_quadrant_l31_31042

def pointInThirdQuadrant (x y : ℝ) := x < 0 ∧ y < 0

theorem range_of_a_in_third_quadrant (a : ℝ) (M : ℝ × ℝ) 
  (hM : M = (-1, a-1)) (hThirdQuad : pointInThirdQuadrant M.1 M.2) : 
  a < 1 :=
by
  sorry

end range_of_a_in_third_quadrant_l31_31042


namespace chuck_total_playable_area_l31_31288

noncomputable def chuck_roaming_area (shed_length shed_width leash_length : ℝ) : ℝ :=
  let larger_arc_area := (3 / 4) * Real.pi * leash_length ^ 2
  let additional_sector_area := (1 / 4) * Real.pi * (leash_length - shed_length) ^ 2
  larger_arc_area + additional_sector_area

theorem chuck_total_playable_area :
  chuck_roaming_area 3 4 5 = 19 * Real.pi :=
  by
  sorry

end chuck_total_playable_area_l31_31288


namespace bill_and_harry_nuts_l31_31574

theorem bill_and_harry_nuts {Bill Harry Sue : ℕ} 
    (h1 : Bill = 6 * Harry) 
    (h2 : Harry = 2 * Sue) 
    (h3 : Sue = 48) : 
    Bill + Harry = 672 := 
by
  sorry

end bill_and_harry_nuts_l31_31574


namespace john_ate_cookies_l31_31391

-- Definitions for conditions
def dozen := 12

-- Given conditions
def initial_cookies : ℕ := 2 * dozen
def cookies_left : ℕ := 21

-- Problem statement
theorem john_ate_cookies : initial_cookies - cookies_left = 3 :=
by
  -- Solution steps omitted, only statement provided
  sorry

end john_ate_cookies_l31_31391


namespace cat_walking_rate_l31_31879

theorem cat_walking_rate :
  let resisting_time := 20 -- minutes
  let total_distance := 64 -- feet
  let total_time := 28 -- minutes
  let walking_time := total_time - resisting_time
  (total_distance / walking_time = 8) :=
by
  let resisting_time := 20
  let total_distance := 64
  let total_time := 28
  let walking_time := total_time - resisting_time
  have : total_distance / walking_time = 8 :=
    by norm_num [total_distance, walking_time]
  exact this

end cat_walking_rate_l31_31879


namespace sum_of_digits_of_x_l31_31035

def two_digit_palindrome (x : ℕ) : Prop :=
  (10 ≤ x ∧ x ≤ 99) ∧ (x = (x % 10) * 10 + (x % 10))

def three_digit_palindrome (y : ℕ) : Prop :=
  (100 ≤ y ∧ y ≤ 999) ∧ (y = (y % 10) * 101 + (y % 10))

theorem sum_of_digits_of_x (x : ℕ) (h1 : two_digit_palindrome x) (h2 : three_digit_palindrome (x + 10)) : 
  (x % 10 + x / 10) = 10 :=
by
  sorry

end sum_of_digits_of_x_l31_31035


namespace daves_apps_count_l31_31211

theorem daves_apps_count (x : ℕ) : 
  let initial_apps : ℕ := 21
  let added_apps : ℕ := 89
  let total_apps : ℕ := initial_apps + added_apps
  let deleted_apps : ℕ := x
  let more_added_apps : ℕ := x + 3
  total_apps - deleted_apps + more_added_apps = 113 :=
by
  sorry

end daves_apps_count_l31_31211


namespace number_of_days_l31_31993

variables (S Wx Wy : ℝ)

-- Given conditions
def condition1 : Prop := S = 36 * Wx
def condition2 : Prop := S = 45 * Wy

-- The lean statement to prove the number of days D = 20
theorem number_of_days (h1 : condition1 S Wx) (h2 : condition2 S Wy) : 
  S / (Wx + Wy) = 20 :=
by
  sorry

end number_of_days_l31_31993


namespace find_a_range_l31_31370

noncomputable def monotonic_func_a_range : Set ℝ :=
  {a : ℝ | ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → (3 * x^2 + a ≥ 0 ∨ 3 * x^2 + a ≤ 0)}

theorem find_a_range :
  monotonic_func_a_range = {a | a ≤ -27} ∪ {a | a ≥ 0} :=
by
  sorry

end find_a_range_l31_31370


namespace trig_identity_l31_31300

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem trig_identity (θ : ℝ) : sin (θ + 75 * Real.pi / 180) + cos (θ + 45 * Real.pi / 180) - Real.sqrt 3 * cos (θ + 15 * Real.pi / 180) = 0 :=
by
  sorry

end trig_identity_l31_31300


namespace labor_day_to_national_day_l31_31216

theorem labor_day_to_national_day :
  let labor_day := 1 -- Monday is represented as 1
  let factor_31 := 31
  let factor_30 := 30
  let total_days := (factor_31 * 3 + factor_30 * 2)
  (labor_day + total_days % 7) % 7 = 0 := -- Since 0 corresponds to Sunday modulo 7
by
  let labor_day := 1
  let factor_31 := 31
  let factor_30 := 30
  let total_days := (factor_31 * 3 + factor_30 * 2)
  have h1 : (labor_day + total_days % 7) % 7 = ((1 + (31 * 3 + 30 * 2) % 7) % 7) := by rfl
  sorry

end labor_day_to_national_day_l31_31216


namespace correct_operation_l31_31503

theorem correct_operation (a : ℝ) :
  (a^5)^2 = a^10 :=
by sorry

end correct_operation_l31_31503


namespace bales_in_barn_now_l31_31392

-- Define the initial number of bales
def initial_bales : ℕ := 28

-- Define the number of bales added by Tim
def added_bales : ℕ := 26

-- Define the total number of bales
def total_bales : ℕ := initial_bales + added_bales

-- Theorem stating the total number of bales
theorem bales_in_barn_now : total_bales = 54 := by
  sorry

end bales_in_barn_now_l31_31392


namespace triangle_largest_angle_l31_31731

theorem triangle_largest_angle (k : ℕ) 
  (h1 : 3 * k + 4 * k + 5 * k = 180)
  (h2 : ∃ k, 3 * k + 4 * k + 5 * k = 180) :
  5 * k = 75 :=
sorry

end triangle_largest_angle_l31_31731


namespace investment_inequality_l31_31222

-- Defining the initial investment
def initial_investment : ℝ := 200

-- Year 1 changes
def alpha_year1 := initial_investment * 1.30
def beta_year1 := initial_investment * 0.80
def gamma_year1 := initial_investment * 1.10
def delta_year1 := initial_investment * 0.90

-- Year 2 changes
def alpha_final := alpha_year1 * 0.85
def beta_final := beta_year1 * 1.30
def gamma_final := gamma_year1 * 0.95
def delta_final := delta_year1 * 1.20

-- Prove the final inequality
theorem investment_inequality : beta_final < gamma_final ∧ gamma_final < delta_final ∧ delta_final < alpha_final :=
by {
  sorry
}

end investment_inequality_l31_31222


namespace problem1_problem2_l31_31000

-- Definitions for the problem

/-- Definition of point P in Cartesian coordinate system -/
def P (x : ℝ) : ℝ × ℝ :=
  (x - 2, x)

-- First proof problem statement
theorem problem1 (x : ℝ) (h : (x - 2) * x < 0) : x = 1 :=
sorry

-- Second proof problem statement
theorem problem2 (x : ℝ) (h1 : x - 2 < 0) (h2 : x > 0) : 0 < x ∧ x < 2 :=
sorry

end problem1_problem2_l31_31000


namespace number_represented_by_B_l31_31269

theorem number_represented_by_B (b : ℤ) : 
  (abs (b - 3) = 5) -> (b = 8 ∨ b = -2) :=
by
  intro h
  sorry

end number_represented_by_B_l31_31269


namespace cupcakes_left_over_l31_31487

def total_cupcakes := 40
def ms_delmont_class := 18
def mrs_donnelly_class := 16
def ms_delmont := 1
def mrs_donnelly := 1
def school_nurse := 1
def school_principal := 1

def total_given_away := ms_delmont_class + mrs_donnelly_class + ms_delmont + mrs_donnelly + school_nurse + school_principal

theorem cupcakes_left_over : total_cupcakes - total_given_away = 2 := by
  sorry

end cupcakes_left_over_l31_31487


namespace min_crossing_time_proof_l31_31303

def min_crossing_time (times : List ℕ) : ℕ :=
  -- Function to compute the minimum crossing time. Note: Actual implementation skipped.
sorry

theorem min_crossing_time_proof
  (times : List ℕ)
  (h_times : times = [2, 4, 8, 16]) :
  min_crossing_time times = 30 :=
sorry

end min_crossing_time_proof_l31_31303


namespace inequality_c_l31_31862

theorem inequality_c (x : ℝ) : x^2 + 1 + 1 / (x^2 + 1) ≥ 2 := sorry

end inequality_c_l31_31862


namespace height_of_cylinder_l31_31227

theorem height_of_cylinder (r_hemisphere : ℝ) (r_cylinder : ℝ) (h_cylinder : ℝ) :
  r_hemisphere = 7 → r_cylinder = 3 → h_cylinder = 2 * Real.sqrt 10 :=
by
  intro r_hemisphere_eq r_cylinder_eq
  sorry

end height_of_cylinder_l31_31227


namespace value_of_M_l31_31889

theorem value_of_M (M : ℝ) (h : (25 / 100) * M = (35 / 100) * 1800) : M = 2520 := 
sorry

end value_of_M_l31_31889


namespace employed_females_percentage_l31_31181

variable (P : ℝ) -- Total population of town X
variable (E_P : ℝ) -- Percentage of the population that is employed
variable (M_E_P : ℝ) -- Percentage of the population that are employed males

-- Conditions
axiom h1 : E_P = 0.64
axiom h2 : M_E_P = 0.55

-- Target: Prove the percentage of employed people in town X that are females
theorem employed_females_percentage (h : P > 0) : 
  (E_P * P - M_E_P * P) / (E_P * P) * 100 = 14.06 := by
sorry

end employed_females_percentage_l31_31181


namespace number_of_real_roots_l31_31200

theorem number_of_real_roots (a : ℝ) :
  (|a| < (2 * Real.sqrt 3 / 9) → ∃ x y z : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ z^3 - z - a = 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  (|a| = (2 * Real.sqrt 3 / 9) → ∃ x y : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ x = y) ∧
  (|a| > (2 * Real.sqrt 3 / 9) → ∃ x : ℝ, x^3 - x - a = 0 ∧ ∀ y : ℝ, y^3 - y - a ≠ 0 ∨ y = x) :=
sorry

end number_of_real_roots_l31_31200


namespace no_positive_integer_triples_l31_31050

theorem no_positive_integer_triples (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) : ¬ (x^2 + y^2 + 41 = 2^n) :=
  sorry

end no_positive_integer_triples_l31_31050


namespace num_articles_cost_price_l31_31662

theorem num_articles_cost_price (N C S : ℝ) (h1 : N * C = 50 * S) (h2 : (S - C) / C * 100 = 10) : N = 55 := 
sorry

end num_articles_cost_price_l31_31662


namespace impossible_coins_l31_31213

theorem impossible_coins (p_1 p_2 : ℝ) 
  (h1 : (1 - p_1) * (1 - p_2) = p_1 * p_2)
  (h2 : p_1 * (1 - p_2) + p_2 * (1 - p_1) = p_1 * p_2) : False := 
sorry

end impossible_coins_l31_31213


namespace M_union_N_eq_l31_31002

open Set

def M : Set ℝ := { x | x^2 - 4 * x < 0 }
def N : Set ℝ := { x | abs x ≤ 2 }

theorem M_union_N_eq : M ∪ N = Ico (-2 : ℝ) 4 := by
  sorry

end M_union_N_eq_l31_31002


namespace smallest_n_divisibility_problem_l31_31793

theorem smallest_n_divisibility_problem :
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), 1 ≤ k → k ≤ n + 2 → n^3 - n ≠ 0 → (n^3 - n) % k = 0) ∧
    (∃ (k : ℕ), 1 ≤ k → k ≤ n + 2 → k ∣ n^3 - n) ∧
    (∃ (k : ℕ), 1 ≤ k → k ≤ n + 2 → ¬ k ∣ n^3 - n) ∧
    (∀ (m : ℕ), m > 0 ∧ (∀ (k : ℕ), 1 ≤ k → k ≤ m + 2 → m^3 - m ≠ 0 → (m^3 - m) % k = 0) ∧
      (∃ (k : ℕ), 1 ≤ k → k ≤ m + 2 → k ∣ m^3 - m) ∧
      (∃ (k : ℕ), 1 ≤ k → k ≤ m + 2 → ¬ k ∣ m^3 - m) → n ≤ m) :=
sorry

end smallest_n_divisibility_problem_l31_31793


namespace avg_daily_production_l31_31045

theorem avg_daily_production (x y : ℕ) (h1 : x + y = 350) (h2 : 2 * x - y = 250) : x = 200 ∧ y = 150 := 
by
  sorry

end avg_daily_production_l31_31045


namespace point_A_in_second_quadrant_l31_31147

def A : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_A_in_second_quadrant : isSecondQuadrant A :=
by
  sorry

end point_A_in_second_quadrant_l31_31147


namespace solution_set_inequality_range_of_t_l31_31384

noncomputable def f (x : ℝ) : ℝ := |x| - 2 * |x + 3|

-- Problem (1)
theorem solution_set_inequality :
  { x : ℝ | f x ≥ 2 } = { x : ℝ | -4 ≤ x ∧ x ≤ - (8 / 3) } :=
by
  sorry

-- Problem (2)
theorem range_of_t (t : ℝ) :
  (∃ x : ℝ, f x - |3 * t - 2| ≥ 0) ↔ (- (1 / 3) ≤ t ∧ t ≤ 5 / 3) :=
by
  sorry

end solution_set_inequality_range_of_t_l31_31384


namespace calc_man_dividend_l31_31315

noncomputable def calc_dividend (investment : ℝ) (face_value : ℝ) (premium : ℝ) (dividend_percent : ℝ) : ℝ :=
  let cost_per_share := face_value * (1 + premium / 100)
  let number_of_shares := investment / cost_per_share
  let dividend_per_share := dividend_percent / 100 * face_value
  let total_dividend := dividend_per_share * number_of_shares
  total_dividend

theorem calc_man_dividend :
  calc_dividend 14400 100 20 5 = 600 :=
by
  sorry

end calc_man_dividend_l31_31315


namespace distance_to_nearest_edge_of_picture_l31_31283

def wall_width : ℕ := 26
def picture_width : ℕ := 4
def distance_from_end (wall picture : ℕ) : ℕ := (wall - picture) / 2

theorem distance_to_nearest_edge_of_picture :
  distance_from_end wall_width picture_width = 11 :=
sorry

end distance_to_nearest_edge_of_picture_l31_31283


namespace minimum_value_ineq_l31_31964

noncomputable def minimum_value (x y z : ℝ) := x^2 + 4 * x * y + 4 * y^2 + 4 * z^2

theorem minimum_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 64) : minimum_value x y z ≥ 192 :=
by {
  sorry
}

end minimum_value_ineq_l31_31964


namespace two_p_in_S_l31_31988

def is_in_S (a b : ℤ) : Prop :=
  ∃ k : ℤ, k = a^2 + 5 * b^2 ∧ Int.gcd a b = 1

def S : Set ℤ := { x | ∃ a b : ℤ, is_in_S a b ∧ a^2 + 5 * b^2 = x }

theorem two_p_in_S (k p n : ℤ) (hp1 : p = 4 * n + 3) (hp2 : Nat.Prime (Int.natAbs p))
  (hk : 0 < k) (hkp : k * p ∈ S) : 2 * p ∈ S := 
sorry

end two_p_in_S_l31_31988


namespace math_study_time_l31_31304

-- Conditions
def science_time : ℕ := 25
def total_time : ℕ := 60

-- Theorem statement
theorem math_study_time :
  total_time - science_time = 35 := by
  -- Proof placeholder
  sorry

end math_study_time_l31_31304


namespace fraction_irreducible_l31_31640

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := 
by 
  sorry

end fraction_irreducible_l31_31640


namespace total_servings_l31_31385

/-- The first jar contains 24 2/3 tablespoons of peanut butter. -/
def first_jar_pb : ℚ := 74 / 3

/-- The second jar contains 19 1/2 tablespoons of peanut butter. -/
def second_jar_pb : ℚ := 39 / 2

/-- One serving size is 3 tablespoons. -/
def serving_size : ℚ := 3

/-- The total servings of peanut butter in both jars is 14 13/18 servings. -/
theorem total_servings : (first_jar_pb + second_jar_pb) / serving_size = 14 + 13 / 18 :=
by
  sorry

end total_servings_l31_31385


namespace circle_area_l31_31960

theorem circle_area (r : ℝ) (h : 8 * (1 / (2 * π * r)) = 2 * r) : π * r^2 = 2 := by
  sorry

end circle_area_l31_31960


namespace product_mod_five_l31_31959

theorem product_mod_five (a b c : ℕ) (h₁ : a = 1236) (h₂ : b = 7483) (h₃ : c = 53) :
  (a * b * c) % 5 = 4 :=
by
  sorry

end product_mod_five_l31_31959


namespace multiple_of_second_lock_time_l31_31953

def first_lock_time := 5
def second_lock_time := 3 * first_lock_time - 3
def combined_lock_time := 60

theorem multiple_of_second_lock_time : combined_lock_time / second_lock_time = 5 := by
  -- Adding a proof placeholder using sorry
  sorry

end multiple_of_second_lock_time_l31_31953


namespace op_example_l31_31873

def op (a b : ℚ) : ℚ := a * b / (a + b)

theorem op_example : op (op 3 5) (op 5 4) = 60 / 59 := by
  sorry

end op_example_l31_31873


namespace part1_part2_l31_31614

open Set Real

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | 2 ≤ x ∧ x < 5 }
def setB : Set ℝ := { x | 1 < x ∧ x < 8 }
def setC (a : ℝ) : Set ℝ := { x | x < a - 1 ∨ x > a }

-- Conditions:
-- - Complement of A
def complementA : Set ℝ := { x | x < 2 ∨ x ≥ 5 }

-- Question parts:
-- (1) Finding intersection of complementA and B
theorem part1 : (complementA ∩ setB) = { x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 8) } := sorry

-- (2) Finding range of a for specific condition on C
theorem part2 (a : ℝ) : (setA ∪ setC a = univ) → (a ≤ 2 ∨ a > 6) := sorry

end part1_part2_l31_31614


namespace bike_cost_l31_31502

-- Defining the problem conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def leftover : ℚ := 20  -- 20 dollars left over
def quarter_value : ℚ := 0.25

-- Define the total quarters Jenn has
def total_quarters := jars * quarters_per_jar

-- Define the total amount of money from quarters
def total_money_quarters := total_quarters * quarter_value

-- Prove that the cost of the bike is $200
theorem bike_cost : total_money_quarters + leftover - 20 = 200 :=
sorry

end bike_cost_l31_31502


namespace log_difference_example_l31_31931

theorem log_difference_example :
  ∀ (log : ℕ → ℝ),
    log 3 * 24 - log 3 * 8 = 1 := 
by
sorry

end log_difference_example_l31_31931


namespace problem1_problem2_l31_31686

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := { x | a - b < x ∧ x < a + b }
def B : Set ℝ := { x | x < -1 ∨ x > 5 }

-- First problem: prove the range of a
theorem problem1 (a : ℝ) (h : A a 1 ⊆ B) : a ≤ -2 ∨ a ≥ 6 := by
  sorry

-- Second problem: prove the range of b
theorem problem2 (b : ℝ) (h : A 1 b ∩ B = ∅) : b ≤ 2 := by
  sorry

end problem1_problem2_l31_31686


namespace algebraic_expression_simplification_l31_31990

theorem algebraic_expression_simplification (x y : ℝ) (h : x + y = 1) : x^3 + y^3 + 3 * x * y = 1 := 
by
  sorry

end algebraic_expression_simplification_l31_31990


namespace range_of_a_range_of_m_l31_31217

-- Definition of proposition p: Equation has real roots
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a * x + a + 3 = 0

-- Definition of proposition q: m - 1 <= a <= m + 1
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 1

-- Part (I): Range of a when ¬p is true
theorem range_of_a (a : ℝ) (hp : ¬ p a) : -2 < a ∧ a < 6 :=
sorry

-- Part (II): Range of m when p is a necessary but not sufficient condition for q
theorem range_of_m (m : ℝ) (hnp : ∀ a, q m a → p a) (hns : ∃ a, q m a ∧ ¬p a) : m ≤ -3 ∨ m ≥ 7 :=
sorry

end range_of_a_range_of_m_l31_31217


namespace Jamie_minimum_4th_quarter_score_l31_31232

theorem Jamie_minimum_4th_quarter_score (q1 q2 q3 : ℤ) (avg : ℤ) (minimum_score : ℤ) :
  q1 = 84 → q2 = 80 → q3 = 83 → avg = 85 → minimum_score = 93 → 4 * avg - (q1 + q2 + q3) = minimum_score :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end Jamie_minimum_4th_quarter_score_l31_31232


namespace fraction_value_l31_31998

theorem fraction_value : (10 + 20 + 30 + 40) / 10 = 10 := by
  sorry

end fraction_value_l31_31998


namespace probability_dice_sum_12_l31_31386

def total_outcomes : ℕ := 216
def favorable_outcomes : ℕ := 25

theorem probability_dice_sum_12 :
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 216 := by
  sorry

end probability_dice_sum_12_l31_31386


namespace complete_square_eqn_l31_31145

theorem complete_square_eqn (d e : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 → (x + d)^2 = e) → d + e = 5 :=
by
  sorry

end complete_square_eqn_l31_31145


namespace value_of_x_plus_y_l31_31547

theorem value_of_x_plus_y (x y : ℤ) (h1 : x + 2 = 10) (h2 : y - 1 = 6) : x + y = 15 :=
by
  sorry

end value_of_x_plus_y_l31_31547


namespace bertha_descendants_without_daughters_l31_31214

-- Definitions based on conditions
def num_daughters : ℕ := 6
def total_daughters_and_granddaughters : ℕ := 30
def daughters_with_daughters := (total_daughters_and_granddaughters - num_daughters) / 6

-- The number of Bertha's daughters who have no daughters:
def daughters_without_daughters := num_daughters - daughters_with_daughters
-- The number of Bertha's granddaughters:
def num_granddaughters := total_daughters_and_granddaughters - num_daughters
-- All granddaughters have no daughters:
def granddaughters_without_daughters := num_granddaughters

-- The total number of daughters and granddaughters without daughters
def total_without_daughters := daughters_without_daughters + granddaughters_without_daughters

-- Main theorem statement
theorem bertha_descendants_without_daughters :
  total_without_daughters = 26 :=
by
  sorry

end bertha_descendants_without_daughters_l31_31214


namespace lcm_220_504_l31_31067

/-- The least common multiple of 220 and 504 is 27720. -/
theorem lcm_220_504 : Nat.lcm 220 504 = 27720 :=
by
  -- This is the final statement of the theorem. The proof is not provided and marked with 'sorry'.
  sorry

end lcm_220_504_l31_31067


namespace batsman_average_19th_inning_l31_31074

theorem batsman_average_19th_inning (initial_avg : ℝ) 
    (scored_19th_inning : ℝ) 
    (new_avg : ℝ) 
    (h1 : scored_19th_inning = 100) 
    (h2 : new_avg = initial_avg + 2)
    (h3 : new_avg = (18 * initial_avg + 100) / 19) :
    new_avg = 64 :=
by
  have h4 : initial_avg = 62 := by
    sorry
  sorry

end batsman_average_19th_inning_l31_31074


namespace minimum_value_of_f_l31_31484

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 4

theorem minimum_value_of_f : ∃ x : ℝ, f x = -5 ∧ ∀ y : ℝ, f y ≥ -5 :=
by
  sorry

end minimum_value_of_f_l31_31484


namespace polynomial_factorization_l31_31983

noncomputable def poly_1 : Polynomial ℤ := (Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.C 1)
noncomputable def poly_2 : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X ^ 12 - Polynomial.C 1 * Polynomial.X ^ 11 +
  Polynomial.C 1 * Polynomial.X ^ 9 - Polynomial.C 1 * Polynomial.X ^ 8 +
  Polynomial.C 1 * Polynomial.X ^ 6 - Polynomial.C 1 * Polynomial.X ^ 4 +
  Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 1 * Polynomial.X + Polynomial.C 1
noncomputable def polynomial_expression : Polynomial ℤ := Polynomial.X ^ 15 + Polynomial.X ^ 10 + Polynomial.C 1

theorem polynomial_factorization : polynomial_expression = poly_1 * poly_2 :=
  by { sorry }

end polynomial_factorization_l31_31983


namespace mart_income_percentage_l31_31630

variables (T J M : ℝ)

theorem mart_income_percentage (h1 : M = 1.60 * T) (h2 : T = 0.50 * J) :
  M = 0.80 * J :=
by
  sorry

end mart_income_percentage_l31_31630


namespace triangle_count_with_perimeter_11_l31_31709

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l31_31709


namespace inequality_holds_l31_31752

theorem inequality_holds (x : ℝ) (n : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 1) (h3 : n > 0) : 
  (1 + x) ^ n ≥ (1 - x) ^ n + 2 * n * x * (1 - x ^ 2) ^ ((n - 1) / 2) :=
sorry

end inequality_holds_l31_31752


namespace decrease_percent_revenue_l31_31544

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let original_revenue := T * C
  let new_tax := 0.68 * T
  let new_consumption := 1.12 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 23.84 := by {
    sorry
  }

end decrease_percent_revenue_l31_31544


namespace sufficient_not_necessary_condition_l31_31297

theorem sufficient_not_necessary_condition (x y : ℝ) (h1 : x ≥ 1) (h2 : y ≥ 2) : 
  x + y ≥ 3 ∧ (¬ (∀ x y : ℝ, x + y ≥ 3 → x ≥ 1 ∧ y ≥ 2)) := 
by {
  sorry -- The actual proof goes here.
}

end sufficient_not_necessary_condition_l31_31297


namespace activity_popularity_order_l31_31782

-- Definitions for the fractions representing activity popularity
def dodgeball_popularity : Rat := 9 / 24
def magic_show_popularity : Rat := 4 / 12
def singing_contest_popularity : Rat := 1 / 3

-- Theorem stating the order of activities based on popularity
theorem activity_popularity_order :
  dodgeball_popularity > magic_show_popularity ∧ magic_show_popularity = singing_contest_popularity :=
by 
  sorry

end activity_popularity_order_l31_31782


namespace length_of_BC_l31_31076

theorem length_of_BC (BD CD : ℝ) (h1 : BD = 3 + 3 * BD) (h2 : CD = 2 + 2 * CD) (h3 : 4 * BD + 3 * CD + 5 = 20) : 2 * CD + 2 = 4 :=
by {
  sorry
}

end length_of_BC_l31_31076


namespace unique_rs_exists_l31_31607

theorem unique_rs_exists (a b : ℕ) (ha : a > 1) (hb : b > 1) (gcd_ab : Nat.gcd a b = 1) :
  ∃! (r s : ℤ), (0 < r ∧ r < b) ∧ (0 < s ∧ s < a) ∧ (a * r - b * s = 1) :=
  sorry

end unique_rs_exists_l31_31607


namespace find_white_towels_l31_31950

variable {W : ℕ} -- Define W as a natural number

-- Define the conditions as Lean definitions
def initial_towel_count (W : ℕ) : ℕ := 35 + W
def remaining_towel_count (W : ℕ) : ℕ := initial_towel_count W - 34

-- Theorem statement: Proving that W = 21 given the conditions
theorem find_white_towels (h : remaining_towel_count W = 22) : W = 21 :=
by
  sorry

end find_white_towels_l31_31950


namespace faye_age_l31_31600

theorem faye_age (D E C F : ℤ)
  (h1 : D = E - 4)
  (h2 : E = C + 5)
  (h3 : F = C + 4)
  (hD : D = 18) :
  F = 21 :=
by
  sorry

end faye_age_l31_31600


namespace c_share_l31_31944

theorem c_share (a b c : ℝ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : a + b + c = 700) : c = 400 :=
by 
  -- Proof goes here
  sorry

end c_share_l31_31944


namespace table_to_chair_ratio_l31_31561

noncomputable def price_chair : ℤ := 20
noncomputable def price_table : ℤ := 60
noncomputable def price_couch : ℤ := 300

theorem table_to_chair_ratio 
  (h1 : price_couch = 300)
  (h2 : price_couch = 5 * price_table)
  (h3 : price_chair + price_table + price_couch = 380)
  : price_table / price_chair = 3 := 
by 
  sorry

end table_to_chair_ratio_l31_31561


namespace equilateral_A1C1E1_l31_31899

variables {A B C D E F A₁ B₁ C₁ D₁ E₁ F₁ : Type*}

-- Defining the convex hexagon and the equilateral triangles.
def is_convex_hexagon (A B C D E F : Type*) : Prop := sorry

def is_equilateral (P Q R : Type*) : Prop := sorry

-- Given conditions
variable (h_hexagon : is_convex_hexagon A B C D E F)
variable (h_eq_triangles :
  is_equilateral A B C₁ ∧ is_equilateral B C D₁ ∧ is_equilateral C D E₁ ∧
  is_equilateral D E F₁ ∧ is_equilateral E F A₁ ∧ is_equilateral F A B₁)
variable (h_B1D1F1 : is_equilateral B₁ D₁ F₁)

-- Statement to be proved
theorem equilateral_A1C1E1 :
  is_equilateral A₁ C₁ E₁ :=
sorry

end equilateral_A1C1E1_l31_31899


namespace total_days_stayed_l31_31126

-- Definitions of given conditions as variables
def cost_first_week := 18
def days_first_week := 7
def cost_additional_week := 13
def total_cost := 334

-- Formulation of the target statement in Lean
theorem total_days_stayed :
  (days_first_week + 
  ((total_cost - (days_first_week * cost_first_week)) / cost_additional_week)) = 23 :=
by
  sorry

end total_days_stayed_l31_31126


namespace hundredth_odd_integer_is_199_sum_of_first_100_odd_integers_is_10000_l31_31356

noncomputable def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

noncomputable def sum_first_n_odd_positive_integers (n : ℕ) : ℕ :=
  n * n

theorem hundredth_odd_integer_is_199 : nth_odd_positive_integer 100 = 199 :=
  by
  sorry

theorem sum_of_first_100_odd_integers_is_10000 : sum_first_n_odd_positive_integers 100 = 10000 :=
  by
  sorry

end hundredth_odd_integer_is_199_sum_of_first_100_odd_integers_is_10000_l31_31356


namespace tg_ctg_sum_l31_31616

theorem tg_ctg_sum (x : Real) 
  (h : Real.cos x ≠ 0 ∧ Real.sin x ≠ 0 ∧ 1 / Real.cos x - 1 / Real.sin x = 4 * Real.sqrt 3) :
  (Real.sin x / Real.cos x + Real.cos x / Real.sin x = 8 ∨ Real.sin x / Real.cos x + Real.cos x / Real.sin x = -6) :=
sorry

end tg_ctg_sum_l31_31616


namespace cube_sum_minus_triple_product_l31_31896

theorem cube_sum_minus_triple_product (x y z : ℝ) (h1 : x + y + z = 8) (h2 : xy + yz + zx = 20) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 32 :=
sorry

end cube_sum_minus_triple_product_l31_31896


namespace races_needed_to_declare_winner_l31_31832

noncomputable def total_sprinters : ℕ := 275
noncomputable def sprinters_per_race : ℕ := 7
noncomputable def sprinters_advance : ℕ := 2
noncomputable def sprinters_eliminated : ℕ := 5

theorem races_needed_to_declare_winner :
  (total_sprinters - 1 + sprinters_eliminated) / sprinters_eliminated = 59 :=
by
  sorry

end races_needed_to_declare_winner_l31_31832


namespace cnc_processing_time_l31_31837

theorem cnc_processing_time :
  (∃ (hours: ℕ), 3 * (960 / hours) = 960 / 3) → 1 * (400 / 5) = 400 / 1 :=
by
  sorry

end cnc_processing_time_l31_31837


namespace neither_necessary_nor_sufficient_l31_31130

def p (x y : ℝ) : Prop := x > 1 ∧ y > 1
def q (x y : ℝ) : Prop := x + y > 3

theorem neither_necessary_nor_sufficient :
  ¬ (∀ x y, q x y → p x y) ∧ ¬ (∀ x y, p x y → q x y) :=
by
  sorry

end neither_necessary_nor_sufficient_l31_31130


namespace spend_amount_7_l31_31380

variable (x y z w : ℕ) (k : ℕ)

theorem spend_amount_7 
  (h1 : 10 * x + 15 * y + 25 * z + 40 * w = 100 * k)
  (h2 : x + y + z + w = 30)
  (h3 : (x = 5 ∨ x = 10) ∧ (y = 5 ∨ y = 10) ∧ (z = 5 ∨ z = 10) ∧ (w = 5 ∨ w = 10)) : 
  k = 7 := 
sorry

end spend_amount_7_l31_31380


namespace circles_max_ab_l31_31715

theorem circles_max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ (x y : ℝ), (x + a)^2 + (y - 2)^2 = 1 ∧ (x - b)^2 + (y - 2)^2 = 4) →
  a + b = 3 →
  ab ≤ 9 / 4 := 
  by
  sorry

end circles_max_ab_l31_31715


namespace clothing_weight_removed_l31_31619

/-- 
In a suitcase, the initial ratio of books to clothes to electronics, by weight measured in pounds, 
is 7:4:3. The electronics weight 9 pounds. Someone removes some pounds of clothing, doubling the ratio of books to clothes. 
This theorem verifies the weight of clothing removed is 1.5 pounds.
-/
theorem clothing_weight_removed 
  (B C E : ℕ) 
  (initial_ratio : B / 7 = C / 4 ∧ C / 4 = E / 3)
  (E_val : E = 9)
  (new_ratio : ∃ x : ℝ, B / (C - x) = 2) : 
  ∃ x : ℝ, x = 1.5 := 
sorry

end clothing_weight_removed_l31_31619


namespace value_of_expression_l31_31027

-- Definitions for the conditions
variables (a b : ℝ)

-- Theorem statement
theorem value_of_expression : (a - 3 * b = 3) → (a + 2 * b - (2 * a - b)) = -3 :=
by
  intro h
  sorry

end value_of_expression_l31_31027


namespace sum_of_midpoints_l31_31904

theorem sum_of_midpoints {a b c : ℝ} (h : a + b + c = 15) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := 
by
  sorry

end sum_of_midpoints_l31_31904


namespace rain_forest_animals_l31_31165

theorem rain_forest_animals (R : ℕ) 
  (h1 : 16 = 3 * R - 5) : R = 7 := 
  by sorry

end rain_forest_animals_l31_31165


namespace find_P_x_l31_31736

noncomputable def P (x : ℝ) : ℝ :=
  (-17 / 3) * x^3 + (68 / 3) * x^2 - (31 / 3) * x - 18

variable (a b c : ℝ)

axiom h1 : a^3 - 4 * a^2 + 2 * a + 3 = 0
axiom h2 : b^3 - 4 * b^2 + 2 * b + 3 = 0
axiom h3 : c^3 - 4 * c^2 + 2 * c + 3 = 0

axiom h4 : P a = b + c
axiom h5 : P b = a + c
axiom h6 : P c = a + b
axiom h7 : a + b + c = 4
axiom h8 : P 4 = -20

theorem find_P_x :
  P x = (-17 / 3) * x^3 + (68 / 3) * x^2 - (31 / 3) * x - 18 := sorry

end find_P_x_l31_31736


namespace minimum_bottles_needed_l31_31765

theorem minimum_bottles_needed (medium_volume jumbo_volume : ℕ) (h_medium : medium_volume = 120) (h_jumbo : jumbo_volume = 2000) : 
  let minimum_bottles := (jumbo_volume + medium_volume - 1) / medium_volume
  minimum_bottles = 17 :=
by
  sorry

end minimum_bottles_needed_l31_31765


namespace sector_area_is_correct_l31_31104

noncomputable def area_of_sector (r : ℝ) (α : ℝ) : ℝ := 1/2 * α * r^2

theorem sector_area_is_correct (circumference : ℝ) (central_angle : ℝ) (r : ℝ) (area : ℝ) 
  (h1 : circumference = 8) 
  (h2 : central_angle = 2) 
  (h3 : circumference = central_angle * r + 2 * r)
  (h4 : r = 2) : area = 4 :=
by
  have h5: area = 1/2 * central_angle * r^2 := sorry
  exact sorry

end sector_area_is_correct_l31_31104


namespace percent_increase_decrease_condition_l31_31779

theorem percent_increase_decrease_condition (p q M : ℝ) (hp : 0 < p) (hq : 0 < q) (hM : 0 < M) (hq50 : q < 50) :
  (M * (1 + p / 100) * (1 - q / 100) < M) ↔ (p < 100 * q / (100 - q)) := 
sorry

end percent_increase_decrease_condition_l31_31779


namespace greatest_value_b_l31_31299

-- Define the polynomial and the inequality condition
def polynomial (b : ℝ) : ℝ := -b^2 + 8*b - 12
#check polynomial
-- State the main theorem with the given condition and the result
theorem greatest_value_b (b : ℝ) : -b^2 + 8*b - 12 ≥ 0 → b ≤ 6 :=
sorry

end greatest_value_b_l31_31299


namespace arithmetic_sequence_sum_l31_31859

noncomputable def sum_of_first_n_terms (n : ℕ) (a d : ℝ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem arithmetic_sequence_sum 
  (a_n : ℕ → ℝ) 
  (h_arith : ∃ d, ∀ n, a_n (n + 1) = a_n n + d) 
  (h1 : a_n 1 + a_n 2 + a_n 3 = 3 )
  (h2 : a_n 28 + a_n 29 + a_n 30 = 165 ) 
  : sum_of_first_n_terms 30 (a_n 1) (a_n 2 - a_n 1) = 840 := 
  sorry

end arithmetic_sequence_sum_l31_31859


namespace problem_I_problem_II_l31_31648

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4 * a * x + 1
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 6 * a^2 * Real.log x + 2 * b + 1
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x + g a b x

theorem problem_I (a : ℝ) (ha : a > 0) :
  ∃ b, b = 5 / 2 * a^2 - 3 * a^2 * Real.log a ∧ ∀ b', b' ≤ 3 / 2 * Real.exp (2 / 3) :=
sorry

theorem problem_II (a x₁ x₂ : ℝ) (ha : a ≥ Real.sqrt 3 - 1) (hx : 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂) :
  (h a b x₂ - h a b x₁) / (x₂ - x₁) > 8 :=
sorry

end problem_I_problem_II_l31_31648


namespace total_money_l31_31105

-- Definitions for the conditions
def Cecil_money : ℕ := 600
def twice_Cecil_money : ℕ := 2 * Cecil_money
def Catherine_money : ℕ := twice_Cecil_money - 250
def Carmela_money : ℕ := twice_Cecil_money + 50

-- Theorem statement to prove
theorem total_money : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- sorry is used since no proof is required.
  sorry

end total_money_l31_31105


namespace cyclic_inequality_l31_31307

theorem cyclic_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y := 
by
  sorry

end cyclic_inequality_l31_31307


namespace B_and_C_complementary_l31_31366

def EventA (selected : List String) : Prop :=
  selected.count "boy" = 1

def EventB (selected : List String) : Prop :=
  selected.count "boy" ≥ 1

def EventC (selected : List String) : Prop :=
  selected.count "girl" = 2

theorem B_and_C_complementary :
  ∀ selected : List String,
    (selected.length = 2 ∧ (EventB selected ∨ EventC selected)) ∧ 
    (¬ (EventB selected ∧ EventC selected)) →
    (EventB selected → ¬ EventC selected) ∧ (EventC selected → ¬ EventB selected) :=
  sorry

end B_and_C_complementary_l31_31366


namespace train_crossing_time_l31_31159

theorem train_crossing_time
    (train_speed_kmph : ℕ)
    (platform_length_meters : ℕ)
    (crossing_time_platform_seconds : ℕ)
    (crossing_time_man_seconds : ℕ)
    (train_speed_mps : ℤ)
    (train_length_meters : ℤ)
    (T : ℤ)
    (h1 : train_speed_kmph = 72)
    (h2 : platform_length_meters = 340)
    (h3 : crossing_time_platform_seconds = 35)
    (h4 : train_speed_mps = 20)
    (h5 : train_length_meters = 360)
    (h6 : train_length_meters = train_speed_mps * crossing_time_man_seconds)
    : T = 18 :=
by
  sorry

end train_crossing_time_l31_31159


namespace hexagon_shaded_area_l31_31635

-- Given conditions
variable (A B C D T : ℝ)
variable (h₁ : A = 2)
variable (h₂ : B = 3)
variable (h₃ : C = 4)
variable (h₄ : T = 20)
variable (h₅ : A + B + C + D = T)

-- The goal is to prove that the area of the shaded region (D) is 11 cm².
theorem hexagon_shaded_area : D = 11 := by
  sorry

end hexagon_shaded_area_l31_31635


namespace perpendicular_bisector_eq_l31_31947

theorem perpendicular_bisector_eq (x y : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2 * x - 5 = 0 ∧ x^2 + y^2 + 2 * x - 4 * y - 4 = 0) →
  x + y - 1 = 0 :=
by
  sorry

end perpendicular_bisector_eq_l31_31947


namespace max_value_of_expr_l31_31926

theorem max_value_of_expr 
  (x y z : ℝ) 
  (h₀ : 0 < x) 
  (h₁ : 0 < y) 
  (h₂ : 0 < z)
  (h : x^2 + y^2 + z^2 = 1) : 
  3 * x * y + y * z ≤ (Real.sqrt 10) / 2 := 
  sorry

end max_value_of_expr_l31_31926


namespace find_f_20_l31_31457

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_20 :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = f (2 - x)) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x - 1 / 2) →
  f 20 = - 1 / 2 :=
sorry

end find_f_20_l31_31457


namespace average_distance_one_hour_l31_31022

theorem average_distance_one_hour (d : ℝ) (t : ℝ) (h1 : d = 100) (h2 : t = 5 / 4) : (d / t) = 80 :=
by
  sorry

end average_distance_one_hour_l31_31022


namespace investment_amount_l31_31493

-- Conditions and given problem rewrite in Lean 4
theorem investment_amount (P y : ℝ) (h1 : P * y * 2 / 100 = 500) (h2 : P * (1 + y / 100) ^ 2 - P = 512.50) : P = 5000 :=
sorry

end investment_amount_l31_31493


namespace fraction_multiplication_division_l31_31174

theorem fraction_multiplication_division :
  ((3 / 4) * (5 / 6)) / (7 / 8) = 5 / 7 :=
by
  sorry

end fraction_multiplication_division_l31_31174


namespace initial_gift_card_value_l31_31419

-- The price per pound of coffee
def cost_per_pound : ℝ := 8.58

-- The number of pounds of coffee bought by Rita
def pounds_bought : ℝ := 4.0

-- The remaining balance on Rita's gift card after buying coffee
def remaining_balance : ℝ := 35.68

-- The total cost of the coffee Rita bought
def total_cost_of_coffee : ℝ := cost_per_pound * pounds_bought

-- The initial value of Rita's gift card
def initial_value_of_gift_card : ℝ := total_cost_of_coffee + remaining_balance

-- Statement of the proof problem
theorem initial_gift_card_value : initial_value_of_gift_card = 70.00 :=
by
  -- Placeholder for the proof
  sorry

end initial_gift_card_value_l31_31419


namespace a_n_formula_b_n_formula_l31_31478

namespace SequenceFormulas

theorem a_n_formula (n : ℕ) (h_pos : 0 < n) : 
  (∃ S : ℕ → ℕ, S n = 2 * n^2 + 2 * n) → ∃ a : ℕ → ℕ, a n = 4 * n :=
by
  sorry

theorem b_n_formula (n : ℕ) (h_pos : 0 < n) : 
  (∃ T : ℕ → ℕ, T n = 2 - (if n > 1 then T (n-1) else 1)) → ∃ b : ℕ → ℝ, b n = (1/2)^(n-1) :=
by
  sorry

end SequenceFormulas


end a_n_formula_b_n_formula_l31_31478


namespace change_factor_l31_31857

theorem change_factor (n : ℕ) (avg_original avg_new : ℕ) (F : ℝ)
  (h1 : n = 10) (h2 : avg_original = 80) (h3 : avg_new = 160) 
  (h4 : F * (n * avg_original) = n * avg_new) :
  F = 2 :=
by
  sorry

end change_factor_l31_31857


namespace compute_expression_l31_31595

theorem compute_expression : 7^2 - 2 * 6 + (3^2 - 1) = 45 :=
by
  sorry

end compute_expression_l31_31595


namespace man_to_son_age_ratio_l31_31692

-- Definitions based on conditions
variable (son_age : ℕ) (man_age : ℕ)
variable (h1 : man_age = son_age + 18) -- The man is 18 years older than his son
variable (h2 : 2 * (son_age + 2) = man_age + 2) -- In two years, the man's age will be a multiple of the son's age
variable (h3 : son_age = 16) -- The present age of the son is 16

-- Theorem statement to prove the desired ratio
theorem man_to_son_age_ratio (son_age man_age : ℕ) (h1 : man_age = son_age + 18) (h2 : 2 * (son_age + 2) = man_age + 2) (h3 : son_age = 16) :
  (man_age + 2) / (son_age + 2) = 2 :=
by
  sorry

end man_to_son_age_ratio_l31_31692


namespace solve_abs_eq_2005_l31_31553

theorem solve_abs_eq_2005 (x : ℝ) : |2005 * x - 2005| = 2005 ↔ x = 0 ∨ x = 2 := by
  sorry

end solve_abs_eq_2005_l31_31553


namespace maximize_profit_l31_31408

def revenue (x : ℝ) : ℝ := 17 * x^2
def cost (x : ℝ) : ℝ := 2 * x^3 - x^2
def profit (x : ℝ) : ℝ := revenue x - cost x

theorem maximize_profit : ∃ x > 0, profit x = 18 * x^2 - 2 * x^3 ∧ (∀ y > 0, y ≠ x → profit y < profit x) :=
by
  sorry

end maximize_profit_l31_31408


namespace keith_remaining_cards_l31_31229

-- Definitions and conditions
def initial_cards := 0
def new_cards := 8
def total_cards_after_purchase := initial_cards + new_cards
def remaining_cards := total_cards_after_purchase / 2

-- Proof statement (in Lean, the following would be a theorem)
theorem keith_remaining_cards : remaining_cards = 4 := sorry

end keith_remaining_cards_l31_31229


namespace invalid_transformation_of_equation_l31_31795

theorem invalid_transformation_of_equation (x y m : ℝ) (h : x = y) :
  (m = 0 → (x = y → x / m = y / m)) = false :=
by
  sorry

end invalid_transformation_of_equation_l31_31795


namespace right_triangle_area_l31_31036

-- Define the initial lengths and the area calculation function.
def area_right_triangle (base height : ℕ) : ℕ :=
  (1 / 2) * base * height

theorem right_triangle_area
  (a : ℕ) (b : ℕ) (c : ℕ)
  (h1 : a = 18)
  (h2 : b = 24)
  (h3 : c = 30)  -- Derived from the solution steps
  (h4 : a ^ 2 + b ^ 2 = c ^ 2) :
  area_right_triangle a b = 216 :=
sorry

end right_triangle_area_l31_31036


namespace relationship_between_a_and_b_l31_31032

theorem relationship_between_a_and_b 
  (a b : ℝ) 
  (h1 : |Real.log (1 / 4) / Real.log a| = Real.log (1 / 4) / Real.log a)
  (h2 : |Real.log a / Real.log b| = -Real.log a / Real.log b) :
  0 < a ∧ a < 1 ∧ 1 < b :=
  sorry

end relationship_between_a_and_b_l31_31032


namespace factorize_polynomial_l31_31270

theorem factorize_polynomial (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := 
by sorry

end factorize_polynomial_l31_31270


namespace angles_in_first_or_third_quadrant_l31_31085

noncomputable def angles_first_quadrant_set : Set ℝ :=
  {α | ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + (Real.pi / 2))}

noncomputable def angles_third_quadrant_set : Set ℝ :=
  {α | ∃ k : ℤ, (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 * Real.pi / 2))}

theorem angles_in_first_or_third_quadrant :
  ∃ S1 S2 : Set ℝ, 
    (S1 = {α | ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + (Real.pi / 2))}) ∧
    (S2 = {α | ∃ k : ℤ, (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 * Real.pi / 2))}) ∧
    (angles_first_quadrant_set = S1 ∧ angles_third_quadrant_set = S2)
  :=
sorry

end angles_in_first_or_third_quadrant_l31_31085


namespace pq_implications_l31_31289

theorem pq_implications (p q : Prop) (hpq_or : p ∨ q) (hpq_and : p ∧ q) : p ∧ q :=
by
  sorry

end pq_implications_l31_31289


namespace problem_statement_l31_31015

theorem problem_statement 
  (a b c : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = 0) : 
  a^19 + b^19 + c^19 = 0 :=
sorry

end problem_statement_l31_31015


namespace hard_candy_food_colouring_l31_31799

theorem hard_candy_food_colouring :
  (∀ lollipop_colour hard_candy_count total_food_colouring lollipop_count hard_candy_food_total_per_lollipop,
    lollipop_colour = 5 →
    lollipop_count = 100 →
    hard_candy_count = 5 →
    total_food_colouring = 600 →
    hard_candy_food_total_per_lollipop = lollipop_colour * lollipop_count →
    total_food_colouring - hard_candy_food_total_per_lollipop = hard_candy_count * hard_candy_food_total_per_candy →
    hard_candy_food_total_per_candy = 20) :=
by
  sorry

end hard_candy_food_colouring_l31_31799


namespace youngest_child_age_l31_31167

theorem youngest_child_age {x : ℝ} (h : x + (x + 1) + (x + 2) + (x + 3) = 12) : x = 1.5 :=
by
  sorry

end youngest_child_age_l31_31167


namespace dot_product_of_vectors_l31_31164

theorem dot_product_of_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-1, 2)
  a.1 * b.1 + a.2 * b.2 = -4 :=
by
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-1, 2)
  sorry

end dot_product_of_vectors_l31_31164


namespace chloe_boxes_l31_31667

/-- Chloe was unboxing some of her old winter clothes. She found some boxes of clothing and
inside each box, there were 2 scarves and 6 mittens. Chloe had a total of 32 pieces of
winter clothing. How many boxes of clothing did Chloe find? -/
theorem chloe_boxes (boxes : ℕ) (total_clothing : ℕ) (pieces_per_box : ℕ) :
  pieces_per_box = 8 -> total_clothing = 32 -> total_clothing / pieces_per_box = boxes -> boxes = 4 :=
by
  intros
  sorry

end chloe_boxes_l31_31667


namespace Yuna_boarding_place_l31_31221

-- Conditions
def Eunji_place : ℕ := 10
def people_after_Eunji : ℕ := 11

-- Proof Problem: Yuna's boarding place calculation
theorem Yuna_boarding_place :
  Eunji_place + people_after_Eunji + 1 = 22 :=
by
  sorry

end Yuna_boarding_place_l31_31221


namespace hyperbola_asymptotes_and_point_l31_31123

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 2 = 1

theorem hyperbola_asymptotes_and_point 
  (x y : ℝ)
  (asymptote1 : ∀ x, y = (1/2) * x)
  (asymptote2 : ∀ x, y = (-1/2) * x)
  (point : (x, y) = (4, Real.sqrt 2))
: hyperbola_equation x y :=
sorry

end hyperbola_asymptotes_and_point_l31_31123


namespace amy_remaining_money_l31_31963

-- Definitions based on conditions
def initial_money : ℕ := 100
def doll_cost : ℕ := 1
def number_of_dolls : ℕ := 3

-- The theorem we want to prove
theorem amy_remaining_money : initial_money - number_of_dolls * doll_cost = 97 :=
by 
  sorry

end amy_remaining_money_l31_31963


namespace purchase_price_l31_31212

noncomputable def cost_price_after_discount (P : ℝ) : ℝ :=
  0.8 * P + 375

theorem purchase_price {P : ℝ} (h : 1.15 * P = 18400) : cost_price_after_discount P = 13175 := by
  sorry

end purchase_price_l31_31212


namespace wrongly_recorded_height_l31_31661

theorem wrongly_recorded_height 
  (avg_incorrect : ℕ → ℕ → ℕ)
  (avg_correct : ℕ → ℕ → ℕ)
  (boy_count : ℕ)
  (incorrect_avg_height : ℕ) 
  (correct_avg_height : ℕ) 
  (actual_height : ℕ) 
  (correct_total_height : ℕ) 
  (incorrect_total_height: ℕ)
  (x : ℕ) :
  avg_incorrect boy_count incorrect_avg_height = incorrect_total_height →
  avg_correct boy_count correct_avg_height = correct_total_height →
  incorrect_total_height - x + actual_height = correct_total_height →
  x = 176 := 
by 
  intros h1 h2 h3
  sorry

end wrongly_recorded_height_l31_31661


namespace factorize_poly_l31_31808

-- Statement of the problem
theorem factorize_poly (x : ℝ) : x^2 - 3 * x = x * (x - 3) :=
sorry

end factorize_poly_l31_31808


namespace value_subtracted_l31_31584

theorem value_subtracted (n v : ℝ) (h1 : 2 * n - v = -12) (h2 : n = -10.0) : v = -8 :=
by
  sorry

end value_subtracted_l31_31584


namespace closest_points_distance_l31_31516

theorem closest_points_distance :
  let center1 := (2, 2)
  let center2 := (17, 10)
  let radius1 := 2
  let radius2 := 10
  let distance_centers := Nat.sqrt ((center2.1 - center1.1) ^ 2 + (center2.2 - center1.2) ^ 2)
  distance_centers = 17 → (distance_centers - radius1 - radius2) = 5 := by
  sorry

end closest_points_distance_l31_31516


namespace arithmetic_sequence_ratio_l31_31263

theorem arithmetic_sequence_ratio
  (x y a1 a2 a3 b1 b2 b3 b4 : ℝ)
  (h1 : x ≠ y)
  (h2 : a1 = x + (1 * (a2 - a1)))
  (h3 : a2 = x + (2 * (a2 - a1)))
  (h4 : a3 = x + (3 * (a2 - a1)))
  (h5 : y = x + (4 * (a2 - a1)))
  (h6 : x = x)
  (h7 : b2 = x + (1 * (b3 - x)))
  (h8 : b3 = x + (2 * (b3 - x)))
  (h9 : y = x + (3 * (b3 - x)))
  (h10 : b4 = x + (4 * (b3 - x))) :
  (b4 - b3) / (a2 - a1) = 8 / 3 := by
  sorry

end arithmetic_sequence_ratio_l31_31263


namespace cannot_determine_right_triangle_l31_31367

-- Define what a right triangle is
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define the conditions
def condition_A (A B C : ℕ) : Prop :=
  A / B = 3 / 4 ∧ A / C = 3 / 5 ∧ B / C = 4 / 5

def condition_B (a b c : ℕ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13

def condition_C (A B C : ℕ) : Prop :=
  A - B = C

def condition_D (a b c : ℕ) : Prop :=
  a^2 = b^2 - c^2

-- Define the problem in Lean
theorem cannot_determine_right_triangle :
  (∃ A B C, condition_A A B C → ¬is_right_triangle A B C) ∧
  (∀ (a b c : ℕ), condition_B a b c → is_right_triangle a b c) ∧
  (∀ A B C, condition_C A B C → A = 90) ∧
  (∀ (a b c : ℕ),  condition_D a b c → is_right_triangle a b c)
:=
by sorry

end cannot_determine_right_triangle_l31_31367


namespace problem_1_2_a_problem_1_2_b_l31_31191

theorem problem_1_2_a (x : ℝ) : x * (1 - x) ≤ 1 / 4 := sorry

theorem problem_1_2_b (x a : ℝ) : x * (a - x) ≤ a^2 / 4 := sorry

end problem_1_2_a_problem_1_2_b_l31_31191


namespace place_signs_correct_l31_31204

theorem place_signs_correct :
  1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99 :=
by
  sorry

end place_signs_correct_l31_31204


namespace tammy_total_distance_l31_31962

-- Define the times and speeds for each segment and breaks
def initial_speed : ℝ := 55   -- miles per hour
def initial_time : ℝ := 2     -- hours
def road_speed : ℝ := 40      -- miles per hour
def road_time : ℝ := 5        -- hours
def first_break : ℝ := 1      -- hour
def drive_after_break_speed : ℝ := 50  -- miles per hour
def drive_after_break_time : ℝ := 15   -- hours
def hilly_speed : ℝ := 35     -- miles per hour
def hilly_time : ℝ := 3       -- hours
def second_break : ℝ := 0.5   -- hours
def finish_speed : ℝ := 60    -- miles per hour
def total_journey_time : ℝ := 36 -- hours

-- Define a function to calculate the segment distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Define the total distance calculation
def total_distance : ℝ :=
  distance initial_speed initial_time +
  distance road_speed road_time +
  distance drive_after_break_speed drive_after_break_time +
  distance hilly_speed hilly_time +
  distance finish_speed (total_journey_time - (initial_time + road_time + drive_after_break_time + hilly_time + first_break + second_break))

-- The final proof statement
theorem tammy_total_distance : total_distance = 1735 :=
  sorry

end tammy_total_distance_l31_31962


namespace base_conversion_l31_31656

theorem base_conversion (x : ℕ) (h : 4 * x + 7 = 71) : x = 16 := 
by {
  sorry
}

end base_conversion_l31_31656


namespace rows_before_change_l31_31749

-- Definitions and conditions
variables {r c : ℕ}

-- The total number of tiles before and after the change
def total_tiles_before (r c : ℕ) := r * c = 30
def total_tiles_after (r c : ℕ) := (r + 4) * (c - 2) = 30

-- Prove that the number of rows before the change is 3
theorem rows_before_change (h1 : total_tiles_before r c) (h2 : total_tiles_after r c) : r = 3 := 
sorry

end rows_before_change_l31_31749


namespace seq_a_ge_two_pow_nine_nine_l31_31918

theorem seq_a_ge_two_pow_nine_nine (a : ℕ → ℤ) 
  (h0 : a 1 > a 0)
  (h1 : a 1 > 0)
  (h2 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2^99 :=
sorry

end seq_a_ge_two_pow_nine_nine_l31_31918


namespace geometric_series_sum_l31_31246

/-- 
The series is given as 1/2^2 + 1/2^3 + 1/2^4 + 1/2^5 + 1/2^6 + 1/2^7 + 1/2^8.
First term a = 1/4 and common ratio r = 1/2 and number of terms n = 7. 
The sum should be 127/256.
-/
theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 2
  let n := 7
  let S := (a * (1 - r^n)) / (1 - r)
  S = 127 / 256 :=
by
  sorry

end geometric_series_sum_l31_31246


namespace g_25_eq_zero_l31_31238

noncomputable def g : ℝ → ℝ := sorry

axiom g_def (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x^2 * g y - y^2 * g x = g (x^2 / y^2)

theorem g_25_eq_zero : g 25 = 0 := by
  sorry

end g_25_eq_zero_l31_31238


namespace members_do_not_play_either_l31_31306

noncomputable def total_members := 30
noncomputable def badminton_players := 16
noncomputable def tennis_players := 19
noncomputable def both_players := 7

theorem members_do_not_play_either : 
  (total_members - (badminton_players + tennis_players - both_players)) = 2 :=
by
  sorry

end members_do_not_play_either_l31_31306


namespace delta_minus2_3_eq_minus14_l31_31845

def delta (a b : Int) : Int := a * b^2 + b + 1

theorem delta_minus2_3_eq_minus14 : delta (-2) 3 = -14 :=
by
  sorry

end delta_minus2_3_eq_minus14_l31_31845


namespace time_spent_on_type_a_l31_31582

theorem time_spent_on_type_a (num_questions : ℕ) 
                             (exam_duration : ℕ)
                             (type_a_count : ℕ)
                             (time_ratio : ℕ)
                             (type_b_count : ℕ)
                             (x : ℕ)
                             (total_time : ℕ) :
  num_questions = 200 ∧
  exam_duration = 180 ∧
  type_a_count = 20 ∧
  time_ratio = 2 ∧
  type_b_count = 180 ∧
  total_time = 36 →
  time_ratio * x * type_a_count + x * type_b_count = exam_duration →
  total_time = 36 :=
by
  sorry

end time_spent_on_type_a_l31_31582


namespace minimum_quadratic_value_l31_31759

theorem minimum_quadratic_value (h : ℝ) (x : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → (x - h)^2 + 1 ≥ 10) ∧ (∃ x, 1 ≤ x ∧ x ≤ 3 ∧ (x - h)^2 + 1 = 10) 
  ↔ h = -2 ∨ h = 6 :=
by
  sorry

end minimum_quadratic_value_l31_31759


namespace angle_y_is_80_l31_31673

def parallel (m n : ℝ) : Prop := sorry

def angle_at_base (θ : ℝ) := θ = 40
def right_angle (θ : ℝ) := θ = 90
def exterior_angle (θ1 θ2 : ℝ) := θ1 + θ2 = 180

theorem angle_y_is_80 (m n : ℝ) (θ1 θ2 θ3 θ_ext : ℝ) :
  parallel m n →
  angle_at_base θ1 →
  right_angle θ2 →
  angle_at_base θ3 →
  exterior_angle θ_ext θ3 →
  θ_ext = 80 := by
  sorry

end angle_y_is_80_l31_31673


namespace round_trip_in_first_trip_l31_31762

def percentage_rt_trip_first_trip := 0.3 -- 30%
def percentage_2t_trip_second_trip := 0.6 -- 60%
def percentage_ow_trip_third_trip := 0.45 -- 45%

theorem round_trip_in_first_trip (P1 P2 P3: ℝ) (C1 C2 C3: ℝ) 
  (h1 : P1 = 0.3) 
  (h2 : 0 < P1 ∧ P1 < 1) 
  (h3 : P2 = 0.6) 
  (h4 : 0 < P2 ∧ P2 < 1) 
  (h5 : P3 = 0.45) 
  (h6 : 0 < P3 ∧ P3 < 1) 
  (h7 : C1 + C2 + C3 = 1) 
  (h8 : (C1 = (1 - P1) * 0.15)) 
  (h9 : C2 = 0.2 * P2) 
  (h10 : C3 = 0.1 * P3) :
  P1 = 0.3 := by
  sorry

end round_trip_in_first_trip_l31_31762


namespace overtaking_time_l31_31412

theorem overtaking_time :
  ∀ t t_k : ℕ,
  (30 * t = 40 * (t - 5)) ∧ 
  (30 * t = 60 * t_k) →
  t = 20 ∧ t_k = 10 ∧ (20 - 10 = 10) :=
by
  sorry

end overtaking_time_l31_31412


namespace triangle_inequality_l31_31467

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_l31_31467


namespace commute_time_l31_31116

theorem commute_time (start_time : ℕ) (first_station_time : ℕ) (work_time : ℕ) 
  (h1 : start_time = 6 * 60) 
  (h2 : first_station_time = 40) 
  (h3 : work_time = 9 * 60) : 
  work_time - (start_time + first_station_time) = 140 :=
by
  sorry

end commute_time_l31_31116


namespace fewer_females_than_males_l31_31100

theorem fewer_females_than_males 
  (total_students : ℕ)
  (female_students : ℕ)
  (h_total : total_students = 280)
  (h_female : female_students = 127) :
  total_students - female_students - female_students = 26 := by
  sorry

end fewer_females_than_males_l31_31100


namespace claire_crafting_hours_l31_31060

theorem claire_crafting_hours (H1 : 24 = 24) (H2 : 8 = 8) (H3 : 4 = 4) (H4 : 2 = 2):
  let total_hours_per_day := 24
  let sleep_hours := 8
  let cleaning_hours := 4
  let cooking_hours := 2
  let working_hours := total_hours_per_day - sleep_hours
  let remaining_hours := working_hours - (cleaning_hours + cooking_hours)
  let crafting_hours := remaining_hours / 2
  crafting_hours = 5 :=
by
  sorry

end claire_crafting_hours_l31_31060


namespace three_pumps_drain_time_l31_31540

-- Definitions of the rates of each pump
def rate1 := 1 / 9
def rate2 := 1 / 6
def rate3 := 1 / 12

-- Combined rate of all three pumps working together
def combined_rate := rate1 + rate2 + rate3

-- Time to drain the lake with all three pumps working together
def time_to_drain := 1 / combined_rate

-- Theorem: The time it takes for three pumps working together to drain the lake is 36/13 hours
theorem three_pumps_drain_time : time_to_drain = 36 / 13 := by
  sorry

end three_pumps_drain_time_l31_31540


namespace pairwise_sums_modulo_l31_31792

theorem pairwise_sums_modulo (n : ℕ) (h : n = 2011) :
  ∃ (sums_div_3 sums_rem_1 : ℕ),
  (sums_div_3 = (n * (n - 1)) / 6) ∧
  (sums_rem_1 = (n * (n - 1)) / 6) := by
  sorry

end pairwise_sums_modulo_l31_31792


namespace solve_for_x_l31_31395

theorem solve_for_x (x : ℚ) (h : 2 / 3 + 1 / x = 7 / 9) : x = 9 :=
sorry

end solve_for_x_l31_31395


namespace rice_mixture_ratio_l31_31687

theorem rice_mixture_ratio (x y : ℝ) (h1 : 7 * x + 8.75 * y = 7.50 * (x + y)) : x / y = 2.5 :=
by
  sorry

end rice_mixture_ratio_l31_31687


namespace binomial_60_3_eq_34220_l31_31997

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l31_31997


namespace find_joe_age_l31_31317

noncomputable def billy_age (joe_age : ℕ) : ℕ := 3 * joe_age
noncomputable def emily_age (billy_age joe_age : ℕ) : ℕ := (billy_age + joe_age) / 2

theorem find_joe_age (joe_age : ℕ) 
    (h1 : billy_age joe_age = 3 * joe_age)
    (h2 : emily_age (billy_age joe_age) joe_age = (billy_age joe_age + joe_age) / 2)
    (h3 : billy_age joe_age + joe_age + emily_age (billy_age joe_age) joe_age = 90) : 
    joe_age = 15 :=
by
  sorry

end find_joe_age_l31_31317


namespace amount_to_add_l31_31939

theorem amount_to_add (students : ℕ) (total_cost : ℕ) (h1 : students = 9) (h2 : total_cost = 143) : 
  ∃ k : ℕ, total_cost + k = students * (total_cost / students + 1) :=
by
  sorry

end amount_to_add_l31_31939


namespace recorded_expenditure_l31_31563

-- Define what it means to record an income and an expenditure
def record_income (y : ℝ) : ℝ := y
def record_expenditure (y : ℝ) : ℝ := -y

-- Define specific instances for the problem
def income_recorded_as : ℝ := 20
def expenditure_value : ℝ := 75

-- Given condition
axiom income_condition : record_income income_recorded_as = 20

-- Theorem to prove the recorded expenditure
theorem recorded_expenditure : record_expenditure expenditure_value = -75 := by
  sorry

end recorded_expenditure_l31_31563


namespace tailor_cut_difference_l31_31949

def skirt_cut : ℝ := 0.75
def pants_cut : ℝ := 0.5

theorem tailor_cut_difference : skirt_cut - pants_cut = 0.25 :=
by
  sorry

end tailor_cut_difference_l31_31949


namespace rbcmul_div7_div89_l31_31828

theorem rbcmul_div7_div89 {r b c : ℕ} (h : (523000 + 100 * r + 10 * b + c) % 7 = 0 ∧ (523000 + 100 * r + 10 * b + c) % 89 = 0) :
  r * b * c = 36 :=
by
  sorry

end rbcmul_div7_div89_l31_31828


namespace largest_n_polynomials_l31_31465

theorem largest_n_polynomials :
  ∃ (P : ℕ → (ℝ → ℝ)), (∀ i j, i ≠ j → ∀ x, P i x + P j x ≠ 0) ∧ (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ∃ x, P i x + P j x + P k x = 0) ↔ n = 3 := 
sorry

end largest_n_polynomials_l31_31465


namespace three_digit_powers_of_two_l31_31313

theorem three_digit_powers_of_two : 
  ∃ (N : ℕ), N = 3 ∧ ∀ (n : ℕ), (100 ≤ 2^n ∧ 2^n < 1000) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
by
  sorry

end three_digit_powers_of_two_l31_31313


namespace colby_mangoes_l31_31048

def mangoes_still_have (t m k : ℕ) : ℕ :=
  let r1 := t - m
  let r2 := r1 / 2
  let r3 := r1 - r2
  r3 * k

theorem colby_mangoes (t m k : ℕ) (h_t : t = 60) (h_m : m = 20) (h_k : k = 8) :
  mangoes_still_have t m k = 160 :=
by
  sorry

end colby_mangoes_l31_31048


namespace addition_of_decimals_l31_31820

theorem addition_of_decimals :
  0.9 + 0.99 = 1.89 :=
by
  sorry

end addition_of_decimals_l31_31820


namespace convert_base8_to_base10_l31_31001

def base8_to_base10 (n : Nat) : Nat := 
  -- Assuming a specific function that converts from base 8 to base 10
  sorry 

theorem convert_base8_to_base10 :
  base8_to_base10 5624 = 2964 :=
by
  sorry

end convert_base8_to_base10_l31_31001


namespace max_inscribed_triangle_area_l31_31666

theorem max_inscribed_triangle_area (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ A, A = (3 * Real.sqrt 3 / 4) * a * b := 
sorry

end max_inscribed_triangle_area_l31_31666


namespace series_sum_eq_negative_one_third_l31_31508

noncomputable def series_sum : ℝ :=
  ∑' n, (2 * n + 1) / (n * (n + 1) * (n + 2) * (n + 3))

theorem series_sum_eq_negative_one_third : series_sum = -1 / 3 := sorry

end series_sum_eq_negative_one_third_l31_31508


namespace cos_theta_four_times_l31_31379

theorem cos_theta_four_times (theta : ℝ) (h : Real.cos theta = 1 / 3) : 
  Real.cos (4 * theta) = 17 / 81 := 
sorry

end cos_theta_four_times_l31_31379


namespace hyperbola_properties_l31_31973

theorem hyperbola_properties :
  (∃ x y : Real,
    (x^2 / 4 - y^2 / 2 = 1) ∧
    (∃ a b c e : Real,
      2 * a = 4 ∧
      2 * b = 2 * Real.sqrt 2 ∧
      c = Real.sqrt (a^2 + b^2) ∧
      2 * c = 2 * Real.sqrt 6 ∧
      e = c / a)) :=
by
  sorry

end hyperbola_properties_l31_31973


namespace tip_percentage_l31_31324

theorem tip_percentage (T : ℝ) 
  (total_cost meal_cost sales_tax : ℝ)
  (h1 : meal_cost = 61.48)
  (h2 : sales_tax = 0.07 * meal_cost)
  (h3 : total_cost = meal_cost + sales_tax + T * meal_cost)
  (h4 : total_cost ≤ 75) :
  T ≤ 0.1499 :=
by
  -- main proof goes here
  sorry

end tip_percentage_l31_31324


namespace total_pencils_is_54_l31_31497

def total_pencils (m a : ℕ) : ℕ :=
  m + a

theorem total_pencils_is_54 : 
  ∃ (m a : ℕ), (m = 30) ∧ (m = a + 6) ∧ total_pencils m a = 54 :=
by
  sorry

end total_pencils_is_54_l31_31497


namespace fixed_point_difference_l31_31326

noncomputable def func (a x : ℝ) : ℝ := a^x + Real.log a

theorem fixed_point_difference (a m n : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (func a 0 = n) ∧ (y = func a x → (x = m) ∧ (y = n)) → (m - n = -2) :=
by 
  intro h
  sorry

end fixed_point_difference_l31_31326


namespace range_of_a_sq_l31_31803

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ m n : ℕ, a (m + n) = a m + a n

theorem range_of_a_sq {n : ℕ}
  (h_arith : arithmetic_sequence a)
  (h_cond : a 1 ^ 2 + a (2 * n + 1) ^ 2 = 1) :
  ∃ (L R : ℝ), (L = 2) ∧ (∀ k : ℕ, a (n+1) ^ 2 + a (3*n+1) ^ 2 ≥ L) := sorry

end range_of_a_sq_l31_31803


namespace count_integers_abs_inequality_l31_31322

theorem count_integers_abs_inequality : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℤ, |(x: ℝ) - 3| ≤ 7.2 ↔ x ∈ {i : ℤ | -4 ≤ i ∧ i ≤ 10} := 
by 
  sorry

end count_integers_abs_inequality_l31_31322


namespace total_time_last_two_videos_l31_31218

theorem total_time_last_two_videos
  (first_video_length : ℕ := 2 * 60)
  (second_video_length : ℕ := 4 * 60 + 30)
  (total_time : ℕ := 510) :
  ∃ t1 t2 : ℕ, t1 ≠ t2 ∧ t1 + t2 = total_time - first_video_length - second_video_length := by
  sorry

end total_time_last_two_videos_l31_31218


namespace eight_lines_no_parallel_no_concurrent_l31_31767

-- Define the number of regions into which n lines divide the plane
def regions (n : ℕ) : ℕ :=
if n = 0 then 1
else if n = 1 then 2
else n * (n - 1) / 2 + n + 1

theorem eight_lines_no_parallel_no_concurrent :
  regions 8 = 37 :=
by
  sorry

end eight_lines_no_parallel_no_concurrent_l31_31767


namespace maximize_profit_l31_31267

theorem maximize_profit 
  (cost_per_product : ℝ)
  (initial_price : ℝ)
  (initial_sales : ℝ)
  (price_increase_effect : ℝ)
  (daily_sales_decrease : ℝ)
  (max_profit_price : ℝ)
  (max_profit : ℝ)
  :
  cost_per_product = 8 ∧ initial_price = 10 ∧ initial_sales = 100 ∧ price_increase_effect = 1 ∧ daily_sales_decrease = 10 → 
  max_profit_price = 14 ∧
  max_profit = 360 :=
by 
  intro h
  have h_cost := h.1
  have h_initial_price := h.2.1
  have h_initial_sales := h.2.2.1
  have h_price_increase_effect := h.2.2.2.1
  have h_daily_sales_decrease := h.2.2.2.2
  sorry

end maximize_profit_l31_31267


namespace remaining_black_cards_l31_31868

-- Define the conditions of the problem
def total_cards : ℕ := 52
def colors : ℕ := 2
def cards_per_color := total_cards / colors
def black_cards_taken_out : ℕ := 5
def total_black_cards : ℕ := cards_per_color

-- Prove the remaining black cards
theorem remaining_black_cards : total_black_cards - black_cards_taken_out = 21 := 
by
  -- Logic to calculate remaining black cards
  sorry

end remaining_black_cards_l31_31868


namespace rudy_first_run_rate_l31_31308

def first_run_rate (R : ℝ) : Prop :=
  let time_first_run := 5 * R
  let time_second_run := 4 * 9.5
  let total_time := time_first_run + time_second_run
  total_time = 88

theorem rudy_first_run_rate : first_run_rate 10 :=
by
  unfold first_run_rate
  simp
  sorry

end rudy_first_run_rate_l31_31308


namespace find_base_b_l31_31070

theorem find_base_b : ∃ b : ℕ, (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 3 * b + 1 ∧ b = 7 := 
by {
  sorry
}

end find_base_b_l31_31070


namespace valid_six_digit_numbers_l31_31402

def is_divisible_by_4 (n : Nat) : Prop :=
  n % 4 = 0

def digit_sum (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def is_divisible_by_9 (n : Nat) : Prop :=
  digit_sum n % 9 = 0

def is_valid_six_digit_number (n : Nat) : Prop :=
  ∃ (a b : Nat), n = b * 100000 + 20140 + a ∧ is_divisible_by_4 (10 * 2014 + a) ∧ is_divisible_by_9 (b * 100000 + 20140 + a)

theorem valid_six_digit_numbers :
  { n | is_valid_six_digit_number n } = {220140, 720144, 320148} :=
by
  sorry

end valid_six_digit_numbers_l31_31402


namespace marks_lost_per_wrong_answer_l31_31604

theorem marks_lost_per_wrong_answer (score_per_correct : ℕ) (total_questions : ℕ) 
(total_score : ℕ) (correct_attempts : ℕ) (wrong_attempts : ℕ) (marks_lost_total : ℕ)
(H1 : score_per_correct = 4)
(H2 : total_questions = 75)
(H3 : total_score = 125)
(H4 : correct_attempts = 40)
(H5 : wrong_attempts = total_questions - correct_attempts)
(H6 : marks_lost_total = (correct_attempts * score_per_correct) - total_score)
: (marks_lost_total / wrong_attempts) = 1 := by
  sorry

end marks_lost_per_wrong_answer_l31_31604


namespace simplest_quadratic_radical_l31_31156

noncomputable def optionA := Real.sqrt 7
noncomputable def optionB := Real.sqrt 9
noncomputable def optionC := Real.sqrt 12
noncomputable def optionD := Real.sqrt (2 / 3)

theorem simplest_quadratic_radical :
  optionA = Real.sqrt 7 ∧
  optionB = Real.sqrt 9 ∧
  optionC = Real.sqrt 12 ∧
  optionD = Real.sqrt (2 / 3) ∧
  (optionB = 3 ∧ optionC = 2 * Real.sqrt 3 ∧ optionD = Real.sqrt 6 / 3) ∧
  (optionA < 3 ∧ optionA < 2 * Real.sqrt 3 ∧ optionA < Real.sqrt 6 / 3) :=
  by {
    sorry
  }

end simplest_quadratic_radical_l31_31156


namespace compare_subtract_one_l31_31245

theorem compare_subtract_one (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end compare_subtract_one_l31_31245


namespace line_length_after_erasing_l31_31942

theorem line_length_after_erasing :
  ∀ (initial_length_m : ℕ) (conversion_factor : ℕ) (erased_length_cm : ℕ),
  initial_length_m = 1 → conversion_factor = 100 → erased_length_cm = 33 →
  initial_length_m * conversion_factor - erased_length_cm = 67 :=
by {
  sorry
}

end line_length_after_erasing_l31_31942


namespace oblique_line_plane_angle_range_l31_31602

/-- 
An oblique line intersects the plane at an angle other than a right angle. 
The angle cannot be $0$ radians or $\frac{\pi}{2}$ radians.
-/
theorem oblique_line_plane_angle_range (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) : 
  0 < θ ∧ θ < π / 2 :=
by {
  exact ⟨h₀, h₁⟩
}

end oblique_line_plane_angle_range_l31_31602


namespace find_h_of_root_l31_31383

theorem find_h_of_root :
  ∀ h : ℝ, (-3)^3 + h * (-3) - 10 = 0 → h = -37/3 := by
  sorry

end find_h_of_root_l31_31383


namespace smallest_sum_l31_31186

theorem smallest_sum (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x ≠ y) (eq : 1/x + 1/y = 1/10) : x + y = 45 :=
by
  sorry

end smallest_sum_l31_31186


namespace part1_part2_l31_31778

-- We state the problem conditions and theorems to be proven accordingly
variable (A B C : Real) (a b c : Real)

-- Condition 1: In triangle ABC, opposite sides a, b, c with angles A, B, C such that a sin(B - C) = b sin(A - C)
axiom condition1 (A B C : Real) (a b c : Real) : a * Real.sin (B - C) = b * Real.sin (A - C)

-- Question 1: Prove that a = b under the given conditions
theorem part1 (A B C : Real) (a b c : Real) (h1 : a * Real.sin (B - C) = b * Real.sin (A - C)) : a = b := sorry

-- Condition 2: If c = 5 and cos C = 12/13
axiom condition2 (c : Real) : c = 5
axiom condition3 (C : Real) : Real.cos C = 12 / 13

-- Question 2: Prove that the area of triangle ABC is 125/4 under the given conditions
theorem part2 (A B C : Real) (a b c : Real) (h1 : a * Real.sin (B - C) = b * Real.sin (A - C)) 
               (h2 : c = 5) (h3 : Real.cos C = 12 / 13): (1 / 2) * a * b * (Real.sin C) = 125 / 4 := sorry

end part1_part2_l31_31778


namespace shifted_parabola_expression_l31_31858

theorem shifted_parabola_expression (x : ℝ) :
  let y_original := x^2
  let y_shifted_right := (x - 1)^2
  let y_shifted_up := y_shifted_right + 2
  y_shifted_up = (x - 1)^2 + 2 :=
by
  sorry

end shifted_parabola_expression_l31_31858


namespace max_ab_ac_bc_l31_31120

noncomputable def maxValue (a b c : ℝ) := a * b + a * c + b * c

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : maxValue a b c ≤ 12 :=
by
  sorry

end max_ab_ac_bc_l31_31120


namespace hyperbola_eccentricity_l31_31727

variable {a b : ℝ}
variable (h1 : a > 0) (h2 : b > 0)
variable (h3 : (a : ℝ) / (b : ℝ) = 3)

theorem hyperbola_eccentricity (h1 : a > 0) (h2 : b > 0) (h3 : b / a = 1 / 3) : 
  (Real.sqrt ((a ^ 2 + b ^ 2) / (a ^ 2))) = Real.sqrt 10 := by sorry

end hyperbola_eccentricity_l31_31727


namespace even_of_even_square_sqrt_two_irrational_l31_31468

-- Problem 1: Prove that if \( p^2 \) is even, then \( p \) is even given \( p \in \mathbb{Z} \).
theorem even_of_even_square (p : ℤ) (h : Even (p * p)) : Even p := 
sorry 

-- Problem 2: Prove that \( \sqrt{2} \) is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ Int.gcd a b = 1 ∧ (a : ℝ) / b = Real.sqrt 2 :=
sorry

end even_of_even_square_sqrt_two_irrational_l31_31468


namespace certain_number_exists_l31_31203

theorem certain_number_exists (a b : ℝ) (C : ℝ) (h1 : a ≠ b) (h2 : a + b = 4) (h3 : a * (a - 4) = C) (h4 : b * (b - 4) = C) : 
  C = -3 := 
sorry

end certain_number_exists_l31_31203


namespace polynomial_evaluation_l31_31647

theorem polynomial_evaluation (x y : ℝ) (h : 2 * x^2 + 3 * y + 3 = 8) : 6 * x^2 + 9 * y + 8 = 23 :=
sorry

end polynomial_evaluation_l31_31647


namespace probability_face_not_red_is_five_sixths_l31_31373

-- Definitions based on the conditions
def total_faces : ℕ := 6
def green_faces : ℕ := 3
def blue_faces : ℕ := 2
def red_faces : ℕ := 1

-- Definition for the probability calculation
def probability_not_red (total : ℕ) (not_red : ℕ) : ℚ := not_red / total

-- The main statement to prove
theorem probability_face_not_red_is_five_sixths :
  probability_not_red total_faces (green_faces + blue_faces) = 5 / 6 :=
by sorry

end probability_face_not_red_is_five_sixths_l31_31373


namespace b_bound_for_tangent_parallel_l31_31773

theorem b_bound_for_tangent_parallel (b : ℝ) (c : ℝ) :
  (∃ x : ℝ, 3 * x^2 - x + b = 0) → b ≤ 1/12 :=
by
  intros h
  -- Placeholder proof
  sorry

end b_bound_for_tangent_parallel_l31_31773


namespace minimize_quadratic_sum_l31_31480

theorem minimize_quadratic_sum (a b : ℝ) : 
  ∃ x : ℝ, y = (x-a)^2 + (x-b)^2 ∧ (∀ x', (x'-a)^2 + (x'-b)^2 ≥ y) ∧ x = (a + b) / 2 := 
sorry

end minimize_quadratic_sum_l31_31480


namespace arjun_starting_amount_l31_31107

theorem arjun_starting_amount (X : ℝ) (h1 : Anoop_investment = 4000) (h2 : Anoop_months = 6) (h3 : Arjun_months = 12) (h4 : (X * 12) = (4000 * 6)) :
  X = 2000 :=
sorry

end arjun_starting_amount_l31_31107


namespace total_spent_is_correct_l31_31954

-- Declare the constants for the prices and quantities
def wallet_cost : ℕ := 50
def sneakers_cost_per_pair : ℕ := 100
def sneakers_pairs : ℕ := 2
def backpack_cost : ℕ := 100
def jeans_cost_per_pair : ℕ := 50
def jeans_pairs : ℕ := 2

-- Define the total amounts spent by Leonard and Michael
def leonard_total : ℕ := wallet_cost + sneakers_cost_per_pair * sneakers_pairs
def michael_total : ℕ := backpack_cost + jeans_cost_per_pair * jeans_pairs

-- The total amount spent by Leonard and Michael
def total_spent : ℕ := leonard_total + michael_total

-- The proof statement
theorem total_spent_is_correct : total_spent = 450 :=
by 
  -- This part is where the proof would go
  sorry

end total_spent_is_correct_l31_31954


namespace pairs_satisfying_equation_l31_31751

theorem pairs_satisfying_equation :
  ∀ x y : ℝ, (x ^ 4 + 1) * (y ^ 4 + 1) = 4 * x^2 * y^2 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by
  intros x y
  sorry

end pairs_satisfying_equation_l31_31751


namespace total_formula_portions_l31_31084

def puppies : ℕ := 7
def feedings_per_day : ℕ := 3
def days : ℕ := 5

theorem total_formula_portions : 
  (feedings_per_day * days * puppies = 105) := 
by
  sorry

end total_formula_portions_l31_31084


namespace topic_preference_order_l31_31248

noncomputable def astronomy_fraction := (8 : ℚ) / 21
noncomputable def botany_fraction := (5 : ℚ) / 14
noncomputable def chemistry_fraction := (9 : ℚ) / 28

theorem topic_preference_order :
  (astronomy_fraction > botany_fraction) ∧ (botany_fraction > chemistry_fraction) :=
by
  sorry

end topic_preference_order_l31_31248


namespace geometric_sequence_sum_l31_31208

variable {α : Type*} [NormedField α] [CompleteSpace α]

def geometric_sum (a r : α) (n : ℕ) : α :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (S : ℕ → α) (a r : α) (hS : ∀ n, S n = geometric_sum a r n) :
  S 2 = 6 → S 4 = 30 → S 6 = 126 :=
by
  sorry

end geometric_sequence_sum_l31_31208


namespace neither_sufficient_nor_necessary_l31_31594

noncomputable def a_b_conditions (a b: ℝ) : Prop :=
∃ (a b: ℝ), ¬((a - b > 0) → (a^2 - b^2 > 0)) ∧ ¬((a^2 - b^2 > 0) → (a - b > 0))

theorem neither_sufficient_nor_necessary (a b: ℝ) : a_b_conditions a b :=
sorry

end neither_sufficient_nor_necessary_l31_31594


namespace vector_x_solution_l31_31219

theorem vector_x_solution (x : ℝ) (a b c : ℝ × ℝ)
  (ha : a = (-2,0))
  (hb : b = (2,1))
  (hc : c = (x,1))
  (collinear : ∃ k : ℝ, 3 • a + b = k • c) :
  x = -4 :=
by
  sorry

end vector_x_solution_l31_31219


namespace sum_ak_div_k2_ge_sum_inv_k_l31_31486

open BigOperators

theorem sum_ak_div_k2_ge_sum_inv_k
  (n : ℕ)
  (a : Fin n → ℕ)
  (hpos : ∀ k, 0 < a k)
  (hdist : Function.Injective a) :
  ∑ k : Fin n, (a k : ℝ) / (k + 1 : ℝ)^2 ≥ ∑ k : Fin n, 1 / (k + 1 : ℝ) := sorry

end sum_ak_div_k2_ge_sum_inv_k_l31_31486


namespace distance_between_points_l31_31550

theorem distance_between_points :
  let x1 := 1
  let y1 := 3
  let z1 := 2
  let x2 := 4
  let y2 := 1
  let z2 := 6
  let distance : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
  distance = Real.sqrt 29 := by
  sorry

end distance_between_points_l31_31550


namespace fifth_number_in_tenth_row_l31_31447

def nth_number_in_row (n k : ℕ) : ℕ :=
  7 * n - (7 - k)

theorem fifth_number_in_tenth_row : nth_number_in_row 10 5 = 68 :=
by
  sorry

end fifth_number_in_tenth_row_l31_31447


namespace length_of_chord_l31_31653

theorem length_of_chord (r AB : ℝ) (h1 : r = 6) (h2 : 0 < AB) (h3 : AB <= 2 * r) : AB ≠ 14 :=
by
  sorry

end length_of_chord_l31_31653


namespace correct_system_of_equations_l31_31567

theorem correct_system_of_equations : 
  ∃ (x y : ℕ), x + y = 12 ∧ 4 * x + 3 * y = 40 := by
  -- we are stating the existence of x and y that satisfy both equations given as conditions.
  sorry

end correct_system_of_equations_l31_31567


namespace intersection_eq_l31_31813

noncomputable def A : Set ℝ := { x | x < 2 }
noncomputable def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_eq : A ∩ B = {-1, 0, 1} :=
by
  sorry

end intersection_eq_l31_31813


namespace frustum_smaller_cone_height_l31_31311

theorem frustum_smaller_cone_height (H frustum_height radius1 radius2 : ℝ) 
  (h : ℝ) (h_eq : h = 30 - 18) : 
  radius1 = 6 → radius2 = 10 → frustum_height = 18 → H = 30 → h = 12 := 
by
  intros
  sorry

end frustum_smaller_cone_height_l31_31311


namespace cl_mass_percentage_in_ccl4_l31_31066

noncomputable def mass_percentage_of_cl_in_ccl4 : ℝ :=
  let mass_C : ℝ := 12.01
  let mass_Cl : ℝ := 35.45
  let num_Cl : ℝ := 4
  let total_mass_Cl : ℝ := num_Cl * mass_Cl
  let total_mass_CCl4 : ℝ := mass_C + total_mass_Cl
  (total_mass_Cl / total_mass_CCl4) * 100

theorem cl_mass_percentage_in_ccl4 :
  abs (mass_percentage_of_cl_in_ccl4 - 92.19) < 0.01 := 
sorry

end cl_mass_percentage_in_ccl4_l31_31066


namespace find_positive_number_l31_31151

theorem find_positive_number (x : ℕ) (h_pos : 0 < x) (h_equation : x * x / 100 + 6 = 10) : x = 20 :=
by
  sorry

end find_positive_number_l31_31151


namespace people_per_car_l31_31010

theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (h1 : total_people = 63) (h2 : num_cars = 3) : total_people / num_cars = 21 :=
by
  sorry

end people_per_car_l31_31010


namespace arithmetic_sequence_root_arithmetic_l31_31119

theorem arithmetic_sequence_root_arithmetic (a : ℕ → ℝ) 
  (h_arith : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0) 
  (h_root : ∀ x : ℝ, x^2 + 12 * x - 8 = 0 → (x = a 2 ∨ x = a 10)) : 
  a 6 = -6 := 
by
  -- We skip the proof as per instructions
  sorry

end arithmetic_sequence_root_arithmetic_l31_31119


namespace trees_left_after_typhoon_l31_31605

theorem trees_left_after_typhoon (trees_grown : ℕ) (trees_died : ℕ) (h1 : trees_grown = 17) (h2 : trees_died = 5) : (trees_grown - trees_died = 12) :=
by
  -- The proof would go here
  sorry

end trees_left_after_typhoon_l31_31605


namespace average_minutes_per_player_is_2_l31_31577

def total_player_footage := 130 + 145 + 85 + 60 + 180
def total_additional_content := 120 + 90 + 30
def pause_transition_time := 15 * (5 + 3) -- 5 players + game footage + interviews + opening/closing scenes - 1
def total_film_time := total_player_footage + total_additional_content + pause_transition_time
def number_of_players := 5
def average_seconds_per_player := total_player_footage / number_of_players
def average_minutes_per_player := average_seconds_per_player / 60

theorem average_minutes_per_player_is_2 :
  average_minutes_per_player = 2 := by
  -- Proof goes here.
  sorry

end average_minutes_per_player_is_2_l31_31577


namespace minimal_volume_block_l31_31064

theorem minimal_volume_block (l m n : ℕ) (h : (l - 1) * (m - 1) * (n - 1) = 297) : l * m * n = 192 :=
sorry

end minimal_volume_block_l31_31064


namespace agatha_remaining_amount_l31_31970

theorem agatha_remaining_amount :
  let initial_amount := 60
  let frame_price := 15
  let frame_discount := 0.10 * frame_price
  let frame_final := frame_price - frame_discount
  let wheel_price := 25
  let wheel_discount := 0.05 * wheel_price
  let wheel_final := wheel_price - wheel_discount
  let seat_price := 8
  let seat_discount := 0.15 * seat_price
  let seat_final := seat_price - seat_discount
  let tape_price := 5
  let total_spent := frame_final + wheel_final + seat_final + tape_price
  let remaining_amount := initial_amount - total_spent
  remaining_amount = 10.95 :=
by
  sorry

end agatha_remaining_amount_l31_31970


namespace domain_f_log2_x_to_domain_f_x_l31_31257

variable {f : ℝ → ℝ}

-- Condition: The domain of y = f(log₂ x) is [1/2, 4]
def domain_f_log2_x : Set ℝ := Set.Icc (1 / 2) 4

-- Proof statement
theorem domain_f_log2_x_to_domain_f_x
  (h : ∀ x, x ∈ domain_f_log2_x → f (Real.log x / Real.log 2) = f x) :
  Set.Icc (-1) 2 = {x : ℝ | ∃ y ∈ domain_f_log2_x, Real.log y / Real.log 2 = x} :=
by
  sorry

end domain_f_log2_x_to_domain_f_x_l31_31257


namespace find_coefficients_l31_31243

theorem find_coefficients (a b p q : ℝ) :
    (∀ x : ℝ, (2 * x - 1) ^ 20 - (a * x + b) ^ 20 = (x^2 + p * x + q) ^ 10) →
    a = -2 * b ∧ (b = 1 ∨ b = -1) ∧ p = -1 ∧ q = 1 / 4 :=
by 
    sorry

end find_coefficients_l31_31243


namespace units_digit_of_n_cubed_minus_n_squared_l31_31735

-- Define n for the purpose of the problem
def n : ℕ := 9867

-- Prove that the units digit of n^3 - n^2 is 4
theorem units_digit_of_n_cubed_minus_n_squared : ∃ d : ℕ, d = (n^3 - n^2) % 10 ∧ d = 4 := by
  sorry

end units_digit_of_n_cubed_minus_n_squared_l31_31735


namespace find_quadruples_l31_31781

def is_prime (n : ℕ) := ∀ m, m ∣ n → m = 1 ∨ m = n

 theorem find_quadruples (p q a b : ℕ) (hp : is_prime p) (hq : is_prime q) (ha : 1 < a)
  : (p^a = 1 + 5 * q^b ↔ ((p = 2 ∧ q = 3 ∧ a = 4 ∧ b = 1) ∨ (p = 3 ∧ q = 2 ∧ a = 4 ∧ b = 4))) :=
by {
  sorry
}

end find_quadruples_l31_31781


namespace sacks_per_day_l31_31082

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (harvest_rate : ℕ)
  (h1 : total_sacks = 498)
  (h2 : days = 6)
  (h3 : harvest_rate = total_sacks / days) :
  harvest_rate = 83 := by
  sorry

end sacks_per_day_l31_31082


namespace ryan_lost_initially_l31_31207

-- Define the number of leaves initially collected
def initial_leaves : ℤ := 89

-- Define the number of leaves broken afterwards
def broken_leaves : ℤ := 43

-- Define the number of leaves left in the collection
def remaining_leaves : ℤ := 22

-- Define the lost leaves
def lost_leaves (L : ℤ) : Prop :=
  initial_leaves - L - broken_leaves = remaining_leaves

theorem ryan_lost_initially : ∃ L : ℤ, lost_leaves L ∧ L = 24 :=
by
  sorry

end ryan_lost_initially_l31_31207


namespace polynomial_divisibility_l31_31281

theorem polynomial_divisibility (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
    (h3 : (a + b^3) % (a^2 + 3 * a * b + 3 * b^2 - 1) = 0) : 
    ∃ k : ℕ, k ≥ 1 ∧ (a^2 + 3 * a * b + 3 * b^2 - 1) % k^3 = 0 :=
    sorry

end polynomial_divisibility_l31_31281


namespace coordinates_of_N_l31_31663

theorem coordinates_of_N
  (M : ℝ × ℝ)
  (a : ℝ × ℝ)
  (x y : ℝ)
  (hM : M = (5, -6))
  (ha : a = (1, -2))
  (hMN : (x - M.1, y - M.2) = (-3 * a.1, -3 * a.2)) :
  (x, y) = (2, 0) :=
by
  sorry

end coordinates_of_N_l31_31663


namespace no_coprime_xy_multiple_l31_31738

theorem no_coprime_xy_multiple (n : ℕ) (hn : ∀ d : ℕ, d ∣ n → d^2 ∣ n → d = 1)
  (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (h_coprime : Nat.gcd x y = 1) :
  ¬ ((x^n + y^n) % ((x + y)^3) = 0) :=
by
  sorry

end no_coprime_xy_multiple_l31_31738


namespace mark_weekly_reading_time_l31_31683

-- Define the conditions
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7
def additional_hours : ℕ := 4

-- State the main theorem to prove
theorem mark_weekly_reading_time : (hours_per_day * days_per_week) + additional_hours = 18 := 
by
  -- The proof steps are omitted as per instructions
  sorry

end mark_weekly_reading_time_l31_31683


namespace average_speed_of_trip_is_correct_l31_31118

-- Definitions
def total_distance : ℕ := 450
def distance_part1 : ℕ := 300
def speed_part1 : ℕ := 20
def distance_part2 : ℕ := 150
def speed_part2 : ℕ := 15

-- The average speed problem
theorem average_speed_of_trip_is_correct :
  (total_distance : ℤ) / (distance_part1 / speed_part1 + distance_part2 / speed_part2 : ℤ) = 18 := by
  sorry

end average_speed_of_trip_is_correct_l31_31118


namespace barbara_total_cost_l31_31625

-- Define conditions
def steak_weight : ℝ := 4.5
def steak_price_per_pound : ℝ := 15.0
def chicken_weight : ℝ := 1.5
def chicken_price_per_pound : ℝ := 8.0

-- Define total cost formula
def total_cost := (steak_weight * steak_price_per_pound) + (chicken_weight * chicken_price_per_pound)

-- Prove that the total cost equals $79.50
theorem barbara_total_cost : total_cost = 79.50 := by
  sorry

end barbara_total_cost_l31_31625


namespace shorten_ellipse_parametric_form_l31_31794

theorem shorten_ellipse_parametric_form :
  ∀ (θ : ℝ), 
  ∃ (x' y' : ℝ),
    x' = 4 * Real.cos θ ∧ y' = 2 * Real.sin θ ∧
    (∃ (x y : ℝ),
      x' = 2 * x ∧ y' = y ∧
      x = 2 * Real.cos θ ∧ y = 2 * Real.sin θ) :=
by
  sorry

end shorten_ellipse_parametric_form_l31_31794


namespace expand_expression_l31_31830

theorem expand_expression (x : ℝ) : (x + 3) * (2 * x ^ 2 - x + 4) = 2 * x ^ 3 + 5 * x ^ 2 + x + 12 :=
by
  sorry

end expand_expression_l31_31830


namespace average_speed_of_Car_X_l31_31103

noncomputable def average_speed_CarX (V_x : ℝ) : Prop :=
  let head_start_time := 1.2
  let distance_traveled_by_CarX := 98
  let speed_CarY := 50
  let time_elapsed := distance_traveled_by_CarX / speed_CarY
  (distance_traveled_by_CarX / time_elapsed) = V_x

theorem average_speed_of_Car_X : average_speed_CarX 50 :=
  sorry

end average_speed_of_Car_X_l31_31103


namespace price_second_day_is_81_percent_l31_31629

-- Define the original price P (for the sake of clarity in the proof statement)
variable (P : ℝ)

-- Define the reductions
def first_reduction (P : ℝ) : ℝ := P - 0.1 * P
def second_reduction (P : ℝ) : ℝ := first_reduction P - 0.1 * first_reduction P

-- Question translated to Lean statement
theorem price_second_day_is_81_percent (P : ℝ) : 
  (second_reduction P / P) * 100 = 81 := by
  sorry

end price_second_day_is_81_percent_l31_31629


namespace probability_winning_probability_not_winning_l31_31565

section Lottery

variable (p1 p2 p3 : ℝ)
variable (h1 : p1 = 0.1)
variable (h2 : p2 = 0.2)
variable (h3 : p3 = 0.4)

theorem probability_winning (h1 : p1 = 0.1) (h2 : p2 = 0.2) (h3 : p3 = 0.4) :
  p1 + p2 + p3 = 0.7 :=
by
  rw [h1, h2, h3]
  norm_num
  done

theorem probability_not_winning (h1 : p1 = 0.1) (h2 : p2 = 0.2) (h3 : p3 = 0.4) :
  1 - (p1 + p2 + p3) = 0.3 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end Lottery

end probability_winning_probability_not_winning_l31_31565


namespace smallest_possible_n_l31_31327

theorem smallest_possible_n
  (n : ℕ)
  (d : ℕ)
  (h_d_pos : d > 0)
  (h_profit : 10 * n - 30 = 100)
  (h_cost_multiple : ∃ k, d = 2 * n * k) :
  n = 13 :=
by {
  sorry
}

end smallest_possible_n_l31_31327


namespace greatest_difference_54_l31_31530

theorem greatest_difference_54 (board : ℕ → ℕ → ℕ) (h : ∀ i j, 1 ≤ board i j ∧ board i j ≤ 100) :
  ∃ i j k l, (i = k ∨ j = l) ∧ (board i j - board k l ≥ 54 ∨ board k l - board i j ≥ 54) :=
sorry

end greatest_difference_54_l31_31530


namespace area_of_shaded_region_l31_31995

open Real

noncomputable def line1 (x : ℝ) : ℝ := -3/10 * x + 5
noncomputable def line2 (x : ℝ) : ℝ := -1.5 * x + 9

theorem area_of_shaded_region : 
  ∫ x in (2:ℝ)..6, (line2 x - line1 x) = 8 :=
by
  sorry

end area_of_shaded_region_l31_31995


namespace angle_between_tangents_l31_31912

theorem angle_between_tangents (R1 R2 : ℝ) (k : ℝ) (h_ratio : R1 = 2 * k ∧ R2 = 3 * k)
  (h_touching : (∃ O1 O2 : ℝ, (R2 - R1 = k))) : 
  ∃ θ : ℝ, θ = 90 := sorry

end angle_between_tangents_l31_31912


namespace min_value_ab2_cd_l31_31938

noncomputable def arithmetic_seq (x a b y : ℝ) : Prop :=
  2 * a = x + b ∧ 2 * b = a + y

noncomputable def geometric_seq (x c d y : ℝ) : Prop :=
  c^2 = x * d ∧ d^2 = c * y

theorem min_value_ab2_cd (x y a b c d : ℝ) :
  (x > 0) → (y > 0) → arithmetic_seq x a b y → geometric_seq x c d y → 
  (a + b) ^ 2 / (c * d) ≥ 4 :=
by
  sorry

end min_value_ab2_cd_l31_31938


namespace work_days_together_l31_31723

theorem work_days_together (A_rate B_rate : ℚ) (h1 : A_rate = 1 / 12) (h2 : B_rate = 5 / 36) : 
  1 / (A_rate + B_rate) = 4.5 := by
  sorry

end work_days_together_l31_31723


namespace smallest_possible_b_l31_31482

theorem smallest_possible_b (a b c : ℚ) (h1 : a < b) (h2 : b < c)
    (arithmetic_seq : 2 * b = a + c) (geometric_seq : c^2 = a * b) :
    b = 1 / 2 :=
by
  let a := 4 * b
  let c := 2 * b - a
  -- rewrite and derived equations will be done in the proof
  sorry

end smallest_possible_b_l31_31482


namespace triangle_area_290_l31_31251

theorem triangle_area_290 
  (P Q R : ℝ × ℝ)
  (h1 : (R.1 - P.1) * (R.1 - Q.1) + (R.2 - P.2) * (R.2 - Q.2) = 0) -- Right triangle condition
  (h2 : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2) -- Length of hypotenuse PQ
  (h3 : ∀ x: ℝ, (x, x - 2) = P) -- Median through P
  (h4 : ∀ x: ℝ, (x, 3 * x + 3) = Q) -- Median through Q
  :
  ∃ (area : ℝ), area = 290 := 
sorry

end triangle_area_290_l31_31251


namespace size_of_angle_B_length_of_side_b_and_area_l31_31863

-- Given problem conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h1 : a < b) (h2 : b < c) (h3 : a / Real.sin A = 2 * b / Real.sqrt 3)

-- Prove that B = π / 3
theorem size_of_angle_B : B = Real.pi / 3 := 
sorry

-- Additional conditions for part (2)
variables (h4 : a = 2) (h5 : c = 3) (h6 : Real.cos B = 1 / 2)

-- Prove b = √7 and the area of triangle ABC
theorem length_of_side_b_and_area :
  b = Real.sqrt 7 ∧ 1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 :=
sorry

end size_of_angle_B_length_of_side_b_and_area_l31_31863


namespace students_not_finding_parents_funny_l31_31897

theorem students_not_finding_parents_funny:
  ∀ (total_students funny_dad funny_mom funny_both : ℕ),
  total_students = 50 →
  funny_dad = 25 →
  funny_mom = 30 →
  funny_both = 18 →
  (total_students - (funny_dad + funny_mom - funny_both) = 13) :=
by
  intros total_students funny_dad funny_mom funny_both
  sorry

end students_not_finding_parents_funny_l31_31897


namespace Grandfather_age_correct_l31_31631

-- Definitions based on the conditions
def Yuna_age : Nat := 9
def Father_age (Yuna_age : Nat) : Nat := Yuna_age + 27
def Grandfather_age (Father_age : Nat) : Nat := Father_age + 23

-- The theorem stating the problem to prove
theorem Grandfather_age_correct : Grandfather_age (Father_age Yuna_age) = 59 := by
  sorry

end Grandfather_age_correct_l31_31631


namespace min_value_of_sum_of_squares_l31_31696

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4.8 :=
sorry

end min_value_of_sum_of_squares_l31_31696


namespace find_unknown_number_l31_31646

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end find_unknown_number_l31_31646


namespace solve_for_x_l31_31743

theorem solve_for_x (x : ℝ) : (1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) → x = 10 :=
by
  sorry

end solve_for_x_l31_31743


namespace find_digits_l31_31528

theorem find_digits (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_digits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h_sum : 100 * a + 10 * b + c = (10 * a + b) + (10 * b + c) + (10 * c + a)) :
  a = 1 ∧ b = 9 ∧ c = 8 := by
  sorry

end find_digits_l31_31528


namespace seven_balls_expected_positions_l31_31272

theorem seven_balls_expected_positions :
  let n := 7
  let swaps := 4
  let p_stay := (1 - 2/7)^4 + 6 * (2/7)^2 * (5/7)^2 + (2/7)^4
  let expected_positions := n * p_stay
  expected_positions = 3.61 :=
by
  let n := 7
  let swaps := 4
  let p_stay := (1 - 2/7)^4 + 6 * (2/7)^2 * (5/7)^2 + (2/7)^4
  let expected_positions := n * p_stay
  exact sorry

end seven_balls_expected_positions_l31_31272


namespace triangle_obtuse_l31_31088

-- We need to set up the definitions for angles and their relationships in triangles.

variable {A B C : ℝ} -- representing the angles of the triangle in radians

structure Triangle (A B C : ℝ) : Prop where
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  sum_to_pi : A + B + C = Real.pi -- representing the sum of angles in a triangle

-- Definition to state the condition in the problem
def triangle_condition (A B C : ℝ) : Prop :=
  Triangle A B C ∧ (Real.cos A * Real.cos B - Real.sin A * Real.sin B > 0)

-- Theorem to prove the triangle is obtuse under the given condition
theorem triangle_obtuse {A B C : ℝ} (h : triangle_condition A B C) : ∃ C', C' = C ∧ C' > Real.pi / 2 :=
sorry

end triangle_obtuse_l31_31088


namespace one_divides_the_other_l31_31668

theorem one_divides_the_other (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) :
  ∃ m n : ℕ, (x = m * y) ∨ (y = n * x) :=
by 
  -- Proof goes here
  sorry

end one_divides_the_other_l31_31668


namespace value_of_x_l31_31132

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 5)

noncomputable def f_inv (y : ℝ) : ℝ := sorry -- Placeholder for the inverse of f

noncomputable def g (x : ℝ) : ℝ := 3 * f_inv x

theorem value_of_x (h : g 18 = 18) : x = 30 / 11 :=
by
  -- Proof is not required.
  sorry

end value_of_x_l31_31132


namespace plane_can_be_colored_l31_31883

-- Define a structure for a triangle and the plane divided into triangles
structure Triangle :=
(vertices : Fin 3 → ℕ) -- vertices labeled with ℕ, interpreted as 0, 1, 2

structure Plane :=
(triangles : Set Triangle)
(adjacent : Triangle → Triangle → Prop)
(labels_correct : ∀ {t1 t2 : Triangle}, adjacent t1 t2 → 
  ∀ i j: Fin 3, t1.vertices i ≠ t1.vertices j)
(adjacent_conditions: ∀ t1 t2: Triangle, adjacent t1 t2 → 
  ∃ v, (∃ i: Fin 3, t1.vertices i = v) ∧ (∃ j: Fin 3, t2.vertices j = v))

theorem plane_can_be_colored (p : Plane) : 
  ∃ (c : Triangle → ℕ), (∀ t1 t2, p.adjacent t1 t2 → c t1 ≠ c t2) :=
sorry

end plane_can_be_colored_l31_31883


namespace smaller_circle_circumference_l31_31442

-- Definitions based on the conditions given in the problem
def AB : ℝ := 24
def BC : ℝ := 45
def CD : ℝ := 28
def DA : ℝ := 53
def smaller_circle_diameter : ℝ := AB

-- Main statement to prove
theorem smaller_circle_circumference :
  let r : ℝ := smaller_circle_diameter / 2
  let circumference := 2 * Real.pi * r
  circumference = 24 * Real.pi := by
  sorry

end smaller_circle_circumference_l31_31442


namespace B_representation_l31_31571

def A : Set ℤ := {-1, 2, 3, 4}

def f (x : ℤ) : ℤ := x^2 - 2*x + 2

def B : Set ℤ := {y | ∃ x ∈ A, y = f x}

theorem B_representation : B = {2, 5, 10} :=
by {
  -- Proof to be provided
  sorry
}

end B_representation_l31_31571


namespace question_l31_31811

section

variable (x : ℝ)
variable (p q : Prop)

-- Define proposition p: ∀ x in [0,1], e^x ≥ 1
def Proposition_p : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → Real.exp x ≥ 1

-- Define proposition q: ∃ x in ℝ such that x^2 + x + 1 < 0
def Proposition_q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- The problem to prove: p ∨ q
theorem question (p q : Prop) (hp : Proposition_p) (hq : ¬ Proposition_q) : p ∨ q := by
  sorry

end

end question_l31_31811


namespace longest_side_in_ratio_5_6_7_l31_31526

theorem longest_side_in_ratio_5_6_7 (x : ℕ) (h : 5 * x + 6 * x + 7 * x = 720) : 7 * x = 280 := 
by
  sorry

end longest_side_in_ratio_5_6_7_l31_31526


namespace lcm_12_18_l31_31925

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l31_31925


namespace price_of_brand_Y_pen_l31_31176

theorem price_of_brand_Y_pen (cost_X : ℝ) (num_X : ℕ) (total_pens : ℕ) (total_cost : ℝ) :
  cost_X = 4 ∧ num_X = 6 ∧ total_pens = 12 ∧ total_cost = 42 →
  (∃ (price_Y : ℝ), price_Y = 3) :=
by
  sorry

end price_of_brand_Y_pen_l31_31176


namespace intersection_PQ_eq_23_l31_31351

def P : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def Q : Set ℝ := {x : ℝ | 2 < x}

theorem intersection_PQ_eq_23 : P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := 
by {
  sorry
}

end intersection_PQ_eq_23_l31_31351


namespace binom_15_3_eq_455_l31_31583

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem statement: Prove that binom 15 3 = 455
theorem binom_15_3_eq_455 : binom 15 3 = 455 := sorry

end binom_15_3_eq_455_l31_31583


namespace cheryl_material_left_l31_31233

def square_yards_left (bought1 bought2 used : ℚ) : ℚ :=
  bought1 + bought2 - used

theorem cheryl_material_left :
  square_yards_left (4/19) (2/13) (0.21052631578947367 : ℚ) = (0.15384615384615385 : ℚ) :=
by
  sorry

end cheryl_material_left_l31_31233


namespace smallest_lucky_number_exists_l31_31641

theorem smallest_lucky_number_exists :
  ∃ (a b c d N: ℕ), 
  N = a^2 + b^2 ∧ 
  N = c^2 + d^2 ∧ 
  a - c = 7 ∧ 
  d - b = 13 ∧ 
  N = 545 := 
by {
  sorry
}

end smallest_lucky_number_exists_l31_31641


namespace benjamin_speed_l31_31435

-- Define the problem conditions
def distance : ℕ := 800 -- Distance in kilometers
def time : ℕ := 10 -- Time in hours

-- Define the main statement
theorem benjamin_speed : distance / time = 80 := by
  sorry

end benjamin_speed_l31_31435


namespace range_of_y_l31_31410

theorem range_of_y (a b y : ℝ) (hab : a + b = 2) (hbl : b ≤ 2) (hy : y = a^2 + 2*a - 2) : y ≥ -2 :=
by
  sorry

end range_of_y_l31_31410


namespace least_number_divisible_by_6_has_remainder_4_is_40_l31_31350

-- Define the least number N which leaves a remainder of 4 when divided by 6
theorem least_number_divisible_by_6_has_remainder_4_is_40 :
  ∃ (N : ℕ), (∀ (k : ℕ), N = 6 * k + 4) ∧ N = 40 := by
  sorry

end least_number_divisible_by_6_has_remainder_4_is_40_l31_31350


namespace Jillian_collected_29_l31_31034

variable (Savannah_shells Clayton_shells total_friends friend_shells : ℕ)

def Jillian_shells : ℕ :=
  let total_shells := friend_shells * total_friends
  let others_shells := Savannah_shells + Clayton_shells
  total_shells - others_shells

theorem Jillian_collected_29 (h_savannah : Savannah_shells = 17) 
                             (h_clayton : Clayton_shells = 8) 
                             (h_friends : total_friends = 2) 
                             (h_friend_shells : friend_shells = 27) : 
  Jillian_shells Savannah_shells Clayton_shells total_friends friend_shells = 29 :=
by
  sorry

end Jillian_collected_29_l31_31034


namespace seven_pow_k_minus_k_pow_seven_l31_31018

theorem seven_pow_k_minus_k_pow_seven (k : ℕ) (h : 21^k ∣ 435961) : 7^k - k^7 = 1 :=
sorry

end seven_pow_k_minus_k_pow_seven_l31_31018


namespace jessica_total_cost_l31_31400

-- Define the costs
def cost_cat_toy : ℝ := 10.22
def cost_cage : ℝ := 11.73

-- Define the total cost
def total_cost : ℝ := cost_cat_toy + cost_cage

-- State the theorem
theorem jessica_total_cost : total_cost = 21.95 := by
  sorry

end jessica_total_cost_l31_31400


namespace average_age_l31_31849

theorem average_age (avg_age_students : ℝ) (num_students : ℕ) (avg_age_teachers : ℝ) (num_teachers : ℕ) :
  avg_age_students = 13 → 
  num_students = 40 → 
  avg_age_teachers = 42 → 
  num_teachers = 60 → 
  (num_students * avg_age_students + num_teachers * avg_age_teachers) / (num_students + num_teachers) = 30.4 :=
by
  intros h1 h2 h3 h4
  sorry

end average_age_l31_31849


namespace remi_spilled_second_time_l31_31529

-- Defining the conditions from the problem
def bottle_capacity : ℕ := 20
def daily_refills : ℕ := 3
def total_days : ℕ := 7
def total_water_consumed : ℕ := 407
def first_spill : ℕ := 5

-- Using the conditions to define the total amount of water that Remi would have drunk without spilling.
def no_spill_total : ℕ := bottle_capacity * daily_refills * total_days

-- Defining the second spill
def second_spill : ℕ := no_spill_total - first_spill - total_water_consumed

-- Stating the theorem that we need to prove
theorem remi_spilled_second_time : second_spill = 8 :=
by
  sorry

end remi_spilled_second_time_l31_31529


namespace no_solutions_eq_l31_31021

theorem no_solutions_eq (x y : ℝ) : (x + y)^2 ≠ x^2 + y^2 + 1 :=
by sorry

end no_solutions_eq_l31_31021


namespace PE_bisects_CD_given_conditions_l31_31915

variables {A B C D E P : Type*}

noncomputable def cyclic_quadrilateral (A B C D : Type*) : Prop := sorry

noncomputable def AD_squared_plus_BC_squared_eq_AB_squared (A B C D : Type*) : Prop := sorry

noncomputable def angles_equality_condition (A B C D P : Type*) : Prop := sorry

noncomputable def line_PE_bisects_CD (P E C D : Type*) : Prop := sorry

theorem PE_bisects_CD_given_conditions
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : AD_squared_plus_BC_squared_eq_AB_squared A B C D)
  (h3 : angles_equality_condition A B C D P) :
  line_PE_bisects_CD P E C D :=
sorry

end PE_bisects_CD_given_conditions_l31_31915


namespace integer_solutions_for_exponential_equation_l31_31340

theorem integer_solutions_for_exponential_equation :
  ∃ (a b c : ℕ), 
  2 ^ a * 3 ^ b + 9 = c ^ 2 ∧ 
  (a = 4 ∧ b = 0 ∧ c = 5) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 9) ∨ 
  (a = 4 ∧ b = 3 ∧ c = 21) ∨ 
  (a = 3 ∧ b = 3 ∧ c = 15) ∨ 
  (a = 4 ∧ b = 5 ∧ c = 51) :=
by {
  -- This is where the proof would go.
  sorry
}

end integer_solutions_for_exponential_equation_l31_31340


namespace trajectory_of_M_l31_31160

-- Define the two circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the condition for the moving circle M being tangent to both circles
def isTangent (Mx My : ℝ) : Prop := 
  let distC1 := (Mx + 3)^2 + My^2
  let distC2 := (Mx - 3)^2 + My^2
  distC2 - distC1 = 4

-- The equation of the trajectory of M
theorem trajectory_of_M (Mx My : ℝ) (h : isTangent Mx My) : 
  Mx^2 - (My^2 / 8) = 1 ∧ Mx < 0 :=
sorry

end trajectory_of_M_l31_31160


namespace total_money_is_305_l31_31886

-- Define the worth of each gold coin, silver coin, and the quantity of each type of coin and cash.
def worth_of_gold_coin := 50
def worth_of_silver_coin := 25
def number_of_gold_coins := 3
def number_of_silver_coins := 5
def cash_in_dollars := 30

-- Define the total money calculation based on given conditions.
def total_gold_value := number_of_gold_coins * worth_of_gold_coin
def total_silver_value := number_of_silver_coins * worth_of_silver_coin
def total_value := total_gold_value + total_silver_value + cash_in_dollars

-- The proof statement asserting the total value.
theorem total_money_is_305 : total_value = 305 := by
  -- Proof omitted for brevity.
  sorry

end total_money_is_305_l31_31886


namespace find_13th_result_l31_31599

theorem find_13th_result 
  (average_25 : ℕ → ℝ) (h1 : average_25 25 = 19)
  (average_first_12 : ℕ → ℝ) (h2 : average_first_12 12 = 14)
  (average_last_12 : ℕ → ℝ) (h3 : average_last_12 12 = 17) :
    let totalSum_25 := 25 * average_25 25
    let totalSum_first_12 := 12 * average_first_12 12
    let totalSum_last_12 := 12 * average_last_12 12
    let result_13 := totalSum_25 - totalSum_first_12 - totalSum_last_12
    result_13 = 103 :=
  by sorry

end find_13th_result_l31_31599


namespace equation_of_tangent_line_l31_31513

noncomputable def f (m x : ℝ) := m * Real.exp x - x - 1

def passes_through_P (m : ℝ) : Prop :=
  f m 0 = 1

theorem equation_of_tangent_line (m : ℝ) (h : passes_through_P m) :
  (f m) 0 = 1 → (2 - 1 = 1) ∧ ((y - 1 = x) → (x - y + 1 = 0)) :=
by
  intro h
  sorry

end equation_of_tangent_line_l31_31513


namespace smallest_n_for_divisibility_property_l31_31242

theorem smallest_n_for_divisibility_property :
  ∃ n : ℕ, 0 < n ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n^2 + n % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n^2 + n % k ≠ 0) ∧ 
  ∀ m : ℕ, 0 < m ∧ m < n → ¬ ((∀ k : ℕ, 1 ≤ k ∧ k ≤ m → m^2 + m % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ m ∧ m^2 + m % k ≠ 0)) := sorry

end smallest_n_for_divisibility_property_l31_31242


namespace first_worker_time_budget_l31_31935

theorem first_worker_time_budget
  (total_time : ℝ := 1)
  (second_worker_time : ℝ := 1 / 3)
  (third_worker_time : ℝ := 1 / 3)
  (x : ℝ) :
  x + second_worker_time + third_worker_time = total_time → x = 1 / 3 :=
by
  sorry

end first_worker_time_budget_l31_31935


namespace solution_set_of_inequality1_solution_set_of_inequality2_l31_31472

-- First inequality problem
theorem solution_set_of_inequality1 :
  {x : ℝ | x^2 + 3*x + 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
sorry

-- Second inequality problem
theorem solution_set_of_inequality2 :
  {x : ℝ | -3*x^2 + 2*x + 2 < 0} =
  {x : ℝ | x ∈ Set.Iio ((1 - Real.sqrt 7) / 3) ∪ Set.Ioi ((1 + Real.sqrt 7) / 3)} :=
sorry

end solution_set_of_inequality1_solution_set_of_inequality2_l31_31472


namespace part1_l31_31730

def setA (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - a^2 - 2*a < 0}
def setB (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3^x - 2*a ∧ x ≤ 2}

theorem part1 (a : ℝ) (h : a = 3) : setA 3 ∪ setB 3 = Set.Ioo (-6) 5 :=
by
  sorry

end part1_l31_31730


namespace n_equal_three_l31_31039

variable (m n : ℝ)

-- Conditions
def in_second_quadrant (m n : ℝ) : Prop := m < 0 ∧ n > 0
def distance_to_x_axis_eq_three (n : ℝ) : Prop := abs n = 3

-- Proof problem statement
theorem n_equal_three 
  (h1 : in_second_quadrant m n) 
  (h2 : distance_to_x_axis_eq_three n) : 
  n = 3 := 
sorry

end n_equal_three_l31_31039


namespace sum_of_cos_series_l31_31053

theorem sum_of_cos_series :
  6 * Real.cos (18 * Real.pi / 180) + 2 * Real.cos (36 * Real.pi / 180) + 
  4 * Real.cos (54 * Real.pi / 180) + 6 * Real.cos (72 * Real.pi / 180) + 
  8 * Real.cos (90 * Real.pi / 180) + 10 * Real.cos (108 * Real.pi / 180) + 
  12 * Real.cos (126 * Real.pi / 180) + 14 * Real.cos (144 * Real.pi / 180) + 
  16 * Real.cos (162 * Real.pi / 180) + 18 * Real.cos (180 * Real.pi / 180) + 
  20 * Real.cos (198 * Real.pi / 180) + 22 * Real.cos (216 * Real.pi / 180) + 
  24 * Real.cos (234 * Real.pi / 180) + 26 * Real.cos (252 * Real.pi / 180) + 
  28 * Real.cos (270 * Real.pi / 180) + 30 * Real.cos (288 * Real.pi / 180) + 
  32 * Real.cos (306 * Real.pi / 180) + 34 * Real.cos (324 * Real.pi / 180) + 
  36 * Real.cos (342 * Real.pi / 180) + 38 * Real.cos (360 * Real.pi / 180) = 10 :=
by
  sorry

end sum_of_cos_series_l31_31053


namespace gcd_polynomial_eq_one_l31_31688

theorem gcd_polynomial_eq_one (b : ℤ) (hb : Even b) (hmb : 431 ∣ b) : 
  Int.gcd (8 * b^2 + 63 * b + 143) (4 * b + 17) = 1 := by
  sorry

end gcd_polynomial_eq_one_l31_31688


namespace range_of_a_l31_31694

def A := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2*(a+1)*x + (a^2 -1) = 0}

theorem range_of_a (a : ℝ) :
  (A ∩ B a = B a) → (a = 1 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l31_31694


namespace impossibility_of_equal_sum_selection_l31_31249

theorem impossibility_of_equal_sum_selection :
  ¬ ∃ (selected non_selected : Fin 10 → ℕ),
    (∀ i, selected i = 1 ∨ selected i = 36 ∨ selected i = 2 ∨ selected i = 35 ∨ 
              selected i = 3 ∨ selected i = 34 ∨ selected i = 4 ∨ selected i = 33 ∨ 
              selected i = 5 ∨ selected i = 32 ∨ selected i = 6 ∨ selected i = 31 ∨ 
              selected i = 7 ∨ selected i = 30 ∨ selected i = 8 ∨ selected i = 29 ∨ 
              selected i = 9 ∨ selected i = 28 ∨ selected i = 10 ∨ selected i = 27) ∧ 
    (∀ i, non_selected i = 1 ∨ non_selected i = 36 ∨ non_selected i = 2 ∨ non_selected i = 35 ∨ 
              non_selected i = 3 ∨ non_selected i = 34 ∨ non_selected i = 4 ∨ non_selected i = 33 ∨ 
              non_selected i = 5 ∨ non_selected i = 32 ∨ non_selected i = 6 ∨ non_selected i = 31 ∨ 
              non_selected i = 7 ∨ non_selected i = 30 ∨ non_selected i = 8 ∨ non_selected i = 29 ∨ 
              non_selected i = 9 ∨ non_selected i = 28 ∨ non_selected i = 10 ∨ non_selected i = 27) ∧ 
    (selected ≠ non_selected) ∧ 
    (Finset.univ.sum selected = Finset.univ.sum non_selected) :=
sorry

end impossibility_of_equal_sum_selection_l31_31249


namespace possible_values_of_b_l31_31014

theorem possible_values_of_b (b : ℝ) (h : ∃ x y : ℝ, y = 2 * x + b ∧ y > 0 ∧ x = 0) : b > 0 :=
sorry

end possible_values_of_b_l31_31014


namespace find_x_l31_31360

def operation_star (a b c d : ℤ) : ℤ × ℤ :=
  (a + c, b - 2 * d)

theorem find_x (x y : ℤ) (h : operation_star (x+1) (y-1) 1 3 = (2, -4)) : x = 0 :=
by 
  sorry

end find_x_l31_31360


namespace star_point_angle_l31_31617

theorem star_point_angle (n : ℕ) (h : n > 4) (h₁ : n ≥ 3) :
  ∃ θ : ℝ, θ = (n-2) * 180 / n :=
by
  sorry

end star_point_angle_l31_31617


namespace maria_trip_distance_l31_31471

theorem maria_trip_distance (D : ℝ) 
  (h1 : D / 2 + ((D / 2) / 4) + 150 = D) 
  (h2 : 150 = 3 * D / 8) : 
  D = 400 :=
by
  -- Placeholder for the actual proof
  sorry

end maria_trip_distance_l31_31471


namespace victor_cannot_escape_k4_l31_31541

theorem victor_cannot_escape_k4
  (r : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ) 
  (k : ℝ)
  (hr : r = 1)
  (hk : k = 4)
  (hA_speed : speed_A = 4 * speed_B)
  (B_starts_at_center : ∃ (B : ℝ), B = 0):
  ¬(∃ (escape_strategy : ℝ → ℝ), escape_strategy 0 = 0 → escape_strategy r = 1) :=
sorry

end victor_cannot_escape_k4_l31_31541


namespace sarah_problem_solution_l31_31405

def two_digit_number := {x : ℕ // 10 ≤ x ∧ x < 100}
def three_digit_number := {y : ℕ // 100 ≤ y ∧ y < 1000}

theorem sarah_problem_solution (x : two_digit_number) (y : three_digit_number) 
    (h_eq : 1000 * x.1 + y.1 = 8 * x.1 * y.1) : 
    x.1 = 15 ∧ y.1 = 126 ∧ (x.1 + y.1 = 141) := 
by 
  sorry

end sarah_problem_solution_l31_31405


namespace octagon_perimeter_l31_31851

def side_length_meters : ℝ := 2.3
def number_of_sides : ℕ := 8
def meter_to_cm (meters : ℝ) : ℝ := meters * 100

def perimeter_cm (side_length_meters : ℝ) (number_of_sides : ℕ) : ℝ :=
  meter_to_cm side_length_meters * number_of_sides

theorem octagon_perimeter :
  perimeter_cm side_length_meters number_of_sides = 1840 :=
by
  sorry

end octagon_perimeter_l31_31851


namespace Jasmine_gets_off_work_at_4pm_l31_31496

-- Conditions
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_clean_time : ℕ := 10
def groomer_time : ℕ := 20
def cook_time : ℕ := 90
def dinner_time : ℕ := 19 * 60  -- 7:00 pm in minutes

-- Question to prove
theorem Jasmine_gets_off_work_at_4pm : 
  (dinner_time - cook_time - groomer_time - dry_clean_time - grocery_time - commute_time = 16 * 60) := sorry

end Jasmine_gets_off_work_at_4pm_l31_31496


namespace students_moved_outside_correct_l31_31908

noncomputable def students_total : ℕ := 90
noncomputable def students_cafeteria_initial : ℕ := (2 * students_total) / 3
noncomputable def students_outside_initial : ℕ := students_total - students_cafeteria_initial
noncomputable def students_ran_inside : ℕ := students_outside_initial / 3
noncomputable def students_cafeteria_now : ℕ := 67
noncomputable def students_moved_outside : ℕ := students_cafeteria_initial + students_ran_inside - students_cafeteria_now

theorem students_moved_outside_correct : students_moved_outside = 3 := by
  sorry

end students_moved_outside_correct_l31_31908


namespace three_pow_m_plus_2n_l31_31681

theorem three_pow_m_plus_2n (m n : ℕ) (h1 : 3^m = 5) (h2 : 9^n = 10) : 3^(m + 2 * n) = 50 :=
by
  sorry

end three_pow_m_plus_2n_l31_31681


namespace find_value_of_x_plus_5_l31_31184

-- Define a variable x
variable (x : ℕ)

-- Define the condition given in the problem
def condition := x - 10 = 15

-- The statement we need to prove
theorem find_value_of_x_plus_5 (h : x - 10 = 15) : x + 5 = 30 := 
by sorry

end find_value_of_x_plus_5_l31_31184


namespace approximation_irrational_quotient_l31_31163

theorem approximation_irrational_quotient 
  (r1 r2 : ℝ) (irrational : ¬ ∃ q : ℚ, r1 = q * r2) 
  (x : ℝ) (p : ℝ) (pos_p : p > 0) : 
  ∃ (k1 k2 : ℤ), |x - (k1 * r1 + k2 * r2)| < p :=
sorry

end approximation_irrational_quotient_l31_31163


namespace directrix_of_parabola_l31_31725

-- Define the parabola x^2 = 16y
def parabola (x y : ℝ) : Prop := x^2 = 16 * y

-- Define the directrix equation
def directrix (y : ℝ) : Prop := y = -4

-- Theorem stating that the directrix of the given parabola is y = -4
theorem directrix_of_parabola : ∀ x y: ℝ, parabola x y → ∃ y, directrix y :=
by
  sorry

end directrix_of_parabola_l31_31725


namespace range_of_sum_l31_31182

variable {x y t : ℝ}

theorem range_of_sum :
  (1 = x^2 + 4*y^2 - 2*x*y) ∧ (x < 0) ∧ (y < 0) →
  -2 <= x + 2*y ∧ x + 2*y < 0 :=
by {
  sorry
}

end range_of_sum_l31_31182


namespace curious_number_is_digit_swap_divisor_l31_31334

theorem curious_number_is_digit_swap_divisor (a b : ℕ) (hab : a ≠ 0 ∧ b ≠ 0) :
  (10 * a + b) ∣ (10 * b + a) → (10 * a + b) = 11 ∨ (10 * a + b) = 22 ∨ (10 * a + b) = 33 ∨ 
  (10 * a + b) = 44 ∨ (10 * a + b) = 55 ∨ (10 * a + b) = 66 ∨ 
  (10 * a + b) = 77 ∨ (10 * a + b) = 88 ∨ (10 * a + b) = 99 :=
by
  sorry

end curious_number_is_digit_swap_divisor_l31_31334


namespace find_sum_l31_31420

theorem find_sum (P R : ℝ) (T : ℝ) (hT : T = 3) (h1 : P * (R + 1) * 3 = P * R * 3 + 2500) : 
  P = 2500 := by
  sorry

end find_sum_l31_31420


namespace find_alpha_plus_beta_l31_31399

open Real

theorem find_alpha_plus_beta 
  (α β : ℝ)
  (h1 : sin α = sqrt 5 / 5)
  (h2 : sin β = sqrt 10 / 10)
  (h3 : π / 2 < α ∧ α < π)
  (h4 : π / 2 < β ∧ β < π) :
  α + β = 7 * π / 4 :=
sorry

end find_alpha_plus_beta_l31_31399


namespace max_value_of_q_l31_31437

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l31_31437


namespace line_equation_l31_31089

theorem line_equation (P A B : ℝ × ℝ) (h1 : P = (-1, 3)) (h2 : A = (1, 2)) (h3 : B = (3, 1)) :
  ∃ c : ℝ, (x - 2*y + c = 0) ∧ (4*x - 2*y - 5 = 0) :=
by
  sorry

end line_equation_l31_31089


namespace t_shirt_cost_l31_31355

theorem t_shirt_cost (n_tshirts : ℕ) (total_cost : ℝ) (cost_per_tshirt : ℝ)
  (h1 : n_tshirts = 25)
  (h2 : total_cost = 248) :
  cost_per_tshirt = 9.92 :=
by
  sorry

end t_shirt_cost_l31_31355


namespace actual_distance_map_l31_31376

theorem actual_distance_map (scale : ℕ) (map_distance : ℕ) (actual_distance_km : ℕ) (h1 : scale = 500000) (h2 : map_distance = 4) :
  actual_distance_km = 20 :=
by
  -- definitions and assumptions
  let actual_distance_cm := map_distance * scale
  have cm_to_km_conversion : actual_distance_km = actual_distance_cm / 100000 := sorry
  -- calculation
  have actual_distance_sol : actual_distance_cm = 4 * 500000 := sorry
  have actual_distance_eq : actual_distance_km = (4 * 500000) / 100000 := sorry
  -- final answer
  have answer_correct : actual_distance_km = 20 := sorry
  exact answer_correct

end actual_distance_map_l31_31376


namespace parabola_standard_equation_oa_dot_ob_value_line_passes_fixed_point_l31_31236

-- Definitions for the problem conditions
def parabola_symmetry_axis := "coordinate axis"
def parabola_vertex := (0, 0)
def directrix_equation := "x = -1"
def intersects_at_two_points (l : ℝ → ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) := (l P.1 = P.2) ∧ (l Q.1 = Q.2) ∧ (P ≠ Q)

-- Main theorem statements
theorem parabola_standard_equation : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") → 
  ∃ p, 0 < p ∧ ∀ y x, y^2 = 4 * p * x := 
  sorry

theorem oa_dot_ob_value (l : ℝ → ℝ) (focus : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") ∧ 
  intersects_at_two_points l P Q ∧ 
  l focus.1 = focus.2 → 
  (P.1 * Q.1 + P.2 * Q.2 = -3) := 
  sorry

theorem line_passes_fixed_point (l : ℝ → ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") ∧ 
  intersects_at_two_points l P Q ∧ 
  (P.1 * Q.1 + P.2 * Q.2 = -4) → 
  ∃ fp, fp = (2,0) := 
  sorry

end parabola_standard_equation_oa_dot_ob_value_line_passes_fixed_point_l31_31236


namespace boy_actual_height_is_236_l31_31852

def actual_height (n : ℕ) (incorrect_avg correct_avg wrong_height : ℕ) : ℕ :=
  let incorrect_total := n * incorrect_avg
  let correct_total := n * correct_avg
  let diff := incorrect_total - correct_total
  wrong_height + diff

theorem boy_actual_height_is_236 :
  ∀ (n incorrect_avg correct_avg wrong_height actual_height : ℕ),
  n = 35 → 
  incorrect_avg = 183 → 
  correct_avg = 181 → 
  wrong_height = 166 → 
  actual_height = wrong_height + (n * incorrect_avg - n * correct_avg) →
  actual_height = 236 :=
by
  intros n incorrect_avg correct_avg wrong_height actual_height hn hic hg hw ha
  rw [hn, hic, hg, hw] at ha
  -- At this point, we would normally proceed to prove the statement.
  -- However, as per the requirements, we just include "sorry" to skip the proof.
  sorry

end boy_actual_height_is_236_l31_31852


namespace segment_length_294_l31_31613

theorem segment_length_294
  (A B P Q : ℝ)   -- Define points A, B, P, Q on the real line
  (h1 : P = A + (3 / 8) * (B - A))   -- P divides AB in the ratio 3:5
  (h2 : Q = A + (4 / 11) * (B - A))  -- Q divides AB in the ratio 4:7
  (h3 : Q - P = 3)                   -- The length of PQ is 3
  : B - A = 294 := 
sorry

end segment_length_294_l31_31613


namespace will_initial_money_l31_31168

theorem will_initial_money (spent_game : ℕ) (number_of_toys : ℕ) (cost_per_toy : ℕ) (initial_money : ℕ) :
  spent_game = 27 →
  number_of_toys = 5 →
  cost_per_toy = 6 →
  initial_money = spent_game + number_of_toys * cost_per_toy →
  initial_money = 57 :=
by
  intros
  sorry

end will_initial_money_l31_31168


namespace temperature_decrease_time_l31_31092

theorem temperature_decrease_time
  (T_initial T_final T_per_hour : ℤ)
  (h_initial : T_initial = -5)
  (h_final : T_final = -25)
  (h_decrease : T_per_hour = -5) :
  (T_final - T_initial) / T_per_hour = 4 := by
sorry

end temperature_decrease_time_l31_31092


namespace train_speed_is_correct_l31_31287

-- Conditions
def train_length := 190.0152  -- in meters
def crossing_time := 17.1     -- in seconds

-- Convert units
def train_length_km := train_length / 1000  -- in kilometers
def crossing_time_hr := crossing_time / 3600  -- in hours

-- Statement of the proof problem
theorem train_speed_is_correct :
  (train_length_km / crossing_time_hr) = 40 :=
sorry

end train_speed_is_correct_l31_31287


namespace four_digit_number_conditions_l31_31776

theorem four_digit_number_conditions :
  ∃ (a b c d : ℕ), 
    (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧ 
    (a * 1000 + b * 100 + c * 10 + d = 10 * 23) ∧ 
    (a + b + c + d = 26) ∧ 
    ((b * d / 10) % 10 = a + c) ∧ 
    ∃ (n : ℕ), (b * d - c^2 = 2^n) ∧ 
    (a * 1000 + b * 100 + c * 10 + d = 1979) :=
sorry

end four_digit_number_conditions_l31_31776


namespace length_of_bridge_is_255_l31_31347

noncomputable def bridge_length (train_length : ℕ) (train_speed_kph : ℕ) (cross_time_sec : ℕ) : ℕ :=
  let train_speed_mps := train_speed_kph * 1000 / (60 * 60)
  let total_distance := train_speed_mps * cross_time_sec
  total_distance - train_length

theorem length_of_bridge_is_255 :
  ∀ (train_length : ℕ) (train_speed_kph : ℕ) (cross_time_sec : ℕ), 
    train_length = 120 →
    train_speed_kph = 45 →
    cross_time_sec = 30 →
    bridge_length train_length train_speed_kph cross_time_sec = 255 :=
by
  intros train_length train_speed_kph cross_time_sec htl htsk hcts
  simp [bridge_length]
  rw [htl, htsk, hcts]
  norm_num
  sorry

end length_of_bridge_is_255_l31_31347


namespace cos_at_min_distance_l31_31626

noncomputable def cosAtMinimumDistance (t : ℝ) (ht : t < 0) : ℝ :=
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  if distance = Real.sqrt 5 then
    x / distance
  else
    0 -- some default value given the condition distance is not sqrt(5), which is impossible in this context

theorem cos_at_min_distance (t : ℝ) (ht : t < 0) :
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  distance = Real.sqrt 5 → cosAtMinimumDistance t ht = - 2 * Real.sqrt 5 / 5 :=
by
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  sorry

end cos_at_min_distance_l31_31626


namespace elimination_method_equation_y_l31_31671

theorem elimination_method_equation_y (x y : ℝ)
    (h1 : 5 * x - 3 * y = -5)
    (h2 : 5 * x + 4 * y = -1) :
    7 * y = 4 :=
by
  -- Adding the required conditions as hypotheses and skipping the proof.
  sorry

end elimination_method_equation_y_l31_31671


namespace daps_to_dips_l31_31348

theorem daps_to_dips : 
  (∀ a b c d : ℝ, (5 * a = 4 * b) → (3 * b = 8 * c) → (c = 48 * d) → (a = 22.5 * d)) := 
by
  intros a b c d h1 h2 h3
  sorry

end daps_to_dips_l31_31348


namespace angle_x_value_l31_31271

theorem angle_x_value 
  (AB CD : Prop) -- AB and CD are straight lines
  (angle_AXB angle_AXZ angle_BXY angle_CYX : ℝ) -- Given angles in the problem
  (h1 : AB) (h2 : CD)
  (h3 : angle_AXB = 180)
  (h4 : angle_AXZ = 60)
  (h5 : angle_BXY = 50)
  (h6 : angle_CYX = 120) : 
  ∃ x : ℝ, x = 50 := by
sorry

end angle_x_value_l31_31271


namespace length_of_XY_l31_31108

theorem length_of_XY (A B C D P Q X Y : ℝ) (h₁ : A = B) (h₂ : C = D) 
  (h₃ : A + B = 13) (h₄ : C + D = 21) (h₅ : A + P = 7) 
  (h₆ : C + Q = 8) (h₇ : P ≠ Q) (h₈ : P + Q = 30) :
  ∃ k : ℝ, XY = 2 * k + 30 + 31 / 15 :=
by sorry

end length_of_XY_l31_31108


namespace jake_present_weight_l31_31976

theorem jake_present_weight (J S : ℕ) 
  (h1 : J - 32 = 2 * S) 
  (h2 : J + S = 212) : 
  J = 152 := 
by 
  sorry

end jake_present_weight_l31_31976


namespace gcd_m_n_is_one_l31_31945

/-- Definition of m -/
def m : ℕ := 130^2 + 241^2 + 352^2

/-- Definition of n -/
def n : ℕ := 129^2 + 240^2 + 353^2 + 2^3

/-- Proof statement: The greatest common divisor of m and n is 1 -/
theorem gcd_m_n_is_one : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_is_one_l31_31945


namespace problem1_problem2_problem3_problem4_l31_31874

-- Statement for problem 1
theorem problem1 : -12 + (-6) - (-28) = 10 :=
  by sorry

-- Statement for problem 2
theorem problem2 : (-8 / 5) * (15 / 4) / (-9) = 2 / 3 :=
  by sorry

-- Statement for problem 3
theorem problem3 : (-3 / 16 - 7 / 24 + 5 / 6) * (-48) = -17 :=
  by sorry

-- Statement for problem 4
theorem problem4 : -3^2 + (7 / 8 - 1) * (-2)^2 = -9.5 :=
  by sorry

end problem1_problem2_problem3_problem4_l31_31874


namespace product_of_repeating_decimals_l31_31518

theorem product_of_repeating_decimals :
  let x := (4 / 9 : ℚ)
  let y := (7 / 9 : ℚ)
  x * y = 28 / 81 :=
by
  sorry

end product_of_repeating_decimals_l31_31518


namespace isosceles_right_triangle_area_l31_31975

theorem isosceles_right_triangle_area (h : ℝ) (area : ℝ) (hypotenuse_condition : h = 6 * Real.sqrt 2) : 
  area = 18 :=
  sorry

end isosceles_right_triangle_area_l31_31975


namespace kara_total_water_intake_l31_31276

-- Define dosages and water intake per tablet
def medicationA_doses_per_day := 3
def medicationB_doses_per_day := 4
def medicationC_doses_per_day := 2
def medicationD_doses_per_day := 1

def water_per_tablet_A := 4
def water_per_tablet_B := 5
def water_per_tablet_C := 6
def water_per_tablet_D := 8

-- Compute weekly water intake
def weekly_water_intake_medication (doses_per_day water_per_tablet : ℕ) (days : ℕ) : ℕ :=
  doses_per_day * water_per_tablet * days

-- Total water intake for two weeks if instructions are followed perfectly
def total_water_no_errors :=
  2 * (weekly_water_intake_medication medicationA_doses_per_day water_per_tablet_A 7 +
       weekly_water_intake_medication medicationB_doses_per_day water_per_tablet_B 7 +
       weekly_water_intake_medication medicationC_doses_per_day water_per_tablet_C 7 +
       weekly_water_intake_medication medicationD_doses_per_day water_per_tablet_D 7)

-- Missed doses in second week
def missed_water_second_week :=
  3 * water_per_tablet_A +
  2 * water_per_tablet_B +
  2 * water_per_tablet_C +
  1 * water_per_tablet_D

-- Total water actually drunk over two weeks
def total_water_real :=
  total_water_no_errors - missed_water_second_week

-- Proof statement
theorem kara_total_water_intake :
  total_water_real = 686 :=
by
  sorry

end kara_total_water_intake_l31_31276


namespace tank_empty_time_l31_31578

noncomputable def capacity : ℝ := 5760
noncomputable def leak_rate_time : ℝ := 6
noncomputable def inlet_rate_per_minute : ℝ := 4

-- leak rate calculation
noncomputable def leak_rate : ℝ := capacity / leak_rate_time

-- inlet rate calculation in litres per hour
noncomputable def inlet_rate : ℝ := inlet_rate_per_minute * 60

-- net emptying rate calculation
noncomputable def net_empty_rate : ℝ := leak_rate - inlet_rate

-- time to empty the tank calculation
noncomputable def time_to_empty : ℝ := capacity / net_empty_rate

-- The statement to prove
theorem tank_empty_time : time_to_empty = 8 :=
by
  -- Definition step
  have h1 : leak_rate = capacity / leak_rate_time := rfl
  have h2 : inlet_rate = inlet_rate_per_minute * 60 := rfl
  have h3 : net_empty_rate = leak_rate - inlet_rate := rfl
  have h4 : time_to_empty = capacity / net_empty_rate := rfl

  -- Final proof (skipped with sorry)
  sorry

end tank_empty_time_l31_31578


namespace problem_statement_l31_31746

-- Definitions of the sets P and Q
def P : Set ℝ := {x : ℝ | x > 1}
def Q : Set ℝ := {x : ℝ | abs x > 0}

-- Statement of the problem to prove that P is not a subset of Q
theorem problem_statement : ¬ (P ⊆ Q) :=
sorry

end problem_statement_l31_31746


namespace sum_of_interior_diagonals_l31_31096

theorem sum_of_interior_diagonals (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 50) (h2 : x * y + y * z + z * x = 47) : 
  4 * Real.sqrt (x^2 + y^2 + z^2) = 20 * Real.sqrt 2 :=
by 
  sorry

end sum_of_interior_diagonals_l31_31096


namespace find_cost_of_two_enchiladas_and_five_tacos_l31_31024

noncomputable def cost_of_two_enchiladas_and_five_tacos (e t : ℝ) : ℝ :=
  2 * e + 5 * t

theorem find_cost_of_two_enchiladas_and_five_tacos (e t : ℝ):
  (e + 4 * t = 3.50) → (4 * e + t = 4.20) → cost_of_two_enchiladas_and_five_tacos e t = 5.04 :=
by
  intro h1 h2
  sorry

end find_cost_of_two_enchiladas_and_five_tacos_l31_31024


namespace range_of_a_l31_31805

noncomputable def g (x : ℝ) : ℝ := abs (x-1) - abs (x-2)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ (g x ≥ a^2 + a + 1)) ↔ (a < -1 ∨ a > 0) :=
by
  sorry

end range_of_a_l31_31805


namespace maximum_n_l31_31555

def arithmetic_sequence_max_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  ∃ d : ℤ, ∀ m : ℕ, a (m + 1) = a m + d

def is_positive_first_term (a : ℕ → ℤ) : Prop :=
  a 0 > 0

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n-1))) / 2

def roots_of_equation (a1006 a1007 : ℤ) : Prop :=
  a1006 * a1007 = -2011 ∧ a1006 + a1007 = 2012

theorem maximum_n (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : arithmetic_sequence_max_n a S 1007)
  (h2 : is_positive_first_term a)
  (h3 : sum_of_first_n_terms a S)
  (h4 : ∃ a1006 a1007, roots_of_equation a1006 a1007 ∧ a 1006 = a1006 ∧ a 1007 = a1007) :
  ∃ n, S n > 0 → n ≤ 1007 := 
sorry

end maximum_n_l31_31555


namespace max_abs_sum_on_circle_l31_31440

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by sorry

end max_abs_sum_on_circle_l31_31440


namespace minimum_value_expression_l31_31821

theorem minimum_value_expression 
  (a b : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_eq : 1 / a + 1 / b = 1) : 
  (∃ (x : ℝ), x = (1 / (a-1) + 9 / (b-1)) ∧ x = 6) :=
sorry

end minimum_value_expression_l31_31821


namespace value_of_a_pow_sum_l31_31675

variable {a : ℝ}
variable {m n : ℕ}

theorem value_of_a_pow_sum (h1 : a^m = 5) (h2 : a^n = 3) : a^(m + n) = 15 := by
  sorry

end value_of_a_pow_sum_l31_31675


namespace expected_number_of_digits_l31_31937

noncomputable def expectedNumberDigits : ℝ :=
  let oneDigitProbability := (9 : ℝ) / 16
  let twoDigitProbability := (7 : ℝ) / 16
  (oneDigitProbability * 1) + (twoDigitProbability * 2)

theorem expected_number_of_digits :
  expectedNumberDigits = 1.4375 := by
  sorry

end expected_number_of_digits_l31_31937


namespace range_of_x_l31_31369

def interval1 : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def interval2 : Set ℝ := {x | x < 1 ∨ x > 4}
def false_statement (x : ℝ) : Prop := x ∈ interval1 ∨ x ∈ interval2

theorem range_of_x (x : ℝ) (h : ¬ false_statement x) : x ∈ Set.Ico 1 2 :=
by
  sorry

end range_of_x_l31_31369


namespace amc_proposed_by_Dorlir_Ahmeti_Albania_l31_31044

-- Define the problem statement, encapsulating the conditions and the final inequality.
theorem amc_proposed_by_Dorlir_Ahmeti_Albania
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_cond : a * b + b * c + c * a = 3) :
  (a / Real.sqrt (a^3 + 5) + b / Real.sqrt (b^3 + 5) + c / Real.sqrt (c^3 + 5) ≤ Real.sqrt 6 / 2) := 
by 
  sorry -- Proof steps go here, which are omitted as per the requirement.

end amc_proposed_by_Dorlir_Ahmeti_Albania_l31_31044


namespace tank_capacity_l31_31224

theorem tank_capacity :
  (∃ c: ℝ, (∃ w: ℝ, w / c = 1/6 ∧ (w + 5) / c = 1/3) → c = 30) :=
by
  sorry

end tank_capacity_l31_31224


namespace correct_statements_l31_31244

theorem correct_statements (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (∀ b, a = 1 - 2 * b → a^2 + b^2 ≥ 1/5) ∧
  (∀ a b, a + 2 * b = 1 → ab ≤ 1/8) ∧
  (∀ a b, a + 2 * b = 1 → 3 + 2 * Real.sqrt 2 ≤ (1 / a + 1 / b)) :=
by
  sorry

end correct_statements_l31_31244


namespace find_a_8_l31_31278

-- Define the arithmetic sequence and its sum formula.
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a (n + 1) = a n + d

-- Define the sum of the first 'n' terms in the arithmetic sequence.
def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) :=
  S n = n * (a 1 + a n) / 2

-- Given conditions
def S_15_eq_90 (S : ℕ → ℕ) : Prop := S 15 = 90

-- Prove that a_8 is 6
theorem find_a_8 (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ)
  (h1 : arithmetic_sequence a d) (h2 : sum_of_first_n_terms S a 15)
  (h3 : S_15_eq_90 S) : a 8 = 6 :=
sorry

end find_a_8_l31_31278


namespace min_m_squared_plus_n_squared_l31_31489

theorem min_m_squared_plus_n_squared {m n : ℝ} (h : 4 * m - 3 * n - 5 * Real.sqrt 2 = 0) :
  m^2 + n^2 = 2 :=
sorry

end min_m_squared_plus_n_squared_l31_31489


namespace max_consecutive_semi_primes_l31_31266

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_semi_prime (n : ℕ) : Prop := 
  n > 25 ∧ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p + q

theorem max_consecutive_semi_primes : ∃ (N : ℕ), N = 5 ∧
  ∀ (a b : ℕ), (a > 25) ∧ (b = a + 4) → 
  (∀ n, a ≤ n ∧ n ≤ b → is_semi_prime n) ↔ N = 5 := sorry

end max_consecutive_semi_primes_l31_31266


namespace batsman_average_30_matches_l31_31777

theorem batsman_average_30_matches (avg_20_matches : ℕ -> ℚ) (avg_10_matches : ℕ -> ℚ)
  (h1 : avg_20_matches 20 = 40)
  (h2 : avg_10_matches 10 = 20)
  : (20 * (avg_20_matches 20) + 10 * (avg_10_matches 10)) / 30 = 33.33 := by
  sorry

end batsman_average_30_matches_l31_31777


namespace multiply_scaled_values_l31_31775

theorem multiply_scaled_values (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by 
  sorry

end multiply_scaled_values_l31_31775


namespace quadratic_inequality_solution_is_interval_l31_31452

noncomputable def quadratic_inequality_solution : Set ℝ :=
  { x : ℝ | -3*x^2 + 9*x + 12 > 0 }

theorem quadratic_inequality_solution_is_interval :
  quadratic_inequality_solution = { x : ℝ | -1 < x ∧ x < 4 } :=
sorry

end quadratic_inequality_solution_is_interval_l31_31452


namespace initial_savings_amount_l31_31201

theorem initial_savings_amount (A : ℝ) (P : ℝ) (r1 r2 t1 t2 : ℝ) (hA : A = 2247.50) (hr1 : r1 = 0.08) (hr2 : r2 = 0.04) (ht1 : t1 = 0.25) (ht2 : t2 = 0.25) :
  P = 2181 :=
by
  sorry

end initial_savings_amount_l31_31201


namespace ratio_alcohol_to_water_l31_31757

theorem ratio_alcohol_to_water (vol_alcohol vol_water : ℚ) 
  (h_alcohol : vol_alcohol = 2/7) 
  (h_water : vol_water = 3/7) : 
  vol_alcohol / vol_water = 2 / 3 := 
by
  sorry

end ratio_alcohol_to_water_l31_31757


namespace initial_distance_l31_31916

/-- Suppose Jack walks at a speed of 3 feet per second toward Christina,
    Christina walks at a speed of 3 feet per second toward Jack, and their dog Lindy
    runs at a speed of 10 feet per second back and forth between Jack and Christina.
    Given that Lindy travels a total of 400 feet when they meet, prove that the initial
    distance between Jack and Christina is 240 feet. -/
theorem initial_distance (initial_distance_jack_christina : ℝ)
  (jack_speed : ℝ := 3)
  (christina_speed : ℝ := 3)
  (lindy_speed : ℝ := 10)
  (lindy_total_distance : ℝ := 400):
  initial_distance_jack_christina = 240 :=
sorry

end initial_distance_l31_31916


namespace right_triangle_second_arm_square_l31_31052

theorem right_triangle_second_arm_square :
  ∀ (k : ℤ) (a : ℤ) (c : ℤ) (b : ℤ),
  a = 2 * k + 1 → 
  c = 2 * k + 3 → 
  a^2 + b^2 = c^2 → 
  b^2 ≠ a * c ∧ b^2 ≠ (c / a) ∧ b^2 ≠ (a + c) ∧ b^2 ≠ (c - a) :=
by sorry

end right_triangle_second_arm_square_l31_31052


namespace M_subset_N_l31_31470

def M : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 2 * a + 2}
def N : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4 * b + 5}

theorem M_subset_N : M ⊆ N := 
by 
  sorry

end M_subset_N_l31_31470


namespace product_of_possible_values_l31_31361

noncomputable def math_problem (x : ℚ) : Prop :=
  |(10 / x) - 4| = 3

theorem product_of_possible_values :
  let x1 := 10 / 7
  let x2 := 10
  (x1 * x2) = (100 / 7) :=
by
  sorry

end product_of_possible_values_l31_31361


namespace min_value_reciprocal_sum_l31_31004

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
sorry

end min_value_reciprocal_sum_l31_31004


namespace find_second_number_l31_31423

theorem find_second_number 
  (h₁ : (20 + 40 + 60) / 3 = (10 + x + 15) / 3 + 5) :
  x = 80 :=
  sorry

end find_second_number_l31_31423


namespace find_s_at_1_l31_31195

variable (t s : ℝ → ℝ)
variable (x : ℝ)

-- Define conditions
def t_def : t x = 4 * x - 9 := by sorry

def s_def : s (t x) = x^2 + 4 * x - 5 := by sorry

-- Prove the question
theorem find_s_at_1 : s 1 = 11.25 := by
  -- Proof goes here
  sorry

end find_s_at_1_l31_31195


namespace newton_method_approximation_bisection_method_approximation_l31_31978

noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 3
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 + 4*x + 3

theorem newton_method_approximation :
  let x0 := -1
  let x1 := x0 - f x0 / f' x0
  let x2 := x1 - f x1 / f' x1
  x2 = -7 / 5 := sorry

theorem bisection_method_approximation :
  let a := -2
  let b := -1
  let midpoint1 := (a + b) / 2
  let new_a := if f midpoint1 < 0 then midpoint1 else a
  let new_b := if f midpoint1 < 0 then b else midpoint1
  let midpoint2 := (new_a + new_b) / 2
  midpoint2 = -11 / 8 := sorry

end newton_method_approximation_bisection_method_approximation_l31_31978


namespace metal_waste_l31_31342

theorem metal_waste (l b : ℝ) (h : l > b) : l * b - (b^2 / 2) = 
  (l * b - (π * (b / 2)^2)) + (π * (b / 2)^2 - (b^2 / 2)) := by
  sorry

end metal_waste_l31_31342


namespace brick_width_l31_31491

variable (w : ℝ)

theorem brick_width :
  ∃ (w : ℝ), 2 * (10 * w + 10 * 3 + 3 * w) = 164 → w = 4 :=
by
  sorry

end brick_width_l31_31491


namespace initial_mixture_amount_l31_31753

theorem initial_mixture_amount (x : ℝ) (h1 : 20 / 100 * x / (x + 3) = 6 / 35) : x = 18 :=
sorry

end initial_mixture_amount_l31_31753


namespace smallest_possible_floor_sum_l31_31569

theorem smallest_possible_floor_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ∃ (a b c : ℝ), ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end smallest_possible_floor_sum_l31_31569


namespace vectors_parallel_iff_m_eq_neg_1_l31_31017

-- Given vectors a and b
def vector_a (m : ℝ) : ℝ × ℝ := (2 * m - 1, m)
def vector_b : ℝ × ℝ := (3, 1)

-- Definition of vectors being parallel
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- The statement to be proven
theorem vectors_parallel_iff_m_eq_neg_1 (m : ℝ) : 
  parallel (vector_a m) vector_b ↔ m = -1 :=
by 
  sorry

end vectors_parallel_iff_m_eq_neg_1_l31_31017


namespace grasshoppers_cannot_return_to_initial_positions_l31_31702

theorem grasshoppers_cannot_return_to_initial_positions :
  (∀ (a b c : ℕ), a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 → a + b + c ≠ 1985) :=
by
  sorry

end grasshoppers_cannot_return_to_initial_positions_l31_31702


namespace sum_remainder_of_consecutive_odds_l31_31822

theorem sum_remainder_of_consecutive_odds :
  (11075 + 11077 + 11079 + 11081 + 11083 + 11085 + 11087) % 14 = 7 :=
by
  -- Adding the proof here
  sorry

end sum_remainder_of_consecutive_odds_l31_31822


namespace first_term_exceeding_1000_l31_31409

variable (a₁ : Int := 2)
variable (d : Int := 3)

def arithmetic_sequence (n : Int) : Int :=
  a₁ + (n - 1) * d

theorem first_term_exceeding_1000 :
  ∃ n : Int, n = 334 ∧ arithmetic_sequence n > 1000 := by
  sorry

end first_term_exceeding_1000_l31_31409


namespace min_faces_n2_min_faces_n3_l31_31943

noncomputable def minimum_faces (n : ℕ) : ℕ := 
  if n = 2 then 2 
  else if n = 3 then 12 
  else sorry 

theorem min_faces_n2 : minimum_faces 2 = 2 := 
  by 
  simp [minimum_faces]

theorem min_faces_n3 : minimum_faces 3 = 12 := 
  by 
  simp [minimum_faces]

end min_faces_n2_min_faces_n3_l31_31943


namespace isosceles_triangle_height_eq_four_times_base_l31_31511

theorem isosceles_triangle_height_eq_four_times_base (b h : ℝ) 
    (same_area : (b * 2 * b) = (1/2 * b * h)) : 
    h = 4 * b :=
by 
  -- sorry allows us to skip the proof steps
  sorry

end isosceles_triangle_height_eq_four_times_base_l31_31511


namespace min_k_period_at_least_15_l31_31406

theorem min_k_period_at_least_15 (a b : ℚ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
    (h_period_a : ∃ m, a = m / (10^30 - 1))
    (h_period_b : ∃ n, b = n / (10^30 - 1))
    (h_period_ab : ∃ p, (a - b) = p / (10^30 - 1) ∧ 10^15 + 1 ∣ p) :
    ∃ k : ℕ, k = 6 ∧ (∃ q, (a + k * b) = q / (10^30 - 1) ∧ 10^15 + 1 ∣ q) :=
sorry

end min_k_period_at_least_15_l31_31406


namespace convert_to_base5_l31_31697

theorem convert_to_base5 : ∀ n : ℕ, n = 1729 → Nat.digits 5 n = [2, 3, 4, 0, 4] :=
by
  intros n hn
  rw [hn]
  -- proof steps can be filled in here
  sorry

end convert_to_base5_l31_31697


namespace cake_piece_volume_l31_31784

theorem cake_piece_volume (h : ℝ) (d : ℝ) (n : ℕ) (V_piece : ℝ) : 
  h = 1/2 ∧ d = 16 ∧ n = 8 → V_piece = 4 * Real.pi :=
by
  sorry

end cake_piece_volume_l31_31784


namespace given_problem_l31_31708

theorem given_problem :
  3^3 + 4^3 + 5^3 = 6^3 :=
by sorry

end given_problem_l31_31708


namespace find_number_l31_31180

theorem find_number (x : ℤ) (h : x = 5 * (x - 4)) : x = 5 :=
by {
  sorry
}

end find_number_l31_31180


namespace no_integer_solution_l31_31655

theorem no_integer_solution (x y z : ℤ) (n : ℕ) (h1 : Prime (x + y)) (h2 : Odd n) : ¬ (x^n + y^n = z^n) :=
sorry

end no_integer_solution_l31_31655


namespace jackson_collection_goal_l31_31533

theorem jackson_collection_goal 
  (days_in_week : ℕ)
  (goal : ℕ)
  (earned_mon : ℕ)
  (earned_tue : ℕ)
  (avg_collect_per_4house : ℕ)
  (remaining_days : ℕ)
  (remaining_goal : ℕ)
  (daily_target : ℕ)
  (collect_per_house : ℚ)
  :
  days_in_week = 5 →
  goal = 1000 →
  earned_mon = 300 →
  earned_tue = 40 →
  avg_collect_per_4house = 10 →
  remaining_goal = goal - earned_mon - earned_tue →
  remaining_days = days_in_week - 2 →
  daily_target = remaining_goal / remaining_days →
  collect_per_house = avg_collect_per_4house / 4 →
  (daily_target : ℚ) / collect_per_house = 88 := 
by sorry

end jackson_collection_goal_l31_31533


namespace area_convex_quadrilateral_l31_31234

theorem area_convex_quadrilateral (x y : ℝ) :
  (x^2 + y^2 = 73 ∧ x * y = 24) →
  -- You can place a formal statement specifying the four vertices here if needed
  ∃ a b c d : ℝ × ℝ,
  a.1^2 + a.2^2 = 73 ∧
  a.1 * a.2 = 24 ∧
  b.1^2 + b.2^2 = 73 ∧
  b.1 * b.2 = 24 ∧
  c.1^2 + c.2^2 = 73 ∧
  c.1 * c.2 = 24 ∧
  d.1^2 + d.2^2 = 73 ∧
  d.1 * d.2 = 24 ∧
  -- Ensure the quadrilateral forms a rectangle (additional conditions here)
  -- Compute the side lengths and area
  -- Specify finally the area and prove it equals 110
  True :=
sorry

end area_convex_quadrilateral_l31_31234


namespace smallest_int_remainder_two_l31_31850

theorem smallest_int_remainder_two (m : ℕ) (hm : m > 1)
  (h3 : m % 3 = 2)
  (h4 : m % 4 = 2)
  (h5 : m % 5 = 2)
  (h6 : m % 6 = 2)
  (h7 : m % 7 = 2) :
  m = 422 :=
sorry

end smallest_int_remainder_two_l31_31850


namespace milk_production_per_cow_l31_31643

theorem milk_production_per_cow :
  ∀ (total_cows : ℕ) (milk_price_per_gallon butter_price_per_stick total_earnings : ℝ)
    (customers customer_milk_demand gallons_per_butter : ℕ),
  total_cows = 12 →
  milk_price_per_gallon = 3 →
  butter_price_per_stick = 1.5 →
  total_earnings = 144 →
  customers = 6 →
  customer_milk_demand = 6 →
  gallons_per_butter = 2 →
  (∀ (total_milk_sold_to_customers produced_milk used_for_butter : ℕ),
    total_milk_sold_to_customers = customers * customer_milk_demand →
    produced_milk = total_milk_sold_to_customers + used_for_butter →
    used_for_butter = (total_earnings - (total_milk_sold_to_customers * milk_price_per_gallon)) / butter_price_per_stick / gallons_per_butter →
    produced_milk / total_cows = 4)
:= by sorry

end milk_production_per_cow_l31_31643


namespace stamps_problem_l31_31928

theorem stamps_problem (x y : ℕ) : 
  2 * x + 6 * x + 5 * y / 2 = 60 → x = 5 ∧ y = 8 ∧ 6 * x = 30 :=
by 
  sorry

end stamps_problem_l31_31928


namespace first_shaded_square_in_each_column_l31_31618

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem first_shaded_square_in_each_column : 
  ∃ n, triangular_number n = 120 ∧ ∀ m < n, ¬ ∀ k < 8, ∃ j ≤ m, ((triangular_number j) % 8) = k := 
by
  sorry

end first_shaded_square_in_each_column_l31_31618


namespace largest_n_divisible_l31_31349

theorem largest_n_divisible (n : ℕ) (h : (n : ℤ) > 0) : 
  (n^3 + 105) % (n + 12) = 0 ↔ n = 93 :=
sorry

end largest_n_divisible_l31_31349


namespace smallest_Q_value_l31_31026

noncomputable def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - x^2 + 4*x + 6

theorem smallest_Q_value :
  min (Q (-1)) (min (6) (min (1 + 2 - 1 + 4 + 6) (sorry))) = Q (-1) :=
by
  sorry

end smallest_Q_value_l31_31026


namespace cartons_loaded_l31_31142

def total_cartons : Nat := 50
def cans_per_carton : Nat := 20
def cans_left_to_load : Nat := 200

theorem cartons_loaded (C : Nat) (h : cans_per_carton ≠ 0) : 
  C = total_cartons - (cans_left_to_load / cans_per_carton) := by
  sorry

end cartons_loaded_l31_31142


namespace current_swans_number_l31_31745

noncomputable def swans_doubling (S : ℕ) : Prop :=
  let S_after_10_years := S * 2^5 -- Doubling every 2 years for 10 years results in multiplying by 2^5
  S_after_10_years = 480

theorem current_swans_number (S : ℕ) (h : swans_doubling S) : S = 15 := by
  sorry

end current_swans_number_l31_31745


namespace negation_of_existential_l31_31870

theorem negation_of_existential (P : Prop) :
  (¬ (∃ x : ℝ, x ^ 3 > 0)) ↔ (∀ x : ℝ, x ^ 3 ≤ 0) :=
by
  sorry

end negation_of_existential_l31_31870


namespace sabrina_herbs_l31_31999

theorem sabrina_herbs (S V : ℕ) 
  (h1 : 2 * S = 12)
  (h2 : 12 + S + V = 29) :
  V - S = 5 := by
  sorry

end sabrina_herbs_l31_31999


namespace probability_not_e_after_n_spins_l31_31809

theorem probability_not_e_after_n_spins
    (S : Type)
    (e b c d : S)
    (p_e : ℝ)
    (p_b : ℝ)
    (p_c : ℝ)
    (p_d : ℝ) :
    (p_e = 0.25) →
    (p_b = 0.25) →
    (p_c = 0.25) →
    (p_d = 0.25) →
    (1 - p_e)^2 = 0.5625 :=
by
  sorry

end probability_not_e_after_n_spins_l31_31809


namespace find_deaf_students_l31_31984

-- Definitions based on conditions
variables (B D : ℕ)
axiom deaf_students_triple_blind_students : D = 3 * B
axiom total_students : D + B = 240

-- Proof statement
theorem find_deaf_students (h1 : D = 3 * B) (h2 : D + B = 240) : D = 180 :=
sorry

end find_deaf_students_l31_31984


namespace set_difference_lt3_gt0_1_leq_x_leq_2_l31_31341

def A := {x : ℝ | |x| < 3}
def B := {x : ℝ | x^2 - 3 * x + 2 > 0}

theorem set_difference_lt3_gt0_1_leq_x_leq_2 : {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end set_difference_lt3_gt0_1_leq_x_leq_2_l31_31341


namespace ratio_bananas_apples_is_3_to_1_l31_31155

def ratio_of_bananas_to_apples (oranges apples bananas peaches total_fruit : ℕ) : ℚ :=
if oranges = 6 ∧ apples = oranges - 2 ∧ peaches = bananas / 2 ∧ total_fruit = 28
   ∧ 6 + apples + bananas + peaches = total_fruit then
    bananas / apples
else 0

theorem ratio_bananas_apples_is_3_to_1 : ratio_of_bananas_to_apples 6 4 12 6 28 = 3 := by
sorry

end ratio_bananas_apples_is_3_to_1_l31_31155


namespace geometric_sequence_divisible_l31_31454

theorem geometric_sequence_divisible (a1 a2 : ℝ) (h1 : a1 = 5 / 8) (h2 : a2 = 25) :
  ∃ n : ℕ, n = 7 ∧ (40^(n-1) * (5/8)) % 10^7 = 0 :=
by
  sorry

end geometric_sequence_divisible_l31_31454


namespace difference_is_167_l31_31919

-- Define the number of boys and girls in each village
def A_village_boys : ℕ := 204
def A_village_girls : ℕ := 468
def B_village_boys : ℕ := 334
def B_village_girls : ℕ := 516
def C_village_boys : ℕ := 427
def C_village_girls : ℕ := 458
def D_village_boys : ℕ := 549
def D_village_girls : ℕ := 239

-- Define total number of boys and girls
def total_boys := A_village_boys + B_village_boys + C_village_boys + D_village_boys
def total_girls := A_village_girls + B_village_girls + C_village_girls + D_village_girls

-- Define the difference between total girls and total boys
def difference := total_girls - total_boys

-- The theorem to prove the difference is 167
theorem difference_is_167 : difference = 167 := by
  sorry

end difference_is_167_l31_31919


namespace factorize_1_factorize_2_l31_31634

theorem factorize_1 {x : ℝ} : 2*x^2 - 4*x = 2*x*(x - 2) := 
by sorry

theorem factorize_2 {a b x y : ℝ} : a^2*(x - y) + b^2*(y - x) = (x - y) * (a + b) * (a - b) := 
by sorry

end factorize_1_factorize_2_l31_31634


namespace sum_of_coordinates_A_l31_31148

-- Define the points A, B, and C and the given conditions
variables (A B C : ℝ × ℝ)
variables (h_ratio1 : dist A C / dist A B = 1 / 3)
variables (h_ratio2 : dist B C / dist A B = 1 / 3)
variables (h_B : B = (2, 8))
variables (h_C : C = (0, 2))

-- Lean 4 statement to prove the sum of the coordinates of A is -14
theorem sum_of_coordinates_A : (A.1 + A.2) = -14 :=
sorry

end sum_of_coordinates_A_l31_31148


namespace parking_spaces_remaining_l31_31980

-- Define the conditions as variables
variable (total_spaces : Nat := 30)
variable (spaces_per_caravan : Nat := 2)
variable (num_caravans : Nat := 3)

-- Prove the number of vehicles that can still park equals 24
theorem parking_spaces_remaining (total_spaces spaces_per_caravan num_caravans : Nat) :
    total_spaces - spaces_per_caravan * num_caravans = 24 :=
by
  -- Filling in the proof is required to fully complete this, but as per instruction we add 'sorry'
  sorry

end parking_spaces_remaining_l31_31980


namespace kanul_cash_percentage_l31_31146

-- Define the conditions
def raw_materials_cost : ℝ := 3000
def machinery_cost : ℝ := 1000
def total_amount : ℝ := 5714.29
def total_spent := raw_materials_cost + machinery_cost
def cash := total_amount - total_spent

-- The goal is to prove the percentage of the total amount as cash is 30%
theorem kanul_cash_percentage :
  (cash / total_amount) * 100 = 30 := 
sorry

end kanul_cash_percentage_l31_31146


namespace profit_ratio_a_to_b_l31_31717

noncomputable def capital_a : ℕ := 3500
noncomputable def time_a : ℕ := 12
noncomputable def capital_b : ℕ := 10500
noncomputable def time_b : ℕ := 6

noncomputable def capital_months (capital : ℕ) (time : ℕ) : ℕ :=
  capital * time

noncomputable def capital_months_a : ℕ :=
  capital_months capital_a time_a

noncomputable def capital_months_b : ℕ :=
  capital_months capital_b time_b

theorem profit_ratio_a_to_b : (capital_months_a / Nat.gcd capital_months_a capital_months_b) =
                             2 ∧
                             (capital_months_b / Nat.gcd capital_months_a capital_months_b) =
                             3 := 
by
  sorry

end profit_ratio_a_to_b_l31_31717


namespace inequality_always_true_l31_31562

theorem inequality_always_true (a : ℝ) (x : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  x^2 + (a - 4) * x + 4 - 2 * a > 0 → (x < 1 ∨ x > 3) :=
by {
  sorry
}

end inequality_always_true_l31_31562


namespace correct_operation_l31_31476

variables (a : ℝ)

-- defining the expressions to be compared
def lhs := 2 * a^2 * a^4
def rhs := 2 * a^6

theorem correct_operation : lhs a = rhs a := 
by sorry

end correct_operation_l31_31476


namespace total_sugar_l31_31893

theorem total_sugar (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
  sugar_frosting + sugar_cake = 0.8 :=
by {
  -- The proof goes here
  sorry
}

end total_sugar_l31_31893


namespace boundary_points_distance_probability_l31_31077

theorem boundary_points_distance_probability
  (a b c : ℕ)
  (h1 : ∀ (x y : ℝ), x ∈ [0, 4] → y ∈ [0, 4] → (|x - y| ≥ 1 / 2 → True))
  (h2 : ∀ (x y : ℝ), x ∈ [0, 4] → y ∈ [0, 4] → True)
  (h3 : ∃ a b c : ℕ, a - b * Real.pi = 2 ∧ c = 4 ∧ Int.gcd (Int.ofNat a) (Int.gcd (Int.ofNat b) (Int.ofNat c)) = 1) :
  (a + b + c = 62) := sorry

end boundary_points_distance_probability_l31_31077


namespace arithmetic_seq_n_possible_values_l31_31011

theorem arithmetic_seq_n_possible_values
  (a1 : ℕ) (a_n : ℕ → ℕ) (d : ℕ) (n : ℕ):
  a1 = 1 → 
  (∀ n, n ≥ 3 → a_n n = 100) → 
  (∃ d : ℕ, ∀ n, n ≥ 3 → a_n n = a1 + (n - 1) * d) → 
  (n = 4 ∨ n = 10 ∨ n = 12 ∨ n = 34 ∨ n = 100) := by
  sorry

end arithmetic_seq_n_possible_values_l31_31011


namespace binomial_expansion_problem_l31_31570

theorem binomial_expansion_problem :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ),
    (1 + 2 * x) ^ 11 =
      a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 +
      a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 +
      a_9 * x^9 + a_10 * x^10 + a_11 * x^11 →
    a_1 - 2 * a_2 + 3 * a_3 - 4 * a_4 + 5 * a_5 - 6 * a_6 +
    7 * a_7 - 8 * a_8 + 9 * a_9 - 10 * a_10 + 11 * a_11 = 22 :=
by
  -- The proof is omitted for this exercise
  sorry

end binomial_expansion_problem_l31_31570


namespace moles_of_NH3_formed_l31_31911

-- Conditions
def moles_NH4Cl : ℕ := 3 -- 3 moles of Ammonium chloride
def total_moles_NH3_formed : ℕ := 3 -- The total moles of Ammonia formed

-- The balanced chemical reaction implies a 1:1 molar ratio
lemma reaction_ratio (n : ℕ) : total_moles_NH3_formed = n := by
  sorry

-- Prove that the number of moles of NH3 formed is equal to 3
theorem moles_of_NH3_formed : total_moles_NH3_formed = moles_NH4Cl := 
reaction_ratio moles_NH4Cl

end moles_of_NH3_formed_l31_31911


namespace find_g_inv_f_neg7_l31_31358

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_def : ∀ x, f_inv (g x) = 5 * x + 3

theorem find_g_inv_f_neg7 : g_inv (f (-7)) = -2 :=
by
  sorry

end find_g_inv_f_neg7_l31_31358


namespace compute_problem_l31_31426

theorem compute_problem : (19^12 / 19^8)^2 = 130321 := by
  sorry

end compute_problem_l31_31426


namespace luncheon_cost_l31_31940

theorem luncheon_cost (s c p : ℝ)
  (h1 : 2 * s + 5 * c + p = 3.00)
  (h2 : 5 * s + 8 * c + p = 5.40)
  (h3 : 3 * s + 4 * c + p = 3.60) :
  2 * s + 2 * c + p = 2.60 :=
sorry

end luncheon_cost_l31_31940


namespace rectangular_plot_perimeter_l31_31387

theorem rectangular_plot_perimeter (w : ℝ) (P : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  (cost_per_meter = 6.5) →
  (total_cost = 1430) →
  (P = 2 * (w + (w + 10))) →
  (cost_per_meter * P = total_cost) →
  P = 220 :=
by
  sorry

end rectangular_plot_perimeter_l31_31387


namespace original_number_of_men_l31_31523

theorem original_number_of_men (x : ℕ) 
  (h1 : 17 * x = 21 * (x - 8)) : x = 42 := 
by {
   -- proof steps can be filled in here
   sorry
}

end original_number_of_men_l31_31523


namespace age_difference_constant_l31_31362

theorem age_difference_constant (a b x : ℕ) : (a + x) - (b + x) = a - b :=
by
  sorry

end age_difference_constant_l31_31362


namespace least_number_to_add_l31_31450

theorem least_number_to_add (n : ℕ) (d : ℕ) (h1 : n = 907223) (h2 : d = 577) : (d - (n % d) = 518) := 
by
  rw [h1, h2]
  sorry

end least_number_to_add_l31_31450


namespace cost_of_agricultural_equipment_max_units_of_type_A_l31_31624

-- Define cost equations
variables (x y : ℝ)

-- Define conditions as hypotheses
def condition1 : Prop := 2 * x + y = 4.2
def condition2 : Prop := x + 3 * y = 5.1

-- Prove the costs are respectively 1.5 and 1.2
theorem cost_of_agricultural_equipment (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 1.5 ∧ y = 1.2 := sorry

-- Define the maximum units constraint
def total_cost (m : ℕ) : ℝ := 1.5 * m + 1.2 * (2 * m - 3)

-- Prove the maximum units of type A is 3
theorem max_units_of_type_A (m : ℕ) (h : total_cost m ≤ 10) : m ≤ 3 := sorry

end cost_of_agricultural_equipment_max_units_of_type_A_l31_31624


namespace problem_statement_l31_31989

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f'' (x : ℝ) : ℝ := -Real.sin x - Real.cos x

theorem problem_statement (a : ℝ) (h : f'' a = 3 * f a) : 
  (Real.sin a)^2 - 3 / (Real.cos a)^2 + 1 = -14 / 9 := 
sorry

end problem_statement_l31_31989


namespace rainfall_sunday_l31_31517

theorem rainfall_sunday 
  (rain_sun rain_mon rain_tue : ℝ)
  (h1 : rain_mon = rain_sun + 3)
  (h2 : rain_tue = 2 * rain_mon)
  (h3 : rain_sun + rain_mon + rain_tue = 25) :
  rain_sun = 4 :=
by
  sorry

end rainfall_sunday_l31_31517


namespace triangle_is_right_triangle_l31_31464

variable (A B C : ℝ) (a b c : ℝ)

-- Conditions definitions
def condition1 : Prop := A + B = C
def condition2 : Prop := a / b = 3 / 4 ∧ b / c = 4 / 5 ∧ a / c = 3 / 5
def condition3 : Prop := A = 90 - B

-- Proof problem
theorem triangle_is_right_triangle (h1 : condition1 A B C) (h2 : condition2 a b c) (h3 : condition3 A B) : C = 90 := 
sorry

end triangle_is_right_triangle_l31_31464


namespace a_squared_plus_b_squared_gt_one_over_four_sequence_is_arithmetic_l31_31398

-- For Question 1
theorem a_squared_plus_b_squared_gt_one_over_four (a b : ℝ) (h : a + b = 1) : a^2 + b^2 > 1/4 :=
sorry

-- For Question 2
theorem sequence_is_arithmetic (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, S n = 2 * (n:ℝ)^2 - 3 * (n:ℝ) - 2) :
  ∃ d, ∀ n, (S n / (2 * (n:ℝ) + 1)) = (S (n + 1) / (2 * (n + 1:ℝ) + 1)) + d :=
sorry

end a_squared_plus_b_squared_gt_one_over_four_sequence_is_arithmetic_l31_31398


namespace largest_negative_is_l31_31144

def largest_of_negatives (a b c d : ℚ) (largest : ℚ) : Prop := largest = max (max a b) (max c d)

theorem largest_negative_is (largest : ℚ) : largest_of_negatives (-2/3) (-2) (-1) (-5) largest → largest = -2/3 :=
by
  intro h
  -- We assume the definition and the theorem are sufficient to say largest = -2/3
  sorry

end largest_negative_is_l31_31144


namespace line_passing_quadrants_l31_31691

theorem line_passing_quadrants (a k : ℝ) (a_nonzero : a ≠ 0)
  (x1 x2 y1 y2 : ℝ) (hx1 : y1 = a * x1^2 - a) (hx2 : y2 = a * x2^2 - a)
  (hx1_y1 : y1 = k * x1) (hx2_y2 : y2 = k * x2) 
  (sum_x : x1 + x2 < 0) : 
  ∃ (q1 q4 : (ℝ × ℝ)), 
  (q1.1 > 0 ∧ q1.2 > 0 ∧ q1.2 = a * q1.1 + k) ∧ (q4.1 > 0 ∧ q4.2 < 0 ∧ q4.2 = a * q4.1 + k) := 
sorry

end line_passing_quadrants_l31_31691


namespace total_time_spent_l31_31445

def one_round_time : ℕ := 30
def saturday_initial_rounds : ℕ := 1
def saturday_additional_rounds : ℕ := 10
def sunday_rounds : ℕ := 15

theorem total_time_spent :
  one_round_time * (saturday_initial_rounds + saturday_additional_rounds + sunday_rounds) = 780 := by
  sorry

end total_time_spent_l31_31445


namespace area_of_triangle_l31_31537

namespace TriangleArea

structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

noncomputable def area (A B C : Point3D) : ℚ :=
  let x1 := A.x
  let y1 := A.y
  let z1 := A.z
  let x2 := B.x
  let y2 := B.y
  let z2 := B.z
  let x3 := C.x
  let y3 := C.y
  let z3 := C.z
  1 / 2 * ( (x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)) )

def A : Point3D := ⟨0, 3, 6⟩
def B : Point3D := ⟨-2, 2, 2⟩
def C : Point3D := ⟨-5, 5, 2⟩

theorem area_of_triangle : area A B C = 4.5 :=
by
  sorry

end TriangleArea

end area_of_triangle_l31_31537
