import Mathlib

namespace NUMINAMATH_GPT_triangle_DEF_area_10_l1674_167414

-- Definitions of vertices and line
def D : ℝ × ℝ := (4, 0)
def E : ℝ × ℝ := (0, 4)
def line (x y : ℝ) : Prop := x + y = 9

-- Definition of point F lying on the given line
axiom F_on_line (F : ℝ × ℝ) : line (F.1) (F.2)

-- The proof statement of the area of triangle DEF being 10
theorem triangle_DEF_area_10 : ∃ F : ℝ × ℝ, line F.1 F.2 ∧ 
  (1 / 2) * abs (D.1 - F.1) * abs E.2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_triangle_DEF_area_10_l1674_167414


namespace NUMINAMATH_GPT_ryan_learning_hours_l1674_167485

theorem ryan_learning_hours (H_E : ℕ) (H_C : ℕ) (h1 : H_E = 6) (h2 : H_C = 2) : H_E - H_C = 4 := by
  sorry

end NUMINAMATH_GPT_ryan_learning_hours_l1674_167485


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1674_167457

def quadratic_real_roots (a : ℝ) : Prop :=
  (∃ x : ℝ, x^2 - 2 * x + a = 0)

theorem sufficient_but_not_necessary (a : ℝ) :
  (quadratic_real_roots 1) ∧ (∀ a > 1, ¬ quadratic_real_roots a) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1674_167457


namespace NUMINAMATH_GPT_number_of_yellow_balls_l1674_167491

theorem number_of_yellow_balls (x : ℕ) (h : (6 : ℝ) / (6 + x) = 0.3) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_number_of_yellow_balls_l1674_167491


namespace NUMINAMATH_GPT_solve_fraction_zero_l1674_167454

theorem solve_fraction_zero (x : ℕ) (h : x ≠ 0) (h_eq : (x - 1) / x = 0) : x = 1 := by 
  sorry

end NUMINAMATH_GPT_solve_fraction_zero_l1674_167454


namespace NUMINAMATH_GPT_range_of_m_l1674_167407

variable (m : ℝ) -- variable m in the real numbers

-- Definition of proposition p
def p : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

-- Definition of proposition q
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- The theorem statement with the given conditions
theorem range_of_m (h : p m ∧ q m) : -2 < m ∧ m < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1674_167407


namespace NUMINAMATH_GPT_find_hypotenuse_l1674_167476

-- Let a, b be the legs of the right triangle, c be the hypotenuse.
-- Let h be the altitude to the hypotenuse and r be the radius of the inscribed circle.
variable (a b c h r : ℝ)

-- Assume conditions of a right-angled triangle
def right_angled (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Given the altitude to the hypotenuse
def altitude (h c : ℝ) : Prop :=
  ∃ a b : ℝ, right_angled a b c ∧ h = a * b / c

-- Given the radius of the inscribed circle
def inscribed_radius (r a b c : ℝ) : Prop :=
  r = (a + b - c) / 2

-- The proof problem statement
theorem find_hypotenuse (a b c h r : ℝ) 
  (h_right_angled : right_angled a b c)
  (h_altitude : altitude h c)
  (h_inscribed_radius : inscribed_radius r a b c) : 
  c = 2 * r^2 / (h - 2 * r) :=
  sorry

end NUMINAMATH_GPT_find_hypotenuse_l1674_167476


namespace NUMINAMATH_GPT_model_to_statue_scale_l1674_167426

theorem model_to_statue_scale
  (statue_height_ft : ℕ)
  (model_height_in : ℕ)
  (ft_to_in : ℕ)
  (statue_height_in : ℕ)
  (scale : ℕ)
  (h1 : statue_height_ft = 120)
  (h2 : model_height_in = 6)
  (h3 : ft_to_in = 12)
  (h4 : statue_height_in = statue_height_ft * ft_to_in)
  (h5 : scale = (statue_height_in / model_height_in) / ft_to_in) : scale = 20 := 
  sorry

end NUMINAMATH_GPT_model_to_statue_scale_l1674_167426


namespace NUMINAMATH_GPT_ab_greater_than_a_plus_b_l1674_167403

theorem ab_greater_than_a_plus_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a - b = a / b) : ab > a + b :=
sorry

end NUMINAMATH_GPT_ab_greater_than_a_plus_b_l1674_167403


namespace NUMINAMATH_GPT_tens_digit_of_23_pow_1987_l1674_167462

theorem tens_digit_of_23_pow_1987 : (23 ^ 1987 % 100 / 10) % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_23_pow_1987_l1674_167462


namespace NUMINAMATH_GPT_smallest_digit_not_in_odd_units_l1674_167429

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end NUMINAMATH_GPT_smallest_digit_not_in_odd_units_l1674_167429


namespace NUMINAMATH_GPT_angle_BDE_60_l1674_167404

noncomputable def is_isosceles_triangle (A B C : Type) (angle_BAC : ℝ) : Prop :=
angle_BAC = 20

noncomputable def equal_sides (BC BD BE : ℝ) : Prop :=
BC = BD ∧ BD = BE

theorem angle_BDE_60 (A B C D E : Type) (BC BD BE : ℝ) 
  (h1 : is_isosceles_triangle A B C 20) 
  (h2 : equal_sides BC BD BE) : 
  ∃ (angle_BDE : ℝ), angle_BDE = 60 :=
by
  sorry

end NUMINAMATH_GPT_angle_BDE_60_l1674_167404


namespace NUMINAMATH_GPT_find_base_l1674_167423

-- Definitions based on the conditions of the problem
def is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
def is_perfect_cube (n : ℕ) := ∃ m : ℕ, m * m * m = n
def is_perfect_fourth (n : ℕ) := ∃ m : ℕ, m * m * m * m = n

-- Define the number A in terms of base a
def A (a : ℕ) : ℕ := 4 * a * a + 4 * a + 1

-- Problem statement: find a base a > 4 such that A is both a perfect cube and a perfect fourth power
theorem find_base (a : ℕ)
  (ha : a > 4)
  (h_square : is_perfect_square (A a)) :
  is_perfect_cube (A a) ∧ is_perfect_fourth (A a) :=
sorry

end NUMINAMATH_GPT_find_base_l1674_167423


namespace NUMINAMATH_GPT_spider_legs_total_l1674_167409

-- Definitions based on given conditions
def spiders : ℕ := 4
def legs_per_spider : ℕ := 8

-- Theorem statement
theorem spider_legs_total : (spiders * legs_per_spider) = 32 := by
  sorry

end NUMINAMATH_GPT_spider_legs_total_l1674_167409


namespace NUMINAMATH_GPT_remainder_of_3_pow_108_plus_5_l1674_167444

theorem remainder_of_3_pow_108_plus_5 :
  (3^108 + 5) % 10 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_108_plus_5_l1674_167444


namespace NUMINAMATH_GPT_cube_side_length_equals_six_l1674_167475

theorem cube_side_length_equals_six {s : ℝ} (h : 6 * s ^ 2 = s ^ 3) : s = 6 :=
by
  sorry

end NUMINAMATH_GPT_cube_side_length_equals_six_l1674_167475


namespace NUMINAMATH_GPT_unique_intersection_of_A_and_B_l1674_167417

-- Define the sets A and B with their respective conditions
def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ x^2 + y^2 = 4 }

def B (r : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - 3)^2 + (y - 4)^2 = r^2 ∧ r > 0 }

-- Define the main theorem statement
theorem unique_intersection_of_A_and_B (r : ℝ) (h : r > 0) : 
  (∃! p, p ∈ A ∧ p ∈ B r) ↔ r = 3 ∨ r = 7 :=
sorry

end NUMINAMATH_GPT_unique_intersection_of_A_and_B_l1674_167417


namespace NUMINAMATH_GPT_solution_of_equations_l1674_167466

variables (x y z w : ℤ)

def system_of_equations :=
  x + y + z + w = 20 ∧
  y + 2 * z - 3 * w = 28 ∧
  x - 2 * y + z = 36 ∧
  -7 * x - y + 5 * z + 3 * w = 84

theorem solution_of_equations (x y z w : ℤ) :
  system_of_equations x y z w → (x, y, z, w) = (4, -6, 20, 2) :=
by sorry

end NUMINAMATH_GPT_solution_of_equations_l1674_167466


namespace NUMINAMATH_GPT_differentiable_function_zero_l1674_167448

noncomputable def f : ℝ → ℝ := sorry

theorem differentiable_function_zero (f : ℝ → ℝ) (h_diff : ∀ x ≥ 0, DifferentiableAt ℝ f x)
  (h_f0 : f 0 = 0) (h_fun : ∀ x ≥ 0, ∀ y ≥ 0, (x = y^2) → deriv f x = f y) : 
  ∀ x ≥ 0, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_differentiable_function_zero_l1674_167448


namespace NUMINAMATH_GPT_find_order_amount_l1674_167442

noncomputable def unit_price : ℝ := 100

def discount_rate (x : ℕ) : ℝ :=
  if x < 250 then 0
  else if x < 500 then 0.05
  else if x < 1000 then 0.10
  else 0.15

theorem find_order_amount (T : ℝ) (x : ℕ)
    (hx : x = 980) (hT : T = 88200) :
  T = unit_price * x * (1 - discount_rate x) :=
by
  rw [hx, hT]
  sorry

end NUMINAMATH_GPT_find_order_amount_l1674_167442


namespace NUMINAMATH_GPT_value_of_expression_l1674_167447

theorem value_of_expression (a b c : ℝ) (h1 : 4 * a = 5 * b) (h2 : 5 * b = 30) (h3 : a + b + c = 15) : 40 * a * b / c = 1200 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1674_167447


namespace NUMINAMATH_GPT_least_subtracted_divisible_l1674_167465

theorem least_subtracted_divisible :
  ∃ k, (5264 - 11) = 17 * k :=
by
  sorry

end NUMINAMATH_GPT_least_subtracted_divisible_l1674_167465


namespace NUMINAMATH_GPT_total_students_l1674_167400

theorem total_students (a : ℕ) (h1: (71 * ((3480 - 69 * a) / 2) + 69 * (a - (3480 - 69 * a) / 2)) = 3480) : a = 50 :=
by
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_total_students_l1674_167400


namespace NUMINAMATH_GPT_min_sum_y1_y2_l1674_167434

theorem min_sum_y1_y2 (y : ℕ → ℕ) (h_seq : ∀ n ≥ 1, y (n+2) = (y n + 2013)/(1 + y (n+1))) : 
  ∃ y1 y2, y1 + y2 = 94 ∧ (∀ n, y n > 0) ∧ (y 1 = y1) ∧ (y 2 = y2) := 
sorry

end NUMINAMATH_GPT_min_sum_y1_y2_l1674_167434


namespace NUMINAMATH_GPT_cost_per_bar_l1674_167494

variable (months_in_year : ℕ := 12)
variable (months_per_bar_of_soap : ℕ := 2)
variable (total_cost_for_year : ℕ := 48)

theorem cost_per_bar (h1 : months_per_bar_of_soap > 0)
                     (h2 : total_cost_for_year > 0) : 
    (total_cost_for_year / (months_in_year / months_per_bar_of_soap)) = 8 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_bar_l1674_167494


namespace NUMINAMATH_GPT_distance_between_clocks_centers_l1674_167406

variable (M m : ℝ)

theorem distance_between_clocks_centers :
  ∃ (c : ℝ), (|c| = (1/2) * (M + m)) := by
  sorry

end NUMINAMATH_GPT_distance_between_clocks_centers_l1674_167406


namespace NUMINAMATH_GPT_circumference_base_of_cone_l1674_167441

-- Define the given conditions
def radius_circle : ℝ := 6
def angle_sector : ℝ := 300

-- Define the problem to prove the circumference of the base of the resulting cone in terms of π
theorem circumference_base_of_cone :
  (angle_sector / 360) * (2 * π * radius_circle) = 10 * π := by
sorry

end NUMINAMATH_GPT_circumference_base_of_cone_l1674_167441


namespace NUMINAMATH_GPT_max_tied_teams_l1674_167498

theorem max_tied_teams (n : ℕ) (h_n : n = 8) (tournament : Fin n → Fin n → Prop)
  (h_symmetric : ∀ i j, tournament i j ↔ tournament j i)
  (h_antisymmetric : ∀ i j, tournament i j → ¬ tournament j i)
  (h_total : ∀ i j, i ≠ j → tournament i j ∨ tournament j i) :
  ∃ (k : ℕ), k = 7 ∧ ∀ (wins : Fin n → ℕ), 
  (∀ i, wins i = 4 → ∃! j, i ≠ j ∧ tournament i j) → True :=
by sorry

end NUMINAMATH_GPT_max_tied_teams_l1674_167498


namespace NUMINAMATH_GPT_carlos_and_dana_rest_days_l1674_167443

structure Schedule where
  days_of_cycle : ℕ
  work_days : ℕ
  rest_days : ℕ

def carlos : Schedule := ⟨7, 5, 2⟩
def dana : Schedule := ⟨13, 9, 4⟩

def days_both_rest (days_count : ℕ) (sched1 sched2 : Schedule) : ℕ :=
  let lcm_cycle := Nat.lcm sched1.days_of_cycle sched2.days_of_cycle
  let coincidences_in_cycle := 2  -- As derived from the solution
  let full_cycles := days_count / lcm_cycle
  coincidences_in_cycle * full_cycles

theorem carlos_and_dana_rest_days :
  days_both_rest 1500 carlos dana = 32 := by
  sorry

end NUMINAMATH_GPT_carlos_and_dana_rest_days_l1674_167443


namespace NUMINAMATH_GPT_jack_weight_l1674_167459

-- Define weights and conditions
def weight_of_rocks : ℕ := 5 * 4
def weight_of_anna : ℕ := 40
def weight_of_jack : ℕ := weight_of_anna - weight_of_rocks

-- Prove that Jack's weight is 20 pounds
theorem jack_weight : weight_of_jack = 20 := by
  sorry

end NUMINAMATH_GPT_jack_weight_l1674_167459


namespace NUMINAMATH_GPT_Jack_Income_Ratio_l1674_167474

noncomputable def Ernie_current_income (x : ℕ) : ℕ :=
  (4 / 5) * x

noncomputable def Jack_current_income (combined_income Ernie_current_income : ℕ) : ℕ :=
  combined_income - Ernie_current_income

theorem Jack_Income_Ratio (Ernie_previous_income combined_income : ℕ) (h₁ : Ernie_previous_income = 6000) (h₂ : combined_income = 16800) :
  let Ernie_current := Ernie_current_income Ernie_previous_income
  let Jack_current := Jack_current_income combined_income Ernie_current
  (Jack_current / Ernie_previous_income) = 2 := by
  sorry

end NUMINAMATH_GPT_Jack_Income_Ratio_l1674_167474


namespace NUMINAMATH_GPT_tan_x_min_x_div_x_min_sin_x_gt_two_range_of_a_l1674_167408

open Real

-- Part 1
theorem tan_x_min_x_div_x_min_sin_x_gt_two (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) :
  (tan x - x) / (x - sin x) > 2 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → tan x + 2 * sin x - a * x > 0) → a ≤ 3 :=
sorry

end NUMINAMATH_GPT_tan_x_min_x_div_x_min_sin_x_gt_two_range_of_a_l1674_167408


namespace NUMINAMATH_GPT_infinite_series_computation_l1674_167473

theorem infinite_series_computation : 
  ∑' k : ℕ, (8^k) / ((2^k - 1) * (2^(k + 1) - 1)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_computation_l1674_167473


namespace NUMINAMATH_GPT_minimum_value_of_f_l1674_167420

noncomputable def f (x : ℝ) : ℝ := sorry

theorem minimum_value_of_f :
  (∀ x : ℝ, f (x + 1) + f (x - 1) = 2 * x^2 - 4 * x) →
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1674_167420


namespace NUMINAMATH_GPT_sphere_radius_ratio_l1674_167437

theorem sphere_radius_ratio (R r : ℝ) (h₁ : (4 / 3) * Real.pi * R ^ 3 = 450 * Real.pi) (h₂ : (4 / 3) * Real.pi * r ^ 3 = 0.25 * 450 * Real.pi) :
  r / R = 1 / 2 :=
sorry

end NUMINAMATH_GPT_sphere_radius_ratio_l1674_167437


namespace NUMINAMATH_GPT_robert_ate_7_chocolates_l1674_167445

-- Define the number of chocolates Nickel ate
def nickel_chocolates : ℕ := 5

-- Define the number of chocolates Robert ate
def robert_chocolates : ℕ := nickel_chocolates + 2

-- Prove that Robert ate 7 chocolates
theorem robert_ate_7_chocolates : robert_chocolates = 7 := by
    sorry

end NUMINAMATH_GPT_robert_ate_7_chocolates_l1674_167445


namespace NUMINAMATH_GPT_supervisors_per_bus_l1674_167464

theorem supervisors_per_bus (total_supervisors : ℕ) (total_buses : ℕ) (H1 : total_supervisors = 21) (H2 : total_buses = 7) : (total_supervisors / total_buses = 3) :=
by
  sorry

end NUMINAMATH_GPT_supervisors_per_bus_l1674_167464


namespace NUMINAMATH_GPT_car_second_hour_speed_l1674_167461

theorem car_second_hour_speed (x : ℝ) 
  (first_hour_speed : ℝ := 20)
  (average_speed : ℝ := 40) 
  (total_time : ℝ := 2)
  (total_distance : ℝ := first_hour_speed + x) 
  : total_distance / total_time = average_speed → x = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_car_second_hour_speed_l1674_167461


namespace NUMINAMATH_GPT_ones_digit_of_power_35_35_pow_17_17_is_five_l1674_167470

theorem ones_digit_of_power_35_35_pow_17_17_is_five :
  (35 ^ (35 * (17 ^ 17))) % 10 = 5 := by
  sorry

end NUMINAMATH_GPT_ones_digit_of_power_35_35_pow_17_17_is_five_l1674_167470


namespace NUMINAMATH_GPT_tree_height_fraction_l1674_167481

theorem tree_height_fraction :
  ∀ (initial_height growth_per_year : ℝ),
  initial_height = 4 ∧ growth_per_year = 0.5 →
  ((initial_height + 6 * growth_per_year) - (initial_height + 4 * growth_per_year)) / (initial_height + 4 * growth_per_year) = 1 / 6 :=
by
  intros initial_height growth_per_year h
  rcases h with ⟨h1, h2⟩
  sorry

end NUMINAMATH_GPT_tree_height_fraction_l1674_167481


namespace NUMINAMATH_GPT_arithmetic_geometric_seq_l1674_167410

theorem arithmetic_geometric_seq (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
  (h_diff : d = 2)
  (h_geom : (a 1)^2 = a 0 * (a 0 + 6)) :
  a 1 = -6 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_seq_l1674_167410


namespace NUMINAMATH_GPT_triangle_formation_inequalities_l1674_167472

theorem triangle_formation_inequalities (a b c d : ℝ)
  (h_abc_pos : 0 < a)
  (h_bcd_pos : 0 < b)
  (h_cde_pos : 0 < c)
  (h_def_pos : 0 < d)
  (tri_ineq_1 : a + b + c > d)
  (tri_ineq_2 : b + c + d > a)
  (tri_ineq_3 : a + d > b + c) :
  (a < (b + c + d) / 2) ∧ (b + c < a + d) ∧ (¬ (c + d < b / 2)) :=
by 
  sorry

end NUMINAMATH_GPT_triangle_formation_inequalities_l1674_167472


namespace NUMINAMATH_GPT_bisectors_form_inscribed_quadrilateral_l1674_167402

noncomputable def angle_sum_opposite_bisectors {α β γ δ : ℝ} (a_bisector b_bisector c_bisector d_bisector : ℝ)
  (cond : α + β + γ + δ = 360) : Prop :=
  (a_bisector + b_bisector + c_bisector + d_bisector) = 180

theorem bisectors_form_inscribed_quadrilateral
  {α β γ δ : ℝ} (convex_quad : α + β + γ + δ = 360) :
  ∃ a_bisector b_bisector c_bisector d_bisector : ℝ,
  angle_sum_opposite_bisectors a_bisector b_bisector c_bisector d_bisector convex_quad := 
sorry

end NUMINAMATH_GPT_bisectors_form_inscribed_quadrilateral_l1674_167402


namespace NUMINAMATH_GPT_total_lives_remaining_l1674_167467

theorem total_lives_remaining (initial_players quit_players : Nat) 
  (lives_3_players lives_4_players lives_2_players bonus_lives : Nat)
  (h1 : initial_players = 16)
  (h2 : quit_players = 7)
  (h3 : lives_3_players = 10)
  (h4 : lives_4_players = 8)
  (h5 : lives_2_players = 6)
  (h6 : bonus_lives = 4)
  (remaining_players : Nat)
  (h7 : remaining_players = initial_players - quit_players)
  (lives_before_bonus : Nat)
  (h8 : lives_before_bonus = 3 * lives_3_players + 4 * lives_4_players + 2 * lives_2_players)
  (bonus_total : Nat)
  (h9 : bonus_total = remaining_players * bonus_lives) :
  3 * lives_3_players + 4 * lives_4_players + 2 * lives_2_players + remaining_players * bonus_lives = 110 :=
by
  sorry

end NUMINAMATH_GPT_total_lives_remaining_l1674_167467


namespace NUMINAMATH_GPT_inequality_proof_l1674_167450

theorem inequality_proof (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hx1y1 : x1 * y1 - z1^2 > 0) (hx2y2 : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 - z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1674_167450


namespace NUMINAMATH_GPT_joyce_new_property_is_10_times_larger_l1674_167401

theorem joyce_new_property_is_10_times_larger :
  let previous_property := 2
  let suitable_acres := 19
  let pond := 1
  let new_property := suitable_acres + pond
  new_property / previous_property = 10 := by {
    let previous_property := 2
    let suitable_acres := 19
    let pond := 1
    let new_property := suitable_acres + pond
    sorry
  }

end NUMINAMATH_GPT_joyce_new_property_is_10_times_larger_l1674_167401


namespace NUMINAMATH_GPT_find_e_l1674_167458

variables (j p t b a : ℝ) (e : ℝ)

theorem find_e
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (e / 100) * p)
  (h4 : b = 1.40 * j)
  (h5 : a = 0.85 * b)
  (h6 : e = 2 * ((p - a) / p) * 100) :
  e = 21.5 := by
  sorry

end NUMINAMATH_GPT_find_e_l1674_167458


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1674_167439

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 4 + a 7 = 39) (h2 : a 2 + a 5 + a 8 = 33) :
  a 3 + a 6 + a 9 = 27 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1674_167439


namespace NUMINAMATH_GPT_question_a_question_b_l1674_167482

-- Definitions
def isSolutionA (a b : ℤ) : Prop :=
  1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 7

def isSolutionB (a b : ℤ) : Prop :=
  1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 25

-- Statements
theorem question_a (a b : ℤ) : isSolutionA a b ↔ (a, b) ∈ [(6, -42), (-42, 6), (8, 56), (56, 8), (14, 14)] :=
sorry

theorem question_b (a b : ℤ) : isSolutionB a b ↔ (a, b) ∈ [(24, -600), (-600, 24), (26, 650), (650, 26), (50, 50)] :=
sorry

end NUMINAMATH_GPT_question_a_question_b_l1674_167482


namespace NUMINAMATH_GPT_problem_a_b_n_geq_1_l1674_167478

theorem problem_a_b_n_geq_1 (a b n : ℕ) (h1 : a > b) (h2 : b > 1) (h3 : Odd b) (h4 : n > 0)
  (h5 : b^n ∣ a^n - 1) : a^b > 3^n / n := 
by 
  sorry

end NUMINAMATH_GPT_problem_a_b_n_geq_1_l1674_167478


namespace NUMINAMATH_GPT_pushkin_family_pension_l1674_167483

def is_survivor_pension (pension : String) (main_provider_deceased : Bool) (provision_lifelong : Bool) (assigned_to_family : Bool) : Prop :=
  pension = "survivor's pension" ↔
    main_provider_deceased = true ∧
    provision_lifelong = true ∧
    assigned_to_family = true

theorem pushkin_family_pension :
  ∀ (pension : String),
    let main_provider_deceased := true
    let provision_lifelong := true
    let assigned_to_family := true
    is_survivor_pension pension main_provider_deceased provision_lifelong assigned_to_family →
    pension = "survivor's pension" :=
by
  intros pension
  intro h
  sorry

end NUMINAMATH_GPT_pushkin_family_pension_l1674_167483


namespace NUMINAMATH_GPT_larger_pile_toys_l1674_167479

-- Define the conditions
def total_toys (small_pile large_pile : ℕ) : Prop := small_pile + large_pile = 120
def larger_pile (small_pile large_pile : ℕ) : Prop := large_pile = 2 * small_pile

-- Define the proof problem
theorem larger_pile_toys (small_pile large_pile : ℕ) (h1 : total_toys small_pile large_pile) (h2 : larger_pile small_pile large_pile) : 
  large_pile = 80 := by
  sorry

end NUMINAMATH_GPT_larger_pile_toys_l1674_167479


namespace NUMINAMATH_GPT_rectangle_diagonal_ratio_l1674_167495

theorem rectangle_diagonal_ratio (s : ℝ) :
  let d := (Real.sqrt 2) * s
  let D := (Real.sqrt 10) * s
  D / d = Real.sqrt 5 :=
by
  let d := (Real.sqrt 2) * s
  let D := (Real.sqrt 10) * s
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_ratio_l1674_167495


namespace NUMINAMATH_GPT_paving_cost_l1674_167455

def length : Real := 5.5
def width : Real := 3.75
def rate : Real := 700
def area : Real := length * width
def cost : Real := area * rate

theorem paving_cost :
  cost = 14437.50 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_paving_cost_l1674_167455


namespace NUMINAMATH_GPT_triangle_angle_bisectors_l1674_167424

theorem triangle_angle_bisectors (α β γ : ℝ) 
  (h1 : α + β + γ = 180)
  (h2 : α = 100) 
  (h3 : β = 30) 
  (h4 : γ = 50) :
  ∃ α' β' γ', α' = 40 ∧ β' = 65 ∧ γ' = 75 :=
sorry

end NUMINAMATH_GPT_triangle_angle_bisectors_l1674_167424


namespace NUMINAMATH_GPT_average_visitors_per_day_in_november_l1674_167469
-- Import the entire Mathlib library for necessary definitions and operations.

-- Define the average visitors per different days of the week.
def sunday_visitors := 510
def monday_visitors := 240
def tuesday_visitors := 240
def wednesday_visitors := 300
def thursday_visitors := 300
def friday_visitors := 200
def saturday_visitors := 200

-- Define the counts of each type of day in November.
def sundays := 5
def mondays := 4
def tuesdays := 4
def wednesdays := 4
def thursdays := 4
def fridays := 4
def saturdays := 4

-- Define the number of days in November.
def days_in_november := 30

-- State the theorem to prove the average number of visitors per day.
theorem average_visitors_per_day_in_november : 
  (5 * sunday_visitors + 
   4 * monday_visitors + 
   4 * tuesday_visitors + 
   4 * wednesday_visitors + 
   4 * thursday_visitors + 
   4 * friday_visitors + 
   4 * saturday_visitors) / days_in_november = 282 :=
by
  sorry

end NUMINAMATH_GPT_average_visitors_per_day_in_november_l1674_167469


namespace NUMINAMATH_GPT_quadratic_monotonic_range_l1674_167421

theorem quadratic_monotonic_range {t : ℝ} (h : ∀ x1 x2 : ℝ, (1 < x1 ∧ x1 < 3) → (1 < x2 ∧ x2 < 3) → x1 < x2 → (x1^2 - 2 * t * x1 + 1 ≤ x2^2 - 2 * t * x2 + 1)) : 
  t ≤ 1 ∨ t ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_monotonic_range_l1674_167421


namespace NUMINAMATH_GPT_base_eight_conversion_l1674_167430

theorem base_eight_conversion :
  (1 * 8^2 + 3 * 8^1 + 2 * 8^0 = 90) := by
  sorry

end NUMINAMATH_GPT_base_eight_conversion_l1674_167430


namespace NUMINAMATH_GPT_cost_price_of_article_l1674_167435

theorem cost_price_of_article (x : ℝ) (h : 57 - x = x - 43) : x = 50 :=
by sorry

end NUMINAMATH_GPT_cost_price_of_article_l1674_167435


namespace NUMINAMATH_GPT_sum_of_numbers_l1674_167460

theorem sum_of_numbers (a b : ℕ) (h : a + 4 * b = 30) : a + b = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l1674_167460


namespace NUMINAMATH_GPT_intersection_of_sets_l1674_167412

open Set Real

theorem intersection_of_sets :
  let M := {x : ℝ | x ≤ 4}
  let N := {x : ℝ | x > 0}
  M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1674_167412


namespace NUMINAMATH_GPT_count_ways_to_write_2010_l1674_167453

theorem count_ways_to_write_2010 : ∃ N : ℕ, 
  (∀ (a_3 a_2 a_1 a_0 : ℕ), a_0 ≤ 99 ∧ a_1 ≤ 99 ∧ a_2 ≤ 99 ∧ a_3 ≤ 99 → 
    2010 = a_3 * 10^3 + a_2 * 10^2 + a_1 * 10 + a_0) ∧ 
    N = 202 :=
sorry

end NUMINAMATH_GPT_count_ways_to_write_2010_l1674_167453


namespace NUMINAMATH_GPT_middle_joints_capacity_l1674_167438

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def bamboo_tube_capacity (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 4.5 ∧ a 6 + a 7 + a 8 = 2.5 ∧ arithmetic_seq a (a 1 - a 0)

theorem middle_joints_capacity (a : ℕ → ℝ) (d : ℝ) (h : bamboo_tube_capacity a) : 
  a 3 + a 4 + a 5 = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_middle_joints_capacity_l1674_167438


namespace NUMINAMATH_GPT_cubes_with_one_face_painted_cubes_with_two_faces_painted_size_of_new_cube_l1674_167431

def cube (n : ℕ) : Type := ℕ × ℕ × ℕ

-- Define a 4x4x4 cube and the painting conditions
def four_by_four_cube := cube 4

-- Determine the number of small cubes with exactly one face painted
theorem cubes_with_one_face_painted : 
  ∃ (count : ℕ), count = 24 :=
by
  -- proof goes here
  sorry

-- Determine the number of small cubes with exactly two faces painted
theorem cubes_with_two_faces_painted : 
  ∃ (count : ℕ), count = 24 :=
by
  -- proof goes here
  sorry

-- Given condition and find the size of the new cube
theorem size_of_new_cube (n : ℕ) : 
  (n - 2) ^ 3 = 3 * 12 * (n - 2) → n = 8 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_cubes_with_one_face_painted_cubes_with_two_faces_painted_size_of_new_cube_l1674_167431


namespace NUMINAMATH_GPT_apple_street_length_l1674_167484

theorem apple_street_length :
  ∀ (n : ℕ) (d : ℕ), 
    (n = 15) → (d = 200) → 
    (∃ l : ℝ, (l = ((n + 1) * d) / 1000) ∧ l = 3.2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_apple_street_length_l1674_167484


namespace NUMINAMATH_GPT_total_area_is_71_l1674_167405

noncomputable def area_of_combined_regions 
  (PQ QR RS TU : ℕ) 
  (PQRSTU_is_rectangle : true) 
  (right_angles : true): ℕ :=
  let Area_PQRSTU := PQ * QR
  let VU := TU - PQ
  let WT := TU - RS
  let Area_triangle_PVU := (1 / 2) * VU * PQ
  let Area_triangle_RWT := (1 / 2) * WT * RS
  Area_PQRSTU + Area_triangle_PVU + Area_triangle_RWT

theorem total_area_is_71
  (PQ QR RS TU : ℕ) 
  (h1 : PQ = 8)
  (h2 : QR = 6)
  (h3 : RS = 5)
  (h4 : TU = 10)
  (PQRSTU_is_rectangle : true)
  (right_angles : true) :
  area_of_combined_regions PQ QR RS TU PQRSTU_is_rectangle right_angles = 71 :=
by
  -- The proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_total_area_is_71_l1674_167405


namespace NUMINAMATH_GPT_fruits_eaten_l1674_167496

theorem fruits_eaten (initial_cherries initial_strawberries initial_blueberries left_cherries left_strawberries left_blueberries : ℕ)
  (h1 : initial_cherries = 16) (h2 : initial_strawberries = 10) (h3 : initial_blueberries = 20)
  (h4 : left_cherries = 6) (h5 : left_strawberries = 8) (h6 : left_blueberries = 15) :
  (initial_cherries - left_cherries) + (initial_strawberries - left_strawberries) + (initial_blueberries - left_blueberries) = 17 := 
by
  sorry

end NUMINAMATH_GPT_fruits_eaten_l1674_167496


namespace NUMINAMATH_GPT_total_heads_l1674_167427

/-- There are H hens and C cows. Each hen has 1 head and 2 feet, and each cow has 1 head and 4 feet.
Given that the total number of feet is 140 and there are 26 hens, prove that the total number of heads is 48. -/
theorem total_heads (H C : ℕ) (h1 : 2 * H + 4 * C = 140) (h2 : H = 26) : H + C = 48 := by
  sorry

end NUMINAMATH_GPT_total_heads_l1674_167427


namespace NUMINAMATH_GPT_range_of_m_l1674_167440

open Real

theorem range_of_m (m : ℝ) : (¬ ∃ x₀ : ℝ, m * x₀^2 + m * x₀ + 1 ≤ 0) ↔ (0 ≤ m ∧ m < 4) := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1674_167440


namespace NUMINAMATH_GPT_max_chocolates_l1674_167433

theorem max_chocolates (b c k : ℕ) (h1 : b + c = 36) (h2 : c = k * b) (h3 : k > 0) : b ≤ 18 :=
sorry

end NUMINAMATH_GPT_max_chocolates_l1674_167433


namespace NUMINAMATH_GPT_combined_shoe_size_l1674_167436

-- Definitions based on conditions
def Jasmine_size : ℕ := 7
def Alexa_size : ℕ := 2 * Jasmine_size
def Clara_size : ℕ := 3 * Jasmine_size

-- Statement to prove
theorem combined_shoe_size : Jasmine_size + Alexa_size + Clara_size = 42 :=
by
  sorry

end NUMINAMATH_GPT_combined_shoe_size_l1674_167436


namespace NUMINAMATH_GPT_final_temperature_correct_l1674_167490

-- Define the initial conditions
def initial_temperature : ℝ := 12
def decrease_per_hour : ℝ := 5
def time_duration : ℕ := 4

-- Define the expected final temperature
def expected_final_temperature : ℝ := -8

-- The theorem to prove that the final temperature after a given time is as expected
theorem final_temperature_correct :
  initial_temperature + (-decrease_per_hour * time_duration) = expected_final_temperature :=
by
  sorry

end NUMINAMATH_GPT_final_temperature_correct_l1674_167490


namespace NUMINAMATH_GPT_tan_seventeen_pi_over_four_l1674_167452

theorem tan_seventeen_pi_over_four : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_seventeen_pi_over_four_l1674_167452


namespace NUMINAMATH_GPT_sandbag_weight_l1674_167463

theorem sandbag_weight (s : ℝ) (f : ℝ) (h : ℝ) : 
  f = 0.75 ∧ s = 450 ∧ h = 0.65 → f * s + h * (f * s) = 556.875 :=
by
  intro hfs
  sorry

end NUMINAMATH_GPT_sandbag_weight_l1674_167463


namespace NUMINAMATH_GPT_problem_statement_l1674_167489

-- Define the universal set
def U : Set ℕ := {x | x ≤ 6}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {4, 5, 6}

-- Define the complement of A with respect to U
def complement_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- Define the intersection of the complement of A and B
def intersect_complement_A_B : Set ℕ := {x | x ∈ complement_A ∧ x ∈ B}

-- Theorem statement to be proven
theorem problem_statement : intersect_complement_A_B = {4, 6} :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1674_167489


namespace NUMINAMATH_GPT_length_of_short_pieces_l1674_167413

def total_length : ℕ := 27
def long_piece_length : ℕ := 4
def number_of_long_pieces : ℕ := total_length / long_piece_length
def remainder_length : ℕ := total_length % long_piece_length
def number_of_short_pieces : ℕ := 3

theorem length_of_short_pieces (h1 : remainder_length = 3) : (remainder_length / number_of_short_pieces) = 1 :=
by
  sorry

end NUMINAMATH_GPT_length_of_short_pieces_l1674_167413


namespace NUMINAMATH_GPT_find_x_that_satisfies_f_l1674_167468

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem find_x_that_satisfies_f (α : ℝ) (x : ℝ) (h : power_function α (-2) = -1/8) : 
  power_function α x = 27 → x = 1/3 :=
  by
  sorry

end NUMINAMATH_GPT_find_x_that_satisfies_f_l1674_167468


namespace NUMINAMATH_GPT_star_angle_sum_l1674_167416

-- Define variables and angles for Petya's and Vasya's stars.
variables {α β γ δ ε : ℝ}
variables {φ χ ψ ω : ℝ}
variables {a b c d e : ℝ}

-- Conditions
def all_acute (a b c d e : ℝ) : Prop := a < 90 ∧ b < 90 ∧ c < 90 ∧ d < 90 ∧ e < 90
def one_obtuse (a b c d e : ℝ) : Prop := (a > 90 ∨ b > 90 ∨ c > 90 ∨ d > 90 ∨ e > 90)

-- Question: Prove the sum of the angles at the vertices of both stars is equal
theorem star_angle_sum : all_acute α β γ δ ε → one_obtuse φ χ ψ ω α → 
  α + β + γ + δ + ε = φ + χ + ψ + ω + α := 
by sorry

end NUMINAMATH_GPT_star_angle_sum_l1674_167416


namespace NUMINAMATH_GPT_single_ticket_cost_l1674_167487

/-- Define the conditions: sales total, attendee count, number of couple tickets, and cost of couple tickets. -/
def total_sales : ℤ := 2280
def total_attendees : ℕ := 128
def couple_tickets_sold : ℕ := 16
def cost_of_couple_ticket : ℤ := 35

/-- Define the derived conditions: people covered by couple tickets, single tickets sold, and sales from couple tickets. -/
def people_covered_by_couple_tickets : ℕ := couple_tickets_sold * 2
def single_tickets_sold : ℕ := total_attendees - people_covered_by_couple_tickets
def sales_from_couple_tickets : ℤ := couple_tickets_sold * cost_of_couple_ticket

/-- Define the core equation that ties single ticket sales to the total sales. -/
def core_equation (x : ℤ) : Bool := 
  sales_from_couple_tickets + single_tickets_sold * x = total_sales

-- Finally, the statement that needs to be proved.
theorem single_ticket_cost :
  ∃ x : ℤ, core_equation x ∧ x = 18 := by
  sorry

end NUMINAMATH_GPT_single_ticket_cost_l1674_167487


namespace NUMINAMATH_GPT_amoebas_after_ten_days_l1674_167446

def amoeba_split_fun (n : Nat) : Nat := 3^n

theorem amoebas_after_ten_days : amoeba_split_fun 10 = 59049 := by
  have h : 3 ^ 10 = 59049 := by norm_num
  exact h

end NUMINAMATH_GPT_amoebas_after_ten_days_l1674_167446


namespace NUMINAMATH_GPT_perpendicular_lines_parallel_lines_l1674_167419

-- Define the given lines
def l1 (m : ℝ) (x y : ℝ) : ℝ := (m-2)*x + 3*y + 2*m
def l2 (m x y : ℝ) : ℝ := x + m*y + 6

-- The slope conditions for the lines to be perpendicular
def slopes_perpendicular (m : ℝ) : Prop :=
  (m - 2) * m = 3

-- The slope conditions for the lines to be parallel
def slopes_parallel (m : ℝ) : Prop :=
  m = -1

-- Perpendicular lines proof statement
theorem perpendicular_lines (m : ℝ) (x y : ℝ)
  (h1 : l1 m x y = 0)
  (h2 : l2 m x y = 0) :
  slopes_perpendicular m :=
sorry

-- Parallel lines proof statement
theorem parallel_lines (m : ℝ) (x y : ℝ)
  (h1 : l1 m x y = 0)
  (h2 : l2 m x y = 0) :
  slopes_parallel m :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_parallel_lines_l1674_167419


namespace NUMINAMATH_GPT_eat_jar_together_time_l1674_167471

-- Define the rate of the child
def child_rate := 1 / 6

-- Define the rate of Karlson who eats twice as fast as the child
def karlson_rate := 2 * child_rate

-- Define the combined rate when both eat together
def combined_rate := child_rate + karlson_rate

-- Prove that the time taken together to eat one jar is 2 minutes
theorem eat_jar_together_time : (1 / combined_rate) = 2 :=
by
  -- Add the proof steps here
  sorry

end NUMINAMATH_GPT_eat_jar_together_time_l1674_167471


namespace NUMINAMATH_GPT_treasure_chest_age_l1674_167456

theorem treasure_chest_age (n : ℕ) (h : n = 3 * 8^2 + 4 * 8^1 + 7 * 8^0) : n = 231 :=
by
  sorry

end NUMINAMATH_GPT_treasure_chest_age_l1674_167456


namespace NUMINAMATH_GPT_area_of_trapezoid_RSQT_l1674_167451

theorem area_of_trapezoid_RSQT
  (PR PQ : ℝ)
  (PR_eq_PQ : PR = PQ)
  (small_triangle_area : ℝ)
  (total_area : ℝ)
  (num_of_small_triangles : ℕ)
  (num_of_triangles_in_trapezoid : ℕ)
  (area_of_trapezoid : ℝ)
  (is_isosceles_triangle : ∀ (a b c : ℝ), a = b → b = c → a = c)
  (are_similar_triangles : ∀ {A B C D E F : ℝ}, 
    A / B = D / E → A / C = D / F → B / A = E / D → C / A = F / D)
  (smallest_triangle_areas : ∀ {n : ℕ}, n = 9 → small_triangle_area = 2 → num_of_small_triangles = 9)
  (triangle_total_area : ∀ (a : ℝ), a = 72 → total_area = 72)
  (contains_3_small_triangles : ∀ (n : ℕ), n = 3 → num_of_triangles_in_trapezoid = 3)
  (parallel_ST_to_PQ : ∀ {x y z : ℝ}, x = z → y = z → x = y)
  : area_of_trapezoid = 39 :=
sorry

end NUMINAMATH_GPT_area_of_trapezoid_RSQT_l1674_167451


namespace NUMINAMATH_GPT_new_ratio_after_2_years_l1674_167499

-- Definitions based on conditions
variable (A : ℕ) -- Current age of a
variable (B : ℕ) -- Current age of b

-- Conditions
def ratio_a_b := A / B = 5 / 3
def current_age_b := B = 6

-- Theorem: New ratio after 2 years is 3:2
theorem new_ratio_after_2_years (h1 : ratio_a_b A B) (h2 : current_age_b B) : (A + 2) / (B + 2) = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_new_ratio_after_2_years_l1674_167499


namespace NUMINAMATH_GPT_find_third_root_l1674_167477

variables (a b : ℝ)

def poly (x : ℝ) : ℝ := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

def root1 := -3
def root2 := 4

axiom root1_cond : poly a b root1 = 0
axiom root2_cond : poly a b root2 = 0

theorem find_third_root (a b : ℝ) (h1 : poly a b root1 = 0) (h2 : poly a b root2 = 0) : 
  ∃ r3 : ℝ, r3 = -1/2 :=
sorry

end NUMINAMATH_GPT_find_third_root_l1674_167477


namespace NUMINAMATH_GPT_square_rectangle_area_ratio_l1674_167411

theorem square_rectangle_area_ratio (l1 l2 : ℕ) (h1 : l1 = 32) (h2 : l2 = 64) (p : ℕ) (s : ℕ) 
  (h3 : p = 256) (h4 : s = p / 4)  :
  (s * s) / (l1 * l2) = 2 := 
by
  sorry

end NUMINAMATH_GPT_square_rectangle_area_ratio_l1674_167411


namespace NUMINAMATH_GPT_agnes_flight_cost_l1674_167492

theorem agnes_flight_cost
  (booking_fee : ℝ) (cost_per_km : ℝ) (distance_XY : ℝ)
  (h1 : booking_fee = 120)
  (h2 : cost_per_km = 0.12)
  (h3 : distance_XY = 4500) :
  booking_fee + cost_per_km * distance_XY = 660 := 
by
  sorry

end NUMINAMATH_GPT_agnes_flight_cost_l1674_167492


namespace NUMINAMATH_GPT_solve_equation_l1674_167493

open Real

theorem solve_equation (t : ℝ) :
  ¬cos t = 0 ∧ ¬cos (2 * t) = 0 → 
  (tan (2 * t) / (cos t)^2 - tan t / (cos (2 * t))^2 = 0 ↔ 
    (∃ k : ℤ, t = π * ↑k) ∨ (∃ n : ℤ, t = π * ↑n + π / 6) ∨ (∃ n : ℤ, t = π * ↑n - π / 6)) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_equation_l1674_167493


namespace NUMINAMATH_GPT_assertion1_false_assertion2_true_assertion3_false_assertion4_false_l1674_167418

section

-- Assertion 1: ∀ x ∈ ℝ, x ≥ 1 is false
theorem assertion1_false : ¬(∀ x : ℝ, x ≥ 1) := 
sorry

-- Assertion 2: ∃ x ∈ ℕ, x ∈ ℝ is true
theorem assertion2_true : ∃ x : ℕ, (x : ℝ) = x := 
sorry

-- Assertion 3: ∀ x ∈ ℝ, x > 2 → x ≥ 3 is false
theorem assertion3_false : ¬(∀ x : ℝ, x > 2 → x ≥ 3) := 
sorry

-- Assertion 4: ∃ n ∈ ℤ, ∀ x ∈ ℝ, n ≤ x < n + 1 is false
theorem assertion4_false : ¬(∃ n : ℤ, ∀ x : ℝ, n ≤ x ∧ x < n + 1) := 
sorry

end

end NUMINAMATH_GPT_assertion1_false_assertion2_true_assertion3_false_assertion4_false_l1674_167418


namespace NUMINAMATH_GPT_master_bedroom_and_bath_area_l1674_167449

-- Definitions of the problem conditions
def guest_bedroom_area : ℕ := 200
def two_guest_bedrooms_area : ℕ := 2 * guest_bedroom_area
def kitchen_guest_bath_living_area : ℕ := 600
def total_rent : ℕ := 3000
def cost_per_sq_ft : ℕ := 2
def total_area_of_house : ℕ := total_rent / cost_per_sq_ft
def expected_master_bedroom_and_bath_area : ℕ := 500

-- Theorem statement to prove the desired area
theorem master_bedroom_and_bath_area :
  total_area_of_house - (two_guest_bedrooms_area + kitchen_guest_bath_living_area) = expected_master_bedroom_and_bath_area :=
by
  sorry

end NUMINAMATH_GPT_master_bedroom_and_bath_area_l1674_167449


namespace NUMINAMATH_GPT_factor_product_l1674_167428

theorem factor_product : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end NUMINAMATH_GPT_factor_product_l1674_167428


namespace NUMINAMATH_GPT_orchestra_members_l1674_167497

theorem orchestra_members (n : ℕ) (h₀ : 100 ≤ n) (h₁ : n ≤ 300)
    (h₂ : n % 4 = 3) (h₃ : n % 5 = 1) (h₄ : n % 7 = 5) : n = 231 := by
  sorry

end NUMINAMATH_GPT_orchestra_members_l1674_167497


namespace NUMINAMATH_GPT_probability_of_team_A_winning_is_11_over_16_l1674_167422

noncomputable def prob_A_wins_series : ℚ :=
  let total_games := 5
  let wins_needed_A := 2
  let wins_needed_B := 3
  -- Assuming equal probability for each game being won by either team
  let equal_chance_of_winning := 0.5
  -- Calculation would follow similar steps omitted for brevity
  -- Assuming the problem statement proven by external logical steps
  11 / 16

theorem probability_of_team_A_winning_is_11_over_16 :
  prob_A_wins_series = 11 / 16 := 
  sorry

end NUMINAMATH_GPT_probability_of_team_A_winning_is_11_over_16_l1674_167422


namespace NUMINAMATH_GPT_rhombus_area_fraction_l1674_167486

theorem rhombus_area_fraction :
  let grid_area := 36
  let vertices := [(2, 2), (4, 2), (3, 3), (3, 1)]
  let rhombus_area := 2
  rhombus_area / grid_area = 1 / 18 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_fraction_l1674_167486


namespace NUMINAMATH_GPT_max_gcd_of_consecutive_terms_seq_b_l1674_167425

-- Define the sequence b_n
def sequence_b (n : ℕ) : ℕ := n.factorial + 3 * n

-- Define the gcd function for two terms in the sequence
def gcd_two_terms (n : ℕ) : ℕ := Nat.gcd (sequence_b n) (sequence_b (n + 1))

-- Define the condition of n being greater than or equal to 0
def n_ge_zero (n : ℕ) : Prop := n ≥ 0

-- The theorem statement
theorem max_gcd_of_consecutive_terms_seq_b : ∃ n : ℕ, n_ge_zero n ∧ gcd_two_terms n = 14 := 
sorry

end NUMINAMATH_GPT_max_gcd_of_consecutive_terms_seq_b_l1674_167425


namespace NUMINAMATH_GPT_cannot_afford_laptop_l1674_167432

theorem cannot_afford_laptop (P_0 : ℝ) : 56358 < P_0 * (1.06)^2 :=
by
  sorry

end NUMINAMATH_GPT_cannot_afford_laptop_l1674_167432


namespace NUMINAMATH_GPT_other_root_l1674_167415

-- Define the condition that one root of the quadratic equation is -3
def is_root (a b c : ℤ) (x : ℚ) : Prop := a * x^2 + b * x + c = 0

-- Define the quadratic equation 7x^2 + mx - 6 = 0
def quadratic_eq (m : ℤ) (x : ℚ) : Prop := is_root 7 m (-6) x

-- Prove that the other root is 2/7 given that one root is -3
theorem other_root (m : ℤ) (h : quadratic_eq m (-3)) : quadratic_eq m (2 / 7) :=
by
  sorry

end NUMINAMATH_GPT_other_root_l1674_167415


namespace NUMINAMATH_GPT_multiples_of_4_count_l1674_167480

theorem multiples_of_4_count (a b : ℕ) (h₁ : a = 100) (h₂ : b = 400) :
  ∃ n : ℕ, n = 75 ∧ ∀ k : ℕ, (k >= a ∧ k <= b ∧ k % 4 = 0) ↔ (k / 4 - 25 ≥ 1 ∧ k / 4 - 25 ≤ n) :=
sorry

end NUMINAMATH_GPT_multiples_of_4_count_l1674_167480


namespace NUMINAMATH_GPT_reciprocal_of_minus_one_over_2023_l1674_167488

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_minus_one_over_2023_l1674_167488
