import Mathlib

namespace inequalities_l1423_142330

variable {a b c : ℝ}

theorem inequalities (ha : a < 0) (hab : a < b) (hbc : b < c) :
  a^2 * b < b^2 * c ∧ a^2 * c < b^2 * c ∧ a^2 * b < a^2 * c :=
by
  sorry

end inequalities_l1423_142330


namespace divisible_by_six_l1423_142346

theorem divisible_by_six (m : ℕ) : 6 ∣ (m^3 + 11 * m) := 
sorry

end divisible_by_six_l1423_142346


namespace solve_for_j_l1423_142335

variable (j : ℝ)
variable (h1 : j > 0)
variable (v1 : ℝ × ℝ × ℝ := (3, 4, 5))
variable (v2 : ℝ × ℝ × ℝ := (2, j, 3))
variable (v3 : ℝ × ℝ × ℝ := (2, 3, j))

theorem solve_for_j :
  |(3 * (j * j - 3 * 3) - 2 * (4 * j - 5 * 3) + 2 * (4 * 3 - 5 * j))| = 36 →
  j = (9 + Real.sqrt 585) / 6 :=
by
  sorry

end solve_for_j_l1423_142335


namespace inequality_proof_l1423_142384

variables {x y : ℝ}

theorem inequality_proof (hx_pos : x > 0) (hy_pos : y > 0) (h1 : x^2 > x + y) (h2 : x^4 > x^3 + y) : x^3 > x^2 + y := 
by 
  sorry

end inequality_proof_l1423_142384


namespace find_explicit_formula_range_of_k_l1423_142398

variable (a b x k : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 - b * x + 4

theorem find_explicit_formula (h_extremum_at_2 : f a b 2 = -4 / 3 ∧ (3 * a * 4 - b = 0)) :
  ∃ a b, f a b x = (1 / 3) * x ^ 3 - 4 * x + 4 :=
sorry

theorem range_of_k (h_extremum_at_2 : f (1 / 3) 4 2 = -4 / 3) :
  ∃ k, -4 / 3 < k ∧ k < 8 / 3 :=
sorry

end find_explicit_formula_range_of_k_l1423_142398


namespace plastering_cost_l1423_142334

variable (l w d : ℝ) (c : ℝ)

theorem plastering_cost :
  l = 60 → w = 25 → d = 10 → c = 0.90 →
    let A_bottom := l * w;
    let A_long_walls := 2 * (l * d);
    let A_short_walls := 2 * (w * d);
    let A_total := A_bottom + A_long_walls + A_short_walls;
    let C_total := A_total * c;
    C_total = 2880 :=
by sorry

end plastering_cost_l1423_142334


namespace minimum_value_of_2x_3y_l1423_142338

noncomputable def minimum_value (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : (2/x) + (3/y) = 1) : ℝ :=
  2*x + 3*y

theorem minimum_value_of_2x_3y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : (2/x) + (3/y) = 1) : minimum_value x y hx hy hxy = 25 :=
sorry

end minimum_value_of_2x_3y_l1423_142338


namespace lines_are_skew_l1423_142357

def line1 (a t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3 * t, 3 + 4 * t, a + 5 * t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 6 * u, 2 + 2 * u, 1 + 2 * u)

theorem lines_are_skew (a : ℝ) :
  ¬(∃ t u : ℝ, line1 a t = line2 u) ↔ a ≠ 5 / 3 :=
sorry

end lines_are_skew_l1423_142357


namespace decimal_to_vulgar_fraction_l1423_142350

theorem decimal_to_vulgar_fraction (h : (34 / 100 : ℚ) = 0.34) : (0.34 : ℚ) = 17 / 50 := by
  sorry

end decimal_to_vulgar_fraction_l1423_142350


namespace simplify_expression_l1423_142302

-- Define the algebraic expressions
def expr1 (x : ℝ) := (3 * x - 4) * (x + 9)
def expr2 (x : ℝ) := (x + 6) * (3 * x + 2)
def combined_expr (x : ℝ) := expr1 x + expr2 x
def result_expr (x : ℝ) := 6 * x^2 + 43 * x - 24

-- Theorem stating the equivalence
theorem simplify_expression (x : ℝ) : combined_expr x = result_expr x := 
by 
  sorry

end simplify_expression_l1423_142302


namespace cylinder_surface_area_l1423_142351

variable (height1 height2 radius1 radius2 : ℝ)
variable (π : ℝ)
variable (C1 : height1 = 6 * π)
variable (C2 : radius1 = 3)
variable (C3 : height2 = 4 * π)
variable (C4 : radius2 = 2)

theorem cylinder_surface_area : 
  (6 * π * 4 * π + 2 * π * radius1 ^ 2) = 24 * π ^ 2 + 18 * π ∨
  (4 * π * 6 * π + 2 * π * radius2 ^ 2) = 24 * π ^ 2 + 8 * π :=
by
  intros
  sorry

end cylinder_surface_area_l1423_142351


namespace problem_l1423_142306

noncomputable def a : ℝ := Real.log 8 / Real.log 3
noncomputable def b : ℝ := Real.log 25 / Real.log 4
noncomputable def c : ℝ := Real.log 24 / Real.log 4

theorem problem : a < c ∧ c < b :=
by
  sorry

end problem_l1423_142306


namespace sphere_surface_area_l1423_142339

-- Let A, B, C, D be distinct points on the same sphere
variables (A B C D : ℝ)

-- Defining edges AB, AC, AD and their lengths
variables (AB AC AD : ℝ)
variable (is_perpendicular : AB * AC = 0 ∧ AB * AD = 0 ∧ AC * AD = 0)

-- Setting specific edge lengths
variables (AB_length : AB = 1) (AC_length : AC = 2) (AD_length : AD = 3)

-- The proof problem: Prove that the surface area of the sphere is 14π
theorem sphere_surface_area : 4 * Real.pi * ((1 + 4 + 9) / 4) = 14 * Real.pi :=
by
  sorry

end sphere_surface_area_l1423_142339


namespace worker_net_salary_change_l1423_142362

theorem worker_net_salary_change (S : ℝ) :
  let final_salary := S * 1.15 * 0.90 * 1.20 * 0.95
  let net_change := final_salary - S
  net_change = 0.0355 * S := by
  -- Proof goes here
  sorry

end worker_net_salary_change_l1423_142362


namespace min_value_2_l1423_142318

noncomputable def min_value (a b : ℝ) : ℝ :=
  1 / a + 1 / (b + 1)

theorem min_value_2 {a b : ℝ} (h1 : a > 0) (h2 : b > -1) (h3 : a + b = 1) : min_value a b = 2 :=
by
  sorry

end min_value_2_l1423_142318


namespace cosine_third_angle_of_triangle_l1423_142376

theorem cosine_third_angle_of_triangle (X Y Z : ℝ)
  (sinX_eq : Real.sin X = 4/5)
  (cosY_eq : Real.cos Y = 12/13)
  (triangle_sum : X + Y + Z = Real.pi) :
  Real.cos Z = -16/65 :=
by
  -- proof will be filled in
  sorry

end cosine_third_angle_of_triangle_l1423_142376


namespace bruce_total_payment_l1423_142373

def cost_of_grapes (quantity rate : ℕ) : ℕ := quantity * rate
def cost_of_mangoes (quantity rate : ℕ) : ℕ := quantity * rate

theorem bruce_total_payment : 
  cost_of_grapes 8 70 + cost_of_mangoes 11 55 = 1165 :=
by 
  sorry

end bruce_total_payment_l1423_142373


namespace sufficient_but_not_necessary_condition_l1423_142344

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  ((∀ x : ℝ, (1 < x) → (x^2 - m * x + 1 > 0)) ↔ (-2 < m ∧ m < 2)) :=
sorry

end sufficient_but_not_necessary_condition_l1423_142344


namespace expected_heads_64_coins_l1423_142368

noncomputable def expected_heads (n : ℕ) (p : ℚ) : ℚ :=
  n * p

theorem expected_heads_64_coins : expected_heads 64 (15/16) = 60 := by
  sorry

end expected_heads_64_coins_l1423_142368


namespace ab_equals_one_l1423_142305

theorem ab_equals_one {a b : ℝ} (h : a ≠ b) (hf : |Real.log a| = |Real.log b|) : a * b = 1 :=
  sorry

end ab_equals_one_l1423_142305


namespace original_pencils_example_l1423_142372

-- Statement of the problem conditions
def original_pencils (total_pencils : ℕ) (added_pencils : ℕ) : ℕ :=
  total_pencils - added_pencils

-- Theorem we need to prove
theorem original_pencils_example : original_pencils 5 3 = 2 := 
by
  -- Proof
  sorry

end original_pencils_example_l1423_142372


namespace cube_properties_l1423_142327

theorem cube_properties (x : ℝ) (h1 : 6 * (2 * (8 * x)^(1/3))^2 = x) : x = 13824 :=
sorry

end cube_properties_l1423_142327


namespace rectangle_ratio_l1423_142308

theorem rectangle_ratio (s y x : ℝ) (hs : s > 0) (hy : y > 0) (hx : x > 0)
  (h1 : s + 2 * y = 3 * s)
  (h2 : x + y = 3 * s)
  (h3 : y = s)
  (h4 : x = 2 * s) :
  x / y = 2 := by
  sorry

end rectangle_ratio_l1423_142308


namespace velocity_at_1_eq_5_l1423_142358

def S (t : ℝ) : ℝ := 2 * t^2 + t

theorem velocity_at_1_eq_5 : (deriv S 1) = 5 :=
by sorry

end velocity_at_1_eq_5_l1423_142358


namespace pool_capacity_l1423_142309

-- Conditions
variables (C : ℝ) -- total capacity of the pool in gallons
variables (h1 : 300 = 0.75 * C - 0.45 * C) -- the pool requires an additional 300 gallons to be filled to 75%
variables (h2 : 300 = 0.30 * C) -- pumping in these additional 300 gallons will increase the amount of water by 30%

-- Goal
theorem pool_capacity : C = 1000 :=
by sorry

end pool_capacity_l1423_142309


namespace daniel_pages_to_read_l1423_142315

-- Definitions from conditions
def total_pages : ℕ := 980
def daniel_read_time_per_page : ℕ := 50
def emma_read_time_per_page : ℕ := 40

-- The theorem that states the solution
theorem daniel_pages_to_read (d : ℕ) :
  d = 436 ↔ daniel_read_time_per_page * d = emma_read_time_per_page * (total_pages - d) :=
by sorry

end daniel_pages_to_read_l1423_142315


namespace smallest_c_minus_a_l1423_142360

theorem smallest_c_minus_a (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_prod : a * b * c = 362880) (h_ineq : a < b ∧ b < c) : 
  c - a = 109 :=
sorry

end smallest_c_minus_a_l1423_142360


namespace milk_amount_at_beginning_l1423_142314

theorem milk_amount_at_beginning (H: 0.69 = 0.6 * total_milk) : total_milk = 1.15 :=
sorry

end milk_amount_at_beginning_l1423_142314


namespace smallest_nat_div_7_and_11_l1423_142337

theorem smallest_nat_div_7_and_11 (n : ℕ) (h1 : n > 1) (h2 : n % 7 = 1) (h3 : n % 11 = 1) : n = 78 :=
by
  sorry

end smallest_nat_div_7_and_11_l1423_142337


namespace range_of_a_l1423_142321

noncomputable def f (a : ℝ) (x : ℝ) := 2 * a * x^2 + 4 * (a - 3) * x + 5

theorem range_of_a (a : ℝ) :
  (∀ x < 3, f a x ≤ f a (3 : ℝ)) ↔ 0 ≤ a ∧ a ≤ 3 / 4 :=
by
  sorry

end range_of_a_l1423_142321


namespace num_parallelograms_4x6_grid_l1423_142331

noncomputable def numberOfParallelograms (m n : ℕ) : ℕ :=
  let numberOfRectangles := (Nat.choose (m + 1) 2) * (Nat.choose (n + 1) 2)
  let numberOfSquares := (m * n) + ((m - 1) * (n - 1)) + ((m - 2) * (n - 2)) + ((m - 3) * (n - 3))
  let numberOfRectanglesWithUnequalSides := numberOfRectangles - numberOfSquares
  2 * numberOfRectanglesWithUnequalSides

theorem num_parallelograms_4x6_grid : numberOfParallelograms 4 6 = 320 := by
  sorry

end num_parallelograms_4x6_grid_l1423_142331


namespace smallest_positive_m_l1423_142379

theorem smallest_positive_m (m : ℕ) : 
  (∃ n : ℤ, (10 * n * (n + 1) = 600) ∧ (m = 10 * (n + (n + 1)))) → (m = 170) :=
by 
  sorry

end smallest_positive_m_l1423_142379


namespace system_solution_l1423_142393

theorem system_solution (x y : ℝ) (h1 : x + y = 1) (h2 : x - y = 3) : x = 2 ∧ y = -1 :=
by
  sorry

end system_solution_l1423_142393


namespace can_capacity_is_14_l1423_142370

noncomputable def capacity_of_can 
    (initial_milk: ℝ) (initial_water: ℝ) 
    (added_milk: ℝ) (ratio_initial: ℝ) (ratio_final: ℝ): ℝ :=
  initial_milk + initial_water + added_milk

theorem can_capacity_is_14
    (M W: ℝ) 
    (ratio_initial : M / W = 1 / 5) 
    (added_milk : ℝ := 2) 
    (ratio_final:  (M + 2) / W = 2.00001 / 5.00001): 
    capacity_of_can M W added_milk (1 / 5) (2.00001 / 5.00001) = 14 := 
  by
    sorry

end can_capacity_is_14_l1423_142370


namespace min_value_of_expression_l1423_142395

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 1) :
  (2 / x + 5 / y) ≥ 2 := sorry

end min_value_of_expression_l1423_142395


namespace perpendicular_lines_l1423_142394

theorem perpendicular_lines (a : ℝ) : 
  (2 * (a + 1) * a + a * 2 = 0) ↔ (a = -2 ∨ a = 0) :=
by 
  sorry

end perpendicular_lines_l1423_142394


namespace line_points_satisfy_equation_l1423_142385

theorem line_points_satisfy_equation (x_2 y_3 : ℝ) 
  (h_slope : ∃ k : ℝ, k = 2) 
  (h_P1 : ∃ P1 : ℝ × ℝ, P1 = (3, 5)) 
  (h_P2 : ∃ P2 : ℝ × ℝ, P2 = (x_2, 7)) 
  (h_P3 : ∃ P3 : ℝ × ℝ, P3 = (-1, y_3)) 
  (h_line : ∀ (x y : ℝ), y - 5 = 2 * (x - 3) ↔ 2 * x - y - 1 = 0) :
  x_2 = 4 ∧ y_3 = -3 :=
sorry

end line_points_satisfy_equation_l1423_142385


namespace least_sum_of_factors_l1423_142353

theorem least_sum_of_factors (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2400) : a + b = 98 :=
sorry

end least_sum_of_factors_l1423_142353


namespace bruce_michael_total_goals_l1423_142375

theorem bruce_michael_total_goals (bruce_goals : ℕ) (michael_goals : ℕ) 
  (h₁ : bruce_goals = 4) (h₂ : michael_goals = 3 * bruce_goals) : bruce_goals + michael_goals = 16 :=
by sorry

end bruce_michael_total_goals_l1423_142375


namespace jackson_sandwiches_l1423_142323

noncomputable def total_sandwiches (weeks : ℕ) (miss_wed : ℕ) (miss_fri : ℕ) : ℕ :=
  let total_wednesdays := weeks - miss_wed
  let total_fridays := weeks - miss_fri
  total_wednesdays + total_fridays

theorem jackson_sandwiches : total_sandwiches 36 1 2 = 69 := by
  sorry

end jackson_sandwiches_l1423_142323


namespace rectangle_area_192_l1423_142371

variable (b l : ℝ) (A : ℝ)

-- Conditions
def length_is_thrice_breadth : Prop :=
  l = 3 * b

def perimeter_is_64 : Prop :=
  2 * (l + b) = 64

-- Area calculation
def area_of_rectangle : ℝ :=
  l * b

theorem rectangle_area_192 (h1 : length_is_thrice_breadth b l) (h2 : perimeter_is_64 b l) :
  area_of_rectangle l b = 192 := by
  sorry

end rectangle_area_192_l1423_142371


namespace train_cars_count_l1423_142336

theorem train_cars_count
  (cars_in_first_15_seconds : ℕ)
  (time_for_first_5_cars : ℕ)
  (total_time_to_pass : ℕ)
  (h_cars_in_first_15_seconds : cars_in_first_15_seconds = 5)
  (h_time_for_first_5_cars : time_for_first_5_cars = 15)
  (h_total_time_to_pass : total_time_to_pass = 210) :
  (total_time_to_pass / time_for_first_5_cars) * cars_in_first_15_seconds = 70 := 
by 
  sorry

end train_cars_count_l1423_142336


namespace angle_measure_of_E_l1423_142369

theorem angle_measure_of_E (E F G H : ℝ) 
  (h1 : E = 3 * F) 
  (h2 : E = 4 * G) 
  (h3 : E = 6 * H) 
  (h_sum : E + F + G + H = 360) : 
  E = 206 := 
by 
  sorry

end angle_measure_of_E_l1423_142369


namespace ball_distribution_l1423_142396

theorem ball_distribution :
  ∃ (f : ℕ → ℕ → ℕ → Prop), 
    (∀ x1 x2 x3, f x1 x2 x3 → x1 + x2 + x3 = 10 ∧ x1 ≥ 1 ∧ x2 ≥ 2 ∧ x3 ≥ 3) ∧
    (∃ (count : ℕ), (count = 15) ∧ (∀ x1 x2 x3, f x1 x2 x3 → count = 15)) :=
sorry

end ball_distribution_l1423_142396


namespace age_of_other_man_replaced_l1423_142328

-- Define the conditions
variables (A : ℝ) (x : ℝ)
variable (average_age_women : ℝ := 50)
variable (num_men : ℕ := 10)
variable (increase_age : ℝ := 6)
variable (one_man_age : ℝ := 22)

-- State the theorem to be proved
theorem age_of_other_man_replaced :
  2 * average_age_women - (one_man_age + x) = 10 * (A + increase_age) - 10 * A →
  x = 18 :=
by
  sorry

end age_of_other_man_replaced_l1423_142328


namespace minimum_slope_tangent_point_coordinates_l1423_142340

theorem minimum_slope_tangent_point_coordinates :
  ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, (2 * x + a / x ≥ 4) ∧ (2 * x + a / x = 4 ↔ x = 1)) → 
  (1, 1) = (1, 1) := by
sorry

end minimum_slope_tangent_point_coordinates_l1423_142340


namespace paper_clips_in_2_cases_l1423_142391

variable (c b : ℕ)

theorem paper_clips_in_2_cases : 2 * (c * b) * 600 = (2 * c * b * 600) := by
  sorry

end paper_clips_in_2_cases_l1423_142391


namespace find_min_n_l1423_142320

variable (a : Nat → Int)
variable (S : Nat → Int)
variable (d : Nat)
variable (n : Nat)

-- Definitions based on given conditions
def arithmetic_sequence (a : Nat → Int) (d : Nat) : Prop :=
  ∀ n, a (n + 1) = a n + d

def a1_eq_neg3 (a : Nat → Int) : Prop :=
  a 1 = -3

def condition (a : Nat → Int) (d : Nat) : Prop :=
  11 * a 5 = 5 * a 8

-- Correct answer condition
def minimized_sum_condition (a : Nat → Int) (S : Nat → Int) (d : Nat) (n : Nat) : Prop :=
  S n ≤ S (n + 1)

theorem find_min_n (a : Nat → Int) (S : Nat → Int) (d : Nat) :
  arithmetic_sequence a d ->
  a1_eq_neg3 a ->
  condition a 2 ->
  minimized_sum_condition a S 2 2 :=
by
  sorry

end find_min_n_l1423_142320


namespace no_all_nine_odd_l1423_142316

theorem no_all_nine_odd
  (a1 a2 a3 a4 a5 b1 b2 b3 b4 : ℤ)
  (h1 : a1 % 2 = 1) (h2 : a2 % 2 = 1) (h3 : a3 % 2 = 1)
  (h4 : a4 % 2 = 1) (h5 : a5 % 2 = 1) (h6 : b1 % 2 = 1)
  (h7 : b2 % 2 = 1) (h8 : b3 % 2 = 1) (h9 : b4 % 2 = 1)
  (sum_eq : a1 + a2 + a3 + a4 + a5 = b1 + b2 + b3 + b4) : 
  false :=
sorry

end no_all_nine_odd_l1423_142316


namespace determine_number_of_students_l1423_142313

theorem determine_number_of_students 
  (n : ℕ) 
  (h1 : n < 600) 
  (h2 : n % 25 = 24) 
  (h3 : n % 19 = 15) : 
  n = 399 :=
by
  -- The proof will be provided here.
  sorry

end determine_number_of_students_l1423_142313


namespace inequality_sum_of_reciprocals_l1423_142352

variable {a b c : ℝ}

theorem inequality_sum_of_reciprocals
  (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
  (hsum : a + b + c = 3) :
  (1 / (2 * a^2 + b^2 + c^2) + 1 / (2 * b^2 + c^2 + a^2) + 1 / (2 * c^2 + a^2 + b^2)) ≤ 3/4 :=
sorry

end inequality_sum_of_reciprocals_l1423_142352


namespace find_sin_value_l1423_142383

variable (x : ℝ)

theorem find_sin_value (h : Real.sin (x + Real.pi / 3) = Real.sqrt 3 / 3) : 
  Real.sin (2 * Real.pi / 3 - x) = Real.sqrt 3 / 3 :=
by 
  sorry

end find_sin_value_l1423_142383


namespace complex_pow_i_2019_l1423_142312

theorem complex_pow_i_2019 : (Complex.I)^2019 = -Complex.I := 
by
  sorry

end complex_pow_i_2019_l1423_142312


namespace find_GQ_in_triangle_XYZ_l1423_142389

noncomputable def GQ_in_triangle_XYZ_centroid : ℝ :=
  let XY := 13
  let XZ := 15
  let YZ := 24
  let centroid_ratio := 1 / 3
  let semi_perimeter := (XY + XZ + YZ) / 2
  let area := Real.sqrt (semi_perimeter * (semi_perimeter - XY) * (semi_perimeter - XZ) * (semi_perimeter - YZ))
  let heightXR := (2 * area) / YZ
  (heightXR * centroid_ratio)

theorem find_GQ_in_triangle_XYZ :
  GQ_in_triangle_XYZ_centroid = 2.4 :=
sorry

end find_GQ_in_triangle_XYZ_l1423_142389


namespace base_5_minus_base_8_in_base_10_l1423_142381

def base_5 := 52143
def base_8 := 4310

theorem base_5_minus_base_8_in_base_10 :
  (5 * 5^4 + 2 * 5^3 + 1 * 5^2 + 4 * 5^1 + 3 * 5^0) -
  (4 * 8^3 + 3 * 8^2 + 1 * 8^1 + 0 * 8^0)
  = 1175 := by
  sorry

end base_5_minus_base_8_in_base_10_l1423_142381


namespace number_of_friends_l1423_142378

theorem number_of_friends (n : ℕ) (h1 : 100 % n = 0) (h2 : 100 % (n + 5) = 0) (h3 : 100 / n - 1 = 100 / (n + 5)) : n = 20 :=
by
  sorry

end number_of_friends_l1423_142378


namespace range_of_a_l1423_142343

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x + a / x + 7 else x + a / x - 7

theorem range_of_a (a : ℝ) (ha : 0 < a)
  (hodd : ∀ x : ℝ, f (-x) a = -f x a)
  (hcond : ∀ x : ℝ, 0 ≤ x → f x a ≥ 1 - a) :
  4 ≤ a := sorry

end range_of_a_l1423_142343


namespace paying_students_pay_7_l1423_142317

/-- At a school, 40% of the students receive a free lunch. 
These lunches are paid for by making sure the price paid by the 
paying students is enough to cover everyone's meal. 
It costs $210 to feed 50 students. 
Prove that each paying student pays $7. -/
theorem paying_students_pay_7 (total_students : ℕ) 
  (free_lunch_percentage : ℤ)
  (cost_per_50_students : ℕ) : 
  free_lunch_percentage = 40 ∧ cost_per_50_students = 210 →
  ∃ (paying_students_pay : ℕ), paying_students_pay = 7 :=
by
  -- Let the proof steps and conditions be set up as follows
  -- (this part is not required, hence using sorry)
  sorry

end paying_students_pay_7_l1423_142317


namespace animals_total_sleep_in_one_week_l1423_142397

-- Define the conditions
def cougar_sleep_per_night := 4 -- Cougar sleeps 4 hours per night
def zebra_extra_sleep := 2 -- Zebra sleeps 2 hours more than cougar

-- Calculate the sleep duration for the zebra
def zebra_sleep_per_night := cougar_sleep_per_night + zebra_extra_sleep

-- Total sleep duration per week
def week_nights := 7

-- Total weekly sleep durations
def cougar_weekly_sleep := cougar_sleep_per_night * week_nights
def zebra_weekly_sleep := zebra_sleep_per_night * week_nights

-- Total sleep time for both animals in one week
def total_weekly_sleep := cougar_weekly_sleep + zebra_weekly_sleep

-- The target theorem
theorem animals_total_sleep_in_one_week : total_weekly_sleep = 70 := by
  sorry

end animals_total_sleep_in_one_week_l1423_142397


namespace length_of_rectangle_l1423_142322

theorem length_of_rectangle (P L B : ℕ) (h₁ : P = 800) (h₂ : B = 300) (h₃ : P = 2 * (L + B)) : L = 100 := by
  sorry

end length_of_rectangle_l1423_142322


namespace inequality_proof_l1423_142361

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c ≤ 3) : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l1423_142361


namespace train_speed_problem_l1423_142390

theorem train_speed_problem (l1 l2 : ℝ) (v2 : ℝ) (t : ℝ) (v1 : ℝ) :
  l1 = 120 → l2 = 280 → v2 = 30 → t = 19.99840012798976 →
  0.4 / (t / 3600) = v1 + v2 → v1 = 42 :=
by
  intros hl1 hl2 hv2 ht hrel
  rw [hl1, hl2, hv2, ht] at *
  sorry

end train_speed_problem_l1423_142390


namespace handshake_count_l1423_142310

-- Definitions based on conditions
def groupA_size : ℕ := 25
def groupB_size : ℕ := 15

-- Total number of handshakes is calculated as product of their sizes
def total_handshakes : ℕ := groupA_size * groupB_size

-- The theorem we need to prove
theorem handshake_count : total_handshakes = 375 :=
by
  -- skipped proof
  sorry

end handshake_count_l1423_142310


namespace distance_y_axis_l1423_142354

def point_M (m : ℝ) : ℝ × ℝ := (2 - m, 1 + 2 * m)

theorem distance_y_axis :
  ∀ m : ℝ, abs (2 - m) = 2 → (point_M m = (2, 1)) ∨ (point_M m = (-2, 9)) :=
by
  sorry

end distance_y_axis_l1423_142354


namespace inequality_correct_l1423_142319

variable (m n c : ℝ)

theorem inequality_correct (h : m > n) : m + c > n + c := 
by sorry

end inequality_correct_l1423_142319


namespace min_students_in_class_l1423_142333

noncomputable def min_possible_students (b g : ℕ) : Prop :=
  (3 * b) / 4 = 2 * (2 * g) / 3 ∧ b = (16 * g) / 9

theorem min_students_in_class : ∃ (b g : ℕ), min_possible_students b g ∧ b + g = 25 :=
by
  sorry

end min_students_in_class_l1423_142333


namespace distance_from_point_to_focus_l1423_142363

theorem distance_from_point_to_focus (P : ℝ × ℝ) (hP : P.2^2 = 8 * P.1) (hX : P.1 = 8) :
  dist P (2, 0) = 10 :=
sorry

end distance_from_point_to_focus_l1423_142363


namespace phone_numbers_divisible_by_13_l1423_142300

theorem phone_numbers_divisible_by_13 :
  ∃ (x y z : ℕ), (x < 10) ∧ (y < 10) ∧ (z < 10) ∧ (100 * x + 10 * y + z) % 13 = 0 ∧ (2 * y = x + z) :=
  sorry

end phone_numbers_divisible_by_13_l1423_142300


namespace slope_of_line_dividing_rectangle_l1423_142342

theorem slope_of_line_dividing_rectangle (h_vertices : 
  ∃ (A B C D : ℝ × ℝ), A = (1, 0) ∧ B = (9, 0) ∧ C = (1, 2) ∧ D = (9, 2) ∧ 
  (∃ line : ℝ × ℝ, line = (0, 0) ∧ line = (5, 1))) : 
  ∃ m : ℝ, m = 1 / 5 :=
sorry

end slope_of_line_dividing_rectangle_l1423_142342


namespace find_x_l1423_142324

-- Definitions for the problem
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b : ℝ × ℝ := (-2, 1)

-- Theorem statement
theorem find_x (x : ℝ) (h : ∃ k : ℝ, a x = k • b) : x = -1/2 := by
  sorry

end find_x_l1423_142324


namespace range_of_a_for_critical_point_l1423_142303

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * Real.log x

theorem range_of_a_for_critical_point :
  ∀ a : ℝ, (∃ x ∈ Set.Icc (a - 1) (a + 1), deriv f x = 0) ↔ 2 < a ∧ a < 4 :=
by
  sorry

end range_of_a_for_critical_point_l1423_142303


namespace q_sufficient_not_necessary_for_p_l1423_142311

def p (x : ℝ) : Prop := abs x < 2
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

theorem q_sufficient_not_necessary_for_p (x : ℝ) : (q x → p x) ∧ ¬(p x → q x) := 
by
  sorry

end q_sufficient_not_necessary_for_p_l1423_142311


namespace find_b_from_root_l1423_142301

theorem find_b_from_root (b : ℝ) :
  (Polynomial.eval (-10) (Polynomial.C 1 * X^2 + Polynomial.C b * X + Polynomial.C (-30)) = 0) →
  b = 7 :=
by
  intro h
  sorry

end find_b_from_root_l1423_142301


namespace sum_congruent_mod_9_l1423_142341

theorem sum_congruent_mod_9 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by 
  -- Proof steps here
  sorry

end sum_congruent_mod_9_l1423_142341


namespace moss_flower_pollen_scientific_notation_l1423_142359

theorem moss_flower_pollen_scientific_notation (d : ℝ) (h : d = 0.0000084) : ∃ n : ℤ, d = 8.4 * 10^n ∧ n = -6 :=
by
  use -6
  rw [h]
  simp
  sorry

end moss_flower_pollen_scientific_notation_l1423_142359


namespace simplify_expression_l1423_142347

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end simplify_expression_l1423_142347


namespace points_per_other_player_l1423_142364

-- Define the conditions as variables
variables (total_points : ℕ) (faye_points : ℕ) (total_players : ℕ)

-- Assume the given conditions
def conditions : Prop :=
  total_points = 68 ∧ faye_points = 28 ∧ total_players = 5

-- Define the proof problem: Prove that the points scored by each of the other players is 10
theorem points_per_other_player :
  conditions total_points faye_points total_players →
  (total_points - faye_points) / (total_players - 1) = 10 :=
by
  sorry

end points_per_other_player_l1423_142364


namespace planks_needed_for_surface_l1423_142365

theorem planks_needed_for_surface
  (total_tables : ℕ := 5)
  (total_planks : ℕ := 45)
  (planks_per_leg : ℕ := 4) :
  ∃ S : ℕ, total_tables * (planks_per_leg + S) = total_planks ∧ S = 5 :=
by
  use 5
  sorry

end planks_needed_for_surface_l1423_142365


namespace conical_tank_volume_l1423_142374

theorem conical_tank_volume
  (diameter : ℝ) (height : ℝ) (depth_linear : ∀ x : ℝ, 0 ≤ x ∧ x ≤ diameter / 2 → height - (height / (diameter / 2)) * x = 0) :
  diameter = 20 → height = 6 → (1 / 3) * Real.pi * (10 ^ 2) * height = 200 * Real.pi :=
by
  sorry

end conical_tank_volume_l1423_142374


namespace dressing_q_vinegar_percentage_l1423_142304

/-- 
Given:
1. P is 30% vinegar and 70% oil.
2. Q is V% vinegar and the rest is oil.
3. The new dressing is produced from 10% of P and 90% of Q and is 12% vinegar.
Prove:
The percentage of vinegar in dressing Q is 10%.
-/
theorem dressing_q_vinegar_percentage (V : ℝ) (h : 0.10 * 0.30 + 0.90 * V = 0.12) : V = 0.10 :=
by 
    sorry

end dressing_q_vinegar_percentage_l1423_142304


namespace problem_one_problem_two_l1423_142388

-- Define the given vectors
def vector_oa : ℝ × ℝ := (-1, 3)
def vector_ob : ℝ × ℝ := (3, -1)
def vector_oc (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define the subtraction of two 2D vectors
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

-- Define the parallel condition (u and v are parallel if u = k*v for some scalar k)
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1  -- equivalent to u = k*v

-- Define the dot product in 2D
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Problem 1
theorem problem_one (m : ℝ) :
  is_parallel (vector_sub vector_ob vector_oa) (vector_oc m) ↔ m = -1 :=
by
-- Proof omitted
sorry

-- Problem 2
theorem problem_two (m : ℝ) :
  dot_product (vector_sub (vector_oc m) vector_oa) (vector_sub (vector_oc m) vector_ob) = 0 ↔
  m = 1 + 2 * Real.sqrt 2 ∨ m = 1 - 2 * Real.sqrt 2 :=
by
-- Proof omitted
sorry

end problem_one_problem_two_l1423_142388


namespace solve_system_of_equations_l1423_142329

-- Definition of the system of equations as conditions
def eq1 (x y : ℤ) : Prop := 3 * x + y = 2
def eq2 (x y : ℤ) : Prop := 2 * x - 3 * y = 27

-- The theorem claiming the solution set is { (3, -7) }
theorem solve_system_of_equations :
  ∀ x y : ℤ, eq1 x y ∧ eq2 x y ↔ (x, y) = (3, -7) :=
by
  sorry

end solve_system_of_equations_l1423_142329


namespace solve_s_l1423_142307

theorem solve_s (s : ℝ) (h_pos : 0 < s) (h_eq : s^3 = 256) : s = 4 :=
sorry

end solve_s_l1423_142307


namespace final_price_is_correct_l1423_142325

-- Define the conditions as constants
def price_smartphone : ℝ := 300
def price_pc : ℝ := price_smartphone + 500
def price_tablet : ℝ := price_smartphone + price_pc
def total_price : ℝ := price_smartphone + price_pc + price_tablet
def discount : ℝ := 0.10 * total_price
def price_after_discount : ℝ := total_price - discount
def sales_tax : ℝ := 0.05 * price_after_discount
def final_price : ℝ := price_after_discount + sales_tax

-- Theorem statement asserting the final price value
theorem final_price_is_correct : final_price = 2079 := by sorry

end final_price_is_correct_l1423_142325


namespace number_of_terriers_groomed_l1423_142380

-- Define the initial constants and the conditions from the problem statement
def time_to_groom_poodle := 30
def time_to_groom_terrier := 15
def number_of_poodles := 3
def total_grooming_time := 210

-- Define the problem to prove that the number of terriers groomed is 8
theorem number_of_terriers_groomed (groom_time_poodle groom_time_terrier num_poodles total_time : ℕ) : 
  groom_time_poodle = time_to_groom_poodle → 
  groom_time_terrier = time_to_groom_terrier →
  num_poodles = number_of_poodles →
  total_time = total_grooming_time →
  ∃ n : ℕ, n * groom_time_terrier + num_poodles * groom_time_poodle = total_time ∧ n = 8 := 
by
  intros h1 h2 h3 h4
  sorry

end number_of_terriers_groomed_l1423_142380


namespace underachievers_l1423_142377

-- Define the variables for the number of students in each group
variables (a b c : ℕ)

-- Given conditions as hypotheses
axiom total_students : a + b + c = 30
axiom top_achievers : a = 19
axiom average_students : c = 12

-- Prove the number of underachievers
theorem underachievers : b = 9 :=
by sorry

end underachievers_l1423_142377


namespace parallel_lines_l1423_142366

/-- Given two lines l1 and l2 are parallel, prove a = -1 or a = 2. -/
def lines_parallel (a : ℝ) : Prop :=
  (a - 1) * a = 2

theorem parallel_lines (a : ℝ) (h : lines_parallel a) : a = -1 ∨ a = 2 :=
by
  sorry

end parallel_lines_l1423_142366


namespace quadratic_roots_identity_l1423_142356

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l1423_142356


namespace range_of_a_l1423_142367

variable (a : ℝ)

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 * a - 1) ^ x < (2 * a - 1) ^ y
def q (a : ℝ) : Prop := ∀ x : ℝ, 2 * a * x^2 - 2 * a * x + 1 > 0

theorem range_of_a (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : (0 ≤ a ∧ a ≤ 1) ∨ (2 ≤ a) :=
by
  sorry

end range_of_a_l1423_142367


namespace average_salary_of_employees_l1423_142355

theorem average_salary_of_employees
  (A : ℝ)  -- Define the average monthly salary A of 18 employees
  (h1 : 18*A + 5800 = 19*(A + 200))  -- Condition given in the problem
  : A = 2000 :=  -- The conclusion we need to prove
by
  sorry

end average_salary_of_employees_l1423_142355


namespace sum_a4_a5_a6_l1423_142387

section ArithmeticSequence

variable {a : ℕ → ℝ}

-- Condition 1: The sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

-- Condition 2: Given information
axiom a2_a8_eq_6 : a 2 + a 8 = 6

-- Question: Prove that a 4 + a 5 + a 6 = 9
theorem sum_a4_a5_a6 : is_arithmetic_sequence a → a 4 + a 5 + a 6 = 9 :=
by
  intro h_arith
  sorry

end ArithmeticSequence

end sum_a4_a5_a6_l1423_142387


namespace nellie_final_legos_l1423_142392

-- Define the conditions
def original_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_away_legos : ℕ := 24

-- The total legos Nellie has now
def remaining_legos (original lost given_away : ℕ) : ℕ := original - lost - given_away

-- Prove that given the conditions, Nellie has 299 legos left
theorem nellie_final_legos : remaining_legos original_legos lost_legos given_away_legos = 299 := by
  sorry

end nellie_final_legos_l1423_142392


namespace smallest_pos_int_ends_in_6_divisible_by_11_l1423_142399

theorem smallest_pos_int_ends_in_6_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 6 ∧ 11 ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m % 10 = 6 ∧ 11 ∣ m → n ≤ m := by
  sorry

end smallest_pos_int_ends_in_6_divisible_by_11_l1423_142399


namespace common_roots_l1423_142386

noncomputable def p (x a : ℝ) := x^3 + a * x^2 + 14 * x + 7
noncomputable def q (x b : ℝ) := x^3 + b * x^2 + 21 * x + 15

theorem common_roots (a b : ℝ) (r s : ℝ) (hr : r ≠ s)
  (hp : p r a = 0) (hp' : p s a = 0)
  (hq : q r b = 0) (hq' : q s b = 0) :
  a = 5 ∧ b = 4 :=
by sorry

end common_roots_l1423_142386


namespace power_congruence_l1423_142326

theorem power_congruence (a b n : ℕ) (h : a ≡ b [MOD n]) : a^n ≡ b^n [MOD n^2] :=
sorry

end power_congruence_l1423_142326


namespace interest_calculation_l1423_142345

variables (P R SI : ℝ) (T : ℕ)

-- Given conditions
def principal := (P = 8)
def rate := (R = 0.05)
def simple_interest := (SI = 4.8)

-- Goal
def time_calculated := (T = 12)

-- Lean statement combining the conditions
theorem interest_calculation : principal P → rate R → simple_interest SI → T = 12 :=
by
  intros hP hR hSI
  sorry

end interest_calculation_l1423_142345


namespace problem_l1423_142349

theorem problem (x y z : ℕ) (h1 : xy + z = 56) (h2 : yz + x = 56) (h3 : zx + y = 56) : x + y + z = 21 :=
sorry

end problem_l1423_142349


namespace mila_social_media_time_week_l1423_142332

theorem mila_social_media_time_week
  (hours_per_day_on_phone : ℕ)
  (half_on_social_media : ℕ)
  (days_in_week : ℕ)
  (h1 : hours_per_day_on_phone = 6)
  (h2 : half_on_social_media = hours_per_day_on_phone / 2)
  (h3 : days_in_week = 7) : 
  half_on_social_media * days_in_week = 21 := 
by
  rw [h2, h3]
  norm_num
  exact h1.symm ▸ rfl

end mila_social_media_time_week_l1423_142332


namespace ads_on_first_web_page_l1423_142348

theorem ads_on_first_web_page 
  (A : ℕ)
  (second_page_ads : ℕ := 2 * A)
  (third_page_ads : ℕ := 2 * A + 24)
  (fourth_page_ads : ℕ := 3 * A / 2)
  (total_ads : ℕ := 68 * 3 / 2)
  (sum_of_ads : A + 2 * A + (2 * A + 24) + 3 * A / 2 = total_ads) :
  A = 12 := 
by
  sorry

end ads_on_first_web_page_l1423_142348


namespace kanul_machinery_expense_l1423_142382

theorem kanul_machinery_expense :
  let Total := 93750
  let RawMaterials := 35000
  let Cash := 0.20 * Total
  let Machinery := Total - (RawMaterials + Cash)
  Machinery = 40000 := by
sorry

end kanul_machinery_expense_l1423_142382
