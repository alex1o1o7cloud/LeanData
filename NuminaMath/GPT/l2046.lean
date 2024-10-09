import Mathlib

namespace find_constant_l2046_204697

variable (constant : ℝ)

theorem find_constant (t : ℝ) (x : ℝ) (y : ℝ)
  (h1 : x = 1 - 2 * t)
  (h2 : y = constant * t - 2)
  (h3 : x = y) : constant = 2 :=
by
  sorry

end find_constant_l2046_204697


namespace profit_when_x_is_6_max_profit_l2046_204618

noncomputable def design_fee : ℝ := 20000 / 10000
noncomputable def production_cost_per_100 : ℝ := 10000 / 10000

noncomputable def P (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x - 0.8
  else 14.7 - 9 / (x - 3)

noncomputable def cost_of_x_sets (x : ℝ) : ℝ :=
  design_fee + x * production_cost_per_100

noncomputable def profit (x : ℝ) : ℝ :=
  P x - cost_of_x_sets x

theorem profit_when_x_is_6 :
  profit 6 = 3.7 := sorry

theorem max_profit :
  ∀ x : ℝ, profit x ≤ 3.7 := sorry

end profit_when_x_is_6_max_profit_l2046_204618


namespace terminating_decimals_count_l2046_204646

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l2046_204646


namespace reporter_earnings_per_hour_l2046_204607

/-- A reporter's expected earnings per hour if she writes the entire time. -/
theorem reporter_earnings_per_hour :
  ∀ (word_earnings: ℝ) (article_earnings: ℝ) (stories: ℕ) (hours: ℝ) (words_per_minute: ℕ),
  word_earnings = 0.1 →
  article_earnings = 60 →
  stories = 3 →
  hours = 4 →
  words_per_minute = 10 →
  (stories * article_earnings + (hours * 60 * words_per_minute) * word_earnings) / hours = 105 :=
by
  intros word_earnings article_earnings stories hours words_per_minute
  intros h_word_earnings h_article_earnings h_stories h_hours h_words_per_minute
  sorry

end reporter_earnings_per_hour_l2046_204607


namespace construct_triangle_from_medians_l2046_204660

theorem construct_triangle_from_medians
    (s_a s_b s_c : ℝ)
    (h1 : s_a + s_b > s_c)
    (h2 : s_a + s_c > s_b)
    (h3 : s_b + s_c > s_a) :
    ∃ (a b c : ℝ), 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    (∃ (median_a median_b median_c : ℝ), 
        median_a = s_a ∧ 
        median_b = s_b ∧ 
        median_c = s_c) :=
sorry

end construct_triangle_from_medians_l2046_204660


namespace least_integer_divisors_l2046_204636

theorem least_integer_divisors (n m k : ℕ)
  (h_divisors : 3003 = 3 * 7 * 11 * 13)
  (h_form : n = m * 30 ^ k)
  (h_no_div_30 : ¬(30 ∣ m))
  (h_divisor_count : ∀ (p : ℕ) (h : n = p), (p + 1) * (p + 1) * (p + 1) * (p + 1) = 3003)
  : m + k = 104978 :=
sorry

end least_integer_divisors_l2046_204636


namespace students_doing_hula_hoops_l2046_204683

def number_of_students_jumping_rope : ℕ := 7
def number_of_students_doing_hula_hoops : ℕ := 5 * number_of_students_jumping_rope

theorem students_doing_hula_hoops : number_of_students_doing_hula_hoops = 35 :=
by
  sorry

end students_doing_hula_hoops_l2046_204683


namespace largest_side_of_rectangle_l2046_204633

theorem largest_side_of_rectangle (l w : ℕ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 1920) : max l w = 101 := 
by
  sorry

end largest_side_of_rectangle_l2046_204633


namespace value_of_expression_l2046_204630

theorem value_of_expression (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 10) : (12 * y - 4)^2 = 80 := 
by 
  sorry

end value_of_expression_l2046_204630


namespace pythagorean_diagonal_l2046_204654

variable (m : ℕ) (h_m : m ≥ 3)

theorem pythagorean_diagonal (h : (2 * m)^2 + a^2 = (a + 2)^2) :
  (a + 2) = m^2 + 1 :=
by
  sorry

end pythagorean_diagonal_l2046_204654


namespace smaller_number_4582_l2046_204621

theorem smaller_number_4582 (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha_b : a < 100) (hb_b : b < 100) (h : a * b = 4582) :
  min a b = 21 :=
sorry

end smaller_number_4582_l2046_204621


namespace solve_equation_l2046_204604

theorem solve_equation :
  ∀ (x : ℝ), (1 / 7 + 7 / x = 15 / x + 1 / 15) → x = 8 :=
by
  intros x h
  sorry

end solve_equation_l2046_204604


namespace find_wind_speed_l2046_204647

-- Definitions from conditions
def speed_with_wind (j w : ℝ) := (j + w) * 6 = 3000
def speed_against_wind (j w : ℝ) := (j - w) * 9 = 3000

-- Theorem to prove the wind speed is 83.335 mph
theorem find_wind_speed (j w : ℝ) (h1 : speed_with_wind j w) (h2 : speed_against_wind j w) : w = 83.335 :=
by 
  -- Here we would prove the theorem using the given conditions
  sorry

end find_wind_speed_l2046_204647


namespace solution_exists_for_100_100_l2046_204676

def exists_positive_integers_sum_of_cubes (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a^3 + b^3 + c^3 + d^3 = x

theorem solution_exists_for_100_100 : exists_positive_integers_sum_of_cubes (100 ^ 100) :=
by
  sorry

end solution_exists_for_100_100_l2046_204676


namespace intersection_A_B_l2046_204674

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l2046_204674


namespace max_value_g_l2046_204677

def g : ℕ → ℕ
| n => if n < 7 then n + 8 else g (n - 3)

theorem max_value_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 14 := by
  sorry

end max_value_g_l2046_204677


namespace determine_abc_l2046_204653

-- Definitions
def parabola_equation (a b c : ℝ) (y : ℝ) : ℝ := a * y^2 + b * y + c

def vertex_condition (a b c : ℝ) : Prop :=
  ∀ y, parabola_equation a b c y = a * (y + 6)^2 + 3

def point_condition (a b c : ℝ) : Prop :=
  parabola_equation a b c (-6) = 3 ∧ parabola_equation a b c (-4) = 2

-- Proposition to prove
theorem determine_abc : 
  ∃ a b c : ℝ, vertex_condition a b c ∧ point_condition a b c
  ∧ (a + b + c = -25/4) :=
sorry

end determine_abc_l2046_204653


namespace probability_heads_l2046_204678

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let y := (n.choose (n / 2)) / total_outcomes
  (1 - y) / 2

theorem probability_heads (h₁ : ∀ (n : ℕ), n = 10 → probability_more_heads_than_tails n = 193 / 512) : 
  probability_more_heads_than_tails 10 = 193 / 512 :=
by
  apply h₁
  exact rfl

end probability_heads_l2046_204678


namespace find_ab_l2046_204616

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 :=
by
  sorry

end find_ab_l2046_204616


namespace at_least_eight_composites_l2046_204632

theorem at_least_eight_composites (n : ℕ) (h : n > 1000) :
  ∃ (comps : Finset ℕ), 
    comps.card ≥ 8 ∧ 
    (∀ x ∈ comps, ¬Prime x) ∧ 
    (∀ k, k < 12 → n + k ∈ comps ∨ Prime (n + k)) :=
by
  sorry

end at_least_eight_composites_l2046_204632


namespace bike_riders_count_l2046_204691

variables (B H : ℕ)

theorem bike_riders_count
  (h₁ : H = B + 178)
  (h₂ : H + B = 676) :
  B = 249 :=
sorry

end bike_riders_count_l2046_204691


namespace smallest_solution_exists_l2046_204620

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l2046_204620


namespace pictures_total_l2046_204685

theorem pictures_total (peter_pics : ℕ) (quincy_extra_pics : ℕ) (randy_pics : ℕ) (quincy_pics : ℕ) (total_pics : ℕ) 
  (h1 : peter_pics = 8)
  (h2 : quincy_extra_pics = 20)
  (h3 : randy_pics = 5)
  (h4 : quincy_pics = peter_pics + quincy_extra_pics)
  (h5 : total_pics = randy_pics + peter_pics + quincy_pics) :
  total_pics = 41 :=
by sorry

end pictures_total_l2046_204685


namespace min_x_plus_2y_l2046_204681

theorem min_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (2 / x) + (1 / y) = 1) : x + 2 * y ≥ 8 :=
sorry

end min_x_plus_2y_l2046_204681


namespace ratio_of_areas_l2046_204675

variable (s : ℝ)
def side_length_square := s
def side_length_longer_rect := 1.2 * s
def side_length_shorter_rect := 0.7 * s
def area_square := s^2
def area_rect := (1.2 * s) * (0.7 * s)

theorem ratio_of_areas (h1 : s > 0) :
  area_rect s / area_square s = 21 / 25 :=
by 
  sorry

end ratio_of_areas_l2046_204675


namespace find_number_l2046_204664

theorem find_number (x : ℝ) (h : 0.30 * x - 70 = 20) : x = 300 :=
sorry

end find_number_l2046_204664


namespace fraction_calculation_l2046_204642

theorem fraction_calculation : (3/10 : ℚ) + (5/100 : ℚ) - (2/1000 : ℚ) = 348/1000 := 
by 
  sorry

end fraction_calculation_l2046_204642


namespace inequality_am_gm_l2046_204658

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c :=
by sorry

end inequality_am_gm_l2046_204658


namespace intersection_of_M_and_N_l2046_204682

def set_M : Set ℝ := {x | -1 < x}
def set_N : Set ℝ := {x | x * (x + 2) ≤ 0}

theorem intersection_of_M_and_N : (set_M ∩ set_N) = {x | -1 < x ∧ x ≤ 0} :=
by
  sorry

end intersection_of_M_and_N_l2046_204682


namespace polygon_number_of_sides_l2046_204638

-- Definitions based on conditions
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def exterior_angle (angle : ℕ) : ℕ := 30

-- The theorem statement
theorem polygon_number_of_sides (n : ℕ) (angle : ℕ) 
  (h1 : sum_of_exterior_angles n = 360)
  (h2 : exterior_angle angle = 30) : 
  n = 12 := 
by
  sorry

end polygon_number_of_sides_l2046_204638


namespace ratio_of_areas_l2046_204651

noncomputable def large_square_side : ℝ := 4
noncomputable def large_square_area : ℝ := large_square_side ^ 2
noncomputable def inscribed_square_side : ℝ := 1  -- As it fits in the definition from the problem description
noncomputable def inscribed_square_area : ℝ := inscribed_square_side ^ 2

theorem ratio_of_areas :
  (inscribed_square_area / large_square_area) = 1 / 16 :=
by
  sorry

end ratio_of_areas_l2046_204651


namespace geom_seq_problem_l2046_204641

variable {a : ℕ → ℝ}  -- positive geometric sequence

-- Conditions
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a n = a 0 * r^n

theorem geom_seq_problem
  (h_geom : geom_seq a)
  (cond : a 0 * a 4 + 2 * a 2 * a 4 + a 2 * a 6 = 25) :
  a 2 + a 4 = 5 :=
sorry

end geom_seq_problem_l2046_204641


namespace order_of_abc_l2046_204614

noncomputable def a := Real.log 6 / Real.log 0.7
noncomputable def b := Real.rpow 6 0.7
noncomputable def c := Real.rpow 0.7 0.6

theorem order_of_abc : b > c ∧ c > a := by
  sorry

end order_of_abc_l2046_204614


namespace relationship_between_p_and_q_l2046_204669

variable (x y : ℝ)

def p := x * y ≥ 0
def q := |x + y| = |x| + |y|

theorem relationship_between_p_and_q : (p x y ↔ q x y) :=
sorry

end relationship_between_p_and_q_l2046_204669


namespace gcd_divisor_l2046_204634

theorem gcd_divisor (p q r s : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (hpq : Nat.gcd p q = 40) (hqr : Nat.gcd q r = 50) (hrs : Nat.gcd r s = 60) (hsp : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) 
  : 13 ∣ p :=
sorry

end gcd_divisor_l2046_204634


namespace valid_number_of_apples_l2046_204661

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l2046_204661


namespace number_of_wheels_on_each_bicycle_l2046_204612

theorem number_of_wheels_on_each_bicycle 
  (num_bicycles : ℕ)
  (num_tricycles : ℕ)
  (wheels_per_tricycle : ℕ)
  (total_wheels : ℕ)
  (h_bicycles : num_bicycles = 24)
  (h_tricycles : num_tricycles = 14)
  (h_wheels_tricycle : wheels_per_tricycle = 3)
  (h_total_wheels : total_wheels = 90) :
  2 * num_bicycles + 3 * num_tricycles = 90 → 
  num_bicycles = 24 → 
  num_tricycles = 14 → 
  wheels_per_tricycle = 3 → 
  total_wheels = 90 → 
  ∃ b : ℕ, b = 2 :=
by
  sorry

end number_of_wheels_on_each_bicycle_l2046_204612


namespace point_inside_circle_l2046_204644

theorem point_inside_circle (a : ℝ) (h : 5 * a^2 - 4 * a - 1 < 0) : -1/5 < a ∧ a < 1 :=
    sorry

end point_inside_circle_l2046_204644


namespace inheritance_shares_l2046_204649

theorem inheritance_shares (A B : ℝ) (h1: A + B = 100) (h2: (1/4) * B - (1/3) * A = 11) : 
  A = 24 ∧ B = 76 := 
by 
  sorry

end inheritance_shares_l2046_204649


namespace printers_finish_tasks_l2046_204693

theorem printers_finish_tasks :
  ∀ (start_time_1 finish_half_time_1 start_time_2 : ℕ) (half_task_duration full_task_duration second_task_duration : ℕ),
    start_time_1 = 9 * 60 ∧
    finish_half_time_1 = 12 * 60 + 30 ∧
    half_task_duration = finish_half_time_1 - start_time_1 ∧
    full_task_duration = 2 * half_task_duration ∧
    start_time_2 = 13 * 60 ∧
    second_task_duration = 2 * 60 ∧
    start_time_1 + full_task_duration = 4 * 60 ∧
    start_time_2 + second_task_duration = 15 * 60 →
  max (start_time_1 + full_task_duration) (start_time_2 + second_task_duration) = 16 * 60 := 
by
  intros start_time_1 finish_half_time_1 start_time_2 half_task_duration full_task_duration second_task_duration
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩
  sorry

end printers_finish_tasks_l2046_204693


namespace mean_of_second_set_l2046_204695

theorem mean_of_second_set (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 90) : 
  (128 + 255 + 511 + 1023 + x) / 5 = 423 :=
by
  sorry

end mean_of_second_set_l2046_204695


namespace longer_side_of_rectangle_l2046_204610

theorem longer_side_of_rectangle 
  (radius : ℝ) (A_rectangle : ℝ) (shorter_side : ℝ) 
  (h1 : radius = 6)
  (h2 : A_rectangle = 3 * (π * radius^2))
  (h3 : shorter_side = 2 * 2 * radius) :
  (A_rectangle / shorter_side) = 4.5 * π :=
by
  sorry

end longer_side_of_rectangle_l2046_204610


namespace value_of_5y_l2046_204688

-- Define positive integers
variables {x y z : ℕ}

-- Define the conditions
def conditions (x y z : ℕ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (5 * y = 6 * z) ∧ (x + y + z = 26)

-- The theorem statement
theorem value_of_5y (x y z : ℕ) (h : conditions x y z) : 5 * y = 30 :=
by
  -- proof skipped (proof goes here)
  sorry

end value_of_5y_l2046_204688


namespace paint_rate_5_l2046_204686
noncomputable def rate_per_sq_meter (L : ℝ) (total_cost : ℝ) (B : ℝ) : ℝ :=
  let Area := L * B
  total_cost / Area

theorem paint_rate_5 : 
  ∀ (L B total_cost rate : ℝ),
    L = 19.595917942265423 →
    total_cost = 640 →
    L = 3 * B →
    rate = rate_per_sq_meter L total_cost B →
    rate = 5 :=
by
  intros L B total_cost rate hL hC hR hRate
  -- Proof goes here
  sorry

end paint_rate_5_l2046_204686


namespace two_point_questions_count_l2046_204671

theorem two_point_questions_count (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 :=
sorry

end two_point_questions_count_l2046_204671


namespace problem_statement_l2046_204670

theorem problem_statement :
  (2 * 3 * 4) * (1/2 + 1/3 + 1/4) = 26 := by
  sorry

end problem_statement_l2046_204670


namespace eval_expr1_eval_expr2_l2046_204687

theorem eval_expr1 (a b : ℝ) (h₁ : a = 7) (h₂ : b = 3) : 
  (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := 
by
  sorry

theorem eval_expr2 (a b : ℝ) (h₁ : a = 7) (h₂ : b = 3) : 
  (a^2 + b^2) / (a + b) = 5.8 :=
by
  sorry

end eval_expr1_eval_expr2_l2046_204687


namespace number_of_triangles_in_lattice_l2046_204615

-- Define the triangular lattice structure
def triangular_lattice_rows : List ℕ := [1, 2, 3, 4]

-- Define the main theorem to state the number of triangles
theorem number_of_triangles_in_lattice :
  let number_of_triangles := 1 + 2 + 3 + 6 + 10
  number_of_triangles = 22 :=
by
  -- here goes the proof, which we skip with "sorry"
  sorry

end number_of_triangles_in_lattice_l2046_204615


namespace problem1_problem2_l2046_204603

theorem problem1 : Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 :=
by
  sorry

theorem problem2 : Real.sqrt 6 / Real.sqrt 18 * Real.sqrt 27 = 3 :=
by
  sorry

end problem1_problem2_l2046_204603


namespace symmetric_origin_l2046_204629

def symmetric_point (p : (Int × Int)) : (Int × Int) :=
  (-p.1, -p.2)

theorem symmetric_origin : symmetric_point (-2, 5) = (2, -5) :=
by
  -- proof goes here
  -- we use sorry to indicate the place where the solution would go
  sorry

end symmetric_origin_l2046_204629


namespace smallest_possible_N_l2046_204652

theorem smallest_possible_N {p q r s t : ℕ} (hp: 0 < p) (hq: 0 < q) (hr: 0 < r) (hs: 0 < s) (ht: 0 < t) 
  (sum_eq: p + q + r + s + t = 3015) :
  ∃ N, N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ N = 1508 := 
sorry

end smallest_possible_N_l2046_204652


namespace how_many_pairs_of_shoes_l2046_204611

theorem how_many_pairs_of_shoes (l k : ℕ) (h_l : l = 52) (h_k : k = 2) : l / k = 26 := by
  sorry

end how_many_pairs_of_shoes_l2046_204611


namespace students_left_zoo_l2046_204606

theorem students_left_zoo
  (students_first_class students_second_class : ℕ)
  (chaperones teachers : ℕ)
  (initial_individuals remaining_individuals : ℕ)
  (chaperones_left remaining_individuals_after_chaperones_left : ℕ)
  (remaining_students initial_students : ℕ)
  (H1 : students_first_class = 10)
  (H2 : students_second_class = 10)
  (H3 : chaperones = 5)
  (H4 : teachers = 2)
  (H5 : initial_individuals = students_first_class + students_second_class + chaperones + teachers) 
  (H6 : initial_individuals = 27)
  (H7 : remaining_individuals = 15)
  (H8 : chaperones_left = 2)
  (H9 : remaining_individuals_after_chaperones_left = remaining_individuals - chaperones_left)
  (H10 : remaining_individuals_after_chaperones_left = 13)
  (H11 : remaining_students = remaining_individuals_after_chaperones_left - teachers)
  (H12 : remaining_students = 11)
  (H13 : initial_students = students_first_class + students_second_class)
  (H14 : initial_students = 20) :
  20 - 11 = 9 :=
by sorry

end students_left_zoo_l2046_204606


namespace equation_equivalence_l2046_204672

theorem equation_equivalence (p q : ℝ) (hp₀ : p ≠ 0) (hp₅ : p ≠ 5) (hq₀ : q ≠ 0) (hq₇ : q ≠ 7) :
  (3 / p + 5 / q = 1 / 3) → p = 9 * q / (q - 15) :=
by
  sorry

end equation_equivalence_l2046_204672


namespace range_of_a_l2046_204643

theorem range_of_a 
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x < 0, f x = a^x)
  (h2 : ∀ x ≥ 0, f x = (a - 3) * x + 4 * a)
  (h3 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0) :
  0 < a ∧ a ≤ 1 / 4 :=
sorry

end range_of_a_l2046_204643


namespace part1_part2_l2046_204662

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (1 - a) * x + (1 - a)

theorem part1 (x : ℝ) : f x 4 ≥ 7 ↔ x ≥ 5 ∨ x ≤ -2 :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, -1 < x → f x a ≥ 0) ↔ a ≤ 1 :=
sorry

end part1_part2_l2046_204662


namespace minimum_value_expression_l2046_204699

variable (a b c k : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a = k ∧ b = k ∧ c = k)

theorem minimum_value_expression : 
  (a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) = 9 / 2 :=
by
  sorry

end minimum_value_expression_l2046_204699


namespace parity_equiv_l2046_204609

open Nat

theorem parity_equiv (p q : ℕ) : (Even (p^3 - q^3) ↔ Even (p + q)) :=
by
  sorry

end parity_equiv_l2046_204609


namespace proof_problem_l2046_204648

def consistent_system (x y : ℕ) : Prop :=
  x + y = 99 ∧ 3 * x + 1 / 3 * y = 97

theorem proof_problem : ∃ (x y : ℕ), consistent_system x y := sorry

end proof_problem_l2046_204648


namespace max_zeros_in_product_l2046_204696

theorem max_zeros_in_product (a b c : ℕ) (h_sum : a + b + c = 1003) : ∃ N, N = 7 ∧ ∀ p : ℕ, (a * b * c = p) → (∃ k, p = 10^k ∧ k ≤ N) ∧ (∀ k, p = 10^k → k ≤ 7) :=
by
  sorry

end max_zeros_in_product_l2046_204696


namespace find_some_number_l2046_204679

theorem find_some_number (a : ℕ) (h1 : a = 105) (h2 : a^3 = some_number * 35 * 45 * 35) : some_number = 1 := by
  sorry

end find_some_number_l2046_204679


namespace subtraction_example_l2046_204656

theorem subtraction_example : 2 - 3 = -1 := 
by {
  -- We need to prove that 2 - 3 = -1
  -- The proof is to be filled here
  sorry
}

end subtraction_example_l2046_204656


namespace sum_of_numbers_l2046_204698

theorem sum_of_numbers (x : ℝ) (h : x^2 + (2 * x)^2 + (4 * x)^2 = 4725) : 
  x + 2 * x + 4 * x = 105 := 
sorry

end sum_of_numbers_l2046_204698


namespace terminal_sides_positions_l2046_204667

def in_third_quadrant (θ : ℝ) (k : ℤ) : Prop :=
  (180 + k * 360 : ℝ) < θ ∧ θ < (270 + k * 360 : ℝ)

theorem terminal_sides_positions (θ : ℝ) (k : ℤ) :
  in_third_quadrant θ k →
  ((2 * θ > 360 + 2 * k * 360 ∧ 2 * θ < 540 + 2 * k * 360) ∨
   (90 + k * 180 < θ / 2 ∧ θ / 2 < 135 + k * 180) ∨
   (2 * θ = 360 + 2 * k * 360) ∨ (2 * θ = 540 + 2 * k * 360) ∨ 
   (θ / 2 = 90 + k * 180) ∨ (θ / 2 = 135 + k * 180)) :=
by
  intro h
  sorry

end terminal_sides_positions_l2046_204667


namespace largest_divisor_even_squares_l2046_204625

theorem largest_divisor_even_squares (m n : ℕ) (hm : Even m) (hn : Even n) (h : n < m) :
  ∃ k, k = 4 ∧ ∀ a b : ℕ, Even a → Even b → b < a → k ∣ (a^2 - b^2) :=
by
  sorry

end largest_divisor_even_squares_l2046_204625


namespace total_amount_l2046_204657

def mark_dollars : ℚ := 5 / 8
def carolyn_dollars : ℚ := 2 / 5
def total_dollars : ℚ := mark_dollars + carolyn_dollars

theorem total_amount : total_dollars = 1.025 := by
  sorry

end total_amount_l2046_204657


namespace B_elements_l2046_204650

def B : Set ℤ := {x | -3 < 2 * x - 1 ∧ 2 * x - 1 < 3}

theorem B_elements : B = {-1, 0, 1} :=
by
  sorry

end B_elements_l2046_204650


namespace normal_price_of_article_l2046_204608

theorem normal_price_of_article (P : ℝ) (h : 0.9 * 0.8 * P = 144) : P = 200 :=
sorry

end normal_price_of_article_l2046_204608


namespace difference_highest_lowest_score_l2046_204605

-- Declare scores of each player
def Zach_score : ℕ := 42
def Ben_score : ℕ := 21
def Emma_score : ℕ := 35
def Leo_score : ℕ := 28

-- Calculate the highest and lowest scores
def highest_score : ℕ := max (max Zach_score Ben_score) (max Emma_score Leo_score)
def lowest_score : ℕ := min (min Zach_score Ben_score) (min Emma_score Leo_score)

-- Calculate the difference
def score_difference : ℕ := highest_score - lowest_score

theorem difference_highest_lowest_score : score_difference = 21 := 
by
  sorry

end difference_highest_lowest_score_l2046_204605


namespace find_radius_l2046_204680

-- Definition of the conditions
def area_of_sector : ℝ := 10 -- The area of the sector in square centimeters
def arc_length : ℝ := 4     -- The arc length of the sector in centimeters

-- The radius of the circle we want to prove
def radius (r : ℝ) : Prop :=
  (r * 4) / 2 = 10

-- The theorem to be proved
theorem find_radius : ∃ r : ℝ, radius r :=
by
  use 5
  unfold radius
  norm_num

end find_radius_l2046_204680


namespace box_volume_l2046_204626

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

theorem box_volume (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : volume_of_box l w h = 72 :=
by
  sorry

end box_volume_l2046_204626


namespace sequence_sum_l2046_204640

theorem sequence_sum (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) = (1/3) * a n) (h_a4a5 : a 4 + a 5 = 4) :
    a 2 + a 3 = 36 :=
    sorry

end sequence_sum_l2046_204640


namespace min_value_of_f_l2046_204637

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 4*x + 5) * (x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = -9 :=
  sorry

end min_value_of_f_l2046_204637


namespace stratified_sampling_young_employees_l2046_204655

-- Given conditions
def total_young : Nat := 350
def total_middle_aged : Nat := 500
def total_elderly : Nat := 150
def total_employees : Nat := total_young + total_middle_aged + total_elderly
def representatives_to_select : Nat := 20
def sampling_ratio : Rat := representatives_to_select / (total_employees : Rat)

-- Proof goal
theorem stratified_sampling_young_employees :
  (total_young : Rat) * sampling_ratio = 7 := 
by
  sorry

end stratified_sampling_young_employees_l2046_204655


namespace sum_eight_smallest_multiples_of_12_l2046_204623

theorem sum_eight_smallest_multiples_of_12 :
  let series := (List.range 8).map (λ k => 12 * (k + 1))
  series.sum = 432 :=
by
  sorry

end sum_eight_smallest_multiples_of_12_l2046_204623


namespace find_fraction_l2046_204602

theorem find_fraction
  (N : ℝ)
  (hN : N = 30)
  (h : 0.5 * N = (x / y) * N + 10):
  x / y = 1 / 6 :=
by
  sorry

end find_fraction_l2046_204602


namespace sequence_term_general_formula_l2046_204613

theorem sequence_term_general_formula (S : ℕ → ℚ) (a : ℕ → ℚ) :
  (∀ n, S n = n^2 + (1/2)*n + 5) →
  (∀ n, (n ≥ 2) → a n = S n - S (n - 1)) →
  a 1 = 13/2 →
  (∀ n, a n = if n = 1 then 13/2 else 2*n - 1/2) :=
by
  intros hS ha h1
  sorry

end sequence_term_general_formula_l2046_204613


namespace sum_of_solutions_l2046_204624

theorem sum_of_solutions :
  (∃ S : Finset ℝ, (∀ x ∈ S, x^2 - 8*x + 21 = abs (x - 5) + 4) ∧ S.sum id = 18) :=
by
  sorry

end sum_of_solutions_l2046_204624


namespace ratio_of_weights_l2046_204663

noncomputable def tyler_weight (sam_weight : ℝ) : ℝ := sam_weight + 25
noncomputable def ratio_of_peter_to_tyler (peter_weight tyler_weight : ℝ) : ℝ := peter_weight / tyler_weight

theorem ratio_of_weights (sam_weight : ℝ) (peter_weight : ℝ) (h_sam : sam_weight = 105) (h_peter : peter_weight = 65) :
  ratio_of_peter_to_tyler peter_weight (tyler_weight sam_weight) = 0.5 := by
  -- We use the conditions to derive the information
  sorry

end ratio_of_weights_l2046_204663


namespace angle_D_in_pentagon_l2046_204694

theorem angle_D_in_pentagon (A B C D E : ℝ) 
  (h1 : A = B) (h2 : B = C) (h3 : D = E) (h4 : A + 40 = D) 
  (h5 : A + B + C + D + E = 540) : D = 132 :=
by
  -- Add proof here if needed
  sorry

end angle_D_in_pentagon_l2046_204694


namespace monotonicity_of_f_range_of_a_l2046_204690

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) - a * x

theorem monotonicity_of_f (a : ℝ) (ha : a ≠ 0) :
  (∀ x < 0, f a x ≥ f a (x + 1)) ∧ (∀ x > 0, f a x ≤ f a (x + 1)) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ Real.sin x - Real.cos x + 2 - a * x) ↔ a ∈ Set.Ici 1 :=
sorry

end monotonicity_of_f_range_of_a_l2046_204690


namespace kim_total_ounces_l2046_204635

def quarts_to_ounces (q : ℚ) : ℚ := q * 32

def bottle_quarts : ℚ := 1.5
def can_ounces : ℚ := 12
def bottle_ounces : ℚ := quarts_to_ounces bottle_quarts

def total_ounces : ℚ := bottle_ounces + can_ounces

theorem kim_total_ounces : total_ounces = 60 :=
by
  -- Proof will go here
  sorry

end kim_total_ounces_l2046_204635


namespace line_intersects_circle_and_focus_condition_l2046_204692

variables {x y k : ℝ}

/-- The line l intersects the circle x^2 + y^2 + 2x - 4y + 1 = 0 at points A and B. If the midpoint of the chord AB is the focus of the parabola x^2 = 4y, then prove that the equation of the line l is x - y + 1 = 0. -/
theorem line_intersects_circle_and_focus_condition :
  (∃ l : ℝ → ℝ, (∀ x y : ℝ, l x = y) ∧
  (∀ A B : ℝ × ℝ, ∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (0, 1)) ∧
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧
  x^2 = 4*y ) → 
  (∀ x y : ℝ, x - y + 1 = 0) :=
sorry

end line_intersects_circle_and_focus_condition_l2046_204692


namespace time_spent_giving_bath_l2046_204673

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

end time_spent_giving_bath_l2046_204673


namespace f_odd_f_decreasing_f_max_min_l2046_204689

noncomputable def f : ℝ → ℝ := sorry

lemma f_add (x y : ℝ) : f (x + y) = f x + f y := sorry
lemma f_neg1 : f (-1) = 2 := sorry
lemma f_positive_less_than_zero {x : ℝ} (hx : x > 0) : f x < 0 := sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

theorem f_decreasing : ∀ x1 x2 : ℝ, x2 > x1 → f x2 < f x1 := sorry

theorem f_max_min : ∀ (f_max f_min : ℝ),
  f_max = f (-2) ∧ f_min = f 4 ∧
  f (-2) = 4 ∧ f 4 = -8 := sorry

end f_odd_f_decreasing_f_max_min_l2046_204689


namespace division_of_fractions_l2046_204601

theorem division_of_fractions : (1 / 10) / (1 / 5) = 1 / 2 :=
by
  sorry

end division_of_fractions_l2046_204601


namespace sample_size_l2046_204622

theorem sample_size (k n : ℕ) (r : 2 * k + 3 * k + 5 * k = 10 * k) (h : 3 * k = 12) : n = 40 :=
by {
    -- here, we will provide a proof to demonstrate that n = 40 given the conditions
    sorry
}

end sample_size_l2046_204622


namespace no_solution_abs_eq_2_l2046_204666

theorem no_solution_abs_eq_2 (x : ℝ) :
  |x - 5| = |x + 3| + 2 → false :=
by sorry

end no_solution_abs_eq_2_l2046_204666


namespace initially_calculated_average_height_l2046_204631

theorem initially_calculated_average_height 
    (students : ℕ) (incorrect_height : ℕ) (correct_height : ℕ) (actual_avg_height : ℝ) 
    (A : ℝ) 
    (h_students : students = 30) 
    (h_incorrect_height : incorrect_height = 151) 
    (h_correct_height : correct_height = 136) 
    (h_actual_avg_height : actual_avg_height = 174.5)
    (h_A_definition : (students : ℝ) * A + (incorrect_height - correct_height) = (students : ℝ) * actual_avg_height) : 
    A = 174 := 
by sorry

end initially_calculated_average_height_l2046_204631


namespace quadratic_eq_complete_square_l2046_204600

theorem quadratic_eq_complete_square (x p q : ℝ) (h : 9 * x^2 - 54 * x + 63 = 0) 
(h_trans : (x + p)^2 = q) : p + q = -1 := sorry

end quadratic_eq_complete_square_l2046_204600


namespace probability_of_distance_less_than_8000_l2046_204665

-- Define distances between cities

noncomputable def distances : List (String × String × ℕ) :=
  [("Bangkok", "Cape Town", 6300),
   ("Bangkok", "Honolulu", 7609),
   ("Bangkok", "London", 5944),
   ("Bangkok", "Tokyo", 2870),
   ("Cape Town", "Honolulu", 11535),
   ("Cape Town", "London", 5989),
   ("Cape Town", "Tokyo", 13400),
   ("Honolulu", "London", 7240),
   ("Honolulu", "Tokyo", 3805),
   ("London", "Tokyo", 5950)]

-- Define the total number of pairs and the pairs with distances less than 8000 miles

noncomputable def total_pairs : ℕ := 10
noncomputable def pairs_less_than_8000 : ℕ := 7

-- Define the statement of the probability being 7/10
theorem probability_of_distance_less_than_8000 :
  pairs_less_than_8000 / total_pairs = 7 / 10 :=
by
  sorry

end probability_of_distance_less_than_8000_l2046_204665


namespace no_such_n_exists_l2046_204645

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, n * sum_of_digits n = 100200300 :=
by
  sorry

end no_such_n_exists_l2046_204645


namespace Matilda_age_is_35_l2046_204659

-- Definitions based on conditions
def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

-- Theorem to prove the question's answer is correct
theorem Matilda_age_is_35 : Matilda_age = 35 :=
by
  -- Adding proof steps
  sorry

end Matilda_age_is_35_l2046_204659


namespace minimum_value_of_expression_l2046_204639

theorem minimum_value_of_expression (x y : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : 2 * x + 3 * y = 8) : 
  (∀ a b, a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 8 → (2 / a + 3 / b) ≥ 25 / 8) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 8 ∧ 2 / a + 3 / b = 25 / 8) :=
sorry

end minimum_value_of_expression_l2046_204639


namespace trajectory_of_point_A_l2046_204619

theorem trajectory_of_point_A (m : ℝ) (A B C : ℝ × ℝ) (hBC : B = (-1, 0) ∧ C = (1, 0)) (hBC_dist : dist B C = 2)
  (hRatio : dist A B / dist A C = m) :
  (m = 1 → ∀ x y : ℝ, A = (x, y) → x = 0) ∧
  (m = 0 → ∀ x y : ℝ, A = (x, y) → x^2 + y^2 - 2 * x + 1 = 0) ∧
  (m ≠ 0 ∧ m ≠ 1 → ∀ x y : ℝ, A = (x, y) → (x + (1 + m^2) / (1 - m^2))^2 + y^2 = (2 * m / (1 - m^2))^2) := 
sorry

end trajectory_of_point_A_l2046_204619


namespace option_a_option_b_l2046_204684

theorem option_a (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 :=
by
  -- Proof goes here
  sorry

theorem option_b (a b : ℝ) (ha : a > 0) (hb : b > 0) : a * b ≤ (a + b)^2 / 4 :=
by
  -- Proof goes here
  sorry

end option_a_option_b_l2046_204684


namespace cost_price_is_800_l2046_204617

theorem cost_price_is_800 (mp sp cp : ℝ) (h1 : mp = 1100) (h2 : sp = 0.8 * mp) (h3 : sp = 1.1 * cp) :
  cp = 800 :=
by
  sorry

end cost_price_is_800_l2046_204617


namespace inequality_proof_l2046_204628

theorem inequality_proof 
  (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (sum_eq_one : a + b + c + d = 1) :
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1/5 := 
by {
  sorry
}

end inequality_proof_l2046_204628


namespace distance_missouri_to_new_york_by_car_l2046_204668

-- Define the given conditions
def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def midway_factor : ℝ := 0.5

-- Define the problem to be proven
theorem distance_missouri_to_new_york_by_car :
  let total_distance : ℝ := distance_plane + (distance_plane * increase_percentage)
  let missouri_to_new_york_distance : ℝ := total_distance * midway_factor
  missouri_to_new_york_distance = 1400 :=
by
  sorry

end distance_missouri_to_new_york_by_car_l2046_204668


namespace password_decryption_probability_l2046_204627

theorem password_decryption_probability :
  let A := (1:ℚ)/5
  let B := (1:ℚ)/3
  let C := (1:ℚ)/4
  let P_decrypt := 1 - (1 - A) * (1 - B) * (1 - C)
  P_decrypt = 3/5 := 
  by
    -- Calculations and logic will be provided here
    sorry

end password_decryption_probability_l2046_204627
