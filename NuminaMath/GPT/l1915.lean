import Mathlib

namespace cowboy_cost_problem_l1915_191506

/-- The cost of a sandwich, a cup of coffee, and a donut adds up to 0.40 dollars given the expenditure details of two cowboys. -/
theorem cowboy_cost_problem (S C D : ℝ) (h1 : 4 * S + C + 10 * D = 1.69) (h2 : 3 * S + C + 7 * D = 1.26) :
  S + C + D = 0.40 :=
by
  sorry

end cowboy_cost_problem_l1915_191506


namespace parallelogram_area_l1915_191535

theorem parallelogram_area (base height : ℝ) (h_base : base = 36) (h_height : height = 24) : 
    base * height = 864 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l1915_191535


namespace Quincy_sold_more_l1915_191593

def ThorSales : ℕ := 200 / 10
def JakeSales : ℕ := ThorSales + 10
def QuincySales : ℕ := 200

theorem Quincy_sold_more (H : QuincySales = 200) : QuincySales - JakeSales = 170 := by
  sorry

end Quincy_sold_more_l1915_191593


namespace smallest_positive_multiple_of_6_and_15_gt_40_l1915_191517

-- Define the LCM function to compute the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Define the statement of the proof problem
theorem smallest_positive_multiple_of_6_and_15_gt_40 : 
  ∃ a : ℕ, (a % 6 = 0) ∧ (a % 15 = 0) ∧ (a > 40) ∧ (∀ b : ℕ, (b % 6 = 0) ∧ (b % 15 = 0) ∧ (b > 40) → a ≤ b) :=
sorry

end smallest_positive_multiple_of_6_and_15_gt_40_l1915_191517


namespace eight_p_plus_one_composite_l1915_191590

theorem eight_p_plus_one_composite 
  (p : ℕ) 
  (hp : Nat.Prime p) 
  (h8p_minus_one : Nat.Prime (8 * p - 1))
  : ¬ (Nat.Prime (8 * p + 1)) :=
sorry

end eight_p_plus_one_composite_l1915_191590


namespace total_nephews_correct_l1915_191599

def alden_nephews_10_years_ago : ℕ := 50

def alden_nephews_now : ℕ :=
  alden_nephews_10_years_ago * 2

def vihaan_nephews_now : ℕ :=
  alden_nephews_now + 60

def total_nephews : ℕ :=
  alden_nephews_now + vihaan_nephews_now

theorem total_nephews_correct : total_nephews = 260 := by
  sorry

end total_nephews_correct_l1915_191599


namespace part_one_part_two_l1915_191592

variable {x : ℝ} {m : ℝ}

-- Question 1
theorem part_one (h : ∀ x : ℝ, mx^2 - mx - 1 < 0) : -4 < m ∧ m <= 0 :=
sorry

-- Question 2
theorem part_two (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → mx^2 - mx - 1 > -m + x - 1) : m > 1 :=
sorry

end part_one_part_two_l1915_191592


namespace josh_marbles_l1915_191543

theorem josh_marbles (initial_marbles lost_marbles : ℕ) (h_initial : initial_marbles = 9) (h_lost : lost_marbles = 5) :
  initial_marbles - lost_marbles = 4 :=
by
  sorry

end josh_marbles_l1915_191543


namespace correct_calculation_l1915_191561

-- Define the variables used in the problem
variables (a x y : ℝ)

-- The main theorem statement
theorem correct_calculation : (2 * x * y^2 - x * y^2 = x * y^2) :=
by sorry

end correct_calculation_l1915_191561


namespace isosceles_base_angle_eq_43_l1915_191549

theorem isosceles_base_angle_eq_43 (α β : ℝ) (h_iso : α = β) (h_sum : α + β + 94 = 180) : α = 43 :=
by
  sorry

end isosceles_base_angle_eq_43_l1915_191549


namespace single_elimination_tournament_l1915_191539

theorem single_elimination_tournament (teams : ℕ) (prelim_games : ℕ) (post_prelim_teams : ℕ) :
  teams = 24 →
  prelim_games = 4 →
  post_prelim_teams = teams - prelim_games →
  post_prelim_teams - 1 + prelim_games = 23 :=
by
  intros
  sorry

end single_elimination_tournament_l1915_191539


namespace Amelia_weekly_sales_l1915_191597

-- Conditions
def monday_sales : ℕ := 45
def tuesday_sales : ℕ := 45 - 16
def remaining_sales : ℕ := 16

-- Question to Answer
def total_weekly_sales : ℕ := 90

-- Lean 4 Statement to Prove
theorem Amelia_weekly_sales : monday_sales + tuesday_sales + remaining_sales = total_weekly_sales :=
by
  sorry

end Amelia_weekly_sales_l1915_191597


namespace cubes_end_same_digits_l1915_191524

theorem cubes_end_same_digits (a b : ℕ) (h : a % 1000 = b % 1000) : (a^3) % 1000 = (b^3) % 1000 := by
  sorry

end cubes_end_same_digits_l1915_191524


namespace trucks_have_160_containers_per_truck_l1915_191566

noncomputable def containers_per_truck: ℕ :=
  let boxes1 := 7 * 20
  let boxes2 := 5 * 12
  let total_boxes := boxes1 + boxes2
  let total_containers := total_boxes * 8
  let trucks := 10
  total_containers / trucks

theorem trucks_have_160_containers_per_truck:
  containers_per_truck = 160 :=
by
  sorry

end trucks_have_160_containers_per_truck_l1915_191566


namespace min_value_reciprocals_l1915_191540

open Real

theorem min_value_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + b = 1) :
  (1 / a + 1 / b) = 4 :=
by
  sorry

end min_value_reciprocals_l1915_191540


namespace transformed_roots_equation_l1915_191562

theorem transformed_roots_equation (α β : ℂ) (h1 : 3 * α^2 + 2 * α + 1 = 0) (h2 : 3 * β^2 + 2 * β + 1 = 0) :
  ∃ (y : ℂ), (y - (3 * α + 2)) * (y - (3 * β + 2)) = y^2 + 4 := 
sorry

end transformed_roots_equation_l1915_191562


namespace beaver_group_count_l1915_191565

theorem beaver_group_count (B : ℕ) (h1 : 3 * B = 60) : B = 20 :=
by sorry

end beaver_group_count_l1915_191565


namespace cubic_root_identity_l1915_191533

theorem cubic_root_identity (x1 x2 x3 : ℝ) (h1 : x1^3 - 3*x1 - 1 = 0) (h2 : x2^3 - 3*x2 - 1 = 0) (h3 : x3^3 - 3*x3 - 1 = 0) (h4 : x1 < x2) (h5 : x2 < x3) :
  x3^2 - x2^2 = x3 - x1 :=
sorry

end cubic_root_identity_l1915_191533


namespace more_apples_than_pears_l1915_191589

-- Define the variables
def apples := 17
def pears := 9

-- Theorem: The number of apples minus the number of pears equals 8
theorem more_apples_than_pears : apples - pears = 8 :=
by
  sorry

end more_apples_than_pears_l1915_191589


namespace bamboo_fifth_section_volume_l1915_191569

theorem bamboo_fifth_section_volume
  (a₁ q : ℝ)
  (h1 : a₁ * (a₁ * q) * (a₁ * q^2) = 3)
  (h2 : (a₁ * q^6) * (a₁ * q^7) * (a₁ * q^8) = 9) :
  a₁ * q^4 = Real.sqrt 3 :=
sorry

end bamboo_fifth_section_volume_l1915_191569


namespace total_emails_received_l1915_191505

theorem total_emails_received :
  let e1 := 16
  let e2 := e1 / 2
  let e3 := e2 / 2
  let e4 := e3 / 2
  e1 + e2 + e3 + e4 = 30 :=
by
  sorry

end total_emails_received_l1915_191505


namespace mrs_hilt_bees_l1915_191522

theorem mrs_hilt_bees (n : ℕ) (h : 3 * n = 432) : n = 144 := by
  sorry

end mrs_hilt_bees_l1915_191522


namespace clock_angle_8_15_l1915_191510

theorem clock_angle_8_15:
  ∃ angle : ℝ, time_on_clock = 8.25 → angle = 157.5 := sorry

end clock_angle_8_15_l1915_191510


namespace simplify_expression_l1915_191573

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := 
by 
  sorry

end simplify_expression_l1915_191573


namespace union_of_A_and_B_l1915_191532

def setA : Set ℝ := {x | 2 * x - 1 > 0}
def setB : Set ℝ := {x | abs x < 1}

theorem union_of_A_and_B : setA ∪ setB = {x : ℝ | x > -1} := 
by {
  sorry
}

end union_of_A_and_B_l1915_191532


namespace total_football_games_l1915_191501

theorem total_football_games (games_this_year : ℕ) (games_last_year : ℕ) (total_games : ℕ) : 
  games_this_year = 14 → games_last_year = 29 → total_games = games_this_year + games_last_year → total_games = 43 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_football_games_l1915_191501


namespace sum_of_roots_eq_h_over_4_l1915_191553

theorem sum_of_roots_eq_h_over_4 (x1 x2 h b : ℝ) (h_ne : x1 ≠ x2)
  (hx1 : 4 * x1 ^ 2 - h * x1 = b) (hx2 : 4 * x2 ^ 2 - h * x2 = b) : x1 + x2 = h / 4 :=
sorry

end sum_of_roots_eq_h_over_4_l1915_191553


namespace original_number_eq_nine_l1915_191564

theorem original_number_eq_nine (N : ℕ) (h1 : ∃ k : ℤ, N - 4 = 5 * k) : N = 9 :=
sorry

end original_number_eq_nine_l1915_191564


namespace weight_of_new_person_l1915_191586

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight new_weight : ℝ) 
  (h_avg_increase : avg_increase = 1.5) (h_num_persons : num_persons = 9) (h_old_weight : old_weight = 65) 
  (h_new_weight_increase : new_weight = old_weight + num_persons * avg_increase) : 
  new_weight = 78.5 :=
sorry

end weight_of_new_person_l1915_191586


namespace sum_of_perimeters_of_squares_l1915_191552

theorem sum_of_perimeters_of_squares (x : ℝ) (h₁ : x = 3) :
  let area1 := x^2 + 4 * x + 4
  let area2 := 4 * x^2 - 12 * x + 9
  let side1 := Real.sqrt area1
  let side2 := Real.sqrt area2
  let perim1 := 4 * side1
  let perim2 := 4 * side2
  perim1 + perim2 = 32 :=
by
  sorry

end sum_of_perimeters_of_squares_l1915_191552


namespace inequality_abc_l1915_191530

variable {a b c : ℝ}

-- Assume a, b, c are positive real numbers
def positive_real_numbers (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Assume the sum of any two numbers is greater than the third
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Lean 4 statement for the proof problem
theorem inequality_abc (h1 : positive_real_numbers a b c) (h2 : triangle_inequality a b c) :
  abc ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
sorry

end inequality_abc_l1915_191530


namespace min_distance_from_start_after_9_minutes_l1915_191568

noncomputable def robot_min_distance : ℝ :=
  let movement_per_minute := 10
  sorry

theorem min_distance_from_start_after_9_minutes :
  robot_min_distance = 10 :=
sorry

end min_distance_from_start_after_9_minutes_l1915_191568


namespace sum_diff_square_cube_l1915_191582

/-- If the sum of two numbers is 25 and the difference between them is 15,
    then the difference between the square of the larger number and the cube of the smaller number is 275. -/
theorem sum_diff_square_cube (x y : ℝ) 
  (h1 : x + y = 25)
  (h2 : x - y = 15) :
  x^2 - y^3 = 275 :=
sorry

end sum_diff_square_cube_l1915_191582


namespace mutual_acquainted_or_unacquainted_l1915_191512

theorem mutual_acquainted_or_unacquainted :
  ∀ (G : SimpleGraph (Fin 6)), 
  ∃ (V : Finset (Fin 6)), V.card = 3 ∧ ((∀ (u v : Fin 6), u ∈ V → v ∈ V → G.Adj u v) ∨ (∀ (u v : Fin 6), u ∈ V → v ∈ V → ¬G.Adj u v)) :=
by
  sorry

end mutual_acquainted_or_unacquainted_l1915_191512


namespace boat_distance_against_stream_l1915_191571

theorem boat_distance_against_stream 
  (v_b : ℝ)
  (v_s : ℝ)
  (distance_downstream : ℝ)
  (t : ℝ)
  (speed_downstream : v_s + v_b = 11)
  (speed_still_water : v_b = 8)
  (time : t = 1) :
  (v_b - (11 - v_b)) * t = 5 :=
by
  -- Here we're given the initial conditions and have to show the final distance against the stream is 5 km
  sorry

end boat_distance_against_stream_l1915_191571


namespace min_c_plus_d_l1915_191556

theorem min_c_plus_d (c d : ℤ) (h : c * d = 36) : c + d = -37 :=
sorry

end min_c_plus_d_l1915_191556


namespace correct_student_answer_l1915_191545

theorem correct_student_answer :
  (9 - (3^2) / 8 = 9 - (9 / 8)) ∧
  (24 - (4 * (3^2)) = 24 - 36) ∧
  ((36 - 12) / (3 / 2) = 24 * (2 / 3)) ∧
  ((-3)^2 / (1 / 3) * 3 = 9 * 3 * 3) →
  (24 * (2 / 3) = 16) :=
by
  sorry

end correct_student_answer_l1915_191545


namespace find_number_exists_l1915_191596

theorem find_number_exists (n : ℤ) : (50 < n ∧ n < 70) ∧
    (n % 5 = 3) ∧
    (n % 7 = 2) ∧
    (n % 8 = 2) → n = 58 := 
sorry

end find_number_exists_l1915_191596


namespace solve_for_b_l1915_191560

theorem solve_for_b (a b c m : ℚ) (h : m = c * a * b / (a - b)) : b = (m * a) / (m + c * a) :=
by
  sorry

end solve_for_b_l1915_191560


namespace gerald_paid_l1915_191579

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 :=
by
  sorry

end gerald_paid_l1915_191579


namespace range_of_m_l1915_191523

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3 * m) ↔ m ≤ -1 ∨ m ≥ 4 :=
by
  sorry

end range_of_m_l1915_191523


namespace log_base_16_of_4_eq_half_l1915_191525

noncomputable def logBase (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_base_16_of_4_eq_half :
  logBase 16 4 = 1 / 2 := by
sorry

end log_base_16_of_4_eq_half_l1915_191525


namespace percentage_difference_l1915_191503

noncomputable def P : ℝ := 40
variables {w x y z : ℝ}
variables (H1 : w = x * (1 - P / 100))
variables (H2 : x = 0.6 * y)
variables (H3 : z = 0.54 * y)
variables (H4 : z = 1.5 * w)

-- Goal
theorem percentage_difference : P = 40 :=
by sorry -- Proof omitted

end percentage_difference_l1915_191503


namespace power_of_product_l1915_191518

variable (x y: ℝ)

theorem power_of_product :
  (-2 * x * y^3)^2 = 4 * x^2 * y^6 := 
by
  sorry

end power_of_product_l1915_191518


namespace gcd_n_cube_plus_16_n_plus_4_l1915_191576

theorem gcd_n_cube_plus_16_n_plus_4 (n : ℕ) (h1 : n > 16) : 
  Nat.gcd (n^3 + 16) (n + 4) = Nat.gcd 48 (n + 4) :=
by
  sorry

end gcd_n_cube_plus_16_n_plus_4_l1915_191576


namespace sin_B_value_cos_A_value_l1915_191581

theorem sin_B_value (A B C S : Real)
  (h1: ∃ (a b c : Real), 
    (a * c * Real.cos (π - B) = (3/2) * (1/2) * a * c * Real.sin B) ∧ 
    (S = (1/2) * a * c * Real.sin B)) : 
  Real.sin B = 4/5 :=
sorry

theorem cos_A_value (A B C : Real)
  (h1: ∃ (a b c : Real), 
    (a * c * Real.cos (π - B) = (3/2) * (1/2) * a * c * Real.sin B) ∧ 
    (S = (1/2) * a * c * Real.sin B)) 
  (h2: A - C = π/4)
  (h3: Real.sin B = 4/5) 
  (h4: Real.cos B = -3/5): 
  Real.cos A = Real.sqrt (50 + 5 * Real.sqrt 2) / 10 :=
sorry

end sin_B_value_cos_A_value_l1915_191581


namespace point_on_x_axis_l1915_191570

theorem point_on_x_axis (a : ℝ) (h₁ : 1 - a = 0) : (3 * a - 6, 1 - a) = (-3, 0) :=
by
  sorry

end point_on_x_axis_l1915_191570


namespace equation_true_when_n_eq_2_l1915_191542

theorem equation_true_when_n_eq_2 : (2 ^ (2 / 2)) = 2 :=
by
  sorry

end equation_true_when_n_eq_2_l1915_191542


namespace part_I_part_II_part_III_l1915_191555

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem part_I : 
  ∀ x:ℝ, f x = x^3 - x :=
by sorry

theorem part_II : 
  ∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (-1:ℝ) 1 ∧ x2 ∈ Set.Icc (-1:ℝ) 1 ∧ (3 * x1^2 - 1) * (3 * x2^2 - 1) = -1 :=
by sorry

theorem part_III (x_n y_m : ℝ) (hx : x_n ∈ Set.Icc (-1:ℝ) 1) (hy : y_m ∈ Set.Icc (-1:ℝ) 1) : 
  |f x_n - f y_m| < 1 :=
by sorry

end part_I_part_II_part_III_l1915_191555


namespace ott_fraction_of_total_money_l1915_191531

-- Definitions for the conditions
def Moe_initial_money (x : ℕ) : ℕ := 3 * x
def Loki_initial_money (x : ℕ) : ℕ := 5 * x
def Nick_initial_money (x : ℕ) : ℕ := 4 * x
def Total_initial_money (x : ℕ) : ℕ := Moe_initial_money x + Loki_initial_money x + Nick_initial_money x
def Ott_received_money (x : ℕ) : ℕ := 3 * x

-- Making the statement we want to prove
theorem ott_fraction_of_total_money (x : ℕ) : 
  (Ott_received_money x) / (Total_initial_money x) = 1 / 4 := by
  sorry

end ott_fraction_of_total_money_l1915_191531


namespace area_of_triangle_ABC_l1915_191554

theorem area_of_triangle_ABC
  {A B C : Type*} 
  (AC BC : ℝ)
  (B : ℝ)
  (h1 : AC = Real.sqrt (13))
  (h2 : BC = 1)
  (h3 : B = Real.sqrt 3 / 2): 
  ∃ area : ℝ, area = Real.sqrt 3 := 
sorry

end area_of_triangle_ABC_l1915_191554


namespace residue_of_neg_1237_mod_29_l1915_191541

theorem residue_of_neg_1237_mod_29 :
  (-1237 : ℤ) % 29 = 10 :=
sorry

end residue_of_neg_1237_mod_29_l1915_191541


namespace solve_for_x_l1915_191521

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (-2, x)
def add_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def sub_vectors (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def is_parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

theorem solve_for_x : ∀ x : ℝ, is_parallel (add_vectors a (b x)) (sub_vectors a (b x)) → x = -4 :=
by
  intros x h_par
  sorry

end solve_for_x_l1915_191521


namespace marble_draw_probability_l1915_191509

def probability_first_white_second_red : ℚ :=
  let total_marbles := 10
  let white_marbles := 6
  let red_marbles := 4
  let first_white_probability := white_marbles / total_marbles
  let remaining_marbles_after_first_draw := total_marbles - 1
  let second_red_probability := red_marbles / remaining_marbles_after_first_draw
  first_white_probability * second_red_probability

theorem marble_draw_probability :
  probability_first_white_second_red = 4 / 15 := by
  sorry

end marble_draw_probability_l1915_191509


namespace combine_polynomials_find_value_profit_or_loss_l1915_191580

-- Problem 1, Part ①
theorem combine_polynomials (a b : ℝ) : -3 * (a+b)^2 - 6 * (a+b)^2 + 8 * (a+b)^2 = -(a+b)^2 := 
sorry

-- Problem 1, Part ②
theorem find_value (a b c d : ℝ) (h1 : a - 2 * b = 5) (h2 : 2 * b - c = -7) (h3 : c - d = 12) : 
  4 * (a - c) + 4 * (2 * b - d) - 4 * (2 * b - c) = 40 := 
sorry

-- Problem 2
theorem profit_or_loss (initial_cost : ℝ) (selling_prices : ℕ → ℝ) (base_price : ℝ) 
  (h_prices : selling_prices 0 = -3) (h_prices1 : selling_prices 1 = 7) 
  (h_prices2 : selling_prices 2 = -8) (h_prices3 : selling_prices 3 = 9) 
  (h_prices4 : selling_prices 4 = -2) (h_prices5 : selling_prices 5 = 0) 
  (h_prices6 : selling_prices 6 = -1) (h_prices7 : selling_prices 7 = -6) 
  (h_initial_cost : initial_cost = 400) (h_base_price : base_price = 56) : 
  (selling_prices 0 + selling_prices 1 + selling_prices 2 + selling_prices 3 + selling_prices 4 + selling_prices 5 + 
  selling_prices 6 + selling_prices 7 + 8 * base_price) - initial_cost > 0 := 
sorry

end combine_polynomials_find_value_profit_or_loss_l1915_191580


namespace calculate_expression_l1915_191558

theorem calculate_expression : 6 * (8 + 1/3) = 50 := by
  sorry

end calculate_expression_l1915_191558


namespace factorize_x_pow_m_minus_x_pow_m_minus_2_l1915_191547

theorem factorize_x_pow_m_minus_x_pow_m_minus_2 (x : ℝ) (m : ℕ) (h : m > 1) : 
  x ^ m - x ^ (m - 2) = (x ^ (m - 2)) * (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_pow_m_minus_x_pow_m_minus_2_l1915_191547


namespace sequence_bound_l1915_191507

theorem sequence_bound (a : ℕ → ℝ) (n : ℕ) 
  (h₁ : a 0 = 0) 
  (h₂ : a (n + 1) = 0)
  (h₃ : ∀ k, 1 ≤ k → k ≤ n → a (k - 1) - 2 * (a k) + (a (k + 1)) ≤ 1) 
  : ∀ k, 0 ≤ k → k ≤ n + 1 → a k ≤ (k * (n + 1 - k)) / 2 :=
sorry

end sequence_bound_l1915_191507


namespace interest_earned_l1915_191538

noncomputable def simple_interest (P R T : ℚ) : ℚ :=
  P * R * T

noncomputable def T_years : ℚ :=
  5 + (8 / 12) + (12 / 365)

def principal : ℚ := 30000
def rate : ℚ := 23.7 / 100

theorem interest_earned :
  simple_interest principal rate T_years = 40524 := by
  sorry

end interest_earned_l1915_191538


namespace fish_population_estimate_l1915_191508

theorem fish_population_estimate 
  (caught_first : ℕ) 
  (caught_first_marked : ℕ) 
  (caught_second : ℕ) 
  (caught_second_marked : ℕ) 
  (proportion_eq : (caught_second_marked : ℚ) / caught_second = (caught_first_marked : ℚ) / caught_first) 
  : caught_first * caught_second / caught_second_marked = 750 := 
by 
  sorry

-- Conditions used as definitions in Lean 4
def pond_fish_total (caught_first : ℕ) (caught_second : ℕ) (caught_second_marked : ℕ) : ℚ :=
  (caught_first : ℚ) * (caught_second : ℚ) / (caught_second_marked : ℚ)

-- Example usage of conditions
example : pond_fish_total 30 50 2 = 750 := 
by
  sorry

end fish_population_estimate_l1915_191508


namespace fraction_option_C_l1915_191520

def is_fraction (expr : String) : Prop := 
  expr = "fraction"

def option_C_fraction (x : ℝ) : Prop :=
  ∃ (numerator : ℝ), ∃ (denominator : ℝ), 
  numerator = 2 ∧ denominator = x + 3

theorem fraction_option_C (x : ℝ) (h : x ≠ -3) :
  is_fraction "fraction" ↔ option_C_fraction x :=
by 
  sorry

end fraction_option_C_l1915_191520


namespace units_digit_sum_42_4_24_4_l1915_191550

theorem units_digit_sum_42_4_24_4 : (42^4 + 24^4) % 10 = 2 := 
by
  sorry

end units_digit_sum_42_4_24_4_l1915_191550


namespace andrew_donates_160_to_homeless_shelter_l1915_191511

/-- Andrew's bake sale earnings -/
def totalEarnings : ℕ := 400

/-- Amount kept by Andrew for ingredients -/
def ingredientsCost : ℕ := 100

/-- Amount Andrew donates from his own piggy bank -/
def piggyBankDonation : ℕ := 10

/-- The total amount Andrew donates to the homeless shelter -/
def totalDonationToHomelessShelter : ℕ :=
  let remaining := totalEarnings - ingredientsCost
  let halfDonation := remaining / 2
  halfDonation + piggyBankDonation

theorem andrew_donates_160_to_homeless_shelter : totalDonationToHomelessShelter = 160 := by
  sorry

end andrew_donates_160_to_homeless_shelter_l1915_191511


namespace find_a_l1915_191598

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + a - 1 = 0}

theorem find_a (a : ℝ) (h : A ∪ B a = A) : a = 2 ∨ a = 3 := by
  sorry

end find_a_l1915_191598


namespace sugarCubeWeight_l1915_191528

theorem sugarCubeWeight
  (ants1 : ℕ) (sugar_cubes1 : ℕ) (weight1 : ℕ) (hours1 : ℕ)
  (ants2 : ℕ) (sugar_cubes2 : ℕ) (hours2 : ℕ) :
  ants1 = 15 →
  sugar_cubes1 = 600 →
  weight1 = 10 →
  hours1 = 5 →
  ants2 = 20 →
  sugar_cubes2 = 960 →
  hours2 = 3 →
  ∃ weight2 : ℕ, weight2 = 5 := by
  sorry

end sugarCubeWeight_l1915_191528


namespace appropriate_selling_price_l1915_191513

-- Define the given conditions
def cost_per_kg : ℝ := 40
def base_price : ℝ := 50
def base_sales_volume : ℝ := 500
def sales_decrease_per_yuan : ℝ := 10
def available_capital : ℝ := 10000
def desired_profit : ℝ := 8000

-- Define the sales volume function dependent on selling price x
def sales_volume (x : ℝ) : ℝ := base_sales_volume - (x - base_price) * sales_decrease_per_yuan

-- Define the profit function dependent on selling price x
def profit (x : ℝ) : ℝ := (x - cost_per_kg) * sales_volume x

-- Prove that the appropriate selling price is 80 yuan
theorem appropriate_selling_price : 
  ∃ x : ℝ, profit x = desired_profit ∧ x = 80 :=
by
  sorry

end appropriate_selling_price_l1915_191513


namespace grantRooms_is_2_l1915_191574

/-- Danielle's apartment has 6 rooms. -/
def danielleRooms : ℕ := 6

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment. -/
def heidiRooms : ℕ := 3 * danielleRooms

/-- Grant's apartment has 1/9 as many rooms as Heidi's apartment. -/
def grantRooms : ℕ := heidiRooms / 9

/-- Prove that Grant's apartment has 2 rooms. -/
theorem grantRooms_is_2 : grantRooms = 2 := by
  sorry

end grantRooms_is_2_l1915_191574


namespace prob1_prob2_l1915_191583

theorem prob1 : -2 + 5 - |(-8 : ℤ)| + (-5) = -10 := 
by
  sorry

theorem prob2 : (-2 : ℤ)^2 * 5 - (-2)^3 / 4 = 22 := 
by
  sorry

end prob1_prob2_l1915_191583


namespace ratio_of_albert_to_mary_l1915_191563

variables (A M B : ℕ) (s : ℕ) 

-- Given conditions as hypotheses
noncomputable def albert_is_multiple_of_mary := A = s * M
noncomputable def albert_is_4_times_betty := A = 4 * B
noncomputable def mary_is_22_years_younger := M = A - 22
noncomputable def betty_is_11 := B = 11

-- Theorem to prove the ratio of Albert's age to Mary's age
theorem ratio_of_albert_to_mary 
  (h1 : albert_is_multiple_of_mary A M s) 
  (h2 : albert_is_4_times_betty A B) 
  (h3 : mary_is_22_years_younger A M) 
  (h4 : betty_is_11 B) : 
  A / M = 2 :=
by
  sorry

end ratio_of_albert_to_mary_l1915_191563


namespace rotation_phenomena_l1915_191529

/-- 
The rotation of the hour hand fits the definition of rotation since it turns around 
the center of the clock, covering specific angles as time passes.
-/
def is_rotation_of_hour_hand : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The rotation of the Ferris wheel fits the definition of rotation since it turns around 
its central axis, making a complete circle.
-/
def is_rotation_of_ferris_wheel : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The annual decline of the groundwater level does not fit the definition of rotation 
since it is a vertical movement (translation).
-/
def is_not_rotation_of_groundwater_level : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The movement of the robots on the conveyor belt does not fit the definition of rotation 
since it is a linear/translational movement.
-/
def is_not_rotation_of_robots_on_conveyor : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
Proof that the phenomena which belong to rotation are exactly the rotation of the hour hand 
and the rotation of the Ferris wheel.
-/
theorem rotation_phenomena :
  is_rotation_of_hour_hand ∧ 
  is_rotation_of_ferris_wheel ∧ 
  is_not_rotation_of_groundwater_level ∧ 
  is_not_rotation_of_robots_on_conveyor →
  "①②" = "①②" :=
by
  intro h
  sorry

end rotation_phenomena_l1915_191529


namespace bar_graph_represents_circle_graph_l1915_191595

theorem bar_graph_represents_circle_graph (r b g : ℕ) 
  (h1 : r = g) 
  (h2 : b = 3 * r) : 
  (r = 1 ∧ b = 3 ∧ g = 1) :=
sorry

end bar_graph_represents_circle_graph_l1915_191595


namespace find_number_l1915_191594

theorem find_number (x : ℕ) (h : x + 20 + x + 30 + x + 40 + x + 10 = 4100) : x = 1000 := 
by
  sorry

end find_number_l1915_191594


namespace football_game_attendance_l1915_191546

theorem football_game_attendance :
  ∃ y : ℕ, (∃ x : ℕ, x + y = 280 ∧ 60 * x + 25 * y = 14000) ∧ y = 80 :=
by
  sorry

end football_game_attendance_l1915_191546


namespace books_sold_l1915_191585

theorem books_sold (initial_books left_books sold_books : ℕ) (h1 : initial_books = 108) (h2 : left_books = 66) : sold_books = 42 :=
by
  have : sold_books = initial_books - left_books := sorry
  rw [h1, h2] at this
  exact this

end books_sold_l1915_191585


namespace point_A_is_closer_to_origin_l1915_191527

theorem point_A_is_closer_to_origin (A B : ℤ) (hA : A = -2) (hB : B = 3) : abs A < abs B := by 
sorry

end point_A_is_closer_to_origin_l1915_191527


namespace sum_of_altitudes_l1915_191534

theorem sum_of_altitudes (x y : ℝ) (hline : 10 * x + 8 * y = 80):
  let A := 1 / 2 * 8 * 10
  let hypotenuse := Real.sqrt (8 ^ 2 + 10 ^ 2)
  let third_altitude := 80 / hypotenuse
  let sum_altitudes := 8 + 10 + third_altitude
  sum_altitudes = 18 + 40 / Real.sqrt 41 := by
  sorry

end sum_of_altitudes_l1915_191534


namespace colored_pencils_count_l1915_191514

-- Given conditions
def bundles := 7
def pencils_per_bundle := 10
def extra_colored_pencils := 3

-- Calculations based on conditions
def total_pencils : ℕ := bundles * pencils_per_bundle
def total_colored_pencils : ℕ := total_pencils + extra_colored_pencils

-- Statement to be proved
theorem colored_pencils_count : total_colored_pencils = 73 := by
  sorry

end colored_pencils_count_l1915_191514


namespace parallelepiped_side_lengths_l1915_191551

theorem parallelepiped_side_lengths (x y z : ℕ) 
  (h1 : x + y + z = 17) 
  (h2 : 2 * x * y + 2 * y * z + 2 * z * x = 180) 
  (h3 : x^2 + y^2 = 100) :
  x = 8 ∧ y = 6 ∧ z = 3 :=
by {
  sorry
}

end parallelepiped_side_lengths_l1915_191551


namespace no_such_integers_l1915_191557

theorem no_such_integers (x y z : ℤ) : ¬ ((x - y)^3 + (y - z)^3 + (z - x)^3 = 2011) :=
sorry

end no_such_integers_l1915_191557


namespace evaluate_f_diff_l1915_191548

def f (x : ℝ) : ℝ := x^4 + 3 * x^3 + 2 * x^2 + 7 * x

theorem evaluate_f_diff:
  f 6 - f (-6) = 1380 := by
  sorry

end evaluate_f_diff_l1915_191548


namespace sum_first_2017_terms_l1915_191504

-- Given sequence definition
def a : ℕ → ℕ
| 0       => 0 -- a_0 (dummy term for 1-based index convenience)
| 1       => 1
| (n + 2) => 3 * 2^(n) - a (n + 1)

-- Sum of the first n terms of the sequence {a_n}
def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + a (n + 1)

-- Theorem to prove
theorem sum_first_2017_terms : S 2017 = 2^2017 - 1 :=
sorry

end sum_first_2017_terms_l1915_191504


namespace possible_values_of_n_are_1_prime_or_prime_squared_l1915_191577

/-- A function that determines if an n x n grid with n marked squares satisfies the condition
    that every rectangle of exactly n grid squares contains at least one marked square. -/
def satisfies_conditions (n : ℕ) (marked_squares : List (ℕ × ℕ)) : Prop :=
  n.succ.succ ≤ marked_squares.length ∧ ∀ (a b : ℕ), a * b = n → ∃ x y, (x, y) ∈ marked_squares ∧ x < n ∧ y < n

/-- The main theorem stating the possible values of n. -/
theorem possible_values_of_n_are_1_prime_or_prime_squared :
  ∀ (n : ℕ), (∃ p : ℕ, Prime p ∧ (n = 1 ∨ n = p ∨ n = p^2)) ↔ satisfies_conditions n marked_squares :=
by
  sorry

end possible_values_of_n_are_1_prime_or_prime_squared_l1915_191577


namespace train_crossing_time_l1915_191502
-- Part a: Identifying the questions and conditions

-- Question: How long does it take for the train to cross the platform?
-- Conditions:
-- 1. Speed of the train: 72 km/hr
-- 2. Length of the goods train: 440 m
-- 3. Length of the platform: 80 m

-- Part b: Identifying the solution steps and the correct answers

-- The solution steps involve:
-- 1. Summing the lengths of the train and the platform to get the total distance the train needs to cover.
-- 2. Converting the speed of the train from km/hr to m/s.
-- 3. Using the formula Time = Distance / Speed to find the time.

-- Correct answer: 26 seconds

-- Part c: Translating the question, conditions, and correct answer to a mathematically equivalent proof problem

-- Goal: Prove that the time it takes for the train to cross the platform is 26 seconds given the provided conditions.

-- Part d: Writing the Lean 4 statement


-- Definitions based on the given conditions
def speed_kmh : ℕ := 72
def length_train : ℕ := 440
def length_platform : ℕ := 80

-- Definition based on the conversion step in the solution
def speed_ms : ℕ := (72 * 1000) / 3600 -- Converting speed from km/hr to m/s

-- Goal: Prove that the time it takes for the train to cross the platform is 26 seconds
theorem train_crossing_time : ((length_train + length_platform) : ℕ) / speed_ms = 26 := by
  sorry

end train_crossing_time_l1915_191502


namespace area_circle_minus_square_l1915_191500

theorem area_circle_minus_square {r : ℝ} (h : r = 1/2) : 
  (π * r^2) - (1^2) = (π / 4) - 1 :=
by
  rw [h]
  sorry

end area_circle_minus_square_l1915_191500


namespace total_soldiers_correct_l1915_191536

-- Definitions based on conditions
def num_generals := 8
def num_vanguards := 8^2
def num_flags := 8^3
def num_team_leaders := 8^4
def num_armored_soldiers := 8^5
def num_soldiers := 8 + 8^2 + 8^3 + 8^4 + 8^5 + 8^6

-- Prove total number of soldiers
theorem total_soldiers_correct : num_soldiers = (1 / 7 : ℝ) * (8^7 - 8) := by
  sorry

end total_soldiers_correct_l1915_191536


namespace mouse_jumps_28_inches_further_than_grasshopper_l1915_191567

theorem mouse_jumps_28_inches_further_than_grasshopper :
  let g_initial := 19
  let g_obstacle := 3
  let g_actual := g_initial - g_obstacle
  let f_difference := 10
  let f_actual := g_initial + f_difference
  let m_difference := 20
  let m_obstacle := 5
  let m_actual := f_actual + m_difference - m_obstacle
  let g_to_m_difference := m_actual - g_actual
  g_to_m_difference = 28 :=
by
  let g_initial := 19
  let g_obstacle := 3
  let g_actual := g_initial - g_obstacle
  let f_difference := 10
  let f_actual := g_initial + f_difference
  let m_difference := 20
  let m_obstacle := 5
  let m_actual := f_actual + m_difference - m_obstacle
  let g_to_m_difference := m_actual - g_actual
  show g_to_m_difference = 28
  sorry

end mouse_jumps_28_inches_further_than_grasshopper_l1915_191567


namespace inequality_transformation_l1915_191537

theorem inequality_transformation (x y : ℝ) (h : 2 * x - 5 < 2 * y - 5) : x < y := 
by 
  sorry

end inequality_transformation_l1915_191537


namespace carnival_tickets_l1915_191588

theorem carnival_tickets (x : ℕ) (won_tickets : ℕ) (found_tickets : ℕ) (ticket_value : ℕ) (total_value : ℕ)
  (h1 : won_tickets = 5 * x)
  (h2 : found_tickets = 5)
  (h3 : ticket_value = 3)
  (h4 : total_value = 30)
  (h5 : total_value = (won_tickets + found_tickets) * ticket_value) :
  x = 1 :=
by
  -- Proof omitted
  sorry

end carnival_tickets_l1915_191588


namespace prove_expression_value_l1915_191526

-- Define the conditions
variables {a b c d m : ℤ}
variable (h1 : a + b = 0)
variable (h2 : |m| = 2)
variable (h3 : c * d = 1)

-- State the theorem
theorem prove_expression_value : (a + b) / (4 * m) + 2 * m ^ 2 - 3 * c * d = 5 :=
by
  -- Proof goes here
  sorry

end prove_expression_value_l1915_191526


namespace find_p_l1915_191516

-- Define the coordinates of the points
structure Point where
  x : Real
  y : Real

def Q := Point.mk 0 15
def A := Point.mk 3 15
def B := Point.mk 15 0
def O := Point.mk 0 0
def C (p : Real) := Point.mk 0 p

-- Given the area of triangle ABC and the coordinates of Q, A, B, O, and C, prove that p = 12.75
theorem find_p (p : Real) (h_area_ABC : 36 = 36) (h_Q : Q = Point.mk 0 15)
                (h_A : A = Point.mk 3 15) (h_B : B = Point.mk 15 0) 
                (h_O : O = Point.mk 0 0) : p = 12.75 := 
sorry

end find_p_l1915_191516


namespace train_crossing_time_l1915_191559

/-!
## Problem Statement
A train 400 m in length crosses a telegraph post. The speed of the train is 90 km/h. Prove that it takes 16 seconds for the train to cross the telegraph post.
-/

-- Defining the given definitions based on the conditions in a)
def train_length : ℕ := 400
def train_speed_kmh : ℕ := 90
def train_speed_ms : ℚ := 25 -- Converting 90 km/h to 25 m/s

-- Proving the problem statement
theorem train_crossing_time : train_length / train_speed_ms = 16 := 
by
  -- convert conditions and show expected result
  sorry

end train_crossing_time_l1915_191559


namespace B_gain_l1915_191584

-- Problem statement and conditions
def principalA : ℝ := 3500
def rateA : ℝ := 0.10
def periodA : ℕ := 2
def principalB : ℝ := 3500
def rateB : ℝ := 0.14
def periodB : ℕ := 3

-- Calculate amount A will receive from B after 2 years
noncomputable def amountA := principalA * (1 + rateA / 1) ^ periodA

-- Calculate amount B will receive from C after 3 years
noncomputable def amountB := principalB * (1 + rateB / 2) ^ (2 * periodB)

-- Calculate B's gain
noncomputable def gainB := amountB - amountA

-- The theorem to prove
theorem B_gain : gainB = 1019.20 := by
  sorry

end B_gain_l1915_191584


namespace volume_of_pyramid_base_isosceles_right_triangle_l1915_191591

theorem volume_of_pyramid_base_isosceles_right_triangle (a h : ℝ) (ha : a = 3) (hh : h = 4) :
  (1 / 3) * (1 / 2) * a * a * h = 6 := by
  sorry

end volume_of_pyramid_base_isosceles_right_triangle_l1915_191591


namespace convex_polyhedron_triangular_face_or_three_edges_vertex_l1915_191515

theorem convex_polyhedron_triangular_face_or_three_edges_vertex
  (M N K : ℕ) 
  (euler_formula : N - M + K = 2) :
  ∃ (f : ℕ), (f ≤ N ∧ f = 3) ∨ ∃ (v : ℕ), (v ≤ K ∧ v = 3) := 
sorry

end convex_polyhedron_triangular_face_or_three_edges_vertex_l1915_191515


namespace number_of_green_hats_l1915_191519

variables (B G : ℕ)

-- Given conditions as definitions
def totalHats : Prop := B + G = 85
def totalCost : Prop := 6 * B + 7 * G = 530

-- The statement we need to prove
theorem number_of_green_hats (h1 : totalHats B G) (h2 : totalCost B G) : G = 20 :=
sorry

end number_of_green_hats_l1915_191519


namespace percentage_democrats_l1915_191575

/-- In a certain city, some percent of the registered voters are Democrats and the rest are Republicans. In a mayoral race, 85 percent of the registered voters who are Democrats and 20 percent of the registered voters who are Republicans are expected to vote for candidate A. Candidate A is expected to get 59 percent of the registered voters' votes. Prove that 60 percent of the registered voters are Democrats. -/
theorem percentage_democrats (D R : ℝ) (h : D + R = 100) (h1 : 0.85 * D + 0.20 * R = 59) : 
  D = 60 :=
by
  sorry

end percentage_democrats_l1915_191575


namespace edward_friend_scores_l1915_191587

theorem edward_friend_scores (total_points friend_points edward_points : ℕ) (h1 : total_points = 13) (h2 : edward_points = 7) (h3 : friend_points = total_points - edward_points) : friend_points = 6 := 
by
  rw [h1, h2] at h3
  exact h3

end edward_friend_scores_l1915_191587


namespace train_length_360_l1915_191544

variable (time_to_cross : ℝ) (speed_of_train : ℝ)

theorem train_length_360 (h1 : time_to_cross = 12) (h2 : speed_of_train = 30) :
  speed_of_train * time_to_cross = 360 :=
by
  rw [h1, h2]
  norm_num

end train_length_360_l1915_191544


namespace least_possible_value_expression_l1915_191578

theorem least_possible_value_expression :
  ∃ min_value : ℝ, ∀ x : ℝ, ((x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019) ≥ min_value ∧ min_value = 2018 :=
by
  sorry

end least_possible_value_expression_l1915_191578


namespace trigonometric_identity_l1915_191572

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end trigonometric_identity_l1915_191572
