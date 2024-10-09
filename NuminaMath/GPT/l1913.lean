import Mathlib

namespace find_fourth_number_l1913_191311

theorem find_fourth_number (x : ℝ) (h : 3 + 33 + 333 + x = 369.63) : x = 0.63 :=
sorry

end find_fourth_number_l1913_191311


namespace geom_seq_min_value_l1913_191329

theorem geom_seq_min_value :
  let a1 := 2
  ∃ r : ℝ, ∀ a2 a3,
    a2 = 2 * r ∧ 
    a3 = 2 * r^2 →
    3 * a2 + 6 * a3 = -3/2 := by
  sorry

end geom_seq_min_value_l1913_191329


namespace pages_read_per_hour_l1913_191359

theorem pages_read_per_hour (lunch_time : ℕ) (book_pages : ℕ) (round_trip_time : ℕ)
  (h1 : lunch_time = round_trip_time)
  (h2 : book_pages = 4000)
  (h3 : round_trip_time = 2 * 4) :
  book_pages / (2 * lunch_time) = 250 :=
by
  sorry

end pages_read_per_hour_l1913_191359


namespace find_original_price_l1913_191381

-- Define the original price P
variable (P : ℝ)

-- Define the conditions as per the given problem
def revenue_equation (P : ℝ) : Prop :=
  820 = (10 * 0.60 * P) + (20 * 0.85 * P) + (18 * P)

-- Prove that the revenue equation implies P = 20
theorem find_original_price (P : ℝ) (h : revenue_equation P) : P = 20 :=
  by sorry

end find_original_price_l1913_191381


namespace original_profit_percentage_l1913_191338

theorem original_profit_percentage (C S : ℝ) (hC : C = 70)
(h1 : S - 14.70 = 1.30 * (C * 0.80)) :
  (S - C) / C * 100 = 25 := by
  sorry

end original_profit_percentage_l1913_191338


namespace find_interval_for_a_l1913_191378

-- Define the system of equations as a predicate
def system_of_equations (a x y z : ℝ) : Prop := 
  x + y + z = 0 ∧ x * y + y * z + a * z * x = 0

-- Define the condition that (0, 0, 0) is the only solution
def unique_solution (a : ℝ) : Prop :=
  ∀ x y z : ℝ, system_of_equations a x y z → x = 0 ∧ y = 0 ∧ z = 0

-- Rewrite the proof problem as a Lean statement
theorem find_interval_for_a :
  ∀ a : ℝ, unique_solution a ↔ 0 < a ∧ a < 4 :=
by
  sorry

end find_interval_for_a_l1913_191378


namespace find_julios_bonus_l1913_191346

def commission (customers: ℕ) : ℕ :=
  customers * 1

def total_commission (week1: ℕ) (week2: ℕ) (week3: ℕ) : ℕ :=
  commission week1 + commission week2 + commission week3

noncomputable def julios_bonus (total_earnings salary total_commission: ℕ) : ℕ :=
  total_earnings - salary - total_commission

theorem find_julios_bonus :
  let week1 := 35
  let week2 := 2 * week1
  let week3 := 3 * week1
  let salary := 500
  let total_earnings := 760
  let total_comm := total_commission week1 week2 week3
  julios_bonus total_earnings salary total_comm = 50 :=
by
  sorry

end find_julios_bonus_l1913_191346


namespace maximum_value_of_expression_is_4_l1913_191321

noncomputable def maximimum_integer_value (x : ℝ) : ℝ :=
    (5 * x^2 + 10 * x + 12) / (5 * x^2 + 10 * x + 2)

theorem maximum_value_of_expression_is_4 :
    ∃ x : ℝ, ∀ y : ℝ, maximimum_integer_value y ≤ 4 ∧ maximimum_integer_value x = 4 := 
by 
  -- Proof omitted for now
  sorry

end maximum_value_of_expression_is_4_l1913_191321


namespace faye_pencils_allocation_l1913_191336

theorem faye_pencils_allocation (pencils total_pencils rows : ℕ) (h_pencils : total_pencils = 6) (h_rows : rows = 2) (h_allocation : pencils = total_pencils / rows) : pencils = 3 := by
  sorry

end faye_pencils_allocation_l1913_191336


namespace cost_of_one_pack_l1913_191374

-- Given condition
def total_cost (packs: ℕ) : ℕ := 110
def number_of_packs : ℕ := 10

-- Question: How much does one pack cost?
-- We need to prove that one pack costs 11 dollars
theorem cost_of_one_pack : (total_cost number_of_packs) / number_of_packs = 11 :=
by
  sorry

end cost_of_one_pack_l1913_191374


namespace winston_initial_gas_l1913_191370

theorem winston_initial_gas (max_gas : ℕ) (store_gas : ℕ) (doctor_gas : ℕ) :
  store_gas = 6 → doctor_gas = 2 → max_gas = 12 → max_gas - (store_gas + doctor_gas) = 4 → max_gas = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end winston_initial_gas_l1913_191370


namespace bus_speed_excluding_stoppages_l1913_191383

theorem bus_speed_excluding_stoppages (S : ℝ) (h₀ : 0 < S) (h₁ : 36 = (2/3) * S) : S = 54 :=
by 
  sorry

end bus_speed_excluding_stoppages_l1913_191383


namespace at_least_one_negative_l1913_191372

theorem at_least_one_negative (a : Fin 7 → ℤ) :
  (∀ i j : Fin 7, i ≠ j → a i ≠ a j) ∧
  (∀ l1 l2 l3 : Fin 7, 
    a l1 + a l2 + a l3 = a l1 + a l2 + a l3) ∧
  (∃ i : Fin 7, a i = 0) →
  (∃ i : Fin 7, a i < 0) :=
  by
  sorry

end at_least_one_negative_l1913_191372


namespace find_number_l1913_191306

theorem find_number (x : ℝ) (h : 0.05 * x = 12.75) : x = 255 :=
by
  sorry

end find_number_l1913_191306


namespace part1_a_eq_zero_part2_range_of_a_l1913_191312

noncomputable def f (x : ℝ) := abs (x + 1)
noncomputable def g (x : ℝ) (a : ℝ) := 2 * abs x + a

theorem part1_a_eq_zero :
  ∀ x, 0 < x + 1 → 0 < 2 * abs x → a = 0 →
  f x ≥ g x a ↔ (-1 / 3 : ℝ) ≤ x ∧ x ≤ 1 :=
sorry

theorem part2_range_of_a :
  ∃ x, f x ≥ g x a ↔ a ≤ 1 :=
sorry

end part1_a_eq_zero_part2_range_of_a_l1913_191312


namespace centroid_of_triangle_l1913_191308

-- Definitions and conditions
def is_lattice_point (p : ℤ × ℤ) : Prop := 
  true -- Placeholder for a more specific definition if necessary

def triangle (A B C : ℤ × ℤ) : Prop := 
  true -- Placeholder for defining a triangle with vertices at integer grid points

def no_other_nodes_on_sides (A B C : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert no other integer grid points on the sides

def exactly_one_node_inside (A B C O : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert exactly one integer grid point inside the triangle

def medians_intersection_is_point_O (A B C O : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert \(O\) is the intersection point of the medians

-- Theorem statement
theorem centroid_of_triangle 
  (A B C O : ℤ × ℤ)
  (h1 : is_lattice_point A)
  (h2 : is_lattice_point B)
  (h3 : is_lattice_point C)
  (h4 : triangle A B C)
  (h5 : no_other_nodes_on_sides A B C)
  (h6 : exactly_one_node_inside A B C O) : 
  medians_intersection_is_point_O A B C O :=
sorry

end centroid_of_triangle_l1913_191308


namespace identify_heaviest_and_lightest_13_weighings_l1913_191318

theorem identify_heaviest_and_lightest_13_weighings (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ f : (Fin 13 → ((Fin 10) × (Fin 10) × ℝ)), true :=
by
  sorry

end identify_heaviest_and_lightest_13_weighings_l1913_191318


namespace gcd_lcm_sum_l1913_191343

-- Definitions
def gcd_42_70 := Nat.gcd 42 70
def lcm_8_32 := Nat.lcm 8 32

-- Theorem statement
theorem gcd_lcm_sum : gcd_42_70 + lcm_8_32 = 46 := by
  sorry

end gcd_lcm_sum_l1913_191343


namespace integers_with_product_72_and_difference_4_have_sum_20_l1913_191335

theorem integers_with_product_72_and_difference_4_have_sum_20 :
  ∃ (x y : ℕ), (x * y = 72) ∧ (x - y = 4) ∧ (x + y = 20) :=
sorry

end integers_with_product_72_and_difference_4_have_sum_20_l1913_191335


namespace water_left_after_experiment_l1913_191365

theorem water_left_after_experiment (initial_water : ℝ) (used_water : ℝ) (result_water : ℝ) 
  (h1 : initial_water = 3) 
  (h2 : used_water = 9 / 4) 
  (h3 : result_water = 3 / 4) : 
  initial_water - used_water = result_water := by
  sorry

end water_left_after_experiment_l1913_191365


namespace amanda_final_quiz_score_l1913_191398

theorem amanda_final_quiz_score
  (average_score_4quizzes : ℕ)
  (total_quizzes : ℕ)
  (average_a : ℕ)
  (current_score : ℕ)
  (required_total_score : ℕ)
  (required_score_final_quiz : ℕ) :
  average_score_4quizzes = 92 →
  total_quizzes = 5 →
  average_a = 93 →
  current_score = 4 * average_score_4quizzes →
  required_total_score = total_quizzes * average_a →
  required_score_final_quiz = required_total_score - current_score →
  required_score_final_quiz = 97 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end amanda_final_quiz_score_l1913_191398


namespace min_value_f_min_value_f_sqrt_min_value_f_2_min_m_l1913_191330

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * Real.log x + b

theorem min_value_f 
  (a b : ℝ) 
  (a_non_pos : a ≤ 1) : 
  f 1 a b = (1 / 2) + b :=
sorry

theorem min_value_f_sqrt 
  (a b : ℝ) 
  (a_pos_range : 1 < a ∧ a < 4) : 
  f (Real.sqrt a) a b = (a / 2) - a * Real.log (Real.sqrt a) + b :=
sorry

theorem min_value_f_2 
  (a b : ℝ) 
  (a_ge_4 : 4 ≤ a) : 
  f 2 a b = 2 - a * Real.log 2 + b :=
sorry

theorem min_m 
  (a : ℝ) 
  (a_range : -2 ≤ a ∧ a < 0):
  ∀x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 2) ∧ (0 < x2 ∧ x2 ≤ 2) →
  ∃m : ℝ, m = 12 ∧ abs (f x1 a 0 - f x2 a 0) ≤ m ^ abs (1 / x1 - 1 / x2) :=
sorry

end min_value_f_min_value_f_sqrt_min_value_f_2_min_m_l1913_191330


namespace factorize_difference_of_squares_l1913_191373

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 81 = (a + 9) * (a - 9) :=
by
  sorry

end factorize_difference_of_squares_l1913_191373


namespace sara_initial_pears_l1913_191323

theorem sara_initial_pears (given_to_dan : ℕ) (left_with_sara : ℕ) (total : ℕ) :
  given_to_dan = 28 ∧ left_with_sara = 7 ∧ total = given_to_dan + left_with_sara → total = 35 :=
by
  sorry

end sara_initial_pears_l1913_191323


namespace max_value_phi_l1913_191355

theorem max_value_phi (φ : ℝ) (hφ : -Real.pi / 2 < φ ∧ φ < Real.pi / 2) :
  (∃ k : ℤ, φ = 2 * k * Real.pi + Real.pi / 2 - Real.pi / 3) →
  φ = Real.pi / 6 :=
by 
  intro h
  sorry

end max_value_phi_l1913_191355


namespace not_prime_a_l1913_191322

theorem not_prime_a 
  (a b : ℕ) 
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : ∃ k : ℤ, (5 * a^4 + a^2) = k * (b^4 + 3 * b^2 + 4))
  : ¬ Nat.Prime a := 
sorry

end not_prime_a_l1913_191322


namespace log_inequality_l1913_191360

theorem log_inequality (a b c : ℝ) (h1 : b^2 - a * c < 0) :
  ∀ x y : ℝ, a * (Real.log x)^2 + 2 * b * (Real.log x) * (Real.log y) + c * (Real.log y)^2 = 1 
  → a * 1^2 + 2 * b * 1 * (-1) + c * (-1)^2 = 1 → 
  -1 / Real.sqrt (a * c - b^2) ≤ Real.log (x * y) ∧ Real.log (x * y) ≤ 1 / Real.sqrt (a * c - b^2) :=
by
  sorry

end log_inequality_l1913_191360


namespace percentage_passed_exam_l1913_191364

theorem percentage_passed_exam (total_students failed_students : ℕ) (h_total : total_students = 540) (h_failed : failed_students = 351) :
  (total_students - failed_students) * 100 / total_students = 35 :=
by
  sorry

end percentage_passed_exam_l1913_191364


namespace jenna_practice_minutes_l1913_191390

theorem jenna_practice_minutes :
  ∀ (practice_6_days practice_2_days target_total target_average: ℕ),
    practice_6_days = 6 * 80 →
    practice_2_days = 2 * 105 →
    target_average = 100 →
    target_total = 9 * target_average →
  ∃ practice_9th_day, (practice_6_days + practice_2_days + practice_9th_day = target_total) ∧ practice_9th_day = 210 :=
by sorry

end jenna_practice_minutes_l1913_191390


namespace number_div_addition_l1913_191366

-- Define the given conditions
def original_number (q d r : ℕ) : ℕ := (q * d) + r

theorem number_div_addition (q d r a b : ℕ) (h1 : d = 6) (h2 : q = 124) (h3 : r = 4) (h4 : a = 24) (h5 : b = 8) :
  ((original_number q d r + a) / b : ℚ) = 96.5 :=
by 
  sorry

end number_div_addition_l1913_191366


namespace pencil_eraser_cost_l1913_191326

theorem pencil_eraser_cost (p e : ℕ) (h1 : 15 * p + 5 * e = 200) (h2 : p > e) (h_p_pos : p > 0) (h_e_pos : e > 0) :
  p + e = 18 :=
  sorry

end pencil_eraser_cost_l1913_191326


namespace nonnegative_for_interval_l1913_191310

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 * (x - 2)^2) / ((1 - x) * (1 + x + x^2))

theorem nonnegative_for_interval (x : ℝ) : (f x >= 0) ↔ (0 <= x) :=
by
  sorry

end nonnegative_for_interval_l1913_191310


namespace sumOfTrianglesIs34_l1913_191348

def triangleOp (a b c : ℕ) : ℕ := a * b - c

theorem sumOfTrianglesIs34 : 
  triangleOp 3 5 2 + triangleOp 4 6 3 = 34 := 
by
  sorry

end sumOfTrianglesIs34_l1913_191348


namespace factorize_diff_of_squares_l1913_191303

theorem factorize_diff_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
  sorry

end factorize_diff_of_squares_l1913_191303


namespace number_of_integers_between_sqrt10_and_sqrt100_l1913_191309

theorem number_of_integers_between_sqrt10_and_sqrt100 : 
  let a := Real.sqrt 10
  let b := Real.sqrt 100
  ∃ (n : ℕ), n = 6 ∧ (∀ x : ℕ, (x > a ∧ x < b) → (4 ≤ x ∧ x ≤ 9)) :=
by
  sorry

end number_of_integers_between_sqrt10_and_sqrt100_l1913_191309


namespace perpendicular_condition_l1913_191354

theorem perpendicular_condition (a : ℝ) :
  (a = 1) ↔ (∀ x : ℝ, (a*x + 1 - ((a - 2)*x - 1)) * ((a * x + 1 - (a * x + 1))) = 0) :=
by
  sorry

end perpendicular_condition_l1913_191354


namespace original_price_of_radio_l1913_191331

theorem original_price_of_radio (P : ℝ) (h : 0.95 * P = 465.5) : P = 490 :=
sorry

end original_price_of_radio_l1913_191331


namespace find_number_multiplied_l1913_191385

theorem find_number_multiplied (m : ℕ) (h : 9999 * m = 325027405) : m = 32505 :=
by {
  sorry
}

end find_number_multiplied_l1913_191385


namespace team_A_games_42_l1913_191377

noncomputable def team_games (a b : ℕ) : Prop :=
  (a * 2 / 3 + 7) = b * 5 / 8

theorem team_A_games_42 (a b : ℕ) (h1 : a * 2 / 3 = b * 5 / 8 - 7)
                                 (h2 : b = a + 14) :
  a = 42 :=
by
  sorry

end team_A_games_42_l1913_191377


namespace ellipse_chord_slope_relation_l1913_191319

theorem ellipse_chord_slope_relation
    (a b : ℝ) (h : a > b) (h1 : b > 0)
    (A B M : ℝ × ℝ)
    (hM_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
    (hAB_slope : A.1 ≠ B.1)
    (K_AB K_OM : ℝ)
    (hK_AB : K_AB = (B.2 - A.2) / (B.1 - A.1))
    (hK_OM : K_OM = (M.2 - 0) / (M.1 - 0)) :
  K_AB * K_OM = - (b ^ 2) / (a ^ 2) := 
  sorry

end ellipse_chord_slope_relation_l1913_191319


namespace divisibility_56786730_polynomial_inequality_l1913_191347

theorem divisibility_56786730 (m n : ℤ) : 56786730 ∣ m * n * (m^60 - n^60) :=
sorry

theorem polynomial_inequality (m n : ℤ) : m^5 + 3 * m^4 * n - 5 * m^3 * n^2 - 15 * m^2 * n^3 + 4 * m * n^4 + 12 * n^5 ≠ 33 :=
sorry

end divisibility_56786730_polynomial_inequality_l1913_191347


namespace next_bell_ringing_time_l1913_191337

theorem next_bell_ringing_time (post_office_interval train_station_interval town_hall_interval start_time : ℕ)
  (h1 : post_office_interval = 18)
  (h2 : train_station_interval = 24)
  (h3 : town_hall_interval = 30)
  (h4 : start_time = 9) :
  let lcm := Nat.lcm post_office_interval (Nat.lcm train_station_interval town_hall_interval)
  lcm + start_time = 15 := by
  sorry

end next_bell_ringing_time_l1913_191337


namespace number_of_students_in_third_batch_l1913_191302

theorem number_of_students_in_third_batch
  (avg1 avg2 avg3 : ℕ)
  (total_avg : ℚ)
  (students1 students2 : ℕ)
  (h_avg1 : avg1 = 45)
  (h_avg2 : avg2 = 55)
  (h_avg3 : avg3 = 65)
  (h_total_avg : total_avg = 56.333333333333336)
  (h_students1 : students1 = 40)
  (h_students2 : students2 = 50) :
  ∃ x : ℕ, (students1 * avg1 + students2 * avg2 + x * avg3 = total_avg * (students1 + students2 + x) ∧ x = 60) :=
by
  sorry

end number_of_students_in_third_batch_l1913_191302


namespace H2O_production_l1913_191384

theorem H2O_production (n : Nat) (m : Nat)
  (h1 : n = 3)
  (h2 : m = 3) :
  n = m → n = 3 := by
  sorry

end H2O_production_l1913_191384


namespace x_plus_p_eq_2p_plus_3_l1913_191317

theorem x_plus_p_eq_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2 * p + 3 := by
  sorry

end x_plus_p_eq_2p_plus_3_l1913_191317


namespace top_three_probability_l1913_191344

-- Definitions for the real-world problem
def total_ways_to_choose_three_cards : ℕ :=
  52 * 51 * 50

def favorable_ways_to_choose_three_specific_suits : ℕ :=
  13 * 13 * 13 * 6

def probability_top_three_inclusive (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

-- The mathematically equivalent proof problem's Lean statement
theorem top_three_probability:
  probability_top_three_inclusive total_ways_to_choose_three_cards favorable_ways_to_choose_three_specific_suits = 2197 / 22100 :=
by
  sorry

end top_three_probability_l1913_191344


namespace total_square_footage_l1913_191386

-- Definitions from the problem conditions
def price_per_square_foot : ℝ := 98
def total_property_value : ℝ := 333200

-- The mathematical statement to prove
theorem total_square_footage : (total_property_value / price_per_square_foot) = 3400 :=
by
  -- Proof goes here (skipped with sorry)
  sorry

end total_square_footage_l1913_191386


namespace find_radius_of_cone_l1913_191314

def slant_height : ℝ := 10
def curved_surface_area : ℝ := 157.07963267948966

theorem find_radius_of_cone
    (l : ℝ) (CSA : ℝ) (h1 : l = slant_height) (h2 : CSA = curved_surface_area) :
    ∃ r : ℝ, r = 5 := 
by
  sorry

end find_radius_of_cone_l1913_191314


namespace wall_building_time_l1913_191361

theorem wall_building_time (m1 m2 d1 d2 k : ℕ) (h1 : m1 = 12) (h2 : d1 = 6) (h3 : m2 = 18) (h4 : k = 72) 
  (condition : m1 * d1 = k) (rate_const : m2 * d2 = k) : d2 = 4 := by
  sorry

end wall_building_time_l1913_191361


namespace triangle_height_l1913_191332

theorem triangle_height (base height area : ℝ) 
(h_base : base = 3) (h_area : area = 9) 
(h_area_eq : area = (base * height) / 2) :
  height = 6 := 
by 
  sorry

end triangle_height_l1913_191332


namespace cube_root_expression_l1913_191379

theorem cube_root_expression (N : ℝ) (h : N > 1) : 
    (N^(1/3)^(1/3)^(1/3)^(1/3)) = N^(40/81) :=
sorry

end cube_root_expression_l1913_191379


namespace negation_of_universal_statement_l1913_191375

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, x^3 - 3 * x > 0) ↔ ∃ x : ℝ, x^3 - 3 * x ≤ 0 :=
by
  sorry

end negation_of_universal_statement_l1913_191375


namespace staplers_left_l1913_191368

-- Definitions based on conditions
def initial_staplers : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

-- Statement of the theorem
theorem staplers_left (h : initial_staplers = 50) (d : dozen = 12) (r : reports_stapled = 3 * dozen) :
  (initial_staplers - reports_stapled) = 14 :=
sorry

end staplers_left_l1913_191368


namespace max_value_l1913_191333

theorem max_value (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) : 
  8 * x + 3 * y + 15 * z ≤ Real.sqrt 298 :=
sorry

end max_value_l1913_191333


namespace common_point_eq_l1913_191353

theorem common_point_eq (a b c d : ℝ) (h₀ : a ≠ b) 
  (h₁ : ∃ x y : ℝ, y = a * x + a ∧ y = b * x + b ∧ y = c * x + d) : 
  d = c :=
by
  sorry

end common_point_eq_l1913_191353


namespace remove_max_rooks_l1913_191371

-- Defines the problem of removing the maximum number of rooks under given conditions
theorem remove_max_rooks (n : ℕ) (attacks_odd : (ℕ × ℕ) → ℕ) :
  (∀ p : ℕ × ℕ, (attacks_odd p) % 2 = 1 → true) →
  n = 8 →
  (∃ m, m = 59) :=
by
  intros _ _
  existsi 59
  sorry

end remove_max_rooks_l1913_191371


namespace intersection_complement_eq_l1913_191394

/-- Define the sets U, A, and B -/
def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {3, 7, 9}
def B : Set ℕ := {1, 9}

/-- Define the complement of B with respect to U -/
def complement_U_B : Set ℕ := U \ B

/-- Theorem stating the intersection of A and the complement of B with respect to U -/
theorem intersection_complement_eq : A ∩ complement_U_B = {3, 7} :=
by
  sorry

end intersection_complement_eq_l1913_191394


namespace jack_keeps_10800_pounds_l1913_191316

def number_of_months_in_a_quarter := 12 / 4
def monthly_hunting_trips := 6
def total_hunting_trips := monthly_hunting_trips * number_of_months_in_a_quarter
def deers_per_trip := 2
def total_deers := total_hunting_trips * deers_per_trip
def weight_per_deer := 600
def total_weight := total_deers * weight_per_deer
def kept_weight_fraction := 1 / 2
def kept_weight := total_weight * kept_weight_fraction

theorem jack_keeps_10800_pounds :
  kept_weight = 10800 :=
by
  -- This is a stub for the automated proof
  sorry

end jack_keeps_10800_pounds_l1913_191316


namespace attraction_ticket_cost_for_parents_l1913_191382

noncomputable def total_cost (children parents adults: ℕ) (entrance_cost child_attraction_cost adult_attraction_cost: ℕ) : ℕ :=
  (children + parents + adults) * entrance_cost + children * child_attraction_cost + adults * (adult_attraction_cost)

theorem attraction_ticket_cost_for_parents
  (children parents adults: ℕ) 
  (entrance_cost child_attraction_cost total_cost_of_family: ℕ) 
  (h_children: children = 4)
  (h_parents: parents = 2)
  (h_adults: adults = 1)
  (h_entrance_cost: entrance_cost = 5)
  (h_child_attraction_cost: child_attraction_cost = 2)
  (h_total_cost_of_family: total_cost_of_family = 55)
  : (total_cost children parents adults entrance_cost child_attraction_cost 4 / 3) = total_cost_of_family - (children + parents + adults) * entrance_cost - children * child_attraction_cost := 
sorry

end attraction_ticket_cost_for_parents_l1913_191382


namespace batsman_avg_increase_l1913_191352

theorem batsman_avg_increase (R : ℕ) (A : ℕ) : 
  (R + 48 = 12 * 26) ∧ (R = 11 * A) → 26 - A = 2 :=
by
  intro h
  have h1 : R + 48 = 312 := h.1
  have h2 : R = 11 * A := h.2
  sorry

end batsman_avg_increase_l1913_191352


namespace total_animals_in_savanna_l1913_191369

/-- Define the number of lions, snakes, and giraffes in Safari National Park. --/
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10

/-- Define the number of lions, snakes, and giraffes in Savanna National Park based on conditions. --/
def savanna_lions : ℕ := 2 * safari_lions
def savanna_snakes : ℕ := 3 * safari_snakes
def savanna_giraffes : ℕ := safari_giraffes + 20

/-- Calculate the total number of animals in Savanna National Park. --/
def total_savanna_animals : ℕ := savanna_lions + savanna_snakes + savanna_giraffes

/-- Proof statement that the total number of animals in Savanna National Park is 410.
My goal is to prove that total_savanna_animals is equal to 410. --/
theorem total_animals_in_savanna : total_savanna_animals = 410 :=
by
  sorry

end total_animals_in_savanna_l1913_191369


namespace redPoints_l1913_191358

open Nat

def isRedPoint (x y : ℕ) : Prop :=
  (y = (x - 36) * (x - 144) - 1991) ∧ (∃ m : ℕ, y = m * m)

theorem redPoints :
  {p : ℕ × ℕ | isRedPoint p.1 p.2} = { (2544, 6017209), (444, 120409) } :=
by
  sorry

end redPoints_l1913_191358


namespace composite_expression_l1913_191327

theorem composite_expression (n : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^3 + 6 * n^2 + 12 * n + 7 = a * b :=
by
  sorry

end composite_expression_l1913_191327


namespace sqrt_computation_l1913_191339

open Real

theorem sqrt_computation : sqrt ((5: ℝ)^2 * (7: ℝ)^4) = 245 :=
by
  -- Proof here
  sorry

end sqrt_computation_l1913_191339


namespace perfectSquareLastFourDigits_l1913_191391

noncomputable def lastThreeDigitsForm (n : ℕ) : Prop :=
  ∃ a : ℕ, a ≤ 9 ∧ n % 1000 = a * 111

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfectSquareLastFourDigits (n : ℕ) :
  lastThreeDigitsForm n →
  isPerfectSquare n →
  (n % 10000 = 0 ∨ n % 10000 = 1444) :=
by {
  sorry
}

end perfectSquareLastFourDigits_l1913_191391


namespace men_in_second_group_l1913_191388

theorem men_in_second_group (M : ℕ) (h1 : 36 * 18 = M * 24) : M = 27 :=
by {
  sorry
}

end men_in_second_group_l1913_191388


namespace grasshopper_max_reach_points_l1913_191356

theorem grasshopper_max_reach_points
  (α : ℝ) (α_eq : α = 36 * Real.pi / 180)
  (L : ℕ)
  (jump_constant : ∀ (n : ℕ), L = L) :
  ∃ (N : ℕ), N ≤ 10 :=
by 
  sorry

end grasshopper_max_reach_points_l1913_191356


namespace find_remainder_in_division_l1913_191324

theorem find_remainder_in_division
  (D : ℕ)
  (r : ℕ) -- the remainder when using the incorrect divisor
  (R : ℕ) -- the remainder when using the correct divisor
  (h1 : D = 12 * 63 + r)
  (h2 : D = 21 * 36 + R)
  : R = 0 :=
by
  sorry

end find_remainder_in_division_l1913_191324


namespace isosceles_right_triangle_area_l1913_191350

theorem isosceles_right_triangle_area (p : ℝ) : 
  ∃ (A : ℝ), A = (3 - 2 * Real.sqrt 2) * p^2 
  → (∃ (x : ℝ), 2 * x + x * Real.sqrt 2 = 2 * p ∧ A = 1 / 2 * x^2) := 
sorry

end isosceles_right_triangle_area_l1913_191350


namespace double_seven_eighth_l1913_191313

theorem double_seven_eighth (n : ℕ) (h : n = 48) : 2 * (7 / 8 * n) = 84 := by
  sorry

end double_seven_eighth_l1913_191313


namespace shrub_height_at_end_of_2_years_l1913_191399

theorem shrub_height_at_end_of_2_years (h₅ : ℕ) (h : ∀ n : ℕ, 0 < n → 243 = 3^5 * h₅) : ∃ h₂ : ℕ, h₂ = 9 :=
by sorry

end shrub_height_at_end_of_2_years_l1913_191399


namespace rationalize_denominator_l1913_191395

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l1913_191395


namespace sum_of_first_9_terms_l1913_191320

-- Define the arithmetic sequence {a_n} and the sum S_n of the first n terms
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

-- Define the conditions given in the problem
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom arith_seq : arithmetic_sequence a
axiom sum_terms : sum_of_first_n_terms a S
axiom S3 : S 3 = 30
axiom S6 : S 6 = 100

-- Goal: Prove that S 9 = 170
theorem sum_of_first_9_terms : S 9 = 170 :=
sorry -- Placeholder for the proof

end sum_of_first_9_terms_l1913_191320


namespace operation_example_l1913_191304

def operation (a b : ℤ) : ℤ := 2 * a * b - b^2

theorem operation_example : operation 1 (-3) = -15 := by
  sorry

end operation_example_l1913_191304


namespace problem_l1913_191334

noncomputable def vector_a (ω φ x : ℝ) : ℝ × ℝ := (Real.sin (ω / 2 * x + φ), 1)
noncomputable def vector_b (ω φ x : ℝ) : ℝ × ℝ := (1, Real.cos (ω / 2 * x + φ))
noncomputable def f (ω φ x : ℝ) : ℝ := 
  let a := vector_a ω φ x
  let b := vector_b ω φ x
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2)

theorem problem 
  (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π / 4)
  (h_period : Function.Periodic (f ω φ) 4)
  (h_point1 : f ω φ 1 = 1 / 2) : 
  ω = π / 2 ∧ ∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ f (π / 2) (π / 12) x ∧ f (π / 2) (π / 12) x ≤ 1 / 2 := 
by
  sorry

end problem_l1913_191334


namespace C_investment_l1913_191349

theorem C_investment (A B C_profit total_profit : ℝ) (hA : A = 24000) (hB : B = 32000) (hC_profit : C_profit = 36000) (h_total_profit : total_profit = 92000) (x : ℝ) (h : x / (A + B + x) = C_profit / total_profit) : x = 36000 := 
by
  sorry

end C_investment_l1913_191349


namespace weekly_earnings_proof_l1913_191376

def minutes_in_hour : ℕ := 60
def hourly_rate : ℕ := 4

def monday_minutes : ℕ := 150
def tuesday_minutes : ℕ := 40
def wednesday_minutes : ℕ := 155
def thursday_minutes : ℕ := 45

def weekly_minutes : ℕ := monday_minutes + tuesday_minutes + wednesday_minutes + thursday_minutes
def weekly_hours : ℕ := weekly_minutes / minutes_in_hour

def sylvia_earnings : ℕ := weekly_hours * hourly_rate

theorem weekly_earnings_proof :
  sylvia_earnings = 26 := by
  sorry

end weekly_earnings_proof_l1913_191376


namespace third_chest_coin_difference_l1913_191363

variable (g1 g2 g3 s1 s2 s3 : ℕ)

-- Conditions
axiom h1 : g1 + g2 + g3 = 40
axiom h2 : s1 + s2 + s3 = 40
axiom h3 : g1 = s1 + 7
axiom h4 : g2 = s2 + 15

-- Goal
theorem third_chest_coin_difference : s3 = g3 + 22 :=
sorry

end third_chest_coin_difference_l1913_191363


namespace min_value_fraction_l1913_191300

theorem min_value_fraction (a b : ℝ) (h1 : 2 * a + b = 3) (h2 : a > 0) (h3 : b > 0) (h4 : ∃ n : ℕ, b = n) : 
  (∃ a b : ℝ, 2 * a + b = 3 ∧ a > 0 ∧ b > 0 ∧ (∃ n : ℕ, b = n) ∧ ((1/(2*a) + 2/b) = 2)) := 
by
  sorry

end min_value_fraction_l1913_191300


namespace find_value_l1913_191328

def set_condition (s : Set ℕ) : Prop := s = {0, 1, 2}

def one_relationship_correct (a b c : ℕ) : Prop :=
  (a ≠ 2 ∧ b ≠ 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b = 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b = 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b = 2 ∧ c = 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b ≠ 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c = 1)
  ∨ (a = 2 ∧ b = 0 ∧ c = 1)
  ∨ (a = 2 ∧ b = 0 ∧ c ≠ 0)
  ∨ (a ≠ 2 ∧ b = 0 ∧ c ≠ 0)

theorem find_value (a b c : ℕ) (h1 : set_condition {a, b, c}) (h2 : one_relationship_correct a b c) :
  100 * c + 10 * b + a = 102 :=
sorry

end find_value_l1913_191328


namespace ab_value_l1913_191393

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l1913_191393


namespace three_tenths_of_number_l1913_191380

theorem three_tenths_of_number (x : ℝ) (h : (1/3) * (1/4) * x = 18) : (3/10) * x = 64.8 :=
sorry

end three_tenths_of_number_l1913_191380


namespace problem_solution_l1913_191341

-- Define the problem conditions and state the theorem
variable (a b : ℝ)
variable (h1 : a^2 - 4 * a + 3 = 0)
variable (h2 : b^2 - 4 * b + 3 = 0)
variable (h3 : a ≠ b)

theorem problem_solution : (a+1)*(b+1) = 8 := by
  sorry

end problem_solution_l1913_191341


namespace blue_balls_count_l1913_191301

def num_purple : Nat := 7
def num_yellow : Nat := 11
def min_tries : Nat := 19

theorem blue_balls_count (num_blue: Nat): num_blue = 1 :=
by
  have worst_case_picks := num_purple + num_yellow
  have h := min_tries
  sorry

end blue_balls_count_l1913_191301


namespace trigonometric_identity_application_l1913_191387

theorem trigonometric_identity_application :
  (1 / 2) * (Real.sin (Real.pi / 12)) * (Real.cos (Real.pi / 12)) = (1 / 8) :=
by
  sorry

end trigonometric_identity_application_l1913_191387


namespace conditional_probabilities_l1913_191305

def PA : ℝ := 0.20
def PB : ℝ := 0.18
def PAB : ℝ := 0.12

theorem conditional_probabilities :
  PAB / PB = 2 / 3 ∧ PAB / PA = 3 / 5 := by
  sorry

end conditional_probabilities_l1913_191305


namespace animal_market_problem_l1913_191397

theorem animal_market_problem:
  ∃ (s c : ℕ), 0 < s ∧ 0 < c ∧ 28 * s + 27 * c = 1200 ∧ c > s :=
by
  sorry

end animal_market_problem_l1913_191397


namespace spurs_team_players_l1913_191345

theorem spurs_team_players (total_basketballs : ℕ) (basketballs_per_player : ℕ) (h : total_basketballs = 242) (h1 : basketballs_per_player = 11) : total_basketballs / basketballs_per_player = 22 :=
by { sorry }

end spurs_team_players_l1913_191345


namespace selling_price_calculation_l1913_191367

-- Given conditions
def cost_price : ℚ := 110
def gain_percent : ℚ := 13.636363636363626

-- Theorem Statement
theorem selling_price_calculation : 
  (cost_price * (1 + gain_percent / 100)) = 125 :=
by
  sorry

end selling_price_calculation_l1913_191367


namespace carpet_shaded_area_l1913_191315

theorem carpet_shaded_area
  (side_length_carpet : ℝ)
  (S : ℝ)
  (T : ℝ)
  (h1 : side_length_carpet = 12)
  (h2 : 12 / S = 4)
  (h3 : S / T = 2) :
  let area_big_square := S^2
  let area_small_squares := 4 * T^2
  area_big_square + area_small_squares = 18 := by
  sorry

end carpet_shaded_area_l1913_191315


namespace pyramid_volume_in_unit_cube_l1913_191396

noncomputable def base_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ := (1 / 3) * base_area * height

theorem pyramid_volume_in_unit_cube : 
  let s := Real.sqrt 2 / 2
  let height := 1
  pyramid_volume (base_area s) height = Real.sqrt 3 / 24 :=
by
  sorry

end pyramid_volume_in_unit_cube_l1913_191396


namespace problem1_solution_problem2_solution_l1913_191342

-- Proof for Problem 1
theorem problem1_solution (x y : ℝ) 
(h1 : x - y - 1 = 4)
(h2 : 4 * (x - y) - y = 5) : 
x = 20 ∧ y = 15 := sorry

-- Proof for Problem 2
theorem problem2_solution (x : ℝ) 
(h1 : 4 * x - 1 ≥ x + 1)
(h2 : (1 - x) / 2 < x) : 
x ≥ 2 / 3 := sorry

end problem1_solution_problem2_solution_l1913_191342


namespace find_x_l1913_191340

theorem find_x {x : ℝ} (hx : x^2 - 5 * x = -4) : x = 1 ∨ x = 4 :=
sorry

end find_x_l1913_191340


namespace tic_tac_toe_winning_boards_l1913_191389

-- Define the board as a 4x4 grid
def Board := Array (Array (Option Bool))

-- Define a function that returns all possible board states after 3 moves
noncomputable def numberOfWinningBoards : Nat := 140

theorem tic_tac_toe_winning_boards:
  numberOfWinningBoards = 140 :=
by
  sorry

end tic_tac_toe_winning_boards_l1913_191389


namespace set_union_inter_example_l1913_191351

open Set

theorem set_union_inter_example :
  let A := ({1, 2} : Set ℕ)
  let B := ({1, 2, 3} : Set ℕ)
  let C := ({2, 3, 4} : Set ℕ)
  (A ∩ B) ∪ C = ({1, 2, 3, 4} : Set ℕ) := by
    let A := ({1, 2} : Set ℕ)
    let B := ({1, 2, 3} : Set ℕ)
    let C := ({2, 3, 4} : Set ℕ)
    sorry

end set_union_inter_example_l1913_191351


namespace count_valid_pairs_l1913_191362

open Nat

-- Define the conditions
def room_conditions (p q : ℕ) : Prop :=
  q > p ∧
  (∃ (p' q' : ℕ), p = p' + 6 ∧ q = q' + 6 ∧ p' * q' = 48)

-- State the theorem to prove the number of valid pairs (p, q)
theorem count_valid_pairs : 
  (∃ l : List (ℕ × ℕ), 
    (∀ pq ∈ l, room_conditions pq.fst pq.snd) ∧ 
    l.length = 5) := 
sorry

end count_valid_pairs_l1913_191362


namespace cars_towards_each_other_cars_same_direction_A_to_B_cars_same_direction_B_to_A_l1913_191357

-- Definitions based on conditions
def distanceAB := 18  -- km
def speedCarA := 54   -- km/h
def speedCarB := 36   -- km/h
def targetDistance := 45  -- km

-- Proof problem statements
theorem cars_towards_each_other {y : ℝ} : 54 * y + 36 * y = 18 + 45 ↔ y = 0.7 :=
by sorry

theorem cars_same_direction_A_to_B {x : ℝ} : 54 * x - (36 * x + 18) = 45 ↔ x = 3.5 :=
by sorry

theorem cars_same_direction_B_to_A {x : ℝ} : 54 * x + 18 - 36 * x = 45 ↔ x = 1.5 :=
by sorry

end cars_towards_each_other_cars_same_direction_A_to_B_cars_same_direction_B_to_A_l1913_191357


namespace sufficient_not_necessary_l1913_191392

variable (x : ℝ)

theorem sufficient_not_necessary (h : |x| > 0) : (x > 0 ↔ true) :=
by 
  sorry

end sufficient_not_necessary_l1913_191392


namespace treasure_hunt_distance_l1913_191307

theorem treasure_hunt_distance (d : ℝ) : 
  (d < 8) → (d > 7) → (d > 9) → False :=
by
  intros h1 h2 h3
  sorry

end treasure_hunt_distance_l1913_191307


namespace monotonicity_decreasing_range_l1913_191325

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem monotonicity_decreasing_range (ω : ℝ) :
  (∀ x y : ℝ, (π / 2 < x ∧ x < π ∧ π / 2 < y ∧ y < π ∧ x < y) → f ω x > f ω y) ↔ (1 / 2 ≤ ω ∧ ω ≤ 5 / 4) :=
sorry

end monotonicity_decreasing_range_l1913_191325
