import Mathlib

namespace zero_in_interval_l84_84544

theorem zero_in_interval (x y : ℝ) (hx_lt_0 : x < 0) (hy_gt_0 : 0 < y) (hy_lt_1 : y < 1) (h : x^5 < y^8 ∧ y^8 < y^3 ∧ y^3 < x^6) : x^5 < 0 ∧ 0 < y^8 :=
by
  sorry

end zero_in_interval_l84_84544


namespace dice_total_correct_l84_84978

-- Define the problem conditions
def IvanDice (x : ℕ) : ℕ := x
def JerryDice (x : ℕ) : ℕ := (1 / 2 * x) ^ 2

-- Define the total dice function
def totalDice (x : ℕ) : ℕ := IvanDice x + JerryDice x

-- The theorem to prove the answer
theorem dice_total_correct (x : ℕ) : totalDice x = x + (1 / 4) * x ^ 2 := 
  sorry

end dice_total_correct_l84_84978


namespace equation_nth_position_l84_84119

theorem equation_nth_position (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * n - 9 :=
by
  sorry

end equation_nth_position_l84_84119


namespace calculate_revolutions_l84_84258

def wheel_diameter : ℝ := 8
def distance_traveled_miles : ℝ := 0.5
def feet_per_mile : ℝ := 5280
def distance_traveled_feet : ℝ := distance_traveled_miles * feet_per_mile

theorem calculate_revolutions :
  let radius : ℝ := wheel_diameter / 2
  let circumference : ℝ := 2 * Real.pi * radius
  let revolutions : ℝ := distance_traveled_feet / circumference
  revolutions = 330 / Real.pi := by
  sorry

end calculate_revolutions_l84_84258


namespace alex_annual_income_l84_84216

theorem alex_annual_income (q : ℝ) (B : ℝ)
  (H1 : 0.01 * q * 50000 + 0.01 * (q + 3) * (B - 50000) = 0.01 * (q + 0.5) * B) :
  B = 60000 :=
by sorry

end alex_annual_income_l84_84216


namespace acute_angle_sine_l84_84232

theorem acute_angle_sine (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 0.58) : (π / 6) < α ∧ α < (π / 4) :=
by
  sorry

end acute_angle_sine_l84_84232


namespace jack_marathon_time_l84_84079

theorem jack_marathon_time :
  ∀ {marathon_distance : ℝ} {jill_time : ℝ} {speed_ratio : ℝ},
    marathon_distance = 40 → 
    jill_time = 4 → 
    speed_ratio = 0.888888888888889 → 
    (marathon_distance / (speed_ratio * (marathon_distance / jill_time))) = 4.5 :=
by
  intros marathon_distance jill_time speed_ratio h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jack_marathon_time_l84_84079


namespace wealth_ratio_l84_84634

theorem wealth_ratio 
  (P W : ℝ)
  (hP_pos : 0 < P)
  (hW_pos : 0 < W)
  (pop_A : ℝ := 0.30 * P)
  (wealth_A : ℝ := 0.40 * W)
  (pop_B : ℝ := 0.20 * P)
  (wealth_B : ℝ := 0.25 * W)
  (avg_wealth_A : ℝ := wealth_A / pop_A)
  (avg_wealth_B : ℝ := wealth_B / pop_B) :
  avg_wealth_A / avg_wealth_B = 16 / 15 :=
by
  sorry

end wealth_ratio_l84_84634


namespace sufficient_not_necessary_condition_l84_84558

open Complex

theorem sufficient_not_necessary_condition (a b : ℝ) (i := Complex.I) :
  (a = 1 ∧ b = 1) → ((a + b * i)^2 = 2 * i) ∧ ¬((a + b * i)^2 = 2 * i → a = 1 ∧ b = 1) :=
by
  sorry

end sufficient_not_necessary_condition_l84_84558


namespace employees_count_l84_84063

theorem employees_count (n : ℕ) (avg_salary : ℝ) (manager_salary : ℝ)
  (new_avg_salary : ℝ) (total_employees_with_manager : ℝ) : 
  avg_salary = 1500 → 
  manager_salary = 3600 → 
  new_avg_salary = avg_salary + 100 → 
  total_employees_with_manager = (n + 1) * 1600 → 
  (n * avg_salary + manager_salary) / total_employees_with_manager = new_avg_salary →
  n = 20 := by
  intros
  sorry

end employees_count_l84_84063


namespace triangle_inequality_sqrt_equality_condition_l84_84074

theorem triangle_inequality_sqrt 
  {a b c : ℝ} 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) 
  ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) := 
sorry

theorem equality_condition 
  {a b c : ℝ} 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) 
  = Real.sqrt a + Real.sqrt b + Real.sqrt c) → 
  (a = b ∧ b = c) := 
sorry

end triangle_inequality_sqrt_equality_condition_l84_84074


namespace difference_divisible_by_9_l84_84626

-- Define the integers a and b
variables (a b : ℤ)

-- Define the theorem statement
theorem difference_divisible_by_9 (a b : ℤ) : 9 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
sorry

end difference_divisible_by_9_l84_84626


namespace school_basketballs_l84_84932

theorem school_basketballs (n_classes n_basketballs_per_class total_basketballs : ℕ)
  (h1 : n_classes = 7)
  (h2 : n_basketballs_per_class = 7)
  (h3 : total_basketballs = n_classes * n_basketballs_per_class) :
  total_basketballs = 49 :=
sorry

end school_basketballs_l84_84932


namespace complement_intersection_l84_84336

def U : Set ℤ := Set.univ
def A : Set ℤ := {1, 2}
def B : Set ℤ := {3, 4}

-- A ∪ B should equal {1, 2, 3, 4}
axiom AUeq : A ∪ B = {1, 2, 3, 4}

theorem complement_intersection : (U \ A) ∩ B = {3, 4} :=
by
  sorry

end complement_intersection_l84_84336


namespace right_triangle_hypotenuse_l84_84806

theorem right_triangle_hypotenuse (x : ℝ) (h : x^2 = 3^2 + 5^2) : x = Real.sqrt 34 :=
by sorry

end right_triangle_hypotenuse_l84_84806


namespace set_intersection_l84_84086

-- Definitions of sets M and N
def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {1, 2, 3}

-- The statement to prove that M ∩ N = {1, 2}
theorem set_intersection :
  M ∩ N = {1, 2} := by
  sorry

end set_intersection_l84_84086


namespace general_term_sum_bn_l84_84852

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + 2 * n
noncomputable def a (n : ℕ) : ℕ := 4 * n
noncomputable def b (n : ℕ) : ℕ := 2 ^ (4 * n)
noncomputable def T (n : ℕ) : ℝ := (16 / 15) * (16^n - 1)

theorem general_term (n : ℕ) (h1 : S n = 2 * n^2 + 2 * n) 
    (h2 : S (n-1) = 2 * (n-1)^2 + 2 * (n-1))
    (h3 : n ≥ 1) : a n = 4 * n :=
by sorry

theorem sum_bn (n : ℕ) (h : ∀ n, (b n, a n) = ((2 ^ (4 * n)), 4 * n)) : 
    T n = (16 / 15) * (16^n - 1) :=
by sorry

end general_term_sum_bn_l84_84852


namespace number_of_friends_l84_84170

/- Define the conditions -/
def sandwiches_per_friend : Nat := 3
def total_sandwiches : Nat := 12

/- Define the mathematical statement to be proven -/
theorem number_of_friends : (total_sandwiches / sandwiches_per_friend) = 4 :=
by
  sorry

end number_of_friends_l84_84170


namespace knicks_win_tournament_probability_l84_84361

noncomputable def knicks_win_probability : ℚ :=
  let knicks_win_proba := 2 / 5
  let heat_win_proba := 3 / 5
  let first_4_games_scenarios := 6 * (knicks_win_proba^2 * heat_win_proba^2)
  first_4_games_scenarios * knicks_win_proba

theorem knicks_win_tournament_probability :
  knicks_win_probability = 432 / 3125 :=
by
  sorry

end knicks_win_tournament_probability_l84_84361


namespace probability_x_add_y_lt_4_in_square_l84_84392

noncomputable def square_area : ℝ := 3 * 3

noncomputable def triangle_area : ℝ := (1 / 2) * 2 * 2

noncomputable def region_area : ℝ := square_area - triangle_area

noncomputable def probability (A B : ℝ) : ℝ := A / B

theorem probability_x_add_y_lt_4_in_square :
  probability region_area square_area = 7 / 9 :=
by 
  sorry

end probability_x_add_y_lt_4_in_square_l84_84392


namespace series_sum_equals_one_l84_84525

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (2 : ℝ)^(2 * (k + 1)) / ((3 : ℝ)^(2 * (k + 1)) - 1)

theorem series_sum_equals_one :
  series_sum = 1 :=
sorry

end series_sum_equals_one_l84_84525


namespace rectangle_perimeter_l84_84608

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem rectangle_perimeter
  (a1 a2 a3 a4 a5 a6 a7 a8 a9 l w : ℕ)
  (h1 : a1 + a2 + a3 = a9)
  (h2 : a1 + a2 = a3)
  (h3 : a1 + a3 = a4)
  (h4 : a3 + a4 = a5)
  (h5 : a4 + a5 = a6)
  (h6 : a2 + a3 + a5 = a7)
  (h7 : a2 + a7 = a8)
  (h8 : a1 + a4 + a6 = a9)
  (h9 : a6 + a9 = a7 + a8)
  (h_rel_prime : relatively_prime l w)
  (h_dimensions : l = 61)
  (h_dimensions_w : w = 69) :
  2 * l + 2 * w = 260 := by
  sorry

end rectangle_perimeter_l84_84608


namespace find_base_number_l84_84814

-- Define the base number
def base_number (x : ℕ) (k : ℕ) : Prop := x ^ k > 4 ^ 22

-- State the theorem based on the problem conditions
theorem find_base_number : ∃ x : ℕ, ∀ k : ℕ, (k = 8) → (base_number x k) → (x = 64) :=
by sorry

end find_base_number_l84_84814


namespace time_to_school_gate_l84_84027

theorem time_to_school_gate (total_time gate_to_building building_to_room time_to_gate : ℕ) 
                            (h1 : total_time = 30)
                            (h2 : gate_to_building = 6)
                            (h3 : building_to_room = 9)
                            (h4 : total_time = time_to_gate + gate_to_building + building_to_room) :
  time_to_gate = 15 :=
  sorry

end time_to_school_gate_l84_84027


namespace product_gcd_lcm_is_correct_l84_84573

-- Define the numbers
def a := 15
def b := 75

-- Definitions related to GCD and LCM
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b
def product_gcd_lcm := gcd_ab * lcm_ab

-- Theorem stating the product of GCD and LCM of a and b is 1125
theorem product_gcd_lcm_is_correct : product_gcd_lcm = 1125 := by
  sorry

end product_gcd_lcm_is_correct_l84_84573


namespace zack_initial_marbles_l84_84442

theorem zack_initial_marbles :
  let a1 := 20
  let a2 := 30
  let a3 := 35
  let a4 := 25
  let a5 := 28
  let a6 := 40
  let r := 7
  let T := a1 + a2 + a3 + a4 + a5 + a6 + r
  T = 185 :=
by
  sorry

end zack_initial_marbles_l84_84442


namespace rate_of_current_l84_84764

theorem rate_of_current (speed_boat_still_water : ℕ) (time_hours : ℚ) (distance_downstream : ℚ)
    (h_speed_boat_still_water : speed_boat_still_water = 20)
    (h_time_hours : time_hours = 15 / 60)
    (h_distance_downstream : distance_downstream = 6.25) :
    ∃ c : ℚ, distance_downstream = (speed_boat_still_water + c) * time_hours ∧ c = 5 :=
by
    sorry

end rate_of_current_l84_84764


namespace education_budget_l84_84851

-- Definitions of the conditions
def total_budget : ℕ := 32 * 10^6  -- 32 million
def policing_budget : ℕ := total_budget / 2
def public_spaces_budget : ℕ := 4 * 10^6  -- 4 million

-- The theorem statement
theorem education_budget :
  total_budget - (policing_budget + public_spaces_budget) = 12 * 10^6 :=
by
  sorry

end education_budget_l84_84851


namespace problem_statement_l84_84649

noncomputable def polynomial_expansion (x : ℚ) : ℚ := (1 - 2 * x) ^ 8

theorem problem_statement :
  (8 * (1 - 2 * 1) ^ 7 * (-2)) = (a_1 : ℚ) + 2 * (a_2 : ℚ) + 3 * (a_3 : ℚ) + 4 * (a_4 : ℚ) +
  5 * (a_5 : ℚ) + 6 * (a_6 : ℚ) + 7 * (a_7 : ℚ) + 8 * (a_8 : ℚ) := by 
  sorry

end problem_statement_l84_84649


namespace x_lt_1_iff_x_abs_x_lt_1_l84_84803

theorem x_lt_1_iff_x_abs_x_lt_1 (x : ℝ) : x < 1 ↔ x * |x| < 1 :=
sorry

end x_lt_1_iff_x_abs_x_lt_1_l84_84803


namespace coordinates_B_l84_84683

theorem coordinates_B (A B : ℝ × ℝ) (distance : ℝ) (A_coords : A = (-1, 3)) 
  (AB_parallel_x : A.snd = B.snd) (AB_distance : abs (A.fst - B.fst) = distance) :
  (B = (-6, 3) ∨ B = (4, 3)) :=
by
  sorry

end coordinates_B_l84_84683


namespace sqrt_meaningful_range_l84_84552

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 3) : 3 ≤ x :=
by
  linarith

end sqrt_meaningful_range_l84_84552


namespace parallelogram_sides_l84_84556

theorem parallelogram_sides (x y : ℝ) (h1 : 12 * y - 2 = 10) (h2 : 5 * x + 15 = 20) : x + y = 2 :=
by
  sorry

end parallelogram_sides_l84_84556


namespace sqrt_mul_example_complex_expression_example_l84_84904

theorem sqrt_mul_example : Real.sqrt 3 * Real.sqrt 27 = 9 :=
by sorry

theorem complex_expression_example : 
  (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) - (Real.sqrt 3 - 2)^2 = 4 * Real.sqrt 3 - 6 :=
by sorry

end sqrt_mul_example_complex_expression_example_l84_84904


namespace distance_between_X_and_Y_l84_84234

def distance_XY := 31

theorem distance_between_X_and_Y
  (yolanda_rate : ℕ) (bob_rate : ℕ) (bob_walked : ℕ) (time_difference : ℕ) :
  yolanda_rate = 1 →
  bob_rate = 2 →
  bob_walked = 20 →
  time_difference = 1 →
  distance_XY = bob_walked + (bob_walked / bob_rate + time_difference) * yolanda_rate :=
by
  intros hy hb hbw htd
  sorry

end distance_between_X_and_Y_l84_84234


namespace lion_room_is_3_l84_84512

/-!
  A lion is hidden in one of three rooms. A note on the door of room 1 reads "The lion is here".
  A note on the door of room 2 reads "The lion is not here". A note on the door of room 3 reads "2+3=2×3".
  Only one of these notes is true. Prove that the lion is in room 3.
-/

def note1 (lion_room : ℕ) : Prop := lion_room = 1
def note2 (lion_room : ℕ) : Prop := lion_room ≠ 2
def note3 (lion_room : ℕ) : Prop := 2 + 3 = 2 * 3
def lion_is_in_room3 : Prop := ∀ lion_room, (note1 lion_room ∨ note2 lion_room ∨ note3 lion_room) ∧
  (note1 lion_room → note2 lion_room = false) ∧ (note1 lion_room → note3 lion_room = false) ∧
  (note2 lion_room → note1 lion_room = false) ∧ (note2 lion_room → note3 lion_room = false) ∧
  (note3 lion_room → note1 lion_room = false) ∧ (note3 lion_room → note2 lion_room = false) → lion_room = 3

theorem lion_room_is_3 : lion_is_in_room3 := 
  by
  sorry

end lion_room_is_3_l84_84512


namespace find_extra_lives_first_level_l84_84333

-- Conditions as definitions
def initial_lives : ℕ := 2
def extra_lives_second_level : ℕ := 11
def total_lives_after_second_level : ℕ := 19

-- Definition representing the extra lives in the first level
def extra_lives_first_level (x : ℕ) : Prop :=
  initial_lives + x + extra_lives_second_level = total_lives_after_second_level

-- The theorem we need to prove
theorem find_extra_lives_first_level : ∃ x : ℕ, extra_lives_first_level x ∧ x = 6 :=
by
  sorry  -- Placeholder for the proof

end find_extra_lives_first_level_l84_84333


namespace simplify_expr1_simplify_expr2_simplify_expr3_l84_84891

-- 1. Proving (1)(2x^{2})^{3}-x^{2}·x^{4} = 7x^{6}
theorem simplify_expr1 (x : ℝ) : (1 : ℝ) * (2 * x^2)^3 - x^2 * x^4 = 7 * x^6 := 
by 
  sorry

-- 2. Proving (a+b)^{2}-b(2a+b) = a^{2}
theorem simplify_expr2 (a b : ℝ) : (a + b)^2 - b * (2 * a + b) = a^2 := 
by 
  sorry

-- 3. Proving (x+1)(x-1)-x^{2} = -1
theorem simplify_expr3 (x : ℝ) : (x + 1) * (x - 1) - x^2 = -1 :=
by 
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l84_84891


namespace initially_planned_days_l84_84779

-- Definitions of the conditions
def total_work_initial (x : ℕ) : ℕ := 50 * x
def total_work_with_reduction (x : ℕ) : ℕ := 25 * (x + 20)

-- The main theorem
theorem initially_planned_days :
  ∀ (x : ℕ), total_work_initial x = total_work_with_reduction x → x = 20 :=
by
  intro x
  intro h
  sorry

end initially_planned_days_l84_84779


namespace ratio_of_A_to_B_l84_84938

theorem ratio_of_A_to_B (v_A v_B : ℝ) (d_A d_B : ℝ) (h1 : d_A = 128) (h2 : d_B = 64) (h3 : d_A / v_A = d_B / v_B) : v_A / v_B = 2 := 
by
  sorry

end ratio_of_A_to_B_l84_84938


namespace no_positive_integer_solutions_l84_84748

theorem no_positive_integer_solutions (x y z : ℕ) (h_cond : x^2 + y^2 = 7 * z^2) : 
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_positive_integer_solutions_l84_84748


namespace everton_college_calculators_l84_84439

theorem everton_college_calculators (total_cost : ℤ) (num_scientific_calculators : ℤ) 
  (cost_per_scientific : ℤ) (cost_per_graphing : ℤ) (total_scientific_cost : ℤ) 
  (num_graphing_calculators : ℤ) (total_graphing_cost : ℤ) (total_calculators : ℤ) :
  total_cost = 1625 ∧
  num_scientific_calculators = 20 ∧
  cost_per_scientific = 10 ∧
  cost_per_graphing = 57 ∧
  total_scientific_cost = num_scientific_calculators * cost_per_scientific ∧
  total_graphing_cost = num_graphing_calculators * cost_per_graphing ∧
  total_cost = total_scientific_cost + total_graphing_cost ∧
  total_calculators = num_scientific_calculators + num_graphing_calculators → 
  total_calculators = 45 :=
by
  intros
  sorry

end everton_college_calculators_l84_84439


namespace minimum_value_of_xy_l84_84300

noncomputable def minimum_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y + 12 = x * y) : ℝ :=
  if hmin : 4 * x + y + 12 = x * y then 36 else sorry

theorem minimum_value_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y + 12 = x * y) : 
  minimum_value_xy x y hx hy h = 36 :=
sorry

end minimum_value_of_xy_l84_84300


namespace positive_integer_iff_positive_real_l84_84646

theorem positive_integer_iff_positive_real (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℕ, n > 0 ∧ abs ((x - 2 * abs x) * abs x) / x = n) ↔ x > 0 :=
by
  sorry

end positive_integer_iff_positive_real_l84_84646


namespace calculate_value_l84_84261

theorem calculate_value :
  ( (3^3 - 1) / (3^3 + 1) ) * ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * ( (7^3 - 1) / (7^3 + 1) )
  = 57 / 84 := by
  sorry

end calculate_value_l84_84261


namespace angle_measure_l84_84103

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l84_84103


namespace directrix_of_parabola_l84_84209

theorem directrix_of_parabola : ∀ (x : ℝ), y = (x^2 - 8*x + 12) / 16 → ∃ (d : ℝ), d = -1/2 := 
sorry

end directrix_of_parabola_l84_84209


namespace solution_set_l84_84151

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set (c1 : ∀ x : ℝ, f x + f' x > 1)
                     (c2 : f 0 = 2) :
  {x : ℝ | e^x * f x > e^x + 1} = {x : ℝ | 0 < x} :=
sorry

end solution_set_l84_84151


namespace lattice_points_condition_l84_84777

/-- A lattice point is a point on the plane with integer coordinates. -/
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

/-- A triangle in the plane with three vertices and at least two lattice points inside. -/
structure Triangle :=
  (A B C : LatticePoint)
  (lattice_points_inside : List LatticePoint)
  (lattice_points_nonempty : lattice_points_inside.length ≥ 2)

noncomputable def exists_lattice_points (T : Triangle) : Prop :=
∃ (X Y : LatticePoint) (hX : X ∈ T.lattice_points_inside) (hY : Y ∈ T.lattice_points_inside), 
  ((∃ (V : LatticePoint), V = T.A ∨ V = T.B ∨ V = T.C ∧ ∃ (k : ℤ), (k : ℝ) * (Y.x - X.x) = (V.x - X.x) ∧ (k : ℝ) * (Y.y - X.y) = (V.y - X.y)) ∨
  (∃ (l m n : ℝ), l * (Y.x - X.x) = m * (T.A.x - T.B.x) ∧ l * (Y.y - X.y) = m * (T.A.y - T.B.y) ∨ l * (Y.x - X.x) = n * (T.B.x - T.C.x) ∧ l * (Y.y - X.y) = n * (T.B.y - T.C.y) ∨ l * (Y.x - X.x) = m * (T.C.x - T.A.x) ∧ l * (Y.y - X.y) = m * (T.C.y - T.A.y)))

theorem lattice_points_condition (T : Triangle) : exists_lattice_points T :=
sorry

end lattice_points_condition_l84_84777


namespace cos_pi_div_4_add_alpha_l84_84444

variable (α : ℝ)

theorem cos_pi_div_4_add_alpha (h : Real.sin (Real.pi / 4 - α) = Real.sqrt 2 / 2) :
  Real.cos (Real.pi / 4 + α) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_pi_div_4_add_alpha_l84_84444


namespace time_spent_on_seals_l84_84571

theorem time_spent_on_seals (s : ℕ) 
  (h1 : 2 * 60 + 10 = 130) 
  (h2 : s + 8 * s + 13 = 130) :
  s = 13 :=
sorry

end time_spent_on_seals_l84_84571


namespace sum_of_areas_of_circles_l84_84802

noncomputable def radius (n : ℕ) : ℝ :=
  3 / 3^n

noncomputable def area (n : ℕ) : ℝ :=
  Real.pi * (radius n)^2

noncomputable def total_area : ℝ :=
  ∑' n, area n

theorem sum_of_areas_of_circles:
  total_area = (9 * Real.pi) / 8 :=
by
  sorry

end sum_of_areas_of_circles_l84_84802


namespace triangle_inequalities_l84_84966

open Real

-- Define a structure for a triangle with its properties
structure Triangle :=
(a b c R ra rb rc : ℝ)

-- Main statement to be proved
theorem triangle_inequalities (Δ : Triangle) (h : 2 * Δ.R ≤ Δ.ra) :
  Δ.a > Δ.b ∧ Δ.a > Δ.c ∧ 2 * Δ.R > Δ.rb ∧ 2 * Δ.R > Δ.rc :=
sorry

end triangle_inequalities_l84_84966


namespace jason_initial_cards_l84_84448

-- Conditions
def cards_given_away : ℕ := 9
def cards_left : ℕ := 4

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 13 :=
by
  sorry

end jason_initial_cards_l84_84448


namespace largest_four_digit_number_l84_84607

def is_four_digit_number (N : ℕ) : Prop := 1000 ≤ N ∧ N ≤ 9999

def sum_of_digits (N : ℕ) : ℕ :=
  let a := N / 1000
  let b := (N % 1000) / 100
  let c := (N % 100) / 10
  let d := N % 10
  a + b + c + d

def is_divisible (N S : ℕ) : Prop := N % S = 0

theorem largest_four_digit_number :
  ∃ N : ℕ, is_four_digit_number N ∧ is_divisible N (sum_of_digits N) ∧
  (∀ M : ℕ, is_four_digit_number M ∧ is_divisible M (sum_of_digits M) → N ≥ M) ∧ N = 9990 :=
by
  sorry

end largest_four_digit_number_l84_84607


namespace circle_area_with_radius_three_is_9pi_l84_84094

theorem circle_area_with_radius_three_is_9pi (r : ℝ) (h : r = 3) : Real.pi * r^2 = 9 * Real.pi := by
  sorry

end circle_area_with_radius_three_is_9pi_l84_84094


namespace sister_weight_difference_is_12_l84_84320

-- Define Antonio's weight
def antonio_weight : ℕ := 50

-- Define the combined weight of Antonio and his sister
def combined_weight : ℕ := 88

-- Define the weight of Antonio's sister
def sister_weight : ℕ := combined_weight - antonio_weight

-- Define the weight difference
def weight_difference : ℕ := antonio_weight - sister_weight

-- Theorem statement to prove the weight difference is 12 kg
theorem sister_weight_difference_is_12 : weight_difference = 12 := by
  sorry

end sister_weight_difference_is_12_l84_84320


namespace krystian_total_books_borrowed_l84_84441

/-
Conditions:
1. Krystian starts on Monday by borrowing 40 books.
2. Each day from Tuesday to Thursday, he borrows 5% more books than he did the previous day.
3. On Friday, his number of borrowed books is 40% higher than on Thursday.
4. During weekends, Krystian borrows books for his friends, and he borrows 2 additional books for every 10 books borrowed during the weekdays.

Theorem: Given these conditions, Krystian borrows a total of 283 books from Monday to Sunday.
-/
theorem krystian_total_books_borrowed : 
  let mon := 40
  let tue := mon + (5 * mon / 100)
  let wed := tue + (5 * tue / 100)
  let thu := wed + (5 * wed / 100)
  let fri := thu + (40 * thu / 100)
  let weekday_total := mon + tue + wed + thu + fri
  let weekend := 2 * (weekday_total / 10)
  weekday_total + weekend = 283 := 
by
  sorry

end krystian_total_books_borrowed_l84_84441


namespace gym_monthly_income_l84_84210

-- Define the conditions
def twice_monthly_charge : ℕ := 18
def monthly_charge_per_member : ℕ := 2 * twice_monthly_charge
def number_of_members : ℕ := 300

-- State the goal: the monthly income of the gym
def monthly_income : ℕ := 36 * 300

-- The theorem to prove
theorem gym_monthly_income : monthly_charge_per_member * number_of_members = 10800 :=
by
  sorry

end gym_monthly_income_l84_84210


namespace length_greater_than_width_l84_84993

theorem length_greater_than_width
  (perimeter : ℕ)
  (P : perimeter = 150)
  (l w difference : ℕ)
  (L : l = 60)
  (W : w = 45)
  (D : difference = l - w) :
  difference = 15 :=
by
  sorry

end length_greater_than_width_l84_84993


namespace central_angle_of_sector_l84_84270

noncomputable def sector_area (α r : ℝ) : ℝ := (1/2) * α * r^2

theorem central_angle_of_sector :
  sector_area 3 2 = 6 :=
by
  unfold sector_area
  norm_num
  done

end central_angle_of_sector_l84_84270


namespace sum_arithmetic_sequence_l84_84497

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_arithmetic_sequence {a : ℕ → ℝ} 
  (h_arith : arithmetic_seq a)
  (h1 : a 2^2 + a 7^2 + 2 * a 2 * a 7 = 9)
  (h2 : ∀ n, a n < 0) : 
  S₁₀ = -15 :=
by
  sorry

end sum_arithmetic_sequence_l84_84497


namespace find_slope_of_intersecting_line_l84_84775

-- Define the conditions
def line_p (x : ℝ) : ℝ := 2 * x + 3
def line_q (x : ℝ) (m : ℝ) : ℝ := m * x + 1

-- Define the point of intersection
def intersection_point : ℝ × ℝ := (4, 11)

-- Prove that the slope m of line q such that both lines intersect at (4, 11) is 2.5
theorem find_slope_of_intersecting_line (m : ℝ) :
  line_q 4 m = 11 → m = 2.5 :=
by
  intro h
  sorry

end find_slope_of_intersecting_line_l84_84775


namespace popped_white_probability_l84_84879

theorem popped_white_probability :
  let P_white := 2 / 3
  let P_yellow := 1 / 3
  let P_pop_given_white := 1 / 2
  let P_pop_given_yellow := 2 / 3

  let P_white_and_pop := P_white * P_pop_given_white
  let P_yellow_and_pop := P_yellow * P_pop_given_yellow
  let P_pop := P_white_and_pop + P_yellow_and_pop

  let P_white_given_pop := P_white_and_pop / P_pop

  P_white_given_pop = 3 / 5 := sorry

end popped_white_probability_l84_84879


namespace equal_division_of_cookie_l84_84159

theorem equal_division_of_cookie (total_area : ℝ) (friends : ℕ) (area_per_person : ℝ) 
  (h1 : total_area = 81.12) 
  (h2 : friends = 6) 
  (h3 : area_per_person = total_area / friends) : 
  area_per_person = 13.52 :=
by 
  sorry

end equal_division_of_cookie_l84_84159


namespace ratio_depends_on_S_and_r_l84_84740

theorem ratio_depends_on_S_and_r
    (S : ℝ) (r : ℝ) (P1 : ℝ) (C2 : ℝ)
    (h1 : P1 = 4 * S)
    (h2 : C2 = 2 * Real.pi * r) :
    (P1 / C2 = 4 * S / (2 * Real.pi * r)) := by
  sorry

end ratio_depends_on_S_and_r_l84_84740


namespace smallest_number_of_coins_l84_84475

theorem smallest_number_of_coins (p n d q h: ℕ) (total: ℕ) 
  (coin_value: ℕ → ℕ)
  (h_p: coin_value 1 = 1) 
  (h_n: coin_value 5 = 5) 
  (h_d: coin_value 10 = 10) 
  (h_q: coin_value 25 = 25) 
  (h_h: coin_value 50 = 50)
  (total_def: total = p * (coin_value 1) + n * (coin_value 5) +
                     d * (coin_value 10) + q * (coin_value 25) + 
                     h * (coin_value 50))
  (h_total: total = 100): 
  p + n + d + q + h = 3 :=
by
  sorry

end smallest_number_of_coins_l84_84475


namespace weavers_in_first_group_l84_84691

theorem weavers_in_first_group :
  (∃ W : ℕ, (W * 4 = 4) ∧ (12 * 12 = 36) ∧ (4 / (W * 4) = 36 / (12 * 12))) -> (W = 4) :=
by
  sorry

end weavers_in_first_group_l84_84691


namespace polynomial_factorization_l84_84954

theorem polynomial_factorization (m n : ℤ) (h₁ : (x + 1) * (x + 3) = x^2 + m * x + n) : m - n = 1 := 
by {
  -- Proof not required
  sorry
}

end polynomial_factorization_l84_84954


namespace find_x_l84_84226

def magic_constant (a b c d e f g h i : ℤ) : Prop :=
  a + b + c = d + e + f ∧ d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧ b + e + h = c + f + i ∧
  a + e + i = c + e + g

def given_magic_square (x : ℤ) : Prop :=
  magic_constant (4017) (2012) (0) 
                 (4015) (x - 2003) (11) 
                 (2014) (9) (x)

theorem find_x (x : ℤ) (h : given_magic_square x) : x = 4003 :=
by {
  sorry
}

end find_x_l84_84226


namespace cylinder_radius_eq_3_l84_84605

theorem cylinder_radius_eq_3 (r : ℝ) : 
  (π * (r + 4)^2 * 3 = π * r^2 * 11) ∧ (r >= 0) → r = 3 :=
by 
  sorry

end cylinder_radius_eq_3_l84_84605


namespace machine_a_produces_6_sprockets_per_hour_l84_84105

theorem machine_a_produces_6_sprockets_per_hour : 
  ∀ (A G T : ℝ), 
  (660 = A * (T + 10)) → 
  (660 = G * T) → 
  (G = 1.10 * A) → 
  A = 6 := 
by
  intros A G T h1 h2 h3
  sorry

end machine_a_produces_6_sprockets_per_hour_l84_84105


namespace directly_proportional_l84_84618

-- Defining conditions
def A (x y : ℝ) : Prop := y = x + 8
def B (x y : ℝ) : Prop := (2 / (5 * y)) = x
def C (x y : ℝ) : Prop := (2 / 3) * x = y

-- Theorem stating that in the given equations, equation C shows direct proportionality
theorem directly_proportional (x y : ℝ) : C x y ↔ (∃ k : ℝ, k ≠ 0 ∧ y = k * x) :=
by
  sorry

end directly_proportional_l84_84618


namespace zoe_pictures_l84_84636

theorem zoe_pictures (P : ℕ) (h1 : P + 16 = 44) : P = 28 :=
by sorry

end zoe_pictures_l84_84636


namespace number_less_than_neg_one_is_neg_two_l84_84325

theorem number_less_than_neg_one_is_neg_two : ∃ x : ℤ, x = -1 - 1 ∧ x = -2 := by
  sorry

end number_less_than_neg_one_is_neg_two_l84_84325


namespace percentage_is_4_l84_84259

-- Define the problem conditions
def percentage_condition (p : ℝ) : Prop := p * 50 = 200

-- State the theorem with the given conditions and the correct answer
theorem percentage_is_4 (p : ℝ) (h : percentage_condition p) : p = 4 := sorry

end percentage_is_4_l84_84259


namespace tan_neg_405_eq_neg_1_l84_84539

theorem tan_neg_405_eq_neg_1 :
  (Real.tan (-405 * Real.pi / 180) = -1) ∧
  (∀ θ : ℝ, Real.tan (θ + 2 * Real.pi) = Real.tan θ) ∧
  (Real.tan θ = Real.sin θ / Real.cos θ) ∧
  (Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2) ∧
  (Real.sin (315 * Real.pi / 180) = - Real.sqrt 2 / 2) :=
sorry

end tan_neg_405_eq_neg_1_l84_84539


namespace find_positive_integer_n_l84_84598

theorem find_positive_integer_n (S : ℕ → ℚ) (hS : ∀ n, S n = n / (n + 1))
  (h : ∃ n : ℕ, S n * S (n + 1) = 3 / 4) : 
  ∃ n : ℕ, n = 6 := 
by {
  sorry
}

end find_positive_integer_n_l84_84598


namespace total_shoes_l84_84482

theorem total_shoes (Brian_shoes : ℕ) (Edward_shoes : ℕ) (Jacob_shoes : ℕ)
  (hBrian : Brian_shoes = 22)
  (hEdward : Edward_shoes = 3 * Brian_shoes)
  (hJacob : Jacob_shoes = Edward_shoes / 2) :
  Brian_shoes + Edward_shoes + Jacob_shoes = 121 :=
by 
  sorry

end total_shoes_l84_84482


namespace greatest_common_divisor_456_108_lt_60_l84_84707

theorem greatest_common_divisor_456_108_lt_60 : 
  let divisors_456 := {d : ℕ | d ∣ 456}
  let divisors_108 := {d : ℕ | d ∣ 108}
  let common_divisors := divisors_456 ∩ divisors_108
  let common_divisors_lt_60 := {d ∈ common_divisors | d < 60}
  ∃ d, d ∈ common_divisors_lt_60 ∧ ∀ e ∈ common_divisors_lt_60, e ≤ d ∧ d = 12 := by {
    sorry
  }

end greatest_common_divisor_456_108_lt_60_l84_84707


namespace no_zero_terms_in_arithmetic_progression_l84_84469

theorem no_zero_terms_in_arithmetic_progression (a d : ℤ) (h : ∃ (n : ℕ), 2 * a + (2 * n - 1) * d = ((3 * n - 1) * (2 * a + (3 * n - 2) * d)) / 2) :
  ∀ (m : ℕ), a + (m - 1) * d ≠ 0 :=
by
  sorry

end no_zero_terms_in_arithmetic_progression_l84_84469


namespace compare_a_b_c_l84_84843

noncomputable def a : ℝ := Real.log (Real.sqrt 2)
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := 1 / Real.exp 1

theorem compare_a_b_c : a < b ∧ b < c := by
  -- Proof will be done here
  sorry

end compare_a_b_c_l84_84843


namespace ellen_total_legos_l84_84687

-- Conditions
def ellen_original_legos : ℝ := 2080.0
def ellen_winning_legos : ℝ := 17.0

-- Theorem statement
theorem ellen_total_legos : ellen_original_legos + ellen_winning_legos = 2097.0 :=
by
  -- The proof would go here, but we will use sorry to indicate it is skipped.
  sorry

end ellen_total_legos_l84_84687


namespace anton_food_cost_l84_84365

def food_cost_julie : ℝ := 10
def food_cost_letitia : ℝ := 20
def tip_per_person : ℝ := 4
def num_people : ℕ := 3
def tip_percentage : ℝ := 0.20

theorem anton_food_cost (A : ℝ) :
  tip_percentage * (food_cost_julie + food_cost_letitia + A) = tip_per_person * num_people →
  A = 30 :=
by
  intro h
  sorry

end anton_food_cost_l84_84365


namespace basic_computer_price_l84_84102

theorem basic_computer_price (C P : ℝ) 
  (h1 : C + P = 2500)
  (h2 : P = 1 / 8 * ((C + 500) + P)) :
  C = 2125 :=
by
  sorry

end basic_computer_price_l84_84102


namespace prove_triangle_inequality_l84_84463

def triangle_inequality (a b c a1 a2 b1 b2 c1 c2 : ℝ) : Prop := 
  a * a1 * a2 + b * b1 * b2 + c * c1 * c2 ≥ a * b * c

theorem prove_triangle_inequality 
  (a b c a1 a2 b1 b2 c1 c2 : ℝ)
  (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c)
  (h4: 0 ≤ a1) (h5: 0 ≤ a2) 
  (h6: 0 ≤ b1) (h7: 0 ≤ b2)
  (h8: 0 ≤ c1) (h9: 0 ≤ c2) : triangle_inequality a b c a1 a2 b1 b2 c1 c2 :=
sorry

end prove_triangle_inequality_l84_84463


namespace problem_subtraction_of_negatives_l84_84182

theorem problem_subtraction_of_negatives :
  12.345 - (-3.256) = 15.601 :=
sorry

end problem_subtraction_of_negatives_l84_84182


namespace larger_number_is_55_l84_84847

theorem larger_number_is_55 (x y : ℤ) (h1 : x + y = 70) (h2 : x = 3 * y + 10) (h3 : y = 15) : x = 55 :=
by
  sorry

end larger_number_is_55_l84_84847


namespace sandy_grew_6_carrots_l84_84551

theorem sandy_grew_6_carrots (sam_grew : ℕ) (total_grew : ℕ) (h1 : sam_grew = 3) (h2 : total_grew = 9) : ∃ sandy_grew : ℕ, sandy_grew = total_grew - sam_grew ∧ sandy_grew = 6 :=
by
  sorry

end sandy_grew_6_carrots_l84_84551


namespace cyclic_inequality_l84_84785

theorem cyclic_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (x + y) * Real.sqrt (y + z) * Real.sqrt (z + x) + (y + z) * Real.sqrt (z + x) * Real.sqrt (x + y) + (z + x) * Real.sqrt (x + y) * Real.sqrt (y + z) ≥ 4 * (x * y + y * z + z * x) :=
by
  sorry

end cyclic_inequality_l84_84785


namespace min_value_x_plus_4y_l84_84202

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 2 * x * y) : x + 4 * y = 9 / 2 :=
by
  sorry

end min_value_x_plus_4y_l84_84202


namespace students_count_l84_84882

theorem students_count (x y : ℕ) (h1 : 3 * x + 20 = y) (h2 : 4 * x - 25 = y) : x = 45 :=
by {
  sorry
}

end students_count_l84_84882


namespace find_k_from_inequality_l84_84656

variable (k x : ℝ)

theorem find_k_from_inequality (h : ∀ x ∈ Set.Ico (-2 : ℝ) 1, 1 + k / (x - 1) ≤ 0)
  (h₂: 1 + k / (-2 - 1) = 0) :
  k = 3 :=
by
  sorry

end find_k_from_inequality_l84_84656


namespace geometric_progression_common_ratio_l84_84430

theorem geometric_progression_common_ratio (a r : ℝ) (h_pos : a > 0)
  (h_eq : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) :
  r = 1/2 :=
sorry

end geometric_progression_common_ratio_l84_84430


namespace harmonic_progression_l84_84917

theorem harmonic_progression (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
(h_harm : 1 / (a : ℝ) + 1 / (c : ℝ) = 2 / (b : ℝ))
(h_div : c % b = 0)
(h_inc : a < b ∧ b < c) :
  a = 20 → 
  (b, c) = (30, 60) ∨ (b, c) = (35, 140) ∨ (b, c) = (36, 180) ∨ (b, c) = (38, 380) ∨ (b, c) = (39, 780) :=
by sorry

end harmonic_progression_l84_84917


namespace john_paintball_times_l84_84085

theorem john_paintball_times (x : ℕ) (cost_per_box : ℕ) (boxes_per_play : ℕ) (monthly_spending : ℕ) :
  (cost_per_box = 25) → (boxes_per_play = 3) → (monthly_spending = 225) → (boxes_per_play * cost_per_box * x = monthly_spending) → x = 3 :=
by
  intros h1 h2 h3 h4
  -- proof would go here
  sorry

end john_paintball_times_l84_84085


namespace polynomial_solution_characterization_l84_84144

theorem polynomial_solution_characterization (P : ℝ → ℝ → ℝ) (h : ∀ x y z : ℝ, P x (2 * y * z) + P y (2 * z * x) + P z (2 * x * y) = P (x + y + z) (x * y + y * z + z * x)) :
  ∃ (a b : ℝ), ∀ x y : ℝ, P x y = a * x + b * (x^2 + 2 * y) :=
sorry

end polynomial_solution_characterization_l84_84144


namespace opposite_of_neg_two_l84_84732

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l84_84732


namespace area_of_triangle_ABC_equation_of_circumcircle_l84_84895

-- Define points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := 1, y := 3 }
def C : Point := { x := 3, y := 6 }

-- Theorem to prove the area of triangle ABC
theorem area_of_triangle_ABC : 
  let base := |B.y - A.y|
  let height := |C.x - A.x|
  (1/2) * base * height = 1 := sorry

-- Theorem to prove the equation of the circumcircle of triangle ABC
theorem equation_of_circumcircle : 
  let D := -10
  let E := -5
  let F := 15
  ∀ (x y : ℝ), (x - 5)^2 + (y - 5/2)^2 = 65/4 ↔ 
                x^2 + y^2 + D * x + E * y + F = 0 := sorry

end area_of_triangle_ABC_equation_of_circumcircle_l84_84895


namespace quadratic_inequality_empty_solution_range_l84_84003

theorem quadratic_inequality_empty_solution_range (b : ℝ) :
  (∀ x : ℝ, ¬ (x^2 + b * x + 1 ≤ 0)) ↔ -2 < b ∧ b < 2 :=
by
  sorry

end quadratic_inequality_empty_solution_range_l84_84003


namespace part1_part2_l84_84637

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2
noncomputable def g (x a : ℝ) : ℝ := x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x

theorem part1 : ∃ xₘ : ℝ, (∀ x > 0, f x ≤ f xₘ) ∧ f xₘ = -1 :=
by sorry

theorem part2 (a : ℝ) : (∀ x > 0, f x + g x a ≥ 0) ↔ a ≤ 1 :=
by sorry

end part1_part2_l84_84637


namespace betty_honey_oats_problem_l84_84774

theorem betty_honey_oats_problem
  (o h : ℝ)
  (h_condition1 : o ≥ 8 + h / 3)
  (h_condition2 : o ≤ 3 * h) :
  h ≥ 3 :=
sorry

end betty_honey_oats_problem_l84_84774


namespace trail_length_proof_l84_84082

theorem trail_length_proof (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 + x2 = 28)
  (h2 : x2 + x3 = 30)
  (h3 : x3 + x4 + x5 = 42)
  (h4 : x1 + x4 = 30) :
  x1 + x2 + x3 + x4 + x5 = 70 := by
  sorry

end trail_length_proof_l84_84082


namespace solve_for_x_values_for_matrix_l84_84833

def matrix_equals_neg_two (x : ℝ) : Prop :=
  let a := 3 * x
  let b := x
  let c := 4
  let d := 2 * x
  (a * b - c * d = -2)

theorem solve_for_x_values_for_matrix : 
  ∃ (x : ℝ), matrix_equals_neg_two x ↔ (x = (4 + Real.sqrt 10) / 3 ∨ x = (4 - Real.sqrt 10) / 3) :=
sorry

end solve_for_x_values_for_matrix_l84_84833


namespace tangent_line_circle_l84_84427

theorem tangent_line_circle : 
  ∃ (k : ℚ), (∀ x y : ℚ, ((x - 3) ^ 2 + (y - 4) ^ 2 = 25) 
               → (3 * x + 4 * y - 25 = 0)) :=
sorry

end tangent_line_circle_l84_84427


namespace manager_salary_correct_l84_84523

-- Define the conditions of the problem
def total_salary_of_24_employees : ℕ := 24 * 2400
def new_average_salary_with_manager : ℕ := 2500
def number_of_people_with_manager : ℕ := 25

-- Define the manager's salary to be proved
def managers_salary : ℕ := 4900

-- Statement of the theorem to prove that the manager's salary is Rs. 4900
theorem manager_salary_correct :
  (number_of_people_with_manager * new_average_salary_with_manager) - total_salary_of_24_employees = managers_salary :=
by
  -- Proof to be filled
  sorry

end manager_salary_correct_l84_84523


namespace incorrect_statement_l84_84130

noncomputable def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem incorrect_statement
  (a b c : ℤ) (h₀ : a ≠ 0)
  (h₁ : 2 * a + b = 0)
  (h₂ : f a b c 1 = 3)
  (h₃ : f a b c 2 = 8) :
  ¬ (f a b c (-1) = 0) :=
sorry

end incorrect_statement_l84_84130


namespace negation_of_p_l84_84081

-- Define the proposition p
def p : Prop := ∀ x : ℝ, Real.exp x > Real.log x

-- Define the negation of p
def neg_p : Prop := ∃ x : ℝ, Real.exp x ≤ Real.log x

-- The statement we want to prove
theorem negation_of_p : ¬p ↔ neg_p :=
by sorry

end negation_of_p_l84_84081


namespace square_value_l84_84937

theorem square_value {square : ℚ} (h : 8 / 12 = square / 3) : square = 2 :=
sorry

end square_value_l84_84937


namespace min_value_of_reciprocal_sum_l84_84780

variable (m n : ℝ)

theorem min_value_of_reciprocal_sum (hmn : m * n > 0) (h_line : m + n = 2) :
  (1 / m + 1 / n = 2) :=
sorry

end min_value_of_reciprocal_sum_l84_84780


namespace range_of_a_l84_84418

noncomputable def equation_has_two_roots (a m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ + a * (2 * x₁ + 2 * m - 4 * Real.exp 1 * x₁) * (Real.log (x₁ + m) - Real.log x₁) = 0 ∧ 
    x₂ + a * (2 * x₂ + 2 * m - 4 * Real.exp 1 * x₂) * (Real.log (x₂ + m) - Real.log x₂) = 0

theorem range_of_a (m : ℝ) (hm : 0 < m) : 
  (∃ a, equation_has_two_roots a m) ↔ (a < 0 ∨ a > 1 / (2 * Real.exp 1)) := 
sorry

end range_of_a_l84_84418


namespace avg_monthly_bill_over_6_months_l84_84229

theorem avg_monthly_bill_over_6_months :
  ∀ (avg_first_4_months avg_last_2_months : ℝ), 
  avg_first_4_months = 30 → 
  avg_last_2_months = 24 → 
  (4 * avg_first_4_months + 2 * avg_last_2_months) / 6 = 28 :=
by
  intros
  sorry

end avg_monthly_bill_over_6_months_l84_84229


namespace sampling_methods_correct_l84_84114

-- Define the conditions given in the problem.
def total_students := 200
def method_1_is_simple_random := true
def method_2_is_systematic := true

-- The proof problem statement, no proof is required.
theorem sampling_methods_correct :
  (method_1_is_simple_random = true) ∧
  (method_2_is_systematic = true) :=
by
  -- using conditions defined above, we state the theorem we need to prove
  sorry

end sampling_methods_correct_l84_84114


namespace equal_real_roots_eq_one_l84_84453

theorem equal_real_roots_eq_one (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y * y = x) ∧ (∀ x y : ℝ, x^2 - 2 * x + m = 0 ↔ (x = y) → b^2 - 4 * a * c = 0) → m = 1 := 
sorry

end equal_real_roots_eq_one_l84_84453


namespace total_spent_is_64_l84_84766

/-- Condition 1: The cost of each deck is 8 dollars -/
def deck_cost : ℕ := 8

/-- Condition 2: Tom bought 3 decks -/
def tom_decks : ℕ := 3

/-- Condition 3: Tom's friend bought 5 decks -/
def friend_decks : ℕ := 5

/-- Total amount spent by Tom and his friend -/
def total_amount_spent : ℕ := (tom_decks * deck_cost) + (friend_decks * deck_cost)

/-- Proof statement: Prove that total amount spent is 64 -/
theorem total_spent_is_64 : total_amount_spent = 64 := by
  sorry

end total_spent_is_64_l84_84766


namespace minimize_abs_a_n_l84_84051

noncomputable def a_n (n : ℕ) : ℝ :=
  14 - (3 / 4) * (n - 1)

theorem minimize_abs_a_n : ∃ n : ℕ, n = 20 ∧ ∀ m : ℕ, |a_n n| ≤ |a_n m| := by
  sorry

end minimize_abs_a_n_l84_84051


namespace max_area_BPC_l84_84870

noncomputable def triangle_area_max (AB BC CA : ℝ) (D : ℝ) : ℝ :=
  if h₁ : AB = 13 ∧ BC = 15 ∧ CA = 14 then
    112.5 - 56.25 * Real.sqrt 3
  else 0

theorem max_area_BPC : triangle_area_max 13 15 14 D = 112.5 - 56.25 * Real.sqrt 3 := by
  sorry

end max_area_BPC_l84_84870


namespace fourth_quadrangle_area_l84_84958

theorem fourth_quadrangle_area (S1 S2 S3 S4 : ℝ) (h : S1 + S4 = S2 + S3) : S4 = S2 + S3 - S1 :=
by
  sorry

end fourth_quadrangle_area_l84_84958


namespace binary_div_remainder_l84_84495

theorem binary_div_remainder (n : ℕ) (h : n = 0b101011100101) : n % 8 = 5 :=
by sorry

end binary_div_remainder_l84_84495


namespace algebraic_expression_correct_l84_84304

variable (x y : ℤ)

theorem algebraic_expression_correct (h : (x - y) / (x + y) = 3) : (2 * (x - y)) / (x + y) - (x + y) / (3 * (x - y)) = 53 / 9 := 
by  
  sorry

end algebraic_expression_correct_l84_84304


namespace thomas_annual_insurance_cost_l84_84080

theorem thomas_annual_insurance_cost (total_cost : ℕ) (number_of_years : ℕ) 
  (h1 : total_cost = 40000) (h2 : number_of_years = 10) : 
  total_cost / number_of_years = 4000 := 
by 
  sorry

end thomas_annual_insurance_cost_l84_84080


namespace total_distance_traveled_l84_84374

/-- Defining the distance Greg travels in each leg of his trip -/
def distance_workplace_to_market : ℕ := 30

def distance_market_to_friend : ℕ := distance_workplace_to_market + 10

def distance_friend_to_aunt : ℕ := 5

def distance_aunt_to_grocery : ℕ := 7

def distance_grocery_to_home : ℕ := 18

/-- The total distance Greg traveled during his entire trip is the sum of all individual distances -/
theorem total_distance_traveled :
  distance_workplace_to_market + distance_market_to_friend + distance_friend_to_aunt + distance_aunt_to_grocery + distance_grocery_to_home = 100 :=
by
  sorry

end total_distance_traveled_l84_84374


namespace common_ratio_geometric_series_l84_84433

theorem common_ratio_geometric_series :
  let a := 2 / 3
  let b := 4 / 9
  let c := 8 / 27
  (b / a = 2 / 3) ∧ (c / b = 2 / 3) → 
  ∃ r : ℚ, r = 2 / 3 ∧ ∀ n : ℕ, (a * r^n) = (a * (2 / 3)^n) :=
by
  sorry

end common_ratio_geometric_series_l84_84433


namespace percentage_of_boys_answered_neither_l84_84595

theorem percentage_of_boys_answered_neither (P_A P_B P_A_and_B : ℝ) (hP_A : P_A = 0.75) (hP_B : P_B = 0.55) (hP_A_and_B : P_A_and_B = 0.50) :
  1 - (P_A + P_B - P_A_and_B) = 0.20 :=
by
  sorry

end percentage_of_boys_answered_neither_l84_84595


namespace completion_time_l84_84652

variables {P E : ℝ}
theorem completion_time (h1 : (20 : ℝ) * P * E / 2 = D * (2.5 * P * E)) : D = 4 :=
by
  -- Given h1 as the condition
  sorry

end completion_time_l84_84652


namespace solution_proof_l84_84305

noncomputable def f (n : ℕ) : ℝ := Real.logb 143 (n^2)

theorem solution_proof : f 7 + f 11 + f 13 = 2 + 2 * Real.logb 143 7 := by
  sorry

end solution_proof_l84_84305


namespace total_hours_charged_l84_84196

variable (K P M : ℕ)

theorem total_hours_charged (h1 : P = 2 * K) (h2 : P = M / 3) (h3 : M = K + 80) : K + P + M = 144 := 
by
  sorry

end total_hours_charged_l84_84196


namespace Tyler_scissors_count_l84_84137

variable (S : ℕ)

def Tyler_initial_money : ℕ := 100
def cost_per_scissors : ℕ := 5
def number_of_erasers : ℕ := 10
def cost_per_eraser : ℕ := 4
def Tyler_remaining_money : ℕ := 20

theorem Tyler_scissors_count :
  Tyler_initial_money - (cost_per_scissors * S + number_of_erasers * cost_per_eraser) = Tyler_remaining_money →
  S = 8 :=
by
  sorry

end Tyler_scissors_count_l84_84137


namespace bridget_apples_l84_84975

variable (x : ℕ)

-- Conditions as definitions
def apples_after_splitting : ℕ := x / 2
def apples_after_giving_to_cassie : ℕ := apples_after_splitting x - 5
def apples_after_finding_hidden : ℕ := apples_after_giving_to_cassie x + 2
def final_apples : ℕ := apples_after_finding_hidden x
def bridget_keeps : ℕ := 6

-- Proof statement
theorem bridget_apples : x / 2 - 5 + 2 = bridget_keeps → x = 18 := by
  intros h
  sorry

end bridget_apples_l84_84975


namespace ellipse_equation_l84_84675

theorem ellipse_equation (e : ℝ) (P : ℝ × ℝ) (d_max : ℝ) (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
    (h3 : e = Real.sqrt 3 / 2) (h4 : P = (0, 3 / 2)) (h5 : ∀ P1 : ℝ × ℝ, (P1.1 ^ 2 / a ^ 2 + P1.2 ^ 2 / b ^ 2 = 1) → 
    ∃ P2 : ℝ × ℝ, dist P P2 = d_max ∧ (P2.1 ^ 2 / a ^ 2 + P2.2 ^ 2 / b ^ 2 = 1)) :
  (a = 2 ∧ b = 1) → (∀ x y : ℝ, (x ^ 2 / 4) + y ^ 2 ≤ 1) := by
  sorry

end ellipse_equation_l84_84675


namespace initial_games_l84_84822

theorem initial_games (X : ℕ) (h1 : X + 31 - 105 = 6) : X = 80 :=
by
  sorry

end initial_games_l84_84822


namespace premium_rate_l84_84480

theorem premium_rate (P : ℝ) : (14400 / (100 + P)) * 5 = 600 → P = 20 :=
by
  intro h
  sorry

end premium_rate_l84_84480


namespace angle_negative_225_in_second_quadrant_l84_84377

def inSecondQuadrant (angle : Int) : Prop :=
  angle % 360 > -270 ∧ angle % 360 <= -180

theorem angle_negative_225_in_second_quadrant :
  inSecondQuadrant (-225) :=
by
  sorry

end angle_negative_225_in_second_quadrant_l84_84377


namespace find_number_l84_84531

theorem find_number : ∃ x : ℝ, 3550 - (1002 / x) = 3500 ∧ x = 20.04 :=
by
  sorry

end find_number_l84_84531


namespace possible_values_of_a_l84_84228

variable (a : ℝ)
def A : Set ℝ := { x | x^2 ≠ 1 }
def B (a : ℝ) : Set ℝ := { x | a * x = 1 }

theorem possible_values_of_a (h : (A ∪ B a) = A) : a = 1 ∨ a = -1 ∨ a = 0 :=
by
  sorry

end possible_values_of_a_l84_84228


namespace M_subset_P_l84_84127

def M := {x : ℕ | ∃ a : ℕ, 0 < a ∧ x = a^2 + 1}
def P := {y : ℕ | ∃ b : ℕ, 0 < b ∧ y = b^2 - 4*b + 5}

theorem M_subset_P : M ⊂ P :=
by
  sorry

end M_subset_P_l84_84127


namespace avg_age_of_women_l84_84420

theorem avg_age_of_women (T : ℕ) (W : ℕ) (T_avg : ℕ) (H1 : T_avg = T / 10)
  (H2 : (T_avg + 6) = ((T - 18 - 22 + W) / 10)) : (W / 2) = 50 :=
sorry

end avg_age_of_women_l84_84420


namespace negation_of_proposition_l84_84340

-- Definitions of the conditions
variables (a b c : ℝ) 

-- Prove the mathematically equivalent statement:
theorem negation_of_proposition :
  (a + b + c ≠ 1) → (a^2 + b^2 + c^2 > 1 / 9) :=
sorry

end negation_of_proposition_l84_84340


namespace horner_eval_at_2_l84_84549

def poly (x : ℝ) : ℝ := 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem horner_eval_at_2 : poly 2 = 373 := by
  sorry

end horner_eval_at_2_l84_84549


namespace total_votes_l84_84241

-- Conditions
variables (V : ℝ)
def candidate_votes := 0.31 * V
def rival_votes := 0.31 * V + 2451

-- Problem statement
theorem total_votes (h : candidate_votes V + rival_votes V = V) : V = 6450 :=
sorry

end total_votes_l84_84241


namespace unit_price_in_range_l84_84681

-- Given definitions and conditions
def Q (x : ℝ) : ℝ := 220 - 2 * x
def f (x : ℝ) : ℝ := x * Q x

-- The desired range for the unit price to maintain a production value of at least 60 million yuan
def valid_unit_price_range (x : ℝ) : Prop := 50 < x ∧ x < 60

-- The main theorem that needs to be proven
theorem unit_price_in_range (x : ℝ) (h₁ : 0 < x) (h₂ : x < 500) (h₃ : f x ≥ 60 * 10^6) : valid_unit_price_range x :=
sorry

end unit_price_in_range_l84_84681


namespace transmission_prob_correct_transmission_scheme_comparison_l84_84671

noncomputable def transmission_prob_single (α β : ℝ) : ℝ :=
  (1 - α) * (1 - β)^2

noncomputable def transmission_prob_triple_sequence (β : ℝ) : ℝ :=
  β * (1 - β)^2

noncomputable def transmission_prob_triple_decoding_one (β : ℝ) : ℝ :=
  β * (1 - β)^2 + (1 - β)^3

noncomputable def transmission_prob_triple_decoding_zero (α : ℝ) : ℝ :=
  3 * α * (1 - α)^2 + (1 - α)^3

noncomputable def transmission_prob_single_decoding_zero (α : ℝ) : ℝ :=
  1 - α

theorem transmission_prob_correct (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  transmission_prob_single α β = (1 - α) * (1 - β)^2 ∧
  transmission_prob_triple_sequence β = β * (1 - β)^2 ∧
  transmission_prob_triple_decoding_one β = β * (1 - β)^2 + (1 - β)^3 :=
sorry

theorem transmission_scheme_comparison (α : ℝ) (hα : 0 < α ∧ α < 0.5) :
  transmission_prob_triple_decoding_zero α > transmission_prob_single_decoding_zero α :=
sorry

end transmission_prob_correct_transmission_scheme_comparison_l84_84671


namespace hyperbola_asymptote_focal_length_l84_84723

theorem hyperbola_asymptote_focal_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : c = 2 * Real.sqrt 5) (h4 : b / a = 2) : a = 2 :=
by
  sorry

end hyperbola_asymptote_focal_length_l84_84723


namespace water_hyacinth_indicates_connection_l84_84188

-- Definitions based on the conditions
def universally_interconnected : Prop := 
  ∀ (a b : Type), a ≠ b → ∃ (c : Type), (a ≠ c) ∧ (b ≠ c)

def connections_diverse : Prop := 
  ∀ (a b : Type), a ≠ b → ∃ (f : a → b), ∀ (x y : a), x ≠ y → f x ≠ f y

def connections_created : Prop :=
  ∃ (a b : Type), a ≠ b ∧ (∀ (f : a → b), False)

def connections_humanized : Prop :=
  ∀ (a b : Type), a ≠ b → (∃ c : Type, a = c) ∧ (∃ d : Type, b = d)

-- Problem statement
theorem water_hyacinth_indicates_connection : 
  universally_interconnected ∧ connections_diverse :=
by
  sorry

end water_hyacinth_indicates_connection_l84_84188


namespace find_q_l84_84248

noncomputable def p : ℝ := -(5 / 6)
noncomputable def g (x : ℝ) : ℝ := p * x^2 + (5 / 6) * x + 5

theorem find_q :
  (∀ x : ℝ, g x = p * x^2 + q * x + r) ∧ 
  (g (-2) = 0) ∧ 
  (g 3 = 0) ∧ 
  (g 1 = 5) 
  → q = 5 / 6 :=
sorry

end find_q_l84_84248


namespace largest_and_smallest_multiples_of_12_l84_84022

theorem largest_and_smallest_multiples_of_12 (k : ℤ) (n₁ n₂ : ℤ) (h₁ : k = -150) (h₂ : n₁ = -156) (h₃ : n₂ = -144) :
  (∃ m1 : ℤ, m1 * 12 = n₁ ∧ n₁ < k) ∧ (¬ (∃ m2 : ℤ, m2 * 12 = n₂ ∧ n₂ > k ∧ ∃ m2' : ℤ, m2' * 12 > k ∧ m2' * 12 < n₂)) :=
by
  sorry

end largest_and_smallest_multiples_of_12_l84_84022


namespace digit_7_count_correct_l84_84393

def base8ToBase10 (n : Nat) : Nat :=
  -- converting base 8 number 1000 to base 10
  1 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0

def countDigit7 (n : Nat) : Nat :=
  -- counts the number of times the digit '7' appears in numbers from 1 to n
  let digits := (List.range (n + 1)).map fun x => x.digits 10
  digits.foldl (fun acc ds => acc + ds.count 7) 0

theorem digit_7_count_correct : countDigit7 512 = 123 := by
  sorry

end digit_7_count_correct_l84_84393


namespace no_four_distinct_real_roots_l84_84317

theorem no_four_distinct_real_roots (a b : ℝ) : ¬ (∃ (x1 x2 x3 x4 : ℝ), 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
  (x1^4 - 4*x1^3 + 6*x1^2 + a*x1 + b = 0) ∧ 
  (x2^4 - 4*x2^3 + 6*x2^2 + a*x2 + b = 0) ∧ 
  (x3^4 - 4*x3^3 + 6*x3^2 + a*x3 + b = 0) ∧ 
  (x4^4 - 4*x4^3 + 6*x4^2 + a*x4 + b = 0)) :=
by
  sorry

end no_four_distinct_real_roots_l84_84317


namespace no_solution_inequalities_l84_84460

theorem no_solution_inequalities (m : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 < 3) → (x > m) → false) ↔ (m ≥ 2) :=
by 
  sorry

end no_solution_inequalities_l84_84460


namespace y_minus_x_is_7_l84_84659

theorem y_minus_x_is_7 (x y : ℕ) (hx : x ≠ y) (h1 : 3 + y = 10) (h2 : 0 + x + 1 = 1) (h3 : 3 + 7 = 10) :
  y - x = 7 :=
by
  sorry

end y_minus_x_is_7_l84_84659


namespace sin_C_in_right_triangle_l84_84416

theorem sin_C_in_right_triangle
  (A B C : ℝ)
  (sin_A : ℝ)
  (sin_B : ℝ)
  (B_right_angle : B = π / 2)
  (sin_A_value : sin_A = 3 / 5)
  (sin_B_value : sin_B = 1)
  (sin_of_C : ℝ)
  (tri_ABC : A + B + C = π ∧ A > 0 ∧ C > 0) :
    sin_of_C = 4 / 5 :=
by
  -- Skipping the proof
  sorry

end sin_C_in_right_triangle_l84_84416


namespace no_solution_for_lcm_gcd_eq_l84_84554

theorem no_solution_for_lcm_gcd_eq (n : ℕ) (h₁ : n ∣ 60) (h₂ : Nat.Prime n) :
  ¬(Nat.lcm n 60 = Nat.gcd n 60 + 200) :=
  sorry

end no_solution_for_lcm_gcd_eq_l84_84554


namespace sum_of_fractions_l84_84486

theorem sum_of_fractions :
  (7:ℚ) / 12 + (11:ℚ) / 15 = 79 / 60 :=
by
  sorry

end sum_of_fractions_l84_84486


namespace symmetric_points_x_axis_l84_84982

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) 
  (hP : P = (a - 3, 1)) (hQ : Q = (2, b + 1)) (hSymm : P.1 = Q.1 ∧ P.2 = -Q.2) :
  a + b = 3 :=
by 
  sorry

end symmetric_points_x_axis_l84_84982


namespace B_completion_time_l84_84459

theorem B_completion_time (A_days : ℕ) (A_efficiency_multiple : ℝ) (B_days_correct : ℝ) :
  A_days = 15 →
  A_efficiency_multiple = 1.8 →
  B_days_correct = 4 + 1 / 6 →
  B_days_correct = 25 / 6 :=
sorry

end B_completion_time_l84_84459


namespace cubes_squares_problem_l84_84046

theorem cubes_squares_problem (h1 : 2^3 - 7^2 = 1) (h2 : 3^3 - 6^2 = 9) (h3 : 5^3 - 9^2 = 16) : 4^3 - 8^2 = 0 := 
by
  sorry

end cubes_squares_problem_l84_84046


namespace sally_balloons_l84_84066

theorem sally_balloons (F S : ℕ) (h1 : F = 3 * S) (h2 : F = 18) : S = 6 :=
by sorry

end sally_balloons_l84_84066


namespace necessary_but_not_sufficient_l84_84805

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 - 3 * x - 4 = 0) -> (x = 4 ∨ x = -1) ∧ ¬(x = 4 ∨ x = -1 -> x = 4) :=
by sorry

end necessary_but_not_sufficient_l84_84805


namespace chemist_salt_solution_l84_84638

theorem chemist_salt_solution (x : ℝ) 
  (hx : 0.60 * x = 0.20 * (1 + x)) : x = 0.5 :=
sorry

end chemist_salt_solution_l84_84638


namespace second_derivative_parametric_l84_84062

noncomputable def x (t : ℝ) := Real.sqrt (t - 1)
noncomputable def y (t : ℝ) := 1 / Real.sqrt t

noncomputable def y_xx (t : ℝ) := (2 * t - 3) * Real.sqrt t / t^3

theorem second_derivative_parametric :
  ∀ t, y_xx t = (2 * t - 3) * Real.sqrt t / t^3 := sorry

end second_derivative_parametric_l84_84062


namespace problem_statement_l84_84155

theorem problem_statement (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) = x * f y + y * f x) →
  (∀ x : ℝ, x > 1 → f x < 0) →

  -- Conclusion 1: f(1) = 0, f(-1) = 0
  f 1 = 0 ∧ f (-1) = 0 ∧

  -- Conclusion 2: f(x) is an odd function: f(-x) = -f(x)
  (∀ x : ℝ, f (-x) = -f x) ∧

  -- Conclusion 3: f(x) is decreasing on (1, +∞)
  (∀ x1 x2 : ℝ, x1 > 1 → x2 > 1 → x1 < x2 → f x1 < f x2) := sorry

end problem_statement_l84_84155


namespace minimum_value_function_inequality_ln_l84_84518

noncomputable def f (x : ℝ) := x * Real.log x

theorem minimum_value_function (t : ℝ) (ht : 0 < t) :
  ∃ (xmin : ℝ), xmin = if (0 < t ∧ t < 1 / Real.exp 1) then -1 / Real.exp 1 else t * Real.log t :=
sorry

theorem inequality_ln (x : ℝ) (hx : 0 < x) : 
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) :=
sorry

end minimum_value_function_inequality_ln_l84_84518


namespace alice_prob_after_three_turns_l84_84357

/-
Definition of conditions:
 - Alice starts with the ball.
 - If Alice has the ball, there is a 1/3 chance that she will toss it to Bob and a 2/3 chance that she will keep the ball.
 - If Bob has the ball, there is a 1/4 chance that he will toss it to Alice and a 3/4 chance that he keeps the ball.
-/

def alice_to_bob : ℚ := 1/3
def alice_keeps : ℚ := 2/3
def bob_to_alice : ℚ := 1/4
def bob_keeps : ℚ := 3/4

theorem alice_prob_after_three_turns :
  alice_to_bob * bob_keeps * bob_to_alice +
  alice_keeps * alice_keeps * alice_keeps +
  alice_to_bob * bob_to_alice * alice_keeps = 179/432 :=
by
  sorry

end alice_prob_after_three_turns_l84_84357


namespace common_ratio_of_geometric_series_l84_84736

-- Definitions of the first two terms of the geometric series
def term1 : ℚ := 4 / 7
def term2 : ℚ := -8 / 3

-- Theorem to prove the common ratio
theorem common_ratio_of_geometric_series : (term2 / term1 = -14 / 3) := by
  sorry

end common_ratio_of_geometric_series_l84_84736


namespace david_marks_in_physics_l84_84064

theorem david_marks_in_physics
  (marks_english : ℤ)
  (marks_math : ℤ)
  (marks_chemistry : ℤ)
  (marks_biology : ℤ)
  (average_marks : ℚ)
  (number_of_subjects : ℤ)
  (h_english : marks_english = 96)
  (h_math : marks_math = 98)
  (h_chemistry : marks_chemistry = 100)
  (h_biology : marks_biology = 98)
  (h_average : average_marks = 98.2)
  (h_subjects : number_of_subjects = 5) : 
  ∃ (marks_physics : ℤ), marks_physics = 99 := 
by {
  sorry
}

end david_marks_in_physics_l84_84064


namespace length_of_cloth_l84_84132

theorem length_of_cloth (L : ℝ) (h : 35 = (L + 4) * (35 / L - 1)) : L = 10 :=
sorry

end length_of_cloth_l84_84132


namespace solution_set_of_inequality_l84_84680

theorem solution_set_of_inequality :
  { x : ℝ | 2 / (x - 1) ≥ 1 } = { x : ℝ | 1 < x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l84_84680


namespace probability_of_B_winning_is_correct_l84_84511

noncomputable def prob_A_wins : ℝ := 0.2
noncomputable def prob_draw : ℝ := 0.5
noncomputable def prob_B_wins : ℝ := 1 - (prob_A_wins + prob_draw)

theorem probability_of_B_winning_is_correct : prob_B_wins = 0.3 := by
  sorry

end probability_of_B_winning_is_correct_l84_84511


namespace frank_remaining_money_l84_84943

noncomputable def cheapest_lamp_cost : ℝ := 20
noncomputable def most_expensive_lamp_cost : ℝ := 3 * cheapest_lamp_cost
noncomputable def frank_initial_money : ℝ := 90

theorem frank_remaining_money : frank_initial_money - most_expensive_lamp_cost = 30 := by
  -- Proof will go here
  sorry

end frank_remaining_money_l84_84943


namespace total_payment_correct_l84_84894

def cost (n : ℕ) : ℕ :=
  if n <= 10 then n * 25
  else 10 * 25 + (n - 10) * (4 * 25 / 5)

def final_cost_with_discount (n : ℕ) : ℕ :=
  let initial_cost := cost n
  if n > 20 then initial_cost - initial_cost / 10
  else initial_cost

def orders_X := 60 * 20 / 100
def orders_Y := 60 * 25 / 100
def orders_Z := 60 * 55 / 100

def cost_X := final_cost_with_discount orders_X
def cost_Y := final_cost_with_discount orders_Y
def cost_Z := final_cost_with_discount orders_Z

theorem total_payment_correct : cost_X + cost_Y + cost_Z = 1279 := by
  sorry

end total_payment_correct_l84_84894


namespace possible_values_of_p1_l84_84935

noncomputable def p (x : ℝ) (n : ℕ) : ℝ := sorry

axiom deg_p (n : ℕ) (h : n ≥ 2) (x : ℝ) : x^n = 1

axiom roots_le_one (r : ℝ) : r ≤ 1

axiom p_at_2 (n : ℕ) (h : n ≥ 2) : p 2 n = 3^n

theorem possible_values_of_p1 (n : ℕ) (h : n ≥ 2) : p 1 n = 0 ∨ p 1 n = (-1)^n * 2^n :=
by
  sorry

end possible_values_of_p1_l84_84935


namespace calculate_paintable_area_l84_84330

noncomputable def bedroom_length : ℝ := 15
noncomputable def bedroom_width : ℝ := 11
noncomputable def bedroom_height : ℝ := 9
noncomputable def door_window_area : ℝ := 70
noncomputable def num_bedrooms : ℝ := 3

theorem calculate_paintable_area :
  (num_bedrooms * ((2 * bedroom_length * bedroom_height) + (2 * bedroom_width * bedroom_height) - door_window_area)) = 1194 := 
by
  -- conditions as definitions
  let total_wall_area := (2 * bedroom_length * bedroom_height) + (2 * bedroom_width * bedroom_height)
  let paintable_wall_in_bedroom := total_wall_area - door_window_area
  let total_paintable_area := num_bedrooms * paintable_wall_in_bedroom
  show total_paintable_area = 1194
  sorry

end calculate_paintable_area_l84_84330


namespace maximum_distance_l84_84753

-- Definitions from the conditions
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def distance_driven : ℝ := 244
def gallons_used : ℝ := 20

-- Problem statement
theorem maximum_distance (h: (distance_driven / gallons_used = highway_mpg)): 
  (distance_driven = 244) :=
sorry

end maximum_distance_l84_84753


namespace e_n_max_value_l84_84223

def b (n : ℕ) : ℕ := (5^n - 1) / 4

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem e_n_max_value (n : ℕ) : e n = 1 := 
by sorry

end e_n_max_value_l84_84223


namespace log_sum_reciprocals_of_logs_l84_84313

-- Problem (1)
theorem log_sum (log_two : Real.log 2 ≠ 0) :
    Real.log 4 / Real.log 10 + Real.log 50 / Real.log 10 - Real.log 2 / Real.log 10 = 2 := by
  sorry

-- Problem (2)
theorem reciprocals_of_logs (a b : Real) (h : 1 + Real.log a / Real.log 2 = 2 + Real.log b / Real.log 3 ∧ (1 + Real.log a / Real.log 2) = Real.log (a + b) / Real.log 6) : 
    1 / a + 1 / b = 6 := by
  sorry

end log_sum_reciprocals_of_logs_l84_84313


namespace probability_of_selection_l84_84150

theorem probability_of_selection (total_students : ℕ) (eliminated_students : ℕ) (groups : ℕ) (selected_students : ℕ)
(h1 : total_students = 1003) 
(h2 : eliminated_students = 3)
(h3 : groups = 20)
(h4 : selected_students = 50) : 
(selected_students : ℝ) / (total_students : ℝ) = 50 / 1003 :=
by
  sorry

end probability_of_selection_l84_84150


namespace solution_set_inequality_l84_84387

theorem solution_set_inequality {a b c : ℝ} (h₁ : a < 0)
  (h₂ : ∀ x : ℝ, (a * x^2 + b * x + c <= 0) ↔ (x <= -(1/3) ∨ 2 <= x)) :
  (∀ x : ℝ, (c * x^2 + b * x + a > 0) ↔ (x < -3 ∨ 1/2 < x)) :=
by
  sorry

end solution_set_inequality_l84_84387


namespace factory_production_schedule_l84_84292

noncomputable def production_equation (x : ℝ) : Prop :=
  (1000 / x) - (1000 / (1.2 * x)) = 2

theorem factory_production_schedule (x : ℝ) (hx : x ≠ 0) : production_equation x := 
by 
  -- Assumptions based on conditions:
  -- Factory plans to produce total of 1000 sets of protective clothing.
  -- Actual production is 20% more than planned.
  -- Task completed 2 days ahead of original schedule.
  -- We need to show: (1000 / x) - (1000 / (1.2 * x)) = 2
  sorry

end factory_production_schedule_l84_84292


namespace average_weight_increase_l84_84840

theorem average_weight_increase 
  (n : ℕ) (old_weight new_weight : ℝ) (group_size := 8) 
  (old_weight := 70) (new_weight := 90) : 
  ((new_weight - old_weight) / group_size) = 2.5 := 
by sorry

end average_weight_increase_l84_84840


namespace find_f_neg_five_half_l84_84088

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
else if 0 ≤ x + 2 ∧ x + 2 ≤ 1 then 2 * (x + 2) * (1 - (x + 2))
     else -2 * abs x * (1 - abs x)

theorem find_f_neg_five_half (x : ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 2) = f x)
  (h_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)) : 
  f (-5 / 2) = -1 / 2 :=
  by sorry

end find_f_neg_five_half_l84_84088


namespace lynne_total_spending_l84_84288

theorem lynne_total_spending :
  let num_books_cats := 7
  let num_books_solar_system := 2
  let num_magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := num_books_cats + num_books_solar_system
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := num_magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 := sorry

end lynne_total_spending_l84_84288


namespace exp_problem_l84_84872

theorem exp_problem (a b c : ℕ) (H1 : a = 1000) (H2 : b = 1000^1000) (H3 : c = 500^1000) :
  a * b / c = 2^1001 * 500 :=
sorry

end exp_problem_l84_84872


namespace log_ratios_l84_84911

noncomputable def ratio_eq : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem log_ratios
  {a b : ℝ}
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : Real.log a / Real.log 8 = Real.log b / Real.log 18)
  (h4 : Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) :
  b / a = ratio_eq :=
sorry

end log_ratios_l84_84911


namespace Hari_contribution_l84_84231

theorem Hari_contribution (P T_P T_H : ℕ) (r1 r2 : ℕ) (H : ℕ) :
  P = 3500 → 
  T_P = 12 → 
  T_H = 7 → 
  r1 = 2 → 
  r2 = 3 →
  (P * T_P) * r2 = (H * T_H) * r1 →
  H = 9000 :=
by
  sorry

end Hari_contribution_l84_84231


namespace number_of_boys_selected_l84_84835

theorem number_of_boys_selected {boys girls selections : ℕ} 
  (h_boys : boys = 11) (h_girls : girls = 10) (h_selections : selections = 6600) : 
  ∃ (k : ℕ), k = 2 :=
sorry

end number_of_boys_selected_l84_84835


namespace find_f_m_l84_84295

-- Definitions based on the conditions
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 3

axiom condition (m a : ℝ) : f (-m) a = 1

-- The statement to be proven
theorem find_f_m (m a : ℝ) (hm : f (-m) a = 1) : f m a = 5 := 
by sorry

end find_f_m_l84_84295


namespace meal_combinations_correct_l84_84985

-- Define the given conditions
def number_of_entrees : Nat := 4
def number_of_drinks : Nat := 4
def number_of_desserts : Nat := 2

-- Define the total number of meal combinations to prove
def total_meal_combinations : Nat := number_of_entrees * number_of_drinks * number_of_desserts

-- The theorem we want to prove
theorem meal_combinations_correct : total_meal_combinations = 32 := 
by 
  sorry

end meal_combinations_correct_l84_84985


namespace negation_of_exists_lt_zero_l84_84000

theorem negation_of_exists_lt_zero (m : ℝ) :
  ¬ (∃ x : ℝ, x < 0 ∧ x^2 + 2 * x - m > 0) ↔ ∀ x : ℝ, x < 0 → x^2 + 2 * x - m ≤ 0 :=
by sorry

end negation_of_exists_lt_zero_l84_84000


namespace math_problem_proof_l84_84972

theorem math_problem_proof : 
  ((9 - 8 + 7) ^ 2 * 6 + 5 - 4 ^ 2 * 3 + 2 ^ 3 - 1) = 347 := 
by sorry

end math_problem_proof_l84_84972


namespace boxes_needed_l84_84214

-- Define the given conditions

def red_pencils : ℕ := 20
def blue_pencils : ℕ := 2 * red_pencils
def yellow_pencils : ℕ := 40
def green_pencils : ℕ := red_pencils + blue_pencils
def total_pencils : ℕ := red_pencils + blue_pencils + green_pencils + yellow_pencils
def pencils_per_box : ℕ := 20

-- Lean theorem statement to prove the number of boxes needed is 8

theorem boxes_needed : total_pencils / pencils_per_box = 8 :=
by
  -- This is where the proof would go
  sorry

end boxes_needed_l84_84214


namespace train_length_is_correct_l84_84592

noncomputable def speed_kmph : ℝ := 72
noncomputable def time_seconds : ℝ := 74.994
noncomputable def tunnel_length_m : ℝ := 1400
noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600
noncomputable def total_distance : ℝ := speed_mps * time_seconds
noncomputable def train_length : ℝ := total_distance - tunnel_length_m

theorem train_length_is_correct :
  train_length = 99.88 := by
  -- the proof will follow here
  sorry

end train_length_is_correct_l84_84592


namespace functional_form_of_f_l84_84980

variable (f : ℝ → ℝ)

-- Define the condition as an axiom
axiom cond_f : ∀ (x y : ℝ), |f (x + y) - f (x - y) - y| ≤ y^2

-- State the theorem to be proved
theorem functional_form_of_f :
  ∃ c : ℝ, ∀ x : ℝ, f x = x / 2 + c := by
  sorry

end functional_form_of_f_l84_84980


namespace felix_chopped_at_least_91_trees_l84_84133

def cost_to_sharpen := 5
def total_spent := 35
def trees_per_sharpen := 13

theorem felix_chopped_at_least_91_trees :
  (total_spent / cost_to_sharpen) * trees_per_sharpen = 91 := by
  sorry

end felix_chopped_at_least_91_trees_l84_84133


namespace bigger_part_of_sum_54_l84_84254

theorem bigger_part_of_sum_54 (x y : ℕ) (h₁ : x + y = 54) (h₂ : 10 * x + 22 * y = 780) : x = 34 :=
sorry

end bigger_part_of_sum_54_l84_84254


namespace geometric_sequence_relation_l84_84153

variables {a : ℕ → ℝ} {q : ℝ}
variables {m n p : ℕ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def are_in_geometric_sequence (a : ℕ → ℝ) (m n p : ℕ) : Prop :=
  a n ^ 2 = a m * a p

-- Theorem
theorem geometric_sequence_relation (h_geom : is_geometric_sequence a q) (h_order : are_in_geometric_sequence a m n p) (hq_ne_one : q ≠ 1) :
  2 * n = m + p :=
sorry

end geometric_sequence_relation_l84_84153


namespace musical_chairs_l84_84487

def is_prime_power (m : ℕ) : Prop :=
  ∃ (p k : ℕ), Nat.Prime p ∧ k > 0 ∧ m = p ^ k

theorem musical_chairs (n m : ℕ) (h1 : 1 < m) (h2 : m ≤ n) (h3 : ¬ is_prime_power m) :
  ∃ f : Fin n → Fin n, (∀ x, f x ≠ x) ∧ (∀ x, (f^[m]) x = x) :=
sorry

end musical_chairs_l84_84487


namespace electronics_weight_is_9_l84_84915

noncomputable def electronics_weight : ℕ :=
  let B : ℕ := sorry -- placeholder for the value of books weight.
  let C : ℕ := 12
  let E : ℕ := 9
  have h1 : (B : ℚ) / (C : ℚ) = 7 / 4 := sorry
  have h2 : (C : ℚ) / (E : ℚ) = 4 / 3 := sorry
  have h3 : (B : ℚ) / (C - 6 : ℚ) = 7 / 2 := sorry
  E

theorem electronics_weight_is_9 : electronics_weight = 9 :=
by
  dsimp [electronics_weight]
  repeat { sorry }

end electronics_weight_is_9_l84_84915


namespace long_furred_brown_dogs_l84_84089

-- Definitions based on given conditions
def T : ℕ := 45
def L : ℕ := 36
def B : ℕ := 27
def N : ℕ := 8

-- The number of long-furred brown dogs (LB) that needs to be proved
def LB : ℕ := 26

-- Lean 4 statement to prove LB
theorem long_furred_brown_dogs :
  L + B - LB = T - N :=
by 
  unfold T L B N LB -- we unfold definitions to simplify the theorem
  sorry

end long_furred_brown_dogs_l84_84089


namespace sums_solved_correctly_l84_84545

theorem sums_solved_correctly (x : ℕ) (h : x + 2 * x = 48) : x = 16 := by
  sorry

end sums_solved_correctly_l84_84545


namespace geometricSeqMinimumValue_l84_84141

noncomputable def isMinimumValue (a : ℕ → ℝ) (n m : ℕ) (value : ℝ) : Prop :=
  ∀ b : ℝ, (1 / a n + b / a m) ≥ value

theorem geometricSeqMinimumValue {a : ℕ → ℝ}
  (h1 : ∀ n, a n > 0)
  (h2 : a 7 = (Real.sqrt 2) / 2)
  (h3 : ∀ n, ∀ m, a n * a m = a (n + m)) :
  isMinimumValue a 3 11 4 :=
sorry

end geometricSeqMinimumValue_l84_84141


namespace weekly_exercise_time_l84_84718

def milesWalked := 3
def walkingSpeed := 3 -- in miles per hour
def milesRan := 10
def runningSpeed := 5 -- in miles per hour
def daysInWeek := 7

theorem weekly_exercise_time : (milesWalked / walkingSpeed + milesRan / runningSpeed) * daysInWeek = 21 := 
by
  -- The actual proof part is intentionally omitted as per the instruction
  sorry

end weekly_exercise_time_l84_84718


namespace gcd_of_polynomial_and_linear_l84_84303

theorem gcd_of_polynomial_and_linear (b : ℤ) (h1 : b % 2 = 1) (h2 : 1019 ∣ b) : 
  Int.gcd (3 * b ^ 2 + 31 * b + 91) (b + 15) = 1 := 
by 
  sorry

end gcd_of_polynomial_and_linear_l84_84303


namespace num_four_digit_integers_with_at_least_one_4_or_7_l84_84744

def count_four_digit_integers_with_4_or_7 : ℕ := 5416

theorem num_four_digit_integers_with_at_least_one_4_or_7 :
  let all_four_digit_integers := 9000
  let valid_digits_first := 7
  let valid_digits := 8
  let integers_without_4_or_7 := valid_digits_first * valid_digits * valid_digits * valid_digits
  all_four_digit_integers - integers_without_4_or_7 = count_four_digit_integers_with_4_or_7 :=
by
  -- Using known values from the problem statement
  let all_four_digit_integers := 9000
  let valid_digits_first := 7
  let valid_digits := 8
  let integers_without_4_or_7 := valid_digits_first * valid_digits * valid_digits * valid_digits
  show all_four_digit_integers - integers_without_4_or_7 = count_four_digit_integers_with_4_or_7
  sorry

end num_four_digit_integers_with_at_least_one_4_or_7_l84_84744


namespace part_I_part_II_l84_84171

namespace VectorProblems

def vector_a : ℝ × ℝ := (3, 2)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (4, 1)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem part_I (m : ℝ) :
  let u := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
  let v := (4 * m + vector_b.1, m + vector_b.2)
  dot_product u v > 0 →
  m ≠ 4 / 7 →
  m > -1 / 2 :=
sorry

theorem part_II (k : ℝ) :
  let u := (vector_a.1 + 4 * k, vector_a.2 + k)
  let v := (2 * vector_b.1 - vector_a.1, 2 * vector_b.2 - vector_a.2)
  dot_product u v = 0 →
  k = -11 / 18 :=
sorry

end VectorProblems

end part_I_part_II_l84_84171


namespace tangent_line_at_1_intervals_of_monotonicity_and_extrema_l84_84413

open Real

noncomputable def f (x : ℝ) := 6 * log x + (1 / 2) * x^2 - 5 * x

theorem tangent_line_at_1 :
  let f' (x : ℝ) := (6 / x) + x - 5
  (f 1 = -9 / 2) →
  (f' 1 = 2) →
  (∀ x y : ℝ, y + 9 / 2 = 2 * (x - 1) → 4 * x - 2 * y - 13 = 0) := 
by
  sorry

theorem intervals_of_monotonicity_and_extrema :
  let f' (x : ℝ) := (x^2 - 5 * x + 6) / x
  (∀ x, 0 < x ∧ x < 2 → f' x > 0) → 
  (∀ x, 3 < x → f' x > 0) →
  (∀ x, 2 < x ∧ x < 3 → f' x < 0) →
  (f 2 = -8 + 6 * log 2) →
  (f 3 = -21 / 2 + 6 * log 3) :=
by
  sorry

end tangent_line_at_1_intervals_of_monotonicity_and_extrema_l84_84413


namespace smallest_number_l84_84326

-- Definitions of the numbers in their respective bases
def num1 := 5 * 9^0 + 8 * 9^1 -- 85_9
def num2 := 0 * 6^0 + 1 * 6^1 + 2 * 6^2 -- 210_6
def num3 := 0 * 4^0 + 0 * 4^1 + 0 * 4^2 + 1 * 4^3 -- 1000_4
def num4 := 1 * 2^0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3 + 1 * 2^4 + 1 * 2^5 -- 111111_2

-- Assert that num4 is the smallest
theorem smallest_number : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 :=
by 
  sorry

end smallest_number_l84_84326


namespace cost_of_lunch_l84_84206

-- Define the conditions: total amount and tip percentage
def total_amount : ℝ := 72.6
def tip_percentage : ℝ := 0.20

-- Define the proof problem
theorem cost_of_lunch (C : ℝ) (h : C + tip_percentage * C = total_amount) : C = 60.5 := 
sorry

end cost_of_lunch_l84_84206


namespace maximum_marks_l84_84230

theorem maximum_marks (M : ℝ) (P : ℝ) 
  (h1 : P = 0.45 * M) -- 45% of the maximum marks to pass
  (h2 : P = 210 + 40) -- Pradeep's marks plus failed marks

  : M = 556 := 
sorry

end maximum_marks_l84_84230


namespace temperature_difference_correct_l84_84503

def avg_high : ℝ := 9
def avg_low : ℝ := -5
def temp_difference : ℝ := avg_high - avg_low

theorem temperature_difference_correct : temp_difference = 14 := by
  sorry

end temperature_difference_correct_l84_84503


namespace find_other_number_l84_84788

theorem find_other_number (a b : ℕ) (h_lcm : Nat.lcm a b = 9240) (h_gcd : Nat.gcd a b = 33) (h_a : a = 231) : b = 1320 :=
sorry

end find_other_number_l84_84788


namespace find_b_l84_84945

noncomputable def general_quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_b (a c : ℝ) (y1 y2 : ℝ) :
  y1 = general_quadratic a 3 c 2 →
  y2 = general_quadratic a 3 c (-2) →
  y1 - y2 = 12 →
  3 = 3 :=
by
  intros h1 h2 h3
  sorry

end find_b_l84_84945


namespace area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l84_84986

-- Definitions and conditions
variable {A B C a b c : ℝ}
variable (cosA : ℝ) (sinA : ℝ)
variable (area : ℝ)
variable (tanA tanB tanC : ℝ)

-- Given conditions
axiom angle_identity : b^2 + c^2 = 3 * b * c * cosA
axiom sin_cos_identity : sinA^2 + cosA^2 = 1
axiom law_of_cosines : a^2 = b^2 + c^2 - 2 * b * c * cosA

-- Part (1) statement
theorem area_of_triangle_is_sqrt_5 (B_eq_C : B = C) (a_eq_2 : a = 2) 
    (cosA_eq_2_3 : cosA = 2/3) 
    (b_eq_sqrt6 : b = Real.sqrt 6) 
    (sinA_eq_sqrt5_3 : sinA = Real.sqrt 5 / 3) 
    : area = Real.sqrt 5 := sorry

-- Part (2) statement
theorem sum_of_tangents_eq_1 (tanA_eq : tanA = sinA / cosA)
    (tanB_eq : tanB = sinA * sinA / (cosA * cosA))
    (tanC_eq : tanC = sinA * sinA / (cosA * cosA))
    : (tanA / tanB) + (tanA / tanC) = 1 := sorry

end area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l84_84986


namespace num_small_boxes_l84_84257

-- Conditions
def chocolates_per_small_box := 25
def total_chocolates := 400

-- Claim: Prove that the number of small boxes is 16
theorem num_small_boxes : (total_chocolates / chocolates_per_small_box) = 16 := 
by sorry

end num_small_boxes_l84_84257


namespace percentage_reduction_l84_84809

theorem percentage_reduction :
  let P := 60
  let R := 45
  (900 / R) - (900 / P) = 5 →
  (P - R) / P * 100 = 25 :=
by 
  intros P R h
  have h1 : R = 45 := rfl
  have h2 : P = 60 := sorry
  rw [h1] at h
  rw [h2]
  sorry -- detailed steps to be filled in the proof

end percentage_reduction_l84_84809


namespace arithmetic_sequence_a1_a9_l84_84407

variable (a : ℕ → ℝ)

-- This statement captures if given condition holds, prove a_1 + a_9 = 18.
theorem arithmetic_sequence_a1_a9 (h : a 4 + a 5 + a 6 = 27)
    (h_seq : ∀ (n : ℕ), a (n + 1) = a n + (a 2 - a 1)) :
    a 1 + a 9 = 18 :=
sorry

end arithmetic_sequence_a1_a9_l84_84407


namespace remaining_money_l84_84578

def initial_amount : ℕ := 10
def spent_on_toy_truck : ℕ := 3
def spent_on_pencil_case : ℕ := 2

theorem remaining_money (initial_amount spent_on_toy_truck spent_on_pencil_case : ℕ) : 
  initial_amount - (spent_on_toy_truck + spent_on_pencil_case) = 5 :=
by
  sorry

end remaining_money_l84_84578


namespace max_value_f_l84_84167

theorem max_value_f (x y z : ℝ) (hxyz : x * y * z = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (1 - y * z + z) * (1 - x * z + x) * (1 - x * y + y) ≤ 1 :=
sorry

end max_value_f_l84_84167


namespace inconsistent_linear_system_l84_84741

theorem inconsistent_linear_system :
  ¬ ∃ (x1 x2 x3 : ℝ), 
    (2 * x1 + 5 * x2 - 4 * x3 = 8) ∧
    (3 * x1 + 15 * x2 - 9 * x3 = 5) ∧
    (5 * x1 + 5 * x2 - 7 * x3 = 1) :=
by
  -- Proof of inconsistency
  sorry

end inconsistent_linear_system_l84_84741


namespace find_numbers_l84_84633

theorem find_numbers (x y : ℤ) (h1 : x > y) (h2 : x^2 - y^2 = 100) : 
  x = 26 ∧ y = 24 := 
  sorry

end find_numbers_l84_84633


namespace abc_sum_is_17_l84_84773

noncomputable def A := 3
noncomputable def B := 5
noncomputable def C := 9

theorem abc_sum_is_17 (A B C : ℕ) (h1 : 100 * A + 10 * B + C = 359) (h2 : 4 * (100 * A + 10 * B + C) = 1436)
  (h3 : A ≠ B) (h4 : B ≠ C) (h5 : A ≠ C) : A + B + C = 17 :=
by
  sorry

end abc_sum_is_17_l84_84773


namespace original_cookies_l84_84001

noncomputable def initial_cookies (final_cookies : ℝ) (ratio : ℝ) (days : ℕ) : ℝ :=
  final_cookies / ratio^days

theorem original_cookies :
  ∀ (final_cookies : ℝ) (ratio : ℝ) (days : ℕ),
  final_cookies = 28 →
  ratio = 0.7 →
  days = 3 →
  initial_cookies final_cookies ratio days = 82 :=
by
  intros final_cookies ratio days h_final h_ratio h_days
  rw [initial_cookies, h_final, h_ratio, h_days]
  norm_num
  sorry

end original_cookies_l84_84001


namespace calc_fractional_product_l84_84542

theorem calc_fractional_product (a b : ℝ) : (1 / 3) * a^2 * (-6 * a * b) = -2 * a^3 * b :=
by
  sorry

end calc_fractional_product_l84_84542


namespace malcolm_initial_white_lights_l84_84914

theorem malcolm_initial_white_lights :
  ∀ (red blue green remaining total_initial : ℕ),
    red = 12 →
    blue = 3 * red →
    green = 6 →
    remaining = 5 →
    total_initial = red + blue + green + remaining →
    total_initial = 59 :=
by
  intros red blue green remaining total_initial h1 h2 h3 h4 h5
  -- Add details if necessary for illustration
  -- sorry typically as per instructions
  sorry

end malcolm_initial_white_lights_l84_84914


namespace ariel_years_fencing_l84_84049

-- Definitions based on given conditions
def fencing_start_year := 2006
def birth_year := 1992
def current_age := 30

-- To find: The number of years Ariel has been fencing
def current_year : ℕ := birth_year + current_age
def years_fencing : ℕ := current_year - fencing_start_year

-- Proof statement
theorem ariel_years_fencing : years_fencing = 16 := by
  sorry

end ariel_years_fencing_l84_84049


namespace tree_height_l84_84593

theorem tree_height (boy_initial_height tree_initial_height boy_final_height boy_growth_rate tree_growth_rate : ℝ) 
  (h1 : boy_initial_height = 24) 
  (h2 : tree_initial_height = 16) 
  (h3 : boy_final_height = 36) 
  (h4 : boy_growth_rate = boy_final_height - boy_initial_height) 
  (h5 : tree_growth_rate = 2 * boy_growth_rate) 
  : tree_initial_height + tree_growth_rate = 40 := 
by
  subst h1 h2 h3 h4 h5;
  sorry

end tree_height_l84_84593


namespace analogical_reasoning_ineq_l84_84318

-- Formalization of the conditions and the theorem to be proved

def positive (a : ℕ → ℝ) (n : ℕ) := ∀ i, 1 ≤ i → i ≤ n → a i > 0

theorem analogical_reasoning_ineq {a : ℕ → ℝ} (hpos : positive a 4) (hsum : a 1 + a 2 + a 3 + a 4 = 1) : 
  (1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4) ≥ 16 := 
sorry

end analogical_reasoning_ineq_l84_84318


namespace find_sequence_formula_l84_84923

variable (a : ℕ → ℝ)

noncomputable def sequence_formula := ∀ n : ℕ, a n = Real.sqrt n

lemma sequence_initial : a 1 = 1 :=
sorry

lemma sequence_recursive (n : ℕ) : a (n+1)^2 - a n^2 = 1 :=
sorry

theorem find_sequence_formula : sequence_formula a :=
sorry

end find_sequence_formula_l84_84923


namespace trigonometric_inequality_l84_84375

theorem trigonometric_inequality (x : Real) (n : Int) :
  (9.286 * (Real.sin x)^3 * Real.sin ((Real.pi / 2) - 3 * x) +
   (Real.cos x)^3 * Real.cos ((Real.pi / 2) - 3 * x) > 
   3 * Real.sqrt 3 / 8) →
   (x > (Real.pi / 12) + (Real.pi * n / 2) ∧
   x < (5 * Real.pi / 12) + (Real.pi * n / 2)) :=
by
  sorry

end trigonometric_inequality_l84_84375


namespace molecular_weight_of_acid_l84_84875

theorem molecular_weight_of_acid (molecular_weight : ℕ) (n : ℕ) (h : molecular_weight = 792) (hn : n = 9) :
  molecular_weight = 792 :=
by 
  sorry

end molecular_weight_of_acid_l84_84875


namespace solve_for_y_l84_84077

theorem solve_for_y (y : ℝ) : 4 * y + 6 * y = 450 - 10 * (y - 5) → y = 25 :=
by
  sorry

end solve_for_y_l84_84077


namespace minutes_past_midnight_l84_84187

-- Definitions for the problem

def degree_per_tick : ℝ := 30
def degree_per_minute_hand : ℝ := 6
def degree_per_hour_hand_hourly : ℝ := 30
def degree_per_hour_hand_minutes : ℝ := 0.5

def condition_minute_hand_degree := 300
def condition_hour_hand_degree := 70

-- Main theorem statement
theorem minutes_past_midnight :
  ∃ (h m: ℝ),
    degree_per_hour_hand_hourly * h + degree_per_hour_hand_minutes * m = condition_hour_hand_degree ∧
    degree_per_minute_hand * m = condition_minute_hand_degree ∧
    h * 60 + m = 110 :=
by
  sorry

end minutes_past_midnight_l84_84187


namespace required_vases_l84_84576

def vase_capacity_roses : Nat := 6
def vase_capacity_tulips : Nat := 8
def vase_capacity_lilies : Nat := 4

def remaining_roses : Nat := 20
def remaining_tulips : Nat := 15
def remaining_lilies : Nat := 5

def vases_for_roses : Nat := (remaining_roses + vase_capacity_roses - 1) / vase_capacity_roses
def vases_for_tulips : Nat := (remaining_tulips + vase_capacity_tulips - 1) / vase_capacity_tulips
def vases_for_lilies : Nat := (remaining_lilies + vase_capacity_lilies - 1) / vase_capacity_lilies

def total_vases_needed : Nat := vases_for_roses + vases_for_tulips + vases_for_lilies

theorem required_vases : total_vases_needed = 8 := by
  sorry

end required_vases_l84_84576


namespace candidates_appeared_in_each_state_equals_7900_l84_84651

theorem candidates_appeared_in_each_state_equals_7900 (x : ℝ) (h : 0.07 * x = 0.06 * x + 79) : x = 7900 :=
sorry

end candidates_appeared_in_each_state_equals_7900_l84_84651


namespace connie_marbles_l84_84140

-- Define the initial number of marbles that Connie had
def initial_marbles : ℝ := 73.5

-- Define the number of marbles that Connie gave away
def marbles_given : ℝ := 70.3

-- Define the expected number of marbles remaining
def marbles_remaining : ℝ := 3.2

-- State the theorem: prove that initial_marbles - marbles_given = marbles_remaining
theorem connie_marbles :
  initial_marbles - marbles_given = marbles_remaining :=
sorry

end connie_marbles_l84_84140


namespace count_special_four_digit_integers_is_100_l84_84699

def count_special_four_digit_integers : Nat := sorry

theorem count_special_four_digit_integers_is_100 :
  count_special_four_digit_integers = 100 :=
sorry

end count_special_four_digit_integers_is_100_l84_84699


namespace alex_casey_meet_probability_l84_84136

noncomputable def probability_meet : ℚ :=
  let L := (1:ℚ) / 3;
  let area_of_square := 1;
  let area_of_triangles := (1 / 2) * L ^ 2;
  let area_of_meeting_region := area_of_square - 2 * area_of_triangles;
  area_of_meeting_region / area_of_square

theorem alex_casey_meet_probability :
  probability_meet = 8 / 9 :=
by
  sorry

end alex_casey_meet_probability_l84_84136


namespace barbara_removed_114_sheets_l84_84566

/-- Given conditions: -/
def bundles (n : ℕ) := 2 * n
def bunches (n : ℕ) := 4 * n
def heaps (n : ℕ) := 20 * n

/-- Barbara removed certain amounts of paper from the chest of drawers. -/
def total_sheets_removed := bundles 3 + bunches 2 + heaps 5

theorem barbara_removed_114_sheets : total_sheets_removed = 114 := by
  -- proof will be inserted here
  sorry

end barbara_removed_114_sheets_l84_84566


namespace find_z_l84_84794

theorem find_z
  (z : ℝ)
  (h : (1 : ℝ) • (2 : ℝ) + 4 • (-1 : ℝ) + z • (3 : ℝ) = 6) :
  z = 8 / 3 :=
by 
  sorry

end find_z_l84_84794


namespace four_machines_save_11_hours_l84_84098

-- Define the conditions
def three_machines_complete_order_in_44_hours := 3 * (1 / (3 * 44)) * 44 = 1

def additional_machine_reduces_time (T : ℝ) := 4 * (1 / (3 * 44)) * T = 1

-- Define the theorem to prove the number of hours saved
theorem four_machines_save_11_hours : 
  (∃ T : ℝ, additional_machine_reduces_time T ∧ three_machines_complete_order_in_44_hours) → 
  44 - 33 = 11 :=
by
  sorry

end four_machines_save_11_hours_l84_84098


namespace find_complex_number_l84_84205

open Complex

theorem find_complex_number (z : ℂ) (hz : z + Complex.abs z = Complex.ofReal 2 + 8 * Complex.I) : 
z = -15 + 8 * Complex.I := by sorry

end find_complex_number_l84_84205


namespace number_div_0_04_eq_200_9_l84_84060

theorem number_div_0_04_eq_200_9 (n : ℝ) (h : n / 0.04 = 200.9) : n = 8.036 :=
sorry

end number_div_0_04_eq_200_9_l84_84060


namespace at_least_one_female_team_l84_84302

open Classical

namespace Probability

-- Define the Problem
noncomputable def prob_at_least_one_female (females males : ℕ) (team_size : ℕ) :=
  let total_students := females + males
  let total_ways := Nat.choose total_students team_size
  let ways_all_males := Nat.choose males team_size
  1 - (ways_all_males / total_ways : ℝ)

-- Verify the given problem against the expected answer
theorem at_least_one_female_team :
  prob_at_least_one_female 1 3 2 = 1 / 2 := by
  sorry

end Probability

end at_least_one_female_team_l84_84302


namespace imaginary_part_z_l84_84661

open Complex

theorem imaginary_part_z : (im ((i - 1) / (i + 1))) = 1 :=
by
  -- The proof goes here, but it can be marked with sorry for now
  sorry

end imaginary_part_z_l84_84661


namespace bananas_per_friend_l84_84057

theorem bananas_per_friend (total_bananas : ℤ) (total_friends : ℤ) (H1 : total_bananas = 21) (H2 : total_friends = 3) : 
  total_bananas / total_friends = 7 :=
by
  sorry

end bananas_per_friend_l84_84057


namespace seq_is_geometric_from_second_l84_84324

namespace sequence_problem

-- Definitions and conditions
def S : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| (n + 1) => 3 * S n - 2 * S (n - 1)

-- Recursive definition for sum of sequence terms
axiom S_rec_relation (n : ℕ) (h : n ≥ 2) : 
  S (n + 1) - 3 * S n + 2 * S (n - 1) = 0

-- Prove the sequence is geometric from the second term
theorem seq_is_geometric_from_second :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 2, a (n + 1) = 2 * a n) ∧ 
  (a 1 = 1) ∧ 
  (a 2 = 1) :=
by
  sorry

end sequence_problem

end seq_is_geometric_from_second_l84_84324


namespace positive_difference_of_two_numbers_l84_84698

theorem positive_difference_of_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 10) 
  (h2 : x^2 - y^2 = 24) : 
  |x - y| = 2.4 := 
sorry

end positive_difference_of_two_numbers_l84_84698


namespace earphone_cost_l84_84244

/-- 
The cost of the earphone purchased on Friday can be calculated given:
1. The mean expenditure over 7 days is 500.
2. The expenditures for Monday, Tuesday, Wednesday, Thursday, Saturday, and Sunday are 450, 600, 400, 500, 550, and 300, respectively.
3. On Friday, the expenditures include a pen costing 30 and a notebook costing 50.
-/
theorem earphone_cost
  (mean_expenditure : ℕ)
  (mon tue wed thur sat sun : ℕ)
  (pen_cost notebook_cost : ℕ)
  (mean_expenditure_eq : mean_expenditure = 500)
  (mon_eq : mon = 450)
  (tue_eq : tue = 600)
  (wed_eq : wed = 400)
  (thur_eq : thur = 500)
  (sat_eq : sat = 550)
  (sun_eq : sun = 300)
  (pen_cost_eq : pen_cost = 30)
  (notebook_cost_eq : notebook_cost = 50)
  : ∃ (earphone_cost : ℕ), earphone_cost = 620 := 
by
  sorry

end earphone_cost_l84_84244


namespace arithmetic_sequence_nth_term_l84_84913

noncomputable def nth_arithmetic_term (a : ℤ) (n : ℕ) : ℤ :=
  let a1 := a - 1
  let a2 := a + 1
  let a3 := 2 * a + 3
  if 2 * (a + 1) = (a - 1) + (2 * a + 3) then
    -1 + (n - 1) * 2
  else
    sorry

theorem arithmetic_sequence_nth_term (a : ℤ) (n : ℕ) (h : 2 * (a + 1) = (a - 1) + (2 * a + 3)) :
  nth_arithmetic_term a n = 2 * (n : ℤ) - 3 :=
by
  sorry

end arithmetic_sequence_nth_term_l84_84913


namespace fish_price_eq_shrimp_price_l84_84667

-- Conditions
variable (x : ℝ) -- regular price for a full pound of fish
variable (h1 : 0.6 * (x / 4) = 1.50) -- quarter-pound fish price after 60% discount
variable (shrimp_price : ℝ) -- price per pound of shrimp
variable (h2 : shrimp_price = 10) -- given shrimp price

-- Proof Statement
theorem fish_price_eq_shrimp_price (h1 : 0.6 * (x / 4) = 1.50) (h2 : shrimp_price = 10) :
  x = 10 ∧ x = shrimp_price :=
by
  sorry

end fish_price_eq_shrimp_price_l84_84667


namespace part_a_exists_part_b_not_exists_l84_84861

theorem part_a_exists :
  ∃ (a b : ℤ), (∀ x : ℝ, x^2 + a*x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + a*x + b = 0) :=
sorry

theorem part_b_not_exists :
  ¬ ∃ (a b : ℤ), (∀ x : ℝ, x^2 + 2*a*x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + 2*a*x + b = 0) :=
sorry

end part_a_exists_part_b_not_exists_l84_84861


namespace largest_divisor_l84_84884

theorem largest_divisor (n : ℕ) (hn : Even n) : ∃ k, ∀ n, Even n → k ∣ (n * (n+2) * (n+4) * (n+6) * (n+8)) ∧ (∀ m, (∀ n, Even n → m ∣ (n * (n+2) * (n+4) * (n+6) * (n+8))) → m ≤ k) :=
by
  use 96
  { sorry }

end largest_divisor_l84_84884


namespace value_of_expression_l84_84600

noncomputable def x := (2 : ℚ) / 3
noncomputable def y := (5 : ℚ) / 2

theorem value_of_expression : (1 / 3) * x^8 * y^9 = (5^9 / (2 * 3^9)) := by
  sorry

end value_of_expression_l84_84600


namespace john_baseball_cards_l84_84881

theorem john_baseball_cards (new_cards old_cards cards_per_page : ℕ) (h1 : new_cards = 8) (h2 : old_cards = 16) (h3 : cards_per_page = 3) :
  (new_cards + old_cards) / cards_per_page = 8 := by
  sorry

end john_baseball_cards_l84_84881


namespace wrapping_paper_area_l84_84100

variable (a b h w : ℝ) (a_gt_b : a > b)

theorem wrapping_paper_area : 
  ∃ total_area, total_area = 4 * (a * b + a * w + b * w + w ^ 2) :=
by
  sorry

end wrapping_paper_area_l84_84100


namespace probability_same_color_set_l84_84470

theorem probability_same_color_set 
  (black_pairs blue_pairs : ℕ)
  (green_pairs : {g : Finset (ℕ × ℕ) // g.card = 3})
  (total_pairs := 15)
  (total_shoes := total_pairs * 2) :
  2 * black_pairs + 2 * blue_pairs + green_pairs.val.card * 2 = total_shoes →
  ∃ probability : ℚ, 
    probability = 89 / 435 :=
by
  intro h_total_shoes
  let black_shoes := black_pairs * 2
  let blue_shoes := blue_pairs * 2
  let green_shoes := green_pairs.val.card * 2
  
  have h_black_probability : ℚ := (black_shoes / total_shoes) * (black_pairs / (total_shoes - 1))
  have h_blue_probability : ℚ := (blue_shoes / total_shoes) * (blue_pairs / (total_shoes - 1))
  have h_green_probability : ℚ := (green_shoes / total_shoes) * (green_pairs.val.card / (total_shoes - 1))
  
  have h_total_probability : ℚ := h_black_probability + h_blue_probability + h_green_probability
  
  use h_total_probability
  sorry

end probability_same_color_set_l84_84470


namespace distance_blown_by_storm_l84_84950

-- Definitions based on conditions
def speed : ℤ := 30
def time_travelled : ℤ := 20
def distance_travelled := speed * time_travelled
def total_distance := 2 * distance_travelled
def fractional_distance_left := total_distance / 3

-- Final statement to prove
theorem distance_blown_by_storm : distance_travelled - fractional_distance_left = 200 := by
  sorry

end distance_blown_by_storm_l84_84950


namespace quadratic_roots_satisfy_condition_l84_84160
variable (x1 x2 m : ℝ)

theorem quadratic_roots_satisfy_condition :
  ( ∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1 + x2 = -m) ∧ 
    (x1 * x2 = 5) ∧ (x1 = 2 * |x2| - 3) ) →
  m = -9 / 2 :=
by
  sorry

end quadratic_roots_satisfy_condition_l84_84160


namespace minimum_distance_focus_to_circle_point_l84_84461

def focus_of_parabola : ℝ × ℝ := (1, 0)
def center_of_circle : ℝ × ℝ := (4, 4)
def radius_of_circle : ℝ := 4
def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 4)^2 = 16

theorem minimum_distance_focus_to_circle_point :
  ∃ P : ℝ × ℝ, circle_equation P.1 P.2 ∧ dist focus_of_parabola P = 5 :=
sorry

end minimum_distance_focus_to_circle_point_l84_84461


namespace compute_expression_l84_84488

theorem compute_expression :
  (143 + 29) * 2 + 25 + 13 = 382 :=
by 
  sorry

end compute_expression_l84_84488


namespace simplify_fraction_l84_84611

theorem simplify_fraction :
  (1 / (1 + Real.sqrt 3) * 1 / (1 - Real.sqrt 5)) = 
  (1 / (1 - Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 15)) :=
by
  sorry

end simplify_fraction_l84_84611


namespace am_gm_problem_l84_84498

theorem am_gm_problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
by 
  sorry

end am_gm_problem_l84_84498


namespace train_crossing_time_l84_84053

-- Condition definitions
def length_train : ℝ := 100
def length_bridge : ℝ := 150
def speed_kmph : ℝ := 54
def speed_mps : ℝ := 15

-- Given the conditions, prove the time to cross the bridge is 16.67 seconds
theorem train_crossing_time :
  (100 + 150) / (54 * 1000 / 3600) = 16.67 := by sorry

end train_crossing_time_l84_84053


namespace find_a10_l84_84653

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Given conditions
variables (a : ℕ → ℤ) (h_arith_seq : arithmetic_sequence a)
variables (h_a2 : a 2 = 2) (h_a6 : a 6 = 10)

-- Goal to prove
theorem find_a10 : a 10 = 18 :=
by
  sorry

end find_a10_l84_84653


namespace nonagon_diagonals_count_l84_84437

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l84_84437


namespace inequality_abc_l84_84664

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2 * b + 3 * c) ^ 2 / (a ^ 2 + 2 * b ^ 2 + 3 * c ^ 2) ≤ 6 :=
sorry

end inequality_abc_l84_84664


namespace exactly_two_succeed_probability_l84_84771

/-- Define the probabilities of three independent events -/
def P1 : ℚ := 1 / 2
def P2 : ℚ := 1 / 3
def P3 : ℚ := 3 / 4

/-- Define the probability that exactly two out of the three people successfully decrypt the password -/
def prob_exactly_two_succeed : ℚ := P1 * P2 * (1 - P3) + P1 * (1 - P2) * P3 + (1 - P1) * P2 * P3

theorem exactly_two_succeed_probability :
  prob_exactly_two_succeed = 5 / 12 :=
sorry

end exactly_two_succeed_probability_l84_84771


namespace find_k_x_l84_84178

-- Define the nonzero polynomial condition
def nonzero_poly (p : Polynomial ℝ) : Prop :=
  ¬ (p = 0)

-- Define the conditions from the problem statement
def conditions (h k : Polynomial ℝ) : Prop :=
  nonzero_poly h ∧ nonzero_poly k ∧ (h.comp k = h * k) ∧ (k.eval 3 = 58)

-- State the main theorem to be proven
theorem find_k_x (h k : Polynomial ℝ) (cond : conditions h k) : 
  k = Polynomial.C 1 + Polynomial.C 49 * Polynomial.X + Polynomial.C (-49) * Polynomial.X^2 :=
sorry

end find_k_x_l84_84178


namespace pirate_treasure_probability_l84_84939

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem pirate_treasure_probability :
  let p_treasure := 1 / 5
  let p_traps := 1 / 10
  let p_neither := 7 / 10
  let num_islands := 8
  let num_treasure := 4
  binomial num_islands num_treasure * p_treasure^num_treasure * p_neither^(num_islands - num_treasure) = 673 / 25000 :=
by
  sorry

end pirate_treasure_probability_l84_84939


namespace ellipse_major_axis_focal_distance_l84_84406

theorem ellipse_major_axis_focal_distance (m : ℝ) (h1 : 10 - m > 0) (h2 : m - 2 > 0) 
  (h3 : ∀ x y, x^2 / (10 - m) + y^2 / (m - 2) = 1) 
  (h4 : ∃ c, 2 * c = 4 ∧ c^2 = (m - 2) - (10 - m)) : m = 8 :=
by
  sorry

end ellipse_major_axis_focal_distance_l84_84406


namespace inequality_B_l84_84084

variable {x y : ℝ}

theorem inequality_B (hx : 0 < x) (hy : 0 < y) (hxy : x > y) : x + 1 / (2 * y) > y + 1 / x :=
sorry

end inequality_B_l84_84084


namespace mean_score_is_93_l84_84855

-- Define Jane's scores as a list
def scores : List ℕ := [98, 97, 92, 85, 93]

-- Define the mean of the scores
noncomputable def mean (lst : List ℕ) : ℚ := 
  (lst.foldl (· + ·) 0 : ℚ) / lst.length

-- The theorem to prove
theorem mean_score_is_93 : mean scores = 93 := by
  sorry

end mean_score_is_93_l84_84855


namespace candies_shared_l84_84565

theorem candies_shared (y b d x : ℕ) (h1 : x = 2 * y + 10) (h2 : x = 3 * b + 18) (h3 : x = 5 * d - 55) (h4 : x + y + b + d = 2013) : x = 990 :=
by
  sorry

end candies_shared_l84_84565


namespace int_coeffs_square_sum_l84_84920

theorem int_coeffs_square_sum (a b c d e f : ℤ)
  (h : ∀ x, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 767 := 
sorry

end int_coeffs_square_sum_l84_84920


namespace tom_pie_share_l84_84252

theorem tom_pie_share :
  (∃ (x : ℚ), 4 * x = (5 / 8) ∧ x = 5 / 32) :=
by
  sorry

end tom_pie_share_l84_84252


namespace range_of_a_l84_84999

theorem range_of_a (a : ℝ) : (5 - a > 1) → (a < 4) := 
by
  sorry

end range_of_a_l84_84999


namespace M_is_infinite_l84_84275

variable (M : Set ℝ)

def has_properties (M : Set ℝ) : Prop :=
  (∃ x y : ℝ, x ∈ M ∧ y ∈ M ∧ x ≠ y) ∧ ∀ x ∈ M, (3*x - 2 ∈ M ∨ -4*x + 5 ∈ M)

theorem M_is_infinite (M : Set ℝ) (h : has_properties M) : ¬Finite M := by
  sorry

end M_is_infinite_l84_84275


namespace neg_square_positive_l84_84152

theorem neg_square_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := sorry

end neg_square_positive_l84_84152


namespace integer_k_values_l84_84910

theorem integer_k_values (a b k : ℝ) (m : ℝ) (ha : a > 0) (hb : b > 0) (hba_int : ∃ n : ℤ, n ≠ 0 ∧ b = (n : ℝ) * a) 
  (hA : a = a * k + m) (hB : 8 * b = b * k + m) : k = 9 ∨ k = 15 := 
by
  sorry

end integer_k_values_l84_84910


namespace cos_5theta_l84_84233

theorem cos_5theta (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (5*θ) = -93/3125 :=
sorry

end cos_5theta_l84_84233


namespace initial_visual_range_is_90_l84_84383

-- Define the initial visual range without the telescope (V).
variable (V : ℝ)

-- Define the condition that the visual range with the telescope is 150 km.
variable (condition1 : V + (2 / 3) * V = 150)

-- Define the proof problem statement.
theorem initial_visual_range_is_90 (V : ℝ) (condition1 : V + (2 / 3) * V = 150) : V = 90 :=
sorry

end initial_visual_range_is_90_l84_84383


namespace find_a_l84_84445

-- Define the quadratic equation with the root condition
def quadratic_with_root_zero (a : ℝ) : Prop :=
  (a - 1) * 0^2 + 0 + a - 2 = 0

-- State the theorem to be proved
theorem find_a (a : ℝ) (h : quadratic_with_root_zero a) : a = 2 :=
by
  -- Statement placeholder, proof omitted
  sorry

end find_a_l84_84445


namespace total_boxes_correct_l84_84662

noncomputable def friday_boxes : ℕ := 40

noncomputable def saturday_boxes : ℕ := 2 * friday_boxes - 10

noncomputable def sunday_boxes : ℕ := saturday_boxes / 2

noncomputable def monday_boxes : ℕ := 
  let extra_boxes := (25 * sunday_boxes + 99) / 100 -- (25/100) * sunday_boxes rounded to nearest integer
  sunday_boxes + extra_boxes

noncomputable def total_boxes : ℕ := 
  friday_boxes + saturday_boxes + sunday_boxes + monday_boxes

theorem total_boxes_correct : total_boxes = 189 := by
  sorry

end total_boxes_correct_l84_84662


namespace problem_statement_l84_84203

-- Define the repeating decimal 0.000272727... as x
noncomputable def repeatingDecimal : ℚ := 3 / 11000

-- Define the given condition for the question
def decimalRepeatsIndefinitely : Prop := 
  repeatingDecimal = 0.0002727272727272727  -- Representation for repeating decimal

-- Definitions of large powers of 10
def ten_pow_5 := 10^5
def ten_pow_3 := 10^3

-- The problem statement
theorem problem_statement : decimalRepeatsIndefinitely →
  (ten_pow_5 - ten_pow_3) * repeatingDecimal = 27 :=
sorry

end problem_statement_l84_84203


namespace find_correct_value_l84_84612

theorem find_correct_value (incorrect_value : ℝ) (subtracted_value : ℝ) (added_value : ℝ) (h_sub : subtracted_value = -added_value)
(h_incorrect : incorrect_value = 8.8) (h_subtracted : subtracted_value = -4.3) (h_added : added_value = 4.3) : incorrect_value + added_value + added_value = 17.4 :=
by
  sorry

end find_correct_value_l84_84612


namespace complement_union_l84_84370

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}
def complement_U_A : Set ℕ := U \ A

theorem complement_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) :
  (complement_U_A ∪ B) = {0, 2, 4} := by
  sorry

end complement_union_l84_84370


namespace calculate_expression_l84_84670

theorem calculate_expression :
  ((16^10 / 16^8) ^ 3 * 8 ^ 3) / 2 ^ 9 = 16777216 := by
  sorry

end calculate_expression_l84_84670


namespace minimum_toothpicks_to_remove_l84_84942

-- Definitions related to the problem statement
def total_toothpicks : Nat := 40
def initial_triangles : Nat := 36

-- Ensure that the minimal number of toothpicks to be removed to destroy all triangles is correct.
theorem minimum_toothpicks_to_remove : ∃ (n : Nat), n = 15 ∧ (∀ (t : Nat), t ≤ total_toothpicks - n → t = 0) :=
sorry

end minimum_toothpicks_to_remove_l84_84942


namespace penny_remaining_money_l84_84379

theorem penny_remaining_money (initial_money : ℤ) (socks_pairs : ℤ) (socks_cost_per_pair : ℤ) (hat_cost : ℤ) :
  initial_money = 20 → socks_pairs = 4 → socks_cost_per_pair = 2 → hat_cost = 7 → 
  initial_money - (socks_pairs * socks_cost_per_pair + hat_cost) = 5 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end penny_remaining_money_l84_84379


namespace fireworks_number_l84_84005

variable (x : ℕ)
variable (fireworks_total : ℕ := 484)
variable (happy_new_year_fireworks : ℕ := 12 * 5)
variable (boxes_of_fireworks : ℕ := 50 * 8)
variable (year_fireworks : ℕ := 4 * x)

theorem fireworks_number :
    4 * x + happy_new_year_fireworks + boxes_of_fireworks = fireworks_total →
    x = 6 := 
by
  sorry

end fireworks_number_l84_84005


namespace fraction_of_sophomores_attending_fair_l84_84516

theorem fraction_of_sophomores_attending_fair
  (s j n : ℕ)
  (h1 : s = j)
  (h2 : j = n)
  (soph_attend : ℚ)
  (junior_attend : ℚ)
  (senior_attend : ℚ)
  (fraction_s : soph_attend = 4/5 * s)
  (fraction_j : junior_attend = 3/4 * j)
  (fraction_n : senior_attend = 1/3 * n) :
  soph_attend / (soph_attend + junior_attend + senior_attend) = 240 / 565 :=
by
  sorry

end fraction_of_sophomores_attending_fair_l84_84516


namespace susan_homework_start_time_l84_84113

def start_time_homework (finish_time : ℕ) (homework_duration : ℕ) (interval_duration : ℕ) : ℕ :=
  finish_time - homework_duration - interval_duration

theorem susan_homework_start_time :
  let finish_time : ℕ := 16 * 60 -- 4:00 p.m. in minutes
  let homework_duration : ℕ := 96 -- Homework duration in minutes
  let interval_duration : ℕ := 25 -- Interval between homework finish and practice in minutes
  start_time_homework finish_time homework_duration interval_duration = 13 * 60 + 59 := -- 13:59 in minutes
by
  sorry

end susan_homework_start_time_l84_84113


namespace max_points_of_intersection_l84_84560

theorem max_points_of_intersection (circles : ℕ) (line : ℕ) (h_circles : circles = 3) (h_line : line = 1) : 
  ∃ points_of_intersection, points_of_intersection = 12 :=
by
  -- Proof here (omitted)
  sorry

end max_points_of_intersection_l84_84560


namespace order_of_p_q_r_l84_84678

theorem order_of_p_q_r (p q r : ℝ) (h1 : p = Real.sqrt 2) (h2 : q = Real.sqrt 7 - Real.sqrt 3) (h3 : r = Real.sqrt 6 - Real.sqrt 2) :
  p > r ∧ r > q :=
by
  sorry

end order_of_p_q_r_l84_84678


namespace find_integer_pairs_l84_84760

def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, k * k = x

theorem find_integer_pairs (m n : ℤ) :
  (is_perfect_square (m^2 + 4 * n) ∧ is_perfect_square (n^2 + 4 * m)) ↔
  (∃ a : ℤ, (m = 0 ∧ n = a^2) ∨ (m = a^2 ∧ n = 0) ∨ (m = -4 ∧ n = -4) ∨ (m = -5 ∧ n = -6) ∨ (m = -6 ∧ n = -5)) :=
by
  sorry

end find_integer_pairs_l84_84760


namespace frank_money_remaining_l84_84709

-- Define the conditions
def cost_cheapest_lamp : ℕ := 20
def factor_most_expensive : ℕ := 3
def initial_money : ℕ := 90

-- Define the cost of the most expensive lamp
def cost_most_expensive_lamp : ℕ := cost_cheapest_lamp * factor_most_expensive

-- Define the money remaining after purchase
def money_remaining : ℕ := initial_money - cost_most_expensive_lamp

-- The theorem we need to prove
theorem frank_money_remaining : money_remaining = 30 := by
  sorry

end frank_money_remaining_l84_84709


namespace possible_lost_rectangle_area_l84_84672

theorem possible_lost_rectangle_area (areas : Fin 10 → ℕ) (total_area : ℕ) (h_total : total_area = 65) :
  (∃ (i : Fin 10), (64 = total_area - areas i) ∨ (49 = total_area - areas i)) ↔
  (∃ (i : Fin 10), (areas i = 1) ∨ (areas i = 16)) :=
by
  sorry

end possible_lost_rectangle_area_l84_84672


namespace condition_swap_l84_84969

variable {p q : Prop}

theorem condition_swap (h : ¬ p → q) (nh : ¬ (¬ p ↔ q)) : (p → ¬ q) ∧ ¬ (¬ (p ↔ ¬ q)) :=
by
  sorry

end condition_swap_l84_84969


namespace isabela_spent_2800_l84_84390

/-- Given:
1. Isabela bought twice as many cucumbers as pencils.
2. Both cucumbers and pencils cost $20 each.
3. Isabela got a 20% discount on the pencils.
4. She bought 100 cucumbers.
Prove that the total amount Isabela spent is $2800. -/
theorem isabela_spent_2800 :
  ∀ (pencils cucumbers : ℕ) (pencil_cost cucumber_cost : ℤ) (discount rate: ℚ)
    (total_cost pencils_cost cucumbers_cost discount_amount : ℤ),
  cucumbers = 100 →
  pencils * 2 = cucumbers →
  pencil_cost = 20 →
  cucumber_cost = 20 →
  rate = 20 / 100 →
  pencils_cost = pencils * pencil_cost →
  discount_amount = pencils_cost * rate →
  total_cost = pencils_cost - discount_amount + cucumbers * cucumber_cost →
  total_cost = 2800 := by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end isabela_spent_2800_l84_84390


namespace pyramid_sphere_area_l84_84712

theorem pyramid_sphere_area (a : ℝ) (PA PB PC : ℝ) 
  (h1 : PA = PB) (h2 : PA = 2 * PC) 
  (h3 : PA = 2 * a) (h4 : PB = 2 * a) 
  (h5 : 4 * π * (PA^2 + PB^2 + PC^2) / 9 = 9 * π) :
  a = 1 :=
by
  sorry

end pyramid_sphere_area_l84_84712


namespace quadratic_to_vertex_form_addition_l84_84138

theorem quadratic_to_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intro h_eq
  sorry

end quadratic_to_vertex_form_addition_l84_84138


namespace tax_diminished_by_20_percent_l84_84815

theorem tax_diminished_by_20_percent
(T C : ℝ) 
(hT : T > 0) 
(hC : C > 0) 
(X : ℝ) 
(h_increased_consumption : ∀ (T C : ℝ), (C * 1.15) = C + 0.15 * C)
(h_decrease_revenue : T * (1 - X / 100) * C * 1.15 = T * C * 0.92) :
X = 20 := 
sorry

end tax_diminished_by_20_percent_l84_84815


namespace quadratic_solution_l84_84628

theorem quadratic_solution (x : ℝ) : x^2 - 5 * x - 6 = 0 ↔ (x = 6 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l84_84628


namespace norris_money_left_l84_84796

-- Define the amounts saved each month
def september_savings : ℕ := 29
def october_savings : ℕ := 25
def november_savings : ℕ := 31

-- Define the total savings
def total_savings : ℕ := september_savings + october_savings + november_savings

-- Define the amount spent on the online game
def amount_spent : ℕ := 75

-- Define the remaining money
def money_left : ℕ := total_savings - amount_spent

-- The theorem stating the problem and the solution
theorem norris_money_left : money_left = 10 := by
  sorry

end norris_money_left_l84_84796


namespace cos_150_degree_l84_84616

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l84_84616


namespace actual_distance_travelled_l84_84863

theorem actual_distance_travelled :
  ∃ (D : ℝ), (D / 10 = (D + 20) / 14) ∧ D = 50 :=
by
  sorry

end actual_distance_travelled_l84_84863


namespace sum_ages_l84_84101

theorem sum_ages (A_years B_years C_years : ℕ) (h1 : B_years = 30)
  (h2 : 10 * (B_years - 10) = (A_years - 10) * 2)
  (h3 : 10 * (B_years - 10) = (C_years - 10) * 3) :
  A_years + B_years + C_years = 90 :=
sorry

end sum_ages_l84_84101


namespace log_one_fifth_25_eq_neg2_l84_84267

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_fifth_25_eq_neg2 :
  log_base (1 / 5) 25 = -2 := by
 sorry

end log_one_fifth_25_eq_neg2_l84_84267


namespace prime_list_count_l84_84263

theorem prime_list_count {L : ℕ → ℕ} 
  (hL₀ : L 0 = 29)
  (hL : ∀ (n : ℕ), L (n + 1) = L n * 101 + L 0) :
  (∃! n, n = 0 ∧ Prime (L n)) ∧ ∀ m > 0, ¬ Prime (L m) := 
by
  sorry

end prime_list_count_l84_84263


namespace slope_of_line_l84_84663

-- Defining the conditions
def intersects_on_line (s x y : ℝ) : Prop :=
  (2 * x + 3 * y = 8 * s + 6) ∧ (x + 2 * y = 5 * s - 1)

-- Theorem stating that the slope of the line on which all intersections lie is 2
theorem slope_of_line {s x y : ℝ} :
  (∃ s x y, intersects_on_line s x y) → (∃ (m : ℝ), m = 2) :=
by sorry

end slope_of_line_l84_84663


namespace minimize_distance_l84_84161

noncomputable def f : ℝ → ℝ := λ x => x ^ 2
noncomputable def g : ℝ → ℝ := λ x => Real.log x
noncomputable def y : ℝ → ℝ := λ x => f x - g x

theorem minimize_distance (t : ℝ) (ht : t = Real.sqrt 2 / 2) :
  ∀ x > 0, y x ≥ y (Real.sqrt 2 / 2) := sorry

end minimize_distance_l84_84161


namespace weaving_sequence_l84_84013

-- Define the arithmetic sequence conditions
def day1_weaving := 5
def total_cloth := 390
def days := 30

-- Mathematical statement to be proved
theorem weaving_sequence : 
    ∃ d : ℚ, 30 * day1_weaving + (days * (days - 1) / 2) * d = total_cloth ∧ d = 16 / 29 :=
by 
  sorry

end weaving_sequence_l84_84013


namespace coins_after_tenth_hour_l84_84521

-- Given variables representing the number of coins added or removed each hour.
def coins_put_in : ℕ :=
  20 + 30 + 30 + 40 + 50 + 60 + 70

def coins_taken_out : ℕ :=
  20 + 15 + 25

-- Definition of the full proof problem
theorem coins_after_tenth_hour :
  coins_put_in - coins_taken_out = 240 :=
by
  sorry

end coins_after_tenth_hour_l84_84521


namespace arithmetic_sequence_solution_l84_84249

variable (a d : ℤ)
variable (n : ℕ)

/-- Given the following conditions:
1. The sum of the first three terms of an arithmetic sequence is -3.
2. The product of the first three terms is 8,
This theorem proves that:
1. The general term formula of the sequence is 3 * n - 7.
2. The sum of the first n terms is (3 / 2) * n ^ 2 - (11 / 2) * n.
-/
theorem arithmetic_sequence_solution
  (h1 : (a - d) + a + (a + d) = -3)
  (h2 : (a - d) * a * (a + d) = 8) :
  (∃ a d : ℤ, (∀ n : ℕ, (n ≥ 1) → (3 * n - 7 = a + (n - 1) * d) ∧ (∃ S : ℕ → ℤ, S n = (3 / 2) * n ^ 2 - (11 / 2) * n))) :=
by
  sorry

end arithmetic_sequence_solution_l84_84249


namespace solution_set_l84_84893

theorem solution_set (x : ℝ) : (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l84_84893


namespace sum_of_roots_of_quadratic_eq_l84_84219

theorem sum_of_roots_of_quadratic_eq :
  ∀ x : ℝ, x^2 + 2023 * x - 2024 = 0 → 
  x = -2023 := 
sorry

end sum_of_roots_of_quadratic_eq_l84_84219


namespace stockings_total_cost_l84_84092

-- Define a function to compute the total cost given the conditions
def total_cost (n_grandchildren n_children : Nat) 
               (stocking_price discount monogram_cost : Nat) : Nat :=
  let total_stockings := n_grandchildren + n_children
  let discounted_price := stocking_price - (stocking_price * discount / 100)
  let total_stockings_cost := discounted_price * total_stockings
  let total_monogram_cost := monogram_cost * total_stockings
  total_stockings_cost + total_monogram_cost

-- Prove that the total cost calculation is correct given the conditions
theorem stockings_total_cost :
  total_cost 5 4 20 10 5 = 207 :=
by
  -- A placeholder for the proof
  sorry

end stockings_total_cost_l84_84092


namespace max_value_f_diff_l84_84197

open Real

noncomputable def f (A ω : ℝ) (x : ℝ) := A * sin (ω * x + π / 6) - 1

theorem max_value_f_diff {A ω : ℝ} (hA : A > 0) (hω : ω > 0)
  (h_sym : (π / 2) = π / (2 * ω))
  (h_initial : f A ω (π / 6) = 1) :
  ∀ (x1 x2 : ℝ), (0 ≤ x1 ∧ x1 ≤ π / 2) ∧ (0 ≤ x2 ∧ x2 ≤ π / 2) →
  (f A ω x1 - f A ω x2 ≤ 3) :=
sorry

end max_value_f_diff_l84_84197


namespace terminating_decimal_l84_84366

theorem terminating_decimal : (47 : ℚ) / (2 * 5^4) = 376 / 10^4 :=
by sorry

end terminating_decimal_l84_84366


namespace remainder_8_pow_1996_mod_5_l84_84645

theorem remainder_8_pow_1996_mod_5 :
  (8: ℕ) ≡ 3 [MOD 5] →
  3^4 ≡ 1 [MOD 5] →
  8^1996 ≡ 1 [MOD 5] :=
by
  sorry

end remainder_8_pow_1996_mod_5_l84_84645


namespace marbles_left_l84_84720

theorem marbles_left (initial_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 64 → given_marbles = 14 → remaining_marbles = (initial_marbles - given_marbles) → remaining_marbles = 50 :=
by
  intros h_initial h_given h_calculation
  rw [h_initial, h_given] at h_calculation
  exact h_calculation

end marbles_left_l84_84720


namespace fractional_expression_evaluation_l84_84555

theorem fractional_expression_evaluation (a : ℝ) (h : a^3 + 3 * a^2 + a = 0) :
  ∃ b : ℝ, b = 0 ∨ b = 1 ∧ b = 2022 * a^2 / (a^4 + 2015 * a^2 + 1) :=
by
  sorry

end fractional_expression_evaluation_l84_84555


namespace scientific_notation_of_3933_billion_l84_84925

-- Definitions and conditions
def is_scientific_notation (a : ℝ) (n : ℤ) :=
  1 ≤ |a| ∧ |a| < 10 ∧ (39.33 * 10^9 = a * 10^n)

-- Theorem (statement only)
theorem scientific_notation_of_3933_billion : 
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n ∧ a = 3.933 ∧ n = 10 :=
by
  sorry

end scientific_notation_of_3933_billion_l84_84925


namespace tangent_line_of_f_eq_kx_l84_84291

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

theorem tangent_line_of_f_eq_kx (k : ℝ) : 
    (∃ x₀, tangent_line k x₀ = f x₀ ∧ deriv f x₀ = k) → 
    (k = 0 ∨ k = 1 ∨ k = -1) := 
  sorry

end tangent_line_of_f_eq_kx_l84_84291


namespace average_of_numbers_is_correct_l84_84041

theorem average_of_numbers_is_correct :
  let nums := [12, 13, 14, 510, 520, 530, 1120, 1, 1252140, 2345]
  let sum_nums := 1253205
  let count_nums := 10
  (sum_nums / count_nums.toFloat) = 125320.5 :=
by {
  sorry
}

end average_of_numbers_is_correct_l84_84041


namespace range_of_a_l84_84378

open Real

theorem range_of_a (a x y : ℝ)
  (h1 : (x - a) ^ 2 + (y - (a + 2)) ^ 2 = 1)
  (h2 : ∃ M : ℝ × ℝ, (M.1 - a) ^ 2 + (M.2 - (a + 2)) ^ 2 = 1
                       ∧ dist M (0, 3) = 2 * dist M (0, 0)) :
  -3 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l84_84378


namespace range_of_a_l84_84685

noncomputable def proof_problem (x : ℝ) (a : ℝ) : Prop :=
  (x^2 - 4*x + 3 < 0) ∧ (x^2 - 6*x + 8 < 0) → (2*x^2 - 9*x + a < 0)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, proof_problem x a) ↔ a ≤ 9 :=
by
  sorry

end range_of_a_l84_84685


namespace fifty_times_reciprocal_of_eight_times_number_three_l84_84154

theorem fifty_times_reciprocal_of_eight_times_number_three (x : ℚ) 
  (h : 8 * x = 3) : 50 * (1 / x) = 133 + 1 / 3 :=
sorry

end fifty_times_reciprocal_of_eight_times_number_three_l84_84154


namespace kimberly_loan_l84_84256

theorem kimberly_loan :
  ∃ (t : ℕ), (1.06 : ℝ)^t > 3 ∧ ∀ (t' : ℕ), t' < t → (1.06 : ℝ)^t' ≤ 3 :=
by
sorry

end kimberly_loan_l84_84256


namespace part1_part2_part3_l84_84603

noncomputable def f (a x : ℝ) : ℝ := Real.log (x + 1) + a * (x^2 - x)

theorem part1 (x : ℝ) (hx : 0 < x) : f 0 x < x := by sorry

theorem part2 (a x : ℝ) :
  (0 ≤ a ∧ a ≤ 8/9 → 0 = 0) ∧
  (a > 8/9 → 2 = 2) ∧
  (a < 0 → 1 = 1) := by sorry

theorem part3 (a : ℝ) (h : ∀ x > 0, f a x ≥ 0) : 0 ≤ a ∧ a ≤ 1 := by sorry

end part1_part2_part3_l84_84603


namespace find_a_l84_84331

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 :=
by
  sorry

end find_a_l84_84331


namespace volvox_pentagons_heptagons_diff_l84_84315

-- Given conditions
variables (V E F f_5 f_6 f_7 : ℕ)

-- Euler's polyhedron formula
axiom euler_formula : V - E + F = 2

-- Each edge is shared by two faces
axiom edge_formula : 2 * E = 5 * f_5 + 6 * f_6 + 7 * f_7

-- Each vertex shared by three faces
axiom vertex_formula : 3 * V = 5 * f_5 + 6 * f_6 + 7 * f_7

-- Total number of faces equals sum of individual face types 
def total_faces : ℕ := f_5 + f_6 + f_7

-- Prove that the number of pentagonal cells exceeds the number of heptagonal cells by 12
theorem volvox_pentagons_heptagons_diff : f_5 - f_7 = 12 := 
sorry

end volvox_pentagons_heptagons_diff_l84_84315


namespace inequality_proof_l84_84949

theorem inequality_proof (a b t : ℝ) (h₀ : 0 < t) (h₁ : t < 1) (h₂ : a * b > 0) : 
  (a^2 / t^3) + (b^2 / (1 - t^3)) ≥ (a + b)^2 :=
by
  sorry

end inequality_proof_l84_84949


namespace factors_of_m_multiples_of_200_l84_84643

theorem factors_of_m_multiples_of_200 (m : ℕ) (h : m = 2^12 * 3^10 * 5^9) : 
  (∃ k, 200 * k ≤ m ∧ ∃ a b c, k = 2^a * 3^b * 5^c ∧ 3 ≤ a ∧ a ≤ 12 ∧ 2 ≤ c ∧ c ≤ 9 ∧ 0 ≤ b ∧ b ≤ 10) := 
by sorry

end factors_of_m_multiples_of_200_l84_84643


namespace reduction_percentage_price_increase_l84_84107

-- Proof Problem 1: Reduction Percentage
theorem reduction_percentage (a : ℝ) (h₁ : (50 * (1 - a)^2 = 32)) : a = 0.2 := by
  sorry

-- Proof Problem 2: Price Increase for Daily Profit
theorem price_increase 
  (x : ℝ)
  (profit_per_kg : ℝ := 10)
  (initial_sales : ℕ := 500)
  (sales_decrease_per_unit : ℝ := 20)
  (required_profit : ℝ := 6000)
  (h₁ : (10 + x) * (initial_sales - sales_decrease_per_unit * x) = required_profit) : 
  x = 5 := by
  sorry

end reduction_percentage_price_increase_l84_84107


namespace average_of_rest_of_class_l84_84930

theorem average_of_rest_of_class
  (n : ℕ)
  (h1 : n > 0)
  (avg_class : ℝ := 84)
  (avg_one_fourth : ℝ := 96)
  (total_sum : ℝ := avg_class * n)
  (sum_one_fourth : ℝ := avg_one_fourth * (n / 4))
  (sum_rest : ℝ := total_sum - sum_one_fourth)
  (num_rest : ℝ := (3 * n) / 4) :
  sum_rest / num_rest = 80 :=
sorry

end average_of_rest_of_class_l84_84930


namespace find_a_plus_b_l84_84353

theorem find_a_plus_b :
  let A := {x : ℝ | -1 < x ∧ x < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  let S := {x : ℝ | -1 < x ∧ x < 2}
  ∃ (a b : ℝ), (∀ x, S x ↔ (x^2 + a * x + b < 0)) ∧ a + b = -3 :=
by
  sorry

end find_a_plus_b_l84_84353


namespace specific_time_l84_84563

theorem specific_time :
  (∀ (s : ℕ), 0 ≤ s ∧ s ≤ 7 → (∃ (t : ℕ), (t ^ 2 + 2 * t) - (3 ^ 2 + 2 * 3) = 20 ∧ t = 5)) :=
  by sorry

end specific_time_l84_84563


namespace complement_of_45_is_45_l84_84869

def angle_complement (A : Real) : Real :=
  90 - A

theorem complement_of_45_is_45:
  angle_complement 45 = 45 :=
by
  sorry

end complement_of_45_is_45_l84_84869


namespace minimum_buses_needed_l84_84695

theorem minimum_buses_needed (bus_capacity : ℕ) (students : ℕ) (h : bus_capacity = 38 ∧ students = 411) :
  ∃ n : ℕ, 38 * n ≥ students ∧ ∀ m : ℕ, 38 * m ≥ students → n ≤ m :=
by sorry

end minimum_buses_needed_l84_84695


namespace polygon_sides_l84_84926

theorem polygon_sides (n : ℕ) (h_interior : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_l84_84926


namespace number_of_flags_l84_84293

theorem number_of_flags (colors : Finset ℕ) (stripes : ℕ) (h_colors : colors.card = 3) (h_stripes : stripes = 3) : 
  (colors.card ^ stripes) = 27 := 
by
  sorry

end number_of_flags_l84_84293


namespace binom_1300_2_l84_84630

theorem binom_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_l84_84630


namespace probability_merlin_dismissed_l84_84955

-- Define the conditions
variables (p : ℝ) (q : ℝ) (hpq : p + q = 1) (hp_pos : 0 < p) (hq_pos : 0 < q)

/--
Given advisor Merlin is equally likely to dismiss as Percival
since they are equally likely to give the correct answer independently,
prove that the probability of Merlin being dismissed is \( \frac{1}{2} \).
-/
theorem probability_merlin_dismissed : (1/2 : ℝ) = 1/2 :=
by 
  sorry

end probability_merlin_dismissed_l84_84955


namespace equation_of_line_intersection_l84_84224

theorem equation_of_line_intersection
  (h1 : ∀ x y : ℝ, x^2 + y^2 = 1)
  (h2 : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0) :
  ∀ x y : ℝ, x - 2*y + 1 = 0 :=
by
  sorry

end equation_of_line_intersection_l84_84224


namespace gcd_between_35_and_7_l84_84091

theorem gcd_between_35_and_7 {n : ℕ} (h1 : 65 < n) (h2 : n < 75) (h3 : gcd 35 n = 7) : n = 70 := 
sorry

end gcd_between_35_and_7_l84_84091


namespace ratio_of_sums_l84_84014

theorem ratio_of_sums (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : a^2 + b^2 + c^2 = 49) 
  (h2 : x^2 + y^2 + z^2 = 64) 
  (h3 : a * x + b * y + c * z = 56) : 
  (a + b + c) / (x + y + z) = 7/8 := 
by 
  sorry

end ratio_of_sums_l84_84014


namespace percentage_seeds_germinated_l84_84432

theorem percentage_seeds_germinated :
  let S1 := 300
  let S2 := 200
  let S3 := 150
  let S4 := 250
  let S5 := 100
  let G1 := 0.20
  let G2 := 0.35
  let G3 := 0.45
  let G4 := 0.25
  let G5 := 0.60
  (G1 * S1 + G2 * S2 + G3 * S3 + G4 * S4 + G5 * S5) / (S1 + S2 + S3 + S4 + S5) * 100 = 32 := 
by
  sorry

end percentage_seeds_germinated_l84_84432


namespace apples_handed_out_to_students_l84_84388

def initial_apples : ℕ := 47
def apples_per_pie : ℕ := 4
def number_of_pies : ℕ := 5
def apples_for_pies : ℕ := number_of_pies * apples_per_pie

theorem apples_handed_out_to_students : 
  initial_apples - apples_for_pies = 27 := 
by
  -- Since 20 apples are used for pies and there were initially 47 apples,
  -- it follows that 27 apples were handed out to students.
  sorry

end apples_handed_out_to_students_l84_84388


namespace time_reduced_fraction_l84_84458

theorem time_reduced_fraction 
  (S : ℝ) (hs : S = 24.000000000000007) 
  (D : ℝ) : 
  1 - (D / (S + 12) / (D / S)) = 1 / 3 :=
by sorry

end time_reduced_fraction_l84_84458


namespace arithmetic_sequence_ninth_term_l84_84028

theorem arithmetic_sequence_ninth_term 
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29)
  : a + 8 * d = 35 :=
by
  sorry

end arithmetic_sequence_ninth_term_l84_84028


namespace non_empty_solution_set_range_l84_84504

theorem non_empty_solution_set_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| - |x - 1| < a) → a > -3 :=
sorry

end non_empty_solution_set_range_l84_84504


namespace find_m_l84_84602

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d 

noncomputable def sum_first_n_terms (a S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

theorem find_m {a S : ℕ → ℤ} (d : ℤ) (m : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : a 1 = 1)
  (h4 : S 3 = a 5)
  (h5 : a m = 2011) :
  m = 1006 :=
sorry

end find_m_l84_84602


namespace hyperbola_focal_product_l84_84596

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

end hyperbola_focal_product_l84_84596


namespace gcd_64_144_l84_84396

theorem gcd_64_144 : Nat.gcd 64 144 = 16 := by
  sorry

end gcd_64_144_l84_84396


namespace arithmetic_avg_salary_technicians_l84_84727

noncomputable def avg_salary_technicians_problem : Prop :=
  let average_salary_all := 8000
  let total_workers := 21
  let average_salary_rest := 6000
  let technician_count := 7
  let total_salary_all := average_salary_all * total_workers
  let total_salary_rest := average_salary_rest * (total_workers - technician_count)
  let total_salary_technicians := total_salary_all - total_salary_rest
  let average_salary_technicians := total_salary_technicians / technician_count
  average_salary_technicians = 12000

theorem arithmetic_avg_salary_technicians :
  avg_salary_technicians_problem :=
by {
  sorry -- Proof is omitted as per instructions.
}

end arithmetic_avg_salary_technicians_l84_84727


namespace solveCubicEquation_l84_84952

-- Define the condition as a hypothesis
def equationCondition (x : ℝ) : Prop := (7 - x)^(1/3) = -5/3

-- State the theorem to be proved
theorem solveCubicEquation : ∃ x : ℝ, equationCondition x ∧ x = 314 / 27 :=
by 
  sorry

end solveCubicEquation_l84_84952


namespace solution_is_111_l84_84286

-- Define the system of equations
def system_of_equations (x y z : ℝ) :=
  (x^2 + 7 * y + 2 = 2 * z + 4 * Real.sqrt (7 * x - 3)) ∧
  (y^2 + 7 * z + 2 = 2 * x + 4 * Real.sqrt (7 * y - 3)) ∧
  (z^2 + 7 * x + 2 = 2 * y + 4 * Real.sqrt (7 * z - 3))

-- Prove that x = 1, y = 1, z = 1 is a solution to the system of equations
theorem solution_is_111 : system_of_equations 1 1 1 :=
by
  sorry

end solution_is_111_l84_84286


namespace range_a_l84_84004

theorem range_a (a : ℝ) :
  (∀ x : ℝ, a ≤ x ∧ x ≤ 3 → -1 ≤ -x^2 + 2 * x + 2 ∧ -x^2 + 2 * x + 2 ≤ 3) →
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l84_84004


namespace triangle_perimeter_l84_84654

-- Definitions for the conditions
def side_length1 : ℕ := 3
def side_length2 : ℕ := 6
def equation (x : ℤ) := x^2 - 6 * x + 8 = 0

-- Perimeter calculation given the sides form a triangle
theorem triangle_perimeter (x : ℤ) (h₁ : equation x) (h₂ : 3 + 6 > x) (h₃ : 3 + x > 6) (h₄ : 6 + x > 3) :
  3 + 6 + x = 13 :=
by sorry

end triangle_perimeter_l84_84654


namespace wide_flags_made_l84_84580

theorem wide_flags_made
  (initial_fabric : ℕ) (square_flag_side : ℕ) (wide_flag_width : ℕ) (wide_flag_height : ℕ)
  (tall_flag_width : ℕ) (tall_flag_height : ℕ) (made_square_flags : ℕ) (made_tall_flags : ℕ)
  (remaining_fabric : ℕ) (used_fabric_for_small_flags : ℕ) (used_fabric_for_tall_flags : ℕ)
  (used_fabric_for_wide_flags : ℕ) (wide_flag_area : ℕ) :
    initial_fabric = 1000 →
    square_flag_side = 4 →
    wide_flag_width = 5 →
    wide_flag_height = 3 →
    tall_flag_width = 3 →
    tall_flag_height = 5 →
    made_square_flags = 16 →
    made_tall_flags = 10 →
    remaining_fabric = 294 →
    used_fabric_for_small_flags = 256 →
    used_fabric_for_tall_flags = 150 →
    used_fabric_for_wide_flags = initial_fabric - remaining_fabric - (used_fabric_for_small_flags + used_fabric_for_tall_flags) →
    wide_flag_area = wide_flag_width * wide_flag_height →
    (used_fabric_for_wide_flags / wide_flag_area) = 20 :=
by
  intros; 
  sorry

end wide_flags_made_l84_84580


namespace original_number_increased_by_110_l84_84401

-- Define the conditions and the proof statement without the solution steps
theorem original_number_increased_by_110 {x : ℝ} (h : x + 1.10 * x = 1680) : x = 800 :=
by 
  sorry

end original_number_increased_by_110_l84_84401


namespace machines_work_together_l84_84957

theorem machines_work_together (x : ℝ) (h_pos : 0 < x) :
  (1 / (x + 2) + 1 / (x + 3) + 1 / (x + 1) = 1 / x) → x = 1 :=
by
  sorry

end machines_work_together_l84_84957


namespace championship_outcome_count_l84_84828

theorem championship_outcome_count (students championships : ℕ) (h_students : students = 8) (h_championships : championships = 3) : students ^ championships = 512 := by
  rw [h_students, h_championships]
  norm_num

end championship_outcome_count_l84_84828


namespace a_parallel_b_l84_84299

variable {Line : Type} (a b c : Line)

-- Definition of parallel lines
def parallel (x y : Line) : Prop := sorry

-- Conditions
axiom a_parallel_c : parallel a c
axiom b_parallel_c : parallel b c

-- Theorem to prove a is parallel to b given the conditions
theorem a_parallel_b : parallel a b :=
by
  sorry

end a_parallel_b_l84_84299


namespace combined_ratio_is_1_l84_84650

-- Conditions
variables (V1 V2 M1 W1 M2 W2 : ℝ)
variables (x : ℝ)
variables (ratio_volumes ratio_milk_water_v1 ratio_milk_water_v2 : ℝ)

-- Given conditions as hypotheses
-- Condition: V1 / V2 = 3 / 5
-- Hypothesis 1: The volume ratio of the first and second vessels
def volume_ratio : Prop :=
  V1 / V2 = 3 / 5

-- Condition: M1 / W1 = 1 / 2 in first vessel
-- Hypothesis 2: The milk to water ratio in the first vessel
def milk_water_ratio_v1 : Prop :=
  M1 / W1 = 1 / 2

-- Condition: M2 / W2 = 3 / 2 in the second vessel
-- Hypothesis 3: The milk to water ratio in the second vessel
def milk_water_ratio_v2 : Prop :=
  M2 / W2 = 3 / 2

-- Definition: Total volumes of milk and water in the larger vessel
def total_milk_water_ratio : Prop :=
  (M1 + M2) / (W1 + W2) = 1 / 1

-- Main theorem: Given the ratios, the ratio of milk to water in the larger vessel is 1:1
theorem combined_ratio_is_1 :
  (volume_ratio V1 V2) →
  (milk_water_ratio_v1 M1 W1) →
  (milk_water_ratio_v2 M2 W2) →
  total_milk_water_ratio M1 W1 M2 W2 :=
by
  -- Proof omitted
  sorry

end combined_ratio_is_1_l84_84650


namespace solve_for_real_a_l84_84033

theorem solve_for_real_a (a : ℝ) (i : ℂ) (h : i^2 = -1) (h1 : (a - i)^2 = 2 * i) : a = -1 :=
by sorry

end solve_for_real_a_l84_84033


namespace least_possible_number_of_straight_lines_l84_84946

theorem least_possible_number_of_straight_lines :
  ∀ (segments : Fin 31 → (Fin 2 → ℝ)), 
  (∀ i j, i ≠ j → (segments i 0 = segments j 0) ∧ (segments i 1 = segments j 1) → false) →
  ∃ (lines_count : ℕ), lines_count = 16 :=
by
  sorry

end least_possible_number_of_straight_lines_l84_84946


namespace eval_expression_l84_84163

theorem eval_expression (a : ℕ) (h : a = 2) : 
  8^3 + 4 * a * 8^2 + 6 * a^2 * 8 + a^3 = 1224 := 
by
  rw [h]
  sorry

end eval_expression_l84_84163


namespace sum_of_abcd_l84_84644

theorem sum_of_abcd (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : ∀ x, x^2 - 8*a*x - 9*b = 0 → x = c ∨ x = d)
  (h2 : ∀ x, x^2 - 8*c*x - 9*d = 0 → x = a ∨ x = b) :
  a + b + c + d = 648 := sorry

end sum_of_abcd_l84_84644


namespace tenth_term_of_sequence_l84_84117

variable (a : ℕ → ℚ) (n : ℕ)

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n-1)

theorem tenth_term_of_sequence :
  let a₁ := (5 : ℚ)
  let r := (5 / 3 : ℚ)
  geometric_sequence a₁ r 10 = (9765625 / 19683 : ℚ) :=
by
  sorry

end tenth_term_of_sequence_l84_84117


namespace total_points_other_five_l84_84198

theorem total_points_other_five
  (x : ℕ) -- total number of points scored by the team
  (d : ℕ) (e : ℕ) (f : ℕ) (y : ℕ) -- points scored by Daniel, Emma, Fiona, and others respectively
  (hd : d = x / 3) -- Daniel scored 1/3 of the team's points
  (he : e = 3 * x / 8) -- Emma scored 3/8 of the team's points
  (hf : f = 18) -- Fiona scored 18 points
  (h_other : ∀ i, 1 ≤ i ∧ i ≤ 5 → y ≤ 15 / 5) -- Other 5 members scored no more than 3 points each
  (h_total : d + e + f + y = x) -- Total points equation
  : y = 14 := sorry -- Final number of points scored by the other 5 members

end total_points_other_five_l84_84198


namespace common_fraction_l84_84535

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end common_fraction_l84_84535


namespace fraction_subtraction_simplified_l84_84528

theorem fraction_subtraction_simplified : (7 / 17) - (4 / 51) = 1 / 3 := by
  sorry

end fraction_subtraction_simplified_l84_84528


namespace ellipse_foci_coordinates_l84_84399

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ),
  (x^2 / 9 + y^2 / 5 = 1) →
  (x, y) = (2, 0) ∨ (x, y) = (-2, 0) :=
by
  sorry

end ellipse_foci_coordinates_l84_84399


namespace picnic_recyclable_collected_l84_84536

theorem picnic_recyclable_collected :
  let guests := 90
  let soda_cans := 50
  let sparkling_water_bottles := 50
  let juice_bottles := 50
  let soda_drinkers := guests / 2
  let sparkling_water_drinkers := guests / 3
  let juice_consumed := juice_bottles * 4 / 5 
  soda_drinkers + sparkling_water_drinkers + juice_consumed = 115 :=
by
  let guests := 90
  let soda_cans := 50
  let sparkling_water_bottles := 50
  let juice_bottles := 50
  let soda_drinkers := guests / 2
  let sparkling_water_drinkers := guests / 3
  let juice_consumed := juice_bottles * 4 / 5 
  show soda_drinkers + sparkling_water_drinkers + juice_consumed = 115
  sorry

end picnic_recyclable_collected_l84_84536


namespace g_50_unique_l84_84526

namespace Proof

-- Define the function g and the condition it should satisfy
variable (g : ℕ → ℕ)
variable (h : ∀ (a b : ℕ), 3 * g (a^2 + b^2) = g a * g b + 2 * (g a + g b))

theorem g_50_unique : ∃ (m t : ℕ), m * t = 0 := by
  -- Existence of m and t fulfilling the condition
  -- Placeholder for the proof
  sorry

end Proof

end g_50_unique_l84_84526


namespace gcd_2pow_2025_minus_1_2pow_2016_minus_1_l84_84400

theorem gcd_2pow_2025_minus_1_2pow_2016_minus_1 :
  Nat.gcd (2^2025 - 1) (2^2016 - 1) = 511 :=
by sorry

end gcd_2pow_2025_minus_1_2pow_2016_minus_1_l84_84400


namespace line_y_axis_intersect_l84_84067

theorem line_y_axis_intersect (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3 ∧ y1 = 27) (h2 : x2 = -7 ∧ y2 = -1) :
  ∃ y : ℝ, (∀ x : ℝ, y = (y2 - y1) / (x2 - x1) * (x - x1) + y1) ∧ y = 18.6 :=
by
  sorry

end line_y_axis_intersect_l84_84067


namespace rain_third_day_l84_84199

theorem rain_third_day (rain_day1 rain_day2 rain_day3 : ℕ)
  (h1 : rain_day1 = 4)
  (h2 : rain_day2 = 5 * rain_day1)
  (h3 : rain_day3 = (rain_day1 + rain_day2) - 6) : 
  rain_day3 = 18 := 
by
  -- Proof omitted
  sorry

end rain_third_day_l84_84199


namespace alcohol_quantity_l84_84841

theorem alcohol_quantity (A W : ℝ) (h1 : A / W = 4 / 3) (h2 : A / (W + 8) = 4 / 5) : A = 16 := 
by
  sorry

end alcohol_quantity_l84_84841


namespace ratio_of_numbers_l84_84271

theorem ratio_of_numbers (x : ℝ) (h_sum : x + 3.5 = 14) : x / 3.5 = 3 :=
by
  sorry

end ratio_of_numbers_l84_84271


namespace sparse_real_nums_l84_84382

noncomputable def is_sparse (r : ℝ) : Prop :=
  ∃n > 0, ∀s : ℝ, s^n = r → s = 1 ∨ s = -1 ∨ s = 0

theorem sparse_real_nums (r : ℝ) : is_sparse r ↔ r = -1 ∨ r = 0 ∨ r = 1 := 
by
  sorry

end sparse_real_nums_l84_84382


namespace club_additional_members_l84_84262

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) :
  current_members = 10 →
  additional_members = (2 * current_members) + 5 - current_members →
  additional_members = 15 :=
by
  intro h1 h2
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l84_84262


namespace angle_CBD_is_10_degrees_l84_84008

theorem angle_CBD_is_10_degrees (angle_ABC angle_ABD : ℝ) (h1 : angle_ABC = 40) (h2 : angle_ABD = 30) :
  angle_ABC - angle_ABD = 10 :=
by
  sorry

end angle_CBD_is_10_degrees_l84_84008


namespace rectangular_to_polar_l84_84801

theorem rectangular_to_polar : 
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (3 * Real.sqrt 2, 7 * Real.pi / 4) := 
by
  sorry

end rectangular_to_polar_l84_84801


namespace tan_half_alpha_eq_one_third_l84_84213

open Real

theorem tan_half_alpha_eq_one_third (α : ℝ) (h1 : 5 * sin (2 * α) = 6 * cos α) (h2 : 0 < α ∧ α < π / 2) :
  tan (α / 2) = 1 / 3 :=
by
  sorry

end tan_half_alpha_eq_one_third_l84_84213


namespace selection_methods_count_l84_84052

/-- Consider a school with 16 teachers, divided into four departments (First grade, Second grade, Third grade, and Administrative department), with 4 teachers each. 
We need to select 3 leaders such that not all leaders are from the same department and at least one leader is from the Administrative department. 
Prove that the number of different selection methods that satisfy these conditions is 336. -/
theorem selection_methods_count :
  let num_teachers := 16
  let teachers_per_department := 4
  ∃ (choose : ℕ → ℕ → ℕ), 
  choose num_teachers 3 = 336 :=
  sorry

end selection_methods_count_l84_84052


namespace square_side_length_l84_84725

noncomputable def diagonal_in_inches : ℝ := 2 * Real.sqrt 2
noncomputable def inches_to_feet : ℝ := 1 / 12
noncomputable def diagonal_in_feet := diagonal_in_inches * inches_to_feet
noncomputable def factor_sqrt_2 : ℝ := 1 / Real.sqrt 2

theorem square_side_length :
  let diagonal_feet := diagonal_in_feet 
  let side_length_feet := diagonal_feet * factor_sqrt_2
  side_length_feet = 1 / 6 :=
sorry

end square_side_length_l84_84725


namespace num_factors_of_M_l84_84050

theorem num_factors_of_M (M : ℕ) 
  (hM : M = (2^5) * (3^4) * (5^3) * (11^2)) : ∃ n : ℕ, n = 360 ∧ M = (2^5) * (3^4) * (5^3) * (11^2) := 
by
  sorry

end num_factors_of_M_l84_84050


namespace power_function_propositions_l84_84493

theorem power_function_propositions : (∀ n : ℤ, n > 0 → ∀ x : ℝ, x > 0 → (x^n) < x) ∧
  (∀ n : ℤ, n < 0 → ∀ x : ℝ, x > 0 → (x^n) > x) :=
by
  sorry

end power_function_propositions_l84_84493


namespace shifted_parabola_eq_l84_84810

theorem shifted_parabola_eq :
  ∀ x, (∃ y, y = 2 * (x - 3)^2 + 2) →
       (∃ y, y = 2 * (x + 0)^2 + 4) :=
by sorry

end shifted_parabola_eq_l84_84810


namespace feet_count_l84_84020

-- We define the basic quantities
def total_heads : ℕ := 50
def num_hens : ℕ := 30
def num_cows : ℕ := total_heads - num_hens
def hens_feet : ℕ := num_hens * 2
def cows_feet : ℕ := num_cows * 4
def total_feet : ℕ := hens_feet + cows_feet

-- The theorem we want to prove
theorem feet_count : total_feet = 140 :=
  by
  sorry

end feet_count_l84_84020


namespace meaningful_expression_range_l84_84624

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by 
  sorry

end meaningful_expression_range_l84_84624


namespace correct_propositions_l84_84446

variables (a b : ℝ) (x : ℝ) (a_max : ℝ)

/-- Given propositions to analyze. -/
noncomputable def propositions :=
  ((a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3) ∧
  ((¬ ∀ x : ℝ, x^2 + x - 2 > 0) ↔ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ∧
  (a_max = 2 ∧ ∀ x > 0, x + 1/x ≥ a_max))

/-- The main theorem stating which propositions are correct -/
theorem correct_propositions (h1 : a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3)
                            (h2 : (¬ ∀ x : ℝ, x^2 + x - 2 > 0) ↔ ∃ x : ℝ, x^2 + x - 2 ≤ 0)
                            (h3 : a_max = 2 ∧ ∀ x > 0, x + 1/x ≥ a_max) :
  propositions a b a_max :=
by
  sorry

end correct_propositions_l84_84446


namespace periodic_odd_function_example_l84_84799

open Real

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem periodic_odd_function_example (f : ℝ → ℝ) 
  (h_odd : odd f) 
  (h_periodic : periodic f 2) : 
  f 1 + f 4 + f 7 = 0 := 
sorry

end periodic_odd_function_example_l84_84799


namespace cuboid_unshaded_face_area_l84_84492

theorem cuboid_unshaded_face_area 
  (x : ℝ)
  (h1 : ∀ a  : ℝ, a = 4*x) -- Condition: each unshaded face area = 4 * shaded face area
  (h2 : 18*x = 72)         -- Condition: total surface area = 72 cm²
  : 4*x = 16 :=            -- Conclusion: area of one visible unshaded face is 16 cm²
by
  sorry

end cuboid_unshaded_face_area_l84_84492


namespace carlos_local_tax_deduction_l84_84693

theorem carlos_local_tax_deduction :
  let hourly_wage_dollars := 25
  let hourly_wage_cents := hourly_wage_dollars * 100
  let tax_rate := 2.5 / 100
  hourly_wage_cents * tax_rate = 62.5 :=
by
  sorry

end carlos_local_tax_deduction_l84_84693


namespace bathroom_cleaning_time_ratio_l84_84099

noncomputable def hourlyRate : ℝ := 5
noncomputable def vacuumingHours : ℝ := 2 -- per session
noncomputable def vacuumingSessions : ℕ := 2
noncomputable def washingDishesTime : ℝ := 0.5
noncomputable def totalEarnings : ℝ := 30

theorem bathroom_cleaning_time_ratio :
  let vacuumingEarnings := vacuumingHours * vacuumingSessions * hourlyRate
  let washingDishesEarnings := washingDishesTime * hourlyRate
  let knownEarnings := vacuumingEarnings + washingDishesEarnings
  let bathroomEarnings := totalEarnings - knownEarnings
  let bathroomCleaningTime := bathroomEarnings / hourlyRate
  bathroomCleaningTime / washingDishesTime = 3 := 
by
  sorry

end bathroom_cleaning_time_ratio_l84_84099


namespace num_pos_pairs_l84_84813

theorem num_pos_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 + 3 * n < 40) :
  ∃ k : ℕ, k = 45 :=
by {
  -- Additional setup and configuration if needed
  -- ...
  sorry
}

end num_pos_pairs_l84_84813


namespace sticks_predict_good_fortune_l84_84537

def good_fortune_probability := 11 / 12

theorem sticks_predict_good_fortune:
  (∃ (α β: ℝ), 0 ≤ α ∧ α ≤ π / 2 ∧ 0 ≤ β ∧ β ≤ π / 2 ∧ (0 ≤ β ∧ β < π - α) ∧ (0 ≤ α ∧ α < π - β)) → 
  good_fortune_probability = 11 / 12 :=
sorry

end sticks_predict_good_fortune_l84_84537


namespace perfect_square_is_289_l84_84953

/-- The teacher tells a three-digit perfect square number by
revealing the hundreds digit to person A, the tens digit to person B,
and the units digit to person C, and tells them that all three digits
are different from each other. Each person only knows their own digit and
not the others. The three people have the following conversation:

Person A: I don't know what the perfect square number is.  
Person B: You don't need to say; I also know that you don't know.  
Person C: I already know what the number is.  
Person A: After hearing Person C, I also know what the number is.  
Person B: After hearing Person A also knows what the number is.

Given these conditions, the three-digit perfect square number is 289. -/
theorem perfect_square_is_289:
  ∃ n : ℕ, n^2 = 289 := by
  sorry

end perfect_square_is_289_l84_84953


namespace regular_polygon_sides_l84_84056

theorem regular_polygon_sides (h : ∀ n : ℕ, n ≥ 3 → (total_internal_angle_sum / n) = 150) :
    n = 12 := by
  sorry

end regular_polygon_sides_l84_84056


namespace grandchildren_ages_l84_84456

theorem grandchildren_ages (x : ℕ) (y : ℕ) :
  (x + y = 30) →
  (5 * (x * (x + 1) + (30 - x) * (31 - x)) = 2410) →
  (x = 16 ∧ y = 14) ∨ (x = 14 ∧ y = 16) :=
by
  sorry

end grandchildren_ages_l84_84456


namespace spending_50_dollars_l84_84928

def receiving_money (r : Int) : Prop := r > 0

def spending_money (s : Int) : Prop := s < 0

theorem spending_50_dollars :
  receiving_money 80 ∧ ∀ r, receiving_money r → spending_money (-r)
  → spending_money (-50) :=
by
  sorry

end spending_50_dollars_l84_84928


namespace factorization_25x2_minus_155x_minus_150_l84_84786

theorem factorization_25x2_minus_155x_minus_150 :
  ∃ (a b : ℤ), (a + b) * 5 = -155 ∧ a * b = -150 ∧ a + 2 * b = 27 :=
by
  sorry

end factorization_25x2_minus_155x_minus_150_l84_84786


namespace sofia_total_time_for_5_laps_sofia_total_time_in_minutes_and_seconds_l84_84166

noncomputable def calculate_time (distance1 distance2 speed1 speed2 : ℕ) : ℕ := 
  (distance1 / speed1) + (distance2 / speed2)

noncomputable def total_time_per_lap := calculate_time 200 100 4 6

theorem sofia_total_time_for_5_laps : total_time_per_lap * 5 = 335 := 
  by sorry

def converted_time (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / 60, total_seconds % 60)

theorem sofia_total_time_in_minutes_and_seconds :
  converted_time (total_time_per_lap * 5) = (5, 35) :=
  by sorry

end sofia_total_time_for_5_laps_sofia_total_time_in_minutes_and_seconds_l84_84166


namespace tangent_line_computation_l84_84260

variables (f : ℝ → ℝ)

theorem tangent_line_computation (h_tangent : ∀ x, (f x = -x + 8) ∧ (∃ y, y = -x + 8 → (f y) = -x + 8 → deriv f x = -1)) :
    f 5 + deriv f 5 = 2 :=
sorry

end tangent_line_computation_l84_84260


namespace number_of_boys_l84_84853

def initial_girls : ℕ := 706
def new_girls : ℕ := 418
def total_pupils : ℕ := 1346
def total_girls := initial_girls + new_girls

theorem number_of_boys : 
  total_pupils = total_girls + 222 := 
by
  sorry

end number_of_boys_l84_84853


namespace ratio_value_l84_84221

theorem ratio_value (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := 
by
  sorry

end ratio_value_l84_84221


namespace ratio_implies_sum_ratio_l84_84310

theorem ratio_implies_sum_ratio (x y : ℝ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end ratio_implies_sum_ratio_l84_84310


namespace find_f2_l84_84329

noncomputable def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 0) : f 2 a b = -16 :=
by {
  sorry
}

end find_f2_l84_84329


namespace product_roots_l84_84665

noncomputable def root1 (x1 : ℝ) : Prop := x1 * Real.log x1 = 2006
noncomputable def root2 (x2 : ℝ) : Prop := x2 * Real.exp x2 = 2006

theorem product_roots (x1 x2 : ℝ) (h1 : root1 x1) (h2 : root2 x2) : x1 * x2 = 2006 := sorry

end product_roots_l84_84665


namespace min_value_of_expression_l84_84694

variable (a b c : ℝ)
variable (h1 : a + b + c = 1)
variable (h2 : 0 < a ∧ a < 1)
variable (h3 : 0 < b ∧ b < 1)
variable (h4 : 0 < c ∧ c < 1)
variable (h5 : 3 * a + 2 * b = 2)

theorem min_value_of_expression : (2 / a + 1 / (3 * b)) ≥ 16 / 3 := 
  sorry

end min_value_of_expression_l84_84694


namespace press_x_squared_three_times_to_exceed_10000_l84_84473

theorem press_x_squared_three_times_to_exceed_10000 :
  ∃ (n : ℕ), n = 3 ∧ (5^(2^n) > 10000) :=
by
  sorry

end press_x_squared_three_times_to_exceed_10000_l84_84473


namespace problem_part1_and_part2_l84_84703

noncomputable def g (x a b : ℝ) : ℝ := a * Real.log x + 0.5 * x ^ 2 + (1 - b) * x

-- Given: the function definition and conditions
variables (a b : ℝ)
variables (x1 x2 : ℝ)
variables (hx1 : x1 ∈ Set.Ioi 0) (hx2 : x2 ∈ Set.Ioi 0)
variables (h_tangent : 8 * 1 - 2 * g 1 a b - 3 = 0)
variables (h_extremes : b = a + 1)

-- Prove the values of a and b as well as the inequality
theorem problem_part1_and_part2 :
  (a = 1 ∧ b = -1) ∧ (g x1 a b + g x2 a b < -4) :=
sorry

end problem_part1_and_part2_l84_84703


namespace average_squares_of_first_10_multiples_of_7_correct_l84_84988

def first_10_multiples_of_7 : List ℕ := List.map (fun n => 7 * n) (List.range 10)

def squares (l : List ℕ) : List ℕ := List.map (fun n => n * n) l

def sum (l : List ℕ) : ℕ := List.foldr (· + ·) 0 l

theorem average_squares_of_first_10_multiples_of_7_correct :
  (sum (squares first_10_multiples_of_7) / 10 : ℚ) = 1686.5 :=
by
  sorry

end average_squares_of_first_10_multiples_of_7_correct_l84_84988


namespace speed_of_first_train_is_correct_l84_84825

-- Define the lengths of the trains
def length_train1 : ℕ := 110
def length_train2 : ℕ := 200

-- Define the speed of the second train in kmph
def speed_train2 : ℕ := 65

-- Define the time they take to clear each other in seconds
def time_clear_seconds : ℚ := 7.695936049253991

-- Define the speed of the first train
def speed_train1 : ℚ :=
  let time_clear_hours : ℚ := time_clear_seconds / 3600
  let total_distance_km : ℚ := (length_train1 + length_train2) / 1000
  let relative_speed_kmph : ℚ := total_distance_km / time_clear_hours 
  relative_speed_kmph - speed_train2

-- The proof problem is to show that the speed of the first train is 80.069 kmph
theorem speed_of_first_train_is_correct : speed_train1 = 80.069 := by
  sorry

end speed_of_first_train_is_correct_l84_84825


namespace algebraic_expression_value_l84_84342

theorem algebraic_expression_value 
  (θ : ℝ)
  (a := (Real.cos θ, Real.sin θ))
  (b := (1, -2))
  (parallel : ∃ k : ℝ, a = (k * 1, k * -2)) :
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 := 
by 
  -- proof goes here 
  sorry

end algebraic_expression_value_l84_84342


namespace rhombus_area_l84_84404

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 30) (h2 : d2 = 12) : (d1 * d2) / 2 = 180 :=
by
  sorry

end rhombus_area_l84_84404


namespace absolute_sum_of_coefficients_l84_84590

theorem absolute_sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) :
  (2 - x)^6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_0 = 2^6 →
  a_0 > 0 ∧ a_2 > 0 ∧ a_4 > 0 ∧ a_6 > 0 ∧
  a_1 < 0 ∧ a_3 < 0 ∧ a_5 < 0 → 
  |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 665 :=
by sorry

end absolute_sum_of_coefficients_l84_84590


namespace time_spent_on_marketing_posts_l84_84464

-- Bryan's conditions
def hours_customer_outreach : ℕ := 4
def hours_advertisement : ℕ := hours_customer_outreach / 2
def total_hours_worked : ℕ := 8

-- Proof statement: Bryan spends 2 hours each day on marketing posts
theorem time_spent_on_marketing_posts : 
  total_hours_worked - (hours_customer_outreach + hours_advertisement) = 2 := by
  sorry

end time_spent_on_marketing_posts_l84_84464


namespace x_value_l84_84455

theorem x_value :
  ∀ (x y : ℝ), x = y - 0.1 * y ∧ y = 125 + 0.1 * 125 → x = 123.75 :=
by
  intros x y h
  sorry

end x_value_l84_84455


namespace find_q_l84_84457

def polynomial_q (x p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_q (p q r : ℝ) (h₀ : r = 3)
  (h₁ : (-p / 3) = -r)
  (h₂ : (-r) = 1 + p + q + r) :
  q = -16 :=
by
  -- h₀ implies r = 3
  -- h₁ becomes (-p / 3) = -3
  -- which results in p = 9
  -- h₂ becomes -3 = 1 + 9 + q + 3
  -- leading to q = -16
  sorry

end find_q_l84_84457


namespace circle_radius_l84_84454

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 120 * π) : r = 10 :=
sorry

end circle_radius_l84_84454


namespace problem1_problem2_l84_84632

theorem problem1 : (Real.sqrt 24 - Real.sqrt 18) - Real.sqrt 6 = Real.sqrt 6 - 3 * Real.sqrt 2 := by
  sorry

theorem problem2 : 2 * Real.sqrt 12 * Real.sqrt (1 / 8) + 5 * Real.sqrt 2 = Real.sqrt 6 + 5 * Real.sqrt 2 := by
  sorry

end problem1_problem2_l84_84632


namespace dodecahedron_edges_l84_84759

noncomputable def regular_dodecahedron := Type

def faces : regular_dodecahedron → ℕ := λ _ => 12
def edges_per_face : regular_dodecahedron → ℕ := λ _ => 5
def shared_edges : regular_dodecahedron → ℕ := λ _ => 2

theorem dodecahedron_edges (d : regular_dodecahedron) :
  (faces d * edges_per_face d) / shared_edges d = 30 :=
by
  sorry

end dodecahedron_edges_l84_84759


namespace min_value_of_vectors_l84_84201

theorem min_value_of_vectors (m n : ℝ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : (m * (n - 2)) + 1 = 0) : (1 / m) + (2 / n) = 2 * Real.sqrt 2 + 3 / 2 :=
by sorry

end min_value_of_vectors_l84_84201


namespace ratio_of_time_l84_84854

theorem ratio_of_time (T_A T_B : ℝ) (h1 : T_A = 8) (h2 : 1 / T_A + 1 / T_B = 0.375) :
  T_B / T_A = 1 / 2 :=
by 
  sorry

end ratio_of_time_l84_84854


namespace initial_butterfly_count_l84_84936

theorem initial_butterfly_count (n : ℕ) (h : (2 / 3 : ℚ) * n = 6) : n = 9 :=
sorry

end initial_butterfly_count_l84_84936


namespace find_dads_dimes_l84_84919

variable (original_dimes mother_dimes total_dimes dad_dimes : ℕ)

def proof_problem (original_dimes mother_dimes total_dimes dad_dimes : ℕ) : Prop :=
  original_dimes = 7 ∧
  mother_dimes = 4 ∧
  total_dimes = 19 ∧
  total_dimes = original_dimes + mother_dimes + dad_dimes

theorem find_dads_dimes (h : proof_problem 7 4 19 8) : dad_dimes = 8 :=
sorry

end find_dads_dimes_l84_84919


namespace steven_amanda_hike_difference_l84_84222

variable (Camila_hikes : ℕ)
variable (Camila_weeks : ℕ)
variable (hikes_per_week : ℕ)

def Amanda_hikes (Camila_hikes : ℕ) : ℕ := 8 * Camila_hikes

def Steven_hikes (Camila_hikes : ℕ)(Camila_weeks : ℕ)(hikes_per_week : ℕ) : ℕ :=
  Camila_hikes + Camila_weeks * hikes_per_week

theorem steven_amanda_hike_difference
  (hCamila : Camila_hikes = 7)
  (hWeeks : Camila_weeks = 16)
  (hHikesPerWeek : hikes_per_week = 4) :
  Steven_hikes Camila_hikes Camila_weeks hikes_per_week - Amanda_hikes Camila_hikes = 15 := by
  sorry

end steven_amanda_hike_difference_l84_84222


namespace length_of_third_side_l84_84731

-- Definitions for sides and perimeter condition
variables (a b : ℕ) (h1 : a = 3) (h2 : b = 10) (p : ℕ) (h3 : p % 6 = 0)
variable (c : ℕ)

-- Definition for the triangle inequality
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statement to prove the length of the third side
theorem length_of_third_side (h4 : triangle_inequality a b c)
  (h5 : p = a + b + c) : c = 11 :=
sorry

end length_of_third_side_l84_84731


namespace binomial_cubes_sum_l84_84615

theorem binomial_cubes_sum (x y : ℤ) :
  let B1 := x^4 + 9 * x * y^3
  let B2 := -(3 * x^3 * y) - 9 * y^4
  (B1 ^ 3 + B2 ^ 3 = x ^ 12 - 729 * y ^ 12) := by
  sorry

end binomial_cubes_sum_l84_84615


namespace paul_peaches_l84_84447

theorem paul_peaches (P : ℕ) (h1 : 26 - P = 22) : P = 4 :=
by {
  sorry
}

end paul_peaches_l84_84447


namespace john_spent_expected_amount_l84_84519

-- Define the original price of each pin
def original_price : ℝ := 20

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the number of pins
def number_of_pins : ℝ := 10

-- Define the sales tax rate
def tax_rate : ℝ := 0.08

-- Calculate the discount on each pin
def discount_per_pin : ℝ := discount_rate * original_price

-- Calculate the discounted price per pin
def discounted_price_per_pin : ℝ := original_price - discount_per_pin

-- Calculate the total discounted price for all pins
def total_discounted_price : ℝ := discounted_price_per_pin * number_of_pins

-- Calculate the sales tax on the total discounted price
def sales_tax : ℝ := tax_rate * total_discounted_price

-- Calculate the total amount spent including sales tax
def total_amount_spent : ℝ := total_discounted_price + sales_tax

-- The theorem that John spent $183.60 on pins including the sales tax
theorem john_spent_expected_amount : total_amount_spent = 183.60 :=
by
  sorry

end john_spent_expected_amount_l84_84519


namespace find_k_for_perpendicular_lines_l84_84015

theorem find_k_for_perpendicular_lines (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (5 - k) * y + 1 = 0) →
  (∀ x y : ℝ, 2 * (k-3) * x - 2 * y + 3 = 0) →
  (k = 1 ∨ k = 4) :=
by
  sorry

end find_k_for_perpendicular_lines_l84_84015


namespace inequality_proof_l84_84272

theorem inequality_proof 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
  sorry

end inequality_proof_l84_84272


namespace smallest_n_probability_l84_84916

theorem smallest_n_probability (n : ℕ) : (1 / (n * (n + 1)) < 1 / 2023) → (n ≥ 45) :=
by
  sorry

end smallest_n_probability_l84_84916


namespace derek_lowest_score_l84_84247

theorem derek_lowest_score:
  ∀ (score1 score2 max_points target_avg min_score tests_needed last_test1 last_test2 : ℕ),
  score1 = 85 →
  score2 = 78 →
  max_points = 100 →
  target_avg = 84 →
  min_score = 60 →
  tests_needed = 4 →
  last_test1 >= min_score →
  last_test2 >= min_score →
  last_test1 <= max_points →
  last_test2 <= max_points →
  (score1 + score2 + last_test1 + last_test2) = target_avg * tests_needed →
  min last_test1 last_test2 = 73 :=
by
  sorry

end derek_lowest_score_l84_84247


namespace hot_drinks_sales_l84_84557

theorem hot_drinks_sales (x: ℝ) (h: x = 4) : abs ((-2.35 * x + 155.47) - 146) < 1 :=
by sorry

end hot_drinks_sales_l84_84557


namespace angle_sum_420_l84_84981

theorem angle_sum_420 (A B C D E F : ℝ) (hE : E = 30) : 
  A + B + C + D + E + F = 420 :=
by
  sorry

end angle_sum_420_l84_84981


namespace khalil_total_payment_l84_84627

def cost_dog := 60
def cost_cat := 40
def cost_parrot := 70
def cost_rabbit := 50

def num_dogs := 25
def num_cats := 45
def num_parrots := 15
def num_rabbits := 10

def total_cost := num_dogs * cost_dog + num_cats * cost_cat + num_parrots * cost_parrot + num_rabbits * cost_rabbit

theorem khalil_total_payment : total_cost = 4850 := by
  sorry

end khalil_total_payment_l84_84627


namespace tina_jumps_more_than_cindy_l84_84391

def cindy_jumps : ℕ := 12
def betsy_jumps : ℕ := cindy_jumps / 2
def tina_jumps : ℕ := betsy_jumps * 3

theorem tina_jumps_more_than_cindy : tina_jumps - cindy_jumps = 6 := by
  sorry

end tina_jumps_more_than_cindy_l84_84391


namespace Freddy_is_18_l84_84963

-- Definitions based on the conditions
def Job_age : Nat := 5
def Stephanie_age : Nat := 4 * Job_age
def Freddy_age : Nat := Stephanie_age - 2

-- Statement to prove
theorem Freddy_is_18 : Freddy_age = 18 := by
  sorry

end Freddy_is_18_l84_84963


namespace candies_distribution_l84_84149

theorem candies_distribution (C : ℕ) (hC : C / 150 = C / 300 + 24) : C / 150 = 48 :=
by sorry

end candies_distribution_l84_84149


namespace max_experiments_fibonacci_search_l84_84987

-- Define the conditions and the theorem
def is_unimodal (f : ℕ → ℕ) : Prop :=
  ∃ k, ∀ n m, (n < k ∧ k ≤ m) → f n < f k ∧ f k > f m

def fibonacci_search_experiments (n : ℕ) : ℕ :=
  -- Placeholder function representing the steps of Fibonacci search
  if n <= 1 then n else fibonacci_search_experiments (n - 1) + fibonacci_search_experiments (n - 2)

theorem max_experiments_fibonacci_search (f : ℕ → ℕ) (n : ℕ) (hn : n = 33) (hf : is_unimodal f) : fibonacci_search_experiments n ≤ 7 :=
  sorry

end max_experiments_fibonacci_search_l84_84987


namespace length_AB_of_parallelogram_l84_84277

theorem length_AB_of_parallelogram
  (AD BC : ℝ) (AB CD : ℝ) 
  (h1 : AD = 5) 
  (h2 : BC = 5) 
  (h3 : AB = CD)
  (h4 : AD + BC + AB + CD = 14) : 
  AB = 2 :=
by
  sorry

end length_AB_of_parallelogram_l84_84277


namespace probability_ephraim_keiko_l84_84594

-- Define the probability that Ephraim gets a certain number of heads tossing two pennies
def prob_heads_ephraim (n : Nat) : ℚ :=
  if n = 2 then 1 / 4
  else if n = 1 then 1 / 2
  else if n = 0 then 1 / 4
  else 0

-- Define the probability that Keiko gets a certain number of heads tossing one penny
def prob_heads_keiko (n : Nat) : ℚ :=
  if n = 1 then 1 / 2
  else if n = 0 then 1 / 2
  else 0

-- Define the probability that Ephraim and Keiko get the same number of heads
def prob_same_heads : ℚ :=
  (prob_heads_ephraim 0 * prob_heads_keiko 0) + (prob_heads_ephraim 1 * prob_heads_keiko 1) + (prob_heads_ephraim 2 * prob_heads_keiko 2)

-- The statement that requires proof
theorem probability_ephraim_keiko : prob_same_heads = 3 / 8 := 
  sorry

end probability_ephraim_keiko_l84_84594


namespace slope_proof_l84_84995

noncomputable def slope_between_midpoints : ℚ :=
  let p1 := (2, 3)
  let p2 := (4, 5)
  let q1 := (7, 3)
  let q2 := (8, 7)

  let midpoint (a b : ℚ × ℚ) : ℚ × ℚ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

  let m1 := midpoint p1 p2
  let m2 := midpoint q1 q2

  (m2.2 - m1.2) / (m2.1 - m1.1)

theorem slope_proof : slope_between_midpoints = 2 / 9 := by
  sorry

end slope_proof_l84_84995


namespace find_negative_number_l84_84586

noncomputable def is_negative (x : ℝ) : Prop := x < 0

theorem find_negative_number : is_negative (-5) := by
  -- Proof steps would go here, but we'll skip them for now.
  sorry

end find_negative_number_l84_84586


namespace determinant_inequality_l84_84768

theorem determinant_inequality (x : ℝ) (h : 2 * x - (3 - x) > 0) : 3 * x - 3 > 0 := 
by
  sorry

end determinant_inequality_l84_84768


namespace max_residents_per_apartment_l84_84778

theorem max_residents_per_apartment (total_floors : ℕ) (floors_with_6_apts : ℕ) (floors_with_5_apts : ℕ)
  (rooms_per_6_floors : ℕ) (rooms_per_5_floors : ℕ) (max_residents : ℕ) : 
  total_floors = 12 ∧ floors_with_6_apts = 6 ∧ floors_with_5_apts = 6 ∧ 
  rooms_per_6_floors = 6 ∧ rooms_per_5_floors = 5 ∧ max_residents = 264 → 
  264 / (6 * 6 + 6 * 5) = 4 := sorry

end max_residents_per_apartment_l84_84778


namespace rectangle_properties_l84_84364

theorem rectangle_properties :
  ∃ (length width : ℝ),
    (length / width = 3) ∧ 
    (length * width = 75) ∧
    (length = 15) ∧
    (width = 5) ∧
    ∀ (side : ℝ), 
      (side^2 = 75) → 
      (side - width > 3) :=
by
  sorry

end rectangle_properties_l84_84364


namespace cheolsu_weight_l84_84974

variable (C M : ℝ)

theorem cheolsu_weight:
  (C = (2/3) * M) →
  (C + 72 = 2 * M) →
  C = 36 :=
by
  intros h1 h2
  sorry

end cheolsu_weight_l84_84974


namespace weight_of_b_l84_84083

theorem weight_of_b (A B C : ℝ) 
  (h1 : A + B + C = 129)
  (h2 : A + B = 96)
  (h3 : B + C = 84) : 
  B = 51 :=
sorry

end weight_of_b_l84_84083


namespace min_value_fraction_l84_84266

theorem min_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : -3 ≤ y ∧ y ≤ 1) : (x + y) / x = 0.8 :=
by
  sorry

end min_value_fraction_l84_84266


namespace annette_miscalculation_l84_84767

theorem annette_miscalculation :
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  x' - y' = 1 :=
by
  let x := 6
  let y := 3
  let x' := 5
  let y' := 4
  sorry

end annette_miscalculation_l84_84767


namespace total_cupcakes_l84_84530

-- Definitions of initial conditions
def cupcakes_initial : ℕ := 42
def cupcakes_sold : ℕ := 22
def cupcakes_made_after : ℕ := 39

-- Proof statement: Total number of cupcakes Robin would have
theorem total_cupcakes : 
  (cupcakes_initial - cupcakes_sold + cupcakes_made_after) = 59 := by
    sorry

end total_cupcakes_l84_84530


namespace min_m_value_arithmetic_seq_l84_84808

theorem min_m_value_arithmetic_seq :
  ∀ (a S : ℕ → ℚ) (m : ℕ),
  (∀ n : ℕ, a (n+2) = 5 ∧ a (n+6) = 21) →
  (∀ n : ℕ, S (n+1) = S n + 1 / a (n+1)) →
  (∀ n : ℕ, S (2 * n + 1) - S n ≤ m / 15) →
  ∀ n : ℕ, m = 5 :=
sorry

end min_m_value_arithmetic_seq_l84_84808


namespace cos_sq_minus_sin_sq_l84_84042

noncomputable def alpha : ℝ := sorry

axiom tan_alpha_eq_two : Real.tan alpha = 2

theorem cos_sq_minus_sin_sq : Real.cos alpha ^ 2 - Real.sin alpha ^ 2 = -3/5 := by
  sorry

end cos_sq_minus_sin_sq_l84_84042


namespace quadratic_eq_solution_1_quadratic_eq_solution_2_l84_84348

theorem quadratic_eq_solution_1 :
    ∀ (x : ℝ), x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by 
  sorry

theorem quadratic_eq_solution_2 :
    ∀ (x : ℝ), x * (x - 2) - x + 2 = 0 ↔ x = 1 ∨ x = 2 :=
by 
  sorry

end quadratic_eq_solution_1_quadratic_eq_solution_2_l84_84348


namespace function_odd_and_decreasing_l84_84283

noncomputable def f (a x : ℝ) : ℝ := (1 / a) ^ x - a ^ x

theorem function_odd_and_decreasing (a : ℝ) (h : a > 1) :
  (∀ x, f a (-x) = -f a x) ∧ (∀ x y, x < y → f a x > f a y) :=
by
  sorry

end function_odd_and_decreasing_l84_84283


namespace max_quotient_l84_84642

theorem max_quotient (x y : ℝ) (hx : 100 ≤ x ∧ x ≤ 300) (hy : 900 ≤ y ∧ y ≤ 1800) : 
  (∀ x y, (100 ≤ x ∧ x ≤ 300) ∧ (900 ≤ y ∧ y ≤ 1800) → y / x ≤ 18) ∧ 
  (∃ x y, (100 ≤ x ∧ x ≤ 300) ∧ (900 ≤ y ∧ y ≤ 1800) ∧ y / x = 18) :=
by
  sorry

end max_quotient_l84_84642


namespace constraint_condition_2000_yuan_wage_l84_84440

-- Definitions based on the given conditions
def wage_carpenter : ℕ := 50
def wage_bricklayer : ℕ := 40
def total_wage : ℕ := 2000

-- Let x be the number of carpenters and y be the number of bricklayers
variable (x y : ℕ)

-- The proof problem statement
theorem constraint_condition_2000_yuan_wage (x y : ℕ) : 
  wage_carpenter * x + wage_bricklayer * y = total_wage → 5 * x + 4 * y = 200 :=
by
  intro h
  -- Simplification step will be placed here
  sorry

end constraint_condition_2000_yuan_wage_l84_84440


namespace total_original_cost_l84_84424

theorem total_original_cost (discounted_price1 discounted_price2 discounted_price3 : ℕ) 
  (discount_rate1 discount_rate2 discount_rate3 : ℚ)
  (h1 : discounted_price1 = 4400)
  (h2 : discount_rate1 = 0.56)
  (h3 : discounted_price2 = 3900)
  (h4 : discount_rate2 = 0.35)
  (h5 : discounted_price3 = 2400)
  (h6 : discount_rate3 = 0.20) :
  (discounted_price1 / (1 - discount_rate1) + discounted_price2 / (1 - discount_rate2) 
    + discounted_price3 / (1 - discount_rate3) = 19000) :=
by
  sorry

end total_original_cost_l84_84424


namespace inequality_problem_l84_84351

open Real

theorem inequality_problem
  (a b c x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h_condition : 1 / x + 1 / y + 1 / z = 1) :
  a^x + b^y + c^z ≥ 4 * a * b * c * x * y * z / (x + y + z - 3) ^ 2 :=
by
  sorry

end inequality_problem_l84_84351


namespace young_people_in_sample_l84_84574

-- Define the conditions
def total_population (elderly middle_aged young : ℕ) : ℕ :=
  elderly + middle_aged + young

def sample_proportion (sample_size total_pop : ℚ) : ℚ :=
  sample_size / total_pop

def stratified_sample (group_size proportion : ℚ) : ℚ :=
  group_size * proportion

-- Main statement to prove
theorem young_people_in_sample (elderly middle_aged young : ℕ) (sample_size : ℚ) :
  total_population elderly middle_aged young = 108 →
  sample_size = 36 →
  stratified_sample (young : ℚ) (sample_proportion sample_size 108) = 17 :=
by
  intros h_total h_sample_size
  sorry -- proof omitted

end young_people_in_sample_l84_84574


namespace class_average_l84_84601

theorem class_average (x : ℝ) :
  (0.25 * 80 + 0.5 * x + 0.25 * 90 = 75) → x = 65 := by
  sorry

end class_average_l84_84601


namespace cost_of_blue_hat_is_six_l84_84419

-- Given conditions
def total_hats : ℕ := 85
def green_hats : ℕ := 40
def blue_hats : ℕ := total_hats - green_hats
def cost_green_hat : ℕ := 7
def total_cost : ℕ := 550
def total_cost_green_hats : ℕ := green_hats * cost_green_hat
def total_cost_blue_hats : ℕ := total_cost - total_cost_green_hats
def cost_blue_hat : ℕ := total_cost_blue_hats / blue_hats

-- Proof statement
theorem cost_of_blue_hat_is_six : cost_blue_hat = 6 := sorry

end cost_of_blue_hat_is_six_l84_84419


namespace parabola_tangent_perp_l84_84961

theorem parabola_tangent_perp (a b : ℝ) : 
  (∃ x y : ℝ, x^2 = 4 * y ∧ y = a ∧ b ≠ 0 ∧ x ≠ 0) ∧
  (∃ x' y' : ℝ, x'^2 = 4 * y' ∧ y' = b ∧ a ≠ 0 ∧ x' ≠ 0) ∧
  (a * b = -1) 
  → a^4 * b^4 = (a^2 + b^2)^3 :=
by
  sorry

end parabola_tangent_perp_l84_84961


namespace football_problem_l84_84529

-- Definitions based on conditions
def total_balls (x y : Nat) : Prop := x + y = 200
def total_cost (x y : Nat) : Prop := 80 * x + 60 * y = 14400
def football_A_profit_per_ball : Nat := 96 - 80
def football_B_profit_per_ball : Nat := 81 - 60
def total_profit (x y : Nat) : Nat :=
  football_A_profit_per_ball * x + football_B_profit_per_ball * y

-- Lean statement proving the conditions lead to the solution
theorem football_problem
  (x y : Nat)
  (h1 : total_balls x y)
  (h2 : total_cost x y)
  (h3 : x = 120)
  (h4 : y = 80) :
  total_profit x y = 3600 := by
  sorry

end football_problem_l84_84529


namespace gcd_factorial_8_6_squared_l84_84412

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_6_squared :
  Nat.gcd (factorial 8) ((factorial 6) ^ 2) = 7200 :=
by
  sorry

end gcd_factorial_8_6_squared_l84_84412


namespace meal_combinations_l84_84657

def menu_items : ℕ := 12
def special_dish_chosen : Prop := true

theorem meal_combinations : (special_dish_chosen → (menu_items - 1) * (menu_items - 1) = 121) :=
by
  sorry

end meal_combinations_l84_84657


namespace solve_for_x_l84_84039

theorem solve_for_x (x : ℝ) (h : 9 / x^2 = x / 81) : x = 9 := 
  sorry

end solve_for_x_l84_84039


namespace original_price_l84_84076

theorem original_price (x : ℝ) (h : x * (1 / 8) = 8) : x = 64 := by
  -- To be proved
  sorry

end original_price_l84_84076


namespace find_number_l84_84845

theorem find_number (n : ℤ) (h : 7 * n - 15 = 2 * n + 10) : n = 5 :=
sorry

end find_number_l84_84845


namespace circle_diameter_l84_84702

theorem circle_diameter (r d : ℝ) (h1 : π * r^2 = 4 * π) (h2 : d = 2 * r) : d = 4 :=
by {
  sorry
}

end circle_diameter_l84_84702


namespace minimum_days_bacteria_count_exceeds_500_l84_84896

theorem minimum_days_bacteria_count_exceeds_500 :
  ∃ n : ℕ, 4 * 3^n > 500 ∧ ∀ m : ℕ, m < n → 4 * 3^m ≤ 500 :=
by
  sorry

end minimum_days_bacteria_count_exceeds_500_l84_84896


namespace smoothie_one_serving_ingredients_in_cups_containers_needed_l84_84874

theorem smoothie_one_serving_ingredients_in_cups :
  (0.2 + 0.1 + 0.2 + 1 * 0.125 + 2 * 0.0625 + 0.5).round = 1.25.round := sorry

theorem containers_needed :
  (5 * 1.25 / 1.5).ceil = 5 := sorry

end smoothie_one_serving_ingredients_in_cups_containers_needed_l84_84874


namespace cos_double_angle_l84_84829

theorem cos_double_angle (α : ℝ) (h : ‖(Real.cos α, Real.sqrt 2 / 2)‖ = Real.sqrt 3 / 2) : Real.cos (2 * α) = -1 / 2 :=
sorry

end cos_double_angle_l84_84829


namespace donna_has_40_bananas_l84_84018

-- Define the number of bananas each person has
variables (dawn lydia donna total : ℕ)

-- State the conditions
axiom h1 : dawn + lydia + donna = total
axiom h2 : dawn = lydia + 40
axiom h3 : lydia = 60
axiom h4 : total = 200

-- State the theorem to be proved
theorem donna_has_40_bananas : donna = 40 :=
by {
  sorry -- Placeholder for the proof
}

end donna_has_40_bananas_l84_84018


namespace smallest_d_l84_84334

theorem smallest_d (d : ℕ) (h_pos : 0 < d) (h_square : ∃ k : ℕ, 3150 * d = k^2) : d = 14 :=
sorry

end smallest_d_l84_84334


namespace exists_long_segment_between_parabolas_l84_84716

def parabola1 (x : ℝ) : ℝ :=
  x ^ 2

def parabola2 (x : ℝ) : ℝ :=
  x ^ 2 - 1

def in_between_parabolas (x y : ℝ) : Prop :=
  (parabola2 x) ≤ y ∧ y ≤ (parabola1 x)

theorem exists_long_segment_between_parabolas :
  ∃ (M1 M2: ℝ × ℝ), in_between_parabolas M1.1 M1.2 ∧ in_between_parabolas M2.1 M2.2 ∧ dist M1 M2 > 10^6 :=
sorry

end exists_long_segment_between_parabolas_l84_84716


namespace joan_football_games_l84_84811

theorem joan_football_games (games_this_year games_last_year total_games: ℕ)
  (h1 : games_this_year = 4)
  (h2 : games_last_year = 9)
  (h3 : total_games = games_this_year + games_last_year) :
  total_games = 13 := 
by
  sorry

end joan_football_games_l84_84811


namespace complex_division_l84_84142

theorem complex_division :
  (1 - 2 * Complex.I) / (2 + Complex.I) = -Complex.I :=
by sorry

end complex_division_l84_84142


namespace sand_exchange_impossible_l84_84255

/-- Given initial conditions for g and p, the goal is to determine if 
the banker can have at least 2 kg of each type of sand in the end. -/
theorem sand_exchange_impossible (g p : ℕ) (G P : ℕ) 
  (initial_g : g = 1001) (initial_p : p = 1001) 
  (initial_G : G = 1) (initial_P : P = 1)
  (exchange_rule : ∀ x y : ℚ, x * p = y * g) 
  (decrement_rule : ∀ k, 1 ≤ k ∧ k ≤ 2000 → 
    (g = 1001 - k ∨ p = 1001 - k)) :
  ¬(G ≥ 2 ∧ P ≥ 2) :=
by
  -- Add a placeholder to skip the proof
  sorry

end sand_exchange_impossible_l84_84255


namespace solve_inequality_l84_84036

theorem solve_inequality (x : ℝ) : x + 1 > 3 → x > 2 := 
sorry

end solve_inequality_l84_84036


namespace neighbors_receive_28_mangoes_l84_84072

/-- 
  Mr. Wong harvested 560 mangoes. He sold half, gave 50 to his family,
  and divided the remaining mangoes equally among 8 neighbors.
  Each neighbor should receive 28 mangoes.
-/
theorem neighbors_receive_28_mangoes : 
  ∀ (initial : ℕ) (sold : ℕ) (given : ℕ) (neighbors : ℕ), 
  initial = 560 → 
  sold = initial / 2 → 
  given = 50 → 
  neighbors = 8 → 
  (initial - sold - given) / neighbors = 28 := 
by 
  intros initial sold given neighbors
  sorry

end neighbors_receive_28_mangoes_l84_84072


namespace real_ratio_sum_values_l84_84025

variables (a b c d : ℝ)

theorem real_ratio_sum_values :
  (a / b + b / c + c / d + d / a = 6) ∧
  (a / c + b / d + c / a + d / b = 8) →
  (a / b + c / d = 2 ∨ a / b + c / d = 4) :=
by
  sorry

end real_ratio_sum_values_l84_84025


namespace cylinder_original_radius_l84_84204

theorem cylinder_original_radius
  (r : ℝ)
  (h_original : ℝ := 4)
  (h_increased : ℝ := 3 * h_original)
  (volume_eq : π * (r + 8)^2 * h_original = π * r^2 * h_increased) :
  r = 4 + 4 * Real.sqrt 5 :=
sorry

end cylinder_original_radius_l84_84204


namespace square_tiles_count_l84_84826

theorem square_tiles_count (a b : ℕ) (h1 : a + b = 25) (h2 : 3 * a + 4 * b = 84) : b = 9 := by
  sorry

end square_tiles_count_l84_84826


namespace minimum_value_of_f_l84_84508

noncomputable def f (x : ℝ) : ℝ := (Real.sin (Real.pi * x) - Real.cos (Real.pi * x) + 2) / Real.sqrt x

theorem minimum_value_of_f :
  ∃ x ∈ Set.Icc (1/4 : ℝ) (5/4 : ℝ), f x = (4 * Real.sqrt 5 / 5 - 2 * Real.sqrt 10 / 5) :=
sorry

end minimum_value_of_f_l84_84508


namespace intersection_points_range_l84_84177

theorem intersection_points_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ a = x₁^3 - 3 * x₁ ∧
  a = x₂^3 - 3 * x₂ ∧ a = x₃^3 - 3 * x₃) ↔ (-2 < a ∧ a < 2) :=
sorry

end intersection_points_range_l84_84177


namespace f_2013_value_l84_84405

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, x ≠ 1 → f (2 * x + 1) + g (3 - x) = x
axiom h2 : ∀ x : ℝ, x ≠ 1 → f ((3 * x + 5) / (x + 1)) + 2 * g ((2 * x + 1) / (x + 1)) = x / (x + 1)

theorem f_2013_value : f 2013 = 1010 / 1007 :=
by
  sorry

end f_2013_value_l84_84405


namespace remainder_div_l84_84599

theorem remainder_div (N : ℕ) (n : ℕ) : 
  (N % 2^n) = (N % 10^n % 2^n) ∧ (N % 5^n) = (N % 10^n % 5^n) := by
  sorry

end remainder_div_l84_84599


namespace max_hardcover_books_l84_84016

-- Define the conditions as provided in the problem
def total_books : ℕ := 36
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ a * b = n

-- The logical statement we need to prove
theorem max_hardcover_books :
  ∃ h : ℕ, (∃ c : ℕ, is_composite c ∧ 2 * h + c = total_books) ∧ 
  ∀ h' c', is_composite c' ∧ 2 * h' + c' = total_books → h' ≤ h :=
sorry

end max_hardcover_books_l84_84016


namespace Valley_Forge_High_School_winter_carnival_l84_84376

noncomputable def number_of_girls (total_students : ℕ) (total_participants : ℕ) (fraction_girls_participating : ℚ) (fraction_boys_participating : ℚ) : ℕ := sorry

theorem Valley_Forge_High_School_winter_carnival
  (total_students : ℕ)
  (total_participants : ℕ)
  (fraction_girls_participating : ℚ)
  (fraction_boys_participating : ℚ)
  (h_total_students : total_students = 1500)
  (h_total_participants : total_participants = 900)
  (h_fraction_girls : fraction_girls_participating = 3 / 4)
  (h_fraction_boys : fraction_boys_participating = 2 / 3) :
  number_of_girls total_students total_participants fraction_girls_participating fraction_boys_participating = 900 := sorry

end Valley_Forge_High_School_winter_carnival_l84_84376


namespace least_pebbles_2021_l84_84287

noncomputable def least_pebbles (n : ℕ) : ℕ :=
  n + n / 2

theorem least_pebbles_2021 :
  least_pebbles 2021 = 3031 :=
by
  sorry

end least_pebbles_2021_l84_84287


namespace sum_first_five_terms_l84_84903

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 1, ∀ n, a (n + 1) = a n * q

theorem sum_first_five_terms (h₁ : is_geometric_sequence a) 
  (h₂ : a 1 > 0) 
  (h₃ : a 1 * a 7 = 64) 
  (h₄ : a 3 + a 5 = 20) : 
  a 1 * (1 - (2 : ℝ) ^ 5) / (1 - 2) = 31 := 
by
  sorry

end sum_first_five_terms_l84_84903


namespace sequence_bounds_l84_84577

theorem sequence_bounds (c : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = ↑n + c / ↑n) 
  (h2 : ∀ n : ℕ+, a n ≥ a 3) : 6 ≤ c ∧ c ≤ 12 :=
by 
  -- We will prove that 6 ≤ c and c ≤ 12 given the conditions stated
  sorry

end sequence_bounds_l84_84577


namespace edge_length_is_correct_l84_84921

-- Define the given conditions
def volume_material : ℕ := 12 * 18 * 6
def edge_length : ℕ := 3
def number_cubes : ℕ := 48
def volume_cube (e : ℕ) : ℕ := e * e * e

-- Problem statement in Lean:
theorem edge_length_is_correct : volume_material = number_cubes * volume_cube edge_length → edge_length = 3 :=
by
  sorry

end edge_length_is_correct_l84_84921


namespace equivalent_multipliers_l84_84494

variable (a b : ℝ)

theorem equivalent_multipliers (a b : ℝ) :
  let a_final := 0.93 * a
  let expr := a_final + 0.05 * b
  expr = 0.93 * a + 0.05 * b  :=
by
  -- Proof placeholder
  sorry

end equivalent_multipliers_l84_84494


namespace range_of_m_l84_84078

noncomputable def setA := {x : ℝ | -2 ≤ x ∧ x ≤ 7}
noncomputable def setB (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem range_of_m (m : ℝ) : (setA ∪ setB m = setA) → m ≤ 4 :=
by
  intro h
  sorry

end range_of_m_l84_84078


namespace range_of_a_for_common_points_l84_84122

theorem range_of_a_for_common_points (a : ℝ) : (∃ x : ℝ, x > 0 ∧ ax^2 = Real.exp x) ↔ a ≥ Real.exp 2 / 4 :=
sorry

end range_of_a_for_common_points_l84_84122


namespace geometric_sequence_product_correct_l84_84327

noncomputable def geometric_sequence_product (a_1 a_5 : ℝ) (a_2 a_3 a_4 : ℝ) :=
  a_1 = 1 / 2 ∧ a_5 = 8 ∧ a_2 * a_4 = a_1 * a_5 ∧ a_3^2 = a_1 * a_5

theorem geometric_sequence_product_correct:
  ∃ a_2 a_3 a_4 : ℝ, geometric_sequence_product (1 / 2) 8 a_2 a_3 a_4 ∧ (a_2 * a_3 * a_4 = 8) :=
by
  sorry

end geometric_sequence_product_correct_l84_84327


namespace greatest_possible_individual_award_l84_84476

variable (prize : ℕ)
variable (total_winners : ℕ)
variable (min_award : ℕ)
variable (fraction_prize : ℚ)
variable (fraction_winners : ℚ)

theorem greatest_possible_individual_award 
  (h1 : prize = 2500)
  (h2 : total_winners = 25)
  (h3 : min_award = 50)
  (h4 : fraction_prize = 3/5)
  (h5 : fraction_winners = 2/5) :
  ∃ award, award = 1300 := by
  sorry

end greatest_possible_individual_award_l84_84476


namespace rachel_envelopes_first_hour_l84_84243

theorem rachel_envelopes_first_hour (total_envelopes : ℕ) (hours : ℕ) (e2 : ℕ) (e_per_hour : ℕ) :
  total_envelopes = 1500 → hours = 8 → e2 = 141 → e_per_hour = 204 →
  ∃ e1 : ℕ, e1 = 135 :=
by
  sorry

end rachel_envelopes_first_hour_l84_84243


namespace difference_between_max_and_min_34_l84_84864

theorem difference_between_max_and_min_34 
  (A B C D E: ℕ) 
  (h_avg: (A + B + C + D + E) / 5 = 50) 
  (h_max: E ≤ 58) 
  (h_distinct: A < B ∧ B < C ∧ C < D ∧ D < E) 
: E - A = 34 := 
sorry

end difference_between_max_and_min_34_l84_84864


namespace rectangle_area_l84_84793

theorem rectangle_area (y : ℝ) (w : ℝ) (h : w > 0) (h_diag : y^2 = 10 * w^2) : 
  (3 * w)^2 * w = 3 * (y^2 / 10) :=
by sorry

end rectangle_area_l84_84793


namespace earnings_difference_l84_84804

def total_earnings : ℕ := 3875
def first_job_earnings : ℕ := 2125
def second_job_earnings := total_earnings - first_job_earnings

theorem earnings_difference : (first_job_earnings - second_job_earnings) = 375 := by
  sorry

end earnings_difference_l84_84804


namespace bess_milk_daily_l84_84989

-- Definitions based on conditions from step a)
variable (B : ℕ) -- B is the number of pails Bess gives every day

def BrownieMilk : ℕ := 3 * B
def DaisyMilk : ℕ := B + 1
def TotalDailyMilk : ℕ := B + BrownieMilk B + DaisyMilk B

-- Conditions definition to be used in Lean to ensure the equivalence
axiom weekly_milk_total : 7 * TotalDailyMilk B = 77
axiom daily_milk_eq : TotalDailyMilk B = 11

-- Prove that Bess gives 2 pails of milk everyday
theorem bess_milk_daily : B = 2 :=
by
  sorry

end bess_milk_daily_l84_84989


namespace min_value_of_expression_l84_84381

open Real

theorem min_value_of_expression {a b c d e f : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
    (h_sum : a + b + c + d + e + f = 10) :
    (∃ x, x = 44.1 ∧ ∀ y, y = 1 / a + 4 / b + 9 / c + 16 / d + 25 / e + 36 / f → x ≤ y) :=
sorry

end min_value_of_expression_l84_84381


namespace remaining_money_after_shopping_l84_84095

theorem remaining_money_after_shopping (initial_money : ℝ) (percentage_spent : ℝ) (final_amount : ℝ) :
  initial_money = 1200 → percentage_spent = 0.30 → final_amount = initial_money - (percentage_spent * initial_money) → final_amount = 840 :=
by
  intros h_initial h_percentage h_final
  sorry

end remaining_money_after_shopping_l84_84095


namespace find_radius_squared_l84_84115

theorem find_radius_squared (r : ℝ) (AB_len CD_len BP : ℝ) (angle_APD : ℝ) (h1 : AB_len = 12)
    (h2 : CD_len = 9) (h3 : BP = 10) (h4 : angle_APD = 60) : r^2 = 111 := by
  have AB_len := h1
  have CD_len := h2
  have BP := h3
  have angle_APD := h4
  sorry

end find_radius_squared_l84_84115


namespace age_of_father_l84_84538

theorem age_of_father (F C : ℕ) 
  (h1 : F = C)
  (h2 : C + 5 * 15 = 2 * (F + 15)) : 
  F = 45 := 
by 
sorry

end age_of_father_l84_84538


namespace part_one_solution_set_part_two_range_of_a_l84_84967

def f (x : ℝ) (a : ℝ) : ℝ := |x - a| - 2

theorem part_one_solution_set (a : ℝ) (h : a = 1) : { x : ℝ | f x a + |2 * x - 3| > 0 } = { x : ℝ | x > 2 ∨ x < 2 / 3 } := 
sorry

theorem part_two_range_of_a : (∃ x : ℝ, f x (a) > |x - 3|) ↔ (a < 1 ∨ a > 5) :=
sorry

end part_one_solution_set_part_two_range_of_a_l84_84967


namespace log2_6_gt_2_sqrt_5_l84_84907

theorem log2_6_gt_2_sqrt_5 : 2 + Real.logb 2 6 > 2 * Real.sqrt 5 := by
  sorry

end log2_6_gt_2_sqrt_5_l84_84907


namespace smallest_k_l84_84850

theorem smallest_k (k : ℕ) (h : 201 ≡ 9 [MOD 24]) : k = 1 := by
  sorry

end smallest_k_l84_84850


namespace triangle_equilateral_from_condition_l84_84467

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
a = b ∧ b = c

theorem triangle_equilateral_from_condition (a b c h_a h_b h_c : ℝ)
  (h : a + h_a = b + h_b ∧ b + h_b = c + h_c) :
  is_equilateral a b c :=
sorry

end triangle_equilateral_from_condition_l84_84467


namespace sum_of_conjugates_eq_30_l84_84169

theorem sum_of_conjugates_eq_30 :
  (15 - Real.sqrt 2023) + (15 + Real.sqrt 2023) = 30 :=
sorry

end sum_of_conjugates_eq_30_l84_84169


namespace imaginary_part_z1_mul_z2_l84_84730

def z1 : ℂ := ⟨1, -1⟩
def z2 : ℂ := ⟨2, 4⟩

theorem imaginary_part_z1_mul_z2 : (z1 * z2).im = 2 := by
  sorry

end imaginary_part_z1_mul_z2_l84_84730


namespace cost_per_tissue_l84_84866

-- Annalise conditions
def boxes : ℕ := 10
def packs_per_box : ℕ := 20
def tissues_per_pack : ℕ := 100
def total_spent : ℝ := 1000

-- Definition for total packs and total tissues
def total_packs : ℕ := boxes * packs_per_box
def total_tissues : ℕ := total_packs * tissues_per_pack

-- The math problem: Prove the cost per tissue
theorem cost_per_tissue : (total_spent / total_tissues) = 0.05 := by
  sorry

end cost_per_tissue_l84_84866


namespace fraction_exponent_product_l84_84215

theorem fraction_exponent_product :
  ( (5/6: ℚ)^2 * (2/3: ℚ)^3 = 50/243 ) :=
by
  sorry

end fraction_exponent_product_l84_84215


namespace max_rectangles_3x5_in_17x22_l84_84449

theorem max_rectangles_3x5_in_17x22 : ∃ n : ℕ, n = 24 ∧ 
  (∀ (cut_3x5_pieces : ℤ), cut_3x5_pieces ≤ n) :=
by
  sorry

end max_rectangles_3x5_in_17x22_l84_84449


namespace complex_norm_solution_l84_84964

noncomputable def complex_norm (z : Complex) : Real :=
  Complex.abs z

theorem complex_norm_solution (w z : Complex) 
  (wz_condition : w * z = 24 - 10 * Complex.I)
  (w_norm_condition : complex_norm w = Real.sqrt 29) :
  complex_norm z = (26 * Real.sqrt 29) / 29 :=
by
  sorry

end complex_norm_solution_l84_84964


namespace sample_size_l84_84859

theorem sample_size (w_under30 : ℕ) (w_30to40 : ℕ) (w_40plus : ℕ) (sample_40plus : ℕ) (total_sample : ℕ) :
  w_under30 = 2400 →
  w_30to40 = 3600 →
  w_40plus = 6000 →
  sample_40plus = 60 →
  total_sample = 120 :=
by
  intros
  sorry

end sample_size_l84_84859


namespace sin_double_angle_plus_pi_over_six_l84_84068

variable (θ : ℝ)
variable (h : 7 * Real.sqrt 3 * Real.sin θ = 1 + 7 * Real.cos θ)

theorem sin_double_angle_plus_pi_over_six :
  Real.sin (2 * θ + π / 6) = 97 / 98 :=
by
  sorry

end sin_double_angle_plus_pi_over_six_l84_84068


namespace P_parity_Q_div_by_3_l84_84251

-- Define polynomial P(x)
def P (x p q : ℤ) : ℤ := x*x + p*x + q

-- Define polynomial Q(x)
def Q (x p q : ℤ) : ℤ := x*x*x + p*x + q

-- Part (a) proof statement
theorem P_parity (p q : ℤ) (h1 : Odd p) (h2 : Even q ∨ Odd q) :
  (∀ x : ℤ, Even (P x p q)) ∨ (∀ x : ℤ, Odd (P x p q)) :=
sorry

-- Part (b) proof statement
theorem Q_div_by_3 (p q : ℤ) (h1 : q % 3 = 0) (h2 : p % 3 = 2) :
  ∀ x : ℤ, Q x p q % 3 = 0 :=
sorry

end P_parity_Q_div_by_3_l84_84251


namespace lines_intersect_at_single_point_l84_84889

theorem lines_intersect_at_single_point (m : ℚ)
    (h1 : ∃ x y : ℚ, y = 4 * x - 8 ∧ y = -3 * x + 9)
    (h2 : ∀ x y : ℚ, (y = 4 * x - 8 ∧ y = -3 * x + 9) → (y = 2 * x + m)) :
    m = -22/7 := by
  sorry

end lines_intersect_at_single_point_l84_84889


namespace inequality_solution_l84_84158

noncomputable def solve_inequality (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution : {x : ℝ | solve_inequality x} = 
  {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry

end inequality_solution_l84_84158


namespace sum_of_digits_third_smallest_multiple_l84_84547

noncomputable def LCM_upto_7 : ℕ := Nat.lcm (Nat.lcm 1 2) (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))

noncomputable def third_smallest_multiple : ℕ := 3 * LCM_upto_7

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_third_smallest_multiple : sum_of_digits third_smallest_multiple = 9 := 
sorry

end sum_of_digits_third_smallest_multiple_l84_84547


namespace inverse_function_correct_inequality_solution_l84_84481

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

noncomputable def f_inv (y : ℝ) : ℝ := Real.log (1 + y) / Real.log (1 - y)

theorem inverse_function_correct (x : ℝ) (hx : -1 < x ∧ x < 1) :
  f_inv (f x) = x :=
sorry

theorem inequality_solution :
  ∀ x, (1 / 2 < x ∧ x < 1) ↔ (f_inv x > Real.log (1 + x) + 1) :=
sorry

end inverse_function_correct_inequality_solution_l84_84481


namespace linear_equations_not_always_solvable_l84_84191

theorem linear_equations_not_always_solvable 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : 
  ¬(∀ x y : ℝ, (a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂) ↔ 
                   a₁ * b₂ - a₂ * b₁ ≠ 0) :=
sorry

end linear_equations_not_always_solvable_l84_84191


namespace segment_area_l84_84483

noncomputable def area_segment_above_triangle (a b c : ℝ) (triangle_area : ℝ) (y : ℝ) :=
  let ellipse_area := Real.pi * a * b
  ellipse_area - triangle_area

theorem segment_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 1) :
  let y := (4 * Real.sqrt 2) / 3
  let triangle_area := (1 / 2) * (2 * (b - y))
  area_segment_above_triangle a b c triangle_area y = 6 * Real.pi - 2 + (4 * Real.sqrt 2) / 3 := by
  sorry

end segment_area_l84_84483


namespace necessary_not_sufficient_condition_l84_84450

theorem necessary_not_sufficient_condition (x : ℝ) :
  ((-6 ≤ x ∧ x ≤ 3) → (-5 ≤ x ∧ x ≤ 3)) ∧
  (¬ ((-5 ≤ x ∧ x ≤ 3) → (-6 ≤ x ∧ x ≤ 3))) :=
by
  -- Need proof steps here
  sorry

end necessary_not_sufficient_condition_l84_84450


namespace equilateral_triangle_on_parallel_lines_l84_84502

theorem equilateral_triangle_on_parallel_lines 
  (l1 l2 l3 : ℝ → Prop)
  (h_parallel_12 : ∀ x y, l1 x → l2 y → ∀ z, l1 z → l2 z)
  (h_parallel_23 : ∀ x y, l2 x → l3 y → ∀ z, l2 z → l3 z) 
  (h_parallel_13 : ∀ x y, l1 x → l3 y → ∀ z, l1 z → l3 z) 
  (A : ℝ) (hA : l1 A)
  (B : ℝ) (hB : l2 B)
  (C : ℝ) (hC : l3 C):
  ∃ A B C : ℝ, l1 A ∧ l2 B ∧ l3 C ∧ (dist A B = dist B C ∧ dist B C = dist C A) :=
by
  sorry

end equilateral_triangle_on_parallel_lines_l84_84502


namespace volume_ratio_l84_84711

theorem volume_ratio (A B C : ℚ) (h1 : (3/4) * A = (2/3) * B) (h2 : (2/3) * B = (1/2) * C) :
  A / C = 2 / 3 :=
sorry

end volume_ratio_l84_84711


namespace clarence_oranges_after_giving_l84_84509

def initial_oranges : ℝ := 5.0
def oranges_given : ℝ := 3.0

theorem clarence_oranges_after_giving : (initial_oranges - oranges_given) = 2.0 :=
by
  sorry

end clarence_oranges_after_giving_l84_84509


namespace men_apples_l84_84677

theorem men_apples (M W : ℕ) (h1 : M = W - 20) (h2 : 2 * M + 3 * W = 210) : M = 30 :=
by
  -- skipping the proof
  sorry

end men_apples_l84_84677


namespace original_price_l84_84899

variable (x : ℝ)

-- Condition 1: Selling at 60% of the original price results in a 20 yuan loss
def condition1 : Prop := 0.6 * x + 20 = x * 0.8 - 15

-- The goal is to prove that the original price is 175 yuan under the given conditions
theorem original_price (h : condition1 x) : x = 175 :=
sorry

end original_price_l84_84899


namespace original_price_of_petrol_l84_84746

variable (P : ℝ)

theorem original_price_of_petrol (h : 0.9 * (200 / P - 200 / (0.9 * P)) = 5) : 
  (P = 20 / 4.5) :=
sorry

end original_price_of_petrol_l84_84746


namespace minimize_acme_cost_l84_84428

theorem minimize_acme_cost (x : ℕ) : 75 + 12 * x < 16 * x → x = 19 :=
by
  intro h
  sorry

end minimize_acme_cost_l84_84428


namespace min_value_f_l84_84706

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x^2 + (1 / x)))

theorem min_value_f : ∃ x > 0, ∀ y > 0, f y ≥ f x ∧ f x = 5 / 2 :=
by
  sorry

end min_value_f_l84_84706


namespace laura_charges_for_truck_l84_84871

theorem laura_charges_for_truck : 
  ∀ (car_wash suv_wash truck_wash total_amount num_suvs num_trucks num_cars : ℕ),
  car_wash = 5 →
  suv_wash = 7 →
  num_suvs = 5 →
  num_trucks = 5 →
  num_cars = 7 →
  total_amount = 100 →
  car_wash * num_cars + suv_wash * num_suvs + truck_wash * num_trucks = total_amount →
  truck_wash = 6 :=
by
  intros car_wash suv_wash truck_wash total_amount num_suvs num_trucks num_cars h1 h2 h3 h4 h5 h6 h7
  sorry

end laura_charges_for_truck_l84_84871


namespace combined_salaries_of_A_B_C_E_is_correct_l84_84301

-- Given conditions
def D_salary : ℕ := 7000
def average_salary : ℕ := 8800
def n_individuals : ℕ := 5

-- Combined salary of A, B, C, and E
def combined_salaries : ℕ := 37000

theorem combined_salaries_of_A_B_C_E_is_correct :
  (average_salary * n_individuals - D_salary) = combined_salaries :=
by
  sorry

end combined_salaries_of_A_B_C_E_is_correct_l84_84301


namespace gcd_lcm_product_l84_84564

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 75) (h2 : b = 90) : Nat.gcd a b * Nat.lcm a b = 6750 :=
by
  sorry

end gcd_lcm_product_l84_84564


namespace xy_value_l84_84704

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 :=
by
  sorry

end xy_value_l84_84704


namespace kerosene_price_increase_l84_84747

theorem kerosene_price_increase (P C : ℝ) (x : ℝ)
  (h1 : 1 = (1 + x / 100) * 0.8) :
  x = 25 := by
  sorry

end kerosene_price_increase_l84_84747


namespace find_carbon_atoms_l84_84200

variable (n : ℕ)
variable (molecular_weight : ℝ := 124.0)
variable (weight_Cu : ℝ := 63.55)
variable (weight_C : ℝ := 12.01)
variable (weight_O : ℝ := 16.00)
variable (num_Cu : ℕ := 1)
variable (num_O : ℕ := 3)

theorem find_carbon_atoms 
  (h : molecular_weight = (num_Cu * weight_Cu) + (n * weight_C) + (num_O * weight_O)) : 
  n = 1 :=
sorry

end find_carbon_atoms_l84_84200


namespace amy_local_calls_l84_84490

-- Define the conditions as hypotheses
variable (L I : ℕ)
variable (h1 : L = (5 / 2 : ℚ) * I)
variable (h2 : L = (5 / 3 : ℚ) * (I + 3))

-- Statement of the theorem
theorem amy_local_calls : L = 15 := by
  sorry

end amy_local_calls_l84_84490


namespace calculate_Y_payment_l84_84309

theorem calculate_Y_payment (X Y : ℝ) (h1 : X + Y = 600) (h2 : X = 1.2 * Y) : Y = 600 / 2.2 :=
by
  sorry

end calculate_Y_payment_l84_84309


namespace no_ordered_triples_exist_l84_84044

theorem no_ordered_triples_exist :
  ¬ ∃ (x y z : ℤ), 
    (x^2 - 3 * x * y + 2 * y^2 - z^2 = 39) ∧
    (-x^2 + 6 * y * z + 2 * z^2 = 40) ∧
    (x^2 + x * y + 8 * z^2 = 96) :=
sorry

end no_ordered_triples_exist_l84_84044


namespace find_integer_k_l84_84674

theorem find_integer_k {k : ℤ} :
  (∀ x : ℝ, (k^2 + 1) * x^2 - (4 - k) * x + 1 = 0 →
    (∃ m n : ℝ, m ≠ n ∧ m * n = 1 / (k^2 + 1) ∧ m + n = (4 - k) / (k^2 + 1) ∧
      ((1 < m ∧ n < 1) ∨ (1 < n ∧ m < 1)))) →
  k = -1 ∨ k = 0 :=
by
  sorry

end find_integer_k_l84_84674


namespace malachi_selfies_total_l84_84347

theorem malachi_selfies_total (x y : ℕ) 
  (h_ratio : 10 * y = 17 * x)
  (h_diff : y = x + 630) : 
  x + y = 2430 :=
sorry

end malachi_selfies_total_l84_84347


namespace tangent_expression_equals_two_l84_84468

noncomputable def eval_tangent_expression : ℝ :=
  (1 + Real.tan (3 * Real.pi / 180)) * (1 + Real.tan (42 * Real.pi / 180))

theorem tangent_expression_equals_two :
  eval_tangent_expression = 2 :=
by sorry

end tangent_expression_equals_two_l84_84468


namespace decimal_to_base_five_l84_84073

theorem decimal_to_base_five : 
  (2 * 5^3 + 1 * 5^1 + 0 * 5^2 + 0 * 5^0 = 255) := 
by
  sorry

end decimal_to_base_five_l84_84073


namespace Bella_catch_correct_l84_84745

def Martha_catch : ℕ := 3 + 7
def Cara_catch : ℕ := 5 * Martha_catch - 3
def T : ℕ := Martha_catch + Cara_catch
def Andrew_catch : ℕ := T^2 + 2
def F : ℕ := Martha_catch + Cara_catch + Andrew_catch
def Bella_catch : ℕ := 2 ^ (F / 3)

theorem Bella_catch_correct : Bella_catch = 2 ^ 1102 := by
  sorry

end Bella_catch_correct_l84_84745


namespace no_complete_divisibility_l84_84179

-- Definition of non-divisibility
def not_divides (m n : ℕ) := ¬ (m ∣ n)

theorem no_complete_divisibility (a b c d : ℕ) (h : a * d - b * c > 1) : 
  not_divides (a * d - b * c) a ∨ not_divides (a * d - b * c) b ∨ not_divides (a * d - b * c) c ∨ not_divides (a * d - b * c) d :=
by 
  sorry

end no_complete_divisibility_l84_84179


namespace simplify_expression_l84_84192

theorem simplify_expression (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  ( (1/(a-b) - 2 * a * b / (a^3 - a^2 * b + a * b^2 - b^3)) / 
    ((a^2 + a * b) / (a^3 + a^2 * b + a * b^2 + b^3) + 
    b / (a^2 + b^2)) ) = (a - b) / (a + b) :=
by
  sorry

end simplify_expression_l84_84192


namespace amount_after_two_years_l84_84927

theorem amount_after_two_years (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)
  (hP : P = 64000) (hr : r = 1 / 6) (hn : n = 2) : 
  A = P * (1 + r) ^ n := by
  sorry

end amount_after_two_years_l84_84927


namespace second_option_cost_per_day_l84_84285

theorem second_option_cost_per_day :
  let distance_one_way := 150
  let rental_first_option := 50
  let kilometers_per_liter := 15
  let cost_per_liter := 0.9
  let savings := 22
  let total_distance := distance_one_way * 2
  let total_liters := total_distance / kilometers_per_liter
  let gasoline_cost := total_liters * cost_per_liter
  let total_cost_first_option := rental_first_option + gasoline_cost
  let second_option_cost := total_cost_first_option + savings
  second_option_cost = 90 :=
by
  sorry

end second_option_cost_per_day_l84_84285


namespace arithmetic_sequence_sum_l84_84443

theorem arithmetic_sequence_sum 
    (a : ℕ → ℤ)
    (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) -- Arithmetic sequence condition
    (h2 : a 5 = 3)
    (h3 : a 6 = -2) :
    (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3) :=
sorry

end arithmetic_sequence_sum_l84_84443


namespace female_employees_l84_84585

theorem female_employees (E M F : ℕ) (h1 : 300 = 300) (h2 : (2/5 : ℚ) * E = (2/5 : ℚ) * M + 300) (h3 : E = M + F) : F = 750 := 
by
  sorry

end female_employees_l84_84585


namespace find_multiple_l84_84356

theorem find_multiple (x m : ℝ) (h₁ : 10 * x = m * x - 36) (h₂ : x = -4.5) : m = 2 :=
by
  sorry

end find_multiple_l84_84356


namespace snowfall_difference_l84_84906

-- Defining all conditions given in the problem
def BaldMountain_snowfall_meters : ℝ := 1.5
def BillyMountain_snowfall_meters : ℝ := 3.5
def MountPilot_snowfall_centimeters : ℝ := 126
def RockstonePeak_snowfall_millimeters : ℝ := 5250
def SunsetRidge_snowfall_meters : ℝ := 2.25

-- Conversion constants
def meters_to_centimeters : ℝ := 100
def millimeters_to_centimeters : ℝ := 0.1

-- Converting snowfall amounts to centimeters
def BaldMountain_snowfall_centimeters : ℝ := BaldMountain_snowfall_meters * meters_to_centimeters
def BillyMountain_snowfall_centimeters : ℝ := BillyMountain_snowfall_meters * meters_to_centimeters
def RockstonePeak_snowfall_centimeters : ℝ := RockstonePeak_snowfall_millimeters * millimeters_to_centimeters
def SunsetRidge_snowfall_centimeters : ℝ := SunsetRidge_snowfall_meters * meters_to_centimeters

-- Defining total combined snowfall
def combined_snowfall_centimeters : ℝ :=
  BillyMountain_snowfall_centimeters + MountPilot_snowfall_centimeters + RockstonePeak_snowfall_centimeters + SunsetRidge_snowfall_centimeters

-- Stating the proof statement
theorem snowfall_difference :
  combined_snowfall_centimeters - BaldMountain_snowfall_centimeters = 1076 := 
  by
    sorry

end snowfall_difference_l84_84906


namespace cost_per_serving_of_pie_l84_84973

theorem cost_per_serving_of_pie 
  (w_gs : ℝ) (p_gs : ℝ) (w_gala : ℝ) (p_gala : ℝ) (w_hc : ℝ) (p_hc : ℝ)
  (pie_crust_cost : ℝ) (lemon_cost : ℝ) (butter_cost : ℝ) (servings : ℕ)
  (total_weight_gs : w_gs = 0.5) (price_gs_per_pound : p_gs = 1.80)
  (total_weight_gala : w_gala = 0.8) (price_gala_per_pound : p_gala = 2.20)
  (total_weight_hc : w_hc = 0.7) (price_hc_per_pound : p_hc = 2.50)
  (cost_pie_crust : pie_crust_cost = 2.50) (cost_lemon : lemon_cost = 0.60)
  (cost_butter : butter_cost = 1.80) (total_servings : servings = 8) :
  (w_gs * p_gs + w_gala * p_gala + w_hc * p_hc + pie_crust_cost + lemon_cost + butter_cost) / servings = 1.16 :=
by 
  sorry

end cost_per_serving_of_pie_l84_84973


namespace right_triangle_max_area_l84_84208

theorem right_triangle_max_area
  (a b : ℝ) (h_a_nonneg : 0 ≤ a) (h_b_nonneg : 0 ≤ b)
  (h_right_triangle : a^2 + b^2 = 20^2)
  (h_perimeter : a + b + 20 = 48) :
  (1 / 2) * a * b = 96 :=
by
  sorry

end right_triangle_max_area_l84_84208


namespace monotonicity_f_l84_84168

open Set

noncomputable def f (a x : ℝ) : ℝ := a * x / (x - 1)

theorem monotonicity_f (a : ℝ) (h : a ≠ 0) :
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → (if a > 0 then f a x1 > f a x2 else if a < 0 then f a x1 < f a x2 else False)) :=
by
  sorry

end monotonicity_f_l84_84168


namespace miles_in_one_hour_eq_8_l84_84890

-- Parameters as given in the conditions
variables (x : ℕ) (h1 : ∀ t : ℕ, t >= 6 → t % 6 = 0 ∨ t % 6 < 6)
variables (miles_in_one_hour : ℕ)
-- Given condition: The car drives 88 miles in 13 hours.
variable (miles_in_13_hours : miles_in_one_hour * 11 = 88)

-- Statement to prove: The car can drive 8 miles in one hour.
theorem miles_in_one_hour_eq_8 : miles_in_one_hour = 8 :=
by {
  -- Proof goes here
  sorry
}

end miles_in_one_hour_eq_8_l84_84890


namespace find_p_over_q_at_0_l84_84941

noncomputable def p (x : ℝ) := 3 * (x - 4) * (x - 1)
noncomputable def q (x : ℝ) := (x + 3) * (x - 1) * (x - 4)

theorem find_p_over_q_at_0 : (p 0) / (q 0) = 1 := 
by
  sorry

end find_p_over_q_at_0_l84_84941


namespace intersection_of_lines_l84_84417

theorem intersection_of_lines :
  ∃ (x y : ℝ), 10 * x - 5 * y = 5 ∧ 8 * x + 2 * y = 22 ∧ x = 2 ∧ y = 3 := by
  sorry

end intersection_of_lines_l84_84417


namespace dinner_customers_l84_84787

theorem dinner_customers 
    (breakfast : ℕ)
    (lunch : ℕ)
    (total_friday : ℕ)
    (H : breakfast = 73)
    (H1 : lunch = 127)
    (H2 : total_friday = 287) :
  (breakfast + lunch + D = total_friday) → D = 87 := by
  sorry

end dinner_customers_l84_84787


namespace max_height_of_basketball_l84_84976

def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 2

theorem max_height_of_basketball : ∃ t : ℝ, h t = 127 :=
by
  use 5
  sorry

end max_height_of_basketball_l84_84976


namespace strawberry_cost_l84_84425

variables (S C : ℝ)

theorem strawberry_cost :
  (C = 6 * S) ∧ (5 * S + 5 * C = 77) → S = 2.2 :=
by
  sorry

end strawberry_cost_l84_84425


namespace atLeastOneTrueRange_exactlyOneTrueRange_l84_84515

-- Definitions of Proposition A and B
def propA (a : ℝ) : Prop := ∀ x, x^2 + (a - 1) * x + a^2 ≤ 0 → false
def propB (a : ℝ) : Prop := ∀ x, (2 * a^2 - a)^x < (2 * a^2 - a)^(x + 1)

-- At least one of A or B is true
def atLeastOneTrue (a : ℝ) : Prop :=
  propA a ∨ propB a

-- Exactly one of A or B is true
def exactlyOneTrue (a : ℝ) : Prop := 
  (propA a ∧ ¬ propB a) ∨ (¬ propA a ∧ propB a)

-- Theorems to prove
theorem atLeastOneTrueRange :
  ∃ a : ℝ, atLeastOneTrue a ↔ (a < -1/2 ∨ a > 1/3) := 
sorry

theorem exactlyOneTrueRange :
  ∃ a : ℝ, exactlyOneTrue a ↔ ((1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2)) :=
sorry

end atLeastOneTrueRange_exactlyOneTrueRange_l84_84515


namespace gcd_3_pow_1007_minus_1_3_pow_1018_minus_1_l84_84426

theorem gcd_3_pow_1007_minus_1_3_pow_1018_minus_1 :
  Nat.gcd (3^1007 - 1) (3^1018 - 1) = 177146 :=
by
  -- Proof follows from the Euclidean algorithm and factoring, skipping the proof here.
  sorry

end gcd_3_pow_1007_minus_1_3_pow_1018_minus_1_l84_84426


namespace C_investment_l84_84075

theorem C_investment (A B total_profit A_share : ℝ) (x : ℝ) :
  A = 6300 → B = 4200 → total_profit = 12600 → A_share = 3780 →
  (A / (A + B + x) = A_share / total_profit) → x = 10500 :=
by
  intros hA hB h_total_profit h_A_share h_ratio
  sorry

end C_investment_l84_84075


namespace certain_event_is_eventC_l84_84306

-- Definitions for the conditions:
def eventA := "A vehicle randomly arriving at an intersection encountering a red light"
def eventB := "The sun rising from the west in the morning"
def eventC := "Two out of 400 people sharing the same birthday"
def eventD := "Tossing a fair coin with the head facing up"

-- The proof goal: proving that event C is the certain event.
theorem certain_event_is_eventC : eventC = "Two out of 400 people sharing the same birthday" :=
sorry

end certain_event_is_eventC_l84_84306


namespace hyperbola_eccentricity_l84_84792

theorem hyperbola_eccentricity
  (a b m : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (PA_perpendicular_to_l2 : (b/a * m) / (m + a) * (-b/a) = -1)
  (PB_parallel_to_l2 : (b/a * m) / (m - a) = -b/a) :
  (∃ e, e = 2) :=
by sorry

end hyperbola_eccentricity_l84_84792


namespace recommended_cups_l84_84617

theorem recommended_cups (current_cups : ℕ) (R : ℕ) : 
  current_cups = 20 →
  R = current_cups + (6 / 10) * current_cups →
  R = 32 :=
by
  intros h1 h2
  sorry

end recommended_cups_l84_84617


namespace solutions_are__l84_84510

def satisfies_system (x y z : ℝ) : Prop :=
  x^2 * y + y^2 * z = 1040 ∧
  x^2 * z + z^2 * y = 260 ∧
  (x - y) * (y - z) * (z - x) = -540

theorem solutions_are_ (x y z : ℝ) :
  satisfies_system x y z ↔ (x = 16 ∧ y = 4 ∧ z = 1) ∨ (x = 1 ∧ y = 16 ∧ z = 4) :=
by
  sorry

end solutions_are__l84_84510


namespace fruit_basket_count_l84_84109

theorem fruit_basket_count :
  let pears := 8
  let bananas := 12
  let total_baskets := (pears + 1) * (bananas + 1) - 1
  total_baskets = 116 :=
by
  sorry

end fruit_basket_count_l84_84109


namespace third_side_length_l84_84791

theorem third_side_length (a b : ℝ) (h : (a - 3) ^ 2 + |b - 4| = 0) :
  ∃ x : ℝ, (a = 3 ∧ b = 4) ∧ (x = 5 ∨ x = Real.sqrt 7) :=
by
  sorry

end third_side_length_l84_84791


namespace area_y_eq_x2_y_eq_x3_l84_84186

noncomputable section

open Real

def area_closed_figure_between_curves : ℝ :=
  ∫ x in (0:ℝ)..(1:ℝ), (x^2 - x^3)

theorem area_y_eq_x2_y_eq_x3 :
  area_closed_figure_between_curves = 1 / 12 := by
  sorry

end area_y_eq_x2_y_eq_x3_l84_84186


namespace central_park_trash_cans_more_than_half_l84_84517

theorem central_park_trash_cans_more_than_half
  (C : ℕ)  -- Original number of trash cans in Central Park
  (V : ℕ := 24)  -- Original number of trash cans in Veteran's Park
  (V_now : ℕ := 34)  -- Number of trash cans in Veteran's Park after the move
  (H_move : (V_now - V) = C / 2)  -- Condition of trash cans moved
  (H_C : C = (1 / 2) * V + x)  -- Central Park had more than half trash cans as Veteran's Park, where x is an excess amount
  : C - (1 / 2) * V = 8 := 
sorry

end central_park_trash_cans_more_than_half_l84_84517


namespace work_together_days_l84_84708

noncomputable def A_per_day := 1 / 78
noncomputable def B_per_day := 1 / 39

theorem work_together_days 
  (A : ℝ) (B : ℝ) 
  (hA : A = 1 / 78)
  (hB : B = 1 / 39) : 
  1 / (A + B) = 26 :=
by
  rw [hA, hB]
  sorry

end work_together_days_l84_84708


namespace sum_of_arithmetic_sequence_l84_84043

variable (S : ℕ → ℝ)

def arithmetic_seq_property (S : ℕ → ℝ) : Prop :=
  S 4 = 4 ∧ S 8 = 12

theorem sum_of_arithmetic_sequence (h : arithmetic_seq_property S) : S 12 = 24 :=
by
  sorry

end sum_of_arithmetic_sequence_l84_84043


namespace red_balls_estimate_l84_84128

/-- There are several red balls and 4 black balls in a bag.
Each ball is identical except for color.
A ball is drawn and put back into the bag. This process is repeated 100 times.
Among those 100 draws, 40 times a black ball is drawn.
Prove that the number of red balls (x) is 6. -/
theorem red_balls_estimate (x : ℕ) (h_condition : (4 / (4 + x) = 40 / 100)) : x = 6 :=
by
    sorry

end red_balls_estimate_l84_84128


namespace div_seven_and_sum_factors_l84_84227

theorem div_seven_and_sum_factors (a b c : ℤ) (h : (a = 0 ∨ b = 0 ∨ c = 0) ∧ ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  ∃ k : ℤ, (a + b + c)^7 - a^7 - b^7 - c^7 = k * 7 * (a + b) * (b + c) * (c + a) :=
by
  sorry

end div_seven_and_sum_factors_l84_84227


namespace part1_part2_l84_84264

-- Part 1
theorem part1 (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  |(3 * x - 4 * x^3)| ≤ 1 := sorry

-- Part 2
theorem part2 (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  |(3 * x - 4 * x^3)| ≤ 1 := sorry

end part1_part2_l84_84264


namespace total_paintable_area_correct_l84_84700

namespace BarnPainting

-- Define the dimensions of the barn
def barn_width : ℕ := 12
def barn_length : ℕ := 15
def barn_height : ℕ := 6

-- Define the dimensions of the windows
def window_width : ℕ := 2
def window_height : ℕ := 3
def num_windows : ℕ := 2

-- Calculate the total number of square yards to be painted
def total_paintable_area : ℕ :=
  let wall1_area := barn_height * barn_width
  let wall2_area := barn_height * barn_length
  let wall_area := 2 * wall1_area + 2 * wall2_area
  let window_area := num_windows * (window_width * window_height)
  let painted_walls_area := wall_area - window_area
  let ceiling_area := barn_width * barn_length
  let total_area := 2 * painted_walls_area + ceiling_area
  total_area

theorem total_paintable_area_correct : total_paintable_area = 780 :=
  by sorry

end BarnPainting

end total_paintable_area_correct_l84_84700


namespace man_swim_upstream_distance_l84_84414

theorem man_swim_upstream_distance (c d : ℝ) (h1 : 15.5 + c ≠ 0) (h2 : 15.5 - c ≠ 0) :
  (15.5 + c) * 2 = 36 ∧ (15.5 - c) * 2 = d → d = 26 := by
  sorry

end man_swim_upstream_distance_l84_84414


namespace remainder_2345678901_div_101_l84_84104

theorem remainder_2345678901_div_101 : 2345678901 % 101 = 12 :=
sorry

end remainder_2345678901_div_101_l84_84104


namespace problem_1_problem_2_l84_84641

-- Definition of sets A and B as in the problem's conditions
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | x > 2 ∨ x < -2}
def C (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

-- Prove that A ∩ B is as described
theorem problem_1 : A ∩ B = {x | 2 < x ∧ x ≤ 5} := by
  sorry

-- Prove that a ≥ 6 given the conditions in the problem
theorem problem_2 (a : ℝ) : (A ⊆ C a) → a ≥ 6 := by
  sorry

end problem_1_problem_2_l84_84641


namespace binomial_9_3_l84_84124

theorem binomial_9_3 : (Nat.choose 9 3) = 84 := by
  sorry

end binomial_9_3_l84_84124


namespace solve_system_of_equations_l84_84729

theorem solve_system_of_equations :
  ∃ x y : ℝ, (x - y = 2) ∧ (2 * x + y = 7) ∧ x = 3 ∧ y = 1 :=
by
  sorry

end solve_system_of_equations_l84_84729


namespace average_payment_52_installments_l84_84606

theorem average_payment_52_installments :
  let first_payment : ℕ := 500
  let remaining_payment : ℕ := first_payment + 100
  let num_first_payments : ℕ := 25
  let num_remaining_payments : ℕ := 27
  let total_payments : ℕ := num_first_payments + num_remaining_payments
  let total_paid_first : ℕ := num_first_payments * first_payment
  let total_paid_remaining : ℕ := num_remaining_payments * remaining_payment
  let total_paid : ℕ := total_paid_first + total_paid_remaining
  let average_payment : ℚ := total_paid / total_payments
  average_payment = 551.92 :=
by
  sorry

end average_payment_52_installments_l84_84606


namespace like_terms_sum_l84_84934

theorem like_terms_sum (m n : ℕ) (h1 : m = 3) (h2 : 4 = n + 2) : m + n = 5 :=
by
  sorry

end like_terms_sum_l84_84934


namespace value_of_expression_l84_84905

variables (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 + b * x ^ 2 + c * x + d

theorem value_of_expression (h : f a b c d (-2) = -3) : 8 * a - 4 * b + 2 * c - d = 3 :=
by {
  sorry
}

end value_of_expression_l84_84905


namespace solve_abs_equation_l84_84157

theorem solve_abs_equation (x : ℝ) (h : |2001 * x - 2001| = 2001) : x = 0 ∨ x = 2 := by
  sorry

end solve_abs_equation_l84_84157


namespace together_complete_days_l84_84623

-- Define the work rates of x and y
def work_rate_x := (1 : ℚ) / 30
def work_rate_y := (1 : ℚ) / 45

-- Define the combined work rate when x and y work together
def combined_work_rate := work_rate_x + work_rate_y

-- Define the number of days to complete the work together
def days_to_complete_work := 1 / combined_work_rate

-- The theorem we want to prove
theorem together_complete_days : days_to_complete_work = 18 := by
  sorry

end together_complete_days_l84_84623


namespace linda_max_servings_is_13_l84_84415

noncomputable def max_servings 
  (recipe_bananas : ℕ) (recipe_yogurt : ℕ) (recipe_honey : ℕ)
  (linda_bananas : ℕ) (linda_yogurt : ℕ) (linda_honey : ℕ)
  (servings_for_recipe : ℕ) : ℕ :=
  min 
    (linda_bananas * servings_for_recipe / recipe_bananas) 
    (min 
      (linda_yogurt * servings_for_recipe / recipe_yogurt)
      (linda_honey * servings_for_recipe / recipe_honey)
    )

theorem linda_max_servings_is_13 : 
  max_servings 3 2 1 10 9 4 4 = 13 :=
  sorry

end linda_max_servings_is_13_l84_84415


namespace cristine_final_lemons_l84_84090

def cristine_lemons_initial : ℕ := 12
def cristine_lemons_given_to_neighbor : ℕ := 1 / 4 * cristine_lemons_initial
def cristine_lemons_left_after_giving : ℕ := cristine_lemons_initial - cristine_lemons_given_to_neighbor
def cristine_lemons_exchanged_for_oranges : ℕ := 1 / 3 * cristine_lemons_left_after_giving
def cristine_lemons_left_after_exchange : ℕ := cristine_lemons_left_after_giving - cristine_lemons_exchanged_for_oranges

theorem cristine_final_lemons : cristine_lemons_left_after_exchange = 6 :=
by
  sorry

end cristine_final_lemons_l84_84090


namespace cosine_between_vectors_l84_84280

noncomputable def vector_cos_angle (a b : ℝ × ℝ) := 
  let dot_product := (a.1 * b.1) + (a.2 * b.2)
  let norm_a := Real.sqrt (a.1 * a.1 + a.2 * a.2)
  let norm_b := Real.sqrt (b.1 * b.1 + b.2 * b.2)
  dot_product / (norm_a * norm_b)

theorem cosine_between_vectors (t : ℝ) 
  (ht : let a := (1, t); let b := (-1, 2 * t);
        (3 * a.1 - b.1) * b.1 + (3 * a.2 - b.2) * b.2 = 0) :
  vector_cos_angle (1, t) (-1, 2 * t) = Real.sqrt 3 / 3 := 
by
  sorry

end cosine_between_vectors_l84_84280


namespace no_positive_x_for_volume_l84_84279

noncomputable def volume (x : ℤ) : ℤ :=
  (x + 5) * (x - 7) * (x^2 + x + 30)

theorem no_positive_x_for_volume : ¬ ∃ x : ℕ, 0 < x ∧ volume x < 800 := by
  sorry

end no_positive_x_for_volume_l84_84279


namespace waiter_customers_l84_84769

-- Define initial conditions
def initial_customers : ℕ := 47
def customers_left : ℕ := 41
def new_customers : ℕ := 20

-- Calculate remaining customers after some left
def remaining_customers : ℕ := initial_customers - customers_left

-- Calculate the total customers after getting new ones
def total_customers : ℕ := remaining_customers + new_customers

-- State the theorem to prove the final total customers
theorem waiter_customers : total_customers = 26 := by
  -- We include sorry for the proof placeholder
  sorry

end waiter_customers_l84_84769


namespace jim_saves_by_buying_gallon_l84_84471

-- Define the conditions as variables
def cost_per_gallon_costco : ℕ := 8
def ounces_per_gallon : ℕ := 128
def cost_per_16oz_bottle_store : ℕ := 3
def ounces_per_bottle : ℕ := 16

-- Define the theorem that needs to be proven
theorem jim_saves_by_buying_gallon (h1 : cost_per_gallon_costco = 8)
                                    (h2 : ounces_per_gallon = 128)
                                    (h3 : cost_per_16oz_bottle_store = 3)
                                    (h4 : ounces_per_bottle = 16) : 
  (8 * 3 - 8) = 16 :=
by sorry

end jim_saves_by_buying_gallon_l84_84471


namespace quadratic_interval_solution_l84_84754

open Set

def quadratic_function (x : ℝ) : ℝ := x^2 + 5 * x + 6

theorem quadratic_interval_solution :
  {x : ℝ | 6 ≤ quadratic_function x ∧ quadratic_function x ≤ 12} = {x | -6 ≤ x ∧ x ≤ -5} ∪ {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end quadratic_interval_solution_l84_84754


namespace lions_min_games_for_90_percent_wins_l84_84823

theorem lions_min_games_for_90_percent_wins : 
  ∀ N : ℕ, (N ≥ 26) ↔ 1 + N ≥ (9 * (4 + N)) / 10 := 
by 
  sorry

end lions_min_games_for_90_percent_wins_l84_84823


namespace jane_bought_two_bagels_l84_84040

variable (b m d k : ℕ)

def problem_conditions : Prop :=
  b + m + d = 6 ∧ 
  (60 * b + 45 * m + 30 * d) = 100 * k

theorem jane_bought_two_bagels (hb : problem_conditions b m d k) : b = 2 :=
  sorry

end jane_bought_two_bagels_l84_84040


namespace part1_part2_l84_84289

def f (x a : ℝ) : ℝ := |x + 1| - |x - a|

theorem part1 (x : ℝ) : (f x 2 > 2) ↔ (x > 3 / 2) :=
sorry

theorem part2 (a : ℝ) (ha : a > 0) : (∀ x, f x a < 2 * a) ↔ (1 < a) :=
sorry

end part1_part2_l84_84289


namespace largest_quantity_l84_84924

noncomputable def A := (2006 / 2005) + (2006 / 2007)
noncomputable def B := (2006 / 2007) + (2008 / 2007)
noncomputable def C := (2007 / 2006) + (2007 / 2008)

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end largest_quantity_l84_84924


namespace unattainable_y_l84_84715

theorem unattainable_y (x : ℚ) (y : ℚ) (h : y = (1 - 2 * x) / (3 * x + 4)) (hx : x ≠ -4 / 3) : y ≠ -2 / 3 :=
by {
  sorry
}

end unattainable_y_l84_84715


namespace area_of_polygon_ABCDEF_l84_84162

-- Definitions based on conditions
def AB : ℕ := 8
def BC : ℕ := 10
def DC : ℕ := 5
def FA : ℕ := 7
def GF : ℕ := 3
def ED : ℕ := 7
def height_GF_ED : ℕ := 2

-- Area calculations based on given conditions
def area_ABCG : ℕ := AB * BC
def area_trapezoid_GFED : ℕ := (GF + ED) * height_GF_ED / 2

-- Proof statement
theorem area_of_polygon_ABCDEF :
  area_ABCG - area_trapezoid_GFED = 70 :=
by
  simp [area_ABCG, area_trapezoid_GFED]
  sorry

end area_of_polygon_ABCDEF_l84_84162


namespace sample_size_second_grade_l84_84193

theorem sample_size_second_grade
    (total_students : ℕ)
    (ratio_first : ℕ)
    (ratio_second : ℕ)
    (ratio_third : ℕ)
    (sample_size : ℕ) :
    total_students = 2000 →
    ratio_first = 5 → ratio_second = 3 → ratio_third = 2 →
    sample_size = 20 →
    (20 * (3 / (5 + 3 + 2)) = 6) :=
by
  intros ht hr1 hr2 hr3 hs
  -- The proof would continue from here, but we're finished as the task only requires the statement.
  sorry

end sample_size_second_grade_l84_84193


namespace poles_needed_l84_84673

theorem poles_needed (L W : ℕ) (dist : ℕ)
  (hL : L = 90) (hW : W = 40) (hdist : dist = 5) :
  (2 * (L + W)) / dist = 52 :=
by 
  sorry

end poles_needed_l84_84673


namespace ratio_of_times_l84_84135

theorem ratio_of_times (D S : ℝ) (hD : D = 27) (hS : S / 2 = D / 2 + 13.5) :
  D / S = 1 / 2 :=
by
  -- the proof will go here
  sorry

end ratio_of_times_l84_84135


namespace table_fill_impossible_l84_84682

/-- Proposition: Given a 7x3 table filled with 0s and 1s, it is impossible to prevent any 2x2 submatrix from having all identical numbers. -/
theorem table_fill_impossible : 
  ¬ ∃ (M : (Fin 7) → (Fin 3) → Fin 2), 
      ∀ i j, (i < 6) → (j < 2) → 
              (M i j = M i.succ j) ∨ 
              (M i j = M i j.succ) ∨ 
              (M i j = M i.succ j.succ) ∨ 
              (M i.succ j = M i j.succ → M i j = M i.succ j.succ) :=
sorry

end table_fill_impossible_l84_84682


namespace find_function_l84_84898

theorem find_function (f : ℝ → ℝ)
  (h₁ : ∀ x : ℝ, x * (f (x + 1) - f x) = f x)
  (h₂ : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ |k| ≤ 1 :=
sorry

end find_function_l84_84898


namespace oxygen_atoms_l84_84583

theorem oxygen_atoms (x : ℤ) (h : 27 + 16 * x + 3 = 78) : x = 3 := 
by 
  sorry

end oxygen_atoms_l84_84583


namespace maximum_xyzw_l84_84532

theorem maximum_xyzw (x y z w : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_pos_w : 0 < w)
(h : (x * y * z) + w = (x + w) * (y + w) * (z + w))
(h_sum : x + y + z + w = 1) :
  xyzw = 1 / 256 :=
sorry

end maximum_xyzw_l84_84532


namespace john_average_speed_l84_84235

/--
John drove continuously from 8:15 a.m. until 2:05 p.m. of the same day 
and covered a distance of 210 miles. Prove that his average speed in 
miles per hour was 36 mph.
-/
theorem john_average_speed :
  (210 : ℝ) / (((2 - 8) * 60 + 5 - 15) / 60) = 36 := by
  sorry

end john_average_speed_l84_84235


namespace initial_speed_is_sixty_l84_84782

variable (D T : ℝ)

-- Condition: Two-thirds of the distance is covered in one-third of the total time.
def two_thirds_distance_in_one_third_time (V : ℝ) : Prop :=
  (2 * D / 3) / V = T / 3

-- Condition: The remaining distance is covered at 15 kmph.
def remaining_distance_at_fifteen_kmph : Prop :=
  (D / 3) / 15 = T - T / 3

-- Given that 30T = D from simplification in the solution.
def distance_time_relationship : Prop :=
  D = 30 * T

-- Prove that the initial speed V is 60 kmph.
theorem initial_speed_is_sixty (V : ℝ) (h1 : two_thirds_distance_in_one_third_time D T V) (h2 : remaining_distance_at_fifteen_kmph D T) (h3 : distance_time_relationship D T) : V = 60 := 
  sorry

end initial_speed_is_sixty_l84_84782


namespace find_a1_l84_84770

variable (a : ℕ → ℚ) (d : ℚ)
variable (S : ℕ → ℚ)
variable (h_seq : ∀ n, a (n + 1) = a n + d)
variable (h_diff : d ≠ 0)
variable (h_prod : (a 2) * (a 3) = (a 4) * (a 5))
variable (h_sum : S 4 = 27)
variable (h_sum_def : ∀ n, S n = n * (a 1 + a n) / 2)

theorem find_a1 : a 1 = 135 / 8 := by
  sorry

end find_a1_l84_84770


namespace exists_integer_root_l84_84290

theorem exists_integer_root (a b c d : ℤ) (ha : a ≠ 0)
  (h : ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * (a * x^3 + b * x^2 + c * x + d) = y * (a * y^3 + b * y^2 + c * y + d)) :
  ∃ z : ℤ, a * z^3 + b * z^2 + c * z + d = 0 :=
by
  sorry

end exists_integer_root_l84_84290


namespace tree_growth_rate_l84_84983

noncomputable def growth_rate_per_week (initial_height final_height : ℝ) (months weeks_per_month : ℕ) : ℝ :=
  (final_height - initial_height) / (months * weeks_per_month)

theorem tree_growth_rate :
  growth_rate_per_week 10 42 4 4 = 2 := 
by
  sorry

end tree_growth_rate_l84_84983


namespace sum_max_min_values_l84_84640

noncomputable def y (x : ℝ) : ℝ := 2 * x^2 + 32 / x

theorem sum_max_min_values :
  y 1 = 34 ∧ y 2 = 24 ∧ y 4 = 40 → ((y 4 + y 2) = 64) :=
by
  sorry

end sum_max_min_values_l84_84640


namespace solution_set_of_inequality_l84_84676

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2
  else if x > 0 then x - 2
  else 0

theorem solution_set_of_inequality :
  {x : ℝ | 2 * f x - 1 < 0} = {x | x < -3 / 2 ∨ (0 ≤ x ∧ x < 5 / 2)} :=
by
  sorry

end solution_set_of_inequality_l84_84676


namespace time_for_model_M_l84_84629

variable (T : ℝ) -- Time taken by model M computer to complete the task in minutes.
variable (n_m : ℝ := 12) -- Number of model M computers
variable (n_n : ℝ := 12) -- Number of model N computers
variable (time_n : ℝ := 18) -- Time taken by model N computer to complete the task in minutes

theorem time_for_model_M :
  n_m / T + n_n / time_n = 1 → T = 36 := by
sorry

end time_for_model_M_l84_84629


namespace relationship_among_abc_l84_84021

noncomputable def a : ℝ := 2 ^ (3 / 2)
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2
noncomputable def c : ℝ := 0.8 ^ 2

theorem relationship_among_abc : b < c ∧ c < a := 
by
  -- these are conditions directly derived from the problem
  let h1 : a = 2 ^ (3 / 2) := rfl
  let h2 : b = Real.log 0.3 / Real.log 2 := rfl
  let h3 : c = 0.8 ^ 2 := rfl
  sorry

end relationship_among_abc_l84_84021


namespace find_prices_min_cost_l84_84284

-- Definitions based on conditions
def price_difference (x y : ℕ) : Prop := x - y = 50
def total_cost (x y : ℕ) : Prop := 2 * x + 3 * y = 250
def cost_function (a : ℕ) : ℕ := 50 * a + 6000
def min_items (a : ℕ) : Prop := a ≥ 80
def total_items : ℕ := 200

-- Lean 4 statements for the proof problem
theorem find_prices (x y : ℕ) (h1 : price_difference x y) (h2 : total_cost x y) :
  (x = 80) ∧ (y = 30) :=
sorry

theorem min_cost (a : ℕ) (h1 : min_items a) :
  cost_function a ≥ 10000 :=
sorry

#check find_prices
#check min_cost

end find_prices_min_cost_l84_84284


namespace circle_center_polar_coords_l84_84717

noncomputable def polar_center (ρ θ : ℝ) : (ℝ × ℝ) :=
  (-1, 0)

theorem circle_center_polar_coords : 
  ∀ ρ θ : ℝ, ρ = -2 * Real.cos θ → polar_center ρ θ = (1, π) :=
by
  intro ρ θ h
  sorry

end circle_center_polar_coords_l84_84717


namespace find_values_of_A_l84_84837

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_values_of_A (A B C : ℕ) :
  sum_of_digits A = B ∧
  sum_of_digits B = C ∧
  A + B + C = 60 →
  (A = 44 ∨ A = 50 ∨ A = 47) :=
by
  sorry

end find_values_of_A_l84_84837


namespace solve_fractional_equation_l84_84609

theorem solve_fractional_equation (x : ℚ) (h1 : x ≠ 4) (h2 : x ≠ -6) :
    (x + 11) / (x - 4) = (x - 3) / (x + 6) ↔ x = -9 / 4 :=
by
  sorry

end solve_fractional_equation_l84_84609


namespace cups_of_ketchup_l84_84274

-- Define variables and conditions
variables (k : ℕ)
def vinegar : ℕ := 1
def honey : ℕ := 1
def sauce_per_burger : ℚ := 1 / 4
def sauce_per_pulled_pork : ℚ := 1 / 6
def burgers : ℕ := 8
def pulled_pork_sandwiches : ℕ := 18

-- Main theorem statement
theorem cups_of_ketchup (h : 8 * sauce_per_burger + 18 * sauce_per_pulled_pork = k + vinegar + honey) : k = 3 :=
  by
    sorry

end cups_of_ketchup_l84_84274


namespace positive_difference_of_numbers_l84_84220

theorem positive_difference_of_numbers (x : ℝ) (h : (30 + x) / 2 = 34) : abs (x - 30) = 8 :=
by
  sorry

end positive_difference_of_numbers_l84_84220


namespace smallest_common_term_larger_than_2023_l84_84323

noncomputable def a_seq (n : ℕ) : ℤ :=
  3 * n - 2

noncomputable def b_seq (m : ℕ) : ℤ :=
  10 * m - 8

theorem smallest_common_term_larger_than_2023 :
  ∃ (n m : ℕ), a_seq n = b_seq m ∧ a_seq n > 2023 ∧ a_seq n = 2032 :=
by {
  sorry
}

end smallest_common_term_larger_than_2023_l84_84323


namespace a0_a1_consecutive_l84_84097

variable (a : ℕ → ℤ)
variable (cond : ∀ i ≥ 2, a i = 2 * a (i - 1) - a (i - 2) ∨ a i = 2 * a (i - 2) - a (i - 1))
variable (consec : |a 2024 - a 2023| = 1)

theorem a0_a1_consecutive :
  |a 1 - a 0| = 1 :=
by
  -- Proof skipped
  sorry

end a0_a1_consecutive_l84_84097


namespace library_books_l84_84147

theorem library_books (a : ℕ) (R L : ℕ) :
  (∃ R, a = 12 * R + 7) ∧ (∃ L, a = 25 * L - 5) ∧ 500 < a ∧ a < 650 → a = 595 :=
by
  sorry

end library_books_l84_84147


namespace find_initial_balance_l84_84011

-- Define the initial balance
variable (X : ℝ)

-- Conditions
def balance_tripled (X : ℝ) : ℝ := 3 * X
def balance_after_withdrawal (X : ℝ) : ℝ := balance_tripled X - 250

-- The problem statement to prove
theorem find_initial_balance (h : balance_after_withdrawal X = 950) : X = 400 :=
by
  sorry

end find_initial_balance_l84_84011


namespace trigonometric_identity_proof_l84_84048

theorem trigonometric_identity_proof :
  3.438 * (Real.sin (84 * Real.pi / 180)) * (Real.sin (24 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * (Real.sin (12 * Real.pi / 180)) = 1 / 16 :=
  sorry

end trigonometric_identity_proof_l84_84048


namespace find_m_and_f_max_l84_84689

noncomputable def f (x m : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin (2 * x) + 2 * (Real.cos x)^2 + m

theorem find_m_and_f_max (m a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≥ 3) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), ∃ y, f y m = 3) →
  (∀ x ∈ Set.Icc a (a + Real.pi), ∃ y, f y m = 6) →
  m = 3 ∧ ∀ x ∈ Set.Icc a (a + Real.pi), f x 3 ≤ 6 :=
sorry

end find_m_and_f_max_l84_84689


namespace tank_overflow_time_l84_84897

noncomputable def pipeARate : ℚ := 1 / 32
noncomputable def pipeBRate : ℚ := 3 * pipeARate
noncomputable def combinedRate (rateA rateB : ℚ) : ℚ := rateA + rateB

theorem tank_overflow_time : 
  combinedRate pipeARate pipeBRate = 1 / 8 ∧ (1 / combinedRate pipeARate pipeBRate = 8) :=
by
  sorry

end tank_overflow_time_l84_84897


namespace runners_meet_again_l84_84979

theorem runners_meet_again :
    ∀ t : ℝ,
      t ≠ 0 →
      (∃ k : ℤ, 3.8 * t - 4 * t = 400 * k) ∧
      (∃ m : ℤ, 4.2 * t - 4 * t = 400 * m) ↔
      t = 2000 := 
by
  sorry

end runners_meet_again_l84_84979


namespace Andy_solves_correct_number_of_problems_l84_84800

-- Define the problem boundaries
def first_problem : ℕ := 80
def last_problem : ℕ := 125

-- The goal is to prove that Andy solves 46 problems given the range
theorem Andy_solves_correct_number_of_problems : (last_problem - first_problem + 1) = 46 :=
by
  sorry

end Andy_solves_correct_number_of_problems_l84_84800


namespace min_m_value_l84_84148

noncomputable def f (x a : ℝ) : ℝ := 2 ^ (abs (x - a))

theorem min_m_value :
  ∀ a, (∀ x, f (1 + x) a = f (1 - x) a) →
  ∃ m : ℝ, (∀ x : ℝ, x ≥ m → ∀ y : ℝ, y ≥ x → f y a ≥ f x a) ∧ m = 1 :=
by
  intros a h
  sorry

end min_m_value_l84_84148


namespace range_of_x_l84_84070

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x - 4| + |x - a|

theorem range_of_x (a : ℝ) (h1 : a > 1) (h2 : ∀ x : ℝ, f x a ≥ |a - 4|) (h3 : |a - 4| = 3) :
  { x : ℝ | f x a ≤ 5 } = { x : ℝ | 3 ≤ x ∧ x ≤ 8 } := 
sorry

end range_of_x_l84_84070


namespace diving_competition_score_l84_84832

theorem diving_competition_score 
  (scores : List ℝ)
  (h : scores = [7.5, 8.0, 9.0, 6.0, 8.8])
  (degree_of_difficulty : ℝ)
  (hd : degree_of_difficulty = 3.2) :
  let sorted_scores := scores.erase 9.0 |>.erase 6.0
  let remaining_sum := sorted_scores.sum
  remaining_sum * degree_of_difficulty = 77.76 :=
by
  sorry

end diving_competition_score_l84_84832


namespace one_number_greater_than_one_l84_84030

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_prod: a * b * c = 1)
  (h_sum: a + b + c > 1 / a + 1 / b + 1 / c) :
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
by
  sorry

end one_number_greater_than_one_l84_84030


namespace max_value_10x_plus_3y_plus_12z_l84_84061

theorem max_value_10x_plus_3y_plus_12z (x y z : ℝ) 
  (h1 : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) 
  (h2 : z = 2 * y) : 
  10 * x + 3 * y + 12 * z ≤ Real.sqrt 253 :=
sorry

end max_value_10x_plus_3y_plus_12z_l84_84061


namespace lab_preparation_is_correct_l84_84992

def correct_operation (m_CuSO4 : ℝ) (m_CuSO4_5H2O : ℝ) (V_solution : ℝ) : Prop :=
  let molar_mass_CuSO4 := 160 -- g/mol
  let molar_mass_CuSO4_5H2O := 250 -- g/mol
  let desired_concentration := 0.1 -- mol/L
  let desired_volume := 0.480 -- L
  let prepared_volume := 0.500 -- L
  (m_CuSO4 = 8.0 ∧ V_solution = 0.500 ∧ m_CuSO4_5H2O = 12.5 ∧ desired_concentration * prepared_volume * molar_mass_CuSO4_5H2O = 12.5)

-- Example proof statement to show the problem with "sorry"
theorem lab_preparation_is_correct : correct_operation 8.0 12.5 0.500 :=
by
  sorry

end lab_preparation_is_correct_l84_84992


namespace scientific_notation_of_million_l84_84246

theorem scientific_notation_of_million : 1000000 = 10^6 :=
by
  sorry

end scientific_notation_of_million_l84_84246


namespace total_weekly_pay_l84_84613

theorem total_weekly_pay (Y_pay: ℝ) (X_pay: ℝ) (Y_weekly: Y_pay = 150) (X_weekly: X_pay = 1.2 * Y_pay) : 
  X_pay + Y_pay = 330 :=
by sorry

end total_weekly_pay_l84_84613


namespace P2011_1_neg1_is_0_2_pow_1006_l84_84411

def P1 (x y : ℤ) : ℤ × ℤ := (x + y, x - y)

def Pn : ℕ → ℤ → ℤ → ℤ × ℤ 
| 0, x, y => (x, y)
| (n + 1), x, y => P1 (Pn n x y).1 (Pn n x y).2

theorem P2011_1_neg1_is_0_2_pow_1006 : Pn 2011 1 (-1) = (0, 2^1006) := by
  sorry

end P2011_1_neg1_is_0_2_pow_1006_l84_84411


namespace sugar_concentration_after_adding_water_l84_84597

def initial_mass_of_sugar_water : ℝ := 90
def initial_sugar_concentration : ℝ := 0.10
def final_sugar_concentration : ℝ := 0.08
def mass_of_water_added : ℝ := 22.5

theorem sugar_concentration_after_adding_water 
  (m_sugar_water : ℝ := initial_mass_of_sugar_water)
  (c_initial : ℝ := initial_sugar_concentration)
  (c_final : ℝ := final_sugar_concentration)
  (m_water_added : ℝ := mass_of_water_added) :
  (m_sugar_water * c_initial = (m_sugar_water + m_water_added) * c_final) := 
sorry

end sugar_concentration_after_adding_water_l84_84597


namespace bao_interest_l84_84918

noncomputable def initial_amount : ℝ := 1000
noncomputable def interest_rate : ℝ := 0.05
noncomputable def periods : ℕ := 6
noncomputable def final_amount : ℝ := initial_amount * (1 + interest_rate) ^ periods
noncomputable def interest_earned : ℝ := final_amount - initial_amount

theorem bao_interest :
  interest_earned = 340.095 := by
  sorry

end bao_interest_l84_84918


namespace roberts_total_sales_l84_84834

theorem roberts_total_sales 
  (basic_salary : ℝ := 1250) 
  (commission_rate : ℝ := 0.10) 
  (savings_rate : ℝ := 0.20) 
  (monthly_expenses : ℝ := 2888) 
  (S : ℝ) : S = 23600 :=
by
  have total_earnings := basic_salary + commission_rate * S
  have used_for_expenses := (1 - savings_rate) * total_earnings
  have expenses_eq : used_for_expenses = monthly_expenses := sorry
  have expense_calc : (1 - savings_rate) * (basic_salary + commission_rate * S) = monthly_expenses := sorry
  have simplify_eq : 0.80 * (1250 + 0.10 * S) = 2888 := sorry
  have open_eq : 1000 + 0.08 * S = 2888 := sorry
  have isolate_S : 0.08 * S = 1888 := sorry
  have solve_S : S = 1888 / 0.08 := sorry
  have final_S : S = 23600 := sorry
  exact final_S

end roberts_total_sales_l84_84834


namespace local_minimum_value_of_f_l84_84238

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem local_minimum_value_of_f : 
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≥ f x) ∧ f x = 1 :=
by
  sorry

end local_minimum_value_of_f_l84_84238


namespace percent_decrease_internet_cost_l84_84940

theorem percent_decrease_internet_cost :
  ∀ (initial_cost final_cost : ℝ), initial_cost = 120 → final_cost = 45 → 
  ((initial_cost - final_cost) / initial_cost) * 100 = 62.5 :=
by
  intros initial_cost final_cost h_initial h_final
  sorry

end percent_decrease_internet_cost_l84_84940


namespace max_leap_years_l84_84977

theorem max_leap_years (years : ℕ) (leap_interval : ℕ) (total_years : ℕ) (leap_years : ℕ)
  (h1 : leap_interval = 5)
  (h2 : total_years = 200)
  (h3 : years = total_years / leap_interval) :
  leap_years = 40 :=
by
  sorry

end max_leap_years_l84_84977


namespace download_time_correct_l84_84812

-- Define the given conditions
def total_size : ℕ := 880
def downloaded : ℕ := 310
def speed : ℕ := 3

-- Calculate the remaining time to download
def time_remaining : ℕ := (total_size - downloaded) / speed

-- Theorem statement that needs to be proven
theorem download_time_correct : time_remaining = 190 := by
  -- Proof goes here
  sorry

end download_time_correct_l84_84812


namespace molecular_weight_one_mole_of_AlPO4_l84_84319

theorem molecular_weight_one_mole_of_AlPO4
  (molecular_weight_4_moles : ℝ)
  (h : molecular_weight_4_moles = 488) :
  molecular_weight_4_moles / 4 = 122 :=
by
  sorry

end molecular_weight_one_mole_of_AlPO4_l84_84319


namespace find_a_l84_84561

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_a (h1 : a ≠ 0) (h2 : f a b c (-1) = 0)
    (h3 : ∀ x : ℝ, x ≤ f a b c x ∧ f a b c x ≤ (1/2) * (x^2 + 1)) :
  a = 1/2 :=
by
  sorry

end find_a_l84_84561


namespace greatest_three_digit_multiple_of_17_l84_84373

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l84_84373


namespace find_n_sequence_sum_l84_84367

theorem find_n_sequence_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₀ : ∀ n, a n = (2^n - 1) / 2^n)
  (h₁ : S 6 = 321 / 64) :
  ∃ n, S n = 321 / 64 ∧ n = 6 := 
by 
  sorry

end find_n_sequence_sum_l84_84367


namespace digit_product_equality_l84_84065

theorem digit_product_equality :
  ∃ (a b c d e f g h i j : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10 ∧
    a * (10 * b + c) * (100 * d + 10 * e + f) = (1000 * g + 100 * h + 10 * i + j) :=
sorry

end digit_product_equality_l84_84065


namespace cubic_as_diff_of_squares_l84_84901

theorem cubic_as_diff_of_squares (n : ℕ) (h : n > 1) :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n^3 = a^2 - b^2 := 
sorry

end cubic_as_diff_of_squares_l84_84901


namespace evaluate_expression_l84_84858

theorem evaluate_expression : 
  (3^4 + 3^4 + 3^4) / (3^(-4) + 3^(-4)) = 9841.5 :=
by
  sorry

end evaluate_expression_l84_84858


namespace fly_distance_from_ceiling_l84_84322

theorem fly_distance_from_ceiling (x y z : ℝ) (hx : x = 2) (hy : y = 6) (hP : x^2 + y^2 + z^2 = 100) : z = 2 * Real.sqrt 15 :=
by
  sorry

end fly_distance_from_ceiling_l84_84322


namespace fish_remaining_l84_84621

def initial_fish : ℝ := 47.0
def given_away_fish : ℝ := 22.5

theorem fish_remaining : initial_fish - given_away_fish = 24.5 :=
by
  sorry

end fish_remaining_l84_84621


namespace square_area_l84_84513

theorem square_area (x : ℝ) (s : ℝ) 
  (h1 : s^2 + s^2 = (2 * x)^2) 
  (h2 : 4 * s = 16 * x) : s^2 = 16 * x^2 :=
by {
  sorry -- Proof not required
}

end square_area_l84_84513


namespace unique_triple_solution_zero_l84_84862

theorem unique_triple_solution_zero (m n k : ℝ) :
  (∃ x : ℝ, m * x ^ 2 + n = 0) ∧
  (∃ x : ℝ, n * x ^ 2 + k = 0) ∧
  (∃ x : ℝ, k * x ^ 2 + m = 0) ↔
  (m = 0 ∧ n = 0 ∧ k = 0) := 
sorry

end unique_triple_solution_zero_l84_84862


namespace find_m_l84_84738

theorem find_m (m : ℕ) :
  (2022 ^ 2 - 4) * (2021 ^ 2 - 4) = 2024 * 2020 * 2019 * m → 
  m = 2023 :=
by
  sorry

end find_m_l84_84738


namespace pine_tree_taller_than_birch_l84_84165

def height_birch : ℚ := 49 / 4
def height_pine : ℚ := 74 / 4

def height_difference : ℚ :=
  height_pine - height_birch

theorem pine_tree_taller_than_birch :
  height_difference = 25 / 4 :=
by
  sorry

end pine_tree_taller_than_birch_l84_84165


namespace local_minimum_point_l84_84570

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem local_minimum_point (a : ℝ) (h : ∃ δ > 0, ∀ x, abs (x - a) < δ → f x ≥ f a) : a = 2 :=
by
  sorry

end local_minimum_point_l84_84570


namespace probability_ephraim_fiona_same_heads_as_keiko_l84_84821

/-- Define a function to calculate the probability that Keiko, Ephraim, and Fiona get the same number of heads. -/
def probability_same_heads : ℚ :=
  let total_outcomes := (2^2) * (2^3) * (2^3)
  let successful_outcomes := 13
  successful_outcomes / total_outcomes

/-- Theorem stating the problem condition and expected probability. -/
theorem probability_ephraim_fiona_same_heads_as_keiko
  (h_keiko : ℕ := 2) -- Keiko tosses two coins
  (h_ephraim : ℕ := 3) -- Ephraim tosses three coins
  (h_fiona : ℕ := 3) -- Fiona tosses three coins
  -- Expected probability that both Ephraim and Fiona get the same number of heads as Keiko
  : probability_same_heads = 13 / 256 :=
sorry

end probability_ephraim_fiona_same_heads_as_keiko_l84_84821


namespace multiply_103_97_l84_84888

theorem multiply_103_97 : 103 * 97 = 9991 := 
by
  sorry

end multiply_103_97_l84_84888


namespace factor_polynomial_l84_84479

theorem factor_polynomial (y : ℝ) :
  y^8 - 4 * y^6 + 6 * y^4 - 4 * y^2 + 1 = ((y - 1) * (y + 1))^4 :=
sorry

end factor_polynomial_l84_84479


namespace rope_for_second_post_l84_84726

theorem rope_for_second_post 
(r1 r2 r3 r4 : ℕ) 
(h_total : r1 + r2 + r3 + r4 = 70)
(h_r1 : r1 = 24)
(h_r3 : r3 = 14)
(h_r4 : r4 = 12) 
: r2 = 20 := 
by 
  sorry

end rope_for_second_post_l84_84726


namespace second_date_sum_eq_80_l84_84543

theorem second_date_sum_eq_80 (a1 a2 a3 a4 a5 : ℕ) (h1 : a1 + a2 + a3 + a4 + a5 = 80)
  (h2 : a2 = a1 + 1) (h3 : a3 = a2 + 1) (h4 : a4 = a3 + 1) (h5 : a5 = a4 + 1): a2 = 15 :=
by
  sorry

end second_date_sum_eq_80_l84_84543


namespace denomination_is_100_l84_84817

-- Define the initial conditions
def num_bills : ℕ := 8
def total_savings : ℕ := 800

-- Define the denomination of the bills
def denomination_bills (num_bills : ℕ) (total_savings : ℕ) : ℕ := 
  total_savings / num_bills

-- The theorem stating the denomination is $100
theorem denomination_is_100 :
  denomination_bills num_bills total_savings = 100 := by
  sorry

end denomination_is_100_l84_84817


namespace johns_meeting_distance_l84_84143

theorem johns_meeting_distance (d t: ℝ) 
    (h1 : d = 40 * (t + 1.5))
    (h2 : d - 40 = 60 * (t - 2)) :
    d = 420 :=
by sorry

end johns_meeting_distance_l84_84143


namespace volume_of_one_wedge_l84_84844

theorem volume_of_one_wedge 
  (circumference : ℝ)
  (h : circumference = 15 * Real.pi) 
  (radius : ℝ) 
  (volume : ℝ) 
  (wedge_volume : ℝ) 
  (h_radius : radius = 7.5)
  (h_volume : volume = (4 / 3) * Real.pi * radius^3)
  (h_wedge_volume : wedge_volume = volume / 5)
  : wedge_volume = 112.5 * Real.pi :=
by
  sorry

end volume_of_one_wedge_l84_84844


namespace bruce_initial_money_l84_84029

-- Definitions of the conditions
def cost_crayons : ℕ := 5 * 5
def cost_books : ℕ := 10 * 5
def cost_calculators : ℕ := 3 * 5
def total_spent : ℕ := cost_crayons + cost_books + cost_calculators
def cost_bags : ℕ := 11 * 10
def initial_money : ℕ := total_spent + cost_bags

-- Theorem statement
theorem bruce_initial_money :
  initial_money = 200 := by
  sorry

end bruce_initial_money_l84_84029


namespace negate_prop_l84_84316

theorem negate_prop :
  ¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
by
  sorry

end negate_prop_l84_84316


namespace problem_1_problem_2_l84_84749

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h₀ : a = 1) (h₁ : ∀ x : ℝ, x^2 - 5 * a * x + 4 * a^2 < 0)
                                    (h₂ : ∀ x : ℝ, (x - 2) * (x - 5) < 0) :
  ∃ x : ℝ, 2 < x ∧ x < 4 :=
by sorry

-- Proof Problem 2
theorem problem_2 (p q : ℝ → Prop) (h₀ : ∀ x : ℝ, p x → q x) 
                                (p_def : ∀ (a : ℝ) (x : ℝ), 0 < a → p x ↔ a < x ∧ x < 4 * a) 
                                (q_def : ∀ x : ℝ, q x ↔ 2 < x ∧ x < 5) :
  ∃ a : ℝ, (5 / 4) ≤ a ∧ a ≤ 2 :=
by sorry

end problem_1_problem_2_l84_84749


namespace pascal_triangle_row_20_sum_l84_84620

theorem pascal_triangle_row_20_sum :
  (Nat.choose 20 2) + (Nat.choose 20 3) + (Nat.choose 20 4) = 6175 :=
by
  sorry

end pascal_triangle_row_20_sum_l84_84620


namespace probability_at_least_3_out_of_6_babies_speak_l84_84211

noncomputable def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * (p^k) * ((1 - p)^(n - k))

noncomputable def prob_at_least_k (total : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  1 - (Finset.range k).sum (λ i => binomial_prob total i p)

theorem probability_at_least_3_out_of_6_babies_speak :
  prob_at_least_k 6 3 (2/5) = 7120/15625 :=
by
  sorry

end probability_at_least_3_out_of_6_babies_speak_l84_84211


namespace tom_balloons_count_l84_84341

-- Define the number of balloons Tom initially has
def balloons_initial : Nat := 30

-- Define the number of balloons Tom gave away
def balloons_given : Nat := 16

-- Define the number of balloons Tom now has
def balloons_remaining : Nat := balloons_initial - balloons_given

theorem tom_balloons_count :
  balloons_remaining = 14 := by
  sorry

end tom_balloons_count_l84_84341


namespace find_angle_F_l84_84714

-- Declaring the necessary angles
variables (E F G H : ℝ) -- Angles are real numbers

-- Declaring the conditions
axiom parallel_lines : E = 3 * H
axiom angle_relation1 : G = 2 * F
axiom supplementary_angles : F + G = 180

-- The theorem statement
theorem find_angle_F (h1 : E = 3 * H) (h2 : G = 2 * F) (h3 : F + G = 180) : F = 60 :=
  sorry

end find_angle_F_l84_84714


namespace tan_315_eq_neg_1_l84_84380

theorem tan_315_eq_neg_1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_1_l84_84380


namespace units_digit_lucas_L10_is_4_l84_84514

def lucas : ℕ → ℕ 
  | 0 => 2
  | 1 => 1
  | n + 2 => lucas (n + 1) + lucas n

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_lucas_L10_is_4 : units_digit (lucas (lucas 10)) = 4 := 
  sorry

end units_digit_lucas_L10_is_4_l84_84514


namespace inequality_wxyz_l84_84798

theorem inequality_wxyz 
  (w x y z : ℝ) 
  (h₁ : w^2 + y^2 ≤ 1) : 
  (w * x + y * z - 1)^2 ≥ (w^2 + y^2 - 1) * (x^2 + z^2 - 1) :=
by
  sorry

end inequality_wxyz_l84_84798


namespace emily_total_beads_l84_84465

-- Let's define the given conditions
def necklaces : ℕ := 11
def beads_per_necklace : ℕ := 28

-- The statement to prove
theorem emily_total_beads : (necklaces * beads_per_necklace) = 308 := by
  sorry

end emily_total_beads_l84_84465


namespace powderman_distance_approximates_275_yards_l84_84524

noncomputable def distance_run (t : ℝ) : ℝ := 6 * t
noncomputable def sound_distance (t : ℝ) : ℝ := 1080 * (t - 45) / 3

theorem powderman_distance_approximates_275_yards : 
  ∃ t : ℝ, t > 45 ∧ 
  (distance_run t = sound_distance t) → 
  abs (distance_run t - 275) < 1 :=
by
  sorry

end powderman_distance_approximates_275_yards_l84_84524


namespace min_speed_to_arrive_before_cara_l84_84047

theorem min_speed_to_arrive_before_cara (d : ℕ) (sc : ℕ) (tc : ℕ) (sd : ℕ) (td : ℕ) (hd : ℕ) :
  d = 180 ∧ sc = 30 ∧ tc = d / sc ∧ hd = 1 ∧ td = tc - hd ∧ sd = d / td ∧ (36 < sd) :=
sorry

end min_speed_to_arrive_before_cara_l84_84047


namespace largest_n_value_l84_84019

theorem largest_n_value (n : ℕ) (h : (1 / 5 : ℝ) + (n / 8 : ℝ) + 1 < 2) : n ≤ 6 :=
by
  sorry

end largest_n_value_l84_84019


namespace remainder_of_sum_div_8_l84_84996

theorem remainder_of_sum_div_8 :
  let a := 2356789
  let b := 211
  (a + b) % 8 = 0 := 
by 
  sorry

end remainder_of_sum_div_8_l84_84996


namespace problem_1_part_1_proof_problem_1_part_2_proof_l84_84533

noncomputable def problem_1_part_1 : Real :=
  2 * Real.sqrt 2 + (Real.sqrt 6) / 2

theorem problem_1_part_1_proof:
  let θ₀ := 3 * Real.pi / 4
  let ρ_A := 4 * Real.cos θ₀
  let ρ_B := Real.sqrt 3 * Real.sin θ₀
  |ρ_A - ρ_B| = 2 * Real.sqrt 2 + (Real.sqrt 6) / 2 :=
  sorry

theorem problem_1_part_2_proof :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x - (Real.sqrt 3)/2 * y = 0) :=
  sorry

end problem_1_part_1_proof_problem_1_part_2_proof_l84_84533


namespace find_b_l84_84176

theorem find_b (g : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, g (-x) = -g x) (h2 : ∃ x, g x ≠ 0) 
               (h3 : a > 0) (h4 : a ≠ 1) (h5 : ∀ x, (1 / (a ^ x - 1) - 1 / b) * g x = (1 / (a ^ (-x) - 1) - 1 / b) * g (-x)) :
    b = -2 :=
sorry

end find_b_l84_84176


namespace min_value_of_a_l84_84568

theorem min_value_of_a (a : ℝ) (h : ∃ x : ℝ, |x - 1| + |x + a| ≤ 8) : -9 ≤ a :=
by
  sorry

end min_value_of_a_l84_84568


namespace domain_of_function_l84_84790

theorem domain_of_function :
  {x : ℝ | x ≥ -1 ∧ x ≠ 1 / 2} =
  {x : ℝ | 2 * x - 1 ≠ 0 ∧ x + 1 ≥ 0} :=
by {
  sorry
}

end domain_of_function_l84_84790


namespace equation_of_line_through_point_l84_84587

theorem equation_of_line_through_point (a T : ℝ) (h : a ≠ 0 ∧ T ≠ 0) :
  ∃ k : ℝ, (k = T / (a^2)) ∧ (k * x + (2 * T / a)) = (k * x + (2 * T / a)) → 
  (T * x - a^2 * y + 2 * T * a = 0) :=
by
  use T / (a^2)
  sorry

end equation_of_line_through_point_l84_84587


namespace quadratic_has_distinct_real_roots_l84_84002

theorem quadratic_has_distinct_real_roots (a : ℝ) (h : a = -2) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * x1 + 3 = 0 ∧ a * x2^2 + 2 * x2 + 3 = 0) :=
by
  sorry

end quadratic_has_distinct_real_roots_l84_84002


namespace smallest_floor_sum_l84_84507

theorem smallest_floor_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (⌊(a + b + d) / c⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + a + d) / b⌋) = 9 :=
sorry

end smallest_floor_sum_l84_84507


namespace red_basket_fruit_count_l84_84755

-- Defining the basket counts
def blue_basket_bananas := 12
def blue_basket_apples := 4
def blue_basket_fruits := blue_basket_bananas + blue_basket_apples
def red_basket_fruits := blue_basket_fruits / 2

-- Statement of the proof problem
theorem red_basket_fruit_count : red_basket_fruits = 8 := by
  sorry

end red_basket_fruit_count_l84_84755


namespace no_solution_abs_eq_l84_84496

theorem no_solution_abs_eq (x : ℝ) (h : x > 0) : |x + 4| = 3 - x → false :=
by
  sorry

end no_solution_abs_eq_l84_84496


namespace width_to_length_ratio_l84_84666

variable (w : ℕ)

def length := 10
def perimeter := 36

theorem width_to_length_ratio
  (h_perimeter : 2 * w + 2 * length = perimeter) :
  w / length = 4 / 5 :=
by
  -- Skipping proof steps, putting sorry
  sorry

end width_to_length_ratio_l84_84666


namespace fraction_of_water_l84_84242

/-- 
  Prove that the fraction of the mixture that is water is (\frac{2}{5}) 
  given the total weight of the mixture is 40 pounds, 
  1/4 of the mixture is sand, 
  and the remaining 14 pounds of the mixture is gravel. 
-/
theorem fraction_of_water 
  (total_weight : ℝ)
  (weight_sand : ℝ)
  (weight_gravel : ℝ)
  (weight_water : ℝ)
  (h1 : total_weight = 40)
  (h2 : weight_sand = (1/4) * total_weight)
  (h3 : weight_gravel = 14)
  (h4 : weight_water = total_weight - (weight_sand + weight_gravel)) :
  (weight_water / total_weight) = 2/5 :=
by
  sorry

end fraction_of_water_l84_84242


namespace color_triangle_vertices_no_same_color_l84_84758

-- Define the colors and the vertices
inductive Color | red | green | blue | yellow
inductive Vertex | A | B | C 

-- Define a function that counts ways to color the triangle given constraints
def count_valid_colorings (colors : List Color) (vertices : List Vertex) : Nat := 
  -- There are 4 choices for the first vertex, 3 for the second, 2 for the third
  4 * 3 * 2

-- The theorem we want to prove
theorem color_triangle_vertices_no_same_color : count_valid_colorings [Color.red, Color.green, Color.blue, Color.yellow] [Vertex.A, Vertex.B, Vertex.C] = 24 := by
  sorry

end color_triangle_vertices_no_same_color_l84_84758


namespace onions_shelf_correct_l84_84956

def onions_on_shelf (initial: ℕ) (sold: ℕ) (added: ℕ) (given_away: ℕ): ℕ :=
  initial - sold + added - given_away

theorem onions_shelf_correct :
  onions_on_shelf 98 65 20 10 = 43 :=
by
  sorry

end onions_shelf_correct_l84_84956


namespace solve_inequality_l84_84688

noncomputable def f (x : ℝ) : ℝ :=
  x^3 + x + 2^x - 2^(-x)

theorem solve_inequality (x : ℝ) : 
  f (Real.exp x - x) ≤ 7/2 ↔ x = 0 := 
sorry

end solve_inequality_l84_84688


namespace find_functions_l84_84180

noncomputable def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (p q : ℝ), p ≠ q → (f q - f p) / (q - p) * 0 + f p - (f q - f p) / (q - p) * p = p * q

theorem find_functions (f : ℝ → ℝ) (c : ℝ) :
  satisfies_condition f → (∀ x : ℝ, f x = x * (c + x)) :=
by
  intros
  sorry

end find_functions_l84_84180


namespace water_saving_percentage_l84_84345

/-- 
Given:
1. The old toilet uses 5 gallons of water per flush.
2. The household flushes 15 times per day.
3. John saved 1800 gallons of water in June.

Prove that the percentage of water saved per flush by the new toilet compared 
to the old one is 80%.
-/
theorem water_saving_percentage 
  (old_toilet_usage_per_flush : ℕ)
  (flushes_per_day : ℕ)
  (savings_in_june : ℕ)
  (days_in_june : ℕ) :
  old_toilet_usage_per_flush = 5 →
  flushes_per_day = 15 →
  savings_in_june = 1800 →
  days_in_june = 30 →
  (old_toilet_usage_per_flush * flushes_per_day * days_in_june - savings_in_june)
  * 100 / (old_toilet_usage_per_flush * flushes_per_day * days_in_june) = 80 :=
by 
  sorry

end water_saving_percentage_l84_84345


namespace range_of_x_l84_84831

theorem range_of_x (m : ℝ) (x : ℝ) (h : 0 < m ∧ m ≤ 5) : 
  (x^2 + (2 * m - 1) * x > 4 * x + 2 * m - 4) ↔ (x < -6 ∨ x > 4) := 
sorry

end range_of_x_l84_84831


namespace largest_n_for_divisibility_l84_84639

theorem largest_n_for_divisibility : 
  ∃ n : ℕ, (n + 12 ∣ n^3 + 150) ∧ (∀ m : ℕ, (m + 12 ∣ m^3 + 150) → m ≤ 246) :=
sorry

end largest_n_for_divisibility_l84_84639


namespace find_values_of_a_and_c_l84_84398

theorem find_values_of_a_and_c
  (a c : ℝ)
  (h1 : ∀ x : ℝ, (1 / 3 < x ∧ x < 1 / 2) ↔ a * x^2 + 5 * x + c > 0) :
  a = -6 ∧ c = -1 :=
by
  sorry

end find_values_of_a_and_c_l84_84398


namespace initial_typists_count_l84_84017

theorem initial_typists_count
  (letters_per_20_min : Nat)
  (letters_total_1_hour : Nat)
  (letters_typists_count : Nat)
  (n_typists_init : Nat)
  (h1 : letters_per_20_min = 46)
  (h2 : letters_typists_count = 30)
  (h3 : letters_total_1_hour = 207) :
  n_typists_init = 20 :=
by {
  sorry
}

end initial_typists_count_l84_84017


namespace percentage_deficit_for_second_side_l84_84394

theorem percentage_deficit_for_second_side
  (L W : ℝ) 
  (measured_first_side : ℝ := 1.12 * L) 
  (error_in_area : ℝ := 1.064) : 
  (∃ x : ℝ, (1.12 * L) * ((1 - 0.01 * x) * W) = 1.064 * (L * W) → x = 5) :=
by
  sorry

end percentage_deficit_for_second_side_l84_84394


namespace four_nat_nums_prime_condition_l84_84763

theorem four_nat_nums_prime_condition (a b c d : ℕ) (h₁ : a = 1) (h₂ : b = 2) (h₃ : c = 3) (h₄ : d = 5) :
  Nat.Prime (a * b + c * d) ∧ Nat.Prime (a * c + b * d) ∧ Nat.Prime (a * d + b * c) :=
by
  sorry

end four_nat_nums_prime_condition_l84_84763


namespace min_value_of_a_plus_2b_l84_84922

theorem min_value_of_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b - 3) :
  a + 2 * b = 4 * Real.sqrt 2 + 3 :=
sorry

end min_value_of_a_plus_2b_l84_84922


namespace vertex_not_neg2_2_l84_84134

theorem vertex_not_neg2_2 (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : a * 1^2 + b * 1 + c = 0)
  (hsymm : ∀ x y, y = a * x^2 + b * x + c → y = a * (4 - x)^2 + b * (4 - x) + c) :
  ¬ ((-b) / (2 * a) = -2 ∧ a * (-2)^2 + b * (-2) + c = 2) :=
by
  sorry

end vertex_not_neg2_2_l84_84134


namespace inequality_proof_l84_84575

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) : 
  x^2 * y^2 + |x^2 - y^2| ≤ π / 2 := 
sorry

end inequality_proof_l84_84575


namespace quadratic_range_l84_84728

open Real

def quadratic (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 5

theorem quadratic_range :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -8 ≤ quadratic x ∧ quadratic x ≤ 19 :=
by
  intro x h
  sorry

end quadratic_range_l84_84728


namespace quadratic_is_perfect_square_l84_84296

theorem quadratic_is_perfect_square (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ d e : ℤ, a*x^2 + b*x + c = (d*x + e)^2) : 
  ∃ d e : ℤ, a = d^2 ∧ b = 2*d*e ∧ c = e^2 :=
sorry

end quadratic_is_perfect_square_l84_84296


namespace unrelated_statement_l84_84733

-- Definitions
def timely_snow_promises_harvest : Prop := true -- assumes it has a related factor
def upper_beam_not_straight_lower_beam_crooked : Prop := true -- assumes it has a related factor
def smoking_harmful_to_health : Prop := true -- assumes it has a related factor
def magpies_signify_joy_crows_signify_mourning : Prop := false -- does not have an inevitable relationship

-- Theorem
theorem unrelated_statement :
  ¬magpies_signify_joy_crows_signify_mourning :=
by 
  -- proof to be provided
  sorry

end unrelated_statement_l84_84733


namespace compound_interest_calculation_l84_84462

theorem compound_interest_calculation : 
  ∀ (x y T SI: ℝ), 
  x = 5000 → T = 2 → SI = 500 → 
  (y = SI * 100 / (x * T)) → 
  (5000 * (1 + (y / 100))^T - 5000 = 512.5) :=
by 
  intros x y T SI hx hT hSI hy
  sorry

end compound_interest_calculation_l84_84462


namespace three_lines_intersect_at_three_points_l84_84739

-- Define the lines as propositions expressing the equations
def line1 (x y : ℝ) := 2 * y - 3 * x = 4
def line2 (x y : ℝ) := x + 3 * y = 3
def line3 (x y : ℝ) := 3 * x - 4.5 * y = 7.5

-- Define a proposition stating that there are exactly 3 unique points of intersection among the three lines
def number_of_intersections : ℕ := 3

-- Prove that the number of unique intersection points is exactly 3 given the lines
theorem three_lines_intersect_at_three_points : 
  ∃! p1 p2 p3 : ℝ × ℝ, 
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
    (line3 p3.1 p3.2 ∧ line1 p3.1 p3.2) :=
sorry

end three_lines_intersect_at_three_points_l84_84739


namespace value_of_x_for_real_y_l84_84129

theorem value_of_x_for_real_y (x y : ℝ) (h : 4 * y^2 - 2 * x * y + 2 * x + 9 = 0) : x ≤ -3 ∨ x ≥ 12 :=
sorry

end value_of_x_for_real_y_l84_84129


namespace work_completion_days_l84_84584

theorem work_completion_days (D : ℕ) 
  (h : 40 * D = 48 * (D - 10)) : D = 60 := 
sorry

end work_completion_days_l84_84584


namespace avg_visitors_per_day_correct_l84_84750

-- Define the given conditions
def avg_sundays : Nat := 540
def avg_other_days : Nat := 240
def num_days : Nat := 30
def sundays_in_month : Nat := 5
def other_days_in_month : Nat := 25

-- Define the total visitors calculation
def total_visitors := (sundays_in_month * avg_sundays) + (other_days_in_month * avg_other_days)

-- Define the average visitors per day calculation
def avg_visitors_per_day := total_visitors / num_days

-- State the proof problem
theorem avg_visitors_per_day_correct : avg_visitors_per_day = 290 :=
by
  sorry

end avg_visitors_per_day_correct_l84_84750


namespace water_added_l84_84397

theorem water_added (x : ℝ) (salt_percent_initial : ℝ) (evaporation_fraction : ℝ) 
(salt_added : ℝ) (resulting_salt_percent : ℝ) 
(hx : x = 119.99999999999996) (h_initial_salt : salt_percent_initial = 0.20) 
(h_evap_fraction : evaporation_fraction = 1/4) (h_salt_added : salt_added = 16)
(h_resulting_salt_percent : resulting_salt_percent = 1/3) : 
∃ (water_added : ℝ), water_added = 30 :=
by
  sorry

end water_added_l84_84397


namespace difference_of_digits_l84_84500

theorem difference_of_digits (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h_diff : (10 * x + y) - (10 * y + x) = 54) : x - y = 6 :=
sorry

end difference_of_digits_l84_84500


namespace minimum_value_of_GP_l84_84761

theorem minimum_value_of_GP (a : ℕ → ℝ) (h : ∀ n, 0 < a n) (h_prod : a 2 * a 10 = 9) :
  a 5 + a 7 = 6 :=
by
  -- proof steps will be filled in here
  sorry

end minimum_value_of_GP_l84_84761


namespace aaron_ends_up_with_24_cards_l84_84569

def initial_cards_aaron : Nat := 5
def found_cards_aaron : Nat := 62
def lost_cards_aaron : Nat := 15
def given_cards_to_arthur : Nat := 28

def final_cards_aaron (initial: Nat) (found: Nat) (lost: Nat) (given: Nat) : Nat :=
  initial + found - lost - given

theorem aaron_ends_up_with_24_cards :
  final_cards_aaron initial_cards_aaron found_cards_aaron lost_cards_aaron given_cards_to_arthur = 24 := by
  sorry

end aaron_ends_up_with_24_cards_l84_84569


namespace solve_for_x_l84_84239

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2 * x - 25) : x = -20 :=
by
  sorry

end solve_for_x_l84_84239


namespace correct_answer_l84_84298

theorem correct_answer (x : ℝ) (h : 3 * x - 10 = 50) : 3 * x + 10 = 70 :=
sorry

end correct_answer_l84_84298


namespace scientific_notation_correct_l84_84506

def n : ℝ := 12910000

theorem scientific_notation_correct : n = 1.291 * 10^7 := 
by
  sorry

end scientific_notation_correct_l84_84506


namespace monotonically_increasing_intervals_exists_a_decreasing_l84_84719

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x - a * x - 1

theorem monotonically_increasing_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, 0 ≤ Real.exp x - a) ∧
  (a > 0 → ∀ x : ℝ, x ≥ Real.log a → 0 ≤ Real.exp x - a) :=
by sorry

theorem exists_a_decreasing (a : ℝ) :
  (a ≥ Real.exp 3) ↔ ∀ x : ℝ, -2 < x ∧ x < 3 → Real.exp x - a ≤ 0 :=
by sorry

end monotonically_increasing_intervals_exists_a_decreasing_l84_84719


namespace number_of_shelves_l84_84355

-- Define the initial conditions and required values
def initial_bears : ℕ := 6
def shipment_bears : ℕ := 18
def bears_per_shelf : ℕ := 6

-- Define the result we want to prove
theorem number_of_shelves : (initial_bears + shipment_bears) / bears_per_shelf = 4 :=
by
    -- Proof steps go here
    sorry

end number_of_shelves_l84_84355


namespace nina_math_homework_l84_84023

theorem nina_math_homework (x : ℕ) :
  let ruby_math := 6
  let ruby_read := 2
  let nina_math := x * ruby_math
  let nina_read := 8 * ruby_read
  let nina_total := nina_math + nina_read
  nina_total = 48 → x = 5 :=
by
  intros
  sorry

end nina_math_homework_l84_84023


namespace hex_351_is_849_l84_84087

noncomputable def hex_to_decimal : ℕ := 1 * 16^0 + 5 * 16^1 + 3 * 16^2

-- The following statement is the core of the proof problem
theorem hex_351_is_849 : hex_to_decimal = 849 := by
  -- Here the proof steps would normally go
  sorry

end hex_351_is_849_l84_84087


namespace part1_part2_l84_84194

theorem part1 (x y : ℝ) (h1 : x + 3 * y = 26) (h2 : 2 * x + y = 22) : x = 8 ∧ y = 6 :=
by
  sorry

theorem part2 (m : ℝ) (h : 8 * m + 6 * (15 - m) ≤ 100) : m ≤ 5 :=
by
  sorry

end part1_part2_l84_84194


namespace lollipops_per_day_l84_84951

variable (Alison_lollipops : ℕ) (Henry_lollipops : ℕ) (Diane_lollipops : ℕ) (Total_lollipops : ℕ) (Days : ℕ)

-- Conditions given in the problem
axiom condition1 : Alison_lollipops = 60
axiom condition2 : Henry_lollipops = Alison_lollipops + 30
axiom condition3 : Alison_lollipops = Diane_lollipops / 2
axiom condition4 : Total_lollipops = Alison_lollipops + Henry_lollipops + Diane_lollipops
axiom condition5 : Days = 6

-- Question to prove
theorem lollipops_per_day : (Total_lollipops / Days) = 45 := sorry

end lollipops_per_day_l84_84951


namespace average_time_per_mile_l84_84534

-- Define the conditions
def total_distance_miles : ℕ := 24
def total_time_hours : ℕ := 3
def total_time_minutes : ℕ := 36
def total_time_in_minutes : ℕ := (total_time_hours * 60) + total_time_minutes

-- State the theorem
theorem average_time_per_mile : total_time_in_minutes / total_distance_miles = 9 :=
by
  sorry

end average_time_per_mile_l84_84534


namespace quadratic_inequality_solution_l84_84038

theorem quadratic_inequality_solution (x : ℝ) :
  (-3 * x^2 + 8 * x + 3 > 0) ↔ (x < -1/3 ∨ x > 3) :=
by
  sorry

end quadratic_inequality_solution_l84_84038


namespace arithmetic_sequence_ninth_term_l84_84781

theorem arithmetic_sequence_ninth_term :
  ∃ a d : ℤ, (a + 2 * d = 23) ∧ (a + 5 * d = 29) ∧ (a + 8 * d = 35) :=
by
  sorry

end arithmetic_sequence_ninth_term_l84_84781


namespace points_five_from_origin_l84_84648

theorem points_five_from_origin (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end points_five_from_origin_l84_84648


namespace shaded_area_correct_l84_84055

-- Define points as vectors in the 2D plane.
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define points K, L, M, J based on the given coordinates.
def K : Point := {x := 0, y := 0}
def L : Point := {x := 5, y := 0}
def M : Point := {x := 5, y := 6}
def J : Point := {x := 0, y := 6}

-- Define intersection point N based on the equations of lines.
def N : Point := {x := 2.5, y := 3}

-- Define the function to calculate area of a trapezoid.
def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  0.5 * (b1 + b2) * h

-- Define the function to calculate area of a triangle.
def triangle_area (b h : ℝ) : ℝ :=
  0.5 * b * h

-- Compute total shaded area according to the problem statement.
def shaded_area (K L M J N : Point) : ℝ :=
  trapezoid_area 5 2.5 3 + triangle_area 2.5 1

theorem shaded_area_correct : shaded_area K L M J N = 12.5 := by
  sorry

end shaded_area_correct_l84_84055


namespace min_value_abs_ab_l84_84451

theorem min_value_abs_ab (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) 
(h_perpendicular : - 1 / (a^2) * (a^2 + 1) / b = -1) :
|a * b| = 2 :=
sorry

end min_value_abs_ab_l84_84451


namespace compare_log_values_l84_84172

noncomputable def a : ℝ := (Real.log 2) / 2
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := (Real.log 5) / 5

theorem compare_log_values : c < a ∧ a < b := by
  -- Proof is omitted
  sorry

end compare_log_values_l84_84172


namespace largest_prime_factor_of_4620_l84_84838

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / m → ¬ (m ∣ n)

def prime_factors (n : ℕ) : List ℕ :=
  -- assumes a well-defined function that generates the prime factor list
  -- this is a placeholder function for demonstrating purposes
  sorry

def largest_prime_factor (l : List ℕ) : ℕ :=
  l.foldr max 0

theorem largest_prime_factor_of_4620 : largest_prime_factor (prime_factors 4620) = 11 :=
by
  sorry

end largest_prime_factor_of_4620_l84_84838


namespace area_difference_l84_84735

theorem area_difference (d : ℝ) (r : ℝ) (ratio : ℝ) (h1 : d = 10) (h2 : ratio = 2) (h3 : r = 5) :
  (π * r^2 - ((d^2 / (ratio^2 + 1)).sqrt * (2 * d^2 / (ratio^2 + 1)).sqrt)) = 38.5 :=
by
  sorry

end area_difference_l84_84735


namespace sum_of_roots_gt_two_l84_84501

noncomputable def f : ℝ → ℝ := λ x => Real.log x - x + 1

theorem sum_of_roots_gt_two (m : ℝ) (x1 x2 : ℝ) (hx1 : f x1 = m) (hx2 : f x2 = m) (hne : x1 ≠ x2) : x1 + x2 > 2 := by
  sorry

end sum_of_roots_gt_two_l84_84501


namespace necessary_but_not_sufficient_l84_84164

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x, x ≥ a → x^2 - x - 2 ≥ 0) ∧ (∃ x, x ≥ a ∧ ¬(x^2 - x - 2 ≥ 0)) ↔ a ≥ 2 := 
sorry

end necessary_but_not_sufficient_l84_84164


namespace students_without_pens_l84_84352

theorem students_without_pens (total_students blue_pens red_pens both_pens : ℕ)
  (h_total : total_students = 40)
  (h_blue : blue_pens = 18)
  (h_red : red_pens = 26)
  (h_both : both_pens = 10) :
  total_students - (blue_pens + red_pens - both_pens) = 6 :=
by
  sorry

end students_without_pens_l84_84352


namespace max_value_f_l84_84037

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem max_value_f : ∃ x ∈ (Set.Icc 0 (Real.pi / 2)), f x = Real.pi / 12 + Real.sqrt 3 / 2 :=
by
  sorry

end max_value_f_l84_84037


namespace carol_rectangle_width_l84_84410

def carol_width (lengthC : ℕ) (widthJ : ℕ) (lengthJ : ℕ) (widthC : ℕ) :=
  lengthC * widthC = lengthJ * widthJ

theorem carol_rectangle_width 
  {lengthC widthJ lengthJ : ℕ} (h1 : lengthC = 8)
  (h2 : widthJ = 30) (h3 : lengthJ = 4)
  (h4 : carol_width lengthC widthJ lengthJ 15) : 
  widthC = 15 :=
by 
  subst h1
  subst h2
  subst h3
  sorry -- proof not required

end carol_rectangle_width_l84_84410


namespace combine_ingredients_l84_84994

theorem combine_ingredients : 
  ∃ (water flour salt : ℕ), 
    water = 10 ∧ flour = 16 ∧ salt = 1 / 2 * flour ∧ 
    (water + flour = 26) ∧ (salt = 8) :=
by
  sorry

end combine_ingredients_l84_84994


namespace find_m_l84_84466

theorem find_m (x1 x2 m : ℝ) (h1 : 2 * x1^2 - 3 * x1 + m = 0) (h2 : 2 * x2^2 - 3 * x2 + m = 0) (h3 : 8 * x1 - 2 * x2 = 7) :
  m = 1 :=
sorry

end find_m_l84_84466


namespace ball_distance_traveled_l84_84820

noncomputable def total_distance (a1 d n : ℕ) : ℕ :=
  n * (a1 + a1 + (n-1) * d) / 2

theorem ball_distance_traveled : 
  total_distance 8 5 20 = 1110 :=
by
  sorry

end ball_distance_traveled_l84_84820


namespace janet_acres_l84_84635

-- Defining the variables and conditions
variable (horse_production : ℕ) (acre_requirement : ℕ) (spread_rate : ℕ) (num_horses : ℕ) (days : ℕ)

-- Assigning the given values to the variables
def horseFertilizer := 5
def acreFertilizer := 400
def janetSpreadRate := 4
def janetHorses := 80
def fertilizingDays := 25

-- Main theorem stating the question and proving the answer
theorem janet_acres : 
  ∀ (horse_production : ℕ) (acre_requirement : ℕ) (spread_rate : ℕ) (num_horses : ℕ) (days : ℕ),
  horse_production = 5 → 
  acre_requirement = 400 →
  spread_rate = 4 →
  num_horses = 80 →
  days = 25 →
  (spread_rate * days = 100) := 
by
  intros
  -- Proof would be inserted here
  sorry

end janet_acres_l84_84635


namespace num_rows_seat_9_people_l84_84705

-- Define the premises of the problem.
def seating_arrangement (x y : ℕ) : Prop := (9 * x + 7 * y = 58)

-- The theorem stating the number of rows seating exactly 9 people.
theorem num_rows_seat_9_people
  (x y : ℕ)
  (h : seating_arrangement x y) :
  x = 1 :=
by
  -- Proof is not required as per the instruction
  sorry

end num_rows_seat_9_people_l84_84705


namespace georgie_entry_exit_ways_l84_84757

-- Defining the conditions
def castle_windows : Nat := 8
def non_exitable_windows : Nat := 2

-- Defining the problem
theorem georgie_entry_exit_ways (total_windows : Nat) (blocked_exits : Nat) (entry_windows : Nat) : 
  total_windows = castle_windows → blocked_exits = non_exitable_windows → 
  entry_windows = castle_windows →
  (entry_windows * (total_windows - 1 - blocked_exits) = 40) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end georgie_entry_exit_ways_l84_84757


namespace graphs_relative_position_and_intersection_l84_84431

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x + 5

theorem graphs_relative_position_and_intersection :
  (1 > -1.5) ∧ ( ∃ y, f 0 = y ∧ g 0 = y ) ∧ f 0 = 5 :=
by
  -- sorry to skip the proof
  sorry

end graphs_relative_position_and_intersection_l84_84431


namespace find_certain_number_l84_84225

theorem find_certain_number (x : ℝ) (h : x + 12.952 - 47.95000000000027 = 3854.002) : x = 3889.000 :=
sorry

end find_certain_number_l84_84225


namespace range_of_f_l84_84909

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+1) + 3

theorem range_of_f : Set.range f = Set.Ici 2 := 
by 
  sorry

end range_of_f_l84_84909


namespace find_m_l84_84839

theorem find_m (x1 x2 m : ℝ) (h1 : x1 + x2 = -3) (h2 : x1 * x2 = m) (h3 : 1 / x1 + 1 / x2 = 1) : m = -3 :=
by
  sorry

end find_m_l84_84839


namespace find_sum_l84_84658

theorem find_sum (P : ℕ) (h_total : P * (4/100 + 6/100 + 8/100) = 2700) : P = 15000 :=
by
  sorry

end find_sum_l84_84658


namespace adults_riding_bicycles_l84_84125

theorem adults_riding_bicycles (A : ℕ) (H1 : 15 * 3 + 2 * A = 57) : A = 6 :=
by
  sorry

end adults_riding_bicycles_l84_84125


namespace square_distance_between_intersections_l84_84123

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 7)^2 + y^2 = 4

-- Problem: Prove the square of the distance between intersection points P and Q
theorem square_distance_between_intersections :
  (∃ (x y1 y2 : ℝ), circle1 x y1 ∧ circle2 x y1 ∧ circle1 x y2 ∧ circle2 x y2 ∧ y1 ≠ y2) →
  ∃ d : ℝ, d^2 = 15.3664 :=
by
  sorry

end square_distance_between_intersections_l84_84123


namespace norma_bananas_count_l84_84944

-- Definitions for the conditions
def initial_bananas : ℕ := 47
def lost_bananas : ℕ := 45

-- The proof problem in Lean 4 statement
theorem norma_bananas_count : initial_bananas - lost_bananas = 2 := by
  -- Proof is omitted
  sorry

end norma_bananas_count_l84_84944


namespace traveling_cost_l84_84772

def area_road_length_parallel (length width : ℕ) := width * length

def area_road_breadth_parallel (length width : ℕ) := width * length

def area_intersection (width : ℕ) := width * width

def total_area_of_roads  (length breadth width : ℕ) : ℕ :=
  (area_road_length_parallel length width) + (area_road_breadth_parallel breadth width) - area_intersection width

def cost_of_traveling_roads (total_area_of_roads cost_per_sq_m : ℕ) := total_area_of_roads * cost_per_sq_m

theorem traveling_cost
  (length breadth width cost_per_sq_m : ℕ)
  (h_length : length = 80)
  (h_breadth : breadth = 50)
  (h_width : width = 10)
  (h_cost_per_sq_m : cost_per_sq_m = 3)
  : cost_of_traveling_roads (total_area_of_roads length breadth width) cost_per_sq_m = 3600 :=
by
  sorry

end traveling_cost_l84_84772


namespace area_of_triangle_formed_by_line_and_axes_l84_84752

-- Definition of the line equation condition
def line_eq (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

-- Statement of the problem to prove
theorem area_of_triangle_formed_by_line_and_axes :
  (∃ x y : ℝ, line_eq x y ∧ x = 0 ∧ y = -2) ∧
  (∃ x y : ℝ, line_eq x y ∧ x = 5 ∧ y = 0) →
  let base : ℝ := 5
  let height : ℝ := 2
  let area := (1 / 2) * base * height
  area = 5 := 
by
  sorry

end area_of_triangle_formed_by_line_and_axes_l84_84752


namespace alyssa_puppies_l84_84422

theorem alyssa_puppies (total_puppies : ℕ) (given_away : ℕ) (remaining_puppies : ℕ) 
  (h1 : total_puppies = 7) (h2 : given_away = 5) 
  : remaining_puppies = total_puppies - given_away → remaining_puppies = 2 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end alyssa_puppies_l84_84422


namespace total_hunts_is_21_l84_84696

-- Define the initial conditions
def Sam_hunts : Nat := 6
def Rob_hunts : Nat := Sam_hunts / 2
def Rob_Sam_total_hunt : Nat := Sam_hunts + Rob_hunts
def Mark_hunts : Nat := Rob_Sam_total_hunt / 3
def Peter_hunts : Nat := Mark_hunts * 3

-- The main theorem to prove
theorem total_hunts_is_21 : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 :=
by
  sorry

end total_hunts_is_21_l84_84696


namespace train_length_l84_84332

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_ms : ℝ) (distance_m : ℝ)
  (h1 : speed_km_hr = 90)
  (h2 : time_seconds = 9)
  (h3 : speed_ms = speed_km_hr * (1000 / 3600))
  (h4 : distance_m = speed_ms * time_seconds) :
  distance_m = 225 :=
by
  sorry

end train_length_l84_84332


namespace always_non_monotonic_l84_84860

noncomputable def f (a t x : ℝ) : ℝ :=
if x ≤ t then (2*a - 1)*x + 3*a - 4 else x^3 - x

theorem always_non_monotonic (a : ℝ) (t : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → f a t x1 ≤ f a t x2 ∨ f a t x1 ≥ f a t x2) → a ≤ 1 / 2 :=
sorry

end always_non_monotonic_l84_84860


namespace find_a_for_perpendicular_lines_l84_84762

theorem find_a_for_perpendicular_lines (a : ℝ) 
    (h_perpendicular : 2 * a + (-1) * (3 - a) = 0) :
    a = 1 :=
by
  sorry

end find_a_for_perpendicular_lines_l84_84762


namespace length_of_each_piece_is_correct_l84_84436

noncomputable def rod_length : ℝ := 38.25
noncomputable def num_pieces : ℕ := 45
noncomputable def length_each_piece_cm : ℝ := 85

theorem length_of_each_piece_is_correct : (rod_length / num_pieces) * 100 = length_each_piece_cm :=
by
  sorry

end length_of_each_piece_is_correct_l84_84436


namespace rhombus_area_l84_84121

theorem rhombus_area
  (d1 d2 : ℝ)
  (hd1 : d1 = 14)
  (hd2 : d2 = 20) :
  (d1 * d2) / 2 = 140 := by
  -- Problem: Given diagonals of length 14 cm and 20 cm,
  -- prove that the area of the rhombus is 140 square centimeters.
  sorry

end rhombus_area_l84_84121


namespace quadratic_inequality_solution_l84_84478

variable (x : ℝ)

theorem quadratic_inequality_solution (hx : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_solution_l84_84478


namespace recurrence_relation_l84_84338

-- Define the function p_nk and prove the recurrence relation
def p (n k : ℕ) : ℝ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) : 
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end recurrence_relation_l84_84338


namespace frankie_pets_l84_84346

variable {C S P D : ℕ}

theorem frankie_pets (h1 : S = C + 6) (h2 : P = C - 1) (h3 : C + D = 6) (h4 : C + S + P + D = 19) : 
  C + S + P + D = 19 :=
  by sorry

end frankie_pets_l84_84346


namespace evaluate_expression_l84_84491

theorem evaluate_expression (x : ℝ) (h : 3 * x^3 - x = 1) : 9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2001 := 
by
  sorry

end evaluate_expression_l84_84491


namespace decimal_to_fraction_l84_84069

theorem decimal_to_fraction (h : 0.36 = 36 / 100): (36 / 100 = 9 / 25) := by
    sorry

end decimal_to_fraction_l84_84069


namespace increase_in_average_age_l84_84776

variable (A : ℝ)
variable (A_increase : ℝ)
variable (orig_age_sum : ℝ)
variable (new_age_sum : ℝ)

def original_total_age (A : ℝ) := 8 * A
def new_total_age (A : ℝ) := original_total_age A - 20 - 22 + 29 + 29

theorem increase_in_average_age (A : ℝ) (orig_age_sum := original_total_age A) (new_age_sum := new_total_age A) : 
  (new_age_sum / 8) = (A + 2) := 
by
  unfold new_total_age
  unfold original_total_age
  sorry

end increase_in_average_age_l84_84776


namespace cheryl_found_more_eggs_l84_84058

theorem cheryl_found_more_eggs :
  let kevin_eggs := 5
  let bonnie_eggs := 13
  let george_eggs := 9
  let cheryl_eggs := 56
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
  sorry

end cheryl_found_more_eggs_l84_84058


namespace complex_number_on_ray_is_specific_l84_84358

open Complex

theorem complex_number_on_ray_is_specific (a b : ℝ) (z : ℂ) (h₁ : z = a + b * I) 
  (h₂ : a = b) (h₃ : abs z = 1) : 
  z = (Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) * I :=
by
  sorry

end complex_number_on_ray_is_specific_l84_84358


namespace length_is_62_l84_84344

noncomputable def length_of_plot (b : ℝ) := b + 24

theorem length_is_62 (b : ℝ) (h1 : length_of_plot b = b + 24) 
  (h2 : 2 * (length_of_plot b + b) = 200) : 
  length_of_plot b = 62 :=
by sorry

end length_is_62_l84_84344


namespace textbook_order_total_cost_l84_84734

theorem textbook_order_total_cost :
  let english_quantity := 35
  let geography_quantity := 35
  let mathematics_quantity := 20
  let science_quantity := 30
  let english_price := 7.50
  let geography_price := 10.50
  let mathematics_price := 12.00
  let science_price := 9.50
  (english_quantity * english_price + geography_quantity * geography_price + mathematics_quantity * mathematics_price + science_quantity * science_price = 1155.00) :=
by sorry

end textbook_order_total_cost_l84_84734


namespace tangent_line_at_point_e_tangent_line_from_origin_l84_84185

-- Problem 1
theorem tangent_line_at_point_e (x y : ℝ) (h : y = Real.exp x) (h_e : x = Real.exp 1) :
    (Real.exp x) * x - y - Real.exp (x + 1) = 0 :=
sorry

-- Problem 2
theorem tangent_line_from_origin (x y : ℝ) (h : y = Real.exp x) :
    x = 1 →  Real.exp x * x - y = 0 :=
sorry

end tangent_line_at_point_e_tangent_line_from_origin_l84_84185


namespace problem_solution_l84_84991

noncomputable def p (x : ℝ) : ℝ := 
  (x - (Real.sin 1)^2) * (x - (Real.sin 3)^2) * (x - (Real.sin 9)^2)

theorem problem_solution : ∃ a b n : ℕ, 
  p (1 / 4) = Real.sin (a * Real.pi / 180) / (n * Real.sin (b * Real.pi / 180)) ∧
  a > 0 ∧ b > 0 ∧ a ≤ 90 ∧ b ≤ 90 ∧ a + b + n = 216 :=
sorry

end problem_solution_l84_84991


namespace current_speed_correct_l84_84328

noncomputable def boat_upstream_speed : ℝ := (1 / 20) * 60
noncomputable def boat_downstream_speed : ℝ := (1 / 9) * 60
noncomputable def speed_of_current : ℝ := (boat_downstream_speed - boat_upstream_speed) / 2

theorem current_speed_correct :
  speed_of_current = 1.835 :=
by
  sorry

end current_speed_correct_l84_84328


namespace abs_ineq_one_abs_ineq_two_l84_84111

-- First proof problem: |x-1| + |x+3| < 6 implies -4 < x < 2
theorem abs_ineq_one (x : ℝ) : |x - 1| + |x + 3| < 6 → -4 < x ∧ x < 2 :=
by
  sorry

-- Second proof problem: 1 < |3x-2| < 4 implies -2/3 ≤ x < 1/3 or 1 < x ≤ 2
theorem abs_ineq_two (x : ℝ) : 1 < |3 * x - 2| ∧ |3 * x - 2| < 4 → (-2/3) ≤ x ∧ x < (1/3) ∨ 1 < x ∧ x ≤ 2 :=
by
  sorry

end abs_ineq_one_abs_ineq_two_l84_84111


namespace move_digit_to_make_equation_correct_l84_84269

theorem move_digit_to_make_equation_correct :
  101 - 102 ≠ 1 → (101 - 10^2 = 1) :=
by
  sorry

end move_digit_to_make_equation_correct_l84_84269


namespace sin_cos_sum_2018_l84_84784

theorem sin_cos_sum_2018 {x : ℝ} (h : Real.sin x + Real.cos x = 1) :
  (Real.sin x)^2018 + (Real.cos x)^2018 = 1 :=
by
  sorry

end sin_cos_sum_2018_l84_84784


namespace pair_not_product_48_l84_84619

theorem pair_not_product_48:
  (∀(a b : ℤ), (a, b) = (-6, -8)                    → a * b = 48) ∧
  (∀(a b : ℤ), (a, b) = (-4, -12)                   → a * b = 48) ∧
  (∀(a b : ℚ), (a, b) = (3/4, -64)                  → a * b ≠ 48) ∧
  (∀(a b : ℤ), (a, b) = (3, 16)                     → a * b = 48) ∧
  (∀(a b : ℚ), (a, b) = (4/3, 36)                   → a * b = 48)
  :=
by
  sorry

end pair_not_product_48_l84_84619


namespace value_of_a_l84_84610

theorem value_of_a (a : ℝ) (h : (a - 3) * x ^ |a - 2| + 4 = 0) : a = 1 :=
by
  sorry

end value_of_a_l84_84610


namespace profit_percentage_is_correct_l84_84900

-- Definitions for the given conditions
def SP : ℝ := 850
def Profit : ℝ := 255
def CP : ℝ := SP - Profit

-- The target proof statement
theorem profit_percentage_is_correct : 
  (Profit / CP) * 100 = 42.86 := by
  sorry

end profit_percentage_is_correct_l84_84900


namespace actual_distance_between_towns_l84_84349

def map_distance := 20 -- distance between towns on the map in inches
def scale := 10 -- scale: 1 inch = 10 miles

theorem actual_distance_between_towns : map_distance * scale = 200 := by
  sorry

end actual_distance_between_towns_l84_84349


namespace power_function_through_point_l84_84131

noncomputable def f : ℝ → ℝ := sorry

theorem power_function_through_point (h : ∀ x, ∃ a : ℝ, f x = x^a) (h1 : f 3 = 27) :
  f x = x^3 :=
sorry

end power_function_through_point_l84_84131


namespace squared_difference_l84_84878

theorem squared_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 10) : (x - y)^2 = 24 :=
by
  sorry

end squared_difference_l84_84878


namespace solve_for_x_l84_84589

theorem solve_for_x (x : ℝ) (h_pos : 0 < x) (h_eq : x^4 = 6561) : x = 9 :=
sorry

end solve_for_x_l84_84589


namespace xyz_inequality_l84_84819

theorem xyz_inequality (x y z : ℝ) (hx : 0 ≤ x) (hx' : x ≤ 1) (hy : 0 ≤ y) (hy' : y ≤ 1) (hz : 0 ≤ z) (hz' : z ≤ 1) :
  (x^2 / (1 + x + x*y*z) + y^2 / (1 + y + x*y*z) + z^2 / (1 + z + x*y*z) ≤ 1) :=
sorry

end xyz_inequality_l84_84819


namespace range_of_sum_l84_84314

theorem range_of_sum (a b : ℝ) (h : a^2 - a * b + b^2 = a + b) :
  0 ≤ a + b ∧ a + b ≤ 4 :=
by
  sorry

end range_of_sum_l84_84314


namespace correctly_transformed_equation_l84_84434

theorem correctly_transformed_equation (s a b x y : ℝ) :
  (s = a * b → a = s / b ∧ b ≠ 0) ∧
  (1/2 * x = 8 → x = 16) ∧
  (-x - 1 = y - 1 → x = -y) ∧
  (a = b → a + 3 = b + 3) :=
by
  sorry

end correctly_transformed_equation_l84_84434


namespace expenditure_increase_l84_84273

theorem expenditure_increase
  (current_expenditure : ℝ)
  (future_expenditure : ℝ)
  (years : ℕ)
  (r : ℝ)
  (h₁ : current_expenditure = 1000)
  (h₂ : future_expenditure = 2197)
  (h₃ : years = 3)
  (h₄ : future_expenditure = current_expenditure * (1 + r / 100) ^ years) :
  r = 30 :=
sorry

end expenditure_increase_l84_84273


namespace ab_leq_one_l84_84265

theorem ab_leq_one (a b x : ℝ) (h1 : (x + a) * (x + b) = 9) (h2 : x = a + b) : a * b ≤ 1 := 
sorry

end ab_leq_one_l84_84265


namespace number_of_chairs_in_first_row_l84_84189

-- Define the number of chairs in each row
def chairs_in_second_row := 23
def chairs_in_third_row := 32
def chairs_in_fourth_row := 41
def chairs_in_fifth_row := 50
def chairs_in_sixth_row := 59

-- Define the pattern increment
def increment := 9

-- Define a function to calculate the number of chairs in a given row, given the increment pattern
def chairs_in_row (n : Nat) : Nat :=
if n = 1 then (chairs_in_second_row - increment)
else if n = 2 then chairs_in_second_row
else if n = 3 then chairs_in_third_row
else if n = 4 then chairs_in_fourth_row
else if n = 5 then chairs_in_fifth_row
else if n = 6 then chairs_in_sixth_row
else chairs_in_second_row + (n - 2) * increment

-- The theorem to prove: The number of chairs in the first row is 14
theorem number_of_chairs_in_first_row : chairs_in_row 1 = 14 :=
  by sorry

end number_of_chairs_in_first_row_l84_84189


namespace max_blocks_fit_l84_84622

-- Define the dimensions of the block and the box
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

-- Define the volumes calculation
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

-- Define the dimensions of the block and the box
def block : Dimensions := { length := 3, width := 1, height := 2 }
def box : Dimensions := { length := 4, width := 3, height := 6 }

-- Prove that the maximum number of blocks that can fit in the box is 12
theorem max_blocks_fit : (volume box) / (volume block) = 12 := by sorry

end max_blocks_fit_l84_84622


namespace arithmetic_sequence_ratio_l84_84970

theorem arithmetic_sequence_ratio (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = (1/2) * n * (2 * a 1 + (n-1) * d))
  (h2 : ∀ n, T n = (1/2) * n * (2 * b 1 + (n-1) * d'))
  (h3 : ∀ n, S n / T n = 7*n / (n + 3)): a 5 / b 5 = 21 / 4 := 
by {
  sorry
}

end arithmetic_sequence_ratio_l84_84970


namespace sufficient_condition_for_ellipse_l84_84722

theorem sufficient_condition_for_ellipse (m : ℝ) (h : m^2 > 5) : m^2 > 4 := by
  sorry

end sufficient_condition_for_ellipse_l84_84722


namespace Amanda_needs_12_more_marbles_l84_84395

theorem Amanda_needs_12_more_marbles (K A M : ℕ)
  (h1 : M = 5 * K)
  (h2 : M = 85)
  (h3 : M = A + 63) :
  A + 12 = 2 * K := 
sorry

end Amanda_needs_12_more_marbles_l84_84395


namespace smallest_obtuse_triangles_l84_84386

def obtuseTrianglesInTriangulation (n : Nat) : Nat :=
  if n < 3 then 0 else (n - 2) - 2

theorem smallest_obtuse_triangles (n : Nat) (h : n = 2003) :
  obtuseTrianglesInTriangulation n = 1999 := by
  sorry

end smallest_obtuse_triangles_l84_84386


namespace polygon_angle_arithmetic_progression_l84_84307

theorem polygon_angle_arithmetic_progression
  (h1 : ∀ {n : ℕ}, n ≥ 3)   -- The polygon is convex and n-sided
  (h2 : ∀ (angles : Fin n → ℝ), (∀ i j, i < j → angles i + 5 = angles j))   -- The interior angles form an arithmetic progression with a common difference of 5°
  (h3 : ∀ (angles : Fin n → ℝ), (∃ i, angles i = 160))  -- The largest angle is 160°
  : n = 9 := sorry

end polygon_angle_arithmetic_progression_l84_84307


namespace savannah_wraps_4_with_third_roll_l84_84031

variable (gifts total_rolls : ℕ)
variable (wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 : ℕ)
variable (no_leftover : Prop)

def savannah_wrapping_presents (gifts total_rolls wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 : ℕ) (no_leftover : Prop) : Prop :=
  gifts = 12 ∧
  total_rolls = 3 ∧
  wrap_with_roll1 = 3 ∧
  wrap_with_roll2 = 5 ∧
  remaining_wrap_with_roll3 = gifts - (wrap_with_roll1 + wrap_with_roll2) ∧
  no_leftover = (total_rolls = 3) ∧ (wrap_with_roll1 + wrap_with_roll2 + remaining_wrap_with_roll3 = gifts)

theorem savannah_wraps_4_with_third_roll
  (h : savannah_wrapping_presents gifts total_rolls wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 no_leftover) :
  remaining_wrap_with_roll3 = 4 :=
by
  sorry

end savannah_wraps_4_with_third_roll_l84_84031


namespace find_cake_box_width_l84_84343

-- Define the dimensions of the carton
def carton_length := 25
def carton_width := 42
def carton_height := 60
def carton_volume := carton_length * carton_width * carton_height

-- Define the dimensions of the cake box
def cake_box_length := 8
variable (cake_box_width : ℝ) -- This is the unknown width we need to find
def cake_box_height := 5
def cake_box_volume := cake_box_length * cake_box_width * cake_box_height

-- Maximum number of cake boxes that can be placed in the carton
def max_cake_boxes := 210
def total_cake_boxes_volume := max_cake_boxes * cake_box_volume cake_box_width

-- Theorem to prove
theorem find_cake_box_width : cake_box_width = 7.5 :=
by
  sorry

end find_cake_box_width_l84_84343


namespace area_of_quadrilateral_NLMK_l84_84669

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_quadrilateral_NLMK 
  (AB BC AC AK CN CL : ℝ)
  (h_AB : AB = 13)
  (h_BC : BC = 20)
  (h_AC : AC = 21)
  (h_AK : AK = 4)
  (h_CN : CN = 1)
  (h_CL : CL = 20 / 21) : 
  triangle_area AB BC AC - 
  (1 * CL / (BC * AC) * triangle_area AB BC AC) - 
  (9 * (BC - CN) / (AB * BC) * triangle_area AB BC AC) -
  (16 * 41 / (169 * 21) * triangle_area AB BC AC) = 
  493737 / 11830 := 
sorry

end area_of_quadrilateral_NLMK_l84_84669


namespace seashells_total_l84_84550

theorem seashells_total :
  let sally := 9.5
  let tom := 7.2
  let jessica := 5.3
  let alex := 12.8
  sally + tom + jessica + alex = 34.8 :=
by
  sorry

end seashells_total_l84_84550


namespace g_at_5_l84_84668

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 22 * x ^ 3 + 47 * x ^ 2 - 44 * x + 24

theorem g_at_5 : g 5 = 104 := by
  sorry

end g_at_5_l84_84668


namespace Adam_ate_more_than_Bill_l84_84877

-- Definitions
def Sierra_ate : ℕ := 12
def Bill_ate : ℕ := Sierra_ate / 2
def total_pies_eaten : ℕ := 27
def Sierra_and_Bill_ate : ℕ := Sierra_ate + Bill_ate
def Adam_ate : ℕ := total_pies_eaten - Sierra_and_Bill_ate
def Adam_more_than_Bill : ℕ := Adam_ate - Bill_ate

-- Statement to prove
theorem Adam_ate_more_than_Bill :
  Adam_more_than_Bill = 3 :=
by
  sorry

end Adam_ate_more_than_Bill_l84_84877


namespace right_triangle_segment_ratio_l84_84856

-- Definitions of the triangle sides and hypotenuse
def right_triangle (AB BC : ℝ) : Prop :=
  AB/BC = 4/3

def hypotenuse (AB BC AC : ℝ) : Prop :=
  AC^2 = AB^2 + BC^2

def perpendicular_segment_ratio (AD CD : ℝ) : Prop :=
  AD / CD = 9/16

-- Final statement of the problem
theorem right_triangle_segment_ratio
  (AB BC AC AD CD : ℝ)
  (h1 : right_triangle AB BC)
  (h2 : hypotenuse AB BC AC)
  (h3 : perpendicular_segment_ratio AD CD) :
  CD / AD = 16/9 := sorry

end right_triangle_segment_ratio_l84_84856


namespace student_weighted_avg_larger_l84_84278

variable {u v w : ℚ}

theorem student_weighted_avg_larger (h1 : u < v) (h2 : v < w) :
  (4 * u + 6 * v + 20 * w) / 30 > (2 * u + 3 * v + 4 * w) / 9 := by
  sorry

end student_weighted_avg_larger_l84_84278


namespace sum_of_numbers_eq_8140_l84_84054

def numbers : List ℤ := [1200, 1300, 1400, 1510, 1530, 1200]

theorem sum_of_numbers_eq_8140 : (numbers.sum = 8140) :=
by
  sorry

end sum_of_numbers_eq_8140_l84_84054


namespace initial_money_correct_l84_84998

def initial_money (total: ℕ) (allowance: ℕ): ℕ :=
  total - allowance

theorem initial_money_correct: initial_money 18 8 = 10 :=
  by sorry

end initial_money_correct_l84_84998


namespace origin_inside_circle_range_l84_84363

theorem origin_inside_circle_range (m : ℝ) :
  ((0 - m)^2 + (0 + m)^2 < 8) → (-2 < m ∧ m < 2) :=
by
  intros h
  sorry

end origin_inside_circle_range_l84_84363


namespace necessary_but_not_sufficient_l84_84857

   theorem necessary_but_not_sufficient (a : ℝ) : a^2 > a → (a > 1) :=
   by {
     sorry
   }
   
end necessary_but_not_sufficient_l84_84857


namespace distance_between_P1_and_P2_l84_84146

-- Define the two points
def P1 : ℝ × ℝ := (2, 3)
def P2 : ℝ × ℝ := (5, 10)

-- Define the distance function
noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Define the theorem we want to prove
theorem distance_between_P1_and_P2 :
  distance P1 P2 = Real.sqrt 58 :=
by sorry

end distance_between_P1_and_P2_l84_84146


namespace other_root_of_quadratic_l84_84268

theorem other_root_of_quadratic (m : ℝ) :
  (∃ t : ℝ, (x^2 + m * x - 20 = 0) ∧ (x = -4 ∨ x = t)) → (t = 5) :=
by
  sorry

end other_root_of_quadratic_l84_84268


namespace factor_polynomial_l84_84112

theorem factor_polynomial :
  (x : ℝ) → (x^2 - 6*x + 9 - 64*x^4) = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by
  intro x
  sorry

end factor_polynomial_l84_84112


namespace vector_at_t5_l84_84647

theorem vector_at_t5 :
  ∃ (a : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ),
    a + (1 : ℝ) • d = (2, -1, 3) ∧
    a + (4 : ℝ) • d = (8, -5, 11) ∧
    a + (5 : ℝ) • d = (10, -19/3, 41/3) := 
sorry

end vector_at_t5_l84_84647


namespace even_function_A_value_l84_84421

-- Given function definition
def f (x : ℝ) (A : ℝ) : ℝ := (x + 1) * (x - A)

-- Statement to prove
theorem even_function_A_value (A : ℝ) (h : ∀ x : ℝ, f x A = f (-x) A) : A = 1 :=
by
  sorry

end even_function_A_value_l84_84421


namespace minimum_value_f_l84_84108

noncomputable def f (x y : ℝ) : ℝ :=
  (x^2 / (y - 2)^2) + (y^2 / (x - 2)^2)

theorem minimum_value_f (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ (a : ℝ), (∀ (b : ℝ), f x y >= b) ∧ a = 10 := sorry

end minimum_value_f_l84_84108


namespace quiz_answer_keys_count_l84_84684

noncomputable def count_answer_keys : ℕ :=
  (Nat.choose 10 5) * (Nat.factorial 6)

theorem quiz_answer_keys_count :
  count_answer_keys = 181440 := 
by
  -- Proof is skipped, using sorry
  sorry

end quiz_answer_keys_count_l84_84684


namespace lcm_of_numbers_is_750_l84_84321

-- Define the two numbers x and y
variables (x y : ℕ)

-- Given conditions as hypotheses
def product_of_numbers := 18750
def hcf_of_numbers := 25

-- The proof problem statement
theorem lcm_of_numbers_is_750 (h_product : x * y = product_of_numbers) 
                              (h_hcf : Nat.gcd x y = hcf_of_numbers) : Nat.lcm x y = 750 :=
by
  sorry

end lcm_of_numbers_is_750_l84_84321


namespace total_weight_is_correct_l84_84297

-- Define the variables
def envelope_weight : ℝ := 8.5
def additional_weight_per_envelope : ℝ := 2
def num_envelopes : ℝ := 880

-- Define the total weight calculation
def total_weight : ℝ := num_envelopes * (envelope_weight + additional_weight_per_envelope)

-- State the theorem to prove that the total weight is as expected
theorem total_weight_is_correct : total_weight = 9240 :=
by
  sorry

end total_weight_is_correct_l84_84297


namespace solve_for_x_l84_84174

theorem solve_for_x : ∃ x : ℝ, 3 * x - 48.2 = 0.25 * (4 * x + 56.8) → x = 31.2 :=
by sorry

end solve_for_x_l84_84174


namespace num_handshakes_ten_women_l84_84818

def num_handshakes (n : ℕ) : ℕ :=
(n * (n - 1)) / 2

theorem num_handshakes_ten_women :
  num_handshakes 10 = 45 :=
by
  sorry

end num_handshakes_ten_women_l84_84818


namespace even_numbers_average_18_l84_84384

variable (n : ℕ)
variable (avg : ℕ)

theorem even_numbers_average_18 (h : avg = 18) : n = 17 := 
    sorry

end even_numbers_average_18_l84_84384


namespace distance_from_y_axis_l84_84846

theorem distance_from_y_axis (P : ℝ × ℝ) (x : ℝ) (hx : P = (x, -9)) 
  (h : (abs (P.2) = 1/2 * abs (P.1))) :
  abs x = 18 :=
by
  sorry

end distance_from_y_axis_l84_84846


namespace find_schnauzers_l84_84789

theorem find_schnauzers (D S : ℕ) (h : 3 * D - 5 + (D - S) = 90) (hD : D = 20) : S = 45 :=
by
  sorry

end find_schnauzers_l84_84789


namespace fraction_sum_is_0_333_l84_84527

theorem fraction_sum_is_0_333 : (3 / 10 : ℝ) + (3 / 100) + (3 / 1000) = 0.333 := 
by
  sorry

end fraction_sum_is_0_333_l84_84527


namespace min_value_of_f_l84_84106

open Real

noncomputable def f (x : ℝ) := x + 1 / (x - 2)

theorem min_value_of_f : ∃ x : ℝ, x > 2 ∧ ∀ y : ℝ, y > 2 → f y ≥ f 3 := by
  sorry

end min_value_of_f_l84_84106


namespace total_length_of_fence_l84_84679

theorem total_length_of_fence (x : ℝ) (h1 : 2 * x * x = 1250) : 2 * x + 2 * x = 100 :=
by
  sorry

end total_length_of_fence_l84_84679


namespace tangency_lines_intersect_at_diagonal_intersection_point_l84_84581

noncomputable def point := Type
noncomputable def line := Type

noncomputable def tangency (C : point) (l : line) : Prop := sorry
noncomputable def circumscribed (Q : point × point × point × point) (C : point) : Prop := sorry
noncomputable def intersects (l1 l2 : line) (P : point) : Prop := sorry
noncomputable def connects_opposite_tangency (Q : point × point × point × point) (l1 l2 : line) : Prop := sorry
noncomputable def diagonals_intersect_at (Q : point × point × point × point) (P : point) : Prop := sorry

theorem tangency_lines_intersect_at_diagonal_intersection_point :
  ∀ (Q : point × point × point × point) (C P : point), 
  circumscribed Q C →
  diagonals_intersect_at Q P →
  ∀ (l1 l2 : line), connects_opposite_tangency Q l1 l2 →
  intersects l1 l2 P :=
sorry

end tangency_lines_intersect_at_diagonal_intersection_point_l84_84581


namespace expand_product_l84_84997

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := 
  sorry

end expand_product_l84_84997


namespace common_divisors_sum_diff_l84_84710

theorem common_divisors_sum_diff (A B : ℤ) (h : Int.gcd A B = 1) : 
  {d : ℤ | d ∣ A + B ∧ d ∣ A - B} = {1, 2} :=
sorry

end common_divisors_sum_diff_l84_84710


namespace smallest_y_76545_l84_84912

theorem smallest_y_76545 (y : ℕ) (h1 : ∀ z : ℕ, 0 < z → (76545 * z = k ^ 2 → (3 ∣ z ∨ 5 ∣ z) → z = y)) : y = 7 :=
sorry

end smallest_y_76545_l84_84912


namespace smallest_value_of_n_l84_84403

/-- Given that Casper has exactly enough money to buy either 
  18 pieces of red candy, 20 pieces of green candy, 
  25 pieces of blue candy, or n pieces of purple candy where 
  each purple candy costs 30 cents, prove that the smallest 
  possible value of n is 30.
-/
theorem smallest_value_of_n
  (r g b n : ℕ)
  (h : 18 * r = 20 * g ∧ 20 * g = 25 * b ∧ 25 * b = 30 * n) : 
  n = 30 :=
sorry

end smallest_value_of_n_l84_84403


namespace problem_part1_problem_part2_l84_84372

def U : Set ℕ := {x | 0 < x ∧ x < 9}

def S : Set ℕ := {1, 3, 5}

def T : Set ℕ := {3, 6}

theorem problem_part1 : S ∩ T = {3} := by
  sorry

theorem problem_part2 : U \ (S ∪ T) = {2, 4, 7, 8} := by
  sorry

end problem_part1_problem_part2_l84_84372


namespace simpsons_hats_l84_84145

variable (S : ℕ)
variable (O : ℕ)

-- Define the conditions: O'Brien's hats before losing one
def obriens_hats_before : Prop := O = 2 * S + 5

-- Define the current number of O'Brien's hats
def obriens_current_hats : Prop := O = 34 + 1

-- Main theorem statement
theorem simpsons_hats : obriens_hats_before S O ∧ obriens_current_hats O → S = 15 := 
by
  sorry

end simpsons_hats_l84_84145


namespace value_of_b_l84_84540

theorem value_of_b (y : ℝ) (b : ℝ) (h_pos : y > 0) (h_eqn : (7 * y) / b + (3 * y) / 10 = 0.6499999999999999 * y) : 
  b = 70 / 61.99999999999999 :=
sorry

end value_of_b_l84_84540


namespace g_g_3_eq_3606651_l84_84093

def g (x: ℤ) : ℤ := 3 * x^3 + 3 * x^2 - x + 1

theorem g_g_3_eq_3606651 : g (g 3) = 3606651 := 
by {
  sorry
}

end g_g_3_eq_3606651_l84_84093


namespace sufficient_but_not_necessary_condition_for_q_l84_84359

variable (p q r : Prop)

theorem sufficient_but_not_necessary_condition_for_q (hp : p → r) (hq1 : r → q) (hq2 : ¬(q → r)) : 
  (p → q) ∧ ¬(q → p) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_q_l84_84359


namespace geometric_sequence_sum_l84_84743

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a n = a 1 * q ^ n

theorem geometric_sequence_sum (h : geometric_sequence a 2) (h_sum : a 1 + a 2 = 3) :
  a 4 + a 5 = 24 := by
  sorry

end geometric_sequence_sum_l84_84743


namespace problem1_problem2_l84_84960

-- Proof for Problem 1
theorem problem1 : (99^2 + 202*99 + 101^2) = 40000 := 
by {
  -- proof
  sorry
}

-- Proof for Problem 2
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : ((1 / (x - 1) - 2) / ((2 * x - 3) / (x^2 - 1))) = -x - 1 :=
by {
  -- proof
  sorry
}

end problem1_problem2_l84_84960


namespace digit_2023_in_fractional_expansion_l84_84588

theorem digit_2023_in_fractional_expansion :
  ∃ d : ℕ, (d = 4) ∧ (∃ n_block : ℕ, n_block = 6 ∧ (∃ p : Nat, p = 2023 ∧ ∃ r : ℕ, r = p % n_block ∧ r = 1)) :=
sorry

end digit_2023_in_fractional_expansion_l84_84588


namespace sum_of_roots_of_quadratic_l84_84795

noncomputable def x1_x2_roots_properties : Prop :=
  ∃ x₁ x₂ : ℝ, (x₁ + x₂ = 3) ∧ (x₁ * x₂ = -4)

theorem sum_of_roots_of_quadratic :
  ∃ x₁ x₂ : ℝ, (x₁ * x₂ = -4) → (x₁ + x₂ = 3) :=
by
  sorry

end sum_of_roots_of_quadratic_l84_84795


namespace bus_seating_capacity_l84_84035

-- Conditions
def left_side_seats : ℕ := 15
def right_side_seats : ℕ := left_side_seats - 3
def seat_capacity : ℕ := 3
def back_seat_capacity : ℕ := 9
def total_seats : ℕ := left_side_seats + right_side_seats

-- Proof problem statement
theorem bus_seating_capacity :
  (total_seats * seat_capacity) + back_seat_capacity = 90 := by
  sorry

end bus_seating_capacity_l84_84035


namespace equation_of_line_with_x_intercept_and_slope_l84_84408

theorem equation_of_line_with_x_intercept_and_slope :
  ∃ (a b c : ℝ), a * x - b * y + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
sorry

end equation_of_line_with_x_intercept_and_slope_l84_84408


namespace imaginary_part_of_conjugate_l84_84541

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

theorem imaginary_part_of_conjugate :
  ∀ (z : ℂ), z = (1+i)^2 / (1-i) → (complex_conjugate z).im = -1 :=
by
  sorry

end imaginary_part_of_conjugate_l84_84541


namespace min_vertical_segment_length_l84_84751

noncomputable def vertical_segment_length (x : ℝ) : ℝ :=
  abs (|x| - (-x^2 - 4*x - 3))

theorem min_vertical_segment_length :
  ∃ x : ℝ, vertical_segment_length x = 3 / 4 :=
by
  sorry

end min_vertical_segment_length_l84_84751


namespace face_value_of_each_ticket_without_tax_l84_84423

theorem face_value_of_each_ticket_without_tax (total_people : ℕ) (total_cost : ℝ) (sales_tax : ℝ) (face_value : ℝ)
  (h1 : total_people = 25)
  (h2 : total_cost = 945)
  (h3 : sales_tax = 0.05)
  (h4 : total_cost = (1 + sales_tax) * face_value * total_people) :
  face_value = 36 := by
  sorry

end face_value_of_each_ticket_without_tax_l84_84423


namespace charles_total_money_l84_84686

-- Definitions based on the conditions in step a)
def number_of_pennies : ℕ := 6
def number_of_nickels : ℕ := 3
def value_of_penny : ℕ := 1
def value_of_nickel : ℕ := 5

-- Calculations in Lean terms
def total_pennies_value : ℕ := number_of_pennies * value_of_penny
def total_nickels_value : ℕ := number_of_nickels * value_of_nickel
def total_money : ℕ := total_pennies_value + total_nickels_value

-- The final proof statement based on step c)
theorem charles_total_money : total_money = 21 := by
  sorry

end charles_total_money_l84_84686


namespace otimes_square_neq_l84_84024

noncomputable def otimes (a b : ℝ) : ℝ :=
  if a > b then a else b

theorem otimes_square_neq (a b : ℝ) (h : a ≠ b) : (otimes a b) ^ 2 ≠ otimes (a ^ 2) (b ^ 2) := by
  sorry

end otimes_square_neq_l84_84024


namespace factory_earnings_l84_84567

-- Definition of constants and functions based on the conditions:
def material_A_production (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def material_B_production (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def convert_B_to_C (material_B : ℕ) : ℕ := material_B / 2
def earnings (amount : ℕ) (price_per_unit : ℕ) : ℕ := amount * price_per_unit

-- Given conditions for the problem:
def hours_machine_1_and_2 : ℕ := 23
def hours_machine_3 : ℕ := 23
def hours_machine_4 : ℕ := 12
def rate_A_machine_1_and_2 : ℕ := 2
def rate_B_machine_1_and_2 : ℕ := 1
def rate_A_machine_3_and_4 : ℕ := 3
def rate_B_machine_3_and_4 : ℕ := 2
def price_A : ℕ := 50
def price_C : ℕ := 100

-- Calculations based on problem conditions:
noncomputable def total_A : ℕ := 
  2 * material_A_production hours_machine_1_and_2 rate_A_machine_1_and_2 + 
  material_A_production hours_machine_3 rate_A_machine_3_and_4 + 
  material_A_production hours_machine_4 rate_A_machine_3_and_4

noncomputable def total_B : ℕ := 
  2 * material_B_production hours_machine_1_and_2 rate_B_machine_1_and_2 + 
  material_B_production hours_machine_3 rate_B_machine_3_and_4 + 
  material_B_production hours_machine_4 rate_B_machine_3_and_4

noncomputable def total_C : ℕ := convert_B_to_C total_B

noncomputable def total_earnings : ℕ :=
  earnings total_A price_A + earnings total_C price_C

-- The theorem to prove the total earnings:
theorem factory_earnings : total_earnings = 15650 :=
by
  sorry

end factory_earnings_l84_84567


namespace quadratic_inequality_l84_84245

theorem quadratic_inequality (a : ℝ) :
  (¬ (∃ x : ℝ, a * x^2 + 2 * x + 3 ≤ 0)) ↔ (a > 1 / 3) :=
by 
  sorry

end quadratic_inequality_l84_84245


namespace clearance_sale_total_earnings_l84_84435

-- Define the variables used in the problem
def total_jackets := 214
def price_before_noon := 31.95
def price_after_noon := 18.95
def jackets_sold_after_noon := 133

-- Calculate the total earnings
def total_earnings_from_clearance_sale : Prop :=
  (133 * 18.95 + (214 - 133) * 31.95) = 5107.30

-- State the theorem to be proven
theorem clearance_sale_total_earnings : total_earnings_from_clearance_sale :=
  by sorry

end clearance_sale_total_earnings_l84_84435


namespace problem_statement_l84_84360

variables {a b c : ℝ}

theorem problem_statement 
  (h1 : a^2 + a * b + b^2 = 9)
  (h2 : b^2 + b * c + c^2 = 52)
  (h3 : c^2 + c * a + a^2 = 49) : 
  (49 * b^2 - 33 * b * c + 9 * c^2) / a^2 = 52 :=
by
  sorry

end problem_statement_l84_84360


namespace range_of_a_l84_84962

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → -3 * x^2 + a ≤ 0) ↔ a ≤ 3 := by
  sorry

end range_of_a_l84_84962


namespace problem1_problem2_l84_84697

-- Problem 1
theorem problem1 (m n : ℚ) (h : m ≠ n) : 
  (m / (m - n)) + (n / (n - m)) = 1 := 
by
  -- Proof steps would go here
  sorry

-- Problem 2
theorem problem2 (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) : 
  (2 / (x^2 - 1)) / (1 / (x + 1)) = 2 / (x - 1) := 
by
  -- Proof steps would go here
  sorry

end problem1_problem2_l84_84697


namespace staff_meeting_doughnuts_l84_84369

theorem staff_meeting_doughnuts (n_d n_s n_l : ℕ) (h₁ : n_d = 50) (h₂ : n_s = 19) (h₃ : n_l = 12) :
  (n_d - n_l) / n_s = 2 :=
by
  sorry

end staff_meeting_doughnuts_l84_84369


namespace triangle_classification_l84_84827

theorem triangle_classification 
  (a b c : ℝ) 
  (h : (b^2 + a^2) * (b - a) = b * c^2 - a * c^2) : 
  (a = b ∨ a^2 + b^2 = c^2) :=
by sorry

end triangle_classification_l84_84827


namespace value_of_x_l84_84311

theorem value_of_x (x : ℝ) (hx_pos : 0 < x) (hx_eq : x^2 = 1024) : x = 32 := 
by
  sorry

end value_of_x_l84_84311


namespace portraits_count_l84_84965

theorem portraits_count (P S : ℕ) (h1 : S = 6 * P) (h2 : P + S = 200) : P = 28 := 
by
  -- The proof will be here.
  sorry

end portraits_count_l84_84965


namespace determine_g_two_l84_84006

variables (a b c d p q r s : ℝ) -- Define variables a, b, c, d, p, q, r, s as real numbers
variables (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) -- The conditions a < b < c < d

noncomputable def f (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d
noncomputable def g (x : ℝ) : ℝ := (x - 1/p) * (x - 1/q) * (x - 1/r) * (x - 1/s)

noncomputable def g_two := g 2
noncomputable def f_two := f 2

theorem determine_g_two :
  g_two a b c d = (16 + 8*a + 4*b + 2*c + d) / (p*q*r*s) :=
sorry

end determine_g_two_l84_84006


namespace larger_of_two_numbers_l84_84173

  theorem larger_of_two_numbers (x y : ℕ) 
    (h₁ : x + y = 37) 
    (h₂ : x - y = 5) 
    : x = 21 :=
  sorry
  
end larger_of_two_numbers_l84_84173


namespace remainder_problem_l84_84409

theorem remainder_problem (n m q1 q2 : ℤ) (h1 : n = 11 * q1 + 1) (h2 : m = 17 * q2 + 3) :
  ∃ r : ℤ, (r = (5 * n + 3 * m) % 11) ∧ (r = (7 * q2 + 3) % 11) :=
by
  sorry

end remainder_problem_l84_84409


namespace measure_of_third_angle_l84_84947

-- Definitions based on given conditions
def angle_sum_of_triangle := 180
def angle1 := 30
def angle2 := 60

-- Problem Statement: Prove the third angle (angle3) in a triangle is 90 degrees
theorem measure_of_third_angle (angle_sum : ℕ := angle_sum_of_triangle) 
  (a1 : ℕ := angle1) (a2 : ℕ := angle2) : (angle_sum - (a1 + a2)) = 90 :=
by
  sorry

end measure_of_third_angle_l84_84947


namespace average_tickets_sold_by_female_l84_84429

-- Define the conditions as Lean expressions.

def totalMembers (M : ℕ) : ℕ := M + 2 * M
def totalTickets (F : ℕ) (M : ℕ) : ℕ := 58 * M + F * 2 * M
def averageTicketsPerMember (F : ℕ) (M : ℕ) : ℕ := (totalTickets F M) / (totalMembers M)

theorem average_tickets_sold_by_female (F M : ℕ) 
  (h1 : 66 * (totalMembers M) = totalTickets F M) :
  F = 70 :=
by
  sorry

end average_tickets_sold_by_female_l84_84429


namespace represent_nat_as_combinations_l84_84886

theorem represent_nat_as_combinations (n : ℕ) :
  ∃ x y z : ℕ,
  (0 ≤ x ∧ x < y ∧ y < z ∨ 0 = x ∧ x = y ∧ y < z) ∧
  (n = Nat.choose x 1 + Nat.choose y 2 + Nat.choose z 3) :=
sorry

end represent_nat_as_combinations_l84_84886


namespace perimeter_of_triangle_is_36_l84_84660

variable (inradius : ℝ)
variable (area : ℝ)
variable (P : ℝ)

theorem perimeter_of_triangle_is_36 (h1 : inradius = 2.5) (h2 : area = 45) : 
  P / 2 * inradius = area → P = 36 :=
sorry

end perimeter_of_triangle_is_36_l84_84660


namespace find_m_l84_84876

theorem find_m (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) 
  (hS : ∀ n, S n = n^2 - 6 * n) :
  (forall m, (5 < a m ∧ a m < 8) → m = 7)
:= 
by
  sorry

end find_m_l84_84876


namespace roots_difference_l84_84339

theorem roots_difference :
  let a := 2 
  let b := 5 
  let c := -12
  let disc := b*b - 4*a*c
  let root1 := (-b + Real.sqrt disc) / (2 * a)
  let root2 := (-b - Real.sqrt disc) / (2 * a)
  let larger_root := max root1 root2
  let smaller_root := min root1 root2
  larger_root - smaller_root = 5.5 := by
  sorry

end roots_difference_l84_84339


namespace ratio_of_a_over_5_to_b_over_4_l84_84126

theorem ratio_of_a_over_5_to_b_over_4 (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) : (a/5) / (b/4) = 1 :=
sorry

end ratio_of_a_over_5_to_b_over_4_l84_84126


namespace alpha_value_l84_84867

open Complex

theorem alpha_value (α β : ℂ) (h1 : β = 2 + 3 * I) (h2 : (α + β).im = 0) (h3 : (I * (2 * α - β)).im = 0) : α = 6 + 4 * I :=
by
  sorry

end alpha_value_l84_84867


namespace additional_income_needed_to_meet_goal_l84_84253

def monthly_current_income : ℤ := 4000
def annual_goal : ℤ := 60000
def additional_amount_per_month (monthly_current_income annual_goal : ℤ) : ℤ :=
  (annual_goal - (monthly_current_income * 12)) / 12

theorem additional_income_needed_to_meet_goal :
  additional_amount_per_month monthly_current_income annual_goal = 1000 :=
by
  sorry

end additional_income_needed_to_meet_goal_l84_84253


namespace unit_vector_norm_equal_l84_84184

variables (a b : EuclideanSpace ℝ (Fin 2)) -- assuming 2D Euclidean space for simplicity

def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop := ‖v‖ = 1

theorem unit_vector_norm_equal {a b : EuclideanSpace ℝ (Fin 2)}
  (ha : is_unit_vector a) (hb : is_unit_vector b) : ‖a‖ = ‖b‖ :=
by 
  sorry

end unit_vector_norm_equal_l84_84184


namespace find_K_l84_84402

theorem find_K : ∃ K : ℕ, (64 ^ (2 / 3) * 16 ^ 2) / 4 = 2 ^ K :=
by
  use 10
  sorry

end find_K_l84_84402


namespace age_of_father_now_l84_84368

variable (M F : ℕ)

theorem age_of_father_now :
  (M = 2 * F / 5) ∧ (M + 14 = (F + 14) / 2) → F = 70 :=
by 
sorry

end age_of_father_now_l84_84368


namespace jan_more_miles_than_ian_l84_84873

noncomputable def distance_diff (d t s : ℝ) : ℝ :=
  let han_distance := (s + 10) * (t + 2)
  let jan_distance := (s + 15) * (t + 3)
  jan_distance - (d + 100)

theorem jan_more_miles_than_ian {d t s : ℝ} (H : d = s * t) (H_han : d + 100 = (s + 10) * (t + 2)) : distance_diff d t s = 165 :=
by {
  sorry
}

end jan_more_miles_than_ian_l84_84873


namespace vertex_of_parabola_l84_84902

theorem vertex_of_parabola :
  ∃ (a b c : ℝ), 
      (4 * a - 2 * b + c = 9) ∧ 
      (16 * a + 4 * b + c = 9) ∧ 
      (49 * a + 7 * b + c = 16) ∧ 
      (-b / (2 * a) = 1) :=
by {
  -- we need to provide the proof here; sorry is a placeholder
  sorry
}

end vertex_of_parabola_l84_84902


namespace express_y_in_terms_of_y_l84_84816

variable (x : ℝ)

theorem express_y_in_terms_of_y (y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
sorry

end express_y_in_terms_of_y_l84_84816


namespace relationship_between_y_values_l84_84797

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 1

variables (m : ℝ) (y1 y2 y3 : ℝ)
variables (h : m > 0)
variables (h1 : y1 = quadratic_function m (-1))
variables (h2 : y2 = quadratic_function m (5 / 2))
variables (h3 : y3 = quadratic_function m 6)

theorem relationship_between_y_values : y3 > y1 ∧ y1 > y2 :=
by
  sorry

end relationship_between_y_values_l84_84797


namespace candle_remaining_length_l84_84765

-- Define the initial length of the candle and the burn rate
def initial_length : ℝ := 20
def burn_rate : ℝ := 5

-- Define the remaining length function
def remaining_length (t : ℝ) : ℝ := initial_length - burn_rate * t

-- Prove the relationship between time and remaining length for the given range of time
theorem candle_remaining_length (t : ℝ) (ht: 0 ≤ t ∧ t ≤ 4) : remaining_length t = 20 - 5 * t :=
by
  dsimp [remaining_length]
  sorry

end candle_remaining_length_l84_84765


namespace g_of_neg5_eq_651_over_16_l84_84892

def f (x : ℝ) : ℝ := 4 * x + 6

def g (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 7

theorem g_of_neg5_eq_651_over_16 : g (-5) = 651 / 16 := by
  sorry

end g_of_neg5_eq_651_over_16_l84_84892


namespace table_height_l84_84350

theorem table_height (l w h : ℝ) (h1 : l + h - w = 38) (h2 : w + h - l = 34) : h = 36 :=
by
  sorry

end table_height_l84_84350


namespace j_at_4_l84_84045

noncomputable def h (x : ℚ) : ℚ := 5 / (3 - x)

noncomputable def h_inv (x : ℚ) : ℚ := (3 * x - 5) / x

noncomputable def j (x : ℚ) : ℚ := (1 / h_inv x) + 7

theorem j_at_4 : j 4 = 53 / 7 :=
by
  -- Proof steps would be inserted here.
  sorry

end j_at_4_l84_84045


namespace common_ratio_q_l84_84276

noncomputable def Sn (n : ℕ) (a1 q : ℝ) := a1 * (1 - q^n) / (1 - q)

theorem common_ratio_q (a1 : ℝ) (q : ℝ) (h : q ≠ 1) (h1 : 6 * Sn 4 a1 q = Sn 5 a1 q + 5 * Sn 6 a1 q) : q = -6/5 := by
  sorry

end common_ratio_q_l84_84276


namespace length_breadth_difference_l84_84371

theorem length_breadth_difference (L W : ℝ) 
  (h1 : W = 1/2 * L) 
  (h2 : L * W = 288) : L - W = 12 :=
by
  sorry

end length_breadth_difference_l84_84371


namespace even_function_a_value_l84_84836

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x^2 + a * x - 1) = ((-x)^2 + a * (-x) - 1)) ↔ a = 0 :=
by
  sorry

end even_function_a_value_l84_84836


namespace probability_sin_in_interval_half_l84_84250

noncomputable def probability_sin_interval : ℝ :=
  let a := - (Real.pi / 2)
  let b := Real.pi / 2
  let interval_length := b - a
  (b - 0) / interval_length

theorem probability_sin_in_interval_half :
  probability_sin_interval = 1 / 2 := by
  sorry

end probability_sin_in_interval_half_l84_84250


namespace gcd_xyz_times_xyz_is_square_l84_84933

theorem gcd_xyz_times_xyz_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, k^2 = Nat.gcd x (Nat.gcd y z) * x * y * z :=
by
  sorry

end gcd_xyz_times_xyz_is_square_l84_84933


namespace trader_excess_donations_l84_84217

-- Define the conditions
def profit : ℤ := 1200
def allocation_percentage : ℤ := 60
def family_donation : ℤ := 250
def friends_donation : ℤ := (20 * family_donation) / 100 + family_donation
def total_family_friends_donation : ℤ := family_donation + friends_donation
def local_association_donation : ℤ := 15 * total_family_friends_donation / 10
def total_donations : ℤ := family_donation + friends_donation + local_association_donation
def allocated_amount : ℤ := allocation_percentage * profit / 100

-- Theorem statement (Question)
theorem trader_excess_donations : total_donations - allocated_amount = 655 :=
by
  sorry

end trader_excess_donations_l84_84217


namespace find_n_l84_84477

theorem find_n (n : ℕ) (h : 2 * 2^2 * 2^n = 2^10) : n = 7 :=
sorry

end find_n_l84_84477


namespace distinct_sequences_ten_flips_l84_84312

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l84_84312


namespace Yasmin_children_count_l84_84337

theorem Yasmin_children_count (Y : ℕ) (h1 : 2 * Y + Y = 6) : Y = 2 :=
by
  sorry

end Yasmin_children_count_l84_84337


namespace probability_of_circle_l84_84931

theorem probability_of_circle :
  let numCircles := 4
  let numSquares := 3
  let numTriangles := 3
  let totalFigures := numCircles + numSquares + numTriangles
  let probability := numCircles / totalFigures
  probability = 2 / 5 :=
by
  sorry

end probability_of_circle_l84_84931


namespace solve_ineq_l84_84887

noncomputable def f (x : ℝ) : ℝ := (2 / (x + 2)) + (4 / (x + 8)) - (7 / 3)

theorem solve_ineq (x : ℝ) : 
  (f x ≤ 0) ↔ (x ∈ Set.Ioc (-8) 4) := 
sorry

end solve_ineq_l84_84887


namespace sum_of_three_consecutive_divisible_by_three_l84_84110

theorem sum_of_three_consecutive_divisible_by_three (n : ℕ) : ∃ k : ℕ, (n + (n + 1) + (n + 2)) = 3 * k := by
  sorry

end sum_of_three_consecutive_divisible_by_three_l84_84110


namespace negation_universal_exists_l84_84335

open Classical

theorem negation_universal_exists :
  (¬ ∀ x : ℝ, x > 0 → (x^2 - x + 3 > 0)) ↔ ∃ x : ℝ, x > 0 ∧ (x^2 - x + 3 ≤ 0) :=
by
  sorry

end negation_universal_exists_l84_84335


namespace maximize_wind_power_l84_84059

variable {C S ρ v_0 : ℝ}

theorem maximize_wind_power : 
  ∃ v : ℝ, (∀ (v' : ℝ),
           let F := (C * S * ρ * (v_0 - v)^2) / 2;
           let N := F * v;
           let N' := (C * S * ρ / 2) * (v_0^2 - 4 * v_0 * v + 3 * v^2);
           N' = 0
         → N ≤ (C * S * ρ / 2) * (v_0^2 * (v_0/3) - 2 * v_0 * (v_0/3)^2 + (v_0/3)^3)) ∧ v = v_0 / 3 :=
by sorry

end maximize_wind_power_l84_84059


namespace megan_bottles_l84_84032

theorem megan_bottles (initial_bottles drank gave_away remaining_bottles : ℕ) 
  (h1 : initial_bottles = 45)
  (h2 : drank = 8)
  (h3 : gave_away = 12) :
  remaining_bottles = initial_bottles - (drank + gave_away) :=
by 
  sorry

end megan_bottles_l84_84032


namespace line_tangent_to_circle_l84_84737

theorem line_tangent_to_circle (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 2 * k + 3 = 0 → x^2 + (y + 1)^2 = 4) → k = 3 / 4 :=
by 
  intro h
  sorry

end line_tangent_to_circle_l84_84737


namespace total_tweets_correct_l84_84948

-- Define the rates at which Polly tweets under different conditions
def happy_rate : ℕ := 18
def hungry_rate : ℕ := 4
def mirror_rate : ℕ := 45

-- Define the durations of each activity
def happy_duration : ℕ := 20
def hungry_duration : ℕ := 20
def mirror_duration : ℕ := 20

-- Compute the total number of tweets
def total_tweets : ℕ := happy_rate * happy_duration + hungry_rate * hungry_duration + mirror_rate * mirror_duration

-- Statement to prove
theorem total_tweets_correct : total_tweets = 1340 := by
  sorry

end total_tweets_correct_l84_84948


namespace a1_greater_than_floor_2n_over_3_l84_84807

theorem a1_greater_than_floor_2n_over_3
  (n : ℕ)
  (a : ℕ → ℕ)
  (h1 : ∀ i j : ℕ, i < j → i ≤ n ∧ j ≤ n → a i < a j)
  (h2 : ∀ i j : ℕ, i ≠ j → i ≤ n ∧ j ≤ n → lcm (a i) (a j) > 2 * n)
  (h_max : ∀ i : ℕ, i ≤ n → a i ≤ 2 * n) :
  a 1 > (2 * n) / 3 :=
by
  sorry

end a1_greater_than_floor_2n_over_3_l84_84807


namespace no_solution_for_inequalities_l84_84175

theorem no_solution_for_inequalities (x : ℝ) : ¬ ((6 * x - 2 < (x + 2) ^ 2) ∧ ((x + 2) ^ 2 < 9 * x - 5)) :=
by sorry

end no_solution_for_inequalities_l84_84175


namespace distribute_items_in_identical_bags_l84_84604

noncomputable def count_ways_to_distribute_items (num_items : ℕ) (num_bags : ℕ) : ℕ :=
  if h : num_items = 5 ∧ num_bags = 3 then 36 else 0

theorem distribute_items_in_identical_bags :
  count_ways_to_distribute_items 5 3 = 36 :=
by
  -- Proof is skipped as per instructions
  sorry

end distribute_items_in_identical_bags_l84_84604


namespace total_sonnets_written_l84_84990

-- Definitions of conditions given in the problem
def lines_per_sonnet : ℕ := 14
def sonnets_read : ℕ := 7
def unread_lines : ℕ := 70

-- Definition of a measuring line for further calculation
def unread_sonnets : ℕ := unread_lines / lines_per_sonnet

-- The assertion we need to prove
theorem total_sonnets_written : 
  unread_sonnets + sonnets_read = 12 := by 
  sorry

end total_sonnets_written_l84_84990


namespace area_of_triangle_PQR_l84_84071

def Point := (ℝ × ℝ)
def area_of_triangle (P Q R : Point) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

def P : Point := (1, 1)
def Q : Point := (4, 5)
def R : Point := (7, 2)

theorem area_of_triangle_PQR :
  area_of_triangle P Q R = 10.5 := by
  sorry

end area_of_triangle_PQR_l84_84071


namespace expression_value_l84_84034

theorem expression_value (a b c d m : ℚ) (h1 : a + b = 0) (h2 : a ≠ 0) (h3 : c * d = 1) (h4 : m = -5 ∨ m = 1) :
  |m| - (a / b) + ((a + b) / 2020) - (c * d) = 1 ∨ |m| - (a / b) + ((a + b) / 2020) - (c * d) = 5 :=
by sorry

end expression_value_l84_84034


namespace smallest_x_l84_84929

theorem smallest_x (y : ℤ) (h1 : 0.9 = (y : ℚ) / (151 + x)) (h2 : 0 < x) (h3 : 0 < y) : x = 9 :=
sorry

end smallest_x_l84_84929


namespace common_tangent_exists_l84_84742

theorem common_tangent_exists:
  ∃ (a b c : ℕ), (a + b + c = 11) ∧
  ( ∀ (x y : ℝ),
      (y = x^2 + 12/5) ∧ 
      (x = y^2 + 99/10) ∧ 
      (a*x + b*y = c) ∧ 
      0 < a ∧ 0 < b ∧ 0 < c ∧ 
      Int.gcd (Int.gcd a b) c = 1
  ) := 
by
  sorry

end common_tangent_exists_l84_84742


namespace determinant_problem_l84_84908

variables {p q r s : ℝ}

theorem determinant_problem
  (h : p * s - q * r = 5) :
  p * (4 * r + 2 * s) - (4 * p + 2 * q) * r = 10 := 
sorry

end determinant_problem_l84_84908


namespace sqrt6_op_sqrt6_l84_84701

variable (x y : ℝ)

noncomputable def op (x y : ℝ) := (x + y)^2 - (x - y)^2

theorem sqrt6_op_sqrt6 : ∀ (x y : ℝ), op (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end sqrt6_op_sqrt6_l84_84701


namespace polly_breakfast_minutes_l84_84308
open Nat

theorem polly_breakfast_minutes (B : ℕ) 
  (lunch_minutes : ℕ)
  (dinner_4_days_minutes : ℕ)
  (dinner_3_days_minutes : ℕ)
  (total_minutes : ℕ)
  (h1 : lunch_minutes = 5 * 7)
  (h2 : dinner_4_days_minutes = 10 * 4)
  (h3 : dinner_3_days_minutes = 30 * 3)
  (h4 : total_minutes = 305) 
  (h5 : 7 * B + lunch_minutes + dinner_4_days_minutes + dinner_3_days_minutes = total_minutes) :
  B = 20 :=
by
  -- proof omitted
  sorry

end polly_breakfast_minutes_l84_84308


namespace negation_of_forall_x_geq_1_l84_84120

theorem negation_of_forall_x_geq_1 :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 1)) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by
  sorry

end negation_of_forall_x_geq_1_l84_84120


namespace rectangle_area_comparison_l84_84520

theorem rectangle_area_comparison 
  {A A' B B' C C' D D': ℝ} 
  (h_A: A ≤ A') 
  (h_B: B ≤ B') 
  (h_C: C ≤ C') 
  (h_D: D ≤ B') : 
  A + B + C + D ≤ A' + B' + C' + D' := 
by 
  sorry

end rectangle_area_comparison_l84_84520


namespace general_formula_for_a_n_l84_84868

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Defining a_n as a function of n assuming it's an arithmetic sequence.
noncomputable def a (x : ℝ) (n : ℕ) : ℝ :=
  if x = 1 then 2 * n - 4 else if x = 3 then 4 - 2 * n else 0

theorem general_formula_for_a_n (x : ℝ) (n : ℕ) (h1 : a x 1 = f (x + 1))
  (h2 : a x 2 = 0) (h3 : a x 3 = f (x - 1)) :
  (x = 1 → a x n = 2 * n - 4) ∧ (x = 3 → a x n = 4 - 2 * n) :=
by sorry

end general_formula_for_a_n_l84_84868


namespace total_cost_six_years_l84_84472

variable {fees : ℕ → ℝ}

-- Conditions
def fee_first_year : fees 1 = 80 := sorry

def fee_increase (n : ℕ) : fees (n + 1) = fees n + (10 + 2 * (n - 1)) := 
sorry

-- Proof problem: Prove that the total cost is 670
theorem total_cost_six_years : (fees 1 + fees 2 + fees 3 + fees 4 + fees 5 + fees 6) = 670 :=
by sorry

end total_cost_six_years_l84_84472


namespace quadratic_completing_square_l84_84631

theorem quadratic_completing_square:
  ∃ (b c : ℝ), (∀ x : ℝ, x^2 + 900 * x + 1800 = (x + b)^2 + c) ∧ (c / b = -446.22222) :=
by
  -- We'll skip the proof steps here
  sorry

end quadratic_completing_square_l84_84631


namespace total_fruits_l84_84655

theorem total_fruits (Mike_fruits Matt_fruits Mark_fruits : ℕ)
  (Mike_receives : Mike_fruits = 3)
  (Matt_receives : Matt_fruits = 2 * Mike_fruits)
  (Mark_receives : Mark_fruits = Mike_fruits + Matt_fruits) :
  Mike_fruits + Matt_fruits + Mark_fruits = 18 := by
  sorry

end total_fruits_l84_84655


namespace area_of_rectangle_l84_84026

theorem area_of_rectangle (AB AC : ℝ) (angle_ABC : ℝ) (h_AB : AB = 15) (h_AC : AC = 17) (h_angle_ABC : angle_ABC = 90) :
  ∃ BC : ℝ, (BC = 8) ∧ (AB * BC = 120) :=
by
  sorry

end area_of_rectangle_l84_84026


namespace f_monotone_on_0_to_2_find_range_a_part2_find_range_a_part3_l84_84139

noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 2^x + a

theorem f_monotone_on_0_to_2 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 > f x2 :=
sorry

theorem find_range_a_part2 : (∀ x1 : ℝ, x1 ∈ (Set.Icc (1/2) 1) → 
  ∃ x2 : ℝ, x2 ∈ (Set.Icc 2 3) ∧ f x1 ≥ g x2 a) → a ≤ 1 :=
sorry

theorem find_range_a_part3 : (∃ x : ℝ, x ∈ (Set.Icc 0 2) ∧ f x ≤ g x a) → a ≥ 0 :=
sorry

end f_monotone_on_0_to_2_find_range_a_part2_find_range_a_part3_l84_84139


namespace ZacharysBusRideLength_l84_84183

theorem ZacharysBusRideLength (vince_ride zach_ride : ℝ) 
  (h1 : vince_ride = 0.625) 
  (h2 : vince_ride = zach_ride + 0.125) : 
  zach_ride = 0.500 := 
by
  sorry

end ZacharysBusRideLength_l84_84183


namespace which_set_can_form_triangle_l84_84984

-- Definition of the triangle inequality theorem
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions for each set of line segments
def setA := (2, 6, 8)
def setB := (4, 6, 7)
def setC := (5, 6, 12)
def setD := (2, 3, 6)

-- Proof problem statement
theorem which_set_can_form_triangle : 
  triangle_inequality 2 6 8 = false ∧
  triangle_inequality 4 6 7 = true ∧
  triangle_inequality 5 6 12 = false ∧
  triangle_inequality 2 3 6 = false := 
by
  sorry -- Proof omitted

end which_set_can_form_triangle_l84_84984


namespace train_distance_900_l84_84118

theorem train_distance_900 (x t : ℝ) (H1 : x = 50 * t) (H2 : x - 100 = 40 * t) : 
  x + (x - 100) = 900 :=
by
  sorry

end train_distance_900_l84_84118


namespace calculate_expression_l84_84625

theorem calculate_expression : 
  (2^10 + (3^6 / 3^2)) = 1105 := 
by 
  -- Steps involve intermediate calculations
  -- for producing (2^10 = 1024), (3^6 = 729), (3^2 = 9)
  -- and then finding (729 / 9 = 81), (1024 + 81 = 1105)
  sorry

end calculate_expression_l84_84625


namespace value_of_f_at_1_l84_84195

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem value_of_f_at_1 : f 1 = 2 :=
by sorry

end value_of_f_at_1_l84_84195


namespace bisecting_line_eq_l84_84572

theorem bisecting_line_eq : ∃ (a : ℝ), (∀ x y : ℝ, (y = a * x) ↔ y = -1 / 6 * x) ∧ 
  (∀ p : ℝ × ℝ, (3 * p.1 - 5 * p.2  = 6 → p.2 = a * p.1) ∧ 
                  (4 * p.1 + p.2 + 6 = 0 → p.2 = a * p.1)) :=
by
  use -1 / 6
  sorry

end bisecting_line_eq_l84_84572


namespace schools_participating_l84_84116

noncomputable def num_schools (students_per_school : ℕ) (total_students : ℕ) : ℕ :=
  total_students / students_per_school

theorem schools_participating (students_per_school : ℕ) (beth_rank : ℕ) 
  (carla_rank : ℕ) (highest_on_team : ℕ) (n : ℕ) :
  students_per_school = 4 ∧ beth_rank = 46 ∧ carla_rank = 79 ∧
  (∀ i, i ≤ 46 → highest_on_team = 40) → 
  num_schools students_per_school ((2 * highest_on_team) - 1) = 19 := 
by
  intros h
  sorry

end schools_participating_l84_84116


namespace find_n_value_l84_84690

theorem find_n_value : ∃ n : ℤ, 3^3 - 7 = 4^2 + n ∧ n = 4 :=
by
  use 4
  sorry

end find_n_value_l84_84690


namespace masha_happy_max_l84_84548

/-- Masha has 2021 weights, all with unique masses. She places weights one at a 
time on a two-pan balance scale without removing previously placed weights. 
Every time the scale balances, Masha feels happy. Prove that the maximum number 
of times she can find the scales in perfect balance is 673. -/
theorem masha_happy_max (weights : Finset ℕ) (h_unique : weights.card = 2021) : 
  ∃ max_happy_times : ℕ, max_happy_times = 673 := 
sorry

end masha_happy_max_l84_84548


namespace initial_girls_count_l84_84012

variable (p : ℕ) -- total number of people initially in the group
variable (initial_girls : ℕ) -- number of girls initially

-- Condition 1: Initially, 50% of the group are girls
def initially_fifty_percent_girls (p : ℕ) (initial_girls : ℕ) : Prop := initial_girls = p / 2

-- Condition 2: Three girls leave and three boys arrive
def after_girls_leave_and_boys_arrive (initial_girls : ℕ) : ℕ := initial_girls - 3

-- Condition 3: After the change, 40% of the group are girls
def after_the_change_forty_percent_girls (p : ℕ) (initial_girls : ℕ) : Prop :=
  (after_girls_leave_and_boys_arrive initial_girls) = 2 * (p / 5)

theorem initial_girls_count (p : ℕ) (initial_girls : ℕ) :
  initially_fifty_percent_girls p initial_girls →
  after_the_change_forty_percent_girls p initial_girls →
  initial_girls = 15 := by
  sorry

end initial_girls_count_l84_84012


namespace inequalities_hold_l84_84181

theorem inequalities_hold 
  (x y z a b c : ℕ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)   -- Given that x, y, z are positive integers
  (ha : a > 0) (hb : b > 0) (hc : c > 0)   -- Given that a, b, c are positive integers
  (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 ≤ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ∧ 
  x^3 + y^3 + z^3 ≤ a^3 + b^3 + c^3 ∧ 
  x^2 * y * z + y^2 * z * x + z^2 * x * y ≤ a^2 * b * c + b^2 * c * a + c^2 * a * b :=
by
  sorry

end inequalities_hold_l84_84181


namespace walking_distance_l84_84485

-- Define the pace in miles per hour.
def pace : ℝ := 2

-- Define the duration in hours.
def duration : ℝ := 8

-- Define the total distance walked.
def total_distance (pace : ℝ) (duration : ℝ) : ℝ := pace * duration

-- Define the theorem we need to prove.
theorem walking_distance :
  total_distance pace duration = 16 := by
  sorry

end walking_distance_l84_84485


namespace total_pens_l84_84354

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l84_84354


namespace gdp_scientific_notation_l84_84849

theorem gdp_scientific_notation (trillion : ℕ) (five_year_growth : ℝ) (gdp : ℝ) :
  trillion = 10^12 ∧ 1 ≤ gdp / 10^14 ∧ gdp / 10^14 < 10 ∧ gdp = 121 * 10^12 → gdp = 1.21 * 10^14
:= by
  sorry

end gdp_scientific_notation_l84_84849


namespace students_in_class_l84_84438

theorem students_in_class (total_spent: ℝ) (packs_per_student: ℝ) (sausages_per_student: ℝ) (cost_pack_noodles: ℝ) (cost_sausage: ℝ) (cost_per_student: ℝ) (num_students: ℝ):
  total_spent = 290 → 
  packs_per_student = 2 → 
  sausages_per_student = 1 → 
  cost_pack_noodles = 3.5 → 
  cost_sausage = 7.5 → 
  cost_per_student = packs_per_student * cost_pack_noodles + sausages_per_student * cost_sausage →
  total_spent = cost_per_student * num_students →
  num_students = 20 := 
by
  sorry

end students_in_class_l84_84438


namespace capital_after_18_years_l84_84281

noncomputable def initial_investment : ℝ := 2000
def rate_of_increase : ℝ := 0.50
def period : ℕ := 3
def total_time : ℕ := 18

theorem capital_after_18_years :
  (initial_investment * (1 + rate_of_increase) ^ (total_time / period)) = 22781.25 :=
by
  sorry

end capital_after_18_years_l84_84281


namespace rectangle_area_l84_84009

theorem rectangle_area
  (b : ℝ)
  (l : ℝ)
  (P : ℝ)
  (h1 : l = 3 * b)
  (h2 : P = 2 * (l + b))
  (h3 : P = 112) :
  l * b = 588 := by
  sorry

end rectangle_area_l84_84009


namespace a_3_and_a_4_sum_l84_84474

theorem a_3_and_a_4_sum (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℚ) :
  (1 - (1 / (2 * x))) ^ 6 = a_0 + a_1 * (1 / x) + a_2 * (1 / x) ^ 2 + a_3 * (1 / x) ^ 3 + 
  a_4 * (1 / x) ^ 4 + a_5 * (1 / x) ^ 5 + a_6 * (1 / x) ^ 6 →
  a_3 + a_4 = -25 / 16 :=
sorry

end a_3_and_a_4_sum_l84_84474


namespace cube_painting_probability_l84_84971

theorem cube_painting_probability :
  let total_configurations := 2^6 * 2^6
  let identical_configurations := 90
  (identical_configurations / total_configurations : ℚ) = 45 / 2048 :=
by
  sorry

end cube_painting_probability_l84_84971


namespace avg_price_six_toys_l84_84362

def avg_price_five_toys : ℝ := 10
def price_sixth_toy : ℝ := 16
def total_toys : ℕ := 5 + 1

theorem avg_price_six_toys (avg_price_five_toys price_sixth_toy : ℝ) (total_toys : ℕ) :
  (avg_price_five_toys * 5 + price_sixth_toy) / total_toys = 11 := by
  sorry

end avg_price_six_toys_l84_84362


namespace annual_interest_rate_last_year_l84_84713

-- Define the conditions
def increased_by_ten_percent (r : ℝ) : ℝ := 1.10 * r

-- Statement of the problem
theorem annual_interest_rate_last_year (r : ℝ) (h : increased_by_ten_percent r = 0.11) : r = 0.10 :=
sorry

end annual_interest_rate_last_year_l84_84713


namespace orange_ratio_l84_84207

theorem orange_ratio (total_oranges alice_oranges : ℕ) (h_total : total_oranges = 180) (h_alice : alice_oranges = 120) :
  alice_oranges / (total_oranges - alice_oranges) = 2 :=
by
  sorry

end orange_ratio_l84_84207


namespace find_percentage_l84_84007

variable (P : ℝ)

def percentage_condition (P : ℝ) : Prop :=
  P * 30 = (0.25 * 16) + 2

theorem find_percentage : percentage_condition P → P = 0.2 :=
by
  intro h
  -- Proof steps go here
  sorry

end find_percentage_l84_84007


namespace correct_calculation_l84_84756

-- Definitions of the equations
def option_A (a : ℝ) : Prop := a + 2 * a = 3 * a^2
def option_B (a b : ℝ) : Prop := (a^2 * b)^3 = a^6 * b^3
def option_C (a : ℝ) (m : ℕ) : Prop := (a^m)^2 = a^(m+2)
def option_D (a : ℝ) : Prop := a^3 * a^2 = a^6

-- The theorem that states option B is correct and others are incorrect
theorem correct_calculation (a b : ℝ) (m : ℕ) : 
  ¬ option_A a ∧ 
  option_B a b ∧ 
  ¬ option_C a m ∧ 
  ¬ option_D a :=
by sorry

end correct_calculation_l84_84756


namespace problem_l84_84294

noncomputable def key_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : Real.sqrt (x * y) ≤ 1) 
    : Prop := ∃ z : ℝ, 0 < z ∧ z = 2 * (x + y) / (x + y + 2)^2

theorem problem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (habcd : a + b + c + d = 2) :
    (a^2 / (a^2 + 1)^2) + (b^2 / (b^2 + 1)^2) + (c^2 / (c^2 + 1)^2) + (d^2 / (d^2 + 1)^2) ≤ 16 / 25 := 
sorry

end problem_l84_84294


namespace simplify_expression_1_combine_terms_l84_84692

variable (a b : ℝ)

-- Problem 1: Simplification
theorem simplify_expression_1 : 2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b := by 
  sorry

-- Problem 2: Combine like terms
theorem combine_terms : 3 * a^2 * b + 2 * a * b^2 - 5 - 3 * a^2 * b - 5 * a * b^2 + 2 = -3 * a * b^2 - 3 := by 
  sorry

end simplify_expression_1_combine_terms_l84_84692


namespace six_digit_numbers_l84_84499

def isNonPerfectPower (n : ℕ) : Prop :=
  ∀ m k : ℕ, m ≥ 2 → k ≥ 2 → m^k ≠ n

theorem six_digit_numbers : ∃ x : ℕ, 
  100000 ≤ x ∧ x < 1000000 ∧ 
  (∃ a b c: ℕ, x = (a^3 * b)^2 ∧ isNonPerfectPower a ∧ isNonPerfectPower b ∧ isNonPerfectPower c ∧ 
    (∃ k : ℤ, k > 1 ∧ 
      (x: ℤ) / (k^3 : ℤ) < 1 ∧ 
      ∃ num denom: ℕ, num < denom ∧ 
      num = n^3 ∧ denom = d^2 ∧ 
      isNonPerfectPower n ∧ isNonPerfectPower d)) := 
sorry

end six_digit_numbers_l84_84499


namespace debut_show_tickets_l84_84489

variable (P : ℕ) -- Number of people who bought tickets for the debut show

-- Conditions
def three_times_more (P : ℕ) : Bool := (3 * P = P + 2 * P)
def ticket_cost : ℕ := 25
def total_revenue (P : ℕ) : ℕ := 4 * P * ticket_cost

-- Main statement
theorem debut_show_tickets (h1 : three_times_more P = true) 
                           (h2 : total_revenue P = 20000) : P = 200 :=
by
  sorry

end debut_show_tickets_l84_84489


namespace binary_to_octal_equivalence_l84_84559

theorem binary_to_octal_equivalence : (1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) 
                                    = (1 * 8^2 + 1 * 8^1 + 5 * 8^0) :=
by sorry

end binary_to_octal_equivalence_l84_84559


namespace train_length_l84_84959

theorem train_length (L : ℝ) (h1 : (L + 120) / 60 = L / 20) : L = 60 := 
sorry

end train_length_l84_84959


namespace triangle_area_from_squares_l84_84865

noncomputable def area_of_triangle (S1 S2 : ℝ) : ℝ :=
  let side1 := Real.sqrt S1
  let side2 := Real.sqrt S2
  0.5 * side1 * side2

theorem triangle_area_from_squares
  (A1 A2 : ℝ)
  (h1 : A1 = 196)
  (h2 : A2 = 100) :
  area_of_triangle A1 A2 = 70 :=
by
  rw [h1, h2]
  unfold area_of_triangle
  rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow]
  norm_num
  sorry

end triangle_area_from_squares_l84_84865


namespace rate_calculation_l84_84096

noncomputable def rate_per_sq_meter
  (lawn_length : ℝ) (lawn_breadth : ℝ)
  (road_width : ℝ) (total_cost : ℝ) : ℝ :=
  let area_road_1 := road_width * lawn_breadth
  let area_road_2 := road_width * lawn_length
  let area_intersection := road_width * road_width
  let total_area_roads := (area_road_1 + area_road_2) - area_intersection
  total_cost / total_area_roads

theorem rate_calculation :
  rate_per_sq_meter 100 60 10 4500 = 3 := by
  sorry

end rate_calculation_l84_84096


namespace find_p_l84_84190

variable (a b p q r1 r2 : ℝ)

-- Given conditions
def roots_eq1 (h_1 : r1 + r2 = -a) (h_2 : r1 * r2 = b) : Prop :=
  -- Using Vieta's Formulas on x^2 + ax + b = 0
  ∀ (r1 r2 : ℝ), r1 + r2 = -a ∧ r1 * r2 = b

def roots_eq2 (r1 r2 : ℝ) (h_3 : r1^2 + r2^2 = -p) (h_4 : r1^2 * r2^2 = q) : Prop :=
  -- Using Vieta's Formulas on x^2 + px + q = 0
  ∀ (r1 r2 : ℝ), r1^2 + r2^2 = -p ∧ r1^2 * r2^2 = q

-- Theorems
theorem find_p (h_1 : r1 + r2 = -a) (h_2 : r1 * r2 = b) (h_3 : r1^2 + r2^2 = -p) :
  p = -a^2 + 2*b := by
  sorry

end find_p_l84_84190


namespace find_w_l84_84484

theorem find_w (k : ℝ) (h1 : z * Real.sqrt w = k)
  (z_w3 : z = 6) (w3 : w = 3) :
  z = 3 / 2 → w = 48 := sorry

end find_w_l84_84484


namespace alice_paid_percentage_of_srp_l84_84452

theorem alice_paid_percentage_of_srp
  (P : ℝ) -- Suggested Retail Price (SRP)
  (MP : ℝ := P * 0.60) -- Marked Price (MP) is 40% less than SRP
  (price_alice_paid : ℝ := MP * 0.60) -- Alice purchased the book for 40% off the marked price
  : (price_alice_paid / P) * 100 = 36 :=
by
  -- only the statement is required, so proof is omitted
  sorry

end alice_paid_percentage_of_srp_l84_84452


namespace find_natural_triples_l84_84522

theorem find_natural_triples (x y z : ℕ) : 
  (x+1) * (y+1) * (z+1) = 3 * x * y * z ↔ 
  (x, y, z) = (2, 2, 3) ∨ (x, y, z) = (2, 3, 2) ∨ (x, y, z) = (3, 2, 2) ∨
  (x, y, z) = (5, 1, 4) ∨ (x, y, z) = (5, 4, 1) ∨ (x, y, z) = (4, 1, 5) ∨ (x, y, z) = (4, 5, 1) ∨ 
  (x, y, z) = (1, 4, 5) ∨ (x, y, z) = (1, 5, 4) ∨ (x, y, z) = (8, 1, 3) ∨ (x, y, z) = (8, 3, 1) ∨
  (x, y, z) = (3, 1, 8) ∨ (x, y, z) = (3, 8, 1) ∨ (x, y, z) = (1, 3, 8) ∨ (x, y, z) = (1, 8, 3) :=
by {
  sorry
}

end find_natural_triples_l84_84522


namespace percent_decrease_in_cost_l84_84562

theorem percent_decrease_in_cost (cost_1990 cost_2010 : ℕ) (h1 : cost_1990 = 35) (h2 : cost_2010 = 5) : 
  ((cost_1990 - cost_2010) * 100 / cost_1990 : ℚ) = 86 := 
by
  sorry

end percent_decrease_in_cost_l84_84562


namespace probability_of_ram_l84_84614

theorem probability_of_ram 
  (P_ravi : ℝ) (P_both : ℝ) 
  (h_ravi : P_ravi = 1 / 5) 
  (h_both : P_both = 0.11428571428571428) : 
  ∃ P_ram : ℝ, P_ram = 0.5714285714285714 :=
by
  sorry

end probability_of_ram_l84_84614


namespace moles_of_CH4_l84_84880

theorem moles_of_CH4 (moles_Be2C moles_H2O : ℕ) (balanced_equation : 1 * Be2C + 4 * H2O = 2 * CH4 + 2 * BeOH2) 
  (h_Be2C : moles_Be2C = 3) (h_H2O : moles_H2O = 12) : 
  6 = 2 * moles_Be2C :=
by
  sorry

end moles_of_CH4_l84_84880


namespace solution_l84_84968

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (4 + 2 * x) / (6 + 3 * x) = (3 + 2 * x) / (5 + 3 * x) ∧ x = -2

theorem solution : problem_statement :=
by
  sorry

end solution_l84_84968


namespace count_consecutive_sequences_l84_84783

def consecutive_sequences (n : ℕ) : ℕ :=
  if n = 15 then 270 else 0

theorem count_consecutive_sequences : consecutive_sequences 15 = 270 :=
by
  sorry

end count_consecutive_sequences_l84_84783


namespace decrypt_message_base7_l84_84582

noncomputable def base7_to_base10 : Nat := 
  2 * 343 + 5 * 49 + 3 * 7 + 4 * 1

theorem decrypt_message_base7 : base7_to_base10 = 956 := 
by 
  sorry

end decrypt_message_base7_l84_84582


namespace derivative_y_l84_84389

noncomputable def u (x : ℝ) := 4 * x - 1 + Real.sqrt (16 * x ^ 2 - 8 * x + 2)
noncomputable def v (x : ℝ) := Real.sqrt (16 * x ^ 2 - 8 * x + 2) * Real.arctan (4 * x - 1)

noncomputable def y (x : ℝ) := Real.log (u x) - v x

theorem derivative_y (x : ℝ) :
  deriv y x = (4 * (1 - 4 * x)) / (Real.sqrt (16 * x ^ 2 - 8 * x + 2)) * Real.arctan (4 * x - 1) :=
by
  sorry

end derivative_y_l84_84389


namespace factorial_simplification_l84_84721

theorem factorial_simplification :
  Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 728 := 
sorry

end factorial_simplification_l84_84721


namespace digits_count_concatenated_l84_84218

-- Define the conditions for the digit count of 2^n and 5^n
def digits_count_2n (n p : ℕ) : Prop := 10^(p-1) ≤ 2^n ∧ 2^n < 10^p
def digits_count_5n (n q : ℕ) : Prop := 10^(q-1) ≤ 5^n ∧ 5^n < 10^q

-- The main theorem to prove the number of digits when 2^n and 5^n are concatenated
theorem digits_count_concatenated (n p q : ℕ) 
  (h1 : digits_count_2n n p) 
  (h2 : digits_count_5n n q): 
  p + q = n + 1 := by 
  sorry

end digits_count_concatenated_l84_84218


namespace n_squared_divisible_by_144_l84_84830

theorem n_squared_divisible_by_144 (n : ℕ) (h1 : 0 < n) (h2 : ∃ t : ℕ, t = 12 ∧ ∀ d : ℕ, d ∣ n → d ≤ t) : 144 ∣ n^2 :=
sorry

end n_squared_divisible_by_144_l84_84830


namespace customers_tried_sample_l84_84010

theorem customers_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left_over : ℕ)
  (samples_per_customer : ℕ := 1)
  (h_samples_per_box : samples_per_box = 20)
  (h_boxes_opened : boxes_opened = 12)
  (h_samples_left_over : samples_left_over = 5) :
  (samples_per_box * boxes_opened - samples_left_over) / samples_per_customer = 235 :=
by
  sorry

end customers_tried_sample_l84_84010


namespace price_returns_to_initial_l84_84724

theorem price_returns_to_initial {P₀ P₁ P₂ P₃ P₄ : ℝ} (y : ℝ) (h₁ : P₀ = 100)
  (h₂ : P₁ = P₀ * 1.30) (h₃ : P₂ = P₁ * 0.70) (h₄ : P₃ = P₂ * 1.40) 
  (h₅ : P₄ = P₃ * (1 - y / 100)) : P₄ = P₀ → y = 22 :=
by
  sorry

end price_returns_to_initial_l84_84724


namespace least_days_to_repay_twice_l84_84156

-- Define the initial conditions
def borrowed_amount : ℝ := 15
def daily_interest_rate : ℝ := 0.10
def interest_per_day : ℝ := borrowed_amount * daily_interest_rate
def total_amount_to_repay : ℝ := 2 * borrowed_amount

-- Define the condition we want to prove
theorem least_days_to_repay_twice : ∃ (x : ℕ), (borrowed_amount + interest_per_day * x) ≥ total_amount_to_repay ∧ x = 10 :=
by
  sorry

end least_days_to_repay_twice_l84_84156


namespace problem_statement_l84_84591

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * sqrt 3 * cos (ω * x + π / 6)

theorem problem_statement (ω : ℝ) (hx : ω = 2 ∨ ω = -2) :
  f ω (π / 3) = -3 ∨ f ω (π / 3) = 0 := by
  unfold f
  cases hx with
  | inl w_eq => sorry
  | inr w_eq => sorry

end problem_statement_l84_84591


namespace Tim_change_l84_84212

theorem Tim_change (initial_amount paid_amount : ℕ) (h₀ : initial_amount = 50) (h₁ : paid_amount = 45) : initial_amount - paid_amount = 5 :=
by
  sorry

end Tim_change_l84_84212


namespace find_p_l84_84848

theorem find_p (p : ℤ)
  (h1 : ∀ (u v : ℤ), u > 0 → v > 0 → 5 * u ^ 2 - 5 * p * u + (66 * p - 1) = 0 ∧
    5 * v ^ 2 - 5 * p * v + (66 * p - 1) = 0) :
  p = 76 :=
sorry

end find_p_l84_84848


namespace tan_alpha_minus_pi_over_4_eq_neg_3_over_4_l84_84237

open Real

theorem tan_alpha_minus_pi_over_4_eq_neg_3_over_4 (α β : ℝ) 
  (h1 : tan (α + β) = 1 / 2) 
  (h2 : tan β = 1 / 3) : 
  tan (α - π / 4) = -3 / 4 :=
sorry

end tan_alpha_minus_pi_over_4_eq_neg_3_over_4_l84_84237


namespace find_number_l84_84579

theorem find_number (x : ℕ) (h : x * 9999 = 724817410) : x = 72492 :=
sorry

end find_number_l84_84579


namespace greatest_possible_remainder_l84_84842

theorem greatest_possible_remainder (x : ℕ) (h: x % 7 ≠ 0) : (∃ r < 7, r = x % 7) ∧ x % 7 ≤ 6 := by
  sorry

end greatest_possible_remainder_l84_84842


namespace isosceles_right_triangle_area_l84_84824

theorem isosceles_right_triangle_area (a : ℝ) (h : ℝ) (p : ℝ) 
  (h_triangle : h = a * Real.sqrt 2) 
  (hypotenuse_is_16 : h = 16) :
  (1 / 2) * a * a = 64 := 
by
  -- Skip the proof as per guidelines
  sorry

end isosceles_right_triangle_area_l84_84824


namespace fraction_of_capacity_l84_84883

theorem fraction_of_capacity
    (bus_capacity : ℕ)
    (x : ℕ)
    (first_pickup : ℕ)
    (second_pickup : ℕ)
    (unable_to_board : ℕ)
    (bus_full : bus_capacity = x + (second_pickup - unable_to_board))
    (carry_fraction : x / bus_capacity = 3 / 5) : 
    true := 
sorry

end fraction_of_capacity_l84_84883


namespace find_r_and_k_l84_84236

-- Define the line equation
def line (x : ℝ) : ℝ := 5 * x - 7

-- Define the parameterization
def param (t r k : ℝ) : ℝ × ℝ := 
  (r + 3 * t, 2 + k * t)

-- Theorem stating that (r, k) = (9/5, 15) satisfies the given conditions
theorem find_r_and_k 
  (r k : ℝ)
  (H1 : param 0 r k = (r, 2))
  (H2 : line r = 2)
  (H3 : param 1 r k = (r + 3, 2 + k))
  (H4 : line (r + 3) = 2 + k)
  : (r, k) = (9/5, 15) :=
sorry

end find_r_and_k_l84_84236


namespace roots_inequality_l84_84553

theorem roots_inequality (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) :
  -1 ≤ z ∧ z ≤ 13 / 3 :=
sorry

end roots_inequality_l84_84553


namespace probability_of_second_ball_red_is_correct_probabilities_of_winning_prizes_distribution_and_expectation_of_X_l84_84282

-- Definitions for balls and initial conditions
def totalBalls : ℕ := 10
def redBalls : ℕ := 2
def whiteBalls : ℕ := 3
def yellowBalls : ℕ := 5

-- Drawing without replacement
noncomputable def probability_second_ball_red : ℚ :=
  (2/10) * (1/9) + (8/10) * (2/9)

-- Probabilities for each case
noncomputable def probability_first_prize : ℚ := 
  (redBalls.choose 1 * whiteBalls.choose 1) / (totalBalls.choose 2)

noncomputable def probability_second_prize : ℚ := 
  (redBalls.choose 2) / (totalBalls.choose 2)

noncomputable def probability_third_prize : ℚ := 
  (whiteBalls.choose 2) / (totalBalls.choose 2)

-- Probability of at least one yellow ball (no prize)
noncomputable def probability_no_prize : ℚ := 
  1 - probability_first_prize - probability_second_prize - probability_third_prize

-- Probability distribution and expectation for number of winners X
noncomputable def winning_probability : ℚ := probability_first_prize + probability_second_prize + probability_third_prize

noncomputable def P_X (n : ℕ) : ℚ :=
  if n = 0 then (7/9)^3
  else if n = 1 then 3 * (2/9) * (7/9)^2
  else if n = 2 then 3 * (2/9)^2 * (7/9)
  else if n = 3 then (2/9)^3
  else 0

noncomputable def expectation_X : ℚ := 
  3 * winning_probability

-- Lean statements
theorem probability_of_second_ball_red_is_correct :
  probability_second_ball_red = 1 / 5 := by
  sorry

theorem probabilities_of_winning_prizes :
  probability_first_prize = 2 / 15 ∧
  probability_second_prize = 1 / 45 ∧
  probability_third_prize = 1 / 15 := by
  sorry

theorem distribution_and_expectation_of_X :
  P_X 0 = 343 / 729 ∧
  P_X 1 = 294 / 729 ∧
  P_X 2 = 84 / 729 ∧
  P_X 3 = 8 / 729 ∧
  expectation_X = 2 / 3 := by
  sorry

end probability_of_second_ball_red_is_correct_probabilities_of_winning_prizes_distribution_and_expectation_of_X_l84_84282


namespace find_solutions_l84_84385

def satisfies_inequality (x : ℝ) : Prop :=
  (Real.cos x)^2018 + (1 / (Real.sin x))^2019 ≤ (Real.sin x)^2018 + (1 / (Real.cos x))^2019

def in_intervals (x : ℝ) : Prop :=
  (x ∈ Set.Ico (-Real.pi / 3) 0) ∨
  (x ∈ Set.Ico (Real.pi / 4) (Real.pi / 2)) ∨
  (x ∈ Set.Ioc Real.pi (5 * Real.pi / 4)) ∨
  (x ∈ Set.Ioc (3 * Real.pi / 2) (5 * Real.pi / 3))

theorem find_solutions :
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi / 3) (5 * Real.pi / 3) →
  satisfies_inequality x ↔ in_intervals x := 
  by sorry

end find_solutions_l84_84385


namespace robin_camera_pictures_l84_84505

-- Given conditions
def pictures_from_phone : Nat := 35
def num_albums : Nat := 5
def pics_per_album : Nat := 8

-- Calculate total pictures and the number of pictures from the camera
theorem robin_camera_pictures : num_albums * pics_per_album - pictures_from_phone = 5 := by
  sorry

end robin_camera_pictures_l84_84505


namespace neg_p_iff_neg_q_l84_84885

theorem neg_p_iff_neg_q (a : ℝ) : (¬ (a < 0)) ↔ (¬ (a^2 > a)) :=
by 
    sorry

end neg_p_iff_neg_q_l84_84885


namespace incorrect_statement_l84_84546

-- Define the general rules of program flowcharts
def isValidStart (box : String) : Prop := box = "start"
def isValidEnd (box : String) : Prop := box = "end"
def isInputBox (box : String) : Prop := box = "input"
def isOutputBox (box : String) : Prop := box = "output"

-- Define the statement to be proved incorrect
def statement (boxes : List String) : Prop :=
  ∀ xs ys, boxes = xs ++ ["start", "input"] ++ ys ->
           ∀ zs ws, boxes = zs ++ ["output", "end"] ++ ws

-- The target theorem stating that the statement is incorrect
theorem incorrect_statement (boxes : List String) :
  ¬ statement boxes :=
sorry

end incorrect_statement_l84_84546


namespace max_pens_given_budget_l84_84240

-- Define the conditions.
def max_pens (x y : ℕ) := 12 * x + 20 * y

-- Define the main theorem stating the proof problem.
theorem max_pens_given_budget : ∃ (x y : ℕ), (10 * x + 15 * y ≤ 173) ∧ (max_pens x y = 224) :=
  sorry

end max_pens_given_budget_l84_84240
