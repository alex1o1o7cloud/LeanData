import Mathlib

namespace central_angle_of_sector_l1736_173679

/-- The central angle of the sector obtained by unfolding the lateral surface of a cone with
    base radius 1 and slant height 2 is \(\pi\). -/
theorem central_angle_of_sector (r_base : ℝ) (r_slant : ℝ) (α : ℝ)
  (h1 : r_base = 1) (h2 : r_slant = 2) (h3 : 2 * π = α * r_slant) : α = π :=
by
  sorry

end central_angle_of_sector_l1736_173679


namespace find_f_neg2_l1736_173663

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2^x + 3*x - 1 else -(2^(-x) + 3*(-x) - 1)

theorem find_f_neg2 : f (-2) = -9 :=
by sorry

end find_f_neg2_l1736_173663


namespace vector_operation_l1736_173651

open Matrix

def u : Matrix (Fin 2) (Fin 1) ℝ := ![![3], ![-6]]
def v : Matrix (Fin 2) (Fin 1) ℝ := ![![1], ![-9]]
def w : Matrix (Fin 2) (Fin 1) ℝ := ![![-1], ![4]]

--\mathbf{u} - 5\mathbf{v} + \mathbf{w} = \begin{pmatrix} = \begin{pmatrix} -3 \\ 43 \end{pmatrix}
theorem vector_operation : u - (5 : ℝ) • v + w = ![![-3], ![43]] :=
by
  sorry

end vector_operation_l1736_173651


namespace sum_and_gap_l1736_173691

-- Define the gap condition
def gap_condition (x : ℝ) : Prop :=
  |5.46 - x| = 3.97

-- Define the main theorem to be proved 
theorem sum_and_gap :
  ∀ (x : ℝ), gap_condition x → x < 5.46 → x + 5.46 = 6.95 := 
by 
  intros x hx hlt
  sorry

end sum_and_gap_l1736_173691


namespace train_length_proof_l1736_173654

-- Definitions based on the conditions given in the problem
def speed_km_per_hr := 45 -- speed of the train in km/hr
def time_seconds := 60 -- time taken to pass the platform in seconds
def length_platform_m := 390 -- length of the platform in meters

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed : ℕ) : ℕ := (speed * 1000) / 3600

-- Calculate the speed in m/s
def speed_m_per_s : ℕ := km_per_hr_to_m_per_s speed_km_per_hr

-- Calculate the total distance covered by the train while passing the platform
def total_distance_m : ℕ := speed_m_per_s * time_seconds

-- Total distance is the sum of the length of the train and the length of the platform
def length_train_m := total_distance_m - length_platform_m

-- The statement to prove the length of the train
theorem train_length_proof : length_train_m = 360 :=
by
  sorry

end train_length_proof_l1736_173654


namespace range_of_a_l1736_173682

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → (x < a ∨ x > a + 4)) ∧ ¬(∀ x : ℝ, (x < a ∨ x > a + 4) → -2 ≤ x ∧ x ≤ 1) ↔
  a > 1 ∨ a < -6 :=
by {
  sorry
}

end range_of_a_l1736_173682


namespace lcm_of_numbers_l1736_173606

theorem lcm_of_numbers (a b lcm hcf : ℕ) (h_prod : a * b = 45276) (h_hcf : hcf = 22) (h_relation : a * b = hcf * lcm) : lcm = 2058 :=
by sorry

end lcm_of_numbers_l1736_173606


namespace speed_of_man_in_still_water_l1736_173623

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : 5 * (v_m + v_s) = 45) (h2 : 5 * (v_m - v_s) = 25) : v_m = 7 :=
by
  sorry

end speed_of_man_in_still_water_l1736_173623


namespace derivative_at_2_l1736_173656

noncomputable def f (x : ℝ) : ℝ := x

theorem derivative_at_2 : (deriv f 2) = 1 :=
by
  -- sorry, proof not included
  sorry

end derivative_at_2_l1736_173656


namespace trapezoid_area_condition_l1736_173642

theorem trapezoid_area_condition
  (a x y z : ℝ)
  (h_sq  : ∀ (ABCD : ℝ), ABCD = a * a)
  (h_trap: ∀ (EBCF : ℝ), EBCF = x * a)
  (h_rec : ∀ (JKHG : ℝ), JKHG = y * z)
  (h_sum : y + z = a)
  (h_area : x * a = a * a - 2 * y * z) :
  x = a / 2 :=
by
  sorry

end trapezoid_area_condition_l1736_173642


namespace large_box_total_chocolate_bars_l1736_173649

def number_of_small_boxes : ℕ := 15
def chocolate_bars_per_small_box : ℕ := 20
def total_chocolate_bars (n : ℕ) (m : ℕ) : ℕ := n * m

theorem large_box_total_chocolate_bars :
  total_chocolate_bars number_of_small_boxes chocolate_bars_per_small_box = 300 :=
by
  sorry

end large_box_total_chocolate_bars_l1736_173649


namespace hexagonal_prism_min_cut_l1736_173685

-- We formulate the problem conditions and the desired proof
def minimum_edges_to_cut (total_edges : ℕ) (uncut_edges : ℕ) : ℕ :=
  total_edges - uncut_edges

theorem hexagonal_prism_min_cut :
  minimum_edges_to_cut 18 7 = 11 :=
by
  sorry

end hexagonal_prism_min_cut_l1736_173685


namespace correct_value_of_3_dollar_neg4_l1736_173695

def special_operation (x y : Int) : Int :=
  x * (y + 2) + x * y + x

theorem correct_value_of_3_dollar_neg4 : special_operation 3 (-4) = -15 :=
by
  sorry

end correct_value_of_3_dollar_neg4_l1736_173695


namespace larger_square_side_length_l1736_173676

theorem larger_square_side_length (x y H : ℝ) 
  (smaller_square_perimeter : 4 * x = H - 20)
  (larger_square_perimeter : 4 * y = H) :
  y = x + 5 :=
by
  sorry

end larger_square_side_length_l1736_173676


namespace rooms_in_second_wing_each_hall_l1736_173673

theorem rooms_in_second_wing_each_hall
  (floors_first_wing : ℕ)
  (halls_per_floor_first_wing : ℕ)
  (rooms_per_hall_first_wing : ℕ)
  (floors_second_wing : ℕ)
  (halls_per_floor_second_wing : ℕ)
  (total_rooms : ℕ)
  (h1 : floors_first_wing = 9)
  (h2 : halls_per_floor_first_wing = 6)
  (h3 : rooms_per_hall_first_wing = 32)
  (h4 : floors_second_wing = 7)
  (h5 : halls_per_floor_second_wing = 9)
  (h6 : total_rooms = 4248) :
  (total_rooms - floors_first_wing * halls_per_floor_first_wing * rooms_per_hall_first_wing) / 
  (floors_second_wing * halls_per_floor_second_wing) = 40 :=
  by {
  sorry
}

end rooms_in_second_wing_each_hall_l1736_173673


namespace cost_comparison_l1736_173675

def cost_function_A (x : ℕ) : ℕ := 450 * x + 1000
def cost_function_B (x : ℕ) : ℕ := 500 * x

theorem cost_comparison (x : ℕ) : 
  if x = 20 then cost_function_A x = cost_function_B x 
  else if x < 20 then cost_function_A x > cost_function_B x 
  else cost_function_A x < cost_function_B x :=
sorry

end cost_comparison_l1736_173675


namespace hair_cut_second_day_l1736_173692

variable (hair_first_day : ℝ) (total_hair_cut : ℝ)

theorem hair_cut_second_day (h1 : hair_first_day = 0.375) (h2 : total_hair_cut = 0.875) :
  total_hair_cut - hair_first_day = 0.500 :=
by sorry

end hair_cut_second_day_l1736_173692


namespace temperature_representation_l1736_173693

theorem temperature_representation (a : ℤ) (b : ℤ) (h1 : a = 8) (h2 : b = -5) :
    b < 0 → b = -5 :=
by
  sorry

end temperature_representation_l1736_173693


namespace steve_travel_time_l1736_173677

noncomputable def total_travel_time (distance: ℕ) (speed_to_work: ℕ) (speed_back: ℕ) : ℕ :=
  (distance / speed_to_work) + (distance / speed_back)

theorem steve_travel_time : 
  ∀ (distance speed_back speed_to_work : ℕ), 
  (speed_to_work = speed_back / 2) → 
  speed_back = 15 → 
  distance = 30 → 
  total_travel_time distance speed_to_work speed_back = 6 := 
by
  intros
  rw [total_travel_time]
  sorry

end steve_travel_time_l1736_173677


namespace ratio_of_pieces_l1736_173686

theorem ratio_of_pieces (total_length : ℝ) (short_piece : ℝ) (total_length_eq : total_length = 70) (short_piece_eq : short_piece = 27.999999999999993) :
  let long_piece := total_length - short_piece
  let ratio := short_piece / long_piece
  ratio = 2 / 3 :=
by
  sorry

end ratio_of_pieces_l1736_173686


namespace find_numerator_l1736_173629

theorem find_numerator (n : ℕ) : 
  (n : ℚ) / 22 = 9545 / 10000 → 
  n = 9545 * 22 / 10000 :=
by sorry

end find_numerator_l1736_173629


namespace xyz_problem_l1736_173613

/-- Given x = 36^2 + 48^2 + 64^3 + 81^2, prove the following:
    - x is a multiple of 3. 
    - x is a multiple of 4.
    - x is a multiple of 9.
    - x is not a multiple of 16. 
-/
theorem xyz_problem (x : ℕ) (h_x : x = 36^2 + 48^2 + 64^3 + 81^2) :
  (x % 3 = 0) ∧ (x % 4 = 0) ∧ (x % 9 = 0) ∧ ¬(x % 16 = 0) := 
by
  have h1 : 36^2 = 1296 := by norm_num
  have h2 : 48^2 = 2304 := by norm_num
  have h3 : 64^3 = 262144 := by norm_num
  have h4 : 81^2 = 6561 := by norm_num
  have hx : x = 1296 + 2304 + 262144 + 6561 := by rw [h_x, h1, h2, h3, h4]
  sorry

end xyz_problem_l1736_173613


namespace equation_squares_l1736_173619

theorem equation_squares (a b c : ℤ) (h : (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = a ^ 2 + b ^ 2 - c ^ 2) :
  ∃ k1 k2 : ℤ, (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = k1 ^ 2 ∧ a ^ 2 + b ^ 2 - c ^ 2 = k2 ^ 2 :=
by
  sorry

end equation_squares_l1736_173619


namespace problem_l1736_173690

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f (-x) = f x)  -- f is an even function
variable (h_mono : ∀ x y : ℝ, 0 < x → x < y → f y < f x)  -- f is monotonically decreasing on (0, +∞)

theorem problem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end problem_l1736_173690


namespace geometric_sequence_sum_l1736_173645

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1 / 4) :
  a 1 * a 2 + a 2 * a 3 + a 3 * a 4 + a 4 * a 5 + a 5 * a 6 = 341 / 32 :=
by sorry

end geometric_sequence_sum_l1736_173645


namespace max_value_a_l1736_173616

theorem max_value_a (a : ℝ) : (∀ x : ℝ, |x - 2| + |x - 8| ≥ a) ↔ a ≤ 6 := by
  sorry

end max_value_a_l1736_173616


namespace maximize_hotel_profit_l1736_173640

theorem maximize_hotel_profit :
  let rooms := 50
  let base_price := 180
  let increase_per_vacancy := 10
  let maintenance_cost := 20
  ∃ (x : ℕ), ((base_price + increase_per_vacancy * x) * (rooms - x) 
    - maintenance_cost * (rooms - x) = 10890) ∧ (base_price + increase_per_vacancy * x = 350) :=
by
  sorry

end maximize_hotel_profit_l1736_173640


namespace least_value_a_plus_b_l1736_173601

theorem least_value_a_plus_b (a b : ℕ) (h : 20 / 19 = 1 + 1 / (1 + a / b)) : a + b = 19 :=
sorry

end least_value_a_plus_b_l1736_173601


namespace range_of_x_l1736_173658

noncomputable def A (x : ℝ) : ℤ := Int.ceil x

theorem range_of_x (x : ℝ) (h₁ : x > 0) (h₂ : A (2 * x * A x) = 5) : x ∈ Set.Ioc 1 (5 / 4 : ℝ) :=
sorry

end range_of_x_l1736_173658


namespace arithmetic_sequence_problem_l1736_173699

theorem arithmetic_sequence_problem
  (a : ℕ → ℚ)
  (h : a 2 + a 4 + a 9 + a 11 = 32) :
  a 6 + a 7 = 16 :=
sorry

end arithmetic_sequence_problem_l1736_173699


namespace problem_solution_l1736_173605

theorem problem_solution {a b : ℝ} (h : a * b + b^2 = 12) : (a + b)^2 - (a + b) * (a - b) = 24 :=
by sorry

end problem_solution_l1736_173605


namespace smallest_next_smallest_sum_l1736_173625

-- Defining the set of numbers as constants
def nums : Set ℕ := {10, 11, 12, 13}

-- Define the smallest number in the set
def smallest : ℕ := 10

-- Define the next smallest number in the set
def next_smallest : ℕ := 11

-- The main theorem statement
theorem smallest_next_smallest_sum : smallest + next_smallest = 21 :=
by 
  sorry

end smallest_next_smallest_sum_l1736_173625


namespace gabriel_forgot_days_l1736_173621

def days_in_july : ℕ := 31
def days_taken : ℕ := 28

theorem gabriel_forgot_days : days_in_july - days_taken = 3 := by
  sorry

end gabriel_forgot_days_l1736_173621


namespace percentage_of_motorists_speeding_l1736_173630

-- Definitions based on the conditions
def total_motorists : Nat := 100
def percent_motorists_receive_tickets : Real := 0.20
def percent_speeders_no_tickets : Real := 0.20

-- Define the variables for the number of speeders
variable (x : Real) -- the percentage of total motorists who speed 

-- Lean statement to formalize the problem
theorem percentage_of_motorists_speeding 
  (h1 : 20 = (0.80 * x) * (total_motorists / 100)) : 
  x = 25 :=
sorry

end percentage_of_motorists_speeding_l1736_173630


namespace max_red_socks_l1736_173638

theorem max_red_socks (x y : ℕ) 
  (h1 : x + y ≤ 2017) 
  (h2 : (x * (x - 1) + y * (y - 1)) = (x + y) * (x + y - 1) / 2) : 
  x ≤ 990 := 
sorry

end max_red_socks_l1736_173638


namespace twelve_sided_die_expected_value_l1736_173678

theorem twelve_sided_die_expected_value : 
  ∃ (E : ℝ), (E = 6.5) :=
by
  let n := 12
  let numerator := n * (n + 1) / 2
  let expected_value := numerator / n
  use expected_value
  sorry

end twelve_sided_die_expected_value_l1736_173678


namespace total_eggs_found_l1736_173662

def eggs_from_club_house : ℕ := 40
def eggs_from_park : ℕ := 25
def eggs_from_town_hall : ℕ := 15

theorem total_eggs_found : eggs_from_club_house + eggs_from_park + eggs_from_town_hall = 80 := by
  -- Proof of this theorem
  sorry

end total_eggs_found_l1736_173662


namespace sum_of_21st_set_l1736_173600

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

def first_element_of_set (n : ℕ) : ℕ := triangular_number n - n + 1

def sum_of_elements_in_set (n : ℕ) : ℕ := 
  n * ((first_element_of_set n + triangular_number n) / 2)

theorem sum_of_21st_set : sum_of_elements_in_set 21 = 4641 := by 
  sorry

end sum_of_21st_set_l1736_173600


namespace product_of_numbers_l1736_173633

theorem product_of_numbers (x y : ℝ) 
  (h₁ : x + y = 8 * (x - y)) 
  (h₂ : x * y = 40 * (x - y)) : x * y = 4032 := 
by
  sorry

end product_of_numbers_l1736_173633


namespace intersection_of_M_and_N_l1736_173608

def M : Set ℤ := { x | -3 < x ∧ x < 3 }
def N : Set ℤ := { x | x < 1 }

theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := by
  sorry

end intersection_of_M_and_N_l1736_173608


namespace karen_has_32_quarters_l1736_173659

variable (k : ℕ)  -- the number of quarters Karen has

-- Define the number of quarters Christopher has
def christopher_quarters : ℕ := 64

-- Define the value of a single quarter in dollars
def quarter_value : ℚ := 0.25

-- Define the amount of money Christopher has
def christopher_money : ℚ := christopher_quarters * quarter_value

-- Define the monetary difference between Christopher and Karen
def money_difference : ℚ := 8

-- Define the amount of money Karen has
def karen_money : ℚ := christopher_money - money_difference

-- Define the number of quarters Karen has
def karen_quarters := karen_money / quarter_value

-- The theorem we need to prove
theorem karen_has_32_quarters : k = 32 :=
by
  sorry

end karen_has_32_quarters_l1736_173659


namespace circle_equation_tangent_x_axis_l1736_173653

theorem circle_equation_tangent_x_axis (x y : ℝ) (center : ℝ × ℝ) (r : ℝ) 
  (h_center : center = (-1, 2)) 
  (h_tangent : r = |2 - 0|) :
  (x + 1)^2 + (y - 2)^2 = 4 := 
sorry

end circle_equation_tangent_x_axis_l1736_173653


namespace tim_change_l1736_173670

theorem tim_change :
  ∀ (initial_amount : ℕ) (amount_paid : ℕ),
  initial_amount = 50 →
  amount_paid = 45 →
  initial_amount - amount_paid = 5 :=
by
  intros
  sorry

end tim_change_l1736_173670


namespace function_value_l1736_173672

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem function_value (a b : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log_base a (2 + b) = 1) (h₃ : log_base a (8 + b) = 2) : a + b = 4 :=
by
  sorry

end function_value_l1736_173672


namespace evaluate_expression_l1736_173660

theorem evaluate_expression :
  (3025^2 : ℝ) / ((305^2 : ℝ) - (295^2 : ℝ)) = 1525.10417 :=
by
  sorry

end evaluate_expression_l1736_173660


namespace first_player_wins_l1736_173635

-- Define the set of points S
def S : Set (ℤ × ℤ) := { p | ∃ x y : ℤ, p = (x, y) ∧ x^2 + y^2 ≤ 1010 }

-- Define the game properties and conditions
def game_property :=
  ∀ (p : ℤ × ℤ), p ∈ S →
  ∀ (q : ℤ × ℤ), q ∈ S →
  p ≠ q →
  -- Forbidden to move to a point symmetric to the current one relative to the origin
  q ≠ (-p.fst, -p.snd) →
  -- Distances of moves must strictly increase
  dist p q > dist q (q.fst, q.snd)

-- The first player always guarantees a win
theorem first_player_wins : game_property → true :=
by
  sorry

end first_player_wins_l1736_173635


namespace factory_correct_decision_prob_l1736_173617

def prob_correct_decision (p : ℝ) : ℝ :=
  let prob_all_correct := p * p * p
  let prob_two_correct_one_incorrect := 3 * p * p * (1 - p)
  prob_all_correct + prob_two_correct_one_incorrect

theorem factory_correct_decision_prob : prob_correct_decision 0.8 = 0.896 :=
by
  sorry

end factory_correct_decision_prob_l1736_173617


namespace cost_per_crayon_l1736_173646

-- Definitions for conditions
def half_dozen := 6
def total_crayons := 4 * half_dozen
def total_cost := 48

-- Problem statement
theorem cost_per_crayon :
  (total_cost / total_crayons) = 2 := 
  by
    sorry

end cost_per_crayon_l1736_173646


namespace necessary_but_not_sufficient_l1736_173683

-- Define the conditions as seen in the problem statement
def condition_x (x : ℝ) : Prop := x < 0
def condition_ln (x : ℝ) : Prop := Real.log (x + 1) < 0

-- State that the condition "x < 0" is necessary but not sufficient for "ln(x + 1) < 0"
theorem necessary_but_not_sufficient :
  ∀ (x : ℝ), (condition_ln x → condition_x x) ∧ ¬(condition_x x → condition_ln x) :=
by
  sorry

end necessary_but_not_sufficient_l1736_173683


namespace union_of_sets_l1736_173689

def A : Set ℝ := {x | x < -1 ∨ x > 3}
def B : Set ℝ := {x | x ≥ 2}

theorem union_of_sets : A ∪ B = {x | x < -1 ∨ x ≥ 2} :=
by
  sorry

end union_of_sets_l1736_173689


namespace tan_a4_a12_eq_neg_sqrt3_l1736_173652

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variables {a : ℕ → ℝ} (h_arith : is_arithmetic_sequence a)
          (h_sum : a 1 + a 8 + a 15 = Real.pi)

-- The main statement to prove
theorem tan_a4_a12_eq_neg_sqrt3 : 
  Real.tan (a 4 + a 12) = -Real.sqrt 3 :=
sorry

end tan_a4_a12_eq_neg_sqrt3_l1736_173652


namespace shape_area_is_36_l1736_173607

def side_length : ℝ := 3
def num_squares : ℕ := 4
def area_square : ℝ := side_length ^ 2
def total_area : ℝ := num_squares * area_square

theorem shape_area_is_36 :
  total_area = 36 := by
  sorry

end shape_area_is_36_l1736_173607


namespace combined_area_of_removed_triangles_l1736_173655

theorem combined_area_of_removed_triangles (s : ℝ) (x : ℝ) (h : 15 = ((s - 2 * x) ^ 2 + (s - 2 * x) ^ 2) ^ (1/2)) :
  2 * x ^ 2 = 28.125 :=
by
  -- The necessary proof will go here
  sorry

end combined_area_of_removed_triangles_l1736_173655


namespace final_amoeba_is_blue_l1736_173661

theorem final_amoeba_is_blue
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ)
  (merge : ∀ (a b : ℕ), a ≠ b → ∃ c, a + b - c = a ∧ a + b - c = b ∧ a + b - c = c)
  (initial_counts : n1 = 26 ∧ n2 = 31 ∧ n3 = 16)
  (final_count : ∃ a, a = 1) :
  ∃ color, color = "blue" := sorry

end final_amoeba_is_blue_l1736_173661


namespace smallest_b_in_ap_l1736_173668

-- Definition of an arithmetic progression
def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

-- Problem statement in Lean
theorem smallest_b_in_ap (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h_ap : is_arithmetic_progression a b c) 
  (h_prod : a * b * c = 216) : 
  b ≥ 6 :=
by
  sorry

end smallest_b_in_ap_l1736_173668


namespace tony_lego_sets_l1736_173643

theorem tony_lego_sets
  (price_lego price_sword price_dough : ℕ)
  (num_sword num_dough total_cost : ℕ)
  (L : ℕ)
  (h1 : price_lego = 250)
  (h2 : price_sword = 120)
  (h3 : price_dough = 35)
  (h4 : num_sword = 7)
  (h5 : num_dough = 10)
  (h6 : total_cost = 1940)
  (h7 : total_cost = price_lego * L + price_sword * num_sword + price_dough * num_dough) :
  L = 3 := 
by
  sorry

end tony_lego_sets_l1736_173643


namespace vector_parallel_unique_solution_l1736_173628

def is_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

theorem vector_parallel_unique_solution (m : ℝ) :
  let a := (m^2 - 1, m + 1)
  let b := (1, -2)
  a ≠ (0, 0) → is_parallel a b → m = 1/2 := by
  sorry

end vector_parallel_unique_solution_l1736_173628


namespace total_arrangements_excluding_zhang_for_shooting_event_l1736_173632

theorem total_arrangements_excluding_zhang_for_shooting_event
  (students : Fin 5) 
  (events : Fin 3)
  (shooting : events ≠ 0) : 
  ∃ arrangements, arrangements = 48 := 
sorry

end total_arrangements_excluding_zhang_for_shooting_event_l1736_173632


namespace number_of_teachers_students_possible_rental_plans_economical_plan_l1736_173609

-- Definitions of conditions

def condition1 (x y : ℕ) : Prop := y - 30 * x = 7
def condition2 (x y : ℕ) : Prop := 31 * x - y = 1
def capacity_condition (m : ℕ) : Prop := 35 * m + 30 * (8 - m) ≥ 255
def rental_fee_condition (m : ℕ) : Prop := 400 * m + 320 * (8 - m) ≤ 3000

-- Problems to prove

theorem number_of_teachers_students (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 8 ∧ y = 247 := 
by sorry

theorem possible_rental_plans (m : ℕ) (h_cap : capacity_condition m) (h_fee : rental_fee_condition m) : m = 3 ∨ m = 4 ∨ m = 5 := 
by sorry

theorem economical_plan (m : ℕ) (h_fee : rental_fee_condition 3) (h_fee_alt1 : rental_fee_condition 4) (h_fee_alt2 : rental_fee_condition 5) : m = 3 := 
by sorry

end number_of_teachers_students_possible_rental_plans_economical_plan_l1736_173609


namespace find_k_l1736_173666

noncomputable def polynomial1 : Polynomial Int := sorry

theorem find_k :
  ∃ P : Polynomial Int,
  (P.eval 1 = 2013) ∧
  (P.eval 2013 = 1) ∧
  (∃ k : Int, P.eval k = k) →
  ∃ k : Int, P.eval k = k ∧ k = 1007 :=
by
  sorry

end find_k_l1736_173666


namespace p_neither_sufficient_nor_necessary_l1736_173611

theorem p_neither_sufficient_nor_necessary (x y : ℝ) :
  (x > 1 ∧ y > 1) ↔ ¬((x > 1 ∧ y > 1) → (x + y > 3)) ∧ ¬((x + y > 3) → (x > 1 ∧ y > 1)) :=
by
  sorry

end p_neither_sufficient_nor_necessary_l1736_173611


namespace worker_assignment_l1736_173669

theorem worker_assignment :
  ∃ (x y : ℕ), x + y = 85 ∧
  (16 * x) / 2 = (10 * y) / 3 ∧
  x = 25 ∧ y = 60 :=
by
  sorry

end worker_assignment_l1736_173669


namespace maximum_angle_B_in_triangle_l1736_173622

theorem maximum_angle_B_in_triangle
  (A B C M : ℝ × ℝ)
  (hM : midpoint ℝ A B = M)
  (h_angle_MAC : ∃ angle_MAC : ℝ, angle_MAC = 15) :
  ∃ angle_B : ℝ, angle_B = 105 := 
by
  sorry

end maximum_angle_B_in_triangle_l1736_173622


namespace rate_per_kg_mangoes_l1736_173637

theorem rate_per_kg_mangoes (kg_apples kg_mangoes total_cost rate_apples total_payment rate_mangoes : ℕ) 
  (h1 : kg_apples = 8) 
  (h2 : rate_apples = 70)
  (h3 : kg_mangoes = 9)
  (h4 : total_payment = 965) :
  rate_mangoes = 45 := 
by
  sorry

end rate_per_kg_mangoes_l1736_173637


namespace find_b_l1736_173644

variable (a b : Prod ℝ ℝ)
variable (x y : ℝ)

theorem find_b (h1 : (Prod.fst a + Prod.fst b = 0) ∧
                    (Real.sqrt ((Prod.snd a + Prod.snd b) ^ 2) = 1))
                    (h2 : a = (2, -1)) :
                    b = (-2, 2) ∨ b = (-2, 0) :=
by sorry

end find_b_l1736_173644


namespace angle_bisector_form_l1736_173602

noncomputable def P : ℝ × ℝ := (-8, 5)
noncomputable def Q : ℝ × ℝ := (-15, -19)
noncomputable def R : ℝ × ℝ := (1, -7)

-- Function to check if the given equation can be in the form ax + 2y + c = 0
-- and that a + c equals 89.
theorem angle_bisector_form (a c : ℝ) : a + c = 89 :=
by
   sorry

end angle_bisector_form_l1736_173602


namespace mark_more_than_kate_l1736_173636

variables {K P M : ℕ}

-- Conditions
def total_hours (P K M : ℕ) : Prop := P + K + M = 189
def pat_as_kate (P K : ℕ) : Prop := P = 2 * K
def pat_as_mark (P M : ℕ) : Prop := P = M / 3

-- Statement
theorem mark_more_than_kate (K P M : ℕ) (h1 : total_hours P K M)
  (h2 : pat_as_kate P K) (h3 : pat_as_mark P M) : M - K = 105 :=
by {
  sorry
}

end mark_more_than_kate_l1736_173636


namespace swimming_pool_paint_area_l1736_173671

theorem swimming_pool_paint_area :
  let length := 20 -- The pool is 20 meters long
  let width := 12  -- The pool is 12 meters wide
  let depth := 2   -- The pool is 2 meters deep
  let area_longer_walls := 2 * length * depth
  let area_shorter_walls := 2 * width * depth
  let total_side_wall_area := area_longer_walls + area_shorter_walls
  let floor_area := length * width
  let total_area_to_paint := total_side_wall_area + floor_area
  total_area_to_paint = 368 :=
by
  sorry

end swimming_pool_paint_area_l1736_173671


namespace parameterized_line_equation_l1736_173696

theorem parameterized_line_equation (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 * t + 6) 
  (h2 : y = 5 * t - 7) : 
  y = (5 / 3) * x - 17 :=
sorry

end parameterized_line_equation_l1736_173696


namespace train_speed_l1736_173664

def train_length : ℕ := 110
def bridge_length : ℕ := 265
def crossing_time : ℕ := 30

def speed_in_m_per_s (d t : ℕ) : ℕ := d / t
def speed_in_km_per_hr (s : ℕ) : ℕ := s * 36 / 10

theorem train_speed :
  speed_in_km_per_hr (speed_in_m_per_s (train_length + bridge_length) crossing_time) = 45 :=
by
  sorry

end train_speed_l1736_173664


namespace correct_calculation_l1736_173627

theorem correct_calculation : (6 + (-13)) = -7 :=
by
  sorry

end correct_calculation_l1736_173627


namespace induction_inequality_term_added_l1736_173667

theorem induction_inequality_term_added (k : ℕ) (h : k > 0) :
  let termAdded := (1 / (2 * (k + 1) - 1 : ℝ)) + (1 / (2 * (k + 1) : ℝ)) - (1 / (k + 1 : ℝ))
  ∃ h : ℝ, termAdded = h :=
by
  sorry

end induction_inequality_term_added_l1736_173667


namespace total_weight_correct_l1736_173648

-- Definitions of the given weights of materials
def weight_concrete : ℝ := 0.17
def weight_bricks : ℝ := 0.237
def weight_sand : ℝ := 0.646
def weight_stone : ℝ := 0.5
def weight_steel : ℝ := 1.73
def weight_wood : ℝ := 0.894

-- Total weight of all materials
def total_weight : ℝ := 
  weight_concrete + weight_bricks + weight_sand + weight_stone + weight_steel + weight_wood

-- The proof statement
theorem total_weight_correct : total_weight = 4.177 := by
  sorry

end total_weight_correct_l1736_173648


namespace find_x_l1736_173626

theorem find_x (x : ℝ) (h : 40 * x - 138 = 102) : x = 6 :=
by 
  sorry

end find_x_l1736_173626


namespace average_of_k_l1736_173647

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l1736_173647


namespace area_of_50th_ring_l1736_173634

-- Definitions based on conditions:
def garden_area : ℕ := 9
def ring_area (n : ℕ) : ℕ := 9 * ((2 * n + 1) ^ 2 - (2 * (n - 1) + 1) ^ 2) / 2

-- Theorem to prove:
theorem area_of_50th_ring : ring_area 50 = 1800 := by sorry

end area_of_50th_ring_l1736_173634


namespace negation_of_p_correct_l1736_173665

def p := ∀ x : ℝ, x > Real.sin x

theorem negation_of_p_correct :
  (¬ p) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end negation_of_p_correct_l1736_173665


namespace brayan_hourly_coffee_l1736_173615

theorem brayan_hourly_coffee (I B : ℕ) (h1 : B = 2 * I) (h2 : I + B = 30) : B / 5 = 4 :=
by
  sorry

end brayan_hourly_coffee_l1736_173615


namespace sec_150_eq_neg_2_sqrt_3_div_3_csc_150_eq_2_l1736_173680

noncomputable def sec (x : ℝ) := 1 / Real.cos x
noncomputable def csc (x : ℝ) := 1 / Real.sin x

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := by
  sorry

theorem csc_150_eq_2 : csc (150 * Real.pi / 180) = 2 := by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_csc_150_eq_2_l1736_173680


namespace tangent_line_eq_f_positive_find_a_l1736_173650

noncomputable def f (x a : ℝ) : ℝ := 1 - (a * x^2) / (Real.exp x)
noncomputable def f' (x a : ℝ) : ℝ := (a * x * (x - 2)) / (Real.exp x)

-- Part 1: equation of tangent line
theorem tangent_line_eq (a : ℝ) (h1 : f' 1 a = 1) (hx : f 1 a = 2) : ∀ x, f 1 a + f' 1 a * (x - 1) = x + 1 :=
sorry

-- Part 2: f(x) > 0 for x > 0 when a = 1
theorem f_positive (x : ℝ) (h : x > 0) : f x 1 > 0 :=
sorry

-- Part 3: minimum value of f(x) is -3, find a
theorem find_a (a : ℝ) (h : ∀ x, f x a ≥ -3) : a = Real.exp 2 :=
sorry

end tangent_line_eq_f_positive_find_a_l1736_173650


namespace evaluate_expression_l1736_173610

variable (m n p : ℝ)

theorem evaluate_expression 
  (h : m / (140 - m) + n / (210 - n) + p / (180 - p) = 9) :
  10 / (140 - m) + 14 / (210 - n) + 12 / (180 - p) = 40 := 
by 
  sorry

end evaluate_expression_l1736_173610


namespace people_per_car_l1736_173657

theorem people_per_car (total_people : ℕ) (total_cars : ℕ) (h_people : total_people = 63) (h_cars : total_cars = 3) : 
  total_people / total_cars = 21 := by
  sorry

end people_per_car_l1736_173657


namespace coefficient_a2_in_expansion_l1736_173603

theorem coefficient_a2_in_expansion:
  let a := (x - 1)^4
  let expansion := a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4
  a2 = 6 :=
by
  sorry

end coefficient_a2_in_expansion_l1736_173603


namespace product_sum_divisible_by_1987_l1736_173694

theorem product_sum_divisible_by_1987 :
  let A : ℕ :=
    List.prod (List.filter (λ x => x % 2 = 1) (List.range (1987 + 1)))
  let B : ℕ :=
    List.prod (List.filter (λ x => x % 2 = 0) (List.range (1987 + 1)))
  A + B ≡ 0 [MOD 1987] := by
  -- The proof goes here
  sorry

end product_sum_divisible_by_1987_l1736_173694


namespace fraction_value_l1736_173698

theorem fraction_value : (20 * 21) / (2 + 0 + 2 + 1) = 84 := by
  sorry

end fraction_value_l1736_173698


namespace incorrect_statement_C_l1736_173614

theorem incorrect_statement_C (a b : ℤ) (h : |a| = |b|) : (a ≠ b ∧ a = -b) :=
by
  sorry

end incorrect_statement_C_l1736_173614


namespace travel_time_reduction_impossible_proof_l1736_173604

noncomputable def travel_time_reduction_impossible : Prop :=
  ∀ (x : ℝ), x > 60 → ¬ (1 / x * 60 = 1 - 1)

theorem travel_time_reduction_impossible_proof : travel_time_reduction_impossible :=
sorry

end travel_time_reduction_impossible_proof_l1736_173604


namespace angle_triple_complement_l1736_173697

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l1736_173697


namespace Janice_earnings_after_deductions_l1736_173641

def dailyEarnings : ℕ := 30
def daysWorked : ℕ := 6
def weekdayOvertimeRate : ℕ := 15
def weekendOvertimeRate : ℕ := 20
def weekdayOvertimeShifts : ℕ := 2
def weekendOvertimeShifts : ℕ := 1
def tipsReceived : ℕ := 10
def taxRate : ℝ := 0.10

noncomputable def calculateEarnings : ℝ :=
  let regularEarnings := dailyEarnings * daysWorked
  let overtimeEarnings := (weekdayOvertimeRate * weekdayOvertimeShifts) + (weekendOvertimeRate * weekendOvertimeShifts)
  let totalEarningsBeforeTax := regularEarnings + overtimeEarnings + tipsReceived
  let taxAmount := totalEarningsBeforeTax * taxRate
  totalEarningsBeforeTax - taxAmount

theorem Janice_earnings_after_deductions :
  calculateEarnings = 216 := by
  sorry

end Janice_earnings_after_deductions_l1736_173641


namespace temperature_at_6_km_l1736_173688

-- Define the initial conditions
def groundTemperature : ℝ := 25
def temperatureDropPerKilometer : ℝ := 5

-- Define the question which is the temperature at a height of 6 kilometers
def temperatureAtHeight (height : ℝ) : ℝ :=
  groundTemperature - temperatureDropPerKilometer * height

-- Prove that the temperature at 6 kilometers is -5 degrees Celsius
theorem temperature_at_6_km : temperatureAtHeight 6 = -5 := by
  -- Use expected proof  
  simp [temperatureAtHeight, groundTemperature, temperatureDropPerKilometer]
  sorry

end temperature_at_6_km_l1736_173688


namespace maggie_sold_2_subscriptions_to_neighbor_l1736_173684

-- Definition of the problem conditions
def maggie_pays_per_subscription : Int := 5
def maggie_subscriptions_to_parents : Int := 4
def maggie_subscriptions_to_grandfather : Int := 1
def maggie_earned_total : Int := 55

-- Define the function to be proven
def subscriptions_sold_to_neighbor (x : Int) : Prop :=
  maggie_pays_per_subscription * (maggie_subscriptions_to_parents + maggie_subscriptions_to_grandfather + x + 2*x) = maggie_earned_total

-- The statement we need to prove
theorem maggie_sold_2_subscriptions_to_neighbor :
  subscriptions_sold_to_neighbor 2 :=
sorry

end maggie_sold_2_subscriptions_to_neighbor_l1736_173684


namespace min_value_expr_l1736_173612

theorem min_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (b^2 + 2) / (a + b) + a^2 / (a * b + 1) ≥ 2 :=
sorry

end min_value_expr_l1736_173612


namespace length_of_jordans_rectangle_l1736_173681

theorem length_of_jordans_rectangle
  (carol_length : ℕ) (carol_width : ℕ) (jordan_width : ℕ) (equal_area : (carol_length * carol_width) = (jordan_width * 2)) :
  (2 = 120 / 60) := by
  sorry

end length_of_jordans_rectangle_l1736_173681


namespace not_sophomores_percentage_l1736_173639

theorem not_sophomores_percentage (total_students : ℕ)
    (juniors_percentage : ℚ) (juniors : ℕ)
    (seniors : ℕ) (freshmen sophomores : ℕ)
    (h1 : total_students = 800)
    (h2 : juniors_percentage = 0.22)
    (h3 : juniors = juniors_percentage * total_students)
    (h4 : seniors = 160)
    (h5 : freshmen = sophomores + 48)
    (h6 : freshmen + sophomores + juniors + seniors = total_students) :
    ((total_students - sophomores : ℚ) / total_students) * 100 = 74 := by
  sorry

end not_sophomores_percentage_l1736_173639


namespace exist_integers_not_div_by_7_l1736_173674

theorem exist_integers_not_div_by_7 (k : ℕ) (hk : 0 < k) :
  ∃ (x y : ℤ), (¬ (7 ∣ x)) ∧ (¬ (7 ∣ y)) ∧ (x^2 + 6 * y^2 = 7^k) :=
sorry

end exist_integers_not_div_by_7_l1736_173674


namespace probability_of_correct_digit_in_two_attempts_l1736_173631

theorem probability_of_correct_digit_in_two_attempts : 
  let num_possible_digits := 10
  let num_attempts := 2
  let total_possible_outcomes := num_possible_digits * (num_possible_digits - 1)
  let total_favorable_outcomes := (num_possible_digits - 1) + (num_possible_digits - 1)
  let probability := (total_favorable_outcomes : ℚ) / (total_possible_outcomes : ℚ)
  probability = (1 / 5 : ℚ) :=
by
  sorry

end probability_of_correct_digit_in_two_attempts_l1736_173631


namespace find_ice_cream_cost_l1736_173687

def chapatis_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def rice_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def mixed_vegetable_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def soup_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def dessert_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def soft_drink_cost (num: ℕ) (price: ℝ) (discount: ℝ) : ℝ := num * price * (1 - discount)
def total_cost (chap: ℝ) (rice: ℝ) (veg: ℝ) (soup: ℝ) (dessert: ℝ) (drink: ℝ) : ℝ := chap + rice + veg + soup + dessert + drink
def total_cost_with_tax (base_cost: ℝ) (tax_rate: ℝ) : ℝ := base_cost * (1 + tax_rate)

theorem find_ice_cream_cost :
  let chapatis := chapatis_cost 16 6
  let rice := rice_cost 5 45
  let veg := mixed_vegetable_cost 7 70
  let soup := soup_cost 4 30
  let dessert := dessert_cost 3 85
  let drinks := soft_drink_cost 2 50 0.1
  let base_cost := total_cost chapatis rice veg soup dessert drinks
  let final_cost := total_cost_with_tax base_cost 0.18
  final_cost + 6 * 108.89 = 2159 := 
  by sorry

end find_ice_cream_cost_l1736_173687


namespace division_multiplication_relation_l1736_173624

theorem division_multiplication_relation (h: 7650 / 306 = 25) :
  25 * 306 = 7650 ∧ 7650 / 25 = 306 := 
by 
  sorry

end division_multiplication_relation_l1736_173624


namespace determine_k_l1736_173620

theorem determine_k (k : ℝ) : 
  (2 * k * (-1/2) - 3 = -7 * 3) → k = 18 :=
by
  intro h
  sorry

end determine_k_l1736_173620


namespace monotonic_intervals_increasing_monotonic_intervals_decreasing_maximum_value_minimum_value_l1736_173618

noncomputable def f (x : ℝ) : ℝ := 3 * (Real.sin (2 * x + (Real.pi / 4)))

theorem monotonic_intervals_increasing (k : ℤ) :
  ∀ x y : ℝ, (k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x ≤ y ∧ y ≤ k * Real.pi + Real.pi / 8) → f x ≤ f y :=
sorry

theorem monotonic_intervals_decreasing (k : ℤ) :
  ∀ x y : ℝ, (k * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ y ∧ y ≤ k * Real.pi + 5 * Real.pi / 8) → f x ≥ f y :=
sorry

theorem maximum_value (k : ℤ) :
  f (k * Real.pi + Real.pi / 8) = 3 :=
sorry

theorem minimum_value (k : ℤ) :
  f (k * Real.pi - 3 * Real.pi / 8) = -3 :=
sorry

end monotonic_intervals_increasing_monotonic_intervals_decreasing_maximum_value_minimum_value_l1736_173618
