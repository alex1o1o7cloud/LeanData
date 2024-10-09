import Mathlib

namespace current_books_l665_66563

def initial_books : ℕ := 743
def sold_instore_saturday : ℕ := 37
def sold_online_saturday : ℕ := 128
def sold_instore_sunday : ℕ := 2 * sold_instore_saturday
def sold_online_sunday : ℕ := sold_online_saturday + 34
def total_books_sold_saturday : ℕ := sold_instore_saturday + sold_online_saturday
def total_books_sold_sunday : ℕ := sold_instore_sunday + sold_online_sunday
def total_books_sold_weekend : ℕ := total_books_sold_saturday + total_books_sold_sunday
def books_received_shipment : ℕ := 160
def net_change_books : ℤ := books_received_shipment - total_books_sold_weekend

theorem current_books
  (initial_books : ℕ) 
  (sold_instore_saturday : ℕ) 
  (sold_online_saturday : ℕ) 
  (sold_instore_sunday : ℕ)
  (sold_online_sunday : ℕ)
  (total_books_sold_saturday : ℕ)
  (total_books_sold_sunday : ℕ)
  (total_books_sold_weekend : ℕ)
  (books_received_shipment : ℕ)
  (net_change_books : ℤ) : (initial_books - net_change_books) = 502 := 
by {
  sorry
}

end current_books_l665_66563


namespace garden_table_bench_cost_l665_66567

theorem garden_table_bench_cost (B T : ℕ) (h1 : T + B = 750) (h2 : T = 2 * B) : B = 250 :=
by
  sorry

end garden_table_bench_cost_l665_66567


namespace gcd_m_n_l665_66561

-- Define the numbers m and n
def m : ℕ := 555555555
def n : ℕ := 1111111111

-- State the problem: Prove that gcd(m, n) = 1
theorem gcd_m_n : Nat.gcd m n = 1 :=
by
  -- Proof goes here
  sorry

end gcd_m_n_l665_66561


namespace smallest_x_for_three_digit_product_l665_66550

theorem smallest_x_for_three_digit_product : ∃ x : ℕ, (27 * x >= 100) ∧ (∀ y < x, 27 * y < 100) :=
by
  sorry

end smallest_x_for_three_digit_product_l665_66550


namespace parallel_vectors_l665_66512

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, -2)) (h_b : b = (-1, m)) (h_parallel : ∃ k : ℝ, b = k • a) : m = 2 :=
by {
  sorry
}

end parallel_vectors_l665_66512


namespace james_sushi_rolls_l665_66548

def fish_for_sushi : ℕ := 40
def total_fish : ℕ := 400
def bad_fish_percentage : ℕ := 20

theorem james_sushi_rolls :
  let good_fish := total_fish - (bad_fish_percentage * total_fish / 100)
  good_fish / fish_for_sushi = 8 :=
by
  sorry

end james_sushi_rolls_l665_66548


namespace product_mod_7_l665_66518

theorem product_mod_7 :
  (2009 % 7 = 4) ∧ (2010 % 7 = 5) ∧ (2011 % 7 = 6) ∧ (2012 % 7 = 0) →
  (2009 * 2010 * 2011 * 2012) % 7 = 0 :=
by
  sorry

end product_mod_7_l665_66518


namespace solve_system_b_zero_solve_system_b_nonzero_solve_second_system_l665_66542

section B_zero

variables {x y z b : ℝ}

-- Given conditions for the first system when b = 0
variables (hb_zero : b = 0)
variables (h1 : x + y + z = 0)
variables (h2 : x^2 + y^2 - z^2 = 0)
variables (h3 : 3 * x * y * z - x^3 - y^3 - z^3 = b^3)

theorem solve_system_b_zero :
  ∃ x y z, 3 * x * y * z - x^3 - y^3 - z^3 = b^3 :=
by { sorry }

end B_zero

section B_nonzero

variables {x y z b : ℝ}

-- Given conditions for the first system when b ≠ 0
variables (hb_nonzero : b ≠ 0)
variables (h1 : x + y + z = 2 * b)
variables (h2 : x^2 + y^2 - z^2 = b^2)
variables (h3 : 3 * x * y * z - x^3 - y^3 - z^3 = b^3)

theorem solve_system_b_nonzero :
  ∃ x y z, 3 * x * y * z - x^3 - y^3 - z^3 = b^3 :=
by { sorry }

end B_nonzero

section Second_System

variables {x y z a : ℝ}

-- Given conditions for the second system
variables (h4 : x^2 + y^2 - 2 * z^2 = 2 * a^2)
variables (h5 : x + y + 2 * z = 4 * (a^2 + 1))
variables (h6 : z^2 - x * y = a^2)

theorem solve_second_system :
  ∃ x y z, z^2 - x * y = a^2 :=
by { sorry }

end Second_System

end solve_system_b_zero_solve_system_b_nonzero_solve_second_system_l665_66542


namespace area_union_after_rotation_l665_66510

-- Define the sides of the triangle
def PQ : ℝ := 11
def QR : ℝ := 13
def PR : ℝ := 12

-- Define the condition that H is the centroid of the triangle PQR
def centroid (P Q R H : ℝ × ℝ) : Prop := sorry -- This definition would require geometric relationships.

-- Statement to prove the area of the union of PQR and P'Q'R' after 180° rotation about H.
theorem area_union_after_rotation (P Q R H : ℝ × ℝ) (hPQ : dist P Q = PQ) (hQR : dist Q R = QR) (hPR : dist P R = PR) (hH : centroid P Q R H) : 
  let s := (PQ + QR + PR) / 2
  let area_PQR := Real.sqrt (s * (s - PQ) * (s - QR) * (s - PR))
  2 * area_PQR = 12 * Real.sqrt 105 :=
sorry

end area_union_after_rotation_l665_66510


namespace total_rooms_to_paint_l665_66535

-- Definitions based on conditions
def hours_per_room : ℕ := 8
def rooms_already_painted : ℕ := 8
def hours_to_paint_rest : ℕ := 16

-- Theorem statement
theorem total_rooms_to_paint :
  rooms_already_painted + hours_to_paint_rest / hours_per_room = 10 :=
  sorry

end total_rooms_to_paint_l665_66535


namespace remaining_stock_weighs_120_l665_66577

noncomputable def total_remaining_weight (green_beans_weight rice_weight sugar_weight : ℕ) :=
  let remaining_rice := rice_weight - (rice_weight / 3)
  let remaining_sugar := sugar_weight - (sugar_weight / 5)
  let remaining_stock := remaining_rice + remaining_sugar + green_beans_weight
  remaining_stock

theorem remaining_stock_weighs_120 : total_remaining_weight 60 30 50 = 120 :=
by
  have h1: 60 - 30 = 30 := by norm_num
  have h2: 60 - 10 = 50 := by norm_num
  have h3: 30 - (30 / 3) = 20 := by norm_num
  have h4: 50 - (50 / 5) = 40 := by norm_num
  have h5: 20 + 40 + 60 = 120 := by norm_num
  exact h5

end remaining_stock_weighs_120_l665_66577


namespace stone_breadth_l665_66574

theorem stone_breadth 
  (hall_length_m : ℕ) (hall_breadth_m : ℕ)
  (stone_length_dm : ℕ) (num_stones : ℕ)
  (hall_area_dm2 : ℕ) (stone_area_dm2 : ℕ) 
  (hall_length_dm hall_breadth_dm : ℕ) (b : ℕ) :
  hall_length_m = 36 → hall_breadth_m = 15 →
  stone_length_dm = 8 → num_stones = 1350 →
  hall_length_dm = hall_length_m * 10 → hall_breadth_dm = hall_breadth_m * 10 →
  hall_area_dm2 = hall_length_dm * hall_breadth_dm →
  stone_area_dm2 = stone_length_dm * b →
  hall_area_dm2 = num_stones * stone_area_dm2 →
  b = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  -- Proof would go here
  sorry

end stone_breadth_l665_66574


namespace minor_axis_length_of_ellipse_l665_66526

theorem minor_axis_length_of_ellipse :
  ∀ (x y : ℝ), (9 * x^2 + y^2 = 36) → 4 = 4 :=
by
  intros x y h
  -- the proof goes here
  sorry

end minor_axis_length_of_ellipse_l665_66526


namespace math_problem_l665_66549

noncomputable def problem_statement (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hxyz : x * y * z = 1) : Prop :=
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + x) * (1 + z)) + z^3 / ((1 + x) * (1 + y))) ≥ 3 / 4

theorem math_problem (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hxyz : x * y * z = 1) :
  problem_statement x y z hx hy hz hxyz :=
sorry

end math_problem_l665_66549


namespace number_of_routes_600_l665_66538

-- Define the problem conditions
def number_of_routes (total_cities : Nat) (pick_cities : Nat) (selected_cities : List Nat) : Nat := sorry

-- The number of ways to pick and order 3 cities from remaining 5
def num_ways_pick_three (total_cities : Nat) (pick_cities : Nat) : Nat :=
  Nat.factorial total_cities / Nat.factorial (total_cities - pick_cities)

-- The number of ways to choose positions for M and N
def num_ways_positions (total_positions : Nat) (pick_positions : Nat) : Nat :=
  Nat.choose total_positions pick_positions

-- The main theorem to prove
theorem number_of_routes_600 :
  number_of_routes 7 5 [M, N] = num_ways_pick_three 5 3 * num_ways_positions 4 2 :=
  by sorry

end number_of_routes_600_l665_66538


namespace compute_expression_l665_66594

theorem compute_expression : 45 * (28 + 72) + 55 * 45 = 6975 := 
  by
  sorry

end compute_expression_l665_66594


namespace radius_of_circular_film_l665_66562

theorem radius_of_circular_film (r_canister h_canister t_film R: ℝ) 
  (V: ℝ) (h1: r_canister = 5) (h2: h_canister = 10) 
  (h3: t_film = 0.2) (h4: V = 250 * Real.pi): R = 25 * Real.sqrt 2 :=
by
  sorry

end radius_of_circular_film_l665_66562


namespace sasha_lives_on_seventh_floor_l665_66571

theorem sasha_lives_on_seventh_floor (N : ℕ) (x : ℕ) 
(h1 : x = (1/3 : ℝ) * N) 
(h2 : N - ((1/3 : ℝ) * N + 1) = (1/2 : ℝ) * N) :
  N + 1 = 7 := 
sorry

end sasha_lives_on_seventh_floor_l665_66571


namespace eight_div_repeat_three_l665_66545

-- Initial condition of the problem
def q : ℚ := 1/3

-- Main theorem to prove
theorem eight_div_repeat_three : (8 : ℚ) / q = 24 := by
  -- proof is omitted with sorry
  sorry

end eight_div_repeat_three_l665_66545


namespace symmetric_point_correct_l665_66579

def symmetric_point (P A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x₁, y₁, z₁) := P
  let (x₀, y₀, z₀) := A
  (2 * x₀ - x₁, 2 * y₀ - y₁, 2 * z₀ - z₁)

def P : ℝ × ℝ × ℝ := (3, -2, 4)
def A : ℝ × ℝ × ℝ := (0, 1, -2)
def expected_result : ℝ × ℝ × ℝ := (-3, 4, -8)

theorem symmetric_point_correct : symmetric_point P A = expected_result :=
  by
    sorry

end symmetric_point_correct_l665_66579


namespace sequence_property_l665_66570

noncomputable def U : ℕ → ℕ
| 0       => 0  -- This definition is added to ensure U 1 corresponds to U_1 = 1
| (n + 1) => U n + (n + 1)

theorem sequence_property (n : ℕ) : U n + U (n + 1) = (n + 1) * (n + 1) :=
  sorry

end sequence_property_l665_66570


namespace quadratic_function_relation_l665_66533

theorem quadratic_function_relation 
  (y : ℝ → ℝ) 
  (y_def : ∀ x : ℝ, y x = x^2 + x + 1) 
  (y1 y2 y3 : ℝ) 
  (hA : y (-3) = y1) 
  (hB : y 2 = y2) 
  (hC : y (1/2) = y3) : 
  y3 < y1 ∧ y1 = y2 := 
sorry

end quadratic_function_relation_l665_66533


namespace breakfast_calories_l665_66572

variable (B : ℝ) 

def lunch_calories := 1.25 * B
def dinner_calories := 2.5 * B
def shakes_calories := 900
def total_calories := 3275

theorem breakfast_calories:
  (B + lunch_calories B + dinner_calories B + shakes_calories = total_calories) → B = 500 :=
by
  sorry

end breakfast_calories_l665_66572


namespace Charles_speed_with_music_l665_66519

theorem Charles_speed_with_music (S : ℝ) (h1 : 40 / 60 + 30 / 60 = 70 / 60) (h2 : S * (40 / 60) + 4 * (30 / 60) = 6) : S = 8 :=
by
  sorry

end Charles_speed_with_music_l665_66519


namespace billy_sisters_count_l665_66540

theorem billy_sisters_count 
  (S B : ℕ) -- S is the number of sisters, B is the number of brothers
  (h1 : B = 2 * S) -- Billy has twice as many brothers as sisters
  (h2 : 2 * (B + S) = 12) -- Billy gives 2 sodas to each sibling to give out the 12 pack
  : S = 2 := 
  by sorry

end billy_sisters_count_l665_66540


namespace intersection_of_sets_l665_66557

def setA : Set ℝ := {x | x^2 < 8}
def setB : Set ℝ := {x | 1 - x ≤ 0}
def setIntersection : Set ℝ := {x | x ∈ setA ∧ x ∈ setB}

theorem intersection_of_sets :
    setIntersection = {x | 1 ≤ x ∧ x < 2 * Real.sqrt 2} :=
by
  sorry

end intersection_of_sets_l665_66557


namespace solve_equation_l665_66569

theorem solve_equation (m n : ℝ) (h₀ : m ≠ 0) (h₁ : n ≠ 0) (h₂ : m ≠ n) :
  ∀ x : ℝ, ((x + m)^2 - 3 * (x + n)^2 = m^2 - 3 * n^2) ↔ (x = 0 ∨ x = m - 3 * n) :=
by
  sorry

end solve_equation_l665_66569


namespace temperature_range_l665_66588

-- Conditions: highest temperature and lowest temperature
def highest_temp : ℝ := 5
def lowest_temp : ℝ := -2
variable (t : ℝ) -- given temperature on February 1, 2018

-- Proof problem statement
theorem temperature_range : lowest_temp ≤ t ∧ t ≤ highest_temp :=
sorry

end temperature_range_l665_66588


namespace smallest_x_l665_66578

theorem smallest_x (x : ℕ) :
  (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 8 = 7) → x = 279 :=
by
  sorry

end smallest_x_l665_66578


namespace shaded_region_perimeter_l665_66534

theorem shaded_region_perimeter (C : Real) (r : Real) (L : Real) (P : Real)
  (h0 : C = 48)
  (h1 : r = C / (2 * Real.pi))
  (h2 : L = (90 / 360) * C)
  (h3 : P = 3 * L) :
  P = 36 := by
  sorry

end shaded_region_perimeter_l665_66534


namespace JerryAge_l665_66564

-- Given definitions
def MickeysAge : ℕ := 20
def AgeRelationship (M J : ℕ) : Prop := M = 2 * J + 10

-- Proof statement
theorem JerryAge : ∃ J : ℕ, AgeRelationship MickeysAge J ∧ J = 5 :=
by
  sorry

end JerryAge_l665_66564


namespace root_sum_greater_than_one_l665_66558

noncomputable def f (x a : ℝ) : ℝ := (x * Real.log x) / (x - 1) - a

noncomputable def h (x a : ℝ) : ℝ := (x^2 - x) * f x a

theorem root_sum_greater_than_one {a m x1 x2 : ℝ} (ha : a < 0)
  (h_eq_m : ∀ x, h x a = m) (hx1_root : h x1 a = m) (hx2_root : h x2 a = m)
  (hx1x2_distinct : x1 ≠ x2) :
  x1 + x2 > 1 := 
sorry

end root_sum_greater_than_one_l665_66558


namespace crayons_count_l665_66575

-- Definitions based on the conditions given in the problem
def total_crayons : Nat := 96
def benny_crayons : Nat := 12
def fred_crayons : Nat := 2 * benny_crayons
def jason_crayons (sarah_crayons : Nat) : Nat := 3 * sarah_crayons

-- Stating the proof goal
theorem crayons_count (sarah_crayons : Nat) :
  fred_crayons + benny_crayons + jason_crayons sarah_crayons + sarah_crayons = total_crayons →
  sarah_crayons = 15 ∧
  fred_crayons = 24 ∧
  jason_crayons sarah_crayons = 45 ∧
  benny_crayons = 12 :=
by
  sorry

end crayons_count_l665_66575


namespace donuts_count_is_correct_l665_66598

-- Define the initial number of donuts
def initial_donuts : ℕ := 50

-- Define the number of donuts Bill eats
def eaten_by_bill : ℕ := 2

-- Define the number of donuts taken by the secretary
def taken_by_secretary : ℕ := 4

-- Calculate the remaining donuts after Bill and the secretary take their portions
def remaining_after_bill_and_secretary : ℕ := initial_donuts - eaten_by_bill - taken_by_secretary

-- Define the number of donuts stolen by coworkers (half of the remaining donuts)
def stolen_by_coworkers : ℕ := remaining_after_bill_and_secretary / 2

-- Define the number of donuts left for the meeting
def donuts_left_for_meeting : ℕ := remaining_after_bill_and_secretary - stolen_by_coworkers

-- The theorem to prove
theorem donuts_count_is_correct : donuts_left_for_meeting = 22 :=
by
  sorry

end donuts_count_is_correct_l665_66598


namespace anna_least_days_l665_66531

theorem anna_least_days (borrow : ℝ) (interest_rate : ℝ) (days : ℕ) :
  (borrow = 20) → (interest_rate = 0.10) → borrow + (borrow * interest_rate * days) ≥ 2 * borrow → days ≥ 10 :=
by
  intros h1 h2 h3
  sorry

end anna_least_days_l665_66531


namespace jello_mix_needed_per_pound_l665_66568

variable (bathtub_volume : ℝ) (gallons_per_cubic_foot : ℝ) 
          (pounds_per_gallon : ℝ) (cost_per_tablespoon : ℝ) 
          (total_cost : ℝ)

theorem jello_mix_needed_per_pound :
  bathtub_volume = 6 ∧
  gallons_per_cubic_foot = 7.5 ∧
  pounds_per_gallon = 8 ∧
  cost_per_tablespoon = 0.50 ∧
  total_cost = 270 →
  (total_cost / cost_per_tablespoon) / 
  (bathtub_volume * gallons_per_cubic_foot * pounds_per_gallon) = 1.5 :=
by
  sorry

end jello_mix_needed_per_pound_l665_66568


namespace uki_cupcakes_per_day_l665_66503

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def daily_cookies : ℝ := 10
def daily_biscuits : ℝ := 20
def total_earnings : ℝ := 350
def days : ℝ := 5

-- Define the number of cupcakes baked per day
def cupcakes_per_day (x : ℝ) : Prop :=
  let earnings_cupcakes := price_cupcake * x * days
  let earnings_cookies := price_cookie * daily_cookies * days
  let earnings_biscuits := price_biscuit * daily_biscuits * days
  earnings_cupcakes + earnings_cookies + earnings_biscuits = total_earnings

-- The statement to be proven
theorem uki_cupcakes_per_day : cupcakes_per_day 20 :=
by 
  sorry

end uki_cupcakes_per_day_l665_66503


namespace f_monotonically_decreasing_in_interval_l665_66580

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

theorem f_monotonically_decreasing_in_interval :
  ∀ x y : ℝ, -2 < x ∧ x < 1 → -2 < y ∧ y < 1 → (y > x → f y < f x) :=
by
  sorry

end f_monotonically_decreasing_in_interval_l665_66580


namespace triangle_perimeter_l665_66555

theorem triangle_perimeter : 
  ∀ (f : ℝ → ℝ), 
  (∀ x, f x = 4 * (1 - x / 3)) →
  ∃ (A B C : ℝ × ℝ), 
  A = (3, 0) ∧ 
  B = (0, 4) ∧ 
  C = (0, 0) ∧ 
  dist A B + dist B C + dist C A = 12 :=
by
  sorry

end triangle_perimeter_l665_66555


namespace solve_inequality_l665_66536

-- Define conditions
def valid_x (x : ℝ) : Prop := x ≠ -3 ∧ x ≠ -8/3

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x + 3) > (4 * x + 5) / (3 * x + 8)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  (-3 < x ∧ x < -8/3) ∨ ((1 - Real.sqrt 89) / 4 < x ∧ x < (1 + Real.sqrt 89) / 4)

-- Prove the equivalence
theorem solve_inequality (x : ℝ) (h : valid_x x) : inequality x ↔ solution_set x :=
by
  sorry

end solve_inequality_l665_66536


namespace sufficient_condition_l665_66530

theorem sufficient_condition (A B C D : Prop) (h : C → D): C → (A > B) := 
by 
  sorry

end sufficient_condition_l665_66530


namespace pigeons_count_l665_66500

theorem pigeons_count :
  let initial_pigeons := 1
  let additional_pigeons := 1
  (initial_pigeons + additional_pigeons) = 2 :=
by
  sorry

end pigeons_count_l665_66500


namespace hexagon_angle_D_135_l665_66532

theorem hexagon_angle_D_135 
  (A B C D E F : ℝ)
  (h1 : A = B ∧ B = C)
  (h2 : D = E ∧ E = F)
  (h3 : A = D - 30)
  (h4 : A + B + C + D + E + F = 720) :
  D = 135 :=
by {
  sorry
}

end hexagon_angle_D_135_l665_66532


namespace intersection_distance_squared_l665_66520

-- Definitions for the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 4)^2 = 9

-- Statement to prove
theorem intersection_distance_squared : 
  ∃ C D : ℝ × ℝ, circle1 C.1 C.2 ∧ circle2 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 D.1 D.2 ∧ 
  (C ≠ D) ∧ ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 224 / 9) :=
sorry

end intersection_distance_squared_l665_66520


namespace total_crackers_l665_66593

-- Define the conditions
def boxes_Darren := 4
def crackers_per_box := 24
def boxes_Calvin := 2 * boxes_Darren - 1

-- Define the mathematical proof problem
theorem total_crackers : 
  let total_Darren := boxes_Darren * crackers_per_box
  let total_Calvin := boxes_Calvin * crackers_per_box
  total_Darren + total_Calvin = 264 :=
by
  sorry

end total_crackers_l665_66593


namespace total_orchids_l665_66524

-- Conditions
def current_orchids : ℕ := 2
def additional_orchids : ℕ := 4

-- Proof statement
theorem total_orchids : current_orchids + additional_orchids = 6 :=
by
  sorry

end total_orchids_l665_66524


namespace jose_profit_share_l665_66543

theorem jose_profit_share :
  ∀ (Tom_investment Jose_investment total_profit month_investment_tom month_investment_jose total_month_investment: ℝ),
    Tom_investment = 30000 →
    ∃ (months_tom months_jose : ℝ), months_tom = 12 ∧ months_jose = 10 →
      Jose_investment = 45000 →
      total_profit = 72000 →
      month_investment_tom = Tom_investment * months_tom →
      month_investment_jose = Jose_investment * months_jose →
      total_month_investment = month_investment_tom + month_investment_jose →
      (Jose_investment * months_jose / total_month_investment) * total_profit = 40000 :=
by
  sorry

end jose_profit_share_l665_66543


namespace find_k_l665_66514

theorem find_k (k : ℕ) : (1/2)^18 * (1/81)^k = (1/18)^18 → k = 9 :=
by
  intro h
  sorry

end find_k_l665_66514


namespace inequality_k_l665_66508

variable {R : Type} [LinearOrderedField R] [Nontrivial R]

theorem inequality_k (x y z : R) (k : ℕ) (h : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) 
  (hineq : (1/x) + (1/y) + (1/z) ≥ x + y + z) :
  (1/x^k) + (1/y^k) + (1/z^k) ≥ x^k + y^k + z^k :=
sorry

end inequality_k_l665_66508


namespace find_m_l665_66504

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : ∃ s : ℝ, (s = (m + 1 - 4) / (2 - m)) ∧ s = Real.sqrt 5) :
  m = (10 - Real.sqrt 5) / 4 :=
by
  sorry

end find_m_l665_66504


namespace annika_return_time_l665_66511

-- Define the rate at which Annika hikes.
def hiking_rate := 10 -- minutes per kilometer

-- Define the distances mentioned in the problem.
def initial_distance_east := 2.5 -- kilometers
def total_distance_east := 3.5 -- kilometers

-- Define the time calculations.
def additional_distance_east := total_distance_east - initial_distance_east

-- Calculate the total time required for Annika to get back to the start.
theorem annika_return_time (rate : ℝ) (initial_dist : ℝ) (total_dist : ℝ) (additional_dist : ℝ) : 
  initial_dist = 2.5 → total_dist = 3.5 → rate = 10 → additional_dist = total_dist - initial_dist → 
  (2.5 * rate + additional_dist * rate * 2) = 45 :=
by
-- Since this is just the statement and no proof is needed, we use sorry
sorry

end annika_return_time_l665_66511


namespace f_one_minus_a_l665_66559

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 2) = f x
axiom f_one_plus_a {a : ℝ} : f (1 + a) = 1

theorem f_one_minus_a (a : ℝ) : f (1 - a) = -1 :=
by
  sorry

end f_one_minus_a_l665_66559


namespace angle_double_of_supplementary_l665_66597

theorem angle_double_of_supplementary (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 2 * (180 - x)) : x = 120 :=
sorry

end angle_double_of_supplementary_l665_66597


namespace a_must_be_negative_l665_66544

variable (a b c d e : ℝ)

theorem a_must_be_negative
  (h1 : a / b < -c / d)
  (hb : b > 0)
  (hd : d > 0)
  (he : e > 0)
  (h2 : a + e > 0) : a < 0 := by
  sorry

end a_must_be_negative_l665_66544


namespace sum_n_div_n4_add_16_eq_9_div_320_l665_66513

theorem sum_n_div_n4_add_16_eq_9_div_320 :
  ∑' n : ℕ, n / (n^4 + 16) = 9 / 320 :=
sorry

end sum_n_div_n4_add_16_eq_9_div_320_l665_66513


namespace part1_part2_l665_66565

def U : Set ℝ := {x : ℝ | True}

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part 1: Prove the range of m when 4 ∈ B(m) is [5/2, 3]
theorem part1 (m : ℝ) : (4 ∈ B m) → (5/2 ≤ m ∧ m ≤ 3) := by
  sorry

-- Part 2: Prove the range of m when x ∈ A is a necessary but not sufficient condition for x ∈ B(m) 
theorem part2 (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ∧ ¬(∀ x, x ∈ A → x ∈ B m) → (m ≤ 3) := by
  sorry

end part1_part2_l665_66565


namespace regular_polygon_perimeter_l665_66596

theorem regular_polygon_perimeter (n : ℕ) (exterior_angle : ℝ) (side_length : ℝ) 
  (h1 : 360 / exterior_angle = n) (h2 : 20 = exterior_angle)
  (h3 : 10 = side_length) : 180 = n * side_length :=
by
  sorry

end regular_polygon_perimeter_l665_66596


namespace greatest_n_and_k_l665_66560

-- (condition): k is a positive integer
def isPositive (k : Nat) : Prop :=
  k > 0

-- (condition): k < n
def lessThan (k n : Nat) : Prop :=
  k < n

/-- Let m = 3^n and k be a positive integer such that k < n.
     Determine the greatest value of n for which 3^n divides 25!,
     and the greatest value of k such that 3^k divides (25! - 3^n). -/
theorem greatest_n_and_k :
  ∃ (n k : Nat), (3^n ∣ Nat.factorial 25) ∧ (isPositive k) ∧ (lessThan k n) ∧ (3^k ∣ (Nat.factorial 25 - 3^n)) ∧ n = 10 ∧ k = 9 := by
    sorry

end greatest_n_and_k_l665_66560


namespace total_cost_correct_l665_66528

-- Define the cost of each category of items
def cost_of_book : ℕ := 16
def cost_of_binders : ℕ := 3 * 2
def cost_of_notebooks : ℕ := 6 * 1

-- Define the total cost calculation
def total_cost : ℕ := cost_of_book + cost_of_binders + cost_of_notebooks

-- Prove that the total cost of Léa's purchases is 28
theorem total_cost_correct : total_cost = 28 :=
by {
  -- This is where the proof would go, but it's omitted for now.
  sorry
}

end total_cost_correct_l665_66528


namespace part_a_l665_66501

open Complex

theorem part_a (z : ℂ) (hz : abs z = 1) :
  (abs (z + 1) - Real.sqrt 2) * (abs (z - 1) - Real.sqrt 2) ≤ 0 :=
by
  -- Proof will go here
  sorry

end part_a_l665_66501


namespace radius_of_unique_circle_l665_66541

noncomputable def circle_radius (z : ℂ) (h k : ℝ) : ℝ :=
  if z = 2 then 1/4 else 0  -- function that determines the circle

def unique_circle_radius : Prop :=
  let x1 := 2
  let y1 := 0
  
  let x2 := 3 / 2
  let y2 := Real.sqrt 11 / 2

  let h := 7 / 4 -- x-coordinate of the circle's center
  let k := 0    -- y-coordinate of the circle's center

  let r := 1 / 4 -- Radius of the circle
  
  -- equation of the circle passing through (x1, y1) and (x2, y2) should satisfy
  -- the radius of the resulting circle is r

  (x1 - h)^2 + y1^2 = r^2 ∧ (x2 - h)^2 + y2^2 = r^2

theorem radius_of_unique_circle :
  unique_circle_radius :=
sorry

end radius_of_unique_circle_l665_66541


namespace counter_example_not_power_of_4_for_25_l665_66599

theorem counter_example_not_power_of_4_for_25 : ∃ n ≥ 2, n = 25 ∧ ¬ ∃ k : ℕ, 2 ^ (2 ^ n) % (2 ^ n - 1) = 4 ^ k :=
by {
  sorry
}

end counter_example_not_power_of_4_for_25_l665_66599


namespace some_number_value_l665_66537

theorem some_number_value (x : ℕ) (some_number : ℕ) : x = 5 → ((x / 5) + some_number = 4) → some_number = 3 :=
by
  intros h1 h2
  sorry

end some_number_value_l665_66537


namespace number_of_possible_flags_l665_66502

def colors : List String := ["purple", "gold"]

noncomputable def num_choices_per_stripe (colors : List String) : Nat := 
  colors.length

theorem number_of_possible_flags :
  (num_choices_per_stripe colors) ^ 3 = 8 := 
by
  -- Proof
  sorry

end number_of_possible_flags_l665_66502


namespace range_of_values_for_a_l665_66523

noncomputable def problem_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (|x - 2| + |x + 3| < a)

theorem range_of_values_for_a (a : ℝ) :
  problem_statement a → a ≤ 5 :=
  sorry

end range_of_values_for_a_l665_66523


namespace probability_even_sum_97_l665_66521

-- You don't need to include numbers since they are directly available in Lean's library
-- This will help to ensure broader compatibility and avoid namespace issues

theorem probability_even_sum_97 (m n : ℕ) (hmn : Nat.gcd m n = 1) 
  (hprob : (224 : ℚ) / 455 = m / n) : 
  m + n = 97 :=
sorry

end probability_even_sum_97_l665_66521


namespace modified_cube_edges_l665_66546

/--
A solid cube with a side length of 4 has different-sized solid cubes removed from three of its corners:
- one corner loses a cube of side length 1,
- another corner loses a cube of side length 2,
- and a third corner loses a cube of side length 1.

The total number of edges of the modified solid is 22.
-/
theorem modified_cube_edges :
  let original_edges := 12
  let edges_removed_1x1 := 6
  let edges_added_2x2 := 16
  original_edges - 2 * edges_removed_1x1 + edges_added_2x2 = 22 := by
  sorry

end modified_cube_edges_l665_66546


namespace arithmetic_sequence_properties_l665_66591

noncomputable def common_difference (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1) * d

theorem arithmetic_sequence_properties :
  ∃ d : ℚ, d = 5 / 9 ∧ ∃ S : ℚ, S = -29 / 3 ∧
  ∀ n : ℕ, ∃ a₁ a₅ a₈ : ℚ, a₁ = -3 ∧
    a₅ = common_difference a₁ d 5 ∧
    a₈ = common_difference a₁ d 8 ∧ 
    11 * a₅ = 5 * a₈ - 13 ∧
    S = (n / 2) * (2 * a₁ + (n - 1) * d) ∧
    n = 6 := 
sorry

end arithmetic_sequence_properties_l665_66591


namespace distance_between_circle_centers_l665_66582

open Real

theorem distance_between_circle_centers :
  let center1 := (1 / 2, 0)
  let center2 := (0, 1 / 2)
  dist center1 center2 = sqrt 2 / 2 :=
by
  sorry

end distance_between_circle_centers_l665_66582


namespace simplify_fraction_l665_66522

variable {a b c k : ℝ}
variable (h : a * b = c * k ∧ a * b ≠ 0)

theorem simplify_fraction (h : a * b = c * k ∧ a * b ≠ 0) : 
  (a - b - c + k) / (a + b + c + k) = (a - c) / (a + c) :=
by
  sorry

end simplify_fraction_l665_66522


namespace third_derivative_y_l665_66587

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.sin (5 * x - 3)

theorem third_derivative_y (x : ℝ) : 
  (deriv^[3] y x) = -150 * x * Real.sin (5 * x - 3) + (30 - 125 * x^2) * Real.cos (5 * x - 3) :=
by
  sorry

end third_derivative_y_l665_66587


namespace least_number_to_add_l665_66551

theorem least_number_to_add (x : ℕ) (h : 1055 % 23 = 20) : x = 3 :=
by
  -- Proof goes here.
  sorry

end least_number_to_add_l665_66551


namespace compare_exponents_l665_66507

noncomputable def a : ℝ := 20 ^ 22
noncomputable def b : ℝ := 21 ^ 21
noncomputable def c : ℝ := 22 ^ 20

theorem compare_exponents : a > b ∧ b > c :=
by {
  sorry
}

end compare_exponents_l665_66507


namespace captain_age_eq_your_age_l665_66584

-- Represent the conditions as assumptions
variables (your_age : ℕ) -- You, the captain, have an age as a natural number

-- Define the statement
theorem captain_age_eq_your_age (H_cap : ∀ captain, captain = your_age) : ∀ captain, captain = your_age := by
  sorry

end captain_age_eq_your_age_l665_66584


namespace larger_solution_of_quadratic_l665_66539

theorem larger_solution_of_quadratic :
  ∀ x y : ℝ, x^2 - 19 * x - 48 = 0 ∧ y^2 - 19 * y - 48 = 0 ∧ x ≠ y →
  max x y = 24 :=
by
  sorry

end larger_solution_of_quadratic_l665_66539


namespace tax_calculation_l665_66506

theorem tax_calculation 
  (total_earnings : ℕ) 
  (deductions : ℕ) 
  (tax_paid : ℕ) 
  (tax_rate_10 : ℚ) 
  (tax_rate_20 : ℚ) 
  (taxable_income : ℕ)
  (X : ℕ)
  (h_total_earnings : total_earnings = 100000)
  (h_deductions : deductions = 30000)
  (h_tax_paid : tax_paid = 12000)
  (h_tax_rate_10 : tax_rate_10 = 10 / 100)
  (h_tax_rate_20 : tax_rate_20 = 20 / 100)
  (h_taxable_income : taxable_income = total_earnings - deductions)
  (h_tax_equation : tax_paid = (tax_rate_10 * X) + (tax_rate_20 * (taxable_income - X))) :
  X = 20000 := 
sorry

end tax_calculation_l665_66506


namespace grant_school_students_l665_66525

theorem grant_school_students (S : ℕ) 
  (h1 : S / 3 = x) 
  (h2 : x / 4 = 15) : 
  S = 180 := 
sorry

end grant_school_students_l665_66525


namespace smallest_satisfying_N_is_2520_l665_66590

open Nat

def smallest_satisfying_N : ℕ :=
  let N := 2520
  if (N + 2) % 2 = 0 ∧
     (N + 3) % 3 = 0 ∧
     (N + 4) % 4 = 0 ∧
     (N + 5) % 5 = 0 ∧
     (N + 6) % 6 = 0 ∧
     (N + 7) % 7 = 0 ∧
     (N + 8) % 8 = 0 ∧
     (N + 9) % 9 = 0 ∧
     (N + 10) % 10 = 0
  then N else 0

-- Statement of the problem in Lean 4
theorem smallest_satisfying_N_is_2520 : smallest_satisfying_N = 2520 :=
  by
    -- Proof would be added here, but is omitted as per instructions
    sorry

end smallest_satisfying_N_is_2520_l665_66590


namespace hens_count_l665_66517

theorem hens_count (H C : ℕ) (h_heads : H + C = 60) (h_feet : 2 * H + 4 * C = 200) : H = 20 :=
by
  sorry

end hens_count_l665_66517


namespace ratio_of_circumscribed_areas_l665_66552

noncomputable def rect_pentagon_circ_ratio (P : ℝ) : ℝ :=
  let s : ℝ := P / 8
  let r_circle : ℝ := (P * Real.sqrt 10) / 16
  let A : ℝ := Real.pi * (r_circle ^ 2)
  let pentagon_side : ℝ := P / 5
  let R_pentagon : ℝ := P / (10 * Real.sin (Real.pi / 5))
  let B : ℝ := Real.pi * (R_pentagon ^ 2)
  A / B

theorem ratio_of_circumscribed_areas (P : ℝ) : rect_pentagon_circ_ratio P = (5 * (5 - Real.sqrt 5)) / 64 :=
by sorry

end ratio_of_circumscribed_areas_l665_66552


namespace relationship_between_p_and_q_l665_66592

variables {x y : ℝ}

def p (x y : ℝ) := (x^2 + y^2) * (x - y)
def q (x y : ℝ) := (x^2 - y^2) * (x + y)

theorem relationship_between_p_and_q (h1 : x < y) (h2 : y < 0) : p x y > q x y := 
  by sorry

end relationship_between_p_and_q_l665_66592


namespace correct_group_l665_66547

def atomic_number (element : String) : Nat :=
  match element with
  | "Be" => 4
  | "C" => 6
  | "B" => 5
  | "Cl" => 17
  | "O" => 8
  | "Li" => 3
  | "Al" => 13
  | "S" => 16
  | "Si" => 14
  | "Mg" => 12
  | _ => 0

def is_descending (lst : List Nat) : Bool :=
  match lst with
  | [] => true
  | [x] => true
  | x :: y :: xs => if x > y then is_descending (y :: xs) else false

theorem correct_group : is_descending [atomic_number "Cl", atomic_number "O", atomic_number "Li"] = true ∧
                        is_descending [atomic_number "Be", atomic_number "C", atomic_number "B"] = false ∧
                        is_descending [atomic_number "Al", atomic_number "S", atomic_number "Si"] = false ∧
                        is_descending [atomic_number "C", atomic_number "S", atomic_number "Mg"] = false :=
by
  -- Prove the given theorem based on the atomic number function and is_descending condition
  sorry

end correct_group_l665_66547


namespace clowns_per_mobile_28_l665_66554

def clowns_in_each_mobile (total_clowns num_mobiles : Nat) (h : total_clowns = 140 ∧ num_mobiles = 5) : Nat :=
  total_clowns / num_mobiles

theorem clowns_per_mobile_28 (total_clowns num_mobiles : Nat) (h : total_clowns = 140 ∧ num_mobiles = 5) :
  clowns_in_each_mobile total_clowns num_mobiles h = 28 :=
by
  sorry

end clowns_per_mobile_28_l665_66554


namespace program1_values_program2_values_l665_66595

theorem program1_values :
  ∃ (a b c : ℤ), a = 3 ∧ b = -5 ∧ c = 8 ∧
  a = b ∧ b = c ∧
  a = -5 ∧ b = 8 ∧ c = 8 :=
by sorry

theorem program2_values :
  ∃ (a b c : ℤ), a = 3 ∧ b = -5 ∧ c = 8 ∧
  a = b ∧ b = c ∧ c = a ∧
  a = -5 ∧ b = 8 ∧ c = -5 :=
by sorry

end program1_values_program2_values_l665_66595


namespace discount_limit_l665_66573

theorem discount_limit {cost_price selling_price : ℕ} (x : ℚ)
  (h1: cost_price = 100)
  (h2: selling_price = 150)
  (h3: ∃ p : ℚ, p = 1.2 * cost_price) : selling_price * (x / 10) - cost_price ≥ 0.2 * cost_price ↔ x ≤ 8 :=
by {
  sorry
}

end discount_limit_l665_66573


namespace max_value_x_minus_2y_l665_66529

open Real

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) : 
  x - 2*y ≤ 10 :=
sorry

end max_value_x_minus_2y_l665_66529


namespace problem_solution_l665_66589

theorem problem_solution (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = a * x^2 + (b - 3) * x + 3) →
  (∀ x : ℝ, f x = f (-x)) →
  (a^2 - 2 = -a) →
  a + b = 4 :=
by
  intros h1 h2 h3
  sorry

end problem_solution_l665_66589


namespace find_x_l665_66586

-- Define the mean of three numbers
def mean_three (a b c : ℕ) : ℚ := (a + b + c) / 3

-- Define the mean of two numbers
def mean_two (x y : ℕ) : ℚ := (x + y) / 2

-- Main theorem: value of x that satisfies the given condition
theorem find_x : 
  (mean_three 6 9 18) = (mean_two x 15) → x = 7 :=
by
  sorry

end find_x_l665_66586


namespace part_a_part_b_l665_66527

theorem part_a (x y : ℂ) : (3 * y + 5 * x * Complex.I = 15 - 7 * Complex.I) ↔ (x = -7/5 ∧ y = 5) := by
  sorry

theorem part_b (x y : ℝ) : (2 * x + 3 * y + (x - y) * Complex.I = 7 + 6 * Complex.I) ↔ (x = 5 ∧ y = -1) := by
  sorry

end part_a_part_b_l665_66527


namespace interest_difference_l665_66583

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := 
  P * (1 + r)^t - P

theorem interest_difference : 
  simple_interest 500 0.20 2 - (500 * (1 + 0.20)^2 - 500) = 20 := by
  sorry

end interest_difference_l665_66583


namespace smallest_number_of_players_l665_66505

theorem smallest_number_of_players :
  ∃ n, n ≡ 1 [MOD 3] ∧ n ≡ 2 [MOD 4] ∧ n ≡ 4 [MOD 6] ∧ ∃ m, n = m * m ∧ ∀ k, (k ≡ 1 [MOD 3] ∧ k ≡ 2 [MOD 4] ∧ k ≡ 4 [MOD 6] ∧ ∃ m, k = m * m) → k ≥ n :=
sorry

end smallest_number_of_players_l665_66505


namespace billy_has_62_crayons_l665_66553

noncomputable def billy_crayons (total_crayons : ℝ) (jane_crayons : ℝ) : ℝ :=
  total_crayons - jane_crayons

theorem billy_has_62_crayons : billy_crayons 114 52.0 = 62 := by
  sorry

end billy_has_62_crayons_l665_66553


namespace probability_of_meeting_l665_66556

noncomputable def meeting_probability : ℝ :=
  let total_area := 10 * 10
  let favorable_area := 51
  favorable_area / total_area

theorem probability_of_meeting : meeting_probability = 51 / 100 :=
by
  sorry

end probability_of_meeting_l665_66556


namespace total_oranges_after_increase_l665_66581

theorem total_oranges_after_increase :
  let Mary := 122
  let Jason := 105
  let Tom := 85
  let Sarah := 134
  let increase_rate := 0.10
  let new_Mary := Mary + Mary * increase_rate
  let new_Jason := Jason + Jason * increase_rate
  let new_Tom := Tom + Tom * increase_rate
  let new_Sarah := Sarah + Sarah * increase_rate
  let total_new_oranges := new_Mary + new_Jason + new_Tom + new_Sarah
  Float.round total_new_oranges = 491 := 
by
  sorry

end total_oranges_after_increase_l665_66581


namespace find_possible_values_a_l665_66585

theorem find_possible_values_a :
  ∃ a : ℤ, ∃ b : ℤ, ∃ c : ℤ, 
  (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) ∧
  ((b + 5) * (c + 5) = 1 ∨ (b + 5) * (c + 5) = 4) ↔ 
  a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 7 :=
by
  sorry

end find_possible_values_a_l665_66585


namespace mod_inverse_11_mod_1105_l665_66576

theorem mod_inverse_11_mod_1105 : (11 * 201) % 1105 = 1 :=
  by 
    sorry

end mod_inverse_11_mod_1105_l665_66576


namespace min_value_expression_l665_66516

theorem min_value_expression (a b : ℝ) : 
  4 + (a + b)^2 ≥ 4 ∧ (4 + (a + b)^2 = 4 ↔ a + b = 0) := by
sorry

end min_value_expression_l665_66516


namespace total_points_scored_l665_66509

-- Definitions based on the conditions
def three_point_shots := 13
def two_point_shots := 20
def free_throws := 5
def missed_free_throws := 2
def points_per_three_point_shot := 3
def points_per_two_point_shot := 2
def points_per_free_throw := 1
def penalty_per_missed_free_throw := 1

-- Main statement proving the total points James scored
theorem total_points_scored :
  three_point_shots * points_per_three_point_shot +
  two_point_shots * points_per_two_point_shot +
  free_throws * points_per_free_throw -
  missed_free_throws * penalty_per_missed_free_throw = 82 :=
by
  sorry

end total_points_scored_l665_66509


namespace maxwell_meets_brad_l665_66566

variable (t : ℝ) -- time in hours
variable (distance_between_homes : ℝ) -- total distance
variable (maxwell_speed : ℝ) -- Maxwell's walking speed
variable (brad_speed : ℝ) -- Brad's running speed
variable (brad_delay : ℝ) -- Brad's start time delay

theorem maxwell_meets_brad 
  (hb: brad_delay = 1)
  (d: distance_between_homes = 34)
  (v_m: maxwell_speed = 4)
  (v_b: brad_speed = 6)
  (h : 4 * t + 6 * (t - 1) = distance_between_homes) :
  t = 4 := 
  sorry

end maxwell_meets_brad_l665_66566


namespace product_of_solutions_abs_eq_l665_66515

theorem product_of_solutions_abs_eq (x1 x2 : ℝ) (h1 : |2 * x1 - 1| + 4 = 24) (h2 : |2 * x2 - 1| + 4 = 24) : x1 * x2 = -99.75 := 
sorry

end product_of_solutions_abs_eq_l665_66515
