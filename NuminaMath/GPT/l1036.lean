import Mathlib

namespace NUMINAMATH_GPT_net_progress_l1036_103654

-- Definitions based on conditions in the problem
def loss := 5
def gain := 9

-- Theorem: Proving the team's net progress
theorem net_progress : (gain - loss) = 4 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_net_progress_l1036_103654


namespace NUMINAMATH_GPT_find_some_number_l1036_103678

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end NUMINAMATH_GPT_find_some_number_l1036_103678


namespace NUMINAMATH_GPT_square_area_from_wire_bent_as_circle_l1036_103670

theorem square_area_from_wire_bent_as_circle 
  (radius : ℝ) 
  (h_radius : radius = 56)
  (π_ineq : π > 3.1415) : 
  ∃ (A : ℝ), A = 784 * π^2 := 
by 
  sorry

end NUMINAMATH_GPT_square_area_from_wire_bent_as_circle_l1036_103670


namespace NUMINAMATH_GPT_certain_number_is_two_l1036_103669

theorem certain_number_is_two (n : ℕ) 
  (h1 : 1 = 62) 
  (h2 : 363 = 3634) 
  (h3 : 3634 = n) 
  (h4 : n = 365) 
  (h5 : 36 = 2) : 
  n = 2 := 
by 
  sorry

end NUMINAMATH_GPT_certain_number_is_two_l1036_103669


namespace NUMINAMATH_GPT_grasshopper_flea_adjacency_l1036_103657

-- Define the types of cells
inductive CellColor
| Red
| White

-- Define the infinite grid as a function from ℤ × ℤ to CellColor
def InfiniteGrid : Type := ℤ × ℤ → CellColor

-- Define the positions of the grasshopper and the flea
variables (g_start f_start : ℤ × ℤ)

-- The conditions for the grid and movement rules
axiom grid_conditions (grid : InfiniteGrid) :
  ∃ g_pos f_pos : ℤ × ℤ, 
  (g_pos = g_start ∧ f_pos = f_start) ∧
  (∀ x y : ℤ × ℤ, grid x = CellColor.Red ∨ grid x = CellColor.White) ∧
  (∀ x y : ℤ × ℤ, grid y = CellColor.Red ∨ grid y = CellColor.White)

-- Define the movement conditions for grasshopper and flea
axiom grasshopper_jumps (grid : InfiniteGrid) (start : ℤ × ℤ) :
  ∃ end_pos : ℤ × ℤ, grid end_pos = CellColor.Red ∧ ((end_pos.1 = start.1 ∨ end_pos.2 = start.2) ∧ abs (end_pos.1 - start.1) ≤ 1 ∧ abs (end_pos.2 - start.2) ≤ 1)

axiom flea_jumps (grid : InfiniteGrid) (start : ℤ × ℤ) :
  ∃ end_pos : ℤ × ℤ, grid end_pos = CellColor.White ∧ ((end_pos.1 = start.1 ∨ end_pos.2 = start.2) ∧ abs (end_pos.1 - start.1) ≤ 1 ∧ abs (end_pos.2 - start.2) ≤ 1)

-- The main theorem statement
theorem grasshopper_flea_adjacency (grid : InfiniteGrid)
    (g_start f_start : ℤ × ℤ) :
    ∃ pos1 pos2 pos3 : ℤ × ℤ,
    (pos1 = g_start ∨ pos1 = f_start) ∧ 
    (pos2 = g_start ∨ pos2 = f_start) ∧ 
    (abs (pos3.1 - g_start.1) + abs (pos3.2 - g_start.2) ≤ 1 ∧ 
    abs (pos3.1 - f_start.1) + abs (pos3.2 - f_start.2) ≤ 1) :=
sorry

end NUMINAMATH_GPT_grasshopper_flea_adjacency_l1036_103657


namespace NUMINAMATH_GPT_chips_reach_end_l1036_103691

theorem chips_reach_end (n k : ℕ) (h : n > k * 2^k) : True := sorry

end NUMINAMATH_GPT_chips_reach_end_l1036_103691


namespace NUMINAMATH_GPT_kopecks_payment_l1036_103649

theorem kopecks_payment (n : ℕ) (h : n ≥ 8) : ∃ (a b : ℕ), n = 3 * a + 5 * b :=
sorry

end NUMINAMATH_GPT_kopecks_payment_l1036_103649


namespace NUMINAMATH_GPT_expression_value_l1036_103644

theorem expression_value (x y : ℝ) (h : x + y = -1) : 
  x^4 + 5 * x^3 * y + x^2 * y + 8 * x^2 * y^2 + x * y^2 + 5 * x * y^3 + y^4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1036_103644


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1036_103646

theorem sufficient_not_necessary_condition (a b : ℝ) (h1 : a > 1) (h2 : b > 2) : a + b > 3 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1036_103646


namespace NUMINAMATH_GPT_green_folder_stickers_l1036_103680

theorem green_folder_stickers (total_stickers red_sheets blue_sheets : ℕ) (red_sticker_per_sheet blue_sticker_per_sheet green_stickers_needed green_sheets : ℕ) :
  total_stickers = 60 →
  red_sticker_per_sheet = 3 →
  blue_sticker_per_sheet = 1 →
  red_sheets = 10 →
  blue_sheets = 10 →
  green_sheets = 10 →
  let red_stickers_total := red_sticker_per_sheet * red_sheets
  let blue_stickers_total := blue_sticker_per_sheet * blue_sheets
  let green_stickers_total := total_stickers - (red_stickers_total + blue_stickers_total)
  green_sticker_per_sheet = green_stickers_total / green_sheets →
  green_sticker_per_sheet = 2 := 
sorry

end NUMINAMATH_GPT_green_folder_stickers_l1036_103680


namespace NUMINAMATH_GPT_anna_has_2_fewer_toys_than_amanda_l1036_103618

-- Define the variables for the number of toys each person has
variables (A B : ℕ)

-- Define the conditions
def conditions (M : ℕ) : Prop :=
  M = 20 ∧ A = 3 * M ∧ A + M + B = 142

-- The theorem to prove
theorem anna_has_2_fewer_toys_than_amanda (M : ℕ) (h : conditions A B M) : B - A = 2 :=
sorry

end NUMINAMATH_GPT_anna_has_2_fewer_toys_than_amanda_l1036_103618


namespace NUMINAMATH_GPT_range_of_a_l1036_103648

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1036_103648


namespace NUMINAMATH_GPT_valentines_left_l1036_103663

def initial_valentines : ℕ := 60
def valentines_given_away : ℕ := 16
def valentines_received : ℕ := 5

theorem valentines_left : (initial_valentines - valentines_given_away + valentines_received) = 49 :=
by sorry

end NUMINAMATH_GPT_valentines_left_l1036_103663


namespace NUMINAMATH_GPT_int_solution_count_l1036_103614

def g (n : ℤ) : ℤ :=
  ⌈97 * n / 98⌉ - ⌊98 * n / 99⌋

theorem int_solution_count :
  (∃! n : ℤ, 1 + ⌊98 * n / 99⌋ = ⌈97 * n / 98⌉) :=
sorry

end NUMINAMATH_GPT_int_solution_count_l1036_103614


namespace NUMINAMATH_GPT_shop_earnings_correct_l1036_103615

theorem shop_earnings_correct :
  let cola_price := 3
  let juice_price := 1.5
  let water_price := 1
  let cola_sold := 15
  let juice_sold := 12
  let water_sold := 25
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold = 88 := 
by 
  sorry

end NUMINAMATH_GPT_shop_earnings_correct_l1036_103615


namespace NUMINAMATH_GPT_probability_of_drawing_red_ball_l1036_103676

theorem probability_of_drawing_red_ball :
  let red_balls := 7
  let black_balls := 3
  let total_balls := red_balls + black_balls
  let probability_red := (red_balls : ℚ) / total_balls
  probability_red = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_red_ball_l1036_103676


namespace NUMINAMATH_GPT_sales_tax_is_5_percent_l1036_103620

theorem sales_tax_is_5_percent :
  let cost_tshirt := 8
  let cost_sweater := 18
  let cost_jacket := 80
  let discount := 0.10
  let num_tshirts := 6
  let num_sweaters := 4
  let num_jackets := 5
  let total_cost_with_tax := 504
  let total_cost_before_discount := (num_jackets * cost_jacket)
  let discount_amount := discount * total_cost_before_discount
  let discounted_cost_jackets := total_cost_before_discount - discount_amount
  let total_cost_before_tax := (num_tshirts * cost_tshirt) + (num_sweaters * cost_sweater) + discounted_cost_jackets
  let sales_tax := (total_cost_with_tax - total_cost_before_tax)
  let sales_tax_percentage := (sales_tax / total_cost_before_tax) * 100
  sales_tax_percentage = 5 := by
  sorry

end NUMINAMATH_GPT_sales_tax_is_5_percent_l1036_103620


namespace NUMINAMATH_GPT_ellipse_foci_y_axis_l1036_103612

theorem ellipse_foci_y_axis (k : ℝ) (h_eq : ∀ x y : ℝ, x^2 + k * y^2 = 2)
  (h_foci : ∀ x y : ℝ, x^2 ≤ 2 ∧ k * y^2 ≤ 2) :
  0 < k ∧ k < 1 :=
  sorry

end NUMINAMATH_GPT_ellipse_foci_y_axis_l1036_103612


namespace NUMINAMATH_GPT_range_of_m_l1036_103636

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x-1)^2
  else if x > 0 then -(x+1)^2
  else 0

theorem range_of_m (m : ℝ) (h : f (m^2 + 2*m) + f m > 0) : -3 < m ∧ m < 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_range_of_m_l1036_103636


namespace NUMINAMATH_GPT_incorrect_contrapositive_l1036_103672

theorem incorrect_contrapositive (x : ℝ) : (x ≠ 1 → ¬ (x^2 - 1 = 0)) ↔ ¬ (x^2 - 1 = 0 → x^2 = 1) := by
  sorry

end NUMINAMATH_GPT_incorrect_contrapositive_l1036_103672


namespace NUMINAMATH_GPT_tiffany_uploaded_7_pics_from_her_phone_l1036_103688

theorem tiffany_uploaded_7_pics_from_her_phone
  (camera_pics : ℕ)
  (albums : ℕ)
  (pics_per_album : ℕ)
  (total_pics : ℕ)
  (h_camera_pics : camera_pics = 13)
  (h_albums : albums = 5)
  (h_pics_per_album : pics_per_album = 4)
  (h_total_pics : total_pics = albums * pics_per_album) :
  total_pics - camera_pics = 7 := by
  sorry

end NUMINAMATH_GPT_tiffany_uploaded_7_pics_from_her_phone_l1036_103688


namespace NUMINAMATH_GPT_world_cup_teams_count_l1036_103658

/-- In the world cup inauguration event, captains and vice-captains of all the teams are invited and awarded welcome gifts. There are some teams participating in the world cup, and 14 gifts are needed for this event. If each team has a captain and a vice-captain, and thus receives 2 gifts, then the number of teams participating is 7. -/
theorem world_cup_teams_count (total_gifts : ℕ) (gifts_per_team : ℕ) (teams : ℕ) 
  (h1 : total_gifts = 14) 
  (h2 : gifts_per_team = 2) 
  (h3 : total_gifts = teams * gifts_per_team) 
: teams = 7 :=
by sorry

end NUMINAMATH_GPT_world_cup_teams_count_l1036_103658


namespace NUMINAMATH_GPT_brandon_initial_skittles_l1036_103602

theorem brandon_initial_skittles (initial_skittles : ℕ) (loss : ℕ) (final_skittles : ℕ) (h1 : final_skittles = 87) (h2 : loss = 9) (h3 : final_skittles = initial_skittles - loss) : initial_skittles = 96 :=
sorry

end NUMINAMATH_GPT_brandon_initial_skittles_l1036_103602


namespace NUMINAMATH_GPT_solve_inequality1_solve_inequality_system_l1036_103619

-- Define the first condition inequality
def inequality1 (x : ℝ) : Prop := 
  (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1

-- Theorem for the first inequality proving x >= -2
theorem solve_inequality1 {x : ℝ} (h : inequality1 x) : x ≥ -2 := 
sorry

-- Define the first condition for the system of inequalities
def inequality2 (x : ℝ) : Prop := 
  x - 3 * (x - 2) ≥ 4

-- Define the second condition for the system of inequalities
def inequality3 (x : ℝ) : Prop := 
  (2 * x - 1) / 5 < (x + 1) / 2

-- Theorem for the system of inequalities proving -7 < x ≤ 1
theorem solve_inequality_system {x : ℝ} (h1 : inequality2 x) (h2 : inequality3 x) : -7 < x ∧ x ≤ 1 := 
sorry

end NUMINAMATH_GPT_solve_inequality1_solve_inequality_system_l1036_103619


namespace NUMINAMATH_GPT_k_2_sufficient_but_not_necessary_l1036_103621

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (1, k^2 - 1) - (2, 1)

def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

theorem k_2_sufficient_but_not_necessary (k : ℝ) :
  k = 2 → perpendicular vector_a (vector_b k) ∧ ∃ k, not (k = 2) ∧ perpendicular vector_a (vector_b k) :=
by
  sorry

end NUMINAMATH_GPT_k_2_sufficient_but_not_necessary_l1036_103621


namespace NUMINAMATH_GPT_profit_per_meter_l1036_103642

theorem profit_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (total_cost_price : ℕ := cost_price_per_meter * total_meters)
  (total_profit : ℕ := selling_price - total_cost_price)
  (profit_per_meter : ℕ := total_profit / total_meters) :
  total_meters = 75 ∧ selling_price = 4950 ∧ cost_price_per_meter = 51 → profit_per_meter = 15 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_profit_per_meter_l1036_103642


namespace NUMINAMATH_GPT_steve_commute_l1036_103697

theorem steve_commute :
  ∃ (D : ℝ), 
    (∃ (V : ℝ), 2 * V = 5 ∧ (D / V + D / (2 * V) = 6)) ∧ D = 10 :=
by
  sorry

end NUMINAMATH_GPT_steve_commute_l1036_103697


namespace NUMINAMATH_GPT_inequality_a_b_c_l1036_103682

theorem inequality_a_b_c 
  (a b c : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_c : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_a_b_c_l1036_103682


namespace NUMINAMATH_GPT_tina_earned_more_l1036_103686

def candy_bar_problem_statement : Prop :=
  let type_a_price := 2
  let type_b_price := 3
  let marvin_type_a_sold := 20
  let marvin_type_b_sold := 15
  let tina_type_a_sold := 70
  let tina_type_b_sold := 35
  let marvin_discount_per_5_type_a := 1
  let tina_discount_per_10_type_b := 2
  let tina_returns_type_b := 2
  let marvin_total_earnings := 
    (marvin_type_a_sold * type_a_price) + 
    (marvin_type_b_sold * type_b_price) -
    (marvin_type_a_sold / 5 * marvin_discount_per_5_type_a)
  let tina_total_earnings := 
    (tina_type_a_sold * type_a_price) + 
    (tina_type_b_sold * type_b_price) -
    (tina_type_b_sold / 10 * tina_discount_per_10_type_b) -
    (tina_returns_type_b * type_b_price)
  let difference := tina_total_earnings - marvin_total_earnings
  difference = 152

theorem tina_earned_more :
  candy_bar_problem_statement :=
by
  sorry

end NUMINAMATH_GPT_tina_earned_more_l1036_103686


namespace NUMINAMATH_GPT_tile_equations_correct_l1036_103641

theorem tile_equations_correct (x y : ℕ) (h1 : 24 * x + 12 * y = 2220) (h2 : y = 2 * x - 15) : 
    (24 * x + 12 * y = 2220) ∧ (y = 2 * x - 15) :=
by
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_tile_equations_correct_l1036_103641


namespace NUMINAMATH_GPT_intersection_A_B_l1036_103606

-- Define the set A
def A : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set B as the set of natural numbers greater than 2.5
def B : Set ℕ := {x : ℕ | 2 * x > 5}

-- Prove that the intersection of A and B is {3, 4, 5}
theorem intersection_A_B : A ∩ B = {3, 4, 5} :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1036_103606


namespace NUMINAMATH_GPT_employee_discount_percentage_l1036_103617

theorem employee_discount_percentage (wholesale_cost retail_price employee_price discount_percentage : ℝ) 
  (h1 : wholesale_cost = 200)
  (h2 : retail_price = wholesale_cost * 1.2)
  (h3 : employee_price = 204)
  (h4 : discount_percentage = ((retail_price - employee_price) / retail_price) * 100) :
  discount_percentage = 15 :=
by
  sorry

end NUMINAMATH_GPT_employee_discount_percentage_l1036_103617


namespace NUMINAMATH_GPT_exhibit_special_13_digit_integer_l1036_103647

open Nat 

def thirteenDigitInteger (N : ℕ) : Prop :=
  N ≥ 10^12 ∧ N < 10^13

def isMultipleOf8192 (N : ℕ) : Prop :=
  8192 ∣ N

def hasOnlyEightOrNineDigits (N : ℕ) : Prop :=
  ∀ d ∈ digits 10 N, d = 8 ∨ d = 9

theorem exhibit_special_13_digit_integer : 
  ∃ N : ℕ, thirteenDigitInteger N ∧ isMultipleOf8192 N ∧ hasOnlyEightOrNineDigits N ∧ N = 8888888888888 := 
by
  sorry 

end NUMINAMATH_GPT_exhibit_special_13_digit_integer_l1036_103647


namespace NUMINAMATH_GPT_cylindrical_log_distance_l1036_103667

def cylinder_radius := 3
def R₁ := 104
def R₂ := 64
def R₃ := 84
def straight_segment := 100

theorem cylindrical_log_distance :
  let adjusted_radius₁ := R₁ - cylinder_radius
  let adjusted_radius₂ := R₂ + cylinder_radius
  let adjusted_radius₃ := R₃ - cylinder_radius
  let arc_distance₁ := π * adjusted_radius₁
  let arc_distance₂ := π * adjusted_radius₂
  let arc_distance₃ := π * adjusted_radius₃
  let total_distance := arc_distance₁ + arc_distance₂ + arc_distance₃ + straight_segment
  total_distance = 249 * π + 100 :=
sorry

end NUMINAMATH_GPT_cylindrical_log_distance_l1036_103667


namespace NUMINAMATH_GPT_train_length_l1036_103605

/-- Given a train traveling at 72 km/hr passing a pole in 8 seconds,
     prove that the length of the train in meters is 160. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (speed_m_s : ℝ) (distance_m : ℝ) :
  speed_kmh = 72 → 
  time_s = 8 → 
  speed_m_s = (speed_kmh * 1000) / 3600 → 
  distance_m = speed_m_s * time_s → 
  distance_m = 160 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1036_103605


namespace NUMINAMATH_GPT_nat_square_iff_divisibility_l1036_103635

theorem nat_square_iff_divisibility (A : ℕ) :
  (∃ k : ℕ, A = k^2) ↔ (∀ n : ℕ, ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ n ∣ ((A + i) * (A + i) - A)) :=
sorry

end NUMINAMATH_GPT_nat_square_iff_divisibility_l1036_103635


namespace NUMINAMATH_GPT_division_and_subtraction_l1036_103671

theorem division_and_subtraction :
  (12 : ℚ) / (1 / 6) - (1 / 3) = 215 / 3 :=
by
  sorry

end NUMINAMATH_GPT_division_and_subtraction_l1036_103671


namespace NUMINAMATH_GPT_find_hcf_l1036_103652

-- Defining the conditions given in the problem
def hcf_of_two_numbers_is_H (A B H : ℕ) : Prop := Nat.gcd A B = H
def lcm_of_A_B (A B : ℕ) (H : ℕ) : Prop := Nat.lcm A B = H * 21 * 23
def larger_number_is_460 (A : ℕ) : Prop := A = 460

-- The propositional goal to prove that H = 20 given the above conditions
theorem find_hcf (A B H : ℕ) (hcf_cond : hcf_of_two_numbers_is_H A B H)
  (lcm_cond : lcm_of_A_B A B H) (larger_cond : larger_number_is_460 A) : H = 20 :=
sorry

end NUMINAMATH_GPT_find_hcf_l1036_103652


namespace NUMINAMATH_GPT_intersection_of_function_and_inverse_l1036_103607

theorem intersection_of_function_and_inverse (c k : ℤ) (f : ℤ → ℤ)
  (hf : ∀ x:ℤ, f x = 4 * x + c) 
  (hf_inv : ∀ y:ℤ, (∃ x:ℤ, f x = y) → (∃ x:ℤ, f y = x))
  (h_intersection : ∀ k:ℤ, f 2 = k ∧ f k = 2 ) 
  : k = 2 :=
sorry

end NUMINAMATH_GPT_intersection_of_function_and_inverse_l1036_103607


namespace NUMINAMATH_GPT_elizabeth_initial_bottles_l1036_103683

theorem elizabeth_initial_bottles (B : ℕ) (H1 : B - 2 - 1 = (3 * X) → 3 * (B - 3) = 21) : B = 10 :=
by
  sorry

end NUMINAMATH_GPT_elizabeth_initial_bottles_l1036_103683


namespace NUMINAMATH_GPT_grid_problem_l1036_103690

theorem grid_problem 
    (A B : ℕ)
    (H1 : 1 ≠ A)
    (H2 : 1 ≠ B)
    (H3 : 2 ≠ A)
    (H4 : 2 ≠ B)
    (H5 : 3 ≠ A)
    (H6 : 3 ≠ B)
    (H7 : A = 2)
    (H8 : B = 1)
    :
    A * B = 2 :=
by
  sorry

end NUMINAMATH_GPT_grid_problem_l1036_103690


namespace NUMINAMATH_GPT_quadrilateral_diagonal_length_l1036_103687

theorem quadrilateral_diagonal_length (d : ℝ) 
  (h_offsets : true) 
  (area_quadrilateral : 195 = ((1 / 2) * d * 9) + ((1 / 2) * d * 6)) : 
  d = 26 :=
by 
  sorry

end NUMINAMATH_GPT_quadrilateral_diagonal_length_l1036_103687


namespace NUMINAMATH_GPT_ball_first_bounce_less_than_30_l1036_103653

theorem ball_first_bounce_less_than_30 (b : ℕ) :
  (243 * ((2: ℝ) / 3) ^ b < 30) ↔ (b ≥ 6) :=
sorry

end NUMINAMATH_GPT_ball_first_bounce_less_than_30_l1036_103653


namespace NUMINAMATH_GPT_hua_luogeng_optimal_selection_method_l1036_103626

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end NUMINAMATH_GPT_hua_luogeng_optimal_selection_method_l1036_103626


namespace NUMINAMATH_GPT_hortense_flower_production_l1036_103624

-- Define the initial conditions
def daisy_seeds : ℕ := 25
def sunflower_seeds : ℕ := 25
def daisy_germination_rate : ℚ := 0.60
def sunflower_germination_rate : ℚ := 0.80
def flower_production_rate : ℚ := 0.80

-- Prove the number of plants that produce flowers
theorem hortense_flower_production :
  (daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) * flower_production_rate = 28 :=
by sorry

end NUMINAMATH_GPT_hortense_flower_production_l1036_103624


namespace NUMINAMATH_GPT_largest_4_digit_congruent_to_7_mod_19_l1036_103637

theorem largest_4_digit_congruent_to_7_mod_19 : 
  ∃ x, (x % 19 = 7) ∧ 1000 ≤ x ∧ x < 10000 ∧ x = 9982 :=
by
  sorry

end NUMINAMATH_GPT_largest_4_digit_congruent_to_7_mod_19_l1036_103637


namespace NUMINAMATH_GPT_sequence_term_is_100th_term_l1036_103625

theorem sequence_term_is_100th_term (a : ℕ → ℝ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, a (n + 1) = 2 * a n / (a n + 2)) :
  (∃ n : ℕ, a n = 2 / 101) ∧ ((∃ n : ℕ, a n = 2 / 101) → n = 100) :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_is_100th_term_l1036_103625


namespace NUMINAMATH_GPT_original_weight_of_marble_l1036_103603

variable (W: ℝ) 

theorem original_weight_of_marble (h: 0.80 * 0.82 * 0.72 * W = 85.0176): W = 144 := 
by
  sorry

end NUMINAMATH_GPT_original_weight_of_marble_l1036_103603


namespace NUMINAMATH_GPT_linear_substitution_correct_l1036_103634

theorem linear_substitution_correct (x y : ℝ) 
  (h1 : y = x - 1) 
  (h2 : x + 2 * y = 7) : 
  x + 2 * x - 2 = 7 := 
by
  sorry

end NUMINAMATH_GPT_linear_substitution_correct_l1036_103634


namespace NUMINAMATH_GPT_find_ab_l1036_103645

theorem find_ab (a b : ℝ) 
  (period_cond : (π / b) = (2 * π / 5)) 
  (point_cond : a * Real.tan (5 * (π / 10) / 2) = 1) :
  a * b = 5 / 2 :=
sorry

end NUMINAMATH_GPT_find_ab_l1036_103645


namespace NUMINAMATH_GPT_horner_multiplications_additions_l1036_103698

def f (x : ℝ) : ℝ := 6 * x^6 + 5

def x : ℝ := 2

theorem horner_multiplications_additions :
  (6 : ℕ) = 6 ∧ (6 : ℕ) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_horner_multiplications_additions_l1036_103698


namespace NUMINAMATH_GPT_planks_from_friends_l1036_103699

theorem planks_from_friends :
  let total_planks := 200
  let planks_from_storage := total_planks / 4
  let planks_from_parents := total_planks / 2
  let planks_from_store := 30
  let planks_from_friends := total_planks - (planks_from_storage + planks_from_parents + planks_from_store)
  planks_from_friends = 20 :=
by
  let total_planks := 200
  let planks_from_storage := total_planks / 4
  let planks_from_parents := total_planks / 2
  let planks_from_store := 30
  let planks_from_friends := total_planks - (planks_from_storage + planks_from_parents + planks_from_store)
  rfl

end NUMINAMATH_GPT_planks_from_friends_l1036_103699


namespace NUMINAMATH_GPT_equal_games_per_month_l1036_103650

-- Define the given conditions
def total_games : ℕ := 27
def months : ℕ := 3
def games_per_month := total_games / months

-- Proposition that needs to be proven
theorem equal_games_per_month : games_per_month = 9 := 
by
  sorry

end NUMINAMATH_GPT_equal_games_per_month_l1036_103650


namespace NUMINAMATH_GPT_simplify_sqrt_power_l1036_103664

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_power_l1036_103664


namespace NUMINAMATH_GPT_sum_of_solutions_l1036_103633

theorem sum_of_solutions (S : Finset ℝ) (h : ∀ x ∈ S, |x^2 - 10 * x + 29| = 3) : S.sum id = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_l1036_103633


namespace NUMINAMATH_GPT_net_profit_correct_l1036_103662

-- Define the conditions
def unit_price : ℝ := 1.25
def selling_price : ℝ := 12
def num_patches : ℕ := 100

-- Define the required total cost
def total_cost : ℝ := num_patches * unit_price

-- Define the required total revenue
def total_revenue : ℝ := num_patches * selling_price

-- Define the net profit calculation
def net_profit : ℝ := total_revenue - total_cost

-- The theorem we need to prove
theorem net_profit_correct : net_profit = 1075 := by
    sorry

end NUMINAMATH_GPT_net_profit_correct_l1036_103662


namespace NUMINAMATH_GPT_point_translation_l1036_103673

theorem point_translation :
  ∃ (x_old y_old x_new y_new : ℤ),
  (x_old = 1 ∧ y_old = -2) ∧
  (x_new = x_old + 2) ∧
  (y_new = y_old + 3) ∧
  (x_new = 3) ∧
  (y_new = 1) :=
sorry

end NUMINAMATH_GPT_point_translation_l1036_103673


namespace NUMINAMATH_GPT_factor_expression_l1036_103679

theorem factor_expression (x : ℝ) : 5 * x^2 * (x - 2) - 9 * (x - 2) = (x - 2) * (5 * x^2 - 9) :=
sorry

end NUMINAMATH_GPT_factor_expression_l1036_103679


namespace NUMINAMATH_GPT_interest_rate_part2_l1036_103665

noncomputable def total_investment : ℝ := 3400
noncomputable def part1 : ℝ := 1300
noncomputable def part2 : ℝ := total_investment - part1
noncomputable def rate1 : ℝ := 0.03
noncomputable def total_interest : ℝ := 144
noncomputable def interest1 : ℝ := part1 * rate1
noncomputable def interest2 : ℝ := total_interest - interest1
noncomputable def rate2 : ℝ := interest2 / part2

theorem interest_rate_part2 : rate2 = 0.05 := sorry

end NUMINAMATH_GPT_interest_rate_part2_l1036_103665


namespace NUMINAMATH_GPT_simplified_expression_correct_l1036_103600

def simplify_expression (x : ℝ) : ℝ :=
  4 * (x ^ 2 - 5 * x) - 5 * (2 * x ^ 2 + 3 * x)

theorem simplified_expression_correct (x : ℝ) :
  simplify_expression x = -6 * x ^ 2 - 35 * x :=
by
  sorry

end NUMINAMATH_GPT_simplified_expression_correct_l1036_103600


namespace NUMINAMATH_GPT_shaded_fraction_is_half_l1036_103616

-- Define the number of rows and columns in the grid
def num_rows : ℕ := 8
def num_columns : ℕ := 8

-- Define the number of shaded triangles based on the pattern explained
def shaded_rows : List ℕ := [1, 3, 5, 7]
def num_shaded_rows : ℕ := 4
def triangles_per_row : ℕ := num_columns
def num_shaded_triangles : ℕ := num_shaded_rows * triangles_per_row

-- Define the total number of triangles
def total_triangles : ℕ := num_rows * num_columns

-- Define the fraction of shaded triangles
def shaded_fraction : ℚ := num_shaded_triangles / total_triangles

-- Prove the shaded fraction is 1/2
theorem shaded_fraction_is_half : shaded_fraction = 1 / 2 :=
by
  -- Provide the calculations
  sorry

end NUMINAMATH_GPT_shaded_fraction_is_half_l1036_103616


namespace NUMINAMATH_GPT_f_log3_54_l1036_103627

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x ∧ x < 1 then 3^x else sorry

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f (x)
def functional_equation (f : ℝ → ℝ) := ∀ x, f (x + 2) = -1 / f (x)

-- Hypotheses based on conditions
variable (f : ℝ → ℝ)
axiom f_is_odd : odd_function f
axiom f_is_periodic : periodic_function f 4
axiom f_functional : functional_equation f

-- Main goal
theorem f_log3_54 : f (Real.log 54 / Real.log 3) = -3 / 2 := by
  sorry

end NUMINAMATH_GPT_f_log3_54_l1036_103627


namespace NUMINAMATH_GPT_max_non_equivalent_100_digit_numbers_l1036_103651

noncomputable def maxPairwiseNonEquivalentNumbers : ℕ := 21^5

theorem max_non_equivalent_100_digit_numbers :
  (∀ (n : ℕ), 0 < n ∧ n < 100 → (∀ (digit : Fin n → Fin 2), 
  ∃ (max_num : ℕ), max_num = maxPairwiseNonEquivalentNumbers)) :=
by sorry

end NUMINAMATH_GPT_max_non_equivalent_100_digit_numbers_l1036_103651


namespace NUMINAMATH_GPT_complement_intersection_l1036_103666

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection :
  ((U \ A) ∩ B) = {0} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1036_103666


namespace NUMINAMATH_GPT_divide_subtract_multiply_l1036_103601

theorem divide_subtract_multiply :
  (-5) / ((1/4) - (1/3)) * 12 = 720 := by
  sorry

end NUMINAMATH_GPT_divide_subtract_multiply_l1036_103601


namespace NUMINAMATH_GPT_find_missing_ratio_l1036_103693

def compounded_ratio (x y : ℚ) : ℚ := (x / y) * (6 / 11) * (11 / 2)

theorem find_missing_ratio (x y : ℚ) (h : compounded_ratio x y = 2) :
  x / y = 2 / 3 :=
sorry

end NUMINAMATH_GPT_find_missing_ratio_l1036_103693


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l1036_103684

noncomputable def parabola_value (x m : ℝ) : ℝ := -x^2 - 4 * x + m

variable (m y1 y2 y3 : ℝ)

def point_A_on_parabola : Prop := y1 = parabola_value (-3) m
def point_B_on_parabola : Prop := y2 = parabola_value (-2) m
def point_C_on_parabola : Prop := y3 = parabola_value 1 m


theorem relationship_y1_y2_y3 (hA : point_A_on_parabola y1 m)
                              (hB : point_B_on_parabola y2 m)
                              (hC : point_C_on_parabola y3 m) :
  y2 > y1 ∧ y1 > y3 := 
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l1036_103684


namespace NUMINAMATH_GPT_stone_123_is_12_l1036_103661

/-- Definitions: 
  1. Fifteen stones arranged in a circle counted in a specific pattern: clockwise and counterclockwise.
  2. The sequence of stones enumerated from 1 to 123
  3. The repeating pattern occurs every 28 stones
-/
def stones_counted (n : Nat) : Nat :=
  if n % 28 <= 15 then (n % 28) else (28 - (n % 28) + 1)

theorem stone_123_is_12 : stones_counted 123 = 12 :=
by
  sorry

end NUMINAMATH_GPT_stone_123_is_12_l1036_103661


namespace NUMINAMATH_GPT_mans_speed_upstream_l1036_103643

-- Define the conditions
def V_downstream : ℝ := 15  -- Speed with the current (downstream)
def V_current : ℝ := 2.5    -- Speed of the current

-- Calculate the man's speed against the current (upstream)
theorem mans_speed_upstream : V_downstream - 2 * V_current = 10 :=
by
  sorry

end NUMINAMATH_GPT_mans_speed_upstream_l1036_103643


namespace NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l1036_103630

theorem solve_quadratic1 (x : ℝ) :
  x^2 - 4 * x - 7 = 0 →
  (x = 2 - Real.sqrt 11) ∨ (x = 2 + Real.sqrt 11) :=
by
  sorry

theorem solve_quadratic2 (x : ℝ) :
  (x - 3)^2 + 2 * (x - 3) = 0 →
  (x = 3) ∨ (x = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l1036_103630


namespace NUMINAMATH_GPT_valid_five_letter_words_l1036_103685

def num_valid_words : Nat :=
  let total_words := 3^5
  let invalid_3_consec := 5 * 2^3 * 1^2
  let invalid_4_consec := 2 * 2^4 * 1
  let invalid_5_consec := 2^5
  total_words - (invalid_3_consec + invalid_4_consec + invalid_5_consec)

theorem valid_five_letter_words : num_valid_words = 139 := by
  sorry

end NUMINAMATH_GPT_valid_five_letter_words_l1036_103685


namespace NUMINAMATH_GPT_circle_area_ratio_l1036_103638

theorem circle_area_ratio (r R : ℝ) (h : π * R^2 - π * r^2 = (3/4) * π * r^2) :
  R / r = Real.sqrt 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_area_ratio_l1036_103638


namespace NUMINAMATH_GPT_find_a8_a12_sum_l1036_103608

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a8_a12_sum
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end NUMINAMATH_GPT_find_a8_a12_sum_l1036_103608


namespace NUMINAMATH_GPT_six_digit_number_condition_l1036_103668

theorem six_digit_number_condition (a b c : ℕ) (h : 1 ≤ a ∧ a ≤ 9) (hb : b < 10) (hc : c < 10) : 
  ∃ k : ℕ, 100000 * a + 10000 * b + 1000 * c + 100 * (2 * a) + 10 * (2 * b) + 2 * c = 2 * k := 
by
  sorry

end NUMINAMATH_GPT_six_digit_number_condition_l1036_103668


namespace NUMINAMATH_GPT_sam_initial_money_l1036_103628

theorem sam_initial_money (num_books cost_per_book money_left initial_money : ℤ) 
  (h1 : num_books = 9) 
  (h2 : cost_per_book = 7) 
  (h3 : money_left = 16)
  (h4 : initial_money = num_books * cost_per_book + money_left) :
  initial_money = 79 := 
by
  -- Proof is not required, hence we use sorry to complete the statement.
  sorry

end NUMINAMATH_GPT_sam_initial_money_l1036_103628


namespace NUMINAMATH_GPT_parabolas_vertex_condition_l1036_103655

theorem parabolas_vertex_condition (p q x₁ x₂ y₁ y₂ : ℝ) (h1: y₂ = p * (x₂ - x₁)^2 + y₁) (h2: y₁ = q * (x₁ - x₂)^2 + y₂) (h3: x₁ ≠ x₂) : p + q = 0 :=
sorry

end NUMINAMATH_GPT_parabolas_vertex_condition_l1036_103655


namespace NUMINAMATH_GPT_parallel_lines_a_eq_3_div_2_l1036_103677

theorem parallel_lines_a_eq_3_div_2 (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → a = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_a_eq_3_div_2_l1036_103677


namespace NUMINAMATH_GPT_total_animals_peppersprayed_l1036_103695

-- Define the conditions
def number_of_raccoons : ℕ := 12
def squirrels_vs_raccoons : ℕ := 6
def number_of_squirrels (raccoons : ℕ) (factor : ℕ) : ℕ := raccoons * factor

-- Define the proof statement
theorem total_animals_peppersprayed : 
  number_of_squirrels number_of_raccoons squirrels_vs_raccoons + number_of_raccoons = 84 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_total_animals_peppersprayed_l1036_103695


namespace NUMINAMATH_GPT_mr_smith_payment_l1036_103622

theorem mr_smith_payment {balance : ℝ} {percentage : ℝ} 
  (h_bal : balance = 150) (h_percent : percentage = 0.02) :
  (balance + balance * percentage) = 153 :=
by
  sorry

end NUMINAMATH_GPT_mr_smith_payment_l1036_103622


namespace NUMINAMATH_GPT_solution_a_eq_2_solution_a_in_real_l1036_103694

-- Define the polynomial inequality for the given conditions
def inequality (x : ℝ) (a : ℝ) : Prop := 12 * x ^ 2 - a * x > a ^ 2

-- Proof statement for when a = 2
theorem solution_a_eq_2 :
  ∀ x : ℝ, inequality x 2 ↔ (x < - (1 : ℝ) / 2) ∨ (x > (2 : ℝ) / 3) :=
sorry

-- Proof statement for when a is in ℝ
theorem solution_a_in_real (a : ℝ) :
  ∀ x : ℝ, inequality x a ↔
    if h : 0 < a then (x < - a / 4) ∨ (x > a / 3)
    else if h : a = 0 then (x ≠ 0)
    else (x < a / 3) ∨ (x > - a / 4) :=
sorry

end NUMINAMATH_GPT_solution_a_eq_2_solution_a_in_real_l1036_103694


namespace NUMINAMATH_GPT_B_wins_four_rounds_prob_is_0_09_C_wins_three_rounds_prob_is_0_162_l1036_103611

namespace GoGame

-- Define the players: A, B, C
inductive Player
| A
| B
| C

open Player

-- Define the probabilities as given
def P_A_beats_B : ℝ := 0.4
def P_B_beats_C : ℝ := 0.5
def P_C_beats_A : ℝ := 0.6

-- Define the game rounds and logic
def probability_B_winning_four_rounds 
  (P_A_beats_B : ℝ) (P_B_beats_C : ℝ) (P_C_beats_A : ℝ) : ℝ :=
(1 - P_A_beats_B)^2 * P_B_beats_C^2

def probability_C_winning_three_rounds 
  (P_A_beats_B : ℝ) (P_B_beats_C : ℝ) (P_C_beats_A : ℝ) : ℝ :=
  P_A_beats_B * P_C_beats_A^2 * P_B_beats_C + 
  (1 - P_A_beats_B) * P_B_beats_C^2 * P_C_beats_A

-- Proof statements
theorem B_wins_four_rounds_prob_is_0_09 : 
  probability_B_winning_four_rounds P_A_beats_B P_B_beats_C P_C_beats_A = 0.09 :=
by
  sorry

theorem C_wins_three_rounds_prob_is_0_162 : 
  probability_C_winning_three_rounds P_A_beats_B P_B_beats_C P_C_beats_A = 0.162 :=
by
  sorry

end GoGame

end NUMINAMATH_GPT_B_wins_four_rounds_prob_is_0_09_C_wins_three_rounds_prob_is_0_162_l1036_103611


namespace NUMINAMATH_GPT_weight_of_new_person_l1036_103681

theorem weight_of_new_person
  (avg_increase : ℝ)
  (num_persons : ℕ)
  (replaced_weight : ℝ)
  (weight_increase_total : ℝ)
  (W : ℝ)
  (h1 : avg_increase = 4.5)
  (h2 : num_persons = 8)
  (h3 : replaced_weight = 65)
  (h4 : weight_increase_total = 8 * 4.5)
  (h5 : W = replaced_weight + weight_increase_total) :
  W = 101 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l1036_103681


namespace NUMINAMATH_GPT_lacrosse_more_than_football_l1036_103610

-- Define the constants and conditions
def total_bottles := 254
def football_players := 11
def bottles_per_football_player := 6
def soccer_bottles := 53
def rugby_bottles := 49

-- Calculate the number of bottles needed by each team
def football_bottles := football_players * bottles_per_football_player
def other_teams_bottles := football_bottles + soccer_bottles + rugby_bottles
def lacrosse_bottles := total_bottles - other_teams_bottles

-- The theorem to be proven
theorem lacrosse_more_than_football : lacrosse_bottles - football_bottles = 20 :=
by
  sorry

end NUMINAMATH_GPT_lacrosse_more_than_football_l1036_103610


namespace NUMINAMATH_GPT_no_positive_integer_n_satisfies_l1036_103696

theorem no_positive_integer_n_satisfies :
  ¬∃ (n : ℕ), (1000 ≤ n / 5 ∧ n / 5 ≤ 9999) ∧ (1000 ≤ 5 * n ∧ 5 * n ≤ 9999) := by
  sorry

end NUMINAMATH_GPT_no_positive_integer_n_satisfies_l1036_103696


namespace NUMINAMATH_GPT_proof_problem_l1036_103674

variable (x : Int) (y : Int) (m : Real)

theorem proof_problem :
  ((x = -6 ∧ y = 1 ∧ m = 7.5) ∨ (x = -1 ∧ y = 2 ∧ m = 4)) ↔
  (-2 * x + 3 * y = 2 * m ∧ x - 5 * y = -11 ∧ x < 0 ∧ y > 0)
:= sorry

end NUMINAMATH_GPT_proof_problem_l1036_103674


namespace NUMINAMATH_GPT_quadratic_roots_eq_k_quadratic_inequality_k_range_l1036_103604

theorem quadratic_roots_eq_k (k : ℝ) (h1 : k > 0)
  (h2 : ∀ x : ℝ, 2 < x → x < 3 → k*x^2 - 2*x + 6*k < 0)
  (h3: (2 + 3) = (2/k)) : k = 2/5 :=
by sorry

theorem quadratic_inequality_k_range (k : ℝ) 
  (h1 : k > 0)
  (h2 : ∀ x : ℝ, 2 < x → x < 3 → k*x^2 - 2*x + 6*k < 0) 
: 0 < k ∧ k <= 2/5 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_eq_k_quadratic_inequality_k_range_l1036_103604


namespace NUMINAMATH_GPT_find_value_of_c_l1036_103656

-- Given: The transformed linear regression equation and the definition of z
theorem find_value_of_c (z : ℝ) (y : ℝ) (x : ℝ) (c : ℝ) (k : ℝ) (h1 : z = 0.4 * x + 2) (h2 : z = Real.log y) (h3 : y = c * Real.exp (k * x)) : 
  c = Real.exp 2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_c_l1036_103656


namespace NUMINAMATH_GPT_inequality_solution_l1036_103623

theorem inequality_solution (x : ℝ) :
  (3 / 20 + |x - 13 / 60| < 7 / 30) ↔ (2 / 15 < x ∧ x < 3 / 10) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1036_103623


namespace NUMINAMATH_GPT_max_abs_cubic_at_least_one_fourth_l1036_103613

def cubic_polynomial (p q r x : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem max_abs_cubic_at_least_one_fourth (p q r : ℝ) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |cubic_polynomial p q r x| ≥ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_abs_cubic_at_least_one_fourth_l1036_103613


namespace NUMINAMATH_GPT_min_employees_needed_l1036_103640

theorem min_employees_needed
  (W A S : Finset ℕ)
  (hW : W.card = 120)
  (hA : A.card = 150)
  (hS : S.card = 100)
  (hWA : (W ∩ A).card = 50)
  (hAS : (A ∩ S).card = 30)
  (hWS : (W ∩ S).card = 20)
  (hWAS : (W ∩ A ∩ S).card = 10) :
  (W ∪ A ∪ S).card = 280 :=
by
  sorry

end NUMINAMATH_GPT_min_employees_needed_l1036_103640


namespace NUMINAMATH_GPT_proposition_P_l1036_103692

theorem proposition_P (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) := 
by 
  sorry

end NUMINAMATH_GPT_proposition_P_l1036_103692


namespace NUMINAMATH_GPT_solve_f_neg_a_l1036_103629

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem solve_f_neg_a (h : f a = 8) : f (-a) = -6 := by
  sorry

end NUMINAMATH_GPT_solve_f_neg_a_l1036_103629


namespace NUMINAMATH_GPT_ratio_of_candy_bar_to_caramel_l1036_103689

noncomputable def price_of_caramel : ℝ := 3
noncomputable def price_of_candy_bar (k : ℝ) : ℝ := k * price_of_caramel
noncomputable def price_of_cotton_candy (C : ℝ) : ℝ := 2 * C 

theorem ratio_of_candy_bar_to_caramel (k : ℝ) (C CC : ℝ) :
  C = price_of_candy_bar k →
  CC = price_of_cotton_candy C →
  6 * C + 3 * price_of_caramel + CC = 57 →
  C / price_of_caramel = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_candy_bar_to_caramel_l1036_103689


namespace NUMINAMATH_GPT_aria_spent_on_cookies_in_march_l1036_103639

/-- Aria purchased 4 cookies each day for the entire month of March,
    each cookie costs 19 dollars, and March has 31 days.
    Prove that the total amount Aria spent on cookies in March is 2356 dollars. -/
theorem aria_spent_on_cookies_in_march :
  (4 * 31) * 19 = 2356 := 
by 
  sorry

end NUMINAMATH_GPT_aria_spent_on_cookies_in_march_l1036_103639


namespace NUMINAMATH_GPT_shares_total_amount_l1036_103632

theorem shares_total_amount (Nina_portion : ℕ) (m n o : ℕ) (m_ratio n_ratio o_ratio : ℕ)
  (h_ratio : m_ratio = 2 ∧ n_ratio = 3 ∧ o_ratio = 9)
  (h_Nina : Nina_portion = 60)
  (hk := Nina_portion / n_ratio)
  (h_shares : m = m_ratio * hk ∧ n = n_ratio * hk ∧ o = o_ratio * hk) :
  m + n + o = 280 :=
by 
  sorry

end NUMINAMATH_GPT_shares_total_amount_l1036_103632


namespace NUMINAMATH_GPT_taxi_division_number_of_ways_to_divide_six_people_l1036_103675

theorem taxi_division (people : Finset ℕ) (h : people.card = 6) (taxi1 taxi2 : Finset ℕ) 
  (h1 : taxi1.card ≤ 4) (h2 : taxi2.card ≤ 4) (h_union : people = taxi1 ∪ taxi2) (h_disjoint : Disjoint taxi1 taxi2) :
  (taxi1.card = 3 ∧ taxi2.card = 3) ∨ 
  (taxi1.card = 4 ∧ taxi2.card = 2) :=
sorry

theorem number_of_ways_to_divide_six_people : 
  ∃ n : ℕ, n = 50 :=
sorry

end NUMINAMATH_GPT_taxi_division_number_of_ways_to_divide_six_people_l1036_103675


namespace NUMINAMATH_GPT_cistern_emptying_time_l1036_103631

theorem cistern_emptying_time (R L : ℝ) (hR : R = 1 / 6) (hL : L = 1 / 6 - 1 / 8) :
    1 / L = 24 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_cistern_emptying_time_l1036_103631


namespace NUMINAMATH_GPT_least_number_of_marbles_divisible_by_2_3_4_5_6_7_l1036_103659

theorem least_number_of_marbles_divisible_by_2_3_4_5_6_7 : 
  ∃ n : ℕ, (∀ k ∈ [2, 3, 4, 5, 6, 7], k ∣ n) ∧ n = 420 :=
  by sorry

end NUMINAMATH_GPT_least_number_of_marbles_divisible_by_2_3_4_5_6_7_l1036_103659


namespace NUMINAMATH_GPT_Keith_initial_picked_l1036_103609

-- Definitions based on the given conditions
def Mike_picked := 12
def Keith_gave_away := 46
def remaining_pears := 13

-- Question: Prove that Keith initially picked 47 pears.
theorem Keith_initial_picked :
  ∃ K : ℕ, K = 47 ∧ (K - Keith_gave_away + Mike_picked = remaining_pears) :=
sorry

end NUMINAMATH_GPT_Keith_initial_picked_l1036_103609


namespace NUMINAMATH_GPT_intersection_M_N_l1036_103660

def M : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N :
  M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1036_103660
