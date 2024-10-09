import Mathlib

namespace part1_part2_l285_28500

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 :=
  sorry

theorem part2 (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → h a x ≤ if a ≥ 0 then 3*a + 3 else if -3 ≤ a then a + 3 else 0) :=
  sorry

end part1_part2_l285_28500


namespace factor_problem_l285_28580

theorem factor_problem 
  (a b : ℕ) (h1 : a > b)
  (h2 : (∀ x, x^2 - 16 * x + 64 = (x - a) * (x - b))) 
  : 3 * b - a = 16 := by
  sorry

end factor_problem_l285_28580


namespace time_between_peanuts_l285_28546

def peanuts_per_bag : ℕ := 30
def number_of_bags : ℕ := 4
def flight_time_hours : ℕ := 2

theorem time_between_peanuts (peanuts_per_bag number_of_bags flight_time_hours : ℕ) (h1 : peanuts_per_bag = 30) (h2 : number_of_bags = 4) (h3 : flight_time_hours = 2) :
  (flight_time_hours * 60) / (peanuts_per_bag * number_of_bags) = 1 := by
  sorry

end time_between_peanuts_l285_28546


namespace minimum_value_of_f_l285_28540

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 6 * x + 1

theorem minimum_value_of_f :
  exists (x : ℝ), x = 1 + 1 / Real.sqrt 3 ∧ ∀ (y : ℝ), f (1 + 1 / Real.sqrt 3) ≤ f y := sorry

end minimum_value_of_f_l285_28540


namespace total_cost_for_round_trip_l285_28537

def time_to_cross_one_way : ℕ := 4 -- time in hours to cross the lake one way
def cost_per_hour : ℕ := 10 -- cost in dollars per hour

def total_time := time_to_cross_one_way * 2 -- total time in hours for a round trip
def total_cost := total_time * cost_per_hour -- total cost in dollars for the assistant

theorem total_cost_for_round_trip : total_cost = 80 := by
  repeat {sorry} -- Leaving the proof for now

end total_cost_for_round_trip_l285_28537


namespace tan_value_l285_28517

open Real

theorem tan_value (α : ℝ) (h : sin (5 * π / 6 - α) = sqrt 3 * cos (α + π / 6)) : 
  tan (α + π / 6) = sqrt 3 := 
  sorry

end tan_value_l285_28517


namespace problem_intersecting_lines_l285_28520

theorem problem_intersecting_lines (c d : ℝ) :
  (3 : ℝ) = (1 / 3 : ℝ) * (6 : ℝ) + c ∧ (6 : ℝ) = (1 / 3 : ℝ) * (3 : ℝ) + d → c + d = 6 :=
by
  intros h
  sorry

end problem_intersecting_lines_l285_28520


namespace total_cost_is_13_l285_28508

-- Definition of pencil cost
def pencil_cost : ℕ := 2

-- Definition of pen cost based on pencil cost
def pen_cost : ℕ := pencil_cost + 9

-- The total cost of both items
def total_cost := pencil_cost + pen_cost

theorem total_cost_is_13 : total_cost = 13 := by
  sorry

end total_cost_is_13_l285_28508


namespace complement_of_A_with_respect_to_U_l285_28504

namespace SetTheory

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def C_UA : Set ℕ := {3, 4, 5}

theorem complement_of_A_with_respect_to_U :
  (U \ A) = C_UA := by
  sorry

end SetTheory

end complement_of_A_with_respect_to_U_l285_28504


namespace charlie_book_pages_l285_28541

theorem charlie_book_pages :
  (2 * 40) + (4 * 45) + 20 = 280 :=
by 
  sorry

end charlie_book_pages_l285_28541


namespace older_brother_catches_up_in_half_hour_l285_28590

-- Defining the parameters according to the conditions
def speed_younger_brother := 4 -- kilometers per hour
def speed_older_brother := 20 -- kilometers per hour
def initial_distance := 8 -- kilometers

-- Calculate the relative speed difference
def speed_difference := speed_older_brother - speed_younger_brother

theorem older_brother_catches_up_in_half_hour:
  ∃ t : ℝ, initial_distance = speed_difference * t ∧ t = 0.5 := by
  use 0.5
  sorry

end older_brother_catches_up_in_half_hour_l285_28590


namespace longest_side_of_region_l285_28511

theorem longest_side_of_region :
  (∃ (x y : ℝ), x + y ≤ 5 ∧ 3 * x + y ≥ 3 ∧ x ≥ 1 ∧ y ≥ 1) →
  (∃ (l : ℝ), l = Real.sqrt 130 / 3 ∧ 
    (l = Real.sqrt ((1 - 1)^2 + (4 - 1)^2) ∨ 
     l = Real.sqrt (((1 + 4 / 3) - 1)^2 + (1 - 1)^2) ∨ 
     l = Real.sqrt ((1 - (1 + 4 / 3))^2 + (1 - 1)^2))) :=
by
  sorry

end longest_side_of_region_l285_28511


namespace function_matches_table_values_l285_28582

variable (f : ℤ → ℤ)

theorem function_matches_table_values (h1 : f (-1) = -2) (h2 : f 0 = 0) (h3 : f 1 = 2) (h4 : f 2 = 4) : 
  ∀ x : ℤ, f x = 2 * x := 
by
  -- Prove that the function satisfying the given table values is f(x) = 2x
  sorry

end function_matches_table_values_l285_28582


namespace cost_of_each_hotdog_l285_28551

theorem cost_of_each_hotdog (number_of_hotdogs : ℕ) (total_cost : ℕ) (cost_per_hotdog : ℕ) 
    (h1 : number_of_hotdogs = 6) (h2 : total_cost = 300) : cost_per_hotdog = 50 :=
by
  have h3 : cost_per_hotdog = total_cost / number_of_hotdogs :=
    sorry -- here we would normally write the division step
  sorry -- here we would show that h3 implies cost_per_hotdog = 50, given h1 and h2

end cost_of_each_hotdog_l285_28551


namespace find_natural_number_l285_28514

theorem find_natural_number (n : ℕ) (h1 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n ∨ d = 3 ∨ d = 5 ∨ d = 9 ∨ d = 15)
  (h2 : 1 + 3 + 5 + 9 + 15 + n = 78) : n = 45 := sorry

end find_natural_number_l285_28514


namespace rectangle_side_deficit_l285_28548

theorem rectangle_side_deficit (L W : ℝ) (p : ℝ)
  (h1 : 1.05 * L * (1 - p) * W - L * W = 0.8 / 100 * L * W)
  (h2 : 0 < L) (h3 : 0 < W) : p = 0.04 :=
by {
  sorry
}

end rectangle_side_deficit_l285_28548


namespace total_selling_price_is_correct_l285_28576

def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

def discount : ℝ := discount_rate * original_price
def sale_price : ℝ := original_price - discount
def tax : ℝ := tax_rate * sale_price
def total_selling_price : ℝ := sale_price + tax

theorem total_selling_price_is_correct : total_selling_price = 96.6 := by
  sorry

end total_selling_price_is_correct_l285_28576


namespace range_of_a_l285_28578

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (h_roots : x1 < 1 ∧ 1 < x2) (h_eq : ∀ x, x^2 + a * x - 2 = (x - x1) * (x - x2)) : a < 1 :=
sorry

end range_of_a_l285_28578


namespace find_x_l285_28558

theorem find_x (x : ℝ) (h : 0.95 * x - 12 = 178) : x = 200 := 
by 
  sorry

end find_x_l285_28558


namespace lindsey_squat_weight_l285_28553

theorem lindsey_squat_weight :
  let bandA := 7
  let bandB := 5
  let bandC := 3
  let leg_weight := 10
  let dumbbell := 15
  let total_weight := (2 * bandA) + (2 * bandB) + (2 * bandC) + (2 * leg_weight) + dumbbell
  total_weight = 65 :=
by
  sorry

end lindsey_squat_weight_l285_28553


namespace compute_g_ggg2_l285_28585

def g (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 1
  else if n < 5 then 2 * n + 2
  else 4 * n - 3

theorem compute_g_ggg2 : g (g (g 2)) = 65 :=
by
  sorry

end compute_g_ggg2_l285_28585


namespace sum_S17_l285_28547

-- Definitions of the required arithmetic sequence elements.
variable (a1 d : ℤ)

-- Definition of the arithmetic sequence
def aₙ (n : ℤ) : ℤ := a1 + (n - 1) * d
def Sₙ (n : ℤ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Theorem for the problem statement
theorem sum_S17 : (aₙ a1 d 7 + aₙ a1 d 5) = (3 + aₙ a1 d 5) → (a1 + 8 * d = 3) → Sₙ a1 d 17 = 51 :=
by
  intros h1 h2
  sorry

end sum_S17_l285_28547


namespace find_a5_l285_28557

variable {a_n : ℕ → ℤ} -- Type of the arithmetic sequence
variable (d : ℤ)       -- Common difference of the sequence

-- Assuming the sequence is defined as an arithmetic progression
axiom arithmetic_seq (a d : ℤ) : ∀ n : ℕ, a_n n = a + n * d

theorem find_a5
  (h : a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 45):
  a_n 5 = 9 :=
by 
  sorry

end find_a5_l285_28557


namespace diamond_op_example_l285_28526

def diamond_op (x y : ℕ) : ℕ := 3 * x + 5 * y

theorem diamond_op_example : diamond_op 2 7 = 41 :=
by {
    -- proof goes here
    sorry
}

end diamond_op_example_l285_28526


namespace new_percentage_water_is_correct_l285_28539

def initial_volume : ℕ := 120
def initial_percentage_water : ℚ := 20 / 100
def added_water : ℕ := 8

def initial_volume_water : ℚ := initial_percentage_water * initial_volume
def initial_volume_wine : ℚ := initial_volume - initial_volume_water
def new_volume_water : ℚ := initial_volume_water + added_water
def new_total_volume : ℚ := initial_volume + added_water

def calculate_new_percentage_water : ℚ :=
  (new_volume_water / new_total_volume) * 100

theorem new_percentage_water_is_correct :
  calculate_new_percentage_water = 25 := 
by
  sorry

end new_percentage_water_is_correct_l285_28539


namespace minimum_inequality_l285_28579

theorem minimum_inequality 
  (x_1 x_2 x_3 x_4 : ℝ) 
  (h1 : x_1 > 0) 
  (h2 : x_2 > 0) 
  (h3 : x_3 > 0) 
  (h4 : x_4 > 0) 
  (h_sum : x_1^2 + x_2^2 + x_3^2 + x_4^2 = 4) :
  (x_1 / (1 - x_1^2) + x_2 / (1 - x_2^2) + x_3 / (1 - x_3^2) + x_4 / (1 - x_4^2)) ≥ 6 * Real.sqrt 3 :=
by
  sorry

end minimum_inequality_l285_28579


namespace simplify_tan_product_l285_28552

noncomputable def tan_deg (d : ℝ) : ℝ := Real.tan (d * Real.pi / 180)

theorem simplify_tan_product :
  (1 + tan_deg 10) * (1 + tan_deg 35) = 2 := 
by
  -- Given conditions
  have h1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have h2 : tan_deg 10 + tan_deg 35 = 1 - tan_deg 10 * tan_deg 35 :=
    by sorry -- Use tan addition formula here
  -- Proof of the theorem follows from here
  sorry

end simplify_tan_product_l285_28552


namespace possible_values_of_k_l285_28521

-- Definition of the proposition
def proposition (k : ℝ) : Prop :=
  ∃ x : ℝ, (k^2 - 1) * x^2 + 4 * (1 - k) * x + 3 ≤ 0

-- The main statement to prove in Lean 4
theorem possible_values_of_k (k : ℝ) : ¬ proposition k ↔ (k = 1 ∨ (1 < k ∧ k < 7)) :=
by 
  sorry

end possible_values_of_k_l285_28521


namespace cobbler_mends_3_pairs_per_hour_l285_28550

def cobbler_hours_per_day_mon_thu := 8
def cobbler_hours_friday := 11 - 8
def cobbler_total_hours_week := 4 * cobbler_hours_per_day_mon_thu + cobbler_hours_friday
def cobbler_pairs_per_week := 105
def cobbler_pairs_per_hour := cobbler_pairs_per_week / cobbler_total_hours_week

theorem cobbler_mends_3_pairs_per_hour : cobbler_pairs_per_hour = 3 := 
by 
  -- Add the steps if necessary but in this scenario, we are skipping proof details
  sorry

end cobbler_mends_3_pairs_per_hour_l285_28550


namespace integer_solution_a_l285_28565

theorem integer_solution_a (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by
  sorry

end integer_solution_a_l285_28565


namespace tan_alpha_sin_cos_half_alpha_l285_28516

variable (α : ℝ)

-- Conditions given in the problem
def cond1 : Real.sin α = 1 / 3 := sorry
def cond2 : 0 < α ∧ α < Real.pi := sorry

-- Lean proof that given the conditions, the solutions are as follows:
theorem tan_alpha (h1 : Real.sin α = 1 / 3) (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = Real.sqrt 2 / 4 ∨ Real.tan α = - Real.sqrt 2 / 4 := sorry

theorem sin_cos_half_alpha (h1 : Real.sin α = 1 / 3) (h2 : 0 < α ∧ α < Real.pi) : 
  Real.sin (α / 2) + Real.cos (α / 2) = 2 * Real.sqrt 3 / 3 := sorry

end tan_alpha_sin_cos_half_alpha_l285_28516


namespace average_marks_l285_28510

theorem average_marks :
  let class1_students := 26
  let class1_avg_marks := 40
  let class2_students := 50
  let class2_avg_marks := 60
  let total_students := class1_students + class2_students
  let total_marks := (class1_students * class1_avg_marks) + (class2_students * class2_avg_marks)
  (total_marks / total_students : ℝ) = 53.16 := by
sorry

end average_marks_l285_28510


namespace linear_valid_arrangements_circular_valid_arrangements_l285_28592

def word := "EFFERVESCES"
def multiplicities := [("E", 4), ("F", 2), ("S", 2), ("R", 1), ("V", 1), ("C", 1)]

-- Number of valid linear arrangements
def linear_arrangements_no_adj_e : ℕ := 88200

-- Number of valid circular arrangements
def circular_arrangements_no_adj_e : ℕ := 6300

theorem linear_valid_arrangements : 
  ∃ n, n = linear_arrangements_no_adj_e := 
  by
    sorry 

theorem circular_valid_arrangements :
  ∃ n, n = circular_arrangements_no_adj_e :=
  by
    sorry

end linear_valid_arrangements_circular_valid_arrangements_l285_28592


namespace certain_number_is_four_l285_28568

theorem certain_number_is_four (k : ℕ) (h₁ : k = 16) : 64 / k = 4 :=
by
  sorry

end certain_number_is_four_l285_28568


namespace taxi_ride_cost_l285_28536

theorem taxi_ride_cost (base_fare : ℝ) (rate_per_mile : ℝ) (additional_charge : ℝ) (distance : ℕ) (cost : ℝ) :
  base_fare = 2 ∧ rate_per_mile = 0.30 ∧ additional_charge = 5 ∧ distance = 12 ∧ 
  cost = base_fare + (rate_per_mile * distance) + additional_charge → cost = 10.60 :=
by
  intros
  sorry

end taxi_ride_cost_l285_28536


namespace initial_calculated_average_l285_28570

theorem initial_calculated_average (S : ℕ) (initial_average correct_average : ℕ) (num_wrongly_read correctly_read wrong_value correct_value : ℕ)
    (h1 : num_wrongly_read = 36) 
    (h2 : correctly_read = 26) 
    (h3 : correct_value = 6)
    (h4 : S = 10 * correct_value) :
    initial_average = (S - (num_wrongly_read - correctly_read)) / 10 → initial_average = 5 :=
sorry

end initial_calculated_average_l285_28570


namespace distinct_real_roots_k_root_condition_k_l285_28588

-- Part (1) condition: The quadratic equation has two distinct real roots
theorem distinct_real_roots_k (k : ℝ) : (∃ x : ℝ, x^2 + 2*x + k = 0) ∧ (∀ x y : ℝ, x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0 → x ≠ y) → k < 1 := 
sorry

-- Part (2) condition: m is a root and satisfies m^2 + 2m = 2
theorem root_condition_k (m k : ℝ) : m^2 + 2*m = 2 → m^2 + 2*m + k = 0 → k = -2 := 
sorry

end distinct_real_roots_k_root_condition_k_l285_28588


namespace solution_for_system_l285_28577
open Real

noncomputable def solve_system (a b x y : ℝ) : Prop :=
  (a * x + b * y = 7 ∧ b * x + a * y = 8)

noncomputable def solve_linear (a b m n : ℝ) : Prop :=
  (a * (m + n) + b * (m - n) = 7 ∧ b * (m + n) + a * (m - n) = 8)

theorem solution_for_system (a b : ℝ) : solve_system a b 2 3 → solve_linear a b (5/2) (-1/2) :=
by {
  sorry
}

end solution_for_system_l285_28577


namespace integer_solutions_of_equation_l285_28572

theorem integer_solutions_of_equation :
  ∀ (x y : ℤ), (x^4 + y^4 = 3 * x^3 * y) → (x = 0 ∧ y = 0) := by
  intros x y h
  sorry

end integer_solutions_of_equation_l285_28572


namespace coloring_impossible_l285_28524

theorem coloring_impossible :
  ¬ ∃ (color : ℕ → Prop), (∀ n m : ℕ, (m = n + 5 → color n ≠ color m) ∧ (m = 2 * n → color n ≠ color m)) :=
sorry

end coloring_impossible_l285_28524


namespace wechat_balance_l285_28569

def transaction1 : ℤ := 48
def transaction2 : ℤ := -30
def transaction3 : ℤ := -50

theorem wechat_balance :
  transaction1 + transaction2 + transaction3 = -32 :=
by
  -- placeholder for proof
  sorry

end wechat_balance_l285_28569


namespace compute_b_l285_28562

-- Defining the polynomial and the root conditions
def poly (x a b : ℝ) := x^3 + a * x^2 + b * x + 21

theorem compute_b (a b : ℚ) (h1 : poly (3 + Real.sqrt 5) a b = 0) (h2 : poly (3 - Real.sqrt 5) a b = 0) : 
  b = -27.5 := 
sorry

end compute_b_l285_28562


namespace three_digit_reverse_sum_to_1777_l285_28581

theorem three_digit_reverse_sum_to_1777 :
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 101 * (a + c) + 20 * b = 1777 :=
by
  sorry

end three_digit_reverse_sum_to_1777_l285_28581


namespace tangent_line_equations_l285_28559

theorem tangent_line_equations (k b : ℝ) :
  (∃ l : ℝ → ℝ, (∀ x, l x = k * x + b) ∧
    (∃ x₁, x₁^2 = k * x₁ + b) ∧ -- Tangency condition with C1: y = x²
    (∃ x₂, -(x₂ - 2)^2 = k * x₂ + b)) -- Tangency condition with C2: y = -(x-2)²
  → ((k = 0 ∧ b = 0) ∨ (k = 4 ∧ b = -4)) := sorry

end tangent_line_equations_l285_28559


namespace sqrt_of_product_of_powers_l285_28566

theorem sqrt_of_product_of_powers :
  (Real.sqrt (4^2 * 5^6) = 500) :=
by
  sorry

end sqrt_of_product_of_powers_l285_28566


namespace shoes_difference_l285_28519

theorem shoes_difference : 
  ∀ (Scott_shoes Anthony_shoes Jim_shoes : ℕ), 
  Scott_shoes = 7 → 
  Anthony_shoes = 3 * Scott_shoes → 
  Jim_shoes = Anthony_shoes - 2 → 
  Anthony_shoes - Jim_shoes = 2 :=
by
  intros Scott_shoes Anthony_shoes Jim_shoes 
  intros h1 h2 h3 
  sorry

end shoes_difference_l285_28519


namespace final_amoeba_is_blue_l285_28574

-- We define the initial counts of each type of amoeba
def initial_red : ℕ := 26
def initial_blue : ℕ := 31
def initial_yellow : ℕ := 16

-- We define the final count of amoebas
def final_amoebas : ℕ := 1

-- The type of the final amoeba (we're proving it's 'blue')
inductive AmoebaColor
| Red
| Blue
| Yellow

-- Given initial counts, we aim to prove the final amoeba is blue
theorem final_amoeba_is_blue :
  initial_red = 26 ∧ initial_blue = 31 ∧ initial_yellow = 16 ∧ final_amoebas = 1 → 
  ∃ c : AmoebaColor, c = AmoebaColor.Blue :=
by sorry

end final_amoeba_is_blue_l285_28574


namespace fourth_root_of_25000000_eq_70_7_l285_28531

theorem fourth_root_of_25000000_eq_70_7 :
  Real.sqrt (Real.sqrt 25000000) = 70.7 :=
sorry

end fourth_root_of_25000000_eq_70_7_l285_28531


namespace angle_equiv_470_110_l285_28534

theorem angle_equiv_470_110 : ∃ (k : ℤ), 470 = k * 360 + 110 :=
by
  use 1
  exact rfl

end angle_equiv_470_110_l285_28534


namespace tangent_slope_at_point_552_32_l285_28575

noncomputable def slope_of_tangent_at_point (cx cy px py : ℚ) : ℚ :=
if py - cy = 0 then 
  0 
else 
  (px - cx) / (py - cy)

theorem tangent_slope_at_point_552_32 : slope_of_tangent_at_point 3 2 5 5 = -2 / 3 :=
by
  -- Conditions from problem
  have h1 : slope_of_tangent_at_point 3 2 5 5 = -2 / 3 := 
    sorry
  
  exact h1

end tangent_slope_at_point_552_32_l285_28575


namespace number_of_children_l285_28532

theorem number_of_children :
  ∃ a : ℕ, (a % 8 = 5) ∧ (a % 10 = 7) ∧ (100 ≤ a) ∧ (a ≤ 150) ∧ (a = 125) :=
by
  sorry

end number_of_children_l285_28532


namespace sum_of_squares_of_roots_l285_28555

theorem sum_of_squares_of_roots
  (x1 x2 : ℝ) (h : 5 * x1^2 + 6 * x1 - 15 = 0) (h' : 5 * x2^2 + 6 * x2 - 15 = 0) :
  x1^2 + x2^2 = 186 / 25 :=
sorry

end sum_of_squares_of_roots_l285_28555


namespace divisor_is_three_l285_28594

noncomputable def find_divisor (n : ℕ) (reduction : ℕ) (result : ℕ) : ℕ :=
  n / result

theorem divisor_is_three (x : ℝ) : 
  (original : ℝ) → (reduction : ℝ) → (new_result : ℝ) → 
  original = 45 → new_result = 45 - 30 → (original / x = new_result) → 
  x = 3 := by 
  intros original reduction new_result h1 h2 h3
  sorry

end divisor_is_three_l285_28594


namespace total_income_l285_28506

theorem total_income (I : ℝ) (h1 : 0.10 * I * 2 + 0.20 * I + 0.06 * (I - 0.40 * I) = 0.46 * I) (h2 : 0.54 * I = 500) : I = 500 / 0.54 :=
by
  sorry

end total_income_l285_28506


namespace triple_angle_l285_28503

theorem triple_angle (α : ℝ) : 3 * α = α + α + α := 
by sorry

end triple_angle_l285_28503


namespace value_of_B_l285_28527

theorem value_of_B (B : ℝ) : 3 * B ^ 2 + 3 * B + 2 = 29 ↔ (B = (-1 + Real.sqrt 37) / 2 ∨ B = (-1 - Real.sqrt 37) / 2) :=
by sorry

end value_of_B_l285_28527


namespace coeff_x_squared_l285_28543

theorem coeff_x_squared (n : ℕ) (t h : ℕ)
  (h_t : t = 4^n) 
  (h_h : h = 2^n) 
  (h_sum : t + h = 272)
  (C : ℕ → ℕ → ℕ) -- binomial coefficient notation, we'll skip the direct proof of properties for simplicity
  : (C 4 4) * (3^0) = 1 := 
by 
  /-
  Proof steps (informal, not needed in Lean statement):
  Since the sum of coefficients is t, we have t = 4^n.
  For the sum of binomial coefficients, we have h = 2^n.
  Given t + h = 272, solve for n:
    4^n + 2^n = 272 
    implies 2^n = 16, so n = 4.
  Substitute into the general term (\(T_{r+1}\):
    T_{r+1} = C_4^r * 3^(4-r) * x^((8+r)/6)
  For x^2 term, set (8+r)/6 = 2, yielding r = 4.
  The coefficient is C_4^4 * 3^0 = 1.
  -/
  sorry

end coeff_x_squared_l285_28543


namespace prove_side_c_prove_sin_B_prove_area_circumcircle_l285_28518

-- Define the given conditions
def triangle_ABC (a b A : ℝ) : Prop :=
  a = Real.sqrt 7 ∧ b = 2 ∧ A = Real.pi / 3

-- Prove that side 'c' is equal to 3
theorem prove_side_c (h : triangle_ABC a b A) : c = 3 := by
  sorry

-- Prove that sin B is equal to \frac{\sqrt{21}}{7}
theorem prove_sin_B (h : triangle_ABC a b A) : Real.sin B = Real.sqrt 21 / 7 := by
  sorry

-- Prove that the area of the circumcircle is \frac{7\pi}{3}
theorem prove_area_circumcircle (h : triangle_ABC a b A) (R : ℝ) : 
  let circumcircle_area := Real.pi * R^2
  circumcircle_area = 7 * Real.pi / 3 := by
  sorry

end prove_side_c_prove_sin_B_prove_area_circumcircle_l285_28518


namespace neg_neg_one_eq_one_l285_28561

theorem neg_neg_one_eq_one : -(-1) = 1 :=
by
  sorry

end neg_neg_one_eq_one_l285_28561


namespace system_solution_l285_28523

theorem system_solution (x y: ℝ) 
  (h1: x + y = 2) 
  (h2: 3 * x + y = 4) : 
  x = 1 ∧ y = 1 :=
sorry

end system_solution_l285_28523


namespace expected_value_is_correct_l285_28528

noncomputable def expected_value_max_two_rolls : ℝ :=
  let p_max_1 := (1/6) * (1/6)
  let p_max_2 := (2/6) * (2/6) - (1/6) * (1/6)
  let p_max_3 := (3/6) * (3/6) - (2/6) * (2/6)
  let p_max_4 := (4/6) * (4/6) - (3/6) * (3/6)
  let p_max_5 := (5/6) * (5/6) - (4/6) * (4/6)
  let p_max_6 := 1 - (5/6) * (5/6)
  1 * p_max_1 + 2 * p_max_2 + 3 * p_max_3 + 4 * p_max_4 + 5 * p_max_5 + 6 * p_max_6

theorem expected_value_is_correct :
  expected_value_max_two_rolls = 4.5 :=
sorry

end expected_value_is_correct_l285_28528


namespace least_odd_prime_factor_2027_l285_28556

-- Definitions for the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def order_divides (a n p : ℕ) : Prop := a ^ n % p = 1

-- Define lean function to denote the problem.
theorem least_odd_prime_factor_2027 :
  ∀ p : ℕ, 
  is_prime p → 
  order_divides 2027 12 p ∧ ¬ order_divides 2027 6 p → 
  p ≡ 1 [MOD 12] → 
  2027^6 + 1 % p = 0 → 
  p = 37 :=
by
  -- skipping proof steps
  sorry

end least_odd_prime_factor_2027_l285_28556


namespace solution_1_solution_2_l285_28530

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (a + 1) * x + Real.log x

def critical_point_condition (a x : ℝ) : Prop :=
  (x = 1 / 4) → deriv (f a) x = 0

def pseudo_symmetry_point_condition (a : ℝ) (x0 : ℝ) : Prop :=
  let f' := fun x => 2 * x^2 - 5 * x + Real.log x
  let g := fun x => (4 * x0^2 - 5 * x0 + 1) / x0 * (x - x0) + 2 * x0^2 - 5 * x0 + Real.log x0
  ∀ x : ℝ, 
    (0 < x ∧ x < x0) → (f' x - g x < 0) ∧ 
    (x > x0) → (f' x - g x > 0)

theorem solution_1 (a : ℝ) (h1 : a > 0) (h2 : critical_point_condition a (1/4)) :
  a = 4 := 
sorry

theorem solution_2 (x0 : ℝ) (h1 : x0 = 1/2) :
  pseudo_symmetry_point_condition 4 x0 :=
sorry


end solution_1_solution_2_l285_28530


namespace gcd_max_value_l285_28597

theorem gcd_max_value (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 1005) : ∃ d, d = Int.gcd a b ∧ d = 335 :=
by {
  sorry
}

end gcd_max_value_l285_28597


namespace triangle_side_value_l285_28533

theorem triangle_side_value
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 1)
  (h2 : b = 4)
  (h3 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h4 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) :
  c = Real.sqrt 13 :=
sorry

end triangle_side_value_l285_28533


namespace solution_set_ineq_min_value_sum_l285_28587

-- Part (1)
theorem solution_set_ineq (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| + |x - 2|) :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 2} :=
sorry

-- Part (2)
theorem min_value_sum (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| + |x - 2|)
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (hx : ∀ x, f x ≥ (1 / m) + (1 / n)) :
  m + n = 8 / 3 :=
sorry

end solution_set_ineq_min_value_sum_l285_28587


namespace calculate_value_l285_28507

theorem calculate_value : (3^2 * 5^4 * 7^2) / 7 = 39375 := by
  sorry

end calculate_value_l285_28507


namespace alcohol_percentage_after_additions_l285_28595

/-
Problem statement:
A 40-liter solution of alcohol and water is 5% alcohol. If 4.5 liters of alcohol and 5.5 liters of water are added to this solution, what percent of the solution produced is alcohol?

Conditions:
1. Initial solution volume = 40 liters
2. Initial percentage of alcohol = 5%
3. Volume of alcohol added = 4.5 liters
4. Volume of water added = 5.5 liters

Correct answer:
The percent of the solution that is alcohol after the additions is 13%.
-/

theorem alcohol_percentage_after_additions (initial_volume : ℝ) (initial_percentage : ℝ) 
  (alcohol_added : ℝ) (water_added : ℝ) :
  initial_volume = 40 ∧ initial_percentage = 5 ∧ alcohol_added = 4.5 ∧ water_added = 5.5 →
  ((initial_percentage / 100 * initial_volume + alcohol_added) / (initial_volume + alcohol_added + water_added) * 100) = 13 :=
by simp; sorry

end alcohol_percentage_after_additions_l285_28595


namespace center_of_circle_l285_28589

theorem center_of_circle (x y : ℝ) (h : x^2 + y^2 - 10 * x + 4 * y = -40) : 
  x + y = 3 := 
sorry

end center_of_circle_l285_28589


namespace friends_received_pebbles_l285_28512

-- Define the conditions as expressions
def total_weight_kg : ℕ := 36
def weight_per_pebble_g : ℕ := 250
def pebbles_per_friend : ℕ := 4

-- Convert the total weight from kilograms to grams
def total_weight_g : ℕ := total_weight_kg * 1000

-- Calculate the total number of pebbles
def total_pebbles : ℕ := total_weight_g / weight_per_pebble_g

-- Calculate the total number of friends who received pebbles
def number_of_friends : ℕ := total_pebbles / pebbles_per_friend

-- The theorem to prove the number of friends
theorem friends_received_pebbles : number_of_friends = 36 := by
  sorry

end friends_received_pebbles_l285_28512


namespace find_b9_l285_28544

theorem find_b9 {b : ℕ → ℕ} 
  (h1 : ∀ n, b (n + 2) = b (n + 1) + b n)
  (h2 : b 8 = 100) :
  b 9 = 194 :=
sorry

end find_b9_l285_28544


namespace evaluate_expression_l285_28598

theorem evaluate_expression : (120 / 6 * 2 / 3 = (40 / 3)) := 
by sorry

end evaluate_expression_l285_28598


namespace jacob_fifth_test_score_l285_28549

theorem jacob_fifth_test_score (s1 s2 s3 s4 s5 : ℕ) :
  s1 = 85 ∧ s2 = 79 ∧ s3 = 92 ∧ s4 = 84 ∧ ((s1 + s2 + s3 + s4 + s5) / 5 = 85) →
  s5 = 85 :=
sorry

end jacob_fifth_test_score_l285_28549


namespace positive_integer_divisibility_l285_28545

theorem positive_integer_divisibility :
  ∀ n : ℕ, 0 < n → (5^(n-1) + 3^(n-1) ∣ 5^n + 3^n) → n = 1 :=
by
  sorry

end positive_integer_divisibility_l285_28545


namespace find_average_of_xyz_l285_28502

variable (x y z k : ℝ)

def system_of_equations : Prop :=
  (2 * x + y - z = 26) ∧
  (x + 2 * y + z = 10) ∧
  (x - y + z = k)

theorem find_average_of_xyz (h : system_of_equations x y z k) : 
  (x + y + z) / 3 = (36 + k) / 6 :=
by sorry

end find_average_of_xyz_l285_28502


namespace trains_clear_time_l285_28564

-- Definitions based on conditions
def length_train1 : ℕ := 160
def length_train2 : ℕ := 280
def speed_train1_kmph : ℕ := 42
def speed_train2_kmph : ℕ := 30

-- Conversion factor from km/h to m/s
def kmph_to_mps (s : ℕ) : ℕ := s * 1000 / 3600

-- Computation of relative speed in m/s
def relative_speed_mps : ℕ := kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

-- Total distance to be covered for the trains to clear each other
def total_distance : ℕ := length_train1 + length_train2

-- Time taken for the trains to clear each other
def time_to_clear_each_other : ℕ := total_distance / relative_speed_mps

-- Theorem stating that time taken is 22 seconds
theorem trains_clear_time : time_to_clear_each_other = 22 := by
  sorry

end trains_clear_time_l285_28564


namespace smallest_sum_of_digits_l285_28505

theorem smallest_sum_of_digits :
  ∃ (a b S : ℕ), 
    (100 ≤ a ∧ a < 1000) ∧ 
    (10 ≤ b ∧ b < 100) ∧ 
    (∃ (d1 d2 d3 d4 d5 : ℕ), 
      (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧ 
      (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧ 
      (d3 ≠ d4) ∧ (d3 ≠ d5) ∧ 
      (d4 ≠ d5) ∧ 
      S = a + b ∧ 100 ≤ S ∧ S < 1000 ∧ 
      (∃ (s : ℕ), 
        s = (S / 100) + ((S % 100) / 10) + (S % 10) ∧ 
        s = 3)) :=
sorry

end smallest_sum_of_digits_l285_28505


namespace job_completion_time_l285_28542

theorem job_completion_time (initial_men : ℕ) (initial_days : ℕ) (extra_men : ℕ) (interval_days : ℕ) (total_days : ℕ) : 
  initial_men = 20 → 
  initial_days = 15 → 
  extra_men = 10 → 
  interval_days = 5 → 
  total_days = 12 → 
  ∀ n, (20 * 5 + (20 + 10) * 5 + (20 + 10 + 10) * n.succ = 300 → n + 10 + n.succ = 12) :=
by
  intro h1 h2 h3 h4 h5
  sorry

end job_completion_time_l285_28542


namespace max_marks_l285_28522

theorem max_marks (M : ℝ) (h1 : 80 + 10 = 90) (h2 : 0.30 * M = 90) : M = 300 :=
by
  sorry

end max_marks_l285_28522


namespace speed_in_still_water_l285_28513

-- Define variables for speed of the boy in still water and speed of the stream.
variables (v s : ℝ)

-- Define the conditions as Lean statements
def downstream_condition (v s : ℝ) : Prop := (v + s) * 7 = 91
def upstream_condition (v s : ℝ) : Prop := (v - s) * 7 = 21

-- The theorem to prove that the speed of the boy in still water is 8 km/h given the conditions
theorem speed_in_still_water
  (h1 : downstream_condition v s)
  (h2 : upstream_condition v s) :
  v = 8 := 
sorry

end speed_in_still_water_l285_28513


namespace trey_total_hours_l285_28573

def num_clean_house := 7
def num_shower := 1
def num_make_dinner := 4
def minutes_per_item := 10
def total_items := num_clean_house + num_shower + num_make_dinner
def total_minutes := total_items * minutes_per_item
def minutes_in_hour := 60

theorem trey_total_hours : total_minutes / minutes_in_hour = 2 := by
  sorry

end trey_total_hours_l285_28573


namespace distance_from_point_to_line_l285_28535

open Real

noncomputable def point_to_line_distance (a b c x0 y0 : ℝ) : ℝ :=
  abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2)

theorem distance_from_point_to_line (a b c x0 y0 : ℝ) :
  point_to_line_distance a b c x0 y0 = abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2) :=
by
  sorry

end distance_from_point_to_line_l285_28535


namespace cube_volume_is_27_l285_28584

noncomputable def original_volume (s : ℝ) : ℝ := s^3
noncomputable def new_solid_volume (s : ℝ) : ℝ := (s + 2) * (s + 2) * (s - 2)

theorem cube_volume_is_27 (s : ℝ) (h : original_volume s - new_solid_volume s = 10) :
  original_volume s = 27 :=
by
  sorry

end cube_volume_is_27_l285_28584


namespace motion_of_Q_is_clockwise_with_2ω_l285_28509

variables {ω t : ℝ} {P Q : ℝ × ℝ}

def moving_counterclockwise (P : ℝ × ℝ) (ω t : ℝ) : Prop :=
  P = (Real.cos (ω * t), Real.sin (ω * t))

def motion_of_Q (x y : ℝ): ℝ × ℝ :=
  (-2 * x * y, y^2 - x^2)

def is_on_unit_circle (Q : ℝ × ℝ) : Prop :=
  Q.fst ^ 2 + Q.snd ^ 2 = 1

theorem motion_of_Q_is_clockwise_with_2ω 
  (P : ℝ × ℝ) (ω t : ℝ) (x y : ℝ) :
  moving_counterclockwise P ω t →
  P = (x, y) →
  is_on_unit_circle P →
  is_on_unit_circle (motion_of_Q x y) ∧
  Q = (x, y) →
  Q.fst = Real.cos (2 * ω * t + 3 * Real.pi / 2) ∧ 
  Q.snd = Real.sin (2 * ω * t + 3 * Real.pi / 2) :=
sorry

end motion_of_Q_is_clockwise_with_2ω_l285_28509


namespace miniature_tank_height_l285_28593

-- Given conditions
def actual_tank_height : ℝ := 50
def actual_tank_volume : ℝ := 200000
def model_tank_volume : ℝ := 0.2

-- Theorem: Calculate the height of the miniature water tank
theorem miniature_tank_height :
  (model_tank_volume / actual_tank_volume) ^ (1/3 : ℝ) * actual_tank_height = 0.5 :=
by
  sorry

end miniature_tank_height_l285_28593


namespace common_year_has_52_weeks_1_day_leap_year_has_52_weeks_2_days_next_year_starts_on_wednesday_next_year_starts_on_thursday_l285_28525

-- a) Prove the statements about the number of weeks and extra days
theorem common_year_has_52_weeks_1_day: 
  ∀ (days_in_common_year : ℕ), 
  days_in_common_year = 365 → 
  (days_in_common_year / 7 = 52 ∧ days_in_common_year % 7 = 1)
:= by
  sorry

theorem leap_year_has_52_weeks_2_days: 
  ∀ (days_in_leap_year : ℕ), 
  days_in_leap_year = 366 → 
  (days_in_leap_year / 7 = 52 ∧ days_in_leap_year % 7 = 2)
:= by
  sorry

-- b) If a common year starts on a Tuesday, prove the following year starts on a Wednesday
theorem next_year_starts_on_wednesday: 
  ∀ (start_day : ℕ), 
  start_day = 2 ∧ (365 % 7 = 1) → 
  ((start_day + 365 % 7) % 7 = 3)
:= by
  sorry

-- c) If a leap year starts on a Tuesday, prove the following year starts on a Thursday
theorem next_year_starts_on_thursday: 
  ∀ (start_day : ℕ), 
  start_day = 2 ∧ (366 % 7 = 2) →
  ((start_day + 366 % 7) % 7 = 4)
:= by
  sorry

end common_year_has_52_weeks_1_day_leap_year_has_52_weeks_2_days_next_year_starts_on_wednesday_next_year_starts_on_thursday_l285_28525


namespace jason_hours_saturday_l285_28515

def hours_after_school (x : ℝ) : ℝ := 4 * x
def hours_saturday (y : ℝ) : ℝ := 6 * y

theorem jason_hours_saturday 
  (x y : ℝ) 
  (total_hours : x + y = 18) 
  (total_earnings : 4 * x + 6 * y = 88) : 
  y = 8 :=
by 
  sorry

end jason_hours_saturday_l285_28515


namespace find_prime_A_l285_28586

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_prime_A (A : ℕ) :
  is_prime A ∧ is_prime (A + 14) ∧ is_prime (A + 18) ∧ is_prime (A + 32) ∧ is_prime (A + 36) → A = 5 := by
  sorry

end find_prime_A_l285_28586


namespace borrowed_quarters_l285_28529

def original_quarters : ℕ := 8
def remaining_quarters : ℕ := 5

theorem borrowed_quarters : original_quarters - remaining_quarters = 3 :=
by
  sorry

end borrowed_quarters_l285_28529


namespace flour_qualification_l285_28538

def acceptable_weight_range := {w : ℝ | 24.75 ≤ w ∧ w ≤ 25.25}

theorem flour_qualification :
  (24.80 ∈ acceptable_weight_range) ∧ 
  (24.70 ∉ acceptable_weight_range) ∧ 
  (25.30 ∉ acceptable_weight_range) ∧ 
  (25.51 ∉ acceptable_weight_range) :=
by 
  -- The proof would go here, but we are adding sorry to skip it.
  sorry

end flour_qualification_l285_28538


namespace number_of_functions_satisfying_conditions_l285_28501

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def f_conditions (f : ℕ → ℕ) : Prop :=
  (∀ s ∈ S, f (f (f s)) = s) ∧ (∀ s ∈ S, (f s - s) % 3 ≠ 0)

theorem number_of_functions_satisfying_conditions :
  (∃ (f : ℕ → ℕ), f_conditions f) ∧ (∃! (n : ℕ), n = 288) :=
by
  sorry

end number_of_functions_satisfying_conditions_l285_28501


namespace graduation_graduates_l285_28591

theorem graduation_graduates :
  ∃ G : ℕ, (∀ (chairs_for_parents chairs_for_teachers chairs_for_admins : ℕ),
    chairs_for_parents = 2 * G ∧
    chairs_for_teachers = 20 ∧
    chairs_for_admins = 10 ∧
    G + chairs_for_parents + chairs_for_teachers + chairs_for_admins = 180) ↔ G = 50 :=
by
  sorry

end graduation_graduates_l285_28591


namespace count_multiples_of_30_between_two_multiples_l285_28560

theorem count_multiples_of_30_between_two_multiples : 
  let lower := 900
  let upper := 27000
  let multiple := 30
  let count := (upper / multiple) - (lower / multiple) + 1
  count = 871 :=
by
  let lower := 900
  let upper := 27000
  let multiple := 30
  let count := (upper / multiple) - (lower / multiple) + 1
  sorry

end count_multiples_of_30_between_two_multiples_l285_28560


namespace factor_expression_l285_28599

theorem factor_expression (b : ℝ) :
  (8 * b^4 - 100 * b^3 + 14 * b^2) - (3 * b^4 - 10 * b^3 + 14 * b^2) = 5 * b^3 * (b - 18) :=
by
  sorry

end factor_expression_l285_28599


namespace problem1_problem2_l285_28563

-- Definitions from the conditions
def A (x : ℝ) : Prop := -1 < x ∧ x < 3

def B (x m : ℝ) : Prop := x^2 - 2 * m * x + m^2 - 1 < 0

-- Intersection problem
theorem problem1 (h₁ : ∀ x, A x ↔ (-1 < x ∧ x < 3))
  (h₂ : ∀ x, B x 3 ↔ (2 < x ∧ x < 4)) :
  ∀ x, (A x ∧ B x 3) ↔ (2 < x ∧ x < 3) := by
  sorry

-- Union problem
theorem problem2 (h₃ : ∀ x, A x ↔ (-1 < x ∧ x < 3))
  (h₄ : ∀ x m, B x m ↔ ((x - m)^2 < 1)) :
  ∀ m, (0 ≤ m ∧ m ≤ 2) ↔ (∀ x, A x ∨ B x m → A x) := by
  sorry

end problem1_problem2_l285_28563


namespace sum_of_possible_values_l285_28554

theorem sum_of_possible_values {x : ℝ} :
  (3 * (x - 3)^2 = (x - 2) * (x + 5)) →
  (∃ (x1 x2 : ℝ), x1 + x2 = 10.5) :=
by sorry

end sum_of_possible_values_l285_28554


namespace intervals_of_monotonicity_when_a_eq_2_no_increasing_intervals_on_1_3_implies_a_ge_19_over_6_l285_28583

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + x^2 - 2 * a * x + a^2

-- Question Ⅰ
theorem intervals_of_monotonicity_when_a_eq_2 :
  (∀ x : ℝ, 0 < x ∧ x < (2 - Real.sqrt 2) / 2 → f x 2 > 0) ∧
  (∀ x : ℝ, (2 - Real.sqrt 2) / 2 < x ∧ x < (2 + Real.sqrt 2) / 2 → f x 2 < 0) ∧
  (∀ x : ℝ, (2 + Real.sqrt 2) / 2 < x → f x 2 > 0) := sorry

-- Question Ⅱ
theorem no_increasing_intervals_on_1_3_implies_a_ge_19_over_6 (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 0) → a ≥ (19 / 6) := sorry

end intervals_of_monotonicity_when_a_eq_2_no_increasing_intervals_on_1_3_implies_a_ge_19_over_6_l285_28583


namespace wire_length_between_poles_l285_28571

theorem wire_length_between_poles :
  let d := 18  -- distance between the bottoms of the poles
  let h1 := 6 + 3  -- effective height of the shorter pole
  let h2 := 20  -- height of the taller pole
  let vertical_distance := h2 - h1 -- vertical distance between the tops of the poles
  let hypotenuse := Real.sqrt (d^2 + vertical_distance^2)
  hypotenuse = Real.sqrt 445 :=
by
  sorry

end wire_length_between_poles_l285_28571


namespace simplify_fraction_l285_28596

variable {a b m : ℝ}

theorem simplify_fraction (h : a + b ≠ 0) : (ma/a + b) + (mb/a + b) = m :=
by
  sorry

end simplify_fraction_l285_28596


namespace dealer_cannot_prevent_l285_28567

theorem dealer_cannot_prevent (m n : ℕ) (h : m < 3 * n ∧ n < 3 * m) :
  ∃ (a b : ℕ), (a = 3 * b ∨ b = 3 * a) ∨ (a = 0 ∧ b = 0):=
sorry

end dealer_cannot_prevent_l285_28567
