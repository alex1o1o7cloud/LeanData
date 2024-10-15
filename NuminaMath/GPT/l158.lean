import Mathlib

namespace NUMINAMATH_GPT_angle_mul_add_proof_solve_equation_proof_l158_15804

-- For (1)
def angle_mul_add_example : Prop :=
  let a := 34 * 3600 + 25 * 60 + 20 -- 34°25'20'' to seconds
  let b := 35 * 60 + 42 * 60        -- 35°42' to total minutes
  let result := a * 3 + b * 60      -- Multiply a by 3 and convert b to seconds
  let final_result := result / 3600 -- Convert back to degrees
  final_result = 138 + (58 / 60)

-- For (2)
def solve_equation_example : Prop :=
  ∀ x : ℚ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 → x = 1 / 9

theorem angle_mul_add_proof : angle_mul_add_example := sorry

theorem solve_equation_proof : solve_equation_example := sorry

end NUMINAMATH_GPT_angle_mul_add_proof_solve_equation_proof_l158_15804


namespace NUMINAMATH_GPT_expand_product_l158_15872

theorem expand_product (y : ℝ) : 3 * (y - 6) * (y + 9) = 3 * y^2 + 9 * y - 162 :=
by sorry

end NUMINAMATH_GPT_expand_product_l158_15872


namespace NUMINAMATH_GPT_four_friends_total_fish_l158_15887

-- Define the number of fish each friend has based on the conditions
def micah_fish : ℕ := 7
def kenneth_fish : ℕ := 3 * micah_fish
def matthias_fish : ℕ := kenneth_fish - 15
def total_three_boys_fish : ℕ := micah_fish + kenneth_fish + matthias_fish
def gabrielle_fish : ℕ := 2 * total_three_boys_fish
def total_fish : ℕ := micah_fish + kenneth_fish + matthias_fish + gabrielle_fish

-- The proof goal
theorem four_friends_total_fish : total_fish = 102 :=
by
  -- We assume the proof steps are correct and leave the proof part as sorry
  sorry

end NUMINAMATH_GPT_four_friends_total_fish_l158_15887


namespace NUMINAMATH_GPT_fred_balloons_remaining_l158_15849

theorem fred_balloons_remaining 
    (initial_balloons : ℕ)         -- Fred starts with these many balloons
    (given_to_sandy : ℕ)           -- Fred gives these many balloons to Sandy
    (given_to_bob : ℕ)             -- Fred gives these many balloons to Bob
    (h1 : initial_balloons = 709) 
    (h2 : given_to_sandy = 221) 
    (h3 : given_to_bob = 153) : 
    (initial_balloons - given_to_sandy - given_to_bob = 335) :=
by
  sorry

end NUMINAMATH_GPT_fred_balloons_remaining_l158_15849


namespace NUMINAMATH_GPT_value_of_4_Y_3_l158_15856

def Y (a b : ℕ) : ℕ := (2 * a ^ 2 - 3 * a * b + b ^ 2) ^ 2

theorem value_of_4_Y_3 : Y 4 3 = 25 := by
  sorry

end NUMINAMATH_GPT_value_of_4_Y_3_l158_15856


namespace NUMINAMATH_GPT_percentage_ethanol_in_fuel_B_l158_15852

-- Definitions from the conditions
def tank_capacity : ℝ := 218
def ethanol_percentage_fuel_A : ℝ := 0.12
def total_ethanol : ℝ := 30
def volume_of_fuel_A : ℝ := 122

-- Expression to calculate ethanol in Fuel A
def ethanol_in_fuel_A : ℝ := ethanol_percentage_fuel_A * volume_of_fuel_A

-- The remaining ethanol in Fuel B = Total ethanol - Ethanol in Fuel A
def ethanol_in_fuel_B : ℝ := total_ethanol - ethanol_in_fuel_A

-- The volume of fuel B used to fill the tank
def volume_of_fuel_B : ℝ := tank_capacity - volume_of_fuel_A

-- Statement to prove:
theorem percentage_ethanol_in_fuel_B : (ethanol_in_fuel_B / volume_of_fuel_B) * 100 = 16 :=
sorry

end NUMINAMATH_GPT_percentage_ethanol_in_fuel_B_l158_15852


namespace NUMINAMATH_GPT_distance_T_S_l158_15847

theorem distance_T_S : 
  let P := -14
  let Q := 46
  let S := P + (3 / 4:ℚ) * (Q - P)
  let T := P + (1 / 3:ℚ) * (Q - P)
  S - T = 25 :=
by
  let P := -14
  let Q := 46
  let S := P + (3 / 4:ℚ) * (Q - P)
  let T := P + (1 / 3:ℚ) * (Q - P)
  show S - T = 25
  sorry

end NUMINAMATH_GPT_distance_T_S_l158_15847


namespace NUMINAMATH_GPT_find_PQ_l158_15834

noncomputable def right_triangle_tan (PQ PR : ℝ) (tan_P : ℝ) (R_right : Prop) : Prop :=
  tan_P = PQ / PR ∧ R_right

theorem find_PQ (PQ PR : ℝ) (tan_P : ℝ) (R_right : Prop)
  (h1 : tan_P = 3 / 2)
  (h2 : PR = 6)
  (h3 : R_right) :
  right_triangle_tan PQ PR tan_P R_right → PQ = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_PQ_l158_15834


namespace NUMINAMATH_GPT_travel_paths_l158_15829

-- Definitions for conditions
def roads_AB : ℕ := 3
def roads_BC : ℕ := 2

-- The theorem statement
theorem travel_paths : roads_AB * roads_BC = 6 := by
  sorry

end NUMINAMATH_GPT_travel_paths_l158_15829


namespace NUMINAMATH_GPT_algebraic_expression_value_l158_15800

-- Definitions for the problem conditions
def x := -1
def y := 1 / 2
def expr := 2 * (x^2 - 5 * x * y) - 3 * (x^2 - 6 * x * y)

-- The problem statement to be proved
theorem algebraic_expression_value : expr = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l158_15800


namespace NUMINAMATH_GPT_middle_digit_base5_l158_15850

theorem middle_digit_base5 {M : ℕ} (x y z : ℕ) (hx : 0 ≤ x ∧ x < 5) (hy : 0 ≤ y ∧ y < 5) (hz : 0 ≤ z ∧ z < 5)
    (h_base5 : M = 25 * x + 5 * y + z) (h_base8 : M = 64 * z + 8 * y + x) : y = 0 :=
sorry

end NUMINAMATH_GPT_middle_digit_base5_l158_15850


namespace NUMINAMATH_GPT_pipe_ratio_l158_15899

theorem pipe_ratio (A B : ℝ) (hA : A = 1 / 12) (hAB : A + B = 1 / 3) : B / A = 3 := by
  sorry

end NUMINAMATH_GPT_pipe_ratio_l158_15899


namespace NUMINAMATH_GPT_proof_l158_15842

-- Define the conditions in Lean
variable {f : ℝ → ℝ}
variable (h1 : ∀ x ∈ (Set.Ioi 0), 0 ≤ f x)
variable (h2 : ∀ x ∈ (Set.Ioi 0), x * f x + f x ≤ 0)

-- Formulate the goal
theorem proof (a b : ℝ) (ha : a ∈ (Set.Ioi 0)) (hb : b ∈ (Set.Ioi 0)) (h : a < b) : 
    b * f a ≤ a * f b :=
by
  sorry  -- Proof omitted

end NUMINAMATH_GPT_proof_l158_15842


namespace NUMINAMATH_GPT_vertical_line_division_l158_15876

theorem vertical_line_division (A B C : ℝ × ℝ)
    (hA : A = (0, 2)) (hB : B = (0, 0)) (hC : C = (6, 0))
    (a : ℝ) (h_area_half : 1 / 2 * 6 * 2 / 2 = 3) :
    a = 3 :=
sorry

end NUMINAMATH_GPT_vertical_line_division_l158_15876


namespace NUMINAMATH_GPT_simplify_expression_l158_15846

theorem simplify_expression :
  (3 * Real.sqrt 8) / 
  (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7 + Real.sqrt 9) =
  - (2 * Real.sqrt 6 - 2 * Real.sqrt 2 + 2 * Real.sqrt 14) / 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l158_15846


namespace NUMINAMATH_GPT_calculation_result_l158_15827

theorem calculation_result :
  3 * 15 + 3 * 16 + 3 * 19 + 11 = 161 :=
sorry

end NUMINAMATH_GPT_calculation_result_l158_15827


namespace NUMINAMATH_GPT_value_of_expression_l158_15863

theorem value_of_expression 
  (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 + 9*x2 + 25*x3 + 49*x4 + 81*x5 + 121*x6 + 169*x7 = 2)
  (h2 : 9*x1 + 25*x2 + 49*x3 + 81*x4 + 121*x5 + 169*x6 + 225*x7 = 24)
  (h3 : 25*x1 + 49*x2 + 81*x3 + 121*x4 + 169*x5 + 225*x6 + 289*x7 = 246) : 
  49*x1 + 81*x2 + 121*x3 + 169*x4 + 225*x5 + 289*x6 + 361*x7 = 668 := 
sorry

end NUMINAMATH_GPT_value_of_expression_l158_15863


namespace NUMINAMATH_GPT_different_books_read_l158_15860

theorem different_books_read (t_books d_books b_books td_same all_same : ℕ)
  (h_t_books : t_books = 23)
  (h_d_books : d_books = 12)
  (h_b_books : b_books = 17)
  (h_td_same : td_same = 3)
  (h_all_same : all_same = 1) : 
  t_books + d_books + b_books - (td_same + all_same) = 48 := 
by
  sorry

end NUMINAMATH_GPT_different_books_read_l158_15860


namespace NUMINAMATH_GPT_leo_current_weight_l158_15868

theorem leo_current_weight (L K : ℝ) 
  (h1 : L + 10 = 1.5 * K) 
  (h2 : L + K = 140) : 
  L = 80 :=
by 
  sorry

end NUMINAMATH_GPT_leo_current_weight_l158_15868


namespace NUMINAMATH_GPT_non_swimmers_play_soccer_percentage_l158_15890

theorem non_swimmers_play_soccer_percentage (N : ℕ) (hN_pos : 0 < N)
 (h1 : (0.7 * N : ℝ) = x)
 (h2 : (0.5 * N : ℝ) = y)
 (h3 : (0.6 * x : ℝ) = z)
 : (0.56 * y = 0.28 * N) := 
 sorry

end NUMINAMATH_GPT_non_swimmers_play_soccer_percentage_l158_15890


namespace NUMINAMATH_GPT_average_temperature_l158_15886

theorem average_temperature (temps : List ℕ) (temps_eq : temps = [40, 47, 45, 41, 39]) :
  (temps.sum : ℚ) / temps.length = 42.4 :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_l158_15886


namespace NUMINAMATH_GPT_sasha_prediction_l158_15821

theorem sasha_prediction (n : ℕ) 
  (white_rook_students : ℕ)
  (black_elephant_students : ℕ)
  (total_games : ℕ) :
  white_rook_students = 15 → 
  black_elephant_students = 20 → 
  total_games = 300 → 
  n = 280 → 
  ∃ s : ℕ, s ≤ white_rook_students ∧ s ≤ black_elephant_students ∧ s * black_elephant_students ≥ total_games - n :=
by
  sorry

end NUMINAMATH_GPT_sasha_prediction_l158_15821


namespace NUMINAMATH_GPT_rectangle_area_l158_15880

variable (x y : ℕ)

theorem rectangle_area
  (h1 : (x + 3) * (y - 1) = x * y)
  (h2 : (x - 3) * (y + 2) = x * y) :
  x * y = 36 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_rectangle_area_l158_15880


namespace NUMINAMATH_GPT_participants_won_more_than_lost_l158_15871

-- Define the conditions given in the problem
def total_participants := 64
def rounds := 6

-- Define a function that calculates the number of participants reaching a given round
def participants_after_round (n : Nat) (r : Nat) : Nat :=
  n / (2 ^ r)

-- The theorem we need to prove
theorem participants_won_more_than_lost :
  participants_after_round total_participants 2 = 16 :=
by 
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_participants_won_more_than_lost_l158_15871


namespace NUMINAMATH_GPT_range_of_m_l158_15812

theorem range_of_m (m : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → (x - 1) * (x - (m - 1)) > 0) → m > 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l158_15812


namespace NUMINAMATH_GPT_find_x_range_l158_15892

theorem find_x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) (h3 : 2 * x - 5 > 0) : x > 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_range_l158_15892


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l158_15817

-- Definitions of arithmetic and geometric sequences
def arithmetic (a_n : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a_n n = a_n 0 + n * d
def geometric (b_n : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, b_n n = b_n 0 * q ^ n
def E (m p r : ℕ) := m < p ∧ p < r
def common_difference_greater_than_one (m p r : ℕ) := (p - m = r - p) ∧ (p - m > 1)

-- Problem (1)
theorem problem1 (a_n b_n : ℕ → ℝ) (d q : ℝ) (h₁: arithmetic a_n d) (h₂: geometric b_n q) (hne: q ≠ 1 ∧ q ≠ -1)
  (h: a_n 0 + b_n 1 = a_n 1 + b_n 2 ∧ a_n 1 + b_n 2 = a_n 2 + b_n 0) :
  q = -1/2 :=
sorry

-- Problem (2)
theorem problem2 (a_n b_n : ℕ → ℝ) (d q : ℝ) (m p r : ℕ) (h₁: arithmetic a_n d) (h₂: geometric b_n q) (hne: q ≠ 1 ∧ q ≠ -1)
  (hE: E m p r) (hDiff: common_difference_greater_than_one m p r)
  (h: a_n m + b_n p = a_n p + b_n r ∧ a_n p + b_n r = a_n r + b_n m) :
  q = - (1/2)^(1/3) :=
sorry

-- Problem (3)
theorem problem3 (a_n b_n : ℕ → ℝ) (m p r : ℕ) (hE: E m p r)
  (hG: ∀ n : ℕ, b_n n = (-1/2)^((n:ℕ)-1)) (h: a_n m + b_n m = 0 ∧ a_n p + b_n p = 0 ∧ a_n r + b_n r = 0) :
  ∃ (E : ℕ × ℕ × ℕ) (a : ℕ → ℝ), (E = ⟨1, 3, 4⟩ ∧ ∀ n : ℕ, a n = 3/8 * n - 11/8) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l158_15817


namespace NUMINAMATH_GPT_avg_width_is_3_5_l158_15889

def book_widths : List ℚ := [4, (3/4), 1.25, 3, 2, 7, 5.5]

noncomputable def average (l : List ℚ) : ℚ :=
  l.sum / l.length

theorem avg_width_is_3_5 : average book_widths = 23.5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_avg_width_is_3_5_l158_15889


namespace NUMINAMATH_GPT_cups_of_sugar_already_put_in_l158_15851

-- Defining the given conditions
variable (f s x : ℕ)

-- The total flour and sugar required
def total_flour_required := 9
def total_sugar_required := 6

-- Mary needs to add 7 more cups of flour than cups of sugar
def remaining_flour_to_sugar_difference := 7

-- Proof goal: to find how many cups of sugar Mary has already put in
theorem cups_of_sugar_already_put_in (total_flour_remaining : ℕ := 9 - 7)
    (remaining_sugar : ℕ := 9 - 7) 
    (already_added_sugar : ℕ := 6 - 2) : already_added_sugar = 4 :=
by sorry

end NUMINAMATH_GPT_cups_of_sugar_already_put_in_l158_15851


namespace NUMINAMATH_GPT_other_root_of_quadratic_l158_15875

theorem other_root_of_quadratic (a b c : ℚ) (x₁ x₂ : ℚ) :
  a ≠ 0 →
  x₁ = 4 / 9 →
  (a * x₁^2 + b * x₁ + c = 0) →
  (a = 81) →
  (b = -145) →
  (c = 64) →
  x₂ = -16 / 9
:=
sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l158_15875


namespace NUMINAMATH_GPT_B_finishes_work_in_4_days_l158_15818

-- Define the work rates of A and B
def work_rate_A : ℚ := 1 / 5
def work_rate_B : ℚ := 1 / 10

-- Combined work rate when A and B work together
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Work done by A and B in 2 days
def work_done_in_2_days : ℚ := 2 * combined_work_rate

-- Remaining work after 2 days
def remaining_work : ℚ := 1 - work_done_in_2_days

-- Time B needs to finish the remaining work
def time_for_B_to_finish_remaining_work : ℚ := remaining_work / work_rate_B

theorem B_finishes_work_in_4_days : time_for_B_to_finish_remaining_work = 4 := by
  sorry

end NUMINAMATH_GPT_B_finishes_work_in_4_days_l158_15818


namespace NUMINAMATH_GPT_absent_children_count_l158_15864

theorem absent_children_count : ∀ (total_children present_children absent_children bananas : ℕ), 
  total_children = 260 → 
  bananas = 2 * total_children → 
  bananas = 4 * present_children → 
  present_children + absent_children = total_children →
  absent_children = 130 :=
by
  intros total_children present_children absent_children bananas h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_absent_children_count_l158_15864


namespace NUMINAMATH_GPT_largest_three_digit_divisible_by_6_l158_15837

-- Defining what it means for a number to be divisible by 6, 2, and 3
def divisible_by (n d : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Conditions extracted from the problem
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def last_digit_even (n : ℕ) : Prop := (n % 10) % 2 = 0
def sum_of_digits_divisible_by_3 (n : ℕ) : Prop := ((n / 100) + (n / 10 % 10) + (n % 10)) % 3 = 0

-- Define what it means for a number to be divisible by 6 according to the conditions
def divisible_by_6 (n : ℕ) : Prop := last_digit_even n ∧ sum_of_digits_divisible_by_3 n

-- Prove that 996 is the largest three-digit number that satisfies these conditions
theorem largest_three_digit_divisible_by_6 (n : ℕ) : is_three_digit n ∧ divisible_by_6 n → n ≤ 996 :=
by
    sorry

end NUMINAMATH_GPT_largest_three_digit_divisible_by_6_l158_15837


namespace NUMINAMATH_GPT_coins_value_percentage_l158_15805

theorem coins_value_percentage :
  let penny_value := 1
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let total_value_cents := (1 * penny_value) + (2 * nickel_value) + (1 * dime_value) + (2 * quarter_value)
  (total_value_cents / 100) * 100 = 71 :=
by
  sorry

end NUMINAMATH_GPT_coins_value_percentage_l158_15805


namespace NUMINAMATH_GPT_harry_total_cost_l158_15845

noncomputable def total_cost : ℝ :=
let small_price := 10
let medium_price := 12
let large_price := 14
let small_topping_price := 1.50
let medium_topping_price := 1.75
let large_topping_price := 2
let small_pizzas := 1
let medium_pizzas := 2
let large_pizzas := 1
let small_toppings := 2
let medium_toppings := 3
let large_toppings := 4
let item_cost : ℝ := (small_pizzas * small_price + medium_pizzas * medium_price + large_pizzas * large_price)
let topping_cost : ℝ := 
  (small_pizzas * small_toppings * small_topping_price) + 
  (medium_pizzas * medium_toppings * medium_topping_price) +
  (large_pizzas * large_toppings * large_topping_price)
let garlic_knots := 2 * 3 -- 2 sets of 5 knots at $3 each
let soda := 2
let replace_total := item_cost + topping_cost
let discounted_total := replace_total - 0.1 * item_cost
let subtotal := discounted_total + garlic_knots + soda
let tax := 0.08 * subtotal
let total_with_tax := subtotal + tax
let tip := 0.25 * total_with_tax
total_with_tax + tip

theorem harry_total_cost : total_cost = 98.15 := by
  sorry

end NUMINAMATH_GPT_harry_total_cost_l158_15845


namespace NUMINAMATH_GPT_range_of_x_l158_15841

theorem range_of_x (x : ℝ) : (|x + 1| + |x - 1| = 2) → (-1 ≤ x ∧ x ≤ 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_x_l158_15841


namespace NUMINAMATH_GPT_number_of_female_students_selected_is_20_l158_15813

noncomputable def number_of_female_students_to_be_selected
(total_students : ℕ) (female_students : ℕ) (students_to_be_selected : ℕ) : ℕ :=
students_to_be_selected * female_students / total_students

theorem number_of_female_students_selected_is_20 :
  number_of_female_students_to_be_selected 2000 800 50 = 20 := 
by
  sorry

end NUMINAMATH_GPT_number_of_female_students_selected_is_20_l158_15813


namespace NUMINAMATH_GPT_percentage_loss_l158_15885

theorem percentage_loss (CP SP : ℝ) (h₁ : CP = 1400) (h₂ : SP = 1232) :
  ((CP - SP) / CP) * 100 = 12 :=
by
  sorry

end NUMINAMATH_GPT_percentage_loss_l158_15885


namespace NUMINAMATH_GPT_regular_tickets_sold_l158_15830

variables (S R : ℕ) (h1 : S + R = 65) (h2 : 10 * S + 15 * R = 855)

theorem regular_tickets_sold : R = 41 :=
sorry

end NUMINAMATH_GPT_regular_tickets_sold_l158_15830


namespace NUMINAMATH_GPT_pencils_per_box_l158_15855

-- Variables and Definitions based on the problem conditions
def num_boxes : ℕ := 10
def pencils_kept : ℕ := 10
def friends : ℕ := 5
def pencils_per_friend : ℕ := 8

-- Theorem to prove the solution
theorem pencils_per_box (pencils_total : ℕ)
  (h1 : pencils_total = pencils_kept + (friends * pencils_per_friend))
  (h2 : pencils_total = num_boxes * (pencils_total / num_boxes)) :
  (pencils_total / num_boxes) = 5 :=
sorry

end NUMINAMATH_GPT_pencils_per_box_l158_15855


namespace NUMINAMATH_GPT_ratio_boys_to_girls_l158_15839

variable (g b : ℕ)

theorem ratio_boys_to_girls (h1 : b = g + 9) (h2 : g + b = 25) : b / g = 17 / 8 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_boys_to_girls_l158_15839


namespace NUMINAMATH_GPT_fruit_trees_l158_15857

theorem fruit_trees (total_streets : ℕ) 
  (fruit_trees_every_other : total_streets % 2 = 0) 
  (equal_fruit_trees : ∀ n : ℕ, 3 * n = total_streets / 2) : 
  ∃ n : ℕ, n = total_streets / 6 :=
by
  sorry

end NUMINAMATH_GPT_fruit_trees_l158_15857


namespace NUMINAMATH_GPT_nephews_count_l158_15806

theorem nephews_count (a_nephews_20_years_ago : ℕ) (third_now_nephews : ℕ) (additional_nephews : ℕ) :
  a_nephews_20_years_ago = 80 →
  third_now_nephews = 3 →
  additional_nephews = 120 →
  ∃ (a_nephews_now : ℕ) (v_nephews_now : ℕ), a_nephews_now = third_now_nephews * a_nephews_20_years_ago ∧ v_nephews_now = a_nephews_now + additional_nephews ∧ (a_nephews_now + v_nephews_now = 600) :=
by
  sorry

end NUMINAMATH_GPT_nephews_count_l158_15806


namespace NUMINAMATH_GPT_percent_dimes_value_is_60_l158_15835

variable (nickels dimes : ℕ)
variable (value_nickel value_dime : ℕ)
variable (num_nickels num_dimes : ℕ)

def total_value (n d : ℕ) (v_n v_d : ℕ) := n * v_n + d * v_d

def percent_value_dimes (total d_value : ℕ) := (d_value * 100) / total

theorem percent_dimes_value_is_60 :
  num_nickels = 40 →
  num_dimes = 30 →
  value_nickel = 5 →
  value_dime = 10 →
  percent_value_dimes (total_value num_nickels num_dimes value_nickel value_dime) (num_dimes * value_dime) = 60 := 
by sorry

end NUMINAMATH_GPT_percent_dimes_value_is_60_l158_15835


namespace NUMINAMATH_GPT_price_per_glass_first_day_l158_15822

theorem price_per_glass_first_day (O W : ℝ) (P1 P2 : ℝ) 
  (h1 : O = W) 
  (h2 : P2 = 0.40)
  (revenue_eq : 2 * O * P1 = 3 * O * P2) 
  : P1 = 0.60 := 
by 
  sorry

end NUMINAMATH_GPT_price_per_glass_first_day_l158_15822


namespace NUMINAMATH_GPT_area_of_square_with_diagonal_l158_15840

theorem area_of_square_with_diagonal (d : ℝ) (s : ℝ) (hsq : d = s * Real.sqrt 2) (hdiagonal : d = 12 * Real.sqrt 2) : 
  s^2 = 144 :=
by
  -- Proof details would go here.
  sorry

end NUMINAMATH_GPT_area_of_square_with_diagonal_l158_15840


namespace NUMINAMATH_GPT_longer_diagonal_of_rhombus_l158_15858

theorem longer_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (h₁ : d1 = 12) (h₂ : area = 120) :
  d2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_longer_diagonal_of_rhombus_l158_15858


namespace NUMINAMATH_GPT_expression_equals_36_l158_15802

def k := 13

theorem expression_equals_36 : 13 * (3 - 3 / 13) = 36 := by
  sorry

end NUMINAMATH_GPT_expression_equals_36_l158_15802


namespace NUMINAMATH_GPT_simplify_expression_l158_15823

noncomputable def a : ℝ := 2 * Real.sqrt 12 - 4 * Real.sqrt 27 + 3 * Real.sqrt 75 + 7 * Real.sqrt 8 - 3 * Real.sqrt 18
noncomputable def b : ℝ := 4 * Real.sqrt 48 - 3 * Real.sqrt 27 - 5 * Real.sqrt 18 + 2 * Real.sqrt 50

theorem simplify_expression : a * b = 97 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l158_15823


namespace NUMINAMATH_GPT_radius_of_sphere_with_surface_area_4pi_l158_15888

noncomputable def sphere_radius (surface_area: ℝ) : ℝ :=
  sorry

theorem radius_of_sphere_with_surface_area_4pi :
  sphere_radius (4 * Real.pi) = 1 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_sphere_with_surface_area_4pi_l158_15888


namespace NUMINAMATH_GPT_upper_limit_b_l158_15859

theorem upper_limit_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : 3 < b) (h4 : (a : ℚ) / b ≤ 3.75) : b ≤ 4 := by
  sorry

end NUMINAMATH_GPT_upper_limit_b_l158_15859


namespace NUMINAMATH_GPT_gumballs_in_packages_l158_15810

theorem gumballs_in_packages (total_gumballs : ℕ) (gumballs_per_package : ℕ) (h1 : total_gumballs = 20) (h2 : gumballs_per_package = 5) :
  total_gumballs / gumballs_per_package = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_gumballs_in_packages_l158_15810


namespace NUMINAMATH_GPT_first_month_sale_l158_15815

theorem first_month_sale (sales_2 : ℕ) (sales_3 : ℕ) (sales_4 : ℕ) (sales_5 : ℕ) (sales_6 : ℕ) (average_sale : ℕ) (total_months : ℕ)
  (H_sales_2 : sales_2 = 6927)
  (H_sales_3 : sales_3 = 6855)
  (H_sales_4 : sales_4 = 7230)
  (H_sales_5 : sales_5 = 6562)
  (H_sales_6 : sales_6 = 5591)
  (H_average_sale : average_sale = 6600)
  (H_total_months : total_months = 6) :
  ∃ (sale_1 : ℕ), sale_1 = 6435 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_first_month_sale_l158_15815


namespace NUMINAMATH_GPT_ferry_routes_ratio_l158_15895

theorem ferry_routes_ratio :
  ∀ (D_P D_Q : ℝ) (speed_P time_P speed_Q time_Q : ℝ),
  speed_P = 8 →
  time_P = 3 →
  speed_Q = speed_P + 4 →
  time_Q = time_P + 1 →
  D_P = speed_P * time_P →
  D_Q = speed_Q * time_Q →
  D_Q / D_P = 2 :=
by sorry

end NUMINAMATH_GPT_ferry_routes_ratio_l158_15895


namespace NUMINAMATH_GPT_inequality_power_cubed_l158_15893

theorem inequality_power_cubed
  (x y a : ℝ)
  (h_condition : (0 < a ∧ a < 1) ∧ a ^ x < a ^ y) : x^3 > y^3 :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_power_cubed_l158_15893


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l158_15861

open Set

-- Definition of set A
def A : Set ℤ := {1, 2, 3}

-- Definition of set B
def B : Set ℤ := {x | x < -1 ∨ 0 < x ∧ x < 2}

-- The theorem to prove A ∩ B = {1}
theorem intersection_of_A_and_B : A ∩ B = {1} := by
  -- Proof logic here
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l158_15861


namespace NUMINAMATH_GPT_hyperbola_properties_l158_15803

theorem hyperbola_properties :
  let h := -3
  let k := 0
  let a := 5
  let c := Real.sqrt 50
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ h + k + a + b = 7 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_properties_l158_15803


namespace NUMINAMATH_GPT_sum_of_three_integers_l158_15867

def three_positive_integers (x y z : ℕ) : Prop :=
  x + y = 2003 ∧ y - z = 1000

theorem sum_of_three_integers (x y z : ℕ) (h1 : x + y = 2003) (h2 : y - z = 1000) (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) : 
  x + y + z = 2004 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_three_integers_l158_15867


namespace NUMINAMATH_GPT_gcd_217_155_l158_15891

theorem gcd_217_155 : Nat.gcd 217 155 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_217_155_l158_15891


namespace NUMINAMATH_GPT_airplane_seat_difference_l158_15833

theorem airplane_seat_difference (F C X : ℕ) 
    (h1 : 387 = F + 310) 
    (h2 : C = 310) 
    (h3 : C = 4 * F + X) :
    X = 2 :=
by
    sorry

end NUMINAMATH_GPT_airplane_seat_difference_l158_15833


namespace NUMINAMATH_GPT_calculate_decimal_sum_and_difference_l158_15819

theorem calculate_decimal_sum_and_difference : 
  (0.5 + 0.003 + 0.070) - 0.008 = 0.565 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_decimal_sum_and_difference_l158_15819


namespace NUMINAMATH_GPT_sum_series_and_convergence_l158_15869

theorem sum_series_and_convergence (x : ℝ) (h : -1 < x ∧ x < 1) :
  ∑' n, (n + 6) * x^(7 * n) = (6 - 5 * x^7) / (1 - x^7)^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_and_convergence_l158_15869


namespace NUMINAMATH_GPT_unique_solution_quadratic_l158_15894

theorem unique_solution_quadratic (x : ℚ) (b : ℚ) (h_b_nonzero : b ≠ 0) (h_discriminant_zero : 625 - 36 * b = 0) : 
  (b = 625 / 36) ∧ (x = -18 / 25) → b * x^2 + 25 * x + 9 = 0 :=
by 
  -- We assume b = 625 / 36 and x = -18 / 25
  rintro ⟨hb, hx⟩
  -- Substitute b and x into the quadratic equation and simplify
  rw [hb, hx]
  -- Show the left-hand side evaluates to zero
  sorry

end NUMINAMATH_GPT_unique_solution_quadratic_l158_15894


namespace NUMINAMATH_GPT_trapezoidal_section_length_l158_15824

theorem trapezoidal_section_length 
  (total_area : ℝ) 
  (rectangular_area : ℝ) 
  (parallel_side1 : ℝ) 
  (parallel_side2 : ℝ) 
  (trapezoidal_area : ℝ)
  (H1 : total_area = 55)
  (H2 : rectangular_area = 30)
  (H3 : parallel_side1 = 3)
  (H4 : parallel_side2 = 6)
  (H5 : trapezoidal_area = total_area - rectangular_area) :
  (trapezoidal_area = 25) → 
  (1/2 * (parallel_side1 + parallel_side2) * L = trapezoidal_area) →
  L = 25 / 4.5 :=
by
  sorry

end NUMINAMATH_GPT_trapezoidal_section_length_l158_15824


namespace NUMINAMATH_GPT_expected_value_is_one_dollar_l158_15838

def star_prob := 1 / 4
def moon_prob := 1 / 2
def sun_prob := 1 / 4

def star_prize := 2
def moon_prize := 4
def sun_penalty := -6

def expected_winnings := star_prob * star_prize + moon_prob * moon_prize + sun_prob * sun_penalty

theorem expected_value_is_one_dollar : expected_winnings = 1 := by
  sorry

end NUMINAMATH_GPT_expected_value_is_one_dollar_l158_15838


namespace NUMINAMATH_GPT_canvas_decreased_by_40_percent_l158_15844

noncomputable def canvas_decrease (P C : ℝ) (x d : ℝ) : Prop :=
  (P = 4 * C) ∧
  ((P - 0.60 * P) + (C - (x / 100) * C) = (1 - d / 100) * (P + C)) ∧
  (d = 55.99999999999999)

theorem canvas_decreased_by_40_percent (P C : ℝ) (x d : ℝ) 
  (h : canvas_decrease P C x d) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_canvas_decreased_by_40_percent_l158_15844


namespace NUMINAMATH_GPT_correct_result_value_at_neg_one_l158_15801

theorem correct_result (x : ℝ) (A : ℝ := 3 * x^2 - x + 1) (incorrect : ℝ := 2 * x^2 - 3 * x - 2) :
  (A - (incorrect - A)) = 4 * x^2 + x + 4 :=
by sorry

theorem value_at_neg_one (x : ℝ := -1) (A : ℝ := 3 * x^2 - x + 1) (incorrect : ℝ := 2 * x^2 - 3 * x - 2) :
  (4 * x^2 + x + 4) = 7 :=
by sorry

end NUMINAMATH_GPT_correct_result_value_at_neg_one_l158_15801


namespace NUMINAMATH_GPT_prove_ab_ge_5_l158_15898

theorem prove_ab_ge_5 (a b c : ℕ) (h : ∀ x, x * (a * x) = b * x + c → 0 ≤ x ∧ x ≤ 1) : 5 ≤ a ∧ 5 ≤ b := 
sorry

end NUMINAMATH_GPT_prove_ab_ge_5_l158_15898


namespace NUMINAMATH_GPT_proof_problem_l158_15884

-- Define sets A and B according to the given conditions
def A : Set ℝ := { x | x ≥ -1 }
def B : Set ℝ := { x | x > 2 }
def complement_B : Set ℝ := { x | ¬ (x > 2) }  -- Complement of B

-- Remaining intersection expression
def intersect_expr : Set ℝ := { x | x ≥ -1 ∧ x ≤ 2 }

-- Statement to prove
theorem proof_problem : (A ∩ complement_B) = intersect_expr :=
sorry

end NUMINAMATH_GPT_proof_problem_l158_15884


namespace NUMINAMATH_GPT_common_divisors_count_48_80_l158_15897

noncomputable def prime_factors_48 : Nat -> Prop
| n => n = 48

noncomputable def prime_factors_80 : Nat -> Prop
| n => n = 80

theorem common_divisors_count_48_80 :
  let gcd_48_80 := 2^4
  let divisors_of_gcd := [1, 2, 4, 8, 16]
  prime_factors_48 48 ∧ prime_factors_80 80 →
  List.length divisors_of_gcd = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_common_divisors_count_48_80_l158_15897


namespace NUMINAMATH_GPT_right_triangle_area_l158_15832

theorem right_triangle_area
    (h : ∀ {a b c : ℕ}, a^2 + b^2 = c^2 → c = 13 → a = 5 ∨ b = 5)
    (hypotenuse : ℕ)
    (leg : ℕ)
    (hypotenuse_eq : hypotenuse = 13)
    (leg_eq : leg = 5) : ∃ (area: ℕ), area = 30 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_right_triangle_area_l158_15832


namespace NUMINAMATH_GPT_range_of_x_l158_15807

def f (x : ℝ) : ℝ := abs (x - 2)

theorem range_of_x (a b x : ℝ) (a_nonzero : a ≠ 0) (ab_real : a ∈ Set.univ ∧ b ∈ Set.univ) : 
  (|a + b| + |a - b| ≥ |a| • f x) ↔ (0 ≤ x ∧ x ≤ 4) :=
sorry

end NUMINAMATH_GPT_range_of_x_l158_15807


namespace NUMINAMATH_GPT_find_other_number_l158_15848

-- Given: 
-- LCM of two numbers is 2310
-- GCD of two numbers is 55
-- One number is 605,
-- Prove: The other number is 210

theorem find_other_number (a b : ℕ) (h_lcm : Nat.lcm a b = 2310) (h_gcd : Nat.gcd a b = 55) (h_b : b = 605) :
  a = 210 :=
sorry

end NUMINAMATH_GPT_find_other_number_l158_15848


namespace NUMINAMATH_GPT_multiple_of_9_l158_15828

noncomputable def digit_sum (x : ℕ) : ℕ := sorry  -- Placeholder for the digit sum function

theorem multiple_of_9 (n : ℕ) (h1 : digit_sum n = digit_sum (3 * n))
  (h2 : ∀ x, x % 9 = digit_sum x % 9) :
  n % 9 = 0 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_9_l158_15828


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l158_15814

-- Let p be the proposition |x| < 2
def p (x : ℝ) : Prop := abs x < 2

-- Let q be the proposition x^2 - x - 2 < 0
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- The proof statement
theorem necessary_but_not_sufficient_condition (x : ℝ) : q x → p x ∧ ¬ (p x → q x) := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l158_15814


namespace NUMINAMATH_GPT_stratified_sampling_community_A_l158_15826

theorem stratified_sampling_community_A :
  let A_households := 360
  let B_households := 270
  let C_households := 180
  let total_households := A_households + B_households + C_households
  let total_units := 90
  (A_households : ℕ) / total_households * total_units = 40 :=
by
  let A_households := 360
  let B_households := 270
  let C_households := 180
  let total_households := A_households + B_households + C_households
  let total_units := 90
  have : total_households = 810 := by sorry
  have : (A_households : ℕ) / total_households * total_units = 40 := by sorry
  exact this

end NUMINAMATH_GPT_stratified_sampling_community_A_l158_15826


namespace NUMINAMATH_GPT_solution_set_l158_15882

open Nat

def is_solution (a b c : ℕ) : Prop :=
  a ^ (b + 20) * (c - 1) = c ^ (b + 21) - 1

theorem solution_set (a b c : ℕ) : 
  (is_solution a b c) ↔ ((c = 0 ∧ a = 1) ∨ (c = 1)) := 
sorry

end NUMINAMATH_GPT_solution_set_l158_15882


namespace NUMINAMATH_GPT_correct_equation_for_annual_consumption_l158_15879

-- Definitions based on the problem conditions
-- average_monthly_consumption_first_half is the average monthly electricity consumption in the first half of the year, assumed to be x
def average_monthly_consumption_first_half (x : ℝ) := x

-- average_monthly_consumption_second_half is the average monthly consumption in the second half of the year, i.e., x - 2000
def average_monthly_consumption_second_half (x : ℝ) := x - 2000

-- total_annual_consumption is the total annual electricity consumption which is 150000 kWh
def total_annual_consumption (x : ℝ) := 6 * average_monthly_consumption_first_half x + 6 * average_monthly_consumption_second_half x

-- The main theorem statement which we need to prove
theorem correct_equation_for_annual_consumption (x : ℝ) : total_annual_consumption x = 150000 :=
by
  -- equation derivation
  sorry

end NUMINAMATH_GPT_correct_equation_for_annual_consumption_l158_15879


namespace NUMINAMATH_GPT_exceeds_500_bacteria_l158_15874

noncomputable def bacteria_count (n : Nat) : Nat :=
  4 * 3^n

theorem exceeds_500_bacteria (n : Nat) (h : 4 * 3^n > 500) : n ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_exceeds_500_bacteria_l158_15874


namespace NUMINAMATH_GPT_intersection_line_l158_15866

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 3*x - y = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y = 0

-- Define the line that we need to prove as the intersection
def line (x y : ℝ) : Prop := x - 2*y = 0

-- The theorem to prove
theorem intersection_line (x y : ℝ) : circle1 x y ∧ circle2 x y → line x y :=
by
  sorry

end NUMINAMATH_GPT_intersection_line_l158_15866


namespace NUMINAMATH_GPT_range_of_a_l158_15831

noncomputable def is_decreasing (a : ℝ) : Prop :=
∀ n : ℕ, 0 < n → n ≤ 6 → (1 - 3 * a) * n + 10 * a > (1 - 3 * a) * (n + 1) + 10 * a ∧ 0 < a ∧ a < 1 ∧ ((1 - 3 * a) * 6 + 10 * a > 1)

theorem range_of_a (a : ℝ) : is_decreasing a ↔ (1/3 < a ∧ a < 5/8) :=
sorry

end NUMINAMATH_GPT_range_of_a_l158_15831


namespace NUMINAMATH_GPT_prob_iff_eq_l158_15877

noncomputable def A (m : ℝ) : Set ℝ := { x | x^2 + m * x + 2 ≥ 0 ∧ x ≥ 0 }
noncomputable def B (m : ℝ) : Set ℝ := { y | ∃ x, x ∈ A m ∧ y = Real.sqrt (x^2 + m * x + 2) }

theorem prob_iff_eq (m : ℝ) : (A m = { y | ∃ x, x ^ 2 + m * x + 2 = y ^ 2 ∧ x ≥ 0 } ↔ m = -2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_prob_iff_eq_l158_15877


namespace NUMINAMATH_GPT_average_percentage_l158_15896

theorem average_percentage (n1 n2 : ℕ) (s1 s2 : ℕ)
  (h1 : n1 = 15) (h2 : s1 = 80) (h3 : n2 = 10) (h4 : s2 = 90) :
  (n1 * s1 + n2 * s2) / (n1 + n2) = 84 :=
by
  sorry

end NUMINAMATH_GPT_average_percentage_l158_15896


namespace NUMINAMATH_GPT_pq_sufficient_not_necessary_l158_15816

theorem pq_sufficient_not_necessary (p q : Prop) :
  (¬ (p ∨ q)) → (¬ p ∧ ¬ q) ∧ ¬ ((¬ p ∧ ¬ q) → (¬ (p ∨ q))) :=
sorry

end NUMINAMATH_GPT_pq_sufficient_not_necessary_l158_15816


namespace NUMINAMATH_GPT_find_daily_rate_second_company_l158_15883

def daily_rate_second_company (x : ℝ) : Prop :=
  let total_cost_1 := 21.95 + 0.19 * 150
  let total_cost_2 := x + 0.21 * 150
  total_cost_1 = total_cost_2

theorem find_daily_rate_second_company : daily_rate_second_company 18.95 :=
  by
  unfold daily_rate_second_company
  sorry

end NUMINAMATH_GPT_find_daily_rate_second_company_l158_15883


namespace NUMINAMATH_GPT_total_fencing_cost_l158_15809

-- Definitions of the given conditions
def length : ℝ := 57
def breadth : ℝ := length - 14
def cost_per_meter : ℝ := 26.50

-- Definition of the total cost calculation
def total_cost : ℝ := 2 * (length + breadth) * cost_per_meter

-- Statement of the theorem to be proved
theorem total_fencing_cost :
  total_cost = 5300 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_fencing_cost_l158_15809


namespace NUMINAMATH_GPT_future_skyscraper_climb_proof_l158_15862

variable {H_f H_c H_fut : ℝ}

theorem future_skyscraper_climb_proof
  (H_f : ℝ)
  (H_c : ℝ := 3 * H_f)
  (H_fut : ℝ := 1.25 * H_c)
  (T_f : ℝ := 1) :
  (H_fut * T_f / H_f) > 2 * T_f :=
by
  -- specific calculations would go here
  sorry

end NUMINAMATH_GPT_future_skyscraper_climb_proof_l158_15862


namespace NUMINAMATH_GPT_randi_peter_ratio_l158_15843

-- Given conditions
def ray_cents := 175
def cents_per_nickel := 5
def peter_cents := 30
def randi_extra_nickels := 6

-- Define the nickels Ray has
def ray_nickels := ray_cents / cents_per_nickel
-- Define the nickels Peter receives
def peter_nickels := peter_cents / cents_per_nickel
-- Define the nickels Randi receives
def randi_nickels := peter_nickels + randi_extra_nickels
-- Define the cents Randi receives
def randi_cents := randi_nickels * cents_per_nickel

-- The goal is to prove the ratio of the cents given to Randi to the cents given to Peter is 2.
theorem randi_peter_ratio : randi_cents / peter_cents = 2 := by
  sorry

end NUMINAMATH_GPT_randi_peter_ratio_l158_15843


namespace NUMINAMATH_GPT_highest_probability_two_out_of_three_probability_l158_15836

structure Student :=
  (name : String)
  (P_T : ℚ)  -- Probability of passing the theoretical examination
  (P_S : ℚ)  -- Probability of passing the social practice examination

noncomputable def P_earn (student : Student) : ℚ :=
  student.P_T * student.P_S

def student_A := Student.mk "A" (5 / 6) (1 / 2)
def student_B := Student.mk "B" (4 / 5) (2 / 3)
def student_C := Student.mk "C" (3 / 4) (5 / 6)

theorem highest_probability : 
  P_earn student_C > P_earn student_B ∧ P_earn student_B > P_earn student_A :=
by sorry

theorem two_out_of_three_probability :
  (1 - P_earn student_A) * P_earn student_B * P_earn student_C +
  P_earn student_A * (1 - P_earn student_B) * P_earn student_C +
  P_earn student_A * P_earn student_B * (1 - P_earn student_C) =
  115 / 288 :=
by sorry

end NUMINAMATH_GPT_highest_probability_two_out_of_three_probability_l158_15836


namespace NUMINAMATH_GPT_finance_to_manufacturing_ratio_l158_15873

theorem finance_to_manufacturing_ratio : 
    let finance_angle := 72
    let manufacturing_angle := 108
    (finance_angle:ℕ) / (Nat.gcd finance_angle manufacturing_angle) = 2 ∧ 
    (manufacturing_angle:ℕ) / (Nat.gcd finance_angle manufacturing_angle) = 3 := 
by 
    sorry

end NUMINAMATH_GPT_finance_to_manufacturing_ratio_l158_15873


namespace NUMINAMATH_GPT_value_of_z_plus_one_over_y_l158_15878

theorem value_of_z_plus_one_over_y
  (x y z : ℝ)
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z)
  (h4 : x * y * z = 1)
  (h5 : x + 1 / z = 3)
  (h6 : y + 1 / x = 31) :
  z + 1 / y = 9 / 23 :=
by
  sorry

end NUMINAMATH_GPT_value_of_z_plus_one_over_y_l158_15878


namespace NUMINAMATH_GPT_quadratic_roots_correct_l158_15881

theorem quadratic_roots_correct (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_correct_l158_15881


namespace NUMINAMATH_GPT_floor_factorial_expression_l158_15820

-- Mathematical definitions (conditions)
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Mathematical proof problem (statement)
theorem floor_factorial_expression :
  Int.floor ((factorial 2007 + factorial 2004 : ℚ) / (factorial 2006 + factorial 2005)) = 2006 :=
sorry

end NUMINAMATH_GPT_floor_factorial_expression_l158_15820


namespace NUMINAMATH_GPT_river_flow_rate_l158_15870

theorem river_flow_rate
  (depth width volume_per_minute : ℝ)
  (h1 : depth = 2)
  (h2 : width = 45)
  (h3 : volume_per_minute = 6000) :
  (volume_per_minute / (depth * width)) * (1 / 1000) * 60 = 4.0002 :=
by
  -- Sorry is used to skip the proof.
  sorry

end NUMINAMATH_GPT_river_flow_rate_l158_15870


namespace NUMINAMATH_GPT_find_circle_radius_l158_15865

/-- Eight congruent copies of the parabola y = x^2 are arranged in the plane so that each vertex 
is tangent to a circle, and each parabola is tangent to its two neighbors at an angle of 45°.
Find the radius of the circle. -/

theorem find_circle_radius
  (r : ℝ)
  (h_tangent_to_circle : ∀ (x : ℝ), (x^2 + r) = x → x^2 - x + r = 0)
  (h_single_tangent_point : ∀ (x : ℝ), (x^2 - x + r = 0) → ((1 : ℝ)^2 - 4 * 1 * r = 0)) :
  r = 1/4 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_find_circle_radius_l158_15865


namespace NUMINAMATH_GPT_smallest_b_l158_15811

noncomputable def geometric_sequence : Prop :=
∃ (a b c r : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ b = a * r ∧ c = a * r^2 ∧ a * b * c = 216

theorem smallest_b (a b c r: ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_geom: b = a * r ∧ c = a * r^2 ∧ a * b * c = 216) : b = 6 :=
sorry

end NUMINAMATH_GPT_smallest_b_l158_15811


namespace NUMINAMATH_GPT_sum_of_x_and_y_l158_15808

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 240) : x + y = 680 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l158_15808


namespace NUMINAMATH_GPT_car_maintenance_expense_l158_15825

-- Define constants and conditions
def miles_per_year : ℕ := 12000
def oil_change_interval : ℕ := 3000
def oil_change_price (quarter : ℕ) : ℕ := 
  if quarter = 1 then 55 
  else if quarter = 2 then 45 
  else if quarter = 3 then 50 
  else 40
def free_oil_changes_per_year : ℕ := 1

def tire_rotation_interval : ℕ := 6000
def tire_rotation_cost : ℕ := 40
def tire_rotation_discount : ℕ := 10 -- In percent

def brake_pad_interval : ℕ := 24000
def brake_pad_cost : ℕ := 200
def brake_pad_discount : ℕ := 20 -- In percent
def brake_pad_membership_cost : ℕ := 60
def membership_duration : ℕ := 2 -- In years

def total_annual_expense : ℕ :=
  let oil_changes := (miles_per_year / oil_change_interval) - free_oil_changes_per_year
  let oil_cost := (oil_change_price 2 + oil_change_price 3 + oil_change_price 4) -- Free oil change in Q1
  let tire_rotations := miles_per_year / tire_rotation_interval
  let tire_cost := (tire_rotation_cost * (100 - tire_rotation_discount) / 100) * tire_rotations
  let brake_pad_cost_per_year := (brake_pad_cost * (100 - brake_pad_discount) / 100) / membership_duration
  let membership_cost_per_year := brake_pad_membership_cost / membership_duration
  oil_cost + tire_cost + (brake_pad_cost_per_year + membership_cost_per_year)

-- Assert the proof problem
theorem car_maintenance_expense : total_annual_expense = 317 := by
  sorry

end NUMINAMATH_GPT_car_maintenance_expense_l158_15825


namespace NUMINAMATH_GPT_find_eccentricity_find_equation_l158_15854

open Real

-- Conditions for the first question
def is_ellipse (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def are_focus (a b : ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = ( - sqrt (a^2 - b^2), 0) ∧ F2 = (sqrt (a^2 - b^2), 0)

def arithmetic_sequence (a b : ℝ) (A B : ℝ × ℝ) (F1 : ℝ × ℝ) : Prop :=
  let dist_AF1 := abs (A.1 - F1.1)
  let dist_BF1 := abs (B.1 - F1.1)
  let dist_AB := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (dist_AF1 + dist_AB + dist_BF1 = 4 * a) ∧
  (dist_AF1 + dist_BF1 = 2 * dist_AB)

-- Proof statement for the eccentricity
theorem find_eccentricity (a b : ℝ) (F1 F2 A B : ℝ × ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : is_ellipse a b)
  (h4 : are_focus a b F1 F2)
  (h5 : arithmetic_sequence a b A B F1) :
  ∃ e : ℝ, e = sqrt 2 / 2 :=
sorry

-- Conditions for the second question
def geometric_property (a b : ℝ) (A B P : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, P = (0, -1) → 
             (x^2 / a^2) + (y^2 / b^2) = 1 → 
             abs ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 
             abs ((P.1 - B.1)^2 + (P.2 - B.2)^2)

-- Proof statement for the equation of the ellipse
theorem find_equation (a b : ℝ) (A B P : ℝ × ℝ)
  (h1 : a = 3 * sqrt 2) (h2 : b = 3) (h3 : P = (0, -1))
  (h4 : is_ellipse a b) (h5 : geometric_property a b A B P) :
  ∃ E : Prop, E = ((x : ℝ) * 2 / 18 + (y : ℝ) * 2 / 9 = 1) :=
sorry

end NUMINAMATH_GPT_find_eccentricity_find_equation_l158_15854


namespace NUMINAMATH_GPT_part_a_avg_area_difference_part_b_prob_same_area_part_c_expected_value_difference_l158_15853

-- Part (a)
theorem part_a_avg_area_difference : 
  let zahid_avg := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6
  let yana_avg := (21 / 6)^2
  zahid_avg - yana_avg = 35 / 12 := sorry

-- Part (b)
theorem part_b_prob_same_area :
  let prob_zahid_min n := (13 - 2 * n) / 36
  let prob_same_area := (1 / 36) * ((11 / 36) + (9 / 36) + (7 / 36) + (5 / 36) + (3 / 36) + (1 / 36))
  prob_same_area = 1 / 24 := sorry

-- Part (c)
theorem part_c_expected_value_difference :
  let yana_avg := 49 / 4
  let zahid_avg := (11 / 36 * 1^2 + 9 / 36 * 2^2 + 7 / 36 * 3^2 + 5 / 36 * 4^2 + 3 / 36 * 5^2 + 1 / 36 * 6^2)
  (yana_avg - zahid_avg) = 35 / 9 := sorry

end NUMINAMATH_GPT_part_a_avg_area_difference_part_b_prob_same_area_part_c_expected_value_difference_l158_15853
