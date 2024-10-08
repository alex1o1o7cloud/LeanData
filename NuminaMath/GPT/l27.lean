import Mathlib

namespace basketball_team_selection_l27_27355

theorem basketball_team_selection :
  (Nat.choose 4 2) * (Nat.choose 14 6) = 18018 := 
by
  -- number of ways to choose 2 out of 4 quadruplets
  -- number of ways to choose 6 out of the remaining 14 players
  -- the product of these combinations equals the required number of ways
  sorry

end basketball_team_selection_l27_27355


namespace geom_seq_prop_l27_27517

-- Definitions from the conditions
def geom_seq (a : ℕ → ℝ) := ∀ (n : ℕ), (a (n + 1)) / (a n) = (a 1) / (a 0) ∧ a n > 0

def condition (a : ℕ → ℝ) :=
  (1 / (a 2 * a 4)) + (2 / (a 4 ^ 2)) + (1 / (a 4 * a 6)) = 81

-- The statement to prove
theorem geom_seq_prop (a : ℕ → ℝ) (hgeom : geom_seq a) (hcond : condition a) :
  (1 / (a 3) + 1 / (a 5)) = 9 :=
sorry

end geom_seq_prop_l27_27517


namespace intersection_M_N_l27_27508

def M : Set ℝ := {x | x < 1/2}
def N : Set ℝ := {y | y ≥ -4}

theorem intersection_M_N :
  (M ∩ N = {x | -4 ≤ x ∧ x < 1/2}) :=
sorry

end intersection_M_N_l27_27508


namespace min_a_for_increasing_interval_l27_27959

def f (x a : ℝ) : ℝ := x^2 + (a - 2) * x - 1

theorem min_a_for_increasing_interval (a : ℝ) : (∀ x : ℝ, x ≥ 2 → f x a ≤ f (x + 1) a) ↔ a ≥ -2 :=
sorry

end min_a_for_increasing_interval_l27_27959


namespace solve_floor_equation_l27_27478

theorem solve_floor_equation (x : ℚ) 
  (h : ⌊(5 + 6 * x) / 8⌋ = (15 * x - 7) / 5) : 
  x = 7 / 15 ∨ x = 4 / 5 := 
sorry

end solve_floor_equation_l27_27478


namespace add_fractions_l27_27812

theorem add_fractions :
  (8:ℚ) / 19 + 5 / 57 = 29 / 57 :=
sorry

end add_fractions_l27_27812


namespace cost_to_fill_pool_l27_27512

/-- Definition of the pool dimensions and constants --/
def pool_length := 20
def pool_width := 6
def pool_depth := 10
def cubic_feet_to_liters := 25
def liter_cost := 3

/-- Calculating the cost to fill the pool --/
def pool_volume := pool_length * pool_width * pool_depth
def total_liters := pool_volume * cubic_feet_to_liters
def total_cost := total_liters * liter_cost

/-- Theorem stating that the total cost to fill the pool is $90,000 --/
theorem cost_to_fill_pool : total_cost = 90000 := by
  sorry

end cost_to_fill_pool_l27_27512


namespace integral_curve_has_inflection_points_l27_27289

theorem integral_curve_has_inflection_points (x y : ℝ) (f : ℝ → ℝ → ℝ) :
  f x y = y - x^2 + 2*x - 2 →
  (∃ y' y'' : ℝ, y' = f x y ∧ y'' = y - x^2 ∧ y'' = 0) ↔ y = x^2 :=
by
  sorry

end integral_curve_has_inflection_points_l27_27289


namespace polynomial_roots_l27_27298

-- The statement that we need to prove
theorem polynomial_roots (a b : ℚ) (h : (2 + Real.sqrt 3) ^ 3 + 4 * (2 + Real.sqrt 3) ^ 2 + a * (2 + Real.sqrt 3) + b = 0) :
  ((Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C (4 : ℚ) * Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b) = Polynomial.X ^ 3 + Polynomial.C (4 : ℚ) * Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b) →
  (2 - Real.sqrt 3) ^ 3 + 4 * (2 - Real.sqrt 3) ^ 2 + a * (2 - Real.sqrt 3) + b = 0 ∧ -8 ^ 3 + 4 * (-8) ^ 2 + a * (-8) + b = 0 := sorry

end polynomial_roots_l27_27298


namespace min_value_xy_l27_27483

theorem min_value_xy (x y : ℝ) (h1 : x + y = -1) (h2 : x < 0) (h3 : y < 0) :
  ∃ (xy_min : ℝ), (∀ (xy : ℝ), xy = x * y → xy + 1 / xy ≥ xy_min) ∧ xy_min = 17 / 4 :=
by
  sorry

end min_value_xy_l27_27483


namespace one_point_one_seven_three_billion_in_scientific_notation_l27_27227

theorem one_point_one_seven_three_billion_in_scientific_notation :
  (1.173 * 10^9 = 1.173 * 1000000000) :=
by
  sorry

end one_point_one_seven_three_billion_in_scientific_notation_l27_27227


namespace sum_after_50_rounds_l27_27317

def initial_states : List ℤ := [1, 0, -1]

def operation (n : ℤ) : ℤ :=
  match n with
  | 1   => n * n * n
  | 0   => n * n
  | -1  => -n
  | _ => n  -- although not necessary for current problem, this covers other possible states

def process_calculator (state : ℤ) (times: ℕ) : ℤ :=
  if state = 1 then state
  else if state = 0 then state
  else if state = -1 then state * (-1) ^ times
  else state

theorem sum_after_50_rounds :
  let final_states := initial_states.map (fun s => process_calculator s 50)
  final_states.sum = 2 := by
  simp only [initial_states, process_calculator]
  simp
  sorry

end sum_after_50_rounds_l27_27317


namespace max_books_l27_27961

theorem max_books (cost_per_book : ℝ) (total_money : ℝ) (h_cost : cost_per_book = 8.75) (h_money : total_money = 250.0) :
  ∃ n : ℕ, n = 28 ∧ cost_per_book * n ≤ total_money ∧ ∀ m : ℕ, cost_per_book * m ≤ total_money → m ≤ 28 :=
by
  sorry

end max_books_l27_27961


namespace river_flow_volume_l27_27899

/-- Given a river depth of 7 meters, width of 75 meters, 
and flow rate of 4 kilometers per hour,
the volume of water running into the sea per minute 
is 35,001.75 cubic meters. -/
theorem river_flow_volume
  (depth : ℝ) (width : ℝ) (rate_kmph : ℝ)
  (depth_val : depth = 7)
  (width_val : width = 75)
  (rate_val : rate_kmph = 4) :
  ( width * depth * (rate_kmph * 1000 / 60) ) = 35001.75 :=
by
  rw [depth_val, width_val, rate_val]
  sorry

end river_flow_volume_l27_27899


namespace percent_of_z_equals_120_percent_of_y_l27_27524

variable {x y z : ℝ}
variable {p : ℝ}

theorem percent_of_z_equals_120_percent_of_y
  (h1 : (p / 100) * z = 1.2 * y)
  (h2 : y = 0.75 * x)
  (h3 : z = 2 * x) :
  p = 45 :=
by sorry

end percent_of_z_equals_120_percent_of_y_l27_27524


namespace rectangle_dimensions_l27_27247

-- Define the dimensions and properties of the rectangle
variables {a b : ℕ}

-- Theorem statement
theorem rectangle_dimensions 
  (h1 : b = a + 3)
  (h2 : 2 * a + 2 * b + a = a * b) : 
  (a = 3 ∧ b = 6) :=
by
  sorry

end rectangle_dimensions_l27_27247


namespace find_a3_l27_27579

-- Define the polynomial equality
def polynomial_equality (x : ℝ) (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) :=
  (1 + x) * (2 - x)^6 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7

-- State the main theorem
theorem find_a3 (a0 a1 a2 a4 a5 a6 a7 : ℝ) :
  (∃ (x : ℝ), polynomial_equality x a0 a1 a2 (-25) a4 a5 a6 a7) :=
sorry

end find_a3_l27_27579


namespace chords_intersect_probability_l27_27266

noncomputable def probability_chords_intersect (n m : ℕ) : ℚ :=
  if (n > 6 ∧ m = 2023) then
    1 / 72
  else
    0

theorem chords_intersect_probability :
  probability_chords_intersect 6 2023 = 1 / 72 :=
by
  sorry

end chords_intersect_probability_l27_27266


namespace max_area_of_triangle_l27_27650

-- Defining the side lengths and constraints
def triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main statement of the area maximization problem
theorem max_area_of_triangle (x : ℝ) (h1 : 2 < x) (h2 : x < 6) :
  triangle_sides 6 x (2 * x) →
  ∃ (S : ℝ), S = 12 :=
by
  sorry

end max_area_of_triangle_l27_27650


namespace person_A_misses_at_least_once_in_4_shots_person_B_stops_after_5_shots_due_to_2_consecutive_misses_l27_27594

-- Define the probability of hitting the target for Person A and Person B
def p_hit_A : ℚ := 2 / 3
def p_hit_B : ℚ := 3 / 4

-- Define the complementary probabilities (missing the target)
def p_miss_A := 1 - p_hit_A
def p_miss_B := 1 - p_hit_B

-- Prove the probability that Person A, shooting 4 times, misses the target at least once
theorem person_A_misses_at_least_once_in_4_shots :
  (1 - (p_hit_A ^ 4)) = 65 / 81 :=
by 
  sorry

-- Prove the probability that Person B stops shooting exactly after 5 shots
-- due to missing the target consecutively 2 times
theorem person_B_stops_after_5_shots_due_to_2_consecutive_misses :
  (p_hit_B * p_hit_B * p_miss_B * (p_miss_B * p_miss_B)) = 45 / 1024 :=
by
  sorry

end person_A_misses_at_least_once_in_4_shots_person_B_stops_after_5_shots_due_to_2_consecutive_misses_l27_27594


namespace chocolate_chips_per_member_l27_27263

/-
Define the problem conditions:
-/
def family_members := 4
def batches_choc_chip := 3
def cookies_per_batch_choc_chip := 12
def chips_per_cookie_choc_chip := 2
def batches_double_choc_chip := 2
def cookies_per_batch_double_choc_chip := 10
def chips_per_cookie_double_choc_chip := 4

/-
State the theorem to be proved:
-/
theorem chocolate_chips_per_member : 
  let total_choc_chip_cookies := batches_choc_chip * cookies_per_batch_choc_chip
  let total_choc_chips_choc_chip := total_choc_chip_cookies * chips_per_cookie_choc_chip
  let total_double_choc_chip_cookies := batches_double_choc_chip * cookies_per_batch_double_choc_chip
  let total_choc_chips_double_choc_chip := total_double_choc_chip_cookies * chips_per_cookie_double_choc_chip
  let total_choc_chips := total_choc_chips_choc_chip + total_choc_chips_double_choc_chip
  let chips_per_member := total_choc_chips / family_members
  chips_per_member = 38 :=
by
  sorry

end chocolate_chips_per_member_l27_27263


namespace noah_uses_36_cups_of_water_l27_27124

theorem noah_uses_36_cups_of_water
  (O : ℕ) (hO : O = 4)
  (S : ℕ) (hS : S = 3 * O)
  (W : ℕ) (hW : W = 3 * S) :
  W = 36 := 
  by sorry

end noah_uses_36_cups_of_water_l27_27124


namespace roots_greater_than_two_l27_27759

variable {x m : ℝ}

theorem roots_greater_than_two (h : ∀ x, x^2 - 2 * m * x + 4 = 0 → (∃ a b : ℝ, a > 2 ∧ b < 2 ∧ x = a ∨ x = b)) : 
  m > 2 :=
by
  sorry

end roots_greater_than_two_l27_27759


namespace calculate_total_students_l27_27426

/-- Define the number of students who like basketball, cricket, and soccer. -/
def likes_basketball : ℕ := 7
def likes_cricket : ℕ := 10
def likes_soccer : ℕ := 8
def likes_all_three : ℕ := 2
def likes_basketball_and_cricket : ℕ := 5
def likes_basketball_and_soccer : ℕ := 4
def likes_cricket_and_soccer : ℕ := 3

/-- Calculate the number of students who like at least one sport using the principle of inclusion-exclusion. -/
def students_who_like_at_least_one_sport (b c s bc bs cs bcs : ℕ) : ℕ :=
  b + c + s - (bc + bs + cs) + bcs

theorem calculate_total_students :
  students_who_like_at_least_one_sport likes_basketball likes_cricket likes_soccer 
    (likes_basketball_and_cricket - likes_all_three) 
    (likes_basketball_and_soccer - likes_all_three) 
    (likes_cricket_and_soccer - likes_all_three) 
    likes_all_three = 21 := 
by
  sorry

end calculate_total_students_l27_27426


namespace valid_license_plates_count_l27_27994

/--
The problem is to prove that the total number of valid license plates under the given format is equal to 45,697,600.
The given conditions are:
1. A valid license plate in Xanadu consists of three letters followed by two digits, and then one more letter at the end.
2. There are 26 choices of letters for each letter spot.
3. There are 10 choices of digits for each digit spot.

We need to conclude that the number of possible license plates is:
26^4 * 10^2 = 45,697,600.
-/

def num_valid_license_plates : Nat :=
  let letter_choices := 26
  let digit_choices := 10
  let total_choices := letter_choices ^ 3 * digit_choices ^ 2 * letter_choices
  total_choices

theorem valid_license_plates_count : num_valid_license_plates = 45697600 := by
  sorry

end valid_license_plates_count_l27_27994


namespace find_ratio_of_hyperbola_asymptotes_l27_27819

theorem find_ratio_of_hyperbola_asymptotes (a b : ℝ) (h : a > b) (hyp : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → |(2 * b / a)| = 1) : 
  a / b = 2 := 
by 
  sorry

end find_ratio_of_hyperbola_asymptotes_l27_27819


namespace operation_1_even_if_input_even_operation_2_even_if_input_even_operation_3_even_if_input_even_operation_4_not_always_even_if_input_even_operation_5_not_always_even_if_input_even_l27_27072

-- Define what an even integer is
def is_even (a : ℤ) : Prop := ∃ k : ℤ, a = 2 * k

-- Define the operations
def add_four (a : ℤ) := a + 4
def subtract_six (a : ℤ) := a - 6
def multiply_by_eight (a : ℤ) := a * 8
def divide_by_two_add_two (a : ℤ) := a / 2 + 2
def average_with_ten (a : ℤ) := (a + 10) / 2

-- The proof statements
theorem operation_1_even_if_input_even (a : ℤ) (h : is_even a) : is_even (add_four a) := sorry
theorem operation_2_even_if_input_even (a : ℤ) (h : is_even a) : is_even (subtract_six a) := sorry
theorem operation_3_even_if_input_even (a : ℤ) (h : is_even a) : is_even (multiply_by_eight a) := sorry
theorem operation_4_not_always_even_if_input_even (a : ℤ) (h : is_even a) : ¬ is_even (divide_by_two_add_two a) := sorry
theorem operation_5_not_always_even_if_input_even (a : ℤ) (h : is_even a) : ¬ is_even (average_with_ten a) := sorry

end operation_1_even_if_input_even_operation_2_even_if_input_even_operation_3_even_if_input_even_operation_4_not_always_even_if_input_even_operation_5_not_always_even_if_input_even_l27_27072


namespace value_of_x_l27_27306

theorem value_of_x (z : ℤ) (h1 : z = 100) (y : ℤ) (h2 : y = z / 10) (x : ℤ) (h3 : x = y / 3) : 
  x = 10 / 3 := 
by
  -- The proof is skipped
  sorry

end value_of_x_l27_27306


namespace cathy_wallet_left_money_l27_27264

noncomputable def amount_left_in_wallet (initial : ℝ) (dad_amount : ℝ) (book_cost : ℝ) (saving_percentage : ℝ) : ℝ :=
  let mom_amount := 2 * dad_amount
  let total_initial := initial + dad_amount + mom_amount
  let after_purchase := total_initial - book_cost
  let saved_amount := saving_percentage * after_purchase
  after_purchase - saved_amount

theorem cathy_wallet_left_money :
  amount_left_in_wallet 12 25 15 0.20 = 57.60 :=
by 
  sorry

end cathy_wallet_left_money_l27_27264


namespace find_f_l27_27548

-- Define the function f and its conditions
def f (x : ℝ) : ℝ := sorry

axiom f_0 : f 0 = 0
axiom f_xy (x y : ℝ) : f (x * y) = f ((x^2 + y^2) / 2) + 3 * (x - y)^2

-- Theorem to be proved
theorem find_f (x : ℝ) : f x = -6 * x + 3 :=
by sorry -- proof goes here

end find_f_l27_27548


namespace find_distance_d_l27_27066

theorem find_distance_d (d : ℝ) (XR : ℝ) (YP : ℝ) (XZ : ℝ) (YZ : ℝ) (XY : ℝ) (h1 : XR = 3) (h2 : YP = 12) (h3 : XZ = 3 + d) (h4 : YZ = 12 + d) (h5 : XY = 15) (h6 : (XZ)^2 + (XY)^2 = (YZ)^2) : d = 5 :=
sorry

end find_distance_d_l27_27066


namespace tan_A_eq_11_l27_27530

variable (A B C : ℝ)

theorem tan_A_eq_11
  (h1 : Real.sin A = 10 * Real.sin B * Real.sin C)
  (h2 : Real.cos A = 10 * Real.cos B * Real.cos C) :
  Real.tan A = 11 := 
sorry

end tan_A_eq_11_l27_27530


namespace price_of_books_sold_at_lower_price_l27_27873

-- Define the conditions
variable (n m p q t : ℕ) (earnings price_high price_low : ℝ)

-- The given conditions
def total_books : ℕ := 10
def books_high_price : ℕ := 2 * total_books / 5 -- 2/5 of total books
def books_low_price : ℕ := total_books - books_high_price
def high_price : ℝ := 2.50
def total_earnings : ℝ := 22

-- The proposition to prove
theorem price_of_books_sold_at_lower_price
  (h_books_high_price : books_high_price = 4)
  (h_books_low_price : books_low_price = 6)
  (h_total_earnings : total_earnings = 22)
  (h_high_price : high_price = 2.50) :
  (price_low = 2) := 
-- Proof goes here 
sorry

end price_of_books_sold_at_lower_price_l27_27873


namespace minimum_cost_to_store_food_l27_27941

-- Define the problem setting
def total_volume : ℕ := 15
def capacity_A : ℕ := 2
def capacity_B : ℕ := 3
def price_A : ℕ := 13
def price_B : ℕ := 15
def cashback_threshold : ℕ := 3
def cashback : ℕ := 10

-- The mathematical theorem statement for the proof problem
theorem minimum_cost_to_store_food : 
  ∃ (x y : ℕ), 
    capacity_A * x + capacity_B * y = total_volume ∧ 
    (y = 5 ∧ price_B * y = 75) ∨ 
    (x = 3 ∧ y = 3 ∧ price_A * x + price_B * y - cashback = 74) :=
sorry

end minimum_cost_to_store_food_l27_27941


namespace intersection_l27_27057

namespace Proof

def A := {x : ℝ | 0 ≤ x ∧ x ≤ 6}
def B := {x : ℝ | 3 * x^2 + x - 8 ≤ 0}

theorem intersection (x : ℝ) : x ∈ A ∩ B ↔ 0 ≤ x ∧ x ≤ (4:ℝ)/3 := 
by 
  sorry  -- proof placeholder

end Proof

end intersection_l27_27057


namespace find_L_l27_27100

theorem find_L (RI G SP T M N : ℝ) (h1 : RI + G + SP = 50) (h2 : RI + T + M = 63) (h3 : G + T + SP = 25) 
(h4 : SP + M = 13) (h5 : M + RI = 48) (h6 : N = 1) :
  ∃ L : ℝ, L * M * T + SP * RI * N * G = 2023 ∧ L = 341 / 40 := 
by
  sorry

end find_L_l27_27100


namespace problem_solution_l27_27110

theorem problem_solution :
  (315^2 - 291^2) / 24 = 606 :=
by
  sorry

end problem_solution_l27_27110


namespace businesses_brandon_can_apply_to_l27_27434

-- Definitions of the given conditions in the problem
variables (x y : ℕ)

-- Define the total, fired, and quit businesses
def total_businesses : ℕ := 72
def fired_businesses : ℕ := 36
def quit_businesses : ℕ := 24

-- Define the unique businesses Brandon can still apply to, considering common businesses and reapplications
def businesses_can_apply_to : ℕ := (12 + x) + y

-- The theorem to prove
theorem businesses_brandon_can_apply_to (x y : ℕ) : businesses_can_apply_to x y = 12 + x + y := by
  unfold businesses_can_apply_to
  sorry

end businesses_brandon_can_apply_to_l27_27434


namespace Frank_read_books_l27_27608

noncomputable def books_read (total_days : ℕ) (days_per_book : ℕ) : ℕ :=
total_days / days_per_book

theorem Frank_read_books : books_read 492 12 = 41 := by
  sorry

end Frank_read_books_l27_27608


namespace correct_exponential_calculation_l27_27048

theorem correct_exponential_calculation (a : ℝ) (ha : a ≠ 0) : 
  (a^4)^4 = a^16 :=
by sorry

end correct_exponential_calculation_l27_27048


namespace main_problem_l27_27501

def arithmetic_sequence (a : ℕ → ℕ) : Prop := ∃ a₁ d, ∀ n, a (n + 1) = a₁ + n * d

def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop := ∀ n, S n = (n * (a 1 + a n)) / 2

def another_sequence (b : ℕ → ℕ) (a : ℕ → ℕ) : Prop := ∀ n, b n = 1 / (a n * a (n + 1))

theorem main_problem (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) 
  (h1 : a_3 = 5) 
  (h2 : S_3 = 9) 
  (h3 : arithmetic_sequence a)
  (h4 : sequence_sum a S)
  (h5 : another_sequence b a) : 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n = n / (2 * n + 1)) := sorry

end main_problem_l27_27501


namespace valerie_laptop_purchase_l27_27412

/-- Valerie wants to buy a new laptop priced at $800. She receives $100 dollars from her parents,
$60 dollars from her uncle, and $40 dollars from her siblings for her graduation.
She also makes $20 dollars each week from tutoring. How many weeks must she save 
her tutoring income, along with her graduation money, to buy the laptop? -/
theorem valerie_laptop_purchase :
  let price_of_laptop : ℕ := 800
  let graduation_money : ℕ := 100 + 60 + 40
  let weekly_tutoring_income : ℕ := 20
  let remaining_amount_needed : ℕ := price_of_laptop - graduation_money
  let weeks_needed := remaining_amount_needed / weekly_tutoring_income
  weeks_needed = 30 :=
by
  sorry

end valerie_laptop_purchase_l27_27412


namespace q_is_false_l27_27696

theorem q_is_false (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end q_is_false_l27_27696


namespace technician_round_trip_completion_l27_27565

theorem technician_round_trip_completion (D : ℝ) (h0 : D > 0) :
  let round_trip := 2 * D
  let to_center := D
  let from_center := 0.30 * D
  let traveled := to_center + from_center
  traveled / round_trip * 100 = 65 := 
by
  sorry

end technician_round_trip_completion_l27_27565


namespace scores_greater_than_18_l27_27224

noncomputable def olympiad_scores (scores : Fin 20 → ℕ) :=
∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k

theorem scores_greater_than_18 (scores : Fin 20 → ℕ) (h1 : ∀ i j, i < j → scores i < scores j)
  (h2 : olympiad_scores scores) : ∀ i, 18 < scores i :=
by
  intro i
  sorry

end scores_greater_than_18_l27_27224


namespace no_nat_x_y_square_l27_27068

theorem no_nat_x_y_square (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ y^2 + x = b^2) := 
by 
  sorry

end no_nat_x_y_square_l27_27068


namespace find_Japanese_students_l27_27507

theorem find_Japanese_students (C K J : ℕ) (hK: K = (6 * C) / 11) (hJ: J = C / 8) (hK_value: K = 48) : J = 11 :=
by
  sorry

end find_Japanese_students_l27_27507


namespace lattice_point_count_l27_27408

noncomputable def countLatticePoints (N : ℤ) : ℤ :=
  2 * N * (N + 1) + 1

theorem lattice_point_count (N : ℤ) (hN : 71 * N > 0) :
    ∃ P, P = countLatticePoints N := sorry

end lattice_point_count_l27_27408


namespace distance_between_house_and_school_l27_27682

theorem distance_between_house_and_school (T D : ℕ) 
    (h1 : D = 10 * (T + 2)) 
    (h2 : D = 20 * (T - 1)) : 
    D = 60 := by
  sorry

end distance_between_house_and_school_l27_27682


namespace other_student_in_sample_18_l27_27302

theorem other_student_in_sample_18 (class_size sample_size : ℕ) (all_students : Finset ℕ) (sample_students : List ℕ)
  (h_class_size : class_size = 60)
  (h_sample_size : sample_size = 4)
  (h_all_students : all_students = Finset.range 60) -- students are numbered from 1 to 60
  (h_sample : sample_students = [3, 33, 48])
  (systematic_sampling : ℕ → ℕ → List ℕ) -- systematic_sampling function that generates the sample based on first element and k
  (k : ℕ) (h_k : k = class_size / sample_size) :
  systematic_sampling 3 k = [3, 18, 33, 48] := 
  sorry

end other_student_in_sample_18_l27_27302


namespace solve_system_l27_27752

theorem solve_system (x y z : ℝ) 
  (h1 : 19 * (x + y) + 17 = 19 * (-x + y) - 21)
  (h2 : 5 * x - 3 * z = 11 * y - 7) : 
  x = -1 ∧ z = -11 * y / 3 + 2 / 3 :=
by sorry

end solve_system_l27_27752


namespace abs_opposite_numbers_l27_27047

theorem abs_opposite_numbers (m n : ℤ) (h : m + n = 0) : |m + n - 1| = 1 := by
  sorry

end abs_opposite_numbers_l27_27047


namespace men_at_yoga_studio_l27_27107

open Real

def yoga_men_count (M : ℕ) (avg_weight_men avg_weight_women avg_weight_total : ℝ) (num_women num_total : ℕ) : Prop :=
  avg_weight_men = 190 ∧
  avg_weight_women = 120 ∧
  num_women = 6 ∧
  num_total = 14 ∧
  avg_weight_total = 160 →
  M + num_women = num_total ∧
  (M * avg_weight_men + num_women * avg_weight_women) / num_total = avg_weight_total ∧
  M = 8

theorem men_at_yoga_studio : ∃ M : ℕ, yoga_men_count M 190 120 160 6 14 :=
  by 
  use 8
  sorry

end men_at_yoga_studio_l27_27107


namespace jessica_can_mail_letter_l27_27745

-- Define the constants
def paper_weight := 1/5 -- each piece of paper weighs 1/5 ounce
def envelope_weight := 2/5 -- envelope weighs 2/5 ounce
def num_papers := 8

-- Calculate the total weight
def total_weight := num_papers * paper_weight + envelope_weight

-- Define stamping rates
def international_rate := 2 -- $2 per ounce internationally

-- Calculate the required postage
def required_postage := total_weight * international_rate

-- Define the available stamp values
inductive Stamp
| one_dollar : Stamp
| fifty_cents : Stamp

-- Function to calculate the total value of a given stamp combination
def stamp_value : List Stamp → ℝ
| [] => 0
| (Stamp.one_dollar :: rest) => 1 + stamp_value rest
| (Stamp.fifty_cents :: rest) => 0.5 + stamp_value rest

-- State the theorem to be proved
theorem jessica_can_mail_letter :
  ∃ stamps : List Stamp, stamp_value stamps = required_postage := by
sorry

end jessica_can_mail_letter_l27_27745


namespace range_of_k_l27_27064

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x^2 - 2 * x + k^2 - 3 > 0) -> (k > 2 ∨ k < -2) :=
by
  sorry

end range_of_k_l27_27064


namespace range_of_a_l27_27993

variables {x a : ℝ}

def p (x : ℝ) : Prop := (x - 5) / (x - 3) ≥ 2
def q (x a : ℝ) : Prop := x ^ 2 - a * x ≤ x - a

theorem range_of_a (h : ¬(∃ x, p x) → ¬(∃ x, q x a)) :
  1 ≤ a ∧ a < 3 :=
by 
  sorry

end range_of_a_l27_27993


namespace frobenius_two_vars_l27_27535

theorem frobenius_two_vars (a b n : ℤ) (ha : 0 < a) (hb : 0 < b) (hgcd : Int.gcd a b = 1) (hn : n > a * b - a - b) :
  ∃ x y : ℕ, n = a * x + b * y :=
by
  sorry

end frobenius_two_vars_l27_27535


namespace incorrect_inequality_l27_27450

variable (a b : ℝ)

theorem incorrect_inequality (h : a > b) : ¬ (-2 * a > -2 * b) :=
by sorry

end incorrect_inequality_l27_27450


namespace find_largest_m_l27_27871

variables (a b c t : ℝ)
def f (x : ℝ) := a * x^2 + b * x + c

theorem find_largest_m (a_ne_zero : a ≠ 0)
  (cond1 : ∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x) ∧ f a b c x ≥ x)
  (cond2 : ∀ x : ℝ, 0 < x ∧ x < 2 → f a b c x ≤ ((x + 1) / 2)^2)
  (cond3 : ∃ x : ℝ, ∀ y : ℝ, f a b c y ≥ f a b c x ∧ f a b c x = 0) :
  ∃ m : ℝ, 1 < m ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f a b c (x + t) ≤ x) ∧ m = 9 := sorry

end find_largest_m_l27_27871


namespace global_school_math_students_l27_27056

theorem global_school_math_students (n : ℕ) (h1 : n < 600) (h2 : n % 28 = 27) (h3 : n % 26 = 20) : n = 615 :=
by
  -- skip the proof
  sorry

end global_school_math_students_l27_27056


namespace price_of_each_sundae_l27_27255

theorem price_of_each_sundae (A B : ℝ) (x y z : ℝ) (hx : 200 * x = 80) (hy : A = y) (hz : y = 0.40)
  (hxy : A - 80 = z) (hyz : 200 * z = B) : y = 0.60 :=
by
  sorry

end price_of_each_sundae_l27_27255


namespace mary_jenny_red_marble_ratio_l27_27095

def mary_red_marble := 30  -- Given that Mary collects the same as Jenny.
def jenny_red_marble := 30 -- Given
def jenny_blue_marble := 25 -- Given
def anie_red_marble := mary_red_marble + 20 -- Anie's red marbles count
def anie_blue_marble := 2 * jenny_blue_marble -- Anie's blue marbles count
def mary_blue_marble := anie_blue_marble / 2 -- Mary's blue marbles count

theorem mary_jenny_red_marble_ratio : 
  mary_red_marble / jenny_red_marble = 1 :=
by
  sorry

end mary_jenny_red_marble_ratio_l27_27095


namespace winner_percentage_l27_27405

theorem winner_percentage (total_votes winner_votes : ℕ) (h1 : winner_votes = 744) (h2 : total_votes - winner_votes = 288) :
  (winner_votes : ℤ) * 100 / total_votes = 62 := 
by
  sorry

end winner_percentage_l27_27405


namespace video_files_initial_l27_27722

theorem video_files_initial (V : ℕ) (h1 : 4 + V - 23 = 2) : V = 21 :=
by 
  sorry

end video_files_initial_l27_27722


namespace sum_of_x_y_is_13_l27_27128

theorem sum_of_x_y_is_13 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h : x^4 + y^4 = 4721) : x + y = 13 :=
sorry

end sum_of_x_y_is_13_l27_27128


namespace a_10_eq_18_l27_27858

variable {a : ℕ → ℕ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

axiom a2 : a 2 = 2
axiom a3 : a 3 = 4
axiom arithmetic_seq : is_arithmetic_sequence a

-- problem: prove a_{10} = 18
theorem a_10_eq_18 : a 10 = 18 :=
sorry

end a_10_eq_18_l27_27858


namespace moles_NaOH_combined_with_HCl_l27_27239

-- Definitions for given conditions
def NaOH : Type := Unit
def HCl : Type := Unit
def NaCl : Type := Unit
def H2O : Type := Unit

def balanced_reaction (nHCl nNaOH nNaCl nH2O : ℕ) : Prop :=
  nHCl = nNaOH ∧ nNaOH = nNaCl ∧ nNaCl = nH2O

def mole_mass_H2O : ℕ := 18

-- Given: certain amount of NaOH combined with 1 mole of HCl
def initial_moles_HCl : ℕ := 1

-- Given: 18 grams of H2O formed
def grams_H2O : ℕ := 18

-- Molar mass of H2O is approximately 18 g/mol, so 18 grams is 1 mole
def moles_H2O : ℕ := grams_H2O / mole_mass_H2O

-- Prove that number of moles of NaOH combined with HCl is 1 mole
theorem moles_NaOH_combined_with_HCl : 
  balanced_reaction initial_moles_HCl 1 1 moles_H2O →
  moles_H2O = 1 →
  1 = 1 :=
by
  intros h1 h2
  sorry

end moles_NaOH_combined_with_HCl_l27_27239


namespace reimbursement_proof_l27_27067

-- Define the rates
def rate_industrial_weekday : ℝ := 0.36
def rate_commercial_weekday : ℝ := 0.42
def rate_weekend : ℝ := 0.45

-- Define the distances for each day
def distance_monday : ℝ := 18
def distance_tuesday : ℝ := 26
def distance_wednesday : ℝ := 20
def distance_thursday : ℝ := 20
def distance_friday : ℝ := 16
def distance_saturday : ℝ := 12

-- Calculate the reimbursement for each day
def reimbursement_monday : ℝ := distance_monday * rate_industrial_weekday
def reimbursement_tuesday : ℝ := distance_tuesday * rate_commercial_weekday
def reimbursement_wednesday : ℝ := distance_wednesday * rate_industrial_weekday
def reimbursement_thursday : ℝ := distance_thursday * rate_commercial_weekday
def reimbursement_friday : ℝ := distance_friday * rate_industrial_weekday
def reimbursement_saturday : ℝ := distance_saturday * rate_weekend

-- Calculate the total reimbursement
def total_reimbursement : ℝ :=
  reimbursement_monday + reimbursement_tuesday + reimbursement_wednesday +
  reimbursement_thursday + reimbursement_friday + reimbursement_saturday

-- State the theorem to be proven
theorem reimbursement_proof : total_reimbursement = 44.16 := by
  sorry

end reimbursement_proof_l27_27067


namespace smallest_n_square_average_l27_27197

theorem smallest_n_square_average (n : ℕ) (h : n > 1)
  (S : ℕ := (n * (n + 1) * (2 * n + 1)) / 6)
  (avg : ℕ := S / n) :
  (∃ k : ℕ, avg = k^2) → n = 337 := by
  sorry

end smallest_n_square_average_l27_27197


namespace tank_water_after_rain_final_l27_27090

theorem tank_water_after_rain_final (initial_water evaporated drained rain_rate rain_time : ℕ)
  (initial_water_eq : initial_water = 6000)
  (evaporated_eq : evaporated = 2000)
  (drained_eq : drained = 3500)
  (rain_rate_eq : rain_rate = 350)
  (rain_time_eq : rain_time = 30) :
  let water_after_evaporation := initial_water - evaporated
  let water_after_drainage := water_after_evaporation - drained 
  let rain_addition := (rain_time / 10) * rain_rate
  let final_water := water_after_drainage + rain_addition
  final_water = 1550 :=
by
  sorry

end tank_water_after_rain_final_l27_27090


namespace evaluate_expression_l27_27598

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end evaluate_expression_l27_27598


namespace milk_mixture_l27_27260

theorem milk_mixture:
  ∀ (x : ℝ), 0.40 * x + 1.6 = 0.20 * (x + 16) → x = 8 := 
by
  intro x
  sorry

end milk_mixture_l27_27260


namespace ratio_Lisa_Claire_l27_27165

-- Definitions
def Claire_photos : ℕ := 6
def Robert_photos : ℕ := Claire_photos + 12
def Lisa_photos : ℕ := Robert_photos

-- Theorem statement
theorem ratio_Lisa_Claire : (Lisa_photos : ℚ) / (Claire_photos : ℚ) = 3 / 1 :=
by
  sorry

end ratio_Lisa_Claire_l27_27165


namespace correct_calculation_l27_27790

theorem correct_calculation (a b : ℝ) : 
  (a + 2 * a = 3 * a) := by
  sorry

end correct_calculation_l27_27790


namespace range_of_a_l27_27953

variable (a x y : ℝ)

theorem range_of_a (h1 : 2 * x + y = 1 + 4 * a) (h2 : x + 2 * y = 2 - a) (h3 : x + y > 0) : a > -1 :=
sorry

end range_of_a_l27_27953


namespace cos_squared_identity_l27_27327

theorem cos_squared_identity (α : ℝ) (h : Real.tan (α + π / 4) = 3 / 4) :
    Real.cos (π / 4 - α) ^ 2 = 9 / 25 := by
  sorry

end cos_squared_identity_l27_27327


namespace math_problem_l27_27591

theorem math_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
sorry

end math_problem_l27_27591


namespace quadratic_has_two_roots_l27_27909

variable {a b c : ℝ}

theorem quadratic_has_two_roots (h1 : b > a + c) (h2 : a > 0) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  -- Using the condition \(b > a + c > 0\),
  -- the proof that the quadratic equation \(a x^2 + b x + c = 0\) has two distinct real roots
  -- would be provided here.
  sorry

end quadratic_has_two_roots_l27_27909


namespace socks_pair_count_l27_27597

theorem socks_pair_count :
  let white := 5
  let brown := 5
  let blue := 3
  let green := 2
  (white * brown) + (white * blue) + (white * green) + (brown * blue) + (brown * green) + (blue * green) = 81 :=
by
  intros
  sorry

end socks_pair_count_l27_27597


namespace measure_angle_T_l27_27973

theorem measure_angle_T (P Q R S T : ℝ) (h₀ : P = R) (h₁ : R = T) (h₂ : Q + S = 180)
  (h_sum : P + Q + R + T + S = 540) : T = 120 :=
by
  sorry

end measure_angle_T_l27_27973


namespace hyperbola_center_l27_27786

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0) : 
  x = 2 ∧ y = 4 :=
sorry

end hyperbola_center_l27_27786


namespace isosceles_triangle_perimeter_l27_27353

-- Define what it means to be a root of the equation x^2 - 5x + 6 = 0
def is_root (x : ℝ) : Prop := x^2 - 5 * x + 6 = 0

-- Define the perimeter based on given conditions
theorem isosceles_triangle_perimeter (x : ℝ) (base : ℝ) (h_base : base = 4) (h_root : is_root x) :
    2 * x + base = 10 :=
by
  -- Insert proof here
  sorry

end isosceles_triangle_perimeter_l27_27353


namespace area_of_PQRS_l27_27982

noncomputable def length_square_EFGH := 6
noncomputable def height_equilateral_triangle := 3 * Real.sqrt 3
noncomputable def diagonal_PQRS := length_square_EFGH + 2 * height_equilateral_triangle
noncomputable def area_PQRS := (1 / 2) * (diagonal_PQRS * diagonal_PQRS)

theorem area_of_PQRS :
  (area_PQRS = 72 + 36 * Real.sqrt 3) :=
sorry

end area_of_PQRS_l27_27982


namespace harry_travel_time_l27_27822

def t_bus1 : ℕ := 15
def t_bus2 : ℕ := 25
def t_bus_journey : ℕ := t_bus1 + t_bus2
def t_walk : ℕ := t_bus_journey / 2
def t_total : ℕ := t_bus_journey + t_walk

theorem harry_travel_time : t_total = 60 := by
  -- Will be proved afterwards
  sorry

end harry_travel_time_l27_27822


namespace correct_statements_l27_27059

variable (a_1 a_2 b_1 b_2 : ℝ)

def ellipse1 := ∀ x y : ℝ, x^2 / a_1^2 + y^2 / b_1^2 = 1
def ellipse2 := ∀ x y : ℝ, x^2 / a_2^2 + y^2 / b_2^2 = 1

axiom a1_pos : a_1 > 0
axiom b1_pos : b_1 > 0
axiom a2_gt_b2_pos : a_2 > b_2 ∧ b_2 > 0
axiom same_foci : a_1^2 - b_1^2 = a_2^2 - b_2^2
axiom a1_gt_a2 : a_1 > a_2

theorem correct_statements : 
  (¬(∃ x y, (x^2 / a_1^2 + y^2 / b_1^2 = 1) ∧ (x^2 / a_2^2 + y^2 / b_2^2 = 1))) ∧ 
  (a_1^2 - a_2^2 = b_1^2 - b_2^2) :=
by 
  sorry

end correct_statements_l27_27059


namespace triangle_area_l27_27032

variable (a b c k : ℝ)
variable (h1 : a = 2 * k)
variable (h2 : b = 3 * k)
variable (h3 : c = k * Real.sqrt 13)

theorem triangle_area (h_right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 * a * b) = 3 * k^2 := 
by 
  sorry

end triangle_area_l27_27032


namespace correct_statements_l27_27927

def f (x : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d
def f_prime (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem correct_statements (b c d : ℝ) :
  (∃ x : ℝ, f x b c d = 4 ∧ f_prime x b c = 0) ∧
  (∃ x : ℝ, f x b c d = 0 ∧ f_prime x b c = 0) :=
by
  sorry

end correct_statements_l27_27927


namespace max_and_min_of_z_in_G_l27_27600

def z (x y : ℝ) : ℝ := x^2 + y^2 - 2*x*y - x - 2*y

def G (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4

theorem max_and_min_of_z_in_G :
  (∃ (x y : ℝ), G x y ∧ z x y = 12) ∧ (∃ (x y : ℝ), G x y ∧ z x y = -1/4) :=
sorry

end max_and_min_of_z_in_G_l27_27600


namespace additional_cards_l27_27130

theorem additional_cards (total_cards : ℕ) (num_decks : ℕ) (cards_per_deck : ℕ) 
  (h1 : total_cards = 319) (h2 : num_decks = 6) (h3 : cards_per_deck = 52) : 
  319 - 6 * 52 = 7 := 
by
  sorry

end additional_cards_l27_27130


namespace simplify_radical_expr_l27_27644

-- Define the variables and expressions
variables {x : ℝ} (hx : 0 ≤ x) 

-- State the problem
theorem simplify_radical_expr (hx : 0 ≤ x) :
  (Real.sqrt (100 * x)) * (Real.sqrt (3 * x)) * (Real.sqrt (18 * x)) = 30 * x * Real.sqrt (6 * x) :=
sorry

end simplify_radical_expr_l27_27644


namespace circle_passes_through_points_l27_27694

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l27_27694


namespace contradiction_example_l27_27338

theorem contradiction_example (x y : ℝ) (h1 : x + y > 2) (h2 : x ≤ 1) (h3 : y ≤ 1) : False :=
by
  sorry

end contradiction_example_l27_27338


namespace birds_problem_l27_27087

-- Define the initial number of birds and the total number of birds as given conditions.
def initial_birds : ℕ := 2
def total_birds : ℕ := 6

-- Define the number of new birds that came to join.
def new_birds : ℕ := total_birds - initial_birds

-- State the theorem to be proved, asserting that the number of new birds is 4.
theorem birds_problem : new_birds = 4 := 
by
  -- required proof goes here
  sorry

end birds_problem_l27_27087


namespace computation_l27_27897

theorem computation : 45 * 52 + 28 * 45 = 3600 := by
  sorry

end computation_l27_27897


namespace point_on_x_axis_coordinates_l27_27299

theorem point_on_x_axis_coordinates (a : ℝ) (hx : a - 3 = 0) : (a + 2, a - 3) = (5, 0) :=
by
  sorry

end point_on_x_axis_coordinates_l27_27299


namespace polynomial_value_at_3_l27_27119

-- Definitions based on given conditions
def f (x : ℕ) : ℕ :=
  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def x := 3

-- Proof statement
theorem polynomial_value_at_3 : f x = 1641 := by
  sorry

end polynomial_value_at_3_l27_27119


namespace anthony_total_pencils_l27_27766

theorem anthony_total_pencils (initial_pencils : ℕ) (pencils_given_by_kathryn : ℕ) (total_pencils : ℕ) :
  initial_pencils = 9 →
  pencils_given_by_kathryn = 56 →
  total_pencils = initial_pencils + pencils_given_by_kathryn →
  total_pencils = 65 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end anthony_total_pencils_l27_27766


namespace largest_difference_l27_27974

def A : ℕ := 3 * 2005^2006
def B : ℕ := 2005^2006
def C : ℕ := 2004 * 2005^2005
def D : ℕ := 3 * 2005^2005
def E : ℕ := 2005^2005
def F : ℕ := 2005^2004

theorem largest_difference : (A - B > B - C) ∧ (A - B > C - D) ∧ (A - B > D - E) ∧ (A - B > E - F) :=
by
  sorry  -- Proof is omitted as per instructions.

end largest_difference_l27_27974


namespace pizzas_needed_l27_27891

theorem pizzas_needed (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) (h_people : people = 18) (h_slices_per_person : slices_per_person = 3) (h_slices_per_pizza : slices_per_pizza = 9) :
  people * slices_per_person / slices_per_pizza = 6 :=
by
  sorry

end pizzas_needed_l27_27891


namespace find_positive_integer_solutions_l27_27193

def is_solution (x y : ℕ) : Prop :=
  4 * x^3 + 4 * x^2 * y - 15 * x * y^2 - 18 * y^3 - 12 * x^2 + 6 * x * y + 36 * y^2 + 5 * x - 10 * y = 0

theorem find_positive_integer_solutions :
  ∀ x y : ℕ, 0 < x ∧ 0 < y → (is_solution x y ↔ (x = 1 ∧ y = 1) ∨ (∃ y', y = y' ∧ x = 2 * y' ∧ 0 < y')) :=
by
  intros x y hxy
  sorry

end find_positive_integer_solutions_l27_27193


namespace hulk_jump_kilometer_l27_27675

theorem hulk_jump_kilometer (n : ℕ) (h : ∀ n : ℕ, n ≥ 1 → (2^(n-1) : ℕ) ≤ 1000 → n-1 < 10) : n = 11 :=
by
  sorry

end hulk_jump_kilometer_l27_27675


namespace imo1989_q3_l27_27304

theorem imo1989_q3 (a b : ℤ) (h1 : ¬ (∃ x : ℕ, a = x ^ 2))
                   (h2 : ¬ (∃ y : ℕ, b = y ^ 2))
                   (h3 : ∃ (x y z w : ℤ), x ^ 2 - a * y ^ 2 - b * z ^ 2 + a * b * w ^ 2 = 0 
                                           ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) :
                   ∃ (x y z : ℤ), x ^ 2 - a * y ^ 2 - b * z ^ 2 = 0 ∧ (x, y, z) ≠ (0, 0, 0) := 
sorry

end imo1989_q3_l27_27304


namespace earnings_correct_l27_27491

def phonePrice : Nat := 11
def laptopPrice : Nat := 15
def computerPrice : Nat := 18
def tabletPrice : Nat := 12
def smartwatchPrice : Nat := 8

def phoneRepairs : Nat := 9
def laptopRepairs : Nat := 5
def computerRepairs : Nat := 4
def tabletRepairs : Nat := 6
def smartwatchRepairs : Nat := 8

def totalEarnings : Nat := 
  phoneRepairs * phonePrice + 
  laptopRepairs * laptopPrice + 
  computerRepairs * computerPrice + 
  tabletRepairs * tabletPrice + 
  smartwatchRepairs * smartwatchPrice

theorem earnings_correct : totalEarnings = 382 := by
  sorry

end earnings_correct_l27_27491


namespace fraction_division_addition_l27_27738

theorem fraction_division_addition :
  (3 / 7 / 4) + (2 / 7) = 11 / 28 := by
  sorry

end fraction_division_addition_l27_27738


namespace find_a_l27_27921

-- Define the function f
def f (a x : ℝ) := a * x^3 - 2 * x

-- State the theorem, asserting that if f passes through the point (-1, 4) then a = -2.
theorem find_a (a : ℝ) (h : f a (-1) = 4) : a = -2 :=
by {
    sorry
}

end find_a_l27_27921


namespace sum_area_of_R_eq_20_l27_27930

noncomputable def sum_m_n : ℝ := 
  let s := 4 + 2 * Real.sqrt 2
  let total_area := s ^ 2
  let small_square_area := 4
  let given_rectangle_area := 4 * Real.sqrt 2
  let area_R := total_area - (small_square_area + given_rectangle_area)
  let m := 20
  let n := 12 * Real.sqrt 2
  m + n

theorem sum_area_of_R_eq_20 :
  let s := 4 + 2 * Real.sqrt 2
  let total_area := s ^ 2
  let small_square_area := 4
  let given_rectangle_area := 4 * Real.sqrt 2
  let area_R := total_area - (small_square_area + given_rectangle_area)
  area_R = 20 + 12 * Real.sqrt 2 :=
by
  sorry

end sum_area_of_R_eq_20_l27_27930


namespace systematic_sampling_l27_27540

theorem systematic_sampling :
  let N := 60
  let n := 5
  let k := N / n
  let initial_sample := 5
  let samples := [initial_sample, initial_sample + k, initial_sample + 2 * k, initial_sample + 3 * k, initial_sample + 4 * k] 
  samples = [5, 17, 29, 41, 53] := sorry

end systematic_sampling_l27_27540


namespace wage_difference_l27_27089

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.2

theorem wage_difference : manager_wage - chef_wage = 3.40 := 
by
  sorry

end wage_difference_l27_27089


namespace rosalina_received_21_gifts_l27_27804

def Emilio_gifts : Nat := 11
def Jorge_gifts : Nat := 6
def Pedro_gifts : Nat := 4

def total_gifts : Nat :=
  Emilio_gifts + Jorge_gifts + Pedro_gifts

theorem rosalina_received_21_gifts : total_gifts = 21 := by
  sorry

end rosalina_received_21_gifts_l27_27804


namespace length_of_the_bridge_l27_27928

theorem length_of_the_bridge
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (cross_time_s : ℕ)
  (h_train_length : train_length = 120)
  (h_train_speed_kmh : train_speed_kmh = 45)
  (h_cross_time_s : cross_time_s = 30) :
  ∃ bridge_length : ℕ, bridge_length = 255 := 
by 
  sorry

end length_of_the_bridge_l27_27928


namespace complex_real_imag_eq_l27_27649

theorem complex_real_imag_eq (b : ℝ) (h : (2 + b) / 5 = (2 * b - 1) / 5) : b = 3 :=
  sorry

end complex_real_imag_eq_l27_27649


namespace boys_laps_eq_27_l27_27485

noncomputable def miles_per_lap : ℝ := 3 / 4
noncomputable def girls_miles : ℝ := 27
noncomputable def girls_extra_laps : ℝ := 9

theorem boys_laps_eq_27 :
  (∃ boys_laps girls_laps : ℝ, 
    girls_laps = girls_miles / miles_per_lap ∧ 
    boys_laps = girls_laps - girls_extra_laps ∧ 
    boys_laps = 27) :=
by
  sorry

end boys_laps_eq_27_l27_27485


namespace ab_equals_6_l27_27781

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l27_27781


namespace selling_price_eq_l27_27631

noncomputable def cost_price : ℝ := 1300
noncomputable def selling_price_loss : ℝ := 1280
noncomputable def selling_price_profit_25_percent : ℝ := 1625

theorem selling_price_eq (cp sp_loss sp_profit sp: ℝ) 
  (h1 : sp_profit = 1.25 * cp)
  (h2 : sp_loss = cp - 20)
  (h3 : sp = cp + 20) :
  sp = 1320 :=
sorry

end selling_price_eq_l27_27631


namespace complex_modulus_problem_l27_27256

open Complex

def modulus_of_z (z : ℂ) (h : (z - 2 * I) * (1 - I) = -2) : Prop :=
  abs z = Real.sqrt 2

theorem complex_modulus_problem (z : ℂ) (h : (z - 2 * I) * (1 - I) = -2) : 
  modulus_of_z z h :=
sorry

end complex_modulus_problem_l27_27256


namespace largest_shaded_area_l27_27580

noncomputable def figureA_shaded_area : ℝ := 16 - 4 * Real.pi
noncomputable def figureB_shaded_area : ℝ := 16 - 4 * Real.pi
noncomputable def figureC_shaded_area : ℝ := 16 - 4 * Real.sqrt 3

theorem largest_shaded_area : 
  figureC_shaded_area > figureA_shaded_area ∧ figureC_shaded_area > figureB_shaded_area :=
by
  sorry

end largest_shaded_area_l27_27580


namespace cost_of_orange_juice_l27_27465

theorem cost_of_orange_juice (total_money : ℕ) (bread_qty : ℕ) (orange_qty : ℕ) 
  (bread_cost : ℕ) (money_left : ℕ) (total_spent : ℕ) (orange_cost : ℕ) 
  (h1 : total_money = 86) (h2 : bread_qty = 3) (h3 : orange_qty = 3) 
  (h4 : bread_cost = 3) (h5 : money_left = 59) :
  (total_money - money_left - bread_qty * bread_cost) / orange_qty = 6 :=
by
  have h6 : total_spent = total_money - money_left := by sorry
  have h7 : total_spent - bread_qty * bread_cost = orange_qty * orange_cost := by sorry
  have h8 : orange_cost = 6 := by sorry
  exact sorry

end cost_of_orange_juice_l27_27465


namespace first_digit_base8_of_473_l27_27466

theorem first_digit_base8_of_473 : 
  ∃ (d : ℕ), (d < 8) ∧ (473 = d * 64 + r ∧ r < 64) ∧ 473 = 7 * 64 + 25 :=
sorry

end first_digit_base8_of_473_l27_27466


namespace quadrilateral_rectangle_ratio_l27_27640

theorem quadrilateral_rectangle_ratio
  (s x y : ℝ)
  (h_area : (s + 2 * x) ^ 2 = 4 * s ^ 2)
  (h_y : 2 * y = s) :
  y / x = 1 :=
by
  sorry

end quadrilateral_rectangle_ratio_l27_27640


namespace harry_has_19_apples_l27_27560

def apples_problem := 
  let A_M := 68  -- Martha's apples
  let A_T := A_M - 30  -- Tim's apples (68 - 30)
  let A_H := A_T / 2  -- Harry's apples (38 / 2)
  A_H = 19

theorem harry_has_19_apples : apples_problem :=
by
  -- prove A_H = 19 given the conditions
  sorry

end harry_has_19_apples_l27_27560


namespace rearrangement_impossible_l27_27295

-- Define the primary problem conditions and goal
theorem rearrangement_impossible :
  ¬ ∃ (f : Fin 100 → Fin 51), 
    (∀ k : Fin 51, ∃ i j : Fin 100, 
      f i = k ∧ f j = k ∧ (i < j ∧ j.val - i.val = k.val + 1)) :=
sorry

end rearrangement_impossible_l27_27295


namespace largest_sum_of_digits_l27_27761

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9) (h4 : 1 ≤ y ∧ y ≤ 10) (h5 : (1000 * (a * 100 + b * 10 + c)) = 1000) : 
  a + b + c = 8 :=
sorry

end largest_sum_of_digits_l27_27761


namespace quadratic_solution_l27_27545

theorem quadratic_solution 
  (x : ℝ)
  (h : x^2 - 2 * x - 1 = 0) : 
  x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 :=
sorry

end quadratic_solution_l27_27545


namespace total_strength_of_college_l27_27888

-- Declare the variables for number of students playing each sport
variables (C B Both : ℕ)

-- Given conditions in the problem
def cricket_players : ℕ := 500
def basketball_players : ℕ := 600
def both_players : ℕ := 220

-- Theorem stating the total strength of the college
theorem total_strength_of_college (h_C : C = cricket_players) 
                                  (h_B : B = basketball_players) 
                                  (h_Both : Both = both_players) : 
                                  C + B - Both = 880 :=
by
  sorry

end total_strength_of_college_l27_27888


namespace find_S6_l27_27983

variable (a : ℕ → ℝ) (S_n : ℕ → ℝ)

-- The sequence {a_n} is given as a geometric sequence
-- Partial sums are given as S_2 = 1 and S_4 = 3

-- Conditions
axiom geom_sequence : ∀ n : ℕ, a (n + 1) / a n = a 1 / a 0
axiom S2 : S_n 2 = 1
axiom S4 : S_n 4 = 3

-- Theorem statement
theorem find_S6 : S_n 6 = 7 :=
sorry

end find_S6_l27_27983


namespace calculate_expression_l27_27088

theorem calculate_expression :
  ( ( (1/6) - (1/8) + (1/9) ) / ( (1/3) - (1/4) + (1/5) ) ) * 3 = 55 / 34 :=
by
  sorry

end calculate_expression_l27_27088


namespace f_f_f_f_f_3_eq_4_l27_27736

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_f_f_f_f_3_eq_4 : f (f (f (f (f 3)))) = 4 := 
  sorry

end f_f_f_f_f_3_eq_4_l27_27736


namespace range_of_a_l27_27661

theorem range_of_a
  (a : ℝ)
  (h : ∀ x : ℝ, |x + 1| + |x - 3| ≥ a) : a ≤ 4 :=
sorry

end range_of_a_l27_27661


namespace square_perimeter_calculation_l27_27951

noncomputable def perimeter_of_square (radius: ℝ) : ℝ := 
  if radius = 4 then 64 * Real.sqrt 2 else 0

theorem square_perimeter_calculation :
  perimeter_of_square 4 = 64 * Real.sqrt 2 :=
by
  sorry

end square_perimeter_calculation_l27_27951


namespace regular_polygon_sides_l27_27427

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l27_27427


namespace arithmetic_sequence_general_term_arithmetic_sequence_max_sum_l27_27687

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) : 
  (∀ n : ℕ, a n = a 1 + (n - 1) * (-2)) → 
  a 2 = 1 → 
  a 5 = -5 → 
  ∀ n : ℕ, a n = -2 * n + 5 :=
by
  intros h₁ h₂ h₅
  sorry

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (-2)) →
  a 2 = 1 → 
  a 5 = -5 → 
  ∃ n : ℕ, n = 2 ∧ S n = 4 :=
by
  intros hSn h₂ h₅
  sorry

end arithmetic_sequence_general_term_arithmetic_sequence_max_sum_l27_27687


namespace units_digit_of_expression_l27_27976

def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def expression := (20 * 21 * 22 * 23 * 24 * 25) / 1000

theorem units_digit_of_expression : units_digit (expression) = 2 :=
by
  sorry

end units_digit_of_expression_l27_27976


namespace find_n_from_lcms_l27_27153

theorem find_n_from_lcms (n : ℕ) (h_pos : n > 0) (h_lcm1 : Nat.lcm 40 n = 200) (h_lcm2 : Nat.lcm n 45 = 180) : n = 100 := 
by
  sorry

end find_n_from_lcms_l27_27153


namespace successfully_served_pizzas_l27_27195

-- Defining the conditions
def total_pizzas_served : ℕ := 9
def pizzas_returned : ℕ := 6

-- Stating the theorem
theorem successfully_served_pizzas :
  total_pizzas_served - pizzas_returned = 3 :=
by
  -- Since this is only the statement, the proof is omitted using sorry
  sorry

end successfully_served_pizzas_l27_27195


namespace fraction_identity_l27_27386

variable (a b : ℚ) (h : a / b = 2 / 3)

theorem fraction_identity : a / (a - b) = -2 :=
by
  sorry

end fraction_identity_l27_27386


namespace isosceles_triangle_sides_l27_27913

theorem isosceles_triangle_sides (r R : ℝ) (a b c : ℝ) (h1 : r = 3 / 2) (h2 : R = 25 / 8)
  (h3 : a = c) (h4 : 5 = a) (h5 : 6 = b) : 
  ∃ a b c, a = 5 ∧ c = 5 ∧ b = 6 := by 
  sorry

end isosceles_triangle_sides_l27_27913


namespace floor_width_l27_27041

theorem floor_width (W : ℕ) (hAreaFloor: 10 * W - 64 = 16) : W = 8 :=
by
  -- the proof should be added here
  sorry

end floor_width_l27_27041


namespace ending_point_divisible_by_9_l27_27544

theorem ending_point_divisible_by_9 (n : ℕ) (ending_point : ℕ) 
  (h1 : n = 11110) 
  (h2 : ∃ k : ℕ, 10 + 9 * k = ending_point) : 
  ending_point = 99999 := 
  sorry

end ending_point_divisible_by_9_l27_27544


namespace log_sum_greater_than_two_l27_27191

variables {x y a m : ℝ}

theorem log_sum_greater_than_two
  (hx : 0 < x) (hxy : x < y) (hya : y < a) (ha1 : a < 1)
  (hm : m = Real.log x / Real.log a + Real.log y / Real.log a) : m > 2 :=
sorry

end log_sum_greater_than_two_l27_27191


namespace shortest_distance_to_circle_l27_27972

def center : ℝ × ℝ := (8, 7)
def radius : ℝ := 5
def point : ℝ × ℝ := (1, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

theorem shortest_distance_to_circle :
  distance point center - radius = Real.sqrt 130 - 5 :=
by
  sorry

end shortest_distance_to_circle_l27_27972


namespace rotate_cd_to_cd_l27_27903

def rotate180 (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, -p.2)

theorem rotate_cd_to_cd' :
  let C := (-1, 2)
  let C' := (1, -2)
  let D := (3, 2)
  let D' := (-3, -2)
  rotate180 C = C' ∧ rotate180 D = D' :=
by
  sorry

end rotate_cd_to_cd_l27_27903


namespace minimum_value_a_plus_3b_plus_9c_l27_27958

open Real

theorem minimum_value_a_plus_3b_plus_9c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 :=
sorry

end minimum_value_a_plus_3b_plus_9c_l27_27958


namespace cross_product_scaled_v_and_w_l27_27372

-- Assume the vectors and their scalar multiple
def v : ℝ × ℝ × ℝ := (3, 1, 4)
def w : ℝ × ℝ × ℝ := (-2, 2, -3)
def v_scaled : ℝ × ℝ × ℝ := (6, 2, 8)

-- Define the cross product function
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.1 * b.2.2 - a.2.2 * b.1,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_scaled_v_and_w :
  cross_product v_scaled w = (-22, -2, 16) :=
by
  sorry

end cross_product_scaled_v_and_w_l27_27372


namespace findB_coords_l27_27882

namespace ProofProblem

-- Define point A with its coordinates.
def A : ℝ × ℝ := (-3, 2)

-- Define a property that checks if a line segment AB is parallel to the x-axis.
def isParallelToXAxis (A B : (ℝ × ℝ)) : Prop :=
  A.2 = B.2

-- Define a property that checks if the length of line segment AB is 4.
def hasLengthFour (A B : (ℝ × ℝ)) : Prop :=
  abs (A.1 - B.1) = 4

-- The proof problem statement.
theorem findB_coords :
  ∃ B : ℝ × ℝ, isParallelToXAxis A B ∧ hasLengthFour A B ∧ (B = (-7, 2) ∨ B = (1, 2)) :=
  sorry

end ProofProblem

end findB_coords_l27_27882


namespace complex_multiplication_l27_27609

theorem complex_multiplication :
  ∀ (i : ℂ), i * i = -1 → i * (1 + i) = -1 + i :=
by
  intros i hi
  sorry

end complex_multiplication_l27_27609


namespace angle_solution_l27_27027

/-!
  Given:
  k + 90° = 360°

  Prove:
  k = 270°
-/

theorem angle_solution (k : ℝ) (h : k + 90 = 360) : k = 270 :=
by
  sorry

end angle_solution_l27_27027


namespace max_value_of_expression_l27_27063

open Real

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) : 
  x^4 * y^2 * z ≤ 1024 / 7^7 :=
sorry

end max_value_of_expression_l27_27063


namespace triangle_BFD_ratio_l27_27538

theorem triangle_BFD_ratio (x : ℝ) : 
  let AF := 3 * x
  let FE := x
  let ED := x
  let DC := 3 * x
  let side_square := AF + FE
  let area_square := side_square^2
  let area_triangle_BFD := area_square - (1/2 * AF * side_square + 1/2 * side_square * FE + 1/2 * ED * DC)
  (area_triangle_BFD / area_square) = 7 / 16 := 
by
  sorry

end triangle_BFD_ratio_l27_27538


namespace roses_to_sister_l27_27380

theorem roses_to_sister (total_roses roses_to_mother roses_to_grandmother roses_kept : ℕ) 
  (h1 : total_roses = 20)
  (h2 : roses_to_mother = 6)
  (h3 : roses_to_grandmother = 9)
  (h4 : roses_kept = 1) : 
  total_roses - (roses_to_mother + roses_to_grandmother + roses_kept) = 4 :=
by
  sorry

end roses_to_sister_l27_27380


namespace find_angle_A_area_bound_given_a_l27_27254

-- (1) Given the condition, prove that \(A = \frac{\pi}{3}\).
theorem find_angle_A
  {A B C : ℝ} {a b c : ℝ}
  (h1 : a / (Real.cos A) + b / (Real.cos B) + c / (Real.cos C) = (Real.sqrt 3) * c * (Real.sin B) / (Real.cos B * Real.cos C)) :
  A = Real.pi / 3 :=
sorry

-- (2) Given a = 4, prove the area S satisfies \(S \leq 4\sqrt{3}\).
theorem area_bound_given_a
  {A B C : ℝ} {a b c S : ℝ}
  (ha : a = 4)
  (hA : A = Real.pi / 3)
  (h1 : a / (Real.cos A) + b / (Real.cos B) + c / (Real.cos C) = (Real.sqrt 3) * c * (Real.sin B) / (Real.cos B * Real.cos C))
  (hS : S = 1 / 2 * b * c * Real.sin A) :
  S ≤ 4 * Real.sqrt 3 :=
sorry

end find_angle_A_area_bound_given_a_l27_27254


namespace ways_to_select_computers_l27_27477

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the number of Type A and Type B computers
def num_type_a := 4
def num_type_b := 5

-- Define the total number of computers to select
def total_selected := 3

-- Define the calculation for number of ways to select the computers ensuring both types are included
def ways_to_select := binomial num_type_a 2 * binomial num_type_b 1 + binomial num_type_a 1 * binomial num_type_b 2

-- State the theorem
theorem ways_to_select_computers : ways_to_select = 70 :=
by
  -- Proof will be provided here
  sorry

end ways_to_select_computers_l27_27477


namespace downstream_speed_l27_27116

def V_u : ℝ := 26
def V_m : ℝ := 28
def V_s : ℝ := V_m - V_u
def V_d : ℝ := V_m + V_s

theorem downstream_speed : V_d = 30 := by
  sorry

end downstream_speed_l27_27116


namespace speed_of_sisters_sailboat_l27_27394

variable (v_j : ℝ) (d : ℝ) (t_wait : ℝ)

-- Conditions
def janet_speed : Prop := v_j = 30
def lake_distance : Prop := d = 60
def janet_wait_time : Prop := t_wait = 3

-- Question to Prove
def sister_speed (v_s : ℝ) : Prop :=
  janet_speed v_j ∧ lake_distance d ∧ janet_wait_time t_wait →
  v_s = 12

-- The main theorem
theorem speed_of_sisters_sailboat (v_j d t_wait : ℝ) (h1 : janet_speed v_j) (h2 : lake_distance d) (h3 : janet_wait_time t_wait) :
  ∃ v_s : ℝ, sister_speed v_j d t_wait v_s :=
by
  sorry

end speed_of_sisters_sailboat_l27_27394


namespace population_definition_l27_27518

variable (students : Type) (weights : students → ℝ) (sample : Fin 50 → students)
variable (total_students : Fin 300 → students)
variable (is_selected : students → Prop)

theorem population_definition :
    (∀ s, is_selected s ↔ ∃ i, sample i = s) →
    (population = {w : ℝ | ∃ s, w = weights s}) ↔
    (population = {w : ℝ | ∃ s, w = weights s ∧ ∃ i, total_students i = s}) := by
  sorry

end population_definition_l27_27518


namespace arithmetic_sequence_n_equals_8_l27_27294

theorem arithmetic_sequence_n_equals_8 :
  (∀ (a b c : ℕ), a + (1 / 4) * c = 2 * (1 / 2) * b) → ∃ n : ℕ, n = 8 :=
by 
  sorry

end arithmetic_sequence_n_equals_8_l27_27294


namespace min_value_expr_l27_27653

theorem min_value_expr (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + ((y / x) - 1)^2 + ((z / y) - 1)^2 + ((5 / z) - 1)^2 = 9 :=
sorry

end min_value_expr_l27_27653


namespace compute_c_over_d_l27_27174

noncomputable def RootsResult (a b c d : ℝ) : Prop :=
  (3 * 4 + 4 * 5 + 5 * 3 = - c / a) ∧ (3 * 4 * 5 = - d / a)

theorem compute_c_over_d (a b c d : ℝ)
  (h1 : (a * 3 ^ 3 + b * 3 ^ 2 + c * 3 + d = 0))
  (h2 : (a * 4 ^ 3 + b * 4 ^ 2 + c * 4 + d = 0))
  (h3 : (a * 5 ^ 3 + b * 5 ^ 2 + c * 5 + d = 0)) 
  (hr : RootsResult a b c d) :
  c / d = 47 / 60 := 
by
  sorry

end compute_c_over_d_l27_27174


namespace maximize_abs_sum_solution_problem_l27_27770

theorem maximize_abs_sum_solution :
ℤ → ℤ → Ennreal := sorry

theorem problem :
  (∃ (x y : ℤ), 6 * x^2 + 5 * x * y + y^2 = 6 * x + 2 * y + 7 ∧ 
  x = -8 ∧ y = 25 ∧ (maximize_abs_sum_solution x y = 33)) := sorry

end maximize_abs_sum_solution_problem_l27_27770


namespace quadratic_inequality_solution_l27_27652

theorem quadratic_inequality_solution (b c : ℝ) 
    (h1 : ∀ x : ℝ, (1 < x ∧ x < 2) → x^2 + b * x + c < 0) :
    b + c = -1 :=
sorry

end quadratic_inequality_solution_l27_27652


namespace number_of_ways_to_score_l27_27036

-- Define the conditions
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def score_red : ℕ := 2
def score_white : ℕ := 1
def total_balls : ℕ := 5
def min_score : ℕ := 7

-- Prove the equivalent proof problem
theorem number_of_ways_to_score :
  ∃ ways : ℕ, 
    (ways = ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
             (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
             (Nat.choose red_balls 2) * (Nat.choose white_balls 3))) ∧
    ways = 186 :=
by
  let ways := ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
               (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
               (Nat.choose red_balls 2) * (Nat.choose white_balls 3))
  use ways
  constructor
  . rfl
  . sorry

end number_of_ways_to_score_l27_27036


namespace inclination_angle_of_line_l27_27750

theorem inclination_angle_of_line (θ : Real) 
  (h : θ = Real.tan 45) : θ = 90 :=
sorry

end inclination_angle_of_line_l27_27750


namespace distance_between_Jay_and_Sarah_l27_27555

theorem distance_between_Jay_and_Sarah 
  (time_in_hours : ℝ)
  (jay_speed_per_12_minutes : ℝ)
  (sarah_speed_per_36_minutes : ℝ)
  (total_distance : ℝ) :
  time_in_hours = 2 →
  jay_speed_per_12_minutes = 1 →
  sarah_speed_per_36_minutes = 3 →
  total_distance = 20 :=
by
  intros time_in_hours_eq jay_speed_eq sarah_speed_eq
  sorry

end distance_between_Jay_and_Sarah_l27_27555


namespace sin_330_correct_l27_27017

noncomputable def sin_330 : ℝ := sorry

theorem sin_330_correct : sin_330 = -1 / 2 :=
  sorry

end sin_330_correct_l27_27017


namespace find_value_of_expression_l27_27251

theorem find_value_of_expression
  (a b c d : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : d ≥ 0)
  (h₄ : a / (b + c + d) = b / (a + c + d))
  (h₅ : b / (a + c + d) = c / (a + b + d))
  (h₆ : c / (a + b + d) = d / (a + b + c))
  (h₇ : d / (a + b + c) = a / (b + c + d)) :
  (a + b) / (c + d) + (b + c) / (a + d) + (c + d) / (a + b) + (d + a) / (b + c) = 4 :=
by sorry

end find_value_of_expression_l27_27251


namespace sin_squared_plus_sin_double_eq_one_l27_27778

variable (α : ℝ)
variable (h : Real.tan α = 1 / 2)

theorem sin_squared_plus_sin_double_eq_one : Real.sin α ^ 2 + Real.sin (2 * α) = 1 :=
by
  -- sorry to indicate the proof is skipped
  sorry

end sin_squared_plus_sin_double_eq_one_l27_27778


namespace smallest_whole_number_larger_than_any_triangle_perimeter_l27_27651

def is_valid_triangle (a b c : ℕ) : Prop := 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem smallest_whole_number_larger_than_any_triangle_perimeter : 
  ∀ (s : ℕ), 16 < s ∧ s < 30 → is_valid_triangle 7 23 s → 
    60 = (Nat.succ (7 + 23 + s - 1)) := 
by 
  sorry

end smallest_whole_number_larger_than_any_triangle_perimeter_l27_27651


namespace cos_alpha_minus_pi_over_4_l27_27214

theorem cos_alpha_minus_pi_over_4 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.tan α = 2) :
  Real.cos (α - π / 4) = 3 * Real.sqrt 10 / 10 :=
sorry

end cos_alpha_minus_pi_over_4_l27_27214


namespace each_boy_makes_14_l27_27201

/-- Proof that each boy makes 14 dollars given the initial conditions and sales scheme. -/
theorem each_boy_makes_14 (victor_shrimp : ℕ)
                          (austin_shrimp : ℕ)
                          (brian_shrimp : ℕ)
                          (total_shrimp : ℕ)
                          (sets_sold : ℕ)
                          (total_earnings : ℕ)
                          (individual_earnings : ℕ)
                          (h1 : victor_shrimp = 26)
                          (h2 : austin_shrimp = victor_shrimp - 8)
                          (h3 : brian_shrimp = (victor_shrimp + austin_shrimp) / 2)
                          (h4 : total_shrimp = victor_shrimp + austin_shrimp + brian_shrimp)
                          (h5 : sets_sold = total_shrimp / 11)
                          (h6 : total_earnings = sets_sold * 7)
                          (h7 : individual_earnings = total_earnings / 3):
  individual_earnings = 14 := 
by
  sorry

end each_boy_makes_14_l27_27201


namespace selling_price_is_320_l27_27699

noncomputable def sales_volume (x : ℝ) : ℝ := 8000 / x

def cost_price : ℝ := 180

def desired_profit : ℝ := 3500

def selling_price_for_desired_profit (x : ℝ) : Prop :=
  (x - cost_price) * sales_volume x = desired_profit

/-- The selling price of the small electrical appliance to achieve a daily sales profit 
    of $3500 dollars is $320 dollars. -/
theorem selling_price_is_320 : selling_price_for_desired_profit 320 :=
by
  -- We skip the proof as per instructions
  sorry

end selling_price_is_320_l27_27699


namespace angle_A_range_l27_27894

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def strictly_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x < y ∧ x ∈ I ∧ y ∈ I → f x < f y

theorem angle_A_range (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_strict_inc : strictly_increasing f {x | 0 < x})
  (h_f_half : f (1 / 2) = 0)
  (A : ℝ)
  (h_cos_A : f (Real.cos A) < 0) :
  (π / 3 < A ∧ A < π / 2) ∨ (2 * π / 3 < A ∧ A < π) :=
by
  sorry

end angle_A_range_l27_27894


namespace sufficient_condition_l27_27898

theorem sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) 2, (1/2 : ℝ) * x^2 - a ≥ 0) → a ≤ 0 :=
by
  sorry

end sufficient_condition_l27_27898


namespace integer_solutions_l27_27534

-- Define the equation to be solved
def equation (x y : ℤ) : Prop := x * y + 3 * x - 5 * y + 3 = 0

-- Define the solutions
def solution_set : List (ℤ × ℤ) := 
  [(-13,-2), (-4,-1), (-1,0), (2, 3), (3, 6), (4, 15), (6, -21),
   (7, -12), (8, -9), (11, -6), (14, -5), (23, -4)]

-- The theorem stating the solutions are correct
theorem integer_solutions : ∀ (x y : ℤ), (x, y) ∈ solution_set → equation x y :=
by
  sorry

end integer_solutions_l27_27534


namespace sum_of_digits_is_10_l27_27135

def sum_of_digits_of_expression : ℕ :=
  let expression := 2^2010 * 5^2008 * 7
  let simplified := 280000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
  2 + 8

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2008 * 7 is 10 -/
theorem sum_of_digits_is_10 :
  sum_of_digits_of_expression = 10 :=
by sorry

end sum_of_digits_is_10_l27_27135


namespace unusual_numbers_exist_l27_27014

noncomputable def n1 : ℕ := 10 ^ 100 - 1
noncomputable def n2 : ℕ := 10 ^ 100 / 2 - 1

theorem unusual_numbers_exist : 
  (n1 ^ 3 % 10 ^ 100 = n1 ∧ n1 ^ 2 % 10 ^ 100 ≠ n1) ∧ 
  (n2 ^ 3 % 10 ^ 100 = n2 ∧ n2 ^ 2 % 10 ^ 100 ≠ n2) :=
by
  sorry

end unusual_numbers_exist_l27_27014


namespace octadecagon_diagonals_l27_27679

def num_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octadecagon_diagonals : num_of_diagonals 18 = 135 := by
  sorry

end octadecagon_diagonals_l27_27679


namespace radius_of_larger_circle_l27_27323

theorem radius_of_larger_circle (r R AC BC AB : ℝ)
  (h1 : R = 4 * r)
  (h2 : AC = 8 * r)
  (h3 : BC^2 + AB^2 = AC^2)
  (h4 : AB = 16) :
  R = 32 :=
by
  sorry

end radius_of_larger_circle_l27_27323


namespace mass_of_man_is_correct_l27_27080

-- Definitions for conditions
def length_of_boat : ℝ := 3
def breadth_of_boat : ℝ := 2
def sinking_depth : ℝ := 0.012
def density_of_water : ℝ := 1000

-- Volume of water displaced
def volume_displaced := length_of_boat * breadth_of_boat * sinking_depth

-- Mass of the man
def mass_of_man := density_of_water * volume_displaced

-- Prove that the mass of the man is 72 kg
theorem mass_of_man_is_correct : mass_of_man = 72 := by
  sorry

end mass_of_man_is_correct_l27_27080


namespace intersection_domains_l27_27190

def domain_f : Set ℝ := {x : ℝ | x < 1}
def domain_g : Set ℝ := {x : ℝ | x > -1}

theorem intersection_domains : {x : ℝ | x < 1} ∩ {x : ℝ | x > -1} = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end intersection_domains_l27_27190


namespace inequality_solution_l27_27157

theorem inequality_solution (a : ℝ)
  (h : ∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → a * x^2 - 2 * x + 2 > 0) :
  a > 1/2 := 
sorry

end inequality_solution_l27_27157


namespace rationalize_denominator_l27_27672

theorem rationalize_denominator (t : ℝ) (h : t = 1 / (1 - Real.sqrt (Real.sqrt 2))) : 
  t = -(1 + Real.sqrt (Real.sqrt 2)) * (1 + Real.sqrt 2) :=
by
  sorry

end rationalize_denominator_l27_27672


namespace find_second_angle_l27_27138

noncomputable def angle_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem find_second_angle
  (A B C : ℝ)
  (hA : A = 32)
  (hC : C = 2 * A - 12)
  (hB : B = 3 * A)
  (h_sum : angle_in_triangle A B C) :
  B = 96 :=
by sorry

end find_second_angle_l27_27138


namespace inequality_proof_l27_27567

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : b > 0) : (1 / a < 1 / b) :=
sorry

end inequality_proof_l27_27567


namespace no_square_pair_l27_27387

/-- 
Given integers a, b, and c, where c > 0, if a(a + 4) = c^2 and (a + 2 + c)(a + 2 - c) = 4, 
then the numbers a(a + 4) and b(b + 4) cannot both be squares.
-/
theorem no_square_pair (a b c : ℤ) (hc_pos : c > 0) (ha_eq : a * (a + 4) = c^2) 
  (hfac_eq : (a + 2 + c) * (a + 2 - c) = 4) : ¬(∃ d e : ℤ, d^2 = a * (a + 4) ∧ e^2 = b * (b + 4)) :=
by sorry

end no_square_pair_l27_27387


namespace positive_integer_pairs_divisibility_l27_27278

theorem positive_integer_pairs_divisibility (a b : ℕ) (h : a * b^2 + b + 7 ∣ a^2 * b + a + b) :
  (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ, k > 0 ∧ a = 7 * k^2 ∧ b = 7 * k :=
sorry

end positive_integer_pairs_divisibility_l27_27278


namespace sum_first_39_natural_numbers_l27_27145

theorem sum_first_39_natural_numbers :
  (39 * (39 + 1)) / 2 = 780 :=
by
  sorry

end sum_first_39_natural_numbers_l27_27145


namespace factorize_expression_l27_27887

theorem factorize_expression (a : ℝ) : 
  (a + 1) * (a + 2) + 1 / 4 = (a + 3 / 2)^2 := 
by 
  sorry

end factorize_expression_l27_27887


namespace determine_constant_l27_27558

/-- If the function f(x) = a * sin x + 3 * cos x has a maximum value of 5,
then the constant a must be ± 4. -/
theorem determine_constant (a : ℝ) (h : ∀ x : ℝ, a * Real.sin x + 3 * Real.cos x ≤ 5) :
  a = 4 ∨ a = -4 :=
sorry

end determine_constant_l27_27558


namespace parabola_vertex_l27_27926

theorem parabola_vertex (a b c : ℝ) :
  (∀ x, y = ax^2 + bx + c ↔ 
   y = a*((x+3)^2) + 4) ∧
   (∀ x y, (x, y) = ((1:ℝ), (2:ℝ))) →
   a + b + c = 3 := by
  sorry

end parabola_vertex_l27_27926


namespace rectangle_side_length_along_hypotenuse_l27_27220

-- Define the right triangle with given sides
def triangle_PQR (PR PQ QR : ℝ) : Prop := 
  PR^2 + PQ^2 = QR^2

-- Condition: Right triangle PQR with PR = 9 and PQ = 12
def PQR : Prop := triangle_PQR 9 12 (Real.sqrt (9^2 + 12^2))

-- Define the property of the rectangle
def rectangle_condition (x : ℝ) (s : ℝ) : Prop := 
  (3 / (Real.sqrt (9^2 + 12^2))) = (x / 9) ∧ s = ((9 - x) * (Real.sqrt (9^2 + 12^2)) / 9)

-- Main theorem
theorem rectangle_side_length_along_hypotenuse : 
  PQR ∧ (∃ x, rectangle_condition x 12) → (∃ s, s = 12) :=
by
  intro h
  sorry

end rectangle_side_length_along_hypotenuse_l27_27220


namespace janet_hourly_wage_l27_27840

theorem janet_hourly_wage : 
  ∃ x : ℝ, 
    (20 * x + (5 * 20 + 7 * 20) = 1640) ∧ 
    x = 70 :=
by
  use 70
  sorry

end janet_hourly_wage_l27_27840


namespace rationalize_denominator_l27_27283

theorem rationalize_denominator :
  (Real.sqrt (5 / 12)) = ((Real.sqrt 15) / 6) :=
sorry

end rationalize_denominator_l27_27283


namespace intersection_complement_M_N_l27_27131

def M : Set ℝ := { x | x > 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }
def complement_M : Set ℝ := { x | x ≤ 1 }

theorem intersection_complement_M_N :
  (complement_M ∩ N) = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_complement_M_N_l27_27131


namespace solve_phi_l27_27615

-- Define the problem
noncomputable def f (phi x : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + phi)
noncomputable def f' (phi x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + phi)
noncomputable def g (phi x : ℝ) : ℝ := f phi x + f' phi x

-- Define the main theorem
theorem solve_phi (phi : ℝ) (h : -Real.pi < phi ∧ phi < 0) 
  (even_g : ∀ x, g phi x = g phi (-x)) : phi = -Real.pi / 3 :=
sorry

end solve_phi_l27_27615


namespace bag_weight_l27_27601

theorem bag_weight (W : ℕ) 
  (h1 : 2 * W + 82 * (2 * W) = 664) : 
  W = 4 := by
  sorry

end bag_weight_l27_27601


namespace mineral_samples_per_shelf_l27_27709

theorem mineral_samples_per_shelf (total_samples : ℕ) (num_shelves : ℕ) (h1 : total_samples = 455) (h2 : num_shelves = 7) :
  total_samples / num_shelves = 65 :=
by
  sorry

end mineral_samples_per_shelf_l27_27709


namespace x_intercept_of_perpendicular_line_l27_27695

theorem x_intercept_of_perpendicular_line (x y : ℝ) (b : ℕ) :
  let line1 := 2 * x + 3 * y
  let slope1 := -2/3
  let slope2 := 3/2
  let y_intercept := -1
  let perp_line := slope2 * x + y_intercept
  let x_intercept := 2/3
  line1 = 12 → perp_line = 0 → x = x_intercept :=
by
  sorry

end x_intercept_of_perpendicular_line_l27_27695


namespace orchid_bushes_total_l27_27925

def current_bushes : ℕ := 47
def bushes_today : ℕ := 37
def bushes_tomorrow : ℕ := 25

theorem orchid_bushes_total : current_bushes + bushes_today + bushes_tomorrow = 109 := 
by sorry

end orchid_bushes_total_l27_27925


namespace arithmetic_sequence_sum_proof_l27_27185

theorem arithmetic_sequence_sum_proof
  (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : S 17 = 170)
  (h2 : a 2000 = 2001)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h4 : ∀ n, a (n + 1) = a n + (a 2 - a 1)) :
  S 2008 = 2019044 :=
  sorry

end arithmetic_sequence_sum_proof_l27_27185


namespace problem_positive_l27_27563

theorem problem_positive : ∀ x : ℝ, x < 0 → -3 * x⁻¹ > 0 :=
by 
  sorry

end problem_positive_l27_27563


namespace find_interest_rate_l27_27192

noncomputable def compound_interest_rate (A P : ℝ) (t n : ℕ) : ℝ := sorry

theorem find_interest_rate :
  compound_interest_rate 676 625 2 1 = 0.04 := 
sorry

end find_interest_rate_l27_27192


namespace runs_in_last_match_l27_27447

theorem runs_in_last_match (W : ℕ) (R x : ℝ) 
    (hW : W = 85) 
    (hR : R = 12.4 * W) 
    (new_average : (R + x) / (W + 5) = 12) : 
    x = 26 := 
by 
  sorry

end runs_in_last_match_l27_27447


namespace initial_balance_l27_27889

-- Define the conditions given in the problem
def transferred_percent_of_balance (X : ℝ) : ℝ := 0.15 * X
def balance_after_transfer (X : ℝ) : ℝ := 0.85 * X
def final_balance_after_refund (X : ℝ) (refund : ℝ) : ℝ := 0.85 * X + refund

-- Define the given values
def refund : ℝ := 450
def final_balance : ℝ := 30000

-- The theorem statement to prove the initial balance
theorem initial_balance (X : ℝ) (h : final_balance_after_refund X refund = final_balance) : 
  X = 34564.71 :=
by
  sorry

end initial_balance_l27_27889


namespace tan_alpha_value_l27_27869

open Real

-- Define the angle alpha in the third quadrant
variable {α : ℝ}

-- Given conditions
def third_quadrant (α : ℝ) : Prop :=  π < α ∧ α < 3 * π / 2
def sin_alpha (α : ℝ) : Prop := sin α = -4 / 5

-- Statement to prove
theorem tan_alpha_value (h1 : third_quadrant α) (h2 : sin_alpha α) : tan α = 4 / 3 :=
sorry

end tan_alpha_value_l27_27869


namespace expected_amoebas_after_one_week_l27_27049

section AmoebaProblem

-- Definitions from conditions
def initial_amoebas : ℕ := 1
def split_probability : ℝ := 0.8
def days : ℕ := 7

-- Function to calculate expected amoebas
def expected_amoebas (n : ℕ) : ℝ :=
  initial_amoebas * ((2 : ℝ) ^ n) * (split_probability ^ n)

-- Theorem statement
theorem expected_amoebas_after_one_week :
  expected_amoebas days = 26.8435456 :=
by sorry

end AmoebaProblem

end expected_amoebas_after_one_week_l27_27049


namespace find_p_range_l27_27152

theorem find_p_range (p : ℝ) (A : ℝ → ℝ) :
  (A = fun x => abs x * x^2 + (p + 2) * x + 1) →
  (∀ x, 0 < x → A x ≠ 0) →
  (-4 < p ∧ p < 0) :=
by
  intro hA h_no_pos_roots
  sorry

end find_p_range_l27_27152


namespace boxes_with_no_items_l27_27030

-- Definitions of each condition as given in the problem
def total_boxes : Nat := 15
def pencil_boxes : Nat := 8
def pen_boxes : Nat := 5
def marker_boxes : Nat := 3
def pen_pencil_boxes : Nat := 4
def all_three_boxes : Nat := 1

-- The theorem to prove
theorem boxes_with_no_items : 
     (total_boxes - ((pen_pencil_boxes - all_three_boxes)
                     + (pencil_boxes - pen_pencil_boxes - all_three_boxes)
                     + (pen_boxes - pen_pencil_boxes - all_three_boxes)
                     + (marker_boxes - all_three_boxes)
                     + all_three_boxes)) = 5 := 
by 
  -- This is where the proof would go, but we'll use sorry to indicate it's skipped.
  sorry

end boxes_with_no_items_l27_27030


namespace election_result_l27_27108

theorem election_result:
  ∀ (Henry_votes India_votes Jenny_votes Ken_votes Lena_votes : ℕ)
    (counted_percentage : ℕ)
    (counted_votes : ℕ), 
    Henry_votes = 14 → 
    India_votes = 11 → 
    Jenny_votes = 10 → 
    Ken_votes = 8 → 
    Lena_votes = 2 → 
    counted_percentage = 90 → 
    counted_votes = 45 → 
    (counted_percentage * Total_votes / 100 = counted_votes) →
    (Total_votes = counted_votes * 100 / counted_percentage) →
    (Remaining_votes = Total_votes - counted_votes) →
    ((Henry_votes + Max_remaining_Votes >= Max_votes) ∨ 
    (India_votes + Max_remaining_Votes >= Max_votes) ∨ 
    (Jenny_votes + Max_remaining_Votes >= Max_votes)) →
    3 = 
    (if Henry_votes + Remaining_votes > Max_votes then 1 else 0) + 
    (if India_votes + Remaining_votes > Max_votes then 1 else 0) + 
    (if Jenny_votes + Remaining_votes > Max_votes then 1 else 0) := 
  sorry

end election_result_l27_27108


namespace cost_of_ice_cream_l27_27212

theorem cost_of_ice_cream (x : ℝ) (h1 : 10 * x = 40) : x = 4 :=
by sorry

end cost_of_ice_cream_l27_27212


namespace union_A_B_correct_l27_27670

def A : Set ℕ := {0, 1}
def B : Set ℕ := {x | 0 < x ∧ x < 3}

theorem union_A_B_correct : A ∪ B = {0, 1, 2} :=
by sorry

end union_A_B_correct_l27_27670


namespace decimal_fraction_to_percentage_l27_27881

theorem decimal_fraction_to_percentage (d : ℝ) (h : d = 0.03) : d * 100 = 3 := by
  sorry

end decimal_fraction_to_percentage_l27_27881


namespace div_40_of_prime_ge7_l27_27788

theorem div_40_of_prime_ge7 (p : ℕ) (hp_prime : Prime p) (hp_ge7 : p ≥ 7) : 40 ∣ (p^2 - 1) :=
sorry

end div_40_of_prime_ge7_l27_27788


namespace fraction_value_l27_27836

theorem fraction_value : (1 + 3 + 5) / (10 + 6 + 2) = 1 / 2 := 
by
  sorry

end fraction_value_l27_27836


namespace necessary_but_not_sufficient_condition_not_sufficient_condition_l27_27129

theorem necessary_but_not_sufficient_condition (x y : ℝ) (h : x > 0) : 
  (x > |y|) → (x > y) :=
by
  sorry

theorem not_sufficient_condition (x y : ℝ) (h : x > 0) :
  ¬ ((x > y) → (x > |y|)) :=
by
  sorry

end necessary_but_not_sufficient_condition_not_sufficient_condition_l27_27129


namespace given_system_solution_l27_27849

noncomputable def solve_system : Prop :=
  ∃ x y z : ℝ, 
  x + y + z = 1 ∧ 
  x^2 + y^2 + z^2 = 1 ∧ 
  x^3 + y^3 + z^3 = 89 / 125 ∧ 
  (x = 2 / 5 ∧ y = (3 + Real.sqrt 33) / 10 ∧ z = (3 - Real.sqrt 33) / 10 ∨ 
   x = 2 / 5 ∧ y = (3 - Real.sqrt 33) / 10 ∧ z = (3 + Real.sqrt 33) / 10 ∨ 
   x = (3 + Real.sqrt 33) / 10 ∧ y = 2 / 5 ∧ z = (3 - Real.sqrt 33) / 10 ∨ 
   x = (3 - Real.sqrt 33) / 10 ∧ y = 2 / 5 ∧ z = (3 + Real.sqrt 33) / 10 ∨ 
   x = (3 + Real.sqrt 33) / 10 ∧ y = (3 - Real.sqrt 33) / 10 ∧ z = 2 / 5 ∨ 
   x = (3 - Real.sqrt 33) / 10 ∧ y = (3 + Real.sqrt 33) / 10 ∧ z = 2 / 5)

theorem given_system_solution : solve_system :=
sorry

end given_system_solution_l27_27849


namespace function_maximum_at_1_l27_27368

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x ^ 2

theorem function_maximum_at_1 :
  ∀ x > 0, (f x ≤ f 1) :=
by
  intro x hx
  have hx_pos : 0 < x := hx
  sorry

end function_maximum_at_1_l27_27368


namespace hall_volume_l27_27838

theorem hall_volume (length width : ℝ) (h : ℝ) 
  (h_length : length = 6) 
  (h_width : width = 6) 
  (h_areas : 2 * (length * width) = 4 * (length * h)) :
  length * width * h = 108 :=
by
  sorry

end hall_volume_l27_27838


namespace problem1_problem2_l27_27361

-- Problem 1
theorem problem1 (b : ℝ) :
  4 * b^2 * (b^3 - 1) - 3 * (1 - 2 * b^2) > 4 * (b^5 - 1) :=
by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) :
  a - a * abs (-a^2 - 1) < 1 - a^2 * (a - 1) :=
by
  sorry

end problem1_problem2_l27_27361


namespace product_at_n_equals_three_l27_27438

theorem product_at_n_equals_three : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) = 120 := by
  sorry

end product_at_n_equals_three_l27_27438


namespace comprehensiveInvestigation_is_Census_l27_27155

def comprehensiveInvestigation (s: String) : Prop :=
  s = "Census"

theorem comprehensiveInvestigation_is_Census :
  comprehensiveInvestigation "Census" :=
by
  sorry

end comprehensiveInvestigation_is_Census_l27_27155


namespace total_animals_l27_27756

namespace Zoo

def snakes := 15
def monkeys := 2 * snakes
def lions := monkeys - 5
def pandas := lions + 8
def dogs := pandas / 3

theorem total_animals : snakes + monkeys + lions + pandas + dogs = 114 := by
  -- definitions from conditions
  have h_snakes : snakes = 15 := rfl
  have h_monkeys : monkeys = 2 * snakes := rfl
  have h_lions : lions = monkeys - 5 := rfl
  have h_pandas : pandas = lions + 8 := rfl
  have h_dogs : dogs = pandas / 3 := rfl
  -- sorry is used as a placeholder for the proof
  sorry

end Zoo

end total_animals_l27_27756


namespace percent_boys_in_class_l27_27413

-- Define the conditions given in the problem
def initial_ratio (b g : ℕ) : Prop := b = 3 * g / 4

def total_students_after_new_girls (total : ℕ) (new_girls : ℕ) : Prop :=
  total = 42 ∧ new_girls = 4

-- Define the percentage calculation correctness
def percentage_of_boys (boys total : ℕ) (percentage : ℚ) : Prop :=
  percentage = (boys : ℚ) / (total : ℚ) * 100

-- State the theorem to be proven
theorem percent_boys_in_class
  (b g : ℕ)   -- Number of boys and initial number of girls
  (total new_girls : ℕ) -- Total students after new girls joined and number of new girls
  (percentage : ℚ) -- The percentage of boys in the class
  (h_initial_ratio : initial_ratio b g)
  (h_total_students : total_students_after_new_girls total new_girls)
  (h_goals : g + new_girls = total - b)
  (h_correct_calc : percentage = 35.71) :
  percentage_of_boys b total percentage :=
by
  sorry

end percent_boys_in_class_l27_27413


namespace sugar_flour_difference_l27_27147

theorem sugar_flour_difference :
  ∀ (flour_required_kg sugar_required_lb flour_added_kg kg_to_lb),
    flour_required_kg = 2.25 →
    sugar_required_lb = 5.5 →
    flour_added_kg = 1 →
    kg_to_lb = 2.205 →
    (sugar_required_lb / kg_to_lb * 1000) - ((flour_required_kg - flour_added_kg) * 1000) = 1244.8 :=
by
  intros flour_required_kg sugar_required_lb flour_added_kg kg_to_lb
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- sorry is used to skip the actual proof
  sorry

end sugar_flour_difference_l27_27147


namespace garden_area_increase_l27_27482

-- Problem: Prove that changing a 40 ft by 10 ft rectangular garden into a square,
-- using the same fencing, increases the area by 225 sq ft.

theorem garden_area_increase :
  let length_orig := 40
  let width_orig := 10
  let perimeter := 2 * (length_orig + width_orig)
  let side_square := perimeter / 4
  let area_orig := length_orig * width_orig
  let area_square := side_square * side_square
  (area_square - area_orig) = 225 := 
sorry

end garden_area_increase_l27_27482


namespace minimum_problem_l27_27446

open BigOperators

theorem minimum_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1 / y) * (x + 1 / y - 2020) + (y + 1 / x) * (y + 1 / x - 2020) ≥ -2040200 := 
sorry

end minimum_problem_l27_27446


namespace yanna_kept_36_apples_l27_27475

-- Define the initial number of apples Yanna has
def initial_apples : ℕ := 60

-- Define the number of apples given to Zenny
def apples_given_to_zenny : ℕ := 18

-- Define the number of apples given to Andrea
def apples_given_to_andrea : ℕ := 6

-- The proof statement that Yanna kept 36 apples
theorem yanna_kept_36_apples : initial_apples - apples_given_to_zenny - apples_given_to_andrea = 36 := by
  sorry

end yanna_kept_36_apples_l27_27475


namespace power_function_value_l27_27231

theorem power_function_value (f : ℝ → ℝ) (h : ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a) (h₁ : f 4 = 1 / 2) :
  f (1 / 16) = 4 :=
sorry

end power_function_value_l27_27231


namespace inequality_holds_l27_27285

variable (x a : ℝ)

def tensor (x y : ℝ) : ℝ :=
  (1 - x) * (1 + y)

theorem inequality_holds (h : ∀ x : ℝ, tensor (x - a) (x + a) < 1) : -2 < a ∧ a < 0 := by
  sorry

end inequality_holds_l27_27285


namespace negation_of_prop_l27_27494

theorem negation_of_prop :
  (¬ ∀ (x y : ℝ), x^2 + y^2 ≥ 0) ↔ (∃ (x y : ℝ), x^2 + y^2 < 0) :=
by
  sorry

end negation_of_prop_l27_27494


namespace find_value_of_x_l27_27815

theorem find_value_of_x :
  ∃ x : ℝ, (0.65 * x = 0.20 * 747.50) ∧ x = 230 :=
by
  sorry

end find_value_of_x_l27_27815


namespace sin_cos_relation_l27_27366

theorem sin_cos_relation (α : ℝ) (h : Real.tan (π / 4 + α) = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1 / 2 :=
by
  sorry

end sin_cos_relation_l27_27366


namespace min_value_of_z_ineq_l27_27194

noncomputable def z (x y : ℝ) : ℝ := 2 * x + 4 * y

theorem min_value_of_z_ineq (k : ℝ) :
  (∃ x y : ℝ, (3 * x + y ≥ 0) ∧ (4 * x + 3 * y ≥ k) ∧ (z x y = -6)) ↔ k = 0 :=
by
  sorry

end min_value_of_z_ineq_l27_27194


namespace derivative_at_pi_l27_27109

noncomputable def f (x : ℝ) : ℝ := (x^2) / (Real.cos x)

theorem derivative_at_pi : deriv f π = -2 * π :=
by
  sorry

end derivative_at_pi_l27_27109


namespace circle_numbers_contradiction_l27_27269

theorem circle_numbers_contradiction :
  ¬ ∃ (f : Fin 25 → Fin 25), ∀ i : Fin 25, 
  let a := f i
  let b := f ((i + 1) % 25)
  (b = a + 10 ∨ b = a - 10 ∨ ∃ k : Int, b = a * k) :=
by
  sorry

end circle_numbers_contradiction_l27_27269


namespace cannot_be_square_of_binomial_B_l27_27610

theorem cannot_be_square_of_binomial_B (x y m n : ℝ) :
  (∃ (a b : ℝ), (3*x + 7*y) * (3*x - 7*y) = a^2 - b^2) ∧
  (∃ (a b : ℝ), ( -0.2*x - 0.3) * ( -0.2*x + 0.3) = a^2 - b^2) ∧
  (∃ (a b : ℝ), ( -3*n - m*n) * ( 3*n - m*n) = a^2 - b^2) ∧
  ¬(∃ (a b : ℝ), ( 5*m - n) * ( n - 5*m) = a^2 - b^2) :=
by
  sorry

end cannot_be_square_of_binomial_B_l27_27610


namespace smallest_k_exists_l27_27629

theorem smallest_k_exists : ∃ (k : ℕ) (n : ℕ), k = 53 ∧ k^2 + 49 = 180 * n :=
sorry

end smallest_k_exists_l27_27629


namespace hypotenuse_length_l27_27015

theorem hypotenuse_length (a b : ℤ) (h₀ : a = 15) (h₁ : b = 36) : 
  ∃ c : ℤ, c^2 = a^2 + b^2 ∧ c = 39 := 
by {
  sorry
}

end hypotenuse_length_l27_27015


namespace median_of_right_triangle_l27_27688

theorem median_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : 
  c / 2 = 5 :=
by
  rw [h3]
  norm_num

end median_of_right_triangle_l27_27688


namespace some_base_value_l27_27860

noncomputable def some_base (x y : ℝ) (h1 : x * y = 1) (h2 : (some_base : ℝ) → (some_base ^ (x + y))^2 / (some_base ^ (x - y))^2 = 2401) : ℝ :=
  7

theorem some_base_value (x y : ℝ) (h1 : x * y = 1) (h2 : ∀ some_base : ℝ, (some_base ^ (x + y))^2 / (some_base ^ (x - y))^2 = 2401) : some_base x y h1 h2 = 7 :=
by
  sorry

end some_base_value_l27_27860


namespace intersect_P_Q_l27_27216

open Set

def P : Set ℤ := { x | (x - 3) * (x - 6) ≤ 0 }
def Q : Set ℤ := { 5, 7 }

theorem intersect_P_Q : P ∩ Q = {5} :=
sorry

end intersect_P_Q_l27_27216


namespace sum_bn_l27_27525

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, a n = a 0 + n * d

def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n-1))) / 2

def geometric_sequence (a : ℕ → ℕ) :=
  ∃ r : ℕ, a 1 = a 0 * r ∧ a 2 = a 1 * r

-- Given S_5 = 35
def S5_property (S : ℕ → ℕ) := S 5 = 35

-- a_1, a_4, a_{13} is a geometric sequence
def a1_a4_a13_geometric_sequence (a : ℕ → ℕ) :=
  ∃ r : ℕ, a 4 = a 1 * r ∧ a 13 = a 4 * r

-- Define the sequence b_n and conditions
def bn_prop (a b : ℕ → ℕ) := ∀ n : ℕ, b n = a n * (2^(n-1))

-- Main theorem
theorem sum_bn {a b : ℕ → ℕ} {S T : ℕ → ℕ} (h_a : arithmetic_sequence a 2) (h_S5 : S5_property S) (h_geo : a1_a4_a13_geometric_sequence a) (h_bn : bn_prop a b)
  : ∀ n : ℕ, T n = 1 + (2 * n - 1) * 2^n := sorry

end sum_bn_l27_27525


namespace a6_is_3_l27_27111

noncomputable def a4 := 8 / 2 -- Placeholder for positive root
noncomputable def a8 := 8 / 2 -- Placeholder for the second root (we know they are both the same for now)
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 2) = (a (n + 1))^2

theorem a6_is_3 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a4_a8: a 4 = a4) (h_a4_a8_root : a 8 = a8) : 
  a 6 = 3 :=
by
  sorry

end a6_is_3_l27_27111


namespace lcm_two_numbers_l27_27183

theorem lcm_two_numbers (a b : ℕ) (h1 : a * b = 17820) (h2 : Nat.gcd a b = 12) : Nat.lcm a b = 1485 := 
by
  sorry

end lcm_two_numbers_l27_27183


namespace simplify_expression_l27_27865

theorem simplify_expression : (- (1 / 343 : ℝ)) ^ (-2 / 3 : ℝ) = 49 :=
by 
  sorry

end simplify_expression_l27_27865


namespace rosy_has_14_fish_l27_27648

-- Define the number of Lilly's fish
def lilly_fish : ℕ := 10

-- Define the total number of fish
def total_fish : ℕ := 24

-- Define the number of Rosy's fish, which we need to prove equals 14
def rosy_fish : ℕ := total_fish - lilly_fish

-- Prove that Rosy has 14 fish
theorem rosy_has_14_fish : rosy_fish = 14 := by
  sorry

end rosy_has_14_fish_l27_27648


namespace workbook_arrangement_l27_27952

-- Define the condition of having different Korean and English workbooks
variables (K1 K2 : Type) (E1 E2 : Type)

-- The main theorem statement
theorem workbook_arrangement :
  ∃ (koreanWorkbooks englishWorkbooks : List (Type)), 
  (koreanWorkbooks.length = 2) ∧
  (englishWorkbooks.length = 2) ∧
  (∀ wb ∈ (koreanWorkbooks ++ englishWorkbooks), wb ≠ wb) ∧
  (∃ arrangements : Nat,
    arrangements = 12) :=
  sorry

end workbook_arrangement_l27_27952


namespace sin_sum_angles_36_108_l27_27880

theorem sin_sum_angles_36_108 (A B C : ℝ) (h_sum : A + B + C = 180)
  (h_angle : A = 36 ∨ A = 108 ∨ B = 36 ∨ B = 108 ∨ C = 36 ∨ C = 108) :
  Real.sin (5 * A) + Real.sin (5 * B) + Real.sin (5 * C) = 0 :=
by
  sorry

end sin_sum_angles_36_108_l27_27880


namespace find_value_of_s_l27_27070

theorem find_value_of_s
  (a b c w s p : ℕ)
  (h₁ : a + b = w)
  (h₂ : w + c = s)
  (h₃ : s + a = p)
  (h₄ : b + c + p = 16) :
  s = 8 :=
sorry

end find_value_of_s_l27_27070


namespace evaluate_expression_l27_27022

theorem evaluate_expression (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 :=
by
  sorry

end evaluate_expression_l27_27022


namespace football_team_gain_l27_27452

theorem football_team_gain (G : ℤ) :
  (-5 + G = 2) → (G = 7) :=
by
  intro h
  sorry

end football_team_gain_l27_27452


namespace outfit_count_l27_27767

theorem outfit_count 
  (S P T J : ℕ) 
  (hS : S = 8) 
  (hP : P = 5) 
  (hT : T = 4) 
  (hJ : J = 3) : 
  S * P * (T + 1) * (J + 1) = 800 := by 
  sorry

end outfit_count_l27_27767


namespace jar_marbles_difference_l27_27440

theorem jar_marbles_difference (a b : ℕ) (h1 : 9 * a = 9 * b) (h2 : 2 * a + b = 135) : 8 * b - 7 * a = 45 := by
  sorry

end jar_marbles_difference_l27_27440


namespace imaginary_number_condition_fourth_quadrant_condition_l27_27515

-- Part 1: Prove that if \( z \) is purely imaginary, then \( m = 0 \)
theorem imaginary_number_condition (m : ℝ) :
  (m * (m + 2) = 0) ∧ (m^2 + m - 2 ≠ 0) → m = 0 :=
by
  sorry

-- Part 2: Prove that if \( z \) is in the fourth quadrant, then \( 0 < m < 1 \)
theorem fourth_quadrant_condition (m : ℝ) :
  (m * (m + 2) > 0) ∧ (m^2 + m - 2 < 0) → (0 < m ∧ m < 1) :=
by
  sorry

end imaginary_number_condition_fourth_quadrant_condition_l27_27515


namespace largest_both_writers_editors_l27_27945

-- Define the conditions
def writers : ℕ := 45
def editors_gt : ℕ := 38
def total_attendees : ℕ := 90
def both_writers_editors (x : ℕ) : ℕ := x
def neither_writers_editors (x : ℕ) : ℕ := x / 2

-- Define the main proof statement
theorem largest_both_writers_editors :
  ∃ x : ℕ, x ≤ 4 ∧
  (writers + (editors_gt + (0 : ℕ)) + neither_writers_editors x + both_writers_editors x = total_attendees) :=
sorry

end largest_both_writers_editors_l27_27945


namespace oil_vinegar_new_ratio_l27_27003

theorem oil_vinegar_new_ratio (initial_oil initial_vinegar new_vinegar : ℕ) 
    (h1 : initial_oil / initial_vinegar = 3 / 1)
    (h2 : new_vinegar = (2 * initial_vinegar)) :
    initial_oil / new_vinegar = 3 / 2 :=
by
  sorry

end oil_vinegar_new_ratio_l27_27003


namespace percentage_of_y_in_relation_to_25_percent_of_x_l27_27680

variable (y x : ℕ) (p : ℕ)

-- Conditions
def condition1 : Prop := (y = (p * 25 * x) / 10000)
def condition2 : Prop := (y * x = 100 * 100)
def condition3 : Prop := (y = 125)

-- The proof goal
theorem percentage_of_y_in_relation_to_25_percent_of_x :
  condition1 y x p ∧ condition2 y x ∧ condition3 y → ((y * 100) / (25 * x / 100) = 625)
:= by
-- Here we would insert the proof steps, but they are omitted as per the requirements.
sorry

end percentage_of_y_in_relation_to_25_percent_of_x_l27_27680


namespace number_of_n_for_prime_l27_27173

theorem number_of_n_for_prime (n : ℕ) : (n > 0) → ∃! n, Nat.Prime (n * (n + 2)) :=
by 
  sorry

end number_of_n_for_prime_l27_27173


namespace find_b_in_triangle_l27_27914

-- Given conditions
variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : a = 3)
variable (h2 : c = 2 * Real.sqrt 3)
variable (h3 : b * Real.sin A = a * Real.cos (B + Real.pi / 6))

-- The proof goal
theorem find_b_in_triangle (h1 : a = 3) (h2 : c = 2 * Real.sqrt 3) (h3 : b * Real.sin A = a * Real.cos (B + Real.pi / 6)) : b = Real.sqrt 3 :=
sorry

end find_b_in_triangle_l27_27914


namespace benny_final_comic_books_l27_27904

-- Define the initial number of comic books
def initial_comic_books : ℕ := 22

-- Define the comic books sold (half of the initial)
def comic_books_sold : ℕ := initial_comic_books / 2

-- Define the comic books left after selling half
def comic_books_left_after_sale : ℕ := initial_comic_books - comic_books_sold

-- Define the number of comic books bought
def comic_books_bought : ℕ := 6

-- Define the final number of comic books
def final_comic_books : ℕ := comic_books_left_after_sale + comic_books_bought

-- Statement to prove that Benny has 17 comic books at the end
theorem benny_final_comic_books : final_comic_books = 17 := by
  sorry

end benny_final_comic_books_l27_27904


namespace union_with_complement_l27_27642

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l27_27642


namespace car_late_speed_l27_27934

theorem car_late_speed :
  ∀ (d : ℝ) (t_on_time : ℝ) (t_late : ℝ) (v_on_time : ℝ) (v_late : ℝ),
  d = 225 →
  v_on_time = 60 →
  t_on_time = d / v_on_time →
  t_late = t_on_time + 0.75 →
  v_late = d / t_late →
  v_late = 50 :=
by
  intros d t_on_time t_late v_on_time v_late hd hv_on_time ht_on_time ht_late hv_late
  sorry

end car_late_speed_l27_27934


namespace CarriesJellybeanCount_l27_27360

-- Definitions based on conditions in part a)
def BertBoxJellybeans : ℕ := 150
def BertBoxVolume : ℕ := 6
def CarriesBoxVolume : ℕ := 3 * 2 * 4 * BertBoxVolume -- (3 * height, 2 * width, 4 * length)

-- Theorem statement in Lean based on part c)
theorem CarriesJellybeanCount : (CarriesBoxVolume / BertBoxVolume) * BertBoxJellybeans = 3600 := by 
  sorry

end CarriesJellybeanCount_l27_27360


namespace Jake_has_8_peaches_l27_27794

variable (Steven Jill Jake : ℕ)

-- Conditions
axiom h1 : Steven = 15
axiom h2 : Steven = Jill + 14
axiom h3 : Jake = Steven - 7

-- Goal
theorem Jake_has_8_peaches : Jake = 8 := by
  sorry

end Jake_has_8_peaches_l27_27794


namespace volume_of_pyramid_l27_27141

theorem volume_of_pyramid (V_cube : ℝ) (h : ℝ) (A : ℝ) (V_pyramid : ℝ) : 
  V_cube = 27 → 
  h = 3 → 
  A = 4.5 → 
  V_pyramid = (1/3) * A * h → 
  V_pyramid = 4.5 := 
by 
  intros V_cube_eq h_eq A_eq V_pyramid_eq 
  sorry

end volume_of_pyramid_l27_27141


namespace product_complex_numbers_l27_27417

noncomputable def Q : ℂ := 3 + 4 * Complex.I
noncomputable def E : ℂ := 2 * Complex.I
noncomputable def D : ℂ := 3 - 4 * Complex.I
noncomputable def R : ℝ := 2

theorem product_complex_numbers : Q * E * D * (R : ℂ) = 100 * Complex.I := by
  sorry

end product_complex_numbers_l27_27417


namespace find_m_if_even_l27_27430

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def my_function (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem find_m_if_even (m : ℝ) :
  is_even_function (my_function m) → m = 2 := 
by
  sorry

end find_m_if_even_l27_27430


namespace total_cost_of_plates_and_cups_l27_27486

theorem total_cost_of_plates_and_cups (P C : ℝ) 
  (h : 20 * P + 40 * C = 1.50) : 
  100 * P + 200 * C = 7.50 :=
by
  -- proof here
  sorry

end total_cost_of_plates_and_cups_l27_27486


namespace hexagon_circle_ratio_correct_l27_27480

noncomputable def hexagon_circle_area_ratio (s r : ℝ) (h : 6 * s = 2 * π * r) : ℝ :=
  let A_hex := (3 * Real.sqrt 3 / 2) * s^2
  let A_circ := π * r^2
  (A_hex / A_circ)

theorem hexagon_circle_ratio_correct (s r : ℝ) (h : 6 * s = 2 * π * r) :
    hexagon_circle_area_ratio s r h = (π * Real.sqrt 3 / 6) :=
sorry

end hexagon_circle_ratio_correct_l27_27480


namespace mink_babies_l27_27441

theorem mink_babies (B : ℕ) (h_coats : 7 * 15 = 105)
    (h_minks: 30 + 30 * B = 210) :
  B = 6 :=
by
  sorry

end mink_babies_l27_27441


namespace intersection_points_number_of_regions_l27_27743

-- Given n lines on a plane, any two of which are not parallel
-- and no three of which intersect at the same point,
-- prove the number of intersection points of these lines

theorem intersection_points (n : ℕ) (h_n : 0 < n) : 
  ∃ a_n : ℕ, a_n = n * (n - 1) / 2 := by
  sorry

-- Given n lines on a plane, any two of which are not parallel
-- and no three of which intersect at the same point,
-- prove the number of regions these lines form

theorem number_of_regions (n : ℕ) (h_n : 0 < n) :
  ∃ R_n : ℕ, R_n = n * (n + 1) / 2 + 1 := by
  sorry

end intersection_points_number_of_regions_l27_27743


namespace determine_selling_price_for_daily_profit_determine_max_profit_and_selling_price_l27_27026

-- Cost price per souvenir
def cost_price : ℕ := 40

-- Minimum selling price
def min_selling_price : ℕ := 44

-- Maximum selling price
def max_selling_price : ℕ := 60

-- Units sold if selling price is min_selling_price
def units_sold_at_min_price : ℕ := 300

-- Units sold decreases by 10 for every 1 yuan increase in selling price
def decrease_in_units (increase : ℕ) : ℕ := 10 * increase

-- Daily profit for a given increase in selling price
def daily_profit (increase : ℕ) : ℕ := (increase + min_selling_price - cost_price) * (units_sold_at_min_price - decrease_in_units increase)

-- Maximum profit calculation
def maximizing_daily_profit (increase : ℕ) : ℕ := (increase + min_selling_price - cost_price) * (units_sold_at_min_price - decrease_in_units increase) 

-- Statement for Problem Part 1
theorem determine_selling_price_for_daily_profit : ∃ P, P = 52 ∧ daily_profit (P - min_selling_price) = 2640 := 
sorry

-- Statement for Problem Part 2
theorem determine_max_profit_and_selling_price : ∃ P, P = 57 ∧ maximizing_daily_profit (P - min_selling_price) = 2890 := 
sorry

end determine_selling_price_for_daily_profit_determine_max_profit_and_selling_price_l27_27026


namespace tenth_term_of_geometric_sequence_l27_27946

theorem tenth_term_of_geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) (tenth_term : ℚ) :
  a = 5 →
  r = 4 / 3 →
  n = 10 →
  tenth_term = a * r ^ (n - 1) →
  tenth_term = 1310720 / 19683 :=
by sorry

end tenth_term_of_geometric_sequence_l27_27946


namespace find_x_l27_27181

def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (4, x)

theorem find_x (x : ℝ) (h : ∃k : ℝ, b x = (k * a.1, k * a.2)) : x = 6 := 
by 
  sorry

end find_x_l27_27181


namespace train_length_l27_27635

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_sec : ℝ := 3
noncomputable def speed_m_s := speed_km_hr * 1000 / 3600
noncomputable def length_of_train := speed_m_s * time_sec

theorem train_length :
  length_of_train = 50.01 := by
  sorry

end train_length_l27_27635


namespace original_price_l27_27748

theorem original_price (P : ℝ) (h1 : 0.76 * P = 820) : P = 1079 :=
by
  sorry

end original_price_l27_27748


namespace find_u_l27_27232

-- Definitions for given points lying on a straight line
def point := (ℝ × ℝ)

-- Points
def p1 : point := (2, 8)
def p2 : point := (6, 20)
def p3 : point := (10, 32)

-- Function to check if point is on the line derived from p1, p2, p3
def is_on_line (x y : ℝ) : Prop :=
  ∃ m b : ℝ, y = m * x + b ∧
  p1.2 = m * p1.1 + b ∧ 
  p2.2 = m * p2.1 + b ∧
  p3.2 = m * p3.1 + b

-- Statement to prove
theorem find_u (u : ℝ) (hu : is_on_line 50 u) : u = 152 :=
sorry

end find_u_l27_27232


namespace exists_unique_continuous_extension_l27_27714

noncomputable def F (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) : ℝ → ℝ :=
  sorry

theorem exists_unique_continuous_extension (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) :
  ∃! F : ℝ → ℝ, Continuous F ∧ ∀ x : ℚ, F x = f x :=
sorry

end exists_unique_continuous_extension_l27_27714


namespace fractions_sum_to_decimal_l27_27526

theorem fractions_sum_to_decimal :
  (2 / 10) + (4 / 100) + (6 / 1000) = 0.246 :=
by 
  sorry

end fractions_sum_to_decimal_l27_27526


namespace minimum_number_of_circles_l27_27830

-- Define the problem conditions
def conditions_of_problem (circles : ℕ) (n : ℕ) (highlighted_lines : ℕ) (sides_of_regular_2011_gon : ℕ) : Prop :=
  circles ≥ n ∧ highlighted_lines = sides_of_regular_2011_gon

-- The main theorem we need to prove
theorem minimum_number_of_circles :
  ∀ (n circles highlighted_lines sides_of_regular_2011_gon : ℕ),
    sides_of_regular_2011_gon = 2011 ∧ (highlighted_lines = sides_of_regular_2011_gon * 2) ∧ conditions_of_problem circles n highlighted_lines sides_of_regular_2011_gon → n = 504 :=
by
  sorry

end minimum_number_of_circles_l27_27830


namespace roots_polynomial_l27_27867

theorem roots_polynomial (n r s : ℚ) (c d : ℚ)
  (h1 : c * c - n * c + 3 = 0)
  (h2 : d * d - n * d + 3 = 0)
  (h3 : (c + 1/d) * (d + 1/c) = s)
  (h4 : c * d = 3) :
  s = 16/3 :=
by
  sorry

end roots_polynomial_l27_27867


namespace skateboarder_speed_l27_27773

theorem skateboarder_speed :
  let distance := 293.33
  let time := 20
  let feet_per_mile := 5280
  let seconds_per_hour := 3600
  let speed_ft_per_sec := distance / time
  let speed_mph := speed_ft_per_sec * (feet_per_mile / seconds_per_hour)
  speed_mph = 21.5 :=
by
  sorry

end skateboarder_speed_l27_27773


namespace part_1_part_2_l27_27855

-- Conditions and definitions
noncomputable def triangle_ABC (a b c S : ℝ) (A B C : ℝ) :=
  a * Real.sin B = -b * Real.sin (A + Real.pi / 3) ∧
  S = Real.sqrt 3 / 4 * c^2

-- 1. Prove A = 5 * Real.pi / 6
theorem part_1 (a b c S A B C : ℝ) (h : triangle_ABC a b c S A B C) :
  A = 5 * Real.pi / 6 :=
  sorry

-- 2. Prove sin C = sqrt 7 / 14 given S = sqrt 3 / 4 * c^2
theorem part_2 (a b c S A B C : ℝ) (h : triangle_ABC a b c S A B C) :
  Real.sin C = Real.sqrt 7 / 14 :=
  sorry

end part_1_part_2_l27_27855


namespace evaluate_4_over_04_eq_400_l27_27406

noncomputable def evaluate_fraction : Float :=
  (0.4)^4 / (0.04)^3

theorem evaluate_4_over_04_eq_400 : evaluate_fraction = 400 :=
by
  sorry

end evaluate_4_over_04_eq_400_l27_27406


namespace three_digit_number_is_657_l27_27496

theorem three_digit_number_is_657 :
  ∃ (a b c : ℕ), (100 * a + 10 * b + c = 657) ∧ (a + b + c = 18) ∧ (a = b + 1) ∧ (c = b + 2) :=
by
  sorry

end three_digit_number_is_657_l27_27496


namespace circle_equation_coefficients_l27_27411

theorem circle_equation_coefficients (a : ℝ) (x y : ℝ) : 
  (a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0) → (a = -1) :=
by 
  sorry

end circle_equation_coefficients_l27_27411


namespace cos_alpha_solution_l27_27011

open Real

theorem cos_alpha_solution
  (α : ℝ)
  (h1 : π < α)
  (h2 : α < 3 * π / 2)
  (h3 : tan α = 2) :
  cos α = -sqrt (1 / (1 + 2^2)) :=
by
  sorry

end cos_alpha_solution_l27_27011


namespace equation_of_tangent_line_l27_27500

theorem equation_of_tangent_line (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 4 * x + a * y - 17 = 0) →
   (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 4 * x - 3 * y + 11 = 0) :=
sorry

end equation_of_tangent_line_l27_27500


namespace negation_of_forall_log_gt_one_l27_27218

noncomputable def negation_of_p : Prop :=
∃ x : ℝ, Real.log x ≤ 1

theorem negation_of_forall_log_gt_one :
  (¬ (∀ x : ℝ, Real.log x > 1)) ↔ negation_of_p :=
by
  sorry

end negation_of_forall_log_gt_one_l27_27218


namespace seokjin_fewer_books_l27_27717

theorem seokjin_fewer_books (init_books : ℕ) (jungkook_initial : ℕ) (seokjin_initial : ℕ) (jungkook_bought : ℕ) (seokjin_bought : ℕ) :
  jungkook_initial = init_books → seokjin_initial = init_books → jungkook_bought = 18 → seokjin_bought = 11 →
  jungkook_initial + jungkook_bought - (seokjin_initial + seokjin_bought) = 7 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  sorry

end seokjin_fewer_books_l27_27717


namespace company_initial_bureaus_l27_27566

theorem company_initial_bureaus (B : ℕ) (offices : ℕ) (extra_bureaus : ℕ) 
  (h1 : offices = 14) 
  (h2 : extra_bureaus = 10) 
  (h3 : (B + extra_bureaus) % offices = 0) : 
  B = 8 := 
by
  sorry

end company_initial_bureaus_l27_27566


namespace symmetric_about_pi_over_4_l27_27429

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + Real.cos x

theorem symmetric_about_pi_over_4 (a : ℝ) :
  (∀ x : ℝ, f a (x + π / 4) = f a (-(x + π / 4))) → a = 1 := by
  unfold f
  sorry

end symmetric_about_pi_over_4_l27_27429


namespace find_f_at_4_l27_27225

theorem find_f_at_4 (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 3 * f x - 2 * f (1 / x) = x) : 
  f 4 = 5 / 2 :=
sorry

end find_f_at_4_l27_27225


namespace square_of_1037_l27_27729

theorem square_of_1037 : (1037 : ℕ)^2 = 1074369 := 
by {
  -- Proof omitted
  sorry
}

end square_of_1037_l27_27729


namespace fraction_pairs_l27_27798

theorem fraction_pairs (n : ℕ) (h : n > 2009) : 
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 1 ≤ a ∧ a ≤ n ∧
  1 ≤ b ∧ b ≤ n ∧ 1 ≤ c ∧ c ≤ n ∧ 1 ≤ d ∧ d ≤ n ∧
  1/a + 1/b = 1/c + 1/d := 
sorry

end fraction_pairs_l27_27798


namespace probability_two_most_expensive_l27_27795

open Nat

noncomputable def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem probability_two_most_expensive :
  (combination 8 1) / (combination 10 3) = 1 / 15 :=
by
  sorry

end probability_two_most_expensive_l27_27795


namespace solve_nine_sections_bamboo_problem_l27_27636

-- Define the bamboo stick problem
noncomputable def nine_sections_bamboo_problem : Prop :=
∃ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a (n + 1) = a n + d) ∧ -- Arithmetic sequence
  (a 1 + a 2 + a 3 + a 4 = 3) ∧ -- Top 4 sections' total volume
  (a 7 + a 8 + a 9 = 4) ∧ -- Bottom 3 sections' total volume
  (a 5 = 67 / 66) -- Volume of the 5th section

theorem solve_nine_sections_bamboo_problem : nine_sections_bamboo_problem :=
sorry

end solve_nine_sections_bamboo_problem_l27_27636


namespace avg_adults_proof_l27_27570

variable (n_total : ℕ) (n_girls : ℕ) (n_boys : ℕ) (n_adults : ℕ)
variable (avg_total : ℕ) (avg_girls : ℕ) (avg_boys : ℕ)

def avg_age_adults (n_total n_girls n_boys n_adults avg_total avg_girls avg_boys : ℕ) : ℕ :=
  let sum_total := n_total * avg_total
  let sum_girls := n_girls * avg_girls
  let sum_boys := n_boys * avg_boys
  let sum_adults := sum_total - sum_girls - sum_boys
  sum_adults / n_adults

theorem avg_adults_proof :
  avg_age_adults 50 25 20 5 21 18 20 = 40 := 
by
  -- Proof will go here
  sorry

end avg_adults_proof_l27_27570


namespace car_speed_is_90_mph_l27_27385

-- Define the given conditions
def distance_yards : ℚ := 22
def time_seconds : ℚ := 0.5
def yards_per_mile : ℚ := 1760

-- Define the car's speed in miles per hour
noncomputable def car_speed_mph : ℚ := (distance_yards / yards_per_mile) * (3600 / time_seconds)

-- The theorem to be proven
theorem car_speed_is_90_mph : car_speed_mph = 90 := by
  sorry

end car_speed_is_90_mph_l27_27385


namespace cylindrical_tank_volume_l27_27589

theorem cylindrical_tank_volume (d h : ℝ) (d_eq_20 : d = 20) (h_eq_10 : h = 10) : 
  π * ((d / 2) ^ 2) * h = 1000 * π :=
by
  sorry

end cylindrical_tank_volume_l27_27589


namespace length_of_platform_l27_27827

theorem length_of_platform
  (length_of_train time_crossing_platform time_crossing_pole : ℝ) 
  (length_of_train_eq : length_of_train = 400)
  (time_crossing_platform_eq : time_crossing_platform = 45)
  (time_crossing_pole_eq : time_crossing_pole = 30) :
  ∃ (L : ℝ), (400 + L) / time_crossing_platform = length_of_train / time_crossing_pole :=
by {
  use 200,
  sorry
}

end length_of_platform_l27_27827


namespace trigonometric_relationship_l27_27816

theorem trigonometric_relationship (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h_tan : Real.tan α = (1 - Real.sin β) / Real.cos β) : 
  2 * α + β = π / 2 := 
sorry

end trigonometric_relationship_l27_27816


namespace no_odd_m_solution_l27_27824

theorem no_odd_m_solution : ∀ (m n : ℕ), 0 < m → 0 < n → (5 * n = m * n - 3 * m) → ¬ Odd m :=
by
  intros m n hm hn h_eq
  sorry

end no_odd_m_solution_l27_27824


namespace olivia_pieces_of_paper_l27_27262

theorem olivia_pieces_of_paper (initial_pieces : ℕ) (used_pieces : ℕ) (pieces_left : ℕ) 
  (h1 : initial_pieces = 81) (h2 : used_pieces = 56) : 
  pieces_left = 81 - 56 :=
by
  sorry

end olivia_pieces_of_paper_l27_27262


namespace first_meeting_time_of_boys_l27_27561

theorem first_meeting_time_of_boys 
  (L : ℝ) (v1_kmh : ℝ) (v2_kmh : ℝ) (v1_ms v2_ms : ℝ) (rel_speed : ℝ) (t : ℝ)
  (hv1_km_to_ms : v1_ms = v1_kmh * 1000 / 3600)
  (hv2_km_to_ms : v2_ms = v2_kmh * 1000 / 3600)
  (hrel_speed : rel_speed = v1_ms + v2_ms)
  (hl : L = 4800)
  (hv1 : v1_kmh = 60)
  (hv2 : v2_kmh = 100)
  (ht : t = L / rel_speed) :
  t = 108 := by
  -- we're providing a placeholder for the proof
  sorry

end first_meeting_time_of_boys_l27_27561


namespace candy_pack_cost_l27_27377

theorem candy_pack_cost (c : ℝ) (h1 : 20 + 78 = 98) (h2 : 2 * c = 98) : c = 49 :=
by {
  sorry
}

end candy_pack_cost_l27_27377


namespace car_cost_l27_27776

def initial_savings : ℕ := 14500
def charge_per_trip : ℚ := 1.5
def percentage_groceries_earnings : ℚ := 0.05
def number_of_trips : ℕ := 40
def total_value_of_groceries : ℕ := 800

theorem car_cost (initial_savings charge_per_trip percentage_groceries_earnings number_of_trips total_value_of_groceries : ℚ) :
  initial_savings + (charge_per_trip * number_of_trips) + (percentage_groceries_earnings * total_value_of_groceries) = 14600 := 
by
  sorry

end car_cost_l27_27776


namespace oranges_for_profit_l27_27912

theorem oranges_for_profit (cost_buy: ℚ) (number_buy: ℚ) (cost_sell: ℚ) (number_sell: ℚ)
  (desired_profit: ℚ) (h₁: cost_buy / number_buy = 3.75) (h₂: cost_sell / number_sell = 4.5)
  (h₃: desired_profit = 120) :
  ∃ (oranges_to_sell: ℚ), oranges_to_sell = 160 ∧ (desired_profit / ((cost_sell / number_sell) - (cost_buy / number_buy))) = oranges_to_sell :=
by
  sorry

end oranges_for_profit_l27_27912


namespace binary_div_mul_l27_27902

-- Define the binary numbers
def a : ℕ := 0b101110
def b : ℕ := 0b110100
def c : ℕ := 0b110

-- Statement to prove the given problem
theorem binary_div_mul : (a * b) / c = 0b101011100 := by
  -- Skipping the proof
  sorry

end binary_div_mul_l27_27902


namespace first_discount_l27_27316

theorem first_discount (P F : ℕ) (D₂ : ℝ) (D₁ : ℝ) 
  (hP : P = 150) 
  (hF : F = 105)
  (hD₂ : D₂ = 12.5)
  (hF_eq : F = P * (1 - D₁ / 100) * (1 - D₂ / 100)) : 
  D₁ = 20 :=
by
  sorry

end first_discount_l27_27316


namespace root_of_quadratic_property_l27_27431

theorem root_of_quadratic_property (m : ℝ) (h : m^2 - 2 * m - 1 = 0) :
  m^2 + (1 / m^2) = 6 :=
sorry

end root_of_quadratic_property_l27_27431


namespace chord_length_of_circle_l27_27370

theorem chord_length_of_circle (x y : ℝ) (h1 : (x - 0)^2 + (y - 2)^2 = 4) (h2 : y = x) : 
  length_of_chord_intercepted_by_line_eq_2sqrt2 :=
sorry

end chord_length_of_circle_l27_27370


namespace cost_of_fencing_per_meter_l27_27791

theorem cost_of_fencing_per_meter
  (length : ℕ) (breadth : ℕ) (total_cost : ℝ) (cost_per_meter : ℝ)
  (h1 : length = 64) 
  (h2 : length = breadth + 28)
  (h3 : total_cost = 5300)
  (h4 : cost_per_meter = total_cost / (2 * (length + breadth))) :
  cost_per_meter = 26.50 :=
by {
  sorry
}

end cost_of_fencing_per_meter_l27_27791


namespace fractions_order_l27_27293

theorem fractions_order : (23 / 18) < (21 / 16) ∧ (21 / 16) < (25 / 19) :=
by
  sorry

end fractions_order_l27_27293


namespace fair_bets_allocation_l27_27999

theorem fair_bets_allocation (p_a : ℚ) (p_b : ℚ) (coins : ℚ) 
  (h_prob : p_a = 3 / 4 ∧ p_b = 1 / 4) (h_coins : coins = 96) : 
  (coins * p_a = 72) ∧ (coins * p_b = 24) :=
by 
  sorry

end fair_bets_allocation_l27_27999


namespace xy_product_l27_27587

variable {x y : ℝ}

theorem xy_product (h1 : x ≠ y) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : x + 3/x = y + 3/y) : x * y = 3 :=
sorry

end xy_product_l27_27587


namespace motorcycle_wheels_l27_27796

/--
In a parking lot, there are cars and motorcycles. Each car has 5 wheels (including one spare) 
and each motorcycle has a certain number of wheels. There are 19 cars in the parking lot.
Altogether all vehicles have 117 wheels. There are 11 motorcycles at the parking lot.
--/
theorem motorcycle_wheels (num_cars num_motorcycles total_wheels wheels_per_car wheels_per_motorcycle : ℕ)
  (h1 : wheels_per_car = 5) 
  (h2 : num_cars = 19) 
  (h3 : total_wheels = 117) 
  (h4 : num_motorcycles = 11) 
  : wheels_per_motorcycle = 2 :=
by
  sorry

end motorcycle_wheels_l27_27796


namespace mia_stops_in_quarter_C_l27_27396

def track_circumference : ℕ := 100 -- The circumference of the track in feet.
def total_distance_run : ℕ := 10560 -- The total distance Mia runs in feet.

-- Define the function to determine the quarter of the circle Mia stops in.
def quarter_mia_stops : ℕ :=
  let quarters := track_circumference / 4 -- Each quarter's length.
  let complete_laps := total_distance_run / track_circumference
  let remaining_distance := total_distance_run % track_circumference
  if remaining_distance < quarters then 1 -- Quarter A
  else if remaining_distance < 2 * quarters then 2 -- Quarter B
  else if remaining_distance < 3 * quarters then 3 -- Quarter C
  else 4 -- Quarter D

theorem mia_stops_in_quarter_C : quarter_mia_stops = 3 := by
  sorry

end mia_stops_in_quarter_C_l27_27396


namespace system_of_equations_solution_l27_27671

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 : ℝ), 
    (x1 + 2 * x2 = 10) ∧
    (3 * x1 + 2 * x2 + x3 = 23) ∧
    (x2 + 2 * x3 = 13) ∧
    (x1 = 4) ∧
    (x2 = 3) ∧
    (x3 = 5) :=
sorry

end system_of_equations_solution_l27_27671


namespace paper_boat_travel_time_l27_27235

-- Defining the conditions as constants
def distance_embankment : ℝ := 50
def speed_downstream : ℝ := 10
def speed_upstream : ℝ := 12.5

-- Definitions for the speeds of the boat and current
noncomputable def v_boat : ℝ := (speed_upstream + speed_downstream) / 2
noncomputable def v_current : ℝ := (speed_downstream - speed_upstream) / 2

-- Statement to prove the time taken for the paper boat
theorem paper_boat_travel_time :
  (distance_embankment / v_current) = 40 := by
  sorry

end paper_boat_travel_time_l27_27235


namespace speed_in_still_water_l27_27350

/-- Conditions -/
def upstream_speed : ℝ := 30
def downstream_speed : ℝ := 40

/-- Theorem: The speed of the man in still water is 35 kmph. -/
theorem speed_in_still_water : 
  (upstream_speed + downstream_speed) / 2 = 35 := 
by 
  sorry

end speed_in_still_water_l27_27350


namespace excircle_side_formula_l27_27646

theorem excircle_side_formula 
  (a b c r_a r_b r_c : ℝ)
  (h1 : r_c = Real.sqrt (r_a * r_b)) :
  c = (a^2 + b^2) / (a + b) :=
sorry

end excircle_side_formula_l27_27646


namespace perpendicular_lines_slope_l27_27721

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, x + a * y = 1 - a ∧ (a - 2) * x + 3 * y + 2 = 0) → a = 1 / 2 := 
by 
  sorry

end perpendicular_lines_slope_l27_27721


namespace range_of_a_l27_27760

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then (x - a) ^ 2 + Real.exp 1 else x / Real.log x + a + 10

theorem range_of_a (a : ℝ) :
    (∀ x, f x a ≥ f 2 a) → (2 ≤ a ∧ a ≤ 6) :=
by
  sorry

end range_of_a_l27_27760


namespace at_least_one_not_less_than_two_l27_27199

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a + (1 / b) ≥ 2 ∨ b + (1 / a) ≥ 2 :=
by
  sorry

end at_least_one_not_less_than_two_l27_27199


namespace equality_am_bn_l27_27333

theorem equality_am_bn (m n : ℝ) (x : ℝ) (a b : ℝ) (hmn : m ≠ n) (hm : m ≠ 0) (hn : n ≠ 0) :
  ((x + m) ^ 2 - (x + n) ^ 2 = (m - n) ^ 2) → (x = am + bn) → (a = 0 ∧ b = -1) :=
by
  intro h1 h2
  sorry

end equality_am_bn_l27_27333


namespace find_smallest_n_l27_27519

/-- 
Define the doubling sum function D(a, n)
-/
def doubling_sum (a : ℕ) (n : ℕ) : ℕ := a * (2^n - 1)

/--
Main theorem statement that proves the smallest n for the given conditions
-/
theorem find_smallest_n :
  ∃ (n : ℕ), (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 6 → ∃ (ai : ℕ), doubling_sum ai i = n) ∧ n = 9765 := 
sorry

end find_smallest_n_l27_27519


namespace find_smaller_number_l27_27573

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : y = 8 :=
by sorry

end find_smaller_number_l27_27573


namespace gasoline_price_increase_l27_27861

theorem gasoline_price_increase
  (P Q : ℝ) -- Prices and quantities
  (x : ℝ) -- The percentage increase in price
  (h1 : (P * (1 + x / 100)) * (Q * 0.95) = P * Q * 1.14) -- Given condition
  : x = 20 := 
sorry

end gasoline_price_increase_l27_27861


namespace melissa_games_played_l27_27542

theorem melissa_games_played (total_points : ℕ) (points_per_game : ℕ) (num_games : ℕ) 
  (h1 : total_points = 81) 
  (h2 : points_per_game = 27) 
  (h3 : num_games = total_points / points_per_game) : 
  num_games = 3 :=
by
  -- Proof goes here
  sorry

end melissa_games_played_l27_27542


namespace age_ratio_l27_27271

theorem age_ratio (B A : ℕ) (h1 : B = 4) (h2 : A - B = 12) :
  A / B = 4 :=
by
  sorry

end age_ratio_l27_27271


namespace photographer_max_photos_l27_27960

-- The initial number of birds of each species
def total_birds : ℕ := 20
def starlings : ℕ := 8
def wagtails : ℕ := 7
def woodpeckers : ℕ := 5

-- Define a function to count the remaining birds of each species after n photos
def remaining_birds (n : ℕ) (species : ℕ) : ℕ := species - (if species ≤ n then species else n)

-- Define the main theorem we want to prove
theorem photographer_max_photos (n : ℕ) (h1 : remaining_birds n starlings ≥ 4) (h2 : remaining_birds n wagtails ≥ 3) : 
  n ≤ 7 :=
by
  sorry

end photographer_max_photos_l27_27960


namespace solve_inequality_l27_27498

theorem solve_inequality : 
  {x : ℝ | -3 * x^2 + 9 * x + 6 < 0} = {x : ℝ | -2 / 3 < x ∧ x < 3} :=
by {
  sorry
}

end solve_inequality_l27_27498


namespace positive_integral_solution_exists_l27_27303

theorem positive_integral_solution_exists :
  ∃ n : ℕ, n > 0 ∧
  ( (n * (n + 1) * (2 * n + 1)) * 100 = 27 * 6 * (n * (n + 1))^2 ) ∧ n = 5 :=
by {
  sorry
}

end positive_integral_solution_exists_l27_27303


namespace problem_l27_27907

theorem problem (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 7 := 
sorry

end problem_l27_27907


namespace polygon_sides_sum_l27_27094

theorem polygon_sides_sum (triangle_hexagon_sum : ℕ) (triangle_sides : ℕ) (hexagon_sides : ℕ) 
  (h1 : triangle_hexagon_sum = 1260) 
  (h2 : triangle_sides = 3) 
  (h3 : hexagon_sides = 6) 
  (convex : ∀ n, 3 <= n) : 
  triangle_sides + hexagon_sides + 4 = 13 :=
by 
  sorry

end polygon_sides_sum_l27_27094


namespace amount_leaked_during_repairs_l27_27277

theorem amount_leaked_during_repairs:
  let total_leaked := 6206
  let leaked_before_repairs := 2475
  total_leaked - leaked_before_repairs = 3731 :=
by
  sorry

end amount_leaked_during_repairs_l27_27277


namespace woods_width_l27_27584

theorem woods_width (Area Length Width : ℝ) (hArea : Area = 24) (hLength : Length = 3) : 
  Width = 8 := 
by
  sorry

end woods_width_l27_27584


namespace sequence_an_square_l27_27223

theorem sequence_an_square (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) > a n) 
  (h3 : ∀ n : ℕ, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) :
  ∀ n : ℕ, a n = n^2 :=
by
  sorry

end sequence_an_square_l27_27223


namespace age_difference_proof_l27_27052

def AlexAge : ℝ := 16.9996700066
def AlexFatherAge (A : ℝ) (F : ℝ) : Prop := F = 2 * A + 4.9996700066
def FatherAgeSixYearsAgo (A : ℝ) (F : ℝ) : Prop := A - 6 = 1 / 3 * (F - 6)

theorem age_difference_proof :
  ∃ (A F : ℝ), A = 16.9996700066 ∧
  (AlexFatherAge A F) ∧
  (FatherAgeSixYearsAgo A F) :=
by
  sorry

end age_difference_proof_l27_27052


namespace min_value_f1_min_value_f1_achieved_max_value_f2_max_value_f2_achieved_l27_27207

-- Problem (Ⅰ)
theorem min_value_f1 (x : ℝ) (h : x > 0) : (12 / x + 3 * x) ≥ 12 :=
sorry

theorem min_value_f1_achieved : (12 / 2 + 3 * 2) = 12 :=
by norm_num

-- Problem (Ⅱ)
theorem max_value_f2 (x : ℝ) (h : 0 < x ∧ x < 1 / 3) : x * (1 - 3 * x) ≤ 1 / 12 :=
sorry

theorem max_value_f2_achieved : (1 / 6) * (1 - 3 * (1 / 6)) = 1 / 12 :=
by norm_num

end min_value_f1_min_value_f1_achieved_max_value_f2_max_value_f2_achieved_l27_27207


namespace gcd_765432_654321_l27_27733

-- Define the two integers 765432 and 654321
def a : ℕ := 765432
def b : ℕ := 654321

-- State the main theorem to prove the gcd
theorem gcd_765432_654321 : Nat.gcd a b = 3 := 
by 
  sorry

end gcd_765432_654321_l27_27733


namespace convert_units_l27_27462

theorem convert_units :
  (0.56 * 10 = 5.6 ∧ 0.6 * 10 = 6) ∧
  (2.05 = 2 + 0.05 ∧ 0.05 * 100 = 5) :=
by 
  sorry

end convert_units_l27_27462


namespace sales_worth_l27_27169

variables (S : ℝ)
variables (old_scheme_remuneration new_scheme_remuneration : ℝ)

def old_scheme := 0.05 * S
def new_scheme := 1300 + 0.025 * (S - 4000)

theorem sales_worth :
  new_scheme S = old_scheme S + 600 →
  S = 24000 :=
by
  intro h
  sorry

end sales_worth_l27_27169


namespace pencils_per_associate_professor_l27_27132

theorem pencils_per_associate_professor
    (A B P : ℕ) -- the number of associate professors, assistant professors, and pencils per associate professor respectively
    (h1 : A + B = 6) -- there are a total of 6 people
    (h2 : A * P + B = 7) -- total number of pencils is 7
    (h3 : A + 2 * B = 11) -- total number of charts is 11
    : P = 2 :=
by
  -- Placeholder for the proof
  sorry

end pencils_per_associate_professor_l27_27132


namespace rectangle_ratio_l27_27415

open Real

theorem rectangle_ratio (A B C D E : Point) (rat : ℚ) : 
  let area_rect := 1
  let area_pentagon := (7 / 10 : ℚ)
  let area_triangle_AEC := 3 / 10
  let area_triangle_ECD := 1 / 5
  let x := 3 * EA
  let y := 2 * EA
  let diag_longer_side := sqrt (5 * EA ^ 2)
  let diag_shorter_side := EA * sqrt 5
  let ratio := sqrt 5 
  ( area_pentagon == area_rect * (7 / 10) ) →
  ( area_triangle_AEC + area_pentagon = area_rect ) →
  ( area_triangle_AEC == area_rect - area_pentagon ) →
  ( ratio == diag_longer_side / diag_shorter_side ) :=
  sorry

end rectangle_ratio_l27_27415


namespace B_and_C_mutually_exclusive_but_not_complementary_l27_27823

-- Define the sample space of the cube
def faces : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define events based on conditions
def event_A (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_B (n : ℕ) : Prop := n = 1 ∨ n = 2
def event_C (n : ℕ) : Prop := n = 4 ∨ n = 5 ∨ n = 6

-- Define mutually exclusive events
def mutually_exclusive (A B : ℕ → Prop) : Prop := ∀ n, A n → ¬ B n

-- Define complementary events (for events over finite sample spaces like faces)
-- Events A and B are complementary if they partition the sample space faces
def complementary (A B : ℕ → Prop) : Prop := (∀ n, n ∈ faces → A n ∨ B n) ∧ (∀ n, A n → ¬ B n) ∧ (∀ n, B n → ¬ A n)

theorem B_and_C_mutually_exclusive_but_not_complementary :
  mutually_exclusive event_B event_C ∧ ¬ complementary event_B event_C := 
by
  sorry

end B_and_C_mutually_exclusive_but_not_complementary_l27_27823


namespace max_rides_day1_max_rides_day2_l27_27832

open List 

def daily_budget : ℤ := 10

def ride_prices_day1 : List (String × ℤ) := 
  [("Ferris wheel", 4), ("Roller coaster", 5), ("Bumper cars", 3), ("Carousel", 2), ("Log flume", 6)]

def ride_prices_day2 : List (String × ℤ) := 
  [("Ferris wheel", 4), ("Roller coaster", 7), ("Bumper cars", 3), ("Carousel", 2), ("Log flume", 6), ("Haunted house", 4)]

def max_rides (budget : ℤ) (prices : List (String × ℤ)) : ℤ :=
  sorry -- We'll assume this calculates the max number of rides correctly based on the given budget and prices.

theorem max_rides_day1 : max_rides daily_budget ride_prices_day1 = 3 := by
  sorry 

theorem max_rides_day2 : max_rides daily_budget ride_prices_day2 = 3 := by
  sorry 

end max_rides_day1_max_rides_day2_l27_27832


namespace math_problem_l27_27021

variables (x y z : ℝ)

theorem math_problem
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( (x^2 / (x + y) >= (3 * x - y) / 4) ) ∧ 
  ( (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) >= (x * y + y * z + z * x) / 2 ) :=
by sorry

end math_problem_l27_27021


namespace exists_word_D_l27_27784

variable {α : Type} [Inhabited α] [DecidableEq α]

def repeats (D : List α) (w : List α) : Prop :=
  ∃ k : ℕ, w = List.join (List.replicate k D)

theorem exists_word_D (A B C : List α)
  (h : (A ++ A ++ B ++ B) = (C ++ C)) :
  ∃ D : List α, repeats D A ∧ repeats D B ∧ repeats D C :=
sorry

end exists_word_D_l27_27784


namespace arithmetic_geometric_progression_l27_27716

-- Define the arithmetic progression terms
def u (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the property that the squares of the 12th, 13th, and 15th terms form a geometric progression
def geometric_progression (a d : ℝ) : Prop :=
  let u12 := u a d 12
  let u13 := u a d 13
  let u15 := u a d 15
  (u13^2 / u12^2 = u15^2 / u13^2)

-- The main statement
theorem arithmetic_geometric_progression (a d : ℝ) (h : geometric_progression a d) :
  d = 0 ∨ 4 * ((a + 11 * d)^2) = (a + 12 *d)^2 * (a + 14 * d)^2 / (a + 12 * d)^2 ∨ (a + 11 * d) * ((a + 11 * d) - 2 *d) = 0 :=
sorry

end arithmetic_geometric_progression_l27_27716


namespace sequence_inequality_l27_27969

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h₁ : a 1 = 0)
  (h₇ : a 7 = 0) :
  ∃ k : ℕ, k ≤ 5 ∧ a k + a (k + 2) ≤ a (k + 1) * Real.sqrt 3 := 
sorry

end sequence_inequality_l27_27969


namespace largest_square_area_l27_27581

theorem largest_square_area (total_string_length : ℕ) (h : total_string_length = 32) : ∃ (area : ℕ), area = 64 := 
  by
    sorry

end largest_square_area_l27_27581


namespace integer_count_n_l27_27617

theorem integer_count_n (n : ℤ) (H1 : n % 3 = 0) (H2 : 3 * n ≥ 1) (H3 : 3 * n ≤ 1000) : 
  ∃ k : ℕ, k = 111 := by
  sorry

end integer_count_n_l27_27617


namespace marie_finishes_fourth_task_at_11_40_am_l27_27989

-- Define the given conditions
def start_time : ℕ := 7 * 60 -- start time in minutes from midnight (7:00 AM)
def second_task_end_time : ℕ := 9 * 60 + 20 -- end time of second task in minutes from midnight (9:20 AM)
def num_tasks : ℕ := 4 -- four tasks
def task_duration : ℕ := (second_task_end_time - start_time) / 2 -- duration of one task

-- Define the goal to prove: the end time of the fourth task
def fourth_task_finish_time : ℕ := second_task_end_time + 2 * task_duration

theorem marie_finishes_fourth_task_at_11_40_am : fourth_task_finish_time = 11 * 60 + 40 := by
  sorry

end marie_finishes_fourth_task_at_11_40_am_l27_27989


namespace percentage_of_candidates_selected_in_State_A_is_6_l27_27841

-- Definitions based on conditions
def candidates_appeared : ℕ := 8400
def candidates_selected_B : ℕ := (7 * candidates_appeared) / 100 -- 7% of 8400
def extra_candidates_selected : ℕ := 84
def candidates_selected_A : ℕ := candidates_selected_B - extra_candidates_selected

-- Definition based on the goal proof
def percentage_selected_A : ℕ := (candidates_selected_A * 100) / candidates_appeared

-- The theorem we need to prove
theorem percentage_of_candidates_selected_in_State_A_is_6 :
  percentage_selected_A = 6 :=
by
  sorry

end percentage_of_candidates_selected_in_State_A_is_6_l27_27841


namespace good_subset_divisible_by_5_l27_27771

noncomputable def num_good_subsets : ℕ :=
  (Nat.factorial 1000) / ((Nat.factorial 201) * (Nat.factorial (1000 - 201)))

theorem good_subset_divisible_by_5 : num_good_subsets / 5 = (1 / 5) * num_good_subsets := 
sorry

end good_subset_divisible_by_5_l27_27771


namespace M_k_max_l27_27801

noncomputable def J_k (k : ℕ) : ℕ := 5^(k+3) * 2^(k+3) + 648

def M (k : ℕ) : ℕ := 
  if k < 3 then k + 3
  else 3

theorem M_k_max (k : ℕ) : M k = 3 :=
by sorry

end M_k_max_l27_27801


namespace find_four_numbers_l27_27662

theorem find_four_numbers (a b c d : ℕ) (h1 : b^2 = a * c) (h2 : a * b * c = 216) (h3 : 2 * c = b + d) (h4 : b + c + d = 12) :
  a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2 :=
sorry

end find_four_numbers_l27_27662


namespace positive_difference_volumes_l27_27084

open Real

noncomputable def charlies_height := 12
noncomputable def charlies_circumference := 10
noncomputable def danas_height := 8
noncomputable def danas_circumference := 10

theorem positive_difference_volumes (hC : ℝ := charlies_height) (CC : ℝ := charlies_circumference)
                                   (hD : ℝ := danas_height) (CD : ℝ := danas_circumference) :
    (π * (π * ((CD / (2 * π)) ^ 2) * hD - π * ((CC / (2 * π)) ^ 2) * hC)) = 100 :=
by
  have rC := CC / (2 * π)
  have VC := π * (rC ^ 2) * hC
  have rD := CD / (2 * π)
  have VD := π * (rD ^ 2) * hD
  sorry

end positive_difference_volumes_l27_27084


namespace f_equals_one_l27_27252

-- Define the functions f, g, h with the given properties

def f : ℕ → ℕ := sorry
def g : ℕ → ℕ := sorry
def h : ℕ → ℕ := sorry

-- Condition 1: h is injective
axiom h_injective : ∀ {a b : ℕ}, h a = h b → a = b

-- Condition 2: g is surjective
axiom g_surjective : ∀ n : ℕ, ∃ m : ℕ, g m = n

-- Condition 3: Definition of f in terms of g and h
axiom f_def : ∀ n : ℕ, f n = g n - h n + 1

-- Prove that f(n) = 1 for all n ∈ ℕ
theorem f_equals_one : ∀ n : ℕ, f n = 1 := by
  sorry

end f_equals_one_l27_27252


namespace product_value_l27_27150

noncomputable def product_of_sequence : ℝ :=
  (1/3) * 9 * (1/27) * 81 * (1/243) * 729 * (1/2187) * 6561

theorem product_value : product_of_sequence = 729 := by
  sorry

end product_value_l27_27150


namespace ellipse_standard_form_l27_27977

theorem ellipse_standard_form (α : ℝ) 
  (x y : ℝ) 
  (hx : x = 5 * Real.cos α) 
  (hy : y = 3 * Real.sin α) : 
  (x^2 / 25) + (y^2 / 9) = 1 := 
by 
  sorry

end ellipse_standard_form_l27_27977


namespace sample_processing_l27_27035

-- Define sample data
def standard: ℕ := 220
def samples: List ℕ := [230, 226, 218, 223, 214, 225, 205, 212]

-- Calculate deviations
def deviations (samples: List ℕ) (standard: ℕ) : List ℤ :=
  samples.map (λ x => x - standard)

-- Total dosage of samples
def total_dosage (samples: List ℕ): ℕ :=
  samples.sum

-- Total cost to process to standard dosage
def total_cost (deviations: List ℤ) (cost_per_ml_adjustment: ℤ) : ℤ :=
  cost_per_ml_adjustment * (deviations.map Int.natAbs).sum

-- Theorem statement
theorem sample_processing :
  let deviation_vals := deviations samples standard;
  let total_dosage_val := total_dosage samples;
  let total_cost_val := total_cost deviation_vals 10;
  deviation_vals = [10, 6, -2, 3, -6, 5, -15, -8] ∧
  total_dosage_val = 1753 ∧
  total_cost_val = 550 :=
by
  sorry

end sample_processing_l27_27035


namespace find_x_l27_27936

theorem find_x : ∃ x, (2015 + x)^2 = x^2 ∧ x = -2015 / 2 := 
by
  sorry

end find_x_l27_27936


namespace four_digit_div_by_99_then_sum_div_by_18_l27_27740

/-- 
If a whole number with at most four digits is divisible by 99, then 
the sum of its digits is divisible by 18. 
-/
theorem four_digit_div_by_99_then_sum_div_by_18 (n : ℕ) (h1 : n < 10000) (h2 : 99 ∣ n) : 
  18 ∣ (n.digits 10).sum := 
sorry

end four_digit_div_by_99_then_sum_div_by_18_l27_27740


namespace possible_orange_cells_l27_27336

theorem possible_orange_cells :
  ∃ (n : ℕ), n = 2021 * 2020 ∨ n = 2022 * 2020 := 
sorry

end possible_orange_cells_l27_27336


namespace fractional_addition_l27_27809

theorem fractional_addition : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by
  sorry

end fractional_addition_l27_27809


namespace cashback_percentage_l27_27730

theorem cashback_percentage
  (total_cost : ℝ) (rebate : ℝ) (final_cost : ℝ)
  (H1 : total_cost = 150) (H2 : rebate = 25) (H3 : final_cost = 110) :
  (total_cost - rebate - final_cost) / (total_cost - rebate) * 100 = 12 := by
  sorry

end cashback_percentage_l27_27730


namespace toy_poodle_height_l27_27399

theorem toy_poodle_height 
  (SP MP TP : ℕ)
  (h1 : SP = MP + 8)
  (h2 : MP = TP + 6)
  (h3 : SP = 28) 
  : TP = 14 := 
    by sorry

end toy_poodle_height_l27_27399


namespace inequality_proof_l27_27325

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b ≥ 1) :
  (a + 2 * b + 2 / (a + 1)) * (b + 2 * a + 2 / (b + 1)) ≥ 16 :=
by
  sorry

end inequality_proof_l27_27325


namespace find_c_value_l27_27621

def finds_c (c : ℝ) : Prop :=
  6 * (-(c / 6)) + 9 * (-(c / 9)) + c = 0 ∧ (-(c / 6) + -(c / 9) = 30)

theorem find_c_value : ∃ c : ℝ, finds_c c ∧ c = -108 :=
by
  use -108
  sorry

end find_c_value_l27_27621


namespace circle_area_from_circumference_l27_27637

theorem circle_area_from_circumference (C : ℝ) (hC : C = 48 * Real.pi) : 
  ∃ m : ℝ, (∀ r : ℝ, C = 2 * Real.pi * r → (Real.pi * r^2 = m * Real.pi)) ∧ m = 576 :=
by
  sorry

end circle_area_from_circumference_l27_27637


namespace hanks_pancakes_needed_l27_27442

/-- Hank's pancake calculation problem -/
theorem hanks_pancakes_needed 
    (pancakes_per_big_stack : ℕ := 5)
    (pancakes_per_short_stack : ℕ := 3)
    (big_stack_orders : ℕ := 6)
    (short_stack_orders : ℕ := 9) :
    (pancakes_per_short_stack * short_stack_orders) + (pancakes_per_big_stack * big_stack_orders) = 57 := by {
  sorry
}

end hanks_pancakes_needed_l27_27442


namespace point_within_region_l27_27800

theorem point_within_region (a : ℝ) (h : 2 * a + 2 < 4) : a < 1 := 
sorry

end point_within_region_l27_27800


namespace regular_hexagon_interior_angles_l27_27402

theorem regular_hexagon_interior_angles (n : ℕ) (h : n = 6) :
  (n - 2) * 180 = 720 :=
by
  subst h
  rfl

end regular_hexagon_interior_angles_l27_27402


namespace abs_sum_condition_l27_27371

theorem abs_sum_condition (a b : ℝ) (h1 : |a| = 7) (h2 : |b| = 3) (h3 : a * b > 0) : a + b = 10 ∨ a + b = -10 :=
by { sorry }

end abs_sum_condition_l27_27371


namespace sequence_general_term_l27_27980

theorem sequence_general_term (a : ℕ → ℚ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n+1) = (n * a n + 2 * (n+1)^2) / (n+2)) :
  ∀ n : ℕ, a n = (1 / 2 : ℚ) * n * (n + 1) := by
  sorry

end sequence_general_term_l27_27980


namespace greatest_possible_remainder_l27_27624

theorem greatest_possible_remainder (x : ℕ) : ∃ r : ℕ, r < 12 ∧ r ≠ 0 ∧ x % 12 = r ∧ r = 11 :=
by 
  sorry

end greatest_possible_remainder_l27_27624


namespace find_set_A_l27_27115

-- Define the set A based on the condition that its elements satisfy a quadratic equation.
def A (a : ℝ) : Set ℝ := {x | x^2 + 2 * x + a = 0}

-- Assume 1 is an element of set A
axiom one_in_A (a : ℝ) (h : 1 ∈ A a) : a = -3

-- The final theorem to prove: Given 1 ∈ A a, A a should be {-3, 1}
theorem find_set_A (a : ℝ) (h : 1 ∈ A a) : A a = {-3, 1} :=
by sorry

end find_set_A_l27_27115


namespace division_theorem_l27_27625

noncomputable def p (z : ℝ) : ℝ := 4 * z ^ 3 - 8 * z ^ 2 + 9 * z - 7
noncomputable def d (z : ℝ) : ℝ := 4 * z + 2
noncomputable def q (z : ℝ) : ℝ := z ^ 2 - 2.5 * z + 3.5
def r : ℝ := -14

theorem division_theorem (z : ℝ) : p z = d z * q z + r := 
by
  sorry

end division_theorem_l27_27625


namespace point_on_x_axis_coordinates_l27_27568

-- Define the conditions
def lies_on_x_axis (M : ℝ × ℝ) : Prop := M.snd = 0

-- State the problem
theorem point_on_x_axis_coordinates (a : ℝ) :
  lies_on_x_axis (a + 3, a + 1) → (a = -1) ∧ ((a + 3, 0) = (2, 0)) :=
by
  intro h
  rw [lies_on_x_axis] at h
  sorry

end point_on_x_axis_coordinates_l27_27568


namespace corrected_mean_l27_27019

theorem corrected_mean (mean_initial : ℝ) (num_obs : ℕ) (obs_incorrect : ℝ) (obs_correct : ℝ) :
  mean_initial = 36 → num_obs = 50 → obs_incorrect = 23 → obs_correct = 30 →
  (mean_initial * ↑num_obs + (obs_correct - obs_incorrect)) / ↑num_obs = 36.14 :=
by
  intros h1 h2 h3 h4
  sorry

end corrected_mean_l27_27019


namespace points_satisfy_diamond_eq_l27_27202

noncomputable def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem points_satisfy_diamond_eq (x y : ℝ) :
  (diamond x y = diamond y x) ↔ ((x = 0) ∨ (y = 0) ∨ (x = y) ∨ (x = -y)) := 
by
  sorry

end points_satisfy_diamond_eq_l27_27202


namespace sum_of_intercepts_l27_27922

theorem sum_of_intercepts (x y : ℝ) (h : y + 3 = -2 * (x + 5)) : 
  (- (13 / 2) : ℝ) + (- 13 : ℝ) = - (39 / 2) :=
by sorry

end sum_of_intercepts_l27_27922


namespace wine_age_problem_l27_27911

theorem wine_age_problem
  (carlo_rosi : ℕ)
  (franzia : ℕ)
  (twin_valley : ℕ)
  (h1 : franzia = 3 * carlo_rosi)
  (h2 : carlo_rosi = 4 * twin_valley)
  (h3 : carlo_rosi = 40) :
  franzia + carlo_rosi + twin_valley = 170 :=
by
  sorry

end wine_age_problem_l27_27911


namespace sixty_fifth_term_is_sixteen_l27_27657

def apply_rule (n : ℕ) : ℕ :=
  if n <= 12 then
    7 * n
  else if n % 2 = 0 then
    n - 7
  else
    n / 3

def sequence_term (a_0 : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate apply_rule n a_0

theorem sixty_fifth_term_is_sixteen : sequence_term 65 64 = 16 := by
  sorry

end sixty_fifth_term_is_sixteen_l27_27657


namespace find_m_l27_27890

noncomputable def vector_a : ℝ × ℝ := (1, -3)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 2)
noncomputable def vector_sum (m : ℝ) : ℝ × ℝ := (1 + m, -1)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_m (m : ℝ) : dot_product vector_a (vector_sum m) = 0 → m = -4 :=
by
  sorry

end find_m_l27_27890


namespace gas_cost_correct_l27_27510

def cost_to_fill_remaining_quarter (initial_fill : ℚ) (final_fill : ℚ) (added_gas : ℚ) (cost_per_litre : ℚ) : ℚ :=
  let tank_capacity := (added_gas * (1 / (final_fill - initial_fill)))
  let remaining_quarter_cost := (tank_capacity * (1 / 4)) * cost_per_litre
  remaining_quarter_cost

theorem gas_cost_correct :
  cost_to_fill_remaining_quarter (1/8) (3/4) 30 1.38 = 16.56 :=
by
  sorry

end gas_cost_correct_l27_27510


namespace find_f_2_l27_27390

noncomputable def f (a b x : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 :=
by
  sorry

end find_f_2_l27_27390


namespace Taso_riddles_correct_l27_27061

-- Definitions based on given conditions
def Josh_riddles : ℕ := 8
def Ivory_riddles : ℕ := Josh_riddles + 4
def Taso_riddles : ℕ := 2 * Ivory_riddles

-- The theorem to prove
theorem Taso_riddles_correct : Taso_riddles = 24 := by
  sorry

end Taso_riddles_correct_l27_27061


namespace no_solution_to_equation_l27_27001

theorem no_solution_to_equation :
  ¬ ∃ x : ℝ, 8 / (x ^ 2 - 4) + 1 = x / (x - 2) :=
by
  sorry

end no_solution_to_equation_l27_27001


namespace find_a_b_value_l27_27658

-- Define the variables
variables {a b : ℤ}

-- Define the conditions for the monomials to be like terms
def exponents_match_x (a : ℤ) : Prop := a + 2 = 1
def exponents_match_y (b : ℤ) : Prop := b + 1 = 3

-- Main statement
theorem find_a_b_value (ha : exponents_match_x a) (hb : exponents_match_y b) : a + b = 1 :=
by
  sorry

end find_a_b_value_l27_27658


namespace no_positive_integer_k_for_rational_solutions_l27_27612

theorem no_positive_integer_k_for_rational_solutions :
  ∀ k : ℕ, k > 0 → ¬ ∃ m : ℤ, 12 * (27 - k ^ 2) = m ^ 2 := by
  sorry

end no_positive_integer_k_for_rational_solutions_l27_27612


namespace eval_to_one_l27_27348

noncomputable def evalExpression (a b c : ℝ) : ℝ :=
  let numerator := (1 / a + 1 / b - 2 * c / (a * b)) * (a + b + 2 * c)
  let denominator := 1 / a^2 + 1 / b^2 + 2 / (a * b) - 4 * c^2 / (a^2 * b^2)
  numerator / denominator

theorem eval_to_one : 
  evalExpression 7.4 (5 / 37) c = 1 := 
by 
  sorry

end eval_to_one_l27_27348


namespace pascal_no_divisible_by_prime_iff_form_l27_27044

theorem pascal_no_divisible_by_prime_iff_form (p : ℕ) (n : ℕ) 
  (hp : Nat.Prime p) :
  (∀ k ≤ n, Nat.choose n k % p ≠ 0) ↔ ∃ s q : ℕ, s ≥ 0 ∧ 0 < q ∧ q < p ∧ n = p^s * q - 1 :=
by
  sorry

end pascal_no_divisible_by_prime_iff_form_l27_27044


namespace eight_S_three_l27_27470

def custom_operation_S (a b : ℤ) : ℤ := 4 * a + 6 * b + 3

theorem eight_S_three : custom_operation_S 8 3 = 53 := by
  sorry

end eight_S_three_l27_27470


namespace expression_equals_16_l27_27739

open Real

theorem expression_equals_16 (x : ℝ) :
  (x + 1) ^ 2 + 2 * (x + 1) * (3 - x) + (3 - x) ^ 2 = 16 :=
sorry

end expression_equals_16_l27_27739


namespace sum_of_three_numbers_l27_27168

theorem sum_of_three_numbers
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 252)
  (h2 : ab + bc + ca = 116) :
  a + b + c = 22 :=
by
  sorry

end sum_of_three_numbers_l27_27168


namespace participants_neither_coffee_nor_tea_l27_27198

-- Define the total number of participants
def total_participants : ℕ := 30

-- Define the number of participants who drank coffee
def coffee_drinkers : ℕ := 15

-- Define the number of participants who drank tea
def tea_drinkers : ℕ := 18

-- Define the number of participants who drank both coffee and tea
def both_drinkers : ℕ := 8

-- The proof statement for the number of participants who drank neither coffee nor tea
theorem participants_neither_coffee_nor_tea :
  total_participants - (coffee_drinkers + tea_drinkers - both_drinkers) = 5 := by
  sorry

end participants_neither_coffee_nor_tea_l27_27198


namespace max_area_triang_ABC_l27_27273

noncomputable def max_area_triang (a b c : ℝ) (M : ℝ) (BM : ℝ := 2) (AM : ℝ := c - b) : ℝ :=
if M = (b + c) / 2 then 2 * Real.sqrt 3 else 0

theorem max_area_triang_ABC (a b c : ℝ) (M : ℝ) (BM : ℝ := 2) (AM : ℝ := c - b) (M_midpoint : M = (b + c) / 2) :
  max_area_triang a b c M BM AM = 2 * Real.sqrt 3 :=
by
  sorry

end max_area_triang_ABC_l27_27273


namespace parabola_vertex_is_two_one_l27_27125

theorem parabola_vertex_is_two_one : 
  ∀ x y : ℝ, (y = (x - 2)^2 + 1) → (2, 1) = (2, 1) :=
by
  intros x y hyp
  sorry

end parabola_vertex_is_two_one_l27_27125


namespace min_positive_announcements_l27_27037

theorem min_positive_announcements (x y : ℕ) 
  (h1 : x * (x - 1) = 110) 
  (h2 : y * (y - 1) + (x - y) * (x - 1 - (y - 1)) = 50) : 
  y >= 5 := 
sorry

end min_positive_announcements_l27_27037


namespace no_solution_exists_only_solution_is_1963_l27_27875

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else n % 10 + sum_of_digits (n / 10)

-- Proof problem for part (a)
theorem no_solution_exists :
  ¬ ∃ x : ℕ, x + sum_of_digits x + sum_of_digits (sum_of_digits x) = 1993 :=
sorry

-- Proof problem for part (b)
theorem only_solution_is_1963 :
  ∃ x : ℕ, (x + sum_of_digits x + sum_of_digits (sum_of_digits x) + sum_of_digits (sum_of_digits (sum_of_digits x)) = 1993) ∧ (x = 1963) :=
sorry

end no_solution_exists_only_solution_is_1963_l27_27875


namespace smallest_integer_sum_to_2020_l27_27984

theorem smallest_integer_sum_to_2020 :
  ∃ B : ℤ, (∃ (n : ℤ), (B * (B + 1) / 2) + ((n * (n + 1)) / 2) = 2020) ∧ (∀ C : ℤ, (∃ (m : ℤ), (C * (C + 1) / 2) + ((m * (m + 1)) / 2) = 2020) → B ≤ C) ∧ B = -2019 :=
by
  sorry

end smallest_integer_sum_to_2020_l27_27984


namespace negation_proposition_iff_l27_27741

-- Define propositions and their components
def P (x : ℝ) : Prop := x > 1
def Q (x : ℝ) : Prop := x^2 > 1

-- State the proof problem
theorem negation_proposition_iff (x : ℝ) : ¬ (P x → Q x) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by 
  sorry

end negation_proposition_iff_l27_27741


namespace sara_height_correct_l27_27162

variable (Roy_height : ℕ)
variable (Joe_height : ℕ)
variable (Sara_height : ℕ)

def problem_conditions (Roy_height Joe_height Sara_height : ℕ) : Prop :=
  Roy_height = 36 ∧
  Joe_height = Roy_height + 3 ∧
  Sara_height = Joe_height + 6

theorem sara_height_correct (Roy_height Joe_height Sara_height : ℕ) :
  problem_conditions Roy_height Joe_height Sara_height → Sara_height = 45 := by
  sorry

end sara_height_correct_l27_27162


namespace shaded_region_area_is_48pi_l27_27734

open Real

noncomputable def small_circle_radius : ℝ := 4
noncomputable def small_circle_area : ℝ := π * small_circle_radius^2
noncomputable def large_circle_radius : ℝ := 2 * small_circle_radius
noncomputable def large_circle_area : ℝ := π * large_circle_radius^2
noncomputable def shaded_region_area : ℝ := large_circle_area - small_circle_area

theorem shaded_region_area_is_48pi :
  shaded_region_area = 48 * π := by
    sorry

end shaded_region_area_is_48pi_l27_27734


namespace second_mechanic_hours_l27_27248

theorem second_mechanic_hours (x y : ℕ) (h1 : 45 * x + 85 * y = 1100) (h2 : x + y = 20) : y = 5 :=
by
  sorry

end second_mechanic_hours_l27_27248


namespace percentage_of_y_l27_27715

theorem percentage_of_y (y : ℝ) : (0.3 * 0.6 * y = 0.18 * y) :=
by {
  sorry
}

end percentage_of_y_l27_27715


namespace value_range_a_for_two_positive_solutions_l27_27550

theorem value_range_a_for_two_positive_solutions (a : ℝ) :
  (∃ (x : ℝ), (|2 * x - 1| - a = 0) ∧ x > 0 ∧ (0 < a ∧ a < 1)) :=
by 
  sorry

end value_range_a_for_two_positive_solutions_l27_27550


namespace bobby_has_candy_left_l27_27445

def initial_candy := 36
def candy_eaten_first := 17
def candy_eaten_second := 15

theorem bobby_has_candy_left : 
  initial_candy - (candy_eaten_first + candy_eaten_second) = 4 := 
by
  sorry


end bobby_has_candy_left_l27_27445


namespace smallest_possible_n_l27_27546

theorem smallest_possible_n (n : ℕ) :
  ∃ n, 17 * n - 3 ≡ 0 [MOD 11] ∧ n = 6 :=
by
  sorry

end smallest_possible_n_l27_27546


namespace consecutive_numbers_sum_digits_l27_27735

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem consecutive_numbers_sum_digits :
  ∃ n : ℕ, sum_of_digits n = 52 ∧ sum_of_digits (n + 4) = 20 := 
sorry

end consecutive_numbers_sum_digits_l27_27735


namespace find_common_ratio_l27_27458

variable {α : Type*} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ n m, ∃ q, a (n + 1) = a n * q ∧ a (m + 1) = a m * q

theorem find_common_ratio 
  (a : ℕ → α) 
  (h : is_geometric_sequence a) 
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 1 / 4) : 
  ∃ q, q = 1 / 2 :=
by
  sorry

end find_common_ratio_l27_27458


namespace can_form_triangle_l27_27237

-- Define the function to check for the triangle inequality
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Problem statement: Prove that only the set (3, 4, 6) can form a triangle
theorem can_form_triangle :
  (¬ is_triangle 3 4 8) ∧
  (¬ is_triangle 5 6 11) ∧
  (¬ is_triangle 5 8 15) ∧
  (is_triangle 3 4 6) :=
by
  sorry

end can_form_triangle_l27_27237


namespace range_of_b_l27_27691

theorem range_of_b (b : ℝ) :
  (∀ x : ℤ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) ↔ 5 < b ∧ b < 7 := 
sorry

end range_of_b_l27_27691


namespace proof_probability_and_expectations_l27_27354

/-- Number of white balls drawn from two boxes --/
def X : ℕ := 1

/-- Number of red balls drawn from two boxes --/
def Y : ℕ := 1

/-- Given the conditions, the probability of drawing one white ball is 1/2, and
the expected value of white balls drawn is greater than the expected value of red balls drawn --/
theorem proof_probability_and_expectations :
  (∃ (P_X : ℚ), P_X = 1 / 2) ∧ (∃ (E_X E_Y : ℚ), E_X > E_Y) :=
by {
  sorry
}

end proof_probability_and_expectations_l27_27354


namespace shifted_quadratic_roots_l27_27920

theorem shifted_quadratic_roots {a h k : ℝ} (h_root_neg3 : a * (-3 + h) ^ 2 + k = 0)
                                 (h_root_2 : a * (2 + h) ^ 2 + k = 0) :
  (a * (-2 + h) ^ 2 + k = 0) ∧ (a * (3 + h) ^ 2 + k = 0) := by
  sorry

end shifted_quadratic_roots_l27_27920


namespace triangle_ABC_properties_l27_27176

open Real

theorem triangle_ABC_properties
  (a b c : ℝ) 
  (A B C : ℝ) 
  (A_eq : A = π / 3) 
  (b_eq : b = sqrt 2) 
  (cond1 : b^2 + sqrt 2 * a * c = a^2 + c^2) 
  (cond2 : a * cos B = b * sin A) 
  (cond3 : sin B + cos B = sqrt 2) : 
  B = π / 4 ∧ (1 / 2) * a * b * sin (π - A - B) = (3 + sqrt 3) / 4 := 
by 
  sorry

end triangle_ABC_properties_l27_27176


namespace total_cost_of_fencing_l27_27137

theorem total_cost_of_fencing (side_count : ℕ) (cost_per_side : ℕ) (h1 : side_count = 4) (h2 : cost_per_side = 79) : side_count * cost_per_side = 316 := by
  sorry

end total_cost_of_fencing_l27_27137


namespace find_A_and_area_l27_27487

open Real

variable (A B C a b c : ℝ)
variable (h1 : 2 * sin A * cos B = 2 * sin C - sin B)
variable (h2 : a = 4 * sqrt 3)
variable (h3 : b + c = 8)
variable (h4 : a^2 = b^2 + c^2 - 2*b*c* cos A)

theorem find_A_and_area :
  A = π / 3 ∧ (1/2 * b * c * sin A = 4 * sqrt 3 / 3) :=
by
  sorry

end find_A_and_area_l27_27487


namespace gcd_lcm_sum_eq_90_l27_27050

def gcd_three (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem gcd_lcm_sum_eq_90 : 
  let A := gcd_three 18 36 72
  let B := lcm_three 18 36 72
  A + B = 90 :=
by
  let A := gcd_three 18 36 72
  let B := lcm_three 18 36 72
  sorry

end gcd_lcm_sum_eq_90_l27_27050


namespace percent_sum_l27_27143

theorem percent_sum (A B C : ℝ)
  (hA : 0.45 * A = 270)
  (hB : 0.35 * B = 210)
  (hC : 0.25 * C = 150) :
  0.75 * A + 0.65 * B + 0.45 * C = 1110 := by
  sorry

end percent_sum_l27_27143


namespace kims_morning_routine_total_time_l27_27818

def time_spent_making_coffee := 5 -- in minutes
def time_spent_per_employee_status_update := 2 -- in minutes
def time_spent_per_employee_payroll_update := 3 -- in minutes
def number_of_employees := 9

theorem kims_morning_routine_total_time :
  time_spent_making_coffee +
  (time_spent_per_employee_status_update + time_spent_per_employee_payroll_update) * number_of_employees = 50 :=
by
  sorry

end kims_morning_routine_total_time_l27_27818


namespace hare_race_l27_27352

theorem hare_race :
  ∃ (total_jumps: ℕ) (final_jump_leg: String), total_jumps = 548 ∧ final_jump_leg = "right leg" :=
by
  sorry

end hare_race_l27_27352


namespace shirts_made_today_l27_27388

def shirts_per_minute : ℕ := 6
def minutes_yesterday : ℕ := 12
def total_shirts : ℕ := 156
def shirts_yesterday : ℕ := shirts_per_minute * minutes_yesterday
def shirts_today : ℕ := total_shirts - shirts_yesterday

theorem shirts_made_today :
  shirts_today = 84 :=
by
  sorry

end shirts_made_today_l27_27388


namespace multiply_repeating_decimals_l27_27834

noncomputable def repeating_decimal_03 : ℚ := 1 / 33
noncomputable def repeating_decimal_8 : ℚ := 8 / 9

theorem multiply_repeating_decimals : repeating_decimal_03 * repeating_decimal_8 = 8 / 297 := by 
  sorry

end multiply_repeating_decimals_l27_27834


namespace fixed_point_sum_l27_27604

theorem fixed_point_sum (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : (m, n) = (1, a * (1-1) + 2)) : m + n = 4 :=
by {
  sorry
}

end fixed_point_sum_l27_27604


namespace smallest_integer_greater_than_power_l27_27166

theorem smallest_integer_greater_than_power (sqrt3 sqrt2 : ℝ) (h1 : (sqrt3 + sqrt2)^6 = 485 + 198 * Real.sqrt 6)
(h2 : (sqrt3 - sqrt2)^6 = 485 - 198 * Real.sqrt 6)
(h3 : 0 < (sqrt3 - sqrt2)^6 ∧ (sqrt3 - sqrt2)^6 < 1) : 
  ⌈(sqrt3 + sqrt2)^6⌉ = 970 := 
sorry

end smallest_integer_greater_than_power_l27_27166


namespace investment_recovery_l27_27792

-- Define the conditions and the goal
theorem investment_recovery (c : ℕ) : 
  (15 * c - 5 * c) ≥ 8000 ↔ c ≥ 800 := 
sorry

end investment_recovery_l27_27792


namespace cart_max_speed_l27_27065

noncomputable def maximum_speed (a R : ℝ) : ℝ :=
  (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4)

theorem cart_max_speed (a R v : ℝ) (h : v = maximum_speed a R) : 
  v = (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4) :=
by
  -- Proof is omitted
  sorry

end cart_max_speed_l27_27065


namespace area_difference_of_circles_l27_27126

theorem area_difference_of_circles : 
  let r1 := 30
  let r2 := 15
  let pi := Real.pi
  900 * pi - 225 * pi = 675 * pi := by
  sorry

end area_difference_of_circles_l27_27126


namespace remaining_soup_feeds_adults_l27_27351

theorem remaining_soup_feeds_adults :
  (∀ (cans : ℕ), cans ≥ 8 ∧ cans / 6 ≥ 24) → (∃ (adults : ℕ), adults = 16) :=
by
  sorry

end remaining_soup_feeds_adults_l27_27351


namespace find_a_and_mono_l27_27249

open Real

noncomputable def f (x : ℝ) (a : ℝ) := (a * 2^x + a - 2) / (2^x + 1)

theorem find_a_and_mono :
  (∀ x : ℝ, f x a + f (-x) a = 0) →
  a = 1 ∧ f 3 1 = 7 / 9 ∧ ∀ x1 x2 : ℝ, x1 < x2 → f x1 1 < f x2 1 :=
by
  sorry

end find_a_and_mono_l27_27249


namespace three_digit_number_cubed_sum_l27_27381

theorem three_digit_number_cubed_sum {n : ℕ} (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 100 * a + 10 * b + c ∧ n = a^3 + b^3 + c^3) ↔
  n = 153 ∨ n = 370 ∨ n = 371 ∨ n = 407 :=
by
  sorry

end three_digit_number_cubed_sum_l27_27381


namespace album_ways_10_l27_27707

noncomputable def total_album_ways : ℕ := 
  let photo_albums := 2
  let stamp_albums := 3
  let total_albums := 4
  let friends := 4
  ((total_albums.choose photo_albums) * (total_albums - photo_albums).choose stamp_albums) / friends

theorem album_ways_10 :
  total_album_ways = 10 := 
by sorry

end album_ways_10_l27_27707


namespace coordinates_of_point_A_in_third_quadrant_l27_27758

def point_in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := abs y

def distance_to_y_axis (x : ℝ) : ℝ := abs x

theorem coordinates_of_point_A_in_third_quadrant 
  (x y : ℝ)
  (h1 : point_in_third_quadrant x y)
  (h2 : distance_to_x_axis y = 2)
  (h3 : distance_to_y_axis x = 3) :
  (x, y) = (-3, -2) :=
  sorry

end coordinates_of_point_A_in_third_quadrant_l27_27758


namespace find_sam_current_age_l27_27885

def Drew_current_age : ℕ := 12

def Drew_age_in_five_years : ℕ := Drew_current_age + 5

def Sam_age_in_five_years : ℕ := 3 * Drew_age_in_five_years

def Sam_current_age : ℕ := Sam_age_in_five_years - 5

theorem find_sam_current_age : Sam_current_age = 46 := by
  sorry

end find_sam_current_age_l27_27885


namespace yearly_return_500_correct_l27_27595

noncomputable def yearly_return_500_investment : ℝ :=
  let total_investment : ℝ := 500 + 1500
  let combined_yearly_return : ℝ := 0.10 * total_investment
  let yearly_return_1500 : ℝ := 0.11 * 1500
  let yearly_return_500 : ℝ := combined_yearly_return - yearly_return_1500
  (yearly_return_500 / 500) * 100

theorem yearly_return_500_correct : yearly_return_500_investment = 7 :=
by
  sorry

end yearly_return_500_correct_l27_27595


namespace Lesha_received_11_gifts_l27_27966

theorem Lesha_received_11_gifts (x : ℕ) 
    (h1 : x < 100) 
    (h2 : x % 2 = 0) 
    (h3 : x % 5 = 0) 
    (h4 : x % 7 = 0) :
    x - (x / 2 + x / 5 + x / 7) = 11 :=
by {
    sorry
}

end Lesha_received_11_gifts_l27_27966


namespace cos_neg_1500_eq_half_l27_27543

theorem cos_neg_1500_eq_half : Real.cos (-1500 * Real.pi / 180) = 1/2 := by
  sorry

end cos_neg_1500_eq_half_l27_27543


namespace ac_bc_nec_not_suff_l27_27793

theorem ac_bc_nec_not_suff (a b c : ℝ) : 
  (a = b → a * c = b * c) ∧ (¬(a * c = b * c → a = b)) := by
  sorry

end ac_bc_nec_not_suff_l27_27793


namespace gage_skating_time_l27_27395

theorem gage_skating_time :
  let gage_times_in_minutes1 := 1 * 60 + 15 -- 1 hour 15 minutes converted to minutes
  let gage_times_in_minutes2 := 2 * 60      -- 2 hours converted to minutes
  let total_skating_time_8_days := 5 * gage_times_in_minutes1 + 3 * gage_times_in_minutes2
  let required_total_time := 10 * 95       -- 10 days * 95 minutes per day
  required_total_time - total_skating_time_8_days = 215 :=
by
  sorry

end gage_skating_time_l27_27395


namespace area_below_line_l27_27901

noncomputable def circle_eqn (x y : ℝ) := 
  x^2 + 2 * x + (y^2 - 6 * y) + 50 = 0

noncomputable def line_eqn (x y : ℝ) := 
  y = x + 1

theorem area_below_line : 
  (∃ (x y : ℝ), circle_eqn x y ∧ y < x + 1) →
  ∃ (a : ℝ), a = 20 * π :=
by
  sorry

end area_below_line_l27_27901


namespace tim_out_of_pocket_cost_l27_27326

noncomputable def totalOutOfPocketCost : ℝ :=
  let mriCost := 1200
  let xrayCost := 500
  let examinationCost := 400 * (45 / 60)
  let feeForBeingSeen := 150
  let consultationFee := 75
  let physicalTherapyCost := 100 * 8
  let totalCostBeforeInsurance := mriCost + xrayCost + examinationCost + feeForBeingSeen + consultationFee + physicalTherapyCost
  let insuranceCoverage := 0.70 * totalCostBeforeInsurance
  let outOfPocketCost := totalCostBeforeInsurance - insuranceCoverage
  outOfPocketCost

theorem tim_out_of_pocket_cost : totalOutOfPocketCost = 907.50 :=
  by
    -- Proof will be provided here
    sorry

end tim_out_of_pocket_cost_l27_27326


namespace functional_equation_solution_l27_27586

theorem functional_equation_solution (f g : ℝ → ℝ)
  (H : ∀ x y : ℝ, f (x^2 - g y) = g x ^ 2 - y) :
  (∀ x : ℝ, f x = x) ∧ (∀ x : ℝ, g x = x) :=
by
  sorry

end functional_equation_solution_l27_27586


namespace perpendicular_vectors_l27_27043

theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) (c : ℝ × ℝ) 
  (h1 : a = (1, 2)) (h2 : b = (1, 1)) 
  (h3 : c = (1 + k, 2 + k))
  (h4 : b.1 * c.1 + b.2 * c.2 = 0) : 
  k = -3 / 2 :=
by
  sorry

end perpendicular_vectors_l27_27043


namespace xy_value_l27_27737

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 15) : x * y = 15 :=
by
  sorry

end xy_value_l27_27737


namespace area_of_right_triangle_l27_27900

-- Define a structure for the triangle with the given conditions
structure Triangle :=
(A B C : ℝ × ℝ)
(right_angle_at_C : (C.1 = 0 ∧ C.2 = 0))
(hypotenuse_length : (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 50 ^ 2)
(median_A : ∀ x: ℝ, A.2 = A.1 + 5)
(median_B : ∀ x: ℝ, B.2 = 2 * B.1 + 2)

-- Theorem statement
theorem area_of_right_triangle (t : Triangle) : 
  ∃ area : ℝ, area = 500 :=
sorry

end area_of_right_triangle_l27_27900


namespace symmetric_pentominoes_count_l27_27569

-- Assume we have exactly fifteen pentominoes
def num_pentominoes : ℕ := 15

-- Define the number of pentominoes with particular symmetrical properties
def num_reflectional_symmetry : ℕ := 8
def num_rotational_symmetry : ℕ := 3
def num_both_symmetries : ℕ := 2

-- The theorem we wish to prove
theorem symmetric_pentominoes_count 
  (n_p : ℕ) (n_r : ℕ) (n_b : ℕ) (n_tot : ℕ)
  (h1 : n_p = num_pentominoes)
  (h2 : n_r = num_reflectional_symmetry)
  (h3 : n_b = num_both_symmetries)
  (h4 : n_tot = n_r + num_rotational_symmetry - n_b) :
  n_tot = 9 := 
sorry

end symmetric_pentominoes_count_l27_27569


namespace percentage_cost_for_overhead_l27_27054

theorem percentage_cost_for_overhead
  (P M N : ℝ)
  (hP : P = 48)
  (hM : M = 50)
  (hN : N = 12) :
  (P + M - P - N) / P * 100 = 79.17 := by
  sorry

end percentage_cost_for_overhead_l27_27054


namespace same_root_implies_a_vals_l27_27681

-- Define the first function f(x) = x - a
def f (x a : ℝ) : ℝ := x - a

-- Define the second function g(x) = x^2 + ax - 2
def g (x a : ℝ) : ℝ := x^2 + a * x - 2

-- Theorem statement
theorem same_root_implies_a_vals (a : ℝ) (x : ℝ) (hf : f x a = 0) (hg : g x a = 0) : a = 1 ∨ a = -1 := 
sorry

end same_root_implies_a_vals_l27_27681


namespace tv_cost_l27_27502

theorem tv_cost (savings original_savings furniture_spent : ℝ) (hs : original_savings = 1000) (hf : furniture_spent = (3/4) * original_savings) (remaining_spent : savings = original_savings - furniture_spent) : savings = 250 := 
by
  sorry

end tv_cost_l27_27502


namespace neg_p_l27_27918

open Real

variable {f : ℝ → ℝ}

theorem neg_p :
  (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0) →
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
sorry

end neg_p_l27_27918


namespace sequence_an_solution_l27_27854

theorem sequence_an_solution {a : ℕ → ℝ} (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, 0 < n → (1 / a (n + 1) = 1 / a n + 1)) : ∀ n : ℕ, 0 < n → (a n = 1 / n) :=
by
  sorry

end sequence_an_solution_l27_27854


namespace tradesman_gain_on_outlay_l27_27186

-- Define the percentage defrauded and the percentage gain in both buying and selling
def defraud_percent := 20
def original_value := 100
def buying_price := original_value * (1 - (defraud_percent / 100))
def selling_price := original_value * (1 + (defraud_percent / 100))
def gain := selling_price - buying_price
def gain_percent := (gain / buying_price) * 100

theorem tradesman_gain_on_outlay :
  gain_percent = 50 := 
sorry

end tradesman_gain_on_outlay_l27_27186


namespace quadratic_floor_eq_solutions_count_l27_27456

theorem quadratic_floor_eq_solutions_count : 
  ∃ s : Finset ℝ, (∀ x : ℝ, x^2 - 4 * ⌊x⌋ + 3 = 0 → x ∈ s) ∧ s.card = 3 :=
by 
  sorry

end quadratic_floor_eq_solutions_count_l27_27456


namespace maximize_distance_l27_27144

def front_tire_lifespan : ℕ := 20000
def rear_tire_lifespan : ℕ := 30000
def max_distance : ℕ := 24000

theorem maximize_distance : max_distance = 24000 := sorry

end maximize_distance_l27_27144


namespace line_parabola_intersection_l27_27379

noncomputable def intersection_range (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + m * x - 1 = 2 * x - 2 * m → -1 ≤ x ∧ x ≤ 3

theorem line_parabola_intersection (m : ℝ) :
  intersection_range m ↔ -3 / 5 < m ∧ m < 5 :=
by
  sorry

end line_parabola_intersection_l27_27379


namespace maximize_profit_l27_27425

-- Define the relationships and constants
def P (x : ℝ) : ℝ := -750 * x + 15000
def material_cost_per_unit : ℝ := 4
def fixed_cost : ℝ := 7000

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - material_cost_per_unit) * P x - fixed_cost

-- The statement of the problem, proving the maximization condition
theorem maximize_profit :
  ∃ x : ℝ, x = 12 ∧ profit 12 = 41000 := by
  sorry

end maximize_profit_l27_27425


namespace plum_purchase_l27_27916

theorem plum_purchase
    (x : ℕ)
    (h1 : ∃ x, 5 * (6 * (4 * x) / 5) - 6 * ((5 * x) / 6) = -30) :
    2 * x = 60 := sorry

end plum_purchase_l27_27916


namespace cost_of_blue_cap_l27_27045

theorem cost_of_blue_cap (cost_tshirt cost_backpack cost_cap total_spent discount: ℝ) 
  (h1 : cost_tshirt = 30) 
  (h2 : cost_backpack = 10) 
  (h3 : discount = 2)
  (h4 : total_spent = 43) 
  (h5 : total_spent = cost_tshirt + cost_backpack + cost_cap - discount) : 
  cost_cap = 5 :=
by sorry

end cost_of_blue_cap_l27_27045


namespace fourth_even_integer_l27_27170

theorem fourth_even_integer (n : ℤ) (h : (n-2) + (n+2) = 92) : n + 4 = 50 := by
  -- This will skip the proof steps and assume the correct answer
  sorry

end fourth_even_integer_l27_27170


namespace emily_sixth_quiz_score_l27_27606

theorem emily_sixth_quiz_score (q1 q2 q3 q4 q5 target_mean : ℕ) (required_sum : ℕ) (current_sum : ℕ) (s6 : ℕ)
  (h1 : q1 = 94) (h2 : q2 = 97) (h3 : q3 = 88) (h4 : q4 = 91) (h5 : q5 = 102) (h_target_mean : target_mean = 95)
  (h_required_sum : required_sum = 6 * target_mean) (h_current_sum : current_sum = q1 + q2 + q3 + q4 + q5)
  (h6 : s6 = required_sum - current_sum) :
  s6 = 98 :=
by
  sorry

end emily_sixth_quiz_score_l27_27606


namespace polynomial_factor_l27_27742

theorem polynomial_factor (a b : ℝ) :
  (∃ (c d : ℝ), a = 4 * c ∧ b = -3 * c + 4 * d ∧ 40 = 2 * c - 3 * d + 18 ∧ -20 = 2 * d - 9 ∧ 9 = 9) →
  a = 11 ∧ b = -121 / 4 :=
by
  sorry

end polynomial_factor_l27_27742


namespace minimum_operations_to_transfer_beer_l27_27700

-- Definition of the initial conditions
structure InitialState where
  barrel_quarts : ℕ := 108
  seven_quart_vessel : ℕ := 0
  five_quart_vessel : ℕ := 0

-- Definition of the desired final state after minimum steps
structure FinalState where
  operations : ℕ := 17

-- Main theorem statement
theorem minimum_operations_to_transfer_beer (s : InitialState) : FinalState :=
  sorry

end minimum_operations_to_transfer_beer_l27_27700


namespace percentage_not_sophomores_l27_27033

variable (Total : ℕ) (Juniors Senior : ℕ) (Freshmen Sophomores : ℕ)

-- Conditions
axiom total_students : Total = 800
axiom percent_juniors : (22 / 100) * Total = Juniors
axiom number_seniors : Senior = 160
axiom freshmen_sophomores_relation : Freshmen = Sophomores + 64
axiom total_composition : Freshmen + Sophomores + Juniors + Senior = Total

-- Proof Objective
theorem percentage_not_sophomores :
  (Total - Sophomores) / Total * 100 = 75 :=
by
  -- proof omitted
  sorry

end percentage_not_sophomores_l27_27033


namespace extra_games_needed_l27_27774

def initial_games : ℕ := 500
def initial_success_rate : ℚ := 0.49
def target_success_rate : ℚ := 0.5

theorem extra_games_needed :
  ∀ (x : ℕ),
  (245 + x) / (initial_games + x) = target_success_rate → x = 10 := 
by
  sorry

end extra_games_needed_l27_27774


namespace andy_wrong_questions_l27_27835

variables (a b c d : ℕ)

theorem andy_wrong_questions 
  (h1 : a + b = c + d) 
  (h2 : a + d = b + c + 6) 
  (h3 : c = 7) : 
  a = 20 :=
sorry

end andy_wrong_questions_l27_27835


namespace transform_polynomial_to_y_l27_27435

theorem transform_polynomial_to_y (x y : ℝ) (h : y = x + 1/x) :
  (x^6 + x^5 - 5*x^4 + x^3 + x + 1 = 0) → 
  (∃ (y_expr : ℝ), (x * y_expr = 0 ∨ (x = 0 ∧ y_expr = y_expr))) :=
sorry

end transform_polynomial_to_y_l27_27435


namespace range_of_x_l27_27744

def valid_domain (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x ≠ 4)

theorem range_of_x : ∀ x : ℝ, valid_domain x ↔ (x ≤ 3) :=
by sorry

end range_of_x_l27_27744


namespace overall_effect_l27_27575
noncomputable def effect (x : ℚ) : ℚ :=
  ((x * (5 / 6)) * (1 / 10)) + (2 / 3)

theorem overall_effect (x : ℚ) : effect x = (x * (5 / 6) * (1 / 10)) + (2 / 3) :=
  by
  sorry

-- Prove for initial number 1
example : effect 1 = 3 / 4 :=
  by
  sorry

end overall_effect_l27_27575


namespace find_k_value_l27_27280

theorem find_k_value (k : ℝ) (x y : ℝ) (h1 : -3 * x + 2 * y = k) (h2 : 0.75 * x + y = 16) (h3 : x = -6) : k = 59 :=
by 
  sorry

end find_k_value_l27_27280


namespace arc_length_of_f_l27_27665

noncomputable def f (x : ℝ) : ℝ := 2 - Real.exp x

theorem arc_length_of_f :
  ∫ x in Real.log (Real.sqrt 3)..Real.log (Real.sqrt 8), Real.sqrt (1 + (Real.exp x)^2) = 1 + 1/2 * Real.log (3 / 2) :=
by
  sorry

end arc_length_of_f_l27_27665


namespace gas_volume_at_12_l27_27697

variable (VolumeTemperature : ℕ → ℕ) -- a function representing the volume of gas at a given temperature 

axiom condition1 : ∀ t : ℕ, VolumeTemperature (t + 4) = VolumeTemperature t + 5

axiom condition2 : VolumeTemperature 28 = 35

theorem gas_volume_at_12 :
  VolumeTemperature 12 = 15 := 
sorry

end gas_volume_at_12_l27_27697


namespace max_mondays_in_51_days_l27_27886

theorem max_mondays_in_51_days : ∀ (first_day : ℕ), first_day ≤ 6 → (∃ mondays : ℕ, mondays = 8) :=
  by
  sorry

end max_mondays_in_51_days_l27_27886


namespace symmetric_line_eq_l27_27230

/-- 
Given two circles O: x^2 + y^2 = 4 and C: x^2 + y^2 + 4x - 4y + 4 = 0, 
prove the equation of the line l such that the two circles are symmetric 
with respect to line l is x - y + 2 = 0.
-/
theorem symmetric_line_eq {x y : ℝ} :
  (∀ x y : ℝ, (x^2 + y^2 = 4) → (x^2 + y^2 + 4*x - 4*y + 4 = 0)) → (∀ x y : ℝ, (x - y + 2 = 0)) :=
  sorry

end symmetric_line_eq_l27_27230


namespace curve_has_axis_of_symmetry_l27_27947

theorem curve_has_axis_of_symmetry (x y : ℝ) :
  (x^2 - x * y + y^2 + x - y - 1 = 0) ↔ (x+y = 0) :=
sorry

end curve_has_axis_of_symmetry_l27_27947


namespace sin_neg_390_eq_neg_half_l27_27472

theorem sin_neg_390_eq_neg_half : Real.sin (-390 * Real.pi / 180) = -1 / 2 :=
  sorry

end sin_neg_390_eq_neg_half_l27_27472


namespace sufficient_condition_l27_27939

theorem sufficient_condition (x y : ℤ) (h : x + y ≠ 2) : x ≠ 1 ∧ y ≠ 1 := 
sorry

end sufficient_condition_l27_27939


namespace quadratic_root_properties_l27_27837

theorem quadratic_root_properties (b : ℝ) (t : ℝ) :
  (∀ x : ℝ, x^2 + b*x - 2 = 0 → (x = 2 ∨ x = t)) →
  b = -1 ∧ t = -1 :=
by
  sorry

end quadratic_root_properties_l27_27837


namespace angle_A_range_find_b_l27_27315

-- Definitions based on problem conditions
variable {a b c S : ℝ}
variable {A B C : ℝ}
variable {x : ℝ}

-- First statement: range of values for A
theorem angle_A_range (h1 : c * b * Real.cos A ≤ 2 * Real.sqrt 3 * S)
                      (h2 : S = 1/2 * b * c * Real.sin A)
                      (h3 : 0 < A ∧ A < π) : π / 6 ≤ A ∧ A < π := 
sorry

-- Second statement: value of b
theorem find_b (h1 : Real.tan A = x ∧ Real.tan B = 2 * x ∧ Real.tan C = 3 * x)
               (h2 : x = 1)
               (h3 : c = 1) : b = 2 * Real.sqrt 2 / 3 :=
sorry

end angle_A_range_find_b_l27_27315


namespace value_of_x_minus_y_l27_27281

theorem value_of_x_minus_y (x y : ℝ) 
    (h1 : 3015 * x + 3020 * y = 3025) 
    (h2 : 3018 * x + 3024 * y = 3030) :
    x - y = 11.1167 :=
sorry

end value_of_x_minus_y_l27_27281


namespace abc_divisibility_l27_27702

theorem abc_divisibility (a b c : ℕ) (h1 : c ∣ a^b) (h2 : a ∣ b^c) (h3 : b ∣ c^a) : abc ∣ (a + b + c)^(a + b + c) := 
sorry

end abc_divisibility_l27_27702


namespace polynomial_divisibility_l27_27643

theorem polynomial_divisibility (a b : ℤ) :
  (∀ x : ℤ, x^2 - 1 ∣ x^5 - 3 * x^4 + a * x^3 + b * x^2 - 5 * x - 5) ↔ (a = 4 ∧ b = 8) :=
sorry

end polynomial_divisibility_l27_27643


namespace smallest_prime_12_less_than_square_l27_27990

def is_perfect_square (n : ℕ) := ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_12_less_than_square : ∃ n : ℕ, is_prime n ∧ (∃ k : ℕ, k * k = n + 12) ∧ n = 13 :=
by
  sorry

end smallest_prime_12_less_than_square_l27_27990


namespace num_trucks_l27_27613

variables (T : ℕ) (num_cars : ℕ := 13) (total_wheels : ℕ := 100) (wheels_per_vehicle : ℕ := 4)

theorem num_trucks :
  (num_cars * wheels_per_vehicle + T * wheels_per_vehicle = total_wheels) -> T = 12 :=
by
  intro h
  -- skipping the proof implementation
  sorry

end num_trucks_l27_27613


namespace interior_angle_ratio_l27_27552

variables (α β γ : ℝ)

theorem interior_angle_ratio
  (h1 : 2 * α + 3 * β = 4 * γ)
  (h2 : α = 4 * β - γ) :
  ∃ k : ℝ, k ≠ 0 ∧ 
  (α = 2 * k ∧ β = 9 * k ∧ γ = 4 * k) :=
sorry

end interior_angle_ratio_l27_27552


namespace tan_sum_formula_l27_27420

theorem tan_sum_formula (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -1 := by
sorry

end tan_sum_formula_l27_27420


namespace part1_part2_l27_27727

theorem part1 (A B C a b c : ℝ) (h1 : 3 * a * Real.cos A = Real.sqrt 6 * (c * Real.cos B + b * Real.cos C)) :
    Real.tan (2 * A) = 2 * Real.sqrt 2 := sorry

theorem part2 (A B C a b c S : ℝ) 
  (h_sin_B : Real.sin (Real.pi / 2 + B) = 2 * Real.sqrt 2 / 3)
  (hc : c = 2 * Real.sqrt 2) :
    S = 2 * Real.sqrt 2 / 3 := sorry

end part1_part2_l27_27727


namespace total_pencils_correct_l27_27967

def pencils_in_drawer : ℕ := 43
def pencils_on_desk_originally : ℕ := 19
def pencils_added_by_dan : ℕ := 16
def total_pencils : ℕ := pencils_in_drawer + pencils_on_desk_originally + pencils_added_by_dan

theorem total_pencils_correct : total_pencils = 78 := by
  sorry

end total_pencils_correct_l27_27967


namespace f_neg_def_l27_27843

variable (f : ℝ → ℝ)
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_def_pos : ∀ x : ℝ, 0 < x → f x = x * (1 + x)

theorem f_neg_def (x : ℝ) (hx : x < 0) : f x = x * (1 - x) := by
  sorry

end f_neg_def_l27_27843


namespace inequality_proof_l27_27692

variable {a b : ℝ}

theorem inequality_proof (h : a > b) : 2 - a < 2 - b :=
by
  sorry

end inequality_proof_l27_27692


namespace math_evening_problem_l27_27146

theorem math_evening_problem
  (S : ℕ)
  (r : ℕ)
  (fifth_graders_per_row : ℕ := 3)
  (sixth_graders_per_row : ℕ := r - fifth_graders_per_row)
  (total_number_of_students : ℕ := r * r) :
  70 < total_number_of_students ∧ total_number_of_students < 90 → 
  r = 9 ∧ 
  6 * r = 54 ∧
  3 * r = 27 :=
sorry

end math_evening_problem_l27_27146


namespace trajectory_of_midpoint_l27_27506

open Real

theorem trajectory_of_midpoint (A : ℝ × ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A = (-2, 0))
    (hP_on_curve : P.1 = 2 * P.2 ^ 2)
    (hM_midpoint : M = ((A.1 + P.1) / 2, (A.2 + P.2) / 2)) :
    M.1 = 4 * M.2 ^ 2 - 1 :=
sorry

end trajectory_of_midpoint_l27_27506


namespace intersecting_lines_l27_27172

variable (a b m : ℝ)

-- Conditions
def condition1 : Prop := 8 = -m + a
def condition2 : Prop := 8 = m + b

-- Statement to prove
theorem intersecting_lines : condition1 a m  → condition2 b m  → a + b = 16 :=
by
  intros h1 h2
  sorry

end intersecting_lines_l27_27172


namespace sum_mod_six_l27_27432

theorem sum_mod_six (n : ℤ) : ((10 - 2 * n) + (4 * n + 2)) % 6 = 0 :=
by {
  sorry
}

end sum_mod_six_l27_27432


namespace problem_l27_27962

theorem problem (a : ℤ) (ha : 0 ≤ a ∧ a < 13) (hdiv : (51 ^ 2016 + a) % 13 = 0) : a = 12 :=
sorry

end problem_l27_27962


namespace yogurt_combinations_l27_27632

-- Definitions: Given conditions from the problem
def num_flavors : ℕ := 5
def num_toppings : ℕ := 7

-- Function to calculate binomial coefficient
def nCr (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement: The problem translated into Lean
theorem yogurt_combinations : 
  (num_flavors * nCr num_toppings 2) = 105 := by
  sorry

end yogurt_combinations_l27_27632


namespace larger_number_l27_27310

theorem larger_number (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 4) : x = 17 :=
by
sorry

end larger_number_l27_27310


namespace total_tickets_l27_27391

theorem total_tickets (tickets_first_day tickets_second_day tickets_third_day : ℕ) 
  (h1 : tickets_first_day = 5 * 4) 
  (h2 : tickets_second_day = 32)
  (h3 : tickets_third_day = 28) :
  tickets_first_day + tickets_second_day + tickets_third_day = 80 := by
  sorry

end total_tickets_l27_27391


namespace calculate_F_2_f_3_l27_27140

def f (a : ℕ) : ℕ := a ^ 2 - 3 * a + 2

def F (a b : ℕ) : ℕ := b ^ 2 + a + 1

theorem calculate_F_2_f_3 : F 2 (f 3) = 7 :=
by
  show F 2 (f 3) = 7
  sorry

end calculate_F_2_f_3_l27_27140


namespace domain_of_myFunction_l27_27344

-- Define the function
def myFunction (x : ℝ) : ℝ := (x + 2) ^ (1 / 2) - (x + 1) ^ 0

-- State the domain constraints as a theorem
theorem domain_of_myFunction (x : ℝ) : 
  (x ≥ -2 ∧ x ≠ -1) →
  ∃ y : ℝ, y = myFunction x := 
sorry

end domain_of_myFunction_l27_27344


namespace find_triple_l27_27006

theorem find_triple (A B C : ℕ) (h1 : A^2 + B - C = 100) (h2 : A + B^2 - C = 124) : 
  (A, B, C) = (12, 13, 57) := 
  sorry

end find_triple_l27_27006


namespace five_point_questions_l27_27559

-- Defining the conditions as Lean statements
def question_count (x y : ℕ) : Prop := x + y = 30
def total_points (x y : ℕ) : Prop := 5 * x + 10 * y = 200

-- The theorem statement that states x equals the number of 5-point questions
theorem five_point_questions (x y : ℕ) (h1 : question_count x y) (h2 : total_points x y) : x = 20 :=
sorry -- Proof is omitted

end five_point_questions_l27_27559


namespace largest_number_is_89_l27_27228

theorem largest_number_is_89 (a b c d : ℕ) 
  (h1 : a + b + c = 180) 
  (h2 : a + b + d = 197) 
  (h3 : a + c + d = 208) 
  (h4 : b + c + d = 222) : 
  max a (max b (max c d)) = 89 := 
by sorry

end largest_number_is_89_l27_27228


namespace remainder_poly_div_l27_27013

theorem remainder_poly_div 
    (x : ℤ) 
    (h1 : (x^2 + x + 1) ∣ (x^3 - 1)) 
    (h2 : x^5 - 1 = (x^3 - 1) * (x^2 + x + 1) - x * (x^2 + x + 1) + 1) : 
  ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 :=
by
  sorry

end remainder_poly_div_l27_27013


namespace f_neg_a_l27_27459

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end f_neg_a_l27_27459


namespace distinguishable_arrangements_l27_27619

-- Define the conditions: number of tiles of each color
def num_brown_tiles := 2
def num_purple_tile := 1
def num_green_tiles := 3
def num_yellow_tiles := 4

-- Total number of tiles
def total_tiles := num_brown_tiles + num_purple_tile + num_green_tiles + num_yellow_tiles

-- Factorials (using Lean's built-in factorial function)
def brown_factorial := Nat.factorial num_brown_tiles
def purple_factorial := Nat.factorial num_purple_tile
def green_factorial := Nat.factorial num_green_tiles
def yellow_factorial := Nat.factorial num_yellow_tiles
def total_factorial := Nat.factorial total_tiles

-- The result of the permutation calculation
def number_of_arrangements := total_factorial / (brown_factorial * purple_factorial * green_factorial * yellow_factorial)

-- The theorem stating the expected correct answer
theorem distinguishable_arrangements : number_of_arrangements = 12600 := 
by
    simp [number_of_arrangements, total_tiles, brown_factorial, purple_factorial, green_factorial, yellow_factorial, total_factorial]
    sorry

end distinguishable_arrangements_l27_27619


namespace find_sum_of_pqr_l27_27004

theorem find_sum_of_pqr (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : p ≠ r) 
(h4 : q = p * (4 - p)) (h5 : r = q * (4 - q)) (h6 : p = r * (4 - r)) : 
p + q + r = 6 :=
sorry

end find_sum_of_pqr_l27_27004


namespace proof_main_proof_l27_27392

noncomputable def main_proof : Prop :=
  2 * Real.logb 5 10 + Real.logb 5 0.25 = 2

theorem proof_main_proof : main_proof :=
  by
    sorry

end proof_main_proof_l27_27392


namespace trigonometric_inequality_l27_27345

open Real

theorem trigonometric_inequality 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < y) 
  (h3 : y < z) 
  (h4 : z < π / 2) : 
  π / 2 + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
  sorry

end trigonometric_inequality_l27_27345


namespace sum_of_squares_ne_sum_of_fourth_powers_l27_27488

theorem sum_of_squares_ne_sum_of_fourth_powers :
  ∀ (a b : ℤ), a^2 + (a + 1)^2 ≠ b^4 + (b + 1)^4 :=
by 
  sorry

end sum_of_squares_ne_sum_of_fourth_powers_l27_27488


namespace box_volume_correct_l27_27523

-- Define the dimensions of the original sheet
def length_original : ℝ := 48
def width_original : ℝ := 36

-- Define the side length of the squares cut from each corner
def side_length_cut : ℝ := 4

-- Define the new dimensions after cutting the squares
def new_length : ℝ := length_original - 2 * side_length_cut
def new_width : ℝ := width_original - 2 * side_length_cut

-- Define the height of the box
def height_box : ℝ := side_length_cut

-- Define the expected volume of the box
def volume_box_expected : ℝ := 4480

-- Prove that the calculated volume is equal to the expected volume
theorem box_volume_correct :
  new_length * new_width * height_box = volume_box_expected := by
  sorry

end box_volume_correct_l27_27523


namespace nth_equation_l27_27083

theorem nth_equation (n : ℕ) (h : 0 < n) : 9 * (n - 1) + n = 10 * n - 9 := 
  sorry

end nth_equation_l27_27083


namespace squares_difference_l27_27055

theorem squares_difference :
  1010^2 - 994^2 - 1008^2 + 996^2 = 8016 :=
by
  sorry

end squares_difference_l27_27055


namespace average_cost_correct_l27_27208

-- Defining the conditions
def groups_of_4_oranges := 11
def cost_of_4_oranges_bundle := 15
def groups_of_7_oranges := 2
def cost_of_7_oranges_bundle := 25

-- Calculating the relevant quantities as per the conditions
def total_cost : ℕ := (groups_of_4_oranges * cost_of_4_oranges_bundle) + (groups_of_7_oranges * cost_of_7_oranges_bundle)
def total_oranges : ℕ := (groups_of_4_oranges * 4) + (groups_of_7_oranges * 7)
def average_cost_per_orange := (total_cost:ℚ) / (total_oranges:ℚ)

-- Proving the average cost per orange matches the correct answer
theorem average_cost_correct : average_cost_per_orange = 215 / 58 := by
  sorry

end average_cost_correct_l27_27208


namespace perpendicular_line_through_point_l27_27236

noncomputable def line_eq_perpendicular (x y : ℝ) : Prop :=
  3 * x - 6 * y = 9

noncomputable def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem perpendicular_line_through_point
  (x y : ℝ)
  (hx : x = 2)
  (hy : y = -3) :
  ∀ x y, line_eq_perpendicular x y →
  ∃ m b, slope_intercept_form m b x y ∧ m = -2 ∧ b = 1 :=
sorry

end perpendicular_line_through_point_l27_27236


namespace largest_circle_radius_l27_27731

theorem largest_circle_radius 
  (h H : ℝ) (h_pos : h > 0) (H_pos : H > 0) :
  ∃ R, R = (h * H) / (h + H) :=
sorry

end largest_circle_radius_l27_27731


namespace vasya_days_without_purchases_l27_27473

theorem vasya_days_without_purchases 
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) : 
  w = 7 := 
sorry

end vasya_days_without_purchases_l27_27473


namespace largest_of_20_consecutive_even_integers_l27_27451

theorem largest_of_20_consecutive_even_integers (x : ℕ) 
  (h : 20 * (x + 19) = 8000) : (x + 38) = 419 :=
  sorry

end largest_of_20_consecutive_even_integers_l27_27451


namespace root_division_simplification_l27_27178

theorem root_division_simplification (a : ℝ) (h1 : a = (7 : ℝ)^(1/4)) (h2 : a = (7 : ℝ)^(1/7)) :
  ((7 : ℝ)^(1/4) / (7 : ℝ)^(1/7)) = (7 : ℝ)^(3/28) :=
sorry

end root_division_simplification_l27_27178


namespace range_of_m_l27_27683

theorem range_of_m :
  ∀ m, (∀ x, m ≤ x ∧ x ≤ 4 → (0 ≤ -x^2 + 4*x ∧ -x^2 + 4*x ≤ 4)) ↔ (0 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l27_27683


namespace find_b_value_l27_27762

theorem find_b_value (a b c : ℝ)
  (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
  (h2 : 6 * b * 7 = 1.5) : b = 15 := by
  sorry

end find_b_value_l27_27762


namespace find_x_l27_27287

theorem find_x (x : ℝ) (h1: x > 0) (h2 : 1 / 2 * x * (3 * x) = 72) : x = 4 * Real.sqrt 3 :=
sorry

end find_x_l27_27287


namespace circle_through_points_and_intercepts_l27_27340

noncomputable def circle_eq (x y D E F : ℝ) : ℝ := x^2 + y^2 + D * x + E * y + F

theorem circle_through_points_and_intercepts :
  ∃ (D E F : ℝ), 
    circle_eq 4 2 D E F = 0 ∧
    circle_eq (-1) 3 D E F = 0 ∧ 
    D + E = -2 ∧
    circle_eq x y (-2) 0 (-12) = 0 :=
by
  unfold circle_eq
  sorry

end circle_through_points_and_intercepts_l27_27340


namespace distinct_dragons_count_l27_27505

theorem distinct_dragons_count : 
  {n : ℕ // n = 7} :=
sorry

end distinct_dragons_count_l27_27505


namespace sum_of_digits_l27_27156

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem sum_of_digits (a b c d : ℕ) (h_distinct : distinct_digits a b c d) (h_eqn : 100*a + 60 + b - (400 + 10*c + d) = 2) :
  a + b + c + d = 10 ∨ a + b + c + d = 18 ∨ a + b + c + d = 19 :=
sorry

end sum_of_digits_l27_27156


namespace find_n_coordinates_l27_27746

variables {a b : ℝ}

def is_perpendicular (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def same_magnitude (m n : ℝ × ℝ) : Prop :=
  m.1 ^ 2 + m.2 ^ 2 = n.1 ^ 2 + n.2 ^ 2

theorem find_n_coordinates (n : ℝ × ℝ) (h1 : is_perpendicular (a, b) n) (h2 : same_magnitude (a, b) n) :
  n = (b, -a) :=
sorry

end find_n_coordinates_l27_27746


namespace necessary_but_not_sufficient_l27_27814

def condition1 (a b : ℝ) : Prop :=
  a > b

def statement (a b : ℝ) : Prop :=
  a > b + 1

theorem necessary_but_not_sufficient (a b : ℝ) (h : condition1 a b) : 
  (∀ a b : ℝ, statement a b → condition1 a b) ∧ ¬ (∀ a b : ℝ, condition1 a b → statement a b) :=
by 
  -- Proof skipped
  sorry

end necessary_but_not_sufficient_l27_27814


namespace charlies_mother_cookies_l27_27917

theorem charlies_mother_cookies 
    (charlie_cookies : ℕ) 
    (father_cookies : ℕ) 
    (total_cookies : ℕ)
    (h_charlie : charlie_cookies = 15)
    (h_father : father_cookies = 10)
    (h_total : total_cookies = 30) : 
    (total_cookies - charlie_cookies - father_cookies = 5) :=
by {
    sorry
}

end charlies_mother_cookies_l27_27917


namespace area_of_square_field_l27_27069

theorem area_of_square_field (x : ℝ) 
  (h₁ : 1.10 * (4 * x - 2) = 732.6) : 
  x = 167 → x ^ 2 = 27889 := by
  sorry

end area_of_square_field_l27_27069


namespace min_sum_p_q_r_s_l27_27320

theorem min_sum_p_q_r_s (p q r s : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
    (h1 : 2 * p = 10 * p - 15 * q)
    (h2 : 2 * q = 6 * p - 9 * q)
    (h3 : 3 * r = 10 * r - 15 * s)
    (h4 : 3 * s = 6 * r - 9 * s) : p + q + r + s = 45 := by
  sorry

end min_sum_p_q_r_s_l27_27320


namespace arithmetic_sequence_geometric_sum_l27_27444

theorem arithmetic_sequence_geometric_sum (a1 : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ (n : ℕ), S 1 = a1)
  (h2 : ∀ (n : ℕ), S 2 = 2 * a1 - 1)
  (h3 : ∀ (n : ℕ), S 4 = 4 * a1 - 6)
  (h4 : (2 * a1 - 1)^2 = a1 * (4 * a1 - 6)) 
  : a1 = -1/2 := 
sorry

end arithmetic_sequence_geometric_sum_l27_27444


namespace cricket_players_count_l27_27654

-- Define the conditions
def total_players_present : ℕ := 50
def hockey_players : ℕ := 17
def football_players : ℕ := 11
def softball_players : ℕ := 10

-- Define the result to prove
def cricket_players : ℕ := total_players_present - (hockey_players + football_players + softball_players)

-- The theorem stating the equivalence of cricket_players and the correct answer
theorem cricket_players_count : cricket_players = 12 := by
  -- A placeholder for the proof
  sorry

end cricket_players_count_l27_27654


namespace terminal_side_in_second_quadrant_l27_27847

theorem terminal_side_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
    0 < α ∧ α < π :=
sorry

end terminal_side_in_second_quadrant_l27_27847


namespace relationship_of_rationals_l27_27539

theorem relationship_of_rationals (a b c : ℚ) (h1 : a - b > 0) (h2 : b - c > 0) : c < b ∧ b < a :=
by {
  sorry
}

end relationship_of_rationals_l27_27539


namespace f_monotone_decreasing_without_min_value_l27_27924

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_monotone_decreasing_without_min_value :
  (∀ x y : ℝ, x < y → f y < f x) ∧ (∃ b : ℝ, ∀ x : ℝ, f x > b) :=
by
  sorry

end f_monotone_decreasing_without_min_value_l27_27924


namespace factory_car_production_l27_27103

theorem factory_car_production :
  let cars_yesterday := 60
  let cars_today := 2 * cars_yesterday
  let total_cars := cars_yesterday + cars_today
  total_cars = 180 :=
by
  sorry

end factory_car_production_l27_27103


namespace find_a_parallel_l27_27244

-- Define the lines
def line1 (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  (a + 1) * x + 2 * y = 2

def line2 (a : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  x + a * y = 1

-- Define the parallel condition
def are_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, line1 a x y → line2 a x y

-- The theorem stating our problem
theorem find_a_parallel (a : ℝ) : are_parallel a → a = -2 :=
by
  sorry

end find_a_parallel_l27_27244


namespace speed_of_stream_l27_27171

theorem speed_of_stream :
  ∃ (v : ℝ), (∀ (swim_speed : ℝ), swim_speed = 1.5 → 
    (∀ (time_upstream : ℝ) (time_downstream : ℝ), 
      time_upstream = 2 * time_downstream → 
      (1.5 + v) / (1.5 - v) = 2)) → v = 0.5 :=
sorry

end speed_of_stream_l27_27171


namespace describe_cylinder_l27_27628

noncomputable def cylinder_geometric_shape (c : ℝ) (r θ z : ℝ) : Prop :=
  r = c

theorem describe_cylinder (c : ℝ) (hc : 0 < c) :
  ∀ r θ z : ℝ, cylinder_geometric_shape c r θ z ↔ (r = c) :=
by
  sorry

end describe_cylinder_l27_27628


namespace bob_average_speed_l27_27944

theorem bob_average_speed
  (lap_distance : ℕ) (lap1_time lap2_time lap3_time total_laps : ℕ)
  (h_lap_distance : lap_distance = 400)
  (h_lap1_time : lap1_time = 70)
  (h_lap2_time : lap2_time = 85)
  (h_lap3_time : lap3_time = 85)
  (h_total_laps : total_laps = 3) : 
  (lap_distance * total_laps) / (lap1_time + lap2_time + lap3_time) = 5 := by
    sorry

end bob_average_speed_l27_27944


namespace question1_question2_l27_27753

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2 * x

-- Problem 1: Prove the valid solution of x when f(x) = 3 and x ∈ [0, 4]
theorem question1 (h₀ : 0 ≤ 3) (h₁ : 4 ≥ 3) : 
  ∃ (x : ℝ), (f x = 3 ∧ 0 ≤ x ∧ x ≤ 4) → x = 3 :=
by
  sorry

-- Problem 2: Prove the range of f(x) when x ∈ [0, 4]
theorem question2 : 
  ∃ (a b : ℝ), (∀ x, 0 ≤ x ∧ x ≤ 4 → a ≤ f x ∧ f x ≤ b) → a = -1 ∧ b = 8 :=
by
  sorry

end question1_question2_l27_27753


namespace solve_quadratic_solve_cubic_l27_27551

theorem solve_quadratic (x : ℝ) (h : 2 * x^2 - 32 = 0) : x = 4 ∨ x = -4 := 
by sorry

theorem solve_cubic (x : ℝ) (h : (x + 4)^3 + 64 = 0) : x = -8 := 
by sorry

end solve_quadratic_solve_cubic_l27_27551


namespace bert_bought_300_stamps_l27_27081

theorem bert_bought_300_stamps (x : ℝ) 
(H1 : x / 2 + x = 450) : x = 300 :=
by
  sorry

end bert_bought_300_stamps_l27_27081


namespace value_of_M_l27_27676

theorem value_of_M (G A M E: ℕ) (hG : G = 15)
(hGAME : G + A + M + E = 50)
(hMEGA : M + E + G + A = 55)
(hAGE : A + G + E = 40) : 
M = 15 := sorry

end value_of_M_l27_27676


namespace digit_divisibility_by_7_l27_27206

theorem digit_divisibility_by_7 (d : ℕ) (h : d < 10) : (10000 + 100 * d + 10) % 7 = 0 ↔ d = 5 :=
by
  sorry

end digit_divisibility_by_7_l27_27206


namespace smallest_prime_divisor_of_sum_of_powers_l27_27241

theorem smallest_prime_divisor_of_sum_of_powers :
  ∃ p, Prime p ∧ p = Nat.gcd (3 ^ 25 + 11 ^ 19) 2 := by
  sorry

end smallest_prime_divisor_of_sum_of_powers_l27_27241


namespace relationship_l27_27669

noncomputable def a : ℝ := Real.log (Real.log Real.pi)
noncomputable def b : ℝ := Real.log Real.pi
noncomputable def c : ℝ := 2^Real.log Real.pi

theorem relationship (a b c : ℝ) (ha : a = Real.log (Real.log Real.pi)) (hb : b = Real.log Real.pi) (hc : c = 2^Real.log Real.pi)
: a < b ∧ b < c := 
by
  sorry

end relationship_l27_27669


namespace printer_time_l27_27975

theorem printer_time (Tx : ℝ) 
  (h1 : ∀ (Ty Tz : ℝ), Ty = 10 → Tz = 20 → 1 / Ty + 1 / Tz = 3 / 20) 
  (h2 : ∀ (T_combined : ℝ), T_combined = 20 / 3 → Tx / T_combined = 2.4) :
  Tx = 16 := 
by 
  sorry

end printer_time_l27_27975


namespace evaluate_expression_l27_27504

variable (a b : ℝ) (h : a > b ∧ b > 0)

theorem evaluate_expression (h : a > b ∧ b > 0) : 
  (a^2 * b^3) / (b^2 * a^3) = (a / b)^(2 - 3) :=
  sorry

end evaluate_expression_l27_27504


namespace hillary_minutes_read_on_saturday_l27_27870

theorem hillary_minutes_read_on_saturday :
  let total_minutes := 60
  let friday_minutes := 16
  let sunday_minutes := 16
  total_minutes - (friday_minutes + sunday_minutes) = 28 := by
sorry

end hillary_minutes_read_on_saturday_l27_27870


namespace gerald_jail_time_l27_27529

theorem gerald_jail_time
    (assault_sentence : ℕ := 3) 
    (poisoning_sentence_years : ℕ := 2) 
    (third_offense_extension : ℕ := 1 / 3) 
    (months_in_year : ℕ := 12)
    : (assault_sentence + poisoning_sentence_years * months_in_year) * (1 + third_offense_extension) = 36 :=
by
  sorry

end gerald_jail_time_l27_27529


namespace jeremy_is_40_l27_27279

-- Definitions for Jeremy (J), Sebastian (S), and Sophia (So)
def JeremyCurrentAge : ℕ := 40
def SebastianCurrentAge : ℕ := JeremyCurrentAge + 4
def SophiaCurrentAge : ℕ := 60 - 3

-- Assertion properties
axiom age_sum_in_3_years : (JeremyCurrentAge + 3) + (SebastianCurrentAge + 3) + (SophiaCurrentAge + 3) = 150
axiom sebastian_older_by_4 : SebastianCurrentAge = JeremyCurrentAge + 4
axiom sophia_age_in_3_years : SophiaCurrentAge + 3 = 60

-- The theorem to prove that Jeremy is currently 40 years old
theorem jeremy_is_40 : JeremyCurrentAge = 40 := by
  sorry

end jeremy_is_40_l27_27279


namespace arithmetic_seq_sum_l27_27210

-- Definition of an arithmetic sequence using a common difference d
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Statement of the problem
theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) 
  (hs : arithmetic_sequence a d)
  (hmean : (a 3 + a 8) / 2 = 10) : 
  a 1 + a 10 = 20 :=
sorry

end arithmetic_seq_sum_l27_27210


namespace divided_scale_length_l27_27931

/-
  The problem definition states that we have a scale that is 6 feet 8 inches long, 
  and we need to prove that when the scale is divided into two equal parts, 
  each part is 3 feet 4 inches long.
-/

/-- Given length conditions in feet and inches --/
def total_length_feet : ℕ := 6
def total_length_inches : ℕ := 8

/-- Convert total length to inches --/
def total_length_in_inches := total_length_feet * 12 + total_length_inches

/-- Proof that if a scale is 6 feet 8 inches long and divided into 2 parts, each part is 3 feet 4 inches --/
theorem divided_scale_length :
  (total_length_in_inches / 2) = 40 ∧ (40 / 12 = 3 ∧ 40 % 12 = 4) :=
by
  sorry

end divided_scale_length_l27_27931


namespace caleb_trip_duration_l27_27363

-- Define the times when the clock hands meet
def startTime := 7 * 60 + 38 -- 7:38 a.m. in minutes from midnight
def endTime := 13 * 60 + 5 -- 1:05 p.m. in minutes from midnight

def duration := endTime - startTime

theorem caleb_trip_duration :
  duration = 5 * 60 + 27 := by
sorry

end caleb_trip_duration_l27_27363


namespace inscribed_circle_radii_rel_l27_27997

theorem inscribed_circle_radii_rel {a b c r r1 r2 : ℝ} :
  (a^2 + b^2 = c^2) ∧
  (r1 = (a / c) * r) ∧
  (r2 = (b / c) * r) →
  r^2 = r1^2 + r2^2 :=
by 
  sorry

end inscribed_circle_radii_rel_l27_27997


namespace divisible_by_five_l27_27358

theorem divisible_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
sorry

end divisible_by_five_l27_27358


namespace part_one_part_two_l27_27428

-- Part 1:
-- Define the function f
def f (x : ℝ) : ℝ := abs (2 * x - 3) + abs (2 * x + 2)

-- Define the inequality problem
theorem part_one (x : ℝ) : f x < x + 5 ↔ 0 < x ∧ x < 2 :=
by sorry

-- Part 2:
-- Define the condition for part 2
theorem part_two (a : ℝ) : (∀ x : ℝ, f x > a + 4 / a) ↔ (a ∈ Set.Ioo 1 4 ∨ a < 0) :=
by sorry

end part_one_part_two_l27_27428


namespace planting_cost_l27_27995

-- Define the costs of the individual items
def cost_of_flowers : ℝ := 9
def cost_of_clay_pot : ℝ := cost_of_flowers + 20
def cost_of_soil : ℝ := cost_of_flowers - 2
def cost_of_fertilizer : ℝ := cost_of_flowers + (0.5 * cost_of_flowers)
def cost_of_tools : ℝ := cost_of_clay_pot - (0.25 * cost_of_clay_pot)

-- Define the total cost
def total_cost : ℝ :=
  cost_of_flowers + cost_of_clay_pot + cost_of_soil + cost_of_fertilizer + cost_of_tools

-- The statement to prove
theorem planting_cost : total_cost = 80.25 :=
by
  sorry

end planting_cost_l27_27995


namespace chess_group_players_l27_27091

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
by {
  sorry
}

end chess_group_players_l27_27091


namespace inequality_proof_l27_27040

theorem inequality_proof (x : ℝ) (n : ℕ) (h : 3 * x ≥ -1) : (1 + x) ^ n ≥ 1 + n * x :=
sorry

end inequality_proof_l27_27040


namespace arithmetic_sequence_T_n_bound_l27_27319

open Nat

theorem arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (h2 : a 2 = 6) (h3_h6 : a 3 + a 6 = 27) :
  (∀ n, a n = 3 * n) := 
by
  sorry

theorem T_n_bound (a : ℕ → ℤ) (S : ℕ → ℤ) (T : ℕ → ℝ) (m : ℝ) (h_general_term : ∀ n, a n = 3 * n) 
  (h_S_n : ∀ n, S n = n^2 + n) (h_T_n : ∀ n, T n = (S n : ℝ) / (3 * (2 : ℝ)^(n-1)))
  (h_bound : ∀ n > 0, T n ≤ m) : 
  m ≥ 3/2 :=
by
  sorry

end arithmetic_sequence_T_n_bound_l27_27319


namespace borrow_years_l27_27479

/-- A person borrows Rs. 5000 at 4% p.a simple interest and lends it at 6% p.a simple interest.
His gain in the transaction per year is Rs. 100. Prove that he borrowed the money for 1 year. --/
theorem borrow_years
  (principal : ℝ)
  (borrow_rate : ℝ)
  (lend_rate : ℝ)
  (gain : ℝ)
  (interest_paid_per_year : ℝ)
  (interest_earned_per_year : ℝ) :
  (principal = 5000) →
  (borrow_rate = 0.04) →
  (lend_rate = 0.06) →
  (gain = 100) →
  (interest_paid_per_year = principal * borrow_rate) →
  (interest_earned_per_year = principal * lend_rate) →
  (interest_earned_per_year - interest_paid_per_year = gain) →
  1 = 1 := 
by
  -- Placeholder for the proof
  sorry

end borrow_years_l27_27479


namespace total_tea_cups_l27_27876

def num_cupboards := 8
def num_compartments_per_cupboard := 5
def num_tea_cups_per_compartment := 85

theorem total_tea_cups :
  num_cupboards * num_compartments_per_cupboard * num_tea_cups_per_compartment = 3400 :=
by
  sorry

end total_tea_cups_l27_27876


namespace correct_fraction_simplification_l27_27401

theorem correct_fraction_simplification (a b : ℝ) (h : a ≠ b) : 
  (∀ (c d : ℝ), (c ≠ d) → (a+2 = c → b+2 = d → (a+2)/d ≠ a/b))
  ∧ (∀ (e f : ℝ), (e ≠ f) → (a-2 = e → b-2 = f → (a-2)/f ≠ a/b))
  ∧ (∀ (g h : ℝ), (g ≠ h) → (a^2 = g → b^2 = h → a^2/h ≠ a/b))
  ∧ (a / b = ( (1/2)*a / (1/2)*b )) := 
sorry

end correct_fraction_simplification_l27_27401


namespace cubes_sum_expr_l27_27443

variable {a b s p : ℝ}

theorem cubes_sum_expr (h1 : s = a + b) (h2 : p = a * b) : a^3 + b^3 = s^3 - 3 * s * p := by
  sorry

end cubes_sum_expr_l27_27443


namespace bananas_used_l27_27690

-- Define the conditions
def bananas_per_loaf : Nat := 4
def loaves_on_monday : Nat := 3
def loaves_on_tuesday : Nat := 2 * loaves_on_monday
def total_loaves : Nat := loaves_on_monday + loaves_on_tuesday

-- Define the total bananas used
def total_bananas : Nat := bananas_per_loaf * total_loaves

-- Prove that the total bananas used is 36
theorem bananas_used : total_bananas = 36 :=
by
  sorry

end bananas_used_l27_27690


namespace four_digit_integer_5533_l27_27023

theorem four_digit_integer_5533
  (a b c d : ℕ)
  (h1 : a + b + c + d = 16)
  (h2 : b + c = 8)
  (h3 : a - d = 2)
  (h4 : (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0) :
  1000 * a + 100 * b + 10 * c + d = 5533 :=
by {
  sorry
}

end four_digit_integer_5533_l27_27023


namespace production_rate_problem_l27_27159

theorem production_rate_problem :
  ∀ (G T : ℕ), 
  (∀ w t, w * 3 * t = 450 * t / 150) ∧
  (∀ w t, w * 2 * t = 300 * t / 150) ∧
  (∀ w t, w * 2 * t = 360 * t / 90) ∧
  (∀ w t, w * (5/2) * t = 450 * t / 90) ∧
  (75 * 2 * 4 = 300) →
  (75 * 2 * 4 = 600) := sorry

end production_rate_problem_l27_27159


namespace max_value_of_y_l27_27117

noncomputable def max_value_of_function : ℝ := 1 + Real.sqrt 2

theorem max_value_of_y : ∀ x : ℝ, (2 * Real.sin x * (Real.sin x + Real.cos x)) ≤ max_value_of_function :=
by
  -- Proof goes here
  sorry

example : ∃ x : ℝ, (2 * Real.sin x * (Real.sin x + Real.cos x)) = max_value_of_function :=
by
  -- Proof goes here
  sorry

end max_value_of_y_l27_27117


namespace min_dot_product_PA_PB_l27_27659

noncomputable def point_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1
noncomputable def point_on_ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem min_dot_product_PA_PB (A B P : ℝ × ℝ)
  (hA : point_on_circle A.1 A.2)
  (hB : point_on_circle B.1 B.2)
  (hAB : A ≠ B ∧ (B.1 = -A.1) ∧ (B.2 = -A.2))
  (hP : point_on_ellipse P.1 P.2) :
  ∃ PA PB : ℝ × ℝ, 
    PA = (P.1 - A.1, P.2 - A.2) ∧ PB = (P.1 - B.1, P.2 - B.2) ∧
    (PA.1 * PB.1 + PA.2 * PB.2) = 2 :=
by sorry

end min_dot_product_PA_PB_l27_27659


namespace parallel_lines_solution_l27_27120

theorem parallel_lines_solution (a : ℝ) :
  (∃ (k1 k2 : ℝ), k1 ≠ 0 ∧ k2 ≠ 0 ∧ 
  ∀ x y : ℝ, x + a^2 * y + 6 = 0 → k1*y = x ∧ 
             (a-2) * x + 3 * a * y + 2 * a = 0 → k2*y = x) 
  → (a = -1 ∨ a = 0) :=
by
  sorry

end parallel_lines_solution_l27_27120


namespace projectile_height_reaches_35_l27_27590

theorem projectile_height_reaches_35 
  (t : ℝ)
  (h_eq : -4.9 * t^2 + 30 * t = 35) :
  t = 2 ∨ t = 50 / 7 ∧ t = min (2 : ℝ) (50 / 7) :=
by
  sorry

end projectile_height_reaches_35_l27_27590


namespace area_of_sector_radius_2_angle_90_l27_27513

-- Given conditions
def radius := 2
def central_angle := 90

-- Required proof: the area of the sector with given conditions equals π.
theorem area_of_sector_radius_2_angle_90 : (90 * Real.pi * (2^2) / 360) = Real.pi := 
by
  sorry

end area_of_sector_radius_2_angle_90_l27_27513


namespace cost_of_dowels_l27_27102

variable (V S : ℝ)

theorem cost_of_dowels 
  (hV : V = 7)
  (h_eq : 0.85 * (V + S) = V + 0.5 * S) :
  S = 3 :=
by
  sorry

end cost_of_dowels_l27_27102


namespace Kyle_rose_cost_l27_27627

/-- Given the number of roses Kyle picked last year, the number of roses he picked this year, 
and the cost of one rose, prove that the total cost he has to spend to buy the remaining roses 
is correct. -/
theorem Kyle_rose_cost (last_year_roses this_year_roses total_roses_needed cost_per_rose : ℕ)
    (h_last_year_roses : last_year_roses = 12) 
    (h_this_year_roses : this_year_roses = last_year_roses / 2) 
    (h_total_roses_needed : total_roses_needed = 2 * last_year_roses) 
    (h_cost_per_rose : cost_per_rose = 3) : 
    (total_roses_needed - this_year_roses) * cost_per_rose = 54 := 
by
sorry

end Kyle_rose_cost_l27_27627


namespace probability_purple_or_orange_face_l27_27071

theorem probability_purple_or_orange_face 
  (total_faces : ℕ) (green_faces : ℕ) (purple_faces : ℕ) (orange_faces : ℕ) 
  (h_total : total_faces = 10) 
  (h_green : green_faces = 5) 
  (h_purple : purple_faces = 3) 
  (h_orange : orange_faces = 2) :
  (purple_faces + orange_faces) / total_faces = 1 / 2 :=
by 
  sorry

end probability_purple_or_orange_face_l27_27071


namespace relative_error_comparison_l27_27806

theorem relative_error_comparison :
  (0.05 / 25 = 0.002) ∧ (0.4 / 200 = 0.002) → (0.002 = 0.002) :=
by
  sorry

end relative_error_comparison_l27_27806


namespace fourth_term_of_sequence_l27_27397

theorem fourth_term_of_sequence (x : ℤ) (h : x^2 - 2 * x - 3 < 0) (hx : x ∈ {n : ℤ | x^2 - 2 * x - 3 < 0}) :
  ∃ a_1 a_2 a_3 a_4 : ℤ, 
  (a_1 = x) ∧ (a_2 = x + 1) ∧ (a_3 = x + 2) ∧ (a_4 = x + 3) ∧ 
  (a_4 = 3 ∨ a_4 = -1) :=
by { sorry }

end fourth_term_of_sequence_l27_27397


namespace division_remainder_l27_27785

theorem division_remainder (dividend divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 15) 
  (h_quotient : quotient = 9) 
  (h_dividend_eq : dividend = 136) 
  (h_eq : dividend = (divisor * quotient) + remainder) : 
  remainder = 1 :=
by
  sorry

end division_remainder_l27_27785


namespace tan_neg_240_eq_neg_sqrt_3_l27_27592

theorem tan_neg_240_eq_neg_sqrt_3 : Real.tan (-4 * Real.pi / 3) = -Real.sqrt 3 :=
by
  sorry

end tan_neg_240_eq_neg_sqrt_3_l27_27592


namespace repeatingDecimals_fraction_eq_l27_27200

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end repeatingDecimals_fraction_eq_l27_27200


namespace daffodil_bulb_cost_l27_27863

theorem daffodil_bulb_cost :
  let total_bulbs := 55
  let crocus_cost := 0.35
  let total_budget := 29.15
  let num_crocus_bulbs := 22
  let total_crocus_cost := num_crocus_bulbs * crocus_cost
  let remaining_budget := total_budget - total_crocus_cost
  let num_daffodil_bulbs := total_bulbs - num_crocus_bulbs
  remaining_budget / num_daffodil_bulbs = 0.65 := 
by
  -- proof to be filled in
  sorry

end daffodil_bulb_cost_l27_27863


namespace XAXAXA_divisible_by_seven_l27_27332

theorem XAXAXA_divisible_by_seven (X A : ℕ) (hX : X < 10) (hA : A < 10) : 
  (101010 * X + 10101 * A) % 7 = 0 := 
by 
  sorry

end XAXAXA_divisible_by_seven_l27_27332


namespace number_of_functions_l27_27895

-- Define the set of conditions
variables (x y : ℝ)

def relation1 := x - y = 0
def relation2 := y^2 = x
def relation3 := |y| = 2 * x
def relation4 := y^2 = x^2
def relation5 := y = 3 - x
def relation6 := y = 2 * x^2 - 1
def relation7 := y = 3 / x

-- Prove that there are 4 unambiguous functions of y with respect to x
theorem number_of_functions : 4 = 4 := sorry

end number_of_functions_l27_27895


namespace fishing_problem_l27_27935

theorem fishing_problem
  (P : ℕ) -- weight of the fish Peter caught
  (H1 : Ali_weight = 2 * P) -- Ali caught twice as much as Peter
  (H2 : Joey_weight = P + 1) -- Joey caught 1 kg more than Peter
  (H3 : P + 2 * P + (P + 1) = 25) -- Together they caught 25 kg
  : Ali_weight = 12 :=
by
  sorry

end fishing_problem_l27_27935


namespace probability_of_valid_quadrilateral_l27_27301

-- Define a regular octagon
def regular_octagon_sides : ℕ := 8

-- Total number of ways to choose 4 sides from 8 sides
def total_ways_choose_four_sides : ℕ := Nat.choose 8 4

-- Number of ways to choose 4 adjacent sides (invalid)
def invalid_adjacent_ways : ℕ := 8

-- Number of ways to choose 4 sides with 3 adjacent unchosen sides (invalid)
def invalid_three_adjacent_unchosen_ways : ℕ := 8 * 3

-- Total number of invalid ways
def total_invalid_ways : ℕ := invalid_adjacent_ways + invalid_three_adjacent_unchosen_ways

-- Total number of valid ways
def total_valid_ways : ℕ := total_ways_choose_four_sides - total_invalid_ways

-- Probability of forming a quadrilateral that contains the octagon
def probability_valid_quadrilateral : ℚ :=
  (total_valid_ways : ℚ) / (total_ways_choose_four_sides : ℚ)

-- Theorem statement
theorem probability_of_valid_quadrilateral :
  probability_valid_quadrilateral = 19 / 35 :=
by
  sorry

end probability_of_valid_quadrilateral_l27_27301


namespace diagonals_diff_heptagon_octagon_l27_27932

-- Define the function to calculate the number of diagonals in a polygon with n sides
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_diff_heptagon_octagon : 
  let A := num_diagonals 7
  let B := num_diagonals 8
  B - A = 6 :=
by
  sorry

end diagonals_diff_heptagon_octagon_l27_27932


namespace jose_share_of_profit_l27_27267

-- Definitions from problem conditions
def tom_investment : ℕ := 30000
def jose_investment : ℕ := 45000
def profit : ℕ := 27000
def months_total : ℕ := 12
def months_jose_investment : ℕ := 10

-- Derived calculations
def tom_month_investment := tom_investment * months_total
def jose_month_investment := jose_investment * months_jose_investment
def total_month_investment := tom_month_investment + jose_month_investment

-- Prove Jose's share of profit
theorem jose_share_of_profit : (jose_month_investment * profit) / total_month_investment = 15000 := by
  -- This is where the step-by-step proof would go
  sorry

end jose_share_of_profit_l27_27267


namespace triangle_inequality_l27_27872

theorem triangle_inequality (x : ℕ) (hx : x > 0) :
  (x ≥ 34) ↔ (x + (10 + x) > 24) ∧ (x + 24 > 10 + x) ∧ ((10 + x) + 24 > x) := by
  sorry

end triangle_inequality_l27_27872


namespace area_of_union_of_rectangle_and_circle_l27_27996

theorem area_of_union_of_rectangle_and_circle :
  let width := 8
  let length := 12
  let radius := 12
  let A_rectangle := length * width
  let A_circle := Real.pi * radius ^ 2
  let A_overlap := (1 / 4) * A_circle
  A_rectangle + A_circle - A_overlap = 96 + 108 * Real.pi :=
by
  sorry

end area_of_union_of_rectangle_and_circle_l27_27996


namespace line_AB_eq_x_plus_3y_zero_l27_27276

/-- 
Consider two circles defined by:
C1: x^2 + y^2 - 4x + 6y = 0
C2: x^2 + y^2 - 6x = 0

Prove that the equation of the line through the intersection points of these two circles (line AB)
is x + 3y = 0.
-/
theorem line_AB_eq_x_plus_3y_zero (x y : ℝ) :
  (x^2 + y^2 - 4 * x + 6 * y = 0) ∧ (x^2 + y^2 - 6 * x = 0) → (x + 3 * y = 0) :=
by
  sorry

end line_AB_eq_x_plus_3y_zero_l27_27276


namespace average_price_of_pencil_correct_l27_27313

def average_price_of_pencil (n_pens n_pencils : ℕ) (total_cost pen_price : ℕ) : ℕ :=
  let pen_cost := n_pens * pen_price
  let pencil_cost := total_cost - pen_cost
  let avg_pencil_price := pencil_cost / n_pencils
  avg_pencil_price

theorem average_price_of_pencil_correct :
  average_price_of_pencil 30 75 450 10 = 2 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end average_price_of_pencil_correct_l27_27313


namespace game_winner_Aerith_first_game_winner_Bob_first_l27_27693

-- Conditions: row of 20 squares, players take turns crossing out one square,
-- game ends when there are two squares left, Aerith wins if two remaining squares
-- are adjacent, Bob wins if they are not adjacent.

-- Definition of the game and winning conditions
inductive Player
| Aerith
| Bob

-- Function to determine the winner given the initial player
def winning_strategy (initial_player : Player) : Player :=
  match initial_player with
  | Player.Aerith => Player.Bob  -- Bob wins if Aerith goes first
  | Player.Bob    => Player.Aerith  -- Aerith wins if Bob goes first

-- Statement to prove
theorem game_winner_Aerith_first : 
  winning_strategy Player.Aerith = Player.Bob :=
by 
  sorry -- Proof is to be done

theorem game_winner_Bob_first :
  winning_strategy Player.Bob = Player.Aerith :=
by
  sorry -- Proof is to be done

end game_winner_Aerith_first_game_winner_Bob_first_l27_27693


namespace part_a_impossible_part_b_possible_l27_27448

-- Part (a)
theorem part_a_impossible (a : ℝ) (h₁ : 1 < a) (h₂ : a ≠ 2) :
  ¬ ∀ (x : ℝ), (1 < x ∧ x < a) ∧ (a < 2*x ∧ 2*x < a^2) :=
sorry

-- Part (b)
theorem part_b_possible (a : ℝ) (h₁ : 1 < a) (h₂ : a ≠ 2) :
  ∃ (x : ℝ), (a < 2*x ∧ 2*x < a^2) ∧ ¬ (1 < x ∧ x < a) :=
sorry

end part_a_impossible_part_b_possible_l27_27448


namespace root_of_function_is_four_l27_27955

noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

theorem root_of_function_is_four (a : ℝ) (h : f a = 0) : a = 4 :=
by
  sorry

end root_of_function_is_four_l27_27955


namespace gcd_1855_1120_l27_27274

theorem gcd_1855_1120 : Int.gcd 1855 1120 = 35 :=
by
  sorry

end gcd_1855_1120_l27_27274


namespace zombies_count_decrease_l27_27374

theorem zombies_count_decrease (z : ℕ) (d : ℕ) : z = 480 → (∀ n, d = 2^n * z) → ∃ t, d / t < 50 :=
by
  intros hz hdz
  let initial_count := 480
  have := 480 / (2 ^ 4)
  sorry

end zombies_count_decrease_l27_27374


namespace different_color_socks_l27_27698

def total_socks := 15
def white_socks := 6
def brown_socks := 5
def blue_socks := 4

theorem different_color_socks (total : ℕ) (white : ℕ) (brown : ℕ) (blue : ℕ) :
  total = white + brown + blue →
  white ≠ 0 → brown ≠ 0 → blue ≠ 0 →
  (white * brown + brown * blue + white * blue) = 74 :=
by
  intros
  -- proof goes here
  sorry

end different_color_socks_l27_27698


namespace part_a_no_solutions_part_a_infinite_solutions_l27_27160

theorem part_a_no_solutions (a : ℝ) (x y : ℝ) : 
    a = -1 → ¬(∃ x y : ℝ, a * x + y = a^2 ∧ x + a * y = 1) :=
sorry

theorem part_a_infinite_solutions (a : ℝ) (x y : ℝ) : 
    a = 1 → ∃ x : ℝ, ∃ y : ℝ, a * x + y = a^2 ∧ x + a * y = 1 :=
sorry

end part_a_no_solutions_part_a_infinite_solutions_l27_27160


namespace percentage_salt_solution_l27_27848

theorem percentage_salt_solution (P : ℝ) (V_initial V_added V_final : ℝ) (C_initial C_final : ℝ) :
  V_initial = 30 ∧ C_initial = 0.20 ∧ V_final = 60 ∧ C_final = 0.40 → 
  V_added = 30 → 
  (C_initial * V_initial + (P / 100) * V_added) / V_final = C_final →
  P = 60 :=
by
  intro h
  sorry

end percentage_salt_solution_l27_27848


namespace pqrs_product_l27_27940

noncomputable def P := (Real.sqrt 2007 + Real.sqrt 2008)
noncomputable def Q := (-Real.sqrt 2007 - Real.sqrt 2008)
noncomputable def R := (Real.sqrt 2007 - Real.sqrt 2008)
noncomputable def S := (-Real.sqrt 2008 + Real.sqrt 2007)

theorem pqrs_product : P * Q * R * S = -1 := by
  sorry

end pqrs_product_l27_27940


namespace gas_usage_l27_27334

def distance_dermatologist : ℕ := 30
def distance_gynecologist : ℕ := 50
def car_efficiency : ℕ := 20

theorem gas_usage (d_1 d_2 e : ℕ) (H1 : d_1 = distance_dermatologist) (H2 : d_2 = distance_gynecologist) (H3 : e = car_efficiency) :
  (2 * d_1 + 2 * d_2) / e = 8 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end gas_usage_l27_27334


namespace relationship_among_vars_l27_27844

theorem relationship_among_vars {a b c d : ℝ} (h : (a + 2 * b) / (b + 2 * c) = (c + 2 * d) / (d + 2 * a)) :
  b = 2 * a ∨ a + b + c + d = 0 :=
sorry

end relationship_among_vars_l27_27844


namespace max_value_7x_10y_z_l27_27769

theorem max_value_7x_10y_z (x y z : ℝ) 
  (h : x^2 + 2 * x + (1 / 5) * y^2 + 7 * z^2 = 6) : 
  7 * x + 10 * y + z ≤ 55 := 
sorry

end max_value_7x_10y_z_l27_27769


namespace part1_part2_part3_l27_27290

-- Part 1
theorem part1 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z :=
sorry

-- Part 2
theorem part2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x^2 + y^2 + z^2 ≥ x * y + y * z + z * x :=
sorry

-- Part 3
theorem part3 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x ^ x * y ^ y * z ^ z ≥ (x * y * z) ^ ((x + y + z) / 3) :=
sorry

#print axioms part1
#print axioms part2
#print axioms part3

end part1_part2_part3_l27_27290


namespace domain_and_range_of_g_l27_27623

noncomputable def f : ℝ → ℝ := sorry-- Given: a function f with domain [0,2] and range [0,1]
noncomputable def g (x : ℝ) := 1 - f (x / 2 + 1)

theorem domain_and_range_of_g :
  let dom_g := { x | -2 ≤ x ∧ x ≤ 2 }
  let range_g := { y | 0 ≤ y ∧ y ≤ 1 }
  ∀ (x : ℝ), (x ∈ dom_g → (g x) ∈ range_g) := 
sorry

end domain_and_range_of_g_l27_27623


namespace total_buyers_in_three_days_l27_27238

theorem total_buyers_in_three_days
  (D_minus_2 : ℕ)
  (D_minus_1 : ℕ)
  (D_0 : ℕ)
  (h1 : D_minus_2 = 50)
  (h2 : D_minus_1 = D_minus_2 / 2)
  (h3 : D_0 = D_minus_1 + 40) :
  D_minus_2 + D_minus_1 + D_0 = 140 :=
by
  sorry

end total_buyers_in_three_days_l27_27238


namespace benito_juarez_birth_year_l27_27000

theorem benito_juarez_birth_year (x : ℕ) (h1 : 1801 ≤ x ∧ x ≤ 1850) (h2 : x*x = 1849) : x = 1806 :=
by sorry

end benito_juarez_birth_year_l27_27000


namespace geometric_series_common_ratio_l27_27121

theorem geometric_series_common_ratio (a r S : ℝ) (h₁ : S = a / (1 - r)) (h₂ : ar^4 / (1 - r) = S / 64) : r = 1 / 2 :=
  by
  sorry

end geometric_series_common_ratio_l27_27121


namespace trapezoid_ratio_l27_27464

structure Trapezoid (α : Type) [LinearOrderedField α] :=
  (AB CD : α)
  (areas : List α)
  (AB_gt_CD : AB > CD)
  (areas_eq : areas = [3, 5, 6, 8])

open Trapezoid

theorem trapezoid_ratio (α : Type) [LinearOrderedField α] (T : Trapezoid α) :
  ∃ ρ : α, T.AB / T.CD = ρ ∧ ρ = 8 / 3 :=
by
  sorry

end trapezoid_ratio_l27_27464


namespace cube_properties_l27_27783

theorem cube_properties (y : ℝ) (s : ℝ) 
  (h_volume : s^3 = 6 * y)
  (h_surface_area : 6 * s^2 = 2 * y) :
  y = 5832 :=
by sorry

end cube_properties_l27_27783


namespace max_value_condition_l27_27253

noncomputable def f (a x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ a then
  Real.log x
else
  if x > a then
    a / x
  else
    0 -- This case should not happen given the domain conditions

theorem max_value_condition (a : ℝ) : 
  (∃ M, ∀ x > 0, x ≤ a → f a x ≤ M) ∧ (∀ x > a, f a x ≤ M) ↔ a ≥ Real.exp 1 :=
sorry

end max_value_condition_l27_27253


namespace triple_hash_100_l27_27038

def hash (N : ℝ) : ℝ :=
  0.5 * N + N

theorem triple_hash_100 : hash (hash (hash 100)) = 337.5 :=
by
  sorry

end triple_hash_100_l27_27038


namespace original_square_area_l27_27008

noncomputable def area_of_original_square (x : ℝ) (w : ℝ) (remaining_area : ℝ) : Prop :=
  x * x = remaining_area + x * w

theorem original_square_area : ∃ (x : ℝ), area_of_original_square x 3 40 → x * x = 64 :=
sorry

end original_square_area_l27_27008


namespace pentagon_sum_of_sides_and_vertices_eq_10_l27_27240

-- Define the number of sides of a pentagon
def number_of_sides : ℕ := 5

-- Define the number of vertices of a pentagon
def number_of_vertices : ℕ := 5

-- Define the sum of sides and vertices
def sum_of_sides_and_vertices : ℕ :=
  number_of_sides + number_of_vertices

-- The theorem to prove that the sum is 10
theorem pentagon_sum_of_sides_and_vertices_eq_10 : sum_of_sides_and_vertices = 10 :=
by
  sorry

end pentagon_sum_of_sides_and_vertices_eq_10_l27_27240


namespace x_eq_1_iff_quadratic_eq_zero_l27_27821

theorem x_eq_1_iff_quadratic_eq_zero :
  ∀ x : ℝ, (x = 1) ↔ (x^2 - 2 * x + 1 = 0) := by
  sorry

end x_eq_1_iff_quadratic_eq_zero_l27_27821


namespace general_term_of_sequence_l27_27557

noncomputable def seq (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 9
  | 3 => 14
  | 4 => 21
  | 5 => 30
  | _ => sorry

theorem general_term_of_sequence :
  ∀ n : ℕ, seq n = 5 + n^2 :=
by
  sorry

end general_term_of_sequence_l27_27557


namespace gcd_4557_1953_5115_l27_27833

theorem gcd_4557_1953_5115 : Nat.gcd (Nat.gcd 4557 1953) 5115 = 93 :=
by
  -- We use 'sorry' to skip the proof part as per the instructions.
  sorry

end gcd_4557_1953_5115_l27_27833


namespace sum_divisible_by_5_and_7_l27_27851

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_divisible_by_5_and_7 (A B : ℕ) (hA_prime : is_prime A) 
  (hB_prime : is_prime B) (hA_minus_3_prime : is_prime (A - 3)) 
  (hA_plus_3_prime : is_prime (A + 3)) (hB_eq_2 : B = 2) : 
  5 ∣ (A + B + (A - 3) + (A + 3)) ∧ 7 ∣ (A + B + (A - 3) + (A + 3)) := by 
  sorry

end sum_divisible_by_5_and_7_l27_27851


namespace combinations_of_balls_and_hats_l27_27039

def validCombinations (b h : ℕ) : Prop :=
  6 * b + 4 * h = 100 ∧ h ≥ 2

theorem combinations_of_balls_and_hats : 
  (∃ (n : ℕ), n = 8 ∧ (∀ b h : ℕ, validCombinations b h → validCombinations b h)) :=
by
  sorry

end combinations_of_balls_and_hats_l27_27039


namespace line_tangent_to_ellipse_l27_27075

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * (m * x + 2)^2 = 3) ∧ 
  (∀ x1 x2 : ℝ, (2 + 3 * m^2) * x1^2 + 12 * m * x1 + 9 = 0 ∧ 
                (2 + 3 * m^2) * x2^2 + 12 * m * x2 + 9 = 0 → x1 = x2) ↔ m^2 = 2 := 
sorry

end line_tangent_to_ellipse_l27_27075


namespace combined_solid_sum_faces_edges_vertices_l27_27018

noncomputable def prism_faces : ℕ := 6
noncomputable def prism_edges : ℕ := 12
noncomputable def prism_vertices : ℕ := 8
noncomputable def new_pyramid_faces : ℕ := 4
noncomputable def new_pyramid_edges : ℕ := 4
noncomputable def new_pyramid_vertex : ℕ := 1

theorem combined_solid_sum_faces_edges_vertices :
  prism_faces - 1 + new_pyramid_faces + prism_edges + new_pyramid_edges + prism_vertices + new_pyramid_vertex = 34 :=
by
  -- proof would go here
  sorry

end combined_solid_sum_faces_edges_vertices_l27_27018


namespace decagon_adjacent_probability_l27_27514

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l27_27514


namespace solve_g_eq_g_inv_l27_27852

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5

noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem solve_g_eq_g_inv : 
  ∃ x : ℝ, g x = g_inv x ∧ x = 5 / 3 :=
by
  sorry

end solve_g_eq_g_inv_l27_27852


namespace hotel_profit_calculation_l27_27582

theorem hotel_profit_calculation
  (operations_expenses : ℝ)
  (meetings_fraction : ℝ) (events_fraction : ℝ) (rooms_fraction : ℝ)
  (meetings_tax_rate : ℝ) (meetings_commission_rate : ℝ)
  (events_tax_rate : ℝ) (events_commission_rate : ℝ)
  (rooms_tax_rate : ℝ) (rooms_commission_rate : ℝ)
  (total_profit : ℝ) :
  operations_expenses = 5000 →
  meetings_fraction = 5/8 →
  events_fraction = 3/10 →
  rooms_fraction = 11/20 →
  meetings_tax_rate = 0.10 →
  meetings_commission_rate = 0.05 →
  events_tax_rate = 0.08 →
  events_commission_rate = 0.06 →
  rooms_tax_rate = 0.12 →
  rooms_commission_rate = 0.03 →
  total_profit = (operations_expenses * (meetings_fraction + events_fraction + rooms_fraction)
                - (operations_expenses
                  + operations_expenses * (meetings_fraction * (meetings_tax_rate + meetings_commission_rate)
                  + events_fraction * (events_tax_rate + events_commission_rate)
                  + rooms_fraction * (rooms_tax_rate + rooms_commission_rate)))) ->
  total_profit = 1283.75 :=
by sorry

end hotel_profit_calculation_l27_27582


namespace approximate_reading_l27_27309

-- Define the given conditions
def arrow_location_between (a b : ℝ) : Prop := a < 42.3 ∧ 42.6 < b

-- Statement of the proof problem
theorem approximate_reading (a b : ℝ) (ha : arrow_location_between a b) :
  a = 42.3 :=
sorry

end approximate_reading_l27_27309


namespace total_number_of_trees_l27_27711

variable {T : ℕ} -- Define T as a natural number (total number of trees)
variable (h1 : 70 / 100 * T + 105 = T) -- Indicates 30% of T is 105

theorem total_number_of_trees (h1 : 70 / 100 * T + 105 = T) : T = 350 :=
by
sorry

end total_number_of_trees_l27_27711


namespace divisor_is_four_l27_27614

theorem divisor_is_four (n d k l : ℤ) (hn : n % d = 3) (h2n : (2 * n) % d = 2) (hd : d > 3) : d = 4 :=
by
  sorry

end divisor_is_four_l27_27614


namespace find_higher_interest_rate_l27_27521

-- Definitions and conditions based on the problem
def total_investment : ℕ := 4725
def higher_rate_investment : ℕ := 1925
def lower_rate_investment : ℕ := total_investment - higher_rate_investment
def lower_rate : ℝ := 0.08
def higher_to_lower_interest_ratio : ℝ := 2

-- The main theorem to prove the higher interest rate
theorem find_higher_interest_rate (r : ℝ) (h1 : higher_rate_investment = 1925) (h2 : lower_rate_investment = 2800) :
  1925 * r = 2 * (2800 * 0.08) → r = 448 / 1925 :=
sorry

end find_higher_interest_rate_l27_27521


namespace percentage_increase_third_year_l27_27189

theorem percentage_increase_third_year
  (initial_price : ℝ)
  (price_2007 : ℝ := initial_price * (1 + 20 / 100))
  (price_2008 : ℝ := price_2007 * (1 - 25 / 100))
  (price_end_third_year : ℝ := initial_price * (108 / 100)) :
  ((price_end_third_year - price_2008) / price_2008) * 100 = 20 :=
by
  sorry

end percentage_increase_third_year_l27_27189


namespace each_shopper_will_receive_amount_l27_27825

/-- Definitions of the given conditions -/
def isabella_has_more_than_sam : ℕ := 45
def isabella_has_more_than_giselle : ℕ := 15
def giselle_money : ℕ := 120

/-- Calculation based on the provided conditions -/
def isabella_money : ℕ := giselle_money + isabella_has_more_than_giselle
def sam_money : ℕ := isabella_money - isabella_has_more_than_sam
def total_money : ℕ := isabella_money + sam_money + giselle_money

/-- The total amount each shopper will receive when the donation is shared equally -/
def money_each_shopper_receives : ℕ := total_money / 3

/-- Main theorem to prove the statement derived from the problem -/
theorem each_shopper_will_receive_amount :
  money_each_shopper_receives = 115 := by
  sorry

end each_shopper_will_receive_amount_l27_27825


namespace constant_term_eq_160_l27_27511

-- Define the binomial coefficients and the binomial theorem
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term of (2x + 1/x)^6 expansion
def general_term_expansion (r : ℕ) : ℤ :=
  2^(6 - r) * binom 6 r

-- Define the proof statement for the required constant term
theorem constant_term_eq_160 : general_term_expansion 3 = 160 := 
by
  sorry

end constant_term_eq_160_l27_27511


namespace problem_solution_l27_27787

theorem problem_solution (a : ℝ) : 
  ( ∀ x : ℝ, (ax - 1) * (x + 1) < 0 ↔ (x ∈ Set.Iio (-1) ∨ x ∈ Set.Ioi (-1 / 2)) ) →
  a = -2 :=
by
  sorry

end problem_solution_l27_27787


namespace camels_in_caravan_l27_27704

theorem camels_in_caravan : 
  ∃ (C : ℕ), 
  (60 + 35 + 10 + C) * 1 + 60 * 2 + 35 * 4 + 10 * 2 + 4 * C - (60 + 35 + 10 + C) = 193 ∧ 
  C = 6 :=
by
  sorry

end camels_in_caravan_l27_27704


namespace meaningful_sqrt_range_l27_27803

theorem meaningful_sqrt_range (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 :=
by
  sorry

end meaningful_sqrt_range_l27_27803


namespace find_a2_l27_27074

theorem find_a2 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * (a n - 1))
  (h2 : S 1 = a 1)
  (h3 : S 2 = a 1 + a 2) :
  a 2 = 4 :=
sorry

end find_a2_l27_27074


namespace find_x_l27_27853

theorem find_x (x : ℝ) (h : 2500 - 1002 / x = 2450) : x = 20.04 :=
by 
  sorry

end find_x_l27_27853


namespace actual_distance_traveled_l27_27211

theorem actual_distance_traveled 
  (D : ℝ)
  (h1 : ∃ (D : ℝ), D/12 = (D + 36)/20)
  : D = 54 :=
sorry

end actual_distance_traveled_l27_27211


namespace fencing_required_l27_27772

theorem fencing_required (length width area : ℕ) (length_eq : length = 30) (area_eq : area = 810) 
  (field_area : length * width = area) : 2 * length + width = 87 := 
by
  sorry

end fencing_required_l27_27772


namespace alex_correct_percentage_l27_27481

theorem alex_correct_percentage 
  (score_quiz : ℤ) (problems_quiz : ℤ)
  (score_test : ℤ) (problems_test : ℤ)
  (score_exam : ℤ) (problems_exam : ℤ)
  (h1 : score_quiz = 75) (h2 : problems_quiz = 30)
  (h3 : score_test = 85) (h4 : problems_test = 50)
  (h5 : score_exam = 80) (h6 : problems_exam = 20) :
  (75 * 30 + 85 * 50 + 80 * 20) / (30 + 50 + 20) = 81 := 
sorry

end alex_correct_percentage_l27_27481


namespace increasing_a_n_not_increasing_n_a_n_increasing_a_n_over_n_not_increasing_a_n_sq_l27_27919

def a_n (n : ℕ) : ℤ := 2 * n - 8

theorem increasing_a_n : ∀ n : ℕ, a_n (n + 1) > a_n n := 
by 
-- Assuming n >= 0
intro n
dsimp [a_n]
sorry

def n_a_n (n : ℕ) : ℤ := n * (2 * n - 8)

theorem not_increasing_n_a_n : ∀ n : ℕ, n > 0 → n_a_n (n + 1) ≤ n_a_n n :=
by
-- Assuming n > 0
intro n hn
dsimp [n_a_n]
sorry

def a_n_over_n (n : ℕ) : ℚ := (2 * n - 8 : ℚ) / n

theorem increasing_a_n_over_n : ∀ n > 0, a_n_over_n (n + 1) > a_n_over_n n :=
by 
-- Assuming n > 0
intro n hn
dsimp [a_n_over_n]
sorry

def a_n_sq (n : ℕ) : ℤ := (2 * n - 8) * (2 * n - 8)

theorem not_increasing_a_n_sq : ∀ n : ℕ, a_n_sq (n + 1) ≤ a_n_sq n :=
by
-- Assuming n >= 0
intro n
dsimp [a_n_sq]
sorry

end increasing_a_n_not_increasing_n_a_n_increasing_a_n_over_n_not_increasing_a_n_sq_l27_27919


namespace decreasing_interval_l27_27184

noncomputable def y (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 15 * x^2 + 36 * x - 24

def has_extremum_at (a : ℝ) (x_ext : ℝ) : Prop :=
  deriv (y a) x_ext = 0

theorem decreasing_interval (a : ℝ) (h_extremum_at : has_extremum_at a 3) :
  a = 2 → ∀ x, (2 < x ∧ x < 3) → deriv (y a) x < 0 :=
sorry

end decreasing_interval_l27_27184


namespace smallest_possible_AC_l27_27275

-- Constants and assumptions
variables (AC CD : ℕ)
def BD_squared : ℕ := 68

-- Prime number constraint for CD
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Given facts
axiom eq_ab_ac (AB : ℕ) : AB = AC
axiom perp_bd_ac (BD AC : ℕ) : BD^2 = BD_squared
axiom int_ac_cd : AC = (CD^2 + BD_squared) / (2 * CD)

theorem smallest_possible_AC :
  ∃ AC : ℕ, (∃ CD : ℕ, is_prime CD ∧ CD < 10 ∧ AC = (CD^2 + BD_squared) / (2 * CD)) ∧ AC = 18 :=
by
  sorry

end smallest_possible_AC_l27_27275


namespace factor_as_complete_square_l27_27673

theorem factor_as_complete_square (k : ℝ) : (∃ a : ℝ, x^2 + k*x + 9 = (x + a)^2) ↔ k = 6 ∨ k = -6 := 
sorry

end factor_as_complete_square_l27_27673


namespace geometric_sequence_a5_l27_27884

variable (a : ℕ → ℝ) (q : ℝ)

axiom pos_terms : ∀ n, a n > 0

axiom a1a3_eq : a 1 * a 3 = 16
axiom a3a4_eq : a 3 + a 4 = 24

theorem geometric_sequence_a5 :
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) → a 5 = 32 :=
by
  sorry

end geometric_sequence_a5_l27_27884


namespace fraction_of_largest_jar_filled_l27_27424

theorem fraction_of_largest_jar_filled
  (C1 C2 C3 : ℝ)
  (h1 : C1 < C2)
  (h2 : C2 < C3)
  (h3 : C1 / 6 = C2 / 5)
  (h4 : C2 / 5 = C3 / 7) :
  (C1 / 6 + C2 / 5) / C3 = 2 / 7 := sorry

end fraction_of_largest_jar_filled_l27_27424


namespace number_of_periods_l27_27846

-- Definitions based on conditions
def students : ℕ := 32
def time_per_student : ℕ := 5
def period_duration : ℕ := 40

-- Theorem stating the equivalent proof problem
theorem number_of_periods :
  (students * time_per_student) / period_duration = 4 :=
sorry

end number_of_periods_l27_27846


namespace largest_eight_digit_with_all_even_digits_l27_27113

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l27_27113


namespace largest_non_expressible_number_l27_27009

theorem largest_non_expressible_number :
  ∀ (x y z : ℕ), 15 * x + 18 * y + 20 * z ≠ 97 :=
by sorry

end largest_non_expressible_number_l27_27009


namespace quadratic_common_root_l27_27400

theorem quadratic_common_root (b : ℤ) :
  (∃ x, 2 * x^2 + (3 * b - 1) * x - 3 = 0 ∧ 6 * x^2 - (2 * b - 3) * x - 1 = 0) ↔ b = 2 := 
sorry

end quadratic_common_root_l27_27400


namespace calculate_z_l27_27092

-- Given conditions
def equally_spaced : Prop := true -- assume equally spaced markings do exist
def total_distance : ℕ := 35
def number_of_steps : ℕ := 7
def step_length : ℕ := total_distance / number_of_steps
def starting_point : ℕ := 10
def steps_forward : ℕ := 4

-- Theorem to prove
theorem calculate_z (h1 : equally_spaced)
(h2 : step_length = 5)
: starting_point + (steps_forward * step_length) = 30 :=
by sorry

end calculate_z_l27_27092


namespace values_of_k_real_equal_roots_l27_27985

theorem values_of_k_real_equal_roots (k : ℝ) :
  (∀ x : ℝ, 3 * x^2 - (k + 2) * x + 12 = 0 → x * x = 0) ↔ (k = 10 ∨ k = -14) :=
by
  sorry

end values_of_k_real_equal_roots_l27_27985


namespace caricatures_sold_on_sunday_l27_27476

def caricature_price : ℕ := 20
def saturday_sales : ℕ := 24
def total_earnings : ℕ := 800

theorem caricatures_sold_on_sunday :
  (total_earnings - saturday_sales * caricature_price) / caricature_price = 16 :=
by
  sorry  -- Proof goes here

end caricatures_sold_on_sunday_l27_27476


namespace question_b_l27_27618

theorem question_b (a b c : ℝ) (h : c ≠ 0) (h_eq : a / c = b / c) : a = b := 
by
  sorry

end question_b_l27_27618


namespace geometric_sequence_a4_l27_27383

-- Define the terms of the geometric sequence
variable {a : ℕ → ℝ}

-- Define the conditions of the problem
def a2_cond : Prop := a 2 = 2
def a6_cond : Prop := a 6 = 32

-- Define the theorem we want to prove
theorem geometric_sequence_a4 (a2_cond : a 2 = 2) (a6_cond : a 6 = 32) : a 4 = 8 := by
  sorry

end geometric_sequence_a4_l27_27383


namespace total_people_after_one_hour_l27_27349

variable (x y Z : ℕ)

def ferris_wheel_line_initial := 50
def bumper_cars_line_initial := 50
def roller_coaster_line_initial := 50

def ferris_wheel_line_after_half_hour := ferris_wheel_line_initial - x
def bumper_cars_line_after_half_hour := bumper_cars_line_initial + y

axiom Z_eq : Z = ferris_wheel_line_after_half_hour + bumper_cars_line_after_half_hour

theorem total_people_after_one_hour : (Z = (50 - x) + (50 + y)) -> (Z + 100) = ((50 - x) + (50 + y) + 100) :=
by {
  sorry
}

end total_people_after_one_hour_l27_27349


namespace inequality_not_true_l27_27754

theorem inequality_not_true (a b : ℝ) (h : a > b) : (a / (-2)) ≤ (b / (-2)) :=
sorry

end inequality_not_true_l27_27754


namespace coin_count_l27_27346

theorem coin_count (x : ℝ) (h₁ : x + 0.50 * x + 0.25 * x = 35) : x = 20 :=
by
  sorry

end coin_count_l27_27346


namespace remainder_17_pow_63_mod_7_l27_27620

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l27_27620


namespace skipping_times_eq_l27_27647

theorem skipping_times_eq (x : ℝ) (h : x > 0) :
  180 / x = 240 / (x + 5) :=
sorry

end skipping_times_eq_l27_27647


namespace asbestos_tiles_width_l27_27329

theorem asbestos_tiles_width (n : ℕ) (h : 0 < n) :
  let width_per_tile := 60
  let overlap := 10
  let effective_width := width_per_tile - overlap
  width_per_tile + (n - 1) * effective_width = 50 * n + 10 := by
sorry

end asbestos_tiles_width_l27_27329


namespace find_initial_average_price_l27_27455

noncomputable def average_initial_price (P : ℚ) : Prop :=
  let total_cost_of_4_cans := 120
  let total_cost_of_returned_cans := 99
  let total_cost_of_6_cans := 6 * P
  total_cost_of_6_cans - total_cost_of_4_cans = total_cost_of_returned_cans

theorem find_initial_average_price (P : ℚ) :
    average_initial_price P → 
    P = 36.5 := sorry

end find_initial_average_price_l27_27455


namespace youngest_child_is_five_l27_27104

-- Define the set of prime numbers
def is_prime (n: ℕ) := n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the ages of the children
def youngest_child_age (x: ℕ) : Prop :=
  is_prime x ∧
  is_prime (x + 2) ∧
  is_prime (x + 6) ∧
  is_prime (x + 8) ∧
  is_prime (x + 12) ∧
  is_prime (x + 14)

-- The main theorem stating the age of the youngest child
theorem youngest_child_is_five : ∃ x: ℕ, youngest_child_age x ∧ x = 5 :=
  sorry

end youngest_child_is_five_l27_27104


namespace divisors_not_multiples_of_14_l27_27764

theorem divisors_not_multiples_of_14 (m : ℕ)
  (h1 : ∃ k : ℕ, m = 2 * k ∧ (k : ℕ) * k = m / 2)  
  (h2 : ∃ k : ℕ, m = 3 * k ∧ (k : ℕ) * k * k = m / 3)  
  (h3 : ∃ k : ℕ, m = 7 * k ∧ (k : ℕ) ^ 7 = m / 7) : 
  let total_divisors := (6 + 1) * (10 + 1) * (7 + 1)
  let divisors_divisible_by_14 := (5 + 1) * (10 + 1) * (6 + 1)
  total_divisors - divisors_divisible_by_14 = 154 :=
by
  sorry

end divisors_not_multiples_of_14_l27_27764


namespace roots_seventh_sum_l27_27674

noncomputable def x1 := (-3 + Real.sqrt 5) / 2
noncomputable def x2 := (-3 - Real.sqrt 5) / 2

theorem roots_seventh_sum :
  (x1 ^ 7 + x2 ^ 7) = -843 :=
by
  -- Given condition: x1 and x2 are roots of x^2 + 3x + 1 = 0
  have h1 : x1^2 + 3 * x1 + 1 = 0 := by sorry
  have h2 : x2^2 + 3 * x2 + 1 = 0 := by sorry
  -- Proof goes here
  sorry

end roots_seventh_sum_l27_27674


namespace min_value_of_quadratic_l27_27243

theorem min_value_of_quadratic : ∃ x : ℝ, 7 * x^2 - 28 * x + 1702 = 1674 ∧ ∀ y : ℝ, 7 * y^2 - 28 * y + 1702 ≥ 1674 :=
by
  sorry

end min_value_of_quadratic_l27_27243


namespace original_price_sarees_l27_27583

theorem original_price_sarees (P : ℝ) (h : 0.80 * P * 0.85 = 231.2) : P = 340 := 
by sorry

end original_price_sarees_l27_27583


namespace drums_per_day_l27_27164

theorem drums_per_day (total_drums : Nat) (days : Nat) (total_drums_eq : total_drums = 6264) (days_eq : days = 58) :
  total_drums / days = 108 :=
by
  sorry

end drums_per_day_l27_27164


namespace problem1_problem2_l27_27656

def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

theorem problem1 (x : ℝ) : f x (-1) ≤ 2 ↔ -1 / 2 ≤ x ∧ x ≤ 1 / 2 :=
by sorry

theorem problem2 (a : ℝ) :
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 1, f x a ≤ |2 * x + 1|) → (0 ≤ a ∧ a ≤ 3) :=
by sorry

end problem1_problem2_l27_27656


namespace inequality_solution_range_l27_27258

theorem inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → x ^ 2 + a * x + 4 < 0) ↔ a < -4 :=
by 
  sorry

end inequality_solution_range_l27_27258


namespace p_q_work_l27_27547

theorem p_q_work (p_rate q_rate : ℝ) (h1: 1 / p_rate + 1 / q_rate = 1 / 6) (h2: p_rate = 15) : q_rate = 10 :=
by
  sorry

end p_q_work_l27_27547


namespace original_selling_price_l27_27981

variable (P : ℝ)

def SP1 := 1.10 * P
def P_new := 0.90 * P
def SP2 := 1.17 * P
def price_diff := SP2 - SP1

theorem original_selling_price : price_diff = 49 → SP1 = 770 :=
by
  sorry

end original_selling_price_l27_27981


namespace average_growth_rate_le_half_sum_l27_27536

variable (p q x : ℝ)

theorem average_growth_rate_le_half_sum : 
  (1 + p) * (1 + q) = (1 + x) ^ 2 → x ≤ (p + q) / 2 :=
by
  intro h
  sorry

end average_growth_rate_le_half_sum_l27_27536


namespace fractions_sum_simplified_l27_27410

noncomputable def frac12over15 : ℚ := 12 / 15
noncomputable def frac7over9 : ℚ := 7 / 9
noncomputable def frac1and1over6 : ℚ := 1 + 1 / 6

theorem fractions_sum_simplified :
  frac12over15 + frac7over9 + frac1and1over6 = 247 / 90 :=
by
  -- This step will be left as a proof to complete.
  sorry

end fractions_sum_simplified_l27_27410


namespace sector_area_l27_27710

theorem sector_area (α : ℝ) (l : ℝ) (S : ℝ) (hα : α = 60 * Real.pi / 180) (hl : l = 6 * Real.pi) : S = 54 * Real.pi :=
sorry

end sector_area_l27_27710


namespace dealer_profit_percentage_l27_27850

-- Define the conditions
def cost_price_kg : ℕ := 1000
def given_weight_kg : ℕ := 575

-- Define the weight saved by the dealer
def weight_saved : ℕ := cost_price_kg - given_weight_kg

-- Define the profit percentage formula
def profit_percentage : ℕ → ℕ → ℚ := λ saved total_weight => (saved : ℚ) / (total_weight : ℚ) * 100

-- The main theorem statement
theorem dealer_profit_percentage : profit_percentage weight_saved cost_price_kg = 42.5 :=
by
  sorry

end dealer_profit_percentage_l27_27850


namespace arithmetic_expression_eval_l27_27133

theorem arithmetic_expression_eval :
  -1 ^ 4 + (4 - ((3 / 8 + 1 / 6 - 3 / 4) * 24)) / 5 = 0.8 := by
  sorry

end arithmetic_expression_eval_l27_27133


namespace eval_sum_l27_27461

theorem eval_sum : 333 + 33 + 3 = 369 :=
by
  sorry

end eval_sum_l27_27461


namespace sum_of_b_for_quadratic_has_one_solution_l27_27151

theorem sum_of_b_for_quadratic_has_one_solution :
  (∀ x : ℝ, 3 * x^2 + (b+6) * x + 1 = 0 → 
    ∀ Δ : ℝ, Δ = (b + 6)^2 - 4 * 3 * 1 → 
    Δ = 0 → 
    b = -6 + 2 * Real.sqrt 3 ∨ b = -6 - 2 * Real.sqrt 3) → 
  (-6 + 2 * Real.sqrt 3 + -6 - 2 * Real.sqrt 3 = -12) := 
by
  sorry

end sum_of_b_for_quadratic_has_one_solution_l27_27151


namespace polynomial_independent_of_m_l27_27307

theorem polynomial_independent_of_m (m : ℝ) (x : ℝ) (h : 6 * x^2 + (1 - 2 * m) * x + 7 * m = 6 * x^2 + x) : 
  x = 7 / 2 :=
by
  sorry

end polynomial_independent_of_m_l27_27307


namespace map_at_three_l27_27808

variable (A B : Type)
variable (a : ℝ)
variable (f : ℝ → ℝ)
variable (h_map : ∀ x : ℝ, f x = a * x - 1)
variable (h_cond : f 2 = 3)

theorem map_at_three : f 3 = 5 := by
  sorry

end map_at_three_l27_27808


namespace half_of_1_point_6_times_10_pow_6_l27_27807

theorem half_of_1_point_6_times_10_pow_6 : (1.6 * 10^6) / 2 = 8 * 10^5 :=
by
  sorry

end half_of_1_point_6_times_10_pow_6_l27_27807


namespace divisor_proof_l27_27010

def original_number : ℕ := 123456789101112131415161718192021222324252627282930313233343536373839404142434481

def remainder : ℕ := 36

theorem divisor_proof (D : ℕ) (Q : ℕ) (h : original_number = D * Q + remainder) : original_number % D = remainder :=
by 
  sorry

end divisor_proof_l27_27010


namespace tony_income_l27_27422

-- Definitions for the given conditions
def investment : ℝ := 3200
def purchase_price : ℝ := 85
def dividend : ℝ := 6.640625

-- Theorem stating Tony's income based on the conditions
theorem tony_income : (investment / purchase_price) * dividend = 250 :=
by
  sorry

end tony_income_l27_27422


namespace david_profit_l27_27308

theorem david_profit (weight : ℕ) (cost sell_price : ℝ) (h_weight : weight = 50) (h_cost : cost = 50) (h_sell_price : sell_price = 1.20) : 
  sell_price * weight - cost = 10 :=
by sorry

end david_profit_l27_27308


namespace maximum_area_of_triangle_OAB_l27_27221

noncomputable def maximum_area_triangle (a b : ℝ) : ℝ :=
  if 2 * a + b = 5 ∧ a > 0 ∧ b > 0 then (1 / 2) * a * b else 0

theorem maximum_area_of_triangle_OAB : 
  (∀ (a b : ℝ), 2 * a + b = 5 ∧ a > 0 ∧ b > 0 → (1 / 2) * a * b ≤ 25 / 16) :=
by
  sorry

end maximum_area_of_triangle_OAB_l27_27221


namespace christopher_sword_length_l27_27373

variable (C J U : ℤ)

def jameson_sword (C : ℤ) : ℤ := 2 * C + 3
def june_sword (J : ℤ) : ℤ := J + 5
def june_sword_christopher (C : ℤ) : ℤ := C + 23

theorem christopher_sword_length (h1 : J = jameson_sword C)
                                (h2 : U = june_sword J)
                                (h3 : U = june_sword_christopher C) :
                                C = 15 :=
by
  sorry

end christopher_sword_length_l27_27373


namespace rain_on_tuesday_l27_27096

theorem rain_on_tuesday 
  (rain_monday : ℝ)
  (rain_less : ℝ) 
  (h1 : rain_monday = 0.9) 
  (h2 : rain_less = 0.7) : 
  (rain_monday - rain_less) = 0.2 :=
by
  sorry

end rain_on_tuesday_l27_27096


namespace seashells_left_l27_27720

theorem seashells_left (initial_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) :
  initial_seashells = 75 → given_seashells = 18 → remaining_seashells = initial_seashells - given_seashells → remaining_seashells = 57 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end seashells_left_l27_27720


namespace max_least_integer_l27_27789

theorem max_least_integer (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 2160) (h_order : x ≤ y ∧ y ≤ z) : x ≤ 10 :=
by
  sorry

end max_least_integer_l27_27789


namespace problem_statement_l27_27641

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a₁ d : ℤ, ∀ n : ℕ, a n = a₁ + n * d

theorem problem_statement :
  ∃ a : ℕ → ℤ, is_arithmetic_sequence a ∧
  a 2 = 7 ∧
  a 4 + a 6 = 26 ∧
  (∀ n : ℕ, a (n + 1) = 2 * n + 1) ∧
  ∃ S : ℕ → ℤ, (S n = n^2 + 2 * n) ∧
  ∃ b : ℕ → ℚ, (∀ n : ℕ, b n = 1 / (a n ^ 2 - 1)) ∧
  ∃ T : ℕ → ℚ, (T n = (n / 4) * (1 / (n + 1))) :=
sorry

end problem_statement_l27_27641


namespace golden_section_BC_length_l27_27376

-- Definition of a golden section point
def is_golden_section_point (A B C : ℝ) : Prop :=
  ∃ (φ : ℝ), φ = (1 + Real.sqrt 5) / 2 ∧ B = φ * C

-- The given problem translated to Lean
theorem golden_section_BC_length (A B C : ℝ) (h1 : is_golden_section_point A B C) (h2 : B - A = 6) : 
  C - B = 3 * Real.sqrt 5 - 3 ∨ C - B = 9 - 3 * Real.sqrt 5 :=
by
  sorry

end golden_section_BC_length_l27_27376


namespace triangle_area_and_coordinates_l27_27182

noncomputable def positive_diff_of_coordinates (A B C R S : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (xr, yr) := R
  let (xs, ys) := S
  if xr = xs then abs (xr - (10 - (x3 - xr)))
  else 0 -- Should never be this case if conditions are properly followed

theorem triangle_area_and_coordinates
  (A B C R S : ℝ × ℝ)
  (h_A : A = (0, 10))
  (h_B : B = (4, 0))
  (h_C : C = (10, 0))
  (h_vertical : R.fst = S.fst)
  (h_intersect_AC : R.snd = -(R.fst - 10))
  (h_intersect_BC : S.snd = 0 ∧ S.fst = 10 - (C.fst - R.fst))
  (h_area : 1/2 * ((R.fst - C.fst) * (R.snd - C.snd)) = 15) :
  positive_diff_of_coordinates A B C R S = 2 * Real.sqrt 30 - 10 := sorry

end triangle_area_and_coordinates_l27_27182


namespace distance_from_C_to_A_is_8_l27_27463

-- Define points A, B, and C as real numbers representing positions
def A : ℝ := 0  -- Starting point
def B : ℝ := A - 15  -- 15 meters west from A
def C : ℝ := B + 23  -- 23 meters east from B

-- Prove that the distance from point C to point A is 8 meters
theorem distance_from_C_to_A_is_8 : abs (C - A) = 8 :=
by
  sorry

end distance_from_C_to_A_is_8_l27_27463


namespace problem1_problem2_l27_27655

-- Define the triangle and the condition a + 2a * cos B = c
variable {A B C : ℝ} (a b c : ℝ)
variable (cos_B : ℝ) -- cosine of angle B

-- Condition: a + 2a * cos B = c
variable (h1 : a + 2 * a * cos_B = c)

-- (I) Prove B = 2A
theorem problem1 (h1 : a + 2 * a * cos_B = c) : B = 2 * A :=
sorry

-- Define the acute triangle condition
variable (Acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)

-- Given: c = 2
variable (h2 : c = 2)

-- (II) Determine the range for a if the triangle is acute and c = 2
theorem problem2 (h1 : a + 2 * a * cos_B = 2) (Acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) : 1 < a ∧ a < 2 :=
sorry

end problem1_problem2_l27_27655


namespace three_over_x_solution_l27_27706

theorem three_over_x_solution (x : ℝ) (h : 1 - 9 / x + 9 / (x^2) = 0) :
  3 / x = (3 - Real.sqrt 5) / 2 ∨ 3 / x = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end three_over_x_solution_l27_27706


namespace no_symmetric_a_l27_27356

noncomputable def f (a x : ℝ) : ℝ := Real.log (((x + 1) / (x - 1)) * (x - 1) * (a - x))

theorem no_symmetric_a (a : ℝ) (h_a : 1 < a) : ¬ ∃ c : ℝ, ∀ d : ℝ, 1 < c - d ∧ c - d < a ∧ 1 < c + d ∧ c + d < a → f a (c - d) = f a (c + d) :=
sorry

end no_symmetric_a_l27_27356


namespace number_of_keepers_l27_27585

theorem number_of_keepers (k : ℕ)
  (hens : ℕ := 50)
  (goats : ℕ := 45)
  (camels : ℕ := 8)
  (hen_feet : ℕ := 2)
  (goat_feet : ℕ := 4)
  (camel_feet : ℕ := 4)
  (keeper_feet : ℕ := 2)
  (feet_more_than_heads : ℕ := 224)
  (total_heads : ℕ := hens + goats + camels + k)
  (total_feet : ℕ := (hens * hen_feet) + (goats * goat_feet) + (camels * camel_feet) + (k * keeper_feet)):
  total_feet = total_heads + feet_more_than_heads → k = 15 :=
by
  sorry

end number_of_keepers_l27_27585


namespace total_ants_correct_l27_27685

-- Define the conditions
def park_width_ft : ℕ := 450
def park_length_ft : ℕ := 600
def ants_per_sq_inch_first_half : ℕ := 2
def ants_per_sq_inch_second_half : ℕ := 4

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Convert width and length from feet to inches
def park_width_inch : ℕ := park_width_ft * feet_to_inches
def park_length_inch : ℕ := park_length_ft * feet_to_inches

-- Define the area of each half of the park in square inches
def half_length_inch : ℕ := park_length_inch / 2
def area_first_half_sq_inch : ℕ := park_width_inch * half_length_inch
def area_second_half_sq_inch : ℕ := park_width_inch * half_length_inch

-- Define the number of ants in each half
def ants_first_half : ℕ := ants_per_sq_inch_first_half * area_first_half_sq_inch
def ants_second_half : ℕ := ants_per_sq_inch_second_half * area_second_half_sq_inch

-- Define the total number of ants
def total_ants : ℕ := ants_first_half + ants_second_half

-- The proof problem
theorem total_ants_correct : total_ants = 116640000 := by
  sorry

end total_ants_correct_l27_27685


namespace bugs_meeting_time_l27_27187

/-- Two circles with radii 7 inches and 3 inches are tangent at a point P. 
Two bugs start crawling at the same time from point P, one along the larger circle 
at 4π inches per minute, and the other along the smaller circle at 3π inches per minute. 
Prove they will meet again after 14 minutes and determine how far each has traveled.

The bug on the larger circle will have traveled 28π inches.
The bug on the smaller circle will have traveled 42π inches.
-/
theorem bugs_meeting_time
  (r₁ r₂ : ℝ) (v₁ v₂ : ℝ)
  (h₁ : r₁ = 7) (h₂ : r₂ = 3) 
  (h₃ : v₁ = 4 * Real.pi) (h₄ : v₂ = 3 * Real.pi) :
  ∃ t d₁ d₂, t = 14 ∧ d₁ = 28 * Real.pi ∧ d₂ = 42 * Real.pi := by
  sorry

end bugs_meeting_time_l27_27187


namespace acute_triangle_condition_l27_27330

theorem acute_triangle_condition (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (|a^2 - b^2| < c^2 ∧ c^2 < a^2 + b^2) ↔ (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) :=
sorry

end acute_triangle_condition_l27_27330


namespace scale_of_diagram_l27_27077

-- Definitions for the given conditions
def length_miniature_component_mm : ℕ := 4
def length_diagram_cm : ℕ := 8
def length_diagram_mm : ℕ := 80  -- Converted length from cm to mm

-- The problem statement
theorem scale_of_diagram :
  (length_diagram_mm : ℕ) / (length_miniature_component_mm : ℕ) = 20 :=
by
  have conversion : length_diagram_mm = length_diagram_cm * 10 := by sorry
  -- conversion states the formula for converting cm to mm
  have ratio : length_diagram_mm / length_miniature_component_mm = 80 / 4 := by sorry
  -- ratio states the initial computed ratio
  exact sorry

end scale_of_diagram_l27_27077


namespace expression_in_terms_of_p_and_q_l27_27684

theorem expression_in_terms_of_p_and_q (x : ℝ) :
  let p := (1 - Real.cos x) * (1 + Real.sin x)
  let q := (1 + Real.cos x) * (1 - Real.sin x)
  (Real.cos x ^ 2 - Real.cos x ^ 4 - Real.sin (2 * x) + 2) = p * q - (p + q) :=
by
  sorry

end expression_in_terms_of_p_and_q_l27_27684


namespace magnitude_squared_l27_27314

-- Let z be the complex number 3 + 4i
def z : ℂ := 3 + 4 * Complex.I

-- Prove that the magnitude of z squared equals 25
theorem magnitude_squared : Complex.abs z ^ 2 = 25 := by
  -- The term "by" starts the proof block, and "sorry" allows us to skip the proof details.
  sorry

end magnitude_squared_l27_27314


namespace jackson_entertainment_expense_l27_27245

noncomputable def total_spent_on_entertainment_computer_game_original_price : ℝ :=
  66 / 0.85

noncomputable def movie_ticket_price_with_tax : ℝ :=
  12 * 1.10

noncomputable def total_movie_tickets_cost : ℝ :=
  3 * movie_ticket_price_with_tax

noncomputable def total_snacks_and_transportation_cost : ℝ :=
  7 + 5

noncomputable def total_spent : ℝ :=
  66 + total_movie_tickets_cost + total_snacks_and_transportation_cost

theorem jackson_entertainment_expense :
  total_spent = 117.60 :=
by
  sorry

end jackson_entertainment_expense_l27_27245


namespace change_back_l27_27490

theorem change_back (price_laptop : ℤ) (price_smartphone : ℤ) (qty_laptops : ℤ) (qty_smartphones : ℤ) (initial_amount : ℤ) (total_cost : ℤ) (change : ℤ) :
  price_laptop = 600 →
  price_smartphone = 400 →
  qty_laptops = 2 →
  qty_smartphones = 4 →
  initial_amount = 3000 →
  total_cost = (price_laptop * qty_laptops) + (price_smartphone * qty_smartphones) →
  change = initial_amount - total_cost →
  change = 200 := by
  sorry

end change_back_l27_27490


namespace problem1_problem2_problem3_problem4_l27_27489

-- Problem 1: 27 - 16 + (-7) - 18 = -14
theorem problem1 : 27 - 16 + (-7) - 18 = -14 := 
by 
  sorry

-- Problem 2: (-6) * (-3/4) / (-3/2) = -3
theorem problem2 : (-6) * (-3/4) / (-3/2) = -3 := 
by
  sorry

-- Problem 3: (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81
theorem problem3 : (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81 := 
by
  sorry

-- Problem 4: -2^4 + 3 * (-1)^4 - (-2)^3 = -5
theorem problem4 : -2^4 + 3 * (-1)^4 - (-2)^3 = -5 := 
by
  sorry

end problem1_problem2_problem3_problem4_l27_27489


namespace xy_diff_l27_27453

theorem xy_diff {x y : ℝ} (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  sorry

end xy_diff_l27_27453


namespace initial_salt_percentage_l27_27877

theorem initial_salt_percentage (P : ℕ) : 
  let initial_solution := 100 
  let added_salt := 20 
  let final_solution := initial_solution + added_salt 
  (P / 100) * initial_solution + added_salt = (25 / 100) * final_solution → 
  P = 10 := 
by
  sorry

end initial_salt_percentage_l27_27877


namespace number_of_bad_cards_l27_27331

-- Define the initial conditions
def janessa_initial_cards : ℕ := 4
def father_given_cards : ℕ := 13
def ordered_cards : ℕ := 36
def cards_given_to_dexter : ℕ := 29
def cards_kept_for_herself : ℕ := 20

-- Define the total cards and cards in bad shape calculation
theorem number_of_bad_cards : 
  let total_initial_cards := janessa_initial_cards + father_given_cards;
  let total_cards := total_initial_cards + ordered_cards;
  let total_distributed_cards := cards_given_to_dexter + cards_kept_for_herself;
  total_cards - total_distributed_cards = 4 :=
by {
  sorry
}

end number_of_bad_cards_l27_27331


namespace ratio_of_sum_to_first_term_l27_27556

-- Definitions and conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 0 * (1 - (2 ^ n)) / (1 - 2)

-- Main statement to be proven
theorem ratio_of_sum_to_first_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geo : geometric_sequence a 2) (h_sum : sum_of_first_n_terms a S) :
  S 3 / a 0 = 7 :=
sorry

end ratio_of_sum_to_first_term_l27_27556


namespace bamboo_break_height_l27_27028

-- Conditions provided in the problem
def original_height : ℝ := 20  -- 20 chi
def distance_tip_to_root : ℝ := 6  -- 6 chi

-- Function to check if the height of the break satisfies the equation
def equationHolds (x : ℝ) : Prop :=
  (original_height - x) ^ 2 - x ^ 2 = distance_tip_to_root ^ 2

-- Main statement to prove the height of the break is 9.1 chi
theorem bamboo_break_height : equationHolds 9.1 :=
by
  sorry

end bamboo_break_height_l27_27028


namespace merchant_profit_percentage_l27_27522

noncomputable def cost_price_of_one_article (C : ℝ) : Prop := ∃ S : ℝ, 20 * C = 16 * S

theorem merchant_profit_percentage (C S : ℝ) (h : cost_price_of_one_article C) : 
  100 * ((S - C) / C) = 25 :=
by 
  sorry

end merchant_profit_percentage_l27_27522


namespace sequence_value_a_l27_27572

theorem sequence_value_a (a : ℚ) (a_n : ℕ → ℚ)
  (h1 : a_n 1 = a) (h2 : a_n 2 = a)
  (h3 : ∀ n ≥ 3, a_n n = a_n (n - 1) + a_n (n - 2))
  (h4 : a_n 8 = 34) :
  a = 34 / 21 :=
by sorry

end sequence_value_a_l27_27572


namespace shaded_area_is_correct_l27_27725

theorem shaded_area_is_correct : 
  ∀ (leg_length : ℕ) (total_partitions : ℕ) (shaded_partitions : ℕ) 
    (tri_area : ℕ) (small_tri_area : ℕ) (shaded_area : ℕ), 
  leg_length = 10 → 
  total_partitions = 25 →
  shaded_partitions = 15 →
  tri_area = (1 / 2 * leg_length * leg_length) → 
  small_tri_area = (tri_area / total_partitions) →
  shaded_area = (shaded_partitions * small_tri_area) →
  shaded_area = 30 :=
by
  intros leg_length total_partitions shaded_partitions tri_area small_tri_area shaded_area
  intros h_leg_length h_total_partitions h_shaded_partitions h_tri_area h_small_tri_area h_shaded_area
  sorry

end shaded_area_is_correct_l27_27725


namespace ellipse_shortest_major_axis_l27_27864

theorem ellipse_shortest_major_axis (P : ℝ × ℝ) (a b : ℝ) 
  (ha : a > b) (hb : b > 0) (hP_on_line : P.2 = P.1 + 2)
  (h_foci_hyperbola : ∃ c : ℝ, c = 1 ∧ a^2 - b^2 = c^2) :
  (∃ a b : ℝ, a^2 = 5 ∧ b^2 = 4 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1)) :=
sorry

end ellipse_shortest_major_axis_l27_27864


namespace billy_trays_l27_27369

def trays_needed (total_ice_cubes : ℕ) (ice_cubes_per_tray : ℕ) : ℕ :=
  total_ice_cubes / ice_cubes_per_tray

theorem billy_trays (total_ice_cubes ice_cubes_per_tray : ℕ) (h1 : total_ice_cubes = 72) (h2 : ice_cubes_per_tray = 9) :
  trays_needed total_ice_cubes ice_cubes_per_tray = 8 :=
by
  sorry

end billy_trays_l27_27369


namespace arithmetic_sequence_next_term_perfect_square_sequence_next_term_l27_27058

theorem arithmetic_sequence_next_term (a : ℕ → ℕ) (n : ℕ) (h₀ : a 0 = 0) (h₁ : ∀ n, a (n + 1) = a n + 3) :
  a 5 = 15 :=
by sorry

theorem perfect_square_sequence_next_term (b : ℕ → ℕ) (k : ℕ) (h₀ : ∀ k, b k = (k + 1) * (k + 1)) :
  b 5 = 36 :=
by sorry

end arithmetic_sequence_next_term_perfect_square_sequence_next_term_l27_27058


namespace find_r_l27_27322

theorem find_r (k r : ℝ) 
  (h1 : 7 = k * 3^r) 
  (h2 : 49 = k * 9^r) : 
  r = Real.log 7 / Real.log 3 :=
by
  sorry

end find_r_l27_27322


namespace apples_per_pie_l27_27634

-- Conditions
def initial_apples : ℕ := 50
def apples_per_teacher_per_child : ℕ := 3
def number_of_teachers : ℕ := 2
def number_of_children : ℕ := 2
def remaining_apples : ℕ := 24

-- Proof goal: the number of apples Jill uses per pie
theorem apples_per_pie : 
  initial_apples 
  - (apples_per_teacher_per_child * number_of_teachers * number_of_children)  - remaining_apples = 14 -> 14 / 2 = 7 := 
by
  sorry

end apples_per_pie_l27_27634


namespace problem_solution_l27_27031

theorem problem_solution (b : ℝ) (i : ℂ) (h : i^2 = -1) (h_cond : (2 - i) * (4 * i) = 4 + b * i) : 
  b = 8 := 
by 
  sorry

end problem_solution_l27_27031


namespace range_of_sum_abs_l27_27777

variable {x y z : ℝ}

theorem range_of_sum_abs : 
  x^2 + y^2 + z = 15 → 
  x + y + z^2 = 27 → 
  xy + yz + zx = 7 → 
  7 ≤ |x + y + z| ∧ |x + y + z| ≤ 8 := by
  sorry

end range_of_sum_abs_l27_27777


namespace dodgeball_tournament_l27_27943

theorem dodgeball_tournament (N : ℕ) (points : ℕ) :
  points = 1151 →
  (∀ {G : ℕ}, G = N * (N - 1) / 2 →
    (∃ (win_points loss_points tie_points : ℕ), 
      win_points = 15 * (N * (N - 1) / 2 - tie_points) ∧ 
      tie_points = 11 * tie_points ∧ 
      points = win_points + tie_points + loss_points)) → 
  N = 12 :=
by
  intro h_points h_games
  sorry

end dodgeball_tournament_l27_27943


namespace sufficient_condition_l27_27666

theorem sufficient_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + a = 0 → a < 1) ↔ 
  (∀ c : ℝ, x^2 - 2 * x + c = 0 ↔ 4 - 4 * c ≥ 0 ∧ c < 1 → ¬ (∀ d : ℝ, d ≤ 1 → d < 1)) := 
by 
sorry

end sufficient_condition_l27_27666


namespace inequality_proof_l27_27268

theorem inequality_proof
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (ha1 : 0 < a1) (hb1 : 0 < b1) (hc1 : 0 < c1)
  (ha2 : 0 < a2) (hb2 : 0 < b2) (hc2 : 0 < c2)
  (h1: b1^2 ≤ a1 * c1)
  (h2: b2^2 ≤ a2 * c2) :
  (a1 + a2 + 5) * (c1 + c2 + 2) > (b1 + b2 + 3)^2 :=
by
  sorry

end inequality_proof_l27_27268


namespace cos_sum_eq_one_l27_27839

theorem cos_sum_eq_one (α β γ : ℝ) 
  (h1 : α + β + γ = Real.pi) 
  (h2 : Real.tan ((β + γ - α) / 4) + Real.tan ((γ + α - β) / 4) + Real.tan ((α + β - γ) / 4) = 1) :
  Real.cos α + Real.cos β + Real.cos γ = 1 :=
sorry

end cos_sum_eq_one_l27_27839


namespace sum_of_squares_of_coeffs_l27_27820

theorem sum_of_squares_of_coeffs (c1 c2 c3 c4 : ℝ) (h1 : c1 = 3) (h2 : c2 = 6) (h3 : c3 = 15) (h4 : c4 = 6) :
  c1^2 + c2^2 + c3^2 + c4^2 = 306 :=
by
  sorry

end sum_of_squares_of_coeffs_l27_27820


namespace deborah_international_letters_l27_27862

theorem deborah_international_letters (standard_postage : ℝ) 
                                      (additional_charge : ℝ) 
                                      (total_letters : ℕ) 
                                      (total_cost : ℝ) 
                                      (h_standard_postage: standard_postage = 1.08)
                                      (h_additional_charge: additional_charge = 0.14)
                                      (h_total_letters: total_letters = 4)
                                      (h_total_cost: total_cost = 4.60) :
                                      ∃ (x : ℕ), x = 2 :=
by
  sorry

end deborah_international_letters_l27_27862


namespace age_problem_l27_27503

theorem age_problem (a b c d : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : b = 3 * d)
  (h4 : a + b + c + d = 87) : 
  b = 30 :=
by sorry

end age_problem_l27_27503


namespace distance_PF_l27_27987

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the point P on the parabola with x-coordinate 4
def point_on_parabola (y : ℝ) : ℝ × ℝ := (4, y)

-- Prove the distance |PF| for given conditions
theorem distance_PF
  (hP : ∃ y : ℝ, parabola 4 y)
  (hF : focus = (2, 0)) :
  ∃ y : ℝ, y^2 = 8 * 4 ∧ abs (4 - 2) + abs y = 6 := 
by
  sorry

end distance_PF_l27_27987


namespace largest_possible_rational_root_l27_27979

noncomputable def rational_root_problem : Prop :=
  ∃ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100) ∧
  ∀ p q : ℤ, (q ≠ 0) → (a * p^2 + b * p + c * q = 0) → 
  (p / q) ≤ -1 / 99

theorem largest_possible_rational_root : rational_root_problem :=
sorry

end largest_possible_rational_root_l27_27979


namespace small_frog_reaches_7th_rung_medium_frog_cannot_reach_1st_rung_large_frog_reaches_3rd_rung_l27_27509

-- 1. Prove that the small frog can reach the 7th rung
theorem small_frog_reaches_7th_rung : ∃ (a b : ℕ), 2 * a + 3 * b = 7 :=
by sorry

-- 2. Prove that the medium frog cannot reach the 1st rung
theorem medium_frog_cannot_reach_1st_rung : ¬(∃ (a b : ℕ), 2 * a + 4 * b = 1) :=
by sorry

-- 3. Prove that the large frog can reach the 3rd rung
theorem large_frog_reaches_3rd_rung : ∃ (a b : ℕ), 6 * a + 9 * b = 3 :=
by sorry

end small_frog_reaches_7th_rung_medium_frog_cannot_reach_1st_rung_large_frog_reaches_3rd_rung_l27_27509


namespace find_a_l27_27968

-- Define the given context (condition)
def condition (a : ℝ) : Prop := 0.5 / 100 * a = 75 / 100 -- since 1 paise = 1/100 rupee

-- Define the statement to prove
theorem find_a (a : ℝ) (h : condition a) : a = 150 := 
sorry

end find_a_l27_27968


namespace simplify_tan_expression_l27_27025

theorem simplify_tan_expression :
  (1 + Real.tan (15 * Real.pi / 180)) * (1 + Real.tan (30 * Real.pi / 180)) = 2 :=
by
  have h1 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h2 : Real.tan (15 * Real.pi / 180 + 30 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h3 : 15 * Real.pi / 180 + 30 * Real.pi / 180 = 45 * Real.pi / 180 := by sorry
  have h4 : Real.tan (45 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h5 : (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) = 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  sorry

end simplify_tan_expression_l27_27025


namespace remainder_of_6_pow_1234_mod_13_l27_27923

theorem remainder_of_6_pow_1234_mod_13 : 6 ^ 1234 % 13 = 10 := 
by 
  sorry

end remainder_of_6_pow_1234_mod_13_l27_27923


namespace reduced_price_l27_27282

theorem reduced_price (P : ℝ) (hP : P = 56)
    (original_qty : ℝ := 800 / P)
    (reduced_qty : ℝ := 800 / (0.65 * P))
    (diff_qty : ℝ := reduced_qty - original_qty)
    (difference_condition : diff_qty = 5) :
  0.65 * P = 36.4 :=
by
  rw [hP]
  sorry

end reduced_price_l27_27282


namespace gcd_of_720_120_168_is_24_l27_27082

theorem gcd_of_720_120_168_is_24 : Int.gcd (Int.gcd 720 120) 168 = 24 := 
by sorry

end gcd_of_720_120_168_is_24_l27_27082


namespace problem_statement_l27_27357

def f (x : ℕ) : ℝ := sorry

theorem problem_statement (h_cond : ∀ k : ℕ, f k ≤ (k : ℝ) ^ 2 → f (k + 1) ≤ (k + 1 : ℝ) ^ 2)
    (h_f7 : f 7 = 50) : ∀ k : ℕ, k ≤ 7 → f k > (k : ℝ) ^ 2 :=
sorry

end problem_statement_l27_27357


namespace yanna_change_l27_27016

theorem yanna_change :
  let shirt_cost := 5
  let sandal_cost := 3
  let num_shirts := 10
  let num_sandals := 3
  let given_amount := 100
  (given_amount - (num_shirts * shirt_cost + num_sandals * sandal_cost)) = 41 :=
by
  sorry

end yanna_change_l27_27016


namespace bags_sold_on_Thursday_l27_27097

theorem bags_sold_on_Thursday 
    (total_bags : ℕ) (sold_Monday : ℕ) (sold_Tuesday : ℕ) (sold_Wednesday : ℕ) (sold_Friday : ℕ) (percent_not_sold : ℕ) :
    total_bags = 600 →
    sold_Monday = 25 →
    sold_Tuesday = 70 →
    sold_Wednesday = 100 →
    sold_Friday = 145 →
    percent_not_sold = 25 →
    ∃ (sold_Thursday : ℕ), sold_Thursday = 110 :=
by
  sorry

end bags_sold_on_Thursday_l27_27097


namespace calculate_power_l27_27291

variable (x y : ℝ)

theorem calculate_power :
  (- (1 / 2) * x^2 * y)^3 = - (1 / 8) * x^6 * y^3 :=
sorry

end calculate_power_l27_27291


namespace centroid_distance_l27_27414

theorem centroid_distance
  (a b m : ℝ)
  (h_a_nonneg : 0 ≤ a)
  (h_b_nonneg : 0 ≤ b)
  (h_m_pos : 0 < m) :
  (∃ d : ℝ, d = m * (b + 2 * a) / (3 * (a + b))) :=
by
  sorry

end centroid_distance_l27_27414


namespace trailing_zeros_310_factorial_l27_27856

def count_trailing_zeros (n : Nat) : Nat :=
  n / 5 + n / 25 + n / 125 + n / 625

theorem trailing_zeros_310_factorial :
  count_trailing_zeros 310 = 76 := by
sorry

end trailing_zeros_310_factorial_l27_27856


namespace top_card_is_queen_probability_l27_27375

theorem top_card_is_queen_probability :
  let num_queens := 4
  let total_cards := 52
  let prob := num_queens / total_cards
  prob = 1 / 13 :=
by 
  sorry

end top_card_is_queen_probability_l27_27375


namespace smallest_four_digit_integer_mod_8_eq_3_l27_27250

theorem smallest_four_digit_integer_mod_8_eq_3 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 3 ∧ n = 1003 := by
  -- Proof will be provided here
  sorry

end smallest_four_digit_integer_mod_8_eq_3_l27_27250


namespace profit_june_correct_l27_27703

-- Define conditions
def profit_in_May : ℝ := 20000
def profit_in_July : ℝ := 28800

-- Define the monthly growth rate variable
variable (x : ℝ)

-- The growth factor per month
def growth_factor : ℝ := 1 + x

-- Given condition translated to an equation
def profit_relation (x : ℝ) : Prop :=
  profit_in_May * (growth_factor x) * (growth_factor x) = profit_in_July

-- The profit in June should be computed
def profit_in_June (x : ℝ) : ℝ :=
  profit_in_May * (growth_factor x)

-- The target profit in June we want to prove
def target_profit_in_June := 24000

-- Statement to prove
theorem profit_june_correct (h : profit_relation x) : profit_in_June x = target_profit_in_June :=
  sorry  -- proof to be completed

end profit_june_correct_l27_27703


namespace repeated_two_digit_number_divisible_by_101_l27_27393

theorem repeated_two_digit_number_divisible_by_101 (a b : ℕ) :
  (10 ≤ a ∧ a ≤ 99 ∧ 0 ≤ b ∧ b ≤ 9) →
  ∃ k, (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) = 101 * k :=
by
  intro h
  sorry

end repeated_two_digit_number_divisible_by_101_l27_27393


namespace integer_solutions_eq_l27_27768

theorem integer_solutions_eq (x y z : ℤ) :
  (x + y + z) ^ 5 = 80 * x * y * z * (x ^ 2 + y ^ 2 + z ^ 2) ↔
  ∃ a : ℤ, (x = a ∧ y = -a ∧ z = 0) ∨ (x = -a ∧ y = a ∧ z = 0) ∨ (x = a ∧ y = 0 ∧ z = -a) ∨ (x = -a ∧ y = 0 ∧ z = a) ∨ (x = 0 ∧ y = a ∧ z = -a) ∨ (x = 0 ∧ y = -a ∧ z = a) :=
by sorry

end integer_solutions_eq_l27_27768


namespace solution_set_of_inequality_l27_27978

theorem solution_set_of_inequality (x : ℝ) : (x^2 + 4*x - 5 < 0) ↔ (-5 < x ∧ x < 1) :=
by sorry

end solution_set_of_inequality_l27_27978


namespace vector_at_t_zero_l27_27686

theorem vector_at_t_zero :
  ∃ a d : ℝ × ℝ, (a + d = (2, 5) ∧ a + 4 * d = (11, -7)) ∧ a = (-1, 9) ∧ a + 0 * d = (-1, 9) :=
by {
  sorry
}

end vector_at_t_zero_l27_27686


namespace percent_value_in_quarters_l27_27436

theorem percent_value_in_quarters (dimes quarters : ℕ) (dime_value quarter_value : ℕ) (dime_count quarter_count : ℕ) :
  dimes = 50 →
  quarters = 20 →
  dime_value = 10 →
  quarter_value = 25 →
  dime_count = dimes * dime_value →
  quarter_count = quarters * quarter_value →
  (quarter_count : ℚ) / (dime_count + quarter_count) * 100 = 50 :=
by
  intros
  sorry

end percent_value_in_quarters_l27_27436


namespace orange_ratio_l27_27335

variable {R U : ℕ}

theorem orange_ratio (h1 : R + U = 96) 
                    (h2 : (3 / 4 : ℝ) * R + (7 / 8 : ℝ) * U = 78) :
  (R : ℝ) / (R + U : ℝ) = 1 / 2 := 
by
  sorry

end orange_ratio_l27_27335


namespace point_inside_circle_l27_27398

theorem point_inside_circle : 
  ∀ (x y : ℝ), 
  (x-2)^2 + (y-3)^2 = 4 → 
  (3-2)^2 + (2-3)^2 < 4 :=
by
  intro x y h
  sorry

end point_inside_circle_l27_27398


namespace solve_equation_l27_27883

theorem solve_equation (x : ℝ) : (x - 2) ^ 2 = 9 ↔ x = 5 ∨ x = -1 :=
by
  sorry -- Proof is skipped

end solve_equation_l27_27883


namespace inequality_holds_l27_27499

variable {x y : ℝ}

theorem inequality_holds (h₀ : 0 < x) (h₁ : x < 1) (h₂ : 0 < y) (h₃ : y < 1) :
  (x^2 / (x + y)) + (y^2 / (1 - x)) + ((1 - x - y)^2 / (1 - y)) ≥ 1 / 2 := by
  sorry

end inequality_holds_l27_27499


namespace count_total_legs_l27_27713

theorem count_total_legs :
  let tables4 := 4 * 4
  let sofa := 1 * 4
  let chairs4 := 2 * 4
  let tables3 := 3 * 3
  let table1 := 1 * 1
  let rocking_chair := 1 * 2
  let total_legs := tables4 + sofa + chairs4 + tables3 + table1 + rocking_chair
  total_legs = 40 :=
by
  sorry

end count_total_legs_l27_27713


namespace jana_walk_distance_l27_27437

-- Define the time taken to walk one mile and the rest period
def walk_time_per_mile : ℕ := 24
def rest_time_per_mile : ℕ := 6

-- Define the total time spent per mile (walking + resting)
def total_time_per_mile : ℕ := walk_time_per_mile + rest_time_per_mile

-- Define the total available time
def total_available_time : ℕ := 78

-- Define the number of complete cycles of walking and resting within the total available time
def complete_cycles : ℕ := total_available_time / total_time_per_mile

-- Define the distance walked per cycle (in miles)
def distance_per_cycle : ℝ := 1.0

-- Define the total distance walked
def total_distance_walked : ℝ := complete_cycles * distance_per_cycle

-- The proof statement
theorem jana_walk_distance : total_distance_walked = 2.0 := by
  sorry

end jana_walk_distance_l27_27437


namespace exists_quadratic_satisfying_conditions_l27_27118

theorem exists_quadratic_satisfying_conditions :
  ∃ (a b c : ℝ), 
  (a - b + c = 0) ∧
  (∀ x : ℝ, x ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ (1 + x^2) / 2) ∧ 
  (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
  sorry

end exists_quadratic_satisfying_conditions_l27_27118


namespace number_of_weavers_is_4_l27_27029

theorem number_of_weavers_is_4
  (mats1 days1 weavers1 mats2 days2 weavers2 : ℕ)
  (h1 : mats1 = 4)
  (h2 : days1 = 4)
  (h3 : weavers2 = 10)
  (h4 : mats2 = 25)
  (h5 : days2 = 10)
  (h_rate_eq : (mats1 / (weavers1 * days1)) = (mats2 / (weavers2 * days2))) :
  weavers1 = 4 :=
by
  sorry

end number_of_weavers_is_4_l27_27029


namespace target_destroyed_probability_l27_27660

noncomputable def probability_hit (p1 p2 p3 : ℝ) : ℝ :=
  let miss1 := 1 - p1
  let miss2 := 1 - p2
  let miss3 := 1 - p3
  let prob_all_miss := miss1 * miss2 * miss3
  let prob_one_hit := (p1 * miss2 * miss3) + (miss1 * p2 * miss3) + (miss1 * miss2 * p3)
  let prob_destroyed := 1 - (prob_all_miss + prob_one_hit)
  prob_destroyed

theorem target_destroyed_probability :
  probability_hit 0.9 0.9 0.8 = 0.954 :=
sorry

end target_destroyed_probability_l27_27660


namespace tangent_line_equation_at_point_l27_27020

theorem tangent_line_equation_at_point 
  (x y : ℝ) (h_curve : y = x^3 - 2 * x) (h_point : (x, y) = (1, -1)) : 
  (x - y - 2 = 0) := 
sorry

end tangent_line_equation_at_point_l27_27020


namespace average_fuel_efficiency_l27_27779

theorem average_fuel_efficiency (d1 d2 : ℝ) (e1 e2 : ℝ) (fuel1 fuel2 : ℝ)
  (h1 : d1 = 150) (h2 : e1 = 35) (h3 : d2 = 180) (h4 : e2 = 18)
  (h_fuel1 : fuel1 = d1 / e1) (h_fuel2 : fuel2 = d2 / e2)
  (total_distance : ℝ := 330)
  (total_fuel : ℝ := fuel1 + fuel2) :
  total_distance / total_fuel = 23 := by
  sorry

end average_fuel_efficiency_l27_27779


namespace sharks_in_Cape_May_August_l27_27382

section
variable {D_J C_J D_A C_A : ℕ}

-- Given conditions
theorem sharks_in_Cape_May_August 
  (h1 : C_J = 2 * D_J) 
  (h2 : C_A = 5 + 3 * D_A) 
  (h3 : D_J = 23) 
  (h4 : D_A = D_J) : 
  C_A = 74 := 
by 
  -- Skipped the proof steps 
  sorry
end

end sharks_in_Cape_May_August_l27_27382


namespace last_three_digits_7_pow_123_l27_27949

theorem last_three_digits_7_pow_123 : (7^123 % 1000) = 717 := sorry

end last_three_digits_7_pow_123_l27_27949


namespace beth_total_crayons_l27_27093

theorem beth_total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) 
  (h1 : packs = 8) (h2 : crayons_per_pack = 20) (h3 : extra_crayons = 15) :
  packs * crayons_per_pack + extra_crayons = 175 :=
by
  sorry

end beth_total_crayons_l27_27093


namespace square_root_25_pm5_l27_27915

-- Define that a number x satisfies the equation x^2 = 25
def square_root_of_25 (x : ℝ) : Prop := x * x = 25

-- The theorem states that the square root of 25 is ±5
theorem square_root_25_pm5 : ∀ x : ℝ, square_root_of_25 x ↔ x = 5 ∨ x = -5 :=
by
  intros x
  sorry

end square_root_25_pm5_l27_27915


namespace tree_distance_l27_27805

theorem tree_distance 
  (num_trees : ℕ) (dist_first_to_fifth : ℕ) (length_of_road : ℤ) 
  (h1 : num_trees = 8) 
  (h2 : dist_first_to_fifth = 100) 
  (h3 : length_of_road = (dist_first_to_fifth * (num_trees - 1)) / 4 + 3 * dist_first_to_fifth) 
  :
  length_of_road = 175 := 
sorry

end tree_distance_l27_27805


namespace find_A_B_l27_27076

theorem find_A_B :
  ∀ (A B : ℝ), (∀ (x : ℝ), 1 < x → ⌊1 / (A * x + B / x)⌋ = 1 / (A * ⌊x⌋ + B / ⌊x⌋)) →
  (A = 0) ∧ (B = 1) :=
by
  sorry

end find_A_B_l27_27076


namespace pow_gt_of_gt_l27_27367

variable {a x1 x2 : ℝ}

theorem pow_gt_of_gt (ha : a > 1) (hx : x1 > x2) : a^x1 > a^x2 :=
by sorry

end pow_gt_of_gt_l27_27367


namespace horner_evaluation_of_f_at_5_l27_27948

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem horner_evaluation_of_f_at_5 : f 5 = 2015 :=
by sorry

end horner_evaluation_of_f_at_5_l27_27948


namespace exists_prime_mod_greater_remainder_l27_27701

theorem exists_prime_mod_greater_remainder (a b : ℕ) (h1 : 0 < a) (h2 : a < b) :
  ∃ p : ℕ, Prime p ∧ a % p > b % p :=
sorry

end exists_prime_mod_greater_remainder_l27_27701


namespace range_of_a_l27_27626

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then -x + 5 else 2 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x, 3 ≤ f x a) ∧ (0 < a) ∧ (a ≠ 1) → 1 < a ∧ a ≤ 2 :=
by
  intro h
  sorry

end range_of_a_l27_27626


namespace seq_eighth_term_l27_27419

theorem seq_eighth_term : (8^2 + 2 * 8 - 1 = 79) :=
by
  sorry

end seq_eighth_term_l27_27419


namespace simplify_expression_l27_27002

theorem simplify_expression (r : ℝ) : (2 * r^2 + 5 * r - 3) + (3 * r^2 - 4 * r + 2) = 5 * r^2 + r - 1 := 
by
  sorry

end simplify_expression_l27_27002


namespace monkey_tree_height_l27_27312

theorem monkey_tree_height (hours: ℕ) (hop ft_per_hour : ℕ) (slip ft_per_hour : ℕ) (net_progress : ℕ) (final_hour : ℕ) (total_height : ℕ) :
  (hours = 18) ∧
  (hop = 3) ∧
  (slip = 2) ∧
  (net_progress = hop - slip) ∧
  (net_progress = 1) ∧
  (final_hour = 1) ∧
  (total_height = (hours - 1) * net_progress + hop) ∧
  (total_height = 20) :=
by
  sorry

end monkey_tree_height_l27_27312


namespace sin_eq_product_one_eighth_l27_27265

open Real

theorem sin_eq_product_one_eighth :
  (∀ (n k m : ℕ), 1 ≤ n → n ≤ 5 → 1 ≤ k → k ≤ 5 → 1 ≤ m → m ≤ 5 →
    sin (π * n / 12) * sin (π * k / 12) * sin (π * m / 12) = 1 / 8) ↔ (n = 2 ∧ k = 2 ∧ m = 2) := by
  sorry

end sin_eq_product_one_eighth_l27_27265


namespace average_capacity_is_3_65_l27_27292

/-- Define the capacities of the jars as a list--/
def jarCapacities : List ℚ := [2, 1/4, 8, 1.5, 0.75, 3, 10]

/-- Calculate the average jar capacity --/
def averageCapacity (capacities : List ℚ) : ℚ :=
  (capacities.sum) / (capacities.length)

/-- The average jar capacity for the given list of jar capacities is 3.65 liters. --/
theorem average_capacity_is_3_65 :
  averageCapacity jarCapacities = 3.65 := 
by
  unfold averageCapacity
  dsimp [jarCapacities]
  norm_num
  sorry

end average_capacity_is_3_65_l27_27292


namespace circle_x_intersect_l27_27616

theorem circle_x_intersect (x y : ℝ) : 
  (x, y) = (0, 0) ∨ (x, y) = (10, 0) → (x = 10) :=
by
  -- conditions:
  -- The endpoints of the diameter are (0,0) and (10,10)
  -- (proving that the second intersect on x-axis has x-coordinate 10)
  sorry

end circle_x_intersect_l27_27616


namespace sum_inequality_l27_27576

theorem sum_inequality 
  {a b c : ℝ}
  (h : a + b + c = 3) : 
  (1 / (a^2 - a + 2) + 1 / (b^2 - b + 2) + 1 / (c^2 - c + 2)) ≤ 3 / 2 := 
sorry

end sum_inequality_l27_27576


namespace subtract_abs_from_local_value_l27_27549

-- Define the local value of 4 in 564823 as 4000
def local_value_of_4_in_564823 : ℕ := 4000

-- Define the absolute value of 4 as 4
def absolute_value_of_4 : ℕ := 4

-- Theorem statement: Prove that subtracting the absolute value of 4 from the local value of 4 in 564823 equals 3996
theorem subtract_abs_from_local_value : (local_value_of_4_in_564823 - absolute_value_of_4) = 3996 :=
by
  sorry

end subtract_abs_from_local_value_l27_27549


namespace mark_notebooks_at_126_percent_l27_27956

variable (L : ℝ) (C : ℝ) (M : ℝ) (S : ℝ)

def merchant_condition1 := C = 0.85 * L
def merchant_condition2 := C = 0.75 * S
def merchant_condition3 := S = 0.9 * M

theorem mark_notebooks_at_126_percent :
    merchant_condition1 L C →
    merchant_condition2 C S →
    merchant_condition3 S M →
    M = 1.259 * L := by
  intros h1 h2 h3
  sorry

end mark_notebooks_at_126_percent_l27_27956


namespace quadratic_fraction_equality_l27_27905

theorem quadratic_fraction_equality (r : ℝ) (h1 : r ≠ 4) (h2 : r ≠ 6) (h3 : r ≠ 5) 
(h4 : r ≠ -4) (h5 : r ≠ -3): 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 18) / (r^2 - 2*r - 24) →
  r = -7/4 :=
by {
  sorry
}

end quadratic_fraction_equality_l27_27905


namespace cos_alpha_sub_beta_sin_alpha_l27_27602

open Real

variables (α β : ℝ)

-- Conditions:
-- 0 < α < π / 2
def alpha_in_first_quadrant := 0 < α ∧ α < π / 2

-- -π / 2 < β < 0
def beta_in_fourth_quadrant := -π / 2 < β ∧ β < 0

-- sin β = -5/13
def sin_beta := sin β = -5 / 13

-- tan(α - β) = 4/3
def tan_alpha_sub_beta := tan (α - β) = 4 / 3

-- Theorem statements (follows directly from the conditions and the equivalence):
theorem cos_alpha_sub_beta : alpha_in_first_quadrant α → beta_in_fourth_quadrant β → sin_beta β → tan_alpha_sub_beta α β → cos (α - β) = 3 / 5 := sorry

theorem sin_alpha : alpha_in_first_quadrant α → beta_in_fourth_quadrant β → sin_beta β → tan_alpha_sub_beta α β → sin α = 33 / 65 := sorry

end cos_alpha_sub_beta_sin_alpha_l27_27602


namespace quotient_is_six_l27_27284

-- Definition of the given conditions
def S : Int := 476
def remainder : Int := 15
def difference : Int := 2395

-- Definition of the larger number based on the given conditions
def L : Int := S + difference

-- The statement we need to prove
theorem quotient_is_six : (L = S * 6 + remainder) := by
  sorry

end quotient_is_six_l27_27284


namespace never_prime_except_three_l27_27286

theorem never_prime_except_three (p : ℕ) (hp : Nat.Prime p) :
  p^2 + 8 = 17 ∨ ∃ k, (k ≠ 1 ∧ k ≠ p^2 + 8 ∧ k ∣ (p^2 + 8)) := by
  sorry

end never_prime_except_three_l27_27286


namespace ben_remaining_money_l27_27457

variable (initial_capital : ℝ := 2000) 
variable (payment_to_supplier : ℝ := 600)
variable (payment_from_debtor : ℝ := 800)
variable (maintenance_cost : ℝ := 1200)
variable (remaining_capital : ℝ := 1000)

theorem ben_remaining_money
  (h1 : initial_capital = 2000)
  (h2 : payment_to_supplier = 600)
  (h3 : payment_from_debtor = 800)
  (h4 : maintenance_cost = 1200) :
  remaining_capital = (initial_capital - payment_to_supplier + payment_from_debtor - maintenance_cost) :=
sorry

end ben_remaining_money_l27_27457


namespace fraction_checked_by_worker_y_l27_27142

theorem fraction_checked_by_worker_y
  (f_X f_Y : ℝ)
  (h1 : f_X + f_Y = 1)
  (h2 : 0.005 * f_X + 0.008 * f_Y = 0.0074) :
  f_Y = 0.8 :=
by
  sorry

end fraction_checked_by_worker_y_l27_27142


namespace original_price_l27_27639

theorem original_price (P: ℝ) (h: 0.80 * 1.15 * P = 46) : P = 50 :=
by sorry

end original_price_l27_27639


namespace flour_needed_l27_27811

theorem flour_needed (flour_per_24_cookies : ℝ) (cookies_per_recipe : ℕ) (desired_cookies : ℕ) 
  (h : flour_per_24_cookies = 1.5) (h1 : cookies_per_recipe = 24) (h2 : desired_cookies = 72) : 
  flour_per_24_cookies / cookies_per_recipe * desired_cookies = 4.5 := 
  by {
    -- The proof is omitted
    sorry
  }

end flour_needed_l27_27811


namespace zero_in_interval_l27_27747

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 2

theorem zero_in_interval : f 1 < 0 ∧ f 2 > 0 → ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 := 
by
  intros h
  sorry

end zero_in_interval_l27_27747


namespace original_number_increased_by_45_percent_is_870_l27_27991

theorem original_number_increased_by_45_percent_is_870 (x : ℝ) (h : x * 1.45 = 870) : x = 870 / 1.45 :=
by sorry

end original_number_increased_by_45_percent_is_870_l27_27991


namespace initial_clothing_count_l27_27564

theorem initial_clothing_count 
  (donated_first : ℕ) 
  (donated_second : ℕ) 
  (thrown_away : ℕ) 
  (remaining : ℕ) 
  (h1 : donated_first = 5) 
  (h2 : donated_second = 3 * donated_first) 
  (h3 : thrown_away = 15) 
  (h4 : remaining = 65) :
  donated_first + donated_second + thrown_away + remaining = 100 :=
by
  sorry

end initial_clothing_count_l27_27564


namespace difference_largest_smallest_l27_27423

noncomputable def ratio_2_3_5 := 2 / 3
noncomputable def ratio_3_5 := 3 / 5
noncomputable def int_sum := 90

theorem difference_largest_smallest :
  ∃ (a b c : ℝ), 
    a + b + c = int_sum ∧
    b / a = ratio_2_3_5 ∧
    c / a = 5 / 2 ∧
    b / a = 3 / 2 ∧
    c - a = 12.846 := 
by
  sorry

end difference_largest_smallest_l27_27423


namespace animal_count_in_hollow_l27_27664

theorem animal_count_in_hollow (heads legs : ℕ) (animals_with_odd_legs animals_with_even_legs : ℕ) :
  heads = 18 →
  legs = 24 →
  (∀ n, n % 2 = 1 → animals_with_odd_legs * 2 = heads - 2 * n) →
  (∀ m, m % 2 = 0 → animals_with_even_legs * 1 = heads - m) →
  (animals_with_odd_legs + animals_with_even_legs = 10 ∨
   animals_with_odd_legs + animals_with_even_legs = 12 ∨
   animals_with_odd_legs + animals_with_even_legs = 14) :=
sorry

end animal_count_in_hollow_l27_27664


namespace find_number_l27_27630

theorem find_number (x y : ℝ) (h1 : x = y + 0.25 * y) (h2 : x = 110) : y = 88 := 
by
  sorry

end find_number_l27_27630


namespace exponential_comparison_l27_27611

theorem exponential_comparison (x y a b : ℝ) (hx : x > y) (hy : y > 1) (ha : 0 < a) (hb : a < b) (hb' : b < 1) : 
  a^x < b^y :=
sorry

end exponential_comparison_l27_27611


namespace sergeant_distance_travel_l27_27497

noncomputable def sergeant_distance (x k : ℝ) : ℝ :=
  let t₁ := 1 / (x * (k - 1))
  let t₂ := 1 / (x * (k + 1))
  let t := t₁ + t₂
  let d := k * 4 / 3
  d

theorem sergeant_distance_travel (x k : ℝ) (h1 : (4 * k) / (k^2 - 1) = 4 / 3) :
  sergeant_distance x k = 8 / 3 := by
  sorry

end sergeant_distance_travel_l27_27497


namespace find_ab_l27_27342

theorem find_ab (a b : ℝ) 
  (H_period : (1 : ℝ) * (π / b) = π / 2)
  (H_point : a * Real.tan (b * (π / 8)) = 4) :
  a * b = 8 :=
sorry

end find_ab_l27_27342


namespace children_count_l27_27106

theorem children_count (W C n : ℝ) (h1 : 4 * W = 1 / 7) (h2 : n * C = 1 / 14) (h3 : 5 * W + 10 * C = 1 / 4) : n = 10 :=
by
  sorry

end children_count_l27_27106


namespace average_of_possible_values_l27_27749

theorem average_of_possible_values 
  (x : ℝ)
  (h : Real.sqrt (2 * x^2 + 5) = Real.sqrt 25) : 
  (x = Real.sqrt 10 ∨ x = -Real.sqrt 10) → (Real.sqrt 10 + (-Real.sqrt 10)) / 2 = 0 :=
by
  sorry

end average_of_possible_values_l27_27749


namespace find_scalars_l27_27261

noncomputable def N : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-2, 0]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

theorem find_scalars (r s : ℤ) (h_r : r = 3) (h_s : s = -8) :
    N * N = r • N + s • I :=
by
  rw [h_r, h_s]
  sorry

end find_scalars_l27_27261


namespace smallest_value_in_geometric_progression_l27_27203

open Real

theorem smallest_value_in_geometric_progression 
  (d : ℝ) : 
  (∀ a b c d : ℝ, 
    a = 5 ∧ b = 5 + d ∧ c = 5 + 2 * d ∧ d = 5 + 3 * d ∧ 
    ∀ a' b' c' d' : ℝ, 
      a' = 5 ∧ b' = 6 + d ∧ c' = 15 + 2 * d ∧ d' = 3 * d ∧ 
      (b' / a' = c' / b' ∧ c' / b' = d' / c')) → 
  (d = (-1 + 4 * sqrt 10) ∨ d = (-1 - 4 * sqrt 10)) → 
  (min (3 * (-1 + 4 * sqrt 10)) (3 * (-1 - 4 * sqrt 10)) = -3 - 12 * sqrt 10) :=
by
  intros ha hd
  sorry

end smallest_value_in_geometric_progression_l27_27203


namespace find_p_from_circle_and_parabola_tangency_l27_27775

theorem find_p_from_circle_and_parabola_tangency :
  (∃ x y : ℝ, (x^2 + y^2 - 6*x - 7 = 0) ∧ (y^2 = 2*p * x) ∧ p > 0) →
  p = 2 :=
by {
  sorry
}

end find_p_from_circle_and_parabola_tangency_l27_27775


namespace exists_triangle_l27_27528

variable (k α m_a : ℝ)

-- Define the main constructibility condition as a noncomputable function.
noncomputable def triangle_constructible (k α m_a : ℝ) : Prop :=
  m_a ≤ (k / 2) * ((1 - Real.sin (α / 2)) / Real.cos (α / 2))

-- Main theorem statement to prove the existence of the triangle
theorem exists_triangle :
  ∃ (k α m_a : ℝ), triangle_constructible k α m_a := 
sorry

end exists_triangle_l27_27528


namespace coeff_x3_product_l27_27879

open Polynomial

noncomputable def poly1 := (C 3 * X ^ 3) + (C 2 * X ^ 2) + (C 4 * X) + (C 5)
noncomputable def poly2 := (C 4 * X ^ 3) + (C 6 * X ^ 2) + (C 5 * X) + (C 2)

theorem coeff_x3_product : coeff (poly1 * poly2) 3 = 10 := by
  sorry

end coeff_x3_product_l27_27879


namespace monotonically_increasing_intervals_sin_value_l27_27364

noncomputable def f (x : Real) : Real := 2 * Real.cos (x - Real.pi / 3) * Real.cos x + 1

theorem monotonically_increasing_intervals :
  ∀ (k : Int), ∃ (a b : Real), a = k * Real.pi - Real.pi / 3 ∧ b = k * Real.pi + Real.pi / 6 ∧
                 ∀ (x y : Real), a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y :=
sorry

theorem sin_value 
  (α : Real) (hα : 0 < α ∧ α < Real.pi / 2) 
  (h : f (α + Real.pi / 12) = 7 / 6) : 
  Real.sin (7 * Real.pi / 6 - 2 * α) = 2 * Real.sqrt 2 / 3 :=
sorry

end monotonically_increasing_intervals_sin_value_l27_27364


namespace haley_magazines_l27_27078

theorem haley_magazines (boxes : ℕ) (magazines_per_box : ℕ) (total_magazines : ℕ) :
  boxes = 7 →
  magazines_per_box = 9 →
  total_magazines = boxes * magazines_per_box →
  total_magazines = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end haley_magazines_l27_27078


namespace total_visit_plans_l27_27134

def exhibitions : List String := ["Opera Culture Exhibition", "Ming Dynasty Imperial Cellar Porcelain Exhibition", "Historical Green Landscape Painting Exhibition", "Zhao Mengfu Calligraphy and Painting Exhibition"]

def painting_exhibitions : List String := ["Historical Green Landscape Painting Exhibition", "Zhao Mengfu Calligraphy and Painting Exhibition"]

def non_painting_exhibitions : List String := ["Opera Culture Exhibition", "Ming Dynasty Imperial Cellar Porcelain Exhibition"]

def num_visit_plans (exhibit_list : List String) (paintings : List String) (non_paintings : List String) : Nat :=
  let case1 := paintings.length * non_paintings.length * 2
  let case2 := if paintings.length >= 2 then 2 else 0
  case1 + case2

theorem total_visit_plans : num_visit_plans exhibitions painting_exhibitions non_painting_exhibitions = 10 :=
  sorry

end total_visit_plans_l27_27134


namespace cube_volume_l27_27341

theorem cube_volume (a : ℝ) (h : (a - 1) * (a - 1) * (a + 1) = a^3 - 7) : a^3 = 8 :=
  sorry

end cube_volume_l27_27341


namespace minimize_fraction_sum_l27_27328

theorem minimize_fraction_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 6) :
  (9 / a + 4 / b + 25 / c) ≥ 50 / 3 :=
sorry

end minimize_fraction_sum_l27_27328


namespace intersection_eq_l27_27012

open Set

variable {α : Type*}

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_eq : M ∩ N = {2, 3} := by
  apply Set.ext
  intro x
  simp [M, N]
  sorry

end intersection_eq_l27_27012


namespace largest_y_value_l27_27158

theorem largest_y_value (y : ℝ) (h : 3*y^2 + 18*y - 90 = y*(y + 17)) : y ≤ 3 :=
by
  sorry

end largest_y_value_l27_27158


namespace bear_small_animal_weight_l27_27449

theorem bear_small_animal_weight :
  let total_weight_needed := 1200
  let berries_weight := 1/5 * total_weight_needed
  let insects_weight := 1/10 * total_weight_needed
  let acorns_weight := 2 * berries_weight
  let honey_weight := 3 * insects_weight
  let total_weight_gained := berries_weight + insects_weight + acorns_weight + honey_weight
  let remaining_weight := total_weight_needed - total_weight_gained
  remaining_weight = 0 -> 0 = 0 := by
  intros total_weight_needed berries_weight insects_weight acorns_weight honey_weight
         total_weight_gained remaining_weight h
  exact Eq.refl 0

end bear_small_animal_weight_l27_27449


namespace roots_poly_sum_cubed_eq_l27_27242

theorem roots_poly_sum_cubed_eq :
  ∀ (r s t : ℝ), (r + s + t = 0) 
  → (∀ x, 9 * x^3 + 2023 * x + 4047 = 0 → x = r ∨ x = s ∨ x = t) 
  → (r + s) ^ 3 + (s + t) ^ 3 + (t + r) ^ 3 = 1349 :=
by
  intros r s t h_sum h_roots
  sorry

end roots_poly_sum_cubed_eq_l27_27242


namespace determine_range_a_l27_27638

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∀ x y : ℝ, 1 ≤ x → x ≤ y → 4 * x^2 - a * x ≤ 4 * y^2 - a * y

theorem determine_range_a (a : ℝ) (h : ¬ prop_p a ∧ (prop_p a ∨ prop_q a)) : 
  a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8) :=
sorry

end determine_range_a_l27_27638


namespace min_value_a_cube_plus_b_cube_no_exist_2a_plus_3b_eq_6_l27_27123

section ProofProblem

theorem min_value_a_cube_plus_b_cube {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hfra : 1/a + 1/b = Real.sqrt (a * b)) :
  a^3 + b^3 = 4 * Real.sqrt 2 := 
sorry

theorem no_exist_2a_plus_3b_eq_6 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hfra : 1/a + 1/b = Real.sqrt (a * b)) :
  ¬ (2 * a + 3 * b = 6) :=
sorry

end ProofProblem

end min_value_a_cube_plus_b_cube_no_exist_2a_plus_3b_eq_6_l27_27123


namespace correct_option_d_l27_27215

variable (m t x1 x2 y1 y2 : ℝ)

theorem correct_option_d (h_m : m > 0)
  (h_y1 : y1 = m * x1^2 - 2 * m * x1 + 1)
  (h_y2 : y2 = m * x2^2 - 2 * m * x2 + 1)
  (h_x1 : t < x1 ∧ x1 < t + 1)
  (h_x2 : t + 2 < x2 ∧ x2 < t + 3)
  (h_t_geq1 : t ≥ 1) :
  y1 < y2 := sorry

end correct_option_d_l27_27215


namespace correct_final_positions_l27_27645

noncomputable def shapes_after_rotation (initial_positions : (String × String) × (String × String) × (String × String)) : (String × String) × (String × String) × (String × String) :=
  match initial_positions with
  | (("Triangle", "Top"), ("Circle", "Lower Left"), ("Pentagon", "Lower Right")) =>
    (("Triangle", "Lower Right"), ("Circle", "Top"), ("Pentagon", "Lower Left"))
  | _ => initial_positions

theorem correct_final_positions :
  shapes_after_rotation (("Triangle", "Top"), ("Circle", "Lower Left"), ("Pentagon", "Lower Right")) = (("Triangle", "Lower Right"), ("Circle", "Top"), ("Pentagon", "Lower Left")) :=
by
  unfold shapes_after_rotation
  rfl

end correct_final_positions_l27_27645


namespace travel_time_at_constant_speed_l27_27188

theorem travel_time_at_constant_speed
  (distance : ℝ) (speed : ℝ) : 
  distance = 100 → speed = 20 → distance / speed = 5 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end travel_time_at_constant_speed_l27_27188


namespace volume_and_surface_area_implies_sum_of_edges_l27_27204

-- Define the problem conditions and prove the required statement
theorem volume_and_surface_area_implies_sum_of_edges :
  ∃ (a r : ℝ), 
    (a / r) * a * (a * r) = 216 ∧ 
    2 * ((a^2 / r) + a^2 * r + a^2) = 288 →
    4 * ((a / r) + a * r + a) = 96 :=
by
  sorry

end volume_and_surface_area_implies_sum_of_edges_l27_27204


namespace calculate_total_notebooks_given_to_tom_l27_27866

noncomputable def total_notebooks_given_to_tom : ℝ :=
  let initial_red := 15
  let initial_blue := 17
  let initial_white := 19
  let red_given_day1 := 4.5
  let blue_given_day1 := initial_blue / 3
  let remaining_red_day1 := initial_red - red_given_day1
  let remaining_blue_day1 := initial_blue - blue_given_day1
  let white_given_day2 := initial_white / 2
  let blue_given_day2 := remaining_blue_day1 * 0.25
  let remaining_white_day2 := initial_white - white_given_day2
  let remaining_blue_day2 := remaining_blue_day1 - blue_given_day2
  let red_given_day3 := 3.5
  let blue_given_day3 := (remaining_blue_day2 * 2) / 5
  let remaining_red_day3 := remaining_red_day1 - red_given_day3
  let remaining_blue_day3 := remaining_blue_day2 - blue_given_day3
  let white_kept_day3 := remaining_white_day2 / 4
  let remaining_white_day3 := initial_white - white_kept_day3
  let remaining_notebooks_day3 := remaining_red_day3 + remaining_blue_day3 + remaining_white_day3
  let notebooks_total_day3 := initial_red + initial_blue + initial_white - red_given_day1 - blue_given_day1 - white_given_day2 - blue_given_day2 - red_given_day3 - blue_given_day3 - white_kept_day3
  let tom_notebooks := red_given_day1 + blue_given_day1
  notebooks_total_day3

theorem calculate_total_notebooks_given_to_tom : total_notebooks_given_to_tom = 10.17 :=
  sorry

end calculate_total_notebooks_given_to_tom_l27_27866


namespace trig_problem_l27_27073

theorem trig_problem (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by
  sorry

end trig_problem_l27_27073


namespace faye_rows_l27_27177

theorem faye_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (rows_created : ℕ) :
  total_pencils = 12 → pencils_per_row = 4 → rows_created = 3 := by
  sorry

end faye_rows_l27_27177


namespace ratio_of_chocolate_to_regular_milk_l27_27484

def total_cartons : Nat := 24
def regular_milk_cartons : Nat := 3
def chocolate_milk_cartons : Nat := total_cartons - regular_milk_cartons

theorem ratio_of_chocolate_to_regular_milk (h1 : total_cartons = 24) (h2 : regular_milk_cartons = 3) :
  chocolate_milk_cartons / regular_milk_cartons = 7 :=
by 
  -- Skipping proof with sorry
  sorry

end ratio_of_chocolate_to_regular_milk_l27_27484


namespace find_a_l27_27571

open Real

def point_in_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 6 * x + 4 * y + 4 = 0

def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 3 = 0

theorem find_a (a : ℝ) :
  point_in_circle 1 a →
  line_equation 1 a →
  a = -2 :=
by
  intro h1 h2
  sorry

end find_a_l27_27571


namespace geometric_series_common_ratio_l27_27296

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l27_27296


namespace age_ratio_proof_l27_27533

variable (j a x : ℕ)

/-- Given conditions about Jack and Alex's ages. -/
axiom h1 : j - 3 = 2 * (a - 3)
axiom h2 : j - 5 = 3 * (a - 5)

def age_ratio_in_years : Prop :=
  (3 * (a + x) = 2 * (j + x)) → (x = 1)

theorem age_ratio_proof : age_ratio_in_years j a x := by
  sorry

end age_ratio_proof_l27_27533


namespace range_of_a_l27_27246

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : (A ∩ B a).Nonempty) : a > 1 :=
sorry

end range_of_a_l27_27246


namespace video_minutes_per_week_l27_27311

theorem video_minutes_per_week
  (daily_videos : ℕ := 3)
  (short_video_length : ℕ := 2)
  (long_video_multiplier : ℕ := 6)
  (days_in_week : ℕ := 7) :
  (2 * short_video_length + long_video_multiplier * short_video_length) * days_in_week = 112 := 
by 
  -- conditions
  let short_videos_per_day := 2
  let long_video_length := long_video_multiplier * short_video_length
  let daily_total := short_videos_per_day * short_video_length + long_video_length
  let weekly_total := daily_total * days_in_week
  -- proof
  sorry

end video_minutes_per_week_l27_27311


namespace perfect_squares_ending_in_5_or_6_lt_2000_l27_27389

theorem perfect_squares_ending_in_5_or_6_lt_2000 :
  ∃ (n : ℕ), n = 9 ∧ ∀ k, 1 ≤ k ∧ k ≤ 44 → 
  (∃ m, m * m < 2000 ∧ (m % 10 = 5 ∨ m % 10 = 6)) :=
by
  sorry

end perfect_squares_ending_in_5_or_6_lt_2000_l27_27389


namespace train_crossing_time_l27_27469

noncomputable def time_to_cross_bridge (l_train : ℕ) (v_train_kmh : ℕ) (l_bridge : ℕ) : ℚ :=
  let total_distance := l_train + l_bridge
  let v_train_ms := (v_train_kmh * 1000 : ℚ) / 3600
  total_distance / v_train_ms

theorem train_crossing_time :
  time_to_cross_bridge 110 72 136 = 12.3 := 
by
  sorry

end train_crossing_time_l27_27469


namespace arithmetic_sequence_num_terms_l27_27474

theorem arithmetic_sequence_num_terms 
  (a : ℕ) (d : ℕ) (l : ℕ) (n : ℕ)
  (h1 : a = 20)
  (h2 : d = 5)
  (h3 : l = 150)
  (h4 : 150 = 20 + (n-1) * 5) :
  n = 27 :=
by sorry

end arithmetic_sequence_num_terms_l27_27474


namespace partition_solution_l27_27318

noncomputable def partitions (a m n x : ℝ) : Prop :=
  a = x + n * (a - m * x)

theorem partition_solution (a m n : ℝ) (h : n * m < 1) :
  partitions a m n (a * (1 - n) / (1 - n * m)) :=
by
  sorry

end partition_solution_l27_27318


namespace number_of_students_l27_27051

theorem number_of_students 
    (N : ℕ) 
    (h_percentage_5 : 28 * N % 100 = 0)
    (h_percentage_4 : 35 * N % 100 = 0)
    (h_percentage_3 : 25 * N % 100 = 0)
    (h_percentage_2 : 12 * N % 100 = 0)
    (h_class_limit : N ≤ 4 * 30) 
    (h_num_classes : 4 * 30 < 120)
    : N = 100 := 
by 
  sorry

end number_of_students_l27_27051


namespace f_of_3_is_log2_3_l27_27439

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (2 ^ x) = x

theorem f_of_3_is_log2_3 : f 3 = Real.log 3 / Real.log 2 := sorry

end f_of_3_is_log2_3_l27_27439


namespace am_gm_inequality_for_x_l27_27520

theorem am_gm_inequality_for_x (x : ℝ) : 1 + x^2 + x^6 + x^8 ≥ 4 * x^4 := by 
  sorry

end am_gm_inequality_for_x_l27_27520


namespace sum_of_coefficients_zero_l27_27404

open Real

theorem sum_of_coefficients_zero (a b c p1 p2 q1 q2 : ℝ)
  (h1 : ∃ p1 p2 : ℝ, p1 ≠ p2 ∧ a * p1^2 + b * p1 + c = 0 ∧ a * p2^2 + b * p2 + c = 0)
  (h2 : ∃ q1 q2 : ℝ, q1 ≠ q2 ∧ c * q1^2 + b * q1 + a = 0 ∧ c * q2^2 + b * q2 + a = 0)
  (h3 : q1 = p1 + (p2 - p1) / 2 ∧ p2 = p1 + (p2 - p1) ∧ q2 = p1 + 3 * (p2 - p1) / 2) :
  a + c = 0 := sorry

end sum_of_coefficients_zero_l27_27404


namespace total_cost_train_and_bus_l27_27378

noncomputable def trainFare := 3.75 + 2.35
noncomputable def busFare := 3.75
noncomputable def totalFare := trainFare + busFare

theorem total_cost_train_and_bus : totalFare = 9.85 :=
by
  -- We'll need a proof here if required.
  sorry

end total_cost_train_and_bus_l27_27378


namespace problem_correct_l27_27167

def decimal_to_fraction_eq_80_5 : Prop :=
  ( (0.5 + 0.25 + 0.125) / (0.5 * 0.25 * 0.125) * ((7 / 18 * (9 / 2) + 1 / 6) / (13 + 1 / 3 - (15 / 4 * 16 / 5))) = 80.5 )

theorem problem_correct : decimal_to_fraction_eq_80_5 :=
  sorry

end problem_correct_l27_27167


namespace fraction_of_money_left_l27_27492

theorem fraction_of_money_left (m : ℝ) (b : ℝ) (h1 : (1 / 4) * m = (1 / 2) * b) :
  m - b - 50 = m / 2 - 50 → (m - b - 50) / m = 1 / 2 - 50 / m :=
by sorry

end fraction_of_money_left_l27_27492


namespace sam_dimes_now_l27_27163

-- Define the initial number of dimes Sam had
def initial_dimes : ℕ := 9

-- Define the number of dimes Sam gave away
def dimes_given : ℕ := 7

-- State the theorem: The number of dimes Sam has now is 2
theorem sam_dimes_now : initial_dimes - dimes_given = 2 := by
  sorry

end sam_dimes_now_l27_27163


namespace polynomial_representation_l27_27053

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 6)

theorem polynomial_representation (x : ℝ) :
  given_expression x = 6 * x^3 - 4 * x^2 - 26 * x + 20 :=
sorry

end polynomial_representation_l27_27053


namespace speed_of_second_car_l27_27161

theorem speed_of_second_car (s1 s2 s : ℕ) (v1 : ℝ) (h_s1 : s1 = 500) (h_s2 : s2 = 700) 
  (h_s : s = 100) (h_v1 : v1 = 10) : 
  (∃ v2 : ℝ, v2 = 12 ∨ v2 = 16) :=
by 
  sorry

end speed_of_second_car_l27_27161


namespace find_x3_minus_y3_l27_27719

theorem find_x3_minus_y3 {x y : ℤ} (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : x^3 - y^3 = 108 :=
by 
  sorry

end find_x3_minus_y3_l27_27719


namespace gross_profit_percentage_is_correct_l27_27347

def selling_price : ℝ := 28
def wholesale_cost : ℝ := 24.56
def gross_profit : ℝ := selling_price - wholesale_cost

-- Define the expected profit percentage as a constant value.
def expected_profit_percentage : ℝ := 14.01

theorem gross_profit_percentage_is_correct :
  ((gross_profit / wholesale_cost) * 100) = expected_profit_percentage :=
by
  -- Placeholder for proof
  sorry

end gross_profit_percentage_is_correct_l27_27347


namespace quadrant_of_angle_l27_27678

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃! (q : ℕ), q = 2 :=
sorry

end quadrant_of_angle_l27_27678


namespace complement_of_intersection_l27_27596

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem complement_of_intersection (AuB AcB : Set ℤ) :
  (A ∪ B) = AuB ∧ (A ∩ B) = AcB → 
  A ∪ B = ∅ ∨ A ∪ B = AuB → 
  (AuB \ AcB) = {-1, 1} :=
by
  -- Proof construction method placeholder.
  sorry

end complement_of_intersection_l27_27596


namespace determine_s_l27_27180

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem determine_s (s : ℝ) (h : g (-3) s = 0) : s = -192 :=
by
  sorry

end determine_s_l27_27180


namespace rainfall_second_week_l27_27454

theorem rainfall_second_week (x : ℝ) (h1 : x + 1.5 * x = 20) : 1.5 * x = 12 := 
by {
  sorry
}

end rainfall_second_week_l27_27454


namespace zero_intercept_and_distinct_roots_l27_27416

noncomputable def Q (x a' b' c' d' : ℝ) : ℝ := x^4 + a' * x^3 + b' * x^2 + c' * x + d'

theorem zero_intercept_and_distinct_roots (a' b' c' d' : ℝ) (u v w : ℝ) (h_distinct : u ≠ v ∧ v ≠ w ∧ u ≠ w) (h_intercept_at_zero : d' = 0)
(h_Q_form : ∀ x, Q x a' b' c' d' = x * (x - u) * (x - v) * (x - w)) : c' ≠ 0 :=
by
  sorry

end zero_intercept_and_distinct_roots_l27_27416


namespace geometric_sequence_constant_l27_27577

theorem geometric_sequence_constant (a : ℕ → ℝ) (q : ℝ)
    (h1 : ∀ n, a (n+1) = q * a n)
    (h2 : ∀ n, a n > 0)
    (h3 : (a 1 + a 3) * (a 5 + a 7) = 4 * (a 4) ^ 2) :
    ∀ n, a n = a 0 :=
by
  sorry

end geometric_sequence_constant_l27_27577


namespace system_of_equations_solution_l27_27842

theorem system_of_equations_solution :
  ∃ (a b : ℤ), (2 * (2 : ℤ) + b = a ∧ (2 : ℤ) + b = 3 ∧ a = 5 ∧ b = 1) :=
by
  sorry

end system_of_equations_solution_l27_27842


namespace remainder_of_sum_division_l27_27868

theorem remainder_of_sum_division (x y : ℕ) (k m : ℕ) 
  (hx : x = 90 * k + 75) (hy : y = 120 * m + 115) :
  (x + y) % 30 = 10 :=
by sorry

end remainder_of_sum_division_l27_27868


namespace birds_initially_sitting_l27_27942

theorem birds_initially_sitting (initial_birds birds_joined total_birds : ℕ) 
  (h1 : birds_joined = 4) (h2 : total_birds = 6) (h3 : total_birds = initial_birds + birds_joined) : 
  initial_birds = 2 :=
by
  sorry

end birds_initially_sitting_l27_27942


namespace find_quotient_from_conditions_l27_27495

variable (x y : ℕ)
variable (k : ℕ)

theorem find_quotient_from_conditions :
  y - x = 1360 ∧ y = 1614 ∧ y % x = 15 → y / x = 6 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end find_quotient_from_conditions_l27_27495


namespace find_g_l27_27099

theorem find_g (g : ℕ) (h : g > 0) :
  (1 / 3) = ((4 + g * (g - 1)) / ((g + 4) * (g + 3))) → g = 5 :=
by
  intro h_eq
  sorry 

end find_g_l27_27099


namespace find_a_l27_27668

open Real

variable (a : ℝ)

theorem find_a (h : 4 * a + -5 * 3 = 0) : a = 15 / 4 :=
sorry

end find_a_l27_27668


namespace andrew_cookies_per_day_l27_27826

/-- Number of days in May --/
def days_in_may : ℤ := 31

/-- Cost per cookie in dollars --/
def cost_per_cookie : ℤ := 15

/-- Total amount spent by Andrew on cookies in dollars --/
def total_amount_spent : ℤ := 1395

/-- Total number of cookies purchased by Andrew --/
def total_cookies : ℤ := total_amount_spent / cost_per_cookie

/-- Number of cookies purchased per day --/
def cookies_per_day : ℤ := total_cookies / days_in_may

theorem andrew_cookies_per_day : cookies_per_day = 3 := by
  sorry

end andrew_cookies_per_day_l27_27826


namespace onions_left_on_scale_l27_27136

-- Define the given weights and conditions
def total_weight_of_40_onions : ℝ := 7680 -- in grams
def avg_weight_remaining_onions : ℝ := 190 -- grams
def avg_weight_removed_onions : ℝ := 206 -- grams

-- Converting original weight from kg to grams
def original_weight_kg_to_g (w_kg : ℝ) : ℝ := w_kg * 1000

-- Proof problem
theorem onions_left_on_scale (w_kg : ℝ) (n_total : ℕ) (n_removed : ℕ) 
    (total_weight : ℝ) (avg_weight_remaining : ℝ) (avg_weight_removed : ℝ)
    (h1 : original_weight_kg_to_g w_kg = total_weight)
    (h2 : n_total = 40)
    (h3 : n_removed = 5)
    (h4 : avg_weight_remaining = avg_weight_remaining_onions)
    (h5 : avg_weight_removed = avg_weight_removed_onions) : 
    n_total - n_removed = 35 :=
sorry

end onions_left_on_scale_l27_27136


namespace complete_square_rewrite_l27_27810

theorem complete_square_rewrite (j i : ℂ) :
  let c := 8
  let p := (3 * i / 8 : ℂ)
  let q := (137 / 8 : ℂ)
  (8 * j^2 + 6 * i * j + 16 = c * (j + p)^2 + q) →
  q / p = - (137 * i / 3) :=
by
  sorry

end complete_square_rewrite_l27_27810


namespace farmer_has_11_goats_l27_27196

theorem farmer_has_11_goats
  (pigs cows goats : ℕ)
  (h1 : pigs = 2 * cows)
  (h2 : cows = goats + 4)
  (h3 : goats + cows + pigs = 56) :
  goats = 11 := by
  sorry

end farmer_has_11_goats_l27_27196


namespace units_produced_today_l27_27270

theorem units_produced_today (n : ℕ) (X : ℕ) 
  (h1 : n = 9) 
  (h2 : (360 + X) / (n + 1) = 45) 
  (h3 : 40 * n = 360) : 
  X = 90 := 
sorry

end units_produced_today_l27_27270


namespace megan_markers_final_count_l27_27062

theorem megan_markers_final_count :
  let initial_markers := 217
  let robert_gave := 109
  let sarah_took := 35
  let teacher_multiplier := 3
  let final_markers := (initial_markers + robert_gave - sarah_took) * (1 + teacher_multiplier) / 2
  final_markers = 582 :=
by
  let initial_markers := 217
  let robert_gave := 109
  let sarah_took := 35
  let teacher_multiplier := 3
  let final_markers := (initial_markers + robert_gave - sarah_took) * (1 + teacher_multiplier) / 2
  have h : final_markers = 582 := sorry
  exact h

end megan_markers_final_count_l27_27062


namespace determine_a_l27_27421

-- Given conditions
variable {a b : ℝ}
variable (h_neg : a < 0) (h_pos : b > 0) (h_max : ∀ x, -2 ≤ a * sin (b * x) ∧ a * sin (b * x) ≤ 2)

-- Statement to prove
theorem determine_a : a = -2 := by
  sorry

end determine_a_l27_27421


namespace complement_M_l27_27409

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x | (x - 1) * (x - 4) = 0}

theorem complement_M :
  (U \ M) = {2, 3} := by
  sorry

end complement_M_l27_27409


namespace little_john_height_l27_27467

theorem little_john_height :
  let m := 2 
  let cm_to_m := 8 * 0.01
  let mm_to_m := 3 * 0.001
  m + cm_to_m + mm_to_m = 2.083 := 
by
  sorry

end little_john_height_l27_27467


namespace find_sum_invested_l27_27971

noncomputable def sum_invested (interest_difference: ℝ) (rate1: ℝ) (rate2: ℝ) (time: ℝ): ℝ := 
  interest_difference * 100 / (time * (rate1 - rate2))

theorem find_sum_invested :
  let interest_difference := 600
  let rate1 := 18 / 100
  let rate2 := 12 / 100
  let time := 2
  sum_invested interest_difference rate1 rate2 time = 5000 :=
by
  sorry

end find_sum_invested_l27_27971


namespace propositions_true_false_l27_27229

theorem propositions_true_false :
  (∃ x : ℝ, x ^ 3 < 1) ∧ 
  ¬ (∃ x : ℚ, x ^ 2 = 2) ∧ 
  ¬ (∀ x : ℕ, x ^ 3 > x ^ 2) ∧ 
  (∀ x : ℝ, x ^ 2 + 1 > 0) :=
by
  sorry

end propositions_true_false_l27_27229


namespace system1_solution_system2_solution_l27_27297

theorem system1_solution : 
  ∃ (x y : ℤ), 2 * x + 3 * y = -1 ∧ y = 4 * x - 5 ∧ x = 1 ∧ y = -1 := by 
    sorry

theorem system2_solution : 
  ∃ (x y : ℤ), 3 * x + 2 * y = 20 ∧ 4 * x - 5 * y = 19 ∧ x = 6 ∧ y = 1 := by 
    sorry

end system1_solution_system2_solution_l27_27297


namespace trains_crossing_time_l27_27205

theorem trains_crossing_time (length : ℕ) (time1 time2 : ℕ) (h1 : length = 120) (h2 : time1 = 10) (h3 : time2 = 20) :
  (2 * length : ℚ) / (length / time1 + length / time2 : ℚ) = 13.33 :=
by
  sorry

end trains_crossing_time_l27_27205


namespace total_hangers_l27_27339

def pink_hangers : ℕ := 7
def green_hangers : ℕ := 4
def blue_hangers : ℕ := green_hangers - 1
def yellow_hangers : ℕ := blue_hangers - 1

theorem total_hangers :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = 16 := by
  sorry

end total_hangers_l27_27339


namespace problem1_problem2_problem3_l27_27780

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^2 + 0.5 * x

theorem problem1 (h : ∀ x : ℝ, f (x + 1) = f x + x + 1) (h0 : f 0 = 0) : 
  ∀ x : ℝ, f x = 0.5 * x^2 + 0.5 * x := by 
  sorry

noncomputable def g (t : ℝ) : ℝ :=
  if t ≤ -1.5 then 0.5 * t^2 + 1.5 * t + 1
  else if -1.5 < t ∧ t < -0.5 then -1 / 8
  else 0.5 * t^2 + 0.5 * t

theorem problem2 (h : ∀ t : ℝ, g t = min (f (t)) (f (t + 1))) : 
  ∀ t : ℝ, g t = 
    if t ≤ -1.5 then 0.5 * t^2 + 1.5 * t + 1
    else if -1.5 < t ∧ t < -0.5 then -1 / 8
    else 0.5 * t^2 + 0.5 * t := by 
  sorry

theorem problem3 (m : ℝ) : (∀ t : ℝ, g t + m ≥ 0) → m ≥ 1 / 8 := by 
  sorry

end problem1_problem2_problem3_l27_27780


namespace oakwood_team_count_l27_27175

theorem oakwood_team_count :
  let girls := 5
  let boys := 7
  let choose_3_girls := Nat.choose girls 3
  let choose_2_boys := Nat.choose boys 2
  choose_3_girls * choose_2_boys = 210 := by
sorry

end oakwood_team_count_l27_27175


namespace haley_candy_l27_27541

theorem haley_candy (X : ℕ) (h : X - 17 + 19 = 35) : X = 33 :=
by
  sorry

end haley_candy_l27_27541


namespace aquarium_water_l27_27179

theorem aquarium_water (T1 T2 T3 T4 : ℕ) (g w : ℕ) (hT1 : T1 = 8) (hT2 : T2 = 8) (hT3 : T3 = 6) (hT4 : T4 = 6):
  (g = T1 + T2 + T3 + T4) → (w = g * 4) → w = 112 :=
by
  sorry

end aquarium_water_l27_27179


namespace solve_system_l27_27046

theorem solve_system (x y : ℝ) (h1 : 2 * x - y = 0) (h2 : x + 2 * y = 1) : 
  x = 1 / 5 ∧ y = 2 / 5 :=
by
  sorry

end solve_system_l27_27046


namespace johns_raise_percentage_increase_l27_27112

theorem johns_raise_percentage_increase (original_amount new_amount : ℝ) (h_original : original_amount = 60) (h_new : new_amount = 70) :
  ((new_amount - original_amount) / original_amount) * 100 = 16.67 := 
  sorry

end johns_raise_percentage_increase_l27_27112


namespace total_time_before_playing_game_l27_27365

theorem total_time_before_playing_game : 
  ∀ (d i t_t t : ℕ), 
  d = 10 → 
  i = d / 2 → 
  t_t = 3 * (d + i) → 
  t = d + i + t_t → 
  t = 60 := 
by
  intros d i t_t t h1 h2 h3 h4
  sorry

end total_time_before_playing_game_l27_27365


namespace find_x_l27_27929

variable {a b x : ℝ}

-- Defining the given conditions
def is_linear_and_unique_solution (a b : ℝ) : Prop :=
  3 * a + 2 * b = 0 ∧ a ≠ 0

-- The proof problem: prove that x = 1.5, given the conditions.
theorem find_x (ha : is_linear_and_unique_solution a b) : x = 1.5 :=
  sorry

end find_x_l27_27929


namespace nicholas_bottle_caps_l27_27362

theorem nicholas_bottle_caps (initial : ℕ) (additional : ℕ) (final : ℕ) (h1 : initial = 8) (h2 : additional = 85) :
  final = 93 :=
by
  sorry

end nicholas_bottle_caps_l27_27362


namespace find_fraction_l27_27965

variables (a b c : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variable (h : a + b + c = 1)

theorem find_fraction :
  (a^3 + b^3 + c^3) / (a * b * c) = (1 + 3 * (a - b)^2) / (a * b * (1 - a - b)) :=
by
  sorry

end find_fraction_l27_27965


namespace Nora_to_Lulu_savings_ratio_l27_27817

-- Definitions
def L : ℕ := 6
def T (N : ℕ) : Prop := N = 3 * (N / 3)
def total_savings (N : ℕ) : Prop := 6 + N + (N / 3) = 46

-- Theorem statement
theorem Nora_to_Lulu_savings_ratio (N : ℕ) (hN_T : T N) (h_total_savings : total_savings N) :
  N / L = 5 :=
by
  -- Proof will be provided here
  sorry

end Nora_to_Lulu_savings_ratio_l27_27817


namespace area_of_shaded_region_l27_27085

noncomputable def shaded_area (length_in_feet : ℝ) (diameter : ℝ) : ℝ :=
  let length_in_inches := length_in_feet * 12
  let radius := diameter / 2
  let num_semicircles := length_in_inches / diameter
  let num_full_circles := num_semicircles / 2
  let area := num_full_circles * (radius ^ 2 * Real.pi)
  area

theorem area_of_shaded_region : shaded_area 1.5 3 = 13.5 * Real.pi :=
by
  sorry

end area_of_shaded_region_l27_27085


namespace fraction_simplification_l27_27892

theorem fraction_simplification :
  (1 / 330) + (19 / 30) = 7 / 11 :=
by
  sorry

end fraction_simplification_l27_27892


namespace intersection_A_B_l27_27114

def A : Set ℝ := {1, 3, 9, 27}
def B : Set ℝ := {y : ℝ | ∃ x ∈ A, y = Real.log x / Real.log 3}
theorem intersection_A_B : A ∩ B = {1, 3} := 
by
  sorry

end intersection_A_B_l27_27114


namespace barbara_wins_iff_multiple_of_6_l27_27468

-- Define the conditions and the statement to be proved
theorem barbara_wins_iff_multiple_of_6 (n : ℕ) (h : n > 1) :
  (∃ a b : ℕ, a > 0 ∧ b > 1 ∧ (b ∣ a ∨ a ∣ b) ∧ ∀ k ≤ 50, (b + k = n ∨ b - k = n)) ↔ 6 ∣ n :=
sorry

end barbara_wins_iff_multiple_of_6_l27_27468


namespace original_people_count_l27_27321

theorem original_people_count (x : ℕ) 
  (H1 : (x - x / 3) / 2 = 15) : x = 45 := by
  sorry

end original_people_count_l27_27321


namespace expression_value_l27_27537

theorem expression_value (a b : ℕ) (ha : a = 45) (hb : b = 15) : (a + b)^2 - (a^2 + b^2) = 1350 := by 
  sorry

end expression_value_l27_27537


namespace total_toothpicks_correct_l27_27101

-- Define the number of vertical lines and toothpicks in them
def num_vertical_lines : ℕ := 41
def num_toothpicks_per_vertical_line : ℕ := 20
def vertical_toothpicks : ℕ := num_vertical_lines * num_toothpicks_per_vertical_line

-- Define the number of horizontal lines and toothpicks in them
def num_horizontal_lines : ℕ := 21
def num_toothpicks_per_horizontal_line : ℕ := 40
def horizontal_toothpicks : ℕ := num_horizontal_lines * num_toothpicks_per_horizontal_line

-- Define the dimensions of the triangle
def triangle_base : ℕ := 20
def triangle_height : ℕ := 20
def triangle_hypotenuse : ℕ := 29 -- approximated

-- Total toothpicks in the triangle
def triangle_toothpicks : ℕ := triangle_height + triangle_hypotenuse

-- Total toothpicks used in the structure
def total_toothpicks : ℕ := vertical_toothpicks + horizontal_toothpicks + triangle_toothpicks

-- Theorem to prove the total number of toothpicks used is 1709
theorem total_toothpicks_correct : total_toothpicks = 1709 := by
  sorry

end total_toothpicks_correct_l27_27101


namespace tricycles_count_l27_27712

-- Define the variables for number of bicycles, tricycles, and scooters.
variables (b t s : ℕ)

-- Define the total number of children and total number of wheels conditions.
def children_condition := b + t + s = 10
def wheels_condition := 2 * b + 3 * t + 2 * s = 27

-- Prove that number of tricycles t is 4 under these conditions.
theorem tricycles_count : children_condition b t s → wheels_condition b t s → t = 4 := by
  sorry

end tricycles_count_l27_27712


namespace no_nat_k_divides_7_l27_27933

theorem no_nat_k_divides_7 (k : ℕ) : ¬ 7 ∣ (2^(2*k - 1) + 2^k + 1) := 
sorry

end no_nat_k_divides_7_l27_27933


namespace range_of_x_l27_27728

noncomputable def f : ℝ → ℝ := sorry -- Define the function f

variable (f_increasing : ∀ x y, x < y → f x < f y) -- f is increasing
variable (f_at_2 : f 2 = 0) -- f(2) = 0

theorem range_of_x (x : ℝ) : f (x - 2) > 0 ↔ x > 4 :=
by
  sorry

end range_of_x_l27_27728


namespace triangle_base_second_l27_27755

theorem triangle_base_second (base1 height1 height2 : ℝ) 
  (h_base1 : base1 = 15) (h_height1 : height1 = 12) (h_height2 : height2 = 18) :
  let area1 := (base1 * height1) / 2
  let area2 := 2 * area1
  let base2 := (2 * area2) / height2
  base2 = 20 :=
by
  sorry

end triangle_base_second_l27_27755


namespace arc_length_correct_l27_27605

noncomputable def chord_length := 2
noncomputable def central_angle := 2
noncomputable def half_chord_length := 1
noncomputable def radius := 1 / Real.sin 1
noncomputable def arc_length := 2 * radius

theorem arc_length_correct :
  arc_length = 2 / Real.sin 1 := by
sorry

end arc_length_correct_l27_27605


namespace largest_a_l27_27588

theorem largest_a (a b : ℕ) (x : ℕ) (h_a_range : 2 < a ∧ a < x) (h_b_range : 4 < b ∧ b < 13) (h_fraction_range : 7 * a = 57) : a = 8 :=
sorry

end largest_a_l27_27588


namespace terry_current_age_l27_27574

theorem terry_current_age (T : ℕ) (nora_current_age : ℕ) (h1 : nora_current_age = 10)
  (h2 : T + 10 = 4 * nora_current_age) : T = 30 :=
by
  sorry

end terry_current_age_l27_27574


namespace painted_cubes_count_l27_27324

def total_painted_cubes : ℕ := 8 + 48

theorem painted_cubes_count : total_painted_cubes = 56 :=
by 
  -- Step 1: Define the number of cubes with 3 faces painted (8 corners)
  let corners := 8
  -- Step 2: Calculate the number of edge cubes with 2 faces painted
  let edge_middle_cubes_per_edge := 2
  let edges := 12
  let edge_cubes := edge_middle_cubes_per_edge * edges -- this should be 24
  -- Step 3: Calculate the number of face-interior cubes with 2 faces painted
  let face_cubes_per_face := 4
  let faces := 6
  let face_cubes := face_cubes_per_face * faces -- this should be 24
  -- Step 4: Sum them up to get total cubes with at least two faces painted
  let total_cubes := corners + edge_cubes + face_cubes
  show total_cubes = total_painted_cubes
  sorry

end painted_cubes_count_l27_27324


namespace number_of_cooks_l27_27359

variable (C W : ℕ)

-- Conditions
def initial_ratio := 3 * W = 8 * C
def new_ratio := 4 * C = W + 12

theorem number_of_cooks (h1 : initial_ratio W C) (h2 : new_ratio W C) : C = 9 := by
  sorry

end number_of_cooks_l27_27359


namespace find_m_l27_27910

theorem find_m (m : ℕ) (h₁ : 0 < m) : 
  144^5 + 91^5 + 56^5 + 19^5 = m^5 → m = 147 := by
  -- Mathematically, we know the sum of powers equals a fifth power of 147
  -- 144^5 = 61917364224
  -- 91^5 = 6240321451
  -- 56^5 = 550731776
  -- 19^5 = 2476099
  -- => 61917364224 + 6240321451 + 550731776 + 2476099 = 68897423550
  -- Find the nearest  m such that m^5 = 68897423550
  sorry

end find_m_l27_27910


namespace regular_octagon_side_length_sum_l27_27667

theorem regular_octagon_side_length_sum (s : ℝ) (h₁ : s = 2.3) (h₂ : 1 = 100) : 
  8 * (s * 100) = 1840 :=
by
  sorry

end regular_octagon_side_length_sum_l27_27667


namespace hyperbola_eccentricity_sqrt2_l27_27222

noncomputable def isHyperbolaPerpendicularAsymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  let asymptote1 := (1/a : ℝ)
  let asymptote2 := (-1/b : ℝ)
  asymptote1 * asymptote2 = -1

theorem hyperbola_eccentricity_sqrt2 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  isHyperbolaPerpendicularAsymptotes a b ha hb →
  let e := Real.sqrt (1 + (b^2 / a^2))
  e = Real.sqrt 2 :=
by
  intro h
  sorry

end hyperbola_eccentricity_sqrt2_l27_27222


namespace determine_angle_A_l27_27079

-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
def sin_rule_condition (a b c A B C : ℝ) : Prop :=
  (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C

-- The proof statement
theorem determine_angle_A (a b c A B C : ℝ) (h : sin_rule_condition a b c A B C) : A = π / 3 :=
  sorry

end determine_angle_A_l27_27079


namespace option_d_correct_l27_27272

variable (a b : ℝ)

theorem option_d_correct : (-a^3)^4 = a^(12) := by sorry

end option_d_correct_l27_27272


namespace alpha_beta_sum_l27_27829

theorem alpha_beta_sum (α β : ℝ) (h1 : α^3 - 3 * α^2 + 5 * α - 17 = 0) (h2 : β^3 - 3 * β^2 + 5 * β + 11 = 0) : α + β = 2 := 
by
  sorry

end alpha_beta_sum_l27_27829


namespace schedule_arrangements_l27_27859

-- Define the initial setup of the problem
def subjects : List String := ["Chinese", "Mathematics", "English", "Physics", "Chemistry", "Biology"]

def periods_morning : List String := ["P1", "P2", "P3", "P4"]
def periods_afternoon : List String := ["P5", "P6", "P7"]

-- Define the constraints
def are_consecutive (subj1 subj2 : String) : Bool := 
  (subj1 = "Chinese" ∧ subj2 = "Mathematics") ∨ 
  (subj1 = "Mathematics" ∧ subj2 = "Chinese")

def can_schedule_max_one_period (subject : String) : Bool :=
  subject = "English" ∨ subject = "Physics" ∨ subject = "Chemistry" ∨ subject = "Biology"

-- Define the math problem as a proof in Lean
theorem schedule_arrangements : 
  ∃ n : Nat, n = 336 :=
by
  -- The detailed proof steps would go here
  sorry

end schedule_arrangements_l27_27859


namespace number_multiplied_by_3_l27_27007

variable (A B C D E : ℝ) -- Declare the five numbers

theorem number_multiplied_by_3 (h1 : (A + B + C + D + E) / 5 = 6.8) 
    (h2 : ∃ X : ℝ, (A + B + C + D + E + 2 * X) / 5 = 9.2) : 
    ∃ X : ℝ, X = 6 := 
  sorry

end number_multiplied_by_3_l27_27007


namespace total_cost_verification_l27_27797

-- Conditions given in the problem
def holstein_cost : ℕ := 260
def jersey_cost : ℕ := 170
def num_hearts_on_card : ℕ := 4
def num_cards_in_deck : ℕ := 52
def cow_ratio_holstein : ℕ := 3
def cow_ratio_jersey : ℕ := 2
def sales_tax : ℝ := 0.05
def transport_cost_per_cow : ℕ := 20

def num_hearts_in_deck := num_cards_in_deck
def total_num_cows := 2 * num_hearts_in_deck
def total_parts_ratio := cow_ratio_holstein + cow_ratio_jersey

-- Total number of cows calculated 
def num_holstein_cows : ℕ := (cow_ratio_holstein * total_num_cows) / total_parts_ratio
def num_jersey_cows : ℕ := (cow_ratio_jersey * total_num_cows) / total_parts_ratio

-- Cost calculations
def holstein_total_cost := num_holstein_cows * holstein_cost
def jersey_total_cost := num_jersey_cows * jersey_cost
def total_cost_before_tax_and_transport := holstein_total_cost + jersey_total_cost
def total_sales_tax := total_cost_before_tax_and_transport * sales_tax
def total_transport_cost := total_num_cows * transport_cost_per_cow
def final_total_cost := total_cost_before_tax_and_transport + total_sales_tax + total_transport_cost

-- Lean statement to prove the result
theorem total_cost_verification : final_total_cost = 26324.50 := by sorry

end total_cost_verification_l27_27797


namespace number_of_planting_methods_l27_27257

theorem number_of_planting_methods :
  let vegetables := ["cucumbers", "cabbages", "rape", "flat beans"]
  let plots := ["plot1", "plot2", "plot3"]
  (∀ v ∈ vegetables, v = "cucumbers") →
  (∃! n : ℕ, n = 18)
:= by
  sorry

end number_of_planting_methods_l27_27257


namespace lcm_18_35_l27_27732

-- Given conditions: Prime factorizations of 18 and 35
def factorization_18 : Prop := (18 = 2^1 * 3^2)
def factorization_35 : Prop := (35 = 5^1 * 7^1)

-- The goal is to prove that the least common multiple of 18 and 35 is 630
theorem lcm_18_35 : factorization_18 ∧ factorization_35 → Nat.lcm 18 35 = 630 := by
  sorry -- Proof to be filled in

end lcm_18_35_l27_27732


namespace total_expenditure_correct_l27_27708

-- Define the weekly costs based on the conditions
def cost_white_bread : Float := 2 * 3.50
def cost_baguette : Float := 1.50
def cost_sourdough_bread : Float := 2 * 4.50
def cost_croissant : Float := 2.00

-- Total weekly cost calculation
def weekly_cost : Float := cost_white_bread + cost_baguette + cost_sourdough_bread + cost_croissant

-- Total cost over 4 weeks
def total_cost_4_weeks (weeks : Float) : Float := weekly_cost * weeks

-- The assertion that needs to be proved
theorem total_expenditure_correct :
  total_cost_4_weeks 4 = 78.00 := by
  sorry

end total_expenditure_correct_l27_27708


namespace polynomial_remainder_distinct_l27_27154

open Nat

theorem polynomial_remainder_distinct (a b c p : ℕ) (hp : Nat.Prime p) (hp_ge5 : p ≥ 5)
  (ha : Nat.gcd a p = 1) (hb : b^2 ≡ 3 * a * c [MOD p]) (hp_mod3 : p ≡ 2 [MOD 3]) :
  ∀ m1 m2 : ℕ, m1 < p ∧ m2 < p → m1 ≠ m2 → (a * m1^3 + b * m1^2 + c * m1) % p ≠ (a * m2^3 + b * m2^2 + c * m2) % p := 
by
  sorry

end polynomial_remainder_distinct_l27_27154


namespace grid_blue_probability_l27_27300

-- Define the problem in Lean
theorem grid_blue_probability :
  let n := 4
  let p_tile_blue := 1 / 2
  let invariant_prob := (p_tile_blue ^ (n / 2))
  let pair_prob := (p_tile_blue * p_tile_blue)
  let total_pairs := (n * n / 2 - n / 2)
  let final_prob := (invariant_prob ^ 2) * (pair_prob ^ total_pairs)
  final_prob = 1 / 65536 := by
  sorry

end grid_blue_probability_l27_27300


namespace speed_of_man_correct_l27_27763

noncomputable def speed_of_man_in_kmph (train_speed_kmph : ℝ) (train_length_m : ℝ) (time_pass_sec : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed_mps := (train_length_m / time_pass_sec)
  let man_speed_mps := relative_speed_mps - train_speed_mps
  man_speed_mps * 3600 / 1000

theorem speed_of_man_correct : 
  speed_of_man_in_kmph 77.993280537557 140 6 = 6.00871946444388 := 
by simp [speed_of_man_in_kmph]; sorry

end speed_of_man_correct_l27_27763


namespace jack_paid_20_l27_27219

-- Define the conditions
def numberOfSandwiches : Nat := 3
def costPerSandwich : Nat := 5
def changeReceived : Nat := 5

-- Define the total cost
def totalCost : Nat := numberOfSandwiches * costPerSandwich

-- Define the amount paid
def amountPaid : Nat := totalCost + changeReceived

-- Prove that the amount paid is 20
theorem jack_paid_20 : amountPaid = 20 := by
  -- You may assume the steps and calculations here, only providing the statement
  sorry

end jack_paid_20_l27_27219


namespace min_value_expression_l27_27723

theorem min_value_expression (a b : ℝ) : ∃ v : ℝ, ∀ (a b : ℝ), (a^2 + a * b + b^2 - a - 2 * b) ≥ v ∧ v = -1 :=
by
  sorry

end min_value_expression_l27_27723


namespace double_variable_for_1600_percent_cost_l27_27874

theorem double_variable_for_1600_percent_cost (t b0 b1 : ℝ) (h : t ≠ 0) :
    (t * b1^4 = 16 * t * b0^4) → b1 = 2 * b0 :=
by
sorry

end double_variable_for_1600_percent_cost_l27_27874


namespace total_cost_is_correct_l27_27986

-- Definitions based on conditions
def bedroomDoorCount : ℕ := 3
def outsideDoorCount : ℕ := 2
def outsideDoorCost : ℕ := 20
def bedroomDoorCost : ℕ := outsideDoorCost / 2

-- Total costs calculations
def totalBedroomCost : ℕ := bedroomDoorCount * bedroomDoorCost
def totalOutsideCost : ℕ := outsideDoorCount * outsideDoorCost
def totalCost : ℕ := totalBedroomCost + totalOutsideCost

-- Proof statement
theorem total_cost_is_correct : totalCost = 70 := 
by
  sorry

end total_cost_is_correct_l27_27986


namespace minimum_common_ratio_l27_27024

theorem minimum_common_ratio (a : ℕ) (n : ℕ) (q : ℝ) (h_pos : ∀ i, i < n → 0 < a * q^i) (h_geom : ∀ i j, i < j → a * q^i < a * q^j) (h_q : 1 < q ∧ q < 2) : q = 6 / 5 :=
by
  sorry

end minimum_common_ratio_l27_27024


namespace total_animals_is_200_l27_27122

-- Definitions for the conditions
def num_cows : Nat := 40
def num_sheep : Nat := 56
def num_goats : Nat := 104

-- The theorem to prove the total number of animals is 200
theorem total_animals_is_200 : num_cows + num_sheep + num_goats = 200 := by
  sorry

end total_animals_is_200_l27_27122


namespace find_wall_width_l27_27337

-- Define the volume of one brick
def volume_of_one_brick : ℚ := 100 * 11.25 * 6

-- Define the total number of bricks
def number_of_bricks : ℕ := 1600

-- Define the volume of all bricks combined
def total_volume_of_bricks : ℚ := volume_of_one_brick * number_of_bricks

-- Define dimensions of the wall
def wall_height : ℚ := 800 -- in cm (since 8 meters = 800 cm)
def wall_depth : ℚ := 22.5 -- in cm

-- Theorem to prove the width of the wall
theorem find_wall_width : ∃ width : ℚ, total_volume_of_bricks = wall_height * width * wall_depth ∧ width = 600 :=
by
  -- skipping the actual proof
  sorry

end find_wall_width_l27_27337


namespace ellipse_x_intercept_other_l27_27433

noncomputable def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 3), (4, 0))
noncomputable def sum_of_distances : ℝ := 7
noncomputable def first_intercept : (ℝ × ℝ) := (0, 0)

theorem ellipse_x_intercept_other 
  (foci : (ℝ × ℝ) × (ℝ × ℝ))
  (sum_of_distances : ℝ)
  (first_intercept : (ℝ × ℝ))
  (hx : foci = ((0, 3), (4, 0)))
  (d_sum : sum_of_distances = 7)
  (intercept : first_intercept = (0, 0)) :
  ∃ (x : ℝ), x > 0 ∧ ((x, 0) = (56 / 11, 0)) := 
sorry

end ellipse_x_intercept_other_l27_27433


namespace unique_zero_of_f_inequality_of_x1_x2_l27_27705

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

theorem unique_zero_of_f (a : ℝ) (h : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem inequality_of_x1_x2 (a x1 x2 : ℝ) (h : f a x1 = g a x1 - g a x2) (hₐ: a ≥ 0) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end unique_zero_of_f_inequality_of_x1_x2_l27_27705


namespace option_C_is_quadratic_l27_27937

theorem option_C_is_quadratic : ∀ (x : ℝ), (x = x^2) ↔ (∃ (a b c : ℝ), a ≠ 0 ∧ a*x^2 + b*x + c = 0) := 
by
  sorry

end option_C_is_quadratic_l27_27937


namespace solve_system_l27_27086

theorem solve_system :
  ∃ x y : ℝ, (x^2 * y + x * y^2 + 3 * x + 3 * y + 24 = 0) ∧ 
              (x^3 * y - x * y^3 + 3 * x^2 - 3 * y^2 - 48 = 0) ∧ 
              (x = -3 ∧ y = -1) :=
  sorry

end solve_system_l27_27086


namespace problem_f_neg2_equals_2_l27_27751

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem problem_f_neg2_equals_2 (f : ℝ → ℝ) (b : ℝ) 
  (h_odd : is_odd_function f)
  (h_def : ∀ x : ℝ, x ≥ 0 → f x = x^2 - 3 * x + b) 
  (h_b : b = 0) :
  f (-2) = 2 :=
by
  sorry

end problem_f_neg2_equals_2_l27_27751


namespace remainder_mod_105_l27_27663

theorem remainder_mod_105 (x : ℤ) 
  (h1 : 3 + x ≡ 4 [ZMOD 27])
  (h2 : 5 + x ≡ 9 [ZMOD 125])
  (h3 : 7 + x ≡ 25 [ZMOD 343]) :
  x % 105 = 4 :=
  sorry

end remainder_mod_105_l27_27663


namespace total_carrots_l27_27878

-- Define the number of carrots grown by Sally and Fred
def sally_carrots := 6
def fred_carrots := 4

-- Theorem: The total number of carrots grown by Sally and Fred
theorem total_carrots : sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l27_27878


namespace red_balls_unchanged_l27_27527

-- Definitions: 
def initial_red_balls : ℕ := 3
def initial_blue_balls : ℕ := 2
def initial_yellow_balls : ℕ := 5

def remove_blue_ball (blue_balls : ℕ) : ℕ :=
  if blue_balls > 0 then blue_balls - 1 else blue_balls

-- Condition after one blue ball is removed
def blue_balls_after_removal := remove_blue_ball initial_blue_balls

-- Prove that the number of red balls remain unchanged
theorem red_balls_unchanged : initial_red_balls = 3 :=
by
  sorry

end red_balls_unchanged_l27_27527


namespace perimeter_division_l27_27034

-- Define the given conditions
def is_pentagon (n : ℕ) : Prop := n = 5
def side_length (s : ℕ) : Prop := s = 25
def perimeter (P : ℕ) (n s : ℕ) : Prop := P = n * s

-- Define the Lean statement to prove
theorem perimeter_division (n s P x : ℕ) 
  (h1 : is_pentagon n) 
  (h2 : side_length s) 
  (h3 : perimeter P n s) 
  (h4 : P = 125) 
  (h5 : s = 25) : 
  P / x = s → x = 5 := 
by
  sorry

end perimeter_division_l27_27034


namespace fraction_neither_cable_nor_vcr_l27_27831

variable (T : ℕ)
variable (units_with_cable : ℕ := T / 5)
variable (units_with_vcrs : ℕ := T / 10)
variable (units_with_cable_and_vcrs : ℕ := (T / 5) / 3)

theorem fraction_neither_cable_nor_vcr (T : ℕ)
  (h1 : units_with_cable = T / 5)
  (h2 : units_with_vcrs = T / 10)
  (h3 : units_with_cable_and_vcrs = (units_with_cable / 3)) :
  (T - (units_with_cable + (units_with_vcrs - units_with_cable_and_vcrs))) / T = 7 / 10 := 
by
  sorry

end fraction_neither_cable_nor_vcr_l27_27831


namespace total_payment_correct_l27_27622

def payment_X (payment_Y : ℝ) : ℝ := 1.2 * payment_Y
def payment_Y : ℝ := 254.55
def total_payment (payment_X payment_Y : ℝ) : ℝ := payment_X + payment_Y

theorem total_payment_correct :
  total_payment (payment_X payment_Y) payment_Y = 560.01 :=
by
  sorry

end total_payment_correct_l27_27622


namespace total_cost_is_83_50_l27_27554

-- Definitions according to the conditions
def cost_adult_ticket : ℝ := 5.50
def cost_child_ticket : ℝ := 3.50
def total_tickets : ℝ := 21
def adult_tickets : ℝ := 5
def child_tickets : ℝ := total_tickets - adult_tickets

-- Total cost calculation based on the conditions
def cost_adult_total : ℝ := adult_tickets * cost_adult_ticket
def cost_child_total : ℝ := child_tickets * cost_child_ticket
def total_cost : ℝ := cost_adult_total + cost_child_total

-- The theorem to prove that the total cost is $83.50
theorem total_cost_is_83_50 : total_cost = 83.50 := by
  sorry

end total_cost_is_83_50_l27_27554


namespace distance_focus_directrix_l27_27726

theorem distance_focus_directrix (y x p : ℝ) (h : y^2 = 4 * x) (hp : 2 * p = 4) : p = 2 :=
by sorry

end distance_focus_directrix_l27_27726


namespace factorial_sum_simplify_l27_27896

theorem factorial_sum_simplify :
  7 * (Nat.factorial 7) + 5 * (Nat.factorial 5) + 3 * (Nat.factorial 3) + (Nat.factorial 3) = 35904 :=
by
  sorry

end factorial_sum_simplify_l27_27896


namespace mul_112_54_l27_27005

theorem mul_112_54 : 112 * 54 = 6048 :=
by
  sorry

end mul_112_54_l27_27005


namespace koala_fiber_intake_l27_27988

theorem koala_fiber_intake (x : ℝ) (h : 0.30 * x = 12) : x = 40 := 
sorry

end koala_fiber_intake_l27_27988


namespace hendecagon_diagonals_l27_27607

-- Define the number of sides n of the hendecagon
def n : ℕ := 11

-- Define the formula for calculating the number of diagonals in an n-sided polygon
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem that there are 44 diagonals in a hendecagon
theorem hendecagon_diagonals : diagonals n = 44 :=
by
  -- Proof is skipped using sorry
  sorry

end hendecagon_diagonals_l27_27607


namespace slope_of_line_through_midpoints_l27_27288

theorem slope_of_line_through_midpoints :
  let P₁ := (1, 2)
  let P₂ := (3, 8)
  let P₃ := (4, 3)
  let P₄ := (7, 9)
  let M₁ := ( (P₁.1 + P₂.1)/2, (P₁.2 + P₂.2)/2 )
  let M₂ := ( (P₃.1 + P₄.1)/2, (P₃.2 + P₄.2)/2 )
  let slope := (M₂.2 - M₁.2) / (M₂.1 - M₁.1)
  slope = 2/7 :=
by
  sorry

end slope_of_line_through_midpoints_l27_27288


namespace smallest_integer_in_range_l27_27233

theorem smallest_integer_in_range :
  ∃ n : ℕ, 
  1 < n ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 ∧ 
  90 < n ∧ n < 119 :=
sorry

end smallest_integer_in_range_l27_27233


namespace horse_cow_difference_l27_27259

def initial_conditions (h c : ℕ) : Prop :=
  4 * c = h

def transaction (h c : ℕ) : Prop :=
  (h - 15) * 7 = (c + 15) * 13

def final_difference (h c : ℕ) : Prop := 
  h - 15 - (c + 15) = 30

theorem horse_cow_difference (h c : ℕ) (hc : initial_conditions h c) (ht : transaction h c) : final_difference h c :=
    by
      sorry

end horse_cow_difference_l27_27259


namespace intersection_complement_l27_27105

universe u

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 2}
def complement_U_B : Set ℤ := {x ∈ U | x ∉ B}

theorem intersection_complement :
  A ∩ complement_U_B = {0, 1} :=
by
  sorry

end intersection_complement_l27_27105


namespace primer_cost_before_discount_l27_27950

theorem primer_cost_before_discount (primer_cost_after_discount : ℝ) (paint_cost : ℝ) (total_cost : ℝ) 
  (rooms : ℕ) (primer_discount : ℝ) (paint_cost_per_gallon : ℝ) :
  (primer_cost_after_discount = total_cost - (rooms * paint_cost_per_gallon)) →
  (rooms * (primer_cost - primer_discount * primer_cost) = primer_cost_after_discount) →
  primer_cost = 30 := by
  sorry

end primer_cost_before_discount_l27_27950


namespace jenny_mother_age_l27_27213

theorem jenny_mother_age:
  (∀ x : ℕ, (50 + x = 2 * (10 + x)) → (2010 + x = 2040)) :=
by
  sorry

end jenny_mother_age_l27_27213


namespace length_of_OP_is_sqrt_200_div_3_l27_27407

open Real

def square (a : ℝ) := a * a

theorem length_of_OP_is_sqrt_200_div_3 (KL MO MP OP : ℝ) (h₁ : KL = 10)
  (h₂: MO = MP) (h₃: square (10) = 100)
  (h₄ : 1 / 6 * 100 = 1 / 2 * (MO * MP)) : OP = sqrt (200/3) :=
by
  sorry

end length_of_OP_is_sqrt_200_div_3_l27_27407


namespace replace_asterisk_l27_27098

theorem replace_asterisk (x : ℕ) (h : (42 / 21) * (42 / x) = 1) : x = 84 := by
  sorry

end replace_asterisk_l27_27098


namespace cheryl_mms_eaten_l27_27127

variable (initial_mms : ℕ) (mms_after_dinner : ℕ) (mms_given_to_sister : ℕ) (total_mms_after_lunch : ℕ)

theorem cheryl_mms_eaten (h1 : initial_mms = 25)
                         (h2 : mms_after_dinner = 5)
                         (h3 : mms_given_to_sister = 13)
                         (h4 : total_mms_after_lunch = initial_mms - mms_after_dinner - mms_given_to_sister) :
                         total_mms_after_lunch = 7 :=
by sorry

end cheryl_mms_eaten_l27_27127


namespace relationship_between_a_b_c_l27_27963

theorem relationship_between_a_b_c :
  let m := 2
  let n := 3
  let f (x : ℝ) := x^3
  let a := f (Real.sqrt 3 / 3)
  let b := f (Real.log Real.pi)
  let c := f (Real.sqrt 2 / 2)
  a < c ∧ c < b :=
by
  sorry

end relationship_between_a_b_c_l27_27963


namespace correct_substitution_l27_27992

theorem correct_substitution (x y : ℤ) (h1 : x = 3 * y - 1) (h2 : x - 2 * y = 4) :
  3 * y - 1 - 2 * y = 4 :=
by
  sorry

end correct_substitution_l27_27992


namespace alan_has_5_20_cent_coins_l27_27938

theorem alan_has_5_20_cent_coins
  (a b c : ℕ)
  (h1 : a + b + c = 20)
  (h2 : ((400 - 15 * a - 10 * b) / 5) + 1 = 24) :
  c = 5 :=
by
  sorry

end alan_has_5_20_cent_coins_l27_27938


namespace radius_range_of_circle_l27_27403

theorem radius_range_of_circle (r : ℝ) :
  (∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 → 
  (abs (4*x - 3*y - 2) = 1)) →
  4 < r ∧ r < 6 :=
by
  sorry

end radius_range_of_circle_l27_27403


namespace acid_solution_l27_27857

theorem acid_solution (m x : ℝ) (h1 : 0 < m) (h2 : m > 50)
  (h3 : (m / 100) * m = (m - 20) / 100 * (m + x)) : x = 20 * m / (m + 20) := 
sorry

end acid_solution_l27_27857


namespace cube_side_length_increase_20_percent_l27_27493

variable {s : ℝ} (initial_side_length_increase : ℝ) (percentage_surface_area_increase : ℝ) (percentage_volume_increase : ℝ)
variable (new_surface_area : ℝ) (new_volume : ℝ)

theorem cube_side_length_increase_20_percent :
  ∀ (s : ℝ),
  (initial_side_length_increase = 1.2 * s) →
  (new_surface_area = 6 * (1.2 * s)^2) →
  (new_volume = (1.2 * s)^3) →
  (percentage_surface_area_increase = ((new_surface_area - (6 * s^2)) / (6 * s^2)) * 100) →
  (percentage_volume_increase = ((new_volume - s^3) / s^3) * 100) →
  5 * (percentage_volume_increase - percentage_surface_area_increase) = 144 := by
  sorry

end cube_side_length_increase_20_percent_l27_27493


namespace minimum_value_of_f_l27_27217

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 2 → f x ≥ 4) ∧ (∃ x : ℝ, x > 2 ∧ f x = 4) :=
by {
  sorry
}

end minimum_value_of_f_l27_27217


namespace factorize_perfect_square_l27_27532

variable (a b : ℤ)

theorem factorize_perfect_square :
  a^2 + 6 * a * b + 9 * b^2 = (a + 3 * b)^2 := 
sorry

end factorize_perfect_square_l27_27532


namespace payment_ways_l27_27828

-- Define basic conditions and variables
variables {x y z : ℕ}

-- Define the main problem as a Lean statement
theorem payment_ways : 
  ∃ (n : ℕ), n = 9 ∧ 
             (∀ x y z : ℕ, 
              x + y + z ≤ 10 ∧ 
              x + 2 * y + 5 * z = 18 ∧ 
              x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
              (x > 0 ∨ y > 0) ∧ (y > 0 ∨ z > 0) ∧ (z > 0 ∨ x > 0) → 
              n = 9) := 
sorry

end payment_ways_l27_27828


namespace arithmetic_sequence_8th_term_is_71_l27_27234

def arithmetic_sequence_8th_term (a d : ℤ) : ℤ := a + 7 * d

theorem arithmetic_sequence_8th_term_is_71 (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  arithmetic_sequence_8th_term a d = 71 :=
by
  sorry

end arithmetic_sequence_8th_term_is_71_l27_27234


namespace coefficient_x4_expansion_eq_7_l27_27139

theorem coefficient_x4_expansion_eq_7 (a : ℝ) : 
  (∀ r : ℕ, 8 - (4 * r) / 3 = 4 → (a ^ r) * (Nat.choose 8 r) = 7) → a = 1 / 2 :=
by
  sorry

end coefficient_x4_expansion_eq_7_l27_27139


namespace average_age_is_correct_l27_27845

-- Define the conditions
def num_men : ℕ := 6
def num_women : ℕ := 9
def average_age_men : ℕ := 57
def average_age_women : ℕ := 52
def total_age_men : ℕ := num_men * average_age_men
def total_age_women : ℕ := num_women * average_age_women
def total_age : ℕ := total_age_men + total_age_women
def total_people : ℕ := num_men + num_women
def average_age_group : ℕ := total_age / total_people

-- The proof will require showing average_age_group is 54, left as sorry.
theorem average_age_is_correct : average_age_group = 54 := sorry

end average_age_is_correct_l27_27845


namespace volume_ratio_proof_l27_27893

-- Definitions:
def height_ratio := 2 / 3
def volume_ratio (r : ℚ) := r^3
def small_pyramid_volume_ratio := volume_ratio height_ratio
def frustum_volume_ratio := 1 - small_pyramid_volume_ratio
def volume_ratio_small_to_frustum (v_small v_frustum : ℚ) := v_small / v_frustum

-- Lean 4 Statement:
theorem volume_ratio_proof
  (height_ratio : ℚ := 2 / 3)
  (small_pyramid_volume_ratio : ℚ := volume_ratio height_ratio)
  (frustum_volume_ratio : ℚ := 1 - small_pyramid_volume_ratio)
  (v_orig : ℚ) :
  volume_ratio_small_to_frustum (small_pyramid_volume_ratio * v_orig) (frustum_volume_ratio * v_orig) = 8 / 19 :=
by
  sorry

end volume_ratio_proof_l27_27893


namespace function_equiv_proof_l27_27908

noncomputable def function_solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) = f x * y

theorem function_equiv_proof : ∀ f : ℝ → ℝ,
  function_solution f ↔ (∀ x : ℝ, f x = 0 ∨ f x = x ∨ f x = -x) := 
sorry

end function_equiv_proof_l27_27908


namespace ratio_of_lost_diaries_to_total_diaries_l27_27957

theorem ratio_of_lost_diaries_to_total_diaries 
  (original_diaries : ℕ)
  (bought_diaries : ℕ)
  (current_diaries : ℕ)
  (h1 : original_diaries = 8)
  (h2 : bought_diaries = 2 * original_diaries)
  (h3 : current_diaries = 18) :
  (original_diaries + bought_diaries - current_diaries) / gcd (original_diaries + bought_diaries - current_diaries) (original_diaries + bought_diaries) 
  = 1 / 4 :=
by
  sorry

end ratio_of_lost_diaries_to_total_diaries_l27_27957


namespace minimum_production_volume_to_avoid_loss_l27_27060

open Real

-- Define the cost function
def cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * (x ^ 2)

-- Define the revenue function
def revenue (x : ℕ) : ℝ := 25 * x

-- Condition: 0 < x < 240 and x ∈ ℕ (naturals greater than 0)
theorem minimum_production_volume_to_avoid_loss (x : ℕ) (hx1 : 0 < x) (hx2 : x < 240) (hx3 : x ∈ (Set.Ioi 0)) :
  revenue x ≥ cost x ↔ x ≥ 150 :=
by
  sorry

end minimum_production_volume_to_avoid_loss_l27_27060


namespace number_of_marbles_l27_27562

theorem number_of_marbles (T : ℕ) (h1 : 12 ≤ T) : 
  (T - 12) * (T - 12) * 16 = 9 * T * T → T = 48 :=
by
  -- Proof omitted
  sorry

end number_of_marbles_l27_27562


namespace Dana_pencils_equals_combined_l27_27998

-- Definitions based on given conditions
def pencils_Jayden : ℕ := 20
def pencils_Marcus (pencils_Jayden : ℕ) : ℕ := pencils_Jayden / 2
def pencils_Dana (pencils_Jayden : ℕ) : ℕ := pencils_Jayden + 15
def pencils_Ella (pencils_Marcus : ℕ) : ℕ := 3 * pencils_Marcus - 5
def combined_pencils (pencils_Marcus : ℕ) (pencils_Ella : ℕ) : ℕ := pencils_Marcus + pencils_Ella

-- Theorem to prove:
theorem Dana_pencils_equals_combined (pencils_Jayden : ℕ := 20) : 
  pencils_Dana pencils_Jayden = combined_pencils (pencils_Marcus pencils_Jayden) (pencils_Ella (pencils_Marcus pencils_Jayden)) := by
  sorry

end Dana_pencils_equals_combined_l27_27998


namespace women_in_the_minority_l27_27418

theorem women_in_the_minority (total_employees : ℕ) (female_employees : ℕ) (h : female_employees < total_employees * 20 / 100) : 
  (female_employees < total_employees / 2) :=
by
  sorry

end women_in_the_minority_l27_27418


namespace ike_mike_total_items_l27_27578

theorem ike_mike_total_items :
  ∃ (s d : ℕ), s + d = 7 ∧ 5 * s + 3/2 * d = 35 :=
by sorry

end ike_mike_total_items_l27_27578


namespace fraction_spent_on_dvd_l27_27677

theorem fraction_spent_on_dvd (r l m d x : ℝ) (h1 : r = 200) (h2 : l = (1/4) * r) (h3 : m = r - l) (h4 : x = 50) (h5 : d = m - x) : d / r = 1 / 2 :=
by
  sorry

end fraction_spent_on_dvd_l27_27677


namespace first_term_of_arithmetic_sequence_l27_27226

theorem first_term_of_arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
  (h1 : a 3 = 3) (h2 : S 9 - S 6 = 27)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d)
  (h4 : ∀ n, S n = n * (a 1 + a n) / 2) : a 1 = 3 / 5 :=
by
  sorry

end first_term_of_arithmetic_sequence_l27_27226


namespace circle_equation_l27_27813

theorem circle_equation :
  ∃ (a : ℝ) (x y : ℝ), 
    (2 * a + y - 1 = 0 ∧ (x = 3 ∧ y = 0) ∧ (x = 0 ∧ y = 1)) →
    (x - 1) ^ 2 + (y + 1) ^ 2 = 5 := by
  sorry

end circle_equation_l27_27813


namespace amount_spent_on_food_l27_27765

-- We define the conditions given in the problem
def Mitzi_brought_money : ℕ := 75
def ticket_cost : ℕ := 30
def tshirt_cost : ℕ := 23
def money_left : ℕ := 9

-- Define the total amount Mitzi spent
def total_spent : ℕ := Mitzi_brought_money - money_left

-- Define the combined cost of the ticket and T-shirt
def combined_cost : ℕ := ticket_cost + tshirt_cost

-- The proof goal
theorem amount_spent_on_food : total_spent - combined_cost = 13 := by
  sorry

end amount_spent_on_food_l27_27765


namespace age_difference_l27_27724

theorem age_difference (M T J X S : ℕ)
  (hM : M = 3)
  (hT : T = 4 * M)
  (hJ : J = T - 5)
  (hX : X = 2 * J)
  (hS : S = 3 * X - 1) :
  S - M = 38 :=
by
  sorry

end age_difference_l27_27724


namespace find_values_l27_27384

theorem find_values (x y : ℤ) 
  (h1 : x / 5 + 7 = y / 4 - 7)
  (h2 : x / 3 - 4 = y / 2 + 4) : 
  x = -660 ∧ y = -472 :=
by 
  sorry

end find_values_l27_27384


namespace no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0_l27_27599

theorem no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0 :
  ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 :=
sorry

end no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0_l27_27599


namespace find_second_number_l27_27553

theorem find_second_number (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 40 :=
by
  sorry

end find_second_number_l27_27553


namespace fg_of_neg2_l27_27471

def f (x : ℤ) : ℤ := x^2
def g (x : ℤ) : ℤ := 2 * x + 5

theorem fg_of_neg2 : f (g (-2)) = 1 := by
  sorry

end fg_of_neg2_l27_27471


namespace desired_salt_percentage_is_ten_percent_l27_27593

-- Define the initial conditions
def initial_pure_water_volume : ℝ := 100
def saline_solution_percentage : ℝ := 0.25
def added_saline_volume : ℝ := 66.67
def total_volume : ℝ := initial_pure_water_volume + added_saline_volume
def added_salt : ℝ := saline_solution_percentage * added_saline_volume
def desired_salt_percentage (P : ℝ) : Prop := added_salt = P * total_volume

-- State the theorem and its result
theorem desired_salt_percentage_is_ten_percent (P : ℝ) (h : desired_salt_percentage P) : P = 0.1 :=
sorry

end desired_salt_percentage_is_ten_percent_l27_27593


namespace common_factor_l27_27906

-- Define the polynomials
def P1 (x : ℝ) : ℝ := x^3 + x^2
def P2 (x : ℝ) : ℝ := x^2 + 2*x + 1
def P3 (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem common_factor (x : ℝ) : ∃ (f : ℝ → ℝ), (f x = x + 1) ∧ (∃ g1 g2 g3 : ℝ → ℝ, P1 x = f x * g1 x ∧ P2 x = f x * g2 x ∧ P3 x = f x * g3 x) :=
sorry

end common_factor_l27_27906


namespace geometric_seq_property_l27_27343

noncomputable def a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

theorem geometric_seq_property (n : ℕ) (h_arith : S (n + 1) + S (n + 1) = 2 * S (n)) (h_condition : a 2 = -2) :
  a 7 = 64 := 
by sorry

end geometric_seq_property_l27_27343


namespace interior_angle_second_quadrant_l27_27802

theorem interior_angle_second_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.sin α * Real.tan α < 0) : 
  π / 2 < α ∧ α < π :=
by
  sorry

end interior_angle_second_quadrant_l27_27802


namespace infinite_geometric_series_sum_l27_27516

theorem infinite_geometric_series_sum
  (a : ℚ) (r : ℚ) (h_a : a = 1) (h_r : r = 2 / 3) (h_r_abs_lt_one : |r| < 1) :
  ∑' (n : ℕ), a * r^n = 3 :=
by
  -- Import necessary lemmas and properties for infinite series
  sorry -- Proof is omitted.

end infinite_geometric_series_sum_l27_27516


namespace gcd_lcm_product_l27_27799

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 3 * 5^2) (h2 : b = 5^3) : 
  Nat.gcd a b * Nat.lcm a b = 9375 := by
  sorry

end gcd_lcm_product_l27_27799


namespace Abby_wins_if_N_2011_Brian_wins_in_31_cases_l27_27148

-- Definitions and assumptions directly from the problem conditions
inductive Player
| Abby
| Brian

def game_condition (N : ℕ) : Prop :=
  ∀ (p : Player), 
    (p = Player.Abby → (∃ k, N = 2 * k + 1)) ∧ 
    (p = Player.Brian → (∃ k, N = 2 * (2^k - 1))) -- This encodes the winning state conditions for simplicity

-- Part (a)
theorem Abby_wins_if_N_2011 : game_condition 2011 :=
by
  sorry

-- Part (b)
theorem Brian_wins_in_31_cases : 
  (∃ S : Finset ℕ, (∀ N ∈ S, N ≤ 2011 ∧ game_condition N) ∧ S.card = 31) :=
by
  sorry

end Abby_wins_if_N_2011_Brian_wins_in_31_cases_l27_27148


namespace sum_of_ages_l27_27305

variable (A1 : ℝ) (A2 : ℝ) (A3 : ℝ) (A4 : ℝ) (A5 : ℝ) (A6 : ℝ) (A7 : ℝ)

noncomputable def age_first_scroll := 4080
noncomputable def age_difference := 2040

theorem sum_of_ages :
  let r := (age_difference:ℝ) / (age_first_scroll:ℝ)
  let A2 := (age_first_scroll:ℝ) + age_difference
  let A3 := A2 + (A2 - age_first_scroll) * r
  let A4 := A3 + (A3 - A2) * r
  let A5 := A4 + (A4 - A3) * r
  let A6 := A5 + (A5 - A4) * r
  let A7 := A6 + (A6 - A5) * r
  (age_first_scroll:ℝ) + A2 + A3 + A4 + A5 + A6 + A7 = 41023.75 := 
  by sorry

end sum_of_ages_l27_27305


namespace find_k_for_line_l27_27460

theorem find_k_for_line : 
  ∃ k : ℚ, (∀ x y : ℚ, (-1 / 3 - 3 * k * x = 4 * y) ∧ (x = 1 / 3) ∧ (y = -8)) → k = 95 / 3 :=
by
  sorry

end find_k_for_line_l27_27460


namespace arithmetic_seq_a4_value_l27_27603

theorem arithmetic_seq_a4_value
  (a : ℕ → ℤ)
  (h : 4 * a 3 + a 11 - 3 * a 5 = 10) :
  a 4 = 5 := 
sorry

end arithmetic_seq_a4_value_l27_27603


namespace dollar_function_twice_l27_27970

noncomputable def f (N : ℝ) : ℝ := 0.4 * N + 2

theorem dollar_function_twice (N : ℝ) (h : N = 30) : (f ∘ f) N = 5 := 
by
  sorry

end dollar_function_twice_l27_27970


namespace igor_number_proof_l27_27633

noncomputable def igor_number (init_lineup : List ℕ) (igor_num : ℕ) : Prop :=
  let after_first_command := [9, 11, 10, 6, 8, 7] -- Results after first command 
  let after_second_command := [9, 11, 10, 8] -- Results after second command
  let after_third_command := [11, 10, 8] -- Results after third command
  ∃ (idx : ℕ), init_lineup.get? idx = some igor_num ∧
    (∀ new_lineup, 
       (new_lineup = after_first_command ∨ new_lineup = after_second_command ∨ new_lineup = after_third_command) →
       igor_num ∉ new_lineup) ∧ 
    after_third_command.length = 3

theorem igor_number_proof : igor_number [9, 1, 11, 2, 10, 3, 6, 4, 8, 5, 7] 5 :=
  sorry 

end igor_number_proof_l27_27633


namespace find_f_8_5_l27_27757

-- Conditions as definitions in Lean
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def segment_function (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 3 * x

-- The main theorem to prove
theorem find_f_8_5 (f : ℝ → ℝ) (h1 : even_function f) (h2 : periodic_function f 3) (h3 : segment_function f)
: f 8.5 = 1.5 :=
sorry

end find_f_8_5_l27_27757


namespace product_mod_32_l27_27954

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  M % 32 = 17 := by
  sorry

end product_mod_32_l27_27954


namespace edmonton_to_calgary_travel_time_l27_27209

theorem edmonton_to_calgary_travel_time :
  let distance_edmonton_red_deer := 220
  let distance_red_deer_calgary := 110
  let speed_to_red_deer := 100
  let detour_distance := 30
  let detour_time := (distance_edmonton_red_deer + detour_distance) / speed_to_red_deer
  let stop_time := 1
  let speed_to_calgary := 90
  let travel_time_to_calgary := distance_red_deer_calgary / speed_to_calgary
  detour_time + stop_time + travel_time_to_calgary = 4.72 := by
  sorry

end edmonton_to_calgary_travel_time_l27_27209


namespace total_amount_is_47_69_l27_27689

noncomputable def Mell_order_cost : ℝ :=
  2 * 4 + 7

noncomputable def friend_order_cost : ℝ :=
  2 * 4 + 7 + 3

noncomputable def total_cost_before_discount : ℝ :=
  Mell_order_cost + 2 * friend_order_cost

noncomputable def discount : ℝ :=
  0.15 * total_cost_before_discount

noncomputable def total_after_discount : ℝ :=
  total_cost_before_discount - discount

noncomputable def sales_tax : ℝ :=
  0.10 * total_after_discount

noncomputable def total_to_pay : ℝ :=
  total_after_discount + sales_tax

theorem total_amount_is_47_69 : total_to_pay = 47.69 :=
by
  sorry

end total_amount_is_47_69_l27_27689


namespace statement_1_statement_2_statement_3_statement_4_main_proof_l27_27531

noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem statement_1 : ¬ ∃ x, x = 2 ∧ ∀ y, f y ≤ f x := sorry

theorem statement_2 : ∃! x, f x - x = 0 := sorry

theorem statement_3 : ¬ ∃ k > 0, ∀ x > 0, f x > k * x := sorry

theorem statement_4 : ∀ x1 x2 : ℝ, x2 > x1 ∧ f x1 = f x2 → x1 + x2 > 4 := sorry

theorem main_proof : (¬ ∃ x, x = 2 ∧ ∀ y, f y ≤ f x) ∧ 
                     (∃! x, f x - x = 0) ∧ 
                     (¬ ∃ k > 0, ∀ x > 0, f x > k * x) ∧ 
                     (∀ x1 x2 : ℝ, x2 > x1 ∧ f x1 = f x2 → x1 + x2 > 4) := 
by
  apply And.intro
  · exact statement_1
  · apply And.intro
    · exact statement_2
    · apply And.intro
      · exact statement_3
      · exact statement_4

end statement_1_statement_2_statement_3_statement_4_main_proof_l27_27531


namespace least_number_to_add_l27_27042

theorem least_number_to_add (n : ℕ) (h : n = 17 * 23 * 29) : 
  ∃ k, k + 1024 ≡ 0 [MOD n] ∧ 
       (∀ m, (m + 1024) ≡ 0 [MOD n] → k ≤ m) ∧ 
       k = 10315 :=
by 
  sorry

end least_number_to_add_l27_27042


namespace evaluate_expression_l27_27149

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 4 - 2 * g (-2) = 47 :=
by
  sorry

end evaluate_expression_l27_27149


namespace tully_twice_kate_in_three_years_l27_27782

-- Definitions for the conditions
def tully_was := 60
def kate_is := 29

-- Number of years from now when Tully will be twice as old as Kate
theorem tully_twice_kate_in_three_years : 
  ∃ (x : ℕ), (tully_was + 1 + x = 2 * (kate_is + x)) ∧ x = 3 :=
by
  sorry

end tully_twice_kate_in_three_years_l27_27782


namespace anne_speed_l27_27964

-- Conditions
def time_hours : ℝ := 3
def distance_miles : ℝ := 6

-- Question with correct answer
theorem anne_speed : distance_miles / time_hours = 2 := by 
  sorry

end anne_speed_l27_27964


namespace ladybugs_calculation_l27_27718

def total_ladybugs : ℕ := 67082
def ladybugs_with_spots : ℕ := 12170
def ladybugs_without_spots : ℕ := 54912

theorem ladybugs_calculation :
  total_ladybugs - ladybugs_with_spots = ladybugs_without_spots :=
by
  sorry

end ladybugs_calculation_l27_27718
