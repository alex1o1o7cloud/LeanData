import Mathlib

namespace polygon_side_count_eq_six_l433_43328

theorem polygon_side_count_eq_six (n : ℕ) 
  (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_side_count_eq_six_l433_43328


namespace expand_product_l433_43341

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9 * x + 18 := 
by sorry

end expand_product_l433_43341


namespace swimmers_meeting_times_l433_43362

theorem swimmers_meeting_times (l : ℕ) (vA vB t : ℕ) (T : ℝ) :
  l = 120 →
  vA = 4 →
  vB = 3 →
  t = 15 →
  T = 21 :=
  sorry

end swimmers_meeting_times_l433_43362


namespace find_number_l433_43325

theorem find_number (x : ℝ) : ((1.5 * x) / 7 = 271.07142857142856) → x = 1265 :=
by
  sorry

end find_number_l433_43325


namespace line_through_circle_center_l433_43321

theorem line_through_circle_center (a : ℝ) :
  (∃ (x y : ℝ), 3 * x + y + a = 0 ∧ x^2 + y^2 + 2 * x - 4 * y = 0) ↔ (a = 1) :=
by
  sorry

end line_through_circle_center_l433_43321


namespace box_with_20_aluminium_80_plastic_weighs_494_l433_43355

def weight_of_box_with_100_aluminium_balls := 510 -- in grams
def weight_of_box_with_100_plastic_balls := 490 -- in grams
def number_of_aluminium_balls := 100
def number_of_plastic_balls := 100

-- Define the weights per ball type by subtracting the weight of the box
def weight_per_aluminium_ball := (weight_of_box_with_100_aluminium_balls - weight_of_box_with_100_plastic_balls) / number_of_aluminium_balls
def weight_per_plastic_ball := (weight_of_box_with_100_plastic_balls - weight_of_box_with_100_plastic_balls) / number_of_plastic_balls

-- Condition: The weight of the box alone (since it's present in both conditions)
def weight_of_empty_box := weight_of_box_with_100_plastic_balls - (weight_per_plastic_ball * number_of_plastic_balls)

-- Function to compute weight of the box with given number of aluminium and plastic balls
def total_weight (num_al : ℕ) (num_pl : ℕ) : ℕ :=
  weight_of_empty_box + (weight_per_aluminium_ball * num_al) + (weight_per_plastic_ball * num_pl)

-- The theorem to be proven
theorem box_with_20_aluminium_80_plastic_weighs_494 :
  total_weight 20 80 = 494 := sorry

end box_with_20_aluminium_80_plastic_weighs_494_l433_43355


namespace find_d_l433_43337

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := c * x + 1

theorem find_d (c d : ℝ) (hx : ∀ x, f (g x c) c = 15 * x + d) : d = 8 :=
sorry

end find_d_l433_43337


namespace count_three_digit_numbers_increased_by_99_when_reversed_l433_43370

def countValidNumbers : Nat := 80

theorem count_three_digit_numbers_increased_by_99_when_reversed :
  ∃ (a b c : Nat), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧
   (100 * a + 10 * b + c + 99 = 100 * c + 10 * b + a) ∧
  (countValidNumbers = 80) :=
sorry

end count_three_digit_numbers_increased_by_99_when_reversed_l433_43370


namespace compute_tensor_operation_l433_43384

def tensor (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

theorem compute_tensor_operation :
  tensor (tensor 8 4) 2 = 202 / 9 :=
by
  sorry

end compute_tensor_operation_l433_43384


namespace percentage_increase_l433_43361

theorem percentage_increase (original_value : ℕ) (percentage_increase : ℚ) :  
  original_value = 1200 → 
  percentage_increase = 0.40 →
  original_value * (1 + percentage_increase) = 1680 :=
by
  intros h1 h2
  sorry

end percentage_increase_l433_43361


namespace cover_large_square_l433_43318

theorem cover_large_square :
  ∃ (small_squares : Fin 8 → Set (ℝ × ℝ)),
    (∀ i, small_squares i = {p : ℝ × ℝ | (p.1 - x_i)^2 + (p.2 - y_i)^2 < (3/2)^2}) ∧
    (∃ (large_square : Set (ℝ × ℝ)),
      large_square = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 7 ∧ 0 ≤ p.2 ∧ p.2 ≤ 7} ∧
      large_square ⊆ ⋃ i, small_squares i) :=
sorry

end cover_large_square_l433_43318


namespace Vovochka_correct_pairs_count_l433_43398

def no_carry_pairs_count (digit_sum: ℕ → ℕ → Prop) : ℕ :=
  let count_pairs (lim: ℕ) : ℕ := (lim * (lim + 1)) / 2
  let digit_valid_pairs : ℕ := count_pairs 9
  (digit_valid_pairs * digit_valid_pairs) * 81

def digit_sum (x y: ℕ) : Prop := (x + y ≤ 9)

theorem Vovochka_correct_pairs_count :
  no_carry_pairs_count digit_sum = 244620 := by
  sorry

end Vovochka_correct_pairs_count_l433_43398


namespace distance_covered_by_train_l433_43373

-- Define the average speed and the total duration of the journey
def speed : ℝ := 10
def time : ℝ := 8

-- Use these definitions to state and prove the distance covered by the train
theorem distance_covered_by_train : speed * time = 80 := by
  sorry

end distance_covered_by_train_l433_43373


namespace speed_of_A_l433_43392
-- Import necessary library

-- Define conditions
def initial_distance : ℝ := 25  -- initial distance between A and B
def speed_B : ℝ := 13  -- speed of B in kmph
def meeting_time : ℝ := 1  -- time duration in hours

-- The speed of A which is to be proven
def speed_A : ℝ := 12

-- The theorem to be proved
theorem speed_of_A (d : ℝ) (vB : ℝ) (t : ℝ) (vA : ℝ) : d = 25 → vB = 13 → t = 1 → 
  d = vA * t + vB * t → vA = 12 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  -- Enforcing the statement to be proved
  have := Eq.symm h4
  simp [speed_A, *] at *
  sorry

end speed_of_A_l433_43392


namespace machine_A_production_rate_l433_43389

theorem machine_A_production_rate :
  ∀ (A B T_A T_B : ℝ),
    500 = A * T_A →
    500 = B * T_B →
    B = 1.25 * A →
    T_A = T_B + 15 →
    A = 100 / 15 :=
by
  intros A B T_A T_B hA hB hRate hTime
  sorry

end machine_A_production_rate_l433_43389


namespace expression_undefined_iff_l433_43393

theorem expression_undefined_iff (a : ℝ) : (a^2 - 9 = 0) ↔ (a = 3 ∨ a = -3) :=
sorry

end expression_undefined_iff_l433_43393


namespace a_in_M_sufficient_not_necessary_l433_43304

-- Defining the sets M and N
def M := {x : ℝ | x^2 < 3 * x}
def N := {x : ℝ | abs (x - 1) < 2}

-- Stating that a ∈ M is a sufficient but not necessary condition for a ∈ N
theorem a_in_M_sufficient_not_necessary (a : ℝ) (h : a ∈ M) : a ∈ N :=
by sorry

end a_in_M_sufficient_not_necessary_l433_43304


namespace least_possible_value_of_y_l433_43303

theorem least_possible_value_of_y (x y z : ℤ) (hx : Even x) (hy : Odd y) (hz : Odd z) 
  (h1 : y - x > 5) (h2 : z - x ≥ 9) : y ≥ 7 :=
by {
  -- sorry allows us to skip the proof
  sorry
}

end least_possible_value_of_y_l433_43303


namespace rect_side_ratio_square_l433_43314

theorem rect_side_ratio_square (a b d : ℝ) (h1 : b = 2 * a) (h2 : d = a * Real.sqrt 5) : (b / a) ^ 2 = 4 := 
by sorry

end rect_side_ratio_square_l433_43314


namespace smallest_possible_n_l433_43333

-- Definitions needed for the problem
variable (x n : ℕ) (hpos : 0 < x)
variable (m : ℕ) (hm : m = 72)

-- The conditions as already stated
def gcd_cond := Nat.gcd 72 n = x + 8
def lcm_cond := Nat.lcm 72 n = x * (x + 8)

-- The proof statement
theorem smallest_possible_n (h_gcd : gcd_cond x n) (h_lcm : lcm_cond x n) : n = 8 :=
by 
  -- Intuitively outline the proof
  sorry

end smallest_possible_n_l433_43333


namespace solve_for_x_l433_43369

theorem solve_for_x :
  ∃ x : ℝ, (2015 + x)^2 = x^2 ∧ x = -2015 / 2 :=
by
  sorry

end solve_for_x_l433_43369


namespace valid_two_digit_numbers_l433_43315

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)

-- Prove the statement about two-digit numbers satisfying the condition
theorem valid_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A ≤ 99 ∧ (sum_of_digits A)^2 = sum_of_digits (A^2)} =
  {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by
  sorry

end valid_two_digit_numbers_l433_43315


namespace restaurant_total_glasses_l433_43368

theorem restaurant_total_glasses (x y t : ℕ) 
  (h1 : y = x + 16)
  (h2 : (12 * x + 16 * y) / (x + y) = 15)
  (h3 : t = 12 * x + 16 * y) : 
  t = 480 :=
by 
  -- Proof omitted
  sorry

end restaurant_total_glasses_l433_43368


namespace total_amount_earned_l433_43365

theorem total_amount_earned (avg_price_per_pair : ℝ) (number_of_pairs : ℕ) (price : avg_price_per_pair = 9.8 ) (pairs : number_of_pairs = 50 ) : 
avg_price_per_pair * number_of_pairs = 490 := by
  -- Given conditions
  sorry

end total_amount_earned_l433_43365


namespace jogger_ahead_distance_l433_43366

-- Definitions of conditions
def jogger_speed : ℝ := 9  -- km/hr
def train_speed : ℝ := 45  -- km/hr
def train_length : ℝ := 150  -- meters
def passing_time : ℝ := 39  -- seconds

-- The main statement that we want to prove
theorem jogger_ahead_distance : 
  let relative_speed := (train_speed - jogger_speed) * (5 / 18)  -- conversion to m/s
  let distance_covered := relative_speed * passing_time
  let jogger_ahead := distance_covered - train_length
  jogger_ahead = 240 :=
by
  sorry

end jogger_ahead_distance_l433_43366


namespace time_to_cross_signal_pole_l433_43350

/-- Definitions representing the given conditions --/
def length_of_train : ℕ := 300
def time_to_cross_platform : ℕ := 39
def length_of_platform : ℕ := 350
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_cross_platform

/-- Main statement to be proven --/
theorem time_to_cross_signal_pole : length_of_train / speed_of_train = 18 := by
  sorry

end time_to_cross_signal_pole_l433_43350


namespace find_m_l433_43344

noncomputable def geometric_sequence_solution (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) (m : ℕ) : Prop :=
  (S 3 + S 6 = 2 * S 9) ∧ (a 2 + a 5 = 2 * a m)

theorem find_m (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) (m : ℕ) (h1 : S 3 + S 6 = 2 * S 9)
  (h2 : a 2 + a 5 = 2 * a m) : m = 8 :=
sorry

end find_m_l433_43344


namespace final_answer_l433_43357

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end final_answer_l433_43357


namespace quadratic_minimum_l433_43319

-- Define the constants p and q as positive real numbers
variables (p q : ℝ) (hp : 0 < p) (hq : 0 < q)

-- Define the quadratic function f
def f (x : ℝ) : ℝ := 3 * x^2 + p * x + q

-- Assertion to prove: the function f reaches its minimum at x = -p / 6
theorem quadratic_minimum : 
  ∃ x : ℝ, x = -p / 6 ∧ (∀ y : ℝ, f y ≥ f x) :=
sorry

end quadratic_minimum_l433_43319


namespace john_task_completion_l433_43306

theorem john_task_completion (J : ℝ) (h : 5 * (1 / J + 1 / 10) + 5 * (1 / J) = 1) : J = 20 :=
by
  sorry

end john_task_completion_l433_43306


namespace scientific_notation_of_361000000_l433_43339

theorem scientific_notation_of_361000000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ abs a) ∧ (abs a < 10) ∧ (361000000 = a * 10^n) ∧ (a = 3.61) ∧ (n = 8) :=
sorry

end scientific_notation_of_361000000_l433_43339


namespace perp_line_eq_l433_43336

theorem perp_line_eq (m : ℝ) (L1 : ∀ (x y : ℝ), m * x - m^2 * y = 1) (P : ℝ × ℝ) (P_def : P = (2, 1)) :
  ∃ d : ℝ, (∀ (x y : ℝ), x + y = d) ∧ P.fst + P.snd = d :=
by
  sorry

end perp_line_eq_l433_43336


namespace cats_in_house_l433_43388

-- Define the conditions
def total_cats (C : ℕ) : Prop :=
  let num_white_cats := 2
  let num_black_cats := C / 4
  let num_grey_cats := 10
  C = num_white_cats + num_black_cats + num_grey_cats

-- State the theorem
theorem cats_in_house : ∃ C : ℕ, total_cats C ∧ C = 16 := 
by
  sorry

end cats_in_house_l433_43388


namespace total_profit_is_2560_l433_43376

noncomputable def basicWashPrice : ℕ := 5
noncomputable def deluxeWashPrice : ℕ := 10
noncomputable def premiumWashPrice : ℕ := 15

noncomputable def basicCarsWeekday : ℕ := 50
noncomputable def deluxeCarsWeekday : ℕ := 40
noncomputable def premiumCarsWeekday : ℕ := 20

noncomputable def employeeADailyWage : ℕ := 110
noncomputable def employeeBDailyWage : ℕ := 90
noncomputable def employeeCDailyWage : ℕ := 100
noncomputable def employeeDDailyWage : ℕ := 80

noncomputable def operatingExpenseWeekday : ℕ := 200

noncomputable def totalProfit : ℕ := 
  let revenueWeekday := (basicCarsWeekday * basicWashPrice) + 
                        (deluxeCarsWeekday * deluxeWashPrice) + 
                        (premiumCarsWeekday * premiumWashPrice)
  let totalRevenue := revenueWeekday * 5
  let wageA := employeeADailyWage * 5
  let wageB := employeeBDailyWage * 2
  let wageC := employeeCDailyWage * 3
  let wageD := employeeDDailyWage * 2
  let totalWages := wageA + wageB + wageC + wageD
  let totalOperatingExpenses := operatingExpenseWeekday * 5
  totalRevenue - (totalWages + totalOperatingExpenses)

theorem total_profit_is_2560 : totalProfit = 2560 := by
  sorry

end total_profit_is_2560_l433_43376


namespace latus_rectum_of_parabola_l433_43371

theorem latus_rectum_of_parabola : 
  ∀ x y : ℝ, x^2 = -y → y = 1/4 :=
by
  -- Proof omitted
  sorry

end latus_rectum_of_parabola_l433_43371


namespace area_two_layers_l433_43301

-- Given conditions
variables (A_total A_covered A_three_layers : ℕ)

-- Conditions from the problem
def condition_1 : Prop := A_total = 204
def condition_2 : Prop := A_covered = 140
def condition_3 : Prop := A_three_layers = 20

-- Mathematical equivalent proof problem
theorem area_two_layers (A_total A_covered A_three_layers : ℕ) 
  (h1 : condition_1 A_total) 
  (h2 : condition_2 A_covered) 
  (h3 : condition_3 A_three_layers) : 
  ∃ A_two_layers : ℕ, A_two_layers = 24 :=
by sorry

end area_two_layers_l433_43301


namespace q_at_2_equals_9_l433_43300

-- Define the sign function
noncomputable def sgn (x : ℝ) : ℝ :=
if x < 0 then -1 else if x = 0 then 0 else 1

-- Define the function q(x)
noncomputable def q (x : ℝ) : ℝ :=
sgn (3 * x - 1) * |3 * x - 1| ^ (1/2) +
3 * sgn (3 * x - 1) * |3 * x - 1| ^ (1/3) +
|3 * x - 1| ^ (1/4)

-- The theorem stating that q(2) equals 9
theorem q_at_2_equals_9 : q 2 = 9 :=
by sorry

end q_at_2_equals_9_l433_43300


namespace find_a_b_c_sum_l433_43310

theorem find_a_b_c_sum (a b c : ℝ) 
  (h_vertex : ∀ x, y = a * x^2 + b * x + c ↔ y = a * (x - 3)^2 + 5)
  (h_passes : a * 1^2 + b * 1 + c = 2) :
  a + b + c = 35 / 4 :=
sorry

end find_a_b_c_sum_l433_43310


namespace cabbage_production_l433_43332

theorem cabbage_production (x y : ℕ) 
  (h1 : y^2 - x^2 = 127) 
  (h2 : y - x = 1) 
  (h3 : 2 * y = 128) : y^2 = 4096 := by
  sorry

end cabbage_production_l433_43332


namespace original_number_is_120_l433_43302

theorem original_number_is_120 (N k : ℤ) (hk : N - 33 = 87 * k) : N = 120 :=
by
  have h : N - 33 = 87 * 1 := by sorry
  have N_eq : N = 87 + 33 := by sorry
  have N_val : N = 120 := by sorry
  exact N_val

end original_number_is_120_l433_43302


namespace min_value_of_squares_l433_43327

theorem min_value_of_squares (a b c : ℝ) (h : a^3 + b^3 + c^3 - 3 * a * b * c = 8) : 
  ∃ m, m ≥ 4 ∧ ∀ a b c, a^3 + b^3 + c^3 - 3 * a * b * c = 8 → a^2 + b^2 + c^2 ≥ m :=
sorry

end min_value_of_squares_l433_43327


namespace eight_pow_three_eq_two_pow_nine_l433_43364

theorem eight_pow_three_eq_two_pow_nine : 8^3 = 2^9 := by
  sorry -- Proof is skipped

end eight_pow_three_eq_two_pow_nine_l433_43364


namespace simplify_expression_l433_43329

theorem simplify_expression (x y : ℝ) (hx : x = 5) (hy : y = 2) :
  (10 * x * y^3) / (15 * x^2 * y^2) = 4 / 15 :=
by
  rw [hx, hy]
  -- here we would simplify but leave a hole
  sorry

end simplify_expression_l433_43329


namespace quadratic_sum_is_zero_l433_43345

-- Definition of a quadratic function with given conditions and the final result to prove
theorem quadratic_sum_is_zero {a b c : ℝ} 
  (h₁ : ∀ x : ℝ, x = 3 → a * (x - 1) * (x - 5) = 36) 
  (h₂ : a * 1^2 + b * 1 + c = 0) 
  (h₃ : a * 5^2 + b * 5 + c = 0) : 
  a + b + c = 0 := 
sorry

end quadratic_sum_is_zero_l433_43345


namespace alice_winning_strategy_l433_43338

theorem alice_winning_strategy (N : ℕ) (hN : N > 0) : 
  (∃! n : ℕ, N = n * n) ↔ (∀ (k : ℕ), ∃ (m : ℕ), m ≠ k ∧ (m ∣ k ∨ k ∣ m)) :=
sorry

end alice_winning_strategy_l433_43338


namespace intersection_A_B_l433_43385

-- Define the set A
def A : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, x + 1) }

-- Define the set B
def B : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, -2*x + 4) }

-- State the theorem to prove A ∩ B = {(1, 2)}
theorem intersection_A_B : A ∩ B = { (1, 2) } :=
by
  sorry

end intersection_A_B_l433_43385


namespace no_solution_for_inequalities_l433_43374

theorem no_solution_for_inequalities (x : ℝ) :
  ¬(5 * x^2 - 7 * x + 1 < 0 ∧ x^2 - 9 * x + 30 < 0) :=
sorry

end no_solution_for_inequalities_l433_43374


namespace coordinates_of_point_P_l433_43305

noncomputable def tangent_slope_4 : Prop :=
  ∀ (x y : ℝ), y = 1 / x → (-1 / (x^2)) = -4 → (x = 1 / 2 ∧ y = 2) ∨ (x = -1 / 2 ∧ y = -2)

theorem coordinates_of_point_P : tangent_slope_4 :=
by sorry

end coordinates_of_point_P_l433_43305


namespace geometric_sequence_second_term_value_l433_43330

theorem geometric_sequence_second_term_value
  (a : ℝ) 
  (r : ℝ) 
  (h1 : 30 * r = a) 
  (h2 : a * r = 7 / 4) 
  (h3 : 0 < a) : 
  a = 7.5 := 
sorry

end geometric_sequence_second_term_value_l433_43330


namespace pure_imaginary_solution_l433_43346

theorem pure_imaginary_solution (m : ℝ) 
  (h : ∃ m : ℝ, (m^2 + m - 2 = 0) ∧ (m^2 - 1 ≠ 0)) : m = -2 :=
sorry

end pure_imaginary_solution_l433_43346


namespace problem_statement_l433_43308

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement :
  (∀ x y : ℝ, f x + f y = f (x + y)) →
  f 3 = 4 →
  f 0 + f (-3) = -4 :=
by
  intros h1 h2
  sorry

end problem_statement_l433_43308


namespace inequality_positive_reals_l433_43367

theorem inequality_positive_reals (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (x / (1 + x^2)) + (y / (1 + y^2)) + (z / (1 + z^2)) ≤ (3 * Real.sqrt 3) / 4 := by
  sorry

end inequality_positive_reals_l433_43367


namespace solution_l433_43353

theorem solution
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (H : (1 / a + 1 / b) * (1 / c + 1 / d) + 1 / (a * b) + 1 / (c * d) = 6 / Real.sqrt (a * b * c * d)) :
  (a^2 + a * c + c^2) / (b^2 - b * d + d^2) = 3 :=
sorry

end solution_l433_43353


namespace largest_divisor_n_l433_43359

theorem largest_divisor_n (n : ℕ) (h₁ : n > 0) (h₂ : 650 ∣ n^3) : 130 ∣ n :=
sorry

end largest_divisor_n_l433_43359


namespace percentage_increase_from_second_to_third_building_l433_43382

theorem percentage_increase_from_second_to_third_building :
  let first_building_units := 4000
  let second_building_units := (2 / 5 : ℝ) * first_building_units
  let total_units := 7520
  let third_building_units := total_units - (first_building_units + second_building_units)
  (third_building_units - second_building_units) / second_building_units * 100 = 20 := by
  let first_building_units := 4000
  let second_building_units : ℝ := (2 / 5 : ℝ) * first_building_units
  let total_units := 7520
  let third_building_units := total_units - (first_building_units + second_building_units)
  have H : (third_building_units - second_building_units) / second_building_units * 100 = 20 := sorry
  exact H

end percentage_increase_from_second_to_third_building_l433_43382


namespace solve_for_r_l433_43351

theorem solve_for_r (r : ℚ) (h : (r + 4) / (r - 3) = (r - 2) / (r + 2)) : r = -2/11 :=
by
  sorry

end solve_for_r_l433_43351


namespace find_digit_B_l433_43348

def six_digit_number (B : ℕ) : ℕ := 303200 + B

def is_prime_six_digit (B : ℕ) : Prop := Prime (six_digit_number B)

theorem find_digit_B :
  ∃ B : ℕ, (B ≤ 9) ∧ (is_prime_six_digit B) ∧ (B = 9) :=
sorry

end find_digit_B_l433_43348


namespace ways_to_draw_balls_eq_total_ways_l433_43323

noncomputable def ways_to_draw_balls (n : Nat) :=
  if h : n = 15 then (15 * 14 * 13 * 12) else 0

noncomputable def valid_combinations : Nat := sorry

noncomputable def total_ways_to_draw : Nat :=
  valid_combinations * 24

theorem ways_to_draw_balls_eq_total_ways :
  ways_to_draw_balls 15 = total_ways_to_draw :=
sorry

end ways_to_draw_balls_eq_total_ways_l433_43323


namespace next_meeting_time_at_B_l433_43347

-- Definitions of conditions
def perimeter := 800 -- Perimeter of the block in meters
def t1 := 1 -- They meet for the first time after 1 minute
def AB := 100 -- Length of side AB in meters
def BC := 300 -- Length of side BC in meters
def CD := 100 -- Length of side CD in meters
def DA := 300 -- Length of side DA in meters

-- Main theorem statement
theorem next_meeting_time_at_B :
  ∃ t : ℕ, t = 9 ∧ (∃ m1 m2 : ℕ, ((t = m1 * m2 + 1) ∧ m2 = 800 / (t1 * (AB + BC + CD + DA))) ∧ m1 = 9) :=
sorry

end next_meeting_time_at_B_l433_43347


namespace find_v_value_l433_43394

theorem find_v_value (x : ℝ) (v : ℝ) (h1 : x = 3.0) (h2 : 5 * x + v = 19) : v = 4 := by
  sorry

end find_v_value_l433_43394


namespace cost_of_single_room_l433_43311

theorem cost_of_single_room
  (total_rooms : ℕ)
  (double_rooms : ℕ)
  (cost_double_room : ℕ)
  (revenue_total : ℕ)
  (cost_single_room : ℕ)
  (H1 : total_rooms = 260)
  (H2 : double_rooms = 196)
  (H3 : cost_double_room = 60)
  (H4 : revenue_total = 14000)
  (H5 : revenue_total = (total_rooms - double_rooms) * cost_single_room + double_rooms * cost_double_room)
  : cost_single_room = 35 :=
sorry

end cost_of_single_room_l433_43311


namespace perimeter_of_tangents_triangle_l433_43335

theorem perimeter_of_tangents_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
    (4 * a * Real.sqrt (a * b)) / (a - b) = 4 * a * (Real.sqrt (a * b) / (a - b)) := 
sorry

end perimeter_of_tangents_triangle_l433_43335


namespace even_function_a_equals_one_l433_43383

theorem even_function_a_equals_one (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x - a) = (1 - x) * (-x - a)) → a = 1 :=
by
  intro h
  sorry

end even_function_a_equals_one_l433_43383


namespace h_at_0_l433_43322

noncomputable def h (x : ℝ) : ℝ := sorry -- the actual polynomial
-- Conditions for h(x)
axiom h_cond1 : h (-2) = -4
axiom h_cond2 : h (1) = -1
axiom h_cond3 : h (-3) = -9
axiom h_cond4 : h (3) = -9
axiom h_cond5 : h (5) = -25

-- Statement of the proof problem
theorem h_at_0 : h (0) = -90 := sorry

end h_at_0_l433_43322


namespace barkley_total_net_buried_bones_l433_43396

def monthly_bones_received (months : ℕ) : (ℕ × ℕ × ℕ) := (10 * months, 6 * months, 4 * months)

def burying_pattern_A (months : ℕ) : ℕ := 6 * months
def eating_pattern_A (months : ℕ) : ℕ := if months > 2 then 3 else 1

def burying_pattern_B (months : ℕ) : ℕ := if months = 5 then 0 else 4 * (months - 1)
def eating_pattern_B (months : ℕ) : ℕ := 2

def burying_pattern_C (months : ℕ) : ℕ := 2 * months
def eating_pattern_C (months : ℕ) : ℕ := 2

def total_net_buried_bones (months : ℕ) : ℕ :=
  let (received_A, received_B, received_C) := monthly_bones_received months
  let net_A := burying_pattern_A months - eating_pattern_A months
  let net_B := burying_pattern_B months - eating_pattern_B months
  let net_C := burying_pattern_C months - eating_pattern_C months
  net_A + net_B + net_C

theorem barkley_total_net_buried_bones : total_net_buried_bones 5 = 49 := by
  sorry

end barkley_total_net_buried_bones_l433_43396


namespace largest_remainder_a_correct_l433_43340

def largest_remainder_a (n : ℕ) (h : n < 150) : ℕ :=
  (269 % n)

theorem largest_remainder_a_correct : ∃ n < 150, largest_remainder_a n sorry = 133 :=
  sorry

end largest_remainder_a_correct_l433_43340


namespace fraction_of_project_completed_in_one_hour_l433_43395

noncomputable def fraction_of_project_completed_together (a b : ℝ) : ℝ :=
  (1 / a) + (1 / b)

theorem fraction_of_project_completed_in_one_hour (a b : ℝ) :
  fraction_of_project_completed_together a b = (1 / a) + (1 / b) := by
  sorry

end fraction_of_project_completed_in_one_hour_l433_43395


namespace option_c_correct_l433_43312

theorem option_c_correct (a b : ℝ) (h : a > b) : 2 + a > 2 + b :=
by sorry

end option_c_correct_l433_43312


namespace equilateral_triangle_perimeter_l433_43363

theorem equilateral_triangle_perimeter (p_ADC : ℝ) (h_ratio : ∀ s1 s2 : ℝ, s1 / s2 = 1 / 2) :
  p_ADC = 9 + 3 * Real.sqrt 3 → (3 * (2 * (3 + Real.sqrt 3)) = 18 + 6 * Real.sqrt 3) :=
by
  intro h
  have h1 : 3 * (2 * (3 + Real.sqrt 3)) = 18 + 6 * Real.sqrt 3 := sorry
  exact h1

end equilateral_triangle_perimeter_l433_43363


namespace mutually_exclusive_event_of_hitting_target_at_least_once_l433_43380

-- Definitions from conditions
def two_shots_fired : Prop := true

def complementary_events (E F : Prop) : Prop :=
  E ∨ F ∧ ¬(E ∧ F)

def hitting_target_at_least_once : Prop := true -- Placeholder for the event of hitting at least one target
def both_shots_miss : Prop := true              -- Placeholder for the event that both shots miss

-- Statement to prove
theorem mutually_exclusive_event_of_hitting_target_at_least_once
  (h1 : two_shots_fired)
  (h2 : complementary_events hitting_target_at_least_once both_shots_miss) :
  hitting_target_at_least_once = ¬both_shots_miss := 
sorry

end mutually_exclusive_event_of_hitting_target_at_least_once_l433_43380


namespace exist_five_natural_numbers_sum_and_product_equal_ten_l433_43342

theorem exist_five_natural_numbers_sum_and_product_equal_ten : 
  ∃ (n_1 n_2 n_3 n_4 n_5 : ℕ), 
  n_1 + n_2 + n_3 + n_4 + n_5 = 10 ∧ 
  n_1 * n_2 * n_3 * n_4 * n_5 = 10 := 
sorry

end exist_five_natural_numbers_sum_and_product_equal_ten_l433_43342


namespace average_speed_for_remaining_part_l433_43377

theorem average_speed_for_remaining_part (D : ℝ) (v : ℝ) 
  (h1 : 0.8 * D / 80 + 0.2 * D / v = D / 50) : v = 20 :=
sorry

end average_speed_for_remaining_part_l433_43377


namespace minimum_oranges_to_profit_l433_43352

/-- 
A boy buys 4 oranges for 12 cents and sells 6 oranges for 25 cents. 
Calculate the minimum number of oranges he needs to sell to make a profit of 150 cents.
--/
theorem minimum_oranges_to_profit (cost_oranges : ℕ) (cost_cents : ℕ)
  (sell_oranges : ℕ) (sell_cents : ℕ) (desired_profit : ℚ) :
  cost_oranges = 4 → cost_cents = 12 →
  sell_oranges = 6 → sell_cents = 25 →
  desired_profit = 150 →
  (∃ n : ℕ, n = 129) :=
by
  sorry

end minimum_oranges_to_profit_l433_43352


namespace no_more_beverages_needed_l433_43331

namespace HydrationPlan

def daily_water_need := 9
def daily_juice_need := 5
def daily_soda_need := 3
def days := 60

def total_water_needed := daily_water_need * days
def total_juice_needed := daily_juice_need * days
def total_soda_needed := daily_soda_need * days

def water_already_have := 617
def juice_already_have := 350
def soda_already_have := 215

theorem no_more_beverages_needed :
  (water_already_have >= total_water_needed) ∧ 
  (juice_already_have >= total_juice_needed) ∧ 
  (soda_already_have >= total_soda_needed) :=
by 
  -- proof goes here
  sorry

end HydrationPlan

end no_more_beverages_needed_l433_43331


namespace fifth_number_in_pascal_row_l433_43390

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l433_43390


namespace compare_M_N_l433_43354

theorem compare_M_N (a : ℝ) : 
  let M := 2 * a * (a - 2) + 7
  let N := (a - 2) * (a - 3)
  M > N :=
by
  sorry

end compare_M_N_l433_43354


namespace solve_for_d_l433_43356

theorem solve_for_d (r s t d c : ℝ)
  (h1 : (t = -r - s))
  (h2 : (c = rs + rt + st))
  (h3 : (t - 1 = -(r + 5) - (s - 4)))
  (h4 : (c = (r + 5) * (s - 4) + (r + 5) * (t - 1) + (s - 4) * (t - 1)))
  (h5 : (d = -r * s * t))
  (h6 : (d + 210 = -(r + 5) * (s - 4) * (t - 1))) :
  d = 240 ∨ d = 420 :=
by
  sorry

end solve_for_d_l433_43356


namespace positive_integers_expressible_l433_43309

theorem positive_integers_expressible :
  ∃ (x y : ℕ), (x > 0) ∧ (y > 0) ∧ (x^2 + y) / (x * y + 1) = 1 ∧
  ∃ (x' y' : ℕ), (x' > 0) ∧ (y' > 0) ∧ (x' ≠ x ∨ y' ≠ y) ∧ (x'^2 + y') / (x' * y' + 1) = 1 :=
by
  sorry

end positive_integers_expressible_l433_43309


namespace find_increase_in_perimeter_l433_43379

variable (L B y : ℕ)

theorem find_increase_in_perimeter (h1 : 2 * (L + y + (B + y)) = 2 * (L + B) + 16) : y = 4 := by
  sorry

end find_increase_in_perimeter_l433_43379


namespace teresa_age_when_michiko_born_l433_43334

theorem teresa_age_when_michiko_born (teresa_current_age morio_current_age morio_age_when_michiko_born : ℕ) 
  (h1 : teresa_current_age = 59) 
  (h2 : morio_current_age = 71) 
  (h3 : morio_age_when_michiko_born = 38) : 
  teresa_current_age - (morio_current_age - morio_age_when_michiko_born) = 26 := 
by 
  sorry

end teresa_age_when_michiko_born_l433_43334


namespace area_of_Q1Q3Q5Q7_l433_43381

def regular_octagon_apothem : ℝ := 3

def area_of_quadrilateral (a : ℝ) : Prop :=
  let s := 6 * (1 - Real.sqrt 2)
  let side_length := s * Real.sqrt 2
  let area := side_length ^ 2
  area = 72 * (3 - 2 * Real.sqrt 2)

theorem area_of_Q1Q3Q5Q7 : area_of_quadrilateral regular_octagon_apothem :=
  sorry

end area_of_Q1Q3Q5Q7_l433_43381


namespace age_of_Rahim_l433_43372

theorem age_of_Rahim (R : ℕ) (h1 : ∀ (a : ℕ), a = (R + 1) → (a + 5) = (2 * R)) (h2 : ∀ (a : ℕ), a = (R + 1) → a = R + 1) :
  R = 6 := by
  sorry

end age_of_Rahim_l433_43372


namespace standard_deviation_of_applicants_l433_43358

theorem standard_deviation_of_applicants (σ : ℕ) 
  (h1 : ∃ avg : ℕ, avg = 30)
  (h2 : ∃ n : ℕ, n = 17)
  (h3 : ∃ range_count : ℕ, range_count = (30 + σ) - (30 - σ) + 1) :
  σ = 8 :=
by
  sorry

end standard_deviation_of_applicants_l433_43358


namespace sector_area_l433_43320

noncomputable def area_of_sector (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) : ℝ :=
  1 / 2 * arc_length * radius

theorem sector_area (R : ℝ)
  (arc_length : ℝ) (central_angle : ℝ)
  (h_arc : arc_length = 4 * Real.pi)
  (h_angle : central_angle = Real.pi / 3)
  (h_radius : arc_length = central_angle * R) :
  area_of_sector arc_length central_angle 12 = 24 * Real.pi :=
by
  -- Proof skipped
  sorry

#check sector_area

end sector_area_l433_43320


namespace rectangle_area_perimeter_eq_l433_43386

theorem rectangle_area_perimeter_eq (x : ℝ) (h : 4 * x * (x + 4) = 2 * 4 * x + 2 * (x + 4)) : x = 1 / 2 :=
sorry

end rectangle_area_perimeter_eq_l433_43386


namespace quadratic_has_two_distinct_real_roots_iff_l433_43378

theorem quadratic_has_two_distinct_real_roots_iff (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 - 2 * x1 + k - 1 = 0 ∧ x2 * x2 - 2 * x2 + k - 1 = 0) ↔ k < 2 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_iff_l433_43378


namespace solve_for_x_l433_43307

theorem solve_for_x (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := 
by
  sorry

end solve_for_x_l433_43307


namespace lcm_of_4_8_9_10_l433_43360

theorem lcm_of_4_8_9_10 : Nat.lcm (Nat.lcm 4 8) (Nat.lcm 9 10) = 360 := by
  sorry

end lcm_of_4_8_9_10_l433_43360


namespace complement_P_l433_43387

def U : Set ℝ := Set.univ

def P : Set ℝ := {x | x^2 < 1}

theorem complement_P : (U \ P) = Set.Iic (-1) ∪ Set.Ici 1 := by
  sorry

end complement_P_l433_43387


namespace black_area_fraction_after_three_changes_l433_43399

theorem black_area_fraction_after_three_changes
  (initial_black_area : ℚ)
  (change_factor : ℚ)
  (h1 : initial_black_area = 1)
  (h2 : change_factor = 2 / 3)
  : (change_factor ^ 3) * initial_black_area = 8 / 27 := 
by
  sorry

end black_area_fraction_after_three_changes_l433_43399


namespace minValue_at_least_9_minValue_is_9_l433_43349

noncomputable def minValue (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4) : ℝ :=
  1 / a + 4 / b + 9 / c

theorem minValue_at_least_9 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4) :
  minValue a b c h_pos h_sum ≥ 9 :=
by
  sorry

theorem minValue_is_9 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4)
  (h_abc : a = 2/3 ∧ b = 4/3 ∧ c = 2) : minValue a b c h_pos h_sum = 9 :=
by
  sorry

end minValue_at_least_9_minValue_is_9_l433_43349


namespace room_length_l433_43326

def area_four_walls (L: ℕ) (w: ℕ) (h: ℕ) : ℕ :=
  2 * (L * h) + 2 * (w * h)

def area_door (d_w: ℕ) (d_h: ℕ) : ℕ :=
  d_w * d_h

def area_windows (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) : ℕ :=
  num_windows * (win_w * win_h)

def total_area_to_whitewash (L: ℕ) (w: ℕ) (h: ℕ) (d_w: ℕ) (d_h: ℕ) (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) : ℕ :=
  area_four_walls L w h - area_door d_w d_h - area_windows win_w win_h num_windows

theorem room_length (cost: ℕ) (rate: ℕ) (w: ℕ) (h: ℕ) (d_w: ℕ) (d_h: ℕ) (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) (L: ℕ) :
  cost = rate * total_area_to_whitewash L w h d_w d_h win_w win_h num_windows →
  L = 25 :=
by
  have h1 : total_area_to_whitewash 25 15 12 6 3 4 3 3 = 24 * 25 + 306 := sorry
  have h2 : rate * (24 * 25 + 306) = 5436 := sorry
  sorry

end room_length_l433_43326


namespace original_salary_condition_l433_43324

variable (S: ℝ)

theorem original_salary_condition (h: 1.10 * 1.08 * 0.95 * 0.93 * S = 6270) :
  S = 6270 / (1.10 * 1.08 * 0.95 * 0.93) :=
by
  sorry

end original_salary_condition_l433_43324


namespace quadratic_equation_roots_l433_43313

theorem quadratic_equation_roots (a b c : ℝ) : 
  (b ^ 6 > 4 * (a ^ 3) * (c ^ 3)) → (b ^ 10 > 4 * (a ^ 5) * (c ^ 5)) :=
by
  sorry

end quadratic_equation_roots_l433_43313


namespace find_f_g_3_l433_43397

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x - 2

theorem find_f_g_3 : f (g 3) = 51 := 
by 
  sorry

end find_f_g_3_l433_43397


namespace prod_gcd_lcm_eq_864_l433_43317

theorem prod_gcd_lcm_eq_864 : 
  let a := 24
  let b := 36
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * lcm_ab = 864 :=
by
  sorry

end prod_gcd_lcm_eq_864_l433_43317


namespace chocolate_eggs_total_weight_l433_43391

def total_weight_after_discarding_box_b : ℕ :=
  let weight_large := 14
  let weight_medium := 10
  let weight_small := 6
  let box_A_weight := 4 * weight_large + 2 * weight_medium
  let box_B_weight := 6 * weight_small + 2 * weight_large
  let box_C_weight := 4 * weight_large + 3 * weight_medium
  let box_D_weight := 4 * weight_medium + 4 * weight_small
  let box_E_weight := 4 * weight_small + 2 * weight_medium
  box_A_weight + box_C_weight + box_D_weight + box_E_weight

theorem chocolate_eggs_total_weight : total_weight_after_discarding_box_b = 270 := by
  sorry

end chocolate_eggs_total_weight_l433_43391


namespace xiao_ming_shopping_l433_43316

theorem xiao_ming_shopping :
  ∃ x : ℕ, x ≤ 16 ∧ 6 * x ≤ 100 ∧ 100 - 6 * x = 28 :=
by
  -- Given that:
  -- 1. x is the same amount spent in each of the six stores.
  -- 2. Total money spent, 6 * x, must be less than or equal to 100.
  -- 3. We seek to prove that Xiao Ming has 28 yuan left.
  sorry

end xiao_ming_shopping_l433_43316


namespace binom_15_4_l433_43343

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the theorem stating the specific binomial coefficient value
theorem binom_15_4 : binom 15 4 = 1365 :=
by
  sorry

end binom_15_4_l433_43343


namespace smallest_n_mod_equality_l433_43375

theorem smallest_n_mod_equality :
  ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 5]) ∧ (∀ m : ℕ, 0 < m ∧ m < n → (7^m ≡ m^7 [MOD 5]) = false) :=
  sorry

end smallest_n_mod_equality_l433_43375
