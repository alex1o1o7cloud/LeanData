import Mathlib

namespace product_of_four_integers_l1314_131420

theorem product_of_four_integers (A B C D : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) (h_pos_D : 0 < D)
  (h_sum : A + B + C + D = 36)
  (h_eq1 : A + 2 = B - 2)
  (h_eq2 : B - 2 = C * 2)
  (h_eq3 : C * 2 = D / 2) :
  A * B * C * D = 3840 :=
by
  sorry

end product_of_four_integers_l1314_131420


namespace cups_of_sugar_already_put_in_l1314_131478

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

end cups_of_sugar_already_put_in_l1314_131478


namespace Joan_initial_money_l1314_131447

def cost_hummus (containers : ℕ) (price_per_container : ℕ) : ℕ := containers * price_per_container
def cost_apple (quantity : ℕ) (price_per_apple : ℕ) : ℕ := quantity * price_per_apple

theorem Joan_initial_money 
  (containers_of_hummus : ℕ)
  (price_per_hummus : ℕ)
  (cost_chicken : ℕ)
  (cost_bacon : ℕ)
  (cost_vegetables : ℕ)
  (quantity_apple : ℕ)
  (price_per_apple : ℕ)
  (total_cost : ℕ)
  (remaining_money : ℕ):
  containers_of_hummus = 2 →
  price_per_hummus = 5 →
  cost_chicken = 20 →
  cost_bacon = 10 →
  cost_vegetables = 10 →
  quantity_apple = 5 →
  price_per_apple = 2 →
  remaining_money = cost_apple quantity_apple price_per_apple →
  total_cost = cost_hummus containers_of_hummus price_per_hummus + cost_chicken + cost_bacon + cost_vegetables + remaining_money →
  total_cost = 60 :=
by
  intros
  sorry

end Joan_initial_money_l1314_131447


namespace calculation_result_l1314_131462

theorem calculation_result :
  3 * 15 + 3 * 16 + 3 * 19 + 11 = 161 :=
sorry

end calculation_result_l1314_131462


namespace highest_probability_two_out_of_three_probability_l1314_131460

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

end highest_probability_two_out_of_three_probability_l1314_131460


namespace ball_reaches_less_than_5_l1314_131488

noncomputable def height_after_bounces (initial_height : ℕ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial_height * (ratio ^ bounces)

theorem ball_reaches_less_than_5 (initial_height : ℕ) (ratio : ℝ) (k : ℕ) (target_height : ℝ) (stop_height : ℝ) 
  (h_initial : initial_height = 500) (h_ratio : ratio = 0.6) (h_target : target_height = 5) (h_stop : stop_height = 0.1) :
  ∃ n, height_after_bounces initial_height ratio n < target_height ∧ 500 * (0.6 ^ 17) < stop_height := by
  sorry

end ball_reaches_less_than_5_l1314_131488


namespace trains_meet_in_time_l1314_131433

noncomputable def time_to_meet (length1 length2 distance_between speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := distance_between + length1 + length2
  total_distance / relative_speed

theorem trains_meet_in_time :
  time_to_meet 150 250 850 110 130 = 18.75 :=
by 
  -- here would go the proof steps, but since we are not required,
  sorry

end trains_meet_in_time_l1314_131433


namespace price_per_glass_first_day_l1314_131464

theorem price_per_glass_first_day (O W : ℝ) (P1 P2 : ℝ) 
  (h1 : O = W) 
  (h2 : P2 = 0.40)
  (revenue_eq : 2 * O * P1 = 3 * O * P2) 
  : P1 = 0.60 := 
by 
  sorry

end price_per_glass_first_day_l1314_131464


namespace tax_free_amount_l1314_131401

theorem tax_free_amount (X : ℝ) (total_value : ℝ) (tax_paid : ℝ) (tax_rate : ℝ) 
(h1 : total_value = 1720) 
(h2 : tax_paid = 134.4) 
(h3 : tax_rate = 0.12) 
(h4 : tax_paid = tax_rate * (total_value - X)) 
: X = 600 := 
sorry

end tax_free_amount_l1314_131401


namespace airplane_seat_difference_l1314_131463

theorem airplane_seat_difference (F C X : ℕ) 
    (h1 : 387 = F + 310) 
    (h2 : C = 310) 
    (h3 : C = 4 * F + X) :
    X = 2 :=
by
    sorry

end airplane_seat_difference_l1314_131463


namespace range_of_m_l1314_131468

theorem range_of_m (m : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → (x - 1) * (x - (m - 1)) > 0) → m > 1 :=
by
  intro h
  sorry

end range_of_m_l1314_131468


namespace bottom_row_bricks_l1314_131410

theorem bottom_row_bricks (n : ℕ) 
  (h1 : (n + (n-1) + (n-2) + (n-3) + (n-4) = 200)) : 
  n = 42 := 
by sorry

end bottom_row_bricks_l1314_131410


namespace minimum_days_to_pay_back_l1314_131449

theorem minimum_days_to_pay_back (x : ℕ) : 
  (50 + 5 * x ≥ 150) → x = 20 :=
sorry

end minimum_days_to_pay_back_l1314_131449


namespace first_month_sale_l1314_131486

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

end first_month_sale_l1314_131486


namespace right_triangle_area_l1314_131479

theorem right_triangle_area
    (h : ∀ {a b c : ℕ}, a^2 + b^2 = c^2 → c = 13 → a = 5 ∨ b = 5)
    (hypotenuse : ℕ)
    (leg : ℕ)
    (hypotenuse_eq : hypotenuse = 13)
    (leg_eq : leg = 5) : ∃ (area: ℕ), area = 30 :=
by
  -- The proof will go here.
  sorry

end right_triangle_area_l1314_131479


namespace largest_three_digit_divisible_by_6_l1314_131461

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

end largest_three_digit_divisible_by_6_l1314_131461


namespace coordinates_reflect_y_axis_l1314_131422

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem coordinates_reflect_y_axis (p : ℝ × ℝ) (h : p = (5, 2)) : reflect_y_axis p = (-5, 2) :=
by
  rw [h]
  rfl

end coordinates_reflect_y_axis_l1314_131422


namespace problem1_problem2_l1314_131427

open Real -- Open the Real namespace for trigonometric functions

-- Part 1: Prove cos(5π + α) * tan(α - 7π) = 4/5 given π < α < 2π and cos α = 3/5
theorem problem1 (α : ℝ) (hα1 : π < α) (hα2 : α < 2 * π) (hcos : cos α = 3 / 5) : 
  cos (5 * π + α) * tan (α - 7 * π) = 4 / 5 := sorry

-- Part 2: Prove sin(π/3 + α) = √3/3 given cos (π/6 - α) = √3/3
theorem problem2 (α : ℝ) (hcos : cos (π / 6 - α) = sqrt 3 / 3) : 
  sin (π / 3 + α) = sqrt 3 / 3 := sorry

end problem1_problem2_l1314_131427


namespace hyperbola_properties_l1314_131493

theorem hyperbola_properties :
  let h := -3
  let k := 0
  let a := 5
  let c := Real.sqrt 50
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ h + k + a + b = 7 :=
by
  sorry

end hyperbola_properties_l1314_131493


namespace percent_dimes_value_is_60_l1314_131489

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

end percent_dimes_value_is_60_l1314_131489


namespace randi_peter_ratio_l1314_131452

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

end randi_peter_ratio_l1314_131452


namespace minimum_value_l1314_131430

open Real

theorem minimum_value (a : ℝ) (m n : ℝ) (h_a : a > 0) (h_a_not_one : a ≠ 1) 
                      (h_mn : m * n > 0) (h_point : -m - n + 1 = 0) :
  (1 / m + 2 / n) = 3 + 2 * sqrt 2 :=
by
  -- proof should go here
  sorry

end minimum_value_l1314_131430


namespace average_velocity_eq_l1314_131419

noncomputable def motion_eq : ℝ → ℝ := λ t => 1 - t + t^2

theorem average_velocity_eq (Δt : ℝ) :
  (motion_eq (3 + Δt) - motion_eq 3) / Δt = 5 + Δt :=
by
  sorry

end average_velocity_eq_l1314_131419


namespace total_fencing_cost_l1314_131499

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

end total_fencing_cost_l1314_131499


namespace square_numbers_divisible_by_5_between_20_and_110_l1314_131432

theorem square_numbers_divisible_by_5_between_20_and_110 :
  ∃ (y : ℕ), (y = 25 ∨ y = 100) ∧ (∃ (n : ℕ), y = n^2) ∧ 5 ∣ y ∧ 20 < y ∧ y < 110 :=
by
  sorry

end square_numbers_divisible_by_5_between_20_and_110_l1314_131432


namespace galya_overtakes_sasha_l1314_131414

variable {L : ℝ} -- Length of the track
variable (Sasha_uphill_speed : ℝ := 8)
variable (Sasha_downhill_speed : ℝ := 24)
variable (Galya_uphill_speed : ℝ := 16)
variable (Galya_downhill_speed : ℝ := 18)

noncomputable def average_speed (uphill_speed: ℝ) (downhill_speed: ℝ) : ℝ :=
  1 / ((1 / (4 * uphill_speed)) + (3 / (4 * downhill_speed)))

noncomputable def time_for_one_lap (L: ℝ) (speed: ℝ) : ℝ :=
  L / speed

theorem galya_overtakes_sasha 
  (L_pos : 0 < L) :
  let v_Sasha := average_speed Sasha_uphill_speed Sasha_downhill_speed
  let v_Galya := average_speed Galya_uphill_speed Galya_downhill_speed
  let t_Sasha := time_for_one_lap L v_Sasha
  let t_Galya := time_for_one_lap L v_Galya
  (L * 11 / v_Galya) < (L * 10 / v_Sasha) :=
by
  sorry

end galya_overtakes_sasha_l1314_131414


namespace no_valid_pairs_l1314_131406

open Nat

theorem no_valid_pairs (l y : ℕ) (h1 : y % 30 = 0) (h2 : l > 1) :
  (∃ n m : ℕ, 180 - 360 / n = y ∧ 180 - 360 / m = l * y ∧ y * l ≤ 180) → False := 
by
  intro h
  sorry

end no_valid_pairs_l1314_131406


namespace find_n_l1314_131404

theorem find_n : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 11] ∧ n = 3 :=
by
  sorry

end find_n_l1314_131404


namespace find_m_l1314_131448

-- Define vectors a and b
def a (m : ℝ) : ℝ × ℝ := (2, -m)
def b : ℝ × ℝ := (1, 3)

-- Define the condition for perpendicular vectors
def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the problem
theorem find_m (m : ℝ) (h : is_perpendicular (a m + b) b) : m = 4 :=
sorry -- proof omitted

end find_m_l1314_131448


namespace ratio_boys_to_girls_l1314_131491

variable (g b : ℕ)

theorem ratio_boys_to_girls (h1 : b = g + 9) (h2 : g + b = 25) : b / g = 17 / 8 := by
  -- Proof goes here
  sorry

end ratio_boys_to_girls_l1314_131491


namespace a_term_b_value_c_value_d_value_l1314_131442

theorem a_term (a x : ℝ) (h1 : a * (x + 1) = x^3 + 3 * x^2 + 3 * x + 1) : a = x^2 + 2 * x + 1 :=
sorry

theorem b_value (a x b : ℝ) (h1 : a - 1 = 0) (h2 : x = 0 ∨ x = b) : b = -2 :=
sorry

theorem c_value (p c b : ℝ) (h1 : p * c^4 = 32) (h2 : p * c = b^2) (h3 : 0 < c) : c = 2 :=
sorry

theorem d_value (A B d : ℝ) (P : ℝ → ℝ) (c : ℝ) (h1 : P (A * B) = P A + P B) (h2 : P A = 1) (h3 : P B = c) (h4 : A = 10^ P A) (h5 : B = 10^ P B) (h6 : d = A * B) : d = 1000 :=
sorry

end a_term_b_value_c_value_d_value_l1314_131442


namespace geometric_sequence_ratio_l1314_131412

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h0 : q ≠ 1) 
  (h1 : ∀ n, S n = a 0 * (1 - q^n) / (1 - q)) 
  (h2 : ∀ n, a n = a 0 * q^n) 
  (h3 : 2 * S 3 = 7 * a 2) :
  (S 5 / a 2 = 31 / 2) ∨ (S 5 / a 2 = 31 / 8) :=
by sorry

end geometric_sequence_ratio_l1314_131412


namespace smallest_b_l1314_131477

noncomputable def geometric_sequence : Prop :=
∃ (a b c r : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ b = a * r ∧ c = a * r^2 ∧ a * b * c = 216

theorem smallest_b (a b c r: ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_geom: b = a * r ∧ c = a * r^2 ∧ a * b * c = 216) : b = 6 :=
sorry

end smallest_b_l1314_131477


namespace prime_gt_three_modulus_l1314_131445

theorem prime_gt_three_modulus (p : ℕ) (hp : Prime p) (hp_gt3 : p > 3) : (p^2 + 12) % 12 = 1 := by
  sorry

end prime_gt_three_modulus_l1314_131445


namespace algebraic_expression_value_l1314_131476

-- Definitions for the problem conditions
def x := -1
def y := 1 / 2
def expr := 2 * (x^2 - 5 * x * y) - 3 * (x^2 - 6 * x * y)

-- The problem statement to be proved
theorem algebraic_expression_value : expr = 3 :=
by
  sorry

end algebraic_expression_value_l1314_131476


namespace problem1_problem2_problem3_l1314_131475

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

end problem1_problem2_problem3_l1314_131475


namespace cuboid_ratio_l1314_131429

theorem cuboid_ratio (length breadth height: ℕ) (h_length: length = 90) (h_breadth: breadth = 75) (h_height: height = 60) : 
(length / Nat.gcd length (Nat.gcd breadth height) = 6) ∧ 
(breadth / Nat.gcd length (Nat.gcd breadth height) = 5) ∧ 
(height / Nat.gcd length (Nat.gcd breadth height) = 4) := by 
  -- intentionally skipped proof 
  sorry

end cuboid_ratio_l1314_131429


namespace target_average_income_l1314_131440

variable (past_incomes : List ℕ) (next_average : ℕ)

def total_past_income := past_incomes.sum
def total_next_income := next_average * 5
def total_ten_week_income := total_past_income past_incomes + total_next_income next_average

theorem target_average_income (h1 : past_incomes = [406, 413, 420, 436, 395])
                              (h2 : next_average = 586) :
  total_ten_week_income past_incomes next_average / 10 = 500 := by
  sorry

end target_average_income_l1314_131440


namespace sum_of_inverses_inequality_l1314_131409

theorem sum_of_inverses_inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum_eq : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 :=
sorry

end sum_of_inverses_inequality_l1314_131409


namespace limping_rook_adjacent_sum_not_divisible_by_4_l1314_131416

/-- Problem statement: A limping rook traversed a 10 × 10 board,
visiting each square exactly once with numbers 1 through 100
written in the order visited.
Prove that the sum of the numbers in any two adjacent cells
is not divisible by 4. -/
theorem limping_rook_adjacent_sum_not_divisible_by_4 :
  ∀ (board : Fin 10 → Fin 10 → ℕ), 
  (∀ (i j : Fin 10), 1 ≤ board i j ∧ board i j ≤ 100) →
  (∀ (i j : Fin 10), (∃ (i' : Fin 10), i = i' + 1 ∨ i = i' - 1)
                 ∨ (∃ (j' : Fin 10), j = j' + 1 ∨ j = j' - 1)) →
  ((∀ (i j : Fin 10) (k l : Fin 10),
      (i = k ∧ (j = l + 1 ∨ j = l - 1) ∨ j = l ∧ (i = k + 1 ∨ i = k - 1)) →
      (board i j + board k l) % 4 ≠ 0)) :=
by
  sorry

end limping_rook_adjacent_sum_not_divisible_by_4_l1314_131416


namespace multiple_of_9_l1314_131473

noncomputable def digit_sum (x : ℕ) : ℕ := sorry  -- Placeholder for the digit sum function

theorem multiple_of_9 (n : ℕ) (h1 : digit_sum n = digit_sum (3 * n))
  (h2 : ∀ x, x % 9 = digit_sum x % 9) :
  n % 9 = 0 :=
by
  sorry

end multiple_of_9_l1314_131473


namespace daisy_dog_toys_l1314_131439

-- Given conditions
def dog_toys_monday : ℕ := 5
def dog_toys_tuesday_left : ℕ := 3
def dog_toys_tuesday_bought : ℕ := 3
def dog_toys_wednesday_all_found : ℕ := 13

-- The question we need to answer
def dog_toys_bought_wednesday : ℕ := 7

-- Statement to prove
theorem daisy_dog_toys :
  (dog_toys_monday - dog_toys_tuesday_left + dog_toys_tuesday_left + dog_toys_tuesday_bought + dog_toys_bought_wednesday = dog_toys_wednesday_all_found) :=
sorry

end daisy_dog_toys_l1314_131439


namespace trapezoidal_section_length_l1314_131466

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

end trapezoidal_section_length_l1314_131466


namespace pq_sufficient_not_necessary_l1314_131474

theorem pq_sufficient_not_necessary (p q : Prop) :
  (¬ (p ∨ q)) → (¬ p ∧ ¬ q) ∧ ¬ ((¬ p ∧ ¬ q) → (¬ (p ∨ q))) :=
sorry

end pq_sufficient_not_necessary_l1314_131474


namespace primes_eq_condition_l1314_131443

theorem primes_eq_condition (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) : 
  p + q^2 = r^4 → p = 7 ∧ q = 3 ∧ r = 2 := 
by
  sorry

end primes_eq_condition_l1314_131443


namespace import_tax_paid_l1314_131408

theorem import_tax_paid (total_value excess_value tax_rate tax_paid : ℝ)
  (h₁ : total_value = 2590)
  (h₂ : excess_value = total_value - 1000)
  (h₃ : tax_rate = 0.07)
  (h₄ : tax_paid = excess_value * tax_rate) : 
  tax_paid = 111.30 := by
  -- variables
  sorry

end import_tax_paid_l1314_131408


namespace sasha_prediction_l1314_131498

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

end sasha_prediction_l1314_131498


namespace Y_minus_X_eq_92_l1314_131400

def arithmetic_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def X : ℕ := arithmetic_sum 10 2 46
def Y : ℕ := arithmetic_sum 12 2 46

theorem Y_minus_X_eq_92 : Y - X = 92 := by
  sorry

end Y_minus_X_eq_92_l1314_131400


namespace line_intersects_circle_and_angle_conditions_l1314_131411

noncomputable def line_circle_intersection_condition (k : ℝ) : Prop :=
  - (Real.sqrt 3) / 3 ≤ k ∧ k ≤ (Real.sqrt 3) / 3

noncomputable def inclination_angle_condition (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

theorem line_intersects_circle_and_angle_conditions (k θ : ℝ) :
  line_circle_intersection_condition k →
  inclination_angle_condition θ →
  ∃ x y : ℝ, (y = k * (x + 1)) ∧ ((x - 1)^2 + y^2 = 1) :=
by
  sorry

end line_intersects_circle_and_angle_conditions_l1314_131411


namespace inequality_am_gm_l1314_131450

theorem inequality_am_gm (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 := by
  sorry

end inequality_am_gm_l1314_131450


namespace jeff_boxes_filled_l1314_131421

def donuts_each_day : ℕ := 10
def days : ℕ := 12
def jeff_eats_per_day : ℕ := 1
def chris_eats : ℕ := 8
def donuts_per_box : ℕ := 10

theorem jeff_boxes_filled : 
  (donuts_each_day * days - jeff_eats_per_day * days - chris_eats) / donuts_per_box = 10 :=
by
  sorry

end jeff_boxes_filled_l1314_131421


namespace solve_equation_l1314_131413

theorem solve_equation (x : ℝ) : x^4 + (3 - x)^4 = 82 ↔ x = 2.5 ∨ x = 0.5 := by
  sorry

end solve_equation_l1314_131413


namespace simplify_expression_l1314_131494

theorem simplify_expression :
  (3 * Real.sqrt 8) / 
  (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7 + Real.sqrt 9) =
  - (2 * Real.sqrt 6 - 2 * Real.sqrt 2 + 2 * Real.sqrt 14) / 5 :=
by
  sorry

end simplify_expression_l1314_131494


namespace intersection_eq_l1314_131418

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end intersection_eq_l1314_131418


namespace correct_result_value_at_neg_one_l1314_131455

theorem correct_result (x : ℝ) (A : ℝ := 3 * x^2 - x + 1) (incorrect : ℝ := 2 * x^2 - 3 * x - 2) :
  (A - (incorrect - A)) = 4 * x^2 + x + 4 :=
by sorry

theorem value_at_neg_one (x : ℝ := -1) (A : ℝ := 3 * x^2 - x + 1) (incorrect : ℝ := 2 * x^2 - 3 * x - 2) :
  (4 * x^2 + x + 4) = 7 :=
by sorry

end correct_result_value_at_neg_one_l1314_131455


namespace necessary_but_not_sufficient_condition_l1314_131457

-- Let p be the proposition |x| < 2
def p (x : ℝ) : Prop := abs x < 2

-- Let q be the proposition x^2 - x - 2 < 0
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- The proof statement
theorem necessary_but_not_sufficient_condition (x : ℝ) : q x → p x ∧ ¬ (p x → q x) := 
sorry

end necessary_but_not_sufficient_condition_l1314_131457


namespace canvas_decreased_by_40_percent_l1314_131453

noncomputable def canvas_decrease (P C : ℝ) (x d : ℝ) : Prop :=
  (P = 4 * C) ∧
  ((P - 0.60 * P) + (C - (x / 100) * C) = (1 - d / 100) * (P + C)) ∧
  (d = 55.99999999999999)

theorem canvas_decreased_by_40_percent (P C : ℝ) (x d : ℝ) 
  (h : canvas_decrease P C x d) : x = 40 :=
by
  sorry

end canvas_decreased_by_40_percent_l1314_131453


namespace multiply_expand_l1314_131437

theorem multiply_expand (x y : ℝ) :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end multiply_expand_l1314_131437


namespace find_other_number_l1314_131472

-- Given: 
-- LCM of two numbers is 2310
-- GCD of two numbers is 55
-- One number is 605,
-- Prove: The other number is 210

theorem find_other_number (a b : ℕ) (h_lcm : Nat.lcm a b = 2310) (h_gcd : Nat.gcd a b = 55) (h_b : b = 605) :
  a = 210 :=
sorry

end find_other_number_l1314_131472


namespace valid_votes_correct_l1314_131431

noncomputable def Total_votes : ℕ := 560000
noncomputable def Percentages_received : Fin 4 → ℚ 
| 0 => 0.4
| 1 => 0.35
| 2 => 0.15
| 3 => 0.1

noncomputable def Percentages_invalid : Fin 4 → ℚ 
| 0 => 0.12
| 1 => 0.18
| 2 => 0.25
| 3 => 0.3

noncomputable def Votes_received (i : Fin 4) : ℚ := Total_votes * Percentages_received i

noncomputable def Invalid_votes (i : Fin 4) : ℚ := Votes_received i * Percentages_invalid i

noncomputable def Valid_votes (i : Fin 4) : ℚ := Votes_received i - Invalid_votes i

def A_valid_votes := 197120
def B_valid_votes := 160720
def C_valid_votes := 63000
def D_valid_votes := 39200

theorem valid_votes_correct :
  Valid_votes 0 = A_valid_votes ∧
  Valid_votes 1 = B_valid_votes ∧
  Valid_votes 2 = C_valid_votes ∧
  Valid_votes 3 = D_valid_votes := by
  sorry

end valid_votes_correct_l1314_131431


namespace expression_equals_36_l1314_131456

def k := 13

theorem expression_equals_36 : 13 * (3 - 3 / 13) = 36 := by
  sorry

end expression_equals_36_l1314_131456


namespace sum_of_x_and_y_l1314_131459

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 240) : x + y = 680 :=
by
  sorry

end sum_of_x_and_y_l1314_131459


namespace projectile_first_reaches_70_feet_l1314_131441

theorem projectile_first_reaches_70_feet :
  ∃ t : ℝ , (t > 0) ∧ (-16 * t^2 + 80 * t = 70) ∧ (∀ t' : ℝ, (t' > 0) ∧ (-16 * t'^2 + 80 * t' = 70) → t ≤ t') :=
sorry

end projectile_first_reaches_70_feet_l1314_131441


namespace nephews_count_l1314_131485

theorem nephews_count (a_nephews_20_years_ago : ℕ) (third_now_nephews : ℕ) (additional_nephews : ℕ) :
  a_nephews_20_years_ago = 80 →
  third_now_nephews = 3 →
  additional_nephews = 120 →
  ∃ (a_nephews_now : ℕ) (v_nephews_now : ℕ), a_nephews_now = third_now_nephews * a_nephews_20_years_ago ∧ v_nephews_now = a_nephews_now + additional_nephews ∧ (a_nephews_now + v_nephews_now = 600) :=
by
  sorry

end nephews_count_l1314_131485


namespace simplify_expression_l1314_131465

noncomputable def a : ℝ := 2 * Real.sqrt 12 - 4 * Real.sqrt 27 + 3 * Real.sqrt 75 + 7 * Real.sqrt 8 - 3 * Real.sqrt 18
noncomputable def b : ℝ := 4 * Real.sqrt 48 - 3 * Real.sqrt 27 - 5 * Real.sqrt 18 + 2 * Real.sqrt 50

theorem simplify_expression : a * b = 97 := by
  sorry

end simplify_expression_l1314_131465


namespace range_of_a_l1314_131446

/-- 
Proof problem statement derived from the given math problem and solution:
Prove that if the conditions:
1. ∀ x > 0, x + 1/x > a
2. ∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0
3. ¬ (∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0) is false
4. (∀ x > 0, x + 1/x > a) ∧ (∃ x0 ∈ ℝ, x0^2 - 2*a*x0 + 1 ≤ 0) is false
hold, then a ≥ 2.
-/
theorem range_of_a (a : ℝ)
  (h1 : ∀ x : ℝ, x > 0 → x + 1 / x > a)
  (h2 : ∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0)
  (h3 : ¬ (¬ (∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0)))
  (h4 : ¬ ((∀ x : ℝ, x > 0 → x + 1 / x > a) ∧ (∃ x0 : ℝ, x0^2 - 2 * a * x0 + 1 ≤ 0))) :
  a ≥ 2 :=
sorry

end range_of_a_l1314_131446


namespace travel_paths_l1314_131482

-- Definitions for conditions
def roads_AB : ℕ := 3
def roads_BC : ℕ := 2

-- The theorem statement
theorem travel_paths : roads_AB * roads_BC = 6 := by
  sorry

end travel_paths_l1314_131482


namespace trip_length_l1314_131415

theorem trip_length 
  (total_time : ℝ) (canoe_speed : ℝ) (hike_speed : ℝ) (hike_distance : ℝ)
  (hike_time_eq : hike_distance / hike_speed = 5.4) 
  (canoe_time_eq : total_time - hike_distance / hike_speed = 0.1)
  (canoe_distance_eq : canoe_speed * (total_time - hike_distance / hike_speed) = 1.2)
  (total_time_val : total_time = 5.5)
  (canoe_speed_val : canoe_speed = 12)
  (hike_speed_val : hike_speed = 5)
  (hike_distance_val : hike_distance = 27) :
  total_time = 5.5 → canoe_speed = 12 → hike_speed = 5 → hike_distance = 27 → hike_distance + canoe_speed * (total_time - hike_distance / hike_speed) = 28.2 := 
by
  intro h_total_time h_canoe_speed h_hike_speed h_hike_distance
  rw [h_total_time, h_canoe_speed, h_hike_speed, h_hike_distance]
  sorry

end trip_length_l1314_131415


namespace range_of_x_l1314_131481

def f (x : ℝ) : ℝ := abs (x - 2)

theorem range_of_x (a b x : ℝ) (a_nonzero : a ≠ 0) (ab_real : a ∈ Set.univ ∧ b ∈ Set.univ) : 
  (|a + b| + |a - b| ≥ |a| • f x) ↔ (0 ≤ x ∧ x ≤ 4) :=
sorry

end range_of_x_l1314_131481


namespace harry_total_cost_l1314_131487

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

end harry_total_cost_l1314_131487


namespace area_of_triangle_ABC_l1314_131402

open Real

noncomputable def triangle_area (b c : ℝ) : ℝ :=
  (sqrt 2 / 4) * (sqrt (4 + b^2)) * (sqrt (4 + c^2))

theorem area_of_triangle_ABC (b c : ℝ) :
  let O : ℝ × ℝ × ℝ := (0, 0, 0)
  let A : ℝ × ℝ × ℝ := (2, 0, 0)
  let B : ℝ × ℝ × ℝ := (0, b, 0)
  let C : ℝ × ℝ × ℝ := (0, 0, c)
  let angle_BAC : ℝ := 45
  (cos (angle_BAC * π / 180) = sqrt 2 / 2) →
  (sin (angle_BAC * π / 180) = sqrt 2 / 2) →
  let AB := sqrt (2^2 + b^2)
  let AC := sqrt (2^2 + c^2)
  let area := (1/2) * AB * AC * (sin (45 * π / 180))
  area = triangle_area b c :=
sorry

end area_of_triangle_ABC_l1314_131402


namespace probability_all_five_dice_even_l1314_131428

-- Definitions of conditions
def standard_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Set ℕ := {2, 4, 6}

-- The statement to be proven
theorem probability_all_five_dice_even : 
  (∀ die ∈ standard_six_sided_die, (∃ n ∈ even_numbers, die = n)) → (1 / 32) = (1 / 2) ^ 5 :=
by
  intro h
  sorry

end probability_all_five_dice_even_l1314_131428


namespace B_finishes_work_in_4_days_l1314_131496

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

end B_finishes_work_in_4_days_l1314_131496


namespace no_valid_two_digit_N_exists_l1314_131417

def is_two_digit_number (N : ℕ) : Prop :=
  10 ≤ N ∧ N < 100

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ (n : ℕ), n ^ 3 = x

def reverse_digits (N : ℕ) : ℕ :=
  match N / 10, N % 10 with
  | a, b => 10 * b + a

theorem no_valid_two_digit_N_exists : ∀ N : ℕ,
  is_two_digit_number N →
  (is_perfect_cube (N - reverse_digits N) ∧ (N - reverse_digits N) ≠ 27) → false :=
by sorry

end no_valid_two_digit_N_exists_l1314_131417


namespace perfect_square_proof_l1314_131436

theorem perfect_square_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := 
sorry

end perfect_square_proof_l1314_131436


namespace total_dollar_amount_l1314_131444

/-- Definitions of base 5 numbers given in the problem -/
def pearls := 1 * 5^0 + 2 * 5^1 + 3 * 5^2 + 4 * 5^3
def silk := 1 * 5^0 + 1 * 5^1 + 1 * 5^2 + 1 * 5^3
def spices := 1 * 5^0 + 2 * 5^1 + 2 * 5^2
def maps := 0 * 5^0 + 1 * 5^1

/-- The theorem to prove the total dollar amount in base 10 -/
theorem total_dollar_amount : pearls + silk + spices + maps = 808 :=
by
  sorry

end total_dollar_amount_l1314_131444


namespace circle_radius_l1314_131435

/-
  Given:
  - The area of the circle x = π r^2
  - The circumference of the circle y = 2π r
  - The sum x + y = 72π

  Prove:
  The radius r = 6
-/
theorem circle_radius (r : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : x = π * r ^ 2) 
  (h₂ : y = 2 * π * r) 
  (h₃ : x + y = 72 * π) : 
  r = 6 := 
sorry

end circle_radius_l1314_131435


namespace intersection_slopes_l1314_131405

theorem intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (4 / 41)) ∨ m ∈ Set.Ici (Real.sqrt (4 / 41)) := 
sorry

end intersection_slopes_l1314_131405


namespace glove_ratio_l1314_131424

theorem glove_ratio (P : ℕ) (G : ℕ) (hf : P = 43) (hg : G = 2 * P) : G / P = 2 := by
  rw [hf, hg]
  norm_num
  sorry

end glove_ratio_l1314_131424


namespace floor_factorial_expression_l1314_131495

-- Mathematical definitions (conditions)
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Mathematical proof problem (statement)
theorem floor_factorial_expression :
  Int.floor ((factorial 2007 + factorial 2004 : ℚ) / (factorial 2006 + factorial 2005)) = 2006 :=
sorry

end floor_factorial_expression_l1314_131495


namespace arithmetic_progression_l1314_131403

theorem arithmetic_progression (a b c : ℝ) (h : a + c = 2 * b) :
  3 * (a^2 + b^2 + c^2) = 6 * (a - b)^2 + (a + b + c)^2 :=
by
  sorry

end arithmetic_progression_l1314_131403


namespace length_of_purple_part_l1314_131434

variables (P : ℝ) (black : ℝ) (blue : ℝ) (total_len : ℝ)

-- The conditions
def conditions := 
  black = 0.5 ∧ 
  blue = 2 ∧ 
  total_len = 4 ∧ 
  P + black + blue = total_len

-- The proof problem statement
theorem length_of_purple_part (h : conditions P 0.5 2 4) : P = 1.5 :=
sorry

end length_of_purple_part_l1314_131434


namespace stratified_sampling_community_A_l1314_131492

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

end stratified_sampling_community_A_l1314_131492


namespace distance_T_S_l1314_131480

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

end distance_T_S_l1314_131480


namespace calculate_decimal_sum_and_difference_l1314_131497

theorem calculate_decimal_sum_and_difference : 
  (0.5 + 0.003 + 0.070) - 0.008 = 0.565 := 
by 
  sorry

end calculate_decimal_sum_and_difference_l1314_131497


namespace find_PQ_l1314_131483

noncomputable def right_triangle_tan (PQ PR : ℝ) (tan_P : ℝ) (R_right : Prop) : Prop :=
  tan_P = PQ / PR ∧ R_right

theorem find_PQ (PQ PR : ℝ) (tan_P : ℝ) (R_right : Prop)
  (h1 : tan_P = 3 / 2)
  (h2 : PR = 6)
  (h3 : R_right) :
  right_triangle_tan PQ PR tan_P R_right → PQ = 9 :=
by
  sorry

end find_PQ_l1314_131483


namespace fraction_decomposition_roots_sum_l1314_131438

theorem fraction_decomposition_roots_sum :
  ∀ (p q r A B C : ℝ),
  p ≠ q → p ≠ r → q ≠ r →
  (∀ (s : ℝ), s ≠ p → s ≠ q → s ≠ r →
          1 / (s^3 - 15 * s^2 + 50 * s - 56) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 225 :=
by
  intros p q r A B C hpq hpr hqr hDecomp
  -- Skip proof
  sorry

end fraction_decomposition_roots_sum_l1314_131438


namespace area_of_square_with_diagonal_l1314_131454

theorem area_of_square_with_diagonal (d : ℝ) (s : ℝ) (hsq : d = s * Real.sqrt 2) (hdiagonal : d = 12 * Real.sqrt 2) : 
  s^2 = 144 :=
by
  -- Proof details would go here.
  sorry

end area_of_square_with_diagonal_l1314_131454


namespace quadratic_root_value_l1314_131425

theorem quadratic_root_value {m : ℝ} (h : m^2 + m - 1 = 0) : 2 * m^2 + 2 * m + 2025 = 2027 :=
sorry

end quadratic_root_value_l1314_131425


namespace middle_digit_base5_l1314_131470

theorem middle_digit_base5 {M : ℕ} (x y z : ℕ) (hx : 0 ≤ x ∧ x < 5) (hy : 0 ≤ y ∧ y < 5) (hz : 0 ≤ z ∧ z < 5)
    (h_base5 : M = 25 * x + 5 * y + z) (h_base8 : M = 64 * z + 8 * y + x) : y = 0 :=
sorry

end middle_digit_base5_l1314_131470


namespace car_maintenance_expense_l1314_131458

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

end car_maintenance_expense_l1314_131458


namespace expected_value_is_one_dollar_l1314_131490

def star_prob := 1 / 4
def moon_prob := 1 / 2
def sun_prob := 1 / 4

def star_prize := 2
def moon_prize := 4
def sun_penalty := -6

def expected_winnings := star_prob * star_prize + moon_prob * moon_prize + sun_prob * sun_penalty

theorem expected_value_is_one_dollar : expected_winnings = 1 := by
  sorry

end expected_value_is_one_dollar_l1314_131490


namespace least_four_digit_perfect_square_and_fourth_power_l1314_131423

theorem least_four_digit_perfect_square_and_fourth_power : 
    ∃ (n : ℕ), (1000 ≤ n) ∧ (n < 10000) ∧ (∃ a : ℕ, n = a^2) ∧ (∃ b : ℕ, n = b^4) ∧ 
    (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (∃ a : ℕ, m = a^2) ∧ (∃ b : ℕ, m = b^4) → n ≤ m) ∧ n = 6561 :=
by
  sorry

end least_four_digit_perfect_square_and_fourth_power_l1314_131423


namespace range_of_a_l1314_131467

noncomputable def is_decreasing (a : ℝ) : Prop :=
∀ n : ℕ, 0 < n → n ≤ 6 → (1 - 3 * a) * n + 10 * a > (1 - 3 * a) * (n + 1) + 10 * a ∧ 0 < a ∧ a < 1 ∧ ((1 - 3 * a) * 6 + 10 * a > 1)

theorem range_of_a (a : ℝ) : is_decreasing a ↔ (1/3 < a ∧ a < 5/8) :=
sorry

end range_of_a_l1314_131467


namespace coins_value_percentage_l1314_131484

theorem coins_value_percentage :
  let penny_value := 1
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let total_value_cents := (1 * penny_value) + (2 * nickel_value) + (1 * dime_value) + (2 * quarter_value)
  (total_value_cents / 100) * 100 = 71 :=
by
  sorry

end coins_value_percentage_l1314_131484


namespace inequality_amgm_l1314_131407

variable {a b c : ℝ}

theorem inequality_amgm (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) : 
  (1 / 2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) <= a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1 / 3) ∧ 
  a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1 / 3) <= (a - b)^2 + (b - c)^2 + (c - a)^2 := 
by 
  sorry

end inequality_amgm_l1314_131407


namespace proof_l1314_131451

-- Define the conditions in Lean
variable {f : ℝ → ℝ}
variable (h1 : ∀ x ∈ (Set.Ioi 0), 0 ≤ f x)
variable (h2 : ∀ x ∈ (Set.Ioi 0), x * f x + f x ≤ 0)

-- Formulate the goal
theorem proof (a b : ℝ) (ha : a ∈ (Set.Ioi 0)) (hb : b ∈ (Set.Ioi 0)) (h : a < b) : 
    b * f a ≤ a * f b :=
by
  sorry  -- Proof omitted

end proof_l1314_131451


namespace number_of_female_students_selected_is_20_l1314_131471

noncomputable def number_of_female_students_to_be_selected
(total_students : ℕ) (female_students : ℕ) (students_to_be_selected : ℕ) : ℕ :=
students_to_be_selected * female_students / total_students

theorem number_of_female_students_selected_is_20 :
  number_of_female_students_to_be_selected 2000 800 50 = 20 := 
by
  sorry

end number_of_female_students_selected_is_20_l1314_131471


namespace fred_balloons_remaining_l1314_131469

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

end fred_balloons_remaining_l1314_131469


namespace expiry_time_correct_l1314_131426

def factorial (n : Nat) : Nat := match n with
| 0 => 1
| n + 1 => (n + 1) * factorial n

def seconds_in_a_day : Nat := 86400
def seconds_in_an_hour : Nat := 3600
def donation_time_seconds : Nat := 8 * seconds_in_an_hour
def expiry_seconds : Nat := factorial 8

def time_of_expiry (donation_time : Nat) (expiry_time : Nat) : Nat :=
  (donation_time + expiry_time) % seconds_in_a_day

def time_to_HM (time_seconds : Nat) : Nat × Nat :=
  let hours := time_seconds / seconds_in_an_hour
  let minutes := (time_seconds % seconds_in_an_hour) / 60
  (hours, minutes)

def is_correct_expiry_time : Prop :=
  let (hours, minutes) := time_to_HM (time_of_expiry donation_time_seconds expiry_seconds)
  hours = 19 ∧ minutes = 12

theorem expiry_time_correct : is_correct_expiry_time := by
  sorry

end expiry_time_correct_l1314_131426
