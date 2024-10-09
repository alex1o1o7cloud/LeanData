import Mathlib

namespace roots_polynomial_sum_pow_l730_73041

open Real

theorem roots_polynomial_sum_pow (a b : ℝ) (h : a^2 - 5 * a + 6 = 0) (h_b : b^2 - 5 * b + 6 = 0) :
  a^5 + a^4 * b + b^5 = -16674 := by
sorry

end roots_polynomial_sum_pow_l730_73041


namespace combined_original_price_of_books_l730_73056

theorem combined_original_price_of_books (p1 p2 : ℝ) (h1 : p1 / 8 = 8) (h2 : p2 / 9 = 9) :
  p1 + p2 = 145 :=
sorry

end combined_original_price_of_books_l730_73056


namespace exists_isosceles_triangle_containing_l730_73009

variables {A B C X Y Z : Type} [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]

noncomputable def triangle (a b c : A) := a + b + c

def is_triangle (a b c : A) := a + b > c ∧ b + c > a ∧ c + a > b

def isosceles_triangle (a b c : A) := (a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem exists_isosceles_triangle_containing
  (a b c : A)
  (h1 : a < 1)
  (h2 : b < 1)
  (h3 : c < 1)
  (h_ABC : is_triangle a b c)
  : ∃ (x y z : A), isosceles_triangle x y z ∧ is_triangle x y z ∧ a < x ∧ b < y ∧ c < z ∧ x < 1 ∧ y < 1 ∧ z < 1 :=
sorry

end exists_isosceles_triangle_containing_l730_73009


namespace determine_S5_l730_73055

noncomputable def S (x : ℝ) (m : ℕ) : ℝ := x^m + 1 / x^m

theorem determine_S5 (x : ℝ) (h : x + 1 / x = 3) : S x 5 = 123 :=
by
  sorry

end determine_S5_l730_73055


namespace smallest_number_divisible_l730_73075

theorem smallest_number_divisible (x : ℕ) : 
  (∃ x, x + 7 % 8 = 0 ∧ x + 7 % 11 = 0 ∧ x + 7 % 24 = 0) ∧
  (∀ y, (y + 7 % 8 = 0 ∧ y + 7 % 11 = 0 ∧ y + 7 % 24 = 0) → 257 ≤ y) :=
by { sorry }

end smallest_number_divisible_l730_73075


namespace employee_B_payment_l730_73003

theorem employee_B_payment (total_payment A_payment B_payment : ℝ) 
    (h1 : total_payment = 450) 
    (h2 : A_payment = 1.5 * B_payment) 
    (h3 : total_payment = A_payment + B_payment) : 
    B_payment = 180 := 
by
  sorry

end employee_B_payment_l730_73003


namespace fraction_paint_used_second_week_l730_73065

noncomputable def total_paint : ℕ := 360
noncomputable def paint_used_first_week : ℕ := total_paint / 4
noncomputable def remaining_paint_after_first_week : ℕ := total_paint - paint_used_first_week
noncomputable def total_paint_used : ℕ := 135
noncomputable def paint_used_second_week : ℕ := total_paint_used - paint_used_first_week
noncomputable def remaining_paint_after_first_week_fraction : ℚ := paint_used_second_week / remaining_paint_after_first_week

theorem fraction_paint_used_second_week : remaining_paint_after_first_week_fraction = 1 / 6 := by
  sorry

end fraction_paint_used_second_week_l730_73065


namespace smallest_odd_integer_of_set_l730_73083

theorem smallest_odd_integer_of_set (S : Set Int) 
  (h1 : ∃ m : Int, m ∈ S ∧ m = 149)
  (h2 : ∃ n : Int, n ∈ S ∧ n = 159)
  (h3 : ∀ a b : Int, a ∈ S → b ∈ S → a ≠ b → (a - b) % 2 = 0) : 
  ∃ s : Int, s ∈ S ∧ s = 137 :=
by sorry

end smallest_odd_integer_of_set_l730_73083


namespace problem_solution_l730_73040

theorem problem_solution (a b c d : ℝ) (h1 : ab + bc + cd + da = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end problem_solution_l730_73040


namespace profit_per_meter_is_35_l730_73001

-- defining the conditions
def meters_sold : ℕ := 85
def selling_price : ℕ := 8925
def cost_price_per_meter : ℕ := 70
def total_cost_price := cost_price_per_meter * meters_sold
def total_selling_price := selling_price
def total_profit := total_selling_price - total_cost_price
def profit_per_meter := total_profit / meters_sold

-- Theorem stating the profit per meter of cloth
theorem profit_per_meter_is_35 : profit_per_meter = 35 := 
by
  sorry

end profit_per_meter_is_35_l730_73001


namespace negation_of_existential_l730_73010

theorem negation_of_existential :
  (¬ (∃ x : ℝ, x^2 - x - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
sorry

end negation_of_existential_l730_73010


namespace initial_students_began_contest_l730_73029

theorem initial_students_began_contest
  (n : ℕ)
  (first_round_fraction : ℚ)
  (second_round_fraction : ℚ)
  (remaining_students : ℕ) :
  first_round_fraction * second_round_fraction * n = remaining_students →
  remaining_students = 18 →
  first_round_fraction = 0.3 →
  second_round_fraction = 0.5 →
  n = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_students_began_contest_l730_73029


namespace number_in_eighth_group_l730_73081

theorem number_in_eighth_group (employees groups n l group_size numbering_drawn starting_number: ℕ) 
(h1: employees = 200) 
(h2: groups = 40) 
(h3: n = 5) 
(h4: number_in_fifth_group = 23) 
(h5: starting_number + 4 * n = number_in_fifth_group) : 
  starting_number + 7 * n = 38 :=
by
  sorry

end number_in_eighth_group_l730_73081


namespace count_positive_integers_in_range_l730_73021

theorem count_positive_integers_in_range :
  ∃ (count : ℕ), count = 11 ∧
    ∀ (n : ℕ), 300 < n^2 ∧ n^2 < 800 → (n ≥ 18 ∧ n ≤ 28) :=
by
  sorry

end count_positive_integers_in_range_l730_73021


namespace books_per_shelf_l730_73046

def initial_coloring_books : ℕ := 86
def sold_coloring_books : ℕ := 37
def shelves : ℕ := 7

theorem books_per_shelf : (initial_coloring_books - sold_coloring_books) / shelves = 7 := by
  sorry

end books_per_shelf_l730_73046


namespace lucy_additional_kilometers_l730_73064

theorem lucy_additional_kilometers
  (mary_distance : ℚ := (3/8) * 24)
  (edna_distance : ℚ := (2/3) * mary_distance)
  (lucy_distance : ℚ := (5/6) * edna_distance) :
  (mary_distance - lucy_distance) = 4 :=
by
  sorry

end lucy_additional_kilometers_l730_73064


namespace least_hourly_number_l730_73044

def is_clock_equivalent (a b : ℕ) : Prop := (a - b) % 12 = 0

theorem least_hourly_number : ∃ n ≥ 6, is_clock_equivalent n (n * n) ∧ ∀ m ≥ 6, is_clock_equivalent m (m * m) → 9 ≤ m → n = 9 := 
by
  sorry

end least_hourly_number_l730_73044


namespace smallest_number_l730_73036

theorem smallest_number (N : ℤ) : (∃ (k : ℤ), N = 24 * k + 34) ∧ ∀ n, (∃ (k : ℤ), n = 24 * k + 10) -> n ≥ 34 := sorry

end smallest_number_l730_73036


namespace min_value_x2y2z2_l730_73033

open Real

noncomputable def condition (x y z : ℝ) : Prop := (1 / x + 1 / y + 1 / z = 3)

theorem min_value_x2y2z2 (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : condition x y z) :
  x^2 * y^2 * z^2 ≥ 1 / 64 :=
by
  sorry

end min_value_x2y2z2_l730_73033


namespace nonneg_real_sum_inequality_l730_73035

theorem nonneg_real_sum_inequality (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end nonneg_real_sum_inequality_l730_73035


namespace choir_members_count_l730_73049

theorem choir_members_count : 
  ∃ n : ℕ, 120 ≤ n ∧ n ≤ 300 ∧
    n % 6 = 1 ∧
    n % 8 = 5 ∧
    n % 9 = 2 ∧
    n = 241 :=
by
  -- Proof will follow
  sorry

end choir_members_count_l730_73049


namespace correct_option_is_A_l730_73063

def second_quadrant (p : ℝ × ℝ) : Prop :=
p.1 < 0 ∧ p.2 > 0

def point_A : ℝ × ℝ := (-1, 2)
def point_B : ℝ × ℝ := (-3, 0)
def point_C : ℝ × ℝ := (0, 4)
def point_D : ℝ × ℝ := (5, -6)

theorem correct_option_is_A :
  (second_quadrant point_A) ∧
  ¬(second_quadrant point_B) ∧
  ¬(second_quadrant point_C) ∧
  ¬(second_quadrant point_D) :=
by sorry

end correct_option_is_A_l730_73063


namespace range_of_m_l730_73002

-- Define the conditions:

/-- Proposition p: the equation represents an ellipse with foci on y-axis -/
def proposition_p (m : ℝ) : Prop :=
  0 < m ∧ m < 9 ∧ 9 - m > 2 * m ∧ 2 * m > 0

/-- Proposition q: the eccentricity of the hyperbola is in the interval (\sqrt(3)/2, \sqrt(2)) -/
def proposition_q (m : ℝ) : Prop :=
  0 < m ∧ (5 / 2 < m ∧ m < 5)

def p_or_q (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def p_and_q (m : ℝ) : Prop := proposition_p m ∧ proposition_q m

-- Mathematically equivalent proof problem in Lean 4:

theorem range_of_m (m : ℝ) : (p_or_q m ∧ ¬p_and_q m) ↔ (m ∈ Set.Ioc 0 (5 / 2) ∪ Set.Icc 3 5) := sorry

end range_of_m_l730_73002


namespace inequality_proof_l730_73087

theorem inequality_proof
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 ≥ x2)
  (h2 : x2 ≥ x3)
  (h3 : x3 ≥ x4)
  (h4 : x2 + x3 + x4 ≥ x1) :
  (x1 + x2 + x3 + x4)^2 ≤ 4 * x1 * x2 * x3 * x4 := 
by
  sorry

end inequality_proof_l730_73087


namespace problem_statement_l730_73022

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Conditions of the problem
def cond1 : Prop := (1 / x) + (1 / y) = 2
def cond2 : Prop := (x * y) + x - y = 6

-- The corresponding theorem to prove: x² - y² = 2
theorem problem_statement (h1 : cond1) (h2 : cond2) : x^2 - y^2 = 2 :=
  sorry

end problem_statement_l730_73022


namespace car_R_average_speed_l730_73095

theorem car_R_average_speed 
  (R P S: ℝ)
  (h1: S = 2 * P)
  (h2: P + 2 = R)
  (h3: P = R + 10)
  (h4: S = R + 20) :
  R = 25 :=
by 
  sorry

end car_R_average_speed_l730_73095


namespace actual_selling_price_l730_73031

-- Define the original price m
variable (m : ℝ)

-- Define the discount rate
def discount_rate : ℝ := 0.2

-- Define the selling price
def selling_price := m * (1 - discount_rate)

-- The theorem states the relationship between the original price and the selling price after discount
theorem actual_selling_price : selling_price m = 0.8 * m :=
by
-- Proof step would go here
sorry

end actual_selling_price_l730_73031


namespace y_in_terms_of_x_l730_73037

theorem y_in_terms_of_x (p x y : ℝ) (h1 : x = 2 + 2^p) (h2 : y = 1 + 2^(-p)) : 
  y = (x-1)/(x-2) :=
by
  sorry

end y_in_terms_of_x_l730_73037


namespace product_equals_one_l730_73059

theorem product_equals_one (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1 / (1 + x + x^2)) + (1 / (1 + y + y^2)) + (1 / (1 + x + y)) = 1) : 
  x * y = 1 :=
by
  sorry

end product_equals_one_l730_73059


namespace maximum_withdraw_l730_73039

theorem maximum_withdraw (initial_amount withdraw deposit : ℕ) (h_initial : initial_amount = 500)
    (h_withdraw : withdraw = 300) (h_deposit : deposit = 198) :
    ∃ x y : ℕ, initial_amount - x * withdraw + y * deposit ≥ 0 ∧ initial_amount - x * withdraw + y * deposit = 194 ∧ initial_amount - x * withdraw = 300 := sorry

end maximum_withdraw_l730_73039


namespace part1_part2_1_part2_2_l730_73042

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a - 2) * x + 4
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x + b - 3) / (a * x^2 + 2)

theorem part1 (a : ℝ) (b : ℝ) :
  (∀ x, f x a = f (-x) a) → b = 3 :=
by sorry

theorem part2_1 (a : ℝ) (b : ℝ) :
  a = 2 → b = 3 →
  ∀ x₁ x₂, -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧ x₁ < x₂ → g x₁ a b < g x₂ a b :=
by sorry

theorem part2_2 (a : ℝ) (b : ℝ) (t : ℝ) :
  a = 2 → b = 3 →
  g (t - 1) a b + g (2 * t) a b < 0 →
  0 < t ∧ t < 1 / 3 :=
by sorry

end part1_part2_1_part2_2_l730_73042


namespace peter_can_transfer_all_money_into_two_accounts_peter_cannot_always_transfer_all_money_into_one_account_l730_73099

-- Define the conditions
variable (a b c : ℕ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)

-- Part 1
theorem peter_can_transfer_all_money_into_two_accounts :
  ∃ x y, (x + y = a + b + c ∧ y = 0) ∨
          (∃ z, (a + b + c = x + y + z ∧ y = 0 ∧ z = 0)) :=
  sorry

-- Part 2
theorem peter_cannot_always_transfer_all_money_into_one_account :
  ((a + b + c) % 2 = 1 → ¬ ∃ x, x = a + b + c) :=
  sorry

end peter_can_transfer_all_money_into_two_accounts_peter_cannot_always_transfer_all_money_into_one_account_l730_73099


namespace initial_children_on_bus_l730_73088

-- Definitions based on conditions
variable (x : ℕ) -- number of children who got off the bus
variable (y : ℕ) -- initial number of children on the bus
variable (after_exchange : ℕ := 30) -- number of children on the bus after exchange
variable (got_on : ℕ := 82) -- number of children who got on the bus
variable (extra_on : ℕ := 2) -- extra children who got on compared to got off

-- Problem translated to Lean 4 statement
theorem initial_children_on_bus (h : got_on = x + extra_on) (hx : y + got_on - x = after_exchange) : y = 28 :=
by
  sorry

end initial_children_on_bus_l730_73088


namespace integer_count_between_cubes_l730_73027

-- Definitions and conditions
def a : ℝ := 10.7
def b : ℝ := 10.8

-- Precomputed values
def a_cubed : ℝ := 1225.043
def b_cubed : ℝ := 1259.712

-- The theorem to prove
theorem integer_count_between_cubes (ha : a ^ 3 = a_cubed) (hb : b ^ 3 = b_cubed) :
  let start := Int.ceil a_cubed
  let end_ := Int.floor b_cubed
  end_ - start + 1 = 34 :=
by
  sorry

end integer_count_between_cubes_l730_73027


namespace sequence_sum_l730_73028

theorem sequence_sum (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) : S 6 - S 2 = 40 :=
by
  sorry

end sequence_sum_l730_73028


namespace marked_price_l730_73030

theorem marked_price (x : ℝ) (payment : ℝ) (discount : ℝ) (hx : (payment = 90) ∧ ((x ≤ 100 ∧ discount = 0.1) ∨ (x > 100 ∧ discount = 0.2))) :
  (x = 100 ∨ x = 112.5) := by
  sorry

end marked_price_l730_73030


namespace division_multiplication_example_l730_73026

theorem division_multiplication_example : 120 / 4 / 2 * 3 = 45 := by
  sorry

end division_multiplication_example_l730_73026


namespace term_of_arithmetic_sequence_l730_73034

variable (a₁ : ℕ) (d : ℕ) (n : ℕ)

theorem term_of_arithmetic_sequence (h₁: a₁ = 2) (h₂: d = 5) (h₃: n = 50) :
    a₁ + (n - 1) * d = 247 := by
  sorry

end term_of_arithmetic_sequence_l730_73034


namespace tanya_erasers_l730_73090

theorem tanya_erasers (H R TR T : ℕ) 
  (h1 : H = 2 * R) 
  (h2 : R = TR / 2 - 3) 
  (h3 : H = 4) 
  (h4 : TR = T / 2) : 
  T = 20 := 
by 
  sorry

end tanya_erasers_l730_73090


namespace jogger_ahead_of_train_l730_73085

theorem jogger_ahead_of_train (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (time_to_pass : ℝ) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 100) 
  (h4 : time_to_pass = 34) : 
  ∃ d : ℝ, d = 240 :=
by
  sorry

end jogger_ahead_of_train_l730_73085


namespace division_result_l730_73066

def numerator : ℕ := 3 * 4 * 5
def denominator : ℕ := 2 * 3
def quotient : ℕ := numerator / denominator

theorem division_result : quotient = 10 := by
  sorry

end division_result_l730_73066


namespace max_value_of_x3_div_y4_l730_73007

theorem max_value_of_x3_div_y4 (x y : ℝ) (h1 : 3 ≤ x * y^2) (h2 : x * y^2 ≤ 8) (h3 : 4 ≤ x^2 / y) (h4 : x^2 / y ≤ 9) :
  ∃ (k : ℝ), k = 27 ∧ ∀ (z : ℝ), z = x^3 / y^4 → z ≤ k :=
by
  sorry

end max_value_of_x3_div_y4_l730_73007


namespace triangle_angle_A_l730_73091

theorem triangle_angle_A (AC BC : ℝ) (angle_B : ℝ) (h_AC : AC = Real.sqrt 2) (h_BC : BC = 1) (h_angle_B : angle_B = 45) :
  ∃ (angle_A : ℝ), angle_A = 30 :=
by
  sorry

end triangle_angle_A_l730_73091


namespace ellipse_focal_distance_l730_73074

theorem ellipse_focal_distance (m : ℝ) :
  (∀ x y : ℝ, (x^2 / 16 + y^2 / m = 1) ∧ (2 * Real.sqrt (16 - m) = 2 * Real.sqrt 7)) → m = 9 :=
by
  intro h
  sorry

end ellipse_focal_distance_l730_73074


namespace find_number_l730_73084

theorem find_number (x : ℚ) (h : (3 * x / 2) + 6 = 11) : x = 10 / 3 :=
sorry

end find_number_l730_73084


namespace triangle_area_l730_73097

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a + b > c) (h3 : a + c > b) (h4 : b + c > a) : 
  a = 3 ∧ b = 4 ∧ c = 5 ∨ a = 4 ∧ b = 3 ∧ c = 5 ∨ a = 5 ∧ b = 4 ∧ c = 3 ∨
  a = 5 ∧ b = 3 ∧ c = 4 ∨ a = 4 ∧ b = 5 ∧ c = 3 ∨ a = 3 ∧ b = 5 ∧ c = 4 → 
  (1 / 2 : ℝ) * ↑a * ↑b = 6 := by
  sorry

end triangle_area_l730_73097


namespace pie_remaining_portion_l730_73089

theorem pie_remaining_portion (Carlos_share Maria_share remaining: ℝ)
  (hCarlos : Carlos_share = 0.65)
  (hRemainingAfterCarlos : remaining = 1 - Carlos_share)
  (hMaria : Maria_share = remaining / 2) :
  remaining - Maria_share = 0.175 :=
by
  sorry

end pie_remaining_portion_l730_73089


namespace rotated_angle_l730_73080

theorem rotated_angle (angle_ACB_initial : ℝ) (rotation_angle : ℝ) (h1 : angle_ACB_initial = 60) (h2 : rotation_angle = 630) : 
  ∃ (angle_ACB_new : ℝ), angle_ACB_new = 30 :=
by
  -- Define the effective rotation
  let effective_rotation := rotation_angle % 360 -- Modulo operation
  
  -- Calculate the new angle
  let angle_new := angle_ACB_initial + effective_rotation
  
  -- Ensure the angle is acute by converting if needed
  let acute_angle_new := if angle_new > 180 then 360 - angle_new else angle_new
  
  -- The acute angle should be 30 degrees
  use acute_angle_new
  have : acute_angle_new = 30 := sorry
  exact this

end rotated_angle_l730_73080


namespace range_of_a_l730_73023

-- Define the propositions
def p (x : ℝ) := (x - 1) * (x - 2) > 0
def q (a x : ℝ) := x^2 + (a - 1) * x - a > 0

-- Define the solution sets
def A := {x : ℝ | p x}
def B (a : ℝ) := {x : ℝ | q a x}

-- State the proof problem
theorem range_of_a (a : ℝ) : 
  (∀ x, p x → q a x) ∧ (∃ x, ¬p x ∧ q a x) → -2 < a ∧ a ≤ -1 :=
by
  sorry

end range_of_a_l730_73023


namespace minimize_quadratic_l730_73086

theorem minimize_quadratic (x : ℝ) :
  (∀ y : ℝ, x^2 + 14*x + 6 ≤ y^2 + 14*y + 6) ↔ x = -7 :=
by
  sorry

end minimize_quadratic_l730_73086


namespace james_twitch_income_l730_73078

theorem james_twitch_income :
  let tier1_base := 120
  let tier2_base := 50
  let tier3_base := 30
  let tier1_gifted := 10
  let tier2_gifted := 25
  let tier3_gifted := 15
  let tier1_new := tier1_base + tier1_gifted
  let tier2_new := tier2_base + tier2_gifted
  let tier3_new := tier3_base + tier3_gifted
  let tier1_income := tier1_new * 4.99
  let tier2_income := tier2_new * 9.99
  let tier3_income := tier3_new * 24.99
  let total_income := tier1_income + tier2_income + tier3_income
  total_income = 2522.50 :=
by
  sorry

end james_twitch_income_l730_73078


namespace work_efficiency_ratio_l730_73098

theorem work_efficiency_ratio (A B : ℝ) (k : ℝ)
  (h1 : A = k * B)
  (h2 : B = 1 / 27)
  (h3 : A + B = 1 / 9) :
  k = 2 :=
by
  sorry

end work_efficiency_ratio_l730_73098


namespace steve_speed_during_race_l730_73012

theorem steve_speed_during_race 
  (distance_gap : ℝ) 
  (john_speed : ℝ) 
  (time : ℝ) 
  (john_ahead : ℝ)
  (steve_speed : ℝ) :
  distance_gap = 16 →
  john_speed = 4.2 →
  time = 36 →
  john_ahead = 2 →
  steve_speed = (151.2 - 18) / 36 :=
by
  sorry

end steve_speed_during_race_l730_73012


namespace majority_vote_is_280_l730_73093

-- Definitions based on conditions from step (a)
def totalVotes : ℕ := 1400
def winningPercentage : ℝ := 0.60
def losingPercentage : ℝ := 0.40

-- Majority computation based on the winning and losing percentages
def majorityVotes : ℝ := totalVotes * winningPercentage - totalVotes * losingPercentage

-- Theorem statement
theorem majority_vote_is_280 : majorityVotes = 280 := by
  sorry

end majority_vote_is_280_l730_73093


namespace students_not_solving_any_problem_l730_73014

variable (A_0 A_1 A_2 A_3 A_4 A_5 A_6 : ℕ)

-- Given conditions
def number_of_students := 2006
def condition_1 := A_1 = 4 * A_2
def condition_2 := A_2 = 4 * A_3
def condition_3 := A_3 = 4 * A_4
def condition_4 := A_4 = 4 * A_5
def condition_5 := A_5 = 4 * A_6
def total_students := A_0 + A_1 = 2006

-- The final statement to be proven
theorem students_not_solving_any_problem : 
  (A_1 = 4 * A_2) →
  (A_2 = 4 * A_3) →
  (A_3 = 4 * A_4) →
  (A_4 = 4 * A_5) →
  (A_5 = 4 * A_6) →
  (A_0 + A_1 = 2006) →
  (A_0 = 982) :=
by
  intro h1 h2 h3 h4 h5 h6
  -- Proof should go here
  sorry

end students_not_solving_any_problem_l730_73014


namespace lcm_of_132_and_315_l730_73094

def n1 : ℕ := 132
def n2 : ℕ := 315

theorem lcm_of_132_and_315 :
  (Nat.lcm n1 n2) = 13860 :=
by
  -- Proof goes here
  sorry

end lcm_of_132_and_315_l730_73094


namespace cost_difference_of_buses_l730_73069

-- Definitions from the conditions
def bus_cost_equations (x y : ℝ) :=
  (x + 2 * y = 260) ∧ (2 * x + y = 280)

-- The statement to prove
theorem cost_difference_of_buses (x y : ℝ) (h : bus_cost_equations x y) :
  x - y = 20 :=
sorry

end cost_difference_of_buses_l730_73069


namespace find_m_range_l730_73050

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + 2 * (m - 1) * x + 2 

theorem find_m_range (m : ℝ) : (∀ x ≤ 4, f x m ≤ f (x + 1) m) → m ≤ -3 :=
by
  sorry

end find_m_range_l730_73050


namespace ping_pong_tournament_l730_73062

theorem ping_pong_tournament :
  ∃ n: ℕ, 
    (∃ m: ℕ, m ≥ 0 ∧ m ≤ 2 ∧ 2 * n + m = 29) ∧
    n = 14 ∧
    (n + 2 = 16) := 
by {
  sorry
}

end ping_pong_tournament_l730_73062


namespace tumblers_count_correct_l730_73045

section MrsPetersonsTumblers

-- Define the cost of one tumbler
def tumbler_cost : ℕ := 45

-- Define the amount paid in total by Mrs. Petersons
def total_paid : ℕ := 5 * 100

-- Define the change received by Mrs. Petersons
def change_received : ℕ := 50

-- Calculate the total amount spent
def total_spent : ℕ := total_paid - change_received

-- Calculate the number of tumblers bought
def tumblers_bought : ℕ := total_spent / tumbler_cost

-- Prove the number of tumblers bought is 10
theorem tumblers_count_correct : tumblers_bought = 10 :=
  by
    -- Proof steps will be filled here
    sorry

end MrsPetersonsTumblers

end tumblers_count_correct_l730_73045


namespace find_min_k_l730_73077

theorem find_min_k (k : ℕ) 
  (h1 : k > 0) 
  (h2 : ∀ (A : Finset ℕ), A ⊆ (Finset.range 26).erase 0 → A.card = k → ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (2 / 3 : ℝ) ≤ x / y ∧ x / y ≤ (3 / 2 : ℝ)) : 
  k = 7 :=
by {
  sorry
}

end find_min_k_l730_73077


namespace total_weight_four_pets_l730_73053

-- Define the weights
def Evan_dog := 63
def Ivan_dog := Evan_dog / 7
def combined_weight_dogs := Evan_dog + Ivan_dog
def Kara_cat := combined_weight_dogs * 5
def combined_weight_dogs_and_cat := Evan_dog + Ivan_dog + Kara_cat
def Lisa_parrot := combined_weight_dogs_and_cat * 3
def total_weight := Evan_dog + Ivan_dog + Kara_cat + Lisa_parrot

-- Total weight of the four pets
theorem total_weight_four_pets : total_weight = 1728 := by
  sorry

end total_weight_four_pets_l730_73053


namespace option_A_option_B_option_C_option_D_l730_73016

theorem option_A (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n, a (n + 1) = a n + n + 1) : a 20 = 211 :=
sorry

theorem option_B (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2^n * a n) : a 5 = 2^10 :=
sorry

theorem option_C (S : ℕ → ℝ) (h₀ : ∀ n, S n = 3^n + 1/2) : ¬(∃ r : ℝ, ∀ n, S n = S 1 * r ^ (n - 1)) :=
sorry

theorem option_D (S : ℕ → ℝ) (a : ℕ → ℝ) (h₀ : S 1 = 1) 
  (h₁ : ∀ n, S n = n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1))
  (h₂ : (S 8) / 8 - (S 4) / 4 = 8) : a 6 = 21 :=
sorry

end option_A_option_B_option_C_option_D_l730_73016


namespace fourth_root_sum_of_square_roots_eq_l730_73025

theorem fourth_root_sum_of_square_roots_eq :
  (1 + Real.sqrt 2 + Real.sqrt 3) = 
    Real.sqrt (Real.sqrt 6400 + Real.sqrt 6144 + Real.sqrt 4800 + Real.sqrt 4608) ^ 4 :=
by
  sorry

end fourth_root_sum_of_square_roots_eq_l730_73025


namespace people_got_off_at_second_stop_l730_73054

theorem people_got_off_at_second_stop (x : ℕ) :
  (10 - x) + 20 - 18 + 2 = 12 → x = 2 :=
  by sorry

end people_got_off_at_second_stop_l730_73054


namespace arthur_walked_distance_in_miles_l730_73004

def blocks_west : ℕ := 8
def blocks_south : ℕ := 10
def block_length_in_miles : ℚ := 1 / 4

theorem arthur_walked_distance_in_miles : 
  (blocks_west + blocks_south) * block_length_in_miles = 4.5 := by
sorry

end arthur_walked_distance_in_miles_l730_73004


namespace prove_mutually_exclusive_l730_73068

def bag : List String := ["red", "red", "red", "black", "black"]

def at_least_one_black (drawn : List String) : Prop :=
  "black" ∈ drawn

def all_red (drawn : List String) : Prop :=
  ∀ b ∈ drawn, b = "red"

def events_mutually_exclusive : Prop :=
  ∀ drawn, at_least_one_black drawn → ¬all_red drawn

theorem prove_mutually_exclusive :
  events_mutually_exclusive
:= by
  sorry

end prove_mutually_exclusive_l730_73068


namespace factor_polynomial_l730_73071

theorem factor_polynomial :
  ∀ u : ℝ, (u^4 - 81 * u^2 + 144) = (u^2 - 72) * (u - 3) * (u + 3) :=
by
  intro u
  -- Establish the polynomial and its factorization in Lean
  have h : u^4 - 81 * u^2 + 144 = (u^2 - 72) * (u - 3) * (u + 3) := sorry
  exact h

end factor_polynomial_l730_73071


namespace find_f_neg_5_l730_73005

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_domain : ∀ x : ℝ, true)
variable (h_positive : ∀ x : ℝ, x > 0 → f x = log 5 x + 1)

theorem find_f_neg_5 : f (-5) = -2 :=
by
  sorry

end find_f_neg_5_l730_73005


namespace probability_at_least_one_visits_guangzhou_l730_73024

-- Define the probabilities of visiting for persons A, B, and C
def p_A : ℚ := 2 / 3
def p_B : ℚ := 1 / 4
def p_C : ℚ := 3 / 5

-- Calculate the probability that no one visits
def p_not_A : ℚ := 1 - p_A
def p_not_B : ℚ := 1 - p_B
def p_not_C : ℚ := 1 - p_C

-- Calculate the probability that at least one person visits
def p_none_visit : ℚ := p_not_A * p_not_B * p_not_C
def p_at_least_one_visit : ℚ := 1 - p_none_visit

-- The statement we need to prove
theorem probability_at_least_one_visits_guangzhou : p_at_least_one_visit = 9 / 10 :=
by 
  sorry

end probability_at_least_one_visits_guangzhou_l730_73024


namespace spaces_per_row_l730_73058

theorem spaces_per_row 
  (kind_of_tomatoes : ℕ)
  (tomatoes_per_kind : ℕ)
  (kind_of_cucumbers : ℕ)
  (cucumbers_per_kind : ℕ)
  (potatoes : ℕ)
  (rows : ℕ)
  (additional_spaces : ℕ)
  (h1 : kind_of_tomatoes = 3)
  (h2 : tomatoes_per_kind = 5)
  (h3 : kind_of_cucumbers = 5)
  (h4 : cucumbers_per_kind = 4)
  (h5 : potatoes = 30)
  (h6 : rows = 10)
  (h7 : additional_spaces = 85) :
  (kind_of_tomatoes * tomatoes_per_kind + kind_of_cucumbers * cucumbers_per_kind + potatoes + additional_spaces) / rows = 15 :=
by
  sorry

end spaces_per_row_l730_73058


namespace unique_solution_l730_73038

theorem unique_solution (a b c : ℝ) (hb : b ≠ 2) (hc : c ≠ 0) : 
  ∃! x : ℝ, 4 * x - 7 + a = 2 * b * x + c ∧ x = (c + 7 - a) / (4 - 2 * b) :=
by
  sorry

end unique_solution_l730_73038


namespace inequality_abc_squared_l730_73067

theorem inequality_abc_squared (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : 2 * (a + b + c + d) ≥ a * b * c * d) : 
  a^2 + b^2 + c^2 + d^2 ≥ a * b * c * d := 
sorry

end inequality_abc_squared_l730_73067


namespace pythagorean_triple_345_l730_73048

theorem pythagorean_triple_345 : (3^2 + 4^2 = 5^2) := 
by 
  -- Here, the proof will be filled in, but we use 'sorry' for now.
  sorry

end pythagorean_triple_345_l730_73048


namespace number_leaves_remainder_five_l730_73051

theorem number_leaves_remainder_five (k : ℕ) (n : ℕ) (least_num : ℕ) 
  (h₁ : least_num = 540)
  (h₂ : ∀ m, m % 12 = 5 → m ≥ least_num)
  (h₃ : n = 107) 
  : 540 % 107 = 5 :=
by sorry

end number_leaves_remainder_five_l730_73051


namespace axis_of_symmetry_shifted_cos_l730_73043

noncomputable def shifted_cos_axis_symmetry (x : ℝ) : Prop :=
  ∃ k : ℤ, x = k * (Real.pi / 2) - (Real.pi / 12)

theorem axis_of_symmetry_shifted_cos :
  shifted_cos_axis_symmetry x :=
sorry

end axis_of_symmetry_shifted_cos_l730_73043


namespace probability_one_solves_l730_73017

theorem probability_one_solves :
  let pA := 0.8
  let pB := 0.7
  (pA * (1 - pB) + pB * (1 - pA)) = 0.38 :=
by
  sorry

end probability_one_solves_l730_73017


namespace david_ate_more_than_emma_l730_73019

-- Definitions and conditions
def contestants : Nat := 8
def pies_david_ate : Nat := 8
def pies_emma_ate : Nat := 2
def pies_by_david (contestants pies_david_ate: Nat) : Prop := pies_david_ate = 8
def pies_by_emma (contestants pies_emma_ate: Nat) : Prop := pies_emma_ate = 2

-- Theorem statement
theorem david_ate_more_than_emma (contestants pies_david_ate pies_emma_ate : Nat) (h_david : pies_by_david contestants pies_david_ate) (h_emma : pies_by_emma contestants pies_emma_ate) : pies_david_ate - pies_emma_ate = 6 :=
by
  sorry

end david_ate_more_than_emma_l730_73019


namespace power_of_two_ends_with_identical_digits_l730_73072

theorem power_of_two_ends_with_identical_digits : ∃ (k : ℕ), k ≥ 10 ∧ (∀ (x y : ℕ), 2^k = 1000 * x + 111 * y → y = 8 → (2^k % 1000 = 888)) :=
by sorry

end power_of_two_ends_with_identical_digits_l730_73072


namespace determinant_modified_l730_73096

variable (a b c d : ℝ)

theorem determinant_modified (h : a * d - b * c = 10) :
  (a + 2 * c) * d - (b + 3 * d) * c = 10 - c * d := by
  sorry

end determinant_modified_l730_73096


namespace c_share_of_profit_l730_73057

-- Definitions for the investments and total profit
def investments_a := 800
def investments_b := 1000
def investments_c := 1200
def total_profit := 1000

-- Definition for the share of profits based on the ratio of investments
def share_of_c : ℕ :=
  let ratio_a := 4
  let ratio_b := 5
  let ratio_c := 6
  let total_ratio := ratio_a + ratio_b + ratio_c
  (ratio_c * total_profit) / total_ratio

-- The theorem to be proved
theorem c_share_of_profit : share_of_c = 400 := by
  sorry

end c_share_of_profit_l730_73057


namespace number_of_subsets_with_four_adj_chairs_l730_73032

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l730_73032


namespace relay_team_orderings_l730_73000

theorem relay_team_orderings (Jordan Mike Friend1 Friend2 Friend3 : Type) :
  ∃ n : ℕ, n = 12 :=
by
  -- Define the team members
  let team : List Type := [Jordan, Mike, Friend1, Friend2, Friend3]
  
  -- Define the number of ways to choose the 4th and 5th runners
  let ways_choose_45 := 2
  
  -- Define the number of ways to order the first 3 runners
  let ways_order_123 := Nat.factorial 3
  
  -- Calculate the total number of ways
  let total_ways := ways_choose_45 * ways_order_123
  
  -- The total ways should be 12
  use total_ways
  have h : total_ways = 12
  sorry
  exact h

end relay_team_orderings_l730_73000


namespace gcd_36_54_l730_73079

-- Add a theorem stating the problem to prove that the gcd of 36 and 54 is 18
theorem gcd_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcd_36_54_l730_73079


namespace intersection_M_N_l730_73070

open Set Real

def M := {x : ℝ | x^2 + x - 6 < 0}
def N := {x : ℝ | abs (x - 1) ≤ 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l730_73070


namespace calculate_total_money_l730_73082

noncomputable def cost_per_gumdrop : ℕ := 4
noncomputable def number_of_gumdrops : ℕ := 20
noncomputable def total_money : ℕ := 80

theorem calculate_total_money : 
  cost_per_gumdrop * number_of_gumdrops = total_money := 
by
  sorry

end calculate_total_money_l730_73082


namespace domain_composite_l730_73076

-- Define the conditions
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

-- The theorem statement
theorem domain_composite (h : ∀ x, domain_f x → 0 ≤ x ∧ x ≤ 4) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2) :=
by
  sorry

end domain_composite_l730_73076


namespace positive_multiples_of_11_ending_with_7_l730_73018

-- Definitions for conditions
def is_multiple_of_11 (n : ℕ) : Prop := (n % 11 = 0)
def ends_with_7 (n : ℕ) : Prop := (n % 10 = 7)

-- Main theorem statement
theorem positive_multiples_of_11_ending_with_7 :
  ∃ n, (n = 13) ∧ ∀ k, is_multiple_of_11 k ∧ ends_with_7 k ∧ 0 < k ∧ k < 1500 → k = 77 + (k / 110) * 110 := 
sorry

end positive_multiples_of_11_ending_with_7_l730_73018


namespace maxProfitAchievable_l730_73015

namespace BarrelProduction

structure ProductionPlan where
  barrelsA : ℕ
  barrelsB : ℕ

def profit (plan : ProductionPlan) : ℕ :=
  300 * plan.barrelsA + 400 * plan.barrelsB

def materialAUsage (plan : ProductionPlan) : ℕ :=
  plan.barrelsA + 2 * plan.barrelsB

def materialBUsage (plan : ProductionPlan) : ℕ :=
  2 * plan.barrelsA + plan.barrelsB

def isValidPlan (plan : ProductionPlan) : Prop :=
  materialAUsage plan ≤ 12 ∧ materialBUsage plan ≤ 12

def maximumProfit : ℕ :=
  2800

theorem maxProfitAchievable : 
  ∃ (plan : ProductionPlan), isValidPlan plan ∧ profit plan = maximumProfit :=
sorry

end BarrelProduction

end maxProfitAchievable_l730_73015


namespace best_years_to_scrap_l730_73052

-- Define the conditions from the problem
def purchase_cost : ℕ := 150000
def annual_cost : ℕ := 15000
def maintenance_initial : ℕ := 3000
def maintenance_difference : ℕ := 3000

-- Define the total_cost function
def total_cost (n : ℕ) : ℕ :=
  purchase_cost + annual_cost * n + (n * (2 * maintenance_initial + (n - 1) * maintenance_difference)) / 2

-- Define the average annual cost function
def average_annual_cost (n : ℕ) : ℕ :=
  total_cost n / n

-- Statement to be proven: the best number of years to minimize average annual cost is 10
theorem best_years_to_scrap : 
  (∀ n : ℕ, average_annual_cost 10 ≤ average_annual_cost n) :=
by
  sorry
  
end best_years_to_scrap_l730_73052


namespace quadratic_real_roots_l730_73008

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k + 1) * x^2 - 2 * x + 1 = 0) → (k ≤ 0 ∧ k ≠ -1) :=
by
  sorry

end quadratic_real_roots_l730_73008


namespace find_x_from_percentage_l730_73013

theorem find_x_from_percentage : 
  ∃ x : ℚ, 0.65 * x = 0.20 * 487.50 := 
sorry

end find_x_from_percentage_l730_73013


namespace nat_gt_10_is_diff_of_hypotenuse_numbers_l730_73020

def is_hypotenuse_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

theorem nat_gt_10_is_diff_of_hypotenuse_numbers (n : ℕ) (h : n > 10) : 
  ∃ (n₁ n₂ : ℕ), is_hypotenuse_number n₁ ∧ is_hypotenuse_number n₂ ∧ n = n₁ - n₂ :=
by
  sorry

end nat_gt_10_is_diff_of_hypotenuse_numbers_l730_73020


namespace value_of_x_minus_2y_l730_73060

theorem value_of_x_minus_2y (x y : ℝ) (h1 : 0.5 * x = y + 20) : x - 2 * y = 40 :=
by
  sorry

end value_of_x_minus_2y_l730_73060


namespace halfway_fraction_between_is_one_fourth_l730_73047

theorem halfway_fraction_between_is_one_fourth : 
  let f1 := (1 / 4 : ℚ)
  let f2 := (1 / 6 : ℚ)
  let f3 := (1 / 3 : ℚ)
  ((f1 + f2 + f3) / 3) = (1 / 4) := 
by
  let f1 := (1 / 4 : ℚ)
  let f2 := (1 / 6 : ℚ)
  let f3 := (1 / 3 : ℚ)
  sorry

end halfway_fraction_between_is_one_fourth_l730_73047


namespace unique_n_degree_polynomial_exists_l730_73073

theorem unique_n_degree_polynomial_exists (n : ℕ) (h : n > 0) :
  ∃! (f : Polynomial ℝ), Polynomial.degree f = n ∧
    f.eval 0 = 1 ∧
    ∀ x : ℝ, (x + 1) * (f.eval x)^2 - 1 = -((x + 1) * (f.eval (-x))^2 - 1) := 
sorry

end unique_n_degree_polynomial_exists_l730_73073


namespace total_bill_is_correct_l730_73092

def number_of_adults : ℕ := 2
def number_of_children : ℕ := 5
def meal_cost : ℕ := 8

-- Define total number of people
def total_people : ℕ := number_of_adults + number_of_children

-- Define the total bill
def total_bill : ℕ := total_people * meal_cost

-- Theorem stating the total bill amount
theorem total_bill_is_correct : total_bill = 56 := by
  sorry

end total_bill_is_correct_l730_73092


namespace b_money_used_for_10_months_l730_73011

theorem b_money_used_for_10_months
  (a_capital_ratio : ℚ)
  (a_time_used : ℕ)
  (b_profit_share : ℚ)
  (h1 : a_capital_ratio = 1 / 4)
  (h2 : a_time_used = 15)
  (h3 : b_profit_share = 2 / 3) :
  ∃ (b_time_used : ℕ), b_time_used = 10 :=
by
  sorry

end b_money_used_for_10_months_l730_73011


namespace ball_radius_and_surface_area_l730_73061

theorem ball_radius_and_surface_area (d h r : ℝ) (radius_eq : d / 2 = 6) (depth_eq : h = 2) 
  (pythagorean : (r - h)^2 + (d / 2)^2 = r^2) :
  r = 10 ∧ (4 * Real.pi * r^2 = 400 * Real.pi) :=
by
  sorry

end ball_radius_and_surface_area_l730_73061


namespace total_nap_duration_l730_73006

def nap1 : ℚ := 1 / 5
def nap2 : ℚ := 1 / 4
def nap3 : ℚ := 1 / 6
def hour_to_minutes : ℚ := 60

theorem total_nap_duration :
  (nap1 + nap2 + nap3) * hour_to_minutes = 37 := by
  sorry

end total_nap_duration_l730_73006
