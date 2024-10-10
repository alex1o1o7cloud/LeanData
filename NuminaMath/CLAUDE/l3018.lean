import Mathlib

namespace fifth_term_of_geometric_sequence_l3018_301841

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_second_term : a 2 = 1/3)
  (h_eighth_term : a 8 = 27) :
  a 5 = 3 ∨ a 5 = -3 :=
sorry

end fifth_term_of_geometric_sequence_l3018_301841


namespace lcm_gcd_1560_1040_l3018_301836

theorem lcm_gcd_1560_1040 :
  (Nat.lcm 1560 1040 = 1560) ∧ (Nat.gcd 1560 1040 = 520) := by
  sorry

end lcm_gcd_1560_1040_l3018_301836


namespace trees_in_yard_l3018_301878

theorem trees_in_yard (yard_length : ℕ) (tree_distance : ℕ) (h1 : yard_length = 300) (h2 : tree_distance = 12) : 
  yard_length / tree_distance + 1 = 26 := by
  sorry

end trees_in_yard_l3018_301878


namespace isosceles_triangle_l3018_301863

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- State the theorem
theorem isosceles_triangle (t : Triangle) 
  (h : 2 * Real.cos t.B * Real.sin t.A = Real.sin t.C) : 
  t.A = t.B := by
  sorry

end isosceles_triangle_l3018_301863


namespace gcd_factorial_problem_l3018_301881

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 11) / (Nat.factorial 4)) = 5040 := by
  sorry

end gcd_factorial_problem_l3018_301881


namespace arithmetic_sequence_first_term_l3018_301880

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_6th : a 6 = 9)
  (h_3rd : a 3 = 3 * a 2) :
  a 1 = -1 := by
sorry

end arithmetic_sequence_first_term_l3018_301880


namespace count_special_numbers_eq_252_l3018_301805

/-- The count of numbers between 1000 and 9999 with four different digits 
    in either strictly increasing or strictly decreasing order -/
def count_special_numbers : ℕ := sorry

/-- A number is considered special if it has four different digits 
    in either strictly increasing or strictly decreasing order -/
def is_special (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  (∃ a b c d : ℕ, 
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    ((a < b ∧ b < c ∧ c < d) ∨ (a > b ∧ b > c ∧ c > d)))

theorem count_special_numbers_eq_252 : 
  count_special_numbers = 252 :=
sorry

end count_special_numbers_eq_252_l3018_301805


namespace fixed_point_of_exponential_function_l3018_301837

theorem fixed_point_of_exponential_function (a : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-2) + 3
  f 2 = 4 :=
by
  sorry

end fixed_point_of_exponential_function_l3018_301837


namespace circle_radius_l3018_301832

theorem circle_radius (x y : ℝ) : 
  (x^2 - 10*x + y^2 + 4*y + 13 = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 4^2 :=
sorry

end circle_radius_l3018_301832


namespace a_investment_amount_verify_profit_share_l3018_301865

/-- Represents the investment scenario described in the problem -/
structure Investment where
  a_amount : ℝ  -- A's investment amount
  b_amount : ℝ := 200  -- B's investment amount
  a_months : ℝ := 12  -- Duration of A's investment in months
  b_months : ℝ := 6  -- Duration of B's investment in months
  total_profit : ℝ := 100  -- Total profit
  a_profit : ℝ := 50  -- A's share of the profit

/-- Theorem stating that A's investment amount must be $100 given the conditions -/
theorem a_investment_amount (inv : Investment) : 
  (inv.a_amount * inv.a_months) / (inv.b_amount * inv.b_months) = 1 →
  inv.a_amount = 100 := by
  sorry

/-- Corollary confirming that the calculated investment satisfies the profit sharing condition -/
theorem verify_profit_share (inv : Investment) : 
  inv.a_amount = 100 →
  (inv.a_amount * inv.a_months) / (inv.b_amount * inv.b_months) = 1 := by
  sorry

end a_investment_amount_verify_profit_share_l3018_301865


namespace kim_dropped_one_class_l3018_301884

/-- Calculates the number of classes dropped given the initial number of classes,
    hours per class, and remaining total hours of classes. -/
def classes_dropped (initial_classes : ℕ) (hours_per_class : ℕ) (remaining_hours : ℕ) : ℕ :=
  (initial_classes * hours_per_class - remaining_hours) / hours_per_class

theorem kim_dropped_one_class :
  classes_dropped 4 2 6 = 1 := by
  sorry

end kim_dropped_one_class_l3018_301884


namespace sum_of_digits_N_l3018_301892

/-- The sum of digits function for natural numbers -/
noncomputable def sumOfDigits (n : ℕ) : ℕ := sorry

/-- N is defined as the positive integer whose square is 36^49 * 49^36 * 81^25 -/
noncomputable def N : ℕ := sorry

/-- Theorem stating that the sum of digits of N is 21 -/
theorem sum_of_digits_N : sumOfDigits N = 21 := by sorry

end sum_of_digits_N_l3018_301892


namespace average_expenditure_feb_to_jul_l3018_301864

def average_expenditure_jan_to_jun : ℝ := 4200
def expenditure_january : ℝ := 1200
def expenditure_july : ℝ := 1500
def num_months : ℕ := 6

theorem average_expenditure_feb_to_jul :
  let total_jan_to_jun := average_expenditure_jan_to_jun * num_months
  let total_feb_to_jun := total_jan_to_jun - expenditure_january
  let total_feb_to_jul := total_feb_to_jun + expenditure_july
  total_feb_to_jul / num_months = 4250 := by
sorry

end average_expenditure_feb_to_jul_l3018_301864


namespace unique_winning_combination_l3018_301886

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 60 ∧ ∃ (m n : ℕ), n = 2^m * 3^n

def is_valid_combination (combo : Finset ℕ) : Prop :=
  combo.card = 5 ∧ 
  ∀ n ∈ combo, is_valid_number n ∧
  ∃ k : ℕ, (combo.prod id) = 12^k

theorem unique_winning_combination : 
  ∃! combo : Finset ℕ, is_valid_combination combo :=
sorry

end unique_winning_combination_l3018_301886


namespace teresa_spent_forty_l3018_301822

/-- The total amount spent by Teresa at the local shop -/
def total_spent (sandwich_price : ℚ) (sandwich_quantity : ℕ)
  (salami_price : ℚ) (olive_price_per_pound : ℚ) (olive_quantity : ℚ)
  (feta_price_per_pound : ℚ) (feta_quantity : ℚ) (bread_price : ℚ) : ℚ :=
  sandwich_price * sandwich_quantity +
  salami_price +
  3 * salami_price +
  olive_price_per_pound * olive_quantity +
  feta_price_per_pound * feta_quantity +
  bread_price

/-- Theorem: Teresa spends $40.00 at the local shop -/
theorem teresa_spent_forty : 
  total_spent 7.75 2 4 10 (1/4) 8 (1/2) 2 = 40 := by
  sorry

end teresa_spent_forty_l3018_301822


namespace solutions_count_2x_3y_763_l3018_301826

theorem solutions_count_2x_3y_763 : 
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + 3 * p.2 = 763 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 764) (Finset.range 764))).card = 127 := by
  sorry

end solutions_count_2x_3y_763_l3018_301826


namespace quadratic_property_l3018_301825

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_property (a b c : ℝ) :
  (∀ x, f a b c x ≥ 10) ∧  -- minimum value is 10
  (f a b c (-2) = 10) ∧    -- minimum occurs at x = -2
  (f a b c 0 = 6) →        -- passes through (0, 6)
  f a b c 5 = -39 :=       -- f(5) = -39
by sorry

end quadratic_property_l3018_301825


namespace geometric_sequence_150th_term_l3018_301856

/-- Given a geometric sequence with first term 5 and second term -10,
    the 150th term is equal to -5 * 2^149 -/
theorem geometric_sequence_150th_term :
  let a₁ : ℝ := 5
  let a₂ : ℝ := -10
  let r : ℝ := a₂ / a₁
  let a₁₅₀ : ℝ := a₁ * r^149
  a₁₅₀ = -5 * 2^149 := by
sorry

end geometric_sequence_150th_term_l3018_301856


namespace intersection_of_A_and_B_union_of_A_and_B_l3018_301809

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

-- Theorem for intersection
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 7} := by sorry

-- Theorem for union
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 10} := by sorry

end intersection_of_A_and_B_union_of_A_and_B_l3018_301809


namespace ratio_K_L_l3018_301827

theorem ratio_K_L : ∃ (K L : ℤ),
  (∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 →
    (K / (x + 3 : ℝ)) + (L / (x^2 - 3*x : ℝ)) = ((x^2 - x + 5) / (x^3 + x^2 - 9*x) : ℝ)) →
  (K : ℚ) / (L : ℚ) = 3 / 5 := by
sorry

end ratio_K_L_l3018_301827


namespace solve_equation_l3018_301818

theorem solve_equation (x : ℝ) : 3*x - 5*x + 4*x + 6 = 138 → x = 66 := by
  sorry

end solve_equation_l3018_301818


namespace inscribed_circle_radius_rhombus_l3018_301815

theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 30) :
  let a := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let r := (d1 * d2) / (4 * a)
  r = (105 * Real.sqrt 274) / 274 := by sorry

end inscribed_circle_radius_rhombus_l3018_301815


namespace tank_emptying_time_difference_l3018_301875

/-- Proves the time difference for emptying a tank with and without an inlet pipe. -/
theorem tank_emptying_time_difference 
  (tank_capacity : ℝ) 
  (outlet_rate : ℝ) 
  (inlet_rate : ℝ) 
  (h1 : tank_capacity = 21600) 
  (h2 : outlet_rate = 2160) 
  (h3 : inlet_rate = 960) : 
  (tank_capacity / outlet_rate) - (tank_capacity / (outlet_rate - inlet_rate)) = 8 := by
  sorry

#check tank_emptying_time_difference

end tank_emptying_time_difference_l3018_301875


namespace second_man_speed_l3018_301814

/-- Given two men walking in the same direction for 1 hour, where one walks at 10 kmph
    and they end up 2 km apart, the speed of the second man is 8 kmph. -/
theorem second_man_speed (speed_first : ℝ) (distance_apart : ℝ) (time : ℝ) (speed_second : ℝ) :
  speed_first = 10 →
  distance_apart = 2 →
  time = 1 →
  speed_first - speed_second = distance_apart / time →
  speed_second = 8 := by
sorry

end second_man_speed_l3018_301814


namespace line_equal_intercepts_l3018_301833

/-- A line passing through (1,2) with equal x and y intercepts has equation x+y-3=0 or 2x-y=0 -/
theorem line_equal_intercepts :
  ∀ (L : Set (ℝ × ℝ)), 
    ((1, 2) ∈ L) →
    (∃ a : ℝ, a ≠ 0 ∧ (a, 0) ∈ L ∧ (0, a) ∈ L) →
    (∀ x y : ℝ, (x, y) ∈ L ↔ (x + y = 3 ∨ 2*x - y = 0)) :=
by sorry

end line_equal_intercepts_l3018_301833


namespace dennis_pants_purchase_l3018_301823

def pants_price : ℝ := 110
def socks_price : ℝ := 60
def discount_rate : ℝ := 0.3
def num_socks : ℕ := 2
def total_spent : ℝ := 392

def discounted_pants_price : ℝ := pants_price * (1 - discount_rate)
def discounted_socks_price : ℝ := socks_price * (1 - discount_rate)

theorem dennis_pants_purchase :
  ∃ (num_pants : ℕ),
    num_pants * discounted_pants_price + num_socks * discounted_socks_price = total_spent ∧
    num_pants = 4 := by
  sorry

end dennis_pants_purchase_l3018_301823


namespace complex_fourth_power_l3018_301876

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end complex_fourth_power_l3018_301876


namespace position_number_difference_l3018_301803

structure Student where
  initial_i : ℤ
  initial_j : ℤ
  new_m : ℤ
  new_n : ℤ

def movement (s : Student) : ℤ × ℤ :=
  (s.initial_i - s.new_m, s.initial_j - s.new_n)

def position_number (s : Student) : ℤ :=
  let (a, b) := movement s
  a + b

def sum_position_numbers (students : List Student) : ℤ :=
  students.map position_number |>.sum

theorem position_number_difference (students : List Student) :
  ∃ (S_max S_min : ℤ),
    (∀ s, sum_position_numbers s ≤ S_max ∧ sum_position_numbers s ≥ S_min) ∧
    S_max - S_min = 12 :=
sorry

end position_number_difference_l3018_301803


namespace range_of_m_l3018_301883

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ 
  (¬ ∃ x : ℝ, x^2 + m * x + 1 < 0) → 
  0 ≤ m ∧ m ≤ 2 := by
  sorry

end range_of_m_l3018_301883


namespace trigonometric_simplification_l3018_301887

theorem trigonometric_simplification :
  (Real.sqrt (1 + 2 * Real.sin (610 * π / 180) * Real.cos (430 * π / 180))) /
  (Real.sin (250 * π / 180) + Real.cos (790 * π / 180)) = -1 := by
  sorry

end trigonometric_simplification_l3018_301887


namespace bianca_not_recycled_bags_l3018_301871

/-- The number of bags Bianca did not recycle -/
def bags_not_recycled (total_bags : ℕ) (points_per_bag : ℕ) (total_points : ℕ) : ℕ :=
  total_bags - (total_points / points_per_bag)

/-- Theorem stating that Bianca did not recycle 8 bags -/
theorem bianca_not_recycled_bags : bags_not_recycled 17 5 45 = 8 := by
  sorry

end bianca_not_recycled_bags_l3018_301871


namespace convention_handshakes_l3018_301858

/-- The number of handshakes at the Annual Mischief Convention --/
def annual_mischief_convention_handshakes (num_gremlins : ℕ) (num_imps : ℕ) : ℕ :=
  let gremlin_handshakes := num_gremlins.choose 2
  let imp_gremlin_handshakes := num_imps * num_gremlins
  gremlin_handshakes + imp_gremlin_handshakes

/-- Theorem stating the number of handshakes at the Annual Mischief Convention --/
theorem convention_handshakes :
  annual_mischief_convention_handshakes 25 20 = 800 := by
  sorry

#eval annual_mischief_convention_handshakes 25 20

end convention_handshakes_l3018_301858


namespace differential_of_y_l3018_301893

noncomputable section

open Real

-- Define the function y
def y (x : ℝ) : ℝ := x * (sin (log x) - cos (log x))

-- State the theorem
theorem differential_of_y (x : ℝ) (h : x > 0) :
  deriv y x = 2 * sin (log x) :=
by sorry

end

end differential_of_y_l3018_301893


namespace bellas_to_annes_height_ratio_l3018_301830

/-- Proves that given the conditions in the problem, the ratio of Bella's height to Anne's height is 3:1 -/
theorem bellas_to_annes_height_ratio : 
  ∀ (anne_height bella_height sister_height : ℝ),
  anne_height = 2 * sister_height →
  anne_height = 80 →
  bella_height - sister_height = 200 →
  bella_height / anne_height = 3 := by
sorry

end bellas_to_annes_height_ratio_l3018_301830


namespace nidas_chocolates_l3018_301808

theorem nidas_chocolates (x : ℕ) 
  (h1 : 3 * x + 5 + 25 = 5 * x) : 3 * x + 5 = 50 := by
  sorry

end nidas_chocolates_l3018_301808


namespace two_coin_toss_probabilities_l3018_301819

theorem two_coin_toss_probabilities (P₁ P₂ P₃ : ℝ) 
  (h1 : P₁ ≥ 0) (h2 : P₂ ≥ 0) (h3 : P₃ ≥ 0)
  (h4 : P₁ ≤ 1) (h5 : P₂ ≤ 1) (h6 : P₃ ≤ 1)
  (h7 : P₁ = (1/2)^2) (h8 : P₂ = (1/2)^2) (h9 : P₃ = 2 * (1/2)^2) : 
  (P₁ + P₂ = P₃) ∧ (P₁ + P₂ + P₃ = 1) ∧ (P₃ = 2*P₁ ∧ P₃ = 2*P₂) := by
  sorry

end two_coin_toss_probabilities_l3018_301819


namespace line_passes_through_fixed_point_l3018_301855

/-- The line (2k-1)x-(k+3)y-(k-11)=0 passes through the point (2, 3) for all values of k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end line_passes_through_fixed_point_l3018_301855


namespace nabla_problem_l3018_301817

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_problem : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end nabla_problem_l3018_301817


namespace solve_linear_equation_l3018_301860

theorem solve_linear_equation :
  ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 :=
by
  use -30
  constructor
  · -- Prove that x = -30 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check solve_linear_equation

end solve_linear_equation_l3018_301860


namespace three_integer_sum_l3018_301849

theorem three_integer_sum (a b c : ℕ) : 
  a > 1 → b > 1 → c > 1 →
  a * b * c = 343000 →
  Nat.gcd a b = 1 → Nat.gcd b c = 1 → Nat.gcd a c = 1 →
  a + b + c = 476 := by
sorry

end three_integer_sum_l3018_301849


namespace bike_ride_time_l3018_301862

/-- The time taken to ride a bike along semicircular paths on a highway -/
theorem bike_ride_time (highway_length : Real) (highway_width : Real) (speed : Real) : 
  highway_length = 2 → 
  highway_width = 60 / 5280 → 
  speed = 6 → 
  (π * highway_length / highway_width) / speed = π / 6 := by
  sorry

end bike_ride_time_l3018_301862


namespace trig_identity_l3018_301824

theorem trig_identity (α β : ℝ) :
  1 - Real.cos (β - α) + Real.cos α - Real.cos β =
  4 * Real.cos (α / 2) * Real.sin (β / 2) * Real.sin ((β - α) / 2) := by
  sorry

end trig_identity_l3018_301824


namespace chord_length_l3018_301859

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (t : ℝ) : 
  let x := 1 + 2*t
  let y := 2 + t
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 9}
  let line := {(x, y) : ℝ × ℝ | ∃ t, x = 1 + 2*t ∧ y = 2 + t}
  let intersection := circle ∩ line
  ∃ p q : ℝ × ℝ, p ∈ intersection ∧ q ∈ intersection ∧ 
    dist p q = 12/5 * Real.sqrt 5 :=
by
  sorry

end chord_length_l3018_301859


namespace g_behavior_at_infinity_l3018_301866

def g (x : ℝ) : ℝ := -3 * x^4 + 5

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M) := by
  sorry

end g_behavior_at_infinity_l3018_301866


namespace power_function_through_2_4_l3018_301888

/-- A power function passing through the point (2, 4) has exponent 2 -/
theorem power_function_through_2_4 :
  ∀ a : ℝ, (2 : ℝ) ^ a = 4 → a = 2 := by
  sorry

end power_function_through_2_4_l3018_301888


namespace steps_climbed_fraction_l3018_301812

/-- Proves that climbing 25 steps in a 6-floor building with 12 steps between floors
    is equivalent to climbing 5/12 of the total steps. -/
theorem steps_climbed_fraction (total_floors : Nat) (steps_per_floor : Nat) (steps_climbed : Nat) :
  total_floors = 6 →
  steps_per_floor = 12 →
  steps_climbed = 25 →
  (steps_climbed : Rat) / ((total_floors - 1) * steps_per_floor) = 5 / 12 := by
  sorry

end steps_climbed_fraction_l3018_301812


namespace ordering_of_expressions_l3018_301851

theorem ordering_of_expressions : 3^(1/5) > 0.2^3 ∧ 0.2^3 > Real.log 0.1 / Real.log 3 := by
  sorry

end ordering_of_expressions_l3018_301851


namespace fourth_root_equation_solution_l3018_301861

theorem fourth_root_equation_solution :
  ∃ x : ℝ, (x^(1/4) * (x^5)^(1/8) = 4) ∧ (x = 4^(8/7)) := by
  sorry

end fourth_root_equation_solution_l3018_301861


namespace triangle_abc_properties_l3018_301807

theorem triangle_abc_properties (b c : ℝ) (A B : ℝ) :
  A = π / 3 →
  3 * b = 2 * c →
  (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 →
  b = 2 ∧ Real.sin B = Real.sqrt 21 / 7 := by
  sorry

end triangle_abc_properties_l3018_301807


namespace triangle_side_length_l3018_301810

/-- Theorem: In a triangle ABC where side b = 2, angle A = 45°, and angle C = 75°, 
    the length of side a is equal to (2/3)√6. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  b = 2 → 
  A = 45 * π / 180 → 
  C = 75 * π / 180 → 
  a = (2 / 3) * Real.sqrt 6 := by
sorry


end triangle_side_length_l3018_301810


namespace cookies_sold_l3018_301835

theorem cookies_sold (total : ℕ) (ratio_brownies : ℕ) (ratio_cookies : ℕ) (cookies : ℕ) : 
  total = 104 →
  ratio_brownies = 7 →
  ratio_cookies = 6 →
  ratio_brownies * cookies = ratio_cookies * (total - cookies) →
  cookies = 48 := by
sorry

end cookies_sold_l3018_301835


namespace ellipse_equivalence_l3018_301829

/-- Given an ellipse with equation 9x^2 + 4y^2 = 36, prove that the ellipse with equation
    x^2/20 + y^2/25 = 1 has the same foci and a minor axis length of 4√5 -/
theorem ellipse_equivalence (x y : ℝ) : 
  (∃ (a b c : ℝ), 9 * x^2 + 4 * y^2 = 36 ∧ 
   c^2 = a^2 - b^2 ∧
   x^2 / 20 + y^2 / 25 = 1 ∧
   b = 2 * (5 : ℝ).sqrt) := by
  sorry

end ellipse_equivalence_l3018_301829


namespace sum_of_roots_l3018_301821

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.log 3 / Real.log (3 * x) + Real.log (3 * x) / Real.log 27 = -4/3

-- Define the roots
def roots : Set ℝ := {x | equation x}

-- Theorem statement
theorem sum_of_roots :
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ a + b = 10/81 :=
sorry

end sum_of_roots_l3018_301821


namespace system_solution_l3018_301891

theorem system_solution (x y : ℝ) : 
  (x = 5 ∧ y = -1) → (2 * x + 3 * y = 7 ∧ x = -2 * y + 3) :=
by sorry

end system_solution_l3018_301891


namespace right_triangle_acute_angles_l3018_301834

theorem right_triangle_acute_angles (α β : ℝ) : 
  α = 60 → β = 90 → α + β + (180 - α - β) = 180 → 180 - α - β = 30 := by
  sorry

end right_triangle_acute_angles_l3018_301834


namespace robot_trajectory_constraint_l3018_301853

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The trajectory of the robot -/
def robotTrajectory : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- The line x = -1 -/
def verticalLine : Line :=
  { slope := 0, yIntercept := -1 }

/-- The point F(1, 0) -/
def pointF : Point :=
  { x := 1, y := 0 }

/-- The point P(-1, 0) -/
def pointP : Point :=
  { x := -1, y := 0 }

/-- The line passing through P(-1, 0) with slope k -/
def lineThroughP (k : ℝ) : Line :=
  { slope := k, yIntercept := k }

/-- The robot's trajectory does not intersect the line through P -/
def noIntersection (k : ℝ) : Prop :=
  ∀ p : Point, p ∈ robotTrajectory → p ∉ {p : Point | p.y = (lineThroughP k).slope * (p.x + 1)}

theorem robot_trajectory_constraint (k : ℝ) :
  noIntersection k ↔ k > 1 ∨ k < -1 :=
sorry

end robot_trajectory_constraint_l3018_301853


namespace function_equation_solution_l3018_301872

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x * f y) = f x * y) : 
  ∀ x : ℝ, f x = x := by
  sorry

end function_equation_solution_l3018_301872


namespace find_k_l3018_301869

theorem find_k (a b c k : ℚ) : 
  (∀ x : ℚ, (a*x^2 + b*x + c + b*x^2 + a*x - 7 + k*x^2 + c*x + 3) / (x^2 - 2*x - 5) = 1) → 
  k = 2 := by
sorry

end find_k_l3018_301869


namespace tan_315_degrees_l3018_301801

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end tan_315_degrees_l3018_301801


namespace octagon_perimeter_in_cm_l3018_301882

/-- Regular octagon with side length in meters -/
structure RegularOctagon where
  side_length : ℝ

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- Sum of all side lengths of a regular octagon in centimeters -/
def sum_side_lengths (octagon : RegularOctagon) : ℝ :=
  8 * octagon.side_length * meters_to_cm

theorem octagon_perimeter_in_cm (octagon : RegularOctagon) 
    (h : octagon.side_length = 2.3) : 
    sum_side_lengths octagon = 1840 := by
  sorry

end octagon_perimeter_in_cm_l3018_301882


namespace completing_square_transformation_l3018_301854

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 8*x + 2 = 0) ↔ ((x - 4)^2 = 14) :=
by sorry

end completing_square_transformation_l3018_301854


namespace turnip_solution_l3018_301873

/-- The number of turnips grown by Melanie, Benny, and Caroline, and the difference between
    the combined turnips of Melanie and Benny versus Caroline's turnips. -/
def turnip_problem (melanie_turnips benny_turnips caroline_turnips : ℕ) : Prop :=
  let combined_turnips := melanie_turnips + benny_turnips
  combined_turnips - caroline_turnips = 80

/-- The theorem stating the solution to the turnip problem -/
theorem turnip_solution : turnip_problem 139 113 172 := by
  sorry

end turnip_solution_l3018_301873


namespace point_movement_and_quadrant_l3018_301847

/-- Given a point P with coordinates (a - 7, 3 - 2a) in the Cartesian coordinate system,
    which is moved 4 units up and 5 units right to obtain point Q (a - 2, 7 - 2a),
    prove that for Q to be in the first quadrant, 2 < a < 3.5,
    and when a is an integer satisfying this condition, P = (-4, -3) and Q = (1, 1). -/
theorem point_movement_and_quadrant (a : ℝ) :
  let P : ℝ × ℝ := (a - 7, 3 - 2*a)
  let Q : ℝ × ℝ := (a - 2, 7 - 2*a)
  (∀ x y, Q = (x, y) → x > 0 ∧ y > 0) ↔ (2 < a ∧ a < 3.5) ∧
  (∃ n : ℤ, ↑n = a ∧ 2 < a ∧ a < 3.5) →
    P = (-4, -3) ∧ Q = (1, 1) :=
by sorry

end point_movement_and_quadrant_l3018_301847


namespace museum_trip_ratio_l3018_301842

theorem museum_trip_ratio : 
  let total_people : ℕ := 123
  let num_boys : ℕ := 50
  let num_staff : ℕ := 3  -- driver, assistant, and teacher
  let num_girls : ℕ := total_people - num_boys - num_staff
  (num_girls > num_boys) →
  (num_girls - num_boys : ℚ) / num_boys = 21 / 50 :=
by
  sorry

end museum_trip_ratio_l3018_301842


namespace percent_democrat_voters_l3018_301857

theorem percent_democrat_voters (D R : ℝ) : 
  D + R = 100 →
  0.85 * D + 0.20 * R = 59 →
  D = 60 :=
by sorry

end percent_democrat_voters_l3018_301857


namespace sequences_properties_l3018_301850

def sequence_a (n : ℕ) : ℤ := (-2) ^ n
def sequence_b (n : ℕ) : ℤ := (-2) ^ (n - 1)
def sequence_c (n : ℕ) : ℕ := 3 * 2 ^ (n - 1)

theorem sequences_properties :
  (sequence_a 6 = 64) ∧
  (sequence_b 7 = 64) ∧
  (sequence_c 7 = 192) ∧
  (sequence_c 11 = 3072) := by
sorry

end sequences_properties_l3018_301850


namespace win_sector_area_l3018_301840

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/7) :
  let total_area := π * r^2
  let win_area := p * total_area
  win_area = 192 * π / 7 := by
sorry

end win_sector_area_l3018_301840


namespace expression_value_l3018_301867

theorem expression_value (a b c : ℝ) (ha : a = 20) (hb : b = 40) (hc : c = 10) :
  (a - (b - c)) - ((a - b) - c) = 20 := by
  sorry

end expression_value_l3018_301867


namespace prime_fraction_equation_l3018_301843

theorem prime_fraction_equation (p q : ℕ) (hp : Prime p) (hq : Prime q) (n : ℕ+) 
  (h : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / (p * q) = (1 : ℚ) / n) :
  (p = 2 ∧ q = 3 ∧ n = 1) ∨ (p = 3 ∧ q = 2 ∧ n = 1) := by
sorry

end prime_fraction_equation_l3018_301843


namespace max_distance_point_to_circle_l3018_301895

/-- The maximum distance between a point and a circle -/
theorem max_distance_point_to_circle :
  let P : ℝ × ℝ := (-1, -1)
  let center : ℝ × ℝ := (3, 0)
  let radius : ℝ := 2
  let circle := {(x, y) : ℝ × ℝ | (x - 3)^2 + y^2 = 4}
  (∀ Q ∈ circle, Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt 17 + 2) ∧
  (∃ Q ∈ circle, Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 17 + 2) :=
by sorry

end max_distance_point_to_circle_l3018_301895


namespace fraction_simplification_l3018_301877

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 * x) / (x^2 - 1) - 1 / (x - 1) = 1 / (x + 1) := by
  sorry

end fraction_simplification_l3018_301877


namespace line_perpendicular_to_plane_l3018_301838

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m α) : 
  perpendicular n α :=
sorry

end line_perpendicular_to_plane_l3018_301838


namespace problem_1_problem_2_l3018_301846

-- Problem 1
theorem problem_1 (a b : ℚ) (h1 : a = 2) (h2 : b = 1/3) :
  3 * (a^2 - a*b + 7) - 2 * (3*a*b - a^2 + 1) + 3 = 36 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h : (x + 2)^2 + |y - 1/2| = 0) :
  5*x^2 - (2*x*y - 3*(1/3*x*y + 2) + 4*x^2) = 11 := by sorry

end problem_1_problem_2_l3018_301846


namespace tan_30_plus_4sin_30_l3018_301852

/-- The tangent of 30 degrees plus 4 times the sine of 30 degrees equals (√3)/3 + 2 -/
theorem tan_30_plus_4sin_30 : Real.tan (30 * π / 180) + 4 * Real.sin (30 * π / 180) = (Real.sqrt 3) / 3 + 2 := by
  sorry

end tan_30_plus_4sin_30_l3018_301852


namespace picture_frame_dimensions_l3018_301839

theorem picture_frame_dimensions (a b : ℕ+) : 
  (a : ℤ) * b = ((a + 2) * (b + 2) : ℤ) - a * b → 
  ((a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4)) :=
by sorry

end picture_frame_dimensions_l3018_301839


namespace golden_section_addition_correct_l3018_301848

/-- The 0.618 method for finding the optimal addition amount --/
def golden_section_addition (a b x : ℝ) : ℝ :=
  a + b - x

/-- Theorem stating the correct formula for the addition point in the 0.618 method --/
theorem golden_section_addition_correct (a b x : ℝ) 
  (h_range : a ≤ x ∧ x ≤ b) 
  (h_good_point : x = a + 0.618 * (b - a)) : 
  golden_section_addition a b x = a + b - x :=
by sorry

end golden_section_addition_correct_l3018_301848


namespace snow_volume_calculation_l3018_301820

/-- Calculates the total volume of snow on a rectangular driveway with two distinct layers. -/
theorem snow_volume_calculation (length width depth1 depth2 : ℝ) 
  (h1 : length = 30) 
  (h2 : width = 4) 
  (h3 : depth1 = 0.5) 
  (h4 : depth2 = 0.3) : 
  length * width * depth1 + length * width * depth2 = 96 := by
  sorry

#check snow_volume_calculation

end snow_volume_calculation_l3018_301820


namespace square_area_from_perimeter_l3018_301897

/-- Given a square with perimeter 40 meters, its area is 100 square meters. -/
theorem square_area_from_perimeter : 
  ∀ s : Real, 
  (4 * s = 40) → -- perimeter is 40 meters
  (s * s = 100)  -- area is 100 square meters
:= by
  sorry

end square_area_from_perimeter_l3018_301897


namespace devonshire_cows_cost_l3018_301800

/-- The number of hearts in a standard deck of 52 playing cards -/
def hearts_in_deck : ℕ := 13

/-- The number of cows in Devonshire -/
def cows_in_devonshire : ℕ := 2 * hearts_in_deck

/-- The price of each cow in dollars -/
def price_per_cow : ℕ := 200

/-- The total cost of all cows in Devonshire when sold -/
def total_cost : ℕ := cows_in_devonshire * price_per_cow

theorem devonshire_cows_cost : total_cost = 5200 := by
  sorry

end devonshire_cows_cost_l3018_301800


namespace bottle_cost_difference_l3018_301811

/-- Represents a bottle of capsules -/
structure Bottle where
  capsules : ℕ
  cost : ℚ

/-- Calculates the cost per capsule for a given bottle -/
def costPerCapsule (b : Bottle) : ℚ := b.cost / b.capsules

/-- The difference in cost per capsule between two bottles -/
def costDifference (b1 b2 : Bottle) : ℚ := costPerCapsule b2 - costPerCapsule b1

theorem bottle_cost_difference :
  let bottleR : Bottle := { capsules := 250, cost := 25/4 }
  let bottleT : Bottle := { capsules := 100, cost := 3 }
  costDifference bottleR bottleT = 1/200
  := by sorry

end bottle_cost_difference_l3018_301811


namespace sum_of_coefficients_l3018_301816

theorem sum_of_coefficients (a : ℝ) : 
  ((1 + a)^5 = -1) → (a = -2) := by
  sorry

end sum_of_coefficients_l3018_301816


namespace min_value_theorem_equality_condition_l3018_301868

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 4 :=
by sorry

theorem equality_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) = 4 ↔ a^2 = 2 ∧ b = Real.sqrt 2 / 2 :=
by sorry

end min_value_theorem_equality_condition_l3018_301868


namespace bagel_store_expenditure_l3018_301804

theorem bagel_store_expenditure (B D : ℝ) : 
  D = B / 2 →
  B = D + 15 →
  B + D = 45 := by sorry

end bagel_store_expenditure_l3018_301804


namespace max_difference_with_broken_calculator_l3018_301870

def is_valid_digit (d : ℕ) (valid_digits : List ℕ) : Prop :=
  d ∈ valid_digits

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem max_difference_with_broken_calculator :
  ∀ (a b c d e f : ℕ),
    is_valid_digit a [3, 5, 9] →
    is_valid_digit b [2, 3, 7] →
    is_valid_digit c [3, 4, 8, 9] →
    is_valid_digit d [2, 3, 7] →
    is_valid_digit e [3, 5, 9] →
    is_valid_digit f [1, 4, 7] →
    is_three_digit_number (100 * a + 10 * b + c) →
    is_three_digit_number (100 * d + 10 * e + f) →
    is_three_digit_number ((100 * a + 10 * b + c) - (100 * d + 10 * e + f)) →
    (100 * a + 10 * b + c) - (100 * d + 10 * e + f) ≤ 529 ∧
    (a = 9 ∧ b = 2 ∧ c = 3 ∧ d = 3 ∧ e = 9 ∧ f = 4 →
      ∀ (x y z u v w : ℕ),
        is_valid_digit x [3, 5, 9] →
        is_valid_digit y [2, 3, 7] →
        is_valid_digit z [3, 4, 8, 9] →
        is_valid_digit u [2, 3, 7] →
        is_valid_digit v [3, 5, 9] →
        is_valid_digit w [1, 4, 7] →
        is_three_digit_number (100 * x + 10 * y + z) →
        is_three_digit_number (100 * u + 10 * v + w) →
        is_three_digit_number ((100 * x + 10 * y + z) - (100 * u + 10 * v + w)) →
        (100 * x + 10 * y + z) - (100 * u + 10 * v + w) ≤ (100 * a + 10 * b + c) - (100 * d + 10 * e + f)) :=
by sorry

end max_difference_with_broken_calculator_l3018_301870


namespace equation_solution_l3018_301898

theorem equation_solution : 
  ∃ (x : ℤ), (1 + 1 / x : ℚ) ^ (x + 1) = (1 + 1 / 2003 : ℚ) ^ 2003 :=
by
  use -2004
  sorry

end equation_solution_l3018_301898


namespace f_iter_formula_l3018_301874

def f (x : ℝ) := 3 * x + 2

def f_iter : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ f_iter n

theorem f_iter_formula (n : ℕ) (x : ℝ) : 
  f_iter n x = 3^n * x + 3^n - 1 := by sorry

end f_iter_formula_l3018_301874


namespace original_number_proof_l3018_301879

theorem original_number_proof (x : ℝ) : 3 * (2 * x + 5) = 123 ↔ x = 18 := by
  sorry

end original_number_proof_l3018_301879


namespace soda_discount_theorem_l3018_301828

/-- Calculates the discounted price for purchasing soda cans -/
def discounted_price (regular_price : ℚ) (num_cans : ℕ) : ℚ :=
  let cases := (num_cans + 23) / 24  -- Round up to nearest case
  let total_regular_price := regular_price * num_cans
  let discount_rate := 
    if cases ≤ 2 then 25/100
    else if cases ≤ 4 then 30/100
    else 35/100
  total_regular_price * (1 - discount_rate)

/-- Theorem stating the discounted price for 70 cans of soda -/
theorem soda_discount_theorem :
  discounted_price (55/100) 70 = 2772/100 := by
  sorry

end soda_discount_theorem_l3018_301828


namespace garden_perimeter_l3018_301885

/-- Given a square garden with area q and perimeter p, if q = 2p + 20, then p = 40 -/
theorem garden_perimeter (q p : ℝ) (h1 : q > 0) (h2 : p > 0) (h3 : q = p^2 / 16) (h4 : q = 2*p + 20) : p = 40 := by
  sorry

end garden_perimeter_l3018_301885


namespace shifted_sine_equals_cosine_l3018_301899

open Real

theorem shifted_sine_equals_cosine (ω φ : ℝ) (h_ω : ω < 0) :
  (∀ x, sin (ω * (x - π / 12) + φ) = cos (2 * x)) →
  ∃ k : ℤ, φ = π / 3 + 2 * π * ↑k := by sorry

end shifted_sine_equals_cosine_l3018_301899


namespace intersection_of_A_and_B_l3018_301894

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 4} := by sorry

end intersection_of_A_and_B_l3018_301894


namespace number_of_pigs_l3018_301806

theorem number_of_pigs (total_cost : ℕ) (num_hens : ℕ) (avg_price_hen : ℕ) (avg_price_pig : ℕ) :
  total_cost = 1200 →
  num_hens = 10 →
  avg_price_hen = 30 →
  avg_price_pig = 300 →
  ∃ (num_pigs : ℕ), num_pigs = 3 ∧ total_cost = num_pigs * avg_price_pig + num_hens * avg_price_hen :=
by
  sorry

end number_of_pigs_l3018_301806


namespace meaningful_fraction_range_l3018_301844

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end meaningful_fraction_range_l3018_301844


namespace at_least_eight_composite_l3018_301813

theorem at_least_eight_composite (n : ℕ) (h : n > 1000) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ 
  (∀ x ∈ s, x ≥ n ∧ x < n + 12) ∧
  (∃ (t : Finset ℕ), t ⊆ s ∧ t.card ≥ 8 ∧ ∀ y ∈ t, ¬ Nat.Prime y) := by
  sorry

end at_least_eight_composite_l3018_301813


namespace certain_number_equation_l3018_301889

theorem certain_number_equation (x : ℝ) : (40 * 30 + (x + 8) * 3) / 5 = 1212 ↔ x = 1612 := by
  sorry

end certain_number_equation_l3018_301889


namespace eight_bead_necklace_arrangements_l3018_301831

/-- The number of distinct arrangements of n beads on a necklace,
    considering rotational and reflectional symmetry -/
def necklace_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements
    of 8 beads on a necklace is 2520 -/
theorem eight_bead_necklace_arrangements :
  necklace_arrangements 8 = 2520 := by
  sorry

end eight_bead_necklace_arrangements_l3018_301831


namespace student_count_l3018_301802

theorem student_count (n : ℕ) (right_rank left_rank : ℕ) 
  (h1 : right_rank = 6) 
  (h2 : left_rank = 5) 
  (h3 : n = right_rank + left_rank - 1) : n = 10 :=
by sorry

end student_count_l3018_301802


namespace mixed_grains_calculation_l3018_301890

/-- Calculates the amount of mixed grains in a batch of rice -/
theorem mixed_grains_calculation (total_rice : ℝ) (sample_size : ℝ) (mixed_in_sample : ℝ) :
  total_rice * (mixed_in_sample / sample_size) = 150 :=
by
  -- Assuming total_rice = 1500, sample_size = 200, and mixed_in_sample = 20
  have h1 : total_rice = 1500 := by sorry
  have h2 : sample_size = 200 := by sorry
  have h3 : mixed_in_sample = 20 := by sorry
  
  -- The proof goes here
  sorry

end mixed_grains_calculation_l3018_301890


namespace sequence_property_l3018_301845

def sequence_a (n : ℕ) : ℕ := sorry

theorem sequence_property : 
  ∃ (b c d : ℤ), 
    (∀ n : ℕ, n > 0 → sequence_a n = b * Int.floor (Real.sqrt (n + c)) + d) ∧ 
    sequence_a 1 = 1 ∧ 
    b + c + d = 1 :=
sorry

end sequence_property_l3018_301845


namespace chorus_group_size_l3018_301896

theorem chorus_group_size :
  let S := {n : ℕ | 100 < n ∧ n < 200 ∧
                    n % 3 = 1 ∧
                    n % 4 = 2 ∧
                    n % 6 = 4 ∧
                    n % 8 = 6}
  S = {118, 142, 166, 190} := by
  sorry

end chorus_group_size_l3018_301896
