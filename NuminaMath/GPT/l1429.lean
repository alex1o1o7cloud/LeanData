import Mathlib

namespace g_18_equals_324_l1429_142928

def is_strictly_increasing (g : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → g (n + 1) > g n

def multiplicative (g : ℕ → ℕ) :=
  ∀ m n : ℕ, m > 0 → n > 0 → g (m * n) = g m * g n

def m_n_condition (g : ℕ → ℕ) :=
  ∀ m n : ℕ, m ≠ n → m > 0 → n > 0 → m ^ n = n ^ m → (g m = n ∨ g n = m)

noncomputable def g : ℕ → ℕ := sorry

theorem g_18_equals_324 :
  is_strictly_increasing g →
  multiplicative g →
  m_n_condition g →
  g 18 = 324 :=
sorry

end g_18_equals_324_l1429_142928


namespace expand_product_l1429_142933

theorem expand_product : ∀ (x : ℝ), (x + 2) * (x^2 - 4 * x + 1) = x^3 - 2 * x^2 - 7 * x + 2 :=
by 
  intro x
  sorry

end expand_product_l1429_142933


namespace unique_two_digit_integer_s_l1429_142919

-- We define s to satisfy the two given conditions.
theorem unique_two_digit_integer_s (s : ℕ) (h1 : 13 * s % 100 = 52) (h2 : 1 ≤ s) (h3 : s ≤ 99) : s = 4 :=
sorry

end unique_two_digit_integer_s_l1429_142919


namespace future_ages_equation_l1429_142985

-- Defining the ages of Joe and James with given conditions
def joe_current_age : ℕ := 22
def james_current_age : ℕ := 12

-- Defining the condition that Joe is 10 years older than James
lemma joe_older_than_james : joe_current_age = james_current_age + 10 := by
  unfold joe_current_age james_current_age
  simp

-- Defining the future age condition equation and the target years y.
theorem future_ages_equation (y : ℕ) :
  2 * (joe_current_age + y) = 3 * (james_current_age + y) → y = 8 := by
  unfold joe_current_age james_current_age
  intro h
  linarith

end future_ages_equation_l1429_142985


namespace second_supply_cost_is_24_l1429_142920

-- Definitions based on the given problem conditions
def cost_first_supply : ℕ := 13
def last_year_remaining : ℕ := 6
def this_year_budget : ℕ := 50
def remaining_budget : ℕ := 19

-- Sum of last year's remaining budget and this year's budget
def total_budget : ℕ := last_year_remaining + this_year_budget

-- Total amount spent on school supplies
def total_spent : ℕ := total_budget - remaining_budget

-- Cost of second school supply
def cost_second_supply : ℕ := total_spent - cost_first_supply

-- The theorem to prove
theorem second_supply_cost_is_24 : cost_second_supply = 24 := by
  sorry

end second_supply_cost_is_24_l1429_142920


namespace problem_l1429_142947

def f (x : ℝ) : ℝ := x^3 + 2 * x

theorem problem : f 5 + f (-5) = 0 := by
  sorry

end problem_l1429_142947


namespace distance_focus_directrix_l1429_142950

theorem distance_focus_directrix (p : ℝ) (x_1 : ℝ) (h1 : 0 < p) (h2 : x_1^2 = 2 * p)
  (h3 : 1 + p / 2 = 3) : p = 4 :=
by
  sorry

end distance_focus_directrix_l1429_142950


namespace tom_purchases_l1429_142914

def total_cost_before_discount (price_per_box : ℝ) (num_boxes : ℕ) : ℝ :=
  price_per_box * num_boxes

def discount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

def total_cost_after_discount (total_cost : ℝ) (discount_amount : ℝ) : ℝ :=
  total_cost - discount_amount

def remaining_boxes (total_boxes : ℕ) (given_boxes : ℕ) : ℕ :=
  total_boxes - given_boxes

def total_pieces (num_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  num_boxes * pieces_per_box

theorem tom_purchases
  (price_per_box : ℝ) (num_boxes : ℕ) (discount_rate : ℝ) (given_boxes : ℕ) (pieces_per_box : ℕ) :
  (price_per_box = 4) →
  (num_boxes = 12) →
  (discount_rate = 0.15) →
  (given_boxes = 7) →
  (pieces_per_box = 6) →
  total_cost_after_discount (total_cost_before_discount price_per_box num_boxes) 
                             (discount (total_cost_before_discount price_per_box num_boxes) discount_rate)
  = 40.80 ∧
  total_pieces (remaining_boxes num_boxes given_boxes) pieces_per_box
  = 30 :=
by
  intros
  sorry

end tom_purchases_l1429_142914


namespace find_value_of_4_minus_2a_l1429_142935

theorem find_value_of_4_minus_2a (a b : ℚ) (h1 : 4 + 2 * a = 5 - b) (h2 : 5 + b = 9 + 3 * a) : 4 - 2 * a = 26 / 5 := 
by
  sorry

end find_value_of_4_minus_2a_l1429_142935


namespace factor_complete_polynomial_l1429_142948

theorem factor_complete_polynomial :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x^2 - 20 * x + 77) :=
sorry

end factor_complete_polynomial_l1429_142948


namespace net_pay_rate_per_hour_l1429_142981

-- Defining the given conditions
def travel_hours : ℕ := 3
def speed_mph : ℕ := 50
def fuel_efficiency : ℕ := 25 -- miles per gallon
def pay_rate_per_mile : ℚ := 0.60 -- dollars per mile
def gas_cost_per_gallon : ℚ := 2.50 -- dollars per gallon

-- Define the statement we want to prove
theorem net_pay_rate_per_hour : 
  (travel_hours * speed_mph * pay_rate_per_mile - 
  (travel_hours * speed_mph / fuel_efficiency) * gas_cost_per_gallon) / 
  travel_hours = 25 :=
by
  repeat {sorry}

end net_pay_rate_per_hour_l1429_142981


namespace minimize_product_l1429_142901

theorem minimize_product
    (a b c : ℕ) 
    (h_positive: a > 0 ∧ b > 0 ∧ c > 0)
    (h_eq: 10 * a^2 - 3 * a * b + 7 * c^2 = 0) : 
    (gcd a b) * (gcd b c) * (gcd c a) = 3 :=
sorry

end minimize_product_l1429_142901


namespace average_marks_second_class_l1429_142956

variable (average_marks_first_class : ℝ) (students_first_class : ℕ)
variable (students_second_class : ℕ) (combined_average_marks : ℝ)

theorem average_marks_second_class (H1 : average_marks_first_class = 60)
  (H2 : students_first_class = 55) (H3 : students_second_class = 48)
  (H4 : combined_average_marks = 59.067961165048544) :
  48 * 57.92 = 103 * 59.067961165048544 - 3300 := by
  sorry

end average_marks_second_class_l1429_142956


namespace joan_games_last_year_l1429_142977

theorem joan_games_last_year (games_this_year : ℕ) (total_games : ℕ) (games_last_year : ℕ) 
  (h1 : games_this_year = 4) 
  (h2 : total_games = 9) 
  (h3 : total_games = games_this_year + games_last_year) : 
  games_last_year = 5 := 
by
  sorry

end joan_games_last_year_l1429_142977


namespace triangle_side_eq_nine_l1429_142992

theorem triangle_side_eq_nine (a b c : ℕ) 
  (h_tri_ineq : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_sqrt_eq : (Nat.sqrt (a - 9)) + (b - 2)^2 = 0)
  (h_c_odd : c % 2 = 1) :
  c = 9 :=
sorry

end triangle_side_eq_nine_l1429_142992


namespace simplify_fraction_l1429_142988

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l1429_142988


namespace slices_of_pizza_left_l1429_142993

theorem slices_of_pizza_left (initial_slices: ℕ) 
  (breakfast_slices: ℕ) (lunch_slices: ℕ) (snack_slices: ℕ) (dinner_slices: ℕ) :
  initial_slices = 15 →
  breakfast_slices = 4 →
  lunch_slices = 2 →
  snack_slices = 2 →
  dinner_slices = 5 →
  (initial_slices - breakfast_slices - lunch_slices - snack_slices - dinner_slices) = 2 :=
by
  intros
  repeat { sorry }

end slices_of_pizza_left_l1429_142993


namespace students_taking_art_l1429_142980

theorem students_taking_art :
  ∀ (total_students music_students both_students neither_students : ℕ),
  total_students = 500 →
  music_students = 30 →
  both_students = 10 →
  neither_students = 460 →
  music_students + both_students + neither_students = total_students →
  ((total_students - neither_students) - (music_students - both_students) + both_students = 20) :=
by
  intros total_students music_students both_students neither_students 
  intro h_total h_music h_both h_neither h_sum 
  sorry

end students_taking_art_l1429_142980


namespace am_gm_inequality_even_sum_l1429_142958

theorem am_gm_inequality_even_sum (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h_even : (a + b) % 2 = 0) :
  (a + b : ℚ) / 2 ≥ Real.sqrt (a * b) :=
sorry

end am_gm_inequality_even_sum_l1429_142958


namespace y_intercept_tangent_line_l1429_142921

/-- Three circles have radii 3, 2, and 1 respectively. The first circle has center at (3,0), 
the second at (7,0), and the third at (11,0). A line is tangent to all three circles 
at points in the first quadrant. Prove the y-intercept of this line is 36.
-/
theorem y_intercept_tangent_line
  (r1 r2 r3 : ℝ) (h1 : r1 = 3) (h2 : r2 = 2) (h3 : r3 = 1)
  (c1 c2 c3 : ℝ × ℝ) (hc1 : c1 = (3, 0)) (hc2 : c2 = (7, 0)) (hc3 : c3 = (11, 0)) :
  ∃ y_intercept : ℝ, y_intercept = 36 :=
sorry

end y_intercept_tangent_line_l1429_142921


namespace probability_is_3888_over_7533_l1429_142987

noncomputable def probability_odd_sum_given_even_product : ℚ := 
  let total_outcomes := 6^5
  let all_odd_outcomes := 3^5
  let at_least_one_even_outcomes := total_outcomes - all_odd_outcomes
  let favorable_outcomes := 5 * 3^4 + 10 * 3^4 + 3^5
  favorable_outcomes / at_least_one_even_outcomes

theorem probability_is_3888_over_7533 :
  probability_odd_sum_given_even_product = 3888 / 7533 := 
sorry

end probability_is_3888_over_7533_l1429_142987


namespace length_of_interval_l1429_142995

theorem length_of_interval (a b : ℝ) (h : 10 = (b - a) / 2) : b - a = 20 :=
by 
  sorry

end length_of_interval_l1429_142995


namespace typhoon_probabilities_l1429_142984

-- Defining the conditions
def probAtLeastOneHit : ℝ := 0.36

-- Defining the events and probabilities
def probOfHit (p : ℝ) := p
def probBothHit (p : ℝ) := p^2

def probAtLeastOne (p : ℝ) : ℝ := p^2 + 2 * p * (1 - p)

-- Defining the variable X as the number of cities hit by the typhoon
def P_X_0 (p : ℝ) : ℝ := (1 - p)^2
def P_X_1 (p : ℝ) : ℝ := 2 * p * (1 - p)
def E_X (p : ℝ) : ℝ := 2 * p

-- Main theorem
theorem typhoon_probabilities :
  ∀ (p : ℝ),
    probAtLeastOne p = probAtLeastOneHit → 
    p = 0.2 ∧ P_X_0 p = 0.64 ∧ P_X_1 p = 0.32 ∧ E_X p = 0.4 :=
by
  intros p h
  sorry

end typhoon_probabilities_l1429_142984


namespace remainder_when_dividing_l1429_142969

theorem remainder_when_dividing (a : ℕ) (h1 : a = 432 * 44) : a % 38 = 8 :=
by
  -- Proof goes here
  sorry

end remainder_when_dividing_l1429_142969


namespace star_is_addition_l1429_142968

theorem star_is_addition (star : ℝ → ℝ → ℝ) 
  (H : ∀ a b c : ℝ, star (star a b) c = a + b + c) : 
  ∀ a b : ℝ, star a b = a + b :=
by
  sorry

end star_is_addition_l1429_142968


namespace minimal_erasure_l1429_142917

noncomputable def min_factors_to_erase : ℕ :=
  2016

theorem minimal_erasure:
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = g x) → 
    (∃ f' g' : ℝ → ℝ, (∀ x, f x ≠ g x) ∧ 
      ((∃ s : Finset ℕ, s.card = min_factors_to_erase ∧ (∀ i ∈ s, f' x = (x - i) * f x)) ∧ 
      (∃ t : Finset ℕ, t.card = min_factors_to_erase ∧ (∀ i ∈ t, g' x = (x - i) * g x)))) :=
by
  sorry

end minimal_erasure_l1429_142917


namespace gain_percent_correct_l1429_142989

variable (CP SP Gain : ℝ)
variable (H₁ : CP = 900)
variable (H₂ : SP = 1125)
variable (H₃ : Gain = SP - CP)

theorem gain_percent_correct : (Gain / CP) * 100 = 25 :=
by
  sorry

end gain_percent_correct_l1429_142989


namespace fraction_B_A_C_l1429_142996

theorem fraction_B_A_C (A B C : ℕ) (x : ℚ) 
  (h1 : A = (1 / 3) * (B + C)) 
  (h2 : A = B + 10) 
  (h3 : A + B + C = 360) : 
  x = 2 / 7 ∧ B = x * (A + C) :=
by
  sorry -- The proof steps can be filled in

end fraction_B_A_C_l1429_142996


namespace plane_through_points_l1429_142931

def point := (ℝ × ℝ × ℝ)

def plane_equation (A B C D : ℤ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_through_points : 
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.gcd (Int.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1) ∧
  plane_equation A B C D 2 (-3) 5 ∧
  plane_equation A B C D (-1) (-3) 7 ∧
  plane_equation A B C D (-4) (-5) 6 ∧
  (A = 2) ∧ (B = -9) ∧ (C = 3) ∧ (D = -46) :=
sorry

end plane_through_points_l1429_142931


namespace sum_of_three_smallest_positive_solutions_l1429_142909

theorem sum_of_three_smallest_positive_solutions :
  let sol1 := 2
  let sol2 := 8 / 3
  let sol3 := 7 / 2
  sol1 + sol2 + sol3 = 8 + 1 / 6 :=
by
  sorry

end sum_of_three_smallest_positive_solutions_l1429_142909


namespace sum_of_sequence_l1429_142945

noncomputable def sequence_sum (n : ℕ) : ℤ :=
  6 * 2^n - (n + 6)

theorem sum_of_sequence (a S : ℕ → ℤ) (n : ℕ) :
  a 1 = 5 →
  (∀ n : ℕ, 1 ≤ n → S (n + 1) = 2 * S n + n + 5) →
  S n = sequence_sum n :=
by sorry

end sum_of_sequence_l1429_142945


namespace find_k_l1429_142966

theorem find_k (k : ℝ) : 
  (1 : ℝ)^2 + k * 1 - 3 = 0 → k = 2 :=
by
  intro h
  sorry

end find_k_l1429_142966


namespace kim_probability_same_color_l1429_142954

noncomputable def probability_same_color (total_shoes : ℕ) (pairs_of_shoes : ℕ) : ℚ :=
  let total_selections := (total_shoes * (total_shoes - 1)) / 2
  let successful_selections := pairs_of_shoes
  successful_selections / total_selections

theorem kim_probability_same_color :
  probability_same_color 10 5 = 1 / 9 :=
by
  unfold probability_same_color
  have h_total : (10 * 9) / 2 = 45 := by norm_num
  have h_success : 5 = 5 := by norm_num
  rw [h_total, h_success]
  norm_num
  done

end kim_probability_same_color_l1429_142954


namespace greatest_divisor_of_976543_and_897623_l1429_142991

theorem greatest_divisor_of_976543_and_897623 :
  Nat.gcd (976543 - 7) (897623 - 11) = 4 := by
  sorry

end greatest_divisor_of_976543_and_897623_l1429_142991


namespace bicycles_purchased_on_Friday_l1429_142934

theorem bicycles_purchased_on_Friday (F : ℕ) : (F - 10) - 4 + 2 = 3 → F = 15 := by
  intro h
  sorry

end bicycles_purchased_on_Friday_l1429_142934


namespace number_of_students_l1429_142911

-- Definitions based on conditions
def candy_bar_cost : ℝ := 2
def chips_cost : ℝ := 0.5
def total_cost_per_student : ℝ := candy_bar_cost + 2 * chips_cost
def total_amount : ℝ := 15

-- Statement to prove
theorem number_of_students : (total_amount / total_cost_per_student) = 5 :=
by
  sorry

end number_of_students_l1429_142911


namespace total_amount_is_4000_l1429_142910

-- Define the amount put at a 3% interest rate
def amount_at_3_percent : ℝ := 2800

-- Define the total annual interest from both investments
def total_annual_interest : ℝ := 144

-- Define the interest rate for the amount put at 3% and 5%
def interest_rate_3_percent : ℝ := 0.03
def interest_rate_5_percent : ℝ := 0.05

-- Define the total amount to be proved
def total_amount_divided (T : ℝ) : Prop :=
  interest_rate_3_percent * amount_at_3_percent + 
  interest_rate_5_percent * (T - amount_at_3_percent) = total_annual_interest

-- The theorem that states the total amount divided is Rs. 4000
theorem total_amount_is_4000 : ∃ T : ℝ, total_amount_divided T ∧ T = 4000 :=
by
  use 4000
  unfold total_amount_divided
  simp
  sorry

end total_amount_is_4000_l1429_142910


namespace point_outside_circle_l1429_142939

theorem point_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, (a * x + b * y = 1) ∧ (x^2 + y^2 = 1)) :
  a^2 + b^2 > 1 := by
  sorry

end point_outside_circle_l1429_142939


namespace set_expression_l1429_142999

def is_natural_number (x : ℚ) : Prop :=
  ∃ n : ℕ, x = n

theorem set_expression :
  {x : ℕ | is_natural_number (6 / (5 - x) : ℚ)} = {2, 3, 4} :=
sorry

end set_expression_l1429_142999


namespace supermarket_spent_more_than_collected_l1429_142927

-- Given conditions
def initial_amount : ℕ := 53
def collected_amount : ℕ := 91
def amount_left : ℕ := 14

-- Finding the total amount before shopping and amount spent in supermarket
def total_amount : ℕ := initial_amount + collected_amount
def spent_amount : ℕ := total_amount - amount_left

-- Prove that the difference between spent amount and collected amount is 39
theorem supermarket_spent_more_than_collected : (spent_amount - collected_amount) = 39 := by
  -- The proof will go here
  sorry

end supermarket_spent_more_than_collected_l1429_142927


namespace solve_fraction_eq_zero_l1429_142967

theorem solve_fraction_eq_zero (x : ℝ) (h₁ : 3 - x = 0) (h₂ : 4 + 2 * x ≠ 0) : x = 3 :=
by sorry

end solve_fraction_eq_zero_l1429_142967


namespace jia_jia_clover_count_l1429_142976

theorem jia_jia_clover_count : ∃ x : ℕ, 3 * x + 4 = 100 ∧ x = 32 := by
  sorry

end jia_jia_clover_count_l1429_142976


namespace solution_exists_unique_n_l1429_142971

theorem solution_exists_unique_n (n : ℕ) : 
  (∀ m : ℕ, (10 * m > 120) ∨ ∃ k1 k2 k3 : ℕ, 10 * k1 + n * k2 + (n + 1) * k3 = 120) = false → 
  n = 16 := by sorry

end solution_exists_unique_n_l1429_142971


namespace average_of_all_digits_l1429_142926

theorem average_of_all_digits (d : List ℕ) (h_len : d.length = 9)
  (h1 : (d.take 4).sum = 32)
  (h2 : (d.drop 4).sum = 130) : 
  (d.sum / d.length : ℚ) = 18 := 
by
  sorry

end average_of_all_digits_l1429_142926


namespace negation_proposition_l1429_142913

theorem negation_proposition :
  ¬(∀ x : ℝ, x^2 > x) ↔ ∃ x : ℝ, x^2 ≤ x :=
sorry

end negation_proposition_l1429_142913


namespace no_equalities_l1429_142916

def f1 (x : ℤ) : ℤ := x * (x - 2007)
def f2 (x : ℤ) : ℤ := (x - 1) * (x - 2006)
def f1004 (x : ℤ) : ℤ := (x - 1003) * (x - 1004)

theorem no_equalities (x : ℤ) (h : 0 ≤ x ∧ x ≤ 2007) :
  ¬(f1 x = f2 x ∨ f1 x = f1004 x ∨ f2 x = f1004 x) :=
by
  sorry

end no_equalities_l1429_142916


namespace estimate_students_in_range_l1429_142964

noncomputable def n_students := 3000
noncomputable def score_range_low := 70
noncomputable def score_range_high := 80
noncomputable def est_students_in_range := 408

theorem estimate_students_in_range : ∀ (n : ℕ) (k : ℕ), n = n_students →
  k = est_students_in_range →
  normal_distribution :=
sorry

end estimate_students_in_range_l1429_142964


namespace second_number_is_40_l1429_142907

-- Defining the problem
theorem second_number_is_40
  (a b c : ℚ)
  (h1 : a + b + c = 120)
  (h2 : a = (3/4 : ℚ) * b)
  (h3 : c = (5/4 : ℚ) * b) :
  b = 40 :=
sorry

end second_number_is_40_l1429_142907


namespace dark_more_than_light_l1429_142961

-- Define the board size
def board_size : ℕ := 9

-- Define the number of dark squares in odd rows
def dark_in_odd_row : ℕ := 5

-- Define the number of light squares in odd rows
def light_in_odd_row : ℕ := 4

-- Define the number of dark squares in even rows
def dark_in_even_row : ℕ := 4

-- Define the number of light squares in even rows
def light_in_even_row : ℕ := 5

-- Calculate the total number of dark squares
def total_dark_squares : ℕ := (dark_in_odd_row * ((board_size + 1) / 2)) + (dark_in_even_row * (board_size / 2))

-- Calculate the total number of light squares
def total_light_squares : ℕ := (light_in_odd_row * ((board_size + 1) / 2)) + (light_in_even_row * (board_size / 2))

-- Define the main theorem
theorem dark_more_than_light : total_dark_squares - total_light_squares = 1 := by
  sorry

end dark_more_than_light_l1429_142961


namespace graph_transform_l1429_142944

-- Define the quadratic function y1 as y = -2x^2 + 4x + 1
def y1 (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

-- Define the quadratic function y2 as y = -2x^2
def y2 (x : ℝ) : ℝ := -2 * x^2

-- Define the transformation function for moving 1 unit to the left and 3 units down
def transform (y : ℝ → ℝ) (x : ℝ) : ℝ := y (x + 1) - 3

-- Statement to prove
theorem graph_transform : ∀ x : ℝ, transform y1 x = y2 x :=
by
  intros x
  sorry

end graph_transform_l1429_142944


namespace largest_possible_n_l1429_142918

theorem largest_possible_n :
  ∃ (m n : ℕ), (0 < m) ∧ (0 < n) ∧ (m + n = 10) ∧ (n = 9) :=
by
  sorry

end largest_possible_n_l1429_142918


namespace opposite_of_reciprocal_negative_one_third_l1429_142902

theorem opposite_of_reciprocal_negative_one_third : -(1 / (-1 / 3)) = 3 := by
  sorry

end opposite_of_reciprocal_negative_one_third_l1429_142902


namespace problem1_problem2_l1429_142952

-- Problem 1 Statement
theorem problem1 : (3 * Real.sqrt 48 - 2 * Real.sqrt 27) / Real.sqrt 3 = 6 :=
by sorry

-- Problem 2 Statement
theorem problem2 : 
  (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + 1 / (2 - Real.sqrt 5) = -3 - Real.sqrt 5 :=
by sorry

end problem1_problem2_l1429_142952


namespace non_zero_number_is_nine_l1429_142905

theorem non_zero_number_is_nine {x : ℝ} (h1 : (x + x^2) / 2 = 5 * x) (h2 : x ≠ 0) : x = 9 :=
by
  sorry

end non_zero_number_is_nine_l1429_142905


namespace bob_hair_length_l1429_142960

theorem bob_hair_length (h_0 : ℝ) (r : ℝ) (t : ℝ) (months_per_year : ℝ) (h : ℝ) :
  h_0 = 6 ∧ r = 0.5 ∧ t = 5 ∧ months_per_year = 12 → h = h_0 + r * months_per_year * t :=
sorry

end bob_hair_length_l1429_142960


namespace Tony_science_degree_years_l1429_142904

theorem Tony_science_degree_years (X : ℕ) (Total : ℕ)
  (h1 : Total = 14)
  (h2 : Total = X + 2 * X + 2) :
  X = 4 :=
by
  sorry

end Tony_science_degree_years_l1429_142904


namespace expand_expression_l1429_142943

theorem expand_expression (y : ℚ) : 5 * (4 * y^3 - 3 * y^2 + 2 * y - 6) = 20 * y^3 - 15 * y^2 + 10 * y - 30 := by
  sorry

end expand_expression_l1429_142943


namespace limit_of_power_seq_l1429_142972

-- Define the problem and its conditions
theorem limit_of_power_seq (a : ℝ) (h : 0 < a ∨ 1 < a) :
  (0 < a ∧ a < 1 → ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, a^n < ε) ∧ 
  (1 < a → ∀ N > 0, ∃ n : ℕ, a^n > N) :=
by
  sorry

end limit_of_power_seq_l1429_142972


namespace line_tangent_to_parabola_l1429_142957

theorem line_tangent_to_parabola (c : ℝ) : (∀ (x y : ℝ), 2 * x - y + c = 0 ∧ x^2 = 4 * y) → c = -4 := by
  sorry

end line_tangent_to_parabola_l1429_142957


namespace parameter_a_solution_exists_l1429_142953

theorem parameter_a_solution_exists (a : ℝ) : 
  (a < -2 / 3 ∨ a > 0) → ∃ b x y : ℝ, 
  x = 6 / a - abs (y - a) ∧ x^2 + y^2 + b^2 + 63 = 2 * (b * y - 8 * x) :=
by
  intro h
  sorry

end parameter_a_solution_exists_l1429_142953


namespace smallest_q_p_difference_l1429_142932

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end smallest_q_p_difference_l1429_142932


namespace find_missing_number_l1429_142951

theorem find_missing_number (x : ℚ) (h : (476 + 424) * 2 - x * 476 * 424 = 2704) : 
  x = -1 / 223 :=
by
  sorry

end find_missing_number_l1429_142951


namespace find_center_of_ellipse_l1429_142924

-- Defining the equation of the ellipse
def ellipse (x y : ℝ) : Prop := 2*x^2 + 2*x*y + y^2 + 2*x + 2*y - 4 = 0

-- The coordinates of the center
def center_of_ellipse : ℝ × ℝ := (0, -1)

-- The theorem asserting the center of the ellipse
theorem find_center_of_ellipse (x y : ℝ) (h : ellipse x y) : (x, y) = center_of_ellipse :=
sorry

end find_center_of_ellipse_l1429_142924


namespace Lewis_found_20_items_l1429_142900

-- Define the number of items Tanya found
def Tanya_items : ℕ := 4

-- Define the number of items Samantha found
def Samantha_items : ℕ := 4 * Tanya_items

-- Define the number of items Lewis found
def Lewis_items : ℕ := Samantha_items + 4

-- Theorem to prove the number of items Lewis found
theorem Lewis_found_20_items : Lewis_items = 20 := by
  sorry

end Lewis_found_20_items_l1429_142900


namespace positivity_of_fraction_l1429_142942

theorem positivity_of_fraction
  (a b c d x1 x2 x3 x4 : ℝ)
  (h_neg_a : a < 0)
  (h_neg_b : b < 0)
  (h_neg_c : c < 0)
  (h_neg_d : d < 0)
  (h_abs : |x1 - a| + |x2 + b| + |x3 - c| + |x4 + d| = 0) :
  (x1 * x2 / (x3 * x4) > 0) := by
  sorry

end positivity_of_fraction_l1429_142942


namespace solve_equation_l1429_142937

theorem solve_equation (x : ℝ) : 
  (x - 1) / 2 - (2 * x + 3) / 3 = 1 ↔ 3 * (x - 1) - 2 * (2 * x + 3) = 6 := 
sorry

end solve_equation_l1429_142937


namespace sin_minus_pi_over_3_eq_neg_four_fifths_l1429_142923

theorem sin_minus_pi_over_3_eq_neg_four_fifths
  (α : ℝ)
  (h : Real.cos (α + π / 6) = 4 / 5) :
  Real.sin (α - π / 3) = - (4 / 5) :=
by
  sorry

end sin_minus_pi_over_3_eq_neg_four_fifths_l1429_142923


namespace simplify_expression_calculate_expression_l1429_142982

-- Problem 1
theorem simplify_expression (x : ℝ) : 
  (x + 1) * (x + 1) - x * (x + 1) = x + 1 := by
  sorry

-- Problem 2
theorem calculate_expression : 
  (-1 : ℝ) ^ 2023 + 2 ^ (-2 : ℝ) + 4 * (Real.cos (Real.pi / 6))^2 = 9 / 4 := by
  sorry

end simplify_expression_calculate_expression_l1429_142982


namespace sum_of_dimensions_l1429_142955

noncomputable def rectangular_prism_dimensions (A B C : ℝ) : Prop :=
  (A * B = 30) ∧ (A * C = 40) ∧ (B * C = 60)

theorem sum_of_dimensions (A B C : ℝ) (h : rectangular_prism_dimensions A B C) : A + B + C = 9 * Real.sqrt 5 :=
by
  sorry

end sum_of_dimensions_l1429_142955


namespace blanch_breakfast_slices_l1429_142970

-- Define the initial number of slices
def initial_slices : ℕ := 15

-- Define the slices eaten at different times
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5

-- Define the number of slices left
def slices_left : ℕ := 2

-- Calculate the total slices eaten during lunch, snack, and dinner
def total_eaten_ex_breakfast : ℕ := lunch_slices + snack_slices + dinner_slices

-- Define the slices eaten during breakfast
def breakfast_slices : ℕ := initial_slices - total_eaten_ex_breakfast - slices_left

-- The theorem to prove
theorem blanch_breakfast_slices : breakfast_slices = 4 := by
  sorry

end blanch_breakfast_slices_l1429_142970


namespace does_not_pass_through_second_quadrant_l1429_142959

def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

theorem does_not_pass_through_second_quadrant :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ x < 0 ∧ y > 0 :=
sorry

end does_not_pass_through_second_quadrant_l1429_142959


namespace additional_laps_needed_l1429_142983

-- Definitions of problem conditions
def total_required_distance : ℕ := 2400
def lap_length : ℕ := 150
def madison_laps : ℕ := 6
def gigi_laps : ℕ := 6

-- Target statement to prove the number of additional laps needed
theorem additional_laps_needed : (total_required_distance - (madison_laps + gigi_laps) * lap_length) / lap_length = 4 := by
  sorry

end additional_laps_needed_l1429_142983


namespace fred_total_cards_l1429_142979

theorem fred_total_cards 
  (initial_cards : ℕ := 26) 
  (cards_given_to_mary : ℕ := 18) 
  (unopened_box_cards : ℕ := 40) : 
  initial_cards - cards_given_to_mary + unopened_box_cards = 48 := 
by 
  sorry

end fred_total_cards_l1429_142979


namespace min_adj_white_pairs_l1429_142936

theorem min_adj_white_pairs (black_cells : Finset (Fin 64)) (h_black_count : black_cells.card = 20) : 
  ∃ rem_white_pairs, rem_white_pairs = 34 := 
sorry

end min_adj_white_pairs_l1429_142936


namespace max_ab_l1429_142997

theorem max_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ab ≤ 1 / 4 :=
sorry

end max_ab_l1429_142997


namespace decompose_series_l1429_142925

-- Define the 11-arithmetic Fibonacci sequence using the given series
def Φ₁₁₀ (n : ℕ) : ℕ :=
  if n % 11 = 0 then 0 else
  if n % 11 = 1 then 1 else
  if n % 11 = 2 then 1 else
  if n % 11 = 3 then 2 else
  if n % 11 = 4 then 3 else
  if n % 11 = 5 then 5 else
  if n % 11 = 6 then 8 else
  if n % 11 = 7 then 2 else
  if n % 11 = 8 then 10 else
  if n % 11 = 9 then 1 else
  0

-- Define the two geometric progressions
def G₁ (n : ℕ) : ℤ := 3 * (8 ^ n)
def G₂ (n : ℕ) : ℤ := 8 * (4 ^ n)

-- The decomposed sequence
def decomposedSequence (n : ℕ) : ℤ := G₁ n + G₂ n

-- The theorem to prove the decomposition
theorem decompose_series : ∀ n : ℕ, Φ₁₁₀ n = decomposedSequence n := by
  sorry

end decompose_series_l1429_142925


namespace find_angle4_l1429_142973

theorem find_angle4
  (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 = 70)
  (h2 : angle2 = 110)
  (h3 : angle3 = 40)
  (h4 : angle2 + angle3 + angle4 = 180) :
  angle4 = 30 := 
  sorry

end find_angle4_l1429_142973


namespace train_length_490_l1429_142994

noncomputable def train_length (speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

theorem train_length_490 :
  train_length 63 28 = 490 := by
  -- Proof goes here
  sorry

end train_length_490_l1429_142994


namespace girl_buys_roses_l1429_142965

theorem girl_buys_roses 
  (x y : ℤ)
  (h1 : y = 1)
  (h2 : x > 0)
  (h3 : (200 : ℤ) / (x + 10) < (100 : ℤ) / x)
  (h4 : (80 : ℤ) / 12 = ((100 : ℤ) / x) - ((200 : ℤ) / (x + 10))) :
  x = 5 ∧ y = 1 :=
by
  sorry

end girl_buys_roses_l1429_142965


namespace ways_to_divide_day_l1429_142998

theorem ways_to_divide_day (n m : ℕ) (h : n * m = 86400) : 
  (∃ k : ℕ, k = 96) :=
  sorry

end ways_to_divide_day_l1429_142998


namespace problem_f_2016_eq_l1429_142975

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem problem_f_2016_eq :
  ∀ (a b : ℝ),
  f a b 2016 + f a b (-2016) + f' a b 2017 - f' a b (-2017) = 8 + 2 * b * 2016^3 :=
by
  intro a b
  sorry

end problem_f_2016_eq_l1429_142975


namespace D_72_eq_81_l1429_142990

-- Definition of the function for the number of decompositions
def D (n : Nat) : Nat :=
  -- D(n) would ideally be implemented here as per the given conditions
  sorry

-- Prime factorization of 72
def prime_factorization_72 : List Nat :=
  [2, 2, 2, 3, 3]

-- Statement to prove
theorem D_72_eq_81 : D 72 = 81 :=
by
  -- Placeholder for actual proof
  sorry

end D_72_eq_81_l1429_142990


namespace red_star_team_wins_l1429_142974

theorem red_star_team_wins (x y : ℕ) (h1 : x + y = 9) (h2 : 3 * x + y = 23) : x = 7 := by
  sorry

end red_star_team_wins_l1429_142974


namespace table_area_l1429_142930

/-- Given the combined area of three table runners is 224 square inches, 
     overlapping the runners to cover 80% of a table results in exactly 24 square inches being covered by 
     two layers, and the area covered by three layers is 30 square inches,
     prove that the area of the table is 175 square inches. -/
theorem table_area (A : ℝ) (S T H : ℝ) (h1 : S + 2 * T + 3 * H = 224)
   (h2 : 0.80 * A = S + T + H) (h3 : T = 24) (h4 : H = 30) : A = 175 := 
sorry

end table_area_l1429_142930


namespace find_integer_l1429_142929

theorem find_integer (N : ℤ) (hN : N^2 + N = 12) (h_pos : 0 < N) : N = 3 :=
sorry

end find_integer_l1429_142929


namespace intersection_M_N_l1429_142908

def M : Set ℝ := { x : ℝ | x + 1 ≥ 0 }
def N : Set ℝ := { x : ℝ | x^2 < 4 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | -1 ≤ x ∧ x < 2 } :=
sorry

end intersection_M_N_l1429_142908


namespace maximize_wz_xy_zx_l1429_142906

-- Variables definition
variables {w x y z : ℝ}

-- Main statement
theorem maximize_wz_xy_zx (h_sum : w + x + y + z = 200) (h_nonneg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) :
  (w * z + x * y + z * x) ≤ 7500 :=
sorry

end maximize_wz_xy_zx_l1429_142906


namespace factor_polynomial_l1429_142962

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l1429_142962


namespace g_of_5_l1429_142940

theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x ≠ 0, 4 * g x - 3 * g (1 / x) = 2 * x) :
  g 5 = 402 / 70 := 
sorry

end g_of_5_l1429_142940


namespace triangle_area_is_14_l1429_142963

def vector : Type := (ℝ × ℝ)
def a : vector := (4, -1)
def b : vector := (2 * 2, 2 * 3)

noncomputable def parallelogram_area (u v : vector) : ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  abs (ux * vy - uy * vx)

noncomputable def triangle_area (u v : vector) : ℝ :=
  (parallelogram_area u v) / 2

theorem triangle_area_is_14 : triangle_area a b = 14 :=
by
  unfold a b triangle_area parallelogram_area
  sorry

end triangle_area_is_14_l1429_142963


namespace ram_leela_money_next_week_l1429_142903

theorem ram_leela_money_next_week (x : ℕ)
  (initial_money : ℕ := 100)
  (total_money_after_52_weeks : ℕ := 1478)
  (sum_of_series : ℕ := 1378) :
  let n := 52
  let a1 := x
  let an := x + 51
  let S := (n / 2) * (a1 + an)
  initial_money + S = total_money_after_52_weeks → x = 1 :=
by
  sorry

end ram_leela_money_next_week_l1429_142903


namespace find_f6_l1429_142915

variable {R : Type} [LinearOrderedField R]

def f : R → R := sorry

theorem find_f6 (h1 : ∀ x y : R, f (x - y) = f x * f y) (h2 : ∀ x : R, f x ≠ 0) : f 6 = 1 :=
sorry

end find_f6_l1429_142915


namespace root_of_quadratic_eq_l1429_142949

theorem root_of_quadratic_eq (a b : ℝ) (h : a + b - 3 = 0) : a + b = 3 :=
sorry

end root_of_quadratic_eq_l1429_142949


namespace find_integer_n_l1429_142912

theorem find_integer_n :
  ∃ n : ℤ, 
    50 ≤ n ∧ n ≤ 120 ∧ (n % 5 = 0) ∧ (n % 6 = 3) ∧ (n % 7 = 4) ∧ n = 165 :=
by
  sorry

end find_integer_n_l1429_142912


namespace additional_interest_due_to_higher_rate_l1429_142938

def principal : ℝ := 2500
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem additional_interest_due_to_higher_rate :
  simple_interest principal rate1 time - simple_interest principal rate2 time = 300 :=
by
  sorry

end additional_interest_due_to_higher_rate_l1429_142938


namespace smallest_two_digit_palindrome_l1429_142978

def is_palindrome {α : Type} [DecidableEq α] (xs : List α) : Prop :=
  xs = xs.reverse

-- A number is a two-digit palindrome in base 5 if it has the form ab5 where a and b are digits 0-4
def two_digit_palindrome_base5 (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 5 ∧ b < 5 ∧ a ≠ 0 ∧ n = a * 5 + b ∧ is_palindrome [a, b]

-- A number is a three-digit palindrome in base 2 if it has the form abc2 where a = c and b can vary (0-1)
def three_digit_palindrome_base2 (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a < 2 ∧ b < 2 ∧ c < 2 ∧ a = c ∧ n = a * 4 + b * 2 + c ∧ is_palindrome [a, b, c]

theorem smallest_two_digit_palindrome :
  ∃ n, two_digit_palindrome_base5 n ∧ three_digit_palindrome_base2 n ∧
       (∀ m, two_digit_palindrome_base5 m ∧ three_digit_palindrome_base2 m → n ≤ m) :=
sorry

end smallest_two_digit_palindrome_l1429_142978


namespace max_value_of_g_l1429_142941

noncomputable def f1 (x : ℝ) : ℝ := 3 * x + 3
noncomputable def f2 (x : ℝ) : ℝ := (1/3) * x + 2
noncomputable def f3 (x : ℝ) : ℝ := -x + 8

noncomputable def g (x : ℝ) : ℝ := min (min (f1 x) (f2 x)) (f3 x)

theorem max_value_of_g : ∃ x : ℝ, g x = 3.5 :=
by
  sorry

end max_value_of_g_l1429_142941


namespace amoeba_reproduction_time_l1429_142946

/--
An amoeba reproduces by fission, splitting itself into two separate amoebae. 
It takes 8 days for one amoeba to divide into 16 amoebae. 

Prove that it takes 2 days for an amoeba to reproduce.
-/
theorem amoeba_reproduction_time (day_per_cycle : ℕ) (n_cycles : ℕ) 
  (h1 : n_cycles * day_per_cycle = 8)
  (h2 : 2^n_cycles = 16) : 
  day_per_cycle = 2 :=
by
  sorry

end amoeba_reproduction_time_l1429_142946


namespace boxes_containing_pans_l1429_142922

def num_boxes : Nat := 26
def num_teacups_per_box : Nat := 20
def num_cups_broken_per_box : Nat := 2
def teacups_left : Nat := 180

def num_teacup_boxes (num_boxes : Nat) (num_teacups_per_box : Nat) (num_cups_broken_per_box : Nat) (teacups_left : Nat) : Nat :=
  teacups_left / (num_teacups_per_box - num_cups_broken_per_box)

def num_remaining_boxes (num_boxes : Nat) (num_teacup_boxes : Nat) : Nat :=
  num_boxes - num_teacup_boxes

def num_pans_boxes (num_remaining_boxes : Nat) : Nat :=
  num_remaining_boxes / 2

theorem boxes_containing_pans : ∀ (num_boxes num_teacups_per_box num_cups_broken_per_box teacups_left : Nat),
  num_boxes = 26 →
  num_teacups_per_box = 20 →
  num_cups_broken_per_box = 2 →
  teacups_left = 180 →
  num_pans_boxes (num_remaining_boxes num_boxes (num_teacup_boxes num_boxes num_teacups_per_box num_cups_broken_per_box teacups_left)) = 8 :=
by
  intros
  sorry

end boxes_containing_pans_l1429_142922


namespace slices_per_large_pizza_l1429_142986

structure PizzaData where
  total_pizzas : Nat
  small_pizzas : Nat
  medium_pizzas : Nat
  slices_per_small : Nat
  slices_per_medium : Nat
  total_slices : Nat

def large_slices (data : PizzaData) : Nat := (data.total_slices - (data.small_pizzas * data.slices_per_small + data.medium_pizzas * data.slices_per_medium)) / (data.total_pizzas - data.small_pizzas - data.medium_pizzas)

def PizzaSlicingConditions := {data : PizzaData // 
  data.total_pizzas = 15 ∧
  data.small_pizzas = 4 ∧
  data.medium_pizzas = 5 ∧
  data.slices_per_small = 6 ∧
  data.slices_per_medium = 8 ∧
  data.total_slices = 136}

theorem slices_per_large_pizza (data : PizzaSlicingConditions) : large_slices data.val = 12 :=
by
  sorry

end slices_per_large_pizza_l1429_142986
