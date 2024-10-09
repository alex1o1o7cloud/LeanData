import Mathlib

namespace domain_of_f_l1256_125693

noncomputable def f (x : ℝ) := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ x + 1 ≠ 1 ∧ 4 - x ≥ 0} = { x : ℝ | (-1 < x ∧ x ≤ 4) ∧ x ≠ 0 } :=
sorry

end domain_of_f_l1256_125693


namespace value_of_x_plus_y_l1256_125665

theorem value_of_x_plus_y (x y : ℝ) (h : (x + 1)^2 + |y - 6| = 0) : x + y = 5 :=
by
sorry

end value_of_x_plus_y_l1256_125665


namespace third_consecutive_odd_integer_l1256_125650

theorem third_consecutive_odd_integer (x : ℤ) (h : 3 * x = 2 * (x + 4) + 3) : x + 4 = 15 :=
sorry

end third_consecutive_odd_integer_l1256_125650


namespace paul_spent_252_dollars_l1256_125615

noncomputable def total_cost_before_discounts : ℝ :=
  let dress_shirts := 4 * 15
  let pants := 2 * 40
  let suit := 150
  let sweaters := 2 * 30
  dress_shirts + pants + suit + sweaters

noncomputable def store_discount : ℝ := 0.20

noncomputable def coupon_discount : ℝ := 0.10

noncomputable def total_cost_after_store_discount : ℝ :=
  let initial_total := total_cost_before_discounts
  initial_total - store_discount * initial_total

noncomputable def final_total : ℝ :=
  let intermediate_total := total_cost_after_store_discount
  intermediate_total - coupon_discount * intermediate_total

theorem paul_spent_252_dollars :
  final_total = 252 := by
  sorry

end paul_spent_252_dollars_l1256_125615


namespace age_of_youngest_child_l1256_125636

theorem age_of_youngest_child (x : ℕ) :
  (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 55) → x = 5 :=
by
  intro h
  sorry

end age_of_youngest_child_l1256_125636


namespace jack_evening_emails_l1256_125601

theorem jack_evening_emails (ema_morning ema_afternoon ema_afternoon_evening ema_evening : ℕ)
  (h1 : ema_morning = 4)
  (h2 : ema_afternoon = 5)
  (h3 : ema_afternoon_evening = 13)
  (h4 : ema_afternoon_evening = ema_afternoon + ema_evening) :
  ema_evening = 8 :=
by
  sorry

end jack_evening_emails_l1256_125601


namespace find_x2_plus_y2_l1256_125610

theorem find_x2_plus_y2 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + x + y = 90) 
  (h2 : x^2 * y + x * y^2 = 1122) : 
  x^2 + y^2 = 1044 :=
sorry

end find_x2_plus_y2_l1256_125610


namespace x_y_sum_vals_l1256_125691

theorem x_y_sum_vals (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 := 
by
  sorry

end x_y_sum_vals_l1256_125691


namespace correct_operation_l1256_125647

theorem correct_operation (x y m c d : ℝ) : (5 * x * y - 4 * x * y = x * y) :=
by sorry

end correct_operation_l1256_125647


namespace bags_of_white_flour_l1256_125644

theorem bags_of_white_flour (total_flour wheat_flour : ℝ) (h1 : total_flour = 0.3) (h2 : wheat_flour = 0.2) : 
  total_flour - wheat_flour = 0.1 :=
by
  sorry

end bags_of_white_flour_l1256_125644


namespace range_of_m_l1256_125628

theorem range_of_m (m : ℝ) (x y : ℝ)
  (h1 : x + y - 3 * m = 0)
  (h2 : 2 * x - y + 2 * m - 1 = 0)
  (h3 : x > 0)
  (h4 : y < 0) : 
  -1 < m ∧ m < 1/8 := 
sorry

end range_of_m_l1256_125628


namespace sufficient_but_not_necessary_l1256_125640

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 3 → x^2 > 4) ∧ ¬(x^2 > 4 → x > 3) :=
by sorry

end sufficient_but_not_necessary_l1256_125640


namespace highest_score_of_D_l1256_125679

theorem highest_score_of_D
  (a b c d : ℕ)
  (h1 : a + b = c + d)
  (h2 : b + d > a + c)
  (h3 : a > b + c) :
  d > a :=
by
  sorry

end highest_score_of_D_l1256_125679


namespace meaningful_expression_range_l1256_125637

theorem meaningful_expression_range (x : ℝ) :
  (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 1) :=
by
  sorry

end meaningful_expression_range_l1256_125637


namespace travel_time_difference_is_58_minutes_l1256_125664

-- Define the distances and speeds for Minnie
def minnie_uphill_distance := 15
def minnie_uphill_speed := 10
def minnie_downhill_distance := 25
def minnie_downhill_speed := 40
def minnie_flat_distance := 30
def minnie_flat_speed := 25

-- Define the distances and speeds for Penny
def penny_flat_distance := 30
def penny_flat_speed := 35
def penny_downhill_distance := 25
def penny_downhill_speed := 50
def penny_uphill_distance := 15
def penny_uphill_speed := 15

-- Calculate Minnie's total travel time in hours
def minnie_time := (minnie_uphill_distance / minnie_uphill_speed) + 
                   (minnie_downhill_distance / minnie_downhill_speed) + 
                   (minnie_flat_distance / minnie_flat_speed)

-- Calculate Penny's total travel time in hours
def penny_time := (penny_flat_distance / penny_flat_speed) + 
                  (penny_downhill_distance / penny_downhill_speed) +
                  (penny_uphill_distance / penny_uphill_speed)

-- Calculate difference in minutes
def time_difference_minutes := (minnie_time - penny_time) * 60

-- The proof statement
theorem travel_time_difference_is_58_minutes :
  time_difference_minutes = 58 := by
  sorry

end travel_time_difference_is_58_minutes_l1256_125664


namespace derivative_f_intervals_of_monotonicity_extrema_l1256_125632

noncomputable def f (x : ℝ) := (x + 1)^2 * (x - 1)

theorem derivative_f (x : ℝ) : deriv f x = 3 * x^2 + 2 * x - 1 := sorry

theorem intervals_of_monotonicity :
  (∀ x, x < -1 → deriv f x > 0) ∧
  (∀ x, -1 < x ∧ x < -1/3 → deriv f x < 0) ∧
  (∀ x, x > -1/3 → deriv f x > 0) := sorry

theorem extrema :
  f (-1) = 0 ∧
  f (-1/3) = -(32 / 27) := sorry

end derivative_f_intervals_of_monotonicity_extrema_l1256_125632


namespace tan_105_eq_neg2_sub_sqrt3_l1256_125600

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l1256_125600


namespace john_average_speed_l1256_125690

noncomputable def time_uphill : ℝ := 45 / 60 -- 45 minutes converted to hours
noncomputable def distance_uphill : ℝ := 2   -- 2 km

noncomputable def time_downhill : ℝ := 15 / 60 -- 15 minutes converted to hours
noncomputable def distance_downhill : ℝ := 2   -- 2 km

noncomputable def total_distance : ℝ := distance_uphill + distance_downhill
noncomputable def total_time : ℝ := time_uphill + time_downhill

theorem john_average_speed : total_distance / total_time = 4 :=
by
  have h1 : total_distance = 4 := by sorry
  have h2 : total_time = 1 := by sorry
  rw [h1, h2]
  norm_num

end john_average_speed_l1256_125690


namespace john_spent_fraction_on_snacks_l1256_125620

theorem john_spent_fraction_on_snacks (x : ℚ) :
  (∀ (x : ℚ), (1 - x) * 20 - (3 / 4) * (1 - x) * 20 = 4) → (x = 1 / 5) :=
by sorry

end john_spent_fraction_on_snacks_l1256_125620


namespace cone_sphere_ratio_l1256_125638

theorem cone_sphere_ratio (r h : ℝ) (π_pos : 0 < π) (r_pos : 0 < r) :
  (1/3) * π * r^2 * h = (1/3) * (4/3) * π * r^3 → h / r = 4/3 :=
by
  sorry

end cone_sphere_ratio_l1256_125638


namespace cost_ratio_l1256_125658

theorem cost_ratio (S J M : ℝ) (h1 : S = 4) (h2 : M = 0.75 * (S + J)) (h3 : S + J + M = 21) : J / S = 2 :=
by
  sorry

end cost_ratio_l1256_125658


namespace alice_questions_wrong_l1256_125607

theorem alice_questions_wrong (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a + d = b + c + 3) 
  (h3 : c = 7) : 
  a = 8.5 := 
by
  sorry

end alice_questions_wrong_l1256_125607


namespace sara_change_l1256_125696

-- Define the costs of individual items
def cost_book_1 : ℝ := 5.5
def cost_book_2 : ℝ := 6.5
def cost_notebook : ℝ := 3
def cost_bookmarks : ℝ := 2

-- Define the discounts and taxes
def discount_books : ℝ := 0.10
def sales_tax : ℝ := 0.05

-- Define the payment amount
def amount_given : ℝ := 20

-- Calculate the total cost, discount, and final amount
def discounted_book_cost := (cost_book_1 + cost_book_2) * (1 - discount_books)
def subtotal := discounted_book_cost + cost_notebook + cost_bookmarks
def total_with_tax := subtotal * (1 + sales_tax)
def change := amount_given - total_with_tax

-- State the theorem
theorem sara_change : change = 3.41 := by
  sorry

end sara_change_l1256_125696


namespace min_guesses_correct_l1256_125678

noncomputable def min_guesses (n k : ℕ) (h : n > k) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h : n > k) :
  min_guesses n k h = if n = 2 * k then 2 else 1 :=
by
  sorry

end min_guesses_correct_l1256_125678


namespace solution_set_of_absolute_inequality_l1256_125674

theorem solution_set_of_absolute_inequality :
  {x : ℝ | |2 * x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_absolute_inequality_l1256_125674


namespace pow_mod_eq_l1256_125688

theorem pow_mod_eq (h : 101 % 100 = 1) : (101 ^ 50) % 100 = 1 :=
by
  -- Proof omitted
  sorry

end pow_mod_eq_l1256_125688


namespace c_share_is_160_l1256_125627

theorem c_share_is_160 (a b c : ℕ) (total : ℕ) (h1 : 4 * a = 5 * b) (h2 : 5 * b = 10 * c) (h_total : a + b + c = 880) : c = 160 :=
by
  sorry

end c_share_is_160_l1256_125627


namespace possible_values_2a_b_l1256_125684

theorem possible_values_2a_b (a b x y z : ℕ) (h1: a^x = 1994^z) (h2: b^y = 1994^z) (h3: 1/x + 1/y = 1/z) : 
  (2 * a + b = 1001) ∨ (2 * a + b = 1996) :=
by
  sorry

end possible_values_2a_b_l1256_125684


namespace mittens_pairing_possible_l1256_125676

/--
In a kindergarten's lost and found basket, there are 30 mittens: 
10 blue, 10 green, 10 red, 15 right-hand, and 15 left-hand. 

Prove that it is always possible to create matching pairs of one right-hand 
and one left-hand mitten of the same color for 5 children.
-/
theorem mittens_pairing_possible : 
  (∃ (right_blue left_blue right_green left_green right_red left_red : ℕ), 
    right_blue + left_blue + right_green + left_green + right_red + left_red = 30 ∧
    right_blue ≤ 10 ∧ left_blue ≤ 10 ∧
    right_green ≤ 10 ∧ left_green ≤ 10 ∧
    right_red ≤ 10 ∧ left_red ≤ 10 ∧
    right_blue + right_green + right_red = 15 ∧
    left_blue + left_green + left_red = 15) →
  (∃ right_blue left_blue right_green left_green right_red left_red,
    min right_blue left_blue + 
    min right_green left_green + 
    min right_red left_red ≥ 5) :=
sorry

end mittens_pairing_possible_l1256_125676


namespace cost_per_rug_proof_l1256_125651

noncomputable def cost_per_rug (price_sold : ℝ) (number_rugs : ℕ) (profit : ℝ) : ℝ :=
  let total_revenue := number_rugs * price_sold
  let total_cost := total_revenue - profit
  total_cost / number_rugs

theorem cost_per_rug_proof : cost_per_rug 60 20 400 = 40 :=
by
  -- Lean will need the proof steps here, which are skipped
  -- The solution steps illustrate how Lean would derive this in a proof
  sorry

end cost_per_rug_proof_l1256_125651


namespace talia_drives_total_distance_l1256_125698

-- Define the distances for each leg of the trip
def distance_house_to_park : ℕ := 5
def distance_park_to_store : ℕ := 3
def distance_store_to_friend : ℕ := 6
def distance_friend_to_house : ℕ := 4

-- Define the total distance Talia drives
def total_distance := distance_house_to_park + distance_park_to_store + distance_store_to_friend + distance_friend_to_house

-- Prove that the total distance is 18 miles
theorem talia_drives_total_distance : total_distance = 18 := by
  sorry

end talia_drives_total_distance_l1256_125698


namespace mn_minus_7_is_negative_one_l1256_125639

def opp (x : Int) : Int := -x
def largest_negative_integer : Int := -1
def m := opp (-6)
def n := opp largest_negative_integer

theorem mn_minus_7_is_negative_one : m * n - 7 = -1 := by
  sorry

end mn_minus_7_is_negative_one_l1256_125639


namespace parabolas_intersect_diff_l1256_125635

theorem parabolas_intersect_diff (a b c d : ℝ) (h1 : c ≥ a)
  (h2 : b = 3 * a^2 - 6 * a + 3)
  (h3 : d = 3 * c^2 - 6 * c + 3)
  (h4 : b = -2 * a^2 - 4 * a + 6)
  (h5 : d = -2 * c^2 - 4 * c + 6) :
  c - a = 1.6 :=
sorry

end parabolas_intersect_diff_l1256_125635


namespace odd_increasing_min_5_then_neg5_max_on_neg_interval_l1256_125694

-- Definitions using the conditions given in the problem statement
variable {f : ℝ → ℝ}

-- Condition 1: f is odd
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Condition 2: f is increasing on the interval [3, 7]
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f (x) ≤ f (y)

-- Condition 3: Minimum value of f on [3, 7] is 5
def min_value_on_interval (f : ℝ → ℝ) (a b : ℝ) (min_val : ℝ) : Prop :=
  ∃ x, a ≤ x ∧ x ≤ b ∧ f (x) = min_val

-- Lean statement for the proof problem
theorem odd_increasing_min_5_then_neg5_max_on_neg_interval
  (f_odd: odd_function f)
  (f_increasing: increasing_on_interval f 3 7)
  (min_val: min_value_on_interval f 3 7 5) :
  increasing_on_interval f (-7) (-3) ∧ min_value_on_interval f (-7) (-3) (-5) :=
by sorry

end odd_increasing_min_5_then_neg5_max_on_neg_interval_l1256_125694


namespace field_length_is_112_l1256_125673

-- Define the conditions
def is_pond_side_length : ℕ := 8
def pond_area : ℕ := is_pond_side_length * is_pond_side_length
def pond_to_field_area_ratio : ℚ := 1 / 98

-- Define the field properties
def field_area (w l : ℕ) : ℕ := w * l

-- Expressing the condition given length is double the width
def length_double_width (w l : ℕ) : Prop := l = 2 * w

-- Equating the areas based on the ratio given
def area_condition (w l : ℕ) : Prop := pond_area = pond_to_field_area_ratio * field_area w l

-- The main theorem
theorem field_length_is_112 : ∃ w l, length_double_width w l ∧ area_condition w l ∧ l = 112 := by
  sorry

end field_length_is_112_l1256_125673


namespace find_angle_ACB_l1256_125681

-- Definitions corresponding to the conditions
def angleABD : ℝ := 145
def angleBAC : ℝ := 105
def supplementary (a b : ℝ) : Prop := a + b = 180
def triangleAngleSum (a b c : ℝ) : Prop := a + b + c = 180

theorem find_angle_ACB :
  ∃ (angleACB : ℝ), 
    supplementary angleABD angleABC ∧
    triangleAngleSum angleBAC angleABC angleACB ∧
    angleACB = 40 := 
sorry

end find_angle_ACB_l1256_125681


namespace probability_sum_odd_correct_l1256_125648

noncomputable def probability_sum_odd : ℚ :=
  let total_ways := 10
  let ways_sum_odd := 6
  ways_sum_odd / total_ways

theorem probability_sum_odd_correct :
  probability_sum_odd = 3 / 5 :=
by
  unfold probability_sum_odd
  rfl

end probability_sum_odd_correct_l1256_125648


namespace chloe_fifth_test_score_l1256_125629

theorem chloe_fifth_test_score (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 84) (h2 : a2 = 87) (h3 : a3 = 78) (h4 : a4 = 90)
  (h_avg : (a1 + a2 + a3 + a4 + a5) / 5 ≥ 85) : 
  a5 ≥ 86 :=
by
  sorry

end chloe_fifth_test_score_l1256_125629


namespace solution_set_inequality_l1256_125633

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_non_neg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x
def f_neg_half_eq_zero (f : ℝ → ℝ) : Prop := f (-1/2) = 0

-- Problem statement
theorem solution_set_inequality (f : ℝ → ℝ) 
  (hf_even : is_even_function f) 
  (hf_decreasing : is_decreasing_on_non_neg f) 
  (hf_neg_half_zero : f_neg_half_eq_zero f) : 
  {x : ℝ | f (Real.logb (1/4) x) < 0} = {x | x > 2} ∪ {x | 0 < x ∧ x < 1/2} :=
  sorry

end solution_set_inequality_l1256_125633


namespace original_price_l1256_125642

variables (p q d : ℝ)


theorem original_price (x : ℝ) (h : x * (1 + p / 100) * (1 - q / 100) = d) :
  x = 100 * d / (100 + p - q - p * q / 100) := 
sorry

end original_price_l1256_125642


namespace new_average_of_remaining_students_l1256_125687

theorem new_average_of_remaining_students 
  (avg_initial_score : ℝ)
  (num_initial_students : ℕ)
  (dropped_score : ℝ)
  (num_remaining_students : ℕ)
  (new_avg_score : ℝ) 
  (h_avg : avg_initial_score = 62.5)
  (h_num_initial : num_initial_students = 16)
  (h_dropped : dropped_score = 55)
  (h_num_remaining : num_remaining_students = 15)
  (h_new_avg : new_avg_score = 63) :
  let total_initial_score := avg_initial_score * num_initial_students
  let total_remaining_score := total_initial_score - dropped_score
  let calculated_new_avg := total_remaining_score / num_remaining_students
  calculated_new_avg = new_avg_score := 
by
  -- The proof will be provided here
  sorry

end new_average_of_remaining_students_l1256_125687


namespace prove_arithmetic_sequence_l1256_125602

def arithmetic_sequence (x : ℝ) : ℕ → ℝ
| 0 => x - 1
| 1 => x + 1
| 2 => 2 * x + 3
| n => sorry

theorem prove_arithmetic_sequence {x : ℝ} (a : ℕ → ℝ)
  (h_terms : a 0 = x - 1 ∧ a 1 = x + 1 ∧ a 2 = 2 * x + 3)
  (h_arithmetic : ∀ n, a n = a 0 + n * (a 1 - a 0)) :
  x = 0 ∧ ∀ n, a n = 2 * n - 3 :=
by
  sorry

end prove_arithmetic_sequence_l1256_125602


namespace eval_difference_of_squares_l1256_125611

theorem eval_difference_of_squares :
  (81^2 - 49^2 = 4160) :=
by
  -- Since the exact mathematical content is established in a formal context, 
  -- we omit the detailed proof steps.
  sorry

end eval_difference_of_squares_l1256_125611


namespace max_value_expression_l1256_125689

theorem max_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (eq_condition : x^2 - 3 * x * y + 4 * y^2 - z = 0) : 
  ∃ (M : ℝ), M = 1 ∧ (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x^2 - 3 * x * y + 4 * y^2 - z = 0 → (2/x + 1/y - 2/z) ≤ M) := 
by
  sorry

end max_value_expression_l1256_125689


namespace balls_in_boxes_l1256_125660

theorem balls_in_boxes (n m : Nat) (h : n = 6) (k : m = 2) : (m ^ n) = 64 := by
  sorry

end balls_in_boxes_l1256_125660


namespace find_second_number_l1256_125613

theorem find_second_number 
  (k : ℕ)
  (h_k_is_1 : k = 1)
  (h_div_1657 : ∃ q1 : ℕ, 1657 = k * q1 + 10)
  (h_div_x : ∃ q2 : ℕ, ∀ x : ℕ, x = k * q2 + 7 → x = 1655) 
: ∃ x : ℕ, x = 1655 :=
by
  sorry

end find_second_number_l1256_125613


namespace bus_stoppage_time_per_hour_l1256_125692

theorem bus_stoppage_time_per_hour
  (speed_excluding_stoppages : ℕ) 
  (speed_including_stoppages : ℕ)
  (h1 : speed_excluding_stoppages = 54) 
  (h2 : speed_including_stoppages = 45) 
  : (60 * (speed_excluding_stoppages - speed_including_stoppages) / speed_excluding_stoppages) = 10 :=
by sorry

end bus_stoppage_time_per_hour_l1256_125692


namespace same_cost_for_same_sheets_l1256_125618

def John's_Photo_World_cost (x : ℕ) : ℝ := 2.75 * x + 125
def Sam's_Picture_Emporium_cost (x : ℕ) : ℝ := 1.50 * x + 140

theorem same_cost_for_same_sheets :
  ∃ (x : ℕ), John's_Photo_World_cost x = Sam's_Picture_Emporium_cost x ∧ x = 12 :=
by
  sorry

end same_cost_for_same_sheets_l1256_125618


namespace rational_solution_l1256_125616

theorem rational_solution (m n : ℤ) (h : a = (m^4 + n^4 + m^2 * n^2) / (4 * m^2 * n^2)) : 
  ∃ a : ℚ, a = (m^4 + n^4 + m^2 * n^2) / (4 * m^2 * n^2) :=
by {
  sorry
}

end rational_solution_l1256_125616


namespace initial_nickels_l1256_125668

theorem initial_nickels (N : ℕ) (h1 : N + 9 + 2 = 18) : N = 7 :=
by sorry

end initial_nickels_l1256_125668


namespace find_r4_l1256_125625

-- Definitions of the problem conditions
variable (r1 r2 r3 r4 r5 r6 r7 : ℝ)
-- Given radius of the smallest circle
axiom smallest_circle : r1 = 6
-- Given radius of the largest circle
axiom largest_circle : r7 = 24
-- Given that radii of circles form a geometric sequence
axiom geometric_sequence : r2 = r1 * (r7 / r1)^(1/6) ∧ 
                            r3 = r1 * (r7 / r1)^(2/6) ∧
                            r4 = r1 * (r7 / r1)^(3/6) ∧
                            r5 = r1 * (r7 / r1)^(4/6) ∧
                            r6 = r1 * (r7 / r1)^(5/6)

-- Statement to prove
theorem find_r4 : r4 = 12 :=
by
  sorry

end find_r4_l1256_125625


namespace solve_mt_eq_l1256_125631

theorem solve_mt_eq (m n : ℤ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^2 + n) * (m + n^2) = (m - n)^3 →
  (m = -1 ∧ n = -1) ∨ (m = 8 ∧ n = -10) ∨ (m = 9 ∧ n = -6) ∨ (m = 9 ∧ n = -21) :=
by
  sorry

end solve_mt_eq_l1256_125631


namespace quadratic_equation_completes_to_square_l1256_125671

theorem quadratic_equation_completes_to_square :
  ∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x + 2)^2 = 2 :=
by
  intro x
  intro h
  sorry

end quadratic_equation_completes_to_square_l1256_125671


namespace simplify_and_evaluate_fraction_l1256_125663

theorem simplify_and_evaluate_fraction (x : ℤ) (hx : x = 5) :
  ((2 * x + 1) / (x - 1) - 1) / ((x + 2) / (x^2 - 2 * x + 1)) = 4 :=
by
  rw [hx]
  sorry

end simplify_and_evaluate_fraction_l1256_125663


namespace maximal_cards_taken_l1256_125672

theorem maximal_cards_taken (cards : Finset ℕ) (h_cards : ∀ n, n ∈ cards ↔ 1 ≤ n ∧ n ≤ 100)
                            (andriy_cards nick_cards : Finset ℕ)
                            (h_card_count : andriy_cards.card = nick_cards.card)
                            (h_card_relation : ∀ n, n ∈ andriy_cards → (2 * n + 2) ∈ nick_cards) :
                            andriy_cards.card + nick_cards.card ≤ 50 := 
sorry

end maximal_cards_taken_l1256_125672


namespace scientific_calculator_ratio_l1256_125656

theorem scientific_calculator_ratio (total : ℕ) (basic_cost : ℕ) (change : ℕ) (sci_ratio : ℕ → ℕ) (graph_ratio : ℕ → ℕ) : 
  total = 100 →
  basic_cost = 8 →
  sci_ratio basic_cost = 8 * x →
  graph_ratio (sci_ratio basic_cost) = 3 * sci_ratio basic_cost →
  change = 28 →
  8 + (8 * x) + (24 * x) = 72 →
  x = 2 :=
by
  sorry

end scientific_calculator_ratio_l1256_125656


namespace halfway_point_l1256_125659

theorem halfway_point (x1 x2 : ℚ) (h1 : x1 = 1 / 6) (h2 : x2 = 5 / 6) : 
  (x1 + x2) / 2 = 1 / 2 :=
by
  sorry

end halfway_point_l1256_125659


namespace line_equation_l1256_125655

noncomputable def arithmetic_sequence (n : ℕ) (a_1 d : ℝ) : ℝ :=
  a_1 + (n - 1) * d

theorem line_equation
  (a_2 a_4 a_5 : ℝ)
  (a_2_cond : a_2 = arithmetic_sequence 2 a_1 d)
  (a_4_cond : a_4 = arithmetic_sequence 4 a_1 d)
  (a_5_cond : a_5 = arithmetic_sequence 5 a_1 d)
  (sum_cond : a_2 + a_4 = 12)
  (a_5_val : a_5 = 10)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * y = 0 ↔ (x - 0)^2 + (y - 1)^2 = 1)
  : ∃ (line : ℝ → ℝ → Prop), line x y ↔ (6 * x - y + 1 = 0) :=
by
  sorry

end line_equation_l1256_125655


namespace average_increase_l1256_125685

theorem average_increase (x : ℝ) (y : ℝ) (h : y = 0.245 * x + 0.321) : 
  ∀ x_increase : ℝ, x_increase = 1 → (0.245 * (x + x_increase) + 0.321) - (0.245 * x + 0.321) = 0.245 :=
by
  intro x_increase
  intro hx
  rw [hx]
  simp
  sorry

end average_increase_l1256_125685


namespace sally_quarters_total_l1256_125608

/--
Sally originally had 760 quarters. She received 418 more quarters. 
Prove that the total number of quarters Sally has now is 1178.
-/
theorem sally_quarters_total : 
  let original_quarters := 760
  let additional_quarters := 418
  original_quarters + additional_quarters = 1178 :=
by
  let original_quarters := 760
  let additional_quarters := 418
  show original_quarters + additional_quarters = 1178
  sorry

end sally_quarters_total_l1256_125608


namespace proposition_check_l1256_125643

variable (P : ℕ → Prop)

theorem proposition_check 
  (h : ∀ k : ℕ, ¬ P (k + 1) → ¬ P k)
  (h2012 : P 2012) : P 2013 :=
by
  sorry

end proposition_check_l1256_125643


namespace solution_set_of_inequality_l1256_125661

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x + 5) / (x - 1) > x ↔ x < -1 ∨ (1 < x ∧ x < 5) :=
sorry

end solution_set_of_inequality_l1256_125661


namespace projection_of_3_neg2_onto_v_l1256_125695

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1 + a.2 * b.2)
  let scalar := (dot_product u v) / (dot_product v v)
  (scalar * v.1, scalar * v.2)

def v : ℝ × ℝ := (2, -8)

theorem projection_of_3_neg2_onto_v :
  projection (3, -2) v = (11/17, -44/17) :=
by sorry

end projection_of_3_neg2_onto_v_l1256_125695


namespace subset_A_B_l1256_125603

def A : Set ℝ := { x | x^2 - 3 * x + 2 < 0 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem subset_A_B : A ⊆ B := sorry

end subset_A_B_l1256_125603


namespace probability_different_colors_l1256_125609

theorem probability_different_colors :
  let total_chips := 18
  let blue_chips := 7
  let red_chips := 6
  let yellow_chips := 5
  let prob_first_blue := blue_chips / total_chips
  let prob_first_red := red_chips / total_chips
  let prob_first_yellow := yellow_chips / total_chips
  let prob_second_not_blue := (red_chips + yellow_chips) / (total_chips - 1)
  let prob_second_not_red := (blue_chips + yellow_chips) / (total_chips - 1)
  let prob_second_not_yellow := (blue_chips + red_chips) / (total_chips - 1)
  (
    prob_first_blue * prob_second_not_blue +
    prob_first_red * prob_second_not_red +
    prob_first_yellow * prob_second_not_yellow
  ) = 122 / 153 :=
by sorry

end probability_different_colors_l1256_125609


namespace quadratic_root_relationship_l1256_125670

theorem quadratic_root_relationship (a b c : ℝ) (α β : ℝ)
  (h1 : a ≠ 0)
  (h2 : α + β = -b / a)
  (h3 : α * β = c / a)
  (h4 : β = 3 * α) : 
  3 * b^2 = 16 * a * c :=
sorry

end quadratic_root_relationship_l1256_125670


namespace sqrt_of_9_eq_pm_3_l1256_125653

theorem sqrt_of_9_eq_pm_3 : (∃ x : ℤ, x * x = 9) → (∃ x : ℤ, x = 3 ∨ x = -3) :=
by
  sorry

end sqrt_of_9_eq_pm_3_l1256_125653


namespace monthly_cost_per_iguana_l1256_125646

theorem monthly_cost_per_iguana
  (gecko_cost snake_cost annual_cost : ℕ)
  (monthly_cost_per_iguana : ℕ)
  (gecko_count iguana_count snake_count : ℕ)
  (annual_cost_eq : annual_cost = 1140)
  (gecko_count_eq : gecko_count = 3)
  (iguana_count_eq : iguana_count = 2)
  (snake_count_eq : snake_count = 4)
  (gecko_cost_eq : gecko_cost = 15)
  (snake_cost_eq : snake_cost = 10)
  (total_annual_cost_eq : gecko_count * gecko_cost + iguana_count * monthly_cost_per_iguana * 12 + snake_count * snake_cost * 12 = annual_cost) :
  monthly_cost_per_iguana = 5 :=
by
  sorry

end monthly_cost_per_iguana_l1256_125646


namespace integer_solution_count_l1256_125614

theorem integer_solution_count :
  (∃ x : ℤ, -4 * x ≥ x + 9 ∧ -3 * x ≤ 15 ∧ -5 * x ≥ 3 * x + 24) ↔
  (∃ n : ℕ, n = 3) :=
by
  sorry

end integer_solution_count_l1256_125614


namespace doughnuts_per_box_l1256_125641

theorem doughnuts_per_box (total_doughnuts : ℕ) (boxes : ℕ) (h_doughnuts : total_doughnuts = 48) (h_boxes : boxes = 4) : 
  total_doughnuts / boxes = 12 :=
by
  -- This is a placeholder for the proof
  sorry

end doughnuts_per_box_l1256_125641


namespace solution_eq_c_l1256_125667

variables (x : ℝ) (a : ℝ) 

def p := ∃ x0 : ℝ, (0 < x0) ∧ (3^x0 + x0 = 2016)
def q := ∃ a : ℝ, (0 < a) ∧ (∀ x : ℝ, (|x| - a * x) = (|(x)| - a * (-x)))

theorem solution_eq_c : p ∧ ¬q :=
by {
  sorry -- proof placeholder
}

end solution_eq_c_l1256_125667


namespace coronavirus_case_ratio_l1256_125675

theorem coronavirus_case_ratio (n_first_wave_cases : ℕ) (total_second_wave_cases : ℕ) (n_days : ℕ) 
  (h1 : n_first_wave_cases = 300) (h2 : total_second_wave_cases = 21000) (h3 : n_days = 14) :
  (total_second_wave_cases / n_days) / n_first_wave_cases = 5 :=
by sorry

end coronavirus_case_ratio_l1256_125675


namespace painting_time_l1256_125682

noncomputable def bob_rate : ℕ := 120 / 8
noncomputable def alice_rate : ℕ := 150 / 10
noncomputable def combined_rate : ℕ := bob_rate + alice_rate
noncomputable def total_area : ℕ := 120 + 150
noncomputable def working_time : ℕ := total_area / combined_rate
noncomputable def lunch_break : ℕ := 1
noncomputable def total_time : ℕ := working_time + lunch_break

theorem painting_time : total_time = 10 := by
  -- Proof skipped
  sorry

end painting_time_l1256_125682


namespace carrots_cost_l1256_125606

/-
Define the problem conditions and parameters.
-/
def num_third_grade_classes := 5
def students_per_third_grade_class := 30
def num_fourth_grade_classes := 4
def students_per_fourth_grade_class := 28
def num_fifth_grade_classes := 4
def students_per_fifth_grade_class := 27

def cost_per_hamburger : ℝ := 2.10
def cost_per_cookie : ℝ := 0.20
def total_lunch_cost : ℝ := 1036

/-
Calculate the total number of students.
-/
def total_students : ℕ :=
  (num_third_grade_classes * students_per_third_grade_class) +
  (num_fourth_grade_classes * students_per_fourth_grade_class) +
  (num_fifth_grade_classes * students_per_fifth_grade_class)

/-
Calculate the cost of hamburgers and cookies.
-/
def hamburgers_cost : ℝ := total_students * cost_per_hamburger
def cookies_cost : ℝ := total_students * cost_per_cookie
def total_hamburgers_and_cookies_cost : ℝ := hamburgers_cost + cookies_cost

/-
State the proof problem: How much do the carrots cost?
-/
theorem carrots_cost : total_lunch_cost - total_hamburgers_and_cookies_cost = 185 :=
by
  -- Proof is omitted
  sorry

end carrots_cost_l1256_125606


namespace ratio_eq_neg_1009_l1256_125623

theorem ratio_eq_neg_1009 (p q : ℝ) (h : (1 / p + 1 / q) / (1 / p - 1 / q) = 1009) : (p + q) / (p - q) = -1009 := 
by 
  sorry

end ratio_eq_neg_1009_l1256_125623


namespace range_of_f_l1256_125657

open Set

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (2 * x - 1)

theorem range_of_f : range f = Ici (1 / 2) :=
by
  sorry

end range_of_f_l1256_125657


namespace b_investment_l1256_125630

theorem b_investment (A_invest C_invest total_profit A_profit x : ℝ) 
(h1 : A_invest = 2400) 
(h2 : C_invest = 9600) 
(h3 : total_profit = 9000) 
(h4 : A_profit = 1125)
(h5 : x = (8100000 / 1125)) : 
x = 7200 := by
  rw [h5]
  sorry

end b_investment_l1256_125630


namespace compound_interest_rate_l1256_125619

theorem compound_interest_rate :
  ∃ r : ℝ, (1000 * (1 + r)^3 = 1331.0000000000005) ∧ r = 0.1 :=
by
  sorry

end compound_interest_rate_l1256_125619


namespace truck_weight_l1256_125669

theorem truck_weight (T R : ℝ) (h1 : T + R = 7000) (h2 : R = 0.5 * T - 200) : T = 4800 :=
by sorry

end truck_weight_l1256_125669


namespace number_of_biscuits_per_day_l1256_125612

theorem number_of_biscuits_per_day 
  (price_cupcake : ℝ) (price_cookie : ℝ) (price_biscuit : ℝ)
  (cupcakes_per_day : ℕ) (cookies_per_day : ℕ) (total_earnings_five_days : ℝ) :
  price_cupcake = 1.5 → 
  price_cookie = 2 → 
  price_biscuit = 1 → 
  cupcakes_per_day = 20 → 
  cookies_per_day = 10 → 
  total_earnings_five_days = 350 →
  (total_earnings_five_days - 
   (5 * (cupcakes_per_day * price_cupcake + cookies_per_day * price_cookie))) / (5 * price_biscuit) = 20 :=
by
  intros price_cupcake_eq price_cookie_eq price_biscuit_eq cupcakes_per_day_eq cookies_per_day_eq total_earnings_five_days_eq
  sorry

end number_of_biscuits_per_day_l1256_125612


namespace quadratic_sum_of_b_and_c_l1256_125697

theorem quadratic_sum_of_b_and_c :
  ∃ b c : ℝ, (∀ x : ℝ, x^2 - 20 * x + 36 = (x + b)^2 + c) ∧ b + c = -74 :=
by
  sorry

end quadratic_sum_of_b_and_c_l1256_125697


namespace part1_part2_l1256_125634

def A (x : ℝ) (a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def B (x : ℝ) : Prop := (x - 3) * (2 - x) ≥ 0

theorem part1 (a : ℝ) (ha1: a = 1) :
  ∀ x, (A x 1 ∧ B x) ↔ (2 ≤ x ∧ x < 3) :=
sorry

theorem part2 (a : ℝ) (ha1: a = 1) :
  ∀ x, (A x 1 ∨ B x) ↔ (1 < x ∧ x ≤ 3) :=
sorry

end part1_part2_l1256_125634


namespace find_TS_l1256_125662

-- Definitions of the conditions as given:
def PQ : ℝ := 25
def PS : ℝ := 25
def QR : ℝ := 15
def RS : ℝ := 15
def PT : ℝ := 15
def ST_parallel_QR : Prop := true  -- ST is parallel to QR (used as a given fact)

-- Main statement in Lean:
theorem find_TS (h1 : PQ = 25) (h2 : PS = 25) (h3 : QR = 15) (h4 : RS = 15) (h5 : PT = 15)
               (h6 : ST_parallel_QR) : TS = 24 :=
by
  sorry

end find_TS_l1256_125662


namespace bus_people_next_pickup_point_l1256_125622

theorem bus_people_next_pickup_point (bus_capacity : ℕ) (fraction_first_pickup : ℚ) (cannot_board : ℕ)
  (h1 : bus_capacity = 80)
  (h2 : fraction_first_pickup = 3 / 5)
  (h3 : cannot_board = 18) : 
  ∃ people_next_pickup : ℕ, people_next_pickup = 50 :=
by
  sorry

end bus_people_next_pickup_point_l1256_125622


namespace exponential_function_example_l1256_125649

def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a > 0, a ≠ 1 ∧ ∀ x, f x = a ^ x

theorem exponential_function_example : is_exponential_function (fun x => 3 ^ x) :=
by
  sorry

end exponential_function_example_l1256_125649


namespace polynomial_sum_l1256_125626

def f (x : ℝ) : ℝ := -4 * x^3 - 3 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 + x - 7
def h (x : ℝ) : ℝ := 3 * x^3 + 6 * x^2 + 3 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = x^3 - 2 * x^2 + 6 * x - 10 := by
  sorry

end polynomial_sum_l1256_125626


namespace polynomial_solution_l1256_125683

noncomputable def polynomial_form (P : ℝ → ℝ) : Prop :=
∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ (2 * x * y * z = x + y + z) →
(P x / (y * z) + P y / (z * x) + P z / (x * y) = P (x - y) + P (y - z) + P (z - x))

theorem polynomial_solution (P : ℝ → ℝ) : polynomial_form P → ∃ c : ℝ, ∀ x : ℝ, P x = c * (x ^ 2 + 3) := 
by 
  sorry

end polynomial_solution_l1256_125683


namespace discriminant_positive_l1256_125605

theorem discriminant_positive
  (a b c : ℝ)
  (h : (a + b + c) * c < 0) : b^2 - 4 * a * c > 0 :=
sorry

end discriminant_positive_l1256_125605


namespace sum_of_solutions_l1256_125604

theorem sum_of_solutions (y1 y2 : ℝ) (h1 : y1 + 16 / y1 = 12) (h2 : y2 + 16 / y2 = 12) : 
  y1 + y2 = 12 :=
by
  sorry

end sum_of_solutions_l1256_125604


namespace distinct_real_c_f_ff_ff_five_l1256_125645

def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem distinct_real_c_f_ff_ff_five : 
  (∀ c : ℝ, f (f (f (f c))) = 5 → False) :=
by
  sorry

end distinct_real_c_f_ff_ff_five_l1256_125645


namespace truncated_cone_volume_l1256_125624

theorem truncated_cone_volume 
  (V_initial : ℝ)
  (r_ratio : ℝ)
  (V_final : ℝ)
  (r_ratio_eq : r_ratio = 1 / 2)
  (V_initial_eq : V_initial = 1) :
  V_final = 7 / 8 :=
  sorry

end truncated_cone_volume_l1256_125624


namespace original_circle_area_l1256_125677

theorem original_circle_area (A : ℝ) (h1 : ∃ sector_area : ℝ, sector_area = 5) (h2 : A / 64 = 5) : A = 320 := 
by sorry

end original_circle_area_l1256_125677


namespace ones_digit_of_73_pow_351_l1256_125621

-- Definition of the problem in Lean 4
theorem ones_digit_of_73_pow_351 : (73 ^ 351) % 10 = 7 := by
  sorry

end ones_digit_of_73_pow_351_l1256_125621


namespace each_girl_gets_2_dollars_after_debt_l1256_125617

variable (Lulu_saved : ℕ)
variable (Nora_saved : ℕ)
variable (Tamara_saved : ℕ)
variable (debt : ℕ)
variable (remaining : ℕ)
variable (each_girl_share : ℕ)

-- Conditions
axiom Lulu_saved_cond : Lulu_saved = 6
axiom Nora_saved_cond : Nora_saved = 5 * Lulu_saved
axiom Nora_Tamara_relation : Nora_saved = 3 * Tamara_saved
axiom debt_cond : debt = 40

-- Question == Answer to prove
theorem each_girl_gets_2_dollars_after_debt (total_saved : ℕ) (remaining: ℕ) (each_girl_share: ℕ) :
  total_saved = Tamara_saved + Nora_saved + Lulu_saved →
  remaining = total_saved - debt →
  each_girl_share = remaining / 3 →
  each_girl_share = 2 := 
sorry

end each_girl_gets_2_dollars_after_debt_l1256_125617


namespace maximal_n_is_k_minus_1_l1256_125686

section
variable (k : ℕ) (n : ℕ)
variable (cards : Finset ℕ)
variable (red : List ℕ) (blue : List (List ℕ))

-- Conditions
axiom h_k_pos : k > 1
axiom h_card_count : cards = Finset.range (2 * n + 1)
axiom h_initial_red : red = (List.range' 1 (2 * n)).reverse
axiom h_initial_blue : blue.length = k

-- Question translated to a goal
theorem maximal_n_is_k_minus_1 (h : ∀ (n' : ℕ), n' ≤ (k - 1)) : n = k - 1 :=
sorry
end

end maximal_n_is_k_minus_1_l1256_125686


namespace min_value_of_z_l1256_125666

theorem min_value_of_z : ∀ x : ℝ, ∃ z : ℝ, z = x^2 + 16 * x + 20 ∧ (∀ y : ℝ, y = x^2 + 16 * x + 20 → z ≤ y) → z = -44 := 
by
  sorry

end min_value_of_z_l1256_125666


namespace ones_digit_of_9_pow_47_l1256_125680

theorem ones_digit_of_9_pow_47 : (9 ^ 47) % 10 = 9 := 
by
  sorry

end ones_digit_of_9_pow_47_l1256_125680


namespace total_cartons_used_l1256_125699

theorem total_cartons_used (x : ℕ) (y : ℕ) (h1 : y = 24) (h2 : 2 * x + 3 * y = 100) : x + y = 38 :=
sorry

end total_cartons_used_l1256_125699


namespace jenna_round_trip_pay_l1256_125654

theorem jenna_round_trip_pay :
  let pay_per_mile := 0.40
  let one_way_miles := 400
  let round_trip_miles := 2 * one_way_miles
  let total_pay := round_trip_miles * pay_per_mile
  total_pay = 320 := 
by
  sorry

end jenna_round_trip_pay_l1256_125654


namespace pirate_islands_probability_l1256_125652

open Finset

/-- There are 7 islands.
There is a 1/5 chance of finding an island with treasure only (no traps).
There is a 1/10 chance of finding an island with treasure and traps.
There is a 1/10 chance of finding an island with traps only (no treasure).
There is a 3/5 chance of finding an island with neither treasure nor traps.
We want to prove that the probability of finding exactly 3 islands
with treasure only and the remaining 4 islands with neither treasure
nor traps is 81/2225. -/
theorem pirate_islands_probability :
  (Nat.choose 7 3 : ℚ) * ((1/5)^3) * ((3/5)^4) = 81 / 2225 :=
by
  /- Here goes the proof -/
  sorry

end pirate_islands_probability_l1256_125652
