import Mathlib

namespace bus_speed_excluding_stoppages_l166_16618

theorem bus_speed_excluding_stoppages 
  (v : ℝ) 
  (speed_incl_stoppages : v * 54 / 60 = 45) : 
  v = 50 := 
  by 
    sorry

end bus_speed_excluding_stoppages_l166_16618


namespace five_times_seven_divided_by_ten_l166_16689

theorem five_times_seven_divided_by_ten : (5 * 7 : ℝ) / 10 = 3.5 := 
by 
  sorry

end five_times_seven_divided_by_ten_l166_16689


namespace books_left_on_Fri_l166_16610

-- Define the conditions as constants or values
def books_at_beginning : ℕ := 98
def books_checked_out_Wed : ℕ := 43
def books_returned_Thu : ℕ := 23
def books_checked_out_Thu : ℕ := 5
def books_returned_Fri : ℕ := 7

-- The proof statement to verify the final number of books
theorem books_left_on_Fri (b : ℕ) :
  b = (books_at_beginning - books_checked_out_Wed) + books_returned_Thu - books_checked_out_Thu + books_returned_Fri := 
  sorry

end books_left_on_Fri_l166_16610


namespace inequality_proof_l166_16668

theorem inequality_proof 
  (a b c x y z : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (h_sum : a + b + c = 1) : 
  (x^2 + y^2 + z^2) * 
  (a^3 / (x^2 + 2 * y^2) + b^3 / (y^2 + 2 * z^2) + c^3 / (z^2 + 2 * x^2)) 
  ≥ 1 / 9 := 
by 
  sorry

end inequality_proof_l166_16668


namespace quadratic_no_third_quadrant_l166_16680

theorem quadratic_no_third_quadrant (x y : ℝ) : 
  (y = x^2 - 2 * x) → ¬(x < 0 ∧ y < 0) :=
by
  intro hy
  sorry

end quadratic_no_third_quadrant_l166_16680


namespace index_card_area_l166_16625

theorem index_card_area
  (L W : ℕ)
  (h1 : L = 4)
  (h2 : W = 6)
  (h3 : (L - 1) * W = 18) :
  (L * (W - 1) = 20) :=
by
  sorry

end index_card_area_l166_16625


namespace manolo_rate_change_after_one_hour_l166_16681

variable (masks_in_first_hour : ℕ)
variable (masks_in_remaining_time : ℕ)
variable (total_masks : ℕ)

-- Define conditions as Lean definitions
def first_hour_rate := 1 / 4  -- masks per minute
def remaining_time_rate := 1 / 6  -- masks per minute
def total_time := 4  -- hours
def masks_produced_in_first_hour (t : ℕ) := t * 15  -- t hours, 60 minutes/hour, at 15 masks/hour
def masks_produced_in_remaining_time (t : ℕ) := t * 10 -- (total_time - 1) hours, 60 minutes/hour, at 10 masks/hour

-- Main proof problem statement
theorem manolo_rate_change_after_one_hour :
  masks_in_first_hour = masks_produced_in_first_hour 1 →
  masks_in_remaining_time = masks_produced_in_remaining_time (total_time - 1) →
  total_masks = masks_in_first_hour + masks_in_remaining_time →
  (∃ t : ℕ, t = 1) :=
by
  -- Placeholder, proof not required
  sorry

end manolo_rate_change_after_one_hour_l166_16681


namespace quadratic_expression_representation_quadratic_expression_integer_iff_l166_16662

theorem quadratic_expression_representation (A B C : ℤ) :
  ∃ (k l m : ℤ), 
    (k = 2 * A) ∧ 
    (l = A + B) ∧ 
    (m = C) ∧ 
    (∀ x : ℤ, A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m) := 
sorry

theorem quadratic_expression_integer_iff (A B C : ℤ) :
  (∀ x : ℤ, ∃ k l m : ℤ, (k = 2 * A) ∧ (l = A + B) ∧ (m = C) ∧ (A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m)) ↔ 
  (A % 1 = 0 ∧ B % 1 = 0 ∧ C % 1 = 0) := 
sorry

end quadratic_expression_representation_quadratic_expression_integer_iff_l166_16662


namespace decreasing_geometric_sums_implications_l166_16693

variable (X : Type)
variable (a1 q : ℝ)
variable (S : ℕ → ℝ)

def is_geometric_sequence (a : ℕ → ℝ) :=
∀ n : ℕ, a (n + 1) = a1 * q^n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
S 0 = a 0 ∧ ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

def is_decreasing_sequence (S : ℕ → ℝ) :=
∀ n : ℕ, S (n + 1) < S n

theorem decreasing_geometric_sums_implications (a1 q : ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S (n + 1) < S n) → a1 < 0 ∧ q > 0 := 
by 
  sorry

end decreasing_geometric_sums_implications_l166_16693


namespace part1_part2_l166_16616

-- Definitions and conditions
def total_length : ℝ := 64
def ratio_larger_square_area : ℝ := 2.25
def total_area : ℝ := 160

-- Given problem parts
theorem part1 (x : ℝ) (h : (64 - 4 * x) / 4 * (64 - 4 * x) / 4 = 2.25 * x * x) : x = 6.4 :=
by
  -- Proof needs to be provided
  sorry

theorem part2 (y : ℝ) (h : (16 - y) * (16 - y) + y * y = 160) : y = 4 ∧ (64 - 4 * y) = 48 :=
by
  -- Proof needs to be provided
  sorry

end part1_part2_l166_16616


namespace simplify_4sqrt2_minus_sqrt2_l166_16661

/-- Prove that 4 * sqrt 2 - sqrt 2 = 3 * sqrt 2 given standard mathematical rules -/
theorem simplify_4sqrt2_minus_sqrt2 : 4 * Real.sqrt 2 - Real.sqrt 2 = 3 * Real.sqrt 2 :=
sorry

end simplify_4sqrt2_minus_sqrt2_l166_16661


namespace incorrect_proposition_statement_l166_16674

theorem incorrect_proposition_statement (p q : Prop) : (p ∨ q) → ¬ (p ∧ q) := 
sorry

end incorrect_proposition_statement_l166_16674


namespace total_time_to_virgo_l166_16649

def train_ride : ℝ := 5
def first_layover : ℝ := 1.5
def bus_ride : ℝ := 4
def second_layover : ℝ := 0.5
def first_flight : ℝ := 6
def third_layover : ℝ := 2
def second_flight : ℝ := 3 * bus_ride
def fourth_layover : ℝ := 3
def car_drive : ℝ := 3.5
def first_boat_ride : ℝ := 1.5
def fifth_layover : ℝ := 0.75
def second_boat_ride : ℝ := 2 * first_boat_ride - 0.5
def final_walk : ℝ := 1.25

def total_time : ℝ := train_ride + first_layover + bus_ride + second_layover + first_flight + third_layover + second_flight + fourth_layover + car_drive + first_boat_ride + fifth_layover + second_boat_ride + final_walk

theorem total_time_to_virgo : total_time = 44 := by
  simp [train_ride, first_layover, bus_ride, second_layover, first_flight, third_layover, second_flight, fourth_layover, car_drive, first_boat_ride, fifth_layover, second_boat_ride, final_walk, total_time]
  sorry

end total_time_to_virgo_l166_16649


namespace correct_phrase_l166_16663

-- Define statements representing each option
def option_A : String := "as twice much"
def option_B : String := "much as twice"
def option_C : String := "twice as much"
def option_D : String := "as much twice"

-- The correct option
def correct_option : String := "twice as much"

-- The main theorem statement
theorem correct_phrase : option_C = correct_option :=
by
  sorry

end correct_phrase_l166_16663


namespace square_side_length_on_hexagon_l166_16688

noncomputable def side_length_of_square (s : ℝ) : Prop :=
  let hexagon_side := 1
  let internal_angle := 120
  ((s * (1 + 1 / Real.sqrt 3)) = 2) → s = (3 - Real.sqrt 3)

theorem square_side_length_on_hexagon : ∃ s : ℝ, side_length_of_square s :=
by
  use 3 - Real.sqrt 3
  -- Proof to be provided
  sorry

end square_side_length_on_hexagon_l166_16688


namespace gumballs_remaining_l166_16685

theorem gumballs_remaining (a b total eaten remaining : ℕ) 
  (hAlicia : a = 20) 
  (hPedro : b = a + 3 * a) 
  (hTotal : total = a + b) 
  (hEaten : eaten = 40 * total / 100) 
  (hRemaining : remaining = total - eaten) : 
  remaining = 60 := by
  sorry

end gumballs_remaining_l166_16685


namespace function_is_zero_l166_16639

-- Define the condition that for any three points A, B, and C forming an equilateral triangle,
-- the sum of their function values is zero.
def has_equilateral_property (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ (A B C : ℝ × ℝ), dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1 → 
  f A + f B + f C = 0

-- Define the theorem that states that a function with the equilateral property is identically zero.
theorem function_is_zero {f : ℝ × ℝ → ℝ} (h : has_equilateral_property f) : 
  ∀ (x : ℝ × ℝ), f x = 0 := 
by
  sorry

end function_is_zero_l166_16639


namespace greatest_possible_x_l166_16682

-- Define the conditions and the target proof in Lean 4
theorem greatest_possible_x 
  (x : ℤ)  -- x is an integer
  (h : 2.134 * (10:ℝ)^x < 21000) : 
  x ≤ 3 :=
sorry

end greatest_possible_x_l166_16682


namespace greatest_possible_value_of_a_l166_16611

theorem greatest_possible_value_of_a (a : ℤ) (h1 : ∃ x : ℤ, x^2 + a*x = -30) (h2 : 0 < a) :
  a ≤ 31 :=
sorry

end greatest_possible_value_of_a_l166_16611


namespace age_of_b_l166_16607

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 27) : b = 10 := by
  sorry

end age_of_b_l166_16607


namespace triangle_angle_contradiction_l166_16672

theorem triangle_angle_contradiction (A B C : ℝ) (h_sum : A + B + C = 180) (h_lt_60 : A < 60 ∧ B < 60 ∧ C < 60) : false := 
sorry

end triangle_angle_contradiction_l166_16672


namespace find_period_l166_16622

-- Definitions based on conditions
def interest_rate_A : ℝ := 0.10
def interest_rate_C : ℝ := 0.115
def principal : ℝ := 4000
def total_gain : ℝ := 180

-- The question to prove
theorem find_period (n : ℝ) : 
  n = 3 :=
by 
  have interest_to_A := interest_rate_A * principal
  have interest_from_C := interest_rate_C * principal
  have annual_gain := interest_from_C - interest_to_A
  have equation := total_gain = annual_gain * n
  sorry

end find_period_l166_16622


namespace distinct_numbers_in_list_l166_16653

def count_distinct_floors (l : List ℕ) : ℕ :=
  l.eraseDups.length

def generate_list : List ℕ :=
  List.map (λ n => Nat.floor ((n * n : ℚ) / 2000)) (List.range' 1 2000)

theorem distinct_numbers_in_list : count_distinct_floors generate_list = 1501 :=
by
  sorry

end distinct_numbers_in_list_l166_16653


namespace tom_final_amount_l166_16643

-- Conditions and definitions from the problem
def initial_amount : ℝ := 74
def spent_percentage : ℝ := 0.15
def earnings : ℝ := 86
def share_percentage : ℝ := 0.60

-- Lean proof statement
theorem tom_final_amount :
  (initial_amount - (spent_percentage * initial_amount)) + (share_percentage * earnings) = 114.5 :=
by
  sorry

end tom_final_amount_l166_16643


namespace solution1_solution2_l166_16676

-- Definition for problem (1)
def problem1 : ℚ :=
  - (1 ^ 4 : ℚ) - (1 / 6) * (2 - (-3 : ℚ) ^ 2) / (-7 : ℚ)

theorem solution1 : problem1 = -7 / 6 :=
by
  sorry

-- Definition for problem (2)
def problem2 : ℚ :=
  ((3 / 2 : ℚ) - (5 / 8) + (7 / 12)) / (-1 / 24) - 8 * ((-1 / 2 : ℚ) ^ 3)

theorem solution2 : problem2 = -34 :=
by
  sorry

end solution1_solution2_l166_16676


namespace parallel_condition_perpendicular_condition_l166_16601

theorem parallel_condition (x : ℝ) (a b : ℝ × ℝ) :
  (a = (x, x + 2)) → (b = (1, 2)) → a.1 * b.2 = a.2 * b.1 → x = 2 := 
sorry

theorem perpendicular_condition (x : ℝ) (a b : ℝ × ℝ) :
  (a = (x, x + 2)) → (b = (1, 2)) → ((a.1 - b.1) * b.1 + (a.2 - b.2) * b.2) = 0 → x = 1 / 3 :=
sorry

end parallel_condition_perpendicular_condition_l166_16601


namespace full_price_tickets_revenue_l166_16659

-- Define the conditions and then prove the statement
theorem full_price_tickets_revenue (f d p : ℕ) (h1 : f + d = 200) (h2 : f * p + d * (p / 3) = 3000) : f * p = 1500 := by
  sorry

end full_price_tickets_revenue_l166_16659


namespace inscribed_circle_radius_l166_16657

theorem inscribed_circle_radius (R : ℝ) (h : 0 < R) : 
  ∃ x : ℝ, (x = R / 3) :=
by
  -- Given conditions
  have h1 : R > 0 := h

  -- Mathematical proof statement derived from conditions
  sorry

end inscribed_circle_radius_l166_16657


namespace inequality_px_qy_l166_16636

theorem inequality_px_qy 
  (p q x y : ℝ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (hpq : p + q < 1) 
  : (p * x + q * y) ^ 2 ≤ p * x ^ 2 + q * y ^ 2 := 
sorry

end inequality_px_qy_l166_16636


namespace original_number_is_115_l166_16658

-- Define the original number N, the least number to be subtracted (given), and the divisor
variable (N : ℤ) (k : ℤ)

-- State the condition based on the problem's requirements
def least_number_condition := ∃ k : ℤ, N - 28 = 87 * k

-- State the proof problem: Given the condition, prove the original number
theorem original_number_is_115 (h : least_number_condition N) : N = 115 := 
by
  sorry

end original_number_is_115_l166_16658


namespace ratio_of_square_areas_l166_16666

theorem ratio_of_square_areas (y : ℝ) (hy : y > 0) : 
  (y^2 / (3 * y)^2) = 1 / 9 :=
sorry

end ratio_of_square_areas_l166_16666


namespace interval_between_births_l166_16644

def youngest_child_age : ℕ := 6

def sum_of_ages (I : ℝ) : ℝ :=
  youngest_child_age + (youngest_child_age + I) + (youngest_child_age + 2 * I) + (youngest_child_age + 3 * I) + (youngest_child_age + 4 * I)

theorem interval_between_births : ∃ (I : ℝ), sum_of_ages I = 60 ∧ I = 3.6 := 
by
  sorry

end interval_between_births_l166_16644


namespace calc_1_calc_2_l166_16626

-- Question 1
theorem calc_1 : (5 / 17 * -4 - 5 / 17 * 15 + -5 / 17 * -2) = -5 :=
by sorry

-- Question 2
theorem calc_2 : (-1^2 + 36 / ((-3)^2) - ((-3 + 3 / 7) * (-7 / 24))) = 2 :=
by sorry

end calc_1_calc_2_l166_16626


namespace problem_inequality_1_problem_inequality_2_l166_16641

theorem problem_inequality_1 (x : ℝ) (α : ℝ) (hx : x > -1) (hα : 0 < α ∧ α < 1) : 
  (1 + x) ^ α ≤ 1 + α * x :=
sorry

theorem problem_inequality_2 (x : ℝ) (α : ℝ) (hx : x > -1) (hα : α < 0 ∨ α > 1) : 
  (1 + x) ^ α ≥ 1 + α * x :=
sorry

end problem_inequality_1_problem_inequality_2_l166_16641


namespace right_triangle_hypotenuse_len_l166_16615

theorem right_triangle_hypotenuse_len (a b : ℕ) (c : ℝ) (h₁ : a = 1) (h₂ : b = 3) 
  (h₃ : a^2 + b^2 = c^2) : c = Real.sqrt 10 := by
  sorry

end right_triangle_hypotenuse_len_l166_16615


namespace equation_of_line_l_l166_16670

theorem equation_of_line_l
  (a : ℝ)
  (l_intersects_circle : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + a = 0)
  (midpoint_chord : ∃ C : ℝ × ℝ, C = (-2, 3) ∧ ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + B.1) / 2 = C.1 ∧ (A.2 + B.2) / 2 = C.2) :
  a < 3 →
  ∃ l : ℝ × ℝ → Prop, (∀ x y : ℝ, l (x, y) ↔ x - y + 5 = 0) :=
by {
  sorry
}

end equation_of_line_l_l166_16670


namespace proposition_true_iff_l166_16645

theorem proposition_true_iff :
  (∀ x y : ℝ, (xy = 1 → x = 1 / y ∧ y = 1 / x) → (x = 1 / y ∧ y = 1 / x → xy = 1)) ∧
  (∀ (A B : Set ℝ), (A ∩ B = B → A ⊆ B) → (A ⊆ B → A ∩ B = B)) ∧
  (∀ m : ℝ, (m > 1 → ∃ x : ℝ, x^2 - 2 * x + m = 0) → (¬(∃ x : ℝ, x^2 - 2 * x + m = 0) → m ≤ 1)) :=
by
  sorry

end proposition_true_iff_l166_16645


namespace range_of_a_l166_16631

noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x

theorem range_of_a (a : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → a < g x) : a < 0 := 
by sorry

end range_of_a_l166_16631


namespace all_buses_have_same_stoppage_time_l166_16669

-- Define the constants for speeds without and with stoppages
def speed_without_stoppage_bus1 := 50
def speed_without_stoppage_bus2 := 60
def speed_without_stoppage_bus3 := 70

def speed_with_stoppage_bus1 := 40
def speed_with_stoppage_bus2 := 48
def speed_with_stoppage_bus3 := 56

-- Stating the stoppage time per hour for each bus
def stoppage_time_per_hour (speed_without : ℕ) (speed_with : ℕ) : ℚ :=
  1 - (speed_with : ℚ) / (speed_without : ℚ)

-- Theorem to prove the stoppage time correctness
theorem all_buses_have_same_stoppage_time :
  stoppage_time_per_hour speed_without_stoppage_bus1 speed_with_stoppage_bus1 = 0.2 ∧
  stoppage_time_per_hour speed_without_stoppage_bus2 speed_with_stoppage_bus2 = 0.2 ∧
  stoppage_time_per_hour speed_without_stoppage_bus3 speed_with_stoppage_bus3 = 0.2 :=
by
  sorry  -- Proof to be completed

end all_buses_have_same_stoppage_time_l166_16669


namespace find_a_and_b_l166_16627

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 6 * a * x^2 + b

theorem find_a_and_b :
  (∃ a b : ℝ, a ≠ 0 ∧
   (∀ x, -1 ≤ x ∧ x ≤ 2 → f a b x ≤ 3) ∧
   (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f a b x = 3) ∧
   (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f a b x = -29)
  ) → ((a = 2 ∧ b = 3) ∨ (a = -2 ∧ b = -29)) :=
sorry

end find_a_and_b_l166_16627


namespace sasha_total_items_l166_16609

/-
  Sasha bought pencils at 13 rubles each and pens at 20 rubles each,
  paying a total of 350 rubles. 
  Prove that the total number of pencils and pens Sasha bought is 23.
-/
theorem sasha_total_items
  (x y : ℕ) -- Define x as the number of pencils and y as the number of pens
  (H: 13 * x + 20 * y = 350) -- Given total cost condition
  : x + y = 23 := 
sorry

end sasha_total_items_l166_16609


namespace trigonometric_identity_l166_16623

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 2 * Real.cos α) :
  Real.sin (π / 2 + 2 * α) = -3 / 5 :=
by
  sorry

end trigonometric_identity_l166_16623


namespace smallest_five_digit_int_equiv_5_mod_9_l166_16605

theorem smallest_five_digit_int_equiv_5_mod_9 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 9 = 5 ∧ ∀ m : ℕ, (10000 ≤ m ∧ m < 100000 ∧ m % 9 = 5) → n ≤ m :=
by
  use 10000
  sorry

end smallest_five_digit_int_equiv_5_mod_9_l166_16605


namespace school_student_count_l166_16691

-- Definition of the conditions
def students_in_school (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 1 ∧
  n % 8 = 2 ∧
  n % 9 = 3

-- The main proof statement
theorem school_student_count : ∃ n, students_in_school n ∧ n = 265 :=
by
  sorry  -- Proof would go here

end school_student_count_l166_16691


namespace max_area_height_l166_16654

theorem max_area_height (h : ℝ) (x : ℝ) 
  (right_trapezoid : True) 
  (angle_30_deg : True) 
  (perimeter_eq_6 : 3 * (x + h) = 6) : 
  h = 1 :=
by 
  sorry

end max_area_height_l166_16654


namespace correct_average_weight_l166_16699

noncomputable def initial_average_weight : ℚ := 58.4
noncomputable def num_boys : ℕ := 20
noncomputable def misread_weight : ℚ := 56
noncomputable def correct_weight : ℚ := 66

theorem correct_average_weight : 
  (initial_average_weight * num_boys + (correct_weight - misread_weight)) / num_boys = 58.9 := 
by
  sorry

end correct_average_weight_l166_16699


namespace simplify_expression_l166_16655

variable (x : ℝ)

theorem simplify_expression :
  2 * x - 3 * (2 - x) + 4 * (1 + 3 * x) - 5 * (1 - x^2) = -5 * x^2 + 17 * x - 7 :=
by
  sorry

end simplify_expression_l166_16655


namespace original_people_in_room_l166_16679

theorem original_people_in_room (x : ℕ) 
  (h1 : 3 * x / 4 - 3 * x / 20 = 16) : x = 27 :=
sorry

end original_people_in_room_l166_16679


namespace first_day_bacteria_exceeds_200_l166_16606

noncomputable def bacteria_growth (n : ℕ) : ℕ := 2 * 3^n

theorem first_day_bacteria_exceeds_200 : ∃ n : ℕ, 2 * 3^n > 200 ∧ ∀ m : ℕ, m < n → 2 * 3^m ≤ 200 :=
by
  -- sorry for skipping proof
  sorry

end first_day_bacteria_exceeds_200_l166_16606


namespace f_satisfies_condition_l166_16664

noncomputable def f (x : ℝ) : ℝ := 2^x

-- Prove that f(x + 1) = 2 * f(x) for the defined function f.
theorem f_satisfies_condition (x : ℝ) : f (x + 1) = 2 * f x := by
  show 2^(x + 1) = 2 * 2^x
  sorry

end f_satisfies_condition_l166_16664


namespace n_minus_one_divides_n_squared_plus_n_sub_two_l166_16604

theorem n_minus_one_divides_n_squared_plus_n_sub_two (n : ℕ) : (n - 1) ∣ (n ^ 2 + n - 2) :=
sorry

end n_minus_one_divides_n_squared_plus_n_sub_two_l166_16604


namespace percent_equivalence_l166_16677

variable (x : ℝ)
axiom condition : 0.30 * 0.15 * x = 18

theorem percent_equivalence :
  0.15 * 0.30 * x = 18 := sorry

end percent_equivalence_l166_16677


namespace simplify_and_evaluate_l166_16635

-- Define the expression
def expression (x : ℝ) := -(2 * x^2 + 3 * x) + 2 * (4 * x + x^2)

-- State the theorem
theorem simplify_and_evaluate : expression (-2) = -10 :=
by
  -- The proof goes here
  sorry

end simplify_and_evaluate_l166_16635


namespace smallest_number_l166_16665

theorem smallest_number (b : ℕ) :
  (b % 3 = 2) ∧ (b % 4 = 2) ∧ (b % 5 = 3) → b = 38 :=
by
  sorry

end smallest_number_l166_16665


namespace number_of_foxes_l166_16633

-- Define the conditions as given in the problem
def num_cows : ℕ := 20
def num_sheep : ℕ := 20
def total_animals : ℕ := 100
def num_zebras (F : ℕ) := 3 * F

-- The theorem we want to prove based on the conditions
theorem number_of_foxes (F : ℕ) :
  num_cows + num_sheep + F + num_zebras F = total_animals → F = 15 :=
by
  sorry

end number_of_foxes_l166_16633


namespace shadow_area_l166_16671

theorem shadow_area (y : ℝ) (cube_side : ℝ) (shadow_excl_area : ℝ) 
  (h₁ : cube_side = 2) 
  (h₂ : shadow_excl_area = 200)
  (h₃ : ((14.28 - 2) / 2 = y)) :
  ⌊1000 * y⌋ = 6140 :=
by
  sorry

end shadow_area_l166_16671


namespace probability_uniform_same_color_l166_16648

noncomputable def probability_same_color (choices : List String) (athleteA: ℕ) (athleteB: ℕ) : ℚ :=
  if choices.length = 3 ∧ athleteA ∈ [0,1,2] ∧ athleteB ∈ [0,1,2] then
    1 / 3
  else
    0

theorem probability_uniform_same_color :
  probability_same_color ["red", "white", "blue"] 0 1 = 1 / 3 :=
by
  sorry

end probability_uniform_same_color_l166_16648


namespace minimum_value_expression_l166_16630

open Real

theorem minimum_value_expression (α β : ℝ) :
  ∃ x y : ℝ, x = 3 * cos α + 4 * sin β ∧ y = 3 * sin α + 4 * cos β ∧
    ((x - 7) ^ 2 + (y - 12) ^ 2) = 242 - 14 * sqrt 193 :=
sorry

end minimum_value_expression_l166_16630


namespace second_concert_attendance_l166_16620

theorem second_concert_attendance (n1 : ℕ) (h1 : n1 = 65899) (h2 : n2 = n1 + 119) : n2 = 66018 :=
by
  -- proof goes here
  sorry

end second_concert_attendance_l166_16620


namespace total_pictures_uploaded_is_65_l166_16613

-- Given conditions
def first_album_pics : ℕ := 17
def album_pics : ℕ := 8
def number_of_albums : ℕ := 6

-- The theorem to be proved
theorem total_pictures_uploaded_is_65 : first_album_pics + number_of_albums * album_pics = 65 :=
by
  sorry

end total_pictures_uploaded_is_65_l166_16613


namespace harry_sandy_midpoint_l166_16697

theorem harry_sandy_midpoint :
  ∃ (x y : ℤ), x = 9 ∧ y = -2 → ∃ (a b : ℤ), a = 1 ∧ b = 6 → ((9 + 1) / 2, (-2 + 6) / 2) = (5, 2) := 
by 
  sorry

end harry_sandy_midpoint_l166_16697


namespace total_volume_of_five_cubes_l166_16683

theorem total_volume_of_five_cubes (edge_length : ℕ) (n : ℕ) (volume_per_cube : ℕ) (total_volume : ℕ) 
  (h1 : edge_length = 5)
  (h2 : n = 5)
  (h3 : volume_per_cube = edge_length ^ 3)
  (h4 : total_volume = n * volume_per_cube) :
  total_volume = 625 :=
sorry

end total_volume_of_five_cubes_l166_16683


namespace faster_train_speed_l166_16678

theorem faster_train_speed (V_s : ℝ) (t : ℝ) (l : ℝ) (V_f : ℝ) : 
  V_s = 36 → t = 20 → l = 200 → V_f = V_s + (l / t) * 3.6 → V_f = 72 
  := by
    intros _ _ _ _
    sorry

end faster_train_speed_l166_16678


namespace quilt_shaded_fraction_l166_16686

theorem quilt_shaded_fraction :
  let original_squares := 9
  let shaded_column_squares := 3
  let fraction_shaded := shaded_column_squares / original_squares 
  fraction_shaded = 1/3 :=
by
  sorry

end quilt_shaded_fraction_l166_16686


namespace electric_energy_consumption_l166_16675

def power_rating_fan : ℕ := 75
def hours_per_day : ℕ := 8
def days_per_month : ℕ := 30
def watts_to_kWh : ℕ := 1000

theorem electric_energy_consumption : power_rating_fan * hours_per_day * days_per_month / watts_to_kWh = 18 := by
  sorry

end electric_energy_consumption_l166_16675


namespace negation_of_universal_proposition_l166_16667

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, |x - 1| - |x + 1| ≤ 3) ↔ ∃ x : ℝ, |x - 1| - |x + 1| > 3 :=
by
  sorry

end negation_of_universal_proposition_l166_16667


namespace largest_angle_l166_16608

-- Definitions for our conditions
def right_angle : ℝ := 90
def sum_of_two_angles (a b : ℝ) : Prop := a + b = (4 / 3) * right_angle
def angle_difference (a b : ℝ) : Prop := b = a + 40

-- Statement of the problem to be proved
theorem largest_angle (a b c : ℝ) (h_sum : sum_of_two_angles a b) (h_diff : angle_difference a b) (h_triangle : a + b + c = 180) : c = 80 :=
by sorry

end largest_angle_l166_16608


namespace longest_segment_in_cylinder_l166_16650

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  ∃ l, l = 10 * Real.sqrt 2 ∧
  (∀ x y z, (x = r * 2) ∧ (y = h) → z = Real.sqrt (x^2 + y^2) → z ≤ l) :=
sorry

end longest_segment_in_cylinder_l166_16650


namespace correct_number_of_statements_l166_16624

theorem correct_number_of_statements (a b : ℤ) :
  (¬ (∃ h₁ : Even (a + 5 * b), ¬ Even (a - 7 * b)) ∧
   ∃ h₂ : a + b % 3 = 0, ¬ ((a % 3 = 0) ∧ (b % 3 = 0)) ∧
   ∃ h₃ : Prime (a + b), Prime (a - b)) →
   1 = 1 :=
by
  sorry

end correct_number_of_statements_l166_16624


namespace solution_set_of_inequality_l166_16684

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2 * Real.exp (x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set_of_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | Real.sqrt 10 < x} :=
by
  sorry

end solution_set_of_inequality_l166_16684


namespace max_collection_l166_16698

theorem max_collection : 
  let Yoongi := 4 
  let Jungkook := 6 / 3 
  let Yuna := 5 
  max Yoongi (max Jungkook Yuna) = 5 :=
by 
  let Yoongi := 4
  let Jungkook := (6 / 3) 
  let Yuna := 5
  show max Yoongi (max Jungkook Yuna) = 5
  sorry

end max_collection_l166_16698


namespace woman_lawyer_probability_l166_16632

-- Defining conditions
def total_members : ℝ := 100
def percent_women : ℝ := 0.90
def percent_women_lawyers : ℝ := 0.60

-- Calculating numbers based on the percentages
def number_women : ℝ := percent_women * total_members
def number_women_lawyers : ℝ := percent_women_lawyers * number_women

-- Statement of the problem in Lean 4
theorem woman_lawyer_probability :
  (number_women_lawyers / total_members) = 0.54 :=
by sorry

end woman_lawyer_probability_l166_16632


namespace find_d_squared_plus_e_squared_l166_16647

theorem find_d_squared_plus_e_squared {a b c d e : ℕ} 
  (h1 : (a + 1) * (3 * b * c + 1) = d + 3 * e + 1)
  (h2 : (b + 1) * (3 * c * a + 1) = 3 * d + e + 13)
  (h3 : (c + 1) * (3 * a * b + 1) = 4 * (26 - d - e) - 1)
  : d ^ 2 + e ^ 2 = 146 := 
sorry

end find_d_squared_plus_e_squared_l166_16647


namespace total_elephants_l166_16600

-- Definitions and Hypotheses
def W : ℕ := 70
def G : ℕ := 3 * W

-- Proposition
theorem total_elephants : W + G = 280 := by
  sorry

end total_elephants_l166_16600


namespace parabola_coordinates_and_area_l166_16692

theorem parabola_coordinates_and_area
  (A B C : ℝ × ℝ)
  (hA : A = (2, 0))
  (hB : B = (3, 0))
  (hC : C = (5 / 2, 1 / 4))
  (h_vertex : ∀ x y, y = -x^2 + 5 * x - 6 → 
                   ((x, y) = A ∨ (x, y) = B ∨ (x, y) = C)) :
  A = (2, 0) ∧ B = (3, 0) ∧ C = (5 / 2, 1 / 4)
  ∧ (1 / 2 * (3 - 2) * (1 / 4) = 1 / 8) := 
by
  sorry

end parabola_coordinates_and_area_l166_16692


namespace min_log_value_l166_16619

theorem min_log_value (x y : ℝ) (h : 2 * x + 3 * y = 3) : ∃ (z : ℝ), z = Real.log (2^(4 * x) + 2^(3 * y)) / Real.log 2 ∧ z = 5 / 2 := 
by
  sorry

end min_log_value_l166_16619


namespace natural_numbers_pq_equal_l166_16673

theorem natural_numbers_pq_equal (p q : ℕ) (h : p^p + q^q = p^q + q^p) : p = q :=
sorry

end natural_numbers_pq_equal_l166_16673


namespace infinite_series_sum_l166_16660

noncomputable def S : ℝ :=
∑' n, (if n % 3 == 0 then 1 / (3 ^ (n / 3)) else if n % 3 == 1 then -1 / (3 ^ (n / 3 + 1)) else -1 / (3 ^ (n / 3 + 2)))

theorem infinite_series_sum : S = 15 / 26 := by
  sorry

end infinite_series_sum_l166_16660


namespace afternoon_sales_l166_16640

theorem afternoon_sales :
  ∀ (morning_sold afternoon_sold total_sold : ℕ),
    afternoon_sold = 2 * morning_sold ∧
    total_sold = morning_sold + afternoon_sold ∧
    total_sold = 510 →
    afternoon_sold = 340 :=
by
  intros morning_sold afternoon_sold total_sold h
  sorry

end afternoon_sales_l166_16640


namespace find_expression_roots_l166_16694

-- Define the roots of the given quadratic equation
def is_root (α : ℝ) : Prop := α ^ 2 - 2 * α - 1 = 0

-- Define the main statement to be proven
theorem find_expression_roots (α β : ℝ) (hα : is_root α) (hβ : is_root β) :
  5 * α ^ 4 + 12 * β ^ 3 = 169 := sorry

end find_expression_roots_l166_16694


namespace factorial_expression_l166_16656

namespace FactorialProblem

-- Definition of factorial function.
def factorial : ℕ → ℕ 
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Theorem stating the problem equivalently.
theorem factorial_expression : (factorial 12 - factorial 10) / factorial 8 = 11790 := by
  sorry

end FactorialProblem

end factorial_expression_l166_16656


namespace num_triples_l166_16696

/-- Theorem statement:
There are exactly 2 triples of positive integers (a, b, c) satisfying the conditions:
1. ab + ac = 60
2. bc + ac = 36
3. ab + bc = 48
--/
theorem num_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (ab + ac = 60) → (bc + ac = 36) → (ab + bc = 48) → 
  (a, b, c) ∈ [(1, 4, 8), (1, 12, 3)] →
  ∃! (a b c : ℕ), (ab + ac = 60) ∧ (bc + ac = 36) ∧ (ab + bc = 48) :=
sorry

end num_triples_l166_16696


namespace diagonal_length_of_cuboid_l166_16651

theorem diagonal_length_of_cuboid
  (a b c : ℝ)
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : c * a = Real.sqrt 6) : 
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 6 := 
sorry

end diagonal_length_of_cuboid_l166_16651


namespace total_visitors_600_l166_16612

variable (Enjoyed Understood : Set ℕ)
variable (TotalVisitors : ℕ)
variable (E U : ℕ)

axiom no_enjoy_no_understand : ∀ v, v ∉ Enjoyed → v ∉ Understood
axiom equal_enjoy_understand : E = U
axiom enjoy_and_understand_fraction : E = 3 / 4 * TotalVisitors
axiom total_visitors_equation : TotalVisitors = E + 150

theorem total_visitors_600 : TotalVisitors = 600 := by
  sorry

end total_visitors_600_l166_16612


namespace union_A_B_l166_16642

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def C : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_A_B : A ∪ B = C := 
by sorry

end union_A_B_l166_16642


namespace age_of_30th_employee_l166_16629

theorem age_of_30th_employee :
  let n := 30
  let group1_avg_age := 24
  let group1_count := 10
  let group2_avg_age := 30
  let group2_count := 12
  let group3_avg_age := 35
  let group3_count := 7
  let remaining_avg_age := 29

  let group1_total_age := group1_count * group1_avg_age
  let group2_total_age := group2_count * group2_avg_age
  let group3_total_age := group3_count * group3_avg_age
  let total_age_29 := group1_total_age + group2_total_age + group3_total_age
  let total_age_30 := remaining_avg_age * n

  let age_30th_employee := total_age_30 - total_age_29

  age_30th_employee = 25 :=
by
  let n := 30
  let group1_avg_age := 24
  let group1_count := 10
  let group2_avg_age := 30
  let group2_count := 12
  let group3_avg_age := 35
  let group3_count := 7
  let remaining_avg_age := 29

  let group1_total_age := group1_count * group1_avg_age
  let group2_total_age := group2_count * group2_avg_age
  let group3_total_age := group3_count * group3_avg_age
  let total_age_29 := group1_total_age + group2_total_age + group3_total_age
  let total_age_30 := remaining_avg_age * n

  let age_30th_employee := total_age_30 - total_age_29

  have h : age_30th_employee = 25 := sorry
  exact h

end age_of_30th_employee_l166_16629


namespace total_storage_l166_16614

variable (barrels largeCasks smallCasks : ℕ)
variable (cap_barrel cap_largeCask cap_smallCask : ℕ)

-- Given conditions
axiom h1 : barrels = 4
axiom h2 : largeCasks = 3
axiom h3 : smallCasks = 5
axiom h4 : cap_largeCask = 20
axiom h5 : cap_smallCask = cap_largeCask / 2
axiom h6 : cap_barrel = 2 * cap_largeCask + 3

-- Target statement
theorem total_storage : 4 * cap_barrel + 3 * cap_largeCask + 5 * cap_smallCask = 282 := 
by
  -- Proof is not required
  sorry

end total_storage_l166_16614


namespace price_of_each_sundae_l166_16690

theorem price_of_each_sundae
  (num_ice_cream_bars : ℕ := 125) 
  (num_sundaes : ℕ := 125) 
  (total_price : ℝ := 225)
  (price_per_ice_cream_bar : ℝ := 0.60) :
  ∃ (price_per_sundae : ℝ), price_per_sundae = 1.20 := 
by
  -- Variables for costs of ice-cream bars and sundaes' total cost
  let cost_ice_cream_bars := num_ice_cream_bars * price_per_ice_cream_bar
  let total_cost_sundaes := total_price - cost_ice_cream_bars
  let price_per_sundae := total_cost_sundaes / num_sundaes
  use price_per_sundae
  sorry

end price_of_each_sundae_l166_16690


namespace find_number_l166_16646

theorem find_number (x : ℝ) : 
  (x + 72 = (2 * x) / (2 / 3)) → x = 36 :=
by
  intro h
  sorry

end find_number_l166_16646


namespace year_population_below_five_percent_l166_16687

def population (P0 : ℕ) (years : ℕ) : ℕ :=
  P0 / 2^years

theorem year_population_below_five_percent (P0 : ℕ) :
  ∃ n, population P0 n < P0 / 20 ∧ (2005 + n) = 2010 := 
by {
  sorry
}

end year_population_below_five_percent_l166_16687


namespace negation_one_zero_l166_16695

theorem negation_one_zero (a b : ℝ) (h : a ≠ 0):
  ¬ (∃! x : ℝ, a * x + b = 0) ↔ (¬ ∃ x : ℝ, a * x + b = 0 ∨ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁ + b = 0 ∧ a * x₂ + b = 0) := by
sorry

end negation_one_zero_l166_16695


namespace number_told_to_sasha_l166_16603

-- Defining concepts
def two_digit_number (a b : ℕ) : Prop := a < 10 ∧ b < 10 ∧ a * b ≥ 1

def product_of_digits (a b : ℕ) (P : ℕ) : Prop := P = a * b

def sum_of_digits (a b : ℕ) (S : ℕ) : Prop := S = a + b

def petya_guesses_in_three_attempts (P : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), P = a * b ∧ P = c * d ∧ P = e * f ∧ 
  (a * b) ≠ (c * d) ∧ (a * b) ≠ (e * f) ∧ (c * d) ≠ (e * f)

def sasha_guesses_in_four_attempts (S : ℕ) : Prop :=
  ∃ (a b c d e f g h i j : ℕ), 
  S = a + b ∧ S = c + d ∧ S = e + f ∧ S = g + h ∧ S = i + j ∧
  (a + b) ≠ (c + d) ∧ (a + b) ≠ (e + f) ∧ (a + b) ≠ (g + h) ∧ (a + b) ≠ (i + j) ∧ 
  (c + d) ≠ (e + f) ∧ (c + d) ≠ (g + h) ∧ (c + d) ≠ (i + j) ∧ 
  (e + f) ≠ (g + h) ∧ (e + f) ≠ (i + j) ∧ 
  (g + h) ≠ (i + j)

theorem number_told_to_sasha : ∃ (S : ℕ), 
  ∀ (a b : ℕ), two_digit_number a b → 
  (product_of_digits a b (a * b) → petya_guesses_in_three_attempts (a * b)) → 
  (sum_of_digits a b S → sasha_guesses_in_four_attempts S) → S = 10 :=
by
  sorry

end number_told_to_sasha_l166_16603


namespace bottle_count_l166_16637

theorem bottle_count :
  ∃ N x : ℕ, 
    N = x^2 + 36 ∧ N = (x + 1)^2 + 3 :=
by 
  sorry

end bottle_count_l166_16637


namespace add_inequality_of_greater_l166_16621

theorem add_inequality_of_greater (a b c d : ℝ) (h₁ : a > b) (h₂ : c > d) : a + c > b + d := 
by sorry

end add_inequality_of_greater_l166_16621


namespace total_stars_correct_l166_16617

-- Define the number of gold stars Shelby earned each day
def monday_stars : ℕ := 4
def tuesday_stars : ℕ := 7
def wednesday_stars : ℕ := 3
def thursday_stars : ℕ := 8
def friday_stars : ℕ := 2

-- Define the total number of gold stars
def total_stars : ℕ := monday_stars + tuesday_stars + wednesday_stars + thursday_stars + friday_stars

-- Prove that the total number of gold stars Shelby earned throughout the week is 24
theorem total_stars_correct : total_stars = 24 :=
by
  -- The proof goes here, using sorry to skip the proof
  sorry

end total_stars_correct_l166_16617


namespace stopped_time_per_hour_A_stopped_time_per_hour_B_stopped_time_per_hour_C_l166_16638

-- Definition of the speeds
def speed_excluding_stoppages_A : ℕ := 60
def speed_including_stoppages_A : ℕ := 48
def speed_excluding_stoppages_B : ℕ := 75
def speed_including_stoppages_B : ℕ := 60
def speed_excluding_stoppages_C : ℕ := 90
def speed_including_stoppages_C : ℕ := 72

-- Theorem to prove the stopped time per hour for each bus
theorem stopped_time_per_hour_A : (speed_excluding_stoppages_A - speed_including_stoppages_A) * 60 / speed_excluding_stoppages_A = 12 := sorry

theorem stopped_time_per_hour_B : (speed_excluding_stoppages_B - speed_including_stoppages_B) * 60 / speed_excluding_stoppages_B = 12 := sorry

theorem stopped_time_per_hour_C : (speed_excluding_stoppages_C - speed_including_stoppages_C) * 60 / speed_excluding_stoppages_C = 12 := sorry

end stopped_time_per_hour_A_stopped_time_per_hour_B_stopped_time_per_hour_C_l166_16638


namespace units_digit_of_8_pow_120_l166_16628

theorem units_digit_of_8_pow_120 : (8 ^ 120) % 10 = 6 := 
by
  sorry

end units_digit_of_8_pow_120_l166_16628


namespace find_A_l166_16652

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) (h : 100 * A + 78 - (200 + B) = 364) : A = 5 :=
by
  sorry

end find_A_l166_16652


namespace part1_part2_part3_l166_16634

noncomputable def seq (a : ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else (1 - a) / n

theorem part1 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (a1_eq : seq a 1 = 1 / 2) (a2_eq : seq a 2 = 1 / 4) : true :=
by trivial

theorem part2 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (n : ℕ) (hn : n > 0) : 0 < seq a n ∧ seq a n < 1 :=
sorry

theorem part3 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (n : ℕ) (hn : n > 0) : seq a n > seq a (n + 1) :=
sorry

end part1_part2_part3_l166_16634


namespace no_function_f_l166_16602

noncomputable def g (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_function_f (a b c : ℝ) (h : ∀ x, g a b c (g a b c x) = x) :
  ¬ ∃ f : ℝ → ℝ, ∀ x, f (f x) = g a b c x := 
sorry

end no_function_f_l166_16602
