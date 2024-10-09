import Mathlib

namespace candy_distribution_l488_48879

theorem candy_distribution :
  let bags := 4
  let candies := 9
  (Nat.choose candies (candies - bags) * Nat.choose (candies - 1) (candies - bags - 1)) = 7056 :=
by
  -- define variables for bags and candies
  let bags := 4
  let candies := 9
  have h : (Nat.choose candies (candies - bags) * Nat.choose (candies - 1) (candies - bags - 1)) = 7056 := sorry
  exact h

end candy_distribution_l488_48879


namespace trader_sold_45_meters_l488_48897

-- Definitions based on conditions
def selling_price_total : ℕ := 4500
def profit_per_meter : ℕ := 12
def cost_price_per_meter : ℕ := 88
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof goal to show that the trader sold 45 meters of cloth
theorem trader_sold_45_meters : ∃ x : ℕ, selling_price_per_meter * x = selling_price_total ∧ x = 45 := 
by
  sorry

end trader_sold_45_meters_l488_48897


namespace grid_possible_configuration_l488_48894

theorem grid_possible_configuration (m n : ℕ) (hm : m > 100) (hn : n > 100) : 
  ∃ grid : ℕ → ℕ → ℕ,
  (∀ i j, grid i j = (if i > 0 then grid (i - 1) j else 0) + 
                       (if i < m - 1 then grid (i + 1) j else 0) + 
                       (if j > 0 then grid i (j - 1) else 0) + 
                       (if j < n - 1 then grid i (j + 1) else 0)) 
  ∧ (∃ i j, grid i j ≠ 0) 
  ∧ m > 14 
  ∧ n > 14 := 
sorry

end grid_possible_configuration_l488_48894


namespace slope_of_line_l488_48850

theorem slope_of_line : 
  let A := Real.sin (Real.pi / 6)
  let B := Real.cos (5 * Real.pi / 6)
  (- A / B) = Real.sqrt 3 / 3 :=
by
  sorry

end slope_of_line_l488_48850


namespace students_helped_on_fourth_day_l488_48870

theorem students_helped_on_fourth_day (total_books : ℕ) (books_per_student : ℕ)
  (day1_students : ℕ) (day2_students : ℕ) (day3_students : ℕ)
  (H1 : total_books = 120) (H2 : books_per_student = 5)
  (H3 : day1_students = 4) (H4 : day2_students = 5) (H5 : day3_students = 6) :
  (total_books - (day1_students * books_per_student + day2_students * books_per_student + day3_students * books_per_student)) / books_per_student = 9 :=
by
  sorry

end students_helped_on_fourth_day_l488_48870


namespace toothpicks_total_l488_48823

-- Definitions based on the conditions
def grid_length : ℕ := 50
def grid_width : ℕ := 40

-- Mathematical statement to prove
theorem toothpicks_total : (grid_length + 1) * grid_width + (grid_width + 1) * grid_length = 4090 := by
  sorry

end toothpicks_total_l488_48823


namespace carmen_reaches_alex_in_17_5_minutes_l488_48800

-- Define the conditions
variable (initial_distance : ℝ := 30) -- Initial distance in kilometers
variable (rate_of_closure : ℝ := 2) -- Rate at which the distance decreases in km per minute
variable (minutes_before_stop : ℝ := 10) -- Minutes before Alex stops

-- Define the speeds
variable (v_A : ℝ) -- Alex's speed in km per hour
variable (v_C : ℝ := 2 * v_A) -- Carmen's speed is twice Alex's speed
variable (total_closure_rate : ℝ := 120) -- Closure rate in km per hour (2 km per minute)

-- Main theorem to prove:
theorem carmen_reaches_alex_in_17_5_minutes : 
  ∃ (v_A v_C : ℝ), v_C = 2 * v_A ∧ v_C + v_A = total_closure_rate ∧ 
    (initial_distance - rate_of_closure * minutes_before_stop 
    - v_C * ((initial_distance - rate_of_closure * minutes_before_stop) / v_C) / 60 = 0) ∧ 
    (minutes_before_stop + ((initial_distance - rate_of_closure * minutes_before_stop) / v_C) * 60 = 17.5) :=
by
  sorry

end carmen_reaches_alex_in_17_5_minutes_l488_48800


namespace probability_both_groups_stop_same_round_l488_48839

noncomputable def probability_same_round : ℚ :=
  let probability_fair_coin_stop (n : ℕ) : ℚ := (1/2)^n
  let probability_biased_coin_stop (n : ℕ) : ℚ := (2/3)^(n-1) * (1/3)
  let probability_fair_coin_group_stop (n : ℕ) : ℚ := (probability_fair_coin_stop n)^3
  let probability_biased_coin_group_stop (n : ℕ) : ℚ := (probability_biased_coin_stop n)^3
  let combined_round_probability (n : ℕ) : ℚ := 
    probability_fair_coin_group_stop n * probability_biased_coin_group_stop n
  let total_probability : ℚ := ∑' n, combined_round_probability n
  total_probability

theorem probability_both_groups_stop_same_round :
  probability_same_round = 1 / 702 := by sorry

end probability_both_groups_stop_same_round_l488_48839


namespace total_number_of_students_l488_48861

theorem total_number_of_students (b h p s : ℕ) 
  (h1 : b = 30)
  (h2 : b = 2 * h)
  (h3 : p = h + 5)
  (h4 : s = 3 * p) :
  b + h + p + s = 125 :=
by sorry

end total_number_of_students_l488_48861


namespace only_prime_satisfying_condition_l488_48890

theorem only_prime_satisfying_condition (p : ℕ) (h_prime : Prime p) : (Prime (p^2 + 14) ↔ p = 3) := 
by
  sorry

end only_prime_satisfying_condition_l488_48890


namespace beta_interval_solution_l488_48820

/-- 
Prove that the values of β in the set {β | β = π/6 + 2*k*π, k ∈ ℤ} 
that satisfy the interval (-2*π, 2*π) are β = π/6 or β = -11*π/6.
-/
theorem beta_interval_solution :
  ∀ β : ℝ, (∃ k : ℤ, β = (π / 6) + 2 * k * π) → (-2 * π < β ∧ β < 2 * π) →
  (β = π / 6 ∨ β = -11 * π / 6) :=
by
  intros β h_exists h_interval
  sorry

end beta_interval_solution_l488_48820


namespace total_crosswalk_lines_l488_48835

theorem total_crosswalk_lines (n m l : ℕ) (h1 : n = 5) (h2 : m = 4) (h3 : l = 20) :
  n * (m * l) = 400 := by
  sorry

end total_crosswalk_lines_l488_48835


namespace exists_rectangle_in_inscribed_right_triangle_l488_48827

theorem exists_rectangle_in_inscribed_right_triangle :
  ∃ (L W : ℝ), 
    (45^2 / (1 + (5/2)^2) = L * L) ∧
    (2 * L = 45) ∧
    (2 * W = 45) ∧
    ((L = 25 ∧ W = 10) ∨ (L = 18.75 ∧ W = 7.5)) :=
by sorry

end exists_rectangle_in_inscribed_right_triangle_l488_48827


namespace seungho_more_marbles_l488_48887

variable (S H : ℕ)

-- Seungho gave 273 marbles to Hyukjin
def given_marbles : ℕ := 273

-- After giving 273 marbles, Seungho has 477 more marbles than Hyukjin
axiom marbles_condition : S - given_marbles = (H + given_marbles) + 477

theorem seungho_more_marbles (S H : ℕ) (marbles_condition : S - 273 = (H + 273) + 477) : S = H + 1023 :=
by
  sorry

end seungho_more_marbles_l488_48887


namespace perpendicular_lines_l488_48865

theorem perpendicular_lines (a : ℝ) :
  (a + 2) * (a - 1) + (1 - a) * (2 * a + 3) = 0 ↔ (a = 1 ∨ a = -1) := 
sorry

end perpendicular_lines_l488_48865


namespace trigonometric_identity_l488_48845

theorem trigonometric_identity 
  (α β : ℝ) 
  (h : α + β = π / 3)  -- Note: 60 degrees is π/3 radians
  (tan_add : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) 
  (tan_60 : Real.tan (π / 3) = Real.sqrt 3) :
  Real.tan α + Real.tan β + Real.sqrt 3 * Real.tan α * Real.tan β = Real.sqrt 3 :=
sorry

end trigonometric_identity_l488_48845


namespace symmetric_line_equation_l488_48810

def line_1 (x y : ℝ) : Prop := 2 * x - y + 3 = 0
def line_2 (x y : ℝ) : Prop := x - y + 2 = 0
def symmetric_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

theorem symmetric_line_equation :
  ∀ x y : ℝ, line_1 x y → line_2 x y → symmetric_line x y := 
sorry

end symmetric_line_equation_l488_48810


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l488_48863

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l488_48863


namespace intersecting_rectangles_area_l488_48884

-- Define the dimensions of the rectangles
def rect1_length : ℝ := 12
def rect1_width : ℝ := 4
def rect2_length : ℝ := 7
def rect2_width : ℝ := 5

-- Define the areas of the individual rectangles
def area_rect1 : ℝ := rect1_length * rect1_width
def area_rect2 : ℝ := rect2_length * rect2_width

-- Assume overlapping region area
def area_overlap : ℝ := rect1_width * rect2_width

-- Define the total shaded area
def shaded_area : ℝ := area_rect1 + area_rect2 - area_overlap

-- Prove the shaded area is 63 square units
theorem intersecting_rectangles_area : shaded_area = 63 :=
by 
  -- Insert proof steps here, we only provide the theorem statement and leave the proof unfinished
  sorry

end intersecting_rectangles_area_l488_48884


namespace zoe_correct_percentage_l488_48878

noncomputable def t : ℝ := sorry  -- total number of problems
noncomputable def chloe_alone_correct : ℝ := 0.70 * (1/3 * t)  -- Chloe's correct answers alone
noncomputable def chloe_total_correct : ℝ := 0.85 * t  -- Chloe's overall correct answers
noncomputable def together_correct : ℝ := chloe_total_correct - chloe_alone_correct  -- Problems solved correctly together
noncomputable def zoe_alone_correct : ℝ := 0.85 * (1/3 * t)  -- Zoe's correct answers alone
noncomputable def zoe_total_correct : ℝ := zoe_alone_correct + together_correct  -- Zoe's total correct answers
noncomputable def zoe_percentage_correct : ℝ := (zoe_total_correct / t) * 100  -- Convert to percentage

theorem zoe_correct_percentage : zoe_percentage_correct = 90 := 
by
  sorry

end zoe_correct_percentage_l488_48878


namespace smallest_n_ineq_l488_48816

theorem smallest_n_ineq : ∃ n : ℕ, 3 * Real.sqrt n - 2 * Real.sqrt (n - 1) < 0.03 ∧ 
  (∀ m : ℕ, (3 * Real.sqrt m - 2 * Real.sqrt (m - 1) < 0.03) → n ≤ m) ∧ n = 433715589 :=
by
  sorry

end smallest_n_ineq_l488_48816


namespace correct_calculation_l488_48855

variable (a b : ℝ)

theorem correct_calculation : (ab)^2 = a^2 * b^2 := by
  sorry

end correct_calculation_l488_48855


namespace solve_for_a4b4_l488_48832

theorem solve_for_a4b4 (
    a1 a2 a3 a4 b1 b2 b3 b4 : ℝ
) (h1 : a1 * b1 + a2 * b3 = 1) 
  (h2 : a1 * b2 + a2 * b4 = 0) 
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) : 
  a4 * b4 = -6 :=
sorry

end solve_for_a4b4_l488_48832


namespace value_of_c_l488_48872

theorem value_of_c (c : ℝ) : (∃ x : ℝ, x^2 + c * x - 36 = 0 ∧ x = -9) → c = 5 :=
by
  sorry

end value_of_c_l488_48872


namespace two_p_plus_q_l488_48868

variable {p q : ℚ}

theorem two_p_plus_q (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 :=
by
  sorry

end two_p_plus_q_l488_48868


namespace find_speed_of_boat_l488_48889

noncomputable def speed_of_boat_in_still_water 
  (v : ℝ) 
  (current_speed : ℝ := 8) 
  (distance : ℝ := 36.67) 
  (time_in_minutes : ℝ := 44) : Prop :=
  v = 42

theorem find_speed_of_boat 
  (v : ℝ)
  (current_speed : ℝ := 8) 
  (distance : ℝ := 36.67) 
  (time_in_minutes : ℝ := 44) 
  (h1 : v + current_speed = distance / (time_in_minutes / 60)) : 
  speed_of_boat_in_still_water v :=
by
  sorry

end find_speed_of_boat_l488_48889


namespace impossible_d_values_count_l488_48828

def triangle_rectangle_difference (d : ℕ) : Prop :=
  ∃ (l w : ℕ),
  l = 2 * w ∧
  6 * w > 0 ∧
  (6 * w + 2 * d) - 6 * w = 1236 ∧
  d > 0

theorem impossible_d_values_count : ∀ d : ℕ, d ≠ 618 → ¬triangle_rectangle_difference d :=
by
  sorry

end impossible_d_values_count_l488_48828


namespace max_y_difference_intersection_l488_48876

noncomputable def f (x : ℝ) : ℝ := 4 - x^2 + x^3
noncomputable def g (x : ℝ) : ℝ := 2 + x^2 + x^3

theorem max_y_difference_intersection :
  let x1 := 1
  let y1 := g x1
  let x2 := -1
  let y2 := g x2
  y1 - y2 = 2 :=
by
  sorry

end max_y_difference_intersection_l488_48876


namespace point_B_value_l488_48844

theorem point_B_value (A : ℝ) (B : ℝ) (hA : A = -5) (hB : B = -1 ∨ B = -9) :
  ∃ B : ℝ, (B = A + 4 ∨ B = A - 4) :=
by sorry

end point_B_value_l488_48844


namespace morning_snowfall_l488_48838

theorem morning_snowfall (total_snowfall afternoon_snowfall morning_snowfall : ℝ) 
  (h1 : total_snowfall = 0.625) 
  (h2 : afternoon_snowfall = 0.5) 
  (h3 : total_snowfall = morning_snowfall + afternoon_snowfall) : 
  morning_snowfall = 0.125 :=
by
  sorry

end morning_snowfall_l488_48838


namespace maximum_k_l488_48819

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

theorem maximum_k (x : ℝ) (h₀ : x > 0) (k : ℤ) (a := 1) (h₁ : (x - k) * f_prime x a + x + 1 > 0) : k = 2 :=
sorry

end maximum_k_l488_48819


namespace MN_length_correct_l488_48848

open Real

noncomputable def MN_segment_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ :=
  sqrt (a * b)

theorem MN_length_correct (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ (MN : ℝ), MN = MN_segment_length a b h1 h2 :=
by
  use sqrt (a * b)
  exact rfl

end MN_length_correct_l488_48848


namespace largest_n_l488_48843

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def valid_n (x y : ℕ) : Prop :=
  x < 10 ∧ y < 10 ∧ x ≠ y ∧ is_prime x ∧ is_prime y ∧ is_prime (10 * y + x) ∧
  100 ≤ x * y * (10 * y + x) ∧ x * y * (10 * y + x) < 1000

theorem largest_n : ∃ x y : ℕ, valid_n x y ∧ x * y * (10 * y + x) = 777 := by
  sorry

end largest_n_l488_48843


namespace hotel_charge_problem_l488_48812

theorem hotel_charge_problem (R G P : ℝ) 
  (h1 : P = 0.5 * R) 
  (h2 : P = 0.9 * G) : 
  (R - G) / G * 100 = 80 :=
by
  sorry

end hotel_charge_problem_l488_48812


namespace greatest_two_digit_with_product_9_l488_48834

theorem greatest_two_digit_with_product_9 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ a b : ℕ, n = 10 * a + b ∧ a * b = 9) ∧ (∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ (∃ c d : ℕ, m = 10 * c + d ∧ c * d = 9) → m ≤ 91) :=
by
  sorry

end greatest_two_digit_with_product_9_l488_48834


namespace solve_for_y_l488_48891

theorem solve_for_y {y : ℝ} : (y - 5)^4 = 16 → y = 7 :=
by
  sorry

end solve_for_y_l488_48891


namespace a4_eq_12_l488_48881

-- Definitions of the sequences and conditions
def S (n : ℕ) : ℕ := 
  -- sum of the first n terms, initially undefined
  sorry  

def a (n : ℕ) : ℕ := 
  -- terms of the sequence, initially undefined
  sorry  

-- Given conditions
axiom a2_eq_3 : a 2 = 3
axiom Sn_recurrence : ∀ n ≥ 2, S (n + 1) = 2 * S n

-- Statement to prove
theorem a4_eq_12 : a 4 = 12 :=
  sorry

end a4_eq_12_l488_48881


namespace johnny_red_pencils_l488_48829

noncomputable def number_of_red_pencils (packs_total : ℕ) (extra_packs : ℕ) (extra_per_pack : ℕ) : ℕ :=
  packs_total + extra_packs * extra_per_pack

theorem johnny_red_pencils : number_of_red_pencils 15 3 2 = 21 := by
  sorry

end johnny_red_pencils_l488_48829


namespace larger_number_is_34_l488_48898

theorem larger_number_is_34 (a b : ℕ) (h1 : a > b) (h2 : (a + b) + (a - b) = 68) : a = 34 := 
by
  sorry

end larger_number_is_34_l488_48898


namespace ab_plus_cd_l488_48806

variables (a b c d : ℝ)

axiom h1 : a + b + c = 1
axiom h2 : a + b + d = 6
axiom h3 : a + c + d = 15
axiom h4 : b + c + d = 10

theorem ab_plus_cd : a * b + c * d = 45.33333333333333 := 
by 
  sorry

end ab_plus_cd_l488_48806


namespace tracy_initial_candies_l488_48852

theorem tracy_initial_candies (x y : ℕ) (h₁ : x = 108) (h₂ : 2 ≤ y ∧ y ≤ 6) : 
  let remaining_after_eating := (3 / 4) * x 
  let remaining_after_giving := (2 / 3) * remaining_after_eating
  let remaining_after_mom := remaining_after_giving - 40
  remaining_after_mom - y = 10 :=
by 
  sorry

end tracy_initial_candies_l488_48852


namespace cost_price_percentage_l488_48862

-- Define the condition that the profit percent is 11.11111111111111%
def profit_percent (CP SP: ℝ) : Prop :=
  ((SP - CP) / CP) * 100 = 11.11111111111111

-- Prove that under this condition, the cost price (CP) is 90% of the selling price (SP).
theorem cost_price_percentage (CP SP : ℝ) (h: profit_percent CP SP) : (CP / SP) * 100 = 90 :=
sorry

end cost_price_percentage_l488_48862


namespace distance_equal_x_value_l488_48805

theorem distance_equal_x_value :
  (∀ P Q R : ℝ × ℝ × ℝ, P = (x, 2, 1) ∧ Q = (1, 1, 2) ∧ R = (2, 1, 1) →
  dist P Q = dist P R →
  x = 1) :=
by
  -- Define the points P, Q, R
  let P := (x, 2, 1)
  let Q := (1, 1, 2)
  let R := (2, 1, 1)

  -- Given the condition
  intro h
  sorry

end distance_equal_x_value_l488_48805


namespace three_digit_divisible_by_8_l488_48856

theorem three_digit_divisible_by_8 : ∃ n : ℕ, n / 100 = 5 ∧ n % 10 = 3 ∧ n % 8 = 0 :=
by
  use 533
  sorry

end three_digit_divisible_by_8_l488_48856


namespace inequality_always_holds_l488_48888

theorem inequality_always_holds (a b : ℝ) (h₀ : a < b) (h₁ : b < 0) : a^2 > ab ∧ ab > b^2 :=
by
  sorry

end inequality_always_holds_l488_48888


namespace range_of_a_l488_48882

-- Define sets A and B and the condition A ∩ B = ∅
def set_A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def set_B (a : ℝ) : Set ℝ := { x | x > a }

-- State the condition: A ∩ B = ∅ implies a ≥ 1
theorem range_of_a (a : ℝ) : (set_A ∩ set_B a = ∅) → a ≥ 1 :=
  by
  sorry

end range_of_a_l488_48882


namespace non_neg_int_solutions_l488_48801

theorem non_neg_int_solutions :
  (∀ m n k : ℕ, 2 * m + 3 * n = k ^ 2 →
    (m = 0 ∧ n = 1 ∧ k = 2) ∨
    (m = 3 ∧ n = 0 ∧ k = 3) ∨
    (m = 4 ∧ n = 2 ∧ k = 5)) :=
by
  intro m n k h
  -- outline proof steps here
  sorry

end non_neg_int_solutions_l488_48801


namespace average_typed_words_per_minute_l488_48804

def rudy_wpm := 64
def joyce_wpm := 76
def gladys_wpm := 91
def lisa_wpm := 80
def mike_wpm := 89
def num_team_members := 5

theorem average_typed_words_per_minute : 
  (rudy_wpm + joyce_wpm + gladys_wpm + lisa_wpm + mike_wpm) / num_team_members = 80 := 
by
  sorry

end average_typed_words_per_minute_l488_48804


namespace trig_identity_l488_48873

open Real

theorem trig_identity :
  3.4173 * sin (2 * pi / 17) + sin (4 * pi / 17) - sin (6 * pi / 17) - (1/2) * sin (8 * pi / 17) =
  8 * (sin (2 * pi / 17))^3 * (cos (pi / 17))^2 :=
by sorry

end trig_identity_l488_48873


namespace derivative_at_2_l488_48811

theorem derivative_at_2 (f : ℝ → ℝ) (h : ∀ x, f x = x^2 * deriv f 2 + 5 * x) :
    deriv f 2 = -5/3 :=
by
  sorry

end derivative_at_2_l488_48811


namespace kerosene_cost_l488_48815

/-- Given that:
    - A dozen eggs cost as much as a pound of rice.
    - A half-liter of kerosene costs as much as 8 eggs.
    - The cost of each pound of rice is $0.33.
    - One dollar has 100 cents.
Prove that a liter of kerosene costs 44 cents.
-/
theorem kerosene_cost :
  let egg_cost := 0.33 / 12  -- Cost per egg in dollars
  let kerosene_half_liter_cost := egg_cost * 8  -- Half-liter of kerosene cost in dollars
  let kerosene_liter_cost := kerosene_half_liter_cost * 2  -- Liter of kerosene cost in dollars
  let kerosene_liter_cost_cents := kerosene_liter_cost * 100  -- Liter of kerosene cost in cents
  kerosene_liter_cost_cents = 44 :=
by
  sorry

end kerosene_cost_l488_48815


namespace min_area_rectangle_l488_48808

theorem min_area_rectangle (l w : ℝ) 
  (hl : 3.5 ≤ l ∧ l ≤ 4.5) 
  (hw : 5.5 ≤ w ∧ w ≤ 6.5) 
  (constraint : l ≥ 2 * w) : 
  l * w = 60.5 := 
sorry

end min_area_rectangle_l488_48808


namespace probability_face_then_number_l488_48877

theorem probability_face_then_number :
  let total_cards := 52
  let total_ways_to_draw_two := total_cards * (total_cards - 1)
  let face_cards := 3 * 4
  let number_cards := 9 * 4
  let probability := (face_cards * number_cards) / total_ways_to_draw_two
  probability = 8 / 49 :=
by
  sorry

end probability_face_then_number_l488_48877


namespace set_operation_example_l488_48807

def set_operation (A B : Set ℝ) := {x | (x ∈ A ∪ B) ∧ (x ∉ A ∩ B)}

def M := {x : ℝ | -2 < x ∧ x < 2}
def N := {x : ℝ | 1 < x ∧ x < 3}

theorem set_operation_example : set_operation M N = {x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (2 ≤ x ∧ x < 3)} :=
by {
  sorry
}

end set_operation_example_l488_48807


namespace divide_P_Q_l488_48885

noncomputable def sequence_of_ones (n : ℕ) : ℕ := (10 ^ n - 1) / 9

theorem divide_P_Q (n : ℕ) (h : 1997 ∣ sequence_of_ones n) :
  1997 ∣ (sequence_of_ones (n + 1) * (10^(3*n) + 9 * 10^(2*n) + 9 * 10^n + 7)) ∧
  1997 ∣ (sequence_of_ones (n + 1) * (10^(3*(n + 1)) + 9 * 10^(2*(n + 1)) + 9 * 10^(n + 1) + 7)) := 
by
  sorry

end divide_P_Q_l488_48885


namespace find_n_l488_48896

-- Declaring the necessary context and parameters.
variable (n : ℕ)

-- Defining the condition described in the problem.
def reposting_equation (n : ℕ) : Prop := 1 + n + n^2 = 111

-- Stating the theorem to prove that for n = 10, the reposting equation holds.
theorem find_n : ∃ (n : ℕ), reposting_equation n ∧ n = 10 :=
by
  use 10
  unfold reposting_equation
  sorry

end find_n_l488_48896


namespace regular_price_of_ticket_l488_48875

theorem regular_price_of_ticket (P : Real) (discount_paid : Real) (discount_rate : Real) (paid : Real)
  (h_discount_rate : discount_rate = 0.40)
  (h_paid : paid = 9)
  (h_discount_paid : discount_paid = P * (1 - discount_rate))
  (h_paid_eq_discount_paid : paid = discount_paid) :
  P = 15 := 
by
  sorry

end regular_price_of_ticket_l488_48875


namespace interior_angle_solution_l488_48880

noncomputable def interior_angle_of_inscribed_triangle (x : ℝ) (h : (2 * x + 40) + (x + 80) + (3 * x - 50) = 360) : ℝ :=
  (1 / 2) * (x + 80)

theorem interior_angle_solution (x : ℝ) (h : (2 * x + 40) + (x + 80) + (3 * x - 50) = 360) :
  interior_angle_of_inscribed_triangle x h = 64 :=
sorry

end interior_angle_solution_l488_48880


namespace five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one_l488_48814

theorem five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one (n : ℕ) (hn : n > 0) : ¬ (4 ^ n - 1 ∣ 5 ^ n - 1) :=
sorry

end five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one_l488_48814


namespace number_of_diagonals_excluding_dividing_diagonals_l488_48851

theorem number_of_diagonals_excluding_dividing_diagonals (n : ℕ) (h1 : n = 150) :
  let totalDiagonals := n * (n - 3) / 2
  let dividingDiagonals := n / 2
  totalDiagonals - dividingDiagonals = 10950 :=
by
  sorry

end number_of_diagonals_excluding_dividing_diagonals_l488_48851


namespace additional_ice_cubes_made_l488_48854

def original_ice_cubes : ℕ := 2
def total_ice_cubes : ℕ := 9

theorem additional_ice_cubes_made :
  (total_ice_cubes - original_ice_cubes) = 7 :=
by
  sorry

end additional_ice_cubes_made_l488_48854


namespace benny_seashells_l488_48803

-- Defining the conditions
def initial_seashells : ℕ := 66
def given_away_seashells : ℕ := 52

-- Statement of the proof problem
theorem benny_seashells : (initial_seashells - given_away_seashells) = 14 :=
by
  sorry

end benny_seashells_l488_48803


namespace original_integer_is_21_l488_48895

theorem original_integer_is_21 (a b c d : ℕ) 
  (h1 : (a + b + c) / 3 + d = 29) 
  (h2 : (a + b + d) / 3 + c = 23) 
  (h3 : (a + c + d) / 3 + b = 21) 
  (h4 : (b + c + d) / 3 + a = 17) : 
  d = 21 :=
sorry

end original_integer_is_21_l488_48895


namespace income_of_deceased_l488_48892

def average_income (total_income : ℕ) (members : ℕ) : ℕ :=
  total_income / members

theorem income_of_deceased
  (total_income_before : ℕ) (members_before : ℕ) (avg_income_before : ℕ)
  (total_income_after : ℕ) (members_after : ℕ) (avg_income_after : ℕ) :
  total_income_before = members_before * avg_income_before →
  total_income_after = members_after * avg_income_after →
  members_before = 4 →
  members_after = 3 →
  avg_income_before = 735 →
  avg_income_after = 650 →
  total_income_before - total_income_after = 990 :=
by
  sorry

end income_of_deceased_l488_48892


namespace child_running_speed_l488_48830

theorem child_running_speed
  (c s t : ℝ)
  (h1 : (74 - s) * 3 = 165)
  (h2 : (74 + s) * t = 372) :
  c = 74 :=
by sorry

end child_running_speed_l488_48830


namespace integer_values_of_a_l488_48831

-- Define the polynomial P(x)
def P (a x : ℤ) : ℤ := x^3 + a * x^2 + 3 * x + 7

-- Define the main theorem
theorem integer_values_of_a (a x : ℤ) (hx : P a x = 0) (hx_is_int : x = 1 ∨ x = -1 ∨ x = 7 ∨ x = -7) :
  a = -11 ∨ a = -3 :=
by
  sorry

end integer_values_of_a_l488_48831


namespace find_a_l488_48883

def set_A : Set ℝ := {x | x^2 + x - 6 = 0}

def set_B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem find_a (a : ℝ) : set_A ∪ set_B a = set_A ↔ a ∈ ({0, 1/3, -1/2} : Set ℝ) := 
by
  sorry

end find_a_l488_48883


namespace isabella_most_efficient_jumper_l488_48864

noncomputable def weight_ricciana : ℝ := 120
noncomputable def jump_ricciana : ℝ := 4

noncomputable def weight_margarita : ℝ := 110
noncomputable def jump_margarita : ℝ := 2 * jump_ricciana - 1

noncomputable def weight_isabella : ℝ := 100
noncomputable def jump_isabella : ℝ := jump_ricciana + 3

noncomputable def ratio_ricciana : ℝ := weight_ricciana / jump_ricciana
noncomputable def ratio_margarita : ℝ := weight_margarita / jump_margarita
noncomputable def ratio_isabella : ℝ := weight_isabella / jump_isabella

theorem isabella_most_efficient_jumper :
  ratio_isabella < ratio_margarita ∧ ratio_isabella < ratio_ricciana :=
by
  sorry

end isabella_most_efficient_jumper_l488_48864


namespace solution_set_of_inequality_l488_48886

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x + 1) * (1 - 2 * x) > 0} = {x : ℝ | -1 / 3 < x ∧ x < 1 / 2} := 
by 
  sorry

end solution_set_of_inequality_l488_48886


namespace smallest_number_is_C_l488_48893

-- Define the conditions
def A := 18 + 38
def B := A - 26
def C := B / 3

-- Proof statement: C is the smallest number among A, B, and C
theorem smallest_number_is_C : C = min A (min B C) :=
by
  sorry

end smallest_number_is_C_l488_48893


namespace model_to_statue_ratio_l488_48817

theorem model_to_statue_ratio 
  (statue_height : ℝ) 
  (model_height_feet : ℝ)
  (model_height_inches : ℝ)
  (conversion_factor : ℝ) :
  statue_height = 45 → model_height_feet = 3 → conversion_factor = 12 → model_height_inches = model_height_feet * conversion_factor →
  (45 / model_height_inches) = 1.25 :=
by
  sorry

end model_to_statue_ratio_l488_48817


namespace chess_tournament_games_l488_48869

def stage1_games (players : ℕ) : ℕ := (players * (players - 1) * 2) / 2
def stage2_games (players : ℕ) : ℕ := (players * (players - 1) * 2) / 2
def stage3_games : ℕ := 4

def total_games (stage1 stage2 stage3 : ℕ) : ℕ := stage1 + stage2 + stage3

theorem chess_tournament_games : total_games (stage1_games 20) (stage2_games 10) stage3_games = 474 :=
by
  unfold stage1_games
  unfold stage2_games
  unfold total_games
  simp
  sorry

end chess_tournament_games_l488_48869


namespace sum_of_a_and_b_l488_48813

theorem sum_of_a_and_b (a b : ℝ) (h : a^2 + b^2 + 2 * a - 4 * b + 5 = 0) :
  a + b = 1 :=
sorry

end sum_of_a_and_b_l488_48813


namespace church_members_l488_48824

theorem church_members (M A C : ℕ) (h1 : A = 4/10 * M)
  (h2 : C = 6/10 * M) (h3 : C = A + 24) : M = 120 := 
  sorry

end church_members_l488_48824


namespace radius_of_smaller_molds_l488_48871

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r ^ 3

theorem radius_of_smaller_molds :
  (64 * hemisphere_volume (1/2)) = hemisphere_volume 2 :=
by
  sorry

end radius_of_smaller_molds_l488_48871


namespace area_equivalence_l488_48842

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def angle_bisector (A B C : Point) : Point := sorry
noncomputable def arc_midpoint (A B C : Point) : Point := sorry
noncomputable def is_concyclic (P Q R S : Point) : Prop := sorry
noncomputable def area_of_quad (A B C D : Point) : ℝ := sorry
noncomputable def area_of_pent (A B C D E : Point) : ℝ := sorry

theorem area_equivalence (A B C I X Y M : Point)
  (h1 : I = incenter A B C)
  (h2 : X = angle_bisector B A C)
  (h3 : Y = angle_bisector C A B)
  (h4 : M = arc_midpoint A B C)
  (h5 : is_concyclic M X I Y) :
  area_of_quad M B I C = area_of_pent B X I Y C := 
sorry

end area_equivalence_l488_48842


namespace lines_parallel_l488_48836

def l1 (x : ℝ) : ℝ := 2 * x + 1
def l2 (x : ℝ) : ℝ := 2 * x + 5

theorem lines_parallel : ∀ x1 x2 : ℝ, l1 x1 = l2 x2 → false := 
by
  intros x1 x2 h
  rw [l1, l2] at h
  sorry

end lines_parallel_l488_48836


namespace prove_u_div_p_l488_48825

theorem prove_u_div_p (p r s u : ℚ) 
  (h1 : p / r = 8)
  (h2 : s / r = 5)
  (h3 : s / u = 1 / 3) : 
  u / p = 15 / 8 := 
by 
  sorry

end prove_u_div_p_l488_48825


namespace evaluate_expression_l488_48818

def ceil (x : ℚ) : ℤ := sorry -- Implement the ceiling function for rational numbers as needed

theorem evaluate_expression :
  (ceil ((23 : ℚ) / 9 - ceil ((35 : ℚ) / 23))) 
  / (ceil ((35 : ℚ) / 9 + ceil ((9 * 23 : ℚ) / 35))) = (1 / 10 : ℚ) :=
by
  intros
  -- Proof goes here
  sorry

end evaluate_expression_l488_48818


namespace find_x_l488_48847

variables (t x : ℕ)

theorem find_x (h1 : 0 < t) (h2 : t = 4) (h3 : ((9 / 10 : ℚ) * (t * x : ℚ)) - 6 = 48) : x = 15 :=
by
  sorry

end find_x_l488_48847


namespace evaluate_cubic_diff_l488_48821

theorem evaluate_cubic_diff (x y : ℝ) (h1 : x + y = 12) (h2 : 2 * x + y = 16) : x^3 - y^3 = -448 := 
by
    sorry

end evaluate_cubic_diff_l488_48821


namespace find_angle_CBO_l488_48874

theorem find_angle_CBO :
  ∀ (BAO CAO CBO ABO ACO BCO AOC : ℝ), 
  BAO = CAO → 
  CBO = ABO → 
  ACO = BCO → 
  AOC = 110 →
  CBO = 20 :=
by
  intros BAO CAO CBO ABO ACO BCO AOC hBAO_CAOC hCBO_ABO hACO_BCO hAOC
  sorry

end find_angle_CBO_l488_48874


namespace area_of_quadrilateral_APQC_l488_48860

-- Define the geometric entities and conditions
structure RightTriangle (a b c : ℝ) :=
  (hypotenuse_eq: c = Real.sqrt (a ^ 2 + b ^ 2))

-- Triangles PAQ and PQC are right triangles with given sides
def PAQ := RightTriangle 9 12 (Real.sqrt (9^2 + 12^2))
def PQC := RightTriangle 12 9 (Real.sqrt (15^2 - 12^2))

-- Prove that the area of quadrilateral APQC is 108 square units
theorem area_of_quadrilateral_APQC :
  let area_PAQ := 1/2 * 9 * 12
  let area_PQC := 1/2 * 12 * 9
  area_PAQ + area_PQC = 108 :=
by
  sorry

end area_of_quadrilateral_APQC_l488_48860


namespace fraction_ordering_l488_48802

theorem fraction_ordering :
  (8 : ℚ) / 24 < (6 : ℚ) / 17 ∧ (6 : ℚ) / 17 < (10 : ℚ) / 27 :=
by
  sorry

end fraction_ordering_l488_48802


namespace cost_prices_l488_48853

theorem cost_prices (C_t C_c C_b : ℝ)
  (h1 : 2 * C_t = 1000)
  (h2 : 1.75 * C_c = 1750)
  (h3 : 0.75 * C_b = 1500) :
  C_t = 500 ∧ C_c = 1000 ∧ C_b = 2000 :=
by
  sorry

end cost_prices_l488_48853


namespace solve_opposite_numbers_product_l488_48809

theorem solve_opposite_numbers_product :
  ∃ (x : ℤ), 3 * x - 2 * (-x) = 30 ∧ x * (-x) = -36 :=
by
  sorry

end solve_opposite_numbers_product_l488_48809


namespace range_of_x_l488_48899

variables {x : Real}

def P (x : Real) : Prop := (x + 1) / (x - 3) ≥ 0
def Q (x : Real) : Prop := abs (1 - x/2) < 1

theorem range_of_x (hP : P x) (hQ : ¬ Q x) : x ≤ -1 ∨ x ≥ 4 :=
  sorry

end range_of_x_l488_48899


namespace total_students_like_sports_l488_48866

def Total_students := 30

def B : ℕ := 12
def C : ℕ := 10
def S : ℕ := 8
def BC : ℕ := 4
def BS : ℕ := 3
def CS : ℕ := 2
def BCS : ℕ := 1

theorem total_students_like_sports : 
  (B + C + S - (BC + BS + CS) + BCS = 22) := by
  sorry

end total_students_like_sports_l488_48866


namespace range_of_a_div_b_l488_48837

theorem range_of_a_div_b (a b : ℝ) (h1 : 1 < a ∧ a < 4) (h2 : 2 < b ∧ b < 8) : 
  1 / 8 < a / b ∧ a / b < 2 :=
sorry

end range_of_a_div_b_l488_48837


namespace largest_three_digit_multiple_of_12_and_sum_of_digits_24_l488_48859

def sum_of_digits (n : ℕ) : ℕ :=
  ((n / 100) + ((n / 10) % 10) + (n % 10))

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

def largest_three_digit_multiple_of_12_with_digits_sum_24 : ℕ :=
  996

theorem largest_three_digit_multiple_of_12_and_sum_of_digits_24 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ sum_of_digits n = 24 ∧ is_multiple_of_12 n ∧ n = largest_three_digit_multiple_of_12_with_digits_sum_24 :=
by 
  sorry

end largest_three_digit_multiple_of_12_and_sum_of_digits_24_l488_48859


namespace triangle_side_relation_l488_48841

theorem triangle_side_relation (a b c : ℝ) (α β γ : ℝ)
  (h1 : 3 * α + 2 * β = 180)
  (h2 : α + β + γ = 180) :
  a^2 + b * c - c^2 = 0 :=
sorry

end triangle_side_relation_l488_48841


namespace roots_sum_product_l488_48858

theorem roots_sum_product (p q : ℝ) (h_sum : p / 3 = 8) (h_prod : q / 3 = 12) : p + q = 60 := 
by 
  sorry

end roots_sum_product_l488_48858


namespace factor_expression_l488_48867

theorem factor_expression :
  (12 * x ^ 6 + 40 * x ^ 4 - 6) - (2 * x ^ 6 - 6 * x ^ 4 - 6) = 2 * x ^ 4 * (5 * x ^ 2 + 23) :=
by sorry

end factor_expression_l488_48867


namespace inequality_b_c_a_l488_48857

-- Define the values of a, b, and c
def a := 8^53
def b := 16^41
def c := 64^27

-- State the theorem to prove the inequality b > c > a
theorem inequality_b_c_a : b > c ∧ c > a := by
  sorry

end inequality_b_c_a_l488_48857


namespace tile_covering_problem_l488_48849

theorem tile_covering_problem :
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12  -- converting feet to inches
  let region_width := 3 * 12   -- converting feet to inches
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  region_area / tile_area = 144 := 
by 
  let tile_length := 5
  let tile_width := 3
  let region_length := 5 * 12
  let region_width := 3 * 12
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  sorry

end tile_covering_problem_l488_48849


namespace can_cut_one_more_square_l488_48840

theorem can_cut_one_more_square (G : Finset (Fin 29 × Fin 29)) (hG : G.card = 99) :
  (∃ S : Finset (Fin 29 × Fin 29), S.card = 4 ∧ (S ⊆ G) ∧ (∀ s1 s2 : Fin 29 × Fin 29, s1 ∈ S → s2 ∈ S → s1 ≠ s2 → (|s1.1 - s2.1| > 2 ∨ |s1.2 - s2.2| > 2))) :=
sorry

end can_cut_one_more_square_l488_48840


namespace factor_in_form_of_2x_l488_48846

theorem factor_in_form_of_2x (w : ℕ) (hw : w = 144) : ∃ x : ℕ, 936 * w = 2^x * P → x = 4 :=
by
  sorry

end factor_in_form_of_2x_l488_48846


namespace slices_remaining_is_correct_l488_48826

def slices_per_pizza : ℕ := 8
def pizzas_ordered : ℕ := 2
def slices_eaten : ℕ := 7
def total_slices : ℕ := slices_per_pizza * pizzas_ordered
def slices_remaining : ℕ := total_slices - slices_eaten

theorem slices_remaining_is_correct : slices_remaining = 9 := by
  sorry

end slices_remaining_is_correct_l488_48826


namespace find_fraction_l488_48833

theorem find_fraction
  (a₁ a₂ b₁ b₂ c₁ c₂ x y : ℚ)
  (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : b₁ = 6) (h₄ : b₂ = 5)
  (h₅ : c₁ = 1) (h₆ : c₂ = 7)
  (h : (a₁ / a₂) / (b₁ / b₂) = (c₁ / c₂) / (x / y)) :
  (x / y) = 2 / 5 := 
by
  sorry

end find_fraction_l488_48833


namespace cost_of_each_steak_meal_l488_48822

variable (x : ℝ)

theorem cost_of_each_steak_meal :
  (2 * x + 2 * 3.5 + 3 * 2 = 99 - 38) → x = 24 := 
by
  intro h
  sorry

end cost_of_each_steak_meal_l488_48822
