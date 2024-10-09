import Mathlib

namespace percentage_of_students_on_trip_l713_71340

-- Define the problem context
variable (total_students : ℕ)
variable (students_more_100 : ℕ)
variable (students_on_trip : ℕ)

-- Define the conditions as per the problem
def condition_1 : Prop := students_more_100 = total_students * 15 / 100
def condition_2 : Prop := students_more_100 = students_on_trip * 25 / 100

-- Define the problem statement
theorem percentage_of_students_on_trip
  (h1 : condition_1 total_students students_more_100)
  (h2 : condition_2 students_more_100 students_on_trip) :
  students_on_trip = total_students * 60 / 100 :=
by
  sorry

end percentage_of_students_on_trip_l713_71340


namespace triangle_perimeter_l713_71359

theorem triangle_perimeter (a b c : ℝ) 
  (h1 : a = 3) 
  (h2 : b = 5) 
  (hc : c ^ 2 - 3 * c = c - 3) 
  (h3 : 3 + 3 > 5) 
  (h4 : 3 + 5 > 3) 
  (h5 : 5 + 3 > 3) : 
  a + b + c = 11 :=
by
  sorry

end triangle_perimeter_l713_71359


namespace find_intersection_pair_l713_71379

def cubic_function (x : ℝ) : ℝ := x^3 - 3*x + 2

def linear_function (x y : ℝ) : Prop := x + 4*y = 4

def intersection_points (x y : ℝ) : Prop := 
  linear_function x y ∧ y = cubic_function x

def sum_x_coord (points : List (ℝ × ℝ)) : ℝ :=
  points.map Prod.fst |>.sum

def sum_y_coord (points : List (ℝ × ℝ)) : ℝ :=
  points.map Prod.snd |>.sum

theorem find_intersection_pair (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1 : intersection_points x1 y1)
  (h2 : intersection_points x2 y2)
  (h3 : intersection_points x3 y3)
  (h_sum_x : sum_x_coord [(x1, y1), (x2, y2), (x3, y3)] = 0) :
  sum_y_coord [(x1, y1), (x2, y2), (x3, y3)] = 3 :=
sorry

end find_intersection_pair_l713_71379


namespace cubic_function_increasing_l713_71339

noncomputable def f (a x : ℝ) := x ^ 3 + a * x ^ 2 + 7 * a * x

theorem cubic_function_increasing (a : ℝ) (h : 0 ≤ a ∧ a ≤ 21) :
    ∀ x y : ℝ, x ≤ y → f a x ≤ f a y :=
sorry

end cubic_function_increasing_l713_71339


namespace largest_angle_in_triangle_PQR_l713_71355

-- Definitions
def is_isosceles_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  (α = β) ∨ (β = γ) ∨ (γ = α)

def is_obtuse_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

variables (P Q R : ℝ)
variables (angleP angleQ angleR : ℝ)

-- Condition: PQR is an obtuse and isosceles triangle, and angle P measures 30 degrees
axiom h1 : is_isosceles_triangle P Q R angleP angleQ angleR
axiom h2 : is_obtuse_triangle P Q R angleP angleQ angleR
axiom h3 : angleP = 30

-- Theorem: The measure of the largest interior angle of triangle PQR is 120 degrees
theorem largest_angle_in_triangle_PQR : max angleP (max angleQ angleR) = 120 :=
  sorry

end largest_angle_in_triangle_PQR_l713_71355


namespace arith_seq_largest_portion_l713_71336

theorem arith_seq_largest_portion (a1 d : ℝ) (h_d_pos : d > 0) 
  (h_sum : 5 * a1 + 10 * d = 100)
  (h_ratio : (3 * a1 + 9 * d) / 7 = 2 * a1 + d) : 
  a1 + 4 * d = 115 / 3 := by
  sorry

end arith_seq_largest_portion_l713_71336


namespace green_yarn_length_l713_71301

/-- The length of the green piece of yarn given the red yarn is 8 cm more 
than three times the length of the green yarn and the total length 
for 2 pieces of yarn is 632 cm. -/
theorem green_yarn_length (G R : ℕ) 
  (h1 : R = 3 * G + 8)
  (h2 : G + R = 632) : 
  G = 156 := 
by
  sorry

end green_yarn_length_l713_71301


namespace pablo_days_to_complete_all_puzzles_l713_71321

def average_pieces_per_hour : ℕ := 100
def puzzles_300_pieces : ℕ := 8
def puzzles_500_pieces : ℕ := 5
def pieces_per_300_puzzle : ℕ := 300
def pieces_per_500_puzzle : ℕ := 500
def max_hours_per_day : ℕ := 7

theorem pablo_days_to_complete_all_puzzles :
  let total_pieces := (puzzles_300_pieces * pieces_per_300_puzzle) + (puzzles_500_pieces * pieces_per_500_puzzle)
  let pieces_per_day := max_hours_per_day * average_pieces_per_hour
  let days_to_complete := total_pieces / pieces_per_day
  days_to_complete = 7 :=
by
  sorry

end pablo_days_to_complete_all_puzzles_l713_71321


namespace num_common_tangents_l713_71368

-- Define the first circle
def circle1 (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 4
-- Define the second circle
def circle2 (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 1) ^ 2 = 9

-- Prove that the number of common tangent lines between the given circles is 2
theorem num_common_tangents : ∃ (n : ℕ), n = 2 ∧
  -- The circles do not intersect nor are they internally tangent
  (∀ (x y : ℝ), ¬(circle1 x y ∧ circle2 x y) ∧ 
  -- There exist exactly n common tangent lines
  ∃ (C : ℕ), C = n) :=
sorry

end num_common_tangents_l713_71368


namespace problem_solution_l713_71302

theorem problem_solution (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : x^11 - 7 * x^7 + x^3 = 0 := 
sorry

end problem_solution_l713_71302


namespace quadratic_completes_square_l713_71357

theorem quadratic_completes_square (b c : ℤ) :
  (∃ b c : ℤ, (∀ x : ℤ, x^2 - 12 * x + 49 = (x + b)^2 + c) ∧ b + c = 7) :=
sorry

end quadratic_completes_square_l713_71357


namespace basket_white_ball_probability_l713_71388

noncomputable def basket_problem_proof : Prop :=
  let P_A := 1 / 2
  let P_B := 1 / 2
  let P_W_given_A := 2 / 5
  let P_W_given_B := 1 / 4
  let P_W := P_A * P_W_given_A + P_B * P_W_given_B
  let P_A_given_W := (P_A * P_W_given_A) / P_W
  P_A_given_W = 8 / 13

theorem basket_white_ball_probability :
  basket_problem_proof :=
  sorry

end basket_white_ball_probability_l713_71388


namespace local_max_2_l713_71384

noncomputable def f (x m n : ℝ) := 2 * Real.log x - (1 / 2) * m * x^2 - n * x

theorem local_max_2 (m n : ℝ) (h : n = 1 - 2 * m) :
  ∃ m : ℝ, -1/2 < m ∧ (∀ x : ℝ, x > 0 → (∃ U : Set ℝ, IsOpen U ∧ (2 ∈ U) ∧ (∀ y ∈ U, f y m n ≤ f 2 m n))) :=
sorry

end local_max_2_l713_71384


namespace proportion_of_boys_correct_l713_71310

noncomputable def proportion_of_boys : ℚ :=
  let p_boy := 1 / 2
  let p_girl := 1 / 2
  let expected_children := 3 -- (2 boys and 1 girl)
  let expected_boys := 2 -- Expected number of boys in each family
  
  expected_boys / expected_children

theorem proportion_of_boys_correct : proportion_of_boys = 2 / 3 := by
  sorry

end proportion_of_boys_correct_l713_71310


namespace geometric_sequence_m_solution_l713_71305

theorem geometric_sequence_m_solution (m : ℝ) (h : ∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 4 ∧ a * c = b^2) :
  m = 2 ∨ m = -2 :=
by
  sorry

end geometric_sequence_m_solution_l713_71305


namespace same_solution_k_value_l713_71330

theorem same_solution_k_value 
  (x : ℝ)
  (k : ℝ)
  (m : ℝ)
  (h₁ : 2 * x + 4 = 4 * (x - 2))
  (h₂ : k * x + m = 2 * x - 1) 
  (h₃ : k = 17) : 
  k = 17 ∧ m = -91 :=
by
  sorry

end same_solution_k_value_l713_71330


namespace parallelogram_area_l713_71332

theorem parallelogram_area {a b : ℝ} (h₁ : a = 9) (h₂ : b = 12) (angle : ℝ) (h₃ : angle = 150) : 
  ∃ (area : ℝ), area = 54 * Real.sqrt 3 :=
by
  sorry

end parallelogram_area_l713_71332


namespace necessary_not_sufficient_condition_l713_71349

-- Define the necessary conditions for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  k > 5 ∨ k < -2

-- Define the condition for k
axiom k_in_real (k : ℝ) : Prop

-- The proof statement
theorem necessary_not_sufficient_condition (k : ℝ) (hk : k_in_real k) :
  (∃ (k_val : ℝ), k_val > 5 ∧ k = k_val) → represents_hyperbola k ∧ ¬ (represents_hyperbola k → k > 5) :=
by
  sorry

end necessary_not_sufficient_condition_l713_71349


namespace fraction_increase_by_five_l713_71300

variable (x y : ℝ)

theorem fraction_increase_by_five :
  let f := fun x y => (x * y) / (2 * x - 3 * y)
  f (5 * x) (5 * y) = 5 * (f x y) :=
by
  sorry

end fraction_increase_by_five_l713_71300


namespace proposition_true_and_negation_false_l713_71393

theorem proposition_true_and_negation_false (a b : ℝ) : 
  (a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ ¬(a + b ≥ 2 → (a < 1 ∧ b < 1)) :=
by {
  sorry
}

end proposition_true_and_negation_false_l713_71393


namespace range_of_a_l713_71354

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - 2 * x ^ 2

theorem range_of_a (a : ℝ) :
  (∀ x0 : ℝ, 0 < x0 ∧ x0 < 1 →
  (0 < (deriv (fun x => f a x - x)) x0)) →
  a > (4 / Real.exp (3 / 4)) :=
by
  intro h
  sorry

end range_of_a_l713_71354


namespace smallest_perfect_square_divisible_by_5_and_6_l713_71309

-- 1. Define the gcd and lcm functionality
def lcm (a b : ℕ) : ℕ :=
  (a * b) / Nat.gcd a b

-- 2. Define the condition that a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- 3. State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 : ∃ n : ℕ, is_perfect_square n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (is_perfect_square m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m :=
  sorry

end smallest_perfect_square_divisible_by_5_and_6_l713_71309


namespace booknote_unique_letters_count_l713_71375

def booknote_set : Finset Char := {'b', 'o', 'k', 'n', 't', 'e'}

theorem booknote_unique_letters_count : booknote_set.card = 6 :=
by
  sorry

end booknote_unique_letters_count_l713_71375


namespace time_to_cross_signal_pole_l713_71398

-- Given conditions
def length_of_train : ℝ := 300
def time_to_cross_platform : ℝ := 39
def length_of_platform : ℝ := 1162.5

-- The question to prove
theorem time_to_cross_signal_pole :
  (length_of_train / ((length_of_train + length_of_platform) / time_to_cross_platform)) = 8 :=
by
  sorry

end time_to_cross_signal_pole_l713_71398


namespace special_discount_percentage_l713_71365

theorem special_discount_percentage (original_price discounted_price : ℝ) (h₀ : original_price = 80) (h₁ : discounted_price = 68) : 
  ((original_price - discounted_price) / original_price) * 100 = 15 :=
by 
  sorry

end special_discount_percentage_l713_71365


namespace num_letters_with_line_no_dot_l713_71308

theorem num_letters_with_line_no_dot :
  ∀ (total_letters with_dot_and_line : ℕ) (with_dot_only with_line_only : ℕ),
    (total_letters = 60) →
    (with_dot_and_line = 20) →
    (with_dot_only = 4) →
    (total_letters = with_dot_and_line + with_dot_only + with_line_only) →
    with_line_only = 36 :=
by
  intros total_letters with_dot_and_line with_dot_only with_line_only
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end num_letters_with_line_no_dot_l713_71308


namespace discount_for_multiple_rides_l713_71390

-- Definitions based on given conditions
def ferris_wheel_cost : ℝ := 2.0
def roller_coaster_cost : ℝ := 7.0
def coupon_value : ℝ := 1.0
def total_tickets_needed : ℝ := 7.0

-- The proof problem
theorem discount_for_multiple_rides : 
  (ferris_wheel_cost + roller_coaster_cost) - (total_tickets_needed - coupon_value) = 2.0 :=
by
  sorry

end discount_for_multiple_rides_l713_71390


namespace petya_recover_x_y_l713_71345

theorem petya_recover_x_y (x y a b c d : ℝ)
    (hx_pos : x > 0) (hy_pos : y > 0)
    (ha : a = x + y) (hb : b = x - y) (hc : c = x / y) (hd : d = x * y) :
    ∃! (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ a = x' + y' ∧ b = x' - y' ∧ c = x' / y' ∧ d = x' * y' :=
sorry

end petya_recover_x_y_l713_71345


namespace parallel_lines_condition_l713_71328

theorem parallel_lines_condition (k_1 k_2 : ℝ) :
  (k_1 = k_2) ↔ (∀ x y : ℝ, k_1 * x + y + 1 = 0 → k_2 * x + y - 1 = 0) :=
sorry

end parallel_lines_condition_l713_71328


namespace perfect_square_condition_l713_71333

def is_perfect_square (x : ℤ) : Prop := ∃ k : ℤ, k^2 = x

noncomputable def a_n (n : ℕ) : ℤ := (10^n - 1) / 9

theorem perfect_square_condition (n b : ℕ) (h1 : 0 < b) (h2 : b < 10) :
  is_perfect_square ((a_n (2 * n)) - b * (a_n n)) ↔ (b = 2 ∨ (b = 7 ∧ n = 1)) := by
  sorry

end perfect_square_condition_l713_71333


namespace product_102_108_l713_71352

theorem product_102_108 : (102 = 105 - 3) → (108 = 105 + 3) → (102 * 108 = 11016) := by
  sorry

end product_102_108_l713_71352


namespace length_of_pencils_l713_71374

theorem length_of_pencils (length_pencil1 : ℕ) (length_pencil2 : ℕ)
  (h1 : length_pencil1 = 12) (h2 : length_pencil2 = 12) : length_pencil1 + length_pencil2 = 24 :=
by
  sorry

end length_of_pencils_l713_71374


namespace avg_speed_of_car_l713_71306

noncomputable def average_speed (distance1 distance2 : ℕ) (time1 time2 : ℕ) : ℕ :=
  (distance1 + distance2) / (time1 + time2)

theorem avg_speed_of_car :
  average_speed 65 45 1 1 = 55 := by
  sorry

end avg_speed_of_car_l713_71306


namespace sum_of_x_intercepts_l713_71343

theorem sum_of_x_intercepts (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (5 : ℤ) * (3 : ℤ) = (a : ℤ) * (b : ℤ)) : 
  ((-5 : ℤ) / (a : ℤ)) + ((-5 : ℤ) / (3 : ℤ)) + ((-1 : ℤ) / (1 : ℤ)) + ((-1 : ℤ) / (15 : ℤ)) = -8 := 
by 
  sorry

end sum_of_x_intercepts_l713_71343


namespace find_a_b_and_m_range_l713_71381

-- Definitions and initial conditions
def f (x : ℝ) (a b m : ℝ) : ℝ := 2*x^3 + a*x^2 + b*x + m
def f_prime (x : ℝ) (a b : ℝ) : ℝ := 6*x^2 + 2*a*x + b

-- Problem statement
theorem find_a_b_and_m_range (a b m : ℝ) :
  (∀ x, f_prime x a b = 6 * (x + 0.5)^2 - k) →
  f_prime 1 a b = 0 →
  a = 3 ∧ b = -12 ∧ -20 < m ∧ m < 7 :=
sorry

end find_a_b_and_m_range_l713_71381


namespace radishes_in_first_basket_l713_71313

theorem radishes_in_first_basket :
  ∃ x : ℕ, ∃ y : ℕ, x + y = 88 ∧ y = x + 14 ∧ x = 37 :=
by
  -- Proof goes here
  sorry

end radishes_in_first_basket_l713_71313


namespace contractor_fine_per_day_l713_71362

theorem contractor_fine_per_day
    (total_days : ℕ) 
    (work_days_fine_amt : ℕ) 
    (total_amt : ℕ) 
    (absent_days : ℕ) 
    (worked_days : ℕ := total_days - absent_days)
    (earned_amt : ℕ := worked_days * work_days_fine_amt)
    (fine_per_day : ℚ)
    (total_fine : ℚ := absent_days * fine_per_day) : 
    (earned_amt - total_fine = total_amt) → 
    fine_per_day = 7.5 :=
by
  intros h
  -- proof here is omitted
  sorry

end contractor_fine_per_day_l713_71362


namespace no_such_P_exists_l713_71387

theorem no_such_P_exists (P : Polynomial ℤ) (r : ℕ) (r_ge_3 : r ≥ 3) (a : Fin r → ℤ)
  (distinct_a : ∀ i j, i ≠ j → a i ≠ a j)
  (P_cycle : ∀ i, P.eval (a i) = a ⟨(i + 1) % r, sorry⟩)
  : False :=
sorry

end no_such_P_exists_l713_71387


namespace ice_skating_rinks_and_ski_resorts_2019_l713_71399

theorem ice_skating_rinks_and_ski_resorts_2019 (x y : ℕ) :
  x + y = 1230 →
  2 * x + 212 + y + 288 = 2560 →
  x = 830 ∧ y = 400 :=
by {
  sorry
}

end ice_skating_rinks_and_ski_resorts_2019_l713_71399


namespace john_average_speed_l713_71304

theorem john_average_speed
  (uphill_distance : ℝ)
  (uphill_time : ℝ)
  (downhill_distance : ℝ)
  (downhill_time : ℝ)
  (uphill_time_is_45_minutes : uphill_time = 45)
  (downhill_time_is_15_minutes : downhill_time = 15)
  (uphill_distance_is_3_km : uphill_distance = 3)
  (downhill_distance_is_3_km : downhill_distance = 3)
  : (uphill_distance + downhill_distance) / ((uphill_time + downhill_time) / 60) = 6 := 
by
  sorry

end john_average_speed_l713_71304


namespace part_a_l713_71356

theorem part_a (a b c : ℝ) : 
  (∀ n : ℝ, (n + 2)^2 = a * (n + 1)^2 + b * n^2 + c * (n - 1)^2) ↔ (a = 3 ∧ b = -3 ∧ c = 1) :=
by 
  sorry

end part_a_l713_71356


namespace set_D_is_empty_l713_71329

-- Definitions based on the conditions from the original problem
def set_A : Set ℝ := {x | x + 3 = 3}
def set_B : Set (ℝ × ℝ) := {(x, y) | y^2 = -x^2}
def set_C : Set ℝ := {x | x^2 ≤ 0}
def set_D : Set ℝ := {x | x^2 - x + 1 = 0}

-- The theorem statement
theorem set_D_is_empty : set_D = ∅ :=
sorry

end set_D_is_empty_l713_71329


namespace max_a_no_lattice_point_l713_71317

theorem max_a_no_lattice_point (a : ℚ) : a = 35 / 51 ↔ 
  (∀ (m : ℚ), (2 / 3 < m ∧ m < a) → 
    (∀ (x : ℤ), (0 < x ∧ x ≤ 50) → 
      ¬ ∃ (y : ℤ), y = m * x + 5)) :=
sorry

end max_a_no_lattice_point_l713_71317


namespace minimal_fence_length_l713_71371

-- Define the conditions as assumptions
axiom side_length : ℝ
axiom num_paths : ℕ
axiom path_length : ℝ

-- Assume the conditions given in the problem
axiom side_length_value : side_length = 50
axiom num_paths_value : num_paths = 13
axiom path_length_value : path_length = 50

-- Define the theorem to be proved
theorem minimal_fence_length : (num_paths * path_length) = 650 := by
  -- The proof goes here
  sorry

end minimal_fence_length_l713_71371


namespace distinct_banners_l713_71331

inductive Color
| red
| white
| blue
| green
| yellow

def adjacent_different (a b : Color) : Prop := a ≠ b

theorem distinct_banners : 
  ∃ n : ℕ, n = 320 ∧ ∀ strips : Fin 4 → Color, 
    adjacent_different (strips 0) (strips 1) ∧ 
    adjacent_different (strips 1) (strips 2) ∧ 
    adjacent_different (strips 2) (strips 3) :=
sorry

end distinct_banners_l713_71331


namespace range_of_f_l713_71394

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |x| - 1 else Real.sin x ^ 2

theorem range_of_f : Set.range f = Set.Ioi (-1) := 
  sorry

end range_of_f_l713_71394


namespace sum_of_tens_and_units_digit_of_8_pow_100_l713_71395

noncomputable def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
noncomputable def units_digit (n : ℕ) : ℕ := n % 10
noncomputable def sum_of_digits (n : ℕ) := tens_digit n + units_digit n

theorem sum_of_tens_and_units_digit_of_8_pow_100 : sum_of_digits (8 ^ 100) = 13 :=
by 
  sorry

end sum_of_tens_and_units_digit_of_8_pow_100_l713_71395


namespace inequality_cannot_hold_l713_71342

variable (a b : ℝ)
variable (h : a < b ∧ b < 0)

theorem inequality_cannot_hold (h : a < b ∧ b < 0) : ¬ (1 / (a - b) > 1 / a) := 
by {
  sorry
}

end inequality_cannot_hold_l713_71342


namespace find_C_l713_71392

theorem find_C (A B C : ℕ) (hA : A = 509) (hAB : A = B + 197) (hCB : C = B - 125) : C = 187 := 
by 
  sorry

end find_C_l713_71392


namespace right_triangle_properties_l713_71347

theorem right_triangle_properties (a b c h : ℝ)
  (ha: a = 5) (hb: b = 12) (h_right_angle: a^2 + b^2 = c^2)
  (h_area: (1/2) * a * b = (1/2) * c * h) :
  c = 13 ∧ h = 60 / 13 :=
by
  sorry

end right_triangle_properties_l713_71347


namespace problem_statement_l713_71361

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l713_71361


namespace sum_of_valid_m_values_l713_71346

-- Variables and assumptions
variable (m x : ℝ)

-- Conditions from the given problem
def inequality_system (m x : ℝ) : Prop :=
  (x - 4) / 3 < x - 4 ∧ (m - x) / 5 < 0

def solution_set_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, inequality_system m x → x > 4

def fractional_equation (m x : ℝ) : Prop :=
  6 / (x - 3) + 1 = (m * x - 3) / (x - 3)

-- Lean statement to prove the sum of integers satisfying the conditions
theorem sum_of_valid_m_values : 
  (∀ m : ℝ, solution_set_condition m ∧ 
            (∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ fractional_equation m x) →
            (∃ (k : ℕ), k = 2 ∨ k = 4) → 
            2 + 4 = 6) :=
sorry

end sum_of_valid_m_values_l713_71346


namespace tangent_line_at_1_1_l713_71386

noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

theorem tangent_line_at_1_1 :
  let m := -((2 * 1 - 1 - 2 * 1) / (2 * 1 - 1)^2) -- Derivative evaluated at x = 1
  let tangent_line (x y : ℝ) := x + y - 2
  ∀ x y : ℝ, tangent_line x y = 0 → (f x = y ∧ x = 1 → y = 1 → m = -1) :=
by
  sorry

end tangent_line_at_1_1_l713_71386


namespace smallest_number_divisible_l713_71396

theorem smallest_number_divisible (n : ℕ) 
    (h1 : (n - 20) % 15 = 0) 
    (h2 : (n - 20) % 30 = 0)
    (h3 : (n - 20) % 45 = 0)
    (h4 : (n - 20) % 60 = 0) : 
    n = 200 :=
sorry

end smallest_number_divisible_l713_71396


namespace max_height_of_rock_l713_71360

theorem max_height_of_rock : 
    ∃ t_max : ℝ, (∀ t : ℝ, -5 * t^2 + 25 * t + 10 ≤ -5 * t_max^2 + 25 * t_max + 10) ∧ (-5 * t_max^2 + 25 * t_max + 10 = 165 / 4) := 
sorry

end max_height_of_rock_l713_71360


namespace total_work_completed_in_days_l713_71378

theorem total_work_completed_in_days (T : ℕ) :
  (amit_days amit_worked ananthu_days remaining_work : ℕ) → 
  amit_days = 3 → amit_worked = amit_days * (1 / 15) → 
  ananthu_days = 36 → 
  remaining_work = 1 - amit_worked  →
  (ananthu_days * (1 / 45)) = remaining_work →
  T = amit_days + ananthu_days →
  T = 39 := 
sorry

end total_work_completed_in_days_l713_71378


namespace matrix_pow_six_identity_l713_71380

variable {n : Type} [Fintype n] [DecidableEq n]
variables {A B C : Matrix n n ℂ}

theorem matrix_pow_six_identity 
  (h1 : A^2 = B^2) (h2 : B^2 = C^2) (h3 : B^3 = A * B * C + 2 * (1 : Matrix n n ℂ)) : 
  A^6 = 1 :=
by 
  sorry

end matrix_pow_six_identity_l713_71380


namespace harrys_mothers_age_l713_71316

theorem harrys_mothers_age 
  (h : ℕ)  -- Harry's age
  (f : ℕ)  -- Father's age
  (m : ℕ)  -- Mother's age
  (h_age : h = 50)
  (f_age : f = h + 24)
  (m_age : m = f - h / 25) 
  : (m - h = 22) := 
by
  sorry

end harrys_mothers_age_l713_71316


namespace solve_equation_l713_71382

noncomputable def equation_solution (x : ℝ) : Prop :=
  (3 / x = 2 / (x - 2)) ∧ x ≠ 0 ∧ x - 2 ≠ 0

theorem solve_equation : (equation_solution 6) :=
  by
    sorry

end solve_equation_l713_71382


namespace fewest_occupied_seats_l713_71397

theorem fewest_occupied_seats (n m : ℕ) (h₁ : n = 150) (h₂ : (m * 4 + 3 < 150)) : m = 37 :=
by
  sorry

end fewest_occupied_seats_l713_71397


namespace cost_per_trip_l713_71376

theorem cost_per_trip (cost_per_pass : ℕ) (num_passes : ℕ) (trips_oldest : ℕ) (trips_youngest : ℕ) :
    cost_per_pass = 100 →
    num_passes = 2 →
    trips_oldest = 35 →
    trips_youngest = 15 →
    (cost_per_pass * num_passes) / (trips_oldest + trips_youngest) = 4 := by
  sorry

end cost_per_trip_l713_71376


namespace trig_evaluation_l713_71303

noncomputable def sin30 := 1 / 2
noncomputable def cos45 := Real.sqrt 2 / 2
noncomputable def tan30 := Real.sqrt 3 / 3
noncomputable def sin60 := Real.sqrt 3 / 2

theorem trig_evaluation : 4 * sin30 - Real.sqrt 2 * cos45 - Real.sqrt 3 * tan30 + 2 * sin60 = Real.sqrt 3 := by
  sorry

end trig_evaluation_l713_71303


namespace total_cost_is_53_l713_71377

-- Defining the costs and quantities as constants
def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 10
def discount : ℕ := 5

-- Get the cost of sandwiches purchased
def cost_of_sandwiches : ℕ := num_sandwiches * sandwich_cost

-- Get the cost of sodas purchased
def cost_of_sodas : ℕ := num_sodas * soda_cost

-- Calculate the total cost before discount
def total_cost_before_discount : ℕ := cost_of_sandwiches + cost_of_sodas

-- Calculate the total cost after discount
def total_cost_after_discount : ℕ := total_cost_before_discount - discount

-- The theorem stating that the total cost is 53 dollars
theorem total_cost_is_53 : total_cost_after_discount = 53 :=
by
  sorry

end total_cost_is_53_l713_71377


namespace tangent_line_MP_l713_71311

theorem tangent_line_MP
  (O : Type)
  (circle : O → O → Prop)
  (K M N P L : O)
  (is_tangent : O → O → Prop)
  (is_diameter : O → O → O)
  (K_tangent : is_tangent K M)
  (eq_segments : ∀ {P Q R}, circle P Q → circle Q R → circle P R → (P, Q) = (Q, R))
  (diam_opposite : L = is_diameter K L)
  (line_intrsc : ∀ {X Y}, is_tangent X Y → circle X Y → (Y = Y) → P = Y)
  (circ : ∀ {X Y}, circle X Y) :
  is_tangent M P :=
by
  sorry

end tangent_line_MP_l713_71311


namespace sphere_radius_l713_71389

theorem sphere_radius (R : ℝ) (h : 4 * Real.pi * R^2 = 4 * Real.pi) : R = 1 :=
by
  sorry

end sphere_radius_l713_71389


namespace fraction_zero_implies_x_half_l713_71370

theorem fraction_zero_implies_x_half (x : ℝ) (h₁ : (2 * x - 1) / (x + 2) = 0) (h₂ : x ≠ -2) : x = 1 / 2 :=
by sorry

end fraction_zero_implies_x_half_l713_71370


namespace Jans_original_speed_l713_71366

theorem Jans_original_speed
  (doubled_speed : ℕ → ℕ) (skips_after_training : ℕ) (time_in_minutes : ℕ) (original_speed : ℕ) :
  (∀ (s : ℕ), doubled_speed s = 2 * s) → 
  skips_after_training = 700 → 
  time_in_minutes = 5 → 
  (original_speed = (700 / 5) / 2) → 
  original_speed = 70 := 
by
  intros h1 h2 h3 h4
  exact h4

end Jans_original_speed_l713_71366


namespace find_k_hyperbola_l713_71334

-- Define the given conditions
variables (k : ℝ)
def condition1 : Prop := k < 0
def condition2 : Prop := 2 * k^2 + k - 2 = -1

-- State the proof goal
theorem find_k_hyperbola (h1 : condition1 k) (h2 : condition2 k) : k = -1 :=
by
  sorry

end find_k_hyperbola_l713_71334


namespace find_ks_l713_71324

def is_valid_function (f : ℕ → ℤ) (k : ℤ) : Prop :=
  ∀ x y : ℕ, f (x * y) = f x + f y + k * f (Nat.gcd x y)

theorem find_ks (f : ℕ → ℤ) :
  (f 2006 = 2007) →
  is_valid_function f k →
  k = 0 ∨ k = -1 :=
sorry

end find_ks_l713_71324


namespace exists_arithmetic_seq_2003_terms_perfect_powers_no_infinite_arithmetic_seq_perfect_powers_l713_71353

-- Part (a): Proving the existence of such an arithmetic sequence with 2003 terms.
theorem exists_arithmetic_seq_2003_terms_perfect_powers :
  ∃ (a : ℕ) (d : ℕ), ∀ n : ℕ, n ≤ 2002 → ∃ (k m : ℕ), m > 1 ∧ a + n * d = k ^ m :=
by
  sorry

-- Part (b): Proving the non-existence of such an infinite arithmetic sequence.
theorem no_infinite_arithmetic_seq_perfect_powers :
  ¬ ∃ (a : ℕ) (d : ℕ), ∀ n : ℕ, ∃ (k m : ℕ), m > 1 ∧ a + n * d = k ^ m :=
by
  sorry

end exists_arithmetic_seq_2003_terms_perfect_powers_no_infinite_arithmetic_seq_perfect_powers_l713_71353


namespace range_of_m_l713_71312

-- Define sets A and B
def A := {x : ℝ | x ≤ 1}
def B (m : ℝ) := {x : ℝ | x ≤ m}

-- Statement: Prove the range of m such that B ⊆ A
theorem range_of_m (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ↔ (m ≤ 1) :=
by sorry

end range_of_m_l713_71312


namespace number_of_possible_values_of_a_l713_71383

theorem number_of_possible_values_of_a :
  ∃ a_values : Finset ℕ, 
    (∀ a ∈ a_values, 5 ∣ a) ∧ 
    (∀ a ∈ a_values, a ∣ 30) ∧ 
    (∀ a ∈ a_values, 0 < a) ∧ 
    a_values.card = 4 :=
by
  sorry

end number_of_possible_values_of_a_l713_71383


namespace find_quadratic_eq_with_given_roots_l713_71369

theorem find_quadratic_eq_with_given_roots (A z x1 x2 : ℝ) 
  (h1 : A * z * x1^2 + x1 * x1 + x2 = 0) 
  (h2 : A * z * x2^2 + x1 * x2 + x2 = 0) : 
  (A * z * x^2 + x1 * x - x2 = 0) :=
by
  sorry

end find_quadratic_eq_with_given_roots_l713_71369


namespace range_of_fx_a_eq_2_range_of_a_increasing_fx_l713_71323

-- Part (1)
theorem range_of_fx_a_eq_2 (x : ℝ) (h : x ∈ Set.Icc (-2 : ℝ) (3 : ℝ)) :
  ∃ y ∈ Set.Icc (-21 / 4 : ℝ) (15 : ℝ), y = x^2 + 3 * x - 3 :=
sorry

-- Part (2)
theorem range_of_a_increasing_fx (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) (3 : ℝ) → 2 * x + 2 * a - 1 ≥ 0) ↔ a ∈ Set.Ici (3 / 2 : ℝ) :=
sorry

end range_of_fx_a_eq_2_range_of_a_increasing_fx_l713_71323


namespace perimeter_of_rectangle_l713_71364

theorem perimeter_of_rectangle (DC BC P : ℝ) (hDC : DC = 12) (hArea : 1/2 * DC * BC = 30) : P = 2 * (DC + BC) → P = 34 :=
by
  sorry

end perimeter_of_rectangle_l713_71364


namespace tangent_line_at_one_l713_71363

noncomputable def f (x : ℝ) := Real.log x + 2 * x^2 - 4 * x

theorem tangent_line_at_one :
  let slope := (1/x + 4*x - 4) 
  let y_val := -2 
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), A = 1 ∧ B = -1 ∧ C = -3 ∧ (∀ (x y : ℝ), f x = y → A * x + B * y + C = 0) :=
by
  sorry

end tangent_line_at_one_l713_71363


namespace min_value_c_plus_d_l713_71314

theorem min_value_c_plus_d (c d : ℤ) (h : c * d = 144) : c + d = -145 :=
sorry

end min_value_c_plus_d_l713_71314


namespace cos_2alpha_2beta_l713_71320

variables (α β : ℝ)

open Real

theorem cos_2alpha_2beta (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) : cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_2alpha_2beta_l713_71320


namespace classroom_width_perimeter_ratio_l713_71351

theorem classroom_width_perimeter_ratio
  (L : Real) (W : Real) (P : Real)
  (hL : L = 15) (hW : W = 10)
  (hP : P = 2 * (L + W)) :
  W / P = 1 / 5 :=
sorry

end classroom_width_perimeter_ratio_l713_71351


namespace one_gallon_fills_one_cubic_foot_l713_71307

theorem one_gallon_fills_one_cubic_foot
  (total_water : ℕ)
  (drinking_cooking : ℕ)
  (shower_water : ℕ)
  (num_showers : ℕ)
  (pool_length : ℕ)
  (pool_width : ℕ)
  (pool_height : ℕ)
  (h_total_water : total_water = 1000)
  (h_drinking_cooking : drinking_cooking = 100)
  (h_shower_water : shower_water = 20)
  (h_num_showers : num_showers = 15)
  (h_pool_length : pool_length = 10)
  (h_pool_width : pool_width = 10)
  (h_pool_height : pool_height = 6) :
  (pool_length * pool_width * pool_height) / 
  (total_water - drinking_cooking - num_showers * shower_water) = 1 := by
  sorry

end one_gallon_fills_one_cubic_foot_l713_71307


namespace ratio_eq_one_l713_71319

variable {a b : ℝ}

theorem ratio_eq_one (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) : (a / 8) / (b / 7) = 1 := 
by
  sorry

end ratio_eq_one_l713_71319


namespace arc_length_of_sector_l713_71344

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h : r = Real.pi ∧ θ = 120) : 
  r * θ / 180 * Real.pi = 2 * Real.pi * Real.pi / 3 :=
by
  sorry

end arc_length_of_sector_l713_71344


namespace theorem_incorrect_statement_D_l713_71372

open Real

def incorrect_statement_D (φ : ℝ) (hφ : φ > 0) (x : ℝ) : Prop :=
  cos (2*x + φ) ≠ cos (2*(x - φ/2))

theorem theorem_incorrect_statement_D (φ : ℝ) (hφ : φ > 0) : 
  ∃ x : ℝ, incorrect_statement_D φ hφ x :=
by
  sorry

end theorem_incorrect_statement_D_l713_71372


namespace sqrt_inequality_l713_71325

theorem sqrt_inequality (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
  sorry

end sqrt_inequality_l713_71325


namespace a_takes_30_minutes_more_l713_71350

noncomputable def speed_ratio := 3 / 4
noncomputable def time_A := 2 -- 2 hours
noncomputable def time_diff (b_time : ℝ) := time_A - b_time

theorem a_takes_30_minutes_more (b_time : ℝ) 
  (h_ratio : speed_ratio = 3 / 4)
  (h_a : time_A = 2) :
  time_diff b_time = 0.5 →  -- because 0.5 hours = 30 minutes
  time_diff b_time * 60 = 30 :=
by sorry

end a_takes_30_minutes_more_l713_71350


namespace largest_divisor_of_n_l713_71326

theorem largest_divisor_of_n (n : ℕ) (hn : 0 < n) (h : 50 ∣ n^2) : 5 ∣ n :=
sorry

end largest_divisor_of_n_l713_71326


namespace y_increase_by_18_when_x_increases_by_12_l713_71373

theorem y_increase_by_18_when_x_increases_by_12
  (h_slope : ∀ x y: ℝ, (4 * y = 6 * x) ↔ (3 * y = 2 * x)) :
  ∀ Δx : ℝ, Δx = 12 → ∃ Δy : ℝ, Δy = 18 :=
by
  sorry

end y_increase_by_18_when_x_increases_by_12_l713_71373


namespace chipmunks_initial_count_l713_71322

variable (C : ℕ) (total : ℕ) (morning_beavers : ℕ) (afternoon_beavers : ℕ) (decrease_chipmunks : ℕ)

axiom chipmunks_count : morning_beavers = 20 
axiom beavers_double : afternoon_beavers = 2 * morning_beavers
axiom decrease_chipmunks_initial : decrease_chipmunks = 10
axiom total_animals : total = 130

theorem chipmunks_initial_count : 
  20 + C + (2 * 20) + (C - 10) = 130 → C = 40 :=
by
  intros h
  sorry

end chipmunks_initial_count_l713_71322


namespace carols_father_gave_5_peanuts_l713_71391

theorem carols_father_gave_5_peanuts : 
  ∀ (c: ℕ) (f: ℕ), c = 2 → c + f = 7 → f = 5 :=
by
  intros c f h1 h2
  sorry

end carols_father_gave_5_peanuts_l713_71391


namespace total_fencing_cost_l713_71315

-- Definitions based on the conditions
def cost_per_side : ℕ := 69
def number_of_sides : ℕ := 4

-- The proof problem statement
theorem total_fencing_cost : number_of_sides * cost_per_side = 276 := by
  sorry

end total_fencing_cost_l713_71315


namespace monthly_food_expense_l713_71338

-- Definitions based on the given conditions
def E : ℕ := 6000
def R : ℕ := 640
def EW : ℕ := E / 4
def I : ℕ := E / 5
def L : ℕ := 2280

-- Define the monthly food expense F
def F : ℕ := E - (R + EW + I) - L

-- The theorem stating that the monthly food expense is 380
theorem monthly_food_expense : F = 380 := 
by
  -- proof goes here
  sorry

end monthly_food_expense_l713_71338


namespace find_k_l713_71358

theorem find_k (a b : ℕ) (k : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (a^2 + b^2) = k * (a * b - 1)) :
  k = 5 :=
sorry

end find_k_l713_71358


namespace find_constants_l713_71335

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
if x < 3 then a * x^2 + b else 10 - 2 * x

theorem find_constants (a b : ℝ)
  (H : ∀ x, f a b (f a b x) = x) :
  a + b = 13 / 3 := by 
  sorry

end find_constants_l713_71335


namespace angles_in_quadrilateral_l713_71367

theorem angles_in_quadrilateral (A B C D : ℝ)
    (h : A / B = 1 / 3 ∧ B / C = 3 / 5 ∧ C / D = 5 / 6)
    (sum_angles : A + B + C + D = 360) :
    A = 24 ∧ D = 144 := 
by
    sorry

end angles_in_quadrilateral_l713_71367


namespace find_nm_2023_l713_71385

theorem find_nm_2023 (n m : ℚ) (h : (n + 9)^2 + |m - 8| = 0) : (n + m) ^ 2023 = -1 := by
  sorry

end find_nm_2023_l713_71385


namespace total_notebooks_l713_71348

-- Definitions from the conditions
def Yoongi_notebooks : Nat := 3
def Jungkook_notebooks : Nat := 3
def Hoseok_notebooks : Nat := 3

-- The proof problem
theorem total_notebooks : Yoongi_notebooks + Jungkook_notebooks + Hoseok_notebooks = 9 := 
by 
  sorry

end total_notebooks_l713_71348


namespace ceil_square_count_ceil_x_eq_15_l713_71337

theorem ceil_square_count_ceil_x_eq_15 : 
  ∀ (x : ℝ), ( ⌈x⌉ = 15 ) → ∃ n : ℕ, n = 29 ∧ ∀ k : ℕ, k = ⌈x^2⌉ → 197 ≤ k ∧ k ≤ 225 :=
sorry

end ceil_square_count_ceil_x_eq_15_l713_71337


namespace ratio_and_lcm_l713_71318

noncomputable def common_factor (a b : ℕ) := ∃ x : ℕ, a = 3 * x ∧ b = 4 * x

theorem ratio_and_lcm (a b : ℕ) (h1 : common_factor a b) (h2 : Nat.lcm a b = 180) (h3 : a = 60) : b = 45 :=
by sorry

end ratio_and_lcm_l713_71318


namespace find_integer_n_l713_71341

theorem find_integer_n (n : ℤ) : 
  (∃ m : ℤ, n = 35 * m + 24) ↔ (5 ∣ (3 * n - 2) ∧ 7 ∣ (2 * n + 1)) :=
by sorry

end find_integer_n_l713_71341


namespace socks_cost_5_l713_71327

theorem socks_cost_5
  (jeans t_shirt socks : ℕ)
  (h1 : jeans = 2 * t_shirt)
  (h2 : t_shirt = socks + 10)
  (h3 : jeans = 30) :
  socks = 5 :=
by
  sorry

end socks_cost_5_l713_71327
