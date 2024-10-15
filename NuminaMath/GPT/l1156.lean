import Mathlib

namespace NUMINAMATH_GPT_water_tank_capacity_l1156_115671

theorem water_tank_capacity (C : ℝ) (h : 0.70 * C - 0.40 * C = 36) : C = 120 :=
sorry

end NUMINAMATH_GPT_water_tank_capacity_l1156_115671


namespace NUMINAMATH_GPT_avg_age_boys_class_l1156_115650

-- Definitions based on conditions
def avg_age_students : ℝ := 15.8
def avg_age_girls : ℝ := 15.4
def ratio_boys_girls : ℝ := 1.0000000000000044

-- Using the given conditions to define the average age of boys
theorem avg_age_boys_class (B G : ℕ) (A_b : ℝ) 
  (h1 : avg_age_students = (B * A_b + G * avg_age_girls) / (B + G)) 
  (h2 : B = ratio_boys_girls * G) : 
  A_b = 16.2 :=
  sorry

end NUMINAMATH_GPT_avg_age_boys_class_l1156_115650


namespace NUMINAMATH_GPT_find_x_l1156_115625

theorem find_x 
  (x : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (hP : P = (x, 6)) 
  (hcos : Real.cos θ = -4/5) 
  : x = -8 := 
sorry

end NUMINAMATH_GPT_find_x_l1156_115625


namespace NUMINAMATH_GPT_ellipse_parabola_four_intersections_intersection_points_lie_on_circle_l1156_115698

-- Part A:
-- Define intersections of a given ellipse and parabola under conditions on m and n
theorem ellipse_parabola_four_intersections (m n : ℝ) :
  (3 / n < m) ∧ (m < (4 * m^2 + 9) / (4 * m)) ∧ (m > 3 / 2) →
  ∃ x y : ℝ, (x^2 / n + y^2 / 9 = 1) ∧ (y = x^2 - m) :=
sorry

-- Part B:
-- Prove four intersection points of given ellipse and parabola lie on same circle for m = n = 4
theorem intersection_points_lie_on_circle (x y : ℝ) :
  (4 / 4 + y^2 / 9 = 1) ∧ (y = x^2 - 4) →
  ∃ k l r : ℝ, ∀ x' y', ((x' - k)^2 + (y' - l)^2 = r^2) :=
sorry

end NUMINAMATH_GPT_ellipse_parabola_four_intersections_intersection_points_lie_on_circle_l1156_115698


namespace NUMINAMATH_GPT_pants_cost_correct_l1156_115691

def shirt_cost : ℕ := 43
def tie_cost : ℕ := 15
def total_paid : ℕ := 200
def change_received : ℕ := 2

def total_spent : ℕ := total_paid - change_received
def combined_cost : ℕ := shirt_cost + tie_cost
def pants_cost : ℕ := total_spent - combined_cost

theorem pants_cost_correct : pants_cost = 140 :=
by
  -- We'll leave the proof as an exercise.
  sorry

end NUMINAMATH_GPT_pants_cost_correct_l1156_115691


namespace NUMINAMATH_GPT_reciprocal_sum_is_1_implies_at_least_one_is_2_l1156_115656

-- Lean statement for the problem
theorem reciprocal_sum_is_1_implies_at_least_one_is_2 (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_sum : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1) : 
  a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 := 
sorry

end NUMINAMATH_GPT_reciprocal_sum_is_1_implies_at_least_one_is_2_l1156_115656


namespace NUMINAMATH_GPT_units_digit_31_2020_units_digit_37_2020_l1156_115683

theorem units_digit_31_2020 : ((31 ^ 2020) % 10) = 1 := by
  sorry

theorem units_digit_37_2020 : ((37 ^ 2020) % 10) = 1 := by
  sorry

end NUMINAMATH_GPT_units_digit_31_2020_units_digit_37_2020_l1156_115683


namespace NUMINAMATH_GPT_unattainable_value_of_y_l1156_115659

noncomputable def f (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

theorem unattainable_value_of_y :
  ∃ y : ℝ, y = -(1 / 3) ∧ ∀ x : ℝ, 3 * x + 4 ≠ 0 → f x ≠ y :=
by
  sorry

end NUMINAMATH_GPT_unattainable_value_of_y_l1156_115659


namespace NUMINAMATH_GPT_thomas_total_training_hours_l1156_115674

-- Define the conditions from the problem statement.
def training_hours_first_15_days : ℕ := 15 * 5
def training_hours_next_15_days : ℕ := (15 - 3) * (4 + 3)
def training_hours_next_12_days : ℕ := (12 - 2) * (4 + 3)

-- Prove that the total training hours equals 229.
theorem thomas_total_training_hours : 
  training_hours_first_15_days + training_hours_next_15_days + training_hours_next_12_days = 229 :=
by
  -- conditions as defined
  let t1 := 15 * 5
  let t2 := (15 - 3) * (4 + 3)
  let t3 := (12 - 2) * (4 + 3)
  show t1 + t2 + t3 = 229
  sorry

end NUMINAMATH_GPT_thomas_total_training_hours_l1156_115674


namespace NUMINAMATH_GPT_max_garden_area_l1156_115693

-- Definitions of conditions
def shorter_side (s : ℕ) := s
def longer_side (s : ℕ) := 2 * s
def total_perimeter (s : ℕ) := 2 * shorter_side s + 2 * longer_side s 
def garden_area (s : ℕ) := shorter_side s * longer_side s

-- Theorem with given conditions and conclusion to be proven
theorem max_garden_area (s : ℕ) (h_perimeter : total_perimeter s = 480) : garden_area s = 12800 :=
by
  sorry

end NUMINAMATH_GPT_max_garden_area_l1156_115693


namespace NUMINAMATH_GPT_eval_x2_sub_y2_l1156_115694

theorem eval_x2_sub_y2 (x y : ℝ) (h1 : x + y = 10) (h2 : 2 * x + y = 13) : x^2 - y^2 = -40 := by
  sorry

end NUMINAMATH_GPT_eval_x2_sub_y2_l1156_115694


namespace NUMINAMATH_GPT_dream_clock_time_condition_l1156_115624

theorem dream_clock_time_condition (x : ℝ) (h1 : 0 < x) (h2 : x < 1)
  (h3 : (120 + 0.5 * 60 * x) = (240 - 6 * 60 * x)) :
  (4 + x) = 4 + 36 + 12 / 13 := by sorry

end NUMINAMATH_GPT_dream_clock_time_condition_l1156_115624


namespace NUMINAMATH_GPT_tangent_slope_at_one_l1156_115658

def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem tangent_slope_at_one : f' 1 = 2 := by
  sorry

end NUMINAMATH_GPT_tangent_slope_at_one_l1156_115658


namespace NUMINAMATH_GPT_value_of_b_l1156_115669

theorem value_of_b (b : ℝ) : 
  (∃ (x : ℝ), x^2 + b * x - 45 = 0 ∧ x = -4) →
  b = -29 / 4 :=
by
  -- Introduce the condition and rewrite it properly
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  -- Proceed with assumption that we have the condition and need to prove the statement
  sorry

end NUMINAMATH_GPT_value_of_b_l1156_115669


namespace NUMINAMATH_GPT_find_m_if_parallel_l1156_115679

theorem find_m_if_parallel 
  (m : ℚ) 
  (a : ℚ × ℚ := (-2, 3)) 
  (b : ℚ × ℚ := (1, m - 3/2)) 
  (h : ∃ k : ℚ, (a.1 = k * b.1) ∧ (a.2 = k * b.2)) : 
  m = 0 := 
  sorry

end NUMINAMATH_GPT_find_m_if_parallel_l1156_115679


namespace NUMINAMATH_GPT_elaine_rent_percentage_l1156_115673

theorem elaine_rent_percentage (E : ℝ) (P : ℝ) 
  (h1 : E > 0) 
  (h2 : P > 0) 
  (h3 : 0.25 * 1.15 * E = 1.4375 * (P / 100) * E) : 
  P = 20 := 
sorry

end NUMINAMATH_GPT_elaine_rent_percentage_l1156_115673


namespace NUMINAMATH_GPT_new_player_weight_l1156_115680

theorem new_player_weight 
  (original_players : ℕ)
  (original_avg_weight : ℝ)
  (new_players : ℕ)
  (new_avg_weight : ℝ)
  (new_total_weight : ℝ) :
  original_players = 20 →
  original_avg_weight = 180 →
  new_players = 21 →
  new_avg_weight = 181.42857142857142 →
  new_total_weight = 3810 →
  (new_total_weight - original_players * original_avg_weight) = 210 :=
by
  intros
  sorry

end NUMINAMATH_GPT_new_player_weight_l1156_115680


namespace NUMINAMATH_GPT_largest_number_l1156_115633

theorem largest_number (n : ℕ) (digits : List ℕ) (h_digits : ∀ d ∈ digits, d = 5 ∨ d = 3 ∨ d = 1) (h_sum : digits.sum = 15) : n = 555 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_l1156_115633


namespace NUMINAMATH_GPT_desired_markup_percentage_l1156_115690

theorem desired_markup_percentage
  (initial_price : ℝ) (markup_rate : ℝ) (wholesale_price : ℝ) (additional_increase : ℝ) 
  (h1 : initial_price = wholesale_price * (1 + markup_rate)) 
  (h2 : initial_price = 34) 
  (h3 : markup_rate = 0.70) 
  (h4 : additional_increase = 6) 
  : ( (initial_price + additional_increase - wholesale_price) / wholesale_price * 100 ) = 100 := 
by
  sorry

end NUMINAMATH_GPT_desired_markup_percentage_l1156_115690


namespace NUMINAMATH_GPT_incorrect_equation_a_neq_b_l1156_115622

theorem incorrect_equation_a_neq_b (a b : ℝ) (h : a ≠ b) : a - b ≠ b - a :=
  sorry

end NUMINAMATH_GPT_incorrect_equation_a_neq_b_l1156_115622


namespace NUMINAMATH_GPT_Matias_sales_l1156_115660

def books_sold (Tuesday Wednesday Thursday : Nat) : Prop :=
  Tuesday = 7 ∧ 
  Wednesday = 3 * Tuesday ∧ 
  Thursday = 3 * Wednesday ∧ 
  Tuesday + Wednesday + Thursday = 91

theorem Matias_sales
  (Tuesday Wednesday Thursday : Nat) :
  books_sold Tuesday Wednesday Thursday := by
  sorry

end NUMINAMATH_GPT_Matias_sales_l1156_115660


namespace NUMINAMATH_GPT_compute_gf3_l1156_115681

def f (x : ℝ) : ℝ := x^3 - 3
def g (x : ℝ) : ℝ := 2 * x^2 - x + 4

theorem compute_gf3 : g (f 3) = 1132 := 
by 
  sorry

end NUMINAMATH_GPT_compute_gf3_l1156_115681


namespace NUMINAMATH_GPT_solve_alcohol_mixture_problem_l1156_115670

theorem solve_alcohol_mixture_problem (x y : ℝ) 
(h1 : x + y = 18) 
(h2 : 0.75 * x + 0.15 * y = 9) 
: x = 10.5 ∧ y = 7.5 :=
by 
  sorry

end NUMINAMATH_GPT_solve_alcohol_mixture_problem_l1156_115670


namespace NUMINAMATH_GPT_sufficient_not_necessary_perpendicular_l1156_115612

theorem sufficient_not_necessary_perpendicular (a : ℝ) :
  (∀ x y : ℝ, (a + 2) * x + 3 * a * y + 1 = 0 ∧
              (a - 2) * x + (a + 2) * y - 3 = 0 → false) ↔ a = -2 :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_perpendicular_l1156_115612


namespace NUMINAMATH_GPT_peter_total_miles_l1156_115651

-- Definitions based on the conditions
def minutes_per_mile : ℝ := 20
def miles_walked_already : ℝ := 1
def additional_minutes : ℝ := 30

-- The value we want to prove
def total_miles_to_walk : ℝ := 2.5

-- Theorem statement corresponding to the proof problem
theorem peter_total_miles :
  (additional_minutes / minutes_per_mile) + miles_walked_already = total_miles_to_walk :=
sorry

end NUMINAMATH_GPT_peter_total_miles_l1156_115651


namespace NUMINAMATH_GPT_number_of_books_bought_l1156_115642

def initial_books : ℕ := 35
def books_given_away : ℕ := 12
def final_books : ℕ := 56

theorem number_of_books_bought : initial_books - books_given_away + (final_books - (initial_books - books_given_away)) = final_books :=
by
  sorry

end NUMINAMATH_GPT_number_of_books_bought_l1156_115642


namespace NUMINAMATH_GPT_complex_magnitude_addition_l1156_115638

theorem complex_magnitude_addition :
  (Complex.abs (3 / 4 - 3 * Complex.I) + 5 / 12) = (9 * Real.sqrt 17 + 5) / 12 := 
  sorry

end NUMINAMATH_GPT_complex_magnitude_addition_l1156_115638


namespace NUMINAMATH_GPT_combined_weight_of_Leo_and_Kendra_l1156_115636

theorem combined_weight_of_Leo_and_Kendra :
  ∃ (K : ℝ), (92 + K = 160) ∧ (102 = 1.5 * K) :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_of_Leo_and_Kendra_l1156_115636


namespace NUMINAMATH_GPT_train_speed_l1156_115646

theorem train_speed (v : ℝ) 
  (h1 : 50 * 2.5 + v * 2.5 = 285) : v = 64 := 
by
  -- h1 unfolds conditions into the mathematical equation
  -- here we would have the proof steps, adding a "sorry" to skip proof steps.
  sorry

end NUMINAMATH_GPT_train_speed_l1156_115646


namespace NUMINAMATH_GPT_susan_annual_percentage_increase_l1156_115668

theorem susan_annual_percentage_increase :
  let initial_jerry := 14400
  let initial_susan := 6250
  let jerry_first_year := initial_jerry * (6 / 5 : ℝ)
  let jerry_second_year := jerry_first_year * (9 / 10 : ℝ)
  let jerry_third_year := jerry_second_year * (6 / 5 : ℝ)
  jerry_third_year = 18662.40 →
  (initial_susan : ℝ) * (1 + r)^3 = 18662.40 →
  r = 0.44 :=
by {
  sorry
}

end NUMINAMATH_GPT_susan_annual_percentage_increase_l1156_115668


namespace NUMINAMATH_GPT_percentage_of_students_absent_l1156_115604

theorem percentage_of_students_absent (total_students : ℕ) (students_present : ℕ) 
(h_total : total_students = 50) (h_present : students_present = 43)
(absent_students := total_students - students_present) :
((absent_students : ℝ) / total_students) * 100 = 14 :=
by sorry

end NUMINAMATH_GPT_percentage_of_students_absent_l1156_115604


namespace NUMINAMATH_GPT_gcd_of_8247_13619_29826_l1156_115653

theorem gcd_of_8247_13619_29826 : Nat.gcd (Nat.gcd 8247 13619) 29826 = 3 := 
sorry

end NUMINAMATH_GPT_gcd_of_8247_13619_29826_l1156_115653


namespace NUMINAMATH_GPT_difference_between_greatest_and_smallest_S_l1156_115631

-- Conditions
def num_students := 47
def rows := 6
def columns := 8

-- The definition of position value calculation
def position_value (i j m n : ℕ) := i - m + (j - n)

-- The definition of S
def S (initial_empty final_empty : (ℕ × ℕ)) : ℤ :=
  let (i_empty, j_empty) := initial_empty
  let (i'_empty, j'_empty) := final_empty
  (i'_empty + j'_empty) - (i_empty + j_empty)

-- Main statement
theorem difference_between_greatest_and_smallest_S :
  let max_S := S (1, 1) (6, 8)
  let min_S := S (6, 8) (1, 1)
  max_S - min_S = 24 :=
sorry

end NUMINAMATH_GPT_difference_between_greatest_and_smallest_S_l1156_115631


namespace NUMINAMATH_GPT_intersection_is_23_l1156_115687

open Set

def setA : Set ℤ := {1, 2, 3, 4}
def setB : Set ℤ := {x | 2 ≤ x ∧ x ≤ 3}

theorem intersection_is_23 : setA ∩ setB = {2, 3} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_is_23_l1156_115687


namespace NUMINAMATH_GPT_toothpicks_needed_base_1001_l1156_115685

-- Define the number of small triangles at the base of the larger triangle
def base_triangle_count := 1001

-- Define the total number of small triangles using the sum of the first 'n' natural numbers
def total_small_triangles (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Calculate the total number of sides for all triangles if there was no sharing
def total_sides (n : ℕ) : ℕ :=
  3 * total_small_triangles n

-- Calculate the number of shared toothpicks
def shared_toothpicks (n : ℕ) : ℕ :=
  total_sides n / 2

-- Calculate the number of unshared perimeter toothpicks
def unshared_perimeter_toothpicks (n : ℕ) : ℕ :=
  3 * n

-- Calculate the total number of toothpicks required
def total_toothpicks (n : ℕ) : ℕ :=
  shared_toothpicks n + unshared_perimeter_toothpicks n

-- Prove that the total toothpicks required for the base of 1001 small triangles is 755255
theorem toothpicks_needed_base_1001 : total_toothpicks base_triangle_count = 755255 :=
by {
  sorry
}

end NUMINAMATH_GPT_toothpicks_needed_base_1001_l1156_115685


namespace NUMINAMATH_GPT_usual_time_to_catch_bus_l1156_115634

variable {S T T' D : ℝ}

theorem usual_time_to_catch_bus (h1 : D = S * T)
  (h2 : D = (4 / 5) * S * T')
  (h3 : T' = T + 4) : T = 16 := by
  sorry

end NUMINAMATH_GPT_usual_time_to_catch_bus_l1156_115634


namespace NUMINAMATH_GPT_find_sixth_number_l1156_115600

theorem find_sixth_number 
  (A : ℕ → ℝ)
  (h1 : ((A 1 + A 2 + A 3 + A 4 + A 5 + A 6 + A 7 + A 8 + A 9 + A 10 + A 11) / 11 = 60))
  (h2 : ((A 1 + A 2 + A 3 + A 4 + A 5 + A 6) / 6 = 58))
  (h3 : ((A 6 + A 7 + A 8 + A 9 + A 10 + A 11) / 6 = 65)) 
  : A 6 = 78 :=
by
  sorry

end NUMINAMATH_GPT_find_sixth_number_l1156_115600


namespace NUMINAMATH_GPT_angle_B_in_arithmetic_sequence_l1156_115619

theorem angle_B_in_arithmetic_sequence (A B C : ℝ) (h_triangle_sum : A + B + C = 180) (h_arithmetic_sequence : 2 * B = A + C) : B = 60 := 
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_angle_B_in_arithmetic_sequence_l1156_115619


namespace NUMINAMATH_GPT_x_n_squared_leq_2007_l1156_115629

def recurrence (x y : ℕ → ℝ) : Prop :=
  x 0 = 1 ∧ y 0 = 2007 ∧
  ∀ n, x (n + 1) = x n - (x n * y n + x (n + 1) * y (n + 1) - 2) * (y n + y (n + 1)) ∧
       y (n + 1) = y n - (x n * y n + x (n + 1) * y (n + 1) - 2) * (x n + x (n + 1))

theorem x_n_squared_leq_2007 (x y : ℕ → ℝ) (h : recurrence x y) : ∀ n, x n ^ 2 ≤ 2007 :=
by sorry

end NUMINAMATH_GPT_x_n_squared_leq_2007_l1156_115629


namespace NUMINAMATH_GPT_train_speed_l1156_115643

theorem train_speed (length time_speed: ℝ) (h1 : length = 400) (h2 : time_speed = 16) : length / time_speed = 25 := 
by
    sorry

end NUMINAMATH_GPT_train_speed_l1156_115643


namespace NUMINAMATH_GPT_range_of_m_l1156_115639

-- Define points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (2, -2)

-- Define the line equation as a predicate
def line_l (m : ℝ) (p : ℝ × ℝ) : Prop := p.1 + m * p.2 + m = 0

-- Define the condition for the line intersecting the segment AB
def intersects_segment_AB (m : ℝ) : Prop :=
  let P : ℝ × ℝ := (0, -1)
  let k_PA := (P.2 - A.2) / (P.1 - A.1) -- Slope of PA
  let k_PB := (P.2 - B.2) / (P.1 - B.1) -- Slope of PB
  (k_PA <= -1 / m) ∧ (-1 / m <= k_PB)

-- State the theorem
theorem range_of_m : ∀ (m : ℝ), intersects_segment_AB m → (1/2 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_GPT_range_of_m_l1156_115639


namespace NUMINAMATH_GPT_additional_people_needed_l1156_115647

-- Definitions corresponding to the given conditions
def person_hours (n : ℕ) (t : ℕ) : ℕ := n * t
def initial_people : ℕ := 8
def initial_time : ℕ := 10
def total_person_hours := person_hours initial_people initial_time

-- Lean statement of the problem
theorem additional_people_needed (new_time : ℕ) (new_people : ℕ) : 
  new_time = 5 → person_hours new_people new_time = total_person_hours → new_people - initial_people = 8 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_additional_people_needed_l1156_115647


namespace NUMINAMATH_GPT_quadratic_real_roots_condition_l1156_115621

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 * x - 1 = 0) ↔ (a ≥ -4 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_GPT_quadratic_real_roots_condition_l1156_115621


namespace NUMINAMATH_GPT_one_hundred_fiftieth_digit_l1156_115664

theorem one_hundred_fiftieth_digit : (0.135 : Real) * 10^150 = 1 :=
by
  sorry

end NUMINAMATH_GPT_one_hundred_fiftieth_digit_l1156_115664


namespace NUMINAMATH_GPT_distinct_zeros_abs_minus_one_l1156_115652

theorem distinct_zeros_abs_minus_one : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (|x₁| - 1 = 0) ∧ (|x₂| - 1 = 0) := 
by
  sorry

end NUMINAMATH_GPT_distinct_zeros_abs_minus_one_l1156_115652


namespace NUMINAMATH_GPT_Mary_paid_on_Tuesday_l1156_115695

theorem Mary_paid_on_Tuesday 
  (credit_limit total_spent paid_on_thursday remaining_payment paid_on_tuesday : ℝ)
  (h1 : credit_limit = 100)
  (h2 : total_spent = credit_limit)
  (h3 : paid_on_thursday = 23)
  (h4 : remaining_payment = 62)
  (h5 : total_spent = paid_on_thursday + remaining_payment + paid_on_tuesday) :
  paid_on_tuesday = 15 :=
sorry

end NUMINAMATH_GPT_Mary_paid_on_Tuesday_l1156_115695


namespace NUMINAMATH_GPT_vans_needed_for_trip_l1156_115628

theorem vans_needed_for_trip (total_people : ℕ) (van_capacity : ℕ) (h_total_people : total_people = 24) (h_van_capacity : van_capacity = 8) : ℕ :=
  let exact_vans := total_people / van_capacity
  let vans_needed := if total_people % van_capacity = 0 then exact_vans else exact_vans + 1
  have h_exact : exact_vans = 3 := by sorry
  have h_vans_needed : vans_needed = 4 := by sorry
  vans_needed

end NUMINAMATH_GPT_vans_needed_for_trip_l1156_115628


namespace NUMINAMATH_GPT_number_of_solutions_l1156_115630

theorem number_of_solutions (θ : ℝ) (h : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  2 - 4 * Real.sin (2 * θ) + 3 * Real.cos (4 * θ) = 0 → 
  ∃ s : Fin 9, s.val = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l1156_115630


namespace NUMINAMATH_GPT_sarah_probability_l1156_115657

noncomputable def probability_odd_product_less_than_20 : ℚ :=
  let total_possibilities := 36
  let favorable_pairs := [(1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (3, 5), (5, 1), (5, 3)]
  let favorable_count := favorable_pairs.length
  let probability := favorable_count / total_possibilities
  probability

theorem sarah_probability : probability_odd_product_less_than_20 = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sarah_probability_l1156_115657


namespace NUMINAMATH_GPT_form_of_reasoning_is_wrong_l1156_115684

-- Let's define the conditions
def some_rat_nums_are_proper_fractions : Prop :=
  ∃ q : ℚ, (q.num : ℤ) ≠ q.den ∧ (q.den : ℤ) ≠ 1 ∧ q.den ≠ 0

def integers_are_rational_numbers : Prop :=
  ∀ n : ℤ, ∃ q : ℚ, q = n

-- The major premise of the syllogism
def major_premise := some_rat_nums_are_proper_fractions

-- The minor premise of the syllogism
def minor_premise := integers_are_rational_numbers

-- The conclusion of the syllogism
def conclusion := ∀ n : ℤ, ∃ q : ℚ, (q.num : ℤ) ≠ q.den ∧ (q.den : ℤ) ≠ 1 ∧ q.den ≠ 0

-- We need to prove that the form of reasoning is wrong
theorem form_of_reasoning_is_wrong (H1 : major_premise) (H2 : minor_premise) : ¬ conclusion :=
by
  sorry -- proof to be filled in

end NUMINAMATH_GPT_form_of_reasoning_is_wrong_l1156_115684


namespace NUMINAMATH_GPT_maximum_value_of_chords_l1156_115609

noncomputable def max_sum_of_chords (r : ℝ) (h1 : r = 5) 
(h2 : ∃ PA PB PC : ℝ, PA = 2 * PB ∧ PA^2 + PB^2 + PC^2 = (2 * r)^2) : ℝ := 
  6 * Real.sqrt 10

theorem maximum_value_of_chords (P : Point) (r : ℝ) (h1 : r = 5) 
(h2 : ∃ PA PB PC : ℝ, PA = 2 * PB ∧ PA^2 + PB^2 + PC^2 = (2 * r)^2) : 
  PA + PB + PC ≤ 6 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_chords_l1156_115609


namespace NUMINAMATH_GPT_sum_invariant_under_permutation_l1156_115678

theorem sum_invariant_under_permutation (b : List ℝ) (σ : List ℕ) (hσ : σ.Perm (List.range b.length)) :
  (List.sum b) = (List.sum (σ.map (b.get!))) := by
  sorry

end NUMINAMATH_GPT_sum_invariant_under_permutation_l1156_115678


namespace NUMINAMATH_GPT_min_value_eq_216_l1156_115667

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c)

theorem min_value_eq_216 {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  min_value a b c = 216 :=
sorry

end NUMINAMATH_GPT_min_value_eq_216_l1156_115667


namespace NUMINAMATH_GPT_neg_p_equivalent_to_forall_x2_ge_1_l1156_115661

open Classical

variable {x : ℝ}

-- Definition of the original proposition p
def p : Prop := ∃ (x : ℝ), x^2 < 1

-- The negation of the proposition p
def not_p : Prop := ∀ (x : ℝ), x^2 ≥ 1

-- The theorem stating the equivalence
theorem neg_p_equivalent_to_forall_x2_ge_1 : ¬ p ↔ not_p := by
  sorry

end NUMINAMATH_GPT_neg_p_equivalent_to_forall_x2_ge_1_l1156_115661


namespace NUMINAMATH_GPT_complement_of_supplement_of_30_degrees_l1156_115618

def supplementary_angle (x : ℕ) : ℕ := 180 - x
def complementary_angle (x : ℕ) : ℕ := if x > 90 then x - 90 else 90 - x

theorem complement_of_supplement_of_30_degrees : complementary_angle (supplementary_angle 30) = 60 := by
  sorry

end NUMINAMATH_GPT_complement_of_supplement_of_30_degrees_l1156_115618


namespace NUMINAMATH_GPT_simplest_common_denominator_of_fractions_l1156_115676

noncomputable def simplestCommonDenominator (a b : ℕ) (x y : ℕ) : ℕ := 6 * (x ^ 2) * (y ^ 3)

theorem simplest_common_denominator_of_fractions :
  simplestCommonDenominator 2 6 x y = 6 * x^2 * y^3 :=
by
  sorry

end NUMINAMATH_GPT_simplest_common_denominator_of_fractions_l1156_115676


namespace NUMINAMATH_GPT_probability_not_pulling_prize_twice_l1156_115675

theorem probability_not_pulling_prize_twice
  (favorable : ℕ)
  (unfavorable : ℕ)
  (total : ℕ := favorable + unfavorable)
  (P_prize : ℚ := favorable / total)
  (P_not_prize : ℚ := 1 - P_prize)
  (P_not_prize_twice : ℚ := P_not_prize * P_not_prize) :
  P_not_prize_twice = 36 / 121 :=
by
  have favorable : ℕ := 5
  have unfavorable : ℕ := 6
  have total : ℕ := favorable + unfavorable
  have P_prize : ℚ := favorable / total
  have P_not_prize : ℚ := 1 - P_prize
  have P_not_prize_twice : ℚ := P_not_prize * P_not_prize
  sorry

end NUMINAMATH_GPT_probability_not_pulling_prize_twice_l1156_115675


namespace NUMINAMATH_GPT_solve_fraction_l1156_115611

variables (w x y : ℝ)

-- Conditions
def condition1 := w / x = 2 / 3
def condition2 := w / y = 6 / 15

-- Statement
theorem solve_fraction (h1 : condition1 w x) (h2 : condition2 w y) : (x + y) / y = 8 / 5 :=
sorry

end NUMINAMATH_GPT_solve_fraction_l1156_115611


namespace NUMINAMATH_GPT_max_intersections_pentagon_triangle_max_intersections_pentagon_quadrilateral_l1156_115617

-- Define the pentagon and various other polygons
inductive PolygonType
| pentagon
| triangle
| quadrilateral

-- Define a function that calculates the maximum number of intersections
def max_intersections (K L : PolygonType) : ℕ :=
  match K, L with
  | PolygonType.pentagon, PolygonType.triangle => 10
  | PolygonType.pentagon, PolygonType.quadrilateral => 16
  | _, _ => 0  -- We only care about the cases specified in our problem

-- Theorem a): When L is a triangle, the intersections should be 10
theorem max_intersections_pentagon_triangle : max_intersections PolygonType.pentagon PolygonType.triangle = 10 :=
  by 
  -- provide proof here, but currently it is skipped with sorry
  sorry

-- Theorem b): When L is a quadrilateral, the intersections should be 16
theorem max_intersections_pentagon_quadrilateral : max_intersections PolygonType.pentagon PolygonType.quadrilateral = 16 :=
  by
  -- provide proof here, but currently it is skipped with sorry
  sorry

end NUMINAMATH_GPT_max_intersections_pentagon_triangle_max_intersections_pentagon_quadrilateral_l1156_115617


namespace NUMINAMATH_GPT_identity_holds_for_all_a_b_l1156_115682

theorem identity_holds_for_all_a_b (a b : ℝ) :
  let x := a + 5 * b
  let y := 5 * a - b
  let z := 3 * a - 2 * b
  let t := 2 * a + 3 * b
  x^2 + y^2 = 2 * (z^2 + t^2) :=
by {
  let x := a + 5 * b
  let y := 5 * a - b
  let z := 3 * a - 2 * b
  let t := 2 * a + 3 * b
  sorry
}

end NUMINAMATH_GPT_identity_holds_for_all_a_b_l1156_115682


namespace NUMINAMATH_GPT_percent_of_male_literate_l1156_115648

noncomputable def female_percentage : ℝ := 0.6
noncomputable def total_employees : ℕ := 1500
noncomputable def literate_percentage : ℝ := 0.62
noncomputable def literate_female_employees : ℕ := 630

theorem percent_of_male_literate :
  let total_females := (female_percentage * total_employees)
  let total_males := total_employees - total_females
  let total_literate := literate_percentage * total_employees
  let literate_male_employees := total_literate - literate_female_employees
  let male_literate_percentage := (literate_male_employees / total_males) * 100
  male_literate_percentage = 50 := by
  sorry

end NUMINAMATH_GPT_percent_of_male_literate_l1156_115648


namespace NUMINAMATH_GPT_movie_ticket_final_price_l1156_115666

noncomputable def final_ticket_price (initial_price : ℝ) : ℝ :=
  let price_year_1 := initial_price * 1.12
  let price_year_2 := price_year_1 * 0.95
  let price_year_3 := price_year_2 * 1.08
  let price_year_4 := price_year_3 * 0.96
  let price_year_5 := price_year_4 * 1.06
  let price_after_tax := price_year_5 * 1.07
  let final_price := price_after_tax * 0.90
  final_price

theorem movie_ticket_final_price :
  final_ticket_price 100 = 112.61 := by
  sorry

end NUMINAMATH_GPT_movie_ticket_final_price_l1156_115666


namespace NUMINAMATH_GPT_haley_marbles_l1156_115665

theorem haley_marbles (boys : ℕ) (marbles_per_boy : ℕ) (total_marbles : ℕ) 
  (h1 : boys = 11) (h2 : marbles_per_boy = 9) : total_marbles = 99 :=
by
  sorry

end NUMINAMATH_GPT_haley_marbles_l1156_115665


namespace NUMINAMATH_GPT_boys_count_l1156_115655

theorem boys_count (B G : ℕ) (h1 : B + G = 41) (h2 : 12 * B + 8 * G = 460) : B = 33 := 
by
  sorry

end NUMINAMATH_GPT_boys_count_l1156_115655


namespace NUMINAMATH_GPT_find_m_l1156_115697

theorem find_m (m : ℝ) (x : ℝ) (y : ℝ) (h_eq_parabola : y = m * x^2)
  (h_directrix : y = 1 / 8) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1156_115697


namespace NUMINAMATH_GPT_soy_sauce_bottle_size_l1156_115610

theorem soy_sauce_bottle_size 
  (ounces_per_cup : ℕ)
  (cups_recipe1 : ℕ)
  (cups_recipe2 : ℕ)
  (cups_recipe3 : ℕ)
  (number_of_bottles : ℕ)
  (total_ounces_needed : ℕ)
  (ounces_per_bottle : ℕ) :
  ounces_per_cup = 8 →
  cups_recipe1 = 2 →
  cups_recipe2 = 1 →
  cups_recipe3 = 3 →
  number_of_bottles = 3 →
  total_ounces_needed = (cups_recipe1 + cups_recipe2 + cups_recipe3) * ounces_per_cup →
  ounces_per_bottle = total_ounces_needed / number_of_bottles →
  ounces_per_bottle = 16 :=
by
  sorry

end NUMINAMATH_GPT_soy_sauce_bottle_size_l1156_115610


namespace NUMINAMATH_GPT_rhombus_area_and_perimeter_l1156_115696

theorem rhombus_area_and_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 26) :
  let area := (d1 * d2) / 2
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let perimeter := 4 * s
  area = 234 ∧ perimeter = 20 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_rhombus_area_and_perimeter_l1156_115696


namespace NUMINAMATH_GPT_smallest_possible_integer_l1156_115606

theorem smallest_possible_integer (a b : ℤ)
  (a_lt_10 : a < 10)
  (b_lt_10 : b < 10)
  (a_lt_b : a < b)
  (sum_eq_45 : a + b + 32 = 45)
  : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_integer_l1156_115606


namespace NUMINAMATH_GPT_horner_rule_v3_is_36_l1156_115616

def f (x : ℤ) : ℤ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_rule_v3_is_36 :
  let v0 := 1;
  let v1 := v0 * 3 + 0;
  let v2 := v1 * 3 + 2;
  let v3 := v2 * 3 + 3;
  v3 = 36 := 
by
  sorry

end NUMINAMATH_GPT_horner_rule_v3_is_36_l1156_115616


namespace NUMINAMATH_GPT_day_of_month_l1156_115623

/--
The 25th day of a particular month is a Monday. 
We need to prove that the 1st day of that month is a Friday.
-/
theorem day_of_month (h : (25 % 7 = 1)) : (1 % 7 = 5) :=
sorry

end NUMINAMATH_GPT_day_of_month_l1156_115623


namespace NUMINAMATH_GPT_find_tan_beta_l1156_115615

variable (α β : ℝ)

def condition1 : Prop := Real.tan α = 3
def condition2 : Prop := Real.tan (α + β) = 2

theorem find_tan_beta (h1 : condition1 α) (h2 : condition2 α β) : Real.tan β = -1 / 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_tan_beta_l1156_115615


namespace NUMINAMATH_GPT_general_formula_a_n_T_n_greater_than_S_n_l1156_115672

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end NUMINAMATH_GPT_general_formula_a_n_T_n_greater_than_S_n_l1156_115672


namespace NUMINAMATH_GPT_stockings_total_cost_l1156_115637

-- Defining the conditions
def total_stockings : ℕ := 9
def original_price_per_stocking : ℝ := 20
def discount_rate : ℝ := 0.10
def monogramming_cost_per_stocking : ℝ := 5

-- Calculate the total cost of stockings
theorem stockings_total_cost :
  total_stockings * ((original_price_per_stocking * (1 - discount_rate)) + monogramming_cost_per_stocking) = 207 := 
by
  sorry

end NUMINAMATH_GPT_stockings_total_cost_l1156_115637


namespace NUMINAMATH_GPT_hyperbola_midpoint_l1156_115601

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end NUMINAMATH_GPT_hyperbola_midpoint_l1156_115601


namespace NUMINAMATH_GPT_money_distribution_l1156_115635

-- Declare the variables and the conditions as hypotheses
theorem money_distribution (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 40) :
  B + C = 340 :=
by
  sorry

end NUMINAMATH_GPT_money_distribution_l1156_115635


namespace NUMINAMATH_GPT_find_third_root_l1156_115677

theorem find_third_root (a b : ℚ) 
  (h1 : a * 1^3 + (a + 3 * b) * 1^2 + (b - 4 * a) * 1 + (6 - a) = 0)
  (h2 : a * (-3)^3 + (a + 3 * b) * (-3)^2 + (b - 4 * a) * (-3) + (6 - a) = 0)
  : ∃ c : ℚ, c = 7 / 13 :=
sorry

end NUMINAMATH_GPT_find_third_root_l1156_115677


namespace NUMINAMATH_GPT_lateral_surface_area_of_cone_l1156_115654

theorem lateral_surface_area_of_cone (r h : ℝ) (r_is_4 : r = 4) (h_is_3 : h = 3) :
  ∃ A : ℝ, A = 20 * Real.pi := by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cone_l1156_115654


namespace NUMINAMATH_GPT_significant_figures_and_precision_l1156_115649

-- Definition of the function to count significant figures
def significant_figures (n : Float) : Nat :=
  -- Implementation of a function that counts significant figures
  -- Skipping actual implementation, assuming it is correct.
  sorry

-- Definition of the function to determine precision
def precision (n : Float) : String :=
  -- Implementation of a function that returns the precision
  -- Skipping actual implementation, assuming it is correct.
  sorry

-- The target number
def num := 0.03020

-- The properties of the number 0.03020
theorem significant_figures_and_precision :
  significant_figures num = 4 ∧ precision num = "ten-thousandth" :=
by
  sorry

end NUMINAMATH_GPT_significant_figures_and_precision_l1156_115649


namespace NUMINAMATH_GPT_infinite_representable_and_nonrepresentable_terms_l1156_115605

def a (n : ℕ) : ℕ :=
  2^n + 2^(n / 2)

def is_representable (k : ℕ) : Prop :=   
  -- A nonnegative integer is defined to be representable if it can
  -- be expressed as a sum of distinct terms from the sequence a(n).
  sorry  -- Definition will depend on the specific notion of representability

theorem infinite_representable_and_nonrepresentable_terms :
  (∃ᶠ n in at_top, is_representable (a n)) ∧ (∃ᶠ n in at_top, ¬is_representable (a n)) :=
sorry  -- This is the main theorem claiming infinitely many representable and non-representable terms.

end NUMINAMATH_GPT_infinite_representable_and_nonrepresentable_terms_l1156_115605


namespace NUMINAMATH_GPT_chocolate_game_winner_l1156_115607

-- Definitions of conditions for the problem
def chocolate_bar (m n : ℕ) := m * n

-- Theorem statement with conditions and conclusion
theorem chocolate_game_winner (m n : ℕ) (h1 : chocolate_bar m n = 48) : 
  ( ∃ first_player_wins : true, true) :=
by sorry

end NUMINAMATH_GPT_chocolate_game_winner_l1156_115607


namespace NUMINAMATH_GPT_find_c2013_l1156_115645

theorem find_c2013 :
  ∀ (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ),
    (a 1 = 3) →
    (b 1 = 3) →
    (∀ n : ℕ, 1 ≤ n → a (n+1) - a n = 3) →
    (∀ n : ℕ, 1 ≤ n → b (n+1) = 3 * b n) →
    (∀ n : ℕ, c n = b (a n)) →
    c 2013 = 27^2013 := by
  sorry

end NUMINAMATH_GPT_find_c2013_l1156_115645


namespace NUMINAMATH_GPT_total_quantity_before_adding_water_l1156_115620

variable (x : ℚ)
variable (milk water : ℚ)
variable (added_water : ℚ)

-- Mixture contains milk and water in the ratio 3:2
def initial_ratio (milk water : ℚ) : Prop := milk / water = 3 / 2

-- Adding 10 liters of water
def added_amount : ℚ := 10

-- New ratio of milk to water becomes 2:3 after adding 10 liters of water
def new_ratio (milk water : ℚ) (added_water : ℚ) : Prop :=
  milk / (water + added_water) = 2 / 3

theorem total_quantity_before_adding_water
  (h_ratio : initial_ratio milk water)
  (h_added : added_water = 10)
  (h_new_ratio : new_ratio milk water added_water) :
  milk + water = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_quantity_before_adding_water_l1156_115620


namespace NUMINAMATH_GPT_triangle_third_side_max_length_l1156_115644

theorem triangle_third_side_max_length (a b : ℕ) (ha : a = 5) (hb : b = 11) : ∃ (c : ℕ), c = 15 ∧ (a + c > b ∧ b + c > a ∧ a + b > c) :=
by 
  sorry

end NUMINAMATH_GPT_triangle_third_side_max_length_l1156_115644


namespace NUMINAMATH_GPT_space_per_bush_l1156_115699

theorem space_per_bush (side_length : ℝ) (num_sides : ℝ) (num_bushes : ℝ) (h1 : side_length = 16) (h2 : num_sides = 3) (h3 : num_bushes = 12) :
  (num_sides * side_length) / num_bushes = 4 :=
by
  sorry

end NUMINAMATH_GPT_space_per_bush_l1156_115699


namespace NUMINAMATH_GPT_alexa_fractions_l1156_115662

theorem alexa_fractions (alexa_days ethans_days : ℕ) 
  (h1 : alexa_days = 9) (h2 : ethans_days = 12) : 
  alexa_days / ethans_days = 3 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_alexa_fractions_l1156_115662


namespace NUMINAMATH_GPT_boxes_with_neither_l1156_115632

-- Definitions based on the conditions given
def total_boxes : Nat := 12
def boxes_with_markers : Nat := 8
def boxes_with_erasers : Nat := 5
def boxes_with_both : Nat := 4

-- The statement we want to prove
theorem boxes_with_neither :
  total_boxes - (boxes_with_markers + boxes_with_erasers - boxes_with_both) = 3 :=
by
  sorry

end NUMINAMATH_GPT_boxes_with_neither_l1156_115632


namespace NUMINAMATH_GPT_carson_giant_slide_rides_l1156_115603

theorem carson_giant_slide_rides :
  let total_hours := 4
  let roller_coaster_wait_time := 30
  let roller_coaster_rides := 4
  let tilt_a_whirl_wait_time := 60
  let tilt_a_whirl_rides := 1
  let giant_slide_wait_time := 15
  -- Convert hours to minutes
  let total_minutes := total_hours * 60
  -- Calculate total wait time for roller coaster
  let roller_coaster_total_wait := roller_coaster_wait_time * roller_coaster_rides
  -- Calculate total wait time for tilt-a-whirl
  let tilt_a_whirl_total_wait := tilt_a_whirl_wait_time * tilt_a_whirl_rides
  -- Calculate total wait time for roller coaster and tilt-a-whirl
  let total_wait := roller_coaster_total_wait + tilt_a_whirl_total_wait
  -- Calculate remaining time
  let remaining_time := total_minutes - total_wait
  -- Calculate how many times Carson can ride the giant slide
  let giant_slide_rides := remaining_time / giant_slide_wait_time
  giant_slide_rides = 4 := by
  let total_hours := 4
  let roller_coaster_wait_time := 30
  let roller_coaster_rides := 4
  let tilt_a_whirl_wait_time := 60
  let tilt_a_whirl_rides := 1
  let giant_slide_wait_time := 15
  let total_minutes := total_hours * 60
  let roller_coaster_total_wait := roller_coaster_wait_time * roller_coaster_rides
  let tilt_a_whirl_total_wait := tilt_a_whirl_wait_time * tilt_a_whirl_rides
  let total_wait := roller_coaster_total_wait + tilt_a_whirl_total_wait
  let remaining_time := total_minutes - total_wait
  let giant_slide_rides := remaining_time / giant_slide_wait_time
  show giant_slide_rides = 4
  sorry

end NUMINAMATH_GPT_carson_giant_slide_rides_l1156_115603


namespace NUMINAMATH_GPT_satisfies_equation_l1156_115688

noncomputable def y (b x : ℝ) : ℝ := (b + x) / (1 + b * x)

theorem satisfies_equation (b x : ℝ) :
  let y_val := y b x
  let y_prime := (1 - b^2) / (1 + b * x)^2
  y_val - x * y_prime = b * (1 + x^2 * y_prime) :=
by
  sorry

end NUMINAMATH_GPT_satisfies_equation_l1156_115688


namespace NUMINAMATH_GPT_hash_fn_triple_40_l1156_115627

def hash_fn (N : ℝ) : ℝ := 0.6 * N + 2

theorem hash_fn_triple_40 : hash_fn (hash_fn (hash_fn 40)) = 12.56 := by
  sorry

end NUMINAMATH_GPT_hash_fn_triple_40_l1156_115627


namespace NUMINAMATH_GPT_hexagonal_pyramid_cross_section_distance_l1156_115689

theorem hexagonal_pyramid_cross_section_distance
  (A1 A2 : ℝ) (distance_between_planes : ℝ)
  (A1_area : A1 = 125 * Real.sqrt 3)
  (A2_area : A2 = 500 * Real.sqrt 3)
  (distance_between_planes_eq : distance_between_planes = 10) :
  ∃ h : ℝ, h = 20 :=
by
  sorry

end NUMINAMATH_GPT_hexagonal_pyramid_cross_section_distance_l1156_115689


namespace NUMINAMATH_GPT_fries_remaining_time_l1156_115641

theorem fries_remaining_time (recommended_time_min : ℕ) (time_in_oven_sec : ℕ)
    (h1 : recommended_time_min = 5)
    (h2 : time_in_oven_sec = 45) :
    (recommended_time_min * 60 - time_in_oven_sec = 255) :=
by
  sorry

end NUMINAMATH_GPT_fries_remaining_time_l1156_115641


namespace NUMINAMATH_GPT_boat_speed_still_water_l1156_115608

variable (V_b V_s : ℝ)

def upstream : Prop := V_b - V_s = 10
def downstream : Prop := V_b + V_s = 40

theorem boat_speed_still_water (h1 : upstream V_b V_s) (h2 : downstream V_b V_s) : V_b = 25 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_still_water_l1156_115608


namespace NUMINAMATH_GPT_point_equidistant_l1156_115640

def A : ℝ × ℝ × ℝ := (10, 0, 0)
def B : ℝ × ℝ × ℝ := (0, -6, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 8)
def D : ℝ × ℝ × ℝ := (0, 0, 0)
def P : ℝ × ℝ × ℝ := (5, -3, 4)

theorem point_equidistant : dist A P = dist B P ∧ dist B P = dist C P ∧ dist C P = dist D P :=
by
  sorry

end NUMINAMATH_GPT_point_equidistant_l1156_115640


namespace NUMINAMATH_GPT_david_completion_time_l1156_115614

theorem david_completion_time :
  (∃ D : ℕ, ∀ t : ℕ, 6 * (1 / D) + 3 * ((1 / D) + (1 / t)) = 1 -> D = 12) :=
sorry

end NUMINAMATH_GPT_david_completion_time_l1156_115614


namespace NUMINAMATH_GPT_fewest_tiles_needed_l1156_115686

def tiles_needed (tile_length tile_width region_length region_width : ℕ) : ℕ :=
  let length_tiles := (region_length + tile_length - 1) / tile_length
  let width_tiles := (region_width + tile_width - 1) / tile_width
  length_tiles * width_tiles

theorem fewest_tiles_needed :
  let tile_length := 2
  let tile_width := 5
  let region_length := 36
  let region_width := 72
  tiles_needed tile_length tile_width region_length region_width = 270 :=
by
  sorry

end NUMINAMATH_GPT_fewest_tiles_needed_l1156_115686


namespace NUMINAMATH_GPT_solve_inequality_l1156_115602

theorem solve_inequality (x : ℝ) : 3 * (x + 1) > 9 → x > 2 :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l1156_115602


namespace NUMINAMATH_GPT_find_f_of_3_l1156_115613

-- Define the function f and its properties
variable {f : ℝ → ℝ}

-- Define the properties given in the problem
axiom f_mono_increasing : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_of_f_minus_exp : ∀ x : ℝ, f (f x - 2^x) = 3

-- The main theorem to prove
theorem find_f_of_3 : f 3 = 9 := 
sorry

end NUMINAMATH_GPT_find_f_of_3_l1156_115613


namespace NUMINAMATH_GPT_first_year_after_2020_with_digit_sum_4_l1156_115663

theorem first_year_after_2020_with_digit_sum_4 :
  ∃ x : ℕ, x > 2020 ∧ (Nat.digits 10 x).sum = 4 ∧ ∀ y : ℕ, y > 2020 ∧ (Nat.digits 10 y).sum = 4 → x ≤ y :=
sorry

end NUMINAMATH_GPT_first_year_after_2020_with_digit_sum_4_l1156_115663


namespace NUMINAMATH_GPT_relationship_y1_y2_l1156_115626

theorem relationship_y1_y2
  (x1 y1 x2 y2 : ℝ)
  (hA : y1 = 3 * x1 + 4)
  (hB : y2 = 3 * x2 + 4)
  (h : x1 < x2) :
  y1 < y2 :=
sorry

end NUMINAMATH_GPT_relationship_y1_y2_l1156_115626


namespace NUMINAMATH_GPT_common_root_and_param_l1156_115692

theorem common_root_and_param :
  ∀ (x : ℤ) (P p : ℚ),
    (P = -((x^2 - x - 2) / (x - 1)) ∧ x ≠ 1) →
    (p = -((x^2 + 2*x - 1) / (x + 2)) ∧ x ≠ -2) →
    (-x + (2 / (x - 1)) = -x + (1 / (x + 2))) →
    x = -5 ∧ p = 14 / 3 :=
by
  intros x P p hP hp hroot
  sorry

end NUMINAMATH_GPT_common_root_and_param_l1156_115692
