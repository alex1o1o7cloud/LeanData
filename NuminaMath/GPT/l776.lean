import Mathlib

namespace NUMINAMATH_GPT_time_after_classes_l776_77608

def time_after_maths : Nat := 60
def time_after_history : Nat := 60 + 90
def time_after_break1 : Nat := time_after_history + 25
def time_after_geography : Nat := time_after_break1 + 45
def time_after_break2 : Nat := time_after_geography + 15
def time_after_science : Nat := time_after_break2 + 75

theorem time_after_classes (start_time : Nat := 12 * 60) : (start_time + time_after_science) % 1440 = 17 * 60 + 10 :=
by
  sorry

end NUMINAMATH_GPT_time_after_classes_l776_77608


namespace NUMINAMATH_GPT_total_pie_eaten_l776_77637

theorem total_pie_eaten (s1 s2 s3 : ℚ) (h1 : s1 = 8/9) (h2 : s2 = 5/6) (h3 : s3 = 2/3) :
  s1 + s2 + s3 = 43/18 := by
  sorry

end NUMINAMATH_GPT_total_pie_eaten_l776_77637


namespace NUMINAMATH_GPT_hexagon_area_correct_l776_77628

structure Point where
  x : ℝ
  y : ℝ

def hexagon : List Point := [
  { x := 0, y := 0 },
  { x := 2, y := 4 },
  { x := 6, y := 4 },
  { x := 8, y := 0 },
  { x := 6, y := -4 },
  { x := 2, y := -4 }
]

def area_of_hexagon (hex : List Point) : ℝ :=
  -- Assume a function that calculates the area of a polygon given a list of vertices
  sorry

theorem hexagon_area_correct : area_of_hexagon hexagon = 16 :=
  sorry

end NUMINAMATH_GPT_hexagon_area_correct_l776_77628


namespace NUMINAMATH_GPT_height_of_frustum_l776_77658

-- Definitions based on the given conditions
def cuts_parallel_to_base (height: ℕ) (ratio: ℕ) : ℕ := 
  height * ratio

-- Define the problem
theorem height_of_frustum 
  (height_smaller_pyramid : ℕ) 
  (ratio_upper_to_lower: ℕ) 
  (h : height_smaller_pyramid = 3) 
  (r : ratio_upper_to_lower = 4) 
  : (cuts_parallel_to_base 3 2) - height_smaller_pyramid = 3 := 
by
  sorry

end NUMINAMATH_GPT_height_of_frustum_l776_77658


namespace NUMINAMATH_GPT_gcd_of_18_and_30_l776_77648

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_of_18_and_30_l776_77648


namespace NUMINAMATH_GPT_perfume_weight_is_six_ounces_l776_77663

def weight_in_pounds (ounces : ℕ) : ℕ := ounces / 16

def initial_weight := 5  -- Initial suitcase weight in pounds
def final_weight := 11   -- Final suitcase weight in pounds
def chocolate := 4       -- Weight of chocolate in pounds
def soap := 2 * 5        -- Weight of 2 bars of soap in ounces
def jam := 2 * 8         -- Weight of 2 jars of jam in ounces

def total_additional_weight :=
  chocolate + (weight_in_pounds soap) + (weight_in_pounds jam)

def perfume_weight_in_pounds := final_weight - initial_weight - total_additional_weight

def perfume_weight_in_ounces := perfume_weight_in_pounds * 16

theorem perfume_weight_is_six_ounces : perfume_weight_in_ounces = 6 := by sorry

end NUMINAMATH_GPT_perfume_weight_is_six_ounces_l776_77663


namespace NUMINAMATH_GPT_total_balls_donated_l776_77678

def num_elem_classes_A := 4
def num_middle_classes_A := 5
def num_elem_classes_B := 5
def num_middle_classes_B := 3
def num_elem_classes_C := 6
def num_middle_classes_C := 4
def balls_per_class := 5

theorem total_balls_donated :
  (num_elem_classes_A + num_middle_classes_A) * balls_per_class +
  (num_elem_classes_B + num_middle_classes_B) * balls_per_class +
  (num_elem_classes_C + num_middle_classes_C) * balls_per_class =
  135 :=
by
  sorry

end NUMINAMATH_GPT_total_balls_donated_l776_77678


namespace NUMINAMATH_GPT_problem_solution_correct_l776_77669

def proposition_p : Prop :=
  ∃ x : ℝ, Real.tan x = 1

def proposition_q : Prop :=
  {x : ℝ | x^2 - 3 * x + 2 < 0} = {x : ℝ | 1 < x ∧ x < 2}

theorem problem_solution_correct :
  (proposition_p ∧ proposition_q) ∧
  (proposition_p ∧ ¬proposition_q) = false ∧
  (¬proposition_p ∨ proposition_q) ∧
  (¬proposition_p ∨ ¬proposition_q) = false :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_correct_l776_77669


namespace NUMINAMATH_GPT_math_proof_problem_l776_77687

theorem math_proof_problem (a : ℝ) : 
  (a^8 / a^4 ≠ a^4) ∧ ((a^2)^3 ≠ a^6) ∧ ((3*a)^3 ≠ 9*a^3) ∧ ((-a)^3 * (-a)^5 = a^8) := 
by 
  sorry

end NUMINAMATH_GPT_math_proof_problem_l776_77687


namespace NUMINAMATH_GPT_area_triangle_ABC_is_correct_l776_77601

noncomputable def radius : ℝ := 4

noncomputable def angleABDiameter : ℝ := 30

noncomputable def ratioAM_MB : ℝ := 2 / 3

theorem area_triangle_ABC_is_correct :
  ∃ (area : ℝ), area = (180 * Real.sqrt 3) / 19 :=
by sorry

end NUMINAMATH_GPT_area_triangle_ABC_is_correct_l776_77601


namespace NUMINAMATH_GPT_equation_true_l776_77626

variables {AB BC CD AD AC BD : ℝ}

theorem equation_true :
  (AD * BC + AB * CD = AC * BD) ∧
  (AD * BC - AB * CD ≠ AC * BD) ∧
  (AB * BC + AC * CD ≠ AC * BD) ∧
  (AB * BC - AC * CD ≠ AC * BD) :=
by
  sorry

end NUMINAMATH_GPT_equation_true_l776_77626


namespace NUMINAMATH_GPT_equation_of_line_l776_77615

theorem equation_of_line 
  (a : ℝ) (h : a < 3) 
  (C : ℝ × ℝ) 
  (hC : C = (-2, 3)) 
  (l_intersects_circle : ∃ A B : ℝ × ℝ, 
    (A.1^2 + A.2^2 + 2 * A.1 - 4 * A.2 + a = 0) ∧ 
    (B.1^2 + B.2^2 + 2 * B.1 - 4 * B.2 + a = 0) ∧ 
    (C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))) : 
  ∃ (m b : ℝ), 
    (m = 1) ∧ 
    (b = -5) ∧ 
    (∀ x y, y - 3 = m * (x + 2) ↔ x - y + 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_line_l776_77615


namespace NUMINAMATH_GPT_possible_sums_of_products_neg11_l776_77631

theorem possible_sums_of_products_neg11 (a b c : ℤ) (h : a * b * c = -11) :
  a + b + c = -9 ∨ a + b + c = 11 ∨ a + b + c = 13 :=
sorry

end NUMINAMATH_GPT_possible_sums_of_products_neg11_l776_77631


namespace NUMINAMATH_GPT_hexagon_side_length_l776_77655

theorem hexagon_side_length (p : ℕ) (s : ℕ) (h₁ : p = 24) (h₂ : s = 6) : p / s = 4 := by
  sorry

end NUMINAMATH_GPT_hexagon_side_length_l776_77655


namespace NUMINAMATH_GPT_sqrt_of_square_of_neg_five_eq_five_l776_77610

theorem sqrt_of_square_of_neg_five_eq_five : Real.sqrt ((-5 : ℤ) ^ 2) = 5 := by
  sorry

end NUMINAMATH_GPT_sqrt_of_square_of_neg_five_eq_five_l776_77610


namespace NUMINAMATH_GPT_shaded_region_area_l776_77689

theorem shaded_region_area
  (n : ℕ) (d : ℝ) 
  (h₁ : n = 25) 
  (h₂ : d = 10) 
  (h₃ : n > 0) : 
  (d^2 / n = 2) ∧ (n * (d^2 / (2 * n)) = 50) :=
by 
  sorry

end NUMINAMATH_GPT_shaded_region_area_l776_77689


namespace NUMINAMATH_GPT_solve_system_l776_77639

noncomputable def f (a b x : ℝ) : ℝ := a^x + b

theorem solve_system (a b : ℝ) :
  (f a b 1 = 4) ∧ (f a b 0 = 2) →
  a = 3 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l776_77639


namespace NUMINAMATH_GPT_find_a_plus_d_l776_77674

variables (a b c d e : ℝ)

theorem find_a_plus_d :
  a + b = 12 ∧ b + c = 9 ∧ c + d = 3 ∧ d + e = 7 ∧ e + a = 10 → a + d = 6 :=
by
  intros h
  have h1 : a + b = 12 := h.1
  have h2 : b + c = 9 := h.2.1
  have h3 : c + d = 3 := h.2.2.1
  have h4 : d + e = 7 := h.2.2.2.1
  have h5 : e + a = 10 := h.2.2.2.2
  sorry

end NUMINAMATH_GPT_find_a_plus_d_l776_77674


namespace NUMINAMATH_GPT_distinct_real_roots_iff_l776_77651

-- Define f(x, a) := |x^2 - a| - x + 2
noncomputable def f (x a : ℝ) : ℝ := abs (x^2 - a) - x + 2

-- The proposition we need to prove
theorem distinct_real_roots_iff (a : ℝ) (h : 0 < a) : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 4 < a :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_iff_l776_77651


namespace NUMINAMATH_GPT_average_speed_is_correct_l776_77632

-- Definitions for the conditions
def speed_first_hour : ℕ := 140
def speed_second_hour : ℕ := 40
def total_distance : ℕ := speed_first_hour + speed_second_hour
def total_time : ℕ := 2

-- The statement we need to prove
theorem average_speed_is_correct : total_distance / total_time = 90 := by
  -- We would place the proof here
  sorry

end NUMINAMATH_GPT_average_speed_is_correct_l776_77632


namespace NUMINAMATH_GPT_add_n_to_constant_l776_77646

theorem add_n_to_constant (y n : ℝ) (h_eq : y^4 - 20 * y + 1 = 22) (h_n : n = 3) : y^4 - 20 * y + 4 = 25 :=
by
  sorry

end NUMINAMATH_GPT_add_n_to_constant_l776_77646


namespace NUMINAMATH_GPT_polynomial_coefficients_l776_77654

theorem polynomial_coefficients (x a₄ a₃ a₂ a₁ a₀ : ℝ) (h : (x - 1)^4 = a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀)
  : a₄ - a₃ + a₂ - a₁ = 15 := by
  sorry

end NUMINAMATH_GPT_polynomial_coefficients_l776_77654


namespace NUMINAMATH_GPT_seashells_total_l776_77635

theorem seashells_total (tim_seashells sally_seashells : ℕ) (ht : tim_seashells = 37) (hs : sally_seashells = 13) :
  tim_seashells + sally_seashells = 50 := 
by 
  sorry

end NUMINAMATH_GPT_seashells_total_l776_77635


namespace NUMINAMATH_GPT_compute_abs_difference_l776_77695

theorem compute_abs_difference (x y : ℝ) 
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 3.6)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 4.5) : 
  |x - y| = 1.1 :=
by 
  sorry

end NUMINAMATH_GPT_compute_abs_difference_l776_77695


namespace NUMINAMATH_GPT_solution_set_of_inequality_l776_77619

theorem solution_set_of_inequality : 
  { x : ℝ | (x + 2) * (1 - x) > 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l776_77619


namespace NUMINAMATH_GPT_div_by_11_l776_77656

theorem div_by_11 (x y : ℤ) (k : ℤ) (h : 14 * x + 13 * y = 11 * k) : 11 ∣ (19 * x + 9 * y) :=
by
  sorry

end NUMINAMATH_GPT_div_by_11_l776_77656


namespace NUMINAMATH_GPT_no_real_x_satisfying_quadratic_inequality_l776_77600

theorem no_real_x_satisfying_quadratic_inequality (a : ℝ) :
  ¬(∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_GPT_no_real_x_satisfying_quadratic_inequality_l776_77600


namespace NUMINAMATH_GPT_num_of_negative_x_l776_77647

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end NUMINAMATH_GPT_num_of_negative_x_l776_77647


namespace NUMINAMATH_GPT_Madeline_hours_left_over_l776_77653

theorem Madeline_hours_left_over :
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  total_hours_per_week - total_busy_hours = 46 :=
by
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  have : total_hours_per_week - total_busy_hours = 168 - 122 := by rfl
  have : 168 - 122 = 46 := by rfl
  exact this

end NUMINAMATH_GPT_Madeline_hours_left_over_l776_77653


namespace NUMINAMATH_GPT_student_attempted_sums_l776_77679

theorem student_attempted_sums (right wrong : ℕ) (h1 : wrong = 2 * right) (h2 : right = 12) : right + wrong = 36 := sorry

end NUMINAMATH_GPT_student_attempted_sums_l776_77679


namespace NUMINAMATH_GPT_equal_pieces_length_l776_77684

theorem equal_pieces_length (total_length_cm : ℕ) (num_pieces : ℕ) (num_equal_pieces : ℕ) (length_remaining_piece_mm : ℕ) :
  total_length_cm = 1165 ∧ num_pieces = 154 ∧ num_equal_pieces = 150 ∧ length_remaining_piece_mm = 100 →
  (total_length_cm * 10 - (num_pieces - num_equal_pieces) * length_remaining_piece_mm) / num_equal_pieces = 75 :=
by
  sorry

end NUMINAMATH_GPT_equal_pieces_length_l776_77684


namespace NUMINAMATH_GPT_danny_age_l776_77697

theorem danny_age (D : ℕ) (h : D - 19 = 3 * (26 - 19)) : D = 40 := by
  sorry

end NUMINAMATH_GPT_danny_age_l776_77697


namespace NUMINAMATH_GPT_valid_votes_other_candidate_l776_77634

theorem valid_votes_other_candidate (total_votes : ℕ)
  (invalid_percentage valid_percentage candidate1_percentage candidate2_percentage : ℕ)
  (h_invalid_valid_sum : invalid_percentage + valid_percentage = 100)
  (h_candidates_sum : candidate1_percentage + candidate2_percentage = 100)
  (h_invalid_percentage : invalid_percentage = 20)
  (h_candidate1_percentage : candidate1_percentage = 55)
  (h_total_votes : total_votes = 7500)
  (h_valid_percentage_eq : valid_percentage = 100 - invalid_percentage)
  (h_candidate2_percentage_eq : candidate2_percentage = 100 - candidate1_percentage) :
  ( ( candidate2_percentage * ( valid_percentage * total_votes / 100) ) / 100 ) = 2700 :=
  sorry

end NUMINAMATH_GPT_valid_votes_other_candidate_l776_77634


namespace NUMINAMATH_GPT_total_leftover_tarts_l776_77643

variable (cherry_tart blueberry_tart peach_tart : ℝ)
variable (h1 : cherry_tart = 0.08)
variable (h2 : blueberry_tart = 0.75)
variable (h3 : peach_tart = 0.08)

theorem total_leftover_tarts : 
  cherry_tart + blueberry_tart + peach_tart = 0.91 := 
by 
  sorry

end NUMINAMATH_GPT_total_leftover_tarts_l776_77643


namespace NUMINAMATH_GPT_composite_quadratic_l776_77622

theorem composite_quadratic (m n : ℤ) (x1 x2 : ℤ)
  (h1 : 2 * x1^2 + m * x1 + 2 - n = 0)
  (h2 : 2 * x2^2 + m * x2 + 2 - n = 0)
  (h3 : x1 ≠ 0) 
  (h4 : x2 ≠ 0) :
  ∃ (k : ℕ), ∃ (l : ℕ), 
    (k > 1) ∧ (l > 1) ∧ (k * l = (m^2 + n^2) / 4) := sorry

end NUMINAMATH_GPT_composite_quadratic_l776_77622


namespace NUMINAMATH_GPT_extrema_of_function_l776_77664

noncomputable def f (x : ℝ) := x / 8 + 2 / x

theorem extrema_of_function : 
  ∀ x ∈ Set.Ioo (-5 : ℝ) (10),
  (x ≠ 0) →
  (f (-4) = -1 ∧ f 4 = 1) ∧
  (∀ x ∈ Set.Ioc (-5) 0, f x ≤ -1) ∧
  (∀ x ∈ Set.Ioo 0 10, f x ≥ 1) := by
  sorry

end NUMINAMATH_GPT_extrema_of_function_l776_77664


namespace NUMINAMATH_GPT_find_train_speed_l776_77645

variable (bridge_length train_length train_crossing_time : ℕ)

def speed_of_train (bridge_length train_length train_crossing_time : ℕ) : ℕ :=
  (bridge_length + train_length) / train_crossing_time

theorem find_train_speed
  (bridge_length : ℕ) (train_length : ℕ) (train_crossing_time : ℕ)
  (h_bridge_length : bridge_length = 180)
  (h_train_length : train_length = 120)
  (h_train_crossing_time : train_crossing_time = 20) :
  speed_of_train bridge_length train_length train_crossing_time = 15 := by
  sorry

end NUMINAMATH_GPT_find_train_speed_l776_77645


namespace NUMINAMATH_GPT_last_number_l776_77604

theorem last_number (A B C D E F G : ℕ)
  (h1 : A + B + C + D = 52)
  (h2 : D + E + F + G = 60)
  (h3 : E + F + G = 55)
  (h4 : D^2 = G) : G = 25 :=
by
  sorry

end NUMINAMATH_GPT_last_number_l776_77604


namespace NUMINAMATH_GPT_largest_three_digit_number_l776_77685

def divisible_by_each_digit (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∀ d ∈ digits, d ≠ 0 ∧ n % d = 0

def sum_of_digits_divisible_by (n : ℕ) (k : ℕ) : Prop :=
  let sum := (n / 100) + ((n / 10) % 10) + (n % 10)
  sum % k = 0

theorem largest_three_digit_number : ∃ n : ℕ, n = 936 ∧
  n >= 100 ∧ n < 1000 ∧
  divisible_by_each_digit n ∧
  sum_of_digits_divisible_by n 6 :=
by
  -- Proof details are omitted
  sorry

end NUMINAMATH_GPT_largest_three_digit_number_l776_77685


namespace NUMINAMATH_GPT_difference_two_numbers_l776_77672

theorem difference_two_numbers (a b : ℕ) (h₁ : a + b = 20250) (h₂ : b % 15 = 0) (h₃ : a = b / 3) : b - a = 10130 :=
by 
  sorry

end NUMINAMATH_GPT_difference_two_numbers_l776_77672


namespace NUMINAMATH_GPT_cube_edge_length_l776_77607

-- Define edge length and surface area
variables (edge_length surface_area : ℝ)

-- Given condition
def surface_area_condition : Prop := surface_area = 294

-- Cube surface area formula
def cube_surface_area : Prop := surface_area = 6 * edge_length^2

-- Proof statement
theorem cube_edge_length (h1: surface_area_condition surface_area) (h2: cube_surface_area edge_length surface_area) : edge_length = 7 := 
by
  sorry

end NUMINAMATH_GPT_cube_edge_length_l776_77607


namespace NUMINAMATH_GPT_total_number_of_fish_l776_77606

theorem total_number_of_fish
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (blue_spotted_fish : ℕ)
  (h1 : blue_fish = total_fish / 3)
  (h2 : blue_spotted_fish = blue_fish / 2)
  (h3 : blue_spotted_fish = 10) :
  total_fish = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_fish_l776_77606


namespace NUMINAMATH_GPT_find_interest_rate_l776_77682

-- Given conditions
def P : ℝ := 4099.999999999999
def t : ℕ := 2
def CI_minus_SI : ℝ := 41

-- Formulas for Simple Interest and Compound Interest
def SI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * r * (t : ℝ)
def CI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * ((1 + r) ^ t) - P

-- Main theorem to prove: the interest rate r is 0.1 (i.e., 10%)
theorem find_interest_rate (r : ℝ) : 
  (CI P r t - SI P r t = CI_minus_SI) → r = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l776_77682


namespace NUMINAMATH_GPT_jill_trips_to_fill_tank_l776_77624

   -- Defining the conditions
   def tank_capacity : ℕ := 600
   def bucket_capacity : ℕ := 5
   def jack_buckets_per_trip : ℕ := 2
   def jill_buckets_per_trip : ℕ := 1
   def jack_trip_rate : ℕ := 3
   def jill_trip_rate : ℕ := 2

   -- Calculate the amount of water Jack and Jill carry per trip
   def jack_gallons_per_trip : ℕ := jack_buckets_per_trip * bucket_capacity
   def jill_gallons_per_trip : ℕ := jill_buckets_per_trip * bucket_capacity

   -- Grouping the trips in the time it takes for Jill to complete her trips
   def total_gallons_per_group : ℕ := (jack_trip_rate * jack_gallons_per_trip) + (jill_trip_rate * jill_gallons_per_trip)

   -- Calculate the number of groups needed to fill the tank
   def groups_needed : ℕ := tank_capacity / total_gallons_per_group

   -- Calculate the total trips Jill makes
   def jill_total_trips : ℕ := groups_needed * jill_trip_rate

   -- The proof statement
   theorem jill_trips_to_fill_tank : jill_total_trips = 30 :=
   by
     -- Skipping the proof
     sorry
   
end NUMINAMATH_GPT_jill_trips_to_fill_tank_l776_77624


namespace NUMINAMATH_GPT_chord_equation_l776_77611

variable {x y k b : ℝ}

-- Define the condition of the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 - 4 = 0

-- Define the condition that the point M(1, 1) is the midpoint
def midpoint_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 1

-- Define the line equation in terms of its slope k and y-intercept b
def line (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

theorem chord_equation :
  (∃ (x₁ x₂ y₁ y₂ : ℝ), ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ midpoint_condition x₁ y₁ x₂ y₂) →
  (∃ (k b : ℝ), line k b x y ∧ k + b = 1 ∧ b = 1 - k) →
  y = -0.5 * x + 1.5 ↔ x + 2 * y - 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_chord_equation_l776_77611


namespace NUMINAMATH_GPT_koala_fiber_eaten_l776_77699

-- Definitions based on conditions
def absorbs_percentage : ℝ := 0.40
def fiber_absorbed : ℝ := 12

-- The theorem statement to prove the total amount of fiber eaten
theorem koala_fiber_eaten : 
  (fiber_absorbed / absorbs_percentage) = 30 :=
by 
  sorry

end NUMINAMATH_GPT_koala_fiber_eaten_l776_77699


namespace NUMINAMATH_GPT_dot_product_result_l776_77650

open scoped BigOperators

-- Define the vectors a and b
def a : ℝ × ℝ := (2, -3)
def b : ℝ × ℝ := (-1, 2)

-- Define the addition of two vectors
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved
theorem dot_product_result : dot_product (vector_add a b) a = 5 := by
  sorry

end NUMINAMATH_GPT_dot_product_result_l776_77650


namespace NUMINAMATH_GPT_find_k_and_x2_l776_77623

theorem find_k_and_x2 (k : ℝ) (x2 : ℝ)
  (h1 : 2 * x2 = k)
  (h2 : 2 + x2 = 6) :
  k = 8 ∧ x2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_and_x2_l776_77623


namespace NUMINAMATH_GPT_Samantha_last_name_length_l776_77605

theorem Samantha_last_name_length :
  ∃ (S B : ℕ), S = B - 3 ∧ B - 2 = 2 * 4 ∧ S = 7 :=
by
  sorry

end NUMINAMATH_GPT_Samantha_last_name_length_l776_77605


namespace NUMINAMATH_GPT_pauly_omelets_l776_77657

/-- Pauly is making omelets for his family. There are three dozen eggs, and he plans to use them all. 
Each omelet requires 4 eggs. Including himself, there are 3 people. 
Prove that each person will get 3 omelets. -/

def total_eggs := 3 * 12

def eggs_per_omelet := 4

def total_omelets := total_eggs / eggs_per_omelet

def number_of_people := 3

def omelets_per_person := total_omelets / number_of_people

theorem pauly_omelets : omelets_per_person = 3 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_pauly_omelets_l776_77657


namespace NUMINAMATH_GPT_binom_10_3_eq_120_l776_77614

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_GPT_binom_10_3_eq_120_l776_77614


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l776_77638

theorem relationship_between_a_and_b (a b : ℝ) (h₀ : a ≠ 0) (max_point : ∃ x, (x = 0 ∨ x = 1/3) ∧ (∀ y, (y = 0 ∨ y = 1/3) → (3 * a * y^2 + 2 * b * y) = 0)) : a + 2 * b = 0 :=
sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l776_77638


namespace NUMINAMATH_GPT_arithmetic_sequence_mod_l776_77683

theorem arithmetic_sequence_mod :
  let a := 2
  let d := 5
  let l := 137
  let n := (l - a) / d + 1
  let S := n * (2 * a + (n - 1) * d) / 2
  n = 28 ∧ S = 1946 →
  S % 20 = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_mod_l776_77683


namespace NUMINAMATH_GPT_photos_difference_is_120_l776_77661

theorem photos_difference_is_120 (initial_photos : ℕ) (final_photos : ℕ) (first_day_factor : ℕ) (first_day_photos : ℕ) (second_day_photos : ℕ) : 
  initial_photos = 400 → 
  final_photos = 920 → 
  first_day_factor = 2 →
  first_day_photos = initial_photos / first_day_factor →
  final_photos = initial_photos + first_day_photos + second_day_photos →
  second_day_photos - first_day_photos = 120 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_photos_difference_is_120_l776_77661


namespace NUMINAMATH_GPT_inequality_problem_l776_77621

noncomputable def a := (3 / 4) * Real.exp (2 / 5)
noncomputable def b := 2 / 5
noncomputable def c := (2 / 5) * Real.exp (3 / 4)

theorem inequality_problem : b < c ∧ c < a := by
  sorry

end NUMINAMATH_GPT_inequality_problem_l776_77621


namespace NUMINAMATH_GPT_locus_of_C_l776_77630

variables (a b : ℝ)
variables (h1 : a > 0) (h2 : b > 0)

theorem locus_of_C :
  ∀ (C : ℝ × ℝ), (C.2 = (b / a) * C.1 ∧ (a * b / Real.sqrt (a ^ 2 + b ^ 2) ≤ C.1) ∧ (C.1 ≤ a)) :=
sorry

end NUMINAMATH_GPT_locus_of_C_l776_77630


namespace NUMINAMATH_GPT_domain_ln_x_plus_one_l776_77670

theorem domain_ln_x_plus_one : 
  { x : ℝ | ∃ y : ℝ, y = x + 1 ∧ y > 0 } = { x : ℝ | x > -1 } :=
by
  sorry

end NUMINAMATH_GPT_domain_ln_x_plus_one_l776_77670


namespace NUMINAMATH_GPT_sum_base_6_l776_77625

-- Define base 6 numbers
def n1 : ℕ := 1 * 6^3 + 4 * 6^2 + 5 * 6^1 + 2 * 6^0
def n2 : ℕ := 2 * 6^3 + 3 * 6^2 + 5 * 6^1 + 4 * 6^0

-- Define the expected result in base 6
def expected_sum : ℕ := 4 * 6^3 + 2 * 6^2 + 5 * 6^1 + 0 * 6^0

-- The theorem to prove
theorem sum_base_6 : n1 + n2 = expected_sum := by
    sorry

end NUMINAMATH_GPT_sum_base_6_l776_77625


namespace NUMINAMATH_GPT_zigzag_lines_divide_regions_l776_77696

-- Define the number of regions created by n zigzag lines
def regions (n : ℕ) : ℕ := (2 * n * (2 * n + 1)) / 2 + 1 - 2 * n

-- Main theorem
theorem zigzag_lines_divide_regions (n : ℕ) : ∃ k : ℕ, k = regions n := by
  sorry

end NUMINAMATH_GPT_zigzag_lines_divide_regions_l776_77696


namespace NUMINAMATH_GPT_medians_sum_square_l776_77618

-- Define the sides of the triangle
variables {a b c : ℝ}

-- Define diameters
variables {D : ℝ}

-- Define medians of the triangle
variables {m_a m_b m_c : ℝ}

-- Defining the theorem statement
theorem medians_sum_square :
  m_a ^ 2 + m_b ^ 2 + m_c ^ 2 = (3 / 4) * (a ^ 2 + b ^ 2 + c ^ 2) + (3 / 4) * D ^ 2 :=
sorry

end NUMINAMATH_GPT_medians_sum_square_l776_77618


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l776_77662

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℤ) (n : ℕ)
  (h₁ : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h₂ : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h₃ : 3 * a 5 - a 1 = 10) :
  S 13 = 117 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l776_77662


namespace NUMINAMATH_GPT_value_of_a8_l776_77644

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → 2 * a n + a (n + 1) = 0

theorem value_of_a8 (a : ℕ → ℝ) (h1 : seq a) (h2 : a 3 = -2) : a 8 = 64 :=
sorry

end NUMINAMATH_GPT_value_of_a8_l776_77644


namespace NUMINAMATH_GPT_solution_set_f_prime_pos_l776_77690

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

theorem solution_set_f_prime_pos : 
  {x : ℝ | 0 < x ∧ (deriv f x > 0)} = {x : ℝ | 2 < x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f_prime_pos_l776_77690


namespace NUMINAMATH_GPT_value_of_six_inch_cube_l776_77693

theorem value_of_six_inch_cube :
  let four_inch_cube_value := 400
  let four_inch_side_length := 4
  let six_inch_side_length := 6
  let volume (s : ℕ) : ℕ := s ^ 3
  (volume six_inch_side_length / volume four_inch_side_length) * four_inch_cube_value = 1350 := by
sorry

end NUMINAMATH_GPT_value_of_six_inch_cube_l776_77693


namespace NUMINAMATH_GPT_required_decrease_l776_77698

noncomputable def price_after_increases (P : ℝ) : ℝ :=
  let P1 := 1.20 * P
  let P2 := 1.10 * P1
  1.15 * P2

noncomputable def price_after_discount (P : ℝ) : ℝ :=
  0.95 * price_after_increases P

noncomputable def price_after_tax (P : ℝ) : ℝ :=
  1.07 * price_after_discount P

theorem required_decrease (P : ℝ) (D : ℝ) : 
  (1 - D / 100) * price_after_tax P = P ↔ D = 35.1852 :=
by
  sorry

end NUMINAMATH_GPT_required_decrease_l776_77698


namespace NUMINAMATH_GPT_washed_shirts_l776_77659

-- Definitions based on the conditions
def short_sleeve_shirts : ℕ := 39
def long_sleeve_shirts : ℕ := 47
def unwashed_shirts : ℕ := 66

-- The total number of shirts is the sum of short and long sleeve shirts
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts

-- The problem to prove that Oliver washed 20 shirts
theorem washed_shirts :
  total_shirts - unwashed_shirts = 20 := 
sorry

end NUMINAMATH_GPT_washed_shirts_l776_77659


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l776_77666

variable {α : Type*} [LinearOrderedField α]

noncomputable def S (n a_1 d : α) : α :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem arithmetic_sequence_sum (a_1 d : α) :
  S 5 a_1 d = 5 → S 9 a_1 d = 27 → S 7 a_1 d = 14 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l776_77666


namespace NUMINAMATH_GPT_band_member_earnings_l776_77620

theorem band_member_earnings :
  let attendees := 500
  let ticket_price := 30
  let band_share_percentage := 70 / 100
  let band_members := 4
  let total_earnings := attendees * ticket_price
  let band_earnings := total_earnings * band_share_percentage
  let earnings_per_member := band_earnings / band_members
  earnings_per_member = 2625 := 
by {
  sorry
}

end NUMINAMATH_GPT_band_member_earnings_l776_77620


namespace NUMINAMATH_GPT_louis_current_age_l776_77633

/-- 
  In 6 years, Carla will be 30 years old. 
  The sum of the current ages of Carla and Louis is 55. 
  Prove that Louis is currently 31 years old.
--/
theorem louis_current_age (C L : ℕ) 
  (h1 : C + 6 = 30) 
  (h2 : C + L = 55) 
  : L = 31 := 
sorry

end NUMINAMATH_GPT_louis_current_age_l776_77633


namespace NUMINAMATH_GPT_B_completes_remaining_work_in_23_days_l776_77636

noncomputable def A_work_rate : ℝ := 1 / 45
noncomputable def B_work_rate : ℝ := 1 / 40
noncomputable def combined_work_rate : ℝ := A_work_rate + B_work_rate
noncomputable def work_done_together_in_9_days : ℝ := combined_work_rate * 9
noncomputable def remaining_work : ℝ := 1 - work_done_together_in_9_days
noncomputable def days_B_completes_remaining_work : ℝ := remaining_work / B_work_rate

theorem B_completes_remaining_work_in_23_days :
  days_B_completes_remaining_work = 23 :=
by 
  -- Proof omitted - please fill in the proof steps
  sorry

end NUMINAMATH_GPT_B_completes_remaining_work_in_23_days_l776_77636


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l776_77676

theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  ∃ x y : ℝ, y = 4 * a * x^2 → (x, y) = (0, 1 / (16 * a)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l776_77676


namespace NUMINAMATH_GPT_max_value_of_expression_l776_77692

theorem max_value_of_expression (A M C : ℕ) (hA : 0 < A) (hM : 0 < M) (hC : 0 < C) (hSum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A + A + M + C ≤ 215 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l776_77692


namespace NUMINAMATH_GPT_area_of_rhombus_l776_77667

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 22) (h2 : d2 = 30) : 
  (d1 * d2) / 2 = 330 :=
by
  rw [h1, h2]
  norm_num

-- Here we state the theorem about the area of the rhombus given its diagonal lengths.

end NUMINAMATH_GPT_area_of_rhombus_l776_77667


namespace NUMINAMATH_GPT_store_profit_l776_77603

theorem store_profit :
  let selling_price : ℝ := 80
  let cost_price_profitable : ℝ := (selling_price - 0.60 * selling_price)
  let cost_price_loss : ℝ := (selling_price + 0.20 * selling_price)
  selling_price + selling_price - cost_price_profitable - cost_price_loss = 10 := by
  sorry

end NUMINAMATH_GPT_store_profit_l776_77603


namespace NUMINAMATH_GPT_meeting_attendance_l776_77688

theorem meeting_attendance (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + 2 * B = 11) : A + B = 6 :=
sorry

end NUMINAMATH_GPT_meeting_attendance_l776_77688


namespace NUMINAMATH_GPT_almost_perfect_numbers_l776_77616

def d (n : Nat) : Nat := 
  -- Implement the function to count the number of positive divisors of n
  sorry

def f (n : Nat) : Nat := 
  -- Implement the function f(n) as given in the problem statement
  sorry

def isAlmostPerfect (n : Nat) : Prop := 
  f n = n

theorem almost_perfect_numbers :
  ∀ n, isAlmostPerfect n → n = 1 ∨ n = 3 ∨ n = 18 ∨ n = 36 :=
by
  sorry

end NUMINAMATH_GPT_almost_perfect_numbers_l776_77616


namespace NUMINAMATH_GPT_translate_line_upwards_by_3_translate_line_right_by_3_l776_77649

theorem translate_line_upwards_by_3 (x : ℝ) :
  let y := 2 * x - 4
  let y' := y + 3
  y' = 2 * x - 1 := 
by
  let y := 2 * x - 4
  let y' := y + 3
  sorry

theorem translate_line_right_by_3 (x : ℝ) :
  let y := 2 * x - 4
  let y_up := y + 3
  let y_right := 2 * (x - 3) - 4
  y_right = 2 * x - 10 :=
by
  let y := 2 * x - 4
  let y_up := y + 3
  let y_right := 2 * (x - 3) - 4
  sorry

end NUMINAMATH_GPT_translate_line_upwards_by_3_translate_line_right_by_3_l776_77649


namespace NUMINAMATH_GPT_only_n_eq_1_divides_2_pow_n_minus_1_l776_77686

theorem only_n_eq_1_divides_2_pow_n_minus_1 (n : ℕ) (h1 : 1 ≤ n) (h2 : n ∣ 2^n - 1) : n = 1 :=
sorry

end NUMINAMATH_GPT_only_n_eq_1_divides_2_pow_n_minus_1_l776_77686


namespace NUMINAMATH_GPT_prove_inequality_l776_77673

variable (x y z : ℝ)
variable (h₁ : x > 0)
variable (h₂ : y > 0)
variable (h₃ : z > 0)
variable (h₄ : x + y + z = 1)

theorem prove_inequality :
  (3 * x^2 - x) / (1 + x^2) +
  (3 * y^2 - y) / (1 + y^2) +
  (3 * z^2 - z) / (1 + z^2) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_prove_inequality_l776_77673


namespace NUMINAMATH_GPT_train_length_l776_77668

theorem train_length 
  (t1 t2 : ℝ)
  (d2 : ℝ)
  (L : ℝ)
  (V : ℝ)
  (h1 : t1 = 18)
  (h2 : t2 = 27)
  (h3 : d2 = 150.00000000000006)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) :
  L = 300.0000000000001 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l776_77668


namespace NUMINAMATH_GPT_diameter_is_twice_radius_l776_77671

theorem diameter_is_twice_radius {r d : ℝ} (h : d = 2 * r) : d = 2 * r :=
by {
  sorry
}

end NUMINAMATH_GPT_diameter_is_twice_radius_l776_77671


namespace NUMINAMATH_GPT_find_DF_l776_77681

noncomputable def triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def median (a b : ℝ) : ℝ := a / 2

theorem find_DF {DE EF DM DF : ℝ} (h1 : DE = 7) (h2 : EF = 10) (h3 : DM = 5) :
  DF = Real.sqrt 51 :=
by
  sorry

end NUMINAMATH_GPT_find_DF_l776_77681


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l776_77617

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3 ∨ b = 3) (h2 : a = 6 ∨ b = 6) 
(h_isosceles : a = b ∨ b = a) : 
  a + b + a = 15 ∨ b + a + b = 15 :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l776_77617


namespace NUMINAMATH_GPT_max_principals_ten_years_l776_77627

theorem max_principals_ten_years : 
  (∀ (P : ℕ → Prop), (∀ n, n ≥ 10 → ∀ i, ¬P (n - i)) → ∀ p, p ≤ 4 → 
  (∃ n ≤ 10, ∀ k, k ≥ n → P k)) :=
sorry

end NUMINAMATH_GPT_max_principals_ten_years_l776_77627


namespace NUMINAMATH_GPT_urea_formation_l776_77602

theorem urea_formation (CO2 NH3 : ℕ) (OCN2 H2O : ℕ) (h1 : CO2 = 3) (h2 : NH3 = 6) :
  (∀ x, CO2 * 1 + NH3 * 2 = x + (2 * x) + x) →
  OCN2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_urea_formation_l776_77602


namespace NUMINAMATH_GPT_terry_tomato_types_l776_77665

theorem terry_tomato_types (T : ℕ) (h1 : 2 * T * 4 * 2 = 48) : T = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_terry_tomato_types_l776_77665


namespace NUMINAMATH_GPT_describe_random_event_l776_77660

def idiom_A : Prop := "海枯石烂" = "extremely improbable or far into the future, not random"
def idiom_B : Prop := "守株待兔" = "represents a random event"
def idiom_C : Prop := "画饼充饥" = "unreal hopes, not random"
def idiom_D : Prop := "瓜熟蒂落" = "natural or expected outcome, not random"

theorem describe_random_event : idiom_B := 
by
  -- Proof omitted; conclusion follows from the given definitions
  sorry

end NUMINAMATH_GPT_describe_random_event_l776_77660


namespace NUMINAMATH_GPT_negation_universal_to_particular_l776_77677

theorem negation_universal_to_particular :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_universal_to_particular_l776_77677


namespace NUMINAMATH_GPT_negation_proposition_l776_77641

-- Definitions based on the conditions
def original_proposition : Prop := ∃ x : ℝ, x^2 + 3*x + 2 < 0

-- Theorem requiring proof
theorem negation_proposition : (¬ original_proposition) = ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l776_77641


namespace NUMINAMATH_GPT_clubsuit_commute_l776_77612

-- Define the operation a ♣ b = a^3 * b - a * b^3
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the proposition to prove
theorem clubsuit_commute (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
by
  sorry

end NUMINAMATH_GPT_clubsuit_commute_l776_77612


namespace NUMINAMATH_GPT_inequalities_not_all_hold_l776_77675

theorem inequalities_not_all_hold (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
    ¬ (a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by
  sorry

end NUMINAMATH_GPT_inequalities_not_all_hold_l776_77675


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l776_77642

theorem problem1 : 128 + 52 / 13 = 132 :=
by
  sorry

theorem problem2 : 132 / 11 * 29 - 178 = 170 :=
by
  sorry

theorem problem3 : 45 * (320 / (4 * 5)) = 720 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l776_77642


namespace NUMINAMATH_GPT_polynomial_divisibility_l776_77694

theorem polynomial_divisibility (C D : ℝ)
  (h : ∀ x, x^2 + x + 1 = 0 → x^102 + C * x + D = 0) :
  C + D = -1 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l776_77694


namespace NUMINAMATH_GPT_rectangle_area_l776_77640

theorem rectangle_area
  (line : ∀ x, 6 = x * x + 4 * x + 3 → x = -2 + Real.sqrt 7 ∨ x = -2 - Real.sqrt 7)
  (shorter_side : ∃ l, l = 2 * Real.sqrt 7 ∧ ∃ s, s = l + 3) :
  ∃ a, a = 28 + 12 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l776_77640


namespace NUMINAMATH_GPT_mass_percentage_C_is_54_55_l776_77609

def mass_percentage (C: String) (percentage: ℝ) : Prop :=
  percentage = 54.55

theorem mass_percentage_C_is_54_55 :
  mass_percentage "C" 54.55 :=
by
  unfold mass_percentage
  rfl

end NUMINAMATH_GPT_mass_percentage_C_is_54_55_l776_77609


namespace NUMINAMATH_GPT_border_material_length_l776_77613

noncomputable def area (r : ℝ) : ℝ := (22 / 7) * r^2

theorem border_material_length (r : ℝ) (C : ℝ) (border : ℝ) : 
  area r = 616 →
  C = 2 * (22 / 7) * r →
  border = C + 3 →
  border = 91 :=
by
  intro h_area h_circumference h_border
  sorry

end NUMINAMATH_GPT_border_material_length_l776_77613


namespace NUMINAMATH_GPT_expand_product_polynomials_l776_77629

noncomputable def poly1 : Polynomial ℤ := 5 * Polynomial.X + 3
noncomputable def poly2 : Polynomial ℤ := 7 * Polynomial.X^2 + 2 * Polynomial.X + 4
noncomputable def expanded_form : Polynomial ℤ := 35 * Polynomial.X^3 + 31 * Polynomial.X^2 + 26 * Polynomial.X + 12

theorem expand_product_polynomials :
  poly1 * poly2 = expanded_form := 
by
  sorry

end NUMINAMATH_GPT_expand_product_polynomials_l776_77629


namespace NUMINAMATH_GPT_cryptarithmetic_puzzle_sol_l776_77691

theorem cryptarithmetic_puzzle_sol (A B C D : ℕ) 
  (h1 : A + B + C = D) 
  (h2 : B + C = 7) 
  (h3 : A - B = 1) : D = 9 := 
by 
  sorry

end NUMINAMATH_GPT_cryptarithmetic_puzzle_sol_l776_77691


namespace NUMINAMATH_GPT_determine_a_l776_77680

theorem determine_a (a x : ℝ) (h : x = 1) (h_eq : a * x + 2 * x = 3) : a = 1 :=
by
  subst h
  simp at h_eq
  linarith

end NUMINAMATH_GPT_determine_a_l776_77680


namespace NUMINAMATH_GPT_total_birds_on_fence_l776_77652

variable (initial_birds : ℕ := 1)
variable (added_birds : ℕ := 4)

theorem total_birds_on_fence : initial_birds + added_birds = 5 := by
  sorry

end NUMINAMATH_GPT_total_birds_on_fence_l776_77652
