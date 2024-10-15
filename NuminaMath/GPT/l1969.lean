import Mathlib

namespace NUMINAMATH_GPT_largest_result_l1969_196932

theorem largest_result :
  let A := (1 / 17 - 1 / 19) / 20
  let B := (1 / 15 - 1 / 21) / 60
  let C := (1 / 13 - 1 / 23) / 100
  let D := (1 / 11 - 1 / 25) / 140
  D > A ∧ D > B ∧ D > C := by
  sorry

end NUMINAMATH_GPT_largest_result_l1969_196932


namespace NUMINAMATH_GPT_largest_of_5_consecutive_odd_integers_l1969_196990

theorem largest_of_5_consecutive_odd_integers (n : ℤ) (h : n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 235) :
  n + 8 = 51 :=
sorry

end NUMINAMATH_GPT_largest_of_5_consecutive_odd_integers_l1969_196990


namespace NUMINAMATH_GPT_simplify_A_minus_B_value_of_A_minus_B_given_condition_l1969_196931

variable (a b : ℝ)

def A := (a + b) ^ 2 - 3 * b ^ 2
def B := 2 * (a + b) * (a - b) - 3 * a * b

theorem simplify_A_minus_B :
  A a b - B a b = -a ^ 2 + 5 * a * b :=
by sorry

theorem value_of_A_minus_B_given_condition :
  (a - 3) ^ 2 + |b - 4| = 0 → A a b - B a b = 51 :=
by sorry

end NUMINAMATH_GPT_simplify_A_minus_B_value_of_A_minus_B_given_condition_l1969_196931


namespace NUMINAMATH_GPT_problem_solution_l1969_196927

theorem problem_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1969_196927


namespace NUMINAMATH_GPT_polynomial_not_factorable_l1969_196919

theorem polynomial_not_factorable :
  ¬ ∃ (A B : Polynomial ℤ), A.degree < 5 ∧ B.degree < 5 ∧ A * B = (Polynomial.C 1 * Polynomial.X ^ 5 - Polynomial.C 3 * Polynomial.X ^ 4 + Polynomial.C 6 * Polynomial.X ^ 3 - Polynomial.C 3 * Polynomial.X ^ 2 + Polynomial.C 9 * Polynomial.X - Polynomial.C 6) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_not_factorable_l1969_196919


namespace NUMINAMATH_GPT_part_I_part_II_l1969_196933

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
noncomputable def g (x : ℝ) := |2 * x - 1|

theorem part_I (x : ℝ) : f x 2 ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1969_196933


namespace NUMINAMATH_GPT_total_days_spent_l1969_196954

theorem total_days_spent {weeks_to_days : ℕ → ℕ} : 
  (weeks_to_days 3 + weeks_to_days 1) + 
  (weeks_to_days (weeks_to_days 3 + weeks_to_days 2) + 3) + 
  (2 * (weeks_to_days (weeks_to_days 3 + weeks_to_days 2))) + 
  (weeks_to_days 5 - weeks_to_days 1) + 
  (weeks_to_days ((weeks_to_days 5 - weeks_to_days 1) - weeks_to_days 3) + 6) + 
  (weeks_to_days (weeks_to_days 5 - weeks_to_days 1) + 4) = 230 :=
by
  sorry

end NUMINAMATH_GPT_total_days_spent_l1969_196954


namespace NUMINAMATH_GPT_price_is_219_l1969_196908

noncomputable def discount_coupon1 (price : ℝ) : ℝ :=
  if price > 50 then 0.1 * price else 0

noncomputable def discount_coupon2 (price : ℝ) : ℝ :=
  if price > 100 then 20 else 0

noncomputable def discount_coupon3 (price : ℝ) : ℝ :=
  if price > 100 then 0.18 * (price - 100) else 0

noncomputable def more_savings_coupon1 (price : ℝ) : Prop :=
  discount_coupon1 price > discount_coupon2 price ∧ discount_coupon1 price > discount_coupon3 price

theorem price_is_219 (price : ℝ) :
  more_savings_coupon1 price → price = 219 :=
by
  sorry

end NUMINAMATH_GPT_price_is_219_l1969_196908


namespace NUMINAMATH_GPT_fruit_shop_problem_l1969_196953

variable (x y z : ℝ)

theorem fruit_shop_problem
  (h1 : x + 4 * y + 2 * z = 27.2)
  (h2 : 2 * x + 6 * y + 2 * z = 32.4) :
  x + 2 * y = 5.2 :=
by
  sorry

end NUMINAMATH_GPT_fruit_shop_problem_l1969_196953


namespace NUMINAMATH_GPT_fan_working_time_each_day_l1969_196936

theorem fan_working_time_each_day
  (airflow_per_second : ℝ)
  (total_airflow_week : ℝ)
  (seconds_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (airy_sector: airflow_per_second = 10)
  (flow_week : total_airflow_week = 42000)
  (sec_per_hr : seconds_per_hour = 3600)
  (hrs_per_day : hours_per_day = 24)
  (days_week : days_per_week = 7) :
  let airflow_per_hour := airflow_per_second * seconds_per_hour
  let total_hours_week := total_airflow_week / airflow_per_hour
  let hours_per_day_given := total_hours_week / days_per_week
  let minutes_per_day := hours_per_day_given * 60
  minutes_per_day = 10 := 
by
  sorry

end NUMINAMATH_GPT_fan_working_time_each_day_l1969_196936


namespace NUMINAMATH_GPT_no_arithmetic_mean_l1969_196955

def eight_thirteen : ℚ := 8 / 13
def eleven_seventeen : ℚ := 11 / 17
def five_eight : ℚ := 5 / 8

-- Define the function to calculate the arithmetic mean of two rational numbers
def arithmetic_mean (a b : ℚ) : ℚ :=
(a + b) / 2

-- The theorem statement
theorem no_arithmetic_mean :
  eight_thirteen ≠ arithmetic_mean eleven_seventeen five_eight ∧
  eleven_seventeen ≠ arithmetic_mean eight_thirteen five_eight ∧
  five_eight ≠ arithmetic_mean eight_thirteen eleven_seventeen :=
sorry

end NUMINAMATH_GPT_no_arithmetic_mean_l1969_196955


namespace NUMINAMATH_GPT_library_books_difference_l1969_196966

theorem library_books_difference :
  let books_old_town := 750
  let books_riverview := 1240
  let books_downtown := 1800
  let books_eastside := 1620
  books_downtown - books_old_town = 1050 :=
by
  sorry

end NUMINAMATH_GPT_library_books_difference_l1969_196966


namespace NUMINAMATH_GPT_total_students_is_17_l1969_196911

def total_students_in_class (students_liking_both_baseball_football : ℕ)
                             (students_only_baseball : ℕ)
                             (students_only_football : ℕ)
                             (students_liking_basketball_as_well : ℕ)
                             (students_liking_basketball_and_football_only : ℕ)
                             (students_liking_all_three : ℕ)
                             (students_liking_none : ℕ) : ℕ :=
  students_liking_both_baseball_football -
  students_liking_all_three +
  students_only_baseball +
  students_only_football +
  students_liking_basketball_and_football_only +
  students_liking_all_three +
  students_liking_none +
  (students_liking_basketball_as_well -
   (students_liking_all_three +
    students_liking_basketball_and_football_only))

theorem total_students_is_17 :
    total_students_in_class 7 3 4 2 1 2 5 = 17 :=
by sorry

end NUMINAMATH_GPT_total_students_is_17_l1969_196911


namespace NUMINAMATH_GPT_combine_like_terms_l1969_196922

theorem combine_like_terms : ∀ (x y : ℝ), -2 * x * y^2 + 2 * x * y^2 = 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_combine_like_terms_l1969_196922


namespace NUMINAMATH_GPT_largest_int_lt_100_remainder_3_div_by_8_l1969_196943

theorem largest_int_lt_100_remainder_3_div_by_8 : 
  ∃ n, n < 100 ∧ n % 8 = 3 ∧ ∀ m, m < 100 ∧ m % 8 = 3 → m ≤ 99 := by
  sorry

end NUMINAMATH_GPT_largest_int_lt_100_remainder_3_div_by_8_l1969_196943


namespace NUMINAMATH_GPT_xy_sum_eq_16_l1969_196999

theorem xy_sum_eq_16 (x y : ℕ) (h1: x > 0) (h2: y > 0) (h3: x < 20) (h4: y < 20) (h5: x + y + x * y = 76) : x + y = 16 :=
  sorry

end NUMINAMATH_GPT_xy_sum_eq_16_l1969_196999


namespace NUMINAMATH_GPT_children_left_l1969_196991

-- Define the initial problem constants and conditions
def totalGuests := 50
def halfGuests := totalGuests / 2
def numberOfMen := 15
def numberOfWomen := halfGuests
def numberOfChildren := totalGuests - (numberOfWomen + numberOfMen)
def proportionMenLeft := numberOfMen / 5
def totalPeopleStayed := 43
def totalPeopleLeft := totalGuests - totalPeopleStayed

-- Define the proposition to prove
theorem children_left : 
  totalPeopleLeft - proportionMenLeft = 4 := by 
    sorry

end NUMINAMATH_GPT_children_left_l1969_196991


namespace NUMINAMATH_GPT_train_length_l1969_196969

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def distance_ahead : ℝ := 270
noncomputable def time_to_pass : ℝ := 39

noncomputable def jogger_speed_mps := jogger_speed_kmph * (1000 / 1) * (1 / 3600)
noncomputable def train_speed_mps := train_speed_kmph * (1000 / 1) * (1 / 3600)

noncomputable def relative_speed_mps := train_speed_mps - jogger_speed_mps

theorem train_length :
  let jogger_speed := 9 * (1000 / 3600)
  let train_speed := 45 * (1000 / 3600)
  let relative_speed := train_speed - jogger_speed
  let distance := 270
  let time := 39
  distance + relative_speed * time = 390 → relative_speed * time = 120 := by
  sorry

end NUMINAMATH_GPT_train_length_l1969_196969


namespace NUMINAMATH_GPT_wall_length_to_height_ratio_l1969_196946

theorem wall_length_to_height_ratio (W H L V : ℝ) (h1 : H = 6 * W) (h2 : V = W * H * L) (h3 : W = 4) (h4 : V = 16128) :
  L / H = 7 :=
by
  -- Note: The proof steps are omitted as per the problem's instructions.
  sorry

end NUMINAMATH_GPT_wall_length_to_height_ratio_l1969_196946


namespace NUMINAMATH_GPT_union_P_Q_l1969_196992

open Set

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5, 6}

theorem union_P_Q : P ∪ Q = {1, 2, 3, 4, 5, 6} :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_union_P_Q_l1969_196992


namespace NUMINAMATH_GPT_Monica_books_read_l1969_196905

theorem Monica_books_read : 
  let books_last_year := 16 
  let books_this_year := 2 * books_last_year
  let books_next_year := 2 * books_this_year + 5
  books_next_year = 69 :=
by
  let books_last_year := 16
  let books_this_year := 2 * books_last_year
  let books_next_year := 2 * books_this_year + 5
  sorry

end NUMINAMATH_GPT_Monica_books_read_l1969_196905


namespace NUMINAMATH_GPT_no_real_solution_for_x_l1969_196977

theorem no_real_solution_for_x
  (y : ℝ)
  (x : ℝ)
  (h1 : y = (x^3 - 8) / (x - 2))
  (h2 : y = 3 * x) :
  ¬ ∃ x : ℝ, y = 3*x ∧ y = (x^3 - 8) / (x - 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_real_solution_for_x_l1969_196977


namespace NUMINAMATH_GPT_trig_expression_value_l1969_196976

theorem trig_expression_value (θ : ℝ)
  (h1 : Real.sin (Real.pi + θ) = 1/4) :
  (Real.cos (Real.pi + θ) / (Real.cos θ * (Real.cos (Real.pi + θ) - 1)) + 
  Real.sin (Real.pi / 2 - θ) / (Real.cos (θ + 2 * Real.pi) * Real.cos (Real.pi + θ) + Real.cos (-θ))) = 32 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l1969_196976


namespace NUMINAMATH_GPT_no_solution_to_system_l1969_196918

open Real

theorem no_solution_to_system (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^(1/3) - y^(1/3) - z^(1/3) = 64) ∧ (x^(1/4) - y^(1/4) - z^(1/4) = 32) ∧ (x^(1/6) - y^(1/6) - z^(1/6) = 8) → False := by
  sorry

end NUMINAMATH_GPT_no_solution_to_system_l1969_196918


namespace NUMINAMATH_GPT_square_area_fraction_shaded_l1969_196956

theorem square_area_fraction_shaded (s : ℝ) :
  let R := (s / 2, s)
  let S := (s, s / 2)
  -- Area of triangle RSV
  let area_RSV := (1 / 2) * (s / 2) * (s * Real.sqrt 2 / 4)
  -- Non-shaded area
  let non_shaded_area := area_RSV
  -- Total area of the square
  let total_area := s^2
  -- Shaded area
  let shaded_area := total_area - non_shaded_area
  -- Fraction shaded
  (shaded_area / total_area) = 1 - Real.sqrt 2 / 16 :=
by
  sorry

end NUMINAMATH_GPT_square_area_fraction_shaded_l1969_196956


namespace NUMINAMATH_GPT_Rachel_and_Mike_l1969_196945

theorem Rachel_and_Mike :
  ∃ b c : ℤ,
    (∀ x : ℝ, |x - 3| = 4 ↔ (x = 7 ∨ x = -1)) ∧
    (∀ x : ℝ, (x - 7) * (x + 1) = 0 ↔ x * x + b * x + c = 0) ∧
    (b, c) = (-6, -7) := by
sorry

end NUMINAMATH_GPT_Rachel_and_Mike_l1969_196945


namespace NUMINAMATH_GPT_degree_of_vertex_angle_of_isosceles_triangle_l1969_196914

theorem degree_of_vertex_angle_of_isosceles_triangle (exterior_angle : ℝ) (h_exterior_angle : exterior_angle = 40) : 
∃ vertex_angle : ℝ, vertex_angle = 140 :=
by 
  sorry

end NUMINAMATH_GPT_degree_of_vertex_angle_of_isosceles_triangle_l1969_196914


namespace NUMINAMATH_GPT_temperature_range_l1969_196988

theorem temperature_range (t : ℕ) : (21 ≤ t ∧ t ≤ 29) :=
by
  sorry

end NUMINAMATH_GPT_temperature_range_l1969_196988


namespace NUMINAMATH_GPT_acute_angle_10_10_l1969_196974

noncomputable def clock_angle_proof : Prop :=
  let minute_hand_position := 60
  let hour_hand_position := 305
  let angle_diff := hour_hand_position - minute_hand_position
  let acute_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  acute_angle = 115

theorem acute_angle_10_10 : clock_angle_proof := by
  sorry

end NUMINAMATH_GPT_acute_angle_10_10_l1969_196974


namespace NUMINAMATH_GPT_total_toys_l1969_196923

theorem total_toys (A M T : ℕ) (h1 : A = 3 * M + M) (h2 : T = A + 2) (h3 : M = 6) : A + M + T = 56 :=
by
  sorry

end NUMINAMATH_GPT_total_toys_l1969_196923


namespace NUMINAMATH_GPT_find_g_x_f_y_l1969_196952

-- Definition of the functions and conditions
variable (f g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, f (x + g y) = -x + y + 1)

-- The theorem to prove
theorem find_g_x_f_y (x y : ℝ) : g (x + f y) = -x + y - 1 := 
sorry

end NUMINAMATH_GPT_find_g_x_f_y_l1969_196952


namespace NUMINAMATH_GPT_shortest_remaining_side_l1969_196929

theorem shortest_remaining_side (a b : ℝ) (h1 : a = 7) (h2 : b = 24) (right_triangle : ∃ c, c^2 = a^2 + b^2) : a = 7 :=
by
  sorry

end NUMINAMATH_GPT_shortest_remaining_side_l1969_196929


namespace NUMINAMATH_GPT_base_8_add_sub_l1969_196935

-- Definitions of the numbers in base 8
def n1 : ℕ := 4 * 8^2 + 5 * 8^1 + 1 * 8^0
def n2 : ℕ := 1 * 8^2 + 6 * 8^1 + 2 * 8^0
def n3 : ℕ := 1 * 8^2 + 2 * 8^1 + 3 * 8^0

-- Convert the result to base 8
def to_base_8 (n : ℕ) : ℕ :=
  let d2 := n / 64
  let rem1 := n % 64
  let d1 := rem1 / 8
  let d0 := rem1 % 8
  d2 * 100 + d1 * 10 + d0

-- Proof statement
theorem base_8_add_sub :
  to_base_8 ((n1 + n2) - n3) = to_base_8 (5 * 8^2 + 1 * 8^1 + 0 * 8^0) :=
by
  sorry

end NUMINAMATH_GPT_base_8_add_sub_l1969_196935


namespace NUMINAMATH_GPT_lcm_technicians_schedule_l1969_196959

theorem lcm_technicians_schedule : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := 
sorry

end NUMINAMATH_GPT_lcm_technicians_schedule_l1969_196959


namespace NUMINAMATH_GPT_map_distance_scaled_l1969_196972

theorem map_distance_scaled (d_map : ℝ) (scale : ℝ) (d_actual : ℝ) :
  d_map = 8 ∧ scale = 1000000 → d_actual = 80 :=
by
  intro h
  rcases h with ⟨h1, h2⟩
  sorry

end NUMINAMATH_GPT_map_distance_scaled_l1969_196972


namespace NUMINAMATH_GPT_total_sheets_l1969_196926

-- Define the conditions
def sheets_in_bundle : ℕ := 10
def bundles : ℕ := 3
def additional_sheets : ℕ := 8

-- Theorem to prove the total number of sheets Jungkook has
theorem total_sheets : bundles * sheets_in_bundle + additional_sheets = 38 := by
  sorry

end NUMINAMATH_GPT_total_sheets_l1969_196926


namespace NUMINAMATH_GPT_max_puzzle_sets_l1969_196985

theorem max_puzzle_sets 
  (total_logic : ℕ) (total_visual : ℕ) (total_word : ℕ)
  (h1 : total_logic = 36) (h2 : total_visual = 27) (h3 : total_word = 15)
  (x y : ℕ)
  (h4 : 7 ≤ 4 * x + 3 * x + y ∧ 4 * x + 3 * x + y ≤ 12)
  (h5 : 4 * x / 3 * x = 4 / 3)
  (h6 : y ≥ 3 * x / 2) :
  5 ≤ total_logic / (4 * x) ∧ 5 ≤ total_visual / (3 * x) ∧ 5 ≤ total_word / y :=
sorry

end NUMINAMATH_GPT_max_puzzle_sets_l1969_196985


namespace NUMINAMATH_GPT_doberman_puppies_count_l1969_196997

theorem doberman_puppies_count (D : ℝ) (S : ℝ) (h1 : S = 55) (h2 : 3 * D - 5 + (D - S) = 90) : D = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_doberman_puppies_count_l1969_196997


namespace NUMINAMATH_GPT_hourly_wage_difference_l1969_196949

theorem hourly_wage_difference (P Q: ℝ) (H_p: ℝ) (H_q: ℝ) (h1: P = 1.5 * Q) (h2: H_q = H_p + 10) (h3: P * H_p = 420) (h4: Q * H_q = 420) : P - Q = 7 := by
  sorry

end NUMINAMATH_GPT_hourly_wage_difference_l1969_196949


namespace NUMINAMATH_GPT_inequality_solution_set_range_of_m_l1969_196917

noncomputable def f (x : ℝ) : ℝ := |x - 1|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m

theorem inequality_solution_set :
  {x : ℝ | f x + x^2 - 1 > 0} = {x : ℝ | x > 1 ∨ x < 0} :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x < g x m) → m > 4 :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_range_of_m_l1969_196917


namespace NUMINAMATH_GPT_work_increase_percent_l1969_196957

theorem work_increase_percent (W p : ℝ) (p_pos : p > 0) :
  (1 / 3 * p) * W / ((2 / 3) * p) - (W / p) = 0.5 * (W / p) :=
by
  sorry

end NUMINAMATH_GPT_work_increase_percent_l1969_196957


namespace NUMINAMATH_GPT_evaluate_f_3_minus_f_neg3_l1969_196947

def f (x : ℝ) : ℝ := x^6 + x^4 + 3*x^3 + 4*x^2 + 8*x

theorem evaluate_f_3_minus_f_neg3 : f 3 - f (-3) = 210 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_3_minus_f_neg3_l1969_196947


namespace NUMINAMATH_GPT_minimum_value_l1969_196930

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

theorem minimum_value : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x ≥ f (-1) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l1969_196930


namespace NUMINAMATH_GPT_sum_of_terms_l1969_196951

noncomputable def arithmetic_sequence : Type :=
  {a : ℕ → ℤ // ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d}

theorem sum_of_terms (a : arithmetic_sequence) (h1 : a.val 1 + a.val 3 = 2) (h2 : a.val 3 + a.val 5 = 4) :
  a.val 5 + a.val 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_terms_l1969_196951


namespace NUMINAMATH_GPT_total_students_registered_l1969_196906

theorem total_students_registered 
  (students_yesterday : ℕ) (absent_today : ℕ) 
  (attended_today : ℕ)
  (h1 : students_yesterday = 70)
  (h2 : absent_today = 30)
  (h3 : attended_today = (2 * students_yesterday) - (10 * (2 * students_yesterday) / 100)) :
  students_yesterday + absent_today = 156 := 
by
  sorry

end NUMINAMATH_GPT_total_students_registered_l1969_196906


namespace NUMINAMATH_GPT_solution_set_part1_solution_set_part2_l1969_196983

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 1

theorem solution_set_part1 :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

theorem solution_set_part2 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_part1_solution_set_part2_l1969_196983


namespace NUMINAMATH_GPT_new_figure_perimeter_equals_5_l1969_196937

-- Defining the side length of the square and the equilateral triangle
def side_length : ℝ := 1

-- Defining the perimeter of the new figure
def new_figure_perimeter : ℝ := 3 * side_length + 2 * side_length

-- Statement: The perimeter of the new figure equals 5
theorem new_figure_perimeter_equals_5 :
  new_figure_perimeter = 5 := by
  sorry

end NUMINAMATH_GPT_new_figure_perimeter_equals_5_l1969_196937


namespace NUMINAMATH_GPT_initial_avg_height_l1969_196967

-- Lean 4 statement for the given problem
theorem initial_avg_height (A : ℝ) (n : ℕ) (wrong_height correct_height actual_avg init_diff : ℝ)
  (h_class_size : n = 35)
  (h_wrong_height : wrong_height = 166)
  (h_correct_height : correct_height = 106)
  (h_actual_avg : actual_avg = 183)
  (h_init_diff : init_diff = wrong_height - correct_height)
  (h_total_height_actual : n * actual_avg = 35 * 183)
  (h_total_height_wrong : n * A = 35 * actual_avg - init_diff) :
  A = 181 :=
by {
  -- The problem and conditions are correctly stated. The proof is skipped with sorry.
  sorry
}

end NUMINAMATH_GPT_initial_avg_height_l1969_196967


namespace NUMINAMATH_GPT_farthings_in_a_pfennig_l1969_196998

theorem farthings_in_a_pfennig (x : ℕ) (h : 54 - 2 * x = 7 * x) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_farthings_in_a_pfennig_l1969_196998


namespace NUMINAMATH_GPT_part1_exists_infinite_rationals_part2_rationals_greater_bound_l1969_196958

theorem part1_exists_infinite_rationals 
  (sqrt5_minus1_div2 := (Real.sqrt 5 - 1) / 2):
  ∀ ε > 0, ∃ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 ∧ abs (q / p - sqrt5_minus1_div2) < 1 / p ^ 2 :=
by sorry

theorem part2_rationals_greater_bound
  (sqrt5_minus1_div2 := (Real.sqrt 5 - 1) / 2)
  (sqrt5_plus1_inv := 1 / (Real.sqrt 5 + 1)):
  ∀ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 → abs (q / p - sqrt5_minus1_div2) > sqrt5_plus1_inv / p ^ 2 :=
by sorry

end NUMINAMATH_GPT_part1_exists_infinite_rationals_part2_rationals_greater_bound_l1969_196958


namespace NUMINAMATH_GPT_parabola_no_intersection_inequality_l1969_196904

-- Definitions for the problem
theorem parabola_no_intersection_inequality
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, (a * x^2 + b * x + c ≠ x) ∧ (a * x^2 + b * x + c ≠ -x)) :
  |b^2 - 4 * a * c| > 1 := 
sorry

end NUMINAMATH_GPT_parabola_no_intersection_inequality_l1969_196904


namespace NUMINAMATH_GPT_inequality_proof_l1969_196995

theorem inequality_proof 
  (a b c d : ℝ) (n : ℕ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_n : 9 ≤ n) :
  a^n + b^n + c^n + d^n ≥ a^(n-9)*b^4*c^3*d^2 + b^(n-9)*c^4*d^3*a^2 + c^(n-9)*d^4*a^3*b^2 + d^(n-9)*a^4*b^3*c^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1969_196995


namespace NUMINAMATH_GPT_find_wrongly_noted_mark_l1969_196902

-- Definitions of given conditions
def average_marks := 100
def number_of_students := 25
def reported_correct_mark := 10
def correct_average_marks := 98
def wrongly_noted_mark : ℕ := sorry

-- Computing the sum with the wrong mark
def incorrect_sum := number_of_students * average_marks

-- Sum corrected by replacing wrong mark with correct mark
def sum_with_correct_replacement (wrongly_noted_mark : ℕ) := 
  incorrect_sum - wrongly_noted_mark + reported_correct_mark

-- Correct total sum for correct average
def correct_sum := number_of_students * correct_average_marks

-- The statement to be proven
theorem find_wrongly_noted_mark : wrongly_noted_mark = 60 :=
by sorry

end NUMINAMATH_GPT_find_wrongly_noted_mark_l1969_196902


namespace NUMINAMATH_GPT_candidate_a_votes_l1969_196916

theorem candidate_a_votes (x : ℕ) (h : 2 * x + x = 21) : 2 * x = 14 :=
by sorry

end NUMINAMATH_GPT_candidate_a_votes_l1969_196916


namespace NUMINAMATH_GPT_painted_cube_problem_l1969_196978

theorem painted_cube_problem (n : ℕ) (h1 : n > 2)
  (h2 : 6 * (n - 2)^2 = (n - 2)^3) : n = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_painted_cube_problem_l1969_196978


namespace NUMINAMATH_GPT_ax_by_n_sum_l1969_196907

theorem ax_by_n_sum {a b x y : ℝ} 
  (h1 : a * x + b * y = 2)
  (h2 : a * x^2 + b * y^2 = 5)
  (h3 : a * x^3 + b * y^3 = 15)
  (h4 : a * x^4 + b * y^4 = 35) :
  a * x^5 + b * y^5 = 10 :=
sorry

end NUMINAMATH_GPT_ax_by_n_sum_l1969_196907


namespace NUMINAMATH_GPT_salt_solution_l1969_196913

variable (x : ℝ) (v_water : ℝ) (c_initial : ℝ) (c_final : ℝ)

theorem salt_solution (h1 : v_water = 1) (h2 : c_initial = 0.60) (h3 : c_final = 0.20)
  (h4 : (v_water + x) * c_final = x * c_initial) :
  x = 0.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_salt_solution_l1969_196913


namespace NUMINAMATH_GPT_participants_initial_count_l1969_196962

theorem participants_initial_count (initial_participants remaining_after_first_round remaining_after_second_round : ℝ) 
  (h1 : remaining_after_first_round = 0.4 * initial_participants)
  (h2 : remaining_after_second_round = (1/4) * remaining_after_first_round)
  (h3 : remaining_after_second_round = 15) : 
  initial_participants = 150 :=
sorry

end NUMINAMATH_GPT_participants_initial_count_l1969_196962


namespace NUMINAMATH_GPT_no_solution_for_inequality_l1969_196934

theorem no_solution_for_inequality (x : ℝ) (h : |x| > 2) : ¬ (5 * x^2 + 6 * x + 8 < 0) := 
by
  sorry

end NUMINAMATH_GPT_no_solution_for_inequality_l1969_196934


namespace NUMINAMATH_GPT_polynomial_remainder_division_l1969_196924

theorem polynomial_remainder_division (x : ℝ) : 
  (x^4 + 1) % (x^2 - 4 * x + 6) = 16 * x - 59 := 
sorry

end NUMINAMATH_GPT_polynomial_remainder_division_l1969_196924


namespace NUMINAMATH_GPT_xyz_identity_l1969_196973

theorem xyz_identity (x y z : ℝ) 
  (h1 : x + y + z = 14) 
  (h2 : xy + xz + yz = 32) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 1400 := 
by 
  -- Proof steps will be placed here, use sorry for now
  sorry

end NUMINAMATH_GPT_xyz_identity_l1969_196973


namespace NUMINAMATH_GPT_sum_of_roots_l1969_196912

theorem sum_of_roots (a b c : ℝ) (h : 6 * a ^ 3 - 7 * a ^ 2 + 2 * a = 0 ∧ 
                                   6 * b ^ 3 - 7 * b ^ 2 + 2 * b = 0 ∧ 
                                   6 * c ^ 3 - 7 * c ^ 2 + 2 * c = 0 ∧ 
                                   a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
    a + b + c = 7 / 6 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1969_196912


namespace NUMINAMATH_GPT_sum_of_possible_values_l1969_196993

variable (a b c d : ℝ)

theorem sum_of_possible_values
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4) :
  (b - a) * (d - c) / ((c - b) * (a - d)) = -4 / 3 :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l1969_196993


namespace NUMINAMATH_GPT_snowdrift_depth_end_of_third_day_l1969_196970

theorem snowdrift_depth_end_of_third_day :
  let depth_ninth_day := 40
  let d_before_eighth_night_snowfall := depth_ninth_day - 10
  let d_before_eighth_day_melting := d_before_eighth_night_snowfall * 4 / 3
  let depth_seventh_day := d_before_eighth_day_melting
  let d_before_sixth_day_snowfall := depth_seventh_day - 20
  let d_before_fifth_day_snowfall := d_before_sixth_day_snowfall - 15
  let d_before_fourth_day_melting := d_before_fifth_day_snowfall * 3 / 2
  depth_ninth_day = 40 →
  d_before_eighth_night_snowfall = depth_ninth_day - 10 →
  d_before_eighth_day_melting = d_before_eighth_night_snowfall * 4 / 3 →
  depth_seventh_day = d_before_eighth_day_melting →
  d_before_sixth_day_snowfall = depth_seventh_day - 20 →
  d_before_fifth_day_snowfall = d_before_sixth_day_snowfall - 15 →
  d_before_fourth_day_melting = d_before_fifth_day_snowfall * 3 / 2 →
  d_before_fourth_day_melting = 7.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_snowdrift_depth_end_of_third_day_l1969_196970


namespace NUMINAMATH_GPT_estimated_watched_students_l1969_196928

-- Definitions for the problem conditions
def total_students : ℕ := 3600
def surveyed_students : ℕ := 200
def watched_students : ℕ := 160

-- Problem statement (proof not included yet)
theorem estimated_watched_students :
  total_students * (watched_students / surveyed_students : ℝ) = 2880 := by
  -- skipping proof step
  sorry

end NUMINAMATH_GPT_estimated_watched_students_l1969_196928


namespace NUMINAMATH_GPT_num_students_second_grade_l1969_196987

structure School :=
(total_students : ℕ)
(prob_male_first_grade : ℝ)

def stratified_sampling (school : School) : ℕ := sorry

theorem num_students_second_grade (school : School) (total_selected : ℕ) : 
    school.total_students = 4000 →
    school.prob_male_first_grade = 0.2 →
    total_selected = 100 →
    stratified_sampling school = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_num_students_second_grade_l1969_196987


namespace NUMINAMATH_GPT_altitude_on_hypotenuse_l1969_196941

theorem altitude_on_hypotenuse (a b : ℝ) (h₁ : a = 5) (h₂ : b = 12) (c : ℝ) (h₃ : c = Real.sqrt (a^2 + b^2)) :
  ∃ h : ℝ, h = (a * b) / c ∧ h = 60 / 13 :=
by
  use (5 * 12) / 13
  -- proof that (60 / 13) is indeed the altitude will be done by verifying calculations
  sorry

end NUMINAMATH_GPT_altitude_on_hypotenuse_l1969_196941


namespace NUMINAMATH_GPT_jamies_class_girls_count_l1969_196901

theorem jamies_class_girls_count 
  (g b : ℕ)
  (h_ratio : 4 * g = 3 * b)
  (h_total : g + b = 35) 
  : g = 15 := 
by 
  sorry 

end NUMINAMATH_GPT_jamies_class_girls_count_l1969_196901


namespace NUMINAMATH_GPT_distance_between_A_and_B_l1969_196900

noncomputable def time_from_A_to_B (D : ℝ) : ℝ := D / 200

noncomputable def time_from_B_to_A (D : ℝ) : ℝ := time_from_A_to_B D + 3

def condition (D : ℝ) : Prop := 
  D = 100 * (time_from_B_to_A D)

theorem distance_between_A_and_B :
  ∃ D : ℝ, condition D ∧ D = 600 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l1969_196900


namespace NUMINAMATH_GPT_luke_trays_l1969_196925

theorem luke_trays 
  (carries_per_trip : ℕ)
  (trips : ℕ)
  (second_table_trays : ℕ)
  (total_trays : carries_per_trip * trips = 36)
  (second_table_value : second_table_trays = 16) : 
  carries_per_trip * trips - second_table_trays = 20 :=
by sorry

end NUMINAMATH_GPT_luke_trays_l1969_196925


namespace NUMINAMATH_GPT_valid_sentence_count_is_208_l1969_196986

def four_words := ["splargh", "glumph", "amr", "flark"]

def valid_sentence (sentence : List String) : Prop :=
  ¬(sentence.contains "glumph amr")

def count_valid_sentences : Nat :=
  let total_sentences := 4^4
  let invalid_sentences := 3 * 4 * 4
  total_sentences - invalid_sentences

theorem valid_sentence_count_is_208 :
  count_valid_sentences = 208 := by
  sorry

end NUMINAMATH_GPT_valid_sentence_count_is_208_l1969_196986


namespace NUMINAMATH_GPT_find_c_l1969_196994

theorem find_c
  (c d : ℝ)
  (h1 : ∀ (x : ℝ), 7 * x^3 + 3 * c * x^2 + 6 * d * x + c = 0)
  (h2 : ∀ (p q r : ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
        7 * p^3 + 3 * c * p^2 + 6 * d * p + c = 0 ∧ 
        7 * q^3 + 3 * c * q^2 + 6 * d * q + c = 0 ∧ 
        7 * r^3 + 3 * c * r^2 + 6 * d * r + c = 0 ∧ 
        Real.log (p * q * r) / Real.log 3 = 3) :
  c = -189 :=
sorry

end NUMINAMATH_GPT_find_c_l1969_196994


namespace NUMINAMATH_GPT_sum_roots_quadratic_l1969_196968

theorem sum_roots_quadratic (a b c : ℝ) (P : ℝ → ℝ) 
  (hP : ∀ x : ℝ, P x = a * x^2 + b * x + c)
  (h : ∀ x : ℝ, P (2 * x^5 + 3 * x) ≥ P (3 * x^4 + 2 * x^2 + 1)) : 
  -b / a = 6 / 5 :=
sorry

end NUMINAMATH_GPT_sum_roots_quadratic_l1969_196968


namespace NUMINAMATH_GPT_regular_decagon_interior_angle_l1969_196975

theorem regular_decagon_interior_angle {n : ℕ} (h1 : n = 10) (h2 : ∀ (k : ℕ), k = 10 → (180 * (k - 2)) / 10 = 144) : 
  (∃ θ : ℕ, θ = 180 * (n - 2) / n ∧ n = 10 ∧ θ = 144) :=
by
  sorry

end NUMINAMATH_GPT_regular_decagon_interior_angle_l1969_196975


namespace NUMINAMATH_GPT_total_matches_l1969_196942

noncomputable def matches_in_tournament (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_matches :
  matches_in_tournament 5 + matches_in_tournament 7 + matches_in_tournament 4 = 37 := 
by 
  sorry

end NUMINAMATH_GPT_total_matches_l1969_196942


namespace NUMINAMATH_GPT_mathland_transport_l1969_196940

theorem mathland_transport (n : ℕ) (h : n ≥ 2) (transport : Fin n -> Fin n -> Prop) :
(∀ i j, transport i j ∨ transport j i) →
(∃ tr : Fin n -> Fin n -> Prop, 
  (∀ i j, transport i j → tr i j) ∨
  (∀ i j, transport j i → tr i j)) :=
by
  sorry

end NUMINAMATH_GPT_mathland_transport_l1969_196940


namespace NUMINAMATH_GPT_problem_l1969_196964

noncomputable def f (a b x : ℝ) := a * x^2 - b * x + 1

theorem problem (a b : ℝ) (h1 : 4 * a - b^2 = 3)
                (h2 : ∀ x : ℝ, f a b (x + 1) = f a b (-x))
                (h3 : b = a + 1) 
                (h4 : 0 ≤ a ∧ a ≤ 1) 
                (h5 : ∀ x ∈ Set.Icc 0 2, ∃ m : ℝ, m ≥ abs (f a b x)) :
  (∀ x : ℝ, f a b x = x^2 - x + 1) ∧ (∃ m : ℝ, m = 1 ∧ ∀ x ∈ Set.Icc 0 2, m ≥ abs (f a b x)) :=
  sorry

end NUMINAMATH_GPT_problem_l1969_196964


namespace NUMINAMATH_GPT_sum_inequality_l1969_196915

noncomputable def f (x : ℝ) : ℝ :=
  (3 * x^2 - x) / (1 + x^2)

theorem sum_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 1) : 
  f x + f y + f z ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_inequality_l1969_196915


namespace NUMINAMATH_GPT_product_of_real_solutions_of_t_cubed_eq_216_l1969_196963

theorem product_of_real_solutions_of_t_cubed_eq_216 : 
  (∃ t : ℝ, t^3 = 216) →
  (∀ t₁ t₂, (t₁ = t₂) → (t₁^3 = 216 → t₂^3 = 216) → (t₁ * t₂ = 6)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_real_solutions_of_t_cubed_eq_216_l1969_196963


namespace NUMINAMATH_GPT_remainder_2007_div_81_l1969_196980

theorem remainder_2007_div_81 : 2007 % 81 = 63 :=
by
  sorry

end NUMINAMATH_GPT_remainder_2007_div_81_l1969_196980


namespace NUMINAMATH_GPT_central_angle_measure_l1969_196960

-- Constants representing the arc length and the area of the sector.
def arc_length : ℝ := 5
def sector_area : ℝ := 5

-- Variables representing the central angle in radians and the radius.
variable (α r : ℝ)

-- Conditions given in the problem.
axiom arc_length_eq : arc_length = α * r
axiom sector_area_eq : sector_area = 1 / 2 * α * r^2

-- The goal to prove that the radian measure of the central angle α is 5 / 2.
theorem central_angle_measure : α = 5 / 2 := by sorry

end NUMINAMATH_GPT_central_angle_measure_l1969_196960


namespace NUMINAMATH_GPT_crayons_left_l1969_196996

def initial_green_crayons : ℝ := 5
def initial_blue_crayons : ℝ := 8
def initial_yellow_crayons : ℝ := 7
def given_green_crayons : ℝ := 3.5
def given_blue_crayons : ℝ := 1.25
def given_yellow_crayons : ℝ := 2.75
def broken_yellow_crayons : ℝ := 0.5

theorem crayons_left (initial_green_crayons initial_blue_crayons initial_yellow_crayons given_green_crayons given_blue_crayons given_yellow_crayons broken_yellow_crayons : ℝ) :
  initial_green_crayons - given_green_crayons + 
  initial_blue_crayons - given_blue_crayons + 
  initial_yellow_crayons - given_yellow_crayons - broken_yellow_crayons = 12 :=
by
  sorry

end NUMINAMATH_GPT_crayons_left_l1969_196996


namespace NUMINAMATH_GPT_find_x_weeks_l1969_196965

-- Definition of the problem conditions:
def archibald_first_two_weeks_apples : Nat := 14
def archibald_next_x_weeks_apples (x : Nat) : Nat := 14
def archibald_last_two_weeks_apples : Nat := 42
def total_weeks : Nat := 7
def weekly_average : Nat := 10

-- Statement of the theorem to prove that x = 2 given the conditions
theorem find_x_weeks :
  ∃ x : Nat, (archibald_first_two_weeks_apples + archibald_next_x_weeks_apples x + archibald_last_two_weeks_apples = total_weeks * weekly_average) 
  ∧ (archibald_next_x_weeks_apples x / x = 7) 
  → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_weeks_l1969_196965


namespace NUMINAMATH_GPT_arrange_p_q_r_l1969_196971

theorem arrange_p_q_r (p : ℝ) (h : 1 < p ∧ p < 1.1) : p < p^p ∧ p^p < p^(p^p) :=
by
  sorry

end NUMINAMATH_GPT_arrange_p_q_r_l1969_196971


namespace NUMINAMATH_GPT_product_of_sums_of_two_squares_l1969_196938

theorem product_of_sums_of_two_squares
  (a b a1 b1 : ℤ) :
  ((a^2 + b^2) * (a1^2 + b1^2)) = ((a * a1 - b * b1)^2 + (a * b1 + b * a1)^2) := 
sorry

end NUMINAMATH_GPT_product_of_sums_of_two_squares_l1969_196938


namespace NUMINAMATH_GPT_linear_independent_vectors_p_value_l1969_196950

theorem linear_independent_vectors_p_value (p : ℝ) :
  (∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ a * (2 : ℝ) + b * (5 : ℝ) = 0 ∧ a * (4 : ℝ) + b * p = 0) ↔ p = 10 :=
by
  sorry

end NUMINAMATH_GPT_linear_independent_vectors_p_value_l1969_196950


namespace NUMINAMATH_GPT_boxes_with_neither_l1969_196910

def total_boxes : ℕ := 15
def boxes_with_crayons : ℕ := 9
def boxes_with_markers : ℕ := 6
def boxes_with_both : ℕ := 4

theorem boxes_with_neither : total_boxes - (boxes_with_crayons + boxes_with_markers - boxes_with_both) = 4 := by
  sorry

end NUMINAMATH_GPT_boxes_with_neither_l1969_196910


namespace NUMINAMATH_GPT_eval_f_at_4_l1969_196989

def f (x : ℕ) : ℕ := 5 * x + 2

theorem eval_f_at_4 : f 4 = 22 :=
by
  sorry

end NUMINAMATH_GPT_eval_f_at_4_l1969_196989


namespace NUMINAMATH_GPT_select_team_ways_l1969_196981

-- Definitions of the conditions and question
def boys := 7
def girls := 10
def boys_needed := 2
def girls_needed := 3
def total_team := 5

-- Theorem statement to prove the number of selecting the team
theorem select_team_ways : (Nat.choose boys boys_needed) * (Nat.choose girls girls_needed) = 2520 := 
by
  -- Place holder for proof
  sorry

end NUMINAMATH_GPT_select_team_ways_l1969_196981


namespace NUMINAMATH_GPT_hand_towels_in_set_l1969_196961

theorem hand_towels_in_set {h : ℕ}
  (hand_towel_sets : ℕ)
  (bath_towel_sets : ℕ)
  (hand_towel_sold : h * hand_towel_sets = 102)
  (bath_towel_sold : 6 * bath_towel_sets = 102)
  (same_sets_sold : hand_towel_sets = bath_towel_sets) :
  h = 17 := 
sorry

end NUMINAMATH_GPT_hand_towels_in_set_l1969_196961


namespace NUMINAMATH_GPT_quiz_answer_key_combinations_l1969_196944

noncomputable def num_ways_answer_key : ℕ :=
  let true_false_combinations := 2^4
  let valid_true_false_combinations := true_false_combinations - 2
  let multi_choice_combinations := 4 * 4
  valid_true_false_combinations * multi_choice_combinations

theorem quiz_answer_key_combinations : num_ways_answer_key = 224 := 
by
  sorry

end NUMINAMATH_GPT_quiz_answer_key_combinations_l1969_196944


namespace NUMINAMATH_GPT_cheese_wedge_volume_l1969_196920

theorem cheese_wedge_volume (r h : ℝ) (n : ℕ) (V : ℝ) (π : ℝ) 
: r = 8 → h = 10 → n = 3 → V = π * r^2 * h → V / n = (640 * π) / 3  :=
by
  intros r_eq h_eq n_eq V_eq
  rw [r_eq, h_eq] at V_eq
  rw [V_eq]
  sorry

end NUMINAMATH_GPT_cheese_wedge_volume_l1969_196920


namespace NUMINAMATH_GPT_stickers_per_student_l1969_196982

theorem stickers_per_student : 
  ∀ (gold silver bronze total : ℕ), 
    gold = 50 →
    silver = 2 * gold →
    bronze = silver - 20 →
    total = gold + silver + bronze →
    total / 5 = 46 :=
by
  intros
  sorry

end NUMINAMATH_GPT_stickers_per_student_l1969_196982


namespace NUMINAMATH_GPT_additional_regular_gift_bags_needed_l1969_196984

-- Defining the conditions given in the question
def confirmed_guests : ℕ := 50
def additional_guests_70pc : ℕ := 30
def additional_guests_40pc : ℕ := 15
def probability_70pc : ℚ := 0.7
def probability_40pc : ℚ := 0.4
def extravagant_bags_prepared : ℕ := 10
def special_bags_prepared : ℕ := 25
def regular_bags_prepared : ℕ := 20

-- Defining the expected number of additional guests based on probabilities
def expected_guests_70pc : ℚ := additional_guests_70pc * probability_70pc
def expected_guests_40pc : ℚ := additional_guests_40pc * probability_40pc

-- Defining the total expected guests including confirmed guests and expected additional guests
def total_expected_guests : ℚ := confirmed_guests + expected_guests_70pc + expected_guests_40pc

-- Defining the problem statement in Lean, proving the additional regular gift bags needed
theorem additional_regular_gift_bags_needed : 
  total_expected_guests = 77 → regular_bags_prepared = 20 → 22 = 22 :=
by
  sorry

end NUMINAMATH_GPT_additional_regular_gift_bags_needed_l1969_196984


namespace NUMINAMATH_GPT_range_of_a_l1969_196979

noncomputable def f (a x : ℝ) : ℝ := Real.log x + 2 * a * (1 - x)

theorem range_of_a (a : ℝ) :
  (∀ x, x > 2 → f a x > f a 2) ↔ (a ∈ Set.Iic 0 ∪ Set.Ici (1 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1969_196979


namespace NUMINAMATH_GPT_intersection_of_sets_l1969_196903

-- Define sets A and B
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- State the theorem
theorem intersection_of_sets : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1969_196903


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1969_196909

theorem quadratic_inequality_solution_set (a b c : ℝ) (h₁ : a < 0) (h₂ : b^2 - 4 * a * c < 0) :
  ∀ x : ℝ, a * x^2 + b * x + c < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1969_196909


namespace NUMINAMATH_GPT_factorize_expression_l1969_196939

theorem factorize_expression : 989 * 1001 * 1007 + 320 = 991 * 997 * 1009 := 
by sorry

end NUMINAMATH_GPT_factorize_expression_l1969_196939


namespace NUMINAMATH_GPT_combinations_of_painting_options_l1969_196948

theorem combinations_of_painting_options : 
  let colors := 6
  let methods := 3
  let finishes := 2
  colors * methods * finishes = 36 := by
  sorry

end NUMINAMATH_GPT_combinations_of_painting_options_l1969_196948


namespace NUMINAMATH_GPT_total_turtles_taken_l1969_196921

theorem total_turtles_taken (number_of_green_turtles number_of_hawksbill_turtles total_number_of_turtles : ℕ)
  (h1 : number_of_green_turtles = 800)
  (h2 : number_of_hawksbill_turtles = 2 * number_of_green_turtles)
  (h3 : total_number_of_turtles = number_of_green_turtles + number_of_hawksbill_turtles) :
  total_number_of_turtles = 2400 :=
by
  sorry

end NUMINAMATH_GPT_total_turtles_taken_l1969_196921
