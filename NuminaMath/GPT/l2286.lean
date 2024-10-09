import Mathlib

namespace pizza_slices_left_over_l2286_228600

def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8
def small_pizzas_purchased : ℕ := 3
def large_pizzas_purchased : ℕ := 2
def george_slices : ℕ := 3
def bob_slices : ℕ := george_slices + 1
def susie_slices : ℕ := bob_slices / 2
def bill_slices : ℕ := 3
def fred_slices : ℕ := 3
def mark_slices : ℕ := 3

def total_pizza_slices : ℕ := (small_pizzas_purchased * small_pizza_slices) + (large_pizzas_purchased * large_pizza_slices)
def total_slices_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

theorem pizza_slices_left_over : total_pizza_slices - total_slices_eaten = 10 :=
by sorry

end pizza_slices_left_over_l2286_228600


namespace point_position_after_time_l2286_228629

noncomputable def final_position (initial : ℝ × ℝ) (velocity : ℝ × ℝ) (time : ℝ) : ℝ × ℝ :=
  (initial.1 + velocity.1 * time, initial.2 + velocity.2 * time)

theorem point_position_after_time :
  final_position (-10, 10) (4, -3) 5 = (10, -5) :=
by
  sorry

end point_position_after_time_l2286_228629


namespace least_number_of_attendees_l2286_228631

-- Definitions based on problem conditions
inductive Person
| Anna
| Bill
| Carl
deriving DecidableEq

inductive Day
| Mon
| Tues
| Wed
| Thurs
| Fri
deriving DecidableEq

def attends : Person → Day → Prop
| Person.Anna, Day.Mon => true
| Person.Anna, Day.Tues => false
| Person.Anna, Day.Wed => true
| Person.Anna, Day.Thurs => false
| Person.Anna, Day.Fri => false
| Person.Bill, Day.Mon => false
| Person.Bill, Day.Tues => true
| Person.Bill, Day.Wed => false
| Person.Bill, Day.Thurs => true
| Person.Bill, Day.Fri => true
| Person.Carl, Day.Mon => true
| Person.Carl, Day.Tues => true
| Person.Carl, Day.Wed => false
| Person.Carl, Day.Thurs => true
| Person.Carl, Day.Fri => false

-- Proof statement
theorem least_number_of_attendees : 
  (∀ d : Day, (∀ p : Person, attends p d → p = Person.Anna ∨ p = Person.Bill ∨ p = Person.Carl) ∧
              (d = Day.Wed ∨ d = Day.Fri → (∃ n : ℕ, n = 2 ∧ (∀ p : Person, attends p d → n = 2))) ∧
              (d = Day.Mon ∨ d = Day.Tues ∨ d = Day.Thurs → (∃ n : ℕ, n = 1 ∧ (∀ p : Person, attends p d → n = 1))) ∧
              ¬ (d = Day.Wed ∨ d = Day.Fri)) :=
sorry

end least_number_of_attendees_l2286_228631


namespace line_through_points_eq_l2286_228618

theorem line_through_points_eq
  (x1 y1 x2 y2 : ℝ)
  (h1 : 2 * x1 + 3 * y1 = 4)
  (h2 : 2 * x2 + 3 * y2 = 4) :
  ∃ m b : ℝ, (∀ x y : ℝ, (y = m * x + b) ↔ (2 * x + 3 * y = 4)) :=
by
  sorry

end line_through_points_eq_l2286_228618


namespace sufficient_condition_for_inequality_l2286_228666

theorem sufficient_condition_for_inequality (a b : ℝ) (h_nonzero : a * b ≠ 0) : (a < b ∧ b < 0) → (1 / a ^ 2 > 1 / b ^ 2) :=
by
  intro h
  sorry

end sufficient_condition_for_inequality_l2286_228666


namespace min_value_of_a_l2286_228687

theorem min_value_of_a :
  ∀ (x y : ℝ), |x| + |y| ≤ 1 → (|2 * x - 3 * y + 3 / 2| + |y - 1| + |2 * y - x - 3| ≤ 23 / 2) :=
by
  intros x y h
  sorry

end min_value_of_a_l2286_228687


namespace xiaoying_school_trip_l2286_228648

theorem xiaoying_school_trip :
  ∃ (x y : ℝ), 
    (1200 / 1000) = (3 / 60) * x + (5 / 60) * y ∧ 
    x + y = 16 :=
by
  sorry

end xiaoying_school_trip_l2286_228648


namespace find_angle_C_l2286_228657

noncomputable def angle_C_value (A B : ℝ) : ℝ :=
  180 - A - B

theorem find_angle_C (A B : ℝ) 
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) :
  angle_C_value A B = 30 :=
sorry

end find_angle_C_l2286_228657


namespace factor_expression_l2286_228651

theorem factor_expression (a b c : ℝ) :
  let num := (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3
  let denom := (a - b)^3 + (b - c)^3 + (c - a)^3
  (denom ≠ 0) →
  num / denom = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_expression_l2286_228651


namespace perfect_cubes_not_divisible_by_10_l2286_228603

-- Definitions based on conditions
def is_divisible_by_10 (n : ℕ) : Prop := 10 ∣ n
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k ^ 3
def erase_last_three_digits (n : ℕ) : ℕ := n / 1000

-- Main statement
theorem perfect_cubes_not_divisible_by_10 (x : ℕ) :
  is_perfect_cube x ∧ ¬ is_divisible_by_10 x ∧ is_perfect_cube (erase_last_three_digits x) →
  x = 1331 ∨ x = 1728 :=
by
  sorry

end perfect_cubes_not_divisible_by_10_l2286_228603


namespace base_conversion_positive_b_l2286_228609

theorem base_conversion_positive_b :
  (∃ (b : ℝ), 3 * 5^1 + 2 * 5^0 = 17 ∧ 1 * b^2 + 2 * b^1 + 0 * b^0 = 17 ∧ b = -1 + 3 * Real.sqrt 2) :=
by
  sorry

end base_conversion_positive_b_l2286_228609


namespace compute_expression_l2286_228638

theorem compute_expression :
  45 * 72 + 28 * 45 = 4500 :=
  sorry

end compute_expression_l2286_228638


namespace find_y_l2286_228610

-- Let s be the result of tripling both the base and exponent of c^d
-- Given the condition s = c^d * y^d, we need to prove y = 27c^2

variable (c d y : ℝ)
variable (h_d : d > 0)
variable (h : (3 * c)^(3 * d) = c^d * y^d)

theorem find_y (h_d : d > 0) (h : (3 * c)^(3 * d) = c^d * y^d) : y = 27 * c ^ 2 :=
by sorry

end find_y_l2286_228610


namespace number_of_pounds_of_vegetables_l2286_228645

-- Defining the conditions
def beef_cost_per_pound : ℕ := 6  -- Beef costs $6 per pound
def vegetable_cost_per_pound : ℕ := 2  -- Vegetables cost $2 per pound
def beef_pounds : ℕ := 4  -- Troy buys 4 pounds of beef
def total_cost : ℕ := 36  -- The total cost of everything is $36

-- Prove the number of pounds of vegetables Troy buys is 6
theorem number_of_pounds_of_vegetables (V : ℕ) :
  beef_cost_per_pound * beef_pounds + vegetable_cost_per_pound * V = total_cost → V = 6 :=
by
  sorry  -- Proof to be filled in later

end number_of_pounds_of_vegetables_l2286_228645


namespace train_times_comparison_l2286_228664

-- Defining the given conditions
variables (V1 T1 T2 D : ℝ)
variables (h1 : T1 = 2) (h2 : T2 = 7/3)
variables (train1_speed : V1 = D / T1)
variables (train2_speed : V2 = (3/5) * V1)

-- The proof statement to show that T2 is 1/3 hour longer than T1
theorem train_times_comparison 
  (h1 : (6/7) * V1 = D / (T1 + 1/3))
  (h2 : (3/5) * V1 = D / (T2 + 1)) :
  T2 - T1 = 1/3 :=
sorry

end train_times_comparison_l2286_228664


namespace find_first_month_sale_l2286_228661

/-- Given the sales for months two to six and the average sales over six months,
    prove the sale in the first month. -/
theorem find_first_month_sale
  (sales_2 : ℤ) (sales_3 : ℤ) (sales_4 : ℤ) (sales_5 : ℤ) (sales_6 : ℤ)
  (avg_sales : ℤ)
  (h2 : sales_2 = 5468) (h3 : sales_3 = 5568) (h4 : sales_4 = 6088)
  (h5 : sales_5 = 6433) (h6 : sales_6 = 5922) (h_avg : avg_sales = 5900) : 
  ∃ (sale_1 : ℤ), sale_1 = 5921 := 
by
  have total_sales : ℤ := avg_sales * 6
  have known_sales_sum : ℤ := sales_2 + sales_3 + sales_4 + sales_5
  use total_sales - known_sales_sum - sales_6
  sorry

end find_first_month_sale_l2286_228661


namespace infinite_series_evaluates_to_12_l2286_228696

noncomputable def infinite_series : ℝ :=
  ∑' k, (k^3) / (3^k)

theorem infinite_series_evaluates_to_12 :
  infinite_series = 12 :=
by
  sorry

end infinite_series_evaluates_to_12_l2286_228696


namespace hot_drink_sales_l2286_228663

theorem hot_drink_sales (x y : ℝ) (h : y = -2.35 * x + 147.7) (hx : x = 2) : y = 143 := 
by sorry

end hot_drink_sales_l2286_228663


namespace twenty_five_percent_less_than_80_is_twenty_five_percent_more_of_l2286_228671

theorem twenty_five_percent_less_than_80_is_twenty_five_percent_more_of (n : ℝ) (h : 1.25 * n = 80 - 0.25 * 80) : n = 48 :=
by
  sorry

end twenty_five_percent_less_than_80_is_twenty_five_percent_more_of_l2286_228671


namespace car_a_distance_behind_car_b_l2286_228693

theorem car_a_distance_behind_car_b :
  ∃ D : ℝ, D = 40 ∧ 
    (∀ (t : ℝ), t = 4 →
    ((58 - 50) * t + 8) = D + 8)
  := by
  sorry

end car_a_distance_behind_car_b_l2286_228693


namespace henry_games_given_l2286_228606

theorem henry_games_given (G : ℕ) (henry_initial : ℕ) (neil_initial : ℕ) (henry_now : ℕ) (neil_now : ℕ) :
  henry_initial = 58 →
  neil_initial = 7 →
  henry_now = henry_initial - G →
  neil_now = neil_initial + G →
  henry_now = 4 * neil_now →
  G = 6 :=
by
  intros h_initial n_initial h_now n_now eq_henry
  sorry

end henry_games_given_l2286_228606


namespace no_such_f_exists_l2286_228685

theorem no_such_f_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x : ℝ), f (f x) = x^2 - 2 := by
  sorry

end no_such_f_exists_l2286_228685


namespace find_total_pupils_l2286_228673

-- Define the conditions for the problem
def diff1 : ℕ := 85 - 45
def diff2 : ℕ := 79 - 49
def diff3 : ℕ := 64 - 34
def total_diff : ℕ := diff1 + diff2 + diff3
def avg_increase : ℕ := 3

-- Assert that the number of pupils n satisfies the given conditions
theorem find_total_pupils (n : ℕ) (h_diff : total_diff = 100) (h_avg_inc : avg_increase * n = total_diff) : n = 33 :=
by
  sorry

end find_total_pupils_l2286_228673


namespace band_row_lengths_l2286_228626

theorem band_row_lengths (n : ℕ) (h1 : n = 108) (h2 : ∃ k, 10 ≤ k ∧ k ≤ 18 ∧ 108 % k = 0) : 
  (∃ count : ℕ, count = 2) :=
by 
  sorry

end band_row_lengths_l2286_228626


namespace min_value_fraction_sum_l2286_228692

theorem min_value_fraction_sum (m n : ℝ) (h₁ : 0 < m) (h₂ : 0 < n) (h₃ : 2 * m + n = 1) : 
  (1 / m + 2 / n) ≥ 8 :=
sorry

end min_value_fraction_sum_l2286_228692


namespace ratio_of_brownies_l2286_228630

def total_brownies : ℕ := 15
def eaten_on_monday : ℕ := 5
def eaten_on_tuesday : ℕ := total_brownies - eaten_on_monday

theorem ratio_of_brownies : eaten_on_tuesday / eaten_on_monday = 2 := 
by
  sorry

end ratio_of_brownies_l2286_228630


namespace halfway_fraction_l2286_228678

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/7) : (a + b) / 2 = 41/56 :=
by
  sorry

end halfway_fraction_l2286_228678


namespace multiply_by_12_correct_result_l2286_228698

theorem multiply_by_12_correct_result (x : ℕ) (h : x / 14 = 42) : x * 12 = 7056 :=
by
  sorry

end multiply_by_12_correct_result_l2286_228698


namespace choir_average_age_l2286_228614

theorem choir_average_age
  (avg_females_age : ℕ)
  (num_females : ℕ)
  (avg_males_age : ℕ)
  (num_males : ℕ)
  (females_avg_condition : avg_females_age = 28)
  (females_num_condition : num_females = 8)
  (males_avg_condition : avg_males_age = 32)
  (males_num_condition : num_males = 17) :
  ((avg_females_age * num_females + avg_males_age * num_males) / (num_females + num_males) = 768 / 25) :=
by
  sorry

end choir_average_age_l2286_228614


namespace range_of_g_l2286_228628

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos x)^2 - (Real.arcsin x)^2

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -((Real.pi^2) / 4) ≤ g x ∧ g x ≤ (3 * (Real.pi^2)) / 4 :=
by
  intros x hx
  sorry

end range_of_g_l2286_228628


namespace f_monotonic_decreasing_interval_l2286_228646

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 2*x)

theorem f_monotonic_decreasing_interval : 
  ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 ≤ x2 → f x2 ≤ f x1 := 
sorry

end f_monotonic_decreasing_interval_l2286_228646


namespace expected_value_full_circles_l2286_228649

-- Definition of the conditions
def num_small_triangles (n : ℕ) : ℕ :=
  n^2

def potential_full_circle_vertices (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2) * (n - 1) / 2

def prob_full_circle : ℚ :=
  1 / 729

-- The expected number of full circles formed
def expected_full_circles (n : ℕ) : ℚ :=
  potential_full_circle_vertices n * prob_full_circle

-- The mathematical equivalence to be proved
theorem expected_value_full_circles (n : ℕ) : expected_full_circles n = (n - 2) * (n - 1) / 1458 := 
  sorry

end expected_value_full_circles_l2286_228649


namespace rectangle_perimeter_l2286_228605

theorem rectangle_perimeter (a b c width : ℕ) (area : ℕ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) 
  (h5 : area = (a * b) / 2) 
  (h6 : width = 5) 
  (h7 : area = width * ((area * 2) / (a * b)))
  : 2 * (width + (area / width)) = 22 := 
by 
  sorry

end rectangle_perimeter_l2286_228605


namespace senior_junior_ratio_l2286_228674

variable (S J : ℕ) (k : ℕ)

theorem senior_junior_ratio (h1 : S = k * J) 
                           (h2 : (1/8 : ℚ) * S + (3/4 : ℚ) * J = (1/3 : ℚ) * (S + J)) : 
                           k = 2 :=
by
  sorry

end senior_junior_ratio_l2286_228674


namespace find_pair_l2286_228675

theorem find_pair (a b : ℤ) :
  (∀ x : ℝ, (a * x^4 + b * x^3 + 20 * x^2 - 12 * x + 10) = (2 * x^2 + 3 * x - 4) * (c * x^2 + d * x + e)) → 
  (a = 2) ∧ (b = 27) :=
sorry

end find_pair_l2286_228675


namespace price_of_silver_l2286_228616

theorem price_of_silver
  (side : ℕ) (side_eq : side = 3)
  (weight_per_cubic_inch : ℕ) (weight_per_cubic_inch_eq : weight_per_cubic_inch = 6)
  (selling_price : ℝ) (selling_price_eq : selling_price = 4455)
  (markup_percentage : ℝ) (markup_percentage_eq : markup_percentage = 1.10)
  : 4050 / 162 = 25 :=
by
  -- Given conditions are side_eq, weight_per_cubic_inch_eq, selling_price_eq, and markup_percentage_eq
  -- The statement requiring proof, i.e., price per ounce calculation, is provided.
  sorry

end price_of_silver_l2286_228616


namespace ticket_cost_l2286_228658

theorem ticket_cost (total_amount_collected : ℕ) (average_tickets_per_day : ℕ) (days : ℕ) 
  (h1 : total_amount_collected = 960) 
  (h2 : average_tickets_per_day = 80) 
  (h3 : days = 3) : 
  total_amount_collected / (average_tickets_per_day * days) = 4 :=
  sorry

end ticket_cost_l2286_228658


namespace escalator_rate_is_15_l2286_228602

noncomputable def rate_escalator_moves (escalator_length : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_length / time) - person_speed

theorem escalator_rate_is_15 :
  rate_escalator_moves 200 5 10 = 15 := by
  sorry

end escalator_rate_is_15_l2286_228602


namespace large_circuit_longer_l2286_228684

theorem large_circuit_longer :
  ∀ (small_circuit_length large_circuit_length : ℕ),
  ∀ (laps_jana laps_father : ℕ),
  laps_jana = 3 →
  laps_father = 4 →
  (laps_father * large_circuit_length = 2 * (laps_jana * small_circuit_length)) →
  small_circuit_length = 400 →
  large_circuit_length - small_circuit_length = 200 :=
by
  intros small_circuit_length large_circuit_length laps_jana laps_father
  intros h_jana_laps h_father_laps h_distance h_small_length
  sorry

end large_circuit_longer_l2286_228684


namespace sum_of_decimals_is_fraction_l2286_228617

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end sum_of_decimals_is_fraction_l2286_228617


namespace max_value_6a_3b_10c_l2286_228682

theorem max_value_6a_3b_10c (a b c : ℝ) (h : 9 * a ^ 2 + 4 * b ^ 2 + 25 * c ^ 2 = 1) : 
  6 * a + 3 * b + 10 * c ≤ (Real.sqrt 41) / 2 :=
sorry

end max_value_6a_3b_10c_l2286_228682


namespace larger_number_is_23_l2286_228642

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l2286_228642


namespace janet_initial_stickers_l2286_228637

variable (x : ℕ)

theorem janet_initial_stickers (h : x + 53 = 56) : x = 3 := by
  sorry

end janet_initial_stickers_l2286_228637


namespace total_cost_john_paid_l2286_228647

theorem total_cost_john_paid 
  (meters_of_cloth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ)
  (h1 : meters_of_cloth = 9.25)
  (h2 : cost_per_meter = 48)
  (h3 : total_cost = meters_of_cloth * cost_per_meter) :
  total_cost = 444 :=
sorry

end total_cost_john_paid_l2286_228647


namespace bran_tuition_fee_l2286_228694

theorem bran_tuition_fee (P : ℝ) (S : ℝ) (M : ℕ) (R : ℝ) (T : ℝ) 
  (h1 : P = 15) (h2 : S = 0.30) (h3 : M = 3) (h4 : R = 18) 
  (h5 : 0.70 * T - (M * P) = R) : T = 90 :=
by
  sorry

end bran_tuition_fee_l2286_228694


namespace find_speed_of_car_y_l2286_228622

noncomputable def average_speed_of_car_y (sₓ : ℝ) (delay : ℝ) (d_afterₓ_started : ℝ) : ℝ :=
  let tₓ_before := delay
  let dₓ_before := sₓ * tₓ_before
  let total_dₓ := dₓ_before + d_afterₓ_started
  let tₓ_after := d_afterₓ_started / sₓ
  let total_time_y := tₓ_after
  d_afterₓ_started / total_time_y

theorem find_speed_of_car_y (h₁ : ∀ t, t = 1.2) (h₂ : ∀ sₓ, sₓ = 35) (h₃ : ∀ d_afterₓ_started, d_afterₓ_started = 42) : 
  average_speed_of_car_y 35 1.2 42 = 35 := by
  unfold average_speed_of_car_y
  simp
  sorry

end find_speed_of_car_y_l2286_228622


namespace y_paisa_for_each_rupee_x_l2286_228615

theorem y_paisa_for_each_rupee_x (p : ℕ) (x : ℕ) (y_share total_amount : ℕ) 
  (h₁ : y_share = 2700) 
  (h₂ : total_amount = 10500) 
  (p_condition : (130 + p) * x = total_amount) 
  (y_condition : p * x = y_share) : 
  p = 45 := 
by
  sorry

end y_paisa_for_each_rupee_x_l2286_228615


namespace successful_pair_exists_another_with_same_arithmetic_mean_l2286_228641

theorem successful_pair_exists_another_with_same_arithmetic_mean
  (a b : ℕ)
  (h_distinct : a ≠ b)
  (h_arith_mean_nat : ∃ m : ℕ, 2 * m = a + b)
  (h_geom_mean_nat : ∃ g : ℕ, g * g = a * b) :
  ∃ (c d : ℕ), c ≠ d ∧ ∃ m' : ℕ, 2 * m' = c + d ∧ ∃ g' : ℕ, g' * g' = c * d ∧ m' = (a + b) / 2 :=
sorry

end successful_pair_exists_another_with_same_arithmetic_mean_l2286_228641


namespace shipping_cost_per_unit_l2286_228601

-- Define the conditions
def cost_per_component : ℝ := 80
def fixed_monthly_cost : ℝ := 16500
def num_components : ℝ := 150
def lowest_selling_price : ℝ := 196.67

-- Define the revenue and total cost
def total_cost (S : ℝ) : ℝ := (cost_per_component * num_components) + fixed_monthly_cost + (num_components * S)
def total_revenue : ℝ := lowest_selling_price * num_components

-- Define the proposition to be proved
theorem shipping_cost_per_unit (S : ℝ) :
  total_cost S ≤ total_revenue → S ≤ 6.67 :=
by sorry

end shipping_cost_per_unit_l2286_228601


namespace collinear_condition_l2286_228619

theorem collinear_condition {a b c d : ℝ} (h₁ : a < b) (h₂ : c < d) (h₃ : a < d) (h₄ : c < b) :
  (a / d) + (c / b) = 1 := 
sorry

end collinear_condition_l2286_228619


namespace find_ordered_pair_l2286_228679

theorem find_ordered_pair (s m : ℚ) :
  (∃ t : ℚ, (5 * s - 7 = 2) ∧ 
           ((∃ (t1 : ℚ), (x = s + 3 * t1) ∧  (y = 2 + m * t1)) 
           → (x = 24 / 5) → (y = 5))) →
  (s = 9 / 5 ∧ m = 3) :=
by
  sorry

end find_ordered_pair_l2286_228679


namespace remaining_student_number_l2286_228660

-- Definitions based on given conditions
def total_students := 48
def sample_size := 6
def sampled_students := [5, 21, 29, 37, 45]

-- Interval calculation and pattern definition based on systematic sampling
def sampling_interval := total_students / sample_size
def sampled_student_numbers (n : Nat) : Nat := 5 + sampling_interval * (n - 1)

-- Prove the student number within the sample
theorem remaining_student_number : ∃ n, n ∉ sampled_students ∧ sampled_student_numbers n = 13 :=
by
  sorry

end remaining_student_number_l2286_228660


namespace PQR_product_l2286_228632

def PQR_condition (P Q R S : ℕ) : Prop :=
  P + Q + R + S = 100 ∧
  ∃ x : ℕ, P = x - 4 ∧ Q = x + 4 ∧ R = x / 4 ∧ S = 4 * x

theorem PQR_product (P Q R S : ℕ) (h : PQR_condition P Q R S) : P * Q * R * S = 61440 :=
by 
  sorry

end PQR_product_l2286_228632


namespace evaluate_expression_l2286_228656

variable (x y : ℚ)

theorem evaluate_expression 
  (hx : x = 2) 
  (hy : y = -1 / 5) : 
  (2 * x - 3)^2 - (x + 2 * y) * (x - 2 * y) - 3 * y^2 + 3 = 1 / 25 :=
by
  sorry

end evaluate_expression_l2286_228656


namespace minimum_distance_l2286_228688

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y + 4 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 8 * x

theorem minimum_distance :
  ∃ (A B : ℝ × ℝ), circle_eq A.1 A.2 ∧ parabola_eq B.1 B.2 ∧ dist A B = 1 / 2 :=
sorry

end minimum_distance_l2286_228688


namespace correct_result_without_mistake_l2286_228639

variable {R : Type*} [CommRing R] (a b c : R)
variable (A : R)

theorem correct_result_without_mistake :
  A + 2 * (ab + 2 * bc - 4 * ac) = (3 * ab - 2 * ac + 5 * bc) → 
  A - 2 * (ab + 2 * bc - 4 * ac) = -ab + 14 * ac - 3 * bc :=
by
  sorry

end correct_result_without_mistake_l2286_228639


namespace ram_total_distance_l2286_228608

noncomputable def total_distance 
  (speed1 speed2 time1 total_time : ℝ) 
  (h_speed1 : speed1 = 20) 
  (h_speed2 : speed2 = 70)
  (h_time1 : time1 = 3.2)
  (h_total_time : total_time = 8) 
  : ℝ := 
  speed1 * time1 + speed2 * (total_time - time1)

theorem ram_total_distance
  (speed1 speed2 time1 total_time : ℝ)
  (h_speed1 : speed1 = 20)
  (h_speed2 : speed2 = 70)
  (h_time1 : time1 = 3.2)
  (h_total_time : total_time = 8)
  : total_distance speed1 speed2 time1 total_time h_speed1 h_speed2 h_time1 h_total_time = 400 :=
  sorry

end ram_total_distance_l2286_228608


namespace odd_multiple_of_9_implies_multiple_of_3_l2286_228604

-- Define an odd number that is a multiple of 9
def odd_multiple_of_nine (m : ℤ) : Prop := 9 * m % 2 = 1

-- Define multiples of 3 and 9
def multiple_of_three (n : ℤ) : Prop := ∃ k : ℤ, n = 3 * k
def multiple_of_nine (n : ℤ) : Prop := ∃ k : ℤ, n = 9 * k

-- The main statement
theorem odd_multiple_of_9_implies_multiple_of_3 (n : ℤ) 
  (h1 : ∀ n, multiple_of_nine n → multiple_of_three n)
  (h2 : odd_multiple_of_nine n ∧ multiple_of_nine n) : 
  multiple_of_three n :=
by sorry

end odd_multiple_of_9_implies_multiple_of_3_l2286_228604


namespace min_distance_from_C_to_circle_l2286_228607

theorem min_distance_from_C_to_circle
  (R : ℝ) (AC : ℝ) (CB : ℝ) (C M : ℝ)
  (hR : R = 6) (hAC : AC = 4) (hCB : CB = 5)
  (hCM_eq : C = 12 - M) :
  C * M = 20 → (M < 6) → M = 2 := 
sorry

end min_distance_from_C_to_circle_l2286_228607


namespace m_greater_than_p_l2286_228681

theorem m_greater_than_p (p m n : ℕ) (pp : Nat.Prime p) (pos_m : m > 0) (pos_n : n > 0) (h : p^2 + m^2 = n^2) : m > p :=
sorry

end m_greater_than_p_l2286_228681


namespace incorrect_statement_B_l2286_228667

-- Define the plane vector operation "☉".
def vector_operation (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

-- Define the mathematical problem based on the given conditions.
theorem incorrect_statement_B (a b : ℝ × ℝ) : vector_operation a b ≠ vector_operation b a := by
  sorry

end incorrect_statement_B_l2286_228667


namespace find_x_l2286_228665

theorem find_x (x : ℝ) : (1 + (1 / (1 + x)) = 2 * (1 / (1 + x))) → x = 0 :=
by
  intro h
  sorry

end find_x_l2286_228665


namespace problem1_problem2_problem3_l2286_228636

-- Definition of operation T
def T (x y m n : ℚ) := (m * x + n * y) * (x + 2 * y)

-- Problem 1: Given T(1, -1) = 0 and T(0, 2) = 8, prove m = 1 and n = 1
theorem problem1 (m n : ℚ) (h1 : T 1 (-1) m n = 0) (h2 : T 0 2 m n = 8) : m = 1 ∧ n = 1 := by
  sorry

-- Problem 2: Given the system of inequalities in terms of p and knowing T(x, y) = (mx + ny)(x + 2y) with m = 1 and n = 1
--            has exactly 3 integer solutions, prove the range of values for a is 42 ≤ a < 54
theorem problem2 (a : ℚ) 
  (h1 : ∃ p : ℚ, T (2 * p) (2 - p) 1 1 > 4 ∧ T (4 * p) (3 - 2 * p) 1 1 ≤ a)
  (h2 : ∃! p : ℤ, -1 < p ∧ p ≤ (a - 18) / 12) : 42 ≤ a ∧ a < 54 := by
  sorry

-- Problem 3: Given T(x, y) = T(y, x) when x^2 ≠ y^2, prove m = 2n
theorem problem3 (m n : ℚ) 
  (h : ∀ x y : ℚ, x^2 ≠ y^2 → T x y m n = T y x m n) : m = 2 * n := by
  sorry

end problem1_problem2_problem3_l2286_228636


namespace convex_quadrilateral_max_two_obtuse_l2286_228624

theorem convex_quadrilateral_max_two_obtuse (a b c d : ℝ)
  (h1 : a + b + c + d = 360)
  (h2 : a < 180) (h3 : b < 180) (h4 : c < 180) (h5 : d < 180)
  : (∃ A1 A2, a = A1 ∧ b = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ c < 90 ∧ d < 90) ∨
    (∃ A1 A2, a = A1 ∧ c = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ b < 90 ∧ d < 90) ∨
    (∃ A1 A2, a = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ b < 90 ∧ c < 90) ∨
    (∃ A1 A2, b = A1 ∧ c = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ d < 90) ∨
    (∃ A1 A2, b = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ c < 90) ∨
    (∃ A1 A2, c = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ b < 90) ∨
    (¬∃ x y z, (x > 90) ∧ (y > 90) ∧ (z > 90) ∧ x + y + z ≤ 360) := sorry

end convex_quadrilateral_max_two_obtuse_l2286_228624


namespace single_elimination_game_count_l2286_228680

theorem single_elimination_game_count (n : Nat) (h : n = 23) : n - 1 = 22 :=
by
  sorry

end single_elimination_game_count_l2286_228680


namespace negation_of_exists_l2286_228668

theorem negation_of_exists (h : ∃ x : ℝ, x > 0 ∧ x^2 + 3*x + 1 < 0) : ∀ x : ℝ, x > 0 → x^2 + 3*x + 1 ≥ 0 :=
sorry

end negation_of_exists_l2286_228668


namespace set_intersection_complement_l2286_228611

-- Definitions of the sets A and B
def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

-- Statement of the problem for Lean 4
theorem set_intersection_complement :
  A ∩ (Set.compl B) = {1, 5, 7} := 
sorry

end set_intersection_complement_l2286_228611


namespace find_k_for_parallel_lines_l2286_228627

theorem find_k_for_parallel_lines (k : ℝ) :
  (∀ x y : ℝ, (k - 2) * x + (4 - k) * y + 1 = 0) →
  (∀ x y : ℝ, 2 * (k - 2) * x - 2 * y + 3 = 0) →
  (k = 2 ∨ k = 5) :=
sorry

end find_k_for_parallel_lines_l2286_228627


namespace cube_division_l2286_228697

theorem cube_division (n : ℕ) (hn1 : 6 ≤ n) (hn2 : n % 2 = 0) : 
  ∃ m : ℕ, (n = 2 * m) ∧ (∀ a : ℕ, ∀ b : ℕ, ∀ c: ℕ, a = m^3 - (m - 1)^3 + 1 → b = 3 * m * (m - 1) + 2 → a = b) :=
by
  sorry

end cube_division_l2286_228697


namespace fourteen_divisible_by_7_twenty_eight_divisible_by_7_thirty_five_divisible_by_7_forty_nine_divisible_by_7_l2286_228659

def is_divisible_by_7 (n: ℕ): Prop := n % 7 = 0

theorem fourteen_divisible_by_7: is_divisible_by_7 14 :=
by
  sorry

theorem twenty_eight_divisible_by_7: is_divisible_by_7 28 :=
by
  sorry

theorem thirty_five_divisible_by_7: is_divisible_by_7 35 :=
by
  sorry

theorem forty_nine_divisible_by_7: is_divisible_by_7 49 :=
by
  sorry

end fourteen_divisible_by_7_twenty_eight_divisible_by_7_thirty_five_divisible_by_7_forty_nine_divisible_by_7_l2286_228659


namespace problem_inequality_solution_set_inequality_proof_l2286_228650

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem problem_inequality_solution_set :
  {x : ℝ | f x < 4} = {x : ℝ | -2 < x ∧ x < 2} :=
sorry

theorem inequality_proof (x y : ℝ) (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) :
  |x + y| < |(x * y) / 2 + 2| :=
sorry

end problem_inequality_solution_set_inequality_proof_l2286_228650


namespace red_crayons_count_l2286_228644

variable (R : ℕ) -- Number of red crayons
variable (B : ℕ) -- Number of blue crayons
variable (Y : ℕ) -- Number of yellow crayons

-- Conditions
axiom h1 : B = R + 5
axiom h2 : Y = 2 * B - 6
axiom h3 : Y = 32

-- Statement to prove
theorem red_crayons_count : R = 14 :=
by
  sorry

end red_crayons_count_l2286_228644


namespace geometric_sequence_term_l2286_228652

theorem geometric_sequence_term (a : ℕ → ℕ) (q : ℕ) (hq : q = 2) (ha2 : a 2 = 8) :
  a 6 = 128 :=
by
  sorry

end geometric_sequence_term_l2286_228652


namespace boat_downstream_distance_l2286_228662

theorem boat_downstream_distance 
  (Vb Vr T D U : ℝ)
  (h1 : Vb + Vr = 21)
  (h2 : Vb - Vr = 12)
  (h3 : U = 48)
  (h4 : T = 4)
  (h5 : D = 20) :
  (Vb + Vr) * D = 420 :=
by
  sorry

end boat_downstream_distance_l2286_228662


namespace ratio_dislikes_to_likes_l2286_228683

theorem ratio_dislikes_to_likes 
  (D : ℕ) 
  (h1 : D + 1000 = 2600) 
  (h2 : 3000 > 0) : 
  D / 3000 = 8 / 15 :=
by sorry

end ratio_dislikes_to_likes_l2286_228683


namespace hcf_of_210_and_671_l2286_228620

theorem hcf_of_210_and_671 :
  let lcm := 2310
  let a := 210
  let b := 671
  gcd a b = 61 :=
by
  let lcm := 2310
  let a := 210
  let b := 671
  let hcf := gcd a b
  have rel : lcm * hcf = a * b := by sorry
  have hcf_eq : hcf = 61 := by sorry
  exact hcf_eq

end hcf_of_210_and_671_l2286_228620


namespace fixed_point_range_l2286_228669

theorem fixed_point_range (a : ℝ) : (∃ x : ℝ, x = x^2 + x + a) → a ≤ 0 :=
sorry

end fixed_point_range_l2286_228669


namespace smallest_period_pi_max_value_min_value_l2286_228699

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

open Real

theorem smallest_period_pi : ∀ x, f (x + π) = f x := by
  unfold f
  intros
  sorry

theorem max_value : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≤ 1 + sqrt 2 := by
  unfold f
  intros
  sorry

theorem min_value : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≥ 0 := by
  unfold f
  intros
  sorry

end smallest_period_pi_max_value_min_value_l2286_228699


namespace parabola_ellipse_tangency_l2286_228686

theorem parabola_ellipse_tangency :
  ∃ (a b : ℝ), (∀ x y, y = x^2 - 5 → (x^2 / a) + (y^2 / b) = 1) →
               (∃ x, y = x^2 - 5 ∧ (x^2 / a) + ((x^2 - 5)^2 / b) = 1) ∧
               a = 1/10 ∧ b = 1 :=
by
  sorry

end parabola_ellipse_tangency_l2286_228686


namespace kite_area_correct_l2286_228634

open Real

structure Point where
  x : ℝ
  y : ℝ

def Kite (p1 p2 p3 p4 : Point) : Prop :=
  let triangle_area (a b c : Point) : ℝ :=
    abs (0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)))
  triangle_area p1 p2 p4 + triangle_area p1 p3 p4 = 102

theorem kite_area_correct : ∃ (p1 p2 p3 p4 : Point), 
  p1 = Point.mk 0 10 ∧ 
  p2 = Point.mk 6 14 ∧ 
  p3 = Point.mk 12 10 ∧ 
  p4 = Point.mk 6 0 ∧ 
  Kite p1 p2 p3 p4 :=
by
  sorry

end kite_area_correct_l2286_228634


namespace ratio_cost_to_marked_price_l2286_228653

theorem ratio_cost_to_marked_price (x : ℝ) 
  (h_discount: ∀ y, y = marked_price → selling_price = (3/4) * y)
  (h_cost: ∀ z, z = selling_price → cost_price = (2/3) * z) :
  cost_price / marked_price = 1 / 2 :=
by
  sorry

end ratio_cost_to_marked_price_l2286_228653


namespace find_m_find_min_value_l2286_228643

-- Conditions
def A (m : ℤ) : Set ℝ := { x | abs (x + 1) + abs (x - m) < 5 }

-- First Problem: Prove m = 3 given 3 ∈ A
theorem find_m (m : ℤ) (h : 3 ∈ A m) : m = 3 := sorry

-- Second Problem: Prove a^2 + b^2 + c^2 ≥ 1 given a + 2b + 2c = 3
theorem find_min_value (a b c : ℝ) (h : a + 2 * b + 2 * c = 3) : (a^2 + b^2 + c^2) ≥ 1 := sorry

end find_m_find_min_value_l2286_228643


namespace geometric_sequence_a_sequence_b_l2286_228654

theorem geometric_sequence_a (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : 2 * a 2 + a 3 = 60) :
  ∀ n, a n = 4 * 3^(n - 1) :=
sorry

theorem sequence_b (b a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : 2 * a 2 + a 3 = 60)
  (h3 : ∀ n, b (n + 1) = b n + a n) (h4 : b 1 = a 2) :
  ∀ n, b n = 2 * 3^n + 10 :=
sorry

end geometric_sequence_a_sequence_b_l2286_228654


namespace find_third_number_l2286_228690

theorem find_third_number (x : ℝ) (third_number : ℝ) : 
  0.6 / 0.96 = third_number / 8 → x = 0.96 → third_number = 5 :=
by
  intro h1 h2
  sorry

end find_third_number_l2286_228690


namespace ellipse_equation_l2286_228613

-- Definitions based on the problem conditions
def hyperbola_foci (x y : ℝ) : Prop := 2 * x^2 - 2 * y^2 = 1
def passes_through_point (p : ℝ × ℝ) (x y : ℝ) : Prop := p = (1, -3 / 2)

-- The statement to be proved
theorem ellipse_equation (c : ℝ) (a b : ℝ) :
    hyperbola_foci (-1) 0 ∧ hyperbola_foci 1 0 ∧
    passes_through_point (1, -3 / 2) 1 (-3 / 2) ∧
    (a = 2) ∧ (b = Real.sqrt 3) ∧ (c = 1)
    → ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 :=
by
  sorry

end ellipse_equation_l2286_228613


namespace eval_custom_op_l2286_228635

def custom_op (a b : ℤ) : ℤ := 2 * b + 5 * a - a^2 - b

theorem eval_custom_op : custom_op 3 4 = 10 :=
by
  sorry

end eval_custom_op_l2286_228635


namespace sequence_inequality_l2286_228621

theorem sequence_inequality (a : ℕ → ℝ) (h1 : ∀ n m : ℕ, a (n + m) ≤ a n + a m)
  (h2 : ∀ n : ℕ, 0 ≤ a n) (n m : ℕ) (hnm : n ≥ m) : 
  a n ≤ m * a 1 + (n / m - 1) * a m :=
sorry

end sequence_inequality_l2286_228621


namespace combined_total_score_l2286_228670

-- Define the conditions
def num_single_answer_questions : ℕ := 50
def num_multiple_answer_questions : ℕ := 20
def single_answer_score : ℕ := 2
def multiple_answer_score : ℕ := 4
def wrong_single_penalty : ℕ := 1
def wrong_multiple_penalty : ℕ := 2
def jose_wrong_single : ℕ := 10
def jose_wrong_multiple : ℕ := 5
def jose_lost_marks : ℕ := (jose_wrong_single * wrong_single_penalty) + (jose_wrong_multiple * wrong_multiple_penalty)
def jose_correct_single : ℕ := num_single_answer_questions - jose_wrong_single
def jose_correct_multiple : ℕ := num_multiple_answer_questions - jose_wrong_multiple
def jose_single_score : ℕ := jose_correct_single * single_answer_score
def jose_multiple_score : ℕ := jose_correct_multiple * multiple_answer_score
def jose_score : ℕ := (jose_single_score + jose_multiple_score) - jose_lost_marks
def alison_score : ℕ := jose_score - 50
def meghan_score : ℕ := jose_score - 30

-- Prove the combined total score
theorem combined_total_score :
  jose_score + alison_score + meghan_score = 280 :=
by
  sorry

end combined_total_score_l2286_228670


namespace vector_c_condition_l2286_228691

variables (a b c : ℝ × ℝ)

def is_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def is_parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v = (k * w.1, k * w.2)

theorem vector_c_condition (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (2, -3)) 
  (hc : c = (7 / 2, -7 / 4)) :
  is_perpendicular c a ∧ is_parallel b (a - c) :=
sorry

end vector_c_condition_l2286_228691


namespace orthocenter_of_triangle_l2286_228677

theorem orthocenter_of_triangle (A : ℝ × ℝ) (x y : ℝ) 
  (h₁ : x + y = 0) (h₂ : 2 * x - 3 * y + 1 = 0) : 
  A = (1, 2) → (x, y) = (-1 / 5, 1 / 5) :=
by
  sorry

end orthocenter_of_triangle_l2286_228677


namespace find_number_l2286_228612

theorem find_number (x : ℝ) (h : x / 5 = 30 + x / 6) : x = 900 :=
  sorry

end find_number_l2286_228612


namespace exists_real_A_l2286_228655

theorem exists_real_A (t : ℝ) (n : ℕ) (h_root: t^2 - 10 * t + 1 = 0) :
  ∃ A : ℝ, (A = t) ∧ ∀ n : ℕ, ∃ k : ℕ, A^n + 1/(A^n) - k^2 = 2 :=
by
  sorry

end exists_real_A_l2286_228655


namespace seedlings_planted_l2286_228676

theorem seedlings_planted (x : ℕ) (h1 : 2 * x + x = 1200) : x = 400 :=
by {
  sorry
}

end seedlings_planted_l2286_228676


namespace max_ab_condition_l2286_228623

theorem max_ab_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 4 = 0)
  (line_check : ∀ x y : ℝ, (x = 1 ∧ y = -2) → 2*a*x - b*y - 2 = 0) : ab ≤ 1/4 :=
by
  sorry

end max_ab_condition_l2286_228623


namespace max_value_2ac_minus_abc_l2286_228625

theorem max_value_2ac_minus_abc (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 7) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c <= 4) : 
  2 * a * c - a * b * c ≤ 28 :=
sorry

end max_value_2ac_minus_abc_l2286_228625


namespace meaningful_iff_x_ne_2_l2286_228672

theorem meaningful_iff_x_ne_2 (x : ℝ) : (x ≠ 2) ↔ (∃ y : ℝ, y = (x - 3) / (x - 2)) := 
by
  sorry

end meaningful_iff_x_ne_2_l2286_228672


namespace other_factor_of_LCM_l2286_228640

-- Definitions and conditions
def A : ℕ := 624
def H : ℕ := 52 
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Hypotheses based on the problem statement
axiom h_hcf : HCF A 52 = 52

-- The desired statement to prove
theorem other_factor_of_LCM (B : ℕ) (y : ℕ) : HCF A B = H → (A * y = 624) → y = 1 := 
by 
  intro h1 h2
  -- Actual proof steps are omitted
  sorry

end other_factor_of_LCM_l2286_228640


namespace drone_height_l2286_228633

theorem drone_height (r s h : ℝ) 
  (h_distance_RS : r^2 + s^2 = 160^2)
  (h_DR : h^2 + r^2 = 170^2) 
  (h_DS : h^2 + s^2 = 150^2) : 
  h = 30 * Real.sqrt 43 :=
by 
  sorry

end drone_height_l2286_228633


namespace domain_of_sqrt_cos_function_l2286_228695

theorem domain_of_sqrt_cos_function:
  (∀ k : ℤ, ∀ x : ℝ, 2 * Real.cos x + 1 ≥ 0 ↔ x ∈ Set.Icc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + 2 * Real.pi / 3)) :=
by
  sorry

end domain_of_sqrt_cos_function_l2286_228695


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l2286_228689

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l2286_228689
