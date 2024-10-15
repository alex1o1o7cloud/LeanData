import Mathlib

namespace NUMINAMATH_GPT_union_of_A_and_B_l372_37251

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l372_37251


namespace NUMINAMATH_GPT_impossible_odd_n_m_l372_37210

theorem impossible_odd_n_m (n m : ℤ) (h : Even (n^2 + m + n * m)) : ¬ (Odd n ∧ Odd m) :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_impossible_odd_n_m_l372_37210


namespace NUMINAMATH_GPT_area_to_be_painted_l372_37240

variable (h_wall : ℕ) (l_wall : ℕ)
variable (h_window : ℕ) (l_window : ℕ)
variable (h_door : ℕ) (l_door : ℕ)

theorem area_to_be_painted :
  ∀ (h_wall : ℕ) (l_wall : ℕ) (h_window : ℕ) (l_window : ℕ) (h_door : ℕ) (l_door : ℕ),
  h_wall = 10 → l_wall = 15 →
  h_window = 3 → l_window = 5 →
  h_door = 2 → l_door = 3 →
  (h_wall * l_wall) - ((h_window * l_window) + (h_door * l_door)) = 129 :=
by
  intros
  sorry

end NUMINAMATH_GPT_area_to_be_painted_l372_37240


namespace NUMINAMATH_GPT_geometric_series_arithmetic_sequence_l372_37224

noncomputable def geometric_seq_ratio (a : ℕ → ℝ) (q : ℝ) : Prop := 
∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_series_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_seq_ratio a q)
  (h_pos : ∀ n, a n > 0)
  (h_arith : a 1 = (a 0 + 2 * a 1) / 2) :
  a 5 / a 3 = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_geometric_series_arithmetic_sequence_l372_37224


namespace NUMINAMATH_GPT_most_likely_outcomes_l372_37273

noncomputable def probability_boy_or_girl : ℚ := 1 / 2

noncomputable def probability_all_boys (n : ℕ) : ℚ := probability_boy_or_girl^n

noncomputable def probability_all_girls (n : ℕ) : ℚ := probability_boy_or_girl^n

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_3_girls_2_boys : ℚ := binom 5 3 * probability_boy_or_girl^5

noncomputable def probability_3_boys_2_girls : ℚ := binom 5 2 * probability_boy_or_girl^5

theorem most_likely_outcomes :
  probability_3_girls_2_boys = 5/16 ∧
  probability_3_boys_2_girls = 5/16 ∧
  probability_all_boys 5 = 1/32 ∧
  probability_all_girls 5 = 1/32 ∧
  (5/16 > 1/32) :=
by
  sorry

end NUMINAMATH_GPT_most_likely_outcomes_l372_37273


namespace NUMINAMATH_GPT_mark_initial_money_l372_37278

theorem mark_initial_money (X : ℝ) 
  (h1 : X = (1/2) * X + 14 + (1/3) * X + 16) : X = 180 := 
  by
  sorry

end NUMINAMATH_GPT_mark_initial_money_l372_37278


namespace NUMINAMATH_GPT_solve_for_f_2012_l372_37216

noncomputable def f : ℝ → ℝ := sorry -- as the exact function definition isn't provided

variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = -f x)
variable (functional_eqn : ∀ x, f (x + 2) = f x + f 2)
variable (f_one : f 1 = 2)

theorem solve_for_f_2012 : f 2012 = 4024 :=
sorry

end NUMINAMATH_GPT_solve_for_f_2012_l372_37216


namespace NUMINAMATH_GPT_mary_marbles_l372_37227

theorem mary_marbles (total_marbles joan_marbles mary_marbles : ℕ) 
  (h1 : total_marbles = 12) 
  (h2 : joan_marbles = 3) 
  (h3 : total_marbles = joan_marbles + mary_marbles) : 
  mary_marbles = 9 := 
by
  rw [h1, h2, add_comm] at h3
  linarith

end NUMINAMATH_GPT_mary_marbles_l372_37227


namespace NUMINAMATH_GPT_product_of_0_25_and_0_75_is_0_1875_l372_37244

noncomputable def product_of_decimals : ℝ := 0.25 * 0.75

theorem product_of_0_25_and_0_75_is_0_1875 :
  product_of_decimals = 0.1875 :=
by
  sorry

end NUMINAMATH_GPT_product_of_0_25_and_0_75_is_0_1875_l372_37244


namespace NUMINAMATH_GPT_find_xyz_l372_37258

theorem find_xyz : ∃ (x y z : ℕ), x + y + z = 12 ∧ 7 * x + 5 * y + 8 * z = 79 ∧ x = 5 ∧ y = 4 ∧ z = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_xyz_l372_37258


namespace NUMINAMATH_GPT_complex_subtraction_l372_37299

def z1 : ℂ := 3 + (1 : ℂ)
def z2 : ℂ := 2 - (1 : ℂ)

theorem complex_subtraction : z1 - z2 = 1 + 2 * (1 : ℂ) :=
by
  sorry

end NUMINAMATH_GPT_complex_subtraction_l372_37299


namespace NUMINAMATH_GPT_tour_group_size_l372_37282

def adult_price : ℕ := 8
def child_price : ℕ := 3
def total_spent : ℕ := 44

theorem tour_group_size :
  ∃ (x y : ℕ), adult_price * x + child_price * y = total_spent ∧ (x + y = 8 ∨ x + y = 13) :=
by
  sorry

end NUMINAMATH_GPT_tour_group_size_l372_37282


namespace NUMINAMATH_GPT_water_consumption_l372_37232

theorem water_consumption (x y : ℝ)
  (h1 : 120 + 20 * x = 3200000 * y)
  (h2 : 120 + 15 * x = 3000000 * y) :
  x = 200 ∧ y = 50 :=
by
  sorry

end NUMINAMATH_GPT_water_consumption_l372_37232


namespace NUMINAMATH_GPT_specificTriangle_perimeter_l372_37267

-- Assume a type to represent triangle sides
structure IsoscelesTriangle (a b : ℕ) : Prop :=
  (equal_sides : a = b ∨ a + b > max a b)

-- Define the condition where we have specific sides
def specificTriangle : Prop :=
  IsoscelesTriangle 5 2

-- Prove that given the specific sides, the perimeter is 12
theorem specificTriangle_perimeter : specificTriangle → 5 + 5 + 2 = 12 :=
by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_specificTriangle_perimeter_l372_37267


namespace NUMINAMATH_GPT_jason_worked_hours_on_saturday_l372_37287

def hours_jason_works (x y : ℝ) : Prop :=
  (4 * x + 6 * y = 88) ∧ (x + y = 18)

theorem jason_worked_hours_on_saturday (x y : ℝ) : hours_jason_works x y → y = 8 := 
by 
  sorry

end NUMINAMATH_GPT_jason_worked_hours_on_saturday_l372_37287


namespace NUMINAMATH_GPT_trains_cross_each_other_in_given_time_l372_37279

noncomputable def trains_crossing_time (length1 length2 speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1 := (speed1_kmph * 1000) / 3600
  let speed2 := (speed2_kmph * 1000) / 3600
  let relative_speed := speed1 + speed2
  let total_distance := length1 + length2
  total_distance / relative_speed

theorem trains_cross_each_other_in_given_time :
  trains_crossing_time 300 400 36 18 = 46.67 :=
by
  -- expected proof here
  sorry

end NUMINAMATH_GPT_trains_cross_each_other_in_given_time_l372_37279


namespace NUMINAMATH_GPT_intersection_of_sets_l372_37285

open Set

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2, 3, 4, 5}) (hB : B = {2, 4, 6}) :
  A ∩ B = {2, 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l372_37285


namespace NUMINAMATH_GPT_giselle_paint_l372_37252

theorem giselle_paint (x : ℚ) (h1 : 5/7 = x/21) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_giselle_paint_l372_37252


namespace NUMINAMATH_GPT_max_yes_answers_100_l372_37238

-- Define the maximum number of "Yes" answers that could be given in a lineup of n people
def maxYesAnswers (n : ℕ) : ℕ :=
  if n = 0 then 0 else 1 + (n - 2)

theorem max_yes_answers_100 : maxYesAnswers 100 = 99 :=
  by sorry

end NUMINAMATH_GPT_max_yes_answers_100_l372_37238


namespace NUMINAMATH_GPT_train_pass_bridge_time_l372_37241

noncomputable def trainLength : ℝ := 360
noncomputable def trainSpeedKMH : ℝ := 45
noncomputable def bridgeLength : ℝ := 160
noncomputable def totalDistance : ℝ := trainLength + bridgeLength
noncomputable def trainSpeedMS : ℝ := trainSpeedKMH * (1000 / 3600)
noncomputable def timeToPassBridge : ℝ := totalDistance / trainSpeedMS

theorem train_pass_bridge_time : timeToPassBridge = 41.6 := sorry

end NUMINAMATH_GPT_train_pass_bridge_time_l372_37241


namespace NUMINAMATH_GPT_school_club_profit_l372_37263

def price_per_bar_buy : ℚ := 5 / 6
def price_per_bar_sell : ℚ := 2 / 3
def total_bars : ℕ := 1200
def total_cost : ℚ := total_bars * price_per_bar_buy
def total_revenue : ℚ := total_bars * price_per_bar_sell
def profit : ℚ := total_revenue - total_cost

theorem school_club_profit : profit = -200 := by
  sorry

end NUMINAMATH_GPT_school_club_profit_l372_37263


namespace NUMINAMATH_GPT_same_speed_4_l372_37275

theorem same_speed_4 {x : ℝ} (hx : x ≠ -7)
  (H1 : ∀ (x : ℝ), (x^2 - 7*x - 60)/(x + 7) = x - 12) 
  (H2 : ∀ (x : ℝ), x^3 - 5*x^2 - 14*x + 104 = x - 12) :
  ∃ (speed : ℝ), speed = 4 :=
by
  sorry

end NUMINAMATH_GPT_same_speed_4_l372_37275


namespace NUMINAMATH_GPT_direction_vector_of_line_l372_37274

theorem direction_vector_of_line : 
  ∃ v : ℝ × ℝ, 
  (∀ x y : ℝ, 2 * y + x = 3 → v = (-2, -1)) :=
by
  sorry

end NUMINAMATH_GPT_direction_vector_of_line_l372_37274


namespace NUMINAMATH_GPT_horses_lcm_l372_37262

theorem horses_lcm :
  let horse_times := [2, 3, 4, 5, 6, 7, 8, 9]
  let lcm_six := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
  let time_T := lcm_six
  lcm_six = 420 ∧ (Nat.digits 10 time_T).sum = 6 := by
    let horse_times := [2, 3, 4, 5, 6, 7, 8, 9]
    let lcm_six := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
    let time_T := lcm_six
    have h1 : lcm_six = 420 := sorry
    have h2 : (Nat.digits 10 time_T).sum = 6 := sorry
    exact ⟨h1, h2⟩

end NUMINAMATH_GPT_horses_lcm_l372_37262


namespace NUMINAMATH_GPT_repeating_decimal_product_l372_37246

theorem repeating_decimal_product (x y : ℚ) (h₁ : x = 8 / 99) (h₂ : y = 1 / 3) :
    x * y = 8 / 297 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_product_l372_37246


namespace NUMINAMATH_GPT_compute_expression_l372_37264

theorem compute_expression (x : ℕ) (h : x = 3) : (x^8 + 8 * x^4 + 16) / (x^4 - 4) = 93 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_compute_expression_l372_37264


namespace NUMINAMATH_GPT_number_of_windows_davids_house_l372_37230

theorem number_of_windows_davids_house
  (windows_per_minute : ℕ → ℕ)
  (h1 : ∀ t, windows_per_minute t = (4 * t) / 10)
  (h2 : windows_per_minute 160 = w)
  : w = 64 :=
by
  sorry

end NUMINAMATH_GPT_number_of_windows_davids_house_l372_37230


namespace NUMINAMATH_GPT_list_price_correct_l372_37221

noncomputable def list_price_satisfied : Prop :=
∃ x : ℝ, 0.25 * (x - 25) + 0.05 * (x - 5) = 0.15 * (x - 15) ∧ x = 28.33

theorem list_price_correct : list_price_satisfied :=
sorry

end NUMINAMATH_GPT_list_price_correct_l372_37221


namespace NUMINAMATH_GPT_inequality_proof_l372_37235

variable (x y z : ℝ)

theorem inequality_proof
  (h : x + 2*y + 3*z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 14 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l372_37235


namespace NUMINAMATH_GPT_simplify_fraction_l372_37237

theorem simplify_fraction :
  (1 / (1 / (1 / 2) ^ 1 + 1 / (1 / 2) ^ 2 + 1 / (1 / 2) ^ 3)) = (1 / 14) :=
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l372_37237


namespace NUMINAMATH_GPT_percentage_increase_is_2_l372_37212

def alan_price := 2000
def john_price := 2040
def percentage_increase (alan_price : ℕ) (john_price : ℕ) : ℕ := (john_price - alan_price) * 100 / alan_price

theorem percentage_increase_is_2 (alan_price john_price : ℕ) (h₁ : alan_price = 2000) (h₂ : john_price = 2040) :
  percentage_increase alan_price john_price = 2 := by
  rw [h₁, h₂]
  sorry

end NUMINAMATH_GPT_percentage_increase_is_2_l372_37212


namespace NUMINAMATH_GPT_milkshake_cost_is_five_l372_37243

def initial_amount : ℝ := 132
def hamburger_cost : ℝ := 4
def num_hamburgers : ℕ := 8
def num_milkshakes : ℕ := 6
def amount_left : ℝ := 70

theorem milkshake_cost_is_five (M : ℝ) (h : initial_amount - (num_hamburgers * hamburger_cost + num_milkshakes * M) = amount_left) : 
  M = 5 :=
by
  sorry

end NUMINAMATH_GPT_milkshake_cost_is_five_l372_37243


namespace NUMINAMATH_GPT_dash_cam_mounts_max_profit_l372_37205

noncomputable def monthly_profit (x t : ℝ) : ℝ :=
  (48 + t / (2 * x)) * x - 32 * x - 3 - t

theorem dash_cam_mounts_max_profit :
  ∃ (x t : ℝ), 1 < x ∧ x < 3 ∧ x = 3 - 2 / (t + 1) ∧
  monthly_profit x t = 37.5 := by
sorry

end NUMINAMATH_GPT_dash_cam_mounts_max_profit_l372_37205


namespace NUMINAMATH_GPT_find_fraction_B_minus_1_over_A_l372_37201

variable (A B : ℝ) (a_n S_n : ℕ → ℝ)
variable (h1 : ∀ n, a_n n + S_n n = A * (n ^ 2) + B * n + 1)
variable (h2 : A ≠ 0)

theorem find_fraction_B_minus_1_over_A : (B - 1) / A = 3 := by
  sorry

end NUMINAMATH_GPT_find_fraction_B_minus_1_over_A_l372_37201


namespace NUMINAMATH_GPT_clocks_sync_again_in_lcm_days_l372_37245

-- Defining the given conditions based on the problem statement.

-- Arthur's clock gains 15 minutes per day, taking 48 days to gain 12 hours (720 minutes).
def arthur_days : ℕ := 48

-- Oleg's clock gains 12 minutes per day, taking 60 days to gain 12 hours (720 minutes).
def oleg_days : ℕ := 60

-- The problem asks to prove that the situation repeats after 240 days, which is the LCM of 48 and 60.
theorem clocks_sync_again_in_lcm_days : Nat.lcm arthur_days oleg_days = 240 := 
by 
  sorry

end NUMINAMATH_GPT_clocks_sync_again_in_lcm_days_l372_37245


namespace NUMINAMATH_GPT_james_spent_6_dollars_l372_37233

-- Define the cost of items
def cost_milk : ℕ := 3
def cost_bananas : ℕ := 2

-- Define the sales tax rate as a decimal
def sales_tax_rate : ℚ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℕ := cost_milk + cost_bananas

-- Define the sales tax amount
def sales_tax_amount : ℚ := sales_tax_rate * total_cost_before_tax

-- Define the total amount spent
def total_amount_spent : ℚ := total_cost_before_tax + sales_tax_amount

-- The proof statement
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end NUMINAMATH_GPT_james_spent_6_dollars_l372_37233


namespace NUMINAMATH_GPT_amplitude_of_f_phase_shift_of_f_vertical_shift_of_f_l372_37294

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 2) + 1

theorem amplitude_of_f : (∀ x y : ℝ, |f x - f y| ≤ 2 * |x - y|) := sorry

theorem phase_shift_of_f : (∃ φ : ℝ, φ = -Real.pi / 8) := sorry

theorem vertical_shift_of_f : (∃ v : ℝ, v = 1) := sorry

end NUMINAMATH_GPT_amplitude_of_f_phase_shift_of_f_vertical_shift_of_f_l372_37294


namespace NUMINAMATH_GPT_smallest_number_divisible_l372_37203

theorem smallest_number_divisible (n : ℕ) : (∃ n : ℕ, (n + 3) % 27 = 0 ∧ (n + 3) % 35 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0) ∧ n = 4722 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_l372_37203


namespace NUMINAMATH_GPT_slope_of_AB_is_1_l372_37276

noncomputable def circle1 := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 + 2 * p.2 - 11 = 0 }
noncomputable def circle2 := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 14 * p.1 + 12 * p.2 + 60 = 0 }
def is_on_circle1 (p : ℝ × ℝ) := p ∈ circle1
def is_on_circle2 (p : ℝ × ℝ) := p ∈ circle2

theorem slope_of_AB_is_1 :
  ∃ A B : ℝ × ℝ,
  is_on_circle1 A ∧ is_on_circle2 A ∧
  is_on_circle1 B ∧ is_on_circle2 B ∧
  (B.2 - A.2) / (B.1 - A.1) = 1 :=
sorry

end NUMINAMATH_GPT_slope_of_AB_is_1_l372_37276


namespace NUMINAMATH_GPT_dan_time_second_hour_tshirts_l372_37229

-- Definition of conditions
def t_shirts_in_first_hour (rate1 : ℕ) (time : ℕ) : ℕ := time / rate1
def total_t_shirts (hour1_ts hour2_ts : ℕ) : ℕ := hour1_ts + hour2_ts
def time_per_t_shirt_in_second_hour (time : ℕ) (hour2_ts : ℕ) : ℕ := time / hour2_ts

-- Main theorem statement (without proof)
theorem dan_time_second_hour_tshirts
  (rate1 : ℕ) (hour1_time : ℕ) (total_ts : ℕ) (hour_time : ℕ)
  (hour1_ts := t_shirts_in_first_hour rate1 hour1_time)
  (hour2_ts := total_ts - hour1_ts) :
  rate1 = 12 → 
  hour1_time = 60 → 
  total_ts = 15 → 
  hour_time = 60 →
  time_per_t_shirt_in_second_hour hour_time hour2_ts = 6 :=
by
  intros rate1_eq hour1_time_eq total_ts_eq hour_time_eq
  sorry

end NUMINAMATH_GPT_dan_time_second_hour_tshirts_l372_37229


namespace NUMINAMATH_GPT_max_U_value_l372_37291

noncomputable def maximum_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) : ℝ :=
  x + y

theorem max_U_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) :
  maximum_value x y h ≤ Real.sqrt 13 :=
  sorry

end NUMINAMATH_GPT_max_U_value_l372_37291


namespace NUMINAMATH_GPT_distance_between_vertices_l372_37271

/-
Problem statement:
Prove that the distance between the vertices of the hyperbola
\(\frac{x^2}{144} - \frac{y^2}{64} = 1\) is 24.
-/

/-- 
We define the given hyperbola equation:
\frac{x^2}{144} - \frac{y^2}{64} = 1
-/
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 64 = 1

/--
We establish that the distance between the vertices of the hyperbola is 24.
-/
theorem distance_between_vertices : 
  (∀ x y : ℝ, hyperbola x y → dist (12, 0) (-12, 0) = 24) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_vertices_l372_37271


namespace NUMINAMATH_GPT_complex_powers_l372_37225

theorem complex_powers (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^(23 : ℕ) + i^(58 : ℕ) = -1 - i :=
by sorry

end NUMINAMATH_GPT_complex_powers_l372_37225


namespace NUMINAMATH_GPT_find_percentage_l372_37215

theorem find_percentage (P : ℝ) (h1 : (P / 100) * 200 = 30 + 0.60 * 50) : P = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l372_37215


namespace NUMINAMATH_GPT_average_marks_first_class_l372_37214

theorem average_marks_first_class (A : ℝ) :
  let students_class1 := 55
  let students_class2 := 48
  let avg_class2 := 58
  let avg_all := 59.067961165048544
  let total_students := 103
  let total_marks := avg_all * total_students
  total_marks = (A * students_class1) + (avg_class2 * students_class2) 
  → A = 60 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_first_class_l372_37214


namespace NUMINAMATH_GPT_sqrt_fraction_simplified_l372_37295

theorem sqrt_fraction_simplified :
  Real.sqrt (4 / 3) = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_GPT_sqrt_fraction_simplified_l372_37295


namespace NUMINAMATH_GPT_vertical_asymptote_x_value_l372_37256

theorem vertical_asymptote_x_value (x : ℝ) : 
  4 * x - 6 = 0 ↔ x = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_vertical_asymptote_x_value_l372_37256


namespace NUMINAMATH_GPT_initial_money_jennifer_l372_37270

theorem initial_money_jennifer (M : ℝ) (h1 : (1/5) * M + (1/6) * M + (1/2) * M + 12 = M) : M = 90 :=
sorry

end NUMINAMATH_GPT_initial_money_jennifer_l372_37270


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l372_37231

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ) 
  (a1 : a 1 = 3) 
  (d : ℕ := 2) 
  (h : ∀ n, a n = a 1 + (n - 1) * d) 
  (h_25 : a n = 25) : 
  n = 12 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l372_37231


namespace NUMINAMATH_GPT_smallest_number_of_integers_l372_37269

theorem smallest_number_of_integers (a b n : ℕ) 
  (h_avg_original : 89 * n = 73 * a + 111 * b) 
  (h_group_sum : a + b = n)
  (h_ratio : 8 * a = 11 * b) : 
  n = 19 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_integers_l372_37269


namespace NUMINAMATH_GPT_min_value_3x_4y_l372_37281

theorem min_value_3x_4y
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  3 * x + 4 * y = 21 :=
sorry

end NUMINAMATH_GPT_min_value_3x_4y_l372_37281


namespace NUMINAMATH_GPT_choose_5_with_exactly_one_twin_l372_37219

theorem choose_5_with_exactly_one_twin :
  let total_players := 12
  let twins := 2
  let players_to_choose := 5
  let remaining_players_after_one_twin := total_players - twins + 1 -- 11 players to choose from
  (2 * Nat.choose remaining_players_after_one_twin (players_to_choose - 1)) = 420 := 
by
  sorry

end NUMINAMATH_GPT_choose_5_with_exactly_one_twin_l372_37219


namespace NUMINAMATH_GPT_total_roses_planted_three_days_l372_37253

-- Definitions based on conditions
def susan_roses_two_days_ago : ℕ := 10
def maria_roses_two_days_ago : ℕ := 2 * susan_roses_two_days_ago
def john_roses_two_days_ago : ℕ := susan_roses_two_days_ago + 10
def roses_two_days_ago : ℕ := susan_roses_two_days_ago + maria_roses_two_days_ago + john_roses_two_days_ago

def roses_yesterday : ℕ := roses_two_days_ago + 20
def susan_roses_yesterday : ℕ := susan_roses_two_days_ago * roses_yesterday / roses_two_days_ago
def maria_roses_yesterday : ℕ := maria_roses_two_days_ago * roses_yesterday / roses_two_days_ago
def john_roses_yesterday : ℕ := john_roses_two_days_ago * roses_yesterday / roses_two_days_ago

def roses_today : ℕ := 2 * roses_two_days_ago
def susan_roses_today : ℕ := susan_roses_two_days_ago
def maria_roses_today : ℕ := maria_roses_two_days_ago + (maria_roses_two_days_ago * 25 / 100)
def john_roses_today : ℕ := john_roses_two_days_ago - (john_roses_two_days_ago * 10 / 100)

def total_roses_planted : ℕ := 
  (susan_roses_two_days_ago + maria_roses_two_days_ago + john_roses_two_days_ago) +
  (susan_roses_yesterday + maria_roses_yesterday + john_roses_yesterday) +
  (susan_roses_today + maria_roses_today + john_roses_today)

-- The statement that needs to be proved
theorem total_roses_planted_three_days : total_roses_planted = 173 := by 
  sorry

end NUMINAMATH_GPT_total_roses_planted_three_days_l372_37253


namespace NUMINAMATH_GPT_nine_cubed_expansion_l372_37255

theorem nine_cubed_expansion : 9^3 + 3 * 9^2 + 3 * 9 + 1 = 1000 := 
by 
  sorry

end NUMINAMATH_GPT_nine_cubed_expansion_l372_37255


namespace NUMINAMATH_GPT_find_9b_l372_37298

variable (a b : ℚ)

theorem find_9b (h1 : 7 * a + 3 * b = 0) (h2 : a = b - 4) : 9 * b = 126 / 5 := 
by
  sorry

end NUMINAMATH_GPT_find_9b_l372_37298


namespace NUMINAMATH_GPT_revised_lemonade_calories_l372_37200

def lemonade (lemon_grams sugar_grams water_grams lemon_calories_per_50grams sugar_calories_per_100grams : ℕ) :=
  let lemon_cals := lemon_calories_per_50grams
  let sugar_cals := (sugar_grams / 100) * sugar_calories_per_100grams
  let water_cals := 0
  lemon_cals + sugar_cals + water_cals

def lemonade_weight (lemon_grams sugar_grams water_grams : ℕ) :=
  lemon_grams + sugar_grams + water_grams

def caloric_density (total_calories : ℕ) (total_weight : ℕ) := (total_calories : ℚ) / total_weight

def calories_in_serving (density : ℚ) (serving : ℕ) := density * serving

theorem revised_lemonade_calories :
  let lemon_calories := 32
  let sugar_calories := 579
  let total_calories := lemonade 50 150 300 lemon_calories sugar_calories
  let total_weight := lemonade_weight 50 150 300
  let density := caloric_density total_calories total_weight
  let serving_calories := calories_in_serving density 250
  serving_calories = 305.5 := sorry

end NUMINAMATH_GPT_revised_lemonade_calories_l372_37200


namespace NUMINAMATH_GPT_cos_alpha_plus_5pi_over_12_eq_neg_1_over_3_l372_37213

theorem cos_alpha_plus_5pi_over_12_eq_neg_1_over_3
  (α : ℝ)
  (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 5 * π / 12) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_plus_5pi_over_12_eq_neg_1_over_3_l372_37213


namespace NUMINAMATH_GPT_option_B_coplanar_l372_37250

-- Define the three vectors in Option B.
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (-2, -4, 6)
def c : ℝ × ℝ × ℝ := (1, 0, 5)

-- Define the coplanarity condition for vectors a, b, and c.
def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = k • a

-- Prove that the vectors in Option B are coplanar.
theorem option_B_coplanar : coplanar a b c :=
sorry

end NUMINAMATH_GPT_option_B_coplanar_l372_37250


namespace NUMINAMATH_GPT_volume_frustum_correct_l372_37292

noncomputable def volume_of_frustum : ℚ :=
  let V_original := (1 / 3 : ℚ) * (16^2) * 10
  let V_smaller := (1 / 3 : ℚ) * (8^2) * 5
  V_original - V_smaller

theorem volume_frustum_correct :
  volume_of_frustum = 2240 / 3 :=
by
  sorry

end NUMINAMATH_GPT_volume_frustum_correct_l372_37292


namespace NUMINAMATH_GPT_work_days_together_l372_37207

theorem work_days_together (p_rate q_rate : ℝ) (fraction_left : ℝ) (d : ℝ) 
  (h₁ : p_rate = 1/15) (h₂ : q_rate = 1/20) (h₃ : fraction_left = 8/15)
  (h₄ : (p_rate + q_rate) * d = 1 - fraction_left) : d = 4 :=
by
  sorry

end NUMINAMATH_GPT_work_days_together_l372_37207


namespace NUMINAMATH_GPT_original_number_is_400_l372_37277

theorem original_number_is_400 (x : ℝ) (h : 1.20 * x = 480) : x = 400 :=
sorry

end NUMINAMATH_GPT_original_number_is_400_l372_37277


namespace NUMINAMATH_GPT_total_workers_is_22_l372_37234

-- Define constants and variables based on conditions
def avg_salary_all : ℝ := 850
def avg_salary_technicians : ℝ := 1000
def avg_salary_rest : ℝ := 780
def num_technicians : ℝ := 7

-- Define the necessary proof statement
theorem total_workers_is_22
  (W : ℝ)
  (h1 : W * avg_salary_all = num_technicians * avg_salary_technicians + (W - num_technicians) * avg_salary_rest) :
  W = 22 :=
by
  sorry

end NUMINAMATH_GPT_total_workers_is_22_l372_37234


namespace NUMINAMATH_GPT_not_divisible_by_10100_l372_37268

theorem not_divisible_by_10100 (n : ℕ) : (3^n + 1) % 10100 ≠ 0 := 
by 
  sorry

end NUMINAMATH_GPT_not_divisible_by_10100_l372_37268


namespace NUMINAMATH_GPT_area_of_enclosed_region_l372_37209

theorem area_of_enclosed_region :
  ∃ (r : ℝ), (∀ (x y : ℝ), x^2 + y^2 + 6 * x - 4 * y - 5 = 0 ↔ (x + 3)^2 + (y - 2)^2 = r^2) ∧ (π * r^2 = 14 * π) := by
  sorry

end NUMINAMATH_GPT_area_of_enclosed_region_l372_37209


namespace NUMINAMATH_GPT_number_of_true_propositions_eq_2_l372_37290

theorem number_of_true_propositions_eq_2 :
  (¬(∀ (a b : ℝ), a < 0 → b > 0 → a + b < 0)) ∧
  (∀ (α β : ℝ), α = 90 → β = 90 → α = β) ∧
  (∀ (α β : ℝ), α + β = 90 → (∀ (γ : ℝ), γ + α = 90 → β = γ)) ∧
  (¬(∀ (ℓ m n : ℕ), (ℓ ≠ m ∧ ℓ ≠ n ∧ m ≠ n) → (∀ (α β : ℝ), α = β))) →
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_eq_2_l372_37290


namespace NUMINAMATH_GPT_find_offset_length_l372_37261

theorem find_offset_length 
  (diagonal_offset_7 : ℝ) 
  (area_of_quadrilateral : ℝ) 
  (diagonal_length : ℝ) 
  (result : ℝ) : 
  (diagonal_length = 10) 
  ∧ (diagonal_offset_7 = 7) 
  ∧ (area_of_quadrilateral = 50) 
  → (∃ x, x = result) :=
by
  sorry

end NUMINAMATH_GPT_find_offset_length_l372_37261


namespace NUMINAMATH_GPT_elaine_earnings_increase_l372_37223

variable (E P : ℝ)

theorem elaine_earnings_increase :
  (0.25 * (E * (1 + P / 100)) = 1.4375 * 0.20 * E) → P = 15 :=
by
  intro h
  -- Start an intermediate transformation here
  sorry

end NUMINAMATH_GPT_elaine_earnings_increase_l372_37223


namespace NUMINAMATH_GPT_stuart_initially_had_20_l372_37226

variable (B T S : ℕ) -- Initial number of marbles for Betty, Tom, and Susan
variable (S_after : ℕ) -- Number of marbles Stuart has after receiving from Betty

-- Given conditions
axiom betty_initially : B = 150
axiom tom_initially : T = 30
axiom susan_initially : S = 20

axiom betty_to_tom : (0.20 : ℚ) * B = 30
axiom betty_to_susan : (0.10 : ℚ) * B = 15
axiom betty_to_stuart : (0.40 : ℚ) * B = 60
axiom stuart_after_receiving : S_after = 80

-- Theorem to prove Stuart initially had 20 marbles
theorem stuart_initially_had_20 : ∃ S_initial : ℕ, S_after - 60 = S_initial ∧ S_initial = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_stuart_initially_had_20_l372_37226


namespace NUMINAMATH_GPT_number_of_charms_l372_37242

-- Let x be the number of charms used to make each necklace
variable (x : ℕ)

-- Each charm costs $15
variable (cost_per_charm : ℕ)
axiom cost_per_charm_is_15 : cost_per_charm = 15

-- Tim sells each necklace for $200
variable (selling_price : ℕ)
axiom selling_price_is_200 : selling_price = 200

-- Tim makes a profit of $1500 if he sells 30 necklaces
variable (total_profit : ℕ)
axiom total_profit_is_1500 : total_profit = 1500

theorem number_of_charms (h : 30 * (selling_price - cost_per_charm * x) = total_profit) : x = 10 :=
sorry

end NUMINAMATH_GPT_number_of_charms_l372_37242


namespace NUMINAMATH_GPT_B_should_be_paid_2307_69_l372_37284

noncomputable def A_work_per_day : ℚ := 1 / 15
noncomputable def B_work_per_day : ℚ := 1 / 10
noncomputable def C_work_per_day : ℚ := 1 / 20
noncomputable def combined_work_per_day : ℚ := A_work_per_day + B_work_per_day + C_work_per_day
noncomputable def total_work : ℚ := 1
noncomputable def total_wages : ℚ := 5000
noncomputable def time_taken : ℚ := total_work / combined_work_per_day
noncomputable def B_share_of_work : ℚ := B_work_per_day / combined_work_per_day
noncomputable def B_share_of_wages : ℚ := B_share_of_work * total_wages

theorem B_should_be_paid_2307_69 : B_share_of_wages = 2307.69 := by
  sorry

end NUMINAMATH_GPT_B_should_be_paid_2307_69_l372_37284


namespace NUMINAMATH_GPT_men_absent_l372_37266

/-- 
A group of men decided to do a work in 20 days, but some of them became absent. 
The rest of the group did the work in 40 days. The original number of men was 20. 
Prove that 10 men became absent. 
--/
theorem men_absent 
    (original_men : ℕ) (absent_men : ℕ) (planned_days : ℕ) (actual_days : ℕ)
    (h1 : original_men = 20) (h2 : planned_days = 20) (h3 : actual_days = 40)
    (h_work : original_men * planned_days = (original_men - absent_men) * actual_days) : 
    absent_men = 10 :=
    by 
    rw [h1, h2, h3] at h_work
    -- Proceed to manually solve the equation, but here we add sorry
    sorry

end NUMINAMATH_GPT_men_absent_l372_37266


namespace NUMINAMATH_GPT_range_of_a_l372_37297

noncomputable def f (x a b : ℝ) : ℝ := (2 * x^2 - a * x + b) * Real.log (x - 1)

theorem range_of_a (a b : ℝ) (h1 : ∀ x > 1, f x a b ≥ 0) : a ≤ 6 :=
by 
  let x := 2
  have hb_eq : b = 2 * a - 8 :=
    by sorry
  have ha_le_6 : a ≤ 6 :=
    by sorry
  exact ha_le_6

end NUMINAMATH_GPT_range_of_a_l372_37297


namespace NUMINAMATH_GPT_find_years_simple_interest_l372_37247

variable (R T : ℝ)
variable (P : ℝ := 6000)
variable (additional_interest : ℝ := 360)
variable (rate_diff : ℝ := 2)
variable (H : P * ((R + rate_diff) / 100) * T = P * (R / 100) * T + additional_interest)

theorem find_years_simple_interest (h : P = 6000) (h₁ : P * ((R + 2) / 100) * T = P * (R / 100) * T + 360) : 
T = 3 :=
sorry

end NUMINAMATH_GPT_find_years_simple_interest_l372_37247


namespace NUMINAMATH_GPT_unit_digit_8_pow_1533_l372_37248

theorem unit_digit_8_pow_1533 : (8^1533 % 10) = 8 := by
  sorry

end NUMINAMATH_GPT_unit_digit_8_pow_1533_l372_37248


namespace NUMINAMATH_GPT_value_of_a_plus_b_l372_37296

theorem value_of_a_plus_b (a b : Int) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l372_37296


namespace NUMINAMATH_GPT_min_ratio_cyl_inscribed_in_sphere_l372_37289

noncomputable def min_surface_area_to_volume_ratio (R r : ℝ) : ℝ :=
  let h := 2 * Real.sqrt (R^2 - r^2)
  let A := 2 * Real.pi * r * (h + r)
  let V := Real.pi * r^2 * h
  A / V

theorem min_ratio_cyl_inscribed_in_sphere (R : ℝ) :
  ∃ r h, h = 2 * Real.sqrt (R^2 - r^2) ∧
         min_surface_area_to_volume_ratio R r = (Real.sqrt (Real.sqrt 4 + 1))^3 / R := 
by {
  sorry
}

end NUMINAMATH_GPT_min_ratio_cyl_inscribed_in_sphere_l372_37289


namespace NUMINAMATH_GPT_simplify_expression_l372_37249

theorem simplify_expression (x : ℝ) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l372_37249


namespace NUMINAMATH_GPT_eggs_in_two_boxes_l372_37260

theorem eggs_in_two_boxes (eggs_per_box : ℕ) (number_of_boxes : ℕ) (total_eggs : ℕ) 
  (h1 : eggs_per_box = 3)
  (h2 : number_of_boxes = 2) :
  total_eggs = eggs_per_box * number_of_boxes :=
sorry

end NUMINAMATH_GPT_eggs_in_two_boxes_l372_37260


namespace NUMINAMATH_GPT_number_of_cars_lifted_l372_37217

def total_cars_lifted : ℕ := 6

theorem number_of_cars_lifted : total_cars_lifted = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_cars_lifted_l372_37217


namespace NUMINAMATH_GPT_arithmetic_mean_twice_y_l372_37259

theorem arithmetic_mean_twice_y (y x : ℝ) (h1 : (8 + y + 24 + 6 + x) / 5 = 12) (h2 : x = 2 * y) :
  y = 22 / 3 ∧ x = 44 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_twice_y_l372_37259


namespace NUMINAMATH_GPT_additional_people_needed_l372_37280

def total_days := 50
def initial_people := 40
def days_passed := 25
def work_completed := 0.40

theorem additional_people_needed : 
  ∃ additional_people : ℕ, additional_people = 8 :=
by
  -- Placeholder for the actual proof skipped with 'sorry'
  sorry

end NUMINAMATH_GPT_additional_people_needed_l372_37280


namespace NUMINAMATH_GPT_train1_speed_l372_37257

noncomputable def total_distance_in_kilometers : ℝ :=
  (630 + 100 + 200) / 1000

noncomputable def time_in_hours : ℝ :=
  13.998880089592832 / 3600

noncomputable def relative_speed : ℝ :=
  total_distance_in_kilometers / time_in_hours

noncomputable def speed_of_train2 : ℝ :=
  72

noncomputable def speed_of_train1 : ℝ :=
  relative_speed - speed_of_train2

theorem train1_speed : speed_of_train1 = 167.076 := by 
  sorry

end NUMINAMATH_GPT_train1_speed_l372_37257


namespace NUMINAMATH_GPT_train_length_is_correct_l372_37206

-- Defining the initial conditions
def train_speed_km_per_hr : Float := 90.0
def time_seconds : Float := 5.0

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : Float) : Float :=
  speed_km_per_hr * (1000.0 / 3600.0)

-- Calculate the length of the train in meters
def length_of_train (speed_km_per_hr : Float) (time_s : Float) : Float :=
  km_per_hr_to_m_per_s speed_km_per_hr * time_s

-- Theorem statement
theorem train_length_is_correct : length_of_train train_speed_km_per_hr time_seconds = 125.0 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l372_37206


namespace NUMINAMATH_GPT_fraction_of_total_money_spent_on_dinner_l372_37286

-- Definitions based on conditions
def aaron_savings : ℝ := 40
def carson_savings : ℝ := 40
def total_savings : ℝ := aaron_savings + carson_savings

def ice_cream_cost_per_scoop : ℝ := 1.5
def scoops_each : ℕ := 6
def total_ice_cream_cost : ℝ := 2 * scoops_each * ice_cream_cost_per_scoop

def total_left : ℝ := 2

def total_spent : ℝ := total_savings - total_left
def dinner_cost : ℝ := total_spent - total_ice_cream_cost

-- Target statement
theorem fraction_of_total_money_spent_on_dinner : 
  (dinner_cost = 60) ∧ (total_savings = 80) → dinner_cost / total_savings = 3 / 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_fraction_of_total_money_spent_on_dinner_l372_37286


namespace NUMINAMATH_GPT_problem_statement_l372_37222

variable {x y : ℝ}

theorem problem_statement (h1 : x * y = -3) (h2 : x + y = -4) : x^2 + 3 * x * y + y^2 = 13 := sorry

end NUMINAMATH_GPT_problem_statement_l372_37222


namespace NUMINAMATH_GPT_kangaroo_fiber_intake_l372_37228

-- Suppose kangaroos absorb only 30% of the fiber they eat
def absorption_rate : ℝ := 0.30

-- If a kangaroo absorbed 15 ounces of fiber in one day
def absorbed_fiber : ℝ := 15.0

-- Prove the kangaroo ate 50 ounces of fiber that day
theorem kangaroo_fiber_intake (x : ℝ) (hx : absorption_rate * x = absorbed_fiber) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_kangaroo_fiber_intake_l372_37228


namespace NUMINAMATH_GPT_bullet_speed_difference_l372_37265

theorem bullet_speed_difference (speed_horse speed_bullet : ℕ) 
    (h_horse : speed_horse = 20) (h_bullet : speed_bullet = 400) :
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    speed_same_direction - speed_opposite_direction = 40 :=
    by
    -- Define the speeds in terms of the given conditions.
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    -- State the equality to prove.
    show speed_same_direction - speed_opposite_direction = 40;
    -- Proof (skipped here).
    -- sorry is used to denote where the formal proof steps would go.
    sorry

end NUMINAMATH_GPT_bullet_speed_difference_l372_37265


namespace NUMINAMATH_GPT_parallelogram_side_length_l372_37254

theorem parallelogram_side_length (x y : ℚ) (h1 : 3 * x + 2 = 12) (h2 : 5 * y - 3 = 9) : x + y = 86 / 15 :=
by 
  sorry

end NUMINAMATH_GPT_parallelogram_side_length_l372_37254


namespace NUMINAMATH_GPT_solve_inequality_l372_37220

theorem solve_inequality (a x : ℝ) : 
  (a < 0 → (x ≤ 3 / a ∨ x ≥ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a = 0 → (x ≥ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (0 < a ∧ a < 3 → (1 ≤ x ∧ x ≤ 3 / a) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a = 3 → (x = 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a > 3 → (3 / a ≤ x ∧ x ≤ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) :=
  sorry

end NUMINAMATH_GPT_solve_inequality_l372_37220


namespace NUMINAMATH_GPT_second_number_in_set_l372_37283

theorem second_number_in_set (avg1 avg2 n1 n2 n3 : ℕ) (h1 : avg1 = (10 + 70 + 19) / 3) (h2 : avg2 = avg1 + 7) (h3 : n1 = 20) (h4 : n3 = 60) :
  n2 = n3 := 
  sorry

end NUMINAMATH_GPT_second_number_in_set_l372_37283


namespace NUMINAMATH_GPT_loraine_wax_usage_proof_l372_37293

-- Conditions
variables (large_animals small_animals : ℕ)
variable (wax : ℕ)

-- Definitions based on conditions
def large_animal_wax := 4
def small_animal_wax := 2
def total_sticks := 20
def small_animals_wax := 12
def small_to_large_ratio := 3

-- Proof statement
theorem loraine_wax_usage_proof (h1 : small_animals_wax = small_animals * small_animal_wax)
  (h2 : small_animals = large_animals * small_to_large_ratio)
  (h3 : wax = small_animals_wax + large_animals * large_animal_wax) :
  wax = total_sticks := by
  sorry

end NUMINAMATH_GPT_loraine_wax_usage_proof_l372_37293


namespace NUMINAMATH_GPT_columns_contain_all_numbers_l372_37202

def rearrange (n m k : ℕ) (a : ℕ → ℕ) : ℕ → ℕ :=
  λ i => if i < n - m then a (i + m + 1)
         else if i < n - k - m then a (i - (n - m) + k + 1)
         else a (i - (n - k))

theorem columns_contain_all_numbers
  (n m k: ℕ)
  (h1 : n > 0)
  (h2 : m < n)
  (h3 : k < n)
  (a : ℕ → ℕ)
  (h4 : ∀ i : ℕ, i < n → a i = i + 1) :
  ∀ j : ℕ, j < n → ∃ i : ℕ, i < n ∧ rearrange n m k a i = j + 1 :=
by
  sorry

end NUMINAMATH_GPT_columns_contain_all_numbers_l372_37202


namespace NUMINAMATH_GPT_youngest_child_age_possible_l372_37288

theorem youngest_child_age_possible 
  (total_bill : ℝ) (mother_charge : ℝ) 
  (yearly_charge_per_child : ℝ) (minimum_charge_per_child : ℝ) 
  (num_children : ℤ) (children_total_bill : ℝ)
  (total_years : ℤ)
  (youngest_possible_age : ℤ) :
  total_bill = 15.30 →
  mother_charge = 6 →
  yearly_charge_per_child = 0.60 →
  minimum_charge_per_child = 0.90 →
  num_children = 3 →
  children_total_bill = total_bill - mother_charge →
  children_total_bill - num_children * minimum_charge_per_child = total_years * yearly_charge_per_child →
  total_years = 11 →
  youngest_possible_age = 1 :=
sorry

end NUMINAMATH_GPT_youngest_child_age_possible_l372_37288


namespace NUMINAMATH_GPT_sin_150_equals_half_l372_37272

theorem sin_150_equals_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_sin_150_equals_half_l372_37272


namespace NUMINAMATH_GPT_calc_g_f_3_l372_37204

def f (x : ℕ) : ℕ := x^3 + 3

def g (x : ℕ) : ℕ := 2 * x^2 + 3 * x + 2

theorem calc_g_f_3 : g (f 3) = 1892 := by
  sorry

end NUMINAMATH_GPT_calc_g_f_3_l372_37204


namespace NUMINAMATH_GPT_find_g_at_1_l372_37211

theorem find_g_at_1 (g : ℝ → ℝ) (h : ∀ x, x ≠ 1/2 → g x + g ((2*x + 1)/(1 - 2*x)) = x) : 
  g 1 = 15 / 7 :=
sorry

end NUMINAMATH_GPT_find_g_at_1_l372_37211


namespace NUMINAMATH_GPT_class_average_weight_l372_37236

theorem class_average_weight (n_A n_B : ℕ) (w_A w_B : ℝ) (h1 : n_A = 50) (h2 : n_B = 40) (h3 : w_A = 50) (h4 : w_B = 70) :
  (n_A * w_A + n_B * w_B) / (n_A + n_B) = 58.89 :=
by
  sorry

end NUMINAMATH_GPT_class_average_weight_l372_37236


namespace NUMINAMATH_GPT_problem1_problem2_l372_37218

variable {a b : ℝ}

-- Proof problem 1
-- Goal: (1)(2a^(2/3)b^(1/2))(-6a^(1/2)b^(1/3)) / (-3a^(1/6)b^(5/6)) = -12a
theorem problem1 (h1 : 0 < a) (h2 : 0 < b) : 
  (1 : ℝ) * (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = -12 * a := 
sorry

-- Proof problem 2
-- Goal: 2(log(sqrt(2)))^2 + log(sqrt(2)) * log(5) + sqrt((log(sqrt(2)))^2 - log(2) + 1) = 1 + (1 / 2) * log(5)
theorem problem2 : 
  2 * (Real.log (Real.sqrt 2))^2 + (Real.log (Real.sqrt 2)) * (Real.log 5) + 
  Real.sqrt ((Real.log (Real.sqrt 2))^2 - Real.log 2 + 1) = 
  1 + 0.5 * (Real.log 5) := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l372_37218


namespace NUMINAMATH_GPT_max_value_of_8q_minus_9p_is_zero_l372_37208

theorem max_value_of_8q_minus_9p_is_zero (p : ℝ) (q : ℝ) (h1 : 0 < p) (h2 : p < 1) (hq : q = 3 * p ^ 2 - 2 * p ^ 3) : 
  8 * q - 9 * p ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_8q_minus_9p_is_zero_l372_37208


namespace NUMINAMATH_GPT_inequality_proof_l372_37239

noncomputable def sum_expression (a b c : ℝ) : ℝ :=
  (1 / (b * c + a + 1 / a)) + (1 / (c * a + b + 1 / b)) + (1 / (a * b + c + 1 / c))

theorem inequality_proof (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  sum_expression a b c ≤ 27 / 31 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l372_37239
