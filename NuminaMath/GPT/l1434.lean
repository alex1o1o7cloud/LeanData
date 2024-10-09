import Mathlib

namespace fraction_value_l1434_143450

theorem fraction_value : (2 * 0.24) / (20 * 2.4) = 0.01 := by
  sorry

end fraction_value_l1434_143450


namespace probability_two_units_of_origin_l1434_143465

def square_vertices (x_min x_max y_min y_max : ℝ) :=
  { p : ℝ × ℝ // x_min ≤ p.1 ∧ p.1 ≤ x_max ∧ y_min ≤ p.2 ∧ p.2 ≤ y_max }

def within_radius (r : ℝ) (origin : ℝ × ℝ) (p : ℝ × ℝ) :=
  (p.1 - origin.1)^2 + (p.2 - origin.2)^2 ≤ r^2

noncomputable def probability_within_radius (x_min x_max y_min y_max r : ℝ) : ℝ :=
  let square_area := (x_max - x_min) * (y_max - y_min)
  let circle_area := r^2 * Real.pi
  circle_area / square_area

theorem probability_two_units_of_origin :
  probability_within_radius (-3) 3 (-3) 3 2 = Real.pi / 9 :=
by
  sorry

end probability_two_units_of_origin_l1434_143465


namespace long_sleeve_shirts_l1434_143444

variable (short_sleeve long_sleeve : Nat)
variable (total_shirts washed_shirts : Nat)
variable (not_washed_shirts : Nat)

-- Given conditions
axiom h1 : short_sleeve = 9
axiom h2 : total_shirts = 29
axiom h3 : not_washed_shirts = 1
axiom h4 : washed_shirts = total_shirts - not_washed_shirts

-- The question to be proved
theorem long_sleeve_shirts : long_sleeve = washed_shirts - short_sleeve := by
  sorry

end long_sleeve_shirts_l1434_143444


namespace maximize_profit_at_14_yuan_and_720_l1434_143458

def initial_cost : ℝ := 8
def initial_price : ℝ := 10
def initial_units_sold : ℝ := 200
def decrease_units_per_half_yuan_increase : ℝ := 10
def increase_price_per_step : ℝ := 0.5

noncomputable def profit (x : ℝ) : ℝ := 
  let selling_price := initial_price + increase_price_per_step * x
  let units_sold := initial_units_sold - decrease_units_per_half_yuan_increase * x
  (selling_price - initial_cost) * units_sold

theorem maximize_profit_at_14_yuan_and_720 :
  profit 8 = 720 ∧ (initial_price + increase_price_per_step * 8 = 14) :=
by
  sorry

end maximize_profit_at_14_yuan_and_720_l1434_143458


namespace solve_system_of_equations_l1434_143466

theorem solve_system_of_equations : ∃ (x y : ℝ), (2 * x - y = 3) ∧ (3 * x + 2 * y = 8) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end solve_system_of_equations_l1434_143466


namespace point_in_fourth_quadrant_l1434_143489

def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant :
  in_fourth_quadrant (1, -2) ∧
  ¬ in_fourth_quadrant (2, 1) ∧
  ¬ in_fourth_quadrant (-2, 1) ∧
  ¬ in_fourth_quadrant (-1, -3) :=
by
  sorry

end point_in_fourth_quadrant_l1434_143489


namespace inverse_function_less_than_zero_l1434_143478

theorem inverse_function_less_than_zero (x : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = 2^x + 1) (h₂ : ∀ y, f (f⁻¹ y) = y) (h₃ : ∀ y, f⁻¹ (f y) = y) :
  {x | f⁻¹ x < 0} = {x | 1 < x ∧ x < 2} :=
by
  sorry

end inverse_function_less_than_zero_l1434_143478


namespace slope_angle_AB_l1434_143439

noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (1, 0)

theorem slope_angle_AB :
  let θ := Real.arctan (↑(B.2 - A.2) / ↑(B.1 - A.1))
  θ = 3 * Real.pi / 4 := 
by
  -- Proof goes here
  sorry

end slope_angle_AB_l1434_143439


namespace fraction_of_two_bedroom_l1434_143457

theorem fraction_of_two_bedroom {x : ℝ} 
    (h1 : 0.17 + x = 0.5) : x = 0.33 :=
by
  sorry

end fraction_of_two_bedroom_l1434_143457


namespace find_symmetric_curve_equation_l1434_143414

def equation_of_curve_symmetric_to_line : Prop :=
  ∀ (x y : ℝ), (5 * x^2 + 12 * x * y - 22 * x - 12 * y - 19 = 0 ∧ x - y + 2 = 0) →
  12 * x * y + 5 * y^2 - 78 * y + 45 = 0

theorem find_symmetric_curve_equation : equation_of_curve_symmetric_to_line :=
sorry

end find_symmetric_curve_equation_l1434_143414


namespace discount_percentage_l1434_143415

theorem discount_percentage 
  (C : ℝ) (S : ℝ) (P : ℝ) (SP : ℝ)
  (h1 : C = 48)
  (h2 : 0.60 * S = C)
  (h3 : P = 16)
  (h4 : P = S - SP)
  (h5 : SP = 80 - 16)
  (h6 : S = 80) :
  (S - SP) / S * 100 = 20 := by
sorry

end discount_percentage_l1434_143415


namespace prob_at_least_3_correct_l1434_143405

-- Define the probability of one patient being cured
def prob_cured : ℝ := 0.9

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the probability of exactly 3 out of 4 patients being cured
def prob_exactly_3 : ℝ :=
  binomial 4 3 * prob_cured^3 * (1 - prob_cured)

-- Define the probability of all 4 patients being cured
def prob_all_4 : ℝ :=
  prob_cured^4

-- Define the probability of at least 3 out of 4 patients being cured
def prob_at_least_3 : ℝ :=
  prob_exactly_3 + prob_all_4

-- The theorem to prove
theorem prob_at_least_3_correct : prob_at_least_3 = 0.9477 :=
  by
  sorry

end prob_at_least_3_correct_l1434_143405


namespace largest_number_l1434_143468

theorem largest_number (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ = 0.9791) 
  (h₂ : x₂ = 0.97019)
  (h₃ : x₃ = 0.97909)
  (h₄ : x₄ = 0.971)
  (h₅ : x₅ = 0.97109)
  : max x₁ (max x₂ (max x₃ (max x₄ x₅))) = 0.9791 :=
  sorry

end largest_number_l1434_143468


namespace ratio_of_line_cutting_median_lines_l1434_143492

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem ratio_of_line_cutting_median_lines (A B C P Q : ℝ × ℝ) 
    (hA : A = (1, 0)) (hB : B = (0, 1)) (hC : C = (0, 0)) 
    (h_mid_AB : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) 
    (h_mid_BC : Q = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) 
    (h_ratio : (Real.sqrt (P.1^2 + P.2^2) / Real.sqrt (Q.1^2 + Q.2^2)) = (Real.sqrt (Q.1^2 + Q.2^2) / Real.sqrt (P.1^2 + P.2^2))) :
  (P.1 / Q.1) = golden_ratio :=
by 
  sorry

end ratio_of_line_cutting_median_lines_l1434_143492


namespace amount_on_table_A_l1434_143441

-- Definitions based on conditions
variables (A B C : ℝ)
variables (h1 : B = 2 * C)
variables (h2 : C = A + 20)
variables (h3 : A + B + C = 220)

-- Theorem statement
theorem amount_on_table_A : A = 40 :=
by
  -- This is expected to be filled in with the proof steps, but we skip it with 'sorry'
  sorry

end amount_on_table_A_l1434_143441


namespace seven_thirteenths_of_3940_percent_25000_l1434_143461

noncomputable def seven_thirteenths (x : ℝ) : ℝ := (7 / 13) * x

noncomputable def percent (part whole : ℝ) : ℝ := (part / whole) * 100

theorem seven_thirteenths_of_3940_percent_25000 :
  percent (seven_thirteenths 3940) 25000 = 8.484 :=
by
  sorry

end seven_thirteenths_of_3940_percent_25000_l1434_143461


namespace reciprocal_of_neg_2023_l1434_143481

theorem reciprocal_of_neg_2023 : (1 : ℝ) / (-2023) = -(1 / 2023) :=
by 
  sorry

end reciprocal_of_neg_2023_l1434_143481


namespace number_of_pupils_in_class_l1434_143499

theorem number_of_pupils_in_class
(U V : ℕ) (increase : ℕ) (avg_increase : ℕ) (n : ℕ) 
(h1 : U = 85) (h2 : V = 45) (h3 : increase = U - V) (h4 : avg_increase = 1 / 2) (h5 : increase / avg_increase = n) :
n = 80 := by
sorry

end number_of_pupils_in_class_l1434_143499


namespace hashN_of_25_l1434_143443

def hashN (N : ℝ) : ℝ := 0.6 * N + 2

theorem hashN_of_25 : hashN (hashN (hashN (hashN 25))) = 7.592 :=
by
  sorry

end hashN_of_25_l1434_143443


namespace averageSpeed_l1434_143484

-- Define the total distance driven by Jane
def totalDistance : ℕ := 200

-- Define the total time duration from 6 a.m. to 11 a.m.
def totalTime : ℕ := 5

-- Theorem stating that the average speed is 40 miles per hour
theorem averageSpeed (h1 : totalDistance = 200) (h2 : totalTime = 5) : totalDistance / totalTime = 40 := 
by
  sorry

end averageSpeed_l1434_143484


namespace min_value_PF_PA_l1434_143477

open Classical

noncomputable section

def parabola_eq (x y : ℝ) : Prop := y^2 = 16 * x

def point_A : ℝ × ℝ := (1, 2)

def focus_F : ℝ × ℝ := (4, 0)  -- Focus of the given parabola y^2 = 16x

def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

def PF_PA (P : ℝ × ℝ) : ℝ :=
  distance P focus_F + distance P point_A

theorem min_value_PF_PA :
  ∃ P : ℝ × ℝ, parabola_eq P.1 P.2 ∧ PF_PA P = 5 :=
sorry

end min_value_PF_PA_l1434_143477


namespace monotonic_intervals_max_value_of_k_l1434_143420

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 2
noncomputable def f_prime (x a : ℝ) : ℝ := Real.exp x - a

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a < f x₂ a) ∧
  (a > 0 → ∀ x₁ x₂ : ℝ,
    x₁ < x₂ → (x₁ < Real.log a → f x₁ a > f x₂ a) ∧ (x₁ > Real.log a → f x₁ a < f x₂ a)) :=
sorry

theorem max_value_of_k (x : ℝ) (k : ℤ) (a : ℝ) (h_a : a = 1)
  (h : ∀ x > 0, (x - k) * f_prime x a + x + 1 > 0) :
  k ≤ 2 :=
sorry

end monotonic_intervals_max_value_of_k_l1434_143420


namespace f_2011_equals_1_l1434_143486

-- Define odd function property
def is_odd_function (f : ℤ → ℤ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Define function with period property
def has_period_3 (f : ℤ → ℤ) : Prop :=
  ∀ x, f (x + 3) = f (x)

-- Define main problem statement
theorem f_2011_equals_1 
  (f : ℤ → ℤ)
  (h1 : is_odd_function f)
  (h2 : has_period_3 f)
  (h3 : f (-1) = -1) 
  : f 2011 = 1 :=
sorry

end f_2011_equals_1_l1434_143486


namespace geometry_biology_overlap_diff_l1434_143482

theorem geometry_biology_overlap_diff :
  ∀ (total_students geometry_students biology_students : ℕ),
  total_students = 232 →
  geometry_students = 144 →
  biology_students = 119 →
  (max geometry_students biology_students - max 0 (geometry_students + biology_students - total_students)) = 88 :=
by
  intros total_students geometry_students biology_students
  sorry

end geometry_biology_overlap_diff_l1434_143482


namespace omega_range_for_monotonically_decreasing_l1434_143495

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem omega_range_for_monotonically_decreasing
  (ω : ℝ)
  (hω : ω > 0)
  (h_decreasing : ∀ x ∈ Set.Ioo (Real.pi / 2) Real.pi, f ω x < f ω (x + 1e-6)) :
  1/2 ≤ ω ∧ ω ≤ 5/4 :=
by
  sorry

end omega_range_for_monotonically_decreasing_l1434_143495


namespace jack_flyers_count_l1434_143448

-- Definitions based on the given conditions
def total_flyers : ℕ := 1236
def rose_flyers : ℕ := 320
def flyers_left : ℕ := 796

-- Statement to prove
theorem jack_flyers_count : total_flyers - (rose_flyers + flyers_left) = 120 := by
  sorry

end jack_flyers_count_l1434_143448


namespace cesaro_sum_51_term_sequence_l1434_143496

noncomputable def cesaro_sum (B : List ℝ) : ℝ :=
  let T := List.scanl (· + ·) 0 B
  T.drop 1 |>.sum / B.length

theorem cesaro_sum_51_term_sequence (B : List ℝ) (h_length : B.length = 49)
  (h_cesaro_sum_49 : cesaro_sum B = 500) :
  cesaro_sum (B ++ [0, 0]) = 1441.18 :=
by
  sorry

end cesaro_sum_51_term_sequence_l1434_143496


namespace determine_x_l1434_143430

theorem determine_x (A B C : ℝ) (x : ℝ) (h1 : C > B) (h2 : B > A) (h3 : A > 0)
  (h4 : A = B - (x / 100) * B) (h5 : C = A + 2 * B) :
  x = 100 * ((B - A) / B) :=
sorry

end determine_x_l1434_143430


namespace simplify_expression_l1434_143411

theorem simplify_expression (a b c : ℝ) (ha : a = 7.4) (hb : b = 5 / 37) :
  1.6 * ((1 / a + 1 / b - 2 * c / (a * b)) * (a + b + 2 * c)) / 
  ((1 / a^2 + 1 / b^2 + 2 / (a * b) - 4 * c^2 / (a^2 * b^2))) = 1.6 :=
by 
  rw [ha, hb] 
  sorry

end simplify_expression_l1434_143411


namespace tank_fill_rate_l1434_143480

theorem tank_fill_rate
  (length width depth : ℝ)
  (time_to_fill : ℝ)
  (h_length : length = 10)
  (h_width : width = 6)
  (h_depth : depth = 5)
  (h_time : time_to_fill = 60) : 
  (length * width * depth) / time_to_fill = 5 :=
by
  -- Proof would go here
  sorry

end tank_fill_rate_l1434_143480


namespace calories_in_250g_mixed_drink_l1434_143422

def calories_in_mixed_drink (grams_cranberry : ℕ) (grams_honey : ℕ) (grams_water : ℕ)
  (calories_per_100g_cranberry : ℕ) (calories_per_100g_honey : ℕ) (calories_per_100g_water : ℕ)
  (total_grams : ℕ) (portion_grams : ℕ) : ℚ :=
  ((grams_cranberry * calories_per_100g_cranberry + grams_honey * calories_per_100g_honey + grams_water * calories_per_100g_water) : ℚ)
  / (total_grams * portion_grams)

theorem calories_in_250g_mixed_drink :
  calories_in_mixed_drink 150 50 300 30 304 0 100 250 = 98.5 := by
  -- The proof will involve arithmetic operations
  sorry

end calories_in_250g_mixed_drink_l1434_143422


namespace number_of_bars_in_box_l1434_143460

variable (x : ℕ)
variable (cost_per_bar : ℕ := 6)
variable (remaining_bars : ℕ := 6)
variable (total_money_made : ℕ := 42)

theorem number_of_bars_in_box :
  cost_per_bar * (x - remaining_bars) = total_money_made → x = 13 :=
by
  intro h
  sorry

end number_of_bars_in_box_l1434_143460


namespace tree_placement_impossible_l1434_143497

theorem tree_placement_impossible
  (length width : ℝ) (h_length : length = 4) (h_width : width = 1) :
  ¬ (∃ (t1 t2 t3 : ℝ × ℝ), 
       dist t1 t2 ≥ 2.5 ∧ 
       dist t2 t3 ≥ 2.5 ∧ 
       dist t1 t3 ≥ 2.5 ∧ 
       t1.1 ≥ 0 ∧ t1.1 ≤ length ∧ t1.2 ≥ 0 ∧ t1.2 ≤ width ∧ 
       t2.1 ≥ 0 ∧ t2.1 ≤ length ∧ t2.2 ≥ 0 ∧ t2.2 ≤ width ∧ 
       t3.1 ≥ 0 ∧ t3.1 ≤ length ∧ t3.2 ≥ 0 ∧ t3.2 ≤ width) := 
by {
  sorry
}

end tree_placement_impossible_l1434_143497


namespace unique_isolating_line_a_eq_2e_l1434_143410

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

theorem unique_isolating_line_a_eq_2e (a : ℝ) (h : a > 0) :
  (∃ k b, ∀ x : ℝ, f x ≥ k * x + b ∧ k * x + b ≥ g a x) → a = 2 * Real.exp 1 :=
sorry

end unique_isolating_line_a_eq_2e_l1434_143410


namespace john_rental_weeks_l1434_143434

noncomputable def camera_value : ℝ := 5000
noncomputable def rental_fee_rate : ℝ := 0.10
noncomputable def friend_payment_rate : ℝ := 0.40
noncomputable def john_total_payment : ℝ := 1200

theorem john_rental_weeks :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let friend_payment := weekly_rental_fee * friend_payment_rate
  let john_weekly_payment := weekly_rental_fee - friend_payment
  let rental_weeks := john_total_payment / john_weekly_payment
  rental_weeks = 4 :=
by
  -- Place for proof steps
  sorry

end john_rental_weeks_l1434_143434


namespace roots_quadratic_eq_l1434_143406

theorem roots_quadratic_eq (a b : ℝ) (h1 : a^2 + 3*a - 4 = 0) (h2 : b^2 + 3*b - 4 = 0) (h3 : a + b = -3) : a^2 + 4*a + b - 3 = -2 :=
by
  sorry

end roots_quadratic_eq_l1434_143406


namespace number_of_distinct_triangle_areas_l1434_143464

noncomputable def distinct_triangle_area_counts : ℕ :=
sorry  -- Placeholder for the proof to derive the correct answer

theorem number_of_distinct_triangle_areas
  (G H I J K L : ℝ × ℝ)
  (h₁ : G.2 = H.2)
  (h₂ : G.2 = I.2)
  (h₃ : G.2 = J.2)
  (h₄ : H.2 = I.2)
  (h₅ : H.2 = J.2)
  (h₆ : I.2 = J.2)
  (h₇ : dist G H = 2)
  (h₈ : dist H I = 2)
  (h₉ : dist I J = 2)
  (h₁₀ : K.2 = L.2 - 2)  -- Assuming constant perpendicular distance between parallel lines
  (h₁₁ : dist K L = 2) : 
  distinct_triangle_area_counts = 3 :=
sorry  -- Placeholder for the proof

end number_of_distinct_triangle_areas_l1434_143464


namespace value_of_last_installment_l1434_143459

noncomputable def total_amount_paid_without_processing_fee : ℝ :=
  36 * 2300

noncomputable def total_interest_paid : ℝ :=
  total_amount_paid_without_processing_fee - 35000

noncomputable def last_installment_value : ℝ :=
  2300 + 1000

theorem value_of_last_installment :
  last_installment_value = 3300 :=
  by
    sorry

end value_of_last_installment_l1434_143459


namespace average_speed_l1434_143404

-- Define the speeds in the first and second hours
def speed_first_hour : ℝ := 90
def speed_second_hour : ℝ := 42

-- Define the time taken for each hour
def time_first_hour : ℝ := 1
def time_second_hour : ℝ := 1

-- Calculate the total distance and total time
def total_distance : ℝ := speed_first_hour + speed_second_hour
def total_time : ℝ := time_first_hour + time_second_hour

-- State the theorem for the average speed
theorem average_speed : total_distance / total_time = 66 := by
  sorry

end average_speed_l1434_143404


namespace sum_of_angles_around_point_l1434_143453

theorem sum_of_angles_around_point (x : ℝ) (h : 6 * x + 3 * x + 4 * x + x + 2 * x = 360) : x = 22.5 :=
by
  sorry

end sum_of_angles_around_point_l1434_143453


namespace triangle_largest_angle_l1434_143456

theorem triangle_largest_angle (x : ℝ) (AB : ℝ) (AC : ℝ) (BC : ℝ) (h1 : AB = x + 5) 
                               (h2 : AC = 2 * x + 3) (h3 : BC = x + 10)
                               (h_angle_A_largest : BC > AB ∧ BC > AC)
                               (triangle_inequality_1 : AB + AC > BC)
                               (triangle_inequality_2 : AB + BC > AC)
                               (triangle_inequality_3 : AC + BC > AB) :
  1 < x ∧ x < 7 ∧ 6 = 6 := 
by {
  sorry
}

end triangle_largest_angle_l1434_143456


namespace range_of_a_l1434_143427

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = a * Real.log x + 1/2 * x^2)
  (h_ineq : ∀ x1 x2 : ℝ, x1 ≠ x2 → 0 < x1 → 0 < x2 → (f x1 - f x2) / (x1 - x2) > 4) : a > 4 :=
sorry

end range_of_a_l1434_143427


namespace find_n_value_l1434_143451

theorem find_n_value :
  ∃ m n : ℝ, (4 * x^2 + 8 * x - 448 = 0 → (x + m)^2 = n) ∧ n = 113 :=
by
  sorry

end find_n_value_l1434_143451


namespace sum_of_areas_squares_l1434_143401

theorem sum_of_areas_squares (a : ℕ) (h1 : (a + 4)^2 - a^2 = 80) : a^2 + (a + 4)^2 = 208 := by
  sorry

end sum_of_areas_squares_l1434_143401


namespace tables_count_is_correct_l1434_143437

-- Definitions based on conditions
def invited_people : ℕ := 18
def people_didnt_show_up : ℕ := 12
def people_per_table : ℕ := 3

-- Calculation based on definitions
def people_attended : ℕ := invited_people - people_didnt_show_up
def tables_needed : ℕ := people_attended / people_per_table

-- The main theorem statement
theorem tables_count_is_correct : tables_needed = 2 := by
  unfold tables_needed
  unfold people_attended
  unfold invited_people
  unfold people_didnt_show_up
  unfold people_per_table
  sorry

end tables_count_is_correct_l1434_143437


namespace gum_pieces_in_each_packet_l1434_143423

theorem gum_pieces_in_each_packet
  (packets : ℕ) (chewed_pieces : ℕ) (remaining_pieces : ℕ) (total_pieces : ℕ)
  (h1 : packets = 8) (h2 : chewed_pieces = 54) (h3 : remaining_pieces = 2) (h4 : total_pieces = chewed_pieces + remaining_pieces)
  (h5 : total_pieces = packets * (total_pieces / packets)) :
  total_pieces / packets = 7 :=
by
  sorry

end gum_pieces_in_each_packet_l1434_143423


namespace contractor_engagement_days_l1434_143402

theorem contractor_engagement_days 
  (days_worked : ℕ) 
  (total_days_absent : ℕ) 
  (work_payment : ℕ → ℤ)
  (absent_fine : ℕ → ℤ)
  (total_payment : ℤ) 
  (total_days : ℕ) 
  (h1 : work_payment days_worked = 25 * days_worked)
  (h2 : absent_fine total_days_absent = 750)
  (h3 : total_payment = (work_payment days_worked) - (absent_fine total_days_absent))
  (h4 : total_payment = 425)
  (h5 : total_days_absent = 10) 
  (h6 : sorry) : -- This assumes the result of x = 20 proving work days 
  total_days = days_worked + total_days_absent := 
  by
    sorry

end contractor_engagement_days_l1434_143402


namespace jack_marathon_time_l1434_143471

noncomputable def marathon_distance : ℝ := 42
noncomputable def jill_time : ℝ := 4.2
noncomputable def speed_ratio : ℝ := 0.7636363636363637

noncomputable def jill_speed : ℝ := marathon_distance / jill_time
noncomputable def jack_speed : ℝ := speed_ratio * jill_speed
noncomputable def jack_time : ℝ := marathon_distance / jack_speed

theorem jack_marathon_time : jack_time = 5.5 := sorry

end jack_marathon_time_l1434_143471


namespace orange_beads_in_necklace_l1434_143440

theorem orange_beads_in_necklace (O : ℕ) : 
    (∀ g w o : ℕ, g = 9 ∧ w = 6 ∧ ∃ t : ℕ, t = 45 ∧ 5 * (g + w + O) = 5 * (9 + 6 + O) ∧ 
    ∃ n : ℕ, n = 5 ∧ n * (45) =
    n * (5 * O)) → O = 9 :=
by
  sorry

end orange_beads_in_necklace_l1434_143440


namespace number_of_rows_of_desks_is_8_l1434_143429

-- Definitions for the conditions
def first_row_desks : ℕ := 10
def desks_increment : ℕ := 2
def total_desks : ℕ := 136

-- Definition for the sum of an arithmetic series
def arithmetic_series_sum (n a1 d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- The proof problem statement
theorem number_of_rows_of_desks_is_8 :
  ∃ n : ℕ, arithmetic_series_sum n first_row_desks desks_increment = total_desks ∧ n = 8 :=
by
  sorry

end number_of_rows_of_desks_is_8_l1434_143429


namespace sum_of_coefficients_l1434_143424

theorem sum_of_coefficients (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ) (hx : (1 - 2 * x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 12 :=
sorry

end sum_of_coefficients_l1434_143424


namespace no_way_to_write_as_sum_l1434_143474

def can_be_written_as_sum (S : ℕ → ℕ) (n : ℕ) (k : ℕ) : Prop :=
  n + k - 1 + (n - 1) * (k - 1) / 2 = 528 ∧ n > 0 ∧ 2 ∣ n ∧ k > 1

theorem no_way_to_write_as_sum : 
  ∀ (S : ℕ → ℕ) (n k : ℕ), can_be_written_as_sum S n k →
    0 = 0 :=
by
  -- Problem states that there are 0 valid ways to write 528 as the sum
  -- of an increasing sequence of two or more consecutive positive integers
  sorry

end no_way_to_write_as_sum_l1434_143474


namespace m_leq_neg_one_l1434_143431

theorem m_leq_neg_one (m : ℝ) :
    (∀ x : ℝ, 2^(-x) + m > 0 → x ≤ 0) → m ≤ -1 :=
by
  sorry

end m_leq_neg_one_l1434_143431


namespace geometric_seq_increasing_l1434_143475

theorem geometric_seq_increasing (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) → 
  (a 1 > a 0) = (∃ a1, (a1 > 0 ∧ q > 1) ∨ (a1 < 0 ∧ 0 < q ∧ q < 1)) :=
sorry

end geometric_seq_increasing_l1434_143475


namespace diane_coins_in_third_piggy_bank_l1434_143436

theorem diane_coins_in_third_piggy_bank :
  ∀ n1 n2 n4 n5 n6 : ℕ, n1 = 72 → n2 = 81 → n4 = 99 → n5 = 108 → n6 = 117 → (n4 - (n4 - 9)) = 90 :=
by
  -- sorry is needed to avoid an incomplete proof, as only the statement is required.
  sorry

end diane_coins_in_third_piggy_bank_l1434_143436


namespace problem_proof_l1434_143428

def P : Set ℝ := {x | x ≤ 3}

theorem problem_proof : {-1} ⊆ P := 
sorry

end problem_proof_l1434_143428


namespace lawn_area_l1434_143432

theorem lawn_area (s l : ℕ) (hs: 5 * s = 10) (hl: 5 * l = 50) (hposts: 2 * (s + l) = 24) (hlen: l + 1 = 3 * (s + 1)) :
  s * l = 500 :=
by {
  sorry
}

end lawn_area_l1434_143432


namespace polygon_sides_l1434_143485

theorem polygon_sides (n : ℕ) 
  (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_sides_l1434_143485


namespace inequality_proof_l1434_143446

variable {α β γ : ℝ}

theorem inequality_proof (h1 : β * γ ≠ 0) (h2 : (1 - γ^2) / (β * γ) ≥ 0) :
  10 * (α^2 + β^2 + γ^2 - β * γ^2) ≥ 2 * α * β + 5 * α * γ :=
sorry

end inequality_proof_l1434_143446


namespace average_growth_rate_inequality_l1434_143435

theorem average_growth_rate_inequality (p q x : ℝ) (h₁ : (1+x)^2 = (1+p)*(1+q)) (h₂ : p ≠ q) :
  x < (p + q) / 2 :=
sorry

end average_growth_rate_inequality_l1434_143435


namespace vartan_recreation_l1434_143455

noncomputable def vartan_recreation_percent (W : ℝ) (P : ℝ) : Prop := 
  let W_this_week := 0.9 * W
  let recreation_last_week := (P / 100) * W
  let recreation_this_week := 0.3 * W_this_week
  recreation_this_week = 1.8 * recreation_last_week

theorem vartan_recreation (W : ℝ) : ∀ P : ℝ, vartan_recreation_percent W P → P = 15 := 
by
  intro P h
  unfold vartan_recreation_percent at h
  sorry

end vartan_recreation_l1434_143455


namespace find_x_l1434_143403

theorem find_x (x : ℝ) : 
  (1 + x) * 0.20 = x * 0.4 → x = 1 :=
by
  intros h
  sorry

end find_x_l1434_143403


namespace smallest_positive_integer_with_18_divisors_l1434_143493

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end smallest_positive_integer_with_18_divisors_l1434_143493


namespace problems_left_to_grade_l1434_143421

-- Definitions based on provided conditions
def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 16
def graded_worksheets : ℕ := 8

-- The statement for the required proof with the correct answer included
theorem problems_left_to_grade : 4 * (16 - 8) = 32 := by
  sorry

end problems_left_to_grade_l1434_143421


namespace inequality_proof_l1434_143426

theorem inequality_proof (x y : ℝ) : 
  -1 / 2 ≤ (x + y) * (1 - x * y) / ((1 + x ^ 2) * (1 + y ^ 2)) ∧
  (x + y) * (1 - x * y) / ((1 + x ^ 2) * (1 + y ^ 2)) ≤ 1 / 2 :=
sorry

end inequality_proof_l1434_143426


namespace arithmetic_geometric_ratio_l1434_143408

variables {a : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem arithmetic_geometric_ratio {a : ℕ → ℝ} {d : ℝ} (h1 : is_arithmetic_sequence a d)
  (h2 : a 9 ≠ a 3) (h3 : is_geometric_sequence (a 1) (a 3) (a 9)):
  (a 2 + a 4 + a 10) / (a 1 + a 3 + a 9) = 16 / 13 :=
sorry

end arithmetic_geometric_ratio_l1434_143408


namespace relationship_of_coefficients_l1434_143425

theorem relationship_of_coefficients (a b c : ℝ) (α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0) 
  (h_eq' : a * β^2 + b * β + c = 0) 
  (h_roots : β = 3 * α) :
  3 * b^2 = 16 * a * c := 
sorry

end relationship_of_coefficients_l1434_143425


namespace score_difference_l1434_143491

noncomputable def mean_score (scores pcts : List ℕ) : ℚ := 
  (List.zipWith (· * ·) scores pcts).sum / 100

def median_score (scores pcts : List ℕ) : ℚ := 75

theorem score_difference :
  let scores := [60, 75, 85, 95]
  let pcts := [20, 50, 15, 15]
  abs (median_score scores pcts - mean_score scores pcts) = 1.5 := by
  sorry

end score_difference_l1434_143491


namespace num_valid_pairs_l1434_143467

theorem num_valid_pairs : ∃ (n : ℕ), n = 8 ∧ (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 150 ∧ ((a + 1 / b) / (1 / a + b) = 17) ↔ (a = 17 * b) ∧ b ≤ 8) :=
by
  sorry

end num_valid_pairs_l1434_143467


namespace chapters_in_first_book_l1434_143412

theorem chapters_in_first_book (x : ℕ) (h1 : 2 * 15 = 30) (h2 : (x + 30) / 2 + x + 30 = 75) : x = 20 :=
sorry

end chapters_in_first_book_l1434_143412


namespace average_eq_instantaneous_velocity_at_t_eq_3_l1434_143479

theorem average_eq_instantaneous_velocity_at_t_eq_3
  (S : ℝ → ℝ) (hS : ∀ t, S t = 24 * t - 3 * t^2) :
  (1 / 6) * (S 6 - S 0) = 24 - 6 * 3 :=
by 
  sorry

end average_eq_instantaneous_velocity_at_t_eq_3_l1434_143479


namespace final_number_of_cards_l1434_143438

def initial_cards : ℕ := 26
def cards_given_to_mary : ℕ := 18
def cards_found_in_box : ℕ := 40
def cards_given_to_john : ℕ := 12
def cards_purchased_at_fleamarket : ℕ := 25

theorem final_number_of_cards :
  (initial_cards - cards_given_to_mary) + (cards_found_in_box - cards_given_to_john) + cards_purchased_at_fleamarket = 61 :=
by sorry

end final_number_of_cards_l1434_143438


namespace family_members_count_l1434_143498

variable (F : ℕ) -- Number of other family members

def annual_cost_per_person : ℕ := 4000 + 12 * 1000
def john_total_cost_for_family (F : ℕ) : ℕ := (F + 1) * annual_cost_per_person / 2

theorem family_members_count :
  john_total_cost_for_family F = 32000 → F = 3 := by
  sorry

end family_members_count_l1434_143498


namespace probability_exactly_two_sunny_days_l1434_143463

-- Define the conditions
def rain_probability : ℝ := 0.8
def sun_probability : ℝ := 1 - rain_probability
def days : ℕ := 5
def sunny_days : ℕ := 2
def rainy_days : ℕ := days - sunny_days

-- Define the combinatorial and probability calculations
def comb (n k : ℕ) : ℕ := Nat.choose n k
def probability_sunny_days : ℝ := comb days sunny_days * (sun_probability ^ sunny_days) * (rain_probability ^ rainy_days)

theorem probability_exactly_two_sunny_days : probability_sunny_days = 51 / 250 := by
  sorry

end probability_exactly_two_sunny_days_l1434_143463


namespace necessary_but_not_sufficient_condition_l1434_143473

variable {M N P : Set α}

theorem necessary_but_not_sufficient_condition (h : M ∩ P = N ∩ P) : 
  (M = N) → (M ∩ P = N ∩ P) :=
sorry

end necessary_but_not_sufficient_condition_l1434_143473


namespace cubic_polynomial_roots_l1434_143417

noncomputable def polynomial := fun x : ℝ => x^3 - 2*x - 2

theorem cubic_polynomial_roots
  (x y z : ℝ) 
  (h1: polynomial x = 0)
  (h2: polynomial y = 0)
  (h3: polynomial z = 0):
  x * (y - z)^2 + y * (z - x)^2 + z * (x - y)^2 = 0 :=
by
  -- Solution steps will be filled here to prove the theorem
  sorry

end cubic_polynomial_roots_l1434_143417


namespace middle_digit_base8_l1434_143447

theorem middle_digit_base8 (M : ℕ) (e : ℕ) (d f : Fin 8) 
  (M_base8 : M = 64 * d + 8 * e + f)
  (M_base10 : M = 100 * f + 10 * e + d) :
  e = 6 :=
by sorry

end middle_digit_base8_l1434_143447


namespace find_slope_of_chord_l1434_143409

noncomputable def slope_of_chord (x1 x2 y1 y2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

theorem find_slope_of_chord :
  (∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1 → ∃ (x1 x2 y1 y2 : ℝ),
    x1 + x2 = 8 ∧ y1 + y2 = 4 ∧ x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2 ∧ slope_of_chord x1 x2 y1 y2 = -1 / 2) := sorry

end find_slope_of_chord_l1434_143409


namespace combined_salaries_A_B_C_D_l1434_143407

-- To ensure the whole calculation is noncomputable due to ℝ
noncomputable section

-- Let's define the variables
def salary_E : ℝ := 9000
def average_salary_group : ℝ := 8400
def num_people : ℕ := 5

-- combined salary A + B + C + D represented as a definition
def combined_salaries : ℝ := (average_salary_group * num_people) - salary_E

-- We need to prove that the combined salaries equals 33000
theorem combined_salaries_A_B_C_D : combined_salaries = 33000 := by
  sorry

end combined_salaries_A_B_C_D_l1434_143407


namespace eight_faucets_fill_time_in_seconds_l1434_143487

open Nat

-- Definitions under the conditions
def four_faucets_rate (gallons : ℕ) (minutes : ℕ) : ℕ := gallons / minutes

def one_faucet_rate (four_faucets_rate : ℕ) : ℕ := four_faucets_rate / 4

def eight_faucets_rate (one_faucet_rate : ℕ) : ℕ := one_faucet_rate * 8

def time_to_fill (rate : ℕ) (gallons : ℕ) : ℕ := gallons / rate

-- Main theorem to prove 
theorem eight_faucets_fill_time_in_seconds (gallons_tub : ℕ) (four_faucets_time : ℕ) :
    let four_faucets_rate := four_faucets_rate 200 8
    let one_faucet_rate := one_faucet_rate four_faucets_rate
    let rate_eight_faucets := eight_faucets_rate one_faucet_rate
    let time_fill := time_to_fill rate_eight_faucets 50
    gallons_tub = 50 ∧ four_faucets_time = 8 ∧ rate_eight_faucets = 50 -> time_fill * 60 = 60 :=
by
    intros
    sorry

end eight_faucets_fill_time_in_seconds_l1434_143487


namespace min_candies_to_remove_l1434_143433

theorem min_candies_to_remove {n : ℕ} (h : n = 31) : (∃ k, (n - k) % 5 = 0) → k = 1 :=
by
  sorry

end min_candies_to_remove_l1434_143433


namespace simplify_expression_l1434_143488

theorem simplify_expression (a : ℝ) (h : a / 2 - 2 / a = 3) : 
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 33 :=
by
  sorry

end simplify_expression_l1434_143488


namespace integer_roots_polynomial_l1434_143418

theorem integer_roots_polynomial 
(m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  (∃ a b c : ℤ, a + b + c = 17 ∧ a * b * c = n^2 ∧ a * b + b * c + c * a = m) ↔ 
  (m, n) = (80, 10) ∨ (m, n) = (88, 12) ∨ (m, n) = (80, 8) ∨ (m, n) = (90, 12) := 
sorry

end integer_roots_polynomial_l1434_143418


namespace min_value_at_x_zero_l1434_143483

noncomputable def f (x : ℝ) := Real.sqrt (x^2 + (x + 1)^2) + Real.sqrt (x^2 + (x - 1)^2)

theorem min_value_at_x_zero : ∀ x : ℝ, f x ≥ f 0 := by
  sorry

end min_value_at_x_zero_l1434_143483


namespace fraction_from_tips_l1434_143400

-- Define the waiter's salary and the conditions given in the problem
variables (S : ℕ) -- S is natural assuming salary is a non-negative integer
def tips := (4/5 : ℚ) * S
def bonus := 2 * (1/10 : ℚ) * S
def total_income := S + tips S + bonus S

-- The theorem to be proven
theorem fraction_from_tips (S : ℕ) :
  (tips S / total_income S) = (2/5 : ℚ) :=
sorry

end fraction_from_tips_l1434_143400


namespace range_of_a_max_value_of_z_l1434_143472

variable (a b : ℝ)

-- Definition of the assumptions
def condition1 := (2 * a + b = 9)
def condition2 := (|9 - b| + |a| < 3)
def condition3 := (a > 0)
def condition4 := (b > 0)
def z := a^2 * b

-- Statement for problem (i)
theorem range_of_a (h1 : condition1 a b) (h2 : condition2 a b) : -1 < a ∧ a < 1 := sorry

-- Statement for problem (ii)
theorem max_value_of_z (h1 : condition1 a b) (h2 : condition3 a) (h3 : condition4 b) : 
  z a b = 27 := sorry

end range_of_a_max_value_of_z_l1434_143472


namespace evie_l1434_143442

variable (Evie_current_age : ℕ) 

theorem evie's_age_in_one_year
  (h : Evie_current_age + 4 = 3 * (Evie_current_age - 2)) : 
  Evie_current_age + 1 = 6 :=
by
  sorry

end evie_l1434_143442


namespace jack_weight_52_l1434_143416

theorem jack_weight_52 (Sam Jack : ℕ) (h1 : Sam + Jack = 96) (h2 : Jack = Sam + 8) : Jack = 52 := 
by
  sorry

end jack_weight_52_l1434_143416


namespace gcd_polynomial_primes_l1434_143462

theorem gcd_polynomial_primes (a : ℤ) (k : ℤ) (ha : a = 2 * 947 * k) : 
  Int.gcd (3 * a^2 + 47 * a + 101) (a + 19) = 1 :=
by
  sorry

end gcd_polynomial_primes_l1434_143462


namespace intersecting_line_circle_condition_l1434_143476

theorem intersecting_line_circle_condition {a b : ℝ} (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x / a + y / b = 1) :
  (1 / a ^ 2) + (1 / b ^ 2) ≥ 1 :=
sorry

end intersecting_line_circle_condition_l1434_143476


namespace correctly_calculated_value_l1434_143470

theorem correctly_calculated_value (x : ℕ) (h : 5 * x = 40) : 2 * x = 16 := 
by {
  sorry
}

end correctly_calculated_value_l1434_143470


namespace larger_inscribed_angle_corresponds_to_larger_chord_l1434_143469

theorem larger_inscribed_angle_corresponds_to_larger_chord
  (R : ℝ) (α β : ℝ) (hα : α < 90) (hβ : β < 90) (h : α < β)
  (BC LM : ℝ) (hBC : BC = 2 * R * Real.sin α) (hLM : LM = 2 * R * Real.sin β) :
  BC < LM :=
sorry

end larger_inscribed_angle_corresponds_to_larger_chord_l1434_143469


namespace checkered_rectangle_minimal_area_checkered_rectangle_possible_perimeters_l1434_143490

-- Define the conditions
def is_checkered_rectangle (S : ℕ) : Prop :=
  (∃ (a b : ℕ), a * b = S) ∧
  (∀ x y k l : ℕ, x * 13 + y * 1 = S) ∧
  (S % 39 = 0)

-- Define that S is minimal satisfying the conditions
def minimal_area_checkered_rectangle (S : ℕ) : Prop :=
  is_checkered_rectangle S ∧
  (∀ (S' : ℕ), S' < S → ¬ is_checkered_rectangle S')

-- Prove that S = 78 is the minimal area
theorem checkered_rectangle_minimal_area : minimal_area_checkered_rectangle 78 :=
  sorry

-- Define the condition for possible perimeters
def possible_perimeters (S : ℕ) (p : ℕ) : Prop :=
  (∀ (a b : ℕ), a * b = S → 2 * (a + b) = p)

-- Prove the possible perimeters for area 78
theorem checkered_rectangle_possible_perimeters :
  ∀ p, p = 38 ∨ p = 58 ∨ p = 82 ↔ possible_perimeters 78 p :=
  sorry

end checkered_rectangle_minimal_area_checkered_rectangle_possible_perimeters_l1434_143490


namespace extreme_points_exactly_one_zero_in_positive_interval_l1434_143445

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (1 / 3) * a * x^3

theorem extreme_points (a : ℝ) (h : a > Real.exp 1) :
  ∃ (x1 x2 x3 : ℝ), (0 < x1) ∧ (x1 < x2) ∧ (x2 < x3) ∧ (deriv (f x) = 0) := sorry

theorem exactly_one_zero_in_positive_interval (a : ℝ) (h : a > Real.exp 1) :
  ∃! x : ℝ, (0 < x) ∧ (f x a = 0) := sorry

end extreme_points_exactly_one_zero_in_positive_interval_l1434_143445


namespace number_of_participants_2005_l1434_143494

variable (participants : ℕ → ℕ)
variable (n : ℕ)

-- Conditions
def initial_participants := participants 2001 = 1000
def increase_till_2003 := ∀ n, 2001 ≤ n ∧ n ≤ 2003 → participants (n + 1) = 2 * participants n
def increase_from_2004 := ∀ n, n ≥ 2004 → participants (n + 1) = 2 * participants n + 500

-- Proof problem
theorem number_of_participants_2005 :
    initial_participants participants →
    increase_till_2003 participants →
    increase_from_2004 participants →
    participants 2005 = 17500 :=
by sorry

end number_of_participants_2005_l1434_143494


namespace prism_dimensions_l1434_143413

theorem prism_dimensions (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 45) (h3 : b * c = 60) : 
  a = 7.2 ∧ b = 9.6 ∧ c = 14.4 :=
by {
  -- Proof skipped for now
  sorry
}

end prism_dimensions_l1434_143413


namespace part1_equation_solution_part2_inequality_solution_l1434_143454

theorem part1_equation_solution (x : ℝ) (h : x / (x - 1) = (x - 1) / (2 * (x - 1))) : 
  x = -1 :=
sorry

theorem part2_inequality_solution (x : ℝ) (h₁ : 5 * x - 1 > 3 * x - 4) (h₂ : - (1 / 3) * x ≤ 2 / 3 - x) : 
  -3 / 2 < x ∧ x ≤ 1 :=
sorry

end part1_equation_solution_part2_inequality_solution_l1434_143454


namespace divides_trans_l1434_143449

theorem divides_trans (m n : ℤ) (h : n ∣ m * (n + 1)) : n ∣ m :=
by
  sorry

end divides_trans_l1434_143449


namespace quadratic_root_a_l1434_143419

theorem quadratic_root_a {a : ℝ} (h : (2 : ℝ) ∈ {x : ℝ | x^2 + 3 * x + a = 0}) : a = -10 :=
by
  sorry

end quadratic_root_a_l1434_143419


namespace alloy_mixture_l1434_143452

theorem alloy_mixture (x y : ℝ) 
  (h1 : x + y = 1000)
  (h2 : 0.25 * x + 0.50 * y = 450) : 
  x = 200 ∧ y = 800 :=
by
  -- Proof will follow here
  sorry

end alloy_mixture_l1434_143452
