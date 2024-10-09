import Mathlib

namespace percentage_politics_not_local_politics_l1805_180570

variables (total_reporters : ℝ) 
variables (reporters_cover_local_politics : ℝ) 
variables (reporters_not_cover_politics : ℝ)

theorem percentage_politics_not_local_politics :
  total_reporters = 100 → 
  reporters_cover_local_politics = 5 → 
  reporters_not_cover_politics = 92.85714285714286 → 
  (total_reporters - reporters_not_cover_politics) - reporters_cover_local_politics = 2.14285714285714 := 
by 
  intros ht hr hn
  rw [ht, hr, hn]
  norm_num


end percentage_politics_not_local_politics_l1805_180570


namespace sector_area_correct_l1805_180583

-- Define the initial conditions
def arc_length := 4 -- Length of the arc in cm
def central_angle := 2 -- Central angle in radians
def radius := arc_length / central_angle -- Radius of the circle

-- Define the formula for the area of the sector
def sector_area := (1 / 2) * radius * arc_length

-- The statement of our theorem
theorem sector_area_correct : sector_area = 4 := by
  -- Proof goes here
  sorry

end sector_area_correct_l1805_180583


namespace different_digits_probability_l1805_180581

noncomputable def number_nonidentical_probability : ℚ :=
  let total_numbers := 900
  let identical_numbers := 9
  -- The probability of identical digits.
  let identical_probability := identical_numbers / total_numbers
  -- The probability of non-identical digits.
  1 - identical_probability

theorem different_digits_probability : number_nonidentical_probability = 99 / 100 := by
  sorry

end different_digits_probability_l1805_180581


namespace frank_fencemaker_fence_length_l1805_180569

theorem frank_fencemaker_fence_length :
  ∃ (L W : ℕ), W = 40 ∧
               (L * W = 200) ∧
               (2 * L + W = 50) :=
by
  sorry

end frank_fencemaker_fence_length_l1805_180569


namespace integral_f_eq_34_l1805_180505

noncomputable def f (x : ℝ) := if x ∈ [0, 1] then (1 / Real.pi) * Real.sqrt (1 - x^2) else 2 - x

theorem integral_f_eq_34 :
  ∫ x in (0 : ℝ)..2, f x = 3 / 4 :=
by
  sorry

end integral_f_eq_34_l1805_180505


namespace find_m_l1805_180533

theorem find_m {m : ℕ} (h1 : Even (m^2 - 2 * m - 3)) (h2 : m^2 - 2 * m - 3 < 0) : m = 1 :=
sorry

end find_m_l1805_180533


namespace marathon_yards_l1805_180536

theorem marathon_yards (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (marathons_run : ℕ) 
  (total_miles : ℕ) (total_yards : ℕ) (h1 : miles_per_marathon = 26) (h2 : yards_per_marathon = 385)
  (h3 : yards_per_mile = 1760) (h4 : marathons_run = 15) (h5 : 
  total_miles = marathons_run * miles_per_marathon + (marathons_run * yards_per_marathon) / yards_per_mile) 
  (h6 : total_yards = (marathons_run * yards_per_marathon) % yards_per_mile) : 
  total_yards = 495 :=
by
  -- This will be our process to verify the transformation
  sorry

end marathon_yards_l1805_180536


namespace probability_of_X_le_1_l1805_180564

noncomputable def C (n k : ℕ) : ℚ := Nat.choose n k

noncomputable def P_X_le_1 := 
  (C 4 3 / C 6 3) + (C 4 2 * C 2 1 / C 6 3)

theorem probability_of_X_le_1 : P_X_le_1 = 4 / 5 := by
  sorry

end probability_of_X_le_1_l1805_180564


namespace three_point_three_seven_five_as_fraction_l1805_180552

theorem three_point_three_seven_five_as_fraction :
  3.375 = (27 / 8 : ℚ) :=
by sorry

end three_point_three_seven_five_as_fraction_l1805_180552


namespace value_of_m_l1805_180580

theorem value_of_m (m : ℤ) : (∃ (f : ℤ → ℤ), ∀ x : ℤ, x^2 + m * x + 16 = (f x)^2) ↔ (m = 8 ∨ m = -8) := 
by
  sorry

end value_of_m_l1805_180580


namespace soccer_team_games_l1805_180550

theorem soccer_team_games (pizzas : ℕ) (slices_per_pizza : ℕ) (average_goals_per_game : ℕ) (total_games : ℕ) 
  (h1 : pizzas = 6) 
  (h2 : slices_per_pizza = 12) 
  (h3 : average_goals_per_game = 9) 
  (h4 : total_games = (pizzas * slices_per_pizza) / average_goals_per_game) :
  total_games = 8 :=
by
  rw [h1, h2, h3] at h4
  exact h4

end soccer_team_games_l1805_180550


namespace betty_garden_total_plants_l1805_180527

theorem betty_garden_total_plants (B O : ℕ) 
  (h1 : B = 5) 
  (h2 : O = 2 + 2 * B) : 
  B + O = 17 := by
  sorry

end betty_garden_total_plants_l1805_180527


namespace product_remainder_mod_5_l1805_180590

theorem product_remainder_mod_5 :
  (1024 * 1455 * 1776 * 2018 * 2222) % 5 = 0 := 
sorry

end product_remainder_mod_5_l1805_180590


namespace range_of_m_values_l1805_180548

theorem range_of_m_values {P Q : ℝ × ℝ} (hP : P = (-1, 1)) (hQ : Q = (2, 2)) (m : ℝ) :
  -3 < m ∧ m < -2 / 3 → (∃ (l : ℝ → ℝ), ∀ x y, y = l x → x + m * y + m = 0) :=
sorry

end range_of_m_values_l1805_180548


namespace common_difference_of_arithmetic_sequence_l1805_180500

variable {a : ℕ → ℝ} (a2 a5 : ℝ)
variable (h1 : a 2 = 9) (h2 : a 5 = 33)

theorem common_difference_of_arithmetic_sequence :
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 8 := by
  sorry

end common_difference_of_arithmetic_sequence_l1805_180500


namespace percent_divisible_by_six_up_to_120_l1805_180537

theorem percent_divisible_by_six_up_to_120 : 
  let total_numbers := 120
  let divisible_by_six := total_numbers / 6
  let percentage := (divisible_by_six * 100) / total_numbers
  percentage = 50 / 3 := sorry

end percent_divisible_by_six_up_to_120_l1805_180537


namespace max_abs_x_minus_2y_plus_1_l1805_180565

theorem max_abs_x_minus_2y_plus_1 (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) :
  |x - 2 * y + 1| ≤ 5 :=
sorry

end max_abs_x_minus_2y_plus_1_l1805_180565


namespace find_a_plus_b_l1805_180560

theorem find_a_plus_b (a b : ℚ) (y : ℚ) (x : ℚ) :
  (y = a + b / x) →
  (2 = a + b / (-2 : ℚ)) →
  (3 = a + b / (-6 : ℚ)) →
  a + b = 13 / 2 :=
by
  intros h₁ h₂ h₃
  sorry

end find_a_plus_b_l1805_180560


namespace geometric_sequence_sum_l1805_180554

theorem geometric_sequence_sum
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (r : ℝ)
  (h1 : r ≠ 1)
  (h2 : ∀ n, S n = a 0 * (1 - r^(n + 1)) / (1 - r))
  (h3 : S 5 = 3)
  (h4 : S 10 = 9) :
  S 15 = 21 :=
sorry

end geometric_sequence_sum_l1805_180554


namespace rectangle_breadth_approx_1_1_l1805_180582

theorem rectangle_breadth_approx_1_1 (s b : ℝ) (h1 : 4 * s = 2 * (16 + b))
  (h2 : abs ((π * s / 2) + s - 21.99) < 0.01) : abs (b - 1.1) < 0.01 :=
sorry

end rectangle_breadth_approx_1_1_l1805_180582


namespace hexagon_height_correct_l1805_180555

-- Define the dimensions of the original rectangle
def original_rectangle_width := 16
def original_rectangle_height := 9
def original_rectangle_area := original_rectangle_width * original_rectangle_height

-- Define the dimensions of the new rectangle formed by the hexagons
def new_rectangle_width := 12
def new_rectangle_height := 12
def new_rectangle_area := new_rectangle_width * new_rectangle_height

-- Define the parameter x, which is the height of the hexagons
def hexagon_height := 6

-- Theorem stating the equivalence of the areas and the specific height x
theorem hexagon_height_correct :
  original_rectangle_area = new_rectangle_area ∧
  hexagon_height * 2 = new_rectangle_height :=
by
  sorry

end hexagon_height_correct_l1805_180555


namespace monotonic_invertible_function_l1805_180523

theorem monotonic_invertible_function (f : ℝ → ℝ) (c : ℝ) (h_mono : ∀ x y, x < y → f x < f y) (h_inv : ∀ x, f (f⁻¹ x) = x) :
  (∀ x, f x + f⁻¹ x = 2 * x) ↔ ∀ x, f x = x + c :=
sorry

end monotonic_invertible_function_l1805_180523


namespace vector_problem_l1805_180556

open Real

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  (v.1 ^ 2 + v.2 ^ 2) ^ (1 / 2)

variables (a b : ℝ × ℝ)
variables (h1 : a ≠ (0, 0)) (h2 : b ≠ (0, 0))
variables (h3 : a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0)
variables (h4 : 2 * magnitude a = magnitude b) (h5 : magnitude b =2)

theorem vector_problem : magnitude (2 * a.1 - b.1, 2 * a.2 - b.2) = 2 :=
sorry

end vector_problem_l1805_180556


namespace total_number_of_water_filled_jars_l1805_180592

theorem total_number_of_water_filled_jars : 
  ∃ (x : ℕ), 28 = x * (1/4 + 1/2 + 1) ∧ 3 * x = 48 :=
by
  sorry

end total_number_of_water_filled_jars_l1805_180592


namespace original_length_of_field_l1805_180526

theorem original_length_of_field (L W : ℕ) 
  (h1 : L * W = 144) 
  (h2 : (L + 6) * W = 198) : 
  L = 16 := 
by 
  sorry

end original_length_of_field_l1805_180526


namespace total_corn_cobs_l1805_180596

-- Definitions for the conditions
def rows_first_field : ℕ := 13
def rows_second_field : ℕ := 16
def cobs_per_row : ℕ := 4

-- Statement to prove
theorem total_corn_cobs : (rows_first_field * cobs_per_row + rows_second_field * cobs_per_row) = 116 :=
by sorry

end total_corn_cobs_l1805_180596


namespace solution_is_correct_l1805_180593

noncomputable def satisfies_inequality (x y : ℝ) : Prop := 
  x + 3 * y + 14 ≤ 0

noncomputable def satisfies_equation (x y : ℝ) : Prop := 
  x^4 + 2 * x^2 * y^2 + y^4 + 64 - 20 * x^2 - 20 * y^2 = 8 * x * y

theorem solution_is_correct : satisfies_inequality (-2) (-4) ∧ satisfies_equation (-2) (-4) :=
  by sorry

end solution_is_correct_l1805_180593


namespace hyperbola_m_range_l1805_180567

theorem hyperbola_m_range (m : ℝ) (h_eq : ∀ x y, (x^2 / m) - (y^2 / (2*m - 1)) = 1) : 
  0 < m ∧ m < 1/2 :=
sorry

end hyperbola_m_range_l1805_180567


namespace josh_points_l1805_180524

variable (x y : ℕ)
variable (three_point_success_rate two_point_success_rate : ℚ)
variable (total_shots : ℕ)
variable (points : ℚ)

theorem josh_points (h1 : three_point_success_rate = 0.25)
                    (h2 : two_point_success_rate = 0.40)
                    (h3 : total_shots = 40)
                    (h4 : x + y = total_shots) :
                    points = 32 :=
by sorry

end josh_points_l1805_180524


namespace greatest_area_difference_l1805_180514

def first_rectangle_perimeter (l w : ℕ) : Prop :=
  2 * l + 2 * w = 156

def second_rectangle_perimeter (l w : ℕ) : Prop :=
  2 * l + 2 * w = 144

theorem greatest_area_difference : 
  ∃ (l1 w1 l2 w2 : ℕ), 
  first_rectangle_perimeter l1 w1 ∧ 
  second_rectangle_perimeter l2 w2 ∧ 
  (l1 * (78 - l1) - l2 * (72 - l2) = 225) := 
sorry

end greatest_area_difference_l1805_180514


namespace semicircle_area_increase_l1805_180574

noncomputable def area_semicircle (r : ℝ) : ℝ :=
  (1 / 2) * Real.pi * r^2

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem semicircle_area_increase :
  let r_long := 6
  let r_short := 4
  let area_long := 2 * area_semicircle r_long
  let area_short := 2 * area_semicircle r_short
  percent_increase area_short area_long = 125 :=
by
  let r_long := 6
  let r_short := 4
  let area_long := 2 * area_semicircle r_long
  let area_short := 2 * area_semicircle r_short
  have : area_semicircle r_long = 18 * Real.pi := by sorry
  have : area_semicircle r_short = 8 * Real.pi := by sorry
  have : area_long = 36 * Real.pi := by sorry
  have : area_short = 16 * Real.pi := by sorry
  have : percent_increase area_short area_long = 125 := by sorry
  exact this

end semicircle_area_increase_l1805_180574


namespace domain_of_ln_function_l1805_180568

theorem domain_of_ln_function (x : ℝ) : 3 - 4 * x > 0 ↔ x < 3 / 4 := 
by
  sorry

end domain_of_ln_function_l1805_180568


namespace mean_score_l1805_180577

theorem mean_score (mu sigma : ℝ) 
  (h1 : 86 = mu - 7 * sigma) 
  (h2 : 90 = mu + 3 * sigma) :
  mu = 88.8 :=
by
  -- skipping the proof
  sorry

end mean_score_l1805_180577


namespace initial_amount_l1805_180530

def pie_cost : Real := 6.75
def juice_cost : Real := 2.50
def gift : Real := 10.00
def mary_final : Real := 52.00

theorem initial_amount (M : Real) :
  M = mary_final + pie_cost + juice_cost + gift :=
by
  sorry

end initial_amount_l1805_180530


namespace largest_integral_value_l1805_180586

theorem largest_integral_value (x : ℤ) : (1 / 3 : ℚ) < x / 5 ∧ x / 5 < 5 / 8 → x = 3 :=
by
  sorry

end largest_integral_value_l1805_180586


namespace measure_angle_PQR_given_conditions_l1805_180508

-- Definitions based on conditions
variables {R P Q S : Type} [LinearOrder R] [AddGroup Q] [LinearOrder P] [LinearOrder S]

-- Assume given conditions
def is_straight_line (r s p : ℝ) : Prop := r + p = 2 * s

def is_isosceles_triangle (p s q : ℝ) : Prop := p = q

def angle (q s p : ℝ) := (q - s) - (s - p)

variables (r p q s : ℝ)

-- Define the given angles and equality conditions
def given_conditions : Prop := 
  is_straight_line r s p ∧
  angle q s p = 60 ∧
  is_isosceles_triangle p s q ∧
  r ≠ q 

-- The theorem we want to prove
theorem measure_angle_PQR_given_conditions : given_conditions r p q s → angle p q r = 120 := by
  sorry

end measure_angle_PQR_given_conditions_l1805_180508


namespace banks_policies_for_seniors_justified_l1805_180551

-- Defining conditions
def better_credit_repayment_reliability : Prop := sorry
def stable_pension_income : Prop := sorry
def indirect_younger_relative_contributions : Prop := sorry
def pensioners_inclination_to_save : Prop := sorry
def regular_monthly_income : Prop := sorry
def preference_for_long_term_deposits : Prop := sorry

-- Lean theorem statement using the conditions
theorem banks_policies_for_seniors_justified :
  better_credit_repayment_reliability →
  stable_pension_income →
  indirect_younger_relative_contributions →
  pensioners_inclination_to_save →
  regular_monthly_income →
  preference_for_long_term_deposits →
  (banks_should_offer_higher_deposit_and_lower_loan_rates_to_seniors : Prop) :=
by
  -- Insert proof here that given all the conditions the conclusion follows
  sorry -- proof not required, so skipping

end banks_policies_for_seniors_justified_l1805_180551


namespace solution_set_of_inequality_l1805_180511

theorem solution_set_of_inequality : {x : ℝ // |x - 2| > x - 2} = {x : ℝ // x < 2} :=
sorry

end solution_set_of_inequality_l1805_180511


namespace product_of_numbers_l1805_180518

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 72) (h2 : x - y = 20) : x * y = 1196 := 
sorry

end product_of_numbers_l1805_180518


namespace find_fx_l1805_180597

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

theorem find_fx (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x : ℝ, x < 0 → f x = x * (x - 1)) :
  ∀ x : ℝ, x > 0 → f x = x * (x + 1) :=
by
  sorry

end find_fx_l1805_180597


namespace quadratic_roots_transformation_l1805_180535

theorem quadratic_roots_transformation {a b c r s : ℝ}
  (h1 : r + s = -b / a)
  (h2 : r * s = c / a) :
  (∃ p q : ℝ, p = a * r + 2 * b ∧ q = a * s + 2 * b ∧ 
     (∀ x, x^2 - 3 * b * x + 2 * b^2 + a * c = (x - p) * (x - q))) :=
by
  sorry

end quadratic_roots_transformation_l1805_180535


namespace max_successful_free_throws_l1805_180599

theorem max_successful_free_throws (a b : ℕ) 
  (h1 : a + b = 105) 
  (h2 : a > 0)
  (h3 : b > 0)
  (ha : a % 3 = 0)
  (hb : b % 5 = 0)
  : (a / 3 + 3 * (b / 5)) ≤ 59 := sorry

end max_successful_free_throws_l1805_180599


namespace average_seq_13_to_52_l1805_180532

-- Define the sequence of natural numbers from 13 to 52
def seq : List ℕ := List.range' 13 52

-- Define the average of a list of natural numbers
def average (xs : List ℕ) : ℚ := (xs.sum : ℚ) / xs.length

-- Define the specific set of numbers and their average
theorem average_seq_13_to_52 : average seq = 32.5 := 
by 
  sorry

end average_seq_13_to_52_l1805_180532


namespace max_value_of_f_l1805_180503

def f (x : ℝ) : ℝ := 9 * x - 4 * x^2

theorem max_value_of_f :
  (∀ x : ℝ, f x ≤ 5.0625) ∧ (∃ x : ℝ, f x = 5.0625) :=
by
  sorry

end max_value_of_f_l1805_180503


namespace f_2015_l1805_180562

noncomputable def f : ℝ → ℝ := sorry
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_periodic : ∀ x : ℝ, f (x - 2) = -f x
axiom f_initial_segment : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) → f x = 2^x

theorem f_2015 : f 2015 = 1 / 2 :=
by
  -- Proof goes here
  sorry

end f_2015_l1805_180562


namespace iPhones_sold_l1805_180540

theorem iPhones_sold (x : ℕ) (h1 : (1000 * x + 18000 + 16000) / (x + 100) = 670) : x = 100 :=
by
  sorry

end iPhones_sold_l1805_180540


namespace pounds_of_coffee_bought_l1805_180507

theorem pounds_of_coffee_bought 
  (total_amount_gift_card : ℝ := 70) 
  (cost_per_pound : ℝ := 8.58) 
  (amount_left_on_card : ℝ := 35.68) :
  (total_amount_gift_card - amount_left_on_card) / cost_per_pound = 4 :=
sorry

end pounds_of_coffee_bought_l1805_180507


namespace quarterly_insurance_payment_l1805_180534

theorem quarterly_insurance_payment (annual_payment : ℕ) (quarters_in_year : ℕ) (quarterly_payment : ℕ) : 
  annual_payment = 1512 → quarters_in_year = 4 → quarterly_payment * quarters_in_year = annual_payment → quarterly_payment = 378 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  sorry

end quarterly_insurance_payment_l1805_180534


namespace imaginary_part_of_complex_z_l1805_180515

noncomputable def complex_z : ℂ := (1 + Complex.I) / (1 - Complex.I) + (1 - Complex.I) ^ 2

theorem imaginary_part_of_complex_z : complex_z.im = -1 := by
  sorry

end imaginary_part_of_complex_z_l1805_180515


namespace solve_equation_l1805_180579

noncomputable def equation_to_solve (x : ℝ) : ℝ :=
  1 / (4^(3*x) - 13 * 4^(2*x) + 51 * 4^x - 60) + 1 / (4^(2*x) - 7 * 4^x + 12)

theorem solve_equation :
  (equation_to_solve (1/2) = 0) ∧ (equation_to_solve (Real.log 6 / Real.log 4) = 0) :=
by {
  sorry
}

end solve_equation_l1805_180579


namespace intersection_eq_l1805_180531

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

-- Prove the intersection of A and B is {0, 1}
theorem intersection_eq : A ∩ B = {0, 1} := by
  sorry

end intersection_eq_l1805_180531


namespace remainder_when_sum_divided_by_30_l1805_180549

theorem remainder_when_sum_divided_by_30 {c d : ℕ} (p q : ℕ)
  (hc : c = 60 * p + 58)
  (hd : d = 90 * q + 85) :
  (c + d) % 30 = 23 :=
by
  sorry

end remainder_when_sum_divided_by_30_l1805_180549


namespace total_running_duration_l1805_180591

-- Conditions
def speed1 := 15 -- speed during the first part in mph
def time1 := 3 -- time during the first part in hours
def speed2 := 19 -- speed during the second part in mph
def distance2 := 190 -- distance during the second part in miles

-- Initialize
def distance1 := speed1 * time1 -- distance covered in the first part in miles

def time2 := distance2 / speed2 -- time to cover the distance in the second part in hours

-- Total duration
def total_duration := time1 + time2

-- Proof statement
theorem total_running_duration : total_duration = 13 :=
by
  sorry

end total_running_duration_l1805_180591


namespace remainder_of_95_times_97_div_12_l1805_180547

theorem remainder_of_95_times_97_div_12 : 
  (95 * 97) % 12 = 11 := by
  sorry

end remainder_of_95_times_97_div_12_l1805_180547


namespace correct_operation_l1805_180566

theorem correct_operation : 
  ¬(3 * x^2 + 2 * x^2 = 6 * x^4) ∧ 
  ¬((-2 * x^2)^3 = -6 * x^6) ∧ 
  ¬(x^3 * x^2 = x^6) ∧ 
  (-6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y) :=
by
  sorry

end correct_operation_l1805_180566


namespace vector_magnitude_parallel_l1805_180545

/-- Given two plane vectors a = (1, 2) and b = (-2, y),
if a is parallel to b, then |2a - b| = 4 * sqrt 5. -/
theorem vector_magnitude_parallel (y : ℝ) 
  (h_parallel : (1 : ℝ) / (-2 : ℝ) = (2 : ℝ) / y) : 
  ‖2 • (1, 2) - (-2, y)‖ = 4 * Real.sqrt 5 := 
by
  sorry

end vector_magnitude_parallel_l1805_180545


namespace largest_gold_coins_l1805_180578

theorem largest_gold_coins (k : ℤ) (h1 : 13 * k + 3 < 100) : 91 ≤ 13 * k + 3 :=
by
  sorry

end largest_gold_coins_l1805_180578


namespace hamburgers_sold_in_winter_l1805_180502

theorem hamburgers_sold_in_winter:
  ∀ (T x : ℕ), 
  (T = 5 * 4) → 
  (5 + 6 + 4 + x = T) →
  (x = 5) :=
by
  intros T x hT hTotal
  sorry

end hamburgers_sold_in_winter_l1805_180502


namespace correct_statement_A_l1805_180529

-- Declare Avogadro's constant
def Avogadro_constant : ℝ := 6.022e23

-- Given conditions
def gas_mass_ethene : ℝ := 5.6 -- grams of ethylene
def gas_mass_cyclopropane : ℝ := 5.6 -- grams of cyclopropane
def gas_combined_carbon_atoms : ℝ := 0.4 * Avogadro_constant

-- Assertion to prove
theorem correct_statement_A :
    gas_combined_carbon_atoms = 0.4 * Avogadro_constant :=
by
  sorry

end correct_statement_A_l1805_180529


namespace output_value_is_3_l1805_180538

-- Define the variables and the program logic
def program (a b : ℕ) : ℕ :=
  if a > b then a else b

-- The theorem statement
theorem output_value_is_3 (a b : ℕ) (ha : a = 2) (hb : b = 3) : program a b = 3 :=
by
  -- Automatically assume the given conditions and conclude the proof. The actual proof is skipped.
  sorry

end output_value_is_3_l1805_180538


namespace exists_k_ge_2_l1805_180513

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def weak (a b n : ℕ) : Prop :=
  ¬ ∃ x y : ℕ, a * x + b * y = n

theorem exists_k_ge_2 (a b n : ℕ) (h_coprime : coprime a b) (h_positive : 0 < n) (h_weak : weak a b n) (h_bound : n < a * b / 6) :
  ∃ k : ℕ, 2 ≤ k ∧ weak a b (k * n) :=
sorry

end exists_k_ge_2_l1805_180513


namespace total_cups_sold_is_46_l1805_180557

-- Define the number of cups sold last week
def cups_sold_last_week : ℕ := 20

-- Define the percentage increase
def percentage_increase : ℕ := 30

-- Calculate the number of cups sold this week
def cups_sold_this_week : ℕ := cups_sold_last_week + (cups_sold_last_week * percentage_increase / 100)

-- Calculate the total number of cups sold over both weeks
def total_cups_sold : ℕ := cups_sold_last_week + cups_sold_this_week

-- State the theorem to prove the total number of cups sold
theorem total_cups_sold_is_46 : total_cups_sold = 46 := sorry

end total_cups_sold_is_46_l1805_180557


namespace rent_3600_yields_88_optimal_rent_is_4100_and_max_revenue_is_304200_l1805_180573

/-
Via conditions:
1. The rental company owns 100 cars.
2. When the monthly rent for each car is set at 3000 yuan, all cars can be rented out.
3. For every 50 yuan increase in the monthly rent per car, there will be one more car that is not rented out.
4. The maintenance cost for each rented car is 200 yuan per month.
-/

noncomputable def num_rented_cars (rent_per_car : ℕ) : ℕ :=
  if rent_per_car < 3000 then 100 else max 0 (100 - (rent_per_car - 3000) / 50)

noncomputable def monthly_revenue (rent_per_car : ℕ) : ℕ :=
  let cars_rented := num_rented_cars rent_per_car
  let maintenance_cost := 200 * cars_rented
  (rent_per_car - maintenance_cost) * cars_rented

theorem rent_3600_yields_88 : num_rented_cars 3600 = 88 :=
  sorry

theorem optimal_rent_is_4100_and_max_revenue_is_304200 :
  ∃ rent_per_car, rent_per_car = 4100 ∧ monthly_revenue rent_per_car = 304200 :=
  sorry

end rent_3600_yields_88_optimal_rent_is_4100_and_max_revenue_is_304200_l1805_180573


namespace sum_gcd_lcm_l1805_180561

theorem sum_gcd_lcm (a b c d : ℕ) (ha : a = 15) (hb : b = 45) (hc : c = 30) :
  Int.gcd a b + Nat.lcm a c = 45 := 
by
  sorry

end sum_gcd_lcm_l1805_180561


namespace part1_solution_set_part2_values_a_b_part3_range_m_l1805_180506

-- Definitions for the given functions
def y1 (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def y2 (x : ℝ) : ℝ := x^2 + x - 2

-- Proof that the solution set for y2 < 0 is (-2, 1)
theorem part1_solution_set : ∀ x : ℝ, y2 x < 0 ↔ (x > -2 ∧ x < 1) :=
sorry

-- Given |y1| ≤ |y2| for all x ∈ ℝ, prove that a = 1 and b = -2
theorem part2_values_a_b (a b : ℝ) : (∀ x : ℝ, |y1 x a b| ≤ |y2 x|) → a = 1 ∧ b = -2 :=
sorry

-- Given y1 > (m-2)x - m for all x > 1 under condition from part 2, prove the range for m is (-∞, 2√2 + 5)
theorem part3_range_m (a b : ℝ) (m : ℝ) : 
  (∀ x : ℝ, |y1 x a b| ≤ |y2 x|) → a = 1 ∧ b = -2 →
  (∀ x : ℝ, x > 1 → y1 x a b > (m-2) * x - m) → m < 2 * Real.sqrt 2 + 5 :=
sorry

end part1_solution_set_part2_values_a_b_part3_range_m_l1805_180506


namespace one_sofa_in_room_l1805_180588

def num_sofas_in_room : ℕ :=
  let num_4_leg_tables := 4
  let num_4_leg_chairs := 2
  let num_3_leg_tables := 3
  let num_1_leg_table := 1
  let num_2_leg_rocking_chairs := 1
  let total_legs := 40

  let legs_of_4_leg_tables := num_4_leg_tables * 4
  let legs_of_4_leg_chairs := num_4_leg_chairs * 4
  let legs_of_3_leg_tables := num_3_leg_tables * 3
  let legs_of_1_leg_table := num_1_leg_table * 1
  let legs_of_2_leg_rocking_chairs := num_2_leg_rocking_chairs * 2

  let accounted_legs := legs_of_4_leg_tables + legs_of_4_leg_chairs + legs_of_3_leg_tables + legs_of_1_leg_table + legs_of_2_leg_rocking_chairs

  let remaining_legs := total_legs - accounted_legs

  let sofa_legs := 4
  remaining_legs / sofa_legs

theorem one_sofa_in_room : num_sofas_in_room = 1 :=
  by
    unfold num_sofas_in_room
    rfl

end one_sofa_in_room_l1805_180588


namespace rows_of_seats_l1805_180575

theorem rows_of_seats (students sections_per_row students_per_section : ℕ) (h1 : students_per_section = 2) (h2 : sections_per_row = 2) (h3 : students = 52) :
  (students / students_per_section / sections_per_row) = 13 :=
sorry

end rows_of_seats_l1805_180575


namespace completing_square_correct_l1805_180572

theorem completing_square_correct :
  ∀ x : ℝ, (x^2 - 4 * x + 2 = 0) ↔ ((x - 2)^2 = 2) := 
by
  intros x
  sorry

end completing_square_correct_l1805_180572


namespace probability_no_prize_l1805_180528

theorem probability_no_prize : (1 : ℚ) - (1 : ℚ) / (50 * 50) = 2499 / 2500 :=
by
  sorry

end probability_no_prize_l1805_180528


namespace skips_in_one_meter_l1805_180510

variable (p q r s t u : ℕ)

theorem skips_in_one_meter (h1 : p * s * u = q * r * t) : 1 = (p * r * t) / (u * s * q) := by
  sorry

end skips_in_one_meter_l1805_180510


namespace nanometers_to_scientific_notation_l1805_180542

theorem nanometers_to_scientific_notation :
  (246 : ℝ) * (10 ^ (-9 : ℝ)) = (2.46 : ℝ) * (10 ^ (-7 : ℝ)) :=
by
  sorry

end nanometers_to_scientific_notation_l1805_180542


namespace parabola_point_value_l1805_180517

variable {x₀ y₀ : ℝ}

theorem parabola_point_value
  (h₁ : y₀^2 = 4 * x₀)
  (h₂ : (Real.sqrt ((x₀ - 1)^2 + y₀^2) = 5/4 * x₀)) :
  x₀ = 4 := by
  sorry

end parabola_point_value_l1805_180517


namespace log_inequality_l1805_180598

theorem log_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  2 * ((Real.log a / Real.log b) / (a + b) + (Real.log b / Real.log c) / (b + c) + (Real.log c / Real.log a) / (c + a)) 
    ≥ 9 / (a + b + c) :=
by
  sorry

end log_inequality_l1805_180598


namespace quadratic_discriminant_one_solution_l1805_180543

theorem quadratic_discriminant_one_solution (m : ℚ) : 
  (3 * (1 : ℚ))^2 - 12 * m = 0 → m = 49 / 12 := 
by {
  sorry
}

end quadratic_discriminant_one_solution_l1805_180543


namespace youth_gathering_l1805_180571

theorem youth_gathering (x : ℕ) (h1 : ∃ x, 9 * (2 * x + 12) = 20 * x) : 
  2 * x + 12 = 120 :=
by sorry

end youth_gathering_l1805_180571


namespace no_sensor_in_option_B_l1805_180544

/-- Define the technologies and whether they involve sensors --/
def technology_involves_sensor (opt : String) : Prop :=
  opt = "A" ∨ opt = "C" ∨ opt = "D"

theorem no_sensor_in_option_B :
  ¬ technology_involves_sensor "B" :=
by
  -- We assume the proof for the sake of this example.
  sorry

end no_sensor_in_option_B_l1805_180544


namespace transformed_point_of_function_l1805_180501

theorem transformed_point_of_function (f : ℝ → ℝ) (h : f 1 = -2) : f (-1) + 1 = -1 :=
by
  sorry

end transformed_point_of_function_l1805_180501


namespace max_sum_combined_shape_l1805_180563

-- Definitions for the initial prism
def faces_prism := 6
def edges_prism := 12
def vertices_prism := 8

-- Definitions for the changes when pyramid is added to a rectangular face
def additional_faces_rect := 4
def additional_edges_rect := 4
def additional_vertices_rect := 1

-- Definition for the maximum sum calculation
def max_sum := faces_prism - 1 + additional_faces_rect + 
               edges_prism + additional_edges_rect + 
               vertices_prism + additional_vertices_rect

-- The theorem to prove the maximum sum
theorem max_sum_combined_shape : max_sum = 34 :=
by
  sorry

end max_sum_combined_shape_l1805_180563


namespace area_of_regular_octagon_l1805_180520

theorem area_of_regular_octagon (BDEF_is_rectangle : true) (AB : ℝ) (BC : ℝ) 
    (capture_regular_octagon : true) (AB_eq_1 : AB = 1) (BC_eq_2 : BC = 2)
    (octagon_perimeter_touch : ∀ x, x = 1) : 
    ∃ A : ℝ, A = 11 :=
by
  sorry

end area_of_regular_octagon_l1805_180520


namespace units_digit_7_pow_3_l1805_180594

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_3 : units_digit (7^3) = 3 :=
by
  -- Proof of the theorem would go here
  sorry

end units_digit_7_pow_3_l1805_180594


namespace average_weight_increase_l1805_180539

theorem average_weight_increase 
  (A : ℝ) (X : ℝ)
  (h1 : 8 * (A + X) = 8 * A + 36) :
  X = 4.5 := 
sorry

end average_weight_increase_l1805_180539


namespace transform_to_quadratic_l1805_180519

theorem transform_to_quadratic :
  (∀ x : ℝ, (x + 1) ^ 2 + (x - 2) * (x + 2) = 1 ↔ 2 * x ^ 2 + 2 * x - 4 = 0) :=
sorry

end transform_to_quadratic_l1805_180519


namespace quadratic_not_proposition_l1805_180595

def is_proposition (P : Prop) : Prop := ∃ (b : Bool), (b = true ∨ b = false)

theorem quadratic_not_proposition : ¬ is_proposition (∃ x : ℝ, x^2 + 2*x - 3 < 0) :=
by 
  sorry

end quadratic_not_proposition_l1805_180595


namespace second_bus_percentage_full_l1805_180546

noncomputable def bus_capacity : ℕ := 150
noncomputable def employees_in_buses : ℕ := 195
noncomputable def first_bus_percentage : ℚ := 0.60

theorem second_bus_percentage_full :
  let employees_first_bus := first_bus_percentage * bus_capacity
  let employees_second_bus := (employees_in_buses : ℚ) - employees_first_bus
  let second_bus_percentage := (employees_second_bus / bus_capacity) * 100
  second_bus_percentage = 70 :=
by
  sorry

end second_bus_percentage_full_l1805_180546


namespace second_discount_correct_l1805_180559

noncomputable def second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : ℝ :=
  let first_discount_amount := first_discount / 100 * original_price
  let price_after_first_discount := original_price - first_discount_amount
  let second_discount_amount := price_after_first_discount - final_price
  (second_discount_amount / price_after_first_discount) * 100

theorem second_discount_correct :
  second_discount_percentage 510 12 381.48 = 15 :=
by
  sorry

end second_discount_correct_l1805_180559


namespace math_problem_l1805_180585

noncomputable def a : ℝ := (0.96)^3 
noncomputable def b : ℝ := (0.1)^3 
noncomputable def c : ℝ := (0.96)^2 
noncomputable def d : ℝ := (0.1)^2 

theorem math_problem : a - b / c + 0.096 + d = 0.989651 := 
by 
  -- skip proof 
  sorry

end math_problem_l1805_180585


namespace intersection_A_B_l1805_180587

open Set Real

def A := { x : ℝ | x ^ 2 - 6 * x + 5 ≤ 0 }
def B := { x : ℝ | ∃ y : ℝ, y = log (x - 2) / log 2 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 2 < x ∧ x ≤ 5 } :=
by
  sorry

end intersection_A_B_l1805_180587


namespace evaluate_g_at_3_l1805_180589

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 - 7 * x + 3

theorem evaluate_g_at_3 : g 3 = 126 := 
by 
  sorry

end evaluate_g_at_3_l1805_180589


namespace coin_flip_probability_l1805_180522

noncomputable def probability_successful_outcomes : ℚ :=
  let total_outcomes := 32
  let successful_outcomes := 3
  successful_outcomes / total_outcomes

theorem coin_flip_probability :
  probability_successful_outcomes = 3 / 32 :=
by
  sorry

end coin_flip_probability_l1805_180522


namespace range_of_m_l1805_180584

-- Define the conditions based on the problem statement
def equation (x m : ℝ) : Prop := (2 * x + m) = (x - 1)

-- The goal is to prove that if there exists a positive solution x to the equation, then m < -1
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, equation x m ∧ x > 0) → m < -1 :=
by
  sorry

end range_of_m_l1805_180584


namespace solution_to_first_equation_solution_to_second_equation_l1805_180558

theorem solution_to_first_equation (x : ℝ) : 
  x^2 - 6 * x + 1 = 0 ↔ x = 3 + 2 * Real.sqrt 2 ∨ x = 3 - 2 * Real.sqrt 2 :=
by sorry

theorem solution_to_second_equation (x : ℝ) : 
  (2 * x - 3)^2 = 5 * (2 * x - 3) ↔ x = 3 / 2 ∨ x = 4 :=
by sorry

end solution_to_first_equation_solution_to_second_equation_l1805_180558


namespace cost_of_football_correct_l1805_180521

-- We define the variables for the costs
def total_amount_spent : ℝ := 20.52
def cost_of_marbles : ℝ := 9.05
def cost_of_baseball : ℝ := 6.52
def cost_of_football : ℝ := total_amount_spent - cost_of_marbles - cost_of_baseball

-- We now state what needs to be proven: that Mike spent $4.95 on the football.
theorem cost_of_football_correct : cost_of_football = 4.95 := by
  sorry

end cost_of_football_correct_l1805_180521


namespace replace_digits_correct_l1805_180525

def digits_eq (a b c d e : ℕ) : Prop :=
  5 * 10 + a + (b * 100) + (c * 10) + 3 = (d * 1000) + (e * 100) + 1

theorem replace_digits_correct :
  ∃ (a b c d e : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
    digits_eq a b c d e ∧ a = 1 ∧ b = 1 ∧ c = 4 ∧ d = 1 ∧ e = 4 :=
by
  sorry

end replace_digits_correct_l1805_180525


namespace other_number_eq_462_l1805_180512

theorem other_number_eq_462 (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 4620) 
  (gcd_ab : Nat.gcd a b = 21) 
  (a_eq : a = 210) : b = 462 := 
by
  sorry

end other_number_eq_462_l1805_180512


namespace relationship_m_n_l1805_180516

theorem relationship_m_n (m n : ℕ) (h : 10 / (m + 10 + n) = (m + n) / (m + 10 + n)) : m + n = 10 := 
by sorry

end relationship_m_n_l1805_180516


namespace carl_garden_area_l1805_180541

theorem carl_garden_area 
  (total_posts : Nat)
  (length_post_distance : Nat)
  (corner_posts : Nat)
  (longer_side_multiplier : Nat)
  (posts_per_shorter_side : Nat)
  (posts_per_longer_side : Nat)
  (shorter_side_distance : Nat)
  (longer_side_distance : Nat) :
  total_posts = 24 →
  length_post_distance = 5 →
  corner_posts = 4 →
  longer_side_multiplier = 2 →
  posts_per_shorter_side = (24 + 4) / 6 →
  posts_per_longer_side = (24 + 4) / 6 * 2 →
  shorter_side_distance = (posts_per_shorter_side - 1) * length_post_distance →
  longer_side_distance = (posts_per_longer_side - 1) * length_post_distance →
  shorter_side_distance * longer_side_distance = 900 :=
by
  intros
  sorry

end carl_garden_area_l1805_180541


namespace find_d_l1805_180509

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_d (d e : ℝ) (h1 : -(-6) / 3 = 2) (h2 : 3 + d + e - 6 = 9) (h3 : -d / 3 = 6) : d = -18 :=
by
  sorry

end find_d_l1805_180509


namespace megan_numbers_difference_l1805_180553

theorem megan_numbers_difference 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h_mean3 : (x1 + x2 + x3) / 3 = -3)
  (h_mean4 : (x1 + x2 + x3 + x4) / 4 = 4)
  (h_mean5 : (x1 + x2 + x3 + x4 + x5) / 5 = -5) :
  x4 - x5 = 66 :=
by
  sorry

end megan_numbers_difference_l1805_180553


namespace moles_H2O_formed_l1805_180576

-- Define the conditions
def moles_HCl : ℕ := 6
def moles_CaCO3 : ℕ := 3
def moles_CaCl2 : ℕ := 3
def moles_CO2 : ℕ := 3

-- Proposition that we need to prove
theorem moles_H2O_formed : moles_CaCl2 = 3 ∧ moles_CO2 = 3 ∧ moles_CaCO3 = 3 ∧ moles_HCl = 6 → moles_CaCO3 = 3 := by
  sorry

end moles_H2O_formed_l1805_180576


namespace isometric_curve_l1805_180504

noncomputable def Q (a b c x y : ℝ) := a * x^2 + 2 * b * x * y + c * y^2

theorem isometric_curve (a b c d e f : ℝ) (h : a * c - b^2 = 0) :
  ∃ (p : ℝ), (Q a b c x y + 2 * d * x + 2 * e * y = f → 
    (y^2 = 2 * p * x) ∨ 
    (∃ c' : ℝ, y^2 = c'^2) ∨ 
    y^2 = 0 ∨ 
    ∀ x y : ℝ, false) :=
sorry

end isometric_curve_l1805_180504
