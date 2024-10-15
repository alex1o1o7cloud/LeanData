import Mathlib

namespace NUMINAMATH_GPT_a_investment_l919_91953

theorem a_investment
  (b_investment : ℝ) (c_investment : ℝ) (c_share_profit : ℝ) (total_profit : ℝ)
  (h1 : b_investment = 45000)
  (h2 : c_investment = 50000)
  (h3 : c_share_profit = 36000)
  (h4 : total_profit = 90000) :
  ∃ A : ℝ, A = 30000 :=
by {
  sorry
}

end NUMINAMATH_GPT_a_investment_l919_91953


namespace NUMINAMATH_GPT_Billy_is_45_l919_91904

variable (B J : ℕ)

-- Condition 1: Billy's age is three times Joe's age
def condition1 : Prop := B = 3 * J

-- Condition 2: The sum of their ages is 60
def condition2 : Prop := B + J = 60

-- The theorem we want to prove: Billy's age is 45
theorem Billy_is_45 (h1 : condition1 B J) (h2 : condition2 B J) : B = 45 := 
sorry

end NUMINAMATH_GPT_Billy_is_45_l919_91904


namespace NUMINAMATH_GPT_extreme_point_condition_l919_91901

variable {R : Type*} [OrderedRing R]

def f (x a b : R) : R := x^3 - a*x - b

theorem extreme_point_condition (a b x0 x1 : R) (h₁ : ∀ x : R, f x a b = x^3 - a*x - b)
  (h₂ : f x0 a b = x0^3 - a*x0 - b)
  (h₃ : f x1 a b = x1^3 - a*x1 - b)
  (has_extreme : ∃ x0 : R, 3*x0^2 = a) 
  (hx1_extreme : f x1 a b = f x0 a b) 
  (hx1_x0_diff : x1 ≠ x0) :
  x1 + 2*x0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_extreme_point_condition_l919_91901


namespace NUMINAMATH_GPT_product_factors_eq_l919_91939

theorem product_factors_eq :
  (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) * (1 - 1/6) * (1 - 1/7) * (1 - 1/8) * (1 - 1/9) * (1 - 1/10) * (1 - 1/11) = 1 / 11 := 
by
  sorry

end NUMINAMATH_GPT_product_factors_eq_l919_91939


namespace NUMINAMATH_GPT_polygon_angle_ratio_pairs_count_l919_91926

theorem polygon_angle_ratio_pairs_count :
  ∃ (m n : ℕ), (∃ (k : ℕ), (k > 0) ∧ (180 - 360 / ↑m) / (180 - 360 / ↑n) = 4 / 3
  ∧ Prime n ∧ (m - 6) * (n + 8) = 48 ∧ 
  ∃! (m n : ℕ), (180 - 360 / ↑m = (4 * (180 - 360 / ↑n)) / 3)) :=
sorry  -- Proof omitted, providing only the statement

end NUMINAMATH_GPT_polygon_angle_ratio_pairs_count_l919_91926


namespace NUMINAMATH_GPT_cube_surface_area_l919_91963

theorem cube_surface_area (a : ℝ) (h : a = 1) :
    6 * a^2 = 6 := by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l919_91963


namespace NUMINAMATH_GPT_algae_cell_count_at_day_nine_l919_91985

noncomputable def initial_cells : ℕ := 5
noncomputable def division_frequency_days : ℕ := 3
noncomputable def total_days : ℕ := 9

def number_of_cycles (total_days division_frequency_days : ℕ) : ℕ :=
  total_days / division_frequency_days

noncomputable def common_ratio : ℕ := 2

noncomputable def number_of_cells_after_n_days (initial_cells common_ratio number_of_cycles : ℕ) : ℕ :=
  initial_cells * common_ratio ^ (number_of_cycles - 1)

theorem algae_cell_count_at_day_nine : number_of_cells_after_n_days initial_cells common_ratio (number_of_cycles total_days division_frequency_days) = 20 :=
by
  sorry

end NUMINAMATH_GPT_algae_cell_count_at_day_nine_l919_91985


namespace NUMINAMATH_GPT_number_of_continents_collected_l919_91928

-- Definitions of the given conditions
def books_per_continent : ℕ := 122
def total_books : ℕ := 488

-- The mathematical statement to be proved
theorem number_of_continents_collected :
  total_books / books_per_continent = 4 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_continents_collected_l919_91928


namespace NUMINAMATH_GPT_greatest_four_digit_number_l919_91983

theorem greatest_four_digit_number (x : ℕ) :
  x ≡ 1 [MOD 7] ∧ x ≡ 5 [MOD 8] ∧ 1000 ≤ x ∧ x < 10000 → x = 9997 :=
by
  sorry

end NUMINAMATH_GPT_greatest_four_digit_number_l919_91983


namespace NUMINAMATH_GPT_totalSleepIsThirtyHours_l919_91906

-- Define constants and conditions
def recommendedSleep : ℝ := 8
def sleepOnTwoDays : ℝ := 3
def percentageSleepOnOtherDays : ℝ := 0.6
def daysInWeek : ℕ := 7
def daysWithThreeHoursSleep : ℕ := 2
def remainingDays : ℕ := daysInWeek - daysWithThreeHoursSleep

-- Define total sleep calculation
theorem totalSleepIsThirtyHours :
  let sleepOnFirstTwoDays := (daysWithThreeHoursSleep : ℝ) * sleepOnTwoDays
  let sleepOnRemainingDays := (remainingDays : ℝ) * (recommendedSleep * percentageSleepOnOtherDays)
  sleepOnFirstTwoDays + sleepOnRemainingDays = 30 := 
by
  sorry

end NUMINAMATH_GPT_totalSleepIsThirtyHours_l919_91906


namespace NUMINAMATH_GPT_daps_equivalent_to_dips_l919_91919

-- Definitions from conditions
def daps (n : ℕ) : ℕ := n
def dops (n : ℕ) : ℕ := n
def dips (n : ℕ) : ℕ := n

-- Given conditions
def equivalence_daps_dops : daps 8 = dops 6 := sorry
def equivalence_dops_dips : dops 3 = dips 11 := sorry

-- Proof problem
theorem daps_equivalent_to_dips (n : ℕ) (h1 : daps 8 = dops 6) (h2 : dops 3 = dips 11) : daps 24 = dips 66 :=
sorry

end NUMINAMATH_GPT_daps_equivalent_to_dips_l919_91919


namespace NUMINAMATH_GPT_b_integer_iff_a_special_form_l919_91975

theorem b_integer_iff_a_special_form (a : ℝ) (b : ℝ) 
  (h1 : a > 0) 
  (h2 : b = (a + Real.sqrt (a ^ 2 + 1)) ^ (1 / 3) + (a - Real.sqrt (a ^ 2 + 1)) ^ (1 / 3)) : 
  (∃ (n : ℕ), a = 1 / 2 * (n * (n^2 + 3))) ↔ (∃ (n : ℕ), b = n) :=
sorry

end NUMINAMATH_GPT_b_integer_iff_a_special_form_l919_91975


namespace NUMINAMATH_GPT_find_wrongly_written_height_l919_91927

def wrongly_written_height
  (n : ℕ)
  (avg_height_incorrect : ℝ)
  (actual_height : ℝ)
  (avg_height_correct : ℝ) : ℝ :=
  let total_height_incorrect := n * avg_height_incorrect
  let total_height_correct := n * avg_height_correct
  let height_difference := total_height_incorrect - total_height_correct
  actual_height + height_difference

theorem find_wrongly_written_height :
  wrongly_written_height 35 182 106 180 = 176 :=
by
  sorry

end NUMINAMATH_GPT_find_wrongly_written_height_l919_91927


namespace NUMINAMATH_GPT_interior_triangle_area_l919_91991

theorem interior_triangle_area (s1 s2 s3 : ℝ) (hs1 : s1 = 15) (hs2 : s2 = 6) (hs3 : s3 = 15) 
  (a1 a2 a3 : ℝ) (ha1 : a1 = 225) (ha2 : a2 = 36) (ha3 : a3 = 225) 
  (h1 : s1 * s1 = a1) (h2 : s2 * s2 = a2) (h3 : s3 * s3 = a3) :
  (1/2) * s1 * s2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_interior_triangle_area_l919_91991


namespace NUMINAMATH_GPT_jane_paid_cashier_l919_91984

-- Define the conditions in Lean
def skirts_bought : ℕ := 2
def price_per_skirt : ℕ := 13
def blouses_bought : ℕ := 3
def price_per_blouse : ℕ := 6
def change_received : ℤ := 56

-- Calculate the total cost in Lean
def cost_of_skirts : ℕ := skirts_bought * price_per_skirt
def cost_of_blouses : ℕ := blouses_bought * price_per_blouse
def total_cost : ℕ := cost_of_skirts + cost_of_blouses
def amount_paid : ℤ := total_cost + change_received

-- Lean statement to prove the question
theorem jane_paid_cashier :
  amount_paid = 100 :=
by
  sorry

end NUMINAMATH_GPT_jane_paid_cashier_l919_91984


namespace NUMINAMATH_GPT_model_price_and_schemes_l919_91946

theorem model_price_and_schemes :
  ∃ (x y : ℕ), 3 * x = 2 * y ∧ x + 2 * y = 80 ∧ x = 20 ∧ y = 30 ∧ 
  ∃ (count m : ℕ), 468 ≤ m ∧ m ≤ 480 ∧ 
                   (20 * m + 30 * (800 - m) ≤ 19320) ∧ 
                   (800 - m ≥ 2 * m / 3) ∧ 
                   count = 13 ∧ 
                   800 - 480 = 320 :=
sorry

end NUMINAMATH_GPT_model_price_and_schemes_l919_91946


namespace NUMINAMATH_GPT_determine_f_2048_l919_91941

theorem determine_f_2048 (f : ℕ → ℝ)
  (A1 : ∀ a b n : ℕ, a > 0 → b > 0 → a * b = 2^n → f a + f b = n^2)
  : f 2048 = 121 := by
  sorry

end NUMINAMATH_GPT_determine_f_2048_l919_91941


namespace NUMINAMATH_GPT_average_speed_ratio_l919_91968

theorem average_speed_ratio 
  (jack_marathon_distance : ℕ) (jack_marathon_time : ℕ) 
  (jill_marathon_distance : ℕ) (jill_marathon_time : ℕ)
  (h1 : jack_marathon_distance = 40) (h2 : jack_marathon_time = 45) 
  (h3 : jill_marathon_distance = 40) (h4 : jill_marathon_time = 40) :
  (889 : ℕ) / 1000 = (jack_marathon_distance / jack_marathon_time) / 
                      (jill_marathon_distance / jill_marathon_time) :=
by
  sorry

end NUMINAMATH_GPT_average_speed_ratio_l919_91968


namespace NUMINAMATH_GPT_number_divided_by_five_is_same_as_three_added_l919_91918

theorem number_divided_by_five_is_same_as_three_added :
  ∃ x : ℚ, x / 5 = x + 3 ∧ x = -15 / 4 :=
by
  sorry

end NUMINAMATH_GPT_number_divided_by_five_is_same_as_three_added_l919_91918


namespace NUMINAMATH_GPT_focus_of_parabola_l919_91995

theorem focus_of_parabola :
  (∃ (x y : ℝ), y = 4 * x ^ 2 - 8 * x - 12 ∧ x = 1 ∧ y = -15.9375) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l919_91995


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_div_four_l919_91966

theorem tan_alpha_plus_pi_div_four
  (α : ℝ)
  (a : ℝ × ℝ := (3, 4))
  (b : ℝ × ℝ := (Real.sin α, Real.cos α))
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) :
  Real.tan (α + Real.pi / 4) = 7 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_div_four_l919_91966


namespace NUMINAMATH_GPT_factorization_of_expression_l919_91960

-- Define variables
variables {a x y : ℝ}

-- State the problem
theorem factorization_of_expression : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) :=
  sorry

end NUMINAMATH_GPT_factorization_of_expression_l919_91960


namespace NUMINAMATH_GPT_total_students_in_class_l919_91974

def period_length : ℕ := 40
def periods_per_student : ℕ := 4
def time_per_student : ℕ := 5

theorem total_students_in_class :
  ((period_length / time_per_student) * periods_per_student) = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l919_91974


namespace NUMINAMATH_GPT_student_percentage_first_subject_l919_91994

theorem student_percentage_first_subject
  (P : ℝ)
  (h1 : (P + 60 + 70) / 3 = 60) : P = 50 :=
  sorry

end NUMINAMATH_GPT_student_percentage_first_subject_l919_91994


namespace NUMINAMATH_GPT_max_distance_increases_l919_91909

noncomputable def largest_n_for_rearrangement (C : ℕ) (marked_points : ℕ) : ℕ :=
  670

theorem max_distance_increases (C : ℕ) (marked_points : ℕ) (n : ℕ) (dist : ℕ → ℕ → ℕ) :
  ∀ i j, i < marked_points → j < marked_points →
    dist i j ≤ n → 
    (∃ rearrangement : ℕ → ℕ, 
    ∀ i j, i < marked_points → j < marked_points → 
      dist (rearrangement i) (rearrangement j) > dist i j) → 
    n ≤ largest_n_for_rearrangement C marked_points := 
by
  sorry

end NUMINAMATH_GPT_max_distance_increases_l919_91909


namespace NUMINAMATH_GPT_distance_between_points_A_and_B_l919_91965

theorem distance_between_points_A_and_B :
  ∃ (d : ℝ), 
    -- Distance must be non-negative
    d ≥ 0 ∧
    -- Condition 1: Car 3 reaches point A at 10:00 AM (3 hours after 7:00 AM)
    (∃ V3 : ℝ, V3 = d / 6) ∧ 
    -- Condition 2: Car 2 reaches point A at 10:30 AM (3.5 hours after 7:00 AM)
    (∃ V2 : ℝ, V2 = 2 * d / 7) ∧ 
    -- Condition 3: When Car 1 and Car 3 meet, Car 2 has traveled exactly 3/8 of d
    (∃ V1 : ℝ, V1 = (d - 84) / 7 ∧ 2 * V1 + 2 * V3 = 8 * V2 / 3) ∧ 
    -- Required: The distance between A and B is 336 km
    d = 336 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_A_and_B_l919_91965


namespace NUMINAMATH_GPT_ratio_of_cost_to_marked_price_l919_91976

variable (p : ℝ)

theorem ratio_of_cost_to_marked_price :
  let selling_price := (3/4) * p
  let cost_price := (5/8) * selling_price
  cost_price / p = 15 / 32 :=
by
  let selling_price := (3 / 4) * p
  let cost_price := (5 / 8) * selling_price
  sorry

end NUMINAMATH_GPT_ratio_of_cost_to_marked_price_l919_91976


namespace NUMINAMATH_GPT_minimize_t_l919_91920

variable (Q : ℝ) (Q_1 Q_2 Q_3 Q_4 Q_5 Q_6 Q_7 Q_8 Q_9 : ℝ)

-- Definition of the sum of undirected lengths
def t (Q : ℝ) := 
  abs (Q - Q_1) + abs (Q - Q_2) + abs (Q - Q_3) + 
  abs (Q - Q_4) + abs (Q - Q_5) + abs (Q - Q_6) + 
  abs (Q - Q_7) + abs (Q - Q_8) + abs (Q - Q_9)

-- Statement that t is minimized when Q = Q_5
theorem minimize_t : ∀ Q : ℝ, t Q ≥ t Q_5 := 
sorry

end NUMINAMATH_GPT_minimize_t_l919_91920


namespace NUMINAMATH_GPT_ordered_pairs_satisfying_condition_l919_91905

theorem ordered_pairs_satisfying_condition : 
  ∃! (pairs : Finset (ℕ × ℕ)),
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ 
      m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 144) ∧ 
    pairs.card = 4 := sorry

end NUMINAMATH_GPT_ordered_pairs_satisfying_condition_l919_91905


namespace NUMINAMATH_GPT_function_is_linear_l919_91925

theorem function_is_linear (f : ℝ → ℝ) :
  (∀ a b c d : ℝ,
    a ≠ b → b ≠ c → c ≠ d → d ≠ a →
    (a ≠ c ∧ b ≠ d ∧ a ≠ d ∧ b ≠ c) →
    (a - b) / (b - c) + (a - d) / (d - c) = 0 →
    (f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f d ∧ f a ≠ f c ∧ f a ≠ f d ∧ f b ≠ f d) →
    (f a - f b) / (f b - f c) + (f a - f d) / (f d - f c) = 0) →
  ∃ m c : ℝ, ∀ x : ℝ, f x = m * x + c :=
by
  sorry

end NUMINAMATH_GPT_function_is_linear_l919_91925


namespace NUMINAMATH_GPT_AM_GM_inequality_example_l919_91993

theorem AM_GM_inequality_example (a b c d : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_prod : a * b * c * d = 1) :
  a^3 + b^3 + c^3 + d^3 ≥ max (a + b + c + d) (1 / a + 1 / b + 1 / c + 1 / d) :=
by
  sorry

end NUMINAMATH_GPT_AM_GM_inequality_example_l919_91993


namespace NUMINAMATH_GPT_max_arithmetic_sequence_terms_l919_91908

theorem max_arithmetic_sequence_terms
  (n : ℕ)
  (a1 : ℝ)
  (d : ℝ) 
  (sum_sq_term_cond : (a1 + (n - 1) * d / 2)^2 + (n - 1) * (a1 + d * (n - 1) / 2) ≤ 100)
  (common_diff : d = 4)
  : n ≤ 8 := 
sorry

end NUMINAMATH_GPT_max_arithmetic_sequence_terms_l919_91908


namespace NUMINAMATH_GPT_central_angle_of_sector_l919_91954

variable (r θ : ℝ)
variable (r_pos : 0 < r) (θ_pos : 0 < θ)

def perimeter_eq : Prop := 2 * r + r * θ = 5
def area_eq : Prop := (1 / 2) * r^2 * θ = 1

theorem central_angle_of_sector :
  perimeter_eq r θ ∧ area_eq r θ → θ = 1 / 2 :=
sorry

end NUMINAMATH_GPT_central_angle_of_sector_l919_91954


namespace NUMINAMATH_GPT_identity_proof_l919_91948

theorem identity_proof (n : ℝ) (h1 : n^2 ≥ 4) (h2 : n ≠ 0) :
    (n^3 - 3*n + (n^2 - 1)*Real.sqrt (n^2 - 4) - 2) / 
    (n^3 - 3*n + (n^2 - 1)*Real.sqrt (n^2 - 4) + 2)
    = ((n + 1) * Real.sqrt (n - 2)) / ((n - 1) * Real.sqrt (n + 2)) := by
  sorry

end NUMINAMATH_GPT_identity_proof_l919_91948


namespace NUMINAMATH_GPT_find_missing_number_l919_91979

theorem find_missing_number (x : ℕ) (h : x * 240 = 173 * 240) : x = 173 :=
sorry

end NUMINAMATH_GPT_find_missing_number_l919_91979


namespace NUMINAMATH_GPT_larger_screen_diagonal_length_l919_91988

theorem larger_screen_diagonal_length :
  (∃ d : ℝ, (∀ a : ℝ, a = 16 → d^2 = 2 * (a^2 + 34)) ∧ d = Real.sqrt 580) :=
by
  sorry

end NUMINAMATH_GPT_larger_screen_diagonal_length_l919_91988


namespace NUMINAMATH_GPT_find_x_l919_91910

theorem find_x (x : ℝ) (h : ⌊x⌋ + x = 15/4) : x = 7/4 :=
sorry

end NUMINAMATH_GPT_find_x_l919_91910


namespace NUMINAMATH_GPT_fraction_power_multiply_l919_91933

theorem fraction_power_multiply :
  ((1 : ℚ) / 3)^4 * ((1 : ℚ) / 5) = (1 / 405 : ℚ) :=
by sorry

end NUMINAMATH_GPT_fraction_power_multiply_l919_91933


namespace NUMINAMATH_GPT_best_fitting_model_l919_91929

theorem best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.98)
  (h2 : R2_2 = 0.80)
  (h3 : R2_3 = 0.50)
  (h4 : R2_4 = 0.25) :
  R2_1 = 0.98 ∧ R2_1 > R2_2 ∧ R2_1 > R2_3 ∧ R2_1 > R2_4 :=
by { sorry }

end NUMINAMATH_GPT_best_fitting_model_l919_91929


namespace NUMINAMATH_GPT_downstream_distance_l919_91942

theorem downstream_distance
  (time_downstream : ℝ) (time_upstream : ℝ)
  (distance_upstream : ℝ) (speed_still_water : ℝ)
  (h1 : time_downstream = 3) (h2 : time_upstream = 3)
  (h3 : distance_upstream = 15) (h4 : speed_still_water = 10) :
  ∃ d : ℝ, d = 45 :=
by
  sorry

end NUMINAMATH_GPT_downstream_distance_l919_91942


namespace NUMINAMATH_GPT_remainder_of_2n_div_7_l919_91915

theorem remainder_of_2n_div_7 (n : ℤ) (k : ℤ) (h : n = 7 * k + 2) : (2 * n) % 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_2n_div_7_l919_91915


namespace NUMINAMATH_GPT_chocolate_candies_total_cost_l919_91937

-- Condition 1: A box of 30 chocolate candies costs $7.50
def box_cost : ℝ := 7.50
def candies_per_box : ℕ := 30

-- Condition 2: The local sales tax rate is 10%
def sales_tax_rate : ℝ := 0.10

-- Total number of candies to be bought
def total_candy_count : ℕ := 540

-- Calculate the number of boxes needed
def number_of_boxes (total_candies : ℕ) (candies_per_box : ℕ) : ℕ :=
  total_candies / candies_per_box

-- Calculate the cost without tax
def cost_without_tax (num_boxes : ℕ) (cost_per_box : ℝ) : ℝ :=
  num_boxes * cost_per_box

-- Calculate the total cost including tax
def total_cost_with_tax (cost : ℝ) (tax_rate : ℝ) : ℝ :=
  cost * (1 + tax_rate)

-- The main statement
theorem chocolate_candies_total_cost :
  total_cost_with_tax 
    (cost_without_tax (number_of_boxes total_candy_count candies_per_box) box_cost)
    sales_tax_rate = 148.50 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_candies_total_cost_l919_91937


namespace NUMINAMATH_GPT_lowest_score_is_C_l919_91969

variable (Score : Type) [LinearOrder Score]
variable (A B C : Score)

-- Translate conditions into Lean
variable (h1 : B ≠ max A (max B C) → A = min A (min B C))
variable (h2 : C ≠ min A (min B C) → A = max A (max B C))

-- Define the proof goal
theorem lowest_score_is_C : min A (min B C) =C :=
by
  sorry

end NUMINAMATH_GPT_lowest_score_is_C_l919_91969


namespace NUMINAMATH_GPT_intersection_domains_l919_91952

def domain_f := {x : ℝ | x < 1}
def domain_g := {x : ℝ | x ≠ 0}

theorem intersection_domains :
  {x : ℝ | x < 1} ∩ {x : ℝ | x ≠ 0} = {x : ℝ | x < 1 ∧ x ≠ 0} :=
by 
  sorry

end NUMINAMATH_GPT_intersection_domains_l919_91952


namespace NUMINAMATH_GPT_existence_of_f_and_g_l919_91971

noncomputable def Set_n (n : ℕ) : Set ℕ := { x | x ≥ 1 ∧ x ≤ n }

theorem existence_of_f_and_g (n : ℕ) (f g : ℕ → ℕ) :
  (∀ x ∈ Set_n n, (f (g x) = x ∨ g (f x) = x) ∧ ¬(f (g x) = x ∧ g (f x) = x)) ↔ Even n := sorry

end NUMINAMATH_GPT_existence_of_f_and_g_l919_91971


namespace NUMINAMATH_GPT_calculation_l919_91990

theorem calculation : 2005^2 - 2003 * 2007 = 4 :=
by
  have h1 : 2003 = 2005 - 2 := by rfl
  have h2 : 2007 = 2005 + 2 := by rfl
  sorry

end NUMINAMATH_GPT_calculation_l919_91990


namespace NUMINAMATH_GPT_exponent_property_l919_91997

theorem exponent_property (a : ℝ) : a^7 = a^3 * a^4 :=
by
  -- The proof statement follows from the properties of exponents:
  -- a^m * a^n = a^(m + n)
  -- Therefore, a^3 * a^4 = a^(3 + 4) = a^7.
  sorry

end NUMINAMATH_GPT_exponent_property_l919_91997


namespace NUMINAMATH_GPT_divisibility_by_1897_l919_91992

theorem divisibility_by_1897 (n : ℕ) : 1897 ∣ (2903 ^ n - 803 ^ n - 464 ^ n + 261 ^ n) :=
sorry

end NUMINAMATH_GPT_divisibility_by_1897_l919_91992


namespace NUMINAMATH_GPT_syllogism_sequence_correct_l919_91917

-- Definitions based on conditions
def square_interior_angles_equal : Prop := ∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D
def rectangle_interior_angles_equal : Prop := ∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D
def square_is_rectangle : Prop := ∀ (S : Type), S = S

-- Final Goal
theorem syllogism_sequence_correct : (rectangle_interior_angles_equal → square_is_rectangle → square_interior_angles_equal) :=
by
  sorry

end NUMINAMATH_GPT_syllogism_sequence_correct_l919_91917


namespace NUMINAMATH_GPT_stacy_paper_shortage_l919_91957

theorem stacy_paper_shortage:
  let bought_sheets : ℕ := 240 + 320
  let daily_mwf : ℕ := 60
  let daily_tt : ℕ := 100
  -- Calculate sheets used in a week
  let used_one_week : ℕ := (daily_mwf * 3) + (daily_tt * 2)
  -- Calculate sheets used in two weeks
  let used_two_weeks : ℕ := used_one_week * 2
  -- Remaining sheets at the end of two weeks
  let remaining_sheets : Int := bought_sheets - used_two_weeks
  remaining_sheets = -200 :=
by sorry

end NUMINAMATH_GPT_stacy_paper_shortage_l919_91957


namespace NUMINAMATH_GPT_integer_part_mod_8_l919_91999

theorem integer_part_mod_8 (n : ℕ) (h : n ≥ 2009) :
  ∃ x : ℝ, x = (3 + Real.sqrt 8)^(2 * n) ∧ Int.floor (x) % 8 = 1 := 
sorry

end NUMINAMATH_GPT_integer_part_mod_8_l919_91999


namespace NUMINAMATH_GPT_line_equation_l919_91912

theorem line_equation (l : ℝ → ℝ → Prop) (a b : ℝ) 
  (h1 : ∀ x y, l x y ↔ y = - (b / a) * x + b) 
  (h2 : l 2 1) 
  (h3 : a + b = 0) : 
  l x y ↔ y = x - 1 ∨ y = x / 2 := 
by
  sorry

end NUMINAMATH_GPT_line_equation_l919_91912


namespace NUMINAMATH_GPT_age_of_15th_student_l919_91900

noncomputable def average_age_15_students := 15
noncomputable def average_age_7_students_1 := 14
noncomputable def average_age_7_students_2 := 16
noncomputable def total_students := 15
noncomputable def group_students := 7

theorem age_of_15th_student :
  let total_age_15_students := total_students * average_age_15_students
  let total_age_7_students_1 := group_students * average_age_7_students_1
  let total_age_7_students_2 := group_students * average_age_7_students_2
  let total_age_14_students := total_age_7_students_1 + total_age_7_students_2
  let age_15th_student := total_age_15_students - total_age_14_students
  age_15th_student = 15 :=
by
  sorry

end NUMINAMATH_GPT_age_of_15th_student_l919_91900


namespace NUMINAMATH_GPT_work_days_for_c_l919_91996

theorem work_days_for_c (A B C : ℝ)
  (h1 : A + B = 1 / 15)
  (h2 : A + B + C = 1 / 11) :
  1 / C = 41.25 :=
by
  sorry

end NUMINAMATH_GPT_work_days_for_c_l919_91996


namespace NUMINAMATH_GPT_total_apples_correctness_l919_91932

-- Define the number of apples each man bought
def applesMen := 30

-- Define the number of apples each woman bought
def applesWomen := applesMen + 20

-- Define the total number of apples bought by the two men
def totalApplesMen := 2 * applesMen

-- Define the total number of apples bought by the three women
def totalApplesWomen := 3 * applesWomen

-- Define the total number of apples bought by the two men and three women
def totalApples := totalApplesMen + totalApplesWomen

-- Prove that the total number of apples bought by two men and three women is 210
theorem total_apples_correctness : totalApples = 210 := by
  sorry

end NUMINAMATH_GPT_total_apples_correctness_l919_91932


namespace NUMINAMATH_GPT_cousin_cards_probability_l919_91930

variable {Isabella_cards : ℕ}
variable {Evan_cards : ℕ}
variable {total_cards : ℕ}

theorem cousin_cards_probability 
  (h1 : Isabella_cards = 8)
  (h2 : Evan_cards = 2)
  (h3 : total_cards = 10) :
  (8 / 10 * 2 / 9) + (2 / 10 * 8 / 9) = 16 / 45 :=
by
  sorry

end NUMINAMATH_GPT_cousin_cards_probability_l919_91930


namespace NUMINAMATH_GPT_sum_of_first_15_terms_l919_91943

open scoped BigOperators

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + n * d

-- Define the condition given in the problem
def condition (a d : ℤ) : Prop :=
  3 * (arithmetic_sequence a d 2 + arithmetic_sequence a d 4) + 
  2 * (arithmetic_sequence a d 6 + arithmetic_sequence a d 11 + arithmetic_sequence a d 16) = 180

-- Prove that the sum of the first 15 terms is 225
theorem sum_of_first_15_terms (a d : ℤ) (h : condition a d) :
  ∑ i in Finset.range 15, arithmetic_sequence a d i = 225 :=
  sorry

end NUMINAMATH_GPT_sum_of_first_15_terms_l919_91943


namespace NUMINAMATH_GPT_determine_value_of_e_l919_91940

theorem determine_value_of_e {a b c d e : ℝ} (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) 
    (h5 : a + b = 32) (h6 : a + c = 36) (h7 : b + c = 37 ∨ a + d = 37) 
    (h8 : c + e = 48) (h9 : d + e = 51) : e = 27.5 :=
sorry

end NUMINAMATH_GPT_determine_value_of_e_l919_91940


namespace NUMINAMATH_GPT_intervals_of_decrease_l919_91987

open Real

noncomputable def func (x : ℝ) : ℝ :=
  cos (2 * x) + 2 * sin x

theorem intervals_of_decrease :
  {x | deriv func x < 0 ∧ 0 < x ∧ x < 2 * π} =
  {x | (π / 6 < x ∧ x < π / 2) ∨ (5 * π / 6 < x ∧ x < 3 * π / 2)} :=
by
  sorry

end NUMINAMATH_GPT_intervals_of_decrease_l919_91987


namespace NUMINAMATH_GPT_triangle_with_altitudes_is_obtuse_l919_91922

theorem triangle_with_altitudes_is_obtuse (h1 h2 h3 : ℝ) (h_pos1 : h1 > 0) (h_pos2 : h2 > 0) (h_pos3 : h3 > 0)
    (h_triangle_ineq1 : 1 / h2 + 1 / h3 > 1 / h1)
    (h_triangle_ineq2 : 1 / h1 + 1 / h3 > 1 / h2)
    (h_triangle_ineq3 : 1 / h1 + 1 / h2 > 1 / h3) : 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a * h1 = b * h2 ∧ b * h2 = c * h3 ∧
    (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2) :=
sorry

end NUMINAMATH_GPT_triangle_with_altitudes_is_obtuse_l919_91922


namespace NUMINAMATH_GPT_James_weight_after_gain_l919_91982

theorem James_weight_after_gain 
    (initial_weight : ℕ)
    (muscle_gain_perc : ℕ)
    (fat_gain_fraction : ℚ)
    (weight_after_gain : ℕ) :
    initial_weight = 120 →
    muscle_gain_perc = 20 →
    fat_gain_fraction = 1/4 →
    weight_after_gain = 150 :=
by
  intros
  sorry

end NUMINAMATH_GPT_James_weight_after_gain_l919_91982


namespace NUMINAMATH_GPT_last_two_digits_10_93_10_31_plus_3_eq_08_l919_91935

def last_two_digits_fraction_floor (n m d : ℕ) : ℕ :=
  let x := 10^n
  let y := 10^m + d
  (x / y) % 100

theorem last_two_digits_10_93_10_31_plus_3_eq_08 :
  last_two_digits_fraction_floor 93 31 3 = 08 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_10_93_10_31_plus_3_eq_08_l919_91935


namespace NUMINAMATH_GPT_arithmetic_expressions_correctness_l919_91945

theorem arithmetic_expressions_correctness :
  ((∀ (a b c : ℚ), (a + b) + c = a + (b + c)) ∧
   (∃ (a b c : ℚ), (a - b) - c ≠ a - (b - c)) ∧
   (∀ (a b c : ℚ), (a * b) * c = a * (b * c)) ∧
   (∃ (a b c : ℚ), a / b / c ≠ a / (b / c))) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expressions_correctness_l919_91945


namespace NUMINAMATH_GPT_inequality_has_solutions_iff_a_ge_4_l919_91986

theorem inequality_has_solutions_iff_a_ge_4 (a x : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) ↔ a ≥ 4 :=
sorry

end NUMINAMATH_GPT_inequality_has_solutions_iff_a_ge_4_l919_91986


namespace NUMINAMATH_GPT_exists_n_for_A_of_non_perfect_square_l919_91934

theorem exists_n_for_A_of_non_perfect_square (A : ℕ) (h : ∀ k : ℕ, k^2 ≠ A) :
  ∃ n : ℕ, A = ⌊ n + Real.sqrt n + 1/2 ⌋ :=
sorry

end NUMINAMATH_GPT_exists_n_for_A_of_non_perfect_square_l919_91934


namespace NUMINAMATH_GPT_sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog_l919_91980

theorem sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog
  (a r : ℝ)
  (volume_cond : a^3 * r^3 = 288)
  (surface_area_cond : 2 * (a^2 * r^4 + a^2 * r^2 + a^2 * r) = 288)
  (geom_prog : True) :
  4 * (a * r^2 + a * r + a) = 92 := 
sorry

end NUMINAMATH_GPT_sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog_l919_91980


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l919_91924

-- Define sets M and N
def M := {x : ℝ | (x + 2) * (x - 1) < 0}
def N := {x : ℝ | x + 1 < 0}

-- State the theorem for the intersection M ∩ N
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < -1} :=
sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l919_91924


namespace NUMINAMATH_GPT_point_A_outside_circle_iff_l919_91903

-- Define the conditions
def B : ℝ := 16
def radius : ℝ := 4
def A_position (t : ℝ) : ℝ := 2 * t

-- Define the theorem
theorem point_A_outside_circle_iff (t : ℝ) : (A_position t < B - radius) ∨ (A_position t > B + radius) ↔ (t < 6 ∨ t > 10) :=
by
  sorry

end NUMINAMATH_GPT_point_A_outside_circle_iff_l919_91903


namespace NUMINAMATH_GPT_systematic_sampling_employee_l919_91923

theorem systematic_sampling_employee {x : ℕ} (h1 : 1 ≤ 6 ∧ 6 ≤ 52) (h2 : 1 ≤ 32 ∧ 32 ≤ 52) (h3 : 1 ≤ 45 ∧ 45 ≤ 52) (h4 : 6 + 45 = x + 32) : x = 19 :=
  by
    sorry

end NUMINAMATH_GPT_systematic_sampling_employee_l919_91923


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l919_91949

theorem solve_equation_1 :
  ∀ x : ℝ, 2 * x^2 - 4 * x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  intro x
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, x^2 - 6 * x - 6 = 0 ↔ (x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l919_91949


namespace NUMINAMATH_GPT_jake_weight_l919_91902

theorem jake_weight {J S : ℝ} (h1 : J - 20 = 2 * S) (h2 : J + S = 224) : J = 156 :=
by
  sorry

end NUMINAMATH_GPT_jake_weight_l919_91902


namespace NUMINAMATH_GPT_second_multiple_of_three_l919_91944

theorem second_multiple_of_three (n : ℕ) (h : 3 * (n - 1) + 3 * (n + 1) = 150) : 3 * n = 75 :=
sorry

end NUMINAMATH_GPT_second_multiple_of_three_l919_91944


namespace NUMINAMATH_GPT_train_speed_l919_91959

theorem train_speed (length : ℝ) (time : ℝ) (length_eq : length = 400) (time_eq : time = 16) :
  (length / time) * (3600 / 1000) = 90 :=
by 
  rw [length_eq, time_eq]
  sorry

end NUMINAMATH_GPT_train_speed_l919_91959


namespace NUMINAMATH_GPT_range_of_x_l919_91970

theorem range_of_x (x : ℤ) : x^2 < 3 * x → x = 1 ∨ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l919_91970


namespace NUMINAMATH_GPT_commutative_op_l919_91931

variable {S : Type} (op : S → S → S)

-- Conditions
axiom cond1 : ∀ (a b : S), op a (op a b) = b
axiom cond2 : ∀ (a b : S), op (op a b) b = a

-- Proof problem statement
theorem commutative_op : ∀ (a b : S), op a b = op b a :=
by
  intros a b
  sorry

end NUMINAMATH_GPT_commutative_op_l919_91931


namespace NUMINAMATH_GPT_min_value_expression_l919_91911

theorem min_value_expression (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1/2) :
  a^2 + 4 * a * b + 12 * b^2 + 8 * b * c + 3 * c^2 ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l919_91911


namespace NUMINAMATH_GPT_laundry_lcm_l919_91955

theorem laundry_lcm :
  Nat.lcm (Nat.lcm 6 9) (Nat.lcm 12 15) = 180 :=
by
  sorry

end NUMINAMATH_GPT_laundry_lcm_l919_91955


namespace NUMINAMATH_GPT_inequality_for_positive_integers_l919_91951

theorem inequality_for_positive_integers (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b + b * c + a * c ≤ 3 * a * b * c :=
sorry

end NUMINAMATH_GPT_inequality_for_positive_integers_l919_91951


namespace NUMINAMATH_GPT_exponent_of_four_l919_91938

theorem exponent_of_four (n : ℕ) (k : ℕ) (h : n = 21) 
  (eq : (↑(4 : ℕ) * 2 ^ (2 * n) = 4 ^ k)) : k = 22 :=
by
  sorry

end NUMINAMATH_GPT_exponent_of_four_l919_91938


namespace NUMINAMATH_GPT_max_value_f_l919_91921

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * (4 : ℝ) * x + 2

theorem max_value_f :
  ∃ x : ℝ, -f x = -18 ∧ (∀ y : ℝ, f y ≤ f x) :=
by
  sorry

end NUMINAMATH_GPT_max_value_f_l919_91921


namespace NUMINAMATH_GPT_initial_pencils_l919_91981

theorem initial_pencils (P : ℕ) (h1 : 84 = P - (P - 15) / 4 + 16 - 12 + 23) : P = 71 :=
by
  sorry

end NUMINAMATH_GPT_initial_pencils_l919_91981


namespace NUMINAMATH_GPT_general_formula_arithmetic_sum_of_geometric_terms_l919_91964

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 2 = 2 ∧ a 5 = 8

noncomputable def geometric_sequence (b : ℕ → ℝ) (a : ℕ → ℤ) : Prop :=
  b 1 = 1 ∧ b 2 + b 3 = a 4

noncomputable def sum_of_terms (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, T n = (2:ℝ)^n - 1

theorem general_formula_arithmetic (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  ∀ n, a n = 2 * n - 2 :=
sorry

theorem sum_of_geometric_terms (a : ℕ → ℤ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h : arithmetic_sequence a) (h2 : geometric_sequence b a) :
  sum_of_terms T b :=
sorry

end NUMINAMATH_GPT_general_formula_arithmetic_sum_of_geometric_terms_l919_91964


namespace NUMINAMATH_GPT_leo_score_l919_91956

-- Definitions for the conditions
def caroline_score : ℕ := 13
def anthony_score : ℕ := 19
def winning_score : ℕ := 21

-- Lean statement for the proof problem
theorem leo_score : ∃ (leo_score : ℕ), leo_score = winning_score := by
  have h_caroline := caroline_score
  have h_anthony := anthony_score
  have h_winning := winning_score
  use 21
  sorry

end NUMINAMATH_GPT_leo_score_l919_91956


namespace NUMINAMATH_GPT_probability_not_exceeding_40_l919_91998

variable (P : ℝ → Prop)

def less_than_30_grams : Prop := P 0.3
def between_30_and_40_grams : Prop := P 0.5

theorem probability_not_exceeding_40 (h1 : less_than_30_grams P) (h2 : between_30_and_40_grams P) : P 0.8 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_exceeding_40_l919_91998


namespace NUMINAMATH_GPT_identify_clothes_l919_91977

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end NUMINAMATH_GPT_identify_clothes_l919_91977


namespace NUMINAMATH_GPT_inequality_system_solution_l919_91947

theorem inequality_system_solution (x : ℤ) :
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1 + 2 * x) / 3 ≥ x - 1) ↔ (x = 3 ∨ x = 4) := sorry

end NUMINAMATH_GPT_inequality_system_solution_l919_91947


namespace NUMINAMATH_GPT_no_lunch_students_l919_91914

variable (total_students : ℕ) (cafeteria_eaters : ℕ) (lunch_bringers : ℕ)

theorem no_lunch_students : 
  total_students = 60 →
  cafeteria_eaters = 10 →
  lunch_bringers = 3 * cafeteria_eaters →
  total_students - (cafeteria_eaters + lunch_bringers) = 20 :=
by
  sorry

end NUMINAMATH_GPT_no_lunch_students_l919_91914


namespace NUMINAMATH_GPT_greatest_visible_unit_cubes_from_one_point_12_l919_91958

def num_unit_cubes (n : ℕ) : ℕ := n * n * n

def face_count (n : ℕ) : ℕ := n * n

def edge_count (n : ℕ) : ℕ := n

def visible_unit_cubes_from_one_point (n : ℕ) : ℕ :=
  let faces := 3 * face_count n
  let edges := 3 * (edge_count n - 1)
  let corner := 1
  faces - edges + corner

theorem greatest_visible_unit_cubes_from_one_point_12 :
  visible_unit_cubes_from_one_point 12 = 400 :=
  by
  sorry

end NUMINAMATH_GPT_greatest_visible_unit_cubes_from_one_point_12_l919_91958


namespace NUMINAMATH_GPT_range_of_k_l919_91978

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem range_of_k :
  (∀ x : ℝ, 2 < x → f x > k) →
  k ≤ -Real.exp 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l919_91978


namespace NUMINAMATH_GPT_num_packages_l919_91913

theorem num_packages (total_shirts : ℕ) (shirts_per_package : ℕ) (h1 : total_shirts = 51) (h2 : shirts_per_package = 3) : total_shirts / shirts_per_package = 17 := by
  sorry

end NUMINAMATH_GPT_num_packages_l919_91913


namespace NUMINAMATH_GPT_solve_inequality_l919_91989

noncomputable def within_interval (x : ℝ) : Prop :=
  x > -3 ∧ x < 5

theorem solve_inequality (x : ℝ) :
  (x^3 - 125) / (x + 3) < 0 ↔ within_interval x :=
sorry

end NUMINAMATH_GPT_solve_inequality_l919_91989


namespace NUMINAMATH_GPT_age_difference_l919_91973

theorem age_difference (C D m : ℕ) 
  (h1 : C = D + m)
  (h2 : C - 1 = 3 * (D - 1)) 
  (h3 : C * D = 72) : 
  m = 9 :=
sorry

end NUMINAMATH_GPT_age_difference_l919_91973


namespace NUMINAMATH_GPT_petya_can_force_difference_2014_l919_91916

theorem petya_can_force_difference_2014 :
  ∀ (p q r : ℚ), ∃ (a b c : ℚ), ∀ (x : ℝ), (x^3 + a * x^2 + b * x + c = 0) → 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2014 :=
by sorry

end NUMINAMATH_GPT_petya_can_force_difference_2014_l919_91916


namespace NUMINAMATH_GPT_player_b_wins_l919_91967

theorem player_b_wins : 
  ∃ B_strategy : (ℕ → ℕ → Prop), (∀ A_turn : ℕ → Prop, 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2019 → (A_turn i ↔ ¬ A_turn (i + 1))) → 
  ((B_strategy 1 2019) ∨ ∃ k : ℕ, 1 ≤ k ∧ k ≤ 2019 ∧ B_strategy k (k + 1) ∧ ¬ A_turn k)) :=
sorry

end NUMINAMATH_GPT_player_b_wins_l919_91967


namespace NUMINAMATH_GPT_solve_for_m_l919_91907

theorem solve_for_m (m : ℝ) : 
  (∀ x : ℝ, (x = 2) → ((m - 2) * x = 5 * (x + 1))) → (m = 19 / 2) :=
by
  intro h
  have h1 := h 2
  sorry  -- proof can be filled in later

end NUMINAMATH_GPT_solve_for_m_l919_91907


namespace NUMINAMATH_GPT_proof_problem_exists_R1_R2_l919_91962

def problem (R1 R2 : ℕ) : Prop :=
  let F1_R1 := (4 * R1 + 5) / (R1^2 - 1)
  let F2_R1 := (5 * R1 + 4) / (R1^2 - 1)
  let F1_R2 := (3 * R2 + 2) / (R2^2 - 1)
  let F2_R2 := (2 * R2 + 3) / (R2^2 - 1)
  F1_R1 = F1_R2 ∧ F2_R1 = F2_R2 ∧ R1 + R2 = 14

theorem proof_problem_exists_R1_R2 : ∃ (R1 R2 : ℕ), problem R1 R2 :=
sorry

end NUMINAMATH_GPT_proof_problem_exists_R1_R2_l919_91962


namespace NUMINAMATH_GPT_number_of_girls_l919_91936

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end NUMINAMATH_GPT_number_of_girls_l919_91936


namespace NUMINAMATH_GPT_correct_result_l919_91961

-- Define the conditions
variables (x : ℤ)
axiom condition1 : (x - 27 + 19 = 84)

-- Define the goal
theorem correct_result : x - 19 + 27 = 100 :=
  sorry

end NUMINAMATH_GPT_correct_result_l919_91961


namespace NUMINAMATH_GPT_minimum_value_of_f_on_interval_l919_91950

noncomputable def f (a x : ℝ) := Real.log x + a * x

theorem minimum_value_of_f_on_interval (a : ℝ) (h : a < 0) :
  ( ( -Real.log 2 ≤ a ∧ a < 0 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≥ a ) ∧
    ( a < -Real.log 2 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≥ (Real.log 2 + 2 * a) )
  ) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_on_interval_l919_91950


namespace NUMINAMATH_GPT_fraction_problem_l919_91972

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20) 
  (h2 : p / n = 5) 
  (h3 : p / q = 1 / 15) : 
  m / q = 4 / 15 :=
sorry

end NUMINAMATH_GPT_fraction_problem_l919_91972
