import Mathlib

namespace NUMINAMATH_GPT_heather_blocks_l1654_165430

theorem heather_blocks (x : ℝ) (h1 : x + 41 = 127) : x = 86 := by
  sorry

end NUMINAMATH_GPT_heather_blocks_l1654_165430


namespace NUMINAMATH_GPT_general_solution_linear_diophantine_l1654_165466

theorem general_solution_linear_diophantine (a b c : ℤ) (h_coprime : Int.gcd a b = 1)
    (x1 y1 : ℤ) (h_particular_solution : a * x1 + b * y1 = c) :
    ∃ (t : ℤ), (∃ (x y : ℤ), x = x1 + b * t ∧ y = y1 - a * t ∧ a * x + b * y = c) ∧
               (∃ (x' y' : ℤ), x' = x1 - b * t ∧ y' = y1 + a * t ∧ a * x' + b * y' = c) :=
by
  sorry

end NUMINAMATH_GPT_general_solution_linear_diophantine_l1654_165466


namespace NUMINAMATH_GPT_interest_calculation_correct_l1654_165404

-- Define the principal amounts and their respective interest rates
def principal1 : ℝ := 3000
def rate1 : ℝ := 0.08
def principal2 : ℝ := 8000 - principal1
def rate2 : ℝ := 0.05

-- Calculate interest for one year
def interest1 : ℝ := principal1 * rate1 * 1
def interest2 : ℝ := principal2 * rate2 * 1

-- Define the total interest
def total_interest : ℝ := interest1 + interest2

-- Prove that the total interest calculated is $490
theorem interest_calculation_correct : total_interest = 490 := by
  sorry

end NUMINAMATH_GPT_interest_calculation_correct_l1654_165404


namespace NUMINAMATH_GPT_find_a_l1654_165468

-- Define the polynomial f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 - 0.8

-- Define the intermediate values v_0, v_1, and v_2 using Horner's method
def v_0 (a : ℝ) : ℝ := a
def v_1 (a : ℝ) (x : ℝ) : ℝ := v_0 a * x + 2
def v_2 (a : ℝ) (x : ℝ) : ℝ := v_1 a x * x + 3.5 * x - 2.6 * x + 13.5

-- The condition for v_2 when x = 5
axiom v2_value (a : ℝ) : v_2 a 5 = 123.5

-- Prove that a = 4
theorem find_a : ∃ a : ℝ, v_2 a 5 = 123.5 ∧ a = 4 := by
  sorry

end NUMINAMATH_GPT_find_a_l1654_165468


namespace NUMINAMATH_GPT_geo_seq_a3_equals_one_l1654_165408

theorem geo_seq_a3_equals_one (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_T5 : a 1 * a 2 * a 3 * a 4 * a 5 = 1) : a 3 = 1 :=
sorry

end NUMINAMATH_GPT_geo_seq_a3_equals_one_l1654_165408


namespace NUMINAMATH_GPT_cube_plus_eleven_mul_divisible_by_six_l1654_165482

theorem cube_plus_eleven_mul_divisible_by_six (a : ℤ) : 6 ∣ (a^3 + 11 * a) := 
by sorry

end NUMINAMATH_GPT_cube_plus_eleven_mul_divisible_by_six_l1654_165482


namespace NUMINAMATH_GPT_fraction_value_l1654_165448

theorem fraction_value
  (a b c d : ℚ)
  (h1 : a / b = 1 / 4)
  (h2 : c / d = 1 / 4)
  (h3 : b ≠ 0)
  (h4 : d ≠ 0)
  (h5 : b + d ≠ 0) :
  (a + 2 * c) / (2 * b + 4 * d) = 1 / 8 :=
sorry

end NUMINAMATH_GPT_fraction_value_l1654_165448


namespace NUMINAMATH_GPT_bucket_capacity_l1654_165411

theorem bucket_capacity (x : ℝ) (h1 : 24 * x = 36 * 9) : x = 13.5 :=
by 
  sorry

end NUMINAMATH_GPT_bucket_capacity_l1654_165411


namespace NUMINAMATH_GPT_find_total_quantities_l1654_165491

theorem find_total_quantities (n S S_3 S_2 : ℕ) (h1 : S = 8 * n) (h2 : S_3 = 4 * 3) (h3 : S_2 = 14 * 2) (h4 : S = S_3 + S_2) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_total_quantities_l1654_165491


namespace NUMINAMATH_GPT_integer_roots_k_values_y1_y2_squared_sum_k_0_y1_y2_squared_sum_k_neg1_l1654_165479

theorem integer_roots_k_values (k : ℤ) :
  (∀ x : ℤ, k * x ^ 2 + (2 * k - 1) * x + k - 1 = 0) →
  k = 0 ∨ k = -1 :=
sorry

theorem y1_y2_squared_sum_k_0 (m y1 y2: ℝ) :
  (m > -2) →
  (k = 0) →
  ((k - 1) * y1 ^ 2 - 3 * y1 + m = 0) →
  ((k - 1) * y2 ^ 2 - 3 * y2 + m = 0) →
  y1^2 + y2^2 = 9 + 2 * m :=
sorry

theorem y1_y2_squared_sum_k_neg1 (m y1 y2: ℝ) :
  (m > -2) →
  (k = -1) →
  ((k - 1) * y1 ^ 2 - 3 * y1 + m = 0) →
  ((k - 1) * y2 ^ 2 - 3 * y2 + m = 0) →
  y1^2 + y2^2 = 9 / 4 + m :=
sorry

end NUMINAMATH_GPT_integer_roots_k_values_y1_y2_squared_sum_k_0_y1_y2_squared_sum_k_neg1_l1654_165479


namespace NUMINAMATH_GPT_horizontal_length_of_rectangle_l1654_165443

theorem horizontal_length_of_rectangle
  (P : ℕ)
  (h v : ℕ)
  (hP : P = 54)
  (hv : v = h - 3) :
  2*h + 2*v = 54 → h = 15 :=
by sorry

end NUMINAMATH_GPT_horizontal_length_of_rectangle_l1654_165443


namespace NUMINAMATH_GPT_basketballs_count_l1654_165416

theorem basketballs_count (x : ℕ) : 
  let num_volleyballs := x
  let num_basketballs := 2 * x
  let num_soccer_balls := x - 8
  num_volleyballs + num_basketballs + num_soccer_balls = 100 →
  num_basketballs = 54 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_basketballs_count_l1654_165416


namespace NUMINAMATH_GPT_complement_P_l1654_165473

def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_P :
  (U \ P) = Set.Iio (-1) ∪ Set.Ioi (1) :=
by
  sorry

end NUMINAMATH_GPT_complement_P_l1654_165473


namespace NUMINAMATH_GPT_price_of_third_variety_l1654_165484

-- Define the given conditions
def price1 : ℝ := 126
def price2 : ℝ := 135
def average_price : ℝ := 153
def ratio1 : ℝ := 1
def ratio2 : ℝ := 1
def ratio3 : ℝ := 2

-- Define the total ratio
def total_ratio : ℝ := ratio1 + ratio2 + ratio3

-- Define the equation based on the given conditions
def weighted_avg_price (P : ℝ) : Prop :=
  (ratio1 * price1 + ratio2 * price2 + ratio3 * P) / total_ratio = average_price

-- Statement of the proof
theorem price_of_third_variety :
  ∃ P : ℝ, weighted_avg_price P ∧ P = 175.5 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_price_of_third_variety_l1654_165484


namespace NUMINAMATH_GPT_unique_xy_exists_l1654_165452

theorem unique_xy_exists (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y) ^ 2 + 3 * x + y) / 2 := 
sorry

end NUMINAMATH_GPT_unique_xy_exists_l1654_165452


namespace NUMINAMATH_GPT_train_length_l1654_165494

theorem train_length
  (speed_kmph : ℕ) (time_s : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_s = 12) :
  speed_kmph * (1000 / 3600 : ℕ) * time_s = 240 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1654_165494


namespace NUMINAMATH_GPT_pants_cost_correct_l1654_165478

-- Define the conditions as variables
def initial_money : ℕ := 71
def shirt_cost : ℕ := 5
def num_shirts : ℕ := 5
def remaining_money : ℕ := 20

-- Define intermediates necessary to show the connection between conditions and the question
def money_spent_on_shirts : ℕ := num_shirts * shirt_cost
def money_left_after_shirts : ℕ := initial_money - money_spent_on_shirts
def pants_cost : ℕ := money_left_after_shirts - remaining_money

-- The main theorem to prove the question is equal to the correct answer
theorem pants_cost_correct : pants_cost = 26 :=
by
  sorry

end NUMINAMATH_GPT_pants_cost_correct_l1654_165478


namespace NUMINAMATH_GPT_sqrt_two_between_one_and_two_l1654_165414

theorem sqrt_two_between_one_and_two : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2 := 
by
  -- sorry placeholder
  sorry

end NUMINAMATH_GPT_sqrt_two_between_one_and_two_l1654_165414


namespace NUMINAMATH_GPT_find_g_of_2_l1654_165461

-- Define the assumptions
variables (g : ℝ → ℝ)
axiom condition : ∀ x : ℝ, x ≠ 0 → 5 * g (1 / x) + (3 * g x) / x = Real.sqrt x

-- State the theorem to prove
theorem find_g_of_2 : g 2 = -(Real.sqrt 2) / 16 :=
by
  sorry

end NUMINAMATH_GPT_find_g_of_2_l1654_165461


namespace NUMINAMATH_GPT_parameter_values_for_roots_l1654_165451

theorem parameter_values_for_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 = 5 * x2 ∧ a * x1^2 - (2 * a + 5) * x1 + 10 = 0 ∧ a * x2^2 - (2 * a + 5) * x2 + 10 = 0)
  ↔ (a = 5 / 3 ∨ a = 5) := 
sorry

end NUMINAMATH_GPT_parameter_values_for_roots_l1654_165451


namespace NUMINAMATH_GPT_train_speed_l1654_165453

theorem train_speed
  (length_of_train : ℕ)
  (time_to_cross_bridge : ℕ)
  (length_of_bridge : ℕ)
  (speed_conversion_factor : ℕ)
  (H1 : length_of_train = 120)
  (H2 : time_to_cross_bridge = 30)
  (H3 : length_of_bridge = 255)
  (H4 : speed_conversion_factor = 36) : 
  (length_of_train + length_of_bridge) / (time_to_cross_bridge / speed_conversion_factor) = 45 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1654_165453


namespace NUMINAMATH_GPT_marla_adds_blue_paint_l1654_165401

variable (M B : ℝ)

theorem marla_adds_blue_paint :
  (20 = 0.10 * M) ∧ (B = 0.70 * M) → B = 140 := 
by 
  sorry

end NUMINAMATH_GPT_marla_adds_blue_paint_l1654_165401


namespace NUMINAMATH_GPT_find_a_plus_b_l1654_165410

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)

def parallel_condition (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.2 - a.2 * b.1 = 0)

theorem find_a_plus_b (m : ℝ) (h_parallel: 
  parallel_condition (⟨vector_a.1 + 2 * (vector_b m).1, vector_a.2 + 2 * (vector_b m).2⟩)
                     (⟨2 * vector_a.1 - (vector_b m).1, 2 * vector_a.2 - (vector_b m).2⟩)) :
  vector_a + vector_b (-1/2) = (-3/2, 3) := 
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1654_165410


namespace NUMINAMATH_GPT_smaller_number_between_5_and_8_l1654_165459

theorem smaller_number_between_5_and_8 :
  min 5 8 = 5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_number_between_5_and_8_l1654_165459


namespace NUMINAMATH_GPT_vertical_angles_equal_l1654_165405

-- Define what it means for two angles to be vertical angles.
def are_vertical_angles (α β : ℝ) : Prop :=
  ∃ (γ δ : ℝ), α + γ = 180 ∧ β + δ = 180 ∧ γ = β ∧ δ = α

-- The theorem statement:
theorem vertical_angles_equal (α β : ℝ) : are_vertical_angles α β → α = β := 
  sorry

end NUMINAMATH_GPT_vertical_angles_equal_l1654_165405


namespace NUMINAMATH_GPT_find_three_numbers_l1654_165472

theorem find_three_numbers (x : ℤ) (a b c : ℤ) :
  a + b + c = (x + 1)^2 ∧ a + b = x^2 ∧ b + c = (x - 1)^2 ∧
  a = 80 ∧ b = 320 ∧ c = 41 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_three_numbers_l1654_165472


namespace NUMINAMATH_GPT_values_of_n_eq_100_l1654_165488

theorem values_of_n_eq_100 :
  ∃ (n_count : ℕ), n_count = 100 ∧
    ∀ (a b c : ℕ),
      a + 11 * b + 111 * c = 900 →
      (∀ (a : ℕ), a ≥ 0) →
      (∃ (n : ℕ), n = a + 2 * b + 3 * c ∧ n_count = 100) :=
sorry

end NUMINAMATH_GPT_values_of_n_eq_100_l1654_165488


namespace NUMINAMATH_GPT_train_speed_kmph_l1654_165426

def length_of_train : ℝ := 120
def time_to_cross_bridge : ℝ := 17.39860811135109
def length_of_bridge : ℝ := 170

theorem train_speed_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = 60 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_kmph_l1654_165426


namespace NUMINAMATH_GPT_find_a_plus_b_l1654_165455

theorem find_a_plus_b (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : a^2 - b^4 = 2009) : a + b = 47 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1654_165455


namespace NUMINAMATH_GPT_roller_coaster_costs_7_tickets_l1654_165499

-- Define the number of tickets for the Ferris wheel, log ride, and the initial and additional tickets Zach needs.
def ferris_wheel_tickets : ℕ := 2
def log_ride_tickets : ℕ := 1
def initial_tickets : ℕ := 1
def additional_tickets : ℕ := 9

-- Define the total number of tickets Zach needs.
def total_tickets : ℕ := initial_tickets + additional_tickets

-- Define the number of tickets needed for the Ferris wheel and log ride together.
def combined_tickets_needed : ℕ := ferris_wheel_tickets + log_ride_tickets

-- Define the number of tickets the roller coaster costs.
def roller_coaster_tickets : ℕ := total_tickets - combined_tickets_needed

-- The theorem stating what we need to prove.
theorem roller_coaster_costs_7_tickets :
  roller_coaster_tickets = 7 :=
by sorry

end NUMINAMATH_GPT_roller_coaster_costs_7_tickets_l1654_165499


namespace NUMINAMATH_GPT_complex_division_l1654_165436

def i : ℂ := Complex.I

theorem complex_division :
  (i^3 / (1 + i)) = -1/2 - 1/2 * i := 
by sorry

end NUMINAMATH_GPT_complex_division_l1654_165436


namespace NUMINAMATH_GPT_cubed_identity_l1654_165474

theorem cubed_identity (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := 
by 
  sorry

end NUMINAMATH_GPT_cubed_identity_l1654_165474


namespace NUMINAMATH_GPT_find_h_plus_k_l1654_165407

theorem find_h_plus_k (h k : ℝ) :
  (∀ (x y : ℝ),
    (x - 3) ^ 2 + (y + 4) ^ 2 = 49) → 
  h = 3 ∧ k = -4 → 
  h + k = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_h_plus_k_l1654_165407


namespace NUMINAMATH_GPT_julia_total_kids_l1654_165469

def kidsMonday : ℕ := 7
def kidsTuesday : ℕ := 13
def kidsThursday : ℕ := 18
def kidsWednesdayCards : ℕ := 20
def kidsWednesdayHideAndSeek : ℕ := 11
def kidsWednesdayPuzzle : ℕ := 9
def kidsFridayBoardGame : ℕ := 15
def kidsFridayDrawingCompetition : ℕ := 12

theorem julia_total_kids : 
  kidsMonday + kidsTuesday + kidsThursday + kidsWednesdayCards + kidsWednesdayHideAndSeek + kidsWednesdayPuzzle + kidsFridayBoardGame + kidsFridayDrawingCompetition = 105 :=
by
  sorry

end NUMINAMATH_GPT_julia_total_kids_l1654_165469


namespace NUMINAMATH_GPT_general_term_formula_l1654_165481

variable (a : ℕ → ℤ) -- A sequence of integers 
variable (d : ℤ) -- The common difference 

-- Conditions provided
axiom h1 : a 1 = 6
axiom h2 : a 3 + a 5 = 0
axiom h_arithmetic : ∀ n, a (n + 1) = a n + d -- Arithmetic progression condition

-- The general term formula we need to prove
theorem general_term_formula : ∀ n, a n = 8 - 2 * n := 
by 
  sorry -- Proof goes here


end NUMINAMATH_GPT_general_term_formula_l1654_165481


namespace NUMINAMATH_GPT_area_of_isosceles_trapezoid_l1654_165419

def isIsoscelesTrapezoid (a b c h : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2

theorem area_of_isosceles_trapezoid :
  ∀ (a b c : ℝ), 
    a = 8 → b = 14 → c = 5 →
    ∃ h: ℝ, isIsoscelesTrapezoid a b c h ∧ ((a + b) / 2 * h = 44) :=
by
  intros a b c ha hb hc
  sorry

end NUMINAMATH_GPT_area_of_isosceles_trapezoid_l1654_165419


namespace NUMINAMATH_GPT_total_sum_is_2696_l1654_165498

def numbers := (100, 4900)

def harmonic_mean (a b : ℕ) : ℕ :=
  2 * a * b / (a + b)

def arithmetic_mean (a b : ℕ) : ℕ :=
  (a + b) / 2

theorem total_sum_is_2696 : 
  harmonic_mean numbers.1 numbers.2 + arithmetic_mean numbers.1 numbers.2 = 2696 :=
by
  sorry

end NUMINAMATH_GPT_total_sum_is_2696_l1654_165498


namespace NUMINAMATH_GPT_equivalent_representations_l1654_165470

theorem equivalent_representations :
  (16 / 20 = 24 / 30) ∧
  (80 / 100 = 4 / 5) ∧
  (4 / 5 = 0.8) :=
by 
  sorry

end NUMINAMATH_GPT_equivalent_representations_l1654_165470


namespace NUMINAMATH_GPT_find_number_l1654_165493

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 15) : x = 7.5 :=
sorry

end NUMINAMATH_GPT_find_number_l1654_165493


namespace NUMINAMATH_GPT_sam_current_dimes_l1654_165412

def original_dimes : ℕ := 8
def sister_borrowed : ℕ := 4
def friend_borrowed : ℕ := 2
def sister_returned : ℕ := 2
def friend_returned : ℕ := 1

theorem sam_current_dimes : 
  (original_dimes - sister_borrowed - friend_borrowed + sister_returned + friend_returned = 5) :=
by
  sorry

end NUMINAMATH_GPT_sam_current_dimes_l1654_165412


namespace NUMINAMATH_GPT_no_integer_a_exists_l1654_165462

theorem no_integer_a_exists (a x : ℤ)
  (h : x^3 - a * x^2 - 6 * a * x + a^2 - 3 = 0)
  (unique_sol : ∀ y : ℤ, (y^3 - a * y^2 - 6 * a * y + a^2 - 3 = 0 → y = x)) :
  false :=
by 
  sorry

end NUMINAMATH_GPT_no_integer_a_exists_l1654_165462


namespace NUMINAMATH_GPT_three_different_suits_probability_l1654_165471

def probability_three_different_suits := (39 / 51) * (35 / 50) = 91 / 170

theorem three_different_suits_probability (deck : Finset (Fin 52)) (h : deck.card = 52) :
  probability_three_different_suits :=
sorry

end NUMINAMATH_GPT_three_different_suits_probability_l1654_165471


namespace NUMINAMATH_GPT_probability_red_or_black_probability_red_black_or_white_l1654_165429

theorem probability_red_or_black (total_balls red_balls black_balls : ℕ) : 
  total_balls = 12 → red_balls = 5 → black_balls = 4 → 
  (red_balls + black_balls) / total_balls = 3 / 4 :=
by
  intros
  sorry

theorem probability_red_black_or_white (total_balls red_balls black_balls white_balls : ℕ) :
  total_balls = 12 → red_balls = 5 → black_balls = 4 → white_balls = 2 → 
  (red_balls + black_balls + white_balls) / total_balls = 11 / 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_red_or_black_probability_red_black_or_white_l1654_165429


namespace NUMINAMATH_GPT_solve_equation_l1654_165487

-- Define the equation to be solved
def equation (x : ℝ) : Prop := (x + 2)^4 + (x - 4)^4 = 272

-- State the theorem we want to prove
theorem solve_equation : ∃ x : ℝ, equation x :=
  sorry

end NUMINAMATH_GPT_solve_equation_l1654_165487


namespace NUMINAMATH_GPT_length_of_AB_l1654_165490

variables {A B P Q : ℝ}
variables (x y : ℝ)

-- Conditions
axiom h1 : A < P ∧ P < Q ∧ Q < B
axiom h2 : P - A = 3 * x
axiom h3 : B - P = 5 * x
axiom h4 : Q - A = 2 * y
axiom h5 : B - Q = 3 * y
axiom h6 : Q - P = 3

-- Theorem statement
theorem length_of_AB : B - A = 120 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l1654_165490


namespace NUMINAMATH_GPT_janet_spending_difference_l1654_165447

-- Definitions for the conditions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℕ := 52

-- The theorem to be proven
theorem janet_spending_difference :
  (piano_hourly_rate * piano_hours_per_week * weeks_per_year - clarinet_hourly_rate * clarinet_hours_per_week * weeks_per_year) = 1040 :=
by
  sorry

end NUMINAMATH_GPT_janet_spending_difference_l1654_165447


namespace NUMINAMATH_GPT_sequence_a7_l1654_165463

theorem sequence_a7 (a b : ℕ) (h1 : a1 = a) (h2 : a2 = b) {a3 a4 a5 a6 a7 : ℕ}
  (h3 : a_3 = a + b)
  (h4 : a_4 = a + 2 * b)
  (h5 : a_5 = 2 * a + 3 * b)
  (h6 : a_6 = 3 * a + 5 * b)
  (h_a6 : a_6 = 50) :
  a_7 = 5 * a + 8 * b :=
by
  sorry

end NUMINAMATH_GPT_sequence_a7_l1654_165463


namespace NUMINAMATH_GPT_angle_measure_l1654_165475

theorem angle_measure (x y : ℝ) 
  (h1 : y = 3 * x + 10) 
  (h2 : x + y = 180) : x = 42.5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_angle_measure_l1654_165475


namespace NUMINAMATH_GPT_a7_is_1_S2022_is_4718_l1654_165434

def harmonious_progressive (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, p > 0 → q > 0 → a p = a q → a (p + 1) = a (q + 1)

variables (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom harmonious_seq : harmonious_progressive a
axiom a1 : a 1 = 1
axiom a2 : a 2 = 2
axiom a4 : a 4 = 1
axiom a6_plus_a8 : a 6 + a 8 = 6

theorem a7_is_1 : a 7 = 1 := sorry

theorem S2022_is_4718 : S 2022 = 4718 := sorry

end NUMINAMATH_GPT_a7_is_1_S2022_is_4718_l1654_165434


namespace NUMINAMATH_GPT_shaded_rectangle_ratio_l1654_165444

variable (a : ℝ) (h : 0 < a)  -- side length of the square is 'a' and it is positive

theorem shaded_rectangle_ratio :
  (∃ l w : ℝ, (l = a / 2 ∧ w = a / 3 ∧ (l * w = a^2 / 6) ∧ (a^2 / 6 = a * a / 6))) → (l / w = 1.5) :=
by {
  -- Proof is to be provided
  sorry
}

end NUMINAMATH_GPT_shaded_rectangle_ratio_l1654_165444


namespace NUMINAMATH_GPT_number_of_BA3_in_sample_l1654_165420

-- Definitions for the conditions
def strains_BA1 : Nat := 60
def strains_BA2 : Nat := 20
def strains_BA3 : Nat := 40
def total_sample_size : Nat := 30

def total_strains : Nat := strains_BA1 + strains_BA2 + strains_BA3

-- Theorem statement translating to the equivalent proof problem
theorem number_of_BA3_in_sample :
  total_sample_size * strains_BA3 / total_strains = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_BA3_in_sample_l1654_165420


namespace NUMINAMATH_GPT_lucca_bread_fraction_l1654_165454

theorem lucca_bread_fraction 
  (total_bread : ℕ)
  (initial_fraction_eaten : ℚ)
  (final_pieces : ℕ)
  (bread_first_day : ℚ)
  (bread_second_day : ℚ)
  (bread_third_day : ℚ)
  (remaining_pieces_after_first_day : ℕ)
  (remaining_pieces_after_second_day : ℕ)
  (remaining_pieces_after_third_day : ℕ) :
  total_bread = 200 →
  initial_fraction_eaten = 1/4 →
  bread_first_day = initial_fraction_eaten * total_bread →
  remaining_pieces_after_first_day = total_bread - bread_first_day →
  bread_second_day = (remaining_pieces_after_first_day * bread_second_day) →
  remaining_pieces_after_second_day = remaining_pieces_after_first_day - bread_second_day →
  bread_third_day = 1/2 * remaining_pieces_after_second_day →
  remaining_pieces_after_third_day = remaining_pieces_after_second_day - bread_third_day →
  remaining_pieces_after_third_day = 45 →
  bread_second_day = 2/5 :=
by
  sorry

end NUMINAMATH_GPT_lucca_bread_fraction_l1654_165454


namespace NUMINAMATH_GPT_porter_previous_painting_price_l1654_165428

variable (P : ℝ)

-- Conditions
def condition1 : Prop := 3.5 * P - 1000 = 49000

-- Correct Answer
def answer : ℝ := 14285.71

-- Theorem stating that the answer holds given the conditions
theorem porter_previous_painting_price (h : condition1 P) : P = answer :=
sorry

end NUMINAMATH_GPT_porter_previous_painting_price_l1654_165428


namespace NUMINAMATH_GPT_general_term_of_sequence_l1654_165485

theorem general_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hSn : ∀ n, S n = 3 * n^2 - n + 1) :
  (∀ n, a n = if n = 1 then 3 else 6 * n - 4) :=
by
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1654_165485


namespace NUMINAMATH_GPT_average_last_4_matches_l1654_165421

theorem average_last_4_matches (avg_10_matches avg_6_matches : ℝ) (matches_10 matches_6 matches_4 : ℕ) :
  avg_10_matches = 38.9 →
  avg_6_matches = 41 →
  matches_10 = 10 →
  matches_6 = 6 →
  matches_4 = 4 →
  (avg_10_matches * matches_10 - avg_6_matches * matches_6) / matches_4 = 35.75 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_average_last_4_matches_l1654_165421


namespace NUMINAMATH_GPT_gcd_of_powers_of_two_l1654_165406

noncomputable def m := 2^2048 - 1
noncomputable def n := 2^2035 - 1

theorem gcd_of_powers_of_two : Int.gcd m n = 8191 := by
  sorry

end NUMINAMATH_GPT_gcd_of_powers_of_two_l1654_165406


namespace NUMINAMATH_GPT_a_eq_zero_l1654_165477

noncomputable def f (x a : ℝ) := x^2 - abs (x + a)

theorem a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_a_eq_zero_l1654_165477


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1654_165446

theorem simplify_and_evaluate (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + 3 * b) * (2 * a - b) - 2 * (a - b)^2 = -23 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1654_165446


namespace NUMINAMATH_GPT_geometric_arithmetic_sequence_l1654_165435

theorem geometric_arithmetic_sequence (a q : ℝ) 
    (h₁ : a + a * q + a * q ^ 2 = 19) 
    (h₂ : a * (q - 1) = -1) : 
  (a = 4 ∧ q = 1.5) ∨ (a = 9 ∧ q = 2/3) :=
by
  sorry

end NUMINAMATH_GPT_geometric_arithmetic_sequence_l1654_165435


namespace NUMINAMATH_GPT_two_digit_numbers_l1654_165476

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem two_digit_numbers (a b : ℕ) (h1 : is_digit a) (h2 : is_digit b) 
  (h3 : a ≠ b) (h4 : (a + b) = 11) : 
  (∃ n m : ℕ, (n = 10 * a + b) ∧ (m = 10 * b + a) ∧ (∃ k : ℕ, (10 * a + b)^2 - (10 * b + a)^2 = k^2)) := 
sorry

end NUMINAMATH_GPT_two_digit_numbers_l1654_165476


namespace NUMINAMATH_GPT_apples_handout_l1654_165415

theorem apples_handout {total_apples pies_needed pies_count handed_out : ℕ}
  (h1 : total_apples = 51)
  (h2 : pies_needed = 5)
  (h3 : pies_count = 2)
  (han : handed_out = total_apples - (pies_needed * pies_count)) :
  handed_out = 41 :=
by {
  sorry
}

end NUMINAMATH_GPT_apples_handout_l1654_165415


namespace NUMINAMATH_GPT_inequality_inequality_l1654_165403

open Real

theorem inequality_inequality (n : ℕ) (k : ℝ) (hn : 0 < n) (hk : 0 < k) : 
  1 - 1/k ≤ n * (k^(1 / n) - 1) ∧ n * (k^(1 / n) - 1) ≤ k - 1 := 
  sorry

end NUMINAMATH_GPT_inequality_inequality_l1654_165403


namespace NUMINAMATH_GPT_problem_statement_l1654_165425

variable (a b c d : ℝ)

noncomputable def circle_condition_1 : Prop := a = (1 : ℝ) / a
noncomputable def circle_condition_2 : Prop := b = (1 : ℝ) / b
noncomputable def circle_condition_3 : Prop := c = (1 : ℝ) / c
noncomputable def circle_condition_4 : Prop := d = (1 : ℝ) / d

theorem problem_statement (h1 : circle_condition_1 a)
                          (h2 : circle_condition_2 b)
                          (h3 : circle_condition_3 c)
                          (h4 : circle_condition_4 d) :
    2 * (a^2 + b^2 + c^2 + d^2) = (a + b + c + d)^2 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1654_165425


namespace NUMINAMATH_GPT_probability_of_event_A_l1654_165465

noncomputable def probability_both_pieces_no_less_than_three_meters (L : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  if h : L = a + b 
  then (if a ≥ 3 ∧ b ≥ 3 then (L - 2 * 3) / L else 0)
  else 0

theorem probability_of_event_A : 
  probability_both_pieces_no_less_than_three_meters 11 6 5 = 5 / 11 :=
by
  -- Additional context to ensure proper definition of the problem
  sorry

end NUMINAMATH_GPT_probability_of_event_A_l1654_165465


namespace NUMINAMATH_GPT_find_natural_numbers_l1654_165457

theorem find_natural_numbers (x : ℕ) : (x % 7 = 3) ∧ (x % 9 = 4) ∧ (x < 100) ↔ (x = 31) ∨ (x = 94) := 
by sorry

end NUMINAMATH_GPT_find_natural_numbers_l1654_165457


namespace NUMINAMATH_GPT_cody_books_second_week_l1654_165402

noncomputable def total_books := 54
noncomputable def books_first_week := 6
noncomputable def books_weeks_after_second := 9
noncomputable def total_weeks := 7

theorem cody_books_second_week :
  let b2 := total_books - (books_first_week + books_weeks_after_second * (total_weeks - 2))
  b2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_cody_books_second_week_l1654_165402


namespace NUMINAMATH_GPT_distinct_connected_stamps_l1654_165441

theorem distinct_connected_stamps (n : ℕ) : 
  ∃ d : ℕ → ℝ, 
    d (n+1) = 1 / 4 * (1 + Real.sqrt 2)^(n + 3) + 1 / 4 * (1 - Real.sqrt 2)^(n + 3) - 2 * n - 7 / 2 :=
sorry

end NUMINAMATH_GPT_distinct_connected_stamps_l1654_165441


namespace NUMINAMATH_GPT_simplify_expression_l1654_165489

theorem simplify_expression (a : ℝ) (h : a = Real.sqrt 3 - 3) : 
  (a^2 - 4 * a + 4) / (a^2 - 4) / ((a - 2) / (a^2 + 2 * a)) + 3 = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1654_165489


namespace NUMINAMATH_GPT_gcd_lcm_identity_l1654_165417

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := a * (b / GCD a b)

theorem gcd_lcm_identity (a b c : ℕ) :
    (LCM a (LCM b c))^2 / (LCM a b * LCM b c * LCM c a) = (GCD a (GCD b c))^2 / (GCD a b * GCD b c * GCD c a) :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_identity_l1654_165417


namespace NUMINAMATH_GPT_units_digit_of_k_squared_plus_2_k_l1654_165483

def k := 2008^2 + 2^2008

theorem units_digit_of_k_squared_plus_2_k : 
  (k^2 + 2^k) % 10 = 7 :=
by {
  -- The proof will be inserted here
  sorry
}

end NUMINAMATH_GPT_units_digit_of_k_squared_plus_2_k_l1654_165483


namespace NUMINAMATH_GPT_find_value_of_expression_l1654_165456

theorem find_value_of_expression
  (x y z : ℝ)
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x + 2 * y - 7 * z = 0)
  (hz : z ≠ 0) :
  (x^2 - 2 * x * y) / (y^2 + 4 * z^2) = -0.252 := 
sorry

end NUMINAMATH_GPT_find_value_of_expression_l1654_165456


namespace NUMINAMATH_GPT_minimize_expression_l1654_165497

theorem minimize_expression (x : ℝ) (h : 0 < x) : 
  x = 9 ↔ (∀ y : ℝ, 0 < y → x + 81 / x ≤ y + 81 / y) :=
sorry

end NUMINAMATH_GPT_minimize_expression_l1654_165497


namespace NUMINAMATH_GPT_students_in_class_l1654_165418

def total_eggs : Nat := 56
def eggs_per_student : Nat := 8
def num_students : Nat := 7

theorem students_in_class :
  total_eggs / eggs_per_student = num_students :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l1654_165418


namespace NUMINAMATH_GPT_orlando_weight_gain_l1654_165486

def weight_gain_statement (x J F : ℝ) : Prop :=
  J = 2 * x + 2 ∧ F = 1/2 * J - 3 ∧ x + J + F = 20

theorem orlando_weight_gain :
  ∃ x J F : ℝ, weight_gain_statement x J F ∧ x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_orlando_weight_gain_l1654_165486


namespace NUMINAMATH_GPT_ratio_y_to_x_l1654_165424

-- Define the setup as given in the conditions
variables (c x y : ℝ)

-- Condition 1: Selling price x results in a loss of 20%
def condition1 : Prop := x = 0.80 * c

-- Condition 2: Selling price y results in a profit of 25%
def condition2 : Prop := y = 1.25 * c

-- Theorem: Prove the ratio of y to x is 25/16 given the conditions
theorem ratio_y_to_x (c : ℝ) (h1 : condition1 c x) (h2 : condition2 c y) : y / x = 25 / 16 := 
sorry

end NUMINAMATH_GPT_ratio_y_to_x_l1654_165424


namespace NUMINAMATH_GPT_lighting_effect_improves_l1654_165458

theorem lighting_effect_improves (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
    (a + m) / (b + m) > a / b := 
sorry

end NUMINAMATH_GPT_lighting_effect_improves_l1654_165458


namespace NUMINAMATH_GPT_find_eagle_feathers_times_l1654_165467

theorem find_eagle_feathers_times (x : ℕ) (hawk_feathers : ℕ) (total_feathers_before_give : ℕ) (total_feathers : ℕ) (left_after_selling : ℕ) :
  hawk_feathers = 6 →
  total_feathers_before_give = 6 + 6 * x →
  total_feathers = total_feathers_before_give - 10 →
  left_after_selling = total_feathers / 2 →
  left_after_selling = 49 →
  x = 17 :=
by
  intros h_hawk h_total_before_give h_total h_left h_after_selling
  sorry

end NUMINAMATH_GPT_find_eagle_feathers_times_l1654_165467


namespace NUMINAMATH_GPT_least_positive_integer_addition_l1654_165431

theorem least_positive_integer_addition (k : ℕ) (h₀ : 525 + k % 5 = 0) (h₁ : 0 < k) : k = 5 := 
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_addition_l1654_165431


namespace NUMINAMATH_GPT_drive_photos_storage_l1654_165496

theorem drive_photos_storage (photo_size: ℝ) (num_photos_with_videos: ℕ) (photo_storage_with_videos: ℝ) (video_size: ℝ) (num_videos_with_photos: ℕ) : 
  num_photos_with_videos * photo_size + num_videos_with_photos * video_size = 3000 → 
  (3000 / photo_size) = 2000 :=
by
  sorry

end NUMINAMATH_GPT_drive_photos_storage_l1654_165496


namespace NUMINAMATH_GPT_arccos_gt_arctan_on_interval_l1654_165450

noncomputable def c : ℝ := sorry -- placeholder for the numerical solution of arccos x = arctan x

theorem arccos_gt_arctan_on_interval (x : ℝ) (hx : -1 ≤ x ∧ x < c) :
  Real.arccos x > Real.arctan x := 
sorry

end NUMINAMATH_GPT_arccos_gt_arctan_on_interval_l1654_165450


namespace NUMINAMATH_GPT_B_values_for_divisibility_l1654_165442

theorem B_values_for_divisibility (B : ℕ) (h : 4 + B + B + B + 2 ≡ 0 [MOD 9]) : B = 1 ∨ B = 4 ∨ B = 7 :=
by sorry

end NUMINAMATH_GPT_B_values_for_divisibility_l1654_165442


namespace NUMINAMATH_GPT_correct_multiplication_result_l1654_165464

theorem correct_multiplication_result :
  ∃ x : ℕ, (x * 9 = 153) ∧ (x * 6 = 102) :=
by
  sorry

end NUMINAMATH_GPT_correct_multiplication_result_l1654_165464


namespace NUMINAMATH_GPT_evaluate_polynomial_l1654_165492

noncomputable def polynomial_evaluation : Prop :=
∀ (x : ℝ), x^2 - 3*x - 9 = 0 ∧ 0 < x → (x^4 - 3*x^3 - 9*x^2 + 27*x - 8) = (65 + 81*(Real.sqrt 5))/2

theorem evaluate_polynomial : polynomial_evaluation :=
sorry

end NUMINAMATH_GPT_evaluate_polynomial_l1654_165492


namespace NUMINAMATH_GPT_log_positive_interval_l1654_165449

noncomputable def f (a x : ℝ) : ℝ := Real.log (2 * x - a) / Real.log a

theorem log_positive_interval (a : ℝ) :
  (∀ x, x ∈ Set.Icc (1 / 2) (2 / 3) → f a x > 0) ↔ (1 / 3 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_GPT_log_positive_interval_l1654_165449


namespace NUMINAMATH_GPT_find_third_month_sale_l1654_165438

theorem find_third_month_sale
  (sale_1 sale_2 sale_3 sale_4 sale_5 sale_6 : ℕ)
  (h1 : sale_1 = 800)
  (h2 : sale_2 = 900)
  (h4 : sale_4 = 700)
  (h5 : sale_5 = 800)
  (h6 : sale_6 = 900)
  (h_avg : (sale_1 + sale_2 + sale_3 + sale_4 + sale_5 + sale_6) / 6 = 850) : 
  sale_3 = 1000 :=
by
  sorry

end NUMINAMATH_GPT_find_third_month_sale_l1654_165438


namespace NUMINAMATH_GPT_megs_cat_weight_l1654_165445

/-- The ratio of the weight of Meg's cat to Anne's cat is 5:7 and Anne's cat weighs 8 kg more than Meg's cat. Prove that the weight of Meg's cat is 20 kg. -/
theorem megs_cat_weight
  (M A : ℝ)
  (h1 : M / A = 5 / 7)
  (h2 : A = M + 8) :
  M = 20 :=
sorry

end NUMINAMATH_GPT_megs_cat_weight_l1654_165445


namespace NUMINAMATH_GPT_sets_equal_l1654_165422

theorem sets_equal :
  {u : ℤ | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l} =
  {u : ℤ | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r} := 
sorry

end NUMINAMATH_GPT_sets_equal_l1654_165422


namespace NUMINAMATH_GPT_squirrel_spiral_distance_l1654_165432

/-- The squirrel runs up a cylindrical post in a perfect spiral path, making one circuit for each rise of 4 feet.
Given the post is 16 feet tall and 3 feet in circumference, the total distance traveled by the squirrel is 20 feet. -/
theorem squirrel_spiral_distance :
  let height : ℝ := 16
  let circumference : ℝ := 3
  let rise_per_circuit : ℝ := 4
  let number_of_circuits := height / rise_per_circuit
  let distance_per_circuit := (circumference^2 + rise_per_circuit^2).sqrt
  number_of_circuits * distance_per_circuit = 20 := by
  sorry

end NUMINAMATH_GPT_squirrel_spiral_distance_l1654_165432


namespace NUMINAMATH_GPT_simplify_expression_l1654_165427

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 25) = 152 * x + 45 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1654_165427


namespace NUMINAMATH_GPT_quadratic_range_l1654_165440

theorem quadratic_range (x y : ℝ) 
    (h1 : y = (x - 1)^2 + 1)
    (h2 : 2 ≤ y ∧ y < 5) : 
    (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_range_l1654_165440


namespace NUMINAMATH_GPT_length_width_percentage_change_l1654_165400

variables (L W : ℝ) (x : ℝ)
noncomputable def area_change_percent : ℝ :=
  (L * (1 + x / 100) * W * (1 - x / 100) - L * W) / (L * W) * 100

theorem length_width_percentage_change (h : area_change_percent L W x = 4) :
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_length_width_percentage_change_l1654_165400


namespace NUMINAMATH_GPT_melanie_marbles_l1654_165480

noncomputable def melanie_blue_marbles : ℕ :=
  let sandy_dozen_marbles := 56
  let dozen := 12
  let sandy_marbles := sandy_dozen_marbles * dozen
  let ratio := 8
  sandy_marbles / ratio

theorem melanie_marbles (h1 : ∀ sandy_dozen_marbles dozen ratio, 56 = sandy_dozen_marbles ∧ sandy_dozen_marbles * dozen = 672 ∧ ratio = 8) : melanie_blue_marbles = 84 := by
  sorry

end NUMINAMATH_GPT_melanie_marbles_l1654_165480


namespace NUMINAMATH_GPT_jugglers_balls_needed_l1654_165413

theorem jugglers_balls_needed (juggler_count balls_per_juggler : ℕ)
  (h_juggler_count : juggler_count = 378)
  (h_balls_per_juggler : balls_per_juggler = 6) :
  juggler_count * balls_per_juggler = 2268 :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_jugglers_balls_needed_l1654_165413


namespace NUMINAMATH_GPT_min_max_sums_l1654_165439

theorem min_max_sums (a b c d e f g : ℝ) 
    (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c)
    (h3 : 0 ≤ d) (h4 : 0 ≤ e) (h5 : 0 ≤ f) 
    (h6 : 0 ≤ g) (h_sum : a + b + c + d + e + f + g = 1) :
    (min (max (a + b + c) 
              (max (b + c + d) 
                   (max (c + d + e) 
                        (max (d + e + f) 
                             (e + f + g))))) = 1 / 3) :=
sorry

end NUMINAMATH_GPT_min_max_sums_l1654_165439


namespace NUMINAMATH_GPT_mixture_ratio_l1654_165495

variables (p q : ℝ)

theorem mixture_ratio 
  (h1 : (5/8) * p + (1/4) * q = 0.5)
  (h2 : (3/8) * p + (3/4) * q = 0.5) : 
  p / q = 1 := 
by 
  sorry

end NUMINAMATH_GPT_mixture_ratio_l1654_165495


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1654_165423

theorem eccentricity_of_ellipse :
  ∀ (x y : ℝ), (x^2) / 25 + (y^2) / 16 = 1 → 
  (∃ (e : ℝ), e = 3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1654_165423


namespace NUMINAMATH_GPT_find_five_digit_number_l1654_165460

theorem find_five_digit_number (x : ℕ) (hx : 10000 ≤ x ∧ x < 100000)
  (h : 10 * x + 1 = 3 * (100000 + x) ∨ 3 * (10 * x + 1) = 100000 + x) :
  x = 42857 :=
sorry

end NUMINAMATH_GPT_find_five_digit_number_l1654_165460


namespace NUMINAMATH_GPT_trajectory_of_moving_point_l1654_165409

theorem trajectory_of_moving_point (P : ℝ × ℝ) (F1 : ℝ × ℝ) (F2 : ℝ × ℝ)
  (hF1 : F1 = (-2, 0)) (hF2 : F2 = (2, 0))
  (h_arith_mean : dist F1 F2 = (dist P F1 + dist P F2) / 2) :
  ∃ a b : ℝ, a = 4 ∧ b^2 = 12 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1) :=
sorry

end NUMINAMATH_GPT_trajectory_of_moving_point_l1654_165409


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_l1654_165433

variable (b l : ℕ)

def length_eq_thrice_breadth (b : ℕ) : ℕ := 3 * b

def area_of_rectangle_eq_2700 (b l : ℕ) : Prop := l * b = 2700

theorem breadth_of_rectangular_plot (h1 : l = 3 * b) (h2 : l * b = 2700) : b = 30 :=
by
  sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_l1654_165433


namespace NUMINAMATH_GPT_quadratic_roots_l1654_165437

variable {a b c : ℝ}

theorem quadratic_roots (h₁ : a > 0) (h₂ : b > 0) (h₃ : c < 0) : 
  ∃ x₁ x₂ : ℝ, (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ 
  (x₁ ≠ x₂) ∧ (x₁ > 0) ∧ (x₂ < 0) ∧ (|x₂| > |x₁|) := 
sorry

end NUMINAMATH_GPT_quadratic_roots_l1654_165437
