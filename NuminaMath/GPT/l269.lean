import Mathlib

namespace division_and_subtraction_l269_269761

theorem division_and_subtraction : (23 ^ 11 / 23 ^ 8) - 15 = 12152 := by
  sorry

end division_and_subtraction_l269_269761


namespace Petya_can_determine_weight_l269_269270

theorem Petya_can_determine_weight (n : ℕ) (distinct_weights : Fin n → ℕ) 
  (device : (Fin 10 → Fin n) → ℕ) (ten_thousand_weights : n = 10000)
  (no_two_same : (∀ i j : Fin n, i ≠ j → distinct_weights i ≠ distinct_weights j)) :
  ∃ i : Fin n, ∃ w : ℕ, distinct_weights i = w :=
by
  sorry

end Petya_can_determine_weight_l269_269270


namespace sequence_general_term_l269_269044

theorem sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1 * 2) ∧ (a 2 = 2 * 3) ∧ (a 3 = 3 * 4) ∧ (a 4 = 4 * 5) ↔ 
    (∀ n, a n = n^2 + n) := sorry

end sequence_general_term_l269_269044


namespace fill_pond_time_l269_269461

-- Define the constants and their types
def pondVolume : ℕ := 200 -- Volume of the pond in gallons
def normalRate : ℕ := 6 -- Normal rate of the hose in gallons per minute

-- Define the reduced rate due to drought restrictions
def reducedRate : ℕ := (2/3 : ℚ) * normalRate

-- Define the time required to fill the pond
def timeToFill : ℚ := pondVolume / reducedRate

-- The main statement to be proven
theorem fill_pond_time : timeToFill = 50 := by
  sorry

end fill_pond_time_l269_269461


namespace problem1_problem2_l269_269047

-- (Problem 1)
def A : Set ℝ := {x | x^2 + 2 * x < 0}
def B : Set ℝ := {x | x ≥ -1}
def complement_A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 0}
def intersection_complement_A_B : Set ℝ := {x | x ≥ 0}

theorem problem1 : (complement_A ∩ B) = intersection_complement_A_B :=
by
  sorry

-- (Problem 2)
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 1}

theorem problem2 {a : ℝ} : (C a ⊆ A) ↔ (a ≤ -1 / 2) :=
by
  sorry

end problem1_problem2_l269_269047


namespace problem_inequality_l269_269686

theorem problem_inequality (x y z : ℝ) (h1 : x + y + z = 0) (h2 : |x| + |y| + |z| ≤ 1) :
  x + (y / 2) + (z / 3) ≤ 1 / 3 :=
sorry

end problem_inequality_l269_269686


namespace solve_quadratic_condition_l269_269903

noncomputable def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
  if h : a ≠ 0 then
    let disc := b^2 - 4 * a * c
    let sqrt_disc := Real.sqrt disc
    ((-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a))
  else (0, 0) -- Not handling degenerate quadratic

theorem solve_quadratic_condition (m : ℝ) :
  let p := quadratic_roots 1 (-4) (m-1)
  let x1 := p.1
  let x2 := p.2
  3 * x1 * x2 - x1 - x2 > 2 → 3 < m ∧ m ≤ 5 ∧ Real.sqrt(16 - 4 * (m-1)) ≥ 0 :=
by
  sorry

end solve_quadratic_condition_l269_269903


namespace opposite_of_neg_nine_is_nine_l269_269702

-- Define the predicate for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that needs to be proved
theorem opposite_of_neg_nine_is_nine : ∃ x, is_opposite (-9) x ∧ x = 9 := 
by {
  -- Use "sorry" to indicate the proof is not provided
  sorry
}

end opposite_of_neg_nine_is_nine_l269_269702


namespace product_of_numbers_l269_269495

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 :=
sorry

end product_of_numbers_l269_269495


namespace find_f3_l269_269615

theorem find_f3 (a b : ℝ) (f : ℝ → ℝ)
  (h1 : f 1 = 4)
  (h2 : f 2 = 10)
  (h3 : ∀ x, f x = a * x^2 + b * x + 2) :
  f 3 = 20 :=
by
  sorry

end find_f3_l269_269615


namespace total_rooms_l269_269579

-- Definitions for the problem conditions
variables (x y : ℕ)

-- Given conditions
def condition1 : Prop := x = 8
def condition2 : Prop := 2 * x + 3 * y = 31

-- The theorem to prove
theorem total_rooms (h1 : condition1 x) (h2 : condition2 x y) : x + y = 13 :=
by sorry

end total_rooms_l269_269579


namespace table_length_l269_269872

theorem table_length (L : ℕ) (H1 : ∃ n : ℕ, 80 = n * L)
  (H2 : L ≥ 16) (H3 : ∃ m : ℕ, 16 = m * 4)
  (H4 : L % 4 = 0) : L = 20 := by 
sorry

end table_length_l269_269872


namespace function_periodicity_l269_269213

theorem function_periodicity
  (f : ℝ → ℝ)
  (H_odd : ∀ x, f (-x) = -f x)
  (H_even_shift : ∀ x, f (x + 2) = f (-x + 2))
  (H_val_neg1 : f (-1) = -1)
  : f 2017 + f 2016 = 1 := 
sorry

end function_periodicity_l269_269213


namespace sum_of_coordinates_l269_269670

theorem sum_of_coordinates (x y : ℚ) (h₁ : y = 5) (h₂ : (y - 0) / (x - 0) = 3/4) : x + y = 35/3 :=
by sorry

end sum_of_coordinates_l269_269670


namespace compute_sqrt_eq_419_l269_269762

theorem compute_sqrt_eq_419 : Real.sqrt ((22 * 21 * 20 * 19) + 1) = 419 :=
by
  sorry

end compute_sqrt_eq_419_l269_269762


namespace apple_percentage_is_23_l269_269695

def total_responses := 70 + 80 + 50 + 30 + 70
def apple_responses := 70

theorem apple_percentage_is_23 :
  (apple_responses : ℝ) / (total_responses : ℝ) * 100 = 23 := 
by
  sorry

end apple_percentage_is_23_l269_269695


namespace part1_part2_l269_269026

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269026


namespace train_speed_l269_269144

theorem train_speed (distance time : ℝ) (h₀ : distance = 180) (h₁ : time = 9) : 
  ((distance / 1000) / (time / 3600)) = 72 :=
by 
  -- below statement will bring the remainder of the setup and will be proved without the steps
  sorry

end train_speed_l269_269144


namespace circle_diameter_l269_269546

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l269_269546


namespace valid_lineups_l269_269667

def total_players : ℕ := 15
def k : ℕ := 2  -- number of twins
def total_chosen : ℕ := 7
def remaining_players := total_players - k

def nCr (n r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.choose n r

def total_choices : ℕ := nCr total_players total_chosen
def restricted_choices : ℕ := nCr remaining_players (total_chosen - k)

theorem valid_lineups : total_choices - restricted_choices = 5148 := by
  sorry

end valid_lineups_l269_269667


namespace bankers_discount_problem_l269_269730

theorem bankers_discount_problem
  (BD : ℚ) (TD : ℚ) (SD : ℚ)
  (h1 : BD = 36)
  (h2 : TD = 30)
  (h3 : BD = TD + TD^2 / SD) :
  SD = 150 := 
sorry

end bankers_discount_problem_l269_269730


namespace infinite_sum_eq_3_over_8_l269_269196

theorem infinite_sum_eq_3_over_8 :
  ∑' n : ℕ, (n : ℝ) / (n^4 + 4) = 3 / 8 :=
sorry

end infinite_sum_eq_3_over_8_l269_269196


namespace power_function_characterization_l269_269631

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_characterization (f : ℝ → ℝ) (h : f 2 = Real.sqrt 2) : 
  ∀ x : ℝ, f x = x ^ (1 / 2) :=
sorry

end power_function_characterization_l269_269631


namespace product_of_two_numbers_l269_269503

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end product_of_two_numbers_l269_269503


namespace simplify_expression_l269_269733

theorem simplify_expression : 
  let i : ℂ := complex.I in
  ( (i^3 = -i) → ((2 + i) * (2 - i) = 5) → (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i ) :=
by
  let i : ℂ := complex.I
  assume h₁ : i^3 = -i
  assume h₂ : (2 + i) * (2 - i) = 5
  sorry

end simplify_expression_l269_269733


namespace solution1_solution2_l269_269340

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l269_269340


namespace right_triangle_area_l269_269413

theorem right_triangle_area (a b : ℝ) (ha : a = 3) (hb : b = 5) : 
  (1 / 2) * a * b = 7.5 := 
by
  rw [ha, hb]
  sorry

end right_triangle_area_l269_269413


namespace find_smaller_number_l269_269040

theorem find_smaller_number (n m : ℕ) (h1 : n - m = 58)
  (h2 : n^2 % 100 = m^2 % 100) : m = 21 :=
by
  sorry

end find_smaller_number_l269_269040


namespace average_age_choir_l269_269104

theorem average_age_choir (S_f S_m S_total : ℕ) (avg_f : ℕ) (avg_m : ℕ) (females males total : ℕ)
  (h1 : females = 8) (h2 : males = 12) (h3 : total = 20)
  (h4 : avg_f = 25) (h5 : avg_m = 40)
  (h6 : S_f = avg_f * females) 
  (h7 : S_m = avg_m * males) 
  (h8 : S_total = S_f + S_m) :
  (S_total / total) = 34 := by
  sorry

end average_age_choir_l269_269104


namespace sum_coordinates_eq_l269_269680

theorem sum_coordinates_eq : ∀ (A B : ℝ × ℝ), A = (0, 0) → (∃ x : ℝ, B = (x, 5)) → 
  (∀ x : ℝ, B = (x, 5) → (5 - 0) / (x - 0) = 3 / 4) → 
  let x := 20 / 3 in
  (x + 5 = 35 / 3) :=
by
  intros A B hA hB hslope
  have hx : ∃ x, B = (x, 5) := hB
  cases hx with x hx_def
  rw hx_def at hslope
  have : x = 20 / 3 := by
    rw hx_def at hslope
    field_simp at hslope
    linarith
  rw [this, hx_def]
  norm_num

end sum_coordinates_eq_l269_269680


namespace average_speed_of_car_l269_269523

/-- The average speed of a car over four hours given specific distances covered each hour. -/
theorem average_speed_of_car
  (d1 d2 d3 d4 : ℝ)
  (t1 t2 t3 t4 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 40)
  (h3 : d3 = 60)
  (h4 : d4 = 100)
  (h5 : t1 = 1)
  (h6 : t2 = 1)
  (h7 : t3 = 1)
  (h8 : t4 = 1) :
  (d1 + d2 + d3 + d4) / (t1 + t2 + t3 + t4) = 55 :=
by sorry

end average_speed_of_car_l269_269523


namespace movie_theater_loss_l269_269558

theorem movie_theater_loss :
  let capacity := 50
  let ticket_price := 8.0
  let tickets_sold := 24
  (capacity * ticket_price - tickets_sold * ticket_price) = 208 := by
  sorry

end movie_theater_loss_l269_269558


namespace rectangle_area_l269_269373

theorem rectangle_area :
  ∃ (x y : ℝ), (x + 3.5) * (y - 1.5) = x * y ∧
               (x - 3.5) * (y + 2.5) = x * y ∧
               2 * (x + 3.5) + 2 * (y - 3.5) = 2 * x + 2 * y ∧
               x * y = 196 :=
by
  sorry

end rectangle_area_l269_269373


namespace min_value_fraction_l269_269781

theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : (1 / m + 2 / n) ≥ 8 :=
sorry

end min_value_fraction_l269_269781


namespace sum_coordinates_eq_l269_269681

theorem sum_coordinates_eq : ∀ (A B : ℝ × ℝ), A = (0, 0) → (∃ x : ℝ, B = (x, 5)) → 
  (∀ x : ℝ, B = (x, 5) → (5 - 0) / (x - 0) = 3 / 4) → 
  let x := 20 / 3 in
  (x + 5 = 35 / 3) :=
by
  intros A B hA hB hslope
  have hx : ∃ x, B = (x, 5) := hB
  cases hx with x hx_def
  rw hx_def at hslope
  have : x = 20 / 3 := by
    rw hx_def at hslope
    field_simp at hslope
    linarith
  rw [this, hx_def]
  norm_num

end sum_coordinates_eq_l269_269681


namespace rational_square_of_1_minus_xy_l269_269228

theorem rational_square_of_1_minus_xy (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : ∃ (q : ℚ), 1 - x * y = q^2 :=
by
  sorry

end rational_square_of_1_minus_xy_l269_269228


namespace range_of_a_l269_269796

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 0 < x ∧ x < 4 → |x - 1| < a) ↔ 3 ≤ a :=
sorry

end range_of_a_l269_269796


namespace john_max_questions_correct_l269_269869

variable (c w b : ℕ)

theorem john_max_questions_correct (H1 : c + w + b = 20) (H2 : 5 * c - 2 * w = 48) : c ≤ 12 := sorry

end john_max_questions_correct_l269_269869


namespace average_speed_for_trip_l269_269359

-- Define the total distance of the trip
def total_distance : ℕ := 850

--  Define the distance and speed for the first part of the trip
def distance1 : ℕ := 400
def speed1 : ℕ := 20

-- Define the distance and speed for the remaining part of the trip
def distance2 : ℕ := 450
def speed2 : ℕ := 15

-- Define the calculated average speed for the entire trip
def average_speed : ℕ := 17

theorem average_speed_for_trip 
  (d_total : ℕ)
  (d1 : ℕ) (s1 : ℕ)
  (d2 : ℕ) (s2 : ℕ)
  (hsum : d1 + d2 = d_total)
  (d1_eq : d1 = distance1)
  (s1_eq : s1 = speed1)
  (d2_eq : d2 = distance2)
  (s2_eq : s2 = speed2) :
  (d_total / ((d1 / s1) + (d2 / s2))) = average_speed := by
  sorry

end average_speed_for_trip_l269_269359


namespace quarter_probability_l269_269156

theorem quarter_probability :
  let quarters_value := 12.00
  let quarter_worth := 0.25
  let nickels_value := 5.00
  let nickel_worth := 0.05
  let pennies_value := 2.00
  let penny_worth := 0.01
  let dimes_value := 10.00
  let dime_worth := 0.10
  let num_quarters := quarters_value / quarter_worth
  let num_nickels := nickels_value / nickel_worth
  let num_pennies := pennies_value / penny_worth
  let num_dimes := dimes_value / dime_worth
  let total_coins := num_quarters + num_nickels + num_pennies + num_dimes
  in
  (num_quarters / total_coins) = (3 : ℝ) / 28 := 
by sorry

end quarter_probability_l269_269156


namespace simplify_complex_expression_l269_269736

noncomputable def i : ℂ := Complex.I

theorem simplify_complex_expression : 
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i := by 
  sorry

end simplify_complex_expression_l269_269736


namespace fraction_computation_l269_269590

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l269_269590


namespace min_value_x_l269_269054

theorem min_value_x (x : ℝ) (h : ∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) : x ≥ -1 := 
sorry

end min_value_x_l269_269054


namespace find_a_and_b_function_value_at_0_function_positive_x_less_than_7_over_6_l269_269913

def linear_function (a b x : ℝ) : ℝ := a * x + b

theorem find_a_and_b : ∃ (a b : ℝ), 
  linear_function a b 1 = 1 ∧ 
  linear_function a b 2 = -5 ∧ 
  a = -6 ∧ 
  b = 7 :=
sorry

theorem function_value_at_0 : 
  ∀ a b, 
  a = -6 → b = 7 → 
  linear_function a b 0 = 7 :=
sorry

theorem function_positive_x_less_than_7_over_6 :
  ∀ a b x, 
  a = -6 → b = 7 → 
  x < 7 / 6 → 
  linear_function a b x > 0 :=
sorry

end find_a_and_b_function_value_at_0_function_positive_x_less_than_7_over_6_l269_269913


namespace tank_filled_to_depth_l269_269367

noncomputable def tank_volume (R H r d : ℝ) : ℝ := R^2 * H * Real.pi - (r^2 * H * Real.pi)

theorem tank_filled_to_depth (R H r d : ℝ) (h_cond : R = 5 ∧ H = 12 ∧ r = 2 ∧ d = 3) :
  tank_volume R H r d = 110 * Real.pi - 96 :=
sorry

end tank_filled_to_depth_l269_269367


namespace movie_theater_loss_l269_269557

theorem movie_theater_loss :
  let capacity := 50
  let ticket_price := 8.0
  let tickets_sold := 24
  (capacity * ticket_price - tickets_sold * ticket_price) = 208 := by
  sorry

end movie_theater_loss_l269_269557


namespace original_square_side_length_l269_269140

theorem original_square_side_length (a : ℕ) (initial_thickness final_thickness : ℕ) (side_length_reduction_factor thickness_doubling_factor : ℕ) (s : ℕ) :
  a = 3 →
  final_thickness = 16 →
  initial_thickness = 1 →
  side_length_reduction_factor = 16 →
  thickness_doubling_factor = 16 →
  s * s = side_length_reduction_factor * a * a →
  s = 12 :=
by
  intros ha hfinal_thickness hin_initial_thickness hside_length_reduction_factor hthickness_doubling_factor h_area_equiv
  sorry

end original_square_side_length_l269_269140


namespace sum_of_x_and_y_l269_269928

-- Definitions of conditions
variables (x y : ℤ)
variable (h1 : x - y = 60)
variable (h2 : x = 37)

-- Statement of the problem to be proven
theorem sum_of_x_and_y : x + y = 14 :=
by
  sorry

end sum_of_x_and_y_l269_269928


namespace sum_of_sequences_l269_269221

noncomputable def arithmetic_sequence (a b : ℤ) : Prop :=
  ∃ k : ℤ, a = 6 + k ∧ b = 6 + 2 * k

noncomputable def geometric_sequence (c d : ℤ) : Prop :=
  ∃ q : ℤ, c = 6 * q ∧ d = 6 * q^2

theorem sum_of_sequences (a b c d : ℤ) 
  (h_arith : arithmetic_sequence a b) 
  (h_geom : geometric_sequence c d) 
  (hb : b = 48) (hd : 6 * c^2 = 48): 
  a + b + c + d = 111 := 
sorry

end sum_of_sequences_l269_269221


namespace batsman_average_after_17th_inning_l269_269134

theorem batsman_average_after_17th_inning 
  (score_17 : ℕ)
  (delta_avg : ℤ)
  (n_before : ℕ)
  (initial_avg : ℤ)
  (h1 : score_17 = 74)
  (h2 : delta_avg = 3)
  (h3 : n_before = 16)
  (h4 : initial_avg = 23) :
  (initial_avg + delta_avg) = 26 := 
by
  sorry

end batsman_average_after_17th_inning_l269_269134


namespace elvins_first_month_bill_l269_269887

-- Define the variables involved
variables (F C : ℝ)

-- State the given conditions
def condition1 : Prop := F + C = 48
def condition2 : Prop := F + 2 * C = 90

-- State the theorem we need to prove
theorem elvins_first_month_bill (F C : ℝ) (h1 : F + C = 48) (h2 : F + 2 * C = 90) : F + C = 48 :=
by sorry

end elvins_first_month_bill_l269_269887


namespace part_1_part_2_l269_269013

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269013


namespace fewer_twos_for_100_l269_269329

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l269_269329


namespace part_I_part_II_part_III_l269_269427

noncomputable def P (A B C : set Ω) (prob : MeasureTheory.Measure Ω) : ℝ := sorry

variables {Ω : Type*} {prob : MeasureTheory.Measure Ω}
variables (A B C : set Ω)

-- Given probabilities for individuals A, B, and C
def P_A : ℝ := 0.6
def P_B : ℝ := 0.8
def P_C : ℝ := 0.9

-- Given independence of the events A, B, and C
axiom independent : ⇑ prob (A ∩ B) * ⇑ prob C = ⇑ prob (A ∩ B ∩ C)

-- Part (I)
theorem part_I : 
  P A B C prob = 0.432 :=
sorry

-- Part (II)
theorem part_II : 
  P (Aᶜ ∩ B ∩ C) prob = 0.288 :=
sorry

-- Part (III)
theorem part_III : 
  P (A ∪ B ∪ C) prob = 0.992 :=
sorry

end part_I_part_II_part_III_l269_269427


namespace jimmy_change_l269_269253

noncomputable def change_back (pen_cost notebook_cost folder_cost highlighter_cost sticky_notes_cost total_paid discount tax : ℝ) : ℝ :=
  let total_before_discount := (5 * pen_cost) + (6 * notebook_cost) + (4 * folder_cost) + (3 * highlighter_cost) + (2 * sticky_notes_cost)
  let total_after_discount := total_before_discount * (1 - discount)
  let final_total := total_after_discount * (1 + tax)
  (total_paid - final_total)

theorem jimmy_change :
  change_back 1.65 3.95 4.35 2.80 1.75 150 0.25 0.085 = 100.16 :=
by
  sorry

end jimmy_change_l269_269253


namespace difference_between_q_and_r_l269_269392

-- Define the variables for shares with respect to the common multiple x
def p_share (x : Nat) : Nat := 3 * x
def q_share (x : Nat) : Nat := 7 * x
def r_share (x : Nat) : Nat := 12 * x

-- Given condition: The difference between q's share and p's share is Rs. 4000
def condition_1 (x : Nat) : Prop := (q_share x - p_share x = 4000)

-- Define the theorem to prove the difference between r and q's share is Rs. 5000
theorem difference_between_q_and_r (x : Nat) (h : condition_1 x) : r_share x - q_share x = 5000 :=
by
  sorry

end difference_between_q_and_r_l269_269392


namespace book_price_is_correct_l269_269147

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage
def discount_percentage : ℝ := 0.30

-- Calculate the CD cost
def cd_cost : ℝ := album_cost * (1 - discount_percentage)

-- Define the additional cost of the book over the CD
def book_cd_diff : ℝ := 4

-- Calculate the book cost
def book_cost : ℝ := cd_cost + book_cd_diff

-- State the proposition to be proved
theorem book_price_is_correct : book_cost = 18 := by
  -- Provide the details of the calculations (optionally)
  sorry

end book_price_is_correct_l269_269147


namespace profit_percentage_is_10_percent_l269_269376

theorem profit_percentage_is_10_percent
  (market_price_per_pen : ℕ)
  (retailer_buys_40_pens_for_36_price : 40 * market_price_per_pen = 36 * market_price_per_pen)
  (discount_percentage : ℕ)
  (selling_price_with_discount : ℕ) :
  discount_percentage = 1 →
  selling_price_with_discount = market_price_per_pen - (market_price_per_pen / 100) →
  (selling_price_with_discount * 40 - 36 * market_price_per_pen) / (36 * market_price_per_pen) * 100 = 10 :=
by
  sorry

end profit_percentage_is_10_percent_l269_269376


namespace catherine_friends_count_l269_269759

/-
Definition and conditions:
- An equal number of pencils and pens, totaling 60 each.
- Gave away 8 pens and 6 pencils to each friend.
- Left with 22 pens and pencils.
Proof:
- The number of friends she gave pens and pencils to equals 7.
-/
theorem catherine_friends_count :
  ∀ (pencils pens friends : ℕ),
  pens = 60 →
  pencils = 60 →
  (pens + pencils) - friends * (8 + 6) = 22 →
  friends = 7 :=
sorry

end catherine_friends_count_l269_269759


namespace line_intersects_circle_and_focus_condition_l269_269699

variables {x y k : ℝ}

/-- The line l intersects the circle x^2 + y^2 + 2x - 4y + 1 = 0 at points A and B. If the midpoint of the chord AB is the focus of the parabola x^2 = 4y, then prove that the equation of the line l is x - y + 1 = 0. -/
theorem line_intersects_circle_and_focus_condition :
  (∃ l : ℝ → ℝ, (∀ x y : ℝ, l x = y) ∧
  (∀ A B : ℝ × ℝ, ∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (0, 1)) ∧
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧
  x^2 = 4*y ) → 
  (∀ x y : ℝ, x - y + 1 = 0) :=
sorry

end line_intersects_circle_and_focus_condition_l269_269699


namespace difference_of_squares_example_product_calculation_factorization_by_completing_square_l269_269828

/-
  Theorem: The transformation in the step \(195 \times 205 = 200^2 - 5^2\) uses the difference of squares formula.
-/

theorem difference_of_squares_example : 
  (195 * 205 = (200 - 5) * (200 + 5)) ∧ ((200 - 5) * (200 + 5) = 200^2 - 5^2) :=
  sorry

/-
  Theorem: Calculate \(9 \times 11 \times 101 \times 10001\) using a simple method.
-/

theorem product_calculation : 
  9 * 11 * 101 * 10001 = 99999999 :=
  sorry

/-
  Theorem: Factorize \(a^2 - 6a + 8\) using the completing the square method.
-/

theorem factorization_by_completing_square (a : ℝ) :
  a^2 - 6 * a + 8 = (a - 2) * (a - 4) :=
  sorry

end difference_of_squares_example_product_calculation_factorization_by_completing_square_l269_269828


namespace find_f_expression_l269_269909

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_expression (x : ℝ) : f(2 * x + 1) = x + 1 → f(x) = (1/2) * (x + 1) :=
by
  intro h
  sorry

end find_f_expression_l269_269909


namespace root_is_neg_one_then_m_eq_neg_3_l269_269614

theorem root_is_neg_one_then_m_eq_neg_3 (m : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + m = 0 ∧ x = -1) : m = -3 :=
sorry

end root_is_neg_one_then_m_eq_neg_3_l269_269614


namespace circle_diameter_l269_269548

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l269_269548


namespace sum_of_midpoints_l269_269492

theorem sum_of_midpoints {a b c : ℝ} (h : a + b + c = 10) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 10 :=
by
  sorry

end sum_of_midpoints_l269_269492


namespace sequence_comparison_l269_269217

-- Define arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define geometric sequence
def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ (∀ n, b (n + 1) = b n * q) ∧ (∀ i, i ≥ 1 → b i > 0)

-- Main theorem to prove
theorem sequence_comparison {a b : ℕ → ℝ} (q : ℝ) (h_a_arith : arithmetic_sequence a) 
  (h_b_geom : geometric_sequence b q) (h_eq_1 : a 1 = b 1) (h_eq_11 : a 11 = b 11) :
  a 6 > b 6 :=
sorry

end sequence_comparison_l269_269217


namespace athletes_meeting_time_and_overtakes_l269_269717

-- Define the constants for the problem
noncomputable def track_length : ℕ := 400
noncomputable def speed1 : ℕ := 155
noncomputable def speed2 : ℕ := 200
noncomputable def speed3 : ℕ := 275

-- The main theorem for the problem statement
theorem athletes_meeting_time_and_overtakes :
  ∃ (t : ℚ) (n_overtakes : ℕ), 
  (t = 80 / 3) ∧
  (n_overtakes = 13) ∧
  (∀ n : ℕ, n * (track_length / 45) = t) ∧
  (∀ k : ℕ, k * (track_length / 120) = t) ∧
  (∀ m : ℕ, m * (track_length / 75) = t) := 
sorry

end athletes_meeting_time_and_overtakes_l269_269717


namespace sum_four_digit_integers_l269_269351

theorem sum_four_digit_integers : 
  ∑ k in Finset.range (9999 - 1000 + 1), (k + 1000) = 49495500 := 
by
  sorry

end sum_four_digit_integers_l269_269351


namespace prob_even_sum_correct_l269_269754

noncomputable def is_even_sum (a b c d e f : ℕ) : Prop :=
  ((a * 100 + b * 10 + c) + (d * 100 + e * 10 + f)) % 2 = 0

noncomputable def prob_even_sum : ℚ :=
  let L := [1, 2, 3, 4, 5, 6] in
  let perms := finset.univ.powerset.filter (λ x, x.card = 3) in
  let even_events := perms.filter (λ abc, is_even_sum abc.1.head abc.1.nth 1 abc.1.nth 2 abc.2.head abc.2.nth 1 abc.2.nth 2) in
  ↑(even_events.card) / ↑(perms.card)

theorem prob_even_sum_correct : prob_even_sum = 9 / 10 :=
by
  sorry

end prob_even_sum_correct_l269_269754


namespace modular_inverse_3_mod_17_l269_269773

theorem modular_inverse_3_mod_17 : ∃ a : ℤ, 0 ≤ a ∧ a < 17 ∧ 3 * a ≡ 1 [MOD 17] := 
by
  use 6
  split; norm_num
  split; norm_num
  exact Nat.ModEq.refl 1

end modular_inverse_3_mod_17_l269_269773


namespace prime_fraction_identity_l269_269410

theorem prime_fraction_identity : ∀ (p q : ℕ),
  Prime p → Prime q → p = 2 → q = 2 →
  (pq + p^p + q^q) / (p + q) = 3 :=
by
  intros p q hp hq hp2 hq2
  sorry

end prime_fraction_identity_l269_269410


namespace samuel_distance_from_hotel_l269_269831

def total_distance (speed1 time1 speed2 time2 : ℕ) : ℕ :=
  (speed1 * time1) + (speed2 * time2)

def distance_remaining (total_distance hotel_distance : ℕ) : ℕ :=
  hotel_distance - total_distance

theorem samuel_distance_from_hotel : 
  ∀ (speed1 time1 speed2 time2 hotel_distance : ℕ),
    speed1 = 50 → time1 = 3 → speed2 = 80 → time2 = 4 → hotel_distance = 600 →
    distance_remaining (total_distance speed1 time1 speed2 time2) hotel_distance = 130 :=
by
  intros speed1 time1 speed2 time2 hotel_distance h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  have hdist : total_distance 50 3 80 4 = 470 := by
    simp [total_distance]
  rw [hdist]
  simp [distance_remaining]
  norm_num
  sorry

end samuel_distance_from_hotel_l269_269831


namespace no_positive_integer_n_satisfies_l269_269417

theorem no_positive_integer_n_satisfies :
  ¬∃ (n : ℕ), (1000 ≤ n / 5 ∧ n / 5 ≤ 9999) ∧ (1000 ≤ 5 * n ∧ 5 * n ≤ 9999) := by
  sorry

end no_positive_integer_n_satisfies_l269_269417


namespace least_positive_four_digit_multiple_of_6_l269_269130

theorem least_positive_four_digit_multiple_of_6 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 6 = 0 → n ≤ m) := 
sorry

end least_positive_four_digit_multiple_of_6_l269_269130


namespace sin_600_eq_neg_sqrt3_div2_l269_269191

theorem sin_600_eq_neg_sqrt3_div2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by sorry

end sin_600_eq_neg_sqrt3_div2_l269_269191


namespace cost_of_book_l269_269150

theorem cost_of_book (cost_album : ℝ) (discount_rate : ℝ) (cost_CD : ℝ) (cost_book : ℝ) 
  (h1 : cost_album = 20)
  (h2 : discount_rate = 0.30)
  (h3 : cost_CD = cost_album * (1 - discount_rate))
  (h4 : cost_book = cost_CD + 4) :
  cost_book = 18 := by
  sorry

end cost_of_book_l269_269150


namespace incorrect_intersections_l269_269605

theorem incorrect_intersections :
  (∃ x, (x = x ∧ x = Real.sqrt (x + 2)) ↔ x = 1 ∨ x = 2) →
  (∃ x, (x^2 - 3 * x + 2 = 2 ∧ x = 2) ↔ x = 1 ∨ x = 2) →
  (∃ x, (Real.sin x = 3 * x - 4 ∧ x = 2) ↔ x = 1 ∨ x = 2) → False :=
by {
  sorry
}

end incorrect_intersections_l269_269605


namespace neg_four_fifth_less_neg_two_third_l269_269180

theorem neg_four_fifth_less_neg_two_third : (-4 : ℚ) / 5 < (-2 : ℚ) / 3 :=
  sorry

end neg_four_fifth_less_neg_two_third_l269_269180


namespace tangent_line_at_one_m_positive_if_equal_vals_ineq_if_equal_vals_l269_269046

noncomputable def f (x m : ℝ) : ℝ := (Real.exp (x - 1) - 0.5 * x^2 + x - m * Real.log x)

theorem tangent_line_at_one (m : ℝ) :
  ∃ (y : ℝ → ℝ), (∀ x, y x = (1 - m) * x + m + 0.5) ∧ y 1 = f 1 m ∧ (tangent_slope : ℝ) = 1 - m ∧
    ∀ x, y x = f x m + y 0 :=
sorry

theorem m_positive_if_equal_vals (m x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ m = f x₂ m) :
  m > 0 :=
sorry

theorem ineq_if_equal_vals (m x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ m = f x₂ m) :
  2 * m > Real.exp (Real.log x₁ + Real.log x₂) :=
sorry

end tangent_line_at_one_m_positive_if_equal_vals_ineq_if_equal_vals_l269_269046


namespace expectation_fair_coin_5_tosses_l269_269719

noncomputable def fairCoinExpectation (n : ℕ) : ℚ :=
  n * (1/2)

theorem expectation_fair_coin_5_tosses :
  fairCoinExpectation 5 = 5 / 2 :=
by
  sorry

end expectation_fair_coin_5_tosses_l269_269719


namespace max_volume_solid_l269_269107

-- Define volumes of individual cubes
def cube_volume (side: ℕ) : ℕ := side * side * side

-- Calculate the total number of cubes in the solid
def total_cubes (base_layer : ℕ) (second_layer : ℕ) : ℕ := base_layer + second_layer

-- Define the base layer and second layer cubes
def base_layer_cubes : ℕ := 4 * 4
def second_layer_cubes : ℕ := 2 * 2

-- Define the total volume of the solid
def total_volume (side_length : ℕ) (base_layer : ℕ) (second_layer : ℕ) : ℕ := 
  total_cubes base_layer second_layer * cube_volume side_length

theorem max_volume_solid :
  total_volume 3 base_layer_cubes second_layer_cubes = 540 := by
  sorry

end max_volume_solid_l269_269107


namespace trigonometric_identity_l269_269185

theorem trigonometric_identity (h1 : cos (70 * Real.pi / 180) ≠ 0) (h2 : sin (70 * Real.pi / 180) ≠ 0) :
  (1 / cos (70 * Real.pi / 180) - (Real.sqrt 3) / sin (70 * Real.pi / 180)) = Real.cot (20 * Real.pi / 180) - 2 :=
by
  sorry

end trigonometric_identity_l269_269185


namespace quadratic_inequality_solution_l269_269204

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 + m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
sorry

end quadratic_inequality_solution_l269_269204


namespace greatest_integer_less_than_neg_19_over_5_l269_269721

theorem greatest_integer_less_than_neg_19_over_5 : 
  let x := - (19 / 5 : ℚ) in
  ∃ n : ℤ, n < x ∧ (∀ m : ℤ, m < x → m ≤ n) := 
by 
  let x : ℚ := - (19 / 5)
  existsi (-4 : ℤ) 
  split 
  · norm_num 
    linarith
  · intros m hm 
    linarith

end greatest_integer_less_than_neg_19_over_5_l269_269721


namespace extra_cost_from_online_purchase_l269_269505

-- Define the in-store price
def inStorePrice : ℝ := 150.00

-- Define the online payment and processing fee
def onlinePayment : ℝ := 35.00
def processingFee : ℝ := 12.00

-- Calculate the total online cost
def totalOnlineCost : ℝ := (4 * onlinePayment) + processingFee

-- Calculate the difference in cents
def differenceInCents : ℝ := (totalOnlineCost - inStorePrice) * 100

-- The proof statement
theorem extra_cost_from_online_purchase : differenceInCents = 200 :=
by
  -- Proof steps go here
  sorry

end extra_cost_from_online_purchase_l269_269505


namespace min_average_annual_growth_rate_l269_269053

theorem min_average_annual_growth_rate (M : ℝ) (x : ℝ) (h : M * (1 + x)^2 = 2 * M) : x = Real.sqrt 2 - 1 :=
by
  sorry

end min_average_annual_growth_rate_l269_269053


namespace part1_part2_l269_269027

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269027


namespace unique_solution_l269_269402

def my_operation (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem unique_solution :
  ∃! y : ℝ, my_operation 4 y = 15 ∧ y = -1/2 :=
by 
  sorry

end unique_solution_l269_269402


namespace no_permutation_exists_l269_269815

open Function Set

theorem no_permutation_exists (f : ℕ → ℕ) (h : ∀ n m : ℕ, f n = f m ↔ n = m) :
  ¬ ∃ n : ℕ, (Finset.range n).image f = Finset.range n :=
by
  sorry

end no_permutation_exists_l269_269815


namespace part1_part2_l269_269033

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269033


namespace num_perfect_square_factors_l269_269758

def prime_factors_9600 (n : ℕ) : Prop :=
  n = 9600

theorem num_perfect_square_factors (n : ℕ) (h : prime_factors_9600 n) : 
  let cond := h
  (n = 9600) → 9600 = 2^6 * 5^2 * 3^1 → (∃ factors_count: ℕ, factors_count = 8) := by 
  sorry

end num_perfect_square_factors_l269_269758


namespace number_of_boxes_needed_l269_269879

theorem number_of_boxes_needed 
  (students : ℕ) (cookies_per_student : ℕ) (cookies_per_box : ℕ) 
  (total_students : students = 134) 
  (cookies_each : cookies_per_student = 7) 
  (cookies_in_box : cookies_per_box = 28) 
  (total_cookies : students * cookies_per_student = 938)
  : Nat.ceil (938 / 28) = 34 := 
by
  sorry

end number_of_boxes_needed_l269_269879


namespace tan_A_plus_C_eq_neg_sqrt3_l269_269452

theorem tan_A_plus_C_eq_neg_sqrt3
  (A B C : Real)
  (hSum : A + B + C = Real.pi)
  (hArithSeq : 2 * B = A + C)
  (hTriangle : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi) :
  Real.tan (A + C) = -Real.sqrt 3 := by
  sorry

end tan_A_plus_C_eq_neg_sqrt3_l269_269452


namespace number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_l269_269061

def num_ways_to_make_125_quacks_using_coins : ℕ :=
  have h : ∃ (a b c d : ℕ), a + 5 * b + 25 * c + 125 * d = 125 := sorry
  82

theorem number_of_ways_to_make_125_quacks_using_1_5_25_125_coins : num_ways_to_make_125_quacks_using_coins = 82 := 
  sorry

end number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_l269_269061


namespace negation_of_P_l269_269922

-- Define the proposition P
def P (x : ℝ) : Prop := x^2 = 1 → x = 1

-- Define the negation of the proposition P
def neg_P (x : ℝ) : Prop := x^2 ≠ 1 → x ≠ 1

theorem negation_of_P (x : ℝ) : ¬P x ↔ neg_P x := by
  sorry

end negation_of_P_l269_269922


namespace sum_of_coordinates_of_B_l269_269823

theorem sum_of_coordinates_of_B (x y : ℕ) (hM : (2 * 6 = x + 10) ∧ (2 * 8 = y + 8)) :
    x + y = 10 :=
sorry

end sum_of_coordinates_of_B_l269_269823


namespace fraction_computation_l269_269588

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l269_269588


namespace tangent_line_at_point_P_l269_269914

-- Define the curve y = x^3 
def curve (x : ℝ) : ℝ := x ^ 3

-- Define the point P(1,1)
def pointP : ℝ × ℝ := (1, 1)

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x ^ 2

-- Define the tangent line equation we need to prove
def tangent_line (x y : ℝ) : Prop := 3 * x - y - 2 = 0

theorem tangent_line_at_point_P :
  ∀ (x y : ℝ), 
  pointP = (1, 1) ∧ curve 1 = 1 ∧ curve_derivative 1 = 3 → 
  tangent_line 1 1 := 
by
  intros x y h
  sorry

end tangent_line_at_point_P_l269_269914


namespace sarah_score_l269_269277

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end sarah_score_l269_269277


namespace false_props_count_is_3_l269_269455

-- Define the propositions and their inferences

noncomputable def original_prop (m n : ℝ) : Prop := m > -n → m^2 > n^2
noncomputable def contrapositive (m n : ℝ) : Prop := ¬(m^2 > n^2) → ¬(m > -n)
noncomputable def inverse (m n : ℝ) : Prop := m^2 > n^2 → m > -n
noncomputable def negation (m n : ℝ) : Prop := ¬(m > -n → m^2 > n^2)

-- The main statement to be proved
theorem false_props_count_is_3 (m n : ℝ) : 
  ¬ (original_prop m n) ∧ ¬ (contrapositive m n) ∧ ¬ (inverse m n) ∧ ¬ (negation m n) →
  (3 = 3) :=
by
  sorry

end false_props_count_is_3_l269_269455


namespace opposite_negative_nine_l269_269706

theorem opposite_negative_nine : 
  (∃ (y : ℤ), -9 + y = 0 ∧ y = 9) :=
by sorry

end opposite_negative_nine_l269_269706


namespace negation_exists_eq_forall_l269_269852

theorem negation_exists_eq_forall (h : ¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) : ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := 
by
  sorry

end negation_exists_eq_forall_l269_269852


namespace cost_per_first_30_kg_is_10_l269_269577

-- Definitions of the constants based on the conditions
def cost_per_33_kg (p q : ℝ) : Prop := 30 * p + 3 * q = 360
def cost_per_36_kg (p q : ℝ) : Prop := 30 * p + 6 * q = 420
def cost_per_25_kg (p : ℝ) : Prop := 25 * p = 250

-- The statement we want to prove
theorem cost_per_first_30_kg_is_10 (p q : ℝ) 
  (h1 : cost_per_33_kg p q)
  (h2 : cost_per_36_kg p q)
  (h3 : cost_per_25_kg p) : 
  p = 10 :=
sorry

end cost_per_first_30_kg_is_10_l269_269577


namespace smallest_root_of_unity_l269_269132

open Complex

theorem smallest_root_of_unity (z : ℂ) (h : z^6 - z^3 + 1 = 0) : ∃ k : ℕ, k < 18 ∧ z = exp (2 * pi * I * k / 18) :=
by
  sorry

end smallest_root_of_unity_l269_269132


namespace student_chose_121_l269_269387

theorem student_chose_121 (x : ℤ) (h : 2 * x - 140 = 102) : x = 121 := by
  sorry

end student_chose_121_l269_269387


namespace jack_last_10_shots_made_l269_269168

theorem jack_last_10_shots_made (initial_shots : ℕ) (initial_percentage : ℚ)
  (additional_shots : ℕ) (new_percentage : ℚ)
  (initial_successful_shots : initial_shots * initial_percentage = 18)
  (total_shots : initial_shots + additional_shots = 40)
  (total_successful_shots : (initial_shots + additional_shots) * new_percentage = 25) :
  ∃ x : ℕ, x = 7 := by
sorry

end jack_last_10_shots_made_l269_269168


namespace theater_loss_l269_269560

/-- 
A movie theater has a total capacity of 50 people and charges $8 per ticket.
On a Tuesday night, they only sold 24 tickets. 
Prove that the revenue lost by not selling out is $208.
-/
theorem theater_loss 
  (capacity : ℕ) 
  (price : ℕ) 
  (sold_tickets : ℕ) 
  (h_cap : capacity = 50) 
  (h_price : price = 8) 
  (h_sold : sold_tickets = 24) : 
  capacity * price - sold_tickets * price = 208 :=
by
  sorry

end theater_loss_l269_269560


namespace amalie_coins_proof_l269_269853

def coins_proof : Prop :=
  ∃ (E A : ℕ),
    (E / A = 10 / 45) ∧
    (E + A = 440) ∧
    ((3 / 4) * A = 270) ∧
    (A - 270 = 90)

theorem amalie_coins_proof : coins_proof :=
  sorry

end amalie_coins_proof_l269_269853


namespace total_money_shared_l269_269390

theorem total_money_shared (A B C : ℕ) (rA rB rC : ℕ) (bens_share : ℕ) 
  (h_ratio : rA = 2 ∧ rB = 3 ∧ rC = 8)
  (h_ben : B = bens_share)
  (h_bensShareGiven : bens_share = 60) : 
  (rA * (bens_share / rB)) + bens_share + (rC * (bens_share / rB)) = 260 :=
by
  -- sorry to skip the proof
  sorry

end total_money_shared_l269_269390


namespace circle_diameter_problem_circle_diameter_l269_269539

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l269_269539


namespace opposite_negative_nine_l269_269705

theorem opposite_negative_nine : 
  (∃ (y : ℤ), -9 + y = 0 ∧ y = 9) :=
by sorry

end opposite_negative_nine_l269_269705


namespace range_of_a_l269_269242

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) : a ≥ 1 :=
sorry

end range_of_a_l269_269242


namespace brochures_multiple_of_6_l269_269940

theorem brochures_multiple_of_6 (n : ℕ) (P : ℕ) (B : ℕ) 
  (hP : P = 12) (hn : n = 6) : ∃ k : ℕ, B = 6 * k := 
sorry

end brochures_multiple_of_6_l269_269940


namespace even_coefficients_count_l269_269786

open Nat

theorem even_coefficients_count :
  let f : ℕ → ℕ := λ k, if (binomial 2008 k) % 2 = 0 then 1 else 0 in
  (List.range 2009).sum (λ k => f k) = 127 :=
by
  let f := λ k, if (binomial 2008 k) % 2 = 0 then 1 else 0
  have odd_count := 128 - 1
  have even_count := 2009 - odd_count
  exact even_count
  sorry

end even_coefficients_count_l269_269786


namespace value_range_of_f_l269_269309

-- Define the function f(x) = 2x - x^2
def f (x : ℝ) : ℝ := 2 * x - x^2

-- State the theorem with the given conditions and prove the correct answer
theorem value_range_of_f :
  (∀ y : ℝ, -3 ≤ y ∧ y ≤ 1 → ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = y) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → -3 ≤ f x ∧ f x ≤ 1) :=
by
  sorry

end value_range_of_f_l269_269309


namespace tan_sum_eq_tan_product_l269_269573

theorem tan_sum_eq_tan_product {α β γ : ℝ} 
  (h_sum : α + β + γ = π) : 
    Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ :=
by
  sorry

end tan_sum_eq_tan_product_l269_269573


namespace samuel_distance_from_hotel_l269_269832

def total_distance (speed1 time1 speed2 time2 : ℕ) : ℕ :=
  (speed1 * time1) + (speed2 * time2)

def distance_remaining (total_distance hotel_distance : ℕ) : ℕ :=
  hotel_distance - total_distance

theorem samuel_distance_from_hotel : 
  ∀ (speed1 time1 speed2 time2 hotel_distance : ℕ),
    speed1 = 50 → time1 = 3 → speed2 = 80 → time2 = 4 → hotel_distance = 600 →
    distance_remaining (total_distance speed1 time1 speed2 time2) hotel_distance = 130 :=
by
  intros speed1 time1 speed2 time2 hotel_distance h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  have hdist : total_distance 50 3 80 4 = 470 := by
    simp [total_distance]
  rw [hdist]
  simp [distance_remaining]
  norm_num
  sorry

end samuel_distance_from_hotel_l269_269832


namespace ratio_of_wins_l269_269121

-- Definitions based on conditions
def W1 : ℕ := 15  -- Number of wins before first loss
def L : ℕ := 2    -- Total number of losses
def W2 : ℕ := 30 - W1  -- Calculate W2 based on W1 and total wins being 28 more than losses

-- Theorem statement: Prove the ratio of wins after her first loss to wins before her first loss is 1:1
theorem ratio_of_wins (h : W1 = 15 ∧ L = 2) : W2 / W1 = 1 := by
  sorry

end ratio_of_wins_l269_269121


namespace part_I_part_II_l269_269797

variables (a b c x y : ℝ) (A B C : ℝ)
variables (m n : EuclideanSpace ℝ (Fin 2)) -- declaring m, n as vectors in 2D Euclidean space

-- Let triangle ABC have sides opposite to angles A, B, and C as a, b, and c respectively
-- Given vectors m and n as indicated:
def m := EuclideanSpace ℝ (Fin 2) := λ (i : Fin 2), if i = 0 then 2 * a + c else b
def n := EuclideanSpace ℝ (Fin 2) := λ (i : Fin 2), if i = 0 then Real.cos B else Real.cos C

-- Condition that m dot n is zero
def dot_product_condition := ((2 * a + c) * Real.cos B + b * Real.cos C = 0)

-- Part (I): Determine angle B
theorem part_I (h : dot_product_condition a b c B C):
  B = 2 * Real.pi / 3 := by
  sorry

-- Part (II): Functional relationship between x and y, minimizing area
theorem part_II (BD_one : BD = 1) (angle_BD : B = 2 * Real.pi / 3) (x_pos : x > 1) :
  y = x / (x - 1) ∧ (x = 2 → arena_triangle_ABC a b c B = Real.sqrt 3) := by
  sorry

end part_I_part_II_l269_269797


namespace find_a_2016_l269_269904

-- Define the sequence a_n and its sum S_n
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom S_n_eq : ∀ n : ℕ, S n + (1 + (2 / n)) * a n = 4
axiom a_1_eq : a 1 = 1
axiom a_rec : ∀ n : ℕ, n ≥ 2 → a n = (n / (2 * (n - 1))) * a (n - 1)

-- The theorem to prove
theorem find_a_2016 : a 2016 = 2016 / 2^2015 := by
  sorry

end find_a_2016_l269_269904


namespace algebraic_expression_value_l269_269900

/-- Given \( x^2 - 5x - 2006 = 0 \), prove that the expression \(\frac{(x-2)^3 - (x-1)^2 + 1}{x-2}\) is equal to 2010. -/
theorem algebraic_expression_value (x : ℝ) (h: x^2 - 5 * x - 2006 = 0) :
  ( (x - 2)^3 - (x - 1)^2 + 1 ) / (x - 2) = 2010 :=
by
  sorry

end algebraic_expression_value_l269_269900


namespace sqrt_x_minus_1_domain_l269_269635

theorem sqrt_x_minus_1_domain (x : ℝ) : (∃ y, y^2 = x - 1) → (x ≥ 1) := 
by 
  sorry

end sqrt_x_minus_1_domain_l269_269635


namespace exists_zero_in_interval_l269_269876

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

theorem exists_zero_in_interval : ∃ c ∈ Set.Ioo 0 (1/2 : ℝ), f c = 0 := by
  -- proof to be filled in
  sorry

end exists_zero_in_interval_l269_269876


namespace no_integer_solutions_for_trapezoid_bases_l269_269843

theorem no_integer_solutions_for_trapezoid_bases :
  ∃ (A h : ℤ) (b1_b2 : ℤ → Prop),
    A = 2800 ∧ h = 80 ∧
    (∀ m n : ℤ, b1_b2 (12 * m) ∧ b1_b2 (12 * n) → (12 * m + 12 * n = 70) → false) :=
by
  sorry

end no_integer_solutions_for_trapezoid_bases_l269_269843


namespace fraction_identity_l269_269599

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l269_269599


namespace range_of_m_l269_269431

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m*x^2 + m*x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end range_of_m_l269_269431


namespace searchlight_reflector_distance_l269_269974

noncomputable def parabola_vertex_distance : Rat :=
  let diameter := 60 -- in cm
  let depth := 40 -- in cm
  let x := 40 -- x-coordinate of the point
  let y := 30 -- y-coordinate of the point
  let p := (y^2) / (2 * x)
  p / 2

theorem searchlight_reflector_distance : parabola_vertex_distance = 45 / 8 := by
  sorry

end searchlight_reflector_distance_l269_269974


namespace probability_of_event_l269_269684

open Set Real

noncomputable def probability_event_interval (x : ℝ) : Prop :=
  1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 3

noncomputable def interval := Icc (0 : ℝ) (3 : ℝ)

noncomputable def event_probability := 1 / 3

theorem probability_of_event :
  ∀ x ∈ interval, probability_event_interval x → (event_probability) = 1 / 3 :=
by
  sorry

end probability_of_event_l269_269684


namespace iron_balls_molded_l269_269394

noncomputable def volume_of_iron_bar (l w h : ℝ) : ℝ :=
  l * w * h

theorem iron_balls_molded (l w h n : ℝ) (volume_of_ball : ℝ) 
  (h_l : l = 12) (h_w : w = 8) (h_h : h = 6) (h_n : n = 10) (h_ball_volume : volume_of_ball = 8) :
  (n * volume_of_iron_bar l w h) / volume_of_ball = 720 :=
by 
  rw [h_l, h_w, h_h, h_n, h_ball_volume]
  rw [volume_of_iron_bar]
  sorry

end iron_balls_molded_l269_269394


namespace point_P_existence_and_distances_l269_269212

variables (a b h : ℝ)

-- Define the conditions under which point P can exist:
def conditions_for_P_existence : Prop :=
  h^2 >= a * b

-- Define the quadratic equation in terms of the distances from P to the bases:
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - h * x + (a * b) / 4 = 0

-- Define the problem statement proving points P can exist and satisfy given conditions:
theorem point_P_existence_and_distances
  (h_pos : h > 0)
  (a_pos : a > 0)
  (b_pos : b > 0) :
  conditions_for_P_existence a b h ↔ ∃ (PM PN : ℝ), quadratic_equation a b h PM ∧ quadratic_equation a b h PN ∧ PM + PN = h :=
by
  sorry

end point_P_existence_and_distances_l269_269212


namespace P_eq_Q_at_x_l269_269942

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 2
def Q (x : ℝ) : ℝ := 0

theorem P_eq_Q_at_x :
  ∃ x : ℝ, P x = Q x ∧ x = 1 :=
by
  sorry

end P_eq_Q_at_x_l269_269942


namespace infinite_sqrt_eval_l269_269890

theorem infinite_sqrt_eval {x : ℝ} (h : x = Real.sqrt (3 - x)) : 
  x = (-1 + Real.sqrt 13) / 2 :=
by sorry

end infinite_sqrt_eval_l269_269890


namespace solve_identity_l269_269412

theorem solve_identity (x : ℝ) (a b p q : ℝ)
  (h : (6 * x + 1) / (6 * x ^ 2 + 19 * x + 15) = a / (x - p) + b / (x - q)) :
  a = -1 ∧ b = 2 ∧ p = -3/4 ∧ q = -5/3 :=
by
  sorry

end solve_identity_l269_269412


namespace exist_indices_eq_l269_269952

theorem exist_indices_eq (p q n : ℕ) (x : ℕ → ℤ) 
    (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_n : 0 < n) 
    (h_pq_n : p + q < n) 
    (h_x0 : x 0 = 0) 
    (h_xn : x n = 0) 
    (h_step : ∀ i, 1 ≤ i ∧ i ≤ n → (x i - x (i - 1) = p ∨ x i - x (i - 1) = -q)) :
    ∃ (i j : ℕ), i < j ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
sorry

end exist_indices_eq_l269_269952


namespace part1_part2_l269_269028

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269028


namespace ratio_of_division_of_chord_l269_269487

theorem ratio_of_division_of_chord (R AP PB O: ℝ) (radius_given: R = 11) (chord_length: AP + PB = 18) (point_distance: O = 7) : 
  (AP / PB = 2 ∨ PB / AP = 2) :=
by 
  -- Proof goes here, to be filled in later
  sorry

end ratio_of_division_of_chord_l269_269487


namespace inequality_solution_l269_269482

noncomputable def inequality (x : ℝ) : Prop :=
  (12 * x ^ 3 + 24 * x ^ 2 - 75 * x - 3) / ((3 * x - 4) * (x + 5)) < 6

theorem inequality_solution (x : ℝ) : inequality x ↔ (x > -5 ∧ x < (4 / 3)) :=
by
  sorry

end inequality_solution_l269_269482


namespace card_2015_in_box_3_l269_269713

-- Define the pattern function for placing cards
def card_placement (n : ℕ) : ℕ :=
  let cycle_length := 12
  let cycle_pos := (n - 1) % cycle_length + 1
  if cycle_pos ≤ 7 then cycle_pos
  else 14 - cycle_pos

-- Define the theorem to prove the position of the 2015th card
theorem card_2015_in_box_3 : card_placement 2015 = 3 := by
  -- sorry is used to skip the proof
  sorry

end card_2015_in_box_3_l269_269713


namespace houses_before_boom_l269_269176

theorem houses_before_boom (T B H : ℕ) (hT : T = 2000) (hB : B = 574) : H = 1426 := by
  sorry

end houses_before_boom_l269_269176


namespace hyperbola_distance_condition_l269_269955

open Real

theorem hyperbola_distance_condition (a b c x: ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
    (h_dist : abs (b^4 / a^2 / (a - c)) < a + sqrt (a^2 + b^2)) :
    0 < b / a ∧ b / a < 1 :=
by
  sorry

end hyperbola_distance_condition_l269_269955


namespace random_events_count_is_five_l269_269623

-- Definitions of the events in the conditions
def event1 := "Classmate A successfully runs for class president"
def event2 := "Stronger team wins in a game between two teams"
def event3 := "A school has a total of 998 students, and at least three students share the same birthday"
def event4 := "If sets A, B, and C satisfy A ⊆ B and B ⊆ C, then A ⊆ C"
def event5 := "In ancient times, a king wanted to execute a painter. Secretly, he wrote 'death' on both slips of paper, then let the painter draw a 'life or death' slip. The painter drew a death slip"
def event6 := "It snows in July"
def event7 := "Choosing any two numbers from 1, 3, 9, and adding them together results in an even number"
def event8 := "Riding through 10 intersections, all lights encountered are red"

-- Tally up the number of random events
def is_random_event (event : String) : Bool :=
  event = event1 ∨
  event = event2 ∨
  event = event3 ∨
  event = event6 ∨
  event = event8

def count_random_events (events : List String) : Nat :=
  (events.map (λ event => if is_random_event event then 1 else 0)).sum

-- List of events
def events := [event1, event2, event3, event4, event5, event6, event7, event8]

-- Theorem statement
theorem random_events_count_is_five : count_random_events events = 5 :=
  by
    sorry

end random_events_count_is_five_l269_269623


namespace negation_equiv_l269_269975

theorem negation_equiv (x : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) := 
by 
  sorry

end negation_equiv_l269_269975


namespace fraction_computation_l269_269586

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l269_269586


namespace probability_exactly_three_heads_in_seven_tosses_l269_269550

def combinations (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) : ℚ :=
  (combinations n k) / (2^n : ℚ)

theorem probability_exactly_three_heads_in_seven_tosses :
  binomial_probability 7 3 = 35 / 128 := 
by 
  sorry

end probability_exactly_three_heads_in_seven_tosses_l269_269550


namespace opposite_neg_9_l269_269709

theorem opposite_neg_9 : 
  ∃ y : Int, -9 + y = 0 ∧ y = 9 :=
by
  sorry

end opposite_neg_9_l269_269709


namespace part1_part2_l269_269036

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269036


namespace survey_blue_percentage_l269_269059

-- Conditions
def red (r : ℕ) := r = 70
def blue (b : ℕ) := b = 80
def green (g : ℕ) := g = 50
def yellow (y : ℕ) := y = 70
def orange (o : ℕ) := o = 30

-- Total responses sum
def total_responses (r b g y o : ℕ) := r + b + g + y + o = 300

-- Percentage of blue respondents
def blue_percentage (b total : ℕ) := (b : ℚ) / total * 100 = 26 + 2/3

-- Theorem statement
theorem survey_blue_percentage (r b g y o : ℕ) (H_red : red r) (H_blue : blue b) (H_green : green g) (H_yellow : yellow y) (H_orange : orange o) (H_total : total_responses r b g y o) : blue_percentage b 300 :=
by {
  sorry
}

end survey_blue_percentage_l269_269059


namespace trigonometric_expression_value_l269_269183

theorem trigonometric_expression_value :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 :=
by
  sorry

end trigonometric_expression_value_l269_269183


namespace prove_OH_squared_l269_269812

noncomputable def circumcenter_orthocenter_identity (a b c R : ℝ) (H O : ℝ) (h1 : R = 10) (h2 : a^2 + b^2 + c^2 = 50) : ℝ :=
  9 * R^2 - (a^2 + b^2 + c^2)

theorem prove_OH_squared :
  let a b c R : ℝ := 10
  let H O : ℝ := sorry
  (9 * 10^2 - (a^2 + b^2 + c^2)) = 850 :=
begin
  have h1 : R = 10 := rfl,
  have h2 : a^2 + b^2 + c^2 = 50 := sorry,
  rw [h1, h2],
  norm_num,
  exact rfl,
end

end prove_OH_squared_l269_269812


namespace sarah_score_l269_269278

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end sarah_score_l269_269278


namespace circle_diameter_l269_269534

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l269_269534


namespace problem_a_problem_b_l269_269994

-- Problem (a)
theorem problem_a (n : ℕ) (hn : n > 0) :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

-- Problem (b)
theorem problem_b (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (coprime_ab : Nat.gcd a b = 1)
  (coprime_ac_or_bc : Nat.gcd c a = 1 ∨ Nat.gcd c b = 1) :
  ∃ᶠ x : ℕ in Filter.atTop, ∃ (y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x^a + y^b = z^c :=
sorry

end problem_a_problem_b_l269_269994


namespace quadratic_graphs_intersect_at_one_point_l269_269435

theorem quadratic_graphs_intersect_at_one_point
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)
  (h_distinct : a1 ≠ a2 ∧ a2 ≠ a3 ∧ a1 ≠ a3)
  (h_intersect_fg : ∃ x₀ : ℝ, (a1 - a2) * x₀^2 + (b1 - b2) * x₀ + (c1 - c2) = 0 ∧ (b1 - b2)^2 - 4 * (a1 - a2) * (c1 - c2) = 0)
  (h_intersect_gh : ∃ x₁ : ℝ, (a2 - a3) * x₁^2 + (b2 - b3) * x₁ + (c2 - c3) = 0 ∧ (b2 - b3)^2 - 4 * (a2 - a3) * (c2 - c3) = 0)
  (h_intersect_fh : ∃ x₂ : ℝ, (a1 - a3) * x₂^2 + (b1 - b3) * x₂ + (c1 - c3) = 0 ∧ (b1 - b3)^2 - 4 * (a1 - a3) * (c1 - c3) = 0) :
  ∃ x : ℝ, (a1 * x^2 + b1 * x + c1 = 0) ∧ (a2 * x^2 + b2 * x + c2 = 0) ∧ (a3 * x^2 + b3 * x + c3 = 0) :=
by
  sorry

end quadratic_graphs_intersect_at_one_point_l269_269435


namespace least_m_plus_n_l269_269654

theorem least_m_plus_n (m n : ℕ) (hmn : Nat.gcd (m + n) 330 = 1) (hm_multiple : m^m % n^n = 0) (hm_not_multiple : ¬ (m % n = 0)) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  m + n = 119 :=
sorry

end least_m_plus_n_l269_269654


namespace proof_problem_l269_269728

/-- 
  Given:
  - r, j, z are Ryan's, Jason's, and Zachary's earnings respectively.
  - Zachary sold 40 games at $5 each.
  - Jason received 30% more money than Zachary.
  - The total amount of money received by all three is $770.
  Prove:
  - Ryan received $50 more than Jason.
--/
def problem_statement : Prop :=
  ∃ (r j z : ℕ), 
    z = 40 * 5 ∧
    j = z + z * 30 / 100 ∧
    r + j + z = 770 ∧ 
    r - j = 50

theorem proof_problem : problem_statement :=
by 
  sorry

end proof_problem_l269_269728


namespace weight_of_b_l269_269486

-- Define the weights of a, b, and c
variables (W_a W_b W_c : ℝ)

-- Define the heights of a, b, and c
variables (h_a h_b h_c : ℝ)

-- Given conditions
axiom average_weight_abc : (W_a + W_b + W_c) / 3 = 45
axiom average_weight_ab : (W_a + W_b) / 2 = 40
axiom average_weight_bc : (W_b + W_c) / 2 = 47
axiom height_condition : h_a + h_c = 2 * h_b
axiom odd_sum_weights : (W_a + W_b + W_c) % 2 = 1

-- Prove that the weight of b is 39 kg
theorem weight_of_b : W_b = 39 :=
by sorry

end weight_of_b_l269_269486


namespace paint_coverage_l269_269313

theorem paint_coverage 
  (width height cost_per_quart money_spent area : ℕ)
  (cover : ℕ → ℕ → ℕ)
  (num_sides quarts_purchased : ℕ)
  (total_area num_quarts : ℕ)
  (sqfeet_per_quart : ℕ) :
  width = 5 
  → height = 4 
  → cost_per_quart = 2 
  → money_spent = 20 
  → num_sides = 2
  → cover width height = area
  → area * num_sides = total_area
  → money_spent / cost_per_quart = quarts_purchased
  → total_area / quarts_purchased = sqfeet_per_quart
  → total_area = 40 
  → quarts_purchased = 10 
  → sqfeet_per_quart = 4 :=
by 
  intros
  sorry

end paint_coverage_l269_269313


namespace solve_equation_l269_269968

theorem solve_equation (x : ℝ) : 2 * x + 17 = 32 - 3 * x → x = 3 := 
by 
  sorry

end solve_equation_l269_269968


namespace complex_division_l269_269219

def imaginary_unit := Complex.I

theorem complex_division :
  (1 - 3 * imaginary_unit) / (2 + imaginary_unit) = -1 / 5 - 7 / 5 * imaginary_unit := by
  sorry

end complex_division_l269_269219


namespace totalNumberOfPeople_l269_269058

def numGirls := 542
def numBoys := 387
def numTeachers := 45
def numStaff := 27

theorem totalNumberOfPeople : numGirls + numBoys + numTeachers + numStaff = 1001 := by
  sorry

end totalNumberOfPeople_l269_269058


namespace trigonometric_expression_value_l269_269182

theorem trigonometric_expression_value :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 :=
by
  sorry

end trigonometric_expression_value_l269_269182


namespace count_indistinguishable_distributions_l269_269232

theorem count_indistinguishable_distributions (balls : ℕ) (boxes : ℕ) (h_balls : balls = 5) (h_boxes : boxes = 4) : 
  ∃ n : ℕ, n = 6 := by
  sorry

end count_indistinguishable_distributions_l269_269232


namespace proof_pyramid_height_proof_dihedral_angle_l269_269112

noncomputable def pyramid_height (a b : ℝ) : ℝ := (real.sqrt (3 * b^2 - a^2)) / (real.sqrt 3)
noncomputable def dihedral_angle (a b : ℝ) : ℝ := 2 * real.arctan (b / real.sqrt (3 * b^2 - a^2))

theorem proof_pyramid_height (a b : ℝ) : 
  pyramid_height a b = (real.sqrt (3 * b^2 - a^2)) / (real.sqrt 3) :=
begin
  rw pyramid_height,
  sorry,  -- Proof to be completed
end

theorem proof_dihedral_angle (a b : ℝ) : 
  dihedral_angle a b = 2 * real.arctan (b / real.sqrt (3 * b^2 - a^2)) :=
begin
  rw dihedral_angle,
  sorry,  -- Proof to be completed
end

end proof_pyramid_height_proof_dihedral_angle_l269_269112


namespace trigonometric_identity_l269_269188

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = (1 / (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180))) :=
by
  sorry

end trigonometric_identity_l269_269188


namespace Tim_is_65_l269_269070

def James_age : Nat := 23
def John_age : Nat := 35
def Tim_age : Nat := 2 * John_age - 5

theorem Tim_is_65 : Tim_age = 65 := by
  sorry

end Tim_is_65_l269_269070


namespace downstream_distance_l269_269113

theorem downstream_distance
    (speed_still_water : ℝ)
    (current_rate : ℝ)
    (travel_time_minutes : ℝ)
    (h_still_water : speed_still_water = 20)
    (h_current_rate : current_rate = 4)
    (h_travel_time : travel_time_minutes = 24) :
    (speed_still_water + current_rate) * (travel_time_minutes / 60) = 9.6 :=
by
  -- Proof goes here
  sorry

end downstream_distance_l269_269113


namespace length_of_arc_AB_proof_area_of_segment_OAB_proof_l269_269210

noncomputable def length_of_arc_AB (r : ℝ) (θ : ℝ) : ℝ :=
  r * θ

noncomputable def area_of_sector_OAB (r : ℝ) (θ : ℝ) : ℝ :=
  0.5 * r^2 * θ

noncomputable def area_of_triangle_OAB (r : ℝ) (sinθ : ℝ) : ℝ :=
  0.5 * r^2 * sinθ

theorem length_of_arc_AB_proof : 
  length_of_arc_AB 6 ((2 / 3) * Real.pi) = 4 * Real.pi := 
by
  sorry

theorem area_of_segment_OAB_proof : 
  area_of_sector_OAB 6 ((2 / 3) * Real.pi) - area_of_triangle_OAB 6 (Real.sin (2 * Real.pi / 3)) = 12 * Real.pi - 9 * Real.sqrt 3 :=
by 
  sorry

end length_of_arc_AB_proof_area_of_segment_OAB_proof_l269_269210


namespace find_m_l269_269049

theorem find_m (x y m : ℝ) (h1 : x = 1) (h2 : y = 3) (h3 : m * x - y = 3) : m = 6 := 
by
  sorry

end find_m_l269_269049


namespace fraction_computation_l269_269589

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l269_269589


namespace athletes_meeting_time_and_overtakes_l269_269718

-- Define the constants for the problem
noncomputable def track_length : ℕ := 400
noncomputable def speed1 : ℕ := 155
noncomputable def speed2 : ℕ := 200
noncomputable def speed3 : ℕ := 275

-- The main theorem for the problem statement
theorem athletes_meeting_time_and_overtakes :
  ∃ (t : ℚ) (n_overtakes : ℕ), 
  (t = 80 / 3) ∧
  (n_overtakes = 13) ∧
  (∀ n : ℕ, n * (track_length / 45) = t) ∧
  (∀ k : ℕ, k * (track_length / 120) = t) ∧
  (∀ m : ℕ, m * (track_length / 75) = t) := 
sorry

end athletes_meeting_time_and_overtakes_l269_269718


namespace nested_radical_solution_l269_269893

theorem nested_radical_solution : 
  (∃ x : ℝ, (x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2)) := 
begin 
  use (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))),
  sorry
end

end nested_radical_solution_l269_269893


namespace harry_pencils_remaining_l269_269172

def num_pencils_anna : ℕ := 50
def num_pencils_harry_initial := 2 * num_pencils_anna
def num_pencils_lost_harry := 19

def pencils_left_harry (pencils_anna : ℕ) (pencils_harry_initial : ℕ) (pencils_lost : ℕ) : ℕ :=
  pencils_harry_initial - pencils_lost

theorem harry_pencils_remaining : pencils_left_harry num_pencils_anna num_pencils_harry_initial num_pencils_lost_harry = 81 :=
by
  sorry

end harry_pencils_remaining_l269_269172


namespace count_ordered_pairs_l269_269416

open Int

theorem count_ordered_pairs : 
  ∑ y in Finset.range 199, ∑ x in Finset.Ico (y+1) 201, 
    ((x % y = 0) ∧ ((x + 2) % (y + 2) = 0)) → 
    Finset.card (Finset.Ico (y + 1, 201)) = ∑ y in Finset.range 199, 
    Nat.floor ((200 - y : ℚ) / ((y * (y + 2) : ℚ))) :=
by condition
sorry

end count_ordered_pairs_l269_269416


namespace meaningful_square_root_l269_269637

theorem meaningful_square_root (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 := 
by {
  -- This part would contain the proof
  sorry
}

end meaningful_square_root_l269_269637


namespace circle_diameter_l269_269542

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l269_269542


namespace cost_of_gravelling_the_path_l269_269990

-- Define the problem conditions
def plot_length : ℝ := 110
def plot_width : ℝ := 65
def path_width : ℝ := 2.5
def cost_per_sq_meter : ℝ := 0.70

-- Define the dimensions of the grassy area without the path
def grassy_length : ℝ := plot_length - 2 * path_width
def grassy_width : ℝ := plot_width - 2 * path_width

-- Define the area of the entire plot and the grassy area without the path
def area_entire_plot : ℝ := plot_length * plot_width
def area_grassy_area : ℝ := grassy_length * grassy_width

-- Define the area of the path
def area_path : ℝ := area_entire_plot - area_grassy_area

-- Define the cost of gravelling the path
def cost_gravelling_path : ℝ := area_path * cost_per_sq_meter

-- State the theorem
theorem cost_of_gravelling_the_path : cost_gravelling_path = 595 := 
by
  -- The proof is omitted
  sorry

end cost_of_gravelling_the_path_l269_269990


namespace candy_bar_profit_l269_269160

theorem candy_bar_profit
  (bars_bought : ℕ)
  (cost_per_six : ℝ)
  (bars_sold : ℕ)
  (price_per_three : ℝ)
  (tax_rate : ℝ)
  (h1 : bars_bought = 800)
  (h2 : cost_per_six = 3)
  (h3 : bars_sold = 800)
  (h4 : price_per_three = 2)
  (h5 : tax_rate = 0.1) :
  let cost_per_bar := cost_per_six / 6
  let total_cost := bars_bought * cost_per_bar
  let price_per_bar := price_per_three / 3
  let total_revenue := bars_sold * price_per_bar
  let tax := tax_rate * total_revenue
  let after_tax_revenue := total_revenue - tax
  let profit_after_tax := after_tax_revenue - total_cost
  profit_after_tax = 80.02 := by
    sorry

end candy_bar_profit_l269_269160


namespace freda_flag_dimensions_l269_269422

/--  
Given the area of the dove is 192 cm², and the perimeter of the dove consists of quarter-circles or straight lines,
prove that the dimensions of the flag are 24 cm by 16 cm.
-/
theorem freda_flag_dimensions (area_dove : ℝ) (h1 : area_dove = 192) : 
∃ (length width : ℝ), length = 24 ∧ width = 16 := 
sorry

end freda_flag_dimensions_l269_269422


namespace distinct_positive_roots_l269_269770

noncomputable def f (a x : ℝ) : ℝ := x^4 - x^3 + 8 * a * x^2 - a * x + a^2

theorem distinct_positive_roots (a : ℝ) :
  0 < a ∧ a < 1/24 → (∀ x1 x2 x3 x4 : ℝ, f a x1 = 0 ∧ 0 < x1 ∧ f a x2 = 0 ∧ 0 < x2 ∧ f a x3 = 0 ∧ 0 < x3 ∧ f a x4 = 0 ∧ 0 < x4 ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ↔ (1/25 < a ∧ a < 1/24) :=
sorry

end distinct_positive_roots_l269_269770


namespace min_value_of_2a_plus_b_l269_269652

variable (a b : ℝ)

def condition := a > 0 ∧ b > 0 ∧ a - 2 * a * b + b = 0

-- Define what needs to be proved
theorem min_value_of_2a_plus_b (h : condition a b) : ∃ a b : ℝ, 2 * a + b = (3 / 2) + Real.sqrt 2 :=
sorry

end min_value_of_2a_plus_b_l269_269652


namespace circle_diameter_l269_269535

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l269_269535


namespace base_7_units_digit_l269_269295

theorem base_7_units_digit (a : ℕ) (b : ℕ) (h₁ : a = 326) (h₂ : b = 57) : ((a * b) % 7) = 4 := by
  sorry

end base_7_units_digit_l269_269295


namespace cherry_pie_degrees_l269_269665

theorem cherry_pie_degrees :
  ∀ (total_students chocolate_students apple_students blueberry_students : ℕ),
  total_students = 36 →
  chocolate_students = 12 →
  apple_students = 8 →
  blueberry_students = 6 →
  (total_students - chocolate_students - apple_students - blueberry_students) / 2 = 5 →
  ((5 : ℕ) * 360 / total_students) = 50 := 
by
  sorry

end cherry_pie_degrees_l269_269665


namespace time_to_fill_pond_l269_269464

-- Conditions:
def pond_capacity : ℕ := 200
def normal_pump_rate : ℕ := 6
def drought_factor : ℚ := 2 / 3

-- The current pumping rate:
def current_pump_rate : ℚ := normal_pump_rate * drought_factor

-- We need to prove the time it takes to fill the pond is 50 minutes:
theorem time_to_fill_pond : 
  (pond_capacity : ℚ) / current_pump_rate = 50 := 
sorry

end time_to_fill_pond_l269_269464


namespace no_solution_l269_269411

theorem no_solution (a : ℝ) :
  (a < -12 ∨ a > 0) →
  ∀ x : ℝ, ¬(6 * (|x - 4 * a|) + (|x - a ^ 2|) + 5 * x - 4 * a = 0) :=
by
  intros ha hx
  sorry

end no_solution_l269_269411


namespace square_root_domain_l269_269633

theorem square_root_domain (x : ℝ) : (∃ y, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end square_root_domain_l269_269633


namespace problem_solution_l269_269878

theorem problem_solution
  (a b c d : ℕ)
  (h1 : a^6 = b^5)
  (h2 : c^4 = d^3)
  (h3 : c - a = 25) :
  d - b = 561 :=
sorry

end problem_solution_l269_269878


namespace polynomial_rewrite_l269_269291

theorem polynomial_rewrite (d : ℤ) (h : d ≠ 0) :
  let a := 20
  let b := 18
  let c := 18
  let e := 8
  (16 * d + 15 + 18 * d^2 + 5 * d^3) + (4 * d + 3 + 3 * d^3) = a * d + b + c * d^2 + e * d^3 ∧ a + b + c + e = 64 := 
by
  sorry

end polynomial_rewrite_l269_269291


namespace inequality_satisfaction_l269_269208

theorem inequality_satisfaction (a b : ℝ) (h : 0 < a ∧ a < b) : 
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b :=
by
  sorry

end inequality_satisfaction_l269_269208


namespace number_of_sheets_l269_269385

theorem number_of_sheets
  (n : ℕ)
  (h₁ : 2 * n + 2 = 74) :
  n / 4 = 9 :=
by
  sorry

end number_of_sheets_l269_269385


namespace find_m_l269_269626

def vec_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vec_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_m (m : ℝ) : dot_product (vec_a m) (vec_b m) = 0 ↔ m = -1/3 := by 
  sorry

end find_m_l269_269626


namespace triangle_angle_sixty_degrees_l269_269683

theorem triangle_angle_sixty_degrees (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) : 
  ∃ (θ : ℝ), θ = 60 ∧ ∃ (a b c : ℝ), a * b * c ≠ 0 ∧ ∀ {α β γ : ℝ}, (a + b + c = α + β + γ + θ) := 
sorry

end triangle_angle_sixty_degrees_l269_269683


namespace shuffleboard_total_games_l269_269649

theorem shuffleboard_total_games
    (jerry_wins : ℕ)
    (dave_wins : ℕ)
    (ken_wins : ℕ)
    (h1 : jerry_wins = 7)
    (h2 : dave_wins = jerry_wins + 3)
    (h3 : ken_wins = dave_wins + 5) :
    jerry_wins + dave_wins + ken_wins = 32 := 
by
  sorry

end shuffleboard_total_games_l269_269649


namespace ratio_sum_is_four_l269_269658

theorem ratio_sum_is_four
  (x y : ℝ)
  (hx : 0 < x) (hy : 0 < y)
  (θ : ℝ)
  (hθ_ne : ∀ n : ℤ, θ ≠ (n * (π / 2)))
  (h1 : (Real.sin θ) / x = (Real.cos θ) / y)
  (h2 : (Real.cos θ)^4 / x^4 + (Real.sin θ)^4 / y^4 = 97 * (Real.sin (2 * θ)) / (x^3 * y + y^3 * x)) :
  (x / y) + (y / x) = 4 := by
  sorry

end ratio_sum_is_four_l269_269658


namespace probability_of_age_less_than_20_l269_269933

noncomputable def total_people : ℕ := 100 
noncomputable def people_more_than_30 : ℕ := 90
noncomputable def people_less_than_20 : ℕ := total_people - people_more_than_30 

theorem probability_of_age_less_than_20 :
  (people_less_than_20 / total_people : ℚ) = 0.1 := by
sorry

end probability_of_age_less_than_20_l269_269933


namespace find_845th_digit_in_decimal_l269_269988

theorem find_845th_digit_in_decimal : 
  let repeating_sequence := "2413793103448275862068965517"
  let cycle_length := 28
  let position := 845
  (repeating_sequence.getNth! ((position % cycle_length) - 1)) = '1' :=
by
  sorry

end find_845th_digit_in_decimal_l269_269988


namespace man_speed_against_current_l269_269518

-- Definitions for the problem conditions
def man_speed_with_current : ℝ := 21
def current_speed : ℝ := 4.3

-- Main proof statement
theorem man_speed_against_current : man_speed_with_current - 2 * current_speed = 12.4 :=
by
  sorry

end man_speed_against_current_l269_269518


namespace fewer_twos_for_100_l269_269326

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l269_269326


namespace man_speed_is_correct_l269_269750

noncomputable def speed_of_man (train_speed_kmh : ℝ) (train_length_m : ℝ) (time_to_pass_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed_ms := train_length_m / time_to_pass_s
  let man_speed_ms := relative_speed_ms - train_speed_ms
  man_speed_ms * 3600 / 1000

theorem man_speed_is_correct : 
  speed_of_man 60 110 5.999520038396929 = 6.0024 := 
by
  sorry

end man_speed_is_correct_l269_269750


namespace irrational_sqrt3_l269_269170

def is_irrational (x : ℝ) : Prop := ∀ (a b : ℤ), b ≠ 0 → x ≠ a / b

theorem irrational_sqrt3 :
  let A := 22 / 7
  let B := 0
  let C := Real.sqrt 3
  let D := 3.14
  is_irrational C :=
by
  sorry

end irrational_sqrt3_l269_269170


namespace time_to_fill_pond_l269_269465

-- Conditions:
def pond_capacity : ℕ := 200
def normal_pump_rate : ℕ := 6
def drought_factor : ℚ := 2 / 3

-- The current pumping rate:
def current_pump_rate : ℚ := normal_pump_rate * drought_factor

-- We need to prove the time it takes to fill the pond is 50 minutes:
theorem time_to_fill_pond : 
  (pond_capacity : ℚ) / current_pump_rate = 50 := 
sorry

end time_to_fill_pond_l269_269465


namespace modulus_of_complex_l269_269925

open Complex

theorem modulus_of_complex (z : ℂ) (h : z = 1 - (1 / Complex.I)) : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end modulus_of_complex_l269_269925


namespace sum_of_a_b_c_d_l269_269077

theorem sum_of_a_b_c_d (a b c d : ℝ) (h1 : c + d = 12 * a) (h2 : c * d = -13 * b) (h3 : a + b = 12 * c) (h4 : a * b = -13 * d) (h_distinct : a ≠ c) : a + b + c + d = 2028 :=
  by 
  -- The proof will go here
  sorry

end sum_of_a_b_c_d_l269_269077


namespace achieve_target_ratio_l269_269362

-- Initial volume and ratio
def initial_volume : ℕ := 20
def initial_milk_ratio : ℕ := 3
def initial_water_ratio : ℕ := 2

-- Mixture removal and addition
def removal_volume : ℕ := 10
def added_milk : ℕ := 10

-- Target ratio of milk to water
def target_milk_ratio : ℕ := 9
def target_water_ratio : ℕ := 1

-- Number of operations required
def operations_needed: ℕ := 2

-- Statement of proof problem
theorem achieve_target_ratio :
  (initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio) - removal_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio) + added_milk * operations_needed) / 
  (initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio) - removal_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) = target_milk_ratio :=
sorry

end achieve_target_ratio_l269_269362


namespace min_value_of_m_l269_269949

open Real

-- Definitions from the conditions
def condition1 (m : ℝ) : Prop :=
  m > 0

def condition2 (m : ℝ) : Prop :=
  ∀ (x : ℝ), 1 < x → 2 * exp (2 * m * x) - (log x) / m ≥ 0

-- The theorem statement for the minimum value of m
theorem min_value_of_m (m : ℝ) : condition1 m → condition2 m → m ≥ 1 / (2 * exp 1) := 
sorry

end min_value_of_m_l269_269949


namespace factorization_correct_l269_269409

theorem factorization_correct (c : ℝ) : (x : ℝ) → x^2 - x + c = (x + 2) * (x - 3) → c = -6 := by
  intro x h
  sorry

end factorization_correct_l269_269409


namespace part1_part2_l269_269774

theorem part1 (m : ℝ) (h_m_not_zero : m ≠ 0) : m ≤ 4 / 3 :=
by
  -- The proof would go here, but we are inserting sorry to skip it.
  sorry

theorem part2 (m : ℕ) (h_m_range : m = 1) :
  ∃ x1 x2 : ℝ, (m * x1^2 - 4 * x1 + 3 = 0) ∧ (m * x2^2 - 4 * x2 + 3 = 0) ∧ x1 = 1 ∧ x2 = 3 :=
by
  -- The proof would go here, but we are inserting sorry to skip it.
  sorry

end part1_part2_l269_269774


namespace gcd_78_182_l269_269128

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := by
  sorry

end gcd_78_182_l269_269128


namespace exists_continuous_nowhere_gen_diff_l269_269143

noncomputable def generalized_derivative (f : ℝ → ℝ) (x_0 : ℝ) : ℝ :=
  lim (λ h, 2 * ((1 / h) * ∫ t in x_0..(x_0 + h), f t - f x_0) / h)

theorem exists_continuous_nowhere_gen_diff :
  ∃ f : ℝ → ℝ, (∀ x : ℝ, ContinuousAt f x) ∧
    ∀ x_0 : ℝ, ¬DifferentiableAt (generalized_derivative f) x_0 :=
sorry

end exists_continuous_nowhere_gen_diff_l269_269143


namespace customers_left_correct_l269_269167

-- Define the initial conditions
def initial_customers : ℕ := 8
def remaining_customers : ℕ := 5

-- Define the statement regarding customers left
def customers_left : ℕ := initial_customers - remaining_customers

-- The theorem we need to prove
theorem customers_left_correct : customers_left = 3 := by
    -- Skipping the actual proof
    sorry

end customers_left_correct_l269_269167


namespace hyperbola_equation_l269_269659

def equation_of_hyperbola (x y a b : ℝ) : Prop :=
  (x^2 / a^2 - y^2 / b^2 = 1) ∧ a > 0 ∧ b > 0

def line_passing_through_focus (x y b: ℝ) : Prop :=
  let l := λ x, -b * (x - 1) in
  l(0) = b ∧ l(1) = 0

def asymptotes (x y a b : ℝ) : Prop :=
  (∃ m : ℝ, m = b / a ∧ y = m * x) ∨ (∃ m : ℝ, m = - b / a ∧ y = m * x)

theorem hyperbola_equation {x y a b : ℝ} (h1 : equation_of_hyperbola x y a b)
    (h2 : line_passing_through_focus x y b)
    (h3 : ∀ (a b : ℝ), asymptotes x y a b → ((b/a = -b) ∨ (b/a * -b = -1 → x^2 - y^2 = 1))) :
  a = 1 ∧ b = 1 → x^2 - y^2 = 1 := sorry

end hyperbola_equation_l269_269659


namespace symmetrical_ring_of_polygons_l269_269748

theorem symmetrical_ring_of_polygons (m n : ℕ) (hn : n ≥ 7) (hm : m ≥ 3) 
  (condition1 : ∀ p1 p2 : ℕ, p1 ≠ p2 → n = 1) 
  (condition2 : ∀ p : ℕ, p * (n - 2) = 4) 
  (condition3 : ∀ p : ℕ, 2 * m - (n - 2) = 4) :
  ∃ k, (k = 6) :=
by
  -- This block is only a placeholder. The actual proof would go here.
  sorry

end symmetrical_ring_of_polygons_l269_269748


namespace deepak_present_age_l269_269304

theorem deepak_present_age (x : ℕ) (rahul deepak rohan : ℕ) 
  (h_ratio : rahul = 5 * x ∧ deepak = 2 * x ∧ rohan = 3 * x)
  (h_rahul_future_age : rahul + 8 = 28) :
  deepak = 8 := 
by
  sorry

end deepak_present_age_l269_269304


namespace positive_rational_solutions_condition_l269_269897

-- Definitions used in Lean 4 statement corresponding to conditions in the problem.
variable (a b : ℚ)

-- Lean Statement encapsulating the mathematical proof problem.
theorem positive_rational_solutions_condition :
  ∃ x y : ℚ, x > 0 ∧ y > 0 ∧ x * y = a ∧ x + y = b ↔ (∃ k : ℚ, k^2 = b^2 - 4 * a ∧ k > 0) :=
by
  sorry

end positive_rational_solutions_condition_l269_269897


namespace fractions_equal_l269_269515

theorem fractions_equal (a b c d e f : ℕ) :
  a = 2 ∧ b = 6 ∧ c = 4 ∧ d = 12 ∧ e = 5 ∧ f = 15 → 
  (a / b : ℚ) = (c / d : ℚ) ∧ (c / d : ℚ) = (e / f : ℚ) :=
by
  intro h
  cases h with ha h1
  cases h1 with hb h2
  cases h2 with hc h3
  cases h3 with hd h4
  cases h4 with he hf
  sorry

end fractions_equal_l269_269515


namespace product_of_two_numbers_l269_269504

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end product_of_two_numbers_l269_269504


namespace fewer_twos_for_100_l269_269331

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l269_269331


namespace M_is_even_l269_269094

def sum_of_digits (n : ℕ) : ℕ := -- Define the digit sum function
  sorry

theorem M_is_even (M : ℕ) (h1 : sum_of_digits M = 100) (h2 : sum_of_digits (5 * M) = 50) : 
  M % 2 = 0 :=
sorry

end M_is_even_l269_269094


namespace price_of_olives_l269_269444

theorem price_of_olives 
  (cherries_price : ℝ)
  (total_cost_with_discount : ℝ)
  (num_bags : ℕ)
  (discount : ℝ)
  (olives_price : ℝ) :
  cherries_price = 5 →
  total_cost_with_discount = 540 →
  num_bags = 50 →
  discount = 0.10 →
  (0.9 * (num_bags * cherries_price + num_bags * olives_price) = total_cost_with_discount) →
  olives_price = 7 :=
by
  intros h_cherries_price h_total_cost h_num_bags h_discount h_equation
  sorry

end price_of_olives_l269_269444


namespace equilateral_triangle_perimeter_l269_269348

-- Define the condition of an equilateral triangle where each side is 7 cm
def side_length : ℕ := 7

def is_equilateral_triangle (a b c : ℕ) : Prop :=
  a = b ∧ b = c

-- Define the perimeter function for a triangle
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

-- Statement to prove
theorem equilateral_triangle_perimeter : is_equilateral_triangle side_length side_length side_length → perimeter side_length side_length side_length = 21 :=
sorry

end equilateral_triangle_perimeter_l269_269348


namespace trigonometric_identity_l269_269189

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = (1 / (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180))) :=
by
  sorry

end trigonometric_identity_l269_269189


namespace volume_of_bag_l269_269116

-- Define the dimensions of the cuboid
def width : ℕ := 9
def length : ℕ := 4
def height : ℕ := 7

-- Define the volume calculation function for a cuboid
def volume (l w h : ℕ) : ℕ :=
  l * w * h

-- Provide the theorem to prove the volume is 252 cubic centimeters
theorem volume_of_bag : volume length width height = 252 := by
  -- Since the proof is not requested, insert sorry to complete the statement.
  sorry

end volume_of_bag_l269_269116


namespace no_real_solutions_l269_269200

noncomputable def equation (x : ℝ) := x + 48 / (x - 3) + 1

theorem no_real_solutions : ∀ x : ℝ, equation x ≠ 0 :=
by
  intro x
  sorry

end no_real_solutions_l269_269200


namespace white_ball_probability_l269_269136

-- Definition of the problem
def combinations : List (List Bool) := 
  [[false, false, false], -- NNN
   [false, false, true],  -- NNW
   [false, true, false],  -- NWN
   [true, false, false],  -- WNN
   [false, true, true],   -- NWW
   [true, false, true],   -- WNW
   [true, true, false],   -- WWN
   [true, true, true]]    -- WWW

noncomputable def prob_white_ball : ℚ :=
  let cases := combinations.map (λ comb, (comb.count true + 1) / 4)
  (cases.sum / cases.length)

theorem white_ball_probability : prob_white_ball = 5 / 8 := 
  sorry

end white_ball_probability_l269_269136


namespace fill_pond_time_l269_269460

-- Define the constants and their types
def pondVolume : ℕ := 200 -- Volume of the pond in gallons
def normalRate : ℕ := 6 -- Normal rate of the hose in gallons per minute

-- Define the reduced rate due to drought restrictions
def reducedRate : ℕ := (2/3 : ℚ) * normalRate

-- Define the time required to fill the pond
def timeToFill : ℚ := pondVolume / reducedRate

-- The main statement to be proven
theorem fill_pond_time : timeToFill = 50 := by
  sorry

end fill_pond_time_l269_269460


namespace number_of_divisors_M_l269_269885

def M : ℕ := 2^5 * 3^4 * 5^2 * 7^3 * 11^1

theorem number_of_divisors_M : (M.factors.prod.divisors.card = 720) :=
sorry

end number_of_divisors_M_l269_269885


namespace find_multiplying_number_l269_269562

variable (a b : ℤ)

theorem find_multiplying_number (h : a^2 * b = 3 * (4 * a + 2)) (ha : a = 1) :
  b = 18 := by
  sorry

end find_multiplying_number_l269_269562


namespace find_sum_of_roots_l269_269655

open Real

theorem find_sum_of_roots (p q r s : ℝ): 
  r + s = 12 * p →
  r * s = 13 * q →
  p + q = 12 * r →
  p * q = 13 * s →
  p ≠ r →
  p + q + r + s = 2028 := by
  intros
  sorry

end find_sum_of_roots_l269_269655


namespace continuous_function_nondecreasing_l269_269650

open Set

variable {α : Type*} [LinearOrder ℝ] [Preorder ℝ]

theorem continuous_function_nondecreasing
  (f : (ℝ)→ ℝ) 
  (h_cont : ContinuousOn f (Ioi 0))
  (h_seq : ∀ x > 0, Monotone (fun n : ℕ => f (n*x))):
  ∀ x y, x ≤ y → f x ≤ f y := 
sorry

end continuous_function_nondecreasing_l269_269650


namespace SarahsScoreIs135_l269_269273

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end SarahsScoreIs135_l269_269273


namespace theater_ticket_cost_l269_269165

theorem theater_ticket_cost
  (O B : ℕ)
  (h1 : O + B = 370)
  (h2 : B = O + 190) 
  : 12 * O + 8 * B = 3320 :=
by
  sorry

end theater_ticket_cost_l269_269165


namespace maximum_value_inequality_l269_269953

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem maximum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
by
  sorry

end maximum_value_inequality_l269_269953


namespace selected_room_l269_269115

theorem selected_room (room_count interval selected initial_room : ℕ) 
  (h_init : initial_room = 5)
  (h_interval : interval = 8)
  (h_room_count : room_count = 64) : 
  ∃ (nth_room : ℕ), nth_room = initial_room + interval * 6 ∧ nth_room = 53 :=
by
  sorry

end selected_room_l269_269115


namespace hyperbola_eqn_l269_269907

theorem hyperbola_eqn
  (P : ℝ × ℝ) (Q : ℝ × ℝ)
  (C1 : P = (-3, 2 * Real.sqrt 7))
  (C2 : Q = (-6 * Real.sqrt 2, -7))
  (asymptote_hyperbola : ∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1)
  (special_point : ℝ × ℝ)
  (C3 : special_point = (2, 2 * Real.sqrt 3)) :
  ∃ (a b : ℝ), ¬(a = 0) ∧ ¬(b = 0) ∧ 
  (∀ x y : ℝ, (y^2 / b - x^2 / a = 1 → 
    ((y^2 / 25 - x^2 / 75 = 1) ∨ 
    (y^2 / 9 - x^2 / 12 = 1)))) :=
by
  sorry

end hyperbola_eqn_l269_269907


namespace fraction_identity_l269_269795

theorem fraction_identity (a b : ℝ) (h₀ : a^2 + a = 4) (h₁ : b^2 + b = 4) (h₂ : a ≠ b) :
  (b / a) + (a / b) = - (9 / 4) :=
sorry

end fraction_identity_l269_269795


namespace probability_of_selecting_same_gender_l269_269289

def number_of_ways_to_choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_of_selecting_same_gender (total_students male_students female_students : ℕ) (h1 : total_students = 10) (h2 : male_students = 2) (h3 : female_students = 8) : 
  let total_combinations := number_of_ways_to_choose_two total_students
  let male_combinations := number_of_ways_to_choose_two male_students
  let female_combinations := number_of_ways_to_choose_two female_students
  let favorable_combinations := male_combinations + female_combinations
  total_combinations = 45 ∧
  male_combinations = 1 ∧
  female_combinations = 28 ∧
  favorable_combinations = 29 ∧
  (favorable_combinations : ℚ) / total_combinations = 29 / 45 :=
by
  sorry

end probability_of_selecting_same_gender_l269_269289


namespace specific_value_is_165_l269_269741

-- Declare x as a specific number and its value
def x : ℕ := 11

-- Declare the specific value as 15 times x
def specific_value : ℕ := 15 * x

-- The theorem to prove
theorem specific_value_is_165 : specific_value = 165 := by
  sorry

end specific_value_is_165_l269_269741


namespace max_M_is_2_l269_269474

theorem max_M_is_2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hdisc : b^2 - 4 * a * c ≥ 0) :
    max (min (b + c / a) (min (c + a / b) (a + b / c))) = 2 := by
    sorry

end max_M_is_2_l269_269474


namespace tetrahedron_three_edges_form_triangle_l269_269827

-- Defining a tetrahedron
structure Tetrahedron := (A B C D : ℝ)
-- length of edges - since it's a geometry problem using the absolute value
def edge_length (x y : ℝ) := abs (x - y)

theorem tetrahedron_three_edges_form_triangle (T : Tetrahedron) :
  ∃ v : ℕ, ∃ e1 e2 e3 : ℝ, 
    (edge_length T.A T.B = e1 ∨ edge_length T.A T.C = e1 ∨ edge_length T.A T.D = e1) ∧ 
    (edge_length T.B T.C = e2 ∨ edge_length T.B T.D = e2 ∨ edge_length T.C T.D = e2) ∧
    (edge_length T.A T.B < e2 + e3 ∧ edge_length T.B T.C < e1 + e3 ∧ edge_length T.C T.D < e1 + e2) := 
sorry

end tetrahedron_three_edges_form_triangle_l269_269827


namespace sum_of_operations_l269_269434

noncomputable def triangle (a b c : ℕ) : ℕ :=
  a + 2 * b - c

theorem sum_of_operations :
  triangle 3 5 7 + triangle 6 1 8 = 6 :=
by
  sorry

end sum_of_operations_l269_269434


namespace card_dealing_probability_l269_269508

-- Define the events and their probabilities
def prob_first_card_ace : ℚ := 4 / 52
def prob_second_card_ten_given_ace : ℚ := 4 / 51
def prob_third_card_jack_given_ace_and_ten : ℚ := 2 / 25

-- Define the overall probability
def overall_probability : ℚ :=
  prob_first_card_ace * 
  prob_second_card_ten_given_ace *
  prob_third_card_jack_given_ace_and_ten

-- State the problem
theorem card_dealing_probability :
  overall_probability = 8 / 16575 := by
  sorry

end card_dealing_probability_l269_269508


namespace A_form_k_l269_269251

theorem A_form_k (m n : ℕ) (h_m : 2 ≤ m) (h_n : 2 ≤ n) :
  ∃ k : ℕ, (A : ℝ) = (n + Real.sqrt (n^2 - 4)) / 2 ^ m → A = (k + Real.sqrt (k^2 - 4)) / 2 :=
by
  sorry

end A_form_k_l269_269251


namespace time_to_fill_pond_l269_269462

noncomputable def pond_capacity : ℝ := 200
noncomputable def normal_pump_rate : ℝ := 6
noncomputable def restriction_factor : ℝ := 2 / 3
noncomputable def restricted_pump_rate : ℝ := restriction_factor * normal_pump_rate

theorem time_to_fill_pond : pond_capacity / restricted_pump_rate = 50 := 
by 
  -- This is where the proof would go
  sorry

end time_to_fill_pond_l269_269462


namespace same_function_absolute_value_l269_269989

theorem same_function_absolute_value :
  (∀ (x : ℝ), |x| = if x > 0 then x else -x) :=
by
  intro x
  split_ifs with h
  · exact abs_of_pos h
  · exact abs_of_nonpos (le_of_not_gt h)

end same_function_absolute_value_l269_269989


namespace part1_part2_l269_269000

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l269_269000


namespace travel_time_to_Virgo_island_l269_269512

theorem travel_time_to_Virgo_island (boat_time : ℝ) (plane_time : ℝ) (total_time : ℝ) 
  (h1 : boat_time ≤ 2) (h2 : plane_time = 4 * boat_time) (h3 : total_time = plane_time + boat_time) : 
  total_time = 10 :=
by
  sorry

end travel_time_to_Virgo_island_l269_269512


namespace smallest_y2_l269_269084

theorem smallest_y2 :
  let y1 := 1 / (-2)
  let y2 := 1 / (-1)
  let y3 := 1 / (1)
  let y4 := 1 / (2)
  y2 < y1 ∧ y2 < y3 ∧ y2 < y4 :=
by
  let y1 := 1 / (-2)
  let y2 := 1 / (-1)
  let y3 := 1 / (1)
  let y4 := 1 / (2)
  show y2 < y1 ∧ y2 < y3 ∧ y2 < y4
  sorry

end smallest_y2_l269_269084


namespace distance_covered_is_9_17_miles_l269_269245

noncomputable def totalDistanceCovered 
  (walkingTimeInMinutes : ℕ) (walkingRate : ℝ)
  (runningTimeInMinutes : ℕ) (runningRate : ℝ)
  (cyclingTimeInMinutes : ℕ) (cyclingRate : ℝ) : ℝ :=
  (walkingRate * (walkingTimeInMinutes / 60.0)) + 
  (runningRate * (runningTimeInMinutes / 60.0)) + 
  (cyclingRate * (cyclingTimeInMinutes / 60.0))

theorem distance_covered_is_9_17_miles :
  totalDistanceCovered 30 3 20 8 25 12 = 9.17 := 
by 
  sorry

end distance_covered_is_9_17_miles_l269_269245


namespace complex_expression_eq_l269_269198

-- Define the complex numbers
def c1 : ℂ := 6 - 3 * Complex.I
def c2 : ℂ := 2 - 7 * Complex.I

-- Define the scale
def scale : ℂ := 3

-- State the theorem
theorem complex_expression_eq : (c1 + scale * c2) = 12 - 24 * Complex.I :=
by
  -- This is the statement only; the proof is omitted with sorry.
  sorry

end complex_expression_eq_l269_269198


namespace gcf_50_75_l269_269347

theorem gcf_50_75 : Nat.gcd 50 75 = 25 := by
  sorry

end gcf_50_75_l269_269347


namespace original_survey_customers_l269_269164

theorem original_survey_customers : ∃ x : ℕ, (7 / x + 0.02).approx (1 / 7) ∧ x ≈ 57 :=
begin
  sorry
end

end original_survey_customers_l269_269164


namespace add_ab_values_l269_269218

theorem add_ab_values (a b : ℝ) (h1 : ∀ x : ℝ, (x^2 + 4*x + 3) = (a*x + b)^2 + 4*(a*x + b) + 3) :
  a + b = -8 ∨ a + b = 4 :=
  by sorry

end add_ab_values_l269_269218


namespace sqrt_product_l269_269763

theorem sqrt_product (h54 : Real.sqrt 54 = 3 * Real.sqrt 6)
                     (h32 : Real.sqrt 32 = 4 * Real.sqrt 2)
                     (h6 : Real.sqrt 6 = Real.sqrt 6) :
    Real.sqrt 54 * Real.sqrt 32 * Real.sqrt 6 = 72 * Real.sqrt 2 := by
  sorry

end sqrt_product_l269_269763


namespace fruit_seller_original_apples_l269_269861

theorem fruit_seller_original_apples (x : ℝ) (h : 0.50 * x = 5000) : x = 10000 :=
sorry

end fruit_seller_original_apples_l269_269861


namespace winner_for_2023_winner_for_2024_l269_269453

-- Definitions for the game conditions
def barbara_moves : List ℕ := [3, 5]
def jenna_moves : List ℕ := [1, 4, 5]

-- Lean theorem statement proving the required answers
theorem winner_for_2023 (coins : ℕ) (barbara_moves : List ℕ) (jenna_moves : List ℕ) :
  coins = 2023 →  -- Specifying that the game starts with 2023 coins
  (∀n, n ∈ barbara_moves → n ≤ 2023) ∧ (∀n, n ∈ jenna_moves → n ≤ 2023) → 
  -- Specifying valid moves for both players
  ∃ winner : String, winner = "Barbara" := 
sorry

theorem winner_for_2024 (coins : ℕ) (barbara_moves : List ℕ) (jenna_moves : List ℕ) :
  coins = 2024 →  -- Specifying that the game starts with 2024 coins
  (∀n, n ∈ barbara_moves → n ≤ 2024) ∧ (∀n, n ∈ jenna_moves → n ≤ 2024) → 
  -- Specifying valid moves for both players
  ∃ winner : String, winner = "Whoever starts" :=
sorry

end winner_for_2023_winner_for_2024_l269_269453


namespace triangle_iff_inequality_l269_269073

variable {a b c : ℝ}

theorem triangle_iff_inequality :
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) := sorry

end triangle_iff_inequality_l269_269073


namespace goldbach_134_l269_269491

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0
noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem goldbach_134 (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h_sum : p + q = 134) (h_diff : p ≠ q) : 
  ∃ (d : ℕ), d = 134 - (2 * p) ∧ d ≤ 128 := 
sorry

end goldbach_134_l269_269491


namespace trigonometric_expression_value_l269_269184

theorem trigonometric_expression_value :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 :=
by
  sorry

end trigonometric_expression_value_l269_269184


namespace range_of_a_l269_269433

variable (a : ℝ)

def condition1 : Prop := a < 0
def condition2 : Prop := -a / 2 ≥ 1
def condition3 : Prop := -1 - a - 5 ≤ a

theorem range_of_a :
  condition1 a ∧ condition2 a ∧ condition3 a → -3 ≤ a ∧ a ≤ -2 :=
by
  sorry

end range_of_a_l269_269433


namespace members_playing_badminton_l269_269451

theorem members_playing_badminton
  (total_members : ℕ := 42)
  (tennis_players : ℕ := 23)
  (neither_players : ℕ := 6)
  (both_players : ℕ := 7) :
  ∃ (badminton_players : ℕ), badminton_players = 20 :=
by
  have union_players := total_members - neither_players
  have badminton_players := union_players - (tennis_players - both_players)
  use badminton_players
  sorry

end members_playing_badminton_l269_269451


namespace average_num_divisors_2019_l269_269898

def num_divisors (n : ℕ) : ℕ :=
  (n.divisors).card

theorem average_num_divisors_2019 :
  1 / 2019 * (Finset.sum (Finset.range 2020) num_divisors) = 15682 / 2019 :=
by
  sorry

end average_num_divisors_2019_l269_269898


namespace fraction_computation_l269_269591

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l269_269591


namespace quiz_winning_probability_l269_269870

noncomputable def probability_win_quiz : ℚ :=
  let p_correct : ℚ := 1 / 3 in
  let p_all_correct := p_correct^4 in
  let p_three_correct_one_wrong := (p_correct^3) * (2 / 3) in
  let num_ways_three_correct := 4 in
  let p_exactly_three_correct := num_ways_three_correct * p_three_correct_one_wrong in
  p_all_correct + p_exactly_three_correct

theorem quiz_winning_probability : probability_win_quiz = 1 / 9 :=
by
  rw [probability_win_quiz]
  dsimp only [probability_win_quiz._match_1, probability_win_quiz._match_2]
  -- calculations matching the solution steps
  have h1 : (1 / 3) ^ 4 = 1 / 81 := by norm_num
  have h2 : (1 / 3) ^ 3 * (2 / 3) = 2 / 81 := by norm_num
  have h3 : 4 * (2 / 81) = 8 / 81 := by norm_num
  calc (1 / 81) + (8 / 81)
       = 9 / 81 : by norm_num
   ... = 1 / 9  : by norm_num
  sorry

end quiz_winning_probability_l269_269870


namespace fraction_simplification_l269_269598

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l269_269598


namespace number_of_sheets_is_9_l269_269378

-- Define the conditions in Lean
variable (n : ℕ) -- Total number of pages

-- The stack is folded and renumbered
axiom folded_and_renumbered : True

-- The sum of numbers on one of the sheets is 74
axiom sum_sheet_is_74 : 2 * n + 2 = 74

-- The number of sheets is the number of pages divided by 4
def number_of_sheets (n : ℕ) : ℕ := n / 4

-- Prove that the number of sheets is 9 given the conditions
theorem number_of_sheets_is_9 : number_of_sheets 36 = 9 :=
by
  sorry

end number_of_sheets_is_9_l269_269378


namespace pick_three_numbers_l269_269824

theorem pick_three_numbers 
  (S : Finset ℕ) (hS : S.card = 4) (hS_set : ∀ x ∈ S, x ∈ Finset.range 21) : 
  ∃ a b c ∈ S, ∃ x : ℤ, (a * x ≡ b [MOD c]) :=
by {
  sorry
}

end pick_three_numbers_l269_269824


namespace fraction_simplification_l269_269595

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l269_269595


namespace min_m_value_l269_269240

theorem min_m_value :
  ∃ (x y m : ℝ), x - y + 2 ≥ 0 ∧ x + y - 2 ≤ 0 ∧ 2 * y ≥ x + 2 ∧
  (m > 0) ∧ (x^2 / 4 + y^2 = m^2) ∧ m = Real.sqrt 2 / 2 :=
sorry

end min_m_value_l269_269240


namespace pqr_value_l269_269260

theorem pqr_value
  (p q r : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (h_sum : p + q + r = 29)
  (h_eq : 1 / p + 1 / q + 1 / r + 392 / (p * q * r) = 1) :
  p * q * r = 630 :=
by
  sorry

end pqr_value_l269_269260


namespace problem1_problem2_l269_269179

theorem problem1 : (-5 : ℝ) ^ 0 - (1 / 3) ^ (-2 : ℝ) + (-2 : ℝ) ^ 2 = -4 := 
by
  sorry

variable (a : ℝ)

theorem problem2 : (-3 * a ^ 3) ^ 2 * 2 * a ^ 3 - 8 * a ^ 12 / (2 * a ^ 3) = 14 * a ^ 9 :=
by
  sorry

end problem1_problem2_l269_269179


namespace fraction_identity_l269_269600

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l269_269600


namespace part1_part2_l269_269001

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l269_269001


namespace geom_seq_a4_l269_269804

theorem geom_seq_a4 (a1 a2 a3 a4 r : ℝ)
  (h1 : a1 + a2 + a3 = 7)
  (h2 : a1 * a2 * a3 = 8)
  (h3 : a1 > 0)
  (h4 : r > 1)
  (h5 : a2 = a1 * r)
  (h6 : a3 = a1 * r^2)
  (h7 : a4 = a1 * r^3) : 
  a4 = 8 :=
sorry

end geom_seq_a4_l269_269804


namespace combined_mpg_l269_269685

theorem combined_mpg (m : ℕ) (ray_mpg tom_mpg : ℕ) (h1 : m = 200) (h2 : ray_mpg = 40) (h3 : tom_mpg = 20) :
  (m / (m / (2 * ray_mpg) + m / (2 * tom_mpg))) = 80 / 3 :=
by
  sorry

end combined_mpg_l269_269685


namespace mn_value_l269_269789

theorem mn_value (m n : ℝ) 
  (h1 : m^2 + 1 = 4)
  (h2 : 2 * m + n = 0) :
  m * n = -6 := 
sorry

end mn_value_l269_269789


namespace triangle_inequality_sqrt_sides_l269_269469

theorem triangle_inequality_sqrt_sides {a b c : ℝ} (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b):
  (Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) 
  ∧ (Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) = Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
sorry

end triangle_inequality_sqrt_sides_l269_269469


namespace work_completion_l269_269732

theorem work_completion (d : ℝ) :
  (9 * (1 / d) + 8 * (1 / 20) = 1) ↔ (d = 15) :=
by
  sorry

end work_completion_l269_269732


namespace trigonometric_identity_l269_269186

theorem trigonometric_identity (h1 : cos (70 * Real.pi / 180) ≠ 0) (h2 : sin (70 * Real.pi / 180) ≠ 0) :
  (1 / cos (70 * Real.pi / 180) - (Real.sqrt 3) / sin (70 * Real.pi / 180)) = Real.cot (20 * Real.pi / 180) - 2 :=
by
  sorry

end trigonometric_identity_l269_269186


namespace solve_for_x_l269_269239

theorem solve_for_x : ∀ (x : ℕ), (y = 2 / (4 * x + 2)) → (y = 1 / 2) → (x = 1/2) :=
by
  sorry

end solve_for_x_l269_269239


namespace carpet_area_proof_l269_269195

noncomputable def carpet_area (main_room_length_ft : ℕ) (main_room_width_ft : ℕ)
  (corridor_length_ft : ℕ) (corridor_width_ft : ℕ) (feet_per_yard : ℕ) : ℚ :=
  let main_room_length_yd := main_room_length_ft / feet_per_yard
  let main_room_width_yd := main_room_width_ft / feet_per_yard
  let corridor_length_yd := corridor_length_ft / feet_per_yard
  let corridor_width_yd := corridor_width_ft / feet_per_yard
  let main_room_area_yd2 := main_room_length_yd * main_room_width_yd
  let corridor_area_yd2 := corridor_length_yd * corridor_width_yd
  main_room_area_yd2 + corridor_area_yd2

theorem carpet_area_proof : carpet_area 15 12 10 3 3 = 23.33 :=
by
  -- Proof steps go here
  sorry

end carpet_area_proof_l269_269195


namespace generatrix_length_of_cone_l269_269784

theorem generatrix_length_of_cone (r : ℝ) (l : ℝ) (h1 : r = 4) (h2 : (2 * Real.pi * r) = (Real.pi / 2) * l) : l = 16 := 
by
  sorry

end generatrix_length_of_cone_l269_269784


namespace find_second_number_l269_269998

theorem find_second_number (X : ℝ) : 
  (0.6 * 50 - 0.3 * X = 27) → X = 10 :=
by
  sorry

end find_second_number_l269_269998


namespace not_possible_sum_2017_l269_269937

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem not_possible_sum_2017 (A B : ℕ) (h1 : A + B = 2017) (h2 : sum_of_digits A = 2 * sum_of_digits B) : false := 
sorry

end not_possible_sum_2017_l269_269937


namespace SarahsScoreIs135_l269_269276

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end SarahsScoreIs135_l269_269276


namespace correct_average_marks_l269_269105

theorem correct_average_marks 
  (avg_marks : ℝ) 
  (num_students : ℕ) 
  (incorrect_marks : ℕ → (ℝ × ℝ)) :
  avg_marks = 85 →
  num_students = 50 →
  incorrect_marks 0 = (95, 45) →
  incorrect_marks 1 = (78, 58) →
  incorrect_marks 2 = (120, 80) →
  (∃ corrected_avg_marks : ℝ, corrected_avg_marks = 82.8) :=
by
  sorry

end correct_average_marks_l269_269105


namespace set_roster_method_l269_269690

open Set

theorem set_roster_method :
  { m : ℤ | ∃ n : ℕ, 12 = n * (m + 1) } = {0, 1, 2, 3, 5, 11} :=
  sorry

end set_roster_method_l269_269690


namespace part1_part2_l269_269034

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269034


namespace sum_of_squares_l269_269479

theorem sum_of_squares (n m : ℕ) (h : 2 * m = n^2 + 1) : ∃ k : ℕ, m = k^2 + (k - 1)^2 :=
sorry

end sum_of_squares_l269_269479


namespace circle_diameter_l269_269528

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l269_269528


namespace expression_evaluation_l269_269400

-- Define expression variable to ensure emphasis on conditions and calculations
def expression : ℤ := 9 - (8 + 7) * 6 + 5^2 - (4 * 3) + 2 - 1

theorem expression_evaluation : expression = -67 :=
by
  -- Use assumptions about the order of operations to conclude
  sorry

end expression_evaluation_l269_269400


namespace maximum_ab_l269_269653

open Real

theorem maximum_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 6 * a + 5 * b = 75) :
  ab ≤ 46.875 :=
by
  -- proof goes here
  sorry

end maximum_ab_l269_269653


namespace iron_balls_molded_l269_269397

-- Define the dimensions of the iron bar
def length_bar : ℝ := 12
def width_bar : ℝ := 8
def height_bar : ℝ := 6

-- Define the volume calculations
def volume_iron_bar : ℝ := length_bar * width_bar * height_bar
def number_of_bars : ℝ := 10
def total_volume_bars : ℝ := volume_iron_bar * number_of_bars
def volume_iron_ball : ℝ := 8

-- Define the goal statement
theorem iron_balls_molded : total_volume_bars / volume_iron_ball = 720 :=
by
  -- Proof is to be filled in here
  sorry

end iron_balls_molded_l269_269397


namespace anne_cleaning_time_l269_269864

variable (B A : ℝ)

theorem anne_cleaning_time :
  (B + A) * 4 = 1 ∧ (B + 2 * A) * 3 = 1 → 1/A = 12 := 
by
  intro h
  sorry

end anne_cleaning_time_l269_269864


namespace product_of_two_numbers_l269_269497

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by sorry

end product_of_two_numbers_l269_269497


namespace value_two_std_dev_less_than_mean_l269_269360

-- Define the given conditions for the problem.
def mean : ℝ := 15
def std_dev : ℝ := 1.5

-- Define the target value that should be 2 standard deviations less than the mean.
def target_value := mean - 2 * std_dev

-- State the theorem that represents the proof problem.
theorem value_two_std_dev_less_than_mean : target_value = 12 := by
  sorry

end value_two_std_dev_less_than_mean_l269_269360


namespace Lorelai_jellybeans_correct_l269_269090

def Gigi_jellybeans : ℕ := 15
def Rory_jellybeans : ℕ := Gigi_jellybeans + 30
def Total_jellybeans : ℕ := Rory_jellybeans + Gigi_jellybeans
def Lorelai_jellybeans : ℕ := 3 * Total_jellybeans

theorem Lorelai_jellybeans_correct : Lorelai_jellybeans = 180 := by
  sorry

end Lorelai_jellybeans_correct_l269_269090


namespace number_of_sheets_l269_269384

theorem number_of_sheets
  (n : ℕ)
  (h₁ : 2 * n + 2 = 74) :
  n / 4 = 9 :=
by
  sorry

end number_of_sheets_l269_269384


namespace relative_positions_of_P_on_AB_l269_269566

theorem relative_positions_of_P_on_AB (A B P : ℝ) : 
  A ≤ B → (A ≤ P ∧ P ≤ B ∨ P = A ∨ P = B ∨ P < A ∨ P > B) :=
by
  intro hAB
  sorry

end relative_positions_of_P_on_AB_l269_269566


namespace range_of_m_l269_269902

theorem range_of_m (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1^2 - 4 * x1 + m - 1 = 0 ∧ x2^2 - 4 * x2 + m - 1 = 0 ∧ x1 ≠ x2) ∧ 
  (3 * (m - 1) - 4 > 2) →

  3 < m ∧ m ≤ 5 :=
sorry

end range_of_m_l269_269902


namespace part1_part2_l269_269038

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269038


namespace third_person_fraction_removed_l269_269775

-- Define the number of teeth for each person and the fractions that are removed
def total_teeth := 32
def total_removed := 40

def first_person_removed := (1 / 4) * total_teeth
def second_person_removed := (3 / 8) * total_teeth
def fourth_person_removed := 4

-- Define the total teeth removed by the first, second, and fourth persons
def known_removed := first_person_removed + second_person_removed + fourth_person_removed

-- Define the total teeth removed by the third person
def third_person_removed := total_removed - known_removed

-- Prove that the third person had 1/2 of his teeth removed
theorem third_person_fraction_removed :
  third_person_removed / total_teeth = 1 / 2 :=
by
  sorry

end third_person_fraction_removed_l269_269775


namespace sum_of_coordinates_l269_269672

theorem sum_of_coordinates (x y : ℚ) (h₁ : y = 5) (h₂ : (y - 0) / (x - 0) = 3/4) : x + y = 35/3 :=
by sorry

end sum_of_coordinates_l269_269672


namespace books_difference_l269_269818

theorem books_difference (maddie_books luisa_books amy_books total_books : ℕ) 
  (h1 : maddie_books = 15) 
  (h2 : luisa_books = 18) 
  (h3 : amy_books = 6) 
  (h4 : total_books = amy_books + luisa_books) :
  total_books - maddie_books = 9 := 
sorry

end books_difference_l269_269818


namespace part_1_part_2_l269_269004

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269004


namespace no_integer_coeff_trinomials_with_integer_roots_l269_269768

theorem no_integer_coeff_trinomials_with_integer_roots :
  ¬ ∃ (a b c : ℤ),
    (∀ x : ℤ, a * x^2 + b * x + c = 0 → (∃ x1 x2 : ℤ, a = 0 ∧ x = x1 ∨ a ≠ 0 ∧ x = x1 ∨ x = x2 ∨ x = x1 ∧ x = x2)) ∧
    (∀ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = 0 → (∃ x1 x2 : ℤ, (a + 1) = 0 ∧ x = x1 ∨ (a + 1) ≠ 0 ∧ x = x1 ∨ x = x2 ∨ x = x1 ∧ x = x2)) :=
by
  sorry

end no_integer_coeff_trinomials_with_integer_roots_l269_269768


namespace fraction_computation_l269_269592

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l269_269592


namespace probability_each_person_selected_l269_269837

-- Define the number of initial participants
def initial_participants := 2007

-- Define the number of participants to exclude
def exclude_participants := 7

-- Define the final number of participants remaining after exclusion
def remaining_participants := initial_participants - exclude_participants

-- Define the number of participants to select
def select_participants := 50

-- Define the probability of each participant being selected
def selection_probability : ℚ :=
  select_participants * remaining_participants / (initial_participants * remaining_participants)

theorem probability_each_person_selected :
  selection_probability = (50 / 2007 : ℚ) :=
sorry

end probability_each_person_selected_l269_269837


namespace fewer_twos_to_hundred_l269_269334

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l269_269334


namespace m_div_x_eq_4_div_5_l269_269731

variable (a b : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_ratio : a / b = 4 / 5)

def x := a * 1.25

def m := b * 0.80

theorem m_div_x_eq_4_div_5 : m / x = 4 / 5 :=
by
  sorry

end m_div_x_eq_4_div_5_l269_269731


namespace fastest_pipe_is_4_l269_269371

/-- There are five pipes with flow rates Q_1, Q_2, Q_3, Q_4, and Q_5.
    The ordering of their flow rates is given by:
    (1) Q_1 > Q_3
    (2) Q_2 < Q_4
    (3) Q_3 < Q_5
    (4) Q_4 > Q_1
    (5) Q_5 < Q_2
    We need to prove that single pipe Q_4 will fill the pool the fastest.
 -/
theorem fastest_pipe_is_4 
  (Q1 Q2 Q3 Q4 Q5 : ℝ)
  (h1 : Q1 > Q3)
  (h2 : Q2 < Q4)
  (h3 : Q3 < Q5)
  (h4 : Q4 > Q1)
  (h5 : Q5 < Q2) :
  Q4 > Q1 ∧ Q4 > Q2 ∧ Q4 > Q3 ∧ Q4 > Q5 :=
by
  sorry

end fastest_pipe_is_4_l269_269371


namespace battery_charging_budget_l269_269230

def cost_per_charge : ℝ := 3.5
def charges : ℕ := 4
def leftover : ℝ := 6
def budget : ℝ := 20

theorem battery_charging_budget :
  (charges : ℝ) * cost_per_charge + leftover = budget :=
by
  sorry

end battery_charging_budget_l269_269230


namespace axis_of_symmetry_l269_269297

-- Definitions for conditions
variable (ω : ℝ) (φ : ℝ) (A B : ℝ)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- Hypotheses
axiom ω_pos : ω > 0
axiom φ_bound : 0 ≤ φ ∧ φ < Real.pi
axiom even_func : ∀ x, f x = f (-x)
axiom dist_AB : abs (B - A) = 4 * Real.sqrt 2

-- Proof statement
theorem axis_of_symmetry : ∃ x : ℝ, x = 4 := 
sorry

end axis_of_symmetry_l269_269297


namespace num_routes_avoiding_danger_l269_269068

-- Definitions based on conditions:
def start := (0, 0)
def end := (4, 3)
def dangerous := (2, 2)
def total_blocks_east := 4
def total_blocks_north := 3

-- Proof Problem:
theorem num_routes_avoiding_danger :
  let total_paths := Nat.choose (total_blocks_east + total_blocks_north) total_blocks_east,
      paths_to_danger := Nat.choose (dangerous.1 + dangerous.2) dangerous.1,
      paths_from_danger := Nat.choose ((end.1 - dangerous.1) + (end.2 - dangerous.2)) (end.1 - dangerous.1),
      paths_via_danger := paths_to_danger * paths_from_danger
  in total_paths - paths_via_danger = 17 := 
sorry

end num_routes_avoiding_danger_l269_269068


namespace solution_interval_l269_269840

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x - x^(1 / 3)

theorem solution_interval (x₀ : ℝ) 
  (h_solution : (1 / 2)^x₀ = x₀^(1 / 3)) : x₀ ∈ Set.Ioo (1 / 3) (1 / 2) :=
by
  sorry

end solution_interval_l269_269840


namespace water_evaporation_correct_l269_269155

noncomputable def water_evaporation_each_day (initial_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ) : ℝ :=
  let total_evaporated := (percentage_evaporated / 100) * initial_water
  total_evaporated / days

theorem water_evaporation_correct :
  water_evaporation_each_day 10 6 30 = 0.02 := by
  sorry

end water_evaporation_correct_l269_269155


namespace total_apples_l269_269238

theorem total_apples (apples_per_person : ℕ) (num_people : ℕ) (h1 : apples_per_person = 25) (h2 : num_people = 6) : apples_per_person * num_people = 150 := by
  sorry

end total_apples_l269_269238


namespace total_games_played_l269_269646

theorem total_games_played (jerry_wins dave_wins ken_wins : ℕ)
  (h1 : jerry_wins = 7)
  (h2 : dave_wins = jerry_wins + 3)
  (h3 : ken_wins = dave_wins + 5) :
  jerry_wins + dave_wins + ken_wins = 32 :=
by
  sorry

end total_games_played_l269_269646


namespace part1_part2_l269_269029

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269029


namespace gravitational_field_height_depth_equality_l269_269666

theorem gravitational_field_height_depth_equality
  (R G ρ : ℝ) (hR : R > 0) :
  ∃ x : ℝ, x = R * ((-1 + Real.sqrt 5) / 2) ∧
  (G * ρ * ((4 / 3) * Real.pi * R^3) / (R + x)^2 = G * ρ * ((4 / 3) * Real.pi * (R - x)^3) / (R - x)^2) :=
by
  sorry

end gravitational_field_height_depth_equality_l269_269666


namespace andy_cavities_l269_269575

def candy_canes_from_parents : ℕ := 2
def candy_canes_per_teacher : ℕ := 3
def number_of_teachers : ℕ := 4
def fraction_to_buy : ℚ := 1 / 7
def cavities_per_candies : ℕ := 4

theorem andy_cavities : (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers 
                         + (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers) * fraction_to_buy)
                         / cavities_per_candies = 4 := by
  sorry

end andy_cavities_l269_269575


namespace sum_of_coordinates_of_other_endpoint_l269_269303

theorem sum_of_coordinates_of_other_endpoint :
  ∀ (x y : ℤ), (7, -15) = ((x + 3) / 2, (y - 5) / 2) → x + y = -14 :=
by
  intros x y h
  sorry

end sum_of_coordinates_of_other_endpoint_l269_269303


namespace largest_sum_of_digits_l269_269440

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) (h1: a < 10) (h2: b < 10) (h3: c < 10) (h4: 0 < y ∧ y ≤ 12) (h5: 1000 * y = abc) :
  a + b + c = 8 := by
  sorry

end largest_sum_of_digits_l269_269440


namespace part_a_part_b_part_c_l269_269307

variable (N : ℕ) (r : Fin N → Fin N → ℝ)

-- Part (a)
theorem part_a (h : ∀ (s : Finset (Fin N)), s.card = 5 → (exists pts : s → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ∃ pts : Fin N → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j :=
sorry

-- Part (b)
theorem part_b (h : ∀ (s : Finset (Fin N)), s.card = 4 → (exists pts : s → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ¬ (∃ pts : Fin N → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j) :=
sorry

-- Part (c)
theorem part_c (h : ∀ (s : Finset (Fin N)), s.card = 6 → (exists pts : s → ℝ × ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ∃ (pts : Fin N → ℝ × ℝ × ℝ), ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j :=
sorry

end part_a_part_b_part_c_l269_269307


namespace book_cost_l269_269148

theorem book_cost (album_cost : ℝ) (h1 : album_cost = 20) (h2 : ∀ cd_cost, cd_cost = album_cost * 0.7)
  (h3 : ∀ book_cost, book_cost = cd_cost + 4) : book_cost = 18 := by
  sorry

end book_cost_l269_269148


namespace value_of_y_l269_269052

theorem value_of_y 
  (x y : ℤ) 
  (h1 : x - y = 10) 
  (h2 : x + y = 8) 
  : y = -1 := by
  sorry

end value_of_y_l269_269052


namespace equal_or_equal_exponents_l269_269521

theorem equal_or_equal_exponents
  (a b c p q r : ℕ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r)
  (h1 : a^p + b^q + c^r = a^q + b^r + c^p)
  (h2 : a^q + b^r + c^p = a^r + b^p + c^q) :
  a = b ∧ b = c ∧ c = a ∨ p = q ∧ q = r ∧ r = p :=
  sorry

end equal_or_equal_exponents_l269_269521


namespace homework_time_decrease_l269_269316

theorem homework_time_decrease (x : ℝ) :
  let T_initial := 100
  let T_final := 70
  T_initial * (1 - x) * (1 - x) = T_final :=
by
  sorry

end homework_time_decrease_l269_269316


namespace sequence_bound_l269_269262

open Real

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ)
  (h₀ : ∀ i : ℕ, 0 < i → 0 ≤ a i ∧ a i ≤ c)
  (h₁ : ∀ (i j : ℕ), 0 < i → 0 < j → i ≠ j → abs (a i - a j) ≥ 1 / (i + j)) :
  c ≥ 1 :=
by {
  sorry
}

end sequence_bound_l269_269262


namespace circle_diameter_l269_269531

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l269_269531


namespace fewer_twos_to_hundred_l269_269336

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l269_269336


namespace function_behavior_on_negative_interval_l269_269551

-- Define the necessary conditions and function properties
variables {f : ℝ → ℝ}

-- Conditions: f is even, increasing on [0, 7], and f(7) = 6
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def f7_eq_6 (f : ℝ → ℝ) : Prop := f 7 = 6

-- The theorem to prove
theorem function_behavior_on_negative_interval (h1 : even_function f) (h2 : increasing_on_interval f 0 7) (h3 : f7_eq_6 f) : 
  (∀ x y, -7 ≤ x → x ≤ y → y ≤ 0 → f x ≥ f y) ∧ (∀ x, -7 ≤ x → x ≤ 0 → f x ≤ 6) :=
sorry

end function_behavior_on_negative_interval_l269_269551


namespace amount_paid_to_shopkeeper_l269_269171

theorem amount_paid_to_shopkeeper :
  let price_of_grapes := 8 * 70
  let price_of_mangoes := 9 * 55
  price_of_grapes + price_of_mangoes = 1055 :=
by
  sorry

end amount_paid_to_shopkeeper_l269_269171


namespace equation1_solutions_equation2_solutions_l269_269096

theorem equation1_solutions (x : ℝ) :
  (4 * x^2 = 12 * x) ↔ (x = 0 ∨ x = 3) := by
sorry

theorem equation2_solutions (x : ℝ) :
  ((3 / 4) * x^2 - 2 * x - (1 / 2) = 0) ↔ (x = (4 + Real.sqrt 22) / 3 ∨ x = (4 - Real.sqrt 22) / 3) := by
sorry

end equation1_solutions_equation2_solutions_l269_269096


namespace increasing_interval_of_f_l269_269851

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 2)

theorem increasing_interval_of_f :
  f x = (1/2)^(x^2 - 2) →
  ∀ x, f (x) ≤ f (x + 0.0001) :=
by
  sorry

end increasing_interval_of_f_l269_269851


namespace portrait_in_silver_box_l269_269506

theorem portrait_in_silver_box
  (gold_box : Prop)
  (silver_box : Prop)
  (lead_box : Prop)
  (p : Prop) (q : Prop) (r : Prop)
  (h1 : p ↔ gold_box)
  (h2 : q ↔ ¬silver_box)
  (h3 : r ↔ ¬gold_box)
  (h4 : (p ∨ q ∨ r) ∧ ¬(p ∧ q) ∧ ¬(q ∧ r) ∧ ¬(r ∧ p)) :
  silver_box :=
sorry

end portrait_in_silver_box_l269_269506


namespace determine_amount_of_substance_l269_269393

noncomputable def amount_of_substance 
  (A : ℝ) (R : ℝ) (delta_T : ℝ) : ℝ :=
  (2 * A) / (R * delta_T)

theorem determine_amount_of_substance 
  (A : ℝ := 831) 
  (R : ℝ := 8.31) 
  (delta_T : ℝ := 100) 
  (nu : ℝ := amount_of_substance A R delta_T) :
  nu = 2 := by
  -- Conditions rewritten as definitions
  -- Definition: A = 831 J
  -- Definition: R = 8.31 J/(mol * K)
  -- Definition: delta_T = 100 K
  -- The correct answer to be proven: nu = 2 mol
  sorry

end determine_amount_of_substance_l269_269393


namespace fewer_twos_for_100_l269_269332

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l269_269332


namespace opposite_negative_nine_l269_269707

theorem opposite_negative_nine : 
  (∃ (y : ℤ), -9 + y = 0 ∧ y = 9) :=
by sorry

end opposite_negative_nine_l269_269707


namespace part1_part2_l269_269018

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269018


namespace gcd_min_b_c_l269_269794

theorem gcd_min_b_c (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : Nat.gcd a b = 294) (h2 : Nat.gcd a c = 1155) :
  Nat.gcd b c = 21 :=
sorry

end gcd_min_b_c_l269_269794


namespace sum_of_reciprocals_of_roots_l269_269308

theorem sum_of_reciprocals_of_roots (p q r : ℝ) (hroots : ∀ x, (x = p ∨ x = q ∨ x = r) ↔ (30*x^3 - 50*x^2 + 22*x - 1 = 0)) 
  (h0 : 0 < p ∧ p < 1) (h1 : 0 < q ∧ q < 1) (h2 : 0 < r ∧ r < 1) 
  (hdistinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) : 
  (1 / (1 - p)) + (1 / (1 - q)) + (1 / (1 - r)) = 12 := 
by 
  sorry

end sum_of_reciprocals_of_roots_l269_269308


namespace hourly_wage_increase_is_10_percent_l269_269563

theorem hourly_wage_increase_is_10_percent :
  ∀ (H W : ℝ), 
    ∀ (H' : ℝ), H' = H * (1 - 0.09090909090909092) →
    (H * W = H' * W') →
    (W' = (100 * W) / 90) := by
  sorry

end hourly_wage_increase_is_10_percent_l269_269563


namespace Jason_current_cards_l269_269806

-- Definitions based on the conditions
def Jason_original_cards : ℕ := 676
def cards_bought_by_Alyssa : ℕ := 224

-- Problem statement: Prove that Jason's current number of Pokemon cards is 452
theorem Jason_current_cards : Jason_original_cards - cards_bought_by_Alyssa = 452 := by
  sorry

end Jason_current_cards_l269_269806


namespace max_distance_traveled_l269_269207

theorem max_distance_traveled (front_lifespan : ℕ) (rear_lifespan : ℕ) : 
  front_lifespan = 21000 ∧ rear_lifespan = 28000 → max_possible_distance = 24000 :=
begin
  intros h,
  sorry,
end

end max_distance_traveled_l269_269207


namespace positive_integer_solutions_3x_5y_eq_501_l269_269701

theorem positive_integer_solutions_3x_5y_eq_501 : ∃ n : ℕ, n = 34 ∧ ∀ (x y : ℕ), 3 * x + 5 * y = 501 → x > 0 ∧ y > 0 → n = 34 :=
by
  sorry

end positive_integer_solutions_3x_5y_eq_501_l269_269701


namespace interval_of_monotonic_increase_l269_269698

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem interval_of_monotonic_increase :
  (∃ α : ℝ, power_function α 2 = 4) →
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → power_function 2 x ≤ power_function 2 y) :=
by
  intro h
  sorry

end interval_of_monotonic_increase_l269_269698


namespace factorize_expression_l269_269199

variable (x y : ℝ)

theorem factorize_expression : (x - y)^2 - (3*x^2 - 3*x*y + y^2) = x * (y - 2*x) := 
by
  sorry

end factorize_expression_l269_269199


namespace find_solutions_l269_269771

theorem find_solutions :
  {x : ℝ | 1 / (x^2 + 12 * x - 9) + 1 / (x^2 + 3 * x - 9) + 1 / (x^2 - 14 * x - 9) = 0} = {1, -9, 3, -3} :=
by
  sorry

end find_solutions_l269_269771


namespace product_form_l269_269290

theorem product_form (a b c d : ℤ) :
    (a^2 - 7*b^2) * (c^2 - 7*d^2) = (a*c + 7*b*d)^2 - 7*(a*d + b*c)^2 :=
by sorry

end product_form_l269_269290


namespace part1_part2_l269_269017

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269017


namespace length_of_segment_AC_l269_269640

theorem length_of_segment_AC :
  ∀ (a b h: ℝ),
    (a = b) →
    (h = a * Real.sqrt 2) →
    (4 = (a + b - h) / 2) →
    a = 4 * Real.sqrt 2 + 8 :=
by
  sorry

end length_of_segment_AC_l269_269640


namespace mara_correct_answers_l269_269661

theorem mara_correct_answers :
  let math_total    := 30
  let science_total := 20
  let history_total := 50
  let math_percent  := 0.85
  let science_percent := 0.75
  let history_percent := 0.65
  let math_correct  := math_percent * math_total
  let science_correct := science_percent * science_total
  let history_correct := history_percent * history_total
  let total_correct := math_correct + science_correct + history_correct
  let total_problems := math_total + science_total + history_total
  let overall_percent := total_correct / total_problems
  overall_percent = 0.73 :=
by
  sorry

end mara_correct_answers_l269_269661


namespace final_solution_sugar_percentage_l269_269822

-- Define the conditions of the problem
def initial_solution_sugar_percentage : ℝ := 0.10
def replacement_fraction : ℝ := 0.25
def second_solution_sugar_percentage : ℝ := 0.26

-- Define the Lean statement that proves the final sugar percentage
theorem final_solution_sugar_percentage:
  (0.10 * (1 - 0.25) + 0.26 * 0.25) * 100 = 14 :=
by
  sorry

end final_solution_sugar_percentage_l269_269822


namespace billy_points_difference_l269_269756

-- Condition Definitions
def billy_points : ℕ := 7
def friend_points : ℕ := 9

-- Theorem stating the problem and the solution
theorem billy_points_difference : friend_points - billy_points = 2 :=
by 
  sorry

end billy_points_difference_l269_269756


namespace intervals_of_monotonicity_range_of_values_l269_269226

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x - a * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  -(1 + a) / x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ :=
  f a x - g a x

theorem intervals_of_monotonicity (a : ℝ) (h_pos : 0 < a) :
  (∀ x > 0, x < 1 + a → h a x < h a (1 + a)) ∧
  (∀ x > 1 + a, h a x > h a (1 + a)) :=
sorry

theorem range_of_values (x0 : ℝ) (h_x0 : 1 ≤ x0 ∧ x0 ≤ Real.exp 1) (h_fx_gx : f a x0 < g a x0) :
  a > (Real.exp 1)^2 + 1 / (Real.exp 1 - 1) ∨ a < -2 :=
sorry

end intervals_of_monotonicity_range_of_values_l269_269226


namespace gcd_78_182_l269_269129

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := by
  sorry

end gcd_78_182_l269_269129


namespace simplify_complex_expression_l269_269735

noncomputable def i : ℂ := Complex.I

theorem simplify_complex_expression : 
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i := by 
  sorry

end simplify_complex_expression_l269_269735


namespace part1_part2_l269_269995

-- Part (1) Lean 4 statement
theorem part1 {x : ℕ} (h : 0 < x ∧ 4 * (x + 2) < 18 + 2 * x) : x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 :=
sorry

-- Part (2) Lean 4 statement
theorem part2 (x : ℝ) (h1 : 5 * x + 2 ≥ 4 * x + 1) (h2 : (x + 1) / 4 > (x - 3) / 2 + 1) : -1 ≤ x ∧ x < 3 :=
sorry

end part1_part2_l269_269995


namespace no_such_integers_l269_269086

theorem no_such_integers :
  ¬ (∃ a b c d : ℤ, a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2) :=
by
  sorry

end no_such_integers_l269_269086


namespace minimum_magnitude_l269_269473

noncomputable def smallest_magnitude_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem minimum_magnitude (z : ℂ) (h : Complex.abs (z - 16) + Complex.abs (z + 3 * Complex.I) = 15) :
  smallest_magnitude_z z = (768 / 265 : ℝ) :=
by
  sorry

end minimum_magnitude_l269_269473


namespace change_in_profit_rate_l269_269388

theorem change_in_profit_rate (A B C : Type) (P : ℝ) (r1 r2 : ℝ) (income_increase : ℝ) (capital : ℝ) :
  (A_receives : ℝ) = (2 / 3) → 
  (B_C_divide : ℝ) = (1 - (2 / 3)) / 2 → 
  income_increase = 300 → 
  capital = 15000 →
  ((2 / 3) * capital * (r2 / 100) - (2 / 3) * capital * (r1 / 100)) = income_increase →
  (r2 - r1) = 3 :=
by
  intros
  sorry

end change_in_profit_rate_l269_269388


namespace product_of_two_numbers_l269_269501

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end product_of_two_numbers_l269_269501


namespace nested_radical_solution_l269_269892

noncomputable def nested_radical : ℝ := sqrt 3 - sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - sqrt 3))))))

theorem nested_radical_solution :
  ∃ x : ℝ, x = sqrt (3 - x) ∧ x = ( -1 + sqrt 13 ) / 2 :=
by {
  let x := ( -1 + sqrt 13 ) / 2,
  use x,
  split,
  {
    exact sqrt_sub_eq (3 : ℝ) x,
    sorry
  },
  {
    sorry
  }
}

end nested_radical_solution_l269_269892


namespace Ruby_math_homework_l269_269268

theorem Ruby_math_homework : 
  ∃ M : ℕ, ∃ R : ℕ, R = 2 ∧ 5 * M + 9 * R = 48 ∧ M = 6 := by
  sorry

end Ruby_math_homework_l269_269268


namespace b_investment_less_c_l269_269874

theorem b_investment_less_c (A B C : ℕ) (y : ℕ) (total_investment : ℕ) (profit : ℕ) (A_share : ℕ)
    (h1 : A + B + C = total_investment)
    (h2 : A = B + 6000)
    (h3 : C = B + y)
    (h4 : profit = 8640)
    (h5 : A_share = 3168) :
    y = 3000 :=
by
  sorry

end b_investment_less_c_l269_269874


namespace contrapositive_of_inequality_l269_269106

theorem contrapositive_of_inequality (a b c : ℝ) (h : a > b → a + c > b + c) : a + c ≤ b + c → a ≤ b :=
by
  intro h_le
  apply not_lt.mp
  intro h_gt
  have h2 := h h_gt
  linarith

end contrapositive_of_inequality_l269_269106


namespace sum_of_primes_l269_269235

open Nat

def is_prime (n : ℕ) : Prop := Prime n

theorem sum_of_primes (a b c : ℕ) (h1 : is_prime a) (h2 : is_prime b) (h3 : is_prime c)
  (h4 : b + c = 13) (h5 : c * c - a * a = 72) : a + b + c = 15 := by
  sorry

end sum_of_primes_l269_269235


namespace equal_split_payment_l269_269956

variable (L M N : ℝ)

theorem equal_split_payment (h1 : L < N) (h2 : L > M) : 
  (L + M + N) / 3 - L = (M + N - 2 * L) / 3 :=
by sorry

end equal_split_payment_l269_269956


namespace distance_CD_l269_269075

theorem distance_CD (C D : ℝ × ℝ) (r₁ r₂ : ℝ) (φ₁ φ₂ : ℝ) 
  (hC : C = (r₁, φ₁)) (hD : D = (r₂, φ₂)) (r₁_eq_5 : r₁ = 5) (r₂_eq_12 : r₂ = 12)
  (angle_diff : φ₁ - φ₂ = π / 3) : dist C D = Real.sqrt 109 :=
  sorry

end distance_CD_l269_269075


namespace sum_of_ages_l269_269971

variables (P M Mo : ℕ)

theorem sum_of_ages (h1 : 5 * P = 3 * M)
                    (h2 : 5 * M = 3 * Mo)
                    (h3 : Mo - P = 32) :
  P + M + Mo = 98 :=
by
  sorry

end sum_of_ages_l269_269971


namespace forty_percent_of_jacquelines_candy_bars_is_120_l269_269776

-- Define the number of candy bars Fred has
def fred_candy_bars : ℕ := 12

-- Define the number of candy bars Uncle Bob has
def uncle_bob_candy_bars : ℕ := fred_candy_bars + 6

-- Define the total number of candy bars Fred and Uncle Bob have together
def total_candy_bars : ℕ := fred_candy_bars + uncle_bob_candy_bars

-- Define the number of candy bars Jacqueline has
def jacqueline_candy_bars : ℕ := 10 * total_candy_bars

-- Define the number of candy bars that is 40% of Jacqueline's total
def forty_percent_jacqueline_candy_bars : ℕ := (40 * jacqueline_candy_bars) / 100

-- The statement to prove
theorem forty_percent_of_jacquelines_candy_bars_is_120 :
  forty_percent_jacqueline_candy_bars = 120 :=
sorry

end forty_percent_of_jacquelines_candy_bars_is_120_l269_269776


namespace fraction_simplification_l269_269597

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l269_269597


namespace find_A_plus_B_l269_269074

def isFourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isMultipleOf5 (n : ℕ) : Prop :=
  n % 5 = 0

def countFourDigitOddNumbers : ℕ :=
  ((9 : ℕ) * 10 * 10 * 5)

def countFourDigitMultiplesOf5 : ℕ :=
  ((9 : ℕ) * 10 * 10 * 2)

theorem find_A_plus_B : countFourDigitOddNumbers + countFourDigitMultiplesOf5 = 6300 := by
  sorry

end find_A_plus_B_l269_269074


namespace evaluate_polynomial_at_neg_one_l269_269414

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

-- Define the value x at which we want to evaluate f
def x_val : ℝ := -1

-- State the theorem with the result using Horner's method
theorem evaluate_polynomial_at_neg_one : f x_val = 6 :=
by
  -- Approach to solution is in solution steps, skipped here
  sorry

end evaluate_polynomial_at_neg_one_l269_269414


namespace gallons_in_pond_after_50_days_l269_269369

def initial_amount : ℕ := 500
def evaporation_rate : ℕ := 1
def days_passed : ℕ := 50
def total_evaporation : ℕ := days_passed * evaporation_rate
def final_amount : ℕ := initial_amount - total_evaporation

theorem gallons_in_pond_after_50_days : final_amount = 450 := by
  sorry

end gallons_in_pond_after_50_days_l269_269369


namespace all_statements_true_l269_269943

theorem all_statements_true (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) :
  (a^2 + b^2 < (a + b)^2) ∧ 
  (ab > 0) ∧ 
  (a > b) ∧ 
  (a > 0) ∧
  (b > 0) :=
by
  sorry

end all_statements_true_l269_269943


namespace number_of_sheets_is_9_l269_269379

-- Define the conditions in Lean
variable (n : ℕ) -- Total number of pages

-- The stack is folded and renumbered
axiom folded_and_renumbered : True

-- The sum of numbers on one of the sheets is 74
axiom sum_sheet_is_74 : 2 * n + 2 = 74

-- The number of sheets is the number of pages divided by 4
def number_of_sheets (n : ℕ) : ℕ := n / 4

-- Prove that the number of sheets is 9 given the conditions
theorem number_of_sheets_is_9 : number_of_sheets 36 = 9 :=
by
  sorry

end number_of_sheets_is_9_l269_269379


namespace no_right_obtuse_triangle_l269_269859

theorem no_right_obtuse_triangle :
  ∀ (α β γ : ℝ),
  (α + β + γ = 180) →
  (α = 90 ∨ β = 90 ∨ γ = 90) →
  (α > 90 ∨ β > 90 ∨ γ > 90) →
  false :=
by
  sorry

end no_right_obtuse_triangle_l269_269859


namespace population_percentage_l269_269742

theorem population_percentage (total_population : ℕ) (percentage : ℕ) (result : ℕ) :
  total_population = 25600 → percentage = 90 → result = (percentage * total_population) / 100 → result = 23040 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end population_percentage_l269_269742


namespace part_1_part_2_l269_269007

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269007


namespace theater_loss_l269_269555

-- Define the conditions
def theater_capacity : Nat := 50
def ticket_price : Nat := 8
def tickets_sold : Nat := 24

-- Define the maximum revenue and actual revenue
def max_revenue : Nat := theater_capacity * ticket_price
def actual_revenue : Nat := tickets_sold * ticket_price

-- Define the money lost by not selling out
def money_lost : Nat := max_revenue - actual_revenue

-- Theorem statement to prove
theorem theater_loss : money_lost = 208 := by
  sorry

end theater_loss_l269_269555


namespace largest_even_number_in_series_l269_269445

/-- 
  If the sum of 25 consecutive even numbers is 10,000,
  what is the largest number among these 25 consecutive even numbers? 
-/
theorem largest_even_number_in_series (n : ℤ) (S : ℤ) (h : S = 25 * (n - 24)) (h_sum : S = 10000) :
  n = 424 :=
by {
  sorry -- proof goes here
}

end largest_even_number_in_series_l269_269445


namespace net_loss_is_1_percent_l269_269377

noncomputable def net_loss_percent (CP SP1 SP2 SP3 SP4 : ℝ) : ℝ :=
  let TCP := 4 * CP
  let TSP := SP1 + SP2 + SP3 + SP4
  ((TCP - TSP) / TCP) * 100

theorem net_loss_is_1_percent
  (CP : ℝ)
  (HCP : CP = 1000)
  (SP1 : ℝ)
  (HSP1 : SP1 = CP * 1.1 * 0.95)
  (SP2 : ℝ)
  (HSP2 : SP2 = (CP * 0.9) * 1.02)
  (SP3 : ℝ)
  (HSP3 : SP3 = (CP * 1.2) * 1.03)
  (SP4 : ℝ)
  (HSP4 : SP4 = (CP * 0.75) * 1.01) :
  abs (net_loss_percent CP SP1 SP2 SP3 SP4 + 1.09) < 0.01 :=
by
  -- Proof omitted
  sorry

end net_loss_is_1_percent_l269_269377


namespace trajectory_equation_find_m_l269_269214

-- Define points A and B.
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for P:
def P_condition (P : ℝ × ℝ) : Prop :=
  let PA_len := Real.sqrt ((P.1 - 1)^2 + P.2^2)
  let AB_len := Real.sqrt ((1 - (-1))^2 + (0 - 0)^2)
  let PB_dot_AB := (P.1 + 1) * (-2)
  PA_len * AB_len = PB_dot_AB

-- Problem (1): The trajectory equation
theorem trajectory_equation (P : ℝ × ℝ) (hP : P_condition P) : P.2^2 = 4 * P.1 :=
sorry

-- Define orthogonality condition
def orthogonal (M N : ℝ × ℝ) : Prop := 
  let OM := M
  let ON := N
  OM.1 * ON.1 + OM.2 * ON.2 = 0

-- Problem (2): Finding the value of m
theorem find_m (m : ℝ) (hm1 : m ≠ 0) (hm2 : m < 1) 
  (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1)
  (M N : ℝ × ℝ) (hM : M.2 = M.1 + m) (hN : N.2 = N.1 + m)
  (hMN : orthogonal M N) : m = -4 :=
sorry

end trajectory_equation_find_m_l269_269214


namespace right_triangle_roots_l269_269430

theorem right_triangle_roots (m a b c : ℝ) 
  (h_eq : ∀ x, x^2 - (2 * m + 1) * x + m^2 + m = 0)
  (h_roots : a^2 - (2 * m + 1) * a + m^2 + m = 0 ∧ b^2 - (2 * m + 1) * b + m^2 + m = 0)
  (h_triangle : a^2 + b^2 = c^2)
  (h_c : c = 5) : 
  m = 3 :=
by sorry

end right_triangle_roots_l269_269430


namespace ludvik_favorite_number_l269_269817

variable (a b : ℕ)
variable (ℓ : ℝ)

theorem ludvik_favorite_number (h1 : 2 * a = (b + 12) * ℓ)
(h2 : a - 42 = (b / 2) * ℓ) : ℓ = 7 :=
sorry

end ludvik_favorite_number_l269_269817


namespace fraction_identity_l269_269603

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l269_269603


namespace sarah_score_l269_269286

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end sarah_score_l269_269286


namespace soccer_boys_percentage_l269_269803

theorem soccer_boys_percentage (total_students boys total_playing_soccer girls_not_playing_soccer : ℕ)
  (h_total_students : total_students = 500)
  (h_boys : boys = 350)
  (h_total_playing_soccer : total_playing_soccer = 250)
  (h_girls_not_playing_soccer : girls_not_playing_soccer = 115) :
  (boys - (total_students - total_playing_soccer) / total_playing_soccer * 100) = 86 :=
by
  sorry

end soccer_boys_percentage_l269_269803


namespace animals_percentage_monkeys_l269_269581

theorem animals_percentage_monkeys (initial_monkeys : ℕ) (initial_birds : ℕ) (birds_eaten : ℕ) (final_monkeys : ℕ) (final_birds : ℕ) : 
  initial_monkeys = 6 → 
  initial_birds = 6 → 
  birds_eaten = 2 → 
  final_monkeys = initial_monkeys → 
  final_birds = initial_birds - birds_eaten → 
  (final_monkeys * 100 / (final_monkeys + final_birds) = 60) := 
by intros
   sorry

end animals_percentage_monkeys_l269_269581


namespace Doug_money_l269_269924

theorem Doug_money (B D : ℝ) (h1 : B + 2*B + D = 68) (h2 : 2*B = (3/4)*D) : D = 32 := by
  sorry

end Doug_money_l269_269924


namespace final_speed_is_zero_l269_269569

-- Define physical constants and conversion
def initial_speed_kmh : ℝ := 189
def initial_speed_ms : ℝ := initial_speed_kmh * 0.277778
def deceleration : ℝ := -0.5
def distance : ℝ := 4000

-- The goal is to prove the final speed is 0 m/s
theorem final_speed_is_zero (v_i : ℝ) (a : ℝ) (d : ℝ) (v_f : ℝ) 
  (hv_i : v_i = initial_speed_ms) 
  (ha : a = deceleration) 
  (hd : d = distance) 
  (h : v_f^2 = v_i^2 + 2 * a * d) : 
  v_f = 0 := 
by 
  sorry 

end final_speed_is_zero_l269_269569


namespace weight_of_B_l269_269138

theorem weight_of_B (A B C : ℝ) 
  (h1 : A + B + C = 135) 
  (h2 : A + B = 80) 
  (h3 : B + C = 86) : 
  B = 31 :=
sorry

end weight_of_B_l269_269138


namespace simplify_expression_l269_269967

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 - x / (x + 1)) / ((x ^ 2 - 1) / (x ^ 2 + 2 * x + 1)) = Real.sqrt 2 / 2 :=
by
  rw [h]
  sorry

end simplify_expression_l269_269967


namespace part1_part2_l269_269015

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269015


namespace two_digit_number_determined_l269_269389

theorem two_digit_number_determined
  (x y : ℕ)
  (hx : 0 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (h : 2 * (5 * x - 3) + y = 21) :
  10 * y + x = 72 := 
sorry

end two_digit_number_determined_l269_269389


namespace derive_units_equivalent_to_velocity_l269_269729

-- Define the unit simplifications
def watt := 1 * (1 * (1 * (1 / 1)))
def newton := 1 * (1 * (1 / (1 * 1)))

-- Define the options
def option_A := watt / newton
def option_B := newton / watt
def option_C := watt / (newton * newton)
def option_D := (watt * watt) / newton
def option_E := (newton * newton) / (watt * watt)

-- Define what it means for a unit to be equivalent to velocity
def is_velocity (unit : ℚ) : Prop := unit = (1 * (1 / 1))

theorem derive_units_equivalent_to_velocity :
  is_velocity option_A ∧ 
  ¬ is_velocity option_B ∧ 
  ¬ is_velocity option_C ∧ 
  ¬ is_velocity option_D ∧ 
  ¬ is_velocity option_E := 
by sorry

end derive_units_equivalent_to_velocity_l269_269729


namespace simplify_expression_l269_269839

variable (x y : ℝ)

theorem simplify_expression : (3 * x + 4 * x + 5 * y + 2 * y) = 7 * x + 7 * y :=
by
  sorry

end simplify_expression_l269_269839


namespace quadratic_inequality_l269_269372

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality
  (a b c : ℝ)
  (h_pos : 0 < a)
  (h_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (x : ℝ) :
  f a b c x + f a b c (x - 1) - f a b c (x + 1) > -4 * a :=
  sorry

end quadratic_inequality_l269_269372


namespace athletes_meet_time_number_of_overtakes_l269_269716

-- Define the speeds of the athletes
def speed1 := 155 -- m/min
def speed2 := 200 -- m/min
def speed3 := 275 -- m/min

-- Define the total length of the track
def track_length := 400 -- meters

-- Prove the minimum time for the athletes to meet again is 80/3 minutes
theorem athletes_meet_time (speed1 speed2 speed3 track_length : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) :
  ∃ t : ℚ, t = (80 / 3 : ℚ) :=
by
  sorry

-- Prove the number of overtakes during this time is 13
theorem number_of_overtakes (speed1 speed2 speed3 track_length t : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) (h5 : t = 80 / 3) :
  ∃ n : ℕ, n = 13 :=
by
  sorry

end athletes_meet_time_number_of_overtakes_l269_269716


namespace sin2alpha_div_1_plus_cos2alpha_eq_3_l269_269620

theorem sin2alpha_div_1_plus_cos2alpha_eq_3 (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 3 := 
  sorry

end sin2alpha_div_1_plus_cos2alpha_eq_3_l269_269620


namespace product_expression_l269_269604

theorem product_expression :
  (3^4 - 1) / (3^4 + 1) * (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) * (6^4 - 1) / (6^4 + 1) * (7^4 - 1) / (7^4 + 1) = 880 / 91 := by
sorry

end product_expression_l269_269604


namespace odd_phone_calls_are_even_l269_269364

theorem odd_phone_calls_are_even (n : ℕ) : Even (2 * n) :=
by
  sorry

end odd_phone_calls_are_even_l269_269364


namespace sum_of_all_four_digit_integers_l269_269352

theorem sum_of_all_four_digit_integers :
  (Finset.range (9999 + 1)).filter (λ x => x ≥ 1000).sum = 49495500 :=
by
  sorry

end sum_of_all_four_digit_integers_l269_269352


namespace adam_and_simon_50_miles_apart_l269_269572

noncomputable def time_when_50_miles_apart (x : ℝ) : Prop :=
  let adam_distance := 10 * x
  let simon_distance := 8 * x
  (adam_distance^2 + simon_distance^2 = 50^2) 

theorem adam_and_simon_50_miles_apart : 
  ∃ x : ℝ, time_when_50_miles_apart x ∧ x = 50 / 12.8 := 
sorry

end adam_and_simon_50_miles_apart_l269_269572


namespace fewer_twos_result_100_l269_269320

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l269_269320


namespace find_w_l269_269607

open Matrix

-- Define matrix B
def B : Matrix (Fin 2) (Fin 2) ℚ := !![2, 0; 0, 2]

-- Define w
def w := !![17 / 31; 1]

-- Define the expression B^4 + B^3 + B^2 + B + I
def expr := B^4 + B^3 + B^2 + B + 1

-- The theorem to prove
theorem find_w : expr.mul_vec w = !![17; 31] :=
by
  sorry

end find_w_l269_269607


namespace find_circle_eqn_range_of_slope_l269_269901

noncomputable def circle_eqn_through_points (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop) :=
  ∃ (C : ℝ × ℝ) (r : ℝ),
    C ∈ {P : ℝ × ℝ | line P.1 P.2} ∧
    dist C M = dist C N ∧
    (∀ (P : ℝ × ℝ), dist P C = r ↔ (P = M ∨ P = N))

noncomputable def circle_standard_eqn (C : ℝ × ℝ) (r : ℝ) :=
  ∀ (P : ℝ × ℝ), dist P C = r ↔ (P.1 - C.1)^2 + P.2^2 = r^2

theorem find_circle_eqn (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (h : circle_eqn_through_points M N line) :
  ∃ r : ℝ, circle_standard_eqn (1, 0) r ∧ r = 5 := 
  sorry

theorem range_of_slope (k : ℝ) :
  0 < k → 8 * k^2 - 15 * k > 0 → k > (15 / 8) :=
  sorry

end find_circle_eqn_range_of_slope_l269_269901


namespace fido_reachable_area_l269_269769

theorem fido_reachable_area (r : ℝ) (a b : ℕ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0)
  (h_leash : ∃ (r : ℝ), r > 0) (h_fraction : (a : ℝ) / b * π = π) : a * b = 1 :=
by
  sorry

end fido_reachable_area_l269_269769


namespace part1_part2_l269_269023

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269023


namespace Alan_ate_1_fewer_pretzel_than_John_l269_269714

/-- Given that there are 95 pretzels in a bowl, John ate 28 pretzels, 
Marcus ate 12 more pretzels than John, and Marcus ate 40 pretzels,
prove that Alan ate 1 fewer pretzel than John. -/
theorem Alan_ate_1_fewer_pretzel_than_John 
  (h95 : 95 = 95)
  (John_ate : 28 = 28)
  (Marcus_ate_more : ∀ (x : ℕ), 40 = x + 12 → x = 28)
  (Marcus_ate : 40 = 40) :
  ∃ (Alan : ℕ), Alan = 27 ∧ 28 - Alan = 1 :=
by
  sorry

end Alan_ate_1_fewer_pretzel_than_John_l269_269714


namespace find_x_l269_269936

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (hxy : x + y + x * y = 71) : x = 8 :=
sorry

end find_x_l269_269936


namespace distance_D_to_plane_l269_269454

-- Given conditions about the distances from points A, B, and C to plane M
variables (a b c : ℝ)

-- Formalizing the distance from vertex D to plane M
theorem distance_D_to_plane (a b c : ℝ) : 
  ∃ d : ℝ, d = |a + b + c| ∨ d = |a + b - c| ∨ d = |a - b + c| ∨ d = |-a + b + c| ∨ 
                    d = |a - b - c| ∨ d = |-a - b + c| ∨ d = |-a + b - c| ∨ d = |-a - b - c| := sorry

end distance_D_to_plane_l269_269454


namespace equipment_value_decrease_l269_269712

theorem equipment_value_decrease (a : ℝ) (b : ℝ) (n : ℕ) :
  (a * (1 - b / 100)^n) = a * (1 - b/100)^n :=
sorry

end equipment_value_decrease_l269_269712


namespace smallest_integer_remainder_l269_269724

theorem smallest_integer_remainder (b : ℕ) :
  (b ≡ 2 [MOD 3]) ∧ (b ≡ 3 [MOD 5]) → b = 8 := 
by
  sorry

end smallest_integer_remainder_l269_269724


namespace xiaoming_original_phone_number_l269_269860

variable (d1 d2 d3 d4 d5 d6 : Nat)

def original_phone_number (d1 d2 d3 d4 d5 d6 : Nat) : Nat :=
  100000 * d1 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6

def upgraded_phone_number (d1 d2 d3 d4 d5 d6 : Nat) : Nat :=
  20000000 + 1000000 * d1 + 80000 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6

theorem xiaoming_original_phone_number :
  let x := original_phone_number d1 d2 d3 d4 d5 d6
  let x' := upgraded_phone_number d1 d2 d3 d4 d5 d6
  (x' = 81 * x) → (x = 282500) :=
by
  sorry

end xiaoming_original_phone_number_l269_269860


namespace loaves_per_hour_in_one_oven_l269_269743

-- Define the problem constants and variables
def loaves_in_3_weeks : ℕ := 1740
def ovens : ℕ := 4
def weekday_hours : ℕ := 5
def weekend_hours : ℕ := 2
def weekdays_per_week : ℕ := 5
def weekends_per_week : ℕ := 2
def weeks : ℕ := 3

-- Calculate the total hours per week
def hours_per_week : ℕ := (weekdays_per_week * weekday_hours) + (weekends_per_week * weekend_hours)

-- Calculate the total oven-hours for 3 weeks
def total_oven_hours : ℕ := hours_per_week * ovens * weeks

-- Provide the proof statement
theorem loaves_per_hour_in_one_oven : (loaves_in_3_weeks = 5 * total_oven_hours) :=
by
  sorry -- Proof omitted

end loaves_per_hour_in_one_oven_l269_269743


namespace circle_diameter_l269_269536

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l269_269536


namespace value_of_x_minus_y_l269_269441

theorem value_of_x_minus_y (x y : ℚ) 
    (h₁ : 3 * x - 5 * y = 5) 
    (h₂ : x / (x + y) = 5 / 7) : x - y = 3 := by
  sorry

end value_of_x_minus_y_l269_269441


namespace product_of_two_numbers_l269_269499

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by sorry

end product_of_two_numbers_l269_269499


namespace circle_diameter_l269_269532

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l269_269532


namespace team_incorrect_answers_l269_269798

theorem team_incorrect_answers (total_questions : ℕ) (riley_mistakes : ℕ) 
  (ofelia_correct : ℕ) :
  total_questions = 35 → riley_mistakes = 3 → 
  ofelia_correct = ((total_questions - riley_mistakes) / 2 + 5) → 
  riley_mistakes + (total_questions - ofelia_correct) = 17 :=
by
  intro h1 h2 h3
  sorry

end team_incorrect_answers_l269_269798


namespace range_of_a_l269_269944

noncomputable def f (a x : ℝ) : ℝ := min (Real.exp x - 2) (Real.exp (2 * x) - a * Real.exp x + a + 24)

def has_three_zeros (f : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0

theorem range_of_a (a : ℝ) :
  has_three_zeros (f a) ↔ 12 < a ∧ a < 28 :=
sorry

end range_of_a_l269_269944


namespace angle_C_45_l269_269446

theorem angle_C_45 (A B C : ℝ) 
(h : (Real.cos A + Real.sin A) * (Real.cos B + Real.sin B) = 2) 
(HA : 0 ≤ A) (HB : 0 ≤ B) (HC : 0 ≤ C):
A + B + C = π → 
A = B →
C = π / 2 - B →
C = π / 4 := 
by
  intros;
  sorry

end angle_C_45_l269_269446


namespace probability_red_white_blue_no_replacement_l269_269363

theorem probability_red_white_blue_no_replacement :
  let total_balls := 5 + 6 + 4 + 3
  let prob_red := (5 : ℚ) / total_balls
  let prob_white := (4 : ℚ) / (total_balls - 1)
  let prob_blue := (3 : ℚ) / (total_balls - 2)
  (prob_red * prob_white * prob_blue = (5 : ℚ) / 408) :=
by
  let total_balls := 18
  let prob_red := (5 : ℚ) / total_balls
  let prob_white := (4 : ℚ) / (total_balls - 1)
  let prob_blue := (3 : ℚ) / (total_balls - 2)
  have h : prob_red * prob_white * prob_blue = (5 : ℚ * 4 * 3) / (total_balls * (total_balls - 1) * (total_balls - 2)) :=
    by norm_num
  rw [h]
  norm_num
  sorry

end probability_red_white_blue_no_replacement_l269_269363


namespace abc_eq_l269_269814

theorem abc_eq (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h : 4 * a * b - 1 ∣ (4 * a * a - 1) ^ 2) : a = b :=
sorry

end abc_eq_l269_269814


namespace team_incorrect_answers_l269_269799

theorem team_incorrect_answers (total_questions : ℕ) (riley_mistakes : ℕ) 
  (ofelia_correct : ℕ) :
  total_questions = 35 → riley_mistakes = 3 → 
  ofelia_correct = ((total_questions - riley_mistakes) / 2 + 5) → 
  riley_mistakes + (total_questions - ofelia_correct) = 17 :=
by
  intro h1 h2 h3
  sorry

end team_incorrect_answers_l269_269799


namespace number_of_sheets_in_stack_l269_269383

theorem number_of_sheets_in_stack (n : ℕ) (h1 : 2 * n + 2 = 74) : n / 4 = 9 := 
by
  sorry

end number_of_sheets_in_stack_l269_269383


namespace possible_values_of_m_plus_n_l269_269884

theorem possible_values_of_m_plus_n (m n : ℕ) (hmn_pos : 0 < m ∧ 0 < n) 
  (cond : Nat.lcm m n - Nat.gcd m n = 103) : m + n = 21 ∨ m + n = 105 ∨ m + n = 309 := by
  sorry

end possible_values_of_m_plus_n_l269_269884


namespace part1_part2_l269_269021

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269021


namespace race_distance_l269_269800

theorem race_distance (d x y z : ℝ) 
  (h1: d / x = (d - 25) / y)
  (h2: d / y = (d - 15) / z)
  (h3: d / x = (d - 35) / z) :
  d = 75 :=
sorry

end race_distance_l269_269800


namespace ratio_rate_down_to_up_l269_269552

noncomputable def rate_up (r_up t_up: ℕ) : ℕ := r_up * t_up
noncomputable def rate_down (d_down t_down: ℕ) : ℕ := d_down / t_down
noncomputable def ratio (r_down r_up: ℕ) : ℚ := r_down / r_up

theorem ratio_rate_down_to_up :
  let r_up := 6
  let t_up := 2
  let d_down := 18
  let t_down := 2
  rate_up 6 2 = 12 ∧ rate_down 18 2 = 9 ∧ ratio 9 6 = 3 / 2 :=
by
  sorry

end ratio_rate_down_to_up_l269_269552


namespace inequality_pow4_geq_sum_l269_269478

theorem inequality_pow4_geq_sum (a b c d e : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) :
  (a / b) ^ 4 + (b / c) ^ 4 + (c / d) ^ 4 + (d / e) ^ 4 + (e / a) ^ 4 ≥ 
  (a / b) + (b / c) + (c / d) + (d / e) + (e / a) :=
by
  sorry

end inequality_pow4_geq_sum_l269_269478


namespace Vasek_solved_18_problems_l269_269079

variables (m v z : ℕ)

theorem Vasek_solved_18_problems (h1 : m + v = 25) (h2 : z + v = 32) (h3 : z = 2 * m) : v = 18 := by 
  sorry

end Vasek_solved_18_problems_l269_269079


namespace total_capacity_both_dressers_l269_269668

/-- Definition of drawers and capacity -/
def first_dresser_drawers : ℕ := 12
def first_dresser_capacity_per_drawer : ℕ := 8
def second_dresser_drawers : ℕ := 6
def second_dresser_capacity_per_drawer : ℕ := 10

/-- Theorem stating the total capacity of both dressers -/
theorem total_capacity_both_dressers :
  (first_dresser_drawers * first_dresser_capacity_per_drawer) +
  (second_dresser_drawers * second_dresser_capacity_per_drawer) = 156 :=
by sorry

end total_capacity_both_dressers_l269_269668


namespace opposite_of_neg_nine_is_nine_l269_269703

-- Define the predicate for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that needs to be proved
theorem opposite_of_neg_nine_is_nine : ∃ x, is_opposite (-9) x ∧ x = 9 := 
by {
  -- Use "sorry" to indicate the proof is not provided
  sorry
}

end opposite_of_neg_nine_is_nine_l269_269703


namespace complement_intersection_l269_269816

open Set

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_intersection :
  compl A ∩ B = {-2, -1} :=
by
  sorry

end complement_intersection_l269_269816


namespace vector_magnitude_example_l269_269788

open Real

noncomputable def vec_len (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem vector_magnitude_example :
  let a := (1, -3)
  let b := (-2, 6)
  let c := (x, y)
  let sum_ab := (a.1 + b.1, a.2 + b.2)
  (x, y) ∈ {c : ℝ × ℝ | (c.1 * sum_ab.1 + c.2 * sum_ab.2) = -10 ∧
                     cos (π / 3) = (a.1 * c.1 + a.2 * c.2) / (vec_len a * vec_len c)} →
  vec_len (x, y) = 2 * sqrt 10 :=
by
  intros
  sorry

end vector_magnitude_example_l269_269788


namespace g_range_l269_269475

noncomputable def g (x y z : ℝ) : ℝ :=
  (x ^ 2) / (x ^ 2 + y ^ 2) +
  (y ^ 2) / (y ^ 2 + z ^ 2) +
  (z ^ 2) / (z ^ 2 + x ^ 2)

theorem g_range (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 < g x y z ∧ g x y z < 2 :=
  sorry

end g_range_l269_269475


namespace sarah_score_l269_269287

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end sarah_score_l269_269287


namespace f_monotonically_decreasing_l269_269227

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_monotonically_decreasing : ∀ x, 0 < x ∧ x < 1 / Real.exp 1 → deriv f x < 0 :=
by
  sorry

end f_monotonically_decreasing_l269_269227


namespace Sarahs_score_l269_269283

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end Sarahs_score_l269_269283


namespace smallest_number_mod_conditions_l269_269725

theorem smallest_number_mod_conditions :
  ∃ b : ℕ, b > 0 ∧ b % 3 = 2 ∧ b % 5 = 3 ∧ ∀ n : ℕ, (n > 0 ∧ n % 3 = 2 ∧ n % 5 = 3) → n ≥ b :=
begin
  use 8,
  split,
  { linarith },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  split,
  { exact nat.mod_eq_of_lt (by norm_num) },
  intros n h,
  cases h with n_pos h_cond,
  cases h_cond with h_mod3 h_mod5,
  have h := chinese_remainder_theorem 3 5 (by norm_num) (by norm_num_ge)
  (λ _, by norm_num) (λ _ _, by norm_num),
  specialize h 2 3,
  rcases h ⟨h_mod3, h_mod5⟩ with ⟨m, rfl⟩,
  linarith,
end

end smallest_number_mod_conditions_l269_269725


namespace trigonometric_identity_l269_269187

theorem trigonometric_identity (h1 : cos (70 * Real.pi / 180) ≠ 0) (h2 : sin (70 * Real.pi / 180) ≠ 0) :
  (1 / cos (70 * Real.pi / 180) - (Real.sqrt 3) / sin (70 * Real.pi / 180)) = Real.cot (20 * Real.pi / 180) - 2 :=
by
  sorry

end trigonometric_identity_l269_269187


namespace ellipse_area_l269_269103

theorem ellipse_area (P : ℝ) (b : ℝ) (a : ℝ) (A : ℝ) (h1 : P = 18)
  (h2 : a = b + 4)
  (h3 : A = π * a * b) :
  A = 5 * π :=
by
  sorry

end ellipse_area_l269_269103


namespace fraction_identity_l269_269602

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l269_269602


namespace hex_B2F_to_dec_l269_269606

theorem hex_B2F_to_dec : 
  let A := 10
  let B := 11
  let C := 12
  let D := 13
  let E := 14
  let F := 15
  let base := 16
  let b2f := B * base^2 + 2 * base^1 + F * base^0
  b2f = 2863 :=
by {
  sorry
}

end hex_B2F_to_dec_l269_269606


namespace solution1_solution2_l269_269346

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l269_269346


namespace parallelogram_area_l269_269842

/-- The area of a parallelogram is given by the product of its base and height. 
Given a parallelogram ABCD with base BC of 4 units and height of 2 units, 
prove its area is 8 square units. --/
theorem parallelogram_area (base height : ℝ) (h_base : base = 4) (h_height : height = 2) : 
  base * height = 8 :=
by
  rw [h_base, h_height]
  norm_num
  done

end parallelogram_area_l269_269842


namespace range_of_a_l269_269471

-- Define an odd function f on ℝ such that f(x) = x^2 for x >= 0
noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 else -(x^2)

-- Prove the range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc a (a + 2) → f (x - a) ≥ f (3 * x + 1)) →
  a ≤ -5 := sorry

end range_of_a_l269_269471


namespace solve_for_x_l269_269945

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem solve_for_x :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by 
  sorry

end solve_for_x_l269_269945


namespace count_valid_n_l269_269790

theorem count_valid_n : ∃ (n : ℕ), n < 200 ∧ (∃ (m : ℕ), (m % 4 = 0) ∧ (∃ (k : ℤ), n = 4 * k + 2 ∧ m = 4 * k * (k + 1))) ∧ (∃ k_range : ℕ, k_range = 50) :=
sorry

end count_valid_n_l269_269790


namespace solve_inequality_l269_269694

noncomputable def within_interval (x : ℝ) : Prop :=
  x > -3 ∧ x < 5

theorem solve_inequality (x : ℝ) :
  (x^3 - 125) / (x + 3) < 0 ↔ within_interval x :=
sorry

end solve_inequality_l269_269694


namespace problem1_problem2_l269_269142

-- For problem (1)
noncomputable def f (x : ℝ) := Real.sqrt ((1 - x) / (1 + x))

theorem problem1 (α : ℝ) (h_alpha : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  f (Real.cos α) + f (-Real.cos α) = 2 / Real.sin α := by
  sorry

-- For problem (2)
theorem problem2 : Real.sin (Real.pi * 50 / 180) * (1 + Real.sqrt 3 * Real.tan (Real.pi * 10 / 180)) = 1 := by
  sorry

end problem1_problem2_l269_269142


namespace number_of_sheets_l269_269386

theorem number_of_sheets
  (n : ℕ)
  (h₁ : 2 * n + 2 = 74) :
  n / 4 = 9 :=
by
  sorry

end number_of_sheets_l269_269386


namespace product_of_numbers_l269_269494

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 :=
sorry

end product_of_numbers_l269_269494


namespace parabola_y_coordinate_l269_269857

theorem parabola_y_coordinate (x y : ℝ) :
  x^2 = 4 * y ∧ (x - 0)^2 + (y - 1)^2 = 16 → y = 3 :=
by
  sorry

end parabola_y_coordinate_l269_269857


namespace rotated_angle_new_measure_l269_269298

theorem rotated_angle_new_measure (θ₀ : ℕ) (rotation : ℕ) : (θ₀ = 60) → (rotation = 630) → 
  let θ₁ := θ₀ + (rotation % 360) in
  let acute_angle := 360 - θ₁ % 360 in
  acute_angle = 30 :=
by
  intros h₀ h_rotation
  let θ₁ := 60 + (630 % 360)
  let θ_final := 360 - (θ₁ % 360)
  have : θ_final = 30 := sorry
  exact this

end rotated_angle_new_measure_l269_269298


namespace prob_even_sum_is_one_third_l269_269296

def is_even_sum_first_last (d1 d2 d3 d4 : Nat) : Prop :=
  (d1 + d4) % 2 = 0

def num_unique_arrangements : Nat := 12

def num_favorable_arrangements : Nat := 4

def prob_even_sum_first_last : Rat :=
  num_favorable_arrangements / num_unique_arrangements

theorem prob_even_sum_is_one_third :
  prob_even_sum_first_last = 1 / 3 := 
  sorry

end prob_even_sum_is_one_third_l269_269296


namespace rational_terms_count_l269_269249

noncomputable def number_of_rational_terms (n : ℕ) (x : ℝ) : ℕ :=
  -- The count of rational terms in the expansion
  17

theorem rational_terms_count (n : ℕ) (x : ℝ) :
  (number_of_rational_terms 100 x) = 17 := by
  sorry

end rational_terms_count_l269_269249


namespace equation_solution_l269_269062

theorem equation_solution (x : ℤ) (h : x + 1 = 2) : x = 1 :=
sorry

end equation_solution_l269_269062


namespace cubic_difference_l269_269783

theorem cubic_difference (x y : ℤ) (h1 : x + y = 14) (h2 : 3 * x + y = 20) : x^3 - y^3 = -1304 :=
sorry

end cubic_difference_l269_269783


namespace angle_proof_l269_269209

-- Variables and assumptions
variable {α : Type} [LinearOrderedField α]    -- using a general type for angles
variable {A B C D E : α}                       -- points of the triangle and extended segment

-- Given conditions
variable (angle_ACB angle_ABC : α)
variable (H1 : angle_ACB = 2 * angle_ABC)      -- angle condition
variable (CD BD AD DE : α)
variable (H2 : CD = 2 * BD)                    -- segment length condition
variable (H3 : AD = DE)                        -- extended segment condition

-- The proof goal in Lean format
theorem angle_proof (H1 : angle_ACB = 2 * angle_ABC) 
  (H2 : CD = 2 * BD) 
  (H3 : AD = DE) :
  angle_ECB + 180 = 2 * angle_EBC := 
sorry  -- proof to be filled in

end angle_proof_l269_269209


namespace SarahsScoreIs135_l269_269274

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end SarahsScoreIs135_l269_269274


namespace rotated_angle_l269_269299

theorem rotated_angle (angle_ACB_initial : ℝ) (rotation_angle : ℝ) (h1 : angle_ACB_initial = 60) (h2 : rotation_angle = 630) : 
  ∃ (angle_ACB_new : ℝ), angle_ACB_new = 30 :=
by
  -- Define the effective rotation
  let effective_rotation := rotation_angle % 360 -- Modulo operation
  
  -- Calculate the new angle
  let angle_new := angle_ACB_initial + effective_rotation
  
  -- Ensure the angle is acute by converting if needed
  let acute_angle_new := if angle_new > 180 then 360 - angle_new else angle_new
  
  -- The acute angle should be 30 degrees
  use acute_angle_new
  have : acute_angle_new = 30 := sorry
  exact this

end rotated_angle_l269_269299


namespace find_b_l269_269691

theorem find_b (b n : ℝ) (h_neg : b < 0) :
  (∀ x, x^2 + b * x + 1 / 4 = (x + n)^2 + 1 / 18) → b = - (Real.sqrt 7) / 3 :=
by
  sorry

end find_b_l269_269691


namespace incorrect_mark_l269_269871

theorem incorrect_mark (n : ℕ) (correct_mark incorrect_entry : ℕ) (average_increase : ℕ) :
  n = 40 → correct_mark = 63 → average_increase = 1/2 →
  incorrect_entry - correct_mark = average_increase * n →
  incorrect_entry = 83 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end incorrect_mark_l269_269871


namespace find_original_comic_books_l269_269092

def comic_books (X : ℕ) : Prop :=
  X / 2 + 6 = 13

theorem find_original_comic_books (X : ℕ) (h : comic_books X) : X = 14 :=
by
  sorry

end find_original_comic_books_l269_269092


namespace linear_equation_in_x_l269_269043

theorem linear_equation_in_x (m : ℤ) (h : |m| = 1) (h₂ : m - 1 ≠ 0) : m = -1 :=
sorry

end linear_equation_in_x_l269_269043


namespace value_of_a_even_function_monotonicity_on_interval_l269_269785

noncomputable def f (x : ℝ) := (1 / x^2) + 0 * x

theorem value_of_a_even_function 
  (f : ℝ → ℝ) 
  (h : ∀ x, f (-x) = f x) : 
  (∃ a : ℝ, ∀ x, f x = (1 / x^2) + a * x) → a = 0 := by
  -- Placeholder for the proof
  sorry

theorem monotonicity_on_interval 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (1 / x^2) + 0 * x) 
  (h2 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f x1 > f x2) : 
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f x1 > f x2 := by
  -- Placeholder for the proof
  sorry

end value_of_a_even_function_monotonicity_on_interval_l269_269785


namespace sum_of_four_digit_integers_l269_269350

theorem sum_of_four_digit_integers : 
  let a := 1000 in
  let l := 9999 in
  let n := l - a + 1 in
  (a + l) * n / 2 = 49495500 :=
by
  let a := 1000
  let l := 9999
  let n := l - a + 1
  have h_sum := ((a + l) * n) / 2
  rw [a, l, n] at h_sum
  exact h_sum
  sorry

end sum_of_four_digit_integers_l269_269350


namespace eval_expression_l269_269406

noncomputable def T := (1 / (Real.sqrt 10 - Real.sqrt 8)) + (1 / (Real.sqrt 8 - Real.sqrt 6)) + (1 / (Real.sqrt 6 - Real.sqrt 4))

theorem eval_expression : T = (Real.sqrt 10 + 2 * Real.sqrt 8 + 2 * Real.sqrt 6 + 2) / 2 := 
by
  sorry

end eval_expression_l269_269406


namespace perp_lines_implies_values_l269_269621

variable (a : ℝ)

def line1_perpendicular (a : ℝ) : Prop :=
  (1 - a) * (2 * a + 3) + a * (a - 1) = 0

theorem perp_lines_implies_values (h : line1_perpendicular a) :
  a = 1 ∨ a = -3 :=
by {
  sorry
}

end perp_lines_implies_values_l269_269621


namespace circle_diameter_l269_269525

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l269_269525


namespace theater_loss_l269_269554

-- Define the conditions
def theater_capacity : Nat := 50
def ticket_price : Nat := 8
def tickets_sold : Nat := 24

-- Define the maximum revenue and actual revenue
def max_revenue : Nat := theater_capacity * ticket_price
def actual_revenue : Nat := tickets_sold * ticket_price

-- Define the money lost by not selling out
def money_lost : Nat := max_revenue - actual_revenue

-- Theorem statement to prove
theorem theater_loss : money_lost = 208 := by
  sorry

end theater_loss_l269_269554


namespace homework_time_decrease_l269_269317

variable (x : ℝ)
variable (initial_time final_time : ℝ)
variable (adjustments : ℕ)

def rate_of_decrease (initial_time final_time : ℝ) (adjustments : ℕ) (x : ℝ) := 
  initial_time * (1 - x)^adjustments = final_time

theorem homework_time_decrease 
  (h_initial : initial_time = 100) 
  (h_final : final_time = 70)
  (h_adjustments : adjustments = 2)
  (h_decrease : rate_of_decrease initial_time final_time adjustments x) : 
  100 * (1 - x)^2 = 70 :=
by
  sorry

end homework_time_decrease_l269_269317


namespace initial_interest_rate_l269_269108

variable (P r : ℕ)

theorem initial_interest_rate (h1 : 405 = (P * r) / 100) (h2 : 450 = (P * (r + 5)) / 100) : r = 45 :=
sorry

end initial_interest_rate_l269_269108


namespace option1_cost_expression_option2_cost_expression_cost_comparison_x_20_more_cost_effective_strategy_cost_x_20_l269_269744

def teapot_price : ℕ := 20
def teacup_price : ℕ := 6
def discount_rate : ℝ := 0.9

def option1_cost (x : ℕ) : ℕ :=
  5 * teapot_price + (x - 5) * teacup_price

def option2_cost (x : ℕ) : ℝ :=
  discount_rate * (5 * teapot_price + x * teacup_price)

theorem option1_cost_expression (x : ℕ) (h : x > 5) : option1_cost x = 6 * x + 70 := by
  sorry

theorem option2_cost_expression (x : ℕ) (h : x > 5) : option2_cost x = 5.4 * x + 90 := by
  sorry

theorem cost_comparison_x_20 : option1_cost 20 < option2_cost 20 := by
  sorry

theorem more_cost_effective_strategy_cost_x_20 : (5 * teapot_price + 15 * teacup_price * discount_rate) = 181 := by
  sorry

end option1_cost_expression_option2_cost_expression_cost_comparison_x_20_more_cost_effective_strategy_cost_x_20_l269_269744


namespace simplify_and_evaluate_l269_269693

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) : (x + 5)^2 - (x - 2) * (x + 2) = 9 :=
by
  rw [h]
  -- Continue with standard proof techniques here
  sorry

end simplify_and_evaluate_l269_269693


namespace cost_of_book_l269_269151

theorem cost_of_book (cost_album : ℝ) (discount_rate : ℝ) (cost_CD : ℝ) (cost_book : ℝ) 
  (h1 : cost_album = 20)
  (h2 : discount_rate = 0.30)
  (h3 : cost_CD = cost_album * (1 - discount_rate))
  (h4 : cost_book = cost_CD + 4) :
  cost_book = 18 := by
  sorry

end cost_of_book_l269_269151


namespace distinct_ball_distributions_l269_269233

theorem distinct_ball_distributions : 
  ∃ (distros : Set (Fin 4 → Fin 6)), 
    distros = { f | f.Sum = 5 ∧
                 ∀ m n : Fin 4, f m ≥ f n ∧ 
                 distros = { (5, 0, 0, 0), (4, 1, 0, 0), (3, 2, 0, 0), (3, 1, 1, 0), 
                             (2, 2, 1, 0), (2, 1, 1, 1) }} ∧
    distros.card = 6 :=
sorry

end distinct_ball_distributions_l269_269233


namespace range_of_x_for_direct_above_inverse_l269_269109

-- The conditions
def is_intersection_point (p : ℝ × ℝ) (k1 k2 : ℝ) : Prop :=
  let (x, y) := p
  y = k1 * x ∧ y = k2 / x

-- The main proof that we need to show
theorem range_of_x_for_direct_above_inverse :
  (∃ k1 k2 : ℝ, is_intersection_point (2, -1/3) k1 k2) →
  {x : ℝ | -1/6 * x > -2/(3 * x)} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} :=
by
  intros
  sorry

end range_of_x_for_direct_above_inverse_l269_269109


namespace Sarahs_score_l269_269281

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end Sarahs_score_l269_269281


namespace rotated_number_divisibility_l269_269826

theorem rotated_number_divisibility 
  (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h : 7 ∣ (10^5 * a1 + 10^4 * a2 + 10^3 * a3 + 10^2 * a4 + 10 * a5 + a6)) :
  7 ∣ (10^5 * a6 + 10^4 * a1 + 10^3 * a2 + 10^2 * a3 + 10 * a4 + a5) := 
sorry

end rotated_number_divisibility_l269_269826


namespace penelope_mandm_candies_l269_269669

theorem penelope_mandm_candies (m n : ℕ) (r : ℝ) :
  (m / n = 5 / 3) → (n = 15) → (m = 25) :=
by
  sorry

end penelope_mandm_candies_l269_269669


namespace circle_diameter_problem_circle_diameter_l269_269537

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l269_269537


namespace max_min_values_l269_269772

theorem max_min_values (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  1 ≤ x^2 + 2*y^2 + 3*z^2 ∧ x^2 + 2*y^2 + 3*z^2 ≤ 3 := by
  sorry

end max_min_values_l269_269772


namespace find_side_c_of_triangle_ABC_l269_269065

theorem find_side_c_of_triangle_ABC
  (a b : ℝ)
  (cosA : ℝ)
  (c : ℝ) :
  a = 4 * Real.sqrt 5 →
  b = 5 →
  cosA = 3 / 5 →
  c^2 - 3 * c - 55 = 0 →
  c = 11 := by
  intros ha hb hcosA hquadratic
  sorry

end find_side_c_of_triangle_ABC_l269_269065


namespace probability_at_least_2_defective_is_one_third_l269_269117

noncomputable def probability_at_least_2_defective (good defective : ℕ) (total_selected : ℕ) : ℚ :=
  let total_ways := Nat.choose (good + defective) total_selected
  let ways_2_defective_1_good := Nat.choose defective 2 * Nat.choose good 1
  let ways_3_defective := Nat.choose defective 3
  (ways_2_defective_1_good + ways_3_defective) / total_ways

theorem probability_at_least_2_defective_is_one_third :
  probability_at_least_2_defective 6 4 3 = 1 / 3 :=
by
  sorry

end probability_at_least_2_defective_is_one_third_l269_269117


namespace total_number_of_members_l269_269301

variables (b g : Nat)
def girls_twice_boys : Prop := g = 2 * b
def boys_twice_remaining_girls (b g : Nat) : Prop := b = 2 * (g - 24)

theorem total_number_of_members (b g : Nat) 
  (h1 : girls_twice_boys b g) 
  (h2 : boys_twice_remaining_girls b g) : 
  b + g = 48 := by
  sorry

end total_number_of_members_l269_269301


namespace find_a_l269_269220

theorem find_a (x : ℝ) (hx1 : 0 < x)
  (hx2 : x + 1/x ≥ 2)
  (hx3 : x + 4/x^2 ≥ 3)
  (hx4 : x + 27/x^3 ≥ 4) :
  (x + a/x^4 ≥ 5) → a = 4^4 :=
sorry

end find_a_l269_269220


namespace relationship_between_a_b_c_l269_269905

-- Define the given parabola function
def parabola (x : ℝ) (k : ℝ) : ℝ := -(x - 2)^2 + k

-- Define the points A, B, C with their respective coordinates and expressions on the parabola
variables {a b c k : ℝ}

-- Conditions: Points lie on the parabola
theorem relationship_between_a_b_c (hA : a = parabola (-2) k)
                                  (hB : b = parabola (-1) k)
                                  (hC : c = parabola 3 k) :
  a < b ∧ b < c :=
by
  sorry

end relationship_between_a_b_c_l269_269905


namespace sector_to_cone_base_area_l269_269965

theorem sector_to_cone_base_area
  (r_sector : ℝ) (theta : ℝ) (h1 : r_sector = 2) (h2 : theta = 120) :
  ∃ (A : ℝ), A = (4 / 9) * Real.pi :=
by
  sorry

end sector_to_cone_base_area_l269_269965


namespace part1_part2_l269_269035

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269035


namespace doctor_lindsay_adult_patients_per_hour_l269_269133

def number_of_adult_patients_per_hour (A : ℕ) : Prop :=
  let children_per_hour := 3
  let cost_per_adult := 50
  let cost_per_child := 25
  let daily_income := 2200
  let hours_worked := 8
  let income_per_hour := daily_income / hours_worked
  let income_from_children_per_hour := children_per_hour * cost_per_child
  let income_from_adults_per_hour := A * cost_per_adult
  income_from_adults_per_hour + income_from_children_per_hour = income_per_hour

theorem doctor_lindsay_adult_patients_per_hour : 
  ∃ A : ℕ, number_of_adult_patients_per_hour A ∧ A = 4 :=
sorry

end doctor_lindsay_adult_patients_per_hour_l269_269133


namespace solution1_solution2_l269_269345

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l269_269345


namespace ratio_sum_pqr_uvw_l269_269629

theorem ratio_sum_pqr_uvw (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end ratio_sum_pqr_uvw_l269_269629


namespace sum_of_five_consecutive_integers_l269_269293

theorem sum_of_five_consecutive_integers : ∀ (n : ℤ), (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 5 * n + 20 := 
by
  -- This would be where the proof goes
  sorry

end sum_of_five_consecutive_integers_l269_269293


namespace new_students_count_l269_269294

-- Define the conditions as given in the problem statement.
def original_average_age := 40
def original_number_students := 17
def new_students_average_age := 32
def decreased_age := 36  -- Since the average decreases by 4 years from 40 to 36

-- Let x be the number of new students, the proof problem is to find x.
def find_new_students (x : ℕ) : Prop :=
  original_average_age * original_number_students + new_students_average_age * x = decreased_age * (original_number_students + x)

-- Prove that find_new_students(x) holds for x = 17
theorem new_students_count : find_new_students 17 :=
by
  sorry -- the proof goes here

end new_students_count_l269_269294


namespace number_of_sets_B_l269_269476

def A : Set ℕ := {1, 2, 3}

theorem number_of_sets_B :
  ∃ B : Set ℕ, (A ∪ B = A ∧ 1 ∈ B ∧ (∃ n : ℕ, n = 4)) :=
by
  sorry

end number_of_sets_B_l269_269476


namespace total_seats_l269_269752

theorem total_seats (s : ℝ) : 
  let first_class := 36
  let business_class := 0.30 * s
  let economy_class := (3/5:ℝ) * s
  let premium_economy := s - (first_class + business_class + economy_class)
  first_class + business_class + economy_class + premium_economy = s := by 
  sorry

end total_seats_l269_269752


namespace travel_time_to_Virgo_island_l269_269511

theorem travel_time_to_Virgo_island (boat_time : ℝ) (plane_time : ℝ) (total_time : ℝ) 
  (h1 : boat_time ≤ 2) (h2 : plane_time = 4 * boat_time) (h3 : total_time = plane_time + boat_time) : 
  total_time = 10 :=
by
  sorry

end travel_time_to_Virgo_island_l269_269511


namespace inscribed_square_ratio_l269_269764

theorem inscribed_square_ratio
  (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 13) (h₁ : a^2 + b^2 = c^2)
  (x y : ℝ) (hx : x = 60 / 17) (hy : y = 144 / 17) :
  (x / y) = 5 / 12 := sorry

end inscribed_square_ratio_l269_269764


namespace joe_collected_cards_l269_269808

theorem joe_collected_cards (boxes : ℕ) (cards_per_box : ℕ) (filled_boxes : boxes = 11) (max_cards_per_box : cards_per_box = 8) : boxes * cards_per_box = 88 := by
  sorry

end joe_collected_cards_l269_269808


namespace fewer_twos_to_hundred_l269_269335

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l269_269335


namespace exponent_neg_power_l269_269141

theorem exponent_neg_power (a : ℝ) : -(a^3)^4 = -a^(3 * 4) := 
by
  sorry

end exponent_neg_power_l269_269141


namespace min_value_of_quadratic_l269_269926

theorem min_value_of_quadratic (m : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2 * x + m) 
  (min_val : ∀ x ≥ 2, f x ≥ -2) : m = -2 := 
by
  sorry

end min_value_of_quadratic_l269_269926


namespace sqrt_fraction_eq_half_l269_269178

-- Define the problem statement in a Lean 4 theorem:
theorem sqrt_fraction_eq_half : Real.sqrt ((25 / 36 : ℚ) - (4 / 9 : ℚ)) = 1 / 2 := by
  sorry

end sqrt_fraction_eq_half_l269_269178


namespace ellipse_hyperbola_tangent_l269_269848

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∃ (x y : ℝ), x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y - 2)^2 = 4) →
  m = 45 / 31 :=
by sorry

end ellipse_hyperbola_tangent_l269_269848


namespace negation_of_forall_exp_positive_l269_269300

theorem negation_of_forall_exp_positive :
  ¬ (∀ x : ℝ, Real.exp x > 0) ↔ ∃ x : ℝ, Real.exp x ≤ 0 :=
by {
  sorry
}

end negation_of_forall_exp_positive_l269_269300


namespace treasure_hunt_distance_l269_269875

theorem treasure_hunt_distance (d : ℝ) : 
  (d < 8) → (d > 7) → (d > 9) → False :=
by
  intros h1 h2 h3
  sorry

end treasure_hunt_distance_l269_269875


namespace pyramid_base_side_length_l269_269844

theorem pyramid_base_side_length
  (area : ℝ)
  (slant_height : ℝ)
  (h : area = 90)
  (sh : slant_height = 15) :
  ∃ (s : ℝ), 90 = 1 / 2 * s * 15 ∧ s = 12 :=
by
  sorry

end pyramid_base_side_length_l269_269844


namespace smallest_value_expression_geq_three_l269_269886

theorem smallest_value_expression_geq_three :
  ∀ (x y : ℝ), 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2 ≥ 3 := 
by
  sorry

end smallest_value_expression_geq_three_l269_269886


namespace chipped_marbles_is_22_l269_269838

def bags : List ℕ := [20, 22, 25, 30, 32, 34, 36]

-- Jane and George take some bags and one bag with chipped marbles is left.
theorem chipped_marbles_is_22
  (h1 : ∃ (jane_bags george_bags : List ℕ) (remaining_bag : ℕ),
    (jane_bags ++ george_bags ++ [remaining_bag] = bags ∧
     jane_bags.length = 3 ∧
     (george_bags.length = 2 ∨ george_bags.length = 3) ∧
     3 * remaining_bag = List.sum jane_bags + List.sum george_bags)) :
  ∃ (c : ℕ), c = 22 := 
sorry

end chipped_marbles_is_22_l269_269838


namespace min_value_f_max_value_bac_l269_269912

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| - |x - 1|

theorem min_value_f : ∃ k : ℝ, (∀ x : ℝ, f x ≥ k) ∧ k = -2 := 
by
  sorry

theorem max_value_bac (a b c : ℝ) 
  (h1 : a^2 + c^2 + b^2 / 2 = 2) : 
  ∃ m : ℝ, (∀ a b c : ℝ, a^2 + c^2 + b^2 / 2 = 2 → b * (a + c) ≤ m) ∧ m = 2 := 
by
  sorry

end min_value_f_max_value_bac_l269_269912


namespace probability_of_green_apples_l269_269069

def total_apples : ℕ := 8
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def apples_chosen : ℕ := 3
noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_green_apples :
  (binomial green_apples apples_chosen : ℚ) / (binomial total_apples apples_chosen : ℚ) = 1 / 56 :=
  sorry

end probability_of_green_apples_l269_269069


namespace inv_three_mod_thirty_seven_l269_269611

theorem inv_three_mod_thirty_seven : (3 * 25) % 37 = 1 :=
by
  -- Explicit mention to skip the proof with sorry
  sorry

end inv_three_mod_thirty_seven_l269_269611


namespace problem_statement_l269_269225

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  λ i j,
    match i, j with
    | 0, 0 => 3
    | 0, 1 => 5
    | 1, 0 => 0
    | 1, 1 => -2

def β : Fin 2 → ℤ :=
  λ i,
    match i with
    | 0 => -1
    | 1 => 1

theorem problem_statement : (A ^ 6).mul_vec β = ![ -64, 64 ] :=
by { sorry }

end problem_statement_l269_269225


namespace athletes_meet_time_number_of_overtakes_l269_269715

-- Define the speeds of the athletes
def speed1 := 155 -- m/min
def speed2 := 200 -- m/min
def speed3 := 275 -- m/min

-- Define the total length of the track
def track_length := 400 -- meters

-- Prove the minimum time for the athletes to meet again is 80/3 minutes
theorem athletes_meet_time (speed1 speed2 speed3 track_length : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) :
  ∃ t : ℚ, t = (80 / 3 : ℚ) :=
by
  sorry

-- Prove the number of overtakes during this time is 13
theorem number_of_overtakes (speed1 speed2 speed3 track_length t : ℕ) (h1 : speed1 = 155) (h2 : speed2 = 200) (h3 : speed3 = 275) (h4 : track_length = 400) (h5 : t = 80 / 3) :
  ∃ n : ℕ, n = 13 :=
by
  sorry

end athletes_meet_time_number_of_overtakes_l269_269715


namespace Lorelai_jellybeans_correct_l269_269091

def Gigi_jellybeans : ℕ := 15
def Rory_jellybeans : ℕ := Gigi_jellybeans + 30
def Total_jellybeans : ℕ := Rory_jellybeans + Gigi_jellybeans
def Lorelai_jellybeans : ℕ := 3 * Total_jellybeans

theorem Lorelai_jellybeans_correct : Lorelai_jellybeans = 180 := by
  sorry

end Lorelai_jellybeans_correct_l269_269091


namespace max_distance_l269_269206

theorem max_distance (front_lifespan : ℕ) (rear_lifespan : ℕ)
  (h_front : front_lifespan = 21000)
  (h_rear : rear_lifespan = 28000) :
  ∃ (max_dist : ℕ), max_dist = 24000 :=
by
  sorry

end max_distance_l269_269206


namespace solve_for_x_l269_269947

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem solve_for_x : {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by
  sorry

end solve_for_x_l269_269947


namespace complex_expr_equals_l269_269737

noncomputable def complex_expr : ℂ := (5 * (1 + complex.i^3)) / ((2 + complex.i) * (2 - complex.i))

theorem complex_expr_equals : complex_expr = (1 - complex.i) := 
sorry

end complex_expr_equals_l269_269737


namespace quadratic_polynomial_with_conditions_l269_269201

theorem quadratic_polynomial_with_conditions :
  ∃ (a b c : ℝ), 
  (∀ x : ℂ, x = -3 - 4 * Complex.I ∨ x = -3 + 4 * Complex.I → a * x^2 + b * x + c = 0)
  ∧ b = -10 
  ∧ a = -5/3 
  ∧ c = -125/3 := 
sorry

end quadratic_polynomial_with_conditions_l269_269201


namespace kevin_hops_exact_distance_l269_269810

theorem kevin_hops_exact_distance :
  let a : ℚ := 1 / 4
  let r : ℚ := 3 / 4
  let S_6 : ℚ := a * (1 - r^6) / (1 - r)
  S_6 = 3367 / 4096 :=
by
  let a : ℚ := 1 / 4
  let r : ℚ := 3 / 4
  let S_6 : ℚ := a * (1 - r^6) / (1 - r)
  have : S_6 = 3367 / 4096 := sorry
  exact this

end kevin_hops_exact_distance_l269_269810


namespace student_count_l269_269980

theorem student_count (ratio : ℝ) (teachers : ℕ) (students : ℕ)
  (h1 : ratio = 27.5)
  (h2 : teachers = 42)
  (h3 : ratio * (teachers : ℝ) = students) :
  students = 1155 :=
sorry

end student_count_l269_269980


namespace largest_multiple_of_9_lt_120_is_117_l269_269985

theorem largest_multiple_of_9_lt_120_is_117 : ∃ k : ℕ, 9 * k < 120 ∧ (∀ m : ℕ, 9 * m < 120 → 9 * m ≤ 9 * k) ∧ 9 * k = 117 := 
by 
  sorry

end largest_multiple_of_9_lt_120_is_117_l269_269985


namespace janek_favorite_number_l269_269457

theorem janek_favorite_number (S : Set ℕ) (n : ℕ) :
  S = {6, 8, 16, 22, 32} →
  n / 2 ∈ S →
  (n + 6) ∈ S →
  (n - 10) ∈ S →
  2 * n ∈ S →
  n = 16 := by
  sorry

end janek_favorite_number_l269_269457


namespace sum_of_squares_eq_frac_squared_l269_269237

theorem sum_of_squares_eq_frac_squared (x y z a b c : ℝ) (hxya : x * y = a) (hxzb : x * z = b) (hyzc : y * z = c)
  (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) (ha0 : a ≠ 0) (hb0 : b ≠ 0) (hc0 : c ≠ 0) :
  x^2 + y^2 + z^2 = ((a * b)^2 + (a * c)^2 + (b * c)^2) / (a * b * c) :=
by
  sorry

end sum_of_squares_eq_frac_squared_l269_269237


namespace count_valid_n_num_valid_ns_final_answer_l269_269792

theorem count_valid_n (n m : ℕ) : 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 49 ∧ n = 4 * k + 2 ∧ m = 4 * k * (k + 1)) → 
  m % 4 = 0 ∧ n < 200 :=
by 
  sorry

theorem num_valid_ns : 
  ∃ (count : ℕ), count = 49 ∧ ∀ n m, (∃ k : ℕ, 1 ≤ k ∧ k ≤ 49 ∧ n = 4 * k + 2 ∧ m = 4 * k * (k + 1)) → 
  (m % 4 = 0 ∧ n < 200) :=
by 
  existsi 49
  split
  case h1 : 
    refl
  case h2 : 
    intros n m h
    exact count_valid_n n m h

theorem final_answer : 
  (∃ (n m : ℕ), (∃ k : ℕ, 1 ≤ k ∧ k ≤ 49 ∧ n = 4 * k + 2 ∧ m = 4 * k * (k + 1)) 
  ∧ m % 4 = 0 ∧ n < 200) 
  → (∃ count : ℕ, count = 49) :=
by 
  intro h
  exact num_valid_ns

end count_valid_n_num_valid_ns_final_answer_l269_269792


namespace price_per_glass_second_day_l269_269959

theorem price_per_glass_second_day 
  (O W : ℕ)  -- O is the amount of orange juice used on each day, W is the amount of water used on the first day
  (V : ℕ)   -- V is the volume of one glass
  (P₁ : ℚ)  -- P₁ is the price per glass on the first day
  (P₂ : ℚ)  -- P₂ is the price per glass on the second day
  (h1 : W = O)  -- First day, water is equal to orange juice
  (h2 : V > 0)  -- Volume of one glass > 0
  (h3 : P₁ = 0.48)  -- Price per glass on the first day
  (h4 : (2 * O / V) * P₁ = (3 * O / V) * P₂)  -- Revenue's are the same
  : P₂ = 0.32 :=  -- Prove that price per glass on the second day is 0.32
by
  sorry

end price_per_glass_second_day_l269_269959


namespace samuel_distance_from_hotel_l269_269836

/-- Samuel's driving problem conditions. -/
structure DrivingConditions where
  total_distance : ℕ -- in miles
  first_speed : ℕ -- in miles per hour
  first_time : ℕ -- in hours
  second_speed : ℕ -- in miles per hour
  second_time : ℕ -- in hours

def distance_remaining (c : DrivingConditions) : ℕ :=
  let distance_covered := (c.first_speed * c.first_time) + (c.second_speed * c.second_time)
  c.total_distance - distance_covered

/-- Prove that Samuel is 130 miles from the hotel. -/
theorem samuel_distance_from_hotel : 
  ∀ (c : DrivingConditions), 
    c.total_distance = 600 ∧
    c.first_speed = 50 ∧
    c.first_time = 3 ∧
    c.second_speed = 80 ∧
    c.second_time = 4 → distance_remaining c = 130 := by
  intros c h
  cases h
  sorry

end samuel_distance_from_hotel_l269_269836


namespace problem_statement_l269_269610

theorem problem_statement : (4^4 / 4^3) * 2^8 = 1024 := by
  sorry

end problem_statement_l269_269610


namespace value_of_xyz_l269_269617

variable (x y z : ℝ)

theorem value_of_xyz (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
                     (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 22) 
                     : x * y * z = 14 / 3 := 
sorry

end value_of_xyz_l269_269617


namespace range_of_a_increasing_f_on_interval_l269_269910

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- Define the condition that f(x) is increasing on [4, +∞)
def isIncreasingOnInterval (a : ℝ) : Prop :=
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → f a x ≤ f a y

theorem range_of_a_increasing_f_on_interval :
  (∀ a : ℝ, isIncreasingOnInterval a → a ≥ -3) := 
by
  sorry

end range_of_a_increasing_f_on_interval_l269_269910


namespace andrena_has_more_dolls_than_debelyn_l269_269192

-- Define the initial number of dolls
def initial_dolls_Debelyn : ℕ := 20
def initial_dolls_Christel : ℕ := 24

-- Define the number of dolls given to Andrena
def dolls_given_by_Debelyn : ℕ := 2
def dolls_given_by_Christel : ℕ := 5

-- Define the condition that Andrena has 2 more dolls than Christel after receiving the dolls
def andrena_more_than_christel : ℕ := 2

-- Define the dolls count after gift exchange
def dolls_Debelyn_after : ℕ := initial_dolls_Debelyn - dolls_given_by_Debelyn
def dolls_Christel_after : ℕ := initial_dolls_Christel - dolls_given_by_Christel
def dolls_Andrena_after : ℕ := dolls_Christel_after + andrena_more_than_christel

-- Define the proof problem
theorem andrena_has_more_dolls_than_debelyn : dolls_Andrena_after - dolls_Debelyn_after = 3 := by
  sorry

end andrena_has_more_dolls_than_debelyn_l269_269192


namespace lorelai_jellybeans_correct_l269_269089

-- Define the number of jellybeans Gigi has
def gigi_jellybeans : Nat := 15

-- Define the number of additional jellybeans Rory has compared to Gigi
def rory_additional_jellybeans : Nat := 30

-- Define the number of jellybeans both girls together have
def total_jellybeans : Nat := gigi_jellybeans + (gigi_jellybeans + rory_additional_jellybeans)

-- Define the number of jellybeans Lorelai has eaten
def lorelai_jellybeans : Nat := 3 * total_jellybeans

-- The theorem to prove the number of jellybeans Lorelai has eaten is 180
theorem lorelai_jellybeans_correct : lorelai_jellybeans = 180 := by
  sorry

end lorelai_jellybeans_correct_l269_269089


namespace arithmetic_sequence_sum_l269_269854

theorem arithmetic_sequence_sum (a b : ℤ) (h1 : 10 - 3 = 7)
  (h2 : a = 10 + 7) (h3 : b = 24 + 7) : a + b = 48 :=
by
  sorry

end arithmetic_sequence_sum_l269_269854


namespace initial_distance_between_fred_and_sam_l269_269421

-- Define the conditions as parameters
variables (initial_distance : ℝ)
          (fred_speed sam_speed meeting_distance : ℝ)
          (h_fred_speed : fred_speed = 5)
          (h_sam_speed : sam_speed = 5)
          (h_meeting_distance : meeting_distance = 25)

-- State the theorem
theorem initial_distance_between_fred_and_sam :
  initial_distance = meeting_distance + meeting_distance :=
by
  -- Inline proof structure (sorry means the proof is omitted here)
  sorry

end initial_distance_between_fred_and_sam_l269_269421


namespace rectangular_solid_depth_l269_269125

def SurfaceArea (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem rectangular_solid_depth
  (l w A : ℝ)
  (hl : l = 10)
  (hw : w = 9)
  (hA : A = 408) :
  ∃ h : ℝ, SurfaceArea l w h = A ∧ h = 6 :=
by
  use 6
  sorry

end rectangular_solid_depth_l269_269125


namespace truck_travel_distance_l269_269873

def original_distance : ℝ := 300
def original_gas : ℝ := 10
def increased_efficiency_percent : ℝ := 1.10
def new_gas : ℝ := 15

theorem truck_travel_distance :
  let original_efficiency := original_distance / original_gas;
  let new_efficiency := original_efficiency * increased_efficiency_percent;
  let distance := new_gas * new_efficiency;
  distance = 495 :=
by
  sorry

end truck_travel_distance_l269_269873


namespace distinct_paths_l269_269438

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem distinct_paths (right_steps up_steps : ℕ) : right_steps = 7 → up_steps = 3 →
  binom (right_steps + up_steps) up_steps = 120 := 
by
  intros h1 h2
  rw [h1, h2]
  unfold binom
  simp
  norm_num
  sorry

end distinct_paths_l269_269438


namespace part1_part2_l269_269032

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269032


namespace type_b_quantity_l269_269866

theorem type_b_quantity 
  (x : ℕ)
  (hx : x + 2 * x + 4 * x = 140) : 
  2 * x = 40 := 
sorry

end type_b_quantity_l269_269866


namespace relatively_prime_powers_of_two_l269_269261

theorem relatively_prime_powers_of_two (a : ℤ) (h₁ : a % 2 = 1) (n m : ℕ) (h₂ : n ≠ m) :
  Int.gcd (a^(2^n) + 2^(2^n)) (a^(2^m) + 2^(2^m)) = 1 :=
by
  sorry

end relatively_prime_powers_of_two_l269_269261


namespace pizza_pieces_per_person_l269_269247

theorem pizza_pieces_per_person (total_people : ℕ) (fraction_eat : ℚ) (total_pizza : ℕ) (remaining_pizza : ℕ)
  (H1 : total_people = 15) (H2 : fraction_eat = 3/5) (H3 : total_pizza = 50) (H4 : remaining_pizza = 14) :
  (total_pizza - remaining_pizza) / (fraction_eat * total_people) = 4 :=
by
  -- proof goes here
  sorry

end pizza_pieces_per_person_l269_269247


namespace team_testing_equation_l269_269934

variable (x : ℝ)

theorem team_testing_equation (h : x > 15) : (600 / x = 500 / (x - 15) * 0.9) :=
sorry

end team_testing_equation_l269_269934


namespace number_of_students_taking_french_l269_269932

def total_students : ℕ := 79
def students_taking_german : ℕ := 22
def students_taking_both : ℕ := 9
def students_not_enrolled_in_either : ℕ := 25

theorem number_of_students_taking_french :
  ∃ F : ℕ, (total_students = F + students_taking_german - students_taking_both + students_not_enrolled_in_either) ∧ F = 41 :=
by
  sorry

end number_of_students_taking_french_l269_269932


namespace hose_removal_rate_l269_269102

def pool_volume (length width depth : ℕ) : ℕ :=
  length * width * depth

def draining_rate (volume time : ℕ) : ℕ :=
  volume / time

theorem hose_removal_rate :
  let length := 150
  let width := 80
  let depth := 10
  let total_volume := pool_volume length width depth
  total_volume = 1200000 ∧
  let time := 2000
  draining_rate total_volume time = 600 :=
by
  sorry

end hose_removal_rate_l269_269102


namespace find_vector_v1_v2_l269_269401

noncomputable def point_on_line_l (t : ℝ) : ℝ × ℝ :=
  (2 + 3 * t, 5 + 2 * t)

noncomputable def point_on_line_m (s : ℝ) : ℝ × ℝ :=
  (3 + 5 * s, 7 + 2 * s)

noncomputable def P_foot_of_perpendicular (B : ℝ × ℝ) : ℝ × ℝ :=
  (4, 8)  -- As derived from the given solution

noncomputable def vector_AB (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def vector_PB (P B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - P.1, B.2 - P.2)

theorem find_vector_v1_v2 :
  ∃ (v1 v2 : ℝ), (v1 + v2 = 1) ∧ (vector_PB (P_foot_of_perpendicular (3,7)) (3,7) = (v1, v2)) :=
  sorry

end find_vector_v1_v2_l269_269401


namespace point_B_coordinates_sum_l269_269673

theorem point_B_coordinates_sum (x : ℚ) (h1 : ∃ (B : ℚ × ℚ), B = (x, 5))
    (h2 : (5 - 0) / (x - 0) = 3/4) :
    x + 5 = 35/3 :=
by
  sorry

end point_B_coordinates_sum_l269_269673


namespace direct_proportion_function_l269_269241

-- Define the conditions for the problem
def condition1 (m : ℝ) : Prop := m ^ 2 - 1 = 0
def condition2 (m : ℝ) : Prop := m - 1 ≠ 0

-- The main theorem we need to prove
theorem direct_proportion_function (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = -1 :=
by
  sorry

end direct_proportion_function_l269_269241


namespace hotel_assignment_l269_269747

noncomputable def numberOfWaysToAssignFriends (rooms friends : ℕ) : ℕ :=
  if rooms = 5 ∧ friends = 6 then 7200 else 0

theorem hotel_assignment : numberOfWaysToAssignFriends 5 6 = 7200 :=
by 
  -- This is the condition already matched in the noncomputable function defined above.
  sorry

end hotel_assignment_l269_269747


namespace largest_number_is_A_l269_269356

def numA : ℝ := 0.989
def numB : ℝ := 0.9879
def numC : ℝ := 0.98809
def numD : ℝ := 0.9807
def numE : ℝ := 0.9819

theorem largest_number_is_A :
  (numA > numB) ∧ (numA > numC) ∧ (numA > numD) ∧ (numA > numE) :=
by sorry

end largest_number_is_A_l269_269356


namespace part1_part2_l269_269020

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269020


namespace fewer_twos_result_100_l269_269319

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l269_269319


namespace part_1_part_2_l269_269003

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269003


namespace inverse_of_49_mod_89_l269_269779

theorem inverse_of_49_mod_89 (h : (7 * 55 ≡ 1 [MOD 89])) : (49 * 1 ≡ 1 [MOD 89]) := 
by
  sorry

end inverse_of_49_mod_89_l269_269779


namespace range_of_a_l269_269624

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ (e^x + 1) * (a * x + 2 * a - 2) < 2) → a < 4 / 3 :=
by
  sorry

end range_of_a_l269_269624


namespace train_length_l269_269123

theorem train_length 
  (L : ℝ) -- Length of each train in meters.
  (speed_fast : ℝ := 56) -- Speed of the faster train in km/hr.
  (speed_slow : ℝ := 36) -- Speed of the slower train in km/hr.
  (time_pass : ℝ := 72) -- Time taken for the faster train to pass the slower train in seconds.
  (km_to_m_s : ℝ := 5 / 18) -- Conversion factor from km/hr to m/s.
  (relative_speed : ℝ := (speed_fast - speed_slow) * km_to_m_s) -- Relative speed in m/s.
  (distance_covered : ℝ := relative_speed * time_pass) -- Distance covered in meters.
  (equal_length : 2 * L = distance_covered) -- Condition of the problem: 2L = distance covered.
  : L = 200.16 :=
sorry

end train_length_l269_269123


namespace neg_four_fifth_less_neg_two_third_l269_269181

theorem neg_four_fifth_less_neg_two_third : (-4 : ℚ) / 5 < (-2 : ℚ) / 3 :=
  sorry

end neg_four_fifth_less_neg_two_third_l269_269181


namespace value_of_a_l269_269041

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - 4*a*x + 5*a^2 - 6*a = 0 → 
    ∃ x₁ x₂, x₁ + x₂ = 4*a ∧ x₁ * x₂ = 5*a^2 - 6*a ∧ |x₁ - x₂| = 6)) → a = 3 :=
by {
  sorry
}

end value_of_a_l269_269041


namespace pages_per_brochure_l269_269564

-- Define the conditions
def single_page_spreads := 20
def double_page_spreads := 2 * single_page_spreads
def pages_per_double_spread := 2
def pages_from_single := single_page_spreads
def pages_from_double := double_page_spreads * pages_per_double_spread
def total_pages_from_spreads := pages_from_single + pages_from_double
def ads_per_4_pages := total_pages_from_spreads / 4
def total_ads_pages := ads_per_4_pages
def total_pages := total_pages_from_spreads + total_ads_pages
def brochures := 25

-- The theorem we want to prove
theorem pages_per_brochure : total_pages / brochures = 5 :=
by
  -- This is a placeholder for the actual proof
  sorry

end pages_per_brochure_l269_269564


namespace numberOfBags_l269_269244

-- Define the given conditions
def totalCookies : Nat := 33
def cookiesPerBag : Nat := 11

-- Define the statement to prove
theorem numberOfBags : totalCookies / cookiesPerBag = 3 := by
  sorry

end numberOfBags_l269_269244


namespace not_detecting_spy_probability_l269_269161

-- Definitions based on conditions
def forest_size : ℝ := 10
def detection_radius : ℝ := 10

-- Inoperative detector - assuming NE corner
def detector_NE_inoperative : Prop := true

-- Probability calculation result
def probability_not_detected : ℝ := 0.087

-- Theorem to prove
theorem not_detecting_spy_probability :
  (forest_size = 10) ∧ (detection_radius = 10) ∧ detector_NE_inoperative →
  probability_not_detected = 0.087 :=
by
  sorry

end not_detecting_spy_probability_l269_269161


namespace tan_alpha_values_l269_269424

theorem tan_alpha_values (α : ℝ) (h : 2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + 5 * Real.cos α ^ 2 = 3) : 
  Real.tan α = 1 ∨ Real.tan α = -2 := 
by sorry

end tan_alpha_values_l269_269424


namespace tickets_needed_for_equal_distribution_l269_269399

theorem tickets_needed_for_equal_distribution :
  ∃ k : ℕ, 865 + k ≡ 0 [MOD 9] ∧ k = 8 := sorry

end tickets_needed_for_equal_distribution_l269_269399


namespace total_puppies_count_l269_269877

theorem total_puppies_count (total_cost sale_cost others_cost: ℕ) 
  (three_puppies_on_sale: ℕ) 
  (one_sale_puppy_cost: ℕ)
  (one_other_puppy_cost: ℕ)
  (h1: total_cost = 800)
  (h2: three_puppies_on_sale = 3)
  (h3: one_sale_puppy_cost = 150)
  (h4: others_cost = total_cost - three_puppies_on_sale * one_sale_puppy_cost)
  (h5: one_other_puppy_cost = 175)
  (h6: ∃ other_puppies : ℕ, other_puppies = others_cost / one_other_puppy_cost) :
  ∃ total_puppies : ℕ,
  total_puppies = three_puppies_on_sale + (others_cost / one_other_puppy_cost) := 
sorry

end total_puppies_count_l269_269877


namespace solution1_solution2_l269_269344

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l269_269344


namespace eval_recursive_sqrt_l269_269891

noncomputable def recursive_sqrt : ℝ := 
  let x := sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...))))
  x 

theorem eval_recursive_sqrt : recursive_sqrt = ( -1 + sqrt 13 ) / 2 := 
sorry

end eval_recursive_sqrt_l269_269891


namespace counterexample_to_proposition_l269_269419

theorem counterexample_to_proposition :
  ∃ (angle1 angle2 : ℝ), angle1 + angle2 = 90 ∧ angle1 = angle2 := 
by {
  existsi 45,
  existsi 45,
  split,
  { norm_num },
  { refl }
}

end counterexample_to_proposition_l269_269419


namespace meaningful_square_root_l269_269636

theorem meaningful_square_root (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 := 
by {
  -- This part would contain the proof
  sorry
}

end meaningful_square_root_l269_269636


namespace veronica_pre_selected_photos_l269_269312

-- Definition: Veronica needs to include 3 or 4 of her pictures
def needs_3_or_4_photos : Prop := True

-- Definition: Veronica has pre-selected a certain number of photos
def pre_selected_photos : ℕ := 15

-- Definition: She has 15 choices
def choices : ℕ := 15

-- The proof statement
theorem veronica_pre_selected_photos : needs_3_or_4_photos → choices = pre_selected_photos :=
by
  intros
  sorry

end veronica_pre_selected_photos_l269_269312


namespace pencils_left_with_Harry_l269_269173

theorem pencils_left_with_Harry :
  (let 
    anna_pencils := 50
    harry_initial_pencils := 2 * anna_pencils
    harry_lost_pencils := 19
    harry_pencils_left := harry_initial_pencils - harry_lost_pencils
  in harry_pencils_left = 81) := 
by
  sorry

end pencils_left_with_Harry_l269_269173


namespace telethon_total_revenue_l269_269254

noncomputable def telethon_revenue (first_period_hours : ℕ) (first_period_rate : ℕ) 
  (additional_percent_increase : ℕ) (second_period_hours : ℕ) : ℕ :=
  let first_revenue := first_period_hours * first_period_rate
  let second_period_rate := first_period_rate + (first_period_rate * additional_percent_increase / 100)
  let second_revenue := second_period_hours * second_period_rate
  first_revenue + second_revenue

theorem telethon_total_revenue : 
  telethon_revenue 12 5000 20 14 = 144000 :=
by 
  rfl -- replace 'rfl' with 'sorry' if the proof is non-trivial and longer

end telethon_total_revenue_l269_269254


namespace solution1_solution2_l269_269343

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l269_269343


namespace items_sold_each_house_l269_269071

-- Define the conditions
def visits_day_one : ℕ := 20
def visits_day_two : ℕ := 2 * visits_day_one
def sale_percentage_day_two : ℝ := 0.8
def total_sales : ℕ := 104

-- Define the number of items sold at each house
variable (x : ℕ)

-- Define the main Lean 4 statement for the proof
theorem items_sold_each_house (h1 : 20 * x + 32 * x = 104) : x = 2 :=
by
  -- Proof would go here
  sorry

end items_sold_each_house_l269_269071


namespace point_B_coordinates_sum_l269_269674

theorem point_B_coordinates_sum (x : ℚ) (h1 : ∃ (B : ℚ × ℚ), B = (x, 5))
    (h2 : (5 - 0) / (x - 0) = 3/4) :
    x + 5 = 35/3 :=
by
  sorry

end point_B_coordinates_sum_l269_269674


namespace shuffleboard_total_games_l269_269648

theorem shuffleboard_total_games
    (jerry_wins : ℕ)
    (dave_wins : ℕ)
    (ken_wins : ℕ)
    (h1 : jerry_wins = 7)
    (h2 : dave_wins = jerry_wins + 3)
    (h3 : ken_wins = dave_wins + 5) :
    jerry_wins + dave_wins + ken_wins = 32 := 
by
  sorry

end shuffleboard_total_games_l269_269648


namespace circle_diameter_l269_269543

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l269_269543


namespace sarah_score_l269_269279

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end sarah_score_l269_269279


namespace jerry_age_l269_269820

theorem jerry_age (M J : ℕ) (h1 : M = 2 * J - 2) (h2 : M = 18) : J = 10 := by
  sorry

end jerry_age_l269_269820


namespace samuel_distance_from_hotel_l269_269835

/-- Samuel's driving problem conditions. -/
structure DrivingConditions where
  total_distance : ℕ -- in miles
  first_speed : ℕ -- in miles per hour
  first_time : ℕ -- in hours
  second_speed : ℕ -- in miles per hour
  second_time : ℕ -- in hours

def distance_remaining (c : DrivingConditions) : ℕ :=
  let distance_covered := (c.first_speed * c.first_time) + (c.second_speed * c.second_time)
  c.total_distance - distance_covered

/-- Prove that Samuel is 130 miles from the hotel. -/
theorem samuel_distance_from_hotel : 
  ∀ (c : DrivingConditions), 
    c.total_distance = 600 ∧
    c.first_speed = 50 ∧
    c.first_time = 3 ∧
    c.second_speed = 80 ∧
    c.second_time = 4 → distance_remaining c = 130 := by
  intros c h
  cases h
  sorry

end samuel_distance_from_hotel_l269_269835


namespace find_selling_price_l269_269111

def cost_price : ℝ := 59
def selling_price_for_loss : ℝ := 52
def loss := cost_price - selling_price_for_loss

theorem find_selling_price (sp : ℝ) : (sp - cost_price = loss) → sp = 66 :=
by
  sorry

end find_selling_price_l269_269111


namespace cubic_root_relation_l269_269403

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem cubic_root_relation
  (x1 x2 x3 : ℝ)
  (hx1x2 : x1 < x2)
  (hx2x3 : x2 < 0)
  (hx3pos : 0 < x3)
  (hfx1 : f x1 = 0)
  (hfx2 : f x2 = 0)
  (hfx3 : f x3 = 0) :
  x3^2 - x2^2 = x3 - x1 :=
sorry

end cubic_root_relation_l269_269403


namespace evaluate_expression_l269_269609

theorem evaluate_expression : (8^5 / 8^2) * 2^12 = 2^21 := by
  sorry

end evaluate_expression_l269_269609


namespace circle_diameter_l269_269545

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l269_269545


namespace hyperbola_equation_l269_269660

theorem hyperbola_equation (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_hyperbola : ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) 
  (h_focus : ∃ (p : ℝ × ℝ), p = (1, 0))
  (h_line_passing_focus : ∀ y, ∃ (m c : ℝ), y = -b * y + c)
  (h_parallel : ∀ x y : ℝ, b/a = -b)
  (h_perpendicular : ∀ x y : ℝ, b/a * (-b) = -1) : 
  ∀ x y : ℝ, x^2 - y^2 = 1 :=
by
  sorry

end hyperbola_equation_l269_269660


namespace fewer_twos_to_hundred_l269_269337

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l269_269337


namespace book_cost_l269_269149

theorem book_cost (album_cost : ℝ) (h1 : album_cost = 20) (h2 : ∀ cd_cost, cd_cost = album_cost * 0.7)
  (h3 : ∀ book_cost, book_cost = cd_cost + 4) : book_cost = 18 := by
  sorry

end book_cost_l269_269149


namespace quadratic_root_other_l269_269055

theorem quadratic_root_other (a : ℝ) (h : (3 : ℝ)*3 - 2*3 + a = 0) : 
  ∃ (b : ℝ), b = -1 ∧ (b : ℝ)*b - 2*b + a = 0 :=
by
  sorry

end quadratic_root_other_l269_269055


namespace circle_diameter_l269_269541

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l269_269541


namespace gcd_of_polynomial_and_linear_l269_269426

theorem gcd_of_polynomial_and_linear (b : ℤ) (h1 : b % 2 = 1) (h2 : 1019 ∣ b) : 
  Int.gcd (3 * b ^ 2 + 31 * b + 91) (b + 15) = 1 := 
by 
  sorry

end gcd_of_polynomial_and_linear_l269_269426


namespace abes_age_after_x_years_l269_269981

-- Given conditions
def A : ℕ := 28
def sum_condition (x : ℕ) : Prop := (A + (A - x) = 35)

-- Proof statement
theorem abes_age_after_x_years
  (x : ℕ)
  (h : sum_condition x) :
  (A + x = 49) :=
  sorry

end abes_age_after_x_years_l269_269981


namespace Sarahs_score_l269_269282

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end Sarahs_score_l269_269282


namespace theater_loss_l269_269553

-- Define the conditions
def theater_capacity : Nat := 50
def ticket_price : Nat := 8
def tickets_sold : Nat := 24

-- Define the maximum revenue and actual revenue
def max_revenue : Nat := theater_capacity * ticket_price
def actual_revenue : Nat := tickets_sold * ticket_price

-- Define the money lost by not selling out
def money_lost : Nat := max_revenue - actual_revenue

-- Theorem statement to prove
theorem theater_loss : money_lost = 208 := by
  sorry

end theater_loss_l269_269553


namespace rectangular_field_area_l269_269159

theorem rectangular_field_area (w l A : ℝ) 
  (h1 : l = 3 * w)
  (h2 : 2 * (w + l) = 80) :
  A = w * l → A = 300 :=
by
  sorry

end rectangular_field_area_l269_269159


namespace part1_part2_l269_269030

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269030


namespace roots_inverse_cubed_l269_269787

-- Define the conditions and the problem statement
theorem roots_inverse_cubed (p q m r s : ℝ) (h1 : r + s = -q / p) (h2 : r * s = m / p) 
  (h3 : ∀ x : ℝ, p * x^2 + q * x + m = 0 → x = r ∨ x = s) : 
  1 / r^3 + 1 / s^3 = (-q^3 + 3 * q * m) / m^3 := 
sorry

end roots_inverse_cubed_l269_269787


namespace horizontal_length_circumference_l269_269080

noncomputable def ratio := 16 / 9
noncomputable def diagonal := 32
noncomputable def computed_length := 32 * 16 / (Real.sqrt 337)
noncomputable def computed_perimeter := 2 * (32 * 16 / (Real.sqrt 337) + 32 * 9 / (Real.sqrt 337))

theorem horizontal_length 
  (ratio : ℝ := 16 / 9) (diagonal : ℝ := 32) : 
  32 * 16 / (Real.sqrt 337) = 512 / (Real.sqrt 337) :=
by sorry

theorem circumference 
  (ratio : ℝ := 16 / 9) (diagonal : ℝ := 32) : 
  2 * (32 * 16 / (Real.sqrt 337) + 32 * 9 / (Real.sqrt 337)) = 1600 / (Real.sqrt 337) :=
by sorry

end horizontal_length_circumference_l269_269080


namespace force_is_correct_l269_269802

noncomputable def force_computation : ℝ :=
  let m : ℝ := 5 -- kg
  let s : ℝ → ℝ := fun t => 2 * t + 3 * t^2 -- cm
  let a : ℝ := 6 / 100 -- acceleration in m/s^2
  m * a

theorem force_is_correct : force_computation = 0.3 := 
by
  -- Initial conditions
  sorry

end force_is_correct_l269_269802


namespace union_intersection_l269_269477

-- Define the sets A, B, and C
def A : Set ℕ := {1, 2, 6}
def B : Set ℕ := {2, 4}
def C : Set ℕ := {1, 2, 3, 4}

-- The theorem stating that (A ∪ B) ∩ C = {1, 2, 4}
theorem union_intersection : (A ∪ B) ∩ C = {1, 2, 4} := sorry

end union_intersection_l269_269477


namespace bridge_length_proof_l269_269749

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_of_train_km_per_hr : ℝ) (time_to_cross_bridge : ℝ) : ℝ :=
  let speed_of_train_m_per_s := speed_of_train_km_per_hr * (1000 / 3600)
  let total_distance := speed_of_train_m_per_s * time_to_cross_bridge
  total_distance - length_of_train

theorem bridge_length_proof : length_of_bridge 100 75 11.279097672186225 = 135 := by
  simp [length_of_bridge]
  sorry

end bridge_length_proof_l269_269749


namespace remainder_proof_l269_269355

theorem remainder_proof (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v ∧ v < y) : 
  (x + y * u^2 + 3 * v) % y = 4 * v % y :=
by
  sorry

end remainder_proof_l269_269355


namespace distance_from_hotel_l269_269833

def total_distance := 600
def speed1 := 50
def time1 := 3
def speed2 := 80
def time2 := 4

theorem distance_from_hotel :
  total_distance - (speed1 * time1 + speed2 * time2) = 130 := 
by
  sorry

end distance_from_hotel_l269_269833


namespace negation_of_existential_proposition_l269_269700

-- Define the propositions
def proposition (x : ℝ) := x^2 - 2 * x + 1 ≤ 0

-- Define the negation of the propositions
def negation_prop (x : ℝ) := x^2 - 2 * x + 1 > 0

-- Theorem to prove that the negation of the existential proposition is the universal proposition
theorem negation_of_existential_proposition
  (h : ¬ ∃ x : ℝ, proposition x) :
  ∀ x : ℝ, negation_prop x :=
by
  sorry

end negation_of_existential_proposition_l269_269700


namespace eval_expression_l269_269407

theorem eval_expression : (500 * 500) - (499 * 501) = 1 := by
  sorry

end eval_expression_l269_269407


namespace part1_part2_l269_269002

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l269_269002


namespace decreased_value_of_expression_l269_269456

theorem decreased_value_of_expression (x y z : ℝ) :
  let x' := 0.6 * x
  let y' := 0.6 * y
  let z' := 0.6 * z
  (x' * y' * z'^2) = 0.1296 * (x * y * z^2) :=
by
  sorry

end decreased_value_of_expression_l269_269456


namespace fraction_simplification_l269_269594

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l269_269594


namespace trajectory_of_midpoint_l269_269311

theorem trajectory_of_midpoint
  (M : ℝ × ℝ)
  (P : ℝ × ℝ) (Q : ℝ × ℝ)
  (hP : P = (4, 0))
  (hQ : Q.1^2 + Q.2^2 = 4)
  (M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 - 2)^2 + M.2^2 = 1 :=
sorry

end trajectory_of_midpoint_l269_269311


namespace player_A_winning_strategy_l269_269118

-- Define the game state and the player's move
inductive Move
| single (index : Nat) : Move
| double (index : Nat) : Move

-- Winning strategy prop
def winning_strategy (n : Nat) (first_player : Bool) : Prop :=
  ∀ moves : List Move, moves.length ≤ n → (first_player → false) → true

-- Main theorem stating that player A always has a winning strategy
theorem player_A_winning_strategy (n : Nat) (h : n ≥ 1) : winning_strategy n true := 
by 
  -- directly prove the statement
  sorry

end player_A_winning_strategy_l269_269118


namespace scientific_notation_21500000_l269_269447

/-- Express the number 21500000 in scientific notation. -/
theorem scientific_notation_21500000 : 21500000 = 2.15 * 10^7 := 
sorry

end scientific_notation_21500000_l269_269447


namespace determine_contents_l269_269120

inductive Color
| White
| Black

open Color

-- Definitions of the mislabeled boxes
def mislabeled (box : Nat → List Color) : Prop :=
  ¬ (box 1 = [Black, Black] ∧ box 2 = [Black, White]
     ∧ box 3 = [White, White])

-- Draw a ball from a box revealing its content
def draw_ball (box : Nat → List Color) (i : Nat) (c : Color) : Prop :=
  c ∈ box i

-- theorem statement
theorem determine_contents (box : Nat → List Color) (c : Color) (h : draw_ball box 3 c) (hl : mislabeled box) :
  (c = White → box 3 = [White, White] ∧ box 2 = [Black, White] ∧ box 1 = [Black, Black]) ∧
  (c = Black → box 3 = [Black, Black] ∧ box 2 = [Black, White] ∧ box 1 = [White, White]) :=
by
  sorry

end determine_contents_l269_269120


namespace cost_of_flight_XY_l269_269841

theorem cost_of_flight_XY :
  let d_XY : ℕ := 4800
  let booking_fee : ℕ := 150
  let cost_per_km : ℚ := 0.12
  ∃ cost : ℚ, cost = d_XY * cost_per_km + booking_fee ∧ cost = 726 := 
by
  sorry

end cost_of_flight_XY_l269_269841


namespace total_amount_spent_l269_269865

noncomputable def food_price : ℝ := 160
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def tip_rate : ℝ := 0.20

theorem total_amount_spent :
  let sales_tax := sales_tax_rate * food_price
  let total_before_tip := food_price + sales_tax
  let tip := tip_rate * total_before_tip
  let total_amount := total_before_tip + tip
  total_amount = 211.20 :=
by
  -- include the proof logic here if necessary
  sorry

end total_amount_spent_l269_269865


namespace solve_logarithmic_equation_l269_269979

/-- The solution to the equation log_2(9^x - 5) = 2 + log_2(3^x - 2) is x = 1. -/
theorem solve_logarithmic_equation (x : ℝ) :
  (Real.logb 2 (9^x - 5) = 2 + Real.logb 2 (3^x - 2)) → x = 1 :=
by
  sorry

end solve_logarithmic_equation_l269_269979


namespace problem1_problem2_l269_269911

noncomputable def f (x : ℝ) : ℝ :=
  if h : 1 ≤ x then x else 1 / x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  a * f x - |x - 2|

def problem1_statement (b : ℝ) : Prop :=
  ∀ x, x > 0 → g x 0 ≤ |x - 1| + b

def problem2_statement : Prop :=
  ∃ x, (0 < x) ∧ ∀ y, (0 < y) → g y 1 ≥ g x 1

theorem problem1 : ∀ b : ℝ, problem1_statement b ↔ b ∈ Set.Ici (-1) := sorry

theorem problem2 : ∃ x, problem2_statement ∧ g x 1 = 0 := sorry

end problem1_problem2_l269_269911


namespace polygon_side_count_eq_six_l269_269057

theorem polygon_side_count_eq_six (n : ℕ) 
  (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_side_count_eq_six_l269_269057


namespace two_thirds_of_x_eq_36_l269_269514

theorem two_thirds_of_x_eq_36 (x : ℚ) (h : (2 / 3) * x = 36) : x = 54 :=
by
  sorry

end two_thirds_of_x_eq_36_l269_269514


namespace solve_for_x_l269_269946

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem solve_for_x :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by 
  sorry

end solve_for_x_l269_269946


namespace work_ratio_of_man_to_boy_l269_269567

theorem work_ratio_of_man_to_boy 
  (M B : ℝ) 
  (work : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = work)
  (h2 : (13 * M + 24 * B) * 4 = work) :
  M / B = 2 :=
by 
  sorry

end work_ratio_of_man_to_boy_l269_269567


namespace initial_spinach_volume_l269_269819

theorem initial_spinach_volume (S : ℝ) (h1 : 0.20 * S + 6 + 4 = 18) : S = 40 :=
by
  sorry

end initial_spinach_volume_l269_269819


namespace gcd_gx_x_is_210_l269_269039

-- Define the conditions
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, y = k * x

-- The main proof problem
theorem gcd_gx_x_is_210 (x : ℕ) (hx : is_multiple_of 17280 x) :
  Nat.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (4 * x + 5)) x = 210 :=
by
  sorry

end gcd_gx_x_is_210_l269_269039


namespace fraction_simplification_l269_269596

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l269_269596


namespace isosceles_triangle_vertex_angle_l269_269935

theorem isosceles_triangle_vertex_angle (A B C : ℝ) (hABC : A + B + C = 180) (h_iso : A = B ∨ B = C ∨ A = C) (h_angle : A = 50 ∨ B = 50 ∨ C = 50) : (A = 50 ∨ A = 80) ∨ (B = 50 ∨ B = 80) ∨ (C = 50 ∨ C = 80) :=
by sorry

end isosceles_triangle_vertex_angle_l269_269935


namespace intersection_A_B_l269_269915

-- Define sets A and B
def A : Set ℤ := {1, 3, 5}
def B : Set ℤ := {-1, 0, 1}

-- Prove that the intersection of A and B is {1}
theorem intersection_A_B : A ∩ B = {1} := by 
  sorry

end intersection_A_B_l269_269915


namespace sum_coords_B_l269_269677

def point (A B : ℝ × ℝ) : Prop :=
A = (0, 0) ∧ B.snd = 5 ∧ (B.snd - A.snd) / (B.fst - A.fst) = 3 / 4

theorem sum_coords_B (x : ℝ) :
  point (0, 0) (x, 5) → x + 5 = 35 / 3 :=
by
  intro h
  sorry

end sum_coords_B_l269_269677


namespace point_B_coordinates_sum_l269_269675

theorem point_B_coordinates_sum (x : ℚ) (h1 : ∃ (B : ℚ × ℚ), B = (x, 5))
    (h2 : (5 - 0) / (x - 0) = 3/4) :
    x + 5 = 35/3 :=
by
  sorry

end point_B_coordinates_sum_l269_269675


namespace greatest_divisor_l269_269613

theorem greatest_divisor (d : ℕ) (h₀ : 1657 % d = 6) (h₁ : 2037 % d = 5) : d = 127 :=
by
  -- Proof skipped
  sorry

end greatest_divisor_l269_269613


namespace logic_problem_l269_269429

variable (p q : Prop)

theorem logic_problem (h₁ : p ∨ q) (h₂ : ¬ p) : ¬ p ∧ q :=
by
  sorry

end logic_problem_l269_269429


namespace fewer_twos_for_100_l269_269328

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l269_269328


namespace xy_inequality_l269_269941

theorem xy_inequality (x y : ℝ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  x * y ≤ (x^(n+2) + y^(n+2)) / (x^n + y^n) :=
sorry

end xy_inequality_l269_269941


namespace find_numbers_l269_269855

theorem find_numbers (u v : ℝ) (h1 : u^2 + v^2 = 20) (h2 : u * v = 8) :
  (u = 2 ∧ v = 4) ∨ (u = 4 ∧ v = 2) ∨ (u = -2 ∧ v = -4) ∨ (u = -4 ∧ v = -2) := by
sorry

end find_numbers_l269_269855


namespace part_1_part_2_l269_269012

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269012


namespace quadratic_decreasing_conditions_l269_269045

theorem quadratic_decreasing_conditions (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → ∃ y : ℝ, y = ax^2 + 4*(a+1)*x - 3 ∧ (∀ z : ℝ, z ≥ x → y ≥ (ax^2 + 4*(a+1)*z - 3))) ↔ a ∈ Set.Iic (-1 / 2) :=
sorry

end quadratic_decreasing_conditions_l269_269045


namespace solution_set_correct_l269_269306

def inequality_solution (x : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3)^2 > 0

theorem solution_set_correct : 
  ∀ x : ℝ, inequality_solution x ↔ (x < 1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) ∨ x > 3) := 
by sorry

end solution_set_correct_l269_269306


namespace fewer_twos_to_hundred_l269_269339

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l269_269339


namespace fewer_twos_result_100_l269_269322

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l269_269322


namespace sum_c_eq_l269_269211

-- Definitions and conditions
def a_n : ℕ → ℝ := λ n => 2 ^ n
def b_n : ℕ → ℝ := λ n => 2 * n
def c_n (n : ℕ) : ℝ := a_n n * b_n n

-- Sum of the first n terms of sequence {c_n}
def sum_c (n : ℕ) : ℝ := (Finset.range n).sum c_n

-- Theorem statement
theorem sum_c_eq (n : ℕ) : sum_c n = (n - 1) * 2 ^ (n + 2) + 4 :=
sorry

end sum_c_eq_l269_269211


namespace sum_even_odd_probability_l269_269122

theorem sum_even_odd_probability :
  (∀ (a b : ℕ), ∃ (P_even P_odd : ℚ),
    P_even = 1/2 ∧ P_odd = 1/2 ∧
    (a % 2 = 0 ∧ b % 2 = 0 ↔ (a + b) % 2 = 0) ∧
    (a % 2 = 1 ∧ b % 2 = 1 ↔ (a + b) % 2 = 0) ∧
    ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0) ↔ (a + b) % 2 = 1)) :=
sorry

end sum_even_odd_probability_l269_269122


namespace fewer_twos_to_hundred_l269_269333

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l269_269333


namespace rosa_peaches_more_than_apples_l269_269459

def steven_peaches : ℕ := 17
def steven_apples  : ℕ := 16
def jake_peaches : ℕ := steven_peaches - 6
def jake_apples  : ℕ := steven_apples + 8
def rosa_peaches : ℕ := 3 * jake_peaches
def rosa_apples  : ℕ := steven_apples / 2

theorem rosa_peaches_more_than_apples : rosa_peaches - rosa_apples = 25 := by
  sorry

end rosa_peaches_more_than_apples_l269_269459


namespace part1_part2_l269_269031

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269031


namespace probability_shots_result_l269_269578

open ProbabilityTheory

noncomputable def P_A := 3 / 4
noncomputable def P_B := 4 / 5
noncomputable def P_not_A := 1 - P_A
noncomputable def P_not_B := 1 - P_B

theorem probability_shots_result :
    (P_not_A * P_not_B * P_A) + (P_not_A * P_not_B * P_not_A * P_B) = 19 / 400 :=
    sorry

end probability_shots_result_l269_269578


namespace max_value_a4_b4_c4_d4_l269_269470

theorem max_value_a4_b4_c4_d4 (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 16) :
  a^4 + b^4 + c^4 + d^4 ≤ 64 :=
sorry

end max_value_a4_b4_c4_d4_l269_269470


namespace number_of_sheets_in_stack_l269_269381

theorem number_of_sheets_in_stack (n : ℕ) (h1 : 2 * n + 2 = 74) : n / 4 = 9 := 
by
  sorry

end number_of_sheets_in_stack_l269_269381


namespace product_of_two_numbers_l269_269498

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by sorry

end product_of_two_numbers_l269_269498


namespace andy_cavities_l269_269576

/-- Andy gets a cavity for every 4 candy canes he eats.
  He gets 2 candy canes from his parents, 
  3 candy canes each from 4 teachers.
  He uses his allowance to buy 1/7 as many candy canes as he was given.
  Prove the total number of cavities Andy gets from eating all his candy canes is 4. -/
theorem andy_cavities :
  let canes_per_teach := 3
  let teachers := 4
  let canes_from_parents := 2
  let bought_fraction := 1 / 7
  let cavities_per_cane := 1 / 4 in
  let canes_from_teachers := canes_per_teach * teachers in
  let total_given := canes_from_teachers + canes_from_parents in
  let canes_bought := total_given * bought_fraction in
  let total_canes := total_given + canes_bought in
  let cavities := total_canes * cavities_per_cane in
  cavities = 4 := 
begin
  sorry
end

end andy_cavities_l269_269576


namespace weight_of_dog_l269_269157

theorem weight_of_dog (k r d : ℕ) (h1 : k + r + d = 30) (h2 : k + r = 2 * d) (h3 : k + d = r) : d = 10 :=
by
  sorry

end weight_of_dog_l269_269157


namespace number_of_sheets_in_stack_l269_269382

theorem number_of_sheets_in_stack (n : ℕ) (h1 : 2 * n + 2 = 74) : n / 4 = 9 := 
by
  sorry

end number_of_sheets_in_stack_l269_269382


namespace probA4Wins_probFifthGame_probCWins_l269_269983

-- Definitions for conditions
def player : Type := A | B | C
def initialMatch : (player × player) := (A, B)
def winProb : ℚ := 1 / 2
def loseTwoConsecutive (p1 p2 : player) : Prop := sorry  -- Definition of losing two consecutive games needed

-- Part (1): Probability of A winning four consecutive games is 1/16.
theorem probA4Wins : 
  let prob := (winProb ^ 4)
  prob = 1 / 16 :=
by
  sorry

-- Part (2): Probability of needing a fifth game to be played is 3/4.
theorem probFifthGame :
  let probEndIn4Games := 4 * (winProb ^ 4)
  let prob := 1 - probEndIn4Games
  prob = 3 / 4 :=
by
  sorry

-- Part (3): Probability of C being the ultimate winner is 7/16.
theorem probCWins :
  let prob := 7 / 16
  prob = 7 / 16 :=
by
  sorry

end probA4Wins_probFifthGame_probCWins_l269_269983


namespace part1_part2_l269_269025

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269025


namespace maximize_revenue_l269_269152

-- Define the conditions
def total_time_condition (x y : ℝ) : Prop := x + y ≤ 300
def total_cost_condition (x y : ℝ) : Prop := 2.5 * x + y ≤ 4500
def non_negative_condition (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0

-- Define the revenue function
def revenue (x y : ℝ) : ℝ := 0.3 * x + 0.2 * y

-- The proof statement
theorem maximize_revenue : 
  ∃ x y, total_time_condition x y ∧ total_cost_condition x y ∧ non_negative_condition x y ∧ 
  revenue x y = 70 := 
by
  sorry

end maximize_revenue_l269_269152


namespace man_swim_upstream_distance_l269_269158

theorem man_swim_upstream_distance (dist_downstream : ℝ) (time_downstream : ℝ) (time_upstream : ℝ) (speed_still_water : ℝ) 
  (effective_speed_downstream : ℝ) (speed_current : ℝ) (effective_speed_upstream : ℝ) (dist_upstream : ℝ) :
  dist_downstream = 36 →
  time_downstream = 6 →
  time_upstream = 6 →
  speed_still_water = 4.5 →
  effective_speed_downstream = dist_downstream / time_downstream →
  effective_speed_downstream = speed_still_water + speed_current →
  effective_speed_upstream = speed_still_water - speed_current →
  dist_upstream = effective_speed_upstream * time_upstream →
  dist_upstream = 18 :=
by
  intros h_dist_downstream h_time_downstream h_time_upstream h_speed_still_water
         h_effective_speed_downstream h_eq_speed_current h_effective_speed_upstream h_dist_upstream
  sorry

end man_swim_upstream_distance_l269_269158


namespace deficit_calculation_l269_269641

theorem deficit_calculation
    (L W : ℝ)  -- Length and Width
    (dW : ℝ)  -- Deficit in width
    (h1 : (1.08 * L) * (W - dW) = 1.026 * (L * W))  -- Condition on the calculated area
    : dW / W = 0.05 := 
by
    sorry

end deficit_calculation_l269_269641


namespace goose_eggs_count_l269_269358

theorem goose_eggs_count 
  (E : ℕ) 
  (hatch_rate : ℚ)
  (survive_first_month_rate : ℚ)
  (survive_first_year_rate : ℚ)
  (geese_survived_first_year : ℕ)
  (no_more_than_one_goose_per_egg : Prop) 
  (hatch_eq : hatch_rate = 2/3) 
  (survive_first_month_eq : survive_first_month_rate = 3/4) 
  (survive_first_year_eq : survive_first_year_rate = 2/5) 
  (geese_survived_eq : geese_survived_first_year = 130):
  E = 650 :=
by
  sorry

end goose_eggs_count_l269_269358


namespace exists_positive_integers_for_hexagon_area_l269_269174

theorem exists_positive_integers_for_hexagon_area (S : ℕ) (a b : ℕ) (hS : S = 2016) :
  2 * (a^2 + b^2 + a * b) = S → ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 2 * (a^2 + b^2 + a * b) = S :=
by
  sorry

end exists_positive_integers_for_hexagon_area_l269_269174


namespace students_not_taking_either_l269_269449

-- Definitions of the conditions
def total_students : ℕ := 28
def students_taking_french : ℕ := 5
def students_taking_spanish : ℕ := 10
def students_taking_both : ℕ := 4

-- Theorem stating the mathematical problem
theorem students_not_taking_either :
  total_students - (students_taking_french + students_taking_spanish + students_taking_both) = 9 :=
sorry

end students_not_taking_either_l269_269449


namespace tickets_to_be_sold_l269_269310

theorem tickets_to_be_sold : 
  let total_tickets := 200
  let jude_tickets := 16
  let andrea_tickets := 4 * jude_tickets
  let sandra_tickets := 2 * jude_tickets + 8
  total_tickets - (jude_tickets + andrea_tickets + sandra_tickets) = 80 := by
  sorry

end tickets_to_be_sold_l269_269310


namespace trigonometric_identity_l269_269190

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = (1 / (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180))) :=
by
  sorry

end trigonometric_identity_l269_269190


namespace apples_in_second_group_l269_269095

theorem apples_in_second_group : 
  ∀ (A O : ℝ) (x : ℕ), 
  6 * A + 3 * O = 1.77 ∧ x * A + 5 * O = 1.27 ∧ A = 0.21 → 
  x = 2 :=
by
  intros A O x h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end apples_in_second_group_l269_269095


namespace fencing_required_l269_269862

theorem fencing_required (L W : ℝ) (hL : L = 20) (hA : 20 * W = 60) : 2 * W + L = 26 :=
by
  sorry

end fencing_required_l269_269862


namespace baron_munchausen_failed_l269_269580

theorem baron_munchausen_failed : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → ¬∃ (d1 d2 : ℕ), ∃ (k : ℕ), n * 100 + (d1 * 10 + d2) = k^2 := 
by
  intros n hn
  obtain ⟨h10, h99⟩ := hn
  sorry

end baron_munchausen_failed_l269_269580


namespace value_a2_plus_b2_l269_269442

noncomputable def a_minus_b : ℝ := 8
noncomputable def ab : ℝ := 49.99999999999999

theorem value_a2_plus_b2 (a b : ℝ) (h1 : a - b = a_minus_b) (h2 : a * b = ab) :
  a^2 + b^2 = 164 := by
  sorry

end value_a2_plus_b2_l269_269442


namespace thirty_percent_less_eq_one_fourth_more_l269_269119

theorem thirty_percent_less_eq_one_fourth_more (x : ℝ) (hx1 : 0.7 * 90 = 63) (hx2 : (5 / 4) * x = 63) : x = 50 :=
sorry

end thirty_percent_less_eq_one_fourth_more_l269_269119


namespace simplify_expression_l269_269439

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : 
  (3 * b + 7 - 5 * b) / 3 = (-2 / 3) * b + (7 / 3) :=
by
  sorry

end simplify_expression_l269_269439


namespace sticker_distribution_ways_l269_269436

theorem sticker_distribution_ways : 
  ∃ ways : ℕ, ways = Nat.choose (9) (4) ∧ ways = 126 :=
by
  sorry

end sticker_distribution_ways_l269_269436


namespace find_number_l269_269524

theorem find_number (a : ℤ) (h : a - a + 99 * (a - 99) = 19802) : a = 299 := 
by 
  sorry

end find_number_l269_269524


namespace distinct_integers_integer_expression_l269_269825

theorem distinct_integers_integer_expression 
  (x y z : ℤ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (n : ℕ) : 
  ∃ k : ℤ, k = (x^n / ((x - y) * (x - z)) + y^n / ((y - x) * (y - z)) + z^n / ((z - x) * (z - y))) := 
sorry

end distinct_integers_integer_expression_l269_269825


namespace fewer_twos_for_100_l269_269330

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l269_269330


namespace negation_of_exists_l269_269616

theorem negation_of_exists (p : Prop) :
  (∃ x : ℝ, x^2 + 2 * x < 0) ↔ ¬ (∀ x : ℝ, x^2 + 2 * x >= 0) :=
sorry

end negation_of_exists_l269_269616


namespace angle_measure_F_l269_269064

theorem angle_measure_F (D E F : ℝ) 
  (h1 : D = 75) 
  (h2 : E = 4 * F - 15) 
  (h3 : D + E + F = 180) : 
  F = 24 := 
sorry

end angle_measure_F_l269_269064


namespace find_f_1988_l269_269292

namespace FunctionalEquation

def f (n : ℕ) : ℕ :=
  sorry -- definition placeholder, since we only need the statement

axiom f_properties (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (f m + f n) = m + n

theorem find_f_1988 (h : ∀ n : ℕ, 0 < n → f n = n) : f 1988 = 1988 :=
  sorry

end FunctionalEquation

end find_f_1988_l269_269292


namespace find_triplet_l269_269203

theorem find_triplet (x y z : ℕ) :
  (1 + (x : ℚ) / (y + z))^2 + (1 + (y : ℚ) / (z + x))^2 + (1 + (z : ℚ) / (x + y))^2 = 27 / 4 ↔ (x, y, z) = (1, 1, 1) :=
by
  sorry

end find_triplet_l269_269203


namespace evaluate_expression_l269_269889

theorem evaluate_expression :
  (5^5 * 5^3) / 3^6 * 2^5 = 12480000 / 729 :=
by sorry

end evaluate_expression_l269_269889


namespace SarahsScoreIs135_l269_269275

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end SarahsScoreIs135_l269_269275


namespace sum_of_primes_l269_269236

theorem sum_of_primes (a b c : ℕ) (h₁ : Nat.Prime a) (h₂ : Nat.Prime b) (h₃ : Nat.Prime c) (h₄ : b + c = 13) (h₅ : c^2 - a^2 = 72) :
  a + b + c = 20 := 
sorry

end sum_of_primes_l269_269236


namespace product_of_two_numbers_l269_269502

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end product_of_two_numbers_l269_269502


namespace gcd_78_182_l269_269126

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := 
by
  sorry

end gcd_78_182_l269_269126


namespace kelly_initial_games_l269_269072

theorem kelly_initial_games (games_given_away : ℕ) (games_left : ℕ)
  (h1 : games_given_away = 91) (h2 : games_left = 92) : 
  games_given_away + games_left = 183 :=
by {
  sorry
}

end kelly_initial_games_l269_269072


namespace fraction_equals_decimal_l269_269849

theorem fraction_equals_decimal : (3 : ℝ) / 2 = 1.5 := 
sorry

end fraction_equals_decimal_l269_269849


namespace fraction_computation_l269_269587

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l269_269587


namespace smallest_AAAB_value_l269_269570

theorem smallest_AAAB_value : ∃ (A B : ℕ), A ≠ B ∧ A < 10 ∧ B < 10 ∧ 111 * A + B = 7 * (10 * A + B) ∧ 111 * A + B = 667 :=
by sorry

end smallest_AAAB_value_l269_269570


namespace sum_coords_B_l269_269678

def point (A B : ℝ × ℝ) : Prop :=
A = (0, 0) ∧ B.snd = 5 ∧ (B.snd - A.snd) / (B.fst - A.fst) = 3 / 4

theorem sum_coords_B (x : ℝ) :
  point (0, 0) (x, 5) → x + 5 = 35 / 3 :=
by
  intro h
  sorry

end sum_coords_B_l269_269678


namespace hyperbola_eccentricity_l269_269625

noncomputable def hyperbola_eccentricity_range (a b e : ℝ) (h_a_pos : 0 < a) (h_a_less_1 : a < 1) (h_b_pos : 0 < b) : Prop :=
  let c := Real.sqrt ((5 * a^2 - a^4) / (1 - a^2))
  let e := c / a
  e > Real.sqrt 5

theorem hyperbola_eccentricity (a b e : ℝ) (h_a_pos : 0 < a) (h_a_less_1 : a < 1) (h_b_pos : 0 < b) :
  hyperbola_eccentricity_range a b e h_a_pos h_a_less_1 h_b_pos := 
sorry

end hyperbola_eccentricity_l269_269625


namespace camel_cost_l269_269740

theorem camel_cost
  (C H O E : ℝ)
  (h1 : 10 * C = 24 * H)
  (h2 : 26 * H = 4 * O)
  (h3 : 6 * O = 4 * E)
  (h4 : 10 * E = 170000) :
  C = 4184.62 :=
by sorry

end camel_cost_l269_269740


namespace circle_points_l269_269960

noncomputable def proof_problem (x1 y1 x2 y2: ℝ) : Prop :=
  (x1^2 + y1^2 = 4) ∧ (x2^2 + y2^2 = 4) ∧ ((x1 - x2)^2 + (y1 - y2)^2 = 12) →
    (x1 * x2 + y1 * y2 = -2)

theorem circle_points (x1 y1 x2 y2 : ℝ) : proof_problem x1 y1 x2 y2 := 
by
  sorry

end circle_points_l269_269960


namespace det_projection_matrix_l269_269258

noncomputable section

variable {𝕜 : Type*} [Field 𝕜]

def projection_matrix (v : Matrix (Fin 3) (Fin 1) 𝕜) :=
  (v ⬝ vᵀ) ⬝ (((vᵀ ⬝ v).det⁻¹) • (1 : Matrix (Fin 3) (Fin 3) 𝕜))

theorem det_projection_matrix (v : Matrix (Fin 3) (Fin 1) 𝕜) : 
  v = ![![3], ![1], ![-4]] →
  (projection_matrix v).det = 0 :=
by
  sorry

end det_projection_matrix_l269_269258


namespace remainder_of_4123_div_by_32_l269_269349

theorem remainder_of_4123_div_by_32 : 
  ∃ r, 0 ≤ r ∧ r < 32 ∧ 4123 = 32 * (4123 / 32) + r ∧ r = 27 := by
  sorry

end remainder_of_4123_div_by_32_l269_269349


namespace pythagorean_theorem_l269_269963

-- Definitions from the conditions
variables {a b c : ℝ}
-- Assuming a right triangle with legs a, b and hypotenuse c
def is_right_triangle (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

-- Statement of the theorem:
theorem pythagorean_theorem (a b c : ℝ) (h : is_right_triangle a b c) : c^2 = a^2 + b^2 :=
sorry

end pythagorean_theorem_l269_269963


namespace circle_diameter_l269_269533

theorem circle_diameter (A : ℝ) (h : A = 4 * π) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l269_269533


namespace lorelai_jellybeans_correct_l269_269088

-- Define the number of jellybeans Gigi has
def gigi_jellybeans : Nat := 15

-- Define the number of additional jellybeans Rory has compared to Gigi
def rory_additional_jellybeans : Nat := 30

-- Define the number of jellybeans both girls together have
def total_jellybeans : Nat := gigi_jellybeans + (gigi_jellybeans + rory_additional_jellybeans)

-- Define the number of jellybeans Lorelai has eaten
def lorelai_jellybeans : Nat := 3 * total_jellybeans

-- The theorem to prove the number of jellybeans Lorelai has eaten is 180
theorem lorelai_jellybeans_correct : lorelai_jellybeans = 180 := by
  sorry

end lorelai_jellybeans_correct_l269_269088


namespace sarah_score_l269_269280

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end sarah_score_l269_269280


namespace sqrt_x_minus_1_domain_l269_269634

theorem sqrt_x_minus_1_domain (x : ℝ) : (∃ y, y^2 = x - 1) → (x ≥ 1) := 
by 
  sorry

end sqrt_x_minus_1_domain_l269_269634


namespace remainder_when_divided_by_5_l269_269608

theorem remainder_when_divided_by_5 : (1234 * 1987 * 2013 * 2021) % 5 = 4 :=
by
  sorry

end remainder_when_divided_by_5_l269_269608


namespace range_of_a_l269_269996

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → (x^2 - 2 * a * x + 2) ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l269_269996


namespace john_tanks_needed_l269_269466

theorem john_tanks_needed 
  (num_balloons : ℕ) 
  (volume_per_balloon : ℕ) 
  (volume_per_tank : ℕ) 
  (H1 : num_balloons = 1000) 
  (H2 : volume_per_balloon = 10) 
  (H3 : volume_per_tank = 500) 
: (num_balloons * volume_per_balloon) / volume_per_tank = 20 := 
by 
  sorry

end john_tanks_needed_l269_269466


namespace Sarahs_score_l269_269284

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end Sarahs_score_l269_269284


namespace fraction_computation_l269_269593

theorem fraction_computation : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by 
  sorry

end fraction_computation_l269_269593


namespace sunny_weather_prob_correct_l269_269571

def rain_prob : ℝ := 0.45
def cloudy_prob : ℝ := 0.20
def sunny_prob : ℝ := 1 - rain_prob - cloudy_prob

theorem sunny_weather_prob_correct : sunny_prob = 0.35 := by
  sorry

end sunny_weather_prob_correct_l269_269571


namespace circle_diameter_l269_269547

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l269_269547


namespace total_spending_correct_l269_269368

-- Define the costs and number of children for each ride and snack
def cost_ferris_wheel := 5 * 5
def cost_roller_coaster := 7 * 3
def cost_merry_go_round := 3 * 8
def cost_bumper_cars := 4 * 6

def cost_ice_cream := 8 * 2 * 5
def cost_hot_dog := 6 * 4
def cost_pizza := 4 * 3

-- Calculate the total cost
def total_cost_rides := cost_ferris_wheel + cost_roller_coaster + cost_merry_go_round + cost_bumper_cars
def total_cost_snacks := cost_ice_cream + cost_hot_dog + cost_pizza
def total_spent := total_cost_rides + total_cost_snacks

-- The statement to prove
theorem total_spending_correct : total_spent = 170 := by
  sorry

end total_spending_correct_l269_269368


namespace smallest_positive_integer_remainder_l269_269723

theorem smallest_positive_integer_remainder : ∃ a : ℕ, 
  (a ≡ 2 [MOD 3]) ∧ (a ≡ 3 [MOD 5]) ∧ (a = 8) := 
by
  use 8
  split
  · exact Nat.modeq.symm (Nat.modeq.modeq_of_dvd _ (by norm_num))
  · split
    · exact Nat.modeq.symm (Nat.modeq.modeq_of_dvd _ (by norm_num))
    · rfl
  sorry  -- The detailed steps of the proof are omitted as per the instructions

end smallest_positive_integer_remainder_l269_269723


namespace multiple_of_one_third_l269_269354

theorem multiple_of_one_third (x : ℚ) (h : x * (1 / 3) = 2 / 9) : x = 2 / 3 :=
sorry

end multiple_of_one_third_l269_269354


namespace iron_balls_molded_l269_269396

-- Define the dimensions of the iron bar
def length_bar : ℝ := 12
def width_bar : ℝ := 8
def height_bar : ℝ := 6

-- Define the volume calculations
def volume_iron_bar : ℝ := length_bar * width_bar * height_bar
def number_of_bars : ℝ := 10
def total_volume_bars : ℝ := volume_iron_bar * number_of_bars
def volume_iron_ball : ℝ := 8

-- Define the goal statement
theorem iron_balls_molded : total_volume_bars / volume_iron_ball = 720 :=
by
  -- Proof is to be filled in here
  sorry

end iron_balls_molded_l269_269396


namespace market_value_of_13_percent_stock_yielding_8_percent_l269_269999

noncomputable def market_value_of_stock (yield rate dividend_per_share : ℝ) : ℝ :=
  (dividend_per_share / yield) * 100

theorem market_value_of_13_percent_stock_yielding_8_percent
  (yield_rate : ℝ) (dividend_per_share : ℝ) (market_value : ℝ)
  (h_yield_rate : yield_rate = 0.08)
  (h_dividend_per_share : dividend_per_share = 13) :
  market_value = 162.50 :=
by
  sorry

end market_value_of_13_percent_stock_yielding_8_percent_l269_269999


namespace opposite_of_neg_nine_is_nine_l269_269704

-- Define the predicate for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that needs to be proved
theorem opposite_of_neg_nine_is_nine : ∃ x, is_opposite (-9) x ∧ x = 9 := 
by {
  -- Use "sorry" to indicate the proof is not provided
  sorry
}

end opposite_of_neg_nine_is_nine_l269_269704


namespace ecuadorian_number_unique_l269_269166

def is_Ecuadorian (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n < 1000 ∧ c ≠ 0 ∧ n % 36 = 0 ∧ (n - (100 * c + 10 * b + a) > 0) ∧ (n - (100 * c + 10 * b + a)) % 36 = 0

theorem ecuadorian_number_unique (n : ℕ) : 
  is_Ecuadorian n → n = 864 :=
sorry

end ecuadorian_number_unique_l269_269166


namespace sales_in_third_month_is_6855_l269_269868

noncomputable def sales_in_third_month : ℕ :=
  let sale_1 := 6435
  let sale_2 := 6927
  let sale_4 := 7230
  let sale_5 := 6562
  let sale_6 := 6791
  let total_sales := 6800 * 6
  total_sales - (sale_1 + sale_2 + sale_4 + sale_5 + sale_6)

theorem sales_in_third_month_is_6855 : sales_in_third_month = 6855 := by
  sorry

end sales_in_third_month_is_6855_l269_269868


namespace johnny_future_years_l269_269939

theorem johnny_future_years (x : ℕ) (h1 : 8 + x = 2 * (8 - 3)) : x = 2 :=
by
  sorry

end johnny_future_years_l269_269939


namespace sum_of_coordinates_l269_269671

theorem sum_of_coordinates (x y : ℚ) (h₁ : y = 5) (h₂ : (y - 0) / (x - 0) = 3/4) : x + y = 35/3 :=
by sorry

end sum_of_coordinates_l269_269671


namespace student_walking_time_l269_269163

-- Define the conditions
def total_time_walking_and_bus : ℕ := 90  -- Total time walking to school and taking the bus back home
def total_time_bus_both_ways : ℕ := 30 -- Total time taking the bus both ways

-- Calculate the time taken for walking both ways
def time_bus_one_way : ℕ := total_time_bus_both_ways / 2
def time_walking_one_way : ℕ := total_time_walking_and_bus - time_bus_one_way
def total_time_walking_both_ways : ℕ := 2 * time_walking_one_way

-- State the theorem to be proved
theorem student_walking_time :
  total_time_walking_both_ways = 150 := by
  sorry

end student_walking_time_l269_269163


namespace find_certain_number_l269_269361

def certain_number (x : ℚ) : Prop := 5 * 1.6 - (1.4 * x) / 1.3 = 4

theorem find_certain_number : certain_number (-(26/7)) :=
by 
  simp [certain_number]
  sorry

end find_certain_number_l269_269361


namespace long_jump_record_l269_269063

theorem long_jump_record 
  (standard_distance : ℝ)
  (jump1 : ℝ)
  (jump2 : ℝ)
  (record1 : ℝ)
  (record2 : ℝ)
  (h1 : standard_distance = 4.00)
  (h2 : jump1 = 4.22)
  (h3 : jump2 = 3.85)
  (h4 : record1 = jump1 - standard_distance)
  (h5 : record2 = jump2 - standard_distance)
  : record2 = -0.15 := 
sorry

end long_jump_record_l269_269063


namespace personal_income_tax_correct_l269_269882

-- Defining the conditions
def monthly_income : ℕ := 30000
def vacation_bonus : ℕ := 20000
def car_sale_income : ℕ := 250000
def land_purchase_cost : ℕ := 300000

def standard_deduction_car_sale : ℕ := 250000
def property_deduction_land_purchase : ℕ := 300000

-- Define total income
def total_income : ℕ := (monthly_income * 12) + vacation_bonus + car_sale_income

-- Define total deductions
def total_deductions : ℕ := standard_deduction_car_sale + property_deduction_land_purchase

-- Define taxable income (total income - total deductions)
def taxable_income : ℕ := total_income - total_deductions

-- Define tax rate
def tax_rate : ℚ := 0.13

-- Define the correct answer for the tax payable
def tax_payable : ℚ := taxable_income * tax_rate

-- Prove the tax payable is 10400 rubles
theorem personal_income_tax_correct : tax_payable = 10400 := by
  sorry

end personal_income_tax_correct_l269_269882


namespace rope_length_equals_120_l269_269507

theorem rope_length_equals_120 (x : ℝ) (l : ℝ)
  (h1 : x + 20 = 3 * x) 
  (h2 : l = 4 * (2 * x)) : 
  l = 120 :=
by
  -- Proof will be provided here
  sorry

end rope_length_equals_120_l269_269507


namespace valid_B_sets_l269_269927

def A : Set ℝ := {x | 0 < x ∧ x < 2}

theorem valid_B_sets (B : Set ℝ) : A ∩ B = B ↔ B = ∅ ∨ B = {1} ∨ B = A :=
by
  sorry

end valid_B_sets_l269_269927


namespace yellow_tint_percent_l269_269370

theorem yellow_tint_percent (total_volume: ℕ) (initial_yellow_percent: ℚ) (yellow_added: ℕ) (answer: ℚ) 
  (h_initial_total: total_volume = 20) 
  (h_initial_yellow: initial_yellow_percent = 0.50) 
  (h_yellow_added: yellow_added = 6) 
  (h_answer: answer = 61.5): 
  (yellow_added + initial_yellow_percent * total_volume) / (total_volume + yellow_added) * 100 = answer := 
by 
  sorry

end yellow_tint_percent_l269_269370


namespace main_theorem_l269_269951

noncomputable def exists_infinitely_many_n (k l m : ℕ) (h_k_pos : 0 < k) (h_l_pos : 0 < l) (h_m_pos : 0 < m) : Prop :=
  ∃ᶠ n in at_top, (Nat.coprime (Nat.choose n k) m ∧ m ∣ Nat.choose n k)

theorem main_theorem (k l m : ℕ) (h_k_pos : 0 < k) (h_l_pos : 0 < l) (h_m_pos : 0 < m) : exists_infinitely_many_n k l m h_k_pos h_l_pos h_m_pos := sorry

end main_theorem_l269_269951


namespace jill_travels_less_than_john_l269_269255

theorem jill_travels_less_than_john :
  ∀ (John Jill Jim : ℕ), 
  John = 15 → 
  Jim = 2 → 
  (Jim = (20 / 100) * Jill) → 
  (John - Jill) = 5 := 
by
  intros John Jill Jim HJohn HJim HJimJill
  -- Skip the proof for now
  sorry

end jill_travels_less_than_john_l269_269255


namespace isosceles_triangle_l269_269066

def triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) : Prop :=
  a^2 + b^2 - (a * Real.cos B + b * Real.cos A)^2 = 2 * a * b * Real.cos B → (B = C)

theorem isosceles_triangle (a b c A B C : ℝ) (h : a^2 + b^2 - (a * Real.cos B + b * Real.cos A)^2 = 2 * a * b * Real.cos B) : B = C :=
  sorry

end isosceles_triangle_l269_269066


namespace k_configurations_count_l269_269437

open Nat

theorem k_configurations_count {n k : ℕ} (h : k ≤ n) :
  (Finset.card ∘ Finset.filter (λ s, s.card = k) ∘ Finset.powerset : Finset (Finset α) → Nat) (Finset.range n) = Nat.choose n k :=
by
  sorry

end k_configurations_count_l269_269437


namespace find_m_l269_269917

theorem find_m (m : ℝ) (a b : ℝ × ℝ)
  (ha : a = (3, m)) (hb : b = (1, -2))
  (h : a.1 * b.1 + a.2 * b.2 = b.1^2 + b.2^2) :
  m = -1 :=
by {
  sorry
}

end find_m_l269_269917


namespace max_f1_l269_269432

-- Define the function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * b * x + a + 2 * b 

-- Define the condition 
def condition (a : ℝ) (b : ℝ) : Prop := f 0 a b = 4

-- State the theorem
theorem max_f1 (a b: ℝ) (h: condition a b) : 
  ∃ b_max, b_max = 1 ∧ ∀ b, f 1 a b ≤ 7 := 
sorry

end max_f1_l269_269432


namespace tshirt_costs_more_than_jersey_l269_269485

open Nat

def cost_tshirt : ℕ := 192
def cost_jersey : ℕ := 34

theorem tshirt_costs_more_than_jersey :
  cost_tshirt - cost_jersey = 158 :=
by sorry

end tshirt_costs_more_than_jersey_l269_269485


namespace area_square_II_is_6a_squared_l269_269696

-- Problem statement:
-- Given the diagonal of square I is 2a and the area of square II is three times the area of square I,
-- prove that the area of square II is 6a^2

noncomputable def area_square_II (a : ℝ) : ℝ :=
  let side_I := (2 * a) / Real.sqrt 2
  let area_I := side_I ^ 2
  3 * area_I

theorem area_square_II_is_6a_squared (a : ℝ) : area_square_II a = 6 * a ^ 2 :=
by
  sorry

end area_square_II_is_6a_squared_l269_269696


namespace opposite_neg_9_l269_269710

theorem opposite_neg_9 : 
  ∃ y : Int, -9 + y = 0 ∧ y = 9 :=
by
  sorry

end opposite_neg_9_l269_269710


namespace positive_integers_n_less_than_200_with_m_divisible_by_4_l269_269791

theorem positive_integers_n_less_than_200_with_m_divisible_by_4 :
  {n : ℕ // n < 200 ∧ ∃ m : ℕ, (∃ k : ℕ, n = 4 * k + 2) ∧ m = 4 * (k^2 + k)}.card = 50 :=
sorry

end positive_integers_n_less_than_200_with_m_divisible_by_4_l269_269791


namespace circle_diameter_l269_269544

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) :
    ∃ d : ℝ, d = 4 :=
by
  -- Create conditions from the given problem
  let r := √(A / π)
  have hr : r = 2, from sorry,
  -- Solve for the diameter
  let d := 2 * r
  have hd : d = 4, from sorry,
  -- Conclude the theorem
  use d
  exact hd

end circle_diameter_l269_269544


namespace solve_equation_1_solve_quadratic_equation_2_l269_269969

theorem solve_equation_1 (x : ℝ) : 2 * (x - 1)^2 = 1 - x ↔ x = 1 ∨ x = 1/2 := sorry

theorem solve_quadratic_equation_2 (x : ℝ) :
  4 * x^2 - 2 * (Real.sqrt 3) * x - 1 = 0 ↔
    x = (Real.sqrt 3 + Real.sqrt 7) / 4 ∨ x = (Real.sqrt 3 - Real.sqrt 7) / 4 := sorry

end solve_equation_1_solve_quadratic_equation_2_l269_269969


namespace circle_diameter_problem_circle_diameter_l269_269540

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l269_269540


namespace tom_travel_time_to_virgo_island_l269_269510

-- Definitions based on conditions
def boat_trip_time : ℕ := 2
def plane_trip_time : ℕ := 4 * boat_trip_time
def total_trip_time : ℕ := plane_trip_time + boat_trip_time

-- Theorem we need to prove
theorem tom_travel_time_to_virgo_island : total_trip_time = 10 := by
  sorry

end tom_travel_time_to_virgo_island_l269_269510


namespace packages_per_hour_A_B_max_A_robots_l269_269100

-- Define the number of packages sorted by each unit of type A and B robots
def packages_by_A_robot (x : ℕ) := x
def packages_by_B_robot (y : ℕ) := y

-- Problem conditions
def cond1 (x y : ℕ) : Prop := 80 * x + 100 * y = 8200
def cond2 (x y : ℕ) : Prop := 50 * x + 50 * y = 4500

-- Part 1: to prove type A and type B robot's packages per hour
theorem packages_per_hour_A_B (x y : ℕ) (h1 : cond1 x y) (h2 : cond2 x y) : x = 40 ∧ y = 50 :=
by sorry

-- Part 2: prove maximum units of type A robots when purchasing 200 robots ensuring not < 9000 packages/hour
def cond3 (m : ℕ) : Prop := 40 * m + 50 * (200 - m) ≥ 9000

theorem max_A_robots (m : ℕ) (h3 : cond3 m) : m ≤ 100 :=
by sorry

end packages_per_hour_A_B_max_A_robots_l269_269100


namespace regular_tetrahedron_ratio_l269_269642

/-- In plane geometry, the ratio of the radius of the circumscribed circle to the 
inscribed circle of an equilateral triangle is 2:1, --/
def ratio_radii_equilateral_triangle : ℚ := 2 / 1

/-- In space geometry, we study the relationship between the radii of the circumscribed
sphere and the inscribed sphere of a regular tetrahedron. --/
def ratio_radii_regular_tetrahedron : ℚ := 3 / 1

/-- Prove the ratio of the radius of the circumscribed sphere to the inscribed sphere
of a regular tetrahedron is 3 : 1, given the ratio is 2 : 1 for the equilateral triangle. --/
theorem regular_tetrahedron_ratio : 
  ratio_radii_equilateral_triangle = 2 / 1 → 
  ratio_radii_regular_tetrahedron = 3 / 1 :=
by
  sorry

end regular_tetrahedron_ratio_l269_269642


namespace dice_probability_l269_269644

theorem dice_probability (p : ℚ) (h : p = (1 / 42)) : 
  p = 0.023809523809523808 := 
sorry

end dice_probability_l269_269644


namespace sue_final_answer_is_67_l269_269755

-- Declare the initial value Ben thinks of
def ben_initial_number : ℕ := 4

-- Ben's calculation function
def ben_number (b : ℕ) : ℕ := ((b + 2) * 3) + 5

-- Sue's calculation function
def sue_number (x : ℕ) : ℕ := ((x - 3) * 3) + 7

-- Define the final number Sue calculates
def final_sue_number : ℕ := sue_number (ben_number ben_initial_number)

-- Prove that Sue's final number is 67
theorem sue_final_answer_is_67 : final_sue_number = 67 :=
by 
  sorry

end sue_final_answer_is_67_l269_269755


namespace largest_multiple_of_9_less_than_120_l269_269986

theorem largest_multiple_of_9_less_than_120 : ∃ n, n < 120 ∧ n % 9 = 0 ∧ ∀ m, m < 120 ∧ m % 9 = 0 → m ≤ n :=
  by {
    use 117,
    split,
    { exact 117 < 120, },
    split,
    { exact 117 % 9 = 0, },
    { intros m hm1 hm2,
      show m ≤ 117,
      sorry
    }
  }

end largest_multiple_of_9_less_than_120_l269_269986


namespace opera_house_rows_l269_269753

variable (R : ℕ)
variable (SeatsPerRow : ℕ)
variable (TicketPrice : ℕ)
variable (TotalEarnings : ℕ)
variable (SeatsTakenPercent : ℝ)

-- Conditions
axiom num_seats_per_row : SeatsPerRow = 10
axiom ticket_price : TicketPrice = 10
axiom total_earnings : TotalEarnings = 12000
axiom seats_taken_percent : SeatsTakenPercent = 0.8

-- Main theorem statement
theorem opera_house_rows
  (h1 : SeatsPerRow = 10)
  (h2 : TicketPrice = 10)
  (h3 : TotalEarnings = 12000)
  (h4 : SeatsTakenPercent = 0.8) :
  R = 150 :=
sorry

end opera_house_rows_l269_269753


namespace part_1_part_2_l269_269014

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269014


namespace least_positive_a_exists_l269_269651

noncomputable def f (x a : ℤ) : ℤ := 5 * x ^ 13 + 13 * x ^ 5 + 9 * a * x

theorem least_positive_a_exists :
  ∃ a : ℕ, (∀ x : ℤ, 65 ∣ f x a) ∧ ∀ b : ℕ, (∀ x : ℤ, 65 ∣ f x b) → a ≤ b :=
sorry

end least_positive_a_exists_l269_269651


namespace kishore_miscellaneous_expenses_l269_269751

theorem kishore_miscellaneous_expenses :
  ∀ (rent milk groceries education petrol savings total_salary total_specified_expenses : ℝ),
  rent = 5000 →
  milk = 1500 →
  groceries = 4500 →
  education = 2500 →
  petrol = 2000 →
  savings = 2300 →
  (savings / 0.10) = total_salary →
  (rent + milk + groceries + education + petrol) = total_specified_expenses →
  (total_salary - (total_specified_expenses + savings)) = 5200 :=
by
  intros rent milk groceries education petrol savings total_salary total_specified_expenses
  sorry

end kishore_miscellaneous_expenses_l269_269751


namespace frog_stops_at_corner_l269_269082

noncomputable def frog_probability_at_corner : ℚ :=
sorry

theorem frog_stops_at_corner :
  frog_probability_at_corner = 3 / 8 :=
sorry

end frog_stops_at_corner_l269_269082


namespace noah_ate_burgers_l269_269662

theorem noah_ate_burgers :
  ∀ (weight_hotdog weight_burger weight_pie : ℕ) 
    (mason_hotdog_weight : ℕ) 
    (jacob_pies noah_burgers mason_hotdogs : ℕ),
    weight_hotdog = 2 →
    weight_burger = 5 →
    weight_pie = 10 →
    (jacob_pies + 3 = noah_burgers) →
    (mason_hotdogs = 3 * jacob_pies) →
    (mason_hotdog_weight = 30) →
    (mason_hotdog_weight / weight_hotdog = mason_hotdogs) →
    noah_burgers = 8 :=
by
  intros weight_hotdog weight_burger weight_pie mason_hotdog_weight
         jacob_pies noah_burgers mason_hotdogs
         h1 h2 h3 h4 h5 h6 h7
  sorry

end noah_ate_burgers_l269_269662


namespace solve_for_x_l269_269428

theorem solve_for_x 
  (a b : ℝ)
  (h1 : ∃ P : ℝ × ℝ, P = (3, -1) ∧ (P.2 = 3 + b) ∧ (P.2 = a * 3 + 2)) :
  (a - 1) * 3 = b - 2 :=
by sorry

end solve_for_x_l269_269428


namespace theater_loss_l269_269559

/-- 
A movie theater has a total capacity of 50 people and charges $8 per ticket.
On a Tuesday night, they only sold 24 tickets. 
Prove that the revenue lost by not selling out is $208.
-/
theorem theater_loss 
  (capacity : ℕ) 
  (price : ℕ) 
  (sold_tickets : ℕ) 
  (h_cap : capacity = 50) 
  (h_price : price = 8) 
  (h_sold : sold_tickets = 24) : 
  capacity * price - sold_tickets * price = 208 :=
by
  sorry

end theater_loss_l269_269559


namespace part1_part2_l269_269037

theorem part1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem part2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt abc) :=
sorry

end part1_part2_l269_269037


namespace mars_bars_count_l269_269265

theorem mars_bars_count (total_candy_bars snickers butterfingers : ℕ) (h_total : total_candy_bars = 12) (h_snickers : snickers = 3) (h_butterfingers : butterfingers = 7) :
  total_candy_bars - (snickers + butterfingers) = 2 :=
by sorry

end mars_bars_count_l269_269265


namespace cube_paint_same_color_l269_269267

theorem cube_paint_same_color (colors : Fin 6) : ∃ ways : ℕ, ways = 6 :=
sorry

end cube_paint_same_color_l269_269267


namespace Ben_total_clothes_l269_269169

-- Definitions of Alex's clothing items
def Alex_shirts := 4.5
def Alex_pants := 3.0
def Alex_shoes := 2.5
def Alex_hats := 1.5
def Alex_jackets := 2.0

-- Definitions of Joe's clothing items
def Joe_shirts := Alex_shirts + 3.5
def Joe_pants := Alex_pants - 2.5
def Joe_shoes := Alex_shoes
def Joe_hats := Alex_hats + 0.3
def Joe_jackets := Alex_jackets - 1.0

-- Definitions of Ben's clothing items
def Ben_shirts := Joe_shirts + 5.3
def Ben_pants := Alex_pants + 5.5
def Ben_shoes := Joe_shoes - 1.7
def Ben_hats := Alex_hats + 0.5
def Ben_jackets := Joe_jackets + 1.5

-- Statement to prove the total number of Ben's clothing items
def total_Ben_clothing_items := Ben_shirts + Ben_pants + Ben_shoes + Ben_hats + Ben_jackets

theorem Ben_total_clothes : total_Ben_clothing_items = 27.1 :=
by
  sorry

end Ben_total_clothes_l269_269169


namespace total_area_of_tickets_is_3_6_m2_l269_269997

def area_of_one_ticket (side_length_cm : ℕ) : ℕ :=
  side_length_cm * side_length_cm

def total_tickets (people : ℕ) (tickets_per_person : ℕ) : ℕ :=
  people * tickets_per_person

def total_area_cm2 (area_per_ticket_cm2 : ℕ) (number_of_tickets : ℕ) : ℕ :=
  area_per_ticket_cm2 * number_of_tickets

def convert_cm2_to_m2 (area_cm2 : ℕ) : ℚ :=
  (area_cm2 : ℚ) / 10000

theorem total_area_of_tickets_is_3_6_m2 :
  let side_length := 30
  let people := 5
  let tickets_per_person := 8
  let one_ticket_area := area_of_one_ticket side_length
  let number_of_tickets := total_tickets people tickets_per_person
  let total_area_cm2 := total_area_cm2 one_ticket_area number_of_tickets
  let total_area_m2 := convert_cm2_to_m2 total_area_cm2
  total_area_m2 = 3.6 := 
by
  sorry

end total_area_of_tickets_is_3_6_m2_l269_269997


namespace dice_probability_sum_17_l269_269899

-- Definitions to be used directly from conditions:
def is_dice_face (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 6

def valid_dice_rolls (x₁ x₂ x₃ x₄ : ℕ) : Prop :=
  is_dice_face x₁ ∧ is_dice_face x₂ ∧ is_dice_face x₃ ∧ is_dice_face x₄

-- Final proof statement with the given correct answer without steps:
theorem dice_probability_sum_17 :
  (∃ s : Finset (ℕ × ℕ × ℕ × ℕ),
    (∀ (x₁ x₂ x₃ x₄ ∈ s), valid_dice_rolls x₁ x₂ x₃ x₄ ∧ x₁ + x₂ + x₃ + x₄ = 17) ∧
    s.card = 56) →
  56 / 1296 = 7 / 162 :=
by sorry

end dice_probability_sum_17_l269_269899


namespace probability_of_multiples_of_4_l269_269881

def number_of_multiples_of_4 (n : ℕ) : ℕ :=
  n / 4

def number_not_multiples_of_4 (n : ℕ) (m : ℕ) : ℕ :=
  n - m

def probability_neither_multiples_of_4 (n : ℕ) (m : ℕ) : ℚ :=
  (m / n : ℚ) * (m / n)

def probability_at_least_one_multiple_of_4 (n : ℕ) (m : ℕ) : ℚ :=
  1 - probability_neither_multiples_of_4 n m

theorem probability_of_multiples_of_4 :
  probability_at_least_one_multiple_of_4 60 45 = 7 / 16 :=
by
  sorry

end probability_of_multiples_of_4_l269_269881


namespace sum_coords_B_l269_269676

def point (A B : ℝ × ℝ) : Prop :=
A = (0, 0) ∧ B.snd = 5 ∧ (B.snd - A.snd) / (B.fst - A.fst) = 3 / 4

theorem sum_coords_B (x : ℝ) :
  point (0, 0) (x, 5) → x + 5 = 35 / 3 :=
by
  intro h
  sorry

end sum_coords_B_l269_269676


namespace main_theorem_l269_269042

-- Define the distribution
def P0 : ℝ := 0.4
def P2 : ℝ := 0.4
def P1 (p : ℝ) : ℝ := p

-- Define a hypothesis that the sum of probabilities is 1
def prob_sum_eq_one (p : ℝ) : Prop := P0 + P1 p + P2 = 1

-- Define the expected value of X
def E_X (p : ℝ) : ℝ := 0 * P0 + 1 * P1 p + 2 * P2

-- Define variance computation
def variance (p : ℝ) : ℝ := P0 * (0 - E_X p) ^ 2 + P1 p * (1 - E_X p) ^ 2 + P2 * (2 - E_X p) ^ 2

-- State the main theorem
theorem main_theorem : (∃ p : ℝ, prob_sum_eq_one p) ∧ variance 0.2 = 0.8 :=
by
  sorry

end main_theorem_l269_269042


namespace complement_A_complement_B_intersection_A_B_complement_union_A_B_l269_269916

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def set_U : Set ℝ := {x | true}  -- This represents U = ℝ
def set_A : Set ℝ := {x | x < -2 ∨ x > 5}
def set_B : Set ℝ := {x | 4 ≤ x ∧ x ≤ 6}

theorem complement_A :
  ∀ x : ℝ, x ∈ set_U \ set_A ↔ -2 ≤ x ∧ x ≤ 5 :=
by
  intro x
  sorry

theorem complement_B :
  ∀ x : ℝ, x ∉ set_B ↔ x < 4 ∨ x > 6 :=
by
  intro x
  sorry

theorem intersection_A_B :
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ 5 < x ∧ x ≤ 6 :=
by
  intro x
  sorry

theorem complement_union_A_B :
  ∀ x : ℝ, x ∈ set_U \ (set_A ∪ set_B) ↔ -2 ≤ x ∧ x < 4 :=
by
  intro x
  sorry

end complement_A_complement_B_intersection_A_B_complement_union_A_B_l269_269916


namespace probability_score_1_2_after_3_serves_probability_fifth_match_in_round_robin_l269_269856

noncomputable section

-- Definitions for Question 1
def A_serves_first := 0.6
def serve_independent := true
def probability_1_2_in_favor_of_B := 0.352

-- Theorem for Question 1
theorem probability_score_1_2_after_3_serves 
  (p_A_serves : ℝ) 
  (independence : bool) 
  : p_A_serves = A_serves_first ∧ independence = serve_independent → probability_1_2_in_favor_of_B = 0.352 :=
sorry

-- Definitions for Question 2
def probability_win_match := 1/2
def probability_fifth_match_needed := 3/4

-- Theorem for Question 2
theorem probability_fifth_match_in_round_robin
  (p_win_match : ℝ) 
  : p_win_match = probability_win_match → probability_fifth_match_needed = 3/4 :=
sorry

end probability_score_1_2_after_3_serves_probability_fifth_match_in_round_robin_l269_269856


namespace nails_per_station_correct_l269_269760

variable (total_nails : ℕ) (total_stations : ℕ) (nails_per_station : ℕ)

theorem nails_per_station_correct :
  total_nails = 140 → total_stations = 20 → nails_per_station = total_nails / total_stations → nails_per_station = 7 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end nails_per_station_correct_l269_269760


namespace circle_diameter_l269_269530

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l269_269530


namespace most_frequent_digit_100000_l269_269124

/- Define the digital root function -/
def digital_root (n : ℕ) : ℕ :=
  if n == 0 then 0 else if n % 9 == 0 then 9 else n % 9

/- Define the problem statement -/
theorem most_frequent_digit_100000 : 
  ∃ digit : ℕ, 
  digit = 1 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → ∃ k : ℕ, k = digital_root n ∧ k = digit) →
  digit = 1 :=
sorry

end most_frequent_digit_100000_l269_269124


namespace usual_time_28_l269_269520

theorem usual_time_28 (R T : ℝ) (h1 : ∀ (d : ℝ), d = R * T)
  (h2 : ∀ (d : ℝ), d = (6/7) * R * (T - 4)) : T = 28 :=
by
  -- Variables:
  -- R : Usual rate of the boy
  -- T : Usual time to reach the school
  -- h1 : Expressing distance in terms of usual rate and time
  -- h2 : Expressing distance in terms of reduced rate and time minus 4
  sorry

end usual_time_28_l269_269520


namespace Arrow_velocity_at_impact_l269_269405

def Edward_initial_distance := 1875 -- \(\text{ft}\)
def Edward_initial_velocity := 0 -- \(\text{ft/s}\)
def Edward_acceleration := 1 -- \(\text{ft/s}^2\)
def Arrow_initial_distance := 0 -- \(\text{ft}\)
def Arrow_initial_velocity := 100 -- \(\text{ft/s}\)
def Arrow_deceleration := -1 -- \(\text{ft/s}^2\)
def time_impact := 25 -- \(\text{s}\)

theorem Arrow_velocity_at_impact : 
  (Arrow_initial_velocity + Arrow_deceleration * time_impact) = 75 := 
by
  sorry

end Arrow_velocity_at_impact_l269_269405


namespace circle_diameter_l269_269526

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l269_269526


namespace solution1_solution2_l269_269341

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l269_269341


namespace part1_part2_l269_269022

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269022


namespace total_growing_space_is_correct_l269_269391

def garden_bed_area (length : ℕ) (width : ℕ) (count : ℕ) : ℕ :=
  length * width * count

def total_growing_space : ℕ :=
  garden_bed_area 5 4 3 +
  garden_bed_area 6 3 4 +
  garden_bed_area 7 5 2 +
  garden_bed_area 8 4 1

theorem total_growing_space_is_correct :
  total_growing_space = 234 := by
  sorry

end total_growing_space_is_correct_l269_269391


namespace complement_of_45_is_45_l269_269921

def angle_complement (A : Real) : Real :=
  90 - A

theorem complement_of_45_is_45:
  angle_complement 45 = 45 :=
by
  sorry

end complement_of_45_is_45_l269_269921


namespace OH_squared_is_given_value_l269_269811

noncomputable def circumcenter_orthocenter_distance_squared (a b c R : ℝ) (hR : R = 10)
  (sides_squared_sum : a^2 + b^2 + c^2 = 50) : ℝ :=
  let OH_squared := 9*R^2 - (a^2 + b^2 + c^2)
  in OH_squared

-- Formalize the statement in Lean
theorem OH_squared_is_given_value (a b c R : ℝ) (hR : R = 10)
  (sides_squared_sum : a^2 + b^2 + c^2 = 50) :
  circumcenter_orthocenter_distance_squared a b c R hR sides_squared_sum = 850 :=
by
  sorry

end OH_squared_is_given_value_l269_269811


namespace ordered_sets_equal_l269_269256

theorem ordered_sets_equal
  (n : ℕ) 
  (h_gcd : gcd n 6 = 1) 
  (a b : ℕ → ℕ) 
  (h_order_a : ∀ {i j}, i < j → a i < a j)
  (h_order_b : ∀ {i j}, i < j → b i < b j) 
  (h_sum : ∀ {j k l : ℕ}, 1 ≤ j → j < k → k < l → l ≤ n → a j + a k + a l = b j + b k + b l) : 
  ∀ (j : ℕ), 1 ≤ j → j ≤ n → a j = b j := 
sorry

end ordered_sets_equal_l269_269256


namespace part1_part2_l269_269019

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269019


namespace min_x_plus_2y_l269_269782

theorem min_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (2 / x) + (1 / y) = 1) : x + 2 * y ≥ 8 :=
sorry

end min_x_plus_2y_l269_269782


namespace total_elephants_l269_269110

-- Define the conditions in Lean
def G (W : ℕ) : ℕ := 3 * W
def N (G : ℕ) : ℕ := 5 * G
def W : ℕ := 70

-- Define the statement to prove
theorem total_elephants :
  G W + W + N (G W) = 1330 :=
by
  -- Proof to be filled in
  sorry

end total_elephants_l269_269110


namespace card_selection_l269_269628

theorem card_selection :
  (Finset.card {S : Finset (Fin 52) | 
                 S.card = 4 ∧ 
                 ∃ s1 s2 s3 s4 : Fin 52, 
                   s1.val / 13 = s2.val / 13 ∧ s1.val / 13 ≠ s3.val / 13 ∧ s1.val / 13 ≠ s4.val / 13}) = 158004 := by
  sorry

end card_selection_l269_269628


namespace ben_min_sales_l269_269519

theorem ben_min_sales 
    (old_salary : ℕ := 75000) 
    (new_base_salary : ℕ := 45000) 
    (commission_rate : ℚ := 0.15) 
    (sale_amount : ℕ := 750) : 
    ∃ (n : ℕ), n ≥ 267 ∧ (old_salary ≤ new_base_salary + n * ⌊commission_rate * sale_amount⌋) :=
by 
  sorry

end ben_min_sales_l269_269519


namespace bobs_total_profit_l269_269177

theorem bobs_total_profit :
  let cost_parent_dog := 250
  let num_parent_dogs := 2
  let num_puppies := 6
  let cost_food_vaccinations := 500
  let cost_advertising := 150
  let selling_price_parent_dog := 200
  let selling_price_puppy := 350
  let total_cost_parent_dogs := num_parent_dogs * cost_parent_dog
  let total_cost_puppies := cost_food_vaccinations + cost_advertising
  let total_revenue_puppies := num_puppies * selling_price_puppy
  let total_revenue_parent_dogs := num_parent_dogs * selling_price_parent_dog
  let total_revenue := total_revenue_puppies + total_revenue_parent_dogs
  let total_cost := total_cost_parent_dogs + total_cost_puppies
  let total_profit := total_revenue - total_cost
  total_profit = 1350 :=
by
  sorry

end bobs_total_profit_l269_269177


namespace greg_rolls_probability_l269_269923

noncomputable def probability_of_more_ones_than_twos_and_threes_combined : ℚ :=
  (3046.5 : ℚ) / 7776

theorem greg_rolls_probability :
  probability_of_more_ones_than_twos_and_threes_combined = (3046.5 : ℚ) / 7776 := 
by 
  sorry

end greg_rolls_probability_l269_269923


namespace solve_inequality_correct_l269_269097

noncomputable def solve_inequality (a x : ℝ) : Set ℝ :=
  if a > 1 ∨ a < 0 then {x | x ≤ a ∨ x ≥ a^2 }
  else if a = 1 ∨ a = 0 then {x | True}
  else {x | x ≤ a^2 ∨ x ≥ a}

theorem solve_inequality_correct (a x : ℝ) :
  (x^2 - (a^2 + a) * x + a^3 ≥ 0) ↔ 
    (if a > 1 ∨ a < 0 then x ≤ a ∨ x ≥ a^2
      else if a = 1 ∨ a = 0 then True
      else x ≤ a^2 ∨ x ≥ a) :=
by sorry

end solve_inequality_correct_l269_269097


namespace quadratic_equation_with_product_of_roots_20_l269_269131

theorem quadratic_equation_with_product_of_roots_20
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c / a = 20) :
  ∃ b : ℝ, ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  use 1
  use 20
  sorry

end quadratic_equation_with_product_of_roots_20_l269_269131


namespace book_price_is_correct_l269_269146

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage
def discount_percentage : ℝ := 0.30

-- Calculate the CD cost
def cd_cost : ℝ := album_cost * (1 - discount_percentage)

-- Define the additional cost of the book over the CD
def book_cd_diff : ℝ := 4

-- Calculate the book cost
def book_cost : ℝ := cd_cost + book_cd_diff

-- State the proposition to be proved
theorem book_price_is_correct : book_cost = 18 := by
  -- Provide the details of the calculations (optionally)
  sorry

end book_price_is_correct_l269_269146


namespace problem_inequality_l269_269688

theorem problem_inequality (x y z : ℝ) (h1 : x + y + z = 0) (h2 : |x| + |y| + |z| ≤ 1) :
  x + (y / 2) + (z / 3) ≤ 1 / 3 :=
sorry

end problem_inequality_l269_269688


namespace no_partition_exists_l269_269962

theorem no_partition_exists : ¬ ∃ (x y : ℕ), 
    (1 ≤ x ∧ x ≤ 15) ∧ 
    (1 ≤ y ∧ y ≤ 15) ∧ 
    (x * y = 120 - x - y) :=
by
  sorry

end no_partition_exists_l269_269962


namespace max_value_x_y2_z3_l269_269970

theorem max_value_x_y2_z3 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) : 
  x + y^2 + z^3 ≤ 1 :=
by
  sorry

end max_value_x_y2_z3_l269_269970


namespace jenny_chocolate_milk_probability_l269_269807

-- Define the binomial probability function.
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  ( Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

-- Given conditions: probability each day and total number of days.
def probability_each_day : ℚ := 2 / 3
def num_days : ℕ := 7
def successful_days : ℕ := 3

-- The problem statement to prove.
theorem jenny_chocolate_milk_probability :
  binomial_probability num_days successful_days probability_each_day = 280 / 2187 :=
by
  sorry

end jenny_chocolate_milk_probability_l269_269807


namespace part_1_part_2_l269_269010

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269010


namespace simplify_trig_expression_l269_269692

open Real

theorem simplify_trig_expression (α : ℝ) : 
  sin (2 * π - α)^2 + (cos (π + α) * cos (π - α)) + 1 = 2 := 
by 
  sorry

end simplify_trig_expression_l269_269692


namespace johns_allowance_is_30_l269_269938

-- Definitions based on conditions
variables (A : ℚ)
def arcade_fraction := 7/15
def books_fraction := 3/10
def clothes_fraction := 1/6
def toy_store_fraction := 2/5
def candy_store_amount := (6/5) / 5 -- equivalent to $1.20 in decimals

-- Calculations based on conditions and correct answer
def total_spent : ℚ := arcade_fraction + books_fraction + clothes_fraction
def remaining_after_initial_spending := 1 - total_spent
def remaining_after_toy_store := remaining_after_initial_spending * (1 - toy_store_fraction)
def final_remaining := remaining_after_toy_store * A

-- The theorem to prove John's weekly allowance is $30
theorem johns_allowance_is_30 : final_remaining = candy_store_amount → A = 30 := by
  sorry

end johns_allowance_is_30_l269_269938


namespace distance_ran_each_morning_l269_269467

-- Definitions based on conditions
def days_ran : ℕ := 3
def total_distance : ℕ := 2700

-- The goal is to prove the distance ran each morning
theorem distance_ran_each_morning : total_distance / days_ran = 900 :=
by
  sorry

end distance_ran_each_morning_l269_269467


namespace y1_mul_y2_eq_one_l269_269780

theorem y1_mul_y2_eq_one (x1 x2 y1 y2 : ℝ) (h1 : y1^2 = x1) (h2 : y2^2 = x2) 
  (h3 : y1 / (y1^2 - 1) = - (y2 / (y2^2 - 1))) (h4 : y1 + y2 ≠ 0) : y1 * y2 = 1 :=
sorry

end y1_mul_y2_eq_one_l269_269780


namespace A_empty_iff_a_gt_9_over_8_A_one_element_l269_269229

-- Definition of A based on a given condition
def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x + 2 = 0}

-- Problem 1: Prove that if A is empty, then a > 9/8
theorem A_empty_iff_a_gt_9_over_8 {a : ℝ} : 
  (A a = ∅) ↔ (a > 9 / 8) := 
sorry

-- Problem 2: Prove the elements in A when it contains only one element
theorem A_one_element {a : ℝ} : 
  (∃! x, x ∈ A a) ↔ (a = 0 ∧ (A a = {2 / 3})) ∨ (a = 9 / 8 ∧ (A a = {4 / 3})) := 
sorry

end A_empty_iff_a_gt_9_over_8_A_one_element_l269_269229


namespace simplify_expression_l269_269734

theorem simplify_expression : 
  let i : ℂ := complex.I in
  ( (i^3 = -i) → ((2 + i) * (2 - i) = 5) → (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i ) :=
by
  let i : ℂ := complex.I
  assume h₁ : i^3 = -i
  assume h₂ : (2 + i) * (2 - i) = 5
  sorry

end simplify_expression_l269_269734


namespace proof_problem_l269_269777

variable {a b : ℝ}

theorem proof_problem (h₁ : a < b) (h₂ : b < 0) : (b/a) + (a/b) > 2 :=
by 
  sorry

end proof_problem_l269_269777


namespace greatest_integer_of_negative_fraction_l269_269722

-- Define the original fraction
def original_fraction : ℚ := -19 / 5

-- Define the greatest integer function
def greatest_integer_less_than (q : ℚ) : ℤ :=
  Int.floor q

-- The proof problem statement:
theorem greatest_integer_of_negative_fraction :
  greatest_integer_less_than original_fraction = -4 :=
sorry

end greatest_integer_of_negative_fraction_l269_269722


namespace gcd_735_1287_l269_269895

theorem gcd_735_1287 : Int.gcd 735 1287 = 3 := by
  sorry

end gcd_735_1287_l269_269895


namespace polygon_sides_and_diagonals_l269_269252

theorem polygon_sides_and_diagonals (n : ℕ) :
  (180 * (n - 2) = 3 * 360 + 180) → n = 9 ∧ (n - 3 = 6) :=
by
  intro h_sum_angles
  -- This is where you would provide the proof.
  sorry

end polygon_sides_and_diagonals_l269_269252


namespace exists_positive_integers_seq_l269_269415

def sum_of_digits (m : ℕ) : ℕ :=
  m.digits 10 |>.sum

def prod_of_digits (m : ℕ) : ℕ :=
  m.digits 10 |>.prod

theorem exists_positive_integers_seq (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n.succ → ℕ),
    (∀ i : Fin n, sum_of_digits (a i) < sum_of_digits (a i.succ)) ∧
    (∀ i : Fin n, sum_of_digits (a i) = prod_of_digits (a i.succ)) ∧
    (∀ i : Fin n, 0 < (a i)) :=
by
  sorry

end exists_positive_integers_seq_l269_269415


namespace initial_blue_marbles_l269_269234

theorem initial_blue_marbles (B R : ℕ) 
    (h1 : 3 * B = 5 * R) 
    (h2 : 4 * (B - 10) = R + 25) : 
    B = 19 := 
sorry

end initial_blue_marbles_l269_269234


namespace distance_from_hotel_l269_269834

def total_distance := 600
def speed1 := 50
def time1 := 3
def speed2 := 80
def time2 := 4

theorem distance_from_hotel :
  total_distance - (speed1 * time1 + speed2 * time2) = 130 := 
by
  sorry

end distance_from_hotel_l269_269834


namespace initial_notebooks_is_10_l269_269574

-- Define the conditions
def ordered_notebooks := 6
def lost_notebooks := 2
def current_notebooks := 14

-- Define the initial number of notebooks
def initial_notebooks (N : ℕ) :=
  N + ordered_notebooks - lost_notebooks = current_notebooks

-- The proof statement
theorem initial_notebooks_is_10 : initial_notebooks 10 :=
by
  sorry

end initial_notebooks_is_10_l269_269574


namespace problem_solution_l269_269468

noncomputable def S : ℝ :=
  1 / (5 - Real.sqrt 19) - 1 / (Real.sqrt 19 - Real.sqrt 18) + 
  1 / (Real.sqrt 18 - Real.sqrt 17) - 1 / (Real.sqrt 17 - 3) + 
  1 / (3 - Real.sqrt 2)

theorem problem_solution : S = 5 + Real.sqrt 2 := by
  sorry

end problem_solution_l269_269468


namespace no_simultaneous_negative_values_l269_269087

theorem no_simultaneous_negative_values (m n : ℝ) :
  ¬ ((3*m^2 + 4*m*n - 2*n^2 < 0) ∧ (-m^2 - 4*m*n + 3*n^2 < 0)) :=
by
  sorry

end no_simultaneous_negative_values_l269_269087


namespace sum_of_ideals_is_ideal_l269_269085

variables {R : Type*} [CommRing R]

theorem sum_of_ideals_is_ideal 
  (n : ℕ) 
  (hn : n ≥ 2)
  (I : Fin n → Ideal R)
  (H_sum_is_ideal : ∀ (H : Finset (Fin n)), H.nonempty → (H.sum (λ h, I h)).IsIdeal) :
  (Ideal.sum (I ∘ (λ k, (Finset.univ.erase k).prod I))).IsIdeal :=
sorry

end sum_of_ideals_is_ideal_l269_269085


namespace convex_maximum_l269_269222

-- Definitions for the given conditions
def f (x : ℝ) (m : ℝ) : ℝ := (1/6) * x^3 - (1/2) * m * x^2 + x

def f' (x : ℝ) (m : ℝ) : ℝ := deriv (λ x, f x m) x

def f'' (x : ℝ) (m : ℝ) : ℝ := deriv (λ x, f' x m) x

-- The statement representing the proof problem
theorem convex_maximum (m : ℝ) :
  (∀ x ∈ set.Ioo (-1 : ℝ) (2 : ℝ), f'' x m < 0) →
  (∃ x ∈ set.Ico (-1 : ℝ) (2 : ℝ), ∀ y ∈ set.Ioo (-1 : ℝ) (2 : ℝ), f y m ≤ f x m ∧ (∃ z ∈ set.Ioo (-1 : ℝ) (2 : ℝ), f' z m ≠ 0)) :=
sorry

end convex_maximum_l269_269222


namespace cone_surface_area_l269_269711

theorem cone_surface_area (r l: ℝ) (θ : ℝ) (h₁ : r = 3) (h₂ : θ = 2 * π / 3) (h₃: 2 * π * r = θ * l) :
  π * r * l + π * r ^ 2 = 36 * π :=
by
  sorry

end cone_surface_area_l269_269711


namespace probability_not_white_l269_269145

def white_balls : ℕ := 6
def yellow_balls : ℕ := 5
def red_balls : ℕ := 4

def total_balls : ℕ := white_balls + yellow_balls + red_balls
def non_white_balls : ℕ := yellow_balls + red_balls

def probability : ℚ := non_white_balls / total_balls

theorem probability_not_white :
  probability = 3 / 5 :=
by
  -- sorry is a placeholder for the proof
  sorry

end probability_not_white_l269_269145


namespace difference_divisible_by_18_l269_269966

theorem difference_divisible_by_18 (a b : ℤ) : 18 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
by
  sorry

end difference_divisible_by_18_l269_269966


namespace slower_train_speed_l269_269513

theorem slower_train_speed (length_train : ℕ) (speed_fast : ℕ) (time_seconds : ℕ) (distance_meters : ℕ): 
  (length_train = 150) → 
  (speed_fast = 46) → 
  (time_seconds = 108) → 
  (distance_meters = 300) → 
  (distance_meters = (speed_fast - speed_slow) * 5 / 18 * time_seconds) → 
  speed_slow = 36 := 
by
    intros h1 h2 h3 h4 h5
    sorry

end slower_train_speed_l269_269513


namespace savings_is_22_77_cents_per_egg_l269_269314

-- Defining the costs and discount condition
def cost_per_large_egg_StoreA : ℚ := 0.55
def cost_per_extra_large_egg_StoreA : ℚ := 0.65
def discounted_cost_of_three_trays_large_StoreB : ℚ := 38
def total_eggs_in_three_trays : ℕ := 90

-- Savings calculation
def savings_per_egg : ℚ := (cost_per_extra_large_egg_StoreA - (discounted_cost_of_three_trays_large_StoreB / total_eggs_in_three_trays)) * 100

-- The statement to prove
theorem savings_is_22_77_cents_per_egg : savings_per_egg = 22.77 :=
by
  -- Here the proof would go, but we are omitting it with sorry
  sorry

end savings_is_22_77_cents_per_egg_l269_269314


namespace equilateral_triangle_AB_length_l269_269067

noncomputable def Q := 2
noncomputable def R := 3
noncomputable def S := 4

theorem equilateral_triangle_AB_length :
  ∀ (AB BC CA : ℝ), 
  AB = BC ∧ BC = CA ∧ (∃ P : ℝ × ℝ, (Q = 2) ∧ (R = 3) ∧ (S = 4)) →
  AB = 6 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_AB_length_l269_269067


namespace nested_radical_value_l269_269894

noncomputable def nested_radical := λ x : ℝ, x = Real.sqrt (3 - x)

theorem nested_radical_value :
  ∃ x : ℝ, nested_radical x ∧ x ≥ 0 ∧ x = (Real.sqrt 13 - 1) / 2 :=
by
  sorry

end nested_radical_value_l269_269894


namespace part_1_part_2_l269_269006

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269006


namespace remainder_mod_41_l269_269076

theorem remainder_mod_41 (M : ℤ) (hM1 : M = 1234567891011123940) : M % 41 = 0 :=
by
  sorry

end remainder_mod_41_l269_269076


namespace mono_increasing_m_value_l269_269443

theorem mono_increasing_m_value (m : ℝ) :
  (∀ x : ℝ, 0 ≤ 3 * x ^ 2 + 4 * x + m) → (m ≥ 4 / 3) :=
by
  intro h
  sorry

end mono_increasing_m_value_l269_269443


namespace rhombus_area_l269_269846

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 30) (h2 : d2 = 12) : (d1 * d2) / 2 = 180 :=
by
  sorry

end rhombus_area_l269_269846


namespace total_cookies_l269_269083

-- Conditions
def Paul_cookies : ℕ := 45
def Paula_cookies : ℕ := Paul_cookies - 3

-- Question and Answer
theorem total_cookies : Paul_cookies + Paula_cookies = 87 := by
  sorry

end total_cookies_l269_269083


namespace total_people_veg_l269_269357

def people_only_veg : ℕ := 13
def people_both_veg_nonveg : ℕ := 8

theorem total_people_veg : people_only_veg + people_both_veg_nonveg = 21 := by
  sorry

end total_people_veg_l269_269357


namespace bacteria_eradication_time_l269_269483

noncomputable def infected_bacteria (n : ℕ) : ℕ := n

theorem bacteria_eradication_time (n : ℕ) : ∃ t : ℕ, t = n ∧ (∃ infect: ℕ → ℕ, ∀ t < n, infect t ≤ n ∧ infect n = n ∧ (∀ k < n, infect k = 2^(n-k))) :=
by sorry

end bacteria_eradication_time_l269_269483


namespace percent_calculation_l269_269987

theorem percent_calculation (y : ℝ) : (0.3 * 0.7 * y - 0.1 * y) = 0.11 * y ∧ (0.11 * y / y * 100 = 11) := by
  sorry

end percent_calculation_l269_269987


namespace value_of_a_l269_269250

theorem value_of_a
  (a : ℝ)
  (h1 : ∀ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 1)
  (h2 : ∀ (ρ : ℝ), ρ = a)
  (h3 : ∃ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 1 ∧ ρ = a ∧ θ = 0)  :
  a = Real.sqrt 2 / 2 := 
sorry

end value_of_a_l269_269250


namespace proof_equiv_l269_269263

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = 2 ^ Real.sqrt (3 + 2 * x - x ^ 2) }
noncomputable def N : Set ℝ := { x | ∃ y : ℝ, y = Real.log (x - 2) }
def I : Set ℝ := Set.univ
def complement_N : Set ℝ := I \ N

theorem proof_equiv : M ∩ complement_N = { y | 1 ≤ y ∧ y ≤ 2 } :=
sorry

end proof_equiv_l269_269263


namespace nina_money_l269_269137

theorem nina_money (C : ℝ) (h1 : C > 0) (h2 : 6 * C = 8 * (C - 2)) : 6 * C = 48 :=
by
  sorry

end nina_money_l269_269137


namespace tan_alpha_plus_pi_over_3_l269_269216

open Real

theorem tan_alpha_plus_pi_over_3 (α : ℝ) 
  (h1 : sin (2 * α) = cos α) 
  (h2 : α ∈ set.Ioo (π / 2) π) :
  tan (α + π / 3) = sqrt 3 / 3 := 
  sorry

end tan_alpha_plus_pi_over_3_l269_269216


namespace part_1_part_2_l269_269008

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269008


namespace parabola_directrix_equation_l269_269488

theorem parabola_directrix_equation (x y a : ℝ) : 
  (x^2 = 4 * y) → (a = 1) → (y = -a) := by
  intro h1 h2
  rw [h2] -- given a = 1
  sorry

end parabola_directrix_equation_l269_269488


namespace dunkers_starting_lineups_l269_269484

theorem dunkers_starting_lineups :
  let players := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
      alex := 0
      ben := 1
      cam := 2
      remaining_players := players \ {alex, ben, cam}
   in 
  (choose (remaining_players.card) 4 + 1) * 3 + choose (remaining_players.card) 5 = 2277 :=
by
  sorry

end dunkers_starting_lineups_l269_269484


namespace total_games_played_l269_269647

theorem total_games_played (jerry_wins dave_wins ken_wins : ℕ)
  (h1 : jerry_wins = 7)
  (h2 : dave_wins = jerry_wins + 3)
  (h3 : ken_wins = dave_wins + 5) :
  jerry_wins + dave_wins + ken_wins = 32 :=
by
  sorry

end total_games_played_l269_269647


namespace two_triangles_not_separable_by_plane_l269_269765

/-- Definition of a point in three-dimensional space -/
structure Point (α : Type) :=
(x : α)
(y : α)
(z : α)

/-- Definition of a segment joining two points -/
structure Segment (α : Type) :=
(p1 : Point α)
(p2 : Point α)

/-- Definition of a triangle formed by three points -/
structure Triangle (α : Type) :=
(a : Point α)
(b : Point α)
(c : Point α)

/-- Definition of a plane given by a normal vector and a point on the plane -/
structure Plane (α : Type) :=
(n : Point α)
(p : Point α)

/-- Definition of separation of two triangles by a plane -/
def separates (plane : Plane ℝ) (t1 t2 : Triangle ℝ) : Prop :=
  -- Placeholder for the actual separation condition
  sorry

/-- The theorem to be proved -/
theorem two_triangles_not_separable_by_plane (points : Fin 6 → Point ℝ) :
  ∃ t1 t2 : Triangle ℝ, ¬∃ plane : Plane ℝ, separates plane t1 t2 :=
sorry

end two_triangles_not_separable_by_plane_l269_269765


namespace part1_part2_l269_269016

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269016


namespace range_of_a_l269_269618

noncomputable def f (a x : ℝ) :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a) ∧
  (∀ x y : ℝ, x < y → f a x ≥ f a y) →
  1 / 7 ≤ a ∧ a < 1 / 3 := 
sorry

end range_of_a_l269_269618


namespace average_speed_is_75_l269_269139

-- Define the conditions
def speed_first_hour : ℕ := 90
def speed_second_hour : ℕ := 60
def total_time : ℕ := 2

-- Define the average speed and prove it is equal to the given answer
theorem average_speed_is_75 : 
  (speed_first_hour + speed_second_hour) / total_time = 75 := 
by 
  -- We will skip the proof for now
  sorry

end average_speed_is_75_l269_269139


namespace height_percentage_differences_l269_269175

variable (B : ℝ) (A : ℝ) (R : ℝ)
variable (h1 : A = 1.25 * B) (h2 : R = 1.0625 * B)

theorem height_percentage_differences :
  (100 * (A - B) / B = 25) ∧
  (100 * (A - R) / A = 15) ∧
  (100 * (R - B) / B = 6.25) :=
by
  sorry

end height_percentage_differences_l269_269175


namespace counterexample_disproving_proposition_l269_269418

theorem counterexample_disproving_proposition (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = angle2) : False :=
by
  have h_contradiction : angle1 ≠ angle2 := sorry
  exact h_contradiction h2

end counterexample_disproving_proposition_l269_269418


namespace rectangle_perimeter_l269_269689

theorem rectangle_perimeter : 
  ∃ (x y a b : ℝ), 
  (x * y = 2016) ∧ 
  (a * b = 2016) ∧ 
  (x^2 + y^2 = 4 * (a^2 - b^2)) → 
  2 * (x + y) = 8 * Real.sqrt 1008 :=
sorry

end rectangle_perimeter_l269_269689


namespace bugs_ate_each_l269_269664

theorem bugs_ate_each : 
  ∀ (total_bugs total_flowers each_bug_flowers : ℕ), 
    total_bugs = 3 ∧ total_flowers = 6 ∧ each_bug_flowers = total_flowers / total_bugs -> each_bug_flowers = 2 := by
  sorry

end bugs_ate_each_l269_269664


namespace original_price_per_pound_l269_269489

theorem original_price_per_pound (P x : ℝ)
  (h1 : 0.2 * x * P = 0.2 * x)
  (h2 : x * P = x * P)
  (h3 : 1.08 * (0.8 * x) * 1.08 = 1.08 * x * P) :
  P = 1.08 :=
sorry

end original_price_per_pound_l269_269489


namespace randy_used_36_blocks_l269_269829

-- Define the initial number of blocks
def initial_blocks : ℕ := 59

-- Define the number of blocks left
def blocks_left : ℕ := 23

-- Define the number of blocks used
def blocks_used (initial left : ℕ) : ℕ := initial - left

-- Prove that Randy used 36 blocks
theorem randy_used_36_blocks : blocks_used initial_blocks blocks_left = 36 := 
by
  -- Proof will be here
  sorry

end randy_used_36_blocks_l269_269829


namespace miriam_cleaning_room_time_l269_269821

theorem miriam_cleaning_room_time
  (laundry_time : Nat := 30)
  (bathroom_time : Nat := 15)
  (homework_time : Nat := 40)
  (total_time : Nat := 120) :
  ∃ room_time : Nat, laundry_time + bathroom_time + homework_time + room_time = total_time ∧
                  room_time = 35 := by
  sorry

end miriam_cleaning_room_time_l269_269821


namespace estimate_high_score_students_l269_269549

noncomputable def students_with_high_scores (num_students : ℕ) (mean : ℝ) (stddev : ℝ) (lower_bound upper_bound : ℝ) (prob_within_bounds : ℝ) : ℕ :=
  if (num_students = 50) ∧ (mean = 105) ∧ (stddev = 10) ∧ (lower_bound = 95) ∧ (upper_bound = 105) ∧ (prob_within_bounds = 0.32) then 9 else 0

theorem estimate_high_score_students :
  students_with_high_scores 50 105 10 95 105 0.32 = 9 :=
by sorry

end estimate_high_score_students_l269_269549


namespace balls_in_boxes_l269_269272

theorem balls_in_boxes : 
  let balls := 4
  let boxes := 3
  (boxes^balls = 81) :=
by sorry

end balls_in_boxes_l269_269272


namespace cosine_expression_value_l269_269657

noncomputable def c : ℝ := 2 * Real.pi / 7

theorem cosine_expression_value :
  (Real.cos (3 * c) * Real.cos (5 * c) * Real.cos (6 * c)) / 
  (Real.cos c * Real.cos (2 * c) * Real.cos (3 * c)) = 1 :=
by
  sorry

end cosine_expression_value_l269_269657


namespace number_of_performances_l269_269205

theorem number_of_performances (hanna_songs : ℕ) (mary_songs : ℕ) (alina_songs : ℕ) (tina_songs : ℕ)
    (hanna_cond : hanna_songs = 4)
    (mary_cond : mary_songs = 7)
    (alina_cond : 4 < alina_songs ∧ alina_songs < 7)
    (tina_cond : 4 < tina_songs ∧ tina_songs < 7) :
    ((hanna_songs + mary_songs + alina_songs + tina_songs) / 3) = 7 :=
by
  -- proof steps would go here
  sorry

end number_of_performances_l269_269205


namespace solution_set_of_x_sq_gt_x_l269_269977

theorem solution_set_of_x_sq_gt_x :
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} := 
sorry

end solution_set_of_x_sq_gt_x_l269_269977


namespace Sam_needs_16_more_hours_l269_269423

noncomputable def Sam_hourly_rate : ℝ :=
  460 / 23

noncomputable def Sam_earnings_Sep_to_Feb : ℝ :=
  8 * Sam_hourly_rate

noncomputable def Sam_total_earnings : ℝ :=
  460 + Sam_earnings_Sep_to_Feb

noncomputable def Sam_remaining_money : ℝ :=
  Sam_total_earnings - 340

noncomputable def Sam_needed_money : ℝ :=
  600 - Sam_remaining_money

noncomputable def Sam_additional_hours_needed : ℝ :=
  Sam_needed_money / Sam_hourly_rate

theorem Sam_needs_16_more_hours : Sam_additional_hours_needed = 16 :=
by 
  sorry

end Sam_needs_16_more_hours_l269_269423


namespace circle_diameter_problem_circle_diameter_l269_269538

theorem circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 2 * r := by
  -- Steps here would include deriving d from the given area
  sorry

theorem problem_circle_diameter (r d : ℝ) (h_area : π * r^2 = 4 * π) : d = 4 := by
  have h_radius : r = 2 := by
    -- Derive r = 2 directly from the given area
    sorry
  have : d = 2 * r := circle_diameter r d h_area
  rw [h_radius] at this
  exact this

end circle_diameter_problem_circle_diameter_l269_269538


namespace cos_half_diff_proof_l269_269224

noncomputable def cos_half_diff (A B C : ℝ) (h_triangle : A + B + C = 180)
  (h_relation : A + C = 2 * B)
  (h_equation : (1 / Real.cos A) + (1 / Real.cos C) = - (Real.sqrt 2 / Real.cos B)) : Real :=
  Real.cos ((A - C) / 2)

theorem cos_half_diff_proof (A B C : ℝ)
  (h_triangle : A + B + C = 180)
  (h_relation : A + C = 2 * B)
  (h_equation : (1 / Real.cos A) + (1 / Real.cos C) = - (Real.sqrt 2 / Real.cos B)) :
  cos_half_diff A B C h_triangle h_relation h_equation = -Real.sqrt 2 / 2 :=
sorry

end cos_half_diff_proof_l269_269224


namespace sector_area_l269_269425

theorem sector_area (r : ℝ) (alpha : ℝ) (h_r : r = 2) (h_alpha : alpha = π / 4) : 
  (1 / 2) * r^2 * alpha = π / 2 :=
by
  rw [h_r, h_alpha]
  -- proof steps would go here
  sorry

end sector_area_l269_269425


namespace factorize_quadratic_l269_269408

theorem factorize_quadratic (x : ℝ) : x^2 - 6 * x + 9 = (x - 3)^2 :=
by {
  sorry  -- Proof goes here
}

end factorize_quadratic_l269_269408


namespace perpendicular_line_through_point_l269_269162

theorem perpendicular_line_through_point (x y : ℝ) : (x, y) = (0, -3) ∧ (∀ x y : ℝ, 2 * x + 3 * y - 6 = 0) → 3 * x - 2 * y - 6 = 0 :=
by
  sorry

end perpendicular_line_through_point_l269_269162


namespace part_1_part_2_l269_269011

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269011


namespace fraction_of_roots_l269_269739

theorem fraction_of_roots (a b : ℝ) (h : a * b = -209) (h_sum : a + b = -8) : 
  (a * b) / (a + b) = 209 / 8 := 
by 
  sorry

end fraction_of_roots_l269_269739


namespace tan_alpha_solution_l269_269050

theorem tan_alpha_solution (α : Real) (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (4 * Real.sin α - Real.cos α) = 2 := 
by 
  sorry

end tan_alpha_solution_l269_269050


namespace induction_step_n_eq_1_l269_269682

theorem induction_step_n_eq_1 : (1 + 2 + 3 = (1+1)*(2*1+1)) :=
by
  -- Proof would go here
  sorry

end induction_step_n_eq_1_l269_269682


namespace numerator_greater_denominator_l269_269490

theorem numerator_greater_denominator (x : ℝ) (h1 : -3 ≤ x) (h2 : x ≤ 3) (h3 : 5 * x + 3 > 8 - 3 * x) : (5 / 8) < x ∧ x ≤ 3 :=
by
  sorry

end numerator_greater_denominator_l269_269490


namespace max_value_l269_269954

noncomputable def max_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  (x + y + z)^2 / (x^2 + y^2 + z^2)

theorem max_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  max_fraction x y z hx hy hz ≤ 3 :=
sorry

end max_value_l269_269954


namespace tom_travel_time_to_virgo_island_l269_269509

-- Definitions based on conditions
def boat_trip_time : ℕ := 2
def plane_trip_time : ℕ := 4 * boat_trip_time
def total_trip_time : ℕ := plane_trip_time + boat_trip_time

-- Theorem we need to prove
theorem tom_travel_time_to_virgo_island : total_trip_time = 10 := by
  sorry

end tom_travel_time_to_virgo_island_l269_269509


namespace cube_of_prism_volume_l269_269374

theorem cube_of_prism_volume (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^3 :=
by
  sorry

end cube_of_prism_volume_l269_269374


namespace g_value_at_4_l269_269850

noncomputable def g : ℝ → ℝ := sorry -- We will define g here

def functional_condition (g : ℝ → ℝ) := ∀ x y : ℝ, x * g y = y * g x
def g_value_at_12 := g 12 = 30

theorem g_value_at_4 (g : ℝ → ℝ) (h₁ : functional_condition g) (h₂ : g_value_at_12) : g 4 = 10 := 
sorry

end g_value_at_4_l269_269850


namespace movie_theater_loss_l269_269556

theorem movie_theater_loss :
  let capacity := 50
  let ticket_price := 8.0
  let tickets_sold := 24
  (capacity * ticket_price - tickets_sold * ticket_price) = 208 := by
  sorry

end movie_theater_loss_l269_269556


namespace phung_more_than_chiu_l269_269663

theorem phung_more_than_chiu
  (C P H : ℕ)
  (h1 : C = 56)
  (h2 : H = P + 5)
  (h3 : C + P + H = 205) :
  P - C = 16 :=
by
  sorry

end phung_more_than_chiu_l269_269663


namespace train_speed_correct_l269_269568

def length_of_train := 280 -- in meters
def time_to_pass_tree := 16 -- in seconds
def speed_of_train := 63 -- in km/hr

theorem train_speed_correct :
  (length_of_train / time_to_pass_tree) * (3600 / 1000) = speed_of_train :=
sorry

end train_speed_correct_l269_269568


namespace correct_mean_l269_269993

theorem correct_mean (mean n incorrect_value correct_value : ℝ) 
  (hmean : mean = 150) (hn : n = 20) (hincorrect : incorrect_value = 135) (hcorrect : correct_value = 160):
  (mean * n - incorrect_value + correct_value) / n = 151.25 :=
by
  sorry

end correct_mean_l269_269993


namespace problem_solution_l269_269051

variable (a : ℝ)

theorem problem_solution (h : a ≠ 0) : a^2 + 1 > 1 :=
sorry

end problem_solution_l269_269051


namespace parabola_intersects_x_axis_l269_269302

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 + 2 * x + m - 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := 4 - 4 * (m - 1)

-- Lean statement to prove the range of m
theorem parabola_intersects_x_axis (m : ℝ) : (∃ x : ℝ, quadratic x m = 0) ↔ m ≤ 2 := by
  sorry

end parabola_intersects_x_axis_l269_269302


namespace prob_A_wins_4_consecutive_games_prob_fifth_game_needed_prob_C_is_ultimate_winner_l269_269982
open Classical

-- Definitions
def player := Type
def game_result := prod player player
def initial_players : player × player × player := (A, B, C)
def initial_conditions : ∀ (x y : player), x ≠ y

-- Functional probabilities
def winning_probability := (1 : ℚ) / 2

-- Proof statements
theorem prob_A_wins_4_consecutive_games 
  (h1 : ∀ (p1 p2 : player), p1 ≠ p2)
  (h2 : ∀ p, winning_probability = (1 : ℚ) / 2)
  : (winning_probability ^ 4) = (1 : ℚ) / 16 :=
  sorry

theorem prob_fifth_game_needed 
  (h1 : ∀ (p1 p2 : player), p1 ≠ p2)
  (h2 : ∀ p, winning_probability = (1 : ℚ) / 2)
  : (1 - (4 * (winning_probability ^ 4))) = 3 / 4 :=
  sorry

theorem prob_C_is_ultimate_winner 
  (h1 : ∀ (p1 p2 : player), p1 ≠ p2)
  (h2 : ∀ p, winning_probability = (1 : ℚ) / 2)
  : ((1 / 8) + (1 / 8) + (1 / 8) + (1 / 16)) = 7 / 16 :=
  sorry

end prob_A_wins_4_consecutive_games_prob_fifth_game_needed_prob_C_is_ultimate_winner_l269_269982


namespace batsman_average_after_17th_match_l269_269135

theorem batsman_average_after_17th_match 
  (A : ℕ) 
  (h1 : (16 * A + 87) / 17 = A + 3) : 
  A + 3 = 39 := 
sorry

end batsman_average_after_17th_match_l269_269135


namespace bologna_sandwiches_l269_269582

variable (C B P : ℕ)

theorem bologna_sandwiches (h1 : C = 1) (h2 : B = 7) (h3 : P = 8)
                          (h4 : C + B + P = 16) (h5 : 80 / 16 = 5) :
                          B * 5 = 35 :=
by
  -- omit the proof part
  sorry

end bologna_sandwiches_l269_269582


namespace fewer_twos_for_100_l269_269327

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l269_269327


namespace find_x_values_l269_269619

open Real

theorem find_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h₁ : x + 1/y = 5) (h₂ : y + 1/x = 7/4) : 
  x = 4/7 ∨ x = 5 := 
by sorry

end find_x_values_l269_269619


namespace cases_in_1990_is_correct_l269_269931

-- Define the initial and final number of cases.
def initial_cases : ℕ := 600000
def final_cases : ℕ := 200

-- Define the years and time spans.
def year_1970 : ℕ := 1970
def year_1985 : ℕ := 1985
def year_2000 : ℕ := 2000

def span_1970_to_1985 : ℕ := year_1985 - year_1970 -- 15 years
def span_1985_to_2000 : ℕ := year_2000 - year_1985 -- 15 years

-- Define the rate of decrease from 1970 to 1985 as r cases per year.
-- Define the rate of decrease from 1985 to 2000 as (r / 2) cases per year.
def rate_of_decrease_1 (r : ℕ) := r
def rate_of_decrease_2 (r : ℕ) := r / 2

-- Define the intermediate number of cases in 1985.
def cases_in_1985 (r : ℕ) : ℕ := initial_cases - (span_1970_to_1985 * rate_of_decrease_1 r)

-- Define the number of cases in 1990.
def cases_in_1990 (r : ℕ) : ℕ := cases_in_1985 r - (5 * rate_of_decrease_2 r) -- 5 years from 1985 to 1990

-- Total decrease in cases over 30 years.
def total_decrease : ℕ := initial_cases - final_cases

-- Formalize the proof that the number of cases in 1990 is 133,450.
theorem cases_in_1990_is_correct : 
  ∃ (r : ℕ), 15 * rate_of_decrease_1 r + 15 * rate_of_decrease_2 r = total_decrease ∧ cases_in_1990 r = 133450 := 
by {
  sorry
}

end cases_in_1990_is_correct_l269_269931


namespace prove_inequality_l269_269778

theorem prove_inequality
  (a : ℕ → ℕ) -- Define a sequence of natural numbers
  (h_initial : a 1 > a 0) -- Initial condition
  (h_recurrence : ∀ n ≥ 2, a n = 3 * a (n - 1) - 2 * a (n - 2)) -- Recurrence relation
  : a 100 > 2^99 := by
  sorry -- Proof placeholder

end prove_inequality_l269_269778


namespace no_roots_one_and_neg_one_l269_269480

theorem no_roots_one_and_neg_one (a b : ℝ) : ¬ ((1 + a + b = 0) ∧ (-1 + a + b = 0)) :=
by
  sorry

end no_roots_one_and_neg_one_l269_269480


namespace product_of_numbers_l269_269496

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 :=
sorry

end product_of_numbers_l269_269496


namespace theater_loss_l269_269561

/-- 
A movie theater has a total capacity of 50 people and charges $8 per ticket.
On a Tuesday night, they only sold 24 tickets. 
Prove that the revenue lost by not selling out is $208.
-/
theorem theater_loss 
  (capacity : ℕ) 
  (price : ℕ) 
  (sold_tickets : ℕ) 
  (h_cap : capacity = 50) 
  (h_price : price = 8) 
  (h_sold : sold_tickets = 24) : 
  capacity * price - sold_tickets * price = 208 :=
by
  sorry

end theater_loss_l269_269561


namespace homework_time_decrease_l269_269318

variable (x : ℝ)
variable (initial_time final_time : ℝ)
variable (adjustments : ℕ)

def rate_of_decrease (initial_time final_time : ℝ) (adjustments : ℕ) (x : ℝ) := 
  initial_time * (1 - x)^adjustments = final_time

theorem homework_time_decrease 
  (h_initial : initial_time = 100) 
  (h_final : final_time = 70)
  (h_adjustments : adjustments = 2)
  (h_decrease : rate_of_decrease initial_time final_time adjustments x) : 
  100 * (1 - x)^2 = 70 :=
by
  sorry

end homework_time_decrease_l269_269318


namespace ratio_enlarged_by_nine_l269_269888

theorem ratio_enlarged_by_nine (a b : ℕ) (h : b ≠ 0) :
  (3 * a) / (b / 3) = 9 * (a / b) :=
by
  have h1 : b / 3 ≠ 0 := by sorry
  have h2 : a * 3 ≠ 0 := by sorry
  sorry

end ratio_enlarged_by_nine_l269_269888


namespace number_of_correct_statements_l269_269420

theorem number_of_correct_statements (a : ℚ) : 
  (¬ (a < 0 → -a < 0) ∧ ¬ (|a| > 0) ∧ ¬ ((a < 0 ∨ -a < 0) ∧ ¬ (a = 0))) 
  → 0 = 0 := 
by
  intro h
  sorry

end number_of_correct_statements_l269_269420


namespace circumcircle_locus_l269_269365

open EuclideanGeometry

variables {R r d : ℝ} {O I : Point}

-- Conditions
axiom cond1 : 0 < r ∧ r < R
axiom cond2 : d = dist O I
axiom cond3 : ∀ (A B : Point), is_chord O A B ∧ is_tangent I A B (chord_tangent_point A B I) 

-- Prove the locus of the centers of circumcircles
theorem circumcircle_locus :
  ∃ (M : Point), ∀ (A B : Point), is_chord O A B ∧ is_tangent I A B (chord_tangent_point A B I) 
    → dist O M = (R^2 - d^2) / (2 * r) :=
by sorry

end circumcircle_locus_l269_269365


namespace circle_diameter_l269_269529

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  sorry

end circle_diameter_l269_269529


namespace garrison_men_initial_l269_269154

theorem garrison_men_initial (M : ℕ) (P : ℕ):
  (P = M * 40) →
  (P / 2 = (M + 2000) * 10) →
  M = 2000 :=
by
  intros h1 h2
  sorry

end garrison_men_initial_l269_269154


namespace sum_four_digit_integers_l269_269353

def a := 1000
def l := 9999
def n := 9999 - 1000 + 1
def S (n : ℕ) (a : ℕ) (l : ℕ) := n / 2 * (a + l)

theorem sum_four_digit_integers : S n a l = 49495500 :=
by
  sorry

end sum_four_digit_integers_l269_269353


namespace tickets_left_l269_269880

-- Define the number of tickets won by Dave
def tickets_won : ℕ := 14

-- Define the number of tickets lost by Dave
def tickets_lost : ℕ := 2

-- Define the number of tickets used to buy toys
def tickets_used : ℕ := 10

-- The theorem to prove that the number of tickets left is 2
theorem tickets_left : tickets_won - tickets_lost - tickets_used = 2 := by
  -- Initial computation of tickets left after losing some
  let tickets_after_lost := tickets_won - tickets_lost
  -- Computation of tickets left after using some
  let tickets_after_used := tickets_after_lost - tickets_used
  show tickets_after_used = 2
  sorry

end tickets_left_l269_269880


namespace second_pipe_fill_time_l269_269858

theorem second_pipe_fill_time (x : ℝ) :
  (1 / 18) + (1 / x) - (1 / 45) = (1 / 15) → x = 30 :=
by
  intro h
  sorry

end second_pipe_fill_time_l269_269858


namespace part_I_part_II_part_III_l269_269622

-- Conditions
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def line_through_origin (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (y = k * x)

def distinct_points_on_ellipse (A B P : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse P.1 P.2 ∧ A ≠ P ∧ B ≠ P

variables {A B P C D E F : ℝ × ℝ}
variables {k1 k2 : ℝ}

-- (I)
theorem part_I (h : distinct_points_on_ellipse A B P) (hA : A = (-2, 0)) (hB : B = (2, 0)) :
  ¬ (∃ P : ℝ × ℝ, ellipse P.1 P.2 ∧ P ≠ A ∧ P ≠ B ∧ (vector_angle (P.1 - A.1, P.2 - A.2) (P.1 - B.1, P.2 - B.2) = real.pi / 2)) :=
sorry

-- (II)
theorem part_II (h : distinct_points_on_ellipse A B P) (h₁ : k1 ≠ 0) (h₂ : k2 ≠ 0) :
  k1 * k2 = -1 / 4 :=
sorry

-- (III)
theorem part_III (h : distinct_points_on_ellipse A B P) (hC : line_through_origin C.1 C.2 ∧ ∀ y, ellipse y (k1 * y) → ellipse C.1 C.2) (hE : line_through_origin E.1 E.2 ∧ ∀ y, ellipse y (k2 * y) → ellipse E.1 E.2) :
  |(dist C D)| ^ 2 + |(dist E F)| ^ 2 = 20 :=
sorry

end part_I_part_II_part_III_l269_269622


namespace correct_statement_l269_269248

variable {a b : Type} -- Let a and b be types representing lines
variable {α β : Type} -- Let α and β be types representing planes

-- Define parallel relations for lines and planes
def parallel (L P : Type) : Prop := sorry

-- Define the subset relation for lines in planes
def subset (L P : Type) : Prop := sorry

-- Now state the theorem corresponding to the correct answer
theorem correct_statement (h1 : parallel α β) (h2 : subset a α) : parallel a β :=
sorry

end correct_statement_l269_269248


namespace find_angle_A_find_sum_b_c_l269_269930

-- Given the necessary conditions
variables (a b c : ℝ)
variables (A B C : ℝ)
variables (sin cos : ℝ → ℝ)

-- Assuming necessary trigonometric identities
axiom sin_squared_add_cos_squared : ∀ (x : ℝ), sin x * sin x + cos x * cos x = 1
axiom cos_sum : ∀ (x y : ℝ), cos (x + y) = cos x * cos y - sin x * sin y

-- Condition: 2 sin^2(A) + 3 cos(B+C) = 0
axiom condition1 : 2 * sin A * sin A + 3 * cos (B + C) = 0

-- Condition: The area of the triangle is S = 5 √3
axiom condition2 : 1 / 2 * b * c * sin A = 5 * Real.sqrt 3

-- Condition: The length of side a = √21
axiom condition3 : a = Real.sqrt 21

-- Part (1): Prove the measure of angle A
theorem find_angle_A : A = π / 3 :=
sorry

-- Part (2): Given S = 5√3 and a = √21, find b + c.
theorem find_sum_b_c : b + c = 9 :=
sorry

end find_angle_A_find_sum_b_c_l269_269930


namespace bus_speed_kmph_l269_269366

theorem bus_speed_kmph : 
  let distance := 600.048 
  let time := 30
  (distance / time) * 3.6 = 72.006 :=
by
  sorry

end bus_speed_kmph_l269_269366


namespace P_in_first_quadrant_l269_269805

def point_in_first_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 > 0

theorem P_in_first_quadrant (k : ℝ) (h : k > 0) : point_in_first_quadrant (3, k) :=
by
  sorry

end P_in_first_quadrant_l269_269805


namespace wrongly_entered_mark_l269_269565

theorem wrongly_entered_mark (x : ℝ) : 
  (∀ (correct_mark avg_increase pupils : ℝ), 
  correct_mark = 45 ∧ avg_increase = 0.5 ∧ pupils = 80 ∧
  (avg_increase * pupils = (x - correct_mark)) →
  x = 85) :=
by 
  intro correct_mark avg_increase pupils
  rintro ⟨hc, ha, hp, h⟩
  sorry

end wrongly_entered_mark_l269_269565


namespace abs_inequality_solution_set_l269_269978

theorem abs_inequality_solution_set (x : ℝ) : 
  (|2 * x - 3| ≤ 1) ↔ (1 ≤ x ∧ x ≤ 2) := 
by
  sorry

end abs_inequality_solution_set_l269_269978


namespace problem_inequality_l269_269687

theorem problem_inequality (x y z : ℝ) (h1 : x + y + z = 0) (h2 : |x| + |y| + |z| ≤ 1) :
  x + (y / 2) + (z / 3) ≤ 1 / 3 :=
sorry

end problem_inequality_l269_269687


namespace nth_term_series_l269_269516

def a_n (n : ℕ) : ℝ :=
  if n % 2 = 1 then -4 else 7

theorem nth_term_series (n : ℕ) : a_n n = 1.5 + 5.5 * (-1) ^ n :=
by
  sorry

end nth_term_series_l269_269516


namespace B_won_third_four_times_l269_269060

noncomputable def first_place := 5
noncomputable def second_place := 2
noncomputable def third_place := 1

structure ContestantScores :=
  (A_score : ℕ)
  (B_score : ℕ)
  (C_score : ℕ)

def competition_results (A B C : ContestantScores) (a b c : ℕ) : Prop :=
  A.A_score = 26 ∧ B.B_score = 11 ∧ C.C_score = 11 ∧ 1 = 1 ∧ -- B won first place once is synonymous to holding true
  a > b ∧ b > c ∧ a = 5 ∧ b = 2 ∧ c = 1

theorem B_won_third_four_times :
  ∃ (A B C : ContestantScores), competition_results A B C first_place second_place third_place → 
  B.B_score = 4 * third_place + first_place := 
sorry

end B_won_third_four_times_l269_269060


namespace fewer_twos_result_100_l269_269325

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l269_269325


namespace weight_of_oil_per_ml_l269_269266

variable (w : ℝ)  -- Weight of the oil per ml
variable (total_volume : ℝ := 150)  -- Bowl volume
variable (oil_fraction : ℝ := 2/3)  -- Fraction of oil
variable (vinegar_fraction : ℝ := 1/3)  -- Fraction of vinegar
variable (vinegar_density : ℝ := 4)  -- Vinegar density in g/ml
variable (total_weight : ℝ := 700)  -- Total weight in grams

theorem weight_of_oil_per_ml :
  (total_volume * oil_fraction * w) + (total_volume * vinegar_fraction * vinegar_density) = total_weight →
  w = 5 := by
  sorry

end weight_of_oil_per_ml_l269_269266


namespace lucas_initial_pet_beds_l269_269264

-- Definitions from the problem conditions
def additional_beds := 8
def beds_per_pet := 2
def pets := 10

-- Statement to prove
theorem lucas_initial_pet_beds :
  (pets * beds_per_pet) - additional_beds = 12 := 
by
  sorry

end lucas_initial_pet_beds_l269_269264


namespace circle_diameter_l269_269527

theorem circle_diameter (A : ℝ) (h : A = 4 * Real.pi) : ∃ d : ℝ, d = 4 :=
by
  let r := real.sqrt (4)
  let d := 2 * r
  use d
  sorry

end circle_diameter_l269_269527


namespace greatest_value_of_b_l269_269767

theorem greatest_value_of_b (b : ℝ) : -b^2 + 8 * b - 15 ≥ 0 → b ≤ 5 := sorry

end greatest_value_of_b_l269_269767


namespace find_g_g2_l269_269630

def g (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 1

theorem find_g_g2 : g (g 2) = 2630 := by
  sorry

end find_g_g2_l269_269630


namespace compound_interest_correct_l269_269056

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem compound_interest_correct (SI R T : ℝ) (hSI : SI = 58) (hR : R = 5) (hT : T = 2) : 
  compound_interest (SI * 100 / (R * T)) R T = 59.45 :=
by
  sorry

end compound_interest_correct_l269_269056


namespace geometric_sequence_x_l269_269906

theorem geometric_sequence_x (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 := by
  sorry

end geometric_sequence_x_l269_269906


namespace max_regions_11_l269_269627

noncomputable def max_regions (n : ℕ) : ℕ :=
  1 + n * (n + 1) / 2

theorem max_regions_11 : max_regions 11 = 67 := by
  unfold max_regions
  norm_num

end max_regions_11_l269_269627


namespace fewer_twos_result_100_l269_269323

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l269_269323


namespace new_cube_edge_length_l269_269847

theorem new_cube_edge_length
  (a1 a2 a3 : ℝ)
  (h1 : a1 = 3) 
  (h2 : a2 = 4) 
  (h3 : a3 = 5) :
  (a1^3 + a2^3 + a3^3)^(1/3) = 6 := by
sorry

end new_cube_edge_length_l269_269847


namespace complex_expr_equals_l269_269738

noncomputable def complex_expr : ℂ := (5 * (1 + complex.i^3)) / ((2 + complex.i) * (2 - complex.i))

theorem complex_expr_equals : complex_expr = (1 - complex.i) := 
sorry

end complex_expr_equals_l269_269738


namespace opposite_neg_9_l269_269708

theorem opposite_neg_9 : 
  ∃ y : Int, -9 + y = 0 ∧ y = 9 :=
by
  sorry

end opposite_neg_9_l269_269708


namespace arithmetic_sequence_length_l269_269404

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ (a_1 d a_n : ℤ), a_1 = -3 ∧ d = 4 ∧ a_n = 45 → n = 13 :=
by
  sorry

end arithmetic_sequence_length_l269_269404


namespace classroom_width_perimeter_ratio_l269_269745

theorem classroom_width_perimeter_ratio
  (L : Real) (W : Real) (P : Real)
  (hL : L = 15) (hW : W = 10)
  (hP : P = 2 * (L + W)) :
  W / P = 1 / 5 :=
sorry

end classroom_width_perimeter_ratio_l269_269745


namespace part1_part2_l269_269257

def A (x : ℝ) : Prop := x ^ 2 - 2 * x - 8 < 0
def B (x : ℝ) : Prop := x ^ 2 + 2 * x - 3 > 0
def C (a : ℝ) (x : ℝ) : Prop := x ^ 2 - 3 * a * x + 2 * a ^ 2 < 0

theorem part1 : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x < 4} := 
by sorry

theorem part2 (a : ℝ) : {x : ℝ | C a x} ⊆ {x : ℝ | A x} ∩ {x : ℝ | B x} ↔ (a = 0 ∨ (1 ≤ a ∧ a ≤ 2)) := 
by sorry

end part1_part2_l269_269257


namespace billy_age_l269_269757

theorem billy_age (B J : ℕ) (h1 : B = 3 * J) (h2 : B + J = 64) : B = 48 :=
by
  sorry

end billy_age_l269_269757


namespace right_triangle_inequality_l269_269950

theorem right_triangle_inequality (a b c : ℝ) (h₁ : a^2 + b^2 = c^2) (h₂ : a ≤ b) (h₃ : b < c) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3 * Real.sqrt 2) * a * b * c :=
by
  sorry

end right_triangle_inequality_l269_269950


namespace clara_sells_total_cookies_l269_269583

theorem clara_sells_total_cookies :
  let cookies_per_box_1 := 12
  let cookies_per_box_2 := 20
  let cookies_per_box_3 := 16
  let cookies_per_box_4 := 18
  let cookies_per_box_5 := 22

  let boxes_sold_1 := 50.5
  let boxes_sold_2 := 80.25
  let boxes_sold_3 := 70.75
  let boxes_sold_4 := 65.5
  let boxes_sold_5 := 55.25

  let total_cookies_1 := cookies_per_box_1 * boxes_sold_1
  let total_cookies_2 := cookies_per_box_2 * boxes_sold_2
  let total_cookies_3 := cookies_per_box_3 * boxes_sold_3
  let total_cookies_4 := cookies_per_box_4 * boxes_sold_4
  let total_cookies_5 := cookies_per_box_5 * boxes_sold_5

  let total_cookies := total_cookies_1 + total_cookies_2 + total_cookies_3 + total_cookies_4 + total_cookies_5

  total_cookies = 5737.5 :=
by
  sorry

end clara_sells_total_cookies_l269_269583


namespace DeAndre_score_prob_l269_269883

theorem DeAndre_score_prob :
  let P_make : ℝ := 0.40 in
  let P_miss : ℝ := 1 - P_make in
  let P_miss_both : ℝ := P_miss * P_miss in
  let P_at_least_one : ℝ := 1 - P_miss_both in
  P_at_least_one = 0.64 :=
by
  let P_make := 0.40
  let P_miss := 1 - P_make
  let P_miss_both := P_miss * P_miss
  let P_at_least_one := 1 - P_miss_both
  have h : P_at_least_one = 0.64 := by sorry
  exact h

end DeAndre_score_prob_l269_269883


namespace gcd_78_182_l269_269127

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := 
by
  sorry

end gcd_78_182_l269_269127


namespace intersect_points_count_l269_269099

open Classical
open Real

noncomputable def f : ℝ → ℝ := sorry
def f_inv : ℝ → ℝ := sorry

axiom f_invertible : ∀ x y : ℝ, f x = f y ↔ x = y

theorem intersect_points_count : ∃ (count : ℕ), count = 3 ∧ ∀ x : ℝ, (f (x ^ 3) = f (x ^ 5)) ↔ (x = 0 ∨ x = 1 ∨ x = -1) :=
by sorry

end intersect_points_count_l269_269099


namespace fewer_twos_result_100_l269_269324

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l269_269324


namespace team_plays_60_games_in_division_l269_269246

noncomputable def number_of_division_games (N M : ℕ) (hNM : N > M) (hM : M > 5) (h_total : 4 * N + 5 * M = 90) : ℕ :=
  4 * N

theorem team_plays_60_games_in_division (N M : ℕ) 
  (hNM : N > M) (hM : M > 5) (h_total : 4 * N + 5 * M = 90) 
  : number_of_division_games N M hNM hM h_total = 60 := 
sorry

end team_plays_60_games_in_division_l269_269246


namespace largest_x_by_equation_l269_269481

theorem largest_x_by_equation : ∃ x : ℚ, 
  (∀ y : ℚ, 6 * (12 * y^2 + 12 * y + 11) = y * (12 * y - 44) → y ≤ x) 
  ∧ 6 * (12 * x^2 + 12 * x + 11) = x * (12 * x - 44) 
  ∧ x = -1 := 
sorry

end largest_x_by_equation_l269_269481


namespace twice_son_plus_father_is_70_l269_269243

section
variable {s f : ℕ}

-- Conditions
def son_age : ℕ := 15
def father_age : ℕ := 40

-- Statement to prove
theorem twice_son_plus_father_is_70 : (2 * son_age + father_age) = 70 :=
by
  sorry
end

end twice_son_plus_father_is_70_l269_269243


namespace false_p_and_q_l269_269215

variable {a : ℝ} 

def p (a : ℝ) := 3 * a / 2 ≤ 1
def q (a : ℝ) := 0 < 2 * a - 1 ∧ 2 * a - 1 < 1

theorem false_p_and_q (a : ℝ) :
  ¬ (p a ∧ q a) ↔ (a ≤ (1 : ℝ) / 2 ∨ a > (2 : ℝ) / 3) :=
by
  sorry

end false_p_and_q_l269_269215


namespace largest_digit_B_divisible_by_4_l269_269643

theorem largest_digit_B_divisible_by_4 :
  ∃ B : ℕ, B = 9 ∧ ∀ k : ℕ, (k ≤ 9 → (∃ n : ℕ, 4 * n = 10 * B + 792 % 100)) :=
by
  sorry

end largest_digit_B_divisible_by_4_l269_269643


namespace abs_equality_holds_if_interval_l269_269193

noncomputable def quadratic_abs_equality (x : ℝ) : Prop :=
  |x^2 - 8 * x + 12| = x^2 - 8 * x + 12

theorem abs_equality_holds_if_interval (x : ℝ) :
  quadratic_abs_equality x ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end abs_equality_holds_if_interval_l269_269193


namespace fraction_identity_l269_269601

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l269_269601


namespace union_complement_eq_universal_l269_269522

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 5}

-- The proof problem
theorem union_complement_eq_universal :
  U = A ∪ (U \ B) :=
by
  sorry

end union_complement_eq_universal_l269_269522


namespace solve_for_x_l269_269948

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem solve_for_x : {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by
  sorry

end solve_for_x_l269_269948


namespace hyperbola_focus_coordinates_l269_269766

theorem hyperbola_focus_coordinates:
  ∀ (x y : ℝ), 
    (x - 5)^2 / 7^2 - (y - 12)^2 / 10^2 = 1 → 
      ∃ (c : ℝ), c = 5 + Real.sqrt 149 ∧ (x, y) = (c, 12) :=
by
  intros x y h
  -- prove the coordinates of the focus with the larger x-coordinate are (5 + sqrt 149, 12)
  sorry

end hyperbola_focus_coordinates_l269_269766


namespace iron_balls_molded_l269_269395

noncomputable def volume_of_iron_bar (l w h : ℝ) : ℝ :=
  l * w * h

theorem iron_balls_molded (l w h n : ℝ) (volume_of_ball : ℝ) 
  (h_l : l = 12) (h_w : w = 8) (h_h : h = 6) (h_n : n = 10) (h_ball_volume : volume_of_ball = 8) :
  (n * volume_of_iron_bar l w h) / volume_of_ball = 720 :=
by 
  rw [h_l, h_w, h_h, h_n, h_ball_volume]
  rw [volume_of_iron_bar]
  sorry

end iron_balls_molded_l269_269395


namespace combined_mpg_correct_l269_269830

def ray_mpg := 30
def tom_mpg := 15
def alice_mpg := 60
def distance_each := 120

-- Total gasoline consumption
def ray_gallons := distance_each / ray_mpg
def tom_gallons := distance_each / tom_mpg
def alice_gallons := distance_each / alice_mpg

def total_gallons := ray_gallons + tom_gallons + alice_gallons
def total_distance := 3 * distance_each

def combined_mpg := total_distance / total_gallons

theorem combined_mpg_correct :
  combined_mpg = 26 :=
by
  -- All the necessary calculations would go here.
  sorry

end combined_mpg_correct_l269_269830


namespace son_time_to_complete_job_l269_269746

theorem son_time_to_complete_job (M S : ℝ) (hM : M = 1 / 5) (hMS : M + S = 1 / 4) : S = 1 / 20 → 1 / S = 20 :=
by
  sorry

end son_time_to_complete_job_l269_269746


namespace minimizes_G_at_7_over_12_l269_269920

def F (p q : ℝ) : ℝ :=
  -2 * p * q + 3 * p * (1 - q) + 3 * (1 - p) * q - 4 * (1 - p) * (1 - q)

noncomputable def G (p : ℝ) : ℝ :=
  max (3 * p - 4) (3 - 5 * p)

theorem minimizes_G_at_7_over_12 :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → (∀ p, G p ≥ G (7 / 12)) ↔ p = 7 / 12 :=
by
  sorry

end minimizes_G_at_7_over_12_l269_269920


namespace fraction_computation_l269_269584

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l269_269584


namespace find_original_comic_books_l269_269093

def comic_books (X : ℕ) : Prop :=
  X / 2 + 6 = 13

theorem find_original_comic_books (X : ℕ) (h : comic_books X) : X = 14 :=
by
  sorry

end find_original_comic_books_l269_269093


namespace min_tiles_needed_l269_269375

-- Definitions for the problem
def tile_width : ℕ := 3
def tile_height : ℕ := 4

def region_width_ft : ℕ := 2
def region_height_ft : ℕ := 5

def inches_in_foot : ℕ := 12

-- Conversion
def region_width_in := region_width_ft * inches_in_foot
def region_height_in := region_height_ft * inches_in_foot

-- Calculations
def region_area := region_width_in * region_height_in
def tile_area := tile_width * tile_height

-- Theorem statement
theorem min_tiles_needed : region_area / tile_area = 120 := 
  sorry

end min_tiles_needed_l269_269375


namespace olivia_total_pieces_l269_269081

def initial_pieces_folder1 : ℕ := 152
def initial_pieces_folder2 : ℕ := 98
def used_pieces_folder1 : ℕ := 78
def used_pieces_folder2 : ℕ := 42

def remaining_pieces_folder1 : ℕ :=
  initial_pieces_folder1 - used_pieces_folder1

def remaining_pieces_folder2 : ℕ :=
  initial_pieces_folder2 - used_pieces_folder2

def total_remaining_pieces : ℕ :=
  remaining_pieces_folder1 + remaining_pieces_folder2

theorem olivia_total_pieces : total_remaining_pieces = 130 :=
  by sorry

end olivia_total_pieces_l269_269081


namespace find_OH_squared_l269_269813

theorem find_OH_squared (R a b c : ℝ) (hR : R = 10) (hsum : a^2 + b^2 + c^2 = 50) : 
  9 * R^2 - (a^2 + b^2 + c^2) = 850 :=
by
  sorry

end find_OH_squared_l269_269813


namespace garden_length_l269_269863

theorem garden_length (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 240) : l = 80 :=
by
  sorry

end garden_length_l269_269863


namespace equidistant_trajectory_l269_269697

theorem equidistant_trajectory (x y : ℝ) (h : abs x = abs y) : y^2 = x^2 :=
by
  sorry

end equidistant_trajectory_l269_269697


namespace measure_of_angle_Q_l269_269918

variables (R S T U Q : ℝ)
variables (angle_R angle_S angle_T angle_U : ℝ)

-- Given conditions
def sum_of_angles_in_pentagon : ℝ := 540
def angle_measure_R : ℝ := 120
def angle_measure_S : ℝ := 94
def angle_measure_T : ℝ := 115
def angle_measure_U : ℝ := 101

theorem measure_of_angle_Q :
  angle_R = angle_measure_R →
  angle_S = angle_measure_S →
  angle_T = angle_measure_T →
  angle_U = angle_measure_U →
  (angle_R + angle_S + angle_T + angle_U + Q = sum_of_angles_in_pentagon) →
  Q = 110 :=
by { sorry }

end measure_of_angle_Q_l269_269918


namespace cube_problem_l269_269517

-- Define the conditions
def cube_volume (s : ℝ) : ℝ := s^3
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

theorem cube_problem (x : ℝ) (s : ℝ) :
  cube_volume s = 8 * x ∧ cube_surface_area s = 4 * x → x = 216 :=
by
  intro h
  sorry

end cube_problem_l269_269517


namespace avg_prime_factors_of_multiples_of_10_l269_269612

theorem avg_prime_factors_of_multiples_of_10 : 
  (2 + 5) / 2 = 3.5 :=
by
  -- The prime factors of 10 are 2 and 5.
  -- Therefore, the average of these prime factors is (2 + 5) / 2.
  sorry

end avg_prime_factors_of_multiples_of_10_l269_269612


namespace number_of_books_l269_269957

theorem number_of_books (Maddie Luisa Amy Noah : ℕ)
  (H1 : Maddie = 15)
  (H2 : Luisa = 18)
  (H3 : Amy + Luisa = Maddie + 9)
  (H4 : Noah = Amy / 3)
  : Amy + Noah = 8 :=
sorry

end number_of_books_l269_269957


namespace cat_toy_cost_l269_269645

-- Define the conditions
def cost_of_cage : ℝ := 11.73
def total_cost_of_purchases : ℝ := 21.95

-- Define the proof statement
theorem cat_toy_cost : (total_cost_of_purchases - cost_of_cage) = 10.22 := by
  sorry

end cat_toy_cost_l269_269645


namespace linear_function_diff_l269_269259

noncomputable def g : ℝ → ℝ := sorry

theorem linear_function_diff (h_linear : ∀ x y z w : ℝ, (g y - g x) / (y - x) = (g w - g z) / (w - z))
                            (h_condition : g 8 - g 1 = 21) : 
  g 16 - g 1 = 45 := 
by 
  sorry

end linear_function_diff_l269_269259


namespace solution1_solution2_l269_269342

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l269_269342


namespace homework_time_decrease_l269_269315

theorem homework_time_decrease (x : ℝ) :
  let T_initial := 100
  let T_final := 70
  T_initial * (1 - x) * (1 - x) = T_final :=
by
  sorry

end homework_time_decrease_l269_269315


namespace cos_of_angle_C_l269_269638

theorem cos_of_angle_C (A B C : ℝ)
  (h1 : Real.sin (π - A) = 3 / 5)
  (h2 : Real.tan (π + B) = 12 / 5)
  (h_cos_A : Real.cos A = 4 / 5) :
  Real.cos C = 16 / 65 :=
sorry

end cos_of_angle_C_l269_269638


namespace sufficient_but_not_necessary_condition_l269_269793

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a = 2 → |a| = 2) ∧ (|a| = 2 → a = 2 ∨ a = -2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l269_269793


namespace greatest_x_inequality_l269_269896

theorem greatest_x_inequality :
  ∃ x, -x^2 + 11 * x - 28 = 0 ∧ (∀ y, -y^2 + 11 * y - 28 ≥ 0 → y ≤ x) ∧ x = 7 :=
sorry

end greatest_x_inequality_l269_269896


namespace vertical_asymptotes_count_l269_269919

theorem vertical_asymptotes_count : 
  let f (x : ℝ) := (x - 2) / (x^2 + 4*x - 5) 
  ∃! c : ℕ, c = 2 :=
by
  sorry

end vertical_asymptotes_count_l269_269919


namespace vertex_of_parabola_l269_269973

theorem vertex_of_parabola :
  (∃ (h k : ℤ), ∀ (x : ℝ), y = (x - h)^2 + k) → (h = 2 ∧ k = -3) := by
  sorry

end vertex_of_parabola_l269_269973


namespace sarah_score_l269_269285

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end sarah_score_l269_269285


namespace each_piglet_ate_9_straws_l269_269720

theorem each_piglet_ate_9_straws (t : ℕ) (h_t : t = 300)
                                 (p : ℕ) (h_p : p = 20)
                                 (f : ℕ) (h_f : f = (3 * t / 5)) :
  f / p = 9 :=
by
  sorry

end each_piglet_ate_9_straws_l269_269720


namespace number_of_men_in_group_l269_269972

-- Define the conditions
variable (n : ℕ) -- number of men in the group
variable (A : ℝ) -- original average age of the group
variable (increase_in_years : ℝ := 2) -- the increase in the average age
variable (ages_before_replacement : ℝ := 21 + 23) -- total age of the men replaced
variable (ages_after_replacement : ℝ := 2 * 37) -- total age of the new men

-- Define the theorem using the conditions
theorem number_of_men_in_group 
  (h1 : n * increase_in_years = ages_after_replacement - ages_before_replacement) :
  n = 15 :=
sorry

end number_of_men_in_group_l269_269972


namespace angle_B_measure_triangle_area_l269_269929

noncomputable def triangle (A B C : ℝ) : Type := sorry

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Given conditions:
axiom eq1 : b * Real.cos C = (2 * a - c) * Real.cos B

-- Part 1: Prove the measure of angle B
theorem angle_B_measure : B = Real.pi / 3 :=
by
  have b_cos_C := eq1
  sorry

-- Part 2: Given additional conditions and find the area
variable (b_value : ℝ := Real.sqrt 7)
variable (sum_ac : ℝ := 4)

theorem triangle_area : (1 / 2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) :=
by
  have b_value_def := b_value
  have sum_ac_def := sum_ac
  sorry

end angle_B_measure_triangle_area_l269_269929


namespace parents_can_catch_ka_liang_l269_269450

-- Definitions according to the problem statement.
-- Define the condition of the roads and the speed of the participants.
def grid_with_roads : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  -- 4 roads forming the sides of a square with side length a
  True ∧
  -- 2 roads connecting the midpoints of opposite sides of the square
  True

def ka_liang_speed : ℝ := 2

def parent_speed : ℝ := 1

-- Condition that Ka Liang, father, and mother can see each other
def mutual_visibility (a b : ℝ) : Prop := True

-- The main proposition
theorem parents_can_catch_ka_liang (a b : ℝ) (hgrid : grid_with_roads)
    (hspeed : ka_liang_speed = 2 * parent_speed) (hvis : mutual_visibility a b) :
  True := 
sorry

end parents_can_catch_ka_liang_l269_269450


namespace number_of_sheets_is_9_l269_269380

-- Define the conditions in Lean
variable (n : ℕ) -- Total number of pages

-- The stack is folded and renumbered
axiom folded_and_renumbered : True

-- The sum of numbers on one of the sheets is 74
axiom sum_sheet_is_74 : 2 * n + 2 = 74

-- The number of sheets is the number of pages divided by 4
def number_of_sheets (n : ℕ) : ℕ := n / 4

-- Prove that the number of sheets is 9 given the conditions
theorem number_of_sheets_is_9 : number_of_sheets 36 = 9 :=
by
  sorry

end number_of_sheets_is_9_l269_269380


namespace find_x_l269_269867

variable (BrandA_millet : ℝ) (Mix_millet : ℝ) (Mix_ratio_A : ℝ) (Mix_ratio_B : ℝ)

axiom BrandA_contains_60_percent_millet : BrandA_millet = 0.60
axiom Mix_contains_50_percent_millet : Mix_millet = 0.50
axiom Mix_composition : Mix_ratio_A = 0.60 ∧ Mix_ratio_B = 0.40

theorem find_x (x : ℝ) :
  Mix_ratio_A * BrandA_millet + Mix_ratio_B * x = Mix_millet →
  x = 0.35 :=
by
  sorry

end find_x_l269_269867


namespace maximum_volume_of_pyramid_l269_269845

theorem maximum_volume_of_pyramid (a b : ℝ) (hb : b > 0) (ha : a > 0):
  ∃ V_max : ℝ, V_max = (a * (4 * b ^ 2 - a ^ 2)) / 12 := 
sorry

end maximum_volume_of_pyramid_l269_269845


namespace square_root_domain_l269_269632

theorem square_root_domain (x : ℝ) : (∃ y, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end square_root_domain_l269_269632


namespace prob_A_wins_4_consecutive_prob_fifth_game_needed_prob_C_ultimate_winner_l269_269984

-- Conditions for the game settings
def initial_conditions (a b c : ℕ) : Prop :=
  a = 0 ∧ b = 0 ∧ c = 0

-- Probability of a player winning any game
def win_probability : ℚ := 1 / 2 

-- Probability calculation for A winning four consecutive games
theorem prob_A_wins_4_consecutive :
  win_probability ^ 4 = 1 / 16 :=
by
  sorry

-- Probability calculation for needing a fifth game to be played
theorem prob_fifth_game_needed :
  1 - 4 * (win_probability ^ 4) = 3 / 4 :=
by
  sorry

-- Probability calculation for C being the ultimate winner
theorem prob_C_ultimate_winner :
  1 - 2 * (9 / 32) = 7 / 16 :=
by
  sorry

end prob_A_wins_4_consecutive_prob_fifth_game_needed_prob_C_ultimate_winner_l269_269984


namespace lisa_interest_earned_l269_269101

/-- Lisa's interest earned after three years from Bank of Springfield's Super High Yield savings account -/
theorem lisa_interest_earned :
  let P := 2000
  let r := 0.02
  let n := 3
  let A := P * (1 + r)^n
  A - P = 122 := by
  sorry

end lisa_interest_earned_l269_269101


namespace directly_above_156_is_133_l269_269726

def row_numbers (k : ℕ) : ℕ := 2 * k - 1

def total_numbers_up_to_row (k : ℕ) : ℕ := k * k

def find_row (n : ℕ) : ℕ :=
  Nat.sqrt (n + 1)

def position_in_row (n k : ℕ) : ℕ :=
  n - (total_numbers_up_to_row (k - 1)) + 1

def number_directly_above (n : ℕ) : ℕ :=
  let k := find_row n
  let pos := position_in_row n k
  (total_numbers_up_to_row (k - 1) - row_numbers (k - 1)) + pos + 1

theorem directly_above_156_is_133 : number_directly_above 156 = 133 := 
  by
  sorry

end directly_above_156_is_133_l269_269726


namespace time_to_fill_pond_l269_269463

noncomputable def pond_capacity : ℝ := 200
noncomputable def normal_pump_rate : ℝ := 6
noncomputable def restriction_factor : ℝ := 2 / 3
noncomputable def restricted_pump_rate : ℝ := restriction_factor * normal_pump_rate

theorem time_to_fill_pond : pond_capacity / restricted_pump_rate = 50 := 
by 
  -- This is where the proof would go
  sorry

end time_to_fill_pond_l269_269463


namespace marker_cost_is_13_l269_269448

theorem marker_cost_is_13 :
  ∃ s m c : ℕ, (s > 20) ∧ (m ≥ 4) ∧ (c > m) ∧ (s * c * m = 3185) ∧ (c = 13) :=
by
  sorry

end marker_cost_is_13_l269_269448


namespace fraction_computation_l269_269585

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l269_269585


namespace correct_propositions_l269_269908

-- Definitions of propositions
def prop1 (f : ℝ → ℝ) : Prop :=
  f (-2) ≠ f (2) → ∀ x : ℝ, f (-x) ≠ f (x)

def prop2 : Prop :=
  ∀ n : ℕ, n = 0 ∨ n = 1 → (∀ x : ℝ, x ≠ 0 → x ^ n ≠ 0)

def prop3 : Prop :=
  ∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0) → (a * b ≠ 0) ∧ (a * b = 0 → a = 0 ∨ b = 0)

def prop4 (a b c d : ℝ) : Prop :=
  ∀ x : ℝ, ∃ k : ℝ, k = d → (3 * a * x ^ 2 + 2 * b * x + c ≠ 0 ∧ b ^ 2 - 3 * a * c ≥ 0)

-- Final proof statement
theorem correct_propositions (f : ℝ → ℝ) (a b c d : ℝ) :
  prop1 f ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 a b c d :=
sorry

end correct_propositions_l269_269908


namespace proof_problem_l269_269223

variable {R : Type} [LinearOrderedField R]

def is_increasing (f : R → R) : Prop :=
  ∀ x y : R, x < y → f x < f y

theorem proof_problem (f : R → R) (a b : R) 
  (inc_f : is_increasing f) 
  (h : f a + f b > f (-a) + f (-b)) : 
  a + b > 0 := 
by
  sorry

end proof_problem_l269_269223


namespace polynomial_expansion_l269_269197

theorem polynomial_expansion (x : ℝ) :
  (5 * x^2 + 3 * x - 7) * (4 * x^3) = 20 * x^5 + 12 * x^4 - 28 * x^3 :=
by 
  sorry

end polynomial_expansion_l269_269197


namespace fraction_of_cats_l269_269098

theorem fraction_of_cats (C D : ℕ) 
  (h1 : C + D = 300)
  (h2 : 4 * D = 400) : 
  (C : ℚ) / (C + D) = 2 / 3 :=
by
  sorry

end fraction_of_cats_l269_269098


namespace max_val_proof_l269_269472

noncomputable def max_val (p q r x y z : ℝ) : ℝ :=
  1 / (p + q) + 1 / (p + r) + 1 / (q + r) + 1 / (x + y) + 1 / (x + z) + 1 / (y + z)

theorem max_val_proof {p q r x y z : ℝ}
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum_pqr : p + q + r = 2) (h_sum_xyz : x + y + z = 1) :
  max_val p q r x y z = 27 / 4 :=
sorry

end max_val_proof_l269_269472


namespace a_received_share_l269_269992

variables (I_a I_b I_c b_share total_investment total_profit a_share : ℕ)
  (h1 : I_a = 11000)
  (h2 : I_b = 15000)
  (h3 : I_c = 23000)
  (h4 : b_share = 3315)
  (h5 : total_investment = I_a + I_b + I_c)
  (h6 : total_profit = b_share * total_investment / I_b)
  (h7 : a_share = I_a * total_profit / total_investment)

theorem a_received_share : a_share = 2662 := by
  sorry

end a_received_share_l269_269992


namespace inequality_problem_l269_269656

variable (a b c : ℝ)

theorem inequality_problem (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
sorry

end inequality_problem_l269_269656


namespace sum_x_coordinates_intersection_mod_9_l269_269958

theorem sum_x_coordinates_intersection_mod_9 :
  ∃ x y : ℤ, (y ≡ 3 * x + 4 [ZMOD 9]) ∧ (y ≡ 7 * x + 2 [ZMOD 9]) ∧ x ≡ 5 [ZMOD 9] := sorry

end sum_x_coordinates_intersection_mod_9_l269_269958


namespace pencil_groups_l269_269114

theorem pencil_groups (total_pencils number_per_group number_of_groups : ℕ) 
  (h_total: total_pencils = 25) 
  (h_group: number_per_group = 5) 
  (h_eq: total_pencils = number_per_group * number_of_groups) : 
  number_of_groups = 5 :=
by
  sorry

end pencil_groups_l269_269114


namespace minimal_overlap_facebook_instagram_l269_269639

variable (P : ℝ → Prop)
variable [Nonempty (Set.Icc 0 1)]

theorem minimal_overlap_facebook_instagram :
  ∀ (f i : ℝ), f = 0.85 → i = 0.75 → ∃ b : ℝ, 0 ≤ b ∧ b ≤ 1 ∧ b = 0.6 :=
by
  intros
  sorry

end minimal_overlap_facebook_instagram_l269_269639


namespace find_x_l269_269727

theorem find_x :
  ∃ x : ℕ, (5 * 12) / (x / 3) + 80 = 81 ∧ x = 180 :=
by
  sorry

end find_x_l269_269727


namespace proof_no_natural_solutions_l269_269961

noncomputable def no_natural_solutions : Prop :=
  ∀ x y : ℕ, y^2 ≠ x^2 + x + 1

theorem proof_no_natural_solutions : no_natural_solutions :=
by
  intros x y
  sorry

end proof_no_natural_solutions_l269_269961


namespace sum_of_Y_l269_269305

open Finset

def X : Finset ℕ := (range 600).map ⟨λ n, n + 1, λ a b, by simp⟩

def multiples (k : ℕ) (s : Finset ℕ) : Finset ℕ :=
  s.filter (λ x, x % k = 0)

def Y : Finset ℕ := (multiples 3 X) ∪ (multiples 4 X)

theorem sum_of_Y : Y.sum id = 90300 :=
by sorry

end sum_of_Y_l269_269305


namespace rachel_total_apples_l269_269964

noncomputable def totalRemainingApples (X : ℕ) : ℕ :=
  let remainingFirstFour := 10 + 40 + 15 + 22
  let remainingOtherTrees := 48 * X
  remainingFirstFour + remainingOtherTrees

theorem rachel_total_apples (X : ℕ) :
  totalRemainingApples X = 87 + 48 * X :=
by
  sorry

end rachel_total_apples_l269_269964


namespace product_of_two_numbers_l269_269500

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by sorry

end product_of_two_numbers_l269_269500


namespace odds_against_C_winning_l269_269231

theorem odds_against_C_winning :
  let P_A := 2 / 7
  let P_B := 1 / 5
  let P_C := 1 - (P_A + P_B)
  (1 - P_C) / P_C = 17 / 18 :=
by
  sorry

end odds_against_C_winning_l269_269231


namespace Trisha_walked_total_distance_l269_269269

theorem Trisha_walked_total_distance 
  (d1 d2 d3 : ℝ) (h_d1 : d1 = 0.11) (h_d2 : d2 = 0.11) (h_d3 : d3 = 0.67) :
  d1 + d2 + d3 = 0.89 :=
by sorry

end Trisha_walked_total_distance_l269_269269


namespace part_1_part_2_l269_269009

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269009


namespace sum_coordinates_eq_l269_269679

theorem sum_coordinates_eq : ∀ (A B : ℝ × ℝ), A = (0, 0) → (∃ x : ℝ, B = (x, 5)) → 
  (∀ x : ℝ, B = (x, 5) → (5 - 0) / (x - 0) = 3 / 4) → 
  let x := 20 / 3 in
  (x + 5 = 35 / 3) :=
by
  intros A B hA hB hslope
  have hx : ∃ x, B = (x, 5) := hB
  cases hx with x hx_def
  rw hx_def at hslope
  have : x = 20 / 3 := by
    rw hx_def at hslope
    field_simp at hslope
    linarith
  rw [this, hx_def]
  norm_num

end sum_coordinates_eq_l269_269679


namespace every_nat_as_diff_of_same_prime_divisors_l269_269271

-- Conditions
def prime_divisors (n : ℕ) : ℕ :=
  -- function to count the number of distinct prime divisors of n
  sorry

-- Tuple translation
theorem every_nat_as_diff_of_same_prime_divisors :
  ∀ n : ℕ, ∃ a b : ℕ, n = a - b ∧ prime_divisors a = prime_divisors b := 
by
  sorry

end every_nat_as_diff_of_same_prime_divisors_l269_269271


namespace dealer_gross_profit_l269_269153

variable (purchase_price : ℝ) (markup_rate : ℝ) (gross_profit : ℝ)

def desk_problem (purchase_price : ℝ) (markup_rate : ℝ) (gross_profit : ℝ) : Prop :=
  ∀ (S : ℝ), S = purchase_price + markup_rate * S → gross_profit = S - purchase_price

theorem dealer_gross_profit : desk_problem 150 0.5 150 :=
by 
  sorry

end dealer_gross_profit_l269_269153


namespace product_of_numbers_l269_269493

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 :=
sorry

end product_of_numbers_l269_269493


namespace part_a_part_b_l269_269048

-- Define the predicate ensuring that among any three consecutive symbols, there is at least one zero
def valid_sequence (s : List Char) : Prop :=
  ∀ (i : Nat), i + 2 < s.length → (s.get! i = '0' ∨ s.get! (i + 1) = '0' ∨ s.get! (i + 2) = '0')

-- Count the valid sequences given the number of 'X's and 'O's
noncomputable def count_valid_sequences (n_zeros n_crosses : Nat) : Nat :=
  sorry -- Implementation of the combinatorial counting

-- Part (a): n = 29
theorem part_a : count_valid_sequences 14 29 = 15 := by
  sorry

-- Part (b): n = 28
theorem part_b : count_valid_sequences 14 28 = 120 := by
  sorry

end part_a_part_b_l269_269048


namespace part_1_part_2_l269_269005

theorem part_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): abc ≤ 1 / 9 :=
sorry

theorem part_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a ^ (3 / 2) + b ^ (3 / 2) + c ^ (3 / 2) = 1): 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (abc)) :=
sorry

end part_1_part_2_l269_269005


namespace riverton_soccer_physics_l269_269398

theorem riverton_soccer_physics : 
  let total_players := 15
  let math_players := 9
  let both_subjects := 3
  let only_physics := total_players - math_players
  let physics_players := only_physics + both_subjects
  physics_players = 9 :=
by
  sorry

end riverton_soccer_physics_l269_269398


namespace part1_part2_l269_269024

variables {a b c : ℝ}

-- Define the hypotheses
def pos_numbers (h : a > 0 ∧ b > 0 ∧ c > 0) := h
def sum_powers_eq_one (h : a^1.5 + b^1.5 + c^1.5 = 1) := h

-- Theorems to prove the given statements under the given conditions
theorem part1 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  abc ≤ 1/9 :=
sorry

theorem part2 (h1 : pos_numbers (a > 0 ∧ b > 0 ∧ c > 0)) (h2 : sum_powers_eq_one (a^1.5 + b^1.5 + c^1.5 = 1)) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * Real.sqrt(abc)) :=
sorry

end part1_part2_l269_269024


namespace triangle_ABC_perimeter_l269_269458

noncomputable def triangle_perimeter (A B C D : Type) (AD BC AC AB : ℝ) : ℝ :=
  AD + BC + AC + AB

theorem triangle_ABC_perimeter (A B C D : Type) (AD BC : ℝ) (cos_BDC : ℝ) (angle_sum : ℝ) (AC : ℝ) (AB : ℝ) :
  AD = 3 → BC = 2 → cos_BDC = 13 / 20 → angle_sum = 180 → 
  (triangle_perimeter A B C D AD BC AC AB = 11) :=
by
  sorry

end triangle_ABC_perimeter_l269_269458


namespace zeroSeq_arithmetic_not_geometric_l269_269976

-- Define what it means for a sequence to be arithmetic
def isArithmeticSequence (seq : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) = seq n + d

-- Define what it means for a sequence to be geometric
def isGeometricSequence (seq : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, seq n ≠ 0 → seq (n + 1) = seq n * q

-- Define the sequence of zeros
def zeroSeq (n : ℕ) : ℝ := 0

theorem zeroSeq_arithmetic_not_geometric :
  isArithmeticSequence zeroSeq ∧ ¬ isGeometricSequence zeroSeq :=
by
  sorry

end zeroSeq_arithmetic_not_geometric_l269_269976


namespace probability_ratio_equality_l269_269194

noncomputable def probability_ratio : ℝ := 
  let A := (Nat.choose 6 2) * (Nat.choose 25 3) * (Nat.choose 22 3) *
           (Nat.choose 19 4) * (Nat.choose 15 4) *
           (Nat.choose 11 4) * (Nat.choose 7 4) / (4! : ℕ)
  let B := (Nat.choose 6 1) * (Nat.choose 25 5) * (Nat.choose 20 4) *
           (Nat.choose 16 4) * (Nat.choose 12 4) *
           (Nat.choose 8 4) * (Nat.choose 4 4)
  A / B

theorem probability_ratio_equality : probability_ratio = 8 := by
  sorry

end probability_ratio_equality_l269_269194


namespace percentage_increase_school_B_l269_269801

theorem percentage_increase_school_B (A B Q_A Q_B : ℝ) 
  (h1 : Q_A = 0.7 * A) 
  (h2 : Q_B = 1.5 * Q_A) 
  (h3 : Q_B = 0.875 * B) :
  (B - A) / A * 100 = 20 :=
by
  sorry

end percentage_increase_school_B_l269_269801


namespace jose_age_is_26_l269_269809

def Maria_age : ℕ := 14
def Jose_age (m : ℕ) : ℕ := m + 12

theorem jose_age_is_26 (m j : ℕ) (h1 : j = m + 12) (h2 : m + j = 40) : j = 26 :=
by
  sorry

end jose_age_is_26_l269_269809


namespace fewer_twos_result_100_l269_269321

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l269_269321


namespace sarah_score_l269_269288

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end sarah_score_l269_269288


namespace smallest_x_value_l269_269202

theorem smallest_x_value {x : ℝ} (h : abs (x + 4) = 15) : x = -19 :=
sorry

end smallest_x_value_l269_269202


namespace fewer_twos_to_hundred_l269_269338

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l269_269338


namespace no_integer_solution_l269_269078

open Polynomial

theorem no_integer_solution (P : Polynomial ℤ) (a b c d : ℤ)
  (h₁ : P.eval a = 2016) (h₂ : P.eval b = 2016) (h₃ : P.eval c = 2016) 
  (h₄ : P.eval d = 2016) (dist : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ¬ ∃ x : ℤ, P.eval x = 2019 :=
sorry

end no_integer_solution_l269_269078


namespace profit_no_discount_l269_269991

theorem profit_no_discount (CP SP ASP : ℝ) (discount profit : ℝ) (h1 : discount = 4 / 100) (h2 : profit = 38 / 100) (h3 : SP = CP + CP * profit) (h4 : ASP = SP - SP * discount) :
  ((SP - CP) / CP) * 100 = 38 :=
by
  sorry

end profit_no_discount_l269_269991
