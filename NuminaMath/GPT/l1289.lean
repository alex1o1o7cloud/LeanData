import Mathlib

namespace NUMINAMATH_GPT_gcd_polynomial_l1289_128913

theorem gcd_polynomial (b : ℤ) (h : 2142 ∣ b) : Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 :=
sorry

end NUMINAMATH_GPT_gcd_polynomial_l1289_128913


namespace NUMINAMATH_GPT_find_x_value_l1289_128970

theorem find_x_value :
  ∃ (x : ℤ), ∀ (y z w : ℤ), (x = 2 * y + 4) → (y = z + 5) → (z = 2 * w + 3) → (w = 50) → x = 220 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l1289_128970


namespace NUMINAMATH_GPT_log_lt_x_squared_for_x_gt_zero_l1289_128925

theorem log_lt_x_squared_for_x_gt_zero (x : ℝ) (h : x > 0) : Real.log (1 + x) < x^2 :=
sorry

end NUMINAMATH_GPT_log_lt_x_squared_for_x_gt_zero_l1289_128925


namespace NUMINAMATH_GPT_units_digit_2_pow_2015_l1289_128906

theorem units_digit_2_pow_2015 : ∃ u : ℕ, (2 ^ 2015 % 10) = u ∧ u = 8 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_2_pow_2015_l1289_128906


namespace NUMINAMATH_GPT_miles_total_instruments_l1289_128963

-- Definitions based on the conditions
def fingers : ℕ := 10
def hands : ℕ := 2
def heads : ℕ := 1
def trumpets : ℕ := fingers - 3
def guitars : ℕ := hands + 2
def trombones : ℕ := heads + 2
def french_horns : ℕ := guitars - 1
def total_instruments : ℕ := trumpets + guitars + trombones + french_horns

-- Main theorem
theorem miles_total_instruments : total_instruments = 17 := 
sorry

end NUMINAMATH_GPT_miles_total_instruments_l1289_128963


namespace NUMINAMATH_GPT_average_weight_of_remaining_students_l1289_128971

theorem average_weight_of_remaining_students
  (M F M' F' : ℝ) (A A' : ℝ)
  (h1 : M + F = 60 * A)
  (h2 : M' + F' = 59 * A')
  (h3 : A' = A + 0.2)
  (h4 : M' = M - 45):
  A' = 57 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_remaining_students_l1289_128971


namespace NUMINAMATH_GPT_original_grain_correct_l1289_128948

-- Define the initial quantities
def grain_spilled : ℕ := 49952
def grain_remaining : ℕ := 918

-- Define the original amount of grain expected
def original_grain : ℕ := 50870

-- Prove that the original amount of grain was correct
theorem original_grain_correct : grain_spilled + grain_remaining = original_grain := 
by
  sorry

end NUMINAMATH_GPT_original_grain_correct_l1289_128948


namespace NUMINAMATH_GPT_find_certain_number_l1289_128912

theorem find_certain_number (n : ℕ)
  (h1 : 3153 + 3 = 3156)
  (h2 : 3156 % 9 = 0)
  (h3 : 3156 % 70 = 0)
  (h4 : 3156 % 25 = 0) :
  3156 % 37 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1289_128912


namespace NUMINAMATH_GPT_composite_10201_base_n_composite_10101_base_n_l1289_128920

-- 1. Prove that 10201_n is composite given n > 2
theorem composite_10201_base_n (n : ℕ) (h : n > 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n^4 + 2*n^2 + 1 := 
sorry

-- 2. Prove that 10101_n is composite given n > 2.
theorem composite_10101_base_n (n : ℕ) (h : n > 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n^4 + n^2 + 1 := 
sorry

end NUMINAMATH_GPT_composite_10201_base_n_composite_10101_base_n_l1289_128920


namespace NUMINAMATH_GPT_man_work_rate_l1289_128937

theorem man_work_rate (W : ℝ) (M S : ℝ)
  (h1 : (M + S) * 3 = W)
  (h2 : S * 5.25 = W) :
  M * 7 = W :=
by 
-- The proof steps will be filled in here.
sorry

end NUMINAMATH_GPT_man_work_rate_l1289_128937


namespace NUMINAMATH_GPT_meetings_percentage_l1289_128929

-- Define all the conditions given in the problem
def first_meeting := 60 -- duration of first meeting in minutes
def second_meeting := 2 * first_meeting -- duration of second meeting in minutes
def third_meeting := first_meeting / 2 -- duration of third meeting in minutes
def total_meeting_time := first_meeting + second_meeting + third_meeting -- total meeting time
def total_workday := 10 * 60 -- total workday time in minutes

-- Statement to prove that the percentage of workday spent in meetings is 35%
def percent_meetings : Prop := (total_meeting_time / total_workday) * 100 = 35

theorem meetings_percentage :
  percent_meetings :=
by
  sorry

end NUMINAMATH_GPT_meetings_percentage_l1289_128929


namespace NUMINAMATH_GPT_total_people_museum_l1289_128977

-- Conditions
def first_bus_people : ℕ := 12
def second_bus_people := 2 * first_bus_people
def third_bus_people := second_bus_people - 6
def fourth_bus_people := first_bus_people + 9

-- Question to prove
theorem total_people_museum : first_bus_people + second_bus_people + third_bus_people + fourth_bus_people = 75 :=
by
  -- The proof is skipped but required to complete the theorem
  sorry

end NUMINAMATH_GPT_total_people_museum_l1289_128977


namespace NUMINAMATH_GPT_jezebel_total_flower_cost_l1289_128982

theorem jezebel_total_flower_cost :
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  (red_rose_count * red_rose_cost + sunflower_count * sunflower_cost = 45) :=
by
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  sorry

end NUMINAMATH_GPT_jezebel_total_flower_cost_l1289_128982


namespace NUMINAMATH_GPT_total_money_spent_l1289_128991

noncomputable def total_expenditure (A : ℝ) : ℝ :=
  let person1_8_expenditure := 8 * 12
  let person9_expenditure := A + 8
  person1_8_expenditure + person9_expenditure

theorem total_money_spent :
  (∃ A : ℝ, total_expenditure A = 9 * A ∧ A = 13) →
  total_expenditure 13 = 117 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_total_money_spent_l1289_128991


namespace NUMINAMATH_GPT_tangerines_in_one_box_l1289_128916

theorem tangerines_in_one_box (total_tangerines boxes remaining_tangerines tangerines_per_box : ℕ) 
  (h1 : total_tangerines = 29)
  (h2 : boxes = 8)
  (h3 : remaining_tangerines = 5)
  (h4 : total_tangerines - remaining_tangerines = boxes * tangerines_per_box) :
  tangerines_per_box = 3 :=
by 
  sorry

end NUMINAMATH_GPT_tangerines_in_one_box_l1289_128916


namespace NUMINAMATH_GPT_largest_integral_value_l1289_128911

theorem largest_integral_value (y : ℤ) (h1 : 0 < y) (h2 : (1 : ℚ)/4 < y / 7) (h3 : y / 7 < 7 / 11) : y = 4 :=
sorry

end NUMINAMATH_GPT_largest_integral_value_l1289_128911


namespace NUMINAMATH_GPT_tangent_lines_through_origin_l1289_128980

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

variable (a : ℝ)

theorem tangent_lines_through_origin 
  (h1 : ∃ m1 m2 : ℝ, m1 ≠ m2 ∧ (f a (-m1) + f a (m1 + 2)) / 2 = f a 1) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (f a t1 * (1 / t1) = f a 0) ∧ (f a t2 * (1 / t2) = f a 0) := 
sorry

end NUMINAMATH_GPT_tangent_lines_through_origin_l1289_128980


namespace NUMINAMATH_GPT_mary_can_keep_warm_l1289_128943

theorem mary_can_keep_warm :
  let chairs := 18
  let chairs_sticks := 6
  let tables := 6
  let tables_sticks := 9
  let stools := 4
  let stools_sticks := 2
  let sticks_per_hour := 5
  let total_sticks := (chairs * chairs_sticks) + (tables * tables_sticks) + (stools * stools_sticks)
  let hours := total_sticks / sticks_per_hour
  hours = 34 := by
{
  sorry
}

end NUMINAMATH_GPT_mary_can_keep_warm_l1289_128943


namespace NUMINAMATH_GPT_total_legs_is_26_l1289_128944

-- Define the number of puppies and chicks
def number_of_puppies : Nat := 3
def number_of_chicks : Nat := 7

-- Define the number of legs per puppy and per chick
def legs_per_puppy : Nat := 4
def legs_per_chick : Nat := 2

-- Calculate the total number of legs
def total_legs := (number_of_puppies * legs_per_puppy) + (number_of_chicks * legs_per_chick)

-- Prove that the total number of legs is 26
theorem total_legs_is_26 : total_legs = 26 := by
  sorry

end NUMINAMATH_GPT_total_legs_is_26_l1289_128944


namespace NUMINAMATH_GPT_longer_piece_length_is_20_l1289_128936

-- Define the rope length
def ropeLength : ℕ := 35

-- Define the ratio of the two pieces
def ratioA : ℕ := 3
def ratioB : ℕ := 4
def totalRatio : ℕ := ratioA + ratioB

-- Define the length of each part
def partLength : ℕ := ropeLength / totalRatio

-- Define the length of the longer piece
def longerPieceLength : ℕ := ratioB * partLength

-- Theorem to prove that the length of the longer piece is 20 inches
theorem longer_piece_length_is_20 : longerPieceLength = 20 := by 
  sorry

end NUMINAMATH_GPT_longer_piece_length_is_20_l1289_128936


namespace NUMINAMATH_GPT_slope_angle_vertical_line_l1289_128999

theorem slope_angle_vertical_line : 
  ∀ α : ℝ, (∀ x y : ℝ, x = 1 → y = α) → α = Real.pi / 2 := 
by 
  sorry

end NUMINAMATH_GPT_slope_angle_vertical_line_l1289_128999


namespace NUMINAMATH_GPT_intersection_A_B_l1289_128990

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x < 4} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1289_128990


namespace NUMINAMATH_GPT_find_pairs_eq_l1289_128942

theorem find_pairs_eq : 
  { (m, n) : ℕ × ℕ | 0 < m ∧ 0 < n ∧ m ^ 2 + 2 * n ^ 2 = 3 * (m + 2 * n) } = {(3, 3), (4, 2)} :=
by sorry

end NUMINAMATH_GPT_find_pairs_eq_l1289_128942


namespace NUMINAMATH_GPT_train_stop_time_per_hour_l1289_128917

theorem train_stop_time_per_hour
    (v1 : ℕ) (v2 : ℕ)
    (h1 : v1 = 45)
    (h2 : v2 = 33) : ∃ (t : ℕ), t = 16 := by
  -- including the proof steps here is unnecessary, so we use sorry
  sorry

end NUMINAMATH_GPT_train_stop_time_per_hour_l1289_128917


namespace NUMINAMATH_GPT_rectangle_width_of_square_l1289_128989

theorem rectangle_width_of_square (side_length_square : ℝ) (length_rectangle : ℝ) (width_rectangle : ℝ)
  (h1 : side_length_square = 3) (h2 : length_rectangle = 3)
  (h3 : (side_length_square ^ 2) = length_rectangle * width_rectangle) : width_rectangle = 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_of_square_l1289_128989


namespace NUMINAMATH_GPT_largest_four_digit_number_divisible_by_4_with_digit_sum_20_l1289_128997

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def digit_sum_is_20 (n : ℕ) : Prop :=
  (n / 1000) + ((n % 1000) / 100) + ((n % 100) / 10) + (n % 10) = 20

theorem largest_four_digit_number_divisible_by_4_with_digit_sum_20 :
  ∃ n : ℕ, is_four_digit n ∧ is_divisible_by_4 n ∧ digit_sum_is_20 n ∧ ∀ m : ℕ, is_four_digit m ∧ is_divisible_by_4 m ∧ digit_sum_is_20 m → m ≤ n :=
  sorry

end NUMINAMATH_GPT_largest_four_digit_number_divisible_by_4_with_digit_sum_20_l1289_128997


namespace NUMINAMATH_GPT_incorrect_statement_D_l1289_128969

theorem incorrect_statement_D :
  ¬ (abs (-1) - abs 1 = 2) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_D_l1289_128969


namespace NUMINAMATH_GPT_inhabitant_eq_resident_l1289_128953

-- Definitions
def inhabitant : Type := String
def resident : Type := String

-- The equivalence theorem
theorem inhabitant_eq_resident :
  ∀ (x : inhabitant), x = "resident" :=
by
  sorry

end NUMINAMATH_GPT_inhabitant_eq_resident_l1289_128953


namespace NUMINAMATH_GPT_points_on_line_l1289_128968

theorem points_on_line (y1 y2 : ℝ) 
  (hA : y1 = - (1 / 2 : ℝ) * 1 - 1) 
  (hB : y2 = - (1 / 2 : ℝ) * 3 - 1) :
  y1 > y2 := 
by
  sorry

end NUMINAMATH_GPT_points_on_line_l1289_128968


namespace NUMINAMATH_GPT_general_formula_for_an_l1289_128972

-- Definitions for the first few terms of the sequence
def a1 : ℚ := 1 / 7
def a2 : ℚ := 3 / 77
def a3 : ℚ := 5 / 777

-- The sequence definition as per the identified pattern
def a_n (n : ℕ) : ℚ := (18 * n - 9) / (7 * (10^n - 1))

-- The theorem to establish that the sequence definition for general n holds given the initial terms 
theorem general_formula_for_an {n : ℕ} :
  (n = 1 → a_n n = a1) ∧
  (n = 2 → a_n n = a2) ∧ 
  (n = 3 → a_n n = a3) ∧ 
  (∀ n > 3, a_n n = (18 * n - 9) / (7 * (10^n - 1))) := 
by
  sorry

end NUMINAMATH_GPT_general_formula_for_an_l1289_128972


namespace NUMINAMATH_GPT_automobile_travel_distance_l1289_128947

theorem automobile_travel_distance (b s : ℝ) (h1 : s > 0) :
  let rate := (b / 8) / s  -- rate in meters per second
  let rate_km_per_min := rate * (1 / 1000) * 60  -- convert to kilometers per minute
  let time := 5  -- time in minutes
  rate_km_per_min * time = 3 * b / 80 / s := sorry

end NUMINAMATH_GPT_automobile_travel_distance_l1289_128947


namespace NUMINAMATH_GPT_at_least_one_genuine_l1289_128994

/-- Given 12 products, of which 10 are genuine and 2 are defective.
    If 3 products are randomly selected, then at least one of the selected products is a genuine product. -/
theorem at_least_one_genuine : 
  ∀ (products : Fin 12 → Prop), 
  (∃ n₁ n₂ : Fin 12, (n₁ ≠ n₂) ∧ 
                   (products n₁ = true) ∧ 
                   (products n₂ = true) ∧ 
                   (∃ n₁' n₂' : Fin 12, (n₁ ≠ n₁' ∧ n₂ ≠ n₂') ∧
                                         products n₁' = products n₂' = true ∧
                                         ∀ j : Fin 3, products j = true)) → 
  (∃ m : Fin 3, products m = true) :=
sorry

end NUMINAMATH_GPT_at_least_one_genuine_l1289_128994


namespace NUMINAMATH_GPT_charlie_has_32_cards_l1289_128985

variable (Chris_cards Charlie_cards : ℕ)

def chris_has_18_cards : Chris_cards = 18 := sorry
def chris_has_14_fewer_cards_than_charlie : Chris_cards + 14 = Charlie_cards := sorry

theorem charlie_has_32_cards (h18 : Chris_cards = 18) (h14 : Chris_cards + 14 = Charlie_cards) : Charlie_cards = 32 := 
sorry

end NUMINAMATH_GPT_charlie_has_32_cards_l1289_128985


namespace NUMINAMATH_GPT_solve_inequality_system_l1289_128903

theorem solve_inequality_system (x : ℝ) :
  (5 * x - 1 > 3 * (x + 1)) ∧ (x - 1 ≤ 7 - x) ↔ (2 < x ∧ x ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l1289_128903


namespace NUMINAMATH_GPT_deposit_is_3000_l1289_128988

-- Define the constants
def cash_price : ℝ := 8000
def monthly_installment : ℝ := 300
def number_of_installments : ℕ := 30
def savings_by_paying_cash : ℝ := 4000

-- Define the total installment payments
def total_installment_payments : ℝ := number_of_installments * monthly_installment

-- Define the total price paid, which includes the deposit and installments
def total_paid : ℝ := cash_price + savings_by_paying_cash

-- Define the deposit
def deposit : ℝ := total_paid - total_installment_payments

-- Statement to be proven
theorem deposit_is_3000 : deposit = 3000 := 
by 
  sorry

end NUMINAMATH_GPT_deposit_is_3000_l1289_128988


namespace NUMINAMATH_GPT_peter_present_age_l1289_128986

def age_problem (P J : ℕ) : Prop :=
  J = P + 12 ∧ P - 10 = (1 / 3 : ℚ) * (J - 10)

theorem peter_present_age : ∃ (P : ℕ), ∃ (J : ℕ), age_problem P J ∧ P = 16 :=
by {
  -- Add the proof here, which is not required
  sorry
}

end NUMINAMATH_GPT_peter_present_age_l1289_128986


namespace NUMINAMATH_GPT_gain_in_transaction_per_year_l1289_128959

noncomputable def compounded_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def gain_per_year (P : ℝ) (t : ℝ) (r1 : ℝ) (n1 : ℕ) (r2 : ℝ) (n2 : ℕ) : ℝ :=
  let amount_repaid := compounded_interest P r1 n1 t
  let amount_received := compounded_interest P r2 n2 t
  (amount_received - amount_repaid) / t

theorem gain_in_transaction_per_year :
  let P := 8000
  let t := 3
  let r1 := 0.05
  let n1 := 2
  let r2 := 0.07
  let n2 := 4
  abs (gain_per_year P t r1 n1 r2 n2 - 191.96) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_gain_in_transaction_per_year_l1289_128959


namespace NUMINAMATH_GPT_shanghai_masters_total_matches_l1289_128975

theorem shanghai_masters_total_matches : 
  let players := 8
  let groups := 2
  let players_per_group := 4
  let round_robin_matches_per_group := (players_per_group * (players_per_group - 1)) / 2
  let round_robin_total_matches := round_robin_matches_per_group * groups
  let elimination_matches := 2 * (groups - 1)  -- semi-final matches
  let final_matches := 2  -- one final and one third-place match
  round_robin_total_matches + elimination_matches + final_matches = 16 :=
by
  sorry

end NUMINAMATH_GPT_shanghai_masters_total_matches_l1289_128975


namespace NUMINAMATH_GPT_value_of_a_l1289_128910

theorem value_of_a (a : ℝ) (h : (1 : ℝ)^2 - 2 * (1 : ℝ) + a = 0) : a = 1 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_a_l1289_128910


namespace NUMINAMATH_GPT_vector_at_t_neg3_l1289_128966

theorem vector_at_t_neg3 :
  let a := (2, 3)
  let b := (12, -37)
  let d := ((b.1 - a.1) / 5, (b.2 - a.2) / 5)
  let line_param (t : ℝ) := (a.1 + t * d.1, a.2 + t * d.2)
  line_param (-3) = (-4, 27) := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_vector_at_t_neg3_l1289_128966


namespace NUMINAMATH_GPT_absent_present_probability_l1289_128934

theorem absent_present_probability : 
  ∀ (p_absent_normal p_absent_workshop p_present_workshop : ℚ), 
    p_absent_normal = 1 / 20 →
    p_absent_workshop = 2 * p_absent_normal →
    p_present_workshop = 1 - p_absent_workshop →
    p_absent_workshop = 1 / 10 →
    (p_present_workshop * p_absent_workshop + p_absent_workshop * p_present_workshop) * 100 = 18 :=
by
  intros
  sorry

end NUMINAMATH_GPT_absent_present_probability_l1289_128934


namespace NUMINAMATH_GPT_correct_statements_about_microbial_counting_l1289_128935

def hemocytometer_counts_bacteria_or_yeast : Prop :=
  true -- based on condition 1

def plate_streaking_allows_colony_counting : Prop :=
  false -- count is not done using the plate streaking method, based on the analysis

def dilution_plating_allows_colony_counting : Prop :=
  true -- based on condition 3  
  
def dilution_plating_count_is_accurate : Prop :=
  false -- colony count is often lower than the actual number, based on the analysis

theorem correct_statements_about_microbial_counting :
  (hemocytometer_counts_bacteria_or_yeast ∧ dilution_plating_allows_colony_counting)
= (plate_streaking_allows_colony_counting ∨ dilution_plating_count_is_accurate) :=
by sorry

end NUMINAMATH_GPT_correct_statements_about_microbial_counting_l1289_128935


namespace NUMINAMATH_GPT_slower_train_passing_time_l1289_128961

/--
Two goods trains, each 500 meters long, are running in opposite directions on parallel tracks. 
Their respective speeds are 45 kilometers per hour and 15 kilometers per hour. 
Prove that the time taken by the slower train to pass the driver of the faster train is 30 seconds.
-/
theorem slower_train_passing_time : 
  ∀ (distance length_speed : ℝ), 
    distance = 500 →
    ∃ (v1 v2 : ℝ), 
      v1 = 45 * (1000 / 3600) → 
      v2 = 15 * (1000 / 3600) →
      (distance / ((v1 + v2) * (3/50)) = 30) :=
by
  sorry

end NUMINAMATH_GPT_slower_train_passing_time_l1289_128961


namespace NUMINAMATH_GPT_smallest_possible_N_l1289_128964

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end NUMINAMATH_GPT_smallest_possible_N_l1289_128964


namespace NUMINAMATH_GPT_problem_l1289_128938

noncomputable def f (x : ℝ) : ℝ :=
  sorry

theorem problem (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x + y) - f y = x * (x + 2 * y + 1))
                (h2 : f 1 = 0) :
  f 0 = -2 ∧ ∀ x : ℝ, f x = x^2 + x - 2 := by
  sorry

end NUMINAMATH_GPT_problem_l1289_128938


namespace NUMINAMATH_GPT_melies_meat_purchase_l1289_128979

-- Define the relevant variables and conditions
variable (initial_amount : ℕ) (amount_left : ℕ) (cost_per_kg : ℕ)

-- State the main theorem we want to prove
theorem melies_meat_purchase (h1 : initial_amount = 180) (h2 : amount_left = 16) (h3 : cost_per_kg = 82) :
  (initial_amount - amount_left) / cost_per_kg = 2 := by
  sorry

end NUMINAMATH_GPT_melies_meat_purchase_l1289_128979


namespace NUMINAMATH_GPT_tan_15_degrees_theta_range_valid_max_f_value_l1289_128907

-- Define the dot product condition
def dot_product_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  AB * BC * (Real.cos θ) = 6

-- Define the sine inequality condition
def sine_inequality_condition (AB BC : ℝ) (θ : ℝ) : Prop :=
  6 * (2 - Real.sqrt 3) ≤ AB * BC * (Real.sin θ) ∧ AB * BC * (Real.sin θ) ≤ 6 * Real.sqrt 3

-- Define the maximum value function
noncomputable def f (θ : ℝ) : ℝ :=
  (1 - Real.sqrt 2 * Real.cos (2 * θ - Real.pi / 4)) / (Real.sin θ)

-- Proof that tan 15 degrees is equal to 2 - sqrt(3)
theorem tan_15_degrees : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 := 
  by sorry

-- Proof for the range of θ
theorem theta_range_valid (AB BC : ℝ) (θ : ℝ) 
  (h1 : dot_product_condition AB BC θ)
  (h2 : sine_inequality_condition AB BC θ) : 
  (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3) := 
  by sorry

-- Proof for the maximum value of the function
theorem max_f_value (θ : ℝ) 
  (h : (Real.pi / 12) ≤ θ ∧ θ ≤ (Real.pi / 3)) : 
  f θ ≤ Real.sqrt 3 - 1 := 
  by sorry

end NUMINAMATH_GPT_tan_15_degrees_theta_range_valid_max_f_value_l1289_128907


namespace NUMINAMATH_GPT_enrique_shredder_pages_l1289_128904

theorem enrique_shredder_pages (total_contracts : ℕ) (num_times : ℕ) (pages_per_time : ℕ) :
  total_contracts = 2132 ∧ num_times = 44 → pages_per_time = 48 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_enrique_shredder_pages_l1289_128904


namespace NUMINAMATH_GPT_perfect_square_iff_all_perfect_squares_l1289_128905

theorem perfect_square_iff_all_perfect_squares
  (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0):
  (∃ k : ℕ, (xy + 1) * (yz + 1) * (zx + 1) = k^2) ↔
  (∃ a b c : ℕ, xy + 1 = a^2 ∧ yz + 1 = b^2 ∧ zx + 1 = c^2) := 
sorry

end NUMINAMATH_GPT_perfect_square_iff_all_perfect_squares_l1289_128905


namespace NUMINAMATH_GPT_calculate_amount_l1289_128933

theorem calculate_amount (p1 p2 p3: ℝ) : 
  p1 = 0.15 * 4000 ∧ 
  p2 = p1 - 0.25 * p1 ∧ 
  p3 = 0.07 * p2 -> 
  (p3 + 0.10 * p3) = 34.65 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_amount_l1289_128933


namespace NUMINAMATH_GPT_count_three_digit_integers_with_tens_7_divisible_by_25_l1289_128926

theorem count_three_digit_integers_with_tens_7_divisible_by_25 :
  ∃ n, n = 33 ∧ ∃ k1 k2 : ℕ, 175 = 25 * k1 ∧ 975 = 25 * k2 ∧ (k2 - k1 + 1 = n) :=
by
  sorry

end NUMINAMATH_GPT_count_three_digit_integers_with_tens_7_divisible_by_25_l1289_128926


namespace NUMINAMATH_GPT_gardener_trees_problem_l1289_128950

theorem gardener_trees_problem 
  (maple_trees : ℕ) (oak_trees : ℕ) (birch_trees : ℕ) 
  (total_trees : ℕ) (valid_positions : ℕ) 
  (total_arrangements : ℕ) (probability_numerator : ℕ) (probability_denominator : ℕ) 
  (reduced_numerator : ℕ) (reduced_denominator : ℕ) (m_plus_n : ℕ) :
  (maple_trees = 5) ∧ 
  (oak_trees = 3) ∧ 
  (birch_trees = 7) ∧ 
  (total_trees = 15) ∧ 
  (valid_positions = 8) ∧ 
  (total_arrangements = 120120) ∧ 
  (probability_numerator = 40) ∧ 
  (probability_denominator = total_arrangements) ∧ 
  (reduced_numerator = 1) ∧ 
  (reduced_denominator = 3003) ∧ 
  (m_plus_n = reduced_numerator + reduced_denominator) → 
  m_plus_n = 3004 := 
by
  intros _
  sorry

end NUMINAMATH_GPT_gardener_trees_problem_l1289_128950


namespace NUMINAMATH_GPT_triangle_problem_l1289_128984

-- Defining the conditions as Lean constructs
variable (a c : ℝ)
variable (b : ℝ := 3)
variable (cosB : ℝ := 1 / 3)
variable (dotProductBACBC : ℝ := 2)
variable (cosB_minus_C : ℝ := 23 / 27)

-- Define the problem as a theorem in Lean 4
theorem triangle_problem
  (h1 : a > c)
  (h2 : a * c * cosB = dotProductBACBC)
  (h3 : a^2 + c^2 = 13) :
  a = 3 ∧ c = 2 ∧ cosB_minus_C = 23 / 27 := by
  sorry

end NUMINAMATH_GPT_triangle_problem_l1289_128984


namespace NUMINAMATH_GPT_max_value_of_function_l1289_128914

noncomputable def y (x : ℝ) : ℝ := 
  Real.sin x - Real.cos x - Real.sin x * Real.cos x

theorem max_value_of_function :
  ∃ x : ℝ, y x = (1 / 2) + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_function_l1289_128914


namespace NUMINAMATH_GPT_eq_abs_piecewise_l1289_128954

theorem eq_abs_piecewise (x : ℝ) : (|x| = if x >= 0 then x else -x) :=
by
  sorry

end NUMINAMATH_GPT_eq_abs_piecewise_l1289_128954


namespace NUMINAMATH_GPT_arithmetic_progression_terms_even_l1289_128995

variable (a d : ℝ) (n : ℕ)

open Real

theorem arithmetic_progression_terms_even {n : ℕ} (hn_even : n % 2 = 0)
  (h_sum_odd : (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 32)
  (h_sum_even : (n / 2 : ℝ) * (2 * a + 2 * d + (n - 2) * d) = 40)
  (h_last_exceeds_first : (a + (n - 1) * d) - a = 8) : n = 16 :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_terms_even_l1289_128995


namespace NUMINAMATH_GPT_clare_milk_cartons_l1289_128960

def money_given := 47
def cost_per_loaf := 2
def loaves_bought := 4
def cost_per_milk := 2
def money_left := 35

theorem clare_milk_cartons : (money_given - money_left - loaves_bought * cost_per_loaf) / cost_per_milk = 2 :=
by
  sorry

end NUMINAMATH_GPT_clare_milk_cartons_l1289_128960


namespace NUMINAMATH_GPT_intersection_correct_l1289_128923

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 2, 3, 4}

theorem intersection_correct : A ∩ B = {2, 3} := sorry

end NUMINAMATH_GPT_intersection_correct_l1289_128923


namespace NUMINAMATH_GPT_arithmetic_example_l1289_128983

theorem arithmetic_example : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_example_l1289_128983


namespace NUMINAMATH_GPT_solve_real_equation_l1289_128998

theorem solve_real_equation (x : ℝ) :
  x^2 * (x + 1)^2 + x^2 = 3 * (x + 1)^2 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_GPT_solve_real_equation_l1289_128998


namespace NUMINAMATH_GPT_parabola_value_f_l1289_128924

theorem parabola_value_f (d e f : ℝ) :
  (∀ y : ℝ, x = d * y ^ 2 + e * y + f) →
  (∀ x y : ℝ, (x + 3) = d * (y - 1) ^ 2) →
  (x = -1 ∧ y = 3) →
  y = 0 →
  f = -2.5 :=
sorry

end NUMINAMATH_GPT_parabola_value_f_l1289_128924


namespace NUMINAMATH_GPT_route_difference_l1289_128974

noncomputable def time_route_A (distance_A : ℝ) (speed_A : ℝ) : ℝ :=
  (distance_A / speed_A) * 60

noncomputable def time_route_B (distance1_B distance2_B distance3_B : ℝ) (speed1_B speed2_B speed3_B : ℝ) : ℝ :=
  ((distance1_B / speed1_B) * 60) + 
  ((distance2_B / speed2_B) * 60) + 
  ((distance3_B / speed3_B) * 60)

theorem route_difference
  (distance_A : ℝ := 8)
  (speed_A : ℝ := 25)
  (distance1_B : ℝ := 2)
  (distance2_B : ℝ := 0.5)
  (speed1_B : ℝ := 50)
  (speed2_B : ℝ := 20)
  (distance_total_B : ℝ := 7)
  (speed3_B : ℝ := 35) :
  time_route_A distance_A speed_A - time_route_B distance1_B distance2_B (distance_total_B - distance1_B - distance2_B) speed1_B speed2_B speed3_B = 7.586 :=
by
  sorry

end NUMINAMATH_GPT_route_difference_l1289_128974


namespace NUMINAMATH_GPT_divisibility_of_powers_l1289_128987

theorem divisibility_of_powers (n : ℤ) : 65 ∣ (7^4 * n - 4^4 * n) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_of_powers_l1289_128987


namespace NUMINAMATH_GPT_acute_triangle_and_angle_relations_l1289_128976

theorem acute_triangle_and_angle_relations (a b c u v w : ℝ) (A B C : ℝ)
  (h₁ : a^2 = u * (v + w - u))
  (h₂ : b^2 = v * (w + u - v))
  (h₃ : c^2 = w * (u + v - w)) :
  (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) ∧
  (∀ U V W : ℝ, U = 180 - 2 * A ∧ V = 180 - 2 * B ∧ W = 180 - 2 * C) :=
by sorry

end NUMINAMATH_GPT_acute_triangle_and_angle_relations_l1289_128976


namespace NUMINAMATH_GPT_perimeter_ABFCDE_l1289_128928

theorem perimeter_ABFCDE 
  (ABCD_perimeter : ℝ)
  (ABCD : ℝ)
  (triangle_BFC : ℝ -> ℝ)
  (translate_BFC : ℝ -> ℝ)
  (ABFCDE : ℝ -> ℝ -> ℝ)
  (h1 : ABCD_perimeter = 40)
  (h2 : ABCD = ABCD_perimeter / 4)
  (h3 : triangle_BFC ABCD = 10 * Real.sqrt 2)
  (h4 : translate_BFC (10 * Real.sqrt 2) = 10 * Real.sqrt 2)
  (h5 : ABFCDE ABCD (10 * Real.sqrt 2) = 40 + 20 * Real.sqrt 2)
  : ABFCDE ABCD (10 * Real.sqrt 2) = 40 + 20 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_perimeter_ABFCDE_l1289_128928


namespace NUMINAMATH_GPT_average_speed_l1289_128945

theorem average_speed (D : ℝ) (h1 : 0 < D) :
  let s1 := 60   -- speed from Q to B in miles per hour
  let s2 := 20   -- speed from B to C in miles per hour
  let d1 := 2 * D  -- distance from Q to B
  let d2 := D     -- distance from B to C
  let t1 := d1 / s1  -- time to travel from Q to B
  let t2 := d2 / s2  -- time to travel from B to C
  let total_distance := d1 + d2  -- total distance
  let total_time := t1 + t2   -- total time
  let average_speed := total_distance / total_time  -- average speed
  average_speed = 36 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l1289_128945


namespace NUMINAMATH_GPT_number_of_black_bears_l1289_128946

-- Definitions of conditions
def brown_bears := 15
def white_bears := 24
def total_bears := 66

-- The proof statement
theorem number_of_black_bears : (total_bears - (brown_bears + white_bears) = 27) := by
  sorry

end NUMINAMATH_GPT_number_of_black_bears_l1289_128946


namespace NUMINAMATH_GPT_simple_interest_double_l1289_128955

theorem simple_interest_double (P : ℝ) (r : ℝ) (t : ℝ) (A : ℝ)
  (h1 : t = 50)
  (h2 : A = 2 * P) 
  (h3 : A - P = P * r * t / 100) :
  r = 2 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_simple_interest_double_l1289_128955


namespace NUMINAMATH_GPT_reflection_y_axis_matrix_correct_l1289_128965

def reflect_y_axis_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(-1 : ℝ), (0 : ℝ)], ![(0 : ℝ), (1 : ℝ)]]

theorem reflection_y_axis_matrix_correct :
  reflect_y_axis_matrix = ![![(-1 : ℝ), (0 : ℝ)], ![(0 : ℝ), (1 : ℝ)]] :=
by
  sorry

end NUMINAMATH_GPT_reflection_y_axis_matrix_correct_l1289_128965


namespace NUMINAMATH_GPT_tens_digit_11_pow_2045_l1289_128996

theorem tens_digit_11_pow_2045 : 
    ((11 ^ 2045) % 100) / 10 % 10 = 5 :=
by
    sorry

end NUMINAMATH_GPT_tens_digit_11_pow_2045_l1289_128996


namespace NUMINAMATH_GPT_extremum_value_l1289_128922

noncomputable def f (x a b : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

theorem extremum_value (a b : ℝ) (h1 : (3 - 6 * a + b = 0)) (h2 : (-1 + 3 * a - b + a^2 = 0)) :
  a - b = -7 :=
by
  sorry

end NUMINAMATH_GPT_extremum_value_l1289_128922


namespace NUMINAMATH_GPT_minimum_p_for_required_profit_l1289_128918

noncomputable def profit (x p : ℝ) : ℝ := p * x - (0.5 * x^2 - 2 * x - 10)
noncomputable def max_profit (p : ℝ) : ℝ := (p + 2)^2 / 2 + 10

theorem minimum_p_for_required_profit : ∀ (p : ℝ), 3 * max_profit p >= 126 → p >= 6 :=
by
  intro p
  unfold max_profit
  -- Given:  3 * ((p + 2)^2 / 2 + 10) >= 126
  sorry

end NUMINAMATH_GPT_minimum_p_for_required_profit_l1289_128918


namespace NUMINAMATH_GPT_function_characterization_l1289_128981

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization :
  (∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0) →
  (∀ x : ℝ, 0 ≤ x → f x = if x < 2 then 2 / (2 - x) else 0) := sorry

end NUMINAMATH_GPT_function_characterization_l1289_128981


namespace NUMINAMATH_GPT_eval_f_at_5_l1289_128915

def f (x : ℝ) : ℝ := 2 * x^7 - 9 * x^6 + 5 * x^5 - 49 * x^4 - 5 * x^3 + 2 * x^2 + x + 1

theorem eval_f_at_5 : f 5 = 56 := 
 by 
   sorry

end NUMINAMATH_GPT_eval_f_at_5_l1289_128915


namespace NUMINAMATH_GPT_triangle_properties_l1289_128951

-- Define the sides of the triangle
def side1 : ℕ := 8
def side2 : ℕ := 15
def hypotenuse : ℕ := 17

-- Using the Pythagorean theorem to assert it is a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Calculate the area of the right triangle
def triangle_area (a b : ℕ) : ℕ :=
  (a * b) / 2

-- Calculate the perimeter of the triangle
def triangle_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem triangle_properties :
  let a := side1
  let b := side2
  let c := hypotenuse
  is_right_triangle a b c →
  triangle_area a b = 60 ∧ triangle_perimeter a b c = 40 := by
  intros h
  sorry

end NUMINAMATH_GPT_triangle_properties_l1289_128951


namespace NUMINAMATH_GPT_charges_are_equal_l1289_128931

variable (a : ℝ)  -- original price for both travel agencies

def charge_A (a : ℝ) : ℝ := a + 2 * 0.7 * a
def charge_B (a : ℝ) : ℝ := 3 * 0.8 * a

theorem charges_are_equal : charge_A a = charge_B a :=
by
  sorry

end NUMINAMATH_GPT_charges_are_equal_l1289_128931


namespace NUMINAMATH_GPT_find_value_of_pow_function_l1289_128956

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem find_value_of_pow_function :
  (∃ α : ℝ, power_function α 4 = 1/2) →
  ∃ α : ℝ, power_function α (1/4) = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_pow_function_l1289_128956


namespace NUMINAMATH_GPT_negation_universal_proposition_l1289_128932

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_universal_proposition_l1289_128932


namespace NUMINAMATH_GPT_binom_mult_eq_6720_l1289_128901

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end NUMINAMATH_GPT_binom_mult_eq_6720_l1289_128901


namespace NUMINAMATH_GPT_find_largest_even_integer_l1289_128930

-- Define the sum of the first 30 positive even integers
def sum_first_30_even : ℕ := 2 * (30 * 31 / 2)

-- Assume five consecutive even integers and their sum
def consecutive_even_sum (m : ℕ) : ℕ := (m - 8) + (m - 6) + (m - 4) + (m - 2) + m

-- Statement of the theorem to be proven
theorem find_largest_even_integer : ∃ (m : ℕ), consecutive_even_sum m = sum_first_30_even ∧ m = 190 :=
by
  sorry

end NUMINAMATH_GPT_find_largest_even_integer_l1289_128930


namespace NUMINAMATH_GPT_max_brownies_l1289_128941

-- Definitions for the conditions given in the problem
def is_interior_pieces (m n : ℕ) : ℕ := (m - 2) * (n - 2)
def is_perimeter_pieces (m n : ℕ) : ℕ := 2 * m + 2 * n - 4

-- The assertion that the number of brownies along the perimeter is twice the number in the interior
def condition (m n : ℕ) : Prop := 2 * is_interior_pieces m n = is_perimeter_pieces m n

-- The statement that the maximum number of brownies under the given condition is 84
theorem max_brownies : ∃ (m n : ℕ), condition m n ∧ m * n = 84 := by
  sorry

end NUMINAMATH_GPT_max_brownies_l1289_128941


namespace NUMINAMATH_GPT_point_in_quadrants_l1289_128973

theorem point_in_quadrants (x y : ℝ) (h1 : 4 * x + 7 * y = 28) (h2 : |x| = |y|) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by
  sorry

end NUMINAMATH_GPT_point_in_quadrants_l1289_128973


namespace NUMINAMATH_GPT_dimes_in_piggy_bank_l1289_128909

variable (q d : ℕ)

def total_coins := q + d = 100
def total_amount := 25 * q + 10 * d = 1975

theorem dimes_in_piggy_bank (h1 : total_coins q d) (h2 : total_amount q d) : d = 35 := by
  sorry

end NUMINAMATH_GPT_dimes_in_piggy_bank_l1289_128909


namespace NUMINAMATH_GPT_total_players_correct_l1289_128967

-- Define the number of players for each type of sport
def cricket_players : Nat := 12
def hockey_players : Nat := 17
def football_players : Nat := 11
def softball_players : Nat := 10

-- The theorem we aim to prove
theorem total_players_correct : 
  cricket_players + hockey_players + football_players + softball_players = 50 := by
  sorry

end NUMINAMATH_GPT_total_players_correct_l1289_128967


namespace NUMINAMATH_GPT_exists_k_such_that_n_eq_k_2010_l1289_128900

theorem exists_k_such_that_n_eq_k_2010 (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h : m * n ∣ m ^ 2010 + n ^ 2010 + n) : ∃ k : ℕ, 0 < k ∧ n = k ^ 2010 := by
  sorry

end NUMINAMATH_GPT_exists_k_such_that_n_eq_k_2010_l1289_128900


namespace NUMINAMATH_GPT_Jake_peach_count_l1289_128949

theorem Jake_peach_count (Steven_peaches : ℕ) (Jake_peach_difference : ℕ) (h1 : Steven_peaches = 19) (h2 : Jake_peach_difference = 12) : 
  Steven_peaches - Jake_peach_difference = 7 :=
by
  sorry

end NUMINAMATH_GPT_Jake_peach_count_l1289_128949


namespace NUMINAMATH_GPT_monkey_reaches_top_l1289_128957

def monkey_climb_time (tree_height : ℕ) (climb_per_hour : ℕ) (slip_per_hour : ℕ) 
  (rest_hours : ℕ) (cycle_hours : ℕ) : ℕ :=
  if (tree_height % (climb_per_hour - slip_per_hour) > climb_per_hour) 
    then (tree_height / (climb_per_hour - slip_per_hour)) + cycle_hours
    else (tree_height / (climb_per_hour - slip_per_hour)) + cycle_hours - 1

theorem monkey_reaches_top :
  monkey_climb_time 253 7 4 1 4 = 109 := 
sorry

end NUMINAMATH_GPT_monkey_reaches_top_l1289_128957


namespace NUMINAMATH_GPT_casey_saves_by_paying_monthly_l1289_128919

theorem casey_saves_by_paying_monthly :
  let weekly_rate := 280
  let monthly_rate := 1000
  let weeks_in_a_month := 4
  let number_of_months := 3
  let total_weeks := number_of_months * weeks_in_a_month
  let total_cost_weekly := total_weeks * weekly_rate
  let total_cost_monthly := number_of_months * monthly_rate
  let savings := total_cost_weekly - total_cost_monthly
  savings = 360 :=
by
  sorry

end NUMINAMATH_GPT_casey_saves_by_paying_monthly_l1289_128919


namespace NUMINAMATH_GPT_min_value_expression_l1289_128993

theorem min_value_expression 
  (a b c : ℝ)
  (h1 : a + b + c = -1)
  (h2 : a * b * c ≤ -3) : 
  (ab + 1) / (a + b) + (bc + 1) / (b + c) + (ca + 1) / (c + a) ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1289_128993


namespace NUMINAMATH_GPT_moles_of_ammonia_combined_l1289_128939

theorem moles_of_ammonia_combined (n_CO2 n_Urea n_NH3 : ℕ) (h1 : n_CO2 = 1) (h2 : n_Urea = 1) (h3 : n_Urea = n_CO2)
  (h4 : n_Urea = 2 * n_NH3): n_NH3 = 2 := 
by
  sorry

end NUMINAMATH_GPT_moles_of_ammonia_combined_l1289_128939


namespace NUMINAMATH_GPT_total_bananas_eq_l1289_128992

def groups_of_bananas : ℕ := 2
def bananas_per_group : ℕ := 145

theorem total_bananas_eq : groups_of_bananas * bananas_per_group = 290 :=
by
  sorry

end NUMINAMATH_GPT_total_bananas_eq_l1289_128992


namespace NUMINAMATH_GPT_fixed_point_coordinates_l1289_128902

theorem fixed_point_coordinates (k : ℝ) (M : ℝ × ℝ) (h : ∀ k : ℝ, M.2 - 2 = k * (M.1 + 1)) :
  M = (-1, 2) :=
sorry

end NUMINAMATH_GPT_fixed_point_coordinates_l1289_128902


namespace NUMINAMATH_GPT_distinct_ordered_pairs_solution_l1289_128940

theorem distinct_ordered_pairs_solution :
  (∃ n : ℕ, ∀ x y : ℕ, (x > 0 ∧ y > 0 ∧ x^4 * y^4 - 24 * x^2 * y^2 + 35 = 0) ↔ n = 1) :=
sorry

end NUMINAMATH_GPT_distinct_ordered_pairs_solution_l1289_128940


namespace NUMINAMATH_GPT_percentage_of_125_equals_75_l1289_128952

theorem percentage_of_125_equals_75 (p : ℝ) (h : p * 125 = 75) : p = 60 / 100 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_125_equals_75_l1289_128952


namespace NUMINAMATH_GPT_relationship_bx_l1289_128908

variable {a b t x : ℝ}

-- Given conditions
variable (h1 : b > a)
variable (h2 : a > 1)
variable (h3 : t > 0)
variable (h4 : a ^ x = a + t)

theorem relationship_bx (h1 : b > a) (h2 : a > 1) (h3 : t > 0) (h4 : a ^ x = a + t) : b ^ x > b + t :=
by
  sorry

end NUMINAMATH_GPT_relationship_bx_l1289_128908


namespace NUMINAMATH_GPT_total_fish_l1289_128921

-- Definition of the number of fish Lilly has
def lilly_fish : Nat := 10

-- Definition of the number of fish Rosy has
def rosy_fish : Nat := 8

-- Statement to prove
theorem total_fish : lilly_fish + rosy_fish = 18 := 
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_total_fish_l1289_128921


namespace NUMINAMATH_GPT_dice_digit_distribution_l1289_128962

theorem dice_digit_distribution : ∃ n : ℕ, n = 10 ∧ 
  (∀ (d1 d2 : Finset ℕ), d1.card = 6 ∧ d2.card = 6 ∧
  (0 ∈ d1) ∧ (1 ∈ d1) ∧ (2 ∈ d1) ∧ 
  (0 ∈ d2) ∧ (1 ∈ d2) ∧ (2 ∈ d2) ∧
  ({3, 4, 5, 6, 7, 8} ⊆ (d1 ∪ d2)) ∧ 
  (∀ i, i ∈ d1 ∪ d2 → i ∈ (Finset.range 10))) := 
  sorry

end NUMINAMATH_GPT_dice_digit_distribution_l1289_128962


namespace NUMINAMATH_GPT_soda_cost_132_cents_l1289_128927

theorem soda_cost_132_cents
  (b s : ℕ)
  (h1 : 3 * b + 2 * s + 30 = 510)
  (h2 : 2 * b + 3 * s = 540) 
  : s = 132 :=
by
  sorry

end NUMINAMATH_GPT_soda_cost_132_cents_l1289_128927


namespace NUMINAMATH_GPT_carmen_candle_burn_time_l1289_128958

theorem carmen_candle_burn_time 
  (burn_time_first_scenario : ℕ)
  (nights_per_candle : ℕ)
  (total_candles_second_scenario : ℕ)
  (total_nights_second_scenario : ℕ)
  (h1 : burn_time_first_scenario = 1)
  (h2 : nights_per_candle = 8)
  (h3 : total_candles_second_scenario = 6)
  (h4 : total_nights_second_scenario = 24) :
  (total_candles_second_scenario * nights_per_candle) / total_nights_second_scenario = 2 :=
by
  sorry

end NUMINAMATH_GPT_carmen_candle_burn_time_l1289_128958


namespace NUMINAMATH_GPT_length_CD_l1289_128978

theorem length_CD (AB AC BD CD : ℝ) (hAB : AB = 2) (hAC : AC = 5) (hBD : BD = 6) :
    CD = 3 :=
by
  sorry

end NUMINAMATH_GPT_length_CD_l1289_128978
