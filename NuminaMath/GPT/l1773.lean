import Mathlib

namespace equation1_solution_equation2_solution_equation3_solution_l1773_177349

theorem equation1_solution :
  ∀ x : ℝ, x^2 - 2 * x - 99 = 0 ↔ x = 11 ∨ x = -9 :=
by
  sorry

theorem equation2_solution :
  ∀ x : ℝ, x^2 + 5 * x = 7 ↔ x = (-5 - Real.sqrt 53) / 2 ∨ x = (-5 + Real.sqrt 53) / 2 :=
by
  sorry

theorem equation3_solution :
  ∀ x : ℝ, 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = -1/2 ∨ x = 3/4 :=
by
  sorry

end equation1_solution_equation2_solution_equation3_solution_l1773_177349


namespace pete_flag_total_circle_square_l1773_177346

theorem pete_flag_total_circle_square : 
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  circles + squares = 54 := 
by
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  show circles + squares = 54
  sorry

end pete_flag_total_circle_square_l1773_177346


namespace convert_ternary_to_octal_2101211_l1773_177373

def ternaryToOctal (n : List ℕ) : ℕ := 
  sorry

theorem convert_ternary_to_octal_2101211 :
  ternaryToOctal [2, 1, 0, 1, 2, 1, 1] = 444
  := sorry

end convert_ternary_to_octal_2101211_l1773_177373


namespace sum_of_roots_l1773_177309

theorem sum_of_roots : 
  ∀ x1 x2 : ℝ, 
  (x1^2 + 2023*x1 = 2024 ∧ x2^2 + 2023*x2 = 2024) → 
  x1 + x2 = -2023 := 
by 
  sorry

end sum_of_roots_l1773_177309


namespace original_price_given_discounts_l1773_177310

theorem original_price_given_discounts (p q d : ℝ) (h : d > 0) :
  ∃ x : ℝ, x * (1 + (p - q) / 100 - p * q / 10000) = d :=
by
  sorry

end original_price_given_discounts_l1773_177310


namespace iggy_wednesday_run_6_l1773_177350

open Nat

noncomputable def iggy_miles_wednesday : ℕ :=
  let total_time := 4 * 60    -- Iggy spends 4 hours running (240 minutes)
  let pace := 10              -- Iggy runs 1 mile in 10 minutes
  let monday := 3
  let tuesday := 4
  let thursday := 8
  let friday := 3
  let total_miles_other_days := monday + tuesday + thursday + friday
  let total_time_other_days := total_miles_other_days * pace
  let wednesday_time := total_time - total_time_other_days
  wednesday_time / pace

theorem iggy_wednesday_run_6 :
  iggy_miles_wednesday = 6 := by
  sorry

end iggy_wednesday_run_6_l1773_177350


namespace problem1_problem2_l1773_177363

-- Problem 1
theorem problem1 (a b : ℝ) (h : a ≠ 0) : 
  (a - b^2 / a) / ((a^2 + 2 * a * b + b^2) / a) = (a - b) / (a + b) :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (6 - 2 * x ≥ 4) ∧ ((1 + 2 * x) / 3 > x - 1) ↔ (x ≤ 1) :=
by
  sorry

end problem1_problem2_l1773_177363


namespace negation_of_statement_l1773_177353

theorem negation_of_statement :
  ¬ (∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 := by
  sorry

end negation_of_statement_l1773_177353


namespace gasoline_tank_capacity_l1773_177319

theorem gasoline_tank_capacity (x : ℝ)
  (h1 : (7 / 8) * x - (1 / 2) * x = 12) : x = 32 := 
sorry

end gasoline_tank_capacity_l1773_177319


namespace unique_4_digit_number_l1773_177357

theorem unique_4_digit_number (P E R U : ℕ) 
  (hP : 0 ≤ P ∧ P < 10)
  (hE : 0 ≤ E ∧ E < 10)
  (hR : 0 ≤ R ∧ R < 10)
  (hU : 0 ≤ U ∧ U < 10)
  (hPERU : 1000 ≤ (P * 1000 + E * 100 + R * 10 + U) ∧ (P * 1000 + E * 100 + R * 10 + U) < 10000) 
  (h_eq : (P * 1000 + E * 100 + R * 10 + U) = (P + E + R + U) ^ U) : 
  (P = 4) ∧ (E = 9) ∧ (R = 1) ∧ (U = 3) ∧ (P * 1000 + E * 100 + R * 10 + U = 4913) :=
sorry

end unique_4_digit_number_l1773_177357


namespace max_remainder_209_lt_120_l1773_177336

theorem max_remainder_209_lt_120 : 
  ∃ n : ℕ, n < 120 ∧ (209 % n = 104) := 
sorry

end max_remainder_209_lt_120_l1773_177336


namespace power_division_l1773_177325

theorem power_division (a b : ℕ) (h₁ : 64 = 8^2) (h₂ : a = 15) (h₃ : b = 7) : 8^a / 64^b = 8 :=
by
  -- Equivalent to 8^15 / 64^7 = 8, given that 64 = 8^2
  sorry

end power_division_l1773_177325


namespace present_age_of_son_is_22_l1773_177387

theorem present_age_of_son_is_22 (S F : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end present_age_of_son_is_22_l1773_177387


namespace area_of_octagon_in_square_l1773_177365

theorem area_of_octagon_in_square (perimeter : ℝ) (side_length : ℝ) (area_square : ℝ)
  (segment_length : ℝ) (area_triangle : ℝ) (total_area_triangles : ℝ) :
  perimeter = 144 →
  side_length = perimeter / 4 →
  segment_length = side_length / 3 →
  area_triangle = (segment_length * segment_length) / 2 →
  total_area_triangles = 4 * area_triangle →
  area_square = side_length * side_length →
  (area_square - total_area_triangles) = 1008 :=
by
  sorry

end area_of_octagon_in_square_l1773_177365


namespace value_of_k_l1773_177375

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem value_of_k
  (a d : ℝ)
  (a1_eq_1 : a = 1)
  (sum_9_eq_sum_4 : 9/2 * (2*a + 8*d) = 4/2 * (2*a + 3*d))
  (k : ℕ)
  (a_k_plus_a_4_eq_0 : arithmetic_sequence a d k + arithmetic_sequence a d 4 = 0) :
  k = 10 :=
by
  sorry

end value_of_k_l1773_177375


namespace contractor_fired_people_l1773_177395

theorem contractor_fired_people :
  ∀ (total_days : ℕ) (initial_people : ℕ) (partial_days : ℕ) 
    (partial_work_fraction : ℚ) (remaining_days : ℕ) 
    (fired_people : ℕ),
  total_days = 100 →
  initial_people = 10 →
  partial_days = 20 →
  partial_work_fraction = 1 / 4 →
  remaining_days = 75 →
  (initial_people - fired_people) * remaining_days * (1 - partial_work_fraction) / partial_days = initial_people * total_days →
  fired_people = 2 :=
by
  intros total_days initial_people partial_days partial_work_fraction remaining_days fired_people
  intro h1 h2 h3 h4 h5 h6
  sorry

end contractor_fired_people_l1773_177395


namespace fuel_at_40_min_fuel_l1773_177372

section FuelConsumption

noncomputable def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

noncomputable def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (100 / x)

theorem fuel_at_40 : total_fuel 40 = 17.5 :=
by sorry

theorem min_fuel : total_fuel 80 = 11.25 ∧ ∀ x, (0 < x ∧ x ≤ 120) → total_fuel x ≥ total_fuel 80 :=
by sorry

end FuelConsumption

end fuel_at_40_min_fuel_l1773_177372


namespace grandma_contribution_l1773_177334

def trip_cost : ℝ := 485
def candy_bar_profit : ℝ := 1.25
def candy_bars_sold : ℕ := 188
def amount_earned_from_selling_candy_bars : ℝ := candy_bars_sold * candy_bar_profit
def amount_grandma_gave : ℝ := trip_cost - amount_earned_from_selling_candy_bars

theorem grandma_contribution :
  amount_grandma_gave = 250 := by
  sorry

end grandma_contribution_l1773_177334


namespace tan_neg_405_eq_neg_1_l1773_177360

theorem tan_neg_405_eq_neg_1 :
  Real.tan (Real.pi * -405 / 180) = -1 := 
sorry

end tan_neg_405_eq_neg_1_l1773_177360


namespace statement_1_equiv_statement_2_equiv_l1773_177352

-- Statement 1
variable (A B C : Prop)

theorem statement_1_equiv : ((A ∨ B) → C) ↔ (A → C) ∧ (B → C) :=
by
  sorry

-- Statement 2
theorem statement_2_equiv : (A → (B ∧ C)) ↔ (A → B) ∧ (A → C) :=
by
  sorry

end statement_1_equiv_statement_2_equiv_l1773_177352


namespace first_divisor_l1773_177326

theorem first_divisor (d x : ℕ) (h1 : ∃ k : ℕ, x = k * d + 11) (h2 : ∃ m : ℕ, x = 9 * m + 2) : d = 3 :=
sorry

end first_divisor_l1773_177326


namespace novel_writing_time_l1773_177323

theorem novel_writing_time :
  ∀ (total_words : ℕ) (first_half_speed second_half_speed : ℕ),
    total_words = 50000 →
    first_half_speed = 600 →
    second_half_speed = 400 →
    (total_words / 2 / first_half_speed + total_words / 2 / second_half_speed : ℚ) = 104.17 :=
by
  -- No proof is required, placeholder using sorry
  sorry

end novel_writing_time_l1773_177323


namespace compare_expressions_l1773_177399

theorem compare_expressions (x : ℝ) : (x - 2) * (x + 3) > x^2 + x - 7 :=
by {
  -- below proof is left as an exercise
  sorry
}

end compare_expressions_l1773_177399


namespace quadrilateral_offset_l1773_177364

-- Define the problem statement
theorem quadrilateral_offset
  (d : ℝ) (x : ℝ) (y : ℝ) (A : ℝ)
  (h₀ : d = 10) 
  (h₁ : y = 3) 
  (h₂ : A = 50) :
  x = 7 :=
by
  -- Assuming the given conditions
  have h₃ : A = 1/2 * d * x + 1/2 * d * y :=
  by
    -- specific formula for area of the quadrilateral
    sorry
  
  -- Given A = 50, d = 10, y = 3, solve for x to show x = 7
  sorry

end quadrilateral_offset_l1773_177364


namespace minimum_transportation_cost_l1773_177355

theorem minimum_transportation_cost :
  ∀ (x : ℕ), 
    (17 - x) + (x - 3) = 12 → 
    (18 - x) + (17 - x) = 14 → 
    (200 * x + 19300 = 19900) → 
    (x = 3) 
:= by sorry

end minimum_transportation_cost_l1773_177355


namespace solve_inequality_l1773_177351

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x ^ 3 - 3 * x ^ 2 + 2 * x) / (x ^ 2 - 3 * x + 2) ≤ 0 ∧
  x ≠ 1 ∧ x ≠ 2

theorem solve_inequality :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | x ≤ 0 ∧ x ≠ 1 ∧ x ≠ 2} :=
  sorry

end solve_inequality_l1773_177351


namespace tom_wins_with_smallest_n_l1773_177302

def tom_and_jerry_game_proof_problem (n : ℕ) : Prop :=
  ∀ (pos : ℕ), pos ≥ 1 ∧ pos ≤ 2018 → 
  ∀ (move : ℕ), move ≥ 1 ∧ move ≤ n →
  (∃ n_min : ℕ, n_min ≤ n ∧ ∀ pos, (pos ≤ n_min ∨ pos > 2018 - n_min) → false)

theorem tom_wins_with_smallest_n : tom_and_jerry_game_proof_problem 1010 :=
sorry

end tom_wins_with_smallest_n_l1773_177302


namespace solve_problem_l1773_177391

def problem_statement : Prop := (245245 % 35 = 0)

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l1773_177391


namespace exists_composite_power_sum_l1773_177308

def is_composite (n : ℕ) : Prop := ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q 

theorem exists_composite_power_sum (a : ℕ) (h1 : 1 < a) (h2 : a ≤ 100) : 
  ∃ n, (n > 0) ∧ (n ≤ 6) ∧ is_composite (a ^ (2 ^ n) + 1) :=
by
  sorry

end exists_composite_power_sum_l1773_177308


namespace sample_avg_std_dev_xy_l1773_177389

theorem sample_avg_std_dev_xy {x y : ℝ} (h1 : (4 + 5 + 6 + x + y) / 5 = 5)
  (h2 : (( (4 - 5)^2 + (5 - 5)^2 + (6 - 5)^2 + (x - 5)^2 + (y - 5)^2 ) / 5) = 2) : x * y = 21 :=
by
  sorry

end sample_avg_std_dev_xy_l1773_177389


namespace speed_in_still_water_l1773_177335

-- Definitions of the conditions
def downstream_condition (v_m v_s : ℝ) : Prop := v_m + v_s = 6
def upstream_condition (v_m v_s : ℝ) : Prop := v_m - v_s = 3

-- The theorem to be proven
theorem speed_in_still_water (v_m v_s : ℝ) 
  (h1 : downstream_condition v_m v_s) 
  (h2 : upstream_condition v_m v_s) : v_m = 4.5 :=
by
  sorry

end speed_in_still_water_l1773_177335


namespace cards_not_in_box_correct_l1773_177383

-- Total number of cards Robie had at the beginning.
def total_cards : ℕ := 75

-- Number of cards in each box.
def cards_per_box : ℕ := 10

-- Number of boxes Robie gave away.
def boxes_given_away : ℕ := 2

-- Number of boxes Robie has with him.
def boxes_with_rob : ℕ := 5

-- The number of cards not placed in a box.
def cards_not_in_box : ℕ :=
  total_cards - (boxes_given_away * cards_per_box + boxes_with_rob * cards_per_box)

theorem cards_not_in_box_correct : cards_not_in_box = 5 :=
by
  unfold cards_not_in_box
  unfold total_cards
  unfold boxes_given_away
  unfold cards_per_box
  unfold boxes_with_rob
  sorry

end cards_not_in_box_correct_l1773_177383


namespace usual_time_to_office_l1773_177380

theorem usual_time_to_office
  (S T : ℝ) 
  (h1 : ∀ D : ℝ, D = S * T)
  (h2 : ∀ D : ℝ, D = (4 / 5) * S * (T + 10)):
  T = 40 := 
by
  sorry

end usual_time_to_office_l1773_177380


namespace M_subset_N_l1773_177342

noncomputable def M_set : Set ℝ := { x | ∃ (k : ℤ), x = k / 4 + 1 / 4 }
noncomputable def N_set : Set ℝ := { x | ∃ (k : ℤ), x = k / 8 - 1 / 4 }

theorem M_subset_N : M_set ⊆ N_set :=
sorry

end M_subset_N_l1773_177342


namespace find_sum_of_squares_l1773_177354

theorem find_sum_of_squares (a b c m : ℤ) (h1 : a + b + c = 0) (h2 : a * b + b * c + a * c = -2023) (h3 : a * b * c = -m) : a^2 + b^2 + c^2 = 4046 := by
  sorry

end find_sum_of_squares_l1773_177354


namespace line_equation_l1773_177333

noncomputable def line_intersects_at_point (a1 a2 b1 b2 c1 c2 : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 * a1 + p.2 * b1 = c1 ∧ p.1 * a2 + p.2 * b2 = c2

noncomputable def point_on_line (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 = c

theorem line_equation
  (p : ℝ × ℝ)
  (h1 : line_intersects_at_point 3 2 2 3 5 5 p)
  (h2 : point_on_line 0 1 (-5) p)
  : ∃ a b c : ℝ,  a * p.1 + b * p.2 + (-5) = 0 :=
sorry

end line_equation_l1773_177333


namespace find_ages_l1773_177312

variables (H J A : ℕ)

def conditions := 
  H + J + A = 90 ∧ 
  H = 2 * J - 5 ∧ 
  H + J - 10 = A

theorem find_ages (h_cond : conditions H J A) : 
  H = 32 ∧ 
  J = 18 ∧ 
  A = 40 :=
sorry

end find_ages_l1773_177312


namespace seats_still_available_l1773_177356

theorem seats_still_available (total_seats : ℕ) (two_fifths_seats : ℕ) (one_tenth_seats : ℕ) 
  (h1 : total_seats = 500) 
  (h2 : two_fifths_seats = (2 * total_seats) / 5) 
  (h3 : one_tenth_seats = total_seats / 10) :
  total_seats - (two_fifths_seats + one_tenth_seats) = 250 :=
by 
  sorry

end seats_still_available_l1773_177356


namespace number_composite_l1773_177376

theorem number_composite : ∃ a1 a2 : ℕ, a1 > 1 ∧ a2 > 1 ∧ 2^17 + 2^5 - 1 = a1 * a2 := 
by
  sorry

end number_composite_l1773_177376


namespace zero_not_in_range_of_g_l1773_177316

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈(Real.cos x) / (x + 3)⌉
  else if x < -3 then ⌊(Real.cos x) / (x + 3)⌋
  else 0 -- arbitrary value since it's undefined

theorem zero_not_in_range_of_g :
  ¬ (∃ x : ℝ, g x = 0) :=
by
  intro h
  sorry

end zero_not_in_range_of_g_l1773_177316


namespace quotient_of_division_l1773_177386

theorem quotient_of_division (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 52) 
  (h2 : divisor = 3) 
  (h3 : remainder = 4) 
  (h4 : dividend = divisor * quotient + remainder) : 
  quotient = 16 :=
by
  sorry

end quotient_of_division_l1773_177386


namespace min_value_abs_sum_exists_min_value_abs_sum_l1773_177388

theorem min_value_abs_sum (x : ℝ) : |x - 1| + |x - 4| ≥ 3 :=
by sorry

theorem exists_min_value_abs_sum : ∃ x : ℝ, |x - 1| + |x - 4| = 3 :=
by sorry

end min_value_abs_sum_exists_min_value_abs_sum_l1773_177388


namespace compute_expression_l1773_177306

theorem compute_expression (y : ℕ) (h : y = 3) : (y^8 + 10 * y^4 + 25) / (y^4 + 5) = 86 :=
by
  rw [h]
  sorry

end compute_expression_l1773_177306


namespace special_hash_value_l1773_177330

def special_hash (a b c d : ℝ) : ℝ :=
  d * b ^ 2 - 4 * a * c

theorem special_hash_value :
  special_hash 2 3 1 (1 / 2) = -3.5 :=
by
  -- Note: Insert proof here
  sorry

end special_hash_value_l1773_177330


namespace store_second_reduction_percentage_l1773_177382

theorem store_second_reduction_percentage (P : ℝ) :
  let first_reduction := 0.88 * P
  let second_reduction := 0.792 * P
  ∃ R : ℝ, (1 - R) * first_reduction = second_reduction ∧ R = 0.1 :=
by
  let first_reduction := 0.88 * P
  let second_reduction := 0.792 * P
  use 0.1
  sorry

end store_second_reduction_percentage_l1773_177382


namespace Rudolph_stop_signs_l1773_177317

def distance : ℕ := 5 + 2
def stopSignsPerMile : ℕ := 2
def totalStopSigns : ℕ := distance * stopSignsPerMile

theorem Rudolph_stop_signs :
  totalStopSigns = 14 := 
  by sorry

end Rudolph_stop_signs_l1773_177317


namespace find_c_l1773_177322

open Real

theorem find_c (a b c d : ℕ) (M : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : d > 1) (hM : M ≠ 1) :
  (M ^ (1 / a) * (M ^ (1 / b) * (M ^ (1 / c) * (M ^ (1 / d))))) ^ (1 / a * b * c * d) = (M ^ 37) ^ (1 / 48) →
  c = 2 :=
by
  sorry

end find_c_l1773_177322


namespace interval_proof_l1773_177337

theorem interval_proof (x : ℝ) (h1 : 2 < 3 * x) (h2 : 3 * x < 3) (h3 : 2 < 4 * x) (h4 : 4 * x < 3) :
    (2 / 3) < x ∧ x < (3 / 4) :=
sorry

end interval_proof_l1773_177337


namespace combined_salaries_l1773_177303

variable (S_A S_B S_C S_D S_E : ℝ)

theorem combined_salaries 
    (h1 : S_C = 16000)
    (h2 : (S_A + S_B + S_C + S_D + S_E) / 5 = 9000) : 
    S_A + S_B + S_D + S_E = 29000 :=
by 
    sorry

end combined_salaries_l1773_177303


namespace unique_solution_l1773_177398
-- Import necessary mathematical library

-- Define mathematical statement
theorem unique_solution (N : ℕ) (hN: N > 0) :
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ (m + (1 / 2 : ℝ) * (m + n - 1) * (m + n - 2) = N) :=
by {
  sorry
}

end unique_solution_l1773_177398


namespace sally_pens_proof_l1773_177370

variable (p : ℕ)  -- define p as a natural number for pens each student received
variable (pensLeft : ℕ)  -- define pensLeft as a natural number for pens left after distributing to students

-- Function representing Sally giving pens to each student
def pens_after_giving_students (p : ℕ) : ℕ := 342 - 44 * p

-- Condition 1: Left half of the remainder in her locker
def locker_pens (p : ℕ) : ℕ := (pens_after_giving_students p) / 2

-- Condition 2: She took 17 pens home
def home_pens : ℕ := 17

-- Main proof statement
theorem sally_pens_proof :
  (locker_pens p + home_pens = pens_after_giving_students p) → p = 7 :=
by
  sorry

end sally_pens_proof_l1773_177370


namespace largest_composite_sequence_l1773_177314

theorem largest_composite_sequence (a b c d e f g : ℕ) (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < e) (h₅ : e < f) (h₆ : f < g) 
  (h₇ : g < 50) (h₈ : a ≥ 10) (h₉ : g ≤ 32)
  (h₁₀ : ¬ Prime a) (h₁₁ : ¬ Prime b) (h₁₂ : ¬ Prime c) (h₁₃ : ¬ Prime d) 
  (h₁₄ : ¬ Prime e) (h₁₅ : ¬ Prime f) (h₁₆ : ¬ Prime g) :
  g = 32 :=
sorry

end largest_composite_sequence_l1773_177314


namespace value_of_a_star_b_l1773_177339

variable (a b : ℤ)

def operation_star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem value_of_a_star_b (h1 : a + b = 7) (h2 : a * b = 12) :
  operation_star a b = 7 / 12 := by
  sorry

end value_of_a_star_b_l1773_177339


namespace draw_white_ball_is_impossible_l1773_177392

-- Definitions based on the conditions
def redBalls : Nat := 2
def blackBalls : Nat := 6
def totalBalls : Nat := redBalls + blackBalls

-- Definition for the white ball drawing event
def whiteBallDraw (redBalls blackBalls : Nat) : Prop :=
  ∀ (n : Nat), n ≠ 0 → n ≤ redBalls + blackBalls → false

-- Theorem to prove the event is impossible
theorem draw_white_ball_is_impossible : whiteBallDraw redBalls blackBalls :=
  by
  sorry

end draw_white_ball_is_impossible_l1773_177392


namespace number_of_ensembles_sold_l1773_177379

-- Define the prices
def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earring_price : ℕ := 10
def ensemble_price : ℕ := 45

-- Define the quantities sold
def necklaces_sold : ℕ := 5
def bracelets_sold : ℕ := 10
def earrings_sold : ℕ := 20

-- Define the total income
def total_income : ℕ := 565

-- Define the function or theorem that determines the number of ensembles sold
theorem number_of_ensembles_sold : 
  (total_income = (necklaces_sold * necklace_price) + (bracelets_sold * bracelet_price) + (earrings_sold * earring_price) + (2 * ensemble_price)) :=
sorry

end number_of_ensembles_sold_l1773_177379


namespace f_odd_function_no_parallel_lines_l1773_177397

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - (1 / a^x))

theorem f_odd_function {a : ℝ} (h_pos : a > 0) (h_ne : a ≠ 1) : 
  ∀ x : ℝ, f a (-x) = -f a x := 
by
  sorry

theorem no_parallel_lines {a : ℝ} (h_pos : a > 0) (h_ne : a ≠ 1) : 
  ∀ x1 x2 : ℝ, x1 ≠ x2 → f a x1 ≠ f a x2 :=
by
  sorry

end f_odd_function_no_parallel_lines_l1773_177397


namespace rationalize_and_divide_l1773_177301

theorem rationalize_and_divide :
  (8 / Real.sqrt 8 / 2) = Real.sqrt 2 :=
by
  sorry

end rationalize_and_divide_l1773_177301


namespace boat_speed_in_still_water_l1773_177321

-- Identifying the speeds of the boat in still water and the stream
variables (b s : ℝ)

-- Conditions stated in terms of equations
axiom boat_along_stream : b + s = 7
axiom boat_against_stream : b - s = 5

-- Prove that the boat speed in still water is 6 km/hr
theorem boat_speed_in_still_water : b = 6 :=
by
  sorry

end boat_speed_in_still_water_l1773_177321


namespace distinguishable_arrangements_l1773_177374

theorem distinguishable_arrangements :
  let n := 9
  let n1 := 3
  let n2 := 2
  let n3 := 4
  (Nat.factorial n) / ((Nat.factorial n1) * (Nat.factorial n2) * (Nat.factorial n3)) = 1260 :=
by sorry

end distinguishable_arrangements_l1773_177374


namespace at_least_one_true_l1773_177367

-- Definitions (Conditions)
variables (p q : Prop)

-- Statement
theorem at_least_one_true (h : p ∨ q) : p ∨ q := by
  sorry

end at_least_one_true_l1773_177367


namespace find_ordered_pair_l1773_177345

theorem find_ordered_pair (s h : ℝ) :
  (∀ (u : ℝ), ∃ (x y : ℝ), x = s + 3 * u ∧ y = -3 + h * u ∧ y = 4 * x + 2) →
  (s, h) = (-5 / 4, 12) :=
by
  sorry

end find_ordered_pair_l1773_177345


namespace find_constants_l1773_177368

open Set

variable {α : Type*} [LinearOrderedField α]

def Set_1 : Set α := {x | x^2 - 3*x + 2 = 0}

def Set_2 (a : α) : Set α := {x | x^2 - a*x + (a-1) = 0}

def Set_3 (m : α) : Set α := {x | x^2 - m*x + 2 = 0}

theorem find_constants (a m : α) :
  (Set_1 ∪ Set_2 a = Set_1) ∧ (Set_1 ∩ Set_2 a = Set_3 m) → 
  a = 3 ∧ m = 3 :=
by sorry

end find_constants_l1773_177368


namespace number_of_disconnected_regions_l1773_177318

theorem number_of_disconnected_regions (n : ℕ) (h : 2 ≤ n) : 
  ∀ R : ℕ → ℕ, (R 1 = 2) → 
  (∀ k, R k = k^2 - k + 2 → R (k + 1) = (k + 1)^2 - (k + 1) + 2) → 
  R n = n^2 - n + 2 :=
sorry

end number_of_disconnected_regions_l1773_177318


namespace find_AD_l1773_177348

theorem find_AD
  (A B C D : Type)
  (BD BC CD AD : ℝ)
  (hBD : BD = 21)
  (hBC : BC = 30)
  (hCD : CD = 15)
  (hAngleBisect : true) -- Encode that D bisects the angle at C internally
  : AD = 35 := by
  sorry

end find_AD_l1773_177348


namespace minimum_boxes_cost_300_muffins_l1773_177361

theorem minimum_boxes_cost_300_muffins :
  ∃ (L_used M_used S_used : ℕ), 
    L_used + M_used + S_used = 28 ∧ 
    (L_used = 10 ∧ M_used = 15 ∧ S_used = 3) ∧ 
    (L_used * 15 + M_used * 9 + S_used * 5 = 300) ∧ 
    (L_used * 5 + M_used * 3 + S_used * 2 = 101) ∧ 
    (L_used ≤ 10 ∧ M_used ≤ 15 ∧ S_used ≤ 25) :=
by
  -- The proof is omitted (theorem statement only).
  sorry

end minimum_boxes_cost_300_muffins_l1773_177361


namespace range_of_t_range_of_a_l1773_177338

-- Proposition P: The curve equation represents an ellipse with foci on the x-axis
def propositionP (t : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (4 - t) + y^2 / (t - 1) = 1)

-- Proof problem for t
theorem range_of_t (t : ℝ) (h : propositionP t) : 1 < t ∧ t < 5 / 2 := 
  sorry

-- Proposition Q: The inequality involving real number t
def propositionQ (t a : ℝ) : Prop := t^2 - (a + 3) * t + (a + 2) < 0

-- Proof problem for a
theorem range_of_a (a : ℝ) (h₁ : ∀ t : ℝ, propositionP t → propositionQ t a) 
                   (h₂ : ∃ t : ℝ, propositionQ t a ∧ ¬ propositionP t) :
  a > 1 / 2 :=
  sorry

end range_of_t_range_of_a_l1773_177338


namespace amare_fabric_needed_l1773_177377

-- Definitions for the conditions
def fabric_per_dress_yards : ℝ := 5.5
def number_of_dresses : ℕ := 4
def fabric_owned_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- Total fabric needed in yards
def total_fabric_needed_yards : ℝ := fabric_per_dress_yards * number_of_dresses

-- Total fabric needed in feet
def total_fabric_needed_feet : ℝ := total_fabric_needed_yards * yard_to_feet

-- Fabric still needed
def fabric_still_needed : ℝ := total_fabric_needed_feet - fabric_owned_feet

-- Proof
theorem amare_fabric_needed : fabric_still_needed = 59 := by
  sorry

end amare_fabric_needed_l1773_177377


namespace pool_capacity_l1773_177324

-- Define the total capacity of the pool as a variable
variable (C : ℝ)

-- Define the conditions
def additional_water_needed (x : ℝ) : Prop :=
  x = 300

def increases_by_25_percent (x : ℝ) (y : ℝ) : Prop :=
  y = x * 0.25

-- State the proof problem
theorem pool_capacity :
  ∃ C : ℝ, additional_water_needed 300 ∧ increases_by_25_percent (0.75 * C) 300 ∧ C = 1200 :=
sorry

end pool_capacity_l1773_177324


namespace pos_int_solutions_l1773_177366

-- defining the condition for a positive integer solution to the equation
def is_pos_int_solution (x y : Int) : Prop :=
  5 * x + 2 * y = 25 ∧ x > 0 ∧ y > 0

-- stating the theorem for positive integer solutions of the equation
theorem pos_int_solutions : 
  ∃ x y : Int, is_pos_int_solution x y ∧ ((x = 1 ∧ y = 10) ∨ (x = 3 ∧ y = 5)) :=
by
  sorry

end pos_int_solutions_l1773_177366


namespace lumberjack_question_l1773_177343

def logs_per_tree (total_firewood : ℕ) (firewood_per_log : ℕ) (trees_chopped : ℕ) : ℕ :=
  total_firewood / firewood_per_log / trees_chopped

theorem lumberjack_question : logs_per_tree 500 5 25 = 4 := by
  sorry

end lumberjack_question_l1773_177343


namespace eval_exponents_l1773_177396

theorem eval_exponents : (2^3)^2 - 4^3 = 0 := by
  sorry

end eval_exponents_l1773_177396


namespace ferry_P_travel_time_l1773_177394

-- Define the conditions based on the problem statement
variables (t : ℝ) -- travel time of ferry P
def speed_P := 6 -- speed of ferry P in km/h
def speed_Q := speed_P + 3 -- speed of ferry Q in km/h
def distance_P := speed_P * t -- distance traveled by ferry P in km
def distance_Q := 3 * distance_P -- distance traveled by ferry Q in km
def time_Q := t + 3 -- travel time of ferry Q

-- Theorem to prove that travel time t for ferry P is 3 hours
theorem ferry_P_travel_time : time_Q * speed_Q = distance_Q → t = 3 :=
by {
  -- Since you've mentioned to include the statement only and not the proof,
  -- Therefore, the proof body is left as an exercise or represented by sorry.
  sorry
}

end ferry_P_travel_time_l1773_177394


namespace add_coefficients_l1773_177384

theorem add_coefficients (a : ℕ) : 2 * a + a = 3 * a :=
by 
  sorry

end add_coefficients_l1773_177384


namespace JillTotalTaxPercentage_l1773_177362

noncomputable def totalTaxPercentage : ℝ :=
  let totalSpending (beforeDiscount : ℝ) : ℝ := 100
  let clothingBeforeDiscount : ℝ := 0.4 * totalSpending 100
  let foodBeforeDiscount : ℝ := 0.2 * totalSpending 100
  let electronicsBeforeDiscount : ℝ := 0.1 * totalSpending 100
  let cosmeticsBeforeDiscount : ℝ := 0.2 * totalSpending 100
  let householdBeforeDiscount : ℝ := 0.1 * totalSpending 100

  let clothingDiscount : ℝ := 0.1 * clothingBeforeDiscount
  let foodDiscount : ℝ := 0.05 * foodBeforeDiscount
  let electronicsDiscount : ℝ := 0.15 * electronicsBeforeDiscount

  let clothingAfterDiscount := clothingBeforeDiscount - clothingDiscount
  let foodAfterDiscount := foodBeforeDiscount - foodDiscount
  let electronicsAfterDiscount := electronicsBeforeDiscount - electronicsDiscount
  
  let taxOnClothing := 0.06 * clothingAfterDiscount
  let taxOnFood := 0.0 * foodAfterDiscount
  let taxOnElectronics := 0.1 * electronicsAfterDiscount
  let taxOnCosmetics := 0.08 * cosmeticsBeforeDiscount
  let taxOnHousehold := 0.04 * householdBeforeDiscount

  let totalTaxPaid := taxOnClothing + taxOnFood + taxOnElectronics + taxOnCosmetics + taxOnHousehold
  (totalTaxPaid / totalSpending 100) * 100

theorem JillTotalTaxPercentage :
  totalTaxPercentage = 5.01 := by
  sorry

end JillTotalTaxPercentage_l1773_177362


namespace female_democrats_count_l1773_177300

theorem female_democrats_count :
  ∃ (F : ℕ) (M : ℕ),
    F + M = 750 ∧
    (F / 2) + (M / 4) = 250 ∧
    1 / 3 * 750 = 250 ∧
    F / 2 = 125 := sorry

end female_democrats_count_l1773_177300


namespace expression_evaluation_l1773_177320

theorem expression_evaluation :
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1 / 3) + Real.sqrt 48) / (2 * Real.sqrt 3) + (Real.sqrt (1 / 3))^2 = 5 :=
by
  sorry

end expression_evaluation_l1773_177320


namespace students_not_A_either_l1773_177358

-- Given conditions as definitions
def total_students : ℕ := 40
def students_A_history : ℕ := 10
def students_A_math : ℕ := 18
def students_A_both : ℕ := 6

-- Statement to prove
theorem students_not_A_either : (total_students - (students_A_history + students_A_math - students_A_both)) = 18 := 
by
  sorry

end students_not_A_either_l1773_177358


namespace solution_set_of_inequality_l1773_177393

-- Define conditions
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- Lean statement of the proof problem
theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_mono_inc : is_monotonically_increasing_on f {x | x ≤ 0}) :
  { x : ℝ | f (3 - 2 * x) > f (1) } = { x : ℝ | 1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_of_inequality_l1773_177393


namespace book_configurations_l1773_177347

theorem book_configurations : 
  (∃ (configurations : Finset ℕ), configurations = {1, 2, 3, 4, 5, 6, 7} ∧ configurations.card = 7) 
  ↔ 
  (∃ (n : ℕ), n = 7) :=
by 
  sorry

end book_configurations_l1773_177347


namespace Michael_rides_six_miles_l1773_177385

theorem Michael_rides_six_miles
  (rate : ℝ)
  (time : ℝ)
  (interval_time : ℝ)
  (interval_distance : ℝ)
  (intervals : ℝ)
  (total_distance : ℝ) :
  rate = 1.5 ∧ time = 40 ∧ interval_time = 10 ∧ interval_distance = 1.5 ∧ intervals = time / interval_time ∧ total_distance = intervals * interval_distance →
  total_distance = 6 :=
by
  intros h
  -- Placeholder for the proof
  sorry

end Michael_rides_six_miles_l1773_177385


namespace general_term_of_sequence_l1773_177359

theorem general_term_of_sequence (a : ℕ → ℝ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = (a n) ^ 2) :
  ∀ n : ℕ, n > 0 → a n = 3 ^ (2 ^ (n - 1)) :=
by
  intros n hn
  sorry

end general_term_of_sequence_l1773_177359


namespace find_original_cost_price_l1773_177331

variables (P : ℝ) (A B C D E : ℝ)

-- Define the conditions as per the problem statement
def with_tax (P : ℝ) : ℝ := P * 1.10
def profit_60 (price : ℝ) : ℝ := price * 1.60
def profit_25 (price : ℝ) : ℝ := price * 1.25
def loss_15 (price : ℝ) : ℝ := price * 0.85
def profit_30 (price : ℝ) : ℝ := price * 1.30

-- The final price E is given.
def final_price (P : ℝ) : ℝ :=
  profit_30 
  (loss_15 
  (profit_25 
  (profit_60 
  (with_tax P))))

-- To find original cost price P given final price of Rs. 500.
theorem find_original_cost_price (h : final_price P = 500) : 
  P = 500 / 2.431 :=
by 
  sorry

end find_original_cost_price_l1773_177331


namespace find_value_of_a_perpendicular_lines_l1773_177329

theorem find_value_of_a_perpendicular_lines :
  ∃ (a : ℝ), (∀ (x y : ℝ), y = a * x - 2 → y = 2 * x + 1 → 
  (a * 2 = -1)) → a = -1/2 :=
by
  sorry

end find_value_of_a_perpendicular_lines_l1773_177329


namespace constant_term_expansion_l1773_177305

-- auxiliary definitions and facts
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def term_constant (n k : ℕ) (a b x : ℂ) : ℂ :=
  binomial_coeff n k * (a * x)^(n-k) * (b / x)^k

-- main theorem statement
theorem constant_term_expansion : ∀ (x : ℂ), (term_constant 8 4 (5 : ℂ) (2 : ℂ) x).re = 1120 :=
by
  intro x
  sorry

end constant_term_expansion_l1773_177305


namespace div_by_64_l1773_177304

theorem div_by_64 (n : ℕ) (h : n ≥ 1) : 64 ∣ (3^(2*n + 2) - 8*n - 9) :=
sorry

end div_by_64_l1773_177304


namespace restaurant_table_difference_l1773_177315

theorem restaurant_table_difference :
  ∃ (N O : ℕ), N + O = 40 ∧ 6 * N + 4 * O = 212 ∧ (N - O) = 12 :=
by
  sorry

end restaurant_table_difference_l1773_177315


namespace geometric_sequence_l1773_177327

theorem geometric_sequence (a b c r : ℤ) (h1 : b = a * r) (h2 : c = a * r^2) (h3 : c = a + 56) : b = 21 :=
by sorry

end geometric_sequence_l1773_177327


namespace geom_seq_sum_a3_a4_a5_l1773_177369

-- Define the geometric sequence terms and sum condition
def geometric_seq (a1 q : ℕ) (n : ℕ) : ℕ :=
  a1 * q^(n - 1)

def sum_first_three (a1 q : ℕ) : ℕ :=
  a1 + a1 * q + a1 * q^2

-- Given conditions
def a1 : ℕ := 3
def S3 : ℕ := 21

-- Define the problem statement
theorem geom_seq_sum_a3_a4_a5 (q : ℕ) (h : sum_first_three a1 q = S3) (h_pos : ∀ n, geometric_seq a1 q n > 0) :
  geometric_seq a1 q 3 + geometric_seq a1 q 4 + geometric_seq a1 q 5 = 84 :=
by sorry

end geom_seq_sum_a3_a4_a5_l1773_177369


namespace average_percentage_reduction_l1773_177341

theorem average_percentage_reduction (x : ℝ) (hx : 0 < x ∧ x < 1)
  (initial_price final_price : ℝ)
  (h_initial : initial_price = 25)
  (h_final : final_price = 16)
  (h_reduction : final_price = initial_price * (1-x)^2) :
  x = 0.2 :=
by {
  --". Convert fraction \( = x / y \)", proof is omitted
  sorry
}

end average_percentage_reduction_l1773_177341


namespace population_reduction_l1773_177311

theorem population_reduction (initial_population : ℕ) (final_population : ℕ) (left_percentage : ℝ)
    (bombardment_percentage : ℝ) :
    initial_population = 7145 →
    final_population = 4555 →
    left_percentage = 0.75 →
    bombardment_percentage = 100 - 84.96 →
    ∃ (x : ℝ), bombardment_percentage = (100 - x) := 
by
    sorry

end population_reduction_l1773_177311


namespace sum_of_cubes_l1773_177332

theorem sum_of_cubes {a b c : ℝ} (h1 : a + b + c = 5) (h2 : a * b + a * c + b * c = 7) (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 29 :=
by
  -- The proof part is intentionally left out.
  sorry

end sum_of_cubes_l1773_177332


namespace anna_and_bob_play_together_l1773_177381

-- Definitions based on the conditions
def total_players := 12
def matches_per_week := 2
def players_per_match := 6
def anna_and_bob := 2
def other_players := total_players - anna_and_bob
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Lean statement based on the equivalent proof problem
theorem anna_and_bob_play_together :
  combination other_players (players_per_match - anna_and_bob) = 210 := by
  -- To use Binomial Theorem in Lean
  -- The mathematical equivalent is C(10, 4) = 210
  sorry

end anna_and_bob_play_together_l1773_177381


namespace expression_value_l1773_177378

theorem expression_value (x y z : ℤ) (hx : x = 26) (hy : y = 3 * x / 2) (hz : z = 11) :
  x - (y - z) - ((x - y) - z) = 22 := 
by
  -- problem statement here
  -- simplified proof goes here
  sorry

end expression_value_l1773_177378


namespace exists_product_sum_20000_l1773_177340

theorem exists_product_sum_20000 :
  ∃ (k m : ℕ), 1 ≤ k ∧ k ≤ 999 ∧ 1 ≤ m ∧ m ≤ 999 ∧ k * (k + 1) + m * (m + 1) = 20000 :=
by 
  sorry

end exists_product_sum_20000_l1773_177340


namespace costForFirstKgs_l1773_177371

noncomputable def applePrice (l : ℝ) (q : ℝ) (x : ℝ) (totalWeight : ℝ) : ℝ :=
  if totalWeight <= x then l * totalWeight else l * x + q * (totalWeight - x)

theorem costForFirstKgs (l q x : ℝ) :
  l = 10 ∧ q = 11 ∧ (applePrice l q x 33 = 333) ∧ (applePrice l q x 36 = 366) ∧ (applePrice l q 15 15 = 150) → x = 30 := 
by
  sorry

end costForFirstKgs_l1773_177371


namespace no_four_digit_numbers_divisible_by_11_l1773_177307

theorem no_four_digit_numbers_divisible_by_11 (a b c d : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) 
(h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : 0 ≤ c) (h₆ : c ≤ 9) (h₇ : 0 ≤ d) (h₈ : d ≤ 9) 
(h₉ : a + b + c + d = 10) (h₁₀ : a + c = b + d) : 
0 = 0 :=
sorry

end no_four_digit_numbers_divisible_by_11_l1773_177307


namespace boat_speed_in_still_water_l1773_177328

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 :=
by 
  sorry

end boat_speed_in_still_water_l1773_177328


namespace mike_practices_hours_on_saturday_l1773_177313

-- Definitions based on conditions
def weekday_hours : ℕ := 3
def weekdays_per_week : ℕ := 5
def total_hours : ℕ := 60
def weeks : ℕ := 3

def calculate_total_weekday_hours (weekday_hours weekdays_per_week weeks : ℕ) : ℕ :=
  weekday_hours * weekdays_per_week * weeks

def calculate_saturday_hours (total_hours total_weekday_hours weeks : ℕ) : ℕ :=
  (total_hours - total_weekday_hours) / weeks

-- Statement to prove
theorem mike_practices_hours_on_saturday :
  calculate_saturday_hours total_hours (calculate_total_weekday_hours weekday_hours weekdays_per_week weeks) weeks = 5 :=
by 
  sorry

end mike_practices_hours_on_saturday_l1773_177313


namespace simplify_and_evaluate_expr_l1773_177344

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3) - x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  rw [h]
  sorry

end simplify_and_evaluate_expr_l1773_177344


namespace exists_consecutive_numbers_with_prime_divisors_l1773_177390

theorem exists_consecutive_numbers_with_prime_divisors (p q : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p < q ∧ q < 2 * p) :
  ∃ n m : ℕ, (m = n + 1) ∧ 
             (Nat.gcd n p = p) ∧ (Nat.gcd m p = 1) ∧ 
             (Nat.gcd m q = q) ∧ (Nat.gcd n q = 1) :=
by
  sorry

end exists_consecutive_numbers_with_prime_divisors_l1773_177390
