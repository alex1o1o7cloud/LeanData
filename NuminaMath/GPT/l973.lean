import Mathlib

namespace NUMINAMATH_GPT_green_shirt_pairs_l973_97329

theorem green_shirt_pairs (r g : ℕ) (p total_pairs red_pairs : ℕ) :
  r = 63 → g = 69 → p = 66 → red_pairs = 25 → (g - (r - red_pairs * 2)) / 2 = 28 :=
by
  intros hr hg hp hred_pairs
  sorry

end NUMINAMATH_GPT_green_shirt_pairs_l973_97329


namespace NUMINAMATH_GPT_max_value_inequality_l973_97379

theorem max_value_inequality : 
  ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) (m : ℝ),
  (∀ n, S_n n = (n * a_n 1 + (1 / 2) * n * (n - 1) * d) ∧
  (∀ n, a_n n ^ 2 + (S_n n ^ 2 / n ^ 2) >= m * (a_n 1) ^ 2)) → 
  m ≤ 1 / 5 := 
sorry

end NUMINAMATH_GPT_max_value_inequality_l973_97379


namespace NUMINAMATH_GPT_count_multiples_12_9_l973_97309

theorem count_multiples_12_9 :
  ∃ n : ℕ, n = 8 ∧ (∀ x : ℕ, x % 36 = 0 ∧ 200 ≤ x ∧ x ≤ 500 ↔ ∃ y : ℕ, (x = 36 * y ∧ 200 ≤ 36 * y ∧ 36 * y ≤ 500)) :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_12_9_l973_97309


namespace NUMINAMATH_GPT_percent_of_y_l973_97366

theorem percent_of_y (y : ℝ) : 0.30 * (0.80 * y) = 0.24 * y :=
by sorry

end NUMINAMATH_GPT_percent_of_y_l973_97366


namespace NUMINAMATH_GPT_sum_of_three_numbers_l973_97373

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : a * b + b * c + c * a = 100) : 
  a + b + c = 21 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l973_97373


namespace NUMINAMATH_GPT_concert_parking_fee_l973_97367

theorem concert_parking_fee :
  let ticket_cost := 50 
  let processing_fee_percentage := 0.15 
  let entrance_fee_per_person := 5 
  let total_cost_concert := 135
  let num_people := 2 

  let total_ticket_cost := ticket_cost * num_people
  let processing_fee := total_ticket_cost * processing_fee_percentage
  let total_ticktet_cost_with_fee := total_ticket_cost + processing_fee
  let total_entrance_fee := entrance_fee_per_person * num_people
  let total_cost_without_parking := total_ticktet_cost_with_fee + total_entrance_fee
  total_cost_concert - total_cost_without_parking = 10 := by 
  sorry

end NUMINAMATH_GPT_concert_parking_fee_l973_97367


namespace NUMINAMATH_GPT_cody_needs_total_steps_l973_97355

theorem cody_needs_total_steps 
  (weekly_steps : ℕ → ℕ)
  (h1 : ∀ n, weekly_steps n = (n + 1) * 1000 * 7)
  (h2 : 4 * 7 * 1000 + 3 * 7 * 1000 + 2 * 7 * 1000 + 1 * 7 * 1000 = 70000) 
  (h3 : 70000 + 30000 = 100000) :
  ∃ total_steps, total_steps = 100000 := 
by
  sorry

end NUMINAMATH_GPT_cody_needs_total_steps_l973_97355


namespace NUMINAMATH_GPT_max_car_passing_400_l973_97360

noncomputable def max_cars_passing (speed : ℕ) (car_length : ℤ) (hour : ℕ) : ℕ :=
  20000 * speed / (5 * (speed + 1))

theorem max_car_passing_400 :
  max_cars_passing 20 5 1 / 10 = 400 := by
  sorry

end NUMINAMATH_GPT_max_car_passing_400_l973_97360


namespace NUMINAMATH_GPT_juggling_contest_l973_97303

theorem juggling_contest (B : ℕ) (rot_baseball : ℕ := 80)
    (rot_per_apple : ℕ := 101) (num_apples : ℕ := 4)
    (winner_rotations : ℕ := 404) :
    (num_apples * rot_per_apple = winner_rotations) :=
by
  sorry

end NUMINAMATH_GPT_juggling_contest_l973_97303


namespace NUMINAMATH_GPT_sqrt_5th_of_x_sqrt_4th_x_l973_97311

theorem sqrt_5th_of_x_sqrt_4th_x (x : ℝ) (hx : 0 < x) : Real.sqrt (x * Real.sqrt (x ^ (1 / 4))) = x ^ (1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_5th_of_x_sqrt_4th_x_l973_97311


namespace NUMINAMATH_GPT_hotel_charge_l973_97372

variable (R G P : ℝ)

theorem hotel_charge (h1 : P = 0.60 * R) (h2 : P = 0.90 * G) : (R - G) / G = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_hotel_charge_l973_97372


namespace NUMINAMATH_GPT_range_of_a_l973_97323

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) → (-1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l973_97323


namespace NUMINAMATH_GPT_solve_for_x_l973_97374

theorem solve_for_x (x : ℝ) (h : x ≠ 0) (h_eq : (8 * x) ^ 16 = (32 * x) ^ 8) : x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l973_97374


namespace NUMINAMATH_GPT_find_ac_bd_l973_97352

variable (a b c d : ℝ)

axiom cond1 : a^2 + b^2 = 1
axiom cond2 : c^2 + d^2 = 1
axiom cond3 : a * d - b * c = 1 / 7

theorem find_ac_bd : a * c + b * d = 4 * Real.sqrt 3 / 7 := by
  sorry

end NUMINAMATH_GPT_find_ac_bd_l973_97352


namespace NUMINAMATH_GPT_monotone_decreasing_interval_3_l973_97378

variable {f : ℝ → ℝ}

theorem monotone_decreasing_interval_3 
  (h1 : ∀ x, f (x + 3) = f (x - 3))
  (h2 : ∀ x, f (x + 3) = f (-x + 3))
  (h3 : ∀ ⦃x y⦄, 0 < x → x < 3 → 0 < y → y < 3 → x < y → f y < f x) :
  f 3.5 < f (-4.5) ∧ f (-4.5) < f 12.5 :=
sorry

end NUMINAMATH_GPT_monotone_decreasing_interval_3_l973_97378


namespace NUMINAMATH_GPT_smallest_k_for_ten_ruble_heads_up_l973_97391

-- Conditions
def num_total_coins : ℕ := 30
def num_ten_ruble_coins : ℕ := 23
def num_five_ruble_coins : ℕ := 7
def num_heads_up : ℕ := 20
def num_tails_up : ℕ := 10

-- Prove the smallest k such that any k coins chosen include at least one ten-ruble coin heads-up.
theorem smallest_k_for_ten_ruble_heads_up (k : ℕ) :
  (∀ (coins : Finset ℕ), coins.card = k → (∃ (coin : ℕ) (h : coin ∈ coins), coin < num_ten_ruble_coins ∧ coin < num_heads_up)) →
  k = 18 :=
sorry

end NUMINAMATH_GPT_smallest_k_for_ten_ruble_heads_up_l973_97391


namespace NUMINAMATH_GPT_problem_l973_97334

open Real

theorem problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (1 / a) + (4 / b) + (9 / c) ≤ 36 / (a + b + c)) 
  : (2 * b + 3 * c) / (a + b + c) = 13 / 6 :=
sorry

end NUMINAMATH_GPT_problem_l973_97334


namespace NUMINAMATH_GPT_perpendicular_lines_l973_97350

theorem perpendicular_lines (a : ℝ) : 
  (a = -1 → (∀ x y : ℝ, 4 * x - (a + 1) * y + 9 = 0 → x ≠ 0 →  y ≠ 0 → 
  ∃ b : ℝ, (b^2 + 1) * x - b * y + 6 = 0)) ∧ 
  (∀ x y : ℝ, (4 * x - (a + 1) * y + 9 = 0) ∧ (∃ x y : ℝ, (a^2 - 1) * x - a * y + 6 = 0) → a ≠ -1) := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_l973_97350


namespace NUMINAMATH_GPT_central_angle_of_probability_l973_97330

theorem central_angle_of_probability (x : ℝ) (h1 : x / 360 = 1 / 6) : x = 60 := by
  have h2 : x = 60 := by
    linarith
  exact h2

end NUMINAMATH_GPT_central_angle_of_probability_l973_97330


namespace NUMINAMATH_GPT_Gretchen_weekend_profit_l973_97388

theorem Gretchen_weekend_profit :
  let saturday_revenue := 24 * 25
  let sunday_revenue := 16 * 15
  let total_revenue := saturday_revenue + sunday_revenue
  let park_fee := 5 * 6 * 2
  let art_supplies_cost := 8 * 2
  let total_expenses := park_fee + art_supplies_cost
  let profit := total_revenue - total_expenses
  profit = 764 :=
by
  sorry

end NUMINAMATH_GPT_Gretchen_weekend_profit_l973_97388


namespace NUMINAMATH_GPT_range_of_m_l973_97394

-- Defining the quadratic function with the given condition
def quadratic (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-1)*x + 2

-- Stating the problem
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, quadratic m x > 0) ↔ 1 ≤ m ∧ m < 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l973_97394


namespace NUMINAMATH_GPT_correct_system_of_equations_l973_97348

theorem correct_system_of_equations (x y : ℝ) 
  (h1 : 3 * x = 5 * y - 6)
  (h2 : y = 2 * x - 10) : 
  (3 * x = 5 * y - 6) ∧ (y = 2 * x - 10) :=
by
  sorry

end NUMINAMATH_GPT_correct_system_of_equations_l973_97348


namespace NUMINAMATH_GPT_find_rate_of_interest_l973_97304

-- Conditions
def principal : ℕ := 4200
def time : ℕ := 2
def interest_12 : ℕ := principal * 12 * time / 100
def additional_interest : ℕ := 504
def total_interest_r : ℕ := interest_12 + additional_interest

-- Theorem Statement
theorem find_rate_of_interest (r : ℕ) (h : 1512 = principal * r * time / 100) : r = 18 :=
by sorry

end NUMINAMATH_GPT_find_rate_of_interest_l973_97304


namespace NUMINAMATH_GPT_sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3_sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2_l973_97332

-- Problem 1
theorem sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3 : |Real.sqrt 3 - Real.sqrt 2| + Real.sqrt 2 = Real.sqrt 3 := by
  sorry

-- Problem 2
theorem sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2 : Real.sqrt 2 * (Real.sqrt 2 + 2) = 2 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3_sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2_l973_97332


namespace NUMINAMATH_GPT_half_angle_in_second_quadrant_l973_97316

theorem half_angle_in_second_quadrant (α : Real) (h1 : 180 < α ∧ α < 270)
        (h2 : |Real.cos (α / 2)| = -Real.cos (α / 2)) :
        90 < α / 2 ∧ α / 2 < 180 :=
sorry

end NUMINAMATH_GPT_half_angle_in_second_quadrant_l973_97316


namespace NUMINAMATH_GPT_divisibility_l973_97326

theorem divisibility (a : ℤ) : (5 ∣ a^3) ↔ (5 ∣ a) := 
by sorry

end NUMINAMATH_GPT_divisibility_l973_97326


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l973_97319

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h : Real.tan α = (1 + Real.sin β) / Real.cos β)

theorem trigonometric_identity_proof : 2 * α - β = π / 2 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l973_97319


namespace NUMINAMATH_GPT_fraction_eaten_on_third_day_l973_97365

theorem fraction_eaten_on_third_day
  (total_pieces : ℕ)
  (first_day_fraction : ℚ)
  (second_day_fraction : ℚ)
  (remaining_after_third_day : ℕ)
  (initial_pieces : total_pieces = 200)
  (first_day_eaten : first_day_fraction = 1/4)
  (second_day_eaten : second_day_fraction = 2/5)
  (remaining_bread_after_third_day : remaining_after_third_day = 45) :
  (1 : ℚ) / 2 = 1/2 := sorry

end NUMINAMATH_GPT_fraction_eaten_on_third_day_l973_97365


namespace NUMINAMATH_GPT_calculate_expression_l973_97362

theorem calculate_expression : ((-1 + 2) * 3 + 2^2 / (-4)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l973_97362


namespace NUMINAMATH_GPT_unique_solution_of_system_l973_97395

theorem unique_solution_of_system :
  ∃! (x y z : ℝ), x + y = 2 ∧ xy - z^2 = 1 ∧ x = 1 ∧ y = 1 ∧ z = 0 := by
  sorry

end NUMINAMATH_GPT_unique_solution_of_system_l973_97395


namespace NUMINAMATH_GPT_vector_subtraction_l973_97340

theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (3, 5)) (h2 : b = (-2, 1)) :
  a - (2 : ℝ) • b = (7, 3) :=
by
  rw [h1, h2]
  simp
  sorry

end NUMINAMATH_GPT_vector_subtraction_l973_97340


namespace NUMINAMATH_GPT_coffee_shop_brewed_cups_in_week_l973_97384

theorem coffee_shop_brewed_cups_in_week 
    (weekday_rate : ℕ) (weekend_rate : ℕ)
    (weekday_hours : ℕ) (saturday_hours : ℕ) (sunday_hours : ℕ)
    (num_weekdays : ℕ) (num_saturdays : ℕ) (num_sundays : ℕ)
    (h1 : weekday_rate = 10)
    (h2 : weekend_rate = 15)
    (h3 : weekday_hours = 5)
    (h4 : saturday_hours = 6)
    (h5 : sunday_hours = 4)
    (h6 : num_weekdays = 5)
    (h7 : num_saturdays = 1)
    (h8 : num_sundays = 1) :
    (weekday_rate * weekday_hours * num_weekdays) + 
    (weekend_rate * saturday_hours * num_saturdays) + 
    (weekend_rate * sunday_hours * num_sundays) = 400 := 
by
  sorry

end NUMINAMATH_GPT_coffee_shop_brewed_cups_in_week_l973_97384


namespace NUMINAMATH_GPT_incorrect_operation_D_l973_97344

theorem incorrect_operation_D (x y: ℝ) : ¬ (-2 * x * (x - y) = -2 * x^2 - 2 * x * y) :=
by sorry

end NUMINAMATH_GPT_incorrect_operation_D_l973_97344


namespace NUMINAMATH_GPT_total_shaded_area_correct_l973_97353
-- Let's import the mathematical library.

-- Define the problem-related conditions.
def first_rectangle_length : ℕ := 4
def first_rectangle_width : ℕ := 15
def second_rectangle_length : ℕ := 5
def second_rectangle_width : ℕ := 12
def third_rectangle_length : ℕ := 2
def third_rectangle_width : ℕ := 2

-- Define the areas based on the problem conditions.
def A1 : ℕ := first_rectangle_length * first_rectangle_width
def A2 : ℕ := second_rectangle_length * second_rectangle_width
def A_overlap_12 : ℕ := first_rectangle_length * second_rectangle_length
def A3 : ℕ := third_rectangle_length * third_rectangle_width

-- Define the total shaded area formula.
def total_shaded_area : ℕ := A1 + A2 - A_overlap_12 + A3

-- Statement of the theorem to prove.
theorem total_shaded_area_correct :
  total_shaded_area = 104 :=
by
  sorry

end NUMINAMATH_GPT_total_shaded_area_correct_l973_97353


namespace NUMINAMATH_GPT_expression_value_l973_97343

theorem expression_value (x y z : ℤ) (h1: x = 2) (h2: y = -3) (h3: z = 1) :
  x^2 + y^2 - 2*z^2 + 3*x*y = -7 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l973_97343


namespace NUMINAMATH_GPT_total_trees_in_forest_l973_97308

theorem total_trees_in_forest (a_street : ℕ) (a_forest : ℕ) 
                              (side_length : ℕ) (trees_per_square_meter : ℕ)
                              (h1 : a_street = side_length * side_length)
                              (h2 : a_forest = 3 * a_street)
                              (h3 : side_length = 100)
                              (h4 : trees_per_square_meter = 4) :
                              a_forest * trees_per_square_meter = 120000 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_trees_in_forest_l973_97308


namespace NUMINAMATH_GPT_find_unknown_number_l973_97396

theorem find_unknown_number (x : ℝ) : 
  (1000 * 7) / (x * 17) = 10000 → x = 24.285714285714286 := by
  sorry

end NUMINAMATH_GPT_find_unknown_number_l973_97396


namespace NUMINAMATH_GPT_contrapositive_x_squared_eq_one_l973_97328

theorem contrapositive_x_squared_eq_one (x : ℝ) : 
  (x^2 = 1 → x = 1 ∨ x = -1) ↔ (x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) := by
  sorry

end NUMINAMATH_GPT_contrapositive_x_squared_eq_one_l973_97328


namespace NUMINAMATH_GPT_perfect_square_trinomial_l973_97320

theorem perfect_square_trinomial (m x : ℝ) : 
  ∃ a b : ℝ, (4 * x^2 + (m - 3) * x + 1 = (a + b)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l973_97320


namespace NUMINAMATH_GPT_anna_plants_needed_l973_97305

def required_salads : ℕ := 12
def salads_per_plant : ℕ := 3
def loss_fraction : ℚ := 1 / 2

theorem anna_plants_needed : 
  ∀ (plants_needed : ℕ), 
  plants_needed = Nat.ceil (required_salads / salads_per_plant * (1 / (1 - (loss_fraction : ℚ)))) :=
by
  sorry

end NUMINAMATH_GPT_anna_plants_needed_l973_97305


namespace NUMINAMATH_GPT_f_2_eq_1_l973_97331

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 1

theorem f_2_eq_1 (a b : ℝ) (h : f a b (-2) = 1) : f a b 2 = 1 :=
by {
  sorry 
}

end NUMINAMATH_GPT_f_2_eq_1_l973_97331


namespace NUMINAMATH_GPT_total_apples_l973_97347

/-- Problem: 
A fruit stand is selling apples for $2 each. Emmy has $200 while Gerry has $100. 
Prove the total number of apples Emmy and Gerry can buy altogether is 150.
-/
theorem total_apples (p E G : ℕ) (h1: p = 2) (h2: E = 200) (h3: G = 100) : 
  (E / p) + (G / p) = 150 :=
by
  sorry

end NUMINAMATH_GPT_total_apples_l973_97347


namespace NUMINAMATH_GPT_triangles_congruent_alternative_condition_l973_97321

theorem triangles_congruent_alternative_condition
  (A B C A' B' C' : Type)
  (AB A'B' AC A'C' : ℝ)
  (angleA angleA' : ℝ)
  (h1 : AB = A'B')
  (h2 : angleA = angleA')
  (h3 : AC = A'C') :
  ∃ (triangleABC triangleA'B'C' : Type), (triangleABC = triangleA'B'C') :=
by sorry

end NUMINAMATH_GPT_triangles_congruent_alternative_condition_l973_97321


namespace NUMINAMATH_GPT_smallest_n_l973_97300

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 5 * n = k1^2) (h2 : ∃ k2, 7 * n = k2^3) : n = 245 :=
sorry

end NUMINAMATH_GPT_smallest_n_l973_97300


namespace NUMINAMATH_GPT_coin_flip_sequences_count_l973_97346

theorem coin_flip_sequences_count : (2 ^ 16) = 65536 :=
by
  sorry

end NUMINAMATH_GPT_coin_flip_sequences_count_l973_97346


namespace NUMINAMATH_GPT_combin_sum_l973_97371

def combin (n m : ℕ) : ℕ := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem combin_sum (n : ℕ) (h₁ : n = 99) : combin n 2 + combin n 3 = 161700 := by
  sorry

end NUMINAMATH_GPT_combin_sum_l973_97371


namespace NUMINAMATH_GPT_janice_overtime_shifts_l973_97382

theorem janice_overtime_shifts (x : ℕ) (h1 : 5 * 30 + 15 * x = 195) : x = 3 :=
by
  -- leaving the proof unfinished, as asked
  sorry

end NUMINAMATH_GPT_janice_overtime_shifts_l973_97382


namespace NUMINAMATH_GPT_maximum_value_fraction_sum_l973_97325

theorem maximum_value_fraction_sum (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : 0 < c) (hd : 0 < d) (h1 : a + c = 20) (h2 : (a : ℝ) / b + (c : ℝ) / d < 1) :
  (a : ℝ) / b + (c : ℝ) / d ≤ 1385 / 1386 :=
sorry

end NUMINAMATH_GPT_maximum_value_fraction_sum_l973_97325


namespace NUMINAMATH_GPT_num_adult_tickets_is_35_l973_97338

noncomputable def num_adult_tickets_sold (A C: ℕ): Prop :=
  A + C = 85 ∧ 5 * A + 2 * C = 275

theorem num_adult_tickets_is_35: ∃ A C: ℕ, num_adult_tickets_sold A C ∧ A = 35 :=
by
  -- Definitions based on the provided conditions
  sorry

end NUMINAMATH_GPT_num_adult_tickets_is_35_l973_97338


namespace NUMINAMATH_GPT_inverse_proportion_relationship_l973_97381

theorem inverse_proportion_relationship (k : ℝ) (y1 y2 y3 : ℝ) :
  y1 = (k^2 + 1) / -1 →
  y2 = (k^2 + 1) / 1 →
  y3 = (k^2 + 1) / 2 →
  y1 < y3 ∧ y3 < y2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_inverse_proportion_relationship_l973_97381


namespace NUMINAMATH_GPT_goods_train_cross_platform_time_l973_97302

noncomputable def time_to_cross_platform (speed_kmph : ℝ) (length_train : ℝ) (length_platform : ℝ) : ℝ :=
  let speed_mps : ℝ := speed_kmph * (1000 / 3600)
  let total_distance : ℝ := length_train + length_platform
  total_distance / speed_mps

theorem goods_train_cross_platform_time :
  time_to_cross_platform 72 290.04 230 = 26.002 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_goods_train_cross_platform_time_l973_97302


namespace NUMINAMATH_GPT_car_arrives_before_bus_l973_97310

theorem car_arrives_before_bus
  (d : ℝ) (s_bus : ℝ) (s_car : ℝ) (v : ℝ)
  (h1 : d = 240)
  (h2 : s_bus = 40)
  (h3 : s_car = v)
  : 56 < v ∧ v < 120 := 
sorry

end NUMINAMATH_GPT_car_arrives_before_bus_l973_97310


namespace NUMINAMATH_GPT_marble_catch_up_time_l973_97386

theorem marble_catch_up_time 
    (a b c : ℝ) 
    (L : ℝ)
    (h1 : a - b = L / 50)
    (h2 : a - c = L / 40) 
    : (110 * (c - b)) / (c - b) = 110 := 
by 
    sorry

end NUMINAMATH_GPT_marble_catch_up_time_l973_97386


namespace NUMINAMATH_GPT_original_flour_quantity_l973_97376

-- Definitions based on conditions
def flour_called (x : ℝ) : Prop := 
  -- total flour Mary uses is x + extra 2 cups, which equals to 9 cups.
  x + 2 = 9

-- The proof statement we need to show
theorem original_flour_quantity : ∃ x : ℝ, flour_called x ∧ x = 7 := 
  sorry

end NUMINAMATH_GPT_original_flour_quantity_l973_97376


namespace NUMINAMATH_GPT_Eve_age_l973_97339

theorem Eve_age (Adam_age : ℕ) (Eve_age : ℕ) (h1 : Adam_age = 9) (h2 : Eve_age = Adam_age + 5)
  (h3 : ∃ k : ℕ, Eve_age + 1 = k * (Adam_age - 4)) : Eve_age = 14 :=
sorry

end NUMINAMATH_GPT_Eve_age_l973_97339


namespace NUMINAMATH_GPT_area_of_region_l973_97364

theorem area_of_region (x y : ℝ) : (x^2 + y^2 + 6 * x - 8 * y = 1) → (π * 26) = 26 * π :=
by
  intro h
  sorry

end NUMINAMATH_GPT_area_of_region_l973_97364


namespace NUMINAMATH_GPT_large_pile_toys_l973_97314

theorem large_pile_toys (x y : ℕ) (h1 : x + y = 120) (h2 : y = 2 * x) : y = 80 := by
  sorry

end NUMINAMATH_GPT_large_pile_toys_l973_97314


namespace NUMINAMATH_GPT_initial_black_pens_correct_l973_97354

-- Define the conditions
def initial_blue_pens : ℕ := 9
def removed_blue_pens : ℕ := 4
def remaining_blue_pens : ℕ := initial_blue_pens - removed_blue_pens

def initial_red_pens : ℕ := 6
def removed_red_pens : ℕ := 0
def remaining_red_pens : ℕ := initial_red_pens - removed_red_pens

def total_remaining_pens : ℕ := 25
def removed_black_pens : ℕ := 7

-- Assume B is the initial number of black pens
def B : ℕ := 21

-- Prove the initial number of black pens condition
theorem initial_black_pens_correct : 
  (initial_blue_pens + B + initial_red_pens) - (removed_blue_pens + removed_black_pens) = total_remaining_pens :=
by 
  have h1 : initial_blue_pens - removed_blue_pens = remaining_blue_pens := rfl
  have h2 : initial_red_pens - removed_red_pens = remaining_red_pens := rfl
  have h3 : remaining_blue_pens + (B - removed_black_pens) + remaining_red_pens = total_remaining_pens := sorry
  exact h3

end NUMINAMATH_GPT_initial_black_pens_correct_l973_97354


namespace NUMINAMATH_GPT_luke_fish_fillets_l973_97335

theorem luke_fish_fillets (daily_fish : ℕ) (days : ℕ) (fillets_per_fish : ℕ) 
  (h1 : daily_fish = 2) (h2 : days = 30) (h3 : fillets_per_fish = 2) : 
  daily_fish * days * fillets_per_fish = 120 := 
by 
  sorry

end NUMINAMATH_GPT_luke_fish_fillets_l973_97335


namespace NUMINAMATH_GPT_scientific_notation_240000_l973_97336

theorem scientific_notation_240000 :
  240000 = 2.4 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_240000_l973_97336


namespace NUMINAMATH_GPT_range_of_a_l973_97301

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x0 : ℝ, 2^x0 - 2 ≤ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) := 
sorry

end NUMINAMATH_GPT_range_of_a_l973_97301


namespace NUMINAMATH_GPT_matrix_diagonal_neg5_l973_97359

variable (M : Matrix (Fin 3) (Fin 3) ℝ)

theorem matrix_diagonal_neg5 
    (h : ∀ v : Fin 3 → ℝ, (M.mulVec v) = -5 • v) : 
    M = !![-5, 0, 0; 0, -5, 0; 0, 0, -5] :=
by
  sorry

end NUMINAMATH_GPT_matrix_diagonal_neg5_l973_97359


namespace NUMINAMATH_GPT_inequality_proof_l973_97385

theorem inequality_proof (x a : ℝ) (h1 : x > a) (h2 : a > 0) : x^2 > ax ∧ ax > a^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l973_97385


namespace NUMINAMATH_GPT_sum_mean_median_mode_l973_97345

theorem sum_mean_median_mode (l : List ℚ) (h : l = [1, 2, 2, 3, 3, 3, 3, 4, 5]) :
    let mean := (1 + 2 + 2 + 3 + 3 + 3 + 3 + 4 + 5) / 9
    let median := 3
    let mode := 3
    mean + median + mode = 8.888 :=
by
  sorry

end NUMINAMATH_GPT_sum_mean_median_mode_l973_97345


namespace NUMINAMATH_GPT_eval_inverse_l973_97399

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h₁ : g 4 = 6)
variable (h₂ : g 7 = 2)
variable (h₃ : g 3 = 7)
variable (h_inv₁ : g_inv 6 = 4)
variable (h_inv₂ : g_inv 7 = 3)

theorem eval_inverse (g : ℕ → ℕ)
(g_inv : ℕ → ℕ)
(h₁ : g 4 = 6)
(h₂ : g 7 = 2)
(h₃ : g 3 = 7)
(h_inv₁ : g_inv 6 = 4)
(h_inv₂ : g_inv 7 = 3) :
g_inv (g_inv 7 + g_inv 6) = 3 := by
  sorry

end NUMINAMATH_GPT_eval_inverse_l973_97399


namespace NUMINAMATH_GPT_proof_problem_l973_97369

def x := 3
def y := 4

theorem proof_problem : 3 * x - 2 * y = 1 := by
  -- We will rely on these definitions and properties of arithmetic to show the result.
  -- The necessary proof steps would follow here, but are skipped for now.
  sorry

end NUMINAMATH_GPT_proof_problem_l973_97369


namespace NUMINAMATH_GPT_repeated_number_divisibility_l973_97390

theorem repeated_number_divisibility (x : ℕ) (h : 1000 ≤ x ∧ x < 10000) :
  73 ∣ (10001 * x) ∧ 137 ∣ (10001 * x) :=
sorry

end NUMINAMATH_GPT_repeated_number_divisibility_l973_97390


namespace NUMINAMATH_GPT_integer_solution_inequality_l973_97356

theorem integer_solution_inequality (x : ℤ) : ((x - 1)^2 ≤ 4) → ([-1, 0, 1, 2, 3].count x = 5) :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_inequality_l973_97356


namespace NUMINAMATH_GPT_quadratic_solutions_l973_97333

-- Define the equation x^2 - 6x + 8 = 0
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Lean statement for the equivalence of solutions
theorem quadratic_solutions : ∀ x : ℝ, quadratic_eq x ↔ x = 2 ∨ x = 4 :=
by
  intro x
  dsimp [quadratic_eq]
  sorry

end NUMINAMATH_GPT_quadratic_solutions_l973_97333


namespace NUMINAMATH_GPT_mr_bird_speed_to_be_on_time_l973_97368

theorem mr_bird_speed_to_be_on_time 
  (d : ℝ) 
  (t : ℝ)
  (h1 : d = 40 * (t + 1/20))
  (h2 : d = 60 * (t - 1/20)) :
  (d / t) = 48 :=
by
  sorry

end NUMINAMATH_GPT_mr_bird_speed_to_be_on_time_l973_97368


namespace NUMINAMATH_GPT_twelfth_term_geometric_sequence_l973_97318

-- Define the first term and common ratio
def a1 : Int := 5
def r : Int := -3

-- Define the formula for the nth term of the geometric sequence
def nth_term (n : Nat) : Int := a1 * r^(n-1)

-- The statement to be proved: that the twelfth term is -885735
theorem twelfth_term_geometric_sequence : nth_term 12 = -885735 := by
  sorry

end NUMINAMATH_GPT_twelfth_term_geometric_sequence_l973_97318


namespace NUMINAMATH_GPT_all_statements_correct_l973_97397

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem all_statements_correct (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (f b b = 1) ∧
  (f b 1 = 0) ∧
  (¬(0 ∈ Set.range (f b))) ∧
  (∀ x, 0 < x ∧ x < b → f b x < 1) ∧
  (∀ x, x > b → f b x > 1) := by
  unfold f
  sorry

end NUMINAMATH_GPT_all_statements_correct_l973_97397


namespace NUMINAMATH_GPT_batsman_average_after_12th_innings_l973_97357

theorem batsman_average_after_12th_innings 
  (A : ℕ) 
  (h1 : 75 = (A + 12)) 
  (h2 : 11 * A + 75 = 12 * (A + 1)) :
  (A + 1) = 64 :=
by 
  sorry

end NUMINAMATH_GPT_batsman_average_after_12th_innings_l973_97357


namespace NUMINAMATH_GPT_expand_product_l973_97370

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x + 6) = 6 * x^2 + 26 * x + 24 := 
by 
  sorry

end NUMINAMATH_GPT_expand_product_l973_97370


namespace NUMINAMATH_GPT_tan_alpha_tan_beta_value_l973_97349

theorem tan_alpha_tan_beta_value
  (α β : ℝ)
  (h1 : Real.cos (α + β) = 1 / 5)
  (h2 : Real.cos (α - β) = 3 / 5) :
  Real.tan α * Real.tan β = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_tan_beta_value_l973_97349


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l973_97341

-- Define arithmetic sequence and given condition
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Lean 4 statement
theorem arithmetic_sequence_property {a : ℕ → ℝ} (h : arithmetic_sequence a) (h1 : a 6 = 30) : a 3 + a 9 = 60 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l973_97341


namespace NUMINAMATH_GPT_fred_more_than_daniel_l973_97393

-- Definitions and conditions from the given problem.
def total_stickers : ℕ := 750
def andrew_kept : ℕ := 130
def daniel_received : ℕ := 250
def fred_received : ℕ := total_stickers - andrew_kept - daniel_received

-- The proof problem statement.
theorem fred_more_than_daniel : fred_received - daniel_received = 120 := by 
  sorry

end NUMINAMATH_GPT_fred_more_than_daniel_l973_97393


namespace NUMINAMATH_GPT_incorrect_conclusion_symmetry_l973_97363

/-- Given the function f(x) = sin(1/5 * x + 13/6 * π), we define another function g(x) as the
translated function of f rightward by 10/3 * π units. We need to show that the graph of g(x)
is not symmetrical about the line x = π/4. -/
theorem incorrect_conclusion_symmetry (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = Real.sin (1/5 * x + 13/6 * Real.pi))
  (h₂ : ∀ x, g x = f (x - 10/3 * Real.pi)) :
  ¬ (∀ x, g (2 * (Real.pi / 4) - x) = g x) :=
sorry

end NUMINAMATH_GPT_incorrect_conclusion_symmetry_l973_97363


namespace NUMINAMATH_GPT_expected_value_is_correct_l973_97351

-- Given conditions
def prob_heads : ℚ := 2 / 5
def prob_tails : ℚ := 3 / 5
def win_amount_heads : ℚ := 5
def loss_amount_tails : ℚ := -3

-- Expected value calculation
def expected_value : ℚ := prob_heads * win_amount_heads + prob_tails * loss_amount_tails

-- Property to prove
theorem expected_value_is_correct : expected_value = 0.2 := sorry

end NUMINAMATH_GPT_expected_value_is_correct_l973_97351


namespace NUMINAMATH_GPT_length_of_marquita_garden_l973_97361

variable (length_marquita_garden : ℕ)

def total_area_mancino_gardens : ℕ := 3 * (16 * 5)
def total_gardens_area : ℕ := 304
def total_area_marquita_gardens : ℕ := total_gardens_area - total_area_mancino_gardens
def area_one_marquita_garden : ℕ := total_area_marquita_gardens / 2

theorem length_of_marquita_garden :
  (4 * length_marquita_garden = area_one_marquita_garden) →
  length_marquita_garden = 8 := by
  sorry

end NUMINAMATH_GPT_length_of_marquita_garden_l973_97361


namespace NUMINAMATH_GPT_james_payment_l973_97322

theorem james_payment (james_meal : ℕ) (friend_meal : ℕ) (tip_percent : ℕ) (final_payment : ℕ) : 
  james_meal = 16 → 
  friend_meal = 14 → 
  tip_percent = 20 → 
  final_payment = 18 :=
by
  -- Definitions
  let total_bill_before_tip := james_meal + friend_meal
  let tip := total_bill_before_tip * tip_percent / 100
  let final_bill := total_bill_before_tip + tip
  let half_bill := final_bill / 2
  -- Proof (to be filled in)
  sorry

end NUMINAMATH_GPT_james_payment_l973_97322


namespace NUMINAMATH_GPT_smallest_nat_number_l973_97317

theorem smallest_nat_number (x : ℕ) 
  (h1 : ∃ z : ℕ, x + 3 = 5 * z) 
  (h2 : ∃ n : ℕ, x - 3 = 6 * n) : x = 27 := 
sorry

end NUMINAMATH_GPT_smallest_nat_number_l973_97317


namespace NUMINAMATH_GPT_particular_solution_ODE_l973_97313

theorem particular_solution_ODE (y : ℝ → ℝ) (h : ∀ x, deriv y x + y x * Real.tan x = 0) (h₀ : y 0 = 2) :
  ∀ x, y x = 2 * Real.cos x :=
sorry

end NUMINAMATH_GPT_particular_solution_ODE_l973_97313


namespace NUMINAMATH_GPT_subset_if_a_neg_third_set_of_real_numbers_for_A_union_B_eq_A_l973_97387

def A : Set ℝ := {x | x ^ 2 - 8 * x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem subset_if_a_neg_third (a : ℝ) (h : a = -1/3) : B a ⊆ A := by
  sorry

theorem set_of_real_numbers_for_A_union_B_eq_A : {a : ℝ | A ∪ B a = A} = {0, -1/3, -1/5} := by
  sorry

end NUMINAMATH_GPT_subset_if_a_neg_third_set_of_real_numbers_for_A_union_B_eq_A_l973_97387


namespace NUMINAMATH_GPT_simplify_and_evaluate_div_expr_l973_97306

variable (m : ℤ)

theorem simplify_and_evaluate_div_expr (h : m = 2) :
  ( (m^2 - 9) / (m^2 - 6 * m + 9) / (1 - 2 / (m - 3)) = -5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_div_expr_l973_97306


namespace NUMINAMATH_GPT_track_width_l973_97327

variable (r1 r2 r3 : ℝ)

def cond1 : Prop := 2 * Real.pi * r2 - 2 * Real.pi * r1 = 20 * Real.pi
def cond2 : Prop := 2 * Real.pi * r3 - 2 * Real.pi * r2 = 30 * Real.pi

theorem track_width (h1 : cond1 r1 r2) (h2 : cond2 r2 r3) : r3 - r1 = 25 := by
  sorry

end NUMINAMATH_GPT_track_width_l973_97327


namespace NUMINAMATH_GPT_largest_coins_l973_97315

theorem largest_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n = 146 :=
by
  sorry

end NUMINAMATH_GPT_largest_coins_l973_97315


namespace NUMINAMATH_GPT_min_value_l973_97324

theorem min_value : ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (a = 1) → (b = 1) → (c = 1) →
  (∃ x, x = (a^2 + 4 * a + 2) / a ∧ x ≥ 6) ∧
  (∃ y, y = (b^2 + 4 * b + 2) / b ∧ y ≥ 6) ∧
  (∃ z, z = (c^2 + 4 * c + 2) / c ∧ z ≥ 6) →
  (∃ m, m = ((a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2)) / (a * b * c) ∧ m = 216) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_l973_97324


namespace NUMINAMATH_GPT_root_of_equation_l973_97375

theorem root_of_equation :
  ∀ x : ℝ, (x - 3)^2 = x - 3 ↔ x = 3 ∨ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_root_of_equation_l973_97375


namespace NUMINAMATH_GPT_perfect_square_of_expression_l973_97358

theorem perfect_square_of_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 := 
by 
  sorry

end NUMINAMATH_GPT_perfect_square_of_expression_l973_97358


namespace NUMINAMATH_GPT_intersection_with_xz_plane_l973_97380

-- Initial points on the line
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def point1 : Point3D := ⟨2, -1, 3⟩
def point2 : Point3D := ⟨6, -4, 7⟩

-- Definition of the line parametrization
def param_line (t : ℝ) : Point3D :=
  ⟨ point1.x + t * (point2.x - point1.x)
  , point1.y + t * (point2.y - point1.y)
  , point1.z + t * (point2.z - point1.z) ⟩

-- Prove that the line intersects the xz-plane at the expected point
theorem intersection_with_xz_plane :
  ∃ t : ℝ, param_line t = ⟨ 2/3, 0, 5/3 ⟩ :=
sorry

end NUMINAMATH_GPT_intersection_with_xz_plane_l973_97380


namespace NUMINAMATH_GPT_least_element_in_T_l973_97307

variable (S : Finset ℕ)
variable (T : Finset ℕ)
variable (hS : S = Finset.range 16 \ {0})
variable (hT : T.card = 5)
variable (hTsubS : T ⊆ S)
variable (hCond : ∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0))

theorem least_element_in_T (S T : Finset ℕ) (hT : T.card = 5) (hTsubS : T ⊆ S)
  (hCond : ∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) : 
  ∃ m ∈ T, m = 5 :=
by
  sorry

end NUMINAMATH_GPT_least_element_in_T_l973_97307


namespace NUMINAMATH_GPT_number_of_girls_more_than_boys_l973_97392

theorem number_of_girls_more_than_boys
    (total_students : ℕ)
    (number_of_boys : ℕ)
    (h1 : total_students = 485)
    (h2 : number_of_boys = 208) :
    total_students - number_of_boys - number_of_boys = 69 :=
by
    sorry

end NUMINAMATH_GPT_number_of_girls_more_than_boys_l973_97392


namespace NUMINAMATH_GPT_original_number_is_seven_l973_97377

theorem original_number_is_seven (x : ℕ) (h : 3 * x - 5 = 16) : x = 7 := by
sorry

end NUMINAMATH_GPT_original_number_is_seven_l973_97377


namespace NUMINAMATH_GPT_ab_value_l973_97342

variables {a b : ℝ}

theorem ab_value (h₁ : a - b = 6) (h₂ : a^2 + b^2 = 50) : ab = 7 :=
sorry

end NUMINAMATH_GPT_ab_value_l973_97342


namespace NUMINAMATH_GPT_sum_of_first_five_terms_l973_97312

theorem sum_of_first_five_terms
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h_sum_n : ∀ n, S n = n / 2 * (a 1 + a n))
  (h_roots : ∀ x, x^2 - x - 3 = 0 → x = a 2 ∨ x = a 4)
  (h_vieta : a 2 + a 4 = 1) :
  S 5 = 5 / 2 :=
  sorry

end NUMINAMATH_GPT_sum_of_first_five_terms_l973_97312


namespace NUMINAMATH_GPT_geometric_sequence_product_l973_97389

theorem geometric_sequence_product (a b : ℝ) (h : 2 * b = a * 16) : a * b = 32 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l973_97389


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l973_97398

theorem batsman_average_after_17th_inning
  (A : ℝ) -- average before 17th inning
  (h1 : (16 * A + 50) / 17 = A + 2) : 
  (A + 2) = 18 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l973_97398


namespace NUMINAMATH_GPT_decorations_cost_l973_97383

def tablecloth_cost : ℕ := 20 * 25
def place_setting_cost : ℕ := 20 * 4 * 10
def rose_cost : ℕ := 20 * 10 * 5
def lily_cost : ℕ := 20 * 15 * 4

theorem decorations_cost :
  tablecloth_cost + place_setting_cost + rose_cost + lily_cost = 3500 :=
by sorry

end NUMINAMATH_GPT_decorations_cost_l973_97383


namespace NUMINAMATH_GPT_tables_count_l973_97337

theorem tables_count (c t : Nat) (h1 : c = 8 * t) (h2 : 3 * c + 5 * t = 580) : t = 20 :=
by
  sorry

end NUMINAMATH_GPT_tables_count_l973_97337
