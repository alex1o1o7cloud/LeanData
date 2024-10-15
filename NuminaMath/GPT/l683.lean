import Mathlib

namespace NUMINAMATH_GPT_min_disks_to_store_files_l683_68308

open Nat

theorem min_disks_to_store_files :
  ∃ minimum_disks : ℕ,
    (minimum_disks = 24) ∧
    ∀ (files : ℕ) (disk_capacity : ℕ) (file_sizes : List ℕ),
      files = 36 →
      disk_capacity = 144 →
      (∃ (size_85 : ℕ) (size_75 : ℕ) (size_45 : ℕ),
         size_85 = 5 ∧
         size_75 = 15 ∧
         size_45 = 16 ∧
         (∀ (disks : ℕ), disks >= minimum_disks →
            ∃ (used_disks_85 : ℕ) (remaining_files_45 : ℕ) (used_disks_45 : ℕ) (used_disks_75 : ℕ),
              remaining_files_45 = size_45 - used_disks_85 ∧
              used_disks_85 = size_85 ∧
              (remaining_files_45 % 3 = 0 → used_disks_45 = remaining_files_45 / 3) ∧
              (remaining_files_45 % 3 ≠ 0 → used_disks_45 = remaining_files_45 / 3 + 1) ∧
              used_disks_75 = size_75 ∧
              disks = used_disks_85 + used_disks_45 + used_disks_75)) :=
by
  sorry

end NUMINAMATH_GPT_min_disks_to_store_files_l683_68308


namespace NUMINAMATH_GPT_primes_p_q_divisibility_l683_68351

theorem primes_p_q_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hq_eq : q = p + 2) :
  (p + q) ∣ (p ^ q + q ^ p) := 
sorry

end NUMINAMATH_GPT_primes_p_q_divisibility_l683_68351


namespace NUMINAMATH_GPT_arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125_l683_68326

theorem arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125 :
  (16 + 23 + 38 + 11.5) / 4 = 22.125 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125_l683_68326


namespace NUMINAMATH_GPT_chicken_cost_l683_68392
noncomputable def chicken_cost_per_plate
  (plates : ℕ) 
  (rice_cost_per_plate : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let total_rice_cost := plates * rice_cost_per_plate
  let total_chicken_cost := total_cost - total_rice_cost
  total_chicken_cost / plates

theorem chicken_cost
  (hplates : plates = 100)
  (hrice_cost_per_plate : rice_cost_per_plate = 0.10)
  (htotal_cost : total_cost = 50) :
  chicken_cost_per_plate 100 0.10 50 = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_chicken_cost_l683_68392


namespace NUMINAMATH_GPT_projective_iff_fractional_linear_l683_68335

def projective_transformation (P : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d))

theorem projective_iff_fractional_linear (P : ℝ → ℝ) : 
  projective_transformation P ↔ ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d)) :=
by 
  sorry

end NUMINAMATH_GPT_projective_iff_fractional_linear_l683_68335


namespace NUMINAMATH_GPT_equidistant_point_x_coord_l683_68324

theorem equidistant_point_x_coord :
  ∃ x y : ℝ, y = x ∧ dist (x, y) (x, 0) = dist (x, y) (0, y) ∧ dist (x, y) (0, y) = dist (x, y) (x, 5 - x)
    → x = 5 / 2 :=
by sorry

end NUMINAMATH_GPT_equidistant_point_x_coord_l683_68324


namespace NUMINAMATH_GPT_rectangle_breadth_l683_68319

theorem rectangle_breadth (sq_area : ℝ) (rect_area : ℝ) (radius_rect_relation : ℝ → ℝ) 
  (rect_length_relation : ℝ → ℝ) (breadth_correct: ℝ) : 
  (sq_area = 3600) →
  (rect_area = 240) →
  (forall r, radius_rect_relation r = r) →
  (forall r, rect_length_relation r = (2/5) * r) →
  breadth_correct = 10 :=
by
  intros h_sq_area h_rect_area h_radius_rect h_rect_length
  sorry

end NUMINAMATH_GPT_rectangle_breadth_l683_68319


namespace NUMINAMATH_GPT_randy_piggy_bank_l683_68312

theorem randy_piggy_bank : 
  ∀ (initial_amount trips_per_month cost_per_trip months_per_year total_spent_left : ℕ),
  initial_amount = 200 →
  cost_per_trip = 2 →
  trips_per_month = 4 →
  months_per_year = 12 →
  total_spent_left = initial_amount - (cost_per_trip * trips_per_month * months_per_year) →
  total_spent_left = 104 :=
by
  intros initial_amount trips_per_month cost_per_trip months_per_year total_spent_left
  sorry

end NUMINAMATH_GPT_randy_piggy_bank_l683_68312


namespace NUMINAMATH_GPT_boys_to_girls_ratio_l683_68394

theorem boys_to_girls_ratio (T G : ℕ) (h : (1 / 2) * G = (1 / 6) * T) : (T - G) = 2 * G := by
  sorry

end NUMINAMATH_GPT_boys_to_girls_ratio_l683_68394


namespace NUMINAMATH_GPT_maximum_gcd_of_sequence_l683_68386

def a_n (n : ℕ) : ℕ := 100 + n^2

def d_n (n : ℕ) : ℕ := Nat.gcd (a_n n) (a_n (n + 1))

theorem maximum_gcd_of_sequence : ∃ n : ℕ, ∀ m : ℕ, d_n n ≤ d_n m ∧ d_n n = 401 := sorry

end NUMINAMATH_GPT_maximum_gcd_of_sequence_l683_68386


namespace NUMINAMATH_GPT_parabola_vertex_and_point_l683_68399

theorem parabola_vertex_and_point (a b c : ℝ) (h_vertex : (1, -2) = (1, a * 1^2 + b * 1 + c))
  (h_point : (3, 7) = (3, a * 3^2 + b * 3 + c)) : a = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_parabola_vertex_and_point_l683_68399


namespace NUMINAMATH_GPT_problem_l683_68381

-- Definitions based on the provided conditions
def frequency_varies (freq : Real) : Prop := true -- Placeholder definition
def probability_is_stable (prob : Real) : Prop := true -- Placeholder definition
def is_random_event (event : Type) : Prop := true -- Placeholder definition
def is_random_experiment (experiment : Type) : Prop := true -- Placeholder definition
def is_sum_of_events (event1 event2 : Prop) : Prop := event1 ∨ event2 -- Definition of sum of events
def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B) -- Definition of mutually exclusive events
def complementary_events (A B : Prop) : Prop := A ↔ ¬B -- Definition of complementary events
def equally_likely_events (events : List Prop) : Prop := true -- Placeholder definition

-- Translation of the questions and correct answers
theorem problem (freq prob : Real) (event experiment : Type) (A B : Prop) (events : List Prop) :
  (¬(frequency_varies freq = probability_is_stable prob)) ∧ -- 1
  ((is_random_event event) ≠ (is_random_experiment experiment)) ∧ -- 2
  (probability_is_stable prob) ∧ -- 3
  (is_sum_of_events A B) ∧ -- 4
  (mutually_exclusive A B → ¬(probability_is_stable (1 - prob))) ∧ -- 5
  (¬(equally_likely_events events)) :=  -- 6
by
  sorry

end NUMINAMATH_GPT_problem_l683_68381


namespace NUMINAMATH_GPT_sam_walked_distance_l683_68300

theorem sam_walked_distance
  (distance_apart : ℝ) (fred_speed : ℝ) (sam_speed : ℝ) (t : ℝ)
  (H1 : distance_apart = 35) (H2 : fred_speed = 2) (H3 : sam_speed = 5)
  (H4 : 2 * t + 5 * t = distance_apart) :
  5 * t = 25 :=
by
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_sam_walked_distance_l683_68300


namespace NUMINAMATH_GPT_necklaces_caught_l683_68354

noncomputable def total_necklaces_caught (boudreaux rhonda latch cecilia : ℕ) : ℕ :=
  boudreaux + rhonda + latch + cecilia

theorem necklaces_caught :
  ∃ (boudreaux rhonda latch cecilia : ℕ), 
    boudreaux = 12 ∧
    rhonda = boudreaux / 2 ∧
    latch = 3 * rhonda - 4 ∧
    cecilia = latch + 3 ∧
    total_necklaces_caught boudreaux rhonda latch cecilia = 49 ∧
    (total_necklaces_caught boudreaux rhonda latch cecilia) % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_necklaces_caught_l683_68354


namespace NUMINAMATH_GPT_original_price_of_article_l683_68346

theorem original_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) (hSP : SP = 374) (hprofit : profit_percent = 0.10) : 
  CP = 340 ↔ SP = CP * (1 + profit_percent) :=
by 
  sorry

end NUMINAMATH_GPT_original_price_of_article_l683_68346


namespace NUMINAMATH_GPT_Rebecca_tent_stakes_l683_68330

theorem Rebecca_tent_stakes : 
  ∃ T D W : ℕ, 
    D = 3 * T ∧ 
    W = T + 2 ∧ 
    T + D + W = 22 ∧ 
    T = 4 := 
by
  sorry

end NUMINAMATH_GPT_Rebecca_tent_stakes_l683_68330


namespace NUMINAMATH_GPT_sum_of_three_numbers_is_seventy_l683_68355

theorem sum_of_three_numbers_is_seventy
  (a b c : ℝ)
  (h1 : a ≤ b ∧ b ≤ c)
  (h2 : (a + b + c) / 3 = a + 20)
  (h3 : (a + b + c) / 3 = c - 30)
  (h4 : b = 10)
  (h5 : a + c = 60) :
  a + b + c = 70 :=
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_is_seventy_l683_68355


namespace NUMINAMATH_GPT_compare_negative_positive_l683_68310

theorem compare_negative_positive : -897 < 0.01 := sorry

end NUMINAMATH_GPT_compare_negative_positive_l683_68310


namespace NUMINAMATH_GPT_winning_candidate_percentage_l683_68372

theorem winning_candidate_percentage (P : ℕ) (majority : ℕ) (total_votes : ℕ) (h1 : majority = 188) (h2 : total_votes = 470) (h3 : 2 * majority = (2 * P - 100) * total_votes) : 
  P = 70 := 
sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l683_68372


namespace NUMINAMATH_GPT_number_of_initials_sets_l683_68356

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end NUMINAMATH_GPT_number_of_initials_sets_l683_68356


namespace NUMINAMATH_GPT_proof_max_ρ_sq_l683_68379

noncomputable def max_ρ_sq (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≥ b) 
    (x y : ℝ) (h₃ : 0 ≤ x) (h₄ : x < a) (h₅ : 0 ≤ y) (h₆ : y < b)
    (h_xy : a^2 + y^2 = b^2 + x^2)
    (h_eq : a^2 + y^2 = (a - x)^2 + (b - y)^2)
    (h_x_le : x ≤ 2 * a / 3) : ℝ :=
  (a / b) ^ 2

theorem proof_max_ρ_sq (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≥ b)
    (x y : ℝ) (h₃ : 0 ≤ x) (h₄ : x < a) (h₅ : 0 ≤ y) (h₆ : y < b)
    (h_xy : a^2 + y^2 = b^2 + x^2)
    (h_eq : a^2 + y^2 = (a - x)^2 + (b - y)^2)
    (h_x_le : x ≤ 2 * a / 3) : (max_ρ_sq a b h₀ h₁ h₂ x y h₃ h₄ h₅ h₆ h_xy h_eq h_x_le) ≤ 9 / 5 := by
  sorry

end NUMINAMATH_GPT_proof_max_ρ_sq_l683_68379


namespace NUMINAMATH_GPT_insulation_cost_l683_68395

def rectangular_prism_surface_area (l w h : ℕ) : ℕ :=
2 * l * w + 2 * l * h + 2 * w * h

theorem insulation_cost
  (l w h : ℕ) (cost_per_square_foot : ℕ)
  (h_l : l = 6) (h_w : w = 3) (h_h : h = 2) (h_cost : cost_per_square_foot = 20) :
  rectangular_prism_surface_area l w h * cost_per_square_foot = 1440 := 
sorry

end NUMINAMATH_GPT_insulation_cost_l683_68395


namespace NUMINAMATH_GPT_oranges_count_l683_68309

theorem oranges_count (N : ℕ) (k : ℕ) (m : ℕ) (j : ℕ) :
  (N ≡ 2 [MOD 10]) ∧ (N ≡ 0 [MOD 12]) → N = 72 :=
by
  sorry

end NUMINAMATH_GPT_oranges_count_l683_68309


namespace NUMINAMATH_GPT_final_amount_after_5_years_l683_68304

-- Define conditions as hypotheses
def principal := 200
def final_amount_after_2_years := 260
def time_2_years := 2

-- Define our final question and answer as a Lean theorem
theorem final_amount_after_5_years : 
  (final_amount_after_2_years - principal) = principal * (rate * time_2_years) →
  (rate * 3) = 90 →
  final_amount_after_2_years + (principal * rate * 3) = 350 :=
by
  intros h1 h2
  -- Proof skipped using sorry
  sorry

end NUMINAMATH_GPT_final_amount_after_5_years_l683_68304


namespace NUMINAMATH_GPT_b_charges_l683_68366

theorem b_charges (total_cost : ℕ) (a_hours b_hours c_hours : ℕ)
  (h_total_cost : total_cost = 720)
  (h_a_hours : a_hours = 9)
  (h_b_hours : b_hours = 10)
  (h_c_hours : c_hours = 13) :
  (total_cost * b_hours / (a_hours + b_hours + c_hours)) = 225 :=
by
  sorry

end NUMINAMATH_GPT_b_charges_l683_68366


namespace NUMINAMATH_GPT_factorize_expression_l683_68337

theorem factorize_expression (x : ℝ) : 4 * x^2 - 4 = 4 * (x + 1) * (x - 1) := 
  sorry

end NUMINAMATH_GPT_factorize_expression_l683_68337


namespace NUMINAMATH_GPT_max_mark_cells_l683_68348

theorem max_mark_cells (n : Nat) (grid : Fin n → Fin n → Bool) :
  (∀ i : Fin n, ∃ j : Fin n, grid i j = true) ∧ 
  (∀ j : Fin n, ∃ i : Fin n, grid i j = true) ∧ 
  (∀ (x1 x2 y1 y2 : Fin n), (x1 ≤ x2 ∧ y1 ≤ y2 ∧ (x2.1 - x1.1 + 1) * (y2.1 - y1.1 + 1) ≥ n) → 
   ∃ i : Fin n, ∃ j : Fin n, grid i j = true ∧ x1 ≤ i ∧ i ≤ x2 ∧ y1 ≤ j ∧ j ≤ y2) → 
  (n ≤ 7) := sorry

end NUMINAMATH_GPT_max_mark_cells_l683_68348


namespace NUMINAMATH_GPT_average_not_1380_l683_68333

-- Define the set of numbers
def numbers := [1200, 1400, 1510, 1520, 1530, 1200]

-- Define the claimed average
def claimed_avg := 1380

-- The sum of the numbers
def sumNumbers := numbers.sum

-- The number of items in the set
def countNumbers := numbers.length

-- The correct average calculation
def correct_avg : ℚ := sumNumbers / countNumbers

-- The proof problem: proving that the correct average is not equal to the claimed average
theorem average_not_1380 : correct_avg ≠ claimed_avg := by
  sorry

end NUMINAMATH_GPT_average_not_1380_l683_68333


namespace NUMINAMATH_GPT_equivalent_modulo_l683_68384

theorem equivalent_modulo :
  ∃ (n : ℤ), 0 ≤ n ∧ n < 31 ∧ -250 ≡ n [ZMOD 31] ∧ n = 29 := 
by
  sorry

end NUMINAMATH_GPT_equivalent_modulo_l683_68384


namespace NUMINAMATH_GPT_dot_product_a_b_l683_68344

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (1, -2)

theorem dot_product_a_b :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = -3 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_a_b_l683_68344


namespace NUMINAMATH_GPT_gina_order_rose_cups_l683_68365

theorem gina_order_rose_cups 
  (rose_cups_per_hour : ℕ) 
  (lily_cups_per_hour : ℕ) 
  (total_lily_cups_order : ℕ) 
  (total_pay : ℕ) 
  (pay_per_hour : ℕ) 
  (total_hours_worked : ℕ) 
  (hours_spent_with_lilies : ℕ)
  (hours_spent_with_roses : ℕ) 
  (rose_cups_order : ℕ) :
  rose_cups_per_hour = 6 →
  lily_cups_per_hour = 7 →
  total_lily_cups_order = 14 →
  total_pay = 90 →
  pay_per_hour = 30 →
  total_hours_worked = total_pay / pay_per_hour →
  hours_spent_with_lilies = total_lily_cups_order / lily_cups_per_hour →
  hours_spent_with_roses = total_hours_worked - hours_spent_with_lilies →
  rose_cups_order = rose_cups_per_hour * hours_spent_with_roses →
  rose_cups_order = 6 := 
by
  sorry

end NUMINAMATH_GPT_gina_order_rose_cups_l683_68365


namespace NUMINAMATH_GPT_units_digit_of_3_pow_y_l683_68367

theorem units_digit_of_3_pow_y
    (x : ℕ)
    (h1 : (2^3)^x = 4096)
    (y : ℕ)
    (h2 : y = x^3) :
    (3^y) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_3_pow_y_l683_68367


namespace NUMINAMATH_GPT_p_squared_plus_41_composite_for_all_primes_l683_68336

theorem p_squared_plus_41_composite_for_all_primes (p : ℕ) (hp : Prime p) : 
  ∃ d : ℕ, d > 1 ∧ d < p^2 + 41 ∧ d ∣ (p^2 + 41) :=
by
  sorry

end NUMINAMATH_GPT_p_squared_plus_41_composite_for_all_primes_l683_68336


namespace NUMINAMATH_GPT_abigail_lost_money_l683_68322

theorem abigail_lost_money (initial_amount spent_first_store spent_second_store remaining_amount_lost: ℝ) 
  (h_initial : initial_amount = 50) 
  (h_spent_first : spent_first_store = 15.25) 
  (h_spent_second : spent_second_store = 8.75) 
  (h_remaining : remaining_amount_lost = 16) : (initial_amount - spent_first_store - spent_second_store - remaining_amount_lost = 10) :=
by
  sorry

end NUMINAMATH_GPT_abigail_lost_money_l683_68322


namespace NUMINAMATH_GPT_trigonometric_identity_l683_68362

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
    Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = -4 / 3 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l683_68362


namespace NUMINAMATH_GPT_tessa_owes_30_l683_68315

-- Definitions based on given conditions
def initial_debt : ℕ := 40
def paid_back : ℕ := initial_debt / 2
def remaining_debt_after_payment : ℕ := initial_debt - paid_back
def additional_borrowing : ℕ := 10
def total_debt : ℕ := remaining_debt_after_payment + additional_borrowing

-- Theorem to be proved
theorem tessa_owes_30 : total_debt = 30 :=
by
  sorry

end NUMINAMATH_GPT_tessa_owes_30_l683_68315


namespace NUMINAMATH_GPT_construct_triangle_condition_l683_68364

theorem construct_triangle_condition (m_a f_a s_a : ℝ) : 
  (m_a < f_a) ∧ (f_a < s_a) ↔ (exists A B C : Type, true) :=
sorry

end NUMINAMATH_GPT_construct_triangle_condition_l683_68364


namespace NUMINAMATH_GPT_visitors_that_day_l683_68360

theorem visitors_that_day (total_visitors : ℕ) (previous_day_visitors : ℕ) 
  (h_total : total_visitors = 406) (h_previous : previous_day_visitors = 274) : 
  total_visitors - previous_day_visitors = 132 :=
by
  sorry

end NUMINAMATH_GPT_visitors_that_day_l683_68360


namespace NUMINAMATH_GPT_mod_3_power_87_plus_5_l683_68357

theorem mod_3_power_87_plus_5 :
  (3 ^ 87 + 5) % 11 = 3 := 
by
  sorry

end NUMINAMATH_GPT_mod_3_power_87_plus_5_l683_68357


namespace NUMINAMATH_GPT_product_of_midpoint_coordinates_l683_68302

theorem product_of_midpoint_coordinates
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 4) (h2 : y1 = -3) (h3 : x2 = -8) (h4 : y2 = 7) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx * my = -4) :=
by
  -- Here we would carry out the proof.
  sorry

end NUMINAMATH_GPT_product_of_midpoint_coordinates_l683_68302


namespace NUMINAMATH_GPT_find_distance_of_post_office_from_village_l683_68398

-- Conditions
def rate_to_post_office : ℝ := 12.5
def rate_back_village : ℝ := 2
def total_time : ℝ := 5.8

-- Statement of the theorem
theorem find_distance_of_post_office_from_village (D : ℝ) 
  (travel_time_to : D / rate_to_post_office = D / 12.5) 
  (travel_time_back : D / rate_back_village = D / 2)
  (journey_time_total : D / 12.5 + D / 2 = total_time) : 
  D = 10 := 
sorry

end NUMINAMATH_GPT_find_distance_of_post_office_from_village_l683_68398


namespace NUMINAMATH_GPT_quilt_width_is_eight_l683_68376

def length := 7
def cost_per_square_foot := 40
def total_cost := 2240
def area := total_cost / cost_per_square_foot

theorem quilt_width_is_eight :
  area / length = 8 := by
  sorry

end NUMINAMATH_GPT_quilt_width_is_eight_l683_68376


namespace NUMINAMATH_GPT_sum_first_five_terms_geometric_sequence_l683_68393

theorem sum_first_five_terms_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ):
  (∀ n, a (n+1) = a 1 * (1/2) ^ n) →
  a 1 = 16 →
  1/2 * (a 4 + a 7) = 9 / 8 →
  S 5 = (a 1 * (1 - (1 / 2) ^ 5)) / (1 - 1 / 2) →
  S 5 = 31 := by
  sorry

end NUMINAMATH_GPT_sum_first_five_terms_geometric_sequence_l683_68393


namespace NUMINAMATH_GPT_complex_shape_perimeter_l683_68389

theorem complex_shape_perimeter :
  ∃ h : ℝ, 12 * h - 20 = 95 ∧
  (24 + ((230 / 12) - 2) + 10 : ℝ) = 51.1667 :=
by
  sorry

end NUMINAMATH_GPT_complex_shape_perimeter_l683_68389


namespace NUMINAMATH_GPT_alligators_not_hiding_l683_68320

theorem alligators_not_hiding (total_alligators hiding_alligators : ℕ) 
  (h1 : total_alligators = 75) 
  (h2 : hiding_alligators = 19) : 
  total_alligators - hiding_alligators = 56 :=
by
  -- The proof will go here, which is currently a placeholder.
  sorry

end NUMINAMATH_GPT_alligators_not_hiding_l683_68320


namespace NUMINAMATH_GPT_duration_period_l683_68369

-- Define the conditions and what we need to prove
theorem duration_period (t : ℝ) (h : 3200 * 0.025 * t = 400) : 
  t = 5 :=
sorry

end NUMINAMATH_GPT_duration_period_l683_68369


namespace NUMINAMATH_GPT_solution_l683_68383

theorem solution (x : ℝ) (h : 6 ∈ ({2, 4, x * x - x} : Set ℝ)) : x = 3 ∨ x = -2 := 
by 
  sorry

end NUMINAMATH_GPT_solution_l683_68383


namespace NUMINAMATH_GPT_addition_example_l683_68305

theorem addition_example : 0.4 + 56.7 = 57.1 := by
  -- Here we need to prove the main statement
  sorry

end NUMINAMATH_GPT_addition_example_l683_68305


namespace NUMINAMATH_GPT_ratio_of_hours_l683_68390

theorem ratio_of_hours (x y z : ℕ) 
  (h1 : x + y + z = 157) 
  (h2 : z = y - 8) 
  (h3 : z = 56) 
  (h4 : y = x + 10) : 
  (y / gcd y x) = 32 ∧ (x / gcd y x) = 27 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_hours_l683_68390


namespace NUMINAMATH_GPT_sum_smallest_largest_eq_2z_l683_68321

theorem sum_smallest_largest_eq_2z (m b z : ℤ) (h1 : m > 0) (h2 : z = (b + (b + 2 * (m - 1))) / 2) :
  b + (b + 2 * (m - 1)) = 2 * z :=
sorry

end NUMINAMATH_GPT_sum_smallest_largest_eq_2z_l683_68321


namespace NUMINAMATH_GPT_simplify_fraction_1_210_plus_17_35_l683_68314

theorem simplify_fraction_1_210_plus_17_35 :
  1 / 210 + 17 / 35 = 103 / 210 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_1_210_plus_17_35_l683_68314


namespace NUMINAMATH_GPT_mask_digits_l683_68301

theorem mask_digits : 
  ∃ (elephant mouse pig panda : ℕ), 
  (elephant ≠ mouse ∧ elephant ≠ pig ∧ elephant ≠ panda ∧ 
   mouse ≠ pig ∧ mouse ≠ panda ∧ pig ≠ panda) ∧
  (4 * 4 = 16) ∧ (7 * 7 = 49) ∧ (8 * 8 = 64) ∧ (9 * 9 = 81) ∧
  (elephant = 6) ∧ (mouse = 4) ∧ (pig = 8) ∧ (panda = 1) :=
by
  sorry

end NUMINAMATH_GPT_mask_digits_l683_68301


namespace NUMINAMATH_GPT_negative_real_root_range_l683_68338

theorem negative_real_root_range (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (1 / Real.pi) ^ x = (1 + a) / (1 - a)) ↔ 0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_negative_real_root_range_l683_68338


namespace NUMINAMATH_GPT_polynomial_roots_sum_l683_68332

theorem polynomial_roots_sum (p q : ℂ) (hp : p + q = 5) (hq : p * q = 7) : 
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 559 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_roots_sum_l683_68332


namespace NUMINAMATH_GPT_min_value_proof_l683_68380

noncomputable def minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3/2) * a + b = 1) : ℝ :=
  (3 / a) + (2 / b)

theorem min_value_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3/2) * a + b = 1) : minimum_value a b h1 h2 h3 = 25 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_proof_l683_68380


namespace NUMINAMATH_GPT_count_remainders_gte_l683_68352

def remainder (a N : ℕ) : ℕ := a % N

theorem count_remainders_gte (N : ℕ) : 
  (∀ a, a > 0 → remainder a 1000 > remainder a 1001 → N ≤ 1000000) →
  N = 499500 :=
by
  sorry

end NUMINAMATH_GPT_count_remainders_gte_l683_68352


namespace NUMINAMATH_GPT_blocks_before_jess_turn_l683_68377

def blocks_at_start : Nat := 54
def players : Nat := 5
def rounds : Nat := 5
def father_removes_block_in_6th_round : Nat := 1

theorem blocks_before_jess_turn :
    (blocks_at_start - (players * rounds + father_removes_block_in_6th_round)) = 28 :=
by 
    sorry

end NUMINAMATH_GPT_blocks_before_jess_turn_l683_68377


namespace NUMINAMATH_GPT_cube_vertices_probability_l683_68316

theorem cube_vertices_probability (totalVertices : ℕ) (selectedVertices : ℕ) 
   (totalCombinations : ℕ) (favorableOutcomes : ℕ) : 
   totalVertices = 8 ∧ selectedVertices = 4 ∧ totalCombinations = 70 ∧ favorableOutcomes = 12 → 
   (favorableOutcomes : ℚ) / totalCombinations = 6 / 35 := by
   sorry

end NUMINAMATH_GPT_cube_vertices_probability_l683_68316


namespace NUMINAMATH_GPT_cos_decreasing_intervals_l683_68334

open Real

def is_cos_decreasing_interval (k : ℤ) : Prop := 
  let f (x : ℝ) := cos (π / 4 - 2 * x)
  ∀ x y : ℝ, (k * π + π / 8 ≤ x) → (x ≤ k * π + 5 * π / 8) → 
             (k * π + π / 8 ≤ y) → (y ≤ k * π + 5 * π / 8) → 
             x < y → f x > f y

theorem cos_decreasing_intervals : ∀ k : ℤ, is_cos_decreasing_interval k :=
by
  sorry

end NUMINAMATH_GPT_cos_decreasing_intervals_l683_68334


namespace NUMINAMATH_GPT_min_tablets_to_ensure_three_each_l683_68303

theorem min_tablets_to_ensure_three_each (A B C : ℕ) (hA : A = 20) (hB : B = 25) (hC : C = 15) : 
  ∃ n, n = 48 ∧ (∀ x y z, x + y + z = n → x ≥ 3 ∧ y ≥ 3 ∧ z ≥ 3) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_min_tablets_to_ensure_three_each_l683_68303


namespace NUMINAMATH_GPT_chord_midpoint_line_eqn_l683_68359

-- Definitions of points and the ellipse condition
def P : ℝ × ℝ := (3, 2)

def is_midpoint (P E F : ℝ × ℝ) := 
  P.1 = (E.1 + F.1) / 2 ∧ P.2 = (E.2 + F.2) / 2

def ellipse (x y : ℝ) := 
  4 * x^2 + 9 * y^2 = 144

theorem chord_midpoint_line_eqn
  (E F : ℝ × ℝ) 
  (h1 : is_midpoint P E F)
  (h2 : ellipse E.1 E.2)
  (h3 : ellipse F.1 F.2):
  ∃ (m b : ℝ), (P.2 = m * P.1 + b) ∧ (2 * P.1 + 3 * P.2 - 12 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_chord_midpoint_line_eqn_l683_68359


namespace NUMINAMATH_GPT_necessary_not_sufficient_x2_minus_3x_plus_2_l683_68345

theorem necessary_not_sufficient_x2_minus_3x_plus_2 (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → x^2 - 3 * x + 2 ≤ 0) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ m ∧ ¬(x^2 - 3 * x + 2 ≤ 0)) →
  m ≥ 2 :=
sorry

end NUMINAMATH_GPT_necessary_not_sufficient_x2_minus_3x_plus_2_l683_68345


namespace NUMINAMATH_GPT_complex_ab_value_l683_68361

open Complex

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h : i = Complex.I) (h₁ : (a + b * i) * (3 + i) = 10 + 10 * i) : a * b = 8 := 
by
  sorry

end NUMINAMATH_GPT_complex_ab_value_l683_68361


namespace NUMINAMATH_GPT_total_number_of_wheels_l683_68382

-- Define the conditions as hypotheses
def cars := 2
def wheels_per_car := 4

def bikes := 2
def trashcans := 1
def wheels_per_bike_or_trashcan := 2

def roller_skates_pair := 1
def wheels_per_skate := 4

def tricycle := 1
def wheels_per_tricycle := 3

-- Prove the total number of wheels
theorem total_number_of_wheels :
  cars * wheels_per_car +
  (bikes + trashcans) * wheels_per_bike_or_trashcan +
  (roller_skates_pair * 2) * wheels_per_skate +
  tricycle * wheels_per_tricycle 
  = 25 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_wheels_l683_68382


namespace NUMINAMATH_GPT_combined_tax_rate_l683_68358

-- Definitions and conditions
def tax_rate_mork : ℝ := 0.45
def tax_rate_mindy : ℝ := 0.20
def income_ratio_mindy_to_mork : ℝ := 4

-- Theorem statement
theorem combined_tax_rate :
  ∀ (M : ℝ), (tax_rate_mork * M + tax_rate_mindy * (income_ratio_mindy_to_mork * M)) / (M + income_ratio_mindy_to_mork * M) = 0.25 :=
by
  intros M
  sorry

end NUMINAMATH_GPT_combined_tax_rate_l683_68358


namespace NUMINAMATH_GPT_range_of_a_l683_68397

open Set Real

noncomputable def A (a : ℝ) : Set ℝ := {x | x * (x - a) < 0}
def B : Set ℝ := {x | x^2 - 7 * x - 18 < 0}

theorem range_of_a (a : ℝ) : A a ⊆ B → (-2 : ℝ) ≤ a ∧ a ≤ 9 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l683_68397


namespace NUMINAMATH_GPT_work_last_duration_l683_68329

theorem work_last_duration
  (work_rate_x : ℚ := 1 / 20)
  (work_rate_y : ℚ := 1 / 12)
  (days_x_worked_alone : ℚ := 4)
  (combined_work_rate : ℚ := work_rate_x + work_rate_y)
  (remaining_work : ℚ := 1 - days_x_worked_alone * work_rate_x) :
  (remaining_work / combined_work_rate + days_x_worked_alone = 10) :=
by
  sorry

end NUMINAMATH_GPT_work_last_duration_l683_68329


namespace NUMINAMATH_GPT_increase_result_l683_68368

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end NUMINAMATH_GPT_increase_result_l683_68368


namespace NUMINAMATH_GPT_part_I_5_continuous_part_I_6_not_continuous_part_II_min_k_for_8_continuous_part_III_min_k_for_20_continuous_l683_68350

def is_continuous_representable (m : ℕ) (Q : List ℤ) : Prop :=
  ∀ n ∈ (List.range (m + 1)).tail, ∃ (sublist : List ℤ), sublist ≠ [] ∧ sublist ∈ Q.sublists' ∧ sublist.sum = n

theorem part_I_5_continuous :
  is_continuous_representable 5 [2, 1, 4] :=
sorry

theorem part_I_6_not_continuous :
  ¬is_continuous_representable 6 [2, 1, 4] :=
sorry

theorem part_II_min_k_for_8_continuous (Q : List ℤ) :
  is_continuous_representable 8 Q → Q.length ≥ 4 :=
sorry

theorem part_III_min_k_for_20_continuous (Q : List ℤ) 
  (h : is_continuous_representable 20 Q) (h_sum : Q.sum < 20) :
  Q.length ≥ 7 :=
sorry

end NUMINAMATH_GPT_part_I_5_continuous_part_I_6_not_continuous_part_II_min_k_for_8_continuous_part_III_min_k_for_20_continuous_l683_68350


namespace NUMINAMATH_GPT_calculate_expression_l683_68388

variable (a : ℝ)

theorem calculate_expression : (-a) ^ 2 * (-a ^ 5) ^ 4 / a ^ 12 * (-2 * a ^ 4) = -2 * a ^ 14 := 
by sorry

end NUMINAMATH_GPT_calculate_expression_l683_68388


namespace NUMINAMATH_GPT_worst_player_is_nephew_l683_68317

-- Define the family members
inductive Player
| father : Player
| sister : Player
| son : Player
| nephew : Player

open Player

-- Define a twin relationship
def is_twin (p1 p2 : Player) : Prop :=
  (p1 = son ∧ p2 = nephew) ∨ (p1 = nephew ∧ p2 = son)

-- Define that two players are of opposite sex
def opposite_sex (p1 p2 : Player) : Prop :=
  (p1 = sister ∧ (p2 = father ∨ p2 = son ∨ p2 = nephew)) ∨
  (p2 = sister ∧ (p1 = father ∨ p1 = son ∨ p1 = nephew))

-- Predicate for the worst player
structure WorstPlayer (p : Player) : Prop :=
  (twin_exists : ∃ twin : Player, is_twin p twin)
  (opposite_sex_best : ∀ twin best, is_twin p twin → best ≠ twin → opposite_sex twin best)

-- The goal is to show that the worst player is the nephew
theorem worst_player_is_nephew : WorstPlayer nephew := sorry

end NUMINAMATH_GPT_worst_player_is_nephew_l683_68317


namespace NUMINAMATH_GPT_smallest_integer_in_consecutive_set_l683_68323

theorem smallest_integer_in_consecutive_set :
  ∃ (n : ℤ), 2 < n ∧ ∀ m : ℤ, m < n → ¬ (m + 6 < 2 * (m + 3) - 2) :=
sorry

end NUMINAMATH_GPT_smallest_integer_in_consecutive_set_l683_68323


namespace NUMINAMATH_GPT_smallest_positive_b_l683_68375

def periodic_10 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 10) = f x

theorem smallest_positive_b
  (f : ℝ → ℝ)
  (h : periodic_10 f) :
  ∀ x, f ((x - 20) / 2) = f (x / 2) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_b_l683_68375


namespace NUMINAMATH_GPT_amplitude_of_cosine_function_is_3_l683_68374

variable (a b : ℝ)
variable (h_a : a > 0)
variable (h_b : b > 0)
variable (h_max : ∀ x : ℝ, a * Real.cos (b * x) ≤ 3)
variable (h_cycle : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ (∀ x : ℝ, a * Real.cos (b * (x + 2 * Real.pi)) = a * Real.cos (b * x)))

theorem amplitude_of_cosine_function_is_3 :
  a = 3 :=
sorry

end NUMINAMATH_GPT_amplitude_of_cosine_function_is_3_l683_68374


namespace NUMINAMATH_GPT_union_M_N_eq_l683_68340

open Set

-- Define set M and set N according to the problem conditions
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = x^2}

-- The theorem we need to prove
theorem union_M_N_eq : M ∪ N = {0, 1, 2, 4} :=
by
  -- Just assert the theorem without proving it
  sorry

end NUMINAMATH_GPT_union_M_N_eq_l683_68340


namespace NUMINAMATH_GPT_find_f_l683_68363

theorem find_f (f : ℕ → ℕ) :
  (∀ a b c : ℕ, ((f a + f b + f c) - a * b - b * c - c * a) ∣ (a * f a + b * f b + c * f c - 3 * a * b * c)) →
  (∀ n : ℕ, f n = n * n) :=
sorry

end NUMINAMATH_GPT_find_f_l683_68363


namespace NUMINAMATH_GPT_find_three_digit_number_l683_68331

theorem find_three_digit_number (A B C D : ℕ) 
  (h1 : A + C = 5) 
  (h2 : B = 3)
  (h3 : A * 100 + B * 10 + C + 124 = D * 111) 
  (h4 : A ≠ B ∧ A ≠ C ∧ B ≠ C) : 
  A * 100 + B * 10 + C = 431 := 
by 
  sorry

end NUMINAMATH_GPT_find_three_digit_number_l683_68331


namespace NUMINAMATH_GPT_black_car_speed_l683_68318

theorem black_car_speed
  (red_speed black_speed : ℝ)
  (initial_distance time : ℝ)
  (red_speed_eq : red_speed = 10)
  (initial_distance_eq : initial_distance = 20)
  (time_eq : time = 0.5)
  (distance_eq : black_speed * time = initial_distance + red_speed * time) :
  black_speed = 50 := by
  rw [red_speed_eq, initial_distance_eq, time_eq] at distance_eq
  sorry

end NUMINAMATH_GPT_black_car_speed_l683_68318


namespace NUMINAMATH_GPT_sarees_shirts_cost_l683_68387

variable (S T : ℕ)

-- Definition of conditions
def condition1 : Prop := 2 * S + 4 * T = 2 * S + 4 * T
def condition2 : Prop := (S + 6 * T) = (2 * S + 4 * T)
def condition3 : Prop := 12 * T = 2400

-- Proof goal
theorem sarees_shirts_cost :
  condition1 S T → condition2 S T → condition3 T → 2 * S + 4 * T = 1600 := by
  sorry

end NUMINAMATH_GPT_sarees_shirts_cost_l683_68387


namespace NUMINAMATH_GPT_abs_inequalities_imply_linear_relationship_l683_68307

theorem abs_inequalities_imply_linear_relationship (a b c : ℝ)
(h1 : |a - b| ≥ |c|)
(h2 : |b - c| ≥ |a|)
(h3 : |c - a| ≥ |b|) :
a = b + c ∨ b = c + a ∨ c = a + b :=
sorry

end NUMINAMATH_GPT_abs_inequalities_imply_linear_relationship_l683_68307


namespace NUMINAMATH_GPT_remaining_tickets_equation_l683_68327

-- Define the constants and variables
variables (x y : ℕ)

-- Conditions from the problem
def tickets_whack_a_mole := 32
def tickets_skee_ball := 25
def tickets_space_invaders : ℕ := x

def spent_hat := 7
def spent_keychain := 10
def spent_toy := 15

-- Define the condition for the total number of tickets spent
def total_tickets_spent := spent_hat + spent_keychain + spent_toy
-- Prove the remaining tickets equation
theorem remaining_tickets_equation : y = (tickets_whack_a_mole + tickets_skee_ball + tickets_space_invaders) - total_tickets_spent ->
                                      y = 25 + x :=
by
  sorry

end NUMINAMATH_GPT_remaining_tickets_equation_l683_68327


namespace NUMINAMATH_GPT_eval_sequence_l683_68313

noncomputable def b : ℕ → ℤ
| 1 => 1
| 2 => 4
| 3 => 9
| n => if h : n > 3 then b (n - 1) * (b (n - 1) - 1) + 1 else 0

theorem eval_sequence :
  b 1 * b 2 * b 3 * b 4 * b 5 * b 6 - (b 1 ^ 2 + b 2 ^ 2 + b 3 ^ 2 + b 4 ^ 2 + b 5 ^ 2 + b 6 ^ 2)
  = -3166598256 :=
by
  /- The proof steps are omitted. -/
  sorry

end NUMINAMATH_GPT_eval_sequence_l683_68313


namespace NUMINAMATH_GPT_prove_f_2013_l683_68378

-- Defining the function f that satisfies the given conditions
variable (f : ℕ → ℕ)

-- Conditions provided in the problem
axiom cond1 : ∀ n, f (f n) + f n = 2 * n + 3
axiom cond2 : f 0 = 1
axiom cond3 : f 2014 = 2015

-- The statement to be proven
theorem prove_f_2013 : f 2013 = 2014 := sorry

end NUMINAMATH_GPT_prove_f_2013_l683_68378


namespace NUMINAMATH_GPT_scientific_notation_2150000_l683_68371

theorem scientific_notation_2150000 : 2150000 = 2.15 * 10^6 :=
  by
  sorry

end NUMINAMATH_GPT_scientific_notation_2150000_l683_68371


namespace NUMINAMATH_GPT_polynomial_remainder_l683_68341

theorem polynomial_remainder :
  ∀ (q : Polynomial ℚ),
  (q.eval 2 = 8) →
  (q.eval (-3) = -10) →
  ∃ c d : ℚ, (q = (Polynomial.C (c : ℚ) * (Polynomial.X - Polynomial.C 2) * (Polynomial.X + Polynomial.C 3)) + (Polynomial.C 3.6 * Polynomial.X + Polynomial.C 0.8)) :=
by intros q h1 h2; sorry

end NUMINAMATH_GPT_polynomial_remainder_l683_68341


namespace NUMINAMATH_GPT_correct_calculation_l683_68349

theorem correct_calculation :
  (3 * Real.sqrt 2) * (2 * Real.sqrt 3) = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l683_68349


namespace NUMINAMATH_GPT_tino_more_jellybeans_than_lee_l683_68328

-- Declare the conditions
variables (arnold_jellybeans lee_jellybeans tino_jellybeans : ℕ)
variables (arnold_jellybeans_half_lee : arnold_jellybeans = lee_jellybeans / 2)
variables (arnold_jellybean_count : arnold_jellybeans = 5)
variables (tino_jellybean_count : tino_jellybeans = 34)

-- The goal is to prove how many more jellybeans Tino has than Lee
theorem tino_more_jellybeans_than_lee : tino_jellybeans - lee_jellybeans = 24 :=
by
  sorry -- proof skipped

end NUMINAMATH_GPT_tino_more_jellybeans_than_lee_l683_68328


namespace NUMINAMATH_GPT_find_k_l683_68391

-- Define vectors a, b, and c
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 3)
def c (k : ℝ) : ℝ × ℝ := (k, 2)

-- Define the dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for perpendicular vectors
def perpendicular_condition (k : ℝ) : Prop :=
  dot_product (a.1 - k, -1) b = 0

-- State the theorem
theorem find_k : ∃ k : ℝ, perpendicular_condition k ∧ k = 0 := by
  sorry

end NUMINAMATH_GPT_find_k_l683_68391


namespace NUMINAMATH_GPT_case1_DC_correct_case2_DC_correct_l683_68306

-- Case 1
theorem case1_DC_correct (AB AD : ℝ) (HM BD DH MD : ℝ) (hAB : AB = 10) (hAD : AD = 4)
  (hHM : HM = 6 / 5) (hBD : BD = 2 * Real.sqrt 21) (hDH : DH = 4 * Real.sqrt 21 / 5)
  (hMD : MD = 6 * (Real.sqrt 21 - 1) / 5):
  (BD - HM : ℝ) == (8 * Real.sqrt 21 - 12) / 5 :=
by {
  sorry
}

-- Case 2
theorem case2_DC_correct (AB AD : ℝ) (HM BD DH MD : ℝ) (hAB : AB = 8 * Real.sqrt 2) (hAD : AD = 4)
  (hHM : HM = Real.sqrt 2) (hBD : BD = 4 * Real.sqrt 7) (hDH : DH = Real.sqrt 14)
  (hMD : MD = Real.sqrt 14 - Real.sqrt 2):
  (BD - HM : ℝ) == 2 * Real.sqrt 14 - 2 * Real.sqrt 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_case1_DC_correct_case2_DC_correct_l683_68306


namespace NUMINAMATH_GPT_width_of_room_l683_68373

theorem width_of_room (length : ℝ) (cost_rate : ℝ) (total_cost : ℝ) (width : ℝ)
  (h1 : length = 5.5)
  (h2 : cost_rate = 800)
  (h3 : total_cost = 16500)
  (h4 : width = total_cost / cost_rate / length) : width = 3.75 :=
by
  sorry

end NUMINAMATH_GPT_width_of_room_l683_68373


namespace NUMINAMATH_GPT_initial_average_weight_l683_68353

theorem initial_average_weight
  (A : ℝ)
  (h : 30 * 27.4 - 10 = 29 * A) : 
  A = 28 := 
by
  sorry

end NUMINAMATH_GPT_initial_average_weight_l683_68353


namespace NUMINAMATH_GPT_fraction_students_walk_home_l683_68325

theorem fraction_students_walk_home :
  let bus := 1/3
  let auto := 1/5
  let bicycle := 1/8
  let total_students := 1
  let other_transportation := bus + auto + bicycle
  let walk_home := total_students - other_transportation
  walk_home = 41/120 :=
by 
  let bus := 1/3
  let auto := 1/5
  let bicycle := 1/8
  let total_students := 1
  let other_transportation := bus + auto + bicycle
  let walk_home := total_students - other_transportation
  have h_bus : bus = 40 / 120 := by sorry
  have h_auto : auto = 24 / 120 := by sorry
  have h_bicycle : bicycle = 15 / 120 := by sorry
  have h_total_transportation : other_transportation = 40 / 120 + 24 / 120 + 15 / 120 := by sorry
  have h_other_transportation_sum : other_transportation = 79 / 120 := by sorry
  have h_walk_home : walk_home = 1 - 79 / 120 := by sorry
  have h_walk_home_simplified : walk_home = 41 / 120 := by sorry
  exact h_walk_home_simplified

end NUMINAMATH_GPT_fraction_students_walk_home_l683_68325


namespace NUMINAMATH_GPT_solution_is_correct_l683_68385

def valid_triple (a b c : ℕ) : Prop :=
  (Nat.gcd a 20 = b) ∧ (Nat.gcd b 15 = c) ∧ (Nat.gcd a c = 5)

def is_solution_set (triples : Set (ℕ × ℕ × ℕ)) : Prop :=
  ∀ a b c, (a, b, c) ∈ triples ↔ 
    (valid_triple a b c) ∧ 
    ((∃ k, a = 20 * k ∧ b = 20 ∧ c = 5) ∨
    (∃ k, a = 20 * k - 10 ∧ b = 10 ∧ c = 5) ∨
    (∃ k, a = 10 * k - 5 ∧ b = 5 ∧ c = 5))

theorem solution_is_correct : ∃ S, is_solution_set S :=
sorry

end NUMINAMATH_GPT_solution_is_correct_l683_68385


namespace NUMINAMATH_GPT_proposition_A_proposition_B_proposition_C_proposition_D_l683_68396

theorem proposition_A (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a ≠ 1) :=
by {
  sorry
}

theorem proposition_B : (¬ ∀ x : ℝ, x^2 + x + 1 < 0) → (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≥ 0) :=
by {
  sorry
}

theorem proposition_C : ¬ ∀ x ≠ 0, x + 1 / x ≥ 2 :=
by {
  sorry
}

theorem proposition_D (m : ℝ) : (∃ x : ℝ, (1 < x ∧ x < 2) ∧ x^2 + m * x + 4 < 0) → m < -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_proposition_A_proposition_B_proposition_C_proposition_D_l683_68396


namespace NUMINAMATH_GPT_central_angle_of_sector_l683_68342

theorem central_angle_of_sector (alpha : ℝ) (l : ℝ) (A : ℝ) (h1 : l = 2 * Real.pi) (h2 : A = 5 * Real.pi) : 
  alpha = 72 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l683_68342


namespace NUMINAMATH_GPT_number_of_players_l683_68311

-- Definitions of the conditions
def initial_bottles : ℕ := 4 * 12
def bottles_remaining : ℕ := 15
def bottles_taken_per_player : ℕ := 2 + 1

-- Total number of bottles taken
def bottles_taken := initial_bottles - bottles_remaining

-- The main theorem stating that the number of players is 11.
theorem number_of_players : (bottles_taken / bottles_taken_per_player) = 11 :=
by
  sorry

end NUMINAMATH_GPT_number_of_players_l683_68311


namespace NUMINAMATH_GPT_geometric_sequences_common_ratios_l683_68347

theorem geometric_sequences_common_ratios 
  (k m n o : ℝ)
  (a_2 a_3 b_2 b_3 c_2 c_3 : ℝ)
  (h1 : a_2 = k * m)
  (h2 : a_3 = k * m^2)
  (h3 : b_2 = k * n)
  (h4 : b_3 = k * n^2)
  (h5 : c_2 = k * o)
  (h6 : c_3 = k * o^2)
  (h7 : a_3 - b_3 + c_3 = 2 * (a_2 - b_2 + c_2))
  (h8 : m ≠ n)
  (h9 : m ≠ o)
  (h10 : n ≠ o) : 
  m + n + o = 1 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequences_common_ratios_l683_68347


namespace NUMINAMATH_GPT_linear_function_max_value_l683_68339

theorem linear_function_max_value (m x : ℝ) (h : -1 ≤ x ∧ x ≤ 3) (y : ℝ) 
  (hl : y = m * x - 2 * m) (hy : y = 6) : m = -2 ∨ m = 6 := 
by 
  sorry

end NUMINAMATH_GPT_linear_function_max_value_l683_68339


namespace NUMINAMATH_GPT_power_function_value_at_half_l683_68370

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2 - x) - 3/4

noncomputable def g (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_value_at_half (a : ℝ) (α : ℝ) 
  (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : f a 2 = 1 / 4) (h4 : g α 2 = 1 / 4) : 
  g α (1/2) = 4 := 
by
  sorry

end NUMINAMATH_GPT_power_function_value_at_half_l683_68370


namespace NUMINAMATH_GPT_avg_annual_growth_rate_profit_exceeds_340_l683_68343

variable (P2018 P2020 : ℝ)
variable (r : ℝ)

theorem avg_annual_growth_rate :
    P2018 = 200 → P2020 = 288 →
    (1 + r)^2 = P2020 / P2018 →
    r = 0.2 :=
by
  intros hP2018 hP2020 hGrowth
  sorry

theorem profit_exceeds_340 (P2020 : ℝ) (r : ℝ) :
    P2020 = 288 → r = 0.2 →
    P2020 * (1 + r) > 340 :=
by
  intros hP2020 hr
  sorry

end NUMINAMATH_GPT_avg_annual_growth_rate_profit_exceeds_340_l683_68343
