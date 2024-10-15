import Mathlib

namespace NUMINAMATH_GPT_random_event_proof_l1043_104331

-- Definitions based on conditions
def event1 := "Tossing a coin twice in a row, and both times it lands heads up."
def event2 := "Opposite charges attract each other."
def event3 := "Water freezes at 1℃ under standard atmospheric pressure."

def is_random_event (event: String) : Prop :=
  event = event1 ∨ event = event2 ∨ event = event3 → event = event1

theorem random_event_proof : is_random_event event1 ∧ ¬is_random_event event2 ∧ ¬is_random_event event3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_random_event_proof_l1043_104331


namespace NUMINAMATH_GPT_number_of_apple_trees_l1043_104351

variable (T : ℕ) -- Declare the number of apple trees as a natural number

-- Define the conditions
def picked_apples := 8 * T
def remaining_apples := 9
def initial_apples := 33

-- The statement to prove Rachel has 3 apple trees
theorem number_of_apple_trees :
  initial_apples - picked_apples + remaining_apples = initial_apples → T = 3 := 
by
  sorry

end NUMINAMATH_GPT_number_of_apple_trees_l1043_104351


namespace NUMINAMATH_GPT_range_of_a_l1043_104381

theorem range_of_a (x y a : ℝ) (h1 : x - y = 2) (h2 : x + y = a) (h3 : x > -1) (h4 : y < 0) : -4 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1043_104381


namespace NUMINAMATH_GPT_possible_permutations_100_l1043_104377

def tasty_permutations (n : ℕ) : ℕ := sorry

theorem possible_permutations_100 :
  2^100 ≤ tasty_permutations 100 ∧ tasty_permutations 100 ≤ 4^100 :=
sorry

end NUMINAMATH_GPT_possible_permutations_100_l1043_104377


namespace NUMINAMATH_GPT_train_a_distance_at_meeting_l1043_104350

noncomputable def train_a_speed : ℝ := 75 / 3
noncomputable def train_b_speed : ℝ := 75 / 2
noncomputable def relative_speed : ℝ := train_a_speed + train_b_speed
noncomputable def time_until_meet : ℝ := 75 / relative_speed
noncomputable def distance_traveled_by_train_a : ℝ := train_a_speed * time_until_meet

theorem train_a_distance_at_meeting : distance_traveled_by_train_a = 30 := by
  sorry

end NUMINAMATH_GPT_train_a_distance_at_meeting_l1043_104350


namespace NUMINAMATH_GPT_find_partition_l1043_104354

open Nat

def isBad (S : Finset ℕ) : Prop :=
  ∃ T : Finset ℕ, T ⊆ S ∧ T.sum id = 2012

def partition_not_bad (S : Finset ℕ) (n : ℕ) : Prop :=
  ∃ (P : Finset (Finset ℕ)), P.card = n ∧ (∀ p ∈ P, isBad p = false) ∧ (S = P.sup id)

theorem find_partition :
  ∃ n : ℕ, n = 2 ∧ partition_not_bad (Finset.range (2012 - 503) \ Finset.range 503) n :=
by
  sorry

end NUMINAMATH_GPT_find_partition_l1043_104354


namespace NUMINAMATH_GPT_part1_part2_l1043_104356

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1043_104356


namespace NUMINAMATH_GPT_P_sufficient_but_not_necessary_for_Q_l1043_104332

-- Definitions based on given conditions
def P (x : ℝ) : Prop := abs (2 * x - 3) < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

-- The theorem to prove that P is sufficient but not necessary for Q
theorem P_sufficient_but_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬P x) :=
by
  sorry

end NUMINAMATH_GPT_P_sufficient_but_not_necessary_for_Q_l1043_104332


namespace NUMINAMATH_GPT_bullet_train_pass_time_l1043_104311

noncomputable def time_to_pass (length_train : ℕ) (speed_train_kmph : ℕ) (speed_man_kmph : ℕ) : ℝ := 
  let relative_speed_kmph := speed_train_kmph + speed_man_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * 1000 / 3600
  length_train / relative_speed_mps

def length_train := 350
def speed_train_kmph := 75
def speed_man_kmph := 12

theorem bullet_train_pass_time : 
  abs (time_to_pass length_train speed_train_kmph speed_man_kmph - 14.47) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_bullet_train_pass_time_l1043_104311


namespace NUMINAMATH_GPT_tickets_per_friend_l1043_104374

-- Defining the conditions
def initial_tickets := 11
def remaining_tickets := 3
def friends := 4

-- Statement to prove
theorem tickets_per_friend (h_tickets_given : initial_tickets - remaining_tickets = 8) : (initial_tickets - remaining_tickets) / friends = 2 :=
by
  sorry

end NUMINAMATH_GPT_tickets_per_friend_l1043_104374


namespace NUMINAMATH_GPT_problem_statement_l1043_104303

theorem problem_statement (x m : ℝ) :
  (¬ (x > m) → ¬ (x^2 + x - 2 > 0)) ∧ (¬ (x > m) ↔ ¬ (x^2 + x - 2 > 0)) → m ≥ 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1043_104303


namespace NUMINAMATH_GPT_min_value_of_reciprocals_l1043_104363

open Real

theorem min_value_of_reciprocals (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 1) :
  (1 / a) + (1 / (b + 1)) ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_reciprocals_l1043_104363


namespace NUMINAMATH_GPT_largest_multiple_of_9_lt_120_is_117_l1043_104372

theorem largest_multiple_of_9_lt_120_is_117 : ∃ k : ℕ, 9 * k < 120 ∧ (∀ m : ℕ, 9 * m < 120 → 9 * m ≤ 9 * k) ∧ 9 * k = 117 := 
by 
  sorry

end NUMINAMATH_GPT_largest_multiple_of_9_lt_120_is_117_l1043_104372


namespace NUMINAMATH_GPT_max_sum_of_segments_l1043_104341

theorem max_sum_of_segments (A B C D : ℝ × ℝ × ℝ)
    (h : (dist A B ≤ 1 ∧ dist A C ≤ 1 ∧ dist A D ≤ 1 ∧ dist B C ≤ 1 ∧ dist B D ≤ 1 ∧ dist C D ≤ 1)
      ∨ (dist A B ≤ 1 ∧ dist A C ≤ 1 ∧ dist A D > 1 ∧ dist B C ≤ 1 ∧ dist B D ≤ 1 ∧ dist C D ≤ 1))
    : dist A B + dist A C + dist A D + dist B C + dist B D + dist C D ≤ 5 + Real.sqrt 3 := sorry

end NUMINAMATH_GPT_max_sum_of_segments_l1043_104341


namespace NUMINAMATH_GPT_repeating_decimal_conversion_l1043_104314

-- Definition of 0.\overline{23} as a rational number
def repeating_decimal_fraction : ℚ := 23 / 99

-- The main statement to prove
theorem repeating_decimal_conversion : (3 / 10) + (repeating_decimal_fraction) = 527 / 990 := 
by
  -- Placeholder for proof steps
  sorry

end NUMINAMATH_GPT_repeating_decimal_conversion_l1043_104314


namespace NUMINAMATH_GPT_vanya_number_l1043_104352

theorem vanya_number (m n : ℕ) (hm : m < 10) (hn : n < 10) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 
  10 * m + n = 81 :=
by sorry

end NUMINAMATH_GPT_vanya_number_l1043_104352


namespace NUMINAMATH_GPT_average_of_middle_three_l1043_104337

theorem average_of_middle_three
  (a b c d e : ℕ)
  (h_distinct : a < b ∧ b < c ∧ c < d ∧ d < e)
  (h_sum : a + b + c + d + e = 25)
  (h_max_diff : ∀ x y : ℕ, x + y = 24 → (e - a) ≥ (y - x)) :
  (b + c + d) / 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_average_of_middle_three_l1043_104337


namespace NUMINAMATH_GPT_second_term_of_geometric_series_l1043_104309

theorem second_term_of_geometric_series 
  (a : ℝ) (r : ℝ) (S : ℝ) :
  r = 1 / 4 → S = 40 → S = a / (1 - r) → a * r = 7.5 :=
by
  intros hr hS hSum
  sorry

end NUMINAMATH_GPT_second_term_of_geometric_series_l1043_104309


namespace NUMINAMATH_GPT_insulin_pills_per_day_l1043_104386

def conditions (I B A : ℕ) : Prop := 
  B = 3 ∧ A = 2 * B ∧ 7 * (I + B + A) = 77

theorem insulin_pills_per_day : ∃ (I : ℕ), ∀ (B A : ℕ), conditions I B A → I = 2 := by
  sorry

end NUMINAMATH_GPT_insulin_pills_per_day_l1043_104386


namespace NUMINAMATH_GPT_tan_product_in_triangle_l1043_104345

theorem tan_product_in_triangle (A B C : ℝ) (h1 : A + B + C = Real.pi)
  (h2 : Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = Real.sin B ^ 2) :
  Real.tan A * Real.tan C = 1 :=
sorry

end NUMINAMATH_GPT_tan_product_in_triangle_l1043_104345


namespace NUMINAMATH_GPT_white_to_brown_eggs_ratio_l1043_104364

-- Define variables W and B (the initial numbers of white and brown eggs respectively)
variable (W B : ℕ)

-- Conditions: 
-- 1. All 5 brown eggs survived.
-- 2. Total number of eggs after dropping is 12.
def egg_conditions : Prop :=
  B = 5 ∧ (W + B) = 12

-- Prove the ratio of white eggs to brown eggs is 7/5 given these conditions.
theorem white_to_brown_eggs_ratio (h : egg_conditions W B) : W / B = 7 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_white_to_brown_eggs_ratio_l1043_104364


namespace NUMINAMATH_GPT_percentage_passed_both_subjects_l1043_104376

def failed_H : ℝ := 0.35
def failed_E : ℝ := 0.45
def failed_HE : ℝ := 0.20

theorem percentage_passed_both_subjects :
  (100 - (failed_H * 100 + failed_E * 100 - failed_HE * 100)) = 40 := 
by
  sorry

end NUMINAMATH_GPT_percentage_passed_both_subjects_l1043_104376


namespace NUMINAMATH_GPT_pq_sum_l1043_104388

theorem pq_sum {p q : ℤ}
  (h : ∀ x : ℤ, 36 * x^2 - 4 * (p^2 + 11) * x + 135 * (p + q) + 576 = 0) :
  p + q = 20 :=
sorry

end NUMINAMATH_GPT_pq_sum_l1043_104388


namespace NUMINAMATH_GPT_largest_possible_s_l1043_104397

theorem largest_possible_s (r s : ℕ) (h1 : r ≥ s) (h2 : s ≥ 3) (h3 : (r - 2) * 180 * s = (s - 2) * 180 * r * 61 / 60) : s = 118 :=
sorry

end NUMINAMATH_GPT_largest_possible_s_l1043_104397


namespace NUMINAMATH_GPT_economical_club_l1043_104319

-- Definitions of cost functions for Club A and Club B
def f (x : ℕ) : ℕ := 5 * x

def g (x : ℕ) : ℕ := if x ≤ 30 then 90 else 2 * x + 30

-- Theorem to determine the more economical club
theorem economical_club (x : ℕ) (hx : 15 ≤ x ∧ x ≤ 40) :
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 30 → f x > g x) ∧
  (30 < x ∧ x ≤ 40 → f x > g x) :=
sorry

end NUMINAMATH_GPT_economical_club_l1043_104319


namespace NUMINAMATH_GPT_taxi_fare_relationship_taxi_fare_relationship_simplified_l1043_104339

variable (x : ℝ) (y : ℝ)

-- Conditions
def starting_fare : ℝ := 14
def additional_fare_per_km : ℝ := 2.4
def initial_distance : ℝ := 3
def total_distance (x : ℝ) := x
def total_fare (x : ℝ) (y : ℝ) := y
def distance_condition (x : ℝ) := x > 3

-- Theorem Statement
theorem taxi_fare_relationship (h : distance_condition x) :
  total_fare x y = additional_fare_per_km * (total_distance x - initial_distance) + starting_fare :=
by
  sorry

-- Simplified Theorem Statement
theorem taxi_fare_relationship_simplified (h : distance_condition x) :
  y = 2.4 * x + 6.8 :=
by
  sorry

end NUMINAMATH_GPT_taxi_fare_relationship_taxi_fare_relationship_simplified_l1043_104339


namespace NUMINAMATH_GPT_students_present_in_class_l1043_104378

theorem students_present_in_class :
  ∀ (total_students absent_percentage : ℕ), 
    total_students = 50 → absent_percentage = 12 → 
    (88 * total_students / 100) = 44 :=
by
  intros total_students absent_percentage h1 h2
  sorry

end NUMINAMATH_GPT_students_present_in_class_l1043_104378


namespace NUMINAMATH_GPT_bread_problem_l1043_104349

variable (x : ℝ)

theorem bread_problem (h1 : x > 0) :
  (15 / x) - 1 = 14 / (x + 2) :=
sorry

end NUMINAMATH_GPT_bread_problem_l1043_104349


namespace NUMINAMATH_GPT_yogurt_count_l1043_104375

theorem yogurt_count (Y : ℕ) 
  (ice_cream_cartons : ℕ := 20)
  (cost_ice_cream_per_carton : ℕ := 6)
  (cost_yogurt_per_carton : ℕ := 1)
  (spent_more_on_ice_cream : ℕ := 118)
  (total_cost_ice_cream : ℕ := ice_cream_cartons * cost_ice_cream_per_carton)
  (total_cost_yogurt : ℕ := Y * cost_yogurt_per_carton)
  (expenditure_condition : total_cost_ice_cream = total_cost_yogurt + spent_more_on_ice_cream) :
  Y = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_yogurt_count_l1043_104375


namespace NUMINAMATH_GPT_digit_makes_5678d_multiple_of_9_l1043_104334

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

theorem digit_makes_5678d_multiple_of_9 (d : Nat) (h : d ≥ 0 ∧ d < 10) :
  is_multiple_of_9 (5 * 10000 + 6 * 1000 + 7 * 100 + 8 * 10 + d) ↔ d = 1 := 
by
  sorry

end NUMINAMATH_GPT_digit_makes_5678d_multiple_of_9_l1043_104334


namespace NUMINAMATH_GPT_xyz_value_l1043_104385

noncomputable def positive (x : ℝ) : Prop := 0 < x

theorem xyz_value (x y z : ℝ) (hx : positive x) (hy : positive y) (hz : positive z): 
  (x + 1/y = 5) → (y + 1/z = 2) → (z + 1/x = 8/3) → x * y * z = (17 + Real.sqrt 285) / 2 :=
by
  sorry

end NUMINAMATH_GPT_xyz_value_l1043_104385


namespace NUMINAMATH_GPT_point_line_real_assoc_l1043_104325

theorem point_line_real_assoc : 
  ∀ (p : ℝ), ∃! (r : ℝ), p = r := 
by 
  sorry

end NUMINAMATH_GPT_point_line_real_assoc_l1043_104325


namespace NUMINAMATH_GPT_range_of_a_l1043_104384

def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

theorem range_of_a (a : ℝ) : (-2 / 3 : ℝ) ≤ a ∧ a < 0 := sorry

end NUMINAMATH_GPT_range_of_a_l1043_104384


namespace NUMINAMATH_GPT_brownies_per_person_l1043_104391

-- Define the conditions as constants
def columns : ℕ := 6
def rows : ℕ := 3
def people : ℕ := 6

-- Define the total number of brownies
def total_brownies : ℕ := columns * rows

-- Define the theorem to be proved
theorem brownies_per_person : total_brownies / people = 3 :=
by sorry

end NUMINAMATH_GPT_brownies_per_person_l1043_104391


namespace NUMINAMATH_GPT_problem_1_problem_2_l1043_104300

-- Proposition p
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2 * a * x + 2 - a)

-- Proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 + 2 * x + a ≥ 0

-- Problem 1: Prove that if p is true then a ≤ -2 or a ≥ 1
theorem problem_1 (a : ℝ) (hp : p a) : a ≤ -2 ∨ a ≥ 1 := sorry

-- Problem 2: Prove that if p ∨ q is true then a ≤ -2 or a ≥ 0
theorem problem_2 (a : ℝ) (hpq : p a ∨ q a) : a ≤ -2 ∨ a ≥ 0 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1043_104300


namespace NUMINAMATH_GPT_parameter_values_for_three_distinct_roots_l1043_104347

theorem parameter_values_for_three_distinct_roots (a : ℝ) :
  (∀ x : ℝ, (|x^3 - a^3| = x - a) → (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ↔ 
  (-2 / Real.sqrt 3 < a ∧ a < -1 / Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_parameter_values_for_three_distinct_roots_l1043_104347


namespace NUMINAMATH_GPT_Maria_drove_approximately_517_miles_l1043_104394

noncomputable def carRentalMaria (daily_rate per_mile_charge discount_rate insurance_rate rental_duration total_invoice : ℝ) (discount_threshold : ℕ) : ℝ :=
  let total_rental_cost := rental_duration * daily_rate
  let discount := if rental_duration ≥ discount_threshold then discount_rate * total_rental_cost else 0
  let discounted_cost := total_rental_cost - discount
  let insurance_cost := rental_duration * insurance_rate
  let cost_without_mileage := discounted_cost + insurance_cost
  let mileage_cost := total_invoice - cost_without_mileage
  mileage_cost / per_mile_charge

noncomputable def approx_equal (a b : ℝ) (epsilon : ℝ := 1) : Prop :=
  abs (a - b) < epsilon

theorem Maria_drove_approximately_517_miles :
  approx_equal (carRentalMaria 35 0.09 0.10 5 4 192.50 3) 517 :=
by
  sorry

end NUMINAMATH_GPT_Maria_drove_approximately_517_miles_l1043_104394


namespace NUMINAMATH_GPT_line_equation_slope_intercept_l1043_104333

theorem line_equation_slope_intercept (m b : ℝ) (h1 : m = -1) (h2 : b = -1) :
  ∀ x y : ℝ, y = m * x + b → x + y + 1 = 0 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_line_equation_slope_intercept_l1043_104333


namespace NUMINAMATH_GPT_thirteenth_term_is_correct_l1043_104320

noncomputable def third_term : ℚ := 2 / 11
noncomputable def twenty_third_term : ℚ := 3 / 7

theorem thirteenth_term_is_correct : 
  (third_term + twenty_third_term) / 2 = 47 / 154 := sorry

end NUMINAMATH_GPT_thirteenth_term_is_correct_l1043_104320


namespace NUMINAMATH_GPT_angle_B_in_equilateral_triangle_l1043_104324

theorem angle_B_in_equilateral_triangle (A B C : ℝ) (h_angle_sum : A + B + C = 180) (h_A : A = 80) (h_BC : B = C) :
  B = 50 :=
by
  -- Conditions
  have h1 : A = 80 := by exact h_A
  have h2 : B = C := by exact h_BC
  have h3 : A + B + C = 180 := by exact h_angle_sum

  sorry -- completing the proof is not required

end NUMINAMATH_GPT_angle_B_in_equilateral_triangle_l1043_104324


namespace NUMINAMATH_GPT_annalise_total_cost_l1043_104399

/-- 
Given conditions:
- 25 boxes of tissues.
- Each box contains 18 packs.
- Each pack contains 150 tissues.
- Each tissue costs $0.06.
- A 10% discount on the total price of the packs in each box.

Prove:
The total amount of money Annalise spent is $3645.
-/
theorem annalise_total_cost :
  let boxes := 25
  let packs_per_box := 18
  let tissues_per_pack := 150
  let cost_per_tissue := 0.06
  let discount_rate := 0.10
  let price_per_box := (packs_per_box * tissues_per_pack * cost_per_tissue)
  let discount_per_box := discount_rate * price_per_box
  let discounted_price_per_box := price_per_box - discount_per_box
  let total_cost := discounted_price_per_box * boxes
  total_cost = 3645 :=
by
  sorry

end NUMINAMATH_GPT_annalise_total_cost_l1043_104399


namespace NUMINAMATH_GPT_gondor_repaired_3_phones_on_monday_l1043_104379

theorem gondor_repaired_3_phones_on_monday :
  ∃ P : ℕ, 
    (10 * P + 10 * 5 + 20 * 2 + 20 * 4 = 200) ∧
    P = 3 :=
by
  sorry

end NUMINAMATH_GPT_gondor_repaired_3_phones_on_monday_l1043_104379


namespace NUMINAMATH_GPT_h_has_only_one_zero_C2_below_C1_l1043_104362

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := 1 - 1/x
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem h_has_only_one_zero (x : ℝ) (hx : x > 0) : 
  ∃! (x0 : ℝ), x0 > 0 ∧ h x0 = 0 := sorry

theorem C2_below_C1 (x : ℝ) (hx : x > 0) (hx1 : x ≠ 1) : 
  g x < f x := sorry

end NUMINAMATH_GPT_h_has_only_one_zero_C2_below_C1_l1043_104362


namespace NUMINAMATH_GPT_gcd_binom_integer_l1043_104358

theorem gcd_binom_integer (n m : ℕ) (hnm : n ≥ m) (hm : m ≥ 1) :
  (Nat.gcd m n) * Nat.choose n m % n = 0 := sorry

end NUMINAMATH_GPT_gcd_binom_integer_l1043_104358


namespace NUMINAMATH_GPT_part1_part2_l1043_104366

-- Part 1
theorem part1 (x : ℝ) (h1 : 2 * x = 3 * x - 1) : x = 1 :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h2 : x < 0) (h3 : |2 * x| + |3 * x - 1| = 16) : x = -3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1043_104366


namespace NUMINAMATH_GPT_calculation_l1043_104369

theorem calculation : 
  ((18 ^ 13 * 18 ^ 11) ^ 2 / 6 ^ 8) * 3 ^ 4 = 2 ^ 40 * 3 ^ 92 :=
by sorry

end NUMINAMATH_GPT_calculation_l1043_104369


namespace NUMINAMATH_GPT_petya_vasya_three_numbers_equal_l1043_104355

theorem petya_vasya_three_numbers_equal (a b c : ℕ) :
  gcd a b = lcm a b ∧ gcd b c = lcm b c ∧ gcd a c = lcm a c → a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_petya_vasya_three_numbers_equal_l1043_104355


namespace NUMINAMATH_GPT_negation_correct_l1043_104335

variable {α : Type*} (A B : Set α)

-- Define the original proposition
def original_proposition : Prop := A ∪ B = A → A ∩ B = B

-- Define the negation of the original proposition
def negation_proposition : Prop := A ∪ B ≠ A → A ∩ B ≠ B

-- State that the negation of the original proposition is equivalent to the negation proposition
theorem negation_correct : ¬(original_proposition A B) ↔ negation_proposition A B := by sorry

end NUMINAMATH_GPT_negation_correct_l1043_104335


namespace NUMINAMATH_GPT_minimum_b_l1043_104327

noncomputable def f (a b x : ℝ) : ℝ := a * x + b

noncomputable def g (a b x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ a then f a b x else f a b (f a b x)

theorem minimum_b {a b : ℝ} (ha : 0 < a) :
  (∀ x : ℝ, 0 ≤ x → g a b x > g a b (x - 1)) → b ≥ 1 / 4 :=
sorry

end NUMINAMATH_GPT_minimum_b_l1043_104327


namespace NUMINAMATH_GPT_butterfat_in_final_mixture_l1043_104344

noncomputable def final_butterfat_percentage (gallons_of_35_percentage : ℕ) 
                                             (percentage_of_35_butterfat : ℝ) 
                                             (total_gallons : ℕ)
                                             (percentage_of_10_butterfat : ℝ) : ℝ :=
  let gallons_of_10 := total_gallons - gallons_of_35_percentage
  let butterfat_35 := gallons_of_35_percentage * percentage_of_35_butterfat
  let butterfat_10 := gallons_of_10 * percentage_of_10_butterfat
  let total_butterfat := butterfat_35 + butterfat_10
  (total_butterfat / total_gallons) * 100

theorem butterfat_in_final_mixture : 
  final_butterfat_percentage 8 0.35 12 0.10 = 26.67 :=
sorry

end NUMINAMATH_GPT_butterfat_in_final_mixture_l1043_104344


namespace NUMINAMATH_GPT_value_of_m_minus_n_l1043_104315

theorem value_of_m_minus_n (m n : ℝ) (h : (-3)^2 + m * (-3) + 3 * n = 0) : m - n = 3 :=
sorry

end NUMINAMATH_GPT_value_of_m_minus_n_l1043_104315


namespace NUMINAMATH_GPT_triangle_area_ratio_l1043_104329

/-
In triangle XYZ, XY=12, YZ=16, and XZ=20. Point D is on XY,
E is on YZ, and F is on XZ. Let XD=p*XY, YE=q*YZ, and ZF=r*XZ,
where p, q, r are positive and satisfy p+q+r=0.9 and p^2+q^2+r^2=0.29.
Prove that the ratio of the area of triangle DEF to the area of triangle XYZ 
can be written in the form m/n where m, n are relatively prime positive 
integers and m+n=137.
-/

theorem triangle_area_ratio :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m + n = 137 ∧ 
  ∃ (p q r : ℝ), p + q + r = 0.9 ∧ p^2 + q^2 + r^2 = 0.29 ∧ 
                  ∀ (XY YZ XZ : ℝ), XY = 12 ∧ YZ = 16 ∧ XZ = 20 → 
                  (1 - (p * (1 - r) + q * (1 - p) + r * (1 - q))) = (37 / 100) :=
by
   sorry

end NUMINAMATH_GPT_triangle_area_ratio_l1043_104329


namespace NUMINAMATH_GPT_find_smallest_N_l1043_104328

-- Define the sum of digits functions as described
def sum_of_digits_base (n : ℕ) (b : ℕ) : ℕ :=
  n.digits b |>.sum

-- Define f(n) which is the sum of digits in base-five representation of n
def f (n : ℕ) : ℕ :=
  sum_of_digits_base n 5

-- Define g(n) which is the sum of digits in base-seven representation of f(n)
def g (n : ℕ) : ℕ :=
  sum_of_digits_base (f n) 7

-- The statement of the problem: find the smallest N such that 
-- g(N) in base-sixteen cannot be represented using only digits 0 to 9
theorem find_smallest_N : ∃ N : ℕ, (g N ≥ 10) ∧ (N % 1000 = 610) :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_N_l1043_104328


namespace NUMINAMATH_GPT_difference_digits_in_base2_l1043_104322

def binaryDigitCount (n : Nat) : Nat := Nat.log2 n + 1

theorem difference_digits_in_base2 : binaryDigitCount 1400 - binaryDigitCount 300 = 2 :=
by
  sorry

end NUMINAMATH_GPT_difference_digits_in_base2_l1043_104322


namespace NUMINAMATH_GPT_min_digits_decimal_correct_l1043_104323

noncomputable def min_digits_decimal : ℕ := 
  let n : ℕ := 123456789
  let d : ℕ := 2^26 * 5^4
  26 -- As per the problem statement

theorem min_digits_decimal_correct :
  let n := 123456789
  let d := 2^26 * 5^4
  ∀ x:ℕ, (∃ k:ℕ, n = k * 10^x) → x ≥ min_digits_decimal := 
by
  sorry

end NUMINAMATH_GPT_min_digits_decimal_correct_l1043_104323


namespace NUMINAMATH_GPT_necessary_not_sufficient_l1043_104389

theorem necessary_not_sufficient (x : ℝ) : (x > 5) → (x > 2) ∧ ¬((x > 2) → (x > 5)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_l1043_104389


namespace NUMINAMATH_GPT_series_sum_l1043_104390

open BigOperators

theorem series_sum :
  (∑ n in Finset.range 99, (1 : ℝ) / ((n + 1) * (n + 2))) = 99 / 100 :=
by
  sorry

end NUMINAMATH_GPT_series_sum_l1043_104390


namespace NUMINAMATH_GPT_johns_previous_earnings_l1043_104348

theorem johns_previous_earnings (new_earnings raise_percentage old_earnings : ℝ) 
  (h1 : new_earnings = 68) (h2 : raise_percentage = 0.1333333333333334)
  (h3 : new_earnings = old_earnings * (1 + raise_percentage)) : old_earnings = 60 :=
sorry

end NUMINAMATH_GPT_johns_previous_earnings_l1043_104348


namespace NUMINAMATH_GPT_unread_pages_when_a_is_11_l1043_104308

variable (a : ℕ)

def total_pages : ℕ := 250
def pages_per_day : ℕ := 15

def unread_pages_after_a_days (a : ℕ) : ℕ := total_pages - pages_per_day * a

theorem unread_pages_when_a_is_11 : unread_pages_after_a_days 11 = 85 :=
by
  sorry

end NUMINAMATH_GPT_unread_pages_when_a_is_11_l1043_104308


namespace NUMINAMATH_GPT_marks_in_biology_l1043_104360

theorem marks_in_biology (E M P C : ℝ) (A B : ℝ)
  (h1 : E = 90)
  (h2 : M = 92)
  (h3 : P = 85)
  (h4 : C = 87)
  (h5 : A = 87.8) 
  (h6 : (E + M + P + C + B) / 5 = A) : 
  B = 85 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_marks_in_biology_l1043_104360


namespace NUMINAMATH_GPT_profit_percentage_approx_l1043_104368

-- Define the cost price of the first item
def CP1 (S1 : ℚ) : ℚ := 0.81 * S1

-- Define the selling price of the second item as 10% less than the first
def S2 (S1 : ℚ) : ℚ := 0.90 * S1

-- Define the cost price of the second item as 81% of its selling price
def CP2 (S1 : ℚ) : ℚ := 0.81 * (S2 S1)

-- Define the total selling price before tax
def TSP (S1 : ℚ) : ℚ := S1 + S2 S1

-- Define the total amount received after a 5% tax
def TAR (S1 : ℚ) : ℚ := TSP S1 * 0.95

-- Define the total cost price of both items
def TCP (S1 : ℚ) : ℚ := CP1 S1 + CP2 S1

-- Define the profit
def P (S1 : ℚ) : ℚ := TAR S1 - TCP S1

-- Define the profit percentage
def ProfitPercentage (S1 : ℚ) : ℚ := (P S1 / TCP S1) * 100

-- Prove the profit percentage is approximately 17.28%
theorem profit_percentage_approx (S1 : ℚ) : abs (ProfitPercentage S1 - 17.28) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_approx_l1043_104368


namespace NUMINAMATH_GPT_current_number_of_people_l1043_104301

theorem current_number_of_people (a b : ℕ) : 0 ≤ a → 0 ≤ b → 48 - a + b ≥ 0 := by
  sorry

end NUMINAMATH_GPT_current_number_of_people_l1043_104301


namespace NUMINAMATH_GPT_problem1_problem2_l1043_104392

def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x | x > 1 ∨ x < -6}

theorem problem1 (a : ℝ) : (setA a ∩ setB = ∅) → (-6 ≤ a ∧ a ≤ -2) := by
  intro h
  sorry

theorem problem2 (a : ℝ) : (setA a ∪ setB = setB) → (a < -9 ∨ a > 1) := by
  intro h
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1043_104392


namespace NUMINAMATH_GPT_sugar_solution_sweeter_l1043_104359

variables (a b m : ℝ)

theorem sugar_solution_sweeter (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : 
  (a / b < (a + m) / (b + m)) :=
sorry

end NUMINAMATH_GPT_sugar_solution_sweeter_l1043_104359


namespace NUMINAMATH_GPT_clara_cookies_l1043_104307

theorem clara_cookies (n : ℕ) :
  (15 * n - 1) % 11 = 0 → n = 3 := 
sorry

end NUMINAMATH_GPT_clara_cookies_l1043_104307


namespace NUMINAMATH_GPT_parabola_vertex_l1043_104371

-- Definition of the quadratic function representing the parabola
def parabola (x : ℝ) : ℝ := (3 * x - 1) ^ 2 + 2

-- Statement asserting the coordinates of the vertex of the given parabola
theorem parabola_vertex :
  ∃ h k : ℝ, ∀ x : ℝ, parabola x = 9 * (x - h) ^ 2 + k ∧ h = 1/3 ∧ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l1043_104371


namespace NUMINAMATH_GPT_dawson_failed_by_36_l1043_104318

-- Define the constants and conditions
def max_marks : ℕ := 220
def passing_percentage : ℝ := 0.3
def marks_obtained : ℕ := 30

-- Calculate the minimum passing marks
noncomputable def min_passing_marks : ℝ :=
  passing_percentage * max_marks

-- Calculate the marks Dawson failed by
noncomputable def marks_failed_by : ℝ :=
  min_passing_marks - marks_obtained

-- State the theorem
theorem dawson_failed_by_36 :
  marks_failed_by = 36 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_dawson_failed_by_36_l1043_104318


namespace NUMINAMATH_GPT_value_of_a3_l1043_104312

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

theorem value_of_a3 (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 + a 4 = 20) :
  a 2 = 4 :=
sorry

end NUMINAMATH_GPT_value_of_a3_l1043_104312


namespace NUMINAMATH_GPT_equilateral_triangle_l1043_104361

variable {a b c : ℝ}

-- Conditions
def condition1 (a b c : ℝ) : Prop :=
  (a + b + c) * (b + c - a) = 3 * b * c

def condition2 (a b c : ℝ) (cos_B cos_C : ℝ) : Prop :=
  c * cos_B = b * cos_C

-- Theorem statement
theorem equilateral_triangle (a b c : ℝ) (cos_B cos_C : ℝ)
  (h1 : condition1 a b c)
  (h2 : condition2 a b c cos_B cos_C) :
  a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_l1043_104361


namespace NUMINAMATH_GPT_line_repr_exists_same_line_iff_scalar_multiple_l1043_104353

-- Given that D is a line in 3D space, there exist a, b, c not all zero
theorem line_repr_exists
  (D : Set (ℝ × ℝ × ℝ)) :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ 
  (D = {p | ∃ (u v w : ℝ), p = (u, v, w) ∧ a * u + b * v + c * w = 0}) :=
sorry

-- Given two lines represented by different coefficients being the same
-- Prove that the coefficients are scalar multiples of each other
theorem same_line_iff_scalar_multiple
  (α1 β1 γ1 α2 β2 γ2 : ℝ) :
  (∀ (u v w : ℝ), α1 * u + β1 * v + γ1 * w = 0 ↔ α2 * u + β2 * v + γ2 * w = 0) ↔
  (∃ k : ℝ, k ≠ 0 ∧ α2 = k * α1 ∧ β2 = k * β1 ∧ γ2 = k * γ1) :=
sorry

end NUMINAMATH_GPT_line_repr_exists_same_line_iff_scalar_multiple_l1043_104353


namespace NUMINAMATH_GPT_solve_for_y_l1043_104395

theorem solve_for_y : ∀ (y : ℝ), 4 + 2.3 * y = 1.7 * y - 20 → y = -40 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1043_104395


namespace NUMINAMATH_GPT_reciprocal_of_2_l1043_104340

theorem reciprocal_of_2 : 1 / 2 = 1 / (2 : ℝ) := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_2_l1043_104340


namespace NUMINAMATH_GPT_Jennifer_has_24_dollars_left_l1043_104380

def remaining_money (initial amount: ℕ) (spent_sandwich spent_museum_ticket spent_book: ℕ) : ℕ :=
  initial - (spent_sandwich + spent_museum_ticket + spent_book)

theorem Jennifer_has_24_dollars_left :
  remaining_money 180 (1/5*180) (1/6*180) (1/2*180) = 24 :=
by
  sorry

end NUMINAMATH_GPT_Jennifer_has_24_dollars_left_l1043_104380


namespace NUMINAMATH_GPT_jake_sausages_cost_l1043_104321

theorem jake_sausages_cost :
  let package_weight := 2
  let num_packages := 3
  let cost_per_pound := 4
  let total_weight := package_weight * num_packages
  let total_cost := total_weight * cost_per_pound
  total_cost = 24 := by
  sorry

end NUMINAMATH_GPT_jake_sausages_cost_l1043_104321


namespace NUMINAMATH_GPT_product_of_solutions_of_t_squared_eq_49_l1043_104382

theorem product_of_solutions_of_t_squared_eq_49 :
  (∃ t₁ t₂ : ℝ, (t₁^2 = 49) ∧ (t₂^2 = 49) ∧ (t₁ ≠ t₂) ∧ (∀ t, t^2 = 49 → (t = t₁ ∨ t = t₂)) → t₁ * t₂ = -49) :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_of_t_squared_eq_49_l1043_104382


namespace NUMINAMATH_GPT_proof_a_eq_neg2x_or_3x_l1043_104316

theorem proof_a_eq_neg2x_or_3x (a b x : ℝ) (h1 : a - b = x) (h2 : a^3 - b^3 = 19 * x^3) (h3 : x ≠ 0) : 
  a = -2 * x ∨ a = 3 * x :=
  sorry

end NUMINAMATH_GPT_proof_a_eq_neg2x_or_3x_l1043_104316


namespace NUMINAMATH_GPT_solution_set_inequality_l1043_104398

variable {x : ℝ}
variable {a b : ℝ}

theorem solution_set_inequality (h₁ : ∀ x : ℝ, (ax^2 + bx - 1 > 0) ↔ (-1/2 < x ∧ x < -1/3)) :
  ∀ x : ℝ, (x^2 - bx - a ≥ 0) ↔ (x ≤ -3 ∨ x ≥ -2) := 
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1043_104398


namespace NUMINAMATH_GPT_no_pos_int_sol_l1043_104365

theorem no_pos_int_sol (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : ¬ ∃ (k : ℕ), (15 * a + b) * (a + 15 * b) = 3^k := 
sorry

end NUMINAMATH_GPT_no_pos_int_sol_l1043_104365


namespace NUMINAMATH_GPT_Carla_total_counts_l1043_104306

def Monday_counts := (60 * 2) + (120 * 2) + (10 * 2)
def Tuesday_counts := (60 * 3) + (120 * 2) + (10 * 1)
def Wednesday_counts := (80 * 4) + (24 * 5)
def Thursday_counts := (60 * 1) + (80 * 2) + (120 * 3) + (10 * 4) + (24 * 5)
def Friday_counts := (60 * 1) + (120 * 2) + (80 * 2) + (10 * 3) + (24 * 3)

def total_counts := Monday_counts + Tuesday_counts + Wednesday_counts + Thursday_counts + Friday_counts

theorem Carla_total_counts : total_counts = 2552 :=
by 
  sorry

end NUMINAMATH_GPT_Carla_total_counts_l1043_104306


namespace NUMINAMATH_GPT_perfect_square_trinomial_k_l1043_104338

theorem perfect_square_trinomial_k (k : ℤ) : (∃ a b : ℤ, (a*x + b)^2 = x^2 + k*x + 9) → (k = 6 ∨ k = -6) :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_k_l1043_104338


namespace NUMINAMATH_GPT_toothpicks_at_200th_stage_l1043_104326

-- Define initial number of toothpicks at stage 1
def a_1 : ℕ := 4

-- Define the function to compute the number of toothpicks at stage n, taking into account the changing common difference
def a (n : ℕ) : ℕ :=
  if n = 1 then 4
  else if n <= 49 then 4 + 4 * (n - 1)
  else if n <= 99 then 200 + 5 * (n - 50)
  else if n <= 149 then 445 + 6 * (n - 100)
  else if n <= 199 then 739 + 7 * (n - 150)
  else 0  -- This covers cases not considered in the problem for clarity

-- State the theorem to check the number of toothpicks at stage 200
theorem toothpicks_at_200th_stage : a 200 = 1082 :=
  sorry

end NUMINAMATH_GPT_toothpicks_at_200th_stage_l1043_104326


namespace NUMINAMATH_GPT_probability_fewer_heads_than_tails_is_793_over_2048_l1043_104310

noncomputable def probability_fewer_heads_than_tails (n : ℕ) : ℝ :=
(793 / 2048 : ℚ)

theorem probability_fewer_heads_than_tails_is_793_over_2048 :
  probability_fewer_heads_than_tails 12 = (793 / 2048 : ℚ) :=
sorry

end NUMINAMATH_GPT_probability_fewer_heads_than_tails_is_793_over_2048_l1043_104310


namespace NUMINAMATH_GPT_rectangle_perimeter_inscribed_l1043_104302

noncomputable def circle_area : ℝ := 32 * Real.pi
noncomputable def rectangle_area : ℝ := 34
noncomputable def rectangle_perimeter : ℝ := 28

theorem rectangle_perimeter_inscribed (area_circle : ℝ := 32 * Real.pi)
  (area_rectangle : ℝ := 34) : ∃ (P : ℝ), P = 28 :=
by
  use rectangle_perimeter
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_inscribed_l1043_104302


namespace NUMINAMATH_GPT_possible_to_fill_grid_l1043_104373

/-- Define the grid as a 2D array where each cell contains either 0 or 1. --/
def grid (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ), i < 5 → j < 5 → f i j = 0 ∨ f i j = 1

/-- Ensure the sum of every 2x2 subgrid is divisible by 3. --/
def divisible_by_3_in_subgrid (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < 4 → j < 4 → (f i j + f (i+1) j + f i (j+1) + f (i+1) (j+1)) % 3 = 0

/-- Ensure both 0 and 1 are present in the grid. --/
def contains_0_and_1 (f : ℕ → ℕ → ℕ) : Prop :=
  (∃ i j, i < 5 ∧ j < 5 ∧ f i j = 0) ∧ (∃ i j, i < 5 ∧ j < 5 ∧ f i j = 1)

/-- The main theorem stating the possibility of such a grid. --/
theorem possible_to_fill_grid :
  ∃ f, grid f ∧ divisible_by_3_in_subgrid f ∧ contains_0_and_1 f :=
sorry

end NUMINAMATH_GPT_possible_to_fill_grid_l1043_104373


namespace NUMINAMATH_GPT_exists_same_color_ratios_l1043_104393

-- Definition of coloring function.
def coloring : ℕ → Fin 2 := sorry

-- Definition of the problem: there exist A, B, C such that A : C = C : B,
-- and A, B, C are of same color.
theorem exists_same_color_ratios :
  ∃ A B C : ℕ, coloring A = coloring B ∧ coloring B = coloring C ∧ 
  (A : ℚ) / C = (C : ℚ) / B := 
sorry

end NUMINAMATH_GPT_exists_same_color_ratios_l1043_104393


namespace NUMINAMATH_GPT_find_c_of_triangle_area_l1043_104336

-- Define the problem in Lean 4 statement.
theorem find_c_of_triangle_area (A : ℝ) (b c : ℝ) (area : ℝ)
  (hA : A = 60 * Real.pi / 180)  -- Converting degrees to radians
  (hb : b = 1)
  (hArea : area = Real.sqrt 3) :
  c = 4 :=
by 
  -- Lean proof goes here (we include sorry to skip)
  sorry

end NUMINAMATH_GPT_find_c_of_triangle_area_l1043_104336


namespace NUMINAMATH_GPT_minimum_distance_between_extrema_is_2_sqrt_pi_l1043_104305

noncomputable def minimum_distance_adjacent_extrema (a : ℝ) (h : a > 0) : ℝ := 2 * Real.sqrt Real.pi

theorem minimum_distance_between_extrema_is_2_sqrt_pi (a : ℝ) (h : a > 0) :
  minimum_distance_adjacent_extrema a h = 2 * Real.sqrt Real.pi := 
sorry

end NUMINAMATH_GPT_minimum_distance_between_extrema_is_2_sqrt_pi_l1043_104305


namespace NUMINAMATH_GPT_root_condition_l1043_104304

-- Let f(x) = x^2 + ax + a^2 - a - 2
noncomputable def f (a x : ℝ) : ℝ := x^2 + a * x + a^2 - a - 2

theorem root_condition (a : ℝ) (h1 : ∀ ζ : ℝ, (ζ > 1 → ζ^2 + a * ζ + a^2 - a - 2 = 0) ∧ (ζ < 1 → ζ^2 + a * ζ + a^2 - a - 2 = 0)) :
  -1 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_root_condition_l1043_104304


namespace NUMINAMATH_GPT_transformed_system_solution_l1043_104343

theorem transformed_system_solution :
  (∀ (a b : ℝ), 2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9 → a = 8.3 ∧ b = 1.2) →
  (∀ (x y : ℝ), 2 * (x + 2) - 3 * (y - 1) = 13 ∧ 3 * (x + 2) + 5 * (y - 1) = 30.9 →
    x = 6.3 ∧ y = 2.2) :=
by
  intro h1
  intro x y
  intro hy
  sorry

end NUMINAMATH_GPT_transformed_system_solution_l1043_104343


namespace NUMINAMATH_GPT_minimum_value_expression_l1043_104387

-- Define the conditions for positive real numbers
variables (a b c : ℝ)
variable (h_a : 0 < a)
variable (h_b : 0 < b)
variable (h_c : 0 < c)

-- State the theorem to prove the minimum value of the expression
theorem minimum_value_expression (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : 
  (a / b) + (b / c) + (c / a) ≥ 3 := 
sorry

end NUMINAMATH_GPT_minimum_value_expression_l1043_104387


namespace NUMINAMATH_GPT_find_three_numbers_l1043_104357

theorem find_three_numbers (x y z : ℝ) 
  (h1 : x - y = 12) 
  (h2 : (x + y) / 4 = 7) 
  (h3 : z = 2 * y) 
  (h4 : x + z = 24) : 
  x = 20 ∧ y = 8 ∧ z = 16 := by
  sorry

end NUMINAMATH_GPT_find_three_numbers_l1043_104357


namespace NUMINAMATH_GPT_candy_pieces_total_l1043_104330

def number_of_packages_of_candy := 45
def pieces_per_package := 9

theorem candy_pieces_total : number_of_packages_of_candy * pieces_per_package = 405 :=
by
  sorry

end NUMINAMATH_GPT_candy_pieces_total_l1043_104330


namespace NUMINAMATH_GPT_omega_value_l1043_104346

theorem omega_value (ω : ℕ) (h : ω > 0) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = Real.sin (ω * x + Real.pi / 4)) 
  (h2 : ∀ x y, (Real.pi / 6 < x ∧ x < Real.pi / 3) → (Real.pi / 6 < y ∧ y < Real.pi / 3) → x < y → f y < f x) :
    ω = 2 ∨ ω = 3 := 
sorry

end NUMINAMATH_GPT_omega_value_l1043_104346


namespace NUMINAMATH_GPT_point_2000_coordinates_l1043_104317

-- Definition to describe the spiral numbering system in the first quadrant
def spiral_number (n : ℕ) : ℕ × ℕ := sorry

-- The task is to prove that the coordinates of the 2000th point are (44, 25).
theorem point_2000_coordinates : spiral_number 2000 = (44, 25) :=
by
  sorry

end NUMINAMATH_GPT_point_2000_coordinates_l1043_104317


namespace NUMINAMATH_GPT_polynomial_value_sum_l1043_104370

theorem polynomial_value_sum
  (a b c d : ℝ)
  (f : ℝ → ℝ)
  (Hf : ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d)
  (H1 : f 1 = 1) (H2 : f 2 = 2) (H3 : f 3 = 3) :
  f 0 + f 4 = 28 :=
sorry

end NUMINAMATH_GPT_polynomial_value_sum_l1043_104370


namespace NUMINAMATH_GPT_g_2002_eq_1_l1043_104367

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ := λ x => f x + 1 - x)

axiom f_one : f 1 = 1
axiom f_inequality_1 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_inequality_2 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

theorem g_2002_eq_1 : g 2002 = 1 := by
  sorry

end NUMINAMATH_GPT_g_2002_eq_1_l1043_104367


namespace NUMINAMATH_GPT_bekahs_reading_l1043_104383

def pages_per_day (total_pages read_pages days_left : ℕ) : ℕ :=
  (total_pages - read_pages) / days_left

theorem bekahs_reading :
  pages_per_day 408 113 5 = 59 := by
  sorry

end NUMINAMATH_GPT_bekahs_reading_l1043_104383


namespace NUMINAMATH_GPT_largest_integral_x_satisfies_ineq_largest_integral_x_is_5_l1043_104342

noncomputable def largest_integral_x_in_ineq (x : ℤ) : Prop :=
  (2 / 5 : ℚ) < (x / 7 : ℚ) ∧ (x / 7 : ℚ) < (8 / 11 : ℚ)

theorem largest_integral_x_satisfies_ineq : largest_integral_x_in_ineq 5 :=
sorry

theorem largest_integral_x_is_5 (x : ℤ) (h : largest_integral_x_in_ineq x) : x ≤ 5 :=
sorry

end NUMINAMATH_GPT_largest_integral_x_satisfies_ineq_largest_integral_x_is_5_l1043_104342


namespace NUMINAMATH_GPT_slower_speed_for_on_time_arrival_l1043_104313

variable (distance : ℝ) (actual_speed : ℝ) (time_early : ℝ)

theorem slower_speed_for_on_time_arrival 
(h1 : distance = 20)
(h2 : actual_speed = 40)
(h3 : time_early = 1 / 15) :
  actual_speed - (600 / 17) = 4.71 :=
by 
  sorry

end NUMINAMATH_GPT_slower_speed_for_on_time_arrival_l1043_104313


namespace NUMINAMATH_GPT_perpendicularity_condition_l1043_104396

theorem perpendicularity_condition 
  (A B C D E F k b : ℝ) 
  (h1 : b ≠ 0)
  (line : ∀ (x : ℝ), y = k * x + b)
  (curve : ∀ (x y : ℝ), A * x^2 + 2 * B * x * y + C * y^2 + 2 * D * x + 2 * E * y + F = 0):
  A * b^2 - 2 * D * k * b + F * k^2 + C * b^2 + 2 * E * b + F = 0 :=
sorry

end NUMINAMATH_GPT_perpendicularity_condition_l1043_104396
