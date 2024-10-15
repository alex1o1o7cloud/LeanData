import Mathlib

namespace NUMINAMATH_GPT_equivalent_single_discount_l765_76548

variable (original_price : ℝ)
variable (first_discount : ℝ)
variable (second_discount : ℝ)

-- Conditions
def sale_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)

def final_price (p : ℝ) (d1 d2 : ℝ) : ℝ :=
  let sale1 := sale_price p d1
  sale_price sale1 d2

-- Prove the equivalent single discount is as described
theorem equivalent_single_discount :
  original_price = 30 → first_discount = 0.2 → second_discount = 0.25 →
  (1 - final_price original_price first_discount second_discount / original_price) * 100 = 40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_equivalent_single_discount_l765_76548


namespace NUMINAMATH_GPT_problem_solve_l765_76502

theorem problem_solve (x y : ℝ) (h1 : x ≠ y) (h2 : x / y + (x + 6 * y) / (y + 6 * x) = 3) : 
    x / y = (8 + Real.sqrt 46) / 6 := 
  sorry

end NUMINAMATH_GPT_problem_solve_l765_76502


namespace NUMINAMATH_GPT_find_b_l765_76576

theorem find_b
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) -> True)
  (h4 : ∃ e, e = (Real.sqrt 5) / 3)
  (h5 : 2 * a = 12) :
  b = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l765_76576


namespace NUMINAMATH_GPT_henry_kombucha_bottles_l765_76522

theorem henry_kombucha_bottles :
  ∀ (monthly_bottles: ℕ) (cost_per_bottle refund_rate: ℝ) (months_in_year total_bottles_in_year: ℕ),
  (monthly_bottles = 15) →
  (cost_per_bottle = 3.0) →
  (refund_rate = 0.10) →
  (months_in_year = 12) →
  (total_bottles_in_year = monthly_bottles * months_in_year) →
  (total_refund = refund_rate * total_bottles_in_year) →
  (bottles_bought_with_refund = total_refund / cost_per_bottle) →
  bottles_bought_with_refund = 6 :=
by
  intros monthly_bottles cost_per_bottle refund_rate months_in_year total_bottles_in_year
  sorry

end NUMINAMATH_GPT_henry_kombucha_bottles_l765_76522


namespace NUMINAMATH_GPT_value_of_k_l765_76560

theorem value_of_k :
  ∀ (k : ℝ), (∃ m : ℝ, m = 4/5 ∧ (21 - (-5)) / (k - 3) = m) →
  k = 35.5 :=
by
  intros k hk
  -- Here hk is the proof that the line through (3, -5) and (k, 21) has the same slope as 4/5
  sorry

end NUMINAMATH_GPT_value_of_k_l765_76560


namespace NUMINAMATH_GPT_measure_four_liters_impossible_l765_76558

theorem measure_four_liters_impossible (a b c : ℕ) (h1 : a = 12) (h2 : b = 9) (h3 : c = 4) :
  ¬ ∃ x y : ℕ, x * a + y * b = c := 
by
  sorry

end NUMINAMATH_GPT_measure_four_liters_impossible_l765_76558


namespace NUMINAMATH_GPT_find_d_l765_76596

theorem find_d
  (a b c d : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (h_d_pos : d > 0)
  (h_max : a * 1 + d = 5)
  (h_min : a * (-1) + d = -3) :
  d = 1 := 
sorry

end NUMINAMATH_GPT_find_d_l765_76596


namespace NUMINAMATH_GPT_arithmetic_mean_of_fractions_l765_76537

theorem arithmetic_mean_of_fractions :
  let a := 7 / 9
  let b := 5 / 6
  let c := 8 / 9
  2 * b = a + c :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_fractions_l765_76537


namespace NUMINAMATH_GPT_total_shaded_area_is_2pi_l765_76584

theorem total_shaded_area_is_2pi (sm_radius large_radius : ℝ) 
  (h_sm_radius : sm_radius = 1) 
  (h_large_radius : large_radius = 2) 
  (sm_circle_area large_circle_area total_shaded_area : ℝ) 
  (h_sm_circle_area : sm_circle_area = π * sm_radius^2) 
  (h_large_circle_area : large_circle_area = π * large_radius^2) 
  (h_total_shaded_area : total_shaded_area = large_circle_area - 2 * sm_circle_area) :
  total_shaded_area = 2 * π :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_shaded_area_is_2pi_l765_76584


namespace NUMINAMATH_GPT_bug_total_distance_l765_76598

theorem bug_total_distance : 
  let start := 3
  let first_point := 9
  let second_point := -4
  let distance_1 := abs (first_point - start)
  let distance_2 := abs (second_point - first_point)
  distance_1 + distance_2 = 19 := 
by
  sorry

end NUMINAMATH_GPT_bug_total_distance_l765_76598


namespace NUMINAMATH_GPT_f_2015_l765_76518

def f : ℤ → ℤ := sorry

axiom f1 : f 1 = 1
axiom f2 : f 2 = 0
axiom functional_eq (x y : ℤ) : f (x + y) = f x * f (1 - y) + f (1 - x) * f y

theorem f_2015 : f 2015 = 1 ∨ f 2015 = -1 :=
sorry

end NUMINAMATH_GPT_f_2015_l765_76518


namespace NUMINAMATH_GPT_john_climbs_9_flights_l765_76540

variable (fl : Real := 10)  -- Each flight of stairs is 10 feet
variable (step_height_inches : Real := 18)  -- Each step is 18 inches
variable (steps : Nat := 60)  -- John climbs 60 steps

theorem john_climbs_9_flights :
  (steps * (step_height_inches / 12) / fl = 9) :=
by
  sorry

end NUMINAMATH_GPT_john_climbs_9_flights_l765_76540


namespace NUMINAMATH_GPT_oil_used_l765_76545

theorem oil_used (total_weight : ℕ) (ratio_oil_peanuts : ℕ) (ratio_total_parts : ℕ) 
  (ratio_peanuts : ℕ) (ratio_parts : ℕ) (peanuts_weight : ℕ) : 
  ratio_oil_peanuts = 2 → 
  ratio_peanuts = 8 → 
  ratio_total_parts = 10 → 
  ratio_parts = 20 →
  peanuts_weight = total_weight / ratio_total_parts →
  total_weight = 20 → 
  2 * peanuts_weight = 4 :=
by sorry

end NUMINAMATH_GPT_oil_used_l765_76545


namespace NUMINAMATH_GPT_square_perimeter_l765_76572

variable (side : ℕ) (P : ℕ)

theorem square_perimeter (h : side = 19) : P = 4 * side → P = 76 := by
  intro hp
  rw [h] at hp
  norm_num at hp
  exact hp

end NUMINAMATH_GPT_square_perimeter_l765_76572


namespace NUMINAMATH_GPT_problem_solution_l765_76556

noncomputable def vector_magnitudes_and_angle 
  (a b : ℝ) (angle_ab : ℝ) (norma normb : ℝ) (k : ℝ) : Prop :=
(a = 4 ∧ b = 8 ∧ angle_ab = 2 * Real.pi / 3 ∧ norma = 4 ∧ normb = 8) →
((norma^2 + normb^2 + 2 * norma * normb * Real.cos angle_ab = 48) ∧
  (16 * k - 32 * k + 16 - 128 = 0))

theorem problem_solution : vector_magnitudes_and_angle 4 8 (2 * Real.pi / 3) 4 8 (-7) := 
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l765_76556


namespace NUMINAMATH_GPT_number_of_classes_l765_76575

theorem number_of_classes (max_val : ℕ) (min_val : ℕ) (class_interval : ℕ) (range : ℕ) (num_classes : ℕ) :
  max_val = 169 → min_val = 143 → class_interval = 3 → range = max_val - min_val → num_classes = (range + 2) / class_interval + 1 :=
sorry

end NUMINAMATH_GPT_number_of_classes_l765_76575


namespace NUMINAMATH_GPT_committee_count_l765_76536

noncomputable def num_acceptable_committees (total_people : ℕ) (committee_size : ℕ) (conditions : List (Set ℕ)) : ℕ := sorry

theorem committee_count :
  num_acceptable_committees 9 5 [ {1, 2}, {3, 4} ] = 41 := sorry

end NUMINAMATH_GPT_committee_count_l765_76536


namespace NUMINAMATH_GPT_difference_of_squares_l765_76530

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : |x^2 - y^2| = 108 :=
  sorry

end NUMINAMATH_GPT_difference_of_squares_l765_76530


namespace NUMINAMATH_GPT_problem1_l765_76514

theorem problem1 : 
  ∀ a b : ℤ, a = 1 → b = -3 → (a - b)^2 - 2 * a * (a + 3 * b) + (a + 2 * b) * (a - 2 * b) = -3 :=
by
  intros a b h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_problem1_l765_76514


namespace NUMINAMATH_GPT_john_bought_two_dozens_l765_76500

theorem john_bought_two_dozens (x : ℕ) (h₁ : 21 + 3 = x * 12) : x = 2 :=
by {
    -- Placeholder for skipping the proof since it's not required.
    sorry
}

end NUMINAMATH_GPT_john_bought_two_dozens_l765_76500


namespace NUMINAMATH_GPT_product_of_two_two_digit_numbers_greater_than_40_is_four_digit_l765_76523

-- Define the condition: both numbers are two-digit numbers greater than 40
def is_two_digit_and_greater_than_40 (n : ℕ) : Prop :=
  40 < n ∧ n < 100

-- Define the problem statement
theorem product_of_two_two_digit_numbers_greater_than_40_is_four_digit
  (a b : ℕ) (ha : is_two_digit_and_greater_than_40 a) (hb : is_two_digit_and_greater_than_40 b) :
  1000 ≤ a * b ∧ a * b < 10000 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_two_digit_numbers_greater_than_40_is_four_digit_l765_76523


namespace NUMINAMATH_GPT_spaceship_distance_l765_76524

-- Define the distance variables and conditions
variables (D : ℝ) -- Distance from Earth to Planet X
variable (T : ℝ) -- Total distance traveled by the spaceship

-- Conditions
variables (hx : T = 0.7) -- Total distance traveled is 0.7 light-years
variables (hy : D + 0.1 + 0.1 = T) -- Sum of distances along the path

-- Theorem statement to prove the distance from Earth to Planet X
theorem spaceship_distance (h1 : T = 0.7) (h2 : D + 0.1 + 0.1 = T) : D = 0.5 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_spaceship_distance_l765_76524


namespace NUMINAMATH_GPT_sin_theta_fourth_quadrant_l765_76567

-- Given conditions
variables {θ : ℝ} (h1 : Real.cos θ = 1 / 3) (h2 : 3 * pi / 2 < θ ∧ θ < 2 * pi)

-- Proof statement
theorem sin_theta_fourth_quadrant : Real.sin θ = -2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_GPT_sin_theta_fourth_quadrant_l765_76567


namespace NUMINAMATH_GPT_relationship_between_b_and_c_l765_76508

-- Definitions based on the given conditions
def y1 (x a b : ℝ) : ℝ := (x + 2 * a) * (x - 2 * b)
def y2 (x b : ℝ) : ℝ := -x + 2 * b
def y (x a b : ℝ) : ℝ := y1 x a b + y2 x b

-- Lean theorem for the proof problem
theorem relationship_between_b_and_c
  (a b c : ℝ)
  (h : a + 2 = b)
  (h_y : y c a b = 0) :
  c = 5 - 2 * b ∨ c = 2 * b :=
by
  -- The proof will go here, currently omitted
  sorry

end NUMINAMATH_GPT_relationship_between_b_and_c_l765_76508


namespace NUMINAMATH_GPT_intersection_is_correct_l765_76547

noncomputable def A : Set ℝ := { x : ℝ | x^2 - x - 2 ≤ 0 }
noncomputable def B : Set ℝ := { x : ℝ | x < 1 }

theorem intersection_is_correct : (A ∩ B) = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l765_76547


namespace NUMINAMATH_GPT_total_cost_chairs_l765_76566

def living_room_chairs : Nat := 3
def kitchen_chairs : Nat := 6
def dining_room_chairs : Nat := 8
def outdoor_patio_chairs : Nat := 12

def living_room_price : Nat := 75
def kitchen_price : Nat := 50
def dining_room_price : Nat := 100
def outdoor_patio_price : Nat := 60

theorem total_cost_chairs : 
  living_room_chairs * living_room_price + 
  kitchen_chairs * kitchen_price + 
  dining_room_chairs * dining_room_price + 
  outdoor_patio_chairs * outdoor_patio_price = 2045 := by
  sorry

end NUMINAMATH_GPT_total_cost_chairs_l765_76566


namespace NUMINAMATH_GPT_point_on_parabola_l765_76525

def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem point_on_parabola : parabola (1/2) = 0 := 
by sorry

end NUMINAMATH_GPT_point_on_parabola_l765_76525


namespace NUMINAMATH_GPT_sum_of_reciprocals_l765_76543

noncomputable def roots (p q r : ℂ) : Prop := 
  p ^ 3 - p + 1 = 0 ∧ q ^ 3 - q + 1 = 0 ∧ r ^ 3 - r + 1 = 0

theorem sum_of_reciprocals (p q r : ℂ) (h : roots p q r) : 
  (1 / (p + 2)) + (1 / (q + 2)) + (1 / (r + 2)) = - (10 / 13) := by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l765_76543


namespace NUMINAMATH_GPT_no_intersection_points_l765_76585

theorem no_intersection_points :
  ¬ ∃ x y : ℝ, y = |3 * x + 4| ∧ y = -|2 * x + 1| :=
by
  sorry

end NUMINAMATH_GPT_no_intersection_points_l765_76585


namespace NUMINAMATH_GPT_markers_per_box_l765_76594

theorem markers_per_box (original_markers new_boxes total_markers : ℕ) 
    (h1 : original_markers = 32) (h2 : new_boxes = 6) (h3 : total_markers = 86) : 
    total_markers - original_markers = new_boxes * 9 :=
by sorry

end NUMINAMATH_GPT_markers_per_box_l765_76594


namespace NUMINAMATH_GPT_relationship_and_range_max_profit_find_a_l765_76565

noncomputable def functional_relationship (x : ℝ) : ℝ :=
if 40 ≤ x ∧ x ≤ 50 then 5
else if 50 < x ∧ x ≤ 100 then 10 - 0.1 * x
else 0  -- default case to handle x out of range, though ideally this should not occur in the context.

theorem relationship_and_range : 
  ∀ (x : ℝ), (40 ≤ x ∧ x ≤ 100) →
    (functional_relationship x = 
    (if 40 ≤ x ∧ x ≤ 50 then 5 else if 50 < x ∧ x ≤ 100 then 10 - 0.1 * x else 0)) :=
sorry

noncomputable def monthly_profit (x : ℝ) : ℝ :=
(x - 40) * functional_relationship x

theorem max_profit : 
  (∀ x, 40 ≤ x ∧ x ≤ 100 → monthly_profit x ≤ 90) ∧
  (monthly_profit 70 = 90) :=
sorry

noncomputable def donation_profit (x a : ℝ) : ℝ :=
(x - 40 - a) * (10 - 0.1 * x)

theorem find_a (a : ℝ) : 
  (∀ x, x ≤ 70 → donation_profit x a ≤ 78) ∧
  (donation_profit 70 a = 78) → 
  a = 4 :=
sorry

end NUMINAMATH_GPT_relationship_and_range_max_profit_find_a_l765_76565


namespace NUMINAMATH_GPT_highlighter_count_l765_76593

-- Define the quantities of highlighters.
def pinkHighlighters := 3
def yellowHighlighters := 7
def blueHighlighters := 5

-- Define the total number of highlighters.
def totalHighlighters := pinkHighlighters + yellowHighlighters + blueHighlighters

-- The theorem states that the total number of highlighters is 15.
theorem highlighter_count : totalHighlighters = 15 := by
  -- Proof skipped for now.
  sorry

end NUMINAMATH_GPT_highlighter_count_l765_76593


namespace NUMINAMATH_GPT_terminating_decimal_count_l765_76599

theorem terminating_decimal_count : ∃ n, n = 23 ∧ (∀ k, 1 ≤ k ∧ k ≤ 499 → (∃ m, k = 21 * m)) :=
by
  sorry

end NUMINAMATH_GPT_terminating_decimal_count_l765_76599


namespace NUMINAMATH_GPT_find_y_l765_76561

theorem find_y :
  ∃ (x y : ℤ), (x - 5) / 7 = 7 ∧ (x - y) / 10 = 3 ∧ y = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l765_76561


namespace NUMINAMATH_GPT_smallest_period_of_f_center_of_symmetry_of_f_range_of_f_on_interval_l765_76552

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * (Real.sin x)^2 - (1 + 2)

theorem smallest_period_of_f : ∀ x : ℝ, f (x + π) = f x := 
by sorry

theorem center_of_symmetry_of_f : ∀ k : ℤ, ∃ c : ℝ, ∀ x : ℝ, f (c - x) = f (c + x) := 
by sorry

theorem range_of_f_on_interval : 
  ∃ a b, (∀ x ∈ Set.Icc (-π / 4) (π / 4), f x ∈ Set.Icc a b) ∧ 
          (∀ y, y ∈ Set.Icc 3 5 → ∃ x ∈ Set.Icc (-π / 4) (π / 4), y = f x) := 
by sorry

end NUMINAMATH_GPT_smallest_period_of_f_center_of_symmetry_of_f_range_of_f_on_interval_l765_76552


namespace NUMINAMATH_GPT_car_speed_is_120_l765_76591

theorem car_speed_is_120 (v t : ℝ) (h1 : v > 0) (h2 : t > 0) (h3 : v * t = 75)
  (h4 : 1.5 * v * (t - (12.5 / 60)) = 75) : v = 120 := by
  sorry

end NUMINAMATH_GPT_car_speed_is_120_l765_76591


namespace NUMINAMATH_GPT_remainder_sum_of_first_eight_primes_div_tenth_prime_l765_76533

theorem remainder_sum_of_first_eight_primes_div_tenth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) % 29 = 19 :=
by norm_num

end NUMINAMATH_GPT_remainder_sum_of_first_eight_primes_div_tenth_prime_l765_76533


namespace NUMINAMATH_GPT_square_of_binomial_l765_76586

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b) ^ 2 = 9 * x^2 - 18 * x + a) ↔ a = 9 :=
by
  sorry

end NUMINAMATH_GPT_square_of_binomial_l765_76586


namespace NUMINAMATH_GPT_irrational_pi_l765_76578

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem irrational_pi : is_irrational π := by
  sorry

end NUMINAMATH_GPT_irrational_pi_l765_76578


namespace NUMINAMATH_GPT_largest_divisor_of_product_of_five_consecutive_integers_l765_76592

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_product_of_five_consecutive_integers_l765_76592


namespace NUMINAMATH_GPT_madeline_water_intake_l765_76587

def water_bottle_capacity : ℕ := 12
def number_of_refills : ℕ := 7
def additional_water_needed : ℕ := 16
def total_water_needed : ℕ := 100

theorem madeline_water_intake : water_bottle_capacity * number_of_refills + additional_water_needed = total_water_needed :=
by
  sorry

end NUMINAMATH_GPT_madeline_water_intake_l765_76587


namespace NUMINAMATH_GPT_find_fraction_of_ab_l765_76532

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def x := a / b

theorem find_fraction_of_ab (h1 : a ≠ b) (h2 : a / b + (3 * a + 4 * b) / (b + 12 * a) = 2) :
  a / b = (5 - Real.sqrt 19) / 6 :=
sorry

end NUMINAMATH_GPT_find_fraction_of_ab_l765_76532


namespace NUMINAMATH_GPT_fill_tank_time_l765_76571

theorem fill_tank_time 
  (tank_capacity : ℕ) (initial_fill : ℕ) (fill_rate : ℝ) 
  (drain_rate1 : ℝ) (drain_rate2 : ℝ) : 
  tank_capacity = 8000 ∧ initial_fill = 4000 ∧ fill_rate = 0.5 ∧ drain_rate1 = 0.25 ∧ drain_rate2 = 0.1667 
  → (initial_fill + fill_rate * t - (drain_rate1 + drain_rate2) * t) = tank_capacity → t = 48 := sorry

end NUMINAMATH_GPT_fill_tank_time_l765_76571


namespace NUMINAMATH_GPT_find_second_cert_interest_rate_l765_76595

theorem find_second_cert_interest_rate
  (initial_investment : ℝ := 12000)
  (first_term_months : ℕ := 8)
  (first_interest_rate : ℝ := 8 / 100)
  (second_term_months : ℕ := 10)
  (final_amount : ℝ := 13058.40)
  : ∃ s : ℝ, (s = 3.984) := sorry

end NUMINAMATH_GPT_find_second_cert_interest_rate_l765_76595


namespace NUMINAMATH_GPT_combined_length_in_scientific_notation_l765_76563

noncomputable def yards_to_inches (yards : ℝ) : ℝ := yards * 36
noncomputable def inches_to_cm (inches : ℝ) : ℝ := inches * 2.54
noncomputable def feet_to_inches (feet : ℝ) : ℝ := feet * 12

def sports_stadium_length_yards : ℝ := 61
def safety_margin_feet : ℝ := 2
def safety_margin_inches : ℝ := 9

theorem combined_length_in_scientific_notation :
  (inches_to_cm (yards_to_inches sports_stadium_length_yards) +
   (inches_to_cm (feet_to_inches safety_margin_feet + safety_margin_inches)) * 2) = 5.74268 * 10^3 :=
by
  sorry

end NUMINAMATH_GPT_combined_length_in_scientific_notation_l765_76563


namespace NUMINAMATH_GPT_tan_45_deg_eq_one_l765_76503

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_45_deg_eq_one_l765_76503


namespace NUMINAMATH_GPT_sequence_value_l765_76513

theorem sequence_value (a : ℕ → ℤ) (h1 : ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q)
                       (h2 : a 2 = -6) : a 10 = -30 :=
by
  sorry

end NUMINAMATH_GPT_sequence_value_l765_76513


namespace NUMINAMATH_GPT_combined_salaries_ABC_E_l765_76573

-- Definitions for the conditions
def salary_D : ℝ := 7000
def avg_salary_ABCDE : ℝ := 8200

-- Defining the combined salary proof
theorem combined_salaries_ABC_E : (A B C E : ℝ) → 
  (A + B + C + D + E = 5 * avg_salary_ABCDE ∧ D = salary_D) → 
  (A + B + C + E = 34000) := 
sorry

end NUMINAMATH_GPT_combined_salaries_ABC_E_l765_76573


namespace NUMINAMATH_GPT_complex_number_solution_l765_76538

open Complex

theorem complex_number_solution (z : ℂ) (h : (z - 2 * I) / z = 2 + I) :
  im z = -1 ∧ abs z = Real.sqrt 2 ∧ z ^ 6 = -8 * I :=
by
  sorry

end NUMINAMATH_GPT_complex_number_solution_l765_76538


namespace NUMINAMATH_GPT_mushroom_distribution_l765_76526

-- Define the total number of mushrooms
def total_mushrooms : ℕ := 120

-- Define the number of girls
def number_of_girls : ℕ := 5

-- Auxiliary function to represent each girl receiving pattern
def mushrooms_received (n :ℕ) (total : ℕ) : ℝ :=
  (n + 20) + 0.04 * (total - (n + 20))

-- Define the equality function to check distribution condition
def equal_distribution (girls : ℕ) (total : ℕ) : Prop :=
  ∀ i j : ℕ, i < girls → j < girls → mushrooms_received i total = mushrooms_received j total

-- Main proof statement about the total mushrooms and number of girls following the distribution
theorem mushroom_distribution :
  total_mushrooms = 120 ∧ number_of_girls = 5 ∧ equal_distribution number_of_girls total_mushrooms := 
by 
  sorry

end NUMINAMATH_GPT_mushroom_distribution_l765_76526


namespace NUMINAMATH_GPT_problem_solution_l765_76597

variables (a b : ℝ)
variables (h1 : a > 0) (h2 : b > 0)
variables (h3 : 3 * log 101 ((1030301 - a - b) / (3 * a * b)) = 3 - 2 * log 101 (a * b))

theorem problem_solution : 101 - (a)^(1/3) - (b)^(1/3) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l765_76597


namespace NUMINAMATH_GPT_abby_correct_percentage_l765_76515

-- Defining the scores and number of problems for each test
def score_test1 := 85 / 100
def score_test2 := 75 / 100
def score_test3 := 60 / 100
def score_test4 := 90 / 100

def problems_test1 := 30
def problems_test2 := 50
def problems_test3 := 20
def problems_test4 := 40

-- Define the total number of problems
def total_problems := problems_test1 + problems_test2 + problems_test3 + problems_test4

-- Calculate the number of problems Abby answered correctly on each test
def correct_problems_test1 := score_test1 * problems_test1
def correct_problems_test2 := score_test2 * problems_test2
def correct_problems_test3 := score_test3 * problems_test3
def correct_problems_test4 := score_test4 * problems_test4

-- Calculate the total number of correctly answered problems
def total_correct_problems := correct_problems_test1 + correct_problems_test2 + correct_problems_test3 + correct_problems_test4

-- Calculate the overall percentage of problems answered correctly
def overall_percentage_correct := (total_correct_problems / total_problems) * 100

-- The theorem to be proved
theorem abby_correct_percentage : overall_percentage_correct = 80 := by
  -- Skipping the actual proof
  sorry

end NUMINAMATH_GPT_abby_correct_percentage_l765_76515


namespace NUMINAMATH_GPT_percentage_increase_Sakshi_Tanya_l765_76546

def efficiency_Sakshi : ℚ := 1 / 5
def efficiency_Tanya : ℚ := 1 / 4
def percentage_increase_in_efficiency (eff_Sakshi eff_Tanya : ℚ) : ℚ :=
  ((eff_Tanya - eff_Sakshi) / eff_Sakshi) * 100

theorem percentage_increase_Sakshi_Tanya :
  percentage_increase_in_efficiency efficiency_Sakshi efficiency_Tanya = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_Sakshi_Tanya_l765_76546


namespace NUMINAMATH_GPT_initial_children_count_l765_76529

theorem initial_children_count (passed retake : ℝ) (h_passed : passed = 105.0) (h_retake : retake = 593) : 
    passed + retake = 698 := 
by
  sorry

end NUMINAMATH_GPT_initial_children_count_l765_76529


namespace NUMINAMATH_GPT_tank_capacity_is_780_l765_76569

noncomputable def tank_capacity : ℕ := 
  let fill_rate_A := 40
  let fill_rate_B := 30
  let drain_rate_C := 20
  let cycle_minutes := 3
  let total_minutes := 48
  let net_fill_per_cycle := fill_rate_A + fill_rate_B - drain_rate_C
  let total_cycles := total_minutes / cycle_minutes
  let total_fill := total_cycles * net_fill_per_cycle
  let final_capacity := total_fill - drain_rate_C -- Adjust for the last minute where C opens
  final_capacity

theorem tank_capacity_is_780 : tank_capacity = 780 := by
  unfold tank_capacity
  -- Proof steps to be filled in
  sorry

end NUMINAMATH_GPT_tank_capacity_is_780_l765_76569


namespace NUMINAMATH_GPT_garden_perimeter_equals_104_l765_76505

theorem garden_perimeter_equals_104 :
  let playground_length := 16
  let playground_width := 12
  let playground_area := playground_length * playground_width
  let garden_width := 4
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * garden_length + 2 * garden_width
  playground_area = 192 ∧ garden_perimeter = 104 :=
by {
  -- Declarations
  let playground_length := 16
  let playground_width := 12
  let playground_area := playground_length * playground_width
  let garden_width := 4
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * garden_length + 2 * garden_width

  -- Assertions
  have area_playground : playground_area = 192 := by sorry
  have perimeter_garden : garden_perimeter = 104 := by sorry

  -- Conclusion
  exact ⟨area_playground, perimeter_garden⟩
}

end NUMINAMATH_GPT_garden_perimeter_equals_104_l765_76505


namespace NUMINAMATH_GPT_positive_difference_of_squares_l765_76510

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 18) : a^2 - b^2 = 1080 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_squares_l765_76510


namespace NUMINAMATH_GPT_number_of_episodes_last_season_more_than_others_l765_76588

-- Definitions based on conditions
def episodes_per_other_season : ℕ := 22
def initial_seasons : ℕ := 9
def duration_per_episode : ℚ := 0.5
def total_hours_after_last_season : ℚ := 112

-- Derived definitions based on conditions (not solution steps)
def total_hours_first_9_seasons := initial_seasons * episodes_per_other_season * duration_per_episode
def additional_hours_last_season := total_hours_after_last_season - total_hours_first_9_seasons
def episodes_last_season := additional_hours_last_season / duration_per_episode

-- Proof problem statement
theorem number_of_episodes_last_season_more_than_others : 
  episodes_last_season = episodes_per_other_season + 4 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_episodes_last_season_more_than_others_l765_76588


namespace NUMINAMATH_GPT_peak_valley_usage_l765_76564

-- Define the electricity rate constants
def normal_rate : ℝ := 0.5380
def peak_rate : ℝ := 0.5680
def valley_rate : ℝ := 0.2880

-- Define the total consumption and the savings
def total_consumption : ℝ := 200
def savings : ℝ := 16.4

-- Define the theorem to prove the peak and off-peak usage
theorem peak_valley_usage :
  ∃ (x y : ℝ), x + y = total_consumption ∧ peak_rate * x + valley_rate * y = total_consumption * normal_rate - savings ∧ x = 120 ∧ y = 80 :=
by
  sorry

end NUMINAMATH_GPT_peak_valley_usage_l765_76564


namespace NUMINAMATH_GPT_parallel_lines_l765_76504

open Real -- Open the real number namespace

/-- Definition of line l1 --/
def line_l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y - 1 = 0

/-- Definition of line l2 --/
def line_l2 (a : ℝ) (x y : ℝ) := x + (a + 1) * y + 4 = 0

/-- The proof statement --/
theorem parallel_lines (a : ℝ) : (a = 1) → (line_l1 a x y) → (line_l2 a x y) := 
sorry

end NUMINAMATH_GPT_parallel_lines_l765_76504


namespace NUMINAMATH_GPT_converse_inverse_contrapositive_l765_76554

-- The original statement
def original_statement (x y : ℕ) : Prop :=
  (x + y = 5) → (x = 3 ∧ y = 2)

-- Converse of the original statement
theorem converse (x y : ℕ) : (x = 3 ∧ y = 2) → (x + y = 5) :=
by
  sorry

-- Inverse of the original statement
theorem inverse (x y : ℕ) : (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2) :=
by
  sorry

-- Contrapositive of the original statement
theorem contrapositive (x y : ℕ) : (x ≠ 3 ∨ y ≠ 2) → (x + y ≠ 5) :=
by
  sorry

end NUMINAMATH_GPT_converse_inverse_contrapositive_l765_76554


namespace NUMINAMATH_GPT_integer_solution_count_l765_76528

theorem integer_solution_count : 
  let condition := ∀ x : ℤ, (x - 2) ^ 2 ≤ 4
  ∃ count : ℕ, count = 5 ∧ (∀ x : ℤ, condition → (0 ≤ x ∧ x ≤ 4)) := 
sorry

end NUMINAMATH_GPT_integer_solution_count_l765_76528


namespace NUMINAMATH_GPT_percentage_decrease_increase_l765_76521

theorem percentage_decrease_increase (S : ℝ) (x : ℝ) (h : S > 0) :
  S * (1 - x / 100) * (1 + x / 100) = S * (64 / 100) → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_increase_l765_76521


namespace NUMINAMATH_GPT_remaining_players_average_points_l765_76577

-- Define the conditions
def total_points : ℕ := 270
def total_players : ℕ := 9
def players_averaged_50 : ℕ := 5
def average_points_50 : ℕ := 50

-- Define the query
theorem remaining_players_average_points :
  (total_points - players_averaged_50 * average_points_50) / (total_players - players_averaged_50) = 5 :=
by
  sorry

end NUMINAMATH_GPT_remaining_players_average_points_l765_76577


namespace NUMINAMATH_GPT_cone_volume_is_correct_l765_76519

theorem cone_volume_is_correct (r l h : ℝ) 
  (h1 : 2 * r = Real.sqrt 2 * l)
  (h2 : π * r * l = 16 * Real.sqrt 2 * π)
  (h3 : h = r) : 
  (1 / 3) * π * r ^ 2 * h = (64 / 3) * π :=
by sorry

end NUMINAMATH_GPT_cone_volume_is_correct_l765_76519


namespace NUMINAMATH_GPT_determinant_of_2x2_matrix_l765_76517

theorem determinant_of_2x2_matrix :
  let a := 2
  let b := 4
  let c := 1
  let d := 3
  a * d - b * c = 2 := by
  sorry

end NUMINAMATH_GPT_determinant_of_2x2_matrix_l765_76517


namespace NUMINAMATH_GPT_largest_number_l765_76568

noncomputable def a : ℝ := 8.12331
noncomputable def b : ℝ := 8.123 + 3 / 10000 * ∑' n, 1 / (10 : ℝ)^n
noncomputable def c : ℝ := 8.12 + 331 / 100000 * ∑' n, 1 / (1000 : ℝ)^n
noncomputable def d : ℝ := 8.1 + 2331 / 1000000 * ∑' n, 1 / (10000 : ℝ)^n
noncomputable def e : ℝ := 8 + 12331 / 100000 * ∑' n, 1 / (10000 : ℝ)^n

theorem largest_number : (b > a) ∧ (b > c) ∧ (b > d) ∧ (b > e) := by
  sorry

end NUMINAMATH_GPT_largest_number_l765_76568


namespace NUMINAMATH_GPT_chocolate_bar_cost_l765_76549

variable (cost_per_bar num_bars : ℝ)

theorem chocolate_bar_cost (num_scouts smores_per_scout smores_per_bar : ℕ) (total_cost : ℝ)
  (h1 : num_scouts = 15)
  (h2 : smores_per_scout = 2)
  (h3 : smores_per_bar = 3)
  (h4 : total_cost = 15)
  (h5 : num_bars = (num_scouts * smores_per_scout) / smores_per_bar)
  (h6 : total_cost = cost_per_bar * num_bars) :
  cost_per_bar = 1.50 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bar_cost_l765_76549


namespace NUMINAMATH_GPT_find_remainder_mod_10_l765_76541

def inv_mod_10 (x : ℕ) : ℕ := 
  if x = 1 then 1 
  else if x = 3 then 7 
  else if x = 7 then 3 
  else if x = 9 then 9 
  else 0 -- invalid, not invertible

theorem find_remainder_mod_10 (a b c d : ℕ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ d) (hd : d ≠ a) 
  (ha' : a < 10) (hb' : b < 10) (hc' : c < 10) (hd' : d < 10)
  (ha_inv : inv_mod_10 a ≠ 0) (hb_inv : inv_mod_10 b ≠ 0)
  (hc_inv : inv_mod_10 c ≠ 0) (hd_inv : inv_mod_10 d ≠ 0) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (inv_mod_10 (a * b * c * d % 10))) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_remainder_mod_10_l765_76541


namespace NUMINAMATH_GPT_savings_percentage_correct_l765_76520

-- Definitions based on conditions
def food_per_week : ℕ := 100
def num_weeks : ℕ := 4
def rent : ℕ := 1500
def video_streaming : ℕ := 30
def cell_phone : ℕ := 50
def savings : ℕ := 198

-- Total spending calculations based on the conditions
def food_total : ℕ := food_per_week * num_weeks
def total_spending : ℕ := food_total + rent + video_streaming + cell_phone

-- Calculation of the percentage
def savings_percentage (savings total_spending : ℕ) : ℕ :=
  (savings * 100) / total_spending

-- The statement to prove
theorem savings_percentage_correct : savings_percentage savings total_spending = 10 := by
  sorry

end NUMINAMATH_GPT_savings_percentage_correct_l765_76520


namespace NUMINAMATH_GPT_find_f_2012_l765_76516

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x / Real.log 2 + b * Real.log x / Real.log 3 + 2

theorem find_f_2012 (a b : ℝ) (h : f (1 / 2012) a b = 5) : f 2012 a b = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2012_l765_76516


namespace NUMINAMATH_GPT_yield_percentage_l765_76550

theorem yield_percentage (d : ℝ) (q : ℝ) (f : ℝ) : d = 12 → q = 150 → f = 100 → (d * f / q) * 100 = 8 :=
by
  intros h_d h_q h_f
  rw [h_d, h_q, h_f]
  sorry

end NUMINAMATH_GPT_yield_percentage_l765_76550


namespace NUMINAMATH_GPT_range_of_angle_A_l765_76511

theorem range_of_angle_A (a b : ℝ) (A : ℝ) (h₁ : a = 2) (h₂ : b = 2 * Real.sqrt 2) 
  (h_triangle : 0 < A ∧ A ≤ Real.pi / 4) :
  (0 < A ∧ A ≤ Real.pi / 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_angle_A_l765_76511


namespace NUMINAMATH_GPT_geometric_sequence_general_term_l765_76580

variable (a : ℕ → ℝ)
variable (n : ℕ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n
  
theorem geometric_sequence_general_term 
  (h_geo : is_geometric_sequence a)
  (h_a3 : a 3 = 3)
  (h_a10 : a 10 = 384) :
  a n = 3 * 2^(n-3) :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_l765_76580


namespace NUMINAMATH_GPT_woman_l765_76553

-- Define the variables and given conditions
variables (W S X : ℕ)
axiom s_eq : S = 27
axiom sum_eq : W + S = 84
axiom w_eq : W = 2 * S + X

theorem woman's_age_more_years : X = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_woman_l765_76553


namespace NUMINAMATH_GPT_f_neg_expression_l765_76555

noncomputable def f : ℝ → ℝ :=
  λ x => if x > 0 then x^2 - 2*x + 3 else sorry

-- Define f by cases: for x > 0 and use the property of odd functions to conclude the expression for x < 0.

theorem f_neg_expression (x : ℝ) (h : x < 0) : f x = -x^2 - 2*x - 3 :=
by
  sorry

end NUMINAMATH_GPT_f_neg_expression_l765_76555


namespace NUMINAMATH_GPT_range_of_alpha_plus_beta_l765_76539

theorem range_of_alpha_plus_beta (α β : ℝ) (h1 : 0 < α - β) (h2 : α - β < π) (h3 : 0 < α + 2 * β) (h4 : α + 2 * β < π) :
  0 < α + β ∧ α + β < π :=
sorry

end NUMINAMATH_GPT_range_of_alpha_plus_beta_l765_76539


namespace NUMINAMATH_GPT_find_P_Q_sum_l765_76535

theorem find_P_Q_sum (P Q : ℤ) 
  (h : ∃ b c : ℤ, x^2 + 3 * x + 2 ∣ x^4 + P * x^2 + Q 
    ∧ b + 3 = 0 
    ∧ c + 3 * b + 6 = P 
    ∧ 3 * c + 2 * b = 0 
    ∧ 2 * c = Q): 
  P + Q = 3 := 
sorry

end NUMINAMATH_GPT_find_P_Q_sum_l765_76535


namespace NUMINAMATH_GPT_geometric_sequence_problem_l765_76562

theorem geometric_sequence_problem 
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h1 : a 7 * a 11 = 6)
  (h2 : a 4 + a 14 = 5) :
  ∃ x : ℝ, x = 2 / 3 ∨ x = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l765_76562


namespace NUMINAMATH_GPT_sum_of_reciprocals_l765_76583

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : (1/x) + (1/y) = 3/8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l765_76583


namespace NUMINAMATH_GPT_residents_rent_contribution_l765_76574

theorem residents_rent_contribution (x R : ℝ) (hx1 : 10 * x + 88 = R) (hx2 : 10.80 * x = 1.025 * R) :
  R / x = 10.54 :=
by sorry

end NUMINAMATH_GPT_residents_rent_contribution_l765_76574


namespace NUMINAMATH_GPT_boatworks_total_canoes_l765_76570

theorem boatworks_total_canoes : 
  let jan := 5
  let feb := 3 * jan
  let mar := 3 * feb
  let apr := 3 * mar
  jan + feb + mar + apr = 200 := 
by 
  sorry

end NUMINAMATH_GPT_boatworks_total_canoes_l765_76570


namespace NUMINAMATH_GPT_parabola_vertex_l765_76531

theorem parabola_vertex (x y : ℝ) : ∀ x y, (y^2 + 8 * y + 2 * x + 11 = 0) → (x = 5 / 2 ∧ y = -4) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_parabola_vertex_l765_76531


namespace NUMINAMATH_GPT_cos_four_times_arccos_val_l765_76534

theorem cos_four_times_arccos_val : 
  ∀ x : ℝ, x = Real.arccos (1 / 4) → Real.cos (4 * x) = 17 / 32 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_cos_four_times_arccos_val_l765_76534


namespace NUMINAMATH_GPT_total_surface_area_of_cube_l765_76527

theorem total_surface_area_of_cube : 
  ∀ (s : Real), 
  (12 * s = 36) → 
  (s * Real.sqrt 3 = 3 * Real.sqrt 3) → 
  6 * s^2 = 54 := 
by
  intros s h1 h2
  sorry

end NUMINAMATH_GPT_total_surface_area_of_cube_l765_76527


namespace NUMINAMATH_GPT_solve_chair_table_fraction_l765_76557

def chair_table_fraction : Prop :=
  ∃ (C T : ℝ), T = 140 ∧ (T + 4 * C = 220) ∧ (C / T = 1 / 7)

theorem solve_chair_table_fraction : chair_table_fraction :=
  sorry

end NUMINAMATH_GPT_solve_chair_table_fraction_l765_76557


namespace NUMINAMATH_GPT_rhombus_side_length_l765_76579

/-
  Define the length of the rhombus diagonal and the area of the rhombus.
-/
def diagonal1 : ℝ := 20
def area : ℝ := 480

/-
  The theorem states that given these conditions, the length of each side of the rhombus is 26 m.
-/
theorem rhombus_side_length (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = diagonal1) (h2 : A = area):
  2 * 26 * 26 * 2 = A * 2 * 2 + (d1 / 2) * (d1 / 2) :=
sorry

end NUMINAMATH_GPT_rhombus_side_length_l765_76579


namespace NUMINAMATH_GPT_average_hamburgers_per_day_l765_76506

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7
def average_per_day : ℕ := total_hamburgers / days_in_week

theorem average_hamburgers_per_day : average_per_day = 9 := by
  sorry

end NUMINAMATH_GPT_average_hamburgers_per_day_l765_76506


namespace NUMINAMATH_GPT_probability_C_l765_76512

-- Variables representing the probabilities of each region
variables (P_A P_B P_C P_D P_E : ℚ)

-- Given conditions
def conditions := P_A = 3/10 ∧ P_B = 1/4 ∧ P_D = 1/5 ∧ P_E = 1/10 ∧ P_A + P_B + P_C + P_D + P_E = 1

-- The statement to prove
theorem probability_C (h : conditions P_A P_B P_C P_D P_E) : P_C = 3/20 := 
by
  sorry

end NUMINAMATH_GPT_probability_C_l765_76512


namespace NUMINAMATH_GPT_chair_arrangements_48_l765_76544

theorem chair_arrangements_48 :
  ∃ (n : ℕ), n = 8 ∧ (∀ (r c : ℕ), r * c = 48 → 2 ≤ r ∧ 2 ≤ c) := 
sorry

end NUMINAMATH_GPT_chair_arrangements_48_l765_76544


namespace NUMINAMATH_GPT_odd_number_divides_3n_plus_1_l765_76559

theorem odd_number_divides_3n_plus_1 (n : ℕ) (h₁ : n % 2 = 1) (h₂ : n ∣ 3^n + 1) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_odd_number_divides_3n_plus_1_l765_76559


namespace NUMINAMATH_GPT_temperature_difference_l765_76507

theorem temperature_difference : 
  let beijing_temp := -6
  let changtai_temp := 15
  changtai_temp - beijing_temp = 21 := 
by
  -- Let the given temperatures
  let beijing_temp := -6
  let changtai_temp := 15
  -- Perform the subtraction and define the expected equality
  show changtai_temp - beijing_temp = 21
  -- Preliminary proof placeholder
  sorry

end NUMINAMATH_GPT_temperature_difference_l765_76507


namespace NUMINAMATH_GPT_smartphones_discount_l765_76542

theorem smartphones_discount
  (discount : ℝ)
  (cost_per_iphone : ℝ)
  (total_saving : ℝ)
  (num_people : ℕ)
  (num_iphones : ℕ)
  (total_cost : ℝ)
  (required_num : ℕ) :
  discount = 0.05 →
  cost_per_iphone = 600 →
  total_saving = 90 →
  num_people = 3 →
  num_iphones = 3 →
  total_cost = num_iphones * cost_per_iphone →
  required_num = num_iphones →
  required_num * cost_per_iphone * discount = total_saving →
  required_num = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_smartphones_discount_l765_76542


namespace NUMINAMATH_GPT_sum_first_seven_terms_geometric_sequence_l765_76582

noncomputable def sum_geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := 
  a * (1 - r^n) / (1 - r)

theorem sum_first_seven_terms_geometric_sequence : 
  sum_geometric_sequence (1/4) (1/4) 7 = 16383 / 49152 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_seven_terms_geometric_sequence_l765_76582


namespace NUMINAMATH_GPT_trip_time_80_minutes_l765_76589

noncomputable def v : ℝ := 1 / 2
noncomputable def speed_highway := 4 * v -- 4 times speed on the highway
noncomputable def time_mountain : ℝ := 20 / v -- Distance on mountain road divided by speed on mountain road
noncomputable def time_highway : ℝ := 80 / speed_highway -- Distance on highway divided by speed on highway
noncomputable def total_time := time_mountain + time_highway

theorem trip_time_80_minutes : total_time = 80 :=
by sorry

end NUMINAMATH_GPT_trip_time_80_minutes_l765_76589


namespace NUMINAMATH_GPT_remainder_37_remainder_73_l765_76590

theorem remainder_37 (N : ℕ) (k : ℕ) (h : N = 1554 * k + 131) : N % 37 = 20 := sorry

theorem remainder_73 (N : ℕ) (k : ℕ) (h : N = 1554 * k + 131) : N % 73 = 58 := sorry

end NUMINAMATH_GPT_remainder_37_remainder_73_l765_76590


namespace NUMINAMATH_GPT_max_value_X2_plus_2XY_plus_3Y2_l765_76509

theorem max_value_X2_plus_2XY_plus_3Y2 
  (x y : ℝ) 
  (h₁ : 0 < x) (h₂ : 0 < y) 
  (h₃ : x^2 - 2 * x * y + 3 * y^2 = 10) : 
  x^2 + 2 * x * y + 3 * y^2 ≤ 30 + 20 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_value_X2_plus_2XY_plus_3Y2_l765_76509


namespace NUMINAMATH_GPT_find_triplet_l765_76501

theorem find_triplet (x y z : ℕ) :
  (1 + (x : ℚ) / (y + z))^2 + (1 + (y : ℚ) / (z + x))^2 + (1 + (z : ℚ) / (x + y))^2 = 27 / 4 ↔ (x, y, z) = (1, 1, 1) :=
by
  sorry

end NUMINAMATH_GPT_find_triplet_l765_76501


namespace NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l765_76581

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l765_76581


namespace NUMINAMATH_GPT_number_of_intersections_l765_76551

theorem number_of_intersections (x y : ℝ) :
  (x^2 + y^2 = 16) ∧ (x = 4) → (x = 4 ∧ y = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_intersections_l765_76551
