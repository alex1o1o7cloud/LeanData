import Mathlib

namespace faster_speed_l260_26019

theorem faster_speed (Speed1 : ℝ) (ExtraDistance : ℝ) (ActualDistance : ℝ) (v : ℝ) : 
  Speed1 = 10 ∧ ExtraDistance = 31 ∧ ActualDistance = 20.67 ∧ 
  (ActualDistance / Speed1 = (ActualDistance + ExtraDistance) / v) → 
  v = 25 :=
by
  sorry

end faster_speed_l260_26019


namespace harry_carries_buckets_rounds_l260_26014

noncomputable def george_rate := 2
noncomputable def total_buckets := 110
noncomputable def total_rounds := 22
noncomputable def harry_buckets_each_round := 3

theorem harry_carries_buckets_rounds :
  (george_rate * total_rounds + harry_buckets_each_round * total_rounds = total_buckets) :=
by sorry

end harry_carries_buckets_rounds_l260_26014


namespace maximum_discount_rate_l260_26061

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l260_26061


namespace largest_mersenne_prime_less_than_500_l260_26035

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_less_than_500 : ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p :=
sorry

end largest_mersenne_prime_less_than_500_l260_26035


namespace geometric_series_first_term_l260_26033

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by {
  sorry
}

end geometric_series_first_term_l260_26033


namespace consecutive_even_product_l260_26002

-- Define that there exist three consecutive even numbers such that the product equals 87526608.
theorem consecutive_even_product (a : ℤ) : 
  (a - 2) * a * (a + 2) = 87526608 → ∃ b : ℤ, b = a - 2 ∧ b % 2 = 0 ∧ ∃ c : ℤ, c = a ∧ c % 2 = 0 ∧ ∃ d : ℤ, d = a + 2 ∧ d % 2 = 0 :=
sorry

end consecutive_even_product_l260_26002


namespace nina_money_l260_26075

theorem nina_money (C : ℝ) (h1 : C > 0) (h2 : 6 * C = 8 * (C - 2)) : 6 * C = 48 :=
by
  sorry

end nina_money_l260_26075


namespace wood_length_equation_l260_26090

theorem wood_length_equation (x : ℝ) :
  (1 / 2) * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l260_26090


namespace fraction_simplification_l260_26041

variable (a b x : ℝ)
variable (h1 : x = a / b)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : a = b * x ^ 2)

theorem fraction_simplification : (a + b) / (a - b) = (x ^ 2 + 1) / (x ^ 2 - 1) := by
  sorry

end fraction_simplification_l260_26041


namespace smallest_n_satisfying_ratio_l260_26067

-- Definitions and conditions from problem
def sum_first_n_odd_numbers_starting_from_3 (n : ℕ) : ℕ := n^2 + 2 * n
def sum_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)

theorem smallest_n_satisfying_ratio :
  ∃ n : ℕ, n > 0 ∧ (sum_first_n_odd_numbers_starting_from_3 n : ℚ) / (sum_first_n_even_numbers n : ℚ) = 49 / 50 ∧ n = 51 :=
by
  use 51
  exact sorry

end smallest_n_satisfying_ratio_l260_26067


namespace length_of_platform_is_correct_l260_26047

noncomputable def length_of_platform : ℝ :=
  let train_length := 200 -- in meters
  let train_speed := 80 * 1000 / 3600 -- kmph to m/s
  let crossing_time := 22 -- in seconds
  (train_speed * crossing_time) - train_length

theorem length_of_platform_is_correct :
  length_of_platform = 2600 / 9 :=
by 
  -- proof would go here
  sorry

end length_of_platform_is_correct_l260_26047


namespace slope_of_line_l260_26023

theorem slope_of_line (x y : ℝ) (h : x + 2 * y + 1 = 0) : y = - (1 / 2) * x - (1 / 2) :=
by
  sorry -- The solution would be filled in here

#check slope_of_line -- additional check to ensure theorem implementation is correct

end slope_of_line_l260_26023


namespace prime_cubed_plus_prime_plus_one_not_square_l260_26091

theorem prime_cubed_plus_prime_plus_one_not_square (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ k : ℕ, k * k = p^3 + p + 1 :=
by
  sorry

end prime_cubed_plus_prime_plus_one_not_square_l260_26091


namespace math_problem_l260_26099

theorem math_problem
  (p q r s : ℕ)
  (hpq : p^3 = q^2)
  (hrs : r^4 = s^3)
  (hrp : r - p = 25) :
  s - q = 73 := by
  sorry

end math_problem_l260_26099


namespace andreas_living_room_floor_area_l260_26056

-- Definitions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_coverage_percentage : ℝ := 0.30
def carpet_area : ℝ := carpet_length * carpet_width

-- Theorem statement
theorem andreas_living_room_floor_area (A : ℝ) 
  (h1 : carpet_coverage_percentage * A = carpet_area) :
  A = 120 :=
by
  sorry

end andreas_living_room_floor_area_l260_26056


namespace conference_center_distance_l260_26021

theorem conference_center_distance
  (d : ℝ)  -- total distance to the conference center
  (t : ℝ)  -- total on-time duration
  (h1 : d = 40 * (t + 1.5))  -- condition from initial speed and late time
  (h2 : d - 40 = 60 * (t - 1.75))  -- condition from increased speed and early arrival
  : d = 310 := 
sorry

end conference_center_distance_l260_26021


namespace simplify_and_evaluate_at_3_l260_26029

noncomputable def expression (x : ℝ) : ℝ := 
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1))

theorem simplify_and_evaluate_at_3 : expression 3 = -5 := 
  sorry

end simplify_and_evaluate_at_3_l260_26029


namespace tree_current_height_l260_26017

theorem tree_current_height 
  (growth_rate_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_height_after_4_months : ℕ) 
  (growth_rate_per_week_eq : growth_rate_per_week = 2)
  (weeks_per_month_eq : weeks_per_month = 4)
  (total_height_after_4_months_eq : total_height_after_4_months = 42) : 
  (∃ (current_height : ℕ), current_height = 10) :=
by
  sorry

end tree_current_height_l260_26017


namespace train_speed_is_72_km_per_hr_l260_26080

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l260_26080


namespace value_of_a_l260_26001

theorem value_of_a (a x : ℝ) (h : (3 * x^2 + 2 * a * x = 0) → (x^3 + a * x^2 - (4 / 3) * a = 0)) :
  a = 0 ∨ a = 3 ∨ a = -3 :=
by
  sorry

end value_of_a_l260_26001


namespace largest_possible_b_l260_26026

theorem largest_possible_b (a b c : ℤ) (h1 : a > b) (h2 : b > c) (h3 : c > 2) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l260_26026


namespace angle_B_l260_26094

/-- 
  Given that the area of triangle ABC is (sqrt 3 / 2) 
  and the dot product of vectors AB and BC is 3, 
  prove that the measure of angle B is 5π/6. 
--/
theorem angle_B (A B C : ℝ) (a c : ℝ) (h1 : 0 ≤ B ∧ B ≤ π)
  (h_area : (1 / 2) * a * c * (Real.sin B) = (Real.sqrt 3 / 2))
  (h_dot : a * c * (Real.cos B) = -3) :
  B = 5 * Real.pi / 6 :=
sorry

end angle_B_l260_26094


namespace find_first_term_of_arithmetic_progression_l260_26049

-- Definitions for the proof
def arithmetic_progression_first_term (L n d : ℕ) : ℕ :=
  L - (n - 1) * d

-- Theorem stating the proof problem
theorem find_first_term_of_arithmetic_progression (L n d : ℕ) (hL : L = 62) (hn : n = 31) (hd : d = 2) :
  arithmetic_progression_first_term L n d = 2 :=
by
  -- proof omitted
  sorry

end find_first_term_of_arithmetic_progression_l260_26049


namespace sum_of_digits_l260_26005

def S (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits :
  (Finset.range 2013).sum S = 28077 :=
by 
  sorry

end sum_of_digits_l260_26005


namespace problem1_problem2_l260_26050

-- Define a and b as real numbers
variables (a b : ℝ)

-- Problem 1: Prove (a-2b)^2 - (b-a)(a+b) = 2a^2 - 4ab + 3b^2
theorem problem1 : (a - 2 * b) ^ 2 - (b - a) * (a + b) = 2 * a ^ 2 - 4 * a * b + 3 * b ^ 2 :=
sorry

-- Problem 2: Prove (2a-b)^2 \cdot (2a+b)^2 = 16a^4 - 8a^2b^2 + b^4
theorem problem2 : (2 * a - b) ^ 2 * (2 * a + b) ^ 2 = 16 * a ^ 4 - 8 * a ^ 2 * b ^ 2 + b ^ 4 :=
sorry

end problem1_problem2_l260_26050


namespace min_bounces_l260_26040

theorem min_bounces
  (h₀ : ℝ := 160)  -- initial height
  (r : ℝ := 3/4)  -- bounce ratio
  (final_h : ℝ := 20)  -- desired height
  (b : ℕ)  -- number of bounces
  : ∃ b, (h₀ * (r ^ b) < final_h ∧ ∀ b', b' < b → ¬(h₀ * (r ^ b') < final_h)) :=
sorry

end min_bounces_l260_26040


namespace max_sin_B_l260_26052

theorem max_sin_B (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (AB BC : ℝ)
    (hAB : AB = 25) (hBC : BC = 20) :
    ∃ sinB : ℝ, sinB = 3 / 5 := sorry

end max_sin_B_l260_26052


namespace polygon_sides_eq_four_l260_26000

theorem polygon_sides_eq_four (n : ℕ)
  (h_interior : (n - 2) * 180 = 360)
  (h_exterior : ∀ (m : ℕ), m = n -> 360 = 360) :
  n = 4 :=
sorry

end polygon_sides_eq_four_l260_26000


namespace distance_from_town_l260_26046

theorem distance_from_town (d : ℝ) :
  (7 < d ∧ d < 8) ↔ (d < 8 ∧ d > 7 ∧ d > 6 ∧ d ≠ 9) :=
by sorry

end distance_from_town_l260_26046


namespace num_five_digit_integers_l260_26077

theorem num_five_digit_integers
  (total_digits : ℕ := 8)
  (repeat_3 : ℕ := 2)
  (repeat_6 : ℕ := 3)
  (repeat_8 : ℕ := 2)
  (arrangements : ℕ := Nat.factorial total_digits / (Nat.factorial repeat_3 * Nat.factorial repeat_6 * Nat.factorial repeat_8)) :
  arrangements = 1680 := by
  sorry

end num_five_digit_integers_l260_26077


namespace collinear_points_l260_26051

variables (a b : ℝ × ℝ) (A B C D : ℝ × ℝ)

-- Define the vectors
noncomputable def vec_AB : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
noncomputable def vec_BC : ℝ × ℝ := (2 * a.1 + 8 * b.1, 2 * a.2 + 8 * b.2)
noncomputable def vec_CD : ℝ × ℝ := (3 * (a.1 - b.1), 3 * (a.2 - b.2))

-- Define the collinearity condition
def collinear (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Translate the problem statement into Lean
theorem collinear_points (h₀ : a ≠ (0, 0)) (h₁ : b ≠ (0, 0)) (h₂ : ¬ (a.1 * b.2 - a.2 * b.1 = 0)):
  collinear (6 * (a.1 + b.1), 6 * (a.2 + b.2)) (5 * (a.1 + b.1, a.2 + b.2)) :=
sorry

end collinear_points_l260_26051


namespace calc_fraction_l260_26055

theorem calc_fraction:
  (125: ℕ) = 5 ^ 3 →
  (25: ℕ) = 5 ^ 2 →
  (25 ^ 40) / (125 ^ 20) = 5 ^ 20 :=
by
  intros h1 h2
  sorry

end calc_fraction_l260_26055


namespace positive_integer_solutions_count_l260_26076

theorem positive_integer_solutions_count :
  ∃ (s : Finset ℕ), (∀ x ∈ s, 24 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 64) ∧ s.card = 4 := 
by
  sorry

end positive_integer_solutions_count_l260_26076


namespace Jane_indisposed_days_l260_26068

-- Definitions based on conditions
def John_completion_days := 18
def Jane_completion_days := 12
def total_task_days := 10.8
def work_per_day_by_john := 1 / John_completion_days
def work_per_day_by_jane := 1 / Jane_completion_days
def work_per_day_together := work_per_day_by_john + work_per_day_by_jane

-- Equivalent proof problem
theorem Jane_indisposed_days : 
  ∃ (x : ℝ), 
    (10.8 - x) * work_per_day_together + x * work_per_day_by_john = 1 ∧
    x = 6 := 
by 
  sorry

end Jane_indisposed_days_l260_26068


namespace quadratic_distinct_real_roots_range_quadratic_root_product_value_l260_26064

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → k > 3 / 4 :=
sorry

theorem quadratic_root_product_value (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → (∀ x₁ x₂ : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → x₁ * x₂ = 5) 
  → k = 2 :=
sorry

end quadratic_distinct_real_roots_range_quadratic_root_product_value_l260_26064


namespace number_exceeds_25_percent_by_150_l260_26095

theorem number_exceeds_25_percent_by_150 (x : ℝ) : (0.25 * x + 150 = x) → x = 200 :=
by
  sorry

end number_exceeds_25_percent_by_150_l260_26095


namespace blue_tint_percentage_in_new_mixture_l260_26004

-- Define the conditions given in the problem
def original_volume : ℝ := 40
def blue_tint_percentage : ℝ := 0.20
def added_blue_tint_volume : ℝ := 8

-- Calculate the original blue tint volume
def original_blue_tint_volume := blue_tint_percentage * original_volume

-- Calculate the new blue tint volume after adding more blue tint
def new_blue_tint_volume := original_blue_tint_volume + added_blue_tint_volume

-- Calculate the new total volume of the mixture
def new_total_volume := original_volume + added_blue_tint_volume

-- Define the expected result in percentage
def expected_blue_tint_percentage : ℝ := 33.3333

-- Statement to prove
theorem blue_tint_percentage_in_new_mixture :
  (new_blue_tint_volume / new_total_volume) * 100 = expected_blue_tint_percentage :=
sorry

end blue_tint_percentage_in_new_mixture_l260_26004


namespace OJ_perpendicular_PQ_l260_26045

noncomputable def quadrilateral (A B C D : Point) : Prop := sorry

noncomputable def inscribed (A B C D : Point) : Prop := sorry

noncomputable def circumscribed (A B C D : Point) : Prop := sorry

noncomputable def no_diameter (A B C D : Point) : Prop := sorry

noncomputable def intersection_of_external_bisectors (A B C D : Point) (P : Point) : Prop := sorry

noncomputable def incenter (A B C D J : Point) : Prop := sorry

noncomputable def circumcenter (A B C D O : Point) : Prop := sorry

noncomputable def PQ_perpendicular (O J P Q : Point) : Prop := sorry

theorem OJ_perpendicular_PQ (A B C D P Q J O : Point) :
  quadrilateral A B C D →
  inscribed A B C D →
  circumscribed A B C D →
  no_diameter A B C D →
  intersection_of_external_bisectors A B C D P →
  intersection_of_external_bisectors C D A B Q →
  incenter A B C D J →
  circumcenter A B C D O →
  PQ_perpendicular O J P Q :=
sorry

end OJ_perpendicular_PQ_l260_26045


namespace original_price_of_cycle_l260_26032

variable (P : ℝ)

theorem original_price_of_cycle (h : 0.92 * P = 1610) : P = 1750 :=
sorry

end original_price_of_cycle_l260_26032


namespace find_b_plus_c_l260_26070

-- Definitions based on the given conditions.
variables {A : ℝ} {a b c : ℝ}

-- The conditions in the problem
theorem find_b_plus_c
  (h_cosA : Real.cos A = 1 / 3)
  (h_a : a = Real.sqrt 3)
  (h_bc : b * c = 3 / 2) :
  b + c = Real.sqrt 7 :=
sorry

end find_b_plus_c_l260_26070


namespace find_smallest_b_l260_26039

theorem find_smallest_b :
  ∃ b : ℕ, 
    (∀ r s : ℤ, r * s = 3960 → r + s ≠ b ∨ r + s > 0) ∧ 
    (∀ r s : ℤ, r * s = 3960 → (r + s < b → r + s ≤ 0)) ∧ 
    b = 126 :=
by
  sorry

end find_smallest_b_l260_26039


namespace divide_90_into_two_parts_l260_26009

theorem divide_90_into_two_parts (x y : ℝ) (h : x + y = 90) 
  (cond : 0.4 * x = 0.3 * y + 15) : x = 60 ∨ y = 60 := 
by
  sorry

end divide_90_into_two_parts_l260_26009


namespace points_four_units_away_l260_26060

theorem points_four_units_away (x : ℤ) : (x - (-1) = 4 ∨ x - (-1) = -4) ↔ (x = 3 ∨ x = -5) :=
by
  sorry

end points_four_units_away_l260_26060


namespace square_of_negative_is_positive_l260_26053

-- Define P as a negative integer
variable (P : ℤ) (hP : P < 0)

-- Theorem statement that P² is always positive.
theorem square_of_negative_is_positive : P^2 > 0 :=
sorry

end square_of_negative_is_positive_l260_26053


namespace add_second_largest_to_sum_l260_26030

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 5 ∨ d = 8

def form_number (d1 d2 d3 : ℕ) : ℕ := 100 * d1 + 10 * d2 + d3

def largest_number : ℕ := form_number 8 5 2
def smallest_number : ℕ := form_number 2 5 8
def second_largest_number : ℕ := form_number 8 2 5

theorem add_second_largest_to_sum : 
  second_largest_number + (largest_number + smallest_number) = 1935 := 
  sorry

end add_second_largest_to_sum_l260_26030


namespace inverse_of_B_cubed_l260_26058

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
def B_inv := Matrix.of ![![3, -2], ![0, -1]]
noncomputable def B_cubed_inv := ((B_inv) 3)^3

theorem inverse_of_B_cubed :
  B_inv = Matrix.of ![![27, -24], ![0, -1]] :=
by
  sorry

end inverse_of_B_cubed_l260_26058


namespace percentage_charge_l260_26072

def car_cost : ℝ := 14600
def initial_savings : ℝ := 14500
def trip_charge : ℝ := 1.5
def number_of_trips : ℕ := 40
def grocery_value : ℝ := 800
def final_savings_needed : ℝ := car_cost - initial_savings

-- The amount earned from trips
def amount_from_trips : ℝ := number_of_trips * trip_charge

-- The amount needed from percentage charge on groceries
def amount_from_percentage (P: ℝ) : ℝ := grocery_value * P

-- The required amount from percentage charge on groceries
def required_amount_from_percentage : ℝ := final_savings_needed - amount_from_trips

theorem percentage_charge (P: ℝ) (h: amount_from_percentage P = required_amount_from_percentage) : P = 0.05 :=
by 
  -- Proof follows from the given condition that amount_from_percentage P = required_amount_from_percentage
  sorry

end percentage_charge_l260_26072


namespace arithmetic_sequence_tenth_term_l260_26062

theorem arithmetic_sequence_tenth_term :
  ∀ (a : ℚ) (a_20 : ℚ) (a_10 : ℚ),
    a = 5 / 11 →
    a_20 = 9 / 11 →
    a_10 = a + (9 * ((a_20 - a) / 19)) →
    a_10 = 1233 / 2309 :=
by
  intros a a_20 a_10 h_a h_a_20 h_a_10
  sorry

end arithmetic_sequence_tenth_term_l260_26062


namespace tasks_completed_correctly_l260_26089

theorem tasks_completed_correctly (x y : ℕ) (h1 : 9 * x - 5 * y = 57) (h2 : x + y ≤ 15) : x = 8 := 
by
  sorry

end tasks_completed_correctly_l260_26089


namespace total_marbles_l260_26085

theorem total_marbles (boxes : ℕ) (marbles_per_box : ℕ) (h1 : boxes = 10) (h2 : marbles_per_box = 100) : (boxes * marbles_per_box = 1000) :=
by
  sorry

end total_marbles_l260_26085


namespace initial_average_mark_l260_26036

-- Define the initial conditions
def num_students : ℕ := 9
def excluded_students_avg : ℕ := 44
def remaining_students_avg : ℕ := 80

-- Define the variables for total marks we calculated in the solution
def total_marks_initial := num_students * (num_students * excluded_students_avg / 5 + remaining_students_avg / (num_students - 5) * (num_students - 5))

-- The theorem we need to prove:
theorem initial_average_mark :
  (num_students * (excluded_students_avg * 5 + remaining_students_avg * (num_students - 5))) / num_students = 60 := 
  by
  -- step-by-step solution proof could go here, but we use sorry as placeholder
  sorry

end initial_average_mark_l260_26036


namespace points_on_ellipse_satisfying_dot_product_l260_26081

theorem points_on_ellipse_satisfying_dot_product :
  ∃ P1 P2 : ℝ × ℝ,
    P1 = (0, 3) ∧ P2 = (0, -3) ∧
    ∀ P : ℝ × ℝ, 
    (P ∈ ({p : ℝ × ℝ | (p.1 / 5)^2 + (p.2 / 3)^2 = 1}) → 
     ((P.1 - (-4)) * (P.1 - 4) + P.2^2 = -7) →
     (P = P1 ∨ P = P2))
:=
sorry

end points_on_ellipse_satisfying_dot_product_l260_26081


namespace Julie_work_hours_per_week_l260_26083

variable (hours_per_week_summer : ℕ) (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (weeks_school_year : ℕ)
variable (earnings_school_year : ℕ)

theorem Julie_work_hours_per_week :
  hours_per_week_summer = 40 →
  weeks_summer = 10 →
  earnings_summer = 4000 →
  weeks_school_year = 40 →
  earnings_school_year = 4000 →
  (∀ rate_per_hour, rate_per_hour = earnings_summer / (hours_per_week_summer * weeks_summer) →
  (earnings_school_year / (weeks_school_year * rate_per_hour) = 10)) :=
by intros h1 h2 h3 h4 h5 rate_per_hour hr; sorry

end Julie_work_hours_per_week_l260_26083


namespace probability_spade_then_ace_l260_26079

theorem probability_spade_then_ace :
  let total_cards := 52
  let total_aces := 4
  let total_spades := 13
  let ace_of_spades := 1
  let non_ace_spades := total_spades - ace_of_spades
  (non_ace_spades / total_cards) * (total_aces / (total_cards - 1)) +
  (ace_of_spades / total_cards) * ((total_aces - ace_of_spades) / (total_cards - 1)) = (1 / 52) :=
by
  sorry

end probability_spade_then_ace_l260_26079


namespace seashells_total_l260_26086

def seashells_sam : ℕ := 18
def seashells_mary : ℕ := 47
def seashells_john : ℕ := 32
def seashells_emily : ℕ := 26

theorem seashells_total : seashells_sam + seashells_mary + seashells_john + seashells_emily = 123 := by
    sorry

end seashells_total_l260_26086


namespace binar_operation_correct_l260_26073

theorem binar_operation_correct : 
  let a := 13  -- 1101_2 in decimal
  let b := 15  -- 1111_2 in decimal
  let c := 9   -- 1001_2 in decimal
  let d := 2   -- 10_2 in decimal
  a + b - c * d = 10 ↔ "1010" = "1010" := 
by 
  intros
  simp
  sorry

end binar_operation_correct_l260_26073


namespace numbers_not_equal_l260_26078

theorem numbers_not_equal
  (a b c S : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h1 : a + b^2 + c^2 = S)
  (h2 : b + a^2 + c^2 = S)
  (h3 : c + a^2 + b^2 = S) :
  ¬ (a = b ∧ b = c) :=
by sorry

end numbers_not_equal_l260_26078


namespace other_factor_of_lcm_l260_26084

theorem other_factor_of_lcm (A B : ℕ) 
  (hcf : Nat.gcd A B = 23) 
  (hA : A = 345) 
  (hcf_factor : 15 ∣ Nat.lcm A B) 
  : 23 ∣ Nat.lcm A B / 15 :=
sorry

end other_factor_of_lcm_l260_26084


namespace find_x_l260_26098

theorem find_x (a b x : ℝ) (h1 : ∀ a b, a * b = 2 * a - b) (h2 : 2 * (6 * x) = 2) : x = 10 := 
sorry

end find_x_l260_26098


namespace uniformity_of_scores_l260_26028

/- Problem statement:
  Randomly select 10 students from class A and class B to participate in an English oral test. 
  The variances of their test scores are S1^2 = 13.2 and S2^2 = 26.26, respectively. 
  Then, we show that the scores of the 10 students from class A are more uniform than 
  those of the 10 students from class B.
-/

theorem uniformity_of_scores (S1 S2 : ℝ) (h1 : S1^2 = 13.2) (h2 : S2^2 = 26.26) : 
    13.2 < 26.26 := 
by 
  sorry

end uniformity_of_scores_l260_26028


namespace max_sum_value_l260_26013

noncomputable def max_sum (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) : ℝ :=
  x + y

theorem max_sum_value :
  ∃ x y : ℝ, ∃ h : 3 * (x^2 + y^2) = x - y, max_sum x y h = 1/3 :=
sorry

end max_sum_value_l260_26013


namespace unlock_probability_l260_26082

/--
Xiao Ming set a six-digit passcode for his phone using the numbers 0-9, but he forgot the last digit.
The probability that Xiao Ming can unlock his phone with just one try is 1/10.
-/
theorem unlock_probability (n : ℕ) (h : n ≥ 0 ∧ n ≤ 9) : 
  1 / 10 = 1 / (10 : ℝ) :=
by
  -- Skipping proof
  sorry

end unlock_probability_l260_26082


namespace cats_weight_difference_l260_26059

-- Define the weights of Anne's and Meg's cats
variables (A M : ℕ)

-- Given conditions:
-- 1. Ratio of weights Meg's cat to Anne's cat is 13:21
-- 2. Meg's cat's weight is 20 kg plus half the weight of Anne's cat

theorem cats_weight_difference (h1 : M = 20 + (A / 2)) (h2 : 13 * A = 21 * M) : A - M = 64 := 
by {
    sorry
}

end cats_weight_difference_l260_26059


namespace c_work_time_l260_26042

theorem c_work_time (A B C : ℝ) 
  (h1 : A + B = 1/10) 
  (h2 : B + C = 1/5) 
  (h3 : C + A = 1/15) : 
  C = 1/12 :=
by
  -- Proof will go here
  sorry

end c_work_time_l260_26042


namespace baker_extra_cakes_l260_26088

-- Defining the conditions
def original_cakes : ℕ := 78
def total_cakes : ℕ := 87
def extra_cakes := total_cakes - original_cakes

-- The statement to prove
theorem baker_extra_cakes : extra_cakes = 9 := by
  sorry

end baker_extra_cakes_l260_26088


namespace power_decomposition_l260_26008

theorem power_decomposition (n m : ℕ) (h1 : n ≥ 2) 
  (h2 : n * n = 1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19) 
  (h3 : Nat.succ 19 = 21) 
  : m + n = 15 := sorry

end power_decomposition_l260_26008


namespace expected_adjacent_black_l260_26066

noncomputable def ExpectedBlackPairs :=
  let totalCards := 104
  let blackCards := 52
  let totalPairs := 103
  let probAdjacentBlack := (blackCards - 1) / (totalPairs)
  blackCards * probAdjacentBlack

theorem expected_adjacent_black :
  ExpectedBlackPairs = 2601 / 103 :=
by
  sorry

end expected_adjacent_black_l260_26066


namespace inequality_preservation_l260_26043

theorem inequality_preservation (a b : ℝ) (h : a > b) : (1/3 : ℝ) * a - 1 > (1/3 : ℝ) * b - 1 := 
by sorry

end inequality_preservation_l260_26043


namespace mike_total_cards_l260_26012

variable (original_cards : ℕ) (birthday_cards : ℕ)

def initial_cards : ℕ := 64
def received_cards : ℕ := 18

theorem mike_total_cards :
  original_cards = 64 →
  birthday_cards = 18 →
  original_cards + birthday_cards = 82 :=
by
  intros
  sorry

end mike_total_cards_l260_26012


namespace tablecloth_diameter_l260_26025

theorem tablecloth_diameter (r : ℝ) (h : r = 5) : 2 * r = 10 :=
by
  simp [h]
  sorry

end tablecloth_diameter_l260_26025


namespace gcd_lcm_sum_l260_26007

theorem gcd_lcm_sum (a b : ℕ) (ha : a = 45) (hb : b = 4050) :
  Nat.gcd a b + Nat.lcm a b = 4095 := by
  sorry

end gcd_lcm_sum_l260_26007


namespace statement_A_statement_E_statement_B_false_statement_C_false_statement_D_false_l260_26010

theorem statement_A : ∃ n : ℤ, 20 = 4 * n := by 
  sorry

theorem statement_E : ∃ n : ℤ, 180 = 9 * n := by 
  sorry

theorem statement_B_false : ¬ (19 ∣ 57) := by 
  sorry

theorem statement_C_false : 30 ∣ 90 := by 
  sorry

theorem statement_D_false : 17 ∣ 51 := by 
  sorry

end statement_A_statement_E_statement_B_false_statement_C_false_statement_D_false_l260_26010


namespace percent_increase_is_fifteen_l260_26006

noncomputable def percent_increase_from_sale_price_to_regular_price (P : ℝ) : ℝ :=
  ((P - (0.87 * P)) / (0.87 * P)) * 100

theorem percent_increase_is_fifteen (P : ℝ) (h : P > 0) :
  percent_increase_from_sale_price_to_regular_price P = 15 :=
by
  -- The proof is not required, so we use sorry.
  sorry

end percent_increase_is_fifteen_l260_26006


namespace arithmetic_sequence_a1_a6_l260_26093

theorem arithmetic_sequence_a1_a6
  (a : ℕ → ℤ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h_a2 : a 2 = 3)
  (h_sum : a 3 + a 4 = 9) : a 1 * a 6 = 14 :=
sorry

end arithmetic_sequence_a1_a6_l260_26093


namespace number_is_160_l260_26087

theorem number_is_160 (x : ℝ) (h : x / 5 + 4 = x / 4 - 4) : x = 160 :=
by
  sorry

end number_is_160_l260_26087


namespace pastries_solution_l260_26074

def pastries_problem : Prop :=
  ∃ (F Calvin Phoebe Grace : ℕ),
  (Calvin = F + 8) ∧
  (Phoebe = F + 8) ∧
  (Grace = 30) ∧
  (F + Calvin + Phoebe + Grace = 97) ∧
  (Grace - Calvin = 5) ∧
  (Grace - Phoebe = 5)

theorem pastries_solution : pastries_problem :=
by
  sorry

end pastries_solution_l260_26074


namespace sum_divisibility_l260_26037

theorem sum_divisibility (a b : ℤ) (h : 6 * a + 11 * b ≡ 0 [ZMOD 31]) : a + 7 * b ≡ 0 [ZMOD 31] :=
sorry

end sum_divisibility_l260_26037


namespace radius_of_shorter_cylinder_l260_26069

theorem radius_of_shorter_cylinder (h r : ℝ) (V_s V_t : ℝ) (π : ℝ) : 
  V_s = 500 → 
  V_t = 500 → 
  V_t = π * 5^2 * 4 * h → 
  V_s = π * r^2 * h → 
  r = 10 :=
by 
  sorry

end radius_of_shorter_cylinder_l260_26069


namespace complete_square_l260_26011

theorem complete_square {x : ℝ} :
  x^2 - 6 * x - 8 = 0 ↔ (x - 3)^2 = 17 :=
sorry

end complete_square_l260_26011


namespace sum_of_interior_angles_l260_26054

theorem sum_of_interior_angles (n : ℕ) (interior_angle : ℝ) :
  (interior_angle = 144) → (180 - 144) * n = 360 → n = 10 → (n - 2) * 180 = 1440 :=
by
  intros h1 h2 h3
  sorry

end sum_of_interior_angles_l260_26054


namespace find_height_of_door_l260_26071

noncomputable def height_of_door (x : ℝ) (w : ℝ) (h : ℝ) : ℝ := h

theorem find_height_of_door :
  ∃ x w h, (w = x - 4) ∧ (h = x - 2) ∧ (x^2 = w^2 + h^2) ∧ height_of_door x w h = 8 :=
by {
  sorry
}

end find_height_of_door_l260_26071


namespace distance_between_cities_l260_26097

-- Definitions
def map_distance : ℝ := 120 -- Distance on the map in cm
def scale_factor : ℝ := 10  -- Scale factor in km per cm

-- Theorem statement
theorem distance_between_cities :
  map_distance * scale_factor = 1200 :=
by
  sorry

end distance_between_cities_l260_26097


namespace power_mod_444_444_l260_26024

open Nat

theorem power_mod_444_444 : (444:ℕ)^444 % 13 = 1 := by
  sorry

end power_mod_444_444_l260_26024


namespace lucky_sum_mod_1000_l260_26020

def is_lucky (n : ℕ) : Prop := ∀ d ∈ n.digits 10, d = 7

def first_twenty_lucky_numbers : List ℕ :=
  [7, 77] ++ List.replicate 18 777

theorem lucky_sum_mod_1000 :
  (first_twenty_lucky_numbers.sum % 1000) = 70 := 
sorry

end lucky_sum_mod_1000_l260_26020


namespace crayons_per_box_l260_26092

theorem crayons_per_box (total_crayons : ℕ) (total_boxes : ℕ)
  (h1 : total_crayons = 321)
  (h2 : total_boxes = 45) :
  (total_crayons / total_boxes) = 7 :=
by
  sorry

end crayons_per_box_l260_26092


namespace rectangular_solid_edges_sum_l260_26015

theorem rectangular_solid_edges_sum
  (b s : ℝ)
  (h_vol : (b / s) * b * (b * s) = 432)
  (h_sa : 2 * ((b ^ 2 / s) + b ^ 2 * s + b ^ 2) = 432)
  (h_gp : 0 < s ∧ s ≠ 1) :
  4 * (b / s + b + b * s) = 144 := 
by
  sorry

end rectangular_solid_edges_sum_l260_26015


namespace floor_of_neg_seven_fourths_l260_26003

theorem floor_of_neg_seven_fourths : 
  Int.floor (-7 / 4 : ℚ) = -2 := 
by sorry

end floor_of_neg_seven_fourths_l260_26003


namespace intersection_of_lines_l260_26096

theorem intersection_of_lines
    (x y : ℚ) 
    (h1 : y = 3 * x - 1)
    (h2 : y + 4 = -6 * x) :
    x = -1 / 3 ∧ y = -2 := 
sorry

end intersection_of_lines_l260_26096


namespace factor_expression_l260_26044

theorem factor_expression (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = (a - b) * (b - c) * (c - a) * (a * b^2 + a * c^2) :=
by 
  sorry

end factor_expression_l260_26044


namespace largest_of_seven_consecutive_integers_l260_26034

theorem largest_of_seven_consecutive_integers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 2821) : 
  n + 6 = 406 := 
by
  -- Proof steps can be added here
  sorry

end largest_of_seven_consecutive_integers_l260_26034


namespace rowing_speed_in_still_water_l260_26063

theorem rowing_speed_in_still_water (speed_of_current : ℝ) (time_seconds : ℝ) (distance_meters : ℝ) (S : ℝ)
  (h_current : speed_of_current = 3) 
  (h_time : time_seconds = 9.390553103577801) 
  (h_distance : distance_meters = 60) 
  (h_S : S = 20) : 
  (distance_meters / 1000) / (time_seconds / 3600) - speed_of_current = S :=
by 
  sorry

end rowing_speed_in_still_water_l260_26063


namespace find_x_y_z_l260_26038

theorem find_x_y_z (x y z : ℝ) (h1 : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z) (h2 : x * y * z = 10)
  (h3 : x ^ Real.log x * y ^ Real.log y * z ^ Real.log z = 10) :
  (x = 1 ∧ y = 1 ∧ z = 10) ∨ (x = 10 ∧ y = 1 ∧ z = 1) ∨ (x = 1 ∧ y = 10 ∧ z = 1) :=
sorry

end find_x_y_z_l260_26038


namespace perimeter_of_square_fence_l260_26016

theorem perimeter_of_square_fence :
  ∀ (n : ℕ) (post_gap post_width : ℝ), 
  4 * n - 4 = 24 →
  post_gap = 6 →
  post_width = 5 / 12 →
  4 * ((n - 1) * post_gap + n * post_width) = 156 :=
by
  intros n post_gap post_width h1 h2 h3
  sorry

end perimeter_of_square_fence_l260_26016


namespace scientific_notation_of_viewers_l260_26022

def million : ℝ := 10^6
def viewers : ℝ := 70.62 * million

theorem scientific_notation_of_viewers : viewers = 7.062 * 10^7 := by
  sorry

end scientific_notation_of_viewers_l260_26022


namespace stratified_sampling_correct_l260_26048

-- Definitions based on the conditions
def total_employees : ℕ := 300
def over_40 : ℕ := 50
def between_30_and_40 : ℕ := 150
def under_30 : ℕ := 100
def sample_size : ℕ := 30
def stratified_ratio : ℕ := 1 / 10  -- sample_size / total_employees

-- Function to compute the number of individuals sampled from each age group
def sampled_from_age_group (group_size : ℕ) : ℕ :=
  group_size * stratified_ratio

-- Mathematical properties to be proved
theorem stratified_sampling_correct :
  sampled_from_age_group over_40 = 5 ∧ 
  sampled_from_age_group between_30_and_40 = 15 ∧ 
  sampled_from_age_group under_30 = 10 := by
  sorry

end stratified_sampling_correct_l260_26048


namespace find_x_l260_26065

noncomputable def x : ℝ := 20

def condition1 (x : ℝ) : Prop := x > 0
def condition2 (x : ℝ) : Prop := x / 100 * 150 - 20 = 10

theorem find_x (x : ℝ) : condition1 x ∧ condition2 x ↔ x = 20 :=
by
  sorry

end find_x_l260_26065


namespace rectangle_area_l260_26057

variable (l w : ℕ)

def length_is_three_times_width := l = 3 * w

def perimeter_is_160 := 2 * l + 2 * w = 160

theorem rectangle_area : 
  length_is_three_times_width l w → 
  perimeter_is_160 l w → 
  l * w = 1200 :=
by
  intros h₁ h₂
  sorry

end rectangle_area_l260_26057


namespace power_function_alpha_l260_26018

theorem power_function_alpha (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^α) (point_condition : f 8 = 2) : 
  α = 1 / 3 :=
by
  sorry

end power_function_alpha_l260_26018


namespace shelves_per_case_l260_26027

noncomputable section

-- Define the total number of ridges
def total_ridges : ℕ := 8640

-- Define the number of ridges per record
def ridges_per_record : ℕ := 60

-- Define the number of records per shelf when the shelf is 60% full
def records_per_shelf : ℕ := (60 * 20) / 100

-- Define the number of ridges per shelf
def ridges_per_shelf : ℕ := records_per_shelf * ridges_per_record

-- Given 4 cases, we need to determine the number of shelves per case
theorem shelves_per_case (cases shelves : ℕ) (h₁ : cases = 4) (h₂ : shelves * ridges_per_shelf = total_ridges) :
  shelves / cases = 3 := by
  sorry

end shelves_per_case_l260_26027


namespace find_n_tan_l260_26031

theorem find_n_tan (n : ℤ) (hn : -90 < n ∧ n < 90) (htan : Real.tan (n * Real.pi / 180) = Real.tan (312 * Real.pi / 180)) : 
  n = -48 := 
sorry

end find_n_tan_l260_26031
