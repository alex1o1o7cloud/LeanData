import Mathlib

namespace girl_from_grade_4_probability_l848_84874

-- Number of girls and boys in grade 3
def girls_grade_3 := 28
def boys_grade_3 := 35
def total_grade_3 := girls_grade_3 + boys_grade_3

-- Number of girls and boys in grade 4
def girls_grade_4 := 45
def boys_grade_4 := 42
def total_grade_4 := girls_grade_4 + boys_grade_4

-- Number of girls and boys in grade 5
def girls_grade_5 := 38
def boys_grade_5 := 51
def total_grade_5 := girls_grade_5 + boys_grade_5

-- Total number of children in playground
def total_children := total_grade_3 + total_grade_4 + total_grade_5

-- Probability that a randomly selected child is a girl from grade 4
def probability_girl_grade_4 := (girls_grade_4: ℚ) / total_children

theorem girl_from_grade_4_probability :
  probability_girl_grade_4 = 45 / 239 := by
  sorry

end girl_from_grade_4_probability_l848_84874


namespace geom_seq_sum_elems_l848_84828

theorem geom_seq_sum_elems (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end geom_seq_sum_elems_l848_84828


namespace car_speed_15_seconds_less_l848_84843

theorem car_speed_15_seconds_less (v : ℝ) : 
  (∀ v, 75 = 3600 / v + 15) → v = 60 :=
by
  intro H
  -- Proof goes here
  sorry

end car_speed_15_seconds_less_l848_84843


namespace dish_heats_up_by_5_degrees_per_minute_l848_84889

theorem dish_heats_up_by_5_degrees_per_minute
  (final_temperature initial_temperature : ℕ)
  (time_taken : ℕ)
  (h1 : final_temperature = 100)
  (h2 : initial_temperature = 20)
  (h3 : time_taken = 16) :
  (final_temperature - initial_temperature) / time_taken = 5 :=
by
  sorry

end dish_heats_up_by_5_degrees_per_minute_l848_84889


namespace remaining_pie_proportion_l848_84852

def carlos_portion : ℝ := 0.6
def maria_share_of_remainder : ℝ := 0.25

theorem remaining_pie_proportion: 
  (1 - carlos_portion) - maria_share_of_remainder * (1 - carlos_portion) = 0.3 := 
by
  -- proof to be implemented here
  sorry

end remaining_pie_proportion_l848_84852


namespace find_last_week_rope_l848_84847

/-- 
Description: Mr. Sanchez bought 4 feet of rope less than he did the previous week. 
Given that he bought 96 inches in total, find how many feet he bought last week.
--/
theorem find_last_week_rope (F : ℕ) :
  12 * (F - 4) = 96 → F = 12 := by
  sorry

end find_last_week_rope_l848_84847


namespace determine_b_from_inequality_l848_84895

theorem determine_b_from_inequality (b : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - b * x + 6 < 0) → b = 5 :=
by
  intro h
  -- Proof can be added here
  sorry

end determine_b_from_inequality_l848_84895


namespace line_eq_l848_84800

theorem line_eq (x_1 y_1 x_2 y_2 : ℝ) (h1 : x_1 + x_2 = 8) (h2 : y_1 + y_2 = 2)
  (h3 : x_1^2 - 4 * y_1^2 = 4) (h4 : x_2^2 - 4 * y_2^2 = 4) :
  ∃ l : ℝ, ∀ x y : ℝ, x - y - 3 = l :=
by sorry

end line_eq_l848_84800


namespace min_number_of_4_dollar_frisbees_l848_84848

theorem min_number_of_4_dollar_frisbees 
  (x y : ℕ) 
  (h1 : x + y = 60)
  (h2 : 3 * x + 4 * y = 200) 
  : y = 20 :=
sorry

end min_number_of_4_dollar_frisbees_l848_84848


namespace log_expression_l848_84846

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression :
  log_base 4 16 - (log_base 2 3 * log_base 3 2) = 1 := by
  sorry

end log_expression_l848_84846


namespace infinite_sum_equals_l848_84881

theorem infinite_sum_equals :
  10 * (79 * (1 / 7)) + (∑' n : ℕ, if n % 2 = 0 then (if n = 0 then 0 else 2 / 7 ^ n) else (1 / 7 ^ n)) = 3 / 16 :=
by
  sorry

end infinite_sum_equals_l848_84881


namespace base8_digits_sum_l848_84875

-- Define digits and their restrictions
variables {A B C : ℕ}

-- Main theorem
theorem base8_digits_sum (h1 : 0 < A ∧ A < 8)
                         (h2 : 0 < B ∧ B < 8)
                         (h3 : 0 < C ∧ C < 8)
                         (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
                         (condition : (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) = (8^2 + 8 + 1) * 8 * A) :
  A + B + C = 8 := 
sorry

end base8_digits_sum_l848_84875


namespace minimize_pollution_park_distance_l848_84829

noncomputable def pollution_index (x : ℝ) : ℝ :=
  (1 / x) + (4 / (30 - x))

theorem minimize_pollution_park_distance : ∃ x : ℝ, (0 < x ∧ x < 30) ∧ pollution_index x = 10 :=
by
  sorry

end minimize_pollution_park_distance_l848_84829


namespace second_container_sand_capacity_l848_84864

def volume (h: ℕ) (w: ℕ) (l: ℕ) : ℕ := h * w * l

def sand_capacity (v1: ℕ) (s1: ℕ) (v2: ℕ) : ℕ := (s1 * v2) / v1

theorem second_container_sand_capacity:
  let h1 := 3
  let w1 := 4
  let l1 := 6
  let s1 := 72
  let h2 := 3 * h1
  let w2 := 2 * w1
  let l2 := l1
  let v1 := volume h1 w1 l1
  let v2 := volume h2 w2 l2
  sand_capacity v1 s1 v2 = 432 :=
by {
  sorry
}

end second_container_sand_capacity_l848_84864


namespace john_runs_with_dog_for_half_hour_l848_84868

noncomputable def time_with_dog_in_hours (t : ℝ) : Prop := 
  let d1 := 6 * t          -- Distance run with the dog
  let d2 := 4 * (1 / 2)    -- Distance run alone
  (d1 + d2 = 5) ∧ (t = 1 / 2)

theorem john_runs_with_dog_for_half_hour : ∃ t : ℝ, time_with_dog_in_hours t := 
by
  use (1 / 2)
  sorry

end john_runs_with_dog_for_half_hour_l848_84868


namespace intersect_sets_l848_84817

def A := {x : ℝ | x > -1}
def B := {x : ℝ | x ≤ 5}

theorem intersect_sets : (A ∩ B) = {x : ℝ | -1 < x ∧ x ≤ 5} := 
by 
  sorry

end intersect_sets_l848_84817


namespace range_of_x_plus_y_l848_84896

theorem range_of_x_plus_y (x y : ℝ) (h : x^2 + 2 * x * y - 1 = 0) : (x + y ≤ -1 ∨ x + y ≥ 1) :=
by
  sorry

end range_of_x_plus_y_l848_84896


namespace largest_whole_number_m_satisfies_inequality_l848_84806

theorem largest_whole_number_m_satisfies_inequality :
  ∃ m : ℕ, (1 / 4 + m / 6 : ℚ) < 3 / 2 ∧ ∀ n : ℕ, (1 / 4 + n / 6 : ℚ) < 3 / 2 → n ≤ 7 :=
by
  sorry

end largest_whole_number_m_satisfies_inequality_l848_84806


namespace correct_product_l848_84835

namespace SarahsMultiplication

theorem correct_product (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (hx' : ∃ (a b : ℕ), x = 10 * a + b ∧ b * 10 + a = x' ∧ 221 = x' * y) : (x * y = 527 ∨ x * y = 923) := by
  sorry

end SarahsMultiplication

end correct_product_l848_84835


namespace sum_of_numerator_and_denominator_l848_84871

def repeating_decimal_to_fraction_sum (x : ℚ) := 
  let numerator := 710
  let denominator := 99
  numerator + denominator

theorem sum_of_numerator_and_denominator : repeating_decimal_to_fraction_sum (71/10 + 7/990) = 809 := by
  sorry

end sum_of_numerator_and_denominator_l848_84871


namespace interval_of_a_l848_84802

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then Real.exp x + x^2 else Real.exp (-x) + x^2

theorem interval_of_a (a : ℝ) :
  f (-a) + f a ≤ 2 * f 1 → -1 ≤ a ∧ a ≤ 1 :=
sorry

end interval_of_a_l848_84802


namespace area_triangle_MNR_l848_84831

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

/-- Given the quadrilateral PQRS with the midpoints M and N of PQ and QR 
and specified lengths, prove the calculated area of triangle MNR. -/
theorem area_triangle_MNR : 
  let P : (ℝ × ℝ) := (0, 5)
  let Q : (ℝ × ℝ) := (10, 5)
  let R : (ℝ × ℝ) := (14, 0)
  let S : (ℝ × ℝ) := (7, 0)
  let M : (ℝ × ℝ) := (5, 5)  -- Midpoint of PQ
  let N : (ℝ × ℝ) := (12, 2.5) -- Midpoint of QR
  distance M.fst M.snd N.fst N.snd = 7.435 →
  ((5 - 0 : ℝ) / 2 = 2.5) →
  (1 / 2 * 7.435 * 2.5) = 9.294375 :=
by
  sorry

end area_triangle_MNR_l848_84831


namespace graph_passes_through_2_2_l848_84855

theorem graph_passes_through_2_2 (a : ℝ) (h : a > 0) (h_ne : a ≠ 1) : (2, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a^(x-2) + 1) } :=
sorry

end graph_passes_through_2_2_l848_84855


namespace sum_of_reciprocals_eq_six_l848_84882

theorem sum_of_reciprocals_eq_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  (1 / x + 1 / y) = 6 :=
by
  sorry

end sum_of_reciprocals_eq_six_l848_84882


namespace arithmetic_sequence_common_difference_l848_84810

theorem arithmetic_sequence_common_difference (d : ℚ) (a₁ : ℚ) (h : a₁ = -10)
  (h₁ : ∀ n ≥ 10, a₁ + (n - 1) * d > 0) :
  10 / 9 < d ∧ d ≤ 5 / 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l848_84810


namespace buns_per_student_correct_l848_84876

variables (packages_per_bun : Nat) (num_packages : Nat)
           (num_classes : Nat) (students_per_class : Nat)

def total_buns (packages_per_bun : Nat) (num_packages : Nat) : Nat :=
  packages_per_bun * num_packages

def total_students (num_classes : Nat) (students_per_class : Nat) : Nat :=
  num_classes * students_per_class

def buns_per_student (total_buns : Nat) (total_students : Nat) : Nat :=
  total_buns / total_students

theorem buns_per_student_correct :
  packages_per_bun = 8 →
  num_packages = 30 →
  num_classes = 4 →
  students_per_class = 30 →
  buns_per_student (total_buns packages_per_bun num_packages) 
                  (total_students num_classes students_per_class) = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end buns_per_student_correct_l848_84876


namespace larger_integer_exists_l848_84808

theorem larger_integer_exists (a b : ℤ) (h1 : a - b = 8) (h2 : a * b = 272) : a = 17 :=
sorry

end larger_integer_exists_l848_84808


namespace cube_increasing_on_reals_l848_84869

theorem cube_increasing_on_reals (a b : ℝ) (h : a < b) : a^3 < b^3 :=
sorry

end cube_increasing_on_reals_l848_84869


namespace total_cash_realized_correct_l848_84824

structure Stock where
  value : ℝ
  return_rate : ℝ
  brokerage_fee_rate : ℝ

def stockA : Stock := { value := 10000, return_rate := 0.14, brokerage_fee_rate := 0.0025 }
def stockB : Stock := { value := 20000, return_rate := 0.10, brokerage_fee_rate := 0.005 }
def stockC : Stock := { value := 30000, return_rate := 0.07, brokerage_fee_rate := 0.0075 }

def cash_realized (s : Stock) : ℝ :=
  let total_with_return := s.value * (1 + s.return_rate)
  total_with_return - (total_with_return * s.brokerage_fee_rate)

noncomputable def total_cash_realized : ℝ :=
  cash_realized stockA + cash_realized stockB + cash_realized stockC

theorem total_cash_realized_correct :
  total_cash_realized = 65120.75 :=
    sorry

end total_cash_realized_correct_l848_84824


namespace cost_of_milk_l848_84837

theorem cost_of_milk (x : ℝ) (h1 : 10 * 0.1 = 1) (h2 : 11 = 1 + x + 3 * x) : x = 2.5 :=
by 
  sorry

end cost_of_milk_l848_84837


namespace derivative_at_pi_over_2_l848_84821

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem derivative_at_pi_over_2 : 
  (deriv f (π / 2)) = Real.exp (π / 2) :=
by
  sorry

end derivative_at_pi_over_2_l848_84821


namespace periodic_function_property_l848_84826

theorem periodic_function_property
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_period : ∀ x, f (x + 2) = f x)
  (h_def1 : ∀ x, -1 ≤ x ∧ x < 0 → f x = a * x + 1)
  (h_def2 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = (b * x + 2) / (x + 1))
  (h_eq : f (1 / 2) = f (3 / 2)) :
  3 * a + 2 * b = -8 := by
  sorry

end periodic_function_property_l848_84826


namespace parity_of_f_and_h_l848_84819

-- Define function f
def f (x : ℝ) : ℝ := x^2

-- Define function h
def h (x : ℝ) : ℝ := x

-- Define even and odd function
def even_fun (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def odd_fun (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = - g x

-- Theorem statement
theorem parity_of_f_and_h :
  even_fun f ∧ odd_fun h :=
by {
  sorry
}

end parity_of_f_and_h_l848_84819


namespace maximum_value_of_N_l848_84894

-- Define J_k based on the conditions given
def J (k : ℕ) : ℕ := 10^(k+3) + 128

-- Define the number of factors of 2 in the prime factorization of J_k
def N (k : ℕ) : ℕ := Nat.factorization (J k) 2

-- The proposition to be proved
theorem maximum_value_of_N (k : ℕ) (hk : k > 0) : N 4 = 7 :=
by
  sorry

end maximum_value_of_N_l848_84894


namespace adam_final_amount_l848_84890

def initial_savings : ℝ := 1579.37
def money_received_monday : ℝ := 21.85
def money_received_tuesday : ℝ := 33.28
def money_spent_wednesday : ℝ := 87.41

def total_money_received : ℝ := money_received_monday + money_received_tuesday
def new_total_after_receiving : ℝ := initial_savings + total_money_received
def final_amount : ℝ := new_total_after_receiving - money_spent_wednesday

theorem adam_final_amount : final_amount = 1547.09 := by
  -- proof omitted
  sorry

end adam_final_amount_l848_84890


namespace ratio_of_pieces_l848_84830

def total_length (len: ℕ) := len = 35
def longer_piece (len: ℕ) := len = 20

theorem ratio_of_pieces (shorter len_shorter : ℕ) : 
  total_length 35 →
  longer_piece 20 →
  shorter = 35 - 20 →
  len_shorter = 15 →
  (20:ℚ) / (len_shorter:ℚ) = (4:ℚ) / (3:ℚ) :=
by
  sorry

end ratio_of_pieces_l848_84830


namespace scientific_notation_of_19400000000_l848_84820

theorem scientific_notation_of_19400000000 :
  ∃ a n, 1 ≤ |a| ∧ |a| < 10 ∧ (19400000000 : ℝ) = a * 10^n ∧ a = 1.94 ∧ n = 10 :=
by
  sorry

end scientific_notation_of_19400000000_l848_84820


namespace sum_of_m_and_n_l848_84833

theorem sum_of_m_and_n :
  ∃ m n : ℝ, (∀ x : ℝ, (x = 2 → m = 6 / x) ∧ (x = -2 → n = 6 / x)) ∧ (m + n = 0) :=
by
  let m := 6 / 2
  let n := 6 / (-2)
  use m, n
  simp
  sorry -- Proof omitted

end sum_of_m_and_n_l848_84833


namespace john_twice_sam_in_years_l848_84873

noncomputable def current_age_sam : ℕ := 9
noncomputable def current_age_john : ℕ := 27

theorem john_twice_sam_in_years (Y : ℕ) :
  (current_age_john + Y = 2 * (current_age_sam + Y)) → Y = 9 := 
by 
  sorry

end john_twice_sam_in_years_l848_84873


namespace find_integers_l848_84858

theorem find_integers (n : ℤ) : (6 ∣ (n - 4)) ∧ (10 ∣ (n - 8)) ↔ (n % 30 = 28) :=
by
  sorry

end find_integers_l848_84858


namespace rectangle_area_k_value_l848_84861

theorem rectangle_area_k_value (d : ℝ) (length width : ℝ) (h1 : 5 * width = 2 * length) (h2 : d^2 = length^2 + width^2) :
  ∃ (k : ℝ), A = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_k_value_l848_84861


namespace sam_won_total_matches_l848_84804

/-- Sam's first 100 matches and he won 50% of them -/
def first_100_matches : ℕ := 100

/-- Sam won 50% of his first 100 matches -/
def win_rate_first : ℕ := 50

/-- Sam's next 100 matches and he won 60% of them -/
def next_100_matches : ℕ := 100

/-- Sam won 60% of his next 100 matches -/
def win_rate_next : ℕ := 60

/-- The total number of matches Sam won -/
def total_matches_won (first_100_matches: ℕ) (win_rate_first: ℕ) (next_100_matches: ℕ) (win_rate_next: ℕ) : ℕ :=
  (first_100_matches * win_rate_first) / 100 + (next_100_matches * win_rate_next) / 100

theorem sam_won_total_matches :
  total_matches_won first_100_matches win_rate_first next_100_matches win_rate_next = 110 :=
by
  sorry

end sam_won_total_matches_l848_84804


namespace hyperbola_sum_l848_84841

theorem hyperbola_sum
  (h k a b : ℝ)
  (center : h = 3 ∧ k = 1)
  (vertex : ∃ (v : ℝ), (v = 4 ∧ h = 3 ∧ a = |k - v|))
  (focus : ∃ (f : ℝ), (f = 10 ∧ h = 3 ∧ (f - k) = 9 ∧ ∃ (c : ℝ), c = |k - f|))
  (relationship : ∀ (c : ℝ), c = 9 → c^2 = a^2 + b^2): 
  h + k + a + b = 7 + 6 * Real.sqrt 2 :=
by 
  sorry

end hyperbola_sum_l848_84841


namespace width_of_rect_prism_l848_84892

theorem width_of_rect_prism (w : ℝ) 
  (h : ℝ := 8) (l : ℝ := 5) (diagonal : ℝ := 17) 
  (h_diag : l^2 + w^2 + h^2 = diagonal^2) :
  w = 10 * Real.sqrt 2 :=
by
  sorry

end width_of_rect_prism_l848_84892


namespace pair_C_does_not_produce_roots_l848_84865

theorem pair_C_does_not_produce_roots (x : ℝ) :
  (x = 0 ∨ x = 2) ↔ (∃ x, y = x ∧ y = x - 2) = false :=
by
  sorry

end pair_C_does_not_produce_roots_l848_84865


namespace minimum_value_problem_l848_84801

theorem minimum_value_problem (x y z w : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) 
  (hxyz : x + y + z + w = 1) :
  (1 / (x + y) + 1 / (x + z) + 1 / (x + w) + 1 / (y + z) + 1 / (y + w) + 1 / (z + w)) ≥ 18 := 
sorry

end minimum_value_problem_l848_84801


namespace equation_solution_l848_84850

theorem equation_solution (x : ℤ) (h : 3 * x - 2 * x + x = 3 - 2 + 1) : x = 2 :=
by
  sorry

end equation_solution_l848_84850


namespace sum_of_solutions_eq_l848_84832

theorem sum_of_solutions_eq (x : ℝ) : (5 * x - 7) * (4 * x + 11) = 0 ->
  -((27 : ℝ) / (20 : ℝ)) =
  - ((5 * - 7) * (4 * x + 11)) / ((5 * x - 7) * 4) :=
by
  intro h
  sorry

end sum_of_solutions_eq_l848_84832


namespace kyle_practice_time_l848_84815

-- Definitions for the conditions
def weightlifting_time : ℕ := 20  -- in minutes
def running_time : ℕ := 2 * weightlifting_time  -- twice the weightlifting time
def total_running_and_weightlifting_time : ℕ := weightlifting_time + running_time  -- total time for running and weightlifting
def shooting_time : ℕ := total_running_and_weightlifting_time  -- because it's half the practice time

-- Total daily practice time, in minutes
def total_practice_time_minutes : ℕ := shooting_time + total_running_and_weightlifting_time

-- Total daily practice time, in hours
def total_practice_time_hours : ℕ := total_practice_time_minutes / 60

-- Theorem stating that Kyle practices for 2 hours every day given the conditions
theorem kyle_practice_time : total_practice_time_hours = 2 := by
  sorry

end kyle_practice_time_l848_84815


namespace nelly_bid_l848_84822

theorem nelly_bid (joe_bid sarah_bid : ℕ) (h1 : joe_bid = 160000) (h2 : sarah_bid = 50000)
  (h3 : ∀ nelly_bid, nelly_bid = 3 * joe_bid + 2000) (h4 : ∀ nelly_bid, nelly_bid = 4 * sarah_bid + 1500) :
  ∃ nelly_bid, nelly_bid = 482000 :=
by
  -- Skipping the proof with sorry
  sorry

end nelly_bid_l848_84822


namespace winning_candidate_votes_l848_84878

theorem winning_candidate_votes (T W : ℕ) (d1 d2 d3 : ℕ) 
  (hT : T = 963)
  (hd1 : d1 = 53) 
  (hd2 : d2 = 79) 
  (hd3 : d3 = 105) 
  (h_sum : T = W + (W - d1) + (W - d2) + (W - d3)) :
  W = 300 := 
by
  sorry

end winning_candidate_votes_l848_84878


namespace quadratic_two_distinct_real_roots_l848_84897

theorem quadratic_two_distinct_real_roots:
  ∃ (α β : ℝ), α ≠ β ∧ (∀ x : ℝ, x * (x - 2) = x - 2 ↔ x = α ∨ x = β) :=
by
  sorry

end quadratic_two_distinct_real_roots_l848_84897


namespace quadratic_inequality_solution_l848_84838

theorem quadratic_inequality_solution 
  (a : ℝ) 
  (h : ∀ x : ℝ, -1 < x ∧ x < a → -x^2 + 2 * a * x + a + 1 > a + 1) : -1 < a ∧ a ≤ -1/2 :=
sorry

end quadratic_inequality_solution_l848_84838


namespace find_line_eq_l848_84884

noncomputable def line_perpendicular (p : ℝ × ℝ) (a b c: ℝ) : Prop :=
  ∃ (m: ℝ) (k: ℝ), k ≠ 0 ∧ (b * m = -a) ∧ p = (m, (c - a * m) / b) ∧
  (∀ x y : ℝ, y = m * x + ((c - a * m) / b) ↔ b * y = -a * x - c)

theorem find_line_eq (p : ℝ × ℝ) (a b c : ℝ) (p_eq : p = (-3, 0)) (perpendicular_eq : a = 2 ∧ b = -1 ∧ c = 3) :
  ∃ (m k : ℝ), (k ≠ 0 ∧ (-1 * (b / a)) = m ∧ line_perpendicular p a b c) ∧ (b * m = -a) ∧ ((k = (-a * m) / b) ∧ (b * k * 0 - (-a * 3)) = c) := sorry

end find_line_eq_l848_84884


namespace product_of_consecutive_numbers_with_25_is_perfect_square_l848_84849

theorem product_of_consecutive_numbers_with_25_is_perfect_square (n : ℕ) : 
  ∃ k : ℕ, 100 * (n * (n + 1)) + 25 = k^2 := 
by
  -- Proof body omitted
  sorry

end product_of_consecutive_numbers_with_25_is_perfect_square_l848_84849


namespace degree_measure_of_subtracted_angle_l848_84845

def angle := 30

theorem degree_measure_of_subtracted_angle :
  let supplement := 180 - angle
  let complement_of_supplement := 90 - supplement
  let twice_complement := 2 * (90 - angle)
  twice_complement - complement_of_supplement = 180 :=
by
  sorry

end degree_measure_of_subtracted_angle_l848_84845


namespace second_train_length_is_correct_l848_84887

noncomputable def length_of_second_train (length_first_train : ℝ) (speed_first_train_kmph : ℝ) (speed_second_train_kmph : ℝ) (time_crossing_seconds : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train_kmph * (1000 / 3600)
  let speed_second_train_mps := speed_second_train_kmph * (1000 / 3600)
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_crossing_seconds
  total_distance - length_first_train

theorem second_train_length_is_correct : length_of_second_train 360 120 80 9 = 139.95 :=
by
  sorry

end second_train_length_is_correct_l848_84887


namespace inequality_gt_zero_l848_84888

theorem inequality_gt_zero (x y : ℝ) : x^2 + 2*y^2 + 2*x*y + 6*y + 10 > 0 :=
  sorry

end inequality_gt_zero_l848_84888


namespace valid_assignment_statement_l848_84883

theorem valid_assignment_statement (S a : ℕ) : (S = a + 1) ∧ ¬(a + 1 = S) ∧ ¬(S - 1 = a) ∧ ¬(S - a = 1) := by
  sorry

end valid_assignment_statement_l848_84883


namespace function_properties_l848_84853

-- Define the function f
def f (x p q : ℝ) : ℝ := x^3 + p * x^2 + 9 * q * x + p + q + 3

-- Stating the main theorem
theorem function_properties (p q : ℝ) :
  ( ∀ x : ℝ, f (-x) p q = -f x p q ) →
  (p = 0 ∧ q = -3 ∧ ∀ x : ℝ, f x 0 (-3) = x^3 - 27 * x ∧
   ( ∀ x ∈ Set.Icc (-1 : ℝ) 4, f x 0 (-3) ≤ 26 ) ∧
   ( ∀ x ∈ Set.Icc (-1 : ℝ) 4, f x 0 (-3) ≥ -54 )) := 
sorry

end function_properties_l848_84853


namespace perpendicular_lines_a_eq_0_or_neg1_l848_84863

theorem perpendicular_lines_a_eq_0_or_neg1 (a : ℝ) :
  (∃ (k₁ k₂: ℝ), (k₁ = a ∧ k₂ = (2 * a - 1)) ∧ ∃ (k₃ k₄: ℝ), (k₃ = 3 ∧ k₄ = a) ∧ k₁ * k₃ + k₂ * k₄ = 0) →
  (a = 0 ∨ a = -1) := 
sorry

end perpendicular_lines_a_eq_0_or_neg1_l848_84863


namespace symmetric_point_Q_l848_84834

-- Definitions based on conditions
def P : ℝ × ℝ := (-3, 2)
def symmetric_with_respect_to_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.fst, -point.snd)

-- Theorem stating that the coordinates of point Q (symmetric to P with respect to the x-axis) are (-3, -2)
theorem symmetric_point_Q : symmetric_with_respect_to_x_axis P = (-3, -2) := 
sorry

end symmetric_point_Q_l848_84834


namespace A_finishes_in_20_days_l848_84867

-- Define the rates and the work
variable (A B W : ℝ)

-- First condition: A and B together can finish the work in 12 days
axiom together_rate : (A + B) * 12 = W

-- Second condition: B alone can finish the work in 30.000000000000007 days
axiom B_rate : B * 30.000000000000007 = W

-- Prove that A alone can finish the work in 20 days
theorem A_finishes_in_20_days : (1 / A) = 20 :=
by 
  sorry

end A_finishes_in_20_days_l848_84867


namespace baskets_containing_neither_l848_84803

-- Definitions representing the conditions
def total_baskets : ℕ := 15
def baskets_with_apples : ℕ := 10
def baskets_with_oranges : ℕ := 8
def baskets_with_both : ℕ := 5

-- Theorem statement to prove the number of baskets containing neither apples nor oranges
theorem baskets_containing_neither : total_baskets - (baskets_with_apples + baskets_with_oranges - baskets_with_both) = 2 :=
by
  sorry

end baskets_containing_neither_l848_84803


namespace largest_share_received_l848_84839

theorem largest_share_received (total_profit : ℝ) (ratios : List ℝ) (h_ratios : ratios = [1, 2, 2, 3, 4, 5]) 
  (h_profit : total_profit = 51000) : 
  let parts := ratios.sum 
  let part_value := total_profit / parts
  let largest_share := 5 * part_value 
  largest_share = 15000 := 
by 
  sorry

end largest_share_received_l848_84839


namespace remaining_number_l848_84811

theorem remaining_number (S : Finset ℕ) (hS : S = Finset.range 51) :
  ∃ n ∈ S, n % 2 = 0 := 
sorry

end remaining_number_l848_84811


namespace coordinate_inequality_l848_84814

theorem coordinate_inequality (x y : ℝ) :
  (xy > 0 → (x - 2)^2 + (y + 1)^2 < 5) ∧ (xy < 0 → (x - 2)^2 + (y + 1)^2 > 5) :=
by
  sorry

end coordinate_inequality_l848_84814


namespace solution_set_for_inequality_l848_84859

theorem solution_set_for_inequality 
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_mono_dec : ∀ x y, 0 < x → x < y → f y ≤ f x)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_for_inequality_l848_84859


namespace solve_quadratic_eq1_solve_quadratic_eq2_l848_84854

-- Define the statement for the first problem
theorem solve_quadratic_eq1 (x : ℝ) : x^2 - 49 = 0 → x = 7 ∨ x = -7 :=
by
  sorry

-- Define the statement for the second problem
theorem solve_quadratic_eq2 (x : ℝ) : 2 * (x + 1)^2 - 49 = 1 → x = 4 ∨ x = -6 :=
by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l848_84854


namespace acute_triangle_area_relation_l848_84809

open Real

variables (A B C R : ℝ)
variables (acute_triangle : Prop)
variables (S p_star : ℝ)

-- Conditions
axiom acute_triangle_condition : acute_triangle
axiom area_formula : S = (R^2 / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C))
axiom semiperimeter_formula : p_star = (R / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C))

-- Theorem to prove
theorem acute_triangle_area_relation (h : acute_triangle) : S = p_star * R := 
by {
  sorry 
}

end acute_triangle_area_relation_l848_84809


namespace peter_ate_7_over_48_l848_84857

-- Define the initial conditions
def total_slices : ℕ := 16
def slices_peter_ate : ℕ := 2
def shared_slice : ℚ := 1/3

-- Define the first part of the problem
def fraction_peter_ate_alone : ℚ := slices_peter_ate / total_slices

-- Define the fraction Peter ate from sharing one slice
def fraction_peter_ate_shared : ℚ := shared_slice / total_slices

-- Define the total fraction Peter ate
def total_fraction_peter_ate : ℚ := fraction_peter_ate_alone + fraction_peter_ate_shared

-- Create the theorem to be proved (statement only)
theorem peter_ate_7_over_48 :
  total_fraction_peter_ate = 7 / 48 :=
by
  sorry

end peter_ate_7_over_48_l848_84857


namespace like_terms_l848_84825

theorem like_terms (x y : ℕ) (h1 : x + 1 = 2) (h2 : x + y = 2) : x = 1 ∧ y = 1 :=
by
  sorry

end like_terms_l848_84825


namespace units_digit_of_8_pow_2022_l848_84856

theorem units_digit_of_8_pow_2022 : (8 ^ 2022) % 10 = 4 := 
by
  -- We here would provide the proof of this theorem
  sorry

end units_digit_of_8_pow_2022_l848_84856


namespace find_positive_integer_tuples_l848_84827

theorem find_positive_integer_tuples
  (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hz_prime : Prime z) :
  z ^ x = y ^ 3 + 1 →
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 2 ∧ y = 2 ∧ z = 3) :=
by
  sorry

end find_positive_integer_tuples_l848_84827


namespace base7_subtraction_correct_l848_84885

-- Define a function converting base 7 number to base 10
def base7_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  let n3 := n2 / 10
  let d3 := n3 % 10
  d3 * 7^3 + d2 * 7^2 + d1 * 7^1 + d0 * 7^0

-- Define the numbers in base 7
def a : Nat := 2456
def b : Nat := 1234

-- Define the expected result in base 7
def result_base7 : Nat := 1222

-- State the theorem: The difference of a and b in base 7 should equal result_base7
theorem base7_subtraction_correct :
  let diff_base10 := (base7_to_base10 a) - (base7_to_base10 b)
  let result_base10 := base7_to_base10 result_base7
  diff_base10 = result_base10 :=
by
  sorry

end base7_subtraction_correct_l848_84885


namespace max_f_5_value_l848_84818

noncomputable def f (x : ℝ) : ℝ := x ^ 2 + 2 * x

noncomputable def f_1 (x : ℝ) : ℝ := f x
noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0       => x -- Not used, as n starts from 1
  | (n + 1) => f (f_n n x)

noncomputable def max_f_5 : ℝ := 3 ^ 32 - 1

theorem max_f_5_value : ∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f_n 5 x ≤ max_f_5 :=
by
  intro x hx
  have := hx
  -- The detailed proof would go here,
  -- but for the statement, we end with sorry.
  sorry

end max_f_5_value_l848_84818


namespace minimum_number_of_tiles_l848_84880

-- Define the measurement conversion and area calculations.
def tile_width := 2
def tile_length := 6
def region_width_feet := 3
def region_length_feet := 4

-- Convert feet to inches.
def region_width_inches := region_width_feet * 12
def region_length_inches := region_length_feet * 12

-- Calculate areas.
def tile_area := tile_width * tile_length
def region_area := region_width_inches * region_length_inches

-- Lean 4 statement to prove the minimum number of tiles required.
theorem minimum_number_of_tiles : region_area / tile_area = 144 := by
  sorry

end minimum_number_of_tiles_l848_84880


namespace solve_for_C_days_l848_84836

noncomputable def A_work_rate : ℚ := 1 / 20
noncomputable def B_work_rate : ℚ := 1 / 15
noncomputable def C_work_rate : ℚ := 1 / 50
noncomputable def total_work_done_by_A_B : ℚ := 6 * (A_work_rate + B_work_rate)
noncomputable def remaining_work : ℚ := 1 - total_work_done_by_A_B

theorem solve_for_C_days : ∃ d : ℚ, d * C_work_rate = remaining_work ∧ d = 15 :=
by
  use 15
  simp [C_work_rate, remaining_work, total_work_done_by_A_B, A_work_rate, B_work_rate]
  sorry

end solve_for_C_days_l848_84836


namespace three_lines_form_triangle_l848_84860

/-- Theorem to prove that for three lines x + y = 0, x - y = 0, and x + ay = 3 to form a triangle, the value of a cannot be ±1. -/
theorem three_lines_form_triangle (a : ℝ) : ¬ (a = 1 ∨ a = -1) :=
sorry

end three_lines_form_triangle_l848_84860


namespace negation_of_existence_l848_84893

theorem negation_of_existence (x : ℝ) (hx : 0 < x) : ¬ (∃ x_0 : ℝ, 0 < x_0 ∧ Real.log x_0 = x_0 - 1) 
  → ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by sorry

end negation_of_existence_l848_84893


namespace algebraic_expression_constant_l848_84807

theorem algebraic_expression_constant (x : ℝ) : x * (x - 6) - (3 - x) ^ 2 = -9 :=
sorry

end algebraic_expression_constant_l848_84807


namespace no_solution_l848_84877

def is_digit (B : ℕ) : Prop := B < 10

def divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

def satisfies_conditions (B : ℕ) : Prop :=
  is_digit B ∧
  divisible_by (12345670 + B) 2 ∧
  divisible_by (12345670 + B) 5 ∧
  divisible_by (12345670 + B) 11

theorem no_solution (B : ℕ) : ¬ satisfies_conditions B :=
sorry

end no_solution_l848_84877


namespace determine_n_l848_84886

-- All the terms used in the conditions
variables (S C M : ℝ)
variables (n : ℝ)

-- Define the conditions as hypotheses
def condition1 := M = 1 / 3 * S
def condition2 := M = 1 / n * C

-- The main theorem statement
theorem determine_n (S C M : ℝ) (n : ℝ) (h1 : condition1 S M) (h2 : condition2 M n C) : n = 2 :=
by sorry

end determine_n_l848_84886


namespace merchant_loss_l848_84840

theorem merchant_loss
  (sp : ℝ)
  (profit_percent: ℝ)
  (loss_percent:  ℝ)
  (sp1 : ℝ)
  (sp2 : ℝ)
  (cp1 cp2 : ℝ)
  (net_loss : ℝ) :
  
  sp = 990 → 
  profit_percent = 0.1 → 
  loss_percent = 0.1 →
  sp1 = sp → 
  sp2 = sp → 
  cp1 = sp1 / (1 + profit_percent) →
  cp2 = sp2 / (1 - loss_percent) →
  net_loss = (cp2 - sp2) - (sp1 - cp1) →
  net_loss = 20 :=
by 
  intros _ _ _ _ _ _ _ _ 
  -- placeholders for intros to bind variables
  sorry

end merchant_loss_l848_84840


namespace find_lesser_number_l848_84870

theorem find_lesser_number (x y : ℕ) (h₁ : x + y = 60) (h₂ : x - y = 10) : y = 25 := by
  sorry

end find_lesser_number_l848_84870


namespace complex_number_properties_l848_84862

open Complex

noncomputable def z : ℂ := (1 - I) / I

theorem complex_number_properties :
  z ^ 2 = 2 * I ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_number_properties_l848_84862


namespace man_is_older_l848_84823

-- Define present age of the son
def son_age : ℕ := 26

-- Define present age of the man (father)
axiom man_age : ℕ

-- Condition: in two years, the man's age will be twice the age of his son
axiom age_condition : man_age + 2 = 2 * (son_age + 2)

-- Prove that the man is 28 years older than his son
theorem man_is_older : man_age - son_age = 28 := sorry

end man_is_older_l848_84823


namespace xy_zero_l848_84851

theorem xy_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = 0 :=
by
  sorry

end xy_zero_l848_84851


namespace min_odd_integers_l848_84898

theorem min_odd_integers :
  ∀ (a b c d e f g h : ℤ),
  a + b + c = 30 →
  a + b + c + d + e + f = 58 →
  a + b + c + d + e + f + g + h = 73 →
  ∃ (odd_count : ℕ), odd_count = 1 :=
by
  sorry

end min_odd_integers_l848_84898


namespace parameterization_of_line_l848_84805

theorem parameterization_of_line : 
  ∀ t : ℝ, ∃ f : ℝ → ℝ, (f t, 20 * t - 14) ∈ { p : ℝ × ℝ | ∃ (x y : ℝ), y = 2 * x - 40 ∧ p = (x, y) } ∧ f t = 10 * t + 13 :=
by
  sorry

end parameterization_of_line_l848_84805


namespace work_completion_l848_84879

theorem work_completion (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a + b = 1/10) (h2 : a = 1/14) : a + b = 1/10 := 
by {
  sorry
}

end work_completion_l848_84879


namespace cuberoot_3375_sum_l848_84866

theorem cuberoot_3375_sum (a b : ℕ) (h : 3375 = 3^3 * 5^3) (h1 : a = 15) (h2 : b = 1) : a + b = 16 := by
  sorry

end cuberoot_3375_sum_l848_84866


namespace prob1_prob2_prob3_prob4_prob5_l848_84899

theorem prob1 : (1 - 27 + (-32) + (-8) + 27) = -40 := sorry

theorem prob2 : (2 * -5 + abs (-3)) = -2 := sorry

theorem prob3 (x y : Int) (h₁ : -x = 3) (h₂ : abs y = 5) : x + y = 2 ∨ x + y = -8 := sorry

theorem prob4 : ((-1 : Int) * (3 / 2) + (5 / 4) + (-5 / 2) - (-13 / 4) - (5 / 4)) = -3 / 4 := sorry

theorem prob5 (a b : Int) (h : abs (a - 4) + abs (b + 5) = 0) : a - b = 9 := sorry

end prob1_prob2_prob3_prob4_prob5_l848_84899


namespace totalCostOfAllPuppies_l848_84891

noncomputable def goldenRetrieverCost : ℕ :=
  let numberOfGoldenRetrievers := 3
  let puppiesPerGoldenRetriever := 4
  let shotsPerPuppy := 2
  let costPerShot := 5
  let vitaminCostPerMonth := 12
  let monthsOfSupplements := 6
  numberOfGoldenRetrievers * puppiesPerGoldenRetriever *
  (shotsPerPuppy * costPerShot + vitaminCostPerMonth * monthsOfSupplements)

noncomputable def germanShepherdCost : ℕ :=
  let numberOfGermanShepherds := 2
  let puppiesPerGermanShepherd := 5
  let shotsPerPuppy := 3
  let costPerShot := 8
  let microchipCost := 25
  let toyCost := 15
  numberOfGermanShepherds * puppiesPerGermanShepherd *
  (shotsPerPuppy * costPerShot + microchipCost + toyCost)

noncomputable def bulldogCost : ℕ :=
  let numberOfBulldogs := 4
  let puppiesPerBulldog := 3
  let shotsPerPuppy := 4
  let costPerShot := 10
  let collarCost := 20
  let chewToyCost := 18
  numberOfBulldogs * puppiesPerBulldog *
  (shotsPerPuppy * costPerShot + collarCost + chewToyCost)

theorem totalCostOfAllPuppies : goldenRetrieverCost + germanShepherdCost + bulldogCost = 2560 :=
by
  sorry

end totalCostOfAllPuppies_l848_84891


namespace XiaoMing_reading_problem_l848_84812

theorem XiaoMing_reading_problem :
  ∀ (total_pages days first_days first_rate remaining_rate : ℕ),
    total_pages = 72 →
    days = 10 →
    first_days = 2 →
    first_rate = 5 →
    (first_days * first_rate) + ((days - first_days) * remaining_rate) ≥ total_pages →
    remaining_rate ≥ 8 :=
by
  intros total_pages days first_days first_rate remaining_rate
  intro h1 h2 h3 h4 h5
  sorry

end XiaoMing_reading_problem_l848_84812


namespace perpendicular_condition_sufficient_but_not_necessary_l848_84813

theorem perpendicular_condition_sufficient_but_not_necessary (a : ℝ) :
  (a = -2) → ((∀ x y : ℝ, ax + (a + 1) * y + 1 = 0 → x + a * y + 2 = 0 ∧ (∃ t : ℝ, t ≠ 0 ∧ x = -t / (a + 1) ∧ y = (t / a))) →
  ¬ (a = -2) ∨ (a + 1 ≠ 0 ∧ ∃ k1 k2 : ℝ, k1 * k2 = -1 ∧ k1 = -a / (a + 1) ∧ k2 = -1 / a)) :=
by
  sorry

end perpendicular_condition_sufficient_but_not_necessary_l848_84813


namespace mat_pow_four_eq_l848_84844

open Matrix

def mat : Matrix (Fin 2) (Fin 2) ℤ :=
  !![2, -2; 1, 1]

def mat_fourth_power : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-14, -6; 3, -17]

theorem mat_pow_four_eq :
  mat ^ 4 = mat_fourth_power :=
by
  sorry

end mat_pow_four_eq_l848_84844


namespace combined_total_time_l848_84872

theorem combined_total_time
  (Katherine_time : Real := 20)
  (Naomi_time : Real := Katherine_time * (1 + 1 / 4))
  (Lucas_time : Real := Katherine_time * (1 + 1 / 3))
  (Isabella_time : Real := Katherine_time * (1 + 1 / 2))
  (Naomi_total : Real := Naomi_time * 10)
  (Lucas_total : Real := Lucas_time * 10)
  (Isabella_total : Real := Isabella_time * 10) :
  Naomi_total + Lucas_total + Isabella_total = 816.7 := sorry

end combined_total_time_l848_84872


namespace base_7_digits_956_l848_84816

theorem base_7_digits_956 : ∃ n : ℕ, ∀ k : ℕ, 956 < 7^k → n = k ∧ 956 ≥ 7^(k-1) := sorry

end base_7_digits_956_l848_84816


namespace trapezoid_base_length_sets_l848_84842

open Nat

theorem trapezoid_base_length_sets :
  ∃ (sets : Finset (ℕ × ℕ)), sets.card = 5 ∧ 
    (∀ p ∈ sets, ∃ (b1 b2 : ℕ), b1 = 10 * p.1 ∧ b2 = 10 * p.2 ∧ b1 + b2 = 90) :=
by
  sorry

end trapezoid_base_length_sets_l848_84842
