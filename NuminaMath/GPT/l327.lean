import Mathlib

namespace sin_theta_value_l327_32792

theorem sin_theta_value 
  (θ : ℝ)
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) :
  Real.sin θ = 3/5 :=
sorry

end sin_theta_value_l327_32792


namespace kekai_ratio_l327_32765

/-
Kekai sells 5 shirts at $1 each,
5 pairs of pants at $3 each,
and he has $10 left after giving some money to his parents.
Our goal is to prove the ratio of the money Kekai gives to his parents
to the total money he earns from selling his clothes is 1:2.
-/

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def shirt_price : ℕ := 1
def pants_price : ℕ := 3
def money_left : ℕ := 10

def total_earnings : ℕ := (shirts_sold * shirt_price) + (pants_sold * pants_price)
def money_given_to_parents : ℕ := total_earnings - money_left
def ratio (a b : ℕ) := (a / Nat.gcd a b, b / Nat.gcd a b)

theorem kekai_ratio : ratio money_given_to_parents total_earnings = (1, 2) :=
  by
    sorry

end kekai_ratio_l327_32765


namespace movies_left_to_watch_l327_32759

theorem movies_left_to_watch (total_movies watched_movies : Nat) (h_total : total_movies = 12) (h_watched : watched_movies = 6) : total_movies - watched_movies = 6 :=
by
  sorry

end movies_left_to_watch_l327_32759


namespace range_of_a_l327_32736

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a_seq n = a + n - 1)
  (h2 : ∀ n : ℕ, b n = (1 + a_seq n) / a_seq n)
  (h3 : ∀ n : ℕ, n > 0 → b n ≤ b 5) :
  -4 < a ∧ a < -3 :=
by
  sorry

end range_of_a_l327_32736


namespace smallest_missing_digit_units_place_cube_l327_32713

theorem smallest_missing_digit_units_place_cube :
  ∀ d : Fin 10, ∃ n : ℕ, (n ^ 3) % 10 = d :=
by
  sorry

end smallest_missing_digit_units_place_cube_l327_32713


namespace find_original_prices_and_discount_l327_32734

theorem find_original_prices_and_discount :
  ∃ x y a : ℝ,
  (6 * x + 5 * y = 1140) ∧
  (3 * x + 7 * y = 1110) ∧
  (((9 * x + 8 * y) - 1062) / (9 * x + 8 * y) = a) ∧
  x = 90 ∧
  y = 120 ∧
  a = 0.4 :=
by
  sorry

end find_original_prices_and_discount_l327_32734


namespace ellipse_equation_l327_32728

theorem ellipse_equation (a b c : ℝ) (h0 : a > b) (h1 : b > 0) (h2 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h3 : dist (3, y) (5 - 5 / 2, 0) = 6.5) (h4 : dist (3, y) (5 + 5 / 2, 0) = 3.5) : 
  ( ∀ x y, (x^2 / 25) + (y^2 / (75 / 4)) = 1 ) :=
sorry

end ellipse_equation_l327_32728


namespace probability_two_consecutive_pairs_of_four_dice_correct_l327_32787

open Classical

noncomputable def probability_two_consecutive_pairs_of_four_dice : ℚ :=
  let total_outcomes := 6^4
  let favorable_outcomes := 48
  favorable_outcomes / total_outcomes

theorem probability_two_consecutive_pairs_of_four_dice_correct :
  probability_two_consecutive_pairs_of_four_dice = 1 / 27 := 
by
  sorry

end probability_two_consecutive_pairs_of_four_dice_correct_l327_32787


namespace inequality_pow_l327_32786

variable {n : ℕ}

theorem inequality_pow (hn : n > 0) : 
  (3:ℝ) / 2 ≤ (1 + (1:ℝ) / (2 * n)) ^ n ∧ (1 + (1:ℝ) / (2 * n)) ^ n < 2 := 
sorry

end inequality_pow_l327_32786


namespace abc_sum_l327_32788

theorem abc_sum (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a + b + c = 13 * Real.sqrt 6 := 
sorry

end abc_sum_l327_32788


namespace jackson_vacuuming_time_l327_32726

-- Definitions based on the conditions
def hourly_wage : ℕ := 5
def washing_dishes_time : ℝ := 0.5
def cleaning_bathroom_time : ℝ := 3 * washing_dishes_time
def total_earnings : ℝ := 30

-- The total time spent on chores
def total_chore_time (V : ℝ) : ℝ :=
  2 * V + washing_dishes_time + cleaning_bathroom_time

-- The main theorem that needs to be proven
theorem jackson_vacuuming_time :
  ∃ V : ℝ, hourly_wage * total_chore_time V = total_earnings ∧ V = 2 :=
by
  sorry

end jackson_vacuuming_time_l327_32726


namespace initial_bucket_capacity_l327_32707

theorem initial_bucket_capacity (x : ℕ) (h1 : x - 3 = 2) : x = 5 := sorry

end initial_bucket_capacity_l327_32707


namespace mileage_interval_l327_32730

-- Define the distances driven each day
def d1 : ℕ := 135
def d2 : ℕ := 135 + 124
def d3 : ℕ := 159
def d4 : ℕ := 189

-- Define the total distance driven
def total_distance : ℕ := d1 + d2 + d3 + d4

-- Define the number of intervals (charges)
def number_of_intervals : ℕ := 6

-- Define the expected mileage interval for charging
def expected_interval : ℕ := 124

-- The theorem to prove that the mileage interval is approximately 124 miles
theorem mileage_interval : total_distance / number_of_intervals = expected_interval := by
  sorry

end mileage_interval_l327_32730


namespace find_complex_number_l327_32774

namespace ComplexProof

open Complex

def satisfies_conditions (z : ℂ) : Prop :=
  (z^2).im = 0 ∧ abs (z - I) = 1

theorem find_complex_number (z : ℂ) (h : satisfies_conditions z) : z = 0 ∨ z = 2 * I :=
sorry

end ComplexProof

end find_complex_number_l327_32774


namespace pretzels_count_l327_32758

-- Define the number of pretzels
def pretzels : ℕ := 64

-- Given conditions
def goldfish (P : ℕ) : ℕ := 4 * P
def suckers : ℕ := 32
def kids : ℕ := 16
def items_per_kid : ℕ := 22
def total_items (P : ℕ) : ℕ := P + goldfish P + suckers

-- The theorem to prove
theorem pretzels_count : total_items pretzels = kids * items_per_kid := by
  sorry

end pretzels_count_l327_32758


namespace isosceles_triangle_base_length_l327_32741

theorem isosceles_triangle_base_length (s a b : ℕ) (h1 : 3 * s = 45)
  (h2 : 2 * a + b = 40) (h3 : a = s) : b = 10 :=
by
  sorry

end isosceles_triangle_base_length_l327_32741


namespace standard_equation_of_parabola_l327_32703

theorem standard_equation_of_parabola (x : ℝ) (y : ℝ) (directrix : ℝ) (eq_directrix : directrix = 1) :
  y^2 = -4 * x :=
sorry

end standard_equation_of_parabola_l327_32703


namespace max_digit_sum_in_24_hour_format_l327_32716

theorem max_digit_sum_in_24_hour_format : 
  ∃ t : ℕ × ℕ, (0 ≤ t.fst ∧ t.fst < 24 ∧ 0 ≤ t.snd ∧ t.snd < 60 ∧ (t.fst / 10 + t.fst % 10 + t.snd / 10 + t.snd % 10 = 24)) :=
sorry

end max_digit_sum_in_24_hour_format_l327_32716


namespace gcd_154_308_462_l327_32717

theorem gcd_154_308_462 : Nat.gcd (Nat.gcd 154 308) 462 = 154 := by
  sorry

end gcd_154_308_462_l327_32717


namespace option_c_correct_l327_32760

theorem option_c_correct (α x1 x2 : ℝ) (hα1 : 0 < α) (hα2 : α < π) (hx1 : 0 < x1) (hx2 : x1 < x2) : 
  (x2 / x1) ^ Real.sin α > 1 :=
by
  sorry

end option_c_correct_l327_32760


namespace total_shingles_for_all_roofs_l327_32799

def roof_A_length : ℕ := 20
def roof_A_width : ℕ := 40
def roof_A_shingles_per_sqft : ℕ := 8

def roof_B_length : ℕ := 25
def roof_B_width : ℕ := 35
def roof_B_shingles_per_sqft : ℕ := 10

def roof_C_length : ℕ := 30
def roof_C_width : ℕ := 30
def roof_C_shingles_per_sqft : ℕ := 12

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def total_area (length : ℕ) (width : ℕ) : ℕ :=
  2 * area length width

def total_shingles_needed (length : ℕ) (width : ℕ) (shingles_per_sqft : ℕ) : ℕ :=
  total_area length width * shingles_per_sqft

theorem total_shingles_for_all_roofs :
  total_shingles_needed roof_A_length roof_A_width roof_A_shingles_per_sqft +
  total_shingles_needed roof_B_length roof_B_width roof_B_shingles_per_sqft +
  total_shingles_needed roof_C_length roof_C_width roof_C_shingles_per_sqft = 51900 :=
by
  sorry

end total_shingles_for_all_roofs_l327_32799


namespace polynomial_coeff_sums_l327_32714

theorem polynomial_coeff_sums (g h : ℤ) (d : ℤ) :
  (7 * d^2 - 3 * d + g) * (3 * d^2 + h * d - 8) = 21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d - 16 →
  g + h = -3 :=
by
  sorry

end polynomial_coeff_sums_l327_32714


namespace percent_increase_l327_32797

theorem percent_increase (x : ℝ) (h : (1 / 2) * x = 1) : ((x - (1 / 2)) / (1 / 2)) * 100 = 300 := by
  sorry

end percent_increase_l327_32797


namespace total_books_from_library_l327_32768

def initialBooks : ℕ := 54
def additionalBooks : ℕ := 23

theorem total_books_from_library : initialBooks + additionalBooks = 77 := by
  sorry

end total_books_from_library_l327_32768


namespace arrangements_correctness_l327_32763

noncomputable def arrangements_of_groups (total mountaineers : ℕ) (familiar_with_route : ℕ) (required_in_each_group : ℕ) : ℕ :=
  sorry

theorem arrangements_correctness :
  arrangements_of_groups 10 4 2 = 120 :=
sorry

end arrangements_correctness_l327_32763


namespace cakes_difference_l327_32784

-- Definitions of the given conditions
def cakes_sold : ℕ := 78
def cakes_bought : ℕ := 31

-- The theorem to prove
theorem cakes_difference : cakes_sold - cakes_bought = 47 :=
by sorry

end cakes_difference_l327_32784


namespace kite_cost_l327_32709

variable (initial_amount : ℕ) (cost_frisbee : ℕ) (amount_left : ℕ)

theorem kite_cost (initial_amount : ℕ) (cost_frisbee : ℕ) (amount_left : ℕ) (h_initial_amount : initial_amount = 78) (h_cost_frisbee : cost_frisbee = 9) (h_amount_left : amount_left = 61) : 
  initial_amount - amount_left - cost_frisbee = 8 :=
by
  -- Proof can be completed here
  sorry

end kite_cost_l327_32709


namespace fourth_term_correct_l327_32777

def fourth_term_sequence : Nat :=
  4^0 + 4^1 + 4^2 + 4^3

theorem fourth_term_correct : fourth_term_sequence = 85 :=
by
  sorry

end fourth_term_correct_l327_32777


namespace no_three_nat_sum_pair_is_pow_of_three_l327_32700

theorem no_three_nat_sum_pair_is_pow_of_three :
  ¬ ∃ (a b c : ℕ) (m n p : ℕ), a + b = 3 ^ m ∧ b + c = 3 ^ n ∧ c + a = 3 ^ p := 
by 
  sorry

end no_three_nat_sum_pair_is_pow_of_three_l327_32700


namespace population_of_missing_village_eq_945_l327_32733

theorem population_of_missing_village_eq_945
  (pop1 pop2 pop3 pop4 pop5 pop6 : ℕ)
  (avg_pop total_population missing_population : ℕ)
  (h1 : pop1 = 803)
  (h2 : pop2 = 900)
  (h3 : pop3 = 1100)
  (h4 : pop4 = 1023)
  (h5 : pop5 = 980)
  (h6 : pop6 = 1249)
  (h_avg : avg_pop = 1000)
  (h_total_population : total_population = avg_pop * 7)
  (h_missing_population : missing_population = total_population - (pop1 + pop2 + pop3 + pop4 + pop5 + pop6)) :
  missing_population = 945 :=
by {
  -- Here would go the proof steps if needed
  sorry 
}

end population_of_missing_village_eq_945_l327_32733


namespace find_b_l327_32781

theorem find_b (a b c : ℝ) (k₁ k₂ k₃ : ℤ) :
  (a + b) / 2 = 40 ∧
  (b + c) / 2 = 43 ∧
  (a + c) / 2 = 44 ∧
  a + b = 5 * k₁ ∧
  b + c = 5 * k₂ ∧
  a + c = 5 * k₃
  → b = 40 :=
by {
  sorry
}

end find_b_l327_32781


namespace total_books_l327_32767

-- Define the conditions
def books_per_shelf : ℕ := 9
def mystery_shelves : ℕ := 6
def picture_shelves : ℕ := 2

-- The proof problem statement
theorem total_books : 
  (mystery_shelves * books_per_shelf) + 
  (picture_shelves * books_per_shelf) = 72 := 
sorry

end total_books_l327_32767


namespace part1_part2_l327_32737

-- Define the solution set M for the inequality
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- Define the problem conditions
variables {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M)

-- First part: Prove that |(1/3)a + (1/6)b| < 1/4
theorem part1 : |(1/3 : ℝ) * a + (1/6 : ℝ) * b| < 1/4 :=
sorry

-- Second part: Prove that |1 - 4 * a * b| > 2 * |a - b|
theorem part2 : |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end part1_part2_l327_32737


namespace lines_intersect_l327_32718

theorem lines_intersect (m : ℝ) : ∃ (x y : ℝ), 3 * x + 2 * y + m = 0 ∧ (m^2 + 1) * x - 3 * y - 3 * m = 0 := 
by {
  sorry
}

end lines_intersect_l327_32718


namespace rachel_homework_difference_l327_32790

def pages_of_math_homework : Nat := 5
def pages_of_reading_homework : Nat := 2

theorem rachel_homework_difference : 
  pages_of_math_homework - pages_of_reading_homework = 3 :=
sorry

end rachel_homework_difference_l327_32790


namespace distance_to_lightning_l327_32780

noncomputable def distance_from_lightning (time_delay : ℕ) (speed_of_sound : ℕ) (feet_per_mile : ℕ) : ℚ :=
  (time_delay * speed_of_sound : ℕ) / feet_per_mile

theorem distance_to_lightning (time_delay : ℕ) (speed_of_sound : ℕ) (feet_per_mile : ℕ) :
  time_delay = 12 → speed_of_sound = 1120 → feet_per_mile = 5280 → distance_from_lightning time_delay speed_of_sound feet_per_mile = 2.5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end distance_to_lightning_l327_32780


namespace figure_50_squares_l327_32744

open Nat

noncomputable def g (n : ℕ) : ℕ := 2 * n ^ 2 + 5 * n + 2

theorem figure_50_squares : g 50 = 5252 :=
by
  sorry

end figure_50_squares_l327_32744


namespace inverse_function_properties_l327_32706

theorem inverse_function_properties {f : ℝ → ℝ} 
  (h_monotonic_decreasing : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 3 → f x2 < f x1)
  (h_range : ∀ y : ℝ, 4 ≤ y ∧ y ≤ 7 ↔ ∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ y = f x)
  (h_inverse_exists : ∃ g : ℝ → ℝ, ∀ x : ℝ, f (g x) = x ∧ g (f x) = x) :
  ∃ g : ℝ → ℝ, (∀ y1 y2 : ℝ, 4 ≤ y1 ∧ y1 < y2 ∧ y2 ≤ 7 → g y2 < g y1) ∧ (∀ y : ℝ, 4 ≤ y ∧ y ≤ 7 → g y ≤ 3) :=
sorry

end inverse_function_properties_l327_32706


namespace ratio_female_to_total_l327_32754

theorem ratio_female_to_total:
  ∃ (F : ℕ), (6 + 7 * F - 9 = (6 + 7 * F) - 9) ∧ 
             (7 * F - 9 = 67 / 100 * ((6 + 7 * F) - 9)) → 
             F = 3 ∧ 6 = 6 → 
             1 / F = 2 / 6 :=
by sorry

end ratio_female_to_total_l327_32754


namespace find_minimum_r_l327_32776

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_minimum_r (r : ℕ) (h_pos : r > 0) (h_perfect : is_perfect_square (4^3 + 4^r + 4^4)) : r = 4 :=
sorry

end find_minimum_r_l327_32776


namespace range_of_m_l327_32743

/-- Define the domain set A where the function f(x) = 1 / sqrt(4 + 3x - x^2) is defined. -/
def A : Set ℝ := {x | -1 < x ∧ x < 4}

/-- Define the range set B where the function g(x) = - x^2 - 2x + 2, with x in [-1, 1], is defined. -/
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

/-- Define the set C in terms of m. -/
def C (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2}

/-- Prove the range of the real number m such that C ∩ (A ∪ B) = C. -/
theorem range_of_m : {m : ℝ | C m ⊆ A ∪ B} = {m | -1 ≤ m ∧ m < 2} :=
by
  sorry

end range_of_m_l327_32743


namespace logarithmic_AMGM_inequality_l327_32775

theorem logarithmic_AMGM_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  2 * ((Real.log b / (a * Real.log a)) / (a + b) + 
       (Real.log c / (b * Real.log b)) / (b + c) + 
       (Real.log a / (c * Real.log c)) / (c + a)) 
  ≥ 9 / (a + b + c) := 
sorry

end logarithmic_AMGM_inequality_l327_32775


namespace student_passing_percentage_l327_32791

variable (marks_obtained failed_by max_marks : ℕ)

def passing_marks (marks_obtained failed_by : ℕ) : ℕ :=
  marks_obtained + failed_by

def percentage_needed (passing_marks max_marks : ℕ) : ℚ :=
  (passing_marks : ℚ) / (max_marks : ℚ) * 100

theorem student_passing_percentage
  (h1 : marks_obtained = 125)
  (h2 : failed_by = 40)
  (h3 : max_marks = 500) :
  percentage_needed (passing_marks marks_obtained failed_by) max_marks = 33 := by
  sorry

end student_passing_percentage_l327_32791


namespace ferris_wheel_seats_l327_32705

variable (total_people : ℕ) (people_per_seat : ℕ)

theorem ferris_wheel_seats (h1 : total_people = 18) (h2 : people_per_seat = 9) : total_people / people_per_seat = 2 := by
  sorry

end ferris_wheel_seats_l327_32705


namespace evaluate_expression_l327_32722

noncomputable def M (x y : ℝ) : ℝ := if x < y then y else x
noncomputable def m (x y : ℝ) : ℝ := if x < y then x else y

theorem evaluate_expression
  (p q r s t : ℝ)
  (h1 : p < q)
  (h2 : q < r)
  (h3 : r < s)
  (h4 : s < t)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ t ∧ t ≠ p ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ s ∧ q ≠ t ∧ r ≠ t):
  M (M p (m q r)) (m s (m p t)) = q := 
sorry

end evaluate_expression_l327_32722


namespace map_distance_l327_32783

theorem map_distance (scale : ℝ) (d_actual_km : ℝ) (d_actual_m : ℝ) (d_actual_cm : ℝ) (d_map : ℝ) :
  scale = 1 / 250000 →
  d_actual_km = 5 →
  d_actual_m = d_actual_km * 1000 →
  d_actual_cm = d_actual_m * 100 →
  d_map = (1 * d_actual_cm) / (1 / scale) →
  d_map = 2 :=
by sorry

end map_distance_l327_32783


namespace check_3x5_board_cannot_be_covered_l327_32735

/-- Define the concept of a checkerboard with a given number of rows and columns. -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Define the number of squares on a checkerboard. -/
def num_squares (cb : Checkerboard) : ℕ :=
  cb.rows * cb.cols

/-- Define whether a board can be completely covered by dominoes. -/
def can_be_covered_by_dominoes (cb : Checkerboard) : Prop :=
  (num_squares cb) % 2 = 0

/-- Instantiate the specific checkerboard scenarios. -/
def board_3x4 := Checkerboard.mk 3 4
def board_3x5 := Checkerboard.mk 3 5
def board_4x4 := Checkerboard.mk 4 4
def board_4x5 := Checkerboard.mk 4 5
def board_6x3 := Checkerboard.mk 6 3

/-- Statement to prove which board cannot be covered completely by dominoes. -/
theorem check_3x5_board_cannot_be_covered : ¬ can_be_covered_by_dominoes board_3x5 :=
by
  /- We leave out the proof steps here as requested. -/
  sorry

end check_3x5_board_cannot_be_covered_l327_32735


namespace alpha_in_second_quadrant_l327_32764

theorem alpha_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 > 0 :=
by
  -- Given conditions
  have : Real.sin α > 0 := h1
  have : Real.cos α < 0 := h2
  sorry

end alpha_in_second_quadrant_l327_32764


namespace tea_sale_price_correct_l327_32750

noncomputable def cost_price (weight: ℕ) (unit_price: ℕ) : ℕ := weight * unit_price
noncomputable def desired_profit (cost: ℕ) (percentage: ℕ) : ℕ := cost * percentage / 100
noncomputable def sale_price (cost: ℕ) (profit: ℕ) : ℕ := cost + profit
noncomputable def sale_price_per_kg (total_sale_price: ℕ) (weight: ℕ) : ℚ := total_sale_price / weight

theorem tea_sale_price_correct :
  ∀ (weight_A weight_B weight_C weight_D cost_per_kg_A cost_per_kg_B cost_per_kg_C cost_per_kg_D
     profit_percent_A profit_percent_B profit_percent_C profit_percent_D : ℕ),

  weight_A = 80 →
  weight_B = 20 →
  weight_C = 50 →
  weight_D = 30 →
  cost_per_kg_A = 15 →
  cost_per_kg_B = 20 →
  cost_per_kg_C = 25 →
  cost_per_kg_D = 30 →
  profit_percent_A = 25 →
  profit_percent_B = 30 →
  profit_percent_C = 20 →
  profit_percent_D = 15 →
  
  sale_price_per_kg (sale_price (cost_price weight_A cost_per_kg_A) (desired_profit (cost_price weight_A cost_per_kg_A) profit_percent_A)) weight_A = 18.75 →
  sale_price_per_kg (sale_price (cost_price weight_B cost_per_kg_B) (desired_profit (cost_price weight_B cost_per_kg_B) profit_percent_B)) weight_B = 26 →
  sale_price_per_kg (sale_price (cost_price weight_C cost_per_kg_C) (desired_profit (cost_price weight_C cost_per_kg_C) profit_percent_C)) weight_C = 30 →
  sale_price_per_kg (sale_price (cost_price weight_D cost_per_kg_D) (desired_profit (cost_price weight_D cost_per_kg_D) profit_percent_D)) weight_D = 34.5 :=
by
  intros
  sorry

end tea_sale_price_correct_l327_32750


namespace polynomial_is_monic_l327_32795

noncomputable def f : ℝ → ℝ := sorry

variables (h1 : f 1 = 3) (h2 : f 2 = 12) (h3 : ∀ x : ℝ, f x = x^2 + 6*x - 4)

theorem polynomial_is_monic (f : ℝ → ℝ) (h1 : f 1 = 3) (h2 : f 2 = 12) (h3 : ∀ x : ℝ, f x = x^2 + x + b) : 
  ∀ x : ℝ, f x = x^2 + 6*x - 4 :=
by sorry

end polynomial_is_monic_l327_32795


namespace triangle_inequality_l327_32766

theorem triangle_inequality (a b c : ℝ) (h : a < b + c) : a^2 - b^2 - c^2 - 2*b*c < 0 := by
  sorry

end triangle_inequality_l327_32766


namespace reading_proof_l327_32749

noncomputable def reading (arrow_pos : ℝ) : ℝ :=
  if arrow_pos > 9.75 ∧ arrow_pos < 10.0 then 9.95 else 0

theorem reading_proof
  (arrow_pos : ℝ)
  (h0 : 9.75 < arrow_pos)
  (h1 : arrow_pos < 10.0)
  (possible_readings : List ℝ)
  (h2 : possible_readings = [9.80, 9.90, 9.95, 10.0, 9.85]) :
  reading arrow_pos = 9.95 := by
  -- Proof would go here
  sorry

end reading_proof_l327_32749


namespace find_uv_non_integer_l327_32782

noncomputable def q (x y : ℝ) (b : ℕ → ℝ) := 
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3

theorem find_uv_non_integer (b : ℕ → ℝ) 
  (h0 : q 0 0 b = 0) 
  (h1 : q 1 0 b = 0) 
  (h2 : q (-1) 0 b = 0) 
  (h3 : q 0 1 b = 0) 
  (h4 : q 0 (-1) b = 0) 
  (h5 : q 1 1 b = 0) 
  (h6 : q 1 (-1) b = 0) 
  (h7 : q 3 3 b = 0) : 
  ∃ u v : ℝ, q u v b = 0 ∧ u = 17/19 ∧ v = 18/19 := 
  sorry

end find_uv_non_integer_l327_32782


namespace total_selling_price_l327_32747

theorem total_selling_price
  (cost1 : ℝ) (cost2 : ℝ) (cost3 : ℝ) 
  (profit_percent1 : ℝ) (profit_percent2 : ℝ) (profit_percent3 : ℝ) :
  cost1 = 600 → cost2 = 450 → cost3 = 750 →
  profit_percent1 = 0.08 → profit_percent2 = 0.10 → profit_percent3 = 0.15 →
  (cost1 * (1 + profit_percent1) + cost2 * (1 + profit_percent2) + cost3 * (1 + profit_percent3)) = 2005.50 :=
by
  intros h1 h2 h3 p1 p2 p3
  simp [h1, h2, h3, p1, p2, p3]
  sorry

end total_selling_price_l327_32747


namespace evaluate_expression_l327_32731

variables (a b c d m : ℝ)

lemma opposite_is_zero (h1 : a + b = 0) : a + b = 0 := h1

lemma reciprocals_equal_one (h2 : c * d = 1) : c * d = 1 := h2

lemma abs_value_two (h3 : |m| = 2) : |m| = 2 := h3

theorem evaluate_expression (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 2) :
  m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 :=
by
  sorry

end evaluate_expression_l327_32731


namespace sum_of_squares_l327_32729

theorem sum_of_squares (n : ℕ) (x : ℕ) (h1 : (x + 1)^3 - x^3 = n^2) (h2 : n > 0) : ∃ a b : ℕ, n = a^2 + b^2 :=
by
  sorry

end sum_of_squares_l327_32729


namespace increased_work_l327_32771

variable (W p : ℕ)

theorem increased_work (hW : W > 0) (hp : p > 0) : 
  (W / (7 * p / 8)) - (W / p) = W / (7 * p) := 
sorry

end increased_work_l327_32771


namespace pure_imaginary_complex_l327_32720

theorem pure_imaginary_complex (m : ℝ) (i : ℂ) (h : i^2 = -1) :
    (∃ (y : ℂ), (2 - m * i) / (1 + i) = y * i) ↔ m = 2 :=
by
  sorry

end pure_imaginary_complex_l327_32720


namespace car_mpg_in_city_l327_32798

theorem car_mpg_in_city
  (H C T : ℕ)
  (h1 : H * T = 462)
  (h2 : C * T = 336)
  (h3 : C = H - 9) : C = 24 := by
  sorry

end car_mpg_in_city_l327_32798


namespace simplified_expression_value_at_4_l327_32762

theorem simplified_expression (x : ℝ) (h : x ≠ 5) : (x^2 - 3*x - 10) / (x - 5) = x + 2 := 
sorry

theorem value_at_4 : (4 : ℝ)^2 - 3*4 - 10 / (4 - 5) = 6 := 
sorry

end simplified_expression_value_at_4_l327_32762


namespace binary_addition_l327_32727

theorem binary_addition :
  0b1101 + 0b101 + 0b1110 + 0b10111 + 0b11000 = 0b11100010 :=
by
  sorry

end binary_addition_l327_32727


namespace sumata_miles_per_day_l327_32789

theorem sumata_miles_per_day (total_miles : ℝ) (total_days : ℝ) (h1 : total_miles = 250.0) (h2 : total_days = 5.0) :
  total_miles / total_days = 50.0 :=
by
  sorry

end sumata_miles_per_day_l327_32789


namespace arithmetic_sequence_sum_l327_32708

theorem arithmetic_sequence_sum :
  ∃ (c d e : ℕ), 
  c = 15 + (9 - 3) ∧ 
  d = c + (9 - 3) ∧ 
  e = d + (9 - 3) ∧ 
  c + d + e = 81 :=
by 
  sorry

end arithmetic_sequence_sum_l327_32708


namespace find_D_l327_32773

theorem find_D (D E F : ℝ) (h : ∀ x : ℝ, x ≠ 1 → x ≠ -2 → (1 / (x^3 - 3*x^2 - 4*x + 12)) = (D / (x - 1)) + (E / (x + 2)) + (F / (x + 2)^2)) :
    D = -1 / 15 :=
by
  -- the proof is omitted as per the instructions
  sorry

end find_D_l327_32773


namespace find_common_tangent_sum_constant_l327_32742

theorem find_common_tangent_sum_constant :
  ∃ (a b c : ℕ), (∀ x y : ℚ, y = x^2 + 169/100 → x = y^2 + 49/4 → a * x + b * y = c) ∧
  (Int.gcd (Int.gcd a b) c = 1) ∧
  (a + b + c = 52) :=
sorry

end find_common_tangent_sum_constant_l327_32742


namespace find_b_minus_a_l327_32702

noncomputable def rotate_90_counterclockwise (x y xc yc : ℝ) : ℝ × ℝ :=
  (xc + (-(y - yc)), yc + (x - xc))

noncomputable def reflect_about_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem find_b_minus_a (a b : ℝ) :
  let xc := 2
  let yc := 3
  let P := (a, b)
  let P_rotated := rotate_90_counterclockwise a b xc yc
  let P_reflected := reflect_about_y_eq_x P_rotated.1 P_rotated.2
  P_reflected = (4, 1) →
  b - a = 1 :=
by
  intros
  sorry

end find_b_minus_a_l327_32702


namespace hyperbola_foci_distance_l327_32748

theorem hyperbola_foci_distance (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 9) :
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 34 := 
by
  sorry

end hyperbola_foci_distance_l327_32748


namespace area_of_triangle_CM_N_l327_32739

noncomputable def triangle_area (a : ℝ) : ℝ :=
  let M := (a / 2, a, a)
  let N := (a, a / 2, a)
  let MN := Real.sqrt ((a - a / 2) ^ 2 + (a / 2 - a) ^ 2)
  let CK := Real.sqrt (a ^ 2 + (a * Real.sqrt 2 / 4) ^ 2)
  (1/2) * MN * CK

theorem area_of_triangle_CM_N 
  (a : ℝ) :
  (a > 0) →
  triangle_area a = (3 * a^2) / 8 :=
by
  intro h
  -- Proof will go here.
  sorry

end area_of_triangle_CM_N_l327_32739


namespace pow_mod_3_225_l327_32772

theorem pow_mod_3_225 :
  (3 ^ 225) % 11 = 1 :=
by
  -- Given condition from problem:
  have h : 3 ^ 5 % 11 = 1 := by norm_num
  -- Proceed to prove based on this condition
  sorry

end pow_mod_3_225_l327_32772


namespace simplify_expression_zero_l327_32704

noncomputable def simplify_expression (a b c d : ℝ) : ℝ :=
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)

theorem simplify_expression_zero (a b c d : ℝ) (h : a + b + c = d)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  simplify_expression a b c d = 0 :=
by
  sorry

end simplify_expression_zero_l327_32704


namespace real_roots_condition_l327_32701

theorem real_roots_condition (a : ℝ) (h : a ≠ -1) : 
    (∃ x : ℝ, x^2 + a * x + (a + 1)^2 = 0) ↔ a ∈ Set.Icc (-2 : ℝ) (-2 / 3) :=
sorry

end real_roots_condition_l327_32701


namespace side_length_of_S2_is_1001_l327_32746

-- Definitions and Conditions
variables (R1 R2 : Type) (S1 S2 S3 : Type)
variables (r s : ℤ)
variables (h_total_width : 2 * r + 3 * s = 4422)
variables (h_total_height : 2 * r + s = 2420)

theorem side_length_of_S2_is_1001 (R1 R2 S1 S2 S3 : Type) (r s : ℤ)
  (h_total_width : 2 * r + 3 * s = 4422)
  (h_total_height : 2 * r + s = 2420) : s = 1001 :=
by
  sorry -- proof to be provided

end side_length_of_S2_is_1001_l327_32746


namespace alyssa_picked_42_l327_32753

variable (totalPears nancyPears : ℕ)
variable (total_picked : totalPears = 59)
variable (nancy_picked : nancyPears = 17)

theorem alyssa_picked_42 (h1 : totalPears = 59) (h2 : nancyPears = 17) :
  totalPears - nancyPears = 42 :=
by
  sorry

end alyssa_picked_42_l327_32753


namespace f_at_five_l327_32785

def f (n : ℕ) : ℕ := n^3 + 2 * n^2 + 3 * n + 17

theorem f_at_five : f 5 = 207 := 
by 
sorry

end f_at_five_l327_32785


namespace domain_of_f_x_squared_l327_32796

theorem domain_of_f_x_squared {f : ℝ → ℝ} (h : ∀ x, -2 ≤ x ∧ x ≤ 3 → ∃ y, f (x + 1) = y) : 
  ∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ y, f (x ^ 2) = y := 
by 
  sorry

end domain_of_f_x_squared_l327_32796


namespace cube_root_of_neg_27_over_8_l327_32757

theorem cube_root_of_neg_27_over_8 :
  (- (3 : ℝ) / 2) ^ 3 = - (27 / 8 : ℝ) := 
by
  sorry

end cube_root_of_neg_27_over_8_l327_32757


namespace sally_paid_peaches_l327_32721

def total_spent : ℝ := 23.86
def amount_spent_on_cherries : ℝ := 11.54
def amount_spent_on_peaches_after_coupon : ℝ := total_spent - amount_spent_on_cherries

theorem sally_paid_peaches : amount_spent_on_peaches_after_coupon = 12.32 :=
by 
  -- The actual proof will involve concrete calculation here.
  -- For now, we skip it with sorry.
  sorry

end sally_paid_peaches_l327_32721


namespace circumcenter_rational_l327_32710

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end circumcenter_rational_l327_32710


namespace line_of_symmetry_is_x_eq_0_l327_32724

variable (f : ℝ → ℝ)

theorem line_of_symmetry_is_x_eq_0 :
  (∀ y, f (10 + y) = f (10 - y)) → ( ∃ l, l = 0 ∧ ∀ x,  f (10 + l + x) = f (10 + l - x)) := 
by
  sorry

end line_of_symmetry_is_x_eq_0_l327_32724


namespace eccentricity_range_of_ellipse_l327_32779

theorem eccentricity_range_of_ellipse 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (P : ℝ × ℝ) (hP_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (h_foci_relation : ∀(θ₁ θ₂ : ℝ), a / (Real.sin θ₁) = c / (Real.sin θ₂)) :
  ∃ (e : ℝ), e = c / a ∧ (Real.sqrt 2 - 1 < e ∧ e < 1) := 
sorry

end eccentricity_range_of_ellipse_l327_32779


namespace slices_per_person_eq_three_l327_32755

variables (num_people : ℕ) (slices_per_pizza : ℕ) (num_pizzas : ℕ)

theorem slices_per_person_eq_three (h1 : num_people = 18) (h2 : slices_per_pizza = 9) (h3 : num_pizzas = 6) : 
  (num_pizzas * slices_per_pizza) / num_people = 3 :=
sorry

end slices_per_person_eq_three_l327_32755


namespace correct_option_is_C_l327_32752

-- Definitions for given conditions
def optionA (x y : ℝ) : Prop := 3 * x + 3 * y = 6 * x * y
def optionB (x y : ℝ) : Prop := 4 * x * y^2 - 5 * x * y^2 = -1
def optionC (x : ℝ) : Prop := -2 * (x - 3) = -2 * x + 6
def optionD (a : ℝ) : Prop := 2 * a + a = 3 * a^2

-- The proof statement to show that Option C is the correct calculation
theorem correct_option_is_C (x y a : ℝ) : 
  ¬ optionA x y ∧ ¬ optionB x y ∧ optionC x ∧ ¬ optionD a :=
by
  -- Proof not required, using sorry to compile successfully
  sorry

end correct_option_is_C_l327_32752


namespace village_population_rate_l327_32751

noncomputable def population_change_X (initial_X : ℕ) (decrease_rate : ℕ) (years : ℕ) : ℕ :=
  initial_X - decrease_rate * years

noncomputable def population_change_Y (initial_Y : ℕ) (increase_rate : ℕ) (years : ℕ) : ℕ :=
  initial_Y + increase_rate * years

theorem village_population_rate (initial_X decrease_rate initial_Y years result : ℕ) 
  (h1 : initial_X = 70000) (h2 : decrease_rate = 1200) 
  (h3 : initial_Y = 42000) (h4 : years = 14) 
  (h5 : initial_X - decrease_rate * years = initial_Y + result * years) 
  : result = 800 :=
  sorry

end village_population_rate_l327_32751


namespace father_dig_time_l327_32738

-- Definitions based on the conditions
variable (T : ℕ) -- Time taken by the father to dig the hole in hours
variable (D : ℕ) -- Depth of the hole dug by the father in feet
variable (M : ℕ) -- Depth of the hole dug by Michael in feet

-- Conditions
def father_hole_depth : Prop := D = 4 * T
def michael_hole_depth : Prop := M = 2 * D - 400
def michael_dig_time : Prop := M = 4 * 700

-- The proof statement, proving T = 400 given the conditions
theorem father_dig_time (T D M : ℕ)
  (h1 : father_hole_depth T D)
  (h2 : michael_hole_depth D M)
  (h3 : michael_dig_time M) : T = 400 := 
by
  sorry

end father_dig_time_l327_32738


namespace janes_stick_shorter_than_sarahs_l327_32732

theorem janes_stick_shorter_than_sarahs :
  ∀ (pat_length jane_length pat_dirt sarah_factor : ℕ),
    pat_length = 30 →
    jane_length = 22 →
    pat_dirt = 7 →
    sarah_factor = 2 →
    (sarah_factor * (pat_length - pat_dirt)) - jane_length = 24 :=
by
  intros pat_length jane_length pat_dirt sarah_factor h1 h2 h3 h4
  -- sorry skips the proof
  sorry

end janes_stick_shorter_than_sarahs_l327_32732


namespace solution_inequality_l327_32719

theorem solution_inequality
  (a a' b b' c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a' ≠ 0)
  (h₃ : (c - b) / a > (c - b') / a') :
  (c - b') / a' < (c - b) / a :=
by
  sorry

end solution_inequality_l327_32719


namespace geometric_sequence_a_eq_neg4_l327_32756

theorem geometric_sequence_a_eq_neg4 
    (a : ℝ)
    (h : (2 * a + 2) ^ 2 = a * (3 * a + 3)) : 
    a = -4 :=
sorry

end geometric_sequence_a_eq_neg4_l327_32756


namespace sale_day_intersection_in_july_l327_32793

def is_multiple_of_five (d : ℕ) : Prop :=
  d % 5 = 0

def shoe_store_sale_days (d : ℕ) : Prop :=
  ∃ (k : ℕ), d = 3 + k * 6

theorem sale_day_intersection_in_july : 
  (∃ d, is_multiple_of_five d ∧ shoe_store_sale_days d ∧ 1 ≤ d ∧ d ≤ 31) = (1 = Nat.card {d | is_multiple_of_five d ∧ shoe_store_sale_days d ∧ 1 ≤ d ∧ d ≤ 31}) :=
by
  sorry

end sale_day_intersection_in_july_l327_32793


namespace solution_set_equivalence_l327_32778

def solution_set_inequality (x : ℝ) : Prop :=
  abs (x - 1) + abs x < 3

theorem solution_set_equivalence :
  { x : ℝ | solution_set_inequality x } = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_equivalence_l327_32778


namespace remainder_gx12_div_gx_l327_32770

-- Definition of the polynomial g(x)
def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Theorem stating the problem
theorem remainder_gx12_div_gx : ∀ x : ℂ, (g (x^12)) % (g x) = 6 := by
  sorry

end remainder_gx12_div_gx_l327_32770


namespace combined_tax_rate_is_correct_l327_32769

noncomputable def combined_tax_rate (john_income : ℝ) (ingrid_income : ℝ) (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) : ℝ :=
  let john_tax := john_tax_rate * john_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let total_tax := john_tax + ingrid_tax
  let total_income := john_income + ingrid_income
  total_tax / total_income

theorem combined_tax_rate_is_correct :
  combined_tax_rate 56000 72000 0.30 0.40 = 0.35625 := 
by
  sorry

end combined_tax_rate_is_correct_l327_32769


namespace determine_pairs_l327_32745

theorem determine_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  (∃ k : ℕ, k > 0 ∧ (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1)) :=
by
  sorry

end determine_pairs_l327_32745


namespace xiao_wang_ways_to_make_8_cents_l327_32761

theorem xiao_wang_ways_to_make_8_cents :
  (∃ c1 c2 c5 : ℕ, c1 ≤ 8 ∧ c2 ≤ 4 ∧ c5 ≤ 1 ∧ c1 + 2 * c2 + 5 * c5 = 8) → (number_of_ways_to_make_8_cents = 7) :=
sorry

end xiao_wang_ways_to_make_8_cents_l327_32761


namespace find_first_term_of_sequence_l327_32723

theorem find_first_term_of_sequence
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n+1) = a n + d)
  (h2 : a 0 + a 1 + a 2 = 12)
  (h3 : a 0 * a 1 * a 2 = 48)
  (h4 : ∀ n m, n < m → a n ≤ a m) :
  a 0 = 2 :=
sorry

end find_first_term_of_sequence_l327_32723


namespace triangular_25_l327_32712

-- Defining the formula for the n-th triangular number.
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Stating that the 25th triangular number is 325.
theorem triangular_25 : triangular 25 = 325 :=
  by
    -- We don't prove it here, so we simply state it requires a proof.
    sorry

end triangular_25_l327_32712


namespace blue_red_area_ratio_l327_32711

theorem blue_red_area_ratio (d1 d2 : ℝ) (h1 : d1 = 2) (h2 : d2 = 6) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let a_red := π * r1^2
  let a_large := π * r2^2
  let a_blue := a_large - a_red
  a_blue / a_red = 8 :=
by
  have r1 := d1 / 2
  have r2 := d2 / 2
  have a_red := π * r1^2
  have a_large := π * r2^2
  have a_blue := a_large - a_red
  sorry

end blue_red_area_ratio_l327_32711


namespace f_of_1_eq_zero_l327_32740

-- Conditions
variables (f : ℝ → ℝ)
-- f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
-- f is a periodic function with a period of 2
def periodic_function (f : ℝ → ℝ) := ∀ x : ℝ, f (x + 2) = f x

-- Theorem statement
theorem f_of_1_eq_zero {f : ℝ → ℝ} (h1 : odd_function f) (h2 : periodic_function f) : f 1 = 0 :=
by { sorry }

end f_of_1_eq_zero_l327_32740


namespace probability_of_red_ball_is_correct_l327_32725

noncomputable def probability_of_drawing_red_ball (white_balls : ℕ) (red_balls : ℕ) :=
  let total_balls := white_balls + red_balls
  let favorable_outcomes := red_balls
  (favorable_outcomes : ℚ) / total_balls

theorem probability_of_red_ball_is_correct :
  probability_of_drawing_red_ball 5 2 = 2 / 7 :=
by
  sorry

end probability_of_red_ball_is_correct_l327_32725


namespace ratio_of_larger_to_smaller_l327_32715

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx : x > y) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l327_32715


namespace other_endpoint_l327_32794

theorem other_endpoint (M : ℝ × ℝ) (A : ℝ × ℝ) (x y : ℝ) :
  M = (2, 3) ∧ A = (5, -1) ∧ (M = ((A.1 + x) / 2, (A.2 + y) / 2)) → (x, y) = (-1, 7) := by
  sorry

end other_endpoint_l327_32794
