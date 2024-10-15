import Mathlib

namespace NUMINAMATH_GPT_correct_propositions_l2379_237945

-- Define propositions
def proposition1 : Prop :=
  ∀ x, 2 * (Real.cos (1/3 * x + Real.pi / 4))^2 - 1 = -Real.sin (2 * x / 3)

def proposition2 : Prop :=
  ∃ α : ℝ, Real.sin α + Real.cos α = 3 / 2

def proposition3 : Prop :=
  ∀ α β : ℝ, (0 < α ∧ α < Real.pi / 2) → (0 < β ∧ β < Real.pi / 2) → α < β → Real.tan α < Real.tan β

def proposition4 : Prop :=
  ∀ x, x = Real.pi / 8 → Real.sin (2 * x + 5 * Real.pi / 4) = -1

def proposition5 : Prop :=
  Real.sin ( 2 * (Real.pi / 12) + Real.pi / 3 ) = 0

-- Define the main theorem combining correct propositions
theorem correct_propositions : 
  proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ proposition4 ∧ ¬proposition5 :=
  by
  -- Since we only need to state the theorem, we use sorry.
  sorry

end NUMINAMATH_GPT_correct_propositions_l2379_237945


namespace NUMINAMATH_GPT_simple_interest_for_2_years_l2379_237912

noncomputable def calculate_simple_interest (P r t : ℝ) : ℝ :=
  (P * r * t) / 100

theorem simple_interest_for_2_years (CI P r t : ℝ) (hCI : CI = P * (1 + r / 100)^t - P)
  (hCI_value : CI = 615) (r_value : r = 5) (t_value : t = 2) : 
  calculate_simple_interest P r t = 600 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_for_2_years_l2379_237912


namespace NUMINAMATH_GPT_find_y_l2379_237957

theorem find_y (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 24) : y = 120 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2379_237957


namespace NUMINAMATH_GPT_max_sum_red_green_balls_l2379_237941

theorem max_sum_red_green_balls (total_balls : ℕ) (green_balls : ℕ) (max_red_balls : ℕ) 
  (h1 : total_balls = 28) (h2 : green_balls = 12) (h3 : max_red_balls ≤ 11) : 
  (max_red_balls + green_balls) = 23 := 
sorry

end NUMINAMATH_GPT_max_sum_red_green_balls_l2379_237941


namespace NUMINAMATH_GPT_blocks_needed_for_enclosure_l2379_237958

noncomputable def volume_of_rectangular_prism (length: ℝ) (width: ℝ) (height: ℝ) : ℝ :=
  length * width * height

theorem blocks_needed_for_enclosure 
  (length width height thickness : ℝ)
  (H_length : length = 15)
  (H_width : width = 12)
  (H_height : height = 6)
  (H_thickness : thickness = 1.5) :
  volume_of_rectangular_prism length width height - 
  volume_of_rectangular_prism (length - 2 * thickness) (width - 2 * thickness) (height - thickness) = 594 :=
by
  sorry

end NUMINAMATH_GPT_blocks_needed_for_enclosure_l2379_237958


namespace NUMINAMATH_GPT_num_terms_arithmetic_sequence_is_41_l2379_237979

-- Definitions and conditions
def first_term : ℤ := 200
def common_difference : ℤ := -5
def last_term : ℤ := 0

-- Definition of the n-th term of arithmetic sequence
def nth_term (a : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  a + (n - 1) * d

-- Statement to prove
theorem num_terms_arithmetic_sequence_is_41 : 
  ∃ n : ℕ, nth_term first_term common_difference n = 0 ∧ n = 41 :=
by 
  sorry

end NUMINAMATH_GPT_num_terms_arithmetic_sequence_is_41_l2379_237979


namespace NUMINAMATH_GPT_clock_correction_calculation_l2379_237913

noncomputable def clock_correction : ℝ :=
  let daily_gain := 5/4
  let hourly_gain := daily_gain / 24
  let total_hours := (9 * 24) + 9
  let total_gain := total_hours * hourly_gain
  total_gain

theorem clock_correction_calculation : clock_correction = 11.72 := by
  sorry

end NUMINAMATH_GPT_clock_correction_calculation_l2379_237913


namespace NUMINAMATH_GPT_greatest_third_side_of_triangle_l2379_237918

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 15) :
  ∃ x : ℕ, 8 < x ∧ x < 22 ∧ (∀ y : ℕ, 8 < y ∧ y < 22 → y ≤ x) ∧ x = 21 :=
by
  sorry

end NUMINAMATH_GPT_greatest_third_side_of_triangle_l2379_237918


namespace NUMINAMATH_GPT_overtakes_in_16_minutes_l2379_237965

def number_of_overtakes (track_length : ℕ) (speed_a : ℕ) (speed_b : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let relative_speed := speed_a - speed_b
  let time_per_overtake := track_length / relative_speed
  time_seconds / time_per_overtake

theorem overtakes_in_16_minutes :
  number_of_overtakes 200 6 4 16 = 9 :=
by
  -- We will insert calculations or detailed proof steps if needed
  sorry

end NUMINAMATH_GPT_overtakes_in_16_minutes_l2379_237965


namespace NUMINAMATH_GPT_slope_of_line_l2379_237926

theorem slope_of_line :
  ∃ (m : ℝ), (∃ b : ℝ, ∀ x y : ℝ, y = m * x + b) ∧
             (b = 2 ∧ ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ = 0 ∧ x₂ = 269 ∧ y₁ = 2 ∧ y₂ = 540 ∧ 
             m = (y₂ - y₁) / (x₂ - x₁)) ∧
             m = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_slope_of_line_l2379_237926


namespace NUMINAMATH_GPT_find_set_l2379_237997

/-- Definition of set A -/
def setA : Set ℝ := { x : ℝ | abs x < 4 }

/-- Definition of set B -/
def setB : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 > 0 }

/-- Definition of the intersection A ∩ B -/
def intersectionAB : Set ℝ := { x : ℝ | abs x < 4 ∧ (x > 3 ∨ x < 1) }

/-- Definition of the set we want to find -/
def setDesired : Set ℝ := { x : ℝ | abs x < 4 ∧ ¬(abs x < 4 ∧ (x > 3 ∨ x < 1)) }

/-- The statement to prove -/
theorem find_set :
  setDesired = { x : ℝ | 1 ≤ x ∧ x ≤ 3 } :=
sorry

end NUMINAMATH_GPT_find_set_l2379_237997


namespace NUMINAMATH_GPT_sum_of_f_l2379_237907

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) / (3^x + (Real.sqrt 3))

theorem sum_of_f :
  (f (-12) + f (-11) + f (-10) + f (-9) + f (-8) + f (-7) + f (-6) + 
   f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f (0) + f (1) + f (2) + 
   f (3) + f (4) + f (5) + f (6) + f (7) + f (8) + f (9) + f (10) + 
   f (11) + f (12) + f (13)) = 13 :=
sorry

end NUMINAMATH_GPT_sum_of_f_l2379_237907


namespace NUMINAMATH_GPT_product_of_common_ratios_l2379_237978

theorem product_of_common_ratios (x p r a2 a3 b2 b3 : ℝ)
  (h1 : a2 = x * p) (h2 : a3 = x * p^2)
  (h3 : b2 = x * r) (h4 : b3 = x * r^2)
  (h5 : 3 * a3 - 4 * b3 = 5 * (3 * a2 - 4 * b2))
  (h_nonconstant : x ≠ 0) (h_diff_ratios : p ≠ r) :
  p * r = 9 :=
by
  sorry

end NUMINAMATH_GPT_product_of_common_ratios_l2379_237978


namespace NUMINAMATH_GPT_prop_B_contrapositive_correct_l2379_237953

/-
Proposition B: The contrapositive of the proposition 
"If x^2 < 1, then -1 < x < 1" is 
"If x ≥ 1 or x ≤ -1, then x^2 ≥ 1".
-/
theorem prop_B_contrapositive_correct :
  (∀ (x : ℝ), x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ (x : ℝ), (x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
sorry

end NUMINAMATH_GPT_prop_B_contrapositive_correct_l2379_237953


namespace NUMINAMATH_GPT_trivia_game_answer_l2379_237903

theorem trivia_game_answer (correct_first_half : Nat)
    (points_per_question : Nat) (final_score : Nat) : 
    correct_first_half = 8 → 
    points_per_question = 8 →
    final_score = 80 →
    (final_score - correct_first_half * points_per_question) / points_per_question = 2 :=
by
    intros h1 h2 h3
    sorry

end NUMINAMATH_GPT_trivia_game_answer_l2379_237903


namespace NUMINAMATH_GPT_second_investment_value_l2379_237947

theorem second_investment_value
  (a : ℝ) (r1 r2 rt : ℝ) (x : ℝ)
  (h1 : a = 500)
  (h2 : r1 = 0.07)
  (h3 : r2 = 0.09)
  (h4 : rt = 0.085)
  (h5 : r1 * a + r2 * x = rt * (a + x)) :
  x = 1500 :=
by 
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_second_investment_value_l2379_237947


namespace NUMINAMATH_GPT_quadrilateral_angle_E_l2379_237916

theorem quadrilateral_angle_E (E F G H : ℝ)
  (h1 : E = 3 * F)
  (h2 : E = 4 * G)
  (h3 : E = 6 * H)
  (h_sum : E + F + G + H = 360) :
  E = 206 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_angle_E_l2379_237916


namespace NUMINAMATH_GPT_no_int_solutions_a_b_l2379_237993

theorem no_int_solutions_a_b :
  ¬ ∃ (a b : ℤ), a^2 + 1998 = b^2 :=
by
  sorry

end NUMINAMATH_GPT_no_int_solutions_a_b_l2379_237993


namespace NUMINAMATH_GPT_kelly_total_snacks_l2379_237999

theorem kelly_total_snacks (peanuts raisins : ℝ) (h₁ : peanuts = 0.1) (h₂ : raisins = 0.4) :
  peanuts + raisins = 0.5 :=
by
  simp [h₁, h₂]
  sorry

end NUMINAMATH_GPT_kelly_total_snacks_l2379_237999


namespace NUMINAMATH_GPT_boyfriend_picks_up_correct_l2379_237944

-- Define the initial condition
def init_pieces : ℕ := 60

-- Define the amount swept by Anne
def swept_pieces (n : ℕ) : ℕ := n / 2

-- Define the number of pieces stolen by the cat
def stolen_pieces : ℕ := 3

-- Define the remaining pieces after the cat steals
def remaining_pieces (n : ℕ) : ℕ := n - stolen_pieces

-- Define how many pieces the boyfriend picks up
def boyfriend_picks_up (n : ℕ) : ℕ := n / 3

-- The main theorem
theorem boyfriend_picks_up_correct : boyfriend_picks_up (remaining_pieces (init_pieces - swept_pieces init_pieces)) = 9 :=
by
  sorry

end NUMINAMATH_GPT_boyfriend_picks_up_correct_l2379_237944


namespace NUMINAMATH_GPT_average_marks_l2379_237981

theorem average_marks (A : ℝ) :
  let marks_first_class := 25 * A
  let marks_second_class := 30 * 60
  let total_marks := 55 * 50.90909090909091
  marks_first_class + marks_second_class = total_marks → A = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_l2379_237981


namespace NUMINAMATH_GPT_ab_cardinals_l2379_237928

open Set

/-- a|A| = b|B| given the conditions.
1. a and b are positive integers.
2. A and B are finite sets of integers such that:
   a. A and B are disjoint.
   b. If an integer i belongs to A or to B, then i + a ∈ A or i - b ∈ B.
-/
theorem ab_cardinals 
  (a b : ℕ) (A B : Finset ℤ) 
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (disjoint_AB : Disjoint A B)
  (condition_2 : ∀ i : ℤ, i ∈ A ∪ B → (i + a) ∈ A ∨ (i - b) ∈ B) :
  a * A.card = b * B.card := 
sorry

end NUMINAMATH_GPT_ab_cardinals_l2379_237928


namespace NUMINAMATH_GPT_balls_problem_l2379_237983

noncomputable def red_balls_initial := 420
noncomputable def total_balls_initial := 600
noncomputable def percent_red_required := 60 / 100

theorem balls_problem :
  ∃ (x : ℕ), 420 - x = (3 / 5) * (600 - x) :=
by
  sorry

end NUMINAMATH_GPT_balls_problem_l2379_237983


namespace NUMINAMATH_GPT_initial_students_count_l2379_237955

-- Definitions based on conditions
def initial_average_age (T : ℕ) (n : ℕ) : Prop := T = 14 * n
def new_average_age_after_adding (T : ℕ) (n : ℕ) : Prop := (T + 5 * 17) / (n + 5) = 15

-- Main proposition stating the problem
theorem initial_students_count (n : ℕ) (T : ℕ) 
  (h1 : initial_average_age T n)
  (h2 : new_average_age_after_adding T n) :
  n = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_students_count_l2379_237955


namespace NUMINAMATH_GPT_combinations_problem_l2379_237969

open Nat

-- Definitions for combinations
def C (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

-- Condition: Number of ways to choose 2 sergeants out of 6
def C_6_2 : Nat := C 6 2

-- Condition: Number of ways to choose 20 soldiers out of 60
def C_60_20 : Nat := C 60 20

-- Theorem statement for the problem
theorem combinations_problem :
  3 * C_6_2 * C_60_20 = 3 * 15 * C 60 20 := by
  simp [C_6_2, C_60_20, C]
  sorry

end NUMINAMATH_GPT_combinations_problem_l2379_237969


namespace NUMINAMATH_GPT_least_positive_whole_number_divisible_by_five_primes_l2379_237950

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_whole_number_divisible_by_five_primes_l2379_237950


namespace NUMINAMATH_GPT_solve_equation_l2379_237914

theorem solve_equation (x : ℚ) : 3 * (x - 2) = 2 - 5 * (x - 2) ↔ x = 9 / 4 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l2379_237914


namespace NUMINAMATH_GPT_total_expense_l2379_237940

noncomputable def sandys_current_age : ℕ := 36 - 2
noncomputable def sandys_monthly_expense : ℕ := 10 * sandys_current_age
noncomputable def alexs_current_age : ℕ := sandys_current_age / 2
noncomputable def alexs_next_month_expense : ℕ := 2 * sandys_monthly_expense

theorem total_expense : 
  sandys_monthly_expense + alexs_next_month_expense = 1020 := 
by 
  sorry

end NUMINAMATH_GPT_total_expense_l2379_237940


namespace NUMINAMATH_GPT_hyperbola_eccentricity_asymptotes_l2379_237963

theorem hyperbola_eccentricity_asymptotes :
  (∃ e: ℝ, ∃ m: ℝ, 
    (∀ x y, (x^2 / 8 - y^2 / 4 = 1) → e = Real.sqrt 6 / 2 ∧ y = m * x) ∧ 
    (m = Real.sqrt 2 / 2 ∨ m = -Real.sqrt 2 / 2)) :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_asymptotes_l2379_237963


namespace NUMINAMATH_GPT_six_a_seven_eight_b_div_by_45_l2379_237984

/-- If the number 6a78b is divisible by 45, then a + b = 6. -/
theorem six_a_seven_eight_b_div_by_45 (a b : ℕ) (h1: 0 ≤ a ∧ a < 10) (h2: 0 ≤ b ∧ b < 10)
  (h3 : (6 * 10^4 + a * 10^3 + 7 * 10^2 + 8 * 10 + b) % 45 = 0) : a + b = 6 := 
by
  sorry

end NUMINAMATH_GPT_six_a_seven_eight_b_div_by_45_l2379_237984


namespace NUMINAMATH_GPT_ski_price_l2379_237915

variable {x y : ℕ}

theorem ski_price (h1 : 2 * x + y = 340) (h2 : 3 * x + 2 * y = 570) : x = 110 ∧ y = 120 := by
  sorry

end NUMINAMATH_GPT_ski_price_l2379_237915


namespace NUMINAMATH_GPT_functional_equation_solution_l2379_237905

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l2379_237905


namespace NUMINAMATH_GPT_longer_side_of_new_rectangle_l2379_237954

theorem longer_side_of_new_rectangle {z : ℕ} (h : ∃x : ℕ, 9 * 16 = 144 ∧ x * z = 144 ∧ z ≠ 9 ∧ z ≠ 16) : z = 18 :=
sorry

end NUMINAMATH_GPT_longer_side_of_new_rectangle_l2379_237954


namespace NUMINAMATH_GPT_number_of_distinct_collections_l2379_237948

def mathe_matical_letters : Multiset Char :=
  {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'A', 'L'}

def vowels : Multiset Char :=
  {'A', 'A', 'A', 'E', 'I'}

def consonants : Multiset Char :=
  {'M', 'T', 'H', 'M', 'T', 'C', 'L', 'C'}

def indistinguishable (s : Multiset Char) :=
  (s.count 'A' = s.count 'A' ∧
   s.count 'E' = 1 ∧
   s.count 'I' = 1 ∧
   s.count 'M' = 2 ∧
   s.count 'T' = 2 ∧
   s.count 'H' = 1 ∧
   s.count 'C' = 2 ∧
   s.count 'L' = 1)

theorem number_of_distinct_collections :
  5 * 16 = 80 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_number_of_distinct_collections_l2379_237948


namespace NUMINAMATH_GPT_smallest_value_arithmetic_geometric_seq_l2379_237968

theorem smallest_value_arithmetic_geometric_seq :
  ∃ (E F G H : ℕ), (E < F) ∧ (F < G) ∧ (F * 4 = G * 7) ∧ (E + G = 2 * F) ∧ (F * F * 49 = G * G * 16) ∧ (E + F + G + H = 97) := 
sorry

end NUMINAMATH_GPT_smallest_value_arithmetic_geometric_seq_l2379_237968


namespace NUMINAMATH_GPT_purchase_total_cost_l2379_237931

theorem purchase_total_cost :
  (1 * 16) + (3 * 2) + (6 * 1) = 28 :=
sorry

end NUMINAMATH_GPT_purchase_total_cost_l2379_237931


namespace NUMINAMATH_GPT_shorter_side_length_l2379_237935

variables (x y : ℝ)
variables (h1 : 2 * x + 2 * y = 60)
variables (h2 : x * y = 200)

theorem shorter_side_length :
  min x y = 10 :=
by
  sorry

end NUMINAMATH_GPT_shorter_side_length_l2379_237935


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l2379_237985

theorem arithmetic_expression_evaluation :
  (-12 * 6) - (-4 * -8) + (-15 * -3) - (36 / (-2)) = -77 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l2379_237985


namespace NUMINAMATH_GPT_area_of_triangle_aef_l2379_237967

noncomputable def length_ab : ℝ := 10
noncomputable def width_ad : ℝ := 6
noncomputable def diagonal_ac : ℝ := Real.sqrt (length_ab^2 + width_ad^2)
noncomputable def segment_length_ac : ℝ := diagonal_ac / 4
noncomputable def area_aef : ℝ := (1/2) * segment_length_ac * ((60 * diagonal_ac) / diagonal_ac^2)

theorem area_of_triangle_aef : area_aef = 7.5 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_aef_l2379_237967


namespace NUMINAMATH_GPT_pq_sum_l2379_237976

theorem pq_sum (p q : ℝ) 
  (h1 : p / 3 = 9) 
  (h2 : q / 3 = 15) : 
  p + q = 72 :=
sorry

end NUMINAMATH_GPT_pq_sum_l2379_237976


namespace NUMINAMATH_GPT_interior_angles_sum_l2379_237925

def sum_of_interior_angles (sides : ℕ) : ℕ :=
  180 * (sides - 2)

theorem interior_angles_sum (n : ℕ) (h : sum_of_interior_angles n = 1800) :
  sum_of_interior_angles (n + 4) = 2520 :=
sorry

end NUMINAMATH_GPT_interior_angles_sum_l2379_237925


namespace NUMINAMATH_GPT_total_wasted_time_is_10_l2379_237917

-- Define the time Martin spends waiting in traffic
def waiting_time : ℕ := 2

-- Define the constant for the multiplier
def multiplier : ℕ := 4

-- Define the time spent trying to get off the freeway
def off_freeway_time : ℕ := waiting_time * multiplier

-- Define the total wasted time
def total_wasted_time : ℕ := waiting_time + off_freeway_time

-- Theorem stating that the total time wasted is 10 hours
theorem total_wasted_time_is_10 : total_wasted_time = 10 :=
by
  sorry

end NUMINAMATH_GPT_total_wasted_time_is_10_l2379_237917


namespace NUMINAMATH_GPT_lydia_ate_24_ounces_l2379_237951

theorem lydia_ate_24_ounces (total_fruit_pounds : ℕ) (mario_oranges_ounces : ℕ) (nicolai_peaches_pounds : ℕ) (total_fruit_ounces mario_oranges_ounces_in_ounces nicolai_peaches_ounces_in_ounces : ℕ) :
  total_fruit_pounds = 8 →
  mario_oranges_ounces = 8 →
  nicolai_peaches_pounds = 6 →
  total_fruit_ounces = total_fruit_pounds * 16 →
  mario_oranges_ounces_in_ounces = mario_oranges_ounces →
  nicolai_peaches_ounces_in_ounces = nicolai_peaches_pounds * 16 →
  (total_fruit_ounces - mario_oranges_ounces_in_ounces - nicolai_peaches_ounces_in_ounces) = 24 :=
by
  sorry

end NUMINAMATH_GPT_lydia_ate_24_ounces_l2379_237951


namespace NUMINAMATH_GPT_parity_of_expression_l2379_237962

theorem parity_of_expression (o1 o2 n : ℕ) (h1 : o1 % 2 = 1) (h2 : o2 % 2 = 1) : 
  ((o1 * o1 + n * (o1 * o2)) % 2 = 1 ↔ n % 2 = 0) :=
by sorry

end NUMINAMATH_GPT_parity_of_expression_l2379_237962


namespace NUMINAMATH_GPT_sales_in_second_month_l2379_237934

-- Given conditions:
def sales_first_month : ℕ := 6400
def sales_third_month : ℕ := 6800
def sales_fourth_month : ℕ := 7200
def sales_fifth_month : ℕ := 6500
def sales_sixth_month : ℕ := 5100
def average_sales : ℕ := 6500

-- Statement to prove:
theorem sales_in_second_month :
  ∃ (sales_second_month : ℕ), 
    average_sales * 6 = sales_first_month + sales_second_month + sales_third_month 
    + sales_fourth_month + sales_fifth_month + sales_sixth_month 
    ∧ sales_second_month = 7000 :=
  sorry

end NUMINAMATH_GPT_sales_in_second_month_l2379_237934


namespace NUMINAMATH_GPT_number_of_2_face_painted_cubes_l2379_237900

-- Condition definitions based on the problem statement
def painted_faces (n : ℕ) (type : String) : ℕ :=
  if type = "corner" then 8
  else if type = "edge" then 12
  else if type = "face" then 24
  else if type = "inner" then 9
  else 0

-- The mathematical proof statement
theorem number_of_2_face_painted_cubes : painted_faces 27 "edge" = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_2_face_painted_cubes_l2379_237900


namespace NUMINAMATH_GPT_christopher_more_than_karen_l2379_237936

-- Define the number of quarters Karen and Christopher have
def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64

-- Define the value of a quarter in dollars
def value_of_quarter : ℚ := 0.25

-- Define the amount of money Christopher has more than Karen in dollars
def christopher_more_money : ℚ := (christopher_quarters - karen_quarters) * value_of_quarter

-- Theorem to prove that Christopher has $8.00 more than Karen
theorem christopher_more_than_karen : christopher_more_money = 8 := by
  sorry

end NUMINAMATH_GPT_christopher_more_than_karen_l2379_237936


namespace NUMINAMATH_GPT_ned_trays_per_trip_l2379_237904

def trays_from_table1 : ℕ := 27
def trays_from_table2 : ℕ := 5
def total_trips : ℕ := 4
def total_trays : ℕ := trays_from_table1 + trays_from_table2
def trays_per_trip : ℕ := total_trays / total_trips

theorem ned_trays_per_trip :
  trays_per_trip = 8 :=
by
  -- proof is skipped
  sorry

end NUMINAMATH_GPT_ned_trays_per_trip_l2379_237904


namespace NUMINAMATH_GPT_general_formula_sum_first_n_terms_l2379_237977

open BigOperators

def geometric_sequence (a_3 : ℚ) (q : ℚ) : ℕ → ℚ
| 0       => 1 -- this is a placeholder since sequence usually start from 1
| (n + 1) => 1 * q ^ n

def sum_geometric_sequence (a_1 q : ℚ) (n : ℕ) : ℚ :=
  a_1 * (1 - q ^ n) / (1 - q)

theorem general_formula (a_3 : ℚ) (q : ℚ) (n : ℕ) (h_a3 : a_3 = 1 / 4) (h_q : q = -1 / 2) :
  geometric_sequence a_3 q (n + 1) = (-1 / 2) ^ n :=
by
  sorry

theorem sum_first_n_terms (a_1 q : ℚ) (n : ℕ) (h_a1 : a_1 = 1) (h_q : q = -1 / 2) :
  sum_geometric_sequence a_1 q n = 2 / 3 * (1 - (-1 / 2) ^ n) :=
by
  sorry

end NUMINAMATH_GPT_general_formula_sum_first_n_terms_l2379_237977


namespace NUMINAMATH_GPT_balloons_in_package_initially_l2379_237996

-- Definition of conditions
def friends : ℕ := 5
def balloons_given_back : ℕ := 11
def balloons_after_giving_back : ℕ := 39

-- Calculation for original balloons each friend had
def original_balloons_each_friend := balloons_after_giving_back + balloons_given_back

-- Theorem: Number of balloons in the package initially
theorem balloons_in_package_initially : 
  (original_balloons_each_friend * friends) = 250 :=
by
  sorry

end NUMINAMATH_GPT_balloons_in_package_initially_l2379_237996


namespace NUMINAMATH_GPT_price_of_skateboard_l2379_237986

-- Given condition (0.20 * p = 300)
variable (p : ℝ)
axiom upfront_payment : 0.20 * p = 300

-- Theorem statement to prove the price of the skateboard
theorem price_of_skateboard : p = 1500 := by
  sorry

end NUMINAMATH_GPT_price_of_skateboard_l2379_237986


namespace NUMINAMATH_GPT_arcsin_one_eq_pi_div_two_l2379_237972

noncomputable def arcsin : ℝ → ℝ := sorry -- Define arcsin function

theorem arcsin_one_eq_pi_div_two : arcsin 1 = Real.pi / 2 := sorry

end NUMINAMATH_GPT_arcsin_one_eq_pi_div_two_l2379_237972


namespace NUMINAMATH_GPT_molecular_weight_cao_is_correct_l2379_237991

-- Define the atomic weights of calcium and oxygen
def atomic_weight_ca : ℝ := 40.08
def atomic_weight_o : ℝ := 16.00

-- Define the molecular weight of CaO
def molecular_weight_cao : ℝ := atomic_weight_ca + atomic_weight_o

-- State the theorem to prove
theorem molecular_weight_cao_is_correct : molecular_weight_cao = 56.08 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_cao_is_correct_l2379_237991


namespace NUMINAMATH_GPT_cylinder_volume_scaling_l2379_237939

theorem cylinder_volume_scaling (r h : ℝ) (V : ℝ) (V' : ℝ) 
  (h_original : V = π * r^2 * h) 
  (h_new : V' = π * (1.5 * r)^2 * (3 * h)) :
  V' = 6.75 * V := by
  sorry

end NUMINAMATH_GPT_cylinder_volume_scaling_l2379_237939


namespace NUMINAMATH_GPT_f_10_equals_1_l2379_237919

noncomputable def f : ℝ → ℝ 
| x => sorry 

axiom odd_f_x_minus_1 : ∀ x : ℝ, f (-x-1) = -f (x-1)
axiom even_f_x_plus_1 : ∀ x : ℝ, f (-x+1) = f (x+1)
axiom f_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = 2^x

theorem f_10_equals_1 : f 10 = 1 :=
by
  sorry -- The actual proof goes here.

end NUMINAMATH_GPT_f_10_equals_1_l2379_237919


namespace NUMINAMATH_GPT_real_root_solution_l2379_237942

theorem real_root_solution (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  ∃ x1 x2 : ℝ, 
    (x1 < b ∧ b < x2) ∧
    (1 / (x1 - a) + 1 / (x1 - b) + 1 / (x1 - c) = 0) ∧ 
    (1 / (x2 - a) + 1 / (x2 - b) + 1 / (x2 - c) = 0) :=
by
  sorry

end NUMINAMATH_GPT_real_root_solution_l2379_237942


namespace NUMINAMATH_GPT_value_of_expression_at_x_eq_2_l2379_237952

theorem value_of_expression_at_x_eq_2 :
  (2 * (2: ℕ)^2 - 3 * 2 + 4 = 6) := 
by sorry

end NUMINAMATH_GPT_value_of_expression_at_x_eq_2_l2379_237952


namespace NUMINAMATH_GPT_exchange_rmb_ways_l2379_237921

theorem exchange_rmb_ways : 
  {n : ℕ // ∃ (x y z : ℕ), x + 2 * y + 5 * z = 10 ∧ n = 10} :=
sorry

end NUMINAMATH_GPT_exchange_rmb_ways_l2379_237921


namespace NUMINAMATH_GPT_minimum_shirts_for_savings_l2379_237964

theorem minimum_shirts_for_savings (x : ℕ) : 75 + 8 * x < 16 * x ↔ 10 ≤ x :=
by
  sorry

end NUMINAMATH_GPT_minimum_shirts_for_savings_l2379_237964


namespace NUMINAMATH_GPT_complement_of_intersection_l2379_237990

def S : Set ℝ := {-2, -1, 0, 1, 2}
def T : Set ℝ := {x | x + 1 ≤ 2}
def complement (A B : Set ℝ) : Set ℝ := {x ∈ B | x ∉ A}

theorem complement_of_intersection :
  complement (S ∩ T) S = {2} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_intersection_l2379_237990


namespace NUMINAMATH_GPT_rectangular_field_area_eq_l2379_237943

-- Definitions based on the problem's conditions
def length (x : ℝ) := x
def width (x : ℝ) := 60 - x
def area (x : ℝ) := x * (60 - x)

-- The proof statement
theorem rectangular_field_area_eq (x : ℝ) (h₀ : x + (60 - x) = 60) (h₁ : area x = 864) :
  x * (60 - x) = 864 :=
by
  -- Using the provided conditions and definitions, we aim to prove the equation.
  sorry

end NUMINAMATH_GPT_rectangular_field_area_eq_l2379_237943


namespace NUMINAMATH_GPT_fruit_salad_total_l2379_237909

def fruit_salad_problem (R_red G R_rasp total_fruit : ℕ) : Prop :=
  R_red = 67 ∧ (3 * G + 7 = 67) ∧ (R_rasp = G - 5) ∧ (total_fruit = R_red + G + R_rasp)

theorem fruit_salad_total (R_red G R_rasp : ℕ) (total_fruit : ℕ) :
  fruit_salad_problem R_red G R_rasp total_fruit → total_fruit = 102 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fruit_salad_total_l2379_237909


namespace NUMINAMATH_GPT_determine_c_l2379_237970

-- Definitions of the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 1

-- Hypothesis for the sequence to be geometric
def geometric_seq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∃ c, ∀ n, a (n + 1) + c = r * (a n + c)

-- The goal to prove
theorem determine_c (a : ℕ → ℕ) (c : ℕ) (r := 2) :
  seq a →
  geometric_seq a c →
  c = 1 :=
by
  intros h_seq h_geo
  sorry

end NUMINAMATH_GPT_determine_c_l2379_237970


namespace NUMINAMATH_GPT_consecutive_odd_integer_sum_l2379_237994

theorem consecutive_odd_integer_sum {n : ℤ} (h1 : n = 17 ∨ n + 2 = 17) (h2 : n + n + 2 ≥ 36) : (n = 17 → n + 2 = 19) ∧ (n + 2 = 17 → n = 15) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_odd_integer_sum_l2379_237994


namespace NUMINAMATH_GPT_man_speed_42_minutes_7_km_l2379_237974

theorem man_speed_42_minutes_7_km 
  (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ)
  (h1 : distance = 7) 
  (h2 : time_minutes = 42) 
  (h3 : time_hours = time_minutes / 60) :
  distance / time_hours = 10 := by
  sorry

end NUMINAMATH_GPT_man_speed_42_minutes_7_km_l2379_237974


namespace NUMINAMATH_GPT_root_condition_l2379_237982

noncomputable def f (a : ℝ) (x : ℝ) := a * x^3 - 3 * x^2 + 1

theorem root_condition (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 0 ∧ ∀ x ≠ x₀, f a x ≠ 0 ∧ x₀ < 0) → a > 2 :=
sorry

end NUMINAMATH_GPT_root_condition_l2379_237982


namespace NUMINAMATH_GPT_f_positive_l2379_237961

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x / Real.log 2

variables (x0 x1 : ℝ)

theorem f_positive (hx0 : f x0 = 0) (hx1 : 0 < x1) (hx0_gt_x1 : x1 < x0) : 0 < f x1 :=
sorry

end NUMINAMATH_GPT_f_positive_l2379_237961


namespace NUMINAMATH_GPT_polynomial_factors_integers_l2379_237989

theorem polynomial_factors_integers (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 500)
  (h₃ : ∃ a : ℤ, n = a * (a + 1)) :
  n ≤ 21 :=
sorry

end NUMINAMATH_GPT_polynomial_factors_integers_l2379_237989


namespace NUMINAMATH_GPT_arithmetic_mean_six_expressions_l2379_237937

theorem arithmetic_mean_six_expressions (x : ℝ) :
  (x + 10 + 17 + 2 * x + 15 + 2 * x + 6 + 3 * x - 5) / 6 = 30 →
  x = 137 / 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_six_expressions_l2379_237937


namespace NUMINAMATH_GPT_simplify_exponential_expression_l2379_237973

theorem simplify_exponential_expression :
  (3 * (-5)^2)^(3/4) = (75)^(3/4) := 
  sorry

end NUMINAMATH_GPT_simplify_exponential_expression_l2379_237973


namespace NUMINAMATH_GPT_daily_avg_for_entire_month_is_correct_l2379_237987

-- conditions
def avg_first_25_days := 63
def days_first_25 := 25
def avg_last_5_days := 33
def days_last_5 := 5
def total_days := days_first_25 + days_last_5

-- question: What is the daily average for the entire month?
theorem daily_avg_for_entire_month_is_correct : 
  (avg_first_25_days * days_first_25 + avg_last_5_days * days_last_5) / total_days = 58 := by
  sorry

end NUMINAMATH_GPT_daily_avg_for_entire_month_is_correct_l2379_237987


namespace NUMINAMATH_GPT_initial_solution_weight_100kg_l2379_237932

theorem initial_solution_weight_100kg
  (W : ℝ)
  (initial_salt_percentage : ℝ)
  (added_salt : ℝ)
  (final_salt_percentage : ℝ)
  (H1 : initial_salt_percentage = 0.10)
  (H2 : added_salt = 12.5)
  (H3 : final_salt_percentage = 0.20)
  (H4 : 0.20 * (W + 12.5) = 0.10 * W + 12.5) :
  W = 100 :=   
by 
  sorry

end NUMINAMATH_GPT_initial_solution_weight_100kg_l2379_237932


namespace NUMINAMATH_GPT_dandelions_initial_l2379_237971

theorem dandelions_initial (y w : ℕ) (h1 : y + w = 35) (h2 : y - 2 = 2 * (w - 6)) : y = 20 ∧ w = 15 :=
by
  sorry

end NUMINAMATH_GPT_dandelions_initial_l2379_237971


namespace NUMINAMATH_GPT_walt_part_time_job_l2379_237906

theorem walt_part_time_job (x : ℝ) 
  (h1 : 0.09 * x + 0.08 * 4000 = 770) : 
  x + 4000 = 9000 := by
  sorry

end NUMINAMATH_GPT_walt_part_time_job_l2379_237906


namespace NUMINAMATH_GPT_fixed_point_coordinates_l2379_237956

theorem fixed_point_coordinates (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : (2, 2) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^(x-2) + 1)} := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fixed_point_coordinates_l2379_237956


namespace NUMINAMATH_GPT_find_complex_number_l2379_237902

open Complex

theorem find_complex_number (a b : ℝ) (z : ℂ) 
  (h₁ : (∀ b: ℝ, (b^2 + 4 * b + 4 = 0) ∧ (b + a = 0))) :
  z = 2 - 2 * Complex.I :=
  sorry

end NUMINAMATH_GPT_find_complex_number_l2379_237902


namespace NUMINAMATH_GPT_find_n_l2379_237930

noncomputable def condition (n : ℕ) : Prop :=
  (1/5)^n * (1/4)^18 = 1 / (2 * 10^35)

theorem find_n (n : ℕ) (h : condition n) : n = 35 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2379_237930


namespace NUMINAMATH_GPT_combine_like_terms_l2379_237992

variable (a : ℝ)

theorem combine_like_terms : 3 * a^2 + 5 * a^2 - a^2 = 7 * a^2 := 
by sorry

end NUMINAMATH_GPT_combine_like_terms_l2379_237992


namespace NUMINAMATH_GPT_ratio_of_tangent_to_circumference_l2379_237975

theorem ratio_of_tangent_to_circumference
  {r x : ℝ}  -- radius of the circle and length of the tangent
  (hT : x = 2 * π * r)  -- given the length of tangent PQ
  (hA : (1 / 2) * x * r = π * r^2)  -- given the area equivalence

  : (x / (2 * π * r)) = 1 :=  -- desired ratio
by
  -- proof omitted, just using sorry to indicate proof
  sorry

end NUMINAMATH_GPT_ratio_of_tangent_to_circumference_l2379_237975


namespace NUMINAMATH_GPT_car_average_speed_l2379_237995

theorem car_average_speed :
  let distance_uphill := 100
  let distance_downhill := 50
  let speed_uphill := 30
  let speed_downhill := 80
  let total_distance := distance_uphill + distance_downhill
  let time_uphill := distance_uphill / speed_uphill
  let time_downhill := distance_downhill / speed_downhill
  let total_time := time_uphill + time_downhill
  let average_speed := total_distance / total_time
  average_speed = 37.92 := by
  sorry

end NUMINAMATH_GPT_car_average_speed_l2379_237995


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l2379_237966

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ^ 2 - 2 * x₁ + m = 0 ∧ x₂ ^ 2 - 2 * x₂ + m = 0) ↔ m < 1 :=
by sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l2379_237966


namespace NUMINAMATH_GPT_problem_l2379_237908

theorem problem (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
  sorry

end NUMINAMATH_GPT_problem_l2379_237908


namespace NUMINAMATH_GPT_total_people_on_bus_l2379_237988

def initial_people := 4
def added_people := 13

theorem total_people_on_bus : initial_people + added_people = 17 := by
  sorry

end NUMINAMATH_GPT_total_people_on_bus_l2379_237988


namespace NUMINAMATH_GPT_kayla_total_items_l2379_237924

theorem kayla_total_items (Tc : ℕ) (Ts : ℕ) (Kc : ℕ) (Ks : ℕ) 
  (h1 : Tc = 2 * Kc) (h2 : Ts = 2 * Ks) (h3 : Tc = 12) (h4 : Ts = 18) : Kc + Ks = 15 :=
by
  sorry

end NUMINAMATH_GPT_kayla_total_items_l2379_237924


namespace NUMINAMATH_GPT_fourth_person_height_l2379_237938

theorem fourth_person_height (H : ℝ) 
  (h_avg : (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 77) : 
  H + 10 = 83 :=
sorry

end NUMINAMATH_GPT_fourth_person_height_l2379_237938


namespace NUMINAMATH_GPT_part1_part2_l2379_237980

noncomputable def h (x : ℝ) : ℝ := x^2

noncomputable def phi (x : ℝ) : ℝ := 2 * Real.exp 1 * Real.log x

noncomputable def F (x : ℝ) : ℝ := h x - phi x

theorem part1 :
  ∃ (x : ℝ), x > 0 ∧ Real.log x = 1 ∧ F x = 0 :=
sorry

theorem part2 :
  ∃ (k b : ℝ), 
  (∀ x > 0, h x ≥ k * x + b) ∧
  (∀ x > 0, phi x ≤ k * x + b) ∧
  (k = 2 * Real.exp 1 ∧ b = -Real.exp 1) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2379_237980


namespace NUMINAMATH_GPT_fraction_to_percentage_l2379_237929

theorem fraction_to_percentage (x : ℝ) (hx : 0 < x) : 
  (x / 50 + x / 25) = 0.06 * x := 
sorry

end NUMINAMATH_GPT_fraction_to_percentage_l2379_237929


namespace NUMINAMATH_GPT_part1_part2_l2379_237927

theorem part1 (x : ℝ) : 3 + 2 * x > - x - 6 ↔ x > -3 := by
  sorry

theorem part2 (x : ℝ) : 2 * x + 1 ≤ x + 3 ∧ (2 * x + 1) / 3 > 1 ↔ 1 < x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2379_237927


namespace NUMINAMATH_GPT_integer_pairs_solution_l2379_237946

theorem integer_pairs_solution (k : ℕ) (h : k ≠ 1) : 
  ∃ (m n : ℤ), 
    ((m - n) ^ 2 = 4 * m * n / (m + n - 1)) ∧ 
    (m = k^2 + k / 2 ∧ n = k^2 - k / 2) ∨ 
    (m = k^2 - k / 2 ∧ n = k^2 + k / 2) :=
sorry

end NUMINAMATH_GPT_integer_pairs_solution_l2379_237946


namespace NUMINAMATH_GPT_initial_walking_speed_l2379_237920

theorem initial_walking_speed
  (t : ℝ) -- Time in minutes for bus to reach the bus stand from when the person starts walking
  (h₁ : 5 = 5 * ((t - 5) / 60)) -- When walking at 5 km/h, person reaches 5 minutes early
  (h₂ : 5 = v * ((t + 10) / 60)) -- At initial speed v, person misses the bus by 10 minutes
  : v = 4 := 
by
  sorry

end NUMINAMATH_GPT_initial_walking_speed_l2379_237920


namespace NUMINAMATH_GPT_compute_large_expression_l2379_237910

theorem compute_large_expression :
  ( (11^4 + 484) * (23^4 + 484) * (35^4 + 484) * (47^4 + 484) * (59^4 + 484) ) / 
  ( (5^4 + 484) * (17^4 + 484) * (29^4 + 484) * (41^4 + 484) * (53^4 + 484) ) = 552.42857 := 
sorry

end NUMINAMATH_GPT_compute_large_expression_l2379_237910


namespace NUMINAMATH_GPT_operations_equivalent_l2379_237923

theorem operations_equivalent (x : ℚ) : 
  ((x * (5 / 6)) / (2 / 3) - 2) = (x * (5 / 4) - 2) :=
sorry

end NUMINAMATH_GPT_operations_equivalent_l2379_237923


namespace NUMINAMATH_GPT_triangle_angle_sum_l2379_237998

theorem triangle_angle_sum (A B C : Type) (angle_ABC angle_BAC angle_ACB : ℝ)
  (h₁ : angle_ABC = 110)
  (h₂ : angle_BAC = 45)
  (triangle_sum : angle_ABC + angle_BAC + angle_ACB = 180) :
  angle_ACB = 25 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l2379_237998


namespace NUMINAMATH_GPT_minimum_distance_AB_l2379_237922

-- Definitions of the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2 - y + 1 = 0
def C2 (x y : ℝ) : Prop := y^2 - x + 1 = 0

theorem minimum_distance_AB :
  ∃ (A B : ℝ × ℝ), C1 A.1 A.2 ∧ C2 B.1 B.2 ∧ dist A B = 3*Real.sqrt 2 / 4 := sorry

end NUMINAMATH_GPT_minimum_distance_AB_l2379_237922


namespace NUMINAMATH_GPT_dealer_selling_price_above_cost_l2379_237911

variable (cost_price : ℝ := 100)
variable (discount_percent : ℝ := 20)
variable (profit_percent : ℝ := 20)

theorem dealer_selling_price_above_cost :
  ∀ (x : ℝ), 
  (0.8 * x = 1.2 * cost_price) → 
  x = cost_price * (1 + profit_percent / 100) :=
by
  sorry

end NUMINAMATH_GPT_dealer_selling_price_above_cost_l2379_237911


namespace NUMINAMATH_GPT_exists_two_elements_l2379_237949

variable (F : Finset (Finset ℕ))
variable (h1 : ∀ (A B : Finset ℕ), A ∈ F → B ∈ F → (A ∪ B) ∈ F)
variable (h2 : ∀ (A : Finset ℕ), A ∈ F → ¬ (3 ∣ A.card))

theorem exists_two_elements : ∃ (x y : ℕ), ∀ (A : Finset ℕ), A ∈ F → x ∈ A ∨ y ∈ A :=
by
  sorry

end NUMINAMATH_GPT_exists_two_elements_l2379_237949


namespace NUMINAMATH_GPT_percent_savings_correct_l2379_237959

theorem percent_savings_correct :
  let cost_of_package := 9
  let num_of_rolls_in_package := 12
  let cost_per_roll_individually := 1
  let cost_per_roll_in_package := cost_of_package / num_of_rolls_in_package
  let savings_per_roll := cost_per_roll_individually - cost_per_roll_in_package
  let percent_savings := (savings_per_roll / cost_per_roll_individually) * 100
  percent_savings = 25 :=
by
  sorry

end NUMINAMATH_GPT_percent_savings_correct_l2379_237959


namespace NUMINAMATH_GPT_angles_geometric_sequence_count_l2379_237901

def is_geometric_sequence (a b c : ℝ) : Prop :=
  (a = b * c) ∨ (b = a * c) ∨ (c = a * b)

theorem angles_geometric_sequence_count : 
  ∃! (angles : Finset ℝ), 
    (∀ θ ∈ angles, 0 < θ ∧ θ < 2 * Real.pi ∧ ¬∃ k : ℤ, θ = k * (Real.pi / 2)) ∧
    ∀ θ ∈ angles,
      is_geometric_sequence (Real.sin θ ^ 2) (Real.cos θ) (Real.tan θ) ∧
    angles.card = 2 := 
sorry

end NUMINAMATH_GPT_angles_geometric_sequence_count_l2379_237901


namespace NUMINAMATH_GPT_find_x_set_l2379_237960

theorem find_x_set (a : ℝ) (h : 0 < a ∧ a < 1) : 
  {x : ℝ | a ^ (x + 3) > a ^ (2 * x)} = {x : ℝ | x > 3} :=
sorry

end NUMINAMATH_GPT_find_x_set_l2379_237960


namespace NUMINAMATH_GPT_problem1_problem2_l2379_237933

-- Define the given sets A and B
def setA (a : ℝ) : Set ℝ := { x | a - 4 < x ∧ x < a + 4 }
def setB : Set ℝ := { x | x < -1 ∨ x > 5 }

-- Problem 1: Prove A ∩ B = { x | -3 < x ∧ x < -1 } when a = 1
theorem problem1 (a : ℝ) (h : a = 1) : 
  (setA a ∩ setB) = { x : ℝ | -3 < x ∧ x < -1 } := sorry

-- Problem 2: Prove range of a given A ∪ B = ℝ is (1, 3)
theorem problem2 (a : ℝ) : 
  (forall x : ℝ, x ∈ (setA a ∪ setB)) ↔ (1 < a ∧ a < 3) := sorry

end NUMINAMATH_GPT_problem1_problem2_l2379_237933
