import Mathlib

namespace NUMINAMATH_GPT_tangent_value_prism_QABC_l866_86632

-- Assuming R is the radius of the sphere and considering the given conditions
variables {R x : ℝ} (P Q A B C M H : Type)

-- Given condition: Angle between lateral face and base of prism P-ABC is 45 degrees
def angle_PABC : ℝ := 45
-- Required to prove: tan(angle between lateral face and base of prism Q-ABC) = 4
def tangent_QABC : ℝ := 4

theorem tangent_value_prism_QABC
  (h1 : angle_PABC = 45)
  (h2 : 5 * x - 2 * R = 0) -- Derived condition from the solution
  (h3 : x = 2 * R / 5) -- x, the distance calculation
: tangent_QABC = 4 := by
  sorry

end NUMINAMATH_GPT_tangent_value_prism_QABC_l866_86632


namespace NUMINAMATH_GPT_probability_of_ace_then_spade_l866_86630

theorem probability_of_ace_then_spade :
  let P := (1 / 52) * (12 / 51) + (3 / 52) * (13 / 51)
  P = (3 / 127) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_ace_then_spade_l866_86630


namespace NUMINAMATH_GPT_black_area_fraction_after_four_changes_l866_86638

/-- 
Problem: Prove that after four changes, the fractional part of the original black area 
remaining black in an equilateral triangle is 81/256, given that each change splits the 
triangle into 4 smaller congruent equilateral triangles, and one of those turns white.
-/

theorem black_area_fraction_after_four_changes :
  (3 / 4) ^ 4 = 81 / 256 := sorry

end NUMINAMATH_GPT_black_area_fraction_after_four_changes_l866_86638


namespace NUMINAMATH_GPT_no_triangles_with_geometric_progression_angles_l866_86695

theorem no_triangles_with_geometric_progression_angles :
  ¬ ∃ (a r : ℕ), a ≥ 10 ∧ (a + a * r + a * r^2 = 180) ∧ (a ≠ a * r) ∧ (a ≠ a * r^2) ∧ (a * r ≠ a * r^2) :=
sorry

end NUMINAMATH_GPT_no_triangles_with_geometric_progression_angles_l866_86695


namespace NUMINAMATH_GPT_rectangular_box_proof_l866_86661

noncomputable def rectangular_box_surface_area
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 140)
  (h2 : a^2 + b^2 + c^2 = 21^2) : ℝ :=
2 * (a * b + b * c + c * a)

theorem rectangular_box_proof
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 140)
  (h2 : a^2 + b^2 + c^2 = 21^2) :
  rectangular_box_surface_area a b c h1 h2 = 784 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_box_proof_l866_86661


namespace NUMINAMATH_GPT_trade_ratio_blue_per_red_l866_86626

-- Define the problem conditions
def initial_total_marbles : ℕ := 10
def blue_percentage : ℕ := 40
def kept_red_marbles : ℕ := 1
def final_total_marbles : ℕ := 15

-- Find the number of blue marbles initially
def initial_blue_marbles : ℕ := (blue_percentage * initial_total_marbles) / 100

-- Calculate the number of red marbles initially
def initial_red_marbles : ℕ := initial_total_marbles - initial_blue_marbles

-- Calculate the number of red marbles traded
def traded_red_marbles : ℕ := initial_red_marbles - kept_red_marbles

-- Calculate the number of marbles received from the trade
def traded_marbles : ℕ := final_total_marbles - (initial_blue_marbles + kept_red_marbles)

-- The number of blue marbles received per each red marble traded
def blue_per_red : ℕ := traded_marbles / traded_red_marbles

-- Theorem stating that Pete's friend trades 2 blue marbles for each red marble
theorem trade_ratio_blue_per_red : blue_per_red = 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_trade_ratio_blue_per_red_l866_86626


namespace NUMINAMATH_GPT_proof_problem_l866_86690

def p : Prop := ∃ k : ℕ, 0 = 2 * k
def q : Prop := ∃ k : ℕ, 3 = 2 * k

theorem proof_problem : p ∨ q :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l866_86690


namespace NUMINAMATH_GPT_stuart_initial_marbles_is_56_l866_86651

-- Define the initial conditions
def betty_initial_marbles : ℕ := 60
def percentage_given_to_stuart : ℚ := 40 / 100
def stuart_marbles_after_receiving : ℕ := 80

-- Define the calculation of how many marbles Betty gave to Stuart
def marbles_given_to_stuart := (percentage_given_to_stuart * betty_initial_marbles)

-- Define the target: Stuart's initial number of marbles
def stuart_initial_marbles := stuart_marbles_after_receiving - marbles_given_to_stuart

-- Main theorem stating the problem
theorem stuart_initial_marbles_is_56 : stuart_initial_marbles = 56 :=
by 
  sorry

end NUMINAMATH_GPT_stuart_initial_marbles_is_56_l866_86651


namespace NUMINAMATH_GPT_solution_set_inequality_l866_86605

theorem solution_set_inequality (a : ℕ) (h : ∀ x : ℝ, (a-2) * x > (a-2) → x < 1) : a = 0 ∨ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l866_86605


namespace NUMINAMATH_GPT_parallel_lines_slope_l866_86608

theorem parallel_lines_slope {m : ℝ} : 
  (∃ m, (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 → 6 * x + m * y + 14 = 0)) ↔ m = 8 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l866_86608


namespace NUMINAMATH_GPT_range_of_a_l866_86623

theorem range_of_a (a : ℝ) (H : ∀ x : ℝ, x ≤ 1 → 4 - a * 2^x > 0) : a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l866_86623


namespace NUMINAMATH_GPT_distance_to_bus_stand_l866_86606

variable (D : ℝ)

theorem distance_to_bus_stand :
  (D / 4 - D / 5 = 1 / 4) → D = 5 :=
sorry

end NUMINAMATH_GPT_distance_to_bus_stand_l866_86606


namespace NUMINAMATH_GPT_goats_difference_l866_86614

-- Definitions of Adam's, Andrew's, and Ahmed's goats
def adam_goats : ℕ := 7
def andrew_goats : ℕ := 2 * adam_goats + 5
def ahmed_goats : ℕ := 13

-- The theorem to prove the difference in goats
theorem goats_difference : andrew_goats - ahmed_goats = 6 :=
by
  sorry

end NUMINAMATH_GPT_goats_difference_l866_86614


namespace NUMINAMATH_GPT_options_implication_l866_86674

theorem options_implication (a b : ℝ) :
  ((b > 0 ∧ a < 0) ∨ (a < 0 ∧ b < 0 ∧ a > b) ∨ (a > 0 ∧ b > 0 ∧ a > b)) → (1 / a < 1 / b) :=
by sorry

end NUMINAMATH_GPT_options_implication_l866_86674


namespace NUMINAMATH_GPT_new_average_l866_86610

theorem new_average (avg : ℕ) (n : ℕ) (k : ℕ) (new_avg : ℕ) 
  (h1 : avg = 23) (h2 : n = 10) (h3 : k = 4) : 
  new_avg = (n * avg + n * k) / n → new_avg = 27 :=
by
  intro H
  sorry

end NUMINAMATH_GPT_new_average_l866_86610


namespace NUMINAMATH_GPT_remainder_1234567_div_145_l866_86654

theorem remainder_1234567_div_145 : 1234567 % 145 = 67 := by
  sorry

end NUMINAMATH_GPT_remainder_1234567_div_145_l866_86654


namespace NUMINAMATH_GPT_find_a_l866_86676

theorem find_a (a : ℝ) (h : Nat.choose 5 2 * (-a)^3 = 10) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l866_86676


namespace NUMINAMATH_GPT_marble_problem_l866_86678

theorem marble_problem {r b : ℕ} 
  (h1 : 9 * r - b = 27) 
  (h2 : 3 * r - b = 3) : r + b = 13 := 
by
  sorry

end NUMINAMATH_GPT_marble_problem_l866_86678


namespace NUMINAMATH_GPT_participants_count_l866_86639

theorem participants_count (x y : ℕ) 
    (h1 : y = x + 41)
    (h2 : y = 3 * x - 35) : 
    x = 38 ∧ y = 79 :=
by
  sorry

end NUMINAMATH_GPT_participants_count_l866_86639


namespace NUMINAMATH_GPT_fraction_simplification_l866_86628

theorem fraction_simplification : 
  (320 / 18) * (9 / 144) * (4 / 5) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_fraction_simplification_l866_86628


namespace NUMINAMATH_GPT_amount_of_c_l866_86673

theorem amount_of_c (A B C : ℕ) (h1 : A + B + C = 350) (h2 : A + C = 200) (h3 : B + C = 350) : C = 200 :=
sorry

end NUMINAMATH_GPT_amount_of_c_l866_86673


namespace NUMINAMATH_GPT_proof_y_pow_x_equal_1_by_9_l866_86664

theorem proof_y_pow_x_equal_1_by_9 
  (x y : ℝ)
  (h : (x - 2)^2 + abs (y + 1/3) = 0) :
  y^x = 1/9 := by
  sorry

end NUMINAMATH_GPT_proof_y_pow_x_equal_1_by_9_l866_86664


namespace NUMINAMATH_GPT_original_length_in_meters_l866_86642

-- Conditions
def erased_length : ℝ := 10 -- 10 cm
def remaining_length : ℝ := 90 -- 90 cm

-- Question: What is the original length of the line in meters?
theorem original_length_in_meters : (remaining_length + erased_length) / 100 = 1 := 
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_original_length_in_meters_l866_86642


namespace NUMINAMATH_GPT_positive_number_property_l866_86679

theorem positive_number_property (x : ℝ) (h : (100 - x) / 100 * x = 16) :
  x = 40 ∨ x = 60 :=
sorry

end NUMINAMATH_GPT_positive_number_property_l866_86679


namespace NUMINAMATH_GPT_incorrect_line_pass_through_Q_l866_86611

theorem incorrect_line_pass_through_Q (a b : ℝ) : 
  (∀ (k : ℝ), ∃ (Q : ℝ × ℝ), Q = (0, b) ∧ y = k * x + b) →
  (¬ ∃ k : ℝ, ∀ y x, y = k * x + b ∧ x = 0)
:= 
sorry

end NUMINAMATH_GPT_incorrect_line_pass_through_Q_l866_86611


namespace NUMINAMATH_GPT_necessarily_positive_l866_86671

theorem necessarily_positive (x y w : ℝ) (h1 : 0 < x ∧ x < 0.5) (h2 : -0.5 < y ∧ y < 0) (h3 : 0.5 < w ∧ w < 1) : 
  0 < w - y :=
sorry

end NUMINAMATH_GPT_necessarily_positive_l866_86671


namespace NUMINAMATH_GPT_equal_student_distribution_l866_86653

theorem equal_student_distribution
  (students_bus1_initial : ℕ)
  (students_bus2_initial : ℕ)
  (students_to_move : ℕ)
  (students_bus1_final : ℕ)
  (students_bus2_final : ℕ)
  (total_students : ℕ) :
  students_bus1_initial = 57 →
  students_bus2_initial = 31 →
  total_students = students_bus1_initial + students_bus2_initial →
  students_to_move = 13 →
  students_bus1_final = students_bus1_initial - students_to_move →
  students_bus2_final = students_bus2_initial + students_to_move →
  students_bus1_final = 44 ∧ students_bus2_final = 44 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_equal_student_distribution_l866_86653


namespace NUMINAMATH_GPT_quadratic_root_exists_in_range_l866_86680

theorem quadratic_root_exists_in_range :
  ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ x^2 + 3 * x - 5 = 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_root_exists_in_range_l866_86680


namespace NUMINAMATH_GPT_jeremy_tylenol_duration_l866_86637

theorem jeremy_tylenol_duration (num_pills : ℕ) (pill_mg : ℕ) (dose_mg : ℕ) (hours_per_dose : ℕ) (hours_per_day : ℕ) 
  (total_tylenol_mg : ℕ := num_pills * pill_mg)
  (num_doses : ℕ := total_tylenol_mg / dose_mg)
  (total_hours : ℕ := num_doses * hours_per_dose) :
  num_pills = 112 → pill_mg = 500 → dose_mg = 1000 → hours_per_dose = 6 → hours_per_day = 24 → 
  total_hours / hours_per_day = 14 := 
by 
  intros; 
  sorry

end NUMINAMATH_GPT_jeremy_tylenol_duration_l866_86637


namespace NUMINAMATH_GPT_tan_theta_solution_l866_86649

theorem tan_theta_solution (θ : ℝ) (h : 2 * Real.sin θ = 1 + Real.cos θ) :
  Real.tan θ = 0 ∨ Real.tan θ = 4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_theta_solution_l866_86649


namespace NUMINAMATH_GPT_average_weight_increase_l866_86683

theorem average_weight_increase (A : ℝ) (X : ℝ) (h : (8 * A - 65 + 93) / 8 = A + X) :
  X = 3.5 :=
sorry

end NUMINAMATH_GPT_average_weight_increase_l866_86683


namespace NUMINAMATH_GPT_min_value_frac_sum_l866_86691

open Real

theorem min_value_frac_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) :
  3 ≤ (1 / (x + y)) + ((x + y) / z) := 
sorry

end NUMINAMATH_GPT_min_value_frac_sum_l866_86691


namespace NUMINAMATH_GPT_distinct_remainders_l866_86621

theorem distinct_remainders
  (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n)
  (h_div : n ∣ a^n - 1) :
  ∀ i j : ℕ, i ∈ (Finset.range n).image (· + 1) →
            j ∈ (Finset.range n).image (· + 1) →
            (a^i + i) % n = (a^j + j) % n →
            i = j :=
by
  intros i j hi hj h
  sorry

end NUMINAMATH_GPT_distinct_remainders_l866_86621


namespace NUMINAMATH_GPT_paperback_copies_sold_l866_86613

theorem paperback_copies_sold
  (H : ℕ) (P : ℕ)
  (h1 : H = 36000)
  (h2 : P = 9 * H)
  (h3 : H + P = 440000) :
  P = 360000 := by
  sorry

end NUMINAMATH_GPT_paperback_copies_sold_l866_86613


namespace NUMINAMATH_GPT_range_of_a_l866_86633

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a - 1)*x - 1 < 0
def r (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1 ∨ a > 3

theorem range_of_a (a : ℝ) (h : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : r a := 
by sorry

end NUMINAMATH_GPT_range_of_a_l866_86633


namespace NUMINAMATH_GPT_old_conveyor_time_l866_86624

theorem old_conveyor_time (x : ℝ) : 
  (1 / x) + (1 / 15) = 1 / 8.75 → 
  x = 21 := 
by 
  intro h 
  sorry

end NUMINAMATH_GPT_old_conveyor_time_l866_86624


namespace NUMINAMATH_GPT_interval_of_monotonic_decrease_range_of_k_l866_86601
open Real

noncomputable def f (x : ℝ) : ℝ := 
  let m := (sqrt 3 * sin (x / 4), 1)
  let n := (cos (x / 4), cos (x / 2))
  m.1 * n.1 + m.2 * n.2 -- vector dot product

-- Prove the interval of monotonic decrease for f(x)
theorem interval_of_monotonic_decrease (k : ℤ) : 
  4 * k * π + 2 * π / 3 ≤ x ∧ x ≤ 4 * k * π + 8 * π / 3 → f x = sin (x / 2 + π / 6) + 1 / 2 :=
sorry

-- Prove the range of k such that the zero condition is satisfied for g(x) - k
theorem range_of_k (k : ℝ) :
  0 ≤ k ∧ k ≤ 3 / 2 → ∃ x ∈ [0, 7 * π / 3], (sin (x / 2 - π / 6) + 1 / 2) - k = 0 :=
sorry

end NUMINAMATH_GPT_interval_of_monotonic_decrease_range_of_k_l866_86601


namespace NUMINAMATH_GPT_constant_term_expansion_l866_86657

theorem constant_term_expansion (x : ℝ) (n : ℕ) (h : (x + 2 + 1/x)^n = 20) : n = 3 :=
by
sorry

end NUMINAMATH_GPT_constant_term_expansion_l866_86657


namespace NUMINAMATH_GPT_mark_age_in_5_years_l866_86656

-- Definitions based on the conditions
def Amy_age := 15
def age_difference := 7

-- Statement specifying the age Mark will be in 5 years
theorem mark_age_in_5_years : (Amy_age + age_difference + 5) = 27 := 
by
  sorry

end NUMINAMATH_GPT_mark_age_in_5_years_l866_86656


namespace NUMINAMATH_GPT_amount_spent_on_machinery_l866_86646

-- Define the given conditions
def raw_materials_spent : ℤ := 80000
def total_amount : ℤ := 137500
def cash_spent : ℤ := (20 * total_amount) / 100

-- The goal is to prove the amount spent on machinery
theorem amount_spent_on_machinery : 
  ∃ M : ℤ, raw_materials_spent + M + cash_spent = total_amount ∧ M = 30000 := by
  sorry

end NUMINAMATH_GPT_amount_spent_on_machinery_l866_86646


namespace NUMINAMATH_GPT_expression_simplification_l866_86629

theorem expression_simplification : (2^2020 + 2^2018) / (2^2020 - 2^2018) = 5 / 3 := 
by 
    sorry

end NUMINAMATH_GPT_expression_simplification_l866_86629


namespace NUMINAMATH_GPT_total_fish_l866_86697

theorem total_fish (fish_lilly fish_rosy : ℕ) (hl : fish_lilly = 10) (hr : fish_rosy = 14) :
  fish_lilly + fish_rosy = 24 := 
by 
  sorry

end NUMINAMATH_GPT_total_fish_l866_86697


namespace NUMINAMATH_GPT_scientific_notation_of_0_0000025_l866_86652

theorem scientific_notation_of_0_0000025 :
  0.0000025 = 2.5 * 10^(-6) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_0_0000025_l866_86652


namespace NUMINAMATH_GPT_distinct_possible_lunches_l866_86662

namespace SchoolCafeteria

def main_courses : List String := ["Hamburger", "Veggie Burger", "Chicken Sandwich", "Pasta"]
def beverages_when_meat_free : List String := ["Water", "Soda"]
def beverages_when_meat : List String := ["Water"]
def snacks : List String := ["Apple Pie", "Fruit Cup"]

-- Count the total number of distinct possible lunches
def count_distinct_lunches : Nat :=
  let count_options (main_course : String) : Nat :=
    if main_course = "Hamburger" ∨ main_course = "Chicken Sandwich" then
      beverages_when_meat.length * snacks.length
    else
      beverages_when_meat_free.length * snacks.length
  (main_courses.map count_options).sum

theorem distinct_possible_lunches : count_distinct_lunches = 12 := by
  sorry

end SchoolCafeteria

end NUMINAMATH_GPT_distinct_possible_lunches_l866_86662


namespace NUMINAMATH_GPT_total_votes_cast_l866_86609

-- Problem statement and conditions
variable (V : ℝ) (candidateVotes : ℝ) (rivalVotes : ℝ)
variable (h1 : candidateVotes = 0.35 * V)
variable (h2 : rivalVotes = candidateVotes + 1350)

-- Target to prove
theorem total_votes_cast : V = 4500 := by
  -- pseudo code proof would be filled here in real Lean environment
  sorry

end NUMINAMATH_GPT_total_votes_cast_l866_86609


namespace NUMINAMATH_GPT_sequence_12th_term_l866_86644

theorem sequence_12th_term (C : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = C / a n) (h4 : C = 12) : a 12 = 4 :=
sorry

end NUMINAMATH_GPT_sequence_12th_term_l866_86644


namespace NUMINAMATH_GPT_max_participants_won_at_least_three_matches_l866_86692

theorem max_participants_won_at_least_three_matches :
  ∀ (n : ℕ), n = 200 → ∃ k : ℕ, k ≤ 66 ∧ ∀ p : ℕ, (p ≥ k ∧ p > 66) → false := by
  sorry

end NUMINAMATH_GPT_max_participants_won_at_least_three_matches_l866_86692


namespace NUMINAMATH_GPT_probability_of_four_of_a_kind_is_correct_l866_86686

noncomputable def probability_four_of_a_kind: ℚ :=
  let total_ways := Nat.choose 52 5
  let successful_ways := 13 * 1 * 12 * 4
  (successful_ways: ℚ) / (total_ways: ℚ)

theorem probability_of_four_of_a_kind_is_correct :
  probability_four_of_a_kind = 13 / 54145 := 
by
  -- sorry is used because we are only writing the statement, no proof required
  sorry

end NUMINAMATH_GPT_probability_of_four_of_a_kind_is_correct_l866_86686


namespace NUMINAMATH_GPT_preston_receives_total_amount_l866_86659

theorem preston_receives_total_amount :
  let price_per_sandwich := 5
  let delivery_fee := 20
  let num_sandwiches := 18
  let tip_percent := 0.10
  let sandwich_cost := num_sandwiches * price_per_sandwich
  let initial_total := sandwich_cost + delivery_fee
  let tip := initial_total * tip_percent
  let final_total := initial_total + tip
  final_total = 121 := 
by
  sorry

end NUMINAMATH_GPT_preston_receives_total_amount_l866_86659


namespace NUMINAMATH_GPT_operation_B_is_not_algorithm_l866_86688

-- Define what constitutes an algorithm.
def is_algorithm (desc : String) : Prop :=
  desc = "clear and finite steps to solve a certain type of problem"

-- Define given operations.
def operation_A : String := "Calculating the area of a circle given its radius"
def operation_B : String := "Calculating the possibility of reaching 24 by randomly drawing 4 playing cards"
def operation_C : String := "Finding the equation of a line given two points in the coordinate plane"
def operation_D : String := "Operations of addition, subtraction, multiplication, and division"

-- Define expected property of an algorithm.
def is_algorithm_A : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"
def is_algorithm_B : Prop := is_algorithm "cannot describe precise steps"
def is_algorithm_C : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"
def is_algorithm_D : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"

theorem operation_B_is_not_algorithm :
  ¬ (is_algorithm operation_B) :=
by
   -- Change this line to the theorem proof.
   sorry

end NUMINAMATH_GPT_operation_B_is_not_algorithm_l866_86688


namespace NUMINAMATH_GPT_find_unknown_towel_rate_l866_86655

theorem find_unknown_towel_rate 
    (cost_known1 : ℕ := 300)
    (cost_known2 : ℕ := 750)
    (total_towels : ℕ := 10)
    (average_price : ℕ := 150)
    (total_cost : ℕ := total_towels * average_price) :
  let total_cost_known := cost_known1 + cost_known2
  let cost_unknown := 2 * x
  300 + 750 + 2 * x = total_cost → x = 225 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_towel_rate_l866_86655


namespace NUMINAMATH_GPT_john_trip_time_l866_86618

theorem john_trip_time (x : ℝ) (h : x + 2 * x + 2 * x = 10) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_trip_time_l866_86618


namespace NUMINAMATH_GPT_minimum_employees_needed_l866_86648

-- Conditions
def water_monitors : ℕ := 95
def air_monitors : ℕ := 80
def soil_monitors : ℕ := 45
def water_and_air : ℕ := 30
def air_and_soil : ℕ := 20
def water_and_soil : ℕ := 15
def all_three : ℕ := 10

-- Theorems/Goals
theorem minimum_employees_needed 
  (water : ℕ := water_monitors)
  (air : ℕ := air_monitors)
  (soil : ℕ := soil_monitors)
  (water_air : ℕ := water_and_air)
  (air_soil : ℕ := air_and_soil)
  (water_soil : ℕ := water_and_soil)
  (all_3 : ℕ := all_three) :
  water + air + soil - water_air - air_soil - water_soil + all_3 = 165 :=
by
  sorry

end NUMINAMATH_GPT_minimum_employees_needed_l866_86648


namespace NUMINAMATH_GPT_reggie_games_lost_l866_86600

-- Given conditions:
def initial_marbles : ℕ := 100
def marbles_per_game : ℕ := 10
def games_played : ℕ := 9
def marbles_after_games : ℕ := 90

-- The statement to prove:
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / marbles_per_game = 1 := 
sorry

end NUMINAMATH_GPT_reggie_games_lost_l866_86600


namespace NUMINAMATH_GPT_time_after_seconds_l866_86615

def initial_time : Nat := 8 * 60 * 60 -- 8:00:00 a.m. in seconds
def seconds_passed : Nat := 8035
def target_time : Nat := (10 * 60 * 60 + 13 * 60 + 35) -- 10:13:35 in seconds

theorem time_after_seconds : initial_time + seconds_passed = target_time := by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_time_after_seconds_l866_86615


namespace NUMINAMATH_GPT_discount_price_l866_86616

theorem discount_price (a : ℝ) (original_price : ℝ) (sold_price : ℝ) :
  original_price = 200 ∧ sold_price = 148 → (original_price * (1 - a/100) * (1 - a/100) = sold_price) :=
by
  sorry

end NUMINAMATH_GPT_discount_price_l866_86616


namespace NUMINAMATH_GPT_sum_x1_x2_l866_86622

open ProbabilityTheory

variable {Ω : Type*} {X : Ω → ℝ}
variable (p1 p2 : ℝ) (x1 x2 : ℝ)
variable (h1 : 2/3 * x1 + 1/3 * x2 = 4/9)
variable (h2 : 2/3 * (x1 - 4/9)^2 + 1/3 * (x2 - 4/9)^2 = 2)
variable (h3 : x1 < x2)

theorem sum_x1_x2 : x1 + x2 = 17/9 :=
by
  sorry

end NUMINAMATH_GPT_sum_x1_x2_l866_86622


namespace NUMINAMATH_GPT_woody_saving_weeks_l866_86640

variable (cost_needed current_savings weekly_allowance : ℕ)

theorem woody_saving_weeks (h₁ : cost_needed = 282)
                           (h₂ : current_savings = 42)
                           (h₃ : weekly_allowance = 24) :
  (cost_needed - current_savings) / weekly_allowance = 10 := by
  sorry

end NUMINAMATH_GPT_woody_saving_weeks_l866_86640


namespace NUMINAMATH_GPT_find_white_balls_l866_86658

-- Define a structure to hold the probabilities and total balls
structure BallProperties where
  totalBalls : Nat
  probRed : Real
  probBlack : Real

-- Given data as conditions
def givenData : BallProperties := 
  { totalBalls := 50, probRed := 0.15, probBlack := 0.45 }

-- The statement to prove the number of white balls
theorem find_white_balls (data : BallProperties) : 
  data.totalBalls = 50 →
  data.probRed = 0.15 →
  data.probBlack = 0.45 →
  ∃ whiteBalls : Nat, whiteBalls = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_white_balls_l866_86658


namespace NUMINAMATH_GPT_coin_count_l866_86602

-- Define the conditions and the proof goal
theorem coin_count (total_value : ℕ) (coin_value_20 : ℕ) (coin_value_25 : ℕ) 
    (num_20_paise_coins : ℕ) (total_value_paise : total_value = 7100)
    (value_20_paise : coin_value_20 = 20) (value_25_paise : coin_value_25 = 25)
    (num_20_paise : num_20_paise_coins = 300) : 
    (300 + 44 = 344) :=
by
  -- The proof would go here, currently omitted with sorry
  sorry

end NUMINAMATH_GPT_coin_count_l866_86602


namespace NUMINAMATH_GPT_find_y_of_set_with_mean_l866_86665

theorem find_y_of_set_with_mean (y : ℝ) (h : ((8 + 15 + 20 + 6 + y) / 5 = 12)) : y = 11 := 
by 
    sorry

end NUMINAMATH_GPT_find_y_of_set_with_mean_l866_86665


namespace NUMINAMATH_GPT_second_digging_breadth_l866_86641

theorem second_digging_breadth :
  ∀ (A B depth1 length1 breadth1 depth2 length2 : ℕ),
  (A / B) = 1 → -- Assuming equal number of days and people
  depth1 = 100 → length1 = 25 → breadth1 = 30 → 
  depth2 = 75 → length2 = 20 → 
  (A = depth1 * length1 * breadth1) → 
  (B = depth2 * length2 * x) →
  x = 50 :=
by sorry

end NUMINAMATH_GPT_second_digging_breadth_l866_86641


namespace NUMINAMATH_GPT_number_of_managers_in_sample_l866_86604

def totalStaff : ℕ := 160
def salespeople : ℕ := 104
def managers : ℕ := 32
def logisticsPersonnel : ℕ := 24
def sampleSize : ℕ := 20

theorem number_of_managers_in_sample : 
  (managers * (sampleSize / totalStaff) = 4) := by
  sorry

end NUMINAMATH_GPT_number_of_managers_in_sample_l866_86604


namespace NUMINAMATH_GPT_base_six_to_ten_2154_l866_86636

def convert_base_six_to_ten (n : ℕ) : ℕ :=
  2 * 6^3 + 1 * 6^2 + 5 * 6^1 + 4 * 6^0

theorem base_six_to_ten_2154 :
  convert_base_six_to_ten 2154 = 502 :=
by
  sorry

end NUMINAMATH_GPT_base_six_to_ten_2154_l866_86636


namespace NUMINAMATH_GPT_find_x_l866_86663

def op (a b : ℕ) : ℕ := a * b - b + b ^ 2

theorem find_x (x : ℕ) : (∃ x : ℕ, op x 8 = 80) :=
  sorry

end NUMINAMATH_GPT_find_x_l866_86663


namespace NUMINAMATH_GPT_colin_speed_l866_86603

variable (B T Br C D : ℝ)

-- Given conditions
axiom cond1 : C = 6 * Br
axiom cond2 : Br = (1/3) * T^2
axiom cond3 : T = 2 * B
axiom cond4 : D = (1/4) * C
axiom cond5 : B = 1

-- Prove Colin's speed C is 8 mph
theorem colin_speed :
  C = 8 :=
by
  sorry

end NUMINAMATH_GPT_colin_speed_l866_86603


namespace NUMINAMATH_GPT_cos_A_eq_a_eq_l866_86689

-- Defining the problem conditions:
variables {A B C a b c : ℝ}
variable (sin_eq : Real.sin (B + C) = 3 * Real.sin (A / 2) ^ 2)
variable (area_eq : 1 / 2 * b * c * Real.sin A = 6)
variable (sum_eq : b + c = 8)
variable (bc_prod_eq : b * c = 13)
variable (cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)

-- Proving the statements:
theorem cos_A_eq : Real.cos A = 5 / 13 :=
sorry

theorem a_eq : a = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_cos_A_eq_a_eq_l866_86689


namespace NUMINAMATH_GPT_ramu_profit_percent_l866_86675

noncomputable def carCost : ℝ := 42000
noncomputable def repairCost : ℝ := 13000
noncomputable def sellingPrice : ℝ := 60900
noncomputable def totalCost : ℝ := carCost + repairCost
noncomputable def profit : ℝ := sellingPrice - totalCost
noncomputable def profitPercent : ℝ := (profit / totalCost) * 100

theorem ramu_profit_percent : profitPercent = 10.73 := 
by
  sorry

end NUMINAMATH_GPT_ramu_profit_percent_l866_86675


namespace NUMINAMATH_GPT_find_point_on_x_axis_l866_86645

theorem find_point_on_x_axis (a : ℝ) (h : abs (3 * a + 6) = 30) : (a = -12) ∨ (a = 8) :=
sorry

end NUMINAMATH_GPT_find_point_on_x_axis_l866_86645


namespace NUMINAMATH_GPT_initially_calculated_average_weight_l866_86607

theorem initially_calculated_average_weight 
  (A : ℚ)
  (h1 : ∀ sum_weight_corr : ℚ, sum_weight_corr = 20 * 58.65)
  (h2 : ∀ misread_weight_corr : ℚ, misread_weight_corr = 56)
  (h3 : ∀ correct_weight_corr : ℚ, correct_weight_corr = 61)
  (h4 : (20 * A + (correct_weight_corr - misread_weight_corr)) = 20 * 58.65) :
  A = 58.4 := 
sorry

end NUMINAMATH_GPT_initially_calculated_average_weight_l866_86607


namespace NUMINAMATH_GPT_expand_polynomial_l866_86681

theorem expand_polynomial (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4 * x + 4) = x^4 + 4 * x^3 - 16 * x - 16 := 
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l866_86681


namespace NUMINAMATH_GPT_decompose_96_l866_86660

theorem decompose_96 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 96) (h4 : a^2 + b^2 = 208) : 
  (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) :=
sorry

end NUMINAMATH_GPT_decompose_96_l866_86660


namespace NUMINAMATH_GPT_smallest_square_length_proof_l866_86668

-- Define square side length required properties
noncomputable def smallest_square_side_length (rect_w rect_h min_side : ℝ) : ℝ :=
  if h : min_side^2 % (rect_w * rect_h) = 0 then min_side 
  else if h : (min_side + 1)^2 % (rect_w * rect_h) = 0 then min_side + 1
  else if h : (min_side + 2)^2 % (rect_w * rect_h) = 0 then min_side + 2
  else if h : (min_side + 3)^2 % (rect_w * rect_h) = 0 then min_side + 3
  else if h : (min_side + 4)^2 % (rect_w * rect_h) = 0 then min_side + 4
  else if h : (min_side + 5)^2 % (rect_w * rect_h) = 0 then min_side + 5
  else if h : (min_side + 6)^2 % (rect_w * rect_h) = 0 then min_side + 6
  else if h : (min_side + 7)^2 % (rect_w * rect_h) = 0 then min_side + 7
  else if h : (min_side + 8)^2 % (rect_w * rect_h) = 0 then min_side + 8
  else if h : (min_side + 9)^2 % (rect_w * rect_h) = 0 then min_side + 9
  else min_side + 2 -- ensuring it can't be less than min_side

-- State the theorem
theorem smallest_square_length_proof : smallest_square_side_length 2 3 10 = 12 :=
by 
  unfold smallest_square_side_length
  norm_num
  sorry

end NUMINAMATH_GPT_smallest_square_length_proof_l866_86668


namespace NUMINAMATH_GPT_impossible_return_l866_86669

def Point := (ℝ × ℝ)

-- Conditions
def is_valid_point (p: Point) : Prop :=
  let (a, b) := p
  ∃ a_int b_int : ℤ, (a = a_int + b_int * Real.sqrt 2 ∧ b = a_int + b_int * Real.sqrt 2)

def valid_movement (p q: Point) : Prop :=
  let (x1, y1) := p
  let (x2, y2) := q
  abs x2 > abs x1 ∧ abs y2 > abs y1 

-- Theorem statement
theorem impossible_return (start: Point) (h: start = (1, Real.sqrt 2)) 
  (valid_start: is_valid_point start) :
  ∀ (p: Point), (is_valid_point p ∧ valid_movement start p) → p ≠ start :=
sorry

end NUMINAMATH_GPT_impossible_return_l866_86669


namespace NUMINAMATH_GPT_minimum_value_of_f_l866_86619

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y ≥ (5 / 2), f y = 1 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l866_86619


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l866_86693

noncomputable def p (x : ℝ) : Prop := |x - 3| < 1
noncomputable def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem p_sufficient_not_necessary_for_q (x : ℝ) :
  (p x → q x) ∧ (¬ (q x → p x)) := by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l866_86693


namespace NUMINAMATH_GPT_find_f_2011_l866_86698

noncomputable def f : ℝ → ℝ := sorry

axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom specific_interval (x : ℝ) (h2 : 2 < x) (h4 : x < 4) : f x = x + 3

theorem find_f_2011 : f 2011 = 6 :=
by {
  -- Leave this part to be filled with the actual proof,
  -- satisfying the initial conditions and concluding f(2011) = 6
  sorry
}

end NUMINAMATH_GPT_find_f_2011_l866_86698


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l866_86631

-- Define the objects and their relationships
noncomputable def α_parallel_β : Prop := sorry
noncomputable def a_parallel_α : Prop := sorry
noncomputable def b_perpendicular_β : Prop := sorry

-- Define the relationship we want to prove
noncomputable def a_perpendicular_b : Prop := sorry

-- The statement we want to prove
theorem relationship_between_a_and_b (h1 : α_parallel_β) (h2 : a_parallel_α) (h3 : b_perpendicular_β) : a_perpendicular_b :=
sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l866_86631


namespace NUMINAMATH_GPT_solution_set_for_log_inequality_l866_86643

noncomputable def log_base_0_1 (x: ℝ) : ℝ := Real.log x / Real.log 0.1

theorem solution_set_for_log_inequality :
  ∀ x : ℝ, (0 < x) → 
  log_base_0_1 (2^x - 1) < 0 ↔ x > 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_for_log_inequality_l866_86643


namespace NUMINAMATH_GPT_quarters_to_dimes_difference_l866_86667

variable (p : ℝ)

theorem quarters_to_dimes_difference :
  let Susan_quarters := 7 * p + 3
  let George_quarters := 2 * p + 9
  let difference_quarters := Susan_quarters - George_quarters
  let difference_dimes := 2.5 * difference_quarters
  difference_dimes = 12.5 * p - 15 :=
by
  let Susan_quarters := 7 * p + 3
  let George_quarters := 2 * p + 9
  let difference_quarters := Susan_quarters - George_quarters
  let difference_dimes := 2.5 * difference_quarters
  sorry

end NUMINAMATH_GPT_quarters_to_dimes_difference_l866_86667


namespace NUMINAMATH_GPT_bananas_in_each_box_l866_86685

theorem bananas_in_each_box 
    (bananas : ℕ) (boxes : ℕ) 
    (h_bananas : bananas = 40) 
    (h_boxes : boxes = 10) : 
    bananas / boxes = 4 := by
  sorry

end NUMINAMATH_GPT_bananas_in_each_box_l866_86685


namespace NUMINAMATH_GPT_simplify_expression_l866_86677

theorem simplify_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = -2 * Real.cos θ :=
sorry

end NUMINAMATH_GPT_simplify_expression_l866_86677


namespace NUMINAMATH_GPT_trains_crossing_time_l866_86672

-- Definitions based on given conditions
noncomputable def length_A : ℝ := 2500
noncomputable def time_A : ℝ := 50
noncomputable def length_B : ℝ := 3500
noncomputable def speed_factor : ℝ := 1.2

-- Speed computations
noncomputable def speed_A : ℝ := length_A / time_A
noncomputable def speed_B : ℝ := speed_A * speed_factor

-- Relative speed when moving in opposite directions
noncomputable def relative_speed : ℝ := speed_A + speed_B

-- Total distance covered when crossing each other
noncomputable def total_distance : ℝ := length_A + length_B

-- Time taken to cross each other
noncomputable def time_to_cross : ℝ := total_distance / relative_speed

-- Proof statement: Time taken is approximately 54.55 seconds
theorem trains_crossing_time :
  |time_to_cross - 54.55| < 0.01 := by
  sorry

end NUMINAMATH_GPT_trains_crossing_time_l866_86672


namespace NUMINAMATH_GPT_committee_membership_l866_86617

theorem committee_membership (n : ℕ) (h1 : 2 * n = 6) (h2 : (n - 1 : ℚ) / 5 = 0.4) : n = 3 := 
sorry

end NUMINAMATH_GPT_committee_membership_l866_86617


namespace NUMINAMATH_GPT_percentage_boys_from_school_A_is_20_l866_86687

-- Definitions and conditions based on the problem
def total_boys : ℕ := 200
def non_science_boys_from_A : ℕ := 28
def science_ratio : ℝ := 0.30
def non_science_ratio : ℝ := 1 - science_ratio

-- To prove: The percentage of the total boys that are from school A is 20%
theorem percentage_boys_from_school_A_is_20 :
  ∃ (x : ℝ), x = 20 ∧ 
  (non_science_ratio * (x / 100 * total_boys) = non_science_boys_from_A) := 
sorry

end NUMINAMATH_GPT_percentage_boys_from_school_A_is_20_l866_86687


namespace NUMINAMATH_GPT_prob_of_three_digit_divisible_by_3_l866_86696

/-- Define the exponents and the given condition --/
def a : ℕ := 5
def b : ℕ := 2
def c : ℕ := 3
def d : ℕ := 1

def condition : Prop := (2^a) * (3^b) * (5^c) * (7^d) = 252000

/-- The probability that a randomly chosen three-digit number formed by any 3 of a, b, c, d 
    is divisible by 3 and less than 250 is 1/4 --/
theorem prob_of_three_digit_divisible_by_3 :
  condition →
  ((sorry : ℝ) = 1/4) := sorry

end NUMINAMATH_GPT_prob_of_three_digit_divisible_by_3_l866_86696


namespace NUMINAMATH_GPT_jack_should_leave_300_in_till_l866_86699

-- Defining the amounts of each type of bill
def num_100_bills := 2
def num_50_bills := 1
def num_20_bills := 5
def num_10_bills := 3
def num_5_bills := 7
def num_1_bills := 27

-- The amount he needs to hand in
def amount_to_hand_in := 142

-- Calculating the total amount in notes
def total_in_notes := 
  (num_100_bills * 100) + 
  (num_50_bills * 50) + 
  (num_20_bills * 20) + 
  (num_10_bills * 10) + 
  (num_5_bills * 5) + 
  (num_1_bills * 1)

-- Calculating the amount to leave in the till
def amount_to_leave := total_in_notes - amount_to_hand_in

-- Proof statement
theorem jack_should_leave_300_in_till :
  amount_to_leave = 300 :=
by sorry

end NUMINAMATH_GPT_jack_should_leave_300_in_till_l866_86699


namespace NUMINAMATH_GPT_tabs_in_all_browsers_l866_86612

-- Definitions based on conditions
def windows_per_browser := 3
def tabs_per_window := 10
def number_of_browsers := 2

-- Total tabs calculation
def total_tabs := number_of_browsers * (windows_per_browser * tabs_per_window)

-- Proving the total number of tabs is 60
theorem tabs_in_all_browsers : total_tabs = 60 := by
  sorry

end NUMINAMATH_GPT_tabs_in_all_browsers_l866_86612


namespace NUMINAMATH_GPT_constant_S13_l866_86620

-- Defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Defining the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (List.range n).map a |> List.sum

-- Defining the given conditions as hypotheses
variable {a : ℕ → ℤ} {d : ℤ}
variable (h_arith : arithmetic_sequence a d)
variable (constant_sum : a 2 + a 4 + a 15 = k)

-- Goal to prove: S_13 is a constant
theorem constant_S13 (k : ℤ) :
  sum_first_n_terms a 13 = k :=
  sorry

end NUMINAMATH_GPT_constant_S13_l866_86620


namespace NUMINAMATH_GPT_perfect_cube_divisor_count_l866_86666

noncomputable def num_perfect_cube_divisors : Nat :=
  let a_choices := Nat.succ (38 / 3)
  let b_choices := Nat.succ (17 / 3)
  let c_choices := Nat.succ (7 / 3)
  let d_choices := Nat.succ (4 / 3)
  a_choices * b_choices * c_choices * d_choices

theorem perfect_cube_divisor_count :
  num_perfect_cube_divisors = 468 :=
by
  sorry

end NUMINAMATH_GPT_perfect_cube_divisor_count_l866_86666


namespace NUMINAMATH_GPT_correct_statement_B_l866_86627

-- Definitions as per the conditions
noncomputable def total_students : ℕ := 6700
noncomputable def selected_students : ℕ := 300

-- Definitions as per the question
def is_population (n : ℕ) : Prop := n = 6700
def is_sample (m n : ℕ) : Prop := m = 300 ∧ n = 6700
def is_individual (m n : ℕ) : Prop := m < n
def is_census (m n : ℕ) : Prop := m = n

-- The statement that needs to be proved
theorem correct_statement_B : 
  is_sample selected_students total_students :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_correct_statement_B_l866_86627


namespace NUMINAMATH_GPT_milk_for_6_cookies_l866_86647

/-- Given conditions for baking cookies -/
def quarts_to_pints : ℕ := 2 -- 2 pints in a quart
def milk_for_24_cookies : ℕ := 5 -- 5 quarts of milk for 24 cookies

/-- Theorem to prove the number of pints needed to bake 6 cookies -/
theorem milk_for_6_cookies : 
  (milk_for_24_cookies * quarts_to_pints * 6 / 24 : ℝ) = 2.5 := 
by 
  sorry -- Proof is omitted

end NUMINAMATH_GPT_milk_for_6_cookies_l866_86647


namespace NUMINAMATH_GPT_geom_seq_S6_l866_86670

theorem geom_seq_S6 :
  ∃ (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ),
  (q = 2) →
  (S 3 = 7) →
  (∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) →
  S 6 = 63 :=
sorry

end NUMINAMATH_GPT_geom_seq_S6_l866_86670


namespace NUMINAMATH_GPT_PQ_sum_l866_86650

theorem PQ_sum (P Q : ℕ) (h1 : 5 / 7 = P / 63) (h2 : 5 / 7 = 70 / Q) : P + Q = 143 :=
by
  sorry

end NUMINAMATH_GPT_PQ_sum_l866_86650


namespace NUMINAMATH_GPT_round_trip_ticket_percentage_l866_86684

theorem round_trip_ticket_percentage (P R : ℝ) 
  (h1 : 0.20 * P = 0.50 * R) : R = 0.40 * P :=
by
  sorry

end NUMINAMATH_GPT_round_trip_ticket_percentage_l866_86684


namespace NUMINAMATH_GPT_initial_percentage_is_30_l866_86694

def percentage_alcohol (P : ℝ) : Prop :=
  let initial_alcohol := (P / 100) * 50
  let mixed_solution_volume := 50 + 30
  let final_percentage_alcohol := 18.75
  let final_alcohol := (final_percentage_alcohol / 100) * mixed_solution_volume
  initial_alcohol = final_alcohol

theorem initial_percentage_is_30 :
  percentage_alcohol 30 :=
by
  unfold percentage_alcohol
  sorry

end NUMINAMATH_GPT_initial_percentage_is_30_l866_86694


namespace NUMINAMATH_GPT_correct_ordering_of_f_values_l866_86634

variable {f : ℝ → ℝ}

theorem correct_ordering_of_f_values
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hf_decreasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ > f x₂) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end NUMINAMATH_GPT_correct_ordering_of_f_values_l866_86634


namespace NUMINAMATH_GPT_dad_steps_l866_86635

theorem dad_steps (dad_steps_ratio: ℕ) (masha_steps_ratio: ℕ) (masha_steps: ℕ)
  (masha_and_yasha_steps: ℕ) (total_steps: ℕ)
  (h1: dad_steps_ratio * 3 = masha_steps_ratio * 5)
  (h2: masha_steps * 3 = masha_and_yasha_steps * 5)
  (h3: masha_and_yasha_steps = total_steps)
  (h4: total_steps = 400) :
  dad_steps_ratio * 30 = 90 :=
by
  sorry

end NUMINAMATH_GPT_dad_steps_l866_86635


namespace NUMINAMATH_GPT_consecutive_integer_sets_sum_27_l866_86682

theorem consecutive_integer_sets_sum_27 :
  ∃! s : Set (List ℕ), ∀ l ∈ s, 
  (∃ n a, n ≥ 3 ∧ l = List.range n ++ [a] ∧ (List.sum l) = 27)
:=
sorry

end NUMINAMATH_GPT_consecutive_integer_sets_sum_27_l866_86682


namespace NUMINAMATH_GPT_mark_remaining_money_l866_86625

theorem mark_remaining_money 
  (initial_money : ℕ) (num_books : ℕ) (cost_per_book : ℕ) (total_cost : ℕ) (remaining_money : ℕ) 
  (H1 : initial_money = 85)
  (H2 : num_books = 10)
  (H3 : cost_per_book = 5)
  (H4 : total_cost = num_books * cost_per_book)
  (H5 : remaining_money = initial_money - total_cost) : 
  remaining_money = 35 := 
by
  sorry

end NUMINAMATH_GPT_mark_remaining_money_l866_86625
