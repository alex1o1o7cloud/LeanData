import Mathlib

namespace minimum_value_condition_l1884_188495

def f (a x : ℝ) : ℝ := -x^3 + 0.5 * (a + 3) * x^2 - a * x - 1

theorem minimum_value_condition (a : ℝ) (h : a ≥ 3) : 
  (∃ x₀ : ℝ, f a x₀ < f a 1) ∨ (f a 1 > f a ((a/3))) := 
sorry

end minimum_value_condition_l1884_188495


namespace theater_ticket_sales_l1884_188455

-- Definitions of the given constants and initialization
def R : ℕ := 25

-- Conditions based on the problem statement
def condition_horror (H : ℕ) := H = 3 * R + 18
def condition_action (A : ℕ) := A = 2 * R
def condition_comedy (C H : ℕ) := 4 * H = 5 * C

-- Desired outcomes based on the solutions
def desired_horror := 93
def desired_action := 50
def desired_comedy := 74

theorem theater_ticket_sales
  (H A C : ℕ)
  (h1 : condition_horror H)
  (h2 : condition_action A)
  (h3 : condition_comedy C H)
  : H = desired_horror ∧ A = desired_action ∧ C = desired_comedy :=
by {
    sorry
}

end theater_ticket_sales_l1884_188455


namespace proof_problem_l1884_188460

noncomputable def a : ℝ := 2 - 0.5
noncomputable def b : ℝ := Real.log (Real.pi) / Real.log 3
noncomputable def c : ℝ := Real.log 2 / Real.log 4

theorem proof_problem : b > a ∧ a > c := 
by
sorry

end proof_problem_l1884_188460


namespace find_missing_fraction_l1884_188467

def f1 := 1/3
def f2 := 1/2
def f3 := 1/5
def f4 := 1/4
def f5 := -9/20
def f6 := -9/20
def total_sum := 45/100
def missing_fraction := 1/15

theorem find_missing_fraction : f1 + f2 + f3 + f4 + f5 + f6 + missing_fraction = total_sum :=
by
  sorry

end find_missing_fraction_l1884_188467


namespace total_texts_received_l1884_188450

open Nat 

-- Definition of conditions
def textsBeforeNoon : Nat := 21
def initialTextsAfterNoon : Nat := 2
def doublingTimeHours : Nat := 12

-- Definition to compute the total texts after noon recursively
def textsAfterNoon (n : Nat) : Nat :=
  if n = 0 then initialTextsAfterNoon
  else 2 * textsAfterNoon (n - 1)

-- Definition to sum the geometric series 
def sumGeometricSeries (a r n : Nat) : Nat :=
  if n = 0 then 0
  else a * (1 - r ^ n) / (1 - r)

-- Total text messages Debby received
def totalTextsReceived : Nat :=
  textsBeforeNoon + sumGeometricSeries initialTextsAfterNoon 2 doublingTimeHours

-- Proof statement
theorem total_texts_received: totalTextsReceived = 8211 := 
by 
  sorry

end total_texts_received_l1884_188450


namespace hyperbola_focal_length_l1884_188440

theorem hyperbola_focal_length :
  let a := 2
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  2 * c = 2 * Real.sqrt 7 := 
by
  sorry

end hyperbola_focal_length_l1884_188440


namespace systematic_sampling_number_l1884_188434

theorem systematic_sampling_number {n m s a b c d : ℕ} (h_n : n = 60) (h_m : m = 4) 
  (h_s : s = 3) (h_a : a = 33) (h_b : b = 48) 
  (h_gcd_1 : ∃ k, s + k * (n / m) = a) (h_gcd_2 : ∃ k, a + k * (n / m) = b) :
  ∃ k, s + k * (n / m) = d → d = 18 := by
  sorry

end systematic_sampling_number_l1884_188434


namespace logarithmic_product_l1884_188429

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem logarithmic_product (a b : ℝ) (h1 : a ≠ b) (h2 : f a = f b) : a * b = 1 := by
  sorry

end logarithmic_product_l1884_188429


namespace integer_solutions_l1884_188490

theorem integer_solutions :
  ∃ (a b c : ℤ), a + b + c = 24 ∧ a^2 + b^2 + c^2 = 210 ∧ a * b * c = 440 ∧
    (a = 5 ∧ b = 8 ∧ c = 11) ∨ (a = 5 ∧ b = 11 ∧ c = 8) ∨ 
    (a = 8 ∧ b = 5 ∧ c = 11) ∨ (a = 8 ∧ b = 11 ∧ c = 5) ∨
    (a = 11 ∧ b = 5 ∧ c = 8) ∨ (a = 11 ∧ b = 8 ∧ c = 5) :=
sorry

end integer_solutions_l1884_188490


namespace Euclid1976_PartA_Problem8_l1884_188459

theorem  Euclid1976_PartA_Problem8 (a b c m n : ℝ) 
  (h1 : Polynomial.eval a (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h4 : Polynomial.eval (-a) (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C n * Polynomial.X + Polynomial.C (-6)) = 0)
  (h5 : Polynomial.eval (-b) (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C n * Polynomial.X + Polynomial.C (-6)) = 0) :
  n = -1 :=
sorry

end Euclid1976_PartA_Problem8_l1884_188459


namespace average_is_correct_l1884_188468

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

def sum_of_numbers : ℕ := numbers.sum
def count_of_numbers : ℕ := numbers.length
def average_of_numbers : ℚ := sum_of_numbers / count_of_numbers

theorem average_is_correct : average_of_numbers = 1380 := 
by 
  -- Here, you would normally put the proof steps.
  sorry

end average_is_correct_l1884_188468


namespace base_angle_isosceles_triangle_l1884_188461

theorem base_angle_isosceles_triangle (α : ℝ) (hα : α = 108) (isosceles : ∀ (a b c : ℝ), a = b ∨ b = c ∨ c = a) : α = 108 →
  α + β + β = 180 → β = 36 :=
by
  sorry

end base_angle_isosceles_triangle_l1884_188461


namespace slower_train_crosses_faster_in_36_seconds_l1884_188402

-- Define the conditions of the problem
def speed_fast_train_kmph : ℚ := 110
def speed_slow_train_kmph : ℚ := 90
def length_fast_train_km : ℚ := 1.10
def length_slow_train_km : ℚ := 0.90

-- Convert speeds to m/s
def speed_fast_train_mps : ℚ := speed_fast_train_kmph * (1000 / 3600)
def speed_slow_train_mps : ℚ := speed_slow_train_kmph * (1000 / 3600)

-- Relative speed when moving in opposite directions
def relative_speed_mps : ℚ := speed_fast_train_mps + speed_slow_train_mps

-- Convert lengths to meters
def length_fast_train_m : ℚ := length_fast_train_km * 1000
def length_slow_train_m : ℚ := length_slow_train_km * 1000

-- Combined length of both trains in meters
def combined_length_m : ℚ := length_fast_train_m + length_slow_train_m

-- Time taken for the slower train to cross the faster train
def crossing_time : ℚ := combined_length_m / relative_speed_mps

theorem slower_train_crosses_faster_in_36_seconds :
  crossing_time = 36 := by
  sorry

end slower_train_crosses_faster_in_36_seconds_l1884_188402


namespace statement_A_statement_B_statement_C_statement_D_l1884_188423

-- Definitions based on the problem conditions
def curve (m : ℝ) (x y : ℝ) : Prop :=
  x^4 + y^4 + m * x^2 * y^2 = 1

def is_symmetric_about_origin (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y ↔ curve m (-x) (-y)

def enclosed_area_eq_pi (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y → (x^2 + y^2)^2 = 1

def does_not_intersect_y_eq_x (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y ∧ x = y → false

def no_common_points_with_region (m : ℝ) : Prop :=
  ∀ x y : ℝ, |x| + |y| < 1 → ¬ curve m x y

-- Statements to prove based on correct answers
theorem statement_A (m : ℝ) : is_symmetric_about_origin m :=
  sorry

theorem statement_B (m : ℝ) (h : m = 2) : enclosed_area_eq_pi m :=
  sorry

theorem statement_C (m : ℝ) (h : m = -2) : ¬ does_not_intersect_y_eq_x m :=
  sorry

theorem statement_D (m : ℝ) (h : m = -1) : no_common_points_with_region m :=
  sorry

end statement_A_statement_B_statement_C_statement_D_l1884_188423


namespace triangle_to_rectangle_ratio_l1884_188488

def triangle_perimeter := 60
def rectangle_perimeter := 60

def is_equilateral_triangle (side_length: ℝ) : Prop :=
  3 * side_length = triangle_perimeter

def is_valid_rectangle (length width: ℝ) : Prop :=
  2 * (length + width) = rectangle_perimeter ∧ length = 2 * width

theorem triangle_to_rectangle_ratio (s l w: ℝ) 
  (ht: is_equilateral_triangle s) 
  (hr: is_valid_rectangle l w) : 
  s / w = 2 := by
  sorry

end triangle_to_rectangle_ratio_l1884_188488


namespace student_can_escape_l1884_188437

open Real

/-- The student can escape the pool given the following conditions:
 1. R is the radius of the circular pool.
 2. The teacher runs 4 times faster than the student swims.
 3. The teacher's running speed is v_T.
 4. The student's swimming speed is v_S = v_T / 4.
 5. The student swims along a circular path of radius r, where
    (1 - π / 4) * R < r < R / 4 -/
theorem student_can_escape (R v_T v_S r : ℝ) (h1 : v_S = v_T / 4)
  (h2 : (1 - π / 4) * R < r) (h3 : r < R / 4) : 
  True :=
sorry

end student_can_escape_l1884_188437


namespace contrapositive_of_ab_eq_zero_l1884_188421

theorem contrapositive_of_ab_eq_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → ab ≠ 0 :=
by
  sorry

end contrapositive_of_ab_eq_zero_l1884_188421


namespace winner_exceeds_second_opponent_l1884_188493

theorem winner_exceeds_second_opponent
  (total_votes : ℕ)
  (votes_winner : ℕ)
  (votes_second : ℕ)
  (votes_third : ℕ)
  (votes_fourth : ℕ) 
  (h_votes_sum : total_votes = votes_winner + votes_second + votes_third + votes_fourth)
  (h_total_votes : total_votes = 963) 
  (h_winner_votes : votes_winner = 195) 
  (h_second_votes : votes_second = 142) 
  (h_third_votes : votes_third = 116) 
  (h_fourth_votes : votes_fourth = 90) :
  votes_winner - votes_second = 53 := by
  sorry

end winner_exceeds_second_opponent_l1884_188493


namespace common_difference_arithmetic_progression_l1884_188487

theorem common_difference_arithmetic_progression {n : ℕ} (x y : ℝ) (a : ℕ → ℝ) 
  (h : ∀ k : ℕ, k ≤ n → a (k+1) = a k + (y - x) / (n + 1)) 
  : (∃ d : ℝ, ∀ i : ℕ, i ≤ n + 1 → a (i+1) = x + i * d) ∧ d = (y - x) / (n + 1) := 
by
  sorry

end common_difference_arithmetic_progression_l1884_188487


namespace complex_square_eq_l1884_188438

theorem complex_square_eq (i : ℂ) (hi : i * i = -1) : (1 + i)^2 = 2 * i := 
by {
  -- marking the end of existing code for clarity
  sorry
}

end complex_square_eq_l1884_188438


namespace staircase_perimeter_l1884_188472

theorem staircase_perimeter (area : ℝ) (side_length : ℝ) (num_sides : ℕ) (right_angles : Prop) :
  area = 85 ∧ side_length = 1 ∧ num_sides = 10 ∧ right_angles → 
  ∃ perimeter : ℝ, perimeter = 30.5 :=
by
  intro h
  sorry

end staircase_perimeter_l1884_188472


namespace females_who_chose_malt_l1884_188448

-- Definitions
def total_cheerleaders : ℕ := 26
def total_males : ℕ := 10
def total_females : ℕ := 16
def males_who_chose_malt : ℕ := 6

-- Main statement
theorem females_who_chose_malt (C M F : ℕ) (hM : M = 2 * C) (h_total : C + M = total_cheerleaders) (h_males_malt : males_who_chose_malt = total_males) : F = 10 :=
sorry

end females_who_chose_malt_l1884_188448


namespace simplify_expression1_simplify_expression2_l1884_188449

-- Define variables as real numbers or appropriate domains
variables {a b x y: ℝ}

-- Problem 1
theorem simplify_expression1 : (2 * a - b) - (2 * b - 3 * a) - 2 * (a - 2 * b) = 3 * a + b :=
by sorry

-- Problem 2
theorem simplify_expression2 : (4 * x^2 - 5 * x * y) - (1 / 3 * y^2 + 2 * x^2) + 2 * (3 * x * y - 1 / 4 * y^2 - 1 / 12 * y^2) = 2 * x^2 + x * y - y^2 :=
by sorry

end simplify_expression1_simplify_expression2_l1884_188449


namespace rational_number_property_l1884_188401

theorem rational_number_property 
  (x : ℚ) (a : ℤ) (ha : 1 ≤ a) : 
  (x ^ (⌊x⌋)) = a / 2 → (∃ k : ℤ, x = k) ∨ x = 3 / 2 :=
by
  sorry

end rational_number_property_l1884_188401


namespace find_a_value_l1884_188499

theorem find_a_value :
  let center := (0.5, Real.sqrt 2)
  let line_dist (a : ℝ) := (abs (0.5 * a + Real.sqrt 2 - Real.sqrt 2)) / Real.sqrt (a^2 + 1)
  line_dist a = Real.sqrt 2 / 4 ↔ (a = 1 ∨ a = -1) :=
by
  sorry

end find_a_value_l1884_188499


namespace count_ways_with_3_in_M_count_ways_with_2_in_M_l1884_188439

structure ArrangementConfig where
  positions : Fin 9 → ℕ
  unique_positions : ∀ (i j : Fin 9) (hi hj : i ≠ j), positions i ≠ positions j
  no_adjacent_same : ∀ (i : Fin 8), positions i ≠ positions (i + 1)

def count_arrangements (fixed_value : ℕ) (fixed_position : Fin 9) : ℕ :=
  -- Implementation of counting the valid arrangements
  sorry

theorem count_ways_with_3_in_M : count_arrangements 3 0 = 6 := sorry

theorem count_ways_with_2_in_M : count_arrangements 2 0 = 12 := sorry

end count_ways_with_3_in_M_count_ways_with_2_in_M_l1884_188439


namespace initial_sheep_count_l1884_188408

theorem initial_sheep_count (S : ℕ) :
  let S1 := S - (S / 3 + 1 / 3)
  let S2 := S1 - (S1 / 4 + 1 / 4)
  let S3 := S2 - (S2 / 5 + 3 / 5)
  S3 = 409
  → S = 1025 := 
by 
  sorry

end initial_sheep_count_l1884_188408


namespace triangle_area_l1884_188444

/-- Given a triangle ABC with BC = 12 cm and AD perpendicular to BC with AD = 15 cm,
    prove that the area of triangle ABC is 90 square centimeters. -/
theorem triangle_area {BC AD : ℝ} (hBC : BC = 12) (hAD : AD = 15) :
  (1 / 2) * BC * AD = 90 := by
  sorry

end triangle_area_l1884_188444


namespace random_events_count_is_five_l1884_188412

-- Definitions of the events in the conditions
def event1 := "Classmate A successfully runs for class president"
def event2 := "Stronger team wins in a game between two teams"
def event3 := "A school has a total of 998 students, and at least three students share the same birthday"
def event4 := "If sets A, B, and C satisfy A ⊆ B and B ⊆ C, then A ⊆ C"
def event5 := "In ancient times, a king wanted to execute a painter. Secretly, he wrote 'death' on both slips of paper, then let the painter draw a 'life or death' slip. The painter drew a death slip"
def event6 := "It snows in July"
def event7 := "Choosing any two numbers from 1, 3, 9, and adding them together results in an even number"
def event8 := "Riding through 10 intersections, all lights encountered are red"

-- Tally up the number of random events
def is_random_event (event : String) : Bool :=
  event = event1 ∨
  event = event2 ∨
  event = event3 ∨
  event = event6 ∨
  event = event8

def count_random_events (events : List String) : Nat :=
  (events.map (λ event => if is_random_event event then 1 else 0)).sum

-- List of events
def events := [event1, event2, event3, event4, event5, event6, event7, event8]

-- Theorem statement
theorem random_events_count_is_five : count_random_events events = 5 :=
  by
    sorry

end random_events_count_is_five_l1884_188412


namespace average_marks_math_chem_l1884_188442

variables (M P C : ℕ)

theorem average_marks_math_chem :
  (M + P = 20) → (C = P + 20) → (M + C) / 2 = 20 := 
by
  sorry

end average_marks_math_chem_l1884_188442


namespace coffee_mug_cost_l1884_188475

theorem coffee_mug_cost (bracelet_cost gold_heart_necklace_cost total_change total_money_spent : ℤ)
    (bracelets_count gold_heart_necklace_count mugs_count : ℤ)
    (h_bracelet_cost : bracelet_cost = 15)
    (h_gold_heart_necklace_cost : gold_heart_necklace_cost = 10)
    (h_total_change : total_change = 15)
    (h_total_money_spent : total_money_spent = 100)
    (h_bracelets_count : bracelets_count = 3)
    (h_gold_heart_necklace_count : gold_heart_necklace_count = 2)
    (h_mugs_count : mugs_count = 1) :
    mugs_count * ((total_money_spent - total_change) - (bracelets_count * bracelet_cost + gold_heart_necklace_count * gold_heart_necklace_cost)) = 20 :=
by
  sorry

end coffee_mug_cost_l1884_188475


namespace slowest_bailing_rate_proof_l1884_188418

def distance : ℝ := 1.5 -- in miles
def rowing_speed : ℝ := 3 -- in miles per hour
def water_intake_rate : ℝ := 8 -- in gallons per minute
def sink_threshold : ℝ := 50 -- in gallons

noncomputable def solve_bailing_rate_proof : ℝ :=
  let time_to_shore_hours : ℝ := distance / rowing_speed
  let time_to_shore_minutes : ℝ := time_to_shore_hours * 60
  let total_water_intake : ℝ := water_intake_rate * time_to_shore_minutes
  let excess_water : ℝ := total_water_intake - sink_threshold
  let bailing_rate_needed : ℝ := excess_water / time_to_shore_minutes
  bailing_rate_needed

theorem slowest_bailing_rate_proof : solve_bailing_rate_proof ≤ 7 :=
  by
    sorry

end slowest_bailing_rate_proof_l1884_188418


namespace minimum_value_l1884_188410

theorem minimum_value (x y : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : x + y = 1) : 
  ∃ z, z = 9 ∧ (forall x y, x > 0 ∧ y > 0 ∧ x + y = 1 → (1/x + 4/y) ≥ z) := 
sorry

end minimum_value_l1884_188410


namespace average_of_remaining_two_nums_l1884_188483

theorem average_of_remaining_two_nums (S S4 : ℕ) (h1 : S / 6 = 8) (h2 : S4 / 4 = 5) :
  ((S - S4) / 2 = 14) :=
by 
  sorry

end average_of_remaining_two_nums_l1884_188483


namespace arithmetic_sequence_10th_term_l1884_188496

theorem arithmetic_sequence_10th_term (a d : ℤ) :
    (a + 4 * d = 26) →
    (a + 7 * d = 50) →
    (a + 9 * d = 66) := by
  intros h1 h2
  sorry

end arithmetic_sequence_10th_term_l1884_188496


namespace negation_of_forall_ge_zero_l1884_188419

theorem negation_of_forall_ge_zero :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 := by
  sorry

end negation_of_forall_ge_zero_l1884_188419


namespace dogs_not_eating_any_foods_l1884_188430

theorem dogs_not_eating_any_foods :
  let total_dogs := 80
  let dogs_like_watermelon := 18
  let dogs_like_salmon := 58
  let dogs_like_both_salmon_watermelon := 7
  let dogs_like_chicken := 16
  let dogs_like_both_chicken_salmon := 6
  let dogs_like_both_chicken_watermelon := 4
  let dogs_like_all_three := 3
  let dogs_like_any_food := dogs_like_watermelon + dogs_like_salmon + dogs_like_chicken - 
                            dogs_like_both_salmon_watermelon - dogs_like_both_chicken_salmon - 
                            dogs_like_both_chicken_watermelon + dogs_like_all_three
  total_dogs - dogs_like_any_food = 2 := by
  sorry

end dogs_not_eating_any_foods_l1884_188430


namespace students_not_next_each_other_l1884_188409

open Nat

theorem students_not_next_each_other (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 5) (h2 : k = 2) (h3 : m = 3)
  (h4 : ∀ (A B : ℕ), A ≠ B) : 
  ∃ (total : ℕ), total = 3! * (choose (5-3+1) 2) := 
by
  sorry

end students_not_next_each_other_l1884_188409


namespace relationship_among_three_numbers_l1884_188471

noncomputable def M (a b : ℝ) : ℝ := a^b
noncomputable def N (a b : ℝ) : ℝ := Real.log a / Real.log b
noncomputable def P (a b : ℝ) : ℝ := b^a

theorem relationship_among_three_numbers (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : N a b < M a b ∧ M a b < P a b := 
by
  sorry

end relationship_among_three_numbers_l1884_188471


namespace chalk_pieces_l1884_188457

theorem chalk_pieces (boxes: ℕ) (pieces_per_box: ℕ) (total_chalk: ℕ) 
  (hb: boxes = 194) (hp: pieces_per_box = 18) : 
  total_chalk = 194 * 18 :=
by 
  sorry

end chalk_pieces_l1884_188457


namespace strawberry_growth_rate_l1884_188476

theorem strawberry_growth_rate
  (initial_plants : ℕ)
  (months : ℕ)
  (plants_given_away : ℕ)
  (total_plants_after : ℕ)
  (growth_rate : ℕ)
  (h_initial : initial_plants = 3)
  (h_months : months = 3)
  (h_given_away : plants_given_away = 4)
  (h_total_after : total_plants_after = 20)
  (h_equation : initial_plants + growth_rate * months - plants_given_away = total_plants_after) :
  growth_rate = 7 :=
sorry

end strawberry_growth_rate_l1884_188476


namespace total_units_per_day_all_work_together_l1884_188445

-- Conditions
def men := 250
def women := 150
def units_per_day_by_men := 15
def units_per_day_by_women := 3

-- Problem statement and proof
theorem total_units_per_day_all_work_together :
  units_per_day_by_men + units_per_day_by_women = 18 :=
sorry

end total_units_per_day_all_work_together_l1884_188445


namespace ada_original_seat_l1884_188484

-- Define the problem conditions
def initial_seats : List ℕ := [1, 2, 3, 4, 5]  -- seat numbers

def bea_move (seat : ℕ) : ℕ := seat + 2  -- Bea moves 2 seats to the right
def ceci_move (seat : ℕ) : ℕ := seat - 1  -- Ceci moves 1 seat to the left
def switch (seats : (ℕ × ℕ)) : (ℕ × ℕ) := (seats.2, seats.1)  -- Dee and Edie switch seats

-- The final seating positions (end seats are 1 or 5 for Ada)
axiom ada_end_seat : ∃ final_seat : ℕ, final_seat ∈ [1, 5]  -- Ada returns to an end seat

-- Prove Ada was originally sitting in seat 2
theorem ada_original_seat (final_seat : ℕ) (h₁ : ∃ (s₁ s₂ : ℕ), s₁ ≠ s₂ ∧ bea_move s₁ ≠ final_seat ∧ ceci_move s₂ ≠ final_seat ∧ switch (s₁, s₂).2 ≠ final_seat) : 2 ∈ initial_seats :=
by
  sorry

end ada_original_seat_l1884_188484


namespace sum_quotient_dividend_divisor_l1884_188413

theorem sum_quotient_dividend_divisor (D : ℕ) (d : ℕ) (Q : ℕ) 
  (h1 : D = 54) (h2 : d = 9) (h3 : D = Q * d) : 
  (Q + D + d) = 69 :=
by
  sorry

end sum_quotient_dividend_divisor_l1884_188413


namespace cos_45_degree_l1884_188431

theorem cos_45_degree : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_45_degree_l1884_188431


namespace filling_tank_ratio_l1884_188474

theorem filling_tank_ratio :
  ∀ (t : ℝ),
    (1 / 40) * t + (1 / 24) * (29.999999999999993 - t) = 1 →
    t / 29.999999999999993 = 1 / 2 :=
by
  intro t
  intro H
  sorry

end filling_tank_ratio_l1884_188474


namespace jenny_ate_65_chocolates_l1884_188494

noncomputable def chocolates_eaten_by_Jenny : ℕ :=
  let chocolates_mike := 20
  let chocolates_john := chocolates_mike / 2
  let combined_chocolates := chocolates_mike + chocolates_john
  let twice_combined_chocolates := 2 * combined_chocolates
  5 + twice_combined_chocolates

theorem jenny_ate_65_chocolates :
  chocolates_eaten_by_Jenny = 65 :=
by
  -- Skipping the proof details
  sorry

end jenny_ate_65_chocolates_l1884_188494


namespace quadratic_root_range_l1884_188414

/-- 
  Define the quadratic function y = ax^2 + bx + c for given values.
  Show that there exists x_1 in the interval (-1, 0) such that y = 0.
-/
theorem quadratic_root_range {a b c : ℝ} (h : a ≠ 0) 
  (h_minus3 : a * (-3)^2 + b * (-3) + c = -11)
  (h_minus2 : a * (-2)^2 + b * (-2) + c = -5)
  (h_minus1 : a * (-1)^2 + b * (-1) + c = -1)
  (h_0 : a * 0^2 + b * 0 + c = 1)
  (h_1 : a * 1^2 + b * 1 + c = 1) : 
  ∃ x1 : ℝ, -1 < x1 ∧ x1 < 0 ∧ a * x1^2 + b * x1 + c = 0 :=
sorry

end quadratic_root_range_l1884_188414


namespace evaluate_expression_l1884_188452

theorem evaluate_expression (a b c : ℕ) (h1 : a = 12) (h2 : b = 8) (h3 : c = 3) :
  (a - b + c - (a - (b + c)) = 6) := by
  sorry

end evaluate_expression_l1884_188452


namespace range_of_x_l1884_188463

theorem range_of_x (x : ℝ) : (6 - 2 * x) ≠ 0 ↔ x ≠ 3 := 
by {
  sorry
}

end range_of_x_l1884_188463


namespace math_problem_l1884_188441

theorem math_problem (a b : ℕ) (h₁ : a = 6) (h₂ : b = 6) : 
  (a^3 + b^3) / (a^2 - a * b + b^2) = 12 :=
by
  sorry

end math_problem_l1884_188441


namespace inequality_proof_l1884_188433

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) : (1 / x) < (1 / y) :=
by
  sorry

end inequality_proof_l1884_188433


namespace sum_of_reciprocals_l1884_188420

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 :=
by
  sorry

end sum_of_reciprocals_l1884_188420


namespace curve_crossing_l1884_188453

structure Point where
  x : ℝ
  y : ℝ

def curve (t : ℝ) : Point :=
  { x := 2 * t^2 - 3, y := 2 * t^4 - 9 * t^2 + 6 }

theorem curve_crossing : ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve 1 = { x := -1, y := -1 } := by
  sorry

end curve_crossing_l1884_188453


namespace press_t_denomination_l1884_188473

def press_f_rate_per_minute := 1000
def press_t_rate_per_minute := 200
def time_in_seconds := 3
def f_denomination := 5
def additional_amount := 50

theorem press_t_denomination : 
  ∃ (x : ℝ), 
  (3 * (5 * (1000 / 60))) = (3 * (x * (200 / 60)) + 50) → 
  x = 20 := 
by 
  -- Proof logic here
  sorry

end press_t_denomination_l1884_188473


namespace solve_fractional_equation_l1884_188482

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 4) : 
  (3 - x) / (x - 4) + 1 / (4 - x) = 1 → x = 3 :=
by {
  sorry
}

end solve_fractional_equation_l1884_188482


namespace trucks_after_redistribution_l1884_188485

/-- Problem Statement:
   Prove that the total number of trucks after redistribution is 10.
-/

theorem trucks_after_redistribution
    (num_trucks1 : ℕ)
    (boxes_per_truck1 : ℕ)
    (num_trucks2 : ℕ)
    (boxes_per_truck2 : ℕ)
    (containers_per_box : ℕ)
    (containers_per_truck_after : ℕ)
    (h1 : num_trucks1 = 7)
    (h2 : boxes_per_truck1 = 20)
    (h3 : num_trucks2 = 5)
    (h4 : boxes_per_truck2 = 12)
    (h5 : containers_per_box = 8)
    (h6 : containers_per_truck_after = 160) :
  (num_trucks1 * boxes_per_truck1 + num_trucks2 * boxes_per_truck2) * containers_per_box / containers_per_truck_after = 10 := by
  sorry

end trucks_after_redistribution_l1884_188485


namespace abs_inequality_solution_l1884_188425

theorem abs_inequality_solution (x : ℝ) : (|x + 3| > x + 3) ↔ (x < -3) :=
by
  sorry

end abs_inequality_solution_l1884_188425


namespace tangential_tetrahedron_triangle_impossibility_l1884_188403

theorem tangential_tetrahedron_triangle_impossibility (a b c d : ℝ) 
  (h : ∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → x > 0) :
  ¬ (∀ (x y z : ℝ) , (x = a ∨ x = b ∨ x = c ∨ x = d) → 
    (y = a ∨ y = b ∨ y = c ∨ y = d) →
    (z = a ∨ z = b ∨ z = c ∨ z = d) → 
    x ≠ y → y ≠ z → z ≠ x → x + y > z ∧ x + z > y ∧ y + z > x) :=
sorry

end tangential_tetrahedron_triangle_impossibility_l1884_188403


namespace price_of_36kgs_l1884_188492

namespace Apples

-- Define the parameters l and q
variables (l q : ℕ)

-- Define the conditions
def cost_first_30kgs (l : ℕ) : ℕ := 30 * l
def cost_first_15kgs : ℕ := 150
def cost_33kgs (l q : ℕ) : ℕ := (30 * l) + (3 * q)
def cost_36kgs (l q : ℕ) : ℕ := (30 * l) + (6 * q)

-- Define the hypothesis for l and q based on given conditions
axiom l_value (h1 : cost_first_15kgs = 150) : l = 10
axiom q_value (h2 : cost_33kgs l q = 333) : q = 11

-- Prove the price of 36 kilograms of apples
theorem price_of_36kgs (h1 : cost_first_15kgs = 150) (h2 : cost_33kgs l q = 333) : cost_36kgs l q = 366 :=
sorry

end Apples

end price_of_36kgs_l1884_188492


namespace constant_sequence_is_AP_and_GP_l1884_188405

theorem constant_sequence_is_AP_and_GP (seq : ℕ → ℕ) (h : ∀ n, seq n = 7) :
  (∃ d, ∀ n, seq n = seq (n + 1) + d) ∧ (∃ r, ∀ n, seq (n + 1) = seq n * r) :=
by
  sorry

end constant_sequence_is_AP_and_GP_l1884_188405


namespace select_2n_comparable_rectangles_l1884_188466

def comparable (A B : Rectangle) : Prop :=
  -- A can be placed into B by translation and rotation
  exists f : Rectangle → Rectangle, f A = B

theorem select_2n_comparable_rectangles (n : ℕ) (h : n > 1) :
  ∃ (rectangles : List Rectangle), rectangles.length = 2 * n ∧
  ∀ (a b : Rectangle), a ∈ rectangles → b ∈ rectangles → comparable a b :=
sorry

end select_2n_comparable_rectangles_l1884_188466


namespace find_second_divisor_l1884_188428

theorem find_second_divisor :
  ∃ x : ℕ, 377 / 13 / x * (1/4 : ℚ) / 2 = 0.125 ∧ x = 29 :=
by
  use 29
  -- Proof steps would go here
  sorry

end find_second_divisor_l1884_188428


namespace sum_of_fourth_powers_eq_square_of_sum_of_squares_l1884_188489

theorem sum_of_fourth_powers_eq_square_of_sum_of_squares 
  (x1 x2 x3 : ℝ) (p q n : ℝ)
  (h1 : x1^3 + p*x1^2 + q*x1 + n = 0)
  (h2 : x2^3 + p*x2^2 + q*x2 + n = 0)
  (h3 : x3^3 + p*x3^2 + q*x3 + n = 0)
  (h_rel : q^2 = 2 * n * p) :
  x1^4 + x2^4 + x3^4 = (x1^2 + x2^2 + x3^2)^2 := 
sorry

end sum_of_fourth_powers_eq_square_of_sum_of_squares_l1884_188489


namespace problem1_problem2_l1884_188416

-- Problem 1
theorem problem1 (α : ℝ) (h : (Real.tan α) / (Real.tan α - 1) = -1) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -5 / 3 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h : (Real.tan α) / (Real.tan α - 1) = -1) (h_quad : π < α ∧ α < 3 * π / 2) :
  Real.cos (-π + α) + Real.cos (π / 2 + α) = 3 * Real.sqrt 5 / 5 :=
by sorry

end problem1_problem2_l1884_188416


namespace James_watch_time_l1884_188497

def Jeopardy_length : ℕ := 20
def Wheel_of_Fortune_length : ℕ := Jeopardy_length * 2
def Jeopardy_episodes : ℕ := 2
def Wheel_of_Fortune_episodes : ℕ := 2

theorem James_watch_time :
  (Jeopardy_episodes * Jeopardy_length + Wheel_of_Fortune_episodes * Wheel_of_Fortune_length) / 60 = 2 :=
by
  sorry

end James_watch_time_l1884_188497


namespace units_digit_G_100_l1884_188427

def G (n : ℕ) : ℕ := 3 ^ (2 ^ n) + 1

theorem units_digit_G_100 : (G 100) % 10 = 2 := 
by
  sorry

end units_digit_G_100_l1884_188427


namespace number_of_ways_to_arrange_BANANA_l1884_188422

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l1884_188422


namespace meaningful_expression_iff_l1884_188478

theorem meaningful_expression_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 3))) ↔ x > 3 := by
  sorry

end meaningful_expression_iff_l1884_188478


namespace age_problem_l1884_188470

-- Define the conditions
variables (a b c : ℕ)

-- Assumptions based on conditions
theorem age_problem (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 37) : b = 14 :=
by {
  sorry   -- Placeholder for the detailed proof
}

end age_problem_l1884_188470


namespace product_of_complex_conjugates_l1884_188443

theorem product_of_complex_conjugates (i : ℂ) (h : i^2 = -1) : (1 + i) * (1 - i) = 2 :=
by
  sorry

end product_of_complex_conjugates_l1884_188443


namespace relationship_between_a_and_b_l1884_188411

-- Definitions for the conditions
variables {a b : ℝ}

-- Main theorem statement
theorem relationship_between_a_and_b (h1 : |Real.log (1 / 4) / Real.log a| = Real.log (1 / 4) / Real.log a)
  (h2 : |Real.log a / Real.log b| = -Real.log a / Real.log b) :
  0 < a ∧ a < 1 ∧ 1 < b :=
by
  sorry

end relationship_between_a_and_b_l1884_188411


namespace production_problem_l1884_188426

theorem production_problem (x y : ℝ) (h₁ : x > 0) (h₂ : ∀ k : ℝ, x * x * x * k = x) : (x * x * y * (1 / (x^2)) = y) :=
by {
  sorry
}

end production_problem_l1884_188426


namespace gcd_36745_59858_l1884_188458

theorem gcd_36745_59858 : Nat.gcd 36745 59858 = 7 :=
sorry

end gcd_36745_59858_l1884_188458


namespace dmitriev_is_older_l1884_188435

variables (Alekseev Borisov Vasilyev Grigoryev Dima Dmitriev : ℤ)

def Lesha := Alekseev + 1
def Borya := Borisov + 2
def Vasya := Vasilyev + 3
def Grisha := Grigoryev + 4

theorem dmitriev_is_older :
  Dima + 10 = Dmitriev :=
sorry

end dmitriev_is_older_l1884_188435


namespace geometric_sequence_first_term_l1884_188481

theorem geometric_sequence_first_term (S_3 S_6 : ℝ) (a_1 q : ℝ)
  (hS3 : S_3 = 6) (hS6 : S_6 = 54)
  (hS3_def : S_3 = a_1 * (1 - q^3) / (1 - q))
  (hS6_def : S_6 = a_1 * (1 - q^6) / (1 - q)) :
  a_1 = 6 / 7 := 
by
  sorry

end geometric_sequence_first_term_l1884_188481


namespace muffin_to_banana_ratio_l1884_188404

-- Definitions of costs
def elaine_cost (m b : ℝ) : ℝ := 5 * m + 4 * b
def derek_cost (m b : ℝ) : ℝ := 3 * m + 18 * b

-- The problem statement
theorem muffin_to_banana_ratio (m b : ℝ) (h : derek_cost m b = 3 * elaine_cost m b) : m / b = 2 :=
by
  sorry

end muffin_to_banana_ratio_l1884_188404


namespace ad_value_l1884_188407

variable (a b c d : ℝ)

-- Conditions
def geom_seq := b^2 = a * c ∧ c^2 = b * d
def vertex_of_parabola := (b = 1 ∧ c = 2)

-- Question
theorem ad_value (h_geom : geom_seq a b c d) (h_vertex : vertex_of_parabola b c) : a * d = 2 := by
  sorry

end ad_value_l1884_188407


namespace division_identity_l1884_188447

theorem division_identity : 45 / 0.05 = 900 :=
by
  sorry

end division_identity_l1884_188447


namespace quadratic_trinomial_form_l1884_188432

noncomputable def quadratic_form (a b c : ℝ) (h : a ≠ 0) : Prop :=
  ∀ x : ℝ, 
    (a * (3.8 * x - 1)^2 + b * (3.8 * x - 1) + c) = (a * (-3.8 * x)^2 + b * (-3.8 * x) + c)

theorem quadratic_trinomial_form (a b c : ℝ) (h : a ≠ 0) : b = a → quadratic_form a b c h :=
by
  intro hba
  unfold quadratic_form
  intro x
  rw [hba]
  sorry

end quadratic_trinomial_form_l1884_188432


namespace yard_length_l1884_188454

theorem yard_length (trees : ℕ) (distance_per_gap : ℕ) (gaps : ℕ) :
  trees = 26 → distance_per_gap = 16 → gaps = trees - 1 → length_of_yard = gaps * distance_per_gap → length_of_yard = 400 :=
by 
  intros h_trees h_distance_per_gap h_gaps h_length_of_yard
  sorry

end yard_length_l1884_188454


namespace dad_strawberries_weight_l1884_188477

-- Definitions for the problem
def weight_marco := 15
def total_weight := 37

-- Theorem statement
theorem dad_strawberries_weight :
  (total_weight - weight_marco = 22) :=
by
  sorry

end dad_strawberries_weight_l1884_188477


namespace symmetric_about_origin_l1884_188462

theorem symmetric_about_origin (x y : ℝ) :
  (∀ (x y : ℝ), (x*y - x^2 = 1) → ((-x)*(-y) - (-x)^2 = 1)) :=
by
  intros x y h
  sorry

end symmetric_about_origin_l1884_188462


namespace max_value_y_l1884_188464

noncomputable def y (x : ℝ) : ℝ := 3 - 3*x - 1/x

theorem max_value_y : (∃ x > 0, ∀ x' > 0, y x' ≤ y x) ∧ (y (1 / Real.sqrt 3) = 3 - 2 * Real.sqrt 3) :=
by
  sorry

end max_value_y_l1884_188464


namespace total_pizza_slices_correct_l1884_188417

-- Define the conditions
def num_pizzas : Nat := 3
def slices_per_first_two_pizzas : Nat := 8
def num_first_two_pizzas : Nat := 2
def slices_third_pizza : Nat := 12

-- Define the total slices based on conditions
def total_slices : Nat := slices_per_first_two_pizzas * num_first_two_pizzas + slices_third_pizza

-- The theorem to be proven
theorem total_pizza_slices_correct : total_slices = 28 := by
  sorry

end total_pizza_slices_correct_l1884_188417


namespace probability_two_doors_open_l1884_188451

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_doors_open :
  let total_doors := 5
  let total_combinations := 2 ^ total_doors
  let favorable_combinations := binomial total_doors 2
  let probability := favorable_combinations / total_combinations
  probability = 5 / 16 :=
by
  sorry

end probability_two_doors_open_l1884_188451


namespace altitude_product_difference_eq_zero_l1884_188480

variables (A B C P Q H : Type*) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited H]
variable {HP HQ BP PC AQ QC AH BH : ℝ}

-- Given conditions
axiom altitude_intersects_at_H : true
axiom HP_val : HP = 3
axiom HQ_val : HQ = 7

-- Statement to prove
theorem altitude_product_difference_eq_zero (h_BP_PC : BP * PC = 3 / (AH + 3))
                                           (h_AQ_QC : AQ * QC = 7 / (BH + 7))
                                           (h_AH_BQ_ratio : AH / BH = 3 / 7) :
  (BP * PC) - (AQ * QC) = 0 :=
by sorry

end altitude_product_difference_eq_zero_l1884_188480


namespace fruits_turned_yellow_on_friday_l1884_188436

theorem fruits_turned_yellow_on_friday :
  ∃ (F : ℕ), F + 2*F = 6 ∧ 14 - F - 2*F = 8 :=
by
  existsi 2
  sorry

end fruits_turned_yellow_on_friday_l1884_188436


namespace area_of_quadrilateral_l1884_188456

theorem area_of_quadrilateral (A B C D H : Type) (AB BC : Real)
    (angle_ABC angle_ADC : Real) (BH h : Real)
    (H1 : AB = BC) (H2 : angle_ABC = 90 ∧ angle_ADC = 90)
    (H3 : BH = h) :
    (∃ area : Real, area = h^2) :=
by
  sorry

end area_of_quadrilateral_l1884_188456


namespace investment_B_l1884_188469

theorem investment_B {x : ℝ} :
  let a_investment := 6300
  let c_investment := 10500
  let total_profit := 12100
  let a_share_profit := 3630
  (6300 / (6300 + x + 10500) = 3630 / 12100) →
  x = 13650 :=
by { sorry }

end investment_B_l1884_188469


namespace fourth_quadrant_negative_half_x_axis_upper_half_plane_l1884_188491

theorem fourth_quadrant (m : ℝ) : ((-7 < m ∧ m < 3) ↔ ((m^2 - 8 * m + 15 > 0) ∧ (m^2 + 3 * m - 28 < 0))) :=
sorry

theorem negative_half_x_axis (m : ℝ) : (m = 4 ↔ ((m^2 - 8 * m + 15 < 0) ∧ (m^2 + 3 * m - 28 = 0))) :=
sorry

theorem upper_half_plane (m : ℝ) : ((m ≥ 4 ∨ m ≤ -7) ↔ (m^2 + 3 * m - 28 ≥ 0)) :=
sorry

end fourth_quadrant_negative_half_x_axis_upper_half_plane_l1884_188491


namespace max_right_angle_triangles_l1884_188406

open Real

theorem max_right_angle_triangles (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x y : ℝ, x^2 + a^2 * y^2 = a^2) :
  ∃n : ℕ, n = 3 := 
by
  sorry

end max_right_angle_triangles_l1884_188406


namespace total_green_ducks_percentage_l1884_188479

def ducks_in_park_A : ℕ := 200
def green_percentage_A : ℕ := 25

def ducks_in_park_B : ℕ := 350
def green_percentage_B : ℕ := 20

def ducks_in_park_C : ℕ := 120
def green_percentage_C : ℕ := 50

def ducks_in_park_D : ℕ := 60
def green_percentage_D : ℕ := 25

def ducks_in_park_E : ℕ := 500
def green_percentage_E : ℕ := 30

theorem total_green_ducks_percentage (green_ducks_A green_ducks_B green_ducks_C green_ducks_D green_ducks_E total_ducks : ℕ)
  (h_A : green_ducks_A = ducks_in_park_A * green_percentage_A / 100)
  (h_B : green_ducks_B = ducks_in_park_B * green_percentage_B / 100)
  (h_C : green_ducks_C = ducks_in_park_C * green_percentage_C / 100)
  (h_D : green_ducks_D = ducks_in_park_D * green_percentage_D / 100)
  (h_E : green_ducks_E = ducks_in_park_E * green_percentage_E / 100)
  (h_total_ducks : total_ducks = ducks_in_park_A + ducks_in_park_B + ducks_in_park_C + ducks_in_park_D + ducks_in_park_E) :
  (green_ducks_A + green_ducks_B + green_ducks_C + green_ducks_D + green_ducks_E) * 100 / total_ducks = 2805 / 100 :=
by sorry

end total_green_ducks_percentage_l1884_188479


namespace days_playing_video_games_l1884_188446

-- Define the conditions
def watchesTVDailyHours : ℕ := 4
def videoGameHoursPerPlay : ℕ := 2
def totalWeeklyHours : ℕ := 34
def weeklyTVDailyHours : ℕ := 7 * watchesTVDailyHours

-- Define the number of days playing video games
def playsVideoGamesDays (d : ℕ) : ℕ := d * videoGameHoursPerPlay

-- Define the number of days Mike plays video games
theorem days_playing_video_games (d : ℕ) :
  weeklyTVDailyHours + playsVideoGamesDays d = totalWeeklyHours → d = 3 :=
by
  -- The proof is omitted
  sorry

end days_playing_video_games_l1884_188446


namespace cube_volume_edge_length_range_l1884_188486

theorem cube_volume_edge_length_range (a : ℝ) (h : a^3 = 9) : 2 < a ∧ a < 2.5 :=
by {
    -- proof will go here
    sorry
}

end cube_volume_edge_length_range_l1884_188486


namespace find_value_of_k_l1884_188400

noncomputable def value_of_k (m n : ℝ) : ℝ :=
  let p := 0.4
  let point1 := (m, n)
  let point2 := (m + 2, n + p)
  let k := 5
  k

theorem find_value_of_k (m n : ℝ) : value_of_k m n = 5 :=
sorry

end find_value_of_k_l1884_188400


namespace inequality_problem_l1884_188498

theorem inequality_problem (a b : ℝ) (h₁ : 1/a < 1/b) (h₂ : 1/b < 0) :
  (∃ (p q : Prop), 
    (p ∧ q) ∧ 
    ((p ↔ (a + b < a * b)) ∧ 
    (¬q ↔ |a| ≤ |b|) ∧ 
    (¬q ↔ a > b) ∧ 
    (q ↔ (b / a + a / b > 2)))) :=
sorry

end inequality_problem_l1884_188498


namespace find_width_of_bobs_tv_l1884_188424

def area (w h : ℕ) : ℕ := w * h

def weight_in_oz (area : ℕ) : ℕ := area * 4

def weight_in_lb (weight_in_oz : ℕ) : ℕ := weight_in_oz / 16

def width_of_bobs_tv (x : ℕ) : Prop :=
  area 48 100 = 4800 ∧
  weight_in_lb (weight_in_oz (area 48 100)) = 1200 ∧
  weight_in_lb (weight_in_oz (area x 60)) = 15 * x ∧
  15 * x = 1350

theorem find_width_of_bobs_tv : ∃ x : ℕ, width_of_bobs_tv x := sorry

end find_width_of_bobs_tv_l1884_188424


namespace smallest_multiple_of_6_and_15_l1884_188415

theorem smallest_multiple_of_6_and_15 : ∃ a : ℕ, a > 0 ∧ a % 6 = 0 ∧ a % 15 = 0 ∧ ∀ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 → a ≤ b :=
  sorry

end smallest_multiple_of_6_and_15_l1884_188415


namespace fill_tank_time_l1884_188465

theorem fill_tank_time :
  ∀ (rate_fill rate_empty : ℝ), 
    rate_fill = 1 / 25 → 
    rate_empty = 1 / 50 → 
    (1/2) / (rate_fill - rate_empty) = 25 :=
by
  intros rate_fill rate_empty h_fill h_empty
  sorry

end fill_tank_time_l1884_188465
