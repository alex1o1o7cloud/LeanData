import Mathlib

namespace NUMINAMATH_GPT_sum_of_first_10_common_elements_eq_13981000_l280_28023

def arithmetic_prog (n : ℕ) : ℕ := 4 + 3 * n
def geometric_prog (k : ℕ) : ℕ := 20 * 2 ^ k

theorem sum_of_first_10_common_elements_eq_13981000 :
  let common_elements : List ℕ := 
    [40, 160, 640, 2560, 10240, 40960, 163840, 655360, 2621440, 10485760]
  let sum_common_elements : ℕ := common_elements.sum
  sum_common_elements = 13981000 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_10_common_elements_eq_13981000_l280_28023


namespace NUMINAMATH_GPT_students_not_next_each_other_l280_28082

open Nat

theorem students_not_next_each_other (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 5) (h2 : k = 2) (h3 : m = 3)
  (h4 : ∀ (A B : ℕ), A ≠ B) : 
  ∃ (total : ℕ), total = 3! * (choose (5-3+1) 2) := 
by
  sorry

end NUMINAMATH_GPT_students_not_next_each_other_l280_28082


namespace NUMINAMATH_GPT_dogs_not_eating_any_foods_l280_28060

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

end NUMINAMATH_GPT_dogs_not_eating_any_foods_l280_28060


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l280_28075

-- Define variables as real numbers or appropriate domains
variables {a b x y: ℝ}

-- Problem 1
theorem simplify_expression1 : (2 * a - b) - (2 * b - 3 * a) - 2 * (a - 2 * b) = 3 * a + b :=
by sorry

-- Problem 2
theorem simplify_expression2 : (4 * x^2 - 5 * x * y) - (1 / 3 * y^2 + 2 * x^2) + 2 * (3 * x * y - 1 / 4 * y^2 - 1 / 12 * y^2) = 2 * x^2 + x * y - y^2 :=
by sorry

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l280_28075


namespace NUMINAMATH_GPT_original_cost_of_tomatoes_correct_l280_28012

noncomputable def original_cost_of_tomatoes := 
  let original_order := 25
  let new_tomatoes := 2.20
  let new_lettuce := 1.75
  let old_lettuce := 1.00
  let new_celery := 2.00
  let old_celery := 1.96
  let delivery_tip := 8
  let new_total_bill := 35
  let new_groceries := new_total_bill - delivery_tip
  let increase_in_cost := (new_lettuce - old_lettuce) + (new_celery - old_celery)
  let difference_due_to_substitutions := new_groceries - original_order
  let x := new_tomatoes + (difference_due_to_substitutions - increase_in_cost)
  x

theorem original_cost_of_tomatoes_correct :
  original_cost_of_tomatoes = 3.41 := by
  sorry

end NUMINAMATH_GPT_original_cost_of_tomatoes_correct_l280_28012


namespace NUMINAMATH_GPT_line_intersects_y_axis_at_0_2_l280_28003

theorem line_intersects_y_axis_at_0_2 (P1 P2 : ℝ × ℝ) (h1 : P1 = (2, 8)) (h2 : P2 = (6, 20)) :
  ∃ y : ℝ, (0, y) = (0, 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_line_intersects_y_axis_at_0_2_l280_28003


namespace NUMINAMATH_GPT_production_problem_l280_28084

theorem production_problem (x y : ℝ) (h₁ : x > 0) (h₂ : ∀ k : ℝ, x * x * x * k = x) : (x * x * y * (1 / (x^2)) = y) :=
by {
  sorry
}

end NUMINAMATH_GPT_production_problem_l280_28084


namespace NUMINAMATH_GPT_correct_operation_l280_28028

theorem correct_operation :
  (∀ {a : ℝ}, a^6 / a^3 = a^3) = false ∧
  (∀ {a b : ℝ}, (a + b) * (a - b) = a^2 - b^2) ∧
  (∀ {a : ℝ}, (-a^3)^3 = -a^9) = false ∧
  (∀ {a : ℝ}, 2 * a^2 + 3 * a^3 = 5 * a^5) = false :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l280_28028


namespace NUMINAMATH_GPT_slowest_bailing_rate_proof_l280_28063

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

end NUMINAMATH_GPT_slowest_bailing_rate_proof_l280_28063


namespace NUMINAMATH_GPT_random_events_count_is_five_l280_28077

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

end NUMINAMATH_GPT_random_events_count_is_five_l280_28077


namespace NUMINAMATH_GPT_statement_A_statement_B_statement_C_statement_D_l280_28053

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

end NUMINAMATH_GPT_statement_A_statement_B_statement_C_statement_D_l280_28053


namespace NUMINAMATH_GPT_geometric_vs_arithmetic_l280_28001

-- Definition of a positive geometric progression
def positive_geometric_progression (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = q * a n ∧ q > 0

-- Definition of an arithmetic progression
def arithmetic_progression (b : ℕ → ℝ) (d : ℝ) := ∀ n, b (n + 1) = b n + d

-- Theorem statement based on the problem and conditions
theorem geometric_vs_arithmetic
  (a : ℕ → ℝ) (b : ℕ → ℝ) (q : ℝ) (d : ℝ)
  (h1 : positive_geometric_progression a q)
  (h2 : arithmetic_progression b d)
  (h3 : a 6 = b 7) :
  a 3 + a 9 ≥ b 4 + b 10 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_vs_arithmetic_l280_28001


namespace NUMINAMATH_GPT_units_digit_of_M_is_1_l280_28027

def Q (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  if units = 0 then 0 else tens / units

def T (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem units_digit_of_M_is_1 (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : b ≤ 9) (h₃ : 10*a + b = Q (10*a + b) + T (10*a + b)) :
  b = 1 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_M_is_1_l280_28027


namespace NUMINAMATH_GPT_max_right_angle_triangles_l280_28035

open Real

theorem max_right_angle_triangles (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x y : ℝ, x^2 + a^2 * y^2 = a^2) :
  ∃n : ℕ, n = 3 := 
by
  sorry

end NUMINAMATH_GPT_max_right_angle_triangles_l280_28035


namespace NUMINAMATH_GPT_probability_two_doors_open_l280_28032

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

end NUMINAMATH_GPT_probability_two_doors_open_l280_28032


namespace NUMINAMATH_GPT_total_texts_received_l280_28072

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

end NUMINAMATH_GPT_total_texts_received_l280_28072


namespace NUMINAMATH_GPT_radius_of_2007_l280_28005

-- Define the conditions
def given_condition (n : ℕ) (r : ℕ → ℝ) : Prop :=
  r 1 = 1 ∧ (∀ i, 1 ≤ i ∧ i < n → r (i + 1) = 3 * r i)

-- State the theorem we want to prove
theorem radius_of_2007 (r : ℕ → ℝ) : given_condition 2007 r → r 2007 = 3^2006 :=
by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_radius_of_2007_l280_28005


namespace NUMINAMATH_GPT_count_ways_with_3_in_M_count_ways_with_2_in_M_l280_28048

structure ArrangementConfig where
  positions : Fin 9 → ℕ
  unique_positions : ∀ (i j : Fin 9) (hi hj : i ≠ j), positions i ≠ positions j
  no_adjacent_same : ∀ (i : Fin 8), positions i ≠ positions (i + 1)

def count_arrangements (fixed_value : ℕ) (fixed_position : Fin 9) : ℕ :=
  -- Implementation of counting the valid arrangements
  sorry

theorem count_ways_with_3_in_M : count_arrangements 3 0 = 6 := sorry

theorem count_ways_with_2_in_M : count_arrangements 2 0 = 12 := sorry

end NUMINAMATH_GPT_count_ways_with_3_in_M_count_ways_with_2_in_M_l280_28048


namespace NUMINAMATH_GPT_visits_365_days_l280_28061

theorem visits_365_days : 
  let alice_visits := 3
  let beatrix_visits := 4
  let claire_visits := 5
  let total_days := 365
  ∃ days_with_exactly_two_visits, days_with_exactly_two_visits = 54 :=
by
  sorry

end NUMINAMATH_GPT_visits_365_days_l280_28061


namespace NUMINAMATH_GPT_rational_number_property_l280_28078

theorem rational_number_property 
  (x : ℚ) (a : ℤ) (ha : 1 ≤ a) : 
  (x ^ (⌊x⌋)) = a / 2 → (∃ k : ℤ, x = k) ∨ x = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_rational_number_property_l280_28078


namespace NUMINAMATH_GPT_overall_ranking_l280_28013

-- Define the given conditions
def total_participants := 99
def rank_number_theory := 16
def rank_combinatorics := 30
def rank_geometry := 23
def exams := ["geometry", "number_theory", "combinatorics"]
def final_ranking_strategy := "sum_of_scores"

-- Given: best possible rank and worst possible rank should be the same in this specific problem (from solution steps).
def best_possible_rank := 67
def worst_possible_rank := 67

-- Mathematically prove that 100 * best possible rank + worst possible rank = 167
theorem overall_ranking :
  100 * best_possible_rank + worst_possible_rank = 167 :=
by {
  -- Add the "sorry" here to skip the proof, as required:
  sorry
}

end NUMINAMATH_GPT_overall_ranking_l280_28013


namespace NUMINAMATH_GPT_seq_properties_l280_28010

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x

theorem seq_properties :
  (∀ n, a_n = -2 * (1 / 3) ^ n) ∧
  (∀ n, b_n = 2 * n - 1) ∧
  (∀ t m, (-1 ≤ m ∧ m ≤ 1) → (t^2 - 2 * m * t + 1/2 > T_n) ↔ (t < -2 ∨ t > 2)) ∧
  (∃ m n, 1 < m ∧ m < n ∧ T_1 * T_n = T_m^2 ∧ m = 2 ∧ n = 12) :=
sorry

end NUMINAMATH_GPT_seq_properties_l280_28010


namespace NUMINAMATH_GPT_julia_fourth_day_candies_l280_28093

-- Definitions based on conditions
def first_day (x : ℚ) := (1/5) * x
def second_day (x : ℚ) := (1/2) * (4/5) * x
def third_day (x : ℚ) := (1/2) * (2/5) * x
def fourth_day (x : ℚ) := (2/5) * x - (1/2) * (2/5) * x

-- The Lean statement to prove
theorem julia_fourth_day_candies (x : ℚ) (h : x ≠ 0): 
  fourth_day x / x = 1/5 :=
by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_julia_fourth_day_candies_l280_28093


namespace NUMINAMATH_GPT_problem1_problem2_l280_28049

-- Problem 1
theorem problem1 (α : ℝ) (h : (Real.tan α) / (Real.tan α - 1) = -1) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -5 / 3 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h : (Real.tan α) / (Real.tan α - 1) = -1) (h_quad : π < α ∧ α < 3 * π / 2) :
  Real.cos (-π + α) + Real.cos (π / 2 + α) = 3 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l280_28049


namespace NUMINAMATH_GPT_tangential_tetrahedron_triangle_impossibility_l280_28059

theorem tangential_tetrahedron_triangle_impossibility (a b c d : ℝ) 
  (h : ∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → x > 0) :
  ¬ (∀ (x y z : ℝ) , (x = a ∨ x = b ∨ x = c ∨ x = d) → 
    (y = a ∨ y = b ∨ y = c ∨ y = d) →
    (z = a ∨ z = b ∨ z = c ∨ z = d) → 
    x ≠ y → y ≠ z → z ≠ x → x + y > z ∧ x + z > y ∧ y + z > x) :=
sorry

end NUMINAMATH_GPT_tangential_tetrahedron_triangle_impossibility_l280_28059


namespace NUMINAMATH_GPT_fruits_turned_yellow_on_friday_l280_28038

theorem fruits_turned_yellow_on_friday :
  ∃ (F : ℕ), F + 2*F = 6 ∧ 14 - F - 2*F = 8 :=
by
  existsi 2
  sorry

end NUMINAMATH_GPT_fruits_turned_yellow_on_friday_l280_28038


namespace NUMINAMATH_GPT_hyperbola_focal_length_l280_28044

theorem hyperbola_focal_length :
  let a := 2
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  2 * c = 2 * Real.sqrt 7 := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_focal_length_l280_28044


namespace NUMINAMATH_GPT_dmitriev_is_older_l280_28083

variables (Alekseev Borisov Vasilyev Grigoryev Dima Dmitriev : ℤ)

def Lesha := Alekseev + 1
def Borya := Borisov + 2
def Vasya := Vasilyev + 3
def Grisha := Grigoryev + 4

theorem dmitriev_is_older :
  Dima + 10 = Dmitriev :=
sorry

end NUMINAMATH_GPT_dmitriev_is_older_l280_28083


namespace NUMINAMATH_GPT_payment_of_employee_B_l280_28024

-- Define the variables and conditions
variables (A B : ℝ) (total_payment : ℝ) (payment_ratio : ℝ)

-- Assume the given conditions
def conditions : Prop := 
  (A + B = total_payment) ∧ 
  (A = payment_ratio * B) ∧ 
  (total_payment = 550) ∧ 
  (payment_ratio = 1.5)

-- Prove the payment of employee B is 220 given the conditions
theorem payment_of_employee_B : conditions A B total_payment payment_ratio → B = 220 := 
by
  sorry

end NUMINAMATH_GPT_payment_of_employee_B_l280_28024


namespace NUMINAMATH_GPT_number_of_tens_in_sum_l280_28094

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end NUMINAMATH_GPT_number_of_tens_in_sum_l280_28094


namespace NUMINAMATH_GPT_flowers_per_bouquet_l280_28087

noncomputable def num_flowers_per_bouquet (total_flowers wilted_flowers bouquets : ℕ) : ℕ :=
  (total_flowers - wilted_flowers) / bouquets

theorem flowers_per_bouquet : num_flowers_per_bouquet 53 18 5 = 7 := by
  sorry

end NUMINAMATH_GPT_flowers_per_bouquet_l280_28087


namespace NUMINAMATH_GPT_num_perfect_squares_l280_28098

theorem num_perfect_squares (a b : ℤ) (h₁ : a = 100) (h₂ : b = 400) : 
  ∃ n : ℕ, (100 < n^2) ∧ (n^2 < 400) ∧ (n = 9) :=
by
  sorry

end NUMINAMATH_GPT_num_perfect_squares_l280_28098


namespace NUMINAMATH_GPT_yoojeong_initial_correct_l280_28017

variable (yoojeong_initial yoojeong_after marbles_given : ℕ)

-- Given conditions
axiom marbles_given_cond : marbles_given = 8
axiom yoojeong_after_cond : yoojeong_after = 24

-- Equation relating initial, given marbles, and marbles left
theorem yoojeong_initial_correct : 
  yoojeong_initial = yoojeong_after + marbles_given := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_yoojeong_initial_correct_l280_28017


namespace NUMINAMATH_GPT_logarithmic_product_l280_28031

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem logarithmic_product (a b : ℝ) (h1 : a ≠ b) (h2 : f a = f b) : a * b = 1 := by
  sorry

end NUMINAMATH_GPT_logarithmic_product_l280_28031


namespace NUMINAMATH_GPT_inequality_log_l280_28021

variable (a b c : ℝ)
variable (h1 : 1 < a)
variable (h2 : 1 < b)
variable (h3 : 1 < c)

theorem inequality_log (a b c : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c) : 
  2 * ( (Real.log a / Real.log b) / (a + b) + (Real.log b / Real.log c) / (b + c) + (Real.log c / Real.log a) / (c + a) ) 
  ≥ 9 / (a + b + c) := 
sorry

end NUMINAMATH_GPT_inequality_log_l280_28021


namespace NUMINAMATH_GPT_ad_value_l280_28036

variable (a b c d : ℝ)

-- Conditions
def geom_seq := b^2 = a * c ∧ c^2 = b * d
def vertex_of_parabola := (b = 1 ∧ c = 2)

-- Question
theorem ad_value (h_geom : geom_seq a b c d) (h_vertex : vertex_of_parabola b c) : a * d = 2 := by
  sorry

end NUMINAMATH_GPT_ad_value_l280_28036


namespace NUMINAMATH_GPT_pond_length_l280_28007

-- Define the dimensions and volume of the pond
def pond_width : ℝ := 15
def pond_depth : ℝ := 5
def pond_volume : ℝ := 1500

-- Define the length variable
variable (L : ℝ)

-- State that the volume relationship holds and L is the length we're solving for
theorem pond_length :
  pond_volume = L * pond_width * pond_depth → L = 20 :=
by
  sorry

end NUMINAMATH_GPT_pond_length_l280_28007


namespace NUMINAMATH_GPT_jimmy_income_l280_28099

variable (J : ℝ)

def rebecca_income : ℝ := 15000
def income_increase : ℝ := 3000
def rebecca_income_after_increase : ℝ := rebecca_income + income_increase
def combined_income : ℝ := 2 * rebecca_income_after_increase

theorem jimmy_income (h : rebecca_income_after_increase + J = combined_income) : 
  J = 18000 := by
  sorry

end NUMINAMATH_GPT_jimmy_income_l280_28099


namespace NUMINAMATH_GPT_joan_initial_balloons_l280_28026

-- Definitions using conditions from a)
def initial_balloons (lost : ℕ) (current : ℕ) : ℕ := lost + current

-- Statement of our equivalent math proof problem
theorem joan_initial_balloons : initial_balloons 2 7 = 9 := 
by
  -- Proof skipped using sorry
  sorry

end NUMINAMATH_GPT_joan_initial_balloons_l280_28026


namespace NUMINAMATH_GPT_slower_train_crosses_faster_in_36_seconds_l280_28058

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

end NUMINAMATH_GPT_slower_train_crosses_faster_in_36_seconds_l280_28058


namespace NUMINAMATH_GPT_simplify_fraction_l280_28009

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2 + 2) = 544 / 121 := 
by sorry

end NUMINAMATH_GPT_simplify_fraction_l280_28009


namespace NUMINAMATH_GPT_quadratic_root_range_l280_28070

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

end NUMINAMATH_GPT_quadratic_root_range_l280_28070


namespace NUMINAMATH_GPT_find_value_of_k_l280_28043

noncomputable def value_of_k (m n : ℝ) : ℝ :=
  let p := 0.4
  let point1 := (m, n)
  let point2 := (m + 2, n + p)
  let k := 5
  k

theorem find_value_of_k (m n : ℝ) : value_of_k m n = 5 :=
sorry

end NUMINAMATH_GPT_find_value_of_k_l280_28043


namespace NUMINAMATH_GPT_total_units_per_day_all_work_together_l280_28080

-- Conditions
def men := 250
def women := 150
def units_per_day_by_men := 15
def units_per_day_by_women := 3

-- Problem statement and proof
theorem total_units_per_day_all_work_together :
  units_per_day_by_men + units_per_day_by_women = 18 :=
sorry

end NUMINAMATH_GPT_total_units_per_day_all_work_together_l280_28080


namespace NUMINAMATH_GPT_sum_of_reciprocals_l280_28067

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l280_28067


namespace NUMINAMATH_GPT_find_width_of_bobs_tv_l280_28057

def area (w h : ℕ) : ℕ := w * h

def weight_in_oz (area : ℕ) : ℕ := area * 4

def weight_in_lb (weight_in_oz : ℕ) : ℕ := weight_in_oz / 16

def width_of_bobs_tv (x : ℕ) : Prop :=
  area 48 100 = 4800 ∧
  weight_in_lb (weight_in_oz (area 48 100)) = 1200 ∧
  weight_in_lb (weight_in_oz (area x 60)) = 15 * x ∧
  15 * x = 1350

theorem find_width_of_bobs_tv : ∃ x : ℕ, width_of_bobs_tv x := sorry

end NUMINAMATH_GPT_find_width_of_bobs_tv_l280_28057


namespace NUMINAMATH_GPT_evaluate_expression_l280_28033

theorem evaluate_expression (a b c : ℕ) (h1 : a = 12) (h2 : b = 8) (h3 : c = 3) :
  (a - b + c - (a - (b + c)) = 6) := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l280_28033


namespace NUMINAMATH_GPT_simple_interest_rate_l280_28006

theorem simple_interest_rate (P R : ℝ) (T : ℕ) (hT : T = 10) (h_double : P * 2 = P + P * R * T / 100) : R = 10 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l280_28006


namespace NUMINAMATH_GPT_math_problem_l280_28046

theorem math_problem (a b : ℕ) (h₁ : a = 6) (h₂ : b = 6) : 
  (a^3 + b^3) / (a^2 - a * b + b^2) = 12 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l280_28046


namespace NUMINAMATH_GPT_units_digit_G_100_l280_28068

def G (n : ℕ) : ℕ := 3 ^ (2 ^ n) + 1

theorem units_digit_G_100 : (G 100) % 10 = 2 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_G_100_l280_28068


namespace NUMINAMATH_GPT_ducks_among_non_falcons_l280_28095

-- Definitions based on conditions
def percentage_birds := 100
def percentage_ducks := 40
def percentage_cranes := 20
def percentage_falcons := 15
def percentage_pigeons := 25

-- Question converted into the statement
theorem ducks_among_non_falcons : 
  (percentage_ducks / (percentage_birds - percentage_falcons) * percentage_birds) = 47 :=
by
  sorry

end NUMINAMATH_GPT_ducks_among_non_falcons_l280_28095


namespace NUMINAMATH_GPT_cos_45_degree_l280_28054

theorem cos_45_degree : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_45_degree_l280_28054


namespace NUMINAMATH_GPT_stacy_height_last_year_l280_28097

-- Definitions for the conditions
def brother_growth := 1
def stacy_growth := brother_growth + 6
def stacy_current_height := 57
def stacy_last_years_height := stacy_current_height - stacy_growth

-- Proof statement
theorem stacy_height_last_year : stacy_last_years_height = 50 :=
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_stacy_height_last_year_l280_28097


namespace NUMINAMATH_GPT_quadratic_trinomial_form_l280_28085

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

end NUMINAMATH_GPT_quadratic_trinomial_form_l280_28085


namespace NUMINAMATH_GPT_total_fish_l280_28018

-- Conditions
def initial_fish : ℕ := 22
def given_fish : ℕ := 47

-- Question: Total fish Mrs. Sheridan has now
theorem total_fish : initial_fish + given_fish = 69 := by
  sorry

end NUMINAMATH_GPT_total_fish_l280_28018


namespace NUMINAMATH_GPT_average_marks_math_chem_l280_28062

variables (M P C : ℕ)

theorem average_marks_math_chem :
  (M + P = 20) → (C = P + 20) → (M + C) / 2 = 20 := 
by
  sorry

end NUMINAMATH_GPT_average_marks_math_chem_l280_28062


namespace NUMINAMATH_GPT_contrapositive_of_ab_eq_zero_l280_28064

theorem contrapositive_of_ab_eq_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → ab ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_ab_eq_zero_l280_28064


namespace NUMINAMATH_GPT_problem_statement_l280_28025

open Function

theorem problem_statement :
  ∃ g : ℝ → ℝ, 
    (g 1 = 2) ∧ 
    (∀ (x y : ℝ), g (x^2 - y^2) = (x - y) * (g x + g y)) ∧ 
    (g 3 = 6) := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l280_28025


namespace NUMINAMATH_GPT_smallest_multiple_of_6_and_15_l280_28071

theorem smallest_multiple_of_6_and_15 : ∃ a : ℕ, a > 0 ∧ a % 6 = 0 ∧ a % 15 = 0 ∧ ∀ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 → a ≤ b :=
  sorry

end NUMINAMATH_GPT_smallest_multiple_of_6_and_15_l280_28071


namespace NUMINAMATH_GPT_noel_baked_dozens_l280_28096

theorem noel_baked_dozens (total_students : ℕ) (percent_like_donuts : ℝ)
    (donuts_per_student : ℕ) (dozen : ℕ) (h_total_students : total_students = 30)
    (h_percent_like_donuts : percent_like_donuts = 0.80)
    (h_donuts_per_student : donuts_per_student = 2)
    (h_dozen : dozen = 12) :
    total_students * percent_like_donuts * donuts_per_student / dozen = 4 := 
by
  sorry

end NUMINAMATH_GPT_noel_baked_dozens_l280_28096


namespace NUMINAMATH_GPT_number_of_ways_to_arrange_BANANA_l280_28052

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_arrange_BANANA_l280_28052


namespace NUMINAMATH_GPT_function_value_sum_l280_28092

namespace MathProof

variable {f : ℝ → ℝ}

theorem function_value_sum :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 5) = f x) →
  f (1 / 3) = 2022 →
  f (1 / 2) = 17 →
  f (-7) + f 12 + f (16 / 3) + f (9 / 2) = 2005 :=
by
  intros h_odd h_periodic h_f13 h_f12
  sorry

end MathProof

end NUMINAMATH_GPT_function_value_sum_l280_28092


namespace NUMINAMATH_GPT_division_identity_l280_28047

theorem division_identity : 45 / 0.05 = 900 :=
by
  sorry

end NUMINAMATH_GPT_division_identity_l280_28047


namespace NUMINAMATH_GPT_probability_XiaoCong_project_A_probability_same_project_not_C_l280_28042

-- Definition of projects and conditions
inductive Project
| A | B | C

def XiaoCong : Project := sorry
def XiaoYing : Project := sorry

-- (1) Probability of Xiao Cong assigned to project A
theorem probability_XiaoCong_project_A : 
  (1 / 3 : ℝ) = 1 / 3 := 
by sorry

-- (2) Probability of Xiao Cong and Xiao Ying being assigned to the same project, given Xiao Ying not assigned to C
theorem probability_same_project_not_C : 
  (2 / 6 : ℝ) = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_probability_XiaoCong_project_A_probability_same_project_not_C_l280_28042


namespace NUMINAMATH_GPT_robotics_club_students_l280_28091

theorem robotics_club_students
  (total_students : ℕ)
  (cs_students : ℕ)
  (electronics_students : ℕ)
  (both_students : ℕ)
  (h1 : total_students = 80)
  (h2 : cs_students = 50)
  (h3 : electronics_students = 35)
  (h4 : both_students = 25) :
  total_students - (cs_students - both_students + electronics_students - both_students + both_students) = 20 :=
by
  sorry

end NUMINAMATH_GPT_robotics_club_students_l280_28091


namespace NUMINAMATH_GPT_product_of_complex_conjugates_l280_28076

theorem product_of_complex_conjugates (i : ℂ) (h : i^2 = -1) : (1 + i) * (1 - i) = 2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_complex_conjugates_l280_28076


namespace NUMINAMATH_GPT_student_can_escape_l280_28039

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

end NUMINAMATH_GPT_student_can_escape_l280_28039


namespace NUMINAMATH_GPT_constant_sequence_is_AP_and_GP_l280_28069

theorem constant_sequence_is_AP_and_GP (seq : ℕ → ℕ) (h : ∀ n, seq n = 7) :
  (∃ d, ∀ n, seq n = seq (n + 1) + d) ∧ (∃ r, ∀ n, seq (n + 1) = seq n * r) :=
by
  sorry

end NUMINAMATH_GPT_constant_sequence_is_AP_and_GP_l280_28069


namespace NUMINAMATH_GPT_inequality_proof_l280_28041

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) : (1 / x) < (1 / y) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l280_28041


namespace NUMINAMATH_GPT_room_volume_correct_l280_28011

variable (Length Width Height : ℕ) (Volume : ℕ)

-- Define the dimensions of the room
def roomLength := 100
def roomWidth := 10
def roomHeight := 10

-- Define the volume function
def roomVolume (l w h : ℕ) : ℕ := l * w * h

-- Theorem to prove the volume of the room
theorem room_volume_correct : roomVolume roomLength roomWidth roomHeight = 10000 := 
by
  -- roomVolume 100 10 10 = 10000
  sorry

end NUMINAMATH_GPT_room_volume_correct_l280_28011


namespace NUMINAMATH_GPT_problem_1_problem_2_l280_28002

def f (x : ℝ) (a : ℝ) : ℝ := |x + 2| - |x + a|

theorem problem_1 (a : ℝ) (h : a = 3) :
  ∀ x, f x a ≤ 1/2 → x ≥ -11/4 := sorry

theorem problem_2 (a : ℝ) :
  (∀ x, f x a ≤ a) → a ≥ 1 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l280_28002


namespace NUMINAMATH_GPT_triangle_area_l280_28079

/-- Given a triangle ABC with BC = 12 cm and AD perpendicular to BC with AD = 15 cm,
    prove that the area of triangle ABC is 90 square centimeters. -/
theorem triangle_area {BC AD : ℝ} (hBC : BC = 12) (hAD : AD = 15) :
  (1 / 2) * BC * AD = 90 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l280_28079


namespace NUMINAMATH_GPT_yard_length_l280_28051

theorem yard_length (trees : ℕ) (distance_per_gap : ℕ) (gaps : ℕ) :
  trees = 26 → distance_per_gap = 16 → gaps = trees - 1 → length_of_yard = gaps * distance_per_gap → length_of_yard = 400 :=
by 
  intros h_trees h_distance_per_gap h_gaps h_length_of_yard
  sorry

end NUMINAMATH_GPT_yard_length_l280_28051


namespace NUMINAMATH_GPT_negation_of_forall_ge_zero_l280_28066

theorem negation_of_forall_ge_zero :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_forall_ge_zero_l280_28066


namespace NUMINAMATH_GPT_curve_crossing_l280_28029

structure Point where
  x : ℝ
  y : ℝ

def curve (t : ℝ) : Point :=
  { x := 2 * t^2 - 3, y := 2 * t^4 - 9 * t^2 + 6 }

theorem curve_crossing : ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve 1 = { x := -1, y := -1 } := by
  sorry

end NUMINAMATH_GPT_curve_crossing_l280_28029


namespace NUMINAMATH_GPT_find_second_divisor_l280_28030

theorem find_second_divisor :
  ∃ x : ℕ, 377 / 13 / x * (1/4 : ℚ) / 2 = 0.125 ∧ x = 29 :=
by
  use 29
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_find_second_divisor_l280_28030


namespace NUMINAMATH_GPT_minimum_value_l280_28073

theorem minimum_value (x y : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : x + y = 1) : 
  ∃ z, z = 9 ∧ (forall x y, x > 0 ∧ y > 0 ∧ x + y = 1 → (1/x + 4/y) ≥ z) := 
sorry

end NUMINAMATH_GPT_minimum_value_l280_28073


namespace NUMINAMATH_GPT_total_gym_cost_l280_28086

def cheap_monthly_fee : ℕ := 10
def cheap_signup_fee : ℕ := 50
def expensive_monthly_fee : ℕ := 3 * cheap_monthly_fee
def expensive_signup_fee : ℕ := 4 * expensive_monthly_fee

def yearly_cost_cheap : ℕ := 12 * cheap_monthly_fee + cheap_signup_fee
def yearly_cost_expensive : ℕ := 12 * expensive_monthly_fee + expensive_signup_fee

theorem total_gym_cost : yearly_cost_cheap + yearly_cost_expensive = 650 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_gym_cost_l280_28086


namespace NUMINAMATH_GPT_systematic_sampling_number_l280_28045

theorem systematic_sampling_number {n m s a b c d : ℕ} (h_n : n = 60) (h_m : m = 4) 
  (h_s : s = 3) (h_a : a = 33) (h_b : b = 48) 
  (h_gcd_1 : ∃ k, s + k * (n / m) = a) (h_gcd_2 : ∃ k, a + k * (n / m) = b) :
  ∃ k, s + k * (n / m) = d → d = 18 := by
  sorry

end NUMINAMATH_GPT_systematic_sampling_number_l280_28045


namespace NUMINAMATH_GPT_systematic_sampling_fourth_group_l280_28019

theorem systematic_sampling_fourth_group (n m k g2 g4 : ℕ) (h_class_size : n = 72)
  (h_sample_size : m = 6) (h_k : k = n / m) (h_group2 : g2 = 16) (h_group4 : g4 = g2 + 2 * k) :
  g4 = 40 := by
  sorry

end NUMINAMATH_GPT_systematic_sampling_fourth_group_l280_28019


namespace NUMINAMATH_GPT_sum_abcd_l280_28090

variable (a b c d : ℝ)

theorem sum_abcd :
  (∃ y : ℝ, 2 * a + 3 = y ∧ 2 * b + 4 = y ∧ 2 * c + 5 = y ∧ 2 * d + 6 = y ∧ a + b + c + d + 10 = y) →
  a + b + c + d = -11 :=
by
  sorry

end NUMINAMATH_GPT_sum_abcd_l280_28090


namespace NUMINAMATH_GPT_jane_doe_total_investment_mutual_funds_l280_28020

theorem jane_doe_total_investment_mutual_funds :
  ∀ (c m : ℝ) (total_investment : ℝ),
  total_investment = 250000 → m = 3 * c → c + m = total_investment → m = 187500 :=
by
  intros c m total_investment h_total h_relation h_sum
  sorry

end NUMINAMATH_GPT_jane_doe_total_investment_mutual_funds_l280_28020


namespace NUMINAMATH_GPT_sum_quotient_dividend_divisor_l280_28034

theorem sum_quotient_dividend_divisor (D : ℕ) (d : ℕ) (Q : ℕ) 
  (h1 : D = 54) (h2 : d = 9) (h3 : D = Q * d) : 
  (Q + D + d) = 69 :=
by
  sorry

end NUMINAMATH_GPT_sum_quotient_dividend_divisor_l280_28034


namespace NUMINAMATH_GPT_abs_inequality_solution_l280_28056

theorem abs_inequality_solution (x : ℝ) : (|x + 3| > x + 3) ↔ (x < -3) :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l280_28056


namespace NUMINAMATH_GPT_tournament_total_players_l280_28089

/--
In a tournament involving n players:
- Each player scored half of all their points in matches against participants who took the last three places.
- Each game results in 1 point.
- Total points from matches among the last three (bad) players = 3.
- The number of games between good and bad players = 3n - 9.
- Total points good players scored from bad players = 3n - 12.
- Games among good players total to (n-3)(n-4)/2 resulting points.
Prove that the total number of participants in the tournament is 9.
-/
theorem tournament_total_players (n : ℕ) :
  3 * (n - 4) = (n - 3) * (n - 4) / 2 → 
  n = 9 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_tournament_total_players_l280_28089


namespace NUMINAMATH_GPT_one_in_M_l280_28000

def N := { x : ℕ | true } -- Define the natural numbers ℕ

def M : Set ℕ := { x ∈ N | 1 / (x - 2) ≤ 0 }

theorem one_in_M : 1 ∈ M :=
  sorry

end NUMINAMATH_GPT_one_in_M_l280_28000


namespace NUMINAMATH_GPT_proof_shortest_side_l280_28016

-- Definitions based on problem conditions
def side_divided (a b : ℕ) : Prop := a + b = 20

def radius (r : ℕ) : Prop := r = 5

noncomputable def shortest_side (a b c : ℕ) : ℕ :=
  if a ≤ b ∧ a ≤ c then a
  else if b ≤ a ∧ b ≤ c then b
  else c

-- Proof problem statement
theorem proof_shortest_side {a b c : ℕ} (h1 : side_divided 9 11) (h2 : radius 5) :
  shortest_side 15 (11 + 9) (2 * 6 + 9) = 14 :=
sorry

end NUMINAMATH_GPT_proof_shortest_side_l280_28016


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l280_28074

-- Definitions for the conditions
variables {a b : ℝ}

-- Main theorem statement
theorem relationship_between_a_and_b (h1 : |Real.log (1 / 4) / Real.log a| = Real.log (1 / 4) / Real.log a)
  (h2 : |Real.log a / Real.log b| = -Real.log a / Real.log b) :
  0 < a ∧ a < 1 ∧ 1 < b :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l280_28074


namespace NUMINAMATH_GPT_spotted_and_fluffy_cats_l280_28004

theorem spotted_and_fluffy_cats (total_cats : ℕ) (total_cats_equiv : total_cats = 120) (one_third_spotted : ℕ → ℕ) (one_fourth_fluffy_spotted : ℕ → ℕ) :
  (one_third_spotted total_cats * one_fourth_fluffy_spotted (one_third_spotted total_cats) = 10) :=
by
  sorry

end NUMINAMATH_GPT_spotted_and_fluffy_cats_l280_28004


namespace NUMINAMATH_GPT_candidate1_fails_by_l280_28014

-- Define the total marks (T), passing marks (P), percentage marks (perc1 and perc2), and the extra marks.
def T : ℝ := 600
def P : ℝ := 160
def perc1 : ℝ := 0.20
def perc2 : ℝ := 0.30
def extra_marks : ℝ := 20

-- Define the marks obtained by the candidates.
def marks_candidate1 : ℝ := perc1 * T
def marks_candidate2 : ℝ := perc2 * T

-- The theorem stating the number of marks by which the first candidate fails.
theorem candidate1_fails_by (h_pass: perc2 * T = P + extra_marks) : P - marks_candidate1 = 40 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_candidate1_fails_by_l280_28014


namespace NUMINAMATH_GPT_days_playing_video_games_l280_28065

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

end NUMINAMATH_GPT_days_playing_video_games_l280_28065


namespace NUMINAMATH_GPT_muffin_to_banana_ratio_l280_28055

-- Definitions of costs
def elaine_cost (m b : ℝ) : ℝ := 5 * m + 4 * b
def derek_cost (m b : ℝ) : ℝ := 3 * m + 18 * b

-- The problem statement
theorem muffin_to_banana_ratio (m b : ℝ) (h : derek_cost m b = 3 * elaine_cost m b) : m / b = 2 :=
by
  sorry

end NUMINAMATH_GPT_muffin_to_banana_ratio_l280_28055


namespace NUMINAMATH_GPT_initial_sheep_count_l280_28037

theorem initial_sheep_count (S : ℕ) :
  let S1 := S - (S / 3 + 1 / 3)
  let S2 := S1 - (S1 / 4 + 1 / 4)
  let S3 := S2 - (S2 / 5 + 3 / 5)
  S3 = 409
  → S = 1025 := 
by 
  sorry

end NUMINAMATH_GPT_initial_sheep_count_l280_28037


namespace NUMINAMATH_GPT_quotient_three_l280_28008

theorem quotient_three (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a * b ∣ a^2 + b^2 + 1) :
  (a^2 + b^2 + 1) / (a * b) = 3 :=
sorry

end NUMINAMATH_GPT_quotient_three_l280_28008


namespace NUMINAMATH_GPT_total_votes_cast_l280_28022

theorem total_votes_cast (S : ℝ) (x : ℝ) (h1 : S = 120) (h2 : S = 0.72 * x - 0.28 * x) : x = 273 := by
  sorry

end NUMINAMATH_GPT_total_votes_cast_l280_28022


namespace NUMINAMATH_GPT_total_pizza_slices_correct_l280_28081

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

end NUMINAMATH_GPT_total_pizza_slices_correct_l280_28081


namespace NUMINAMATH_GPT_sample_capacity_is_480_l280_28015

-- Problem conditions
def total_people : ℕ := 500 + 400 + 300
def selection_probability : ℝ := 0.4

-- Statement: Prove that sample capacity n equals 480
theorem sample_capacity_is_480 (n : ℕ) (h : n / total_people = selection_probability) : n = 480 := by
  sorry

end NUMINAMATH_GPT_sample_capacity_is_480_l280_28015


namespace NUMINAMATH_GPT_females_who_chose_malt_l280_28050

-- Definitions
def total_cheerleaders : ℕ := 26
def total_males : ℕ := 10
def total_females : ℕ := 16
def males_who_chose_malt : ℕ := 6

-- Main statement
theorem females_who_chose_malt (C M F : ℕ) (hM : M = 2 * C) (h_total : C + M = total_cheerleaders) (h_males_malt : males_who_chose_malt = total_males) : F = 10 :=
sorry

end NUMINAMATH_GPT_females_who_chose_malt_l280_28050


namespace NUMINAMATH_GPT_find_M_N_sum_l280_28088

theorem find_M_N_sum
  (M N : ℕ)
  (h1 : 3 * 75 = 5 * M)
  (h2 : 3 * N = 5 * 90) :
  M + N = 195 := 
sorry

end NUMINAMATH_GPT_find_M_N_sum_l280_28088


namespace NUMINAMATH_GPT_complex_square_eq_l280_28040

theorem complex_square_eq (i : ℂ) (hi : i * i = -1) : (1 + i)^2 = 2 * i := 
by {
  -- marking the end of existing code for clarity
  sorry
}

end NUMINAMATH_GPT_complex_square_eq_l280_28040
