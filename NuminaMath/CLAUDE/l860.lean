import Mathlib

namespace NUMINAMATH_CALUDE_rainfall_problem_l860_86004

/-- Rainfall problem --/
theorem rainfall_problem (monday_rain tuesday_rain wednesday_rain thursday_rain friday_rain : ℝ)
  (h_monday : monday_rain = 3)
  (h_tuesday : tuesday_rain = 2 * monday_rain)
  (h_wednesday : wednesday_rain = 0)
  (h_friday : friday_rain = monday_rain + tuesday_rain + wednesday_rain + thursday_rain)
  (h_average : (monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain) / 7 = 4) :
  thursday_rain = 5 := by
sorry

end NUMINAMATH_CALUDE_rainfall_problem_l860_86004


namespace NUMINAMATH_CALUDE_sqrt_inequality_and_trig_identity_l860_86001

theorem sqrt_inequality_and_trig_identity :
  (∀ (α : Real),
    Real.sqrt 8 - Real.sqrt 6 < Real.sqrt 5 - Real.sqrt 3 ∧
    Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_and_trig_identity_l860_86001


namespace NUMINAMATH_CALUDE_at_least_one_to_beijing_l860_86065

-- Define the probabilities
def prob_A : ℝ := 0.9
def prob_B : ℝ := 0.8
def prob_C : ℝ := 0.7

-- Define the theorem
theorem at_least_one_to_beijing :
  1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) = 0.994 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_to_beijing_l860_86065


namespace NUMINAMATH_CALUDE_rationalize_denominator_l860_86009

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 2 + Real.sqrt 5 + Real.sqrt 7) =
    (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 7 + D * Real.sqrt E) / F ∧
    A = 5 ∧ B = 4 ∧ C = -1 ∧ D = 1 ∧ E = 70 ∧ F = 20 ∧ F > 0 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l860_86009


namespace NUMINAMATH_CALUDE_green_hats_count_l860_86077

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_cost : ℚ) :
  total_hats = 85 →
  blue_cost = 6 →
  green_cost = 7 →
  total_cost = 540 →
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_cost ∧
    green_hats = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_green_hats_count_l860_86077


namespace NUMINAMATH_CALUDE_inequality_proof_l860_86093

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_increasing : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 5 → f x < f y

-- State the theorem
theorem inequality_proof : f 4 > f (-Real.pi) ∧ f (-Real.pi) > f 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l860_86093


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l860_86046

theorem sin_cos_sum_equals_one : 
  Real.sin (65 * π / 180) * Real.sin (115 * π / 180) + 
  Real.cos (65 * π / 180) * Real.sin (25 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l860_86046


namespace NUMINAMATH_CALUDE_fib_equation_solutions_l860_86061

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The set of solutions to the Fibonacci equation -/
def fibSolutions : Set (ℕ × ℕ) :=
  {p : ℕ × ℕ | 5 * fib p.1 - 3 * fib p.2 = 1}

theorem fib_equation_solutions :
  fibSolutions = {(2, 3), (5, 8), (8, 13)} := by sorry

end NUMINAMATH_CALUDE_fib_equation_solutions_l860_86061


namespace NUMINAMATH_CALUDE_remainder_after_adding_1470_l860_86029

theorem remainder_after_adding_1470 (n : ℤ) (h : n % 7 = 2) : (n + 1470) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_1470_l860_86029


namespace NUMINAMATH_CALUDE_fourth_side_length_l860_86038

/-- A quadrilateral inscribed in a circle with specific properties -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The length of three sides of the quadrilateral -/
  side_length : ℝ
  /-- Assertion that the quadrilateral is a kite with two equal consecutive sides -/
  is_kite : Prop
  /-- Assertion that one diagonal is a diameter of the circle -/
  diagonal_is_diameter : Prop

/-- The theorem stating the length of the fourth side of the quadrilateral -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
  (h1 : q.radius = 150 * Real.sqrt 2)
  (h2 : q.side_length = 150) :
  ∃ (fourth_side : ℝ), fourth_side = 150 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l860_86038


namespace NUMINAMATH_CALUDE_cups_per_girl_l860_86081

theorem cups_per_girl (total_students : Nat) (boys : Nat) (cups_per_boy : Nat) (total_cups : Nat)
  (h1 : total_students = 30)
  (h2 : boys = 10)
  (h3 : cups_per_boy = 5)
  (h4 : total_cups = 90)
  (h5 : boys * 2 = total_students - boys) :
  (total_cups - boys * cups_per_boy) / (total_students - boys) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cups_per_girl_l860_86081


namespace NUMINAMATH_CALUDE_bridge_length_l860_86032

/-- The length of a bridge given a train crossing it -/
theorem bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 100 →
  crossing_time = 60 →
  train_speed = 5 →
  train_speed * crossing_time - train_length = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l860_86032


namespace NUMINAMATH_CALUDE_value_calculation_l860_86010

theorem value_calculation (number : ℕ) (value : ℕ) (h1 : number = 16) (h2 : value = 2 * number - 12) : value = 20 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l860_86010


namespace NUMINAMATH_CALUDE_total_turkey_cost_l860_86031

def turkey_cost (weight : ℕ) (price_per_kg : ℕ) : ℕ := weight * price_per_kg

theorem total_turkey_cost : 
  let first_turkey := 6
  let second_turkey := 9
  let third_turkey := 2 * second_turkey
  let price_per_kg := 2
  turkey_cost first_turkey price_per_kg + 
  turkey_cost second_turkey price_per_kg + 
  turkey_cost third_turkey price_per_kg = 66 := by
sorry

end NUMINAMATH_CALUDE_total_turkey_cost_l860_86031


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l860_86041

theorem complex_fraction_evaluation (u v : ℂ) (hu : u ≠ 0) (hv : v ≠ 0) 
  (h : u^2 + u*v + v^2 = 0) : 
  (u^7 + v^7) / (u + v)^7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l860_86041


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l860_86049

/-- A quadratic equation with parameter m -/
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 + 2 * m * x + m + 1

/-- Condition for the quadratic equation to have two distinct real roots -/
def has_distinct_real_roots (m : ℝ) : Prop :=
  (2 * m)^2 - 4 * (m - 3) * (m + 1) > 0

/-- Condition for the roots not being opposites of each other -/
def roots_not_opposite (m : ℝ) : Prop := m ≠ 0

/-- The range of m satisfying both conditions -/
def valid_m_range (m : ℝ) : Prop :=
  m > -3/2 ∧ m ≠ 0 ∧ m ≠ 3

theorem quadratic_equation_roots_condition :
  ∀ m : ℝ, has_distinct_real_roots m ∧ roots_not_opposite m ↔ valid_m_range m :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l860_86049


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l860_86063

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ ∀ x, ¬ P x :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 4 < 0) ↔ (∀ x : ℝ, x^2 - x + 4 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l860_86063


namespace NUMINAMATH_CALUDE_base_8_45327_equals_19159_l860_86079

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_45327_equals_19159 :
  base_8_to_10 [7, 2, 3, 5, 4] = 19159 := by
  sorry

end NUMINAMATH_CALUDE_base_8_45327_equals_19159_l860_86079


namespace NUMINAMATH_CALUDE_novelists_to_poets_ratio_l860_86068

def total_people : ℕ := 24
def novelists : ℕ := 15

def poets : ℕ := total_people - novelists

theorem novelists_to_poets_ratio :
  (novelists : ℚ) / (poets : ℚ) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_novelists_to_poets_ratio_l860_86068


namespace NUMINAMATH_CALUDE_number_with_fraction_difference_l860_86024

theorem number_with_fraction_difference (x : ℤ) : x - (7 : ℤ) * x / (13 : ℤ) = 110 ↔ x = 237 := by
  sorry

end NUMINAMATH_CALUDE_number_with_fraction_difference_l860_86024


namespace NUMINAMATH_CALUDE_yellow_peaches_count_red_yellow_relation_l860_86083

/-- The number of yellow peaches in the basket -/
def yellow_peaches : ℕ := 11

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 19

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 12

/-- The difference between red and yellow peaches -/
def red_yellow_difference : ℕ := 8

theorem yellow_peaches_count : yellow_peaches = 11 := by
  sorry

theorem red_yellow_relation : red_peaches = yellow_peaches + red_yellow_difference := by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_count_red_yellow_relation_l860_86083


namespace NUMINAMATH_CALUDE_base_prime_rep_1170_l860_86036

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def base_prime_representation (n : ℕ) : List ℕ := sorry

theorem base_prime_rep_1170 :
  base_prime_representation 1170 = [1, 2, 1, 0, 0, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_base_prime_rep_1170_l860_86036


namespace NUMINAMATH_CALUDE_cube_volume_increase_cube_volume_not_8_times_l860_86067

theorem cube_volume_increase (edge : ℝ) (edge_positive : 0 < edge) : 
  (2 * edge)^3 = 27 * edge^3 := by sorry

theorem cube_volume_not_8_times (edge : ℝ) (edge_positive : 0 < edge) : 
  (2 * edge)^3 ≠ 8 * edge^3 := by sorry

end NUMINAMATH_CALUDE_cube_volume_increase_cube_volume_not_8_times_l860_86067


namespace NUMINAMATH_CALUDE_value_added_to_numbers_l860_86026

theorem value_added_to_numbers (n : ℕ) (original_avg new_avg x : ℝ) 
  (h1 : n = 15)
  (h2 : original_avg = 40)
  (h3 : new_avg = 53)
  (h4 : n * new_avg = n * original_avg + n * x) :
  x = 13 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_numbers_l860_86026


namespace NUMINAMATH_CALUDE_sequence_property_l860_86019

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := 2 * a n - a 1

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n, sequence_sum a n = 2 * a n - a 1) →
  (2 * (a 2 + 1) = a 3 + a 1) →
  ∀ n, a n = 2^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l860_86019


namespace NUMINAMATH_CALUDE_diamond_crack_l860_86006

theorem diamond_crack (p p₁ p₂ : ℝ) (h₁ : p > 0) (h₂ : p₁ > 0) (h₃ : p₂ > 0) 
  (h₄ : p₁ + p₂ = p) (h₅ : p₁^2 + p₂^2 = 0.68 * p^2) : 
  min p₁ p₂ / p = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_diamond_crack_l860_86006


namespace NUMINAMATH_CALUDE_octal_digit_reversal_difference_l860_86094

theorem octal_digit_reversal_difference (A B : Nat) : 
  A ≠ B → 
  A < 8 → 
  B < 8 → 
  ∃ k : Int, (8 * A + B) - (8 * B + A) = 7 * k ∧ k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_octal_digit_reversal_difference_l860_86094


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l860_86085

theorem triangle_angle_difference (a b c : ℝ) : 
  a = 32 →
  b = 96 →
  c = 52 →
  b = 3 * a →
  2 * a - c = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_difference_l860_86085


namespace NUMINAMATH_CALUDE_table_cost_l860_86097

theorem table_cost (chair_cost : ℚ → ℚ) (total_cost : ℚ → ℚ) : 
  (∀ t, chair_cost t = t / 7) →
  (∀ t, total_cost t = t + 4 * (chair_cost t)) →
  (∃ t, total_cost t = 220) →
  (∃ t, t = 140 ∧ total_cost t = 220) :=
by sorry

end NUMINAMATH_CALUDE_table_cost_l860_86097


namespace NUMINAMATH_CALUDE_popped_kernels_problem_l860_86086

theorem popped_kernels_problem (bag1_popped bag1_total bag2_popped bag2_total bag3_popped : ℕ)
  (h1 : bag1_popped = 60)
  (h2 : bag1_total = 75)
  (h3 : bag2_popped = 42)
  (h4 : bag2_total = 50)
  (h5 : bag3_popped = 82)
  (h6 : (bag1_popped : ℚ) / bag1_total + (bag2_popped : ℚ) / bag2_total + (bag3_popped : ℚ) / bag3_total = 82 * 3 / 100) :
  bag3_total = 100 := by
  sorry

end NUMINAMATH_CALUDE_popped_kernels_problem_l860_86086


namespace NUMINAMATH_CALUDE_tangent_lines_existence_l860_86095

/-- The range of a for which there exist two different lines tangent to both f(x) and g(x) -/
theorem tangent_lines_existence (a : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₃ ∧ x₁ > 0 ∧ x₃ > 0 ∧
    (2 + a * Real.log x₁) + (a / x₁) * (x₂ - x₁) = (a * x₂^2 + 1) ∧
    (2 + a * Real.log x₃) + (a / x₃) * (x₄ - x₃) = (a * x₄^2 + 1)) ↔
  (a < 0 ∨ a > 2 / (1 + Real.log 2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_existence_l860_86095


namespace NUMINAMATH_CALUDE_prob_three_diff_suits_probability_three_different_suits_l860_86020

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- The number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- The number of cards in each suit -/
def CardsPerSuit : ℕ := StandardDeck / NumSuits

/-- The probability of picking three cards of different suits from a standard deck without replacement -/
theorem prob_three_diff_suits : 
  (39 / 51) * (26 / 50) = 169 / 425 := by sorry

/-- The main theorem: probability of picking three cards of different suits -/
theorem probability_three_different_suits :
  let p := (CardsPerSuit * (NumSuits - 1) / (StandardDeck - 1)) * 
           (CardsPerSuit * (NumSuits - 2) / (StandardDeck - 2))
  p = 169 / 425 := by sorry

end NUMINAMATH_CALUDE_prob_three_diff_suits_probability_three_different_suits_l860_86020


namespace NUMINAMATH_CALUDE_smallest_prime_after_five_nonprimes_l860_86044

/-- A function that returns true if a natural number is prime, false otherwise -/
def is_prime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nth_prime (n : ℕ) : ℕ := sorry

/-- A function that returns true if there are at least five consecutive nonprime numbers before n, false otherwise -/
def five_consecutive_nonprimes_before (n : ℕ) : Prop := sorry

theorem smallest_prime_after_five_nonprimes : 
  ∃ (n : ℕ), is_prime n ∧ five_consecutive_nonprimes_before n ∧ 
  ∀ (m : ℕ), m < n → ¬(is_prime m ∧ five_consecutive_nonprimes_before m) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_five_nonprimes_l860_86044


namespace NUMINAMATH_CALUDE_smallest_m_for_equal_notebooks_and_pencils_l860_86033

theorem smallest_m_for_equal_notebooks_and_pencils :
  ∃ (M : ℕ+), (M = 5) ∧
  (∀ (k : ℕ+), k < M → ¬∃ (n : ℕ+), 3 * k = 5 * n) ∧
  (∃ (n : ℕ+), 3 * M = 5 * n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_for_equal_notebooks_and_pencils_l860_86033


namespace NUMINAMATH_CALUDE_fraction_expression_equality_l860_86016

theorem fraction_expression_equality : (3/7 + 5/8) / (5/12 + 2/9) = 531/322 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_equality_l860_86016


namespace NUMINAMATH_CALUDE_box_fill_rate_l860_86069

-- Define the box dimensions
def box_length : ℝ := 7
def box_width : ℝ := 6
def box_height : ℝ := 2

-- Define the time to fill the box
def fill_time : ℝ := 21

-- Calculate the volume of the box
def box_volume : ℝ := box_length * box_width * box_height

-- Define the theorem
theorem box_fill_rate :
  box_volume / fill_time = 4 := by sorry

end NUMINAMATH_CALUDE_box_fill_rate_l860_86069


namespace NUMINAMATH_CALUDE_garden_length_ratio_l860_86072

/-- Given a rectangular property and a rectangular garden, this theorem proves
    the ratio of the garden's length to the property's length. -/
theorem garden_length_ratio
  (property_length : ℝ)
  (property_width : ℝ)
  (garden_area : ℝ)
  (h_property_length : property_length = 2250)
  (h_property_width : property_width = 1000)
  (h_garden_area : garden_area = 28125)
  (garden_width : ℝ)
  (h_garden_width_pos : garden_width > 0) :
  garden_area / garden_width / property_length = 12.5 / garden_width :=
by sorry

end NUMINAMATH_CALUDE_garden_length_ratio_l860_86072


namespace NUMINAMATH_CALUDE_cooking_dishes_time_is_one_point_five_l860_86030

/-- Represents the daily schedule of a working mom -/
structure DailySchedule where
  total_awake_time : ℝ
  work_time : ℝ
  gym_time : ℝ
  bathing_time : ℝ
  homework_bedtime : ℝ
  packing_lunches : ℝ
  cleaning_time : ℝ
  shower_leisure : ℝ

/-- Calculates the time spent on cooking and dishes -/
def cooking_dishes_time (schedule : DailySchedule) : ℝ :=
  schedule.total_awake_time - (schedule.work_time + schedule.gym_time + 
  schedule.bathing_time + schedule.homework_bedtime + schedule.packing_lunches + 
  schedule.cleaning_time + schedule.shower_leisure)

/-- Theorem stating that the cooking and dishes time for the given schedule is 1.5 hours -/
theorem cooking_dishes_time_is_one_point_five (schedule : DailySchedule) 
  (h1 : schedule.total_awake_time = 16)
  (h2 : schedule.work_time = 8)
  (h3 : schedule.gym_time = 2)
  (h4 : schedule.bathing_time = 0.5)
  (h5 : schedule.homework_bedtime = 1)
  (h6 : schedule.packing_lunches = 0.5)
  (h7 : schedule.cleaning_time = 0.5)
  (h8 : schedule.shower_leisure = 2) :
  cooking_dishes_time schedule = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_cooking_dishes_time_is_one_point_five_l860_86030


namespace NUMINAMATH_CALUDE_license_plate_combinations_license_plate_count_l860_86052

theorem license_plate_combinations : ℕ :=
  let letter_choices : ℕ := 26
  let digit_choices : ℕ := 10
  let letter_positions : ℕ := 3
  let digit_positions : ℕ := 4
  letter_choices ^ letter_positions * digit_choices ^ digit_positions

theorem license_plate_count : license_plate_combinations = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_license_plate_count_l860_86052


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l860_86042

/-- Given a quadratic function f(x) = ax² + bx + c, if f(2) - f(-2) = 4, then b = 1. -/
theorem quadratic_coefficient (a b c : ℝ) (y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = 4 →
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l860_86042


namespace NUMINAMATH_CALUDE_valid_selection_count_l860_86059

/-- Represents a dad in the TV show -/
structure Dad :=
  (id : Nat)

/-- Represents a kid in the TV show -/
structure Kid :=
  (id : Nat)
  (isGirl : Bool)
  (dad : Dad)

/-- Represents the selection of one dad and three kids -/
structure Selection :=
  (selectedDad : Dad)
  (selectedKids : Finset Kid)

/-- The set of all dads -/
def allDads : Finset Dad := sorry

/-- The set of all kids -/
def allKids : Finset Kid := sorry

/-- Kimi is a boy -/
def kimi : Kid := sorry

/-- Stone is a boy -/
def stone : Kid := sorry

/-- Predicate to check if a selection is valid -/
def isValidSelection (s : Selection) : Prop :=
  s.selectedKids.card = 3 ∧
  (∃ k ∈ s.selectedKids, k.isGirl) ∧
  (kimi ∈ s.selectedKids ↔ kimi.dad = s.selectedDad) ∧
  (stone ∈ s.selectedKids ↔ stone.dad ≠ s.selectedDad)

/-- The set of all possible valid selections -/
def allValidSelections : Finset Selection :=
  sorry

theorem valid_selection_count :
  allValidSelections.card = 12 :=
sorry

end NUMINAMATH_CALUDE_valid_selection_count_l860_86059


namespace NUMINAMATH_CALUDE_exists_special_sequence_l860_86022

def sequence_condition (a : ℕ → ℕ) : Prop :=
  ∀ i j p q r, i ≠ j ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r →
    Nat.gcd (a i + a j) (a p + a q + a r) = 1

theorem exists_special_sequence :
  ∃ a : ℕ → ℕ, sequence_condition a ∧ (∀ n, a n < a (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_exists_special_sequence_l860_86022


namespace NUMINAMATH_CALUDE_money_division_l860_86011

/-- Proves that the total amount of money divided among A, B, and C is 980,
    given the specified conditions. -/
theorem money_division (a b c : ℕ) : 
  b = 290 →            -- B's share is 290
  a = b + 40 →         -- A has 40 more than B
  c = a + 30 →         -- C has 30 more than A
  a + b + c = 980 :=   -- Total amount is 980
by
  sorry


end NUMINAMATH_CALUDE_money_division_l860_86011


namespace NUMINAMATH_CALUDE_largest_divisor_with_remainders_l860_86015

theorem largest_divisor_with_remainders : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℤ), 69 = k * n + 5) ∧ 
  (∃ (l : ℤ), 86 = l * n + 6) ∧ 
  (∀ (m : ℕ), m > n → 
    (¬∃ (k : ℤ), 69 = k * m + 5) ∨ 
    (¬∃ (l : ℤ), 86 = l * m + 6)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_with_remainders_l860_86015


namespace NUMINAMATH_CALUDE_f_extrema_g_negativity_l860_86008

noncomputable def f (x : ℝ) : ℝ := -x^2 + Real.log x

noncomputable def g (a x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x - 2*a*x

def interval : Set ℝ := Set.Icc (1/Real.exp 1) (Real.exp 1)

theorem f_extrema :
  ∃ (x_min x_max : ℝ), x_min ∈ interval ∧ x_max ∈ interval ∧
  (∀ x ∈ interval, f x ≥ f x_min) ∧
  (∀ x ∈ interval, f x ≤ f x_max) ∧
  f x_min = 1 - Real.exp 2 ∧
  f x_max = -1/2 - 1/2 * Real.log 2 :=
sorry

theorem g_negativity :
  ∀ a : ℝ, (∀ x > 2, g a x < 0) ↔ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_f_extrema_g_negativity_l860_86008


namespace NUMINAMATH_CALUDE_partnership_profit_l860_86058

/-- Represents the investment and profit distribution in a partnership --/
structure Partnership where
  /-- A's investment ratio relative to B --/
  a_ratio : ℚ
  /-- B's investment ratio relative to C --/
  b_ratio : ℚ
  /-- B's share of the profit --/
  b_share : ℕ

/-- Calculates the total profit given the partnership details --/
def calculate_total_profit (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that given the specified partnership details, the total profit is 7700 --/
theorem partnership_profit (p : Partnership) 
  (h1 : p.a_ratio = 3)
  (h2 : p.b_ratio = 2/3)
  (h3 : p.b_share = 1400) :
  calculate_total_profit p = 7700 :=
sorry

end NUMINAMATH_CALUDE_partnership_profit_l860_86058


namespace NUMINAMATH_CALUDE_truncated_trigonal_pyramid_theorem_l860_86023

/-- A truncated trigonal pyramid circumscribed around a sphere -/
structure TruncatedTrigonalPyramid where
  /-- The altitude of the pyramid -/
  h : ℝ
  /-- The circumradius of the lower base -/
  R₁ : ℝ
  /-- The circumradius of the upper base -/
  R₂ : ℝ
  /-- The distance from the circumcenter of the lower base to the point where the sphere touches it -/
  O₁T₁ : ℝ
  /-- The distance from the circumcenter of the upper base to the point where the sphere touches it -/
  O₂T₂ : ℝ
  /-- The sphere touches both bases of the pyramid -/
  touches_bases : True

/-- The main theorem about the relationship between the measurements of a truncated trigonal pyramid -/
theorem truncated_trigonal_pyramid_theorem (p : TruncatedTrigonalPyramid) :
  p.R₁ * p.R₂ * p.h^2 = (p.R₁^2 - p.O₁T₁^2) * (p.R₂^2 - p.O₂T₂^2) := by
  sorry

end NUMINAMATH_CALUDE_truncated_trigonal_pyramid_theorem_l860_86023


namespace NUMINAMATH_CALUDE_complex_equation_sum_l860_86070

theorem complex_equation_sum (x y : ℝ) :
  (x / (1 - Complex.I)) + (y / (1 - 2 * Complex.I)) = 5 / (1 - 3 * Complex.I) →
  x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l860_86070


namespace NUMINAMATH_CALUDE_population_change_factors_l860_86074

-- Define the factors that can affect population change
inductive PopulationFactor
  | NaturalGrowth
  | Migration
  | Mortality
  | BirthRate

-- Define a function that determines if a factor affects population change
def affectsPopulationChange (factor : PopulationFactor) : Prop :=
  match factor with
  | PopulationFactor.NaturalGrowth => true
  | PopulationFactor.Migration => true
  | _ => false

-- Theorem stating that population change is determined by natural growth and migration
theorem population_change_factors :
  ∀ (factor : PopulationFactor),
    affectsPopulationChange factor ↔
      (factor = PopulationFactor.NaturalGrowth ∨ factor = PopulationFactor.Migration) :=
by
  sorry


end NUMINAMATH_CALUDE_population_change_factors_l860_86074


namespace NUMINAMATH_CALUDE_main_theorem_l860_86002

/-- Proposition p: for all positive x, x + a/x ≥ 2 -/
def p (a : ℝ) : Prop :=
  ∀ x > 0, x + a / x ≥ 2

/-- Proposition q: for all real k, the line kx - y + 2 = 0 intersects with the ellipse x^2 + y^2/a^2 = 1 -/
def q (a : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x y : ℝ, k * x - y + 2 = 0 ∧ x^2 + y^2 / a^2 = 1

/-- The main theorem: (p ∨ q) ∧ ¬(p ∧ q) is true if and only if 1 ≤ a < 2 -/
theorem main_theorem (a : ℝ) (h : a > 0) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ 1 ≤ a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l860_86002


namespace NUMINAMATH_CALUDE_expression_simplification_l860_86073

theorem expression_simplification (n : ℝ) (h : n = Real.sqrt 2 + 1) :
  ((n + 3) / (n^2 - 1) - 1 / (n + 1)) / (2 / (n + 1)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l860_86073


namespace NUMINAMATH_CALUDE_total_discount_calculation_l860_86089

theorem total_discount_calculation (cost_price_A cost_price_B cost_price_C : ℝ)
  (markup_percentage : ℝ) (loss_percentage_A loss_percentage_B loss_percentage_C : ℝ)
  (h1 : cost_price_A = 200)
  (h2 : cost_price_B = 150)
  (h3 : cost_price_C = 100)
  (h4 : markup_percentage = 0.5)
  (h5 : loss_percentage_A = 0.01)
  (h6 : loss_percentage_B = 0.03)
  (h7 : loss_percentage_C = 0.02) :
  let marked_price (cp : ℝ) := cp * (1 + markup_percentage)
  let selling_price (cp : ℝ) (loss : ℝ) := cp * (1 - loss)
  let discount (cp : ℝ) (loss : ℝ) := marked_price cp - selling_price cp loss
  discount cost_price_A loss_percentage_A +
  discount cost_price_B loss_percentage_B +
  discount cost_price_C loss_percentage_C = 233.5 :=
by sorry


end NUMINAMATH_CALUDE_total_discount_calculation_l860_86089


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l860_86035

/-- The hyperbola defined by (x^2/9) - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/9 - y^2 = 1

/-- The line defined by y = (1/3)(x+1) -/
def line (x y : ℝ) : Prop := y = (1/3)*(x+1)

/-- The number of intersection points between the hyperbola and the line -/
def intersection_count : ℕ := 1

theorem hyperbola_line_intersection :
  ∃! p : ℝ × ℝ, hyperbola p.1 p.2 ∧ line p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l860_86035


namespace NUMINAMATH_CALUDE_correct_calculation_l860_86012

theorem correct_calculation (x : ℝ) : 3 * x = 135 → x / 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l860_86012


namespace NUMINAMATH_CALUDE_fgh_supermarket_difference_l860_86040

theorem fgh_supermarket_difference (total : ℕ) (us : ℕ) (h1 : total = 84) (h2 : us = 47) (h3 : us > total - us) : us - (total - us) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarket_difference_l860_86040


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l860_86047

theorem largest_five_digit_divisible_by_8 : 
  ∀ n : ℕ, n ≤ 99999 ∧ n ≥ 10000 ∧ n % 8 = 0 → n ≤ 99992 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l860_86047


namespace NUMINAMATH_CALUDE_sequence_general_term_l860_86098

/-- The sequence a_n defined by a_1 = 2 and a_{n+1} = 2a_n for n ≥ 1 has the general term a_n = 2^n -/
theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, a (n + 1) = 2 * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^n :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l860_86098


namespace NUMINAMATH_CALUDE_family_age_average_l860_86087

/-- Given the ages of four family members with specific relationships, 
    prove that their average age is 31.5 years. -/
theorem family_age_average (devin_age eden_age mom_age grandfather_age : ℕ) :
  devin_age = 12 →
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  grandfather_age = (devin_age + eden_age + mom_age) / 2 →
  (devin_age + eden_age + mom_age + grandfather_age : ℚ) / 4 = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_family_age_average_l860_86087


namespace NUMINAMATH_CALUDE_triangle_properties_l860_86013

/-- Triangle ABC with vertices A(-1,4), B(-2,-1), and C(2,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from BC in triangle ABC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => p.1 + p.2 - 3 = 0

/-- The area of triangle ABC -/
def area (t : Triangle) : ℝ := 8

theorem triangle_properties :
  let t : Triangle := { A := (-1, 4), B := (-2, -1), C := (2, 3) }
  (∀ p, altitude t p ↔ p.1 + p.2 - 3 = 0) ∧ area t = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l860_86013


namespace NUMINAMATH_CALUDE_count_D_two_eq_30_l860_86078

/-- The number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 127 for which D(n) = 2 -/
def count_D_two : ℕ := sorry

theorem count_D_two_eq_30 : count_D_two = 30 := by sorry

end NUMINAMATH_CALUDE_count_D_two_eq_30_l860_86078


namespace NUMINAMATH_CALUDE_claire_earnings_l860_86005

/-- Represents the total earnings from selling roses with discounts applied -/
def total_earnings (total_flowers : ℕ) (tulips : ℕ) (white_roses : ℕ) 
  (small_red_roses : ℕ) (medium_red_roses : ℕ) 
  (small_price : ℚ) (medium_price : ℚ) (large_price : ℚ) : ℚ :=
  let total_roses := total_flowers - tulips
  let red_roses := total_roses - white_roses
  let large_red_roses := red_roses - small_red_roses - medium_red_roses
  let small_sold := small_red_roses / 2
  let medium_sold := medium_red_roses / 2
  let large_sold := large_red_roses / 2
  let small_earnings := small_sold * small_price * (1 - 0.1)  -- 10% discount
  let medium_earnings := medium_sold * medium_price * (1 - 0.15)  -- 15% discount
  let large_earnings := large_sold * large_price * (1 - 0.15)  -- 15% discount
  small_earnings + medium_earnings + large_earnings

/-- Theorem stating that Claire's earnings are $92.13 -/
theorem claire_earnings : 
  total_earnings 400 120 80 40 60 0.75 1 1.25 = 92.13 := by
  sorry


end NUMINAMATH_CALUDE_claire_earnings_l860_86005


namespace NUMINAMATH_CALUDE_vector_dot_product_l860_86057

def vector_a : ℝ × ℝ := (-1, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (m, m - 4)
def vector_c (m : ℝ) : ℝ × ℝ := (2*m, 3)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product (m : ℝ) :
  parallel vector_a (vector_b m) →
  dot_product (vector_b m) (vector_c m) = -7 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l860_86057


namespace NUMINAMATH_CALUDE_andrew_fruit_purchase_cost_l860_86000

/-- Calculates the total cost of fruits purchased by Andrew -/
theorem andrew_fruit_purchase_cost : 
  let grapes_quantity : ℕ := 14
  let grapes_price : ℕ := 54
  let mangoes_quantity : ℕ := 10
  let mangoes_price : ℕ := 62
  let pineapple_quantity : ℕ := 8
  let pineapple_price : ℕ := 40
  let kiwi_quantity : ℕ := 5
  let kiwi_price : ℕ := 30
  let total_cost := 
    grapes_quantity * grapes_price + 
    mangoes_quantity * mangoes_price + 
    pineapple_quantity * pineapple_price + 
    kiwi_quantity * kiwi_price
  total_cost = 1846 := by
  sorry


end NUMINAMATH_CALUDE_andrew_fruit_purchase_cost_l860_86000


namespace NUMINAMATH_CALUDE_factorization_proof_l860_86003

theorem factorization_proof (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l860_86003


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l860_86017

theorem opposite_of_negative_two : 
  (∃ x : ℝ, -2 + x = 0) → (∃ x : ℝ, -2 + x = 0 ∧ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l860_86017


namespace NUMINAMATH_CALUDE_parallel_line_length_l860_86034

theorem parallel_line_length (base : ℝ) (h1 : base = 24) : 
  ∃ (parallel_line : ℝ), 
    parallel_line^2 / base^2 = 1/2 ∧ 
    parallel_line = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_length_l860_86034


namespace NUMINAMATH_CALUDE_reflect_x_of_P_l860_86018

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The theorem stating that reflecting the point P(-2, √5) across the x-axis 
    results in the point (-2, -√5) -/
theorem reflect_x_of_P : 
  let P : Point := { x := -2, y := Real.sqrt 5 }
  reflect_x P = { x := -2, y := -Real.sqrt 5 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_of_P_l860_86018


namespace NUMINAMATH_CALUDE_factorization_equality_l860_86037

theorem factorization_equality (x y : ℝ) : 1 - 2*(x - y) + (x - y)^2 = (1 - x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l860_86037


namespace NUMINAMATH_CALUDE_fifteenth_prime_l860_86091

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime :
  (nth_prime 5 = 11) → (nth_prime 15 = 47) :=
sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l860_86091


namespace NUMINAMATH_CALUDE_mean_median_difference_l860_86056

-- Define the score distribution
def score_60_percent : ℝ := 0.20
def score_75_percent : ℝ := 0.40
def score_85_percent : ℝ := 0.25
def score_95_percent : ℝ := 1 - (score_60_percent + score_75_percent + score_85_percent)

-- Define the scores
def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_85 : ℝ := 85
def score_95 : ℝ := 95

-- Calculate the mean score
def mean_score : ℝ :=
  score_60_percent * score_60 +
  score_75_percent * score_75 +
  score_85_percent * score_85 +
  score_95_percent * score_95

-- Define the median score
def median_score : ℝ := score_75

-- Theorem stating the difference between mean and median
theorem mean_median_difference :
  |mean_score - median_score| = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l860_86056


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l860_86099

theorem six_digit_divisibility (a b c : ℕ) 
  (h1 : a ≥ 1 ∧ a ≤ 9) 
  (h2 : b ≥ 0 ∧ b ≤ 9) 
  (h3 : c ≥ 0 ∧ c ≤ 9) : 
  (100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c) % 1001 = 0 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l860_86099


namespace NUMINAMATH_CALUDE_division_invariance_l860_86025

theorem division_invariance (a b : ℝ) (h : b ≠ 0) : (10 * a) / (10 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_division_invariance_l860_86025


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l860_86076

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  -- The area of the original quadrilateral
  area : ℝ
  -- The lengths of the sides and their extensions
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ

/-- Theorem stating the area of the extended quadrilateral -/
theorem extended_quadrilateral_area (q : ExtendedQuadrilateral)
  (h_area : q.area = 25)
  (h_ef : q.ef = 5)
  (h_fg : q.fg = 7)
  (h_gh : q.gh = 9)
  (h_he : q.he = 8) :
  q.area + 2 * q.area = 75 := by
  sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l860_86076


namespace NUMINAMATH_CALUDE_jqk_base14_to_binary_digits_l860_86090

def base14_to_decimal (j k q : ℕ) : ℕ := j * 14^2 + q * 14 + k

def count_binary_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem jqk_base14_to_binary_digits : 
  count_binary_digits (base14_to_decimal 11 13 12) = 11 := by
  sorry

end NUMINAMATH_CALUDE_jqk_base14_to_binary_digits_l860_86090


namespace NUMINAMATH_CALUDE_rectangle_perimeter_from_triangle_l860_86054

/-- Given a triangle with sides 5, 12, and 13 units, and a rectangle with width 5 units
    and area equal to the triangle's area, the perimeter of the rectangle is 22 units. -/
theorem rectangle_perimeter_from_triangle : 
  ∀ (triangle_side1 triangle_side2 triangle_side3 rectangle_width : ℝ),
  triangle_side1 = 5 →
  triangle_side2 = 12 →
  triangle_side3 = 13 →
  rectangle_width = 5 →
  (1/2) * triangle_side1 * triangle_side2 = rectangle_width * (1/2 * triangle_side1 * triangle_side2 / rectangle_width) →
  2 * (rectangle_width + (1/2 * triangle_side1 * triangle_side2 / rectangle_width)) = 22 :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_from_triangle_l860_86054


namespace NUMINAMATH_CALUDE_sum_x_and_5_nonpositive_l860_86053

theorem sum_x_and_5_nonpositive (x : ℝ) : (x + 5 ≤ 0) ↔ (∀ y : ℝ, y ≤ 0 → x + 5 ≤ y) := by
  sorry

end NUMINAMATH_CALUDE_sum_x_and_5_nonpositive_l860_86053


namespace NUMINAMATH_CALUDE_rectangle_max_area_rectangle_max_area_value_l860_86048

/-- Represents a rectangle with length, width, and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter : ℝ
  perimeterConstraint : perimeter = 2 * (length + width)

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: The area of a rectangle with fixed perimeter is maximized when it's a square -/
theorem rectangle_max_area (p : ℝ) (hp : p > 0) :
  ∃ (r : Rectangle), r.perimeter = p ∧
    ∀ (s : Rectangle), s.perimeter = p → r.area ≥ s.area ∧
    r.length = p / 4 ∧ r.width = p / 4 :=
  sorry

/-- Corollary: The maximum area of a rectangle with perimeter p is p^2 / 16 -/
theorem rectangle_max_area_value (p : ℝ) (hp : p > 0) :
  ∃ (r : Rectangle), r.perimeter = p ∧
    ∀ (s : Rectangle), s.perimeter = p → r.area ≥ s.area ∧
    r.area = p^2 / 16 :=
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_rectangle_max_area_value_l860_86048


namespace NUMINAMATH_CALUDE_houses_with_neither_l860_86028

theorem houses_with_neither (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ)
  (h_total : total = 70)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_both : both = 35) :
  total - (garage + pool - both) = 15 :=
by sorry

end NUMINAMATH_CALUDE_houses_with_neither_l860_86028


namespace NUMINAMATH_CALUDE_power_digit_cycle_l860_86084

theorem power_digit_cycle (n : ℤ) (k : ℕ) : n^(k+4) ≡ n^k [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_power_digit_cycle_l860_86084


namespace NUMINAMATH_CALUDE_fourth_term_equals_seven_l860_86039

/-- Given a sequence {a_n} where the sum of the first n terms S_n = n^2, prove that a_4 = 7 -/
theorem fourth_term_equals_seven (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2) : 
    a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_equals_seven_l860_86039


namespace NUMINAMATH_CALUDE_combined_molecular_weight_l860_86045

/-- The atomic mass of Calcium in g/mol -/
def Ca_mass : ℝ := 40.08

/-- The atomic mass of Bromine in g/mol -/
def Br_mass : ℝ := 79.904

/-- The atomic mass of Strontium in g/mol -/
def Sr_mass : ℝ := 87.62

/-- The atomic mass of Chlorine in g/mol -/
def Cl_mass : ℝ := 35.453

/-- The number of moles of Calcium bromide -/
def CaBr2_moles : ℝ := 4

/-- The number of moles of Strontium chloride -/
def SrCl2_moles : ℝ := 3

/-- The combined molecular weight of Calcium bromide and Strontium chloride -/
theorem combined_molecular_weight :
  CaBr2_moles * (Ca_mass + 2 * Br_mass) + SrCl2_moles * (Sr_mass + 2 * Cl_mass) = 1275.13 := by
  sorry

end NUMINAMATH_CALUDE_combined_molecular_weight_l860_86045


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l860_86021

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 576) 
  (h2 : height = 48) 
  (h3 : area = base * height) : 
  base = 12 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l860_86021


namespace NUMINAMATH_CALUDE_expression_evaluation_l860_86007

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^(y + 1) + 6 * y^(x + 1) + 2 * x * y = 2775 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l860_86007


namespace NUMINAMATH_CALUDE_imaginary_part_of_two_over_one_plus_i_l860_86062

theorem imaginary_part_of_two_over_one_plus_i :
  Complex.im (2 / (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_two_over_one_plus_i_l860_86062


namespace NUMINAMATH_CALUDE_fly_journey_l860_86082

theorem fly_journey (r : ℝ) (s : ℝ) (h1 : r = 65) (h2 : s = 90) :
  let d := 2 * r
  let b := Real.sqrt (d^2 - s^2)
  d + s + b = 314 :=
by sorry

end NUMINAMATH_CALUDE_fly_journey_l860_86082


namespace NUMINAMATH_CALUDE_product_relation_l860_86092

theorem product_relation (x y z : ℝ) (h : x^2 + y^2 = x*y*(z + 1/z)) :
  x = y*z ∨ y = x*z :=
by sorry

end NUMINAMATH_CALUDE_product_relation_l860_86092


namespace NUMINAMATH_CALUDE_existence_of_xy_for_nonconstant_f_l860_86027

theorem existence_of_xy_for_nonconstant_f (f : ℝ → ℝ) (h_nonconstant : ∃ a b, f a ≠ f b) :
  ∃ x y : ℝ, f (x + y) < f (x * y) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_for_nonconstant_f_l860_86027


namespace NUMINAMATH_CALUDE_flooring_rate_calculation_l860_86066

/-- Given a rectangular room with specified dimensions and total flooring cost,
    calculate the rate per square meter for flooring. -/
theorem flooring_rate_calculation
  (length : ℝ) (width : ℝ) (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500)
  : (total_cost / (length * width)) = 800 := by
  sorry

end NUMINAMATH_CALUDE_flooring_rate_calculation_l860_86066


namespace NUMINAMATH_CALUDE_perfect_squares_between_50_and_250_l860_86071

theorem perfect_squares_between_50_and_250 : 
  (Finset.filter (fun n => 50 < n * n ∧ n * n < 250) (Finset.range 16)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_50_and_250_l860_86071


namespace NUMINAMATH_CALUDE_average_equation_solution_l860_86014

theorem average_equation_solution (x : ℝ) : 
  ((x + 8) + (7 * x + 3) + (3 * x + 9)) / 3 = 5 * x - 10 → x = 12.5 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l860_86014


namespace NUMINAMATH_CALUDE_count_numbers_with_square_factor_eq_41_l860_86080

def perfect_squares : List Nat := [4, 9, 16, 25, 36, 49, 64, 100]

def is_divisible_by_square (n : Nat) : Bool :=
  perfect_squares.any (λ s => n % s = 0)

def count_numbers_with_square_factor : Nat :=
  (List.range 100).filter is_divisible_by_square |>.length

theorem count_numbers_with_square_factor_eq_41 :
  count_numbers_with_square_factor = 41 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_square_factor_eq_41_l860_86080


namespace NUMINAMATH_CALUDE_complex_exp_form_l860_86050

theorem complex_exp_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → Complex.arg z = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_form_l860_86050


namespace NUMINAMATH_CALUDE_min_value_theorem_l860_86096

theorem min_value_theorem (x y a b : ℝ) :
  x - y - 1 ≤ 0 →
  2 * x - y - 3 ≥ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', x' - y' - 1 ≤ 0 → 2 * x' - y' - 3 ≥ 0 → a * x' + b * y' ≥ a * x + b * y) →
  a * x + b * y = 3 →
  (∀ a' b', a' > 0 → b' > 0 → 2 / a' + 1 / b' ≥ 2 / a + 1 / b) →
  2 / a + 1 / b = 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l860_86096


namespace NUMINAMATH_CALUDE_triangle_problem_l860_86075

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if √3 * sin(C) + c * cos(A) = a + b, c = 2, and the area is √3,
    then C = π/3 and the perimeter is 6. -/
theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.sqrt 3 * Real.sin C + c * Real.cos A = a + b →
  c = 2 →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 →
  C = π/3 ∧ a + b + c = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l860_86075


namespace NUMINAMATH_CALUDE_central_cell_value_l860_86051

theorem central_cell_value (a b c d e f g h i : ℝ) 
  (row_products : a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10)
  (col_products : a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10)
  (square_products : a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3) :
  e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l860_86051


namespace NUMINAMATH_CALUDE_student_pairs_l860_86043

theorem student_pairs (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_l860_86043


namespace NUMINAMATH_CALUDE_sum_even_coefficients_l860_86088

theorem sum_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 * (x + 2)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
                                    a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₂ + a₄ + a₆ + a₈ = -24 := by
sorry

end NUMINAMATH_CALUDE_sum_even_coefficients_l860_86088


namespace NUMINAMATH_CALUDE_retail_price_problem_l860_86060

/-- The retail price problem -/
theorem retail_price_problem
  (wholesale_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.2)
  (retail_price : ℝ) :
  retail_price = 120 ↔
    retail_price * (1 - discount_rate) = 
      wholesale_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_retail_price_problem_l860_86060


namespace NUMINAMATH_CALUDE_quadratic_function_max_min_l860_86055

theorem quadratic_function_max_min (a b : ℝ) (h1 : a ≠ 0) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 2 * a * x + b
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, f y ≤ f x) ∧
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, f y ≥ f x) ∧
  (∃ x ∈ Set.Icc 1 2, f x = 0) ∧
  (∃ x ∈ Set.Icc 1 2, f x = -1) →
  ((a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_min_l860_86055


namespace NUMINAMATH_CALUDE_blake_milkshake_problem_l860_86064

/-- Calculates the amount of ice cream needed per milkshake given the initial milk,
    ice cream amount, milk per milkshake, and remaining milk. -/
def ice_cream_per_milkshake (initial_milk : ℚ) (ice_cream : ℚ) 
                             (milk_per_milkshake : ℚ) (remaining_milk : ℚ) : ℚ :=
  ice_cream / ((initial_milk - remaining_milk) / milk_per_milkshake)

/-- Proves that given the specified conditions, the amount of ice cream
    needed per milkshake is 12 ounces. -/
theorem blake_milkshake_problem :
  ice_cream_per_milkshake 72 192 4 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_blake_milkshake_problem_l860_86064
