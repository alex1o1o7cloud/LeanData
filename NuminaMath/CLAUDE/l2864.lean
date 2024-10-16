import Mathlib

namespace NUMINAMATH_CALUDE_floor_slab_rate_l2864_286489

/-- Proves that for a rectangular room with given dimensions and total flooring cost,
    the rate per square meter is 900 Rs. -/
theorem floor_slab_rate (length width total_cost : ℝ) :
  length = 5 →
  width = 4.75 →
  total_cost = 21375 →
  total_cost / (length * width) = 900 := by
sorry

end NUMINAMATH_CALUDE_floor_slab_rate_l2864_286489


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2864_286447

theorem simplify_and_evaluate :
  let x : ℚ := -1/3
  let a : ℤ := -2
  let b : ℤ := -1
  (6*x^2 + 5*x^2 - 2*(3*x - 2*x^2) = 11/3) ∧
  (5*a^2 - a*b - 2*(3*a*b - (a*b - 2*a^2)) = -6) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2864_286447


namespace NUMINAMATH_CALUDE_chocolate_bars_bought_correct_number_of_bars_l2864_286457

def sugar_per_chocolate_bar : ℕ := 10
def sugar_in_lollipop : ℕ := 37
def calories_in_lollipop : ℕ := 190
def total_sugar : ℕ := 177

theorem chocolate_bars_bought : ℕ :=
  (total_sugar - sugar_in_lollipop) / sugar_per_chocolate_bar

theorem correct_number_of_bars : chocolate_bars_bought = 14 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_bought_correct_number_of_bars_l2864_286457


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2864_286475

theorem quadratic_inequality_equivalence (x : ℝ) :
  x^2 + 5*x - 14 < 0 ↔ -7 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2864_286475


namespace NUMINAMATH_CALUDE_rose_pollen_diameter_scientific_notation_l2864_286432

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The diameter of the rose pollen in meters -/
def rose_pollen_diameter : ℝ := 0.0000028

/-- The scientific notation representation of the rose pollen diameter -/
def rose_pollen_scientific : ScientificNotation :=
  { coefficient := 2.8
  , exponent := -6
  , is_valid := by sorry }

/-- Theorem stating that the rose pollen diameter is correctly expressed in scientific notation -/
theorem rose_pollen_diameter_scientific_notation :
  rose_pollen_diameter = rose_pollen_scientific.coefficient * (10 : ℝ) ^ rose_pollen_scientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_rose_pollen_diameter_scientific_notation_l2864_286432


namespace NUMINAMATH_CALUDE_sqrt_65_greater_than_8_l2864_286451

theorem sqrt_65_greater_than_8 : Real.sqrt 65 > 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_65_greater_than_8_l2864_286451


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l2864_286407

theorem circle_radius_is_three (r : ℝ) (h : 3 * (2 * π * r) = 2 * (π * r^2)) : r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l2864_286407


namespace NUMINAMATH_CALUDE_triangle_properties_l2864_286414

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  tan C = (sin A + sin B) / (cos A + cos B) →
  C = π / 3 ∧
  (∀ r : ℝ, r > 0 → 2 * r = 1 →
    3/4 < a^2 + b^2 ∧ a^2 + b^2 ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2864_286414


namespace NUMINAMATH_CALUDE_smallest_value_zero_l2864_286492

-- Define the function y
def y (x p q : ℝ) : ℝ := x^3 + x^2 + p*x + q

-- State the theorem
theorem smallest_value_zero (p : ℝ) :
  ∃ q : ℝ, (∀ x : ℝ, y x p q ≥ 0) ∧ (∃ x : ℝ, y x p q = 0) ∧ q = -2/27 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_zero_l2864_286492


namespace NUMINAMATH_CALUDE_inverse_proportion_value_l2864_286441

/-- For the inverse proportion function y = -8/x, when x = -2, y = 4 -/
theorem inverse_proportion_value : 
  let f : ℝ → ℝ := λ x => -8 / x
  f (-2) = 4 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_value_l2864_286441


namespace NUMINAMATH_CALUDE_product_of_special_reals_l2864_286467

/-- Given two real numbers a and b satisfying certain conditions, 
    their product is approximately 17.26 -/
theorem product_of_special_reals (a b : ℝ) 
  (sum_eq : a + b = 8)
  (fourth_power_sum : a^4 + b^4 = 272) :
  ∃ ε > 0, |a * b - 17.26| < ε :=
sorry

end NUMINAMATH_CALUDE_product_of_special_reals_l2864_286467


namespace NUMINAMATH_CALUDE_unique_zero_in_interval_l2864_286404

def f (x : ℝ) := 2*x + x^3 - 2

theorem unique_zero_in_interval : ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_in_interval_l2864_286404


namespace NUMINAMATH_CALUDE_f_derivative_l2864_286421

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.cos x)^5 * (Real.arctan x)^7 * (Real.log x)^4 * (Real.arcsin x)^10

theorem f_derivative (x : ℝ) (hx : x ≠ 0 ∧ x^2 < 1) : 
  deriv f x = f x * (3/x - 5*Real.tan x + 7/(Real.arctan x * (1 + x^2)) + 4/(x * Real.log x) + 10/(Real.arcsin x * Real.sqrt (1 - x^2))) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l2864_286421


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l2864_286491

/-- The length of a bridge given train specifications --/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the bridge length problem --/
theorem bridge_length_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |bridge_length 180 60 45 - 570.15| < ε :=
sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l2864_286491


namespace NUMINAMATH_CALUDE_probability_two_red_marbles_l2864_286419

/-- The probability of drawing two red marbles without replacement from a bag -/
theorem probability_two_red_marbles (red : ℕ) (white : ℕ) : 
  red = 5 → white = 7 → (red / (red + white)) * ((red - 1) / (red + white - 1)) = 5 / 33 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_marbles_l2864_286419


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l2864_286423

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def C1 : Circle := { center := (0, -1), radius := 5 }
def C2 : Circle := { center := (0, 2), radius := 1 }

def is_tangent_inside (M : ℝ × ℝ) (C : Circle) : Prop :=
  (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = (C.radius - 1)^2

def is_tangent_outside (M : ℝ × ℝ) (C : Circle) : Prop :=
  (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = (C.radius + 1)^2

theorem trajectory_of_moving_circle (x y : ℝ) :
  is_tangent_inside (x, y) C1 → is_tangent_outside (x, y) C2 →
  y ≠ 3 → y^2/9 + x^2/5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l2864_286423


namespace NUMINAMATH_CALUDE_meeting_arrangements_l2864_286438

def num_schools : ℕ := 3
def members_per_school : ℕ := 6
def host_representatives : ℕ := 3
def other_representatives : ℕ := 1

def arrange_meeting : ℕ := 
  num_schools * (members_per_school.choose host_representatives) * 
  ((members_per_school.choose other_representatives) ^ (num_schools - 1))

theorem meeting_arrangements :
  arrange_meeting = 2160 := by
  sorry

end NUMINAMATH_CALUDE_meeting_arrangements_l2864_286438


namespace NUMINAMATH_CALUDE_parallel_transitivity_l2864_286416

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a relation for parallel lines
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (l₁ l₂ l₃ : Line) :
  Parallel l₁ l₃ → Parallel l₂ l₃ → Parallel l₁ l₂ :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l2864_286416


namespace NUMINAMATH_CALUDE_expression_simplification_l2864_286437

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 5) (hz : z ≠ 7) :
  (x - 2) / (6 - z) * (y - 5) / (2 - x) * (z - 7) / (5 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2864_286437


namespace NUMINAMATH_CALUDE_cashback_less_profitable_l2864_286484

structure Bank where
  forecasted_cashback : ℝ
  actual_cashback : ℝ

structure Customer where
  is_savvy : Bool
  cashback_optimized : ℝ

def cashback_program (b : Bank) (customers : List Customer) : Prop :=
  b.actual_cashback > b.forecasted_cashback

def growing_savvy_customers (customers : List Customer) : Prop :=
  (customers.filter (λ c => c.is_savvy)).length > 
  (customers.filter (λ c => !c.is_savvy)).length

theorem cashback_less_profitable 
  (b : Bank) 
  (customers : List Customer) 
  (h1 : growing_savvy_customers customers) 
  (h2 : ∀ c ∈ customers, c.is_savvy → c.cashback_optimized > 0) :
  cashback_program b customers :=
by
  sorry

#check cashback_less_profitable

end NUMINAMATH_CALUDE_cashback_less_profitable_l2864_286484


namespace NUMINAMATH_CALUDE_game_correct_answers_l2864_286433

theorem game_correct_answers : 
  ∀ (total_questions : ℕ) 
    (correct_reward incorrect_penalty : ℕ) 
    (correct_answers : ℕ),
  total_questions = 50 →
  correct_reward = 7 →
  incorrect_penalty = 3 →
  correct_answers * correct_reward = 
    (total_questions - correct_answers) * incorrect_penalty →
  correct_answers = 15 := by
sorry

end NUMINAMATH_CALUDE_game_correct_answers_l2864_286433


namespace NUMINAMATH_CALUDE_fencing_requirement_l2864_286446

/-- Given a rectangular field with area 60 sq. feet and one side 20 feet,
    prove that the sum of the other three sides is 26 feet. -/
theorem fencing_requirement (length width : ℝ) : 
  length * width = 60 →
  length = 20 →
  length + 2 * width = 26 := by
  sorry

end NUMINAMATH_CALUDE_fencing_requirement_l2864_286446


namespace NUMINAMATH_CALUDE_travel_time_difference_proof_l2864_286494

/-- The length of Telegraph Road in kilometers -/
def telegraph_road_length : ℝ := 162

/-- The lengths of the four detours on Telegraph Road in kilometers -/
def telegraph_detours : List ℝ := [5.2, 2.7, 3.8, 4.4]

/-- The length of Pardee Road in meters -/
def pardee_road_length : ℝ := 12000

/-- The increase in length of Pardee Road due to road work in kilometers -/
def pardee_road_increase : ℝ := 2.5

/-- The constant speed of travel in kilometers per hour -/
def travel_speed : ℝ := 80

/-- The difference in travel time between Telegraph Road and Pardee Road in minutes -/
def travel_time_difference : ℝ := 122.7

theorem travel_time_difference_proof :
  let telegraph_total := telegraph_road_length + (telegraph_detours.sum)
  let pardee_total := (pardee_road_length / 1000) + pardee_road_increase
  let telegraph_time := (telegraph_total / travel_speed) * 60
  let pardee_time := (pardee_total / travel_speed) * 60
  telegraph_time - pardee_time = travel_time_difference := by
  sorry

end NUMINAMATH_CALUDE_travel_time_difference_proof_l2864_286494


namespace NUMINAMATH_CALUDE_bread_slices_left_l2864_286464

theorem bread_slices_left (
  initial_slices : Nat) 
  (days_in_week : Nat)
  (slices_per_sandwich : Nat)
  (extra_sandwiches : Nat) :
  initial_slices = 22 →
  days_in_week = 7 →
  slices_per_sandwich = 2 →
  extra_sandwiches = 1 →
  initial_slices - (days_in_week + extra_sandwiches) * slices_per_sandwich = 6 :=
by sorry

end NUMINAMATH_CALUDE_bread_slices_left_l2864_286464


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l2864_286459

theorem consecutive_pages_sum (x : ℕ) : x > 0 ∧ x + (x + 1) = 137 → x + 1 = 69 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l2864_286459


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l2864_286463

/-- Proves that three successive discounts are equivalent to a single discount -/
theorem successive_discounts_equivalence (original_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) (equivalent_discount : ℝ) : 
  original_price = 60 ∧ 
  discount1 = 0.15 ∧ 
  discount2 = 0.10 ∧ 
  discount3 = 0.20 ∧ 
  equivalent_discount = 0.388 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 
  original_price * (1 - equivalent_discount) :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l2864_286463


namespace NUMINAMATH_CALUDE_stating_solve_age_problem_l2864_286450

/-- Represents the age-related problem described in the question. -/
def AgeProblem (current_age : ℕ) (years_ago : ℕ) : Prop :=
  3 * (current_age + 3) - 3 * (current_age - years_ago) = current_age

/-- 
Theorem stating that given the person's current age of 18, 
the number of years ago referred to in their statement is 3.
-/
theorem solve_age_problem : 
  ∃ (years_ago : ℕ), AgeProblem 18 years_ago ∧ years_ago = 3 :=
sorry

end NUMINAMATH_CALUDE_stating_solve_age_problem_l2864_286450


namespace NUMINAMATH_CALUDE_inequality_proof_l2864_286488

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (1 + 2*a)) + (1 / (1 + 2*b)) + (1 / (1 + 2*c)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2864_286488


namespace NUMINAMATH_CALUDE_IMO_2001_max_sum_l2864_286411

theorem IMO_2001_max_sum : 
  ∀ I M O : ℕ+,
  I ≠ M → I ≠ O → M ≠ O →
  I * M * O = 2001 →
  I + M + O ≤ 671 :=
by
  sorry

end NUMINAMATH_CALUDE_IMO_2001_max_sum_l2864_286411


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2864_286490

/-- Two lines in the form ax + by + c = 0 and dx + ey + f = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Perpendicular property for two lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_a_value :
  ∀ (a : ℝ),
  let l1 : Line := ⟨a, 2, 1⟩
  let l2 : Line := ⟨1, 3, -2⟩
  perpendicular l1 l2 → a = -6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2864_286490


namespace NUMINAMATH_CALUDE_sum_lent_calculation_l2864_286462

-- Define the interest rate and time period
def interest_rate : ℚ := 3 / 100
def time_period : ℕ := 3

-- Define the theorem
theorem sum_lent_calculation (P : ℚ) : 
  P * interest_rate * time_period = P - 1820 → P = 2000 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_calculation_l2864_286462


namespace NUMINAMATH_CALUDE_lines_skew_iff_b_neq_l2864_286420

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∀ t u : ℝ, l1.point + t • l1.direction ≠ l2.point + u • l2.direction

/-- The main theorem -/
theorem lines_skew_iff_b_neq (b : ℝ) : 
  are_skew 
    (Line3D.mk (2, 1, b) (3, 4, 5)) 
    (Line3D.mk (3, 5, 2) (7, 3, 1)) 
  ↔ b ≠ -79/19 := by sorry

end NUMINAMATH_CALUDE_lines_skew_iff_b_neq_l2864_286420


namespace NUMINAMATH_CALUDE_a_less_than_sqrt_a_iff_l2864_286439

theorem a_less_than_sqrt_a_iff (a : ℝ) : 0 < a ∧ a < 1 ↔ a < Real.sqrt a := by sorry

end NUMINAMATH_CALUDE_a_less_than_sqrt_a_iff_l2864_286439


namespace NUMINAMATH_CALUDE_total_erasers_l2864_286417

/-- Given an initial number of erasers and a number of erasers added, 
    the total number of erasers is equal to the sum of the initial number and the added number. -/
theorem total_erasers (initial_erasers added_erasers : ℕ) :
  initial_erasers + added_erasers = initial_erasers + added_erasers :=
by sorry

end NUMINAMATH_CALUDE_total_erasers_l2864_286417


namespace NUMINAMATH_CALUDE_smallest_n_with_properties_l2864_286429

def has_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = a * 10^c + d₁ * 10^b + d₂

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

theorem smallest_n_with_properties : 
  ∀ n : ℕ, 
    n > 0 ∧ 
    is_terminating_decimal n ∧ 
    has_digits n 9 5 → 
    n ≥ 9000000 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_properties_l2864_286429


namespace NUMINAMATH_CALUDE_original_speed_is_30_l2864_286465

/-- Represents the driving scenario with given conditions -/
def DrivingScenario (original_speed : ℝ) : Prop :=
  let total_distance : ℝ := 100
  let breakdown_time : ℝ := 2
  let repair_time : ℝ := 0.5
  let speed_increase_factor : ℝ := 1.6
  
  -- Time equation: total time = time before breakdown + repair time + time after repair
  total_distance / original_speed = 
    breakdown_time + repair_time + 
    (total_distance - breakdown_time * original_speed) / (speed_increase_factor * original_speed)

/-- Theorem stating that the original speed satisfying the driving scenario is 30 km/h -/
theorem original_speed_is_30 : 
  ∃ (speed : ℝ), DrivingScenario speed ∧ speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_original_speed_is_30_l2864_286465


namespace NUMINAMATH_CALUDE_unique_divisible_by_nine_l2864_286481

theorem unique_divisible_by_nine : ∃! x : ℕ, 
  x ≥ 0 ∧ x ≤ 9 ∧ (13800 + x * 10 + 6) % 9 = 0 := by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_nine_l2864_286481


namespace NUMINAMATH_CALUDE_employee_salary_proof_l2864_286469

def total_salary : ℝ := 572
def m_salary_ratio : ℝ := 1.2

theorem employee_salary_proof (n_salary : ℝ) 
  (h1 : n_salary + m_salary_ratio * n_salary = total_salary) :
  n_salary = 260 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_proof_l2864_286469


namespace NUMINAMATH_CALUDE_books_and_students_count_l2864_286477

/-- The number of books distributed to students -/
def total_books : ℕ := 26

/-- The number of students receiving books -/
def total_students : ℕ := 6

/-- Condition 1: If each person receives 3 books, there will be 8 books left -/
axiom condition1 : total_books = 3 * total_students + 8

/-- Condition 2: If each of the previous students receives 5 books, 
    then the last person will not receive 2 books -/
axiom condition2 : 
  total_books - 5 * (total_students - 1) < 2 ∧ 
  total_books - 5 * (total_students - 1) ≥ 0

/-- Theorem: Given the conditions, prove that the number of books is 26 
    and the number of students is 6 -/
theorem books_and_students_count : 
  total_books = 26 ∧ total_students = 6 := by
  sorry

end NUMINAMATH_CALUDE_books_and_students_count_l2864_286477


namespace NUMINAMATH_CALUDE_inverse_f_90_l2864_286444

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_f_90 : f⁻¹ 90 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_90_l2864_286444


namespace NUMINAMATH_CALUDE_alicia_remaining_art_l2864_286442

/-- Represents the types of art in Alicia's collection -/
inductive ArtType
  | Medieval
  | Renaissance
  | Modern

/-- Calculates the remaining art pieces after donation -/
def remaining_art (initial : Nat) (donate_percent : Nat) : Nat :=
  initial - (initial * donate_percent / 100)

/-- Theorem stating the remaining art pieces after Alicia's donations -/
theorem alicia_remaining_art :
  (remaining_art 70 65 = 25) ∧
  (remaining_art 120 30 = 84) ∧
  (remaining_art 150 45 = 83) := by
  sorry

#check alicia_remaining_art

end NUMINAMATH_CALUDE_alicia_remaining_art_l2864_286442


namespace NUMINAMATH_CALUDE_field_trip_girls_fraction_l2864_286435

theorem field_trip_girls_fraction (total_boys : ℕ) (total_girls : ℕ) 
  (boys_fraction : ℚ) (girls_fraction : ℚ) :
  total_boys = 200 →
  total_girls = 150 →
  boys_fraction = 3 / 5 →
  girls_fraction = 4 / 5 →
  let boys_on_trip := (boys_fraction * total_boys : ℚ)
  let girls_on_trip := (girls_fraction * total_girls : ℚ)
  let total_on_trip := boys_on_trip + girls_on_trip
  (girls_on_trip / total_on_trip : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_field_trip_girls_fraction_l2864_286435


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2864_286409

theorem floor_ceil_sum : ⌊(-3.75 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ + (1/2 : ℝ) = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2864_286409


namespace NUMINAMATH_CALUDE_classroom_gpa_proof_l2864_286470

/-- Proves that the grade point average of one third of a classroom is 30,
    given the grade point average of two thirds is 33 and the overall average is 32. -/
theorem classroom_gpa_proof (gpa_two_thirds : ℝ) (gpa_overall : ℝ) : ℝ :=
  let gpa_one_third : ℝ := 30
  by
    have h1 : gpa_two_thirds = 33 := by sorry
    have h2 : gpa_overall = 32 := by sorry
    have h3 : (1/3 : ℝ) * gpa_one_third + (2/3 : ℝ) * gpa_two_thirds = gpa_overall := by sorry
    sorry

end NUMINAMATH_CALUDE_classroom_gpa_proof_l2864_286470


namespace NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l2864_286482

/-- Given a business investment scenario where:
  * The total loss is 1000
  * Pyarelal's loss is 900
  * Ashok's loss is the remaining amount
  * The ratio of losses is proportional to the ratio of investments

  This theorem proves that the ratio of Ashok's capital to Pyarelal's capital is 1:9.
-/
theorem ashok_pyarelal_capital_ratio 
  (total_loss : ℕ) 
  (pyarelal_loss : ℕ) 
  (h_total_loss : total_loss = 1000)
  (h_pyarelal_loss : pyarelal_loss = 900)
  (ashok_loss : ℕ := total_loss - pyarelal_loss)
  (ashok_capital pyarelal_capital : ℚ)
  (h_loss_ratio : ashok_loss / pyarelal_loss = ashok_capital / pyarelal_capital) :
  ashok_capital / pyarelal_capital = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l2864_286482


namespace NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l2864_286406

theorem max_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  2*x + y ≤ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 2/y₀ = 1 ∧ 2*x₀ + y₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l2864_286406


namespace NUMINAMATH_CALUDE_product_of_roots_l2864_286428

theorem product_of_roots : Real.sqrt 16 * (27 ^ (1/3 : ℝ)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2864_286428


namespace NUMINAMATH_CALUDE_least_clock_equivalent_after_six_twelve_is_clock_equivalent_twelve_is_least_clock_equivalent_after_six_l2864_286476

def clock_equivalent (h : ℕ) : Prop :=
  (h ^ 2 - h) % 24 = 0

theorem least_clock_equivalent_after_six :
  ∀ h : ℕ, h > 6 → clock_equivalent h → h ≥ 12 :=
by sorry

theorem twelve_is_clock_equivalent : clock_equivalent 12 :=
by sorry

theorem twelve_is_least_clock_equivalent_after_six :
  ∀ h : ℕ, h > 6 → clock_equivalent h → h = 12 ∨ h > 12 :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_after_six_twelve_is_clock_equivalent_twelve_is_least_clock_equivalent_after_six_l2864_286476


namespace NUMINAMATH_CALUDE_palindrome_divisible_by_seven_probability_l2864_286452

/-- A function that checks if a number is a palindrome -/
def is_palindrome (n : ℕ) : Prop := sorry

/-- A function that generates all 5-digit palindromes -/
def five_digit_palindromes : Finset ℕ := sorry

/-- A function that counts the number of elements in a finite set satisfying a predicate -/
def count_satisfying {α : Type*} (s : Finset α) (p : α → Prop) : ℕ := sorry

/-- The main theorem -/
theorem palindrome_divisible_by_seven_probability :
  ∃ k : ℕ, (k : ℚ) / 900 = (count_satisfying five_digit_palindromes 
    (λ n => (n % 7 = 0) ∧ (is_palindrome (n / 7)))) / (five_digit_palindromes.card) :=
sorry

end NUMINAMATH_CALUDE_palindrome_divisible_by_seven_probability_l2864_286452


namespace NUMINAMATH_CALUDE_octal_year_to_decimal_l2864_286415

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the year -/
def octal_year : List Nat := [7, 4, 2]

/-- Theorem stating that the octal year 742 is equal to 482 in decimal -/
theorem octal_year_to_decimal :
  octal_to_decimal octal_year = 482 := by sorry

end NUMINAMATH_CALUDE_octal_year_to_decimal_l2864_286415


namespace NUMINAMATH_CALUDE_max_integers_greater_than_20_l2864_286413

theorem max_integers_greater_than_20 (integers : List ℤ) : 
  integers.length = 8 → 
  integers.sum = -20 → 
  (integers.filter (λ x => x > 20)).length ≤ 7 ∧ 
  ∃ (valid_list : List ℤ), 
    valid_list.length = 8 ∧ 
    valid_list.sum = -20 ∧ 
    (valid_list.filter (λ x => x > 20)).length = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_integers_greater_than_20_l2864_286413


namespace NUMINAMATH_CALUDE_equation_solutions_l2864_286472

-- Define the function representing the left side of the equation
def f (x : ℝ) : ℝ := (18 * x - 1) ^ (1/3) - (10 * x + 1) ^ (1/3) - 3 * x ^ (1/3)

-- Define the set of solutions
def solutions : Set ℝ := {0, -5/8317, -60/1614}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℝ, f x = 0 ↔ x ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2864_286472


namespace NUMINAMATH_CALUDE_inequality_problem_l2864_286400

theorem inequality_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = 1) :
  (ab ≥ 16) ∧ 
  (2*a + b ≥ 6 + 4*Real.sqrt 2) ∧ 
  (1/a^2 + 16/b^2 ≥ 1/2) ∧
  ¬(∀ (a b : ℝ), a > 0 → b > 0 → 1/a + 4/b = 1 → a - b < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l2864_286400


namespace NUMINAMATH_CALUDE_bank_account_difference_l2864_286499

theorem bank_account_difference (bob_amount jenna_amount phil_amount : ℝ) : 
  bob_amount = 60 →
  phil_amount = (1/3) * bob_amount →
  jenna_amount = 2 * phil_amount →
  bob_amount - jenna_amount = 20 := by
sorry

end NUMINAMATH_CALUDE_bank_account_difference_l2864_286499


namespace NUMINAMATH_CALUDE_abc_relationship_l2864_286445

theorem abc_relationship :
  let a : Real := Real.rpow 0.3 0.2
  let b : Real := Real.rpow 0.2 0.3
  let c : Real := Real.rpow 0.3 0.3
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_abc_relationship_l2864_286445


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_range_l2864_286455

def integerRange : List Int := List.range 10 |> List.map (λ x => x - 4)

theorem arithmetic_mean_of_range : 
  (integerRange.sum : ℚ) / integerRange.length = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_range_l2864_286455


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2864_286448

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + 3 * x^5 + 2 * x^4 + x + 15) - (x^6 + 4 * x^5 + 5 * x^4 - 2 * x^3 + 20) = 
  x^6 - x^5 - 3 * x^4 + 2 * x^3 + x - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2864_286448


namespace NUMINAMATH_CALUDE_cube_difference_l2864_286480

theorem cube_difference (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 35) : a^3 - b^3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l2864_286480


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2864_286418

theorem smallest_positive_integer_with_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ (y : ℕ), y > 0 → 
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → 
    x ≤ y ∧
  x = 167 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2864_286418


namespace NUMINAMATH_CALUDE_digit_sum_divisibility_pairs_l2864_286408

/-- Sum of decimal digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem digit_sum_divisibility_pairs :
  ∀ a b : ℕ, 
    a > b → 
    a > 0 → 
    b > 0 → 
    a ∣ (b + sum_of_digits a) → 
    b ∣ (a + sum_of_digits b) → 
    ((a = 18 ∧ b = 9) ∨ (a = 27 ∧ b = 18)) :=
sorry

end NUMINAMATH_CALUDE_digit_sum_divisibility_pairs_l2864_286408


namespace NUMINAMATH_CALUDE_x_needs_seven_days_l2864_286473

/-- The number of days x needs to finish the remaining work -/
def remaining_days_for_x (x_days y_days y_worked_days : ℕ) : ℚ :=
  (y_days - y_worked_days) * x_days / y_days

/-- Theorem stating that x needs 7 days to finish the remaining work -/
theorem x_needs_seven_days (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 21)
  (hy : y_days = 15)
  (hw : y_worked_days = 10) :
  remaining_days_for_x x_days y_days y_worked_days = 7 := by
  sorry

#eval remaining_days_for_x 21 15 10

end NUMINAMATH_CALUDE_x_needs_seven_days_l2864_286473


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l2864_286498

/-- The number of balls in the bin -/
def n : ℕ := 15

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from n balls -/
def possible_lists (n k : ℕ) : ℕ := n ^ k

theorem ball_drawing_theorem : possible_lists n k = 50625 := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l2864_286498


namespace NUMINAMATH_CALUDE_intersection_constraint_l2864_286495

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + 2*p.2^2 = 3}

def N (m b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m*p.1 + b}

theorem intersection_constraint (b : ℝ) :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) → b ∈ Set.Icc (-Real.sqrt (3/2)) (Real.sqrt (3/2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_constraint_l2864_286495


namespace NUMINAMATH_CALUDE_inversion_similarity_l2864_286440

/-- Inversion of a point with respect to a circle -/
def inversion (O : ℝ × ℝ) (R : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Similarity of triangles -/
def triangles_similar (A B C D E F : ℝ × ℝ) : Prop := sorry

theorem inversion_similarity 
  (O A B : ℝ × ℝ) 
  (R : ℝ) 
  (A' B' : ℝ × ℝ) 
  (h1 : A' = inversion O R A) 
  (h2 : B' = inversion O R B) : 
  triangles_similar O A B B' O A' := 
sorry

end NUMINAMATH_CALUDE_inversion_similarity_l2864_286440


namespace NUMINAMATH_CALUDE_rectangle_width_l2864_286474

/-- Given a rectangle with length 4 times its width and area 196 square inches, 
    prove that its width is 7 inches. -/
theorem rectangle_width (w : ℝ) (h1 : w > 0) (h2 : w * (4 * w) = 196) : w = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l2864_286474


namespace NUMINAMATH_CALUDE_no_function_satisfies_inequality_l2864_286425

theorem no_function_satisfies_inequality :
  ∀ f : ℕ → ℕ, ∃ m n : ℕ, (m + f n)^2 < 3 * (f m)^2 + n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_inequality_l2864_286425


namespace NUMINAMATH_CALUDE_history_book_cost_l2864_286496

theorem history_book_cost 
  (total_books : ℕ) 
  (math_books : ℕ) 
  (math_book_cost : ℚ) 
  (total_price : ℚ) 
  (h1 : total_books = 90)
  (h2 : math_books = 60)
  (h3 : math_book_cost = 4)
  (h4 : total_price = 390) :
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 := by
  sorry

end NUMINAMATH_CALUDE_history_book_cost_l2864_286496


namespace NUMINAMATH_CALUDE_cookies_per_bag_l2864_286479

/-- Given 33 cookies distributed equally among 3 bags, prove that each bag contains 11 cookies. -/
theorem cookies_per_bag :
  ∀ (total_cookies : ℕ) (num_bags : ℕ) (cookies_per_bag : ℕ),
    total_cookies = 33 →
    num_bags = 3 →
    total_cookies = num_bags * cookies_per_bag →
    cookies_per_bag = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l2864_286479


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2864_286486

/-- The longest segment in a cylinder -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2864_286486


namespace NUMINAMATH_CALUDE_yvonne_swims_10_laps_l2864_286412

/-- The number of laps Yvonne can swim -/
def yvonne_laps : ℕ := sorry

/-- The number of laps Yvonne's younger sister can swim -/
def sister_laps : ℕ := sorry

/-- The number of laps Joel can swim -/
def joel_laps : ℕ := 15

theorem yvonne_swims_10_laps :
  (sister_laps = yvonne_laps / 2) →
  (joel_laps = 3 * sister_laps) →
  (yvonne_laps = 10) :=
by sorry

end NUMINAMATH_CALUDE_yvonne_swims_10_laps_l2864_286412


namespace NUMINAMATH_CALUDE_mathlon_solution_l2864_286449

/-- A Mathlon competition with M events and three participants -/
structure Mathlon where
  M : ℕ
  p₁ : ℕ
  p₂ : ℕ
  p₃ : ℕ
  scoreA : ℕ
  scoreB : ℕ
  scoreC : ℕ
  B_won_100m : Bool

/-- The conditions of the Mathlon problem -/
def mathlon_conditions (m : Mathlon) : Prop :=
  m.M > 0 ∧
  m.p₁ > m.p₂ ∧ m.p₂ > m.p₃ ∧ m.p₃ > 0 ∧
  m.scoreA = 22 ∧ m.scoreB = 9 ∧ m.scoreC = 9 ∧
  m.B_won_100m = true

/-- The theorem to prove -/
theorem mathlon_solution (m : Mathlon) (h : mathlon_conditions m) : 
  m.M = 5 ∧ ∃ (events : Fin m.M → Fin 3), 
    (∃ i, events i = 1) ∧  -- B wins one event (100m)
    (∃ i, events i = 2)    -- C is second in one event (high jump)
    := by sorry

end NUMINAMATH_CALUDE_mathlon_solution_l2864_286449


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l2864_286410

/-- Converts a base 5 number (represented as a list of digits) to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The loot values in base 5 --/
def silverware : List Nat := [3, 1, 2, 4]
def diamondTiaras : List Nat := [1, 0, 1, 3]
def silkScarves : List Nat := [2, 0, 2]

/-- The theorem to prove --/
theorem pirate_loot_sum :
  base5ToBase10 silverware + base5ToBase10 diamondTiaras + base5ToBase10 silkScarves = 1011 := by
  sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l2864_286410


namespace NUMINAMATH_CALUDE_melanie_dimes_problem_l2864_286497

/-- The number of dimes Melanie gave her mother -/
def dimes_given_to_mother (initial dimes_from_dad final : ℕ) : ℕ :=
  initial + dimes_from_dad - final

theorem melanie_dimes_problem (initial dimes_from_dad final : ℕ) 
  (h1 : initial = 7)
  (h2 : dimes_from_dad = 8)
  (h3 : final = 11) :
  dimes_given_to_mother initial dimes_from_dad final = 4 := by
sorry

end NUMINAMATH_CALUDE_melanie_dimes_problem_l2864_286497


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2864_286483

/-- A line y = 3x + d is tangent to the parabola y^2 = 12x if and only if d = 1 -/
theorem line_tangent_to_parabola (d : ℝ) : 
  (∃! x : ℝ, (3 * x + d)^2 = 12 * x) ↔ d = 1 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2864_286483


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_ratios_ge_two_l2864_286454

theorem sum_of_reciprocal_ratios_ge_two (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  b / a + a / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_ratios_ge_two_l2864_286454


namespace NUMINAMATH_CALUDE_remaining_bottles_l2864_286405

/-- Calculates the number of remaining bottles of juice after some are broken -/
theorem remaining_bottles (total_crates : ℕ) (bottles_per_crate : ℕ) (broken_crates : ℕ) :
  total_crates = 7 →
  bottles_per_crate = 6 →
  broken_crates = 3 →
  total_crates * bottles_per_crate - broken_crates * bottles_per_crate = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_bottles_l2864_286405


namespace NUMINAMATH_CALUDE_document_typing_time_l2864_286402

theorem document_typing_time (barbara_speed jim_speed : ℕ) (document_length : ℕ) (jim_time : ℕ) :
  barbara_speed = 172 →
  jim_speed = 100 →
  document_length = 3440 →
  jim_time = 20 →
  ∃ t : ℕ, t < jim_time ∧ t * (barbara_speed + jim_speed) ≥ document_length :=
by sorry

end NUMINAMATH_CALUDE_document_typing_time_l2864_286402


namespace NUMINAMATH_CALUDE_inequalities_proof_l2864_286487

theorem inequalities_proof (a b r s : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : r > 0) (hs : s > 0) 
  (hrs : 1/r + 1/s = 1) : 
  (a^2 * b ≤ 4 * ((a + b) / 3)^3) ∧ 
  ((a^r / r) + (b^s / s) ≥ a * b) := by
sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2864_286487


namespace NUMINAMATH_CALUDE_fraction_ordering_l2864_286443

theorem fraction_ordering : 19/15 < 17/13 ∧ 17/13 < 15/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2864_286443


namespace NUMINAMATH_CALUDE_second_car_speed_l2864_286430

/-- Theorem: Given two cars starting from opposite ends of a 500-mile highway
    at the same time, with one car traveling at 40 mph and both cars meeting
    after 5 hours, the speed of the second car is 60 mph. -/
theorem second_car_speed
  (highway_length : ℝ)
  (first_car_speed : ℝ)
  (meeting_time : ℝ)
  (second_car_speed : ℝ) :
  highway_length = 500 →
  first_car_speed = 40 →
  meeting_time = 5 →
  highway_length = first_car_speed * meeting_time + second_car_speed * meeting_time →
  second_car_speed = 60 :=
by
  sorry

#check second_car_speed

end NUMINAMATH_CALUDE_second_car_speed_l2864_286430


namespace NUMINAMATH_CALUDE_tangent_point_abscissa_l2864_286434

/-- The curve function f(x) = x³ - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 1

/-- The slope of the line perpendicular to x + 2y - 1 = 0 -/
def m : ℝ := 2

theorem tangent_point_abscissa :
  ∃ (x : ℝ), (f' x = m) ∧ (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_abscissa_l2864_286434


namespace NUMINAMATH_CALUDE_seans_sandwiches_l2864_286468

/-- Calculates the number of sandwiches Sean bought given the costs of items and total cost -/
theorem seans_sandwiches
  (soda_cost : ℕ)
  (soup_cost : ℕ)
  (sandwich_cost : ℕ)
  (total_cost : ℕ)
  (h1 : soda_cost = 3)
  (h2 : soup_cost = 6)
  (h3 : sandwich_cost = 9)
  (h4 : total_cost = 18)
  : (total_cost - soda_cost - soup_cost) / sandwich_cost = 1 := by
  sorry

#check seans_sandwiches

end NUMINAMATH_CALUDE_seans_sandwiches_l2864_286468


namespace NUMINAMATH_CALUDE_ornamental_rings_ratio_l2864_286426

theorem ornamental_rings_ratio (initial_purchase : ℕ) (mother_purchase : ℕ) (sold_after : ℕ) (remaining : ℕ) :
  initial_purchase = 200 →
  mother_purchase = 300 →
  sold_after = 150 →
  remaining = 225 →
  ∃ (original_stock : ℕ),
    initial_purchase + original_stock > 0 ∧
    (1 / 4 : ℚ) * (initial_purchase + original_stock : ℚ) + (mother_purchase : ℚ) - (sold_after : ℚ) = (remaining : ℚ) ∧
    (initial_purchase : ℚ) / (original_stock : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ornamental_rings_ratio_l2864_286426


namespace NUMINAMATH_CALUDE_ducks_in_other_flock_other_flock_size_l2864_286493

/-- Calculates the number of ducks in the other flock given the conditions of the problem -/
theorem ducks_in_other_flock (original_flock : ℕ) (net_increase_per_year : ℕ) (years : ℕ) (combined_flock : ℕ) : ℕ :=
  let final_original_flock := original_flock + net_increase_per_year * years
  combined_flock - final_original_flock

/-- Proves that the number of ducks in the other flock is 150 given the problem conditions -/
theorem other_flock_size :
  ducks_in_other_flock 100 10 5 300 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_other_flock_other_flock_size_l2864_286493


namespace NUMINAMATH_CALUDE_equation_solution_l2864_286461

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2864_286461


namespace NUMINAMATH_CALUDE_count_with_3_or_7_l2864_286401

/-- The set of digits that are neither 3 nor 7 -/
def other_digits : Finset Nat := {0, 1, 2, 4, 5, 6, 8, 9}

/-- The set of non-zero digits that are neither 3 nor 7 -/
def non_zero_other_digits : Finset Nat := {1, 2, 4, 5, 6, 8, 9}

/-- The count of four-digit numbers without 3 or 7 -/
def count_without_3_or_7 : Nat :=
  (Finset.card non_zero_other_digits) * (Finset.card other_digits)^3

/-- The total count of four-digit numbers -/
def total_four_digit_numbers : Nat := 9000

theorem count_with_3_or_7 :
  total_four_digit_numbers - count_without_3_or_7 = 5416 := by
  sorry

end NUMINAMATH_CALUDE_count_with_3_or_7_l2864_286401


namespace NUMINAMATH_CALUDE_daniel_purchase_cost_l2864_286427

/-- The total amount spent on a magazine and pencil after applying a coupon discount -/
def total_spent (magazine_cost pencil_cost coupon_discount : ℚ) : ℚ :=
  magazine_cost + pencil_cost - coupon_discount

/-- Theorem stating that given specific costs and discount, the total spent is $1.00 -/
theorem daniel_purchase_cost :
  total_spent 0.85 0.50 0.35 = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_daniel_purchase_cost_l2864_286427


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2864_286436

theorem smallest_candy_count : 
  ∃ (n : ℕ), 
    100 ≤ n ∧ n < 1000 ∧ 
    (n + 7) % 9 = 0 ∧ 
    (n - 9) % 7 = 0 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0 → false :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2864_286436


namespace NUMINAMATH_CALUDE_solve_equation_l2864_286424

theorem solve_equation : ∃ x : ℝ, 2 * x = (26 - x) + 19 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2864_286424


namespace NUMINAMATH_CALUDE_no_solution_iff_a_in_open_interval_l2864_286460

theorem no_solution_iff_a_in_open_interval :
  ∀ a : ℝ, (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) ↔ -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_in_open_interval_l2864_286460


namespace NUMINAMATH_CALUDE_directrix_of_specific_parabola_l2864_286422

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Defines the directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop := sorry

/-- The specific parabola with equation x^2 = 8y -/
def specific_parabola : Parabola :=
  { equation := fun x y => x^2 = 8*y }

/-- Theorem stating that the directrix of the specific parabola is y = -2 -/
theorem directrix_of_specific_parabola :
  directrix specific_parabola = fun y => y = -2 := by sorry

end NUMINAMATH_CALUDE_directrix_of_specific_parabola_l2864_286422


namespace NUMINAMATH_CALUDE_mean_proportional_sqrt45_and_7_3_pi_l2864_286453

theorem mean_proportional_sqrt45_and_7_3_pi :
  let a := Real.sqrt 45
  let b := 7/3 * Real.pi
  Real.sqrt (a * b) = Real.sqrt (7 * Real.sqrt 5 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_sqrt45_and_7_3_pi_l2864_286453


namespace NUMINAMATH_CALUDE_boat_travel_time_l2864_286471

/-- Given a boat that travels 2 miles in 5 minutes, prove that it takes 90 minutes to travel 36 miles at the same speed. -/
theorem boat_travel_time (distance : ℝ) (time : ℝ) (total_distance : ℝ) 
  (h1 : distance = 2) 
  (h2 : time = 5) 
  (h3 : total_distance = 36) : 
  (total_distance / (distance / time)) = 90 := by
  sorry


end NUMINAMATH_CALUDE_boat_travel_time_l2864_286471


namespace NUMINAMATH_CALUDE_parabola_sum_l2864_286478

/-- A parabola with equation y = px^2 + qx + r, vertex (3, 7), vertical axis of symmetry, and containing the point (0, 4) -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 7
  point_x : ℝ := 0
  point_y : ℝ := 4
  eq_at_point : p * point_x^2 + q * point_x + r = point_y
  vertex_form : ∀ x y, y = p * (x - vertex_x)^2 + vertex_y

theorem parabola_sum (par : Parabola) : par.p + par.q + par.r = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l2864_286478


namespace NUMINAMATH_CALUDE_not_perfect_squares_l2864_286458

theorem not_perfect_squares : 
  ¬(∃ n : ℕ, n^2 = 12345678) ∧ 
  ¬(∃ n : ℕ, n^2 = 987654) ∧ 
  ¬(∃ n : ℕ, n^2 = 1234560) ∧ 
  ¬(∃ n : ℕ, n^2 = 98765445) := by
sorry

end NUMINAMATH_CALUDE_not_perfect_squares_l2864_286458


namespace NUMINAMATH_CALUDE_committee_selection_l2864_286456

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committee_selection (total : ℕ) (committee_size : ℕ) (bill_karl : ℕ) (alice_jane : ℕ) :
  total = 9 ∧ committee_size = 5 ∧ bill_karl = 2 ∧ alice_jane = 2 →
  (choose (total - bill_karl) (committee_size - bill_karl) - 
   choose (total - bill_karl - alice_jane) 1) +
  (choose (total - bill_karl) committee_size - 
   choose (total - bill_karl - alice_jane) 3) = 41 :=
by sorry

end NUMINAMATH_CALUDE_committee_selection_l2864_286456


namespace NUMINAMATH_CALUDE_liquid_mixture_problem_l2864_286403

/-- Proves that the initial amount of liquid A is 21 litres given the conditions of the problem -/
theorem liquid_mixture_problem (initial_ratio_A : ℚ) (initial_ratio_B : ℚ) 
  (drawn_off : ℚ) (new_ratio_A : ℚ) (new_ratio_B : ℚ) :
  initial_ratio_A = 7 ∧ 
  initial_ratio_B = 5 ∧ 
  drawn_off = 9 ∧ 
  new_ratio_A = 7 ∧ 
  new_ratio_B = 9 → 
  ∃ (x : ℚ), 
    7 * x = 21 ∧ 
    (7 * x - (7 / 12) * drawn_off) / (5 * x - (5 / 12) * drawn_off + drawn_off) = new_ratio_A / new_ratio_B :=
by sorry

end NUMINAMATH_CALUDE_liquid_mixture_problem_l2864_286403


namespace NUMINAMATH_CALUDE_expression_value_l2864_286431

theorem expression_value (x : ℝ) (h : 5 * x^2 - x - 2 = 0) :
  (2*x + 1) * (2*x - 1) + x * (x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2864_286431


namespace NUMINAMATH_CALUDE_divisible_by_27_l2864_286466

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, 2^(5*n + 1) + 5^(n + 2) = 27 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_27_l2864_286466


namespace NUMINAMATH_CALUDE_motorcycle_theorem_l2864_286485

def motorcycle_problem (k r t : ℝ) (h1 : k > 0) (h2 : r > 0) (h3 : t > 0) : Prop :=
  ∃ (v1 v2 : ℝ), v1 > v2 ∧ v2 > 0 ∧
    r * (v1 - v2) = k ∧
    t * (v1 + v2) = k ∧
    v1 / v2 = |r + t| / |r - t|

theorem motorcycle_theorem (k r t : ℝ) (h1 : k > 0) (h2 : r > 0) (h3 : t > 0) :
  motorcycle_problem k r t h1 h2 h3 :=
sorry

end NUMINAMATH_CALUDE_motorcycle_theorem_l2864_286485
