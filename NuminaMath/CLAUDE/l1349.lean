import Mathlib

namespace NUMINAMATH_CALUDE_expression_factorization_l1349_134944

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 45 * x^2 - 3) - (-3 * x^3 + 6 * x^2 - 3) = 3 * x^2 * (5 * x + 13) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1349_134944


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1349_134996

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (m : ℕ), (1101 + m) % 24 = 0 → m ≥ n) ∧
  (1101 + n) % 24 = 0 := by
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1349_134996


namespace NUMINAMATH_CALUDE_square_sum_theorem_l1349_134976

theorem square_sum_theorem (x y : ℝ) 
  (h1 : (x + y)^4 + (x - y)^4 = 4112)
  (h2 : x^2 - y^2 = 16) : 
  x^2 + y^2 = 34 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l1349_134976


namespace NUMINAMATH_CALUDE_horizontal_line_slope_line_2023_slope_l1349_134946

/-- The slope of a horizontal line y = k is 0 -/
theorem horizontal_line_slope (k : ℝ) : 
  let f : ℝ → ℝ := λ x => k
  (∀ x : ℝ, (f x) = k) → 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) = 0) :=
by
  sorry

/-- The slope of the line y = 2023 is 0 -/
theorem line_2023_slope : 
  let f : ℝ → ℝ := λ x => 2023
  (∀ x : ℝ, (f x) = 2023) → 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_horizontal_line_slope_line_2023_slope_l1349_134946


namespace NUMINAMATH_CALUDE_root_one_when_sum_zero_reciprocal_roots_l1349_134969

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Theorem 1: If a + b + c = 0, then x = 1 is a root
theorem root_one_when_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hsum : a + b + c = 0) :
  quadratic a b c 1 := by sorry

-- Theorem 2: If x1 and x2 are roots of ax^2 + bx + c = 0 where x1 ≠ x2 ≠ 0,
-- then 1/x1 and 1/x2 are roots of cx^2 + bx + a = 0 (c ≠ 0)
theorem reciprocal_roots (a b c x1 x2 : ℝ) (ha : a ≠ 0) (hc : c ≠ 0)
  (hx1 : x1 ≠ 0) (hx2 : x2 ≠ 0) (hx1x2 : x1 ≠ x2)
  (hroot1 : quadratic a b c x1) (hroot2 : quadratic a b c x2) :
  quadratic c b a (1/x1) ∧ quadratic c b a (1/x2) := by sorry

end NUMINAMATH_CALUDE_root_one_when_sum_zero_reciprocal_roots_l1349_134969


namespace NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l1349_134935

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Creates a triangle from a line and coordinate axes -/
def triangleFromLine (l : Line) : Triangle :=
  sorry

/-- Calculates the sum of altitudes of a triangle -/
def sumOfAltitudes (t : Triangle) : ℝ :=
  sorry

/-- The main theorem -/
theorem sum_of_altitudes_for_specific_line :
  let l : Line := { a := 15, b := 8, c := 120 }
  let t : Triangle := triangleFromLine l
  sumOfAltitudes t = 391 / 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l1349_134935


namespace NUMINAMATH_CALUDE_overlap_percentage_l1349_134945

theorem overlap_percentage (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 10 →
  rect_length = 18 →
  rect_width = 10 →
  (2 * square_side - rect_length) * rect_width / (rect_length * rect_width) * 100 = 11.11 := by
sorry

end NUMINAMATH_CALUDE_overlap_percentage_l1349_134945


namespace NUMINAMATH_CALUDE_race_car_time_problem_l1349_134978

theorem race_car_time_problem (time_A time_sync : ℕ) (time_B : ℕ) : 
  time_A = 28 →
  time_sync = 168 →
  time_sync % time_A = 0 →
  time_sync % time_B = 0 →
  time_B > time_A →
  time_B < time_sync →
  (time_sync / time_A) % (time_sync / time_B) = 0 →
  time_B = 42 :=
by sorry

end NUMINAMATH_CALUDE_race_car_time_problem_l1349_134978


namespace NUMINAMATH_CALUDE_arccos_cos_ten_equals_two_l1349_134974

theorem arccos_cos_ten_equals_two : Real.arccos (Real.cos 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_ten_equals_two_l1349_134974


namespace NUMINAMATH_CALUDE_number_puzzle_l1349_134960

theorem number_puzzle (x : ℝ) : (x / 8) - 160 = 12 → x = 1376 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1349_134960


namespace NUMINAMATH_CALUDE_insurance_cost_calculation_l1349_134964

def apartment_cost : ℝ := 7000000
def loan_amount : ℝ := 4000000
def interest_rate : ℝ := 0.101
def property_insurance_rate : ℝ := 0.0009
def life_health_insurance_female : ℝ := 0.0017
def life_health_insurance_male : ℝ := 0.0019
def title_insurance_rate : ℝ := 0.0027
def svetlana_ratio : ℝ := 0.2
def dmitry_ratio : ℝ := 0.8

def total_insurance_cost : ℝ :=
  let total_loan := loan_amount * (1 + interest_rate)
  let property_insurance := total_loan * property_insurance_rate
  let title_insurance := total_loan * title_insurance_rate
  let svetlana_insurance := total_loan * svetlana_ratio * life_health_insurance_female
  let dmitry_insurance := total_loan * dmitry_ratio * life_health_insurance_male
  property_insurance + title_insurance + svetlana_insurance + dmitry_insurance

theorem insurance_cost_calculation :
  total_insurance_cost = 24045.84 := by sorry

end NUMINAMATH_CALUDE_insurance_cost_calculation_l1349_134964


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_first_five_l1349_134981

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

def sum_arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_first_five
  (a d : ℤ)
  (h1 : arithmetic_sequence a d 6 = 10)
  (h2 : arithmetic_sequence a d 7 = 15)
  (h3 : arithmetic_sequence a d 8 = 20) :
  sum_arithmetic_sequence a d 5 = -25 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_first_five_l1349_134981


namespace NUMINAMATH_CALUDE_selection_problem_l1349_134961

theorem selection_problem (n r : ℕ) (h : r < n) :
  -- Number of ways to select r people from 2n people in a row with no adjacent selections
  (Nat.choose (2*n - r + 1) r) = 
    (Nat.choose (2*n - r + 1) r) ∧
  -- Number of ways to select r people from 2n people in a circle with no adjacent selections
  ((2*n : ℚ) / (2*n - r : ℚ)) * (Nat.choose (2*n - r) r) = 
    ((2*n : ℚ) / (2*n - r : ℚ)) * (Nat.choose (2*n - r) r) := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l1349_134961


namespace NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l1349_134907

/-- The number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

theorem students_play_both_football_and_cricket :
  students_play_both 450 325 175 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l1349_134907


namespace NUMINAMATH_CALUDE_half_correct_probability_l1349_134957

def num_questions : ℕ := 10
def num_correct : ℕ := 5
def probability_correct : ℚ := 1/2

theorem half_correct_probability :
  (Nat.choose num_questions num_correct) * (probability_correct ^ num_correct) * ((1 - probability_correct) ^ (num_questions - num_correct)) = 63/256 := by
  sorry

end NUMINAMATH_CALUDE_half_correct_probability_l1349_134957


namespace NUMINAMATH_CALUDE_festival_lineup_theorem_l1349_134952

/-- The minimum number of Gennadys required for the festival lineup -/
def min_gennadys (num_alexanders num_borises num_vasilys : ℕ) : ℕ :=
  max 0 (num_borises - 1 - (num_alexanders + num_vasilys))

/-- Theorem stating the minimum number of Gennadys required for the festival lineup -/
theorem festival_lineup_theorem (num_alexanders num_borises num_vasilys : ℕ) 
  (h1 : num_alexanders = 45)
  (h2 : num_borises = 122)
  (h3 : num_vasilys = 27) :
  min_gennadys num_alexanders num_borises num_vasilys = 49 := by
  sorry

#eval min_gennadys 45 122 27

end NUMINAMATH_CALUDE_festival_lineup_theorem_l1349_134952


namespace NUMINAMATH_CALUDE_room_dimension_is_15_l1349_134968

/-- Represents the dimensions and properties of a room to be whitewashed -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  doorArea : ℝ
  windowArea : ℝ
  windowCount : ℕ
  whitewashCost : ℝ
  totalCost : ℝ

/-- Calculates the total area to be whitewashed in the room -/
def areaToWhitewash (r : Room) : ℝ :=
  2 * (r.length * r.height + r.width * r.height) - (r.doorArea + r.windowCount * r.windowArea)

/-- Theorem stating that the unknown dimension of the room is 15 feet -/
theorem room_dimension_is_15 (r : Room) 
  (h1 : r.length = 25)
  (h2 : r.height = 12)
  (h3 : r.doorArea = 18)
  (h4 : r.windowArea = 12)
  (h5 : r.windowCount = 3)
  (h6 : r.whitewashCost = 5)
  (h7 : r.totalCost = 4530)
  (h8 : r.totalCost = r.whitewashCost * areaToWhitewash r) :
  r.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_room_dimension_is_15_l1349_134968


namespace NUMINAMATH_CALUDE_new_person_weight_l1349_134915

theorem new_person_weight (n : ℕ) (avg_increase weight_replaced : ℝ) :
  n = 7 →
  avg_increase = 6.2 →
  weight_replaced = 76 →
  n * avg_increase + weight_replaced = 119.4 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1349_134915


namespace NUMINAMATH_CALUDE_salmon_migration_result_l1349_134937

/-- The total number of salmon in a river after migration -/
def total_salmon (initial : ℕ) (increase_factor : ℕ) : ℕ :=
  initial + initial * increase_factor

/-- Theorem: Given 500 initial salmon and a tenfold increase, the total is 5500 -/
theorem salmon_migration_result :
  total_salmon 500 10 = 5500 := by
  sorry

end NUMINAMATH_CALUDE_salmon_migration_result_l1349_134937


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l1349_134922

/-- The volume of a slice of pizza -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) :
  thickness = 1/2 →
  diameter = 10 →
  num_slices = 10 →
  (π * (diameter/2)^2 * thickness) / num_slices = 5*π/4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_volume_l1349_134922


namespace NUMINAMATH_CALUDE_B_equals_interval_A_union_C_equals_A_l1349_134995

-- Define sets A, B, and C
def A : Set ℝ := {x | 2 * x^2 - 9 * x + 4 > 0}
def B : Set ℝ := {y | ∃ x ∈ (Set.univ \ A), y = -x^2 + 2 * x}
def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x ≤ 2 * m - 1}

-- Theorem 1: B is equal to the closed interval [-8, 1]
theorem B_equals_interval : B = Set.Icc (-8) 1 := by sorry

-- Theorem 2: A ∪ C = A if and only if m ≤ 2 or m ≥ 3
theorem A_union_C_equals_A (m : ℝ) : A ∪ C m = A ↔ m ≤ 2 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_B_equals_interval_A_union_C_equals_A_l1349_134995


namespace NUMINAMATH_CALUDE_collinear_vectors_dot_product_l1349_134994

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), b.1 = t * a.1 ∧ b.2 = t * a.2

/-- The dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem collinear_vectors_dot_product :
  let a : ℝ × ℝ := (3/2, 1)
  let b : ℝ × ℝ := (3, k)
  collinear a b →
  dot_product (a.1 - b.1, a.2 - b.2) (2 * a.1 + b.1, 2 * a.2 + b.2) = -13 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_dot_product_l1349_134994


namespace NUMINAMATH_CALUDE_work_completion_time_l1349_134998

/-- The number of days it takes for person A to complete the work alone -/
def days_A : ℝ := 15

/-- The fraction of work completed by A and B together in 5 days -/
def work_completed : ℝ := 0.5

/-- The number of days A and B work together -/
def days_together : ℝ := 5

/-- The number of days it takes for person B to complete the work alone -/
def days_B : ℝ := 30

theorem work_completion_time :
  (1 / days_A + 1 / days_B) * days_together = work_completed := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1349_134998


namespace NUMINAMATH_CALUDE_equation_roots_l1349_134977

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => -x * (x + 3) - x * (x + 3)
  (f 0 = 0 ∧ f (-3) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 0 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l1349_134977


namespace NUMINAMATH_CALUDE_min_difference_l1349_134966

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x - 3)
noncomputable def g (x : ℝ) : ℝ := 1 / 4 + Real.log (x / 2)

theorem min_difference (m n : ℝ) (h : f m = g n) :
  ∃ (d : ℝ), d = 1 / 2 + Real.log 2 ∧ n - m ≥ d ∧ ∃ (m₀ n₀ : ℝ), f m₀ = g n₀ ∧ n₀ - m₀ = d :=
sorry

end NUMINAMATH_CALUDE_min_difference_l1349_134966


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l1349_134908

def number_of_students : ℕ := 4
def fixed_position : ℕ := 1
def available_positions : ℕ := number_of_students - fixed_position

theorem relay_race_arrangements :
  (available_positions.factorial) = 6 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l1349_134908


namespace NUMINAMATH_CALUDE_percentage_of_whole_l1349_134971

theorem percentage_of_whole (whole : ℝ) (part : ℝ) (h : whole = 450 ∧ part = 229.5) :
  (part / whole) * 100 = 51 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_whole_l1349_134971


namespace NUMINAMATH_CALUDE_cos_negative_thirteen_pi_over_four_l1349_134991

theorem cos_negative_thirteen_pi_over_four :
  Real.cos (-13 * π / 4) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_negative_thirteen_pi_over_four_l1349_134991


namespace NUMINAMATH_CALUDE_new_person_weight_l1349_134986

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (leaving_weight : ℝ) (average_increase : ℝ) : ℝ :=
  leaving_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 65 kg -/
theorem new_person_weight :
  weight_of_new_person 8 45 2.5 = 65 := by
  sorry

#eval weight_of_new_person 8 45 2.5

end NUMINAMATH_CALUDE_new_person_weight_l1349_134986


namespace NUMINAMATH_CALUDE_average_of_remaining_digits_l1349_134940

theorem average_of_remaining_digits 
  (total_count : Nat) 
  (subset_count : Nat) 
  (total_average : ℝ) 
  (subset_average : ℝ) 
  (h1 : total_count = 20) 
  (h2 : subset_count = 14) 
  (h3 : total_average = 500) 
  (h4 : subset_average = 390) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 756.67 := by
sorry

#eval (20 * 500 - 14 * 390) / 6

end NUMINAMATH_CALUDE_average_of_remaining_digits_l1349_134940


namespace NUMINAMATH_CALUDE_sum_mod_seven_l1349_134927

theorem sum_mod_seven : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l1349_134927


namespace NUMINAMATH_CALUDE_regular_tetrahedron_edges_l1349_134920

/-- A regular tetrahedron is a tetrahedron in which all faces are congruent equilateral triangles. -/
def RegularTetrahedron : Type := sorry

/-- The number of edges in a geometric shape. -/
def num_edges (shape : Type) : ℕ := sorry

/-- Theorem: A regular tetrahedron has 6 edges. -/
theorem regular_tetrahedron_edges : num_edges RegularTetrahedron = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_edges_l1349_134920


namespace NUMINAMATH_CALUDE_unique_residue_mod_11_l1349_134942

theorem unique_residue_mod_11 :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_unique_residue_mod_11_l1349_134942


namespace NUMINAMATH_CALUDE_max_profit_multimedia_devices_l1349_134923

/-- Represents the profit function for multimedia devices -/
def profit_function (x : ℝ) : ℝ := -0.1 * x + 20

/-- Represents the constraint on the quantity of devices -/
def quantity_constraint (x : ℝ) : Prop := 4 * x ≥ 50 - x

/-- Theorem stating the maximum profit and optimal quantity of type A devices -/
theorem max_profit_multimedia_devices :
  ∃ (x : ℝ), 
    quantity_constraint x ∧ 
    profit_function x = 19 ∧ 
    x = 10 ∧
    ∀ (y : ℝ), quantity_constraint y → profit_function y ≤ profit_function x :=
by
  sorry


end NUMINAMATH_CALUDE_max_profit_multimedia_devices_l1349_134923


namespace NUMINAMATH_CALUDE_equal_value_proof_l1349_134913

theorem equal_value_proof (a b : ℝ) (h1 : 10 * a = 6 * b) (h2 : 120 * a * b = 800) :
  10 * a = 20 ∧ 6 * b = 20 := by
  sorry

end NUMINAMATH_CALUDE_equal_value_proof_l1349_134913


namespace NUMINAMATH_CALUDE_circle_point_x_coordinate_l1349_134992

theorem circle_point_x_coordinate :
  ∀ (x : ℝ),
  let center_x : ℝ := (-3 + 21) / 2
  let center_y : ℝ := 0
  let radius : ℝ := (21 - (-3)) / 2
  (x - center_x)^2 + (12 - center_y)^2 = radius^2 →
  x = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_point_x_coordinate_l1349_134992


namespace NUMINAMATH_CALUDE_train_crossing_time_l1349_134932

theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 100 ∧ train_speed_kmh = 90 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1349_134932


namespace NUMINAMATH_CALUDE_ladybug_dots_count_l1349_134948

/-- The number of ladybugs Andre caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of ladybugs Andre caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The number of dots each ladybug has -/
def dots_per_ladybug : ℕ := 6

/-- The total number of dots on all ladybugs caught by Andre -/
def total_dots : ℕ := (monday_ladybugs + tuesday_ladybugs) * dots_per_ladybug

theorem ladybug_dots_count : total_dots = 78 := by
  sorry

end NUMINAMATH_CALUDE_ladybug_dots_count_l1349_134948


namespace NUMINAMATH_CALUDE_set_of_possible_a_l1349_134983

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N (a : ℝ) : Set ℝ := {x | x^2 - a*x + 3*a - 5 = 0}

theorem set_of_possible_a (a : ℝ) : M ∪ N a = M → 2 ≤ a ∧ a < 10 := by
  sorry

end NUMINAMATH_CALUDE_set_of_possible_a_l1349_134983


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l1349_134982

-- First inequality
theorem inequality_one (x : ℝ) : 
  (|1 - (2*x - 1)/3| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 5) :=
sorry

-- Second inequality
theorem inequality_two (x : ℝ) :
  ((2 - x)*(x + 3) < 2 - x) ↔ (x > 2 ∨ x < -2) :=
sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l1349_134982


namespace NUMINAMATH_CALUDE_increased_value_l1349_134972

theorem increased_value (x : ℝ) (p : ℝ) (h1 : x = 1200) (h2 : p = 40) :
  x * (1 + p / 100) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_increased_value_l1349_134972


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l1349_134958

-- Define the triangle
structure ObtuseIsoscelesTriangle where
  -- The largest angle in degrees
  largest_angle : ℝ
  -- One of the two equal angles in degrees
  equal_angle : ℝ
  -- Conditions
  is_obtuse : largest_angle > 90
  is_isosceles : equal_angle = equal_angle
  angle_sum : largest_angle + 2 * equal_angle = 180

-- Theorem statement
theorem smallest_angle_measure (t : ObtuseIsoscelesTriangle) 
  (h : t.largest_angle = 108) : t.equal_angle = 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l1349_134958


namespace NUMINAMATH_CALUDE_solve_potatoes_problem_l1349_134926

def potatoes_problem (total : ℕ) (gina : ℕ) : Prop :=
  let tom := 2 * gina
  let anne := tom / 3
  let remaining := total - (gina + tom + anne)
  remaining = 47

theorem solve_potatoes_problem :
  potatoes_problem 300 69 := by
  sorry

end NUMINAMATH_CALUDE_solve_potatoes_problem_l1349_134926


namespace NUMINAMATH_CALUDE_soft_drink_bottles_sold_l1349_134939

theorem soft_drink_bottles_sold (small_bottles : ℕ) (big_bottles : ℕ) 
  (small_sold_percent : ℚ) (total_remaining : ℕ) : 
  small_bottles = 6000 →
  big_bottles = 14000 →
  small_sold_percent = 1/5 →
  total_remaining = 15580 →
  (big_bottles - (total_remaining - (small_bottles - small_bottles * small_sold_percent))) / big_bottles = 23/100 := by
  sorry

end NUMINAMATH_CALUDE_soft_drink_bottles_sold_l1349_134939


namespace NUMINAMATH_CALUDE_balloon_cost_theorem_l1349_134959

/-- Represents the cost of balloons for a person -/
structure BalloonCost where
  count : ℕ
  price : ℚ

/-- Calculates the total cost for a person's balloons -/
def totalCost (bc : BalloonCost) : ℚ :=
  bc.count * bc.price

theorem balloon_cost_theorem (fred sam dan : BalloonCost)
  (h_fred : fred = ⟨10, 1⟩)
  (h_sam : sam = ⟨46, (3/2)⟩)
  (h_dan : dan = ⟨16, (3/4)⟩) :
  totalCost fred + totalCost sam + totalCost dan = 91 := by
  sorry

end NUMINAMATH_CALUDE_balloon_cost_theorem_l1349_134959


namespace NUMINAMATH_CALUDE_diane_gambling_problem_l1349_134910

theorem diane_gambling_problem (initial_amount : ℝ) : 
  (initial_amount + 65 + 50 = 215) → initial_amount = 100 := by
sorry

end NUMINAMATH_CALUDE_diane_gambling_problem_l1349_134910


namespace NUMINAMATH_CALUDE_count_arrangements_no_adjacent_girls_count_arrangements_AB_adjacent_l1349_134943

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 4

/-- The number of arrangements where no two girls are adjacent -/
def arrangements_no_adjacent_girls : ℕ := 144

/-- The number of arrangements where boys A and B are adjacent -/
def arrangements_AB_adjacent : ℕ := 240

/-- Theorem stating the number of arrangements where no two girls are adjacent -/
theorem count_arrangements_no_adjacent_girls :
  (num_boys.factorial * num_girls.factorial) = arrangements_no_adjacent_girls := by
  sorry

/-- Theorem stating the number of arrangements where boys A and B are adjacent -/
theorem count_arrangements_AB_adjacent :
  ((num_boys + num_girls - 1).factorial * 2) = arrangements_AB_adjacent := by
  sorry

end NUMINAMATH_CALUDE_count_arrangements_no_adjacent_girls_count_arrangements_AB_adjacent_l1349_134943


namespace NUMINAMATH_CALUDE_students_liking_table_tennis_not_basketball_l1349_134993

theorem students_liking_table_tennis_not_basketball
  (total : ℕ)
  (basketball : ℕ)
  (table_tennis : ℕ)
  (neither : ℕ)
  (h1 : total = 40)
  (h2 : basketball = 17)
  (h3 : table_tennis = 20)
  (h4 : neither = 8)
  : ∃ (x : ℕ), x = table_tennis - (total - neither - (basketball + table_tennis - x)) ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_table_tennis_not_basketball_l1349_134993


namespace NUMINAMATH_CALUDE_inequality_solution_l1349_134962

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 18*x + 35) / (x^2 - 4*x + 8) ∧
  (x^2 - 18*x + 35) / (x^2 - 4*x + 8) < 2 →
  3 < x ∧ x < 17/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1349_134962


namespace NUMINAMATH_CALUDE_total_bottles_l1349_134924

theorem total_bottles (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 30) (h2 : diet_soda = 8) : 
  regular_soda + diet_soda = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_l1349_134924


namespace NUMINAMATH_CALUDE_sochi_price_decrease_in_euros_l1349_134980

/-- Represents the price decrease in Sochi apartments in euros -/
def sochi_price_decrease_euros : ℝ := 32.5

/-- The price decrease of Moscow apartments in rubles -/
def moscow_price_decrease_rubles : ℝ := 20

/-- The price decrease of Moscow apartments in euros -/
def moscow_price_decrease_euros : ℝ := 40

/-- The price decrease of Sochi apartments in rubles -/
def sochi_price_decrease_rubles : ℝ := 10

theorem sochi_price_decrease_in_euros :
  let initial_price_rubles : ℝ := 100  -- Arbitrary initial price
  let initial_price_euros : ℝ := 100   -- Arbitrary initial price
  let moscow_new_price_rubles : ℝ := initial_price_rubles * (1 - moscow_price_decrease_rubles / 100)
  let moscow_new_price_euros : ℝ := initial_price_euros * (1 - moscow_price_decrease_euros / 100)
  let sochi_new_price_rubles : ℝ := initial_price_rubles * (1 - sochi_price_decrease_rubles / 100)
  let exchange_rate : ℝ := moscow_new_price_rubles / moscow_new_price_euros
  let sochi_new_price_euros : ℝ := sochi_new_price_rubles / exchange_rate
  (initial_price_euros - sochi_new_price_euros) / initial_price_euros * 100 = sochi_price_decrease_euros :=
by sorry

end NUMINAMATH_CALUDE_sochi_price_decrease_in_euros_l1349_134980


namespace NUMINAMATH_CALUDE_abs_negative_2023_l1349_134917

theorem abs_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l1349_134917


namespace NUMINAMATH_CALUDE_average_speed_last_hour_l1349_134911

theorem average_speed_last_hour (total_distance : ℝ) (total_time : ℝ) 
  (first_30_speed : ℝ) (next_30_speed : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  first_30_speed = 50 →
  next_30_speed = 70 →
  let first_30_distance := first_30_speed * (30 / 60)
  let next_30_distance := next_30_speed * (30 / 60)
  let last_60_distance := total_distance - (first_30_distance + next_30_distance)
  let last_60_time := 60 / 60
  last_60_distance / last_60_time = 60 := by
  sorry

#check average_speed_last_hour

end NUMINAMATH_CALUDE_average_speed_last_hour_l1349_134911


namespace NUMINAMATH_CALUDE_saree_original_price_l1349_134936

theorem saree_original_price (P : ℝ) : 
  (P * (1 - 0.2) * (1 - 0.3) = 313.6) → P = 560 := by
  sorry

end NUMINAMATH_CALUDE_saree_original_price_l1349_134936


namespace NUMINAMATH_CALUDE_project_completion_time_l1349_134938

/-- The number of days it takes A to complete the project alone -/
def a_days : ℝ := 10

/-- The number of days it takes B to complete the project alone -/
def b_days : ℝ := 30

/-- The number of days before project completion that A quits -/
def a_quit_days : ℝ := 10

/-- The total number of days to complete the project with A and B working together, with A quitting early -/
def total_days : ℝ := 15

theorem project_completion_time :
  let a_rate : ℝ := 1 / a_days
  let b_rate : ℝ := 1 / b_days
  (total_days - a_quit_days) * a_rate + total_days * b_rate = 1 :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l1349_134938


namespace NUMINAMATH_CALUDE_triangle_heights_theorem_l1349_134903

/-- A triangle with given heights -/
structure Triangle where
  ha : ℝ
  hb : ℝ
  hc : ℝ

/-- Definition of an acute triangle based on heights -/
def is_acute (t : Triangle) : Prop :=
  t.ha > 0 ∧ t.hb > 0 ∧ t.hc > 0 ∧ t.ha ≠ t.hb ∧ t.hb ≠ t.hc ∧ t.ha ≠ t.hc

/-- Definition of triangle existence based on heights -/
def triangle_exists (t : Triangle) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    t.ha = (2 * (a * b * c) / (a * b + b * c + c * a)) / a ∧
    t.hb = (2 * (a * b * c) / (a * b + b * c + c * a)) / b ∧
    t.hc = (2 * (a * b * c) / (a * b + b * c + c * a)) / c

theorem triangle_heights_theorem :
  (let t1 : Triangle := ⟨4, 5, 6⟩
   is_acute t1) ∧
  (let t2 : Triangle := ⟨2, 3, 6⟩
   ¬ triangle_exists t2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_heights_theorem_l1349_134903


namespace NUMINAMATH_CALUDE_chocolates_in_cost_price_l1349_134956

/-- The number of chocolates in the cost price -/
def n : ℕ := sorry

/-- The cost price of one chocolate -/
def C : ℝ := sorry

/-- The selling price of one chocolate -/
def S : ℝ := sorry

/-- The cost price of n chocolates equals the selling price of 16 chocolates -/
axiom cost_price_eq_selling_price : n * C = 16 * S

/-- The gain percent is 50% -/
axiom gain_percent : S = 1.5 * C

theorem chocolates_in_cost_price : n = 24 := by sorry

end NUMINAMATH_CALUDE_chocolates_in_cost_price_l1349_134956


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1349_134984

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def intersection_points : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points = 210 :=
sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l1349_134984


namespace NUMINAMATH_CALUDE_min_sum_distances_l1349_134921

theorem min_sum_distances (a b : ℝ) :
  Real.sqrt ((a - 1)^2 + (b - 1)^2) + Real.sqrt ((a + 1)^2 + (b + 1)^2) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_distances_l1349_134921


namespace NUMINAMATH_CALUDE_complement_of_57_13_l1349_134949

/-- Represents an angle in degrees and minutes -/
structure Angle where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the complement of an angle -/
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60,
    valid := by sorry }

/-- The main theorem stating that the complement of 57°13' is 32°47' -/
theorem complement_of_57_13 :
  complement { degrees := 57, minutes := 13, valid := by sorry } =
  { degrees := 32, minutes := 47, valid := by sorry } := by
  sorry

end NUMINAMATH_CALUDE_complement_of_57_13_l1349_134949


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l1349_134985

theorem coefficient_x_cubed_in_binomial_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) * (1 ^ (5 - k)) * (1 ^ k) * (if k = 3 then 1 else 0)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l1349_134985


namespace NUMINAMATH_CALUDE_kelly_gave_away_64_games_l1349_134955

/-- The number of games Kelly gave away -/
def games_given_away (initial_games final_games : ℕ) : ℕ :=
  initial_games - final_games

/-- Theorem: Kelly gave away 64 games -/
theorem kelly_gave_away_64_games :
  games_given_away 106 42 = 64 := by
  sorry

end NUMINAMATH_CALUDE_kelly_gave_away_64_games_l1349_134955


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1349_134904

/-- Represents the interest rate calculation problem --/
theorem interest_rate_calculation 
  (principal : ℝ) 
  (amount : ℝ) 
  (time : ℝ) 
  (h1 : principal = 896) 
  (h2 : amount = 1120) 
  (h3 : time = 5) :
  (amount - principal) / (principal * time) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1349_134904


namespace NUMINAMATH_CALUDE_minimum_discount_l1349_134901

theorem minimum_discount (n : ℕ) : n = 38 ↔ 
  (n > 0) ∧
  (∀ m : ℕ, m < n → 
    ((1 - m / 100 : ℝ) ≥ (1 - 0.20) * (1 - 0.10) ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.08)^4 ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.30) * (1 - 0.10))) ∧
  ((1 - n / 100 : ℝ) < (1 - 0.20) * (1 - 0.10) ∧
   (1 - n / 100 : ℝ) < (1 - 0.08)^4 ∧
   (1 - n / 100 : ℝ) < (1 - 0.30) * (1 - 0.10)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_discount_l1349_134901


namespace NUMINAMATH_CALUDE_weekly_diaper_sales_revenue_l1349_134905

/-- Represents the weekly diaper sales revenue calculation --/
theorem weekly_diaper_sales_revenue :
  let boxes_per_week : ℕ := 30
  let packs_per_box : ℕ := 40
  let diapers_per_pack : ℕ := 160
  let price_per_diaper : ℚ := 4
  let bundle_discount : ℚ := 0.05
  let special_discount : ℚ := 0.05
  let tax_rate : ℚ := 0.10

  let total_diapers : ℕ := boxes_per_week * packs_per_box * diapers_per_pack
  let base_revenue : ℚ := total_diapers * price_per_diaper
  let after_bundle_discount : ℚ := base_revenue * (1 - bundle_discount)
  let after_special_discount : ℚ := after_bundle_discount * (1 - special_discount)
  let final_revenue : ℚ := after_special_discount * (1 + tax_rate)

  final_revenue = 762432 :=
by sorry


end NUMINAMATH_CALUDE_weekly_diaper_sales_revenue_l1349_134905


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1349_134999

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + 8*x + b = (x + k)^2) → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1349_134999


namespace NUMINAMATH_CALUDE_larger_number_proof_l1349_134973

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 25) (h2 : Nat.lcm a b = 25 * 14 * 16) :
  max a b = 400 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1349_134973


namespace NUMINAMATH_CALUDE_units_digit_product_minus_power_l1349_134965

def units_digit (n : ℤ) : ℕ :=
  (n % 10).toNat

theorem units_digit_product_minus_power : units_digit (8 * 18 * 1988 - 8^4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_minus_power_l1349_134965


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1349_134919

theorem quadratic_function_range (a : ℝ) : 
  (∃ x₀ : ℝ, |x₀^2 + a*x₀ + 1| ≤ 1/4 ∧ |(x₀+1)^2 + a*(x₀+1) + 1| ≤ 1/4) → 
  a ∈ Set.Icc (-Real.sqrt 6) (-2) ∪ Set.Icc 2 (Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1349_134919


namespace NUMINAMATH_CALUDE_jills_nickels_l1349_134914

/-- Proves that Jill has 30 nickels given the conditions of the problem -/
theorem jills_nickels (total_coins : ℕ) (total_value : ℚ) (nickel_value dime_value : ℚ) :
  total_coins = 50 →
  total_value = (350 : ℚ) / 100 →
  nickel_value = (5 : ℚ) / 100 →
  dime_value = (10 : ℚ) / 100 →
  ∃ (nickels dimes : ℕ),
    nickels + dimes = total_coins ∧
    nickels * nickel_value + dimes * dime_value = total_value ∧
    nickels = 30 :=
by sorry

end NUMINAMATH_CALUDE_jills_nickels_l1349_134914


namespace NUMINAMATH_CALUDE_lions_volleyball_games_l1349_134902

theorem lions_volleyball_games 
  (initial_win_rate : Real) 
  (initial_win_rate_value : initial_win_rate = 0.60)
  (final_win_rate : Real) 
  (final_win_rate_value : final_win_rate = 0.55)
  (tournament_wins : Nat) 
  (tournament_wins_value : tournament_wins = 8)
  (tournament_losses : Nat) 
  (tournament_losses_value : tournament_losses = 4) :
  ∃ (total_games : Nat), 
    total_games = 40 ∧ 
    (initial_win_rate * (total_games - tournament_wins - tournament_losses) + tournament_wins) / total_games = final_win_rate :=
by sorry

end NUMINAMATH_CALUDE_lions_volleyball_games_l1349_134902


namespace NUMINAMATH_CALUDE_soda_sales_difference_l1349_134953

/-- Calculates the difference between evening and morning sales for Remy and Nick's soda business -/
theorem soda_sales_difference (remy_morning : ℕ) (nick_difference : ℕ) (price : ℚ) (evening_sales : ℚ) : 
  remy_morning = 55 →
  nick_difference = 6 →
  price = 1/2 →
  evening_sales = 55 →
  evening_sales - (price * (remy_morning + (remy_morning - nick_difference))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_soda_sales_difference_l1349_134953


namespace NUMINAMATH_CALUDE_remaining_money_l1349_134925

def savings : ℕ := 5376
def ticket_cost : ℕ := 1350

def octal_to_decimal (n : ℕ) : ℕ := sorry

theorem remaining_money : 
  octal_to_decimal savings - ticket_cost = 1464 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l1349_134925


namespace NUMINAMATH_CALUDE_two_solutions_l1349_134967

/-- A solution to the system of equations is a triple of positive integers (x, y, z) 
    satisfying the given conditions. -/
def IsSolution (x y z : ℕ+) : Prop :=
  x * y + y * z = 63 ∧ x * z + y * z = 23

/-- The theorem states that there are exactly two solutions to the system of equations. -/
theorem two_solutions : 
  ∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    (∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ IsSolution x y z) ∧ 
    Finset.card s = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_l1349_134967


namespace NUMINAMATH_CALUDE_min_k_value_l1349_134906

theorem min_k_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ k : ℝ, (1/a + 1/b + k/(a+b) ≥ 0)) : 
  ∃ k_min : ℝ, k_min = -4 ∧ ∀ k : ℝ, (1/a + 1/b + k/(a+b) ≥ 0) → k ≥ k_min :=
sorry

end NUMINAMATH_CALUDE_min_k_value_l1349_134906


namespace NUMINAMATH_CALUDE_parallel_line_through_point_line_equation_proof_l1349_134987

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point (given_line : Line) (point : ℝ × ℝ) :
  ∃ (parallel_line : Line),
    parallel_line.contains point.1 point.2 ∧
    Line.parallel parallel_line given_line :=
by
  sorry

/-- The main theorem to prove -/
theorem line_equation_proof :
  let given_line : Line := { a := 1, b := -2, c := -2 }
  let point : ℝ × ℝ := (1, 1)
  let parallel_line : Line := { a := 1, b := -2, c := 1 }
  parallel_line.contains point.1 point.2 ∧
  Line.parallel parallel_line given_line :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_line_equation_proof_l1349_134987


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1349_134979

theorem imaginary_part_of_z (z : ℂ) : 
  z * (1 + 2 * I ^ 6) = (2 - 3 * I) / I → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1349_134979


namespace NUMINAMATH_CALUDE_min_value_of_g_l1349_134916

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - 2 * a^(-x)

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2 * f a x

-- Theorem statement
theorem min_value_of_g (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3) :
  ∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, g a x ≤ g a y ∧ g a x = -2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_g_l1349_134916


namespace NUMINAMATH_CALUDE_function_properties_imply_cosine_and_value_l1349_134997

/-- The function f(x) = sin(ωx + φ) with given properties -/
noncomputable def f (ω φ : ℝ) : ℝ → ℝ := fun x ↦ Real.sin (ω * x + φ)

/-- The theorem statement -/
theorem function_properties_imply_cosine_and_value
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 ≤ φ ∧ φ ≤ π)
  (h_even : ∀ x, f ω φ x = f ω φ (-x))
  (h_distance : ∃ (x₁ x₂ : ℝ), abs (x₁ - x₂) = π ∧ abs (f ω φ x₁ - f ω φ x₂) = 2)
  (α : ℝ)
  (h_sum : Real.sin α + f ω φ α = 2/3) :
  (∀ x, f ω φ x = Real.cos x) ∧
  ((Real.sqrt 2 * Real.sin (2*α - π/4) + 1) / (1 + Real.tan α) = -5/9) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_imply_cosine_and_value_l1349_134997


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1349_134954

theorem circle_area_ratio : 
  let d1 : ℝ := 2  -- diameter of smallest circle
  let d2 : ℝ := 6  -- diameter of middle circle
  let d3 : ℝ := 10 -- diameter of largest circle
  let r1 : ℝ := d1 / 2  -- radius of smallest circle
  let r2 : ℝ := d2 / 2  -- radius of middle circle
  let r3 : ℝ := d3 / 2  -- radius of largest circle
  let area_smallest : ℝ := π * r1^2
  let area_middle : ℝ := π * r2^2
  let area_largest : ℝ := π * r3^2
  let area_green : ℝ := area_largest - area_middle
  let area_red : ℝ := area_smallest
  (area_green / area_red : ℝ) = 16
  := by sorry


end NUMINAMATH_CALUDE_circle_area_ratio_l1349_134954


namespace NUMINAMATH_CALUDE_remaining_red_cards_l1349_134941

theorem remaining_red_cards (total_cards : ℕ) (red_cards : ℕ) (removed_cards : ℕ) : 
  total_cards = 52 → 
  red_cards = total_cards / 2 →
  removed_cards = 10 →
  red_cards - removed_cards = 16 := by
  sorry

end NUMINAMATH_CALUDE_remaining_red_cards_l1349_134941


namespace NUMINAMATH_CALUDE_mixed_gender_more_likely_l1349_134930

def num_children : ℕ := 5
def prob_boy : ℚ := 1/2
def prob_girl : ℚ := 1/2

def prob_all_boys : ℚ := prob_boy ^ num_children
def prob_all_girls : ℚ := prob_girl ^ num_children

def prob_three_girls_two_boys : ℚ := (Nat.choose num_children 3) * (prob_girl ^ 3) * (prob_boy ^ 2)
def prob_three_boys_two_girls : ℚ := (Nat.choose num_children 3) * (prob_boy ^ 3) * (prob_girl ^ 2)

theorem mixed_gender_more_likely :
  prob_three_girls_two_boys > prob_all_boys ∧
  prob_three_girls_two_boys > prob_all_girls ∧
  prob_three_boys_two_girls > prob_all_boys ∧
  prob_three_boys_two_girls > prob_all_girls :=
sorry

end NUMINAMATH_CALUDE_mixed_gender_more_likely_l1349_134930


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1349_134912

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0), if an isosceles right triangle
    MF₁F₂ is constructed with F₁ as the right-angle vertex and the midpoint of side MF₁ lies on the
    hyperbola, then the eccentricity of the hyperbola is (√5 + 1)/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)  -- Definition of eccentricity for a hyperbola
  ∃ (x y : ℝ), 
    x^2 / a^2 - y^2 / b^2 = 1 ∧  -- Point (x, y) is on the hyperbola
    x = -Real.sqrt (a^2 + b^2) / 2 ∧  -- x-coordinate of the midpoint of MF₁
    y = b^2 / (2*a) →  -- y-coordinate of the midpoint of MF₁
  e = (Real.sqrt 5 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1349_134912


namespace NUMINAMATH_CALUDE_inequality_implies_m_range_l1349_134989

theorem inequality_implies_m_range (m : ℝ) :
  (∀ x : ℝ, 4^x - m * 2^x + 1 > 0) → -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_m_range_l1349_134989


namespace NUMINAMATH_CALUDE_cube_inequality_l1349_134900

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l1349_134900


namespace NUMINAMATH_CALUDE_two_digit_numbers_satisfying_condition_l1349_134975

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  (sum_of_digits n)^2 = sum_of_digits (n^2)

theorem two_digit_numbers_satisfying_condition :
  {n : ℕ | is_two_digit n ∧ satisfies_condition n} =
  {10, 11, 12, 13, 20, 21, 22, 30, 31} := by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_satisfying_condition_l1349_134975


namespace NUMINAMATH_CALUDE_cheesecake_slice_price_l1349_134929

theorem cheesecake_slice_price (slices_per_pie : ℕ) (pies_sold : ℕ) (total_revenue : ℕ) : 
  slices_per_pie = 6 →
  pies_sold = 7 →
  total_revenue = 294 →
  total_revenue / (slices_per_pie * pies_sold) = 7 := by
sorry

end NUMINAMATH_CALUDE_cheesecake_slice_price_l1349_134929


namespace NUMINAMATH_CALUDE_unique_nonzero_solution_sum_of_squares_l1349_134951

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * y - 2 * y - 3 * x = 0
def equation2 (y z : ℝ) : Prop := y * z - 3 * z - 5 * y = 0
def equation3 (x z : ℝ) : Prop := x * z - 5 * x - 2 * z = 0

-- Define the theorem
theorem unique_nonzero_solution_sum_of_squares :
  ∃! (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
    equation1 a b ∧ equation2 b c ∧ equation3 a c →
    a^2 + b^2 + c^2 = 152 :=
by sorry

end NUMINAMATH_CALUDE_unique_nonzero_solution_sum_of_squares_l1349_134951


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1349_134970

theorem similar_triangle_perimeter (a b c d e f : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for the original triangle
  d^2 + e^2 = f^2 →  -- Pythagorean theorem for the similar triangle
  d / a = e / b →    -- Similarity condition
  d / a = f / c →    -- Similarity condition
  a = 6 →            -- Given length of shorter leg of original triangle
  b = 8 →            -- Given length of longer leg of original triangle
  d = 15 →           -- Given length of shorter leg of similar triangle
  d + e + f = 60 :=  -- Perimeter of the similar triangle
by
  sorry


end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1349_134970


namespace NUMINAMATH_CALUDE_product_remainder_mod_ten_l1349_134909

theorem product_remainder_mod_ten (a b c : ℕ) : 
  a % 10 = 7 → b % 10 = 1 → c % 10 = 3 → (a * b * c) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_ten_l1349_134909


namespace NUMINAMATH_CALUDE_jar_problem_l1349_134928

/-- Represents the number of small jars -/
def small_jars : ℕ := 62

/-- Represents the number of large jars -/
def large_jars : ℕ := 100 - small_jars

/-- Capacity of a small jar in liters -/
def small_jar_capacity : ℕ := 3

/-- Capacity of a large jar in liters -/
def large_jar_capacity : ℕ := 5

/-- Total number of jars -/
def total_jars : ℕ := 100

/-- Total capacity of all jars in liters -/
def total_capacity : ℕ := 376

theorem jar_problem :
  small_jars + large_jars = total_jars ∧
  small_jars * small_jar_capacity + large_jars * large_jar_capacity = total_capacity :=
by sorry

end NUMINAMATH_CALUDE_jar_problem_l1349_134928


namespace NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l1349_134990

def number_of_divisors (n : ℕ) : ℕ := sorry

def is_prime_factorization (n : ℕ) (factors : List (ℕ × ℕ)) : Prop := sorry

theorem smallest_number_with_2020_divisors :
  ∃ (n : ℕ) (factors : List (ℕ × ℕ)),
    is_prime_factorization n factors ∧
    number_of_divisors n = 2020 ∧
    (∀ m : ℕ, m < n → number_of_divisors m ≠ 2020) ∧
    factors = [(2, 100), (3, 4), (5, 1), (7, 1)] :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l1349_134990


namespace NUMINAMATH_CALUDE_tree_rings_l1349_134933

theorem tree_rings (thin_rings : ℕ) : 
  (∀ (fat_rings : ℕ), fat_rings = 2) →
  (70 * (fat_rings + thin_rings) = 40 * (fat_rings + thin_rings) + 180) →
  thin_rings = 4 := by
sorry

end NUMINAMATH_CALUDE_tree_rings_l1349_134933


namespace NUMINAMATH_CALUDE_cafeteria_earnings_l1349_134950

/-- Calculates the total earnings from selling apples and oranges in a cafeteria. -/
theorem cafeteria_earnings (initial_apples initial_oranges : ℕ)
                           (apple_price orange_price : ℚ)
                           (remaining_apples remaining_oranges : ℕ)
                           (h1 : initial_apples = 50)
                           (h2 : initial_oranges = 40)
                           (h3 : apple_price = 0.80)
                           (h4 : orange_price = 0.50)
                           (h5 : remaining_apples = 10)
                           (h6 : remaining_oranges = 6) :
  (initial_apples - remaining_apples) * apple_price +
  (initial_oranges - remaining_oranges) * orange_price = 49 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_earnings_l1349_134950


namespace NUMINAMATH_CALUDE_valid_assignment_d_plus_5_l1349_134947

/-- Represents a programming language variable --/
structure Variable where
  name : String

/-- Represents a programming language expression --/
inductive Expression where
  | Var : Variable → Expression
  | Const : Int → Expression
  | Add : Expression → Expression → Expression

/-- Represents an assignment statement --/
structure Assignment where
  lhs : Variable
  rhs : Expression

/-- Predicate to check if an assignment is valid --/
def is_valid_assignment (a : Assignment) : Prop :=
  ∃ (d : Variable), a.lhs = d ∧ 
    a.rhs = Expression.Add (Expression.Var d) (Expression.Const 5)

/-- Theorem stating that "d = d + 5" is a valid assignment --/
theorem valid_assignment_d_plus_5 :
  ∃ (a : Assignment), is_valid_assignment a :=
sorry

end NUMINAMATH_CALUDE_valid_assignment_d_plus_5_l1349_134947


namespace NUMINAMATH_CALUDE_optimal_profit_distribution_l1349_134963

/-- Represents the profit and production setup for handicrafts A and B --/
structure HandicraftSetup where
  profit_diff : ℝ  -- Profit difference between B and A
  profit_A_equal : ℝ  -- Profit of A when quantities are equal
  profit_B_equal : ℝ  -- Profit of B when quantities are equal
  total_workers : ℕ  -- Total number of workers
  A_production_rate : ℕ  -- Number of A pieces one worker can produce
  B_production_rate : ℕ  -- Number of B pieces one worker can produce
  min_B_production : ℕ  -- Minimum number of B pieces to be produced
  profit_decrease_rate : ℝ  -- Rate of profit decrease per extra B piece

/-- Calculates the maximum profit for the given handicraft setup --/
def max_profit (setup : HandicraftSetup) : ℝ :=
  let profit_A := setup.profit_A_equal * setup.profit_B_equal / (setup.profit_B_equal - setup.profit_diff)
  let profit_B := profit_A + setup.profit_diff
  let m := setup.total_workers / 2  -- Approximate midpoint for worker distribution
  (-2) * (m - 25)^2 + 3200

/-- Theorem stating the maximum profit and optimal worker distribution --/
theorem optimal_profit_distribution (setup : HandicraftSetup) :
  setup.profit_diff = 105 ∧
  setup.profit_A_equal = 30 ∧
  setup.profit_B_equal = 240 ∧
  setup.total_workers = 65 ∧
  setup.A_production_rate = 2 ∧
  setup.B_production_rate = 1 ∧
  setup.min_B_production = 5 ∧
  setup.profit_decrease_rate = 2 →
  max_profit setup = 3200 ∧
  ∃ (workers_A workers_B : ℕ),
    workers_A = 40 ∧
    workers_B = 25 ∧
    workers_A + workers_B = setup.total_workers :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_profit_distribution_l1349_134963


namespace NUMINAMATH_CALUDE_f_satisfies_data_points_l1349_134988

/-- The function f that we want to prove satisfies the given data points -/
def f (x : ℤ) : ℤ := 2 * x^2 + 2 * x - 1

/-- The list of data points given in the table -/
def data_points : List (ℤ × ℤ) := [(1, 3), (2, 11), (3, 23), (4, 39), (5, 59)]

/-- Theorem stating that the function f satisfies all the given data points -/
theorem f_satisfies_data_points : ∀ (point : ℤ × ℤ), point ∈ data_points → f point.1 = point.2 := by
  sorry

#check f_satisfies_data_points

end NUMINAMATH_CALUDE_f_satisfies_data_points_l1349_134988


namespace NUMINAMATH_CALUDE_bird_on_time_speed_l1349_134918

/-- Represents the problem of Mr. Bird's commute --/
structure BirdCommute where
  distance : ℝ
  time_on_time : ℝ
  speed_late : ℝ
  speed_early : ℝ
  late_time : ℝ
  early_time : ℝ

/-- The theorem stating the correct speed for Mr. Bird to arrive on time --/
theorem bird_on_time_speed (b : BirdCommute) 
  (h1 : b.speed_late = 30)
  (h2 : b.speed_early = 50)
  (h3 : b.late_time = 5 / 60)
  (h4 : b.early_time = 5 / 60)
  (h5 : b.distance = b.speed_late * (b.time_on_time + b.late_time))
  (h6 : b.distance = b.speed_early * (b.time_on_time - b.early_time)) :
  b.distance / b.time_on_time = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_bird_on_time_speed_l1349_134918


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_m_range_l1349_134931

theorem quadratic_distinct_roots_m_range :
  ∀ m : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + m*x₁ + (m + 3) = 0 ∧
    x₂^2 + m*x₂ + (m + 3) = 0) ↔
  m < -2 ∨ m > 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_m_range_l1349_134931


namespace NUMINAMATH_CALUDE_test_question_points_l1349_134934

theorem test_question_points : 
  ∀ (other_point_value : ℕ),
    (40 : ℕ) = 10 + (100 - 10 * 4) / other_point_value →
    other_point_value = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_test_question_points_l1349_134934
