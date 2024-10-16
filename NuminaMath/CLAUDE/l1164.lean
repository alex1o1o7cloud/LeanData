import Mathlib

namespace NUMINAMATH_CALUDE_cube_coverage_l1164_116413

/-- Represents a paper strip of size 3 × 1 -/
structure PaperStrip :=
  (length : Nat := 3)
  (width : Nat := 1)

/-- Represents a cube of size n × n × n -/
structure Cube (n : Nat) :=
  (side_length : Nat := n)

/-- Predicate to check if a number is divisible by 3 -/
def divisible_by_three (n : Nat) : Prop := n % 3 = 0

/-- Predicate to check if it's possible to cover three sides of a cube with paper strips -/
def can_cover_sides (c : Cube n) (p : PaperStrip) : Prop :=
  divisible_by_three n

/-- Theorem stating the condition for covering three sides of a cube with paper strips -/
theorem cube_coverage (n : Nat) :
  ∀ (c : Cube n) (p : PaperStrip),
    can_cover_sides c p ↔ divisible_by_three n :=
by sorry

end NUMINAMATH_CALUDE_cube_coverage_l1164_116413


namespace NUMINAMATH_CALUDE_broadway_show_attendance_l1164_116490

/-- The number of children attending a Broadway show -/
def num_children : ℕ := 200

/-- The number of adults attending the Broadway show -/
def num_adults : ℕ := 400

/-- The price of a child's ticket in dollars -/
def child_ticket_price : ℕ := 16

/-- The price of an adult ticket in dollars -/
def adult_ticket_price : ℕ := 32

/-- The total amount collected from ticket sales in dollars -/
def total_amount : ℕ := 16000

theorem broadway_show_attendance :
  num_children = 200 ∧
  num_adults = 400 ∧
  adult_ticket_price = 2 * child_ticket_price ∧
  adult_ticket_price = 32 ∧
  total_amount = num_adults * adult_ticket_price + num_children * child_ticket_price :=
by sorry

end NUMINAMATH_CALUDE_broadway_show_attendance_l1164_116490


namespace NUMINAMATH_CALUDE_ant_probability_l1164_116468

/-- Represents a vertex of a cube -/
inductive Vertex : Type
| A | B | C | D | E | F | G | H

/-- Represents the movement of an ant from one vertex to another -/
def Move : Type := Vertex → Vertex

/-- The set of all possible moves for all 8 ants -/
def AllMoves : Type := Fin 8 → Move

/-- Checks if a move is valid (i.e., to an adjacent vertex) -/
def isValidMove (m : Move) : Prop := sorry

/-- Checks if a set of moves results in no two ants on the same vertex -/
def noCollisions (moves : AllMoves) : Prop := sorry

/-- The total number of possible movement combinations -/
def totalMoves : ℕ := 3^8

/-- The number of valid movement combinations where no two ants collide -/
def validMoves : ℕ := 240

/-- The probability of no two ants arriving at the same vertex -/
theorem ant_probability : 
  (validMoves : ℚ) / totalMoves = 240 / 6561 := by sorry

end NUMINAMATH_CALUDE_ant_probability_l1164_116468


namespace NUMINAMATH_CALUDE_function_range_in_unit_interval_l1164_116457

theorem function_range_in_unit_interval (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ z : ℝ, 0 ≤ f z ∧ f z ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_in_unit_interval_l1164_116457


namespace NUMINAMATH_CALUDE_fraction_equality_l1164_116480

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - 4*b ≠ 0) (h4 : 4*a - b ≠ 0)
  (h5 : (4*a + 2*b) / (a - 4*b) = 3) : (a + 4*b) / (4*a - b) = 10/57 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1164_116480


namespace NUMINAMATH_CALUDE_vector_points_to_line_and_parallel_l1164_116493

/-- The line is parameterized by x = 3t + 1, y = t + 1 -/
def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 1, t + 1)

/-- The direction vector -/
def direction : ℝ × ℝ := (3, 1)

/-- The vector we want to prove -/
def vector : ℝ × ℝ := (9, 3)

theorem vector_points_to_line_and_parallel :
  (∃ t : ℝ, line_param t = vector) ∧ 
  (∃ k : ℝ, vector = (k * direction.1, k * direction.2)) :=
sorry

end NUMINAMATH_CALUDE_vector_points_to_line_and_parallel_l1164_116493


namespace NUMINAMATH_CALUDE_anniversary_day_theorem_l1164_116410

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of leap years in a 300-year span -/
def leapYearsIn300Years : Nat := 73

/-- Calculates the number of regular years in a 300-year span -/
def regularYearsIn300Years : Nat := 300 - leapYearsIn300Years

/-- Calculates the total days to move backward in 300 years -/
def totalDaysBackward : Nat :=
  regularYearsIn300Years + 2 * leapYearsIn300Years

/-- Theorem: If a 300th anniversary falls on a Thursday, the original date was a Tuesday -/
theorem anniversary_day_theorem (anniversaryDay : DayOfWeek) :
  anniversaryDay = DayOfWeek.Thursday →
  (totalDaysBackward % 7 : Nat) = 2 →
  ∃ (originalDay : DayOfWeek), originalDay = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_anniversary_day_theorem_l1164_116410


namespace NUMINAMATH_CALUDE_f_increasing_condition_f_extremum_at_3_f_max_value_f_min_value_l1164_116403

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - 3*x

-- Part 1: f(x) is increasing on [1, +∞) iff a ≤ 0
theorem f_increasing_condition (a : ℝ) :
  (∀ x ≥ 1, Monotone (f a)) ↔ a ≤ 0 := by sorry

-- Part 2: When x = 3 is an extremum point
theorem f_extremum_at_3 (a : ℝ) :
  (∃ x, HasDerivAt (f a) 0 x) → a = 6 := by sorry

-- Maximum value of f(x) on [1, 6] is -6
theorem f_max_value :
  ∃ x ∈ Set.Icc 1 6, ∀ y ∈ Set.Icc 1 6, f 6 y ≤ f 6 x ∧ f 6 x = -6 := by sorry

-- Minimum value of f(x) on [1, 6] is -18
theorem f_min_value :
  ∃ x ∈ Set.Icc 1 6, ∀ y ∈ Set.Icc 1 6, f 6 x ≤ f 6 y ∧ f 6 x = -18 := by sorry

end NUMINAMATH_CALUDE_f_increasing_condition_f_extremum_at_3_f_max_value_f_min_value_l1164_116403


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1164_116418

/-- Given an arithmetic sequence with first term 3, second term 10, third term 17, and sixth term 38,
    the sum of the fourth and fifth terms is 55. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  a 0 = 3 ∧ a 1 = 10 ∧ a 2 = 17 ∧ a 5 = 38 ∧
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →
  a 3 + a 4 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1164_116418


namespace NUMINAMATH_CALUDE_f_neg_a_eq_zero_l1164_116458

noncomputable def f (x : ℝ) : ℝ := x * Real.log (Real.exp (2 * x) + 1) - x^2 + 1

theorem f_neg_a_eq_zero (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_a_eq_zero_l1164_116458


namespace NUMINAMATH_CALUDE_bisection_interval_valid_l1164_116428

-- Define the function f(x) = x^3 + 5
def f (x : ℝ) : ℝ := x^3 + 5

-- Theorem statement
theorem bisection_interval_valid :
  f (-2) * f 1 < 0 := by sorry

end NUMINAMATH_CALUDE_bisection_interval_valid_l1164_116428


namespace NUMINAMATH_CALUDE_import_value_calculation_l1164_116424

/-- Given the export value and its relationship to the import value, 
    calculate the import value. -/
theorem import_value_calculation (export_value : ℝ) (import_value : ℝ) : 
  export_value = 8.07 ∧ 
  export_value = 1.5 * import_value + 1.11 → 
  sorry

end NUMINAMATH_CALUDE_import_value_calculation_l1164_116424


namespace NUMINAMATH_CALUDE_students_catching_up_on_homework_l1164_116464

theorem students_catching_up_on_homework (total : ℕ) (silent_reading : ℕ) (board_games : ℕ) :
  total = 24 →
  silent_reading = total / 2 →
  board_games = total / 3 →
  total - (silent_reading + board_games) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_students_catching_up_on_homework_l1164_116464


namespace NUMINAMATH_CALUDE_stating_remaining_pieces_l1164_116455

/-- The number of pieces on a standard chessboard at the start of the game. -/
def initial_pieces : ℕ := 32

/-- The number of pieces Audrey lost. -/
def audrey_lost : ℕ := 6

/-- The number of pieces Thomas lost. -/
def thomas_lost : ℕ := 5

/-- The total number of pieces lost by both players. -/
def total_lost : ℕ := audrey_lost + thomas_lost

/-- 
  Theorem stating that the number of pieces remaining on the chessboard is 21,
  given the initial number of pieces and the number of pieces lost by each player.
-/
theorem remaining_pieces :
  initial_pieces - total_lost = 21 := by sorry

end NUMINAMATH_CALUDE_stating_remaining_pieces_l1164_116455


namespace NUMINAMATH_CALUDE_remaining_money_l1164_116443

def octal_to_decimal (n : ℕ) : ℕ := sorry

def john_savings : ℕ := octal_to_decimal 5372

def ticket_cost : ℕ := 1200

theorem remaining_money :
  john_savings - ticket_cost = 1610 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l1164_116443


namespace NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l1164_116452

theorem power_of_eight_sum_equals_power_of_two : 8^17 + 8^17 + 8^17 + 8^17 = 2^53 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_sum_equals_power_of_two_l1164_116452


namespace NUMINAMATH_CALUDE_water_volume_cylindrical_tank_l1164_116491

/-- The volume of water in a cylindrical tank lying on its side -/
theorem water_volume_cylindrical_tank 
  (r : ℝ) -- radius of the tank
  (l : ℝ) -- length of the tank
  (d : ℝ) -- depth of water
  (h_r : r = 5) -- radius is 5 feet
  (h_l : l = 10) -- length is 10 feet
  (h_d : d = 4) -- depth of water is 4 feet
  : ∃ (volume : ℝ), volume = 343 * Real.pi - 20 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_water_volume_cylindrical_tank_l1164_116491


namespace NUMINAMATH_CALUDE_min_largest_group_size_l1164_116429

theorem min_largest_group_size (total_boxes : ℕ) (min_apples max_apples : ℕ) : 
  total_boxes = 128 →
  min_apples = 120 →
  max_apples = 144 →
  ∃ (n : ℕ), n = 6 ∧ 
    (∀ (group_size : ℕ), 
      (group_size * (max_apples - min_apples + 1) ≥ total_boxes → group_size ≥ n) ∧
      (∃ (distribution : List ℕ), 
        distribution.length = max_apples - min_apples + 1 ∧
        distribution.sum = total_boxes ∧
        ∀ (x : ℕ), x ∈ distribution → x ≤ n)) :=
by sorry

end NUMINAMATH_CALUDE_min_largest_group_size_l1164_116429


namespace NUMINAMATH_CALUDE_class_b_wins_l1164_116441

/-- Represents the grades in a class --/
structure ClassGrades where
  excellent : ℕ
  good : ℕ
  average : ℕ
  satisfactory : ℕ

/-- Calculates the average grade for a class --/
def averageGrade (cg : ClassGrades) (totalStudents : ℕ) : ℚ :=
  (5 * cg.excellent + 4 * cg.good + 3 * cg.average + 2 * cg.satisfactory) / totalStudents

theorem class_b_wins (classA classB : ClassGrades) : 
  classA.excellent = 6 ∧
  classA.good = 16 ∧
  classA.average = 10 ∧
  classA.satisfactory = 8 ∧
  classB.excellent = 5 ∧
  classB.good = 15 ∧
  classB.average = 15 ∧
  classB.satisfactory = 3 →
  averageGrade classB 38 > averageGrade classA 40 := by
  sorry

#eval averageGrade ⟨6, 16, 10, 8⟩ 40
#eval averageGrade ⟨5, 15, 15, 3⟩ 38

end NUMINAMATH_CALUDE_class_b_wins_l1164_116441


namespace NUMINAMATH_CALUDE_insufficient_apples_l1164_116449

def apples_picked : ℕ := 150
def num_children : ℕ := 4
def apples_per_child_per_day : ℕ := 12
def days_in_week : ℕ := 7
def apples_per_pie : ℕ := 12
def num_pies : ℕ := 2
def apples_per_salad : ℕ := 15
def salads_per_week : ℕ := 2
def apples_taken_by_sister : ℕ := 5

theorem insufficient_apples :
  apples_picked < 
    (num_children * apples_per_child_per_day * days_in_week) +
    (num_pies * apples_per_pie) +
    (apples_per_salad * salads_per_week) +
    apples_taken_by_sister := by
  sorry

end NUMINAMATH_CALUDE_insufficient_apples_l1164_116449


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l1164_116495

theorem max_digits_product_5_4 : ∀ a b : ℕ, 
  10000 ≤ a ∧ a < 100000 → 1000 ≤ b ∧ b < 10000 → 
  a * b < 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l1164_116495


namespace NUMINAMATH_CALUDE_machine_does_not_require_repair_no_repair_needed_l1164_116459

/-- Represents the nominal portion weight in grams -/
def nominal_weight : ℝ := 390

/-- Represents the greatest deviation from the mean among preserved measurements in grams -/
def max_deviation : ℝ := 39

/-- Represents the threshold for requiring repair in grams -/
def repair_threshold : ℝ := 39

/-- Condition: The greatest deviation does not exceed 10% of the nominal weight -/
axiom max_deviation_condition : max_deviation ≤ 0.1 * nominal_weight

/-- Condition: All deviations are no more than the maximum deviation -/
axiom all_deviations_bounded (deviation : ℝ) : deviation ≤ max_deviation

/-- Condition: The standard deviation does not exceed the greatest deviation -/
axiom standard_deviation_bounded (σ : ℝ) : σ ≤ max_deviation

/-- Theorem: The standard deviation is no more than the repair threshold -/
theorem machine_does_not_require_repair (σ : ℝ) : 
  σ ≤ repair_threshold :=
sorry

/-- Corollary: The machine does not require repair -/
theorem no_repair_needed : 
  ∃ (σ : ℝ), σ ≤ repair_threshold :=
sorry

end NUMINAMATH_CALUDE_machine_does_not_require_repair_no_repair_needed_l1164_116459


namespace NUMINAMATH_CALUDE_puzzle_solution_l1164_116442

theorem puzzle_solution (A B C : ℤ) 
  (eq1 : A + C = 10)
  (eq2 : A + B + 1 = C + 10)
  (eq3 : A + 1 = B) :
  A = 6 ∧ B = 7 ∧ C = 4 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1164_116442


namespace NUMINAMATH_CALUDE_soccer_season_length_l1164_116487

theorem soccer_season_length (total_games : ℕ) (games_per_month : ℕ) (h1 : total_games = 27) (h2 : games_per_month = 9) :
  total_games / games_per_month = 3 := by
  sorry

end NUMINAMATH_CALUDE_soccer_season_length_l1164_116487


namespace NUMINAMATH_CALUDE_alternating_sequence_property_l1164_116438

def alternatingSequence (n : ℕ) : ℤ := (-1) ^ (n + 1)

theorem alternating_sequence_property : ∀ n : ℕ, 
  (alternatingSequence n = 1 ∧ alternatingSequence (n + 1) = -1) ∨
  (alternatingSequence n = -1 ∧ alternatingSequence (n + 1) = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_alternating_sequence_property_l1164_116438


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1164_116420

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ x : ℝ, (1 + x > a ∧ 2 * x - 4 ≤ 0)) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1164_116420


namespace NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l1164_116499

def num_candidates : ℕ := 5
def num_chosen : ℕ := 3

theorem probability_of_selecting_A_and_B :
  let total_combinations := Nat.choose num_candidates num_chosen
  let combinations_with_A_and_B := Nat.choose (num_candidates - 2) (num_chosen - 2)
  (combinations_with_A_and_B : ℚ) / total_combinations = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l1164_116499


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1164_116404

def A : Set ℕ := {70, 1946, 1997, 2003}
def B : Set ℕ := {1, 10, 70, 2016}

theorem intersection_of_A_and_B : A ∩ B = {70} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1164_116404


namespace NUMINAMATH_CALUDE_potato_ratio_l1164_116454

def potato_distribution (initial : ℕ) (gina : ℕ) (remaining : ℕ) : Prop :=
  ∃ (tom anne : ℕ),
    tom = 2 * gina ∧
    initial = gina + tom + anne + remaining ∧
    anne * 3 = tom

theorem potato_ratio (initial : ℕ) (gina : ℕ) (remaining : ℕ) 
  (h : potato_distribution initial gina remaining) :
  potato_distribution 300 69 47 :=
by sorry

end NUMINAMATH_CALUDE_potato_ratio_l1164_116454


namespace NUMINAMATH_CALUDE_cube_from_wire_l1164_116476

/-- Given a wire of length 60 cm formed into a cube frame, prove that the volume is 125 cm³ and the surface area is 150 cm². -/
theorem cube_from_wire (wire_length : ℝ) (h_wire : wire_length = 60) :
  let edge_length : ℝ := wire_length / 12
  let volume : ℝ := edge_length ^ 3
  let surface_area : ℝ := 6 * edge_length ^ 2
  volume = 125 ∧ surface_area = 150 := by sorry

end NUMINAMATH_CALUDE_cube_from_wire_l1164_116476


namespace NUMINAMATH_CALUDE_stanley_lemonade_sales_l1164_116416

/-- The number of cups of lemonade Carl sells per hour -/
def carl_cups_per_hour : ℕ := 7

/-- The number of hours considered -/
def hours : ℕ := 3

/-- The difference in cups sold between Carl and Stanley over 3 hours -/
def difference_in_cups : ℕ := 9

/-- The number of cups of lemonade Stanley sells per hour -/
def stanley_cups_per_hour : ℕ := 4

theorem stanley_lemonade_sales :
  stanley_cups_per_hour * hours + difference_in_cups = carl_cups_per_hour * hours := by
  sorry

end NUMINAMATH_CALUDE_stanley_lemonade_sales_l1164_116416


namespace NUMINAMATH_CALUDE_sum_of_k_values_l1164_116408

theorem sum_of_k_values (a b c k : ℂ) : 
  a ≠ b ∧ b ≠ c ∧ c ≠ a →
  (a + 1) / (2 - b) = k ∧
  (b + 1) / (2 - c) = k ∧
  (c + 1) / (2 - a) = k →
  ∃ k₁ k₂ : ℂ, k = k₁ ∨ k = k₂ ∧ k₁ + k₂ = (3/2 : ℂ) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_k_values_l1164_116408


namespace NUMINAMATH_CALUDE_smallest_integer_with_16_divisors_l1164_116492

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a given positive integer has exactly 16 positive divisors -/
def has_16_divisors (n : ℕ+) : Prop := num_divisors n = 16

theorem smallest_integer_with_16_divisors :
  (∃ (n : ℕ+), has_16_divisors n) ∧
  (∀ (m : ℕ+), has_16_divisors m → 384 ≤ m) ∧
  has_16_divisors 384 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_16_divisors_l1164_116492


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1164_116467

theorem max_value_of_expression (y : ℝ) :
  (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) ≤ 15 ∧
  ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1164_116467


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitivity_l1164_116483

-- Define the types for lines and planes
def Line : Type := Real × Real × Real → Prop
def Plane : Type := Real × Real × Real → Prop

-- Define the relations
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem perpendicular_parallel_transitivity 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perpendicular_line_plane m α) 
  (h3 : parallel m n) : 
  perpendicular_line_plane n α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitivity_l1164_116483


namespace NUMINAMATH_CALUDE_power_relation_l1164_116465

theorem power_relation (a : ℝ) (b : ℝ) (h : a ^ b = 1 / 8) : a ^ (-3 * b) = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l1164_116465


namespace NUMINAMATH_CALUDE_triangle_inequality_l1164_116498

theorem triangle_inequality (a b c : ℝ) (h_area : (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = 1/4) (h_circumradius : (a * b * c) / (4 * (1/4)) = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1/a + 1/b + 1/c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1164_116498


namespace NUMINAMATH_CALUDE_cylinder_volume_in_sphere_l1164_116406

-- Define the sphere and cylinder
def sphere_diameter : ℝ := 2
def cylinder_height : ℝ := 1

-- Theorem statement
theorem cylinder_volume_in_sphere :
  let sphere_radius : ℝ := sphere_diameter / 2
  let cylinder_base_radius : ℝ := sphere_radius
  let cylinder_volume : ℝ := π * cylinder_base_radius^2 * cylinder_height / 2
  cylinder_volume = π / 2 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_in_sphere_l1164_116406


namespace NUMINAMATH_CALUDE_teresa_jogging_time_l1164_116460

-- Define the constants
def distance : ℝ := 45  -- kilometers
def speed : ℝ := 7      -- kilometers per hour
def break_time : ℝ := 0.5  -- hours (30 minutes)

-- Define the theorem
theorem teresa_jogging_time :
  let jogging_time := distance / speed
  let total_time := jogging_time + break_time
  total_time = 6.93 :=
by
  sorry


end NUMINAMATH_CALUDE_teresa_jogging_time_l1164_116460


namespace NUMINAMATH_CALUDE_orange_painted_cubes_l1164_116430

/-- Represents a cube construction with small cubes -/
structure CubeConstruction where
  small_edge : ℝ
  large_edge : ℝ
  all_sides_painted : Bool

/-- Calculates the number of small cubes with only one side painted -/
def cubes_with_one_side_painted (c : CubeConstruction) : ℕ :=
  sorry

/-- Theorem stating the number of small cubes with one side painted in the given construction -/
theorem orange_painted_cubes (c : CubeConstruction) 
  (h1 : c.small_edge = 2)
  (h2 : c.large_edge = 10)
  (h3 : c.all_sides_painted = true) :
  cubes_with_one_side_painted c = 54 := by
  sorry

end NUMINAMATH_CALUDE_orange_painted_cubes_l1164_116430


namespace NUMINAMATH_CALUDE_divisibility_proof_l1164_116488

theorem divisibility_proof (a b c : ℝ) 
  (h : (a ≠ 0 ∧ b ≠ 0) ∨ (a ≠ 0 ∧ c ≠ 0) ∨ (b ≠ 0 ∧ c ≠ 0)) :
  ∃ k : ℤ, (a + b + c)^7 - a^7 - b^7 - c^7 = k * (7 * (a + b) * (b + c) * (c + a)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_proof_l1164_116488


namespace NUMINAMATH_CALUDE_balance_after_transfer_l1164_116440

/-- The initial balance in Christina's bank account before the transfer -/
def initial_balance : ℕ := 27004

/-- The amount Christina transferred out of her account -/
def transferred_amount : ℕ := 69

/-- The remaining balance in Christina's account after the transfer -/
def remaining_balance : ℕ := 26935

/-- Theorem stating that the initial balance minus the transferred amount equals the remaining balance -/
theorem balance_after_transfer : 
  initial_balance - transferred_amount = remaining_balance := by sorry

end NUMINAMATH_CALUDE_balance_after_transfer_l1164_116440


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l1164_116450

theorem sum_of_cubes_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : a / (2 * (b - c)) + b / (2 * (c - a)) + c / (2 * (a - b)) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l1164_116450


namespace NUMINAMATH_CALUDE_meadow_orders_30_boxes_l1164_116485

/-- Represents Meadow's diaper business --/
structure DiaperBusiness where
  packs_per_box : ℕ
  diapers_per_pack : ℕ
  price_per_diaper : ℕ
  total_revenue : ℕ

/-- Calculates the number of boxes ordered weekly --/
def boxes_ordered (business : DiaperBusiness) : ℕ :=
  business.total_revenue / (business.price_per_diaper * business.diapers_per_pack * business.packs_per_box)

/-- Theorem: Given the conditions, Meadow orders 30 boxes weekly --/
theorem meadow_orders_30_boxes :
  let business : DiaperBusiness := {
    packs_per_box := 40,
    diapers_per_pack := 160,
    price_per_diaper := 5,
    total_revenue := 960000
  }
  boxes_ordered business = 30 := by
  sorry

end NUMINAMATH_CALUDE_meadow_orders_30_boxes_l1164_116485


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l1164_116463

theorem christmas_tree_lights (total : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : total = 95)
  (h2 : yellow = 37)
  (h3 : blue = 32) :
  total - (yellow + blue) = 26 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l1164_116463


namespace NUMINAMATH_CALUDE_cricket_match_average_l1164_116419

/-- Given five cricket match scores x, y, a, b, and c, prove that their average is 36 under certain conditions. -/
theorem cricket_match_average (x y a b c : ℝ) 
  (avg_first_two : (x + y) / 2 = 30)
  (avg_last_three : (a + b + c) / 3 = 40)
  (max_score : x ≤ 60 ∧ y ≤ 60 ∧ a ≤ 60 ∧ b ≤ 60 ∧ c ≤ 60)
  (century_condition : x + y ≥ 100 ∨ a + b + c ≥ 100) :
  (x + y + a + b + c) / 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cricket_match_average_l1164_116419


namespace NUMINAMATH_CALUDE_root_product_sum_l1164_116478

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧ 
  (Real.sqrt 1008) * x₁^3 - 2016 * x₁^2 + 5 * x₁ + 2 = 0 ∧
  (Real.sqrt 1008) * x₂^3 - 2016 * x₂^2 + 5 * x₂ + 2 = 0 ∧
  (Real.sqrt 1008) * x₃^3 - 2016 * x₃^2 + 5 * x₃ + 2 = 0 →
  x₂ * (x₁ + x₃) = 1010 / 1008 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l1164_116478


namespace NUMINAMATH_CALUDE_power_equality_comparisons_l1164_116489

theorem power_equality_comparisons :
  (-2^3 = (-2)^3) ∧
  (3^2 ≠ 2^3) ∧
  (-3^2 ≠ (-3)^2) ∧
  (-(3 * 2)^2 ≠ -3 * 2^2) := by sorry

end NUMINAMATH_CALUDE_power_equality_comparisons_l1164_116489


namespace NUMINAMATH_CALUDE_pentagon_diagonals_from_vertex_l1164_116462

/-- The number of diagonals that can be drawn from a vertex of an n-sided polygon. -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A pentagon has 5 sides. -/
def pentagon_sides : ℕ := 5

theorem pentagon_diagonals_from_vertex :
  diagonals_from_vertex pentagon_sides = 2 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_diagonals_from_vertex_l1164_116462


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1164_116433

-- Define a geometric sequence
def is_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ q : ℝ, y = x * q ∧ z = y * q

-- Define the problem statement
theorem geometric_sequence_problem (a b c d : ℝ) 
  (h : is_geometric_sequence a c d) :
  (is_geometric_sequence (a*b) (b+c) (c+d) ∨
   is_geometric_sequence (a*b) (b*c) (c*d) ∨
   is_geometric_sequence (a*b) (b-c) (-d)) ∧
  ¬(is_geometric_sequence (a*b) (b+c) (c+d) ∧
    is_geometric_sequence (a*b) (b*c) (c*d)) ∧
  ¬(is_geometric_sequence (a*b) (b+c) (c+d) ∧
    is_geometric_sequence (a*b) (b-c) (-d)) ∧
  ¬(is_geometric_sequence (a*b) (b*c) (c*d) ∧
    is_geometric_sequence (a*b) (b-c) (-d)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1164_116433


namespace NUMINAMATH_CALUDE_value_of_y_l1164_116411

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 20) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l1164_116411


namespace NUMINAMATH_CALUDE_total_tea_gallons_l1164_116402

-- Define the number of containers
def num_containers : ℕ := 80

-- Define the relationship between containers and pints
def containers_to_pints : ℚ := 7 / (7/2)

-- Define the conversion rate from pints to gallons
def pints_per_gallon : ℕ := 8

-- Theorem stating the total amount of tea in gallons
theorem total_tea_gallons : 
  (↑num_containers * containers_to_pints) / ↑pints_per_gallon = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_tea_gallons_l1164_116402


namespace NUMINAMATH_CALUDE_square_sum_plus_sum_squares_l1164_116426

theorem square_sum_plus_sum_squares : (5 + 9)^2 + (5^2 + 9^2) = 302 := by sorry

end NUMINAMATH_CALUDE_square_sum_plus_sum_squares_l1164_116426


namespace NUMINAMATH_CALUDE_parallel_lines_equal_angles_with_plane_l1164_116448

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the relation for a line forming an angle with a plane
variable (angle_with_plane : Line → Plane → ℝ)

-- State the theorem
theorem parallel_lines_equal_angles_with_plane
  (m n : Line) (α : Plane) :
  (parallel m n → angle_with_plane m α = angle_with_plane n α) ∧
  ¬(angle_with_plane m α = angle_with_plane n α → parallel m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_equal_angles_with_plane_l1164_116448


namespace NUMINAMATH_CALUDE_balanced_polynomial_existence_balanced_polynomial_equality_l1164_116486

-- Define what it means for an integer to be balanced
def IsBalanced (n : ℤ) : Prop :=
  n = 1 ∨ ∃ (k : ℕ) (p : List ℤ), k % 2 = 0 ∧ n = p.prod ∧ ∀ x ∈ p, Nat.Prime x.natAbs

-- Define the polynomial P(x) = (x+a)(x+b)
def P (a b : ℤ) (x : ℤ) : ℤ := (x + a) * (x + b)

theorem balanced_polynomial_existence :
  ∃ (a b : ℤ), a ≠ b ∧ a > 0 ∧ b > 0 ∧ ∀ n : ℤ, 1 ≤ n ∧ n ≤ 50 → IsBalanced (P a b n) :=
sorry

theorem balanced_polynomial_equality (a b : ℤ) (h : ∀ n : ℤ, IsBalanced (P a b n)) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_balanced_polynomial_existence_balanced_polynomial_equality_l1164_116486


namespace NUMINAMATH_CALUDE_fraction_comparison_l1164_116447

theorem fraction_comparison : ((3 / 5 : ℚ) * 320 + (5 / 9 : ℚ) * 540) - ((7 / 12 : ℚ) * 450) = 229.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1164_116447


namespace NUMINAMATH_CALUDE_value_of_a_l1164_116427

def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem value_of_a : ∀ a : ℝ, A a ∪ B a = {0, 1, 2, 4, 16} → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1164_116427


namespace NUMINAMATH_CALUDE_random_events_l1164_116405

-- Define the type for events
inductive Event
  | CoinToss
  | ChargeAttraction
  | WaterFreezing
  | DiceRoll

-- Define a function to check if an event is random
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.CoinToss => true
  | Event.ChargeAttraction => false
  | Event.WaterFreezing => false
  | Event.DiceRoll => true

-- Theorem stating which events are random
theorem random_events :
  (isRandomEvent Event.CoinToss) ∧
  (¬isRandomEvent Event.ChargeAttraction) ∧
  (¬isRandomEvent Event.WaterFreezing) ∧
  (isRandomEvent Event.DiceRoll) := by
  sorry

#check random_events

end NUMINAMATH_CALUDE_random_events_l1164_116405


namespace NUMINAMATH_CALUDE_special_rectangle_area_l1164_116482

/-- A rectangle ABCD with specific properties -/
structure SpecialRectangle where
  -- AB, BC, CD are sides of the rectangle
  AB : ℝ
  BC : ℝ
  CD : ℝ
  -- E is the midpoint of BC
  BE : ℝ
  -- Conditions
  rectangle_condition : AB = CD
  perimeter_condition : AB + BC + CD = 20
  midpoint_condition : BE = BC / 2
  diagonal_condition : AB^2 + BE^2 = 9^2

/-- The area of a SpecialRectangle is 19 -/
theorem special_rectangle_area (r : SpecialRectangle) : r.AB * r.BC = 19 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_l1164_116482


namespace NUMINAMATH_CALUDE_max_k_value_l1164_116439

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8 = 0

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the condition for a point to be a valid center
def valid_center (k : ℝ) (x y : ℝ) : Prop :=
  line k x y ∧ ∃ (x' y' : ℝ), circle_C x' y' ∧ (x - x')^2 + (y - y')^2 = 1

-- Theorem statement
theorem max_k_value :
  (∃ k : ℝ, ∀ k' : ℝ, (∃ x y : ℝ, valid_center k' x y) → k' ≤ k) ∧
  (∃ x y : ℝ, valid_center (12/5) x y) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1164_116439


namespace NUMINAMATH_CALUDE_g_pow_6_eq_id_l1164_116432

/-- Definition of the function g -/
def g (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, c) := v
  (a + b, b + c, a + c)

/-- Definition of g^n for n ≥ 2 -/
def g_pow (n : ℕ) : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) :=
  match n with
  | 0 => id
  | 1 => g
  | n + 2 => g ∘ (g_pow (n + 1))

/-- Main theorem -/
theorem g_pow_6_eq_id (v : ℝ × ℝ × ℝ) (h1 : v ≠ (0, 0, 0)) 
    (h2 : ∃ (n : ℕ+), g_pow n v = v) : 
  g_pow 6 v = v := by
  sorry

end NUMINAMATH_CALUDE_g_pow_6_eq_id_l1164_116432


namespace NUMINAMATH_CALUDE_heidi_has_five_more_than_kim_l1164_116446

/-- The number of nail polishes each person has -/
structure NailPolishes where
  kim : ℕ
  heidi : ℕ
  karen : ℕ

/-- The conditions of the nail polish problem -/
def nail_polish_problem (np : NailPolishes) : Prop :=
  np.kim = 12 ∧
  np.heidi > np.kim ∧
  np.karen = np.kim - 4 ∧
  np.karen + np.heidi = 25

/-- The theorem stating that Heidi has 5 more nail polishes than Kim -/
theorem heidi_has_five_more_than_kim (np : NailPolishes) 
  (h : nail_polish_problem np) : np.heidi - np.kim = 5 := by
  sorry

end NUMINAMATH_CALUDE_heidi_has_five_more_than_kim_l1164_116446


namespace NUMINAMATH_CALUDE_quadratic_form_h_l1164_116436

theorem quadratic_form_h (a k : ℝ) (h : ℝ) :
  (∀ x, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) →
  h = -3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_h_l1164_116436


namespace NUMINAMATH_CALUDE_fraction_problem_l1164_116456

theorem fraction_problem (a b m : ℚ) : 
  (2 * (1/2) - b = 0) →  -- Fraction is undefined when x = 0.5
  ((-2 + a) / (2 * (-2) - b) = 0) →  -- Fraction equals 0 when x = -2
  ((m + a) / (2 * m - b) = 1) →  -- Fraction equals 1 when x = m
  m = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1164_116456


namespace NUMINAMATH_CALUDE_mayo_bottle_size_l1164_116484

/-- Proves the size of a mayo bottle at a normal store given bulk pricing information -/
theorem mayo_bottle_size 
  (costco_price : ℝ) 
  (normal_store_price : ℝ) 
  (savings : ℝ) 
  (gallon_in_ounces : ℝ) 
  (h1 : costco_price = 8) 
  (h2 : normal_store_price = 3) 
  (h3 : savings = 16) 
  (h4 : gallon_in_ounces = 128) : 
  (gallon_in_ounces / ((savings + costco_price) / normal_store_price)) = 16 :=
by sorry

end NUMINAMATH_CALUDE_mayo_bottle_size_l1164_116484


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_1000_l1164_116421

def hulk_jump (n : ℕ) : ℝ := 3^n

theorem hulk_jump_exceeds_1000 : 
  (∀ k < 7, hulk_jump k ≤ 1000) ∧ hulk_jump 7 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_1000_l1164_116421


namespace NUMINAMATH_CALUDE_min_squared_distance_to_origin_l1164_116400

theorem min_squared_distance_to_origin (x y : ℝ) : 
  (x + 5)^2 + (y - 12)^2 = 14^2 → 
  ∃ (min : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2) ∧ min = 1 :=
sorry

end NUMINAMATH_CALUDE_min_squared_distance_to_origin_l1164_116400


namespace NUMINAMATH_CALUDE_owen_june_burger_expense_l1164_116435

/-- The amount Owen spent on burgers in June -/
def owen_burger_expense (burgers_per_day : ℕ) (burger_cost : ℕ) (days_in_june : ℕ) : ℕ :=
  burgers_per_day * days_in_june * burger_cost

/-- Theorem stating that Owen's burger expense in June is 720 dollars -/
theorem owen_june_burger_expense :
  owen_burger_expense 2 12 30 = 720 :=
by sorry

end NUMINAMATH_CALUDE_owen_june_burger_expense_l1164_116435


namespace NUMINAMATH_CALUDE_no_single_common_tangent_for_equal_circles_l1164_116422

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a function to count common tangents between two circles
def countCommonTangents (c1 c2 : Circle) : ℕ := sorry

-- Theorem statement
theorem no_single_common_tangent_for_equal_circles (c1 c2 : Circle) :
  c1.radius = c2.radius → c1 ≠ c2 → countCommonTangents c1 c2 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_single_common_tangent_for_equal_circles_l1164_116422


namespace NUMINAMATH_CALUDE_square_of_negative_two_times_a_cubed_l1164_116473

theorem square_of_negative_two_times_a_cubed (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_times_a_cubed_l1164_116473


namespace NUMINAMATH_CALUDE_sum_of_squares_of_divisors_1800_l1164_116469

def sumOfSquaresOfDivisors (n : ℕ) : ℕ := sorry

theorem sum_of_squares_of_divisors_1800 :
  sumOfSquaresOfDivisors 1800 = 5035485 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_divisors_1800_l1164_116469


namespace NUMINAMATH_CALUDE_consecutive_integer_sum_l1164_116466

theorem consecutive_integer_sum (n : ℕ) :
  (∃ k : ℤ, (k - 2) + (k - 1) + k + (k + 1) + (k + 2) = n) ∧
  (¬ ∃ m : ℤ, (m - 1) + m + (m + 1) + (m + 2) = n) :=
by
  sorry

#check consecutive_integer_sum 225

end NUMINAMATH_CALUDE_consecutive_integer_sum_l1164_116466


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1164_116445

theorem max_value_polynomial (a b : ℝ) (h : a^2 + 4*b^2 = 4) :
  ∃ M : ℝ, M = 16 ∧ ∀ x y : ℝ, x^2 + 4*y^2 = 4 → 3*x^5*y - 40*x^3*y^3 + 48*x*y^5 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1164_116445


namespace NUMINAMATH_CALUDE_f_even_and_decreasing_l1164_116417

def f (x : ℝ) := -x^2 + 1

theorem f_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_decreasing_l1164_116417


namespace NUMINAMATH_CALUDE_daily_profit_properties_l1164_116415

/-- Represents the daily sales profit function for a company -/
def daily_profit (x : ℝ) : ℝ := 10 * x^2 - 80 * x

/-- Theorem stating the properties of the daily sales profit function -/
theorem daily_profit_properties :
  -- The daily profit function is correct
  (∀ x, daily_profit x = 10 * x^2 - 80 * x) ∧
  -- When the selling price increases by 3 yuan, the daily profit is 350 yuan
  (daily_profit 3 = 350) ∧
  -- When the daily profit is 360 yuan, the selling price has increased by 4 yuan
  (daily_profit 4 = 360) := by
  sorry


end NUMINAMATH_CALUDE_daily_profit_properties_l1164_116415


namespace NUMINAMATH_CALUDE_lindas_coins_l1164_116414

/-- Represents the number of coins Linda has initially -/
structure InitialCoins where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- Represents the number of coins Linda's mother gives her -/
structure AdditionalCoins where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- The problem statement -/
theorem lindas_coins (initial : InitialCoins) (additional : AdditionalCoins) 
    (h1 : initial.quarters = 6)
    (h2 : initial.nickels = 5)
    (h3 : additional.dimes = 2)
    (h4 : additional.quarters = 10)
    (h5 : additional.nickels = 2 * initial.nickels)
    (h6 : initial.dimes + initial.quarters + initial.nickels + 
          additional.dimes + additional.quarters + additional.nickels = 35) :
    initial.dimes = 4 := by
  sorry


end NUMINAMATH_CALUDE_lindas_coins_l1164_116414


namespace NUMINAMATH_CALUDE_three_circles_sum_l1164_116451

theorem three_circles_sum (triangle circle : ℚ) 
  (eq1 : 5 * triangle + 2 * circle = 27)
  (eq2 : 2 * triangle + 5 * circle = 29) :
  3 * circle = 13 := by
sorry

end NUMINAMATH_CALUDE_three_circles_sum_l1164_116451


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1164_116479

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 4 * p - 7 = 0) → 
  (3 * q^2 + 4 * q - 7 = 0) → 
  (p - 2) * (q - 2) = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1164_116479


namespace NUMINAMATH_CALUDE_lines_without_common_point_are_parallel_or_skew_l1164_116496

-- Define a type for straight lines in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- or any other suitable representation
  -- This is just a placeholder structure
  mk :: (dummy : Unit)

-- Define the property of two lines not having a common point
def noCommonPoint (a b : Line3D) : Prop :=
  -- The actual implementation would depend on how you define Line3D
  sorry

-- Define the property of two lines being parallel
def parallel (a b : Line3D) : Prop :=
  -- The actual implementation would depend on how you define Line3D
  sorry

-- Define the property of two lines being skew
def skew (a b : Line3D) : Prop :=
  -- The actual implementation would depend on how you define Line3D
  sorry

-- The theorem statement
theorem lines_without_common_point_are_parallel_or_skew 
  (a b : Line3D) (h : noCommonPoint a b) : 
  parallel a b ∨ skew a b :=
sorry

end NUMINAMATH_CALUDE_lines_without_common_point_are_parallel_or_skew_l1164_116496


namespace NUMINAMATH_CALUDE_absolute_value_equation_implies_power_l1164_116434

theorem absolute_value_equation_implies_power (x : ℝ) :
  |x| = 3 * x + 1 → (4 * x + 2)^2005 = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_implies_power_l1164_116434


namespace NUMINAMATH_CALUDE_inequality_proof_l1164_116481

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * (a + b)) + Real.sqrt (b * c * (b + c)) + Real.sqrt (c * a * (c + a)) >
  Real.sqrt ((a + b) * (b + c) * (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1164_116481


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l1164_116444

-- Define what it means for an angle to be in the third quadrant
def in_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, k * 2 * Real.pi + Real.pi < α ∧ α < k * 2 * Real.pi + 3 * Real.pi / 2

-- Define what it means for an angle to be in the second or fourth quadrant
def in_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, (k * Real.pi + Real.pi / 2 < α ∧ α < k * Real.pi + Real.pi) ∨
             (k * Real.pi + 3 * Real.pi / 2 < α ∧ α < (k + 1) * Real.pi)

-- State the theorem
theorem half_angle_quadrant (α : Real) :
  in_third_quadrant α → in_second_or_fourth_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l1164_116444


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l1164_116494

/-- If 55 cows eat 55 bags of husk in 55 days, then one cow will eat one bag of husk in 55 days -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 55 ∧ bags = 55 ∧ days = 55) :
  (1 : ℕ) * bags = (1 : ℕ) * cows * days := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l1164_116494


namespace NUMINAMATH_CALUDE_feed_has_greatest_value_l1164_116471

/-- The value of a letter in the alphabet (A to F) -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | _ => 0

/-- The value of a word, which is the sum of its letter values -/
def word_value (w : String) : ℕ :=
  w.data.map letter_value |>.sum

/-- The list of words to compare -/
def words : List String := ["BEEF", "FADE", "FEED", "FACE", "DEAF"]

theorem feed_has_greatest_value :
  ∀ w ∈ words, word_value "FEED" ≥ word_value w :=
by sorry

end NUMINAMATH_CALUDE_feed_has_greatest_value_l1164_116471


namespace NUMINAMATH_CALUDE_calories_burned_proof_l1164_116477

/-- The number of times players run up and down the bleachers -/
def num_runs : ℕ := 40

/-- The number of stairs climbed in one direction -/
def stairs_one_way : ℕ := 32

/-- The number of calories burned per stair -/
def calories_per_stair : ℕ := 2

/-- Calculates the total calories burned during the exercise -/
def total_calories_burned : ℕ :=
  num_runs * (2 * stairs_one_way) * calories_per_stair

/-- Theorem stating that the total calories burned is 5120 -/
theorem calories_burned_proof : total_calories_burned = 5120 := by
  sorry

end NUMINAMATH_CALUDE_calories_burned_proof_l1164_116477


namespace NUMINAMATH_CALUDE_exists_left_identity_element_l1164_116431

variable {T : Type*} [Fintype T]

def LeftIdentityElement (star : T → T → T) (a : T) : Prop :=
  ∀ b : T, star a b = a

theorem exists_left_identity_element
  (star : T → T → T)
  (assoc : ∀ a b c : T, star (star a b) c = star a (star b c))
  (comm : ∀ a b : T, star a b = star b a) :
  ∃ a : T, LeftIdentityElement star a :=
by
  sorry

end NUMINAMATH_CALUDE_exists_left_identity_element_l1164_116431


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l1164_116470

theorem binomial_expansion_example : 
  (0.5 : ℝ)^3 + 3 * (0.5 : ℝ)^2 * (-1.5) + 3 * (0.5 : ℝ) * (-1.5)^2 + (-1.5)^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l1164_116470


namespace NUMINAMATH_CALUDE_smallest_q_for_inequality_l1164_116497

theorem smallest_q_for_inequality : ∃ (q : ℕ+), 
  (q = 2015) ∧ 
  (∀ (q' : ℕ+), q' < q → 
    ∃ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 ∧ 
      ∀ (n : ℤ), (↑m / 1007 : ℚ) * ↑q' ≥ ↑n ∨ ↑n ≥ (↑(m + 1) / 1008 : ℚ) * ↑q') ∧
  (∀ (m : ℕ), 1 ≤ m → m ≤ 1006 → 
    ∃ (n : ℤ), (↑m / 1007 : ℚ) * ↑q < ↑n ∧ ↑n < (↑(m + 1) / 1008 : ℚ) * ↑q) :=
by sorry

end NUMINAMATH_CALUDE_smallest_q_for_inequality_l1164_116497


namespace NUMINAMATH_CALUDE_function_property_l1164_116472

theorem function_property (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x - f y) - f (f x) = -f y - 1) →
  (∀ x : ℤ, f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l1164_116472


namespace NUMINAMATH_CALUDE_bowling_team_size_l1164_116401

theorem bowling_team_size (original_avg : ℝ) (new_player1_weight : ℝ) (new_player2_weight : ℝ) (new_avg : ℝ) :
  original_avg = 121 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  new_avg = 113 →
  ∃ n : ℕ, n > 0 ∧ 
    (n * original_avg + new_player1_weight + new_player2_weight) / (n + 2) = new_avg ∧
    n = 7 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_size_l1164_116401


namespace NUMINAMATH_CALUDE_unique_modular_solution_l1164_116453

theorem unique_modular_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -567 [ZMOD 13] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l1164_116453


namespace NUMINAMATH_CALUDE_minimum_rental_fee_for_360_people_l1164_116407

/-- Represents a bus type with its seat capacity and rental fee -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the minimum rental fee for transporting a given number of people -/
def minimumRentalFee (totalPeople : ℕ) (typeA typeB : BusType) : ℕ :=
  sorry

theorem minimum_rental_fee_for_360_people :
  let typeA : BusType := ⟨40, 400⟩
  let typeB : BusType := ⟨50, 480⟩
  minimumRentalFee 360 typeA typeB = 3520 := by
  sorry

end NUMINAMATH_CALUDE_minimum_rental_fee_for_360_people_l1164_116407


namespace NUMINAMATH_CALUDE_triangle_inequality_equality_condition_l1164_116461

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point P
def Point := ℝ × ℝ

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define the sine of an angle in a triangle
def sine (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

-- Define whether a point lies on the circumcircle of a triangle
def onCircumcircle (t : Triangle) (p : Point) : Prop := sorry

theorem triangle_inequality (t : Triangle) (p : Point) :
  distance p t.A * sine t t.A ≤ distance p t.B * sine t t.B + distance p t.C * sine t t.C :=
sorry

theorem equality_condition (t : Triangle) (p : Point) :
  distance p t.A * sine t t.A = distance p t.B * sine t t.B + distance p t.C * sine t t.C ↔
  onCircumcircle t p :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_equality_condition_l1164_116461


namespace NUMINAMATH_CALUDE_quadratic_properties_l1164_116425

/-- A quadratic function y = x² + mx + n -/
def quadratic (m n : ℝ) (x : ℝ) : ℝ := x^2 + m*x + n

theorem quadratic_properties (m n : ℝ) :
  (∀ y₁ y₂ : ℝ, quadratic m n 1 = y₁ ∧ quadratic m n 3 = y₂ ∧ y₁ = y₂ → m = -4) ∧
  (m = -4 ∧ ∃! x, quadratic m n x = 0 → n = 4) ∧
  (∀ a b₁ b₂ : ℝ, quadratic m n a = b₁ ∧ quadratic m n 3 = b₂ ∧ b₁ > b₂ → a < 1 ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1164_116425


namespace NUMINAMATH_CALUDE_no_solutions_for_prime_power_equation_l1164_116437

theorem no_solutions_for_prime_power_equation (p m n k : ℕ) : 
  Nat.Prime p → 
  p % 2 = 1 → 
  0 < n → 
  n ≤ m → 
  m ≤ 3 * n → 
  p^m + p^n + 1 = k^2 → 
  False :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_for_prime_power_equation_l1164_116437


namespace NUMINAMATH_CALUDE_correct_sum_and_digit_sum_l1164_116409

def num1 : ℕ := 943587
def num2 : ℕ := 329430
def incorrect_sum : ℕ := 1412017

def change_digit (n : ℕ) (d e : ℕ) : ℕ := 
  sorry

theorem correct_sum_and_digit_sum :
  ∃ (d e : ℕ),
    (change_digit num1 d e + change_digit num2 d e ≠ incorrect_sum) ∧
    (change_digit num1 d e + change_digit num2 d e = num1 + change_digit num2 d e) ∧
    (d + e = 7) :=
  sorry

end NUMINAMATH_CALUDE_correct_sum_and_digit_sum_l1164_116409


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l1164_116412

/-- The mass of a man who causes a boat to sink by a certain amount -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : Real) : Real :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating the mass of the man in the given problem -/
theorem mass_of_man_on_boat :
  let boat_length : Real := 4
  let boat_breadth : Real := 3
  let boat_sink_height : Real := 0.01
  let water_density : Real := 1000
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 120 := by
  sorry

#check mass_of_man_on_boat

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l1164_116412


namespace NUMINAMATH_CALUDE_sphere_surface_area_tangent_to_cube_l1164_116475

/-- The surface area of a sphere tangent to all six faces of a cube with edge length 2 is 4π. -/
theorem sphere_surface_area_tangent_to_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) : 
  cube_edge_length = 2 → 
  sphere_radius = cube_edge_length / 2 → 
  4 * Real.pi * sphere_radius^2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_tangent_to_cube_l1164_116475


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l1164_116474

theorem five_digit_multiple_of_nine : ∃ (d : ℕ), d < 10 ∧ (56780 + d) % 9 = 0 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l1164_116474


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1164_116423

theorem tangent_line_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x → (∃! x₀ y₀ : ℝ, y₀ = 3*x₀ + c ∧ y₀^2 = 12*x₀)) → 
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1164_116423
