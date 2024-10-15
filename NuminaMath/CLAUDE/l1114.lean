import Mathlib

namespace NUMINAMATH_CALUDE_mariela_cards_total_l1114_111471

/-- Calculates the total number of cards Mariela received based on the given quantities -/
def total_cards (hospital_dozens : ℕ) (hospital_hundreds : ℕ) (home_dozens : ℕ) (home_hundreds : ℕ) : ℕ :=
  (hospital_dozens * 12 + hospital_hundreds * 100) + (home_dozens * 12 + home_hundreds * 100)

/-- Proves that Mariela received 1768 cards in total -/
theorem mariela_cards_total : total_cards 25 7 39 3 = 1768 := by
  sorry

end NUMINAMATH_CALUDE_mariela_cards_total_l1114_111471


namespace NUMINAMATH_CALUDE_integer_part_of_sum_of_roots_l1114_111465

theorem integer_part_of_sum_of_roots (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x*y + y*z + z*x = 1) : 
  ⌊Real.sqrt (3*x*y + 1) + Real.sqrt (3*y*z + 1) + Real.sqrt (3*z*x + 1)⌋ = 4 :=
sorry

end NUMINAMATH_CALUDE_integer_part_of_sum_of_roots_l1114_111465


namespace NUMINAMATH_CALUDE_midpoint_distance_after_movement_l1114_111472

/-- Given two points A and B in a Cartesian plane, if A moves 5 units right and 6 units up,
    and B moves 12 units left and 4 units down, then the distance between the original
    midpoint M and the new midpoint M' is √53/2. -/
theorem midpoint_distance_after_movement (p q r s : ℝ) : 
  let A : ℝ × ℝ := (p, q)
  let B : ℝ × ℝ := (r, s)
  let M : ℝ × ℝ := ((p + r) / 2, (q + s) / 2)
  let A' : ℝ × ℝ := (p + 5, q + 6)
  let B' : ℝ × ℝ := (r - 12, s - 4)
  let M' : ℝ × ℝ := ((p + 5 + r - 12) / 2, (q + 6 + s - 4) / 2)
  Real.sqrt ((M.1 - M'.1)^2 + (M.2 - M'.2)^2) = Real.sqrt 53 / 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_after_movement_l1114_111472


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1114_111477

theorem tan_alpha_value (α : ℝ) 
  (h : (Real.sin (α + Real.pi) + Real.cos (Real.pi - α)) / 
       (Real.sin (Real.pi / 2 - α) + Real.sin (2 * Real.pi - α)) = 5) : 
  Real.tan α = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1114_111477


namespace NUMINAMATH_CALUDE_average_sum_is_six_l1114_111422

theorem average_sum_is_six (a b c d e : ℕ) (h : a + b + c + d + e > 0) :
  let teacher_avg := (5*a + 4*b + 3*c + 2*d + e) / (a + b + c + d + e)
  let kati_avg := (5*e + 4*d + 3*c + 2*b + a) / (a + b + c + d + e)
  teacher_avg + kati_avg = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_sum_is_six_l1114_111422


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_l1114_111487

theorem min_perimeter_rectangle (w l : ℝ) (h1 : w > 0) (h2 : l > 0) (h3 : l = 2 * w) (h4 : w * l ≥ 500) :
  2 * w + 2 * l ≥ 30 * Real.sqrt 10 ∧ 
  (2 * w + 2 * l = 30 * Real.sqrt 10 → w = 5 * Real.sqrt 10 ∧ l = 10 * Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_l1114_111487


namespace NUMINAMATH_CALUDE_notebook_cost_l1114_111495

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buyers : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
  total_students = 35 ∧
  total_cost = 2013 ∧
  buyers > total_students / 2 ∧
  notebooks_per_student % 2 = 0 ∧
  notebooks_per_student > 2 ∧
  cost_per_notebook > notebooks_per_student ∧
  buyers * notebooks_per_student * cost_per_notebook = total_cost ∧
  cost_per_notebook = 61 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l1114_111495


namespace NUMINAMATH_CALUDE_two_problems_without_conditional_statements_l1114_111478

/-- Represents a mathematical problem that may or may not require conditional statements --/
inductive Problem
| OppositeNumber
| SquarePerimeter
| MaximumOfThree
| PiecewiseFunction

/-- Determines if a problem requires conditional statements --/
def requiresConditionalStatements (p : Problem) : Bool :=
  match p with
  | Problem.OppositeNumber => false
  | Problem.SquarePerimeter => false
  | Problem.MaximumOfThree => true
  | Problem.PiecewiseFunction => true

/-- The list of all problems --/
def allProblems : List Problem :=
  [Problem.OppositeNumber, Problem.SquarePerimeter, Problem.MaximumOfThree, Problem.PiecewiseFunction]

/-- Theorem stating that the number of problems not requiring conditional statements is 2 --/
theorem two_problems_without_conditional_statements :
  (allProblems.filter (fun p => ¬(requiresConditionalStatements p))).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_problems_without_conditional_statements_l1114_111478


namespace NUMINAMATH_CALUDE_min_side_in_triangle_l1114_111434

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if c = 2b and the area of the triangle is 1, then the minimum value of a is √3.
-/
theorem min_side_in_triangle (a b c : ℝ) (A B C : ℝ) :
  c = 2 * b →
  (1 / 2) * b * c * Real.sin A = 1 →
  ∃ (a_min : ℝ), a_min = Real.sqrt 3 ∧ ∀ a', a' ≥ a_min := by
  sorry

end NUMINAMATH_CALUDE_min_side_in_triangle_l1114_111434


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1114_111469

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1114_111469


namespace NUMINAMATH_CALUDE_roosters_count_l1114_111406

/-- Given a total number of chickens and a proportion of roosters to hens to chicks,
    calculate the number of roosters. -/
def count_roosters (total_chickens : ℕ) (rooster_parts hen_parts chick_parts : ℕ) : ℕ :=
  let total_parts := rooster_parts + hen_parts + chick_parts
  let chickens_per_part := total_chickens / total_parts
  rooster_parts * chickens_per_part

/-- Theorem stating that given 9000 total chickens and a proportion of 2:1:3 for
    roosters:hens:chicks, the number of roosters is 3000. -/
theorem roosters_count :
  count_roosters 9000 2 1 3 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_roosters_count_l1114_111406


namespace NUMINAMATH_CALUDE_smallest_k_sum_digits_l1114_111418

/-- Sum of digits function -/
def s (n : ℕ) : ℕ := sorry

/-- Theorem stating that 9999 is the smallest positive integer k satisfying the condition -/
theorem smallest_k_sum_digits : 
  (∀ m : ℕ, m ∈ Finset.range 2014 → s ((m + 1) * 9999) = s 9999) ∧ 
  (∀ k : ℕ, k < 9999 → ∃ m : ℕ, m ∈ Finset.range 2014 ∧ s ((m + 1) * k) ≠ s k) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_sum_digits_l1114_111418


namespace NUMINAMATH_CALUDE_simplify_negative_x_powers_l1114_111450

theorem simplify_negative_x_powers (x : ℝ) : (-x)^3 * (-x)^2 = -x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_negative_x_powers_l1114_111450


namespace NUMINAMATH_CALUDE_project_scientists_l1114_111463

/-- The total number of scientists in the project -/
def S : ℕ := 70

/-- The number of scientists from Europe -/
def europe : ℕ := S / 2

/-- The number of scientists from Canada -/
def canada : ℕ := S / 5

/-- The number of scientists from the USA -/
def usa : ℕ := 21

/-- Theorem stating that the sum of scientists from Europe, Canada, and USA equals the total number of scientists -/
theorem project_scientists : europe + canada + usa = S := by sorry

end NUMINAMATH_CALUDE_project_scientists_l1114_111463


namespace NUMINAMATH_CALUDE_line_perp_plane_condition_l1114_111430

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the relation of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- The theorem statement
theorem line_perp_plane_condition 
  (m n : Line) (α β : Plane) 
  (h1 : perp_planes α β)
  (h2 : intersect α β = m)
  (h3 : contained_in n α) :
  perp_line_plane n β ↔ perp_lines n m :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_condition_l1114_111430


namespace NUMINAMATH_CALUDE_multiply_polynomials_l1114_111427

theorem multiply_polynomials (x : ℝ) : 
  (x^4 + 10*x^2 + 25) * (x^2 - 25) = x^4 + 10*x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomials_l1114_111427


namespace NUMINAMATH_CALUDE_f_3_is_even_l1114_111467

/-- Given a function f(x) = a(x-1)³ + bx + c where a is real and b, c are integers,
    if f(-1) = 2, then f(3) must be even. -/
theorem f_3_is_even (a : ℝ) (b c : ℤ) :
  let f : ℝ → ℝ := λ x => a * (x - 1)^3 + b * x + c
  (f (-1) = 2) → ∃ k : ℤ, f 3 = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_f_3_is_even_l1114_111467


namespace NUMINAMATH_CALUDE_count_symmetric_scanning_codes_l1114_111483

/-- A symmetric scanning code is a 7x7 grid of black and white squares that is invariant under 90° rotations and reflections across diagonals and midlines. -/
def SymmetricScanningCode := Fin 7 → Fin 7 → Bool

/-- A scanning code is valid if it has at least one black and one white square. -/
def is_valid (code : SymmetricScanningCode) : Prop :=
  (∃ i j, code i j = true) ∧ (∃ i j, code i j = false)

/-- A scanning code is symmetric if it's invariant under 90° rotations and reflections. -/
def is_symmetric (code : SymmetricScanningCode) : Prop :=
  (∀ i j, code i j = code (6-j) i) ∧  -- 90° rotation
  (∀ i j, code i j = code j i) ∧      -- diagonal reflection
  (∀ i j, code i j = code (6-i) (6-j))  -- midline reflection

/-- The number of valid symmetric scanning codes -/
def num_valid_symmetric_codes : ℕ := sorry

theorem count_symmetric_scanning_codes :
  num_valid_symmetric_codes = 1022 :=
sorry

end NUMINAMATH_CALUDE_count_symmetric_scanning_codes_l1114_111483


namespace NUMINAMATH_CALUDE_watch_time_loss_l1114_111498

/-- Represents the number of minutes lost by a watch per day -/
def minutes_lost_per_day : ℚ := 13/4

/-- Represents the number of hours between 1 P.M. on March 15 and 3 P.M. on March 22 -/
def hours_passed : ℕ := 7 * 24 + 2

/-- Theorem stating that the watch loses 221/96 minutes over the given period -/
theorem watch_time_loss : 
  (minutes_lost_per_day * (hours_passed : ℚ) / 24) = 221/96 := by sorry

end NUMINAMATH_CALUDE_watch_time_loss_l1114_111498


namespace NUMINAMATH_CALUDE_data_transmission_time_l1114_111407

theorem data_transmission_time (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) :
  blocks = 100 →
  chunks_per_block = 800 →
  transmission_rate = 200 →
  (blocks * chunks_per_block : ℝ) / transmission_rate / 60 = 6.666666666666667 :=
by sorry

end NUMINAMATH_CALUDE_data_transmission_time_l1114_111407


namespace NUMINAMATH_CALUDE_julia_tag_game_l1114_111466

theorem julia_tag_game (tuesday_kids : ℕ) (monday_difference : ℕ) : 
  tuesday_kids = 5 → monday_difference = 1 → tuesday_kids + monday_difference = 6 :=
by sorry

end NUMINAMATH_CALUDE_julia_tag_game_l1114_111466


namespace NUMINAMATH_CALUDE_solve_employee_pay_l1114_111428

def employee_pay_problem (pay_B : ℝ) (percent_A : ℝ) : Prop :=
  let pay_A : ℝ := percent_A * pay_B
  let total_pay : ℝ := pay_A + pay_B
  pay_B = 228 ∧ percent_A = 1.5 → total_pay = 570

theorem solve_employee_pay : employee_pay_problem 228 1.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_employee_pay_l1114_111428


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1114_111484

theorem complex_fraction_simplification :
  (7 + 15 * Complex.I) / (3 - 4 * Complex.I) = -39/25 + (73/25) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1114_111484


namespace NUMINAMATH_CALUDE_f_not_in_second_quadrant_l1114_111460

/-- A linear function f(x) = 2x - 1 -/
def f (x : ℝ) : ℝ := 2 * x - 1

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem: The graph of f(x) = 2x - 1 does not pass through the second quadrant -/
theorem f_not_in_second_quadrant :
  ∀ x y : ℝ, f x = y → ¬(second_quadrant x y) :=
by sorry

end NUMINAMATH_CALUDE_f_not_in_second_quadrant_l1114_111460


namespace NUMINAMATH_CALUDE_exponent_difference_equals_204_l1114_111420

theorem exponent_difference_equals_204 : 3^(1*(2+3)) - (3^1 + 3^2 + 3^3) = 204 := by
  sorry

end NUMINAMATH_CALUDE_exponent_difference_equals_204_l1114_111420


namespace NUMINAMATH_CALUDE_net_error_is_24x_l1114_111426

/-- The net error in cents due to the cashier's miscounting -/
def net_error (x : ℕ) : ℤ :=
  let penny_value : ℤ := 1
  let nickel_value : ℤ := 5
  let dime_value : ℤ := 10
  let quarter_value : ℤ := 25
  let penny_to_nickel_error := x * (nickel_value - penny_value)
  let nickel_to_dime_error := x * (dime_value - nickel_value)
  let dime_to_quarter_error := x * (quarter_value - dime_value)
  penny_to_nickel_error + nickel_to_dime_error + dime_to_quarter_error

theorem net_error_is_24x (x : ℕ) : net_error x = 24 * x :=
sorry

end NUMINAMATH_CALUDE_net_error_is_24x_l1114_111426


namespace NUMINAMATH_CALUDE_trip_cost_calculation_l1114_111453

theorem trip_cost_calculation (original_price discount : ℕ) (num_people : ℕ) : 
  original_price = 147 → 
  discount = 14 → 
  num_people = 2 → 
  (original_price - discount) * num_people = 266 := by
sorry

end NUMINAMATH_CALUDE_trip_cost_calculation_l1114_111453


namespace NUMINAMATH_CALUDE_tea_preparation_time_l1114_111424

/-- Represents the time required for each task in minutes -/
structure TaskTimes where
  washKettle : ℕ
  boilWater : ℕ
  washTeapot : ℕ
  washTeacups : ℕ
  getTeaLeaves : ℕ

/-- Calculates the minimum time required to complete all tasks -/
def minTimeRequired (times : TaskTimes) : ℕ :=
  max times.washKettle (times.boilWater + times.washKettle)

/-- Theorem stating that the minimum time required is 16 minutes -/
theorem tea_preparation_time (times : TaskTimes) 
  (h1 : times.washKettle = 1)
  (h2 : times.boilWater = 15)
  (h3 : times.washTeapot = 1)
  (h4 : times.washTeacups = 1)
  (h5 : times.getTeaLeaves = 2) :
  minTimeRequired times = 16 := by
  sorry


end NUMINAMATH_CALUDE_tea_preparation_time_l1114_111424


namespace NUMINAMATH_CALUDE_school_average_age_l1114_111485

/-- Given a school with the following properties:
  * Total number of students is 600
  * Average age of boys is 12 years
  * Average age of girls is 11 years
  * Number of girls is 150
  Prove that the average age of the school is 11.75 years -/
theorem school_average_age 
  (total_students : ℕ) 
  (boys_avg_age girls_avg_age : ℚ)
  (num_girls : ℕ) :
  total_students = 600 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  num_girls = 150 →
  let num_boys := total_students - num_girls
  let total_age := boys_avg_age * num_boys + girls_avg_age * num_girls
  total_age / total_students = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_school_average_age_l1114_111485


namespace NUMINAMATH_CALUDE_solve_kitchen_supplies_l1114_111438

def kitchen_supplies_problem (angela_pots : ℕ) (angela_plates : ℕ) (angela_cutlery : ℕ) 
  (sharon_total : ℕ) : Prop :=
  angela_pots = 20 ∧
  angela_plates > 3 * angela_pots ∧
  angela_cutlery = angela_plates / 2 ∧
  sharon_total = 254 ∧
  sharon_total = angela_pots / 2 + (3 * angela_plates - 20) + 2 * angela_cutlery ∧
  angela_plates - 3 * angela_pots = 6

theorem solve_kitchen_supplies : 
  ∃ (angela_pots angela_plates angela_cutlery : ℕ),
    kitchen_supplies_problem angela_pots angela_plates angela_cutlery 254 :=
sorry

end NUMINAMATH_CALUDE_solve_kitchen_supplies_l1114_111438


namespace NUMINAMATH_CALUDE_quadratic_roots_l1114_111402

theorem quadratic_roots (a : ℝ) : 
  (2 : ℝ)^2 + 2 - a = 0 → (-3 : ℝ)^2 + (-3) - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1114_111402


namespace NUMINAMATH_CALUDE_certain_number_proof_l1114_111458

theorem certain_number_proof (N : ℝ) : (5/6) * N = (5/16) * N + 50 → N = 96 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1114_111458


namespace NUMINAMATH_CALUDE_probability_of_losing_l1114_111468

theorem probability_of_losing (odds_win odds_lose : ℕ) 
  (h_odds : odds_win = 5 ∧ odds_lose = 3) : 
  (odds_lose : ℚ) / (odds_win + odds_lose) = 3 / 8 :=
by
  sorry

#check probability_of_losing

end NUMINAMATH_CALUDE_probability_of_losing_l1114_111468


namespace NUMINAMATH_CALUDE_smallest_class_size_l1114_111491

theorem smallest_class_size : ∃ n : ℕ, n > 0 ∧ 
  n % 6 = 3 ∧ 
  n % 8 = 5 ∧ 
  n % 9 = 7 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 3 → m % 8 = 5 → m % 9 = 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l1114_111491


namespace NUMINAMATH_CALUDE_math_club_team_selection_l1114_111499

def math_club_selection (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) (boys_in_team : ℕ) (girls_in_team : ℕ) : ℕ :=
  (total_boys.choose boys_in_team) * (total_girls.choose girls_in_team)

theorem math_club_team_selection :
  math_club_selection 10 12 8 4 4 = 103950 := by
sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l1114_111499


namespace NUMINAMATH_CALUDE_ellipse_max_product_l1114_111423

theorem ellipse_max_product (x y : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  (x^2 / 25 + y^2 / 9 = 1) →
  (P = (x, y)) →
  (F₁ ≠ F₂) →
  (∀ (x' y' : ℝ), x'^2 / 25 + y'^2 / 9 = 1 → 
    dist P F₁ + dist P F₂ = dist (x', y') F₁ + dist (x', y') F₂) →
  (∃ (M : ℝ), ∀ (x' y' : ℝ), x'^2 / 25 + y'^2 / 9 = 1 → 
    dist (x', y') F₁ * dist (x', y') F₂ ≤ M ∧ 
    ∃ (x'' y'' : ℝ), x''^2 / 25 + y''^2 / 9 = 1 ∧ 
      dist (x'', y'') F₁ * dist (x'', y'') F₂ = M) →
  M = 25 := by
sorry

end NUMINAMATH_CALUDE_ellipse_max_product_l1114_111423


namespace NUMINAMATH_CALUDE_lcm_of_18_50_120_l1114_111454

theorem lcm_of_18_50_120 : Nat.lcm (Nat.lcm 18 50) 120 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_18_50_120_l1114_111454


namespace NUMINAMATH_CALUDE_remainder_theorem_l1114_111433

-- Define the polynomial p(x)
variable (p : ℝ → ℝ)

-- Define the conditions
axiom remainder_x_minus_3 : ∃ q : ℝ → ℝ, ∀ x, p x = (x - 3) * q x + 7
axiom remainder_x_plus_2 : ∃ q : ℝ → ℝ, ∀ x, p x = (x + 2) * q x - 3

-- Theorem statement
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 3) * (x + 2) * q x + (2 * x + 1) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1114_111433


namespace NUMINAMATH_CALUDE_triangle_height_l1114_111448

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 3 → area = 6 → area = (base * height) / 2 → height = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l1114_111448


namespace NUMINAMATH_CALUDE_positive_integer_pairs_l1114_111441

theorem positive_integer_pairs (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (∃ k : ℤ, (a^3 * b - 1 : ℤ) = k * (a + 1)) ∧
  (∃ m : ℤ, (b^3 * a + 1 : ℤ) = m * (b - 1)) →
  ((a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_l1114_111441


namespace NUMINAMATH_CALUDE_wood_length_after_sawing_l1114_111431

theorem wood_length_after_sawing (original_length saw_length : Real) 
  (h1 : original_length = 0.41)
  (h2 : saw_length = 0.33) :
  original_length - saw_length = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_after_sawing_l1114_111431


namespace NUMINAMATH_CALUDE_incorrect_statement_l1114_111455

theorem incorrect_statement (P Q : Prop) (h1 : P ↔ (2 + 2 = 5)) (h2 : Q ↔ (3 > 2)) : 
  ¬((¬(P ∧ Q)) ∧ (¬¬P)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_statement_l1114_111455


namespace NUMINAMATH_CALUDE_power_two_gt_square_plus_one_l1114_111443

theorem power_two_gt_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_gt_square_plus_one_l1114_111443


namespace NUMINAMATH_CALUDE_marks_books_l1114_111457

/-- Given Mark's initial amount, cost per book, and remaining amount, prove the number of books he bought. -/
theorem marks_books (initial_amount : ℕ) (cost_per_book : ℕ) (remaining_amount : ℕ) :
  initial_amount = 85 →
  cost_per_book = 5 →
  remaining_amount = 35 →
  (initial_amount - remaining_amount) / cost_per_book = 10 :=
by sorry

end NUMINAMATH_CALUDE_marks_books_l1114_111457


namespace NUMINAMATH_CALUDE_swim_time_ratio_l1114_111413

/-- Proves that the ratio of time taken to swim upstream to downstream is 2:1 given specific speeds -/
theorem swim_time_ratio (man_speed stream_speed : ℝ) 
  (h1 : man_speed = 3)
  (h2 : stream_speed = 1) :
  (man_speed - stream_speed)⁻¹ / (man_speed + stream_speed)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_swim_time_ratio_l1114_111413


namespace NUMINAMATH_CALUDE_video_game_lives_l1114_111414

theorem video_game_lives (initial_lives lost_lives gained_lives : ℕ) 
  (h1 : initial_lives = 47)
  (h2 : lost_lives = 23)
  (h3 : gained_lives = 46) :
  initial_lives - lost_lives + gained_lives = 70 := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l1114_111414


namespace NUMINAMATH_CALUDE_printer_Z_time_l1114_111449

/-- The time it takes for printer Z to do the job alone -/
def T_Z : ℝ := 18

/-- The time it takes for printer X to do the job alone -/
def T_X : ℝ := 15

/-- The time it takes for printer Y to do the job alone -/
def T_Y : ℝ := 12

/-- The ratio of X's time to Y and Z's combined time -/
def ratio : ℝ := 2.0833333333333335

theorem printer_Z_time :
  T_Z = 18 ∧
  T_X = 15 ∧
  T_Y = 12 ∧
  ratio = 15 / (1 / (1 / T_Y + 1 / T_Z)) :=
by sorry

end NUMINAMATH_CALUDE_printer_Z_time_l1114_111449


namespace NUMINAMATH_CALUDE_problem_solution_l1114_111447

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := 
  fun (a, b) (c, d) ↦ (a - c, b + d)

/-- Theorem stating that given the conditions, a = 2 -/
theorem problem_solution : 
  ∃ (a b : ℤ), star (5, 2) (1, 1) = (a, b) ∧ star (a, b) (0, 2) = (2, 5) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1114_111447


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1114_111456

-- Define the sides of the triangle
def side1 : ℝ := 9
def side2 : ℝ := 9
def side3 : ℝ := 4

-- Define the isosceles triangle condition
def is_isosceles (a b c : ℝ) : Prop := (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

-- Define the triangle inequality
def satisfies_triangle_inequality (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  is_isosceles side1 side2 side3 ∧
  satisfies_triangle_inequality side1 side2 side3 →
  perimeter side1 side2 side3 = 22 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1114_111456


namespace NUMINAMATH_CALUDE_smaller_octagon_area_ratio_l1114_111419

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The smaller octagon formed by connecting midpoints of sides of a regular octagon -/
def smallerOctagon (oct : RegularOctagon) : RegularOctagon := sorry

/-- The area of a regular octagon -/
def area (oct : RegularOctagon) : ℝ := sorry

/-- Theorem: The area of the smaller octagon is half the area of the larger octagon -/
theorem smaller_octagon_area_ratio (oct : RegularOctagon) : 
  area (smallerOctagon oct) = (1/2 : ℝ) * area oct := by sorry

end NUMINAMATH_CALUDE_smaller_octagon_area_ratio_l1114_111419


namespace NUMINAMATH_CALUDE_lottery_profit_lilys_profit_l1114_111444

/-- Calculates the profit from selling lottery tickets -/
theorem lottery_profit (n : ℕ) (first_price : ℕ) (prize : ℕ) : ℕ :=
  let total_revenue := n * (2 * first_price + (n - 1)) / 2
  total_revenue - prize

/-- Proves that Lily's profit is $4 given the specified conditions -/
theorem lilys_profit :
  lottery_profit 5 1 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_lottery_profit_lilys_profit_l1114_111444


namespace NUMINAMATH_CALUDE_food_allocation_l1114_111452

/-- Given a total budget allocated among three categories in a specific ratio,
    calculate the amount allocated to the second category. -/
def allocate_budget (total : ℚ) (ratio1 ratio2 ratio3 : ℕ) : ℚ :=
  (total * ratio2) / (ratio1 + ratio2 + ratio3)

/-- Theorem stating that given a total budget of 1800 allocated in the ratio 5:4:1,
    the amount allocated to the second category is 720. -/
theorem food_allocation :
  allocate_budget 1800 5 4 1 = 720 := by
  sorry

end NUMINAMATH_CALUDE_food_allocation_l1114_111452


namespace NUMINAMATH_CALUDE_nine_point_circle_triangles_l1114_111492

/-- Given 9 points on a circle, this function calculates the number of distinct triangles
    formed by the intersection points of chords inside the circle. --/
def count_triangles (n : ℕ) : ℕ :=
  Nat.choose n 6

/-- Theorem stating that for 9 points on a circle, with chords connecting every pair of points
    and no three chords intersecting at a single point inside the circle, the number of
    distinct triangles formed by the intersection points of these chords inside the circle is 84. --/
theorem nine_point_circle_triangles :
  count_triangles 9 = 84 := by
  sorry

end NUMINAMATH_CALUDE_nine_point_circle_triangles_l1114_111492


namespace NUMINAMATH_CALUDE_f_even_h_odd_l1114_111493

-- Define the functions f and h
def f (x : ℝ) : ℝ := x^2
def h (x : ℝ) : ℝ := x

-- State the theorem
theorem f_even_h_odd : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, h (-x) = -h x) := by
  sorry

end NUMINAMATH_CALUDE_f_even_h_odd_l1114_111493


namespace NUMINAMATH_CALUDE_total_spent_is_450_l1114_111462

/-- The total amount spent by Leonard and Michael on presents for their father -/
def total_spent (leonard_wallet : ℕ) (leonard_sneakers : ℕ) (leonard_sneakers_pairs : ℕ)
  (michael_backpack : ℕ) (michael_jeans : ℕ) (michael_jeans_pairs : ℕ) : ℕ :=
  leonard_wallet + leonard_sneakers * leonard_sneakers_pairs +
  michael_backpack + michael_jeans * michael_jeans_pairs

/-- Theorem stating that the total amount spent is $450 -/
theorem total_spent_is_450 :
  total_spent 50 100 2 100 50 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_450_l1114_111462


namespace NUMINAMATH_CALUDE_quadratic_ratio_l1114_111432

theorem quadratic_ratio (x : ℝ) : 
  ∃ (d e : ℝ), x^2 + 2600*x + 2600 = (x + d)^2 + e ∧ e / d = -1298 := by
sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l1114_111432


namespace NUMINAMATH_CALUDE_P_roots_properties_l1114_111464

/-- Definition of the polynomial sequence P_n(x) -/
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | n + 1, x => x^(5*(n+1)) - P n x

/-- Theorem stating the properties of real roots for P_n(x) -/
theorem P_roots_properties :
  (∀ n : ℕ, Odd n → (∃! x : ℝ, P n x = 0 ∧ x = 1)) ∧
  (∀ n : ℕ, Even n → ∀ x : ℝ, P n x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_P_roots_properties_l1114_111464


namespace NUMINAMATH_CALUDE_fourth_month_sale_is_13792_l1114_111488

/-- Represents the sales data for a grocery shop over 6 months -/
structure SalesData where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Calculates the sale in the fourth month given the sales data and average -/
def fourthMonthSale (data : SalesData) (average : ℕ) : ℕ :=
  6 * average - (data.month1 + data.month2 + data.month3 + data.month5 + data.month6)

/-- Theorem stating that the fourth month's sale is 13792 given the conditions -/
theorem fourth_month_sale_is_13792 :
  let data : SalesData := {
    month1 := 6635,
    month2 := 6927,
    month3 := 6855,
    month4 := 0,  -- Unknown, to be calculated
    month5 := 6562,
    month6 := 4791
  }
  let average := 6500
  fourthMonthSale data average = 13792 := by
  sorry

#eval fourthMonthSale
  { month1 := 6635,
    month2 := 6927,
    month3 := 6855,
    month4 := 0,
    month5 := 6562,
    month6 := 4791 }
  6500

end NUMINAMATH_CALUDE_fourth_month_sale_is_13792_l1114_111488


namespace NUMINAMATH_CALUDE_trig_problem_l1114_111497

theorem trig_problem (a : Real) (h1 : 0 < a) (h2 : a < Real.pi) (h3 : Real.tan a = -2) :
  (Real.cos a = -Real.sqrt 5 / 5) ∧
  (2 * Real.sin a ^ 2 - Real.sin a * Real.cos a + Real.cos a ^ 2 = 11 / 5) := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l1114_111497


namespace NUMINAMATH_CALUDE_will_hero_count_l1114_111401

/-- Represents the number of heroes drawn on a sheet of paper -/
structure HeroCount where
  front : Nat
  back : Nat
  third : Nat

/-- Calculates the total number of heroes drawn -/
def totalHeroes (h : HeroCount) : Nat :=
  h.front + h.back + h.third

/-- Theorem: Given the specific hero counts, the total is 19 -/
theorem will_hero_count :
  ∃ (h : HeroCount), h.front = 4 ∧ h.back = 9 ∧ h.third = 6 ∧ totalHeroes h = 19 :=
by sorry

end NUMINAMATH_CALUDE_will_hero_count_l1114_111401


namespace NUMINAMATH_CALUDE_brown_mms_problem_l1114_111403

theorem brown_mms_problem (bag1 bag2 bag3 bag4 bag5 : ℕ) 
  (h1 : bag1 = 9)
  (h2 : bag2 = 12)
  (h5 : bag5 = 3)
  (h_avg : (bag1 + bag2 + bag3 + bag4 + bag5) / 5 = 8) :
  bag3 + bag4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_brown_mms_problem_l1114_111403


namespace NUMINAMATH_CALUDE_adam_tickets_bought_l1114_111489

def tickets_bought (tickets_left : ℕ) (ticket_cost : ℕ) (amount_spent : ℕ) : ℕ :=
  tickets_left + amount_spent / ticket_cost

theorem adam_tickets_bought :
  tickets_bought 4 9 81 = 13 := by
  sorry

end NUMINAMATH_CALUDE_adam_tickets_bought_l1114_111489


namespace NUMINAMATH_CALUDE_worker_schedule_solution_correct_l1114_111496

/-- Represents the worker payment schedule problem over a 30-day period. -/
structure WorkerSchedule where
  total_days : ℕ
  daily_wage : ℕ
  daily_penalty : ℕ
  total_earnings : ℤ

/-- The solution to the worker schedule problem. -/
def solve_worker_schedule (ws : WorkerSchedule) : ℕ :=
  sorry

/-- Theorem stating the correctness of the solution for the given problem. -/
theorem worker_schedule_solution_correct (ws : WorkerSchedule) : 
  ws.total_days = 30 ∧ 
  ws.daily_wage = 100 ∧ 
  ws.daily_penalty = 25 ∧ 
  ws.total_earnings = 0 →
  solve_worker_schedule ws = 24 :=
sorry

end NUMINAMATH_CALUDE_worker_schedule_solution_correct_l1114_111496


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1114_111490

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  n : ℕ -- number of terms
  d : ℝ -- common difference
  a₁ : ℝ -- first term

/-- Sum of magnitudes of terms in an arithmetic sequence -/
def sumOfMagnitudes (seq : ArithmeticSequence) : ℝ := 
  sorry

/-- New sequence obtained by adding a constant to all terms -/
def addConstant (seq : ArithmeticSequence) (c : ℝ) : ArithmeticSequence :=
  sorry

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  sumOfMagnitudes seq = 250 ∧
  sumOfMagnitudes (addConstant seq 1) = 250 ∧
  sumOfMagnitudes (addConstant seq 2) = 250 →
  seq.n^2 * seq.d = 1000 ∨ seq.n^2 * seq.d = -1000 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1114_111490


namespace NUMINAMATH_CALUDE_jia_steps_to_meet_yi_l1114_111412

theorem jia_steps_to_meet_yi (distance : ℝ) (speed_ratio : ℝ) (step_length : ℝ) :
  distance = 10560 ∧ speed_ratio = 5 ∧ step_length = 2.5 →
  (distance / (1 + speed_ratio)) / step_length = 704 := by
  sorry

end NUMINAMATH_CALUDE_jia_steps_to_meet_yi_l1114_111412


namespace NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l1114_111461

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_fraction_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((a + 3 * Complex.I) / (1 - Complex.I)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l1114_111461


namespace NUMINAMATH_CALUDE_average_playing_time_l1114_111415

/-- The average playing time for children playing table tennis -/
theorem average_playing_time
  (num_children : ℕ)
  (total_time : ℝ)
  (h_num_children : num_children = 5)
  (h_total_time : total_time = 15)
  : total_time / num_children = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_playing_time_l1114_111415


namespace NUMINAMATH_CALUDE_pizza_combinations_l1114_111442

theorem pizza_combinations : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l1114_111442


namespace NUMINAMATH_CALUDE_power_function_value_l1114_111474

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f (1/2) = 8 → f 2 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l1114_111474


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l1114_111437

/-- The area of a stripe on a cylindrical water tower -/
theorem stripe_area_on_cylinder (diameter : ℝ) (stripe_width : ℝ) (revolutions : ℕ) :
  diameter = 20 ∧ stripe_width = 4 ∧ revolutions = 3 →
  stripe_width * revolutions * π * diameter = 240 * π :=
by sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l1114_111437


namespace NUMINAMATH_CALUDE_nested_radical_equality_l1114_111451

theorem nested_radical_equality : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_equality_l1114_111451


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l1114_111480

theorem no_linear_term_condition (p q : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x^2 - p*x + q)*(x - 3) = a*x^3 + b*x^2 + c) → 
  q + 3*p = 0 := by
sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l1114_111480


namespace NUMINAMATH_CALUDE_can_reach_ten_white_marbles_l1114_111479

-- Define the state of the urn
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

-- Define the possible operations
inductive Operation
  | op1 -- 4B -> 2B
  | op2 -- 3B + W -> B
  | op3 -- 2B + 2W -> W + B
  | op4 -- B + 3W -> 2W
  | op5 -- 4W -> B

-- Define a function to apply an operation to the urn state
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.op1 => ⟨state.white, state.black - 2⟩
  | Operation.op2 => ⟨state.white - 1, state.black - 2⟩
  | Operation.op3 => ⟨state.white - 1, state.black - 1⟩
  | Operation.op4 => ⟨state.white - 1, state.black - 1⟩
  | Operation.op5 => ⟨state.white - 4, state.black + 1⟩

-- Define the initial state
def initialState : UrnState := ⟨50, 150⟩

-- Theorem: It is possible to reach exactly 10 white marbles
theorem can_reach_ten_white_marbles :
  ∃ (operations : List Operation),
    (operations.foldl applyOperation initialState).white = 10 :=
sorry

end NUMINAMATH_CALUDE_can_reach_ten_white_marbles_l1114_111479


namespace NUMINAMATH_CALUDE_simplify_expression_l1114_111421

theorem simplify_expression : (512 : ℝ)^(1/3) * (343 : ℝ)^(1/2) = 56 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1114_111421


namespace NUMINAMATH_CALUDE_sqrt_diff_inequality_l1114_111476

theorem sqrt_diff_inequality (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n - 1) - Real.sqrt n < Real.sqrt n - Real.sqrt (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_diff_inequality_l1114_111476


namespace NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l1114_111446

theorem quadratic_equation_rational_solutions :
  ∃! (c₁ c₂ : ℕ+), 
    (∃ (x : ℚ), 7 * x^2 + 13 * x + c₁.val = 0) ∧
    (∃ (x : ℚ), 7 * x^2 + 13 * x + c₂.val = 0) ∧
    c₁ = c₂ ∧ c₁ = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_rational_solutions_l1114_111446


namespace NUMINAMATH_CALUDE_newer_model_travels_200_miles_l1114_111436

/-- The distance traveled by the older model car -/
def older_model_distance : ℝ := 160

/-- The percentage increase in distance for the newer model -/
def newer_model_percentage : ℝ := 0.25

/-- The distance traveled by the newer model car -/
def newer_model_distance : ℝ := older_model_distance * (1 + newer_model_percentage)

/-- Theorem stating that the newer model travels 200 miles -/
theorem newer_model_travels_200_miles :
  newer_model_distance = 200 := by sorry

end NUMINAMATH_CALUDE_newer_model_travels_200_miles_l1114_111436


namespace NUMINAMATH_CALUDE_kevins_record_is_72_l1114_111417

/-- Calculates the number of wings in Kevin's hot wing eating record --/
def kevins_record (duration : ℕ) (alans_rate : ℕ) (additional_wings_needed : ℕ) : ℕ :=
  duration * (alans_rate + additional_wings_needed)

theorem kevins_record_is_72 :
  kevins_record 8 5 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_kevins_record_is_72_l1114_111417


namespace NUMINAMATH_CALUDE_salary_increase_proof_l1114_111435

/-- Calculates the increase in average salary when adding a manager to a group of employees -/
def salary_increase (num_employees : ℕ) (initial_avg : ℚ) (manager_salary : ℚ) : ℚ :=
  let new_total := num_employees * initial_avg + manager_salary
  let new_avg := new_total / (num_employees + 1)
  new_avg - initial_avg

/-- The increase in average salary when adding a manager's salary of 3300 to a group of 20 employees with an initial average salary of 1200 is equal to 100 -/
theorem salary_increase_proof :
  salary_increase 20 1200 3300 = 100 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l1114_111435


namespace NUMINAMATH_CALUDE_custom_bowling_ball_volume_l1114_111481

/-- The volume of a customized bowling ball -/
theorem custom_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 6
  let small_hole_diameter : ℝ := 2.5
  let large_hole_diameter : ℝ := 4
  let sphere_volume := (4/3) * π * (sphere_diameter/2)^3
  let small_hole_volume := π * (small_hole_diameter/2)^2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter/2)^2 * hole_depth
  sphere_volume - 2 * small_hole_volume - large_hole_volume = 2261.25 * π :=
by sorry

end NUMINAMATH_CALUDE_custom_bowling_ball_volume_l1114_111481


namespace NUMINAMATH_CALUDE_rohan_salary_calculation_l1114_111486

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 12500

/-- The percentage of salary Rohan spends on food -/
def food_expense_percent : ℝ := 40

/-- The percentage of salary Rohan spends on house rent -/
def rent_expense_percent : ℝ := 20

/-- The percentage of salary Rohan spends on entertainment -/
def entertainment_expense_percent : ℝ := 10

/-- The percentage of salary Rohan spends on conveyance -/
def conveyance_expense_percent : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 2500

/-- Theorem stating that given the conditions, Rohan's monthly salary is Rs. 12500 -/
theorem rohan_salary_calculation :
  (food_expense_percent + rent_expense_percent + entertainment_expense_percent + conveyance_expense_percent) / 100 * monthly_salary + savings = monthly_salary :=
by sorry

end NUMINAMATH_CALUDE_rohan_salary_calculation_l1114_111486


namespace NUMINAMATH_CALUDE_distance_traveled_l1114_111440

-- Define the velocity function
def velocity (t : ℝ) : ℝ := t^2 + 1

-- Define the theorem
theorem distance_traveled (v : ℝ → ℝ) (a b : ℝ) : 
  (v = velocity) → (a = 0) → (b = 3) → ∫ x in a..b, v x = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1114_111440


namespace NUMINAMATH_CALUDE_square_side_equals_pi_l1114_111445

theorem square_side_equals_pi :
  ∀ x : ℝ,
  (4 * x = 2 * π * 2) →
  x = π :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_equals_pi_l1114_111445


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l1114_111405

theorem arithmetic_sqrt_of_nine (x : ℝ) :
  (x ≥ 0 ∧ x^2 = 9) → x = 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l1114_111405


namespace NUMINAMATH_CALUDE_two_solutions_l1114_111409

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line 2x - 3y + 5 = 0 -/
def onLine (p : Point) : Prop :=
  2 * p.x - 3 * p.y + 5 = 0

/-- The distance between two points is √13 -/
def hasDistance13 (p : Point) : Prop :=
  (p.x - 2)^2 + (p.y - 3)^2 = 13

/-- The two solutions -/
def solution1 : Point := ⟨-1, 1⟩
def solution2 : Point := ⟨5, 5⟩

theorem two_solutions :
  ∀ p : Point, (onLine p ∧ hasDistance13 p) ↔ (p = solution1 ∨ p = solution2) := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_l1114_111409


namespace NUMINAMATH_CALUDE_ladder_distance_l1114_111404

theorem ladder_distance (ladder_length height : ℝ) 
  (h1 : ladder_length = 25)
  (h2 : height = 20) :
  ∃ (distance : ℝ), distance^2 + height^2 = ladder_length^2 ∧ distance = 15 :=
sorry

end NUMINAMATH_CALUDE_ladder_distance_l1114_111404


namespace NUMINAMATH_CALUDE_dot_only_count_l1114_111429

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total : ℕ)
  (dot_and_line : ℕ)
  (line_only : ℕ)
  (all_have_dot_or_line : Prop)

/-- The number of letters containing a dot but not a straight line -/
def dot_only (α : Alphabet) : ℕ :=
  α.total - α.dot_and_line - α.line_only

/-- Theorem stating the number of letters with only a dot in the given alphabet -/
theorem dot_only_count (α : Alphabet) 
  (h1 : α.total = 40)
  (h2 : α.dot_and_line = 13)
  (h3 : α.line_only = 24) :
  dot_only α = 16 := by
  sorry

end NUMINAMATH_CALUDE_dot_only_count_l1114_111429


namespace NUMINAMATH_CALUDE_total_spent_on_tickets_l1114_111459

def this_year_prices : List ℕ := [35, 45, 50, 62]
def last_year_prices : List ℕ := [25, 30, 40, 45, 55, 60, 65, 70, 75]

theorem total_spent_on_tickets : 
  (this_year_prices.sum + last_year_prices.sum : ℕ) = 657 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_tickets_l1114_111459


namespace NUMINAMATH_CALUDE_sin_1035_degrees_l1114_111400

theorem sin_1035_degrees : Real.sin (1035 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1035_degrees_l1114_111400


namespace NUMINAMATH_CALUDE_chess_tournament_attendance_l1114_111475

theorem chess_tournament_attendance (total_students : ℕ) 
  (h1 : total_students = 24) 
  (h2 : ∃ chess_students : ℕ, chess_students = total_students / 3)
  (h3 : ∃ tournament_students : ℕ, tournament_students = (total_students / 3) / 2) :
  ∃ tournament_students : ℕ, tournament_students = 4 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_attendance_l1114_111475


namespace NUMINAMATH_CALUDE_expression_evaluation_l1114_111410

theorem expression_evaluation : (-2 : ℤ) ^ (4^2) + 1^(3^3) = 65537 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1114_111410


namespace NUMINAMATH_CALUDE_class_size_l1114_111473

theorem class_size (hockey : ℕ) (basketball : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : hockey = 15)
  (h2 : basketball = 16)
  (h3 : both = 10)
  (h4 : neither = 4) :
  hockey + basketball - both + neither = 25 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1114_111473


namespace NUMINAMATH_CALUDE_negation_equivalence_l1114_111411

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1114_111411


namespace NUMINAMATH_CALUDE_ellipse_conditions_l1114_111416

-- Define what it means for an equation to represent an ellipse
def represents_ellipse (a b : ℝ) : Prop :=
  ∃ (h k : ℝ) (A B : ℝ), A ≠ B ∧ A > 0 ∧ B > 0 ∧
    ∀ (x y : ℝ), a * (x - h)^2 + b * (y - k)^2 = 1 ↔ 
      ((x - h)^2 / A^2) + ((y - k)^2 / B^2) = 1

-- State the theorem
theorem ellipse_conditions (a b : ℝ) :
  (a > 0 ∧ b > 0 ∧ represents_ellipse a b) ∧
  ¬(a > 0 ∧ b > 0 → represents_ellipse a b) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_conditions_l1114_111416


namespace NUMINAMATH_CALUDE_triangle_angle_determination_l1114_111425

theorem triangle_angle_determination (a b : ℝ) (A B : Real) :
  a = 40 →
  b = 20 * Real.sqrt 2 →
  A = π / 4 →
  Real.sin B = (b * Real.sin A) / a →
  0 < B →
  B < π / 4 →
  B = π / 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_determination_l1114_111425


namespace NUMINAMATH_CALUDE_expression_value_l1114_111408

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : abs m = 3)  -- |m| = 3
  : m + c * d - (a + b) / (m^2) = 4 ∨ m + c * d - (a + b) / (m^2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1114_111408


namespace NUMINAMATH_CALUDE_sin_double_theta_l1114_111439

theorem sin_double_theta (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5 →
  Real.sin (2 * θ) = 6 * Real.sqrt 8 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_theta_l1114_111439


namespace NUMINAMATH_CALUDE_mass_percentage_cl_in_mixture_mass_percentage_cl_approx_43_85_l1114_111482

/-- Mass percentage of Cl in a mixture of NaClO and NaClO2 -/
theorem mass_percentage_cl_in_mixture (moles_NaClO moles_NaClO2 : ℝ) 
  (mass_Na mass_Cl mass_O : ℝ) : ℝ :=
  let molar_mass_NaClO := mass_Na + mass_Cl + mass_O
  let molar_mass_NaClO2 := mass_Na + mass_Cl + 2 * mass_O
  let mass_Cl_NaClO := moles_NaClO * mass_Cl
  let mass_Cl_NaClO2 := moles_NaClO2 * mass_Cl
  let total_mass_Cl := mass_Cl_NaClO + mass_Cl_NaClO2
  let total_mass_mixture := moles_NaClO * molar_mass_NaClO + moles_NaClO2 * molar_mass_NaClO2
  let mass_percentage_Cl := (total_mass_Cl / total_mass_mixture) * 100
  mass_percentage_Cl

/-- The mass percentage of Cl in the given mixture is approximately 43.85% -/
theorem mass_percentage_cl_approx_43_85 :
  abs (mass_percentage_cl_in_mixture 3 2 22.99 35.45 16 - 43.85) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_mass_percentage_cl_in_mixture_mass_percentage_cl_approx_43_85_l1114_111482


namespace NUMINAMATH_CALUDE_new_person_weight_l1114_111494

theorem new_person_weight (n : ℕ) (old_weight avg_increase : ℝ) :
  n = 8 ∧ 
  old_weight = 70 ∧ 
  avg_increase = 3 →
  (n * avg_increase + old_weight : ℝ) = 94 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1114_111494


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1114_111470

theorem polynomial_remainder (x : ℤ) : (x^15 - 2) % (x + 2) = -32770 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1114_111470
