import Mathlib

namespace NUMINAMATH_CALUDE_probability_point_in_circle_l1424_142478

/-- The probability of a point randomly selected from a square with side length 4
    being within a circle of radius 2 centered at the origin is π/4. -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) :
  square_side = 4 →
  circle_radius = 2 →
  (π * circle_radius^2) / (square_side^2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l1424_142478


namespace NUMINAMATH_CALUDE_set_operations_l1424_142498

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}

theorem set_operations :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1424_142498


namespace NUMINAMATH_CALUDE_fifth_term_is_14_l1424_142479

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_second_third : a 2 + a 3 = 13

/-- The fifth term of the arithmetic sequence equals 14 -/
theorem fifth_term_is_14 (seq : ArithmeticSequence) : seq.a 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_14_l1424_142479


namespace NUMINAMATH_CALUDE_shop_equations_correct_l1424_142420

/-- Represents a shop with rooms and guests -/
structure Shop where
  rooms : ℕ
  guests : ℕ

/-- The system of equations for the shop problem -/
def shop_equations (s : Shop) : Prop :=
  (7 * s.rooms + 7 = s.guests) ∧ (9 * (s.rooms - 1) = s.guests)

/-- Theorem stating that the shop equations correctly represent the given conditions -/
theorem shop_equations_correct (s : Shop) :
  (∀ (r : ℕ), r * s.rooms + 7 = s.guests → r = 7) ∧
  (∀ (r : ℕ), r * (s.rooms - 1) = s.guests → r = 9) →
  shop_equations s :=
sorry

end NUMINAMATH_CALUDE_shop_equations_correct_l1424_142420


namespace NUMINAMATH_CALUDE_find_m_l1424_142497

theorem find_m : ∃ m : ℝ, ∀ x : ℝ, (x - 4) * (x + 3) = x^2 + m*x - 12 → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1424_142497


namespace NUMINAMATH_CALUDE_toy_boxes_theorem_l1424_142467

/-- Represents the number of toy cars in each box -/
structure ToyBoxes :=
  (box1 : ℕ)
  (box2 : ℕ)
  (box3 : ℕ)
  (box4 : ℕ)
  (box5 : ℕ)

/-- The initial state of the toy boxes -/
def initial_state : ToyBoxes :=
  { box1 := 21
  , box2 := 31
  , box3 := 19
  , box4 := 45
  , box5 := 27 }

/-- The final state of the toy boxes after moving 12 cars from box1 to box4 -/
def final_state : ToyBoxes :=
  { box1 := 9
  , box2 := 31
  , box3 := 19
  , box4 := 57
  , box5 := 27 }

/-- The number of cars moved from box1 to box4 -/
def cars_moved : ℕ := 12

theorem toy_boxes_theorem (initial : ToyBoxes) (final : ToyBoxes) (moved : ℕ) :
  initial = initial_state →
  moved = cars_moved →
  final.box1 = initial.box1 - moved ∧
  final.box2 = initial.box2 ∧
  final.box3 = initial.box3 ∧
  final.box4 = initial.box4 + moved ∧
  final.box5 = initial.box5 →
  final = final_state :=
by sorry

end NUMINAMATH_CALUDE_toy_boxes_theorem_l1424_142467


namespace NUMINAMATH_CALUDE_union_M_N_l1424_142476

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N as the difference between U and complement_U_N
def N : Set ℝ := U \ complement_U_N

-- Theorem statement
theorem union_M_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_l1424_142476


namespace NUMINAMATH_CALUDE_equation_solution_l1424_142458

theorem equation_solution (x : ℝ) :
  x ≠ 2/3 →
  ((4*x + 3) / (3*x^2 + 4*x - 4) = 3*x / (3*x - 2)) ↔
  (x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1424_142458


namespace NUMINAMATH_CALUDE_flight_cost_X_to_Y_l1424_142464

/-- Represents a city in the travel problem -/
inductive City : Type
| X : City
| Y : City
| Z : City

/-- The distance between two cities in kilometers -/
def distance : City → City → ℝ
| City.X, City.Y => 4800
| City.X, City.Z => 4000
| _, _ => 0  -- We don't need other distances for this problem

/-- The cost per kilometer for bus travel -/
def busCostPerKm : ℝ := 0.15

/-- The cost per kilometer for air travel -/
def airCostPerKm : ℝ := 0.12

/-- The booking fee for air travel -/
def airBookingFee : ℝ := 150

/-- The cost of flying between two cities -/
def flightCost (c1 c2 : City) : ℝ :=
  airBookingFee + airCostPerKm * distance c1 c2

/-- The main theorem: The cost of flying from X to Y is $726 -/
theorem flight_cost_X_to_Y : flightCost City.X City.Y = 726 := by
  sorry

end NUMINAMATH_CALUDE_flight_cost_X_to_Y_l1424_142464


namespace NUMINAMATH_CALUDE_module_stock_worth_l1424_142469

/-- Calculates the total worth of a stock of modules -/
theorem module_stock_worth (total_modules : ℕ) (cheap_modules : ℕ) (expensive_cost : ℚ) (cheap_cost : ℚ) 
  (h1 : total_modules = 22)
  (h2 : cheap_modules = 21)
  (h3 : expensive_cost = 10)
  (h4 : cheap_cost = 5/2)
  : (total_modules - cheap_modules) * expensive_cost + cheap_modules * cheap_cost = 125/2 := by
  sorry

#eval (22 - 21) * 10 + 21 * (5/2)  -- This should output 62.5

end NUMINAMATH_CALUDE_module_stock_worth_l1424_142469


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1424_142496

theorem inequality_solution_set (x : ℝ) :
  (3 * x + 2 ≥ 5) ↔ (x ≥ 1) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1424_142496


namespace NUMINAMATH_CALUDE_zoo_animal_count_l1424_142423

/-- Calculates the total number of animals in a zoo with specific enclosure arrangements --/
def total_animals_in_zoo : ℕ :=
  let tiger_enclosures := 4
  let tigers_per_enclosure := 4
  let zebra_enclosures := (tiger_enclosures / 2) * 3
  let zebras_per_enclosure := 10
  let elephant_giraffe_pattern_repetitions := 4
  let elephants_per_enclosure := 3
  let giraffes_per_enclosure := 2
  let rhino_enclosures := 5
  let rhinos_per_enclosure := 1
  let chimpanzee_enclosures := rhino_enclosures * 2
  let chimpanzees_per_enclosure := 8

  let total_tigers := tiger_enclosures * tigers_per_enclosure
  let total_zebras := zebra_enclosures * zebras_per_enclosure
  let total_elephants := elephant_giraffe_pattern_repetitions * elephants_per_enclosure
  let total_giraffes := elephant_giraffe_pattern_repetitions * 2 * giraffes_per_enclosure
  let total_rhinos := rhino_enclosures * rhinos_per_enclosure
  let total_chimpanzees := chimpanzee_enclosures * chimpanzees_per_enclosure

  total_tigers + total_zebras + total_elephants + total_giraffes + total_rhinos + total_chimpanzees

theorem zoo_animal_count : total_animals_in_zoo = 189 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l1424_142423


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1424_142418

/-- A geometric sequence with positive terms satisfying a specific condition -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r, ∀ n, a (n + 1) = r * a n) ∧
  (a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100)

/-- The sum of the 4th and 6th terms of the geometric sequence is 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) : a 4 + a 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1424_142418


namespace NUMINAMATH_CALUDE_eighth_term_is_one_thirty_second_l1424_142446

/-- The sequence defined by a_n = (-1)^n * n / 2^n -/
def a (n : ℕ) : ℚ := (-1)^n * n / 2^n

/-- The 8th term of the sequence is 1/32 -/
theorem eighth_term_is_one_thirty_second : a 8 = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_one_thirty_second_l1424_142446


namespace NUMINAMATH_CALUDE_function_property_k_value_l1424_142474

theorem function_property_k_value (f : ℝ → ℝ) (k : ℝ) 
  (h1 : f 1 = 4)
  (h2 : ∀ x y, f (x + y) = f x + f y + k * x * y + 4)
  (h3 : f 2 + f 5 = 125) :
  k = 7 := by sorry

end NUMINAMATH_CALUDE_function_property_k_value_l1424_142474


namespace NUMINAMATH_CALUDE_island_population_theorem_l1424_142456

theorem island_population_theorem (a b c d : ℝ) 
  (h1 : a / (a + b) = 0.65)  -- 65% of blue-eyed are brunettes
  (h2 : b / (b + c) = 0.7)   -- 70% of blondes have blue eyes
  (h3 : c / (c + d) = 0.1)   -- 10% of green-eyed are blondes
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) -- All populations are positive
  : d / (a + b + c + d) = 0.54 := by
  sorry

#check island_population_theorem

end NUMINAMATH_CALUDE_island_population_theorem_l1424_142456


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1424_142461

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9 ∧ x * y = 20) →
  m + n = 87 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1424_142461


namespace NUMINAMATH_CALUDE_tangent_condition_zeros_condition_l1424_142403

noncomputable section

def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 1

def g (k x : ℝ) := k * x + 1 - Real.log x

def h (k x : ℝ) := min (f x) (g k x)

def has_exactly_two_tangents (a : ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
  (∀ t : ℝ, (f t - (-4) = (6 * t^2 - 6 * t) * (t - a)) ↔ (t = t₁ ∨ t = t₂))

def has_exactly_three_zeros (k : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x : ℝ, x > 0 → (h k x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃))

theorem tangent_condition (a : ℝ) :
  has_exactly_two_tangents a ↔ (a = -1 ∨ a = 7/2) :=
sorry

theorem zeros_condition (k : ℝ) :
  has_exactly_three_zeros k ↔ (0 < k ∧ k < Real.exp (-2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_zeros_condition_l1424_142403


namespace NUMINAMATH_CALUDE_percentage_increase_l1424_142421

theorem percentage_increase (initial_earnings new_earnings : ℝ) (h1 : initial_earnings = 60) (h2 : new_earnings = 72) :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1424_142421


namespace NUMINAMATH_CALUDE_no_double_reverse_number_l1424_142494

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem: There does not exist a positive integer N such that 
    when its digits are reversed, the resulting number is exactly twice N -/
theorem no_double_reverse_number : ¬ ∃ (N : ℕ+), reverseDigits N = 2 * N := by
  sorry

end NUMINAMATH_CALUDE_no_double_reverse_number_l1424_142494


namespace NUMINAMATH_CALUDE_pillars_count_l1424_142499

/-- The length of the circular track in meters -/
def track_length : ℕ := 1200

/-- The interval between pillars in meters -/
def pillar_interval : ℕ := 30

/-- The number of pillars along the circular track -/
def num_pillars : ℕ := track_length / pillar_interval

theorem pillars_count : num_pillars = 40 := by
  sorry

end NUMINAMATH_CALUDE_pillars_count_l1424_142499


namespace NUMINAMATH_CALUDE_equation_represents_point_l1424_142400

theorem equation_represents_point :
  ∀ x y : ℝ, (Real.sqrt (x - 2) + (y + 2)^2 = 0) ↔ (x = 2 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_point_l1424_142400


namespace NUMINAMATH_CALUDE_fifteen_plus_neg_twentythree_l1424_142451

-- Define the operation for adding a positive and negative rational number
def add_pos_neg (a b : ℚ) : ℚ := -(b - a)

-- State the theorem
theorem fifteen_plus_neg_twentythree :
  15 + (-23) = add_pos_neg 15 23 :=
by sorry

end NUMINAMATH_CALUDE_fifteen_plus_neg_twentythree_l1424_142451


namespace NUMINAMATH_CALUDE_one_more_green_than_red_peaches_l1424_142484

/-- Given a basket of peaches with specified quantities of red, yellow, and green peaches,
    prove that there is one more green peach than red peaches. -/
theorem one_more_green_than_red_peaches 
  (red : ℕ) (yellow : ℕ) (green : ℕ)
  (h_red : red = 7)
  (h_yellow : yellow = 71)
  (h_green : green = 8) :
  green - red = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_more_green_than_red_peaches_l1424_142484


namespace NUMINAMATH_CALUDE_paving_cost_l1424_142463

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) : 
  length = 6.5 → 
  width = 2.75 → 
  rate = 600 → 
  length * width * rate = 10725 := by
sorry

end NUMINAMATH_CALUDE_paving_cost_l1424_142463


namespace NUMINAMATH_CALUDE_mistake_position_l1424_142452

theorem mistake_position (n : ℕ) (a₁ : ℤ) (d : ℤ) (sum : ℤ) (k : ℕ) : 
  n = 21 →
  a₁ = 51 →
  d = 5 →
  sum = 2021 →
  k ∈ Finset.range n →
  sum = (n * (2 * a₁ + (n - 1) * d)) / 2 - 10 * k →
  k = 10 :=
by sorry

end NUMINAMATH_CALUDE_mistake_position_l1424_142452


namespace NUMINAMATH_CALUDE_product_remainder_by_10_l1424_142460

theorem product_remainder_by_10 : (2583 * 7462 * 93215) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_10_l1424_142460


namespace NUMINAMATH_CALUDE_sum_770_product_not_divisible_l1424_142457

theorem sum_770_product_not_divisible (a b : ℕ) : 
  a + b = 770 → ¬(770 ∣ (a * b)) := by
sorry

end NUMINAMATH_CALUDE_sum_770_product_not_divisible_l1424_142457


namespace NUMINAMATH_CALUDE_area_enclosed_by_line_and_curve_l1424_142448

def dataSet : List ℝ := [1, 2, 0, 0, 8, 7, 6, 5]

def median (l : List ℝ) : ℝ := sorry

def areaEnclosed (a : ℝ) : ℝ := sorry

theorem area_enclosed_by_line_and_curve (a : ℝ) :
  a ∈ dataSet →
  median dataSet = 4 →
  areaEnclosed a = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_line_and_curve_l1424_142448


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l1424_142408

theorem complex_subtraction_simplification :
  (-5 + 3 * Complex.I) - (2 - 7 * Complex.I) = -7 + 10 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l1424_142408


namespace NUMINAMATH_CALUDE_pass_rate_calculation_l1424_142417

/-- Pass rate calculation for a batch of parts -/
theorem pass_rate_calculation (inspected : ℕ) (qualified : ℕ) (h1 : inspected = 40) (h2 : qualified = 38) :
  (qualified : ℚ) / (inspected : ℚ) * 100 = 95 := by
  sorry

end NUMINAMATH_CALUDE_pass_rate_calculation_l1424_142417


namespace NUMINAMATH_CALUDE_chloe_wins_l1424_142486

/-- Given that the ratio of Chloe's wins to Max's wins is 8:3 and Max won 9 times,
    prove that Chloe won 24 times. -/
theorem chloe_wins (ratio_chloe : ℕ) (ratio_max : ℕ) (max_wins : ℕ) 
    (h1 : ratio_chloe = 8)
    (h2 : ratio_max = 3)
    (h3 : max_wins = 9) : 
  (ratio_chloe * max_wins) / ratio_max = 24 := by
  sorry

#check chloe_wins

end NUMINAMATH_CALUDE_chloe_wins_l1424_142486


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1424_142430

-- Define the concept of opposite (additive inverse)
def opposite (a : ℤ) : ℤ := -a

-- State the theorem
theorem opposite_of_2023 : opposite 2023 = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1424_142430


namespace NUMINAMATH_CALUDE_fourth_root_of_1250000_l1424_142431

theorem fourth_root_of_1250000 : (1250000 : ℝ) ^ (1/4 : ℝ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_1250000_l1424_142431


namespace NUMINAMATH_CALUDE_company_j_salary_difference_l1424_142453

/-- Represents the company J with its payroll information -/
structure CompanyJ where
  factory_workers : ℕ
  office_workers : ℕ
  factory_payroll : ℕ
  office_payroll : ℕ

/-- Calculates the difference between average monthly salaries of office and factory workers -/
def salary_difference (company : CompanyJ) : ℚ :=
  (company.office_payroll / company.office_workers) - (company.factory_payroll / company.factory_workers)

/-- Theorem stating the salary difference in Company J -/
theorem company_j_salary_difference :
  ∃ (company : CompanyJ),
    company.factory_workers = 15 ∧
    company.office_workers = 30 ∧
    company.factory_payroll = 30000 ∧
    company.office_payroll = 75000 ∧
    salary_difference company = 500 := by
  sorry

end NUMINAMATH_CALUDE_company_j_salary_difference_l1424_142453


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1424_142407

-- Define the hyperbola
def Hyperbola (a b : ℝ) := {(x, y) : ℝ × ℝ | y^2 / a^2 - x^2 / b^2 = 1}

-- Define the foci
def Foci (a b : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

-- Define a point on the hyperbola
def PointOnHyperbola (a b : ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def Distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the eccentricity
def Eccentricity (a b : ℝ) : ℝ := sorry

theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let (f1x, f1y, f2x, f2y) := Foci a b
  let p := PointOnHyperbola a b
  let d1 := Distance p (f1x, f1y)
  let d2 := Distance p (f2x, f2y)
  d1 = 3 * d2 →
  let e := Eccentricity a b
  1 < e ∧ e ≤ 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1424_142407


namespace NUMINAMATH_CALUDE_tangent_slope_tan_at_pi_over_four_l1424_142489

theorem tangent_slope_tan_at_pi_over_four :
  let f : ℝ → ℝ := λ x ↦ Real.tan x
  let x₀ : ℝ := π / 4
  (deriv f) x₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_tan_at_pi_over_four_l1424_142489


namespace NUMINAMATH_CALUDE_thirteen_ceilings_left_l1424_142492

/-- Represents the number of ceilings left to paint after next week -/
def ceilings_left_after_next_week (stories : ℕ) (rooms_per_floor : ℕ) (ceilings_painted_this_week : ℕ) : ℕ :=
  let total_room_ceilings := stories * rooms_per_floor
  let total_hallway_ceilings := stories
  let total_ceilings := total_room_ceilings + total_hallway_ceilings
  let ceilings_left_after_this_week := total_ceilings - ceilings_painted_this_week
  let ceilings_to_paint_next_week := ceilings_painted_this_week / 4 + stories
  ceilings_left_after_this_week - ceilings_to_paint_next_week

/-- Theorem stating that 13 ceilings will be left to paint after next week -/
theorem thirteen_ceilings_left : ceilings_left_after_next_week 4 7 12 = 13 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_ceilings_left_l1424_142492


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l1424_142491

theorem no_solutions_for_equation : ¬∃ (x y : ℕ+), x^12 = 26*y^3 + 2023 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l1424_142491


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1424_142441

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with axes parallel to coordinate axes -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  ((p.x - e.center.x)^2 / e.semi_major_axis^2) + ((p.y - e.center.y)^2 / e.semi_minor_axis^2) = 1

theorem ellipse_minor_axis_length : 
  ∀ (e : Ellipse),
    let p1 : Point := ⟨-2, 1⟩
    let p2 : Point := ⟨0, 0⟩
    let p3 : Point := ⟨0, 3⟩
    let p4 : Point := ⟨4, 0⟩
    let p5 : Point := ⟨4, 3⟩
    (¬ collinear p1 p2 p3 ∧ ¬ collinear p1 p2 p4 ∧ ¬ collinear p1 p2 p5 ∧ 
     ¬ collinear p1 p3 p4 ∧ ¬ collinear p1 p3 p5 ∧ ¬ collinear p1 p4 p5 ∧ 
     ¬ collinear p2 p3 p4 ∧ ¬ collinear p2 p3 p5 ∧ ¬ collinear p2 p4 p5 ∧ 
     ¬ collinear p3 p4 p5) →
    (pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ pointOnEllipse p3 e ∧ 
     pointOnEllipse p4 e ∧ pointOnEllipse p5 e) →
    2 * e.semi_minor_axis = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1424_142441


namespace NUMINAMATH_CALUDE_range_of_a_l1424_142449

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then x else a * x^2 + 2 * x

/-- The range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  -1 ≤ a ∧ a ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1424_142449


namespace NUMINAMATH_CALUDE_three_bushes_same_flowers_l1424_142459

theorem three_bushes_same_flowers (garden : Finset ℕ) (flower_count : ℕ → ℕ) :
  garden.card = 201 →
  (∀ bush ∈ garden, 1 ≤ flower_count bush ∧ flower_count bush ≤ 100) →
  ∃ n : ℕ, ∃ bush₁ bush₂ bush₃ : ℕ,
    bush₁ ∈ garden ∧ bush₂ ∈ garden ∧ bush₃ ∈ garden ∧
    bush₁ ≠ bush₂ ∧ bush₁ ≠ bush₃ ∧ bush₂ ≠ bush₃ ∧
    flower_count bush₁ = n ∧ flower_count bush₂ = n ∧ flower_count bush₃ = n :=
by sorry

end NUMINAMATH_CALUDE_three_bushes_same_flowers_l1424_142459


namespace NUMINAMATH_CALUDE_specific_arrangement_probability_l1424_142444

def total_tiles : ℕ := 9
def x_tiles : ℕ := 5
def o_tiles : ℕ := 4

theorem specific_arrangement_probability :
  (1 : ℚ) / Nat.choose total_tiles x_tiles = (1 : ℚ) / 126 := by
  sorry

end NUMINAMATH_CALUDE_specific_arrangement_probability_l1424_142444


namespace NUMINAMATH_CALUDE_angle_b_measure_l1424_142485

theorem angle_b_measure (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : C = 3 * A) (h3 : B = 2 * A) : B = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_measure_l1424_142485


namespace NUMINAMATH_CALUDE_arithmetic_progression_zero_term_l1424_142480

/-- An arithmetic progression with a_{2n} / a_{2m} = -1 has a zero term at position n+m -/
theorem arithmetic_progression_zero_term 
  (a : ℕ → ℝ) 
  (h_arith : ∃ (d : ℝ), ∀ (k : ℕ), a (k + 1) = a k + d) 
  (n m : ℕ) 
  (h_ratio : a (2 * n) / a (2 * m) = -1) :
  ∃ (k : ℕ), k = n + m ∧ a k = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_zero_term_l1424_142480


namespace NUMINAMATH_CALUDE_solve_equation_l1424_142402

theorem solve_equation : 
  ∃ X : ℝ, 1.5 * ((3.6 * X * 2.50) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002 ∧ 
  X = 0.4800000000000001 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1424_142402


namespace NUMINAMATH_CALUDE_negation_equivalence_l1424_142471

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x > 0 ∧ -x^2 + 2*x - 1 > 0) ↔ 
  (∀ x : ℝ, x > 0 → -x^2 + 2*x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1424_142471


namespace NUMINAMATH_CALUDE_simplify_expression_l1424_142412

theorem simplify_expression :
  4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1424_142412


namespace NUMINAMATH_CALUDE_max_absolute_value_on_circle_l1424_142477

theorem max_absolute_value_on_circle (z : ℂ) (h : Complex.abs (z - (1 - Complex.I)) = 1) :
  Complex.abs z ≤ Real.sqrt 2 + 1 ∧ ∃ w : ℂ, Complex.abs (w - (1 - Complex.I)) = 1 ∧ Complex.abs w = Real.sqrt 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_max_absolute_value_on_circle_l1424_142477


namespace NUMINAMATH_CALUDE_solve_for_t_l1424_142450

theorem solve_for_t (s t : ℚ) (eq1 : 11 * s + 7 * t = 170) (eq2 : s = 2 * t - 3) : t = 203 / 29 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l1424_142450


namespace NUMINAMATH_CALUDE_negation_of_implication_l1424_142473

theorem negation_of_implication (x : ℝ) :
  ¬(x^2 = 1 → x = 1 ∨ x = -1) ↔ (x^2 ≠ 1 → x ≠ 1 ∧ x ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1424_142473


namespace NUMINAMATH_CALUDE_programming_contest_grouping_l1424_142438

/-- The number of programmers in the contest -/
def num_programmers : ℕ := 2008

/-- The number of rounds needed -/
def num_rounds : ℕ := 11

/-- A function that represents the grouping of programmers in each round -/
def grouping (round : ℕ) (programmer : ℕ) : Bool :=
  sorry

theorem programming_contest_grouping :
  (∀ (p1 p2 : ℕ), p1 < num_programmers → p2 < num_programmers → p1 ≠ p2 →
    ∃ (r : ℕ), r < num_rounds ∧ grouping r p1 ≠ grouping r p2) ∧
  (∀ (n : ℕ), n < num_rounds →
    ∃ (p1 p2 : ℕ), p1 < num_programmers ∧ p2 < num_programmers ∧ p1 ≠ p2 ∧
      ∀ (r : ℕ), r < n → grouping r p1 = grouping r p2) :=
sorry

end NUMINAMATH_CALUDE_programming_contest_grouping_l1424_142438


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l1424_142409

/-- A random variable following a normal distribution with mean μ and standard deviation σ -/
structure NormalRV (μ σ : ℝ) where
  X : ℝ → ℝ  -- The random variable as a function

/-- The probability that a random variable X is greater than a given value -/
noncomputable def prob_gt (X : ℝ → ℝ) (x : ℝ) : ℝ := sorry

/-- The probability that a random variable X is less than a given value -/
noncomputable def prob_lt (X : ℝ → ℝ) (x : ℝ) : ℝ := sorry

/-- Theorem: For a normal distribution N(3,1), if P(X > 2c-1) = P(X < c+3), then c = 4/3 -/
theorem normal_distribution_symmetry (c : ℝ) (X : NormalRV 3 1) :
  prob_gt X.X (2*c - 1) = prob_lt X.X (c + 3) → c = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l1424_142409


namespace NUMINAMATH_CALUDE_cocoa_powder_calculation_l1424_142440

theorem cocoa_powder_calculation (already_has : ℕ) (still_needs : ℕ) 
  (h1 : already_has = 259)
  (h2 : still_needs = 47) :
  already_has + still_needs = 306 := by
sorry

end NUMINAMATH_CALUDE_cocoa_powder_calculation_l1424_142440


namespace NUMINAMATH_CALUDE_four_liters_possible_l1424_142413

/-- Represents the state of water in two vessels -/
structure WaterState :=
  (small : ℕ)  -- Amount of water in the 3-liter vessel
  (large : ℕ)  -- Amount of water in the 5-liter vessel

/-- Represents a pouring operation between vessels -/
inductive PourOperation
  | FillSmall
  | FillLarge
  | EmptySmall
  | EmptyLarge
  | SmallToLarge
  | LargeToSmall

/-- Applies a pouring operation to a water state -/
def applyOperation (state : WaterState) (op : PourOperation) : WaterState :=
  match op with
  | PourOperation.FillSmall => ⟨3, state.large⟩
  | PourOperation.FillLarge => ⟨state.small, 5⟩
  | PourOperation.EmptySmall => ⟨0, state.large⟩
  | PourOperation.EmptyLarge => ⟨state.small, 0⟩
  | PourOperation.SmallToLarge =>
      let amount := min state.small (5 - state.large)
      ⟨state.small - amount, state.large + amount⟩
  | PourOperation.LargeToSmall =>
      let amount := min state.large (3 - state.small)
      ⟨state.small + amount, state.large - amount⟩

/-- Theorem stating that it's possible to obtain 4 liters in the 5-liter vessel -/
theorem four_liters_possible : ∃ (ops : List PourOperation),
  (ops.foldl applyOperation ⟨0, 0⟩).large = 4 :=
sorry

end NUMINAMATH_CALUDE_four_liters_possible_l1424_142413


namespace NUMINAMATH_CALUDE_crayon_production_in_four_hours_l1424_142404

/-- Represents a crayon factory with given specifications -/
structure CrayonFactory where
  colors : Nat
  crayonsPerColorPerBox : Nat
  boxesPerHour : Nat

/-- Calculates the total number of crayons produced in a given number of hours -/
def totalCrayonsProduced (factory : CrayonFactory) (hours : Nat) : Nat :=
  factory.colors * factory.crayonsPerColorPerBox * factory.boxesPerHour * hours

/-- Theorem stating that a factory with given specifications produces 160 crayons in 4 hours -/
theorem crayon_production_in_four_hours :
  ∀ (factory : CrayonFactory),
    factory.colors = 4 →
    factory.crayonsPerColorPerBox = 2 →
    factory.boxesPerHour = 5 →
    totalCrayonsProduced factory 4 = 160 :=
by sorry

end NUMINAMATH_CALUDE_crayon_production_in_four_hours_l1424_142404


namespace NUMINAMATH_CALUDE_orange_count_correct_l1424_142490

/-- The number of oranges in the box -/
def num_oranges : ℕ := 24

/-- The initial number of kiwis in the box -/
def initial_kiwis : ℕ := 30

/-- The number of kiwis added to the box -/
def added_kiwis : ℕ := 26

/-- The percentage of oranges after adding kiwis -/
def orange_percentage : ℚ := 30 / 100

theorem orange_count_correct :
  (orange_percentage * (num_oranges + initial_kiwis + added_kiwis) : ℚ) = num_oranges := by
  sorry

end NUMINAMATH_CALUDE_orange_count_correct_l1424_142490


namespace NUMINAMATH_CALUDE_expression_simplification_system_of_equations_solution_l1424_142429

-- Part 1: Simplifying the Expression
theorem expression_simplification :
  (Real.sqrt 6 - Real.sqrt (8/3)) * Real.sqrt 3 - (2 + Real.sqrt 3) * (2 - Real.sqrt 3) = Real.sqrt 2 - 1 := by
  sorry

-- Part 2: Solving the System of Equations
theorem system_of_equations_solution :
  ∃ (x y : ℝ), 2*x - 5*y = 7 ∧ 3*x + 2*y = 1 ∧ x = 1 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_system_of_equations_solution_l1424_142429


namespace NUMINAMATH_CALUDE_prize_pricing_and_quantity_l1424_142475

/-- Represents the price of a type A prize -/
def price_A : ℕ := sorry

/-- Represents the price of a type B prize -/
def price_B : ℕ := sorry

/-- The cost of one type A prize and two type B prizes -/
def cost_combination1 : ℕ := 220

/-- The cost of two type A prizes and three type B prizes -/
def cost_combination2 : ℕ := 360

/-- The total number of prizes to be purchased -/
def total_prizes : ℕ := 30

/-- The maximum total cost allowed -/
def max_total_cost : ℕ := 2300

/-- The minimum number of type A prizes that can be purchased -/
def min_type_A_prizes : ℕ := sorry

theorem prize_pricing_and_quantity :
  (price_A + 2 * price_B = cost_combination1) ∧
  (2 * price_A + 3 * price_B = cost_combination2) ∧
  (price_A = 60) ∧
  (price_B = 80) ∧
  (∀ m : ℕ, m ≥ min_type_A_prizes →
    price_A * m + price_B * (total_prizes - m) ≤ max_total_cost) ∧
  (min_type_A_prizes = 5) := by sorry

end NUMINAMATH_CALUDE_prize_pricing_and_quantity_l1424_142475


namespace NUMINAMATH_CALUDE_school_seat_cost_l1424_142424

/-- Calculates the total cost of seats with discounts applied --/
def totalCostWithDiscounts (
  rows1 : ℕ) (seats1 : ℕ) (price1 : ℕ) (discount1 : ℚ)
  (rows2 : ℕ) (seats2 : ℕ) (price2 : ℕ) (discount2 : ℚ) (extraDiscount2 : ℚ)
  (rows3 : ℕ) (seats3 : ℕ) (price3 : ℕ) (discount3 : ℚ) : ℚ :=
  let totalSeats1 := rows1 * seats1
  let totalSeats2 := rows2 * seats2
  let totalSeats3 := rows3 * seats3
  let cost1 := totalSeats1 * price1
  let cost2 := totalSeats2 * price2
  let cost3 := totalSeats3 * price3
  let discountedCost1 := cost1 * (1 - discount1 * (totalSeats1 / seats1))
  let discountedCost2 := 
    if totalSeats2 ≥ 30 then
      cost2 * (1 - discount2 * (totalSeats2 / seats2)) * (1 - extraDiscount2)
    else
      cost2 * (1 - discount2 * (totalSeats2 / seats2))
  let discountedCost3 := cost3 * (1 - discount3 * (totalSeats3 / seats3))
  discountedCost1 + discountedCost2 + discountedCost3

/-- Theorem stating the total cost for the school --/
theorem school_seat_cost : 
  totalCostWithDiscounts 10 20 60 (12/100)
                         10 15 50 (10/100) (3/100)
                         5 10 40 (8/100) = 18947.50 := by
  sorry

end NUMINAMATH_CALUDE_school_seat_cost_l1424_142424


namespace NUMINAMATH_CALUDE_problem_statement_l1424_142411

theorem problem_statement :
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)
  M = (Real.sqrt 57 - 6 * Real.sqrt 6 + 4) / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1424_142411


namespace NUMINAMATH_CALUDE_no_solution_exists_l1424_142482

theorem no_solution_exists : ¬∃ x : ℝ, 2 < 3 * x ∧ 3 * x < 4 ∧ 1 < 5 * x ∧ 5 * x < 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1424_142482


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1424_142436

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n ∧ a n > 0

/-- Three terms form an arithmetic sequence -/
def ArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticSequence (3 * a 1) ((1 / 2) * a 3) (2 * a 2) →
  (a 11 + a 13) / (a 8 + a 10) = 27 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1424_142436


namespace NUMINAMATH_CALUDE_league_games_l1424_142488

theorem league_games (n : ℕ) (h1 : n = 8) : (n.choose 2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l1424_142488


namespace NUMINAMATH_CALUDE_emily_candy_duration_l1424_142410

/-- The number of days Emily's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  (neighbors_candy + sister_candy) / daily_consumption

/-- Theorem stating that Emily's candy will last 2 days -/
theorem emily_candy_duration :
  candy_duration 5 13 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_candy_duration_l1424_142410


namespace NUMINAMATH_CALUDE_real_axis_length_l1424_142416

/-- A hyperbola with equation x²/a² - y²/b² = 1/4 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The standard hyperbola with equation x²/9 - y²/16 = 1 -/
def standard_hyperbola : Hyperbola where
  a := 3
  b := 4
  h_positive := by norm_num

theorem real_axis_length
  (C : Hyperbola)
  (h_asymptotes : C.a / C.b = standard_hyperbola.a / standard_hyperbola.b)
  (h_point : C.a^2 * 9 - C.b^2 * 12 = C.a^2 * C.b^2) :
  2 * C.a = 3 := by
  sorry

#check real_axis_length

end NUMINAMATH_CALUDE_real_axis_length_l1424_142416


namespace NUMINAMATH_CALUDE_kateDisprovesPeter_l1424_142481

/-- Represents a card with a character on one side and a natural number on the other -/
structure Card where
  letter : Char
  number : Nat

/-- Checks if a given character is a vowel -/
def isVowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']

/-- Checks if a given natural number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- Represents Peter's statement about vowels and even numbers -/
def petersStatement (c : Card) : Bool :=
  isVowel c.letter → isEven c.number

/-- The set of cards on the table -/
def cardsOnTable : List Card := [
  ⟨'A', 0⟩,  -- Placeholder number
  ⟨'B', 0⟩,  -- Placeholder number
  ⟨'C', 1⟩,  -- Assuming 'C' for the third card
  ⟨'D', 7⟩,  -- The fourth card we know about
  ⟨'U', 0⟩   -- Placeholder number
]

theorem kateDisprovesPeter :
  ∃ (c : Card), c ∈ cardsOnTable ∧ ¬(petersStatement c) ∧ c.number = 7 := by
  sorry

#check kateDisprovesPeter

end NUMINAMATH_CALUDE_kateDisprovesPeter_l1424_142481


namespace NUMINAMATH_CALUDE_fourth_proportional_l1424_142443

theorem fourth_proportional (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x : ℝ, x > 0 ∧ a * x = b * c := by
sorry

end NUMINAMATH_CALUDE_fourth_proportional_l1424_142443


namespace NUMINAMATH_CALUDE_participant_age_l1424_142406

/-- Represents the initial state of the lecture rooms -/
structure LectureRooms where
  room1_count : ℕ
  room1_avg_age : ℕ
  room2_count : ℕ
  room2_avg_age : ℕ

/-- Calculates the total age sum of all participants -/
def total_age_sum (rooms : LectureRooms) : ℕ :=
  rooms.room1_count * rooms.room1_avg_age + rooms.room2_count * rooms.room2_avg_age

/-- Calculates the total number of participants -/
def total_count (rooms : LectureRooms) : ℕ :=
  rooms.room1_count + rooms.room2_count

/-- Theorem stating the age of the participant who left -/
theorem participant_age (rooms : LectureRooms) 
  (h1 : rooms.room1_count = 8)
  (h2 : rooms.room1_avg_age = 20)
  (h3 : rooms.room2_count = 12)
  (h4 : rooms.room2_avg_age = 45)
  (h5 : (total_age_sum rooms - x) / (total_count rooms - 1) = (total_age_sum rooms) / (total_count rooms) + 1) :
  x = 16 :=
sorry


end NUMINAMATH_CALUDE_participant_age_l1424_142406


namespace NUMINAMATH_CALUDE_least_common_addition_primes_l1424_142470

theorem least_common_addition_primes (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x < y → x + y = 36 → 4 * x + y = 51 := by
  sorry

end NUMINAMATH_CALUDE_least_common_addition_primes_l1424_142470


namespace NUMINAMATH_CALUDE_restaurant_hotdogs_l1424_142468

theorem restaurant_hotdogs (hotdogs : ℕ) (pizzas : ℕ) : 
  pizzas = hotdogs + 40 →
  30 * (hotdogs + pizzas) = 4800 →
  hotdogs = 60 := by
sorry

end NUMINAMATH_CALUDE_restaurant_hotdogs_l1424_142468


namespace NUMINAMATH_CALUDE_chemical_mixture_volume_l1424_142445

theorem chemical_mixture_volume : 
  ∀ (initial_volume : ℝ),
  initial_volume > 0 →
  0.30 * initial_volume + 20 = 0.44 * (initial_volume + 20) →
  initial_volume = 80 :=
λ initial_volume h_positive h_equation =>
  sorry

end NUMINAMATH_CALUDE_chemical_mixture_volume_l1424_142445


namespace NUMINAMATH_CALUDE_min_subsets_to_guess_l1424_142455

/-- The set of possible choices for player A -/
def S : Set Nat := Finset.range 1001

/-- The condition that ensures B can always guess correctly -/
def can_guess (k₁ k₂ k₃ : Nat) : Prop :=
  (k₁ + 1) * (k₂ + 1) * (k₃ + 1) ≥ 1001

/-- The sum of subsets chosen by B -/
def total_subsets (k₁ k₂ k₃ : Nat) : Nat :=
  k₁ + k₂ + k₃

/-- The theorem stating that 28 is the minimum value -/
theorem min_subsets_to_guess :
  ∃ k₁ k₂ k₃ : Nat,
    can_guess k₁ k₂ k₃ ∧
    total_subsets k₁ k₂ k₃ = 28 ∧
    ∀ k₁' k₂' k₃' : Nat,
      can_guess k₁' k₂' k₃' →
      total_subsets k₁' k₂' k₃' ≥ 28 :=
sorry

end NUMINAMATH_CALUDE_min_subsets_to_guess_l1424_142455


namespace NUMINAMATH_CALUDE_simplify_fraction_ratio_l1424_142433

theorem simplify_fraction_ratio (k : ℤ) : 
  ∃ (a b : ℤ), (4*k + 8) / 4 = a*k + b ∧ a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_ratio_l1424_142433


namespace NUMINAMATH_CALUDE_principal_square_root_nine_sixteenths_l1424_142454

theorem principal_square_root_nine_sixteenths (x : ℝ) : x = Real.sqrt (9 / 16) → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_principal_square_root_nine_sixteenths_l1424_142454


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1424_142439

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ),
    (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
      (x^2 + 5*x - 2) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1)) ∧
    A = 2 ∧ B = -1 ∧ C = 5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1424_142439


namespace NUMINAMATH_CALUDE_barbie_coconuts_l1424_142462

theorem barbie_coconuts (total_coconuts : ℕ) (trips : ℕ) (bruno_capacity : ℕ) 
  (h1 : total_coconuts = 144)
  (h2 : trips = 12)
  (h3 : bruno_capacity = 8) :
  ∃ barbie_capacity : ℕ, 
    barbie_capacity * trips + bruno_capacity * trips = total_coconuts ∧ 
    barbie_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_barbie_coconuts_l1424_142462


namespace NUMINAMATH_CALUDE_circle_in_rectangle_l1424_142427

theorem circle_in_rectangle (r x : ℝ) : 
  r > 0 →  -- radius is positive
  2 * r = x →  -- width of rectangle is diameter of circle
  r + (2 * x) / 3 + r = 10 →  -- length of rectangle
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_in_rectangle_l1424_142427


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1424_142465

theorem hyperbola_focal_length (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m - y^2 / 4 = 1) →  -- Equation of the hyperbola
  (∃ c : ℝ, c = 3) →                    -- Focal length is 6 (2c = 6, so c = 3)
  (∃ a b : ℝ, a^2 = m ∧ b^2 = 4 ∧ c^2 = a^2 + b^2) →  -- Relationship between a, b, c
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1424_142465


namespace NUMINAMATH_CALUDE_money_left_l1424_142466

/-- The amount of money Mrs. Hilt had initially, in cents. -/
def initial_amount : ℕ := 15

/-- The cost of the pencil, in cents. -/
def pencil_cost : ℕ := 11

/-- The theorem stating how much money Mrs. Hilt had left after buying the pencil. -/
theorem money_left : initial_amount - pencil_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l1424_142466


namespace NUMINAMATH_CALUDE_trees_in_park_l1424_142432

/-- The number of trees after n years, given an initial number and annual growth rate. -/
def trees_after_years (initial : ℕ) (growth_rate : ℚ) (years : ℕ) : ℚ :=
  initial * (1 + growth_rate) ^ years

/-- Theorem stating that given 5000 trees initially and 30% annual growth,
    the number of trees after 3 years is 10985. -/
theorem trees_in_park (initial : ℕ) (growth_rate : ℚ) (years : ℕ) 
  (h_initial : initial = 5000)
  (h_growth : growth_rate = 3/10)
  (h_years : years = 3) :
  trees_after_years initial growth_rate years = 10985 := by
  sorry

#eval trees_after_years 5000 (3/10) 3

end NUMINAMATH_CALUDE_trees_in_park_l1424_142432


namespace NUMINAMATH_CALUDE_sin_cos_power_six_sum_one_l1424_142493

theorem sin_cos_power_six_sum_one (α : Real) (h : Real.sin α + Real.cos α = 1) :
  Real.sin α ^ 6 + Real.cos α ^ 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_power_six_sum_one_l1424_142493


namespace NUMINAMATH_CALUDE_geometric_sequence_shift_l1424_142483

theorem geometric_sequence_shift (a : ℕ → ℝ) (q c : ℝ) :
  (q ≠ 1) →
  (∀ n, a (n + 1) = q * a n) →
  (∃ r, ∀ n, (a (n + 1) + c) = r * (a n + c)) →
  c = 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_shift_l1424_142483


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_three_l1424_142437

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {3, 4, 2*a - 4}
def B (a : ℝ) : Set ℝ := {a}

-- State the theorem
theorem intersection_implies_a_equals_three (a : ℝ) :
  (A a ∩ B a).Nonempty → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_three_l1424_142437


namespace NUMINAMATH_CALUDE_division_problem_l1424_142495

theorem division_problem (x y : ℕ+) 
  (h1 : x = 10 * y + 3)
  (h2 : 2 * x = 21 * y + 1) :
  11 * y - x = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1424_142495


namespace NUMINAMATH_CALUDE_john_hourly_rate_is_20_l1424_142419

/-- Represents John's car repair scenario -/
structure CarRepairScenario where
  total_cars : ℕ
  standard_repair_time : ℕ  -- in minutes
  longer_repair_factor : ℚ  -- factor for longer repair time
  standard_repair_count : ℕ
  total_earnings : ℕ        -- in dollars

/-- Calculates John's hourly rate given the car repair scenario -/
def calculate_hourly_rate (scenario : CarRepairScenario) : ℚ :=
  -- Function body to be implemented
  sorry

/-- Theorem stating that John's hourly rate is $20 -/
theorem john_hourly_rate_is_20 (scenario : CarRepairScenario) 
  (h1 : scenario.total_cars = 5)
  (h2 : scenario.standard_repair_time = 40)
  (h3 : scenario.longer_repair_factor = 3/2)
  (h4 : scenario.standard_repair_count = 3)
  (h5 : scenario.total_earnings = 80) :
  calculate_hourly_rate scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_hourly_rate_is_20_l1424_142419


namespace NUMINAMATH_CALUDE_power_45_equals_a_squared_b_l1424_142405

theorem power_45_equals_a_squared_b (x a b : ℝ) (h1 : 3^x = a) (h2 : 5^x = b) : 45^x = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_power_45_equals_a_squared_b_l1424_142405


namespace NUMINAMATH_CALUDE_fraction_equality_l1424_142415

theorem fraction_equality (a b : ℝ) (h : (1/a + 1/b)/(1/a - 1/b) = 2023) : (a + b)/(a - b) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1424_142415


namespace NUMINAMATH_CALUDE_finite_solutions_factorial_cube_plus_eight_l1424_142487

theorem finite_solutions_factorial_cube_plus_eight :
  {p : ℕ × ℕ | (p.1.factorial = p.2^3 + 8)}.Finite := by
  sorry

end NUMINAMATH_CALUDE_finite_solutions_factorial_cube_plus_eight_l1424_142487


namespace NUMINAMATH_CALUDE_set_operation_result_l1424_142422

def A : Set ℕ := {1, 3, 4, 5}
def B : Set ℕ := {2, 4, 6}
def C : Set ℕ := {0, 1, 2, 3, 4}

theorem set_operation_result : (A ∪ B) ∩ C = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l1424_142422


namespace NUMINAMATH_CALUDE_peppers_total_weight_l1424_142435

theorem peppers_total_weight (green : Real) (red : Real) (yellow : Real) (jalapeno : Real) (habanero : Real)
  (h1 : green = 1.45)
  (h2 : red = 0.68)
  (h3 : yellow = 1.6)
  (h4 : jalapeno = 2.25)
  (h5 : habanero = 3.2) :
  green + red + yellow + jalapeno + habanero = 9.18 := by
  sorry

end NUMINAMATH_CALUDE_peppers_total_weight_l1424_142435


namespace NUMINAMATH_CALUDE_exam_grading_problem_l1424_142401

theorem exam_grading_problem (X : ℝ) 
  (monday_graded : X * 0.6 = X - (X * 0.4))
  (tuesday_graded : X * 0.4 * 0.75 = X * 0.4 - (X * 0.1))
  (wednesday_remaining : X * 0.1 = 12) :
  X = 120 := by
sorry

end NUMINAMATH_CALUDE_exam_grading_problem_l1424_142401


namespace NUMINAMATH_CALUDE_emilys_spending_l1424_142426

/-- Given Emily's spending pattern over four days and the total amount spent,
    prove that the amount she spent on Friday is equal to the total divided by 18. -/
theorem emilys_spending (X Y : ℝ) : 
  X > 0 →  -- Assuming X is positive
  Y > 0 →  -- Assuming Y is positive
  X + 2*X + 3*X + 4*(3*X) = Y →  -- Total spending equation
  X = Y / 18 := by
sorry

end NUMINAMATH_CALUDE_emilys_spending_l1424_142426


namespace NUMINAMATH_CALUDE_university_size_l1424_142472

/-- Represents the total number of students in a university --/
def total_students (sample_size : ℕ) (other_grades_sample : ℕ) (other_grades_total : ℕ) : ℕ :=
  (other_grades_total * sample_size) / other_grades_sample

/-- Theorem stating the total number of students in the university --/
theorem university_size :
  let sample_size : ℕ := 500
  let freshmen_sample : ℕ := 200
  let sophomore_sample : ℕ := 100
  let other_grades_sample : ℕ := sample_size - freshmen_sample - sophomore_sample
  let other_grades_total : ℕ := 3000
  total_students sample_size other_grades_sample other_grades_total = 7500 := by
  sorry

end NUMINAMATH_CALUDE_university_size_l1424_142472


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l1424_142428

def f (x : ℝ) := (x - 2)^2

theorem f_derivative_at_one : 
  deriv f 1 = -2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l1424_142428


namespace NUMINAMATH_CALUDE_mistaken_calculation_correction_l1424_142414

theorem mistaken_calculation_correction (x : ℝ) : 
  x * 4 = 166.08 → x / 4 = 10.38 := by
sorry

end NUMINAMATH_CALUDE_mistaken_calculation_correction_l1424_142414


namespace NUMINAMATH_CALUDE_max_round_value_l1424_142447

/-- Represents a digit assignment for the letter puzzle --/
structure DigitAssignment where
  H : Fin 10
  M : Fin 10
  T : Fin 10
  G : Fin 10
  U : Fin 10
  S : Fin 10
  R : Fin 10
  O : Fin 10
  N : Fin 10
  D : Fin 10

/-- Checks if all digits in the assignment are distinct --/
def allDistinct (a : DigitAssignment) : Prop :=
  a.H ≠ a.M ∧ a.H ≠ a.T ∧ a.H ≠ a.G ∧ a.H ≠ a.U ∧ a.H ≠ a.S ∧ a.H ≠ a.R ∧ a.H ≠ a.O ∧ a.H ≠ a.N ∧ a.H ≠ a.D ∧
  a.M ≠ a.T ∧ a.M ≠ a.G ∧ a.M ≠ a.U ∧ a.M ≠ a.S ∧ a.M ≠ a.R ∧ a.M ≠ a.O ∧ a.M ≠ a.N ∧ a.M ≠ a.D ∧
  a.T ≠ a.G ∧ a.T ≠ a.U ∧ a.T ≠ a.S ∧ a.T ≠ a.R ∧ a.T ≠ a.O ∧ a.T ≠ a.N ∧ a.T ≠ a.D ∧
  a.G ≠ a.U ∧ a.G ≠ a.S ∧ a.G ≠ a.R ∧ a.G ≠ a.O ∧ a.G ≠ a.N ∧ a.G ≠ a.D ∧
  a.U ≠ a.S ∧ a.U ≠ a.R ∧ a.U ≠ a.O ∧ a.U ≠ a.N ∧ a.U ≠ a.D ∧
  a.S ≠ a.R ∧ a.S ≠ a.O ∧ a.S ≠ a.N ∧ a.S ≠ a.D ∧
  a.R ≠ a.O ∧ a.R ≠ a.N ∧ a.R ≠ a.D ∧
  a.O ≠ a.N ∧ a.O ≠ a.D ∧
  a.N ≠ a.D

/-- Checks if the equation HMMT + GUTS = ROUND is satisfied --/
def equationSatisfied (a : DigitAssignment) : Prop :=
  1000 * a.H.val + 100 * a.M.val + 10 * a.M.val + a.T.val +
  1000 * a.G.val + 100 * a.U.val + 10 * a.T.val + a.S.val =
  10000 * a.R.val + 1000 * a.O.val + 100 * a.U.val + 10 * a.N.val + a.D.val

/-- Checks if there are no leading zeroes --/
def noLeadingZeroes (a : DigitAssignment) : Prop :=
  a.H ≠ 0 ∧ a.G ≠ 0 ∧ a.R ≠ 0

/-- The value of ROUND for a given digit assignment --/
def roundValue (a : DigitAssignment) : ℕ :=
  10000 * a.R.val + 1000 * a.O.val + 100 * a.U.val + 10 * a.N.val + a.D.val

/-- The main theorem statement --/
theorem max_round_value :
  ∀ a : DigitAssignment,
    allDistinct a →
    equationSatisfied a →
    noLeadingZeroes a →
    roundValue a ≤ 16352 :=
sorry

end NUMINAMATH_CALUDE_max_round_value_l1424_142447


namespace NUMINAMATH_CALUDE_third_blue_after_fifth_probability_l1424_142442

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 5

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles

/-- The probability of drawing the third blue marble after the fifth draw -/
def probability_third_blue_after_fifth : ℚ := 23 / 28

theorem third_blue_after_fifth_probability :
  probability_third_blue_after_fifth = 
    (Nat.choose 5 2 * Nat.choose 3 1 + 
     Nat.choose 5 1 * Nat.choose 3 2 + 
     Nat.choose 5 0 * Nat.choose 3 3) / 
    Nat.choose total_marbles blue_marbles :=
by sorry

end NUMINAMATH_CALUDE_third_blue_after_fifth_probability_l1424_142442


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l1424_142434

/-- Calculates the total weight of a zinc-copper mixture given the ratio and zinc weight -/
theorem zinc_copper_mixture_weight 
  (zinc_ratio : ℚ) 
  (copper_ratio : ℚ) 
  (zinc_weight : ℚ) 
  (h1 : zinc_ratio = 9) 
  (h2 : copper_ratio = 11) 
  (h3 : zinc_weight = 26.1) : 
  zinc_weight + (copper_ratio / zinc_ratio) * zinc_weight = 58 := by
sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l1424_142434


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1424_142425

/-- The quadratic function f(x) = ax² + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

/-- The maximum value of f(x) on the interval [-3, 2] -/
def max_value : ℝ := 9

/-- The lower bound of the interval -/
def lower_bound : ℝ := -3

/-- The upper bound of the interval -/
def upper_bound : ℝ := 2

theorem quadratic_max_value (a : ℝ) :
  (∀ x, lower_bound ≤ x ∧ x ≤ upper_bound → f a x ≤ max_value) ∧
  (∃ x, lower_bound ≤ x ∧ x ≤ upper_bound ∧ f a x = max_value) →
  a = 1 ∨ a = -8 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1424_142425
