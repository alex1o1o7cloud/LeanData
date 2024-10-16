import Mathlib

namespace NUMINAMATH_CALUDE_sum_greater_two_necessary_not_sufficient_l2885_288597

theorem sum_greater_two_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2) ∧
  (∃ a b : ℝ, a + b > 2 ∧ (a ≤ 1 ∨ b ≤ 1)) :=
sorry

end NUMINAMATH_CALUDE_sum_greater_two_necessary_not_sufficient_l2885_288597


namespace NUMINAMATH_CALUDE_sum_first_two_terms_l2885_288559

/-- A geometric sequence with third term 12 and fourth term 18 -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), q ≠ 0 ∧ (∀ n, a (n + 1) = a n * q) ∧ a 3 = 12 ∧ a 4 = 18

/-- The sum of the first and second terms of the geometric sequence is 40/3 -/
theorem sum_first_two_terms (a : ℕ → ℚ) (h : GeometricSequence a) :
  a 1 + a 2 = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_two_terms_l2885_288559


namespace NUMINAMATH_CALUDE_circus_performers_standing_time_l2885_288552

/-- The combined time that Pulsar, Polly, and Petra stand on their back legs is 45 minutes. -/
theorem circus_performers_standing_time : 
  let pulsar_time : ℕ := 10
  let polly_time : ℕ := 3 * pulsar_time
  let petra_time : ℕ := polly_time / 6
  pulsar_time + polly_time + petra_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_circus_performers_standing_time_l2885_288552


namespace NUMINAMATH_CALUDE_polygon_sides_sum_l2885_288584

/-- The sum of interior angles of a convex polygon with n sides --/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The number of sides in a triangle --/
def triangle_sides : ℕ := 3

/-- The number of sides in a hexagon --/
def hexagon_sides : ℕ := 6

/-- The sum of interior angles of the given polygons --/
def total_angle_sum : ℝ := 1260

theorem polygon_sides_sum :
  ∃ (n : ℕ), n = triangle_sides + hexagon_sides ∧
  total_angle_sum = interior_angle_sum triangle_sides + interior_angle_sum hexagon_sides + interior_angle_sum (n - triangle_sides - hexagon_sides) :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_sum_l2885_288584


namespace NUMINAMATH_CALUDE_pastry_distribution_l2885_288585

/-- The number of pastries the Hatter initially had -/
def total_pastries : ℕ := 32

/-- The fraction of pastries March Hare ate -/
def march_hare_fraction : ℚ := 5/16

/-- The fraction of remaining pastries Dormouse ate -/
def dormouse_fraction : ℚ := 7/11

/-- The number of pastries left for the Hatter -/
def hatter_leftover : ℕ := 8

/-- The number of pastries March Hare ate -/
def march_hare_eaten : ℕ := 10

/-- The number of pastries Dormouse ate -/
def dormouse_eaten : ℕ := 14

theorem pastry_distribution :
  (march_hare_eaten = (total_pastries : ℚ) * march_hare_fraction) ∧
  (dormouse_eaten = ((total_pastries - march_hare_eaten) : ℚ) * dormouse_fraction) ∧
  (hatter_leftover = total_pastries - march_hare_eaten - dormouse_eaten) :=
by sorry

end NUMINAMATH_CALUDE_pastry_distribution_l2885_288585


namespace NUMINAMATH_CALUDE_electricity_bill_theorem_l2885_288516

/-- Represents a meter reading with three tariff zones -/
structure MeterReading where
  peak : ℝ
  night : ℝ
  half_peak : ℝ

/-- Represents tariff rates for electricity -/
structure TariffRates where
  peak : ℝ
  night : ℝ
  half_peak : ℝ

/-- Calculates the electricity bill based on meter readings and tariff rates -/
def calculate_bill (previous : MeterReading) (current : MeterReading) (rates : TariffRates) : ℝ :=
  (current.peak - previous.peak) * rates.peak +
  (current.night - previous.night) * rates.night +
  (current.half_peak - previous.half_peak) * rates.half_peak

/-- Theorem: Maximum additional payment and expected difference -/
theorem electricity_bill_theorem 
  (previous : MeterReading)
  (current : MeterReading)
  (rates : TariffRates)
  (actual_payment : ℝ)
  (h1 : rates.peak = 4.03)
  (h2 : rates.night = 1.01)
  (h3 : rates.half_peak = 3.39)
  (h4 : actual_payment = 660.72)
  (h5 : current.peak > previous.peak)
  (h6 : current.night > previous.night)
  (h7 : current.half_peak > previous.half_peak) :
  ∃ (max_additional_payment expected_difference : ℝ),
    max_additional_payment = 397.34 ∧
    expected_difference = 19.30 :=
sorry

end NUMINAMATH_CALUDE_electricity_bill_theorem_l2885_288516


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2885_288568

theorem sqrt_equation_solution : 
  ∃ x : ℝ, x > 0 ∧ Real.sqrt 289 - Real.sqrt 625 / Real.sqrt x = 12 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2885_288568


namespace NUMINAMATH_CALUDE_max_a_value_l2885_288595

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 3 -/
def LineEquation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 3

/-- Predicate for a line not passing through any lattice point in the given range -/
def NoLatticePointIntersection (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 150 → LatticePoint x y → ¬LineEquation m x y

/-- The theorem statement -/
theorem max_a_value :
  (∀ m : ℚ, 1/3 < m → m < 50/149 → NoLatticePointIntersection m) ∧
  ¬(∀ m : ℚ, 1/3 < m → m < 50/149 + ε → NoLatticePointIntersection m) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2885_288595


namespace NUMINAMATH_CALUDE_right_triangle_properties_l2885_288515

theorem right_triangle_properties (A B C : ℝ) (h_right : A = 90) (h_tan : Real.tan C = 5) (h_hypotenuse : A = 80) :
  let AB := 80 * (5 / Real.sqrt 26)
  let BC := 80 / Real.sqrt 26
  (AB = 80 * (5 / Real.sqrt 26)) ∧ (BC / AB = 1 / 5) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l2885_288515


namespace NUMINAMATH_CALUDE_equation_set_solution_inequality_set_solution_l2885_288549

-- Equation set
theorem equation_set_solution :
  ∃! (x y : ℝ), x - y - 1 = 4 ∧ 4 * (x - y) - y = 5 ∧ x = 20 ∧ y = 15 := by sorry

-- Inequality set
theorem inequality_set_solution :
  ∀ x : ℝ, (4 * x - 1 ≥ x + 1 ∧ (1 - x) / 2 < x) ↔ x ≥ 2/3 := by sorry

end NUMINAMATH_CALUDE_equation_set_solution_inequality_set_solution_l2885_288549


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2885_288551

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (a*x + b)^2) ↔ k = 100 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2885_288551


namespace NUMINAMATH_CALUDE_income_comparison_l2885_288570

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = 0.4 * juan)
  (h2 : mary = 0.6400000000000001 * juan) :
  (mary - tim) / tim = 0.6 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l2885_288570


namespace NUMINAMATH_CALUDE_polynomial_roots_imply_a_ge_5_l2885_288528

theorem polynomial_roots_imply_a_ge_5 (a b c : ℤ) (ha : a > 0) 
  (h_roots : ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
    a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) : 
  a ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_imply_a_ge_5_l2885_288528


namespace NUMINAMATH_CALUDE_student_competition_theorem_l2885_288583

/-- The number of ways students can sign up for competitions -/
def signup_ways (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_competitions ^ num_students

/-- The number of possible outcomes for championship winners -/
def championship_outcomes (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_students ^ num_competitions

/-- Theorem stating the correct number of ways for signup and championship outcomes -/
theorem student_competition_theorem :
  let num_students : ℕ := 5
  let num_competitions : ℕ := 4
  signup_ways num_students num_competitions = 4^5 ∧
  championship_outcomes num_students num_competitions = 5^4 := by
  sorry

end NUMINAMATH_CALUDE_student_competition_theorem_l2885_288583


namespace NUMINAMATH_CALUDE_triangle_problem_l2885_288563

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  a = 2 * Real.sin B / Real.sqrt 3 →
  a = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2885_288563


namespace NUMINAMATH_CALUDE_exists_skew_line_l2885_288598

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of a line intersecting a plane
variable (intersects : Line → Plane → Prop)

-- Define the property of a line being in a plane
variable (inPlane : Line → Plane → Prop)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem exists_skew_line 
  (l : Line) (α : Plane) 
  (h : intersects l α) : 
  ∃ m : Line, inPlane m α ∧ skew l m :=
sorry

end NUMINAMATH_CALUDE_exists_skew_line_l2885_288598


namespace NUMINAMATH_CALUDE_soccer_balls_in_bag_l2885_288573

theorem soccer_balls_in_bag (initial_balls : ℕ) (additional_balls : ℕ) : 
  initial_balls = 6 → additional_balls = 18 → initial_balls + additional_balls = 24 :=
by sorry

end NUMINAMATH_CALUDE_soccer_balls_in_bag_l2885_288573


namespace NUMINAMATH_CALUDE_kyle_caught_14_fish_l2885_288578

/-- The number of fish Kyle caught given the conditions of the problem -/
def kyles_fish (total : ℕ) (carlas : ℕ) : ℕ :=
  (total - carlas) / 2

/-- Theorem stating that Kyle caught 14 fish under the given conditions -/
theorem kyle_caught_14_fish (total : ℕ) (carlas : ℕ) 
  (h1 : total = 36) 
  (h2 : carlas = 8) : 
  kyles_fish total carlas = 14 := by
  sorry

#eval kyles_fish 36 8

end NUMINAMATH_CALUDE_kyle_caught_14_fish_l2885_288578


namespace NUMINAMATH_CALUDE_most_probable_hits_l2885_288546

-- Define the parameters
def n : ℕ := 5
def p : ℝ := 0.6

-- Define the binomial probability mass function
def binomialPMF (k : ℕ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem most_probable_hits :
  ∃ (k : ℕ), k ≤ n ∧ 
  (∀ (j : ℕ), j ≤ n → binomialPMF j ≤ binomialPMF k) ∧
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_most_probable_hits_l2885_288546


namespace NUMINAMATH_CALUDE_two_machines_total_copies_l2885_288558

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  rate : ℕ  -- copies per minute

/-- Calculates the total number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Theorem: Two copy machines working together for 30 minutes will produce 3000 copies -/
theorem two_machines_total_copies 
  (machine1 : CopyMachine) 
  (machine2 : CopyMachine) 
  (h1 : machine1.rate = 35) 
  (h2 : machine2.rate = 65) : 
  copies_made machine1 30 + copies_made machine2 30 = 3000 := by
  sorry

#check two_machines_total_copies

end NUMINAMATH_CALUDE_two_machines_total_copies_l2885_288558


namespace NUMINAMATH_CALUDE_expand_product_l2885_288500

theorem expand_product (x : ℝ) : (2*x + 3) * (x + 10) = 2*x^2 + 23*x + 30 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2885_288500


namespace NUMINAMATH_CALUDE_sum_of_powers_of_three_l2885_288503

theorem sum_of_powers_of_three : (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_three_l2885_288503


namespace NUMINAMATH_CALUDE_ohara_triple_49_16_l2885_288594

/-- Definition of O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: If (49, 16, x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_49_16 (x : ℕ) :
  is_ohara_triple 49 16 x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_16_l2885_288594


namespace NUMINAMATH_CALUDE_original_cube_side_length_l2885_288572

/-- Given a cube of side length s that is painted and cut into smaller cubes of side 3,
    if there are exactly 12 smaller cubes with paint on 2 sides, then s = 6 -/
theorem original_cube_side_length (s : ℕ) : 
  s > 0 →  -- ensure the side length is positive
  (12 * (s / 3 - 1) = 12) →  -- condition for 12 smaller cubes with paint on 2 sides
  s = 6 := by
sorry

end NUMINAMATH_CALUDE_original_cube_side_length_l2885_288572


namespace NUMINAMATH_CALUDE_debt_calculation_l2885_288539

theorem debt_calculation (initial_debt additional_borrowing : ℕ) :
  initial_debt = 20 →
  additional_borrowing = 15 →
  initial_debt + additional_borrowing = 35 :=
by sorry

end NUMINAMATH_CALUDE_debt_calculation_l2885_288539


namespace NUMINAMATH_CALUDE_sqrt_30_between_5_and_6_l2885_288532

theorem sqrt_30_between_5_and_6 : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_30_between_5_and_6_l2885_288532


namespace NUMINAMATH_CALUDE_remainder_2519_div_9_l2885_288575

theorem remainder_2519_div_9 : 2519 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_9_l2885_288575


namespace NUMINAMATH_CALUDE_tangent_circles_m_value_l2885_288588

/-- The value of m for which the circle x² + y² = 1 is tangent to the circle x² + y² + 6x - 8y + m = 0 -/
theorem tangent_circles_m_value : ∃ m : ℝ, 
  (∀ x y : ℝ, x^2 + y^2 = 1 → x^2 + y^2 + 6*x - 8*y + m = 0 → 
    (x + 3)^2 + (y - 4)^2 = 5^2 ∨ (x + 3)^2 + (y - 4)^2 = 4^2) ∧
  (m = -11 ∨ m = 9) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_m_value_l2885_288588


namespace NUMINAMATH_CALUDE_no_negative_exponents_l2885_288560

theorem no_negative_exponents (a b c d : Int) 
  (h1 : (5 : ℝ)^a + (5 : ℝ)^b = (3 : ℝ)^c + (3 : ℝ)^d)
  (h2 : Even a) (h3 : Even b) (h4 : Even c) (h5 : Even d) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_negative_exponents_l2885_288560


namespace NUMINAMATH_CALUDE_bus_ride_distance_l2885_288521

theorem bus_ride_distance 
  (total_time : ℝ) 
  (bus_speed : ℝ) 
  (walking_speed : ℝ) 
  (h1 : total_time = 8) 
  (h2 : bus_speed = 9) 
  (h3 : walking_speed = 3) : 
  ∃ d : ℝ, d = 18 ∧ d / bus_speed + d / walking_speed = total_time :=
by
  sorry

end NUMINAMATH_CALUDE_bus_ride_distance_l2885_288521


namespace NUMINAMATH_CALUDE_store_display_cans_l2885_288524

/-- Represents the number of cans in each layer of the display -/
def canSequence : ℕ → ℚ
  | 0 => 30
  | n + 1 => canSequence n - 3

/-- The number of layers in the display -/
def numLayers : ℕ := 11

/-- The total number of cans in the display -/
def totalCans : ℚ := (numLayers : ℚ) * (canSequence 0 + canSequence (numLayers - 1)) / 2

theorem store_display_cans : totalCans = 170.5 := by
  sorry

end NUMINAMATH_CALUDE_store_display_cans_l2885_288524


namespace NUMINAMATH_CALUDE_average_marks_l2885_288530

theorem average_marks (num_subjects : ℕ) (avg_five : ℝ) (sixth_mark : ℝ) :
  num_subjects = 6 →
  avg_five = 74 →
  sixth_mark = 98 →
  ((avg_five * 5 + sixth_mark) / num_subjects : ℝ) = 78 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l2885_288530


namespace NUMINAMATH_CALUDE_product_xy_is_eight_l2885_288534

theorem product_xy_is_eight (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 64)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(4*y) = 256) : 
  x * y = 8 := by
sorry

end NUMINAMATH_CALUDE_product_xy_is_eight_l2885_288534


namespace NUMINAMATH_CALUDE_four_digit_sum_reverse_equals_4983_l2885_288566

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_sum_reverse_equals_4983 :
  ∃ (n : ℕ), is_four_digit n ∧ n + reverse_number n = 4983 :=
sorry

end NUMINAMATH_CALUDE_four_digit_sum_reverse_equals_4983_l2885_288566


namespace NUMINAMATH_CALUDE_expression_one_equality_l2885_288501

theorem expression_one_equality : 2 - (-4) + 8 / (-2) + (-3) = -1 := by sorry

end NUMINAMATH_CALUDE_expression_one_equality_l2885_288501


namespace NUMINAMATH_CALUDE_stating_comprehensive_investigation_is_census_l2885_288590

/-- Represents a comprehensive investigation. -/
structure ComprehensiveInvestigation where
  subject : String
  purpose : String

/-- Defines what a census is. -/
def Census : Type := ComprehensiveInvestigation

/-- 
Theorem stating that a comprehensive investigation on the subject of examination 
for a specific purpose is equivalent to a census.
-/
theorem comprehensive_investigation_is_census 
  (investigation : ComprehensiveInvestigation) 
  (h1 : investigation.subject = "examination") 
  (h2 : investigation.purpose ≠ "") : 
  ∃ (c : Census), c = investigation :=
sorry

end NUMINAMATH_CALUDE_stating_comprehensive_investigation_is_census_l2885_288590


namespace NUMINAMATH_CALUDE_number_of_possible_lists_l2885_288542

def number_of_balls : ℕ := 15
def list_length : ℕ := 4

theorem number_of_possible_lists :
  (number_of_balls ^ list_length : ℕ) = 50625 := by
sorry

end NUMINAMATH_CALUDE_number_of_possible_lists_l2885_288542


namespace NUMINAMATH_CALUDE_polynomial_root_implies_k_value_l2885_288581

theorem polynomial_root_implies_k_value : ∀ k : ℚ,
  (3 : ℚ)^3 + k * 3 + 20 = 0 → k = -47/3 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_k_value_l2885_288581


namespace NUMINAMATH_CALUDE_reorganize_32_city_graph_l2885_288596

/-- A graph with n vertices, where each pair of vertices is connected by a directed edge. -/
structure DirectedGraph (n : ℕ) where
  edges : Fin n → Fin n → Bool

/-- The number of steps required to reorganize a directed graph with n vertices
    such that the resulting graph has no cycles. -/
def reorganization_steps (n : ℕ) : ℕ :=
  if n ≤ 2 then 0 else 2^(n-2) * (2^n - n - 1)

/-- Theorem stating that for a graph with 32 vertices, it's possible to reorganize
    the edge directions in at most 208 steps to eliminate all cycles. -/
theorem reorganize_32_city_graph :
  reorganization_steps 32 ≤ 208 :=
sorry

end NUMINAMATH_CALUDE_reorganize_32_city_graph_l2885_288596


namespace NUMINAMATH_CALUDE_library_book_combinations_l2885_288545

theorem library_book_combinations : Nat.choose 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_library_book_combinations_l2885_288545


namespace NUMINAMATH_CALUDE_shiela_paintings_l2885_288547

/-- The number of paintings Shiela can give to each grandmother -/
def paintings_per_grandmother : ℕ := 9

/-- The number of grandmothers Shiela has -/
def number_of_grandmothers : ℕ := 2

/-- The total number of paintings Shiela has -/
def total_paintings : ℕ := paintings_per_grandmother * number_of_grandmothers

theorem shiela_paintings : total_paintings = 18 := by
  sorry

end NUMINAMATH_CALUDE_shiela_paintings_l2885_288547


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l2885_288527

theorem smallest_number_in_sequence (a b c d : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- Four positive integers
  (a + b + c + d) / 4 = 30 →  -- Arithmetic mean is 30
  b = 33 →  -- Second largest is 33
  d = b + 3 →  -- Largest is 3 more than second largest
  a < b ∧ b < c ∧ c < d →  -- Ascending order
  a = 17 :=  -- The smallest number is 17
by sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l2885_288527


namespace NUMINAMATH_CALUDE_log_relation_l2885_288507

theorem log_relation (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = m * y) → 
  m = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l2885_288507


namespace NUMINAMATH_CALUDE_johns_share_is_18_l2885_288557

/-- The amount one person pays when splitting the cost of multiple items equally -/
def split_cost (num_items : ℕ) (price_per_item : ℚ) (num_people : ℕ) : ℚ :=
  (num_items : ℚ) * price_per_item / (num_people : ℚ)

/-- Theorem: John's share of the cake cost is $18 -/
theorem johns_share_is_18 :
  split_cost 3 12 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_johns_share_is_18_l2885_288557


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2885_288548

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 6 * x₁ - 1 = 0) → 
  (2 * x₂^2 + 6 * x₂ - 1 = 0) → 
  (x₁ + x₂ = -3) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2885_288548


namespace NUMINAMATH_CALUDE_right_triangle_yz_l2885_288577

/-- In a right triangle XYZ, given angle X, angle Y, and hypotenuse XZ, calculate YZ --/
theorem right_triangle_yz (X Y Z : ℝ) (angleX : ℝ) (angleY : ℝ) (XZ : ℝ) : 
  angleX = 25 * π / 180 →  -- Convert 25° to radians
  angleY = π / 2 →         -- 90° in radians
  XZ = 18 →
  abs (Y - (XZ * Real.sin angleX)) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_yz_l2885_288577


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2885_288564

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The condition c < 0 is sufficient but not necessary for f(x) < 0 -/
theorem sufficient_not_necessary (b c : ℝ) :
  (c < 0 → ∃ x, f b c x < 0) ∧
  ∃ b' c' x', c' ≥ 0 ∧ f b' c' x' < 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2885_288564


namespace NUMINAMATH_CALUDE_base8_563_to_base3_l2885_288544

/-- Converts a base 8 number to base 10 --/
def base8ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- Converts a base 10 number to base 3 --/
def base10ToBase3 (n : Nat) : List Nat :=
  sorry  -- Implementation details omitted

theorem base8_563_to_base3 :
  base10ToBase3 (base8ToBase10 563) = [1, 1, 1, 2, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_base8_563_to_base3_l2885_288544


namespace NUMINAMATH_CALUDE_minimize_sum_squared_distances_l2885_288538

-- Define the points A, B, C
def A : ℝ × ℝ := (3, -1)
def B : ℝ × ℝ := (-1, 4)
def C : ℝ × ℝ := (1, -6)

-- Define the function to calculate the sum of squared distances
def sumSquaredDistances (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x - A.1)^2 + (y - A.2)^2 +
  (x - B.1)^2 + (y - B.2)^2 +
  (x - C.1)^2 + (y - C.2)^2

-- Define the point P
def P : ℝ × ℝ := (1, -1)

-- Theorem statement
theorem minimize_sum_squared_distances :
  ∀ Q : ℝ × ℝ, sumSquaredDistances P ≤ sumSquaredDistances Q :=
sorry

end NUMINAMATH_CALUDE_minimize_sum_squared_distances_l2885_288538


namespace NUMINAMATH_CALUDE_cake_icing_theorem_l2885_288508

/-- Represents a rectangular prism cake with icing on specific sides -/
structure CakeWithIcing where
  length : ℕ
  width : ℕ
  height : ℕ
  hasTopIcing : Bool
  hasFrontIcing : Bool
  hasBackIcing : Bool

/-- Counts the number of 1x1x1 cubes with icing on exactly two sides -/
def countCubesWithTwoSidesIced (cake : CakeWithIcing) : ℕ :=
  sorry

/-- The main theorem stating that a 5x5x3 cake with top, front, and back icing
    will have exactly 30 small cubes with icing on two sides when divided into 1x1x1 cubes -/
theorem cake_icing_theorem :
  let cake : CakeWithIcing := {
    length := 5,
    width := 5,
    height := 3,
    hasTopIcing := true,
    hasFrontIcing := true,
    hasBackIcing := true
  }
  countCubesWithTwoSidesIced cake = 30 := by
  sorry

end NUMINAMATH_CALUDE_cake_icing_theorem_l2885_288508


namespace NUMINAMATH_CALUDE_johnson_family_seating_l2885_288506

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem johnson_family_seating (boys girls : ℕ) (total : ℕ) :
  boys = 5 →
  girls = 4 →
  total = boys + girls →
  factorial total - (factorial boys * factorial girls) = 360000 :=
by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l2885_288506


namespace NUMINAMATH_CALUDE_bob_monthly_hours_l2885_288543

/-- Calculates the total hours worked in a month given daily hours, workdays per week, and average weeks per month. -/
def total_monthly_hours (daily_hours : ℝ) (workdays_per_week : ℝ) (avg_weeks_per_month : ℝ) : ℝ :=
  daily_hours * workdays_per_week * avg_weeks_per_month

/-- Proves that Bob's total monthly hours are approximately 216.5 -/
theorem bob_monthly_hours :
  let daily_hours : ℝ := 10
  let workdays_per_week : ℝ := 5
  let avg_weeks_per_month : ℝ := 4.33
  abs (total_monthly_hours daily_hours workdays_per_week avg_weeks_per_month - 216.5) < 0.1 := by
  sorry

#eval total_monthly_hours 10 5 4.33

end NUMINAMATH_CALUDE_bob_monthly_hours_l2885_288543


namespace NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l2885_288599

/-- A quadrilateral with perpendicular diagonals, an inscribed circle, and a circumscribed circle -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perpendicular_diagonals : Bool
  /-- The quadrilateral has an inscribed circle -/
  has_inscribed_circle : Bool
  /-- The quadrilateral has a circumscribed circle -/
  has_circumscribed_circle : Bool

/-- Definition of a square -/
def is_square (q : SpecialQuadrilateral) : Bool := sorry

/-- Theorem: A quadrilateral with perpendicular diagonals, an inscribed circle, and a circumscribed circle is not necessarily a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ q : SpecialQuadrilateral,
    q.perpendicular_diagonals ∧
    q.has_inscribed_circle ∧
    q.has_circumscribed_circle ∧
    ¬(is_square q) :=
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l2885_288599


namespace NUMINAMATH_CALUDE_problem_parallelogram_area_l2885_288536

/-- A parallelogram in 2D space defined by four vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- The specific parallelogram from the problem -/
def problem_parallelogram : Parallelogram :=
  { v1 := (0, 0)
    v2 := (4, 0)
    v3 := (1, 5)
    v4 := (5, 5) }

/-- Theorem stating that the area of the problem parallelogram is 20 -/
theorem problem_parallelogram_area :
  area problem_parallelogram = 20 := by sorry

end NUMINAMATH_CALUDE_problem_parallelogram_area_l2885_288536


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_nine_l2885_288553

theorem sqrt_sum_equals_nine :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) - 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_nine_l2885_288553


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l2885_288513

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 16

-- Define the line m that bisects the circle
def line_m (x y : ℝ) : Prop :=
  3*x - y = 0

-- Define the line l passing through D(0,-1) with slope k
def line_l (k x y : ℝ) : Prop :=
  y = k*x - 1

-- Theorem statement
theorem circle_and_line_intersection :
  -- Circle C passes through A(1,-1) and B(5,3)
  circle_C 1 (-1) ∧ circle_C 5 3 ∧
  -- Circle C is bisected by line m
  (∀ x y, circle_C x y → line_m x y → x = 1 ∧ y = 3) →
  -- Part 1: Prove the equation of circle C
  (∀ x y, circle_C x y ↔ (x - 1)^2 + (y - 3)^2 = 16) ∧
  -- Part 2: Prove the range of k for which line l intersects circle C at two distinct points
  (∀ k, (∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂) ↔
        (k < -8/15 ∨ k > 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l2885_288513


namespace NUMINAMATH_CALUDE_inequality_proof_l2885_288504

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : y * z + z * x + x * y = 1) : 
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ (4/9) * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2885_288504


namespace NUMINAMATH_CALUDE_point_movement_to_y_axis_l2885_288567

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The y-axis -/
def yAxis : Set Point := {p : Point | p.x = 0}

theorem point_movement_to_y_axis (m : ℝ) :
  let P : Point := ⟨m + 2, 3⟩
  let P' : Point := ⟨P.x + 3, P.y⟩
  P' ∈ yAxis → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_to_y_axis_l2885_288567


namespace NUMINAMATH_CALUDE_sugar_for_recipe_l2885_288592

/-- The amount of sugar required for a cake recipe -/
theorem sugar_for_recipe (sugar_frosting sugar_cake : ℚ) 
  (h1 : sugar_frosting = 6/10)
  (h2 : sugar_cake = 2/10) :
  sugar_frosting + sugar_cake = 8/10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_for_recipe_l2885_288592


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l2885_288510

theorem thirty_percent_less_than_eighty (x : ℝ) : x + (1/4) * x = 80 - (30/100) * 80 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l2885_288510


namespace NUMINAMATH_CALUDE_cylinder_radius_comparison_l2885_288587

theorem cylinder_radius_comparison (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let original_volume := (6/7) * π * r^2 * h
  let new_height := (7/10) * h
  let new_volume := original_volume
  let new_radius := Real.sqrt ((5/3) * new_volume / (π * new_height))
  (new_radius - r) / r = 3/7 := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_comparison_l2885_288587


namespace NUMINAMATH_CALUDE_equation_solution_l2885_288519

theorem equation_solution : 
  ∃ x : ℝ, 3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * x)) = 2800.0000000000005 ∧ x = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2885_288519


namespace NUMINAMATH_CALUDE_german_students_l2885_288574

theorem german_students (total : ℕ) (both : ℕ) (german : ℕ) (spanish : ℕ) :
  total = 30 ∧ 
  both = 2 ∧ 
  german + spanish + both = total ∧ 
  german = 3 * spanish →
  german - both = 20 := by
  sorry

end NUMINAMATH_CALUDE_german_students_l2885_288574


namespace NUMINAMATH_CALUDE_min_sum_of_slopes_l2885_288555

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop :=
  x^2 - 4*(x+y) + y^2 = 2*x*y + 8

/-- Tangent line to the parabola at point (a, b) -/
def tangent_line (a b x y : ℝ) : Prop :=
  y - b = ((b - a + 2) / (b - a - 2)) * (x - a)

/-- Intersection point of the tangent lines -/
def intersection_point (p q : ℝ) : Prop :=
  p + q = -32

theorem min_sum_of_slopes :
  ∃ (a b p q : ℝ),
    parabola a b ∧
    parabola b a ∧
    intersection_point p q ∧
    tangent_line a b p q ∧
    tangent_line b a p q ∧
    ((b - a + 2) / (b - a - 2) + (a - b + 2) / (a - b - 2) ≥ 62 / 29) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_slopes_l2885_288555


namespace NUMINAMATH_CALUDE_club_officer_selection_l2885_288580

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members : ℕ) (founding_members : ℕ) (positions : ℕ) : ℕ :=
  founding_members * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose officers in the given scenario --/
theorem club_officer_selection :
  choose_officers 12 4 5 = 25920 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l2885_288580


namespace NUMINAMATH_CALUDE_equation_proof_l2885_288522

theorem equation_proof : ((12 : ℝ)^2 * (6 : ℝ)^4 / 432)^(1/2) = 4 * 3 * (3 : ℝ)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2885_288522


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2885_288535

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

theorem parallel_lines_m_value (m : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y => 2 * x + m * y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y => y = 3 * x - 1
  parallel (-1/2) (1/3) → m = -2/3 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l2885_288535


namespace NUMINAMATH_CALUDE_probability_exact_scenario_verify_conditions_l2885_288540

/-- The probability of drawing all red balls by the 4th draw in a specific scenario -/
def probability_all_red_by_fourth_draw (total_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) : ℚ :=
  let total_ways := (total_balls.choose 4)
  let favorable_ways := (red_balls.choose 2) * (white_balls.choose 2)
  (favorable_ways : ℚ) / total_ways

/-- The main theorem stating the probability for the given scenario -/
theorem probability_exact_scenario : 
  probability_all_red_by_fourth_draw 10 8 2 = 4338 / 91125 := by
  sorry

/-- Verifies that the conditions of the problem are met -/
theorem verify_conditions : 
  10 = 8 + 2 ∧ 
  8 ≥ 0 ∧ 
  2 ≥ 0 ∧
  10 > 0 := by
  sorry

end NUMINAMATH_CALUDE_probability_exact_scenario_verify_conditions_l2885_288540


namespace NUMINAMATH_CALUDE_x_over_y_value_l2885_288502

theorem x_over_y_value (x y : ℝ) 
  (h1 : 3 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 6)
  (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -2 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_value_l2885_288502


namespace NUMINAMATH_CALUDE_minimum_race_distance_minimum_race_distance_is_1200_l2885_288541

/-- The minimum distance a runner must travel in a race with given conditions -/
theorem minimum_race_distance (A B : ℝ × ℝ) (wall_length : ℝ) : ℝ :=
  let wall_start := A
  let wall_end := (A.1, A.2 + wall_length)
  let distance := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  
  1200

/-- The theorem states that the minimum race distance is 1200 meters -/
theorem minimum_race_distance_is_1200 :
  minimum_race_distance (0, 0) (1000, 600) 1000 = 1200 := by
  sorry

#check minimum_race_distance_is_1200

end NUMINAMATH_CALUDE_minimum_race_distance_minimum_race_distance_is_1200_l2885_288541


namespace NUMINAMATH_CALUDE_fraction_calculation_l2885_288556

theorem fraction_calculation : (7 / 9 - 5 / 6 + 5 / 18) * 18 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2885_288556


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l2885_288520

/-- Proves that the percentage increase in rent for one friend is 16% given the conditions of the problem -/
theorem rent_increase_percentage (num_friends : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (initial_rent : ℝ) : 
  num_friends = 4 →
  initial_avg = 800 →
  new_avg = 850 →
  initial_rent = 1250 →
  let total_initial := initial_avg * num_friends
  let new_rent := (new_avg * num_friends) - (total_initial - initial_rent)
  let percentage_increase := (new_rent - initial_rent) / initial_rent * 100
  percentage_increase = 16 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l2885_288520


namespace NUMINAMATH_CALUDE_abc_sum_bound_l2885_288554

theorem abc_sum_bound (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (x : ℝ), x ≤ 1/2 ∧ ∃ (a' b' c' : ℝ), a' + b' + c' = 1 ∧ a'*b' + a'*c' + b'*c' = x :=
sorry

end NUMINAMATH_CALUDE_abc_sum_bound_l2885_288554


namespace NUMINAMATH_CALUDE_population_after_10_years_l2885_288537

/-- Given an initial population and growth rate, calculates the population after n years -/
def population (M : ℝ) (p : ℝ) (n : ℕ) : ℝ := M * (1 + p) ^ n

/-- Theorem: The population after 10 years with initial population M and growth rate p is M(1+p)^10 -/
theorem population_after_10_years (M p : ℝ) : 
  population M p 10 = M * (1 + p)^10 := by
  sorry

end NUMINAMATH_CALUDE_population_after_10_years_l2885_288537


namespace NUMINAMATH_CALUDE_parallel_to_line_if_equal_perpendicular_distances_l2885_288586

structure Geometry2D where
  Point : Type
  Line : Type
  perpendicular_distance : Point → Line → ℝ
  on_line : Point → Line → Prop
  parallel : Line → Line → Prop

variable {G : Geometry2D}

theorem parallel_to_line_if_equal_perpendicular_distances
  (A B : G.Point) (l : G.Line) :
  G.perpendicular_distance A l = G.perpendicular_distance B l →
  ∃ (AB : G.Line), G.on_line A AB ∧ G.on_line B AB ∧ G.parallel AB l :=
sorry

end NUMINAMATH_CALUDE_parallel_to_line_if_equal_perpendicular_distances_l2885_288586


namespace NUMINAMATH_CALUDE_polygon_vertices_from_diagonals_l2885_288576

/-- The number of vertices in a polygon given the number of diagonals from a single vertex -/
def num_vertices (diagonals_from_vertex : ℕ) : ℕ :=
  diagonals_from_vertex + 3

theorem polygon_vertices_from_diagonals (diagonals : ℕ) (h : diagonals = 6) :
  num_vertices diagonals = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_vertices_from_diagonals_l2885_288576


namespace NUMINAMATH_CALUDE_fraction_addition_l2885_288569

theorem fraction_addition : (3 / 4) / (5 / 8) + 1 / 2 = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2885_288569


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2885_288571

/-- The area of a square field with a given diagonal length -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 98.00000000000001) :
  d^2 / 2 = 4802.000000000001 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2885_288571


namespace NUMINAMATH_CALUDE_stratified_sample_female_count_l2885_288526

theorem stratified_sample_female_count (male_count : ℕ) (female_count : ℕ) (sample_size : ℕ) :
  male_count = 48 →
  female_count = 36 →
  sample_size = 35 →
  (female_count : ℚ) / (male_count + female_count) * sample_size = 15 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_female_count_l2885_288526


namespace NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l2885_288505

/-- Given a geometric sequence {a_n}, prove that if the sum of four consecutive terms
    and the product of two middle terms satisfy certain conditions, then the sum of
    their reciprocals is equal to a specific value. -/
theorem geometric_sequence_reciprocal_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n) →  -- a_n is a geometric sequence
  a 5 + a 6 + a 7 + a 8 = 15 / 8 →           -- sum condition
  a 6 * a 7 = -9 / 8 →                       -- product condition
  1 / a 5 + 1 / a 6 + 1 / a 7 + 1 / a 8 = -5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l2885_288505


namespace NUMINAMATH_CALUDE_euclidean_division_37_by_5_l2885_288531

theorem euclidean_division_37_by_5 :
  ∃ (q r : ℤ), 37 = 5 * q + r ∧ 0 ≤ r ∧ r < 5 ∧ q = 7 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_euclidean_division_37_by_5_l2885_288531


namespace NUMINAMATH_CALUDE_xiaoming_walking_speed_l2885_288518

theorem xiaoming_walking_speed (distance : ℝ) (min_time max_time : ℝ) (h1 : distance = 3500)
  (h2 : min_time = 40) (h3 : max_time = 50) :
  let speed_range := {x : ℝ | distance / max_time ≤ x ∧ x ≤ distance / min_time}
  ∀ x ∈ speed_range, 70 ≤ x ∧ x ≤ 87.5 :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_walking_speed_l2885_288518


namespace NUMINAMATH_CALUDE_largest_angle_in_specific_triangle_l2885_288533

/-- The largest angle in a triangle with sides 3√2, 6, and 3√10 is 135° --/
theorem largest_angle_in_specific_triangle : 
  ∀ (a b c : ℝ) (θ : ℝ),
  a = 3 * Real.sqrt 2 →
  b = 6 →
  c = 3 * Real.sqrt 10 →
  c > a ∧ c > b →
  θ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) →
  θ = 135 * (π / 180) := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_specific_triangle_l2885_288533


namespace NUMINAMATH_CALUDE_vector_collinearity_l2885_288512

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 1]
def c : Fin 2 → ℝ := ![2, 1]

def collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, v = fun i => t * u i

theorem vector_collinearity (k : ℝ) :
  collinear (fun i => k * a i + b i) c → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2885_288512


namespace NUMINAMATH_CALUDE_negative_two_star_negative_three_l2885_288562

-- Define the new operation
def star (a b : ℤ) : ℤ := b^2 - a

-- State the theorem
theorem negative_two_star_negative_three : star (-2) (-3) = 11 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_star_negative_three_l2885_288562


namespace NUMINAMATH_CALUDE_range_of_a_for_no_real_roots_l2885_288561

theorem range_of_a_for_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a + 1) * x + 1 > 0) ↔ a ∈ Set.Ioo (-3 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_no_real_roots_l2885_288561


namespace NUMINAMATH_CALUDE_boys_girls_arrangement_l2885_288511

/-- The number of ways to arrange boys and girls in a row with alternating genders -/
def alternating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  (Nat.factorial num_boys) * (Nat.factorial num_girls)

/-- Theorem stating that there are 144 ways to arrange 4 boys and 3 girls
    in a row such that no two boys or two girls stand next to each other -/
theorem boys_girls_arrangement :
  alternating_arrangements 4 3 = 144 := by
  sorry

#check boys_girls_arrangement

end NUMINAMATH_CALUDE_boys_girls_arrangement_l2885_288511


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2885_288550

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a ≤ 1 ∧ b ≤ 1 → a + b ≤ 2) ∧
  (∃ a b : ℝ, a + b ≤ 2 ∧ ¬(a ≤ 1 ∧ b ≤ 1)) := by
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2885_288550


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l2885_288509

/-- The sum of interior angles of a regular polygon with exterior angles of 20 degrees -/
theorem sum_interior_angles_regular_polygon (n : ℕ) (h : n * 20 = 360) : 
  (n - 2) * 180 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l2885_288509


namespace NUMINAMATH_CALUDE_unique_values_theorem_l2885_288525

/-- Definition of the sequence P_n -/
def P (a b : ℝ) : ℕ → ℝ × ℝ
  | 0 => (1, 0)
  | n + 1 => let (x, y) := P a b n; (a * x - b * y, b * x + a * y)

/-- Condition (i): P_0 = P_6 -/
def condition_i (a b : ℝ) : Prop := P a b 0 = P a b 6

/-- Condition (ii): All P_0, P_1, P_2, P_3, P_4, P_5 are distinct -/
def condition_ii (a b : ℝ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < j ∧ j < 6 → P a b i ≠ P a b j

/-- The main theorem -/
theorem unique_values_theorem :
  {(a, b) : ℝ × ℝ | condition_i a b ∧ condition_ii a b} =
  {(1/2, Real.sqrt 3/2), (1/2, -Real.sqrt 3/2)} :=
sorry

end NUMINAMATH_CALUDE_unique_values_theorem_l2885_288525


namespace NUMINAMATH_CALUDE_train_crossing_time_l2885_288517

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length platform_length : ℝ) (train_speed : ℝ) : 
  train_length = platform_length →
  train_length = 750 →
  train_speed = 90 * 1000 / 3600 →
  (train_length + platform_length) / train_speed = 60 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2885_288517


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2885_288591

/-- An isosceles triangle with perimeter 16 and one side of length 6 has a base of either 6 or 4. -/
theorem isosceles_triangle_base_length (a b : ℝ) : 
  a > 0 → b > 0 → 
  a + b + b = 16 → 
  (a = 6 ∨ b = 6) → 
  (a = 6 ∧ b = 5) ∨ (a = 4 ∧ b = 6) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2885_288591


namespace NUMINAMATH_CALUDE_calculation_proof_l2885_288514

theorem calculation_proof : (1/4 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * 72 + (1/8 : ℚ) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2885_288514


namespace NUMINAMATH_CALUDE_battery_current_l2885_288565

/-- Given a battery with voltage 48V, prove that when the resistance R is 12Ω, 
    the current I is 4A, where I is related to R by the function I = 48 / R. -/
theorem battery_current (R : ℝ) (I : ℝ) : 
  (I = 48 / R) → (R = 12) → (I = 4) := by
  sorry

end NUMINAMATH_CALUDE_battery_current_l2885_288565


namespace NUMINAMATH_CALUDE_square_difference_formula_l2885_288529

theorem square_difference_formula (a b : ℚ) 
  (sum_eq : a + b = 3/4)
  (diff_eq : a - b = 1/8) : 
  a^2 - b^2 = 3/32 := by
sorry

end NUMINAMATH_CALUDE_square_difference_formula_l2885_288529


namespace NUMINAMATH_CALUDE_matrix_sum_of_squares_l2885_288593

theorem matrix_sum_of_squares (x y z w : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (B.transpose = 2 • B⁻¹) → x^2 + y^2 + z^2 + w^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_of_squares_l2885_288593


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l2885_288589

/-- A function that checks if all digits in the decimal representation of a natural number are greater than or equal to 7. -/
def allDigitsAtLeastSeven (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≥ 7

/-- A function that generates a pair of natural numbers based on the input n. -/
noncomputable def f (n : ℕ) : ℕ × ℕ :=
  let a := (887 : ℕ).pow n
  let b := 10^(3*n) - 123
  (a, b)

/-- Theorem stating that there exist infinitely many pairs of integers satisfying the given conditions. -/
theorem infinitely_many_pairs_exist :
  ∀ n : ℕ, 
    let (a, b) := f n
    allDigitsAtLeastSeven a ∧
    allDigitsAtLeastSeven b ∧
    allDigitsAtLeastSeven (a * b) :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l2885_288589


namespace NUMINAMATH_CALUDE_prism_surface_area_l2885_288523

/-- A rectangular prism with prime edge lengths and volume 627 has surface area 598 -/
theorem prism_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 627 →
  2 * (a * b + b * c + c * a) = 598 := by
sorry

end NUMINAMATH_CALUDE_prism_surface_area_l2885_288523


namespace NUMINAMATH_CALUDE_percentage_problem_l2885_288582

theorem percentage_problem (P : ℝ) : 
  (5 / 100 * 6400 = P / 100 * 650 + 190) → P = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2885_288582


namespace NUMINAMATH_CALUDE_largest_c_for_quadratic_range_l2885_288579

theorem largest_c_for_quadratic_range (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 2) ↔ c ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_quadratic_range_l2885_288579
