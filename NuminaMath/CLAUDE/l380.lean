import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_a3_equals_5_l380_38089

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that in an arithmetic sequence, a_3 = 5 given the conditions -/
theorem arithmetic_sequence_a3_equals_5 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 3 + a 5 = 15) : 
  a 3 = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a3_equals_5_l380_38089


namespace NUMINAMATH_CALUDE_negation_equivalence_l380_38000

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, ∃ n : ℕ+, n ≥ x^2) ↔ (∃ x : ℝ, ∀ n : ℕ+, n < x^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l380_38000


namespace NUMINAMATH_CALUDE_reduced_rates_fraction_l380_38052

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the number of hours in a day
def hours_in_day : ℕ := 24

-- Define the number of weekdays (Monday to Friday)
def weekdays : ℕ := 5

-- Define the number of weekend days (Saturday and Sunday)
def weekend_days : ℕ := 2

-- Define the number of hours with reduced rates on weekdays (8 p.m. to 8 a.m.)
def reduced_hours_weekday : ℕ := 12

-- Define the number of hours with reduced rates on weekend days (24 hours)
def reduced_hours_weekend : ℕ := 24

-- Theorem stating that the fraction of a week with reduced rates is 9/14
theorem reduced_rates_fraction :
  (weekdays * reduced_hours_weekday + weekend_days * reduced_hours_weekend) / 
  (days_in_week * hours_in_day) = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_reduced_rates_fraction_l380_38052


namespace NUMINAMATH_CALUDE_sequential_structure_essential_l380_38037

/-- Represents the different types of algorithm structures -/
inductive AlgorithmStructure
  | Logical
  | Selection
  | Loop
  | Sequential

/-- Represents an algorithm -/
structure Algorithm where
  structures : List AlgorithmStructure

/-- Defines what it means for a structure to be essential for all algorithms -/
def isEssentialStructure (s : AlgorithmStructure) : Prop :=
  ∀ (a : Algorithm), s ∈ a.structures

/-- States that an algorithm can exist without Logical, Selection, or Loop structures -/
axiom non_essential_structures :
  ∃ (a : Algorithm),
    AlgorithmStructure.Logical ∉ a.structures ∧
    AlgorithmStructure.Selection ∉ a.structures ∧
    AlgorithmStructure.Loop ∉ a.structures

/-- The main theorem: Sequential structure is the only essential structure -/
theorem sequential_structure_essential :
  isEssentialStructure AlgorithmStructure.Sequential ∧
  (∀ s : AlgorithmStructure, s ≠ AlgorithmStructure.Sequential → ¬isEssentialStructure s) :=
sorry

end NUMINAMATH_CALUDE_sequential_structure_essential_l380_38037


namespace NUMINAMATH_CALUDE_sally_quarters_l380_38092

/-- The number of quarters Sally spent -/
def quarters_spent : ℕ := 418

/-- The number of quarters Sally has left -/
def quarters_left : ℕ := 342

/-- The initial number of quarters Sally had -/
def initial_quarters : ℕ := quarters_spent + quarters_left

theorem sally_quarters : initial_quarters = 760 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_l380_38092


namespace NUMINAMATH_CALUDE_composite_prime_calculation_l380_38019

def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]
def next_six_composites : List Nat := [14, 15, 16, 18, 20, 21]
def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem composite_prime_calculation :
  (((first_six_composites.prod : ℚ) / (next_six_composites.prod : ℚ)) * (first_five_primes.prod : ℚ)) = 377.55102040816324 := by
  sorry

end NUMINAMATH_CALUDE_composite_prime_calculation_l380_38019


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l380_38035

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x + 4

-- State the theorem
theorem f_decreasing_interval :
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 6 ∧ f x₂ = 2) →  -- Maximum and minimum conditions
  (∀ x, f x ≤ 6) →                           -- 6 is the maximum value
  (∀ x, f x ≥ 2) →                           -- 2 is the minimum value
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l380_38035


namespace NUMINAMATH_CALUDE_inequality_solution_l380_38090

theorem inequality_solution (x : ℝ) : (x + 3) * (x - 1) < 0 ↔ -3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l380_38090


namespace NUMINAMATH_CALUDE_book_arrangement_l380_38068

theorem book_arrangement (n m : ℕ) (hn : n = 3) (hm : m = 4) :
  (Nat.choose (n + m) n) = 35 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_l380_38068


namespace NUMINAMATH_CALUDE_rose_difference_l380_38015

theorem rose_difference (total : ℕ) (red_fraction : ℚ) (yellow_fraction : ℚ)
  (h_total : total = 48)
  (h_red : red_fraction = 3/8)
  (h_yellow : yellow_fraction = 5/16) :
  ↑total * red_fraction - ↑total * yellow_fraction = 3 :=
by sorry

end NUMINAMATH_CALUDE_rose_difference_l380_38015


namespace NUMINAMATH_CALUDE_train_speed_ratio_l380_38038

/-- Proves that the ratio of speeds of two trains is 2:1 given specific conditions --/
theorem train_speed_ratio :
  ∀ (v_A v_B : ℝ),
  v_A > 0 ∧ v_B > 0 →
  v_A = 2 * v_B →
  ∃ (L_A L_B : ℝ),
    L_A > 0 ∧ L_B > 0 ∧
    L_A / v_A = 27 ∧
    L_B / v_B = 17 ∧
    (L_A + L_B) / (v_A + v_B) = 22 ∧
    v_A + v_B ≤ 60 →
  v_A / v_B = 2 := by
sorry

end NUMINAMATH_CALUDE_train_speed_ratio_l380_38038


namespace NUMINAMATH_CALUDE_f_monotonically_decreasing_l380_38093

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

-- Theorem statement
theorem f_monotonically_decreasing :
  ∀ x y, -2 < x ∧ x < y ∧ y < 1 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_f_monotonically_decreasing_l380_38093


namespace NUMINAMATH_CALUDE_toy_store_shelves_l380_38080

/-- Calculates the number of shelves needed to store bears in a toy store. -/
def shelves_needed (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Proves that given the specific conditions, the number of shelves needed is 5. -/
theorem toy_store_shelves : shelves_needed 15 45 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l380_38080


namespace NUMINAMATH_CALUDE_lacy_correct_percentage_l380_38079

theorem lacy_correct_percentage (x : ℝ) (h : x > 0) : 
  let total := 6 * x
  let missed := 2 * x
  let correct := total - missed
  (correct / total) * 100 = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_lacy_correct_percentage_l380_38079


namespace NUMINAMATH_CALUDE_one_pair_percentage_l380_38062

def five_digit_numbers : ℕ := 90000

def numbers_with_one_pair : ℕ := 10 * 10 * 9 * 8 * 7

theorem one_pair_percentage : 
  (numbers_with_one_pair : ℚ) / five_digit_numbers * 100 = 56 :=
by sorry

end NUMINAMATH_CALUDE_one_pair_percentage_l380_38062


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l380_38075

/-- Theorem: Volume of cylinder X in terms of cylinder Y's height -/
theorem cylinder_volume_relation (h : ℝ) (h_pos : h > 0) : ∃ (r_x r_y h_x : ℝ),
  r_y = 2 * h_x ∧ 
  h_x = 3 * r_y ∧ 
  h_x = 3 * h ∧
  r_x = 6 * h ∧
  π * r_x^2 * h_x = 3 * (π * r_y^2 * h) ∧
  π * r_x^2 * h_x = 108 * π * h^3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l380_38075


namespace NUMINAMATH_CALUDE_carol_weight_l380_38063

/-- Given two people's weights satisfying certain conditions, prove that one person's weight is 165 pounds. -/
theorem carol_weight (alice_weight carol_weight : ℝ) 
  (sum_condition : alice_weight + carol_weight = 220)
  (difference_condition : carol_weight - alice_weight = (2/3) * carol_weight) :
  carol_weight = 165 := by
  sorry

end NUMINAMATH_CALUDE_carol_weight_l380_38063


namespace NUMINAMATH_CALUDE_min_ferries_required_l380_38074

def ferry_capacity : ℕ := 45
def people_to_transport : ℕ := 523

theorem min_ferries_required : 
  ∃ (n : ℕ), n * ferry_capacity ≥ people_to_transport ∧ 
  ∀ (m : ℕ), m * ferry_capacity ≥ people_to_transport → m ≥ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_min_ferries_required_l380_38074


namespace NUMINAMATH_CALUDE_line_equation_proof_l380_38044

/-- A line passing through a point with a given direction vector -/
structure DirectedLine where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The general form of a line equation: ax + by + c = 0 -/
structure GeneralLineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line passing through (5,4) with direction vector (1,2),
    its general form equation is 2x - y - 6 = 0 -/
theorem line_equation_proof (l : DirectedLine) 
    (h1 : l.point = (5, 4))
    (h2 : l.direction = (1, 2)) :
    ∃ (eq : GeneralLineEquation), 
      eq.a = 2 ∧ eq.b = -1 ∧ eq.c = -6 ∧
      ∀ (x y : ℝ), eq.a * x + eq.b * y + eq.c = 0 ↔ 
        ∃ (t : ℝ), (x, y) = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l380_38044


namespace NUMINAMATH_CALUDE_carries_money_from_mom_l380_38048

def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11
def money_left : ℕ := 50

theorem carries_money_from_mom : 
  sweater_cost + tshirt_cost + shoes_cost + money_left = 91 := by
  sorry

end NUMINAMATH_CALUDE_carries_money_from_mom_l380_38048


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l380_38026

theorem trigonometric_expression_equality : 
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) / 
  (Real.sin (12 * π / 180) * (4 * (Real.cos (12 * π / 180))^2 - 2)) = 
  -4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l380_38026


namespace NUMINAMATH_CALUDE_hexagon_diagonals_intersect_at_nine_point_center_l380_38011

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Finset Point
  is_convex : Bool

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point := sorry

/-- The perpendicular bisector of a line segment -/
def perp_bisector (p1 p2 : Point) : Set Point := sorry

/-- The intersection points of a line with a triangle's sides -/
def intersections_with_triangle (line : Set Point) (t : Triangle) : Finset Point := sorry

/-- The hexagon formed by the intersections of perpendicular bisectors with triangle sides -/
def form_hexagon (t : Triangle) (h : Point) : Hexagon := sorry

/-- The main diagonals of a hexagon -/
def main_diagonals (h : Hexagon) : Finset (Set Point) := sorry

/-- The intersection point of the main diagonals of a hexagon -/
def diagonals_intersection (h : Hexagon) : Option Point := sorry

/-- The center of the nine-point circle of a triangle -/
def nine_point_center (t : Triangle) : Point := sorry

/-- The theorem to be proved -/
theorem hexagon_diagonals_intersect_at_nine_point_center 
  (t : Triangle) (is_acute : Bool) : 
  let h := orthocenter t
  let hexagon := form_hexagon t h
  diagonals_intersection hexagon = some (nine_point_center t) := by sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_intersect_at_nine_point_center_l380_38011


namespace NUMINAMATH_CALUDE_trivia_team_size_l380_38055

theorem trivia_team_size :
  ∀ (original_members : ℕ),
  (original_members ≥ 2) →
  (4 * (original_members - 2) = 20) →
  original_members = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_team_size_l380_38055


namespace NUMINAMATH_CALUDE_solve_for_x_l380_38081

theorem solve_for_x (x y : ℚ) (h1 : x / y = 9 / 5) (h2 : y = 25) : x = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l380_38081


namespace NUMINAMATH_CALUDE_road_trash_cans_l380_38084

/-- The number of trash cans on both sides of a road -/
def trashCanCount (roadLength : ℕ) (interval : ℕ) : ℕ :=
  2 * ((roadLength / interval) - 1)

/-- Theorem: The total number of trash cans on a 400-meter road with 20-meter intervals is 38 -/
theorem road_trash_cans :
  trashCanCount 400 20 = 38 := by
  sorry

end NUMINAMATH_CALUDE_road_trash_cans_l380_38084


namespace NUMINAMATH_CALUDE_correct_tax_distribution_l380_38001

/-- Represents the tax calculation for three individuals based on their yields -/
def tax_calculation (total_tax : ℚ) (yield1 yield2 yield3 : ℕ) : Prop :=
  let total_yield := yield1 + yield2 + yield3
  let tax1 := total_tax * (yield1 : ℚ) / total_yield
  let tax2 := total_tax * (yield2 : ℚ) / total_yield
  let tax3 := total_tax * (yield3 : ℚ) / total_yield
  (tax1 = 1 + 3/32) ∧ (tax2 = 1 + 1/4) ∧ (tax3 = 1 + 13/32)

/-- Theorem stating the correct tax distribution for the given problem -/
theorem correct_tax_distribution :
  tax_calculation (15/4) 7 8 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_tax_distribution_l380_38001


namespace NUMINAMATH_CALUDE_vector_equation_l380_38083

theorem vector_equation (a b : ℝ × ℝ) : 
  a = (1, 2) → 2 • a + b = (3, 2) → b = (1, -2) := by sorry

end NUMINAMATH_CALUDE_vector_equation_l380_38083


namespace NUMINAMATH_CALUDE_triangle_inequality_ratio_three_fourths_is_optimal_l380_38054

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq_ab : c < a + b
  triangle_ineq_bc : a < b + c
  triangle_ineq_ca : b < c + a

-- Theorem statement
theorem triangle_inequality_ratio (t : Triangle) :
  (t.a^2 + t.b^2 + t.a * t.b) / t.c^2 ≥ (3/4 : ℝ) :=
sorry

-- Theorem for the optimality of the bound
theorem three_fourths_is_optimal :
  ∀ ε > 0, ∃ t : Triangle, (t.a^2 + t.b^2 + t.a * t.b) / t.c^2 < 3/4 + ε :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_ratio_three_fourths_is_optimal_l380_38054


namespace NUMINAMATH_CALUDE_line_equation_proof_l380_38043

/-- A line passing through point (1, -2) with slope 3 has the equation 3x - y - 5 = 0 -/
theorem line_equation_proof (x y : ℝ) : 
  (y - (-2) = 3 * (x - 1)) ↔ (3 * x - y - 5 = 0) := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l380_38043


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l380_38022

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : m % 2 = 0) (h3 : m % 221 = 0) :
  ∃ (d : ℕ), d > 221 ∧ m % d = 0 ∧ (∀ (x : ℕ), 221 < x ∧ x < d → m % x ≠ 0) → d = 247 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l380_38022


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l380_38012

theorem arithmetic_mean_of_fractions :
  let f1 : ℚ := 3 / 8
  let f2 : ℚ := 5 / 9
  let f3 : ℚ := 7 / 12
  let mean : ℚ := (f1 + f2 + f3) / 3
  mean = 109 / 216 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l380_38012


namespace NUMINAMATH_CALUDE_share_of_a_l380_38047

theorem share_of_a (total : ℝ) (a b c : ℝ) : 
  total = 500 →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a = 125 := by
sorry

end NUMINAMATH_CALUDE_share_of_a_l380_38047


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l380_38017

theorem sufficient_not_necessary : ∀ x : ℝ, 
  (∀ x, x > 5 → x > 3) ∧ 
  (∃ x, x > 3 ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l380_38017


namespace NUMINAMATH_CALUDE_impossible_inequality_l380_38004

theorem impossible_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_log : Real.log x / Real.log 2 = Real.log y / Real.log 3 ∧ 
           Real.log y / Real.log 3 = Real.log z / Real.log 5 ∧
           Real.log z / Real.log 5 > 0) :
  ¬(y / 3 < z / 5 ∧ z / 5 < x / 2) := by
sorry

end NUMINAMATH_CALUDE_impossible_inequality_l380_38004


namespace NUMINAMATH_CALUDE_total_path_satisfies_conditions_l380_38014

/-- The total length of Gyeongyeon's travel path --/
def total_path : ℝ := 2200

/-- Gyeongyeon's travel segments --/
structure TravelSegments where
  bicycle : ℝ
  first_walk : ℝ
  bus : ℝ
  final_walk : ℝ

/-- Conditions of Gyeongyeon's travel --/
def travel_conditions (d : ℝ) : Prop :=
  ∃ (segments : TravelSegments),
    segments.bicycle = d / 2 ∧
    segments.first_walk = 300 ∧
    segments.bus = (d / 2 - 300) / 2 ∧
    segments.final_walk = 400 ∧
    segments.bicycle + segments.first_walk + segments.bus + segments.final_walk = d

/-- Theorem stating that the total path length satisfies the travel conditions --/
theorem total_path_satisfies_conditions : travel_conditions total_path := by
  sorry


end NUMINAMATH_CALUDE_total_path_satisfies_conditions_l380_38014


namespace NUMINAMATH_CALUDE_probability_yellow_second_marble_l380_38009

-- Define the number of marbles in each bag
def bag_A_white : ℕ := 5
def bag_A_black : ℕ := 2
def bag_B_yellow : ℕ := 4
def bag_B_blue : ℕ := 5
def bag_C_yellow : ℕ := 3
def bag_C_blue : ℕ := 4
def bag_D_yellow : ℕ := 8
def bag_D_blue : ℕ := 2

-- Define the probabilities of drawing from each bag
def prob_white_A : ℚ := bag_A_white / (bag_A_white + bag_A_black)
def prob_black_A : ℚ := bag_A_black / (bag_A_white + bag_A_black)
def prob_yellow_B : ℚ := bag_B_yellow / (bag_B_yellow + bag_B_blue)
def prob_yellow_C : ℚ := bag_C_yellow / (bag_C_yellow + bag_C_blue)
def prob_yellow_D : ℚ := bag_D_yellow / (bag_D_yellow + bag_D_blue)

-- Assume equal probability of odd and even weight for black marbles
def prob_odd_weight : ℚ := 1/2
def prob_even_weight : ℚ := 1/2

-- Define the theorem
theorem probability_yellow_second_marble :
  prob_white_A * prob_yellow_B +
  prob_black_A * prob_odd_weight * prob_yellow_C +
  prob_black_A * prob_even_weight * prob_yellow_D = 211/245 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_second_marble_l380_38009


namespace NUMINAMATH_CALUDE_angle4_measure_l380_38050

-- Define the angles
def angle1 : ℝ := 85
def angle2 : ℝ := 34
def angle3 : ℝ := 20

-- Define the theorem
theorem angle4_measure : 
  ∀ (angle4 angle5 angle6 : ℝ),
  -- Conditions
  (angle1 + angle2 + angle3 + angle5 + angle6 = 180) →
  (angle4 + angle5 + angle6 = 180) →
  -- Conclusion
  angle4 = 139 := by
sorry

end NUMINAMATH_CALUDE_angle4_measure_l380_38050


namespace NUMINAMATH_CALUDE_point_coordinates_l380_38077

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the Cartesian coordinate system -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates :
  ∀ (p : Point),
    second_quadrant p →
    distance_to_x_axis p = 3 →
    distance_to_y_axis p = 7 →
    p = Point.mk (-7) 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l380_38077


namespace NUMINAMATH_CALUDE_businessmen_drink_count_l380_38042

theorem businessmen_drink_count (total : ℕ) (coffee : ℕ) (tea : ℕ) (coffee_and_tea : ℕ) 
  (juice : ℕ) (juice_and_tea_not_coffee : ℕ) 
  (h1 : total = 35)
  (h2 : coffee = 18)
  (h3 : tea = 15)
  (h4 : coffee_and_tea = 7)
  (h5 : juice = 6)
  (h6 : juice_and_tea_not_coffee = 3) : 
  total - ((coffee + tea - coffee_and_tea) + (juice - juice_and_tea_not_coffee)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_drink_count_l380_38042


namespace NUMINAMATH_CALUDE_cubic_factorization_l380_38021

theorem cubic_factorization (x : ℝ) : x^3 - 16*x = x*(x+4)*(x-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l380_38021


namespace NUMINAMATH_CALUDE_hash_one_two_three_l380_38088

/-- The operation # defined for real numbers a, b, and c -/
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem stating that #(1, 2, 3) = -8 -/
theorem hash_one_two_three : hash 1 2 3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_hash_one_two_three_l380_38088


namespace NUMINAMATH_CALUDE_triangle_area_l380_38010

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  (1/2) * a * b = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l380_38010


namespace NUMINAMATH_CALUDE_inequality_solution_l380_38049

theorem inequality_solution (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 8) ≥ 1 / 2 ↔ x ∈ Set.Ioc (-8) (-4) ∪ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l380_38049


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l380_38016

-- Define the vectors
def a : ℝ × ℝ := (4, -3)
def b (x : ℝ) : ℝ × ℝ := (x, 6)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem parallel_vectors_x_value :
  parallel a (b x) → x = -8 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l380_38016


namespace NUMINAMATH_CALUDE_math_interest_group_problem_l380_38027

theorem math_interest_group_problem (m n : ℕ) : 
  m * (m - 1) / 2 + m * n + n = 51 → m = 6 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_math_interest_group_problem_l380_38027


namespace NUMINAMATH_CALUDE_video_votes_l380_38057

theorem video_votes (up_votes : ℕ) (ratio_up : ℕ) (ratio_down : ℕ) (down_votes : ℕ) : 
  up_votes = 18 →
  ratio_up = 9 →
  ratio_down = 2 →
  up_votes * ratio_down = down_votes * ratio_up →
  down_votes = 4 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l380_38057


namespace NUMINAMATH_CALUDE_complement_A_in_U_l380_38006

-- Define the universal set U
def U : Set ℝ := {x | x > 1}

-- Define set A
def A : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l380_38006


namespace NUMINAMATH_CALUDE_minsu_running_time_l380_38028

theorem minsu_running_time 
  (total_distance : Real) 
  (speed : Real) 
  (distance_remaining : Real) : Real :=
  let distance_run := total_distance - distance_remaining
  let time_elapsed := distance_run / speed
  have h1 : total_distance = 120 := by sorry
  have h2 : speed = 4 := by sorry
  have h3 : distance_remaining = 20 := by sorry
  have h4 : time_elapsed = 25 := by sorry
  time_elapsed

#check minsu_running_time

end NUMINAMATH_CALUDE_minsu_running_time_l380_38028


namespace NUMINAMATH_CALUDE_divisor_problem_l380_38066

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a number is divisible by another -/
def is_divisible_by (a b : ℕ) : Prop := sorry

theorem divisor_problem (n : ℕ+) (k : ℕ) :
  num_divisors n = 72 →
  num_divisors (5 * n) = 120 →
  (∀ m : ℕ, m > k → ¬ is_divisible_by n (5^m)) →
  is_divisible_by n (5^k) →
  k = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l380_38066


namespace NUMINAMATH_CALUDE_adult_meal_cost_l380_38060

/-- Calculates the cost of an adult meal given the total number of people,
    number of kids, and total cost for a group at a restaurant where kids eat free. -/
theorem adult_meal_cost (total_people : ℕ) (num_kids : ℕ) (total_cost : ℚ) :
  total_people = 12 →
  num_kids = 7 →
  total_cost = 15 →
  (total_cost / (total_people - num_kids) : ℚ) = 3 := by
  sorry

#check adult_meal_cost

end NUMINAMATH_CALUDE_adult_meal_cost_l380_38060


namespace NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l380_38029

/-- Given a cube where the sum of the lengths of all edges is 48 cm, 
    prove that its volume is 64 cubic centimeters. -/
theorem cube_volume_from_total_edge_length : 
  ∀ (edge_length : ℝ), 
    (12 * edge_length = 48) →
    (edge_length^3 = 64) := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l380_38029


namespace NUMINAMATH_CALUDE_sum_base6_1452_2354_l380_38098

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- The main theorem: sum of 1452₆ and 2354₆ in base 6 is 4250₆ -/
theorem sum_base6_1452_2354 :
  decimalToBase6 (base6ToDecimal [1, 4, 5, 2] + base6ToDecimal [2, 3, 5, 4]) = [4, 2, 5, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_base6_1452_2354_l380_38098


namespace NUMINAMATH_CALUDE_complex_product_equals_33_l380_38097

theorem complex_product_equals_33 (x : ℂ) (h : x = Complex.exp (2 * π * I / 9)) :
  (2 * x + x^2) * (2 * x^2 + x^4) * (2 * x^3 + x^6) * (2 * x^4 + x^8) = 33 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_33_l380_38097


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l380_38095

def f (x : ℝ) : ℝ := x^4 - 2*x^3

theorem tangent_line_at_one (x y : ℝ) :
  (y - f 1 = (4 - 6) * (x - 1)) ↔ (y = -2*x + 1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l380_38095


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l380_38007

theorem smallest_whole_number_above_sum : ∃ n : ℕ, 
  (n : ℚ) > (3 + 1/3 : ℚ) + (4 + 1/6 : ℚ) + (5 + 1/12 : ℚ) + (6 + 1/8 : ℚ) ∧
  ∀ m : ℕ, (m : ℚ) > (3 + 1/3 : ℚ) + (4 + 1/6 : ℚ) + (5 + 1/12 : ℚ) + (6 + 1/8 : ℚ) → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l380_38007


namespace NUMINAMATH_CALUDE_consecutive_circle_selections_l380_38078

/-- Represents the arrangement of circles in the figure -/
structure CircleArrangement where
  total_circles : Nat
  long_side_rows : Nat
  long_side_ways : Nat
  diagonal_ways : Nat

/-- The specific arrangement for our problem -/
def problem_arrangement : CircleArrangement :=
  { total_circles := 33
  , long_side_rows := 6
  , long_side_ways := 21
  , diagonal_ways := 18 }

/-- Calculates the total number of ways to select three consecutive circles -/
def count_consecutive_selections (arr : CircleArrangement) : Nat :=
  arr.long_side_ways + 2 * arr.diagonal_ways

/-- Theorem stating that there are 57 ways to select three consecutive circles -/
theorem consecutive_circle_selections :
  count_consecutive_selections problem_arrangement = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_circle_selections_l380_38078


namespace NUMINAMATH_CALUDE_consecutive_squares_not_perfect_square_l380_38096

theorem consecutive_squares_not_perfect_square (n : ℕ) : 
  ∃ k : ℕ, (n - 1)^2 + n^2 + (n + 1)^2 ≠ k^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_not_perfect_square_l380_38096


namespace NUMINAMATH_CALUDE_inequalities_and_range_l380_38059

theorem inequalities_and_range :
  (∀ x : ℝ, x > 1 → 2 * Real.log x < x - 1/x) ∧
  (∀ a : ℝ, a > 0 → (∀ t : ℝ, t > 0 → (1 + a/t) * Real.log (1 + t) > a) ↔ 0 < a ∧ a ≤ 2) ∧
  ((9/10 : ℝ)^19 < 1/Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_and_range_l380_38059


namespace NUMINAMATH_CALUDE_special_function_at_one_fifth_l380_38031

/-- A monotonic function on (0, +∞) satisfying f(f(x) - 1/x) = 2 for all x > 0 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y ∨ f x > f y) ∧
  (∀ x, 0 < x → f (f x - 1/x) = 2)

/-- The value of f(1/5) for a special function f -/
theorem special_function_at_one_fifth
    (f : ℝ → ℝ) (h : special_function f) : f (1/5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_one_fifth_l380_38031


namespace NUMINAMATH_CALUDE_product_of_eight_consecutive_odd_numbers_divisible_by_ten_l380_38091

theorem product_of_eight_consecutive_odd_numbers_divisible_by_ten (n : ℕ) (h : Odd n) :
  ∃ k : ℕ, (n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) * (n + 14)) = 10 * k :=
by
  sorry

#check product_of_eight_consecutive_odd_numbers_divisible_by_ten

end NUMINAMATH_CALUDE_product_of_eight_consecutive_odd_numbers_divisible_by_ten_l380_38091


namespace NUMINAMATH_CALUDE_f_derivative_and_tangent_line_l380_38034

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + x * Real.log x

-- State the theorem
theorem f_derivative_and_tangent_line :
  -- The derivative of f(x)
  (∀ x : ℝ, x > 0 → HasDerivAt f (2*x + Real.log x + 1) x) ∧
  -- The equation of the tangent line at x = 1
  (∃ A B C : ℝ, A = 3 ∧ B = -1 ∧ C = -2 ∧
    ∀ x y : ℝ, (x = 1 ∧ y = f 1) → (A*x + B*y + C = 0)) := by
  sorry

end

end NUMINAMATH_CALUDE_f_derivative_and_tangent_line_l380_38034


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l380_38041

theorem min_value_geometric_sequence (a₁ a₂ a₃ : ℝ) :
  a₁ = 1 →
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) →
  (∀ b₂ b₃ : ℝ, (∃ s : ℝ, b₂ = a₁ * s ∧ b₃ = b₂ * s) → 4 * a₂ + 5 * a₃ ≤ 4 * b₂ + 5 * b₃) →
  4 * a₂ + 5 * a₃ = -4/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l380_38041


namespace NUMINAMATH_CALUDE_leadership_selection_count_l380_38033

/-- The number of people in the group -/
def n : ℕ := 5

/-- The number of positions to be filled (leader and deputy) -/
def k : ℕ := 2

/-- The number of ways to select a leader and deputy with no restrictions -/
def total_selections : ℕ := n * (n - 1)

/-- The number of invalid selections (when the restricted person is deputy) -/
def invalid_selections : ℕ := n - 1

/-- The number of valid selections -/
def valid_selections : ℕ := total_selections - invalid_selections

theorem leadership_selection_count :
  valid_selections = 16 :=
sorry

end NUMINAMATH_CALUDE_leadership_selection_count_l380_38033


namespace NUMINAMATH_CALUDE_polynomial_value_l380_38036

theorem polynomial_value (a b : ℝ) (h : |a - 2| + (b + 1/2)^2 = 0) :
  (2*a*b^2 + a^2*b) - (3*a*b^2 + a^2*b - 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l380_38036


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l380_38070

def ends_in_6 (n : ℕ) : Prop := n % 10 = 6

def move_6_to_front (n : ℕ) : ℕ :=
  let d := (Nat.log 10 n) + 1
  6 * (10 ^ d) + (n / 10)

theorem smallest_number_with_properties :
  ∃ (n : ℕ),
    ends_in_6 n ∧
    move_6_to_front n = 4 * n ∧
    ∀ (m : ℕ), (ends_in_6 m ∧ move_6_to_front m = 4 * m) → n ≤ m ∧
    n = 153846 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_properties_l380_38070


namespace NUMINAMATH_CALUDE_nell_card_difference_l380_38023

/-- Represents Nell's card collection --/
structure CardCollection where
  initial_baseball : Nat
  initial_ace : Nat
  current_baseball : Nat
  current_ace : Nat

/-- Calculates the difference between baseball and Ace cards --/
def card_difference (c : CardCollection) : Int :=
  c.current_baseball - c.current_ace

/-- Theorem stating the difference between Nell's baseball and Ace cards --/
theorem nell_card_difference (nell : CardCollection)
  (h1 : nell.initial_baseball = 438)
  (h2 : nell.initial_ace = 18)
  (h3 : nell.current_baseball = 178)
  (h4 : nell.current_ace = 55) :
  card_difference nell = 123 := by
  sorry

end NUMINAMATH_CALUDE_nell_card_difference_l380_38023


namespace NUMINAMATH_CALUDE_original_number_of_people_l380_38003

theorem original_number_of_people (x : ℕ) : 
  (x / 2 : ℚ) = 18 → x = 36 := by sorry

end NUMINAMATH_CALUDE_original_number_of_people_l380_38003


namespace NUMINAMATH_CALUDE_jackets_sold_after_noon_l380_38065

theorem jackets_sold_after_noon :
  let total_jackets : ℕ := 214
  let price_before_noon : ℚ := 31.95
  let price_after_noon : ℚ := 18.95
  let total_receipts : ℚ := 5108.30
  let jackets_after_noon : ℕ := 133
  let jackets_before_noon : ℕ := total_jackets - jackets_after_noon
  (jackets_before_noon : ℚ) * price_before_noon + (jackets_after_noon : ℚ) * price_after_noon = total_receipts :=
by
  sorry

#check jackets_sold_after_noon

end NUMINAMATH_CALUDE_jackets_sold_after_noon_l380_38065


namespace NUMINAMATH_CALUDE_base_nine_to_ten_l380_38085

theorem base_nine_to_ten : 
  (3 * 9^4 + 9 * 9^3 + 4 * 9^2 + 5 * 9^1 + 7 * 9^0) = 26620 := by
  sorry

end NUMINAMATH_CALUDE_base_nine_to_ten_l380_38085


namespace NUMINAMATH_CALUDE_simplify_expression_l380_38040

theorem simplify_expression (w : ℝ) : 2*w + 3 - 4*w - 5 + 6*w + 2^3 - 8*w - 9 = -4*w - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l380_38040


namespace NUMINAMATH_CALUDE_solve_equation_l380_38013

theorem solve_equation (x : ℝ) (h : (128 / x) + (75 / x) + (57 / x) = 6.5) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l380_38013


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_fifty_l380_38099

theorem largest_multiple_of_seven_below_negative_fifty :
  ∀ n : ℤ, n * 7 < -50 → n * 7 ≤ -56 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_fifty_l380_38099


namespace NUMINAMATH_CALUDE_least_tiles_required_l380_38008

/-- The length of the room in centimeters -/
def room_length : ℕ := 1517

/-- The breadth of the room in centimeters -/
def room_breadth : ℕ := 902

/-- The greatest common divisor of the room length and breadth -/
def tile_side : ℕ := Nat.gcd room_length room_breadth

/-- The area of the room in square centimeters -/
def room_area : ℕ := room_length * room_breadth

/-- The area of a single tile in square centimeters -/
def tile_area : ℕ := tile_side * tile_side

/-- The number of tiles required to pave the room -/
def num_tiles : ℕ := (room_area + tile_area - 1) / tile_area

theorem least_tiles_required :
  num_tiles = 814 :=
sorry

end NUMINAMATH_CALUDE_least_tiles_required_l380_38008


namespace NUMINAMATH_CALUDE_right_triangle_sides_l380_38020

theorem right_triangle_sides (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 30 →
  x^2 + y^2 + z^2 = 338 →
  x^2 + y^2 = z^2 →
  ((x = 5 ∧ y = 12 ∧ z = 13) ∨ (x = 12 ∧ y = 5 ∧ z = 13)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l380_38020


namespace NUMINAMATH_CALUDE_distance_traveled_l380_38025

/-- Given a person traveling at 40 km/hr for 6 hours, prove that the distance traveled is 240 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 40) (h2 : time = 6) :
  speed * time = 240 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l380_38025


namespace NUMINAMATH_CALUDE_difference_of_squares_l380_38051

theorem difference_of_squares (x : ℝ) : x^2 - 25 = (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l380_38051


namespace NUMINAMATH_CALUDE_topological_minor_theorem_l380_38076

-- Define the average degree of a graph
def average_degree (G : Graph) : ℝ := sorry

-- Define what it means for a graph to contain another graph as a topological minor
def contains_topological_minor (G H : Graph) : Prop := sorry

-- Define the complete graph on r vertices
def complete_graph (r : ℕ) : Graph := sorry

theorem topological_minor_theorem :
  ∃ (c : ℝ), c = 10 ∧
  ∀ (r : ℕ) (G : Graph),
    average_degree G ≥ c * r^2 →
    contains_topological_minor G (complete_graph r) :=
sorry

end NUMINAMATH_CALUDE_topological_minor_theorem_l380_38076


namespace NUMINAMATH_CALUDE_min_hours_sixth_week_l380_38061

/-- The required average number of hours per week -/
def required_average : ℚ := 12

/-- The number of weeks -/
def total_weeks : ℕ := 6

/-- The hours worked in the first 5 weeks -/
def first_five_weeks : List ℕ := [9, 10, 14, 11, 8]

/-- The sum of hours worked in the first 5 weeks -/
def sum_first_five : ℕ := first_five_weeks.sum

theorem min_hours_sixth_week : 
  ∀ x : ℕ, 
    (sum_first_five + x : ℚ) / total_weeks ≥ required_average → 
    x ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_min_hours_sixth_week_l380_38061


namespace NUMINAMATH_CALUDE_some_flying_creatures_are_magical_l380_38082

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (unicorn : U → Prop)
variable (flying : U → Prop)
variable (magical : U → Prop)

-- State the theorem
theorem some_flying_creatures_are_magical :
  (∀ x, unicorn x → flying x) →  -- All unicorns are capable of flying
  (∃ x, magical x ∧ unicorn x) →  -- Some magical creatures are unicorns
  (∃ x, flying x ∧ magical x) :=  -- Some flying creatures are magical creatures
by
  sorry

end NUMINAMATH_CALUDE_some_flying_creatures_are_magical_l380_38082


namespace NUMINAMATH_CALUDE_function_decreasing_implies_a_range_l380_38069

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_function_decreasing_implies_a_range_l380_38069


namespace NUMINAMATH_CALUDE_point_in_region_l380_38045

theorem point_in_region (m : ℝ) : 
  (2 * m + 3 * 1 - 5 > 0) ↔ (m > 1) := by sorry

end NUMINAMATH_CALUDE_point_in_region_l380_38045


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l380_38086

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 3 * n ≡ 1356 [ZMOD 22]) → n ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l380_38086


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l380_38064

theorem rhombus_perimeter (d : ℝ) (h1 : d = 20) : 
  let longer_diagonal := 1.3 * d
  let side := Real.sqrt ((d/2)^2 + (longer_diagonal/2)^2)
  4 * side = 4 * Real.sqrt 269 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l380_38064


namespace NUMINAMATH_CALUDE_quadratic_intercept_distance_l380_38005

/-- Given a quadratic function f(x) = x² + ax + b, where the line from (0, b) to one x-intercept
    is perpendicular to y = x, prove that the distance from (0, 0) to the other x-intercept is 1. -/
theorem quadratic_intercept_distance (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  let x₁ := -b  -- One x-intercept
  let x₂ := 1   -- The other x-intercept (to be proven)
  (∀ x, f x = 0 → x = x₁ ∨ x = x₂) →  -- x₁ and x₂ are the only roots
  (x₁ + x₂ = -a ∧ x₁ * x₂ = b) →      -- Vieta's formulas
  (b ≠ 0) →                           -- Ensuring non-zero y-intercept
  (∀ x y, y = -x + b → f x = y) →     -- Line from (0, b) to (x₁, 0) has equation y = -x + b
  x₂ = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intercept_distance_l380_38005


namespace NUMINAMATH_CALUDE_debate_schedule_ways_l380_38067

/-- Number of debaters from each school -/
def num_debaters : ℕ := 4

/-- Total number of debates -/
def total_debates : ℕ := num_debaters * num_debaters

/-- Maximum number of debates per session -/
def max_debates_per_session : ℕ := 3

/-- Number of ways to schedule debates -/
def schedule_ways : ℕ := 20922789888000

/-- Theorem stating the number of ways to schedule debates -/
theorem debate_schedule_ways :
  (total_debates.factorial) / (max_debates_per_session.factorial ^ 5 * 1) = schedule_ways := by
  sorry

end NUMINAMATH_CALUDE_debate_schedule_ways_l380_38067


namespace NUMINAMATH_CALUDE_percentage_calculation_l380_38094

theorem percentage_calculation (N P : ℝ) (h1 : N = 75) (h2 : N = (P / 100) * N + 63) : P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l380_38094


namespace NUMINAMATH_CALUDE_min_value_trigonometric_fraction_l380_38030

theorem min_value_trigonometric_fraction (a b : ℝ) (θ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hθ : θ ∈ Set.Ioo 0 (π / 2)) :
  a / Real.sin θ + b / Real.cos θ ≥ (Real.rpow a (2/3) + Real.rpow b (2/3))^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_fraction_l380_38030


namespace NUMINAMATH_CALUDE_reflect_2_5_across_x_axis_l380_38018

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Theorem: Reflecting the point (2,5) across the x-axis results in (2,-5) -/
theorem reflect_2_5_across_x_axis :
  reflectAcrossXAxis { x := 2, y := 5 } = { x := 2, y := -5 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_2_5_across_x_axis_l380_38018


namespace NUMINAMATH_CALUDE_preston_order_calculation_l380_38073

/-- The total amount Preston received from Abra Company's order -/
def total_received (sandwich_price : ℚ) (delivery_fee : ℚ) (num_sandwiches : ℕ) (tip_percentage : ℚ) : ℚ :=
  let subtotal := sandwich_price * num_sandwiches + delivery_fee
  subtotal + subtotal * tip_percentage

/-- Preston's sandwich shop order calculation -/
theorem preston_order_calculation :
  total_received 5 20 18 (1/10) = 121 := by
  sorry

end NUMINAMATH_CALUDE_preston_order_calculation_l380_38073


namespace NUMINAMATH_CALUDE_equality_of_ratios_l380_38032

theorem equality_of_ratios (a b c d : ℕ) 
  (h1 : a / c = b / d) 
  (h2 : a / c = (a * b + 1) / (c * d + 1)) 
  (h3 : b / d = (a * b + 1) / (c * d + 1)) : 
  a = c ∧ b = d := by
  sorry

end NUMINAMATH_CALUDE_equality_of_ratios_l380_38032


namespace NUMINAMATH_CALUDE_diplomats_language_theorem_l380_38058

theorem diplomats_language_theorem (total : ℕ) (japanese : ℕ) (not_russian : ℕ) (both_percent : ℚ) :
  total = 120 →
  japanese = 20 →
  not_russian = 32 →
  both_percent = 1/10 →
  (↑(total - (japanese + (total - not_russian) - (both_percent * ↑total).num)) / ↑total : ℚ) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_diplomats_language_theorem_l380_38058


namespace NUMINAMATH_CALUDE_banana_cost_l380_38002

/-- Given that 5 dozen bananas cost $24.00, prove that 4 dozen bananas at the same rate will cost $19.20 -/
theorem banana_cost (total_cost : ℝ) (total_dozens : ℕ) (target_dozens : ℕ) 
  (h1 : total_cost = 24)
  (h2 : total_dozens = 5)
  (h3 : target_dozens = 4) :
  (target_dozens : ℝ) * (total_cost / total_dozens) = 19.2 :=
by sorry

end NUMINAMATH_CALUDE_banana_cost_l380_38002


namespace NUMINAMATH_CALUDE_percentage_increase_men_is_twenty_percent_l380_38053

/-- Represents the population data and conditions --/
structure PopulationData where
  men_1990 : ℕ
  women_1990 : ℕ
  boys_1990 : ℕ
  total_1994 : ℕ
  boys_1994 : ℕ

/-- Calculates the percentage increase in men given the population data --/
def percentageIncreaseMen (data : PopulationData) : ℚ :=
  let women_1994 := data.women_1990 + data.boys_1990 * data.women_1990 / data.women_1990
  let men_1994 := data.total_1994 - women_1994 - data.boys_1994
  (men_1994 - data.men_1990) * 100 / data.men_1990

/-- Theorem stating that the percentage increase in men is 20% --/
theorem percentage_increase_men_is_twenty_percent (data : PopulationData) 
  (h1 : data.men_1990 = 5000)
  (h2 : data.women_1990 = 3000)
  (h3 : data.boys_1990 = 2000)
  (h4 : data.total_1994 = 13000)
  (h5 : data.boys_1994 = data.boys_1990) :
  percentageIncreaseMen data = 20 := by
  sorry

#eval percentageIncreaseMen {
  men_1990 := 5000,
  women_1990 := 3000,
  boys_1990 := 2000,
  total_1994 := 13000,
  boys_1994 := 2000
}

end NUMINAMATH_CALUDE_percentage_increase_men_is_twenty_percent_l380_38053


namespace NUMINAMATH_CALUDE_expand_expression_l380_38024

theorem expand_expression (m n : ℝ) : (2*m + n - 1) * (2*m - n + 1) = 4*m^2 - n^2 + 2*n - 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l380_38024


namespace NUMINAMATH_CALUDE_solve_watermelon_problem_l380_38071

def watermelon_problem (n : ℕ) (initial_avg : ℝ) (new_weight : ℝ) (new_avg : ℝ) : Prop :=
  let total_initial := n * initial_avg
  let replaced_weight := total_initial + new_weight - n * new_avg
  replaced_weight = 3

theorem solve_watermelon_problem :
  watermelon_problem 10 4.2 5 4.4 := by sorry

end NUMINAMATH_CALUDE_solve_watermelon_problem_l380_38071


namespace NUMINAMATH_CALUDE_consecutive_integer_averages_l380_38087

theorem consecutive_integer_averages (a b : ℤ) (h_positive : a > 0) : 
  ((7 * a + 21) / 7 = b) → 
  ((7 * b + 21) / 7 = a + 6) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integer_averages_l380_38087


namespace NUMINAMATH_CALUDE_cube_volume_problem_l380_38039

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) →
  ((a + 2) * (a - 2) * a = a^3 - 8) →
  (a^3 = 8) :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l380_38039


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_to_plane_l380_38046

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_parallel_perpendicular_to_plane 
  (a b : Line) (α : Plane) :
  parallel a b → perpendicular b α → perpendicular a α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_to_plane_l380_38046


namespace NUMINAMATH_CALUDE_plate_count_l380_38072

theorem plate_count (n : ℕ) 
  (h1 : 500 < n ∧ n < 600)
  (h2 : n % 10 = 7)
  (h3 : n % 12 = 7) : 
  n = 547 := by
sorry

end NUMINAMATH_CALUDE_plate_count_l380_38072


namespace NUMINAMATH_CALUDE_chef_dinner_meals_l380_38056

/-- Calculates the number of meals prepared for dinner given lunch and dinner information -/
def meals_prepared_for_dinner (lunch_prepared : ℕ) (lunch_sold : ℕ) (dinner_total : ℕ) : ℕ :=
  dinner_total - (lunch_prepared - lunch_sold)

/-- Proves that the chef prepared 5 meals for dinner -/
theorem chef_dinner_meals :
  meals_prepared_for_dinner 17 12 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_chef_dinner_meals_l380_38056
