import Mathlib

namespace NUMINAMATH_CALUDE_binary_arithmetic_theorem_l373_37347

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ ⟨i, bi⟩ acc => acc + if bi then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

def binary_add (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a + binary_to_decimal b)

def binary_sub (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a - binary_to_decimal b)

theorem binary_arithmetic_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [false, true, false, true] -- 1010₂
  let d := [true, false, false, true] -- 1001₂
  binary_add (binary_sub (binary_add a b) c) d = [true, false, false, false, true] -- 10001₂
  := by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_theorem_l373_37347


namespace NUMINAMATH_CALUDE_flag_pole_height_l373_37323

/-- Given a tree and a flag pole casting shadows at the same time, 
    calculate the height of the flag pole. -/
theorem flag_pole_height 
  (tree_height : ℝ) 
  (tree_shadow : ℝ) 
  (flag_shadow : ℝ) 
  (h_tree_height : tree_height = 12)
  (h_tree_shadow : tree_shadow = 8)
  (h_flag_shadow : flag_shadow = 100) :
  (tree_height / tree_shadow) * flag_shadow = 150 :=
by
  sorry

#check flag_pole_height

end NUMINAMATH_CALUDE_flag_pole_height_l373_37323


namespace NUMINAMATH_CALUDE_net_profit_is_107_70_l373_37337

/-- Laundry shop rates and quantities for a three-day period --/
structure LaundryData where
  regular_rate : ℝ
  delicate_rate : ℝ
  business_rate : ℝ
  bulky_rate : ℝ
  discount_rate : ℝ
  day1_regular : ℝ
  day1_delicate : ℝ
  day1_business : ℝ
  day1_bulky : ℝ
  day2_regular : ℝ
  day2_delicate : ℝ
  day2_business : ℝ
  day2_bulky : ℝ
  day3_regular : ℝ
  day3_delicate : ℝ
  day3_business : ℝ
  day3_bulky : ℝ
  overhead_costs : ℝ

/-- Calculate the net profit for a three-day period in a laundry shop --/
def calculate_net_profit (data : LaundryData) : ℝ :=
  let day1_total := data.regular_rate * data.day1_regular +
                    data.delicate_rate * data.day1_delicate +
                    data.business_rate * data.day1_business +
                    data.bulky_rate * data.day1_bulky
  let day2_total := data.regular_rate * data.day2_regular +
                    data.delicate_rate * data.day2_delicate +
                    data.business_rate * data.day2_business +
                    data.bulky_rate * data.day2_bulky
  let day3_total := (data.regular_rate * data.day3_regular +
                    data.delicate_rate * data.day3_delicate +
                    data.business_rate * data.day3_business +
                    data.bulky_rate * data.day3_bulky) * (1 - data.discount_rate)
  day1_total + day2_total + day3_total - data.overhead_costs

/-- Theorem: The net profit for the given three-day period is $107.70 --/
theorem net_profit_is_107_70 :
  let data := LaundryData.mk 3 4 5 6 0.1 7 4 3 2 10 6 4 3 20 4 5 2 150
  calculate_net_profit data = 107.7 := by
  sorry

end NUMINAMATH_CALUDE_net_profit_is_107_70_l373_37337


namespace NUMINAMATH_CALUDE_line_points_property_l373_37359

theorem line_points_property (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -2 * x₁ + 3)
  (h2 : y₂ = -2 * x₂ + 3)
  (h3 : y₃ = -2 * x₃ + 3)
  (h4 : x₁ < x₂)
  (h5 : x₂ < x₃)
  (h6 : x₂ * x₃ < 0) :
  y₁ * y₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_points_property_l373_37359


namespace NUMINAMATH_CALUDE_last_three_digits_of_8_to_1000_l373_37319

theorem last_three_digits_of_8_to_1000 (h : 8^125 ≡ 2 [ZMOD 1250]) :
  8^1000 ≡ 256 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_8_to_1000_l373_37319


namespace NUMINAMATH_CALUDE_base6_subtraction_431_254_l373_37308

/-- Represents a number in base 6 using a list of digits -/
def Base6 : Type := List Nat

/-- Converts a base 6 number to its decimal representation -/
def to_decimal (n : Base6) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Subtracts two base 6 numbers -/
def base6_sub (a b : Base6) : Base6 :=
  sorry -- Implementation details omitted

theorem base6_subtraction_431_254 :
  base6_sub [1, 3, 4] [4, 5, 2] = [3, 3, 1] :=
by sorry

end NUMINAMATH_CALUDE_base6_subtraction_431_254_l373_37308


namespace NUMINAMATH_CALUDE_min_value_of_squares_l373_37390

theorem min_value_of_squares (a b t c : ℝ) (h : a + b = t) :
  ∃ m : ℝ, m = (t^2 + c^2 - 2*t*c + 2*c^2) / 2 ∧ 
  ∀ x y : ℝ, x + y = t → x^2 + (y + c)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l373_37390


namespace NUMINAMATH_CALUDE_route_redistribution_possible_l373_37309

/-- Represents an airline with its routes -/
structure Airline where
  id : Nat
  routes : Finset (Nat × Nat)

/-- Represents the initial configuration of airlines -/
def initial_airlines (k : Nat) : Finset Airline :=
  sorry

/-- Checks if an airline complies with the one-route-per-city law -/
def complies_with_law (a : Airline) : Prop :=
  sorry

/-- Checks if all airlines have the same number of routes -/
def equal_routes (airlines : Finset Airline) : Prop :=
  sorry

theorem route_redistribution_possible (k : Nat) :
  ∃ (new_airlines : Finset Airline),
    (∀ a ∈ new_airlines, complies_with_law a) ∧
    equal_routes new_airlines ∧
    new_airlines.card = (initial_airlines k).card :=
  sorry

end NUMINAMATH_CALUDE_route_redistribution_possible_l373_37309


namespace NUMINAMATH_CALUDE_percentage_of_B_grades_l373_37366

def scores : List ℕ := [86, 73, 55, 98, 76, 93, 88, 72, 77, 62, 81, 79, 68, 82, 91]

def is_grade_B (score : ℕ) : Bool :=
  87 ≤ score ∧ score ≤ 93

def count_grade_B (scores : List ℕ) : ℕ :=
  (scores.filter is_grade_B).length

theorem percentage_of_B_grades :
  (count_grade_B scores : ℚ) / (scores.length : ℚ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_B_grades_l373_37366


namespace NUMINAMATH_CALUDE_neighborhood_cleanup_weight_l373_37338

/-- The total weight of litter collected during a neighborhood clean-up. -/
def total_litter_weight (gina_bags : ℕ) (neighborhood_multiplier : ℕ) (bag_weight : ℕ) : ℕ :=
  (gina_bags + neighborhood_multiplier * gina_bags) * bag_weight

/-- Theorem stating that the total weight of litter collected is 664 pounds. -/
theorem neighborhood_cleanup_weight :
  total_litter_weight 2 82 4 = 664 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_cleanup_weight_l373_37338


namespace NUMINAMATH_CALUDE_max_value_of_expressions_l373_37307

theorem max_value_of_expressions :
  let expr1 := 3 + 1 + 2 + 4
  let expr2 := 3 * 1 + 2 + 4
  let expr3 := 3 + 1 * 2 + 4
  let expr4 := 3 + 1 + 2 * 4
  let expr5 := 3 * 1 * 2 * 4
  max expr1 (max expr2 (max expr3 (max expr4 expr5))) = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expressions_l373_37307


namespace NUMINAMATH_CALUDE_billy_hike_distance_l373_37335

theorem billy_hike_distance (east north : ℝ) (h1 : east = 7) (h2 : north = 8 * (Real.sqrt 2) / 2) :
  Real.sqrt (east^2 + north^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_billy_hike_distance_l373_37335


namespace NUMINAMATH_CALUDE_no_solution_set_characterization_l373_37379

/-- The quadratic function f(x) = x² - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The set of values k for which f(x) = k has no real solutions -/
def no_solution_set : Set ℝ := {k | ∀ x, f x ≠ k}

/-- Theorem stating that the no_solution_set is equivalent to {k | k < 1} -/
theorem no_solution_set_characterization :
  no_solution_set = {k | k < 1} := by sorry

end NUMINAMATH_CALUDE_no_solution_set_characterization_l373_37379


namespace NUMINAMATH_CALUDE_smallest_proportional_part_l373_37313

theorem smallest_proportional_part (total : ℕ) (parts : List ℕ) : 
  total = 360 → 
  parts = [5, 7, 4, 8] → 
  (parts.sum : ℚ) > 0 → 
  let proportional_parts := parts.map (λ p => (p : ℚ) * total / parts.sum)
  List.minimum proportional_parts = some 60 := by
sorry

end NUMINAMATH_CALUDE_smallest_proportional_part_l373_37313


namespace NUMINAMATH_CALUDE_divisible_by_120_l373_37355

theorem divisible_by_120 (n : ℕ+) : 120 ∣ n * (n^2 - 1) * (n^2 - 5*n + 26) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_120_l373_37355


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l373_37370

theorem sarahs_bowling_score (greg_score sarah_score : ℕ) : 
  sarah_score = greg_score + 60 →
  (sarah_score + greg_score) / 2 = 108 →
  sarah_score = 138 := by
sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l373_37370


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_expression_equals_one_l373_37317

theorem tan_alpha_two_implies_expression_equals_one (α : Real) 
  (h : Real.tan α = 2) : 
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_expression_equals_one_l373_37317


namespace NUMINAMATH_CALUDE_solution_pairs_l373_37349

theorem solution_pairs (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_solution_pairs_l373_37349


namespace NUMINAMATH_CALUDE_jr_high_selection_theorem_l373_37357

/-- Represents the structure of a school with different grade levels and classes --/
structure School where
  elem_grades : Nat
  elem_classes_per_grade : Nat
  jr_high_grades : Nat
  jr_high_classes_per_grade : Nat
  high_grades : Nat
  high_classes_per_grade : Nat

/-- Calculates the total number of classes in the school --/
def total_classes (s : School) : Nat :=
  s.elem_grades * s.elem_classes_per_grade +
  s.jr_high_grades * s.jr_high_classes_per_grade +
  s.high_grades * s.high_classes_per_grade

/-- Calculates the number of classes to be selected from each grade in junior high --/
def jr_high_classes_selected (s : School) (total_selected : Nat) : Nat :=
  (total_selected * s.jr_high_classes_per_grade) / (total_classes s)

theorem jr_high_selection_theorem (s : School) (total_selected : Nat) :
  s.elem_grades = 6 →
  s.elem_classes_per_grade = 6 →
  s.jr_high_grades = 3 →
  s.jr_high_classes_per_grade = 8 →
  s.high_grades = 3 →
  s.high_classes_per_grade = 12 →
  total_selected = 36 →
  jr_high_classes_selected s total_selected = 2 := by
  sorry

end NUMINAMATH_CALUDE_jr_high_selection_theorem_l373_37357


namespace NUMINAMATH_CALUDE_hypotenuse_increase_bound_l373_37305

theorem hypotenuse_increase_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let c := Real.sqrt (x^2 + y^2)
  let c' := Real.sqrt ((x + 1)^2 + (y + 1)^2)
  c' - c ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_increase_bound_l373_37305


namespace NUMINAMATH_CALUDE_line_intersects_circle_l373_37328

/-- Given a point M(x₀, y₀) outside the circle x² + y² = 2,
    prove that the line x₀x + y₀y = 2 intersects the circle. -/
theorem line_intersects_circle (x₀ y₀ : ℝ) (h : x₀^2 + y₀^2 > 2) :
  ∃ (x y : ℝ), x^2 + y^2 = 2 ∧ x₀*x + y₀*y = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l373_37328


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l373_37318

theorem imaginary_part_of_complex_expression :
  let z : ℂ := 1 - I
  (z^2 + 2/z).im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l373_37318


namespace NUMINAMATH_CALUDE_smallest_divisible_page_number_l373_37316

theorem smallest_divisible_page_number : 
  (∀ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 13 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧ 17 ∣ n → n ≥ 68068) ∧ 
  (4 ∣ 68068) ∧ (13 ∣ 68068) ∧ (7 ∣ 68068) ∧ (11 ∣ 68068) ∧ (17 ∣ 68068) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_page_number_l373_37316


namespace NUMINAMATH_CALUDE_number_of_planes_l373_37334

/-- Given an air exhibition with commercial planes, prove the number of planes
    when the total number of wings and wings per plane are known. -/
theorem number_of_planes (total_wings : ℕ) (wings_per_plane : ℕ) (h1 : total_wings = 90) (h2 : wings_per_plane = 2) :
  total_wings / wings_per_plane = 45 := by
  sorry

#check number_of_planes

end NUMINAMATH_CALUDE_number_of_planes_l373_37334


namespace NUMINAMATH_CALUDE_sqrt_problem_l373_37320

theorem sqrt_problem (h1 : Real.sqrt 15129 = 123) (h2 : Real.sqrt x = 0.123) : x = 0.015129 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_problem_l373_37320


namespace NUMINAMATH_CALUDE_min_root_of_negated_quadratic_l373_37397

theorem min_root_of_negated_quadratic (p : ℝ) (r₁ r₂ : ℝ) :
  (∀ x, (x - 19) * (x - 83) = p ↔ x = r₁ ∨ x = r₂) →
  (∃ x, (x - r₁) * (x - r₂) = -p) →
  (∀ x, (x - r₁) * (x - r₂) = -p → x ≥ -19) ∧
  (∃ x, (x - r₁) * (x - r₂) = -p ∧ x = -19) :=
by sorry

end NUMINAMATH_CALUDE_min_root_of_negated_quadratic_l373_37397


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l373_37306

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define a point on the parabola in the first quadrant
def point_on_parabola (Q : ℝ × ℝ) : Prop :=
  parabola Q.1 Q.2 ∧ Q.1 > 0 ∧ Q.2 > 0

-- Define the condition for vector PQ and QF
def vector_condition (P Q : ℝ × ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 2 * ((Q.1 - focus.1)^2 + (Q.2 - focus.2)^2)

-- Main theorem
theorem parabola_focus_distance 
  (Q : ℝ × ℝ) 
  (h1 : point_on_parabola Q) 
  (h2 : ∃ P, vector_condition P Q) : 
  (Q.1 - focus.1)^2 + (Q.2 - focus.2)^2 = (8 + 4*Real.sqrt 2)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l373_37306


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l373_37300

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h : is_arithmetic_sequence a) (h2 : a 2 = 2) (h3 : a 3 = -4) :
  ∃ d : ℤ, d = -6 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l373_37300


namespace NUMINAMATH_CALUDE_fault_line_movement_l373_37310

/-- The total movement of a fault line over two years, given its movement in each year -/
def total_movement (movement_year1 : ℝ) (movement_year2 : ℝ) : ℝ :=
  movement_year1 + movement_year2

/-- Theorem stating that the total movement of the fault line over two years is 6.50 inches -/
theorem fault_line_movement :
  let movement_year1 : ℝ := 1.25
  let movement_year2 : ℝ := 5.25
  total_movement movement_year1 movement_year2 = 6.50 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_movement_l373_37310


namespace NUMINAMATH_CALUDE_eighth_number_in_list_l373_37364

theorem eighth_number_in_list (numbers : List ℕ) : 
  numbers.length = 9 ∧ 
  (numbers.sum : ℚ) / numbers.length = 60 ∧
  numbers.count 54 = 1 ∧
  numbers.count 55 = 1 ∧
  numbers.count 57 = 1 ∧
  numbers.count 58 = 1 ∧
  numbers.count 59 = 1 ∧
  numbers.count 62 = 2 ∧
  numbers.count 65 = 2 →
  numbers.count 53 = 1 := by
sorry

end NUMINAMATH_CALUDE_eighth_number_in_list_l373_37364


namespace NUMINAMATH_CALUDE_age_ratio_proof_l373_37353

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 22 →  -- sum of ages is 22
  b = 8 →  -- b is 8 years old
  b = 2 * c  -- ratio of b's age to c's age is 2:1
:= by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l373_37353


namespace NUMINAMATH_CALUDE_base6_arithmetic_equality_l373_37324

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ := sorry

theorem base6_arithmetic_equality :
  base10ToBase6 ((base6ToBase10 45321 - base6ToBase10 23454) + base6ToBase10 14553) = 45550 := by
  sorry

end NUMINAMATH_CALUDE_base6_arithmetic_equality_l373_37324


namespace NUMINAMATH_CALUDE_rug_area_calculation_l373_37343

/-- Calculates the area of a rug on a rectangular floor with uncovered strips along the edges -/
def rugArea (floorLength floorWidth stripWidth : ℝ) : ℝ :=
  (floorLength - 2 * stripWidth) * (floorWidth - 2 * stripWidth)

/-- Theorem stating that the area of the rug is 204 square meters given the specific dimensions -/
theorem rug_area_calculation :
  rugArea 25 20 4 = 204 := by
  sorry

end NUMINAMATH_CALUDE_rug_area_calculation_l373_37343


namespace NUMINAMATH_CALUDE_x_in_terms_of_y_l373_37371

theorem x_in_terms_of_y (x y : ℚ) (h : 2 * x - 7 * y = 5) : x = (7 * y + 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_in_terms_of_y_l373_37371


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l373_37365

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | -3 < x ∧ x < 2}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x > -3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l373_37365


namespace NUMINAMATH_CALUDE_square_area_difference_l373_37362

/-- Given two line segments where one is 2 cm longer than the other, and the difference 
    of the areas of squares drawn on these line segments is 32 sq. cm, 
    prove that the length of the longer line segment is 9 cm. -/
theorem square_area_difference (x : ℝ) 
  (h1 : (x + 2)^2 - x^2 = 32) : 
  x + 2 = 9 := by sorry

end NUMINAMATH_CALUDE_square_area_difference_l373_37362


namespace NUMINAMATH_CALUDE_passing_marks_l373_37329

theorem passing_marks (T : ℝ) (P : ℝ) 
  (h1 : 0.20 * T = P - 40)
  (h2 : 0.30 * T = P + 20) : 
  P = 160 := by
sorry

end NUMINAMATH_CALUDE_passing_marks_l373_37329


namespace NUMINAMATH_CALUDE_factorization_4a_squared_minus_2a_l373_37336

-- Define what it means for an expression to be a factorization from left to right
def is_factorization_left_to_right (f g : ℝ → ℝ) : Prop :=
  ∃ (h k : ℝ → ℝ), (∀ x, f x = h x * k x) ∧ (∀ x, g x = h x * k x) ∧ (f ≠ g)

-- Define the left side of the equation
def left_side (a : ℝ) : ℝ := 4 * a^2 - 2 * a

-- Define the right side of the equation
def right_side (a : ℝ) : ℝ := 2 * a * (2 * a - 1)

-- Theorem statement
theorem factorization_4a_squared_minus_2a :
  is_factorization_left_to_right left_side right_side :=
sorry

end NUMINAMATH_CALUDE_factorization_4a_squared_minus_2a_l373_37336


namespace NUMINAMATH_CALUDE_only_isosceles_trapezoid_axially_not_centrally_symmetric_l373_37381

-- Define the set of geometric figures
inductive GeometricFigure
  | LineSegment
  | Square
  | Circle
  | IsoscelesTrapezoid
  | Parallelogram

-- Define axial symmetry
def is_axially_symmetric (figure : GeometricFigure) : Prop :=
  match figure with
  | GeometricFigure.LineSegment => true
  | GeometricFigure.Square => true
  | GeometricFigure.Circle => true
  | GeometricFigure.IsoscelesTrapezoid => true
  | GeometricFigure.Parallelogram => false

-- Define central symmetry
def is_centrally_symmetric (figure : GeometricFigure) : Prop :=
  match figure with
  | GeometricFigure.LineSegment => true
  | GeometricFigure.Square => true
  | GeometricFigure.Circle => true
  | GeometricFigure.IsoscelesTrapezoid => false
  | GeometricFigure.Parallelogram => true

-- Theorem stating that only the isosceles trapezoid satisfies the condition
theorem only_isosceles_trapezoid_axially_not_centrally_symmetric :
  ∀ (figure : GeometricFigure),
    (is_axially_symmetric figure ∧ ¬is_centrally_symmetric figure) ↔
    (figure = GeometricFigure.IsoscelesTrapezoid) :=
by
  sorry

end NUMINAMATH_CALUDE_only_isosceles_trapezoid_axially_not_centrally_symmetric_l373_37381


namespace NUMINAMATH_CALUDE_group_size_l373_37301

theorem group_size (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : 
  average_increase = 2 →
  old_weight = 65 →
  new_weight = 81 →
  (new_weight - old_weight) / average_increase = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_l373_37301


namespace NUMINAMATH_CALUDE_f_values_l373_37388

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x

-- Theorem stating that f(2) = 14 and f(-2) = 2
theorem f_values : f 2 = 14 ∧ f (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_values_l373_37388


namespace NUMINAMATH_CALUDE_angle_measure_problem_l373_37351

/-- Two angles are complementary if their measures sum to 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- Given two complementary angles A and B, where A is 5 times B, prove A is 75 degrees -/
theorem angle_measure_problem (A B : ℝ) 
  (h1 : complementary A B) 
  (h2 : A = 5 * B) : 
  A = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l373_37351


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l373_37374

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the number of intersection points between three circles
def intersectionPoints (c1 c2 c3 : Circle) : ℕ := sorry

theorem circle_intersection_theorem :
  -- There exist three circles that intersect at exactly one point
  (∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 1) ∧
  -- There exist three circles that intersect at exactly two points
  (∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 2) ∧
  -- There do not exist three circles that intersect at exactly three points
  (¬ ∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 3) ∧
  -- There do not exist three circles that intersect at exactly four points
  (¬ ∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l373_37374


namespace NUMINAMATH_CALUDE_notecard_problem_l373_37369

theorem notecard_problem (N E : ℕ) : 
  N - E = 80 →  -- Bill used all envelopes and had 80 notecards left
  3 * E = N →   -- John used all notecards, each letter used 3 notecards
  N = 120       -- The number of notecards in each set is 120
:= by sorry

end NUMINAMATH_CALUDE_notecard_problem_l373_37369


namespace NUMINAMATH_CALUDE_encyclopedia_total_pages_l373_37389

/-- Represents a chapter in the encyclopedia -/
structure Chapter where
  main_pages : ℕ
  sub_chapters : ℕ
  sub_chapter_pages : ℕ

/-- The encyclopedia with 12 chapters -/
def encyclopedia : Vector Chapter 12 :=
  Vector.ofFn fun i =>
    match i with
    | 0 => ⟨450, 3, 90⟩
    | 1 => ⟨650, 5, 68⟩
    | 2 => ⟨712, 4, 75⟩
    | 3 => ⟨820, 6, 120⟩
    | 4 => ⟨530, 2, 110⟩
    | 5 => ⟨900, 7, 95⟩
    | 6 => ⟨680, 4, 80⟩
    | 7 => ⟨555, 3, 180⟩
    | 8 => ⟨990, 5, 53⟩
    | 9 => ⟨825, 6, 150⟩
    | 10 => ⟨410, 2, 200⟩
    | 11 => ⟨1014, 7, 69⟩

/-- Total pages in a chapter -/
def total_pages_in_chapter (c : Chapter) : ℕ :=
  c.main_pages + c.sub_chapters * c.sub_chapter_pages

/-- Total pages in the encyclopedia -/
def total_pages : ℕ :=
  (encyclopedia.toList.map total_pages_in_chapter).sum

theorem encyclopedia_total_pages :
  total_pages = 13659 := by sorry

end NUMINAMATH_CALUDE_encyclopedia_total_pages_l373_37389


namespace NUMINAMATH_CALUDE_board_numbers_l373_37341

theorem board_numbers (a b : ℕ+) : 
  (a.val - b.val)^2 = a.val^2 - b.val^2 - 4038 →
  ((a.val = 2020 ∧ b.val = 1) ∨
   (a.val = 2020 ∧ b.val = 2019) ∨
   (a.val = 676 ∧ b.val = 3) ∨
   (a.val = 676 ∧ b.val = 673)) :=
by sorry

end NUMINAMATH_CALUDE_board_numbers_l373_37341


namespace NUMINAMATH_CALUDE_data_analysis_l373_37384

def data : List ℕ := [11, 10, 11, 13, 11, 13, 15]

def mode (l : List ℕ) : ℕ := sorry

def mean (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem data_analysis (d : List ℕ) (h : d = data) : 
  mode d = 11 ∧ 
  mean d = 12 ∧ 
  variance d = 18/7 ∧ 
  median d = 11 := by sorry

end NUMINAMATH_CALUDE_data_analysis_l373_37384


namespace NUMINAMATH_CALUDE_sqrt_x_plus_2_meaningful_l373_37383

theorem sqrt_x_plus_2_meaningful (x : ℝ) : Real.sqrt (x + 2) ≥ 0 ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_2_meaningful_l373_37383


namespace NUMINAMATH_CALUDE_function_inequality_l373_37333

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivative condition
variable (h : ∀ x, deriv f x > deriv g x)

-- Define the theorem
theorem function_inequality (a x b : ℝ) (h_order : a < x ∧ x < b) :
  f x + g a > g x + f a := by sorry

end NUMINAMATH_CALUDE_function_inequality_l373_37333


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l373_37391

theorem arithmetic_evaluation : 8 + 15 / 3 * 2 - 5 + 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l373_37391


namespace NUMINAMATH_CALUDE_lineup_ways_proof_l373_37377

/-- The number of ways to arrange 5 people in a line with restrictions -/
def lineupWays : ℕ := 72

/-- The number of people in the line -/
def totalPeople : ℕ := 5

/-- The number of positions where the youngest person can be placed -/
def youngestPositions : ℕ := 3

/-- The number of choices for the first position -/
def firstPositionChoices : ℕ := 4

/-- The number of ways to arrange the remaining people after placing the youngest -/
def remainingArrangements : ℕ := 6

theorem lineup_ways_proof :
  lineupWays = firstPositionChoices * youngestPositions * remainingArrangements :=
sorry

end NUMINAMATH_CALUDE_lineup_ways_proof_l373_37377


namespace NUMINAMATH_CALUDE_gp_special_term_l373_37398

def geometric_progression (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem gp_special_term (b : ℕ → ℝ) (α : ℝ) :
  geometric_progression b →
  (0 < α) ∧ (α < Real.pi / 2) →
  b 25 = 2 * Real.tan α →
  b 31 = 2 * Real.sin α →
  b 37 = Real.sin (2 * α) :=
by sorry

end NUMINAMATH_CALUDE_gp_special_term_l373_37398


namespace NUMINAMATH_CALUDE_blue_stamps_count_l373_37352

theorem blue_stamps_count (red_count : ℕ) (yellow_count : ℕ) (blue_count : ℕ) 
  (red_price : ℚ) (blue_price : ℚ) (yellow_price : ℚ) (total_earnings : ℚ) :
  red_count = 20 →
  yellow_count = 7 →
  red_price = 11/10 →
  blue_price = 4/5 →
  yellow_price = 2 →
  total_earnings = 100 →
  (red_count : ℚ) * red_price + (blue_count : ℚ) * blue_price + (yellow_count : ℚ) * yellow_price = total_earnings →
  blue_count = 80 := by
  sorry

end NUMINAMATH_CALUDE_blue_stamps_count_l373_37352


namespace NUMINAMATH_CALUDE_guess_number_in_three_questions_l373_37331

theorem guess_number_in_three_questions :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 8 →
  ∃ (q₁ q₂ q₃ : ℕ → Prop),
    ∀ m : ℕ, 1 ≤ m ∧ m ≤ 8 →
      (q₁ m = q₁ n ∧ q₂ m = q₂ n ∧ q₃ m = q₃ n) → m = n :=
by sorry

end NUMINAMATH_CALUDE_guess_number_in_three_questions_l373_37331


namespace NUMINAMATH_CALUDE_probability_isosceles_triangle_l373_37345

def roll_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

def is_isosceles (a b : ℕ) : Bool :=
  a + b > 5

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (roll_die.product roll_die).filter (fun (a, b) => is_isosceles a b)

theorem probability_isosceles_triangle :
  (favorable_outcomes.card : ℚ) / (roll_die.card * roll_die.card) = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_isosceles_triangle_l373_37345


namespace NUMINAMATH_CALUDE_similar_squares_side_length_l373_37339

theorem similar_squares_side_length (s1 s2 : ℝ) (h1 : s1 > 0) (h2 : s2 > 0) : 
  (s1 ^ 2 : ℝ) / (s2 ^ 2) = 9 → s2 = 5 → s1 = 15 := by sorry

end NUMINAMATH_CALUDE_similar_squares_side_length_l373_37339


namespace NUMINAMATH_CALUDE_john_bought_36_rolls_l373_37360

def price_per_dozen : ℕ := 5
def amount_spent : ℕ := 15
def rolls_per_dozen : ℕ := 12

theorem john_bought_36_rolls : 
  amount_spent / price_per_dozen * rolls_per_dozen = 36 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_36_rolls_l373_37360


namespace NUMINAMATH_CALUDE_ratio_NBQ_ABQ_l373_37382

-- Define the points
variable (A B C P Q N : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- BP and BQ divide ∠ABC into three equal parts
axiom divide_three_equal : angle A B P = angle P B Q ∧ angle P B Q = angle Q B C

-- BN bisects ∠QBP
axiom bisect_QBP : angle Q B N = angle N B P

-- Theorem to prove
theorem ratio_NBQ_ABQ : 
  (angle N B Q) / (angle A B Q) = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_ratio_NBQ_ABQ_l373_37382


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l373_37395

theorem sqrt_equation_solution : ∃ x : ℝ, x = 225 / 16 ∧ Real.sqrt x + Real.sqrt (x + 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l373_37395


namespace NUMINAMATH_CALUDE_chord_intersection_ratio_l373_37373

theorem chord_intersection_ratio (EQ FQ GQ HQ : ℝ) :
  EQ = 5 →
  GQ = 12 →
  HQ = 3 →
  EQ * FQ = GQ * HQ →
  FQ / HQ = 12 / 5 := by
sorry

end NUMINAMATH_CALUDE_chord_intersection_ratio_l373_37373


namespace NUMINAMATH_CALUDE_tree_planting_equation_l373_37302

/-- Represents the tree planting scenario -/
structure TreePlanting where
  total_trees : ℕ := 480
  days_saved : ℕ := 4
  new_rate : ℝ
  original_rate : ℝ

/-- The new rate is 1/3 more than the original rate -/
axiom rate_increase {tp : TreePlanting} : tp.new_rate = (4/3) * tp.original_rate

/-- The equation correctly represents the tree planting scenario -/
theorem tree_planting_equation (tp : TreePlanting) :
  (tp.total_trees / (tp.original_rate)) - (tp.total_trees / tp.new_rate) = tp.days_saved := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_equation_l373_37302


namespace NUMINAMATH_CALUDE_pat_worked_57_days_l373_37387

/-- Represents the number of days Pat worked at the summer camp -/
def days_worked : ℕ := sorry

/-- The total number of days Pat spent at the summer camp -/
def total_days : ℕ := 70

/-- Daily wage for working days -/
def daily_wage : ℕ := 100

/-- Daily food cost for non-working days -/
def daily_food_cost : ℕ := 20

/-- The net amount Pat earned after 70 days -/
def net_earnings : ℕ := 5440

theorem pat_worked_57_days :
  days_worked * daily_wage - (total_days - days_worked) * daily_food_cost = net_earnings ∧
  days_worked = 57 := by sorry

end NUMINAMATH_CALUDE_pat_worked_57_days_l373_37387


namespace NUMINAMATH_CALUDE_f_2017_value_l373_37327

theorem f_2017_value (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - x * (deriv f 0) - 1) :
  f 2017 = 2016 * 2018 := by
  sorry

end NUMINAMATH_CALUDE_f_2017_value_l373_37327


namespace NUMINAMATH_CALUDE_probability_point_closer_to_center_l373_37321

theorem probability_point_closer_to_center (R : ℝ) (r : ℝ) : R > 0 → r > 0 → R = 3 * r →
  (π * (2 * r)^2) / (π * R^2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_closer_to_center_l373_37321


namespace NUMINAMATH_CALUDE_age_difference_is_four_l373_37386

/-- The difference between the ages of Albert's parents -/
def age_difference (albert_age : ℕ) : ℕ :=
  let father_age := albert_age + 48
  let brother_age := albert_age - 2
  let mother_age := brother_age + 46
  father_age - mother_age

/-- Theorem stating that the difference between the ages of Albert's parents is 4 years -/
theorem age_difference_is_four (albert_age : ℕ) (h : albert_age ≥ 2) :
  age_difference albert_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_four_l373_37386


namespace NUMINAMATH_CALUDE_system_solution_l373_37385

-- Define the system of equations
def system_equations (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (n ≥ 3) ∧
  (∀ i ∈ Finset.range (n - 1), x i ^ 3 = x ((i + 1) % n) + x ((i + 2) % n) + 1) ∧
  (x (n - 1) ^ 3 = x 0 + x 1 + 1)

-- Define the solution set
def solution_set : Set ℝ :=
  {-1, (1 + Real.sqrt 5) / 2, (1 - Real.sqrt 5) / 2}

-- Theorem statement
theorem system_solution (n : ℕ) (x : ℕ → ℝ) :
  system_equations n x →
  (∀ i ∈ Finset.range n, x i ∈ solution_set) ∧
  (∃ t ∈ solution_set, ∀ i ∈ Finset.range n, x i = t) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l373_37385


namespace NUMINAMATH_CALUDE_return_trip_time_l373_37342

/-- Represents the flight scenario between two cities --/
structure FlightScenario where
  d : ℝ  -- distance between cities
  v : ℝ  -- speed of plane in still air
  u : ℝ  -- speed of wind
  outboundTime : ℝ  -- time from A to B against wind
  returnTimeDifference : ℝ  -- difference in return time compared to calm air

/-- Conditions for the flight scenario --/
def flightConditions (s : FlightScenario) : Prop :=
  s.v > 0 ∧ s.u > 0 ∧ s.d > 0 ∧
  s.outboundTime = 60 ∧
  s.returnTimeDifference = 10 ∧
  s.d = s.outboundTime * (s.v - s.u) ∧
  s.d / (s.v + s.u) = s.d / s.v - s.returnTimeDifference

/-- The theorem stating that the return trip takes 20 minutes --/
theorem return_trip_time (s : FlightScenario) 
  (h : flightConditions s) : s.d / (s.v + s.u) = 20 := by
  sorry


end NUMINAMATH_CALUDE_return_trip_time_l373_37342


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achievable_l373_37394

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) 
  (pos₁ : 0 < y₁) (pos₂ : 0 < y₂) (pos₃ : 0 < y₃)
  (sum_constraint : y₁ + 3 * y₂ + 5 * y₃ = 120) :
  720 / 7 ≤ y₁^2 + y₂^2 + y₃^2 :=
by sorry

theorem min_sum_squares_achievable :
  ∃ (y₁ y₂ y₃ : ℝ), 0 < y₁ ∧ 0 < y₂ ∧ 0 < y₃ ∧ 
  y₁ + 3 * y₂ + 5 * y₃ = 120 ∧
  y₁^2 + y₂^2 + y₃^2 = 720 / 7 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achievable_l373_37394


namespace NUMINAMATH_CALUDE_mild_curries_count_mild_curries_proof_l373_37314

/-- The number of peppers needed for different curry types -/
def peppers_per_curry : List Nat := [3, 2, 1]

/-- The number of curries of each type previously bought -/
def previous_curries : List Nat := [30, 30, 10]

/-- The number of spicy curries now bought -/
def current_spicy_curries : Nat := 15

/-- The reduction in total peppers bought -/
def pepper_reduction : Nat := 40

/-- Calculate the total number of peppers previously bought -/
def previous_total_peppers : Nat :=
  List.sum (List.zipWith (· * ·) peppers_per_curry previous_curries)

/-- Calculate the current total number of peppers bought -/
def current_total_peppers : Nat := previous_total_peppers - pepper_reduction

/-- Calculate the number of peppers used for current spicy curries -/
def current_spicy_peppers : Nat := peppers_per_curry[1] * current_spicy_curries

theorem mild_curries_count : Nat :=
  current_total_peppers - current_spicy_peppers

theorem mild_curries_proof : mild_curries_count = 90 := by
  sorry

end NUMINAMATH_CALUDE_mild_curries_count_mild_curries_proof_l373_37314


namespace NUMINAMATH_CALUDE_sugar_salt_diff_is_one_l373_37315

/-- A baking recipe with specified ingredient amounts -/
structure Recipe where
  flour : ℕ
  sugar : ℕ
  salt : ℕ

/-- The difference in cups between sugar and salt in a recipe -/
def sugar_salt_difference (r : Recipe) : ℤ :=
  r.sugar - r.salt

/-- Theorem: The difference between sugar and salt in the given recipe is 1 cup -/
theorem sugar_salt_diff_is_one (r : Recipe) (h : r.flour = 6 ∧ r.sugar = 8 ∧ r.salt = 7) : 
  sugar_salt_difference r = 1 := by
  sorry

#eval sugar_salt_difference {flour := 6, sugar := 8, salt := 7}

end NUMINAMATH_CALUDE_sugar_salt_diff_is_one_l373_37315


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l373_37367

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 2 * a 8 = a 4 * a 4) →     -- a_2, a_4, a_8 form a geometric sequence
  a 4 = 12 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l373_37367


namespace NUMINAMATH_CALUDE_tangent_line_and_a_range_l373_37303

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

theorem tangent_line_and_a_range (a : ℝ) :
  -- Condition: Tangent line at (1, f(1)) is parallel to 2x - y + 1 = 0
  (f_derivative a 1 = 2) →
  -- Condition: f(x) is decreasing on the interval [-2/3, -1/3]
  (∀ x ∈ Set.Icc (-2/3) (-1/3), f_derivative a x ≤ 0) →
  -- Conclusion 1: Equations of tangent lines passing through (0, 1)
  ((∃ x₀ y₀ : ℝ, f a x₀ = y₀ ∧ f_derivative a x₀ = (y₀ - 1) / x₀ ∧
    ((y₀ = 1 ∧ x₀ = 0) ∨ (y₀ = 11/8 ∧ x₀ = 1/2))) ∧
   (∀ x y : ℝ, (y = x + 1) ∨ (y = 3/4 * x + 1))) ∧
  -- Conclusion 2: Range of a
  (a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_a_range_l373_37303


namespace NUMINAMATH_CALUDE_andy_cookies_l373_37380

/-- Represents the number of cookies taken by each basketball team member -/
def basketballTeamCookies (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of cookies taken by all basketball team members -/
def totalTeamCookies (teamSize : ℕ) : ℕ :=
  (teamSize * (basketballTeamCookies 1 + basketballTeamCookies teamSize)) / 2

theorem andy_cookies (initialCookies brotherCookies teamSize : ℕ) 
  (h1 : initialCookies = 72)
  (h2 : brotherCookies = 5)
  (h3 : teamSize = 8)
  (h4 : totalTeamCookies teamSize + brotherCookies < initialCookies) :
  initialCookies - (totalTeamCookies teamSize + brotherCookies) = 3 := by
  sorry

end NUMINAMATH_CALUDE_andy_cookies_l373_37380


namespace NUMINAMATH_CALUDE_H_range_l373_37304

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem H_range : ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ -5 ≤ y ∧ y ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_H_range_l373_37304


namespace NUMINAMATH_CALUDE_quadratic_functions_problem_l373_37344

/-- A quadratic function -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex (f : QuadraticFunction) : ℝ := sorry

/-- The x-intercepts of a quadratic function -/
def x_intercepts (f : QuadraticFunction) : Set ℝ := sorry

theorem quadratic_functions_problem 
  (f g : QuadraticFunction)
  (h1 : ∀ x, g.f x = -f.f (100 - x))
  (h2 : vertex f ∈ x_intercepts g)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h3 : {x₁, x₂, x₃, x₄} ⊆ x_intercepts f ∪ x_intercepts g)
  (h4 : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
  (h5 : x₃ - x₂ = 150)
  : x₄ - x₁ = 450 + 300 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_functions_problem_l373_37344


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l373_37368

theorem ratio_x_to_y (x y : ℝ) (h : y = x * (1 - 0.6666666666666666)) :
  x / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l373_37368


namespace NUMINAMATH_CALUDE_product_mb_range_l373_37330

/-- Given a line y = mx + b with slope m = 3/4 and y-intercept b = -1/3,
    the product mb satisfies -1 < mb < 0. -/
theorem product_mb_range (m b : ℚ) : 
  m = 3/4 → b = -1/3 → -1 < m * b ∧ m * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mb_range_l373_37330


namespace NUMINAMATH_CALUDE_decreasing_function_a_range_l373_37346

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a
  else Real.log x / Real.log a

-- Theorem statement
theorem decreasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  (1 / 7 : ℝ) ≤ a ∧ a < (1 / 3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_a_range_l373_37346


namespace NUMINAMATH_CALUDE_translation_transforms_function_l373_37358

/-- The translation vector -/
def translation_vector : ℝ × ℝ := (2, -3)

/-- The original function -/
def original_function (x : ℝ) : ℝ := x^2 + 4*x + 7

/-- The translated function -/
def translated_function (x : ℝ) : ℝ := x^2

theorem translation_transforms_function :
  ∀ x y : ℝ,
  original_function (x - translation_vector.1) + translation_vector.2 = translated_function x :=
by sorry

end NUMINAMATH_CALUDE_translation_transforms_function_l373_37358


namespace NUMINAMATH_CALUDE_jackson_running_program_l373_37376

/-- Calculates the final running distance after a given number of days,
    given an initial distance and daily increase. -/
def finalRunningDistance (initialDistance : ℝ) (dailyIncrease : ℝ) (days : ℕ) : ℝ :=
  initialDistance + dailyIncrease * (days - 1)

/-- Theorem stating that given the initial conditions of Jackson's running program,
    the final running distance on the last day is 16.5 miles. -/
theorem jackson_running_program :
  let initialDistance : ℝ := 3
  let dailyIncrease : ℝ := 0.5
  let programDays : ℕ := 28
  finalRunningDistance initialDistance dailyIncrease programDays = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_jackson_running_program_l373_37376


namespace NUMINAMATH_CALUDE_small_planter_capacity_l373_37375

/-- Given the total number of seeds, the number and capacity of large planters,
    and the number of small planters, prove that each small planter can hold 4 seeds. -/
theorem small_planter_capacity
  (total_seeds : ℕ)
  (large_planters : ℕ)
  (large_planter_capacity : ℕ)
  (small_planters : ℕ)
  (h1 : total_seeds = 200)
  (h2 : large_planters = 4)
  (h3 : large_planter_capacity = 20)
  (h4 : small_planters = 30)
  : (total_seeds - large_planters * large_planter_capacity) / small_planters = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_planter_capacity_l373_37375


namespace NUMINAMATH_CALUDE_no_double_composition_inverse_l373_37326

/-- A quadratic function g(x) = ax^2 + bx + c -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The statement that g(g(x)) = x has four distinct real roots -/
def has_four_distinct_roots (a b c : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∀ x : ℝ, g a b c (g a b c x) = x ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

/-- The main theorem -/
theorem no_double_composition_inverse (a b c : ℝ) (h : has_four_distinct_roots a b c) :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = g a b c x := by
  sorry

end NUMINAMATH_CALUDE_no_double_composition_inverse_l373_37326


namespace NUMINAMATH_CALUDE_complex_multiplication_complex_division_l373_37325

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem 1
theorem complex_multiplication :
  (4 - i) * (6 + 2 * i^3) = 22 - 14 * i :=
by sorry

-- Theorem 2
theorem complex_division :
  (5 * (4 + i)^2) / (i * (2 + i)) = 1 - 38 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_complex_division_l373_37325


namespace NUMINAMATH_CALUDE_buy_one_get_one_free_promotion_l373_37332

/-- Calculates the total number of items received in a "buy one get one free" promotion --/
def itemsReceived (itemCost : ℕ) (totalPaid : ℕ) : ℕ :=
  2 * (totalPaid / itemCost)

/-- Theorem: Given a "buy one get one free" promotion where each item costs $3
    and a total payment of $15, the number of items received is 10 --/
theorem buy_one_get_one_free_promotion (itemCost : ℕ) (totalPaid : ℕ) 
    (h1 : itemCost = 3) (h2 : totalPaid = 15) : 
    itemsReceived itemCost totalPaid = 10 := by
  sorry

#eval itemsReceived 3 15  -- Should output 10

end NUMINAMATH_CALUDE_buy_one_get_one_free_promotion_l373_37332


namespace NUMINAMATH_CALUDE_magical_stack_size_157_l373_37322

/-- A stack of cards is magical if it satisfies certain conditions --/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (card_157_position : ℕ)
  (card_157_retains_position : card_157_position = 157)

/-- The number of cards in a magical stack where card 157 retains its position --/
def magical_stack_size (stack : MagicalStack) : ℕ := stack.total_cards

/-- Theorem: The size of a magical stack where card 157 retains its position is 470 --/
theorem magical_stack_size_157 (stack : MagicalStack) : 
  magical_stack_size stack = 470 := by sorry

end NUMINAMATH_CALUDE_magical_stack_size_157_l373_37322


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l373_37393

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 39 →
  football = 26 →
  tennis = 20 →
  neither = 10 →
  ∃ (both : ℕ), both = 17 ∧
    total = football + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l373_37393


namespace NUMINAMATH_CALUDE_mice_breeding_experiment_l373_37378

/-- Calculates the final number of mice after two generations -/
def final_mice_count (initial_mice : ℕ) (pups_per_mouse : ℕ) (pups_eaten : ℕ) : ℕ :=
  let first_gen_pups := initial_mice * pups_per_mouse
  let total_after_first_gen := initial_mice + first_gen_pups
  let surviving_pups_per_mouse := pups_per_mouse - pups_eaten
  let second_gen_pups := total_after_first_gen * surviving_pups_per_mouse
  total_after_first_gen + second_gen_pups

/-- Theorem stating that the final number of mice is 280 given the initial conditions -/
theorem mice_breeding_experiment :
  final_mice_count 8 6 2 = 280 := by
  sorry

end NUMINAMATH_CALUDE_mice_breeding_experiment_l373_37378


namespace NUMINAMATH_CALUDE_photo_area_l373_37354

theorem photo_area (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a + 4) * (b + 5) = 77) :
  a * b = 18 ∨ a * b = 14 := by
  sorry

end NUMINAMATH_CALUDE_photo_area_l373_37354


namespace NUMINAMATH_CALUDE_probability_through_C_l373_37348

theorem probability_through_C (total_paths : ℕ) (paths_A_to_C : ℕ) (paths_C_to_B : ℕ) :
  total_paths = Nat.choose 6 3 →
  paths_A_to_C = Nat.choose 3 2 →
  paths_C_to_B = Nat.choose 3 1 →
  (paths_A_to_C * paths_C_to_B : ℚ) / total_paths = 21 / 32 :=
by sorry

end NUMINAMATH_CALUDE_probability_through_C_l373_37348


namespace NUMINAMATH_CALUDE_fruit_picking_orders_l373_37399

/-- The number of fruits in the basket -/
def n : ℕ := 5

/-- The number of fruits to be picked -/
def k : ℕ := 2

/-- Calculates the number of permutations of n items taken k at a time -/
def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Theorem stating that picking 2 fruits out of 5 distinct fruits, where order matters, results in 20 different orders -/
theorem fruit_picking_orders : permutations n k = 20 := by sorry

end NUMINAMATH_CALUDE_fruit_picking_orders_l373_37399


namespace NUMINAMATH_CALUDE_fraction_product_l373_37372

theorem fraction_product : (2 : ℚ) / 5 * (3 : ℚ) / 5 = (6 : ℚ) / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l373_37372


namespace NUMINAMATH_CALUDE_boat_current_rate_l373_37392

/-- Proves that given a boat with a speed of 22 km/hr in still water,
    traveling 10.4 km downstream in 24 minutes, the rate of the current is 4 km/hr. -/
theorem boat_current_rate
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : boat_speed = 22)
  (h2 : downstream_distance = 10.4)
  (h3 : downstream_time = 24 / 60) :
  ∃ current_rate : ℝ,
    current_rate = 4 ∧
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_current_rate_l373_37392


namespace NUMINAMATH_CALUDE_theater_population_l373_37340

theorem theater_population :
  ∀ (total : ℕ),
  (19 : ℕ) + (total / 2) + (total / 4) + 6 = total →
  total = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_population_l373_37340


namespace NUMINAMATH_CALUDE_movie_tickets_correct_l373_37396

/-- The number of movie tickets sold for the given estimation --/
def movie_tickets : ℕ := 6

/-- The price of a pack of grain crackers --/
def cracker_price : ℚ := 2.25

/-- The price of a bottle of beverage --/
def beverage_price : ℚ := 1.5

/-- The price of a chocolate bar --/
def chocolate_price : ℚ := 1

/-- The average amount of estimated snack sales per movie ticket --/
def avg_sales_per_ticket : ℚ := 2.79

/-- Theorem stating that the number of movie tickets sold is correct --/
theorem movie_tickets_correct : 
  (3 * cracker_price + 4 * beverage_price + 4 * chocolate_price) / avg_sales_per_ticket = movie_tickets :=
by sorry

end NUMINAMATH_CALUDE_movie_tickets_correct_l373_37396


namespace NUMINAMATH_CALUDE_no_xy_term_implies_m_eq_6_l373_37312

/-- The polynomial that does not contain the xy term -/
def polynomial (x y m : ℝ) : ℝ := x^2 - m*x*y - y^2 + 6*x*y - 1

/-- The coefficient of xy in the polynomial -/
def xy_coefficient (m : ℝ) : ℝ := -m + 6

theorem no_xy_term_implies_m_eq_6 (m : ℝ) :
  (∀ x y : ℝ, polynomial x y m = x^2 - y^2 - 1) → m = 6 :=
by sorry

end NUMINAMATH_CALUDE_no_xy_term_implies_m_eq_6_l373_37312


namespace NUMINAMATH_CALUDE_trader_cloth_sale_l373_37311

/-- The number of meters of cloth sold by a trader -/
def meters_sold (total_price profit_per_meter cost_per_meter : ℚ) : ℚ :=
  total_price / (cost_per_meter + profit_per_meter)

/-- Theorem: The trader sold 85 meters of cloth -/
theorem trader_cloth_sale : meters_sold 8925 5 100 = 85 := by
  sorry

end NUMINAMATH_CALUDE_trader_cloth_sale_l373_37311


namespace NUMINAMATH_CALUDE_photo_survey_result_l373_37361

/-- Represents the number of students with each attitude towards photography -/
structure PhotoAttitudes where
  dislike : ℕ
  neutral : ℕ
  like : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (a : PhotoAttitudes) : Prop :=
  a.neutral = a.dislike + 12 ∧
  1 * (a.neutral + a.dislike + a.like) = 9 * a.dislike ∧
  3 * (a.neutral + a.dislike + a.like) = 9 * a.neutral ∧
  5 * (a.neutral + a.dislike + a.like) = 9 * a.like

/-- The theorem to be proved -/
theorem photo_survey_result :
  ∃ (a : PhotoAttitudes), satisfiesConditions a ∧ a.like = 30 := by
  sorry

end NUMINAMATH_CALUDE_photo_survey_result_l373_37361


namespace NUMINAMATH_CALUDE_magnitude_of_z_squared_l373_37363

theorem magnitude_of_z_squared (z : ℂ) : z = 5 + 2*I → Complex.abs (z^2) = 29 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_squared_l373_37363


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l373_37356

theorem arithmetic_sequence_sum (a₁ aₙ d : ℕ) (n : ℕ) (h1 : a₁ = 1) (h2 : aₙ = 31) (h3 : d = 3) (h4 : n = (aₙ - a₁) / d + 1) :
  (n : ℝ) / 2 * (a₁ + aₙ) = 176 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l373_37356


namespace NUMINAMATH_CALUDE_girls_to_boys_fraction_l373_37350

theorem girls_to_boys_fraction (total : ℕ) (girls : ℕ) (h1 : total = 35) (h2 : girls = 10) :
  (girls : ℚ) / ((total - girls) : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_fraction_l373_37350
