import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l3707_370770

theorem problem_statement : (2025^2 - 2025) / 2025 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3707_370770


namespace NUMINAMATH_CALUDE_corn_field_fraction_theorem_l3707_370716

/-- Represents a trapezoid field -/
structure TrapezoidField where
  short_side : ℝ
  long_side : ℝ
  angle : ℝ

/-- The fraction of a trapezoid field's area that is closer to its longest side -/
def fraction_closest_to_longest_side (field : TrapezoidField) : ℝ :=
  sorry

theorem corn_field_fraction_theorem (field : TrapezoidField) 
  (h1 : field.short_side = 120)
  (h2 : field.long_side = 240)
  (h3 : field.angle = 60) :
  fraction_closest_to_longest_side field = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_corn_field_fraction_theorem_l3707_370716


namespace NUMINAMATH_CALUDE_sum_and_double_l3707_370747

theorem sum_and_double : 2 * (2/20 + 3/30 + 4/40) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_double_l3707_370747


namespace NUMINAMATH_CALUDE_x_value_l3707_370785

theorem x_value : ∃ x : ℝ, x = 88 * (1 + 0.5) ∧ x = 132 := by sorry

end NUMINAMATH_CALUDE_x_value_l3707_370785


namespace NUMINAMATH_CALUDE_jake_weight_loss_l3707_370786

/-- Jake needs to lose weight to weigh twice as much as his sister. -/
theorem jake_weight_loss (total_weight sister_weight jake_weight : ℕ) 
  (h1 : total_weight = 153)
  (h2 : jake_weight = 113)
  (h3 : total_weight = sister_weight + jake_weight) :
  jake_weight - 2 * sister_weight = 33 := by
sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l3707_370786


namespace NUMINAMATH_CALUDE_journey_ratio_l3707_370767

/-- Proves the ratio of distance after storm to total journey distance -/
theorem journey_ratio (speed : ℝ) (time : ℝ) (storm_distance : ℝ) : 
  speed = 30 ∧ time = 20 ∧ storm_distance = 200 →
  (speed * time - storm_distance) / (2 * speed * time) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_journey_ratio_l3707_370767


namespace NUMINAMATH_CALUDE_inverse_negation_false_l3707_370755

theorem inverse_negation_false : 
  ¬(∀ x : ℝ, (x^2 = 1 ∧ x ≠ 1) → x^2 ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_inverse_negation_false_l3707_370755


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3707_370717

theorem quadratic_expression_value (x y z : ℝ) 
  (eq1 : 4*x + 2*y + z = 20)
  (eq2 : x + 4*y + 2*z = 26)
  (eq3 : 2*x + y + 4*z = 28) :
  20*x^2 + 24*x*y + 20*y^2 + 12*z^2 = 500 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3707_370717


namespace NUMINAMATH_CALUDE_tom_program_duration_l3707_370748

def combined_program_duration (bs_duration ph_d_duration : ℕ) : ℕ :=
  bs_duration + ph_d_duration

def accelerated_duration (total_duration : ℕ) (acceleration_factor : ℚ) : ℚ :=
  (total_duration : ℚ) * acceleration_factor

theorem tom_program_duration :
  let bs_duration : ℕ := 3
  let ph_d_duration : ℕ := 5
  let acceleration_factor : ℚ := 3 / 4
  let total_duration := combined_program_duration bs_duration ph_d_duration
  accelerated_duration total_duration acceleration_factor = 6 := by
  sorry

end NUMINAMATH_CALUDE_tom_program_duration_l3707_370748


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_100_l3707_370763

theorem largest_multiple_of_9_less_than_100 : 
  ∀ n : ℕ, n * 9 < 100 → n * 9 ≤ 99 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_100_l3707_370763


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3707_370715

theorem quadratic_equations_solutions : ∃ (s1 s2 : Set ℝ),
  (∀ x : ℝ, x ∈ s1 ↔ 3 * x^2 = 6 * x) ∧
  (∀ x : ℝ, x ∈ s2 ↔ x^2 - 6 * x + 5 = 0) ∧
  s1 = {0, 2} ∧
  s2 = {5, 1} := by
sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3707_370715


namespace NUMINAMATH_CALUDE_fractional_equation_simplification_l3707_370759

theorem fractional_equation_simplification (x : ℝ) :
  (x / (2 * x - 1) - 3 = 2 / (1 - 2 * x)) ↔ (x - 3 * (2 * x - 1) = -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_simplification_l3707_370759


namespace NUMINAMATH_CALUDE_complement_of_union_l3707_370738

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union (h : U = {1, 2, 3, 4} ∧ M = {1, 2} ∧ N = {2, 3}) :
  U \ (M ∪ N) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l3707_370738


namespace NUMINAMATH_CALUDE_average_of_nine_numbers_l3707_370713

theorem average_of_nine_numbers (numbers : Fin 9 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4) / 5 = 99)
  (h2 : (numbers 4 + numbers 5 + numbers 6 + numbers 7 + numbers 8) / 5 = 100)
  (h3 : numbers 4 = 59) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + 
   numbers 5 + numbers 6 + numbers 7 + numbers 8) / 9 = 104 := by
sorry

end NUMINAMATH_CALUDE_average_of_nine_numbers_l3707_370713


namespace NUMINAMATH_CALUDE_rice_bag_problem_l3707_370704

theorem rice_bag_problem (initial_stock : ℕ) : 
  initial_stock - 23 + 132 = 164 → initial_stock = 55 := by
  sorry

end NUMINAMATH_CALUDE_rice_bag_problem_l3707_370704


namespace NUMINAMATH_CALUDE_select_blocks_count_l3707_370724

/-- The number of ways to select 4 blocks from a 6x6 grid with no two in the same row or column -/
def select_blocks : ℕ :=
  (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

/-- Theorem stating that the number of ways to select 4 blocks from a 6x6 grid
    with no two in the same row or column is 5400 -/
theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end NUMINAMATH_CALUDE_select_blocks_count_l3707_370724


namespace NUMINAMATH_CALUDE_rent_utilities_percentage_l3707_370749

-- Define the previous monthly income
def previous_income : ℝ := 1000

-- Define the salary increase
def salary_increase : ℝ := 600

-- Define the percentage spent on rent and utilities after the increase
def new_percentage : ℝ := 0.25

-- Define the function to calculate the amount spent on rent and utilities
def rent_utilities (income : ℝ) (percentage : ℝ) : ℝ := income * percentage

-- Theorem statement
theorem rent_utilities_percentage :
  ∃ (old_percentage : ℝ),
    rent_utilities previous_income old_percentage = 
    rent_utilities (previous_income + salary_increase) new_percentage ∧
    old_percentage = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_rent_utilities_percentage_l3707_370749


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3707_370722

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 + (m+2)*x - 2 = 0 ∧ x = 1) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3707_370722


namespace NUMINAMATH_CALUDE_not_passes_third_quadrant_l3707_370732

/-- A linear function f(x) = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.first  => x > 0 ∧ y > 0
  | Quadrant.second => x < 0 ∧ y > 0
  | Quadrant.third  => x < 0 ∧ y < 0
  | Quadrant.fourth => x > 0 ∧ y < 0

/-- A linear function passes through a quadrant if there exists a point (x, y) in that quadrant satisfying the function equation -/
def passesThroughQuadrant (f : LinearFunction) (q : Quadrant) : Prop :=
  ∃ x y : ℝ, y = f.m * x + f.b ∧ inQuadrant x y q

/-- The main theorem: the graph of y = -3x + 2 does not pass through the third quadrant -/
theorem not_passes_third_quadrant :
  ¬ passesThroughQuadrant { m := -3, b := 2 } Quadrant.third := by
  sorry

end NUMINAMATH_CALUDE_not_passes_third_quadrant_l3707_370732


namespace NUMINAMATH_CALUDE_area_of_triangle_APQ_l3707_370720

/-- Two perpendicular lines intersecting at A(9,12) with y-intercepts P and Q -/
structure PerpendicularLines where
  A : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  perpendicular : Bool
  intersect_at_A : Bool
  y_intercept_diff : ℝ

/-- The specific configuration of perpendicular lines for our problem -/
def problem_lines : PerpendicularLines where
  A := (9, 12)
  P := (0, 0)
  Q := (0, 6)
  perpendicular := true
  intersect_at_A := true
  y_intercept_diff := 6

/-- The area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of triangle APQ is 27 -/
theorem area_of_triangle_APQ : 
  triangle_area problem_lines.A problem_lines.P problem_lines.Q = 27 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_APQ_l3707_370720


namespace NUMINAMATH_CALUDE_chloe_first_round_score_l3707_370718

/-- Chloe's trivia game score calculation -/
theorem chloe_first_round_score (first_round : ℤ) 
  (h1 : first_round + 50 - 4 = 86) : first_round = 40 := by
  sorry

end NUMINAMATH_CALUDE_chloe_first_round_score_l3707_370718


namespace NUMINAMATH_CALUDE_bisection_diagram_type_l3707_370789

/-- The function we're finding the root for -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- Represents the types of diagrams -/
inductive DiagramType
| ProcessFlowchart
| KnowledgeStructureDiagram
| ProgramFlowchart
| OrganizationalStructureDiagram

/-- Properties of the bisection method -/
structure BisectionMethod where
  continuous : ∀ a b, a < b → ContinuousOn f (Set.Icc a b)
  oppositeSign : ∃ a b, a < b ∧ f a * f b < 0
  iterative : ∀ a b, a < b → ∃ c, a < c ∧ c < b ∧ f c = (f a + f b) / 2

/-- The theorem stating that the bisection method for x^2 - 2 = 0 is represented by a Program Flowchart -/
theorem bisection_diagram_type (bm : BisectionMethod) : 
  ∃ d : DiagramType, d = DiagramType.ProgramFlowchart :=
sorry

end NUMINAMATH_CALUDE_bisection_diagram_type_l3707_370789


namespace NUMINAMATH_CALUDE_two_thirds_of_45_minus_7_l3707_370797

theorem two_thirds_of_45_minus_7 : (2 / 3 : ℚ) * 45 - 7 = 23 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_45_minus_7_l3707_370797


namespace NUMINAMATH_CALUDE_johns_age_l3707_370710

/-- Given the ages of John, his dad, and his sister, prove that John is 25 years old. -/
theorem johns_age (john dad sister : ℕ) 
  (h1 : john + 30 = dad)
  (h2 : john + dad = 80)
  (h3 : sister = john - 5) :
  john = 25 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l3707_370710


namespace NUMINAMATH_CALUDE_ship_passengers_with_round_trip_tickets_l3707_370739

theorem ship_passengers_with_round_trip_tickets 
  (total_passengers : ℝ) 
  (h1 : total_passengers > 0) 
  (round_trip_with_car : ℝ) 
  (h2 : round_trip_with_car = 0.15 * total_passengers) 
  (h3 : round_trip_with_car > 0) 
  (round_trip_without_car_ratio : ℝ) 
  (h4 : round_trip_without_car_ratio = 0.6) :
  (round_trip_with_car / (1 - round_trip_without_car_ratio)) / total_passengers = 0.375 := by
sorry

end NUMINAMATH_CALUDE_ship_passengers_with_round_trip_tickets_l3707_370739


namespace NUMINAMATH_CALUDE_sector_central_angle_l3707_370734

/-- Given a sector with circumference 10 and area 4, prove that its central angle is π/2 radians -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 10) (h2 : (1/2) * l * r = 4) :
  l / r = 1/2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3707_370734


namespace NUMINAMATH_CALUDE_system_solution_l3707_370737

theorem system_solution (x y : ℝ) : 
  (2 * x + 5 * y = 26 ∧ 4 * x - 2 * y = 4) ↔ (x = 3 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3707_370737


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_values_l3707_370707

-- Define the two lines
def line1 (t k : ℝ) : ℝ × ℝ × ℝ := (1 + t, 2 + 2*t, 3 - k*t)
def line2 (u k : ℝ) : ℝ × ℝ × ℝ := (2 + u, 5 + k*u, 6 + u)

-- Define the condition for the lines to be coplanar
def are_coplanar (k : ℝ) : Prop :=
  ∃ t u, line1 t k = line2 u k

-- State the theorem
theorem lines_coplanar_iff_k_values :
  ∀ k : ℝ, are_coplanar k ↔ (k = -2 + Real.sqrt 6 ∨ k = -2 - Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_values_l3707_370707


namespace NUMINAMATH_CALUDE_plot_length_proof_l3707_370779

/-- Given a rectangular plot with width 50 meters, prove that if 56 poles
    are needed when placed 5 meters apart along the perimeter,
    then the length of the plot is 80 meters. -/
theorem plot_length_proof (width : ℝ) (num_poles : ℕ) (pole_distance : ℝ) (length : ℝ) :
  width = 50 →
  num_poles = 56 →
  pole_distance = 5 →
  2 * ((length / pole_distance) + 1) + 2 * ((width / pole_distance) + 1) = num_poles →
  length = 80 := by
  sorry


end NUMINAMATH_CALUDE_plot_length_proof_l3707_370779


namespace NUMINAMATH_CALUDE_coin_flip_sequences_l3707_370752

theorem coin_flip_sequences (n k : ℕ) (hn : n = 10) (hk : k = 6) :
  (Nat.choose n k) = 210 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_sequences_l3707_370752


namespace NUMINAMATH_CALUDE_belt_length_sufficient_l3707_370791

/-- Given three pulleys with parallel axes and identical radii, prove that 
    a 54 cm cord is sufficient for the belt connecting them. -/
theorem belt_length_sufficient 
  (r : ℝ) 
  (O₁O₂ O₁O₃ O₃_to_plane : ℝ) 
  (h_r : r = 2)
  (h_O₁O₂ : O₁O₂ = 12)
  (h_O₁O₃ : O₁O₃ = 10)
  (h_O₃_to_plane : O₃_to_plane = 8) :
  ∃ (belt_length : ℝ), 
    belt_length < 54 ∧ 
    belt_length = 
      O₁O₂ + O₁O₃ + Real.sqrt (O₁O₂^2 + O₁O₃^2 - 2 * O₁O₂ * O₁O₃ * (O₃_to_plane / O₁O₃)) + 
      2 * π * r :=
by sorry

end NUMINAMATH_CALUDE_belt_length_sufficient_l3707_370791


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3707_370782

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (h1 : a > b) (h2 : b > 0) : 
  a + b > 2 * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3707_370782


namespace NUMINAMATH_CALUDE_mary_turnips_count_l3707_370790

/-- The number of turnips grown by Sally -/
def sally_turnips : ℕ := 113

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := 242

/-- The number of turnips grown by Mary -/
def mary_turnips : ℕ := total_turnips - sally_turnips

theorem mary_turnips_count : mary_turnips = 129 := by
  sorry

end NUMINAMATH_CALUDE_mary_turnips_count_l3707_370790


namespace NUMINAMATH_CALUDE_special_set_properties_l3707_370753

/-- A set M with specific closure properties -/
structure SpecialSet (M : Set ℝ) : Prop where
  zero_in : (0 : ℝ) ∈ M
  one_in : (1 : ℝ) ∈ M
  closed_sub : ∀ x y, x ∈ M → y ∈ M → (x - y) ∈ M
  closed_inv : ∀ x, x ∈ M → x ≠ 0 → (1 / x) ∈ M

/-- Properties of the special set M -/
theorem special_set_properties (M : Set ℝ) (h : SpecialSet M) :
  (1 / 3 ∈ M) ∧
  (-1 ∈ M) ∧
  (∀ x y, x ∈ M → y ∈ M → (x + y) ∈ M) ∧
  (∀ x, x ∈ M → x^2 ∈ M) := by
  sorry

end NUMINAMATH_CALUDE_special_set_properties_l3707_370753


namespace NUMINAMATH_CALUDE_sum_of_roots_l3707_370751

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x

-- State the theorem
theorem sum_of_roots (h k : ℝ) (h_root : p h = 1) (k_root : p k = 5) : h + k = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3707_370751


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l3707_370702

-- Define the line ax - by - 2 = 0
def line (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y - 2 = 0

-- Define the curve y = x^3
def curve (x y : ℝ) : Prop := y = x^3

-- Define the point P(1, 1)
def point_P : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line to the curve at P
def tangent_slope_curve : ℝ := 3

-- Define the condition that the tangent lines are mutually perpendicular
def perpendicular_tangents (a b : ℝ) : Prop :=
  (a / b) * tangent_slope_curve = -1

theorem perpendicular_tangents_ratio (a b : ℝ) :
  line a b point_P.1 point_P.2 →
  curve point_P.1 point_P.2 →
  perpendicular_tangents a b →
  a / b = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l3707_370702


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l3707_370778

-- Define the function f
def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + x + c

-- Define the property of being an odd function on an interval
def is_odd_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f (-x) = -f x) ∧ a + b = 0

-- Theorem statement
theorem odd_function_sum_zero (a b c : ℝ) :
  is_odd_on (f a c) a b → a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l3707_370778


namespace NUMINAMATH_CALUDE_yarn_cost_calculation_l3707_370768

/-- Proves that the cost of each ball of yarn is $6 given the conditions of Chantal's sweater business -/
theorem yarn_cost_calculation (num_sweaters : ℕ) (yarn_per_sweater : ℕ) (price_per_sweater : ℚ) (total_profit : ℚ) : 
  num_sweaters = 28 →
  yarn_per_sweater = 4 →
  price_per_sweater = 35 →
  total_profit = 308 →
  (num_sweaters * price_per_sweater - total_profit) / (num_sweaters * yarn_per_sweater) = 6 := by
sorry

end NUMINAMATH_CALUDE_yarn_cost_calculation_l3707_370768


namespace NUMINAMATH_CALUDE_bianca_birthday_money_l3707_370775

theorem bianca_birthday_money (total_amount : ℕ) (num_friends : ℕ) (amount_per_friend : ℕ) 
  (h1 : total_amount = 30) 
  (h2 : num_friends = 5) 
  (h3 : total_amount = num_friends * amount_per_friend) : 
  amount_per_friend = 6 := by
  sorry

end NUMINAMATH_CALUDE_bianca_birthday_money_l3707_370775


namespace NUMINAMATH_CALUDE_lineup_probability_probability_no_more_than_five_girls_between_boys_l3707_370731

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem lineup_probability :
  (C 14 9 + 6 * C 13 8) / C 20 9 =
  (↑(C 14 9 + 6 * C 13 8) : ℚ) / (C 20 9 : ℚ) :=
by sorry

theorem probability_no_more_than_five_girls_between_boys :
  (↑(C 14 9 + 6 * C 13 8) : ℚ) / (C 20 9 : ℚ) =
  9724 / 167960 :=
by sorry

end NUMINAMATH_CALUDE_lineup_probability_probability_no_more_than_five_girls_between_boys_l3707_370731


namespace NUMINAMATH_CALUDE_student_number_problem_l3707_370799

theorem student_number_problem :
  ∃ x : ℝ, 2 * x - 138 = 104 ∧ x = 121 := by sorry

end NUMINAMATH_CALUDE_student_number_problem_l3707_370799


namespace NUMINAMATH_CALUDE_twice_total_credits_l3707_370703

/-- Given the high school credits of three students (Aria, Emily, and Spencer),
    where Emily has 20 credits, Aria has twice as many credits as Emily,
    and Emily has twice as many credits as Spencer,
    prove that twice the total number of credits for all three is 140. -/
theorem twice_total_credits (emily_credits : ℕ) 
  (h1 : emily_credits = 20)
  (h2 : ∃ aria_credits : ℕ, aria_credits = 2 * emily_credits)
  (h3 : ∃ spencer_credits : ℕ, emily_credits = 2 * spencer_credits) :
  2 * (emily_credits + 2 * emily_credits + emily_credits / 2) = 140 :=
by sorry

end NUMINAMATH_CALUDE_twice_total_credits_l3707_370703


namespace NUMINAMATH_CALUDE_quadrilateral_equality_l3707_370793

/-- Given a quadrilateral ABCD where AD is parallel to BC, 
    prove that AC^2 + BD^2 = AB^2 + CD^2 + 2AD · BC. -/
theorem quadrilateral_equality (A B C D : ℝ × ℝ) 
    (h_parallel : (D.2 - A.2) / (D.1 - A.1) = (C.2 - B.2) / (C.1 - B.1)) : 
    (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - D.1)^2 + (B.2 - D.2)^2 = 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 + 
    2 * ((D.1 - A.1) * (C.1 - B.1) + (D.2 - A.2) * (C.2 - B.2)) := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_equality_l3707_370793


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l3707_370771

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement is valid. -/
def is_valid_seating (table : CircularTable) : Prop :=
  table.seated_people > 0 ∧
  table.seated_people ≤ table.total_chairs ∧
  ∀ (new_seat : ℕ), new_seat < table.total_chairs →
    ∃ (occupied_seat : ℕ), occupied_seat < table.total_chairs ∧
      (new_seat = (occupied_seat + 1) % table.total_chairs ∨
       new_seat = (occupied_seat - 1 + table.total_chairs) % table.total_chairs)

/-- The main theorem stating the smallest valid number of seated people. -/
theorem smallest_valid_seating :
  ∀ (table : CircularTable),
    table.total_chairs = 80 →
    (is_valid_seating table ↔ table.seated_people ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l3707_370771


namespace NUMINAMATH_CALUDE_third_team_pieces_l3707_370708

theorem third_team_pieces (total : ℕ) (first_team : ℕ) (second_team : ℕ) 
  (h1 : total = 500) 
  (h2 : first_team = 189) 
  (h3 : second_team = 131) : 
  total - (first_team + second_team) = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_third_team_pieces_l3707_370708


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3707_370772

theorem inequality_solution_set : 
  ∀ x : ℝ, -x^2 + 3*x + 4 > 0 ↔ -1 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3707_370772


namespace NUMINAMATH_CALUDE_phi_range_for_monotonic_interval_l3707_370706

/-- Given a function f(x) = -2 sin(2x + φ) where |φ| < π, 
    if (π/5, 5π/8) is a monotonically increasing interval of f(x),
    then π/10 ≤ φ ≤ π/4 -/
theorem phi_range_for_monotonic_interval (φ : Real) :
  (|φ| < π) →
  (∀ x₁ x₂, π/5 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5*π/8 → 
    (-2 * Real.sin (2*x₁ + φ)) < (-2 * Real.sin (2*x₂ + φ))) →
  π/10 ≤ φ ∧ φ ≤ π/4 := by
  sorry

end NUMINAMATH_CALUDE_phi_range_for_monotonic_interval_l3707_370706


namespace NUMINAMATH_CALUDE_min_pouches_is_sixty_l3707_370729

/-- Represents the number of gold coins Flint has. -/
def total_coins : ℕ := 60

/-- Represents the possible number of sailors among whom the coins might be distributed. -/
def possible_sailors : List ℕ := [2, 3, 4, 5]

/-- Defines a valid distribution as one where each sailor receives an equal number of coins. -/
def is_valid_distribution (num_pouches : ℕ) : Prop :=
  ∀ n ∈ possible_sailors, (total_coins / num_pouches) * n = total_coins

/-- States that the number of pouches is minimal if no smaller number satisfies the distribution criteria. -/
def is_minimal (num_pouches : ℕ) : Prop :=
  is_valid_distribution num_pouches ∧
  ∀ k < num_pouches, ¬is_valid_distribution k

/-- The main theorem stating that 60 is the minimum number of pouches required for valid distribution. -/
theorem min_pouches_is_sixty :
  is_minimal total_coins :=
sorry

end NUMINAMATH_CALUDE_min_pouches_is_sixty_l3707_370729


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3707_370700

/-- Given a line described by the equation y - 3 = -3(x + 2), 
    the sum of its x-intercept and y-intercept is -4 -/
theorem line_intercepts_sum : 
  ∀ (x y : ℝ), y - 3 = -3*(x + 2) → 
  ∃ (x_int y_int : ℝ), 
    (0 - 3 = -3*(x_int + 2)) ∧ 
    (y_int - 3 = -3*(0 + 2)) ∧ 
    (x_int + y_int = -4) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3707_370700


namespace NUMINAMATH_CALUDE_friday_texts_l3707_370730

/-- Represents the number of texts sent to each friend on a given day -/
structure DailyTexts where
  allison : ℕ
  brittney : ℕ
  carol : ℕ
  dylan : ℕ

/-- Calculates the total number of texts sent in a day -/
def totalTexts (d : DailyTexts) : ℕ := d.allison + d.brittney + d.carol + d.dylan

/-- Sydney's texting schedule from Monday to Thursday -/
def textSchedule : List DailyTexts := [
  ⟨5, 5, 5, 5⟩,        -- Monday
  ⟨15, 10, 12, 8⟩,     -- Tuesday
  ⟨20, 18, 7, 14⟩,     -- Wednesday
  ⟨0, 25, 10, 5⟩       -- Thursday
]

/-- Cost of a single text in cents -/
def textCost : ℕ := 10

/-- Weekly budget in cents -/
def weeklyBudget : ℕ := 2000

/-- Theorem: Sydney can send 36 texts on Friday given her schedule and budget -/
theorem friday_texts : 
  (weeklyBudget - (textSchedule.map totalTexts).sum * textCost) / textCost = 36 := by
  sorry

end NUMINAMATH_CALUDE_friday_texts_l3707_370730


namespace NUMINAMATH_CALUDE_three_statements_incorrect_l3707_370757

-- Define the four statements
def statement1 : Prop := ∀ (a : ℕ → ℝ) (S : ℕ → ℝ), 
  (∀ n, a (n+1) - a n = a (n+2) - a (n+1)) → 
  (a 6 + a 7 > 0 ↔ S 9 ≥ S 3)

def statement2 : Prop := 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x < 1)

def statement3 : Prop := 
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 → (x = 1 ∨ x = 3)) ↔
  (∀ x : ℝ, (x ≠ 1 ∨ x ≠ 3) → x^2 - 4*x + 3 ≠ 0)

def statement4 : Prop := 
  ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

-- Theorem stating that exactly 3 statements are incorrect
theorem three_statements_incorrect : 
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ statement4) :=
sorry

end NUMINAMATH_CALUDE_three_statements_incorrect_l3707_370757


namespace NUMINAMATH_CALUDE_total_paper_weight_is_2074_l3707_370792

/-- Calculates the total weight of paper Barbara removed from the chest of drawers. -/
def total_paper_weight : ℕ :=
  let bundle_size : ℕ := 2
  let bunch_size : ℕ := 4
  let heap_size : ℕ := 20
  let pile_size : ℕ := 10
  let stack_size : ℕ := 5

  let colored_bundles : ℕ := 3
  let white_bunches : ℕ := 2
  let scrap_heaps : ℕ := 5
  let glossy_piles : ℕ := 4
  let cardstock_stacks : ℕ := 3

  let colored_weight : ℕ := 8
  let white_weight : ℕ := 12
  let scrap_weight : ℕ := 10
  let glossy_weight : ℕ := 15
  let cardstock_weight : ℕ := 22

  let colored_total := colored_bundles * bundle_size * colored_weight
  let white_total := white_bunches * bunch_size * white_weight
  let scrap_total := scrap_heaps * heap_size * scrap_weight
  let glossy_total := glossy_piles * pile_size * glossy_weight
  let cardstock_total := cardstock_stacks * stack_size * cardstock_weight

  colored_total + white_total + scrap_total + glossy_total + cardstock_total

theorem total_paper_weight_is_2074 : total_paper_weight = 2074 := by
  sorry

end NUMINAMATH_CALUDE_total_paper_weight_is_2074_l3707_370792


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3707_370762

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) / n = 144 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3707_370762


namespace NUMINAMATH_CALUDE_sin_30_degrees_l3707_370783

theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l3707_370783


namespace NUMINAMATH_CALUDE_vacation_cost_per_person_l3707_370774

theorem vacation_cost_per_person 
  (num_people : ℕ) 
  (airbnb_cost car_cost : ℚ) 
  (h1 : num_people = 8)
  (h2 : airbnb_cost = 3200)
  (h3 : car_cost = 800) :
  (airbnb_cost + car_cost) / num_people = 500 :=
by sorry

end NUMINAMATH_CALUDE_vacation_cost_per_person_l3707_370774


namespace NUMINAMATH_CALUDE_farm_animals_count_l3707_370794

/-- Represents the farm with goats and sheep -/
structure Farm where
  goats : ℕ
  sheep : ℕ

/-- Calculates the total number of animals on the farm -/
def Farm.total (f : Farm) : ℕ := f.goats + f.sheep

/-- Theorem: Given the conditions, the total number of animals on the farm is 1524 -/
theorem farm_animals_count (f : Farm) 
  (ratio : f.goats * 7 = f.sheep * 5)
  (sale_amount : (f.goats / 2) * 40 + (f.sheep * 2 / 3) * 30 = 7200) : 
  f.total = 1524 := by
  sorry


end NUMINAMATH_CALUDE_farm_animals_count_l3707_370794


namespace NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l3707_370745

theorem belt_and_road_population_scientific_notation :
  (4500000000 : ℝ) = 4.5 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l3707_370745


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_17_l3707_370773

theorem three_digit_divisible_by_17 : 
  (Finset.filter (fun k : ℕ => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_17_l3707_370773


namespace NUMINAMATH_CALUDE_angle_complement_theorem_l3707_370705

theorem angle_complement_theorem (x : ℝ) : 
  (90 - x) = (3 * x + 10) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_theorem_l3707_370705


namespace NUMINAMATH_CALUDE_plan_D_most_reasonable_l3707_370781

/-- Represents a survey plan for testing vision of junior high school students -/
inductive SurveyPlan
| A  : SurveyPlan  -- Test students in a certain middle school
| B  : SurveyPlan  -- Test all students in a certain district
| C  : SurveyPlan  -- Test all students in the entire city
| D  : SurveyPlan  -- Select 5 schools from each district and test their students

/-- Represents a city with districts and schools -/
structure City where
  numDistricts : Nat
  numSchoolsPerDistrict : Nat

/-- Determines if a survey plan is reasonable based on representativeness and practicality -/
def isReasonable (plan : SurveyPlan) (city : City) : Prop :=
  match plan with
  | SurveyPlan.D => city.numDistricts = 9 ∧ city.numSchoolsPerDistrict ≥ 5
  | _ => False

/-- Theorem stating that plan D is the most reasonable for a city with 9 districts -/
theorem plan_D_most_reasonable (city : City) :
  city.numDistricts = 9 → city.numSchoolsPerDistrict ≥ 5 → 
  ∀ (plan : SurveyPlan), isReasonable plan city → plan = SurveyPlan.D :=
by sorry

end NUMINAMATH_CALUDE_plan_D_most_reasonable_l3707_370781


namespace NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l3707_370788

/-- Given the conditions of the police force duty, prove the percentage of female officers on duty -/
theorem percentage_female_officers_on_duty
  (total_on_duty : ℕ)
  (total_female_officers : ℕ)
  (half_on_duty_female : total_on_duty / 2 = total_on_duty - total_on_duty / 2)
  (h_total_on_duty : total_on_duty = 180)
  (h_total_female : total_female_officers = 500) :
  (((total_on_duty / 2 : ℚ) / total_female_officers) * 100 : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l3707_370788


namespace NUMINAMATH_CALUDE_shaded_triangle_probability_l3707_370733

/-- The total number of triangles in the diagram -/
def total_triangles : ℕ := 10

/-- The number of shaded or partially shaded triangles -/
def shaded_triangles : ℕ := 3

/-- Each triangle has an equal probability of being selected -/
axiom equal_probability : True

/-- The probability of selecting a shaded or partially shaded triangle -/
def shaded_probability : ℚ := shaded_triangles / total_triangles

theorem shaded_triangle_probability : 
  shaded_probability = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_shaded_triangle_probability_l3707_370733


namespace NUMINAMATH_CALUDE_unique_solution_sum_l3707_370719

theorem unique_solution_sum (x : ℝ) (a b c : ℕ+) : 
  x = Real.sqrt ((Real.sqrt 65) / 2 + 5 / 2) →
  x^100 = 3*x^98 + 18*x^96 + 13*x^94 - x^50 + a*x^46 + b*x^44 + c*x^40 →
  ∃! (a b c : ℕ+), x^100 = 3*x^98 + 18*x^96 + 13*x^94 - x^50 + a*x^46 + b*x^44 + c*x^40 →
  a + b + c = 105 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_sum_l3707_370719


namespace NUMINAMATH_CALUDE_fenced_area_blocks_l3707_370795

def total_blocks : ℕ := 344
def building_blocks : ℕ := 80
def farmhouse_blocks : ℕ := 123
def remaining_blocks : ℕ := 84

theorem fenced_area_blocks :
  total_blocks - building_blocks - farmhouse_blocks - remaining_blocks = 57 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_blocks_l3707_370795


namespace NUMINAMATH_CALUDE_total_crayons_l3707_370750

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) : 
  crayons_per_child = 5 → num_children = 10 → crayons_per_child * num_children = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l3707_370750


namespace NUMINAMATH_CALUDE_profit_ratio_problem_l3707_370798

/-- The profit ratio problem -/
theorem profit_ratio_problem (profit_3_shirts profit_7_shirts_3_sandals : ℚ) 
  (h1 : profit_3_shirts = 21)
  (h2 : profit_7_shirts_3_sandals = 175) :
  (2 * profit_3_shirts) = ((profit_7_shirts_3_sandals - (7 / 3) * profit_3_shirts) / 3 * 2) :=
by sorry

end NUMINAMATH_CALUDE_profit_ratio_problem_l3707_370798


namespace NUMINAMATH_CALUDE_shot_cost_calculation_l3707_370746

def total_shot_cost (num_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) : ℕ :=
  num_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot

theorem shot_cost_calculation :
  total_shot_cost 3 4 2 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_shot_cost_calculation_l3707_370746


namespace NUMINAMATH_CALUDE_geoffrey_game_cost_l3707_370701

theorem geoffrey_game_cost (initial_money : ℕ) : 
  initial_money + 20 + 25 + 30 = 125 → 
  ∃ (game_cost : ℕ), 
    game_cost * 3 = 125 - 20 ∧ 
    game_cost = 35 := by
  sorry

end NUMINAMATH_CALUDE_geoffrey_game_cost_l3707_370701


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l3707_370742

theorem max_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 36) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ Real.sqrt 261 := by
  sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l3707_370742


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3707_370769

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 →
  side_length = 7 →
  exterior_angle = 90 →
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 28 := by
  sorry

#check regular_polygon_perimeter

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3707_370769


namespace NUMINAMATH_CALUDE_equation_solution_l3707_370764

theorem equation_solution : ∃! x : ℝ, (64 : ℝ)^(x - 1) / (4 : ℝ)^(x - 1) = (256 : ℝ)^(2*x) ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3707_370764


namespace NUMINAMATH_CALUDE_k_h_negative_three_equals_fifteen_l3707_370754

-- Define the function h
def h (x : ℝ) : ℝ := 5 * x^2 - 12

-- Define a variable k as a function from ℝ to ℝ
variable (k : ℝ → ℝ)

-- State the theorem
theorem k_h_negative_three_equals_fifteen
  (h_def : ∀ x, h x = 5 * x^2 - 12)
  (k_h_three : k (h 3) = 15) :
  k (h (-3)) = 15 := by
sorry

end NUMINAMATH_CALUDE_k_h_negative_three_equals_fifteen_l3707_370754


namespace NUMINAMATH_CALUDE_brick_height_calculation_l3707_370711

/-- Calculates the height of a brick given the wall dimensions, mortar percentage, brick dimensions, and number of bricks --/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ)
  (mortar_percentage : ℝ) (brick_length brick_width : ℝ) (num_bricks : ℕ) :
  wall_length = 10 →
  wall_width = 4 →
  wall_height = 5 →
  mortar_percentage = 0.1 →
  brick_length = 0.25 →
  brick_width = 0.15 →
  num_bricks = 6000 →
  ∃ (brick_height : ℝ),
    brick_height = 0.8 ∧
    (1 - mortar_percentage) * wall_length * wall_width * wall_height =
    (brick_length * brick_width * brick_height) * num_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_height_calculation_l3707_370711


namespace NUMINAMATH_CALUDE_largest_special_number_l3707_370776

/-- A function that returns true if all digits in a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- A function that returns true if a natural number is divisible by all of its digits -/
def divisible_by_all_digits (n : ℕ) : Prop := sorry

/-- A function that returns true if a natural number contains the digit 5 -/
def contains_digit_five (n : ℕ) : Prop := sorry

theorem largest_special_number : 
  ∀ n : ℕ, 
    has_distinct_digits n ∧ 
    divisible_by_all_digits n ∧ 
    contains_digit_five n →
    n ≤ 9315 :=
sorry

end NUMINAMATH_CALUDE_largest_special_number_l3707_370776


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_equals_half_l3707_370756

theorem cos_sixty_degrees_equals_half : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_equals_half_l3707_370756


namespace NUMINAMATH_CALUDE_jones_family_puzzle_l3707_370784

/-- Represents a 4-digit number where one digit appears three times and another once -/
structure LicensePlate where
  digits : Fin 4 → Nat
  pattern : ∃ (a b : Nat), (∀ i, digits i = a ∨ digits i = b) ∧
                           (∃ j, digits j ≠ digits ((j + 1) % 4))

/-- Mr. Jones' family setup -/
structure JonesFamily where
  license : LicensePlate
  children_ages : Finset Nat
  jones_age : Nat
  h1 : children_ages.card = 8
  h2 : 12 ∈ children_ages
  h3 : ∀ age ∈ children_ages, license.digits 0 * 1000 + license.digits 1 * 100 + 
                               license.digits 2 * 10 + license.digits 3 % age = 0
  h4 : jones_age = license.digits 1 * 10 + license.digits 0

theorem jones_family_puzzle (family : JonesFamily) : 11 ∉ family.children_ages := by
  sorry

end NUMINAMATH_CALUDE_jones_family_puzzle_l3707_370784


namespace NUMINAMATH_CALUDE_rice_consumption_l3707_370740

theorem rice_consumption (initial_rice : ℕ) (daily_consumption : ℕ) (days : ℕ) 
  (h1 : initial_rice = 52)
  (h2 : daily_consumption = 9)
  (h3 : days = 3) :
  initial_rice - (daily_consumption * days) = 25 := by
  sorry

end NUMINAMATH_CALUDE_rice_consumption_l3707_370740


namespace NUMINAMATH_CALUDE_four_projectors_illuminate_plane_l3707_370726

/-- Represents a point on a plane with a projector --/
structure ProjectorPoint where
  x : ℝ
  y : ℝ
  direction : Nat -- 0: North, 1: East, 2: South, 3: West

/-- Represents the illuminated area by a projector --/
def illuminatedArea (p : ProjectorPoint) : Set (ℝ × ℝ) :=
  sorry

/-- The entire plane --/
def entirePlane : Set (ℝ × ℝ) :=
  sorry

/-- Theorem stating that four projector points can illuminate the entire plane --/
theorem four_projectors_illuminate_plane (p1 p2 p3 p4 : ProjectorPoint) :
  ∃ (d1 d2 d3 d4 : Nat), 
    (d1 < 4 ∧ d2 < 4 ∧ d3 < 4 ∧ d4 < 4) ∧
    (illuminatedArea {p1 with direction := d1} ∪ 
     illuminatedArea {p2 with direction := d2} ∪
     illuminatedArea {p3 with direction := d3} ∪
     illuminatedArea {p4 with direction := d4}) = entirePlane :=
  sorry

end NUMINAMATH_CALUDE_four_projectors_illuminate_plane_l3707_370726


namespace NUMINAMATH_CALUDE_orange_pricing_and_purchase_l3707_370766

-- Define variables
variable (x y m : ℝ)

-- Define the theorem
theorem orange_pricing_and_purchase :
  -- Conditions
  (3 * x + 2 * y = 78) →
  (2 * x + 3 * y = 72) →
  (18 * m + 12 * (100 - m) ≤ 1440) →
  (m ≤ 100) →
  -- Conclusions
  (x = 18 ∧ y = 12) ∧
  (∀ n, n ≤ 100 ∧ 18 * n + 12 * (100 - n) ≤ 1440 → n ≤ 40) :=
by sorry

end NUMINAMATH_CALUDE_orange_pricing_and_purchase_l3707_370766


namespace NUMINAMATH_CALUDE_contradiction_assumption_l3707_370787

theorem contradiction_assumption (a b : ℝ) : ¬(a > b) ↔ (a ≤ b) := by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l3707_370787


namespace NUMINAMATH_CALUDE_triangle_properties_l3707_370727

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B + t.b * Real.cos t.A = t.c / (2 * Real.cos t.C) ∧
  t.c = 6 ∧
  2 * Real.sqrt 3 = t.c * Real.sin t.C / 2

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.C = π / 3 ∧ t.a + t.b + t.c = 6 * Real.sqrt 3 + 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3707_370727


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l3707_370712

theorem coefficient_of_x_squared (x : ℝ) :
  ∃ (k n : ℝ), (3 * x + 2) * (2 * x - 7) = 6 * x^2 + k * x + n := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l3707_370712


namespace NUMINAMATH_CALUDE_max_stuck_guests_l3707_370743

/-- Represents a guest with their galosh size -/
structure Guest where
  size : Nat

/-- Represents the state of remaining guests and galoshes -/
structure State where
  guests : List Guest
  galoshes : List Nat

/-- Checks if a guest can wear a galosh -/
def canWear (g : Guest) (s : Nat) : Bool :=
  g.size ≤ s

/-- Defines a valid initial state with 10 guests and galoshes -/
def validInitialState (s : State) : Prop :=
  s.guests.length = 10 ∧ 
  s.galoshes.length = 10 ∧
  s.guests.map Guest.size = s.galoshes ∧
  s.galoshes.Nodup

/-- Defines a stuck state where no remaining guest can wear any remaining galosh -/
def isStuckState (s : State) : Prop :=
  ∀ g ∈ s.guests, ∀ sz ∈ s.galoshes, ¬ canWear g sz

/-- The main theorem stating the maximum number of guests that could be left -/
theorem max_stuck_guests (s : State) (h : validInitialState s) :
  ∀ s' : State, (∃ seq : List (Guest × Nat), s.guests.Sublist s'.guests ∧ 
                                             s.galoshes.Sublist s'.galoshes ∧
                                             isStuckState s') →
    s'.guests.length ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_stuck_guests_l3707_370743


namespace NUMINAMATH_CALUDE_largest_of_twenty_consecutive_even_integers_with_sum_3000_l3707_370780

/-- Represents a sequence of consecutive even integers -/
structure ConsecutiveEvenIntegers where
  start : ℤ
  count : ℕ
  is_even : Even start

/-- The sum of the sequence -/
def sum_sequence (seq : ConsecutiveEvenIntegers) : ℤ :=
  seq.count * (2 * seq.start + (seq.count - 1) * 2) / 2

/-- The largest integer in the sequence -/
def largest_integer (seq : ConsecutiveEvenIntegers) : ℤ :=
  seq.start + 2 * (seq.count - 1)

theorem largest_of_twenty_consecutive_even_integers_with_sum_3000 :
  ∀ seq : ConsecutiveEvenIntegers,
    seq.count = 20 →
    sum_sequence seq = 3000 →
    largest_integer seq = 169 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_twenty_consecutive_even_integers_with_sum_3000_l3707_370780


namespace NUMINAMATH_CALUDE_inequality_solution_l3707_370725

theorem inequality_solution (x : ℝ) : 
  1 / (x * (x + 1)) - 1 / ((x + 2) * (x + 3)) < 1 / 5 ↔ 
  x < -3 ∨ (-2 < x ∧ x < -1) ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3707_370725


namespace NUMINAMATH_CALUDE_triangle_t_range_l3707_370735

theorem triangle_t_range (A B C : ℝ) (a b c : ℝ) (t : ℝ) :
  0 < B → B < π / 2 →  -- B is acute
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  a * c = (1 / 4) * b ^ 2 →  -- given condition
  Real.sin A + Real.sin C = t * Real.sin B →  -- given condition
  t ∈ Set.Ioo (Real.sqrt 6 / 2) (Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_t_range_l3707_370735


namespace NUMINAMATH_CALUDE_investment_partnership_profit_share_l3707_370744

/-- Investment partnership problem -/
theorem investment_partnership_profit_share
  (invest_A invest_B invest_C invest_D : ℚ)
  (total_profit : ℚ)
  (h1 : invest_A = 3 * invest_B)
  (h2 : invest_B = 2/3 * invest_C)
  (h3 : invest_D = 1/2 * invest_A)
  (h4 : total_profit = 19900) :
  invest_B / (invest_A + invest_B + invest_C + invest_D) * total_profit = 2842.86 := by
sorry


end NUMINAMATH_CALUDE_investment_partnership_profit_share_l3707_370744


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3707_370723

theorem complex_product_magnitude : Complex.abs (3 - 5*Complex.I) * Complex.abs (3 + 5*Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3707_370723


namespace NUMINAMATH_CALUDE_product_and_sum_of_three_two_digit_integers_l3707_370741

theorem product_and_sum_of_three_two_digit_integers : ∃ (a b c : ℕ), 
  10 ≤ a ∧ a < 100 ∧
  10 ≤ b ∧ b < 100 ∧
  10 ≤ c ∧ c < 100 ∧
  a * b * c = 636405 ∧
  a + b + c = 259 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_three_two_digit_integers_l3707_370741


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3707_370765

theorem geometric_sequence_middle_term (b : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 30 * r = b ∧ b * r = 9/4) → b = 3 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3707_370765


namespace NUMINAMATH_CALUDE_car_sale_percentage_increase_l3707_370760

theorem car_sale_percentage_increase 
  (P : ℝ) 
  (discount : ℝ) 
  (profit : ℝ) 
  (buying_price : ℝ) 
  (selling_price : ℝ) :
  discount = 0.1 →
  profit = 0.62000000000000014 →
  buying_price = P * (1 - discount) →
  selling_price = P * (1 + profit) →
  (selling_price - buying_price) / buying_price = 0.8000000000000002 :=
by sorry

end NUMINAMATH_CALUDE_car_sale_percentage_increase_l3707_370760


namespace NUMINAMATH_CALUDE_original_number_proof_l3707_370796

theorem original_number_proof (x : ℚ) : 1 + 1 / x = 11 / 5 → x = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3707_370796


namespace NUMINAMATH_CALUDE_university_admission_problem_l3707_370736

theorem university_admission_problem :
  let n_universities : ℕ := 8
  let n_selected_universities : ℕ := 2
  let n_students : ℕ := 3
  
  (Nat.choose n_universities n_selected_universities) * (2 ^ n_students) = 224 :=
by sorry

end NUMINAMATH_CALUDE_university_admission_problem_l3707_370736


namespace NUMINAMATH_CALUDE_exist_special_integers_l3707_370777

theorem exist_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
  sorry

end NUMINAMATH_CALUDE_exist_special_integers_l3707_370777


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3707_370728

/-- Given two parallel vectors a = (-1, 4) and b = (x, 2), prove that x = -1/2 -/
theorem parallel_vectors_x_value (x : ℚ) :
  let a : ℚ × ℚ := (-1, 4)
  let b : ℚ × ℚ := (x, 2)
  (∃ (k : ℚ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2) →
  x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3707_370728


namespace NUMINAMATH_CALUDE_point_symmetry_and_quadrant_l3707_370721

theorem point_symmetry_and_quadrant (a : ℤ) : 
  (-1 - 2*a > 0) ∧ (2*a - 1 > 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_and_quadrant_l3707_370721


namespace NUMINAMATH_CALUDE_contractor_absent_days_l3707_370758

/-- Proves that given the contract conditions, the number of absent days is 6 -/
theorem contractor_absent_days
  (total_days : ℕ)
  (pay_per_day : ℚ)
  (fine_per_day : ℚ)
  (total_amount : ℚ)
  (h1 : total_days = 30)
  (h2 : pay_per_day = 25)
  (h3 : fine_per_day = 7.5)
  (h4 : total_amount = 555) :
  ∃ (absent_days : ℕ),
    absent_days = 6 ∧
    (pay_per_day * (total_days - absent_days) - fine_per_day * absent_days = total_amount) :=
by sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l3707_370758


namespace NUMINAMATH_CALUDE_initial_speed_calculation_l3707_370761

/-- Represents a baseball player's training progress -/
structure BaseballTraining where
  initialSpeed : ℝ
  trainingWeeks : ℕ
  speedGainPerWeek : ℝ
  finalSpeedIncrease : ℝ

/-- Theorem stating the initial speed of a baseball player given their training progress -/
theorem initial_speed_calculation (training : BaseballTraining)
  (h1 : training.trainingWeeks = 16)
  (h2 : training.speedGainPerWeek = 1)
  (h3 : training.finalSpeedIncrease = 0.2)
  : training.initialSpeed = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_calculation_l3707_370761


namespace NUMINAMATH_CALUDE_constructible_prism_dimensions_l3707_370709

/-- Represents a brick with dimensions 1 × 2 × 4 -/
structure Brick :=
  (length : ℕ := 1)
  (width : ℕ := 2)
  (height : ℕ := 4)

/-- Represents a rectangular prism -/
structure RectangularPrism :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Predicate to check if a prism can be constructed from bricks -/
def can_construct (p : RectangularPrism) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    p.length = a ∧ p.width = 2 * b ∧ p.height = 4 * c

/-- Theorem stating that any constructible prism has dimensions a × 2b × 4c -/
theorem constructible_prism_dimensions (p : RectangularPrism) :
  can_construct p ↔ ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    p.length = a ∧ p.width = 2 * b ∧ p.height = 4 * c :=
sorry

end NUMINAMATH_CALUDE_constructible_prism_dimensions_l3707_370709


namespace NUMINAMATH_CALUDE_divisibility_problem_l3707_370714

theorem divisibility_problem (x q : ℤ) (hx : x > 0) (h_pos : q * x + 197 > 0) 
  (h_197 : 197 % x = 3) : (q * x + 197) % x = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3707_370714
