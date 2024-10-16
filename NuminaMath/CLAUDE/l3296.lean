import Mathlib

namespace NUMINAMATH_CALUDE_special_quadratic_a_range_l3296_329659

/-- A quadratic function satisfying the given conditions -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  max_at_midpoint : ∀ a : ℝ, ∀ x : ℝ, f x ≤ f ((1 - 2*a) / 2)
  decreasing_away_from_zero : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ + x₂ ≠ 0 → f x₁ > f x₂

/-- The range of a for a SpecialQuadratic function -/
theorem special_quadratic_a_range (sq : SpecialQuadratic) : 
  ∀ a : ℝ, (∀ x : ℝ, sq.f x ≤ sq.f ((1 - 2*a) / 2)) → a > 1/2 :=
sorry

end NUMINAMATH_CALUDE_special_quadratic_a_range_l3296_329659


namespace NUMINAMATH_CALUDE_trajectory_is_straight_line_l3296_329670

/-- The set of points (x, y) in ℝ² where x + y = 0 forms a straight line -/
theorem trajectory_is_straight_line :
  {p : ℝ × ℝ | p.1 + p.2 = 0} = {p : ℝ × ℝ | ∃ (t : ℝ), p = (t, -t)} := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_straight_line_l3296_329670


namespace NUMINAMATH_CALUDE_f_properties_l3296_329650

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 1) - a

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x, f a x = f a x → x ∈ {x : ℝ | x < -1 ∨ x > -1}) ∧
  (∀ x, f a x = -f a (-x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3296_329650


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l3296_329671

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age_diff : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 24)
  (h3 : wicket_keeper_age_diff = 7)
  : ∃ (team_avg_age : ℚ),
    team_avg_age = 23 ∧
    (team_size : ℚ) * team_avg_age = 
      captain_age + (captain_age + wicket_keeper_age_diff) + 
      ((team_size - 2) : ℚ) * (team_avg_age - 1) :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l3296_329671


namespace NUMINAMATH_CALUDE_middle_number_is_four_l3296_329691

/-- Represents a triple of positive integers -/
structure Triple where
  left : Nat
  middle : Nat
  right : Nat
  left_pos : 0 < left
  middle_pos : 0 < middle
  right_pos : 0 < right

/-- Checks if a triple satisfies the problem conditions -/
def validTriple (t : Triple) : Prop :=
  t.left < t.middle ∧ t.middle < t.right ∧ t.left + t.middle + t.right = 15

/-- Casey cannot determine the other two numbers -/
def caseyUncertain (t : Triple) : Prop :=
  ∃ t' : Triple, t'.left = t.left ∧ validTriple t' ∧ t' ≠ t

/-- Tracy cannot determine the other two numbers -/
def tracyUncertain (t : Triple) : Prop :=
  ∃ t' : Triple, t'.right = t.right ∧ validTriple t' ∧ t' ≠ t

/-- Stacy cannot determine the other two numbers -/
def stacyUncertain (t : Triple) : Prop :=
  ∃ t' : Triple, t'.middle = t.middle ∧ validTriple t' ∧ t' ≠ t

/-- The main theorem stating that the middle number must be 4 -/
theorem middle_number_is_four :
  ∀ t : Triple,
    validTriple t →
    caseyUncertain t →
    tracyUncertain t →
    stacyUncertain t →
    t.middle = 4 := by
  sorry


end NUMINAMATH_CALUDE_middle_number_is_four_l3296_329691


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l3296_329656

/-- Parabola structure -/
structure Parabola where
  focus : Point
  directrix : Line
  axis_of_symmetry : Line

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Area of a triangle given three points -/
def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- Checks if a line is perpendicular to another line -/
def is_perpendicular (l1 l2 : Line) : Prop := sorry

/-- Checks if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

/-- Checks if a line passes through a point -/
def line_passes_through (l : Line) (p : Point) : Prop := sorry

/-- Main theorem -/
theorem parabola_triangle_area 
  (C : Parabola) 
  (l : Line) 
  (A B P : Point) :
  line_passes_through l C.focus →
  is_perpendicular l C.axis_of_symmetry →
  point_on_line A l →
  point_on_line B l →
  distance A B = 12 →
  point_on_line P C.directrix →
  triangle_area A B P = 36 := by sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l3296_329656


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3296_329676

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 11) (h2 : a * b = 21) :
  a^3 + b^3 = 638 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3296_329676


namespace NUMINAMATH_CALUDE_expected_remaining_bullets_value_l3296_329637

/-- The probability of hitting the target with each shot -/
def hit_probability : ℝ := 0.6

/-- The total number of available bullets -/
def total_bullets : ℕ := 4

/-- The expected number of remaining bullets after the first hit -/
def expected_remaining_bullets : ℝ :=
  3 * hit_probability +
  2 * hit_probability * (1 - hit_probability) +
  1 * hit_probability * (1 - hit_probability)^2 +
  0 * (1 - hit_probability)^3

theorem expected_remaining_bullets_value :
  expected_remaining_bullets = 2.376 := by sorry

end NUMINAMATH_CALUDE_expected_remaining_bullets_value_l3296_329637


namespace NUMINAMATH_CALUDE_total_cost_of_tickets_l3296_329604

def total_tickets : ℕ := 29
def cheap_ticket_price : ℕ := 7
def expensive_ticket_price : ℕ := 9
def expensive_tickets : ℕ := 11

theorem total_cost_of_tickets : 
  cheap_ticket_price * (total_tickets - expensive_tickets) + 
  expensive_ticket_price * expensive_tickets = 225 := by sorry

end NUMINAMATH_CALUDE_total_cost_of_tickets_l3296_329604


namespace NUMINAMATH_CALUDE_sequence_is_constant_l3296_329649

/-- A sequence of positive integers -/
def Sequence := ℕ → ℕ

/-- The divisibility condition for the sequence -/
def DivisibilityCondition (a : Sequence) : Prop :=
  ∀ i j : ℕ, i > j → (i - j)^(2*(i - j)) + 1 ∣ a i - a j

/-- The theorem stating that a sequence satisfying the divisibility condition is constant -/
theorem sequence_is_constant (a : Sequence) (h : DivisibilityCondition a) :
  ∀ n m : ℕ, a n = a m :=
sorry

end NUMINAMATH_CALUDE_sequence_is_constant_l3296_329649


namespace NUMINAMATH_CALUDE_line_slope_l3296_329694

theorem line_slope (x y : ℝ) : 
  x - Real.sqrt 3 * y + 3 = 0 → 
  (y - Real.sqrt 3) / (x - (-3)) = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_l3296_329694


namespace NUMINAMATH_CALUDE_four_valid_digits_l3296_329689

-- Define a function to represent a three-digit number in the form 2C4
def number (C : Nat) : Nat := 200 + 10 * C + 4

-- Define the condition for C to be a valid digit
def valid_digit (C : Nat) : Prop := C ≥ 0 ∧ C ≤ 9

-- Define the divisibility condition
def divisible_by_four (C : Nat) : Prop := number C % 4 = 0

-- The main theorem
theorem four_valid_digits :
  ∃ (S : Finset Nat), (∀ C ∈ S, valid_digit C ∧ divisible_by_four C) ∧ S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_valid_digits_l3296_329689


namespace NUMINAMATH_CALUDE_union_of_sets_l3296_329646

def setA : Set ℝ := {x | x + 2 > 0}
def setB : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem union_of_sets : setA ∪ setB = Set.Ioi (-2) := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l3296_329646


namespace NUMINAMATH_CALUDE_linear_function_increasing_l3296_329696

/-- A linear function y = mx + b where m = k - 2 and b = 3 -/
def linearFunction (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + 3

/-- The property that y increases as x increases -/
def increasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem linear_function_increasing (k : ℝ) :
  increasingFunction (linearFunction k) → k > 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l3296_329696


namespace NUMINAMATH_CALUDE_plane_equation_l3296_329638

theorem plane_equation (s t x y z : ℝ) : 
  (∃ (s t : ℝ), x = 3 + 2*s - 3*t ∧ y = 1 + s ∧ z = 4 - 3*s + t) ↔ 
  (x - 7*y + 3*z - 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_l3296_329638


namespace NUMINAMATH_CALUDE_inequality_preservation_l3296_329642

theorem inequality_preservation (x y : ℝ) (h : x > y) : x - 3 > y - 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3296_329642


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l3296_329673

/-- Given a rhombus with area 150 square units and diagonals in ratio 4:3, prove its longest diagonal is 20 units -/
theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℚ) (d₁ d₂ : ℝ) : 
  area = 150 →
  ratio = 4/3 →
  d₁/d₂ = ratio →
  area = (1/2) * d₁ * d₂ →
  d₁ > d₂ →
  d₁ = 20 := by sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l3296_329673


namespace NUMINAMATH_CALUDE_phantom_needs_43_more_l3296_329603

/-- The amount of money Phantom's mom gave him initially -/
def initial_amount : ℕ := 50

/-- The cost of one black printer ink -/
def black_ink_cost : ℕ := 11

/-- The number of black printer inks Phantom wants to buy -/
def black_ink_count : ℕ := 2

/-- The cost of one red printer ink -/
def red_ink_cost : ℕ := 15

/-- The number of red printer inks Phantom wants to buy -/
def red_ink_count : ℕ := 3

/-- The cost of one yellow printer ink -/
def yellow_ink_cost : ℕ := 13

/-- The number of yellow printer inks Phantom wants to buy -/
def yellow_ink_count : ℕ := 2

/-- The additional amount Phantom needs to ask his mom -/
def additional_amount : ℕ := 43

theorem phantom_needs_43_more :
  (black_ink_cost * black_ink_count +
   red_ink_cost * red_ink_count +
   yellow_ink_cost * yellow_ink_count) - initial_amount = additional_amount := by
  sorry

end NUMINAMATH_CALUDE_phantom_needs_43_more_l3296_329603


namespace NUMINAMATH_CALUDE_min_arcs_for_circle_l3296_329609

theorem min_arcs_for_circle (arc_measure : ℝ) (n : ℕ) : 
  arc_measure = 120 → 
  (n : ℝ) * arc_measure = 360 → 
  n ≥ 3 ∧ ∀ m : ℕ, m < n → (m : ℝ) * arc_measure ≠ 360 :=
by sorry

end NUMINAMATH_CALUDE_min_arcs_for_circle_l3296_329609


namespace NUMINAMATH_CALUDE_binary_sum_equals_638_l3296_329685

/-- The sum of the binary numbers 111111111₂ and 1111111₂ is equal to 638 in base 10. -/
theorem binary_sum_equals_638 : 
  (2^9 - 1) + (2^7 - 1) = 638 := by
  sorry

#check binary_sum_equals_638

end NUMINAMATH_CALUDE_binary_sum_equals_638_l3296_329685


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l3296_329634

noncomputable def f (x : ℝ) : ℝ := (x^5 + 1) / (x^4 + 1)

theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := deriv f x₀
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ → y = (1/2) * x + 1/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l3296_329634


namespace NUMINAMATH_CALUDE_max_value_x_plus_y_l3296_329684

theorem max_value_x_plus_y (x y : ℝ) (h : x - 3 * Real.sqrt (x + 1) = 3 * Real.sqrt (y + 2) - y) :
  (∀ a b : ℝ, a - 3 * Real.sqrt (a + 1) = 3 * Real.sqrt (b + 2) - b → x + y ≥ a + b) ∧
  x + y ≤ 9 + 3 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_y_l3296_329684


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_129_l3296_329648

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithCircle where
  -- The lengths of the parallel sides
  shorterBase : ℝ
  longerBase : ℝ
  -- The length of the leg (equal for both legs in an isosceles trapezoid)
  leg : ℝ
  -- The height of the trapezoid
  height : ℝ
  -- The radius of the inscribed circle
  radius : ℝ
  -- Conditions
  height_positive : height > 0
  radius_positive : radius > 0
  longer_than_shorter : longerBase > shorterBase
  circle_touches_base : shorterBase = 2 * radius
  circle_touches_leg : leg^2 = (longerBase - shorterBase)^2 / 4 + height^2

/-- The perimeter of the trapezoid -/
def perimeter (t : IsoscelesTrapezoidWithCircle) : ℝ :=
  t.shorterBase + t.longerBase + 2 * t.leg

theorem trapezoid_perimeter_is_129
  (t : IsoscelesTrapezoidWithCircle)
  (h₁ : t.height = 36)
  (h₂ : t.radius = 11) :
  perimeter t = 129 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_129_l3296_329648


namespace NUMINAMATH_CALUDE_train_speed_l3296_329695

theorem train_speed (train_length : Real) (man_speed : Real) (passing_time : Real) :
  train_length = 220 →
  man_speed = 6 →
  passing_time = 12 →
  (train_length / 1000) / (passing_time / 3600) - man_speed = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3296_329695


namespace NUMINAMATH_CALUDE_prism_pyramid_sum_l3296_329682

/-- A pentagonal prism with a pyramid attached to one of its pentagonal faces -/
structure PrismWithPyramid where
  prism_faces : Nat
  prism_vertices : Nat
  prism_edges : Nat
  pyramid_faces : Nat
  pyramid_vertex : Nat
  pyramid_edges : Nat

/-- The total number of exterior elements (faces, vertices, edges) of the combined solid -/
def total_elements (solid : PrismWithPyramid) : Nat :=
  (solid.prism_faces - 1 + solid.pyramid_faces) +
  (solid.prism_vertices + solid.pyramid_vertex) +
  (solid.prism_edges + solid.pyramid_edges)

/-- Theorem stating that the sum of exterior faces, vertices, and edges is 42 -/
theorem prism_pyramid_sum (solid : PrismWithPyramid)
  (h1 : solid.prism_faces = 7)
  (h2 : solid.prism_vertices = 10)
  (h3 : solid.prism_edges = 15)
  (h4 : solid.pyramid_faces = 5)
  (h5 : solid.pyramid_vertex = 1)
  (h6 : solid.pyramid_edges = 5) :
  total_elements solid = 42 := by
  sorry

end NUMINAMATH_CALUDE_prism_pyramid_sum_l3296_329682


namespace NUMINAMATH_CALUDE_correct_quotient_l3296_329690

theorem correct_quotient (N : ℕ) (h1 : N % 21 = 0) (h2 : N / 12 = 63) : N / 21 = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_l3296_329690


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l3296_329660

theorem consecutive_integers_product (a : ℕ) : 
  (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) = 15120) → (a + 4 = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l3296_329660


namespace NUMINAMATH_CALUDE_largest_multiple_of_13_less_than_neg_124_l3296_329647

theorem largest_multiple_of_13_less_than_neg_124 :
  ∀ n : ℤ, n * 13 < -124 → n * 13 ≤ -130 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_13_less_than_neg_124_l3296_329647


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l3296_329651

/-- Given vectors a, b, and c in ℝ², prove that if 3a + b is collinear with c, then x = 4 -/
theorem collinear_vectors_x_value (a b c : ℝ × ℝ) (x : ℝ) 
  (ha : a = (-2, 0)) 
  (hb : b = (2, 1)) 
  (hc : c = (x, -1)) 
  (hcollinear : ∃ (k : ℝ), k ≠ 0 ∧ 3 • a + b = k • c) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l3296_329651


namespace NUMINAMATH_CALUDE_rectangle_formations_l3296_329645

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 6

theorem rectangle_formations : 
  (Nat.choose horizontal_lines 2) * (Nat.choose vertical_lines 2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formations_l3296_329645


namespace NUMINAMATH_CALUDE_perfect_square_digit_sum_l3296_329681

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem perfect_square_digit_sum :
  (¬ ∃ n : ℕ, ∃ m : ℕ, n = m^2 ∧ sum_of_digits n = 20) ∧
  (∃ n : ℕ, ∃ m : ℕ, n = m^2 ∧ sum_of_digits n = 10) := by sorry

end NUMINAMATH_CALUDE_perfect_square_digit_sum_l3296_329681


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3296_329640

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₁ + a₃ + a₅ = 21, 
    prove that a₃ + a₅ + a₇ = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : a 1 = 3) 
    (h2 : a 1 + a 3 + a 5 = 21) 
    (h3 : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) : 
  a 3 + a 5 + a 7 = 42 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3296_329640


namespace NUMINAMATH_CALUDE_total_players_on_ground_l3296_329680

theorem total_players_on_ground (cricket_players hockey_players football_players softball_players : ℕ) 
  (h1 : cricket_players = 10)
  (h2 : hockey_players = 12)
  (h3 : football_players = 16)
  (h4 : softball_players = 13) :
  cricket_players + hockey_players + football_players + softball_players = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l3296_329680


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_algebraic_expression_equality_l3296_329654

-- Part 1
theorem sqrt_expression_equality : 2 * Real.sqrt 20 - Real.sqrt 5 + 2 * Real.sqrt (1/5) = (17 * Real.sqrt 5) / 5 := by sorry

-- Part 2
theorem algebraic_expression_equality : 
  (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_algebraic_expression_equality_l3296_329654


namespace NUMINAMATH_CALUDE_student_group_size_l3296_329666

theorem student_group_size (n : ℕ) (h : n > 1) :
  (2 : ℚ) / n = (1 : ℚ) / 5 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_student_group_size_l3296_329666


namespace NUMINAMATH_CALUDE_abs_neg_three_halves_l3296_329643

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_halves_l3296_329643


namespace NUMINAMATH_CALUDE_representatives_selection_count_l3296_329667

def male_students : ℕ := 6
def female_students : ℕ := 3
def total_students : ℕ := male_students + female_students
def representatives : ℕ := 4

theorem representatives_selection_count :
  (Nat.choose total_students representatives) - (Nat.choose male_students representatives) = 111 := by
  sorry

end NUMINAMATH_CALUDE_representatives_selection_count_l3296_329667


namespace NUMINAMATH_CALUDE_abc_product_l3296_329672

theorem abc_product (a b c : ℝ) 
  (h1 : 1/a + 1/b + 1/c = 4)
  (h2 : 4 * (1/(a+b) + 1/(b+c) + 1/(c+a)) = 4)
  (h3 : c/(a+b) + a/(b+c) + b/(c+a) = 4) :
  a * b * c = 49/23 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3296_329672


namespace NUMINAMATH_CALUDE_linear_equation_condition_l3296_329622

/-- A linear equation in two variables x and y of the form mx + 3y = 4x - 1 -/
def linear_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x + 3 * y = 4 * x - 1

/-- The condition for the equation to be linear in two variables -/
def is_linear_in_two_variables (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y, linear_equation m x y ↔ a * x + b * y = c

theorem linear_equation_condition (m : ℝ) :
  is_linear_in_two_variables m ↔ m ≠ 4 :=
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l3296_329622


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3296_329624

theorem solve_linear_equation :
  ∀ x : ℝ, 7 - 2 * x = 15 → x = -4 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3296_329624


namespace NUMINAMATH_CALUDE_difference_even_odd_sums_l3296_329625

/-- Sum of first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := 2 * sum_first_n n

/-- Sum of first n positive odd integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n * n

theorem difference_even_odd_sums : 
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 := by
  sorry

end NUMINAMATH_CALUDE_difference_even_odd_sums_l3296_329625


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3296_329608

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (2 * x + 14) = 10 → x = 43 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3296_329608


namespace NUMINAMATH_CALUDE_remainder_proof_l3296_329662

def smallest_prime_greater_than_1000 : ℕ → Prop :=
  λ x => Prime x ∧ x > 1000 ∧ ∀ y, Prime y ∧ y > 1000 → x ≤ y

theorem remainder_proof (x : ℕ) (h : smallest_prime_greater_than_1000 x) :
  (10000 - 999) % x = 945 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l3296_329662


namespace NUMINAMATH_CALUDE_average_and_differences_l3296_329664

theorem average_and_differences (y : ℝ) : 
  (50 + y) / 2 = 60 →
  y = 70 ∧ 
  |50 - y| = 20 ∧ 
  50 - y = -20 := by
sorry

end NUMINAMATH_CALUDE_average_and_differences_l3296_329664


namespace NUMINAMATH_CALUDE_specific_sequence_terms_l3296_329658

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℕ
  last : ℕ
  diff : ℕ

/-- Calculates the number of terms in an arithmetic sequence -/
def numTerms (seq : ArithmeticSequence) : ℕ :=
  (seq.last - seq.first) / seq.diff + 1

theorem specific_sequence_terms : 
  let seq := ArithmeticSequence.mk 2 3007 5
  numTerms seq = 602 := by
  sorry

end NUMINAMATH_CALUDE_specific_sequence_terms_l3296_329658


namespace NUMINAMATH_CALUDE_sphere_minus_cylinder_volume_l3296_329688

/-- The volume of the space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_minus_cylinder_volume (r_sphere r_cylinder : ℝ) : 
  r_sphere = 7 → r_cylinder = 4 → 
  (4/3 * π * r_sphere^3) - (π * r_cylinder^2 * (2 * r_sphere)) = 700/3 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_minus_cylinder_volume_l3296_329688


namespace NUMINAMATH_CALUDE_volume_bound_l3296_329693

/-- 
Given a body in 3D space, its volume does not exceed the square root of the product 
of the areas of its projections onto the coordinate planes.
-/
theorem volume_bound (S₁ S₂ S₃ V : ℝ) 
  (h₁ : S₁ > 0) (h₂ : S₂ > 0) (h₃ : S₃ > 0) (h₄ : V > 0)
  (h_S₁ : S₁ = area_projection_xy)
  (h_S₂ : S₂ = area_projection_yz)
  (h_S₃ : S₃ = area_projection_zx)
  (h_V : V = volume_of_body) : 
  V ≤ Real.sqrt (S₁ * S₂ * S₃) := by
  sorry

end NUMINAMATH_CALUDE_volume_bound_l3296_329693


namespace NUMINAMATH_CALUDE_problem_solution_l3296_329692

theorem problem_solution (a : ℝ) (h : a = 1 / (Real.sqrt 2 - 1)) : 4 * a^2 - 8 * a - 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3296_329692


namespace NUMINAMATH_CALUDE_checkerboard_squares_l3296_329665

/-- The number of squares of a given size on a rectangular grid -/
def count_squares (rows : ℕ) (cols : ℕ) (size : ℕ) : ℕ :=
  (rows - size + 1) * (cols - size + 1)

/-- The total number of squares on a 3x4 checkerboard -/
def total_squares : ℕ :=
  count_squares 3 4 1 + count_squares 3 4 2 + count_squares 3 4 3

/-- Theorem stating that the total number of squares on a 3x4 checkerboard is 20 -/
theorem checkerboard_squares :
  total_squares = 20 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_squares_l3296_329665


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l3296_329636

/-- The minimum sum of dimensions for a rectangular box with volume 1645 and positive integer dimensions -/
theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 1645 → l + w + h ≥ 129 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l3296_329636


namespace NUMINAMATH_CALUDE_sampling_methods_classification_l3296_329669

-- Define the characteristics of sampling methods
def is_systematic_sampling (method : String) : Prop :=
  method = "Samples at equal time intervals"

def is_simple_random_sampling (method : String) : Prop :=
  method = "Selects individuals from a small population with little difference among them"

-- Define the two sampling methods
def sampling_method_1 : String :=
  "Samples a bag for inspection every 30 minutes in a milk production line"

def sampling_method_2 : String :=
  "Selects 3 students from a group of 30 math enthusiasts in a middle school"

-- Theorem to prove
theorem sampling_methods_classification :
  is_systematic_sampling sampling_method_1 ∧
  is_simple_random_sampling sampling_method_2 := by
  sorry


end NUMINAMATH_CALUDE_sampling_methods_classification_l3296_329669


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_144_l3296_329678

theorem factor_x_squared_minus_144 (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_144_l3296_329678


namespace NUMINAMATH_CALUDE_initial_pens_count_prove_initial_pens_l3296_329652

theorem initial_pens_count : ℕ → Prop :=
  fun initial_pens =>
    let after_mike := initial_pens + 20
    let after_cindy := 2 * after_mike
    let after_sharon := after_cindy - 10
    after_sharon = initial_pens ∧ initial_pens = 30

theorem prove_initial_pens : ∃ (n : ℕ), initial_pens_count n := by
  sorry

end NUMINAMATH_CALUDE_initial_pens_count_prove_initial_pens_l3296_329652


namespace NUMINAMATH_CALUDE_min_votes_for_a_to_win_l3296_329616

/-- Represents the minimum number of votes candidate A needs to win the election -/
def min_votes_to_win (total_votes : ℕ) (first_votes : ℕ) (a_votes : ℕ) (b_votes : ℕ) (c_votes : ℕ) : ℕ :=
  let remaining_votes := total_votes - first_votes
  let a_deficit := b_votes - a_votes
  (remaining_votes - a_deficit) / 2 + a_deficit + 1

theorem min_votes_for_a_to_win :
  let total_votes : ℕ := 1500
  let first_votes : ℕ := 1000
  let a_votes : ℕ := 350
  let b_votes : ℕ := 370
  let c_votes : ℕ := 280
  min_votes_to_win total_votes first_votes a_votes b_votes c_votes = 261 := by
  sorry

#eval min_votes_to_win 1500 1000 350 370 280

end NUMINAMATH_CALUDE_min_votes_for_a_to_win_l3296_329616


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_l3296_329615

theorem quadratic_roots_imply_m (m : ℝ) : 
  (∀ x : ℂ, 8 * x^2 + 4 * x + m = 0 ↔ x = (-2 + Complex.I * Real.sqrt 88) / 8 ∨ x = (-2 - Complex.I * Real.sqrt 88) / 8) → 
  m = 13 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_l3296_329615


namespace NUMINAMATH_CALUDE_no_solution_for_system_l3296_329663

theorem no_solution_for_system : ¬∃ x : ℝ, 
  (1 / (x + 2) + 8 / (x + 6) ≥ 2) ∧ (5 / (x + 1) - 2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_system_l3296_329663


namespace NUMINAMATH_CALUDE_scalar_multiplication_distributivity_l3296_329607

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem scalar_multiplication_distributivity
  (m n : ℝ) (a : V)
  (hm : m ≠ 0) (hn : n ≠ 0) (ha : a ≠ 0) :
  (m + n) • a = m • a + n • a :=
by sorry

end NUMINAMATH_CALUDE_scalar_multiplication_distributivity_l3296_329607


namespace NUMINAMATH_CALUDE_cannot_cut_square_l3296_329630

theorem cannot_cut_square (rectangle_area : ℝ) (square_area : ℝ) 
  (h_rectangle_area : rectangle_area = 582) 
  (h_square_area : square_area = 400) : ¬ ∃ (l w : ℝ), 
  l * w = rectangle_area ∧ 
  l / w = 3 / 2 ∧ 
  w ≥ Real.sqrt square_area := by
sorry

end NUMINAMATH_CALUDE_cannot_cut_square_l3296_329630


namespace NUMINAMATH_CALUDE_power_of_power_l3296_329697

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3296_329697


namespace NUMINAMATH_CALUDE_hemisphere_radius_from_cylinder_l3296_329610

/-- The radius of a hemisphere formed from a cylinder of equal volume --/
theorem hemisphere_radius_from_cylinder (r h R : ℝ) : 
  r = 2 * (2 : ℝ)^(1/3) → 
  h = 12 → 
  π * r^2 * h = (2/3) * π * R^3 → 
  R = 2 * 3^(1/3) := by
sorry

end NUMINAMATH_CALUDE_hemisphere_radius_from_cylinder_l3296_329610


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l3296_329617

theorem gcd_digits_bound (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) →
  (10^6 ≤ b ∧ b < 10^7) →
  (10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) →
  Nat.gcd a b < 10^3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l3296_329617


namespace NUMINAMATH_CALUDE_cube_root_of_special_sum_l3296_329633

theorem cube_root_of_special_sum (m n : ℚ) 
  (h : m + 2*n + Real.sqrt 2 * (2 - n) = Real.sqrt 2 * (Real.sqrt 2 + 6) + 15) :
  (((m : ℝ).sqrt + n) ^ 100) ^ (1/3 : ℝ) = 1 :=
sorry

end NUMINAMATH_CALUDE_cube_root_of_special_sum_l3296_329633


namespace NUMINAMATH_CALUDE_cart_distance_proof_l3296_329661

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ → ℕ :=
  fun i => a₁ + (i - 1) * d

def sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem cart_distance_proof (a₁ d n : ℕ) 
  (h₁ : a₁ = 5) 
  (h₂ : d = 7) 
  (h₃ : n = 30) : 
  sequence_sum a₁ d n = 3195 :=
sorry

end NUMINAMATH_CALUDE_cart_distance_proof_l3296_329661


namespace NUMINAMATH_CALUDE_factorization_equality_l3296_329618

theorem factorization_equality (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3296_329618


namespace NUMINAMATH_CALUDE_line_through_point_l3296_329683

/-- 
Given a line with equation -1/3 - 3kx = 4y that passes through the point (1/3, -8),
prove that k = 95/3.
-/
theorem line_through_point (k : ℚ) : 
  (-1/3 : ℚ) - 3 * k * (1/3 : ℚ) = 4 * (-8 : ℚ) → k = 95/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3296_329683


namespace NUMINAMATH_CALUDE_unique_solution_l3296_329614

/-- A pair of natural numbers (r, x) where r is the base and x is a number in that base -/
structure BaseNumber :=
  (r : ℕ)
  (x : ℕ)

/-- Check if a BaseNumber satisfies the given conditions -/
def satisfiesConditions (bn : BaseNumber) : Prop :=
  -- r is at most 70
  bn.r ≤ 70 ∧
  -- x is represented by repeating a pair of digits
  ∃ (n : ℕ) (a b : ℕ), 
    a < bn.r ∧ b < bn.r ∧
    bn.x = (a * bn.r + b) * (bn.r^(2*n) - 1) / (bn.r^2 - 1) ∧
  -- x^2 in base r consists of 4n ones
  ∃ (n : ℕ), bn.x^2 = (bn.r^(4*n) - 1) / (bn.r - 1)

/-- The theorem stating that (7, 26₇) is the only solution -/
theorem unique_solution : 
  ∀ (bn : BaseNumber), satisfiesConditions bn ↔ bn.r = 7 ∧ bn.x = 26 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3296_329614


namespace NUMINAMATH_CALUDE_rationalize_and_divide_l3296_329641

theorem rationalize_and_divide : (8 / Real.sqrt 8) / 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_divide_l3296_329641


namespace NUMINAMATH_CALUDE_beach_count_theorem_l3296_329632

/-- The total count of oysters and crabs over two days -/
def total_count (initial_oysters initial_crabs : ℕ) : ℕ :=
  initial_oysters + (initial_oysters / 2) +
  initial_crabs + (initial_crabs * 2 / 3)

/-- Theorem stating the total count for the given initial numbers -/
theorem beach_count_theorem :
  total_count 50 72 = 195 := by
  sorry

end NUMINAMATH_CALUDE_beach_count_theorem_l3296_329632


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l3296_329629

/-- Proves that the cost of fencing per meter for a rectangular plot is 26.5 Rs. -/
theorem fencing_cost_per_meter 
  (length : ℝ) 
  (breadth : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 70) 
  (h2 : length = breadth + 40) 
  (h3 : total_cost = 5300) : 
  total_cost / (2 * length + 2 * breadth) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l3296_329629


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_positive_l3296_329668

def S : Set Int := {2, 5, -7, 8, -10}

theorem smallest_sum_of_three_positive : 
  (∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
   a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
   a > 0 ∧ b > 0 ∧ c > 0 ∧
   a + b + c = 15 ∧
   (∀ (x y z : Int), x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y → y ≠ z → x ≠ z → 
    x > 0 → y > 0 → z > 0 → 
    x + y + z ≥ 15)) := by
  sorry

#check smallest_sum_of_three_positive

end NUMINAMATH_CALUDE_smallest_sum_of_three_positive_l3296_329668


namespace NUMINAMATH_CALUDE_inscribed_circle_sector_ratio_l3296_329611

theorem inscribed_circle_sector_ratio :
  ∀ (R r : ℝ),
  R > 0 → r > 0 →
  R = (2 * Real.sqrt 3 + 3) * r / 3 →
  (π * r^2) / ((π * R^2) / 6) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_sector_ratio_l3296_329611


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l3296_329631

/-- An ellipse with equation x²/4 + y²/2 = 1 -/
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 2 = 1

/-- A line with equation y = k(x-1) -/
def Line (k x y : ℝ) : Prop :=
  y = k * (x - 1)

/-- The area of a triangle given three points -/
noncomputable def TriangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- The main theorem -/
theorem ellipse_line_intersection (k : ℝ) :
  (∃ x1 y1 x2 y2,
    Ellipse x1 y1 ∧ Ellipse x2 y2 ∧
    Line k x1 y1 ∧ Line k x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    TriangleArea 2 0 x1 y1 x2 y2 = Real.sqrt 10 / 3) ↔
  k = 1 ∨ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l3296_329631


namespace NUMINAMATH_CALUDE_total_shaded_area_is_one_third_l3296_329612

/-- Represents the fractional area shaded at each step of the square division process -/
def shadedAreaSequence : ℕ → ℚ
  | 0 => 5 / 16
  | n + 1 => shadedAreaSequence n + 5 / 16^(n + 2)

/-- The theorem stating that the total shaded area is 1/3 of the square -/
theorem total_shaded_area_is_one_third :
  (∑' n, shadedAreaSequence n) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_total_shaded_area_is_one_third_l3296_329612


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l3296_329627

/-- The nth term of a geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

/-- The 12th term of the specific geometric sequence is 1/6561 -/
theorem twelfth_term_of_specific_sequence :
  geometric_sequence 27 (1/3) 12 = 1/6561 := by sorry

end NUMINAMATH_CALUDE_twelfth_term_of_specific_sequence_l3296_329627


namespace NUMINAMATH_CALUDE_mirror_to_wall_area_ratio_l3296_329644

/-- The ratio of a square mirror's area to a rectangular wall's area --/
theorem mirror_to_wall_area_ratio
  (mirror_side : ℝ)
  (wall_width : ℝ)
  (wall_length : ℝ)
  (h1 : mirror_side = 24)
  (h2 : wall_width = 42)
  (h3 : wall_length = 27.428571428571427)
  : (mirror_side ^ 2) / (wall_width * wall_length) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_mirror_to_wall_area_ratio_l3296_329644


namespace NUMINAMATH_CALUDE_intersection_M_N_l3296_329686

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x < 2}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3296_329686


namespace NUMINAMATH_CALUDE_smallest_concatenated_multiple_of_2016_l3296_329657

def concatenate (n : ℕ) : ℕ :=
  n * 1001

theorem smallest_concatenated_multiple_of_2016 :
  ∀ A : ℕ, A > 0 →
    (∃ k : ℕ, concatenate A = 2016 * k) →
    A ≥ 288 :=
by sorry

end NUMINAMATH_CALUDE_smallest_concatenated_multiple_of_2016_l3296_329657


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_slope_product_l3296_329601

/-- Given an ellipse and a line intersecting it, proves the relationship between the slopes of the intersecting line and the line connecting the origin to the midpoint of the intersection points. -/
theorem ellipse_line_intersection_slope_product (k1 k2 : ℝ) 
  (h1 : k1 ≠ 0) 
  (h2 : ∃ (P1 P2 P : ℝ × ℝ), 
    (P1.1^2 + 2*P1.2^2 = 2) ∧ 
    (P2.1^2 + 2*P2.2^2 = 2) ∧ 
    (P = ((P1.1 + P2.1)/2, (P1.2 + P2.2)/2)) ∧ 
    (k1 = (P2.2 - P1.2)/(P2.1 - P1.1)) ∧ 
    (k2 = P.2/P.1)) : 
  k1 * k2 = -1/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_slope_product_l3296_329601


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l3296_329679

-- Define the repeating decimals
def x : ℚ := 0.142857142857142857
def y : ℚ := 2.857142857142857142

-- State the theorem
theorem repeating_decimal_fraction : x / y = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l3296_329679


namespace NUMINAMATH_CALUDE_school_sections_l3296_329606

theorem school_sections (num_boys num_girls : ℕ) 
  (h_boys : num_boys = 408) 
  (h_girls : num_girls = 312) : 
  (num_boys / (Nat.gcd num_boys num_girls)) + (num_girls / (Nat.gcd num_boys num_girls)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_l3296_329606


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3296_329675

theorem min_value_sum_squares (x y z : ℝ) (h : x - 2*y - 3*z = 4) :
  ∃ (m : ℝ), m = 8/7 ∧ (∀ x y z : ℝ, x - 2*y - 3*z = 4 → x^2 + y^2 + z^2 ≥ m) ∧
  (∃ x y z : ℝ, x - 2*y - 3*z = 4 ∧ x^2 + y^2 + z^2 = m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3296_329675


namespace NUMINAMATH_CALUDE_cos_750_degrees_l3296_329605

theorem cos_750_degrees : Real.cos (750 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_750_degrees_l3296_329605


namespace NUMINAMATH_CALUDE_more_diamonds_than_rubies_l3296_329698

theorem more_diamonds_than_rubies (diamonds : ℕ) (rubies : ℕ) 
  (h1 : diamonds = 421) (h2 : rubies = 377) : 
  diamonds - rubies = 44 := by sorry

end NUMINAMATH_CALUDE_more_diamonds_than_rubies_l3296_329698


namespace NUMINAMATH_CALUDE_no_real_roots_l3296_329621

theorem no_real_roots : ¬∃ (x : ℝ), x + Real.sqrt (2*x - 6) = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3296_329621


namespace NUMINAMATH_CALUDE_tom_bonus_points_l3296_329674

/-- The number of bonus points earned by an employee based on customers served --/
def bonus_points (customers : ℕ) : ℕ := (customers * 20) / 100

/-- The total number of customers served by Tom --/
def total_customers (customers_per_hour hours : ℕ) : ℕ := customers_per_hour * hours

theorem tom_bonus_points :
  let customers_per_hour : ℕ := 10
  let hours_worked : ℕ := 8
  bonus_points (total_customers customers_per_hour hours_worked) = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_bonus_points_l3296_329674


namespace NUMINAMATH_CALUDE_generated_number_is_square_l3296_329687

/-- Generates a number with n threes followed by 34 -/
def generateNumber (n : ℕ) : ℕ :=
  3 * (10^n - 1) / 9 * 10 + 34

/-- Theorem stating that the generated number is always a perfect square -/
theorem generated_number_is_square (n : ℕ) :
  ∃ k : ℕ, (generateNumber n) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_generated_number_is_square_l3296_329687


namespace NUMINAMATH_CALUDE_M_greater_than_N_l3296_329620

theorem M_greater_than_N (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) : a * b > a + b - 1 := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l3296_329620


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3296_329619

theorem unique_four_digit_number : ∃! x : ℕ,
  1000 ≤ x ∧ x ≤ 9999 ∧
  x % 7 = 0 ∧
  x % 29 = 0 ∧
  (19 * x) % 37 = 3 ∧
  x = 5075 := by
sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3296_329619


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3296_329653

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 635 * n ≡ 1251 * n [ZMOD 30] ∧ 
  ∀ (m : ℕ), m > 0 → 635 * m ≡ 1251 * m [ZMOD 30] → n ≤ m :=
by
  use 15
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3296_329653


namespace NUMINAMATH_CALUDE_andy_cookies_count_l3296_329699

/-- The number of cookies Andy ate -/
def andy_ate : Nat := 3

/-- The number of cookies Andy gave to his brother -/
def brother_cookies : Nat := 5

/-- The number of players in Andy's basketball team -/
def team_size : Nat := 8

/-- The sequence of cookies taken by each team member -/
def team_sequence (n : Nat) : Nat := 2 * n - 1

/-- The total number of cookies Andy had at the start -/
def total_cookies : Nat := andy_ate + brother_cookies + (Finset.sum (Finset.range team_size) team_sequence)

theorem andy_cookies_count : total_cookies = 72 := by sorry

end NUMINAMATH_CALUDE_andy_cookies_count_l3296_329699


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3296_329655

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, x < a ∧ x < 3 ↔ x < a) → a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3296_329655


namespace NUMINAMATH_CALUDE_largest_power_of_five_dividing_factorial_sum_l3296_329623

def factorial (n : ℕ) : ℕ := Nat.factorial n

def divides_exactly (x n y : ℕ) : Prop :=
  (x^n ∣ y) ∧ ¬(x^(n+1) ∣ y)

theorem largest_power_of_five_dividing_factorial_sum :
  ∃ (n : ℕ), n = 26 ∧ divides_exactly 5 n (factorial 98 + factorial 99 + factorial 100) ∧
  ∀ (m : ℕ), m > n → ¬(divides_exactly 5 m (factorial 98 + factorial 99 + factorial 100)) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_five_dividing_factorial_sum_l3296_329623


namespace NUMINAMATH_CALUDE_binomial_sum_27_mod_9_l3296_329613

def binomial_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => Nat.choose n k)

theorem binomial_sum_27_mod_9 :
  (binomial_sum 27 - Nat.choose 27 0) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_27_mod_9_l3296_329613


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3296_329639

theorem triangle_angle_measure (D E F : ℝ) : 
  D = E →                         -- Two angles are congruent
  F = D + 40 →                    -- One angle is 40 degrees more than the congruent angles
  D + E + F = 180 →               -- Sum of angles in a triangle is 180 degrees
  F = 86.67 :=                    -- The measure of angle F is 86.67 degrees
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3296_329639


namespace NUMINAMATH_CALUDE_same_color_probability_l3296_329635

/-- The probability of drawing two balls of the same color from a bag containing
    8 green balls, 5 red balls, and 7 blue balls, with replacement. -/
theorem same_color_probability :
  let total_balls : ℕ := 8 + 5 + 7
  let p_green : ℚ := 8 / total_balls
  let p_red : ℚ := 5 / total_balls
  let p_blue : ℚ := 7 / total_balls
  let p_same_color : ℚ := p_green ^ 2 + p_red ^ 2 + p_blue ^ 2
  p_same_color = 117 / 200 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3296_329635


namespace NUMINAMATH_CALUDE_thirteen_people_handshakes_l3296_329626

/-- The number of handshakes in a room with n people, where each person shakes hands with everyone else. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a room with 13 people, where each person shakes hands with everyone else, the total number of handshakes is 78. -/
theorem thirteen_people_handshakes :
  handshakes 13 = 78 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_people_handshakes_l3296_329626


namespace NUMINAMATH_CALUDE_fourth_week_sugar_l3296_329600

def sugar_reduction (initial_amount : ℚ) (weeks : ℕ) : ℚ :=
  initial_amount / (2 ^ weeks)

theorem fourth_week_sugar : sugar_reduction 24 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_week_sugar_l3296_329600


namespace NUMINAMATH_CALUDE_expression_equality_l3296_329628

theorem expression_equality : 
  Real.sqrt 4 + |1 - Real.sqrt 3| - (1/2)⁻¹ + 2023^0 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3296_329628


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3296_329677

theorem rectangular_field_area (m : ℝ) : ∃ m : ℝ, (3*m + 8)*(m - 3) = 100 ∧ m > 0 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3296_329677


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3296_329602

theorem simplify_sqrt_expression :
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3296_329602
