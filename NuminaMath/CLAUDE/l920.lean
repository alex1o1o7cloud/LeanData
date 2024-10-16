import Mathlib

namespace NUMINAMATH_CALUDE_cost_function_property_l920_92098

/-- A function representing the cost with respect to some parameter b -/
def cost_function (f : ℝ → ℝ) : Prop :=
  ∀ b : ℝ, f (2 * b) = 16 * f b

/-- Theorem stating that if doubling the input results in a cost that is 1600% of the original,
    then f(2b) = 16f(b) for any value of b -/
theorem cost_function_property (f : ℝ → ℝ) (h : ∀ b : ℝ, f (2 * b) = 16 * f b) :
  cost_function f := by sorry

end NUMINAMATH_CALUDE_cost_function_property_l920_92098


namespace NUMINAMATH_CALUDE_sphere_circle_paint_equivalence_l920_92044

theorem sphere_circle_paint_equivalence (r_sphere r_circle : ℝ) : 
  r_sphere = 3 → 
  4 * π * r_sphere^2 = π * r_circle^2 → 
  r_circle = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_circle_paint_equivalence_l920_92044


namespace NUMINAMATH_CALUDE_share_ratio_a_to_b_l920_92022

/-- Proof of the ratio of shares between A and B --/
theorem share_ratio_a_to_b (total amount : ℕ) (a_share b_share c_share : ℕ) :
  amount = 510 →
  a_share = 360 →
  b_share = 90 →
  c_share = 60 →
  b_share = c_share / 4 →
  a_share / b_share = 4 :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_a_to_b_l920_92022


namespace NUMINAMATH_CALUDE_income_growth_equation_correct_l920_92065

/-- Represents the growth of per capita disposable income in China from 2020 to 2022 -/
def income_growth (x : ℝ) : Prop :=
  let income_2020 : ℝ := 3.2  -- in ten thousand yuan
  let income_2022 : ℝ := 3.7  -- in ten thousand yuan
  let years : ℕ := 2
  income_2020 * (1 + x) ^ years = income_2022

/-- Theorem stating that the equation correctly represents the income growth -/
theorem income_growth_equation_correct :
  ∃ x : ℝ, income_growth x := by
  sorry

end NUMINAMATH_CALUDE_income_growth_equation_correct_l920_92065


namespace NUMINAMATH_CALUDE_quadratic_radical_for_all_reals_l920_92073

theorem quadratic_radical_for_all_reals (a : ℝ) : ∃ (x : ℝ), x^2 = a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_for_all_reals_l920_92073


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l920_92047

-- Define the properties of triangles
def Equilateral (t : Type) : Prop := sorry
def Isosceles (t : Type) : Prop := sorry

-- State the given true statement
axiom equilateral_implies_isosceles : ∀ t : Type, Equilateral t → Isosceles t

-- Define the converse and inverse
def converse : Prop := ∀ t : Type, Isosceles t → Equilateral t
def inverse : Prop := ∀ t : Type, ¬(Equilateral t) → ¬(Isosceles t)

-- Theorem to prove
theorem converse_and_inverse_false : ¬converse ∧ ¬inverse := by
  sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l920_92047


namespace NUMINAMATH_CALUDE_sixth_power_sum_of_roots_l920_92060

theorem sixth_power_sum_of_roots (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 5 + 2 = 0 → 
  s^2 - 2*s*Real.sqrt 5 + 2 = 0 → 
  r^6 + s^6 = 3904 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_of_roots_l920_92060


namespace NUMINAMATH_CALUDE_pens_count_l920_92066

/-- Given a ratio of pens to markers as 2:5 and 25 markers, prove that the number of pens is 10 -/
theorem pens_count (markers : ℕ) (h1 : markers = 25) : 
  (2 : ℚ) / 5 * markers = 10 := by
  sorry

#check pens_count

end NUMINAMATH_CALUDE_pens_count_l920_92066


namespace NUMINAMATH_CALUDE_inheritance_calculation_l920_92003

theorem inheritance_calculation (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 20000) → x = 55172 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l920_92003


namespace NUMINAMATH_CALUDE_economics_test_absentees_l920_92045

theorem economics_test_absentees (total_students : ℕ) (q1_correct : ℕ) (q2_correct : ℕ) (both_correct : ℕ) 
  (h1 : total_students = 29)
  (h2 : q1_correct = 19)
  (h3 : q2_correct = 24)
  (h4 : both_correct = 19) :
  total_students - (q1_correct + q2_correct - both_correct) = 5 := by
  sorry


end NUMINAMATH_CALUDE_economics_test_absentees_l920_92045


namespace NUMINAMATH_CALUDE_cards_per_student_l920_92036

/-- Given that Joseph had 357 cards initially, has 15 students, and had 12 cards left after distribution,
    prove that the number of cards given to each student is 23. -/
theorem cards_per_student (total_cards : Nat) (num_students : Nat) (remaining_cards : Nat)
    (h1 : total_cards = 357)
    (h2 : num_students = 15)
    (h3 : remaining_cards = 12) :
    (total_cards - remaining_cards) / num_students = 23 :=
by sorry

end NUMINAMATH_CALUDE_cards_per_student_l920_92036


namespace NUMINAMATH_CALUDE_only_A_in_first_quadrant_l920_92023

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def point_A : ℝ × ℝ := (3, 2)
def point_B : ℝ × ℝ := (-3, 2)
def point_C : ℝ × ℝ := (3, -2)
def point_D : ℝ × ℝ := (-3, -2)

theorem only_A_in_first_quadrant :
  first_quadrant point_A.1 point_A.2 ∧
  ¬first_quadrant point_B.1 point_B.2 ∧
  ¬first_quadrant point_C.1 point_C.2 ∧
  ¬first_quadrant point_D.1 point_D.2 := by
  sorry

end NUMINAMATH_CALUDE_only_A_in_first_quadrant_l920_92023


namespace NUMINAMATH_CALUDE_difference_of_numbers_l920_92093

theorem difference_of_numbers (x y : ℝ) : 
  x + y = 20 → x^2 - y^2 = 160 → x - y = 8 := by sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l920_92093


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l920_92048

/-- The mass of a man who causes a boat to sink by a certain amount -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating the mass of the man in the given problem -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 8
  let boat_breadth : ℝ := 2
  let boat_sink_height : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000     -- kg/m³
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 160 := by
  sorry

#check mass_of_man_on_boat

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l920_92048


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l920_92089

def numKnights : ℕ := 30
def chosenKnights : ℕ := 4

def prob_adjacent_knights : ℚ :=
  1 - (Nat.choose (numKnights - chosenKnights + 1) (chosenKnights - 1) : ℚ) /
      (Nat.choose numKnights chosenKnights : ℚ)

theorem adjacent_knights_probability :
  prob_adjacent_knights = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l920_92089


namespace NUMINAMATH_CALUDE_line_equation_through_two_points_l920_92085

/-- The equation of a line passing through two points is x + y = 1 -/
theorem line_equation_through_two_points :
  ∀ (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ),
  A = (1, -2) →
  B = (-3, 2) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ ((x - 1) * (2 - (-2)) = (y - (-2)) * ((-3) - 1))) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x + y = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_two_points_l920_92085


namespace NUMINAMATH_CALUDE_bag_of_balls_l920_92056

theorem bag_of_balls (white green yellow red purple : ℕ) 
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 17)
  (h4 : red = 3)
  (h5 : purple = 1)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 95 / 100) :
  white + green + yellow + red + purple = 80 := by
  sorry

end NUMINAMATH_CALUDE_bag_of_balls_l920_92056


namespace NUMINAMATH_CALUDE_translation_of_parabola_l920_92084

theorem translation_of_parabola (t m : ℝ) : 
  (∀ x : ℝ, (x - 3)^2 = (t - 3)^2 → x = t) →  -- P is on y=(x-3)^2
  (t - m)^2 = (t - 3)^2 →                     -- Q is on y=x^2
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_translation_of_parabola_l920_92084


namespace NUMINAMATH_CALUDE_least_zogs_for_dropping_advantage_l920_92061

/-- Score for dropping n zogs -/
def drop_score (n : ℕ) : ℕ := n * (n + 1)

/-- Score for eating n zogs -/
def eat_score (n : ℕ) : ℕ := 8 * n

/-- Predicate for when dropping earns more points than eating -/
def dropping_beats_eating (n : ℕ) : Prop := drop_score n > eat_score n

theorem least_zogs_for_dropping_advantage : 
  (∀ k < 8, ¬dropping_beats_eating k) ∧ dropping_beats_eating 8 := by sorry

end NUMINAMATH_CALUDE_least_zogs_for_dropping_advantage_l920_92061


namespace NUMINAMATH_CALUDE_second_half_speed_l920_92088

/-- Represents the speed of a car during a trip -/
structure TripSpeed where
  average : ℝ
  firstHalf : ℝ
  secondHalf : ℝ

/-- Theorem stating the speed of the car in the second half of the trip -/
theorem second_half_speed (trip : TripSpeed) (h1 : trip.average = 60) (h2 : trip.firstHalf = 75) :
  trip.secondHalf = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_half_speed_l920_92088


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l920_92043

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity : ∀ (a b : ℝ) (P : ℝ × ℝ),
  a > 0 → b > 0 →
  -- Hyperbola equation
  P.1^2 / a^2 - P.2^2 / b^2 = 1 →
  -- P is on the curve y = √x
  P.2 = Real.sqrt P.1 →
  -- Tangent line passes through the left focus (-1, 0)
  (Real.sqrt P.1 - 0) / (P.1 - (-1)) = 1 / (2 * Real.sqrt P.1) →
  -- The eccentricity is (√5 + 1) / 2
  a / Real.sqrt (a^2 + b^2) = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l920_92043


namespace NUMINAMATH_CALUDE_roof_length_width_difference_l920_92067

-- Define the trapezoidal roof
structure TrapezoidalRoof where
  width : ℝ
  length : ℝ
  height : ℝ
  area : ℝ

-- Define the conditions of the problem
def roof_conditions (roof : TrapezoidalRoof) : Prop :=
  roof.length = 3 * roof.width ∧
  roof.height = 25 ∧
  roof.area = 675 ∧
  roof.area = (1 / 2) * (roof.width + roof.length) * roof.height

-- Theorem to prove
theorem roof_length_width_difference (roof : TrapezoidalRoof) 
  (h : roof_conditions roof) : roof.length - roof.width = 27 := by
  sorry


end NUMINAMATH_CALUDE_roof_length_width_difference_l920_92067


namespace NUMINAMATH_CALUDE_probability_one_third_implies_five_l920_92059

def integer_list : List ℕ := [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

def count (n : ℕ) (l : List ℕ) : ℕ := (l.filter (· = n)).length

theorem probability_one_third_implies_five :
  ∀ n : ℕ, 
  (count n integer_list : ℚ) / (integer_list.length : ℚ) = 1 / 3 →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_probability_one_third_implies_five_l920_92059


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l920_92004

/-- Given two parallel lines 3x - 2y - 1 = 0 and 6x + ay + c = 0 with a distance of 2√13/13 between them, prove that (c + 2)/a = 1 -/
theorem parallel_lines_distance (a c : ℝ) : 
  (∀ x y : ℝ, 3 * x - 2 * y - 1 = 0 ↔ 6 * x + a * y + c = 0) →  -- lines are equivalent
  (∃ k : ℝ, k ≠ 0 ∧ 3 = k * 6 ∧ -2 = k * a) →  -- lines are parallel
  (|c/2 + 1| / Real.sqrt 13 = 2 * Real.sqrt 13 / 13) →  -- distance between lines
  (c + 2) / a = 1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l920_92004


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l920_92026

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 2000)
  (h2 : final_price = 1620)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ (x : ℝ), x > 0 ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price ∧ x = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l920_92026


namespace NUMINAMATH_CALUDE_thirteenth_term_l920_92090

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (a 1 + a 9 = 16) ∧
  (a 4 = 1)

/-- The 13th term of the arithmetic sequence is 64 -/
theorem thirteenth_term (a : ℕ → ℚ) (h : arithmetic_sequence a) : a 13 = 64 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_term_l920_92090


namespace NUMINAMATH_CALUDE_complex_norm_problem_l920_92046

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 12)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z - w) = 7) :
  Complex.abs w = Real.sqrt 36.75 :=
sorry

end NUMINAMATH_CALUDE_complex_norm_problem_l920_92046


namespace NUMINAMATH_CALUDE_sin_symmetry_l920_92050

theorem sin_symmetry (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x + π / 3)
  let g (x : ℝ) := f (x - π / 12)
  ∀ t, g ((-π / 12) + t) = g ((-π / 12) - t) :=
by sorry

end NUMINAMATH_CALUDE_sin_symmetry_l920_92050


namespace NUMINAMATH_CALUDE_line_passes_through_intersections_l920_92031

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 3*x - y = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y = 0

/-- Line equation -/
def line (x y : ℝ) : Prop := x - 2*y = 0

/-- Theorem stating that the line passes through the intersection points of the circles -/
theorem line_passes_through_intersections :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_intersections_l920_92031


namespace NUMINAMATH_CALUDE_table_relationship_l920_92094

def f (x : ℝ) : ℝ := 200 - 3*x - 6*x^2

theorem table_relationship : 
  (f 0 = 200) ∧ 
  (f 2 = 152) ∧ 
  (f 4 = 80) ∧ 
  (f 6 = -16) ∧ 
  (f 8 = -128) := by
  sorry

end NUMINAMATH_CALUDE_table_relationship_l920_92094


namespace NUMINAMATH_CALUDE_no_integer_solution_l920_92053

theorem no_integer_solution : ¬ ∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l920_92053


namespace NUMINAMATH_CALUDE_expected_value_theorem_l920_92032

def N : ℕ := 123456789

/-- The expected value of N' when two distinct digits of N are randomly swapped -/
def expected_value_N_prime : ℚ := 555555555

/-- Theorem stating that the expected value of N' is 555555555 -/
theorem expected_value_theorem : expected_value_N_prime = 555555555 := by sorry

end NUMINAMATH_CALUDE_expected_value_theorem_l920_92032


namespace NUMINAMATH_CALUDE_parabola_midpoint_distance_l920_92063

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = x^2 + 3*x + 2

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

-- Define the square of the distance between two points
def square_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2

-- Theorem statement
theorem parabola_midpoint_distance 
  (C D : PointOnParabola) 
  (h : is_midpoint C.x C.y D.x D.y) : 
  square_distance C.x C.y D.x D.y = 16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_midpoint_distance_l920_92063


namespace NUMINAMATH_CALUDE_min_value_cube_sum_plus_inverse_cube_equality_condition_l920_92015

theorem min_value_cube_sum_plus_inverse_cube (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + 1 / (a + b)^3 ≥ 4^(1/4) :=
sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + 1 / (a + b)^3 = 4^(1/4) ↔ a = b ∧ a = (4^(1/4) / 2)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_cube_sum_plus_inverse_cube_equality_condition_l920_92015


namespace NUMINAMATH_CALUDE_coefficient_of_x_l920_92029

theorem coefficient_of_x (x : ℝ) : 
  let expression := 5*(x - 6) + 3*(9 - 3*x^2 + 2*x) - 10*(3*x - 2)
  ∃ (a b c : ℝ), expression = a*x^2 + (-19)*x + c :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l920_92029


namespace NUMINAMATH_CALUDE_set_classification_l920_92091

/-- The set of numbers we're working with -/
def S : Set ℝ := {-2, -3.14, 0.3, 0, Real.pi/3, 22/7, -0.1212212221}

/-- The set of positive numbers in S -/
def positiveS : Set ℝ := {x ∈ S | x > 0}

/-- The set of negative numbers in S -/
def negativeS : Set ℝ := {x ∈ S | x < 0}

/-- The set of integers in S -/
def integerS : Set ℝ := {x ∈ S | ∃ n : ℤ, x = n}

/-- The set of rational numbers in S -/
def rationalS : Set ℝ := {x ∈ S | ∃ p q : ℤ, q ≠ 0 ∧ x = p / q}

theorem set_classification :
  positiveS = {0.3, Real.pi/3, 22/7} ∧
  negativeS = {-2, -3.14, -0.1212212221} ∧
  integerS = {-2, 0} ∧
  rationalS = {-2, 0, 0.3, 22/7} := by
  sorry

end NUMINAMATH_CALUDE_set_classification_l920_92091


namespace NUMINAMATH_CALUDE_remaining_oil_after_350km_distance_when_8_liters_left_l920_92021

-- Define the initial conditions
def initial_oil : ℝ := 56
def oil_consumption_rate : ℝ := 0.08

-- Define the relationship between remaining oil and distance traveled
def remaining_oil (x : ℝ) : ℝ := initial_oil - oil_consumption_rate * x

-- Theorem to prove the remaining oil after 350 km
theorem remaining_oil_after_350km :
  remaining_oil 350 = 28 := by sorry

-- Theorem to prove the distance traveled when 8 liters are left
theorem distance_when_8_liters_left :
  ∃ x : ℝ, remaining_oil x = 8 ∧ x = 600 := by sorry

end NUMINAMATH_CALUDE_remaining_oil_after_350km_distance_when_8_liters_left_l920_92021


namespace NUMINAMATH_CALUDE_num_valid_colorings_is_7776_l920_92011

/-- A graph representing the extended figure described in the problem -/
def ExtendedFigureGraph : Type := Unit

/-- The number of vertices in the extended figure graph -/
def num_vertices : Nat := 12

/-- The number of available colors -/
def num_colors : Nat := 4

/-- A function that determines if two vertices are adjacent in the extended figure graph -/
def are_adjacent (v1 v2 : Fin num_vertices) : Bool := sorry

/-- A coloring of the graph is a function from vertices to colors -/
def Coloring := Fin num_vertices → Fin num_colors

/-- A predicate that determines if a coloring is valid (no adjacent vertices have the same color) -/
def is_valid_coloring (c : Coloring) : Prop :=
  ∀ v1 v2 : Fin num_vertices, are_adjacent v1 v2 → c v1 ≠ c v2

/-- The number of valid colorings for the extended figure graph -/
def num_valid_colorings : Nat := sorry

/-- The main theorem stating that the number of valid colorings is 7776 -/
theorem num_valid_colorings_is_7776 : num_valid_colorings = 7776 := by sorry

end NUMINAMATH_CALUDE_num_valid_colorings_is_7776_l920_92011


namespace NUMINAMATH_CALUDE_triangle_problem_l920_92072

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem statement --/
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * Real.sin t.B * Real.sin t.C + Real.cos t.B + 2 * Real.cos (t.B + t.C) = 0)
  (h2 : Real.sin t.B ≠ 1)
  (h3 : 5 * Real.sin t.B = 3 * Real.sin t.A)
  (h4 : (1/2) * t.a * t.b * Real.sin t.C = 15 * Real.sqrt 3 / 4) :
  t.C = 2 * Real.pi / 3 ∧ t.a + t.b + t.c = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l920_92072


namespace NUMINAMATH_CALUDE_distribution_count_7_l920_92097

/-- The number of ways to distribute n distinct objects into 3 distinct containers
    labeled 1, 2, and 3, such that each container has at least as many objects as its label -/
def distribution_count (n : ℕ) : ℕ :=
  let ways_221 := (n.choose 2) * ((n - 2).choose 2)
  let ways_133 := (n.choose 1) * ((n - 1).choose 3)
  let ways_124 := (n.choose 1) * ((n - 1).choose 2)
  ways_221 + ways_133 + ways_124

/-- Theorem stating that there are 455 ways to distribute 7 distinct objects
    into 3 distinct containers with the given constraints -/
theorem distribution_count_7 : distribution_count 7 = 455 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_7_l920_92097


namespace NUMINAMATH_CALUDE_tyler_double_flips_l920_92005

/-- Represents the number of flips in a single move for each gymnast -/
def triple_flip : ℕ := 3
def double_flip : ℕ := 2

/-- Represents the number of triple-flips Jen performed -/
def jen_triple_flips : ℕ := 16

/-- Calculates the total number of flips Jen performed -/
def jen_total_flips : ℕ := jen_triple_flips * triple_flip

/-- Calculates the total number of flips Tyler performed -/
def tyler_total_flips : ℕ := jen_total_flips / 2

/-- Theorem: Given the conditions, Tyler performed 12 double-flips -/
theorem tyler_double_flips : tyler_total_flips / double_flip = 12 := by
  sorry

end NUMINAMATH_CALUDE_tyler_double_flips_l920_92005


namespace NUMINAMATH_CALUDE_square_difference_problem_l920_92030

theorem square_difference_problem (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) :
  |x^2 - y^2| = 108 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_problem_l920_92030


namespace NUMINAMATH_CALUDE_cut_cube_theorem_l920_92057

/-- Given a cube cut into equal smaller cubes, this function calculates
    the total number of smaller cubes created. -/
def total_smaller_cubes (n : ℕ) : ℕ := (n + 1)^3

/-- This function calculates the number of smaller cubes painted on exactly 2 faces. -/
def cubes_with_two_painted_faces (n : ℕ) : ℕ := 12 * (n - 1)

/-- Theorem stating that when a cube is cut such that 12 smaller cubes are painted
    on exactly 2 faces, the total number of smaller cubes is 27. -/
theorem cut_cube_theorem :
  ∃ n : ℕ, cubes_with_two_painted_faces n = 12 ∧ total_smaller_cubes n = 27 :=
sorry

end NUMINAMATH_CALUDE_cut_cube_theorem_l920_92057


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l920_92001

def A : Set ℝ := {-2, -1, 0, 1}
def B : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l920_92001


namespace NUMINAMATH_CALUDE_marys_story_characters_l920_92012

theorem marys_story_characters (total : ℕ) (a c d e : ℕ) : 
  total = 60 →
  a = total / 2 →
  c = a / 2 →
  d + e = total - a - c →
  d = 2 * e →
  d = 10 := by
  sorry

end NUMINAMATH_CALUDE_marys_story_characters_l920_92012


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_one_fourth_l920_92080

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_one_fourth : arithmetic_sqrt (1/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_one_fourth_l920_92080


namespace NUMINAMATH_CALUDE_triangle_max_area_l920_92006

theorem triangle_max_area (a b c : ℝ) (A : ℝ) (h_a : a = 4) (h_A : A = π/3) :
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l920_92006


namespace NUMINAMATH_CALUDE_A_intersect_B_l920_92055

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ m : ℕ, x = 2 * m}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l920_92055


namespace NUMINAMATH_CALUDE_double_dimensions_cylinder_l920_92008

/-- A cylindrical container with original volume and new volume after doubling dimensions -/
structure Container where
  originalVolume : ℝ
  newVolume : ℝ

/-- The volume of a cylinder doubles when its radius is doubled -/
def volumeDoubledRadius (v : ℝ) : ℝ := 4 * v

/-- The volume of a cylinder doubles when its height is doubled -/
def volumeDoubledHeight (v : ℝ) : ℝ := 2 * v

/-- Theorem: Doubling all dimensions of a 5-gallon cylindrical container results in a 40-gallon container -/
theorem double_dimensions_cylinder (c : Container) 
  (h₁ : c.originalVolume = 5)
  (h₂ : c.newVolume = volumeDoubledHeight (volumeDoubledRadius c.originalVolume)) :
  c.newVolume = 40 := by
  sorry

#check double_dimensions_cylinder

end NUMINAMATH_CALUDE_double_dimensions_cylinder_l920_92008


namespace NUMINAMATH_CALUDE_min_value_a_minus_2b_l920_92087

/-- A quadratic function f(x) = x^2 - ax + b with one root in [-1, 1] and another in [1, 2] -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  root_in_neg_one_to_one : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ x^2 - a*x + b = 0
  root_in_one_to_two : ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 - a*x + b = 0

/-- The minimum value of a - 2b for the given quadratic function is -1 -/
theorem min_value_a_minus_2b (f : QuadraticFunction) :
  ∃ m : ℝ, m = -1 ∧ ∀ x : ℝ, f.a - 2*f.b ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_a_minus_2b_l920_92087


namespace NUMINAMATH_CALUDE_optimal_purchase_plan_l920_92040

-- Define the types and constants
def UnitPriceA : ℝ → ℝ := λ x => x + 800
def UnitPriceB : ℝ := 2400
def SellingPriceA : ℝ := 3700
def SellingPriceB : ℝ := 2700
def Budget : ℝ := 28000

-- Define the conditions
axiom price_difference : UnitPriceA UnitPriceB = UnitPriceB + 800
axiom quantity_equality : 38400 / (UnitPriceA UnitPriceB) = 28800 / UnitPriceB

-- Define the purchase plan type
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

-- Define the profit function
def profit : PurchasePlan → ℝ
  | ⟨a, b⟩ => (SellingPriceA - UnitPriceA UnitPriceB) * a + (SellingPriceB - UnitPriceB) * b

-- Define the theorem to be proved
theorem optimal_purchase_plan :
  ∃ (plan : PurchasePlan),
    (UnitPriceA UnitPriceB * plan.typeA + UnitPriceB * plan.typeB = Budget) ∧
    (∀ (other : PurchasePlan),
      (UnitPriceA UnitPriceB * other.typeA + UnitPriceB * other.typeB = Budget) →
      profit plan ≥ profit other) ∧
    plan.typeA = 8 ∧
    plan.typeB = 1 ∧
    profit plan = 4300 :=
  sorry

end NUMINAMATH_CALUDE_optimal_purchase_plan_l920_92040


namespace NUMINAMATH_CALUDE_trajectory_and_circle_properties_l920_92079

-- Define the vectors a and b
def a (m x y : ℝ) : ℝ × ℝ := (m * x, y + 1)
def b (x y : ℝ) : ℝ × ℝ := (x, y - 1)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the perpendicularity condition
def perpendicular (m x y : ℝ) : Prop := dot_product (a m x y) (b x y) = 0

-- Define the equation of trajectory E
def trajectory_equation (m x y : ℝ) : Prop := m * x^2 + y^2 = 1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4/5

-- Define a tangent line to the circle
def tangent_line (k t x y : ℝ) : Prop := y = k * x + t

-- Define the perpendicularity condition for OA and OB
def OA_perp_OB (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem
theorem trajectory_and_circle_properties (m : ℝ) :
  (∀ x y : ℝ, perpendicular m x y → trajectory_equation m x y) ∧
  (m = 1/4 →
    ∃ k t x1 y1 x2 y2 : ℝ,
      tangent_line k t x1 y1 ∧
      tangent_line k t x2 y2 ∧
      trajectory_equation m x1 y1 ∧
      trajectory_equation m x2 y2 ∧
      circle_equation x1 y1 ∧
      circle_equation x2 y2 ∧
      OA_perp_OB x1 y1 x2 y2) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_circle_properties_l920_92079


namespace NUMINAMATH_CALUDE_parallelepiped_with_surroundings_volume_l920_92071

/-- The volume of a set consisting of a rectangular parallelepiped and its surrounding elements -/
theorem parallelepiped_with_surroundings_volume 
  (l w h : ℝ) 
  (hl : l = 2) 
  (hw : w = 3) 
  (hh : h = 6) 
  (r : ℝ) 
  (hr : r = 1) : 
  (l * w * h) + 
  (2 * (r * w * h + r * l * h + r * l * w)) + 
  (π * r^2 * (l + w + h)) + 
  (2 * π * r^3) = 
  108 + (41/3) * π := by sorry

end NUMINAMATH_CALUDE_parallelepiped_with_surroundings_volume_l920_92071


namespace NUMINAMATH_CALUDE_bell_pepper_ratio_l920_92069

/-- Represents the number of bell peppers --/
def num_peppers : ℕ := 5

/-- Represents the number of large slices per bell pepper --/
def slices_per_pepper : ℕ := 20

/-- Represents the total number of slices and pieces in the meal --/
def total_pieces : ℕ := 200

/-- Represents the number of smaller pieces each large slice is cut into --/
def pieces_per_slice : ℕ := 3

/-- Calculates the total number of large slices --/
def total_large_slices : ℕ := num_peppers * slices_per_pepper

/-- Theorem stating the ratio of large slices cut into smaller pieces to total large slices --/
theorem bell_pepper_ratio : 
  ∃ (x : ℕ), x * pieces_per_slice + (total_large_slices - x) = total_pieces ∧ 
             x = 33 ∧
             (x : ℚ) / (total_large_slices : ℚ) = 33 / 100 := by
  sorry

end NUMINAMATH_CALUDE_bell_pepper_ratio_l920_92069


namespace NUMINAMATH_CALUDE_square_overlap_area_l920_92025

/-- The area of overlapping regions in a rectangle with four squares -/
theorem square_overlap_area (total_square_area sum_individual_areas uncovered_area : ℝ) :
  total_square_area = 27.5 ∧ 
  sum_individual_areas = 30 ∧ 
  uncovered_area = 1.5 →
  sum_individual_areas - total_square_area + uncovered_area = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_overlap_area_l920_92025


namespace NUMINAMATH_CALUDE_temperature_difference_l920_92035

theorem temperature_difference (highest lowest : ℤ) (h1 : highest = 12) (h2 : lowest = -1) :
  highest - lowest = 13 := by sorry

end NUMINAMATH_CALUDE_temperature_difference_l920_92035


namespace NUMINAMATH_CALUDE_sand_weight_formula_l920_92049

/-- Given a number of bags n, where each full bag contains 65 pounds of sand,
    and one bag is not full containing 42 pounds of sand,
    the total weight of sand W is (n-1) * 65 + 42 pounds. -/
theorem sand_weight_formula (n : ℕ) (W : ℕ) : W = (n - 1) * 65 + 42 :=
by sorry

end NUMINAMATH_CALUDE_sand_weight_formula_l920_92049


namespace NUMINAMATH_CALUDE_binomial_30_choose_3_l920_92019

theorem binomial_30_choose_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_choose_3_l920_92019


namespace NUMINAMATH_CALUDE_selection_schemes_l920_92042

theorem selection_schemes (num_boys num_girls : ℕ) (h1 : num_boys = 4) (h2 : num_girls = 2) :
  (num_boys : ℕ) * (num_girls : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_selection_schemes_l920_92042


namespace NUMINAMATH_CALUDE_doubling_function_m_range_l920_92058

/-- A function f is a doubling function if there exists an interval [a, b] such that f([a, b]) = [2a, 2b] -/
def DoublingFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧ Set.image f (Set.Icc a b) = Set.Icc (2*a) (2*b)

/-- The main theorem stating that if ln(e^x + m) is a doubling function, then m is in the open interval (-1/4, 0) -/
theorem doubling_function_m_range :
  ∀ m : ℝ, DoublingFunction (fun x ↦ Real.log (Real.exp x + m)) → m ∈ Set.Ioo (-1/4) 0 :=
by sorry

end NUMINAMATH_CALUDE_doubling_function_m_range_l920_92058


namespace NUMINAMATH_CALUDE_negation_square_positive_negation_root_equation_negation_sum_positive_negation_prime_odd_l920_92075

-- 1. The square of every natural number is positive.
theorem negation_square_positive : 
  (∀ n : ℕ, n^2 > 0) ↔ ¬(∃ n : ℕ, ¬(n^2 > 0)) :=
by sorry

-- 2. Every real number x is a root of the equation 5x-12=0.
theorem negation_root_equation : 
  (∀ x : ℝ, 5*x - 12 = 0) ↔ ¬(∃ x : ℝ, 5*x - 12 ≠ 0) :=
by sorry

-- 3. For every real number x, there exists a real number y such that x+y>0.
theorem negation_sum_positive : 
  (∀ x : ℝ, ∃ y : ℝ, x + y > 0) ↔ ¬(∃ x : ℝ, ∀ y : ℝ, x + y ≤ 0) :=
by sorry

-- 4. Some prime numbers are odd.
theorem negation_prime_odd : 
  (∃ p : ℕ, Prime p ∧ Odd p) ↔ ¬(∀ p : ℕ, Prime p → ¬Odd p) :=
by sorry

end NUMINAMATH_CALUDE_negation_square_positive_negation_root_equation_negation_sum_positive_negation_prime_odd_l920_92075


namespace NUMINAMATH_CALUDE_sum_and_square_difference_l920_92062

theorem sum_and_square_difference (x y : ℝ) : 
  x + y = 15 → x^2 - y^2 = 150 → x - y = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_and_square_difference_l920_92062


namespace NUMINAMATH_CALUDE_density_difference_of_cubes_l920_92037

theorem density_difference_of_cubes (m₁ : ℝ) (a₁ : ℝ) (m₁_pos : m₁ > 0) (a₁_pos : a₁ > 0) :
  let m₂ := 0.75 * m₁
  let a₂ := 1.25 * a₁
  let ρ₁ := m₁ / (a₁^3)
  let ρ₂ := m₂ / (a₂^3)
  (ρ₁ - ρ₂) / ρ₁ = 0.616 := by
sorry

end NUMINAMATH_CALUDE_density_difference_of_cubes_l920_92037


namespace NUMINAMATH_CALUDE_simplify_expression_l920_92096

theorem simplify_expression (a : ℝ) : (a + 4) * (a - 4) - (a - 1)^2 = 2 * a - 17 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l920_92096


namespace NUMINAMATH_CALUDE_cos_pi_third_plus_alpha_l920_92078

theorem cos_pi_third_plus_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (π / 3 + α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_plus_alpha_l920_92078


namespace NUMINAMATH_CALUDE_solve_equation_l920_92024

theorem solve_equation : ∃ x : ℝ, (5 - x = 8) ∧ (x = -3) := by sorry

end NUMINAMATH_CALUDE_solve_equation_l920_92024


namespace NUMINAMATH_CALUDE_polynomial_simplification_l920_92018

theorem polynomial_simplification (x : ℝ) :
  2 * x^2 * (4 * x^3 - 3 * x + 1) - 7 * (x^3 - 3 * x^2 + 2 * x - 8) =
  8 * x^5 - 13 * x^3 + 23 * x^2 - 14 * x + 56 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l920_92018


namespace NUMINAMATH_CALUDE_weekly_earnings_is_1454000_l920_92068

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of computers produced on a given day -/
def production_rate (d : Day) : ℕ :=
  match d with
  | Day.Monday    => 1200
  | Day.Tuesday   => 1500
  | Day.Wednesday => 1800
  | Day.Thursday  => 1600
  | Day.Friday    => 1400
  | Day.Saturday  => 1000
  | Day.Sunday    => 800

/-- Returns the selling price per computer on a given day -/
def selling_price (d : Day) : ℕ :=
  match d with
  | Day.Monday    => 150
  | Day.Tuesday   => 160
  | Day.Wednesday => 170
  | Day.Thursday  => 155
  | Day.Friday    => 145
  | Day.Saturday  => 165
  | Day.Sunday    => 140

/-- Calculates the earnings for a given day -/
def daily_earnings (d : Day) : ℕ :=
  production_rate d * selling_price d

/-- Calculates the total earnings for the week -/
def total_weekly_earnings : ℕ :=
  daily_earnings Day.Monday +
  daily_earnings Day.Tuesday +
  daily_earnings Day.Wednesday +
  daily_earnings Day.Thursday +
  daily_earnings Day.Friday +
  daily_earnings Day.Saturday +
  daily_earnings Day.Sunday

/-- Theorem stating that the total weekly earnings is $1,454,000 -/
theorem weekly_earnings_is_1454000 :
  total_weekly_earnings = 1454000 := by
  sorry

end NUMINAMATH_CALUDE_weekly_earnings_is_1454000_l920_92068


namespace NUMINAMATH_CALUDE_problem_solution_l920_92064

theorem problem_solution (x : ℝ) : (0.25 * x = 0.15 * 1500 - 30) → x = 780 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l920_92064


namespace NUMINAMATH_CALUDE_special_triangle_angles_l920_92010

/-- A triangle with a special property -/
structure SpecialTriangle where
  /-- The angle at vertex A -/
  angle_a : Real
  /-- The angle at vertex B -/
  angle_b : Real
  /-- The angle at vertex C -/
  angle_c : Real
  /-- The sum of angles is 180° -/
  angle_sum : angle_a + angle_b + angle_c = Real.pi
  /-- The altitude, angle bisector, and median from vertex A divide the angle into four equal parts -/
  special_property : ∃ (α : Real), angle_a = 4 * α ∧ 0 < α ∧ α < Real.pi / 2

/-- The theorem stating the angles of the special triangle -/
theorem special_triangle_angles (t : SpecialTriangle) :
  t.angle_a = Real.pi / 2 ∧ 
  t.angle_b = Real.pi / 8 ∧ 
  t.angle_c = 3 * Real.pi / 8 := by
  sorry

#check special_triangle_angles

end NUMINAMATH_CALUDE_special_triangle_angles_l920_92010


namespace NUMINAMATH_CALUDE_divisibility_property_l920_92086

theorem divisibility_property (a b c d : ℤ) (h1 : a ≠ b) (h2 : (a - b) ∣ (a * c + b * d)) :
  (a - b) ∣ (a * d + b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l920_92086


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l920_92020

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((2*a + b + c)^2) / (2*a^2 + (b + c)^2) +
  ((2*b + c + a)^2) / (2*b^2 + (c + a)^2) +
  ((2*c + a + b)^2) / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l920_92020


namespace NUMINAMATH_CALUDE_no_divisible_by_four_exists_l920_92013

theorem no_divisible_by_four_exists : 
  ¬ ∃ (B : ℕ), B < 10 ∧ (8000000 + 100000 * B + 4000 + 635 + 1) % 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_divisible_by_four_exists_l920_92013


namespace NUMINAMATH_CALUDE_total_spent_correct_l920_92016

def regular_fee : ℝ := 150
def discount_rate : ℝ := 0.075
def tax_rate : ℝ := 0.06
def total_teachers : ℕ := 22
def special_diet_teachers : ℕ := 3
def regular_food_allowance : ℝ := 10
def special_food_allowance : ℝ := 15

def total_spent : ℝ :=
  let discounted_fee := regular_fee * (1 - discount_rate) * total_teachers
  let taxed_fee := discounted_fee * (1 + tax_rate)
  let food_allowance := regular_food_allowance * (total_teachers - special_diet_teachers) +
                        special_food_allowance * special_diet_teachers
  taxed_fee + food_allowance

theorem total_spent_correct :
  total_spent = 3470.65 := by sorry

end NUMINAMATH_CALUDE_total_spent_correct_l920_92016


namespace NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achieved_l920_92041

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 64) : x + y ≤ 8 := by
  sorry

theorem max_sum_achieved : ∃ (x y : ℤ), x^2 + y^2 = 64 ∧ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achieved_l920_92041


namespace NUMINAMATH_CALUDE_set_union_equality_implies_m_range_l920_92076

theorem set_union_equality_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
  let B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  A ∪ B = A → m ∈ Set.Icc (-3) 3 := by
  sorry

end NUMINAMATH_CALUDE_set_union_equality_implies_m_range_l920_92076


namespace NUMINAMATH_CALUDE_john_reading_probability_l920_92002

/-- Probability of John reading a book on Monday -/
def prob_read_monday : ℝ := 0.8

/-- Probability of John playing soccer on Tuesday -/
def prob_soccer_tuesday : ℝ := 0.5

/-- Independence of activities -/
axiom activities_independent : True

/-- John reads every day when he decides to play soccer on the previous day -/
axiom reads_after_soccer : True

/-- Probability of John reading a book on both Monday and Tuesday -/
def prob_read_both_days : ℝ := prob_read_monday * prob_soccer_tuesday * prob_read_monday

theorem john_reading_probability :
  prob_read_both_days = 0.32 :=
sorry

end NUMINAMATH_CALUDE_john_reading_probability_l920_92002


namespace NUMINAMATH_CALUDE_a_eq_3_sufficient_not_necessary_l920_92014

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line2D) : Prop :=
  l₁.a * l₂.b = l₂.a * l₁.b

/-- The first line: ax - 5y - 1 = 0 -/
def l₁ (a : ℝ) : Line2D :=
  { a := a, b := -5, c := -1 }

/-- The second line: 3x - (a+2)y + 4 = 0 -/
def l₂ (a : ℝ) : Line2D :=
  { a := 3, b := -(a+2), c := 4 }

/-- The statement that a = 3 is a sufficient but not necessary condition for l₁ ∥ l₂ -/
theorem a_eq_3_sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ 3 ∧ parallel (l₁ a) (l₂ a)) ∧
  (parallel (l₁ 3) (l₂ 3)) := by
  sorry

end NUMINAMATH_CALUDE_a_eq_3_sufficient_not_necessary_l920_92014


namespace NUMINAMATH_CALUDE_estimations_correct_l920_92034

/-- A function that performs rounding to the nearest hundred. -/
def roundToHundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

/-- The acceptable error margin for estimation. -/
def ε : ℚ := 100

/-- Theorem stating that the estimations are correct within the error margin. -/
theorem estimations_correct :
  let e1 := |212 + 384 - roundToHundred 212 - roundToHundred 384|
  let e2 := |903 - 497 - (roundToHundred 903 - roundToHundred 497)|
  let e3 := |206 + 3060 - roundToHundred 206 - roundToHundred 3060|
  let e4 := |523 + 386 - roundToHundred 523 - roundToHundred 386|
  (e1 ≤ ε) ∧ (e2 ≤ ε) ∧ (e3 ≤ ε) ∧ (e4 ≤ ε) := by
  sorry

end NUMINAMATH_CALUDE_estimations_correct_l920_92034


namespace NUMINAMATH_CALUDE_at_least_one_negative_l920_92070

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) :
  a < 0 ∨ b < 0 := by sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l920_92070


namespace NUMINAMATH_CALUDE_guests_per_table_l920_92099

theorem guests_per_table (tables : ℝ) (total_guests : ℕ) 
  (h1 : tables = 252.0) 
  (h2 : total_guests = 1008) : 
  (total_guests : ℝ) / tables = 4 := by
  sorry

end NUMINAMATH_CALUDE_guests_per_table_l920_92099


namespace NUMINAMATH_CALUDE_no_prime_solution_l920_92095

theorem no_prime_solution (p : ℕ) (hp : Prime p) : ¬(2^p + p ∣ 3^p + p) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l920_92095


namespace NUMINAMATH_CALUDE_accessory_percentage_l920_92027

def computer_cost : ℝ := 3000
def initial_money : ℝ := 3 * computer_cost
def money_left : ℝ := 2700

theorem accessory_percentage :
  let total_spent := initial_money - money_left
  let accessory_cost := total_spent - computer_cost
  (accessory_cost / computer_cost) * 100 = 110 := by sorry

end NUMINAMATH_CALUDE_accessory_percentage_l920_92027


namespace NUMINAMATH_CALUDE_area_of_square_C_is_144_l920_92074

-- Define squares A, B, and C
def square_A : Real → Real := λ s ↦ s * s
def square_B : Real → Real := λ t ↦ t * t
def square_C : Real → Real := λ u ↦ u * u

-- Define the perimeter function for squares
def perimeter (side : Real) : Real := 4 * side

-- Theorem statement
theorem area_of_square_C_is_144 
  (side_A : Real) 
  (side_B : Real) 
  (side_C : Real) 
  (h1 : perimeter side_A = 16) 
  (h2 : perimeter side_B = 32) 
  (h3 : side_C = side_A + side_B) : 
  square_C side_C = 144 := by
  sorry

end NUMINAMATH_CALUDE_area_of_square_C_is_144_l920_92074


namespace NUMINAMATH_CALUDE_park_area_is_102400_l920_92000

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  ratio : length = 4 * breadth

/-- Calculates the perimeter of the park -/
def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.breadth)

/-- Calculates the area of the park -/
def area (park : RectangularPark) : ℝ :=
  park.length * park.breadth

/-- Theorem: The area of the park is 102400 square meters -/
theorem park_area_is_102400 (park : RectangularPark) 
    (h_perimeter : perimeter park = 12 * 8 / 60 * 1000) : 
    area park = 102400 := by
  sorry


end NUMINAMATH_CALUDE_park_area_is_102400_l920_92000


namespace NUMINAMATH_CALUDE_hexagon_CF_length_l920_92054

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  square_side : ℝ
  other_side : ℝ
  h_square : square_side = 20
  h_other : other_side = 23
  h_square_ABDE : A.x = 0 ∧ A.y = 0 ∧ B.x = square_side ∧ B.y = 0 ∧
                  D.x = square_side ∧ D.y = square_side ∧ E.x = 0 ∧ E.y = square_side
  h_parallel : C.x = B.x ∧ F.x = A.x
  h_BC : (C.x - B.x)^2 + (C.y - B.y)^2 = other_side^2
  h_CD : (D.x - C.x)^2 + (D.y - C.y)^2 = other_side^2
  h_EF : (F.x - E.x)^2 + (F.y - E.y)^2 = other_side^2
  h_FA : (A.x - F.x)^2 + (A.y - F.y)^2 = other_side^2

/-- The theorem to be proved -/
theorem hexagon_CF_length (h : Hexagon) :
  ∃ n : ℕ, n = 28 ∧ n = ⌊Real.sqrt ((h.C.x - h.F.x)^2 + (h.C.y - h.F.y)^2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_hexagon_CF_length_l920_92054


namespace NUMINAMATH_CALUDE_complex_root_quadratic_l920_92052

theorem complex_root_quadratic (a : ℝ) : 
  (∃ x : ℂ, x^2 - 2*a*x + a^2 - 4*a + 6 = 0) ∧ 
  (Complex.I^2 = -1) ∧
  ((1 : ℂ) + Complex.I * Real.sqrt 2)^2 - 2*a*((1 : ℂ) + Complex.I * Real.sqrt 2) + a^2 - 4*a + 6 = 0
  → a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_root_quadratic_l920_92052


namespace NUMINAMATH_CALUDE_triangle_side_product_greater_than_circle_diameters_l920_92017

theorem triangle_side_product_greater_than_circle_diameters 
  (a b c r R : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ R > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_inradius : r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * (a + b + c)))
  (h_circumradius : R = a * b * c / (4 * (a + b - c) * (b + c - a) * (c + a - b))) :
  a * b > 4 * r * R :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_product_greater_than_circle_diameters_l920_92017


namespace NUMINAMATH_CALUDE_janes_mean_score_l920_92051

def janes_scores : List ℝ := [98, 97, 92, 85, 93]

theorem janes_mean_score :
  (janes_scores.sum / janes_scores.length : ℝ) = 93 := by
  sorry

end NUMINAMATH_CALUDE_janes_mean_score_l920_92051


namespace NUMINAMATH_CALUDE_plums_picked_total_l920_92077

/-- The number of plums Alyssa picked -/
def alyssas_plums : ℕ := 17

/-- The number of plums Jason picked -/
def jasons_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := alyssas_plums + jasons_plums

theorem plums_picked_total :
  total_plums = 27 := by sorry

end NUMINAMATH_CALUDE_plums_picked_total_l920_92077


namespace NUMINAMATH_CALUDE_quadratic_real_root_l920_92092

theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l920_92092


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l920_92033

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricPointXAxis (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

theorem symmetric_point_x_axis :
  let original := Point3D.mk (-2) 1 4
  symmetricPointXAxis original = Point3D.mk (-2) (-1) (-4) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l920_92033


namespace NUMINAMATH_CALUDE_red_squares_less_than_half_l920_92038

/-- Represents a cube with side length 3, composed of 27 unit cubes -/
structure LargeCube where
  total_units : Nat
  red_units : Nat
  blue_units : Nat
  side_length : Nat

/-- Calculates the total number of visible unit squares on the surface of the large cube -/
def total_surface_squares (cube : LargeCube) : Nat :=
  6 * (cube.side_length * cube.side_length)

/-- Calculates the maximum number of red unit squares that can be visible on the surface -/
def max_red_surface_squares (cube : LargeCube) : Nat :=
  (cube.side_length - 1) * (cube.side_length - 1) * 3 + (cube.side_length - 1) * 3 * 2 + 8 * 3

/-- Theorem stating that the maximum number of red squares on the surface is less than half the total -/
theorem red_squares_less_than_half (cube : LargeCube) 
  (h1 : cube.total_units = 27)
  (h2 : cube.red_units = 9)
  (h3 : cube.blue_units = 18)
  (h4 : cube.side_length = 3)
  : max_red_surface_squares cube < (total_surface_squares cube) / 2 := by
  sorry

end NUMINAMATH_CALUDE_red_squares_less_than_half_l920_92038


namespace NUMINAMATH_CALUDE_min_ellipse_area_l920_92082

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b

/-- A circle with center (h, 0) and radius 1 -/
structure Circle where
  h : ℝ

/-- The ellipse is tangent to the circle -/
def is_tangent (e : Ellipse) (c : Circle) : Prop :=
  ∃ x y : ℝ, (x^2 / e.a^2) + (y^2 / e.b^2) = 1 ∧ (x - c.h)^2 + y^2 = 1

/-- The theorem stating the minimum area of the ellipse -/
theorem min_ellipse_area (e : Ellipse) (c1 c2 : Circle) 
  (h1 : is_tangent e c1) (h2 : is_tangent e c2) (h3 : c1.h = 2) (h4 : c2.h = -2) :
  e.a * e.b * π ≥ (10 * Real.sqrt 15 / 3) * π :=
sorry

end NUMINAMATH_CALUDE_min_ellipse_area_l920_92082


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l920_92009

theorem arithmetic_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + q) →  -- arithmetic sequence with common ratio q
  (∀ n, S n = (n * (2 * a 1 + (n - 1) * q)) / 2) →  -- sum formula for arithmetic sequence
  S 2 = 3 * a 2 + 2 →
  S 4 = 3 * a 4 + 2 →
  q = -1 ∨ q = 3/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l920_92009


namespace NUMINAMATH_CALUDE_phone_price_is_3000_l920_92007

/-- Represents the payment plan for a phone purchase -/
structure PaymentPlan where
  initialPayment : ℕ
  monthlyPayment : ℕ
  duration : ℕ

/-- Calculates the total cost of a payment plan -/
def totalCost (plan : PaymentPlan) : ℕ :=
  plan.initialPayment + plan.monthlyPayment * (plan.duration - 1)

/-- Represents the two-part payment plan -/
structure TwoPartPlan where
  firstHalfPayment : ℕ
  secondHalfPayment : ℕ
  duration : ℕ

/-- Calculates the total cost of a two-part payment plan -/
def twoPartTotalCost (plan : TwoPartPlan) : ℕ :=
  (plan.firstHalfPayment * (plan.duration / 2)) + (plan.secondHalfPayment * (plan.duration / 2))

/-- The theorem stating that the phone price is 3000 yuan given the described payment plans -/
theorem phone_price_is_3000 (plan1 : PaymentPlan) (plan2 : TwoPartPlan) 
    (h1 : plan1.initialPayment = 800)
    (h2 : plan1.monthlyPayment = 200)
    (h3 : plan2.firstHalfPayment = 350)
    (h4 : plan2.secondHalfPayment = 150)
    (h5 : plan1.duration = plan2.duration)
    (h6 : totalCost plan1 = twoPartTotalCost plan2) :
    totalCost plan1 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_phone_price_is_3000_l920_92007


namespace NUMINAMATH_CALUDE_A_intersect_B_l920_92083

def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {0, 1, 2}

theorem A_intersect_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l920_92083


namespace NUMINAMATH_CALUDE_fathers_age_l920_92081

/-- Represents the ages of family members and proves the father's age -/
theorem fathers_age (total_age sister_age kaydence_age : ℕ) 
  (h1 : total_age = 200)
  (h2 : sister_age = 40)
  (h3 : kaydence_age = 12) :
  ∃ (father_age : ℕ),
    father_age = 60 ∧
    ∃ (mother_age brother_age : ℕ),
      mother_age = father_age - 2 ∧
      brother_age = father_age / 2 ∧
      father_age + mother_age + brother_age + sister_age + kaydence_age = total_age :=
by
  sorry


end NUMINAMATH_CALUDE_fathers_age_l920_92081


namespace NUMINAMATH_CALUDE_school_capacity_l920_92028

theorem school_capacity (total_capacity : ℕ) (known_school_capacity : ℕ) (num_schools : ℕ) (num_known_schools : ℕ) :
  total_capacity = 1480 →
  known_school_capacity = 400 →
  num_schools = 4 →
  num_known_schools = 2 →
  (total_capacity - num_known_schools * known_school_capacity) / (num_schools - num_known_schools) = 340 := by
  sorry

end NUMINAMATH_CALUDE_school_capacity_l920_92028


namespace NUMINAMATH_CALUDE_percentage_equality_l920_92039

theorem percentage_equality (x : ℝ) : (90 / 100 * 600 = 50 / 100 * x) → x = 1080 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l920_92039
