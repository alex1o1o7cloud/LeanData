import Mathlib

namespace NUMINAMATH_CALUDE_ratio_a_to_b_l4101_410134

theorem ratio_a_to_b (a b : ℚ) (h : (12*a - 5*b) / (17*a - 3*b) = 4/7) : a/b = 23/16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l4101_410134


namespace NUMINAMATH_CALUDE_unique_solution_for_n_l4101_410192

theorem unique_solution_for_n : ∃! (n : ℕ+), ∃ (x : ℕ+),
  n = 2^(2*x.val-1) - 5*x.val - 3 ∧
  n = (2^(x.val-1) - 1) * (2^x.val + 1) ∧
  n = 2015 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_n_l4101_410192


namespace NUMINAMATH_CALUDE_min_value_theorem_l4101_410165

theorem min_value_theorem (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  ∃ (min_val : ℝ), min_val = -9/8 ∧ ∀ (a b : ℝ), 2 * a^2 + 3 * a * b + 2 * b^2 = 1 →
    x + y + x * y ≥ min_val ∧ a + b + a * b ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4101_410165


namespace NUMINAMATH_CALUDE_student_marks_difference_l4101_410193

/-- Given a student's marks in physics, chemistry, and mathematics,
    prove that the total marks exceed the physics marks by 140,
    given that the average of chemistry and mathematics marks is 70. -/
theorem student_marks_difference 
  (P C M : ℕ)  -- Marks in Physics, Chemistry, and Mathematics
  (h_avg : (C + M) / 2 = 70)  -- Average of Chemistry and Mathematics is 70
  : (P + C + M) - P = 140 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_difference_l4101_410193


namespace NUMINAMATH_CALUDE_round_trips_per_day_l4101_410162

def apartment_stories : ℕ := 5
def story_height : ℕ := 10
def total_vertical_distance : ℕ := 2100
def days_in_week : ℕ := 7

theorem round_trips_per_day :
  ∃ (trips_per_day : ℕ),
    trips_per_day * days_in_week * 2 * apartment_stories * story_height = total_vertical_distance ∧
    trips_per_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_round_trips_per_day_l4101_410162


namespace NUMINAMATH_CALUDE_problem_statement_l4101_410153

theorem problem_statement (x y : ℝ) (h1 : x = 3) (h2 : y = 1) :
  let n := x - y^(2*(x+y))
  n = 2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l4101_410153


namespace NUMINAMATH_CALUDE_geometric_ratio_is_four_l4101_410185

/-- An arithmetic sequence with a_1 = 2 and non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d

/-- Three terms of an arithmetic sequence form a geometric sequence -/
def forms_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ a 3 = a 1 * q ∧ a 11 = a 1 * q^2

/-- The common ratio of the geometric sequence formed by a_1, a_3, and a_11 is 4 -/
theorem geometric_ratio_is_four
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : forms_geometric_sequence a) :
  ∃ q : ℝ, q = 4 ∧ a 3 = a 1 * q ∧ a 11 = a 1 * q^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_ratio_is_four_l4101_410185


namespace NUMINAMATH_CALUDE_power_of_i_2023_l4101_410177

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_2023 : i^2023 = -i := by sorry

end NUMINAMATH_CALUDE_power_of_i_2023_l4101_410177


namespace NUMINAMATH_CALUDE_reservoir_water_ratio_l4101_410198

theorem reservoir_water_ratio (current_amount : ℝ) (total_capacity : ℝ) (normal_level : ℝ)
  (h1 : current_amount = 14000000)
  (h2 : current_amount = 0.7 * total_capacity)
  (h3 : normal_level = total_capacity - 10000000) :
  current_amount / normal_level = 1.4 := by
sorry

end NUMINAMATH_CALUDE_reservoir_water_ratio_l4101_410198


namespace NUMINAMATH_CALUDE_total_tiles_is_183_l4101_410106

/-- Calculates the number of tiles needed for a room with given dimensions and tile specifications. -/
def calculate_tiles (room_length room_width border_width : ℕ) 
  (border_tile_size inner_tile_size : ℕ) : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let border_tiles := 2 * (room_length + room_width) * (border_width / border_tile_size) +
                      4 * (border_width / border_tile_size) ^ 2
  let inner_tiles := (inner_length * inner_width) / (inner_tile_size ^ 2)
  border_tiles + inner_tiles

/-- Theorem stating that the total number of tiles for the given room specifications is 183. -/
theorem total_tiles_is_183 :
  calculate_tiles 24 18 2 1 3 = 183 := by sorry

end NUMINAMATH_CALUDE_total_tiles_is_183_l4101_410106


namespace NUMINAMATH_CALUDE_parabola_perpendicular_chords_locus_l4101_410180

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a chord of the parabola -/
structure Chord where
  slope : ℝ

/-- The locus of point M -/
def locus (p : ℝ) (x y : ℝ) : Prop :=
  (x - 2*p)^2 + y^2 = 4*p^2

theorem parabola_perpendicular_chords_locus 
  (para : Parabola) 
  (chord1 chord2 : Chord) 
  (O M : Point) :
  O.x = 0 ∧ O.y = 0 ∧  -- Vertex O at origin
  (chord1.slope * chord2.slope = -1) →  -- Perpendicular chords
  locus para.p M.x M.y  -- Locus of projection M
  := by sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_chords_locus_l4101_410180


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l4101_410140

/-- Given a group of children with various emotional states and genders, 
    prove the number of boys who are neither happy nor sad. -/
theorem boys_neither_happy_nor_sad 
  (total_children : ℕ) 
  (happy_children sad_children confused_children excited_children neither_happy_nor_sad : ℕ)
  (total_boys total_girls : ℕ)
  (happy_boys sad_girls confused_boys excited_girls : ℕ)
  (h1 : total_children = 80)
  (h2 : happy_children = 35)
  (h3 : sad_children = 15)
  (h4 : confused_children = 10)
  (h5 : excited_children = 5)
  (h6 : neither_happy_nor_sad = 15)
  (h7 : total_boys = 45)
  (h8 : total_girls = 35)
  (h9 : happy_boys = 8)
  (h10 : sad_girls = 7)
  (h11 : confused_boys = 4)
  (h12 : excited_girls = 3)
  (h13 : total_children = happy_children + sad_children + confused_children + excited_children + neither_happy_nor_sad)
  (h14 : total_children = total_boys + total_girls) :
  total_boys - (happy_boys + (sad_children - sad_girls) + confused_boys + (excited_children - excited_girls)) = 23 :=
by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l4101_410140


namespace NUMINAMATH_CALUDE_equation_solutions_l4101_410157

theorem equation_solutions :
  (∃ x : ℚ, 8 * x = -2 * (x + 5) ∧ x = -1) ∧
  (∃ x : ℚ, (x - 1) / 4 = (5 * x - 7) / 6 + 1 ∧ x = -1 / 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4101_410157


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l4101_410183

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a3 : a 3 = 4)
  (h_a5 : a 5 = m)
  (h_a7 : a 7 = 16) :
  m = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l4101_410183


namespace NUMINAMATH_CALUDE_only_taller_students_not_set_l4101_410128

-- Define the options
inductive SetOption
  | PrimesUpTo20
  | RootsOfEquation
  | TallerStudents
  | AllSquares

-- Define a predicate for well-defined sets
def is_well_defined_set (option : SetOption) : Prop :=
  match option with
  | SetOption.PrimesUpTo20 => true
  | SetOption.RootsOfEquation => true
  | SetOption.TallerStudents => false
  | SetOption.AllSquares => true

-- Theorem statement
theorem only_taller_students_not_set :
  ∀ (option : SetOption),
    ¬(is_well_defined_set option) ↔ option = SetOption.TallerStudents :=
by sorry

end NUMINAMATH_CALUDE_only_taller_students_not_set_l4101_410128


namespace NUMINAMATH_CALUDE_moon_distance_scientific_notation_l4101_410119

/-- The average distance between the Earth and the Moon in kilometers -/
def moon_distance : ℝ := 384000

/-- Theorem stating that the moon distance in scientific notation is 3.84 × 10^5 -/
theorem moon_distance_scientific_notation : moon_distance = 3.84 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_moon_distance_scientific_notation_l4101_410119


namespace NUMINAMATH_CALUDE_new_average_after_removal_l4101_410103

theorem new_average_after_removal (numbers : List ℝ) : 
  numbers.length = 12 → 
  numbers.sum / numbers.length = 90 → 
  80 ∈ numbers → 
  84 ∈ numbers → 
  let remaining := numbers.filter (λ x => x ≠ 80 ∧ x ≠ 84)
  remaining.sum / remaining.length = 91.6 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_removal_l4101_410103


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_incorrect_l4101_410131

theorem quadratic_inequality_condition_incorrect 
  (a b c : ℝ) (h1 : a < 0) (h2 : b^2 - 4*a*c ≤ 0) :
  ¬ (∀ x : ℝ, a*x^2 + b*x + c ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_incorrect_l4101_410131


namespace NUMINAMATH_CALUDE_log3_graph_properties_l4101_410184

-- Define the logarithm function base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the graph of y = log₃(x)
def graph_log3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = log3 p.1 ∧ p.1 > 0}

-- Define the x-axis and y-axis
def x_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}
def y_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

-- Theorem statement
theorem log3_graph_properties :
  (∃ p, p ∈ graph_log3 ∩ x_axis) ∧
  (graph_log3 ∩ y_axis = ∅) :=
by sorry

end NUMINAMATH_CALUDE_log3_graph_properties_l4101_410184


namespace NUMINAMATH_CALUDE_log_simplification_l4101_410125

-- Define variables
variable (a b c d x y : ℝ)
-- Assume all variables are positive to ensure logarithms are defined
variable (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hx : x > 0) (hy : y > 0)

-- Define the theorem
theorem log_simplification :
  Real.log (2*a/(3*b)) + Real.log (5*b/(4*c)) + Real.log (6*c/(7*d)) - Real.log (20*a*y/(21*d*x)) = Real.log (3*x/(4*y)) :=
by sorry

end NUMINAMATH_CALUDE_log_simplification_l4101_410125


namespace NUMINAMATH_CALUDE_visitors_in_scientific_notation_l4101_410197

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem visitors_in_scientific_notation :
  toScientificNotation 203000 = ScientificNotation.mk 2.03 5 sorry := by
  sorry

end NUMINAMATH_CALUDE_visitors_in_scientific_notation_l4101_410197


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l4101_410126

theorem contrapositive_equivalence (a b c d : ℝ) :
  ((a = b ∧ c = d) → (a + c = b + d)) ↔ ((a + c ≠ b + d) → (a ≠ b ∨ c ≠ d)) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l4101_410126


namespace NUMINAMATH_CALUDE_arithmetic_progression_properties_l4101_410173

-- Define the arithmetic progression
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

-- Define the geometric progression condition
def geometric_progression_condition (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, a 4 = a 2 * r ∧ a 8 = a 4 * r

-- Main theorem
theorem arithmetic_progression_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_progression a)
  (h_a1 : a 1 = 1)
  (h_geom : geometric_progression_condition a) :
  (∀ n : ℕ, a n = n) ∧
  (∀ n : ℕ, n ≤ 98 ↔ 100 * (1 - 1 / (n + 1)) < 99) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_properties_l4101_410173


namespace NUMINAMATH_CALUDE_star_comm_star_assoc_star_disprove_l4101_410187

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem 1: Commutativity of *
theorem star_comm (a b : ℝ) : star a b = star b a := by sorry

-- Theorem 2: Associativity of *
theorem star_assoc (a b c : ℝ) : star (star a b) c = star a (star b c) := by sorry

-- Theorem 3: Disprove the given property
theorem star_disprove : ¬(∀ (a b : ℝ), star (a + 1) b = star a b + star 1 b) := by sorry

end NUMINAMATH_CALUDE_star_comm_star_assoc_star_disprove_l4101_410187


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l4101_410107

open Real

theorem min_value_trigonometric_expression (θ : ℝ) (h : 0 < θ ∧ θ < π/2) :
  (4 * cos θ + 3 / sin θ + 2 * sqrt 2 * tan θ) ≥ 6 * sqrt 3 * (2 ^ (1/6)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l4101_410107


namespace NUMINAMATH_CALUDE_acid_concentration_problem_l4101_410138

theorem acid_concentration_problem (acid1 acid2 acid3 : ℝ) (water : ℝ) :
  acid1 = 10 →
  acid2 = 20 →
  acid3 = 30 →
  acid1 / (acid1 + (water * (1/20))) = 1/20 →
  acid2 / (acid2 + (water * (13/30))) = 7/30 →
  acid3 / (acid3 + water) = 21/200 :=
by sorry

end NUMINAMATH_CALUDE_acid_concentration_problem_l4101_410138


namespace NUMINAMATH_CALUDE_okinawa_sales_ratio_l4101_410137

/-- Proves the ratio of Okinawa-flavored milk tea sales to total sales -/
theorem okinawa_sales_ratio (total_sales : ℕ) (winter_melon_sales : ℕ) (chocolate_sales : ℕ) 
  (h1 : total_sales = 50)
  (h2 : winter_melon_sales = 2 * total_sales / 5)
  (h3 : chocolate_sales = 15)
  (h4 : winter_melon_sales + chocolate_sales + (total_sales - winter_melon_sales - chocolate_sales) = total_sales) :
  (total_sales - winter_melon_sales - chocolate_sales) / total_sales = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_okinawa_sales_ratio_l4101_410137


namespace NUMINAMATH_CALUDE_point_movement_l4101_410188

/-- Given a point P at (-1, 2), moving it 2 units left and 1 unit up results in point M at (-3, 3) -/
theorem point_movement :
  let P : ℝ × ℝ := (-1, 2)
  let M : ℝ × ℝ := (P.1 - 2, P.2 + 1)
  M = (-3, 3) := by sorry

end NUMINAMATH_CALUDE_point_movement_l4101_410188


namespace NUMINAMATH_CALUDE_star_difference_l4101_410171

-- Define the ⭐ operation
def star (x y : ℝ) : ℝ := x^2 * y - 3 * x + y

-- Theorem statement
theorem star_difference : star 3 5 - star 5 3 = -22 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l4101_410171


namespace NUMINAMATH_CALUDE_range_of_a_l4101_410181

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem range_of_a (a : ℝ) :
  (∀ x y, x < y → x < -1 → y < -1 → f a y < f a x) →
  (∀ x y, x < y → 1 < x → 1 < y → f a x < f a y) →
  a ∈ Set.Icc (-1) 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4101_410181


namespace NUMINAMATH_CALUDE_parabola_focus_l4101_410150

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x = (1/4) * y^2

/-- The focus of a parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola x = (1/4)y^2 is at (1, 0) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola_equation x y → 
  (∃ (p : ℝ × ℝ), p = focus ∧ 
   (x - p.1)^2 + (y - p.2)^2 = (x + p.1)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l4101_410150


namespace NUMINAMATH_CALUDE_sqrt_26_is_7th_term_l4101_410110

theorem sqrt_26_is_7th_term (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n = Real.sqrt (4 * n - 2)) →
  a 7 = Real.sqrt 26 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_26_is_7th_term_l4101_410110


namespace NUMINAMATH_CALUDE_exists_valid_division_l4101_410151

/-- A tiling of a 6x6 board with 2x1 dominos -/
def Tiling := Fin 6 → Fin 6 → Fin 18

/-- A division of the board into two rectangles -/
structure Division where
  horizontal : Bool
  position : Fin 6

/-- Checks if a domino crosses the dividing line -/
def crossesDivision (t : Tiling) (d : Division) : Prop :=
  ∃ (i j : Fin 6), 
    (d.horizontal ∧ i = d.position ∧ t i j = t (i + 1) j) ∨
    (¬d.horizontal ∧ j = d.position ∧ t i j = t i (j + 1))

/-- The main theorem -/
theorem exists_valid_division (t : Tiling) : 
  ∃ (d : Division), ¬crossesDivision t d := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_division_l4101_410151


namespace NUMINAMATH_CALUDE_chord_length_circle_line_l4101_410142

/-- The chord length cut by a circle from a line -/
theorem chord_length_circle_line (r : ℝ) (a b c : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | ∃ t, x = a + b*t ∧ y = c + t}
  let chord_length := 2 * Real.sqrt (r^2 - (a^2 + b^2 - 2*a*c + c^2) / (b^2 + 1))
  r = 3 ∧ a = 1 ∧ b = 2 ∧ c = 2 → chord_length = 12 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_chord_length_circle_line_l4101_410142


namespace NUMINAMATH_CALUDE_fuel_left_in_tank_l4101_410121

/-- Calculates the remaining fuel in a plane's tank given the fuel consumption rate and remaining flight time. -/
def remaining_fuel (fuel_rate : ℝ) (flight_time : ℝ) : ℝ :=
  fuel_rate * flight_time

/-- Proves that given a plane using fuel at a rate of 9.5 gallons per hour and can continue flying for 0.6667 hours, the amount of fuel left in the tank is approximately 6.33365 gallons. -/
theorem fuel_left_in_tank : 
  let fuel_rate := 9.5
  let flight_time := 0.6667
  abs (remaining_fuel fuel_rate flight_time - 6.33365) < 0.00001 := by
sorry

end NUMINAMATH_CALUDE_fuel_left_in_tank_l4101_410121


namespace NUMINAMATH_CALUDE_acute_angle_between_l1_l2_l4101_410132

/-- The acute angle formed by the intersection of two lines in a 2D plane. -/
def acuteAngleBetweenLines (l1 l2 : ℝ → ℝ → Prop) : ℝ := sorry

/-- Line l1: √3x - y + 1 = 0 -/
def l1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0

/-- Line l2: x + 5 = 0 -/
def l2 (x y : ℝ) : Prop := x + 5 = 0

/-- The acute angle formed by the intersection of l1 and l2 is 30° -/
theorem acute_angle_between_l1_l2 : acuteAngleBetweenLines l1 l2 = 30 * Real.pi / 180 := by sorry

end NUMINAMATH_CALUDE_acute_angle_between_l1_l2_l4101_410132


namespace NUMINAMATH_CALUDE_intersection_implies_difference_l4101_410135

def set_A (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = a * p.1 + 6}
def set_B : Set (ℝ × ℝ) := {p | p.2 = 5 * p.1 - 3}

theorem intersection_implies_difference (a b : ℝ) :
  (1, b) ∈ set_A a ∩ set_B → a - b = -6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_difference_l4101_410135


namespace NUMINAMATH_CALUDE_ellipse_parabola_problem_l4101_410129

/-- Given an ellipse and a parabola with specific properties, prove the equation of the ellipse,
    the coordinates of a point, and the range of a certain expression. -/
theorem ellipse_parabola_problem (a b p : ℝ) (F : ℝ × ℝ) :
  a > b ∧ b > 0 ∧ p > 0 ∧  -- Conditions on a, b, and p
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ y^2 = 2*p*x) ∧  -- C₁ and C₂ have a common point
  (F.1 - F.2 + 1)^2 / 2 = 2 ∧  -- Distance from F to x - y + 1 = 0 is √2
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ y^2 = 2*p*x ∧ (x - 3/2)^2 + y^2 = 6) →  -- Common chord length is 2√6
  (a^2 = 9 ∧ b^2 = 8 ∧ F = (1, 0)) ∧  -- Equation of C₁ and coordinates of F
  (∀ k : ℝ, k ≠ 0 → 
    1/6 < (21*k^2 + 8)/(48*(k^2 + 1)) ∧ (21*k^2 + 8)/(48*(k^2 + 1)) ≤ 7/16) -- Range of 1/|AB| + 1/|CD|
  := by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_problem_l4101_410129


namespace NUMINAMATH_CALUDE_chord_length_l4101_410136

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by sorry

end NUMINAMATH_CALUDE_chord_length_l4101_410136


namespace NUMINAMATH_CALUDE_no_real_solutions_l4101_410160

/-- Given a function f(x) = x^2 + 2x + a, where f(bx) = 9x^2 - 6x + 2,
    prove that the equation f(ax + b) = 0 has no real solutions. -/
theorem no_real_solutions (a b : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = x^2 + 2*x + a) ∧
   (∀ x, f (b*x) = 9*x^2 - 6*x + 2)) →
  (∀ x, (x^2 + 2*x + a) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l4101_410160


namespace NUMINAMATH_CALUDE_vector_equation_solution_l4101_410163

/-- Given vector a and an equation involving a and b, prove that b equals (1, -2) -/
theorem vector_equation_solution (a b : ℝ × ℝ) : 
  a = (1, 2) → 
  (2 • a) + b = (3, 2) → 
  b = (1, -2) := by
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l4101_410163


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_I_l4101_410166

def I : Finset ℕ := {1, 2, 3, 4, 5, 6}
def A : Finset ℕ := {1, 3, 4}

theorem complement_of_A_wrt_I :
  I \ A = {2, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_I_l4101_410166


namespace NUMINAMATH_CALUDE_ellipse_tangent_property_l4101_410112

/-- Ellipse passing through a point with specific tangent properties -/
theorem ellipse_tangent_property (m : ℝ) (r : ℝ) (h_m : m > 0) (h_r : r > 0) :
  (∃ (E F : ℝ × ℝ),
    -- E and F are on the ellipse
    (E.1^2 / 4 + E.2^2 / m = 1) ∧
    (F.1^2 / 4 + F.2^2 / m = 1) ∧
    -- A is on the ellipse
    (1^2 / 4 + (3/2)^2 / m = 1) ∧
    -- Slopes form arithmetic sequence
    (∃ (k : ℝ),
      (F.2 - 3/2) / (F.1 - 1) = k ∧
      (E.2 - 3/2) / (E.1 - 1) = -k ∧
      (F.2 - E.2) / (F.1 - E.1) = 3*k) ∧
    -- AE and AF are tangent to the circle
    ((1 - 2)^2 + (3/2 - 3/2)^2 = r^2)) →
  r = Real.sqrt 37 / 37 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_property_l4101_410112


namespace NUMINAMATH_CALUDE_log_equality_implies_p_q_equal_three_l4101_410117

theorem log_equality_implies_p_q_equal_three (p q : ℝ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) 
  (h_log : Real.log p + Real.log q = Real.log (2*p + q)) : 
  p = 3 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_p_q_equal_three_l4101_410117


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l4101_410139

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (cross_time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * cross_time_s

/-- Proof that a train's length is approximately 100.02 meters -/
theorem train_length_proof (speed_kmh : ℝ) (cross_time_s : ℝ) 
  (h1 : speed_kmh = 60) 
  (h2 : cross_time_s = 6) : 
  ∃ ε > 0, |train_length speed_kmh cross_time_s - 100.02| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l4101_410139


namespace NUMINAMATH_CALUDE_union_complement_equal_l4101_410168

def U : Set ℕ := {x | x < 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_complement_equal : B ∪ (U \ A) = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equal_l4101_410168


namespace NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l4101_410114

/-- The value of m for which the parabola y = 2x^2 + 3 is tangent to the hyperbola 4y^2 - mx^2 = 9 -/
def tangent_value : ℝ := 48

/-- The equation of the parabola -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2 + 3

/-- The equation of the hyperbola -/
def hyperbola (m x y : ℝ) : Prop := 4 * y^2 - m * x^2 = 9

/-- The parabola is tangent to the hyperbola when m equals the tangent_value -/
theorem parabola_tangent_hyperbola :
  ∃ (x y : ℝ), parabola x y ∧ hyperbola tangent_value x y ∧
  ∀ (x' y' : ℝ), parabola x' y' ∧ hyperbola tangent_value x' y' → (x', y') = (x, y) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_hyperbola_l4101_410114


namespace NUMINAMATH_CALUDE_dream_car_gas_consumption_l4101_410152

/-- Represents the gas consumption problem for Dream's car -/
theorem dream_car_gas_consumption 
  (gas_per_mile : ℝ) 
  (miles_today : ℝ) 
  (miles_tomorrow : ℝ) 
  (total_gas : ℝ) :
  miles_today = 400 →
  miles_tomorrow = miles_today + 200 →
  total_gas = 4000 →
  gas_per_mile * miles_today + gas_per_mile * miles_tomorrow = total_gas →
  gas_per_mile = 4 := by
sorry

end NUMINAMATH_CALUDE_dream_car_gas_consumption_l4101_410152


namespace NUMINAMATH_CALUDE_smaller_area_with_center_l4101_410144

/-- Represents a circular sector with a central angle of 60 degrees -/
structure Sector60 where
  radius : ℝ
  center : ℝ × ℝ

/-- Represents the line that cuts the sector -/
structure CuttingLine where
  slope : ℝ
  intercept : ℝ

/-- Represents the two parts after cutting the sector -/
structure SectorParts where
  part_with_center : Set (ℝ × ℝ)
  other_part : Set (ℝ × ℝ)

/-- Function to cut the sector -/
def cut_sector (s : Sector60) (l : CuttingLine) : SectorParts := sorry

/-- Function to calculate perimeter of a part -/
def perimeter (part : Set (ℝ × ℝ)) : ℝ := sorry

/-- Function to calculate area of a part -/
def area (part : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem smaller_area_with_center (s : Sector60) :
  ∃ (l : CuttingLine),
    let parts := cut_sector s l
    perimeter parts.part_with_center = perimeter parts.other_part →
    area parts.part_with_center < area parts.other_part :=
  sorry

end NUMINAMATH_CALUDE_smaller_area_with_center_l4101_410144


namespace NUMINAMATH_CALUDE_pure_imaginary_power_l4101_410158

theorem pure_imaginary_power (a : ℝ) (z : ℂ) : 
  z = a + (a + 1) * Complex.I → (z.im ≠ 0 ∧ z.re = 0) → z^2010 = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_power_l4101_410158


namespace NUMINAMATH_CALUDE_smallest_square_enclosing_circle_l4101_410100

theorem smallest_square_enclosing_circle (r : ℝ) (h : r = 5) : 
  (2 * r) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_enclosing_circle_l4101_410100


namespace NUMINAMATH_CALUDE_set_separation_iff_disjoint_l4101_410159

universe u

theorem set_separation_iff_disjoint {U : Type u} (A B : Set U) :
  (∃ C : Set U, A ⊆ C ∧ B ⊆ Cᶜ) ↔ A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_set_separation_iff_disjoint_l4101_410159


namespace NUMINAMATH_CALUDE_parabola_directrix_l4101_410123

/-- Represents a parabola with equation y = -4x^2 + 4 -/
structure Parabola where
  /-- The y-coordinate of the focus -/
  f : ℝ
  /-- The y-coordinate of the directrix -/
  d : ℝ

/-- Theorem: The directrix of the parabola y = -4x^2 + 4 is y = 65/16 -/
theorem parabola_directrix (p : Parabola) : p.d = 65/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4101_410123


namespace NUMINAMATH_CALUDE_apples_left_after_pie_l4101_410120

theorem apples_left_after_pie (initial_apples : Real) (anita_contribution : Real) (pie_requirement : Real) :
  initial_apples = 10.0 →
  anita_contribution = 5.0 →
  pie_requirement = 4.0 →
  initial_apples + anita_contribution - pie_requirement = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_after_pie_l4101_410120


namespace NUMINAMATH_CALUDE_smallest_n_for_array_formation_l4101_410191

theorem smallest_n_for_array_formation : 
  ∃ n k : ℕ+, 
    (∀ m k' : ℕ+, 8 * m = 225 * k' + 3 → n ≤ m) ∧ 
    (8 * n = 225 * k + 3) ∧
    n = 141 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_array_formation_l4101_410191


namespace NUMINAMATH_CALUDE_triangle_value_l4101_410146

theorem triangle_value (p : ℚ) (triangle : ℚ) 
  (eq1 : triangle * p + p = 72)
  (eq2 : (triangle * p + p) + p = 111) :
  triangle = 11 / 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_value_l4101_410146


namespace NUMINAMATH_CALUDE_short_trees_count_l4101_410164

/-- The number of short trees in the park after planting -/
def short_trees_after_planting (initial_short_trees planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + planted_short_trees

/-- Theorem: The number of short trees after planting is 95 -/
theorem short_trees_count : short_trees_after_planting 31 64 = 95 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_count_l4101_410164


namespace NUMINAMATH_CALUDE_vloggers_earnings_per_view_l4101_410194

/-- Represents the earnings and viewership of a vlogger -/
structure Vlogger where
  name : String
  daily_viewers : ℕ
  weekly_earnings : ℚ

/-- Calculates the earnings per view for a vlogger -/
def earnings_per_view (v : Vlogger) : ℚ :=
  v.weekly_earnings / (v.daily_viewers * 7)

theorem vloggers_earnings_per_view 
  (voltaire leila : Vlogger)
  (h1 : voltaire.daily_viewers = 50)
  (h2 : leila.daily_viewers = 2 * voltaire.daily_viewers)
  (h3 : leila.weekly_earnings = 350) :
  earnings_per_view voltaire = earnings_per_view leila ∧ 
  earnings_per_view voltaire = 1/2 := by
  sorry

#check vloggers_earnings_per_view

end NUMINAMATH_CALUDE_vloggers_earnings_per_view_l4101_410194


namespace NUMINAMATH_CALUDE_jamesFinalNumber_l4101_410115

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define Kyle's result
def kylesResult : ℕ := sumOfDigits (2014^2014)

-- Define Shannon's result
def shannonsResult : ℕ := sumOfDigits kylesResult

-- Theorem to prove
theorem jamesFinalNumber : sumOfDigits shannonsResult = 7 := by sorry

end NUMINAMATH_CALUDE_jamesFinalNumber_l4101_410115


namespace NUMINAMATH_CALUDE_probability_of_even_product_l4101_410169

-- Define the set of chips in each box
def chips : Set ℕ := {1, 2, 4}

-- Define the function to check if a number is even
def isEven (n : ℕ) : Prop := n % 2 = 0

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := 27

-- Define the number of favorable outcomes (even products)
def favorableOutcomes : ℕ := 26

-- Theorem statement
theorem probability_of_even_product :
  (favorableOutcomes : ℚ) / totalOutcomes = 26 / 27 := by sorry

end NUMINAMATH_CALUDE_probability_of_even_product_l4101_410169


namespace NUMINAMATH_CALUDE_line_through_hyperbola_points_l4101_410127

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := 2*x + 8*y + 7 = 0

/-- Theorem stating that the line passing through two points on the given hyperbola
    with midpoint (1/2, -1) has the equation 2x + 8y + 7 = 0 -/
theorem line_through_hyperbola_points :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧
    (x₁ + x₂)/2 = 1/2 ∧
    (y₁ + y₂)/2 = -1 ∧
    (∀ (x y : ℝ), (x - x₁)*(y₂ - y₁) = (y - y₁)*(x₂ - x₁) ↔ line x y) :=
by sorry

end NUMINAMATH_CALUDE_line_through_hyperbola_points_l4101_410127


namespace NUMINAMATH_CALUDE_indeterminate_equation_solutions_l4101_410195

theorem indeterminate_equation_solutions :
  ∀ x y : ℤ, 2 * (x + y) = x * y + 7 ↔ 
    (x = 3 ∧ y = -1) ∨ (x = 5 ∧ y = 1) ∨ (x = 1 ∧ y = 5) ∨ (x = -1 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_indeterminate_equation_solutions_l4101_410195


namespace NUMINAMATH_CALUDE_ceiling_square_count_l4101_410122

theorem ceiling_square_count (x : ℝ) (h : ⌈x⌉ = 15) : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ y : ℝ, ⌈y⌉ = 15 ∧ ⌈y^2⌉ = n) ∧ S.card = 29 :=
sorry

end NUMINAMATH_CALUDE_ceiling_square_count_l4101_410122


namespace NUMINAMATH_CALUDE_olympic_medals_l4101_410108

/-- Olympic Medals Theorem -/
theorem olympic_medals (china_total russia_total us_total : ℕ)
  (china_gold china_silver china_bronze : ℕ)
  (russia_gold russia_silver russia_bronze : ℕ)
  (us_gold us_silver us_bronze : ℕ)
  (h1 : china_total = 100)
  (h2 : russia_total = 72)
  (h3 : us_total = 110)
  (h4 : china_silver + china_bronze = russia_silver + russia_bronze)
  (h5 : russia_gold + 28 = china_gold)
  (h6 : us_gold = russia_gold + 13)
  (h7 : us_gold = us_bronze)
  (h8 : us_silver = us_gold + 2)
  (h9 : china_bronze = china_silver + 7)
  (h10 : china_total = china_gold + china_silver + china_bronze)
  (h11 : russia_total = russia_gold + russia_silver + russia_bronze)
  (h12 : us_total = us_gold + us_silver + us_bronze) :
  china_gold = 51 ∧ us_silver = 38 ∧ russia_bronze = 28 := by
  sorry


end NUMINAMATH_CALUDE_olympic_medals_l4101_410108


namespace NUMINAMATH_CALUDE_triangle_angles_theorem_l4101_410113

noncomputable def triangle_angles (a b c : ℝ) : ℝ × ℝ × ℝ := sorry

theorem triangle_angles_theorem :
  let side1 := 3
  let side2 := 3
  let side3 := Real.sqrt 8 - Real.sqrt 3
  let (angle_A, angle_B, angle_C) := triangle_angles side1 side2 side3
  angle_C = Real.arccos ((7 / 18) + (2 * Real.sqrt 6 / 9)) ∧
  angle_A = (π - angle_C) / 2 ∧
  angle_B = (π - angle_C) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_angles_theorem_l4101_410113


namespace NUMINAMATH_CALUDE_second_discount_percentage_l4101_410186

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (third_discount : ℝ) (final_price : ℝ) :
  original_price = 9795.3216374269 →
  first_discount = 20 →
  third_discount = 5 →
  final_price = 6700 →
  ∃ (second_discount : ℝ), 
    (original_price * (1 - first_discount / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) = final_price) ∧
    (abs (second_discount - 10) < 0.0000000001) := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l4101_410186


namespace NUMINAMATH_CALUDE_dogs_added_on_monday_l4101_410149

theorem dogs_added_on_monday
  (initial_dogs : ℕ)
  (sunday_dogs : ℕ)
  (total_dogs : ℕ)
  (h1 : initial_dogs = 2)
  (h2 : sunday_dogs = 5)
  (h3 : total_dogs = 10)
  : total_dogs - (initial_dogs + sunday_dogs) = 3 :=
by sorry

end NUMINAMATH_CALUDE_dogs_added_on_monday_l4101_410149


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_is_five_halves_l4101_410111

theorem sqrt_x_div_sqrt_y_is_five_halves (x y : ℝ) 
  (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*x)/(61*y)) : 
  Real.sqrt x / Real.sqrt y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_is_five_halves_l4101_410111


namespace NUMINAMATH_CALUDE_game_draw_fraction_l4101_410124

theorem game_draw_fraction (ben_win : ℚ) (tom_win : ℚ) (draw : ℚ) : 
  ben_win = 4/9 → tom_win = 1/3 → draw = 1 - (ben_win + tom_win) → draw = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_game_draw_fraction_l4101_410124


namespace NUMINAMATH_CALUDE_min_velocity_increase_is_6_l4101_410172

/-- Represents a car with its velocity -/
structure Car where
  velocity : ℝ

/-- Represents the road scenario -/
structure RoadScenario where
  carA : Car
  carB : Car
  carC : Car
  initialDistanceAB : ℝ
  initialDistanceAC : ℝ

/-- Calculates the minimum velocity increase needed for car A -/
def minVelocityIncrease (scenario : RoadScenario) : ℝ :=
  sorry

/-- Theorem stating the minimum velocity increase for the given scenario -/
theorem min_velocity_increase_is_6 (scenario : RoadScenario) 
  (h1 : scenario.carA.velocity > scenario.carB.velocity)
  (h2 : scenario.initialDistanceAB = 50)
  (h3 : scenario.initialDistanceAC = 300)
  (h4 : scenario.carB.velocity = 50)
  (h5 : scenario.carC.velocity = 70)
  (h6 : scenario.carA.velocity = 68) :
  minVelocityIncrease scenario = 6 :=
sorry

end NUMINAMATH_CALUDE_min_velocity_increase_is_6_l4101_410172


namespace NUMINAMATH_CALUDE_distinct_lines_theorem_l4101_410178

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to determine if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Function to create a line from two points -/
def line_from_points (p q : Point) : Line :=
  { a := q.y - p.y,
    b := p.x - q.x,
    c := p.y * q.x - p.x * q.y }

/-- Function to check if two lines are distinct -/
def distinct_lines (l1 l2 : Line) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a ∨ l1.a * l2.c ≠ l1.c * l2.a ∨ l1.b * l2.c ≠ l1.c * l2.b

/-- Theorem: For n points on a plane, not all collinear, there are at least n distinct lines -/
theorem distinct_lines_theorem (n : ℕ) (points : Fin n → Point) 
  (h : ∃ i j k : Fin n, ¬collinear (points i) (points j) (points k)) :
  ∃ (lines : Fin n → Line), ∀ i j : Fin n, i ≠ j → distinct_lines (lines i) (lines j) :=
sorry

end NUMINAMATH_CALUDE_distinct_lines_theorem_l4101_410178


namespace NUMINAMATH_CALUDE_inequality_solution_l4101_410148

theorem inequality_solution (x : ℝ) : (x - 2) * (6 + 2*x) > 0 ↔ x > 2 ∨ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l4101_410148


namespace NUMINAMATH_CALUDE_x_amount_proof_l4101_410143

def total_amount : ℝ := 5000
def ratio_x : ℝ := 2
def ratio_y : ℝ := 8

theorem x_amount_proof :
  let total_ratio := ratio_x + ratio_y
  let amount_per_part := total_amount / total_ratio
  let x_amount := amount_per_part * ratio_x
  x_amount = 1000 := by sorry

end NUMINAMATH_CALUDE_x_amount_proof_l4101_410143


namespace NUMINAMATH_CALUDE_olivias_chips_quarters_l4101_410154

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The total amount Olivia pays in dollars -/
def total_dollars : ℕ := 4

/-- The number of quarters Olivia pays for soda -/
def quarters_for_soda : ℕ := 12

/-- The number of quarters Olivia pays for chips -/
def quarters_for_chips : ℕ := total_dollars * quarters_per_dollar - quarters_for_soda

theorem olivias_chips_quarters : quarters_for_chips = 4 := by
  sorry

end NUMINAMATH_CALUDE_olivias_chips_quarters_l4101_410154


namespace NUMINAMATH_CALUDE_average_hamburgers_is_nine_l4101_410105

-- Define the total number of hamburgers sold
def total_hamburgers : ℕ := 63

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the average number of hamburgers sold per day
def average_hamburgers : ℚ := total_hamburgers / days_in_week

-- Theorem to prove
theorem average_hamburgers_is_nine : average_hamburgers = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_hamburgers_is_nine_l4101_410105


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tangent_l4101_410196

/-- Given an arithmetic sequence {a_n} where S_n is the sum of its first n terms,
    prove that if S_11 = 22π/3, then tan(a_6) = -√3 -/
theorem arithmetic_sequence_tangent (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  S 11 = 22 * Real.pi / 3 →             -- Given condition
  Real.tan (a 6) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tangent_l4101_410196


namespace NUMINAMATH_CALUDE_divisor_congruence_l4101_410190

theorem divisor_congruence (p n d : ℕ) : 
  Prime p → d ∣ ((n + 1)^p - n^p) → d ≡ 1 [MOD p] := by sorry

end NUMINAMATH_CALUDE_divisor_congruence_l4101_410190


namespace NUMINAMATH_CALUDE_set_operations_and_inclusion_l4101_410104

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 3 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x - a - 1 < 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem set_operations_and_inclusion (a : ℝ) : 
  (Set.compl A ∪ Set.compl B = {x | x ≤ 3 ∨ x ≥ 6}) ∧
  (B ⊆ C a ↔ a ≥ 8) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_inclusion_l4101_410104


namespace NUMINAMATH_CALUDE_f_seven_halves_eq_neg_sqrt_two_l4101_410133

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_seven_halves_eq_neg_sqrt_two
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_exp : ∀ x ∈ Set.Ioo 0 1, f x = 2^x) :
  f (7/2) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_seven_halves_eq_neg_sqrt_two_l4101_410133


namespace NUMINAMATH_CALUDE_semicircle_area_l4101_410102

theorem semicircle_area (diameter : ℝ) (h : diameter = 3) : 
  let radius : ℝ := diameter / 2
  let semicircle_area : ℝ := (π * radius^2) / 2
  semicircle_area = 9 * π / 8 := by
sorry

end NUMINAMATH_CALUDE_semicircle_area_l4101_410102


namespace NUMINAMATH_CALUDE_exists_n_pow_half_n_eq_ten_l4101_410116

theorem exists_n_pow_half_n_eq_ten : ∃ n : ℝ, n ^ (n / 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_pow_half_n_eq_ten_l4101_410116


namespace NUMINAMATH_CALUDE_max_floors_theorem_l4101_410199

/-- Represents a building with elevators and floors -/
structure Building where
  num_elevators : ℕ
  num_floors : ℕ
  stops_per_elevator : ℕ
  all_pairs_connected : Bool

/-- The maximum number of floors possible for a building with given constraints -/
def max_floors (b : Building) : ℕ :=
  sorry

/-- Theorem stating that for a building with 7 elevators, each stopping on 6 floors,
    and all pairs of floors connected, the maximum number of floors is 14 -/
theorem max_floors_theorem (b : Building) 
  (h1 : b.num_elevators = 7)
  (h2 : b.stops_per_elevator = 6)
  (h3 : b.all_pairs_connected = true) :
  max_floors b = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_floors_theorem_l4101_410199


namespace NUMINAMATH_CALUDE_total_mail_sent_l4101_410167

def mail_problem (monday tuesday wednesday thursday : ℕ) : Prop :=
  (tuesday = monday + 10) ∧
  (wednesday = tuesday - 5) ∧
  (thursday = wednesday + 15)

theorem total_mail_sent (monday : ℕ) : 
  monday = 65 → 
  ∃ (tuesday wednesday thursday : ℕ),
    mail_problem monday tuesday wednesday thursday ∧
    monday + tuesday + wednesday + thursday = 295 := by
  sorry

end NUMINAMATH_CALUDE_total_mail_sent_l4101_410167


namespace NUMINAMATH_CALUDE_calculate_overall_profit_l4101_410101

/-- Calculate the overall profit from selling two items with given purchase prices and profit/loss percentages -/
theorem calculate_overall_profit
  (grinder_price mobile_price : ℕ)
  (grinder_loss_percent mobile_profit_percent : ℚ)
  (h1 : grinder_price = 15000)
  (h2 : mobile_price = 10000)
  (h3 : grinder_loss_percent = 4 / 100)
  (h4 : mobile_profit_percent = 10 / 100)
  : ↑grinder_price * (1 - grinder_loss_percent) + 
    ↑mobile_price * (1 + mobile_profit_percent) - 
    ↑(grinder_price + mobile_price) = 400 := by
  sorry


end NUMINAMATH_CALUDE_calculate_overall_profit_l4101_410101


namespace NUMINAMATH_CALUDE_cattle_count_farm_cattle_count_l4101_410174

theorem cattle_count (cow_ratio : ℕ) (bull_ratio : ℕ) (bull_count : ℕ) : ℕ :=
  let total_ratio := cow_ratio + bull_ratio
  let parts := bull_count / bull_ratio
  let total_cattle := parts * total_ratio
  total_cattle

/-- Given a ratio of cows to bulls of 10:27 and 405 bulls, the total number of cattle is 675. -/
theorem farm_cattle_count : cattle_count 10 27 405 = 675 := by
  sorry

end NUMINAMATH_CALUDE_cattle_count_farm_cattle_count_l4101_410174


namespace NUMINAMATH_CALUDE_seven_boys_without_calculators_l4101_410189

/-- Represents the number of boys in Miss Parker's class who did not bring calculators. -/
def boys_without_calculators (total_students : ℕ) (boys_in_class : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ) : ℕ :=
  boys_in_class - (students_with_calculators - girls_with_calculators)

/-- Theorem stating that 7 boys in Miss Parker's class did not bring calculators. -/
theorem seven_boys_without_calculators :
  boys_without_calculators 24 18 26 15 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_boys_without_calculators_l4101_410189


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_nonpositive_l4101_410130

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem f_increasing_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_nonpositive_l4101_410130


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l4101_410155

theorem simplify_cube_roots : 
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 112 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l4101_410155


namespace NUMINAMATH_CALUDE_circumscribed_sphere_area_l4101_410182

theorem circumscribed_sphere_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 2 * Real.sqrt 6) :
  let diagonal_squared := a^2 + b^2 + c^2
  let sphere_radius := Real.sqrt (diagonal_squared / 4)
  let sphere_surface_area := 4 * Real.pi * sphere_radius^2
  sphere_surface_area = 49 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_area_l4101_410182


namespace NUMINAMATH_CALUDE_square_sum_equation_l4101_410161

theorem square_sum_equation : ∃ (x y : ℕ), x^2 + 12 = y^2 ∧ x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equation_l4101_410161


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l4101_410141

def A : Set ℤ := {0, -2}
def B : Set ℤ := {-4, 0}

theorem union_of_A_and_B :
  A ∪ B = {-4, -2, 0} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l4101_410141


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l4101_410176

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_three_digit_product (n p q : ℕ) : 
  n ≥ 100 ∧ n < 1000 ∧
  is_prime p ∧ p < 10 ∧
  q < 10 ∧
  n = p * q * (10 * p + q) ∧
  p ≠ q ∧ p ≠ (10 * p + q) ∧ q ≠ (10 * p + q) →
  n ≤ 777 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l4101_410176


namespace NUMINAMATH_CALUDE_lily_pad_coverage_l4101_410118

/-- Represents the number of days required for the lily pad patch to cover half the lake -/
def days_to_half_coverage : ℕ := 33

/-- Represents the number of days required for the lily pad patch to cover the entire lake -/
def days_to_full_coverage : ℕ := days_to_half_coverage + 1

/-- Theorem stating that the number of days to cover the entire lake is equal to
    the number of days to cover half the lake plus one -/
theorem lily_pad_coverage :
  days_to_full_coverage = days_to_half_coverage + 1 :=
by sorry

end NUMINAMATH_CALUDE_lily_pad_coverage_l4101_410118


namespace NUMINAMATH_CALUDE_ratio_squares_sum_l4101_410156

theorem ratio_squares_sum (a b c : ℝ) : 
  a / b = 3 / 2 ∧ 
  c / b = 5 / 2 ∧ 
  b = 14 → 
  a^2 + b^2 + c^2 = 1862 := by
sorry

end NUMINAMATH_CALUDE_ratio_squares_sum_l4101_410156


namespace NUMINAMATH_CALUDE_equivalent_ratios_l4101_410170

theorem equivalent_ratios (x : ℚ) : (3 : ℚ) / 12 = 3 / x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_ratios_l4101_410170


namespace NUMINAMATH_CALUDE_inequalities_proof_l4101_410147

theorem inequalities_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : a < b) (h3 : b < c) : 
  (a * b < b * c) ∧ 
  (a * c < b * c) ∧ 
  (a * b < a * c) ∧ 
  (a + b < b + c) := by
sorry

end NUMINAMATH_CALUDE_inequalities_proof_l4101_410147


namespace NUMINAMATH_CALUDE_sine_cosine_sum_equals_one_l4101_410179

theorem sine_cosine_sum_equals_one :
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_equals_one_l4101_410179


namespace NUMINAMATH_CALUDE_max_product_sum_l4101_410175

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({7, 8, 9, 10} : Set ℕ) →
  g ∈ ({7, 8, 9, 10} : Set ℕ) →
  h ∈ ({7, 8, 9, 10} : Set ℕ) →
  j ∈ ({7, 8, 9, 10} : Set ℕ) →
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j →
  (f * g + g * h + h * j + f * j) ≤ 289 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l4101_410175


namespace NUMINAMATH_CALUDE_reading_homework_pages_l4101_410109

theorem reading_homework_pages (total_pages math_pages : ℕ) 
  (h1 : total_pages = 7) 
  (h2 : math_pages = 5) : 
  total_pages - math_pages = 2 := by
sorry

end NUMINAMATH_CALUDE_reading_homework_pages_l4101_410109


namespace NUMINAMATH_CALUDE_scientific_notation_of_10374_billion_l4101_410145

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The value to be converted (10,374 billion yuan) -/
def originalValue : ℝ := 10374 * 1000000000

/-- The number of significant figures to retain -/
def sigFigures : ℕ := 3

theorem scientific_notation_of_10374_billion :
  toScientificNotation originalValue sigFigures =
    ScientificNotation.mk 1.037 13 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_10374_billion_l4101_410145
