import Mathlib

namespace NUMINAMATH_CALUDE_relay_race_arrangements_l3615_361519

def number_of_students : ℕ := 4
def fixed_position : ℕ := 1
def available_positions : ℕ := number_of_students - fixed_position

theorem relay_race_arrangements :
  (available_positions.factorial) = 6 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l3615_361519


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l3615_361595

/-- Probability of selecting two non-defective pens from a box -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 16) (h2 : defective_pens = 3) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l3615_361595


namespace NUMINAMATH_CALUDE_triangle_properties_l3615_361548

/-- Triangle ABC with given points and conditions -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  h_A : A = (-2, 1)
  h_B : B = (4, 3)

/-- The equation of a line in general form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point lies on a line -/
def lies_on (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Predicate to check if a line is perpendicular to another line -/
def perpendicular (l1 l2 : LineEquation) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem triangle_properties (t : Triangle) :
  (t.C = (3, -2) →
    ∃ (l : LineEquation), l.a = 1 ∧ l.b = 5 ∧ l.c = -3 ∧
    lies_on t.A l ∧
    ∃ (bc : LineEquation), lies_on t.B bc ∧ lies_on t.C bc ∧ perpendicular l bc) ∧
  (t.M = (3, 1) ∧ t.M.1 = (t.A.1 + t.C.1) / 2 ∧ t.M.2 = (t.A.2 + t.C.2) / 2 →
    ∃ (l : LineEquation), l.a = 1 ∧ l.b = 2 ∧ l.c = -10 ∧
    lies_on t.B l ∧ lies_on t.C l) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3615_361548


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3615_361524

/-- A right triangle with sides 6, 8, and 10 (hypotenuse) -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right : side1 = 6 ∧ side2 = 8 ∧ hypotenuse = 10

/-- A square inscribed in the triangle with one vertex at the right angle -/
def inscribed_square_at_right_angle (t : RightTriangle) (x : ℝ) : Prop :=
  0 < x ∧ x < t.side1 ∧ x < t.side2

/-- A square inscribed in the triangle with one side on the hypotenuse -/
def inscribed_square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  0 < y ∧ y < t.side1 ∧ y < t.side2

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ)
  (h1 : inscribed_square_at_right_angle t x)
  (h2 : inscribed_square_on_hypotenuse t y) :
  x / y = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3615_361524


namespace NUMINAMATH_CALUDE_picture_area_l3615_361596

theorem picture_area (x y : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 * x + 5) * (y + 3) = 60) : x * y = 27 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l3615_361596


namespace NUMINAMATH_CALUDE_limit_is_nonzero_real_l3615_361565

noncomputable def f (x : ℝ) : ℝ := x^(5/3) * ((x + 1)^(1/3) + (x - 1)^(1/3) - 2 * x^(1/3))

theorem limit_is_nonzero_real : ∃ (L : ℝ), L ≠ 0 ∧ Filter.Tendsto f Filter.atTop (nhds L) := by
  sorry

end NUMINAMATH_CALUDE_limit_is_nonzero_real_l3615_361565


namespace NUMINAMATH_CALUDE_min_digits_theorem_l3615_361526

/-- The minimum number of digits to the right of the decimal point needed to express the given fraction as a decimal -/
def min_decimal_digits : ℕ := 30

/-- The numerator of the fraction -/
def numerator : ℕ := 987654321

/-- The denominator of the fraction -/
def denominator : ℕ := 2^30 * 5^6

/-- Theorem stating that the minimum number of digits to the right of the decimal point
    needed to express the fraction numerator/denominator as a decimal is min_decimal_digits -/
theorem min_digits_theorem :
  (∀ n : ℕ, n < min_decimal_digits → ∃ m : ℕ, m * denominator ≠ numerator * 10^n) ∧
  (∃ m : ℕ, m * denominator = numerator * 10^min_decimal_digits) :=
sorry

end NUMINAMATH_CALUDE_min_digits_theorem_l3615_361526


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l3615_361586

theorem square_of_binomial_constant (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 4*x^2 + 16*x + m = (a*x + b)^2) → m = 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l3615_361586


namespace NUMINAMATH_CALUDE_exists_square_with_1983_nines_l3615_361509

theorem exists_square_with_1983_nines : ∃ n : ℕ, ∃ m : ℕ, n^2 = 10^3968 - 10^1985 + m ∧ m < 10^1985 := by
  sorry

end NUMINAMATH_CALUDE_exists_square_with_1983_nines_l3615_361509


namespace NUMINAMATH_CALUDE_rectangle_sides_l3615_361599

theorem rectangle_sides (x y : ℝ) : 
  (2 * (x + y) = 124) →  -- Perimeter of rectangle is 124 cm
  (4 * Real.sqrt ((x/2)^2 + ((124/2 - x)/2)^2) = 100) →  -- Perimeter of rhombus is 100 cm
  ((x = 48 ∧ y = 14) ∨ (x = 14 ∧ y = 48)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_sides_l3615_361599


namespace NUMINAMATH_CALUDE_correct_average_calculation_l3615_361589

/-- Proves that the correct average is 22 given the conditions of the problem -/
theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) 
  (hn : n = 10) 
  (hinitial : initial_avg = 18) 
  (hincorrect : incorrect_num = 26)
  (hcorrect : correct_num = 66) :
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 22 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l3615_361589


namespace NUMINAMATH_CALUDE_binomial_identity_sum_identity_l3615_361517

def binomial (n p : ℕ) : ℕ := if p ≤ n then n.factorial / (p.factorial * (n - p).factorial) else 0

theorem binomial_identity (n p : ℕ) (h : n ≥ p ∧ p ≥ 1) :
  binomial n p = (Finset.range (n - p + 1)).sum (fun i => binomial (n - 1 - i) (p - 1)) :=
sorry

theorem sum_identity :
  (Finset.range 97).sum (fun k => (k + 1) * (k + 2) * (k + 3)) = 23527350 :=
sorry

end NUMINAMATH_CALUDE_binomial_identity_sum_identity_l3615_361517


namespace NUMINAMATH_CALUDE_poem_line_increase_l3615_361566

theorem poem_line_increase (initial_lines : ℕ) (target_lines : ℕ) (lines_per_month : ℕ) (months : ℕ) : 
  initial_lines = 24 →
  target_lines = 90 →
  lines_per_month = 3 →
  initial_lines + months * lines_per_month = target_lines →
  months = 22 := by
sorry

end NUMINAMATH_CALUDE_poem_line_increase_l3615_361566


namespace NUMINAMATH_CALUDE_airplane_seats_theorem_l3615_361593

/-- Represents the total number of seats in an airplane -/
def total_seats : ℕ := 180

/-- Represents the number of seats in First Class -/
def first_class_seats : ℕ := 36

/-- Represents the fraction of total seats in Business Class -/
def business_class_fraction : ℚ := 1/5

/-- Represents the fraction of total seats in Economy Class -/
def economy_class_fraction : ℚ := 3/5

/-- Theorem stating that the total number of seats is correct given the conditions -/
theorem airplane_seats_theorem :
  (first_class_seats : ℚ) + 
  business_class_fraction * total_seats + 
  economy_class_fraction * total_seats = total_seats := by sorry

end NUMINAMATH_CALUDE_airplane_seats_theorem_l3615_361593


namespace NUMINAMATH_CALUDE_village_population_panic_l3615_361561

theorem village_population_panic (original_population : ℕ) (final_population : ℕ) 
  (h1 : original_population = 7600)
  (h2 : final_population = 5130) :
  let remaining_after_initial := original_population - original_population / 10
  let left_during_panic := remaining_after_initial - final_population
  (left_during_panic : ℚ) / remaining_after_initial * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_village_population_panic_l3615_361561


namespace NUMINAMATH_CALUDE_opposite_face_is_B_l3615_361510

-- Define the faces of the cube
inductive Face : Type
| X | A | B | C | D | E

-- Define the net structure
structure Net :=
  (faces : Finset Face)
  (center : Face)
  (surrounding : List Face)
  (adjacent_to_A : Face)
  (adjacent_to_D : Face)

-- Define the property of being opposite in a cube
def is_opposite (f1 f2 : Face) : Prop := sorry

-- Define the cube folding function
def fold_to_cube (n : Net) : Prop := sorry

-- Theorem statement
theorem opposite_face_is_B (n : Net) : 
  n.faces.card = 6 ∧ 
  n.center = Face.X ∧ 
  n.surrounding = [Face.A, Face.B, Face.D] ∧
  n.adjacent_to_A = Face.C ∧
  n.adjacent_to_D = Face.E ∧
  fold_to_cube n →
  is_opposite Face.X Face.B :=
sorry

end NUMINAMATH_CALUDE_opposite_face_is_B_l3615_361510


namespace NUMINAMATH_CALUDE_perimeter_ratio_specific_triangle_l3615_361541

/-- Right triangle DEF with altitude FG and external point J -/
structure RightTriangleWithAltitude where
  /-- Length of side DF -/
  df : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of hypotenuse DE -/
  de : ℝ
  /-- Length of altitude FG -/
  fg : ℝ
  /-- Length of tangent from J to circle with diameter FG -/
  tj : ℝ
  /-- de² = df² + ef² (Pythagorean theorem) -/
  pythagoras : de^2 = df^2 + ef^2
  /-- fg² = df * ef (geometric mean property of altitude) -/
  altitude_property : fg^2 = df * ef
  /-- tj² = df * (de - df) (tangent-secant theorem) -/
  tangent_secant : tj^2 = df * (de - df)

/-- Theorem: Perimeter ratio for specific right triangle -/
theorem perimeter_ratio_specific_triangle :
  ∀ t : RightTriangleWithAltitude,
  t.df = 9 →
  t.ef = 40 →
  (t.de + 2 * t.tj) / t.de = 49 / 41 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_specific_triangle_l3615_361541


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3615_361554

theorem greatest_power_of_two_factor (n : ℕ) : n = 1003 →
  ∃ k : ℕ, (2^n : ℤ) ∣ (10^n - 4^(n/2)) ∧
  ∀ m : ℕ, m > n → ¬((2^m : ℤ) ∣ (10^n - 4^(n/2))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3615_361554


namespace NUMINAMATH_CALUDE_train_speed_l3615_361588

/-- The speed of a train passing through a tunnel -/
theorem train_speed (train_length : Real) (tunnel_length : Real) (time_minutes : Real) :
  train_length = 0.1 →
  tunnel_length = 2.9 →
  time_minutes = 2.5 →
  ∃ (speed : Real), abs (speed - 71.94) < 0.01 ∧ 
    speed = (tunnel_length + train_length) / (time_minutes / 60) := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l3615_361588


namespace NUMINAMATH_CALUDE_jake_roll_combinations_l3615_361542

/-- The number of different combinations of rolls Jake could buy -/
def num_combinations : ℕ := 3

/-- The number of types of rolls available -/
def num_roll_types : ℕ := 3

/-- The total number of rolls Jake needs to purchase -/
def total_rolls : ℕ := 7

/-- The minimum number of each type of roll Jake must purchase -/
def min_per_type : ℕ := 2

theorem jake_roll_combinations :
  num_combinations = 3 ∧
  num_roll_types = 3 ∧
  total_rolls = 7 ∧
  min_per_type = 2 ∧
  total_rolls = num_roll_types * min_per_type + 1 →
  num_combinations = num_roll_types :=
by sorry

end NUMINAMATH_CALUDE_jake_roll_combinations_l3615_361542


namespace NUMINAMATH_CALUDE_ratio_of_fourth_power_equality_l3615_361527

theorem ratio_of_fourth_power_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4) :
  b / a = 1 := by sorry

end NUMINAMATH_CALUDE_ratio_of_fourth_power_equality_l3615_361527


namespace NUMINAMATH_CALUDE_mathematician_project_time_l3615_361568

theorem mathematician_project_time (project1 : ℕ) (project2 : ℕ) (daily_questions : ℕ) : 
  project1 = 518 → project2 = 476 → daily_questions = 142 → 
  (project1 + project2) / daily_questions = 7 := by
  sorry

end NUMINAMATH_CALUDE_mathematician_project_time_l3615_361568


namespace NUMINAMATH_CALUDE_test_questions_count_l3615_361573

theorem test_questions_count :
  ∀ (total_questions : ℕ),
    total_questions % 5 = 0 →
    32 > (70 * total_questions) / 100 →
    32 < (77 * total_questions) / 100 →
    total_questions = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l3615_361573


namespace NUMINAMATH_CALUDE_brad_read_more_books_l3615_361545

/-- Proves that Brad read 4 more books than William across two months --/
theorem brad_read_more_books (william_last_month : ℕ) (brad_this_month : ℕ) : 
  william_last_month = 6 →
  brad_this_month = 8 →
  (3 * william_last_month + brad_this_month) - (william_last_month + 2 * brad_this_month) = 4 := by
sorry

end NUMINAMATH_CALUDE_brad_read_more_books_l3615_361545


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3615_361534

theorem no_solution_for_equation :
  ¬∃ (x : ℝ), x ≠ 1 ∧ (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3615_361534


namespace NUMINAMATH_CALUDE_prob_no_red_square_is_127_128_l3615_361547

/-- Represents a 4-by-4 grid where each cell can be colored red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Returns true if the grid has a 3-by-3 red square starting at (i, j) -/
def has_red_square (g : Grid) (i j : Fin 2) : Prop :=
  ∀ (x y : Fin 3), g (i + x) (j + y) = true

/-- The probability of a grid not having any 3-by-3 red square -/
def prob_no_red_square : ℚ :=
  1 - (4 : ℚ) / 2^9

theorem prob_no_red_square_is_127_128 :
  prob_no_red_square = 127 / 128 := by sorry

#check prob_no_red_square_is_127_128

end NUMINAMATH_CALUDE_prob_no_red_square_is_127_128_l3615_361547


namespace NUMINAMATH_CALUDE_smallest_equal_gum_pieces_l3615_361598

theorem smallest_equal_gum_pieces (n : ℕ) : n > 0 ∧ n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 → n ≥ 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_equal_gum_pieces_l3615_361598


namespace NUMINAMATH_CALUDE_complex_fraction_sum_simplification_l3615_361500

theorem complex_fraction_sum_simplification :
  let i : ℂ := Complex.I
  ((4 + 7 * i) / (4 - 7 * i) + (4 - 7 * i) / (4 + 7 * i)) = (-66 : ℚ) / 65 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_simplification_l3615_361500


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3615_361585

theorem inequality_system_solutions (m : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ 
   3 - 2 * (x : ℝ) ≥ 0 ∧ (x : ℝ) ≥ m ∧
   3 - 2 * (y : ℝ) ≥ 0 ∧ (y : ℝ) ≥ m ∧
   (∀ z : ℤ, z ≠ x ∧ z ≠ y → ¬(3 - 2 * (z : ℝ) ≥ 0 ∧ (z : ℝ) ≥ m))) →
  -1 < m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3615_361585


namespace NUMINAMATH_CALUDE_hexagon_enclosed_by_polygons_l3615_361525

/-- A regular hexagon is enclosed by m regular n-sided polygons, where three polygons meet at each vertex of the hexagon. -/
theorem hexagon_enclosed_by_polygons (m : ℕ) (n : ℕ) : n = 18 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_enclosed_by_polygons_l3615_361525


namespace NUMINAMATH_CALUDE_first_place_points_l3615_361594

def second_place_points : Nat := 7
def third_place_points : Nat := 5
def fourth_place_points : Nat := 2
def total_participations : Nat := 7
def product_of_points : Nat := 38500

theorem first_place_points (first_place : Nat) 
  (h1 : ∃ (a b c d : Nat), a + b + c + d = total_participations ∧ 
                           a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
                           first_place^a * second_place_points^b * 
                           third_place_points^c * fourth_place_points^d = product_of_points) : 
  first_place = 11 := by
sorry

end NUMINAMATH_CALUDE_first_place_points_l3615_361594


namespace NUMINAMATH_CALUDE_valentines_day_cards_l3615_361516

theorem valentines_day_cards (total_students : ℕ) (card_cost : ℚ) (total_money : ℚ) 
  (spend_percentage : ℚ) (h1 : total_students = 30) (h2 : card_cost = 2) 
  (h3 : total_money = 40) (h4 : spend_percentage = 0.9) : 
  (((total_money * spend_percentage) / card_cost) / total_students) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_valentines_day_cards_l3615_361516


namespace NUMINAMATH_CALUDE_no_non_multiple_ghosts_l3615_361555

/-- Definition of the sequence S -/
def S (p : ℕ) : ℕ → ℕ
  | n => if n < p then n else sorry

/-- A number is a ghost if it doesn't appear in S -/
def is_ghost (p : ℕ) (k : ℕ) : Prop :=
  ∀ n, S p n ≠ k

/-- Main theorem: There are no ghosts that are not multiples of p -/
theorem no_non_multiple_ghosts (p : ℕ) (hp : Prime p) (hp_odd : Odd p) :
  ∀ k, ¬(p ∣ k) → ¬(is_ghost p k) := by sorry

end NUMINAMATH_CALUDE_no_non_multiple_ghosts_l3615_361555


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3615_361550

/-- A quadratic equation ax^2 + bx + c = 0 has two distinct real roots if and only if its discriminant is positive -/
axiom quadratic_two_roots (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b^2 - 4*a*c > 0

/-- If c < 1/4, then the quadratic equation x^2 + 2x + 4c = 0 has two distinct real roots -/
theorem quadratic_roots_condition (c : ℝ) (h : c < 1/4) :
  ∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + 4*c = 0 ∧ y^2 + 2*y + 4*c = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3615_361550


namespace NUMINAMATH_CALUDE_cable_length_l3615_361537

/-- The length of the curve defined by the intersection of a plane and a sphere --/
theorem cable_length (x y z : ℝ) : 
  x + y + z = 10 → 
  x * y + y * z + x * z = -22 → 
  (∃ (l : ℝ), l = 4 * Real.pi * Real.sqrt (83 / 3) ∧ 
   l = 2 * Real.pi * Real.sqrt (144 - (10^2 / 3))) :=
by sorry

end NUMINAMATH_CALUDE_cable_length_l3615_361537


namespace NUMINAMATH_CALUDE_nested_square_roots_simplification_l3615_361531

theorem nested_square_roots_simplification :
  Real.sqrt (36 * Real.sqrt (18 * Real.sqrt 9)) = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_roots_simplification_l3615_361531


namespace NUMINAMATH_CALUDE_exists_right_triangle_different_colors_l3615_361520

-- Define the color type
inductive Color
| Blue
| Green
| Red

-- Define the plane as a type
def Plane := ℝ × ℝ

-- Define a coloring function
def coloring : Plane → Color := sorry

-- Define the existence of at least one point of each color
axiom exists_blue : ∃ p : Plane, coloring p = Color.Blue
axiom exists_green : ∃ p : Plane, coloring p = Color.Green
axiom exists_red : ∃ p : Plane, coloring p = Color.Red

-- Define a right triangle
def is_right_triangle (p q r : Plane) : Prop := sorry

-- Theorem statement
theorem exists_right_triangle_different_colors :
  ∃ p q r : Plane, 
    is_right_triangle p q r ∧ 
    coloring p ≠ coloring q ∧ 
    coloring q ≠ coloring r ∧ 
    coloring r ≠ coloring p :=
sorry

end NUMINAMATH_CALUDE_exists_right_triangle_different_colors_l3615_361520


namespace NUMINAMATH_CALUDE_subset_M_l3615_361546

def M : Set ℕ := {x : ℕ | (1 : ℚ) / (x - 2 : ℚ) ≤ 0}

theorem subset_M : {1} ⊆ M := by sorry

end NUMINAMATH_CALUDE_subset_M_l3615_361546


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3615_361502

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem complement_of_A_in_U :
  (U \ A) = {x | 0 < x ∧ x ≤ 1} ∪ {x | x ≥ 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3615_361502


namespace NUMINAMATH_CALUDE_micheal_work_days_l3615_361569

/-- Represents the total amount of work to be done -/
def W : ℝ := 1

/-- Represents the rate at which Micheal works (fraction of work done per day) -/
def M : ℝ := sorry

/-- Represents the rate at which Adam works (fraction of work done per day) -/
def A : ℝ := sorry

/-- Micheal and Adam can do the work together in 20 days -/
axiom combined_rate : M + A = W / 20

/-- After working together for 14 days, the remaining work is completed by Adam in 10 days -/
axiom remaining_work : A * 10 = W - 14 * (M + A)

theorem micheal_work_days : M = W / 50 := by sorry

end NUMINAMATH_CALUDE_micheal_work_days_l3615_361569


namespace NUMINAMATH_CALUDE_lunch_cakes_count_l3615_361584

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := sorry

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served -/
def total_cakes : ℕ := 14

/-- Theorem stating that the number of cakes served during lunch today is 5 -/
theorem lunch_cakes_count : lunch_cakes = 5 := by
  sorry

#check lunch_cakes_count

end NUMINAMATH_CALUDE_lunch_cakes_count_l3615_361584


namespace NUMINAMATH_CALUDE_vasya_reading_time_difference_l3615_361580

/-- Represents the number of books Vasya planned to read each week -/
def planned_books_per_week : ℕ := sorry

/-- Represents the total number of books in the reading list -/
def total_books : ℕ := 12 * planned_books_per_week

/-- Represents the number of weeks it took Vasya to finish when reading one less book per week -/
def actual_weeks : ℕ := 12 + 3

theorem vasya_reading_time_difference :
  (total_books / (planned_books_per_week + 1) = 10) ∧
  (10 = 12 - 2) :=
by sorry

end NUMINAMATH_CALUDE_vasya_reading_time_difference_l3615_361580


namespace NUMINAMATH_CALUDE_product_of_positive_real_part_roots_l3615_361544

theorem product_of_positive_real_part_roots : ∃ (roots : Finset ℂ),
  (∀ z ∈ roots, z^6 = -64) ∧
  (∀ z ∈ roots, (z.re : ℝ) > 0) ∧
  (roots.prod id = 4) := by
sorry

end NUMINAMATH_CALUDE_product_of_positive_real_part_roots_l3615_361544


namespace NUMINAMATH_CALUDE_add_negative_two_and_two_equals_zero_l3615_361539

theorem add_negative_two_and_two_equals_zero : (-2) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_add_negative_two_and_two_equals_zero_l3615_361539


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l3615_361558

theorem power_equality_implies_exponent (p : ℕ) : 16^10 = 4^p → p = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l3615_361558


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l3615_361549

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 134 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l3615_361549


namespace NUMINAMATH_CALUDE_students_without_A_l3615_361538

theorem students_without_A (total : ℕ) (chem : ℕ) (phys : ℕ) (both : ℕ) : 
  total = 40 → chem = 10 → phys = 18 → both = 6 →
  total - (chem + phys - both) = 18 := by sorry

end NUMINAMATH_CALUDE_students_without_A_l3615_361538


namespace NUMINAMATH_CALUDE_point_on_parabola_l3615_361579

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 1

-- Define the theorem
theorem point_on_parabola (y w : ℝ) :
  parabola 3 = y → w = 2 → y = 4 * w := by
  sorry

end NUMINAMATH_CALUDE_point_on_parabola_l3615_361579


namespace NUMINAMATH_CALUDE_gcd_of_225_and_135_l3615_361522

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_225_and_135_l3615_361522


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l3615_361570

/-- A parabola passing through (-3, 0) with axis of symmetry x = -1 has coefficient sum of 0 -/
theorem parabola_coefficient_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = -3 ∨ x = 1) →
  -b / (2 * a) = -1 →
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l3615_361570


namespace NUMINAMATH_CALUDE_apple_pricing_l3615_361592

/-- The cost of apples per kilogram for the first 30 kgs -/
def l : ℝ := 0.362

/-- The cost of apples per kilogram for each additional kg after 30 kgs -/
def m : ℝ := 0.27

/-- The price of 33 kilograms of apples -/
def price_33kg : ℝ := 11.67

/-- The price of 36 kilograms of apples -/
def price_36kg : ℝ := 12.48

/-- The cost of the first 10 kgs of apples -/
def cost_10kg : ℝ := 3.62

theorem apple_pricing :
  (10 * l = cost_10kg) ∧
  (30 * l + 3 * m = price_33kg) ∧
  (30 * l + 6 * m = price_36kg) →
  m = 0.27 := by
  sorry

end NUMINAMATH_CALUDE_apple_pricing_l3615_361592


namespace NUMINAMATH_CALUDE_semi_circle_area_l3615_361513

/-- The area of a semi-circle with diameter 10 meters is 12.5π square meters. -/
theorem semi_circle_area (π : ℝ) : 
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let semi_circle_area : ℝ := π * radius^2 / 2
  semi_circle_area = 12.5 * π := by
  sorry

end NUMINAMATH_CALUDE_semi_circle_area_l3615_361513


namespace NUMINAMATH_CALUDE_incorrect_height_calculation_l3615_361577

theorem incorrect_height_calculation (n : ℕ) (initial_avg real_avg actual_height : ℝ) 
  (h1 : n = 35)
  (h2 : initial_avg = 185)
  (h3 : real_avg = 183)
  (h4 : actual_height = 106) :
  ∃ (incorrect_height : ℝ),
    incorrect_height = n * initial_avg - (n * real_avg - actual_height) ∧
    incorrect_height = 176 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_height_calculation_l3615_361577


namespace NUMINAMATH_CALUDE_expression_simplification_l3615_361591

theorem expression_simplification (b : ℝ) (h : b ≠ -1) :
  1 - (1 / (1 - b / (1 + b))) = -b :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3615_361591


namespace NUMINAMATH_CALUDE_example_monomial_properties_l3615_361575

/-- Represents a monomial with integer coefficient and variables x, y, and z -/
structure Monomial where
  coeff : Int
  x_exp : Nat
  y_exp : Nat
  z_exp : Nat

/-- Calculates the coefficient of a monomial -/
def coefficient (m : Monomial) : Int :=
  m.coeff

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : Nat :=
  m.x_exp + m.y_exp + m.z_exp

/-- The monomial -3^2 * x * y * z^2 -/
def example_monomial : Monomial :=
  { coeff := -9, x_exp := 1, y_exp := 1, z_exp := 2 }

theorem example_monomial_properties :
  (coefficient example_monomial = -9) ∧ (degree example_monomial = 4) := by
  sorry


end NUMINAMATH_CALUDE_example_monomial_properties_l3615_361575


namespace NUMINAMATH_CALUDE_problem_solution_l3615_361511

theorem problem_solution (a b c : ℝ) (h1 : a = 8 - b) (h2 : c^2 = a*b - 16) : 
  a + c = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3615_361511


namespace NUMINAMATH_CALUDE_first_candidate_marks_l3615_361574

/-- Represents the total marks in the exam -/
def total_marks : ℝ := 600

/-- Represents the passing marks -/
def passing_marks : ℝ := 240

/-- Represents the percentage of marks obtained by the first candidate -/
def first_candidate_percentage : ℝ := 30

/-- Theorem stating the percentage of marks obtained by the first candidate -/
theorem first_candidate_marks :
  let second_candidate_marks := 0.45 * total_marks
  let first_candidate_marks := (first_candidate_percentage / 100) * total_marks
  (second_candidate_marks = passing_marks + 30) ∧
  (first_candidate_marks = passing_marks - 60) →
  first_candidate_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_first_candidate_marks_l3615_361574


namespace NUMINAMATH_CALUDE_problem_paths_l3615_361514

/-- Represents the graph of points and their connections -/
structure PointGraph where
  blue_points : Nat
  red_points : Nat
  red_connected_to_blue : Bool
  blue_connected_to_each_other : Bool

/-- Calculates the number of paths between red points -/
def count_paths (g : PointGraph) : Nat :=
  sorry

/-- The specific graph configuration from the problem -/
def problem_graph : PointGraph :=
  { blue_points := 8
  , red_points := 2
  , red_connected_to_blue := true
  , blue_connected_to_each_other := true }

/-- Theorem stating the number of paths in the problem -/
theorem problem_paths :
  count_paths problem_graph = 645120 :=
by sorry

end NUMINAMATH_CALUDE_problem_paths_l3615_361514


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3615_361515

theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  a + b = 105 →
  b = a + 40 →
  max a (max b c) = 75 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3615_361515


namespace NUMINAMATH_CALUDE_smallest_c_for_max_at_zero_l3615_361572

/-- Given a function y = a * cos(b * x + c) where a, b, and c are positive constants,
    and the graph reaches a maximum at x = 0, the smallest possible value of c is 0. -/
theorem smallest_c_for_max_at_zero (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos c) →
  (∀ ε > 0, ∃ x, a * Real.cos (b * x + (c - ε)) > a * Real.cos (c - ε)) →
  c = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_max_at_zero_l3615_361572


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l3615_361508

def base_seven_to_decimal (n : ℕ) : ℕ := 
  2 * 7^6 + 1 * 7^5 + 0 * 7^4 + 2 * 7^3 + 0 * 7^2 + 1 * 7^1 + 2 * 7^0

def number : ℕ := base_seven_to_decimal 2102012

theorem largest_prime_divisor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ number ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ number → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l3615_361508


namespace NUMINAMATH_CALUDE_minimize_triangle_side_l3615_361507

noncomputable def minimizeTriangleSide (t : ℝ) (C : ℝ) : ℝ × ℝ × ℝ :=
  let a := (2 * t / Real.sin C) ^ (1/2)
  let b := a
  let c := 2 * (t * Real.tan (C/2)) ^ (1/2)
  (a, b, c)

theorem minimize_triangle_side (t : ℝ) (C : ℝ) (h1 : t > 0) (h2 : 0 < C ∧ C < π) :
  let (a, b, c) := minimizeTriangleSide t C
  (∀ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 2 * t = a' * b' * Real.sin C →
    c ≤ (a'^2 + b'^2 - 2*a'*b'*Real.cos C)^(1/2)) ∧
  a = b ∧
  c = 2 * (t * Real.tan (C/2))^(1/2) :=
by sorry

end NUMINAMATH_CALUDE_minimize_triangle_side_l3615_361507


namespace NUMINAMATH_CALUDE_cube_root_of_x_sqrt_x_l3615_361504

theorem cube_root_of_x_sqrt_x (x : ℝ) (hx : x > 0) : 
  (x * Real.sqrt x) ^ (1/3 : ℝ) = x ^ (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_x_sqrt_x_l3615_361504


namespace NUMINAMATH_CALUDE_copper_content_bounds_l3615_361557

/-- Represents the composition of an alloy --/
structure Alloy where
  nickel : ℝ
  copper : ℝ
  manganese : ℝ
  sum_to_one : nickel + copper + manganese = 1

/-- The three initial alloys --/
def alloy1 : Alloy := ⟨0.3, 0.7, 0, by norm_num⟩
def alloy2 : Alloy := ⟨0, 0.1, 0.9, by norm_num⟩
def alloy3 : Alloy := ⟨0.15, 0.25, 0.6, by norm_num⟩

/-- The fraction of each initial alloy in the new alloy --/
structure Fractions where
  x1 : ℝ
  x2 : ℝ
  x3 : ℝ
  sum_to_one : x1 + x2 + x3 = 1
  manganese_constraint : 0.9 * x2 + 0.6 * x3 = 0.4

/-- The copper content in the new alloy --/
def copper_content (f : Fractions) : ℝ :=
  0.7 * f.x1 + 0.1 * f.x2 + 0.25 * f.x3

/-- The main theorem stating the bounds of copper content --/
theorem copper_content_bounds (f : Fractions) : 
  0.4 ≤ copper_content f ∧ copper_content f ≤ 13/30 := by sorry

end NUMINAMATH_CALUDE_copper_content_bounds_l3615_361557


namespace NUMINAMATH_CALUDE_min_factors_to_remove_for_2_l3615_361503

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def endsIn2 (n : ℕ) : Prop := n % 10 = 2

def factorsToRemove (n : ℕ) : ℕ := 
  let multiples_of_5 := n / 5
  multiples_of_5 + 1

theorem min_factors_to_remove_for_2 : 
  ∃ (removed : Finset ℕ), 
    removed.card = factorsToRemove 99 ∧ 
    endsIn2 ((factorial 99) / (removed.prod id)) ∧
    ∀ (other : Finset ℕ), other.card < factorsToRemove 99 → 
      ¬(endsIn2 ((factorial 99) / (other.prod id))) :=
sorry

end NUMINAMATH_CALUDE_min_factors_to_remove_for_2_l3615_361503


namespace NUMINAMATH_CALUDE_value_k_std_dev_below_mean_l3615_361551

-- Define the properties of the normal distribution
def mean : ℝ := 12
def std_dev : ℝ := 1.2

-- Define the range for k
def k_range (k : ℝ) : Prop := 2 < k ∧ k < 3 ∧ k ≠ ⌊k⌋

-- Theorem statement
theorem value_k_std_dev_below_mean (k : ℝ) (h : k_range k) :
  ∃ (value : ℝ), value = mean - k * std_dev :=
sorry

end NUMINAMATH_CALUDE_value_k_std_dev_below_mean_l3615_361551


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3615_361528

theorem inequality_solution_set (x : ℝ) : (3 - 2*x) * (x + 1) ≤ 0 ↔ x < -1 ∨ x ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3615_361528


namespace NUMINAMATH_CALUDE_optimal_strategy_probability_l3615_361562

/-- Represents the color of a hat -/
inductive HatColor
| Red
| Blue

/-- A strategy for guessing hat colors -/
def Strategy := (n : Nat) → (Vector HatColor n) → Vector Bool n

/-- The probability of all prisoners guessing correctly given a strategy -/
def SuccessProbability (n : Nat) (s : Strategy) : ℚ :=
  sorry

/-- Theorem stating that the maximum success probability is 1/2 -/
theorem optimal_strategy_probability (n : Nat) :
  ∀ s : Strategy, SuccessProbability n s ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_probability_l3615_361562


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3615_361564

/-- Proves that given a person's age is 40, and 7 years earlier they were 11 times their daughter's age,
    the ratio of their age to their daughter's age today is 4:1 -/
theorem age_ratio_proof (your_age : ℕ) (daughter_age : ℕ) : 
  your_age = 40 →
  your_age - 7 = 11 * (daughter_age - 7) →
  your_age / daughter_age = 4 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3615_361564


namespace NUMINAMATH_CALUDE_two_sunny_days_probability_l3615_361553

/-- The probability of exactly k successes in n independent trials,
    each with probability p of success. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

theorem two_sunny_days_probability :
  binomial_probability 5 2 (3/10 : ℝ) = 3087/10000 := by
  sorry

end NUMINAMATH_CALUDE_two_sunny_days_probability_l3615_361553


namespace NUMINAMATH_CALUDE_certain_part_of_number_l3615_361578

theorem certain_part_of_number (x y : ℝ) : 
  x = 1925 → 
  (1 / 7) * x = y + 100 → 
  y = 175 := by
  sorry

end NUMINAMATH_CALUDE_certain_part_of_number_l3615_361578


namespace NUMINAMATH_CALUDE_area_relation_l3615_361543

/-- A triangle is acute-angled if all its angles are less than 90 degrees. -/
def IsAcuteAngledTriangle (A B C : ℝ × ℝ) : Prop := sorry

/-- The orthocentre of a triangle is the point where all three altitudes intersect. -/
def Orthocentre (A B C H : ℝ × ℝ) : Prop := sorry

/-- The centroid of a triangle is the arithmetic mean position of all points in the triangle. -/
def Centroid (A B C G : ℝ × ℝ) : Prop := sorry

/-- The area of a triangle given its vertices. -/
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem area_relation (A B C H G₁ G₂ G₃ : ℝ × ℝ) :
  IsAcuteAngledTriangle A B C →
  Orthocentre A B C H →
  Centroid H B C G₁ →
  Centroid H C A G₂ →
  Centroid H A B G₃ →
  TriangleArea G₁ G₂ G₃ = 7 →
  TriangleArea A B C = 63 := by
  sorry

end NUMINAMATH_CALUDE_area_relation_l3615_361543


namespace NUMINAMATH_CALUDE_symmetry_point_xOy_l3615_361540

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xOy plane -/
def symmetryXOy (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetry_point_xOy :
  let P : Point3D := { x := -3, y := 2, z := -1 }
  symmetryXOy P = { x := -3, y := 2, z := 1 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_xOy_l3615_361540


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l3615_361518

theorem stratified_sampling_medium_supermarkets 
  (total_sample : ℕ) 
  (large_supermarkets : ℕ) 
  (medium_supermarkets : ℕ) 
  (small_supermarkets : ℕ) 
  (h_total_sample : total_sample = 100)
  (h_large : large_supermarkets = 200)
  (h_medium : medium_supermarkets = 400)
  (h_small : small_supermarkets = 1400) : 
  (total_sample * medium_supermarkets) / (large_supermarkets + medium_supermarkets + small_supermarkets) = 20 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l3615_361518


namespace NUMINAMATH_CALUDE_jerry_total_games_l3615_361536

/-- Calculates the total number of games Jerry has after his birthday and trade --/
def total_games_after (initial_action : ℕ) (initial_strategy : ℕ) 
  (action_increase_percent : ℕ) (strategy_increase_percent : ℕ) 
  (action_traded : ℕ) (sports_received : ℕ) : ℕ :=
  let action_increase := (initial_action * action_increase_percent) / 100
  let strategy_increase := (initial_strategy * strategy_increase_percent) / 100
  let final_action := initial_action + action_increase - action_traded
  let final_strategy := initial_strategy + strategy_increase
  final_action + final_strategy + sports_received

/-- Theorem stating that Jerry's total games after birthday and trade is 16 --/
theorem jerry_total_games : 
  total_games_after 7 5 30 20 2 3 = 16 := by sorry

end NUMINAMATH_CALUDE_jerry_total_games_l3615_361536


namespace NUMINAMATH_CALUDE_circle_M_equation_l3615_361529

-- Define the circle M
def circle_M (a r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = r^2}

-- Define the line l₁: x = -2
def line_l₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -2}

-- Define the line l₂: 2x - √5y - 4 = 0
def line_l₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - Real.sqrt 5 * p.2 - 4 = 0}

theorem circle_M_equation 
  (a : ℝ) 
  (h1 : a > -2)
  (h2 : ∃ r : ℝ, 
    -- The chord formed by the intersection of M and l₁ has length 2√3
    (3 : ℝ) + (a + 2)^2 = r^2 ∧ 
    -- M is tangent to l₂
    r = |2 * a - 4| / 3) :
  circle_M a 2 = circle_M 1 2 := by sorry

end NUMINAMATH_CALUDE_circle_M_equation_l3615_361529


namespace NUMINAMATH_CALUDE_olivia_paper_usage_l3615_361597

/-- The number of pieces of paper Olivia initially had -/
def initial_pieces : ℕ := 81

/-- The number of pieces of paper Olivia has left -/
def remaining_pieces : ℕ := 25

/-- The number of pieces of paper Olivia used -/
def used_pieces : ℕ := initial_pieces - remaining_pieces

theorem olivia_paper_usage :
  used_pieces = 56 :=
sorry

end NUMINAMATH_CALUDE_olivia_paper_usage_l3615_361597


namespace NUMINAMATH_CALUDE_problem_statement_l3615_361567

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem problem_statement (f : ℝ → ℝ) (α : ℝ) 
    (h_odd : IsOdd f)
    (h_period : HasPeriod f 5)
    (h_f_neg_three : f (-3) = 1)
    (h_tan_α : Real.tan α = 2) :
    f (20 * Real.sin α * Real.cos α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3615_361567


namespace NUMINAMATH_CALUDE_students_walking_home_l3615_361582

theorem students_walking_home (total : ℚ) (bus carpool scooter walk : ℚ) : 
  bus = 1/3 * total →
  carpool = 1/5 * total →
  scooter = 1/8 * total →
  walk = total - (bus + carpool + scooter) →
  walk = 41/120 * total := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l3615_361582


namespace NUMINAMATH_CALUDE_wanda_blocks_count_l3615_361560

/-- The total number of blocks Wanda has after receiving more blocks from Theresa -/
def total_blocks (initial : ℕ) (additional : ℕ) : ℕ := initial + additional

/-- Theorem stating that given the initial and additional blocks, Wanda has 83 blocks in total -/
theorem wanda_blocks_count : total_blocks 4 79 = 83 := by
  sorry

end NUMINAMATH_CALUDE_wanda_blocks_count_l3615_361560


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3615_361587

theorem smallest_n_square_and_cube : 
  (∀ m : ℕ, m > 0 ∧ m < 1875 → ¬(∃ a : ℕ, 3 * m = a ^ 2) ∨ ¬(∃ b : ℕ, 5 * m = b ^ 3)) ∧ 
  (∃ a : ℕ, 3 * 1875 = a ^ 2) ∧ 
  (∃ b : ℕ, 5 * 1875 = b ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3615_361587


namespace NUMINAMATH_CALUDE_proposition_truth_values_l3615_361556

theorem proposition_truth_values (p q : Prop) (h1 : ¬p) (h2 : q) :
  ¬p ∧ ¬(p ∧ q) ∧ ¬(¬q) ∧ (p ∨ q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l3615_361556


namespace NUMINAMATH_CALUDE_log_1458_between_consecutive_integers_l3615_361506

theorem log_1458_between_consecutive_integers (c d : ℤ) : 
  (c : ℝ) < Real.log 1458 / Real.log 10 ∧ 
  Real.log 1458 / Real.log 10 < (d : ℝ) ∧ 
  d = c + 1 → 
  c + d = 7 := by
sorry

end NUMINAMATH_CALUDE_log_1458_between_consecutive_integers_l3615_361506


namespace NUMINAMATH_CALUDE_farm_tax_land_percentage_l3615_361521

theorem farm_tax_land_percentage 
  (total_tax : ℝ) 
  (individual_tax : ℝ) 
  (h1 : total_tax = 3840) 
  (h2 : individual_tax = 480) :
  individual_tax / total_tax = 0.125 := by
sorry

end NUMINAMATH_CALUDE_farm_tax_land_percentage_l3615_361521


namespace NUMINAMATH_CALUDE_melies_remaining_money_l3615_361552

/-- Calculates the remaining money after Méliès buys groceries -/
theorem melies_remaining_money :
  let meat_weight : ℝ := 3.5
  let meat_price_per_kg : ℝ := 95
  let vegetable_weight : ℝ := 4
  let vegetable_price_per_kg : ℝ := 18
  let fruit_weight : ℝ := 2.5
  let fruit_price_per_kg : ℝ := 12
  let initial_money : ℝ := 450
  let total_cost : ℝ := meat_weight * meat_price_per_kg +
                        vegetable_weight * vegetable_price_per_kg +
                        fruit_weight * fruit_price_per_kg
  let remaining_money : ℝ := initial_money - total_cost
  remaining_money = 15.5 := by sorry

end NUMINAMATH_CALUDE_melies_remaining_money_l3615_361552


namespace NUMINAMATH_CALUDE_logans_average_speed_l3615_361512

/-- Prove Logan's average speed given the driving conditions of Tamika and Logan -/
theorem logans_average_speed 
  (tamika_time : ℝ) 
  (tamika_speed : ℝ) 
  (logan_time : ℝ) 
  (distance_difference : ℝ) 
  (h1 : tamika_time = 8) 
  (h2 : tamika_speed = 45) 
  (h3 : logan_time = 5) 
  (h4 : tamika_time * tamika_speed = logan_time * logan_speed + distance_difference) 
  (h5 : distance_difference = 85) : 
  logan_speed = 55 := by
  sorry

end NUMINAMATH_CALUDE_logans_average_speed_l3615_361512


namespace NUMINAMATH_CALUDE_barbara_candies_l3615_361501

/-- The number of candies Barbara has left after using some -/
def candies_left (initial : ℝ) (used : ℝ) : ℝ :=
  initial - used

theorem barbara_candies : 
  let initial_candies : ℝ := 18.0
  let used_candies : ℝ := 9.0
  candies_left initial_candies used_candies = 9.0 := by
sorry

end NUMINAMATH_CALUDE_barbara_candies_l3615_361501


namespace NUMINAMATH_CALUDE_expression_equals_one_l3615_361505

theorem expression_equals_one (x : ℝ) (h1 : x^3 ≠ 1) (h2 : x^3 ≠ -1) :
  ((x^2 + 2*x + 2)^2 * (x^4 - x^2 + 1)^2) / (x^3 + 1)^3 *
  ((x^2 - 2*x + 2)^2 * (x^4 + x^2 + 1)^2) / (x^3 - 1)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3615_361505


namespace NUMINAMATH_CALUDE_down_payment_calculation_l3615_361590

/-- Proves that the down payment is $4 given the specified conditions -/
theorem down_payment_calculation (purchase_price : ℝ) (monthly_payment : ℝ) 
  (num_payments : ℕ) (interest_rate : ℝ) (down_payment : ℝ) : 
  purchase_price = 112 →
  monthly_payment = 10 →
  num_payments = 12 →
  interest_rate = 0.10714285714285714 →
  down_payment + num_payments * monthly_payment = purchase_price * (1 + interest_rate) →
  down_payment = 4 := by
sorry

end NUMINAMATH_CALUDE_down_payment_calculation_l3615_361590


namespace NUMINAMATH_CALUDE_square_diff_over_hundred_l3615_361581

theorem square_diff_over_hundred : (2200 - 2100)^2 / 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_over_hundred_l3615_361581


namespace NUMINAMATH_CALUDE_basketball_team_wins_l3615_361532

theorem basketball_team_wins (total_games : ℕ) (win_loss_difference : ℕ) 
  (h1 : total_games = 62) 
  (h2 : win_loss_difference = 28) : 
  let games_won := (total_games + win_loss_difference) / 2
  games_won = 45 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_wins_l3615_361532


namespace NUMINAMATH_CALUDE_constrained_words_count_l3615_361571

/-- The number of possible letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- A five-letter word with the given constraints -/
structure ConstrainedWord :=
  (first : Fin alphabet_size)
  (second : Fin alphabet_size)
  (third : Fin alphabet_size)

/-- The total number of constrained words -/
def total_constrained_words : ℕ := alphabet_size ^ 3

theorem constrained_words_count :
  total_constrained_words = 17576 := by
  sorry

end NUMINAMATH_CALUDE_constrained_words_count_l3615_361571


namespace NUMINAMATH_CALUDE_max_mondays_in_51_days_l3615_361563

theorem max_mondays_in_51_days : ∀ (start_day : Nat),
  (start_day < 7) →
  (∃ (monday_count : Nat),
    monday_count = (51 / 7 : Nat) + (if start_day ≤ 1 then 1 else 0) ∧
    monday_count ≤ 8 ∧
    ∀ (other_count : Nat),
      (∃ (other_start : Nat), other_start < 7 ∧
        other_count = (51 / 7 : Nat) + (if other_start ≤ 1 then 1 else 0)) →
      other_count ≤ monday_count) :=
by sorry

end NUMINAMATH_CALUDE_max_mondays_in_51_days_l3615_361563


namespace NUMINAMATH_CALUDE_school_trip_photos_l3615_361533

theorem school_trip_photos (c : ℕ) : 
  (3 * c = c + 12) →  -- Lisa and Robert have the same number of photos
  c = 6               -- Claire took 6 photos
  := by sorry

end NUMINAMATH_CALUDE_school_trip_photos_l3615_361533


namespace NUMINAMATH_CALUDE_equation_equivalence_l3615_361559

theorem equation_equivalence (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) :
  10 * (6 * x + 14 * Real.pi) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3615_361559


namespace NUMINAMATH_CALUDE_classroom_size_l3615_361576

theorem classroom_size (x : ℕ) 
  (h1 : (11 * x : ℝ) = (10 * (x - 1) + 30 : ℝ)) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_classroom_size_l3615_361576


namespace NUMINAMATH_CALUDE_not_perfect_square_special_number_l3615_361523

/-- A 100-digit number with all digits as fives except one is not a perfect square. -/
theorem not_perfect_square_special_number : 
  ∀ n : ℕ, 
  (n ≥ 10^99 ∧ n < 10^100) →  -- 100-digit number
  (∃! d : ℕ, d < 10 ∧ d ≠ 5 ∧ 
    ∀ i : ℕ, i < 100 → 
      (n / 10^i) % 10 = if (n / 10^i) % 10 = d then d else 5) →  -- All digits are fives except one
  ¬∃ m : ℕ, n = m^2 :=  -- Not a perfect square
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_special_number_l3615_361523


namespace NUMINAMATH_CALUDE_fold_line_length_squared_fold_line_theorem_l3615_361583

/-- Represents an equilateral triangle with side length 15 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 15

/-- Represents the folded triangle -/
structure FoldedTriangle extends EquilateralTriangle where
  fold_distance : ℝ
  is_valid_fold : fold_distance = 11

/-- The theorem stating the square of the fold line length -/
theorem fold_line_length_squared (t : FoldedTriangle) : ℝ :=
  2174209 / 78281

/-- The main theorem to be proved -/
theorem fold_line_theorem (t : FoldedTriangle) : 
  fold_line_length_squared t = 2174209 / 78281 := by
  sorry

end NUMINAMATH_CALUDE_fold_line_length_squared_fold_line_theorem_l3615_361583


namespace NUMINAMATH_CALUDE_equation_solution_l3615_361530

/-- The set of solutions for the equation x! + y! = 8z + 2017 -/
def SolutionSet : Set (ℕ × ℕ × ℤ) :=
  {(1, 4, -249), (4, 1, -249), (1, 5, -237), (5, 1, -237)}

/-- The equation x! + y! = 8z + 2017 -/
def Equation (x y : ℕ) (z : ℤ) : Prop :=
  Nat.factorial x + Nat.factorial y = 8 * z + 2017

/-- z is an odd integer -/
def IsOdd (z : ℤ) : Prop :=
  ∃ k : ℤ, z = 2 * k + 1

theorem equation_solution :
  ∀ x y : ℕ, ∀ z : ℤ,
    Equation x y z ∧ IsOdd z ↔ (x, y, z) ∈ SolutionSet :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3615_361530


namespace NUMINAMATH_CALUDE_f_properties_l3615_361535

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / 4^x - 1 / 2^x else 2^x - 4^x

theorem f_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-x) = -f x) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, f x = 1 / 4^x - 1 / 2^x) ∧
  (∀ x ∈ Set.Icc 0 1, f x = 2^x - 4^x) ∧
  (∀ x ∈ Set.Icc 0 1, f x ≤ 0) ∧
  (∃ x ∈ Set.Icc 0 1, f x = 0) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l3615_361535
