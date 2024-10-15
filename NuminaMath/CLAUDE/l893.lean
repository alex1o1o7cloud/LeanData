import Mathlib

namespace NUMINAMATH_CALUDE_average_pastry_sales_l893_89339

def pastry_sales : List Nat := [2, 3, 4, 5, 6, 7, 8]

theorem average_pastry_sales : 
  (List.sum pastry_sales) / pastry_sales.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_pastry_sales_l893_89339


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l893_89340

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  sides : ℕ
  is_irregular : Bool
  is_convex : Bool
  right_angles : ℕ

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem stating that a nine-sided polygon has 27 diagonals -/
theorem nine_sided_polygon_diagonals (P : Polygon 9) 
  (h1 : P.is_irregular = true) 
  (h2 : P.is_convex = true) 
  (h3 : P.right_angles = 2) : 
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l893_89340


namespace NUMINAMATH_CALUDE_min_value_f_l893_89377

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_period (x : ℝ) : f (x + 2) = 3 * f x

axiom f_def (x : ℝ) (h : x ∈ Set.Icc 0 2) : f x = x^2 - 2*x

-- Define the theorem
theorem min_value_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-4) (-2) ∧
  f x = -1/9 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-4) (-2) → f y ≥ -1/9 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_l893_89377


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l893_89399

def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry 
  (a b c : ℝ) 
  (h_symmetry : ∀ x, p a b c x = p a b c (30 - x))
  (h_p25 : p a b c 25 = 9)
  (h_p0 : p a b c 0 = 1) :
  p a b c 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l893_89399


namespace NUMINAMATH_CALUDE_flowchart_properties_l893_89368

/-- A flowchart is a type of diagram that represents a process or algorithm. -/
def Flowchart : Type := sorry

/-- A block in a flowchart represents a step or decision in the process. -/
def Block : Type := sorry

/-- The start block of a flowchart. -/
def start_block : Block := sorry

/-- The end block of a flowchart. -/
def end_block : Block := sorry

/-- An input block in a flowchart. -/
def input_block : Block := sorry

/-- An output block in a flowchart. -/
def output_block : Block := sorry

/-- A decision block in a flowchart. -/
def decision_block : Block := sorry

/-- A function that checks if a flowchart has both start and end blocks. -/
def has_start_and_end (f : Flowchart) : Prop := sorry

/-- A function that checks if input blocks are only after the start block. -/
def input_after_start (f : Flowchart) : Prop := sorry

/-- A function that checks if output blocks are only before the end block. -/
def output_before_end (f : Flowchart) : Prop := sorry

/-- A function that checks if decision blocks are the only ones with multiple exit points. -/
def decision_multiple_exits (f : Flowchart) : Prop := sorry

/-- A function that checks if the way conditions are described in decision blocks is unique. -/
def unique_decision_conditions (f : Flowchart) : Prop := sorry

theorem flowchart_properties (f : Flowchart) :
  (has_start_and_end f ∧ 
   input_after_start f ∧ 
   output_before_end f ∧ 
   decision_multiple_exits f) ∧
  ¬(unique_decision_conditions f) := by sorry

end NUMINAMATH_CALUDE_flowchart_properties_l893_89368


namespace NUMINAMATH_CALUDE_square_cut_divisible_by_four_l893_89336

/-- A rectangle on a grid --/
structure GridRectangle where
  length : ℕ
  width : ℕ

/-- A square on a grid --/
structure GridSquare where
  side : ℕ

/-- Function to cut a square into rectangles along grid lines --/
def cutSquareIntoRectangles (square : GridSquare) : List GridRectangle :=
  sorry

/-- Function to calculate the perimeter of a rectangle --/
def rectanglePerimeter (rect : GridRectangle) : ℕ :=
  2 * (rect.length + rect.width)

theorem square_cut_divisible_by_four (square : GridSquare) 
    (h : square.side = 2009) :
    ∃ (rect : GridRectangle), rect ∈ cutSquareIntoRectangles square ∧ 
    (rectanglePerimeter rect) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_cut_divisible_by_four_l893_89336


namespace NUMINAMATH_CALUDE_mildred_weight_l893_89343

/-- Mildred's weight problem -/
theorem mildred_weight (carol_weight : ℕ) (weight_difference : ℕ) 
  (h1 : carol_weight = 9)
  (h2 : weight_difference = 50) :
  carol_weight + weight_difference = 59 := by
  sorry

end NUMINAMATH_CALUDE_mildred_weight_l893_89343


namespace NUMINAMATH_CALUDE_right_triangle_case1_right_triangle_case2_right_triangle_case3_l893_89308

-- Define a right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  right_angle : angleC = 90
  angle_sum : angleA + angleB + angleC = 180

-- Case 1
theorem right_triangle_case1 (t : RightTriangle) (h1 : t.angleB = 60) (h2 : t.a = 4) :
  t.b = 4 * Real.sqrt 3 ∧ t.c = 8 := by
  sorry

-- Case 2
theorem right_triangle_case2 (t : RightTriangle) (h1 : t.a = Real.sqrt 3 - 1) (h2 : t.b = 3 - Real.sqrt 3) :
  t.angleB = 60 ∧ t.angleA = 30 ∧ t.c = 2 * Real.sqrt 3 - 2 := by
  sorry

-- Case 3
theorem right_triangle_case3 (t : RightTriangle) (h1 : t.angleA = 60) (h2 : t.c = 2 + Real.sqrt 3) :
  t.angleB = 30 ∧ t.a = Real.sqrt 3 + 3/2 ∧ t.b = (2 + Real.sqrt 3)/2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_case1_right_triangle_case2_right_triangle_case3_l893_89308


namespace NUMINAMATH_CALUDE_conference_drinks_l893_89370

theorem conference_drinks (total : ℕ) (coffee : ℕ) (juice : ℕ) (both : ℕ) :
  total = 30 →
  coffee = 15 →
  juice = 18 →
  both = 7 →
  total - (coffee + juice - both) = 4 :=
by sorry

end NUMINAMATH_CALUDE_conference_drinks_l893_89370


namespace NUMINAMATH_CALUDE_parabola_perpendicular_range_l893_89355

/-- Given point A(0,2) and two points B and C on the parabola y^2 = x + 4 such that AB ⟂ BC,
    the y-coordinate of point C satisfies y ≤ 0 or y ≥ 4. -/
theorem parabola_perpendicular_range (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (0, 2)
  let on_parabola (p : ℝ × ℝ) := p.2^2 = p.1 + 4
  let perpendicular (p q r : ℝ × ℝ) := 
    (q.2 - p.2) * (r.2 - q.2) = -(q.1 - p.1) * (r.1 - q.1)
  on_parabola B ∧ on_parabola C ∧ perpendicular A B C →
  C.2 ≤ 0 ∨ C.2 ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_range_l893_89355


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l893_89324

/-- A function that satisfies f(xy) = f(x) + f(y) + 1 for all x and y -/
def f (x : ℝ) : ℝ := -1

/-- Theorem stating that f satisfies the given functional equation -/
theorem f_satisfies_equation (x y : ℝ) : f (x * y) = f x + f y + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l893_89324


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l893_89395

/-- Prove that (1, 1, 1) is the solution to the given system of equations -/
theorem solution_satisfies_system :
  let x₁ : ℝ := 1
  let x₂ : ℝ := 1
  let x₃ : ℝ := 1
  (x₁ + 2*x₂ + x₃ = 4) ∧
  (3*x₁ - 5*x₂ + 3*x₃ = 1) ∧
  (2*x₁ + 7*x₂ - x₃ = 8) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l893_89395


namespace NUMINAMATH_CALUDE_lottery_blank_probability_l893_89322

theorem lottery_blank_probability :
  let num_prizes : ℕ := 10
  let num_blanks : ℕ := 25
  let total_outcomes : ℕ := num_prizes + num_blanks
  (num_blanks : ℚ) / (total_outcomes : ℚ) = 5 / 7 :=
by sorry

end NUMINAMATH_CALUDE_lottery_blank_probability_l893_89322


namespace NUMINAMATH_CALUDE_sara_pumpkins_l893_89362

/-- The number of pumpkins Sara grew -/
def pumpkins_grown : ℕ := 43

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The number of pumpkins Sara has left -/
def pumpkins_left : ℕ := pumpkins_grown - pumpkins_eaten

theorem sara_pumpkins : pumpkins_left = 20 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l893_89362


namespace NUMINAMATH_CALUDE_arithmetic_proof_l893_89374

theorem arithmetic_proof : 4 * (8 - 6) - 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l893_89374


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l893_89329

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) : 
  M / N = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l893_89329


namespace NUMINAMATH_CALUDE_time_per_cut_l893_89372

/-- Given 3 pieces of wood, each cut into 3 sections, in 18 minutes total, prove the time per cut is 3 minutes -/
theorem time_per_cut (num_pieces : ℕ) (sections_per_piece : ℕ) (total_time : ℕ) :
  num_pieces = 3 →
  sections_per_piece = 3 →
  total_time = 18 →
  (total_time : ℚ) / (num_pieces * (sections_per_piece - 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_per_cut_l893_89372


namespace NUMINAMATH_CALUDE_sum_of_ten_consecutive_squares_not_perfect_square_l893_89302

theorem sum_of_ten_consecutive_squares_not_perfect_square (x : ℤ) :
  ∃ (y : ℤ), 5 * (2 * x^2 + 10 * x + 29) ≠ y^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ten_consecutive_squares_not_perfect_square_l893_89302


namespace NUMINAMATH_CALUDE_divisibility_by_x2_plus_x_plus_1_l893_89393

theorem divisibility_by_x2_plus_x_plus_1 (n : ℕ) (hn : n > 0) :
  ∃ q : Polynomial ℚ, (X + 1 : Polynomial ℚ)^(2*n + 1) + X^(n + 2) = (X^2 + X + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_x2_plus_x_plus_1_l893_89393


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l893_89371

theorem complex_magnitude_problem (a b : ℝ) (h : a^2 - 4 + b * Complex.I - Complex.I = 0) :
  Complex.abs (a + b * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l893_89371


namespace NUMINAMATH_CALUDE_arithmetic_sequence_slope_l893_89365

/-- For an arithmetic sequence {a_n} where a_2 - a_4 = 2, 
    the slope of the line containing points (n, a_n) is -1 -/
theorem arithmetic_sequence_slope (a : ℕ → ℝ) (h : a 2 - a 4 = 2) :
  ∃ b : ℝ, ∀ n : ℕ, a n = -n + b := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_slope_l893_89365


namespace NUMINAMATH_CALUDE_five_dozen_apples_cost_l893_89389

/-- The cost of a given number of dozens of apples, given the cost of two dozens. -/
def apple_cost (two_dozen_cost : ℚ) (dozens : ℚ) : ℚ :=
  (dozens / 2) * two_dozen_cost

/-- Theorem: If two dozen apples cost $15.60, then five dozen apples cost $39.00 -/
theorem five_dozen_apples_cost (two_dozen_cost : ℚ) 
  (h : two_dozen_cost = 15.6) : 
  apple_cost two_dozen_cost 5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_five_dozen_apples_cost_l893_89389


namespace NUMINAMATH_CALUDE_count_valid_numbers_l893_89333

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧  -- nine-digit number
  (∃ (digits : List ℕ), 
    digits.length = 9 ∧
    digits.count 3 = 8 ∧
    digits.count 0 = 1 ∧
    digits.foldl (λ acc d => acc * 10 + d) 0 = n)

def leaves_remainder_one (n : ℕ) : Prop :=
  n % 4 = 1

theorem count_valid_numbers : 
  (∃ (S : Finset ℕ), 
    (∀ n ∈ S, is_valid_number n ∧ leaves_remainder_one n) ∧
    S.card = 7 ∧
    (∀ n, is_valid_number n ∧ leaves_remainder_one n → n ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l893_89333


namespace NUMINAMATH_CALUDE_total_persimmons_l893_89344

/-- Given that the total weight of persimmons is 3 kg and 5 persimmons weigh 1 kg,
    prove that the total number of persimmons is 15. -/
theorem total_persimmons (total_weight : ℝ) (weight_of_five : ℝ) (num_in_five : ℕ) :
  total_weight = 3 →
  weight_of_five = 1 →
  num_in_five = 5 →
  (total_weight / weight_of_five) * num_in_five = 15 := by
  sorry

#check total_persimmons

end NUMINAMATH_CALUDE_total_persimmons_l893_89344


namespace NUMINAMATH_CALUDE_nested_f_application_l893_89391

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^3 else x + 9

theorem nested_f_application : f (f (f (f (f 3)))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_nested_f_application_l893_89391


namespace NUMINAMATH_CALUDE_least_n_for_product_exceeding_million_l893_89314

theorem least_n_for_product_exceeding_million (n : ℕ) : 
  (∀ k < 23, (2 : ℝ) ^ ((k * (k + 1)) / 26) ≤ 1000000) ∧
  (2 : ℝ) ^ ((23 * 24) / 26) > 1000000 := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_product_exceeding_million_l893_89314


namespace NUMINAMATH_CALUDE_tan_315_degrees_l893_89328

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l893_89328


namespace NUMINAMATH_CALUDE_fraction_power_product_l893_89303

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l893_89303


namespace NUMINAMATH_CALUDE_paytons_score_l893_89315

theorem paytons_score (total_students : ℕ) (students_without_payton : ℕ) 
  (avg_without_payton : ℝ) (avg_with_payton : ℝ) :
  total_students = 15 →
  students_without_payton = 14 →
  avg_without_payton = 80 →
  avg_with_payton = 81 →
  (students_without_payton * avg_without_payton + 
    (total_students - students_without_payton) * 
    ((total_students * avg_with_payton - students_without_payton * avg_without_payton) / 
    (total_students - students_without_payton))) / total_students = avg_with_payton →
  (total_students * avg_with_payton - students_without_payton * avg_without_payton) / 
  (total_students - students_without_payton) = 95 := by
sorry

end NUMINAMATH_CALUDE_paytons_score_l893_89315


namespace NUMINAMATH_CALUDE_fraction_simplification_l893_89382

theorem fraction_simplification (d : ℝ) : 
  (5 + 4*d) / 9 - 3 + 1/3 = (4*d - 19) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l893_89382


namespace NUMINAMATH_CALUDE_angle_bisector_length_l893_89356

/-- Given a triangle ABC, this theorem states that the length of the angle bisector
    from vertex C to the opposite side AB can be calculated using the formula:
    l₃ = (2ab)/(a+b) * cos(C/2), where a and b are the lengths of sides BC and AC
    respectively, and C is the angle at vertex C. -/
theorem angle_bisector_length (a b C l₃ : ℝ) :
  (a > 0) → (b > 0) → (C > 0) → (C < π) →
  l₃ = (2 * a * b) / (a + b) * Real.cos (C / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l893_89356


namespace NUMINAMATH_CALUDE_base_k_conversion_l893_89311

theorem base_k_conversion (k : ℕ) (h : k > 0) : 
  (1 * k^2 + 3 * k + 2 = 30) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_k_conversion_l893_89311


namespace NUMINAMATH_CALUDE_symmetry_implies_phase_shift_l893_89354

/-- Given a function f(x) = sin x + √3 cos x, prove that if y = f(x + φ) is symmetric about x = 0, then φ = π/6 -/
theorem symmetry_implies_phase_shift (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin x + Real.sqrt 3 * Real.cos x) →
  (∀ x, f (x + φ) = f (-x + φ)) →
  φ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_phase_shift_l893_89354


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_seven_l893_89327

theorem power_of_three_plus_five_mod_seven : (3^90 + 5) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_seven_l893_89327


namespace NUMINAMATH_CALUDE_product_change_l893_89386

theorem product_change (a b : ℕ) (h : (a + 3) * (b - 3) - a * b = 600) : 
  a * b - (a - 3) * (b + 3) = 618 := by
sorry

end NUMINAMATH_CALUDE_product_change_l893_89386


namespace NUMINAMATH_CALUDE_twin_primes_sum_divisible_by_12_l893_89376

theorem twin_primes_sum_divisible_by_12 (p : ℕ) (h1 : p > 3) (h2 : Prime p) (h3 : Prime (p + 2)) :
  12 ∣ (p + (p + 2)) :=
sorry

end NUMINAMATH_CALUDE_twin_primes_sum_divisible_by_12_l893_89376


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l893_89348

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l893_89348


namespace NUMINAMATH_CALUDE_man_speed_is_4_l893_89367

/-- Represents the speed of water in a stream. -/
def stream_speed : ℝ := sorry

/-- Represents the speed of a man swimming in still water. -/
def man_speed : ℝ := sorry

/-- The distance traveled downstream. -/
def downstream_distance : ℝ := 30

/-- The distance traveled upstream. -/
def upstream_distance : ℝ := 18

/-- The time taken for both downstream and upstream swims. -/
def swim_time : ℝ := 6

/-- Theorem stating that the man's speed in still water is 4 km/h. -/
theorem man_speed_is_4 : 
  downstream_distance = (man_speed + stream_speed) * swim_time ∧ 
  upstream_distance = (man_speed - stream_speed) * swim_time → 
  man_speed = 4 := by sorry

end NUMINAMATH_CALUDE_man_speed_is_4_l893_89367


namespace NUMINAMATH_CALUDE_circle_graph_percentage_l893_89305

theorem circle_graph_percentage (sector_degrees : ℝ) (total_degrees : ℝ) 
  (h1 : sector_degrees = 18)
  (h2 : total_degrees = 360) :
  (sector_degrees / total_degrees) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_circle_graph_percentage_l893_89305


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l893_89312

theorem difference_of_squares_factorization (x : ℝ) : 9 - 4 * x^2 = (3 - 2*x) * (3 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l893_89312


namespace NUMINAMATH_CALUDE_parabola_equation_l893_89375

/-- Represents a parabola with vertex at the origin and coordinate axes as axes of symmetry -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = -2*a*x

/-- The parabola passes through the given point -/
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

theorem parabola_equation : 
  ∃ (p : Parabola), passes_through p (-2) (-4) ∧ p.a = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l893_89375


namespace NUMINAMATH_CALUDE_farm_field_area_l893_89369

/-- Represents the farm field ploughing scenario -/
structure FarmField where
  planned_daily_rate : ℝ
  actual_daily_rate : ℝ
  extra_days : ℕ
  remaining_area : ℝ

/-- Calculates the total area of the farm field -/
def total_area (f : FarmField) : ℝ :=
  sorry

/-- Theorem stating that the total area of the farm field is 312 hectares -/
theorem farm_field_area (f : FarmField) 
  (h1 : f.planned_daily_rate = 260)
  (h2 : f.actual_daily_rate = 85)
  (h3 : f.extra_days = 2)
  (h4 : f.remaining_area = 40) :
  total_area f = 312 :=
sorry

end NUMINAMATH_CALUDE_farm_field_area_l893_89369


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l893_89349

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_x_value :
  let A : Point := ⟨3, -2⟩
  let B : Point := ⟨-9, 4⟩
  let C : Point := ⟨x, 0⟩
  collinear A B C → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l893_89349


namespace NUMINAMATH_CALUDE_incorrect_calculation_l893_89384

theorem incorrect_calculation (x : ℝ) : 
  25 * ((1/25) * x^2 - (1/10) * x + 1) ≠ x^2 - (5/2) * x + 25 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l893_89384


namespace NUMINAMATH_CALUDE_rectangle_width_range_l893_89316

/-- Given a wire of length 20 cm shaped into a rectangle with length at least 6 cm,
    prove that the width x satisfies 0 < x ≤ 20/3 -/
theorem rectangle_width_range :
  ∀ x : ℝ,
  (∃ l : ℝ, l ≥ 6 ∧ 2 * (x + l) = 20) →
  (0 < x ∧ x ≤ 20 / 3) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_range_l893_89316


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_l893_89334

/-- The probability of selecting at least one female student when choosing two students from a group of three male and two female students is 7/10. -/
theorem prob_at_least_one_female (total : ℕ) (male : ℕ) (female : ℕ) (select : ℕ) :
  total = male + female →
  total = 5 →
  male = 3 →
  female = 2 →
  select = 2 →
  (1 : ℚ) - (Nat.choose male select : ℚ) / (Nat.choose total select : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_female_l893_89334


namespace NUMINAMATH_CALUDE_product_five_cubed_sum_l893_89390

theorem product_five_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 5^3 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 31 := by sorry

end NUMINAMATH_CALUDE_product_five_cubed_sum_l893_89390


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l893_89306

-- Define the function f
def f (x m : ℝ) : ℝ := |2 * x - m|

-- State the theorem
theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, f x m ≤ 6 ↔ -2 ≤ x ∧ x ≤ 4) →
  (m = 2 ∧
   ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 2 →
     (∀ x : ℝ, f x m + f ((1/2) * x + 3) m ≤ 8/a + 2/b ↔ -3 ≤ x ∧ x ≤ 7/3)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l893_89306


namespace NUMINAMATH_CALUDE_evaluate_expression_l893_89383

theorem evaluate_expression : 15 * 30 + 45 * 15 - 15 * 10 = 975 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l893_89383


namespace NUMINAMATH_CALUDE_p_tilde_at_two_l893_89350

def p (x : ℝ) : ℝ := x^2 - 4*x + 3

def p_tilde (x : ℝ) : ℝ := p (p x)

theorem p_tilde_at_two : p_tilde 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_p_tilde_at_two_l893_89350


namespace NUMINAMATH_CALUDE_remaining_students_l893_89347

/-- The number of remaining students in the class -/
def n : ℕ := sorry

/-- The weight of the student who left the class -/
def weight_left : ℝ := 45

/-- The increase in average weight after the student left -/
def weight_increase : ℝ := 0.2

/-- The average weight of the remaining students -/
def avg_weight_remaining : ℝ := 57

/-- Theorem stating that the number of remaining students is 59 -/
theorem remaining_students : n = 59 := by
  sorry

end NUMINAMATH_CALUDE_remaining_students_l893_89347


namespace NUMINAMATH_CALUDE_remainder_theorem_l893_89331

theorem remainder_theorem : (43^43 + 43) % 44 = 42 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l893_89331


namespace NUMINAMATH_CALUDE_joe_hvac_cost_per_vent_l893_89304

/-- The cost per vent of an HVAC system -/
def cost_per_vent (total_cost : ℕ) (num_zones : ℕ) (vents_per_zone : ℕ) : ℚ :=
  total_cost / (num_zones * vents_per_zone)

/-- Theorem: The cost per vent of Joe's HVAC system is $2,000 -/
theorem joe_hvac_cost_per_vent :
  cost_per_vent 20000 2 5 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_joe_hvac_cost_per_vent_l893_89304


namespace NUMINAMATH_CALUDE_function_values_and_range_l893_89373

noncomputable def f (b c x : ℝ) : ℝ := -1/3 * x^3 + b * x^2 + c * x + b * c

noncomputable def g (a x : ℝ) : ℝ := a * x^2 - 2 * Real.log x

theorem function_values_and_range :
  ∀ b c : ℝ,
  (∃ x : ℝ, f b c x = -4/3 ∧ ∀ y : ℝ, f b c y ≤ f b c x) →
  (b = -1 ∧ c = 3) ∧
  ∀ a : ℝ,
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 3 ∧ 0 < x₂ ∧ x₂ < 3 ∧ |f b c x₁ - g a x₂| < 1) →
  (2 * Real.log 3 - 13) / 9 ≤ a ∧ a ≤ (6 * Real.log 3 - 1) / 27 :=
by sorry

end NUMINAMATH_CALUDE_function_values_and_range_l893_89373


namespace NUMINAMATH_CALUDE_grocery_shop_sales_l893_89323

theorem grocery_shop_sales (sales1 sales3 sales4 sales5 sales6 : ℕ) 
  (h1 : sales1 = 6335)
  (h3 : sales3 = 7230)
  (h4 : sales4 = 6562)
  (h5 : sales5 = 6855)
  (h6 : sales6 = 5091)
  (h_avg : (sales1 + sales3 + sales4 + sales5 + sales6 + 6927) / 6 = 6500) :
  ∃ sales2 : ℕ, sales2 = 6927 := by
  sorry

end NUMINAMATH_CALUDE_grocery_shop_sales_l893_89323


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l893_89396

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- Definition of the foci -/
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-4, 0) ∧ F₂ = (4, 0)

/-- Theorem: Perimeter of triangle PF₁F₂ is 18 for any point P on the ellipse -/
theorem ellipse_triangle_perimeter 
  (x y : ℝ) 
  (F₁ F₂ : ℝ × ℝ) 
  (h_ellipse : is_on_ellipse x y) 
  (h_foci : foci F₁ F₂) : 
  let P := (x, y)
  ‖P - F₁‖ + ‖P - F₂‖ + ‖F₁ - F₂‖ = 18 :=
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l893_89396


namespace NUMINAMATH_CALUDE_three_digit_numbers_decreasing_by_factor_of_six_l893_89361

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ (a b c : ℕ), 
    n = 100 * a + 10 * b + c ∧
    a ≠ 0 ∧
    10 * b + c = (100 * a + 10 * b + c) / 6

theorem three_digit_numbers_decreasing_by_factor_of_six : 
  {n : ℕ | is_valid_number n} = {120, 240, 360, 480} := by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_decreasing_by_factor_of_six_l893_89361


namespace NUMINAMATH_CALUDE_zainab_works_two_hours_per_day_l893_89301

/-- Represents Zainab's flyer distribution job --/
structure FlyerJob where
  hourly_rate : ℕ
  days_per_week : ℕ
  total_weeks : ℕ
  total_earnings : ℕ

/-- Calculates the number of hours worked per day --/
def hours_per_day (job : FlyerJob) : ℚ :=
  job.total_earnings / (job.hourly_rate * job.days_per_week * job.total_weeks)

/-- Theorem stating that Zainab works 2 hours per day --/
theorem zainab_works_two_hours_per_day :
  let job := FlyerJob.mk 2 3 4 96
  hours_per_day job = 2 := by sorry

end NUMINAMATH_CALUDE_zainab_works_two_hours_per_day_l893_89301


namespace NUMINAMATH_CALUDE_acme_savings_threshold_l893_89335

/-- Acme T-Shirt Plus Company's pricing structure -/
def acme_cost (x : ℕ) : ℚ := 75 + 8 * x

/-- Gamma T-shirt Company's pricing structure -/
def gamma_cost (x : ℕ) : ℚ := 12 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme_savings : ℕ := 19

theorem acme_savings_threshold :
  (∀ x : ℕ, x ≥ min_shirts_for_acme_savings → acme_cost x < gamma_cost x) ∧
  (∀ x : ℕ, x < min_shirts_for_acme_savings → acme_cost x ≥ gamma_cost x) :=
sorry

end NUMINAMATH_CALUDE_acme_savings_threshold_l893_89335


namespace NUMINAMATH_CALUDE_complex_division_simplification_l893_89345

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  (2 * i) / (1 + i) = 1 + i :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l893_89345


namespace NUMINAMATH_CALUDE_second_difference_quadratic_constant_second_difference_implies_A_second_difference_one_implies_A_half_second_difference_seven_implies_A_seven_half_l893_89364

/-- Second difference of a function f at point n -/
def secondDifference (f : ℕ → ℚ) (n : ℕ) : ℚ :=
  f (n + 2) - 2 * f (n + 1) + f n

/-- Quadratic function with rational coefficients -/
def quadraticFunction (A B C : ℚ) (n : ℕ) : ℚ :=
  A * n^2 + B * n + C

theorem second_difference_quadratic (A B C : ℚ) :
  ∀ n : ℕ, secondDifference (quadraticFunction A B C) n = 2 * A :=
sorry

theorem constant_second_difference_implies_A (A B C k : ℚ) :
  (∀ n : ℕ, secondDifference (quadraticFunction A B C) n = k) → A = k / 2 :=
sorry

theorem second_difference_one_implies_A_half :
  ∀ A B C : ℚ,
  (∀ n : ℕ, secondDifference (quadraticFunction A B C) n = 1) →
  A = 1 / 2 :=
sorry

theorem second_difference_seven_implies_A_seven_half :
  ∀ A B C : ℚ,
  (∀ n : ℕ, secondDifference (quadraticFunction A B C) n = 7) →
  A = 7 / 2 :=
sorry

end NUMINAMATH_CALUDE_second_difference_quadratic_constant_second_difference_implies_A_second_difference_one_implies_A_half_second_difference_seven_implies_A_seven_half_l893_89364


namespace NUMINAMATH_CALUDE_total_prank_combinations_l893_89309

/-- The number of different combinations of people Tim could involve in the prank --/
def prank_combinations (day1 day2 day3 day4 day5 : ℕ) : ℕ :=
  day1 * day2 * day3 * day4 * day5

/-- Theorem stating the total number of different combinations for Tim's prank --/
theorem total_prank_combinations :
  prank_combinations 1 2 5 4 1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_prank_combinations_l893_89309


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l893_89357

theorem min_value_of_sum_of_squares (x y : ℝ) (h : 2 * (x^2 + y^2) = x^2 + y + x*y) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b : ℝ), 2 * (a^2 + b^2) = a^2 + b + a*b → x^2 + y^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l893_89357


namespace NUMINAMATH_CALUDE_stool_height_is_80_l893_89300

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height : ℝ :=
  let ceiling_height : ℝ := 300
  let light_bulb_below_ceiling : ℝ := 15
  let alice_height : ℝ := 150
  let alice_reach : ℝ := 50
  let decoration_below_bulb : ℝ := 5
  let light_bulb_height : ℝ := ceiling_height - light_bulb_below_ceiling
  let effective_reach_height : ℝ := light_bulb_height - decoration_below_bulb
  effective_reach_height - (alice_height + alice_reach)

theorem stool_height_is_80 :
  stool_height = 80 := by
  sorry

end NUMINAMATH_CALUDE_stool_height_is_80_l893_89300


namespace NUMINAMATH_CALUDE_quadrilateral_side_sum_l893_89319

/-- Represents a quadrilateral with side lengths a, b, c, d --/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Predicate to check if angles are in arithmetic progression --/
def angles_in_arithmetic_progression (q : Quadrilateral) : Prop :=
  sorry

/-- Predicate to check if the largest side is opposite the largest angle --/
def largest_side_opposite_largest_angle (q : Quadrilateral) : Prop :=
  sorry

/-- The main theorem --/
theorem quadrilateral_side_sum (q : Quadrilateral) 
  (h1 : angles_in_arithmetic_progression q)
  (h2 : largest_side_opposite_largest_angle q)
  (h3 : q.a = 7)
  (h4 : q.b = 8)
  (h5 : ∃ (a b c : ℕ), q.c = a + Real.sqrt b + Real.sqrt c ∧ a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (a b c : ℕ), q.c = a + Real.sqrt b + Real.sqrt c ∧ a + b + c = 113 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_side_sum_l893_89319


namespace NUMINAMATH_CALUDE_annika_hiking_rate_l893_89363

/-- Annika's hiking problem -/
theorem annika_hiking_rate (initial_distance : Real) (total_east_distance : Real) (return_time : Real) :
  initial_distance = 2.75 →
  total_east_distance = 3.625 →
  return_time = 45 →
  let additional_east := total_east_distance - initial_distance
  let total_distance := initial_distance + 2 * additional_east
  total_distance / return_time * 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_annika_hiking_rate_l893_89363


namespace NUMINAMATH_CALUDE_a_formula_S_formula_min_t_value_l893_89378

-- Define the arithmetic sequence and its sum
def a (n : ℕ) : ℚ := sorry
def S (n : ℕ) : ℚ := sorry

-- Define conditions
axiom S_9 : S 9 = 90
axiom S_15 : S 15 = 240

-- Define b_n and its sum
def b (n : ℕ) : ℚ := 1 / (2 * n * (n + 1))
def S_b (n : ℕ) : ℚ := (1 / 2) * (1 - 1 / (n + 1))

-- Theorem statements
theorem a_formula (n : ℕ) : a n = 2 * n := sorry

theorem S_formula (n : ℕ) : S n = n * (n + 1) := sorry

theorem min_t_value : 
  ∀ t : ℚ, (∀ n : ℕ, n > 0 → S_b n < t) → t ≥ 1/2 := sorry

end NUMINAMATH_CALUDE_a_formula_S_formula_min_t_value_l893_89378


namespace NUMINAMATH_CALUDE_triangle_side_length_l893_89307

theorem triangle_side_length (A B C : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let cos_C := (AB^2 + AC^2 - BC^2) / (2 * AB * AC)
  AB = Real.sqrt 5 ∧ AC = 5 ∧ cos_C = 9/10 →
  BC = 4 ∨ BC = 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l893_89307


namespace NUMINAMATH_CALUDE_total_books_collected_l893_89380

def north_america : ℕ := 581
def south_america : ℕ := 435
def africa : ℕ := 524
def europe : ℕ := 688
def australia : ℕ := 319
def asia : ℕ := 526
def antarctica : ℕ := 276

theorem total_books_collected :
  north_america + south_america + africa + europe + australia + asia + antarctica = 3349 := by
  sorry

end NUMINAMATH_CALUDE_total_books_collected_l893_89380


namespace NUMINAMATH_CALUDE_john_started_five_days_ago_l893_89338

/-- Represents the number of days John has worked -/
def days_worked : ℕ := sorry

/-- Represents the daily wage John earns -/
def daily_wage : ℚ := sorry

/-- The total amount John has earned so far -/
def current_earnings : ℚ := 250

/-- The number of additional days John needs to work -/
def additional_days : ℕ := 10

theorem john_started_five_days_ago :
  days_worked = 5 ∧
  daily_wage * days_worked = current_earnings ∧
  daily_wage * (days_worked + additional_days) = 2 * current_earnings :=
sorry

end NUMINAMATH_CALUDE_john_started_five_days_ago_l893_89338


namespace NUMINAMATH_CALUDE_representation_2015_l893_89366

theorem representation_2015 : ∃ (a b c : ℤ), 
  a + b + c = 2015 ∧ 
  Nat.Prime a.natAbs ∧ 
  ∃ (k : ℤ), b = 3 * k ∧
  400 < c ∧ c < 500 ∧
  ¬∃ (m : ℤ), c = 3 * m :=
sorry

end NUMINAMATH_CALUDE_representation_2015_l893_89366


namespace NUMINAMATH_CALUDE_earnings_exceed_goal_l893_89381

/-- Represents the berry-picking job scenario --/
structure BerryPicking where
  lingonberry_rate : ℝ
  cloudberry_rate : ℝ
  blueberry_rate : ℝ
  monday_lingonberry : ℝ
  monday_cloudberry : ℝ
  monday_blueberry : ℝ
  tuesday_lingonberry_factor : ℝ
  tuesday_cloudberry_factor : ℝ
  tuesday_blueberry : ℝ
  goal : ℝ

/-- Calculates the total earnings for Monday and Tuesday --/
def total_earnings (job : BerryPicking) : ℝ :=
  let monday_earnings := 
    job.lingonberry_rate * job.monday_lingonberry +
    job.cloudberry_rate * job.monday_cloudberry +
    job.blueberry_rate * job.monday_blueberry
  let tuesday_earnings := 
    job.lingonberry_rate * (job.tuesday_lingonberry_factor * job.monday_lingonberry) +
    job.cloudberry_rate * (job.tuesday_cloudberry_factor * job.monday_cloudberry) +
    job.blueberry_rate * job.tuesday_blueberry
  monday_earnings + tuesday_earnings

/-- Theorem: Steve's earnings exceed his goal after two days --/
theorem earnings_exceed_goal (job : BerryPicking) 
  (h1 : job.lingonberry_rate = 2)
  (h2 : job.cloudberry_rate = 3)
  (h3 : job.blueberry_rate = 5)
  (h4 : job.monday_lingonberry = 8)
  (h5 : job.monday_cloudberry = 10)
  (h6 : job.monday_blueberry = 0)
  (h7 : job.tuesday_lingonberry_factor = 3)
  (h8 : job.tuesday_cloudberry_factor = 2)
  (h9 : job.tuesday_blueberry = 5)
  (h10 : job.goal = 150) :
  total_earnings job > job.goal := by
  sorry

end NUMINAMATH_CALUDE_earnings_exceed_goal_l893_89381


namespace NUMINAMATH_CALUDE_find_n_l893_89351

theorem find_n : ∃ n : ℤ, (11 : ℝ) ^ (4 * n) = (1 / 11 : ℝ) ^ (n - 30) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l893_89351


namespace NUMINAMATH_CALUDE_prob_queen_first_three_cards_l893_89325

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : Type := Unit

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The probability of drawing at least one Queen in the first three cards -/
def prob_at_least_one_queen (d : StandardDeck) : ℚ :=
  1 - (deck_size - num_queens) * (deck_size - num_queens - 1) * (deck_size - num_queens - 2) /
      (deck_size * (deck_size - 1) * (deck_size - 2))

theorem prob_queen_first_three_cards :
  ∀ d : StandardDeck, prob_at_least_one_queen d = 2174 / 10000 :=
by sorry

end NUMINAMATH_CALUDE_prob_queen_first_three_cards_l893_89325


namespace NUMINAMATH_CALUDE_emma_account_balance_l893_89320

def remaining_balance (initial_balance : ℕ) (daily_spending : ℕ) (days : ℕ) (bill_denomination : ℕ) : ℕ :=
  let balance_after_spending := initial_balance - daily_spending * days
  let withdrawal_amount := (balance_after_spending / bill_denomination) * bill_denomination
  balance_after_spending - withdrawal_amount

theorem emma_account_balance :
  remaining_balance 100 8 7 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_emma_account_balance_l893_89320


namespace NUMINAMATH_CALUDE_square_rotation_octagon_l893_89394

/-- Represents a regular polygon with n sides -/
structure RegularPolygon where
  sides : ℕ
  mk_sides_pos : sides > 0

/-- Represents a square -/
structure Square

/-- Represents the position of an object on a square -/
inductive Position
  | Top
  | Right
  | Bottom
  | Left

/-- Calculates the inner angle of a regular polygon -/
def inner_angle (p : RegularPolygon) : ℚ :=
  (p.sides - 2 : ℚ) * 180 / p.sides

/-- Calculates the rotation per movement when a square rolls around a regular polygon -/
def rotation_per_movement (p : RegularPolygon) : ℚ :=
  360 - (inner_angle p + 90)

/-- Theorem: After a full rotation around an octagon, an object on a square returns to its original position -/
theorem square_rotation_octagon (s : Square) (initial_pos : Position) :
  let octagon : RegularPolygon := ⟨8, by norm_num⟩
  let total_rotation : ℚ := 8 * rotation_per_movement octagon
  total_rotation % 360 = 0 → initial_pos = Position.Bottom → initial_pos = Position.Bottom :=
by
  sorry


end NUMINAMATH_CALUDE_square_rotation_octagon_l893_89394


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l893_89387

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (Real.sqrt 3, 1))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a + 2 • b))^2 + (Prod.snd (a + 2 • b))^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l893_89387


namespace NUMINAMATH_CALUDE_chord_length_l893_89353

-- Define the circle and chord
def circle_radius : ℝ := 5
def center_to_chord : ℝ := 4

-- Theorem statement
theorem chord_length :
  ∀ (chord_length : ℝ),
  circle_radius = 5 ∧
  center_to_chord = 4 →
  chord_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l893_89353


namespace NUMINAMATH_CALUDE_product_H₁_H₂_is_square_l893_89332

/-- For any positive integer n, H₁ is the set of odd numbers from 1 to 2n-1 -/
def H₁ (n : ℕ+) : Finset ℕ :=
  Finset.range n |>.image (fun i => 2*i + 1)

/-- For any positive integers n and k, H₂ is the set obtained by adding k to each element of H₁ -/
def H₂ (n : ℕ+) (k : ℕ+) : Finset ℕ :=
  H₁ n |>.image (fun x => x + k)

/-- The product of all elements in the union of H₁ and H₂ -/
def product_H₁_H₂ (n : ℕ+) (k : ℕ+) : ℕ :=
  (H₁ n ∪ H₂ n k).prod id

/-- For any positive integer n, when k = 2n + 1, the product of all elements in H₁ ∪ H₂ is a perfect square -/
theorem product_H₁_H₂_is_square (n : ℕ+) :
  ∃ m : ℕ, product_H₁_H₂ n (2*n + 1) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_product_H₁_H₂_is_square_l893_89332


namespace NUMINAMATH_CALUDE_integral_ln_sin_x_l893_89321

theorem integral_ln_sin_x (x : ℝ) : 
  ∫ x in (0)..(π/2), Real.log (Real.sin x) = -(π/2) * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_integral_ln_sin_x_l893_89321


namespace NUMINAMATH_CALUDE_sum_remainder_zero_l893_89397

theorem sum_remainder_zero : (9152 + 9153 + 9154 + 9155 + 9156) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_zero_l893_89397


namespace NUMINAMATH_CALUDE_inequality_solution_range_l893_89317

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → (a < 1 ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l893_89317


namespace NUMINAMATH_CALUDE_inequality_proof_l893_89326

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 + x / y)^3 + (1 + y / x)^3 ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l893_89326


namespace NUMINAMATH_CALUDE_percentage_unsold_books_l893_89341

def initial_stock : ℕ := 900
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

theorem percentage_unsold_books :
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales
  let unsold_books := initial_stock - total_sales
  (unsold_books : ℚ) / initial_stock * 100 = 55.33 := by
  sorry

end NUMINAMATH_CALUDE_percentage_unsold_books_l893_89341


namespace NUMINAMATH_CALUDE_prism_volume_l893_89318

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the properties of our specific rectangular prism -/
def RectangularPrism (d : PrismDimensions) : Prop :=
  d.x * d.y = 18 ∧ 
  d.y * d.z = 12 ∧ 
  d.x * d.z = 8 ∧
  d.y = 2 * min d.x d.z

theorem prism_volume (d : PrismDimensions) 
  (h : RectangularPrism d) : d.x * d.y * d.z = 16 := by
  sorry

#check prism_volume

end NUMINAMATH_CALUDE_prism_volume_l893_89318


namespace NUMINAMATH_CALUDE_adjusted_work_hours_sufficient_l893_89379

/-- Proves that working 27 hours per week for 9 weeks will result in at least $3000 earnings,
    given the initial plan of 20 hours per week for 12 weeks to earn $3000. -/
theorem adjusted_work_hours_sufficient
  (initial_hours_per_week : ℕ)
  (initial_weeks : ℕ)
  (target_earnings : ℕ)
  (missed_weeks : ℕ)
  (adjusted_hours_per_week : ℕ)
  (h1 : initial_hours_per_week = 20)
  (h2 : initial_weeks = 12)
  (h3 : target_earnings = 3000)
  (h4 : missed_weeks = 3)
  (h5 : adjusted_hours_per_week = 27) :
  (adjusted_hours_per_week : ℚ) * (initial_weeks - missed_weeks) ≥ (target_earnings : ℚ) := by
  sorry

#check adjusted_work_hours_sufficient

end NUMINAMATH_CALUDE_adjusted_work_hours_sufficient_l893_89379


namespace NUMINAMATH_CALUDE_revenue_change_l893_89388

/-- Given a price increase of 80% and a sales decrease of 35%, prove that revenue increases by 17% -/
theorem revenue_change (P Q : ℝ) (h_P : P > 0) (h_Q : Q > 0) : 
  let R := P * Q
  let P_new := P * (1 + 0.80)
  let Q_new := Q * (1 - 0.35)
  let R_new := P_new * Q_new
  (R_new - R) / R = 0.17 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l893_89388


namespace NUMINAMATH_CALUDE_cos_double_angle_fourth_quadrant_l893_89342

/-- Prove that for an angle in the fourth quadrant, if the sum of coordinates of its terminal point on the unit circle is -1/3, then cos 2θ = -√17/9 -/
theorem cos_double_angle_fourth_quadrant (θ : ℝ) (x₀ y₀ : ℝ) :
  (π < θ ∧ θ < 2*π) →  -- θ is in the fourth quadrant
  x₀^2 + y₀^2 = 1 →    -- point (x₀, y₀) is on the unit circle
  x₀ = Real.cos θ →    -- x₀ is the cosine of θ
  y₀ = Real.sin θ →    -- y₀ is the sine of θ
  x₀ + y₀ = -1/3 →     -- sum of coordinates is -1/3
  Real.cos (2*θ) = -Real.sqrt 17 / 9 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_fourth_quadrant_l893_89342


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l893_89385

/-- Represents a college with a name and number of officers -/
structure College where
  name : String
  officers : Nat

/-- Represents the result of stratified sampling -/
structure SamplingResult where
  m : Nat
  n : Nat
  s : Nat

/-- Calculates the stratified sampling result -/
def stratifiedSampling (colleges : List College) (totalSample : Nat) : SamplingResult :=
  sorry

/-- Calculates the probability of selecting two officers from the same college -/
def probabilitySameCollege (result : SamplingResult) : Rat :=
  sorry

theorem stratified_sampling_theorem (m n s : College) (h1 : m.officers = 36) (h2 : n.officers = 24) (h3 : s.officers = 12) :
  let colleges := [m, n, s]
  let result := stratifiedSampling colleges 6
  result.m = 3 ∧ result.n = 2 ∧ result.s = 1 ∧ probabilitySameCollege result = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l893_89385


namespace NUMINAMATH_CALUDE_pencil_color_fraction_l893_89313

theorem pencil_color_fraction (total_length : ℝ) (green_fraction : ℝ) (white_fraction : ℝ) :
  total_length = 2 →
  green_fraction = 7 / 10 →
  white_fraction = 1 / 2 →
  (total_length - green_fraction * total_length) / 2 = 
  (1 - white_fraction) * (total_length - green_fraction * total_length) :=
by sorry

end NUMINAMATH_CALUDE_pencil_color_fraction_l893_89313


namespace NUMINAMATH_CALUDE_election_votes_l893_89360

theorem election_votes (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    loser_votes = (30 * total_votes) / 100 ∧
    winner_votes - loser_votes = 180) →
  total_votes = 450 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l893_89360


namespace NUMINAMATH_CALUDE_bicycle_sales_cost_price_l893_89346

theorem bicycle_sales_cost_price 
  (profit_A_to_B : Real) 
  (profit_B_to_C : Real) 
  (final_price : Real) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  final_price = 225 →
  ∃ (initial_cost : Real) (profit_C_to_D : Real),
    initial_cost = 150 ∧
    final_price = initial_cost * (1 + profit_A_to_B) * (1 + profit_B_to_C) * (1 + profit_C_to_D) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_sales_cost_price_l893_89346


namespace NUMINAMATH_CALUDE_ellipse_area_l893_89398

/-- The area of an ellipse with semi-major axis a and semi-minor axis b -/
theorem ellipse_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∫ x in -a..a, 2 * b * Real.sqrt (1 - x^2 / a^2)) = π * a * b :=
sorry

end NUMINAMATH_CALUDE_ellipse_area_l893_89398


namespace NUMINAMATH_CALUDE_sum_45_52_base4_l893_89310

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number in base 10 -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_45_52_base4 : 
  toBase4 (45 + 52) = [1, 2, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_45_52_base4_l893_89310


namespace NUMINAMATH_CALUDE_log_ratio_squared_l893_89337

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hx_neq : x ≠ 1) (hy_neq : y ≠ 1)
  (h_log : Real.log x / Real.log 2 = Real.log 16 / Real.log y)
  (h_prod : x * y = 64) : 
  ((Real.log x - Real.log y) / Real.log 2)^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l893_89337


namespace NUMINAMATH_CALUDE_estate_distribution_theorem_l893_89330

/-- Represents the estate distribution problem --/
structure EstateDistribution where
  total : ℚ
  daughter_share : ℚ
  son_share : ℚ
  wife_share : ℚ
  nephew_share : ℚ
  gardener_share : ℚ

/-- Theorem stating the conditions and the result to be proved --/
theorem estate_distribution_theorem (e : EstateDistribution) : 
  e.daughter_share + e.son_share = (2 : ℚ) / (3 : ℚ) * e.total ∧ 
  e.daughter_share = (5 : ℚ) / (9 : ℚ) * ((2 : ℚ) / (3 : ℚ) * e.total) ∧
  e.son_share = (4 : ℚ) / (9 : ℚ) * ((2 : ℚ) / (3 : ℚ) * e.total) ∧
  e.wife_share = 3 * e.son_share ∧
  e.nephew_share = 1000 ∧
  e.gardener_share = 600 ∧
  e.total = e.daughter_share + e.son_share + e.wife_share + e.nephew_share + e.gardener_share
  →
  e.total = 2880 := by
  sorry


end NUMINAMATH_CALUDE_estate_distribution_theorem_l893_89330


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l893_89358

/-- The time taken for a train to completely cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed : ℝ) 
  (h1 : train_length = 400) 
  (h2 : bridge_length = 300) 
  (h3 : train_speed = 55.99999999999999) : 
  (train_length + bridge_length) / train_speed = (400 + 300) / 55.99999999999999 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l893_89358


namespace NUMINAMATH_CALUDE_finite_decimals_are_rational_l893_89352

theorem finite_decimals_are_rational : 
  ∀ x : ℝ, (∃ n : ℕ, ∃ m : ℤ, x = m / (10 ^ n)) → ∃ a b : ℤ, x = a / b ∧ b ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_finite_decimals_are_rational_l893_89352


namespace NUMINAMATH_CALUDE_two_number_problem_l893_89359

-- Define the types for logicians and their knowledge states
inductive Logician | A | B
inductive Knowledge | Known | Unknown

-- Define a function to represent the state of knowledge after each exchange
def knowledge_state (exchange : ℕ) (l : Logician) : Knowledge := sorry

-- Define the conditions for the sum and sum of squares
def sum_condition (u v : ℕ) : Prop := u + v = 17

def sum_of_squares_condition (u v : ℕ) : Prop := u^2 + v^2 = 145

-- Define the main theorem
theorem two_number_problem (u v : ℕ) :
  u > 0 ∧ v > 0 ∧
  sum_condition u v ∧
  sum_of_squares_condition u v ∧
  (∀ e, e < 6 → knowledge_state e Logician.A = Knowledge.Unknown) ∧
  (∀ e, e < 6 → knowledge_state e Logician.B = Knowledge.Unknown) ∧
  knowledge_state 6 Logician.B = Knowledge.Known
  → (u = 8 ∧ v = 9) ∨ (u = 9 ∧ v = 8) := by sorry

end NUMINAMATH_CALUDE_two_number_problem_l893_89359


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l893_89392

theorem largest_integer_less_than_100_remainder_5_mod_8 :
  ∀ n : ℕ, n < 100 → n % 8 = 5 → n ≤ 99 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l893_89392
