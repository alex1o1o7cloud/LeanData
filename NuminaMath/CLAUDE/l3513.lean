import Mathlib

namespace NUMINAMATH_CALUDE_lloyds_work_hours_l3513_351307

/-- Calculates the total hours worked given the conditions of Lloyd's work and pay --/
theorem lloyds_work_hours
  (regular_hours : ℝ)
  (regular_rate : ℝ)
  (overtime_multiplier : ℝ)
  (total_pay : ℝ)
  (h1 : regular_hours = 7.5)
  (h2 : regular_rate = 4)
  (h3 : overtime_multiplier = 1.5)
  (h4 : total_pay = 48) :
  ∃ (total_hours : ℝ), total_hours = 10.5 ∧
    total_pay = regular_hours * regular_rate +
                (total_hours - regular_hours) * (regular_rate * overtime_multiplier) :=
by sorry

end NUMINAMATH_CALUDE_lloyds_work_hours_l3513_351307


namespace NUMINAMATH_CALUDE_distinct_permutations_with_repetitions_l3513_351365

-- Define the number of elements
def n : ℕ := 5

-- Define the number of repetitions for the first digit
def r1 : ℕ := 3

-- Define the number of repetitions for the second digit
def r2 : ℕ := 2

-- State the theorem
theorem distinct_permutations_with_repetitions :
  (n.factorial) / (r1.factorial * r2.factorial) = 10 := by
  sorry

end NUMINAMATH_CALUDE_distinct_permutations_with_repetitions_l3513_351365


namespace NUMINAMATH_CALUDE_percentage_runs_by_running_approx_l3513_351361

-- Define the given conditions
def total_runs : ℕ := 134
def boundaries : ℕ := 12
def sixes : ℕ := 2

-- Define the runs per boundary and six
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

-- Calculate runs from boundaries and sixes
def runs_from_boundaries_and_sixes : ℕ := boundaries * runs_per_boundary + sixes * runs_per_six

-- Calculate runs made by running between wickets
def runs_by_running : ℕ := total_runs - runs_from_boundaries_and_sixes

-- Define the percentage of runs made by running
def percentage_runs_by_running : ℚ := (runs_by_running : ℚ) / (total_runs : ℚ) * 100

-- Theorem to prove
theorem percentage_runs_by_running_approx :
  abs (percentage_runs_by_running - 55.22) < 0.01 := by sorry

end NUMINAMATH_CALUDE_percentage_runs_by_running_approx_l3513_351361


namespace NUMINAMATH_CALUDE_number_ordering_l3513_351319

theorem number_ordering : 6^10 < 3^20 ∧ 3^20 < 2^30 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l3513_351319


namespace NUMINAMATH_CALUDE_function_is_linear_l3513_351335

/-- Given a function f: ℝ → ℝ satisfying f(x²-y²) = x f(x) - y f(y) for all x, y ∈ ℝ,
    prove that f is a linear function. -/
theorem function_is_linear (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) :
    ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
  sorry

end NUMINAMATH_CALUDE_function_is_linear_l3513_351335


namespace NUMINAMATH_CALUDE_machines_working_time_l3513_351375

theorem machines_working_time (x : ℝ) : 
  (1 / (x + 10) + 1 / (x + 3) + 1 / (2 * x) = 1 / x) → x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_machines_working_time_l3513_351375


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3513_351378

def num_ingredients : ℕ := 7

theorem sandwich_combinations :
  (Nat.choose num_ingredients 3 = 35) ∧ (Nat.choose num_ingredients 4 = 35) := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3513_351378


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3513_351398

theorem triangle_perimeter (m n : ℝ) : 
  let side1 := 3 * m
  let side2 := side1 - (m - n)
  let side3 := side2 + 2 * n
  side1 + side2 + side3 = 7 * m + 4 * n :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3513_351398


namespace NUMINAMATH_CALUDE_min_value_quadratic_function_l3513_351380

theorem min_value_quadratic_function :
  ∃ (min : ℝ), min = -11.25 ∧
  ∀ (x y : ℝ), 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_function_l3513_351380


namespace NUMINAMATH_CALUDE_stairs_theorem_l3513_351389

def stairs_problem (samir veronica ravi : ℕ) : Prop :=
  samir = 318 ∧
  veronica = (samir / 2) + 18 ∧
  ravi = 2 * veronica ∧
  samir + veronica + ravi = 849

theorem stairs_theorem : ∃ samir veronica ravi : ℕ, stairs_problem samir veronica ravi :=
  sorry

end NUMINAMATH_CALUDE_stairs_theorem_l3513_351389


namespace NUMINAMATH_CALUDE_painter_problem_l3513_351371

/-- Given two painters with a work ratio of 2:7 painting a total area of 270 square feet,
    the painter with the larger share paints 210 square feet. -/
theorem painter_problem (total_area : ℕ) (ratio_small : ℕ) (ratio_large : ℕ) :
  total_area = 270 →
  ratio_small = 2 →
  ratio_large = 7 →
  (ratio_large * total_area) / (ratio_small + ratio_large) = 210 :=
by sorry

end NUMINAMATH_CALUDE_painter_problem_l3513_351371


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l3513_351308

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a 10-player round-robin tournament, there are 45 matches. -/
theorem ten_player_tournament_matches : num_matches 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l3513_351308


namespace NUMINAMATH_CALUDE_min_value_theorem_l3513_351384

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + a*b + a*c + b*c = 4) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + x*y + x*z + y*z = 4 → 2*a + b + c ≤ 2*x + y + z :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3513_351384


namespace NUMINAMATH_CALUDE_road_repaving_l3513_351323

theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ) 
  (h1 : total_repaved = 4938)
  (h2 : repaved_today = 805) :
  total_repaved - repaved_today = 4133 :=
by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l3513_351323


namespace NUMINAMATH_CALUDE_f_difference_at_five_l3513_351372

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^4 + x^3 + 5*x

-- State the theorem
theorem f_difference_at_five : f 5 - f (-5) = 6550 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_five_l3513_351372


namespace NUMINAMATH_CALUDE_hiking_distance_proof_l3513_351364

def hiking_distance (total distance_car_to_stream distance_meadow_to_campsite : ℝ) : Prop :=
  ∃ distance_stream_to_meadow : ℝ,
    distance_stream_to_meadow = total - (distance_car_to_stream + distance_meadow_to_campsite) ∧
    distance_stream_to_meadow = 0.4

theorem hiking_distance_proof :
  hiking_distance 0.7 0.2 0.1 :=
by
  sorry

end NUMINAMATH_CALUDE_hiking_distance_proof_l3513_351364


namespace NUMINAMATH_CALUDE_tenth_power_sum_l3513_351324

theorem tenth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_tenth_power_sum_l3513_351324


namespace NUMINAMATH_CALUDE_pebble_distribution_theorem_l3513_351325

/-- Represents a point on a 2D integer grid -/
structure Point where
  x : Int
  y : Int

/-- Represents the state of the pebble distribution -/
def PebbleState := Point → Nat

/-- Represents an operation on the pebble distribution -/
def Operation := PebbleState → Option PebbleState

/-- The initial state has 2009 pebbles distributed on integer coordinate points -/
def initial_state : PebbleState := sorry

/-- An operation is valid if it removes 4 pebbles from a point with at least 4 pebbles
    and adds 1 pebble to each of its four adjacent points -/
def valid_operation (op : Operation) : Prop := sorry

/-- A sequence of operations is valid if each operation in the sequence is valid -/
def valid_sequence (seq : List Operation) : Prop := sorry

/-- The final state after applying a sequence of operations -/
def final_state (init : PebbleState) (seq : List Operation) : PebbleState := sorry

/-- A state is stable if no point has more than 3 pebbles -/
def is_stable (state : PebbleState) : Prop := sorry

theorem pebble_distribution_theorem :
  ∀ (seq : List Operation),
    valid_sequence seq →
    ∃ (n : Nat),
      (is_stable (final_state initial_state (seq.take n))) ∧
      (∀ (seq' : List Operation),
        valid_sequence seq' →
        is_stable (final_state initial_state seq') →
        final_state initial_state seq = final_state initial_state seq') :=
sorry

end NUMINAMATH_CALUDE_pebble_distribution_theorem_l3513_351325


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3513_351348

/-- The length of the major axis of an ellipse formed by intersecting a plane with a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The length of the major axis of the ellipse is 8 --/
theorem ellipse_major_axis_length :
  let cylinder_radius : ℝ := 2
  let major_minor_ratio : ℝ := 2
  major_axis_length cylinder_radius major_minor_ratio = 8 := by
  sorry

#check ellipse_major_axis_length

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3513_351348


namespace NUMINAMATH_CALUDE_quadratic_roots_l3513_351356

theorem quadratic_roots (a : ℝ) : 
  (3 : ℝ) ^ 2 - 2 * 3 + a = 0 → 
  (-1 : ℝ) ^ 2 - 2 * (-1) + a = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3513_351356


namespace NUMINAMATH_CALUDE_point_c_coordinates_l3513_351316

/-- Given points A and B, and a point C on line AB satisfying a vector relationship,
    prove that C has specific coordinates. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) : 
  A = (-1, -1) →
  B = (2, 5) →
  (∃ t : ℝ, C = (1 - t) • A + t • B) →  -- C is on line AB
  (C.1 - A.1, C.2 - A.2) = 5 • (B.1 - C.1, B.2 - C.2) →  -- Vector relationship
  C = (3/2, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l3513_351316


namespace NUMINAMATH_CALUDE_zlatoust_miass_distance_l3513_351385

theorem zlatoust_miass_distance :
  ∀ (g m k : ℝ),  -- speeds of GAZ, MAZ, and KamAZ
  (∀ x : ℝ, 
    (x + 18) / k = (x - 18) / m ∧
    (x + 25) / k = (x - 25) / g ∧
    (x + 8) / m = (x - 8) / g) →
  ∃ x : ℝ, x = 60 ∧ 
    (x + 18) / k = (x - 18) / m ∧
    (x + 25) / k = (x - 25) / g ∧
    (x + 8) / m = (x - 8) / g :=
by sorry

end NUMINAMATH_CALUDE_zlatoust_miass_distance_l3513_351385


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3513_351374

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3513_351374


namespace NUMINAMATH_CALUDE_at_op_difference_l3513_351320

/-- Definition of the @ operation -/
def at_op (x y : ℤ) : ℤ := 3 * x * y - 2 * x + y

/-- Theorem stating that (6@4) - (4@6) = -6 -/
theorem at_op_difference : at_op 6 4 - at_op 4 6 = -6 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l3513_351320


namespace NUMINAMATH_CALUDE_portias_school_students_l3513_351382

theorem portias_school_students (portia_students lara_students : ℕ) 
  (h1 : portia_students = 4 * lara_students)
  (h2 : portia_students + lara_students = 2500) : 
  portia_students = 2000 := by
  sorry

end NUMINAMATH_CALUDE_portias_school_students_l3513_351382


namespace NUMINAMATH_CALUDE_julia_tag_difference_l3513_351302

theorem julia_tag_difference (x y : ℕ) (hx : x = 45) (hy : y = 28) : x - y = 17 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_difference_l3513_351302


namespace NUMINAMATH_CALUDE_decimal_multiplication_correction_l3513_351305

theorem decimal_multiplication_correction (a b : ℚ) (x y : ℕ) :
  a = 0.085 →
  b = 3.45 →
  x = 85 →
  y = 345 →
  x * y = 29325 →
  a * b = 0.29325 :=
sorry

end NUMINAMATH_CALUDE_decimal_multiplication_correction_l3513_351305


namespace NUMINAMATH_CALUDE_coat_drive_l3513_351326

theorem coat_drive (total_coats high_school_coats : ℕ) 
  (h1 : total_coats = 9437)
  (h2 : high_school_coats = 6922) :
  total_coats - high_school_coats = 2515 := by
  sorry

end NUMINAMATH_CALUDE_coat_drive_l3513_351326


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3513_351359

/-- Two lines in the form ax + by + c = 0 and dx + ey + f = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Perpendicular property for two lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_a_value :
  ∀ (a : ℝ),
  let l1 : Line := ⟨a, 2, 1⟩
  let l2 : Line := ⟨1, 3, -2⟩
  perpendicular l1 l2 → a = -6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3513_351359


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3513_351373

/-- Proves that the first discount percentage is 20% given the conditions of the problem -/
theorem first_discount_percentage (original_price : ℝ) (second_discount : ℝ) (final_price : ℝ)
  (h1 : original_price = 400)
  (h2 : second_discount = 15)
  (h3 : final_price = 272)
  (h4 : final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100)) :
  first_discount = 20 := by
  sorry


end NUMINAMATH_CALUDE_first_discount_percentage_l3513_351373


namespace NUMINAMATH_CALUDE_scientific_notation_826_million_l3513_351366

theorem scientific_notation_826_million : 
  826000000 = 8.26 * (10 : ℝ)^8 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_826_million_l3513_351366


namespace NUMINAMATH_CALUDE_james_birthday_stickers_l3513_351352

/-- The number of stickers James got for his birthday -/
def birthday_stickers (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem james_birthday_stickers :
  birthday_stickers 39 61 = 22 := by
  sorry

end NUMINAMATH_CALUDE_james_birthday_stickers_l3513_351352


namespace NUMINAMATH_CALUDE_distance_between_axes_of_symmetry_l3513_351334

/-- The distance between two adjacent axes of symmetry in the graph of y = 3sin(2x + π/4) is π/2 -/
theorem distance_between_axes_of_symmetry :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (2 * x + π / 4)
  ∃ d : ℝ, d = π / 2 ∧ ∀ x : ℝ, f (x + d) = f x := by sorry

end NUMINAMATH_CALUDE_distance_between_axes_of_symmetry_l3513_351334


namespace NUMINAMATH_CALUDE_f_properties_l3513_351340

def f (x : ℝ) : ℝ := x * (x + 1) * (x - 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x > 2 ∧ y > x → f x < f y) ∧ 
  (∃! a b c, a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3513_351340


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_fraction_sum_l3513_351345

/-- Two prime numbers that are roots of a quadratic equation --/
def QuadraticPrimeRoots (a b : ℕ) : Prop :=
  Prime a ∧ Prime b ∧ ∃ t : ℤ, a^2 - 21*a + t = 0 ∧ b^2 - 21*b + t = 0

/-- The main theorem --/
theorem quadratic_prime_roots_fraction_sum
  (a b : ℕ) (h : QuadraticPrimeRoots a b) :
  (b : ℚ) / a + (a : ℚ) / b = 365 / 38 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_fraction_sum_l3513_351345


namespace NUMINAMATH_CALUDE_percentage_difference_theorem_l3513_351346

theorem percentage_difference_theorem (x : ℝ) : 
  (0.35 * x = 0.50 * x - 24) → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_theorem_l3513_351346


namespace NUMINAMATH_CALUDE_parabola_intersection_and_area_minimization_l3513_351341

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the line passing through M and intersecting the parabola
def line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Define the dot product of vectors OA and OB
def dot_product (x1 x2 : ℝ) : ℝ := x1 * x2 + (x1^2) * (x2^2)

theorem parabola_intersection_and_area_minimization 
  (k m : ℝ) -- Parameters of the line
  (x1 x2 : ℝ) -- x-coordinates of intersection points A and B
  (h1 : parabola x1 = line k m x1) -- A is on both parabola and line
  (h2 : parabola x2 = line k m x2) -- B is on both parabola and line
  (h3 : dot_product x1 x2 = 2) -- Given condition
  (h4 : m = 2) -- Line passes through (0, 2)
  : 
  (∃ (x : ℝ), line k m x = 2) ∧ -- Line passes through (0, 2)
  (∃ (area : ℝ), area = 3 ∧ 
    ∀ (x : ℝ), x > 0 → x + 9/(4*x) ≥ area) -- Minimum area is 3
  := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_and_area_minimization_l3513_351341


namespace NUMINAMATH_CALUDE_min_value_sin_squares_l3513_351332

theorem min_value_sin_squares (α β : Real) 
  (h : -5 * Real.sin α ^ 2 + Real.sin β ^ 2 = 3 * Real.sin α) :
  ∃ (y : Real), y = Real.sin α ^ 2 + Real.sin β ^ 2 ∧ 
  (∀ (z : Real), z = Real.sin α ^ 2 + Real.sin β ^ 2 → y ≤ z) ∧
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_sin_squares_l3513_351332


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3513_351317

/-- A right-angled triangle with area 150 cm² and perimeter 60 cm has sides of length 15 cm, 20 cm, and 25 cm. -/
theorem right_triangle_sides (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 150 →
  a + b + c = 60 →
  ((a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15)) ∧ c = 25 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3513_351317


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3513_351342

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (x + 1/x - 2)^5
  ∃ (p : ℝ → ℝ), expansion = p x ∧ p 0 = -252 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3513_351342


namespace NUMINAMATH_CALUDE_one_of_each_color_probability_l3513_351362

def total_marbles : ℕ := 6
def red_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def green_marbles : ℕ := 2
def selected_marbles : ℕ := 3

theorem one_of_each_color_probability :
  (red_marbles * blue_marbles * green_marbles) / Nat.choose total_marbles selected_marbles = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_of_each_color_probability_l3513_351362


namespace NUMINAMATH_CALUDE_our_ellipse_correct_l3513_351333

/-- An ellipse with foci at (-2, 0) and (2, 0) passing through (2, 3) -/
structure Ellipse where
  -- The equation of the ellipse
  equation : ℝ → ℝ → Prop
  -- The foci are at (-2, 0) and (2, 0)
  foci_x : equation (-2) 0 ∧ equation 2 0
  -- The ellipse passes through (2, 3)
  passes_through : equation 2 3
  -- The equation is of the form x^2/a^2 + y^2/b^2 = 1 for some a, b
  is_standard_form : ∃ a b : ℝ, ∀ x y : ℝ, equation x y ↔ x^2/a^2 + y^2/b^2 = 1

/-- The specific ellipse we're interested in -/
def our_ellipse : Ellipse where
  equation := fun x y => x^2/16 + y^2/12 = 1
  foci_x := sorry
  passes_through := sorry
  is_standard_form := sorry

/-- The theorem stating that our_ellipse satisfies all the conditions -/
theorem our_ellipse_correct : 
  our_ellipse.equation = fun x y => x^2/16 + y^2/12 = 1 := by sorry

end NUMINAMATH_CALUDE_our_ellipse_correct_l3513_351333


namespace NUMINAMATH_CALUDE_polynomial_difference_divisibility_l3513_351322

/-- For any polynomial P with integer coefficients and any integers a and b,
    (a - b) divides (P(a) - P(b)) in ℤ. -/
theorem polynomial_difference_divisibility (P : Polynomial ℤ) (a b : ℤ) :
  (a - b) ∣ (P.eval a - P.eval b) :=
sorry

end NUMINAMATH_CALUDE_polynomial_difference_divisibility_l3513_351322


namespace NUMINAMATH_CALUDE_five_points_on_circle_l3513_351310

-- Define a type for lines in general position
structure GeneralPositionLine where
  -- Add necessary fields

-- Define a type for points
structure Point where
  -- Add necessary fields

-- Define a type for circles
structure Circle where
  -- Add necessary fields

-- Function to get the intersection point of two lines
def lineIntersection (l1 l2 : GeneralPositionLine) : Point :=
  sorry

-- Function to get the circle passing through three points
def circleThrough3Points (p1 p2 p3 : Point) : Circle :=
  sorry

-- Function to get the intersection point of two circles
def circleIntersection (c1 c2 : Circle) : Point :=
  sorry

-- Function to check if a point lies on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  sorry

-- Main theorem
theorem five_points_on_circle 
  (l1 l2 l3 l4 l5 : GeneralPositionLine) : 
  ∃ (c : Circle),
    let s12 := circleThrough3Points (lineIntersection l3 l4) (lineIntersection l3 l5) (lineIntersection l4 l5)
    let s13 := circleThrough3Points (lineIntersection l2 l4) (lineIntersection l2 l5) (lineIntersection l4 l5)
    let s14 := circleThrough3Points (lineIntersection l2 l3) (lineIntersection l2 l5) (lineIntersection l3 l5)
    let s15 := circleThrough3Points (lineIntersection l2 l3) (lineIntersection l2 l4) (lineIntersection l3 l4)
    let s23 := circleThrough3Points (lineIntersection l1 l4) (lineIntersection l1 l5) (lineIntersection l4 l5)
    let s24 := circleThrough3Points (lineIntersection l1 l3) (lineIntersection l1 l5) (lineIntersection l3 l5)
    let s25 := circleThrough3Points (lineIntersection l1 l3) (lineIntersection l1 l4) (lineIntersection l3 l4)
    let s34 := circleThrough3Points (lineIntersection l1 l2) (lineIntersection l1 l5) (lineIntersection l2 l5)
    let s35 := circleThrough3Points (lineIntersection l1 l2) (lineIntersection l1 l4) (lineIntersection l2 l4)
    let s45 := circleThrough3Points (lineIntersection l1 l2) (lineIntersection l1 l3) (lineIntersection l2 l3)
    let a1 := circleIntersection s23 s24
    let a2 := circleIntersection s13 s14
    let a3 := circleIntersection s12 s14
    let a4 := circleIntersection s12 s13
    let a5 := circleIntersection s12 s23
    pointOnCircle a1 c ∧ 
    pointOnCircle a2 c ∧ 
    pointOnCircle a3 c ∧ 
    pointOnCircle a4 c ∧ 
    pointOnCircle a5 c :=
  sorry


end NUMINAMATH_CALUDE_five_points_on_circle_l3513_351310


namespace NUMINAMATH_CALUDE_apples_in_first_group_l3513_351343

-- Define the cost of an apple
def apple_cost : ℚ := 21/100

-- Define the equation for the first group
def first_group (x : ℚ) (orange_cost : ℚ) : Prop :=
  x * apple_cost + 3 * orange_cost = 177/100

-- Define the equation for the second group
def second_group (orange_cost : ℚ) : Prop :=
  2 * apple_cost + 5 * orange_cost = 127/100

-- Theorem stating that the number of apples in the first group is 6
theorem apples_in_first_group :
  ∃ (orange_cost : ℚ), first_group 6 orange_cost ∧ second_group orange_cost :=
sorry

end NUMINAMATH_CALUDE_apples_in_first_group_l3513_351343


namespace NUMINAMATH_CALUDE_remainder_doubling_l3513_351303

theorem remainder_doubling (N : ℤ) : 
  N % 367 = 241 → (2 * N) % 367 = 115 := by
sorry

end NUMINAMATH_CALUDE_remainder_doubling_l3513_351303


namespace NUMINAMATH_CALUDE_round_windmill_iff_on_diagonal_l3513_351370

/-- A square in a 2D plane. -/
structure Square :=
  (A B C D : ℝ × ℝ)

/-- A point in a 2D plane. -/
def Point := ℝ × ℝ

/-- A line in a 2D plane. -/
structure Line :=
  (p1 p2 : Point)

/-- A windmill configuration. -/
structure Windmill :=
  (center : Point)
  (l1 l2 : Line)

/-- Checks if a point is inside a square. -/
def isInside (s : Square) (p : Point) : Prop := sorry

/-- Checks if two lines are perpendicular. -/
def arePerpendicular (l1 l2 : Line) : Prop := sorry

/-- Checks if a quadrilateral is cyclic. -/
def isCyclic (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a point lies on the diagonal of a square. -/
def isOnDiagonal (s : Square) (p : Point) : Prop := sorry

/-- Theorem: A point P inside a square ABCD produces a round windmill for all
    possible configurations if and only if P lies on the diagonals of the square. -/
theorem round_windmill_iff_on_diagonal (s : Square) (p : Point) :
  isInside s p →
  (∀ (w : Windmill), w.center = p →
    arePerpendicular w.l1 w.l2 →
    (∃ W X Y Z, isCyclic W X Y Z)) ↔
  isOnDiagonal s p :=
sorry

end NUMINAMATH_CALUDE_round_windmill_iff_on_diagonal_l3513_351370


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3513_351331

/-- Given a line segment with midpoint (1, -2) and one endpoint at (4, 5), 
    prove that the other endpoint is at (-2, -9) -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (1, -2) → endpoint1 = (4, 5) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) → 
  endpoint2 = (-2, -9) := by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3513_351331


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3513_351336

theorem x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1/x^4) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3513_351336


namespace NUMINAMATH_CALUDE_rearrangements_without_substring_l3513_351355

def string_length : ℕ := 8
def h_count : ℕ := 2
def m_count : ℕ := 4
def t_count : ℕ := 2

def total_arrangements : ℕ := string_length.factorial / (h_count.factorial * m_count.factorial * t_count.factorial)

def substring_length : ℕ := 4
def remaining_string_length : ℕ := string_length - substring_length + 1

def arrangements_with_substring : ℕ := 
  (remaining_string_length.factorial / (h_count.pred.factorial * m_count.pred.pred.pred.factorial)) * 
  (substring_length.factorial / m_count.pred.factorial)

theorem rearrangements_without_substring : 
  total_arrangements - arrangements_with_substring + 1 = 361 := by sorry

end NUMINAMATH_CALUDE_rearrangements_without_substring_l3513_351355


namespace NUMINAMATH_CALUDE_equation_solution_l3513_351321

theorem equation_solution : ∃ n : ℝ, (1 / (n + 2) + 2 / (n + 2) + n / (n + 2) = 2) ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3513_351321


namespace NUMINAMATH_CALUDE_fraction_value_l3513_351357

theorem fraction_value (x y : ℝ) (h : (1 / x) + (1 / y) = 2) :
  (-2 * y + x * y - 2 * x) / (3 * x + x * y + 3 * y) = -3 / 7 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l3513_351357


namespace NUMINAMATH_CALUDE_hall_dark_tile_fraction_l3513_351363

/-- Represents a tiling pattern on a floor -/
structure TilingPattern :=
  (size : Nat)
  (dark_tiles_in_section : Nat)
  (section_size : Nat)

/-- The fraction of dark tiles in a tiling pattern -/
def dark_tile_fraction (pattern : TilingPattern) : Rat :=
  pattern.dark_tiles_in_section / (pattern.section_size * pattern.section_size)

/-- Theorem stating that for the given tiling pattern, the fraction of dark tiles is 5/8 -/
theorem hall_dark_tile_fraction :
  ∀ (pattern : TilingPattern),
    pattern.size = 8 ∧
    pattern.section_size = 4 ∧
    pattern.dark_tiles_in_section = 10 →
    dark_tile_fraction pattern = 5 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_hall_dark_tile_fraction_l3513_351363


namespace NUMINAMATH_CALUDE_wyatt_remaining_money_l3513_351367

/-- Calculates the remaining money after shopping, given the initial amount,
    costs of items, quantities, and discount rate. -/
def remaining_money (initial_amount : ℚ)
                    (bread_cost loaves orange_juice_cost cartons
                     cookie_cost boxes apple_cost pounds
                     chocolate_cost bars : ℚ)
                    (discount_rate : ℚ) : ℚ :=
  let total_cost := bread_cost * loaves +
                    orange_juice_cost * cartons +
                    cookie_cost * boxes +
                    apple_cost * pounds +
                    chocolate_cost * bars
  let discounted_cost := total_cost * (1 - discount_rate)
  initial_amount - discounted_cost

/-- Theorem stating that Wyatt has $127.60 left after shopping -/
theorem wyatt_remaining_money :
  remaining_money 200
                  6.50 5 3.25 4 2.10 7 1.75 3 2.50 6
                  0.10 = 127.60 := by
  sorry

end NUMINAMATH_CALUDE_wyatt_remaining_money_l3513_351367


namespace NUMINAMATH_CALUDE_gym_monthly_income_l3513_351301

-- Define the gym's charging structure
def twice_monthly_charge : ℕ := 18

-- Define the number of members
def number_of_members : ℕ := 300

-- Define the monthly income
def monthly_income : ℕ := twice_monthly_charge * 2 * number_of_members

-- Theorem statement
theorem gym_monthly_income :
  monthly_income = 10800 :=
by sorry

end NUMINAMATH_CALUDE_gym_monthly_income_l3513_351301


namespace NUMINAMATH_CALUDE_probability_non_defective_pencils_l3513_351353

/-- The probability of selecting 3 non-defective pencils from a box of 11 pencils with 2 defective ones -/
theorem probability_non_defective_pencils (total_pencils : Nat) (defective_pencils : Nat) (selected_pencils : Nat) :
  total_pencils = 11 →
  defective_pencils = 2 →
  selected_pencils = 3 →
  (Nat.choose (total_pencils - defective_pencils) selected_pencils : ℚ) / 
  (Nat.choose total_pencils selected_pencils : ℚ) = 28 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_defective_pencils_l3513_351353


namespace NUMINAMATH_CALUDE_three_hits_in_five_shots_l3513_351347

/-- The probability of hitting the target exactly k times in n independent shots,
    where p is the probability of hitting the target in each shot. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of hitting the target exactly 3 times in 5 shots,
    where the probability of hitting the target in each shot is 0.6,
    is equal to 0.3456. -/
theorem three_hits_in_five_shots :
  binomial_probability 5 3 0.6 = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_three_hits_in_five_shots_l3513_351347


namespace NUMINAMATH_CALUDE_sum_100_to_120_l3513_351377

def sum_inclusive_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_100_to_120 : sum_inclusive_range 100 120 = 2310 := by sorry

end NUMINAMATH_CALUDE_sum_100_to_120_l3513_351377


namespace NUMINAMATH_CALUDE_remaining_balance_calculation_l3513_351314

/-- Calculates the remaining balance for a product purchase with given conditions -/
theorem remaining_balance_calculation (deposit : ℝ) (deposit_rate : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (service_charge : ℝ) :
  deposit = 110 →
  deposit_rate = 0.10 →
  tax_rate = 0.15 →
  discount_rate = 0.05 →
  service_charge = 50 →
  ∃ (total_price : ℝ),
    total_price = deposit / deposit_rate ∧
    (total_price * (1 + tax_rate) * (1 - discount_rate) + service_charge - deposit) = 1141.75 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_calculation_l3513_351314


namespace NUMINAMATH_CALUDE_nancy_washed_19_shirts_l3513_351386

/-- The number of shirts Nancy had to wash -/
def num_shirts (machine_capacity : ℕ) (num_loads : ℕ) (num_sweaters : ℕ) : ℕ :=
  machine_capacity * num_loads - num_sweaters

/-- Proof that Nancy washed 19 shirts -/
theorem nancy_washed_19_shirts :
  num_shirts 9 3 8 = 19 := by
  sorry

end NUMINAMATH_CALUDE_nancy_washed_19_shirts_l3513_351386


namespace NUMINAMATH_CALUDE_equation_solution_l3513_351390

theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt (7 * x - 2) - Real.sqrt (3 * x - 1) = 2) ∧ (x = 0.515625) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3513_351390


namespace NUMINAMATH_CALUDE_exponential_equation_comparison_l3513_351368

theorem exponential_equation_comparison
  (c d k n : ℝ)
  (hc : c > 0)
  (hk : k > 0)
  (hd : d ≠ 0)
  (hn : n ≠ 0) :
  (Real.log c / Real.log k < d / n) ↔
  ((1 / d) * Real.log (1 / c) < (1 / n) * Real.log (1 / k)) :=
sorry

end NUMINAMATH_CALUDE_exponential_equation_comparison_l3513_351368


namespace NUMINAMATH_CALUDE_equation_solution_l3513_351311

theorem equation_solution : ∃! y : ℝ, 5 * (y + 2) + 9 = 3 * (1 - y) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3513_351311


namespace NUMINAMATH_CALUDE_certain_number_problem_l3513_351381

theorem certain_number_problem (x : ℤ) : 17 * (x + 99) = 3111 ↔ x = 84 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3513_351381


namespace NUMINAMATH_CALUDE_remainder_seven_divisors_l3513_351395

theorem remainder_seven_divisors (n : ℕ) : 
  (∃ (divisors : Finset ℕ), 
    divisors = {d : ℕ | d > 7 ∧ 54 % d = 0} ∧ 
    Finset.card divisors = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_divisors_l3513_351395


namespace NUMINAMATH_CALUDE_range_of_a_l3513_351369

/-- The exponential function f(x) = (2a - 6)^x is monotonically decreasing on ℝ -/
def P (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2*a - 6)^x > (2*a - 6)^y

/-- Both real roots of the equation x^2 - 3ax + 2a^2 + 1 = 0 are greater than 3 -/
def Q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 3*a*x + 2*a^2 + 1 = 0 → x > 3

theorem range_of_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 4) :
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) ↔ a ≥ 3.5 ∧ a < 5 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3513_351369


namespace NUMINAMATH_CALUDE_shared_fixed_points_l3513_351306

/-- A function that represents f(x) = x^2 - 2 --/
def f (x : ℝ) : ℝ := x^2 - 2

/-- A function that represents g(x) = 2x^2 - c --/
def g (c : ℝ) (x : ℝ) : ℝ := 2*x^2 - c

/-- The theorem stating the conditions for shared fixed points --/
theorem shared_fixed_points (c : ℝ) : 
  (c = 3 ∨ c = 6) ↔ ∃ x : ℝ, (f x = x ∧ g c x = x) :=
sorry

end NUMINAMATH_CALUDE_shared_fixed_points_l3513_351306


namespace NUMINAMATH_CALUDE_license_plate_increase_l3513_351388

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letters in an old license plate. -/
def old_letters : ℕ := 3

/-- The number of digits in an old license plate. -/
def old_digits : ℕ := 2

/-- The number of letters in a new license plate. -/
def new_letters : ℕ := 2

/-- The number of digits in a new license plate. -/
def new_digits : ℕ := 4

/-- The theorem stating the increase in the number of possible license plates. -/
theorem license_plate_increase :
  (num_letters ^ new_letters * num_digits ^ new_digits) /
  (num_letters ^ old_letters * num_digits ^ old_digits) = 50 / 13 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_increase_l3513_351388


namespace NUMINAMATH_CALUDE_couscous_dishes_proof_l3513_351379

/-- Calculates the number of dishes that can be made from couscous shipments -/
def couscous_dishes (shipment1 shipment2 shipment3 pounds_per_dish : ℕ) : ℕ :=
  (shipment1 + shipment2 + shipment3) / pounds_per_dish

/-- Proves that given the specified shipments and dish requirement, 13 dishes can be made -/
theorem couscous_dishes_proof :
  couscous_dishes 7 13 45 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_couscous_dishes_proof_l3513_351379


namespace NUMINAMATH_CALUDE_six_pencil_sharpeners_affordable_remaining_money_buys_four_pencil_cases_remaining_money_after_ten_pencil_cases_l3513_351399

-- Define the given prices and budget
def total_budget : ℕ := 100
def pencil_sharpener_price : ℕ := 15
def notebooks_6_price : ℕ := 24
def pencil_case_price : ℕ := 5
def colored_pencils_2boxes_price : ℕ := 16

-- Theorem 1: 6 pencil sharpeners cost less than or equal to 100 yuan
theorem six_pencil_sharpeners_affordable :
  6 * pencil_sharpener_price ≤ total_budget :=
sorry

-- Theorem 2: After buying 20 notebooks, the remaining money can buy exactly 4 pencil cases
theorem remaining_money_buys_four_pencil_cases :
  (total_budget - (20 * (notebooks_6_price / 6))) / pencil_case_price = 4 :=
sorry

-- Theorem 3: After buying 10 pencil cases, the remaining money is 50 yuan
theorem remaining_money_after_ten_pencil_cases :
  total_budget - (10 * pencil_case_price) = 50 :=
sorry

end NUMINAMATH_CALUDE_six_pencil_sharpeners_affordable_remaining_money_buys_four_pencil_cases_remaining_money_after_ten_pencil_cases_l3513_351399


namespace NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l3513_351376

theorem unique_solution_ceiling_equation :
  ∃! x : ℝ, x > 0 ∧ x + ⌈x⌉ = 21.3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l3513_351376


namespace NUMINAMATH_CALUDE_bailey_credit_cards_l3513_351329

/-- The number of credit cards Bailey used to split the charges for her pet supplies purchase. -/
def number_of_credit_cards : ℕ :=
  let dog_treats : ℕ := 8
  let chew_toys : ℕ := 2
  let rawhide_bones : ℕ := 10
  let items_per_charge : ℕ := 5
  let total_items : ℕ := dog_treats + chew_toys + rawhide_bones
  total_items / items_per_charge

theorem bailey_credit_cards :
  number_of_credit_cards = 4 := by
  sorry

end NUMINAMATH_CALUDE_bailey_credit_cards_l3513_351329


namespace NUMINAMATH_CALUDE_truck_license_combinations_l3513_351351

/-- The number of possible letters for a truck license -/
def num_letters : ℕ := 3

/-- The number of digits in a truck license -/
def num_digits : ℕ := 6

/-- The number of possible digits (0-9) for each position -/
def digits_per_position : ℕ := 10

/-- The total number of possible truck license combinations -/
def total_combinations : ℕ := num_letters * (digits_per_position ^ num_digits)

theorem truck_license_combinations :
  total_combinations = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_truck_license_combinations_l3513_351351


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_is_seven_tenths_l3513_351358

/-- Represents the composition of a research team -/
structure ResearchTeam where
  total : Nat
  males : Nat
  females : Nat

/-- Calculates the probability of at least one female being selected
    when choosing two representatives from a research team -/
def probAtLeastOneFemale (team : ResearchTeam) : Rat :=
  sorry

/-- The main theorem stating the probability for the given team composition -/
theorem prob_at_least_one_female_is_seven_tenths :
  let team : ResearchTeam := ⟨5, 3, 2⟩
  probAtLeastOneFemale team = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_female_is_seven_tenths_l3513_351358


namespace NUMINAMATH_CALUDE_max_xy_value_l3513_351396

theorem max_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : 2 * x + y = 1) :
  x * y ≤ 1 / 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ x * y = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l3513_351396


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l3513_351328

theorem cryptarithmetic_puzzle (F I V E N : ℕ) : 
  F = 8 → 
  E % 3 = 0 →
  E % 2 = 0 →
  E > 0 →
  E + E ≡ E [ZMOD 10] →
  I + I ≡ N [ZMOD 10] →
  F + F = 10 + N →
  N = 1 →
  (F * 1000 + I * 100 + V * 10 + E) + (F * 1000 + I * 100 + V * 10 + E) = N * 1000 + I * 100 + N * 10 + E →
  I = 5 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l3513_351328


namespace NUMINAMATH_CALUDE_reading_time_calculation_gwendolyn_reading_time_l3513_351392

theorem reading_time_calculation (reading_speed : ℕ) (paragraphs_per_page : ℕ) 
  (sentences_per_paragraph : ℕ) (total_pages : ℕ) : ℕ :=
  let sentences_per_page := paragraphs_per_page * sentences_per_paragraph
  let total_sentences := sentences_per_page * total_pages
  total_sentences / reading_speed

theorem gwendolyn_reading_time : 
  reading_time_calculation 300 40 20 150 = 400 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_gwendolyn_reading_time_l3513_351392


namespace NUMINAMATH_CALUDE_f_above_g_implies_m_less_than_5_l3513_351312

/-- The function f(x) = |x - 2| -/
def f (x : ℝ) : ℝ := |x - 2|

/-- The function g(x) = -|x + 3| + m -/
def g (x m : ℝ) : ℝ := -|x + 3| + m

/-- Theorem: If f(x) is always above g(x) for all real x, then m < 5 -/
theorem f_above_g_implies_m_less_than_5 (m : ℝ) :
  (∀ x : ℝ, f x > g x m) → m < 5 :=
by
  sorry


end NUMINAMATH_CALUDE_f_above_g_implies_m_less_than_5_l3513_351312


namespace NUMINAMATH_CALUDE_intersection_A_B_l3513_351327

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | |x| < 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3513_351327


namespace NUMINAMATH_CALUDE_history_book_cost_l3513_351394

theorem history_book_cost 
  (total_books : ℕ) 
  (math_books : ℕ) 
  (math_book_cost : ℚ) 
  (total_price : ℚ) 
  (h1 : total_books = 90)
  (h2 : math_books = 60)
  (h3 : math_book_cost = 4)
  (h4 : total_price = 390) :
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 := by
  sorry

end NUMINAMATH_CALUDE_history_book_cost_l3513_351394


namespace NUMINAMATH_CALUDE_intersection_constraint_l3513_351393

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + 2*p.2^2 = 3}

def N (m b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m*p.1 + b}

theorem intersection_constraint (b : ℝ) :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) → b ∈ Set.Icc (-Real.sqrt (3/2)) (Real.sqrt (3/2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_constraint_l3513_351393


namespace NUMINAMATH_CALUDE_floor_plus_half_l3513_351391

theorem floor_plus_half (x : ℝ) : 
  ⌊x + 0.5⌋ = ⌊x⌋ ∨ ⌊x + 0.5⌋ = ⌊x⌋ + 1 := by sorry

end NUMINAMATH_CALUDE_floor_plus_half_l3513_351391


namespace NUMINAMATH_CALUDE_min_distance_squared_to_point_l3513_351313

/-- The minimum distance squared from a point on the line x - y - 1 = 0 to the point (2, 2) is 1/2 -/
theorem min_distance_squared_to_point : 
  ∀ x y : ℝ, x - y - 1 = 0 → ∃ m : ℝ, m = (1 : ℝ) / 2 ∧ ∀ a b : ℝ, a - b - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≤ (a - 2)^2 + (b - 2)^2 := by
  sorry


end NUMINAMATH_CALUDE_min_distance_squared_to_point_l3513_351313


namespace NUMINAMATH_CALUDE_vector_magnitude_l3513_351318

/-- Given vectors a and b in ℝ², where a · b = 0, prove that |b| = √5 -/
theorem vector_magnitude (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) 
  (ha : a = (1, 2)) (hb : b.1 = 2) : Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

#check vector_magnitude

end NUMINAMATH_CALUDE_vector_magnitude_l3513_351318


namespace NUMINAMATH_CALUDE_candy_box_problem_l3513_351354

/-- Given the number of chocolate boxes, caramel boxes, and total pieces of candy,
    calculate the number of pieces in each box. -/
def pieces_per_box (chocolate_boxes caramel_boxes total_pieces : ℕ) : ℕ :=
  total_pieces / (chocolate_boxes + caramel_boxes)

/-- Theorem stating that given 7 boxes of chocolate candy, 3 boxes of caramel candy,
    and a total of 80 pieces, there are 8 pieces in each box. -/
theorem candy_box_problem :
  pieces_per_box 7 3 80 = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_problem_l3513_351354


namespace NUMINAMATH_CALUDE_crayons_erasers_difference_l3513_351387

/-- Given the initial numbers of crayons and erasers, and the final number of crayons,
    prove that the difference between the number of crayons left and the number of erasers is 66. -/
theorem crayons_erasers_difference 
  (initial_crayons : ℕ) 
  (initial_erasers : ℕ) 
  (final_crayons : ℕ) 
  (h1 : initial_crayons = 617) 
  (h2 : initial_erasers = 457) 
  (h3 : final_crayons = 523) : 
  final_crayons - initial_erasers = 66 := by
  sorry

#check crayons_erasers_difference

end NUMINAMATH_CALUDE_crayons_erasers_difference_l3513_351387


namespace NUMINAMATH_CALUDE_fourth_power_difference_l3513_351360

theorem fourth_power_difference (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_difference_l3513_351360


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_three_answer_is_199999_l3513_351344

theorem largest_n_divisible_by_three (n : ℕ) : 
  n < 200000 → 
  (3 ∣ (10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36)) → 
  n ≤ 199999 :=
by sorry

theorem answer_is_199999 : 
  199999 < 200000 ∧ 
  (3 ∣ (10 * (199999 - 3)^5 - 2 * 199999^2 + 20 * 199999 - 36)) ∧
  ∀ m : ℕ, m > 199999 → m < 200000 → 
    ¬(3 ∣ (10 * (m - 3)^5 - 2 * m^2 + 20 * m - 36)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_three_answer_is_199999_l3513_351344


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l3513_351339

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = x^2 - 4*x + 3}
def B : Set ℝ := {y | ∃ x, y = -x^2 - 2*x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {y | -1 ≤ y ∧ y ≤ 1} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l3513_351339


namespace NUMINAMATH_CALUDE_fifteen_is_zero_l3513_351337

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ (∀ x, f (x + 2) = -f x)

/-- Theorem stating that any function satisfying the conditions has f(15) = 0 -/
theorem fifteen_is_zero (f : ℝ → ℝ) (h : special_function f) : f 15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_is_zero_l3513_351337


namespace NUMINAMATH_CALUDE_complement_of_A_l3513_351350

-- Define the set A
def A : Set ℝ := {x : ℝ | x ≤ 1}

-- State the theorem
theorem complement_of_A : 
  {x : ℝ | x ∉ A} = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3513_351350


namespace NUMINAMATH_CALUDE_total_bike_cost_l3513_351330

def marion_bike_cost : ℕ := 356
def stephanie_bike_cost : ℕ := 2 * marion_bike_cost

theorem total_bike_cost : marion_bike_cost + stephanie_bike_cost = 1068 := by
  sorry

end NUMINAMATH_CALUDE_total_bike_cost_l3513_351330


namespace NUMINAMATH_CALUDE_unique_solution_xyz_l3513_351304

theorem unique_solution_xyz (x y z : ℕ) :
  x > 1 → y > 1 → z > 1 → (x + 1)^y - x^z = 1 → x = 2 ∧ y = 2 ∧ z = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_xyz_l3513_351304


namespace NUMINAMATH_CALUDE_price_increase_l3513_351300

theorem price_increase (x : ℝ) : 
  (1 + x / 100) * (1 + x / 100) = 1 + 32.25 / 100 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_l3513_351300


namespace NUMINAMATH_CALUDE_tournament_committee_count_l3513_351309

/-- Represents a frisbee league -/
structure FrisbeeLeague where
  teams : Nat
  membersPerTeam : Nat
  committeeSize : Nat
  hostTeamMembers : Nat
  nonHostTeamMembers : Nat

/-- The specific frisbee league described in the problem -/
def regionalLeague : FrisbeeLeague :=
  { teams := 5
  , membersPerTeam := 8
  , committeeSize := 11
  , hostTeamMembers := 4
  , nonHostTeamMembers := 3 }

/-- The number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

/-- The number of possible tournament committees -/
def numberOfCommittees (league : FrisbeeLeague) : Nat :=
  league.teams *
  (choose (league.membersPerTeam - 1) (league.hostTeamMembers - 1)) *
  (choose league.membersPerTeam league.nonHostTeamMembers ^ (league.teams - 1))

/-- Theorem stating the number of possible tournament committees -/
theorem tournament_committee_count :
  numberOfCommittees regionalLeague = 1723286800 := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l3513_351309


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_two_point_five_l3513_351383

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A point is inside a square --/
def isInside (p : ℝ × ℝ) (s : Square) : Prop :=
  s.bottomLeft.1 ≤ p.1 ∧ p.1 ≤ s.topRight.1 ∧
  s.bottomLeft.2 ≤ p.2 ∧ p.2 ≤ s.topRight.2

/-- The probability of an event for a uniformly distributed point in a square --/
def probability (s : Square) (event : ℝ × ℝ → Prop) : ℝ :=
  sorry

theorem probability_x_plus_y_less_than_two_point_five :
  let s : Square := { bottomLeft := (0, 0), topRight := (3, 3) }
  probability s (fun p => p.1 + p.2 < 2.5) = 125 / 360 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_two_point_five_l3513_351383


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3513_351338

theorem simplify_and_evaluate (x : ℝ) (h : x = 2) : 
  (1 + 1 / (x + 1)) / ((x + 2) / (x^2 - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3513_351338


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3513_351315

/-- The total surface area of a cylinder with height 12 and radius 4 is 128π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 12
  let r : ℝ := 4
  let circle_area : ℝ := π * r^2
  let lateral_area : ℝ := 2 * π * r * h
  circle_area * 2 + lateral_area = 128 * π := by
sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3513_351315


namespace NUMINAMATH_CALUDE_angle_bisector_inequality_l3513_351349

/-- Represents a triangle with side lengths and angle bisectors -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  aa_prime : ℝ → ℝ → ℝ
  bb_prime : ℝ → ℝ → ℝ

/-- Theorem: In a triangle ABC with angle bisectors AA' and BB', if a > b, then CA' > CB' and BA' > AB' -/
theorem angle_bisector_inequality (t : Triangle) (h : t.a > t.b) :
  (t.c * t.a) / (t.b + t.c) > (t.a * t.b) / (t.a + t.c) ∧
  (t.a * t.b) / (t.b + t.c) > (t.c * t.b) / (t.a + t.c) := by
  sorry


end NUMINAMATH_CALUDE_angle_bisector_inequality_l3513_351349


namespace NUMINAMATH_CALUDE_must_divide_p_l3513_351397

theorem must_divide_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 28)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 63)
  (h4 : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) :
  11 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_must_divide_p_l3513_351397
