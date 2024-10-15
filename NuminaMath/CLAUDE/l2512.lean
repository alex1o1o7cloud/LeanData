import Mathlib

namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_l2512_251276

theorem min_value_cyclic_fraction (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a / b + b / c + c / d + d / a ≥ 4 ∧ 
  (a / b + b / c + c / d + d / a = 4 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_l2512_251276


namespace NUMINAMATH_CALUDE_equal_sum_squared_distances_exist_l2512_251278

-- Define a triangle as a tuple of three points in a plane
def Triangle := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Define a function to calculate the sum of squared distances from a point to triangle vertices
def sumSquaredDistances (p : ℝ × ℝ) (t : Triangle) : ℝ :=
  let (a, b, c) := t
  (p.1 - a.1)^2 + (p.2 - a.2)^2 +
  (p.1 - b.1)^2 + (p.2 - b.2)^2 +
  (p.1 - c.1)^2 + (p.2 - c.2)^2

-- State the theorem
theorem equal_sum_squared_distances_exist (t1 t2 t3 : Triangle) :
  ∃ (p : ℝ × ℝ), sumSquaredDistances p t1 = sumSquaredDistances p t2 ∧
                 sumSquaredDistances p t2 = sumSquaredDistances p t3 :=
sorry

end NUMINAMATH_CALUDE_equal_sum_squared_distances_exist_l2512_251278


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l2512_251298

theorem largest_stamps_per_page : Nat.gcd 840 1008 = 168 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l2512_251298


namespace NUMINAMATH_CALUDE_road_repair_hours_l2512_251283

theorem road_repair_hours (people1 people2 days1 days2 hours2 : ℕ) 
  (h1 : people1 = 42)
  (h2 : people2 = 30)
  (h3 : days1 = 12)
  (h4 : days2 = 14)
  (h5 : hours2 = 6)
  (h6 : people1 * days1 * (people1 * days1 * hours2 / (people2 * days2)) = people2 * days2 * hours2) :
  people1 * days1 * hours2 / (people2 * days2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_hours_l2512_251283


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2512_251279

def geometric_sequence (a : ℕ → ℚ) := ∀ n, a (n + 1) = a n * (a 2 / a 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geom : geometric_sequence a) 
  (h_a1 : a 1 = 1/8) 
  (h_a4 : a 4 = -1) : 
  a 2 / a 1 = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2512_251279


namespace NUMINAMATH_CALUDE_expression_value_l2512_251273

theorem expression_value : 2 * Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + 2 * Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2512_251273


namespace NUMINAMATH_CALUDE_g_is_linear_l2512_251248

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom integral_condition : ∀ x : ℝ, f x + g x = ∫ t in x..(x+1), 2*t

axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem g_is_linear : (∀ x : ℝ, f x + g x = ∫ t in x..(x+1), 2*t) → 
                      (∀ x : ℝ, f (-x) = -f x) → 
                      (∀ x : ℝ, g x = 1 + x) :=
sorry

end NUMINAMATH_CALUDE_g_is_linear_l2512_251248


namespace NUMINAMATH_CALUDE_nested_series_sum_l2512_251226

def nested_series (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | k + 1 => 2 * (1 + nested_series k)

theorem nested_series_sum : nested_series 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_nested_series_sum_l2512_251226


namespace NUMINAMATH_CALUDE_fraction_value_l2512_251255

theorem fraction_value (x y : ℝ) (h1 : -1 < (y - x) / (x + y)) (h2 : (y - x) / (x + y) < 2) 
  (h3 : ∃ n : ℤ, y / x = n) : y / x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2512_251255


namespace NUMINAMATH_CALUDE_stan_playlist_additional_time_l2512_251242

/-- The number of additional minutes needed for a playlist -/
def additional_minutes_needed (three_minute_songs : ℕ) (two_minute_songs : ℕ) (total_run_time : ℕ) : ℕ :=
  total_run_time - (three_minute_songs * 3 + two_minute_songs * 2)

/-- Theorem: Given Stan's playlist and run time, he needs 40 more minutes of songs -/
theorem stan_playlist_additional_time :
  additional_minutes_needed 10 15 100 = 40 := by
  sorry

#eval additional_minutes_needed 10 15 100

end NUMINAMATH_CALUDE_stan_playlist_additional_time_l2512_251242


namespace NUMINAMATH_CALUDE_real_part_of_z_l2512_251218

theorem real_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) : 
  Complex.re z = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2512_251218


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l2512_251213

theorem vector_dot_product_problem (a b : ℝ × ℝ) : 
  a = (0, 1) → b = (-1, 1) → (3 • a + 2 • b) • b = 7 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l2512_251213


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l2512_251231

/-- The number of ways to arrange the digits 3, 0, 5, 7, 0 into a 5-digit number -/
def digit_arrangements : ℕ :=
  let digits : Multiset ℕ := {3, 0, 5, 7, 0}
  let total_arrangements := Nat.factorial 5 / (Nat.factorial 2)  -- Total permutations with repetition
  let arrangements_starting_with_zero := Nat.factorial 4 / (Nat.factorial 2)  -- Arrangements starting with 0
  total_arrangements - arrangements_starting_with_zero

/-- The theorem stating that the number of valid arrangements is 48 -/
theorem valid_arrangements_count : digit_arrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l2512_251231


namespace NUMINAMATH_CALUDE_calculation_proof_l2512_251237

theorem calculation_proof :
  (3 / (-1/2) - (2/5 - 1/3) * 15 = -7) ∧
  ((-3)^2 - (-2)^3 * (-1/4) - (-1 + 6) = 2) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l2512_251237


namespace NUMINAMATH_CALUDE_simplify_equation_l2512_251227

theorem simplify_equation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 7*x^2 + x + 1 = 0 ↔ x^2*(y^2 + y - 9) = 0 :=
by sorry

end NUMINAMATH_CALUDE_simplify_equation_l2512_251227


namespace NUMINAMATH_CALUDE_side_face_area_is_288_l2512_251220

/-- Represents a rectangular box with length, width, and height. -/
structure RectangularBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular box. -/
def volume (box : RectangularBox) : ℝ :=
  box.length * box.width * box.height

/-- Calculates the area of the front face of a rectangular box. -/
def frontFaceArea (box : RectangularBox) : ℝ :=
  box.length * box.width

/-- Calculates the area of the top face of a rectangular box. -/
def topFaceArea (box : RectangularBox) : ℝ :=
  box.length * box.height

/-- Calculates the area of the side face of a rectangular box. -/
def sideFaceArea (box : RectangularBox) : ℝ :=
  box.width * box.height

/-- Theorem stating that given the conditions, the area of the side face is 288. -/
theorem side_face_area_is_288 (box : RectangularBox) 
  (h1 : frontFaceArea box = (1/2) * topFaceArea box)
  (h2 : topFaceArea box = (3/2) * sideFaceArea box)
  (h3 : volume box = 5184) :
  sideFaceArea box = 288 := by
  sorry

end NUMINAMATH_CALUDE_side_face_area_is_288_l2512_251220


namespace NUMINAMATH_CALUDE_trampoline_jumps_l2512_251291

/-- The number of times Ronald jumped on the trampoline -/
def ronald_jumps : ℕ := 157

/-- The additional number of times Rupert jumped compared to Ronald -/
def rupert_additional_jumps : ℕ := 86

/-- The number of times Rupert jumped on the trampoline -/
def rupert_jumps : ℕ := ronald_jumps + rupert_additional_jumps

/-- The average number of jumps between the two brothers -/
def average_jumps : ℕ := (ronald_jumps + rupert_jumps) / 2

/-- The total number of jumps made by both Rupert and Ronald -/
def total_jumps : ℕ := ronald_jumps + rupert_jumps

theorem trampoline_jumps :
  average_jumps = 200 ∧ total_jumps = 400 := by
  sorry

end NUMINAMATH_CALUDE_trampoline_jumps_l2512_251291


namespace NUMINAMATH_CALUDE_first_part_length_l2512_251263

/-- Proves that given a 60 km trip with two parts, where the second part is traveled at half the speed
    of the first part, and the average speed of the entire trip is 32 km/h, the length of the first
    part of the trip is 30 km. -/
theorem first_part_length
  (total_distance : ℝ)
  (speed_first_part : ℝ)
  (speed_second_part : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : speed_second_part = speed_first_part / 2)
  (h3 : average_speed = 32)
  : ∃ (first_part_length : ℝ),
    first_part_length = 30 ∧
    first_part_length / speed_first_part +
    (total_distance - first_part_length) / speed_second_part =
    total_distance / average_speed :=
by sorry

end NUMINAMATH_CALUDE_first_part_length_l2512_251263


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2512_251244

theorem quadratic_function_range (k m : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 2 * k * x + m > 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4 * x₁ + k = 0 ∧ x₂^2 - 4 * x₂ + k = 0) →
  (∀ k' : ℤ, k' > k → 
    (∃ x : ℝ, 2 * x^2 - 2 * k' * x + m ≤ 0) ∨
    (∀ x₁ x₂ : ℝ, x₁ = x₂ ∨ x₁^2 - 4 * x₁ + k' ≠ 0 ∨ x₂^2 - 4 * x₂ + k' ≠ 0)) →
  m > 9/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l2512_251244


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_l2512_251262

theorem units_digit_factorial_sum : 
  (1 + 2 + 6 + (24 % 10)) % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_l2512_251262


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l2512_251285

theorem reciprocal_of_negative_half :
  (1 : ℚ) / (-1/2 : ℚ) = -2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l2512_251285


namespace NUMINAMATH_CALUDE_david_biology_marks_l2512_251200

/-- Calculates David's marks in Biology given his marks in other subjects and his average -/
def davidsBiologyMarks (english : ℕ) (mathematics : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + mathematics + physics + chemistry)

theorem david_biology_marks :
  davidsBiologyMarks 51 65 82 67 70 = 85 := by
  sorry

end NUMINAMATH_CALUDE_david_biology_marks_l2512_251200


namespace NUMINAMATH_CALUDE_evaluate_expression_l2512_251205

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  y * (y - 5 * x + 2) = -15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2512_251205


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l2512_251280

/-- Calculates the length of the second train given the speeds of two trains, 
    the time they take to clear each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (clear_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * 1000 / 3600
  let total_distance := relative_speed_ms * clear_time
  total_distance - length1

/-- The length of the second train is approximately 165.12 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, |second_train_length 80 65 7.596633648618456 141 - 165.12| < ε :=
sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l2512_251280


namespace NUMINAMATH_CALUDE_min_value_sum_l2512_251207

theorem min_value_sum (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁^2 + x₂^2 + x₃^2 + x₄^2 = 4) : 
  x₁ / (1 - x₁^2) + x₂ / (1 - x₂^2) + x₃ / (1 - x₃^2) + x₄ / (1 - x₄^2) ≥ 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l2512_251207


namespace NUMINAMATH_CALUDE_football_players_count_l2512_251260

/-- Represents the number of players for each sport type -/
structure PlayerCounts where
  cricket : Nat
  hockey : Nat
  softball : Nat
  total : Nat

/-- Calculates the number of football players given the counts of other players -/
def footballPlayers (counts : PlayerCounts) : Nat :=
  counts.total - (counts.cricket + counts.hockey + counts.softball)

/-- Theorem stating that the number of football players is 11 given the specific counts -/
theorem football_players_count (counts : PlayerCounts)
  (h1 : counts.cricket = 12)
  (h2 : counts.hockey = 17)
  (h3 : counts.softball = 10)
  (h4 : counts.total = 50) :
  footballPlayers counts = 11 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l2512_251260


namespace NUMINAMATH_CALUDE_efficient_elimination_of_y_l2512_251204

theorem efficient_elimination_of_y (x y : ℝ) :
  (3 * x - 2 * y = 3) →
  (4 * x + y = 15) →
  ∃ k : ℝ, (2 * (4 * x + y) + (3 * x - 2 * y) = k) ∧ (11 * x = 33) :=
by
  sorry

end NUMINAMATH_CALUDE_efficient_elimination_of_y_l2512_251204


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l2512_251293

theorem sum_of_fractions_geq_one (x y z : ℝ) :
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l2512_251293


namespace NUMINAMATH_CALUDE_iron_conducts_electricity_is_deductive_l2512_251212

-- Define the set of all substances
def Substance : Type := String

-- Define the property of conducting electricity
def conductsElectricity : Substance → Prop := sorry

-- Define the property of being a metal
def isMetal : Substance → Prop := sorry

-- Define iron as a substance
def iron : Substance := "iron"

-- Define the concept of deductive reasoning
def isDeductiveReasoning (premise1 premise2 conclusion : Prop) : Prop := sorry

-- Theorem statement
theorem iron_conducts_electricity_is_deductive :
  (∀ x, isMetal x → conductsElectricity x) →  -- All metals conduct electricity
  isMetal iron →                              -- Iron is a metal
  isDeductiveReasoning 
    (∀ x, isMetal x → conductsElectricity x)
    (isMetal iron)
    (conductsElectricity iron) :=
by
  sorry

end NUMINAMATH_CALUDE_iron_conducts_electricity_is_deductive_l2512_251212


namespace NUMINAMATH_CALUDE_convention_center_tables_l2512_251269

/-- The number of tables in the convention center. -/
def num_tables : ℕ := 26

/-- The number of chairs around each table. -/
def chairs_per_table : ℕ := 8

/-- The number of legs each chair has. -/
def legs_per_chair : ℕ := 4

/-- The number of legs each table has. -/
def legs_per_table : ℕ := 5

/-- The number of extra chairs not linked with any table. -/
def extra_chairs : ℕ := 10

/-- The total number of legs from tables and chairs. -/
def total_legs : ℕ := 1010

theorem convention_center_tables :
  num_tables * chairs_per_table * legs_per_chair +
  num_tables * legs_per_table +
  extra_chairs * legs_per_chair = total_legs :=
by sorry

end NUMINAMATH_CALUDE_convention_center_tables_l2512_251269


namespace NUMINAMATH_CALUDE_cube_root_nested_l2512_251271

theorem cube_root_nested (N : ℝ) (h : N > 1) :
  (N * (N * N^(1/3))^(1/3))^(1/3) = N^(13/27) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_nested_l2512_251271


namespace NUMINAMATH_CALUDE_proportional_relation_l2512_251261

theorem proportional_relation (x y z : ℝ) (k₁ k₂ : ℝ) :
  (∃ m : ℝ, x = m * y^3) →  -- x is directly proportional to y^3
  (∃ n : ℝ, y * z = n) →    -- y is inversely proportional to z
  (x = 5 ∧ z = 16) →        -- x = 5 when z = 16
  (z = 64 → x = 5/64) :=    -- x = 5/64 when z = 64
by sorry

end NUMINAMATH_CALUDE_proportional_relation_l2512_251261


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l2512_251274

theorem rectangle_area_perimeter_relation :
  ∀ a b : ℕ,
  a > 10 →
  a * b = 5 * (2 * a + 2 * b) →
  2 * a + 2 * b = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l2512_251274


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_l2512_251272

structure GeometricSpace where
  Line : Type
  Plane : Type
  is_parallel : Line → Plane → Prop
  intersect : Plane → Plane → Line
  line_parallel : Line → Line → Prop

theorem line_parallel_to_intersection
  (S : GeometricSpace)
  (l : S.Line)
  (p1 p2 : S.Plane)
  (h1 : S.is_parallel l p1)
  (h2 : S.is_parallel l p2)
  (h3 : p1 ≠ p2) :
  S.line_parallel l (S.intersect p1 p2) :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_l2512_251272


namespace NUMINAMATH_CALUDE_solve_for_y_l2512_251206

theorem solve_for_y (x y : ℤ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2512_251206


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l2512_251297

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Fibonacci sequence modulo 9 -/
def fibMod9 (n : ℕ) : Fin 9 :=
  (fib n).mod 9

/-- The period of Fibonacci sequence modulo 9 -/
def fibMod9Period : ℕ := 24

theorem fib_150_mod_9 :
  fibMod9 150 = 8 := by sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l2512_251297


namespace NUMINAMATH_CALUDE_toys_ratio_l2512_251230

theorem toys_ratio (s : ℚ) : 
  (s * 20 = (142 - 20 - s * 20) - 2) →
  (s * 20 + (142 - 20 - s * 20) + 20 = 142) →
  (s = 3) :=
by sorry

end NUMINAMATH_CALUDE_toys_ratio_l2512_251230


namespace NUMINAMATH_CALUDE_total_problems_solved_l2512_251257

theorem total_problems_solved (initial_problems : Nat) (additional_problems : Nat) : 
  initial_problems = 45 → additional_problems = 18 → initial_problems + additional_problems = 63 :=
by sorry

end NUMINAMATH_CALUDE_total_problems_solved_l2512_251257


namespace NUMINAMATH_CALUDE_superior_rainbow_max_quantity_l2512_251265

/-- Represents the mixing ratios for Superior Rainbow paint -/
structure MixingRatios where
  red : Rat
  white : Rat
  blue : Rat
  yellow : Rat

/-- Represents the available paint quantities -/
structure AvailablePaint where
  red : Nat
  white : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the maximum quantity of Superior Rainbow paint -/
def maxSuperiorRainbow (ratios : MixingRatios) (available : AvailablePaint) : Nat :=
  sorry

/-- Theorem: The maximum quantity of Superior Rainbow paint is 121 pints -/
theorem superior_rainbow_max_quantity :
  let ratios : MixingRatios := ⟨3/4, 2/3, 1/4, 1/6⟩
  let available : AvailablePaint := ⟨50, 45, 20, 15⟩
  maxSuperiorRainbow ratios available = 121 := by
  sorry

end NUMINAMATH_CALUDE_superior_rainbow_max_quantity_l2512_251265


namespace NUMINAMATH_CALUDE_condition1_correct_condition2_correct_condition3_correct_l2512_251284

-- Define the number of teachers, male students, and female students
def num_teachers : ℕ := 2
def num_male_students : ℕ := 3
def num_female_students : ℕ := 3

-- Define the total number of people
def total_people : ℕ := num_teachers + num_male_students + num_female_students

-- Function to calculate the number of arrangements for condition 1
def arrangements_condition1 : ℕ := sorry

-- Function to calculate the number of arrangements for condition 2
def arrangements_condition2 : ℕ := sorry

-- Function to calculate the number of arrangements for condition 3
def arrangements_condition3 : ℕ := sorry

-- Theorem for condition 1
theorem condition1_correct : arrangements_condition1 = 4320 := by sorry

-- Theorem for condition 2
theorem condition2_correct : arrangements_condition2 = 30240 := by sorry

-- Theorem for condition 3
theorem condition3_correct : arrangements_condition3 = 6720 := by sorry

end NUMINAMATH_CALUDE_condition1_correct_condition2_correct_condition3_correct_l2512_251284


namespace NUMINAMATH_CALUDE_marble_remainder_l2512_251288

theorem marble_remainder (n m : ℤ) : ∃ k : ℤ, (8*n + 5) + (8*m + 7) + 6 = 8*k + 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_remainder_l2512_251288


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2512_251222

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2512_251222


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2512_251281

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12 -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 2 →  -- Two sides are 5, one side is 2
  (a = b ∨ b = c ∨ a = c) →  -- The triangle is isosceles
  a + b + c = 12 :=  -- The perimeter is 12
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2512_251281


namespace NUMINAMATH_CALUDE_square_of_binomial_l2512_251292

theorem square_of_binomial (a : ℝ) : 
  (∃ b c : ℝ, ∀ x, 9*x^2 - 18*x + a = (b*x + c)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2512_251292


namespace NUMINAMATH_CALUDE_eighth_term_is_23_l2512_251258

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem eighth_term_is_23 :
  arithmetic_sequence 2 3 8 = 23 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_23_l2512_251258


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2512_251224

theorem sqrt_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a * b = c * d) (h2 : a + b > c + d) : 
  Real.sqrt a + Real.sqrt b > Real.sqrt c + Real.sqrt d :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2512_251224


namespace NUMINAMATH_CALUDE_f_is_log_x_range_l2512_251221

noncomputable section

variable (a : ℝ) (f g : ℝ → ℝ)

-- Define g(x) = a^x
def g_def : g = fun x ↦ a^x := by sorry

-- Define f(x) as symmetric to g(x) with respect to y = x
def f_symmetric : ∀ x y, f x = y ↔ g y = x := by sorry

-- Part 1: Prove that f(x) = log_a x
theorem f_is_log : f = fun x ↦ Real.log x / Real.log a := by sorry

-- Part 2: Prove the range of x when a > 1 and f(x) < f(2)
theorem x_range (h : a > 1) : 
  ∀ x, f x < f 2 ↔ 0 < x ∧ x < a^2 := by sorry

end

end NUMINAMATH_CALUDE_f_is_log_x_range_l2512_251221


namespace NUMINAMATH_CALUDE_total_birds_and_storks_l2512_251254

def birds_and_storks (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ) : ℕ :=
  initial_birds + initial_storks + additional_storks

theorem total_birds_and_storks :
  birds_and_storks 3 4 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_and_storks_l2512_251254


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2512_251234

/-- Perimeter of triangle PF₁F₂ for a specific ellipse -/
theorem ellipse_triangle_perimeter :
  ∀ (a b c : ℝ) (P F₁ F₂ : ℝ × ℝ),
  -- Ellipse equation
  (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | x^2 / a^2 + y^2 / 2 = 1} → x^2 / a^2 + y^2 / 2 = 1) →
  -- F₁ and F₂ are foci
  F₁.1 = -c ∧ F₁.2 = 0 ∧ F₂.1 = c ∧ F₂.2 = 0 →
  -- P is on the ellipse
  P ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / 2 = 1} →
  -- F₁ is symmetric to y = -x at P
  P.1 = -F₁.2 ∧ P.2 = -F₁.1 →
  -- Perimeter of triangle PF₁F₂
  dist P F₁ + dist P F₂ + dist F₁ F₂ = 4 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2512_251234


namespace NUMINAMATH_CALUDE_sin_squared_value_l2512_251289

theorem sin_squared_value (θ : Real) 
  (h : Real.cos θ ^ 4 + Real.sin θ ^ 4 + (Real.cos θ * Real.sin θ) ^ 4 + 
       1 / (Real.cos θ ^ 4 + Real.sin θ ^ 4) = 41 / 16) : 
  Real.sin θ ^ 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_value_l2512_251289


namespace NUMINAMATH_CALUDE_golden_ratio_cubic_l2512_251286

theorem golden_ratio_cubic (p q : ℚ) : 
  let x : ℝ := (Real.sqrt 5 - 1) / 2
  x^3 + p * x + q = 0 → p + q = -1 := by
sorry

end NUMINAMATH_CALUDE_golden_ratio_cubic_l2512_251286


namespace NUMINAMATH_CALUDE_product_decimal_places_l2512_251240

/-- A function that returns the number of decimal places in a decimal number -/
def decimal_places (x : ℚ) : ℕ :=
  sorry

/-- The product of two decimal numbers with one and two decimal places respectively has three decimal places -/
theorem product_decimal_places (a b : ℚ) :
  decimal_places a = 1 → decimal_places b = 2 → decimal_places (a * b) = 3 :=
sorry

end NUMINAMATH_CALUDE_product_decimal_places_l2512_251240


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2512_251225

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h1 : area = 180) (h2 : leg = 30) :
  ∃ (other_leg : ℝ) (hypotenuse : ℝ),
    area = (1 / 2) * leg * other_leg ∧
    hypotenuse^2 = leg^2 + other_leg^2 ∧
    leg + other_leg + hypotenuse = 42 + 2 * Real.sqrt 261 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2512_251225


namespace NUMINAMATH_CALUDE_gcd_65_130_l2512_251210

theorem gcd_65_130 : Nat.gcd 65 130 = 65 := by
  sorry

end NUMINAMATH_CALUDE_gcd_65_130_l2512_251210


namespace NUMINAMATH_CALUDE_pharmacist_weights_existence_l2512_251253

theorem pharmacist_weights_existence :
  ∃ (a b c : ℝ), 
    a < b ∧ b < c ∧
    a + b = 100 ∧
    a + c = 101 ∧
    b + c = 102 ∧
    a < 90 ∧ b < 90 ∧ c < 90 := by
  sorry

end NUMINAMATH_CALUDE_pharmacist_weights_existence_l2512_251253


namespace NUMINAMATH_CALUDE_average_cat_weight_l2512_251209

def cat_weights : List ℝ := [12, 12, 14.7, 9.3]

theorem average_cat_weight :
  (cat_weights.sum / cat_weights.length) = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_cat_weight_l2512_251209


namespace NUMINAMATH_CALUDE_inverse_38_mod_53_l2512_251239

theorem inverse_38_mod_53 (h : (16⁻¹ : ZMod 53) = 20) : (38⁻¹ : ZMod 53) = 25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_38_mod_53_l2512_251239


namespace NUMINAMATH_CALUDE_equation_solution_l2512_251270

theorem equation_solution :
  let f (x : ℂ) := -x^2 - (4*x + 2)/(x + 2)
  ∃ (s : Finset ℂ), s.card = 3 ∧ 
    (∀ x ∈ s, f x = 0) ∧
    (∃ (a b : ℂ), s = {-1, a, b} ∧ a + b = -2 ∧ a * b = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2512_251270


namespace NUMINAMATH_CALUDE_min_max_values_l2512_251295

theorem min_max_values : 
  (∀ a b : ℝ, a > 0 → b > 0 → a * b = 2 → a + 2 * b ≥ 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 2 ∧ a + 2 * b = 4) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = 1 → a + b ≤ Real.sqrt 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = 1 ∧ a + b = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_l2512_251295


namespace NUMINAMATH_CALUDE_abs_lt_sufficient_not_necessary_l2512_251268

theorem abs_lt_sufficient_not_necessary (a b : ℝ) (ha : a > 0) :
  (∀ a b, (abs a < b) → (-a < b)) ∧
  (∃ a b, a > 0 ∧ (-a < b) ∧ (abs a ≥ b)) :=
sorry

end NUMINAMATH_CALUDE_abs_lt_sufficient_not_necessary_l2512_251268


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l2512_251259

theorem complex_in_fourth_quadrant : ∃ (z : ℂ), z = Complex.I * (-2 - Complex.I) ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l2512_251259


namespace NUMINAMATH_CALUDE_abc_inequality_l2512_251229

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b * c * (a + b + c) = a * b + b * c + c * a) :
  5 * (a + b + c) ≥ 7 + 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2512_251229


namespace NUMINAMATH_CALUDE_carpet_cut_length_l2512_251290

theorem carpet_cut_length (square_area : ℝ) (room_area : ℝ) : 
  square_area = 169 →
  room_area = 143 →
  (Real.sqrt square_area - room_area / Real.sqrt square_area) = 2 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cut_length_l2512_251290


namespace NUMINAMATH_CALUDE_car_gasoline_theorem_l2512_251277

/-- Represents the relationship between remaining gasoline and distance traveled for a car --/
def gasoline_function (x : ℝ) : ℝ := 50 - 0.1 * x

/-- Represents the valid range for the distance traveled --/
def valid_distance (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 500

theorem car_gasoline_theorem :
  ∀ x : ℝ,
  valid_distance x →
  (∀ y : ℝ, y = gasoline_function x → y = 50 - 0.1 * x) ∧
  (x = 200 → gasoline_function x = 30) :=
by sorry

end NUMINAMATH_CALUDE_car_gasoline_theorem_l2512_251277


namespace NUMINAMATH_CALUDE_not_always_swappable_renumbering_l2512_251264

-- Define a type for cities
def City : Type := ℕ

-- Define a type for the connection list
def ConnectionList : Type := List (City × City)

-- Function to check if a list is valid (placeholder)
def isValidList (list : ConnectionList) : Prop := sorry

-- Function to represent renumbering of cities
def renumber (oldNum newNum : City) (list : ConnectionList) : ConnectionList := sorry

-- Theorem statement
theorem not_always_swappable_renumbering :
  ∃ (list : ConnectionList) (M N : City),
    isValidList list ∧
    (∀ X Y : City, isValidList (renumber X Y list)) ∧
    ¬(isValidList (renumber M N (renumber N M list))) :=
sorry

end NUMINAMATH_CALUDE_not_always_swappable_renumbering_l2512_251264


namespace NUMINAMATH_CALUDE_ab_equals_zero_l2512_251217

theorem ab_equals_zero (a b : ℝ) 
  (h1 : (4 : ℝ) ^ a = 256 ^ (b + 1))
  (h2 : (27 : ℝ) ^ b = 3 ^ (a - 2)) : 
  a * b = 0 := by sorry

end NUMINAMATH_CALUDE_ab_equals_zero_l2512_251217


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2512_251241

theorem equilateral_triangle_area_perimeter_ratio :
  ∀ s : ℝ,
  s > 0 →
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  s = 6 →
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2512_251241


namespace NUMINAMATH_CALUDE_fixed_points_equality_implies_a_bound_l2512_251214

/-- Given a function f(x) = x^2 - 2x + a, if the set of fixed points of f is equal to the set of fixed points of f ∘ f, then a is greater than or equal to 5/4. -/
theorem fixed_points_equality_implies_a_bound (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + a
  ({x : ℝ | f x = x} = {x : ℝ | f (f x) = x}) →
  a ≥ 5/4 := by
sorry

end NUMINAMATH_CALUDE_fixed_points_equality_implies_a_bound_l2512_251214


namespace NUMINAMATH_CALUDE_odd_function_and_inequality_l2512_251235

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (b - 2^x) / (2^(x+1) + a)

theorem odd_function_and_inequality (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1) ∧
  (∀ t, f 2 1 (t^2 - 2*t) + f 2 1 (2*t^2 - 1) < 0 ↔ t > 1 ∨ t < -1/3) :=
sorry

end NUMINAMATH_CALUDE_odd_function_and_inequality_l2512_251235


namespace NUMINAMATH_CALUDE_bicyclist_average_speed_l2512_251250

/-- Proves that the average speed of a bicyclist is 18 km/h given the specified conditions -/
theorem bicyclist_average_speed :
  let total_distance : ℝ := 450
  let first_part_distance : ℝ := 300
  let second_part_distance : ℝ := total_distance - first_part_distance
  let first_part_speed : ℝ := 20
  let second_part_speed : ℝ := 15
  let first_part_time : ℝ := first_part_distance / first_part_speed
  let second_part_time : ℝ := second_part_distance / second_part_speed
  let total_time : ℝ := first_part_time + second_part_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_bicyclist_average_speed_l2512_251250


namespace NUMINAMATH_CALUDE_quadratic_roots_and_reciprocals_l2512_251266

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * (k + 1) * x + k - 1

-- Theorem statement
theorem quadratic_roots_and_reciprocals (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0) ↔ 
  (k > -1/3 ∧ k ≠ 0) ∧
  ¬∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧ 1/x₁ + 1/x₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_reciprocals_l2512_251266


namespace NUMINAMATH_CALUDE_probability_of_identical_cubes_l2512_251219

/-- Represents the three possible colors for painting cube faces -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents a cube with six colored faces -/
def Cube := Fin 6 → Color

/-- The number of ways to paint a single cube -/
def waysToColorCube : ℕ := 729

/-- The total number of ways to paint three cubes -/
def totalWaysToPaintThreeCubes : ℕ := 387420489

/-- The number of ways to paint three cubes so they are rotationally identical -/
def waysToColorIdenticalCubes : ℕ := 633

/-- Checks if two cubes are rotationally identical -/
def areRotationallyIdentical (c1 c2 : Cube) : Prop := sorry

/-- The probability of three independently painted cubes being rotationally identical -/
theorem probability_of_identical_cubes :
  (waysToColorIdenticalCubes : ℚ) / totalWaysToPaintThreeCubes = 211 / 129140163 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_identical_cubes_l2512_251219


namespace NUMINAMATH_CALUDE_nines_in_range_70_l2512_251236

def count_nines (n : ℕ) : ℕ :=
  (n / 10) + (if n % 10 ≥ 9 then 1 else 0)

theorem nines_in_range_70 : count_nines 70 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nines_in_range_70_l2512_251236


namespace NUMINAMATH_CALUDE_not_all_axially_symmetric_figures_have_one_axis_l2512_251251

/-- A type representing geometric figures -/
structure Figure where
  -- Add necessary fields here
  
/-- Predicate to check if a figure is axially symmetric -/
def is_axially_symmetric (f : Figure) : Prop :=
  sorry

/-- Function to count the number of axes of symmetry for a figure -/
def count_axes_of_symmetry (f : Figure) : ℕ :=
  sorry

/-- Theorem stating that not all axially symmetric figures have only one axis of symmetry -/
theorem not_all_axially_symmetric_figures_have_one_axis :
  ¬ (∀ f : Figure, is_axially_symmetric f → count_axes_of_symmetry f = 1) :=
sorry

end NUMINAMATH_CALUDE_not_all_axially_symmetric_figures_have_one_axis_l2512_251251


namespace NUMINAMATH_CALUDE_parabola_single_intersection_l2512_251299

/-- A parabola y = x^2 + 4x + 5 - m intersects the x-axis at only one point if and only if m = 1 -/
theorem parabola_single_intersection (m : ℝ) : 
  (∃! x, x^2 + 4*x + 5 - m = 0) ↔ m = 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_single_intersection_l2512_251299


namespace NUMINAMATH_CALUDE_correct_calculation_l2512_251275

theorem correct_calculation (a b x p q : ℝ) : 
  (∀ a ≠ 0, (a * b^4)^4 ≠ a * b^8) ∧ 
  (∀ p q, (-3 * p * q)^2 ≠ -6 * p^2 * q^2) ∧ 
  (∀ x, x^2 - 1/2 * x + 1/4 ≠ (x - 1/2)^2) ∧ 
  (∀ a, 3 * (a^2)^3 - 6 * a^6 = -3 * a^6) := by
sorry

end NUMINAMATH_CALUDE_correct_calculation_l2512_251275


namespace NUMINAMATH_CALUDE_min_abs_a_for_solvable_equation_l2512_251287

theorem min_abs_a_for_solvable_equation :
  ∀ (a b : ℤ),
  (a + 2 * b = 32) →
  (∀ a' : ℤ, a' > 0 ∧ (∃ b' : ℤ, a' + 2 * b' = 32) → a' ≥ 4) →
  (∃ b'' : ℤ, (-2) + 2 * b'' = 32) →
  (∃ a₀ : ℤ, |a₀| = 2 ∧ (∃ b₀ : ℤ, a₀ + 2 * b₀ = 32) ∧
    ∀ a' : ℤ, (∃ b' : ℤ, a' + 2 * b' = 32) → |a'| ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_abs_a_for_solvable_equation_l2512_251287


namespace NUMINAMATH_CALUDE_handicraft_sale_properties_l2512_251233

/-- Represents the daily profit function for a handicraft item sale --/
def daily_profit (x : ℝ) : ℝ :=
  -5 * x^2 + 800 * x - 27500

/-- Represents the daily sales volume function --/
def daily_sales (x : ℝ) : ℝ :=
  50 + 5 * (100 - x)

/-- Theorem stating the properties of the handicraft item sale --/
theorem handicraft_sale_properties :
  let cost : ℝ := 50
  let base_price : ℝ := 100
  let base_sales : ℝ := 50
  ∀ x : ℝ, cost ≤ x ∧ x ≤ base_price →
    (daily_profit x = (x - cost) * daily_sales x) ∧
    (∃ max_profit max_price, 
      max_profit = daily_profit max_price ∧
      max_price = 80 ∧ 
      max_profit = 4500 ∧
      ∀ y, cost ≤ y ∧ y ≤ base_price → daily_profit y ≤ max_profit) ∧
    (∃ min_total_cost,
      min_total_cost = 5000 ∧
      ∀ z, cost ≤ z ∧ z ≤ base_price →
        daily_profit z ≥ 4000 → cost * daily_sales z ≥ min_total_cost) := by
  sorry


end NUMINAMATH_CALUDE_handicraft_sale_properties_l2512_251233


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l2512_251252

theorem largest_n_satisfying_conditions : ∃ (n : ℤ), n = 181 ∧ 
  (∃ (m : ℤ), n^2 = (m+1)^3 - m^3) ∧ 
  (∃ (k : ℤ), 2*n + 79 = k^2) ∧
  (∀ (n' : ℤ), n' > n → 
    (¬∃ (m : ℤ), n'^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ (k : ℤ), 2*n' + 79 = k^2)) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l2512_251252


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l2512_251256

/-- For an infinite geometric series with common ratio 1/4 and sum 40, the second term is 7.5 -/
theorem geometric_series_second_term : 
  ∀ (a : ℝ), 
  (∑' n, a * (1/4)^n) = 40 → 
  a * (1/4) = 7.5 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l2512_251256


namespace NUMINAMATH_CALUDE_correct_sums_count_l2512_251211

theorem correct_sums_count (total : ℕ) (correct : ℕ) (incorrect : ℕ)
  (h1 : incorrect = 2 * correct)
  (h2 : total = correct + incorrect)
  (h3 : total = 24) :
  correct = 8 := by
  sorry

end NUMINAMATH_CALUDE_correct_sums_count_l2512_251211


namespace NUMINAMATH_CALUDE_women_to_total_ratio_l2512_251202

theorem women_to_total_ratio (total_passengers : ℕ) (seated_men : ℕ) : 
  total_passengers = 48 →
  seated_men = 14 →
  ∃ (women men standing_men : ℕ),
    women + men = total_passengers ∧
    standing_men + seated_men = men ∧
    standing_men = men / 8 ∧
    women * 3 = total_passengers * 2 := by
  sorry

end NUMINAMATH_CALUDE_women_to_total_ratio_l2512_251202


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l2512_251296

theorem sum_of_squares_and_products (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 52)
  (h5 : x*y + y*z + z*x = 24) : 
  x + y + z = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l2512_251296


namespace NUMINAMATH_CALUDE_online_store_commission_l2512_251267

/-- Calculates the commission percentage of an online store given the cost price,
    desired profit percentage, and final observed price. -/
theorem online_store_commission
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (observed_price : ℝ)
  (h1 : cost_price = 15)
  (h2 : profit_percentage = 0.1)
  (h3 : observed_price = 19.8) :
  let distributor_price := cost_price * (1 + profit_percentage)
  let commission_percentage := (observed_price / distributor_price - 1) * 100
  commission_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_online_store_commission_l2512_251267


namespace NUMINAMATH_CALUDE_bounded_sequence_with_lcm_condition_l2512_251232

theorem bounded_sequence_with_lcm_condition (n : ℕ) (k : ℕ) (a : Fin k → ℕ) :
  (∀ i : Fin k, 1 ≤ a i) →
  (∀ i j : Fin k, i < j → a i < a j) →
  (∀ i : Fin k, a i ≤ n) →
  (∀ i j : Fin k, Nat.lcm (a i) (a j) ≤ n) →
  k ≤ 2 * Int.floor (Real.sqrt n) :=
by sorry

end NUMINAMATH_CALUDE_bounded_sequence_with_lcm_condition_l2512_251232


namespace NUMINAMATH_CALUDE_sequence_formula_l2512_251247

theorem sequence_formula (a : ℕ+ → ℚ) :
  (∀ n : ℕ+, a (n + 1) / a n = (n + 2 : ℚ) / n) →
  a 1 = 1 →
  ∀ n : ℕ+, a n = n * (n + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l2512_251247


namespace NUMINAMATH_CALUDE_binomial_coefficient_12_5_l2512_251246

theorem binomial_coefficient_12_5 : Nat.choose 12 5 = 792 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_12_5_l2512_251246


namespace NUMINAMATH_CALUDE_triangle_side_length_l2512_251223

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) : 
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →  -- Area condition
  B = Real.pi / 3 →  -- 60° in radians
  a^2 + c^2 = 3 * a * c →  -- Given equation
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2512_251223


namespace NUMINAMATH_CALUDE_password_probability_l2512_251215

-- Define the set of all possible symbols
def AllSymbols : Finset Char := {'!', '@', '#', '$', '%'}

-- Define the set of allowed symbols
def AllowedSymbols : Finset Char := {'!', '@', '#'}

-- Define the set of all possible single digits
def AllDigits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the set of even single digits
def EvenDigits : Finset Nat := {0, 2, 4, 6, 8}

-- Define the set of non-zero single digits
def NonZeroDigits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Theorem statement
theorem password_probability :
  (Finset.card EvenDigits : ℚ) / (Finset.card AllDigits) *
  (Finset.card AllowedSymbols : ℚ) / (Finset.card AllSymbols) *
  (Finset.card NonZeroDigits : ℚ) / (Finset.card AllDigits) =
  27 / 100 := by
sorry

end NUMINAMATH_CALUDE_password_probability_l2512_251215


namespace NUMINAMATH_CALUDE_angle_C_measure_l2512_251282

-- Define the triangle ABC
variable (A B C : ℝ)

-- Define the conditions
axiom scalene : A ≠ B ∧ B ≠ C ∧ A ≠ C
axiom angle_sum : A + B + C = 180
axiom angle_relation : C = A + 40
axiom angle_B : B = 2 * A

-- Theorem to prove
theorem angle_C_measure : C = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l2512_251282


namespace NUMINAMATH_CALUDE_intersection_distance_is_2b_l2512_251249

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- Represents a parabola with focus (p, 0) and directrix x = -p -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- The distance between intersection points of an ellipse and a parabola -/
def intersection_distance (e : Ellipse) (p : Parabola) : ℝ :=
  sorry

/-- Theorem stating the distance between intersection points -/
theorem intersection_distance_is_2b 
  (e : Ellipse) 
  (p : Parabola) 
  (h1 : e.a = 5 ∧ e.b = 4)  -- Ellipse equation condition
  (h2 : p.p = 3)  -- Shared focus condition
  (h3 : ∃ b : ℝ, b > 0 ∧ 
    (b^2 / 6 + 1.5)^2 / 25 + b^2 / 16 = 1)  -- Intersection condition
  : 
  ∃ b : ℝ, intersection_distance e p = 2 * b ∧ 
    b > 0 ∧ 
    (b^2 / 6 + 1.5)^2 / 25 + b^2 / 16 = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_is_2b_l2512_251249


namespace NUMINAMATH_CALUDE_linear_function_max_value_l2512_251201

/-- The maximum value of a linear function y = (5/3)x + 2 over the interval [-3, 3] is 7 -/
theorem linear_function_max_value (x : ℝ) :
  x ∈ Set.Icc (-3 : ℝ) 3 →
  (5/3 : ℝ) * x + 2 ≤ 7 ∧ ∃ x₀, x₀ ∈ Set.Icc (-3 : ℝ) 3 ∧ (5/3 : ℝ) * x₀ + 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_max_value_l2512_251201


namespace NUMINAMATH_CALUDE_min_sum_of_sides_l2512_251228

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (a + b)^2 - c^2 = 4 and C = 60°, then the minimum value of a + b is 4√3/3 -/
theorem min_sum_of_sides (a b c : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : Real.cos (Real.pi / 3) = (a^2 + b^2 - c^2) / (2 * a * b)) :
  ∃ (min_sum : ℝ), min_sum = 4 * Real.sqrt 3 / 3 ∧ ∀ x y, (x + y)^2 - c^2 = 4 → x + y ≥ min_sum :=
sorry


end NUMINAMATH_CALUDE_min_sum_of_sides_l2512_251228


namespace NUMINAMATH_CALUDE_complex_sum_argument_l2512_251208

theorem complex_sum_argument : 
  let z : ℂ := Complex.exp (11 * Real.pi * I / 60) + 
                Complex.exp (31 * Real.pi * I / 60) + 
                Complex.exp (51 * Real.pi * I / 60) + 
                Complex.exp (71 * Real.pi * I / 60) + 
                Complex.exp (91 * Real.pi * I / 60)
  ∃ (r : ℝ), z = r * Complex.exp (17 * Real.pi * I / 20) ∧ r > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_argument_l2512_251208


namespace NUMINAMATH_CALUDE_two_x_equals_two_l2512_251216

theorem two_x_equals_two (h : 1 = x) : 2 * x = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_x_equals_two_l2512_251216


namespace NUMINAMATH_CALUDE_shirt_costs_15_l2512_251294

/-- The cost of one pair of jeans -/
def jeans_cost : ℚ := sorry

/-- The cost of one shirt -/
def shirt_cost : ℚ := sorry

/-- First condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jeans_cost + 2 * shirt_cost = 69

/-- Second condition: 2 pairs of jeans and 3 shirts cost $71 -/
axiom condition2 : 2 * jeans_cost + 3 * shirt_cost = 71

/-- Theorem: The cost of one shirt is $15 -/
theorem shirt_costs_15 : shirt_cost = 15 := by sorry

end NUMINAMATH_CALUDE_shirt_costs_15_l2512_251294


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2512_251243

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem sufficient_not_necessary : 
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2512_251243


namespace NUMINAMATH_CALUDE_model_shop_purchase_l2512_251203

theorem model_shop_purchase : ∃ (c t : ℕ), c > 0 ∧ t > 0 ∧ 5 * c + 8 * t = 31 ∧ c + t = 5 := by
  sorry

end NUMINAMATH_CALUDE_model_shop_purchase_l2512_251203


namespace NUMINAMATH_CALUDE_max_rectangle_area_l2512_251245

theorem max_rectangle_area (l w : ℝ) (h_perimeter : l + w = 10) :
  l * w ≤ 25 :=
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l2512_251245


namespace NUMINAMATH_CALUDE_product_of_sines_equals_one_fourth_l2512_251238

theorem product_of_sines_equals_one_fourth :
  (1 - Real.sin (π/8)) * (1 - Real.sin (3*π/8)) * (1 - Real.sin (5*π/8)) * (1 - Real.sin (7*π/8)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_equals_one_fourth_l2512_251238
