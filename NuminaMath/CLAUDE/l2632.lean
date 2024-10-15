import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_proposition_l2632_263228

theorem negation_of_proposition :
  (¬ ∀ (a : ℝ) (n : ℕ), n > 0 → (a ≠ n → a * n ≠ 2 * n)) ↔
  (∃ (a : ℝ) (n : ℕ), n > 0 ∧ a ≠ n ∧ a * n = 2 * n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2632_263228


namespace NUMINAMATH_CALUDE_sandys_age_l2632_263224

/-- Given that Molly is 16 years older than Sandy and their ages are in the ratio 7:9, prove that Sandy is 56 years old. -/
theorem sandys_age (sandy_age molly_age : ℕ) : 
  molly_age = sandy_age + 16 →
  sandy_age * 9 = molly_age * 7 →
  sandy_age = 56 := by
  sorry

end NUMINAMATH_CALUDE_sandys_age_l2632_263224


namespace NUMINAMATH_CALUDE_hyperbola_from_ellipse_and_asymptote_l2632_263225

/-- Given ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- Given asymptote equation -/
def asymptote_equation (x y : ℝ) : Prop := x - Real.sqrt 2 * y = 0

/-- Hyperbola equation to be proved -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / (8 * Real.sqrt 3 / 3) - y^2 / (4 * Real.sqrt 3 / 3) = 1

theorem hyperbola_from_ellipse_and_asymptote :
  ∀ x y : ℝ,
  (∃ a b : ℝ, ellipse_equation a b ∧
    (∀ c d : ℝ, hyperbola_equation c d → (c - a)^2 + (d - b)^2 = (c + a)^2 + (d + b)^2)) →
  asymptote_equation x y →
  hyperbola_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_from_ellipse_and_asymptote_l2632_263225


namespace NUMINAMATH_CALUDE_probability_A_B_same_group_l2632_263232

-- Define the score ranges and their frequencies
def score_ranges : List (ℕ × ℕ × ℕ) := [
  (60, 75, 2),
  (75, 90, 3),
  (90, 105, 14),
  (105, 120, 15),
  (120, 135, 12),
  (135, 150, 4)
]

-- Define the total number of students
def total_students : ℕ := 50

-- Define student A's score
def score_A : ℕ := 62

-- Define student B's score
def score_B : ℕ := 140

-- Define the "two-help-one" group formation rule
def two_help_one (s1 s2 s3 : ℕ) : Prop :=
  (s1 ≥ 135 ∧ s1 ≤ 150) ∧ (s2 ≥ 135 ∧ s2 ≤ 150) ∧ (s3 ≥ 60 ∧ s3 < 75)

-- Theorem to prove
theorem probability_A_B_same_group :
  ∃ (p : ℚ), p = 1/4 ∧ 
  (p = (number_of_groups_with_A_and_B : ℚ) / (total_number_of_possible_groups : ℚ)) :=
sorry

end NUMINAMATH_CALUDE_probability_A_B_same_group_l2632_263232


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2632_263229

theorem trigonometric_identities :
  (Real.cos (2 * Real.pi / 5) - Real.cos (4 * Real.pi / 5) = Real.sqrt 5 / 2) ∧
  (Real.sin (2 * Real.pi / 7) + Real.sin (4 * Real.pi / 7) - Real.sin (6 * Real.pi / 7) = Real.sqrt 7 / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2632_263229


namespace NUMINAMATH_CALUDE_sqrt_256_squared_plus_100_l2632_263218

theorem sqrt_256_squared_plus_100 : (Real.sqrt 256)^2 + 100 = 356 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_256_squared_plus_100_l2632_263218


namespace NUMINAMATH_CALUDE_misread_number_correction_l2632_263295

theorem misread_number_correction (n : ℕ) (incorrect_avg correct_avg misread_value : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 14)
  (h3 : correct_avg = 15)
  (h4 : misread_value = 26) : 
  ∃ (actual_value : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * incorrect_avg = misread_value - actual_value ∧ 
    actual_value = 16 := by
  sorry

end NUMINAMATH_CALUDE_misread_number_correction_l2632_263295


namespace NUMINAMATH_CALUDE_min_value_theorem_l2632_263227

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.log (x + y) = 0 → 2/x + 3/y ≥ 5 + 2 * Real.sqrt 6) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.log (x + y) = 0 ∧ 2/x + 3/y = 5 + 2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2632_263227


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2632_263203

/-- Given two vectors a and b in R², prove that if k*a + b is perpendicular to a - b, then k = 1. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 1))
  (h2 : b = (-1, 1))
  (h3 : (k * a.1 + b.1) * (a.1 - b.1) + (k * a.2 + b.2) * (a.2 - b.2) = 0) :
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2632_263203


namespace NUMINAMATH_CALUDE_team_b_size_l2632_263259

/-- Proves that Team B has 9 people given the competition conditions -/
theorem team_b_size (team_a_avg : ℝ) (team_b_avg : ℝ) (total_avg : ℝ) (size_diff : ℕ) :
  team_a_avg = 75 →
  team_b_avg = 73 →
  total_avg = 73.5 →
  size_diff = 6 →
  ∃ (x : ℕ), x + size_diff = 9 ∧
    (team_a_avg * x + team_b_avg * (x + size_diff)) / (x + (x + size_diff)) = total_avg :=
by
  sorry

#check team_b_size

end NUMINAMATH_CALUDE_team_b_size_l2632_263259


namespace NUMINAMATH_CALUDE_odd_function_extension_l2632_263239

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_extension :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  (∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x) →  -- f(x) = x^2 - 2x when x ≥ 0
  (∀ x : ℝ, x < 0 → f x = -x^2 + 2*x) :=  -- f(x) = -x^2 + 2x when x < 0
by sorry

end NUMINAMATH_CALUDE_odd_function_extension_l2632_263239


namespace NUMINAMATH_CALUDE_shaded_region_correct_l2632_263208

-- Define the universal set U and subsets A and B
variable (U : Type) (A B : Set U)

-- Define the shaded region
def shaded_region (U : Type) (A B : Set U) : Set U :=
  (Aᶜ) ∩ (Bᶜ)

-- Theorem statement
theorem shaded_region_correct (U : Type) (A B : Set U) :
  shaded_region U A B = (Aᶜ) ∩ (Bᶜ) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_shaded_region_correct_l2632_263208


namespace NUMINAMATH_CALUDE_sum_of_occurrences_l2632_263214

theorem sum_of_occurrences (a₀ a₁ a₂ a₃ a₄ : ℕ) 
  (sum_constraint : a₀ + a₁ + a₂ + a₃ + a₄ = 5)
  (value_constraint : 0*a₀ + 1*a₁ + 2*a₂ + 3*a₃ + 4*a₄ = 5) :
  a₀ + a₁ + a₂ + a₃ = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_occurrences_l2632_263214


namespace NUMINAMATH_CALUDE_smallest_square_area_l2632_263276

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum side length of a square that can contain two rectangles,
    one of which is rotated 90 degrees -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r1.height) (max r2.width r2.height)

/-- Theorem stating the smallest possible area of the square -/
theorem smallest_square_area (r1 r2 : Rectangle)
  (h1 : r1 = ⟨4, 2⟩ ∨ r1 = ⟨2, 4⟩)
  (h2 : r2 = ⟨5, 3⟩ ∨ r2 = ⟨3, 5⟩) :
  (minSquareSide r1 r2) ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l2632_263276


namespace NUMINAMATH_CALUDE_sum_of_roots_for_f_l2632_263223

def f (x : ℝ) : ℝ := (4*x)^2 - (4*x) + 2

theorem sum_of_roots_for_f (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 10 ∧ f z₂ = 10 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 1/16) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_for_f_l2632_263223


namespace NUMINAMATH_CALUDE_reflected_light_equation_l2632_263241

/-- The incident light line -/
def incident_line (x y : ℝ) : Prop := 2 * x - y + 6 = 0

/-- The reflection line -/
def reflection_line (x y : ℝ) : Prop := y = x

/-- The reflected light line -/
def reflected_line (x y : ℝ) : Prop := x + 2 * y + 18 = 0

/-- 
Given an incident light line 2x - y + 6 = 0 striking the line y = x, 
prove that the reflected light line has the equation x + 2y + 18 = 0.
-/
theorem reflected_light_equation :
  ∀ x y : ℝ, incident_line x y ∧ reflection_line x y → reflected_line x y :=
by sorry

end NUMINAMATH_CALUDE_reflected_light_equation_l2632_263241


namespace NUMINAMATH_CALUDE_chocolate_squares_multiple_l2632_263258

theorem chocolate_squares_multiple (mike_squares jenny_squares : ℕ) 
  (h1 : mike_squares = 20) 
  (h2 : jenny_squares = 65) 
  (h3 : ∃ m : ℕ, jenny_squares = mike_squares * m + 5) : 
  ∃ m : ℕ, m = 3 ∧ jenny_squares = mike_squares * m + 5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_squares_multiple_l2632_263258


namespace NUMINAMATH_CALUDE_exists_correct_coloring_l2632_263248

/-- Represents the color of a square on the board -/
inductive Color
| White
| Black

/-- Represents a position on the board -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the game board -/
def Board := Position → Color

/-- Checks if two positions are adjacent -/
def adjacent (p1 p2 : Position) : Bool :=
  (p1.row = p2.row ∧ (p1.col + 1 = p2.col ∨ p2.col + 1 = p1.col)) ∨
  (p1.col = p2.col ∧ (p1.row + 1 = p2.row ∨ p2.row + 1 = p1.row))

/-- Checks if a position is within the 4x8 board -/
def validPosition (p : Position) : Bool :=
  p.row < 4 ∧ p.col < 8

/-- Inverts the color -/
def invertColor (c : Color) : Color :=
  match c with
  | Color.White => Color.Black
  | Color.Black => Color.White

/-- Applies a move to the board -/
def applyMove (board : Board) (topLeft : Position) : Board :=
  λ p => if p.row ∈ [topLeft.row, topLeft.row + 1] ∧ 
            p.col ∈ [topLeft.col, topLeft.col + 1]
         then invertColor (board p)
         else board p

/-- Checks if the board is correctly colored -/
def isCorrectlyColored (board : Board) : Prop :=
  ∀ p1 p2, validPosition p1 ∧ validPosition p2 ∧ adjacent p1 p2 →
    board p1 ≠ board p2

/-- The main theorem to prove -/
theorem exists_correct_coloring :
  ∃ (finalBoard : Board),
    (∃ (moves : List Position), 
      finalBoard = (moves.foldl applyMove (λ _ => Color.White)) ∧
      isCorrectlyColored finalBoard) :=
sorry

end NUMINAMATH_CALUDE_exists_correct_coloring_l2632_263248


namespace NUMINAMATH_CALUDE_tricycles_in_garage_l2632_263274

/-- The number of tricycles in Zoe's garage --/
def num_tricycles : ℕ := sorry

/-- The total number of wheels in the garage --/
def total_wheels : ℕ := 25

/-- The number of bicycles in the garage --/
def num_bicycles : ℕ := 3

/-- The number of unicycles in the garage --/
def num_unicycles : ℕ := 7

/-- The number of wheels on a bicycle --/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle --/
def wheels_per_tricycle : ℕ := 3

/-- The number of wheels on a unicycle --/
def wheels_per_unicycle : ℕ := 1

/-- Theorem stating that there are 4 tricycles in the garage --/
theorem tricycles_in_garage : num_tricycles = 4 := by
  sorry

end NUMINAMATH_CALUDE_tricycles_in_garage_l2632_263274


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2632_263252

theorem complex_equation_solution (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2632_263252


namespace NUMINAMATH_CALUDE_x_value_when_one_in_set_l2632_263243

theorem x_value_when_one_in_set (x : ℝ) : 
  (1 ∈ ({x, x^2} : Set ℝ)) → x ≠ x^2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_one_in_set_l2632_263243


namespace NUMINAMATH_CALUDE_abie_chips_count_l2632_263282

theorem abie_chips_count (initial : Nat) (given : Nat) (bought : Nat) (final : Nat) : 
  initial = 20 → given = 4 → bought = 6 → final = initial - given + bought → final = 22 := by
  sorry

end NUMINAMATH_CALUDE_abie_chips_count_l2632_263282


namespace NUMINAMATH_CALUDE_inverse_composition_l2632_263291

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- State the given condition
axiom condition : ∀ x, f_inv (g x) = 5 * x + 3

-- State the theorem to be proved
theorem inverse_composition : g_inv (f (-7)) = -2 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_l2632_263291


namespace NUMINAMATH_CALUDE_evaluate_expression_l2632_263217

theorem evaluate_expression (x : ℕ) (h : x = 3) : x^2 + x * (x^(Nat.factorial x)) = 2196 :=
by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2632_263217


namespace NUMINAMATH_CALUDE_sum_equals_350_l2632_263278

theorem sum_equals_350 : 124 + 129 + 106 + 141 + 237 - 500 + 113 = 350 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_350_l2632_263278


namespace NUMINAMATH_CALUDE_problems_per_worksheet_l2632_263268

/-- Given a set of worksheets with some graded and some problems left to grade,
    calculate the number of problems per worksheet. -/
theorem problems_per_worksheet
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (problems_left : ℕ)
  (h1 : total_worksheets = 14)
  (h2 : graded_worksheets = 7)
  (h3 : problems_left = 14)
  (h4 : graded_worksheets < total_worksheets) :
  problems_left / (total_worksheets - graded_worksheets) = 2 :=
by
  sorry

#check problems_per_worksheet

end NUMINAMATH_CALUDE_problems_per_worksheet_l2632_263268


namespace NUMINAMATH_CALUDE_time_after_minutes_l2632_263279

def minutes_after_midnight : ℕ := 2345

def hours_in_day : ℕ := 24

def minutes_in_hour : ℕ := 60

def start_date : String := "January 1, 2022"

theorem time_after_minutes (m : ℕ) (h : m = minutes_after_midnight) :
  (start_date, m) = ("January 2", 15 * minutes_in_hour + 5) := by sorry

end NUMINAMATH_CALUDE_time_after_minutes_l2632_263279


namespace NUMINAMATH_CALUDE_alicia_singles_stats_l2632_263262

/-- Represents a baseball player's hit statistics -/
structure HitStats where
  total : ℕ
  homeRuns : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the number of singles and their percentage of total hits -/
def singlesStats (stats : HitStats) : (ℕ × ℚ) :=
  let singles := stats.total - (stats.homeRuns + stats.triples + stats.doubles)
  let percentage := (singles : ℚ) / (stats.total : ℚ) * 100
  (singles, percentage)

/-- Theorem: Given Alicia's hit statistics, prove that she had 38 singles
    which constitute 76% of her total hits -/
theorem alicia_singles_stats :
  let alicia : HitStats := ⟨50, 2, 3, 7⟩
  singlesStats alicia = (38, 76) := by sorry

end NUMINAMATH_CALUDE_alicia_singles_stats_l2632_263262


namespace NUMINAMATH_CALUDE_right_triangle_existence_l2632_263233

/-- A right triangle with hypotenuse c and angle bisector f of the right angle -/
structure RightTriangle where
  c : ℝ  -- Length of hypotenuse
  f : ℝ  -- Length of angle bisector of right angle
  c_pos : c > 0
  f_pos : f > 0

/-- The condition for the existence of a right triangle given its hypotenuse and angle bisector -/
def constructible (t : RightTriangle) : Prop :=
  t.f < t.c / 2

/-- Theorem stating the condition for the existence of a right triangle -/
theorem right_triangle_existence (t : RightTriangle) :
  constructible t ↔ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = t.c^2 ∧ 
    t.f = (a * b) / (a + b) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l2632_263233


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l2632_263293

/-- Represents a trapezoid with diagonals and sum of bases -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  sum_of_bases : ℝ

/-- Calculates the area of a trapezoid given its diagonals and sum of bases -/
def area (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that a trapezoid with diagonals 12 and 6, and sum of bases 14, has an area of 16√5 -/
theorem trapezoid_area_theorem :
  let t : Trapezoid := { diagonal1 := 12, diagonal2 := 6, sum_of_bases := 14 }
  area t = 16 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l2632_263293


namespace NUMINAMATH_CALUDE_adjacent_i_probability_l2632_263242

theorem adjacent_i_probability : 
  let total_letters : ℕ := 10
  let unique_letters : ℕ := 9
  let repeated_letter : ℕ := 1
  let favorable_arrangements : ℕ := unique_letters.factorial
  let total_arrangements : ℕ := total_letters.factorial / (repeated_letter + 1).factorial
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_adjacent_i_probability_l2632_263242


namespace NUMINAMATH_CALUDE_book_arrangement_problem_l2632_263281

/-- Represents the number of arrangements of books on a shelf. -/
def num_arrangements (n : ℕ) (chinese : ℕ) (math : ℕ) (physics : ℕ) : ℕ := sorry

/-- Theorem stating the number of arrangements for the given problem. -/
theorem book_arrangement_problem :
  num_arrangements 5 2 2 1 = 48 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_problem_l2632_263281


namespace NUMINAMATH_CALUDE_system_solution_unique_equation_no_solution_l2632_263254

-- Problem 1
theorem system_solution_unique (x y : ℝ) : 
  x - 3*y = 4 ∧ 2*x - y = 3 ↔ x = 1 ∧ y = -1 :=
sorry

-- Problem 2
theorem equation_no_solution : 
  ¬∃ x : ℝ, (x ≠ 2) ∧ (1 / (x - 2) + 3 = (1 - x) / (2 - x)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_unique_equation_no_solution_l2632_263254


namespace NUMINAMATH_CALUDE_distinct_values_x9_mod_999_l2632_263270

theorem distinct_values_x9_mod_999 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n < 999) ∧ 
  (∀ x : ℕ, ∃ n ∈ S, x^9 ≡ n [ZMOD 999]) ∧
  Finset.card S = 15 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_x9_mod_999_l2632_263270


namespace NUMINAMATH_CALUDE_parking_lot_car_ratio_l2632_263294

theorem parking_lot_car_ratio :
  let red_cars : ℕ := 28
  let black_cars : ℕ := 75
  (red_cars : ℚ) / black_cars = 28 / 75 :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_car_ratio_l2632_263294


namespace NUMINAMATH_CALUDE_dress_discount_price_l2632_263297

theorem dress_discount_price (d : ℝ) (h : d > 0) : 
  d * (1 - 0.45) * (1 - 0.4) = d * 0.33 := by
sorry

end NUMINAMATH_CALUDE_dress_discount_price_l2632_263297


namespace NUMINAMATH_CALUDE_square_garden_multiple_l2632_263212

theorem square_garden_multiple (a p : ℝ) (h1 : p = 38) (h2 : a = (p / 4)^2) (h3 : ∃ m : ℝ, a = m * p + 14.25) : 
  ∃ m : ℝ, a = m * p + 14.25 ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_square_garden_multiple_l2632_263212


namespace NUMINAMATH_CALUDE_triangle_side_length_triangle_angle_relation_l2632_263264

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C

-- Theorem 1
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 3 * t.c) 
  (h2 : t.b = Real.sqrt 2) 
  (h3 : Real.cos t.B = 2/3) : 
  t.c = Real.sqrt 3 / 3 := by
  sorry

-- Theorem 2
theorem triangle_angle_relation (t : Triangle) 
  (h : Real.sin t.A / t.a = Real.cos t.B / (2 * t.b)) : 
  Real.sin (t.B + π/2) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_triangle_angle_relation_l2632_263264


namespace NUMINAMATH_CALUDE_polynomial_equality_constants_l2632_263284

theorem polynomial_equality_constants (k1 k2 k3 : ℤ) : 
  (∀ x : ℝ, -x^4 - (k1 + 11)*x^3 - k2*x^2 - 8*x - k3 = -(x - 2)*(x^3 - 6*x^2 + 8*x - 4)) ↔ 
  (k1 = -19 ∧ k2 = 20 ∧ k3 = 8) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_constants_l2632_263284


namespace NUMINAMATH_CALUDE_nala_seashell_count_l2632_263240

/-- The number of seashells Nala found on Monday -/
def monday_shells : ℕ := 5

/-- The number of seashells Nala found on Tuesday -/
def tuesday_shells : ℕ := 7

/-- The number of seashells Nala discarded on Tuesday -/
def tuesday_discarded : ℕ := 3

/-- The number of seashells Nala found on Wednesday relative to Monday -/
def wednesday_multiplier : ℕ := 2

/-- The fraction of seashells Nala discarded on Wednesday -/
def wednesday_discard_fraction : ℚ := 1/2

/-- The number of seashells Nala found on Thursday relative to Tuesday -/
def thursday_multiplier : ℕ := 3

/-- The total number of unbroken seashells Nala has by the end of Thursday -/
def total_shells : ℕ := 35

theorem nala_seashell_count : 
  monday_shells + 
  (tuesday_shells - tuesday_discarded) + 
  (wednesday_multiplier * monday_shells - Nat.floor (↑(wednesday_multiplier * monday_shells) * wednesday_discard_fraction)) + 
  (thursday_multiplier * tuesday_shells) = total_shells := by
  sorry

end NUMINAMATH_CALUDE_nala_seashell_count_l2632_263240


namespace NUMINAMATH_CALUDE_corn_stalk_calculation_hilary_corn_stalks_l2632_263213

theorem corn_stalk_calculation (ears_per_stalk : ℕ) 
  (kernels_low : ℕ) (kernels_high : ℕ) (total_kernels : ℕ) : ℕ :=
  let avg_kernels := (kernels_low + kernels_high) / 2
  let total_ears := total_kernels / avg_kernels
  total_ears / ears_per_stalk

theorem hilary_corn_stalks : 
  corn_stalk_calculation 4 500 600 237600 = 108 := by
  sorry

end NUMINAMATH_CALUDE_corn_stalk_calculation_hilary_corn_stalks_l2632_263213


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2632_263220

theorem isosceles_right_triangle_area 
  (h : ℝ) -- hypotenuse length
  (is_isosceles_right : True) -- condition that the triangle is isosceles right
  (hyp_length : h = 6 * Real.sqrt 2) : -- condition for the hypotenuse length
  (1/2) * ((h / Real.sqrt 2) ^ 2) = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2632_263220


namespace NUMINAMATH_CALUDE_median_of_special_list_l2632_263251

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of elements in the list -/
def total_elements : ℕ := sum_of_first_n 200

/-- The position of the median elements -/
def median_position : ℕ × ℕ := (total_elements / 2, total_elements / 2 + 1)

/-- The value that appears at the median positions -/
def median_value : ℕ := 141

/-- The median of the list -/
def list_median : ℚ := (median_value : ℚ)

theorem median_of_special_list : list_median = 141 := by sorry

end NUMINAMATH_CALUDE_median_of_special_list_l2632_263251


namespace NUMINAMATH_CALUDE_fraction_operation_equivalence_l2632_263266

theorem fraction_operation_equivalence (x : ℚ) :
  x * (5/6) / (2/7) = x * (35/12) := by
sorry

end NUMINAMATH_CALUDE_fraction_operation_equivalence_l2632_263266


namespace NUMINAMATH_CALUDE_tree_planting_activity_l2632_263269

theorem tree_planting_activity (boys girls : ℕ) : 
  (boys = 2 * girls + 15) →
  (girls = boys / 3 + 6) →
  (boys = 81 ∧ girls = 33) :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_activity_l2632_263269


namespace NUMINAMATH_CALUDE_joes_fast_food_cost_l2632_263231

/-- Calculates the cost of a purchase at Joe's Fast Food -/
def calculate_cost (sandwich_price : ℕ) (soda_price : ℕ) (sandwich_count : ℕ) (soda_count : ℕ) (bulk_discount : ℕ) (bulk_threshold : ℕ) : ℕ :=
  let total_items := sandwich_count + soda_count
  let subtotal := sandwich_price * sandwich_count + soda_price * soda_count
  if total_items > bulk_threshold then subtotal - bulk_discount else subtotal

/-- The cost of purchasing 6 sandwiches and 6 sodas at Joe's Fast Food is 37 dollars -/
theorem joes_fast_food_cost : calculate_cost 4 3 6 6 5 10 = 37 := by
  sorry

end NUMINAMATH_CALUDE_joes_fast_food_cost_l2632_263231


namespace NUMINAMATH_CALUDE_choose_4_from_10_l2632_263206

theorem choose_4_from_10 : Nat.choose 10 4 = 210 := by sorry

end NUMINAMATH_CALUDE_choose_4_from_10_l2632_263206


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l2632_263236

/-- Given three two-digit numbers and a fourth unknown two-digit number,
    if the sum of the digits of all four numbers is 1/4 of their total sum,
    then the smallest possible value for the unknown number is 70. -/
theorem smallest_fourth_number (x : ℕ) :
  x ≥ 10 ∧ x < 100 →
  (34 + 21 + 63 + x : ℕ) = 4 * ((3 + 4 + 2 + 1 + 6 + 3 + (x / 10) + (x % 10)) : ℕ) →
  ∀ y : ℕ, y ≥ 10 ∧ y < 100 →
    (34 + 21 + 63 + y : ℕ) = 4 * ((3 + 4 + 2 + 1 + 6 + 3 + (y / 10) + (y % 10)) : ℕ) →
    x ≤ y →
  x = 70 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l2632_263236


namespace NUMINAMATH_CALUDE_max_product_l2632_263256

def digits : List ℕ := [1, 3, 5, 8, 9]

def is_valid_combination (a b c d e : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product (a b c d e : ℕ) : ℕ := (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product :
  ∀ a b c d e : ℕ,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 9 3 1 8 5 :=
by sorry

end NUMINAMATH_CALUDE_max_product_l2632_263256


namespace NUMINAMATH_CALUDE_shirt_cost_l2632_263244

-- Define the number of $10 bills
def num_10_bills : ℕ := 2

-- Define the number of $20 bills
def num_20_bills : ℕ := num_10_bills + 1

-- Define the value of a $10 bill
def value_10_bill : ℕ := 10

-- Define the value of a $20 bill
def value_20_bill : ℕ := 20

-- Theorem: The cost of the shirt is $80
theorem shirt_cost : 
  num_10_bills * value_10_bill + num_20_bills * value_20_bill = 80 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l2632_263244


namespace NUMINAMATH_CALUDE_secret_ballot_best_for_new_member_l2632_263273

/-- Represents a voting method -/
inductive VotingMethod
  | ShowOfHandsAgree
  | ShowOfHandsDisagree
  | SecretBallot
  | RecordedVote

/-- Represents the context of the vote -/
structure VoteContext where
  purpose : String

/-- Defines what it means for a voting method to reflect the true will of students -/
def reflectsTrueWill (method : VotingMethod) (context : VoteContext) : Prop := sorry

/-- Theorem stating that secret ballot best reflects the true will of students for adding a new class committee member -/
theorem secret_ballot_best_for_new_member :
  ∀ (context : VoteContext),
  context.purpose = "adding a new class committee member" →
  ∀ (method : VotingMethod),
  reflectsTrueWill VotingMethod.SecretBallot context →
  reflectsTrueWill method context →
  method = VotingMethod.SecretBallot :=
sorry

end NUMINAMATH_CALUDE_secret_ballot_best_for_new_member_l2632_263273


namespace NUMINAMATH_CALUDE_line_arrangements_l2632_263253

theorem line_arrangements (n : ℕ) (h : n = 6) :
  (n - 1) * Nat.factorial (n - 1) = 600 :=
by sorry

end NUMINAMATH_CALUDE_line_arrangements_l2632_263253


namespace NUMINAMATH_CALUDE_saras_quarters_l2632_263255

/-- The number of quarters Sara has after receiving some from her dad -/
def total_quarters (initial_quarters given_quarters : ℝ) : ℝ :=
  initial_quarters + given_quarters

/-- Theorem stating that Sara's total quarters is the sum of her initial quarters and those given by her dad -/
theorem saras_quarters (initial_quarters given_quarters : ℝ) :
  total_quarters initial_quarters given_quarters = initial_quarters + given_quarters :=
by sorry

end NUMINAMATH_CALUDE_saras_quarters_l2632_263255


namespace NUMINAMATH_CALUDE_martha_butterflies_l2632_263250

theorem martha_butterflies (total : ℕ) (blue : ℕ) (yellow : ℕ) (black : ℕ) : 
  total = 19 → 
  blue = 2 * yellow → 
  blue = 6 → 
  black = total - (blue + yellow) → 
  black = 10 := by
sorry

end NUMINAMATH_CALUDE_martha_butterflies_l2632_263250


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l2632_263238

theorem matrix_equation_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-21, -2; 13, 1]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![71/14, -109/14; -43/14, 67/14]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l2632_263238


namespace NUMINAMATH_CALUDE_difference_of_squares_302_298_l2632_263209

theorem difference_of_squares_302_298 : 302^2 - 298^2 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_302_298_l2632_263209


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l2632_263230

/-- The product of the coordinates of the midpoint of a line segment with endpoints (5, -3) and (-7, 11) is -4. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 5
  let y1 : ℝ := -3
  let x2 : ℝ := -7
  let y2 : ℝ := 11
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l2632_263230


namespace NUMINAMATH_CALUDE_solve_for_d_l2632_263288

theorem solve_for_d (x d : ℝ) (h1 : x = 0.3) 
  (h2 : (10 * x + 2) / 4 - (d * x - 6) / 18 = (2 * x + 4) / 3) : d = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_d_l2632_263288


namespace NUMINAMATH_CALUDE_subtract_squared_terms_l2632_263299

theorem subtract_squared_terms (x : ℝ) : 3 * x^2 - 4 * x^2 = -x^2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_squared_terms_l2632_263299


namespace NUMINAMATH_CALUDE_f_symmetry_l2632_263265

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2 + 3 * Real.sin x + 2

theorem f_symmetry (a : ℝ) (h : f a = 1) : f (-a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l2632_263265


namespace NUMINAMATH_CALUDE_hiking_trip_calculation_l2632_263202

structure HikingSegment where
  distance : Float
  speed : Float

def total_distance (segments : List HikingSegment) : Float :=
  segments.map (λ s => s.distance) |> List.sum

def total_time (segments : List HikingSegment) : Float :=
  segments.map (λ s => s.distance / s.speed) |> List.sum

def hiking_segments : List HikingSegment := [
  { distance := 0.5, speed := 3.0 },
  { distance := 1.2, speed := 2.5 },
  { distance := 0.8, speed := 2.0 },
  { distance := 0.6, speed := 2.8 }
]

theorem hiking_trip_calculation :
  total_distance hiking_segments = 3.1 ∧
  (total_time hiking_segments * 60).round = 76 := by
  sorry

#eval total_distance hiking_segments
#eval (total_time hiking_segments * 60).round

end NUMINAMATH_CALUDE_hiking_trip_calculation_l2632_263202


namespace NUMINAMATH_CALUDE_vectors_orthogonal_l2632_263210

theorem vectors_orthogonal (x : ℝ) : 
  x = 28/3 → (3 * x + 4 * (-7) = 0) := by sorry

end NUMINAMATH_CALUDE_vectors_orthogonal_l2632_263210


namespace NUMINAMATH_CALUDE_power_sum_equals_zero_l2632_263263

theorem power_sum_equals_zero : (-1 : ℤ) ^ (5^2) + (1 : ℤ) ^ (2^5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_zero_l2632_263263


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2632_263246

theorem unique_solution_quadratic (c : ℝ) (h : c ≠ 0) :
  (∃! b : ℝ, b > 0 ∧ (∃! x : ℝ, x^2 + 3 * (b + 1/b) * x + c = 0)) ↔ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2632_263246


namespace NUMINAMATH_CALUDE_triangle_DEF_angle_D_l2632_263286

theorem triangle_DEF_angle_D (D E F : ℝ) : 
  E = 3 * F → F = 15 → D + E + F = 180 → D = 120 := by sorry

end NUMINAMATH_CALUDE_triangle_DEF_angle_D_l2632_263286


namespace NUMINAMATH_CALUDE_man_upstream_speed_l2632_263234

/-- Calculates the upstream speed of a man given his downstream speed and the stream speed -/
def upstream_speed (downstream_speed stream_speed : ℝ) : ℝ :=
  downstream_speed - 2 * stream_speed

/-- Theorem stating that given a downstream speed of 13 kmph and a stream speed of 2.5 kmph, 
    the upstream speed is 8 kmph -/
theorem man_upstream_speed :
  upstream_speed 13 2.5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_man_upstream_speed_l2632_263234


namespace NUMINAMATH_CALUDE_line_through_points_circle_through_points_circle_center_on_y_axis_l2632_263280

-- Define the points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 3)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + (y-2)^2 = 2

-- Theorem for the line equation
theorem line_through_points : 
  line_eq A.1 A.2 ∧ line_eq B.1 B.2 := by sorry

-- Theorem for the circle equation
theorem circle_through_points : 
  circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 := by sorry

-- Theorem to prove the center of the circle is on the y-axis
theorem circle_center_on_y_axis : 
  ∃ y : ℝ, ∀ x : ℝ, circle_eq 0 y → circle_eq x y → x = 0 := by sorry

end NUMINAMATH_CALUDE_line_through_points_circle_through_points_circle_center_on_y_axis_l2632_263280


namespace NUMINAMATH_CALUDE_systematic_sample_smallest_element_l2632_263298

/-- Represents a systematic sample -/
structure SystematicSample where
  total : ℕ
  sampleSize : ℕ
  interval : ℕ
  containsElement : ℕ

/-- The smallest element in a systematic sample -/
def smallestElement (s : SystematicSample) : ℕ :=
  s.interval * (s.containsElement / s.interval)

theorem systematic_sample_smallest_element 
  (s : SystematicSample) 
  (h1 : s.total = 360)
  (h2 : s.sampleSize = 30)
  (h3 : s.interval = s.total / s.sampleSize)
  (h4 : s.containsElement = 105)
  (h5 : s.containsElement ≤ s.total)
  : smallestElement s = 96 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_smallest_element_l2632_263298


namespace NUMINAMATH_CALUDE_AB_length_l2632_263201

-- Define the points and lengths
variable (A B C D E F G : ℝ)

-- Define the midpoint relationships
axiom C_midpoint : C = (A + B) / 2
axiom D_midpoint : D = (A + C) / 2
axiom E_midpoint : E = (A + D) / 2
axiom F_midpoint : F = (A + E) / 2
axiom G_midpoint : G = (A + F) / 2

-- Given condition
axiom AG_length : G - A = 5

-- Theorem to prove
theorem AB_length : B - A = 160 := by
  sorry

end NUMINAMATH_CALUDE_AB_length_l2632_263201


namespace NUMINAMATH_CALUDE_exists_divisible_by_13_in_79_consecutive_l2632_263287

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: In any sequence of 79 consecutive positive integers, 
    there exists at least one integer whose sum of digits is divisible by 13 -/
theorem exists_divisible_by_13_in_79_consecutive (start : ℕ) : 
  ∃ k ∈ Finset.range 79, (sum_of_digits (start + k)) % 13 = 0 := by sorry

end NUMINAMATH_CALUDE_exists_divisible_by_13_in_79_consecutive_l2632_263287


namespace NUMINAMATH_CALUDE_necklace_packing_condition_l2632_263222

/-- Represents a necklace of cubes -/
structure CubeNecklace where
  n : ℕ
  numCubes : ℕ
  isLooped : Bool

/-- Represents a cubic box -/
structure CubicBox where
  edgeLength : ℕ

/-- Predicate to check if a necklace can be packed into a box -/
def canBePacked (necklace : CubeNecklace) (box : CubicBox) : Prop :=
  necklace.numCubes = box.edgeLength ^ 3 ∧
  necklace.isLooped = true

/-- Theorem stating the condition for packing the necklace -/
theorem necklace_packing_condition (n : ℕ) :
  let necklace := CubeNecklace.mk n (n^3) true
  let box := CubicBox.mk n
  canBePacked necklace box ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_necklace_packing_condition_l2632_263222


namespace NUMINAMATH_CALUDE_range_of_a_l2632_263277

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → x^3 - a*x + 1 ≥ 0) → 
  0 ≤ a ∧ a ≤ 3 * (2 : ℝ)^(1/3) / 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2632_263277


namespace NUMINAMATH_CALUDE_multiplication_proof_l2632_263283

theorem multiplication_proof : 287 * 23 = 6601 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_proof_l2632_263283


namespace NUMINAMATH_CALUDE_linear_function_triangle_area_l2632_263207

theorem linear_function_triangle_area (k : ℝ) : 
  (1/2 * 3 * |3/k| = 24) → (k = 3/16 ∨ k = -3/16) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_triangle_area_l2632_263207


namespace NUMINAMATH_CALUDE_count_six_digit_numbers_with_at_least_two_zeros_l2632_263245

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with no zeros -/
def numbers_with_no_zeros : ℕ := 531441

/-- The number of 6-digit numbers with exactly one zero -/
def numbers_with_one_zero : ℕ := 295245

/-- The number of 6-digit numbers with at least two zeros -/
def numbers_with_at_least_two_zeros : ℕ := 
  total_six_digit_numbers - (numbers_with_no_zeros + numbers_with_one_zero)

theorem count_six_digit_numbers_with_at_least_two_zeros : 
  numbers_with_at_least_two_zeros = 73314 := by
  sorry

end NUMINAMATH_CALUDE_count_six_digit_numbers_with_at_least_two_zeros_l2632_263245


namespace NUMINAMATH_CALUDE_painted_subcubes_count_l2632_263205

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  all_faces_painted : Bool

/-- Calculates the number of 1x1x1 subcubes with at least two painted faces in a painted cube -/
def subcubes_with_two_or_more_painted_faces (c : Cube 4) : ℕ :=
  if c.all_faces_painted then
    -- Corner cubes (3 faces painted)
    8 +
    -- Edge cubes without corners (2 faces painted)
    (12 * 2) +
    -- Middle-edge face cubes (2 faces painted)
    (6 * 4)
  else
    0

/-- Theorem: In a 4x4x4 cube with all faces painted, there are 56 subcubes with at least two painted faces -/
theorem painted_subcubes_count (c : Cube 4) (h : c.all_faces_painted = true) :
  subcubes_with_two_or_more_painted_faces c = 56 := by
  sorry

#check painted_subcubes_count

end NUMINAMATH_CALUDE_painted_subcubes_count_l2632_263205


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2632_263260

theorem geometric_sequence_product (a r : ℝ) (n : ℕ) (h_even : Even n) :
  let S := a * (1 - r^n) / (1 - r)
  let S' := (1 / (2*a)) * (r^n - 1) / (r - 1) * r^(1-n)
  let P := (2*a)^n * r^(n*(n-1)/2)
  P = (S * S')^(n/2) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2632_263260


namespace NUMINAMATH_CALUDE_malvina_card_sum_l2632_263272

open Real MeasureTheory

theorem malvina_card_sum : ∀ x : ℝ,
  90 * π / 180 < x ∧ x < π →
  (∀ y : ℝ, 90 * π / 180 < y ∧ y < π →
    sin y > 0 ∧ cos y < 0 ∧ tan y < 0) →
  (∫ y in Set.Icc (90 * π / 180) π, sin y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_malvina_card_sum_l2632_263272


namespace NUMINAMATH_CALUDE_watch_synchronization_l2632_263211

/-- The number of seconds in a full rotation of a standard watch -/
def full_rotation : ℕ := 12 * 60 * 60

/-- The number of seconds Glafira's watch gains per day -/
def glafira_gain : ℕ := 12

/-- The number of seconds Gavrila's watch loses per day -/
def gavrila_loss : ℕ := 18

/-- The combined deviation of both watches per day -/
def combined_deviation : ℕ := glafira_gain + gavrila_loss

theorem watch_synchronization :
  (full_rotation / combined_deviation : ℚ) = 1440 := by sorry

end NUMINAMATH_CALUDE_watch_synchronization_l2632_263211


namespace NUMINAMATH_CALUDE_equal_probability_for_all_probability_independent_of_method_l2632_263285

/-- The probability of selecting a product given the described selection method -/
def selection_probability (total : ℕ) (remove : ℕ) (select : ℕ) : ℚ :=
  select / total

/-- The selection method ensures equal probability for all products -/
theorem equal_probability_for_all (total : ℕ) (remove : ℕ) (select : ℕ) 
  (h1 : total = 2003)
  (h2 : remove = 3)
  (h3 : select = 50) :
  selection_probability total remove select = 50 / 2003 := by
  sorry

/-- The probability is independent of the specific selection method -/
theorem probability_independent_of_method 
  (simple_random_sampling : (ℕ → ℕ → ℕ → ℚ))
  (systematic_sampling : (ℕ → ℕ → ℕ → ℚ))
  (total : ℕ) (remove : ℕ) (select : ℕ)
  (h1 : total = 2003)
  (h2 : remove = 3)
  (h3 : select = 50) :
  simple_random_sampling total remove select = systematic_sampling (total - remove) select select ∧
  simple_random_sampling total remove select = selection_probability total remove select := by
  sorry

end NUMINAMATH_CALUDE_equal_probability_for_all_probability_independent_of_method_l2632_263285


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2632_263289

theorem exponent_multiplication (x : ℝ) : x^3 * (2*x^4) = 2*x^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2632_263289


namespace NUMINAMATH_CALUDE_quadratic_touches_x_axis_at_one_point_l2632_263235

/-- A quadratic function g(x) = x^2 - 6x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + k

/-- The discriminant of the quadratic function g -/
def discriminant (k : ℝ) : ℝ := (-6)^2 - 4*1*k

/-- Theorem: The value of k that makes g(x) touch the x-axis at exactly one point is 9 -/
theorem quadratic_touches_x_axis_at_one_point :
  ∃ (k : ℝ), (discriminant k = 0) ∧ (k = 9) := by sorry

end NUMINAMATH_CALUDE_quadratic_touches_x_axis_at_one_point_l2632_263235


namespace NUMINAMATH_CALUDE_hayley_sticker_distribution_l2632_263249

def distribute_stickers (total_stickers : ℕ) (num_friends : ℕ) : ℕ :=
  total_stickers / num_friends

theorem hayley_sticker_distribution :
  let total_stickers : ℕ := 72
  let num_friends : ℕ := 9
  distribute_stickers total_stickers num_friends = 8 := by sorry

end NUMINAMATH_CALUDE_hayley_sticker_distribution_l2632_263249


namespace NUMINAMATH_CALUDE_eating_contest_l2632_263296

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ)
  (noah_burgers jacob_pies mason_hotdogs : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  noah_burgers = 8 →
  mason_hotdogs * hot_dog_weight = 30 := by
sorry

end NUMINAMATH_CALUDE_eating_contest_l2632_263296


namespace NUMINAMATH_CALUDE_weight_problem_l2632_263290

theorem weight_problem (c d e f : ℝ) 
  (h1 : c + d = 330)
  (h2 : d + e = 290)
  (h3 : e + f = 310) :
  c + f = 350 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l2632_263290


namespace NUMINAMATH_CALUDE_root_sum_powers_l2632_263292

theorem root_sum_powers (α β : ℝ) : 
  α^2 - 5*α + 6 = 0 → β^2 - 5*β + 6 = 0 → 3*α^3 + 10*β^4 = 2305 := by
sorry

end NUMINAMATH_CALUDE_root_sum_powers_l2632_263292


namespace NUMINAMATH_CALUDE_angle_triple_supplement_l2632_263271

theorem angle_triple_supplement (x : ℝ) : 
  (x = 3 * (180 - x)) → x = 135 := by sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_l2632_263271


namespace NUMINAMATH_CALUDE_f1_times_g0_l2632_263219

-- Define f as an odd function on ℝ
def f : ℝ → ℝ := sorry

-- Define g as an even function on ℝ
def g : ℝ → ℝ := sorry

-- Define the relationship between f and g
axiom fg_relation : ∀ x : ℝ, f x - g x = 2^x

-- Define the property of odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define the property of even function
axiom g_even : ∀ x : ℝ, g (-x) = g x

-- Theorem to prove
theorem f1_times_g0 : f 1 * g 0 = -3/4 := by sorry

end NUMINAMATH_CALUDE_f1_times_g0_l2632_263219


namespace NUMINAMATH_CALUDE_geometric_sequence_a_value_l2632_263267

theorem geometric_sequence_a_value (a : ℝ) :
  (1 / (a - 1)) * (a + 1) = (a + 1) * (a^2 - 1) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a_value_l2632_263267


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2632_263261

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^3 > x) ↔ (∀ x : ℝ, x^3 ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2632_263261


namespace NUMINAMATH_CALUDE_statement_a_is_false_l2632_263204

/-- Represents an element in the periodic table -/
structure Element where
  atomic_number : ℕ
  isotopes : List (ℕ × ℝ)  -- List of (mass number, abundance) pairs

/-- Calculates the relative atomic mass of an element -/
def relative_atomic_mass (e : Element) : ℝ :=
  (e.isotopes.map (λ (mass, abundance) => mass * abundance)).sum

/-- Represents a single atom of an element -/
structure Atom where
  protons : ℕ
  neutrons : ℕ

/-- The statement we want to prove false -/
def statement_a (e : Element) (a : Atom) : Prop :=
  relative_atomic_mass e = a.protons + a.neutrons

/-- Theorem stating that the statement is false -/
theorem statement_a_is_false :
  ∃ (e : Element) (a : Atom), ¬(statement_a e a) :=
sorry

end NUMINAMATH_CALUDE_statement_a_is_false_l2632_263204


namespace NUMINAMATH_CALUDE_max_product_of_roots_l2632_263215

/-- Given a quadratic equation 5x^2 - 10x + m = 0 with real roots,
    the maximum value of the product of its roots is 1. -/
theorem max_product_of_roots :
  ∀ m : ℝ,
  (∃ x : ℝ, 5 * x^2 - 10 * x + m = 0) →
  (∀ k : ℝ, (∃ x : ℝ, 5 * x^2 - 10 * x + k = 0) → m / 5 ≥ k / 5) →
  m / 5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_roots_l2632_263215


namespace NUMINAMATH_CALUDE_floor_plus_self_equals_twenty_l2632_263237

theorem floor_plus_self_equals_twenty (s : ℝ) : ⌊s⌋ + s = 20 ↔ s = 10 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_equals_twenty_l2632_263237


namespace NUMINAMATH_CALUDE_f_value_at_2_l2632_263275

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem f_value_at_2 : 
  (∀ x : ℝ, f (2 * x + 1) = x^2 - 2*x) → f 2 = -3/4 := by sorry

end NUMINAMATH_CALUDE_f_value_at_2_l2632_263275


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2632_263247

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2632_263247


namespace NUMINAMATH_CALUDE_sqrt_1001_irreducible_l2632_263221

theorem sqrt_1001_irreducible : ∀ a b : ℕ, a * a = 1001 * (b * b) → a = 1001 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_1001_irreducible_l2632_263221


namespace NUMINAMATH_CALUDE_math_only_students_l2632_263226

theorem math_only_students (total : ℕ) (math foreign_lang science : Finset ℕ) :
  total = 120 →
  (∀ s, s ∈ math ∪ foreign_lang ∪ science) →
  math.card = 85 →
  foreign_lang.card = 65 →
  science.card = 50 →
  (math ∩ foreign_lang ∩ science).card = 20 →
  (math \ (foreign_lang ∪ science)).card = 52 :=
by sorry

end NUMINAMATH_CALUDE_math_only_students_l2632_263226


namespace NUMINAMATH_CALUDE_expression_evaluation_l2632_263200

theorem expression_evaluation : -1^2008 + (-1)^2009 + 1^2010 + (-1)^2011 + 1^2012 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2632_263200


namespace NUMINAMATH_CALUDE_course_class_duration_l2632_263257

/-- Proves the duration of each class in a course given the total course duration and other parameters. -/
theorem course_class_duration 
  (weeks : ℕ) 
  (unknown_classes_per_week : ℕ) 
  (known_class_duration : ℕ) 
  (homework_duration : ℕ) 
  (total_course_time : ℕ) 
  (h1 : weeks = 24)
  (h2 : unknown_classes_per_week = 2)
  (h3 : known_class_duration = 4)
  (h4 : homework_duration = 4)
  (h5 : total_course_time = 336) :
  ∃ x : ℕ, x * unknown_classes_per_week * weeks + known_class_duration * weeks + homework_duration * weeks = total_course_time ∧ x = 3 := by
  sorry

#check course_class_duration

end NUMINAMATH_CALUDE_course_class_duration_l2632_263257


namespace NUMINAMATH_CALUDE_perpendicular_nonzero_vectors_exist_l2632_263216

theorem perpendicular_nonzero_vectors_exist :
  ∃ (a b : ℝ × ℝ), a ≠ (0, 0) ∧ b ≠ (0, 0) ∧ a.1 * b.1 + a.2 * b.2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_nonzero_vectors_exist_l2632_263216
