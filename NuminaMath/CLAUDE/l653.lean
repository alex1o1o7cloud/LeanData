import Mathlib

namespace NUMINAMATH_CALUDE_vector_angle_constraint_l653_65338

def a (k : ℝ) : Fin 2 → ℝ := ![-k, 4]
def b (k : ℝ) : Fin 2 → ℝ := ![k, k+3]

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

def is_acute_angle (v w : Fin 2 → ℝ) : Prop :=
  dot_product v w > 0 ∧ v ≠ w

theorem vector_angle_constraint (k : ℝ) :
  is_acute_angle (a k) (b k) → k ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 6 :=
sorry

end NUMINAMATH_CALUDE_vector_angle_constraint_l653_65338


namespace NUMINAMATH_CALUDE_odd_integer_divisibility_l653_65340

theorem odd_integer_divisibility (n : ℤ) (h : Odd n) :
  ∃ x : ℤ, (n^2 : ℤ) ∣ (x^2 - n*x - 1) := by sorry

end NUMINAMATH_CALUDE_odd_integer_divisibility_l653_65340


namespace NUMINAMATH_CALUDE_sin_360_degrees_equals_zero_l653_65397

theorem sin_360_degrees_equals_zero : Real.sin (2 * Real.pi) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_360_degrees_equals_zero_l653_65397


namespace NUMINAMATH_CALUDE_two_equal_intercept_lines_l653_65323

/-- A line passing through (5,2) with equal x and y intercepts -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (5,2) -/
  passes_through : 2 = m * 5 + b
  /-- The line has equal x and y intercepts -/
  equal_intercepts : b = m * b

/-- There are exactly two distinct lines passing through (5,2) with equal x and y intercepts -/
theorem two_equal_intercept_lines : 
  ∃ (l₁ l₂ : EqualInterceptLine), l₁ ≠ l₂ ∧ 
  ∀ (l : EqualInterceptLine), l = l₁ ∨ l = l₂ :=
sorry

end NUMINAMATH_CALUDE_two_equal_intercept_lines_l653_65323


namespace NUMINAMATH_CALUDE_equal_expressions_l653_65391

theorem equal_expressions (x : ℝ) : 2 * x - 1 = 3 * x + 3 ↔ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_l653_65391


namespace NUMINAMATH_CALUDE_meters_in_one_kilometer_l653_65392

/-- Conversion factor from kilometers to hectometers -/
def km_to_hm : ℝ := 5

/-- Conversion factor from hectometers to dekameters -/
def hm_to_dam : ℝ := 10

/-- Conversion factor from dekameters to meters -/
def dam_to_m : ℝ := 15

/-- The number of meters in one kilometer -/
def meters_in_km : ℝ := km_to_hm * hm_to_dam * dam_to_m

theorem meters_in_one_kilometer :
  meters_in_km = 750 := by sorry

end NUMINAMATH_CALUDE_meters_in_one_kilometer_l653_65392


namespace NUMINAMATH_CALUDE_chess_tournament_attendees_l653_65367

theorem chess_tournament_attendees (total_students : ℕ) 
  (h1 : total_students = 24) 
  (chess_program_fraction : ℚ) 
  (h2 : chess_program_fraction = 1 / 3) 
  (tournament_fraction : ℚ) 
  (h3 : tournament_fraction = 1 / 2) : ℕ :=
  by
    sorry

#check chess_tournament_attendees

end NUMINAMATH_CALUDE_chess_tournament_attendees_l653_65367


namespace NUMINAMATH_CALUDE_min_students_for_question_distribution_l653_65320

theorem min_students_for_question_distribution (total_questions : Nat) 
  (folder_size : Nat) (num_folders : Nat) (max_unsolved : Nat) :
  total_questions = 2010 →
  folder_size = 670 →
  num_folders = 3 →
  max_unsolved = 2 →
  ∃ (min_students : Nat), 
    (∀ (n : Nat), n < min_students → 
      ¬(∀ (folder : Finset Nat), folder.card = folder_size → 
        ∃ (solved_by : Finset Nat), solved_by.card ≥ num_folders ∧ 
          ∀ (q : Nat), q ∈ folder → (n - solved_by.card) ≤ max_unsolved)) ∧
    (∀ (folder : Finset Nat), folder.card = folder_size → 
      ∃ (solved_by : Finset Nat), solved_by.card ≥ num_folders ∧ 
        ∀ (q : Nat), q ∈ folder → (min_students - solved_by.card) ≤ max_unsolved) ∧
    min_students = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_students_for_question_distribution_l653_65320


namespace NUMINAMATH_CALUDE_total_chairs_count_l653_65360

theorem total_chairs_count : ℕ := by
  -- Define the number of rows and chairs per row for each section
  let first_section_rows : ℕ := 5
  let first_section_chairs_per_row : ℕ := 10
  let second_section_rows : ℕ := 8
  let second_section_chairs_per_row : ℕ := 12

  -- Define the number of late arrivals and extra chairs per late arrival
  let late_arrivals : ℕ := 20
  let extra_chairs_per_late_arrival : ℕ := 3

  -- Calculate the total number of chairs
  let total_chairs := 
    (first_section_rows * first_section_chairs_per_row) +
    (second_section_rows * second_section_chairs_per_row) +
    (late_arrivals * extra_chairs_per_late_arrival)

  -- Prove that the total number of chairs is 206
  have h : total_chairs = 206 := by sorry

  exact 206


end NUMINAMATH_CALUDE_total_chairs_count_l653_65360


namespace NUMINAMATH_CALUDE_second_investment_rate_l653_65365

theorem second_investment_rate
  (total_investment : ℝ)
  (first_rate : ℝ)
  (first_amount : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 6000)
  (h2 : first_rate = 0.09)
  (h3 : first_amount = 1800)
  (h4 : total_interest = 624)
  : (total_interest - first_amount * first_rate) / (total_investment - first_amount) = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_rate_l653_65365


namespace NUMINAMATH_CALUDE_unique_solution_l653_65308

def system_solution (x y : ℝ) : Prop :=
  x + y = 1 ∧ x - y = -1

theorem unique_solution : 
  ∃! p : ℝ × ℝ, system_solution p.1 p.2 ∧ p = (0, 1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l653_65308


namespace NUMINAMATH_CALUDE_next_base3_number_l653_65302

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The base 3 representation of M -/
def M : List Nat := [0, 2, 0, 1]

theorem next_base3_number (h : base3ToDecimal M = base3ToDecimal [0, 2, 0, 1]) :
  decimalToBase3 (base3ToDecimal M + 1) = [1, 2, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_next_base3_number_l653_65302


namespace NUMINAMATH_CALUDE_number_of_boys_l653_65324

theorem number_of_boys (total_children happy_children sad_children neutral_children girls happy_boys sad_girls : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  girls = 42 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = happy_children + sad_children + neutral_children →
  ∃ boys, boys = total_children - girls ∧ boys = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l653_65324


namespace NUMINAMATH_CALUDE_power_function_through_point_l653_65316

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 4 = 2 → f 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l653_65316


namespace NUMINAMATH_CALUDE_ellipse_equation_l653_65359

/-- Given an ellipse with the following properties:
  1. The axes of symmetry lie on the coordinate axes
  2. One endpoint of the minor axis and the two foci form an equilateral triangle
  3. The distance from the foci to the same vertex is √3
  Then the standard equation of the ellipse is x²/12 + y²/9 = 1 or y²/12 + x²/9 = 1 -/
theorem ellipse_equation (a c : ℝ) (h1 : a = 2 * c) (h2 : a - c = Real.sqrt 3) :
  ∃ (x y : ℝ), (x^2 / 12 + y^2 / 9 = 1) ∨ (y^2 / 12 + x^2 / 9 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l653_65359


namespace NUMINAMATH_CALUDE_no_equidistant_points_l653_65366

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {P | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the parallel tangents
def ParallelTangents (O : ℝ × ℝ) (d : ℝ) : Set (ℝ × ℝ) :=
  {P | |P.2 - O.2| = d}

-- Define a point equidistant from circle and tangents
def IsEquidistant (P : ℝ × ℝ) (O : ℝ × ℝ) (r d : ℝ) : Prop :=
  abs (((P.1 - O.1)^2 + (P.2 - O.2)^2).sqrt - r) = abs (|P.2 - O.2| - d)

theorem no_equidistant_points (O : ℝ × ℝ) (r d : ℝ) (h : d > r) :
  ¬∃P, IsEquidistant P O r d :=
by sorry

end NUMINAMATH_CALUDE_no_equidistant_points_l653_65366


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l653_65396

theorem factorial_difference_quotient : (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l653_65396


namespace NUMINAMATH_CALUDE_geometric_sequence_10th_term_l653_65364

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_10th_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_4th : a 4 = 16)
  (h_7th : a 7 = 128) :
  a 10 = 1024 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_10th_term_l653_65364


namespace NUMINAMATH_CALUDE_prob_not_red_or_purple_is_correct_l653_65349

-- Define the total number of balls and the number of each color
def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def yellow_balls : ℕ := 7
def red_balls : ℕ := 15
def purple_balls : ℕ := 6

-- Define the probability of choosing a ball that is neither red nor purple
def prob_not_red_or_purple : ℚ := (white_balls + green_balls + yellow_balls) / total_balls

-- Theorem statement
theorem prob_not_red_or_purple_is_correct :
  prob_not_red_or_purple = 13/20 := by sorry

end NUMINAMATH_CALUDE_prob_not_red_or_purple_is_correct_l653_65349


namespace NUMINAMATH_CALUDE_power_fraction_equality_l653_65394

theorem power_fraction_equality : (2^2020 + 2^2016) / (2^2020 - 2^2016) = 17/15 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l653_65394


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_105_l653_65386

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of n consecutive positive integers starting from a -/
def sum_consecutive (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

theorem largest_consecutive_sum_105 :
  (∃ (a : ℕ), a > 0 ∧ sum_consecutive a 14 = 105) ∧
  (∀ (n : ℕ), n > 14 → ¬∃ (a : ℕ), a > 0 ∧ sum_consecutive a n = 105) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_105_l653_65386


namespace NUMINAMATH_CALUDE_store_revenue_calculation_l653_65315

/-- Represents the revenue calculation for Linda's store --/
def store_revenue (jean_price tee_price_low tee_price_high jacket_price jacket_discount tee_count_low tee_count_high jean_count jacket_count_regular jacket_count_discount sales_tax : ℚ) : ℚ :=
  let tee_revenue := tee_price_low * tee_count_low + tee_price_high * tee_count_high
  let jean_revenue := jean_price * jean_count
  let jacket_revenue_regular := jacket_price * jacket_count_regular
  let jacket_revenue_discount := jacket_price * (1 - jacket_discount) * jacket_count_discount
  let total_revenue := tee_revenue + jean_revenue + jacket_revenue_regular + jacket_revenue_discount
  let total_with_tax := total_revenue * (1 + sales_tax)
  total_with_tax

/-- Theorem stating that the store revenue matches the calculated amount --/
theorem store_revenue_calculation :
  store_revenue 22 15 20 37 0.1 4 3 4 2 3 0.07 = 408.63 :=
by sorry

end NUMINAMATH_CALUDE_store_revenue_calculation_l653_65315


namespace NUMINAMATH_CALUDE_set_B_equality_l653_65326

def A : Set ℤ := {-1, 0, 1, 2}

def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2*x}

theorem set_B_equality : B = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_set_B_equality_l653_65326


namespace NUMINAMATH_CALUDE_simplify_expression_l653_65325

theorem simplify_expression (a b : ℝ) : (2*a - b) - 2*(a - 2*b) = 3*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l653_65325


namespace NUMINAMATH_CALUDE_price_change_l653_65350

theorem price_change (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.3
  let final_price := increased_price * 0.75
  final_price = original_price * 0.975 :=
by sorry

end NUMINAMATH_CALUDE_price_change_l653_65350


namespace NUMINAMATH_CALUDE_julia_drove_214_miles_l653_65345

/-- Calculates the number of miles driven given the total cost, daily rental rate, and per-mile rate -/
def miles_driven (total_cost daily_rate mile_rate : ℚ) : ℚ :=
  (total_cost - daily_rate) / mile_rate

/-- Proves that Julia drove 214 miles given the rental conditions -/
theorem julia_drove_214_miles :
  let total_cost : ℚ := 46.12
  let daily_rate : ℚ := 29
  let mile_rate : ℚ := 0.08
  miles_driven total_cost daily_rate mile_rate = 214 := by
    sorry

#eval miles_driven 46.12 29 0.08

end NUMINAMATH_CALUDE_julia_drove_214_miles_l653_65345


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l653_65313

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b ^ 16) ^ (1 / 8)) ^ (1 / 4)) ^ 3 * (((b ^ 16) ^ (1 / 4)) ^ (1 / 8)) ^ 3 = b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l653_65313


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l653_65378

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l653_65378


namespace NUMINAMATH_CALUDE_die_roll_counts_l653_65381

/-- Represents the number of sides on a standard die -/
def dieSides : ℕ := 6

/-- Calculates the number of three-digit numbers with all distinct digits -/
def distinctDigits : ℕ := dieSides * (dieSides - 1) * (dieSides - 2)

/-- Calculates the total number of different three-digit numbers -/
def totalNumbers : ℕ := dieSides ^ 3

/-- Calculates the number of three-digit numbers with exactly two digits the same -/
def twoSameDigits : ℕ := 3 * dieSides * (dieSides - 1)

theorem die_roll_counts :
  distinctDigits = 120 ∧ totalNumbers = 216 ∧ twoSameDigits = 90 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_counts_l653_65381


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l653_65303

/-- Two circles are externally tangent when the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂

/-- The problem statement -/
theorem circles_externally_tangent :
  let r₁ : ℝ := 1
  let r₂ : ℝ := 3
  let d : ℝ := 4
  externally_tangent r₁ r₂ d :=
by
  sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l653_65303


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l653_65341

/-- Three points in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point2D) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

theorem collinear_points_x_value :
  let p : Point2D := ⟨1, 1⟩
  let a : Point2D := ⟨2, -4⟩
  let b : Point2D := ⟨x, -9⟩
  collinear p a b → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l653_65341


namespace NUMINAMATH_CALUDE_video_votes_l653_65300

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 120 ∧ 
  like_percentage = 3/4 ∧ 
  (∀ (total_votes : ℕ), 
    (↑total_votes : ℚ) * like_percentage - (↑total_votes : ℚ) * (1 - like_percentage) = score) →
  ∃ (total_votes : ℕ), total_votes = 240 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l653_65300


namespace NUMINAMATH_CALUDE_world_expo_visitors_l653_65354

def cost_per_person (n : ℕ) : ℕ :=
  if n ≤ 30 then 120
  else max 90 (120 - 2 * (n - 30))

def total_cost (n : ℕ) : ℕ :=
  n * cost_per_person n

theorem world_expo_visitors :
  ∃ n : ℕ, n > 30 ∧ total_cost n = 4000 ∧
  ∀ m : ℕ, m ≠ n → total_cost m ≠ 4000 :=
sorry

end NUMINAMATH_CALUDE_world_expo_visitors_l653_65354


namespace NUMINAMATH_CALUDE_gideon_age_proof_l653_65310

/-- The number of years in a century -/
def years_in_century : ℕ := 100

/-- Gideon's current age -/
def gideon_age : ℕ := 45

/-- The number of marbles Gideon has -/
def gideon_marbles : ℕ := years_in_century

/-- Gideon's age five years from now -/
def gideon_future_age : ℕ := gideon_age + 5

theorem gideon_age_proof :
  gideon_age = 45 ∧
  gideon_marbles = years_in_century ∧
  gideon_future_age = 2 * (gideon_marbles / 4) :=
by sorry

end NUMINAMATH_CALUDE_gideon_age_proof_l653_65310


namespace NUMINAMATH_CALUDE_integer_triple_solution_l653_65371

theorem integer_triple_solution (x y z : ℤ) :
  x * (y + z) = y^2 + z^2 - 2 ∧
  y * (z + x) = z^2 + x^2 - 2 ∧
  z * (x + y) = x^2 + y^2 - 2 →
  (x = 1 ∧ y = 0 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = 0) ∨
  (x = 0 ∧ y = 1 ∧ z = -1) ∨
  (x = 0 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 0) ∨
  (x = -1 ∧ y = 0 ∧ z = 1) :=
by sorry


end NUMINAMATH_CALUDE_integer_triple_solution_l653_65371


namespace NUMINAMATH_CALUDE_min_fence_posts_for_grazing_area_l653_65322

/-- Calculates the number of fence posts needed for a rectangular grazing area -/
def fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side_posts := length / post_spacing + 1
  let short_side_posts := width / post_spacing
  long_side_posts + 2 * short_side_posts

/-- Theorem stating the minimum number of fence posts required -/
theorem min_fence_posts_for_grazing_area :
  fence_posts 100 40 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_min_fence_posts_for_grazing_area_l653_65322


namespace NUMINAMATH_CALUDE_intersection_point_unique_l653_65319

/-- Two lines in a 2D plane --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in a 2D plane --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def lies_on (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The intersection point of two lines --/
def intersection_point (l1 l2 : Line2D) : Point2D :=
  { x := 1, y := 0 }

theorem intersection_point_unique (l1 l2 : Line2D) :
  l1 = Line2D.mk 1 (-4) (-1) →
  l2 = Line2D.mk 2 1 (-2) →
  let p := intersection_point l1 l2
  lies_on p l1 ∧ lies_on p l2 ∧
  ∀ q : Point2D, lies_on q l1 → lies_on q l2 → q = p :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l653_65319


namespace NUMINAMATH_CALUDE_equation_solution_l653_65375

theorem equation_solution : 
  ∃ y : ℝ, (1/8: ℝ)^(3*y+12) = (64 : ℝ)^(y+4) ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l653_65375


namespace NUMINAMATH_CALUDE_unique_solution_system_l653_65304

theorem unique_solution_system (x y z : ℝ) : 
  (Real.sqrt (x - 997) + Real.sqrt (y - 932) + Real.sqrt (z - 796) = 100) ∧
  (Real.sqrt (x - 1237) + Real.sqrt (y - 1121) + Real.sqrt (3045 - z) = 90) ∧
  (Real.sqrt (x - 1621) + Real.sqrt (2805 - y) + Real.sqrt (z - 997) = 80) ∧
  (Real.sqrt (2102 - x) + Real.sqrt (y - 1237) + Real.sqrt (z - 932) = 70) →
  x = 2021 ∧ y = 2021 ∧ z = 2021 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l653_65304


namespace NUMINAMATH_CALUDE_sum_of_squares_l653_65388

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 30) : x^2 + y^2 = 840 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l653_65388


namespace NUMINAMATH_CALUDE_six_students_five_lectures_l653_65348

/-- The number of ways students can choose lectures -/
def lecture_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 6 students choosing from 5 lectures results in 5^6 possibilities -/
theorem six_students_five_lectures :
  lecture_choices 6 5 = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_six_students_five_lectures_l653_65348


namespace NUMINAMATH_CALUDE_water_speed_calculation_l653_65327

/-- Proves that the speed of the water is 8 km/h given the swimming conditions -/
theorem water_speed_calculation (swimming_speed : ℝ) (distance : ℝ) (time : ℝ) :
  swimming_speed = 16 →
  distance = 12 →
  time = 1.5 →
  swimming_speed - (distance / time) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l653_65327


namespace NUMINAMATH_CALUDE_saltwater_animals_count_l653_65374

-- Define the number of saltwater aquariums
def saltwater_aquariums : ℕ := 22

-- Define the number of animals per aquarium
def animals_per_aquarium : ℕ := 46

-- Theorem to prove the number of saltwater animals
theorem saltwater_animals_count :
  saltwater_aquariums * animals_per_aquarium = 1012 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_animals_count_l653_65374


namespace NUMINAMATH_CALUDE_line_equation_from_slope_and_point_l653_65311

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a line with slope 3 passing through (1, -2), its equation is 3x - y - 5 = 0 -/
theorem line_equation_from_slope_and_point :
  ∀ (l : Line),
  l.slope = 3 ∧ l.point = (1, -2) →
  ∃ (eq : LineEquation),
  eq.a = 3 ∧ eq.b = -1 ∧ eq.c = -5 ∧
  ∀ (x y : ℝ), eq.a * x + eq.b * y + eq.c = 0 ↔ y = l.slope * (x - l.point.1) + l.point.2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_slope_and_point_l653_65311


namespace NUMINAMATH_CALUDE_ramanujan_identity_a_l653_65352

theorem ramanujan_identity_a : 
  (((2 : ℝ) ^ (1/3) - 1) ^ (1/3) = (1/9 : ℝ) ^ (1/3) - (2/9 : ℝ) ^ (1/3) + (4/9 : ℝ) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_identity_a_l653_65352


namespace NUMINAMATH_CALUDE_no_solution_to_system_l653_65334

theorem no_solution_to_system :
  ∀ x : ℝ, ¬(x^5 + 3*x^4 + 5*x^3 + 5*x^2 + 6*x + 2 = 0 ∧ x^3 + 3*x^2 + 4*x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l653_65334


namespace NUMINAMATH_CALUDE_cone_height_l653_65389

/-- For a cone with slant height 2 and lateral area 4 times the area of its base, 
    the height of the cone is π/2. -/
theorem cone_height (r : ℝ) (h : ℝ) : 
  r > 0 → h > 0 → 
  r^2 + h^2 = 4 → -- slant height is 2
  2 * π * r = 4 * π * r^2 → -- lateral area is 4 times base area
  h = π / 2 := by
sorry

end NUMINAMATH_CALUDE_cone_height_l653_65389


namespace NUMINAMATH_CALUDE_triangle_abc_area_l653_65385

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  Real.sqrt (1/4 * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))

theorem triangle_abc_area :
  ∀ (A B C : ℝ) (a b c : ℝ),
  (Real.sin A - Real.sin B) * (Real.sin A + Real.sin B) = Real.sin A * Real.sin C - Real.sin C^2 →
  c = 2*a ∧ c = 2 * Real.sqrt 2 →
  triangle_area a b c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l653_65385


namespace NUMINAMATH_CALUDE_bubble_sort_probability_bubble_sort_probability_proof_l653_65399

/-- The probability that the 10th element in a random sequence of 50 distinct elements 
    will end up in the 25th position after one bubble pass -/
theorem bubble_sort_probability (n : ℕ) (h : n = 50) : ℝ :=
  24 / 25

/-- Proof of the bubble_sort_probability theorem -/
theorem bubble_sort_probability_proof (n : ℕ) (h : n = 50) : 
  bubble_sort_probability n h = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_probability_bubble_sort_probability_proof_l653_65399


namespace NUMINAMATH_CALUDE_school_supplies_cost_l653_65346

/-- Calculate the total amount spent on school supplies --/
theorem school_supplies_cost 
  (original_backpack_price : ℕ) 
  (original_binder_price : ℕ) 
  (backpack_price_increase : ℕ) 
  (binder_price_decrease : ℕ) 
  (num_binders : ℕ) 
  (h1 : original_backpack_price = 50)
  (h2 : original_binder_price = 20)
  (h3 : backpack_price_increase = 5)
  (h4 : binder_price_decrease = 2)
  (h5 : num_binders = 3) :
  (original_backpack_price + backpack_price_increase) + 
  num_binders * (original_binder_price - binder_price_decrease) = 109 :=
by sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l653_65346


namespace NUMINAMATH_CALUDE_domain_of_h_l653_65344

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-12) 6

-- Define the function h in terms of f
def h (x : ℝ) : ℝ := f (-3 * x)

-- State the theorem about the domain of h
theorem domain_of_h :
  {x : ℝ | h x ∈ Set.range f} = Set.Icc (-2) 4 := by sorry

end NUMINAMATH_CALUDE_domain_of_h_l653_65344


namespace NUMINAMATH_CALUDE_base_number_proof_l653_65347

theorem base_number_proof (y : ℝ) (base : ℝ) 
  (h1 : 9^y = base^16) (h2 : y = 8) : base = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l653_65347


namespace NUMINAMATH_CALUDE_library_shelf_capacity_l653_65357

/-- Given a library with a total number of books and shelves, 
    calculate the number of books per shelf. -/
def books_per_shelf (total_books : ℕ) (total_shelves : ℕ) : ℕ :=
  total_books / total_shelves

/-- Theorem stating that in a library with 14240 books and 1780 shelves,
    each shelf holds 8 books. -/
theorem library_shelf_capacity : books_per_shelf 14240 1780 = 8 := by
  sorry

end NUMINAMATH_CALUDE_library_shelf_capacity_l653_65357


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l653_65351

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 4) (h2 : x^2 + y^2 = 8) : x^3 + y^3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l653_65351


namespace NUMINAMATH_CALUDE_total_spent_usd_value_l653_65382

/-- The total amount spent on souvenirs in US dollars -/
def total_spent_usd (key_chain_bracelet_cost : ℝ) (tshirt_cost_diff : ℝ) 
  (tshirt_discount : ℝ) (key_chain_tax : ℝ) (bracelet_tax : ℝ) 
  (conversion_rate : ℝ) : ℝ :=
  let tshirt_cost := key_chain_bracelet_cost - tshirt_cost_diff
  let tshirt_actual := tshirt_cost * (1 - tshirt_discount)
  let key_chain_bracelet_actual := key_chain_bracelet_cost * (1 + key_chain_tax + bracelet_tax)
  (tshirt_actual + key_chain_bracelet_actual) * conversion_rate

/-- Theorem stating the total amount spent on souvenirs in US dollars -/
theorem total_spent_usd_value :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_spent_usd 347 146 0.1 0.12 0.08 0.75 - 447.98| < ε :=
sorry

end NUMINAMATH_CALUDE_total_spent_usd_value_l653_65382


namespace NUMINAMATH_CALUDE_circle_properties_l653_65398

-- Define the circle C: (x-2)²+y²=1
def C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point P(m,n) on the circle C
def P (m n : ℝ) : Prop := C m n

-- Theorem statement
theorem circle_properties :
  (∃ (m₀ n₀ : ℝ), P m₀ n₀ ∧ ∀ (m n : ℝ), P m n → |n / m| ≤ |n₀ / m₀| ∧ |n₀ / m₀| = Real.sqrt 3 / 3) ∧
  (∃ (m₁ n₁ : ℝ), P m₁ n₁ ∧ ∀ (m n : ℝ), P m n → m^2 + n^2 ≤ m₁^2 + n₁^2 ∧ m₁^2 + n₁^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l653_65398


namespace NUMINAMATH_CALUDE_blue_balls_removed_l653_65339

theorem blue_balls_removed (total_initial : ℕ) (blue_initial : ℕ) (prob_after : ℚ) : 
  total_initial = 18 → 
  blue_initial = 6 → 
  prob_after = 1/5 → 
  ∃ (removed : ℕ), 
    removed ≤ blue_initial ∧ 
    (blue_initial - removed : ℚ) / (total_initial - removed : ℚ) = prob_after ∧
    removed = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_removed_l653_65339


namespace NUMINAMATH_CALUDE_alpha_30_sufficient_not_necessary_for_sin_half_l653_65380

theorem alpha_30_sufficient_not_necessary_for_sin_half :
  (∀ α : Real, α = 30 * π / 180 → Real.sin α = 1 / 2) ∧
  (∃ α : Real, Real.sin α = 1 / 2 ∧ α ≠ 30 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_alpha_30_sufficient_not_necessary_for_sin_half_l653_65380


namespace NUMINAMATH_CALUDE_score_difference_proof_l653_65307

def score_distribution : List (ℝ × ℝ) := [
  (60, 0.15),
  (75, 0.20),
  (85, 0.25),
  (90, 0.10),
  (100, 0.30)
]

def mean_score : ℝ := (score_distribution.map (fun (score, percent) => score * percent)).sum

def median_score : ℝ := 85

theorem score_difference_proof :
  mean_score - median_score = -0.75 := by sorry

end NUMINAMATH_CALUDE_score_difference_proof_l653_65307


namespace NUMINAMATH_CALUDE_gcd_ABC_l653_65333

-- Define the constants
def a : ℕ := 177
def b : ℕ := 173

-- Define A, B, and C using the given formulas
def A : ℕ := a^5 + (a*b) * b^3 - b^5
def B : ℕ := b^5 + (a*b) * a^3 - a^5
def C : ℕ := b^4 + (a*b)^2 + a^4

-- State the theorem
theorem gcd_ABC : 
  Nat.gcd A C = 30637 ∧ Nat.gcd B C = 30637 := by
  sorry

end NUMINAMATH_CALUDE_gcd_ABC_l653_65333


namespace NUMINAMATH_CALUDE_equation_result_l653_65372

theorem equation_result (x y : ℤ) (h1 : 2 * x - y = 20) (h2 : 3 * y^2 = 48) :
  3 * x + y = 40 := by
  sorry

end NUMINAMATH_CALUDE_equation_result_l653_65372


namespace NUMINAMATH_CALUDE_line_constant_value_l653_65387

theorem line_constant_value (m n p : ℝ) (h : p = 1/3) :
  ∃ C : ℝ, (m = 6*n + C ∧ m + 2 = 6*(n + p) + C) → C = 0 :=
sorry

end NUMINAMATH_CALUDE_line_constant_value_l653_65387


namespace NUMINAMATH_CALUDE_fraction_simplification_l653_65361

theorem fraction_simplification : 
  (1 / (1 + 1 / (3 + 1 / 4))) = 13 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l653_65361


namespace NUMINAMATH_CALUDE_new_mean_after_removal_l653_65384

def original_mean : ℝ := 42
def original_count : ℕ := 65
def removed_score : ℝ := 50
def removed_count : ℕ := 6

theorem new_mean_after_removal :
  let original_sum := original_mean * original_count
  let removed_sum := removed_score * removed_count
  let new_sum := original_sum - removed_sum
  let new_count := original_count - removed_count
  let new_mean := new_sum / new_count
  ∃ ε > 0, |new_mean - 41.2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_new_mean_after_removal_l653_65384


namespace NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_27_l653_65383

theorem sqrt_12_minus_sqrt_27 : Real.sqrt 12 - Real.sqrt 27 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_27_l653_65383


namespace NUMINAMATH_CALUDE_EF_length_l653_65395

-- Define the segment AB and points C, D, E, F
def AB : ℝ := 26
def AC : ℝ := 1
def AD : ℝ := 8

-- Define the semicircle with diameter AB
def semicircle (x y : ℝ) : Prop :=
  x ≥ 0 ∧ x ≤ AB ∧ y ≥ 0 ∧ x * (AB - x) = y^2

-- Define the perpendicularity condition
def perpendicular (x y : ℝ) : Prop :=
  semicircle x y ∧ (x = AC ∨ x = AD)

-- Theorem statement
theorem EF_length :
  ∃ (xE yE xF yF : ℝ),
    perpendicular xE yE ∧
    perpendicular xF yF ∧
    xE = AC ∧
    xF = AD ∧
    (yF - yE)^2 + (xF - xE)^2 = (7 * Real.sqrt 2)^2 :=
sorry

end NUMINAMATH_CALUDE_EF_length_l653_65395


namespace NUMINAMATH_CALUDE_total_pencils_l653_65353

theorem total_pencils (drawer : ℕ) (desk : ℕ) (added : ℕ) 
  (h1 : drawer = 43)
  (h2 : desk = 19)
  (h3 : added = 16) :
  drawer + desk + added = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l653_65353


namespace NUMINAMATH_CALUDE_coffee_order_total_cost_l653_65369

theorem coffee_order_total_cost : 
  let drip_coffee_price : ℝ := 2.25
  let drip_coffee_discount : ℝ := 0.1
  let espresso_price : ℝ := 3.50
  let espresso_tax : ℝ := 0.15
  let latte_price : ℝ := 4.00
  let vanilla_syrup_price : ℝ := 0.50
  let vanilla_syrup_tax : ℝ := 0.20
  let cold_brew_price : ℝ := 2.50
  let cold_brew_discount : ℝ := 1.00
  let cappuccino_price : ℝ := 3.50
  let cappuccino_tip : ℝ := 0.05

  let drip_coffee_cost := 2 * drip_coffee_price * (1 - drip_coffee_discount)
  let espresso_cost := espresso_price * (1 + espresso_tax)
  let latte_cost := latte_price + (latte_price / 2) + (vanilla_syrup_price * (1 + vanilla_syrup_tax))
  let cold_brew_cost := 2 * cold_brew_price - cold_brew_discount
  let cappuccino_cost := cappuccino_price * (1 + cappuccino_tip)

  let total_cost := drip_coffee_cost + espresso_cost + latte_cost + cold_brew_cost + cappuccino_cost

  total_cost = 22.35 := by sorry

end NUMINAMATH_CALUDE_coffee_order_total_cost_l653_65369


namespace NUMINAMATH_CALUDE_bond_coupon_income_is_135_l653_65363

/-- Represents a bond with its characteristics -/
structure Bond where
  purchase_price : ℝ
  face_value : ℝ
  current_yield : ℝ
  duration : ℕ

/-- Calculates the annual coupon income for a given bond -/
def annual_coupon_income (b : Bond) : ℝ :=
  b.current_yield * b.purchase_price

/-- Theorem stating that for the given bond, the annual coupon income is 135 rubles -/
theorem bond_coupon_income_is_135 (b : Bond) 
  (h1 : b.purchase_price = 900)
  (h2 : b.face_value = 1000)
  (h3 : b.current_yield = 0.15)
  (h4 : b.duration = 3) :
  annual_coupon_income b = 135 := by
  sorry

end NUMINAMATH_CALUDE_bond_coupon_income_is_135_l653_65363


namespace NUMINAMATH_CALUDE_counterexample_disproves_conjecture_l653_65379

def isOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def isPrime (p : ℤ) : Prop := p > 1 ∧ ∀ m : ℤ, m > 1 → m < p → ¬(p % m = 0)

def isSumOfThreePrimes (n : ℤ) : Prop :=
  ∃ p q r : ℤ, isPrime p ∧ isPrime q ∧ isPrime r ∧ n = p + q + r

theorem counterexample_disproves_conjecture :
  ∃ n : ℤ, n > 5 ∧ isOdd n ∧ ¬(isSumOfThreePrimes n) →
  ¬(∀ m : ℤ, m > 5 → isOdd m → isSumOfThreePrimes m) :=
sorry

end NUMINAMATH_CALUDE_counterexample_disproves_conjecture_l653_65379


namespace NUMINAMATH_CALUDE_polynomial_properties_l653_65309

theorem polynomial_properties :
  (∀ x : ℝ, x^2 + 2*x - 3 = (x-1)*(x+3)) ∧
  (∀ x : ℝ, x^2 + 4*x + 5 ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_properties_l653_65309


namespace NUMINAMATH_CALUDE_no_two_digit_double_square_sum_l653_65355

theorem no_two_digit_double_square_sum :
  ¬ ∃ (N : ℕ), 
    (10 ≤ N ∧ N ≤ 99) ∧ 
    (∃ (k : ℕ), N + (10 * (N % 10) + N / 10) = 2 * k^2) :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_double_square_sum_l653_65355


namespace NUMINAMATH_CALUDE_james_calculator_problem_l653_65321

theorem james_calculator_problem (x y : ℚ) :
  x = 0.005 →
  y = 3.24 →
  (5 : ℕ) * 324 = 1620 →
  x * y = 0.0162 := by
sorry

end NUMINAMATH_CALUDE_james_calculator_problem_l653_65321


namespace NUMINAMATH_CALUDE_stratified_sample_second_year_l653_65368

/-- Represents the number of students in each year of high school -/
structure HighSchool where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the total number of students in the high school -/
def total_students (hs : HighSchool) : ℕ :=
  hs.first_year + hs.second_year + hs.third_year

/-- Calculates the number of students from a specific year in a stratified sample -/
def stratified_sample_size (hs : HighSchool) (year_size : ℕ) (sample_size : ℕ) : ℕ :=
  (year_size * sample_size) / total_students hs

/-- Theorem: In a stratified sample of 100 students from a high school with 1000 first-year,
    800 second-year, and 700 third-year students, the number of second-year students
    in the sample is 32. -/
theorem stratified_sample_second_year :
  let hs : HighSchool := ⟨1000, 800, 700⟩
  stratified_sample_size hs hs.second_year 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_second_year_l653_65368


namespace NUMINAMATH_CALUDE_city_rentals_cost_per_mile_l653_65336

/-- Proves that the cost per mile for City Rentals is $0.16 given the rental rates and equal cost for 48.0 miles. -/
theorem city_rentals_cost_per_mile :
  let sunshine_daily_rate : ℝ := 17.99
  let sunshine_per_mile : ℝ := 0.18
  let city_daily_rate : ℝ := 18.95
  let miles : ℝ := 48.0
  ∀ city_per_mile : ℝ,
    sunshine_daily_rate + sunshine_per_mile * miles = city_daily_rate + city_per_mile * miles →
    city_per_mile = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_city_rentals_cost_per_mile_l653_65336


namespace NUMINAMATH_CALUDE_three_numbers_problem_l653_65328

theorem three_numbers_problem (x y z : ℤ) : 
  x - y = 12 ∧ 
  (x + y) / 4 = 7 ∧ 
  z = 2 * y ∧ 
  x + z = 24 → 
  x = 20 ∧ y = 8 ∧ z = 16 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l653_65328


namespace NUMINAMATH_CALUDE_derivative_at_one_l653_65377

-- Define the function f(x) = (x-2)²
def f (x : ℝ) : ℝ := (x - 2)^2

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = -2 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l653_65377


namespace NUMINAMATH_CALUDE_total_money_divided_l653_65335

/-- The total amount of money divided among A, B, and C is 120, given the specified conditions. -/
theorem total_money_divided (a b c : ℕ) : 
  b = 20 → a = b + 20 → c = a + 20 → a + b + c = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_money_divided_l653_65335


namespace NUMINAMATH_CALUDE_angle_complement_theorem_l653_65370

theorem angle_complement_theorem (A : ℝ) :
  (90 - A = 60) → A = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_theorem_l653_65370


namespace NUMINAMATH_CALUDE_division_with_equal_quotient_and_remainder_l653_65306

theorem division_with_equal_quotient_and_remainder :
  {N : ℕ | ∃ k : ℕ, 2014 = N * k + k ∧ k < N} = {2013, 1006, 105, 52} := by
  sorry

end NUMINAMATH_CALUDE_division_with_equal_quotient_and_remainder_l653_65306


namespace NUMINAMATH_CALUDE_reginas_earnings_l653_65329

/-- Represents Regina's farm and calculates her earnings -/
def ReginasFarm : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun num_cows num_pigs num_goats num_chickens num_rabbits
      cow_price pig_price goat_price chicken_price rabbit_price
      milk_income_per_cow rabbit_income_per_year maintenance_cost =>
    let total_animal_sale := num_cows * cow_price + num_pigs * pig_price +
                             num_goats * goat_price + num_chickens * chicken_price +
                             num_rabbits * rabbit_price
    let total_product_income := num_cows * milk_income_per_cow +
                                num_rabbits * rabbit_income_per_year
    total_animal_sale + total_product_income - maintenance_cost

/-- Theorem stating Regina's final earnings -/
theorem reginas_earnings :
  ReginasFarm 20 (4 * 20) ((4 * 20) / 2) (2 * 20) 30
               800 400 600 50 25
               500 10 10000 = 75050 := by
  sorry

end NUMINAMATH_CALUDE_reginas_earnings_l653_65329


namespace NUMINAMATH_CALUDE_half_angle_formulas_l653_65318

/-- For a triangle with sides a, b, c, angle α opposite side a, and semi-perimeter p = (a + b + c) / 2,
    we prove the half-angle formulas for cos and sin. -/
theorem half_angle_formulas (a b c : ℝ) (α : Real) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) 
    (h_angle : α = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) : 
  let p := (a + b + c) / 2
  (Real.cos (α / 2))^2 = p * (p - a) / (b * c) ∧ 
  (Real.sin (α / 2))^2 = (p - b) * (p - c) / (b * c) := by
sorry

end NUMINAMATH_CALUDE_half_angle_formulas_l653_65318


namespace NUMINAMATH_CALUDE_equation_solutions_l653_65356

theorem equation_solutions :
  (∃ x : ℝ, 6 * x - 7 = 4 * x - 5 ∧ x = 1) ∧
  (∃ x : ℝ, (1/2) * x - 6 = (3/4) * x ∧ x = -24) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l653_65356


namespace NUMINAMATH_CALUDE_x_squared_plus_2x_is_quadratic_l653_65331

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 + 2x = 0 is a quadratic equation -/
theorem x_squared_plus_2x_is_quadratic :
  is_quadratic_equation (λ x => x^2 + 2*x) :=
by
  sorry


end NUMINAMATH_CALUDE_x_squared_plus_2x_is_quadratic_l653_65331


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l653_65358

-- Define the quadratic function
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Theorem statement
theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c x = p a b c (21 - x)) →  -- Axis of symmetry at x = 10.5
  p a b c 0 = -4 →                           -- p(0) = -4
  p a b c 21 = -4 :=                         -- Conclusion: p(21) = -4
by
  sorry


end NUMINAMATH_CALUDE_quadratic_symmetry_l653_65358


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l653_65332

theorem complex_magnitude_problem (z : ℂ) (h : (1 - I) * z - 3 * I = 1) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l653_65332


namespace NUMINAMATH_CALUDE_valentines_packs_given_away_l653_65390

def initial_valentines : ℕ := 450
def remaining_valentines : ℕ := 70
def valentines_per_pack : ℕ := 10

theorem valentines_packs_given_away : 
  (initial_valentines - remaining_valentines) / valentines_per_pack = 38 := by
  sorry

end NUMINAMATH_CALUDE_valentines_packs_given_away_l653_65390


namespace NUMINAMATH_CALUDE_intersection_chord_length_l653_65312

/-- The line C in the Cartesian plane -/
def line_C (x y : ℝ) : Prop := x - y - 1 = 0

/-- The circle P in the Cartesian plane -/
def circle_P (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

/-- The theorem stating that the length of the chord formed by the intersection
    of line C and circle P is √2 -/
theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    line_C A.1 A.2 ∧ line_C B.1 B.2 ∧
    circle_P A.1 A.2 ∧ circle_P B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l653_65312


namespace NUMINAMATH_CALUDE_quadratic_function_evaluation_l653_65337

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem quadratic_function_evaluation :
  3 * g 2 + 2 * g (-2) = 85 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_evaluation_l653_65337


namespace NUMINAMATH_CALUDE_joan_has_ten_books_l653_65342

/-- The number of books Tom has -/
def tom_books : ℕ := 38

/-- The total number of books Joan and Tom have together -/
def total_books : ℕ := 48

/-- The number of books Joan has -/
def joan_books : ℕ := total_books - tom_books

theorem joan_has_ten_books : joan_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_ten_books_l653_65342


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l653_65330

-- Define the function representing the sum of distances
def f (x : ℝ) : ℝ := |x - 5| + |x + 1|

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x < 8} = Set.Ioo (-2 : ℝ) (6 : ℝ) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l653_65330


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l653_65314

/-- An ellipse passing through two given points with a focus on a coordinate axis. -/
structure Ellipse where
  -- The coefficients of the ellipse equation x²/a² + y²/b² = 1
  a : ℝ
  b : ℝ
  -- Condition: a > 0 and b > 0
  ha : a > 0
  hb : b > 0
  -- Condition: Passes through P1(-√6, 1)
  passes_p1 : 6 / a^2 + 1 / b^2 = 1
  -- Condition: Passes through P2(√3, -√2)
  passes_p2 : 3 / a^2 + 2 / b^2 = 1
  -- Condition: One focus on coordinate axis, perpendicular to minor axis vertices, passes through (-3, 3√2/2)
  focus_condition : 9 / a^2 + (9/2) / b^2 = 1

/-- The standard equation of the ellipse satisfies one of the given forms. -/
theorem ellipse_standard_equation (e : Ellipse) : 
  (e.a^2 = 9 ∧ e.b^2 = 3) ∨ 
  (e.a^2 = 18 ∧ e.b^2 = 9) ∨ 
  (e.a^2 = 45/4 ∧ e.b^2 = 45/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l653_65314


namespace NUMINAMATH_CALUDE_inequality_proof_l653_65376

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) ≥ (a*b + b*c + c*a)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l653_65376


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l653_65362

/-- Given line l1 with equation 4x + 5y - 8 = 0 -/
def l1 : ℝ → ℝ → Prop :=
  λ x y => 4*x + 5*y - 8 = 0

/-- Point A with coordinates (3, 2) -/
def A : ℝ × ℝ := (3, 2)

/-- The perpendicular line l2 passing through A -/
def l2 : ℝ → ℝ → Prop :=
  λ x y => 5*x - 4*y - 7 = 0

theorem perpendicular_line_through_point :
  (∀ x y, l2 x y ↔ 5*x - 4*y - 7 = 0) ∧
  l2 A.1 A.2 ∧
  (∀ x1 y1 x2 y2, l1 x1 y1 → l1 x2 y2 → l2 x1 y1 → l2 x2 y2 →
    (x2 - x1) * (4) + (y2 - y1) * (5) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l653_65362


namespace NUMINAMATH_CALUDE_charity_distribution_l653_65305

theorem charity_distribution (total_amount : ℝ) (donation_percentage : ℝ) (num_organizations : ℕ) : 
  total_amount = 2500 →
  donation_percentage = 0.80 →
  num_organizations = 8 →
  (total_amount * donation_percentage) / num_organizations = 250 := by
sorry

end NUMINAMATH_CALUDE_charity_distribution_l653_65305


namespace NUMINAMATH_CALUDE_cylinder_no_triangular_front_view_l653_65373

-- Define the set of solid geometries
inductive SolidGeometry
| Cylinder
| Cone
| Tetrahedron
| TriangularPrism

-- Define a function that determines if a solid geometry can have a triangular front view
def canHaveTriangularFrontView (s : SolidGeometry) : Prop :=
  match s with
  | SolidGeometry.Cylinder => False
  | _ => True

-- Theorem statement
theorem cylinder_no_triangular_front_view :
  ∀ s : SolidGeometry, ¬(canHaveTriangularFrontView s) ↔ s = SolidGeometry.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_cylinder_no_triangular_front_view_l653_65373


namespace NUMINAMATH_CALUDE_power_of_fraction_cube_l653_65301

theorem power_of_fraction_cube (x : ℝ) : ((1/2) * x^3)^2 = (1/4) * x^6 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_cube_l653_65301


namespace NUMINAMATH_CALUDE_marble_count_l653_65393

theorem marble_count (g y p : ℕ) : 
  y + p = 7 →  -- all but 7 are green
  g + p = 10 → -- all but 10 are yellow
  g + y = 5 →  -- all but 5 are purple
  g + y + p = 11 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l653_65393


namespace NUMINAMATH_CALUDE_triangle_area_is_16_l653_65343

/-- The area of the triangle formed by the intersection of three lines -/
def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ := sorry

/-- Line 1: y = 6 -/
def line1 : ℝ → ℝ := fun x ↦ 6

/-- Line 2: y = 2 + x -/
def line2 : ℝ → ℝ := fun x ↦ 2 + x

/-- Line 3: y = 2 - x -/
def line3 : ℝ → ℝ := fun x ↦ 2 - x

theorem triangle_area_is_16 : triangleArea line1 line2 line3 = 16 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_16_l653_65343


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l653_65317

/-- Given a circle with four angles around its center measured as 7x°, 3x°, 4x°, and x°,
    prove that x = 24°. -/
theorem angle_sum_in_circle (x : ℝ) : 
  (7 * x + 3 * x + 4 * x + x : ℝ) = 360 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l653_65317
