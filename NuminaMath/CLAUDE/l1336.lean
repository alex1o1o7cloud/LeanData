import Mathlib

namespace NUMINAMATH_CALUDE_numerical_trick_l1336_133675

theorem numerical_trick (x : ℝ) : ((6 * x - 21) / 3) - 2 * x = -7 := by
  sorry

end NUMINAMATH_CALUDE_numerical_trick_l1336_133675


namespace NUMINAMATH_CALUDE_divisibility_count_l1336_133668

theorem divisibility_count : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (1638 : ℤ) % (n^2 - 3) = 0) ∧ 
    (∀ n : ℕ, n > 0 ∧ (1638 : ℤ) % (n^2 - 3) = 0 → n ∈ S) ∧
    S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_count_l1336_133668


namespace NUMINAMATH_CALUDE_family_income_increase_l1336_133687

theorem family_income_increase (I : ℝ) (S M F G : ℝ) : 
  I > 0 →
  S = 0.05 * I →
  M = 0.15 * I →
  F = 0.25 * I →
  G = I - S - M - F →
  (2 * G - G) / I = 0.55 := by
sorry

end NUMINAMATH_CALUDE_family_income_increase_l1336_133687


namespace NUMINAMATH_CALUDE_min_value_theorem_l1336_133623

-- Define the equation
def equation (x y : ℝ) : Prop := y^2 - 2*x + 4 = 0

-- Define the expression to minimize
def expression (x y : ℝ) : ℝ := x^2 + y^2 + 2*x

-- Theorem statement
theorem min_value_theorem :
  ∃ (min : ℝ), min = -8 ∧
  (∀ (x y : ℝ), equation x y → expression x y ≥ min) ∧
  (∃ (x y : ℝ), equation x y ∧ expression x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1336_133623


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l1336_133689

/-- A hyperbola with given asymptote and point -/
structure Hyperbola where
  -- Asymptote equation: 3x + 4y = 0
  asymptote_slope : ℝ
  asymptote_slope_eq : asymptote_slope = -3/4
  -- Point on the hyperbola
  point : ℝ × ℝ
  point_eq : point = (4, 6)

/-- The standard form of a hyperbola -/
def standard_form (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating the standard form of the hyperbola -/
theorem hyperbola_standard_form (h : Hyperbola) :
  ∃ (a b : ℝ), a^2 = 48 ∧ b^2 = 27 ∧ 
  ∀ (x y : ℝ), standard_form a b x y ↔ 
    (∃ (t : ℝ), x = 3*t ∧ y = -4*t) ∨ (x, y) = h.point :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l1336_133689


namespace NUMINAMATH_CALUDE_trig_expression_equality_l1336_133638

theorem trig_expression_equality : 
  (Real.sin (20 * π / 180) * Real.sqrt (1 + Real.cos (40 * π / 180))) / 
  (Real.cos (50 * π / 180)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l1336_133638


namespace NUMINAMATH_CALUDE_sheetrock_length_l1336_133640

/-- Represents the properties of a rectangular sheetrock -/
structure Sheetrock where
  width : ℝ
  area : ℝ

/-- Theorem stating that a sheetrock with width 5 and area 30 has length 6 -/
theorem sheetrock_length (s : Sheetrock) (h1 : s.width = 5) (h2 : s.area = 30) :
  s.area / s.width = 6 := by
  sorry


end NUMINAMATH_CALUDE_sheetrock_length_l1336_133640


namespace NUMINAMATH_CALUDE_solution_set_l1336_133696

/-- A function representing the quadratic expression inside the absolute value -/
def f (a b x : ℝ) : ℝ := x^2 + 2*a*x + 3*a + b

/-- The condition for the inequality to have exactly one solution -/
def has_unique_solution (a b : ℝ) : Prop :=
  ∃! x, |f a b x| ≤ 4

/-- The theorem stating the solution set -/
theorem solution_set :
  ∀ a : ℝ, has_unique_solution a (a^2 - 3*a + 4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_l1336_133696


namespace NUMINAMATH_CALUDE_cube_difference_of_sum_and_diff_l1336_133609

theorem cube_difference_of_sum_and_diff (x y : ℕ) 
  (sum_eq : x + y = 64) 
  (diff_eq : x - y = 16) 
  (x_pos : x > 0) 
  (y_pos : y > 0) : 
  x^3 - y^3 = 50176 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_of_sum_and_diff_l1336_133609


namespace NUMINAMATH_CALUDE_ram_exam_result_l1336_133664

/-- The percentage of marks Ram got in his exam -/
def ram_percentage (marks_obtained : ℕ) (total_marks : ℕ) : ℚ :=
  (marks_obtained : ℚ) / (total_marks : ℚ) * 100

/-- Theorem stating that Ram's percentage is 90% -/
theorem ram_exam_result : ram_percentage 450 500 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ram_exam_result_l1336_133664


namespace NUMINAMATH_CALUDE_roll_sum_less_than_12_prob_value_l1336_133690

def roll_sum_less_than_12_prob : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := total_outcomes - 15
  favorable_outcomes / total_outcomes

theorem roll_sum_less_than_12_prob_value : 
  roll_sum_less_than_12_prob = 49 / 64 := by sorry

end NUMINAMATH_CALUDE_roll_sum_less_than_12_prob_value_l1336_133690


namespace NUMINAMATH_CALUDE_lollipop_cost_theorem_lollipop_cost_correct_l1336_133655

/-- The cost of n lollipops given that 2 lollipops cost $2.40 and 6 lollipops cost $7.20 -/
def lollipop_cost (n : ℕ) : ℚ :=
  1.20 * n

/-- Theorem stating that the lollipop_cost function satisfies the given conditions -/
theorem lollipop_cost_theorem :
  lollipop_cost 2 = 2.40 ∧ lollipop_cost 6 = 7.20 :=
by sorry

/-- Theorem proving that the lollipop_cost function is correct for all non-negative integers -/
theorem lollipop_cost_correct (n : ℕ) :
  lollipop_cost n = 1.20 * n :=
by sorry

end NUMINAMATH_CALUDE_lollipop_cost_theorem_lollipop_cost_correct_l1336_133655


namespace NUMINAMATH_CALUDE_zero_in_interval_l1336_133662

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1336_133662


namespace NUMINAMATH_CALUDE_inequality_always_holds_l1336_133635

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l1336_133635


namespace NUMINAMATH_CALUDE_binary_to_hex_conversion_l1336_133674

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its hexadecimal representation -/
def decimal_to_hex (n : Nat) : String :=
  let rec aux (m : Nat) (acc : String) : String :=
    if m = 0 then
      if acc.isEmpty then "0" else acc
    else
      let digit := m % 16
      let hex_digit := if digit < 10 then 
        Char.toString (Char.ofNat (digit + 48))
      else
        Char.toString (Char.ofNat (digit + 55))
      aux (m / 16) (hex_digit ++ acc)
  aux n ""

/-- The binary number 1011101₂ -/
def binary_number : List Bool := [true, false, true, true, true, false, true]

theorem binary_to_hex_conversion :
  (binary_to_decimal binary_number = 93) ∧
  (decimal_to_hex 93 = "5D") := by
  sorry

end NUMINAMATH_CALUDE_binary_to_hex_conversion_l1336_133674


namespace NUMINAMATH_CALUDE_hex_grid_half_path_l1336_133666

/-- Represents a point in the hexagonal grid -/
structure HexPoint where
  x : ℤ
  y : ℤ

/-- Represents a direction in the hexagonal grid -/
inductive HexDirection
  | Horizontal
  | LeftDiagonal
  | RightDiagonal

/-- Calculates the distance between two points in the hexagonal grid -/
def hexDistance (a b : HexPoint) : ℕ :=
  sorry

/-- Represents a path in the hexagonal grid -/
def HexPath := List HexDirection

/-- Checks if a path is valid (follows the grid lines) -/
def isValidPath (path : HexPath) (start finish : HexPoint) : Prop :=
  sorry

/-- Calculates the length of a path -/
def pathLength (path : HexPath) : ℕ :=
  sorry

/-- Checks if a path is the shortest between two points -/
def isShortestPath (path : HexPath) (start finish : HexPoint) : Prop :=
  isValidPath path start finish ∧
  pathLength path = hexDistance start finish

/-- Counts the number of steps in a single direction -/
def countDirectionSteps (path : HexPath) (direction : HexDirection) : ℕ :=
  sorry

theorem hex_grid_half_path (a b : HexPoint) (path : HexPath) :
  isShortestPath path a b →
  hexDistance a b = 100 →
  ∃ (direction : HexDirection), countDirectionSteps path direction = 50 :=
sorry

end NUMINAMATH_CALUDE_hex_grid_half_path_l1336_133666


namespace NUMINAMATH_CALUDE_min_value_function_l1336_133637

theorem min_value_function (x : ℝ) (h : x > 2) : 
  (x^2 - 4*x + 8) / (x - 2) ≥ 4 ∧ ∃ y > 2, (y^2 - 4*y + 8) / (y - 2) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_function_l1336_133637


namespace NUMINAMATH_CALUDE_quadratic_roots_property_integer_values_k_l1336_133665

theorem quadratic_roots_property (k : ℝ) (x₁ x₂ : ℝ) : 
  (4 * k * x₁^2 - 4 * k * x₁ + k + 1 = 0 ∧ 
   4 * k * x₂^2 - 4 * k * x₂ + k + 1 = 0) → 
  (2 * x₁ - x₂) * (x₁ - 2 * x₂) ≠ -3/2 :=
sorry

theorem integer_values_k (k : ℤ) :
  (∃ x₁ x₂ : ℝ, 4 * (k : ℝ) * x₁^2 - 4 * (k : ℝ) * x₁ + (k : ℝ) + 1 = 0 ∧
                4 * (k : ℝ) * x₂^2 - 4 * (k : ℝ) * x₂ + (k : ℝ) + 1 = 0 ∧
                ∃ n : ℤ, (x₁ / x₂ + x₂ / x₁ - 2 : ℝ) = n) ↔
  k = -2 ∨ k = -3 ∨ k = -5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_integer_values_k_l1336_133665


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1336_133620

theorem solve_linear_equation (x : ℝ) :
  2*x - 3*x + 5*x = 80 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1336_133620


namespace NUMINAMATH_CALUDE_digit_seven_place_value_l1336_133627

theorem digit_seven_place_value (n : ℕ) (p : ℕ) : 
  7 * 10^p - 7 = 693 → p = 2 := by sorry

end NUMINAMATH_CALUDE_digit_seven_place_value_l1336_133627


namespace NUMINAMATH_CALUDE_integer_values_less_than_sqrt2_l1336_133621

theorem integer_values_less_than_sqrt2 (x : ℤ) : 
  (|x| : ℝ) < Real.sqrt 2 → x = -1 ∨ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_values_less_than_sqrt2_l1336_133621


namespace NUMINAMATH_CALUDE_point_on_line_l1336_133603

theorem point_on_line (m n p : ℝ) : 
  (m = n / 5 - 2 / 5) ∧ (m + p = (n + 15) / 5 - 2 / 5) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1336_133603


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1336_133622

theorem quadratic_inequality_solution (x : ℝ) :
  -3 * x^2 + 5 * x + 4 < 0 ↔ x < 3/4 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1336_133622


namespace NUMINAMATH_CALUDE_machine_production_l1336_133641

/-- The number of shirts a machine can make per minute -/
def shirts_per_minute : ℕ := 8

/-- The number of minutes the machine worked -/
def minutes_worked : ℕ := 2

/-- The number of shirts made by the machine -/
def shirts_made : ℕ := shirts_per_minute * minutes_worked

theorem machine_production :
  shirts_made = 16 := by sorry

end NUMINAMATH_CALUDE_machine_production_l1336_133641


namespace NUMINAMATH_CALUDE_beau_and_sons_ages_equality_l1336_133695

/-- Represents the problem of finding when Beau's age equaled the sum of his sons' ages --/
theorem beau_and_sons_ages_equality (beau_age_today : ℕ) (sons_age_today : ℕ) : 
  beau_age_today = 42 →
  sons_age_today = 16 →
  ∃ (years_ago : ℕ), 
    beau_age_today - years_ago = 3 * (sons_age_today - years_ago) ∧
    years_ago = 3 := by
sorry

end NUMINAMATH_CALUDE_beau_and_sons_ages_equality_l1336_133695


namespace NUMINAMATH_CALUDE_seven_valid_methods_l1336_133600

/-- The number of valid purchasing methods for software and tapes -/
def validPurchaseMethods : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    60 * p.1 + 70 * p.2 ≤ 500 ∧
    p.1 ≥ 3 ∧
    p.2 ≥ 2)
    (Finset.product (Finset.range 9) (Finset.range 8))).card

/-- Theorem stating that there are exactly 7 valid purchasing methods -/
theorem seven_valid_methods : validPurchaseMethods = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_valid_methods_l1336_133600


namespace NUMINAMATH_CALUDE_taxi_fare_fraction_l1336_133611

/-- Represents the taxi fare structure and proves the fraction of a mile for each part. -/
theorem taxi_fare_fraction (first_part_cost : ℚ) (additional_part_cost : ℚ) 
  (total_distance : ℚ) (total_cost : ℚ) : 
  first_part_cost = 21/10 →
  additional_part_cost = 2/5 →
  total_distance = 8 →
  total_cost = 177/10 →
  ∃ (part_fraction : ℚ), 
    part_fraction = 7/39 ∧
    first_part_cost + (total_distance - 1) * (additional_part_cost / part_fraction) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_taxi_fare_fraction_l1336_133611


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1336_133657

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) *
  (Real.sqrt 7 / Real.sqrt 8) * (Real.sqrt 9 / Real.sqrt 10) =
  3 * Real.sqrt 1050 / 320 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1336_133657


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l1336_133608

/-- Proves that the ratio of time taken to row against the stream to the time taken to row in favor of the stream is 2:1, given that the boat's speed in still water is 3 times the stream's speed. -/
theorem rowing_time_ratio (B S D : ℝ) (h_positive : B > 0 ∧ S > 0 ∧ D > 0) (h_speed_ratio : B = 3 * S) :
  (D / (B - S)) / (D / (B + S)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_ratio_l1336_133608


namespace NUMINAMATH_CALUDE_sin_C_value_sin_law_variation_area_inequality_l1336_133648

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.S = (t.a + t.b)^2 - t.c^2 ∧ t.a + t.b = 4

-- Theorem statements
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) : 
  Real.sin t.C = 8 / 17 := by sorry

theorem sin_law_variation (t : Triangle) : 
  (t.a^2 - t.b^2) / t.c^2 = Real.sin (t.A - t.B) / Real.sin t.C := by sorry

theorem area_inequality (t : Triangle) : 
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * Real.sqrt 3 * t.S := by sorry

end NUMINAMATH_CALUDE_sin_C_value_sin_law_variation_area_inequality_l1336_133648


namespace NUMINAMATH_CALUDE_seven_balls_two_boxes_l1336_133677

def distribute_balls (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n / 2 + 1)) (λ k => Nat.choose n k)

theorem seven_balls_two_boxes : distribute_balls 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_two_boxes_l1336_133677


namespace NUMINAMATH_CALUDE_cinema_rows_l1336_133659

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 5^2 + tens * 5^1 + ones * 5^0

/-- Calculates the number of rows needed in a cinema -/
def calculateRows (seats : Nat) (peoplePerRow : Nat) : Nat :=
  (seats + peoplePerRow - 1) / peoplePerRow

theorem cinema_rows :
  let seatsBase5 : Nat := 312
  let peoplePerRow : Nat := 3
  let seatsBase10 : Nat := base5ToBase10 seatsBase5
  calculateRows seatsBase10 peoplePerRow = 28 := by
  sorry

end NUMINAMATH_CALUDE_cinema_rows_l1336_133659


namespace NUMINAMATH_CALUDE_simplify_expression_l1336_133633

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 4*x + 4) = |x - 2| + |x + 2| :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1336_133633


namespace NUMINAMATH_CALUDE_altitude_equation_correct_l1336_133663

-- Define the triangle vertices
def A : ℝ × ℝ := (-5, 3)
def B : ℝ × ℝ := (3, 7)
def C : ℝ × ℝ := (4, -1)

-- Define the vector BC
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define the altitude equation
def altitudeEquation (x y : ℝ) : Prop := x - 8*y + 29 = 0

-- Theorem statement
theorem altitude_equation_correct :
  ∀ x y : ℝ, altitudeEquation x y ↔
  ((x - A.1, y - A.2) • BC = 0 ∧ ∃ t : ℝ, (x, y) = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2))) :=
by sorry

end NUMINAMATH_CALUDE_altitude_equation_correct_l1336_133663


namespace NUMINAMATH_CALUDE_similar_right_triangles_shortest_side_l1336_133632

theorem similar_right_triangles_shortest_side 
  (leg1 : ℝ) (hyp1 : ℝ) (hyp2 : ℝ) 
  (h_right : leg1 ^ 2 + (hyp1 ^ 2 - leg1 ^ 2) = hyp1 ^ 2) 
  (h_leg1 : leg1 = 15) 
  (h_hyp1 : hyp1 = 17) 
  (h_hyp2 : hyp2 = 51) : 
  (leg1 * hyp2 / hyp1) = 24 := by sorry

end NUMINAMATH_CALUDE_similar_right_triangles_shortest_side_l1336_133632


namespace NUMINAMATH_CALUDE_slope_theorem_l1336_133614

/-- Given two points R(-3,9) and S(3,y) on a coordinate plane, 
    if the slope of the line through R and S is -2, then y = -3. -/
theorem slope_theorem (y : ℝ) : 
  let R : ℝ × ℝ := (-3, 9)
  let S : ℝ × ℝ := (3, y)
  let slope := (S.2 - R.2) / (S.1 - R.1)
  slope = -2 → y = -3 := by
sorry

end NUMINAMATH_CALUDE_slope_theorem_l1336_133614


namespace NUMINAMATH_CALUDE_green_eyes_count_l1336_133678

/-- The number of students with green eyes in Mrs. Jensen's preschool class -/
def green_eyes : ℕ := sorry

theorem green_eyes_count : green_eyes = 12 := by
  have total_students : ℕ := 40
  have red_hair : ℕ := 3 * green_eyes
  have both : ℕ := 8
  have neither : ℕ := 4

  have h1 : total_students = (green_eyes - both) + (red_hair - both) + both + neither := by sorry
  
  sorry

end NUMINAMATH_CALUDE_green_eyes_count_l1336_133678


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1336_133671

-- Define a quadratic trinomial
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Theorem statement
theorem quadratic_roots_property (a b c : ℝ) (h : discriminant a b c ≥ 0) :
  ∃ (x : ℝ), ¬(∀ (y : ℝ), discriminant (a^2) (b^2) (c^2) ≥ 0) ∧
  (∀ (z : ℝ), discriminant (a^3) (b^3) (c^3) ≥ 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1336_133671


namespace NUMINAMATH_CALUDE_solution_sum_l1336_133693

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

theorem solution_sum (a b : ℝ) :
  a > b →
  f (a^2) * f a = 0.72 →
  f (b^2) * f b = 0.72 →
  19 * a + 7 * b = 134 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l1336_133693


namespace NUMINAMATH_CALUDE_johns_socks_theorem_l1336_133629

/-- The number of pairs of matched socks John initially had -/
def initial_pairs : ℕ := 9

/-- The number of individual socks John loses -/
def lost_socks : ℕ := 5

/-- The greatest number of pairs of matched socks John can have left after losing socks -/
def remaining_pairs : ℕ := 7

theorem johns_socks_theorem :
  (2 * initial_pairs - lost_socks ≥ 2 * remaining_pairs) ∧
  (2 * (initial_pairs - 1) - lost_socks < 2 * remaining_pairs) := by
  sorry

end NUMINAMATH_CALUDE_johns_socks_theorem_l1336_133629


namespace NUMINAMATH_CALUDE_red_jellybeans_count_l1336_133691

/-- The number of red jellybeans in a jar -/
def num_red_jellybeans (total blue purple orange pink yellow : ℕ) : ℕ :=
  total - (blue + purple + orange + pink + yellow)

/-- Theorem stating the number of red jellybeans in the jar -/
theorem red_jellybeans_count :
  num_red_jellybeans 237 14 26 40 7 21 = 129 := by
  sorry

end NUMINAMATH_CALUDE_red_jellybeans_count_l1336_133691


namespace NUMINAMATH_CALUDE_initial_distance_calculation_l1336_133697

/-- Represents the scenario of two trucks traveling on the same route --/
structure TruckScenario where
  initial_distance : ℝ
  speed_x : ℝ
  speed_y : ℝ
  overtake_time : ℝ
  final_distance : ℝ

/-- Theorem stating the initial distance between trucks given the scenario conditions --/
theorem initial_distance_calculation (scenario : TruckScenario)
  (h1 : scenario.speed_x = 57)
  (h2 : scenario.speed_y = 63)
  (h3 : scenario.overtake_time = 3)
  (h4 : scenario.final_distance = 4)
  (h5 : scenario.speed_y > scenario.speed_x) :
  scenario.initial_distance = 14 := by
  sorry


end NUMINAMATH_CALUDE_initial_distance_calculation_l1336_133697


namespace NUMINAMATH_CALUDE_express_train_speed_l1336_133642

/-- 
Given two trains traveling towards each other from towns 390 km apart,
where the freight train travels 30 km/h slower than the express train,
and they pass each other after 3 hours,
prove that the speed of the express train is 80 km/h.
-/
theorem express_train_speed (distance : ℝ) (time : ℝ) (speed_difference : ℝ) 
  (h1 : distance = 390)
  (h2 : time = 3)
  (h3 : speed_difference = 30) : 
  ∃ (express_speed : ℝ), 
    express_speed * time + (express_speed - speed_difference) * time = distance ∧ 
    express_speed = 80 := by
  sorry

end NUMINAMATH_CALUDE_express_train_speed_l1336_133642


namespace NUMINAMATH_CALUDE_probability_at_least_twice_value_l1336_133684

def single_shot_probability : ℝ := 0.6
def number_of_shots : ℕ := 3

def probability_at_least_twice : ℝ :=
  (Nat.choose number_of_shots 2) * (single_shot_probability ^ 2) * (1 - single_shot_probability) +
  (Nat.choose number_of_shots 3) * (single_shot_probability ^ 3)

theorem probability_at_least_twice_value : 
  probability_at_least_twice = 81 / 125 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_twice_value_l1336_133684


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1336_133676

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x) / (3 * x - 1) > 1 ↔ 1/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1336_133676


namespace NUMINAMATH_CALUDE_point_on_line_k_l1336_133688

/-- A line passing through the origin with slope 1/5 -/
def line_k (x y : ℝ) : Prop := y = (1/5) * x

theorem point_on_line_k (x : ℝ) : 
  line_k x 1 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_k_l1336_133688


namespace NUMINAMATH_CALUDE_range_of_a_l1336_133660

theorem range_of_a (x a : ℝ) : 
  (∀ x, (|x + 1| > 2 → x > a) ∧ 
  (|x + 1| ≤ 2 → x ≤ a) ∧ 
  (∃ x, x ≤ a ∧ |x + 1| > 2)) → 
  a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1336_133660


namespace NUMINAMATH_CALUDE_unique_top_coloring_l1336_133658

/-- Represents the colors used for the cube corners -/
inductive Color
  | Red
  | Green
  | Blue
  | Purple

/-- Represents a corner of the cube -/
structure Corner where
  position : Fin 8
  color : Color

/-- Represents a cube with colored corners -/
structure ColoredCube where
  corners : Fin 8 → Corner

/-- Checks if all corners on a face have different colors -/
def faceHasDifferentColors (cube : ColoredCube) (face : Fin 6) : Prop := sorry

/-- Checks if the bottom four corners of the cube are colored with four different colors -/
def bottomCornersAreDifferent (cube : ColoredCube) : Prop := sorry

/-- The main theorem stating that there is only one way to color the top corners -/
theorem unique_top_coloring (cube : ColoredCube) : 
  bottomCornersAreDifferent cube →
  (∀ face, faceHasDifferentColors cube face) →
  ∃! topColoring : Fin 4 → Color, 
    ∀ i : Fin 4, (cube.corners (i + 4)).color = topColoring i :=
sorry

end NUMINAMATH_CALUDE_unique_top_coloring_l1336_133658


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1336_133698

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- The theorem states that the opposite of 2023 is -2023. -/
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1336_133698


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1336_133667

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 4 → 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1336_133667


namespace NUMINAMATH_CALUDE_domain_of_f_l1336_133639

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (5 - 2 * x) ^ (1/3)

-- State the theorem
theorem domain_of_f :
  ∀ x : ℝ, ∃ y : ℝ, f x = y :=
by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_l1336_133639


namespace NUMINAMATH_CALUDE_inequality_implies_sign_conditions_l1336_133694

theorem inequality_implies_sign_conditions (a b : ℝ) 
  (h : (|abs a - (a + b)| : ℝ) < |a - abs (a + b)|) : 
  a < 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_sign_conditions_l1336_133694


namespace NUMINAMATH_CALUDE_max_min_sum_f_l1336_133682

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + x) / (x^2 + 1)

theorem max_min_sum_f :
  (∃ M m : ℝ, (∀ x : ℝ, f x ≤ M ∧ m ≤ f x) ∧ (∃ x₁ x₂ : ℝ, f x₁ = M ∧ f x₂ = m) ∧ M + m = 2) :=
sorry

end NUMINAMATH_CALUDE_max_min_sum_f_l1336_133682


namespace NUMINAMATH_CALUDE_hexagon_area_right_triangle_l1336_133650

/-- Given a right-angled triangle with hypotenuse c and sum of legs d,
    the area of the hexagon formed by the outer vertices of squares
    drawn on the sides of the triangle is c^2 + d^2. -/
theorem hexagon_area_right_triangle (c d : ℝ) (h : c > 0) (h' : d > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = d ∧ a^2 + b^2 = c^2 ∧
  (c^2 + a^2 + b^2 + 2*a*b : ℝ) = c^2 + d^2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_right_triangle_l1336_133650


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l1336_133626

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 887 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l1336_133626


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l1336_133699

theorem greatest_prime_factor_of_sum (p : ℕ) :
  (∃ (q : ℕ), Nat.Prime q ∧ q ∣ (5^7 + 6^6) ∧ q ≥ p) →
  p ≤ 211 :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l1336_133699


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1336_133685

theorem quadratic_rewrite :
  ∃ (a b c : ℤ), a > 0 ∧
  (∀ x, 64 * x^2 + 80 * x - 72 = 0 ↔ (a * x + b)^2 = c) ∧
  a + b + c = 110 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1336_133685


namespace NUMINAMATH_CALUDE_students_speaking_both_languages_l1336_133656

/-- Theorem: In a class of 150 students, given that 55 speak English, 85 speak Telugu, 
    and 30 speak neither English nor Telugu, prove that 20 students speak both English and Telugu. -/
theorem students_speaking_both_languages (total : ℕ) (english : ℕ) (telugu : ℕ) (neither : ℕ) :
  total = 150 →
  english = 55 →
  telugu = 85 →
  neither = 30 →
  english + telugu - (total - neither) = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_students_speaking_both_languages_l1336_133656


namespace NUMINAMATH_CALUDE_average_of_B_and_C_l1336_133647

theorem average_of_B_and_C (A B C : ℕ) : 
  A + B + C = 111 →
  (A + B) / 2 = 31 →
  (A + C) / 2 = 37 →
  (B + C) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_of_B_and_C_l1336_133647


namespace NUMINAMATH_CALUDE_cos_negative_seventy_nine_pi_sixths_l1336_133630

theorem cos_negative_seventy_nine_pi_sixths :
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_negative_seventy_nine_pi_sixths_l1336_133630


namespace NUMINAMATH_CALUDE_tims_weekly_earnings_l1336_133634

def visitors_per_day : ℕ := 100
def days_normal : ℕ := 6
def earnings_per_visit : ℚ := 1 / 100

def total_visitors : ℕ := visitors_per_day * days_normal + 2 * (visitors_per_day * days_normal)

def total_earnings : ℚ := (total_visitors : ℚ) * earnings_per_visit

theorem tims_weekly_earnings : total_earnings = 18 := by
  sorry

end NUMINAMATH_CALUDE_tims_weekly_earnings_l1336_133634


namespace NUMINAMATH_CALUDE_tv_sales_effect_l1336_133661

theorem tv_sales_effect (P Q : ℝ) (h_pos_P : P > 0) (h_pos_Q : Q > 0) :
  let new_price := 0.8 * P
  let new_quantity := 1.8 * Q
  let original_value := P * Q
  let new_value := new_price * new_quantity
  (new_value - original_value) / original_value = 0.44 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_effect_l1336_133661


namespace NUMINAMATH_CALUDE_parabola_equation_l1336_133628

-- Define a parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the condition that the parabola passes through a point
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

-- Define the condition that the vertex is at the origin
def vertex_at_origin (p : Parabola) : Prop :=
  p.equation 0 0

-- Define the condition that the axis of symmetry is a coordinate axis
def axis_is_coordinate (p : Parabola) : Prop :=
  (∀ x y : ℝ, p.equation x y ↔ p.equation x (-y)) ∨
  (∀ x y : ℝ, p.equation x y ↔ p.equation (-x) y)

-- Theorem statement
theorem parabola_equation 
  (p : Parabola) 
  (h1 : vertex_at_origin p) 
  (h2 : axis_is_coordinate p) 
  (h3 : passes_through p (-2) (-4)) :
  (∀ x y : ℝ, p.equation x y ↔ y^2 = -8*x) ∨
  (∀ x y : ℝ, p.equation x y ↔ x^2 = -y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1336_133628


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1336_133616

def polynomial (x : ℝ) : ℝ := 5*x^8 - 3*x^7 + 2*x^6 - 8*x^4 + 6*x^3 - 9

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem polynomial_remainder :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 26207 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1336_133616


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1336_133602

theorem complex_fraction_simplification :
  (3 + 5*Complex.I) / (-2 + 3*Complex.I) = 9/13 - 19/13 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1336_133602


namespace NUMINAMATH_CALUDE_sqrt_25_equals_5_l1336_133683

theorem sqrt_25_equals_5 : Real.sqrt 25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_25_equals_5_l1336_133683


namespace NUMINAMATH_CALUDE_total_weight_of_sand_l1336_133612

/-- The total weight of sand in two jugs with different capacities and sand densities -/
theorem total_weight_of_sand (jug1_capacity jug2_capacity : ℝ)
  (fill_percentage : ℝ)
  (density1 density2 : ℝ) :
  jug1_capacity = 2 →
  jug2_capacity = 3 →
  fill_percentage = 0.7 →
  density1 = 4 →
  density2 = 5 →
  jug1_capacity * fill_percentage * density1 +
  jug2_capacity * fill_percentage * density2 = 16.1 := by
  sorry

#check total_weight_of_sand

end NUMINAMATH_CALUDE_total_weight_of_sand_l1336_133612


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1336_133619

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The first line equation: y = 3x + 7 -/
def line1 (x y : ℝ) : Prop := y = 3 * x + 7

/-- The second line equation: 4y + bx = 12 -/
def line2 (x y b : ℝ) : Prop := 4 * y + b * x = 12

/-- The theorem stating that if the two given lines are perpendicular, then b = 4/3 -/
theorem perpendicular_lines_b_value (b : ℝ) :
  (∀ x y, line1 x y → line2 x y b → perpendicular 3 (-b/4)) →
  b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1336_133619


namespace NUMINAMATH_CALUDE_domain_of_f_sqrt_x_minus_2_l1336_133649

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-1) 0

-- State the theorem
theorem domain_of_f_sqrt_x_minus_2 :
  (∀ x ∈ domain_f_x_plus_1, f (x + 1) ∈ Set.Icc 0 1) →
  {x : ℝ | f (Real.sqrt x - 2) ∈ Set.Icc 0 1} = Set.Icc 4 9 :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_sqrt_x_minus_2_l1336_133649


namespace NUMINAMATH_CALUDE_measure_six_liters_possible_l1336_133692

/-- Represents the state of milk distribution among containers -/
structure MilkState :=
  (container : ℕ)
  (jug9 : ℕ)
  (jug5 : ℕ)
  (bucket10 : ℕ)

/-- Represents a pouring action between two containers -/
inductive PourAction
  | ContainerTo9
  | ContainerTo5
  | NineToContainer
  | NineTo10
  | NineTo5
  | FiveTo9
  | FiveTo10
  | FiveToContainer

/-- Applies a pouring action to a milk state -/
def applyAction (state : MilkState) (action : PourAction) : MilkState :=
  sorry

/-- Checks if the given sequence of actions results in 6 liters in the 10-liter bucket -/
def isValidSolution (actions : List PourAction) : Bool :=
  sorry

/-- Proves that it's possible to measure out 6 liters using given containers -/
theorem measure_six_liters_possible :
  ∃ (actions : List PourAction), isValidSolution actions = true :=
sorry

end NUMINAMATH_CALUDE_measure_six_liters_possible_l1336_133692


namespace NUMINAMATH_CALUDE_basket_weight_proof_l1336_133653

def basket_problem (num_pears : ℕ) (pear_weight : ℝ) (total_weight : ℝ) : Prop :=
  let pears_weight := num_pears * pear_weight
  let basket_weight := total_weight - pears_weight
  basket_weight = 0.46

theorem basket_weight_proof :
  basket_problem 30 0.36 11.26 := by
  sorry

end NUMINAMATH_CALUDE_basket_weight_proof_l1336_133653


namespace NUMINAMATH_CALUDE_arithmetic_progression_implies_linear_l1336_133651

/-- A function f: ℚ → ℚ satisfies the arithmetic progression property if
    for all rational numbers x < y < z < t in arithmetic progression,
    f(y) + f(z) = f(x) + f(t) -/
def ArithmeticProgressionProperty (f : ℚ → ℚ) : Prop :=
  ∀ (x y z t : ℚ), x < y ∧ y < z ∧ z < t ∧ (y - x = z - y) ∧ (z - y = t - z) →
    f y + f z = f x + f t

/-- The main theorem stating that any function satisfying the arithmetic progression property
    is a linear function -/
theorem arithmetic_progression_implies_linear
  (f : ℚ → ℚ) (h : ArithmeticProgressionProperty f) :
  ∃ (C : ℚ), ∀ (x : ℚ), f x = C * x := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_implies_linear_l1336_133651


namespace NUMINAMATH_CALUDE_cubic_expression_equals_1000_l1336_133681

theorem cubic_expression_equals_1000 (α : ℝ) (h : α = 6) : 
  α^3 + 3*α^2*4 + 3*α*16 + 64 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equals_1000_l1336_133681


namespace NUMINAMATH_CALUDE_road_breadth_from_fallen_tree_l1336_133606

/-- The breadth of a road when a tree falls across it -/
theorem road_breadth_from_fallen_tree (tree_height : ℝ) (break_height : ℝ) (road_breadth : ℝ) : 
  tree_height = 36 →
  break_height = 16 →
  (tree_height - break_height) ^ 2 = road_breadth ^ 2 + break_height ^ 2 →
  road_breadth = 12 := by
sorry

end NUMINAMATH_CALUDE_road_breadth_from_fallen_tree_l1336_133606


namespace NUMINAMATH_CALUDE_factorization_exists_l1336_133625

theorem factorization_exists : ∃ (a b c : ℤ), ∀ x : ℝ,
  (x - a) * (x - 10) + 1 = (x + b) * (x + c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_exists_l1336_133625


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_property_l1336_133613

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a line -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Checks if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is inscribed in a circle -/
def Quadrilateral.inscribed (q : Quadrilateral) (c : Circle) : Prop :=
  q.A.onCircle c ∧ q.B.onCircle c ∧ q.C.onCircle c ∧ q.D.onCircle c

/-- Represents the tangent line at a point on a circle -/
def tangentLine (p : Point) (c : Circle) : Line :=
  sorry

/-- Checks if two lines intersect at a point -/
def Line.intersectAt (l1 l2 : Line) (p : Point) : Prop :=
  p.onLine l1 ∧ p.onLine l2

/-- Calculates the distance between two points -/
def Point.dist (p1 p2 : Point) : ℝ :=
  sorry

theorem inscribed_quadrilateral_property (c : Circle) (q : Quadrilateral) (K : Point) :
  q.inscribed c →
  Line.intersectAt (tangentLine q.B c) (tangentLine q.D c) K →
  K.onLine (Line.mk 0 1 0) →  -- Assuming y-axis is the line AC
  q.A.dist q.B * q.C.dist q.D = q.B.dist q.C * q.A.dist q.D ∧
  ∀ (P Q R : Point) (l : Line),
    l.intersectAt (Line.mk 1 0 0) P →  -- Assuming x-axis is the line BA
    l.intersectAt (Line.mk 1 1 0) Q →  -- Assuming y=x is the line BD
    l.intersectAt (Line.mk 0 1 0) R →  -- Assuming y-axis is the line BC
    P.dist Q = Q.dist R :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_property_l1336_133613


namespace NUMINAMATH_CALUDE_same_terminal_side_l1336_133679

theorem same_terminal_side (k : ℤ) : 
  ∃ k : ℤ, -390 = k * 360 + 330 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l1336_133679


namespace NUMINAMATH_CALUDE_base_three_20121_equals_178_l1336_133610

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem base_three_20121_equals_178 :
  base_three_to_decimal [1, 2, 1, 0, 2] = 178 := by
  sorry

end NUMINAMATH_CALUDE_base_three_20121_equals_178_l1336_133610


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1336_133672

/-- A parabola with parameter p > 0 and two points A and B on it, intersected by a line through its focus F -/
structure ParabolaWithIntersection where
  p : ℝ
  hp : p > 0
  A : ℝ × ℝ
  B : ℝ × ℝ
  hA : A.2^2 = 2 * p * A.1
  hB : B.2^2 = 2 * p * B.1
  hAF : A.1 = 3 - p/2
  hBF : B.1 = 2 - p/2

/-- The theorem stating that under the given conditions, p = 12/5 -/
theorem parabola_intersection_theorem (pwi : ParabolaWithIntersection) : pwi.p = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1336_133672


namespace NUMINAMATH_CALUDE_josh_film_purchase_l1336_133605

/-- The number of films Josh bought -/
def num_films : ℕ := 9

/-- The number of books Josh bought -/
def num_books : ℕ := 4

/-- The number of CDs Josh bought -/
def num_cds : ℕ := 6

/-- The cost of each film in dollars -/
def cost_per_film : ℕ := 5

/-- The cost of each book in dollars -/
def cost_per_book : ℕ := 4

/-- The cost of each CD in dollars -/
def cost_per_cd : ℕ := 3

/-- The total amount Josh spent in dollars -/
def total_spent : ℕ := 79

/-- Theorem stating that the number of films Josh bought is correct -/
theorem josh_film_purchase :
  num_films * cost_per_film + num_books * cost_per_book + num_cds * cost_per_cd = total_spent :=
by sorry

end NUMINAMATH_CALUDE_josh_film_purchase_l1336_133605


namespace NUMINAMATH_CALUDE_triangle_angle_c_ninety_degrees_l1336_133644

/-- Given a triangle ABC with sides a, b, c and angles A, B, C respectively,
    if b^2 + c^2 - bc = a^2 and a/b = √3, then angle C = 90°. -/
theorem triangle_angle_c_ninety_degrees 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : b^2 + c^2 - b*c = a^2)
  (h_ratio : a/b = Real.sqrt 3) : 
  C = Real.pi/2 := by
sorry


end NUMINAMATH_CALUDE_triangle_angle_c_ninety_degrees_l1336_133644


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l1336_133646

theorem isosceles_triangle_largest_angle (α β γ : Real) :
  -- The triangle is isosceles
  (α = β) →
  -- One of the angles opposite an equal side is 50°
  α = 50 →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 80°
  γ = 80 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l1336_133646


namespace NUMINAMATH_CALUDE_same_result_as_five_minus_seven_l1336_133686

theorem same_result_as_five_minus_seven : 5 - 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_same_result_as_five_minus_seven_l1336_133686


namespace NUMINAMATH_CALUDE_height_difference_is_9_l1336_133604

/-- The height of the Empire State Building in meters -/
def empire_state_height : ℝ := 443

/-- The height of the Petronas Towers in meters -/
def petronas_height : ℝ := 452

/-- The height difference between the Petronas Towers and the Empire State Building -/
def height_difference : ℝ := petronas_height - empire_state_height

theorem height_difference_is_9 : height_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_is_9_l1336_133604


namespace NUMINAMATH_CALUDE_students_wanting_fruit_l1336_133645

theorem students_wanting_fruit (red_apples green_apples extra_fruit : ℕ) 
  (h1 : red_apples = 42)
  (h2 : green_apples = 7)
  (h3 : extra_fruit = 40) :
  red_apples + green_apples - extra_fruit = 40 := by
  sorry

end NUMINAMATH_CALUDE_students_wanting_fruit_l1336_133645


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1336_133601

/-- Given a circle and a line intersecting at two points, prove the value of m and the equation of the circle with the intersection points as diameter. -/
theorem circle_intersection_theorem (x y : ℝ) (m : ℝ) : 
  let circle := x^2 + y^2 - 2*x - 4*y + m
  let line := x + 2*y - 4
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (circle = 0 ∧ line = 0 → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) ∧
    (x₁ * x₂ + y₁ * y₂ = 0) →
    (m = 8/5 ∧ 
     ∀ (x y : ℝ), x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔ 
       ((x - x₁)*(x - x₂) + (y - y₁)*(y - y₂) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1336_133601


namespace NUMINAMATH_CALUDE_fruit_count_l1336_133615

theorem fruit_count (total fruits apples oranges bananas : ℕ) : 
  total = 12 → 
  apples = 3 → 
  oranges = 5 → 
  total = apples + oranges + bananas → 
  bananas = 4 :=
by sorry

end NUMINAMATH_CALUDE_fruit_count_l1336_133615


namespace NUMINAMATH_CALUDE_evaluate_expressions_l1336_133652

-- Define the logarithm functions
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10
noncomputable def ln (x : ℝ) := Real.log x

-- Define the main theorem
theorem evaluate_expressions :
  (2 * Real.sqrt 3 * (12 ^ (1/6)) * (3 ^ (3/2)) = 6) ∧
  ((1/2) * lg 25 + lg 2 + ln (Real.sqrt (Real.exp 1)) - 
   (Real.log 27 / Real.log 2) * (Real.log 2 / Real.log 3) - 
   7 ^ (Real.log 3 / Real.log 7) = -9/2) := by
sorry

end NUMINAMATH_CALUDE_evaluate_expressions_l1336_133652


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l1336_133669

/-- A trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference

/-- The properties of an arithmetic trapezoid -/
def ArithmeticTrapezoid.valid (t : ArithmeticTrapezoid) : Prop :=
  -- Sum of interior angles is 360°
  t.a + (t.a + t.d) + (t.a + 2*t.d) + (t.a + 3*t.d) = 360 ∧
  -- Largest angle is 150°
  t.a + 3*t.d = 150

theorem smallest_angle_measure (t : ArithmeticTrapezoid) (h : t.valid) :
  t.a = 15 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l1336_133669


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1336_133617

theorem fraction_equation_solution (x : ℚ) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) → x = -2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1336_133617


namespace NUMINAMATH_CALUDE_initial_customers_l1336_133631

theorem initial_customers (stayed : ℕ) (left : ℕ) : stayed = 3 → left = stayed + 5 → stayed + left = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_customers_l1336_133631


namespace NUMINAMATH_CALUDE_magic_trick_basis_l1336_133624

/-- The set of valid dice face pairs -/
def DicePairs : Set (ℕ × ℕ) :=
  {p | 1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ 6}

/-- The set of possible numbers of dice in the spectator's pocket -/
def PocketCounts : Set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 21}

/-- The statement of the magic trick's mathematical basis -/
theorem magic_trick_basis :
  ∃ f : DicePairs → PocketCounts, Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_magic_trick_basis_l1336_133624


namespace NUMINAMATH_CALUDE_train_length_l1336_133670

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 24 → ∃ length : ℝ, 
  (abs (length - 199.92) < 0.01 ∧ length = speed * (1000 / 3600) * time) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1336_133670


namespace NUMINAMATH_CALUDE_problem_part1_problem_part2_l1336_133654

-- Part 1
theorem problem_part1 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |a * x - 1|) 
  (h2 : Set.Icc (-3) 1 = {x | f x ≤ 2}) : a = -1 := by sorry

-- Part 2
theorem problem_part2 (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) 
  (m : ℝ) (h2 : ∃ x, f (2 * x + 1) - f (x - 1) ≤ 3 - 2 * m) : m ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_problem_part1_problem_part2_l1336_133654


namespace NUMINAMATH_CALUDE_distribution_plans_count_l1336_133607

/-- The number of ways to distribute 3 distinct items into 3 distinct boxes,
    where each box must contain at least one item -/
def distribution_count : ℕ := 12

/-- Theorem stating that the number of distribution plans is correct -/
theorem distribution_plans_count : distribution_count = 12 := by
  sorry

end NUMINAMATH_CALUDE_distribution_plans_count_l1336_133607


namespace NUMINAMATH_CALUDE_greater_number_problem_l1336_133636

theorem greater_number_problem (x y : ℝ) (h1 : x ≥ y) (h2 : x > 0) (h3 : y > 0) 
  (h4 : x * y = 2048) (h5 : x + y - (x - y) = 64) : x = 64 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l1336_133636


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1336_133680

/-- The length of the major axis of an ellipse with given foci and y-axis tangency -/
theorem ellipse_major_axis_length : 
  let f₁ : ℝ × ℝ := (1, -3 + 2 * Real.sqrt 3)
  let f₂ : ℝ × ℝ := (1, -3 - 2 * Real.sqrt 3)
  ∀ (e : Set (ℝ × ℝ)), 
    (∃ (p : ℝ × ℝ), p ∈ e ∧ p.1 = 0) →  -- Tangent to y-axis
    (∀ (q : ℝ × ℝ), q ∈ e → ∃ (a b : ℝ), a * (q.1 - f₁.1)^2 + b * (q.2 - f₁.2)^2 = 1 ∧ 
                                         a * (q.1 - f₂.1)^2 + b * (q.2 - f₂.2)^2 = 1) →
    (∃ (major_axis : ℝ), major_axis = 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1336_133680


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1336_133673

theorem complex_equation_solution (z : ℂ) (h : Complex.I * z = 2 + 3 * Complex.I) : z = 3 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1336_133673


namespace NUMINAMATH_CALUDE_geese_count_l1336_133643

def geese_problem (n : ℕ) : Prop :=
  -- The number of geese is an integer (implied by ℕ)
  -- After each lake, the number of remaining geese is an integer
  (∀ k : ℕ, k ≤ 7 → ∃ m : ℕ, n * 2^(7 - k) - (2^(7 - k) - 1) = m) ∧
  -- The process continues for exactly 7 lakes
  -- At each lake, half of the remaining geese plus half a goose land (implied by the formula)
  -- After 7 lakes, no geese remain
  n * 2^0 - (2^0 - 1) = 0

theorem geese_count : ∃ n : ℕ, geese_problem n ∧ n = 127 := by
  sorry

end NUMINAMATH_CALUDE_geese_count_l1336_133643


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1336_133618

theorem cubic_equation_solutions (t : ℝ) :
  let f := fun x : ℝ => x^3 - 2*t*x^2 + t^3
  (f t = 0) ∧
  (f ((t*(1+Real.sqrt 5))/2) = 0) ∧
  (f ((t*(1-Real.sqrt 5))/2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1336_133618
