import Mathlib

namespace NUMINAMATH_CALUDE_square_minus_twice_plus_nine_equals_eleven_l2426_242650

theorem square_minus_twice_plus_nine_equals_eleven :
  let a : ℝ := 2 / (Real.sqrt 3 - 1)
  a^2 - 2*a + 9 = 11 := by sorry

end NUMINAMATH_CALUDE_square_minus_twice_plus_nine_equals_eleven_l2426_242650


namespace NUMINAMATH_CALUDE_prime_expressions_solution_l2426_242610

def f (n : ℤ) : ℤ := |n^3 - 4*n^2 + 3*n - 35|
def g (n : ℤ) : ℤ := |n^2 + 4*n + 8|

theorem prime_expressions_solution :
  {n : ℤ | Nat.Prime (f n).natAbs ∧ Nat.Prime (g n).natAbs} = {-3, -1, 5} := by
sorry

end NUMINAMATH_CALUDE_prime_expressions_solution_l2426_242610


namespace NUMINAMATH_CALUDE_max_viewers_per_week_l2426_242696

/-- Represents the number of times a series is broadcast per week -/
structure BroadcastCount where
  seriesA : ℕ
  seriesB : ℕ

/-- Calculates the total program time for a given broadcast count -/
def totalProgramTime (bc : BroadcastCount) : ℕ :=
  80 * bc.seriesA + 40 * bc.seriesB

/-- Calculates the total commercial time for a given broadcast count -/
def totalCommercialTime (bc : BroadcastCount) : ℕ :=
  bc.seriesA + bc.seriesB

/-- Calculates the total number of viewers for a given broadcast count -/
def totalViewers (bc : BroadcastCount) : ℕ :=
  600000 * bc.seriesA + 200000 * bc.seriesB

/-- Represents the constraints for the broadcast schedule -/
def validBroadcastCount (bc : BroadcastCount) : Prop :=
  totalProgramTime bc ≤ 320 ∧ totalCommercialTime bc ≥ 6

/-- Theorem: The maximum number of viewers per week is 2,000,000 -/
theorem max_viewers_per_week :
  ∃ (bc : BroadcastCount), validBroadcastCount bc ∧
  ∀ (bc' : BroadcastCount), validBroadcastCount bc' →
  totalViewers bc' ≤ 2000000 :=
sorry

end NUMINAMATH_CALUDE_max_viewers_per_week_l2426_242696


namespace NUMINAMATH_CALUDE_chord_length_inequality_l2426_242636

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define the line y = kx + 1
def line1 (k x y : ℝ) : Prop := y = k*x + 1

-- Define the line kx + y - 2 = 0
def line2 (k x y : ℝ) : Prop := k*x + y - 2 = 0

-- Define a function to calculate the chord length
noncomputable def chordLength (k : ℝ) (line : ℝ → ℝ → ℝ → Prop) : ℝ :=
  sorry -- Actual calculation of chord length would go here

-- Theorem statement
theorem chord_length_inequality (k : ℝ) :
  chordLength k line1 ≠ chordLength k line2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_inequality_l2426_242636


namespace NUMINAMATH_CALUDE_divisible_by_three_l2426_242643

theorem divisible_by_three (a b : ℕ) (h : 3 ∣ (a * b)) : 3 ∣ a ∨ 3 ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l2426_242643


namespace NUMINAMATH_CALUDE_young_employees_count_l2426_242654

theorem young_employees_count (young middle elderly : ℕ) 
  (ratio : young = 10 * (middle / 8) ∧ young = 10 * (elderly / 7))
  (sample_size : ℕ) (sample_prob : ℚ)
  (h_sample : sample_size = 200)
  (h_prob : sample_prob = 1/5) :
  young = 400 := by
  sorry

end NUMINAMATH_CALUDE_young_employees_count_l2426_242654


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l2426_242618

theorem certain_fraction_proof : 
  ∃ (x y : ℚ), (3 / 7) / (x / y) = (2 / 5) / (1 / 7) ∧ x / y = 15 / 98 :=
by sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l2426_242618


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2426_242648

theorem quadratic_inequality_solution_set (x : ℝ) :
  {x | x^2 - 5*x - 6 > 0} = {x | x < -1 ∨ x > 6} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2426_242648


namespace NUMINAMATH_CALUDE_walking_problem_solution_l2426_242632

/-- Two people walking in opposite directions --/
structure WalkingProblem where
  time : ℝ
  distance : ℝ
  speed1 : ℝ
  speed2 : ℝ

/-- The conditions of our specific problem --/
def problem : WalkingProblem where
  time := 5
  distance := 75
  speed1 := 10
  speed2 := 5 -- This is what we want to prove

theorem walking_problem_solution (p : WalkingProblem) 
  (h1 : p.time * (p.speed1 + p.speed2) = p.distance)
  (h2 : p.time = 5)
  (h3 : p.distance = 75)
  (h4 : p.speed1 = 10) :
  p.speed2 = 5 := by
  sorry

#check walking_problem_solution problem

end NUMINAMATH_CALUDE_walking_problem_solution_l2426_242632


namespace NUMINAMATH_CALUDE_smaller_circle_area_l2426_242694

-- Define the radius of the smaller circle
def r : ℝ := sorry

-- Define the radius of the larger circle
def R : ℝ := 3 * r

-- Define the length of the common tangent
def tangent_length : ℝ := 5

-- Theorem statement
theorem smaller_circle_area : 
  r^2 + tangent_length^2 = (R - r)^2 → π * r^2 = 25 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_area_l2426_242694


namespace NUMINAMATH_CALUDE_rectangle_to_square_width_third_l2426_242600

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- Theorem: Given a 9x27 rectangle that can be cut into two congruent hexagons
    which can be repositioned to form a square, one third of the rectangle's width is 9 -/
theorem rectangle_to_square_width_third (rect : Rectangle) (sq : Square) :
  rect.width = 27 ∧ 
  rect.height = 9 ∧ 
  sq.side ^ 2 = rect.width * rect.height →
  rect.width / 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_width_third_l2426_242600


namespace NUMINAMATH_CALUDE_expression_value_l2426_242639

theorem expression_value
  (a b c d e : ℝ)
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |e| = 1) :
  e^2 + 2023*c*d - (a + b)/20 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2426_242639


namespace NUMINAMATH_CALUDE_polynomial_positive_root_l2426_242698

/-- The polynomial has at least one positive real root if and only if q ≥ 3/2 -/
theorem polynomial_positive_root (q : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^6 + 3*q*x^4 + 3*x^4 + 3*q*x^2 + x^2 + 3*q + 1 = 0) ↔ q ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_positive_root_l2426_242698


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2426_242697

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 6) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 32 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2426_242697


namespace NUMINAMATH_CALUDE_expression_values_l2426_242627

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let expr := a / abs a + b / abs b + c / abs c + (a * b * c) / abs (a * b * c)
  expr = -4 ∨ expr = 0 ∨ expr = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l2426_242627


namespace NUMINAMATH_CALUDE_perfect_square_sum_l2426_242692

theorem perfect_square_sum : ∃ k : ℕ, 2^8 + 2^11 + 2^12 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l2426_242692


namespace NUMINAMATH_CALUDE_shaded_area_five_circles_plus_one_l2426_242679

/-- The area of the shaded region formed by five circles of radius 5 units
    intersecting at the origin, with an additional circle creating 10 similar sectors. -/
theorem shaded_area_five_circles_plus_one (r : ℝ) (n : ℕ) : 
  r = 5 → n = 10 → (n : ℝ) * (π * r^2 / 4 - r^2 / 2) = 62.5 * π - 125 := by
  sorry

#check shaded_area_five_circles_plus_one

end NUMINAMATH_CALUDE_shaded_area_five_circles_plus_one_l2426_242679


namespace NUMINAMATH_CALUDE_problem_statement_l2426_242631

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2426_242631


namespace NUMINAMATH_CALUDE_female_officers_count_l2426_242609

theorem female_officers_count (total_on_duty : ℕ) (male_on_duty : ℕ) 
  (female_on_duty_percentage : ℚ) :
  total_on_duty = 475 →
  male_on_duty = 315 →
  female_on_duty_percentage = 65/100 →
  ∃ (total_female : ℕ), 
    (total_female : ℚ) * female_on_duty_percentage = (total_on_duty - male_on_duty : ℚ) ∧
    total_female = 246 :=
by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2426_242609


namespace NUMINAMATH_CALUDE_unique_natural_solution_l2426_242670

theorem unique_natural_solution :
  ∃! (x y : ℕ), 3 * x + 7 * y = 23 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_natural_solution_l2426_242670


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_l2426_242605

-- Define a convex polygon
def ConvexPolygon (M : Set (ℝ × ℝ)) : Prop := sorry

-- Define an inscribed triangle in a polygon
def InscribedTriangle (T : Set (ℝ × ℝ)) (M : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (T : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a triangle formed by three vertices of a polygon
def VertexTriangle (T : Set (ℝ × ℝ)) (M : Set (ℝ × ℝ)) : Prop := sorry

theorem largest_inscribed_triangle (M : Set (ℝ × ℝ)) (h : ConvexPolygon M) :
  ∃ (T : Set (ℝ × ℝ)), VertexTriangle T M ∧
    ∀ (S : Set (ℝ × ℝ)), InscribedTriangle S M → TriangleArea S ≤ TriangleArea T :=
sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_l2426_242605


namespace NUMINAMATH_CALUDE_great_eight_teams_l2426_242665

/-- The number of teams in the GREAT EIGHT conference -/
def num_teams : ℕ := 9

/-- The total number of games played in the conference -/
def total_games : ℕ := 36

/-- The number of games played by one team -/
def games_per_team : ℕ := 8

/-- Calculates the number of games in a round-robin tournament -/
def round_robin_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem great_eight_teams :
  (round_robin_games num_teams = total_games) ∧
  (num_teams - 1 = games_per_team) := by
  sorry

end NUMINAMATH_CALUDE_great_eight_teams_l2426_242665


namespace NUMINAMATH_CALUDE_production_value_decrease_l2426_242601

theorem production_value_decrease (a : ℝ) :
  let increase_percent := a
  let decrease_percent := |a / (100 + a)|
  increase_percent > -100 →
  decrease_percent = |1 - 1 / (1 + a / 100)| :=
by sorry

end NUMINAMATH_CALUDE_production_value_decrease_l2426_242601


namespace NUMINAMATH_CALUDE_oil_price_reduction_is_fifty_percent_l2426_242673

/-- Calculates the percentage reduction in oil price given the reduced price and additional quantity -/
def oil_price_reduction (reduced_price : ℚ) (additional_quantity : ℚ) : ℚ :=
  let original_price := (800 : ℚ) / (((800 : ℚ) / reduced_price) - additional_quantity)
  ((original_price - reduced_price) / original_price) * 100

/-- Theorem stating that under the given conditions, the oil price reduction is 50% -/
theorem oil_price_reduction_is_fifty_percent :
  oil_price_reduction 80 5 = 50 := by sorry

end NUMINAMATH_CALUDE_oil_price_reduction_is_fifty_percent_l2426_242673


namespace NUMINAMATH_CALUDE_det_equals_nine_l2426_242659

-- Define the determinant for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- State the theorem
theorem det_equals_nine (x : ℝ) (h : x^2 - 2*x - 5 = 0) : 
  det2x2 (x + 1) x (4 - x) (x - 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_det_equals_nine_l2426_242659


namespace NUMINAMATH_CALUDE_distributive_property_example_l2426_242671

theorem distributive_property_example :
  (3/4 + 7/12 - 5/9) * (-36) = 3/4 * (-36) + 7/12 * (-36) - 5/9 * (-36) := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_example_l2426_242671


namespace NUMINAMATH_CALUDE_valid_sequences_l2426_242677

def is_valid_sequence (s : List Nat) : Prop :=
  s.length = 8 ∧
  s.count 1 = 2 ∧
  s.count 2 = 2 ∧
  s.count 3 = 2 ∧
  s.count 4 = 2 ∧
  (∃ i, s.get? i = some 1 ∧ s.get? (i + 2) = some 1) ∧
  (∃ i, s.get? i = some 2 ∧ s.get? (i + 3) = some 2) ∧
  (∃ i, s.get? i = some 3 ∧ s.get? (i + 4) = some 3) ∧
  (∃ i, s.get? i = some 4 ∧ s.get? (i + 5) = some 4)

theorem valid_sequences :
  is_valid_sequence [4, 1, 3, 1, 2, 4, 3, 2] ∧
  is_valid_sequence [2, 3, 4, 2, 1, 3, 1, 4] :=
by sorry

end NUMINAMATH_CALUDE_valid_sequences_l2426_242677


namespace NUMINAMATH_CALUDE_inequality_proof_l2426_242652

theorem inequality_proof (a b c : ℝ) 
  (h : a^2 * b * c + a * b^2 * c + a * b * c^2 = 1) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2426_242652


namespace NUMINAMATH_CALUDE_expression_zero_l2426_242626

theorem expression_zero (a b c : ℝ) (h : c = b + 2) :
  b = -2 ∧ c = 0 → (a - (b + c)) - ((a + c) - b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_zero_l2426_242626


namespace NUMINAMATH_CALUDE_blue_paint_cans_l2426_242603

theorem blue_paint_cans (total_cans : ℕ) (blue_ratio yellow_ratio : ℕ) 
  (h1 : total_cans = 42)
  (h2 : blue_ratio = 4)
  (h3 : yellow_ratio = 3) : 
  (blue_ratio * total_cans) / (blue_ratio + yellow_ratio) = 24 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_cans_l2426_242603


namespace NUMINAMATH_CALUDE_or_false_implies_both_false_l2426_242622

theorem or_false_implies_both_false (p q : Prop) : 
  (¬p ∨ ¬q) → (¬p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_or_false_implies_both_false_l2426_242622


namespace NUMINAMATH_CALUDE_art_team_arrangement_l2426_242674

/-- Given a team of 1000 members arranged in rows where each row from the second onward
    has one more person than the previous row, prove that there are 25 rows with 28 members
    in the first row. -/
theorem art_team_arrangement (k m : ℕ) : k > 16 →
  (k * (2 * m + k - 1)) / 2 = 1000 → k = 25 ∧ m = 28 := by
  sorry

end NUMINAMATH_CALUDE_art_team_arrangement_l2426_242674


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l2426_242625

theorem polynomial_product_expansion (x : ℝ) :
  (3 * x^2 + 4) * (2 * x^3 + x^2 + 5) = 6 * x^5 + 3 * x^4 + 8 * x^3 + 19 * x^2 + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l2426_242625


namespace NUMINAMATH_CALUDE_det_B_squared_minus_3B_l2426_242612

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det (B ^ 2 - 3 • B) = 88 := by
  sorry

end NUMINAMATH_CALUDE_det_B_squared_minus_3B_l2426_242612


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2426_242641

/-- Proves the number of yellow balls in a box given specific conditions -/
theorem yellow_balls_count (red yellow green : ℕ) : 
  red + yellow + green = 68 →
  yellow = 2 * red →
  3 * green = 4 * yellow →
  yellow = 24 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2426_242641


namespace NUMINAMATH_CALUDE_customers_in_other_countries_l2426_242656

/-- Represents the number of customers in different regions --/
structure CustomerDistribution where
  total : Nat
  usa : Nat
  canada : Nat

/-- Calculates the number of customers in other countries --/
def customersInOtherCountries (d : CustomerDistribution) : Nat :=
  d.total - (d.usa + d.canada)

/-- Theorem stating the number of customers in other countries --/
theorem customers_in_other_countries :
  let d : CustomerDistribution := {
    total := 7422,
    usa := 723,
    canada := 1297
  }
  customersInOtherCountries d = 5402 := by
  sorry

#eval customersInOtherCountries {total := 7422, usa := 723, canada := 1297}

end NUMINAMATH_CALUDE_customers_in_other_countries_l2426_242656


namespace NUMINAMATH_CALUDE_f_2021_value_l2426_242651

-- Define the set A
def A : Set ℚ := {x : ℚ | x ≠ -1 ∧ x ≠ 0}

-- Define the function property
def has_property (f : A → ℝ) : Prop :=
  ∀ x : A, f x + f ⟨1 + 1 / x, sorry⟩ = (1/2) * Real.log (abs (x : ℝ))

-- State the theorem
theorem f_2021_value (f : A → ℝ) (h : has_property f) :
  f ⟨2021, sorry⟩ = (1/2) * Real.log 2021 := by sorry

end NUMINAMATH_CALUDE_f_2021_value_l2426_242651


namespace NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_line_theorem_l2426_242614

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y + 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 2)

-- Define line l1
def line_l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- Define line l2
def line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y + 2 = 0

-- Theorem for parallel line l1
theorem parallel_line_theorem :
  (∀ x y : ℝ, line_l1 x y ↔ 3 * x + 4 * y - 11 = 0) ∧
  (line_l1 (point_A.1) (point_A.2)) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, line_l x y ↔ line_l1 (k * x) (k * y)) :=
sorry

-- Theorem for perpendicular line l2
theorem perpendicular_line_theorem :
  (∀ x y : ℝ, line_l2 x y ↔ 4 * x - 3 * y + 2 = 0) ∧
  (line_l2 (point_A.1) (point_A.2)) ∧
  (∀ x1 y1 x2 y2 : ℝ, line_l x1 y1 → line_l x2 y2 →
    3 * (x2 - x1) + 4 * (y2 - y1) = 0 →
    4 * (x2 - x1) - 3 * (y2 - y1) = 0) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_line_theorem_l2426_242614


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l2426_242666

/-- Given a parabola y^2 = 4x, point A(3,0), and a point P on the parabola,
    if a line through P intersects perpendicularly with x = -1 at B,
    and |PB| = |PA|, then the x-coordinate of P is 2. -/
theorem parabola_point_coordinates (P : ℝ × ℝ) :
  P.2^2 = 4 * P.1 →  -- P is on the parabola y^2 = 4x
  ∃ B : ℝ × ℝ, 
    B.1 = -1 ∧  -- B is on the line x = -1
    (P.2 - B.2) * (P.1 - B.1) = -1 ∧  -- PB is perpendicular to x = -1
    (P.1 - B.1)^2 + (P.2 - B.2)^2 = (P.1 - 3)^2 + P.2^2 →  -- |PB| = |PA|
  P.1 = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l2426_242666


namespace NUMINAMATH_CALUDE_tangent_and_roots_l2426_242676

noncomputable section

def F (x : ℝ) := x * Real.log x

def tangent_line (x y : ℝ) := 2 * x - y - Real.exp 1 = 0

def has_two_roots (t : ℝ) :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    Real.exp (-2) ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧
    F x₁ = t ∧ F x₂ = t

theorem tangent_and_roots :
  (∀ x y, F x = y → x = Real.exp 1 → tangent_line x y) ∧
  (∀ t, has_two_roots t ↔ -Real.exp (-1) < t ∧ t ≤ -2 * Real.exp (-2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_roots_l2426_242676


namespace NUMINAMATH_CALUDE_find_y_l2426_242646

def rotation_equivalence (y : ℝ) : Prop :=
  (480 % 360 : ℝ) = (360 - y) % 360 ∧ y < 360

theorem find_y : ∃ y : ℝ, rotation_equivalence y ∧ y = 240 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2426_242646


namespace NUMINAMATH_CALUDE_right_triangle_sin_cos_relation_l2426_242686

theorem right_triangle_sin_cos_relation (A B C : ℝ) :
  A = Real.pi / 2 →  -- ∠A = 90°
  Real.cos B = 3 / 5 →
  Real.sin C = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_cos_relation_l2426_242686


namespace NUMINAMATH_CALUDE_candy_distribution_l2426_242620

theorem candy_distribution (total_candy : ℕ) (candy_per_student : ℕ) (num_students : ℕ) : 
  total_candy = 18 → candy_per_student = 2 → total_candy = candy_per_student * num_students → num_students = 9 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2426_242620


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l2426_242684

theorem pure_imaginary_complex_fraction (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l2426_242684


namespace NUMINAMATH_CALUDE_same_solution_l2426_242608

theorem same_solution (x y : ℝ) : 
  (4 * x - 8 * y - 5 = 0) ↔ (8 * x - 16 * y - 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_same_solution_l2426_242608


namespace NUMINAMATH_CALUDE_smallest_coprime_to_210_l2426_242660

theorem smallest_coprime_to_210 :
  ∀ y : ℕ, y > 1 → y < 11 → Nat.gcd y 210 ≠ 1 ∧ Nat.gcd 11 210 = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_coprime_to_210_l2426_242660


namespace NUMINAMATH_CALUDE_sum_not_ending_in_seven_l2426_242687

theorem sum_not_ending_in_seven (n : ℕ) : ¬ (∃ k : ℕ, n * (n + 1) / 2 = 10 * k + 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_ending_in_seven_l2426_242687


namespace NUMINAMATH_CALUDE_simplify_expression_solve_equation_solve_system_l2426_242617

-- Part 1
theorem simplify_expression (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

-- Part 2
theorem solve_equation (x y : ℝ) (h : x^2 - 2*y = 4) :
  23 - 3*x^2 + 6*y = 11 := by sorry

-- Part 3
theorem solve_system (a b c d : ℝ) 
  (h1 : a - 2*b = 3) (h2 : 2*b - c = -5) (h3 : c - d = -9) :
  (a - c) + (2*b - d) - (2*b - c) = -11 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_equation_solve_system_l2426_242617


namespace NUMINAMATH_CALUDE_race_head_start_l2426_242661

/-- Proves the head start distance in a race with given conditions -/
theorem race_head_start 
  (race_distance : ℝ) 
  (speed_ratio : ℝ) 
  (win_margin : ℝ) 
  (h1 : race_distance = 600)
  (h2 : speed_ratio = 5/4)
  (h3 : win_margin = 200) :
  ∃ (head_start : ℝ), 
    head_start = 100 ∧ 
    (race_distance - head_start) / speed_ratio = (race_distance - win_margin) / 1 :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l2426_242661


namespace NUMINAMATH_CALUDE_point_P_coordinates_l2426_242619

-- Define the coordinate system and points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, -1)

-- Define vectors
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- State the theorem
theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, 
    vec A P = (2 : ℝ) • vec P B ∧ 
    P = (7/3, 1/3) := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l2426_242619


namespace NUMINAMATH_CALUDE_first_week_gain_l2426_242655

/-- Proves that the percentage gain in the first week was 25% --/
theorem first_week_gain (initial_investment : ℝ) (final_value : ℝ) : 
  initial_investment = 400 →
  final_value = 750 →
  ∃ (x : ℝ), 
    (initial_investment + x / 100 * initial_investment) * 1.5 = final_value ∧
    x = 25 := by
  sorry

#check first_week_gain

end NUMINAMATH_CALUDE_first_week_gain_l2426_242655


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l2426_242649

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : (x * x^(1/3))^(1/4) = x^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l2426_242649


namespace NUMINAMATH_CALUDE_geometric_subsequence_ratio_l2426_242672

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0

/-- The property that a_1, a_3, and a_7 form a geometric sequence -/
def IsGeometricSubsequence (seq : ArithmeticSequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ seq.a 3 = seq.a 1 * q ∧ seq.a 7 = seq.a 3 * q

/-- The theorem stating that the common ratio of the geometric subsequence is 2 -/
theorem geometric_subsequence_ratio (seq : ArithmeticSequence) 
  (h_geom : IsGeometricSubsequence seq) : 
  ∃ q : ℝ, q = 2 ∧ seq.a 3 = seq.a 1 * q ∧ seq.a 7 = seq.a 3 * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_subsequence_ratio_l2426_242672


namespace NUMINAMATH_CALUDE_sector_arc_length_l2426_242634

theorem sector_arc_length (θ : Real) (r : Real) (h1 : θ = 90) (h2 : r = 6) :
  (θ / 360) * (2 * Real.pi * r) = 3 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2426_242634


namespace NUMINAMATH_CALUDE_doghouse_accessible_area_l2426_242628

-- Define the doghouse
def doghouse_side_length : ℝ := 2

-- Define the tether length
def tether_length : ℝ := 3

-- Theorem statement
theorem doghouse_accessible_area :
  let total_sector_area := π * tether_length^2 * (240 / 360)
  let small_sector_area := 2 * (π * doghouse_side_length^2 * (60 / 360))
  total_sector_area + small_sector_area = (22 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_doghouse_accessible_area_l2426_242628


namespace NUMINAMATH_CALUDE_library_experience_l2426_242615

/-- Given two employees' years of experience satisfying certain conditions,
    prove that one employee has 10 years of experience. -/
theorem library_experience (b j : ℝ) 
  (h1 : j - 5 = 3 * (b - 5))
  (h2 : j = 2 * b) : 
  b = 10 := by sorry

end NUMINAMATH_CALUDE_library_experience_l2426_242615


namespace NUMINAMATH_CALUDE_checkerboard_square_count_l2426_242658

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  topLeftRow : Nat
  topLeftCol : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 10

/-- Checks if a square contains at least 5 black squares -/
def containsAtLeast5Black (s : Square) : Bool :=
  sorry

/-- Counts the number of valid squares of a given size -/
def countValidSquares (size : Nat) : Nat :=
  sorry

/-- Counts the total number of squares containing at least 5 black squares -/
def totalValidSquares : Nat :=
  sorry

/-- Main theorem: The number of distinct squares containing at least 5 black squares is 172 -/
theorem checkerboard_square_count : totalValidSquares = 172 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_square_count_l2426_242658


namespace NUMINAMATH_CALUDE_rowing_current_rate_l2426_242602

/-- Proves that the rate of the current is 1.4 km/hr given the conditions of the rowing problem -/
theorem rowing_current_rate (rowing_speed : ℝ) (upstream_time downstream_time : ℝ) : 
  rowing_speed = 4.2 →
  upstream_time = 2 * downstream_time →
  let current_rate := (rowing_speed / 3 : ℝ)
  current_rate = 1.4 := by sorry

end NUMINAMATH_CALUDE_rowing_current_rate_l2426_242602


namespace NUMINAMATH_CALUDE_gcd_consecutive_b_terms_l2426_242640

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem gcd_consecutive_b_terms (n : ℕ) : Nat.gcd (b n) (b (n + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_consecutive_b_terms_l2426_242640


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_line_l_equation_l2426_242662

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y - 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + y - 3 = 0

-- Define perpendicularity of lines
def perpendicular (m : ℝ) : Prop := 
  (m = 0) ∨ (m ≠ 0 ∧ (m + 2) / m * m = -1)

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (1, 2 * m)

-- Define the line l
def l (k : ℝ) (x y : ℝ) : Prop := y - 2 = k * (x - 1)

-- Define the intercept condition
def intercept_condition (k : ℝ) : Prop :=
  (k - 2) / k = 2 * (2 - k)

theorem perpendicular_lines_m (m : ℝ) : 
  perpendicular m → m = -3 ∨ m = 0 :=
sorry

theorem line_l_equation (m : ℝ) :
  l₂ m 1 (2 * m) →
  (∃ k, l k 1 (2 * m) ∧ intercept_condition k) →
  (∀ x y, l 2 x y ∨ l (-1/2) x y) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_line_l_equation_l2426_242662


namespace NUMINAMATH_CALUDE_y_value_l2426_242613

theorem y_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2426_242613


namespace NUMINAMATH_CALUDE_honey_production_l2426_242690

theorem honey_production (num_hives : ℕ) (jar_capacity : ℚ) (jars_for_half : ℕ) 
  (h1 : num_hives = 5)
  (h2 : jar_capacity = 1/2)
  (h3 : jars_for_half = 100) :
  (2 * jars_for_half : ℚ) * jar_capacity / num_hives = 20 := by
  sorry

end NUMINAMATH_CALUDE_honey_production_l2426_242690


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l2426_242635

/-- A function v is odd if v(-x) = -v(x) for all x in its domain -/
def IsOdd (v : ℝ → ℝ) : Prop := ∀ x, v (-x) = -v x

/-- The sum of v(-3.14), v(-1.57), v(1.57), and v(3.14) is zero for any odd function v -/
theorem odd_function_sum_zero (v : ℝ → ℝ) (h : IsOdd v) :
  v (-3.14) + v (-1.57) + v 1.57 + v 3.14 = 0 :=
sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l2426_242635


namespace NUMINAMATH_CALUDE_last_digit_of_2_pow_2010_l2426_242653

-- Define the function that gives the last digit of 2^n
def lastDigitOfPowerOfTwo (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 0 => 6
  | _ => 0  -- This case should never occur

-- Theorem statement
theorem last_digit_of_2_pow_2010 :
  lastDigitOfPowerOfTwo 2010 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_2_pow_2010_l2426_242653


namespace NUMINAMATH_CALUDE_max_sum_with_constraint_l2426_242699

theorem max_sum_with_constraint (a b c d e : ℕ) 
  (h : 625 * a + 250 * b + 100 * c + 40 * d + 16 * e = 15^3) :
  a + b + c + d + e ≤ 153 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_constraint_l2426_242699


namespace NUMINAMATH_CALUDE_point_coordinates_in_fourth_quadrant_l2426_242604

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the fourth quadrant
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

-- Define distance to x-axis
def distance_to_x_axis (p : Point2D) : ℝ :=
  |p.y|

-- Define distance to y-axis
def distance_to_y_axis (p : Point2D) : ℝ :=
  |p.x|

-- Theorem statement
theorem point_coordinates_in_fourth_quadrant (p : Point2D) 
  (h1 : in_fourth_quadrant p)
  (h2 : distance_to_x_axis p = 3)
  (h3 : distance_to_y_axis p = 8) :
  p = Point2D.mk 8 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_in_fourth_quadrant_l2426_242604


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l2426_242629

theorem algebraic_expression_equality (x y : ℝ) (h : x - 2*y + 8 = 18) :
  3*x - 6*y + 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l2426_242629


namespace NUMINAMATH_CALUDE_book_has_two_chapters_l2426_242607

/-- A book with chapters and pages -/
structure Book where
  total_pages : ℕ
  first_chapter_pages : ℕ
  second_chapter_pages : ℕ

/-- The number of chapters in a book -/
def num_chapters (b : Book) : ℕ :=
  if b.first_chapter_pages + b.second_chapter_pages = b.total_pages then 2 else 0

theorem book_has_two_chapters (b : Book) 
  (h1 : b.total_pages = 81) 
  (h2 : b.first_chapter_pages = 13) 
  (h3 : b.second_chapter_pages = 68) : 
  num_chapters b = 2 := by
  sorry

end NUMINAMATH_CALUDE_book_has_two_chapters_l2426_242607


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2426_242647

theorem unique_solution_for_equation : 
  ∃! (x : ℕ+), (1 : ℕ)^(x.val + 2) + 2^(x.val + 1) + 3^(x.val - 1) + 4^x.val = 1170 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2426_242647


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l2426_242657

structure Community where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

structure Survey where
  sample_size : Nat
  population_size : Nat

inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

def survey1 : Survey := {
  sample_size := 100,
  population_size := 125 + 280 + 95
}

def survey2 : Survey := {
  sample_size := 3,
  population_size := 12
}

def community : Community := {
  high_income := 125,
  middle_income := 280,
  low_income := 95
}

def optimal_sampling_method (s : Survey) (c : Option Community) : SamplingMethod :=
  sorry

theorem optimal_sampling_methods :
  optimal_sampling_method survey1 (some community) = SamplingMethod.Stratified ∧
  optimal_sampling_method survey2 none = SamplingMethod.SimpleRandom :=
sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_l2426_242657


namespace NUMINAMATH_CALUDE_total_sales_theorem_l2426_242638

/-- Calculate total sales from lettuce and tomatoes -/
def total_sales (customers : ℕ) (lettuce_per_customer : ℕ) (lettuce_price : ℚ) 
  (tomatoes_per_customer : ℕ) (tomato_price : ℚ) : ℚ :=
  (customers * lettuce_per_customer * lettuce_price) + 
  (customers * tomatoes_per_customer * tomato_price)

/-- Theorem: Total sales from lettuce and tomatoes is $2000 per month -/
theorem total_sales_theorem : 
  total_sales 500 2 1 4 (1/2) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_theorem_l2426_242638


namespace NUMINAMATH_CALUDE_c_work_time_l2426_242606

-- Define the work rates for each worker
def work_rate_a : ℚ := 1 / 36
def work_rate_b : ℚ := 1 / 18

-- Define the combined work rate
def combined_work_rate : ℚ := 1 / 4

-- Define the relationship between c and d's work rates
def d_work_rate (c : ℚ) : ℚ := c / 2

-- Theorem statement
theorem c_work_time :
  ∃ (c : ℚ), 
    work_rate_a + work_rate_b + c + d_work_rate c = combined_work_rate ∧
    c = 1 / 9 :=
by sorry

end NUMINAMATH_CALUDE_c_work_time_l2426_242606


namespace NUMINAMATH_CALUDE_BC_length_l2426_242685

-- Define the points and segments
variable (A B C D E : ℝ × ℝ)

-- Define the lengths of segments
def length (P Q : ℝ × ℝ) : ℝ := sorry

-- Define the conditions
def on_segment (P Q R : ℝ × ℝ) : Prop := sorry

axiom D_on_AE : on_segment A D E
axiom B_on_AD : on_segment A B D
axiom C_on_DE : on_segment D C E

axiom AB_length : length A B = 3 + 3 * length B D
axiom CE_length : length C E = 2 + 2 * length C D
axiom AE_length : length A E = 20

-- Theorem to prove
theorem BC_length : length B C = 4 := by sorry

end NUMINAMATH_CALUDE_BC_length_l2426_242685


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l2426_242678

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2015 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l2426_242678


namespace NUMINAMATH_CALUDE_only_cylinder_has_quadrilateral_cross_section_l2426_242637

-- Define the types of solids
inductive Solid
| Cone
| Cylinder
| Sphere

-- Define a function that determines if a solid can have a quadrilateral cross-section
def has_quadrilateral_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.Cone => False
  | Solid.Cylinder => True
  | Solid.Sphere => False

-- Theorem statement
theorem only_cylinder_has_quadrilateral_cross_section :
  ∀ s : Solid, has_quadrilateral_cross_section s ↔ s = Solid.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_only_cylinder_has_quadrilateral_cross_section_l2426_242637


namespace NUMINAMATH_CALUDE_sqrt_11_parts_sum_l2426_242616

theorem sqrt_11_parts_sum (x y : ℝ) : 
  (x = ⌊Real.sqrt 11⌋) → 
  (y = Real.sqrt 11 - x) → 
  (2 * x * y + y^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_11_parts_sum_l2426_242616


namespace NUMINAMATH_CALUDE_z_to_twelve_equals_one_l2426_242621

theorem z_to_twelve_equals_one :
  let z : ℂ := (Real.sqrt 3 - Complex.I) / 2
  z^12 = 1 := by sorry

end NUMINAMATH_CALUDE_z_to_twelve_equals_one_l2426_242621


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_m_l2426_242663

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is nonzero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_m (m : ℝ) : 
  is_pure_imaginary ((m^2 - 5*m + 6 : ℝ) + (m^2 - 3*m : ℝ) * I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_m_l2426_242663


namespace NUMINAMATH_CALUDE_product_congruence_l2426_242623

theorem product_congruence : ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (93 * 59 * 84) % 25 = m ∧ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_congruence_l2426_242623


namespace NUMINAMATH_CALUDE_translation_theorem_l2426_242630

-- Define the original function
def f (x : ℝ) : ℝ := (x - 2)^2 + 2

-- Define the translated function
def g (x : ℝ) : ℝ := (x - 1)^2 + 3

-- Theorem statement
theorem translation_theorem :
  ∀ x : ℝ, g x = f (x + 1) + 1 :=
sorry

end NUMINAMATH_CALUDE_translation_theorem_l2426_242630


namespace NUMINAMATH_CALUDE_circle_equation_l2426_242669

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 2)

-- Define the radius of the circle
def radius : ℝ := 4

-- State the theorem
theorem circle_equation :
  ∀ (x y : ℝ), ((x + 1)^2 + (y - 2)^2 = 16) ↔ 
  ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2426_242669


namespace NUMINAMATH_CALUDE_abie_spent_64_dollars_l2426_242624

def initial_bags : ℕ := 20
def original_price : ℚ := 2
def shared_fraction : ℚ := 2/5
def half_price_bags : ℕ := 18
def coupon_bags : ℕ := 4
def coupon_price_fraction : ℚ := 3/4

def total_spent : ℚ :=
  initial_bags * original_price +
  half_price_bags * (original_price / 2) +
  coupon_bags * (original_price * coupon_price_fraction)

theorem abie_spent_64_dollars : total_spent = 64 := by
  sorry

end NUMINAMATH_CALUDE_abie_spent_64_dollars_l2426_242624


namespace NUMINAMATH_CALUDE_total_problems_eq_480_l2426_242680

/-- The number of math problems Marvin solved yesterday -/
def marvin_yesterday : ℕ := 40

/-- The number of math problems Marvin solved today -/
def marvin_today : ℕ := 3 * marvin_yesterday

/-- The total number of math problems Marvin solved over two days -/
def marvin_total : ℕ := marvin_yesterday + marvin_today

/-- The number of math problems Arvin solved over two days -/
def arvin_total : ℕ := 2 * marvin_total

/-- The total number of math problems solved by both Marvin and Arvin -/
def total_problems : ℕ := marvin_total + arvin_total

theorem total_problems_eq_480 : total_problems = 480 := by sorry

end NUMINAMATH_CALUDE_total_problems_eq_480_l2426_242680


namespace NUMINAMATH_CALUDE_no_multiple_of_five_l2426_242688

theorem no_multiple_of_five (C : ℕ) : 
  (100 ≤ 100 + 10 * C + 4) ∧ (100 + 10 * C + 4 < 1000) ∧ (C < 10) →
  ¬(∃ k : ℕ, 100 + 10 * C + 4 = 5 * k) := by
sorry

end NUMINAMATH_CALUDE_no_multiple_of_five_l2426_242688


namespace NUMINAMATH_CALUDE_count_quadruples_l2426_242693

theorem count_quadruples : 
  let S := {q : Fin 10 × Fin 10 × Fin 10 × Fin 10 | true}
  Fintype.card S = 10000 := by sorry

end NUMINAMATH_CALUDE_count_quadruples_l2426_242693


namespace NUMINAMATH_CALUDE_first_load_pieces_l2426_242695

theorem first_load_pieces (total : ℕ) (equal_loads : ℕ) (pieces_per_load : ℕ)
  (h1 : total = 36)
  (h2 : equal_loads = 2)
  (h3 : pieces_per_load = 9)
  : total - (equal_loads * pieces_per_load) = 18 :=
by sorry

end NUMINAMATH_CALUDE_first_load_pieces_l2426_242695


namespace NUMINAMATH_CALUDE_two_lines_condition_l2426_242642

theorem two_lines_condition (m : ℝ) : 
  (∃ (a b c d : ℝ), ∀ (x y : ℝ), 
    (x^2 - m*y^2 + 2*x + 2*y = 0) ↔ ((a*x + b*y + c = 0) ∧ (a*x + b*y + d = 0))) 
  → m = 1 := by
sorry

end NUMINAMATH_CALUDE_two_lines_condition_l2426_242642


namespace NUMINAMATH_CALUDE_min_of_three_correct_specific_case_l2426_242668

def min_of_three (a b c : ℕ) : ℕ :=
  if b < a then
    if c < b then c else b
  else
    if c < a then c else a

theorem min_of_three_correct (a b c : ℕ) :
  min_of_three a b c = min a (min b c) := by sorry

theorem specific_case :
  min_of_three 3 6 2 = 2 := by sorry

end NUMINAMATH_CALUDE_min_of_three_correct_specific_case_l2426_242668


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2426_242633

theorem complex_fraction_simplification :
  (2 - I) / (2 + I) = 3/5 - 4/5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2426_242633


namespace NUMINAMATH_CALUDE_mary_has_29_nickels_l2426_242645

/-- Calculates the total number of nickels Mary has after receiving gifts and doing chores. -/
def marys_nickels (initial : ℕ) (from_dad : ℕ) (mom_multiplier : ℕ) (from_chores : ℕ) : ℕ :=
  initial + from_dad + (mom_multiplier * from_dad) + from_chores

/-- Theorem stating that Mary has 29 nickels after all transactions. -/
theorem mary_has_29_nickels : 
  marys_nickels 7 5 3 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_29_nickels_l2426_242645


namespace NUMINAMATH_CALUDE_sin_graph_shift_symmetry_l2426_242611

open Real

theorem sin_graph_shift_symmetry (φ : ℝ) :
  (∀ x, ∃ y, y = sin (2*x + φ)) →
  (abs φ < π) →
  (∀ x, ∃ y, y = sin (2*(x + π/6) + φ)) →
  (∀ x, sin (2*(x + π/6) + φ) = -sin (2*(-x + π/6) + φ)) →
  (φ = -π/3 ∨ φ = 2*π/3) := by
sorry

end NUMINAMATH_CALUDE_sin_graph_shift_symmetry_l2426_242611


namespace NUMINAMATH_CALUDE_coprime_divisibility_implies_one_l2426_242667

theorem coprime_divisibility_implies_one (a b c : ℕ+) :
  Nat.Coprime a.val b.val →
  Nat.Coprime a.val c.val →
  Nat.Coprime b.val c.val →
  a.val^2 ∣ (b.val^3 + c.val^3) →
  b.val^2 ∣ (a.val^3 + c.val^3) →
  c.val^2 ∣ (a.val^3 + b.val^3) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_coprime_divisibility_implies_one_l2426_242667


namespace NUMINAMATH_CALUDE_power_5_2048_mod_17_l2426_242682

theorem power_5_2048_mod_17 : 5^2048 % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_5_2048_mod_17_l2426_242682


namespace NUMINAMATH_CALUDE_polynomial_with_arithmetic_progression_roots_l2426_242691

/-- A polynomial of the form x^4 + mx^2 + nx + 144 with four distinct real roots in arithmetic progression has m = -40 -/
theorem polynomial_with_arithmetic_progression_roots (m n : ℝ) : 
  (∃ (b d : ℝ) (h_distinct : d ≠ 0), 
    (∀ x : ℝ, x^4 + m*x^2 + n*x + 144 = (x - b)*(x - (b + d))*(x - (b + 2*d))*(x - (b + 3*d))) ∧
    (b ≠ b + d) ∧ (b + d ≠ b + 2*d) ∧ (b + 2*d ≠ b + 3*d)) →
  m = -40 := by
sorry

end NUMINAMATH_CALUDE_polynomial_with_arithmetic_progression_roots_l2426_242691


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l2426_242675

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) :
  total_clips = 81 →
  num_boxes = 9 →
  total_clips = num_boxes * clips_per_box →
  clips_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l2426_242675


namespace NUMINAMATH_CALUDE_clean_room_together_l2426_242681

/-- The time it takes for Lisa and Kay to clean their room together -/
theorem clean_room_together (lisa_rate kay_rate : ℝ) (h1 : lisa_rate = 1 / 8) (h2 : kay_rate = 1 / 12) :
  1 / (lisa_rate + kay_rate) = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_clean_room_together_l2426_242681


namespace NUMINAMATH_CALUDE_reciprocal_sum_one_third_three_fourths_l2426_242689

theorem reciprocal_sum_one_third_three_fourths (x : ℚ) :
  x = (1/3 + 3/4)⁻¹ → x = 12/13 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_one_third_three_fourths_l2426_242689


namespace NUMINAMATH_CALUDE_p_oplus_q_equals_result_l2426_242664

def P : Set ℤ := {4, 5}
def Q : Set ℤ := {1, 2, 3}

def setDifference (P Q : Set ℤ) : Set ℤ :=
  {x | ∃ p ∈ P, ∃ q ∈ Q, x = p - q}

theorem p_oplus_q_equals_result : setDifference P Q = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_p_oplus_q_equals_result_l2426_242664


namespace NUMINAMATH_CALUDE_intersection_complement_problem_l2426_242683

open Set

theorem intersection_complement_problem :
  let U : Set ℝ := Set.univ
  let A : Set ℝ := {x | x > 0}
  let B : Set ℝ := {x | x > 1}
  A ∩ (U \ B) = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_problem_l2426_242683


namespace NUMINAMATH_CALUDE_optimal_production_solution_l2426_242644

/-- Represents the production problem with given parameters -/
structure ProductionProblem where
  total_units : ℕ
  workers : ℕ
  a_per_unit : ℕ
  b_per_unit : ℕ
  c_per_unit : ℕ
  a_per_worker : ℕ
  b_per_worker : ℕ
  c_per_worker : ℕ

/-- Calculates the completion time for a given worker distribution -/
def completion_time (prob : ProductionProblem) (x k : ℕ) : ℚ :=
  max (prob.a_per_unit * prob.total_units / (prob.a_per_worker * x : ℚ))
    (max (prob.b_per_unit * prob.total_units / (prob.b_per_worker * k * x : ℚ))
         (prob.c_per_unit * prob.total_units / (prob.c_per_worker * (prob.workers - (1 + k) * x) : ℚ)))

/-- The main theorem stating the optimal solution -/
theorem optimal_production_solution (prob : ProductionProblem) 
    (h_prob : prob.total_units = 3000 ∧ prob.workers = 200 ∧ 
              prob.a_per_unit = 2 ∧ prob.b_per_unit = 2 ∧ prob.c_per_unit = 1 ∧
              prob.a_per_worker = 6 ∧ prob.b_per_worker = 3 ∧ prob.c_per_worker = 2) :
    ∃ (x : ℕ), x > 0 ∧ x < prob.workers ∧ 
    completion_time prob x 2 = 250 / 11 ∧
    ∀ (y k : ℕ), y > 0 → y < prob.workers → k > 0 → 
    completion_time prob y k ≥ 250 / 11 := by
  sorry

end NUMINAMATH_CALUDE_optimal_production_solution_l2426_242644
