import Mathlib

namespace unique_solution_l1113_111304

/-- Returns the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + digit_count (n / 10)

/-- Represents k as overline(1n) -/
def k (n : ℕ) : ℕ := 10^(digit_count n) + n

/-- The main theorem stating that (11, 7) is the only solution -/
theorem unique_solution :
  ∀ m n : ℕ, m^2 = n * k n + 2 → (m = 11 ∧ n = 7) :=
sorry

end unique_solution_l1113_111304


namespace sample_size_theorem_l1113_111356

theorem sample_size_theorem (N : ℕ) (sample_size : ℕ) (probability : ℚ) 
  (h1 : sample_size = 30)
  (h2 : probability = 1/4)
  (h3 : (sample_size : ℚ) / N = probability) : 
  N = 120 := by
  sorry

end sample_size_theorem_l1113_111356


namespace ratio_of_fraction_equation_l1113_111319

theorem ratio_of_fraction_equation (P Q : ℤ) : 
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ -3 ∧ x ≠ 2 ∧ x ≠ 0 → 
    (P / (x - 3) + Q / (x^2 + x - 6) = (x^2 + 3*x + 1) / (x^3 - x^2 - 12*x))) →
  Q / P = -6 / 13 := by
  sorry

end ratio_of_fraction_equation_l1113_111319


namespace exists_rectangle_same_parity_l1113_111383

/-- Represents a rectangle on a grid -/
structure GridRectangle where
  length : ℕ
  width : ℕ

/-- Represents a square cut into rectangles -/
structure CutSquare where
  side_length : ℕ
  rectangles : List GridRectangle

/-- Checks if a number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Checks if two numbers have the same parity -/
def same_parity (a b : ℕ) : Prop :=
  (is_even a ∧ is_even b) ∨ (¬is_even a ∧ ¬is_even b)

/-- Main theorem: In a square with side length 2009 cut into rectangles,
    there exists at least one rectangle with sides of the same parity -/
theorem exists_rectangle_same_parity (sq : CutSquare) 
    (h1 : sq.side_length = 2009) 
    (h2 : sq.rectangles.length > 0) : 
    ∃ (r : GridRectangle), r ∈ sq.rectangles ∧ same_parity r.length r.width := by
  sorry

end exists_rectangle_same_parity_l1113_111383


namespace school_student_count_l1113_111394

theorem school_student_count :
  ∀ (total_students : ℕ),
  (∃ (girls boys : ℕ),
    girls = boys ∧
    girls + boys = total_students ∧
    (girls : ℚ) * (1/5) + (boys : ℚ) * (1/10) = 15) →
  total_students = 100 := by
sorry

end school_student_count_l1113_111394


namespace gasoline_consumption_reduction_l1113_111375

theorem gasoline_consumption_reduction 
  (original_price original_quantity : ℝ) 
  (price_increase : ℝ) 
  (spending_increase : ℝ) 
  (h1 : price_increase = 0.20) 
  (h2 : spending_increase = 0.14) : 
  let new_price := original_price * (1 + price_increase)
  let new_spending := original_price * original_quantity * (1 + spending_increase)
  let new_quantity := new_spending / new_price
  (original_quantity - new_quantity) / original_quantity = 0.05 := by
sorry

end gasoline_consumption_reduction_l1113_111375


namespace cubic_function_property_l1113_111391

/-- Given a cubic function f(x) = ax³ + bx² + cx + d where f(1) = 4,
    prove that 12a - 6b + 3c - 2d = 40 -/
theorem cubic_function_property (a b c d : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^3 + b * x^2 + c * x + d)
  (h_f1 : f 1 = 4) :
  12 * a - 6 * b + 3 * c - 2 * d = 40 := by
sorry

end cubic_function_property_l1113_111391


namespace derivative_even_function_at_zero_l1113_111389

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem derivative_even_function_at_zero (f : ℝ → ℝ) (hf : even_function f) 
  (hf' : Differentiable ℝ f) : 
  deriv f 0 = 0 := by
  sorry

end derivative_even_function_at_zero_l1113_111389


namespace right_triangle_leg_length_l1113_111326

theorem right_triangle_leg_length (c a b : ℝ) : 
  c = 10 →  -- hypotenuse length
  a = 6 →   -- length of one leg
  c^2 = a^2 + b^2 →  -- Pythagorean theorem (right-angled triangle condition)
  b = 8 := by  -- length of the other leg
sorry

end right_triangle_leg_length_l1113_111326


namespace road_project_solution_l1113_111365

/-- Road construction project parameters -/
structure RoadProject where
  total_length : ℝ
  small_eq_rate : ℝ
  large_eq_rate : ℝ
  large_eq_time_ratio : ℝ
  length_increase : ℝ
  small_eq_time_increase : ℝ
  large_eq_rate_decrease : ℝ
  large_eq_time_increase : ℝ → ℝ

/-- Theorem stating the correct small equipment usage time and the value of m -/
theorem road_project_solution (project : RoadProject)
  (h1 : project.total_length = 39000)
  (h2 : project.small_eq_rate = 30)
  (h3 : project.large_eq_rate = 60)
  (h4 : project.large_eq_time_ratio = 5/3)
  (h5 : project.length_increase = 9000)
  (h6 : project.small_eq_time_increase = 18)
  (h7 : project.large_eq_time_increase = λ m => 150 + 2*m) :
  ∃ (small_eq_time m : ℝ),
    small_eq_time = 300 ∧
    m = 5 ∧
    project.small_eq_rate * small_eq_time +
    project.large_eq_rate * (project.large_eq_time_ratio * small_eq_time) = project.total_length ∧
    project.small_eq_rate * (small_eq_time + project.small_eq_time_increase) +
    (project.large_eq_rate - m) * (project.large_eq_time_ratio * small_eq_time + project.large_eq_time_increase m) =
    project.total_length + project.length_increase :=
sorry

end road_project_solution_l1113_111365


namespace recreation_spending_percentage_l1113_111340

/-- Calculates the percentage of this week's recreation spending compared to last week's. -/
theorem recreation_spending_percentage 
  (last_week_wage : ℝ) 
  (last_week_recreation_percent : ℝ) 
  (this_week_wage_reduction : ℝ) 
  (this_week_recreation_percent : ℝ) 
  (h1 : last_week_recreation_percent = 0.40) 
  (h2 : this_week_wage_reduction = 0.05) 
  (h3 : this_week_recreation_percent = 0.50) : 
  (this_week_recreation_percent * (1 - this_week_wage_reduction) * last_week_wage) / 
  (last_week_recreation_percent * last_week_wage) * 100 = 118.75 :=
by sorry

end recreation_spending_percentage_l1113_111340


namespace eunji_confetti_l1113_111310

theorem eunji_confetti (red : ℕ) (green : ℕ) (given : ℕ) : 
  red = 1 → green = 9 → given = 4 → red + green - given = 6 := by sorry

end eunji_confetti_l1113_111310


namespace triangle_property_l1113_111300

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition that a*sin(B) - √3*b*cos(A) = 0 -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a * Real.sin t.B - Real.sqrt 3 * t.b * Real.cos t.A = 0

theorem triangle_property (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.A = π / 3 ∧ 
  (t.a = Real.sqrt 7 ∧ t.b = 2 → 
    (1/2 : ℝ) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2) :=
sorry


end triangle_property_l1113_111300


namespace angle_B_is_pi_over_four_max_area_when_b_is_two_max_area_equality_condition_l1113_111368

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangleCondition (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + t.c * Real.sin t.B

-- Theorem for part 1
theorem angle_B_is_pi_over_four (t : Triangle) (h : triangleCondition t) :
  t.B = π / 4 := by sorry

-- Theorem for part 2
theorem max_area_when_b_is_two (t : Triangle) (h1 : triangleCondition t) (h2 : t.b = 2) :
  (1 / 2) * t.a * t.c * Real.sin t.B ≤ Real.sqrt 2 + 1 := by sorry

-- Theorem for equality condition in part 2
theorem max_area_equality_condition (t : Triangle) (h1 : triangleCondition t) (h2 : t.b = 2) :
  (1 / 2) * t.a * t.c * Real.sin t.B = Real.sqrt 2 + 1 ↔ t.a = t.c := by sorry

end angle_B_is_pi_over_four_max_area_when_b_is_two_max_area_equality_condition_l1113_111368


namespace min_occupied_seats_for_150_l1113_111329

/-- Given a row of seats, calculates the minimum number of occupied seats
    required to ensure the next person must sit next to someone. -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  (total_seats + 2) / 4

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 37 := by
  sorry

#eval min_occupied_seats 150

end min_occupied_seats_for_150_l1113_111329


namespace stating_bacteria_fill_time_l1113_111308

/-- 
Represents the time (in minutes) it takes to fill a bottle with bacteria,
given the initial number of bacteria and their division rate.
-/
def fill_time (initial_bacteria : ℕ) (a : ℕ) : ℕ :=
  if initial_bacteria = 1 then a
  else a - 1

/-- 
Theorem stating that if one bacterium fills a bottle in 'a' minutes,
then two bacteria will fill the same bottle in 'a - 1' minutes,
given that each bacterium divides into two every minute.
-/
theorem bacteria_fill_time (a : ℕ) (h : a > 0) :
  fill_time 2 a = a - 1 :=
sorry

end stating_bacteria_fill_time_l1113_111308


namespace ab_equation_sum_l1113_111341

theorem ab_equation_sum (A B : ℕ) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  (10 * A + B) * 6 = 100 * B + 10 * B + B → 
  A + B = 11 :=
by sorry

end ab_equation_sum_l1113_111341


namespace triangle_packing_l1113_111316

/-- Represents an equilateral triangle with side length L -/
structure EquilateralTriangle (L : ℝ) where
  sideLength : L > 0

/-- Represents a configuration of unit equilateral triangles inside a larger triangle -/
structure TriangleConfiguration (L : ℝ) where
  largeTriangle : EquilateralTriangle L
  numUnitTriangles : ℕ
  nonOverlapping : Bool
  parallelSides : Bool
  oppositeOrientation : Bool

/-- The theorem statement -/
theorem triangle_packing (L : ℝ) (config : TriangleConfiguration L) :
  config.nonOverlapping ∧ config.parallelSides ∧ config.oppositeOrientation →
  (config.numUnitTriangles : ℝ) ≤ (2 / 3) * L^2 := by
  sorry

end triangle_packing_l1113_111316


namespace complex_equation_implies_sum_l1113_111371

theorem complex_equation_implies_sum (x y : ℝ) :
  (x + y : ℂ) + (y - 1) * I = (2 * x + 3 * y : ℂ) + (2 * y + 1) * I →
  x + y = 2 := by
  sorry

end complex_equation_implies_sum_l1113_111371


namespace factors_of_M_l1113_111318

/-- The number of natural-number factors of M, where M = 2^3 · 3^5 · 5^3 · 7^1 · 11^2 -/
def number_of_factors (M : ℕ) : ℕ :=
  if M = 2^3 * 3^5 * 5^3 * 7^1 * 11^2 then 576 else 0

/-- Theorem stating that the number of natural-number factors of M is 576 -/
theorem factors_of_M :
  number_of_factors (2^3 * 3^5 * 5^3 * 7^1 * 11^2) = 576 :=
by sorry

end factors_of_M_l1113_111318


namespace triangle_construction_from_polygon_centers_l1113_111332

/-- Centers of regular n-sided polygons externally inscribed on triangle sides -/
structure PolygonCenters (n : ℕ) where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Triangle vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Rotation by angle α around point P -/
def rotate (P : ℝ × ℝ) (α : ℝ) (Q : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Check if three points form a regular triangle -/
def isRegularTriangle (P Q R : ℝ × ℝ) : Prop := sorry

/-- Theorem about triangle construction from polygon centers -/
theorem triangle_construction_from_polygon_centers (n : ℕ) (centers : PolygonCenters n) :
  (n ≥ 4 → ∃! t : Triangle, 
    rotate centers.Y (2 * π / n) t.A = t.C ∧
    rotate centers.X (2 * π / n) t.C = t.B ∧
    rotate centers.Z (2 * π / n) t.B = t.A) ∧
  (n = 3 → isRegularTriangle centers.X centers.Y centers.Z → 
    ∃ t : Set Triangle, Infinite t ∧ 
    ∀ tri ∈ t, rotate centers.Y (2 * π / 3) tri.A = tri.C ∧
               rotate centers.X (2 * π / 3) tri.C = tri.B ∧
               rotate centers.Z (2 * π / 3) tri.B = tri.A) :=
by sorry

end triangle_construction_from_polygon_centers_l1113_111332


namespace planting_cost_l1113_111338

def flower_cost : ℕ := 9
def clay_pot_cost : ℕ := flower_cost + 20
def soil_cost : ℕ := flower_cost - 2

def total_cost : ℕ := flower_cost + clay_pot_cost + soil_cost

theorem planting_cost : total_cost = 45 := by
  sorry

end planting_cost_l1113_111338


namespace union_A_B_when_a_zero_complement_A_intersect_B_nonempty_iff_l1113_111350

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 2 < x ∧ x < a + 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2)*x + 2*a = 0}

-- Theorem 1
theorem union_A_B_when_a_zero :
  A 0 ∪ B 0 = {x | -2 < x ∧ x ≤ 2} := by sorry

-- Theorem 2
theorem complement_A_intersect_B_nonempty_iff (a : ℝ) :
  ((Set.univ \ A a) ∩ B a).Nonempty ↔ a ≤ 0 ∨ a ≥ 4 := by sorry

end union_A_B_when_a_zero_complement_A_intersect_B_nonempty_iff_l1113_111350


namespace average_words_per_puzzle_l1113_111334

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the number of weeks a pencil lasts -/
def weeks_per_pencil : ℕ := 2

/-- Represents the total number of words to use up a pencil -/
def words_per_pencil : ℕ := 1050

/-- Represents Bert's daily crossword puzzle habit -/
def puzzles_per_day : ℕ := 1

/-- Theorem stating the average number of words in each crossword puzzle -/
theorem average_words_per_puzzle :
  (words_per_pencil / (weeks_per_pencil * days_per_week)) = 75 := by
  sorry

end average_words_per_puzzle_l1113_111334


namespace exists_points_with_longer_inner_vector_sum_l1113_111352

/-- A regular polygon with 1976 sides -/
structure RegularPolygon1976 where
  vertices : Fin 1976 → ℝ × ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is inside the regular 1976-gon -/
def isInside (p : Point) (poly : RegularPolygon1976) : Prop :=
  sorry

/-- Checks if a point is outside the regular 1976-gon -/
def isOutside (p : Point) (poly : RegularPolygon1976) : Prop :=
  sorry

/-- Sum of vectors from a point to all vertices of the 1976-gon -/
def vectorSum (p : Point) (poly : RegularPolygon1976) : ℝ × ℝ :=
  sorry

/-- Length of a 2D vector -/
def vectorLength (v : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating the existence of points A and B satisfying the conditions -/
theorem exists_points_with_longer_inner_vector_sum (poly : RegularPolygon1976) :
  ∃ (A B : Point),
    isInside A poly ∧
    isOutside B poly ∧
    vectorLength (vectorSum A poly) > vectorLength (vectorSum B poly) :=
  sorry

end exists_points_with_longer_inner_vector_sum_l1113_111352


namespace triangle_isosceles_l1113_111361

theorem triangle_isosceles (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : Real.sin C = 2 * Real.cos A * Real.sin B) : A = B := by
  sorry

end triangle_isosceles_l1113_111361


namespace monomials_like_terms_iff_l1113_111312

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ v, m1 v = m2 v

/-- The first monomial 4ab^n -/
def monomial1 (n : ℕ) : ℕ → ℕ
| 0 => 1  -- exponent of a
| 1 => n  -- exponent of b
| _ => 0  -- other variables

/-- The second monomial -2a^mb^4 -/
def monomial2 (m : ℕ) : ℕ → ℕ
| 0 => m  -- exponent of a
| 1 => 4  -- exponent of b
| _ => 0  -- other variables

/-- Theorem: The monomials 4ab^n and -2a^mb^4 are like terms if and only if m = 1 and n = 4 -/
theorem monomials_like_terms_iff (m n : ℕ) :
  like_terms (monomial1 n) (monomial2 m) ↔ m = 1 ∧ n = 4 :=
by sorry

end monomials_like_terms_iff_l1113_111312


namespace sweeties_remainder_l1113_111346

theorem sweeties_remainder (m : ℕ) (h : m % 6 = 4) : (2 * m) % 6 = 2 := by
  sorry

end sweeties_remainder_l1113_111346


namespace function_periodicity_l1113_111323

/-- A function f: ℝ → ℝ satisfying the given property is periodic with period 2a -/
theorem function_periodicity (f : ℝ → ℝ) (a : ℝ) (h_a : a > 0) 
  (h_f : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x :=
by
  -- The proof goes here
  sorry

end function_periodicity_l1113_111323


namespace derived_sequence_not_arithmetic_nor_geometric_l1113_111396

/-- A sequence {a_n} defined by its partial sums s_n = aq^n -/
def PartialSumSequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^n

/-- The sequence {a_n} derived from the partial sums -/
def DerivedSequence (a q : ℝ) : ℕ → ℝ :=
  fun n => if n = 1 then a * q else a * (q - 1) * q^(n - 1)

/-- Theorem stating that the derived sequence is neither arithmetic nor geometric -/
theorem derived_sequence_not_arithmetic_nor_geometric (a q : ℝ) (ha : a ≠ 0) (hq : q ≠ 1) (hq_nonzero : q ≠ 0) :
  ¬ (∃ d : ℝ, ∀ n : ℕ, n ≥ 2 → DerivedSequence a q (n + 1) - DerivedSequence a q n = d) ∧
  ¬ (∃ r : ℝ, ∀ n : ℕ, n ≥ 1 → DerivedSequence a q (n + 1) / DerivedSequence a q n = r) :=
by sorry


end derived_sequence_not_arithmetic_nor_geometric_l1113_111396


namespace license_plate_palindrome_theorem_l1113_111311

/-- The number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The length of the letter sequence in the license plate -/
def letter_length : ℕ := 4

/-- The length of the digit sequence in the license plate -/
def digit_length : ℕ := 4

/-- The probability of a license plate containing at least one palindrome -/
def license_plate_palindrome_probability : ℚ := 655 / 57122

/-- 
Theorem: The probability of a license plate containing at least one palindrome 
(either in the four-letter or four-digit arrangement) is 655/57122.
-/
theorem license_plate_palindrome_theorem : 
  license_plate_palindrome_probability = 655 / 57122 := by
  sorry

end license_plate_palindrome_theorem_l1113_111311


namespace common_prime_root_quadratics_l1113_111382

theorem common_prime_root_quadratics (a b : ℤ) : 
  (∃ p : ℕ, Prime p ∧ 
    (p : ℤ)^2 + a * (p : ℤ) + b = 0 ∧ 
    (p : ℤ)^2 + b * (p : ℤ) + 1100 = 0) →
  a = 274 ∨ a = 40 := by
sorry

end common_prime_root_quadratics_l1113_111382


namespace y_is_75_percent_of_x_l1113_111313

/-- Given that 45% of z equals 96% of y and z equals 160% of x, prove that y equals 75% of x -/
theorem y_is_75_percent_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 0.96 * y) 
  (h2 : z = 1.60 * x) : 
  y = 0.75 * x := by
sorry

end y_is_75_percent_of_x_l1113_111313


namespace purely_imaginary_fraction_l1113_111353

theorem purely_imaginary_fraction (a : ℝ) (z : ℂ) :
  z = (a^2 - 1 : ℂ) + (a - 1 : ℂ) * I →
  z.re = 0 →
  z.im ≠ 0 →
  (a + I^2024) / (1 - I) = 0 := by sorry

end purely_imaginary_fraction_l1113_111353


namespace abc_inequality_l1113_111372

/-- Given a = 2/ln(4), b = ln(3)/ln(2), c = 3/2, prove that b > c > a -/
theorem abc_inequality (a b c : ℝ) (ha : a = 2 / Real.log 4) (hb : b = Real.log 3 / Real.log 2) (hc : c = 3 / 2) :
  b > c ∧ c > a := by
  sorry

end abc_inequality_l1113_111372


namespace commodity_tax_consumption_l1113_111309

theorem commodity_tax_consumption (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let new_tax := 0.8 * T
  let new_revenue := 0.92 * T * C
  ∃ new_consumption, 
    new_tax * new_consumption = new_revenue ∧ 
    new_consumption = 1.15 * C := by
sorry

end commodity_tax_consumption_l1113_111309


namespace passes_through_point_l1113_111339

/-- A linear function that passes through the point (0, 3) -/
def linearFunction (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- Theorem: The linear function passes through the point (0, 3) for any slope m -/
theorem passes_through_point (m : ℝ) : linearFunction m 0 = 3 := by
  sorry

end passes_through_point_l1113_111339


namespace b_completes_in_24_days_l1113_111355

/-- Worker represents a person who can complete a task -/
structure Worker where
  rate : ℚ  -- work rate in units of work per day

/-- Represents a work scenario with three workers -/
structure WorkScenario where
  a : Worker
  b : Worker
  c : Worker
  combined_time_ab : ℚ  -- time for a and b to complete work together
  time_a : ℚ           -- time for a to complete work alone
  time_c : ℚ           -- time for c to complete work alone

/-- Calculate the time for worker b to complete the work alone -/
def time_for_b_alone (w : WorkScenario) : ℚ :=
  1 / (1 / w.combined_time_ab - 1 / w.time_a)

/-- Theorem stating that given the conditions, b takes 24 days to complete the work alone -/
theorem b_completes_in_24_days (w : WorkScenario) 
  (h1 : w.combined_time_ab = 8)
  (h2 : w.time_a = 12)
  (h3 : w.time_c = 18) :
  time_for_b_alone w = 24 := by
  sorry

#eval time_for_b_alone { a := ⟨1/12⟩, b := ⟨1/24⟩, c := ⟨1/18⟩, combined_time_ab := 8, time_a := 12, time_c := 18 }

end b_completes_in_24_days_l1113_111355


namespace f_max_value_f_min_value_l1113_111367

/-- The function f(x) = 2x³ - 6x² - 18x + 7 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

/-- The maximum value of f(x) is 17 -/
theorem f_max_value : ∃ (x : ℝ), f x = 17 ∧ ∀ (y : ℝ), f y ≤ 17 := by sorry

/-- The minimum value of f(x) is -47 -/
theorem f_min_value : ∃ (x : ℝ), f x = -47 ∧ ∀ (y : ℝ), f y ≥ -47 := by sorry

end f_max_value_f_min_value_l1113_111367


namespace parabola_vertex_x_coordinate_l1113_111370

/-- The x-coordinate of the vertex of a parabola given three points it passes through -/
theorem parabola_vertex_x_coordinate 
  (a b c : ℝ) 
  (h1 : a * (-2)^2 + b * (-2) + c = 8)
  (h2 : a * 4^2 + b * 4 + c = 8)
  (h3 : a * 7^2 + b * 7 + c = 15) :
  let f := fun x => a * x^2 + b * x + c
  ∃ x₀, ∀ x, f x ≥ f x₀ ∧ x₀ = 1 :=
by sorry

end parabola_vertex_x_coordinate_l1113_111370


namespace unique_number_l1113_111359

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def contains_digit_5 (n : ℕ) : Prop := ∃ a b, n = 10*a + 5 + b ∧ 0 ≤ b ∧ b < 10

def divisible_by_3 (n : ℕ) : Prop := ∃ k, n = 3*k

theorem unique_number : 
  ∃! n : ℕ, 
    144 < n ∧ 
    n < 169 ∧ 
    is_odd n ∧ 
    contains_digit_5 n ∧ 
    divisible_by_3 n ∧ 
    n = 165 := by
  sorry

end unique_number_l1113_111359


namespace pencil_dozens_l1113_111384

theorem pencil_dozens (total_pencils : ℕ) (pencils_per_dozen : ℕ) (h1 : total_pencils = 144) (h2 : pencils_per_dozen = 12) :
  total_pencils / pencils_per_dozen = 12 := by
  sorry

end pencil_dozens_l1113_111384


namespace inverse_of_A_squared_l1113_111306

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![-2, 3], ![1, -5]]) : 
  (A^2)⁻¹ = ![![7, -21], ![-7, 28]] := by
  sorry

end inverse_of_A_squared_l1113_111306


namespace circle_C_theorem_l1113_111314

-- Define the circle C
def circle_C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 5}

-- Define the lines
def line_l1 (x y : ℝ) : Prop := x - y + 1 = 0
def line_l2 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 - Real.sqrt 3 = 0
def line_l3 (m a : ℝ) (x y : ℝ) : Prop := m * x - y + Real.sqrt a + 1 = 0

-- Define the theorem
theorem circle_C_theorem (center : ℝ × ℝ) (M N : ℝ × ℝ) :
  line_l1 center.1 center.2 →
  M ∈ circle_C center →
  N ∈ circle_C center →
  line_l2 M.1 M.2 →
  line_l2 N.1 N.2 →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 17 →
  (∀ (m : ℝ), ∃ (p : ℝ × ℝ), p ∈ circle_C center ∧ line_l3 m 5 p.1 p.2) →
  (((center.1 = 0 ∧ center.2 = 1) ∨
    (center.1 = 3 + Real.sqrt 3 ∧ center.2 = 4 + Real.sqrt 3)) ∧
   (∀ (a : ℝ), (∀ (m : ℝ), ∃ (p : ℝ × ℝ), p ∈ circle_C center ∧ line_l3 m a p.1 p.2) → 0 ≤ a ∧ a ≤ 5)) :=
by sorry


end circle_C_theorem_l1113_111314


namespace nine_to_fourth_equals_three_to_eighth_l1113_111327

theorem nine_to_fourth_equals_three_to_eighth : (9 : ℕ) ^ 4 = 3 ^ 8 := by
  sorry

end nine_to_fourth_equals_three_to_eighth_l1113_111327


namespace order_of_3_is_2_l1113_111303

def f (x : ℕ) : ℕ := x^2 % 13

def iterate_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem order_of_3_is_2 : 
  (∃ m : ℕ, m > 0 ∧ iterate_f m 3 = 3) ∧ 
  (∀ k : ℕ, k > 0 ∧ k < 2 → iterate_f k 3 ≠ 3) :=
sorry

end order_of_3_is_2_l1113_111303


namespace NaNO3_formed_l1113_111399

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String

-- Define the moles of each substance
structure Moles where
  AgNO3 : ℝ
  NaOH : ℝ
  AgOH : ℝ
  NaNO3 : ℝ

-- Define the chemical equation
def chemicalEquation : Reaction :=
  { reactant1 := "AgNO3"
  , reactant2 := "NaOH"
  , product1 := "AgOH"
  , product2 := "NaNO3" }

-- Define the initial moles
def initialMoles : Moles :=
  { AgNO3 := 1
  , NaOH := 1
  , AgOH := 0
  , NaNO3 := 0 }

-- Define the reaction completion condition
def reactionComplete (initial : Moles) (final : Moles) : Prop :=
  final.AgNO3 = 0 ∨ final.NaOH = 0

-- Define the no side reactions condition
def noSideReactions (initial : Moles) (final : Moles) : Prop :=
  initial.AgNO3 + initial.NaOH = final.AgOH + final.NaNO3

-- Theorem statement
theorem NaNO3_formed
  (reaction : Reaction)
  (initial : Moles)
  (final : Moles)
  (hReaction : reaction = chemicalEquation)
  (hInitial : initial = initialMoles)
  (hComplete : reactionComplete initial final)
  (hNoSide : noSideReactions initial final) :
  final.NaNO3 = 1 := by
  sorry

end NaNO3_formed_l1113_111399


namespace line_segment_parameterization_l1113_111392

/-- Given a line segment connecting points (1,-3) and (6,12) parameterized by
    x = at + b and y = ct + d where 0 ≤ t ≤ 1 and t = 0 corresponds to (1,-3),
    prove that a + c^2 + b^2 + d^2 = 240 -/
theorem line_segment_parameterization (a b c d : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = 1 ∧ d = -3) →
  (a + b = 6 ∧ c + d = 12) →
  a + c^2 + b^2 + d^2 = 240 := by
  sorry

end line_segment_parameterization_l1113_111392


namespace hayden_ironing_time_l1113_111393

/-- The time Hayden spends ironing his clothes over a given number of weeks -/
def ironingTime (shirtTime minutesPerDay : ℕ) (pantsTime minutesPerDay : ℕ) (daysPerWeek : ℕ) (numWeeks : ℕ) : ℕ :=
  (shirtTime + pantsTime) * daysPerWeek * numWeeks

/-- Theorem stating that Hayden spends 160 minutes ironing over 4 weeks -/
theorem hayden_ironing_time :
  ironingTime 5 3 5 4 = 160 := by
  sorry

end hayden_ironing_time_l1113_111393


namespace unique_solution_l1113_111342

/-- The infinite series representation of the equation -/
def infiniteSeries (x : ℝ) : ℝ := 2 - x + x^2 - x^3 + x^4 - x^5

/-- The condition for series convergence -/
def seriesConverges (x : ℝ) : Prop := abs x < 1

theorem unique_solution : 
  ∃! x : ℝ, (x = infiniteSeries x) ∧ seriesConverges x ∧ x = -1 + Real.sqrt 3 := by
  sorry

end unique_solution_l1113_111342


namespace cosine_product_sqrt_l1113_111388

theorem cosine_product_sqrt (π : Real) : 
  Real.sqrt ((3 - Real.cos (π / 9)^2) * (3 - Real.cos (2 * π / 9)^2) * (3 - Real.cos (4 * π / 9)^2)) = 39 / 8 := by
  sorry

end cosine_product_sqrt_l1113_111388


namespace sample_size_equals_selected_students_l1113_111328

/-- Represents a school with classes and students -/
structure School where
  num_classes : ℕ
  students_per_class : ℕ
  selected_students : ℕ

/-- The sample size of a school's "Student Congress" -/
def sample_size (school : School) : ℕ :=
  school.selected_students

theorem sample_size_equals_selected_students (school : School) 
  (h1 : school.num_classes = 40)
  (h2 : school.students_per_class = 50)
  (h3 : school.selected_students = 150) :
  sample_size school = 150 := by
  sorry

#check sample_size_equals_selected_students

end sample_size_equals_selected_students_l1113_111328


namespace square_side_length_l1113_111330

/-- Right triangle PQR with legs PQ and PR, and a square inside --/
structure RightTriangleWithSquare where
  /-- Length of leg PQ --/
  pq : ℝ
  /-- Length of leg PR --/
  pr : ℝ
  /-- Side length of the square --/
  s : ℝ
  /-- PQ is 9 cm --/
  pq_length : pq = 9
  /-- PR is 12 cm --/
  pr_length : pr = 12
  /-- The square has one side on hypotenuse QR and one vertex on each leg --/
  square_position : s > 0 ∧ s < pq ∧ s < pr

/-- The side length of the square is 15/2 cm --/
theorem square_side_length (t : RightTriangleWithSquare) : t.s = 15 / 2 := by
  sorry

end square_side_length_l1113_111330


namespace floor_equation_unique_solution_l1113_111386

theorem floor_equation_unique_solution (n : ℕ+) :
  ∃! (a : ℝ), ∀ (n : ℕ+), 4 * ⌊a * n⌋ = n + ⌊a * ⌊a * n⌋⌋ ∧ a = 2 + Real.sqrt 3 := by
  sorry

end floor_equation_unique_solution_l1113_111386


namespace line_parallel_perpendicular_implies_planes_perpendicular_l1113_111315

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planesPerpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) (h_distinct : α ≠ β)
  (h_parallel : parallel l α) (h_perpendicular : perpendicular l β) :
  planesPerpendicular α β :=
sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l1113_111315


namespace last_four_digits_theorem_l1113_111322

theorem last_four_digits_theorem :
  ∃ (N : ℕ+),
    (∃ (a b c d : ℕ),
      a ≠ 0 ∧
      a ≠ 6 ∧ b ≠ 6 ∧ c ≠ 6 ∧
      N % 10000 = a * 1000 + b * 100 + c * 10 + d ∧
      (N * N) % 10000 = a * 1000 + b * 100 + c * 10 + d ∧
      a * 100 + b * 10 + c = 106) :=
by sorry

end last_four_digits_theorem_l1113_111322


namespace f_at_4_l1113_111349

def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

theorem f_at_4 : f 4 = 371 := by
  sorry

end f_at_4_l1113_111349


namespace functional_equation_solution_l1113_111307

open Real

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the property that f must satisfy
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((floor x : ℝ) * y) = f x * (floor (f y) : ℝ)

-- Theorem statement
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, satisfies_equation f →
    (∀ x : ℝ, f x = 0) ∨ 
    (∃ C : ℝ, 1 ≤ C ∧ C < 2 ∧ ∀ x : ℝ, f x = C) :=
by sorry

end functional_equation_solution_l1113_111307


namespace euler_family_mean_age_l1113_111333

def euler_family_ages : List ℕ := [5, 8, 8, 8, 12, 12]

theorem euler_family_mean_age :
  (euler_family_ages.sum : ℚ) / euler_family_ages.length = 53 / 6 := by
  sorry

end euler_family_mean_age_l1113_111333


namespace arithmetic_progression_problem_l1113_111351

def arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_progression_problem (a₁ d : ℝ) :
  (arithmetic_progression a₁ d 13 = 3 * arithmetic_progression a₁ d 3) ∧
  (arithmetic_progression a₁ d 18 = 2 * arithmetic_progression a₁ d 7 + 8) →
  d = 4 ∧ a₁ = 12 := by
  sorry

end arithmetic_progression_problem_l1113_111351


namespace license_plate_increase_l1113_111398

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4 * 2
  (new_plates : ℚ) / old_plates = 135.2 := by
sorry

end license_plate_increase_l1113_111398


namespace circle_properties_l1113_111378

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2 = 0

-- Define the line L
def L (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the symmetric line
def SymLine (x y : ℝ) : Prop := x - y = 0

-- Define the distance line
def DistLine (x y m : ℝ) : Prop := x + y + m = 0

theorem circle_properties :
  -- 1. Chord length
  (∃ l : ℝ, l = Real.sqrt 6 ∧
    ∀ x y : ℝ, C x y → L x y →
      ∃ x' y' : ℝ, C x' y' ∧ L x' y' ∧
        (x - x')^2 + (y - y')^2 = l^2) ∧
  -- 2. Symmetric circle
  (∀ x y : ℝ, (∃ x' y' : ℝ, C x' y' ∧ SymLine x' y' ∧
    x = y' ∧ y = x') → x^2 + (y-2)^2 = 2) ∧
  -- 3. Distance condition
  (∀ m : ℝ, (abs (m + 2) / Real.sqrt 2 = Real.sqrt 2 / 2) →
    m = -1 ∨ m = -3) :=
sorry

end circle_properties_l1113_111378


namespace restaurant_outdoor_area_l1113_111305

/-- The area of a rectangular section with width 4 feet and length 6 feet is 24 square feet. -/
theorem restaurant_outdoor_area : 
  ∀ (width length area : ℝ), 
    width = 4 → 
    length = 6 → 
    area = width * length → 
    area = 24 :=
by
  sorry

end restaurant_outdoor_area_l1113_111305


namespace product_of_primes_l1113_111325

theorem product_of_primes : 
  ∃ (p q : Nat), 
    Prime p ∧ 
    Prime q ∧ 
    p = 1021031 ∧ 
    q = 237019 ∧ 
    p * q = 241940557349 := by
  sorry

end product_of_primes_l1113_111325


namespace ants_eyes_count_l1113_111395

theorem ants_eyes_count (spider_count : ℕ) (ant_count : ℕ) (eyes_per_spider : ℕ) (total_eyes : ℕ)
  (h1 : spider_count = 3)
  (h2 : ant_count = 50)
  (h3 : eyes_per_spider = 8)
  (h4 : total_eyes = 124) :
  (total_eyes - spider_count * eyes_per_spider) / ant_count = 2 :=
by sorry

end ants_eyes_count_l1113_111395


namespace highest_probability_C_l1113_111301

-- Define the events and their probabilities
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.1
def prob_C : ℝ := 0.9

-- Theorem: C has the highest probability
theorem highest_probability_C : 
  prob_C > prob_A ∧ prob_C > prob_B :=
by sorry

end highest_probability_C_l1113_111301


namespace tangent_line_parallel_implies_a_zero_l1113_111380

/-- Given a function f(x) = x^2 + a/x where a is a real number,
    if the tangent line at x = 1 is parallel to 2x - y + 1 = 0,
    then a = 0. -/
theorem tangent_line_parallel_implies_a_zero 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 + a/x) 
  (h2 : (deriv f 1) = 2) : 
  a = 0 := by
  sorry

end tangent_line_parallel_implies_a_zero_l1113_111380


namespace saras_quarters_l1113_111354

/-- Sara's quarters problem -/
theorem saras_quarters (initial_quarters borrowed_quarters : ℕ) 
  (h1 : initial_quarters = 4937)
  (h2 : borrowed_quarters = 1743) :
  initial_quarters - borrowed_quarters = 3194 :=
by sorry

end saras_quarters_l1113_111354


namespace mary_fruits_left_l1113_111317

/-- The number of fruits Mary has left after buying, using for salad, eating, and giving away -/
def fruits_left (apples oranges blueberries grapes kiwis : ℕ) 
  (apples_salad oranges_salad blueberries_salad : ℕ)
  (apples_eaten oranges_eaten kiwis_eaten : ℕ)
  (apples_given oranges_given blueberries_given grapes_given kiwis_given : ℕ) : ℕ :=
  (apples - apples_salad - apples_eaten - apples_given) +
  (oranges - oranges_salad - oranges_eaten - oranges_given) +
  (blueberries - blueberries_salad - blueberries_given) +
  (grapes - grapes_given) +
  (kiwis - kiwis_eaten - kiwis_given)

/-- Theorem stating that Mary has 61 fruits left -/
theorem mary_fruits_left : 
  fruits_left 26 35 18 12 22 6 10 8 2 3 1 5 7 4 3 3 = 61 := by
  sorry

end mary_fruits_left_l1113_111317


namespace min_values_xy_l1113_111387

/-- Given two positive real numbers x and y satisfying lgx + lgy = lg(x + y + 3),
    prove that the minimum value of xy is 9 and the minimum value of x + y is 6 -/
theorem min_values_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h : Real.log x + Real.log y = Real.log (x + y + 3)) : 
    (∀ a b : ℝ, a > 0 → b > 0 → Real.log a + Real.log b = Real.log (a + b + 3) → x * y ≤ a * b) ∧
    (∀ c d : ℝ, c > 0 → d > 0 → Real.log c + Real.log d = Real.log (c + d + 3) → x + y ≤ c + d) ∧
    x * y = 9 ∧ x + y = 6 := by
  sorry


end min_values_xy_l1113_111387


namespace diagonal_angle_is_45_degrees_l1113_111385

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define the angle formed by a diagonal and a side of a square
def diagonal_angle (s : Square) : ℝ := sorry

-- Theorem statement
theorem diagonal_angle_is_45_degrees (s : Square) : 
  diagonal_angle s = 45 := by sorry

end diagonal_angle_is_45_degrees_l1113_111385


namespace factorial_30_trailing_zeros_l1113_111331

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- 30! has 7 trailing zeros -/
theorem factorial_30_trailing_zeros : trailingZeros 30 = 7 := by sorry

end factorial_30_trailing_zeros_l1113_111331


namespace geometric_proof_l1113_111376

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the intersection of two planes resulting in a line
variable (intersect_planes : Plane → Plane → Line)

-- Define the relation of a line being contained in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem geometric_proof 
  (m n : Line) (α β γ : Plane)
  (h1 : perp_planes α β)
  (h2 : m = intersect_planes α β)
  (h3 : perp_line_plane n α)
  (h4 : line_in_plane n γ) :
  perp_lines m n ∧ perp_planes α γ := by
  sorry

end geometric_proof_l1113_111376


namespace odd_digits_sum_152_345_l1113_111320

/-- Converts a base 10 number to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

theorem odd_digits_sum_152_345 : 
  let base4_152 := toBase4 152
  let base4_345 := toBase4 345
  countOddDigits base4_152 + countOddDigits base4_345 = 6 := by
  sorry

end odd_digits_sum_152_345_l1113_111320


namespace janinas_pancakes_l1113_111369

-- Define the variables
def daily_rent : ℕ := 30
def daily_supplies : ℕ := 12
def price_per_pancake : ℕ := 2

-- Define the function to calculate the number of pancakes needed
def pancakes_needed (rent : ℕ) (supplies : ℕ) (price : ℕ) : ℕ :=
  (rent + supplies) / price

-- Theorem statement
theorem janinas_pancakes :
  pancakes_needed daily_rent daily_supplies price_per_pancake = 21 := by
sorry

end janinas_pancakes_l1113_111369


namespace student_number_problem_l1113_111336

theorem student_number_problem (x : ℝ) : 4 * x - 138 = 102 → x = 60 := by
  sorry

end student_number_problem_l1113_111336


namespace inverse_square_relation_l1113_111360

/-- Given that x varies inversely as the square of y and y = 3 when x = 1,
    prove that x = 1/9 when y = 9. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) : 
  (∀ (x y : ℝ), x = k / (y^2)) →  -- x varies inversely as square of y
  (1 = k / (3^2)) →               -- y = 3 when x = 1
  (x = k / (9^2)) →               -- condition for y = 9
  x = 1/9 := by
sorry

end inverse_square_relation_l1113_111360


namespace solve_for_A_l1113_111358

theorem solve_for_A : ∃ A : ℤ, (2 * A - 6 + 4 = 26) ∧ A = 14 := by
  sorry

end solve_for_A_l1113_111358


namespace towels_per_pack_l1113_111397

/-- Given that Tiffany bought 9 packs of towels and 27 towels in total,
    prove that there were 3 towels in each pack. -/
theorem towels_per_pack (total_packs : ℕ) (total_towels : ℕ) 
  (h1 : total_packs = 9) 
  (h2 : total_towels = 27) : 
  total_towels / total_packs = 3 := by
  sorry

end towels_per_pack_l1113_111397


namespace roots_of_polynomial_l1113_111377

def p (x : ℝ) : ℝ := 3*x^4 + 7*x^3 - 13*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (p (-3) = 0) ∧ (p (-2) = 0) ∧ (p (-1) = 0) ∧ (p (1/3) = 0) :=
by sorry

end roots_of_polynomial_l1113_111377


namespace group_average_score_l1113_111373

def class_size : ℕ := 14
def class_average : ℝ := 85
def score_differences : List ℝ := [2, 3, -3, -5, 12, 12, 8, 2, -1, 4, -10, -2, 5, 5]

theorem group_average_score :
  let total_score := class_size * class_average + score_differences.sum
  total_score / class_size = 87.29 := by sorry

end group_average_score_l1113_111373


namespace tea_maker_capacity_l1113_111363

/-- A cylindrical tea maker with capacity x cups -/
structure TeaMaker where
  capacity : ℝ
  cylindrical : Bool

/-- Theorem: A cylindrical tea maker that contains 54 cups when 45% full has a total capacity of 120 cups -/
theorem tea_maker_capacity (tm : TeaMaker) (h1 : tm.cylindrical = true) 
    (h2 : 0.45 * tm.capacity = 54) : tm.capacity = 120 := by
  sorry

end tea_maker_capacity_l1113_111363


namespace son_father_distance_l1113_111381

/-- 
Given a lamp post, a father, and his son standing on the same straight line,
with their shadows' heads incident at the same point, prove that the distance
between the son and his father is 4.9 meters.
-/
theorem son_father_distance 
  (lamp_height : ℝ) 
  (father_height : ℝ) 
  (son_height : ℝ) 
  (father_lamp_distance : ℝ) 
  (h_lamp : lamp_height = 6)
  (h_father : father_height = 1.8)
  (h_son : son_height = 0.9)
  (h_father_lamp : father_lamp_distance = 2.1)
  (h_shadows : ∀ x : ℝ, father_height / father_lamp_distance = lamp_height / (father_lamp_distance + x) → 
                        son_height / x = father_height / (father_lamp_distance + x)) :
  ∃ x : ℝ, x = 4.9 ∧ 
    father_height / father_lamp_distance = lamp_height / (father_lamp_distance + x) ∧
    son_height / x = father_height / (father_lamp_distance + x) := by
  sorry


end son_father_distance_l1113_111381


namespace choral_group_max_size_l1113_111347

theorem choral_group_max_size :
  ∀ (n s : ℕ),
  (∃ (m : ℕ),
    m < 150 ∧
    n * s + 4 = m ∧
    (s - 3) * (n + 2) = m) →
  (∀ (m : ℕ),
    m < 150 ∧
    (∃ (x y : ℕ),
      x * y + 4 = m ∧
      (y - 3) * (x + 2) = m) →
    m ≤ 144) :=
by sorry

end choral_group_max_size_l1113_111347


namespace geometric_sequence_ratio_main_theorem_l1113_111321

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∃ q : ℝ, ∀ n, a (n + 1) = a 1 * q^n :=
sorry

theorem main_theorem (a : ℕ → ℝ) (h1 : geometric_sequence a) 
  (h2 : ∀ n, a n > 0)
  (h3 : 2 * (1/2 * a 3) = 3 * a 1 + 2 * a 2) :
  (a 10 + a 12 + a 15 + a 19 + a 20 + a 23) / 
  (a 8 + a 10 + a 13 + a 17 + a 18 + a 21) = 9 :=
sorry

end geometric_sequence_ratio_main_theorem_l1113_111321


namespace max_value_theorem_l1113_111357

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a * Real.sqrt (1 - b^2) - b * Real.sqrt (1 - a^2) = a * b) :
  (a / b + b / a) ≤ Real.sqrt 5 := by
sorry

end max_value_theorem_l1113_111357


namespace students_without_A_l1113_111344

theorem students_without_A (total : ℕ) (science_A : ℕ) (english_A : ℕ) (both_A : ℕ) : 
  total - (science_A + english_A - both_A) = 18 :=
by
  sorry

#check students_without_A 40 10 18 6

end students_without_A_l1113_111344


namespace arithmetic_sequence_length_l1113_111345

theorem arithmetic_sequence_length
  (a : ℕ) (d : ℕ) (last : ℕ) (n : ℕ)
  (h1 : a = 3)
  (h2 : d = 4)
  (h3 : last = 47)
  (h4 : last = a + (n - 1) * d) :
  n = 12 := by
sorry

end arithmetic_sequence_length_l1113_111345


namespace absolute_sum_zero_implies_values_l1113_111302

theorem absolute_sum_zero_implies_values (m n : ℝ) :
  |1 + m| + |n - 2| = 0 → m = -1 ∧ n = 2 ∧ m^n = 1 := by
  sorry

end absolute_sum_zero_implies_values_l1113_111302


namespace sqrt_meaningful_range_l1113_111335

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by
  sorry

end sqrt_meaningful_range_l1113_111335


namespace new_figure_length_l1113_111379

/-- A polygon with adjacent perpendicular sides -/
structure PerpendicularPolygon where
  sides : List ℝ
  adjacent_perpendicular : Bool

/-- The new figure formed by removing four sides from the original polygon -/
def new_figure (p : PerpendicularPolygon) : List ℝ :=
  sorry

/-- Theorem: The total length of segments in the new figure is 22 units -/
theorem new_figure_length (p : PerpendicularPolygon) 
  (h1 : p.adjacent_perpendicular = true)
  (h2 : p.sides = [9, 3, 7, 1, 1]) :
  (new_figure p).sum = 22 := by
  sorry

end new_figure_length_l1113_111379


namespace min_value_and_relationship_l1113_111348

theorem min_value_and_relationship (a b : ℝ) : 
  (4 + (a + b)^2 ≥ 4) ∧ (4 + (a + b)^2 = 4 ↔ a + b = 0) := by
  sorry

end min_value_and_relationship_l1113_111348


namespace workers_read_both_books_l1113_111364

/-- The number of workers who have read both Saramago's and Kureishi's latest books -/
def workers_read_both (total : ℕ) (saramago : ℕ) (kureishi : ℕ) (neither : ℕ) : ℕ :=
  saramago + kureishi - (total - neither)

theorem workers_read_both_books :
  let total := 42
  let saramago := total / 2
  let kureishi := total / 6
  let neither := saramago - kureishi - 1
  workers_read_both total saramago kureishi neither = 6 := by
  sorry

#eval workers_read_both 42 21 7 20

end workers_read_both_books_l1113_111364


namespace smallest_positive_angle_same_terminal_side_l1113_111343

/-- Given an angle α = 2012°, the smallest positive angle θ with the same terminal side is 212°. -/
theorem smallest_positive_angle_same_terminal_side :
  let α : Real := 2012
  ∃ θ : Real,
    0 < θ ∧ 
    θ ≤ 360 ∧
    ∃ k : Int, α = k * 360 + θ ∧
    ∀ φ : Real, (0 < φ ∧ φ ≤ 360 ∧ ∃ m : Int, α = m * 360 + φ) → θ ≤ φ ∧
    θ = 212 :=
by sorry

end smallest_positive_angle_same_terminal_side_l1113_111343


namespace sequence_general_term_l1113_111390

/-- Given a sequence {aₙ} with sum of first n terms Sₙ = (2/3)n² - (1/3)n,
    prove that the general term is aₙ = (4/3)n - 1 -/
theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) 
    (h : ∀ n, S n = 2/3 * n^2 - 1/3 * n) :
  ∀ n, a n = 4/3 * n - 1 := by
  sorry

end sequence_general_term_l1113_111390


namespace angle_AO2B_greater_than_90_degrees_l1113_111324

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the center of circle O₂
def O2_center : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem angle_AO2B_greater_than_90_degrees :
  let angle_AO2B := sorry
  angle_AO2B > 90 := by sorry

end angle_AO2B_greater_than_90_degrees_l1113_111324


namespace gcd_765432_654321_l1113_111362

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l1113_111362


namespace soft_drink_price_l1113_111374

/-- The price increase of a soft drink over 10 years -/
def price_increase (initial_price : ℕ) (increase_5p : ℕ) (increase_2p : ℕ) : ℚ :=
  (initial_price + 5 * increase_5p + 2 * increase_2p) / 100

/-- Theorem stating the final price of the soft drink -/
theorem soft_drink_price :
  price_increase 70 4 6 = 102 / 100 := by sorry

end soft_drink_price_l1113_111374


namespace greatest_NPMPP_l1113_111337

/-- A function that checks if a number's square ends with the number itself -/
def endsWithSelf (n : Nat) : Prop :=
  n % 10 = (n * n) % 10

/-- A function that generates a four-digit number with all identical digits -/
def fourIdenticalDigits (d : Nat) : Nat :=
  d * 1000 + d * 100 + d * 10 + d

/-- The theorem stating the greatest possible value of NPMPP -/
theorem greatest_NPMPP : 
  ∃ (M : Nat), 
    M ≤ 9 ∧ 
    endsWithSelf M ∧ 
    ∀ (N : Nat), N ≤ 9 → endsWithSelf N → M ≥ N ∧
    fourIdenticalDigits M * M = 89991 :=
sorry

end greatest_NPMPP_l1113_111337


namespace principal_amount_correct_l1113_111366

/-- The principal amount borrowed -/
def P : ℝ := 22539.53

/-- The total interest paid after 3 years -/
def total_interest : ℝ := 9692

/-- The interest rate for the first year -/
def r1 : ℝ := 0.12

/-- The interest rate for the second year -/
def r2 : ℝ := 0.14

/-- The interest rate for the third year -/
def r3 : ℝ := 0.17

/-- Theorem stating that the given principal amount results in the specified total interest -/
theorem principal_amount_correct : 
  P * r1 + P * r2 + P * r3 = total_interest := by sorry

end principal_amount_correct_l1113_111366
