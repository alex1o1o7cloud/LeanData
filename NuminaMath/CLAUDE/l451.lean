import Mathlib

namespace NUMINAMATH_CALUDE_total_apples_to_pack_l451_45130

/-- The number of apples in one dozen -/
def apples_per_dozen : ℕ := 12

/-- The number of boxes needed -/
def boxes_needed : ℕ := 90

/-- Theorem stating the total number of apples to be packed -/
theorem total_apples_to_pack : apples_per_dozen * boxes_needed = 1080 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_to_pack_l451_45130


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l451_45196

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 3*x - 4 = 0 ↔ x = x₁ ∨ x = x₂) → x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l451_45196


namespace NUMINAMATH_CALUDE_current_rate_calculation_l451_45191

theorem current_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) :
  boat_speed = 21 →
  distance = 6.283333333333333 →
  time = 13 / 60 →
  ∃ current_rate : ℝ, 
    distance = (boat_speed + current_rate) * time ∧
    current_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l451_45191


namespace NUMINAMATH_CALUDE_books_sold_correct_l451_45107

/-- The number of books Paul sold in a garage sale -/
def books_sold : ℕ := 94

/-- The initial number of books Paul had -/
def initial_books : ℕ := 2

/-- The number of new books Paul bought -/
def new_books : ℕ := 150

/-- The final number of books Paul has -/
def final_books : ℕ := 58

/-- Theorem stating that the number of books Paul sold is correct -/
theorem books_sold_correct : 
  initial_books - books_sold + new_books = final_books :=
by sorry

end NUMINAMATH_CALUDE_books_sold_correct_l451_45107


namespace NUMINAMATH_CALUDE_combination_permutation_relation_combination_symmetry_pascal_identity_permutation_recursive_l451_45164

-- Define C_n_m as the number of combinations of n items taken m at a time
def C (n m : ℕ) : ℕ := Nat.choose n m

-- Define A_n_m as the number of permutations of n items taken m at a time
def A (n m : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - m)

theorem combination_permutation_relation (n m : ℕ) (h : m ≤ n) :
  C n m = A n m / Nat.factorial m := by sorry

theorem combination_symmetry (n m : ℕ) (h : m ≤ n) :
  C n m = C n (n - m) := by sorry

theorem pascal_identity (n r : ℕ) (h : r ≤ n) :
  C (n + 1) r = C n r + C n (r - 1) := by sorry

theorem permutation_recursive (n m : ℕ) (h : m ≤ n) :
  A (n + 2) (m + 2) = (n + 2) * (n + 1) * A n m := by sorry

end NUMINAMATH_CALUDE_combination_permutation_relation_combination_symmetry_pascal_identity_permutation_recursive_l451_45164


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l451_45149

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 18) :
  let r := 9 - 9 * Real.sqrt 2 / 2
  (4 / 3 : ℝ) * Real.pi * r^3 = (4 / 3 : ℝ) * Real.pi * (9 - 9 * Real.sqrt 2 / 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l451_45149


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l451_45147

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the equation
def equation (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the theorem
theorem arithmetic_sequence_properties
  (a : ℝ) (d : ℝ)
  (h1 : equation a 1 = 0)
  (h2 : equation a d = 0) :
  ∃ (S_n : ℕ → ℝ) (T_n : ℕ → ℝ),
    (∀ n : ℕ, arithmetic_sequence a d n = n + 1) ∧
    (∀ n : ℕ, S_n n = (n^2 + 3*n) / 2) ∧
    (∀ n : ℕ, T_n n = 1 + (n - 1) * 3^n + (3^n - 1) / 2) :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l451_45147


namespace NUMINAMATH_CALUDE_hannah_reading_finish_day_l451_45100

def days_to_read (n : ℕ) : ℕ := n * (n + 1) / 2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem hannah_reading_finish_day (start_day : ℕ) (num_books : ℕ) :
  start_day = 5 →  -- Friday is represented as 5 (0 = Sunday, 1 = Monday, etc.)
  num_books = 20 →
  day_of_week start_day (days_to_read num_books) = start_day :=
by sorry

end NUMINAMATH_CALUDE_hannah_reading_finish_day_l451_45100


namespace NUMINAMATH_CALUDE_cube_side_length_is_one_l451_45174

/-- The side length of a cube -/
def m : ℕ := sorry

/-- The number of blue faces on the unit cubes -/
def blue_faces : ℕ := 2 * m^2

/-- The total number of faces on all unit cubes -/
def total_faces : ℕ := 6 * m^3

/-- The theorem stating that if one-third of the total faces are blue, then m = 1 -/
theorem cube_side_length_is_one : 
  (blue_faces : ℚ) / total_faces = 1 / 3 → m = 1 := by sorry

end NUMINAMATH_CALUDE_cube_side_length_is_one_l451_45174


namespace NUMINAMATH_CALUDE_polygonal_chains_10_9_l451_45189

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The number of sides in each polygonal chain -/
def sides : ℕ := 9

/-- A function that calculates the number of non-closed, non-self-intersecting 
    polygonal chains with 'sides' sides that can be formed from 'n' points on a circle -/
def polygonal_chains (n : ℕ) (sides : ℕ) : ℕ :=
  if sides ≤ n ∧ sides > 2 then
    (n * 2^(sides - 1)) / 2
  else
    0

/-- Theorem stating that the number of non-closed, non-self-intersecting 9-sided 
    polygonal chains that can be formed with 10 points on a circle as vertices is 1280 -/
theorem polygonal_chains_10_9 : polygonal_chains n sides = 1280 := by
  sorry

end NUMINAMATH_CALUDE_polygonal_chains_10_9_l451_45189


namespace NUMINAMATH_CALUDE_karthik_weight_average_l451_45110

def karthik_weight_range (w : ℝ) : Prop :=
  55 < w ∧ w < 62 ∧ 50 < w ∧ w < 60 ∧ w < 58

theorem karthik_weight_average :
  ∃ (min max : ℝ),
    (∀ w, karthik_weight_range w → min ≤ w ∧ w ≤ max) ∧
    (∃ w₁ w₂, karthik_weight_range w₁ ∧ karthik_weight_range w₂ ∧ w₁ = min ∧ w₂ = max) ∧
    (min + max) / 2 = 56.5 :=
sorry

end NUMINAMATH_CALUDE_karthik_weight_average_l451_45110


namespace NUMINAMATH_CALUDE_wrong_number_calculation_l451_45166

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg correct_num wrong_num : ℚ) : 
  n = 10 →
  initial_avg = 18 →
  correct_avg = 19 →
  correct_num = 36 →
  (n : ℚ) * initial_avg + (correct_num - wrong_num) = (n : ℚ) * correct_avg →
  wrong_num = 26 := by
sorry

end NUMINAMATH_CALUDE_wrong_number_calculation_l451_45166


namespace NUMINAMATH_CALUDE_fireworks_count_l451_45160

/-- The number of fireworks Henry and his friend have now -/
def total_fireworks (henry_new : ℕ) (friend_new : ℕ) (last_year : ℕ) : ℕ :=
  henry_new + friend_new + last_year

/-- Proof that Henry and his friend have 11 fireworks in total -/
theorem fireworks_count : total_fireworks 2 3 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fireworks_count_l451_45160


namespace NUMINAMATH_CALUDE_triangle_properties_l451_45151

/-- Triangle represented by three points in 2D space -/
structure Triangle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- Calculate the squared distance between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Check if a triangle is a right triangle -/
def isRightTriangle (t : Triangle) : Prop :=
  let a := distanceSquared t.p1 t.p2
  let b := distanceSquared t.p2 t.p3
  let c := distanceSquared t.p3 t.p1
  (a + b = c) ∨ (b + c = a) ∨ (c + a = b)

/-- Triangle A -/
def triangleA : Triangle :=
  { p1 := (0, 0), p2 := (3, 4), p3 := (0, 8) }

/-- Triangle B -/
def triangleB : Triangle :=
  { p1 := (3, 4), p2 := (10, 4), p3 := (3, 0) }

theorem triangle_properties :
  ¬(isRightTriangle triangleA) ∧
  (isRightTriangle triangleB) ∧
  (distanceSquared triangleB.p1 triangleB.p2 = 65 ∨
   distanceSquared triangleB.p2 triangleB.p3 = 65 ∨
   distanceSquared triangleB.p3 triangleB.p1 = 65) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l451_45151


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_is_2_root_6_l451_45179

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ :=
  sorry

/-- The specific quadrilateral from the problem -/
def problem_quadrilateral : Quadrilateral :=
  ⟨13, 10, 8, 11⟩

theorem largest_inscribed_circle_radius_is_2_root_6 :
  largest_inscribed_circle_radius problem_quadrilateral = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_is_2_root_6_l451_45179


namespace NUMINAMATH_CALUDE_vector_magnitude_l451_45118

def a : ℝ × ℝ := (1, 2)
def b : ℝ → ℝ × ℝ := λ t ↦ (2, t)

theorem vector_magnitude (t : ℝ) (h : a.1 * (b t).1 + a.2 * (b t).2 = 0) :
  Real.sqrt ((b t).1^2 + (b t).2^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l451_45118


namespace NUMINAMATH_CALUDE_pipe_filling_time_l451_45136

/-- Given a tank and two pipes, prove that if one pipe takes T minutes to fill the tank,
    another pipe takes 12 minutes, and both pipes together take 4.8 minutes,
    then T = 8 minutes. -/
theorem pipe_filling_time (T : ℝ) : 
  (T > 0) →  -- T is positive (implied by the context)
  (1 / T + 1 / 12 = 1 / 4.8) →  -- Combined rate equation
  T = 8 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l451_45136


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l451_45143

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if the points (1,-2), (3,4), and (6,k/3) are collinear, then k = 39. -/
theorem collinear_points_k_value :
  ∀ k : ℝ, collinear 1 (-2) 3 4 6 (k/3) → k = 39 := by
  sorry

#check collinear_points_k_value

end NUMINAMATH_CALUDE_collinear_points_k_value_l451_45143


namespace NUMINAMATH_CALUDE_rationalize_denominator_cube_root_l451_45173

theorem rationalize_denominator_cube_root (x : ℝ) (h : x > 0) :
  x / (x^(1/3)) = x^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_cube_root_l451_45173


namespace NUMINAMATH_CALUDE_special_triangle_properties_l451_45148

/-- A right-angled triangle with special properties -/
structure SpecialTriangle where
  -- The hypotenuse of the triangle
  hypotenuse : ℝ
  -- The shorter leg of the triangle
  short_leg : ℝ
  -- The longer leg of the triangle
  long_leg : ℝ
  -- The hypotenuse is 1
  hyp_is_one : hypotenuse = 1
  -- The shorter leg is (√5 - 1) / 2
  short_leg_value : short_leg = (Real.sqrt 5 - 1) / 2
  -- The longer leg is the square root of the shorter leg
  long_leg_value : long_leg = Real.sqrt short_leg

/-- Theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  -- The longer leg is the geometric mean of the hypotenuse and shorter leg
  t.long_leg ^ 2 = t.hypotenuse * t.short_leg ∧
  -- All segments formed by successive altitudes are powers of the longer leg
  ∀ n : ℕ, ∃ segment : ℝ, segment = t.long_leg ^ n ∧ 0 ≤ n ∧ n ≤ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l451_45148


namespace NUMINAMATH_CALUDE_complex_norm_problem_l451_45145

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 17)
  (h2 : Complex.abs (z + 3 * w) = 4)
  (h3 : Complex.abs (z + w) = 6) :
  Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_complex_norm_problem_l451_45145


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l451_45112

/-- Given a line passing through points A(-2, m) and B(m, 4) that is parallel to the line 2x + y + 1 = 0, prove that m = 8. -/
theorem parallel_lines_m_value (m : ℝ) : 
  (∃ (k : ℝ), k * (m - (-2)) = 4 - m ∧ k = -2) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l451_45112


namespace NUMINAMATH_CALUDE_f_composition_value_l451_45134

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else Real.exp (x + 1) - 2

theorem f_composition_value : f (f (1 / Real.exp 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l451_45134


namespace NUMINAMATH_CALUDE_strawberry_count_l451_45171

/-- Calculates the total number of strawberries after picking more -/
def total_strawberries (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: The total number of strawberries is the sum of initial and additional strawberries -/
theorem strawberry_count (initial additional : ℕ) :
  total_strawberries initial additional = initial + additional := by
  sorry

end NUMINAMATH_CALUDE_strawberry_count_l451_45171


namespace NUMINAMATH_CALUDE_original_books_l451_45198

/-- The number of books person A has -/
def books_A : ℕ := sorry

/-- The number of books person B has -/
def books_B : ℕ := sorry

/-- If A gives 10 books to B, they have an equal number of books -/
axiom equal_books : books_A - 10 = books_B + 10

/-- If B gives 10 books to A, A has twice the number of books B has left -/
axiom double_books : books_A + 10 = 2 * (books_B - 10)

theorem original_books : books_A = 70 ∧ books_B = 50 := by sorry

end NUMINAMATH_CALUDE_original_books_l451_45198


namespace NUMINAMATH_CALUDE_water_in_tank_after_rain_l451_45180

/-- Calculates the final amount of water in a tank after rainfall, considering inflow, leakage, and evaporation. -/
def final_water_amount (initial_water : ℝ) (inflow_rate : ℝ) (leakage_rate : ℝ) (evaporation_rate : ℝ) (duration : ℝ) : ℝ :=
  initial_water + (inflow_rate - leakage_rate - evaporation_rate) * duration

/-- Theorem stating that the final amount of water in the tank is 226 L -/
theorem water_in_tank_after_rain (initial_water : ℝ) (inflow_rate : ℝ) (leakage_rate : ℝ) (evaporation_rate : ℝ) (duration : ℝ) :
  initial_water = 100 ∧
  inflow_rate = 2 ∧
  leakage_rate = 0.5 ∧
  evaporation_rate = 0.1 ∧
  duration = 90 →
  final_water_amount initial_water inflow_rate leakage_rate evaporation_rate duration = 226 :=
by sorry

end NUMINAMATH_CALUDE_water_in_tank_after_rain_l451_45180


namespace NUMINAMATH_CALUDE_log_equation_implies_relationship_l451_45113

theorem log_equation_implies_relationship (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) (hy1 : y ≠ 1) :
  9 * (Real.log y / Real.log c)^2 + 5 * (Real.log y / Real.log d)^2 = 18 * (Real.log y)^2 / (Real.log c * Real.log d) →
  d = c^(1/Real.sqrt 3) ∨ d = c^(Real.sqrt 3) ∨ d = c^(1/Real.sqrt 0.6) ∨ d = c^(Real.sqrt 0.6) := by
sorry

end NUMINAMATH_CALUDE_log_equation_implies_relationship_l451_45113


namespace NUMINAMATH_CALUDE_triangle_inequality_l451_45139

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = π) :
  let f (x : ℝ) := 1 - Real.sqrt (Real.sqrt 3 * Real.tan (x / 2)) + Real.sqrt 3 * Real.tan (x / 2)
  (f A * f B) + (f B * f C) + (f C * f A) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l451_45139


namespace NUMINAMATH_CALUDE_min_value_expression_l451_45195

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x*y*z) ≥ 512 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) = 512 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l451_45195


namespace NUMINAMATH_CALUDE_sqrt_sum_zero_implies_power_l451_45137

theorem sqrt_sum_zero_implies_power (a b : ℝ) : 
  Real.sqrt (a + 3) + Real.sqrt (2 - b) = 0 → a^b = 9 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_zero_implies_power_l451_45137


namespace NUMINAMATH_CALUDE_triangle_area_l451_45175

theorem triangle_area (base height : ℝ) (h1 : base = 4) (h2 : height = 6) :
  (base * height) / 2 = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l451_45175


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_quadratics_l451_45140

theorem unique_prime_with_prime_quadratics :
  ∃! p : ℕ, Prime p ∧ Prime (4 * p^2 + 1) ∧ Prime (6 * p^2 + 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_quadratics_l451_45140


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l451_45172

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    a + b + c + d = 18 ∧
    b + c = 7 ∧
    a - d = 3 ∧
    n % 9 = 0

theorem unique_four_digit_number :
  ∃! (n : ℕ), is_valid_number n ∧ n = 6453 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l451_45172


namespace NUMINAMATH_CALUDE_second_concert_attendance_l451_45154

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (additional_people : ℕ) 
  (h1 : first_concert = 65899)
  (h2 : additional_people = 119) : 
  first_concert + additional_people = 66018 := by
sorry

end NUMINAMATH_CALUDE_second_concert_attendance_l451_45154


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l451_45177

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The GDP value in yuan -/
def gdp : ℝ := 338.8e9

theorem gdp_scientific_notation :
  toScientificNotation gdp = ScientificNotation.mk 3.388 10 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l451_45177


namespace NUMINAMATH_CALUDE_inequality_solution_l451_45186

theorem inequality_solution (a : ℝ) (h : a < 0) :
  let solution := {x : ℝ | a * x^2 + (1 - a) * x - 1 > 0}
  ((-1 < a ∧ a < 0) → solution = {x | 1 < x ∧ x < -1/a}) ∧
  (a = -1 → solution = ∅) ∧
  (a < -1 → solution = {x | -1/a < x ∧ x < 1}) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l451_45186


namespace NUMINAMATH_CALUDE_product_of_fractions_l451_45185

theorem product_of_fractions : (2 : ℚ) / 9 * (5 : ℚ) / 4 = (5 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l451_45185


namespace NUMINAMATH_CALUDE_product_of_integers_l451_45163

theorem product_of_integers (A B C D : ℕ+) : 
  A + B + C + D = 100 →
  2^(A:ℕ) = B - 4 →
  C + 6 = D →
  B + C = D + 10 →
  A * B * C * D = 33280 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l451_45163


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l451_45182

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |2*x - a|

theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x < 2} = {x : ℝ | 1/4 < x ∧ x < 5/4} := by sorry

theorem range_of_a :
  (∀ x : ℝ, f a x ≥ 3*a + 2) ↔ -3/2 ≤ a ∧ a ≤ -1/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l451_45182


namespace NUMINAMATH_CALUDE_winning_scores_count_l451_45138

/-- Represents a cross country meet with specific rules --/
structure CrossCountryMeet where
  runners_per_team : Nat
  total_runners : Nat
  min_score : Nat
  max_score : Nat

/-- Calculates the total score of all runners --/
def total_meet_score (meet : CrossCountryMeet) : Nat :=
  meet.total_runners * (meet.total_runners + 1) / 2

/-- Defines a valid cross country meet with given parameters --/
def valid_meet : CrossCountryMeet :=
  { runners_per_team := 6
  , total_runners := 12
  , min_score := 21
  , max_score := 38 }

/-- Theorem stating the number of possible winning scores --/
theorem winning_scores_count (meet : CrossCountryMeet) 
  (h1 : meet = valid_meet) 
  (h2 : meet.total_runners = 2 * meet.runners_per_team) 
  (h3 : total_meet_score meet = 78) : 
  (meet.max_score - meet.min_score + 1 : Nat) = 18 := by
  sorry

end NUMINAMATH_CALUDE_winning_scores_count_l451_45138


namespace NUMINAMATH_CALUDE_rob_has_five_nickels_l451_45153

/-- Represents the number of coins of each type Rob has -/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in cents for a given CoinCount -/
def totalValueInCents (coins : CoinCount) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Proves that Rob has 5 nickels given the conditions -/
theorem rob_has_five_nickels :
  ∃ (robsCoins : CoinCount),
    robsCoins.quarters = 7 ∧
    robsCoins.dimes = 3 ∧
    robsCoins.pennies = 12 ∧
    totalValueInCents robsCoins = 242 ∧
    robsCoins.nickels = 5 := by
  sorry


end NUMINAMATH_CALUDE_rob_has_five_nickels_l451_45153


namespace NUMINAMATH_CALUDE_perfect_square_condition_l451_45114

theorem perfect_square_condition (X M : ℕ) : 
  (1000 < X ∧ X < 8000) → 
  (M > 1) → 
  (X = M * M^2) → 
  (∃ k : ℕ, X = k^2) → 
  M = 16 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l451_45114


namespace NUMINAMATH_CALUDE_randys_fathers_biscuits_l451_45193

/-- Proves that Randy's father gave him 13 biscuits given the initial conditions and final result. -/
theorem randys_fathers_biscuits :
  ∀ (initial mother_gave brother_ate final father_gave : ℕ),
  initial = 32 →
  mother_gave = 15 →
  brother_ate = 20 →
  final = 40 →
  initial + mother_gave + father_gave - brother_ate = final →
  father_gave = 13 := by
  sorry

end NUMINAMATH_CALUDE_randys_fathers_biscuits_l451_45193


namespace NUMINAMATH_CALUDE_prob_even_sum_is_one_third_l451_45104

/-- Probability of an even outcome for the first wheel -/
def p_even_1 : ℚ := 1/2

/-- Probability of an even outcome for the second wheel -/
def p_even_2 : ℚ := 1/3

/-- Probability of an even outcome for the third wheel -/
def p_even_3 : ℚ := 3/4

/-- The probability of getting an even sum from three independent events -/
def prob_even_sum (p1 p2 p3 : ℚ) : ℚ :=
  p1 * p2 * p3 +
  (1 - p1) * p2 * p3 +
  p1 * (1 - p2) * p3 +
  p1 * p2 * (1 - p3)

/-- Theorem stating that the probability of an even sum is 1/3 given the specific probabilities -/
theorem prob_even_sum_is_one_third :
  prob_even_sum p_even_1 p_even_2 p_even_3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_one_third_l451_45104


namespace NUMINAMATH_CALUDE_not_right_triangle_with_angle_ratio_l451_45199

theorem not_right_triangle_with_angle_ratio (A B C : ℝ) (h : A + B + C = 180) 
  (ratio : A / 3 = B / 4 ∧ A / 3 = C / 5) : 
  ¬(A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_with_angle_ratio_l451_45199


namespace NUMINAMATH_CALUDE_no_real_roots_l451_45126

theorem no_real_roots : ∀ x : ℝ, (x + 1) * |x + 1| - x * |x| + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l451_45126


namespace NUMINAMATH_CALUDE_min_sum_squares_l451_45128

theorem min_sum_squares (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ x^2 + y^2 + z^2 ≥ m ∧ ∃ (a b c : ℝ), a + b + c = 1 ∧ a^2 + b^2 + c^2 = m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l451_45128


namespace NUMINAMATH_CALUDE_inequality_system_solution_l451_45106

theorem inequality_system_solution (x : ℝ) :
  (1 - x > 0) ∧ ((x + 2) / 3 - 1 ≤ x) → -1/2 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l451_45106


namespace NUMINAMATH_CALUDE_triangle_theorem_l451_45142

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c - t.b = 2 * t.b * Real.cos t.A ∧
  Real.cos t.B = 3/4 ∧
  t.c = 5

-- Theorem to prove
theorem triangle_theorem (t : Triangle) 
  (h : triangle_conditions t) : 
  t.A = 2 * t.B ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = (15/4) * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l451_45142


namespace NUMINAMATH_CALUDE_evaluate_expression_l451_45192

theorem evaluate_expression : (2^2003 * 3^2005) / 6^2004 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l451_45192


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l451_45150

theorem x_range_for_inequality (x : ℝ) : 
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) → 
  x > 3 ∨ x < -1 := by
sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l451_45150


namespace NUMINAMATH_CALUDE_min_value_expression_l451_45184

theorem min_value_expression (x y : ℝ) : (x*y - 2)^2 + (x^2 + y^2)^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l451_45184


namespace NUMINAMATH_CALUDE_triangle_side_length_l451_45109

-- Define the triangle PQR
structure Triangle (P Q R : ℝ) where
  angleSum : P + Q + R = Real.pi
  positive : 0 < P ∧ 0 < Q ∧ 0 < R

-- Define the side lengths
def sideLength (a b : ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_side_length 
  (P Q R : ℝ) 
  (tri : Triangle P Q R) 
  (h1 : Real.cos (2 * P - Q) + Real.sin (P + 2 * Q) = 1)
  (h2 : sideLength P Q = 5)
  (h3 : sideLength P Q + sideLength Q R + sideLength R P = 12) :
  sideLength Q R = 3.5 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l451_45109


namespace NUMINAMATH_CALUDE_second_part_interest_rate_l451_45155

-- Define the total sum and the two parts
def total_sum : ℚ := 2717
def second_part : ℚ := 1672
def first_part : ℚ := total_sum - second_part

-- Define the interest rates and time periods
def first_rate : ℚ := 3 / 100
def first_time : ℚ := 8
def second_time : ℚ := 3

-- Define the theorem
theorem second_part_interest_rate :
  ∃ (r : ℚ), 
    (first_part * first_rate * first_time = second_part * r * second_time) ∧
    (r = 5 / 100) := by
  sorry

end NUMINAMATH_CALUDE_second_part_interest_rate_l451_45155


namespace NUMINAMATH_CALUDE_recurrence_sequence_x7_l451_45168

/-- A sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (x : ℕ → ℕ) : Prop :=
  (∀ n, x n > 0) ∧
  (∀ n ∈ ({1, 2, 3, 4} : Finset ℕ), x (n + 3) = x (n + 2) * (x (n + 1) + x n))

theorem recurrence_sequence_x7 (x : ℕ → ℕ) (h : RecurrenceSequence x) (h6 : x 6 = 144) :
  x 7 = 3456 := by
  sorry

#check recurrence_sequence_x7

end NUMINAMATH_CALUDE_recurrence_sequence_x7_l451_45168


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l451_45125

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 8*x - 4*k = 0 ∧ 
   ∀ y : ℝ, y^2 - 8*y - 4*k = 0 → y = x) → 
  k = -4 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l451_45125


namespace NUMINAMATH_CALUDE_solution_distribution_l451_45132

def test_tube_volumes : List ℝ := [7, 4, 5, 4, 6, 8, 7, 3, 9, 6]
def num_beakers : ℕ := 5

theorem solution_distribution (volumes : List ℝ) (num_beakers : ℕ) 
  (h1 : volumes = test_tube_volumes) 
  (h2 : num_beakers = 5) : 
  (volumes.sum / num_beakers : ℝ) = 11.8 := by
  sorry

#check solution_distribution

end NUMINAMATH_CALUDE_solution_distribution_l451_45132


namespace NUMINAMATH_CALUDE_base_eight_sum_l451_45119

theorem base_eight_sum (A B C : ℕ) : 
  A ≠ 0 → B ≠ 0 → C ≠ 0 →
  A ≠ B → B ≠ C → A ≠ C →
  A < 8 → B < 8 → C < 8 →
  (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) = 8^3 * A + 8^2 * A + 8 * A →
  B + C = 7 := by
sorry

end NUMINAMATH_CALUDE_base_eight_sum_l451_45119


namespace NUMINAMATH_CALUDE_puppies_left_l451_45170

/-- The number of puppies Alyssa had initially -/
def initial_puppies : ℕ := 7

/-- The number of puppies Alyssa gave to her friends -/
def given_puppies : ℕ := 5

/-- Theorem: Alyssa is left with 2 puppies -/
theorem puppies_left : initial_puppies - given_puppies = 2 := by
  sorry

end NUMINAMATH_CALUDE_puppies_left_l451_45170


namespace NUMINAMATH_CALUDE_sum_of_integers_30_to_50_l451_45162

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_of_integers_30_to_50 (x y : ℕ) :
  x = sum_of_integers 30 50 →
  y = count_even_integers 30 50 →
  x + y = 851 →
  x = 840 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_30_to_50_l451_45162


namespace NUMINAMATH_CALUDE_probability_at_least_three_hits_l451_45131

def probability_hit_single_shot : ℝ := 0.8
def number_of_shots : ℕ := 4
def minimum_hits : ℕ := 3

theorem probability_at_least_three_hits :
  let p := probability_hit_single_shot
  let n := number_of_shots
  let k := minimum_hits
  (Finset.sum (Finset.range (n - k + 1))
    (λ i => (n.choose (k + i)) * p^(k + i) * (1 - p)^(n - k - i))) = 0.8192 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_three_hits_l451_45131


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l451_45117

theorem geometric_sequence_sum (a r : ℝ) (h1 : a * (1 - r^1000) / (1 - r) = 1024) 
  (h2 : a * (1 - r^2000) / (1 - r) = 2040) : 
  a * (1 - r^3000) / (1 - r) = 3048 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l451_45117


namespace NUMINAMATH_CALUDE_barrel_contents_l451_45188

theorem barrel_contents :
  ∀ (x : ℝ),
  (x > 0) →
  (x / 6 = x - 5 * x / 6) →
  (5 * x / 30 = 5 * x / 6 - 2 * x / 3) →
  (x / 6 = 2 * x / 3 - x / 2) →
  ((x + 120) + (5 * x / 6 + 120) = 4 * (x / 2)) →
  (x = 1440 ∧ 
   5 * x / 6 = 1200 ∧ 
   2 * x / 3 = 960 ∧ 
   x / 2 = 720) :=
by sorry

end NUMINAMATH_CALUDE_barrel_contents_l451_45188


namespace NUMINAMATH_CALUDE_flag_problem_l451_45129

theorem flag_problem (x : ℝ) : 
  (8 * 5 : ℝ) + (x * 7) + (5 * 5) = 15 * 9 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_flag_problem_l451_45129


namespace NUMINAMATH_CALUDE_lcm_problem_l451_45157

theorem lcm_problem (a b : ℕ) (h1 : a < b) (h2 : a + b = 78) (h3 : Nat.lcm a b = 252) : b - a = 6 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l451_45157


namespace NUMINAMATH_CALUDE_postal_stamps_theorem_l451_45158

/-- The number of color stamps sold -/
def color_stamps : ℕ := 578833

/-- The total number of stamps sold -/
def total_stamps : ℕ := 1102609

/-- The number of black-and-white stamps sold -/
def bw_stamps : ℕ := total_stamps - color_stamps

theorem postal_stamps_theorem : 
  bw_stamps = 523776 := by sorry

end NUMINAMATH_CALUDE_postal_stamps_theorem_l451_45158


namespace NUMINAMATH_CALUDE_sequence_properties_l451_45141

/-- Given a sequence {aₙ} with sum Sₙ satisfying Sₙ = t(Sₙ - aₙ + 1) where t ≠ 0 and t ≠ 1,
    and a sequence {bₙ} defined as bₙ = aₙ² + Sₙ · aₙ which is geometric,
    prove that {aₙ} is geometric and find the general term of {bₙ}. -/
theorem sequence_properties (t : ℝ) (a b : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : t ≠ 0) (h2 : t ≠ 1)
  (h3 : ∀ n, S n = t * (S n - a n + 1))
  (h4 : ∀ n, b n = a n ^ 2 + S n * a n)
  (h5 : ∃ q, ∀ n, b (n + 1) = q * b n) :
  (∀ n, a (n + 1) = t * a n) ∧
  (∀ n, b n = t^(n + 1) * (2 * t + 1)^(n - 1) / 2^(n - 2)) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l451_45141


namespace NUMINAMATH_CALUDE_other_replaced_man_age_proof_l451_45167

/-- The age of the other replaced man in a group of three men -/
def other_replaced_man_age : ℕ := 26

theorem other_replaced_man_age_proof 
  (initial_men : ℕ) 
  (replaced_men : ℕ) 
  (known_replaced_age : ℕ) 
  (new_men_avg_age : ℝ) 
  (h1 : initial_men = 3)
  (h2 : replaced_men = 2)
  (h3 : known_replaced_age = 23)
  (h4 : new_men_avg_age = 25)
  (h5 : ∀ (initial_avg new_avg : ℝ), new_avg > initial_avg) :
  other_replaced_man_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_other_replaced_man_age_proof_l451_45167


namespace NUMINAMATH_CALUDE_probability_all_white_balls_l451_45144

/-- The probability of drawing 7 white balls from a box containing 7 white and 8 black balls -/
theorem probability_all_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) :
  total_balls = white_balls + black_balls →
  total_balls = 15 →
  white_balls = 7 →
  black_balls = 8 →
  drawn_balls = 7 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 6435 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_white_balls_l451_45144


namespace NUMINAMATH_CALUDE_specific_glued_cubes_surface_area_l451_45124

/-- Represents a 3D shape formed by gluing two cubes --/
structure GluedCubes where
  large_edge_length : ℝ
  small_edge_length : ℝ

/-- Calculates the surface area of the GluedCubes shape --/
def surface_area (shape : GluedCubes) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the specific GluedCubes shape is 136 --/
theorem specific_glued_cubes_surface_area :
  ∃ (shape : GluedCubes),
    shape.large_edge_length = 4 ∧
    shape.small_edge_length = 1 ∧
    surface_area shape = 136 :=
  sorry

end NUMINAMATH_CALUDE_specific_glued_cubes_surface_area_l451_45124


namespace NUMINAMATH_CALUDE_angle_measure_in_pentagon_and_triangle_l451_45190

/-- Given a pentagon with angles A, B, C, E, and F, where angles D, E, and F form a triangle,
    this theorem proves that if m∠A = 80°, m∠B = 30°, and m∠C = 20°, then m∠D = 130°. -/
theorem angle_measure_in_pentagon_and_triangle 
  (A B C D E F : Real) 
  (pentagon : A + B + C + E + F = 540) 
  (triangle : D + E + F = 180) 
  (angle_A : A = 80) 
  (angle_B : B = 30) 
  (angle_C : C = 20) : 
  D = 130 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_pentagon_and_triangle_l451_45190


namespace NUMINAMATH_CALUDE_third_card_value_l451_45159

theorem third_card_value (a b c : ℕ) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h2 : 1 ≤ a ∧ a ≤ 13)
  (h3 : 1 ≤ b ∧ b ≤ 13)
  (h4 : 1 ≤ c ∧ c ≤ 13)
  (h5 : a + b = 25)
  (h6 : b + c = 13) :
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_third_card_value_l451_45159


namespace NUMINAMATH_CALUDE_age_height_not_function_l451_45187

-- Define a type for age and height
def Age := ℕ
def Height := ℝ

-- Define a relation between age and height
def AgeHeightRelation := Age → Set Height

-- Define what it means for a relation to be a function
def IsFunction (R : α → Set β) : Prop :=
  ∀ x : α, ∃! y : β, y ∈ R x

-- State the theorem
theorem age_height_not_function :
  ∃ R : AgeHeightRelation, ¬ IsFunction R :=
sorry

end NUMINAMATH_CALUDE_age_height_not_function_l451_45187


namespace NUMINAMATH_CALUDE_first_month_sale_l451_45123

def sales_month_2_to_6 : List ℕ := [6927, 6855, 7230, 6562, 4891]
def average_sale : ℕ := 6500
def number_of_months : ℕ := 6

theorem first_month_sale :
  (average_sale * number_of_months - sales_month_2_to_6.sum) = 6535 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l451_45123


namespace NUMINAMATH_CALUDE_horatio_sonnets_count_l451_45116

/-- Represents the number of sonnets Horatio wrote -/
def total_sonnets : ℕ := 12

/-- Represents the number of lines in each sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of sonnets Horatio's lady fair heard -/
def sonnets_heard : ℕ := 7

/-- Represents the number of romantic lines that were never heard -/
def unheard_lines : ℕ := 70

/-- Theorem stating that the total number of sonnets Horatio wrote is correct -/
theorem horatio_sonnets_count :
  total_sonnets = sonnets_heard + (unheard_lines / lines_per_sonnet) := by
  sorry

end NUMINAMATH_CALUDE_horatio_sonnets_count_l451_45116


namespace NUMINAMATH_CALUDE_tree_height_difference_l451_45105

def maple_height : ℚ := 10 + 3/4
def pine_height : ℚ := 12 + 7/8

theorem tree_height_difference :
  pine_height - maple_height = 2 + 1/8 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l451_45105


namespace NUMINAMATH_CALUDE_max_cars_with_ac_no_stripes_l451_45120

theorem max_cars_with_ac_no_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (cars_with_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 49)
  (h3 : cars_with_stripes ≥ 51) :
  ∃ (max_cars : ℕ), 
    max_cars ≤ total_cars - cars_without_ac ∧
    max_cars ≤ total_cars - cars_with_stripes ∧
    ∀ (n : ℕ), n ≤ total_cars - cars_without_ac ∧ 
               n ≤ total_cars - cars_with_stripes → 
               n ≤ max_cars ∧
    max_cars = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_cars_with_ac_no_stripes_l451_45120


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l451_45121

theorem largest_four_digit_divisible_by_six :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 6 = 0 → n ≤ 9960 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l451_45121


namespace NUMINAMATH_CALUDE_donny_apple_purchase_cost_l451_45152

-- Define the prices of apples
def small_apple_price : ℚ := 1.5
def medium_apple_price : ℚ := 2
def big_apple_price : ℚ := 3

-- Define the number of apples Donny bought
def small_apples_bought : ℕ := 6
def medium_apples_bought : ℕ := 6
def big_apples_bought : ℕ := 8

-- Calculate the total cost
def total_cost : ℚ := 
  small_apple_price * small_apples_bought +
  medium_apple_price * medium_apples_bought +
  big_apple_price * big_apples_bought

-- Theorem statement
theorem donny_apple_purchase_cost : total_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_donny_apple_purchase_cost_l451_45152


namespace NUMINAMATH_CALUDE_grid_paths_count_l451_45111

/-- Represents a grid of roads between two locations -/
structure Grid where
  north_paths : Nat
  east_paths : Nat

/-- Calculates the total number of paths in a grid -/
def total_paths (g : Grid) : Nat :=
  g.north_paths * g.east_paths

/-- Theorem stating that the total number of paths in the given grid is 15 -/
theorem grid_paths_count : 
  ∀ g : Grid, g.north_paths = 3 → g.east_paths = 5 → total_paths g = 15 := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_count_l451_45111


namespace NUMINAMATH_CALUDE_marble_probability_l451_45183

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  sum_check : total = black + white

/-- The problem setup -/
def marble_problem (box1 box2 : MarbleBox) : Prop :=
  box1.total + box2.total = 30 ∧
  box1.black = 3 * box2.black ∧
  (box1.black : ℚ) / box1.total * (box2.black : ℚ) / box2.total = 1/2

theorem marble_probability (box1 box2 : MarbleBox) 
  (h : marble_problem box1 box2) : 
  (box1.white : ℚ) / box1.total * (box2.white : ℚ) / box2.total = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l451_45183


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l451_45101

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l451_45101


namespace NUMINAMATH_CALUDE_geometric_sequence_special_term_l451_45156

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The 14th term of a geometric sequence -/
def a_14 (a : ℕ → ℝ) : ℝ := a 14

/-- The 4th term of a geometric sequence -/
def a_4 (a : ℕ → ℝ) : ℝ := a 4

/-- The 24th term of a geometric sequence -/
def a_24 (a : ℕ → ℝ) : ℝ := a 24

/-- Theorem: In a geometric sequence, if a_4 and a_24 are roots of 3x^2 - 2014x + 9 = 0, then a_14 = √3 -/
theorem geometric_sequence_special_term (a : ℕ → ℝ) :
  geometric_sequence a →
  (3 * (a_4 a)^2 - 2014 * (a_4 a) + 9 = 0) →
  (3 * (a_24 a)^2 - 2014 * (a_24 a) + 9 = 0) →
  a_14 a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_term_l451_45156


namespace NUMINAMATH_CALUDE_triangle_inequality_l451_45197

theorem triangle_inequality (x y z : ℝ) (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi) :
  (x + y + z)^2 ≥ 4 * (y * z * Real.sin A^2 + z * x * Real.sin B^2 + x * y * Real.sin C^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l451_45197


namespace NUMINAMATH_CALUDE_intersection_size_lower_bound_l451_45178

theorem intersection_size_lower_bound 
  (n k : ℕ) 
  (A : Fin (k + 1) → Finset (Fin (4 * n))) 
  (h1 : ∀ i, (A i).card = 2 * n) :
  ∃ i j, i ≠ j ∧ (A i ∩ A j).card ≥ n - n / k := by
  sorry

end NUMINAMATH_CALUDE_intersection_size_lower_bound_l451_45178


namespace NUMINAMATH_CALUDE_f_min_value_f_attains_min_l451_45165

def f (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem f_min_value : ∀ x : ℝ, f x ≥ 2 := by sorry

theorem f_attains_min : ∃ x : ℝ, f x = 2 := by sorry

end NUMINAMATH_CALUDE_f_min_value_f_attains_min_l451_45165


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l451_45169

theorem x_squared_plus_reciprocal_squared (x : ℝ) (h : x + 1/x = 3.5) : 
  x^2 + 1/x^2 = 10.25 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l451_45169


namespace NUMINAMATH_CALUDE_bus_profit_properties_l451_45161

/-- Represents the daily profit of a bus given the number of passengers -/
def daily_profit (x : ℕ) : ℤ :=
  2 * x - 600

theorem bus_profit_properties :
  let min_passengers_no_loss := 300
  let profit_500_passengers := daily_profit 500
  let relationship (x : ℕ) := daily_profit x = 2 * x - 600
  (∀ x : ℕ, x ≥ min_passengers_no_loss → daily_profit x ≥ 0) ∧
  (profit_500_passengers = 400) ∧
  (∀ x : ℕ, relationship x) :=
by sorry

end NUMINAMATH_CALUDE_bus_profit_properties_l451_45161


namespace NUMINAMATH_CALUDE_richard_twice_scott_age_l451_45102

/-- Represents the ages of the three brothers -/
structure BrothersAges where
  david : ℕ
  richard : ℕ
  scott : ℕ

/-- The current ages of the brothers -/
def currentAges : BrothersAges :=
  { david := 14
    richard := 20
    scott := 6 }

/-- The conditions given in the problem -/
axiom age_difference_richard_david : currentAges.richard = currentAges.david + 6
axiom age_difference_david_scott : currentAges.david = currentAges.scott + 8
axiom david_age_three_years_ago : currentAges.david = 11 + 3

/-- The theorem to be proved -/
theorem richard_twice_scott_age (x : ℕ) :
  x = 8 ↔ currentAges.richard + x = 2 * (currentAges.scott + x) :=
sorry

end NUMINAMATH_CALUDE_richard_twice_scott_age_l451_45102


namespace NUMINAMATH_CALUDE_unique_integer_solution_l451_45133

theorem unique_integer_solution : 
  ∃! (x : ℤ), (abs x : ℝ) < 5 * Real.pi ∧ x^2 - 4*x + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l451_45133


namespace NUMINAMATH_CALUDE_last_number_proof_l451_45135

theorem last_number_proof (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : (B + C + D) / 3 = 3)
  (h3 : A + D = 13) :
  D = 2 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l451_45135


namespace NUMINAMATH_CALUDE_computer_peripherals_cost_fraction_l451_45194

theorem computer_peripherals_cost_fraction :
  let computer_cost : ℚ := 1500
  let base_video_card_cost : ℚ := 300
  let upgraded_video_card_cost : ℚ := 2 * base_video_card_cost
  let total_spent : ℚ := 2100
  let computer_with_upgrade_cost : ℚ := computer_cost + upgraded_video_card_cost - base_video_card_cost
  let peripherals_cost : ℚ := total_spent - computer_with_upgrade_cost
  peripherals_cost / computer_cost = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_computer_peripherals_cost_fraction_l451_45194


namespace NUMINAMATH_CALUDE_non_intersecting_path_count_l451_45181

/-- A path on a grid from (0,0) to (n,n) that can only move top or right -/
def GridPath (n : ℕ) := List (Bool)

/-- Two paths are non-intersecting if they don't share any point except (0,0) and (n,n) -/
def NonIntersecting (n : ℕ) (p1 p2 : GridPath n) : Prop := sorry

/-- The number of non-intersecting pairs of paths from (0,0) to (n,n) -/
def NonIntersectingPathCount (n : ℕ) : ℕ := sorry

theorem non_intersecting_path_count (n : ℕ) : 
  NonIntersectingPathCount n = (Nat.choose (2*n-2) (n-1))^2 - (Nat.choose (2*n-2) (n-2))^2 := by sorry

end NUMINAMATH_CALUDE_non_intersecting_path_count_l451_45181


namespace NUMINAMATH_CALUDE_polygon_sides_l451_45103

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 3 * 360 + 180) →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l451_45103


namespace NUMINAMATH_CALUDE_initial_student_count_l451_45127

/-- Given the initial average weight, new average weight after admitting a new student,
    and the weight of the new student, prove that the initial number of students is 19. -/
theorem initial_student_count
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (new_student_weight : ℝ)
  (h1 : initial_avg = 15)
  (h2 : new_avg = 14.8)
  (h3 : new_student_weight = 11) :
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 19 := by
  sorry

#check initial_student_count

end NUMINAMATH_CALUDE_initial_student_count_l451_45127


namespace NUMINAMATH_CALUDE_ellipse_equation_l451_45115

/-- An ellipse with center at the origin, foci on the x-axis, eccentricity 1/2, 
    and the perimeter of triangle PF₁F₂ equal to 12 -/
structure Ellipse where
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The perimeter of triangle PF₁F₂ -/
  perimeter : ℝ
  /-- The eccentricity is 1/2 -/
  h_e : e = 1/2
  /-- The perimeter is 12 -/
  h_perimeter : perimeter = 12

/-- The standard equation of the ellipse -/
def standardEquation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2/16 + y^2/12 = 1

/-- Theorem stating that the given ellipse satisfies the standard equation -/
theorem ellipse_equation (E : Ellipse) (x y : ℝ) : 
  standardEquation E x y := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l451_45115


namespace NUMINAMATH_CALUDE_total_price_after_increase_l451_45122

/-- Calculates the total price for a buyer purchasing jewelry and paintings 
    after a price increase. -/
theorem total_price_after_increase 
  (initial_jewelry_price : ℝ) 
  (initial_painting_price : ℝ)
  (jewelry_price_increase : ℝ)
  (painting_price_increase_percent : ℝ)
  (jewelry_quantity : ℕ)
  (painting_quantity : ℕ)
  (h1 : initial_jewelry_price = 30)
  (h2 : initial_painting_price = 100)
  (h3 : jewelry_price_increase = 10)
  (h4 : painting_price_increase_percent = 20)
  (h5 : jewelry_quantity = 2)
  (h6 : painting_quantity = 5) :
  let new_jewelry_price := initial_jewelry_price + jewelry_price_increase
  let new_painting_price := initial_painting_price * (1 + painting_price_increase_percent / 100)
  let total_price := new_jewelry_price * jewelry_quantity + new_painting_price * painting_quantity
  total_price = 680 := by
sorry


end NUMINAMATH_CALUDE_total_price_after_increase_l451_45122


namespace NUMINAMATH_CALUDE_plot_perimeter_is_220_l451_45176

/-- Represents a rectangular plot with the given conditions -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ
  lengthWidthRelation : length = width + 10
  fencingCostRelation : fencingCostPerMeter * (2 * (length + width)) = totalFencingCost

/-- The perimeter of the rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.width)

/-- Theorem stating that the perimeter of the plot is 220 meters -/
theorem plot_perimeter_is_220 (plot : RectangularPlot) 
    (h1 : plot.fencingCostPerMeter = 6.5)
    (h2 : plot.totalFencingCost = 1430) : 
  perimeter plot = 220 := by
  sorry

end NUMINAMATH_CALUDE_plot_perimeter_is_220_l451_45176


namespace NUMINAMATH_CALUDE_sequence_kth_term_value_l451_45108

/-- Given a sequence {a_n} with sum S_n = n^2 - 9n and 5 < a_k < 8, prove k = 8 -/
theorem sequence_kth_term_value (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) :
  (∀ n, S n = n^2 - 9*n) →
  (∀ n, a n = 2*n - 10) →
  (5 < a k ∧ a k < 8) →
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_kth_term_value_l451_45108


namespace NUMINAMATH_CALUDE_probability_select_four_or_five_l451_45146

/-- The probability of selecting a product with a number not less than 4 from 5 products -/
theorem probability_select_four_or_five (n : ℕ) (h : n = 5) :
  (Finset.filter (λ i => i ≥ 4) (Finset.range n)).card / n = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_select_four_or_five_l451_45146
