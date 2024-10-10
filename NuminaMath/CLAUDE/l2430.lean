import Mathlib

namespace quadratic_equation_condition_l2430_243032

theorem quadratic_equation_condition (m : ℝ) : 
  (∀ x, (m - 3) * x^(|m| + 2) + 2 * x - 7 = 0 ↔ ∃ a b c, a ≠ 0 ∧ a * x^2 + b * x + c = 0) ↔ m = 0 :=
by sorry

end quadratic_equation_condition_l2430_243032


namespace garage_cleanup_l2430_243022

theorem garage_cleanup (total_trips : ℕ) (jean_extra_trips : ℕ) (total_capacity : ℝ) (actual_weight : ℝ) 
  (h1 : total_trips = 40)
  (h2 : jean_extra_trips = 6)
  (h3 : total_capacity = 8000)
  (h4 : actual_weight = 7850) : 
  let bill_trips := (total_trips - jean_extra_trips) / 2
  let jean_trips := bill_trips + jean_extra_trips
  let avg_weight := actual_weight / total_trips
  jean_trips = 23 ∧ avg_weight = 196.25 := by
  sorry

end garage_cleanup_l2430_243022


namespace right_triangle_hypotenuse_l2430_243043

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 1 → 
    b = 3 → 
    c^2 = a^2 + b^2 → 
    c = Real.sqrt 10 := by
  sorry

end right_triangle_hypotenuse_l2430_243043


namespace min_value_expression_min_value_attainable_l2430_243038

theorem min_value_expression (x y : ℝ) : (x*y - 2)^2 + (x^2 + y^2) ≥ 4 := by
  sorry

theorem min_value_attainable : ∃ x y : ℝ, (x*y - 2)^2 + (x^2 + y^2) = 4 := by
  sorry

end min_value_expression_min_value_attainable_l2430_243038


namespace function_constancy_l2430_243064

def is_constant {α : Type*} (f : α → ℕ) : Prop :=
  ∀ x y, f x = f y

theorem function_constancy (f : ℤ × ℤ → ℕ) 
  (h : ∀ (x y : ℤ), 4 * f (x, y) = f (x - 1, y) + f (x, y + 1) + f (x + 1, y) + f (x, y - 1)) :
  is_constant f := by
  sorry

end function_constancy_l2430_243064


namespace tangent_line_circle_min_sum_l2430_243092

theorem tangent_line_circle_min_sum (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 ∧ 
               (x - 1)^2 + (y - 1)^2 = 1 ∧
               ∀ a b : ℝ, (x - a)^2 + (y - b)^2 ≤ 1 → 
                          (m + 1) * a + (n + 1) * b - 2 ≠ 0) →
  (∀ p q : ℝ, p > 0 → q > 0 → 
    (∃ x y : ℝ, (p + 1) * x + (q + 1) * y - 2 = 0 ∧ 
                (x - 1)^2 + (y - 1)^2 = 1 ∧
                ∀ a b : ℝ, (x - a)^2 + (y - b)^2 ≤ 1 → 
                           (p + 1) * a + (q + 1) * b - 2 ≠ 0) →
    m + n ≤ p + q) →
  m + n = 2 + 2 * Real.sqrt 2 :=
by sorry

end tangent_line_circle_min_sum_l2430_243092


namespace min_abs_z_on_line_segment_l2430_243003

theorem min_abs_z_on_line_segment (z : ℂ) (h : Complex.abs (z - 6 * Complex.I) + Complex.abs (z - 5) = 7) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 30 / Real.sqrt 61 := by
  sorry

end min_abs_z_on_line_segment_l2430_243003


namespace factorial_sum_equality_l2430_243001

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 5 * Nat.factorial 5 = 5760 := by
  sorry

end factorial_sum_equality_l2430_243001


namespace balls_in_boxes_l2430_243000

/-- The number of ways to place balls into boxes -/
def place_balls (num_balls : ℕ) (num_boxes : ℕ) (max_per_box : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to place the balls -/
theorem balls_in_boxes : place_balls 3 4 2 = 16 := by
  sorry

end balls_in_boxes_l2430_243000


namespace missing_score_l2430_243033

theorem missing_score (scores : List ℕ) (mean : ℚ) : 
  scores = [73, 83, 86, 73] ∧ 
  mean = 79.2 ∧ 
  (scores.sum + (missing : ℕ)) / 5 = mean → 
  missing = 81 :=
by
  sorry

end missing_score_l2430_243033


namespace total_removed_volume_is_one_forty_eighth_l2430_243011

/-- A unit cube with corners cut off such that each face forms a regular hexagon -/
structure ModifiedCube where
  /-- The original cube is a unit cube -/
  is_unit_cube : Bool
  /-- Each face of the modified cube forms a regular hexagon -/
  faces_are_hexagons : Bool

/-- The volume of a single removed triangular pyramid -/
def single_pyramid_volume (cube : ModifiedCube) : ℝ :=
  sorry

/-- The total number of removed triangular pyramids -/
def num_pyramids : ℕ := 8

/-- The total volume of all removed triangular pyramids -/
def total_removed_volume (cube : ModifiedCube) : ℝ :=
  (single_pyramid_volume cube) * (num_pyramids : ℝ)

/-- Theorem: The total volume of removed triangular pyramids is 1/48 -/
theorem total_removed_volume_is_one_forty_eighth (cube : ModifiedCube) :
  cube.is_unit_cube ∧ cube.faces_are_hexagons →
  total_removed_volume cube = 1 / 48 :=
sorry

end total_removed_volume_is_one_forty_eighth_l2430_243011


namespace junk_mail_total_l2430_243050

/-- Calculates the total number of junk mail pieces a mailman should give --/
theorem junk_mail_total (houses_per_block : ℕ) (num_blocks : ℕ) (mail_per_house : ℕ) : 
  houses_per_block = 50 → num_blocks = 3 → mail_per_house = 45 → 
  houses_per_block * num_blocks * mail_per_house = 6750 := by
  sorry

#check junk_mail_total

end junk_mail_total_l2430_243050


namespace complex_magnitude_product_l2430_243086

theorem complex_magnitude_product : Complex.abs ((7 - 4*I) * (3 + 11*I)) = Real.sqrt 8450 := by sorry

end complex_magnitude_product_l2430_243086


namespace acute_inclination_implies_ab_negative_l2430_243089

-- Define a line with coefficients a, b, and c
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of having an acute angle of inclination
def hasAcuteInclination (l : Line) : Prop :=
  0 < -l.a / l.b ∧ -l.a / l.b < 1

-- Theorem statement
theorem acute_inclination_implies_ab_negative (l : Line) :
  hasAcuteInclination l → l.a * l.b < 0 := by
  sorry

end acute_inclination_implies_ab_negative_l2430_243089


namespace t_greater_than_a_squared_l2430_243079

/-- An equilateral triangle with a point on one of its sides -/
structure EquilateralTriangleWithPoint where
  a : ℝ  -- Side length of the equilateral triangle
  x : ℝ  -- Distance from A to P on side AB
  h1 : 0 < a  -- Side length is positive
  h2 : 0 ≤ x ∧ x ≤ a  -- P is on side AB

/-- The expression t = AP^2 + PB^2 + CP^2 -/
def t (triangle : EquilateralTriangleWithPoint) : ℝ :=
  let a := triangle.a
  let x := triangle.x
  x^2 + (a - x)^2 + (a^2 - a*x + x^2)

/-- Theorem: t is always greater than a^2 -/
theorem t_greater_than_a_squared (triangle : EquilateralTriangleWithPoint) :
  t triangle > triangle.a^2 := by
  sorry

end t_greater_than_a_squared_l2430_243079


namespace sqrt_eight_simplification_l2430_243067

theorem sqrt_eight_simplification : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_eight_simplification_l2430_243067


namespace eccentricity_range_l2430_243052

/-- An ellipse with center O and endpoint A of its major axis -/
structure Ellipse where
  center : ℝ × ℝ
  majorAxis : ℝ
  eccentricity : ℝ

/-- The condition that there is no point P on the ellipse such that ∠OPA = π/2 -/
def noRightAngle (e : Ellipse) : Prop :=
  ∀ p : ℝ × ℝ, p ≠ e.center → p ≠ (e.center.1 + e.majorAxis, e.center.2) →
    (p.1 - e.center.1)^2 + (p.2 - e.center.2)^2 = e.majorAxis^2 * (1 - e.eccentricity^2) →
    (p.1 - e.center.1) * (p.1 - (e.center.1 + e.majorAxis)) +
    (p.2 - e.center.2) * p.2 ≠ 0

/-- The theorem stating the range of eccentricity -/
theorem eccentricity_range (e : Ellipse) :
  0 < e.eccentricity ∧ e.eccentricity < 1 ∧ noRightAngle e →
  0 < e.eccentricity ∧ e.eccentricity ≤ Real.sqrt 2 / 2 :=
sorry

end eccentricity_range_l2430_243052


namespace dividend_calculation_l2430_243029

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 9) : 
  divisor * quotient + remainder = 162 := by
  sorry

end dividend_calculation_l2430_243029


namespace line_parallel_to_parallel_plane_l2430_243072

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (a b : Line) (α β : Plane)
  (diff_lines : a ≠ b)
  (diff_planes : α ≠ β)
  (h1 : parallel_planes α β)
  (h2 : contained_in a α) :
  parallel_line_plane a β :=
sorry

end line_parallel_to_parallel_plane_l2430_243072


namespace girls_fraction_in_class_l2430_243014

theorem girls_fraction_in_class (T G B : ℚ) (h1 : T > 0) (h2 : G > 0) (h3 : B > 0)
  (h4 : T = G + B) (h5 : B / G = 5 / 3) :
  ∃ X : ℚ, X * G = (1 / 4) * T ∧ X = 2 / 3 := by
sorry

end girls_fraction_in_class_l2430_243014


namespace quadratic_equations_solutions_l2430_243051

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - x₁ - 2 = 0 ∧ x₂^2 - x₂ - 2 = 0 ∧ x₁ = 2 ∧ x₂ = -1) ∧
  (∃ y₁ y₂ : ℝ, 2*y₁^2 + 2*y₁ - 1 = 0 ∧ 2*y₂^2 + 2*y₂ - 1 = 0 ∧ 
    y₁ = (-1 + Real.sqrt 3) / 2 ∧ y₂ = (-1 - Real.sqrt 3) / 2) :=
by sorry

end quadratic_equations_solutions_l2430_243051


namespace cylinder_volume_l2430_243049

/-- The volume of a cylinder given water displacement measurements -/
theorem cylinder_volume
  (initial_water_level : ℝ)
  (final_water_level : ℝ)
  (cylinder_min_marking : ℝ)
  (cylinder_max_marking : ℝ)
  (h1 : initial_water_level = 30)
  (h2 : final_water_level = 35)
  (h3 : cylinder_min_marking = 15)
  (h4 : cylinder_max_marking = 45) :
  let water_displaced := final_water_level - initial_water_level
  let cylinder_marking_range := cylinder_max_marking - cylinder_min_marking
  let submerged_proportion := (final_water_level - cylinder_min_marking) / cylinder_marking_range
  cylinder_marking_range / (final_water_level - cylinder_min_marking) * water_displaced = 7.5 :=
by sorry

end cylinder_volume_l2430_243049


namespace valid_numbers_l2430_243048

def is_valid_number (n : ℕ) : Prop :=
  523000 ≤ n ∧ n ≤ 523999 ∧ n % 7 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 523152 ∨ n = 523656 := by sorry

end valid_numbers_l2430_243048


namespace area_swept_specific_triangle_l2430_243025

/-- Represents a triangle with sides and height -/
structure Triangle where
  bc : ℝ
  ab : ℝ
  ad : ℝ

/-- Calculates the area swept by a triangle moving upward -/
def area_swept (t : Triangle) (speed : ℝ) (time : ℝ) : ℝ :=
  sorry

/-- Theorem stating the area swept by the specific triangle -/
theorem area_swept_specific_triangle :
  let t : Triangle := { bc := 6, ab := 5, ad := 4 }
  area_swept t 3 2 = 66 := by sorry

end area_swept_specific_triangle_l2430_243025


namespace bus_assignment_count_l2430_243045

def num_boys : ℕ := 6
def num_girls : ℕ := 4
def num_buses : ℕ := 5
def attendants_per_bus : ℕ := 2

theorem bus_assignment_count : 
  (Nat.choose num_buses 3) * 
  (Nat.factorial num_boys / (Nat.factorial attendants_per_bus ^ 3)) * 
  (Nat.factorial num_girls / (Nat.factorial attendants_per_bus ^ 2)) * 
  (1 / Nat.factorial 3) * 
  (1 / Nat.factorial 2) * 
  Nat.factorial num_buses = 54000 := by
sorry

end bus_assignment_count_l2430_243045


namespace song_book_cost_l2430_243036

def trumpet_cost : ℝ := 149.16
def music_tool_cost : ℝ := 9.98
def total_spent : ℝ := 163.28

theorem song_book_cost :
  total_spent - (trumpet_cost + music_tool_cost) = 4.14 := by sorry

end song_book_cost_l2430_243036


namespace quadratic_polynomial_half_coefficient_integer_values_l2430_243071

theorem quadratic_polynomial_half_coefficient_integer_values :
  ∃ (b c : ℚ), ∀ (x : ℤ), ∃ (y : ℤ), ((1/2 : ℚ) * x^2 + b * x + c : ℚ) = y := by
  sorry

end quadratic_polynomial_half_coefficient_integer_values_l2430_243071


namespace systematic_sample_valid_l2430_243070

/-- Checks if a list of integers forms a valid systematic sample -/
def is_valid_systematic_sample (population_size : ℕ) (sample_size : ℕ) (sample : List ℕ) : Prop :=
  let interval := population_size / sample_size
  sample.length = sample_size ∧
  ∀ i j, i < j → j < sample.length →
    sample[j]! - sample[i]! = (j - i) * interval

theorem systematic_sample_valid :
  let population_size := 50
  let sample_size := 5
  let sample := [3, 13, 23, 33, 43]
  is_valid_systematic_sample population_size sample_size sample := by
  sorry

end systematic_sample_valid_l2430_243070


namespace change_in_expression_l2430_243074

theorem change_in_expression (x : ℝ) (b : ℕ+) : 
  let f : ℝ → ℝ := λ t => t^2 - 5*t + 6
  (f (x + b) - f x = 2*b*x + b^2 - 5*b) ∧ 
  (f (x - b) - f x = -2*b*x + b^2 + 5*b) := by
  sorry

end change_in_expression_l2430_243074


namespace complex_equation_real_solution_l2430_243017

theorem complex_equation_real_solution (a : ℝ) : 
  (((a : ℂ) / (1 + Complex.I)) + ((1 + Complex.I) / 2)).im = 0 → a = 1 := by
  sorry

end complex_equation_real_solution_l2430_243017


namespace recurrence_equals_explicit_l2430_243084

def recurrence_sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 5
  | 1 => 10
  | n + 2 => 5 * recurrence_sequence (n + 1) - 6 * recurrence_sequence n + 2 * (n + 2) - 3

def explicit_form (n : ℕ) : ℤ :=
  2^(n + 1) + 3^n + n + 2

theorem recurrence_equals_explicit : ∀ n : ℕ, recurrence_sequence n = explicit_form n :=
  sorry

end recurrence_equals_explicit_l2430_243084


namespace max_a_value_l2430_243063

theorem max_a_value (a : ℝ) : (∀ x : ℝ, x * a ≤ Real.exp (x - 1) + x^2 + 1) → a ≤ 3 := by
  sorry

end max_a_value_l2430_243063


namespace lg_sqrt_sum_l2430_243069

-- Define lg as the base 10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_sqrt_sum : lg (Real.sqrt 5) + lg (Real.sqrt 20) = 1 := by
  sorry

end lg_sqrt_sum_l2430_243069


namespace total_books_total_books_specific_l2430_243073

theorem total_books (stu_books : ℕ) (albert_multiplier : ℕ) : ℕ :=
  let albert_books := albert_multiplier * stu_books
  stu_books + albert_books

theorem total_books_specific : total_books 9 4 = 45 := by
  sorry

end total_books_total_books_specific_l2430_243073


namespace savings_calculation_l2430_243087

theorem savings_calculation (savings : ℚ) (tv_cost : ℚ) 
  (h1 : tv_cost = 150)
  (h2 : (1 : ℚ) / 4 * savings = tv_cost) : 
  savings = 600 := by
  sorry

end savings_calculation_l2430_243087


namespace right_triangle_hypotenuse_l2430_243024

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive sides
  a + b + c = 40 →  -- perimeter condition
  (1/2) * a * b = 30 →  -- area condition
  a^2 + b^2 = c^2 →  -- right triangle (Pythagorean theorem)
  c = 18.5 := by
sorry

end right_triangle_hypotenuse_l2430_243024


namespace prob_two_heads_in_three_fair_coin_l2430_243099

/-- A fair coin is a coin with probability 1/2 of landing heads -/
def fairCoin (p : ℝ) : Prop := p = (1 : ℝ) / 2

/-- The probability of getting exactly two heads in three independent coin flips -/
def probTwoHeadsInThree (p : ℝ) : ℝ := 3 * p^2 * (1 - p)

/-- Theorem: The probability of getting exactly two heads in three flips of a fair coin is 3/8 -/
theorem prob_two_heads_in_three_fair_coin :
  ∀ p : ℝ, fairCoin p → probTwoHeadsInThree p = (3 : ℝ) / 8 :=
by sorry

end prob_two_heads_in_three_fair_coin_l2430_243099


namespace sum_of_decimal_and_fraction_l2430_243035

theorem sum_of_decimal_and_fraction : 7.31 + (1 / 5 : ℚ) = 7.51 := by sorry

end sum_of_decimal_and_fraction_l2430_243035


namespace researchers_distribution_l2430_243046

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    with at least one object in each box. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of schools -/
def num_schools : ℕ := 3

/-- The number of researchers -/
def num_researchers : ℕ := 4

/-- The theorem stating that the number of ways to distribute 4 researchers
    to 3 schools, with at least one researcher in each school, is 36. -/
theorem researchers_distribution :
  distribute num_researchers num_schools = 36 := by sorry

end researchers_distribution_l2430_243046


namespace salary_after_changes_l2430_243010

def initial_salary : ℝ := 3000
def raise_percentage : ℝ := 0.15
def cut_percentage : ℝ := 0.25

theorem salary_after_changes : 
  initial_salary * (1 + raise_percentage) * (1 - cut_percentage) = 2587.5 := by
  sorry

end salary_after_changes_l2430_243010


namespace area_distance_relation_l2430_243002

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  a : ℝ  -- Length of the rectangle
  b : ℝ  -- Width of the rectangle
  t : ℝ  -- Area of the original rectangle
  t₁ : ℝ  -- Area of the first smaller rectangle
  t₂ : ℝ  -- Area of the second smaller rectangle
  t₃ : ℝ  -- Area of the third smaller rectangle
  t₄ : ℝ  -- Area of the fourth smaller rectangle
  z : ℝ  -- Distance from the center of the original rectangle to line e
  z₁ : ℝ  -- Distance from the center of the first smaller rectangle to line e
  z₂ : ℝ  -- Distance from the center of the second smaller rectangle to line e
  z₃ : ℝ  -- Distance from the center of the third smaller rectangle to line e
  z₄ : ℝ  -- Distance from the center of the fourth smaller rectangle to line e
  h_positive : a > 0 ∧ b > 0  -- Ensure positive dimensions
  h_area : t = a * b  -- Area of the original rectangle
  h_sum_areas : t = t₁ + t₂ + t₃ + t₄  -- Sum of areas of smaller rectangles

/-- The theorem stating the relationship between areas and distances -/
theorem area_distance_relation (r : DividedRectangle) :
    r.t₁ * r.z₁ + r.t₂ * r.z₂ + r.t₃ * r.z₃ + r.t₄ * r.z₄ = r.t * r.z := by
  sorry

end area_distance_relation_l2430_243002


namespace locus_is_apollonian_circle_l2430_243005

/-- An Apollonian circle is the locus of points with a constant ratio of distances to two fixed points. -/
def ApollonianCircle (A B : ℝ × ℝ) (k : ℝ) : Set (ℝ × ℝ) :=
  {M | dist A M / dist M B = k}

/-- The locus of points M satisfying |AM| : |MB| = k ≠ 1, where A and B are fixed points, is an Apollonian circle. -/
theorem locus_is_apollonian_circle (A B : ℝ × ℝ) (k : ℝ) (h : k ≠ 1) :
  {M : ℝ × ℝ | dist A M / dist M B = k} = ApollonianCircle A B k := by
  sorry

end locus_is_apollonian_circle_l2430_243005


namespace BA_equals_AB_l2430_243080

-- Define the matrices A and B
variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

-- Define the given conditions
def condition1 : Prop := A + B = A * B
def condition2 : Prop := A * B = !![12, -6; 9, -3]

-- State the theorem
theorem BA_equals_AB (h1 : condition1 A B) (h2 : condition2 A B) : 
  B * A = !![12, -6; 9, -3] := by sorry

end BA_equals_AB_l2430_243080


namespace division_problem_l2430_243037

theorem division_problem (k : ℕ) (h : k = 14) : 56 / k = 4 := by
  sorry

end division_problem_l2430_243037


namespace quadrilateral_is_right_angled_trapezoid_l2430_243059

/-- A quadrilateral in 2D space --/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Vector from point P to point Q --/
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Dot product of two 2D vectors --/
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Definition of a right-angled trapezoid --/
def is_right_angled_trapezoid (q : Quadrilateral) : Prop :=
  ∃ (k : ℝ), k ≠ 1 ∧ vec q.A q.B = k • (vec q.D q.C) ∧
  dot (vec q.A q.D) (vec q.A q.B) = 0

/-- The main theorem --/
theorem quadrilateral_is_right_angled_trapezoid (q : Quadrilateral) 
  (h1 : vec q.A q.B = 2 • (vec q.D q.C))
  (h2 : dot (vec q.C q.D - vec q.C q.A) (vec q.A q.B) = 0) :
  is_right_angled_trapezoid q := by
  sorry

end quadrilateral_is_right_angled_trapezoid_l2430_243059


namespace bowling_ball_volume_l2430_243085

/-- The volume of a sphere with two cylindrical holes drilled into it -/
theorem bowling_ball_volume 
  (sphere_diameter : ℝ) 
  (hole1_depth hole1_diameter : ℝ) 
  (hole2_depth hole2_diameter : ℝ) 
  (h1 : sphere_diameter = 24)
  (h2 : hole1_depth = 6)
  (h3 : hole1_diameter = 3)
  (h4 : hole2_depth = 6)
  (h5 : hole2_diameter = 4) : 
  (4 / 3 * π * (sphere_diameter / 2) ^ 3) - 
  (π * (hole1_diameter / 2) ^ 2 * hole1_depth) - 
  (π * (hole2_diameter / 2) ^ 2 * hole2_depth) = 2266.5 * π := by
  sorry

end bowling_ball_volume_l2430_243085


namespace apples_in_basket_after_removal_l2430_243047

/-- Given a total number of apples and baskets, and a number of apples removed from each basket,
    calculate the number of apples remaining in each basket. -/
def applesPerBasket (totalApples : ℕ) (numBaskets : ℕ) (applesRemoved : ℕ) : ℕ :=
  (totalApples / numBaskets) - applesRemoved

/-- Theorem stating that for the given problem, each basket contains 9 apples after removal. -/
theorem apples_in_basket_after_removal :
  applesPerBasket 128 8 7 = 9 := by
  sorry

end apples_in_basket_after_removal_l2430_243047


namespace tan_theta_in_terms_of_x_l2430_243012

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2) 
  (h_x : x > 1) 
  (h_cos : Real.cos (θ / 2) = Real.sqrt ((x - 1) / (2 * x))) : 
  Real.tan θ = -x * Real.sqrt (1 - 1 / x^2) := by
  sorry

end tan_theta_in_terms_of_x_l2430_243012


namespace dihedral_angle_at_apex_is_45_degrees_l2430_243062

/-- A regular square pyramid with coinciding centers of inscribed and circumscribed spheres -/
structure RegularSquarePyramid where
  /-- The centers of the inscribed and circumscribed spheres coincide -/
  coinciding_centers : Bool

/-- The dihedral angle at the apex of the pyramid -/
def dihedral_angle_at_apex (p : RegularSquarePyramid) : ℝ :=
  sorry

/-- Theorem: The dihedral angle at the apex of a regular square pyramid 
    with coinciding centers of inscribed and circumscribed spheres is 45° -/
theorem dihedral_angle_at_apex_is_45_degrees (p : RegularSquarePyramid) 
    (h : p.coinciding_centers = true) : 
    dihedral_angle_at_apex p = 45 := by
  sorry

end dihedral_angle_at_apex_is_45_degrees_l2430_243062


namespace experts_win_probability_l2430_243088

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Viewers winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def expert_score : ℕ := 3

/-- The current score of Viewers -/
def viewer_score : ℕ := 4

/-- The number of rounds needed to win the game -/
def winning_score : ℕ := 6

/-- The probability that the Experts will eventually win the game -/
theorem experts_win_probability : 
  p^4 + 4 * p^3 * q = 0.4752 := by sorry

end experts_win_probability_l2430_243088


namespace range_of_m_l2430_243013

/-- Statement p: For any real number x, the inequality x^2 - 2x + m ≥ 0 always holds -/
def statement_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m ≥ 0

/-- Statement q: The equation (x^2)/(m-3) - (y^2)/m = 1 represents a hyperbola with foci on the x-axis -/
def statement_q (m : ℝ) : Prop :=
  m > 3 ∧ ∀ x y : ℝ, x^2 / (m - 3) - y^2 / m = 1

/-- The range of m when "p ∨ q" is true and "p ∧ q" is false -/
theorem range_of_m :
  ∀ m : ℝ, (statement_p m ∨ statement_q m) ∧ ¬(statement_p m ∧ statement_q m) →
  1 ≤ m ∧ m ≤ 3 :=
by sorry

end range_of_m_l2430_243013


namespace alcohol_mixture_proof_l2430_243031

/-- Proves that adding 300 mL of solution Y to 100 mL of solution X
    creates a solution that is 25% alcohol by volume. -/
theorem alcohol_mixture_proof 
  (x_conc : Real) -- Concentration of alcohol in solution X
  (y_conc : Real) -- Concentration of alcohol in solution Y
  (x_vol : Real)  -- Volume of solution X
  (y_vol : Real)  -- Volume of solution Y to be added
  (h1 : x_conc = 0.10) -- Solution X is 10% alcohol
  (h2 : y_conc = 0.30) -- Solution Y is 30% alcohol
  (h3 : x_vol = 100)   -- We start with 100 mL of solution X
  (h4 : y_vol = 300)   -- We add 300 mL of solution Y
  : (x_conc * x_vol + y_conc * y_vol) / (x_vol + y_vol) = 0.25 := by
  sorry

#check alcohol_mixture_proof

end alcohol_mixture_proof_l2430_243031


namespace sector_area_l2430_243078

/-- The area of a sector with a central angle of 60° in a circle passing through two given points -/
theorem sector_area (P Q : ℝ × ℝ) (h : P = (2, -2) ∧ Q = (8, 6)) : 
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (1/6 : ℝ) * π * r^2 = 50*π/3 := by sorry

end sector_area_l2430_243078


namespace volume_of_sphere_containing_pyramid_l2430_243015

/-- Regular triangular pyramid with base on sphere -/
structure RegularTriangularPyramid where
  /-- Base edge length -/
  baseEdge : ℝ
  /-- Volume of the pyramid -/
  volume : ℝ
  /-- Radius of the circumscribed sphere -/
  sphereRadius : ℝ

/-- Theorem: Volume of sphere containing regular triangular pyramid -/
theorem volume_of_sphere_containing_pyramid (p : RegularTriangularPyramid) 
  (h1 : p.baseEdge = 2 * Real.sqrt 3)
  (h2 : p.volume = Real.sqrt 3) :
  (4 / 3) * Real.pi * p.sphereRadius ^ 3 = (20 * Real.sqrt 5 * Real.pi) / 3 := by
  sorry

end volume_of_sphere_containing_pyramid_l2430_243015


namespace least_value_with_specific_remainders_l2430_243091

theorem least_value_with_specific_remainders :
  ∃ (N : ℕ), 
    N > 0 ∧
    N % 6 = 5 ∧
    N % 5 = 4 ∧
    N % 4 = 3 ∧
    N % 3 = 2 ∧
    N % 2 = 1 ∧
    (∀ (M : ℕ), M > 0 ∧ 
      M % 6 = 5 ∧
      M % 5 = 4 ∧
      M % 4 = 3 ∧
      M % 3 = 2 ∧
      M % 2 = 1 → M ≥ N) ∧
    N = 59 :=
by sorry

end least_value_with_specific_remainders_l2430_243091


namespace sum_reciprocals_equals_six_l2430_243034

theorem sum_reciprocals_equals_six (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * a * b) :
  1 / a + 1 / b = 6 := by
sorry

end sum_reciprocals_equals_six_l2430_243034


namespace simplify_expressions_l2430_243082

variable (a b : ℝ)

theorem simplify_expressions :
  (4 * a^3 - a^2 + 1 - a^2 - 2 * a^3 = 2 * a^3 - 2 * a^2 + 1) ∧
  (2 * a - 3 * (5 * a - b) + 7 * (a + 2 * b) = -6 * a + 17 * b) := by
  sorry

end simplify_expressions_l2430_243082


namespace sum_of_numbers_ge_threshold_l2430_243075

theorem sum_of_numbers_ge_threshold : 
  let numbers : List ℝ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let threshold : ℝ := 1.1
  (numbers.filter (λ x => x ≥ threshold)).sum = 3.9 := by
sorry

end sum_of_numbers_ge_threshold_l2430_243075


namespace yellow_pencils_count_l2430_243076

/-- Represents a grid of colored pencils -/
structure PencilGrid :=
  (size : ℕ)
  (perimeter_color : String)
  (inside_color : String)

/-- Calculates the number of pencils of the inside color in the grid -/
def count_inside_pencils (grid : PencilGrid) : ℕ :=
  grid.size * grid.size - (4 * grid.size - 4)

/-- The theorem to be proved -/
theorem yellow_pencils_count (grid : PencilGrid) 
  (h1 : grid.size = 10)
  (h2 : grid.perimeter_color = "red")
  (h3 : grid.inside_color = "yellow") :
  count_inside_pencils grid = 64 := by
  sorry

end yellow_pencils_count_l2430_243076


namespace circle_partition_exists_l2430_243083

/-- Represents a person with their country and position -/
structure Person where
  country : Fin 25
  position : Fin 100

/-- Defines the arrangement of people in a circle -/
def arrangement : Fin 100 → Person :=
  sorry

/-- Checks if two people are adjacent in the circle -/
def are_adjacent (p1 p2 : Person) : Prop :=
  sorry

/-- Represents a partition of people into 4 groups -/
def Partition := Fin 100 → Fin 4

/-- Checks if a partition is valid according to the problem conditions -/
def is_valid_partition (p : Partition) : Prop :=
  ∀ i j : Fin 100,
    i ≠ j →
    (arrangement i).country = (arrangement j).country ∨ are_adjacent (arrangement i) (arrangement j) →
    p i ≠ p j

theorem circle_partition_exists :
  ∃ p : Partition, is_valid_partition p :=
sorry

end circle_partition_exists_l2430_243083


namespace fair_coin_same_side_five_tosses_l2430_243020

/-- A fair coin is a coin with equal probability of landing on either side -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of a sequence of independent events -/
def prob_sequence (p : ℝ) (n : ℕ) : ℝ := p ^ n

/-- The number of tosses -/
def num_tosses : ℕ := 5

/-- Theorem: The probability of a fair coin landing on the same side for 5 tosses is 1/32 -/
theorem fair_coin_same_side_five_tosses (p : ℝ) (h : fair_coin p) :
  prob_sequence p num_tosses = 1 / 32 := by
  sorry


end fair_coin_same_side_five_tosses_l2430_243020


namespace find_number_l2430_243068

/-- Given two positive integers with specific LCM and HCF, prove one number given the other -/
theorem find_number (A B : ℕ+) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 30) (h3 : B = 150) :
  A = 462 := by
  sorry

end find_number_l2430_243068


namespace waiter_tip_calculation_l2430_243026

theorem waiter_tip_calculation (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tips : ℕ) :
  total_customers = 7 →
  non_tipping_customers = 4 →
  total_tips = 27 →
  (total_tips / (total_customers - non_tipping_customers) : ℚ) = 9 := by
  sorry

end waiter_tip_calculation_l2430_243026


namespace max_a4b4_l2430_243096

/-- Given an arithmetic sequence a and a geometric sequence b satisfying
    certain conditions, the maximum value of a₄b₄ is 37/4 -/
theorem max_a4b4 (a b : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_geom : ∀ n, b (n + 1) / b n = b (n + 2) / b (n + 1))
  (h1 : a 1 * b 1 = 20)
  (h2 : a 2 * b 2 = 19)
  (h3 : a 3 * b 3 = 14) :
  (∀ x, a 4 * b 4 ≤ x) → x = 37/4 :=
sorry

end max_a4b4_l2430_243096


namespace tangent_lines_with_slope_one_to_cubic_l2430_243060

/-- The number of tangent lines with slope 1 to the curve y = x³ -/
theorem tangent_lines_with_slope_one_to_cubic (x : ℝ) :
  (∃ m : ℝ, 3 * m^2 = 1) ∧ (∀ m₁ m₂ : ℝ, 3 * m₁^2 = 1 ∧ 3 * m₂^2 = 1 → m₁ = m₂ ∨ m₁ = -m₂) :=
by sorry

end tangent_lines_with_slope_one_to_cubic_l2430_243060


namespace smallest_n0_for_inequality_l2430_243097

theorem smallest_n0_for_inequality : ∃ (n0 : ℕ), n0 = 5 ∧ 
  (∀ n : ℕ, n ≥ n0 → 2^n > n^2 + 1) ∧ 
  (∀ m : ℕ, m < n0 → ¬(2^m > m^2 + 1)) :=
sorry

end smallest_n0_for_inequality_l2430_243097


namespace man_walked_40_minutes_l2430_243042

/-- Represents the scenario of a man meeting his wife at the train station and going home. -/
structure TrainScenario where
  T : ℕ  -- usual arrival time at the station
  X : ℕ  -- usual driving time from station to home

/-- Calculates the time spent walking in the given scenario. -/
def time_walking (s : TrainScenario) : ℕ :=
  s.X - 40

/-- Theorem stating that the man spent 40 minutes walking. -/
theorem man_walked_40_minutes (s : TrainScenario) :
  time_walking s = 40 :=
by
  sorry


end man_walked_40_minutes_l2430_243042


namespace karlson_candies_theorem_l2430_243021

/-- Represents the maximum number of candies Karlson can eat -/
def max_candies (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating that the maximum number of candies Karlson can eat with 31 initial ones is 465 -/
theorem karlson_candies_theorem :
  max_candies 31 = 465 := by
  sorry

#eval max_candies 31

end karlson_candies_theorem_l2430_243021


namespace cube_minus_cylinder_volume_l2430_243028

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_minus_cylinder_volume (cube_side : ℝ) (cylinder_radius : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  cube_side ^ 3 - π * cylinder_radius ^ 2 * cube_side = 216 - 54 * π := by
  sorry

end cube_minus_cylinder_volume_l2430_243028


namespace upstream_distance_is_96_l2430_243055

/-- Represents the boat's journey on a river -/
structure RiverJourney where
  boatSpeed : ℝ
  riverSpeed : ℝ
  downstreamDistance : ℝ
  downstreamTime : ℝ
  upstreamTime : ℝ

/-- Calculates the upstream distance for a given river journey -/
def upstreamDistance (journey : RiverJourney) : ℝ :=
  (journey.boatSpeed - journey.riverSpeed) * journey.upstreamTime

/-- Theorem stating that for the given conditions, the upstream distance is 96 km -/
theorem upstream_distance_is_96 (journey : RiverJourney) 
  (h1 : journey.boatSpeed = 14)
  (h2 : journey.downstreamDistance = 200)
  (h3 : journey.downstreamTime = 10)
  (h4 : journey.upstreamTime = 12)
  (h5 : journey.downstreamDistance = (journey.boatSpeed + journey.riverSpeed) * journey.downstreamTime) :
  upstreamDistance journey = 96 := by
  sorry

end upstream_distance_is_96_l2430_243055


namespace stone_exit_and_return_velocity_range_l2430_243093

/-- 
Theorem: Stone Exit and Return Velocity Range

For a stone thrown upwards in a well with the following properties:
- Well depth: h = 10 meters
- Cover cycle: opens for 1 second, closes for 1 second
- Stone thrown 0.5 seconds before cover opens
- Acceleration due to gravity: g = 10 m/s²

The initial velocities V for which the stone will exit the well and fall back onto the cover
are in the range (85/6, 33/2) ∪ (285/14, 45/2).
-/
theorem stone_exit_and_return_velocity_range (h g τ : ℝ) (V : ℝ) : 
  h = 10 → 
  g = 10 → 
  τ = 1 → 
  (V ∈ Set.Ioo (85/6) (33/2) ∪ Set.Ioo (285/14) (45/2)) ↔ 
  (∃ t : ℝ, 
    t > 0 ∧ 
    V * t - (1/2) * g * t^2 ≥ h ∧
    ∃ t' : ℝ, t' > t ∧ V * t' - (1/2) * g * t'^2 = 0 ∧
    (∃ n : ℕ, t' = (2*n + 3/2) * τ ∨ t' = (2*n + 7/2) * τ)) :=
by sorry

end stone_exit_and_return_velocity_range_l2430_243093


namespace prob_three_odds_eq_4_35_l2430_243098

/-- The set of numbers from which we select -/
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The set of odd numbers in S -/
def odds : Finset ℕ := S.filter (fun n => n % 2 = 1)

/-- The number of elements to select -/
def k : ℕ := 3

/-- The probability of selecting three distinct odd numbers from S -/
theorem prob_three_odds_eq_4_35 : 
  (Finset.card (odds.powersetCard k)) / (Finset.card (S.powersetCard k)) = 4 / 35 := by
  sorry

end prob_three_odds_eq_4_35_l2430_243098


namespace squad_size_problem_l2430_243030

theorem squad_size_problem (total : ℕ) (transfer : ℕ) 
  (h1 : total = 146) (h2 : transfer = 11) : 
  (∃ (first second : ℕ), 
    first + second = total ∧ 
    first - transfer = second + transfer ∧
    first = 84 ∧ 
    second = 62) := by
  sorry

end squad_size_problem_l2430_243030


namespace inequality_of_positive_numbers_l2430_243081

theorem inequality_of_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^2 * b + a * b^2 ≤ a^3 + b^3 := by
  sorry

end inequality_of_positive_numbers_l2430_243081


namespace problem_book_solution_l2430_243007

/-- The number of problems solved by Taeyeon and Yura -/
def total_problems_solved (taeyeon_per_day : ℕ) (taeyeon_days : ℕ) (yura_per_day : ℕ) (yura_days : ℕ) : ℕ :=
  taeyeon_per_day * taeyeon_days + yura_per_day * yura_days

/-- Theorem stating that Taeyeon and Yura solved 262 problems in total -/
theorem problem_book_solution :
  total_problems_solved 16 7 25 6 = 262 := by
  sorry

end problem_book_solution_l2430_243007


namespace cubic_equation_unique_solution_l2430_243039

theorem cubic_equation_unique_solution :
  ∃! (x : ℤ), x^3 + (x+1)^3 + (x+2)^3 = (x+3)^3 ∧ x = 3 := by
  sorry

end cubic_equation_unique_solution_l2430_243039


namespace horner_rule_f_3_l2430_243094

/-- Horner's Rule evaluation of a polynomial -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [5, 4, 3, 2, 1, 1]

theorem horner_rule_f_3 :
  f 3 = horner_eval f_coeffs 3 ∧ horner_eval f_coeffs 3 = 1642 :=
sorry

end horner_rule_f_3_l2430_243094


namespace third_month_sale_l2430_243066

def average_sale : ℕ := 3500
def number_of_months : ℕ := 6
def sale_month1 : ℕ := 3435
def sale_month2 : ℕ := 3920
def sale_month4 : ℕ := 4230
def sale_month5 : ℕ := 3560
def sale_month6 : ℕ := 2000

theorem third_month_sale :
  let total_sales := average_sale * number_of_months
  let known_sales := sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 3855 := by
sorry

end third_month_sale_l2430_243066


namespace expansion_term_count_l2430_243006

/-- The number of terms in a polynomial -/
def num_terms (p : Polynomial ℚ) : ℕ := sorry

/-- The expansion of the product of two polynomials -/
def expand_product (p q : Polynomial ℚ) : Polynomial ℚ := sorry

theorem expansion_term_count :
  let p := X + Y + Z
  let q := U + V + W + X
  num_terms (expand_product p q) = 12 := by sorry

end expansion_term_count_l2430_243006


namespace matthew_crackers_left_l2430_243095

/-- Calculates the number of crackers Matthew has left after distributing them to friends and the friends eating some. -/
def crackers_left (initial_crackers : ℕ) (num_friends : ℕ) (crackers_eaten_per_friend : ℕ) : ℕ :=
  let distributed_crackers := initial_crackers - 1
  let crackers_per_friend := distributed_crackers / num_friends
  let remaining_with_friends := (crackers_per_friend - crackers_eaten_per_friend) * num_friends
  1 + remaining_with_friends

/-- Proves that Matthew has 11 crackers left given the initial conditions. -/
theorem matthew_crackers_left :
  crackers_left 23 2 6 = 11 := by
  sorry

end matthew_crackers_left_l2430_243095


namespace solid_max_volume_l2430_243044

/-- The side length of each cube in centimeters -/
def cube_side_length : ℝ := 3

/-- The number of cubes in the base layer -/
def base_layer_cubes : ℕ := 4 * 4

/-- The number of cubes in the second layer -/
def second_layer_cubes : ℕ := 2 * 2

/-- The total number of cubes in the solid -/
def total_cubes : ℕ := base_layer_cubes + second_layer_cubes

/-- The volume of a single cube in cubic centimeters -/
def single_cube_volume : ℝ := cube_side_length ^ 3

/-- The maximum volume of the solid in cubic centimeters -/
def max_volume : ℝ := (total_cubes : ℝ) * single_cube_volume

theorem solid_max_volume : max_volume = 540 := by sorry

end solid_max_volume_l2430_243044


namespace minimum_buses_needed_l2430_243090

def students : ℕ := 535
def bus_capacity : ℕ := 45

theorem minimum_buses_needed : 
  ∃ (n : ℕ), n * bus_capacity ≥ students ∧ 
  ∀ (m : ℕ), m * bus_capacity ≥ students → n ≤ m :=
by sorry

end minimum_buses_needed_l2430_243090


namespace infinite_solutions_imply_d_equals_five_l2430_243009

theorem infinite_solutions_imply_d_equals_five (d : ℝ) :
  (∀ (S : Set ℝ), S.Infinite → (∀ x ∈ S, 3 * (5 + d * x) = 15 * x + 15)) →
  d = 5 := by
sorry

end infinite_solutions_imply_d_equals_five_l2430_243009


namespace fixed_fee_is_7_42_l2430_243018

/-- Represents the billing structure and usage for an online service provider -/
structure BillingInfo where
  fixedFee : ℝ
  hourlyCharge : ℝ
  decemberUsage : ℝ
  januaryUsage : ℝ

/-- Calculates the total bill based on fixed fee, hourly charge, and usage -/
def calculateBill (info : BillingInfo) (usage : ℝ) : ℝ :=
  info.fixedFee + info.hourlyCharge * usage

/-- Theorem stating that under given conditions, the fixed monthly fee is $7.42 -/
theorem fixed_fee_is_7_42 (info : BillingInfo) :
  calculateBill info info.decemberUsage = 12.48 →
  calculateBill info info.januaryUsage = 17.54 →
  info.januaryUsage = 2 * info.decemberUsage →
  info.fixedFee = 7.42 := by
  sorry

#eval (7.42 : Float)

end fixed_fee_is_7_42_l2430_243018


namespace trigonometric_identity_l2430_243019

theorem trigonometric_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end trigonometric_identity_l2430_243019


namespace triangle_isosceles_or_right_angled_l2430_243023

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

/-- The theorem stating that if a/cos(B) = b/cos(A) in a triangle, 
    then the triangle is either isosceles or right-angled. -/
theorem triangle_isosceles_or_right_angled (t : Triangle) 
  (h : t.a / Real.cos t.B = t.b / Real.cos t.A) : 
  (t.A = t.B) ∨ (t.C = π/2) := by
  sorry


end triangle_isosceles_or_right_angled_l2430_243023


namespace freshmen_in_liberal_arts_l2430_243004

theorem freshmen_in_liberal_arts 
  (total_students : ℝ) 
  (freshmen_ratio : ℝ) 
  (psych_majors_ratio : ℝ) 
  (freshmen_psych_lib_arts_ratio : ℝ) 
  (h1 : freshmen_ratio = 0.4)
  (h2 : psych_majors_ratio = 0.5)
  (h3 : freshmen_psych_lib_arts_ratio = 0.1) :
  (freshmen_psych_lib_arts_ratio * total_students) / (psych_majors_ratio * (freshmen_ratio * total_students)) = 0.5 := by
  sorry

end freshmen_in_liberal_arts_l2430_243004


namespace shifted_data_invariants_l2430_243008

variable {n : ℕ}
variable (X Y : Fin n → ℝ)
variable (c : ℝ)

def is_shifted (X Y : Fin n → ℝ) (c : ℝ) : Prop :=
  ∀ i, Y i = X i + c

def standard_deviation (X : Fin n → ℝ) : ℝ := sorry

def range (X : Fin n → ℝ) : ℝ := sorry

theorem shifted_data_invariants (h : is_shifted X Y c) (h_nonzero : c ≠ 0) :
  standard_deviation Y = standard_deviation X ∧ range Y = range X := by sorry

end shifted_data_invariants_l2430_243008


namespace world_expo_ticket_sales_l2430_243016

theorem world_expo_ticket_sales :
  let regular_price : ℕ := 200
  let concession_price : ℕ := 120
  let total_tickets : ℕ := 1200
  let total_revenue : ℕ := 216000
  ∃ (regular_tickets concession_tickets : ℕ),
    regular_tickets + concession_tickets = total_tickets ∧
    regular_tickets * regular_price + concession_tickets * concession_price = total_revenue ∧
    regular_tickets = 900 ∧
    concession_tickets = 300 := by
sorry

end world_expo_ticket_sales_l2430_243016


namespace birthday_cookies_l2430_243077

theorem birthday_cookies (friends : ℕ) (packages : ℕ) (cookies_per_package : ℕ) :
  friends = 7 →
  packages = 5 →
  cookies_per_package = 36 →
  (packages * cookies_per_package) / (friends + 1) = 22 :=
by
  sorry

end birthday_cookies_l2430_243077


namespace fixed_point_of_exponential_function_l2430_243040

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end fixed_point_of_exponential_function_l2430_243040


namespace betty_order_cost_l2430_243027

/-- The total cost of Betty's order -/
def total_cost (slipper_price lipstick_price hair_color_price sunglasses_price tshirt_price : ℚ) 
  (slipper_qty lipstick_qty hair_color_qty sunglasses_qty tshirt_qty : ℕ) : ℚ :=
  slipper_price * slipper_qty + 
  lipstick_price * lipstick_qty + 
  hair_color_price * hair_color_qty + 
  sunglasses_price * sunglasses_qty + 
  tshirt_price * tshirt_qty

/-- The theorem stating that Betty's total order cost is $110.25 -/
theorem betty_order_cost : 
  total_cost 2.5 1.25 3 5.75 12.25 6 4 8 3 4 = 110.25 := by
  sorry

end betty_order_cost_l2430_243027


namespace same_number_on_four_dice_l2430_243041

theorem same_number_on_four_dice (n : ℕ) (h : n = 8) :
  (1 : ℚ) / (n ^ 3) = 1 / 512 :=
by sorry

end same_number_on_four_dice_l2430_243041


namespace unreserved_seat_cost_l2430_243053

theorem unreserved_seat_cost (total_revenue : ℚ) (reserved_seat_cost : ℚ) 
  (reserved_tickets : ℕ) (unreserved_tickets : ℕ) :
  let unreserved_seat_cost := (total_revenue - reserved_seat_cost * reserved_tickets) / unreserved_tickets
  total_revenue = 26170 ∧ 
  reserved_seat_cost = 25 ∧ 
  reserved_tickets = 246 ∧ 
  unreserved_tickets = 246 → 
  unreserved_seat_cost = 81.3 := by
sorry

end unreserved_seat_cost_l2430_243053


namespace office_canteen_chairs_l2430_243057

/-- The number of round tables in the office canteen -/
def num_round_tables : ℕ := 2

/-- The number of rectangular tables in the office canteen -/
def num_rectangular_tables : ℕ := 2

/-- The number of chairs per round table -/
def chairs_per_round_table : ℕ := 6

/-- The number of chairs per rectangular table -/
def chairs_per_rectangular_table : ℕ := 7

/-- The total number of chairs in the office canteen -/
def total_chairs : ℕ := num_round_tables * chairs_per_round_table + num_rectangular_tables * chairs_per_rectangular_table

theorem office_canteen_chairs : total_chairs = 26 := by
  sorry

end office_canteen_chairs_l2430_243057


namespace coffee_conference_theorem_l2430_243054

/-- Represents the number of participants who went for coffee -/
def coffee_goers (n : ℕ) : Set ℕ :=
  {k : ℕ | ∃ (remaining : ℕ), 
    remaining > 0 ∧ 
    remaining < n ∧ 
    remaining % 2 = 0 ∧ 
    k = n - remaining}

/-- The theorem stating the possible number of coffee goers -/
theorem coffee_conference_theorem :
  coffee_goers 14 = {6, 8, 10, 12} :=
sorry


end coffee_conference_theorem_l2430_243054


namespace rectangle_ratio_l2430_243058

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : s > 0)
  (h4 : s + 2*x = 3*s) -- The outer boundary is 3 times the inner square side
  (h5 : 2*y = s) -- The shorter side spans half the inner square side
  : x / y = 2 := by
sorry

end rectangle_ratio_l2430_243058


namespace farmer_milk_production_l2430_243061

/-- Calculates the total milk production for a given number of cows over a week -/
def totalMilkProduction (numCows : ℕ) (milkPerDay : ℕ) : ℕ :=
  numCows * milkPerDay * 7

/-- Proves that 52 cows producing 5 liters of milk per day will produce 1820 liters in a week -/
theorem farmer_milk_production :
  totalMilkProduction 52 5 = 1820 := by
  sorry

#eval totalMilkProduction 52 5

end farmer_milk_production_l2430_243061


namespace max_m_and_min_sum_l2430_243065

theorem max_m_and_min_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ m : ℝ, (3 / a + 1 / b ≥ m / (a + 3 * b)) → m ≤ 12) ∧
  (a + 2 * b + 2 * a * b = 8 → a + 2 * b ≥ 4) := by
  sorry

end max_m_and_min_sum_l2430_243065


namespace trigonometric_simplification_l2430_243056

theorem trigonometric_simplification :
  let numerator := Real.sin (15 * π / 180) + Real.sin (25 * π / 180) + 
                   Real.sin (35 * π / 180) + Real.sin (45 * π / 180) + 
                   Real.sin (55 * π / 180) + Real.sin (65 * π / 180) + 
                   Real.sin (75 * π / 180) + Real.sin (85 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (15 * π / 180) * Real.cos (25 * π / 180)
  numerator / denominator = 4 * Real.sin (50 * π / 180) := by
  sorry

end trigonometric_simplification_l2430_243056
