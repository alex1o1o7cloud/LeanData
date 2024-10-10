import Mathlib

namespace gcd_2023_2048_l1910_191015

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end gcd_2023_2048_l1910_191015


namespace min_product_of_geometric_sequence_l1910_191034

theorem min_product_of_geometric_sequence (x y : ℝ) 
  (hx : x > 1) (hy : y > 1) 
  (h_seq : (Real.log x) * (Real.log y) = (1/2)^2) : 
  x * y ≥ Real.exp 1 ∧ ∃ x y, x > 1 ∧ y > 1 ∧ (Real.log x) * (Real.log y) = (1/2)^2 ∧ x * y = Real.exp 1 :=
sorry

end min_product_of_geometric_sequence_l1910_191034


namespace percentage_of_boys_l1910_191099

theorem percentage_of_boys (total_students : ℕ) (boys : ℕ) (percentage : ℚ) : 
  total_students = 220 →
  242 = (220 / 100) * boys →
  percentage = (boys / total_students) * 100 →
  percentage = 50 := by
sorry

end percentage_of_boys_l1910_191099


namespace union_A_B_minus_three_intersection_A_B_equals_B_iff_l1910_191094

-- Define set A
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Statement 1: A ∪ B when m = -3
theorem union_A_B_minus_three : 
  A ∪ B (-3) = {x : ℝ | -7 ≤ x ∧ x ≤ 4} := by sorry

-- Statement 2: A ∩ B = B iff m ≥ -1
theorem intersection_A_B_equals_B_iff (m : ℝ) : 
  A ∩ B m = B m ↔ m ≥ -1 := by sorry

end union_A_B_minus_three_intersection_A_B_equals_B_iff_l1910_191094


namespace quadratic_relationship_l1910_191064

theorem quadratic_relationship (x : ℕ) (z : ℕ) : 
  (x = 1 ∧ z = 5) ∨ 
  (x = 2 ∧ z = 12) ∨ 
  (x = 3 ∧ z = 23) ∨ 
  (x = 4 ∧ z = 38) ∨ 
  (x = 5 ∧ z = 57) → 
  z = 2 * x^2 + x + 2 :=
by sorry

end quadratic_relationship_l1910_191064


namespace number_comparison_l1910_191039

theorem number_comparison : 0.6^7 < 0.7^6 ∧ 0.7^6 < 6^0.7 := by
  sorry

end number_comparison_l1910_191039


namespace average_cost_is_seven_l1910_191074

/-- The average cost per book in cents, rounded to the nearest whole number -/
def average_cost_per_book (num_books : ℕ) (lot_cost : ℚ) (delivery_fee : ℚ) : ℕ :=
  let total_cost_cents := (lot_cost + delivery_fee) * 100
  let average_cost := total_cost_cents / num_books
  (average_cost + 1/2).floor.toNat

/-- Theorem stating that the average cost per book is 7 cents -/
theorem average_cost_is_seven :
  average_cost_per_book 350 (15.30) (9.25) = 7 := by
  sorry

end average_cost_is_seven_l1910_191074


namespace expansion_contains_constant_term_l1910_191037

/-- The expansion of (√x - 2/x)^n contains a constant term for some positive integer n -/
theorem expansion_contains_constant_term : ∃ (n : ℕ+), 
  ∃ (r : ℕ), n = 3 * r := by
  sorry

end expansion_contains_constant_term_l1910_191037


namespace range_of_negative_values_l1910_191049

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on (-∞, 0] if f(x) ≥ f(y) for all x, y ∈ (-∞, 0] with x ≤ y -/
def IsDecreasingOnNegative (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → x ≤ 0 → y ≤ 0 → f x ≥ f y

/-- The theorem stating the range of x for which f(x) < 0 -/
theorem range_of_negative_values (f : ℝ → ℝ) 
  (h_even : IsEven f) 
  (h_decreasing : IsDecreasingOnNegative f) 
  (h_zero : f 2 = 0) : 
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 := by
  sorry

end range_of_negative_values_l1910_191049


namespace quadratic_sum_l1910_191028

/-- A quadratic function with specific properties -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: For a quadratic function g(x) = ax^2 + bx + c with vertex at (-2, 6) 
    and passing through (0, 2), the value of a + 2b + c is -7 -/
theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, g a b c x = a * x^2 + b * x + c) →  -- Definition of g
  g a b c (-2) = 6 →                        -- Vertex at (-2, 6)
  (∀ x, g a b c x ≤ 6) →                   -- (-2, 6) is the maximum point
  g a b c 0 = 2 →                          -- Point (0, 2) on the graph
  a + 2*b + c = -7 :=
by
  sorry

end quadratic_sum_l1910_191028


namespace garden_width_to_perimeter_ratio_l1910_191041

/-- Given a rectangular garden with length 23 feet and width 15 feet, 
    the ratio of its width to its perimeter is 15:76. -/
theorem garden_width_to_perimeter_ratio :
  let garden_length : ℕ := 23
  let garden_width : ℕ := 15
  let perimeter : ℕ := 2 * (garden_length + garden_width)
  (garden_width : ℚ) / perimeter = 15 / 76 := by
  sorry

end garden_width_to_perimeter_ratio_l1910_191041


namespace intersection_union_problem_l1910_191098

theorem intersection_union_problem (m : ℝ) : 
  let A : Set ℝ := {3, 4, m^2 - 3*m - 1}
  let B : Set ℝ := {2*m, -3}
  (A ∩ B = {-3}) → (m = 1 ∧ A ∪ B = {-3, 2, 3, 4}) :=
by sorry

end intersection_union_problem_l1910_191098


namespace cylinder_height_in_hemisphere_l1910_191007

/-- The height of a right circular cylinder inscribed in a hemisphere -/
theorem cylinder_height_in_hemisphere (r_cylinder : ℝ) (r_hemisphere : ℝ) 
  (h_cylinder : r_cylinder = 3)
  (h_hemisphere : r_hemisphere = 7) :
  Real.sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2) = 2 * Real.sqrt 10 := by
  sorry

end cylinder_height_in_hemisphere_l1910_191007


namespace broken_line_enclosing_circle_l1910_191082

/-- A closed broken line in a metric space -/
structure ClosedBrokenLine (α : Type*) [MetricSpace α] where
  points : Set α
  is_closed : IsClosed points
  is_connected : IsConnected points
  perimeter : ℝ

/-- Theorem: Any closed broken line can be enclosed in a circle with radius not exceeding its perimeter divided by 4 -/
theorem broken_line_enclosing_circle 
  {α : Type*} [MetricSpace α] (L : ClosedBrokenLine α) :
  ∃ (center : α), ∀ (p : α), p ∈ L.points → dist center p ≤ L.perimeter / 4 := by
  sorry

end broken_line_enclosing_circle_l1910_191082


namespace min_balls_for_twenty_of_one_color_l1910_191088

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in our problem -/
def ourBox : BallCounts :=
  { red := 30, green := 22, yellow := 18, blue := 15, white := 10, black := 6 }

/-- The theorem to be proved -/
theorem min_balls_for_twenty_of_one_color :
  minBallsForColor ourBox 20 = 88 := by
  sorry

end min_balls_for_twenty_of_one_color_l1910_191088


namespace parabola_axis_of_symmetry_l1910_191013

/-- A parabola is defined by its coefficients a, b, and c in the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a parabola -/
def lies_on (p : Point) (par : Parabola) : Prop :=
  p.y = par.a * p.x^2 + par.b * p.x + par.c

/-- The axis of symmetry of a parabola -/
def axis_of_symmetry (par : Parabola) : ℝ := 3

/-- Theorem: The axis of symmetry of a parabola y = ax^2 + bx + c is x = 3, 
    given that the points (2,5) and (4,5) lie on the parabola -/
theorem parabola_axis_of_symmetry (par : Parabola) 
  (h1 : lies_on ⟨2, 5⟩ par) 
  (h2 : lies_on ⟨4, 5⟩ par) : 
  axis_of_symmetry par = 3 := by
  sorry

end parabola_axis_of_symmetry_l1910_191013


namespace unique_base_for_good_number_l1910_191061

def is_good_number (m : ℕ) : Prop :=
  ∃ (p n : ℕ), n ≥ 2 ∧ Nat.Prime p ∧ m = p^n

theorem unique_base_for_good_number :
  ∀ b : ℕ, (is_good_number (b^2 - 2*b - 3)) ↔ b = 7 :=
by sorry

end unique_base_for_good_number_l1910_191061


namespace count_divisible_sum_l1910_191084

theorem count_divisible_sum : ∃ (S : Finset Nat), 
  (∀ n ∈ S, n > 0 ∧ (10 * n) % ((n * (n + 1)) / 2) = 0) ∧
  (∀ n : Nat, n > 0 ∧ (10 * n) % ((n * (n + 1)) / 2) = 0 → n ∈ S) ∧
  Finset.card S = 5 := by
  sorry

end count_divisible_sum_l1910_191084


namespace line_through_point_parallel_to_line_line_equation_proof_l1910_191047

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_line : Line)
  (given_point : Point)
  (result_line : Line) : Prop :=
  (given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 5) →
  (given_point.x = -2 ∧ given_point.y = 1) →
  (result_line.a = 2 ∧ result_line.b = -3 ∧ result_line.c = 7) →
  (given_point.liesOn result_line ∧ result_line.isParallelTo given_line)

-- The proof of the theorem
theorem line_equation_proof : line_through_point_parallel_to_line 
  (Line.mk 2 (-3) 5) 
  (Point.mk (-2) 1) 
  (Line.mk 2 (-3) 7) := by
  sorry

end line_through_point_parallel_to_line_line_equation_proof_l1910_191047


namespace reporters_coverage_l1910_191051

theorem reporters_coverage (total : ℕ) (h_total : total > 0) : 
  let local_politics := (28 : ℕ) * total / 100
  let not_politics := (60 : ℕ) * total / 100
  let politics := total - not_politics
  (politics - local_politics) * 100 / politics = 30 :=
by sorry

end reporters_coverage_l1910_191051


namespace crayon_count_theorem_l1910_191009

/-- Represents the number of crayons in various states --/
structure CrayonCounts where
  initial : ℕ
  givenAway : ℕ
  lost : ℕ
  remaining : ℕ

/-- Theorem stating the relationship between crayons lost, given away, and the total --/
theorem crayon_count_theorem (c : CrayonCounts) 
  (h1 : c.givenAway = 52)
  (h2 : c.lost = 535)
  (h3 : c.remaining = 492) :
  c.givenAway + c.lost = 587 := by
  sorry

end crayon_count_theorem_l1910_191009


namespace shaded_angle_is_fifteen_degrees_l1910_191081

/-- A configuration of three identical isosceles triangles in a square -/
structure TrianglesInSquare where
  /-- The measure of the angle where three triangles meet at a corner of the square -/
  corner_angle : ℝ
  /-- The measure of each of the two equal angles in each isosceles triangle -/
  isosceles_angle : ℝ
  /-- Axiom: The corner angle is formed by three equal parts -/
  corner_angle_eq : corner_angle = 90 / 3
  /-- Axiom: The sum of angles in each isosceles triangle is 180° -/
  triangle_sum : corner_angle + 2 * isosceles_angle = 180

/-- The theorem to be proved -/
theorem shaded_angle_is_fifteen_degrees (t : TrianglesInSquare) :
  90 - t.isosceles_angle = 15 := by
  sorry

end shaded_angle_is_fifteen_degrees_l1910_191081


namespace completing_square_equivalence_l1910_191073

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end completing_square_equivalence_l1910_191073


namespace cubic_root_sum_l1910_191095

theorem cubic_root_sum (a b : ℝ) : 
  (∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧ r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
   (∀ x : ℝ, 4*x^3 + 7*a*x^2 + 6*b*x + 2*a = 0 ↔ (x = r ∨ x = s ∨ x = t)) ∧
   (r + s + t)^3 = 125) →
  a = -20/7 := by
sorry

end cubic_root_sum_l1910_191095


namespace specific_conference_handshakes_l1910_191059

/-- The number of distinct handshakes in a conference --/
def conference_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes for the specific conference scenario --/
theorem specific_conference_handshakes :
  conference_handshakes 3 5 = 75 := by
  sorry

#eval conference_handshakes 3 5

end specific_conference_handshakes_l1910_191059


namespace parabola_translation_l1910_191093

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (-1) 0 2
  let translated := translate original 2 (-3)
  y = -(x - 2)^2 - 1 ↔ y = translated.a * x^2 + translated.b * x + translated.c :=
sorry

end parabola_translation_l1910_191093


namespace birthday_stickers_l1910_191030

theorem birthday_stickers (initial_stickers total_stickers : ℕ) 
  (h1 : initial_stickers = 269)
  (h2 : total_stickers = 423) : 
  total_stickers - initial_stickers = 154 := by
  sorry

end birthday_stickers_l1910_191030


namespace range_of_a_l1910_191065

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → a ∈ Set.Icc (-2) 2 := by
  sorry

end range_of_a_l1910_191065


namespace vector_properties_l1910_191026

def a : ℝ × ℝ := (-3, 2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (3, -1)

theorem vector_properties :
  (∃ (t : ℝ), ∀ (s : ℝ), ‖a + s • b‖ ≥ ‖a + t • b‖ ∧ ‖a + t • b‖ = (7 * Real.sqrt 5) / 5) ∧
  (∃ (t : ℝ), ∃ (k : ℝ), a - t • b = k • c) :=
sorry

end vector_properties_l1910_191026


namespace complex_fraction_ratio_l1910_191045

theorem complex_fraction_ratio : 
  let z : ℂ := (2 + Complex.I) / Complex.I
  let a : ℝ := z.re
  let b : ℝ := z.im
  b / a = -2 := by sorry

end complex_fraction_ratio_l1910_191045


namespace expand_expressions_l1910_191092

theorem expand_expressions (x m n : ℝ) :
  ((-3*x - 5) * (5 - 3*x) = 9*x^2 - 25) ∧
  ((-3*x - 5) * (5 + 3*x) = -9*x^2 - 30*x - 25) ∧
  ((2*m - 3*n + 1) * (2*m + 1 + 3*n) = 4*m^2 + 4*m + 1 - 9*n^2) := by
  sorry

end expand_expressions_l1910_191092


namespace f_satisfies_data_points_l1910_191008

/-- The function f that we want to prove fits the data points -/
def f (x : ℝ) : ℝ := x^2 + 3*x + 1

/-- The set of data points given in the problem -/
def data_points : List (ℝ × ℝ) := [(1, 5), (2, 11), (3, 19), (4, 29), (5, 41)]

/-- Theorem stating that f satisfies all given data points -/
theorem f_satisfies_data_points : ∀ (point : ℝ × ℝ), point ∈ data_points → f point.1 = point.2 := by
  sorry

end f_satisfies_data_points_l1910_191008


namespace secret_spreading_day_l1910_191067

/-- The number of students who know the secret on the nth day -/
def students_knowing_secret (n : ℕ) : ℕ := 3^(n+1) - 1

/-- The day when 3280 students know the secret -/
theorem secret_spreading_day : ∃ (n : ℕ), students_knowing_secret n = 3280 ∧ n = 7 := by
  sorry

end secret_spreading_day_l1910_191067


namespace complex_equation_sum_l1910_191071

theorem complex_equation_sum (a b : ℝ) : 
  (Complex.mk a 3 + Complex.mk 2 (-1) = Complex.mk 5 b) → a + b = 5 := by
  sorry

end complex_equation_sum_l1910_191071


namespace total_interest_is_330_l1910_191027

/-- Calculates the total interest for a stock over 5 years with increasing rates -/
def stockInterest (initialRate : ℚ) : ℚ :=
  let faceValue : ℚ := 100
  let yearlyIncrease : ℚ := 2 / 100
  (initialRate + yearlyIncrease) * faceValue +
  (initialRate + 2 * yearlyIncrease) * faceValue +
  (initialRate + 3 * yearlyIncrease) * faceValue +
  (initialRate + 4 * yearlyIncrease) * faceValue +
  (initialRate + 5 * yearlyIncrease) * faceValue

/-- Calculates the total interest for all three stocks over 5 years -/
def totalInterest : ℚ :=
  let stock1 : ℚ := 16 / 100
  let stock2 : ℚ := 12 / 100
  let stock3 : ℚ := 20 / 100
  stockInterest stock1 + stockInterest stock2 + stockInterest stock3

theorem total_interest_is_330 : totalInterest = 330 := by
  sorry

end total_interest_is_330_l1910_191027


namespace cosine_sine_equation_solutions_l1910_191087

open Real

theorem cosine_sine_equation_solutions (a α : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   cos (x₁ - a) - sin (x₁ + 2*α) = 0 ∧
   cos (x₂ - a) - sin (x₂ + 2*α) = 0 ∧
   ¬ ∃ k : ℤ, x₁ - x₂ = k * π) ↔ 
  ∃ t : ℤ, a = π * (4*t + 1) / 6 :=
by sorry

end cosine_sine_equation_solutions_l1910_191087


namespace correct_fraction_is_five_thirds_l1910_191054

/-- The percentage error when using an incorrect fraction instead of the correct one. -/
def percentage_error : ℚ := 64.00000000000001

/-- The incorrect fraction used by the student. -/
def incorrect_fraction : ℚ := 3/5

/-- The correct fraction that should have been used. -/
def correct_fraction : ℚ := 5/3

/-- Theorem stating that given the percentage error and incorrect fraction, 
    the correct fraction is 5/3. -/
theorem correct_fraction_is_five_thirds :
  (1 - percentage_error / 100) * correct_fraction = incorrect_fraction :=
sorry

end correct_fraction_is_five_thirds_l1910_191054


namespace tables_required_l1910_191033

-- Define the base-5 number
def base5_seating : ℕ := 3 * 5^2 + 2 * 5^1 + 1 * 5^0

-- Define the number of people per table
def people_per_table : ℕ := 3

-- Theorem to prove
theorem tables_required :
  (base5_seating + people_per_table - 1) / people_per_table = 29 := by
  sorry

end tables_required_l1910_191033


namespace polynomial_inequality_solution_l1910_191077

theorem polynomial_inequality_solution (x : ℝ) : 
  x^4 - 15*x^3 + 80*x^2 - 200*x > 0 ↔ (0 < x ∧ x < 5) ∨ x > 10 := by
  sorry

end polynomial_inequality_solution_l1910_191077


namespace distribute_five_to_three_l1910_191055

/-- The number of ways to distribute n students among k universities,
    with each university receiving at least one student. -/
def distribute_students (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to partition n elements into k non-empty subsets. -/
def stirling2 (n k : ℕ) : ℕ :=
  sorry

theorem distribute_five_to_three :
  distribute_students 5 3 = 150 :=
sorry

end distribute_five_to_three_l1910_191055


namespace arithmetic_properties_l1910_191031

variable (a : ℤ)

theorem arithmetic_properties :
  (216 + 35 + 84 = 35 + (216 + 84)) ∧
  (298 - 35 - 165 = 298 - (35 + 165)) ∧
  (400 / 25 / 4 = 400 / (25 * 4)) ∧
  (a * 6 + 6 * 15 = 6 * (a + 15)) := by
  sorry

end arithmetic_properties_l1910_191031


namespace hyperbola_eccentricity_sqrt_three_l1910_191032

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Represents a point on the right branch of a hyperbola -/
structure RightBranchPoint (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1
  on_right_branch : x > 0

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Predicate to check if three points form an equilateral triangle -/
def is_equilateral_triangle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- Main theorem: If a line through the right focus intersects the right branch
    at two points forming an equilateral triangle with the left focus,
    then the eccentricity of the hyperbola is √3 -/
theorem hyperbola_eccentricity_sqrt_three (h : Hyperbola a b)
  (M N : RightBranchPoint h)
  (h_line : ∃ (t : ℝ), (M.x, M.y) = right_focus h + t • ((N.x, N.y) - right_focus h))
  (h_equilateral : is_equilateral_triangle (M.x, M.y) (N.x, N.y) (left_focus h)) :
  eccentricity h = Real.sqrt 3 := by sorry

end hyperbola_eccentricity_sqrt_three_l1910_191032


namespace basketball_time_l1910_191090

theorem basketball_time (n : ℕ) (last_activity_time : ℝ) : 
  n = 5 ∧ last_activity_time = 160 →
  (let seq := fun i => (2 ^ i) * (last_activity_time / (2 ^ (n - 1)))
   seq 0 = 10) := by
  sorry

end basketball_time_l1910_191090


namespace part_one_part_two_l1910_191080

open Real

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Theorem for part (1)
theorem part_one (t : Triangle) 
  (h1 : t.a^2 = 4 * Real.sqrt 3 * t.S)
  (h2 : t.C = π/3)
  (h3 : t.b = 1) : 
  t.a = 3 := by sorry

-- Theorem for part (2)
theorem part_two (t : Triangle)
  (h1 : t.a^2 = 4 * Real.sqrt 3 * t.S)
  (h2 : t.c / t.b = 2 + Real.sqrt 3) :
  t.A = π/3 := by sorry

end part_one_part_two_l1910_191080


namespace derivative_f_at_2_l1910_191052

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2

-- State the theorem
theorem derivative_f_at_2 : 
  (deriv f) 2 = 12 := by sorry

end derivative_f_at_2_l1910_191052


namespace deceased_member_income_family_income_problem_l1910_191079

theorem deceased_member_income 
  (initial_members : ℕ) 
  (initial_avg_income : ℚ) 
  (final_members : ℕ) 
  (final_avg_income : ℚ) : ℚ :=
  let initial_total_income := initial_members * initial_avg_income
  let final_total_income := final_members * final_avg_income
  initial_total_income - final_total_income

theorem family_income_problem 
  (h1 : initial_members = 4)
  (h2 : initial_avg_income = 840)
  (h3 : final_members = 3)
  (h4 : final_avg_income = 650) : 
  deceased_member_income initial_members initial_avg_income final_members final_avg_income = 1410 := by
  sorry

end deceased_member_income_family_income_problem_l1910_191079


namespace larger_number_proof_l1910_191004

theorem larger_number_proof (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  (Nat.gcd a b = 23) → 
  (Nat.lcm a b = 23 * 12 * 13) → 
  (max a b = 299) := by
sorry

end larger_number_proof_l1910_191004


namespace sin_sum_max_in_acute_triangle_l1910_191000

-- Define the convexity property for a function on an interval
def IsConvex (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y t : ℝ, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ 0 ≤ t ∧ t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

-- State the theorem
theorem sin_sum_max_in_acute_triangle :
  IsConvex Real.sin 0 (Real.pi / 2) →
  ∀ A B C : ℝ,
    0 < A ∧ A < Real.pi / 2 →
    0 < B ∧ B < Real.pi / 2 →
    0 < C ∧ C < Real.pi / 2 →
    A + B + C = Real.pi →
    Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
by sorry

end sin_sum_max_in_acute_triangle_l1910_191000


namespace certain_number_problem_l1910_191044

theorem certain_number_problem (h : 2994 / 14.5 = 173) : 
  ∃ x : ℝ, x / 1.45 = 17.3 ∧ x = 25.085 := by
sorry

end certain_number_problem_l1910_191044


namespace grid_midpoint_theorem_l1910_191078

theorem grid_midpoint_theorem (points : Finset (ℤ × ℤ)) 
  (h : points.card = 5) :
  ∃ p1 p2 : ℤ × ℤ, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ 
  (∃ m : ℤ × ℤ, m.1 * 2 = p1.1 + p2.1 ∧ m.2 * 2 = p1.2 + p2.2) :=
sorry

end grid_midpoint_theorem_l1910_191078


namespace min_beta_delta_sum_l1910_191014

open Complex

/-- The complex number i -/
def i : ℂ := Complex.I

/-- The function g as defined in the problem -/
def g (β δ : ℂ) (z : ℂ) : ℂ := (3 + 2*i)*z^2 + β*z + δ

/-- The theorem statement -/
theorem min_beta_delta_sum :
  ∀ β δ : ℂ, (g β δ 1).im = 0 → (g β δ (-i)).im = 0 → 
  ∃ (min : ℝ), min = 2 * Real.sqrt 2 ∧ 
  ∀ β' δ' : ℂ, (g β' δ' 1).im = 0 → (g β' δ' (-i)).im = 0 → 
  Complex.abs β' + Complex.abs δ' ≥ min :=
sorry

end min_beta_delta_sum_l1910_191014


namespace equation_solution_l1910_191018

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end equation_solution_l1910_191018


namespace water_park_admission_l1910_191063

/-- The admission charge for a child in a water park. -/
def child_admission : ℚ :=
  3⁻¹ * (13 / 4 - 1)

/-- The total amount paid by an adult. -/
def total_paid : ℚ := 13 / 4

/-- The number of children accompanying the adult. -/
def num_children : ℕ := 3

/-- The admission charge for an adult. -/
def adult_admission : ℚ := 1

theorem water_park_admission :
  child_admission * num_children + adult_admission = total_paid :=
sorry

end water_park_admission_l1910_191063


namespace distinct_color_selections_eq_62_l1910_191069

/-- The number of ways to select 6 objects from 5 red and 5 blue objects, where order matters only for color. -/
def distinct_color_selections : ℕ :=
  let red := 5
  let blue := 5
  let total_select := 6
  (2 * (Nat.choose total_select 1) +  -- 5 of one color, 1 of the other
   2 * (Nat.choose total_select 2) +  -- 4 of one color, 2 of the other
   Nat.choose total_select 3)         -- 3 of each color

/-- Theorem stating that the number of distinct color selections is 62. -/
theorem distinct_color_selections_eq_62 : distinct_color_selections = 62 := by
  sorry

end distinct_color_selections_eq_62_l1910_191069


namespace min_value_sqrt_sum_reciprocal_sum_equality_condition_l1910_191046

theorem min_value_sqrt_sum_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a + b) * (1 / Real.sqrt a + 1 / Real.sqrt b) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a + b) * (1 / Real.sqrt a + 1 / Real.sqrt b) = 2 * Real.sqrt 2 ↔ a = b :=
by sorry

end min_value_sqrt_sum_reciprocal_sum_equality_condition_l1910_191046


namespace fractional_equation_positive_root_l1910_191022

theorem fractional_equation_positive_root (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ m / (x - 2) = (1 - x) / (2 - x) - 3) →
  m = 1 := by
sorry

end fractional_equation_positive_root_l1910_191022


namespace quadratic_inequality_result_l1910_191089

theorem quadratic_inequality_result (x : ℝ) :
  x^2 - 5*x + 6 < 0 → x^2 - 5*x + 10 = 4 := by
  sorry

end quadratic_inequality_result_l1910_191089


namespace trisected_right_triangle_product_l1910_191035

/-- A right triangle with trisected angle -/
structure TrisectedRightTriangle where
  -- The length of side XY
  xy : ℝ
  -- The length of side YZ
  yz : ℝ
  -- Point P on XZ
  p : ℝ × ℝ
  -- Point Q on XZ
  q : ℝ × ℝ
  -- The angle at Y is trisected
  angle_trisected : Bool
  -- X, P, Q, Z lie on XZ in that order
  point_order : Bool

/-- The main theorem -/
theorem trisected_right_triangle_product (t : TrisectedRightTriangle)
  (h_xy : t.xy = 228)
  (h_yz : t.yz = 2004)
  (h_trisected : t.angle_trisected = true)
  (h_order : t.point_order = true) :
  (Real.sqrt ((t.p.1 - 0)^2 + (t.p.2 - t.yz)^2) + t.yz) *
  (Real.sqrt ((t.q.1 - 0)^2 + (t.q.2 - t.yz)^2) + t.xy) = 1370736 := by
  sorry

end trisected_right_triangle_product_l1910_191035


namespace teachers_gathering_problem_l1910_191002

theorem teachers_gathering_problem (male_teachers female_teachers : ℕ) 
  (h1 : female_teachers = male_teachers + 12)
  (h2 : (male_teachers : ℚ) / (male_teachers + female_teachers) = 9 / 20) :
  male_teachers + female_teachers = 120 := by
sorry

end teachers_gathering_problem_l1910_191002


namespace expression_equality_l1910_191016

theorem expression_equality : (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5/4 := by
  sorry

end expression_equality_l1910_191016


namespace problem_solution_l1910_191068

theorem problem_solution (a b : ℝ) 
  (h1 : 1 < a) 
  (h2 : a < b) 
  (h3 : 1 / a + 1 / b = 1) 
  (h4 : a * b = 6) : 
  b = 3 + Real.sqrt 3 := by
sorry

end problem_solution_l1910_191068


namespace total_amount_shared_l1910_191053

theorem total_amount_shared (T : ℝ) : 
  (0.4 * T = 0.3 * T + 5) → T = 50 := by
  sorry

end total_amount_shared_l1910_191053


namespace jeremys_beads_l1910_191057

theorem jeremys_beads (n : ℕ) : n > 1 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 ∧ 
  n % 9 = 2 → 
  (∀ m : ℕ, m > 1 ∧ m % 5 = 2 ∧ m % 7 = 2 ∧ m % 9 = 2 → m ≥ n) →
  n = 317 := by
sorry

end jeremys_beads_l1910_191057


namespace square_difference_81_49_l1910_191070

theorem square_difference_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end square_difference_81_49_l1910_191070


namespace percent_of_decimal_l1910_191076

theorem percent_of_decimal (part whole : ℝ) (percent : ℝ) : 
  part = 0.01 → whole = 0.1 → percent = 10 → (part / whole) * 100 = percent := by
  sorry

end percent_of_decimal_l1910_191076


namespace g_zero_at_seven_fifths_l1910_191086

def g (x : ℝ) : ℝ := 5 * x - 7

theorem g_zero_at_seven_fifths : g (7 / 5) = 0 := by
  sorry

end g_zero_at_seven_fifths_l1910_191086


namespace gcd_459_357_l1910_191050

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by sorry

end gcd_459_357_l1910_191050


namespace binomial_coefficient_congruence_l1910_191062

theorem binomial_coefficient_congruence (p a b : ℕ) : 
  Nat.Prime p → p > 3 → a > b → b > 1 → 
  (Nat.choose (a * p) (b * p)) ≡ (Nat.choose a b) [MOD p^3] := by
  sorry

end binomial_coefficient_congruence_l1910_191062


namespace geometric_sequence_ratio_l1910_191023

def is_increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ q > 1

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : is_increasing_geometric_sequence a q)
  (h_sum : a 1 + a 5 = 17)
  (h_prod : a 2 * a 4 = 16) :
  q = 2 :=
sorry

end geometric_sequence_ratio_l1910_191023


namespace maintenance_check_increase_maintenance_check_theorem_l1910_191025

theorem maintenance_check_increase (initial_interval : ℝ) 
  (additive_a_percent : ℝ) (additive_b_percent : ℝ) : ℝ :=
  let interval_after_a := initial_interval * (1 + additive_a_percent)
  let interval_after_b := interval_after_a * (1 + additive_b_percent)
  let total_increase_percent := (interval_after_b - initial_interval) / initial_interval * 100
  total_increase_percent

theorem maintenance_check_theorem :
  maintenance_check_increase 45 0.35 0.20 = 62 := by
  sorry

end maintenance_check_increase_maintenance_check_theorem_l1910_191025


namespace largest_fraction_less_than_16_23_l1910_191075

def F : Set ℚ := {q : ℚ | ∃ m n : ℕ+, q = m / n ∧ m + n ≤ 2005}

theorem largest_fraction_less_than_16_23 :
  ∀ q ∈ F, q < 16/23 → q ≤ 816/1189 :=
by sorry

end largest_fraction_less_than_16_23_l1910_191075


namespace circle_theorem_l1910_191060

/-- Represents the type of person in the circle -/
inductive PersonType
| Knight
| Liar

/-- Checks if a given number is a valid k value -/
def is_valid_k (k : ℕ) : Prop :=
  k < 100 ∧ ∃ (m : ℕ), 100 = m * (k + 1)

/-- The set of all valid k values -/
def valid_k_set : Set ℕ :=
  {1, 3, 4, 9, 19, 24, 49, 99}

/-- A circle of 100 people -/
def Circle := Fin 100 → PersonType

theorem circle_theorem (circle : Circle) :
  ∃ (k : ℕ), is_valid_k k ∧
  (∀ (i : Fin 100),
    (circle i = PersonType.Knight →
      ∀ (j : Fin 100), j < k → circle ((i + j + 1) % 100) = PersonType.Liar) ∧
    (circle i = PersonType.Liar →
      ∃ (j : Fin 100), j < k ∧ circle ((i + j + 1) % 100) = PersonType.Knight)) ↔
  ∃ (k : ℕ), k ∈ valid_k_set :=
sorry

end circle_theorem_l1910_191060


namespace heartsuit_three_eight_l1910_191040

-- Define the ⬥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x - 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = -36 := by
  sorry

end heartsuit_three_eight_l1910_191040


namespace limit_of_function_l1910_191006

theorem limit_of_function : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
  0 < |x| ∧ |x| < δ → 
  |((1 + x * Real.sin x - Real.cos (2 * x)) / (Real.sin x)^2) - 3| < ε := by
sorry

end limit_of_function_l1910_191006


namespace diameters_intersect_l1910_191091

-- Define a convex set in a plane
def ConvexSet (S : Set (Real × Real)) : Prop :=
  ∀ x y : Real × Real, x ∈ S → y ∈ S → ∀ t : Real, 0 ≤ t ∧ t ≤ 1 →
    (t * x.1 + (1 - t) * y.1, t * x.2 + (1 - t) * y.2) ∈ S

-- Define a diameter of a convex set
def Diameter (S : Set (Real × Real)) (d : Set (Real × Real)) : Prop :=
  ConvexSet S ∧ d ⊆ S ∧ ∀ x y : Real × Real, x ∈ S → y ∈ S →
    ∃ a b : Real × Real, a ∈ d ∧ b ∈ d ∧ 
      (a.1 - b.1)^2 + (a.2 - b.2)^2 ≥ (x.1 - y.1)^2 + (x.2 - y.2)^2

-- Theorem statement
theorem diameters_intersect (S : Set (Real × Real)) (d1 d2 : Set (Real × Real)) :
  ConvexSet S → Diameter S d1 → Diameter S d2 → (d1 ∩ d2).Nonempty := by
  sorry

end diameters_intersect_l1910_191091


namespace expression_evaluation_l1910_191024

theorem expression_evaluation :
  (-1)^2008 + (-1)^2009 + 2^2006 * (-1)^2007 + 1^2010 = -2^2006 + 1 := by
  sorry

end expression_evaluation_l1910_191024


namespace otimes_equation_solution_l1910_191021

-- Define the custom operation
noncomputable def otimes (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem otimes_equation_solution :
  ∃ z : ℝ, otimes 3 z = 27 ∧ z = 72 :=
by sorry

end otimes_equation_solution_l1910_191021


namespace linear_function_k_value_l1910_191083

theorem linear_function_k_value (k : ℝ) : 
  k ≠ 0 → (1 : ℝ) = k * 3 - 2 → k = 1 := by sorry

end linear_function_k_value_l1910_191083


namespace decimal_point_problem_l1910_191003

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 9 * (1 / x)) : 
  x = 3 * Real.sqrt 10 / 100 := by
sorry

end decimal_point_problem_l1910_191003


namespace quadratic_factorization_l1910_191043

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 2 * y^2 + 3 * y - 35 = (2 * y + a) * (y + b)) →
  a - b = -12 := by
  sorry

end quadratic_factorization_l1910_191043


namespace three_element_subsets_of_eight_l1910_191020

theorem three_element_subsets_of_eight (S : Finset Nat) :
  S.card = 8 → (S.powerset.filter (fun s => s.card = 3)).card = 56 := by
  sorry

end three_element_subsets_of_eight_l1910_191020


namespace cos_A_in_third_quadrant_l1910_191038

theorem cos_A_in_third_quadrant (A : Real) :
  (A > π ∧ A < 3*π/2) →  -- Angle A is in the third quadrant
  (Real.sin A = -1/3) →  -- sin A = -1/3
  (Real.cos A = -2*Real.sqrt 2/3) :=  -- cos A = -2√2/3
by sorry

end cos_A_in_third_quadrant_l1910_191038


namespace tetrahedron_volume_and_height_l1910_191085

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (a b c d : Point3D) : ℝ := sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (a b c d : Point3D) : ℝ := sorry

theorem tetrahedron_volume_and_height :
  let a₁ : Point3D := ⟨-2, -1, -1⟩
  let a₂ : Point3D := ⟨0, 3, 2⟩
  let a₃ : Point3D := ⟨3, 1, -4⟩
  let a₄ : Point3D := ⟨-4, 7, 3⟩
  (tetrahedronVolume a₁ a₂ a₃ a₄ = 70/3) ∧
  (tetrahedronHeight a₄ a₁ a₂ a₃ = 140 / Real.sqrt 1021) := by
  sorry

end tetrahedron_volume_and_height_l1910_191085


namespace triangle_area_l1910_191042

/-- The area of a triangle with base 2t and height 3t - 1 is t(3t - 1) -/
theorem triangle_area (t : ℝ) : 
  let base : ℝ := 2 * t
  let height : ℝ := 3 * t - 1
  (1 / 2 : ℝ) * base * height = t * (3 * t - 1) := by
  sorry

end triangle_area_l1910_191042


namespace group_division_arrangements_l1910_191096

/-- The number of teachers --/
def num_teachers : ℕ := 2

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of groups --/
def num_groups : ℕ := 2

/-- The number of teachers per group --/
def teachers_per_group : ℕ := 1

/-- The number of students per group --/
def students_per_group : ℕ := 2

/-- The total number of arrangements --/
def total_arrangements : ℕ := 12

theorem group_division_arrangements :
  (Nat.choose num_teachers teachers_per_group) *
  (Nat.choose num_students students_per_group) =
  total_arrangements :=
sorry

end group_division_arrangements_l1910_191096


namespace vann_teeth_cleaning_l1910_191012

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of teeth a pig has -/
def pig_teeth : ℕ := 28

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 5

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 10

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 7

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := dog_teeth * num_dogs + cat_teeth * num_cats + pig_teeth * num_pigs

theorem vann_teeth_cleaning :
  total_teeth = 706 := by sorry

end vann_teeth_cleaning_l1910_191012


namespace divide_l_shaped_ice_sheet_l1910_191066

/-- Represents an L-shaped ice sheet composed of three unit squares -/
structure LShapedIceSheet :=
  (area : ℝ := 3)

/-- Represents a part of the divided ice sheet -/
structure IceSheetPart :=
  (area : ℝ)

/-- Theorem stating that the L-shaped ice sheet can be divided into four equal parts -/
theorem divide_l_shaped_ice_sheet (sheet : LShapedIceSheet) :
  ∃ (part1 part2 part3 part4 : IceSheetPart),
    part1.area = 3/4 ∧
    part2.area = 3/4 ∧
    part3.area = 3/4 ∧
    part4.area = 3/4 ∧
    part1.area + part2.area + part3.area + part4.area = sheet.area :=
sorry

end divide_l_shaped_ice_sheet_l1910_191066


namespace quadratic_root_expression_l1910_191058

theorem quadratic_root_expression (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ + 2 = 0 →
  x₂^2 - 4*x₂ + 2 = 0 →
  x₁ + x₂ = 4 →
  x₁ * x₂ = 2 →
  x₁^2 - 4*x₁ + 2*x₁*x₂ = 2 := by
sorry

end quadratic_root_expression_l1910_191058


namespace sum_of_row_and_column_for_2023_l1910_191072

/-- Represents the value in the table at a given row and column -/
def tableValue (row : ℕ) (col : ℕ) : ℕ :=
  if row % 2 = 1 then
    (row - 1) * 20 + (col - 1) * 2 + 1
  else
    row * 20 - (col - 1) * 2 - 1

/-- The row where 2023 is located -/
def m : ℕ := 253

/-- The column where 2023 is located -/
def n : ℕ := 5

theorem sum_of_row_and_column_for_2023 :
  tableValue m n = 2023 → m + n = 258 := by
  sorry

end sum_of_row_and_column_for_2023_l1910_191072


namespace sufficient_condition_for_reciprocal_inequality_l1910_191019

theorem sufficient_condition_for_reciprocal_inequality (a b : ℝ) :
  b < a ∧ a < 0 → (1 : ℝ) / a < (1 : ℝ) / b :=
by sorry

end sufficient_condition_for_reciprocal_inequality_l1910_191019


namespace people_in_room_l1910_191005

theorem people_in_room (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) :
  total_chairs = 25 →
  seated_people = (3 * total_people) / 4 →
  seated_people = (2 * total_chairs) / 3 →
  total_people = 23 :=
by
  sorry

end people_in_room_l1910_191005


namespace initial_men_correct_l1910_191017

/-- The number of men initially working -/
def initial_men : ℕ := 72

/-- The depth dug by the initial group in meters -/
def initial_depth : ℕ := 30

/-- The hours worked by the initial group per day -/
def initial_hours : ℕ := 8

/-- The new depth to be dug in meters -/
def new_depth : ℕ := 50

/-- The new hours to be worked per day -/
def new_hours : ℕ := 6

/-- The number of extra men needed for the new task -/
def extra_men : ℕ := 88

/-- Theorem stating that the initial number of men is correct -/
theorem initial_men_correct : 
  (initial_depth : ℚ) / (initial_hours * initial_men) = 
  (new_depth : ℚ) / (new_hours * (initial_men + extra_men)) :=
sorry

end initial_men_correct_l1910_191017


namespace function_is_identity_l1910_191036

open Real

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x^2 + y * f z + 1) = x * f x + z * f y + 1

theorem function_is_identity (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x : ℝ, f x = x) ∧ f 5 = 5 := by
  sorry


end function_is_identity_l1910_191036


namespace cubic_root_sum_squares_l1910_191010

/-- Given a cubic equation x^3 - ax^2 + bx - c = 0 with roots r, s, and t,
    prove that r^2 + s^2 + t^2 = a^2 - 2b -/
theorem cubic_root_sum_squares (a b c r s t : ℝ) : 
  (r^3 - a*r^2 + b*r - c = 0) → 
  (s^3 - a*s^2 + b*s - c = 0) → 
  (t^3 - a*t^2 + b*t - c = 0) → 
  r^2 + s^2 + t^2 = a^2 - 2*b := by
  sorry

end cubic_root_sum_squares_l1910_191010


namespace cone_generatrix_length_l1910_191011

/-- The length of the generatrix of a cone formed by a semi-circular iron sheet -/
def generatrix_length (base_radius : ℝ) : ℝ :=
  2 * base_radius

/-- Theorem: The length of the generatrix of the cone is 8 cm -/
theorem cone_generatrix_length :
  let base_radius : ℝ := 4
  generatrix_length base_radius = 8 := by
  sorry

#check cone_generatrix_length

end cone_generatrix_length_l1910_191011


namespace weight_of_rod_l1910_191097

/-- Represents the weight of a uniform rod -/
structure UniformRod where
  /-- Weight per meter of the rod -/
  weight_per_meter : ℝ
  /-- The rod is uniform (constant weight per meter) -/
  uniform : True

/-- Calculate the weight of a given length of a uniform rod -/
def weight_of_length (rod : UniformRod) (length : ℝ) : ℝ :=
  rod.weight_per_meter * length

/-- Theorem: Given a uniform rod where 8 m weighs 30.4 kg, the weight of 11.25 m is 42.75 kg -/
theorem weight_of_rod (rod : UniformRod) 
  (h : weight_of_length rod 8 = 30.4) : 
  weight_of_length rod 11.25 = 42.75 := by
  sorry

end weight_of_rod_l1910_191097


namespace race_time_proof_l1910_191056

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

/-- The conditions of the specific race described in the problem -/
def race_conditions (r : Race) : Prop :=
  r.distance = 1000 ∧
  r.runner_a.speed * r.runner_a.time = r.distance ∧
  r.runner_b.speed * r.runner_b.time = r.distance ∧
  (r.runner_a.speed * r.runner_a.time - r.runner_b.speed * r.runner_a.time = 50 ∨
   r.runner_b.time - r.runner_a.time = 20)

theorem race_time_proof (r : Race) (h : race_conditions r) : r.runner_a.time = 400 := by
  sorry

end race_time_proof_l1910_191056


namespace fraction_equality_l1910_191048

theorem fraction_equality : 
  (3 + 6 - 12 + 24 + 48 - 96 + 192 - 384) / (6 + 12 - 24 + 48 + 96 - 192 + 384 - 768) = 1 / 2 := by
  sorry

end fraction_equality_l1910_191048


namespace complex_equation_solution_l1910_191001

theorem complex_equation_solution (z : ℂ) :
  (3 + 4 * Complex.I) * z = 1 - 2 * Complex.I →
  z = -1/5 - 2/5 * Complex.I :=
by sorry

end complex_equation_solution_l1910_191001


namespace parkway_fifth_grade_count_l1910_191029

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := sorry

/-- The number of boys in the fifth grade -/
def num_boys : ℕ := 312

/-- The number of students playing soccer -/
def num_soccer : ℕ := 250

/-- The proportion of boys among students playing soccer -/
def prop_boys_soccer : ℚ := 78 / 100

/-- The number of girls not playing soccer -/
def num_girls_not_soccer : ℕ := 53

theorem parkway_fifth_grade_count :
  total_students = 420 :=
by sorry

end parkway_fifth_grade_count_l1910_191029
