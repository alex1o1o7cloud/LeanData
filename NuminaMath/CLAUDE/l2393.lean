import Mathlib

namespace log_difference_equality_l2393_239311

theorem log_difference_equality : 
  Real.sqrt (Real.log 12 / Real.log 4 - Real.log 12 / Real.log 5) = 
  Real.sqrt ((Real.log 12 * Real.log 1.25) / (Real.log 4 * Real.log 5)) := by
  sorry

end log_difference_equality_l2393_239311


namespace imaginary_part_of_complex_fraction_l2393_239353

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.im (2 / (1 - i)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l2393_239353


namespace root_sum_reciprocal_l2393_239367

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - p - 1 = 0) → 
  (q^3 - q - 1 = 0) → 
  (r^3 - r - 1 = 0) → 
  (1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 11 / 7) := by
  sorry

end root_sum_reciprocal_l2393_239367


namespace rectangular_solid_diagonal_l2393_239362

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + a * c) = 34)
  (edge_length : 4 * (a + b + c) = 40) :
  ∃ d : ℝ, d^2 = 66 ∧ d^2 = a^2 + b^2 + c^2 := by sorry

end rectangular_solid_diagonal_l2393_239362


namespace line_equation_sum_l2393_239383

/-- Given two points on a line and the general form of the line equation,
    prove that the sum of the slope and y-intercept equals 7. -/
theorem line_equation_sum (m b : ℝ) : 
  (1 = m * (-3) + b) →   -- Point (-3,1) satisfies the equation
  (7 = m * 1 + b) →      -- Point (1,7) satisfies the equation
  m + b = 7 := by
sorry


end line_equation_sum_l2393_239383


namespace line_CR_tangent_to_circumcircle_l2393_239347

-- Define the square ABCD
structure Square (A B C D : ℝ × ℝ) : Prop where
  is_square : A = (0, 0) ∧ B = (0, 1) ∧ C = (1, 1) ∧ D = (1, 0)

-- Define point P on BC
def P (k : ℝ) : ℝ × ℝ := (k, 1)

-- Define square APRS
structure SquareAPRS (A P R S : ℝ × ℝ) : Prop where
  is_square : A = (0, 0) ∧ P.1 = k ∧ P.2 = 1 ∧
              S = (1, -k) ∧ R = (1+k, 1-k)

-- Define the circumcircle of triangle ABC
def CircumcircleABC (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 0.5)^2 + (p.2 - 0.5)^2 = 0.5^2}

-- Define the line CR
def LineCR (C R : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 - C.2) = -1 * (p.1 - C.1)}

-- Theorem statement
theorem line_CR_tangent_to_circumcircle 
  (A B C D : ℝ × ℝ) 
  (k : ℝ) 
  (P R S : ℝ × ℝ) 
  (h1 : Square A B C D) 
  (h2 : 0 ≤ k ∧ k ≤ 1) 
  (h3 : P = (k, 1)) 
  (h4 : SquareAPRS A P R S) :
  ∃ (x : ℝ × ℝ), x ∈ CircumcircleABC A B C ∧ x ∈ LineCR C R ∧
  ∀ (y : ℝ × ℝ), y ≠ x → y ∈ CircumcircleABC A B C → y ∉ LineCR C R :=
sorry


end line_CR_tangent_to_circumcircle_l2393_239347


namespace computer_cost_computer_cost_proof_l2393_239388

theorem computer_cost (accessories_cost : ℕ) (playstation_worth : ℕ) (discount_percent : ℕ) (out_of_pocket : ℕ) : ℕ :=
  let playstation_sold := playstation_worth - (playstation_worth * discount_percent / 100)
  let total_paid := playstation_sold + out_of_pocket
  total_paid - accessories_cost

#check computer_cost 200 400 20 580 = 700

theorem computer_cost_proof :
  computer_cost 200 400 20 580 = 700 := by
  sorry

end computer_cost_computer_cost_proof_l2393_239388


namespace beautiful_points_of_A_beautiful_points_coincide_original_point_C_l2393_239358

-- Define the type for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the beautiful points of a given point
def beautifulPoints (p : Point2D) : (Point2D × Point2D) :=
  let a := -p.x
  let b := p.x - p.y
  ({x := a, y := b}, {x := b, y := a})

-- Theorem 1: Beautiful points of A(4,1)
theorem beautiful_points_of_A :
  let A : Point2D := {x := 4, y := 1}
  let (M, N) := beautifulPoints A
  M = {x := -4, y := 3} ∧ N = {x := 3, y := -4} := by sorry

-- Theorem 2: When beautiful points of B(2,y) coincide
theorem beautiful_points_coincide :
  ∀ y : ℝ, let B : Point2D := {x := 2, y := y}
  let (M, N) := beautifulPoints B
  M = N → y = 4 := by sorry

-- Theorem 3: Original point C given a beautiful point (-2,7)
theorem original_point_C :
  ∀ C : Point2D, let (M, N) := beautifulPoints C
  (M = {x := -2, y := 7} ∨ N = {x := -2, y := 7}) →
  (C = {x := 2, y := -5} ∨ C = {x := -7, y := -5}) := by sorry

end beautiful_points_of_A_beautiful_points_coincide_original_point_C_l2393_239358


namespace line_chart_for_weekly_temperature_l2393_239312

/-- A type representing different chart types -/
inductive ChartType
  | Bar
  | Line
  | Pie
  | Scatter

/-- A structure representing data over time -/
structure TimeSeriesData where
  time_period : String
  has_continuous_change : Bool

/-- A function to determine the most appropriate chart type for a given data set -/
def most_appropriate_chart (data : TimeSeriesData) : ChartType :=
  if data.has_continuous_change then ChartType.Line else ChartType.Bar

/-- Theorem stating that a line chart is most appropriate for weekly temperature data -/
theorem line_chart_for_weekly_temperature :
  let weekly_temp_data : TimeSeriesData := { time_period := "Week", has_continuous_change := true }
  most_appropriate_chart weekly_temp_data = ChartType.Line :=
by
  sorry


end line_chart_for_weekly_temperature_l2393_239312


namespace unique_x_with_three_prime_divisors_l2393_239380

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 11))) →
  11 ∣ x →
  x = 59048 :=
by sorry

end unique_x_with_three_prime_divisors_l2393_239380


namespace tangent_circle_area_l2393_239351

/-- A circle passing through two given points with tangent lines intersecting on x-axis --/
structure TangentCircle where
  /-- The center of the circle --/
  center : ℝ × ℝ
  /-- The radius of the circle --/
  radius : ℝ
  /-- The circle passes through point A --/
  passes_through_A : (center.1 - 7)^2 + (center.2 - 14)^2 = radius^2
  /-- The circle passes through point B --/
  passes_through_B : (center.1 - 13)^2 + (center.2 - 12)^2 = radius^2
  /-- The tangent lines at A and B intersect on the x-axis --/
  tangents_intersect_x_axis : ∃ x : ℝ, 
    (x - 7) * (center.2 - 14) = (center.1 - 7) * 14 ∧
    (x - 13) * (center.2 - 12) = (center.1 - 13) * 12

/-- The theorem stating that the area of the circle is 196π --/
theorem tangent_circle_area (ω : TangentCircle) : π * ω.radius^2 = 196 * π :=
sorry

end tangent_circle_area_l2393_239351


namespace no_solution_implies_a_leq_8_l2393_239330

theorem no_solution_implies_a_leq_8 (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end no_solution_implies_a_leq_8_l2393_239330


namespace function_transformation_l2393_239301

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 0) : 
  f 1 + 1 = 1 := by
sorry

end function_transformation_l2393_239301


namespace valentines_theorem_l2393_239344

theorem valentines_theorem (n x y : ℕ+) (h : x * y = x + y + 36) : x * y = 76 := by
  sorry

end valentines_theorem_l2393_239344


namespace right_triangle_hypotenuse_l2393_239315

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by
sorry

end right_triangle_hypotenuse_l2393_239315


namespace section_area_theorem_l2393_239321

/-- Regular quadrilateral pyramid with given properties -/
structure RegularPyramid where
  -- Base side length
  base_side : ℝ
  -- Distance from apex to cutting plane
  apex_distance : ℝ

/-- Area of the section formed by a plane in the pyramid -/
def section_area (p : RegularPyramid) : ℝ := sorry

/-- Theorem stating the area of the section for the given pyramid -/
theorem section_area_theorem (p : RegularPyramid) 
  (h1 : p.base_side = 8 / Real.sqrt 7)
  (h2 : p.apex_distance = 2 / 3) : 
  section_area p = 6 := by sorry

end section_area_theorem_l2393_239321


namespace six_valid_cuts_l2393_239348

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertex : Point3D
  base : (Point3D × Point3D × Point3D)

/-- Represents a plane -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  vertex1 : Point3D
  vertex2 : Point3D
  vertex3 : Point3D

/-- Function to check if a plane cuts a tetrahedron such that 
    the first projection is an isosceles right triangle -/
def validCut (t : Tetrahedron) (p : Plane) : Bool :=
  sorry

/-- Function to count the number of valid cutting planes -/
def countValidCuts (t : Tetrahedron) : Nat :=
  sorry

/-- Theorem stating that there are exactly 6 valid cutting planes -/
theorem six_valid_cuts (t : Tetrahedron) : 
  countValidCuts t = 6 := by sorry

end six_valid_cuts_l2393_239348


namespace polynomial_symmetry_condition_l2393_239309

/-- A polynomial function of degree 4 -/
def polynomial (a b c d e : ℝ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- Symmetry condition for a function -/
def isSymmetric (f : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, ∀ x : ℝ, f x = f (2 * t - x)

theorem polynomial_symmetry_condition
  (a b c d e : ℝ) (h : a ≠ 0) :
  isSymmetric (polynomial a b c d e) ↔ b^3 - a*b*c + 8*a^2*d = 0 :=
sorry

end polynomial_symmetry_condition_l2393_239309


namespace apple_tree_production_l2393_239329

/-- Apple tree production over three years -/
theorem apple_tree_production : 
  let first_year : ℕ := 40
  let second_year : ℕ := 8 + 2 * first_year
  let third_year : ℕ := second_year - second_year / 4
  first_year + second_year + third_year = 194 := by
sorry

end apple_tree_production_l2393_239329


namespace third_derivative_y_l2393_239379

open Real

noncomputable def y (x : ℝ) : ℝ := (log (2 * x + 5)) / (2 * x + 5)

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = (88 - 48 * log (2 * x + 5)) / (2 * x + 5)^4 :=
by sorry

end third_derivative_y_l2393_239379


namespace least_integer_abs_value_l2393_239323

theorem least_integer_abs_value (y : ℤ) : 
  (∀ z : ℤ, 3 * |z| + 2 < 20 → y ≤ z) ↔ y = -5 := by sorry

end least_integer_abs_value_l2393_239323


namespace ranch_problem_l2393_239307

theorem ranch_problem : ∃! (s c : ℕ), s > 0 ∧ c > 0 ∧ 35 * s + 40 * c = 1200 ∧ c > s := by
  sorry

end ranch_problem_l2393_239307


namespace solve_nested_equation_l2393_239332

theorem solve_nested_equation : 
  ∃ x : ℤ, 45 - (28 - (x - (15 - 16))) = 55 ∧ x = 37 :=
by sorry

end solve_nested_equation_l2393_239332


namespace largest_prime_factor_of_1729_l2393_239334

theorem largest_prime_factor_of_1729 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p := by
  sorry

end largest_prime_factor_of_1729_l2393_239334


namespace integers_between_cubes_l2393_239357

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(10.3 : ℝ)^3⌋ - ⌈(10.2 : ℝ)^3⌉ + 1) ∧ n = 155 := by
  sorry

end integers_between_cubes_l2393_239357


namespace four_digit_numbers_two_repeated_l2393_239338

/-- The number of ways to choose 3 different digits from 0 to 9 -/
def three_digit_choices : ℕ := 10 * 9 * 8

/-- The number of ways to arrange 3 different digits with one repeated (forming a 4-digit number) -/
def repeated_digit_arrangements : ℕ := 6

/-- The number of four-digit numbers with exactly two repeated digits, including those starting with 0 -/
def total_with_leading_zero : ℕ := three_digit_choices * repeated_digit_arrangements

/-- The number of three-digit numbers with exactly two repeated digits (those starting with 0) -/
def starting_with_zero : ℕ := 9 * 8 * repeated_digit_arrangements

/-- The number of four-digit numbers with exactly two repeated digits -/
def four_digit_repeated : ℕ := total_with_leading_zero - starting_with_zero

theorem four_digit_numbers_two_repeated : four_digit_repeated = 3888 := by
  sorry

end four_digit_numbers_two_repeated_l2393_239338


namespace stratified_sample_size_l2393_239343

/-- Given a population with three groups in the ratio 2:3:5, 
    if a stratified sample contains 16 items from the first group, 
    then the total sample size is 80. -/
theorem stratified_sample_size 
  (population_ratio : Fin 3 → ℕ)
  (h_ratio : population_ratio = ![2, 3, 5])
  (sample_size : ℕ)
  (first_group_sample : ℕ)
  (h_first_group : first_group_sample = 16)
  (h_stratified : (population_ratio 0 : ℚ) / (population_ratio 0 + population_ratio 1 + population_ratio 2) 
                = first_group_sample / sample_size) :
  sample_size = 80 :=
sorry

end stratified_sample_size_l2393_239343


namespace symmetric_point_on_parabola_l2393_239300

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y = (x-h)^2 + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Checks if a point lies on a given parabola -/
def isOnParabola (p : Point) (parab : Parabola) : Prop :=
  p.y = (p.x - parab.h)^2 + parab.k

/-- Finds the symmetric point with respect to the axis of symmetry of a parabola -/
def symmetricPoint (p : Point) (parab : Parabola) : Point :=
  ⟨2 * parab.h - p.x, p.y⟩

theorem symmetric_point_on_parabola (parab : Parabola) (p : Point) :
  isOnParabola p parab → p.x = -1 → 
  symmetricPoint p parab = Point.mk 3 6 := by
  sorry

#check symmetric_point_on_parabola

end symmetric_point_on_parabola_l2393_239300


namespace sum_of_specific_digits_l2393_239371

/-- A sequence where each positive integer n is repeated n times in increasing order -/
def special_sequence : ℕ → ℕ
  | 0 => 0
  | n + 1 => sorry

/-- The nth digit of the special sequence -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the 4501st and 4052nd digits of the special sequence is 13 -/
theorem sum_of_specific_digits :
  nth_digit 4501 + nth_digit 4052 = 13 := by sorry

end sum_of_specific_digits_l2393_239371


namespace boat_travel_theorem_l2393_239384

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (boat_speed stream_speed : ℝ) : ℝ := boat_speed - stream_speed

/-- Proves that a boat traveling 11 km along the stream in one hour will travel 7 km against the stream in one hour, given its still water speed is 9 km/hr -/
theorem boat_travel_theorem (boat_speed : ℝ) (h1 : boat_speed = 9) 
  (h2 : boat_speed + (11 - boat_speed) = 11) : 
  boat_distance boat_speed (11 - boat_speed) = 7 := by
  sorry

#check boat_travel_theorem

end boat_travel_theorem_l2393_239384


namespace lifting_ratio_after_training_l2393_239310

/-- Calculates the ratio of lifting total to bodyweight after training -/
theorem lifting_ratio_after_training 
  (initial_total : ℝ)
  (initial_weight : ℝ)
  (total_increase_percent : ℝ)
  (weight_increase : ℝ)
  (h1 : initial_total = 2200)
  (h2 : initial_weight = 245)
  (h3 : total_increase_percent = 0.15)
  (h4 : weight_increase = 8) :
  (initial_total * (1 + total_increase_percent)) / (initial_weight + weight_increase) = 10 :=
by
  sorry

#check lifting_ratio_after_training

end lifting_ratio_after_training_l2393_239310


namespace acorn_theorem_l2393_239387

/-- The number of acorns Shawna, Sheila, and Danny have altogether -/
theorem acorn_theorem (shawna sheila danny : ℕ) : 
  shawna = 7 →
  sheila = 5 * shawna →
  danny = sheila + 3 →
  shawna + sheila + danny = 80 := by
  sorry

end acorn_theorem_l2393_239387


namespace f_has_minimum_at_negative_four_l2393_239354

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 8*x + 2

-- Theorem stating that f has a minimum at x = -4
theorem f_has_minimum_at_negative_four :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ x₀ = -4 :=
by
  sorry

end f_has_minimum_at_negative_four_l2393_239354


namespace smallest_negative_quadratic_l2393_239304

theorem smallest_negative_quadratic (n : ℤ) : 
  (∀ m : ℤ, m < n → 4 * m^2 - 28 * m + 48 ≥ 0) ∧ 
  (4 * n^2 - 28 * n + 48 < 0) → 
  n = 4 := by
sorry

end smallest_negative_quadratic_l2393_239304


namespace reading_completion_time_l2393_239328

/-- Represents a reader with their reading speed and number of books to read -/
structure Reader where
  speed : ℕ  -- hours per book
  books : ℕ

/-- Represents the reading schedule constraints -/
structure ReadingConstraints where
  hours_per_day : ℕ

/-- Calculate the total reading time for a reader -/
def total_reading_time (reader : Reader) : ℕ :=
  reader.speed * reader.books

/-- Calculate the number of days needed to finish reading -/
def days_to_finish (reader : Reader) (constraints : ReadingConstraints) : ℕ :=
  (total_reading_time reader + constraints.hours_per_day - 1) / constraints.hours_per_day

theorem reading_completion_time 
  (peter kristin : Reader) 
  (constraints : ReadingConstraints) 
  (h1 : peter.speed = 12)
  (h2 : kristin.speed = 3 * peter.speed)
  (h3 : peter.books = 20)
  (h4 : kristin.books = 20)
  (h5 : constraints.hours_per_day = 16) :
  kristin.speed = 36 ∧ 
  days_to_finish peter constraints = days_to_finish kristin constraints ∧
  days_to_finish kristin constraints = 45 := by
  sorry

end reading_completion_time_l2393_239328


namespace discounted_cd_cost_l2393_239377

/-- The cost of five CDs with a 10% discount, given the cost of two CDs and the discount condition -/
theorem discounted_cd_cost (cost_of_two : ℝ) (discount_rate : ℝ) : 
  cost_of_two = 40 →
  discount_rate = 0.1 →
  (5 : ℝ) * (cost_of_two / 2) * (1 - discount_rate) = 90 := by
sorry

end discounted_cd_cost_l2393_239377


namespace sqrt_80_bound_l2393_239352

theorem sqrt_80_bound (k : ℤ) : k < Real.sqrt 80 ∧ Real.sqrt 80 < k + 1 → k = 8 := by
  sorry

end sqrt_80_bound_l2393_239352


namespace integers_less_than_four_abs_value_l2393_239361

theorem integers_less_than_four_abs_value :
  {x : ℤ | |x| < 4} = {-3, -2, -1, 0, 1, 2, 3} := by sorry

end integers_less_than_four_abs_value_l2393_239361


namespace minimum_team_size_l2393_239333

theorem minimum_team_size : ∃ n : ℕ, n > 0 ∧ n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ ∀ m : ℕ, m > 0 → m % 8 = 0 → m % 9 = 0 → m % 10 = 0 → n ≤ m :=
by sorry

end minimum_team_size_l2393_239333


namespace otimes_inequality_implies_a_unrestricted_l2393_239397

/-- Custom operation ⊗ defined on real numbers -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating that if (x-a) ⊗ (x+a) < 1 holds for all real x, then a can be any real number -/
theorem otimes_inequality_implies_a_unrestricted :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → a ∈ Set.univ :=
by sorry

end otimes_inequality_implies_a_unrestricted_l2393_239397


namespace equation_system_equivalence_l2393_239369

theorem equation_system_equivalence (x y : ℝ) :
  (3 * x^2 + 9 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 4 = 0) →
  4 * y^2 + 19 * y - 14 = 0 := by
  sorry

end equation_system_equivalence_l2393_239369


namespace average_MTWT_is_48_l2393_239325

/-- The average temperature for some days -/
def average_some_days : ℝ := 48

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday -/
def average_TWTF : ℝ := 46

/-- The temperature on Monday -/
def temp_Monday : ℝ := 42

/-- The temperature on Friday -/
def temp_Friday : ℝ := 34

/-- The number of days in the TWTF group -/
def num_days_TWTF : ℕ := 4

/-- The number of days in the MTWT group -/
def num_days_MTWT : ℕ := 4

/-- Theorem: The average temperature for Monday, Tuesday, Wednesday, and Thursday is 48 degrees -/
theorem average_MTWT_is_48 : 
  (temp_Monday + (average_TWTF * num_days_TWTF - temp_Friday)) / num_days_MTWT = 48 := by
  sorry

end average_MTWT_is_48_l2393_239325


namespace average_of_numbers_l2393_239308

def numbers : List ℕ := [12, 13, 14, 510, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 125790 := by sorry

end average_of_numbers_l2393_239308


namespace chess_tournament_draw_fraction_l2393_239356

theorem chess_tournament_draw_fraction 
  (peter_wins : Rat) 
  (marc_wins : Rat) 
  (h1 : peter_wins = 2 / 5)
  (h2 : marc_wins = 1 / 4)
  : 1 - (peter_wins + marc_wins) = 7 / 20 := by
  sorry

end chess_tournament_draw_fraction_l2393_239356


namespace identity_function_unique_l2393_239374

theorem identity_function_unique (f : ℤ → ℤ) 
  (h1 : ∀ x : ℤ, f (f x) = x)
  (h2 : ∀ x y : ℤ, Odd (x + y) → f x + f y ≥ x + y) :
  ∀ x : ℤ, f x = x := by sorry

end identity_function_unique_l2393_239374


namespace square_sum_of_xy_l2393_239378

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 35)
  (h2 : x^2 * y + x * y^2 = 306) : 
  x^2 + y^2 = 290 := by
sorry

end square_sum_of_xy_l2393_239378


namespace dictionary_correct_and_complete_l2393_239382

-- Define the types for words and sentences
def Word : Type := String
def Sentence : Type := List Word

-- Define the type for a dictionary
def Dictionary : Type := List (Word × Word)

-- Define the Russian sentences
def russian_sentences : List Sentence := [
  ["Мышка", "ночью", "пошла", "гулять"],
  ["Кошка", "ночью", "видит", "мышка"],
  ["Мышку", "кошка", "пошла", "поймать"]
]

-- Define the Am-Yam sentences
def amyam_sentences : List Sentence := [
  ["ту", "ам", "ям", "му"],
  ["ля", "ам", "бу", "ту"],
  ["ту", "ля", "ям", "ям"]
]

-- Define the correct dictionary fragment
def correct_dictionary : Dictionary := [
  ("гулять", "му"),
  ("видит", "бу"),
  ("поймать", "ям"),
  ("мышка", "ту"),
  ("ночью", "ам"),
  ("пошла", "ям"),
  ("кошка", "ля")
]

-- Function to create dictionary from sentence pairs
def create_dictionary (russian : List Sentence) (amyam : List Sentence) : Dictionary :=
  sorry

-- Theorem statement
theorem dictionary_correct_and_complete 
  (russian : List Sentence := russian_sentences)
  (amyam : List Sentence := amyam_sentences)
  (correct : Dictionary := correct_dictionary) :
  create_dictionary russian amyam = correct :=
sorry

end dictionary_correct_and_complete_l2393_239382


namespace angle_I_measures_138_l2393_239302

/-- A convex pentagon with specific angle properties -/
structure ConvexPentagon where
  -- Angles in degrees
  F : ℝ
  G : ℝ
  H : ℝ
  I : ℝ
  J : ℝ
  -- Angle sum in a pentagon is 540°
  sum_eq_540 : F + G + H + I + J = 540
  -- Angles F, G, and H are congruent
  F_eq_G : F = G
  G_eq_H : G = H
  -- Angles I and J are congruent
  I_eq_J : I = J
  -- Angle F is 50° less than angle I
  F_eq_I_minus_50 : F = I - 50

/-- Theorem: In a convex pentagon with the given properties, angle I measures 138° -/
theorem angle_I_measures_138 (p : ConvexPentagon) : p.I = 138 := by
  sorry

end angle_I_measures_138_l2393_239302


namespace hryzka_nuts_theorem_l2393_239341

/-- Represents the two types of days in Hryzka's eating schedule -/
inductive DayType
  | Diet
  | Normal

/-- Calculates the number of nuts eaten on a given day type -/
def nutsEaten (d : DayType) : ℕ :=
  match d with
  | DayType.Diet => 1
  | DayType.Normal => 3

/-- Represents a sequence of day types -/
def Schedule := List DayType

/-- Generates an alternating schedule of the given length -/
def generateSchedule (startWithDiet : Bool) (length : ℕ) : Schedule :=
  sorry

/-- Calculates the total nuts eaten for a given schedule -/
def totalNutsEaten (s : Schedule) : ℕ :=
  sorry

theorem hryzka_nuts_theorem :
  let dietFirst := generateSchedule true 19
  let normalFirst := generateSchedule false 19
  (totalNutsEaten dietFirst = 37 ∧ totalNutsEaten normalFirst = 39) ∧
  (∀ (s : Schedule), s.length = 19 → totalNutsEaten s ≥ 37 ∧ totalNutsEaten s ≤ 39) :=
  sorry

#check hryzka_nuts_theorem

end hryzka_nuts_theorem_l2393_239341


namespace integer_pairs_satisfying_equation_l2393_239350

theorem integer_pairs_satisfying_equation :
  {(x, y) : ℤ × ℤ | 8 * x^2 * y^2 + x^2 + y^2 = 10 * x * y} =
  {(0, 0), (1, 1), (-1, -1)} :=
by sorry

end integer_pairs_satisfying_equation_l2393_239350


namespace certain_number_problem_l2393_239305

theorem certain_number_problem : ∃ x : ℤ, (3005 - 3000 + x = 2705) ∧ (x = 2700) := by
  sorry

end certain_number_problem_l2393_239305


namespace no_x_exists_rational_l2393_239335

theorem no_x_exists_rational : ¬ ∃ (x : ℝ), (∃ (a b : ℚ), (x + Real.sqrt 2 = a) ∧ (x^3 + Real.sqrt 2 = b)) := by
  sorry

end no_x_exists_rational_l2393_239335


namespace line_intersection_l2393_239303

theorem line_intersection :
  ∃! p : ℚ × ℚ, 
    (3 * p.2 = -2 * p.1 + 6) ∧ 
    (-2 * p.2 = 6 * p.1 + 4) ∧ 
    p = (-12/7, 22/7) := by
  sorry

end line_intersection_l2393_239303


namespace circle_op_proof_l2393_239327

def circle_op (M N : Set ℕ) : Set ℕ := {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

theorem circle_op_proof (M N : Set ℕ) 
  (hM : M = {0, 2, 4, 6, 8, 10}) 
  (hN : N = {0, 3, 6, 9, 12, 15}) : 
  (circle_op (circle_op M N) M) = N := by
  sorry

#check circle_op_proof

end circle_op_proof_l2393_239327


namespace int_coord_triangle_area_rational_l2393_239368

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a triangle with three integer points
structure IntTriangle where
  p1 : IntPoint
  p2 : IntPoint
  p3 : IntPoint

-- Function to calculate the area of a triangle
def triangleArea (t : IntTriangle) : ℚ :=
  let x1 := t.p1.x
  let y1 := t.p1.y
  let x2 := t.p2.x
  let y2 := t.p2.y
  let x3 := t.p3.x
  let y3 := t.p3.y
  (1/2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Theorem stating that the area of a triangle with integer coordinates is rational
theorem int_coord_triangle_area_rational (t : IntTriangle) : 
  ∃ q : ℚ, triangleArea t = q :=
sorry

end int_coord_triangle_area_rational_l2393_239368


namespace square_sum_solution_l2393_239317

theorem square_sum_solution (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 130) : 
  x^2 + y^2 = 3049 / 49 := by
  sorry

end square_sum_solution_l2393_239317


namespace deal_or_no_deal_elimination_l2393_239355

/-- The total number of boxes in the game -/
def total_boxes : ℕ := 26

/-- The number of boxes containing at least $200,000 -/
def high_value_boxes : ℕ := 9

/-- The probability threshold for holding a high-value box -/
def probability_threshold : ℚ := 1/2

/-- The minimum number of boxes that need to be eliminated -/
def boxes_to_eliminate : ℕ := 9

theorem deal_or_no_deal_elimination :
  boxes_to_eliminate = total_boxes - high_value_boxes - (total_boxes - high_value_boxes) / 2 :=
by sorry

end deal_or_no_deal_elimination_l2393_239355


namespace partial_fraction_decomposition_l2393_239366

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ), ∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
    (x^2 + 2) / ((x - 1) * (x - 4) * (x - 6)) = 
    P / (x - 1) + Q / (x - 4) + R / (x - 6) ∧
    P = 1/5 ∧ Q = -3 ∧ R = 19/5 := by
  sorry

end partial_fraction_decomposition_l2393_239366


namespace excluded_students_count_l2393_239390

theorem excluded_students_count (total_students : ℕ) (initial_avg : ℚ) 
  (excluded_avg : ℚ) (new_avg : ℚ) (h1 : total_students = 10) 
  (h2 : initial_avg = 80) (h3 : excluded_avg = 70) (h4 : new_avg = 90) :
  ∃ (excluded : ℕ), 
    excluded = 5 ∧ 
    (initial_avg * total_students : ℚ) = 
      excluded_avg * excluded + new_avg * (total_students - excluded) :=
by sorry

end excluded_students_count_l2393_239390


namespace ellipse_sum_l2393_239363

/-- Represents an ellipse with center (h, k) and semi-axes lengths a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

theorem ellipse_sum (e : Ellipse) :
  e.h = 3 ∧ e.k = -5 ∧ e.a = 7 ∧ e.b = 4 →
  e.h + e.k + e.a + e.b = 9 := by
  sorry

end ellipse_sum_l2393_239363


namespace parallel_lines_distance_l2393_239372

/-- Given three equally spaced parallel lines intersecting a circle and creating chords of lengths 40, 36, and 32, 
    the distance between two adjacent parallel lines is √(576/31). -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ), 
    chord1 = 40 ∧ 
    chord2 = 36 ∧ 
    chord3 = 32 ∧ 
    400 + (5/4) * d^2 = r^2 ∧ 
    256 + (36/4) * d^2 = r^2) → 
  d = Real.sqrt (576/31) := by
sorry

end parallel_lines_distance_l2393_239372


namespace soccer_ball_cost_l2393_239349

theorem soccer_ball_cost (total_cost : ℕ) (num_soccer_balls : ℕ) (num_volleyballs : ℕ) (volleyball_cost : ℕ) :
  total_cost = 980 ∧ num_soccer_balls = 5 ∧ num_volleyballs = 4 ∧ volleyball_cost = 65 →
  ∃ (soccer_ball_cost : ℕ), soccer_ball_cost = 144 ∧ 
    total_cost = num_soccer_balls * soccer_ball_cost + num_volleyballs * volleyball_cost :=
by
  sorry

end soccer_ball_cost_l2393_239349


namespace expand_polynomial_l2393_239320

theorem expand_polynomial (x : ℝ) : 
  (x + 3) * (4 * x^2 - 2 * x - 5) = 4 * x^3 + 10 * x^2 - 11 * x - 15 := by
  sorry

end expand_polynomial_l2393_239320


namespace sum_of_roots_special_quadratic_l2393_239360

theorem sum_of_roots_special_quadratic :
  let f : ℝ → ℝ := λ x ↦ (x - 7)^2 - 16
  ∃ r₁ r₂ : ℝ, (f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂) ∧ r₁ + r₂ = 14 :=
by sorry

end sum_of_roots_special_quadratic_l2393_239360


namespace spruce_tree_height_l2393_239340

theorem spruce_tree_height 
  (height_maple : ℝ) 
  (height_pine : ℝ) 
  (height_spruce : ℝ) 
  (h1 : height_maple = height_pine + 1)
  (h2 : height_pine = height_spruce - 4)
  (h3 : height_maple / height_spruce = 25 / 64) :
  height_spruce = 64 / 13 := by
  sorry

end spruce_tree_height_l2393_239340


namespace fourth_root_of_sum_of_cubes_l2393_239393

theorem fourth_root_of_sum_of_cubes : ∃ n : ℕ, n > 0 ∧ n^4 = 5508^3 + 5625^3 + 5742^3 ∧ n = 855 := by
  sorry

end fourth_root_of_sum_of_cubes_l2393_239393


namespace minimum_value_theorem_l2393_239399

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n

theorem minimum_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  GeometricSequence a →
  a 3 = a 2 + 2 * a 1 →
  a m * a n = 64 * (a 1)^2 →
  (∀ k l : ℕ, a k * a l = 64 * (a 1)^2 → 1 / k + 9 / l ≥ 1 / m + 9 / n) →
  1 / m + 9 / n = 2 := by
  sorry

end minimum_value_theorem_l2393_239399


namespace square_diagonal_perimeter_l2393_239324

theorem square_diagonal_perimeter (d : ℝ) (s : ℝ) (P : ℝ) :
  d = 2 * Real.sqrt 2 →  -- diagonal length
  d = s * Real.sqrt 2 →  -- relation between diagonal and side length
  P = 4 * s →           -- perimeter definition
  P = 8 := by
sorry

end square_diagonal_perimeter_l2393_239324


namespace sherlock_lock_combination_l2393_239373

def is_valid_solution (d : ℕ) (S E N D R : ℕ) : Prop :=
  S < d ∧ E < d ∧ N < d ∧ D < d ∧ R < d ∧
  S ≠ E ∧ S ≠ N ∧ S ≠ D ∧ S ≠ R ∧
  E ≠ N ∧ E ≠ D ∧ E ≠ R ∧
  N ≠ D ∧ N ≠ R ∧
  D ≠ R ∧
  (S * d^3 + E * d^2 + N * d + D) +
  (E * d^2 + N * d + D) +
  (R * d^2 + E * d + D) =
  (D * d^3 + E * d^2 + E * d + R)

theorem sherlock_lock_combination :
  ∃ (d : ℕ), ∃ (S E N D R : ℕ),
    is_valid_solution d S E N D R ∧
    R * d^2 + E * d + D = 879 :=
sorry

end sherlock_lock_combination_l2393_239373


namespace vector_difference_magnitude_l2393_239336

/-- Given two vectors in R², prove that the magnitude of their difference is 5. -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by
  sorry

end vector_difference_magnitude_l2393_239336


namespace max_mice_two_kittens_max_mice_two_males_l2393_239326

/-- Represents the production possibility frontier (PPF) for a kitten --/
structure KittenPPF where
  maxMice : ℕ  -- Maximum number of mice caught when K = 0
  slope : ℚ    -- Rate of decrease in mice caught per hour of therapy

/-- Calculates the number of mice caught given hours of therapy --/
def micesCaught (ppf : KittenPPF) (therapyHours : ℚ) : ℚ :=
  ppf.maxMice - ppf.slope * therapyHours

/-- Male kitten PPF --/
def malePPF : KittenPPF := { maxMice := 80, slope := 4 }

/-- Female kitten PPF --/
def femalePPF : KittenPPF := { maxMice := 16, slope := 1/4 }

/-- Theorem: The maximum number of mice caught by 2 kittens is 160 --/
theorem max_mice_two_kittens :
  ∀ (k1 k2 : KittenPPF), ∀ (h1 h2 : ℚ),
    micesCaught k1 h1 + micesCaught k2 h2 ≤ 160 :=
by sorry

/-- Corollary: The maximum is achieved with two male kittens and zero therapy hours --/
theorem max_mice_two_males :
  micesCaught malePPF 0 + micesCaught malePPF 0 = 160 :=
by sorry

end max_mice_two_kittens_max_mice_two_males_l2393_239326


namespace running_speed_is_six_l2393_239391

/-- Calculates the running speed given swimming speed and average speed -/
def calculate_running_speed (swimming_speed average_speed : ℝ) : ℝ :=
  2 * average_speed - swimming_speed

/-- Proves that given a swimming speed of 1 mph and an average speed of 3.5 mph
    for equal time spent swimming and running, the running speed is 6 mph -/
theorem running_speed_is_six :
  let swimming_speed : ℝ := 1
  let average_speed : ℝ := 3.5
  calculate_running_speed swimming_speed average_speed = 6 := by
  sorry

#eval calculate_running_speed 1 3.5

end running_speed_is_six_l2393_239391


namespace wife_departure_time_l2393_239345

/-- Proves that given the conditions of the problem, the wife left 24 minutes after the man -/
theorem wife_departure_time (man_speed wife_speed : ℝ) (meeting_time : ℝ) :
  man_speed = 40 →
  wife_speed = 50 →
  meeting_time = 2 →
  ∃ (t : ℝ), t * wife_speed = meeting_time * man_speed ∧ t = 24 / 60 :=
by sorry

end wife_departure_time_l2393_239345


namespace hyperbola_foci_l2393_239394

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

/-- The foci of the hyperbola -/
def foci : Set (ℝ × ℝ) :=
  {(-5, 0), (5, 0)}

/-- Theorem: The given foci are the correct foci of the hyperbola -/
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y →
  ∃ (f : ℝ × ℝ), f ∈ foci ∧
  (x - f.1)^2 + y^2 = (x + f.1)^2 + y^2 :=
sorry

end hyperbola_foci_l2393_239394


namespace pipe_A_fill_time_l2393_239314

-- Define the time it takes for Pipe A to fill the tank
variable (A : ℝ)

-- Define the time it takes for Pipe B to empty the tank
def B : ℝ := 24

-- Define the total time to fill the tank when both pipes are used
def total_time : ℝ := 30

-- Define the time Pipe B is open
def B_open_time : ℝ := 24

-- Define the theorem
theorem pipe_A_fill_time :
  (1 / A - 1 / B) * B_open_time + (1 / A) * (total_time - B_open_time) = 1 →
  A = 15 := by
sorry

end pipe_A_fill_time_l2393_239314


namespace smallest_n_for_no_real_roots_l2393_239370

theorem smallest_n_for_no_real_roots :
  ∀ n : ℤ, (∀ x : ℝ, 3 * x * (n * x + 3) - 2 * x^2 - 9 ≠ 0) →
  n ≥ -1 ∧ ∀ m : ℤ, m < -1 → ∃ x : ℝ, 3 * x * (m * x + 3) - 2 * x^2 - 9 = 0 :=
by sorry

end smallest_n_for_no_real_roots_l2393_239370


namespace power_of_three_mod_seven_l2393_239389

theorem power_of_three_mod_seven : 3^20 % 7 = 2 := by
  sorry

end power_of_three_mod_seven_l2393_239389


namespace max_time_digit_sum_l2393_239381

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours ≤ 23
  minutes_valid : minutes ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum possible sum of digits for any Time24 -/
def maxTimeDigitSum : Nat := 24

theorem max_time_digit_sum :
  ∀ t : Time24, timeDigitSum t ≤ maxTimeDigitSum :=
by sorry

end max_time_digit_sum_l2393_239381


namespace x_value_proof_l2393_239375

theorem x_value_proof (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 := by
  sorry

end x_value_proof_l2393_239375


namespace system_solution_l2393_239385

theorem system_solution (a b c : ℝ) 
  (eq1 : b + c = 10 - 4*a)
  (eq2 : a + c = -16 - 4*b)
  (eq3 : a + b = 9 - 4*c) :
  2*a + 2*b + 2*c = 1 := by
sorry

end system_solution_l2393_239385


namespace initial_number_theorem_l2393_239386

theorem initial_number_theorem (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end initial_number_theorem_l2393_239386


namespace complex_equality_l2393_239322

theorem complex_equality (z : ℂ) : z = -1 + (7/2) * I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧
  Complex.abs (z - 2) = Complex.abs (z + I) :=
by
  sorry

end complex_equality_l2393_239322


namespace new_individuals_weight_l2393_239318

/-- The total weight of three new individuals joining a group, given specific conditions -/
theorem new_individuals_weight (W : ℝ) : 
  let initial_group_size : ℕ := 10
  let leaving_weights : List ℝ := [75, 80, 90]
  let average_weight_increase : ℝ := 6.5
  let new_individuals_count : ℕ := 3
  W - (initial_group_size : ℝ) * average_weight_increase = 
    (W - leaving_weights.sum) + (new_individuals_count : ℝ) * average_weight_increase →
  (∃ X : ℝ, X = (new_individuals_count : ℝ) * average_weight_increase ∧ X = 65) := by
sorry


end new_individuals_weight_l2393_239318


namespace product_inequality_l2393_239342

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end product_inequality_l2393_239342


namespace inequality_proof_l2393_239359

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 1) : 
  x * y + y * z + z * x ≤ 2 / 7 + 9 * x * y * z / 7 := by
  sorry

end inequality_proof_l2393_239359


namespace solve_for_m_l2393_239306

def f (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + m

def g (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + 5*m

theorem solve_for_m :
  ∃ m : ℚ, 3 * (f m 5) = 2 * (g m 5) ∧ m = 10/7 := by
  sorry

end solve_for_m_l2393_239306


namespace circle_area_ratio_l2393_239337

theorem circle_area_ratio (R : ℝ) (h : R > 0) : 
  let total_area := π * R^2
  let part_area := total_area / 8
  let shaded_area := 2 * part_area
  let unshaded_area := total_area - shaded_area
  shaded_area / unshaded_area = 1 / 3 := by sorry

end circle_area_ratio_l2393_239337


namespace crease_line_equation_l2393_239346

/-- Given a circle with radius R and a point A inside the circle at distance a from the center,
    the set of all points (x, y) on the crease lines formed by folding the paper so that any point
    on the circumference coincides with A satisfies the equation:
    (2x - a)^2 / R^2 + 4y^2 / (R^2 - a^2) = 1 -/
theorem crease_line_equation (R a x y : ℝ) (h1 : R > 0) (h2 : 0 ≤ a) (h3 : a < R) :
  (∃ (A' : ℝ × ℝ), (A'.1^2 + A'.2^2 = R^2) ∧
   ((x - A'.1)^2 + (y - A'.2)^2 = (x - a)^2 + y^2)) ↔
  (2*x - a)^2 / R^2 + 4*y^2 / (R^2 - a^2) = 1 :=
sorry

end crease_line_equation_l2393_239346


namespace regular_hexagon_perimeter_l2393_239398

/-- The perimeter of a regular hexagon with side length 5 cm is 30 cm. -/
theorem regular_hexagon_perimeter :
  ∀ (side_length : ℝ), side_length = 5 →
  (6 : ℝ) * side_length = 30 := by sorry

end regular_hexagon_perimeter_l2393_239398


namespace greatest_difference_of_unit_digits_l2393_239392

def is_multiple_of_four (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k

def three_digit_72X (n : ℕ) : Prop := ∃ x : ℕ, n = 720 + x ∧ x < 10

def possible_unit_digit (x : ℕ) : Prop :=
  ∃ n : ℕ, three_digit_72X n ∧ is_multiple_of_four n ∧ n % 10 = x

theorem greatest_difference_of_unit_digits :
  (∃ x y : ℕ, possible_unit_digit x ∧ possible_unit_digit y ∧ x - y = 8) ∧
  (∀ a b : ℕ, possible_unit_digit a → possible_unit_digit b → a - b ≤ 8) :=
sorry

end greatest_difference_of_unit_digits_l2393_239392


namespace banknote_replacement_theorem_l2393_239396

/-- Represents the banknote replacement problem in the Magical Kingdom treasury --/
structure BanknoteReplacement where
  total_banknotes : ℕ
  machine_startup_cost : ℕ
  major_repair_cost : ℕ
  post_repair_capacity : ℕ
  budget : ℕ

/-- Calculates the number of banknotes replaced in a given number of days --/
def banknotes_replaced (br : BanknoteReplacement) (days : ℕ) : ℕ :=
  sorry

/-- Checks if all banknotes can be replaced within the budget --/
def can_replace_all (br : BanknoteReplacement) : Prop :=
  sorry

/-- The main theorem about banknote replacement --/
theorem banknote_replacement_theorem (br : BanknoteReplacement) 
  (h1 : br.total_banknotes = 3628800)
  (h2 : br.machine_startup_cost = 90000)
  (h3 : br.major_repair_cost = 700000)
  (h4 : br.post_repair_capacity = 1000000)
  (h5 : br.budget = 1000000) :
  (banknotes_replaced br 3 ≥ br.total_banknotes * 9 / 10) ∧
  (can_replace_all br) :=
sorry

end banknote_replacement_theorem_l2393_239396


namespace max_books_borrowed_l2393_239319

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℕ) (h1 : total_students = 20) (h2 : zero_books = 3) (h3 : one_book = 9) 
  (h4 : two_books = 4) (h5 : avg_books = 2) : 
  ∃ (max_books : ℕ), max_books = 14 ∧ 
  ∀ (student_books : ℕ), student_books ≤ max_books ∧
  (zero_books * 0 + one_book * 1 + two_books * 2 + 
   (total_students - zero_books - one_book - two_books) * 3 + 
   (max_books - 3) ≤ total_students * avg_books) :=
by sorry

end max_books_borrowed_l2393_239319


namespace min_value_of_expression_lower_bound_achievable_l2393_239331

theorem min_value_of_expression (x y : ℝ) : (x * y + 1)^2 + (x - y)^2 ≥ 1 := by
  sorry

theorem lower_bound_achievable : ∃ x y : ℝ, (x * y + 1)^2 + (x - y)^2 = 1 := by
  sorry

end min_value_of_expression_lower_bound_achievable_l2393_239331


namespace power_sum_theorem_l2393_239364

theorem power_sum_theorem (k : ℕ) :
  (∃ (n m : ℕ), m ≥ 2 ∧ 3^k + 5^k = n^m) → k = 1 :=
by sorry

end power_sum_theorem_l2393_239364


namespace pencil_count_multiple_of_ten_l2393_239339

/-- Given that 1230 pens and some pencils are distributed among students, 
    with each student receiving the same number of pens and pencils, 
    and the maximum number of students is 10, 
    prove that the total number of pencils is a multiple of 10. -/
theorem pencil_count_multiple_of_ten (total_pens : ℕ) (total_pencils : ℕ) (num_students : ℕ) :
  total_pens = 1230 →
  num_students ≤ 10 →
  num_students ∣ total_pens →
  num_students ∣ total_pencils →
  num_students = 10 →
  10 ∣ total_pencils :=
by sorry

end pencil_count_multiple_of_ten_l2393_239339


namespace arithmetic_expression_evaluation_l2393_239313

theorem arithmetic_expression_evaluation : 2 + (4 * 3 - 2) / 2 * 3 + 5 = 22 := by
  sorry

end arithmetic_expression_evaluation_l2393_239313


namespace football_shape_area_l2393_239395

/-- The area of the football-shaped region formed by two circular sectors -/
theorem football_shape_area 
  (r1 : ℝ) 
  (r2 : ℝ) 
  (h1 : r1 = 2 * Real.sqrt 2) 
  (h2 : r2 = 2) 
  (θ : ℝ) 
  (h3 : θ = π / 2) : 
  (θ / (2 * π)) * π * r1^2 - (θ / (2 * π)) * π * r2^2 = π := by
  sorry

end football_shape_area_l2393_239395


namespace rectangle_quadrilateral_inequality_l2393_239376

theorem rectangle_quadrilateral_inequality 
  (m n a b c d : ℝ) 
  (hm : m > 0) 
  (hn : n > 0) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (h_rectangle : ∃ (x y z s t u v w : ℝ), 
    x + w = m ∧ y + z = n ∧ s + t = n ∧ u + v = m ∧
    a^2 = x^2 + y^2 ∧ b^2 = z^2 + s^2 ∧ c^2 = t^2 + u^2 ∧ d^2 = v^2 + w^2) :
  1 ≤ (a^2 + b^2 + c^2 + d^2) / (m^2 + n^2) ∧ (a^2 + b^2 + c^2 + d^2) / (m^2 + n^2) ≤ 2 :=
by sorry

end rectangle_quadrilateral_inequality_l2393_239376


namespace inverse_function_decomposition_l2393_239365

noncomputable section

def PeriodOn (h : ℝ → ℝ) (d : ℝ) : Prop :=
  ∀ x, h (x + d) = h x

def IsPeriodic (h : ℝ → ℝ) : Prop :=
  ∃ d ≠ 0, PeriodOn h d

def MutuallyInverse (f g : ℝ → ℝ) : Prop :=
  (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

theorem inverse_function_decomposition
  (f g : ℝ → ℝ)
  (h : ℝ → ℝ)
  (k : ℝ)
  (h_inv : MutuallyInverse f g)
  (h_periodic : IsPeriodic h)
  (h_decomp : ∀ x, f x = k * x + h x) :
  ∃ p : ℝ → ℝ, (IsPeriodic p) ∧ (∀ y, g y = (1/k) * y + p y) :=
sorry

end inverse_function_decomposition_l2393_239365


namespace square_root_squared_l2393_239316

theorem square_root_squared (x : ℝ) (hx : x = 49) : (Real.sqrt x)^2 = x := by
  sorry

end square_root_squared_l2393_239316
