import Mathlib

namespace quadratic_symmetry_l69_6939

def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry 
  (a b c : ℝ) 
  (h_symmetry : ∀ x, p a b c x = p a b c (30 - x))
  (h_p25 : p a b c 25 = 9)
  (h_p0 : p a b c 0 = 1) :
  p a b c 5 = 9 := by
sorry

end quadratic_symmetry_l69_6939


namespace line_through_point_with_equal_intercepts_l69_6971

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (a ≠ 0 ∨ b ≠ 0)

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a line has equal intercepts on both axes --/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  (l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c/l.a = -l.c/l.b) ∨
  (l.a = 0 ∧ l.b = 0 ∧ l.c = 0)

/-- The main theorem --/
theorem line_through_point_with_equal_intercepts :
  ∃ (l₁ l₂ : Line),
    (Point.liesOn ⟨1, 5⟩ l₁) ∧
    (Point.liesOn ⟨1, 5⟩ l₂) ∧
    l₁.hasEqualIntercepts ∧
    l₂.hasEqualIntercepts ∧
    ((l₁.a = 1 ∧ l₁.b = 1 ∧ l₁.c = -6) ∨
     (l₂.a = 5 ∧ l₂.b = -1 ∧ l₂.c = 0)) :=
by sorry

end line_through_point_with_equal_intercepts_l69_6971


namespace east_northwest_angle_l69_6982

/-- Given a circle with ten equally spaced rays, where one ray points due North,
    the smaller angle between the rays pointing East and Northwest is 36°. -/
theorem east_northwest_angle (n : ℕ) (ray_angle : ℝ) : 
  n = 10 ∧ ray_angle = 360 / n → 36 = ray_angle := by sorry

end east_northwest_angle_l69_6982


namespace abs_eq_sqrt_square_l69_6922

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end abs_eq_sqrt_square_l69_6922


namespace complex_division_simplification_l69_6946

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  (2 * i) / (1 + i) = 1 + i :=
by sorry

end complex_division_simplification_l69_6946


namespace arithmetic_proof_l69_6941

theorem arithmetic_proof : 4 * (8 - 6) - 7 = 1 := by
  sorry

end arithmetic_proof_l69_6941


namespace smallest_prime_20_less_than_square_l69_6978

theorem smallest_prime_20_less_than_square : ∃ (m : ℕ), 
  (∀ (n : ℕ), n > 0 ∧ Nat.Prime n ∧ (∃ (k : ℕ), n = k^2 - 20) → n ≥ 5) ∧
  5 > 0 ∧ Nat.Prime 5 ∧ 5 = m^2 - 20 :=
by sorry

end smallest_prime_20_less_than_square_l69_6978


namespace quadratic_minimum_l69_6992

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 10

/-- The point where the minimum occurs -/
def min_point : ℝ := -4

theorem quadratic_minimum :
  ∀ x : ℝ, f x ≥ f min_point :=
sorry

end quadratic_minimum_l69_6992


namespace binomial_coefficient_modulo_prime_l69_6928

theorem binomial_coefficient_modulo_prime (p n q : ℕ) : 
  Prime p → 
  0 < n → 
  0 < q → 
  (n ≠ q * (p - 1) → Nat.choose n (p - 1) % p = 0) ∧
  (n = q * (p - 1) → Nat.choose n (p - 1) % p = 1) :=
by sorry

end binomial_coefficient_modulo_prime_l69_6928


namespace gcd_of_consecutive_odd_terms_l69_6965

theorem gcd_of_consecutive_odd_terms (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  Nat.gcd ((n + 1) * (n + 3) * (n + 7) * (n + 9)) 15 = 15 :=
by sorry

end gcd_of_consecutive_odd_terms_l69_6965


namespace point_on_line_through_two_points_l69_6927

/-- A point lies on a line if it satisfies the line equation --/
def point_on_line (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

/-- The theorem statement --/
theorem point_on_line_through_two_points :
  point_on_line 1 2 5 10 3 6 := by
  sorry

end point_on_line_through_two_points_l69_6927


namespace sam_and_tina_distances_l69_6972

/-- Calculates the distance traveled given speed and time -/
def distance (speed time : ℝ) : ℝ := speed * time

theorem sam_and_tina_distances 
  (marguerite_distance marguerite_time sam_time tina_time : ℝ) 
  (marguerite_distance_positive : marguerite_distance > 0)
  (marguerite_time_positive : marguerite_time > 0)
  (sam_time_positive : sam_time > 0)
  (tina_time_positive : tina_time > 0)
  (h_marguerite_distance : marguerite_distance = 150)
  (h_marguerite_time : marguerite_time = 3)
  (h_sam_time : sam_time = 4)
  (h_tina_time : tina_time = 2) :
  let marguerite_speed := marguerite_distance / marguerite_time
  (distance marguerite_speed sam_time = 200) ∧ 
  (distance marguerite_speed tina_time = 100) := by
  sorry

end sam_and_tina_distances_l69_6972


namespace fraction_product_cubed_main_proof_l69_6979

theorem fraction_product_cubed (a b c d : ℚ) : 
  (a / b) ^ 3 * (c / d) ^ 3 = ((a * c) / (b * d)) ^ 3 :=
by sorry

theorem main_proof : (5 / 8 : ℚ) ^ 3 * (4 / 9 : ℚ) ^ 3 = 125 / 5832 :=
by sorry

end fraction_product_cubed_main_proof_l69_6979


namespace solution_satisfies_system_l69_6901

/-- Prove that (1, 1, 1) is the solution to the given system of equations -/
theorem solution_satisfies_system :
  let x₁ : ℝ := 1
  let x₂ : ℝ := 1
  let x₃ : ℝ := 1
  (x₁ + 2*x₂ + x₃ = 4) ∧
  (3*x₁ - 5*x₂ + 3*x₃ = 1) ∧
  (2*x₁ + 7*x₂ - x₃ = 8) := by
  sorry

end solution_satisfies_system_l69_6901


namespace decreasing_quadratic_function_l69_6952

theorem decreasing_quadratic_function (a : ℝ) :
  (∀ x < 4, (∀ y < x, x^2 + 2*(a-1)*x + 2 < y^2 + 2*(a-1)*y + 2)) →
  a ≤ -3 := by
  sorry

end decreasing_quadratic_function_l69_6952


namespace number_pattern_equality_l69_6926

theorem number_pattern_equality (n : ℕ) (h : n > 1) :
  3 * (6 * (10^n - 1) / 9)^3 = 
    8 * ((10^n - 1) / 9) * 10^(2*n+1) + 
    6 * 10^(2*n) + 
    2 * ((10^n - 1) / 9) * 10^(n+1) + 
    4 * 10^n + 
    8 * ((10^n - 1) / 9) := by
  sorry

end number_pattern_equality_l69_6926


namespace expression_simplification_l69_6903

theorem expression_simplification :
  ∀ p : ℝ, ((7*p+3)-3*p*5)*(2)+(5-2/4)*(8*p-12) = 20*p - 48 := by
sorry

end expression_simplification_l69_6903


namespace incorrect_calculation_l69_6964

theorem incorrect_calculation (x : ℝ) : 
  25 * ((1/25) * x^2 - (1/10) * x + 1) ≠ x^2 - (5/2) * x + 25 := by
  sorry

end incorrect_calculation_l69_6964


namespace complex_fraction_equality_l69_6961

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) + (a - b) / (a + b) = 2) :
  (a^4 + b^4) / (a^4 - b^4) + (a^4 - b^4) / (a^4 + b^4) = 2 := by
  sorry

end complex_fraction_equality_l69_6961


namespace function_symmetry_implies_m_range_l69_6962

theorem function_symmetry_implies_m_range 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h_f : ∀ x, f x = m * 4^x - 2^x) 
  (h_symmetry : ∃ x_0 : ℝ, x_0 ≠ 0 ∧ f (-x_0) = f x_0) : 
  0 < m ∧ m < 1/2 := by
  sorry

end function_symmetry_implies_m_range_l69_6962


namespace simultaneous_processing_equation_l69_6981

/-- Represents the total number of workers --/
def total_workers : ℕ := 26

/-- Represents the number of type A parts to process --/
def type_a_parts : ℕ := 2100

/-- Represents the number of type B parts to process --/
def type_b_parts : ℕ := 1200

/-- Represents the number of type A parts a worker can process per day --/
def type_a_rate : ℕ := 30

/-- Represents the number of type B parts a worker can process per day --/
def type_b_rate : ℕ := 20

/-- Theorem stating that the equation correctly represents the simultaneous processing of both types of parts --/
theorem simultaneous_processing_equation (x : ℝ) (h1 : 0 < x) (h2 : x < total_workers) :
  (type_a_parts : ℝ) / (type_a_rate * x) = (type_b_parts : ℝ) / (type_b_rate * (total_workers - x)) :=
by sorry

end simultaneous_processing_equation_l69_6981


namespace angle_bisector_length_l69_6950

/-- Given a triangle ABC, this theorem states that the length of the angle bisector
    from vertex C to the opposite side AB can be calculated using the formula:
    l₃ = (2ab)/(a+b) * cos(C/2), where a and b are the lengths of sides BC and AC
    respectively, and C is the angle at vertex C. -/
theorem angle_bisector_length (a b C l₃ : ℝ) :
  (a > 0) → (b > 0) → (C > 0) → (C < π) →
  l₃ = (2 * a * b) / (a + b) * Real.cos (C / 2) :=
by sorry

end angle_bisector_length_l69_6950


namespace circle_area_difference_l69_6974

theorem circle_area_difference : 
  let d₁ : ℝ := 30
  let r₁ : ℝ := d₁ / 2
  let r₂ : ℝ := 10
  let r₃ : ℝ := 5
  (π * r₁^2) - (π * r₂^2) - (π * r₃^2) = 100 * π := by
  sorry

end circle_area_difference_l69_6974


namespace ellipse_triangle_perimeter_l69_6902

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- Definition of the foci -/
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-4, 0) ∧ F₂ = (4, 0)

/-- Theorem: Perimeter of triangle PF₁F₂ is 18 for any point P on the ellipse -/
theorem ellipse_triangle_perimeter 
  (x y : ℝ) 
  (F₁ F₂ : ℝ × ℝ) 
  (h_ellipse : is_on_ellipse x y) 
  (h_foci : foci F₁ F₂) : 
  let P := (x, y)
  ‖P - F₁‖ + ‖P - F₂‖ + ‖F₁ - F₂‖ = 18 :=
sorry

end ellipse_triangle_perimeter_l69_6902


namespace hyperbola_quadrants_l69_6986

theorem hyperbola_quadrants (k : ℝ) : k < 0 ∧ 2 * k^2 + k - 2 = -1 → k = -1 := by
  sorry

end hyperbola_quadrants_l69_6986


namespace right_triangle_max_sum_l69_6994

theorem right_triangle_max_sum (a b c : ℝ) : 
  c = 5 →
  a ≤ 3 →
  b ≥ 3 →
  a^2 + b^2 = c^2 →
  a + b ≤ 7 :=
by sorry

end right_triangle_max_sum_l69_6994


namespace no_snow_probability_l69_6969

theorem no_snow_probability (p : ℚ) (h : p = 2/3) : (1 - p)^5 = 1/243 := by
  sorry

end no_snow_probability_l69_6969


namespace point_A_in_first_quadrant_l69_6996

-- Define the Cartesian coordinate system
def CartesianCoordinate := ℝ × ℝ

-- Define the point A
def A : CartesianCoordinate := (1, 2)

-- Define the first quadrant
def FirstQuadrant (p : CartesianCoordinate) : Prop :=
  p.1 > 0 ∧ p.2 > 0

-- Theorem statement
theorem point_A_in_first_quadrant : FirstQuadrant A := by
  sorry

end point_A_in_first_quadrant_l69_6996


namespace probability_at_least_3_of_6_l69_6959

def probability_at_least_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  Finset.sum (Finset.range (n - k + 1))
    (λ i => Nat.choose n (k + i) * p ^ (k + i) * (1 - p) ^ (n - k - i))

theorem probability_at_least_3_of_6 :
  probability_at_least_k_successes 6 3 (2/3) = 656/729 := by
  sorry

end probability_at_least_3_of_6_l69_6959


namespace product_change_l69_6900

theorem product_change (a b : ℕ) (h : (a + 3) * (b - 3) - a * b = 600) : 
  a * b - (a - 3) * (b + 3) = 618 := by
sorry

end product_change_l69_6900


namespace perfect_squares_digits_parity_l69_6954

/-- A natural number is a perfect square if it is equal to the square of some natural number. -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- The units digit of a natural number. -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The tens digit of a natural number. -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem perfect_squares_digits_parity (a b : ℕ) (x y : ℕ) :
  is_perfect_square a →
  is_perfect_square b →
  units_digit a = 1 →
  tens_digit a = x →
  units_digit b = 6 →
  tens_digit b = y →
  Even x ∧ Odd y :=
sorry

end perfect_squares_digits_parity_l69_6954


namespace time_to_write_117639_l69_6907

def digits_count (n : ℕ) : ℕ := 
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else if n < 10000 then 4
  else if n < 100000 then 5
  else 6

def total_digits (n : ℕ) : ℕ := 
  (List.range n).map digits_count |>.sum

def time_to_write (n : ℕ) (digits_per_minute : ℕ) : ℕ := 
  (total_digits n + digits_per_minute - 1) / digits_per_minute

theorem time_to_write_117639 : 
  time_to_write 117639 93 = 4 * 24 * 60 + 10 * 60 + 34 := by sorry

end time_to_write_117639_l69_6907


namespace translation_right_2_units_l69_6910

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_2_units :
  let A : Point := { x := 3, y := -2 }
  let A' : Point := translateRight A 2
  A'.x = 5 ∧ A'.y = -2 := by
  sorry

end translation_right_2_units_l69_6910


namespace percentage_problem_l69_6904

theorem percentage_problem (x : ℝ) (h : x = 300) : 
  ∃ P : ℝ, P * x / 100 = x / 3 + 110 := by sorry

end percentage_problem_l69_6904


namespace sara_pumpkins_l69_6951

/-- The number of pumpkins Sara grew -/
def pumpkins_grown : ℕ := 43

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The number of pumpkins Sara has left -/
def pumpkins_left : ℕ := pumpkins_grown - pumpkins_eaten

theorem sara_pumpkins : pumpkins_left = 20 := by
  sorry

end sara_pumpkins_l69_6951


namespace representation_2015_l69_6916

theorem representation_2015 : ∃ (a b c : ℤ), 
  a + b + c = 2015 ∧ 
  Nat.Prime a.natAbs ∧ 
  ∃ (k : ℤ), b = 3 * k ∧
  400 < c ∧ c < 500 ∧
  ¬∃ (m : ℤ), c = 3 * m :=
sorry

end representation_2015_l69_6916


namespace intersection_point_value_l69_6985

/-- Given three lines that intersect at a single point, prove that the value of a is -1 --/
theorem intersection_point_value (a : ℝ) :
  (∃! p : ℝ × ℝ, (a * p.1 + 2 * p.2 + 8 = 0) ∧
                 (4 * p.1 + 3 * p.2 = 10) ∧
                 (2 * p.1 - p.2 = 10)) →
  a = -1 := by
sorry

end intersection_point_value_l69_6985


namespace min_value_of_sum_of_squares_l69_6919

theorem min_value_of_sum_of_squares (x y : ℝ) (h : 2 * (x^2 + y^2) = x^2 + y + x*y) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b : ℝ), 2 * (a^2 + b^2) = a^2 + b + a*b → x^2 + y^2 ≥ m :=
sorry

end min_value_of_sum_of_squares_l69_6919


namespace collinearity_ABD_l69_6983

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two non-zero vectors are not collinear -/
def not_collinear (a b : V) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ ¬∃ (r : ℝ), a = r • b

/-- Three points are collinear if the vector from the first to the third is a scalar multiple of the vector from the first to the second -/
def collinear (A B D : V) : Prop := ∃ (t : ℝ), D - A = t • (B - A)

theorem collinearity_ABD 
  (a b : V) 
  (h_not_collinear : not_collinear a b)
  (h_AB : B - A = a + b)
  (h_BC : C - B = a + 10 • b)
  (h_CD : D - C = 3 • (a - 2 • b)) :
  collinear A B D :=
sorry

end collinearity_ABD_l69_6983


namespace adjusted_work_hours_sufficient_l69_6958

/-- Proves that working 27 hours per week for 9 weeks will result in at least $3000 earnings,
    given the initial plan of 20 hours per week for 12 weeks to earn $3000. -/
theorem adjusted_work_hours_sufficient
  (initial_hours_per_week : ℕ)
  (initial_weeks : ℕ)
  (target_earnings : ℕ)
  (missed_weeks : ℕ)
  (adjusted_hours_per_week : ℕ)
  (h1 : initial_hours_per_week = 20)
  (h2 : initial_weeks = 12)
  (h3 : target_earnings = 3000)
  (h4 : missed_weeks = 3)
  (h5 : adjusted_hours_per_week = 27) :
  (adjusted_hours_per_week : ℚ) * (initial_weeks - missed_weeks) ≥ (target_earnings : ℚ) := by
  sorry

#check adjusted_work_hours_sufficient

end adjusted_work_hours_sufficient_l69_6958


namespace tangent_line_parallel_proof_l69_6908

/-- The parabola y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- The slope of the tangent line to the parabola at point (a, a^2) -/
def tangent_slope (a : ℝ) : ℝ := 2 * a

/-- The slope of the line 2x - y + 4 = 0 -/
def given_line_slope : ℝ := 2

/-- The equation of the tangent line at point (a, a^2) -/
def tangent_line_eq (a : ℝ) (x y : ℝ) : Prop :=
  y - a^2 = tangent_slope a * (x - a)

theorem tangent_line_parallel_proof (a : ℝ) :
  tangent_slope a = given_line_slope →
  a = 1 ∧
  ∀ x y : ℝ, tangent_line_eq a x y ↔ 2*x - y - 1 = 0 :=
by sorry

end tangent_line_parallel_proof_l69_6908


namespace police_coverage_l69_6934

-- Define the set of intersections
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

-- Define the streets as sets of intersections
def horizontal1 : Set Intersection := {Intersection.A, Intersection.B, Intersection.C, Intersection.D}
def horizontal2 : Set Intersection := {Intersection.E, Intersection.F, Intersection.G}
def horizontal3 : Set Intersection := {Intersection.H, Intersection.I, Intersection.J, Intersection.K}
def vertical1 : Set Intersection := {Intersection.A, Intersection.E, Intersection.H}
def vertical2 : Set Intersection := {Intersection.B, Intersection.F, Intersection.I}
def vertical3 : Set Intersection := {Intersection.D, Intersection.G, Intersection.J}
def diagonal1 : Set Intersection := {Intersection.H, Intersection.F, Intersection.C}
def diagonal2 : Set Intersection := {Intersection.C, Intersection.G, Intersection.K}

-- Define the set of all streets
def allStreets : Set (Set Intersection) :=
  {horizontal1, horizontal2, horizontal3, vertical1, vertical2, vertical3, diagonal1, diagonal2}

-- Define the set of intersections with police officers
def policeLocations : Set Intersection := {Intersection.B, Intersection.G, Intersection.H}

-- Theorem statement
theorem police_coverage :
  ∀ street ∈ allStreets, ∃ intersection ∈ street, intersection ∈ policeLocations :=
by sorry

end police_coverage_l69_6934


namespace square_difference_153_147_l69_6968

theorem square_difference_153_147 : 153^2 - 147^2 = 1800 := by sorry

end square_difference_153_147_l69_6968


namespace flowchart_properties_l69_6912

/-- A flowchart is a type of diagram that represents a process or algorithm. -/
def Flowchart : Type := sorry

/-- A block in a flowchart represents a step or decision in the process. -/
def Block : Type := sorry

/-- The start block of a flowchart. -/
def start_block : Block := sorry

/-- The end block of a flowchart. -/
def end_block : Block := sorry

/-- An input block in a flowchart. -/
def input_block : Block := sorry

/-- An output block in a flowchart. -/
def output_block : Block := sorry

/-- A decision block in a flowchart. -/
def decision_block : Block := sorry

/-- A function that checks if a flowchart has both start and end blocks. -/
def has_start_and_end (f : Flowchart) : Prop := sorry

/-- A function that checks if input blocks are only after the start block. -/
def input_after_start (f : Flowchart) : Prop := sorry

/-- A function that checks if output blocks are only before the end block. -/
def output_before_end (f : Flowchart) : Prop := sorry

/-- A function that checks if decision blocks are the only ones with multiple exit points. -/
def decision_multiple_exits (f : Flowchart) : Prop := sorry

/-- A function that checks if the way conditions are described in decision blocks is unique. -/
def unique_decision_conditions (f : Flowchart) : Prop := sorry

theorem flowchart_properties (f : Flowchart) :
  (has_start_and_end f ∧ 
   input_after_start f ∧ 
   output_before_end f ∧ 
   decision_multiple_exits f) ∧
  ¬(unique_decision_conditions f) := by sorry

end flowchart_properties_l69_6912


namespace time_per_cut_l69_6932

/-- Given 3 pieces of wood, each cut into 3 sections, in 18 minutes total, prove the time per cut is 3 minutes -/
theorem time_per_cut (num_pieces : ℕ) (sections_per_piece : ℕ) (total_time : ℕ) :
  num_pieces = 3 →
  sections_per_piece = 3 →
  total_time = 18 →
  (total_time : ℚ) / (num_pieces * (sections_per_piece - 1)) = 3 := by
  sorry

end time_per_cut_l69_6932


namespace quadratic_inequalities_solution_sets_l69_6976

theorem quadratic_inequalities_solution_sets :
  (∀ x : ℝ, -3 * x^2 + x + 1 > 0 ↔ x ∈ Set.Ioo ((1 - Real.sqrt 13) / 6) ((1 + Real.sqrt 13) / 6)) ∧
  (∀ x : ℝ, x^2 - 2*x + 1 ≤ 0 ↔ x = 1) := by
  sorry

end quadratic_inequalities_solution_sets_l69_6976


namespace derivative_of_f_l69_6989

-- Define the function f
def f (x : ℝ) : ℝ := (3*x - 5)^2

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x => 6 * (3*x - 5) := by sorry

end derivative_of_f_l69_6989


namespace trapezoid_theorem_l69_6980

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Checks if a point is on a line segment -/
def isOnSegment (P Q R : Point) : Prop := sorry

/-- Checks if two line segments intersect -/
def intersect (P Q R S : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

/-- Main theorem -/
theorem trapezoid_theorem (ABCD : Trapezoid) (E O P : Point) :
  isOnSegment ABCD.A E ABCD.D →
  distance ABCD.A E = distance ABCD.B ABCD.C →
  intersect ABCD.C ABCD.A ABCD.B ABCD.D →
  intersect ABCD.C E ABCD.B ABCD.D →
  intersect ABCD.C ABCD.A ABCD.B O →
  intersect ABCD.C E ABCD.B P →
  distance ABCD.B O = distance P ABCD.D →
  (distance ABCD.A ABCD.D)^2 = (distance ABCD.B ABCD.C)^2 + (distance ABCD.A ABCD.D) * (distance ABCD.B ABCD.C) := by
  sorry

end trapezoid_theorem_l69_6980


namespace axiom_1_l69_6990

-- Define the types for points, lines, and planes
variable {Point Line Plane : Type}

-- Define the relations for points being on lines and planes
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)

-- Define the subset relation for lines and planes
variable (line_on_plane : Line → Plane → Prop)

-- State the theorem
theorem axiom_1 (l : Line) (α : Plane) :
  (∃ (A B : Point), on_line A l ∧ on_line B l ∧ on_plane A α ∧ on_plane B α) →
  line_on_plane l α :=
sorry

end axiom_1_l69_6990


namespace no_root_greater_than_three_l69_6975

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem no_root_greater_than_three (a b c : ℝ) :
  (quadratic a b c (-1) = -1) →
  (quadratic a b c 0 = 2) →
  (quadratic a b c 2 = 2) →
  (quadratic a b c 4 = -6) →
  ∀ x > 3, quadratic a b c x ≠ 0 :=
by
  sorry

end no_root_greater_than_three_l69_6975


namespace symmetry_implies_phase_shift_l69_6914

/-- Given a function f(x) = sin x + √3 cos x, prove that if y = f(x + φ) is symmetric about x = 0, then φ = π/6 -/
theorem symmetry_implies_phase_shift (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin x + Real.sqrt 3 * Real.cos x) →
  (∀ x, f (x + φ) = f (-x + φ)) →
  φ = π / 6 := by
  sorry

end symmetry_implies_phase_shift_l69_6914


namespace remaining_students_l69_6931

/-- The number of remaining students in the class -/
def n : ℕ := sorry

/-- The weight of the student who left the class -/
def weight_left : ℝ := 45

/-- The increase in average weight after the student left -/
def weight_increase : ℝ := 0.2

/-- The average weight of the remaining students -/
def avg_weight_remaining : ℝ := 57

/-- Theorem stating that the number of remaining students is 59 -/
theorem remaining_students : n = 59 := by
  sorry

end remaining_students_l69_6931


namespace function_values_and_range_l69_6940

noncomputable def f (b c x : ℝ) : ℝ := -1/3 * x^3 + b * x^2 + c * x + b * c

noncomputable def g (a x : ℝ) : ℝ := a * x^2 - 2 * Real.log x

theorem function_values_and_range :
  ∀ b c : ℝ,
  (∃ x : ℝ, f b c x = -4/3 ∧ ∀ y : ℝ, f b c y ≤ f b c x) →
  (b = -1 ∧ c = 3) ∧
  ∀ a : ℝ,
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 3 ∧ 0 < x₂ ∧ x₂ < 3 ∧ |f b c x₁ - g a x₂| < 1) →
  (2 * Real.log 3 - 13) / 9 ≤ a ∧ a ≤ (6 * Real.log 3 - 1) / 27 :=
by sorry

end function_values_and_range_l69_6940


namespace man_speed_is_4_l69_6953

/-- Represents the speed of water in a stream. -/
def stream_speed : ℝ := sorry

/-- Represents the speed of a man swimming in still water. -/
def man_speed : ℝ := sorry

/-- The distance traveled downstream. -/
def downstream_distance : ℝ := 30

/-- The distance traveled upstream. -/
def upstream_distance : ℝ := 18

/-- The time taken for both downstream and upstream swims. -/
def swim_time : ℝ := 6

/-- Theorem stating that the man's speed in still water is 4 km/h. -/
theorem man_speed_is_4 : 
  downstream_distance = (man_speed + stream_speed) * swim_time ∧ 
  upstream_distance = (man_speed - stream_speed) * swim_time → 
  man_speed = 4 := by sorry

end man_speed_is_4_l69_6953


namespace compute_expression_l69_6933

theorem compute_expression : 10 + 8 * (2 - 9)^2 = 402 := by
  sorry

end compute_expression_l69_6933


namespace total_persimmons_l69_6945

/-- Given that the total weight of persimmons is 3 kg and 5 persimmons weigh 1 kg,
    prove that the total number of persimmons is 15. -/
theorem total_persimmons (total_weight : ℝ) (weight_of_five : ℝ) (num_in_five : ℕ) :
  total_weight = 3 →
  weight_of_five = 1 →
  num_in_five = 5 →
  (total_weight / weight_of_five) * num_in_five = 15 := by
  sorry

#check total_persimmons

end total_persimmons_l69_6945


namespace grade_A_students_over_three_years_l69_6909

theorem grade_A_students_over_three_years 
  (total : ℕ) 
  (first_year : ℕ) 
  (growth_rate : ℝ) 
  (h1 : total = 728)
  (h2 : first_year = 200)
  (h3 : first_year + first_year * (1 + growth_rate) + first_year * (1 + growth_rate)^2 = total) :
  first_year + first_year * (1 + growth_rate) + first_year * (1 + growth_rate)^2 = 728 := by
sorry

end grade_A_students_over_three_years_l69_6909


namespace collinear_points_x_value_l69_6923

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_x_value :
  let A : Point := ⟨3, -2⟩
  let B : Point := ⟨-9, 4⟩
  let C : Point := ⟨x, 0⟩
  collinear A B C → x = -1 := by
  sorry

end collinear_points_x_value_l69_6923


namespace chord_length_l69_6948

-- Define the circle and chord
def circle_radius : ℝ := 5
def center_to_chord : ℝ := 4

-- Theorem statement
theorem chord_length :
  ∀ (chord_length : ℝ),
  circle_radius = 5 ∧
  center_to_chord = 4 →
  chord_length = 6 :=
by sorry

end chord_length_l69_6948


namespace rectangle_area_from_diagonal_l69_6995

/-- Theorem: Area of a rectangle with length thrice its width and diagonal x -/
theorem rectangle_area_from_diagonal (x : ℝ) (h : x > 0) : 
  ∃ w l : ℝ, w > 0 ∧ l = 3 * w ∧ w^2 + l^2 = x^2 ∧ w * l = (3/10) * x^2 := by
  sorry

end rectangle_area_from_diagonal_l69_6995


namespace higher_selling_price_is_360_l69_6987

/-- The higher selling price of an article, given its cost and profit conditions -/
def higherSellingPrice (cost : ℚ) (lowerPrice : ℚ) : ℚ :=
  let profitAtLowerPrice := lowerPrice - cost
  let additionalProfit := (5 / 100) * cost
  cost + profitAtLowerPrice + additionalProfit

/-- Theorem stating that the higher selling price is 360, given the conditions -/
theorem higher_selling_price_is_360 :
  higherSellingPrice 400 340 = 360 := by
  sorry

#eval higherSellingPrice 400 340

end higher_selling_price_is_360_l69_6987


namespace solve_puppy_problem_l69_6997

def puppyProblem (initialPuppies : ℕ) (givenAway : ℕ) (kept : ℕ) (sellingPrice : ℕ) (profit : ℕ) : Prop :=
  let remainingAfterGiveaway := initialPuppies - givenAway
  let soldPuppies := remainingAfterGiveaway - kept
  let revenue := soldPuppies * sellingPrice
  let amountToStud := revenue - profit
  amountToStud = 300

theorem solve_puppy_problem :
  puppyProblem 8 4 1 600 1500 := by
  sorry

end solve_puppy_problem_l69_6997


namespace complex_coordinate_proof_l69_6920

theorem complex_coordinate_proof (z : ℂ) : (z - 2*I) * (1 + I) = I → z = 1/2 + 5/2*I := by
  sorry

end complex_coordinate_proof_l69_6920


namespace arithmetic_progression_square_sum_l69_6993

/-- For real numbers a, b, c forming an arithmetic progression,
    3(a² + b² + c²) = 6(a-b)² + (a+b+c)² -/
theorem arithmetic_progression_square_sum (a b c : ℝ) 
  (h : a + c = 2 * b) : 
  3 * (a^2 + b^2 + c^2) = 6 * (a - b)^2 + (a + b + c)^2 := by
  sorry

end arithmetic_progression_square_sum_l69_6993


namespace triangle_ABC_proof_l69_6944

theorem triangle_ABC_proof (A B C : Real) (a b c : Real) :
  -- Conditions
  A + B + C = π →
  2 * Real.sin (B + C) ^ 2 - 3 * Real.cos A = 0 →
  B = π / 4 →
  a = 2 * Real.sqrt 3 →
  -- Conclusions
  A = π / 3 ∧ c = Real.sqrt 6 + Real.sqrt 2 := by
  sorry


end triangle_ABC_proof_l69_6944


namespace original_price_calculation_l69_6936

theorem original_price_calculation (a b : ℝ) : 
  ∃ x : ℝ, (x - a) * (1 - 0.4) = b ∧ x = a + (5/3) * b := by
  sorry

end original_price_calculation_l69_6936


namespace base4_21012_equals_582_l69_6935

/-- Converts a base 4 digit to its base 10 equivalent -/
def base4_digit_to_base10 (d : Nat) : Nat :=
  if d < 4 then d else 0

/-- Represents the base 4 number 21012 -/
def base4_number : List Nat := [2, 1, 0, 1, 2]

/-- Converts a list of base 4 digits to a base 10 number -/
def base4_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + base4_digit_to_base10 d * (4 ^ (digits.length - 1 - i))) 0

theorem base4_21012_equals_582 :
  base4_to_base10 base4_number = 582 := by
  sorry

end base4_21012_equals_582_l69_6935


namespace system_solution_l69_6942

theorem system_solution (x y : Real) (k₁ k₂ : Int) : 
  (Real.sqrt 2 * Real.sin x = Real.sin y) →
  (Real.sqrt 2 * Real.cos x = Real.sqrt 3 * Real.cos y) →
  (∃ n₁ n₂ : Int, x = n₁ * π / 6 + k₂ * π ∧ y = n₂ * π / 4 + k₁ * π ∧ 
   (n₁ = 1 ∨ n₁ = -1) ∧ (n₂ = 1 ∨ n₂ = -1) ∧ k₁ % 2 = k₂ % 2) :=
by sorry

end system_solution_l69_6942


namespace shaded_area_is_108pi_l69_6918

/-- Represents a point on a line -/
structure Point :=
  (x : ℝ)

/-- Represents a semicircle -/
structure Semicircle :=
  (center : Point)
  (radius : ℝ)

/-- The configuration of points and semicircles -/
structure Configuration :=
  (A B C D E F : Point)
  (AF AB BC CD DE EF : Semicircle)

/-- The conditions of the problem -/
def problem_conditions (config : Configuration) : Prop :=
  let {A, B, C, D, E, F, AF, AB, BC, CD, DE, EF} := config
  (B.x - A.x = 6) ∧ 
  (C.x - B.x = 6) ∧ 
  (D.x - C.x = 6) ∧ 
  (E.x - D.x = 6) ∧ 
  (F.x - E.x = 6) ∧
  (AF.radius = 15) ∧
  (AB.radius = 3) ∧
  (BC.radius = 3) ∧
  (CD.radius = 3) ∧
  (DE.radius = 3) ∧
  (EF.radius = 3)

/-- The area of the shaded region -/
def shaded_area (config : Configuration) : ℝ :=
  sorry  -- Actual calculation would go here

/-- The theorem stating that the shaded area is 108π -/
theorem shaded_area_is_108pi (config : Configuration) 
  (h : problem_conditions config) : shaded_area config = 108 * Real.pi := by
  sorry

end shaded_area_is_108pi_l69_6918


namespace expression_simplification_and_evaluation_l69_6970

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (((3 / (x - 1)) - x - 1) / ((x^2 - 4*x + 4) / (x - 1))) = (2 + x) / (2 - x) ∧
  (((3 / (0 - 1)) - 0 - 1) / ((0^2 - 4*0 + 4) / (0 - 1))) = 1 := by
  sorry

end expression_simplification_and_evaluation_l69_6970


namespace finite_decimals_are_rational_l69_6947

theorem finite_decimals_are_rational : 
  ∀ x : ℝ, (∃ n : ℕ, ∃ m : ℤ, x = m / (10 ^ n)) → ∃ a b : ℤ, x = a / b ∧ b ≠ 0 :=
sorry

end finite_decimals_are_rational_l69_6947


namespace green_guards_with_shields_l69_6977

theorem green_guards_with_shields (total : ℝ) (green : ℝ) (yellow : ℝ) (special : ℝ) 
  (h1 : green = (3/8) * total)
  (h2 : yellow = (5/8) * total)
  (h3 : special = (1/5) * total)
  (h4 : ∃ (r s : ℝ), (green * (r/s) + yellow * (r/(3*s)) = special) ∧ (r/s > 0) ∧ (s ≠ 0)) :
  ∃ (r s : ℝ), (r/s = 12/35) ∧ (green * (r/s) = (3/5) * special) := by
  sorry

end green_guards_with_shields_l69_6977


namespace quadrupled_base_and_exponent_l69_6921

theorem quadrupled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4 * a)^(4 * b) = a^b * x^(2 * b) → x = 16 * a^(3/2) := by
  sorry

end quadrupled_base_and_exponent_l69_6921


namespace divisibility_by_x2_plus_x_plus_1_l69_6911

theorem divisibility_by_x2_plus_x_plus_1 (n : ℕ) (hn : n > 0) :
  ∃ q : Polynomial ℚ, (X + 1 : Polynomial ℚ)^(2*n + 1) + X^(n + 2) = (X^2 + X + 1) * q := by
  sorry

end divisibility_by_x2_plus_x_plus_1_l69_6911


namespace sum_remainder_zero_l69_6937

theorem sum_remainder_zero : (9152 + 9153 + 9154 + 9155 + 9156) % 10 = 0 := by
  sorry

end sum_remainder_zero_l69_6937


namespace no_significant_relationship_l69_6973

-- Define the contingency table data
def boys_enthusiasts : ℕ := 45
def boys_non_enthusiasts : ℕ := 10
def girls_enthusiasts : ℕ := 30
def girls_non_enthusiasts : ℕ := 15

-- Define the total number of students
def total_students : ℕ := boys_enthusiasts + boys_non_enthusiasts + girls_enthusiasts + girls_non_enthusiasts

-- Define the K² calculation function
def calculate_k_squared (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n : ℚ) * (a * d - b * c : ℚ)^2 / ((a + b : ℚ) * (c + d : ℚ) * (a + c : ℚ) * (b + d : ℚ))

-- Define the critical value for 95% confidence
def critical_value : ℚ := 3841 / 1000

-- Theorem statement
theorem no_significant_relationship : 
  calculate_k_squared boys_enthusiasts boys_non_enthusiasts girls_enthusiasts girls_non_enthusiasts < critical_value := by
  sorry


end no_significant_relationship_l69_6973


namespace arithmetic_sequence_slope_l69_6956

/-- For an arithmetic sequence {a_n} where a_2 - a_4 = 2, 
    the slope of the line containing points (n, a_n) is -1 -/
theorem arithmetic_sequence_slope (a : ℕ → ℝ) (h : a 2 - a 4 = 2) :
  ∃ b : ℝ, ∀ n : ℕ, a n = -n + b := by
  sorry

end arithmetic_sequence_slope_l69_6956


namespace parabola_perpendicular_range_l69_6949

/-- Given point A(0,2) and two points B and C on the parabola y^2 = x + 4 such that AB ⟂ BC,
    the y-coordinate of point C satisfies y ≤ 0 or y ≥ 4. -/
theorem parabola_perpendicular_range (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (0, 2)
  let on_parabola (p : ℝ × ℝ) := p.2^2 = p.1 + 4
  let perpendicular (p q r : ℝ × ℝ) := 
    (q.2 - p.2) * (r.2 - q.2) = -(q.1 - p.1) * (r.1 - q.1)
  on_parabola B ∧ on_parabola C ∧ perpendicular A B C →
  C.2 ≤ 0 ∨ C.2 ≥ 4 := by
sorry

end parabola_perpendicular_range_l69_6949


namespace annika_hiking_rate_l69_6917

/-- Annika's hiking problem -/
theorem annika_hiking_rate (initial_distance : Real) (total_east_distance : Real) (return_time : Real) :
  initial_distance = 2.75 →
  total_east_distance = 3.625 →
  return_time = 45 →
  let additional_east := total_east_distance - initial_distance
  let total_distance := initial_distance + 2 * additional_east
  total_distance / return_time * 60 = 10 := by
  sorry

end annika_hiking_rate_l69_6917


namespace tangent_line_passes_fixed_point_l69_6915

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 2

-- Define a point P on line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line_l : line_l x y

-- Define the tangent condition
def is_tangent (P : Point_P) (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
  (∃ t : ℝ, A.1 = P.x + t * (P.y - 2) ∧ A.2 = P.y - t * P.x) ∧
  (∃ t : ℝ, B.1 = P.x + t * (P.y - 2) ∧ B.2 = P.y - t * P.x)

-- Theorem statement
theorem tangent_line_passes_fixed_point (P : Point_P) (A B : ℝ × ℝ) :
  is_tangent P A B →
  ∃ t : ℝ, t * A.1 + (1 - t) * B.1 = 1/2 ∧ t * A.2 + (1 - t) * B.2 = 1/2 :=
sorry

end tangent_line_passes_fixed_point_l69_6915


namespace largest_integer_less_than_100_remainder_5_mod_8_l69_6967

theorem largest_integer_less_than_100_remainder_5_mod_8 :
  ∀ n : ℕ, n < 100 → n % 8 = 5 → n ≤ 99 :=
by
  sorry

end largest_integer_less_than_100_remainder_5_mod_8_l69_6967


namespace farm_field_area_l69_6913

/-- Represents the farm field ploughing scenario -/
structure FarmField where
  planned_daily_rate : ℝ
  actual_daily_rate : ℝ
  extra_days : ℕ
  remaining_area : ℝ

/-- Calculates the total area of the farm field -/
def total_area (f : FarmField) : ℝ :=
  sorry

/-- Theorem stating that the total area of the farm field is 312 hectares -/
theorem farm_field_area (f : FarmField) 
  (h1 : f.planned_daily_rate = 260)
  (h2 : f.actual_daily_rate = 85)
  (h3 : f.extra_days = 2)
  (h4 : f.remaining_area = 40) :
  total_area f = 312 :=
sorry

end farm_field_area_l69_6913


namespace ellipse_area_l69_6938

/-- The area of an ellipse with semi-major axis a and semi-minor axis b -/
theorem ellipse_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∫ x in -a..a, 2 * b * Real.sqrt (1 - x^2 / a^2)) = π * a * b :=
sorry

end ellipse_area_l69_6938


namespace square_rotation_octagon_l69_6925

/-- Represents a regular polygon with n sides -/
structure RegularPolygon where
  sides : ℕ
  mk_sides_pos : sides > 0

/-- Represents a square -/
structure Square

/-- Represents the position of an object on a square -/
inductive Position
  | Top
  | Right
  | Bottom
  | Left

/-- Calculates the inner angle of a regular polygon -/
def inner_angle (p : RegularPolygon) : ℚ :=
  (p.sides - 2 : ℚ) * 180 / p.sides

/-- Calculates the rotation per movement when a square rolls around a regular polygon -/
def rotation_per_movement (p : RegularPolygon) : ℚ :=
  360 - (inner_angle p + 90)

/-- Theorem: After a full rotation around an octagon, an object on a square returns to its original position -/
theorem square_rotation_octagon (s : Square) (initial_pos : Position) :
  let octagon : RegularPolygon := ⟨8, by norm_num⟩
  let total_rotation : ℚ := 8 * rotation_per_movement octagon
  total_rotation % 360 = 0 → initial_pos = Position.Bottom → initial_pos = Position.Bottom :=
by
  sorry


end square_rotation_octagon_l69_6925


namespace circle_and_line_problem_l69_6991

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + (y + 2)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the line m
def line_m (x y : ℝ) : Prop := x = 1 ∨ (5/12)*x - y + 43/12 = 0

-- Theorem statement
theorem circle_and_line_problem :
  -- Given conditions
  (circle_C 0 2) ∧ 
  (circle_C 2 (-2)) ∧ 
  (∃ (x y : ℝ), circle_C x y ∧ line_l x y) ∧
  (line_m 1 4) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    circle_C x₁ y₁ ∧ 
    circle_C x₂ y₂ ∧ 
    line_m x₁ y₁ ∧ 
    line_m x₂ y₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) →
  -- Conclusion
  (∀ (x y : ℝ), circle_C x y ↔ (x + 3)^2 + (y + 2)^2 = 25) ∧
  (∀ (x y : ℝ), line_m x y ↔ (x = 1 ∨ (5/12)*x - y + 43/12 = 0)) :=
by
  sorry

end circle_and_line_problem_l69_6991


namespace equation_roots_exist_l69_6906

/-- Proves that the equation x|x| + px + q = 0 can have real roots even when p^2 - 4q < 0 -/
theorem equation_roots_exist (p q : ℝ) (h : p^2 - 4*q < 0) : 
  ∃ x : ℝ, x * |x| + p * x + q = 0 := by
  sorry

end equation_roots_exist_l69_6906


namespace evaluate_expression_l69_6963

theorem evaluate_expression : 15 * 30 + 45 * 15 - 15 * 10 = 975 := by
  sorry

end evaluate_expression_l69_6963


namespace quadratic_inequality_l69_6905

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- Theorem: If f(-3) = f(1) for a quadratic function f(x) = x^2 + bx + c, 
    then f(1) > c > f(-1) -/
theorem quadratic_inequality (b c : ℝ) : 
  f b c (-3) = f b c 1 → f b c 1 > c ∧ c > f b c (-1) := by
  sorry

end quadratic_inequality_l69_6905


namespace francie_remaining_money_l69_6999

def initial_allowance : ℕ := 5
def initial_weeks : ℕ := 8
def raised_allowance : ℕ := 6
def raised_weeks : ℕ := 6
def cash_gift : ℕ := 20
def investment_amount : ℕ := 10
def investment_return_rate : ℚ := 5 / 100
def video_game_cost : ℕ := 35

def total_savings : ℚ :=
  (initial_allowance * initial_weeks +
   raised_allowance * raised_weeks +
   cash_gift : ℚ)

def total_with_investment : ℚ :=
  total_savings + investment_amount * investment_return_rate

def remaining_after_clothes : ℚ :=
  total_with_investment / 2

theorem francie_remaining_money :
  remaining_after_clothes - video_game_cost = 13.25 := by
  sorry

end francie_remaining_money_l69_6999


namespace second_difference_quadratic_constant_second_difference_implies_A_second_difference_one_implies_A_half_second_difference_seven_implies_A_seven_half_l69_6955

/-- Second difference of a function f at point n -/
def secondDifference (f : ℕ → ℚ) (n : ℕ) : ℚ :=
  f (n + 2) - 2 * f (n + 1) + f n

/-- Quadratic function with rational coefficients -/
def quadraticFunction (A B C : ℚ) (n : ℕ) : ℚ :=
  A * n^2 + B * n + C

theorem second_difference_quadratic (A B C : ℚ) :
  ∀ n : ℕ, secondDifference (quadraticFunction A B C) n = 2 * A :=
sorry

theorem constant_second_difference_implies_A (A B C k : ℚ) :
  (∀ n : ℕ, secondDifference (quadraticFunction A B C) n = k) → A = k / 2 :=
sorry

theorem second_difference_one_implies_A_half :
  ∀ A B C : ℚ,
  (∀ n : ℕ, secondDifference (quadraticFunction A B C) n = 1) →
  A = 1 / 2 :=
sorry

theorem second_difference_seven_implies_A_seven_half :
  ∀ A B C : ℚ,
  (∀ n : ℕ, secondDifference (quadraticFunction A B C) n = 7) →
  A = 7 / 2 :=
sorry

end second_difference_quadratic_constant_second_difference_implies_A_second_difference_one_implies_A_half_second_difference_seven_implies_A_seven_half_l69_6955


namespace chemical_representations_correct_l69_6984

/-- Represents a chemical element -/
inductive Element : Type
| C : Element
| H : Element
| O : Element
| N : Element
| Si : Element
| P : Element

/-- Represents a chemical formula -/
structure ChemicalFormula :=
  (elements : List (Element × ℕ))

/-- Represents a structural formula -/
structure StructuralFormula :=
  (formula : String)

/-- Definition of starch chemical formula -/
def starchFormula : ChemicalFormula :=
  ⟨[(Element.C, 6), (Element.H, 10), (Element.O, 5)]⟩

/-- Definition of glycine structural formula -/
def glycineFormula : StructuralFormula :=
  ⟨"H₂N-CH₂-COOH"⟩

/-- Definition of silicate-containing materials -/
def silicateProducts : List String :=
  ["glass", "ceramics", "cement"]

/-- Definition of red tide causing elements -/
def redTideElements : List Element :=
  [Element.N, Element.P]

/-- Theorem stating the correctness of the chemical representations -/
theorem chemical_representations_correct :
  (starchFormula.elements = [(Element.C, 6), (Element.H, 10), (Element.O, 5)]) ∧
  (glycineFormula.formula = "H₂N-CH₂-COOH") ∧
  (∀ product ∈ silicateProducts, ∃ e ∈ product.toList, e = 'S') ∧
  (redTideElements = [Element.N, Element.P]) :=
sorry


end chemical_representations_correct_l69_6984


namespace mitchell_antonio_pencil_difference_l69_6998

theorem mitchell_antonio_pencil_difference :
  ∀ (mitchell_pencils antonio_pencils : ℕ),
    mitchell_pencils = 30 →
    mitchell_pencils + antonio_pencils = 54 →
    mitchell_pencils > antonio_pencils →
    mitchell_pencils - antonio_pencils = 6 :=
by
  sorry

end mitchell_antonio_pencil_difference_l69_6998


namespace a_formula_S_formula_min_t_value_l69_6957

-- Define the arithmetic sequence and its sum
def a (n : ℕ) : ℚ := sorry
def S (n : ℕ) : ℚ := sorry

-- Define conditions
axiom S_9 : S 9 = 90
axiom S_15 : S 15 = 240

-- Define b_n and its sum
def b (n : ℕ) : ℚ := 1 / (2 * n * (n + 1))
def S_b (n : ℕ) : ℚ := (1 / 2) * (1 - 1 / (n + 1))

-- Theorem statements
theorem a_formula (n : ℕ) : a n = 2 * n := sorry

theorem S_formula (n : ℕ) : S n = n * (n + 1) := sorry

theorem min_t_value : 
  ∀ t : ℚ, (∀ n : ℕ, n > 0 → S_b n < t) → t ≥ 1/2 := sorry

end a_formula_S_formula_min_t_value_l69_6957


namespace bicycle_sales_cost_price_l69_6930

theorem bicycle_sales_cost_price 
  (profit_A_to_B : Real) 
  (profit_B_to_C : Real) 
  (final_price : Real) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  final_price = 225 →
  ∃ (initial_cost : Real) (profit_C_to_D : Real),
    initial_cost = 150 ∧
    final_price = initial_cost * (1 + profit_A_to_B) * (1 + profit_B_to_C) * (1 + profit_C_to_D) :=
by sorry

end bicycle_sales_cost_price_l69_6930


namespace great_wall_scientific_notation_l69_6988

theorem great_wall_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 6700000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 6.7 ∧ n = 6 := by
  sorry

end great_wall_scientific_notation_l69_6988


namespace nested_f_application_l69_6966

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^3 else x + 9

theorem nested_f_application : f (f (f (f (f 3)))) = 0 := by
  sorry

end nested_f_application_l69_6966


namespace p_tilde_at_two_l69_6924

def p (x : ℝ) : ℝ := x^2 - 4*x + 3

def p_tilde (x : ℝ) : ℝ := p (p x)

theorem p_tilde_at_two : p_tilde 2 = -4 := by
  sorry

end p_tilde_at_two_l69_6924


namespace roots_equation_value_l69_6943

theorem roots_equation_value (α β : ℝ) : 
  α^2 - α - 1 = 0 → β^2 - β - 1 = 0 → α^4 + 3*β = 5 := by
  sorry

end roots_equation_value_l69_6943


namespace solve_equation_for_k_l69_6960

theorem solve_equation_for_k (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k + 1) = x^3 + k * (x^2 - x - 4)) →
  k = -3 :=
by sorry

end solve_equation_for_k_l69_6960


namespace specific_wall_has_30_bricks_l69_6929

/-- Represents a brick wall with a specific pattern -/
structure BrickWall where
  num_rows : ℕ
  bottom_row_bricks : ℕ
  brick_decrease : ℕ

/-- Calculates the total number of bricks in the wall -/
def total_bricks (wall : BrickWall) : ℕ :=
  sorry

/-- Theorem stating that a specific brick wall configuration has 30 bricks in total -/
theorem specific_wall_has_30_bricks :
  let wall : BrickWall := {
    num_rows := 5,
    bottom_row_bricks := 8,
    brick_decrease := 1
  }
  total_bricks wall = 30 := by
  sorry

end specific_wall_has_30_bricks_l69_6929
