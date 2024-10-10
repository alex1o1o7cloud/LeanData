import Mathlib

namespace inscribed_quadrilateral_sides_l2758_275814

-- Define the quadrilateral
structure InscribedQuadrilateral where
  radius : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  perpendicular : Bool

-- Define the theorem
theorem inscribed_quadrilateral_sides
  (q : InscribedQuadrilateral)
  (h1 : q.radius = 10)
  (h2 : q.diagonal1 = 12)
  (h3 : q.diagonal2 = 10 * Real.sqrt 3)
  (h4 : q.perpendicular = true) :
  ∃ (s1 s2 s3 s4 : ℝ),
    (s1 = 4 * Real.sqrt 15 + 2 * Real.sqrt 5 ∧
     s2 = 4 * Real.sqrt 15 - 2 * Real.sqrt 5 ∧
     s3 = 4 * Real.sqrt 5 + 2 * Real.sqrt 15 ∧
     s4 = 4 * Real.sqrt 5 - 2 * Real.sqrt 15) ∨
    (s1 = 4 * Real.sqrt 15 + 2 * Real.sqrt 5 ∧
     s2 = 4 * Real.sqrt 15 - 2 * Real.sqrt 5 ∧
     s3 = 4 * Real.sqrt 5 - 2 * Real.sqrt 15 ∧
     s4 = 4 * Real.sqrt 5 + 2 * Real.sqrt 15) :=
by sorry


end inscribed_quadrilateral_sides_l2758_275814


namespace race_result_l2758_275879

/-- Represents a runner in the race -/
structure Runner where
  position : ℝ
  speed : ℝ

/-- The race setup and result -/
theorem race_result 
  (race_length : ℝ) 
  (a b : Runner) 
  (h1 : race_length = 3000)
  (h2 : a.position = race_length - 500)
  (h3 : b.position = race_length - 600)
  (h4 : a.speed > 0)
  (h5 : b.speed > 0) :
  let time_to_finish_a := (race_length - a.position) / a.speed
  let b_final_position := b.position + b.speed * time_to_finish_a
  race_length - b_final_position = 120 := by
sorry

end race_result_l2758_275879


namespace trapezoid_EFGH_area_l2758_275873

/-- Trapezoid with vertices E(0,0), F(0,3), G(5,0), and H(5,7) -/
structure Trapezoid where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

/-- The area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ := sorry

/-- The theorem stating the area of the specific trapezoid EFGH -/
theorem trapezoid_EFGH_area :
  let t : Trapezoid := {
    E := (0, 0),
    F := (0, 3),
    G := (5, 0),
    H := (5, 7)
  }
  trapezoidArea t = 25 := by sorry

end trapezoid_EFGH_area_l2758_275873


namespace employment_agency_payroll_l2758_275881

/-- Calculates the total payroll for an employment agency given the number of employees,
    number of laborers, and pay rates for heavy operators and laborers. -/
theorem employment_agency_payroll
  (total_employees : ℕ)
  (num_laborers : ℕ)
  (heavy_operator_pay : ℕ)
  (laborer_pay : ℕ)
  (h1 : total_employees = 31)
  (h2 : num_laborers = 1)
  (h3 : heavy_operator_pay = 129)
  (h4 : laborer_pay = 82) :
  (total_employees - num_laborers) * heavy_operator_pay + num_laborers * laborer_pay = 3952 :=
by
  sorry


end employment_agency_payroll_l2758_275881


namespace picture_difference_is_eight_l2758_275847

/-- The number of pictures Ralph has -/
def ralph_pictures : ℕ := 26

/-- The number of pictures Derrick has -/
def derrick_pictures : ℕ := 34

/-- The difference in the number of pictures between Derrick and Ralph -/
def picture_difference : ℕ := derrick_pictures - ralph_pictures

theorem picture_difference_is_eight : picture_difference = 8 := by
  sorry

end picture_difference_is_eight_l2758_275847


namespace unique_positive_integer_solution_l2758_275888

theorem unique_positive_integer_solution : 
  ∃! (x : ℕ), x > 0 ∧ 12 * x = x^2 + 36 :=
by
  -- Proof goes here
  sorry

end unique_positive_integer_solution_l2758_275888


namespace items_per_crate_l2758_275868

theorem items_per_crate (novels : ℕ) (comics : ℕ) (documentaries : ℕ) (albums : ℕ) (crates : ℕ) :
  novels = 145 →
  comics = 271 →
  documentaries = 419 →
  albums = 209 →
  crates = 116 →
  (novels + comics + documentaries + albums) / crates = 9 := by
sorry

end items_per_crate_l2758_275868


namespace inscribe_two_equal_circles_l2758_275824

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle is tangent to a line segment --/
def isTangentToSide (c : Circle) (p1 p2 : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if two circles are tangent to each other --/
def areTangent (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a circle is inside a triangle --/
def isInside (c : Circle) (t : Triangle) : Prop := sorry

/-- Theorem stating that two equal circles can be inscribed in any triangle,
    each touching two sides of the triangle and the other circle --/
theorem inscribe_two_equal_circles (t : Triangle) : 
  ∃ (c1 c2 : Circle), 
    c1.radius = c2.radius ∧ 
    isInside c1 t ∧ 
    isInside c2 t ∧ 
    (isTangentToSide c1 t.A t.B ∨ isTangentToSide c1 t.B t.C ∨ isTangentToSide c1 t.C t.A) ∧
    (isTangentToSide c1 t.A t.B ∨ isTangentToSide c1 t.B t.C ∨ isTangentToSide c1 t.C t.A) ∧
    (isTangentToSide c2 t.A t.B ∨ isTangentToSide c2 t.B t.C ∨ isTangentToSide c2 t.C t.A) ∧
    (isTangentToSide c2 t.A t.B ∨ isTangentToSide c2 t.B t.C ∨ isTangentToSide c2 t.C t.A) ∧
    areTangent c1 c2 := by
  sorry

end inscribe_two_equal_circles_l2758_275824


namespace perpendicular_line_equation_l2758_275807

/-- The slope of the given line y = 2x -/
def slope_given : ℚ := 2

/-- The point through which the perpendicular line passes -/
def point : ℚ × ℚ := (1, 1)

/-- The equation of the line to be proved -/
def line_equation (x y : ℚ) : Prop := x + 2 * y - 3 = 0

/-- Theorem stating that the line equation represents the perpendicular line -/
theorem perpendicular_line_equation :
  (∀ x y, line_equation x y ↔ y - point.2 = (-1 / slope_given) * (x - point.1)) ∧
  line_equation point.1 point.2 := by
  sorry

end perpendicular_line_equation_l2758_275807


namespace irrational_numbers_have_square_roots_l2758_275846

theorem irrational_numbers_have_square_roots : ∃ (x : ℝ), Irrational x ∧ ∃ (y : ℝ), y^2 = x := by
  sorry

end irrational_numbers_have_square_roots_l2758_275846


namespace league_games_l2758_275882

theorem league_games (num_teams : ℕ) (total_games : ℕ) (games_per_matchup : ℕ) : 
  num_teams = 50 → 
  total_games = 4900 → 
  games_per_matchup * (num_teams - 1) * num_teams = 2 * total_games → 
  games_per_matchup = 2 :=
by
  sorry

end league_games_l2758_275882


namespace geometric_sequence_inequality_l2758_275805

/-- A geometric sequence with positive terms and common ratio greater than 1 -/
structure GeometricSequence where
  b : ℕ → ℝ
  positive : ∀ n, b n > 0
  q : ℝ
  q_gt_one : q > 1
  geometric : ∀ n, b (n + 1) = q * b n

/-- The inequality holds for the 4th, 5th, 7th, and 8th terms of the geometric sequence -/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  seq.b 4 * seq.b 8 > seq.b 5 * seq.b 7 := by sorry

end geometric_sequence_inequality_l2758_275805


namespace calculation_result_l2758_275886

theorem calculation_result : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end calculation_result_l2758_275886


namespace complex_function_inequality_l2758_275845

/-- Given a ∈ (0,1) and f(z) = z^2 - z + a for z ∈ ℂ,
    for any z ∈ ℂ with |z| ≥ 1, there exists z₀ ∈ ℂ with |z₀| = 1
    such that |f(z₀)| ≤ |f(z)| -/
theorem complex_function_inequality (a : ℝ) (ha : 0 < a ∧ a < 1) :
  let f : ℂ → ℂ := fun z ↦ z^2 - z + a
  ∀ z : ℂ, Complex.abs z ≥ 1 →
    ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ Complex.abs (f z₀) ≤ Complex.abs (f z) :=
by
  sorry

end complex_function_inequality_l2758_275845


namespace chord_midpoint_trajectory_midpoint_PQ_trajectory_l2758_275890

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 12*y + 24 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 5)

-- Theorem for the trajectory of chord midpoints
theorem chord_midpoint_trajectory (x y : ℝ) : 
  (∃ (a b : ℝ), circle_C a b ∧ circle_C (2*x - a) (2*y - b) ∧ (x + a = 0 ∨ y + b = 5)) →
  x^2 + y^2 + 2*x - 11*y + 30 = 0 :=
sorry

-- Theorem for the trajectory of midpoint M of PQ
theorem midpoint_PQ_trajectory (x y : ℝ) :
  (∃ (q_x q_y : ℝ), circle_C q_x q_y ∧ x = (q_x + point_P.1) / 2 ∧ y = (q_y + point_P.2) / 2) →
  x^2 + y^2 + 2*x - 11*y - 11/4 = 0 :=
sorry

end chord_midpoint_trajectory_midpoint_PQ_trajectory_l2758_275890


namespace trapezoid_side_length_l2758_275887

/-- Represents a trapezoid ABCD with sides AB and CD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem stating the relationship between the sides of the trapezoid
    given the area ratio of triangles ABC and ADC -/
theorem trapezoid_side_length (ABCD : Trapezoid)
    (h1 : (ABCD.AB / ABCD.CD) = (7 : ℝ) / 3)
    (h2 : ABCD.AB + ABCD.CD = 210) :
    ABCD.AB = 147 := by
  sorry

end trapezoid_side_length_l2758_275887


namespace tourist_money_theorem_l2758_275896

/-- Represents the amount of money a tourist has at the end of each day -/
def money_after_day (initial_money : ℚ) (day : ℕ) : ℚ :=
  match day with
  | 0 => initial_money
  | n + 1 => (money_after_day initial_money n) / 2 - 100

/-- Theorem stating that if a tourist spends half their money plus 100 Ft each day for 5 days
    and ends up with no money, they must have started with 6200 Ft -/
theorem tourist_money_theorem :
  ∃ (initial_money : ℚ), 
    (money_after_day initial_money 5 = 0) ∧ 
    (initial_money = 6200) :=
by sorry

end tourist_money_theorem_l2758_275896


namespace disjunction_truth_l2758_275893

theorem disjunction_truth (p q : Prop) : (p ∨ q) → (p ∨ q) :=
  sorry

end disjunction_truth_l2758_275893


namespace imaginary_part_of_z_l2758_275851

theorem imaginary_part_of_z (z : ℂ) (h : 1 + 2*I = I * z) : z.im = -1 := by
  sorry

end imaginary_part_of_z_l2758_275851


namespace divisibility_by_24_l2758_275829

theorem divisibility_by_24 (n : ℕ) (h_odd : Odd n) (h_not_div_3 : ¬3 ∣ n) :
  24 ∣ (n^2 - 1) := by
  sorry

end divisibility_by_24_l2758_275829


namespace largest_integer_in_inequality_l2758_275821

theorem largest_integer_in_inequality : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 7/11 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/11) → y ≤ x :=
by
  use 4
  sorry

end largest_integer_in_inequality_l2758_275821


namespace triangle_property_l2758_275816

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) →
  -- Given condition
  (a - 2*c) * (Real.cos B) + b * (Real.cos A) = 0 →
  -- Given value for sin A
  Real.sin A = 3 * (Real.sqrt 10) / 10 →
  -- Prove these
  Real.cos B = 1/3 ∧ b/c = Real.sqrt 7 := by
  sorry

end triangle_property_l2758_275816


namespace vector_perpendicular_l2758_275849

/-- Given vectors a and b, prove that a - b is perpendicular to b -/
theorem vector_perpendicular (a b : ℝ × ℝ) (h1 : a = (1, 0)) (h2 : b = (1/2, 1/2)) :
  (a - b) • b = 0 := by
  sorry

end vector_perpendicular_l2758_275849


namespace stock_worth_equation_l2758_275835

/-- Proves that the total worth of stock satisfies the given equation based on the problem conditions --/
theorem stock_worth_equation (W : ℝ) 
  (h1 : 0.25 * W * 0.15 - 0.40 * W * 0.05 + 0.35 * W * 0.10 = 750) : 
  0.0525 * W = 750 := by
  sorry

end stock_worth_equation_l2758_275835


namespace children_doing_both_A_and_B_l2758_275828

/-- The number of children who can do both A and B -/
def X : ℕ := 19

/-- The total number of children -/
def total : ℕ := 48

/-- The number of children who can do A -/
def A : ℕ := 38

/-- The number of children who can do B -/
def B : ℕ := 29

theorem children_doing_both_A_and_B :
  X = A + B - total :=
by sorry

end children_doing_both_A_and_B_l2758_275828


namespace water_added_to_container_l2758_275864

theorem water_added_to_container (capacity : ℝ) (initial_percentage : ℝ) (final_fraction : ℝ) :
  capacity = 120 →
  initial_percentage = 0.35 →
  final_fraction = 3/4 →
  (final_fraction * capacity) - (initial_percentage * capacity) = 48 :=
by sorry

end water_added_to_container_l2758_275864


namespace tangent_line_minimum_value_l2758_275863

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

/-- Point A -/
def A : ℝ × ℝ := (2, 2)

/-- The line on which point A lies -/
def line (m n l : ℝ) (x y : ℝ) : Prop := m*x + n*y = l

theorem tangent_line_minimum_value (m n l : ℝ) (hm : m > 0) (hn : n > 0) :
  line m n l A.1 A.2 →
  f' A.1 = 4 →
  ∀ k₁ k₂ : ℝ, k₁ > 0 → k₂ > 0 → line k₁ k₂ l A.1 A.2 → 
  1/k₁ + 2/k₂ ≥ 6 + 4*Real.sqrt 2 :=
sorry

end tangent_line_minimum_value_l2758_275863


namespace prime_divisor_congruent_to_one_l2758_275822

theorem prime_divisor_congruent_to_one (p : ℕ) (hp : Prime p) :
  ∃ q : ℕ, Prime q ∧ q ∣ (p^p - 1) ∧ q % p = 1 := by
  sorry

end prime_divisor_congruent_to_one_l2758_275822


namespace triangle_theorem_l2758_275839

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.c * Real.sin (t.A - t.B) = t.b * Real.sin (t.C - t.A)) :
  (t.a ^ 2 = t.b * t.c → t.A = π / 3) ∧
  (t.a = 2 ∧ Real.cos t.A = 4 / 5 → t.a + t.b + t.c = 2 + Real.sqrt 13) :=
by sorry

end triangle_theorem_l2758_275839


namespace jackie_free_time_l2758_275856

/-- Calculates the free time given the time spent on various activities and the total time available. -/
def free_time (work_hours exercise_hours sleep_hours total_hours : ℕ) : ℕ :=
  total_hours - (work_hours + exercise_hours + sleep_hours)

/-- Proves that Jackie has 5 hours of free time given her daily schedule. -/
theorem jackie_free_time :
  let work_hours : ℕ := 8
  let exercise_hours : ℕ := 3
  let sleep_hours : ℕ := 8
  let total_hours : ℕ := 24
  free_time work_hours exercise_hours sleep_hours total_hours = 5 := by
  sorry

end jackie_free_time_l2758_275856


namespace equation_proof_l2758_275842

theorem equation_proof : 529 + 2 * 23 * 3 + 9 = 676 := by
  sorry

end equation_proof_l2758_275842


namespace point_outside_region_implies_a_range_l2758_275889

theorem point_outside_region_implies_a_range (a : ℝ) : 
  (2 - (4 * a^2 + 3 * a - 2) * 2 - 4 ≥ 0) → 
  (a ∈ Set.Icc (-1 : ℝ) (1/4 : ℝ)) := by
  sorry

end point_outside_region_implies_a_range_l2758_275889


namespace chocolate_gain_percent_l2758_275844

/-- Given that the cost price of 121 chocolates equals the selling price of 77 chocolates,
    the gain percent is (4400 / 77)%. -/
theorem chocolate_gain_percent :
  ∀ (cost_price selling_price : ℝ),
  cost_price > 0 →
  selling_price > 0 →
  121 * cost_price = 77 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = 4400 / 77 := by
sorry

end chocolate_gain_percent_l2758_275844


namespace equation_satisfied_at_nine_l2758_275858

/-- The sum of an infinite geometric series with first term a and common ratio r. -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Left-hand side of the equation -/
noncomputable def leftHandSide : ℝ := 
  (geometricSum 1 (1/3)) * (geometricSum 1 (-1/3))

/-- Right-hand side of the equation -/
noncomputable def rightHandSide (y : ℝ) : ℝ := 
  geometricSum 1 (1/y)

/-- The theorem stating that the equation is satisfied when y = 9 -/
theorem equation_satisfied_at_nine : 
  leftHandSide = rightHandSide 9 := by sorry

end equation_satisfied_at_nine_l2758_275858


namespace students_in_both_band_and_chorus_l2758_275866

/-- Calculates the number of students in both band and chorus -/
def students_in_both (total : ℕ) (band : ℕ) (chorus : ℕ) (band_or_chorus : ℕ) : ℕ :=
  band + chorus - band_or_chorus

/-- Proves that the number of students in both band and chorus is 50 -/
theorem students_in_both_band_and_chorus :
  students_in_both 300 120 180 250 = 50 := by
  sorry

end students_in_both_band_and_chorus_l2758_275866


namespace page_added_thrice_l2758_275802

/-- Given a book with pages numbered from 2 to n, if one page number p
    is added three times instead of once, resulting in a total sum of 4090,
    then p = 43. -/
theorem page_added_thrice (n : ℕ) (p : ℕ) (h1 : n ≥ 2) 
    (h2 : n * (n + 1) / 2 - 1 + 2 * p = 4090) : p = 43 := by
  sorry

end page_added_thrice_l2758_275802


namespace problem_solution_l2758_275841

theorem problem_solution : ∃ y : ℕ, (8000 * 6000 : ℕ) = 480 * (10 ^ y) ∧ y = 5 := by
  sorry

end problem_solution_l2758_275841


namespace new_person_weight_l2758_275815

/-- Proves that the weight of a new person is 85 kg given the conditions of the problem -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 65 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 85 :=
by sorry

end new_person_weight_l2758_275815


namespace fraction_sum_equals_two_thirds_l2758_275877

theorem fraction_sum_equals_two_thirds : 
  2 / 10 + 4 / 40 + 6 / 60 + 8 / 30 = 2 / 3 := by
  sorry

end fraction_sum_equals_two_thirds_l2758_275877


namespace unique_solution_l2758_275878

theorem unique_solution (x y z : ℝ) 
  (hx : x > 2) (hy : y > 2) (hz : z > 2)
  (heq : ((x + 3)^2) / (y + z - 3) + ((y + 5)^2) / (z + x - 5) + ((z + 7)^2) / (x + y - 7) = 45) :
  x = 7 ∧ y = 5 ∧ z = 3 := by
sorry

end unique_solution_l2758_275878


namespace tan_45_degrees_l2758_275883

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l2758_275883


namespace vertex_when_m_3_n_values_max_3_m_range_two_points_l2758_275865

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + m - 1

-- Theorem 1: Vertex when m = 3
theorem vertex_when_m_3 :
  let m := 3
  ∃ (x y : ℝ), x = 2 ∧ y = 6 ∧ 
    ∀ (t : ℝ), f m t ≤ f m x :=
sorry

-- Theorem 2: Values of n when maximum is 3
theorem n_values_max_3 :
  let m := 3
  ∀ (n : ℝ), (∀ (x : ℝ), n ≤ x ∧ x ≤ n + 2 → f m x ≤ 3) ∧
             (∃ (x : ℝ), n ≤ x ∧ x ≤ n + 2 ∧ f m x = 3) →
    n = 2 + Real.sqrt 3 ∨ n = -Real.sqrt 3 :=
sorry

-- Theorem 3: Range of m for exactly two points 3 units from x-axis
theorem m_range_two_points :
  ∀ (m : ℝ), (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (f m x₁ = 3 ∨ f m x₁ = -3) ∧ (f m x₂ = 3 ∨ f m x₂ = -3)) ↔
    -6 < m ∧ m < 0 :=
sorry

end vertex_when_m_3_n_values_max_3_m_range_two_points_l2758_275865


namespace solve_equation_l2758_275838

theorem solve_equation (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 7 → y = 29 / 3 := by
  sorry

end solve_equation_l2758_275838


namespace complex_equation_unit_modulus_l2758_275834

theorem complex_equation_unit_modulus (z : ℂ) (h : 11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0) : Complex.abs z = 1 := by
  sorry

end complex_equation_unit_modulus_l2758_275834


namespace correct_calculation_l2758_275859

theorem correct_calculation (x : ℤ) (h : x + 44 - 39 = 63) : (x + 39) - 44 = 53 := by
  sorry

end correct_calculation_l2758_275859


namespace exists_positive_c_less_than_sum_l2758_275806

theorem exists_positive_c_less_than_sum (a b : ℝ) (h : a < b) :
  ∃ c : ℝ, c > 0 ∧ a < b + c := by
sorry

end exists_positive_c_less_than_sum_l2758_275806


namespace negation_of_conditional_l2758_275895

theorem negation_of_conditional (x y : ℝ) :
  ¬(((x - 1) * (y + 2) = 0) → (x = 1 ∨ y = -2)) ↔
  (((x - 1) * (y + 2) ≠ 0) → (x ≠ 1 ∧ y ≠ -2)) :=
by sorry

end negation_of_conditional_l2758_275895


namespace triangle_problem_l2758_275825

/-- Triangle ABC with angles A, B, C opposite sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions and theorem -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.A = π / 6)
  (h2 : (1 + Real.sqrt 3) * t.c = 2 * t.b) :
  t.C = π / 4 ∧ 
  (t.b * t.a * Real.cos t.C = 1 + Real.sqrt 3 → 
    t.a = Real.sqrt 2 ∧ t.b = 1 + Real.sqrt 3 ∧ t.c = 2) := by
  sorry

end triangle_problem_l2758_275825


namespace mixture_weight_theorem_l2758_275832

/-- Atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Phosphorus in g/mol -/
def P_weight : ℝ := 30.97

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Sodium in g/mol -/
def Na_weight : ℝ := 22.99

/-- Atomic weight of Sulfur in g/mol -/
def S_weight : ℝ := 32.07

/-- Molecular weight of Aluminum phosphate (AlPO4) in g/mol -/
def AlPO4_weight : ℝ := Al_weight + P_weight + 4 * O_weight

/-- Molecular weight of Sodium sulfate (Na2SO4) in g/mol -/
def Na2SO4_weight : ℝ := 2 * Na_weight + S_weight + 4 * O_weight

/-- Total weight of the mixture in grams -/
def total_mixture_weight : ℝ := 5 * AlPO4_weight + 3 * Na2SO4_weight

theorem mixture_weight_theorem :
  total_mixture_weight = 1035.90 := by sorry

end mixture_weight_theorem_l2758_275832


namespace prime_cube_equation_solutions_l2758_275899

theorem prime_cube_equation_solutions :
  ∀ m n p : ℕ+,
    Nat.Prime p.val →
    (m.val^3 + n.val) * (n.val^3 + m.val) = p.val^3 →
    ((m = 2 ∧ n = 1 ∧ p = 3) ∨ (m = 1 ∧ n = 2 ∧ p = 3)) :=
by sorry

end prime_cube_equation_solutions_l2758_275899


namespace parallelogram_bisector_intersection_inside_l2758_275833

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : sorry

/-- An angle bisector of a parallelogram is a line that bisects one of its angles. -/
def angle_bisector (p : Parallelogram) (i : Fin 4) : Set (ℝ × ℝ) := sorry

/-- The pairwise intersections of angle bisectors of a parallelogram. -/
def bisector_intersections (p : Parallelogram) : Set (ℝ × ℝ) := sorry

/-- A point is inside a parallelogram if it's in the interior of the parallelogram. -/
def inside_parallelogram (p : Parallelogram) (point : ℝ × ℝ) : Prop := sorry

theorem parallelogram_bisector_intersection_inside 
  (p : Parallelogram) : 
  ∃ (point : ℝ × ℝ), point ∈ bisector_intersections p ∧ inside_parallelogram p point :=
sorry

end parallelogram_bisector_intersection_inside_l2758_275833


namespace consecutive_odd_divisibility_l2758_275867

theorem consecutive_odd_divisibility (m n : ℤ) : 
  (∃ k : ℤ, m = 2*k + 1 ∧ n = 2*k + 3) → 
  (∃ l : ℤ, 7*m^2 - 5*n^2 - 2 = 8*l) :=
sorry

end consecutive_odd_divisibility_l2758_275867


namespace bella_apple_consumption_l2758_275848

/-- The fraction of apples Bella consumes from what Grace picks -/
def bella_fraction : ℚ := 1 / 18

/-- The number of apples Bella eats per day -/
def bella_daily_apples : ℕ := 6

/-- The number of apples Grace has left after 6 weeks -/
def grace_remaining_apples : ℕ := 504

/-- The number of weeks in the problem -/
def weeks : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem bella_apple_consumption :
  bella_fraction = 1 / 18 :=
sorry

end bella_apple_consumption_l2758_275848


namespace root_in_interval_l2758_275853

def f (x : ℝ) := x^3 + x - 8

theorem root_in_interval :
  f 1 < 0 →
  f 1.5 < 0 →
  f 1.75 < 0 →
  f 2 > 0 →
  ∃ x, x ∈ Set.Ioo 1.75 2 ∧ f x = 0 :=
by sorry

end root_in_interval_l2758_275853


namespace sqrt_inequality_l2758_275843

theorem sqrt_inequality : Real.sqrt 7 - 1 > Real.sqrt 11 - Real.sqrt 5 := by
  sorry

end sqrt_inequality_l2758_275843


namespace friendly_point_properties_l2758_275894

def is_friendly_point (x y : ℝ) : Prop :=
  ∃ m n : ℝ, m - n = 6 ∧ m - 1 = x ∧ 3*n + 1 = y

theorem friendly_point_properties :
  (¬ is_friendly_point 7 1) ∧ 
  (is_friendly_point 6 4) ∧
  (∀ x y t : ℝ, x + y = 2 → 2*x - y = t → is_friendly_point x y → t = 10) := by
  sorry

end friendly_point_properties_l2758_275894


namespace average_marks_proof_l2758_275876

def scores : List ℕ := [76, 65, 82, 67, 55, 89, 74, 63, 78, 71]

theorem average_marks_proof :
  (scores.sum / scores.length : ℚ) = 72 := by
  sorry

end average_marks_proof_l2758_275876


namespace calc_difference_l2758_275803

-- Define the correct calculation (Mark's method)
def correct_calc : ℤ := 12 - (3 + 6)

-- Define the incorrect calculation (Jane's method)
def incorrect_calc : ℤ := 12 - 3 + 6

-- Theorem statement
theorem calc_difference : correct_calc - incorrect_calc = -12 := by
  sorry

end calc_difference_l2758_275803


namespace cone_surface_area_l2758_275811

theorem cone_surface_area (θ : Real) (S_lateral : Real) (S_total : Real) : 
  θ = 2 * Real.pi / 3 →  -- 120° in radians
  S_lateral = 3 * Real.pi →
  S_total = 4 * Real.pi :=
by sorry

end cone_surface_area_l2758_275811


namespace chess_tournament_green_teams_l2758_275817

theorem chess_tournament_green_teams (red_players green_players total_players total_teams red_red_teams : ℕ)
  (h1 : red_players = 64)
  (h2 : green_players = 68)
  (h3 : total_players = red_players + green_players)
  (h4 : total_teams = 66)
  (h5 : total_players = 2 * total_teams)
  (h6 : red_red_teams = 20) :
  ∃ green_green_teams : ℕ, green_green_teams = 22 ∧ 
  green_green_teams = total_teams - red_red_teams - (red_players - 2 * red_red_teams) := by
  sorry

#check chess_tournament_green_teams

end chess_tournament_green_teams_l2758_275817


namespace paint_calculation_l2758_275861

theorem paint_calculation (P : ℚ) : 
  (1/6 : ℚ) * P + (1/5 : ℚ) * (P - (1/6 : ℚ) * P) = 120 → P = 360 := by
sorry

end paint_calculation_l2758_275861


namespace salary_restoration_l2758_275870

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) : 
  let reduced_salary := original_salary * (1 - 0.2)
  reduced_salary * (1 + 0.25) = original_salary := by
sorry

end salary_restoration_l2758_275870


namespace quadratic_inequality_solution_l2758_275857

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + 3 * x + 2 > 0) ↔ (b < x ∧ x < 1)) →
  (a = -5 ∧ b = -2/5) :=
by sorry

end quadratic_inequality_solution_l2758_275857


namespace soft_drink_bottles_l2758_275892

theorem soft_drink_bottles (small_bottles : ℕ) : 
  (10000 : ℕ) * (85 : ℕ) / 100 + small_bottles * (88 : ℕ) / 100 = (13780 : ℕ) →
  small_bottles = (6000 : ℕ) :=
by sorry

end soft_drink_bottles_l2758_275892


namespace simplify_fraction_l2758_275850

theorem simplify_fraction (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 2*b) (h3 : a ≠ b) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end simplify_fraction_l2758_275850


namespace composite_has_at_least_three_divisors_l2758_275898

-- Define what it means for a number to be composite
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ k ∣ n

-- Define the number of divisors function
def NumDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- Theorem statement
theorem composite_has_at_least_three_divisors (n : ℕ) (h : IsComposite n) :
  NumDivisors n ≥ 3 := by
  sorry

end composite_has_at_least_three_divisors_l2758_275898


namespace average_first_10_even_numbers_l2758_275818

theorem average_first_10_even_numbers : 
  let first_10_even : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
  (first_10_even.sum / first_10_even.length : ℚ) = 11 := by
  sorry

end average_first_10_even_numbers_l2758_275818


namespace lawrence_county_kids_at_home_l2758_275880

/-- The number of kids who stay home during the break in Lawrence county -/
def kids_staying_home (total_kids : ℕ) (kids_at_camp : ℕ) : ℕ :=
  total_kids - kids_at_camp

/-- Theorem stating the number of kids staying home during the break in Lawrence county -/
theorem lawrence_county_kids_at_home :
  kids_staying_home 313473 38608 = 274865 := by
  sorry

end lawrence_county_kids_at_home_l2758_275880


namespace two_week_riding_hours_l2758_275875

/-- Represents the number of hours Bethany rides on a given day -/
def daily_riding_hours (day : Nat) : Real :=
  match day % 7 with
  | 1 | 3 | 5 => 1    -- Monday, Wednesday, Friday
  | 2 | 4 => 0.5      -- Tuesday, Thursday
  | 6 => 2            -- Saturday
  | _ => 0            -- Sunday

/-- Calculates the total riding hours over a given number of days -/
def total_riding_hours (days : Nat) : Real :=
  (List.range days).map daily_riding_hours |>.sum

/-- Proves that Bethany rides for 12 hours over a 2-week period -/
theorem two_week_riding_hours :
  total_riding_hours 14 = 12 := by
  sorry

end two_week_riding_hours_l2758_275875


namespace diophantine_equation_prime_divisor_l2758_275813

theorem diophantine_equation_prime_divisor (b : ℕ+) (h : Nat.gcd b.val 6 = 1) :
  (∃ (x y : ℕ+), (1 : ℚ) / x.val + (1 : ℚ) / y.val = (3 : ℚ) / b.val) ↔
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ b.val ∧ ∃ (k : ℕ), p = 6 * k - 1 := by
  sorry

end diophantine_equation_prime_divisor_l2758_275813


namespace chinese_gcd_168_93_l2758_275836

def chinese_gcd (a b : ℕ) : ℕ := sorry

def chinese_gcd_sequence (a b : ℕ) : List (ℕ × ℕ) := sorry

theorem chinese_gcd_168_93 :
  let seq := chinese_gcd_sequence 168 93
  (57, 18) ∈ seq ∧
  (3, 18) ∈ seq ∧
  (3, 3) ∈ seq ∧
  (6, 9) ∉ seq ∧
  chinese_gcd 168 93 = 3 := by sorry

end chinese_gcd_168_93_l2758_275836


namespace floor_ceil_sum_l2758_275809

theorem floor_ceil_sum : ⌊(1.002 : ℝ)⌋ + ⌈(3.998 : ℝ)⌉ + ⌈(-0.999 : ℝ)⌉ = 5 := by
  sorry

end floor_ceil_sum_l2758_275809


namespace raisin_count_proof_l2758_275840

/-- Given 5 boxes of raisins with a total of 437 raisins, where one box has 72 raisins,
    another has 74 raisins, and the remaining three boxes have an equal number of raisins,
    prove that each of these three boxes contains 97 raisins. -/
theorem raisin_count_proof (total_raisins : ℕ) (total_boxes : ℕ) 
  (box1_raisins : ℕ) (box2_raisins : ℕ) (other_boxes_raisins : ℕ) :
  total_raisins = 437 →
  total_boxes = 5 →
  box1_raisins = 72 →
  box2_raisins = 74 →
  total_raisins = box1_raisins + box2_raisins + 3 * other_boxes_raisins →
  other_boxes_raisins = 97 := by
  sorry

end raisin_count_proof_l2758_275840


namespace geometric_sequence_middle_term_l2758_275884

theorem geometric_sequence_middle_term (y : ℝ) :
  (3^2 : ℝ) < y ∧ y < (3^4 : ℝ) ∧ 
  (y / (3^2 : ℝ)) = ((3^4 : ℝ) / y) →
  y = 27 :=
by sorry

end geometric_sequence_middle_term_l2758_275884


namespace hilary_pakora_orders_l2758_275862

/-- Represents the cost of a meal at Delicious Delhi restaurant -/
structure MealCost where
  samosas : ℕ
  pakoras : ℕ
  lassi : ℕ
  tip_percent : ℚ
  total_with_tax : ℚ

/-- Calculates the number of pakora orders given the meal cost details -/
def calculate_pakora_orders (meal : MealCost) : ℚ :=
  let samosa_cost := 2 * meal.samosas
  let lassi_cost := 2 * meal.lassi
  let pakora_cost := 3 * meal.pakoras
  let subtotal := samosa_cost + lassi_cost + pakora_cost
  let total_with_tip := subtotal * (1 + meal.tip_percent)
  (meal.total_with_tax - total_with_tip) / 3

/-- Theorem stating that Hilary bought 4 orders of pakoras -/
theorem hilary_pakora_orders :
  let meal := MealCost.mk 3 4 1 (1/4) 25
  calculate_pakora_orders meal = 4 := by
  sorry

end hilary_pakora_orders_l2758_275862


namespace perpendicular_lines_parallel_l2758_275827

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end perpendicular_lines_parallel_l2758_275827


namespace equation_solution_l2758_275812

theorem equation_solution : ∃! x : ℝ, 5 * x - 3 * x = 210 + 6 * (x + 4) ∧ x = -58.5 := by
  sorry

end equation_solution_l2758_275812


namespace solution_range_l2758_275808

theorem solution_range (x m : ℝ) : 
  (x + m) / 3 - (2 * x - 1) / 2 = m ∧ x ≤ 0 → m ≥ 3/4 := by
  sorry

end solution_range_l2758_275808


namespace grid_with_sequence_exists_l2758_275819

-- Define the grid type
def Grid := Matrix (Fin 6) (Fin 6) (Fin 4)

-- Define a predicate for valid subgrids
def valid_subgrid (g : Grid) (i j : Fin 2) : Prop :=
  ∀ n : Fin 4, ∃! x y : Fin 2, g (2 * i + x) (2 * j + y) = n

-- Define a predicate for adjacent cells being different
def adjacent_different (g : Grid) : Prop :=
  ∀ i j i' j' : Fin 6, 
    (i = i' ∧ |j - j'| = 1) ∨ 
    (j = j' ∧ |i - i'| = 1) ∨ 
    (|i - i'| = 1 ∧ |j - j'| = 1) → 
    g i j ≠ g i' j'

-- Define the existence of the sequence 3521 in the grid
def sequence_exists (g : Grid) : Prop :=
  ∃ i₁ j₁ i₂ j₂ i₃ j₃ i₄ j₄ : Fin 6,
    g i₁ j₁ = 3 ∧ g i₂ j₂ = 5 ∧ g i₃ j₃ = 2 ∧ g i₄ j₄ = 1

-- The main theorem
theorem grid_with_sequence_exists : 
  ∃ g : Grid, 
    (∀ i j : Fin 2, valid_subgrid g i j) ∧ 
    adjacent_different g ∧
    sequence_exists g :=
sorry

end grid_with_sequence_exists_l2758_275819


namespace complement_of_A_l2758_275860

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x - 2 > 0}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≤ 2} := by sorry

end complement_of_A_l2758_275860


namespace tan_negative_210_degrees_l2758_275826

theorem tan_negative_210_degrees : 
  Real.tan (-(210 * π / 180)) = -(Real.sqrt 3 / 3) := by sorry

end tan_negative_210_degrees_l2758_275826


namespace cube_sum_geq_mixed_terms_l2758_275810

theorem cube_sum_geq_mixed_terms (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^3 + y^3 ≥ x^2*y + x*y^2 := by
sorry

end cube_sum_geq_mixed_terms_l2758_275810


namespace intersection_A_complement_B_l2758_275804

def I : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 4, 5}
def B : Set Nat := {1, 4}

theorem intersection_A_complement_B :
  A ∩ (I \ B) = {3, 5} := by sorry

end intersection_A_complement_B_l2758_275804


namespace solve_cottage_problem_l2758_275885

def cottage_problem (hourly_rate : ℚ) (jack_paid : ℚ) (jill_paid : ℚ) : Prop :=
  let total_paid := jack_paid + jill_paid
  let hours_rented := total_paid / hourly_rate
  hours_rented = 8

theorem solve_cottage_problem :
  cottage_problem 5 20 20 := by sorry

end solve_cottage_problem_l2758_275885


namespace triangle_properties_l2758_275852

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) : 
  (t.a * Real.cos t.C + (t.c - 3 * t.b) * Real.cos t.A = 0) → 
  (Real.cos t.A = 1 / 3) ∧
  (Real.sqrt 2 = 1 / 2 * t.b * t.c * Real.sin t.A) →
  (t.b - t.c = 2) →
  (t.a = 2 * Real.sqrt 2) := by
  sorry


end triangle_properties_l2758_275852


namespace percentage_six_years_or_more_l2758_275871

def employee_distribution (x : ℕ) : List ℕ :=
  [4*x, 7*x, 5*x, 4*x, 3*x, 3*x, 2*x, 2*x, 2*x, 2*x]

def total_employees (x : ℕ) : ℕ :=
  List.sum (employee_distribution x)

def employees_six_years_or_more (x : ℕ) : ℕ :=
  List.sum (List.drop 6 (employee_distribution x))

theorem percentage_six_years_or_more (x : ℕ) :
  (employees_six_years_or_more x : ℚ) / (total_employees x : ℚ) * 100 = 2222 / 100 :=
sorry

end percentage_six_years_or_more_l2758_275871


namespace cos_nineteen_pi_fourths_l2758_275837

theorem cos_nineteen_pi_fourths : Real.cos (19 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end cos_nineteen_pi_fourths_l2758_275837


namespace polynomial_roots_unit_circle_l2758_275891

theorem polynomial_roots_unit_circle (a b c : ℂ) :
  (∀ w : ℂ, w^3 + Complex.abs a * w^2 + Complex.abs b * w + Complex.abs c = 0 → Complex.abs w = 1) →
  (Complex.abs c = 1 ∧ 
   ∀ x : ℂ, x^3 + Complex.abs a * x^2 + Complex.abs b * x + Complex.abs c = 0 ↔ 
            x^3 + Complex.abs a * x^2 + Complex.abs a * x + 1 = 0) :=
by sorry

end polynomial_roots_unit_circle_l2758_275891


namespace equiangular_equilateral_parallelogram_is_square_l2758_275830

-- Define a parallelogram
class Parallelogram (P : Type) where
  -- Add any necessary properties of a parallelogram

-- Define the property of being equiangular
class Equiangular (P : Type) where
  -- All angles are equal

-- Define the property of being equilateral
class Equilateral (P : Type) where
  -- All sides have equal length

-- Define a square
class Square (P : Type) extends Parallelogram P where
  -- A square is a parallelogram with additional properties

-- Theorem statement
theorem equiangular_equilateral_parallelogram_is_square 
  (P : Type) [Parallelogram P] [Equiangular P] [Equilateral P] : Square P :=
by sorry

end equiangular_equilateral_parallelogram_is_square_l2758_275830


namespace min_sum_of_indices_l2758_275800

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem min_sum_of_indices (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n : ℕ, a m + a n = 4 * a 1) →
  ∃ m n : ℕ, a m + a n = 4 * a 1 ∧ m + n = 4 ∧ ∀ k l : ℕ, a k + a l = 4 * a 1 → k + l ≥ 4 := by
  sorry

end min_sum_of_indices_l2758_275800


namespace sqrt_three_irrational_neg_one_rational_two_rational_three_rational_l2758_275897

theorem sqrt_three_irrational :
  ∀ (x : ℝ), x ^ 2 = 3 → ¬ (∃ (a b : ℤ), b ≠ 0 ∧ x = a / b) :=
by sorry

theorem neg_one_rational : ∃ (a b : ℤ), b ≠ 0 ∧ -1 = a / b :=
by sorry

theorem two_rational : ∃ (a b : ℤ), b ≠ 0 ∧ 2 = a / b :=
by sorry

theorem three_rational : ∃ (a b : ℤ), b ≠ 0 ∧ 3 = a / b :=
by sorry

end sqrt_three_irrational_neg_one_rational_two_rational_three_rational_l2758_275897


namespace alpha_values_l2758_275874

theorem alpha_values (α : ℂ) 
  (h1 : α ≠ Complex.I ∧ α ≠ -Complex.I)
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : (Complex.abs (α^4 - 1))^2 = 9 * (Complex.abs (α - 1))^2) :
  α = (1/2 : ℂ) + Complex.I * (Real.sqrt 35 / 2) ∨ 
  α = (1/2 : ℂ) - Complex.I * (Real.sqrt 35 / 2) :=
sorry

end alpha_values_l2758_275874


namespace quadratic_roots_range_l2758_275855

/-- A quadratic equation of the form kx^2 - 2x - 1 = 0 has two distinct real roots -/
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x - 1 = 0 ∧ k * y^2 - 2*y - 1 = 0

/-- The range of k for which the quadratic equation has two distinct real roots -/
theorem quadratic_roots_range :
  ∀ k : ℝ, has_two_distinct_real_roots k ↔ k > -1 ∧ k ≠ 0 :=
by sorry

end quadratic_roots_range_l2758_275855


namespace geometric_sequence_third_term_l2758_275869

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem states that in a geometric sequence where the product of the first and fifth terms is 16,
    the third term is either 4 or -4. -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_prod : a 1 * a 5 = 16) :
  a 3 = 4 ∨ a 3 = -4 := by
sorry

end geometric_sequence_third_term_l2758_275869


namespace polynomial_simplification_l2758_275801

theorem polynomial_simplification (x : ℝ) :
  x * (4 * x^3 + 3 * x^2 - 5) - 7 * (x^3 - 4 * x^2 + 2 * x - 6) =
  4 * x^4 - 4 * x^3 + 28 * x^2 - 19 * x + 42 := by
  sorry

end polynomial_simplification_l2758_275801


namespace marble_combination_count_l2758_275831

def num_marbles_per_color : ℕ := 2
def num_colors : ℕ := 4
def total_marbles : ℕ := num_marbles_per_color * num_colors

def choose_two_same_color : ℕ := num_colors * (num_marbles_per_color.choose 2)
def choose_two_diff_colors : ℕ := (num_colors.choose 2) * num_marbles_per_color * num_marbles_per_color

theorem marble_combination_count :
  choose_two_same_color + choose_two_diff_colors = 28 := by
  sorry

end marble_combination_count_l2758_275831


namespace fraction_identity_l2758_275854

theorem fraction_identity (M N a b x : ℝ) (h1 : x ≠ a) (h2 : x ≠ b) (h3 : a ≠ b) :
  (M * x + N) / ((x - a) * (x - b)) = 
  (M * a + N) / (a - b) * (1 / (x - a)) - (M * b + N) / (a - b) * (1 / (x - b)) := by
  sorry

end fraction_identity_l2758_275854


namespace integer_points_count_l2758_275872

/-- Represents a line segment on a number line -/
structure LineSegment where
  start : ℝ
  length : ℝ

/-- Counts the number of integer points covered by a line segment -/
def count_integer_points (segment : LineSegment) : ℕ :=
  sorry

/-- Theorem stating that a line segment of length 2020 covers either 2020 or 2021 integer points -/
theorem integer_points_count (segment : LineSegment) :
  segment.length = 2020 → count_integer_points segment = 2020 ∨ count_integer_points segment = 2021 :=
sorry

end integer_points_count_l2758_275872


namespace smallest_circle_equation_l2758_275820

/-- A parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola y^2 = 4x -/
def Focus : ℝ × ℝ := (1, 0)

/-- A circle with center on the parabola and passing through the focus -/
def CircleOnParabola (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = (center.1 - Focus.1)^2 + (center.2 - Focus.2)^2}

/-- The theorem stating that the circle with smallest radius has equation x^2 + y^2 = 1 -/
theorem smallest_circle_equation :
  ∃ (center : ℝ × ℝ),
    center ∈ Parabola ∧
    Focus ∈ CircleOnParabola center ∧
    (∀ (other_center : ℝ × ℝ),
      other_center ∈ Parabola →
      Focus ∈ CircleOnParabola other_center →
      (center.1 - Focus.1)^2 + (center.2 - Focus.2)^2 ≤ (other_center.1 - Focus.1)^2 + (other_center.2 - Focus.2)^2) →
    CircleOnParabola center = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} :=
sorry

end smallest_circle_equation_l2758_275820


namespace jerry_initial_money_l2758_275823

-- Define Jerry's financial situation
def jerry_spent : ℕ := 6
def jerry_left : ℕ := 12

-- Theorem to prove
theorem jerry_initial_money : 
  jerry_spent + jerry_left = 18 := by sorry

end jerry_initial_money_l2758_275823
