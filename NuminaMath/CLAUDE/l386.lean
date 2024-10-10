import Mathlib

namespace fraction_numerator_proof_l386_38696

theorem fraction_numerator_proof (x : ℚ) : 
  (x / (4 * x + 4) = 3 / 7) → x = -12 / 5 := by
  sorry

end fraction_numerator_proof_l386_38696


namespace polynomial_evaluation_l386_38697

theorem polynomial_evaluation :
  ∀ y : ℝ, y > 0 → y^2 - 3*y - 9 = 0 → y^3 - 3*y^2 - 9*y + 3 = 3 := by
  sorry

end polynomial_evaluation_l386_38697


namespace product_inequality_l386_38617

theorem product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d < b * c := by
sorry

end product_inequality_l386_38617


namespace solution_set_quadratic_inequality_l386_38604

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + 4*x - 5 > 0} = {x : ℝ | x < -5 ∨ x > 1} := by sorry

end solution_set_quadratic_inequality_l386_38604


namespace prism_volume_l386_38689

/-- The volume of a right prism with an equilateral triangular base -/
theorem prism_volume (a : ℝ) (h : ℝ) : 
  a = 5 → -- Side length of the equilateral triangle base
  (a * h * 2 + a^2 * Real.sqrt 3 / 4) = 40 → -- Sum of areas of three adjacent faces
  a * a * Real.sqrt 3 / 4 * h = 625 / 160 * (3 - Real.sqrt 3) :=
by sorry

end prism_volume_l386_38689


namespace sin_cos_cube_difference_l386_38600

theorem sin_cos_cube_difference (α : ℝ) (n : ℝ) (h : Real.sin α - Real.cos α = n) :
  Real.sin α ^ 3 - Real.cos α ^ 3 = (3 * n - n^3) / 2 := by
  sorry

end sin_cos_cube_difference_l386_38600


namespace lemonade_problem_l386_38673

theorem lemonade_problem (lemons_for_60 : ℕ) (gallons : ℕ) (lemon_cost : ℚ) :
  lemons_for_60 = 36 →
  gallons = 15 →
  lemon_cost = 1/2 →
  (lemons_for_60 * gallons) / 60 = 9 ∧
  (lemons_for_60 * gallons) / 60 * lemon_cost = 9/2 := by
  sorry

end lemonade_problem_l386_38673


namespace successful_table_filling_l386_38669

theorem successful_table_filling :
  ∃ (t : Fin 6 → Fin 3 → Bool),
    ∀ (r1 r2 : Fin 6) (c1 c2 : Fin 3),
      r1 ≠ r2 → c1 ≠ c2 →
        (t r1 c1 = t r1 c2 ∧ t r1 c1 = t r2 c1 ∧ t r1 c1 = t r2 c2) = False :=
by sorry

end successful_table_filling_l386_38669


namespace greatest_third_side_l386_38687

theorem greatest_third_side (a b : ℝ) (ha : a = 7) (hb : b = 10) :
  ∃ (c : ℕ), c = 16 ∧ 
  (∀ (x : ℕ), x > c → ¬(a + b > x ∧ b + x > a ∧ x + a > b)) :=
sorry

end greatest_third_side_l386_38687


namespace tomato_picking_second_week_l386_38681

/-- Represents the number of tomatoes picked in each week -/
structure TomatoPicking where
  initial : ℕ
  first_week : ℕ
  second_week : ℕ
  third_week : ℕ
  remaining : ℕ

/-- Checks if the tomato picking satisfies the given conditions -/
def is_valid_picking (p : TomatoPicking) : Prop :=
  p.initial = 100 ∧
  p.first_week = p.initial / 4 ∧
  p.third_week = 2 * p.second_week ∧
  p.remaining = 15 ∧
  p.first_week + p.second_week + p.third_week + p.remaining = p.initial

theorem tomato_picking_second_week :
  ∀ p : TomatoPicking, is_valid_picking p → p.second_week = 20 := by
  sorry

end tomato_picking_second_week_l386_38681


namespace square_of_6y_minus_2_l386_38645

-- Define the condition
def satisfies_equation (y : ℝ) : Prop := 3 * y^2 + 2 = 5 * y + 7

-- State the theorem
theorem square_of_6y_minus_2 (y : ℝ) (h : satisfies_equation y) : (6 * y - 2)^2 = 94 := by
  sorry

end square_of_6y_minus_2_l386_38645


namespace arrangement_two_rows_arrangement_person_not_at_ends_arrangement_girls_together_arrangement_boys_not_adjacent_l386_38685

-- 1
theorem arrangement_two_rows (n : ℕ) (m : ℕ) (h : n + m = 7) :
  (Nat.factorial 7) = 5040 :=
sorry

-- 2
theorem arrangement_person_not_at_ends (n : ℕ) (h : n = 7) :
  5 * (Nat.factorial 6) = 3600 :=
sorry

-- 3
theorem arrangement_girls_together (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 4) :
  (Nat.factorial 4) * (Nat.factorial 4) = 576 :=
sorry

-- 4
theorem arrangement_boys_not_adjacent (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 4) :
  (Nat.factorial 4) * (Nat.factorial 5 / Nat.factorial 2) = 1440 :=
sorry

end arrangement_two_rows_arrangement_person_not_at_ends_arrangement_girls_together_arrangement_boys_not_adjacent_l386_38685


namespace problem_solution_l386_38661

/-- Given that 2x^5 - x^3 + 4x^2 + 3x - 5 + g(x) = 7x^3 - 4x + 2,
    prove that g(x) = -2x^5 + 6x^3 - 4x^2 - x + 7 -/
theorem problem_solution (x : ℝ) :
  let g : ℝ → ℝ := λ x => -2*x^5 + 6*x^3 - 4*x^2 - x + 7
  2*x^5 - x^3 + 4*x^2 + 3*x - 5 + g x = 7*x^3 - 4*x + 2 :=
by sorry

end problem_solution_l386_38661


namespace common_tangent_theorem_l386_38694

/-- The value of 'a' for which the graphs of f(x) = ln(x) and g(x) = x^2 + ax 
    have a common tangent line parallel to y = x -/
def tangent_condition (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), 
    x₁ > 0 ∧ 
    (1 / x₁ = 1) ∧ 
    (2 * x₂ + a = 1) ∧ 
    (x₂^2 + a * x₂ = x₂ - 1)

theorem common_tangent_theorem :
  ∀ a : ℝ, tangent_condition a → (a = 3 ∨ a = -1) :=
sorry

end common_tangent_theorem_l386_38694


namespace coins_missing_l386_38666

theorem coins_missing (x : ℚ) (h : x > 0) : 
  let lost := (2 : ℚ) / 3 * x
  let found := (3 : ℚ) / 4 * lost
  let remaining := x - lost + found
  (x - remaining) / x = 1 / 6 := by
sorry

end coins_missing_l386_38666


namespace imaginary_part_of_complex_number_l386_38690

theorem imaginary_part_of_complex_number : 
  Complex.im ((2 : ℂ) + Complex.I * Complex.I) = 2 := by sorry

end imaginary_part_of_complex_number_l386_38690


namespace line_direction_vector_l386_38629

/-- Given a line passing through two points and a direction vector, prove the scalar value. -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (a : ℝ) :
  p1 = (-3, 2) →
  p2 = (2, -3) →
  (a, -2) = (p2.1 - p1.1, p2.2 - p1.2) →
  a = 2 := by
  sorry

end line_direction_vector_l386_38629


namespace sunglasses_sign_cost_l386_38621

theorem sunglasses_sign_cost (selling_price cost_price : ℕ) (pairs_sold : ℕ) : 
  selling_price = 30 →
  cost_price = 26 →
  pairs_sold = 10 →
  (pairs_sold * (selling_price - cost_price)) / 2 = 20 :=
by sorry

end sunglasses_sign_cost_l386_38621


namespace right_triangle_hypotenuse_l386_38657

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 120 → b = 160 → c^2 = a^2 + b^2 → c = 200 := by sorry

end right_triangle_hypotenuse_l386_38657


namespace f_properties_l386_38632

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then x^2 + 4*x + 3 else Real.log (x - 1) + 1

-- Theorem statement
theorem f_properties :
  (f (Real.exp 1 + 1) = 2) ∧
  (Set.range f = Set.Ici (-1 : ℝ)) :=
by sorry

end f_properties_l386_38632


namespace paper_sheets_calculation_l386_38608

theorem paper_sheets_calculation (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) :
  num_classes = 4 →
  students_per_class = 20 →
  sheets_per_student = 5 →
  num_classes * students_per_class * sheets_per_student = 400 :=
by
  sorry

end paper_sheets_calculation_l386_38608


namespace sum_of_coefficients_l386_38686

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 243 :=
by
  sorry

end sum_of_coefficients_l386_38686


namespace rectangle_to_square_l386_38618

theorem rectangle_to_square (k : ℕ) (h1 : k > 5) :
  (∃ n : ℕ, k * (k - 5) = n^2) → k * (k - 5) = 6^2 :=
by sorry

end rectangle_to_square_l386_38618


namespace condition_one_condition_two_l386_38658

-- Define set A
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- Define set B
def B : Set ℝ := {x | x < -1 ∨ x > 3}

-- Theorem for condition 1
theorem condition_one (a : ℝ) : A a ∩ B = A a → a < -3 ∨ a > 3 := by
  sorry

-- Theorem for condition 2
theorem condition_two (a : ℝ) : (A a ∩ B).Nonempty → a < -1 ∨ a > 1 := by
  sorry

end condition_one_condition_two_l386_38658


namespace largest_number_l386_38668

theorem largest_number : ∀ (a b c d : ℝ), 
  a = -3 → b = 0 → c = Real.sqrt 5 → d = 2 → 
  c ≥ a ∧ c ≥ b ∧ c ≥ d := by
  sorry

end largest_number_l386_38668


namespace sum_of_integers_l386_38682

theorem sum_of_integers (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t ∧
  (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -120 →
  p + q + r + s + t = 22 := by
sorry

end sum_of_integers_l386_38682


namespace divisibility_by_x_squared_minus_one_cubed_l386_38634

theorem divisibility_by_x_squared_minus_one_cubed (n : ℕ) :
  ∃ P : Polynomial ℚ, 
    X^(4*n+2) - (2*n+1) * X^(2*n+2) + (2*n+1) * X^(2*n) - 1 = 
    (X^2 - 1)^3 * P :=
by sorry

end divisibility_by_x_squared_minus_one_cubed_l386_38634


namespace min_ab_in_triangle_l386_38605

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if 2c cos B = 2a + b and the area S = √3 c, then ab ≥ 48. -/
theorem min_ab_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * c * Real.cos B = 2 * a + b →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 * c →
  ab ≥ 48 := by
sorry

end min_ab_in_triangle_l386_38605


namespace meeting_at_64th_lamp_l386_38612

def meet_point (total_intervals : ℕ) (petya_progress : ℕ) (vasya_progress : ℕ) : ℕ :=
  3 * petya_progress + 1

theorem meeting_at_64th_lamp (total_lamps : ℕ) (petya_at : ℕ) (vasya_at : ℕ) 
  (h1 : total_lamps = 100)
  (h2 : petya_at = 22)
  (h3 : vasya_at = 88) :
  meet_point (total_lamps - 1) (petya_at - 1) (total_lamps - vasya_at) = 64 := by
  sorry

end meeting_at_64th_lamp_l386_38612


namespace rectilinear_polygon_odd_area_l386_38692

/-- A rectilinear polygon with integer vertex coordinates and odd side lengths -/
structure RectilinearPolygon where
  vertices : List (Int × Int)
  sides_parallel_to_axes : Bool
  all_sides_odd_length : Bool

/-- The area of a rectilinear polygon -/
noncomputable def area (p : RectilinearPolygon) : ℝ :=
  sorry

/-- A predicate to check if a number is odd -/
def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem rectilinear_polygon_odd_area
  (p : RectilinearPolygon)
  (h_sides : p.vertices.length = 100)
  (h_parallel : p.sides_parallel_to_axes = true)
  (h_odd_sides : p.all_sides_odd_length = true) :
  is_odd (Int.floor (area p)) :=
sorry

end rectilinear_polygon_odd_area_l386_38692


namespace investment_calculation_l386_38651

/-- Given two investors P and Q, where the profit is divided in the ratio 4:6 and P invested 60000, 
    prove that Q invested 90000. -/
theorem investment_calculation (P Q : ℕ) (profit_ratio : Rat) (P_investment : ℕ) : 
  profit_ratio = 4 / 6 →
  P_investment = 60000 →
  Q = 90000 := by
  sorry

end investment_calculation_l386_38651


namespace car_speed_calculation_l386_38698

/-- Represents the speed of a car during a journey -/
structure CarJourney where
  first_speed : ℝ  -- Speed for the first 160 km
  second_speed : ℝ  -- Speed for the next 160 km
  average_speed : ℝ  -- Average speed for the entire 320 km

/-- Theorem stating the speed of the car during the next 160 km -/
theorem car_speed_calculation (journey : CarJourney) 
  (h1 : journey.first_speed = 70)
  (h2 : journey.average_speed = 74.67) : 
  journey.second_speed = 80 := by
  sorry

#check car_speed_calculation

end car_speed_calculation_l386_38698


namespace number_of_employees_l386_38660

def average_salary_without_manager : ℝ := 1200
def average_salary_with_manager : ℝ := 1300
def manager_salary : ℝ := 3300

theorem number_of_employees : 
  ∃ (E : ℕ), 
    (E * average_salary_without_manager + manager_salary) / (E + 1) = average_salary_with_manager ∧ 
    E = 20 := by
  sorry

end number_of_employees_l386_38660


namespace polynomial_ascending_powers_l386_38670

theorem polynomial_ascending_powers (x : ℝ) :
  x^2 - 2 - 5*x^4 + 3*x^3 = -2 + x^2 + 3*x^3 - 5*x^4 :=
by sorry

end polynomial_ascending_powers_l386_38670


namespace line_E_passes_through_points_l386_38614

def point := ℝ × ℝ

-- Define the line equations
def line_A (p : point) : Prop := 3 * p.1 - 2 * p.2 + 1 = 0
def line_B (p : point) : Prop := 4 * p.1 - 5 * p.2 + 13 = 0
def line_C (p : point) : Prop := 5 * p.1 + 2 * p.2 - 17 = 0
def line_D (p : point) : Prop := p.1 + 7 * p.2 - 24 = 0
def line_E (p : point) : Prop := p.1 - 4 * p.2 + 10 = 0

-- Define the given point and the endpoints of the line segment
def given_point : point := (4, 3)
def segment_start : point := (2, 7)
def segment_end : point := (8, -2)

-- Define the trisection points
def trisection_point1 : point := (4, 4)
def trisection_point2 : point := (6, 1)

-- Theorem statement
theorem line_E_passes_through_points :
  (line_E given_point ∨ line_E trisection_point1 ∨ line_E trisection_point2) ∧
  ¬(line_A given_point ∨ line_A trisection_point1 ∨ line_A trisection_point2) ∧
  ¬(line_B given_point ∨ line_B trisection_point1 ∨ line_B trisection_point2) ∧
  ¬(line_C given_point ∨ line_C trisection_point1 ∨ line_C trisection_point2) ∧
  ¬(line_D given_point ∨ line_D trisection_point1 ∨ line_D trisection_point2) :=
by sorry

end line_E_passes_through_points_l386_38614


namespace quadratic_function_property_l386_38627

/-- A quadratic function with specific properties -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, f x = a * x^2 + b * x + c) ∧
    (∀ x, f x ≤ f 3) ∧
    (f 3 = 10) ∧
    (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 4)

/-- Theorem stating that f(5) = 0 for the specified quadratic function -/
theorem quadratic_function_property (f : ℝ → ℝ) (h : quadratic_function f) : f 5 = 0 := by
  sorry

end quadratic_function_property_l386_38627


namespace abs_x_minus_3_minus_sqrt_x_minus_4_squared_l386_38636

theorem abs_x_minus_3_minus_sqrt_x_minus_4_squared (x : ℝ) (h : x < 3) :
  |x - 3 - Real.sqrt ((x - 4)^2)| = 7 - 2*x := by sorry

end abs_x_minus_3_minus_sqrt_x_minus_4_squared_l386_38636


namespace gasoline_tank_problem_l386_38623

/-- Proves properties of a gasoline tank given initial and final fill levels -/
theorem gasoline_tank_problem (x : ℚ) 
  (h1 : 5/6 * x - 2/3 * x = 18) 
  (h2 : x > 0) : 
  x = 108 ∧ 18 * 4 = 72 := by
  sorry

#check gasoline_tank_problem

end gasoline_tank_problem_l386_38623


namespace fisher_algebra_eligibility_l386_38679

/-- Determines if a student is eligible for algebra based on their quarterly scores -/
def isEligible (q1 q2 q3 q4 : ℚ) : Prop :=
  (q1 + q2 + q3 + q4) / 4 ≥ 83

/-- Fisher's minimum required score for the 4th quarter -/
def fisherMinScore : ℚ := 98

theorem fisher_algebra_eligibility :
  ∀ q4 : ℚ,
  isEligible 82 77 75 q4 ↔ q4 ≥ fisherMinScore :=
by sorry

#check fisher_algebra_eligibility

end fisher_algebra_eligibility_l386_38679


namespace area_of_four_squares_l386_38641

/-- The area of a shape composed of four identical squares with side length 3 cm is 36 cm² -/
theorem area_of_four_squares (side_length : ℝ) (h1 : side_length = 3) : 
  4 * (side_length ^ 2) = 36 := by
  sorry

end area_of_four_squares_l386_38641


namespace intersection_points_l386_38699

-- Define the polar equations
def line_equation (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 4
def curve_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define the constraints
def valid_polar_coord (ρ θ : ℝ) : Prop := ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Theorem statement
theorem intersection_points :
  ∀ ρ θ, valid_polar_coord ρ θ →
    (line_equation ρ θ ∧ curve_equation ρ θ) →
    ((ρ = 4 ∧ θ = 0) ∨ (ρ = 2 * Real.sqrt 2 ∧ θ = Real.pi / 4)) :=
by sorry

end intersection_points_l386_38699


namespace box_packing_problem_l386_38601

theorem box_packing_problem (x y : ℤ) : 
  (3 * x + 4 * y = 108) → 
  (2 * x + 3 * y = 76) → 
  (x = 20 ∧ y = 12) := by
sorry

end box_packing_problem_l386_38601


namespace angle_measure_l386_38616

theorem angle_measure (x : Real) : 
  (0.4 * (180 - x) = 90 - x) → x = 30 := by
  sorry

end angle_measure_l386_38616


namespace concentric_circles_radius_change_l386_38639

theorem concentric_circles_radius_change (R_o R_i : ℝ) 
  (h1 : R_o = 6)
  (h2 : R_i = 4)
  (h3 : R_o > R_i)
  (h4 : 0 < R_i)
  (h5 : 0 < R_o) :
  let A_original := π * (R_o^2 - R_i^2)
  let R_i_new := R_i * 0.75
  let A_new := A_original * 3.6
  ∃ x : ℝ, 
    (π * ((R_o * (1 + x/100))^2 - R_i_new^2) = A_new) ∧
    x = 50 :=
sorry

end concentric_circles_radius_change_l386_38639


namespace sqrt_inequality_l386_38643

theorem sqrt_inequality : Real.sqrt 6 - Real.sqrt 5 > 2 * Real.sqrt 2 - Real.sqrt 7 := by
  sorry

end sqrt_inequality_l386_38643


namespace min_value_of_a_l386_38688

theorem min_value_of_a (x a : ℝ) : 
  (∃ x, |x - 1| + |x + a| ≤ 8) → a ≥ -9 :=
sorry

end min_value_of_a_l386_38688


namespace sum_of_squares_equals_165_l386_38659

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Property of combination numbers -/
axiom comb_property (n m : ℕ) : binomial n (m-1) + binomial n m = binomial (n+1) m

/-- Special case of combination numbers -/
axiom comb_special_case : binomial 2 2 = binomial 3 3

/-- The sum of squares of binomial coefficients from C(2,2) to C(10,2) -/
def sum_of_squares : ℕ := 
  binomial 2 2 + binomial 3 2 + binomial 4 2 + binomial 5 2 + 
  binomial 6 2 + binomial 7 2 + binomial 8 2 + binomial 9 2 + binomial 10 2

theorem sum_of_squares_equals_165 : sum_of_squares = 165 := by
  sorry

end sum_of_squares_equals_165_l386_38659


namespace ellipse_hyperbola_intersection_l386_38624

-- Define the ellipse and hyperbola
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / 10 + y^2 / m = 1
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2 / b = 1

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := x = Real.sqrt 10 / 3

-- Define that the ellipse and hyperbola have the same foci
def same_foci (m b : ℝ) : Prop := 10 - m = 1 + b

-- Theorem statement
theorem ellipse_hyperbola_intersection (m b : ℝ) :
  (∃ y, ellipse m (Real.sqrt 10 / 3) y ∧ 
        hyperbola b (Real.sqrt 10 / 3) y ∧
        intersection_point (Real.sqrt 10 / 3) y) →
  same_foci m b →
  m = 1 ∧ b = 8 := by
  sorry

end ellipse_hyperbola_intersection_l386_38624


namespace simplify_exponential_expression_l386_38603

theorem simplify_exponential_expression (a : ℝ) (h : a ≠ 0) :
  (a^9 * a^15) / a^3 = a^21 := by
  sorry

end simplify_exponential_expression_l386_38603


namespace parallel_line_through_point_l386_38609

/-- A line passing through the point (-3, -1) and parallel to x - 3y - 1 = 0 has the equation x - 3y = 0 -/
theorem parallel_line_through_point : 
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ x - 3*y = 0) →
    (-3, -1) ∈ l →
    (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (t : ℝ), x = t ∧ y = (t - 1) / 3) →
    True :=
by sorry

end parallel_line_through_point_l386_38609


namespace boot_price_calculation_l386_38677

theorem boot_price_calculation (discount_percent : ℝ) (discounted_price : ℝ) : 
  discount_percent = 20 → discounted_price = 72 → 
  discounted_price / (1 - discount_percent / 100) = 90 := by
sorry

end boot_price_calculation_l386_38677


namespace weight_difference_e_d_l386_38626

/-- Given the weights of individuals A, B, C, D, and E, prove that E weighs 3 kg more than D. -/
theorem weight_difference_e_d (w_a w_b w_c w_d w_e : ℝ) : 
  w_a = 81 →
  (w_a + w_b + w_c) / 3 = 70 →
  (w_a + w_b + w_c + w_d) / 4 = 70 →
  (w_b + w_c + w_d + w_e) / 4 = 68 →
  w_e > w_d →
  w_e - w_d = 3 := by
  sorry

end weight_difference_e_d_l386_38626


namespace expand_product_l386_38620

theorem expand_product (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * ((7 / y) + 8 * y^2 - 3 * y) = 3 / y + (24 * y^2 - 9 * y) / 7 := by
  sorry

end expand_product_l386_38620


namespace line_intercepts_l386_38628

/-- Given a line with equation 4x + 6y = 24, prove its x-intercept and y-intercept -/
theorem line_intercepts (x y : ℝ) :
  4 * x + 6 * y = 24 →
  (x = 6 ∧ y = 0) ∨ (x = 0 ∧ y = 4) :=
by sorry

end line_intercepts_l386_38628


namespace sin_minus_cos_sqrt_two_l386_38613

theorem sin_minus_cos_sqrt_two (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi →
  Real.sin x - Real.cos x = Real.sqrt 2 →
  x = 3 * Real.pi / 4 := by
sorry

end sin_minus_cos_sqrt_two_l386_38613


namespace ten_row_triangle_pieces_l386_38606

/-- The number of rods in the nth row of the triangle -/
def rods_in_row (n : ℕ) : ℕ := 3 * n

/-- The total number of rods in a triangle with n rows -/
def total_rods (n : ℕ) : ℕ := (n * (n + 1) * 3) / 2

/-- The number of connectors in a triangle with n rows of rods -/
def total_connectors (n : ℕ) : ℕ := ((n + 1) * (n + 2)) / 2

/-- The total number of pieces in a triangle with n rows of rods -/
def total_pieces (n : ℕ) : ℕ := total_rods n + total_connectors n

theorem ten_row_triangle_pieces :
  total_pieces 10 = 231 := by sorry

end ten_row_triangle_pieces_l386_38606


namespace thirtieth_triangular_number_sum_of_30th_and_29th_triangular_numbers_l386_38684

-- Define the triangular number function
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem for the 30th triangular number
theorem thirtieth_triangular_number : triangularNumber 30 = 465 := by
  sorry

-- Theorem for the sum of 30th and 29th triangular numbers
theorem sum_of_30th_and_29th_triangular_numbers :
  triangularNumber 30 + triangularNumber 29 = 900 := by
  sorry

end thirtieth_triangular_number_sum_of_30th_and_29th_triangular_numbers_l386_38684


namespace six_digit_numbers_with_zero_count_l386_38652

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with no zeros -/
def six_digit_numbers_no_zero : ℕ := 531441

/-- The number of 6-digit numbers with at least one zero -/
def six_digit_numbers_with_zero : ℕ := total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_zero_count : six_digit_numbers_with_zero = 368559 := by
  sorry

end six_digit_numbers_with_zero_count_l386_38652


namespace arithmetic_sequence_sum_l386_38667

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 48) →
  (a 2 + a 5 + a 8 = 40) →
  (a 3 + a 6 + a 9 = 32) :=
by sorry

end arithmetic_sequence_sum_l386_38667


namespace marbles_theorem_l386_38683

def marbles_problem (total : ℕ) (colors : ℕ) (red_lost : ℕ) : ℕ :=
  let marbles_per_color := total / colors
  let red_remaining := marbles_per_color - red_lost
  let blue_remaining := marbles_per_color - (2 * red_lost)
  let yellow_remaining := marbles_per_color - (3 * red_lost)
  red_remaining + blue_remaining + yellow_remaining

theorem marbles_theorem :
  marbles_problem 72 3 5 = 42 := by sorry

end marbles_theorem_l386_38683


namespace min_value_interval_min_value_interval_converse_l386_38630

/-- The function f(x) = x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- The theorem stating the possible values of a -/
theorem min_value_interval (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 6), f x ≥ 9) ∧
  (∃ x ∈ Set.Icc a (a + 6), f x = 9) →
  a = 2 ∨ a = -10 := by
  sorry

/-- The converse theorem -/
theorem min_value_interval_converse :
  ∀ a : ℝ, (a = 2 ∨ a = -10) →
  (∀ x ∈ Set.Icc a (a + 6), f x ≥ 9) ∧
  (∃ x ∈ Set.Icc a (a + 6), f x = 9) := by
  sorry

end min_value_interval_min_value_interval_converse_l386_38630


namespace brown_leaves_percentage_l386_38625

/-- Given a collection of leaves with known percentages of green and yellow leaves,
    calculate the percentage of brown leaves. -/
theorem brown_leaves_percentage
  (total_leaves : ℕ)
  (green_percentage : ℚ)
  (yellow_count : ℕ)
  (h1 : total_leaves = 25)
  (h2 : green_percentage = 1/5)
  (h3 : yellow_count = 15) :
  (total_leaves : ℚ) - green_percentage * total_leaves - yellow_count = 1/5 * total_leaves :=
sorry

end brown_leaves_percentage_l386_38625


namespace line_always_intersects_ellipse_iff_m_in_range_l386_38678

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the ellipse equation
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / 5 + y^2 / m = 1

-- Theorem statement
theorem line_always_intersects_ellipse_iff_m_in_range :
  ∀ m : ℝ, (∀ k : ℝ, ∃ x y : ℝ, line k x = y ∧ ellipse m x y) ↔ 
  (m ≥ 1 ∧ m ≠ 5) :=
sorry

end line_always_intersects_ellipse_iff_m_in_range_l386_38678


namespace solve_for_y_l386_38615

theorem solve_for_y (x y p : ℝ) (h : p = (5 * x * y) / (x - y)) : 
  y = (p * x) / (5 * x + p) := by
sorry

end solve_for_y_l386_38615


namespace count_b_k_divisible_by_11_l386_38649

-- Define b_k as a function that takes k and returns the concatenated number
def b (k : ℕ) : ℕ := sorry

-- Define a function to count how many b_k are divisible by 11 for 1 ≤ k ≤ 50
def count_divisible_by_11 : ℕ := sorry

-- Theorem stating that the count of b_k divisible by 11 for 1 ≤ k ≤ 50 is equal to X
theorem count_b_k_divisible_by_11 : count_divisible_by_11 = X := by sorry

end count_b_k_divisible_by_11_l386_38649


namespace independence_test_not_always_correct_l386_38611

-- Define what an independence test is
def IndependenceTest : Type := sorry

-- Define a function that represents the conclusion of an independence test
def conclusion (test : IndependenceTest) : Prop := sorry

-- Theorem stating that the conclusion of an independence test is not always correct
theorem independence_test_not_always_correct :
  ¬ (∀ (test : IndependenceTest), conclusion test) := by sorry

end independence_test_not_always_correct_l386_38611


namespace robin_total_bottles_l386_38675

/-- The total number of water bottles Robin drank throughout the day -/
def total_bottles (morning afternoon evening night : ℕ) : ℕ :=
  morning + afternoon + evening + night

/-- Theorem stating that Robin drank 24 bottles in total -/
theorem robin_total_bottles : 
  total_bottles 7 9 5 3 = 24 := by
  sorry

end robin_total_bottles_l386_38675


namespace non_zero_terms_count_l386_38672

/-- The expression to be expanded and simplified -/
def expression (x : ℝ) : ℝ := (x - 3) * (x^2 + 5*x + 8) + 2 * (x^3 + 3*x^2 - x - 4)

/-- The expanded and simplified form of the expression -/
def simplified_expression (x : ℝ) : ℝ := 3*x^3 + 8*x^2 - 9*x - 32

/-- Theorem stating that the number of non-zero terms in the simplified expression is 4 -/
theorem non_zero_terms_count : 
  (∃ (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0), 
    ∀ x, simplified_expression x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ (a b c d e : ℝ), ¬(∀ x, simplified_expression x = a*x^4 + b*x^3 + c*x^2 + d*x + e)) :=
sorry

end non_zero_terms_count_l386_38672


namespace car_A_time_l386_38619

/-- Proves that Car A takes 8 hours to reach its destination given the specified conditions -/
theorem car_A_time (speed_A speed_B time_B : ℝ) (ratio : ℝ) : 
  speed_A = 50 →
  speed_B = 25 →
  time_B = 4 →
  ratio = 4 →
  speed_A * (ratio * speed_B * time_B) / speed_A = 8 :=
by
  sorry

#check car_A_time

end car_A_time_l386_38619


namespace property_transaction_outcome_l386_38602

def initial_value : ℝ := 15000
def profit_percentage : ℝ := 0.15
def loss_percentage : ℝ := 0.05

theorem property_transaction_outcome :
  let first_sale := initial_value * (1 + profit_percentage)
  let second_sale := first_sale * (1 - loss_percentage)
  first_sale - second_sale = 862.50 := by sorry

end property_transaction_outcome_l386_38602


namespace six_factorial_divisors_l386_38656

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define function to count positive divisors
def count_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- Theorem statement
theorem six_factorial_divisors :
  count_divisors (factorial 6) = 30 := by
  sorry

end six_factorial_divisors_l386_38656


namespace corner_removed_cube_edge_count_l386_38633

/-- Represents a cube with a given side length -/
structure Cube :=
  (sideLength : ℝ)

/-- Represents the solid formed by removing smaller cubes from the corners of a larger cube -/
structure CornerRemovedCube :=
  (originalCube : Cube)
  (removedCubeSize : ℝ)

/-- Calculates the number of edges in the solid formed by removing smaller cubes from the corners of a larger cube -/
def edgeCount (c : CornerRemovedCube) : ℕ :=
  12 * 2  -- Each original edge is split into two

/-- Theorem stating that removing cubes of side length 2 from each corner of a cube with side length 4 results in a solid with 24 edges -/
theorem corner_removed_cube_edge_count :
  let originalCube : Cube := ⟨4⟩
  let cornerRemovedCube : CornerRemovedCube := ⟨originalCube, 2⟩
  edgeCount cornerRemovedCube = 24 :=
by sorry

end corner_removed_cube_edge_count_l386_38633


namespace car_trip_duration_l386_38662

/-- Proves that a car trip with given conditions has a total duration of 8 hours -/
theorem car_trip_duration (initial_speed : ℝ) (initial_time : ℝ) (later_speed : ℝ) (avg_speed : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : initial_time = 6)
  (h3 : later_speed = 46)
  (h4 : avg_speed = 34) :
  ∃ (total_time : ℝ), 
    (initial_speed * initial_time + later_speed * (total_time - initial_time)) / total_time = avg_speed ∧
    total_time = 8 := by
  sorry


end car_trip_duration_l386_38662


namespace tan_theta_two_implies_expression_equals_six_fifths_l386_38693

theorem tan_theta_two_implies_expression_equals_six_fifths (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / 
  (Real.sqrt 2 * Real.cos (θ - π / 4)) = 6 / 5 := by
  sorry

end tan_theta_two_implies_expression_equals_six_fifths_l386_38693


namespace dartboard_area_ratio_l386_38695

theorem dartboard_area_ratio :
  let outer_square_side : ℝ := 4
  let inner_square_side : ℝ := 2
  let triangle_leg : ℝ := 1 / Real.sqrt 2
  let s : ℝ := (1 / 2) * triangle_leg * triangle_leg
  let p : ℝ := (1 / 2) * (inner_square_side + outer_square_side) * (outer_square_side / 2 - triangle_leg)
  p / s = 12 := by sorry

end dartboard_area_ratio_l386_38695


namespace inequality_proof_l386_38610

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_abc : a + b + c = 1) :
  (7 + 2*b) / (1 + a) + (7 + 2*c) / (1 + b) + (7 + 2*a) / (1 + c) ≥ 69/4 := by
  sorry

end inequality_proof_l386_38610


namespace hcf_problem_l386_38638

theorem hcf_problem (a b : ℕ) (h : ℕ) : 
  (max a b = 600) →
  (∃ (k : ℕ), lcm a b = h * 11 * 12) →
  gcd a b = 12 :=
by
  sorry

end hcf_problem_l386_38638


namespace geometric_sequence_sum_l386_38691

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 1 + a 4 + a 7 = 10 →
  a 3 + a 6 + a 9 = 20 := by
sorry

end geometric_sequence_sum_l386_38691


namespace a_in_A_sufficient_not_necessary_for_a_in_B_l386_38647

def A : Set ℝ := {1, 2, 3}
def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem a_in_A_sufficient_not_necessary_for_a_in_B :
  (∀ a, a ∈ A → a ∈ B) ∧ (∃ a, a ∈ B ∧ a ∉ A) := by sorry

end a_in_A_sufficient_not_necessary_for_a_in_B_l386_38647


namespace movie_production_cost_ratio_l386_38680

/-- Proves that the ratio of equipment rental cost to the combined cost of food and actors is 2:1 --/
theorem movie_production_cost_ratio :
  let actor_cost : ℕ := 1200
  let num_people : ℕ := 50
  let food_cost_per_person : ℕ := 3
  let total_food_cost : ℕ := num_people * food_cost_per_person
  let combined_cost : ℕ := actor_cost + total_food_cost
  let selling_price : ℕ := 10000
  let profit : ℕ := 5950
  let total_cost : ℕ := selling_price - profit
  let equipment_cost : ℕ := total_cost - combined_cost
  equipment_cost / combined_cost = 2 := by sorry

end movie_production_cost_ratio_l386_38680


namespace sum_of_cubes_of_roots_l386_38637

theorem sum_of_cubes_of_roots (a b c : ℝ) : 
  (a^3 - 2*a^2 + 2*a - 3 = 0) →
  (b^3 - 2*b^2 + 2*b - 3 = 0) →
  (c^3 - 2*c^2 + 2*c - 3 = 0) →
  a^3 + b^3 + c^3 = 5 := by
sorry

end sum_of_cubes_of_roots_l386_38637


namespace derivative_of_y_l386_38631

noncomputable def y (x : ℝ) : ℝ := 
  -1 / (3 * Real.sin x ^ 3) - 1 / Real.sin x + 1 / 2 * Real.log ((1 + Real.sin x) / (1 - Real.sin x))

theorem derivative_of_y (x : ℝ) (h : x ∉ Set.range (fun n => n * π)) :
  deriv y x = 1 / (Real.cos x * Real.sin x ^ 4) :=
sorry

end derivative_of_y_l386_38631


namespace sum_proper_divisors_81_l386_38607

def proper_divisors (n : ℕ) : Set ℕ :=
  {d : ℕ | d ∣ n ∧ d ≠ n}

theorem sum_proper_divisors_81 :
  (Finset.sum (Finset.filter (· ≠ 81) (Finset.range 82)) (λ x => if x ∣ 81 then x else 0)) = 40 := by
  sorry

end sum_proper_divisors_81_l386_38607


namespace john_shopping_expense_l386_38655

/-- Given John's shopping scenario, prove the amount spent on pants. -/
theorem john_shopping_expense (tshirt_count : ℕ) (tshirt_price : ℕ) (total_spent : ℕ) 
  (h1 : tshirt_count = 3)
  (h2 : tshirt_price = 20)
  (h3 : total_spent = 110) :
  total_spent - (tshirt_count * tshirt_price) = 50 := by
  sorry

end john_shopping_expense_l386_38655


namespace dave_winfield_home_runs_l386_38653

theorem dave_winfield_home_runs : ∃ (x : ℕ), 
  (755 = 2 * x - 175) ∧ x = 465 := by sorry

end dave_winfield_home_runs_l386_38653


namespace jills_uphill_speed_l386_38674

/-- Jill's speed running up the hill -/
def uphill_speed : ℝ := 9

/-- Jill's speed running down the hill -/
def downhill_speed : ℝ := 12

/-- Hill height in feet -/
def hill_height : ℝ := 900

/-- Total time for running up and down the hill in seconds -/
def total_time : ℝ := 175

theorem jills_uphill_speed :
  (hill_height / uphill_speed + hill_height / downhill_speed = total_time) ∧
  (uphill_speed > 0) ∧
  (downhill_speed > 0) ∧
  (hill_height > 0) ∧
  (total_time > 0) := by
  sorry

end jills_uphill_speed_l386_38674


namespace inverse_composition_l386_38654

-- Define the functions h and k
noncomputable def h : ℝ → ℝ := sorry
noncomputable def k : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_composition (x : ℝ) : 
  (h⁻¹ ∘ k) x = 3 * x - 4 → k⁻¹ (h 8) = 8 := by sorry

end inverse_composition_l386_38654


namespace grain_movement_representation_l386_38664

-- Define the type for grain movement
inductive GrainMovement
  | arrival
  | departure

-- Define a function to represent the sign of grain movement
def signOfMovement (g : GrainMovement) : Int :=
  match g with
  | GrainMovement.arrival => 1
  | GrainMovement.departure => -1

-- Define the theorem
theorem grain_movement_representation :
  ∀ (quantity : ℕ),
  (signOfMovement GrainMovement.arrival * quantity = 30) →
  (signOfMovement GrainMovement.departure * quantity = -30) :=
by
  sorry


end grain_movement_representation_l386_38664


namespace perfect_square_pair_iff_in_solution_set_l386_38671

/-- A pair of integers (a, b) satisfies the perfect square property if
    a^2 + 4b and b^2 + 4a are both perfect squares. -/
def PerfectSquarePair (a b : ℤ) : Prop :=
  ∃ (m n : ℤ), a^2 + 4*b = m^2 ∧ b^2 + 4*a = n^2

/-- The set of solutions for the perfect square pair problem. -/
def SolutionSet : Set (ℤ × ℤ) :=
  {p | ∃ (k : ℤ), p = (k^2, 0) ∨ p = (0, k^2) ∨ p = (k, 1-k) ∨
                   p = (-6, -5) ∨ p = (-5, -6) ∨ p = (-4, -4)}

/-- The main theorem stating that a pair (a, b) satisfies the perfect square property
    if and only if it belongs to the solution set. -/
theorem perfect_square_pair_iff_in_solution_set (a b : ℤ) :
  PerfectSquarePair a b ↔ (a, b) ∈ SolutionSet := by
  sorry

end perfect_square_pair_iff_in_solution_set_l386_38671


namespace equation_solutions_count_l386_38676

theorem equation_solutions_count :
  let f : ℝ → ℝ := λ θ => (Real.sin θ ^ 2 - 1) * (2 * Real.sin θ ^ 2 - 1)
  ∃! (s : Finset ℝ), s.card = 6 ∧ 
    (∀ θ ∈ s, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ f θ = 0) ∧
    (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ f θ = 0 → θ ∈ s) :=
by
  sorry

end equation_solutions_count_l386_38676


namespace twenty_sixth_term_is_79_l386_38646

/-- An arithmetic sequence with first term 4 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

/-- The 26th term of the arithmetic sequence is 79 -/
theorem twenty_sixth_term_is_79 : arithmetic_sequence 26 = 79 := by
  sorry

end twenty_sixth_term_is_79_l386_38646


namespace jerry_added_six_figures_l386_38648

/-- Given that Jerry initially had 4 action figures and ended up with 10 action figures in total,
    prove that he added 6 action figures. -/
theorem jerry_added_six_figures (initial : ℕ) (total : ℕ) (added : ℕ)
    (h1 : initial = 4)
    (h2 : total = 10)
    (h3 : total = initial + added) :
  added = 6 := by
  sorry

end jerry_added_six_figures_l386_38648


namespace no_real_solutions_l386_38644

theorem no_real_solutions : ¬∃ (x : ℝ), x + 64 / (x + 3) = -13 := by
  sorry

end no_real_solutions_l386_38644


namespace fish_population_estimate_l386_38622

/-- Estimate the number of fish in a reservoir using the capture-recapture method. -/
theorem fish_population_estimate
  (M : ℕ) -- Number of fish initially captured, marked, and released
  (m : ℕ) -- Number of fish captured in the second round
  (n : ℕ) -- Number of marked fish found in the second capture
  (h1 : M > 0)
  (h2 : m > 0)
  (h3 : n > 0)
  (h4 : n ≤ m)
  (h5 : n ≤ M) :
  ∃ x : ℚ, x = (M * m : ℚ) / n ∧ x > 0 :=
sorry

end fish_population_estimate_l386_38622


namespace divisor_problem_l386_38642

theorem divisor_problem (original : Nat) (subtracted : Nat) (remaining : Nat) :
  original = 165826 →
  subtracted = 2 →
  remaining = original - subtracted →
  (∃ (d : Nat), d > 1 ∧ remaining % d = 0 ∧ ∀ (k : Nat), k > d → remaining % k ≠ 0) →
  (∃ (d : Nat), d = 2 ∧ remaining % d = 0 ∧ ∀ (k : Nat), k > d → remaining % k ≠ 0) :=
by sorry

end divisor_problem_l386_38642


namespace orange_juice_bottles_l386_38650

/-- Proves that the number of orange juice bottles is 42 given the conditions of the problem -/
theorem orange_juice_bottles (orange_cost apple_cost total_bottles total_cost : ℚ)
  (h1 : orange_cost = 70/100)
  (h2 : apple_cost = 60/100)
  (h3 : total_bottles = 70)
  (h4 : total_cost = 4620/100)
  (h5 : ∃ (orange apple : ℚ), orange + apple = total_bottles ∧ 
                               orange * orange_cost + apple * apple_cost = total_cost) :
  ∃ (orange : ℚ), orange = 42 ∧ 
    ∃ (apple : ℚ), orange + apple = total_bottles ∧ 
                    orange * orange_cost + apple * apple_cost = total_cost :=
by sorry


end orange_juice_bottles_l386_38650


namespace p_and_q_iff_a_in_range_l386_38640

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x, x^2 + 2*a*x + a + 2 = 0

def q (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

-- State the theorem
theorem p_and_q_iff_a_in_range (a : ℝ) : 
  (p a ∧ q a) ↔ a ∈ Set.Iic (-1) :=
sorry

end p_and_q_iff_a_in_range_l386_38640


namespace first_number_equation_l386_38663

theorem first_number_equation : ∃ x : ℝ, 
  x + 17.0005 - 9.1103 = 20.011399999999995 ∧ 
  x = 12.121199999999995 := by sorry

end first_number_equation_l386_38663


namespace inverse_proposition_false_l386_38635

theorem inverse_proposition_false : 
  ¬(∀ (a b c : ℝ), a > b → a / (c^2) > b / (c^2)) := by
sorry

end inverse_proposition_false_l386_38635


namespace num_subsets_eq_two_pow_l386_38665

/-- The number of subsets of a finite set -/
def num_subsets (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of subsets of a set with n elements is 2^n -/
theorem num_subsets_eq_two_pow (n : ℕ) : num_subsets n = 2^n := by
  sorry

end num_subsets_eq_two_pow_l386_38665
