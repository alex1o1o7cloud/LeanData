import Mathlib

namespace cylinder_height_from_cube_water_l3437_343763

/-- The height of a cylinder filled with water from a cube -/
theorem cylinder_height_from_cube_water (cube_edge : ℝ) (cylinder_base_area : ℝ) 
  (h_cube_edge : cube_edge = 6)
  (h_cylinder_base : cylinder_base_area = 18)
  (h_water_conserved : cube_edge ^ 3 = cylinder_base_area * cylinder_height) :
  cylinder_height = 12 := by
  sorry

#check cylinder_height_from_cube_water

end cylinder_height_from_cube_water_l3437_343763


namespace function_derivative_at_zero_l3437_343706

theorem function_derivative_at_zero 
  (f : ℝ → ℝ) 
  (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x = x^2 + 2 * (deriv f 2) * x + 3) : 
  deriv f 0 = -8 := by
sorry

end function_derivative_at_zero_l3437_343706


namespace four_lines_theorem_l3437_343767

-- Define the type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define the type for points
def Point : Type := ℝ × ℝ

-- Define a function to check if a point is on a line
def PointOnLine (p : Point) (l : Line) : Prop := l p.1 p.2

-- Define a function to check if a point is on a circle
def PointOnCircle (p : Point) (c : Point → Prop) : Prop := c p

-- Define a function to get the intersection point of two lines
def Intersection (l1 l2 : Line) : Point := sorry

-- Define a function to get the circle passing through three points
def CircleThrough (p1 p2 p3 : Point) : Point → Prop := sorry

-- Define a function to get the point corresponding to a triple of lines
def CorrespondingPoint (l1 l2 l3 : Line) : Point := sorry

-- State the theorem
theorem four_lines_theorem 
  (l1 l2 l3 l4 : Line) 
  (p1 p2 p3 p4 : Point) 
  (c : Point → Prop) 
  (h1 : PointOnLine p1 l1) 
  (h2 : PointOnLine p2 l2) 
  (h3 : PointOnLine p3 l3) 
  (h4 : PointOnLine p4 l4) 
  (hc1 : PointOnCircle p1 c) 
  (hc2 : PointOnCircle p2 c) 
  (hc3 : PointOnCircle p3 c) 
  (hc4 : PointOnCircle p4 c) :
  ∃ (c' : Point → Prop), 
    PointOnCircle (CorrespondingPoint l2 l3 l4) c' ∧ 
    PointOnCircle (CorrespondingPoint l1 l3 l4) c' ∧ 
    PointOnCircle (CorrespondingPoint l1 l2 l4) c' ∧ 
    PointOnCircle (CorrespondingPoint l1 l2 l3) c' :=
sorry

end four_lines_theorem_l3437_343767


namespace remainder_theorem_remainder_problem_l3437_343781

/-- A polynomial of the form Dx^4 + Ex^2 + Fx - 2 -/
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x - 2

/-- The remainder theorem -/
theorem remainder_theorem {p : ℝ → ℝ} {a r : ℝ} :
  (∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + r) ↔ p a = r :=
sorry

theorem remainder_problem (D E F : ℝ) :
  (∃ r : ℝ, ∀ x, q D E F x = (x - 2) * r + 14) →
  (∃ s : ℝ, ∀ x, q D E F x = (x + 2) * s - 18) :=
sorry

end remainder_theorem_remainder_problem_l3437_343781


namespace subject_score_proof_l3437_343745

theorem subject_score_proof (physics chemistry mathematics : ℕ) : 
  (physics + chemistry + mathematics) / 3 = 85 →
  (physics + mathematics) / 2 = 90 →
  (physics + chemistry) / 2 = 70 →
  physics = 65 →
  mathematics = 115 := by
sorry

end subject_score_proof_l3437_343745


namespace exists_word_with_multiple_associations_l3437_343736

-- Define the alphabet A
def A : Type := Char

-- Define the set of all words over A
def A_star : Type := List A

-- Define the transducer T'
def T' : A_star → Set A_star := sorry

-- Define the property of a word having multiple associations
def has_multiple_associations (v : A_star) : Prop :=
  ∃ (w1 w2 : A_star), w1 ∈ T' v ∧ w2 ∈ T' v ∧ w1 ≠ w2

-- Theorem statement
theorem exists_word_with_multiple_associations :
  ∃ (v : A_star), has_multiple_associations v := by sorry

end exists_word_with_multiple_associations_l3437_343736


namespace three_numbers_theorem_l3437_343707

theorem three_numbers_theorem (x y z : ℝ) 
  (h1 : (x + y + z)^2 = x^2 + y^2 + z^2)
  (h2 : x * y = z^2) :
  (x = 0 ∧ z = 0) ∨ (y = 0 ∧ z = 0) :=
by sorry

end three_numbers_theorem_l3437_343707


namespace ellipse_properties_hyperbola_properties_parabola_properties_l3437_343778

/-- Ellipse properties -/
theorem ellipse_properties (x y : ℝ) :
  x^2 / 4 + y^2 = 1 →
  ∃ (a b : ℝ), a = 2 * b ∧ a > 0 ∧ b > 0 ∧
    x^2 / a^2 + y^2 / b^2 = 1 ∧
    (2 : ℝ)^2 / a^2 + 0^2 / b^2 = 1 :=
sorry

/-- Hyperbola properties -/
theorem hyperbola_properties (x y : ℝ) :
  y^2 / 20 - x^2 / 16 = 1 →
  ∃ (a b : ℝ), a = 2 * Real.sqrt 5 ∧ a > 0 ∧ b > 0 ∧
    y^2 / a^2 - x^2 / b^2 = 1 ∧
    5^2 / a^2 - 2^2 / b^2 = 1 :=
sorry

/-- Parabola properties -/
theorem parabola_properties (x y : ℝ) :
  y^2 = 4 * x →
  ∃ (p : ℝ), p > 0 ∧
    y^2 = 4 * p * x ∧
    (-2)^2 = 4 * p * 1 ∧
    (∀ (x₀ y₀ : ℝ), y₀^2 = 4 * p * x₀ → x₀ = 0 → y₀ = 0) :=
sorry

end ellipse_properties_hyperbola_properties_parabola_properties_l3437_343778


namespace framed_painting_ratio_l3437_343703

theorem framed_painting_ratio : 
  ∀ (y : ℝ), 
    y > 0 → 
    (15 + 2*y) * (20 + 6*y) = 2 * 15 * 20 → 
    (min (15 + 2*y) (20 + 6*y)) / (max (15 + 2*y) (20 + 6*y)) = 4 / 7 := by
  sorry

end framed_painting_ratio_l3437_343703


namespace greater_number_is_eighteen_l3437_343760

theorem greater_number_is_eighteen (x y : ℝ) 
  (sum : x + y = 30)
  (diff : x - y = 6)
  (y_lower_bound : y ≥ 10)
  (x_greater : x > y) :
  x = 18 := by
sorry

end greater_number_is_eighteen_l3437_343760


namespace solve_equation_l3437_343750

theorem solve_equation (x : ℝ) (h : 5*x - 8 = 15*x + 14) : 6*(x + 3) = 4.8 := by
  sorry

end solve_equation_l3437_343750


namespace sphere_volume_increase_on_doubling_radius_l3437_343795

theorem sphere_volume_increase_on_doubling_radius :
  ∀ (r : ℝ), r > 0 →
  (4 / 3 * Real.pi * (2 * r)^3) = 8 * (4 / 3 * Real.pi * r^3) :=
by
  sorry

end sphere_volume_increase_on_doubling_radius_l3437_343795


namespace sum_of_integers_l3437_343790

theorem sum_of_integers (x y : ℤ) 
  (h1 : x^2 + y^2 = 290) 
  (h2 : x * y = 96) : 
  x + y = 22 := by
sorry

end sum_of_integers_l3437_343790


namespace delta_max_success_ratio_l3437_343757

/-- Represents a participant's scores in a two-day math competition -/
structure Participant where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ

/-- The maximum possible two-day success ratio for Delta given the competition conditions -/
theorem delta_max_success_ratio 
  (gamma : Participant)
  (total_points : ℕ)
  (h_total : gamma.day1_attempted + gamma.day2_attempted = total_points)
  (h_gamma_day1 : gamma.day1_score = 210 ∧ gamma.day1_attempted = 360)
  (h_gamma_day2 : gamma.day2_score = 150 ∧ gamma.day2_attempted = 240)
  (h_gamma_ratio : (gamma.day1_score + gamma.day2_score : ℚ) / total_points = 3/5) :
  ∃ (delta : Participant),
    (delta.day1_attempted + delta.day2_attempted = total_points) ∧
    (delta.day1_attempted ≠ gamma.day1_attempted) ∧
    (delta.day1_score > 0 ∧ delta.day2_score > 0) ∧
    ((delta.day1_score : ℚ) / delta.day1_attempted < (gamma.day1_score : ℚ) / gamma.day1_attempted) ∧
    ((delta.day2_score : ℚ) / delta.day2_attempted < (gamma.day2_score : ℚ) / gamma.day2_attempted) ∧
    ((delta.day1_score + delta.day2_score : ℚ) / total_points ≤ 1/4) ∧
    ∀ (delta' : Participant),
      (delta'.day1_attempted + delta'.day2_attempted = total_points) →
      (delta'.day1_attempted ≠ gamma.day1_attempted) →
      (delta'.day1_score > 0 ∧ delta'.day2_score > 0) →
      ((delta'.day1_score : ℚ) / delta'.day1_attempted < (gamma.day1_score : ℚ) / gamma.day1_attempted) →
      ((delta'.day2_score : ℚ) / delta'.day2_attempted < (gamma.day2_score : ℚ) / gamma.day2_attempted) →
      ((delta'.day1_score + delta'.day2_score : ℚ) / total_points ≤ (delta.day1_score + delta.day2_score : ℚ) / total_points) := by
  sorry


end delta_max_success_ratio_l3437_343757


namespace max_value_constraint_l3437_343785

theorem max_value_constraint (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) : 
  ∃ (M : ℝ), M = 3.2 ∧ ∀ (x y z : ℝ), 9*x^2 + 4*y^2 + 25*z^2 = 1 → 6*x + 3*y + 10*z ≤ M :=
sorry

end max_value_constraint_l3437_343785


namespace mean_home_runs_l3437_343734

def home_runs : List ℕ := [5, 6, 7, 8, 9]
def players : List ℕ := [4, 5, 3, 2, 2]

theorem mean_home_runs :
  let total_hrs := (List.zip home_runs players).map (fun (hr, p) => hr * p) |>.sum
  let total_players := players.sum
  (total_hrs : ℚ) / total_players = 105 / 16 := by sorry

end mean_home_runs_l3437_343734


namespace line_through_points_with_45_degree_angle_l3437_343776

/-- Given a line passing through points (-1, 3) and (2, a) with an inclination angle of 45°, prove that a = 6 -/
theorem line_through_points_with_45_degree_angle (a : ℝ) : 
  (∃ (line : ℝ → ℝ), 
    line (-1) = 3 ∧ 
    line 2 = a ∧ 
    (∀ x y : ℝ, y = line x → (y - 3) / (x - (-1)) = 1)) → 
  a = 6 := by
sorry

end line_through_points_with_45_degree_angle_l3437_343776


namespace tinas_time_l3437_343727

/-- Represents the mile times of three runners with specific speed relationships -/
structure RunnerTimes where
  tom : ℝ
  tina : ℝ
  tony : ℝ
  tina_slower : tina = 3 * tom
  tony_faster : tony = tina / 2
  total_time : tom + tina + tony = 11

/-- Theorem stating that given the conditions, Tina's mile time is 6 minutes -/
theorem tinas_time (rt : RunnerTimes) : rt.tina = 6 := by
  sorry

end tinas_time_l3437_343727


namespace diagonals_bisect_in_rhombus_rectangle_square_l3437_343751

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define the property of diagonals bisecting each other
def diagonals_bisect (q : Quadrilateral) : Prop :=
  let d1_mid := ((q.vertices 0 + q.vertices 2) : ℝ × ℝ) / 2
  let d2_mid := ((q.vertices 1 + q.vertices 3) : ℝ × ℝ) / 2
  d1_mid = d2_mid

-- Define rhombus, rectangle, and square as specific types of quadrilaterals
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- State the theorem
theorem diagonals_bisect_in_rhombus_rectangle_square (q : Quadrilateral) :
  (is_rhombus q ∨ is_rectangle q ∨ is_square q) → diagonals_bisect q :=
by sorry

end diagonals_bisect_in_rhombus_rectangle_square_l3437_343751


namespace total_lives_calculation_l3437_343743

theorem total_lives_calculation (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) : 
  initial_players = 8 → additional_players = 2 → lives_per_player = 6 →
  (initial_players + additional_players) * lives_per_player = 60 := by
  sorry

end total_lives_calculation_l3437_343743


namespace valid_numbers_count_and_max_l3437_343769

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ),
    is_prime x ∧ is_prime y ∧
    n = 4000 + 100 * a + 10 * b + 5 ∧
    n = 5 * x * 11 * y

theorem valid_numbers_count_and_max :
  (∃! (s : Finset ℕ),
    (∀ n ∈ s, is_valid_number n) ∧
    (∀ n, is_valid_number n → n ∈ s) ∧
    s.card = 3) ∧
  (∃ m : ℕ, is_valid_number m ∧ ∀ n, is_valid_number n → n ≤ m) ∧
  (∃ m : ℕ, is_valid_number m ∧ m = 4785) :=
sorry

end valid_numbers_count_and_max_l3437_343769


namespace square_sum_zero_implies_both_zero_l3437_343779

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l3437_343779


namespace factorable_implies_even_b_l3437_343771

/-- A quadratic expression of the form 15x^2 + bx + 15 -/
def quadratic_expr (b : ℤ) (x : ℝ) : ℝ := 15 * x^2 + b * x + 15

/-- Represents a linear binomial factor with integer coefficients -/
structure LinearFactor where
  c : ℤ
  d : ℤ

/-- Checks if a quadratic expression can be factored into two linear binomial factors -/
def is_factorable (b : ℤ) : Prop :=
  ∃ (f1 f2 : LinearFactor), ∀ x, 
    quadratic_expr b x = (f1.c * x + f1.d) * (f2.c * x + f2.d)

theorem factorable_implies_even_b :
  ∀ b : ℤ, is_factorable b → Even b :=
sorry

end factorable_implies_even_b_l3437_343771


namespace truck_distance_from_start_l3437_343704

-- Define the truck's travel distances
def north_distance1 : ℝ := 20
def east_distance : ℝ := 30
def north_distance2 : ℝ := 20

-- Define the total north distance
def total_north_distance : ℝ := north_distance1 + north_distance2

-- Theorem to prove
theorem truck_distance_from_start : 
  Real.sqrt (total_north_distance ^ 2 + east_distance ^ 2) = 50 := by
  sorry

end truck_distance_from_start_l3437_343704


namespace parallelogram_with_equilateral_triangles_l3437_343761

-- Define the points
variable (A B C D P Q : ℝ × ℝ)

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  B.1 - A.1 = D.1 - C.1 ∧ B.2 - A.2 = D.2 - C.2 ∧
  A.1 - D.1 = B.1 - C.1 ∧ A.2 - D.2 = B.2 - C.2

-- Define an equilateral triangle
def is_equilateral_triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 ∧
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = (Z.1 - X.1)^2 + (Z.2 - X.2)^2

-- State the theorem
theorem parallelogram_with_equilateral_triangles
  (h1 : is_parallelogram A B C D)
  (h2 : is_equilateral_triangle B C P)
  (h3 : is_equilateral_triangle C D Q) :
  is_equilateral_triangle A P Q :=
sorry

end parallelogram_with_equilateral_triangles_l3437_343761


namespace post_office_distance_l3437_343774

/-- Proves that the distance of a round trip journey is 10 km given specific conditions -/
theorem post_office_distance (outward_speed return_speed total_time : ℝ) 
  (h1 : outward_speed = 12.5)
  (h2 : return_speed = 2)
  (h3 : total_time = 5.8) : 
  (total_time * outward_speed * return_speed) / (outward_speed + return_speed) = 10 := by
  sorry

end post_office_distance_l3437_343774


namespace exam_students_count_l3437_343728

theorem exam_students_count (N : ℕ) (T : ℕ) : 
  N * 85 = T ∧
  (N - 5) * 90 = T - 300 ∧
  (N - 8) * 95 = T - 465 ∧
  (N - 15) * 100 = T - 955 →
  N = 30 := by
  sorry

end exam_students_count_l3437_343728


namespace function_symmetry_translation_l3437_343718

def symmetric_wrt_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

/-- If the graph of f(x+1) is symmetric to e^x with respect to the y-axis,
    then f(x) = e^(-(x+1)) -/
theorem function_symmetry_translation (f : ℝ → ℝ) :
  symmetric_wrt_y_axis (λ x => f (x + 1)) Real.exp →
  f = λ x => Real.exp (-(x + 1)) :=
by sorry

end function_symmetry_translation_l3437_343718


namespace rotate_D_90_clockwise_l3437_343756

-- Define the rotation matrix for 90° clockwise rotation
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

-- Define the original point D
def D : ℝ × ℝ := (-2, 3)

-- Theorem to prove
theorem rotate_D_90_clockwise :
  rotate90Clockwise D = (3, 2) := by
  sorry

end rotate_D_90_clockwise_l3437_343756


namespace two_draw_probability_l3437_343726

/-- The probability of drawing either a red and blue chip or a blue and green chip
    in two draws with replacement from a bag containing 6 red, 4 blue, and 2 green chips -/
theorem two_draw_probability (red blue green : ℕ) (total : ℕ) : 
  red = 6 → blue = 4 → green = 2 → total = red + blue + green →
  (red / total * blue / total + blue / total * red / total +
   blue / total * green / total + green / total * blue / total : ℚ) = 4 / 9 := by
sorry

end two_draw_probability_l3437_343726


namespace train_length_l3437_343721

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : speed = 45 → time = 16 → speed * time * (1000 / 3600) = 200 := by
  sorry

end train_length_l3437_343721


namespace worm_coverage_l3437_343764

/-- A continuous curve in the plane -/
def ContinuousCurve := Set (ℝ × ℝ)

/-- The length of a continuous curve -/
noncomputable def length (γ : ContinuousCurve) : ℝ := sorry

/-- A semicircle in the plane -/
def Semicircle (center : ℝ × ℝ) (diameter : ℝ) : Set (ℝ × ℝ) := sorry

/-- Whether a set covers another set -/
def covers (A B : Set (ℝ × ℝ)) : Prop := B ⊆ A

theorem worm_coverage (γ : ContinuousCurve) (h : length γ = 1) :
  ∃ (center : ℝ × ℝ), covers (Semicircle center 1) γ := by sorry

end worm_coverage_l3437_343764


namespace point_P_coordinates_l3437_343737

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem point_P_coordinates :
  ∃ (P : ℝ × ℝ), (f' P.1 = 3) ∧ ((P = (-1, -1)) ∨ (P = (1, 1))) :=
sorry

end point_P_coordinates_l3437_343737


namespace linear_function_decreasing_l3437_343773

/-- A linear function y = (m-3)x + 6 + 2m decreases as x increases if and only if m < 3 -/
theorem linear_function_decreasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((m - 3) * x₁ + 6 + 2 * m) > ((m - 3) * x₂ + 6 + 2 * m)) ↔ m < 3 :=
sorry

end linear_function_decreasing_l3437_343773


namespace method_of_continuous_subtraction_equiv_euclid_algorithm_l3437_343748

/-- The Method of Continuous Subtraction as used in ancient Chinese mathematics -/
def methodOfContinuousSubtraction (a b : ℕ) : ℕ :=
  sorry

/-- Euclid's algorithm for finding the greatest common divisor -/
def euclidAlgorithm (a b : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the Method of Continuous Subtraction is equivalent to Euclid's algorithm -/
theorem method_of_continuous_subtraction_equiv_euclid_algorithm :
  ∀ a b : ℕ, methodOfContinuousSubtraction a b = euclidAlgorithm a b :=
sorry

end method_of_continuous_subtraction_equiv_euclid_algorithm_l3437_343748


namespace repeating_decimal_sum_l3437_343709

theorem repeating_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) + (6 / 9999 : ℚ) = 3793 / 9999 := by
  sorry

#check repeating_decimal_sum

end repeating_decimal_sum_l3437_343709


namespace one_third_of_five_times_seven_l3437_343731

theorem one_third_of_five_times_seven :
  (1/3 : ℚ) * (5 * 7) = 35/3 := by
sorry

end one_third_of_five_times_seven_l3437_343731


namespace least_months_to_triple_l3437_343705

/-- The initial borrowed amount in dollars -/
def initial_amount : ℝ := 1000

/-- The monthly interest rate as a decimal -/
def monthly_rate : ℝ := 0.06

/-- The function that calculates the amount owed after t months -/
def amount_owed (t : ℕ) : ℝ := initial_amount * (1 + monthly_rate) ^ t

/-- Theorem stating that 17 is the least number of months for which the amount owed exceeds three times the initial amount -/
theorem least_months_to_triple : 
  (∀ k : ℕ, k < 17 → amount_owed k ≤ 3 * initial_amount) ∧ 
  amount_owed 17 > 3 * initial_amount :=
sorry

end least_months_to_triple_l3437_343705


namespace bell_weight_ratio_l3437_343793

/-- Given three bells with specific weight relationships, prove the ratio of the third to second bell's weight --/
theorem bell_weight_ratio :
  ∀ (bell1 bell2 bell3 : ℝ),
  bell1 = 50 →
  bell2 = 2 * bell1 →
  bell1 + bell2 + bell3 = 550 →
  bell3 / bell2 = 4 :=
by
  sorry

end bell_weight_ratio_l3437_343793


namespace race_solution_l3437_343724

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  half_lap_time : ℝ

/-- Represents the race configuration -/
structure RaceConfig where
  track_length : ℝ
  α : Runner
  β : Runner
  initial_distance : ℝ
  symmetry_time : ℝ
  β_to_Q_time : ℝ
  α_to_finish_time : ℝ

/-- The theorem statement -/
theorem race_solution (config : RaceConfig) 
  (h1 : config.initial_distance = 16)
  (h2 : config.β_to_Q_time = 1 + 2/15)
  (h3 : config.α_to_finish_time = 13 + 13/15)
  (h4 : config.α.speed = config.track_length / (2 * config.α.half_lap_time))
  (h5 : config.β.speed = config.track_length / (2 * config.β.half_lap_time))
  (h6 : config.α.half_lap_time + config.symmetry_time + config.β_to_Q_time + config.α_to_finish_time = 2 * config.α.half_lap_time)
  (h7 : config.β.half_lap_time = config.α.half_lap_time + config.symmetry_time + config.β_to_Q_time)
  (h8 : config.track_length / 2 = config.α.speed * config.α.half_lap_time)
  (h9 : config.track_length / 2 = config.β.speed * config.β.half_lap_time)
  (h10 : config.α.speed * (config.β_to_Q_time + config.α_to_finish_time) = config.track_length / 2) :
  config.α.speed = 8.5 ∧ config.β.speed = 7.5 ∧ config.track_length = 272 := by
  sorry


end race_solution_l3437_343724


namespace exists_grade_to_move_l3437_343708

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem exists_grade_to_move :
  ∃ g ∈ group1,
    average (group1.filter (· ≠ g)) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end exists_grade_to_move_l3437_343708


namespace trig_expression_value_l3437_343742

theorem trig_expression_value (α : Real) (h : Real.tan (α / 2) = 4) :
  (6 * Real.sin α - 7 * Real.cos α + 1) / (8 * Real.sin α + 9 * Real.cos α - 1) = -85/44 := by
  sorry

end trig_expression_value_l3437_343742


namespace matrix_product_is_zero_l3437_343741

variable (a b c : ℝ)

def matrix1 : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2*c, -2*b],
    ![-2*c, 0, 2*a],
    ![2*b, -2*a, 0]]

def matrix2 : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a^2, 2*a*b, 2*a*c],
    ![2*a*b, b^2, 2*b*c],
    ![2*a*c, 2*b*c, c^2]]

theorem matrix_product_is_zero :
  matrix1 a b c * matrix2 a b c = 0 := by sorry

end matrix_product_is_zero_l3437_343741


namespace congruence_problem_l3437_343797

theorem congruence_problem (x : ℤ) : (3 * x + 7) % 16 = 2 → (2 * x + 11) % 16 = 13 := by
  sorry

end congruence_problem_l3437_343797


namespace derivative_f_at_negative_one_l3437_343775

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 1

-- State the theorem
theorem derivative_f_at_negative_one :
  deriv f (-1) = -1 := by sorry

end derivative_f_at_negative_one_l3437_343775


namespace linda_savings_proof_l3437_343713

def linda_savings (total : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : Prop :=
  furniture_fraction = 3/4 ∧ 
  (1 - furniture_fraction) * total = tv_cost ∧
  tv_cost = 450

theorem linda_savings_proof (total : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) :
  linda_savings total furniture_fraction tv_cost → total = 1800 := by
  sorry

end linda_savings_proof_l3437_343713


namespace lychee_ratio_l3437_343758

theorem lychee_ratio (total : ℕ) (remaining : ℕ) : 
  total = 500 → 
  remaining = 100 → 
  (total / 2 - remaining : ℚ) / (total / 2 : ℚ) = 3 / 5 := by
  sorry

end lychee_ratio_l3437_343758


namespace f_properties_l3437_343715

noncomputable def f (x : ℝ) := Real.sin x - Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), (f x)^2 = (f (x + p))^2 ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), (f x)^2 = (f (x + q))^2) → p ≤ q) ∧
  (∀ (x : ℝ), f (2*x - Real.pi/2) = Real.sqrt 2 * Real.sin (x/2)) ∧
  (∃ (M : ℝ), M = 1 + Real.sqrt 3 / 2 ∧
    ∀ (x : ℝ), (f x + Real.cos x) * (Real.sqrt 3 * Real.sin x + Real.cos x) ≤ M ∧
    ∃ (x₀ : ℝ), (f x₀ + Real.cos x₀) * (Real.sqrt 3 * Real.sin x₀ + Real.cos x₀) = M) :=
by sorry

end f_properties_l3437_343715


namespace largest_three_digit_congruence_l3437_343755

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), n = 991 ∧ 
  n < 1000 ∧ 
  n > 99 ∧
  55 * n ≡ 165 [MOD 260] ∧
  ∀ (m : ℕ), m < 1000 ∧ m > 99 ∧ 55 * m ≡ 165 [MOD 260] → m ≤ n :=
by sorry

end largest_three_digit_congruence_l3437_343755


namespace mans_speed_against_current_l3437_343770

/-- Calculates the man's speed against the current with wind resistance -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) (wind_resistance_factor : ℝ) (current_increase_factor : ℝ) : ℝ :=
  let speed_still_water := speed_with_current - current_speed
  let effective_speed_still_water := speed_still_water * (1 - wind_resistance_factor)
  let new_current_speed := current_speed * (1 + current_increase_factor)
  effective_speed_still_water - new_current_speed

/-- Theorem stating the man's speed against the current -/
theorem mans_speed_against_current :
  speed_against_current 22 5 0.15 0.1 = 8.95 := by
  sorry

end mans_speed_against_current_l3437_343770


namespace trigonometric_sum_zero_l3437_343735

theorem trigonometric_sum_zero (α : ℝ) : 
  Real.sin (2 * α - 3/2 * Real.pi) + Real.cos (2 * α - 8/3 * Real.pi) + Real.cos (2/3 * Real.pi + 2 * α) = 0 := by
  sorry

end trigonometric_sum_zero_l3437_343735


namespace max_intersections_three_lines_circle_l3437_343717

/-- A line in a 2D plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A circle in a 2D plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- The number of intersection points between a line and a circle -/
def line_circle_intersections (l : Line) (c : Circle) : ℕ := sorry

/-- The number of intersection points between two lines -/
def line_line_intersections (l1 l2 : Line) : ℕ := sorry

/-- Three distinct lines -/
def three_distinct_lines : Prop :=
  ∃ (l1 l2 l3 : Line), l1 ≠ l2 ∧ l1 ≠ l3 ∧ l2 ≠ l3

theorem max_intersections_three_lines_circle :
  ∀ (l1 l2 l3 : Line) (c : Circle),
  three_distinct_lines →
  (line_circle_intersections l1 c +
   line_circle_intersections l2 c +
   line_circle_intersections l3 c +
   line_line_intersections l1 l2 +
   line_line_intersections l1 l3 +
   line_line_intersections l2 l3) ≤ 9 ∧
  ∃ (l1' l2' l3' : Line) (c' : Circle),
    three_distinct_lines →
    (line_circle_intersections l1' c' +
     line_circle_intersections l2' c' +
     line_circle_intersections l3' c' +
     line_line_intersections l1' l2' +
     line_line_intersections l1' l3' +
     line_line_intersections l2' l3') = 9 :=
sorry

end max_intersections_three_lines_circle_l3437_343717


namespace sine_cosine_product_l3437_343784

theorem sine_cosine_product (α : Real) : 
  (∃ P : ℝ × ℝ, P.1 = Real.cos α ∧ P.2 = Real.sin α ∧ P.2 = -2 * P.1) →
  Real.sin α * Real.cos α = -2/5 := by
sorry

end sine_cosine_product_l3437_343784


namespace path_length_of_rotating_triangle_l3437_343753

/-- Represents a square with side length 4 inches -/
def Square := {s : ℝ // s = 4}

/-- Represents an equilateral triangle with side length 2 inches -/
def EquilateralTriangle := {t : ℝ // t = 2}

/-- Calculates the path length of vertex P during rotations -/
noncomputable def pathLength (square : Square) (triangle : EquilateralTriangle) : ℝ :=
  sorry

/-- Theorem stating the path length of vertex P -/
theorem path_length_of_rotating_triangle 
  (square : Square) 
  (triangle : EquilateralTriangle) : 
  pathLength square triangle = (40 * Real.pi) / 3 := by
  sorry

end path_length_of_rotating_triangle_l3437_343753


namespace expression_equals_sixteen_times_twelve_to_1001_l3437_343740

theorem expression_equals_sixteen_times_twelve_to_1001 :
  (3^1001 + 4^1002)^2 - (3^1001 - 4^1002)^2 = 16 * 12^1001 := by
  sorry

end expression_equals_sixteen_times_twelve_to_1001_l3437_343740


namespace odd_red_faces_count_l3437_343783

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with odd number of red faces -/
def count_odd_red_faces (dims : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the correct number of cubes with odd red faces -/
theorem odd_red_faces_count (block : BlockDimensions) 
  (h1 : block.length = 5)
  (h2 : block.width = 5)
  (h3 : block.height = 1) : 
  count_odd_red_faces block = 13 := by
  sorry

end odd_red_faces_count_l3437_343783


namespace complex_equation_solution_l3437_343702

theorem complex_equation_solution (x : ℝ) :
  (x - 2 * Complex.I) * Complex.I = 2 + Complex.I → x = 1 := by
  sorry

end complex_equation_solution_l3437_343702


namespace quadratic_congruence_solution_l3437_343720

theorem quadratic_congruence_solution (p : ℕ) (hp : Nat.Prime p) :
  ∃ n : ℤ, (6 * n^2 + 5 * n + 1) % p = 0 :=
sorry

end quadratic_congruence_solution_l3437_343720


namespace mans_usual_time_l3437_343723

/-- 
Given a man whose walking time increases by 24 minutes when his speed is reduced to 50% of his usual speed,
prove that his usual time to cover the distance is 24 minutes.
-/
theorem mans_usual_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0)
  (h3 : usual_speed / (0.5 * usual_speed) = (usual_time + 24) / usual_time) : 
  usual_time = 24 := by
sorry

end mans_usual_time_l3437_343723


namespace unique_two_digit_number_l3437_343739

theorem unique_two_digit_number (t : ℕ) : 
  (10 ≤ t ∧ t < 100) ∧ (13 * t ≡ 42 [ZMOD 100]) ↔ t = 34 :=
sorry

end unique_two_digit_number_l3437_343739


namespace positive_correlation_groups_l3437_343730

structure Variable where
  name : String

structure VariableGroup where
  var1 : Variable
  var2 : Variable

def has_positive_correlation (group : VariableGroup) : Prop :=
  sorry

def selling_price : Variable := ⟨"selling price"⟩
def sales_volume : Variable := ⟨"sales volume"⟩
def id_number : Variable := ⟨"ID number"⟩
def math_score : Variable := ⟨"math score"⟩
def breakfast_eaters : Variable := ⟨"number of people who eat breakfast daily"⟩
def stomach_diseases : Variable := ⟨"number of people with stomach diseases"⟩
def temperature : Variable := ⟨"temperature"⟩
def cold_drink_sales : Variable := ⟨"cold drink sales volume"⟩
def ebike_weight : Variable := ⟨"weight of an electric bicycle"⟩
def electricity_consumption : Variable := ⟨"electricity consumption per kilometer"⟩

def group1 : VariableGroup := ⟨selling_price, sales_volume⟩
def group2 : VariableGroup := ⟨id_number, math_score⟩
def group3 : VariableGroup := ⟨breakfast_eaters, stomach_diseases⟩
def group4 : VariableGroup := ⟨temperature, cold_drink_sales⟩
def group5 : VariableGroup := ⟨ebike_weight, electricity_consumption⟩

theorem positive_correlation_groups :
  has_positive_correlation group4 ∧ 
  has_positive_correlation group5 ∧
  ¬has_positive_correlation group1 ∧
  ¬has_positive_correlation group2 ∧
  ¬has_positive_correlation group3 :=
by sorry

end positive_correlation_groups_l3437_343730


namespace kirills_height_l3437_343712

/-- Proves that Kirill's height is 49 cm given the conditions -/
theorem kirills_height (brother_height : ℕ) 
  (h1 : brother_height - 14 + brother_height = 112) : 
  brother_height - 14 = 49 := by
  sorry

#check kirills_height

end kirills_height_l3437_343712


namespace hyperbola_vertex_distance_l3437_343729

/-- The distance between the vertices of a hyperbola with equation x²/144 - y²/49 = 1 is 24 -/
theorem hyperbola_vertex_distance : 
  ∀ (x y : ℝ), x^2/144 - y^2/49 = 1 → ∃ (d : ℝ), d = 24 ∧ d = 2 * (Real.sqrt 144) := by
  sorry

end hyperbola_vertex_distance_l3437_343729


namespace quadratic_trinomial_zero_discriminant_sum_l3437_343722

/-- A quadratic trinomial can be represented as the sum of two quadratic trinomials with zero discriminants -/
theorem quadratic_trinomial_zero_discriminant_sum (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (f g : ℝ → ℝ),
    (∀ x, a * x^2 + b * x + c = f x + g x) ∧
    (∃ (a₁ b₁ c₁ : ℝ), ∀ x, f x = a₁ * x^2 + b₁ * x + c₁) ∧
    (∃ (a₂ b₂ c₂ : ℝ), ∀ x, g x = a₂ * x^2 + b₂ * x + c₂) ∧
    (b₁^2 - 4 * a₁ * c₁ = 0) ∧
    (b₂^2 - 4 * a₂ * c₂ = 0) :=
by sorry

end quadratic_trinomial_zero_discriminant_sum_l3437_343722


namespace time_after_2500_minutes_l3437_343711

-- Define a custom datetime type
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

def startDateTime : DateTime :=
  { year := 2011, month := 1, day := 1, hour := 0, minute := 0 }

def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry  -- Implementation details omitted

theorem time_after_2500_minutes :
  addMinutes startDateTime 2500 =
    { year := 2011, month := 1, day := 2, hour := 17, minute := 40 } :=
by sorry

end time_after_2500_minutes_l3437_343711


namespace comparison_theorem_l3437_343738

theorem comparison_theorem (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : n ≥ 2) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end comparison_theorem_l3437_343738


namespace negation_equivalence_l3437_343749

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end negation_equivalence_l3437_343749


namespace expression_factorization_l3437_343725

theorem expression_factorization (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + 3 * a * b * c * (a - b) * (b - c) * (c - a) =
  (a - b) * (b - c) * (c - a) * (a + b + c + 3 * a * b * c) := by
  sorry

end expression_factorization_l3437_343725


namespace seven_minus_sqrt_five_floor_l3437_343768

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- State the theorem
theorem seven_minus_sqrt_five_floor : integerPart (7 - Real.sqrt 5) = 4 := by
  sorry

end seven_minus_sqrt_five_floor_l3437_343768


namespace negation_of_existence_negation_of_proposition_l3437_343787

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by sorry

end negation_of_existence_negation_of_proposition_l3437_343787


namespace derivative_ln_plus_reciprocal_l3437_343716

theorem derivative_ln_plus_reciprocal (x : ℝ) (hx : x > 0) :
  deriv (λ x => Real.log x + x⁻¹) x = (x - 1) / x^2 := by sorry

end derivative_ln_plus_reciprocal_l3437_343716


namespace mildred_blocks_l3437_343744

/-- The number of blocks Mildred found -/
def blocks_found (initial final : ℕ) : ℕ := final - initial

/-- Proof that Mildred found 84 blocks -/
theorem mildred_blocks : blocks_found 2 86 = 84 := by
  sorry

end mildred_blocks_l3437_343744


namespace tangent_sum_product_l3437_343732

theorem tangent_sum_product (α β γ : Real) (h : α + β + γ = 2 * Real.pi) :
  Real.tan (α / 2) + Real.tan (β / 2) + Real.tan (γ / 2) = 
  Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) := by
  sorry

end tangent_sum_product_l3437_343732


namespace A_intersect_B_is_singleton_one_l3437_343794

def A : Set ℝ := {0.1, 1, 10}

def B : Set ℝ := { y | ∃ x ∈ A, y = Real.log x / Real.log 10 }

theorem A_intersect_B_is_singleton_one : A ∩ B = {1} := by sorry

end A_intersect_B_is_singleton_one_l3437_343794


namespace factorization_sum_l3437_343789

theorem factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 12*x + 35 = (x + a)*(x + b)) → 
  (∀ x : ℝ, x^2 - 15*x + 56 = (x - b)*(x - c)) → 
  a + b + c = 20 := by
sorry

end factorization_sum_l3437_343789


namespace triangle_median_inequality_l3437_343710

/-- Given a triangle with sides a, b, and c, and s_c as the length of the median to side c,
    this theorem proves the inequality relating these measurements. -/
theorem triangle_median_inequality (a b c s_c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hs_c : 0 < s_c)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
    (h_median : 2 * s_c^2 = (2 * a^2 + 2 * b^2 - c^2) / 4) :
    (c^2 - (a - b)^2) / (2 * (a + b)) ≤ a + b - 2 * s_c ∧ 
    a + b - 2 * s_c < (c^2 + (a - b)^2) / (4 * s_c) := by
  sorry

end triangle_median_inequality_l3437_343710


namespace minimum_votes_to_win_l3437_343714

theorem minimum_votes_to_win (total_votes remaining_votes : ℕ)
  (a_votes b_votes c_votes : ℕ) (h1 : total_votes = 1500)
  (h2 : remaining_votes = 500) (h3 : a_votes + b_votes + c_votes = 1000)
  (h4 : a_votes = 350) (h5 : b_votes = 370) (h6 : c_votes = 280) :
  (∀ x : ℕ, x < 261 → 
    ∃ y : ℕ, y ≤ remaining_votes - x ∧ 
      a_votes + x ≤ b_votes + y) ∧
  (∃ z : ℕ, z = 261 ∧ 
    ∀ y : ℕ, y ≤ remaining_votes - z → 
      a_votes + z > b_votes + y) :=
by sorry

end minimum_votes_to_win_l3437_343714


namespace intersection_of_A_and_B_l3437_343780

def A : Set ℝ := {x | -5 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 9 < 0}

theorem intersection_of_A_and_B : A ∩ B = {x | -3 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l3437_343780


namespace gcd_13m_plus_4_7m_plus_2_max_l3437_343747

theorem gcd_13m_plus_4_7m_plus_2_max (m : ℕ+) : 
  (Nat.gcd (13 * m.val + 4) (7 * m.val + 2) ≤ 2) ∧ 
  (∃ m : ℕ+, Nat.gcd (13 * m.val + 4) (7 * m.val + 2) = 2) :=
by sorry

end gcd_13m_plus_4_7m_plus_2_max_l3437_343747


namespace fraction_proof_l3437_343701

theorem fraction_proof (w x y F : ℝ) 
  (h1 : 5 / w + F = 5 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  F = 10 := by
  sorry

end fraction_proof_l3437_343701


namespace percent_of_percent_equality_l3437_343752

theorem percent_of_percent_equality (y : ℝ) (h : y ≠ 0) :
  (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end percent_of_percent_equality_l3437_343752


namespace find_A_l3437_343746

theorem find_A : ∃ A : ℝ, (12 + 3) * (12 - A) = 120 ∧ A = 4 := by
  sorry

end find_A_l3437_343746


namespace smallest_fraction_between_l3437_343798

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 8 → q' ≥ q) →
  p + q = 21 :=
by sorry

end smallest_fraction_between_l3437_343798


namespace power_of_two_equation_l3437_343759

theorem power_of_two_equation (l : ℤ) : 
  2^2000 - 2^1999 - 3 * 2^1998 + 2^1997 = l * 2^1997 → l = -1 := by
  sorry

end power_of_two_equation_l3437_343759


namespace wheel_distance_l3437_343765

/-- The distance covered by a wheel with given radius and number of revolutions -/
theorem wheel_distance (radius : ℝ) (revolutions : ℕ) : 
  radius = Real.sqrt 157 → revolutions = 1000 → 
  2 * Real.pi * radius * revolutions = 78740 := by
  sorry

end wheel_distance_l3437_343765


namespace remainder_3042_div_98_l3437_343799

theorem remainder_3042_div_98 : 3042 % 98 = 4 := by
  sorry

end remainder_3042_div_98_l3437_343799


namespace june_maths_books_l3437_343719

/-- The number of maths books June bought -/
def num_maths_books : ℕ := sorry

/-- The total amount June has for school supplies -/
def total_amount : ℕ := 500

/-- The cost of each maths book -/
def maths_book_cost : ℕ := 20

/-- The cost of each science book -/
def science_book_cost : ℕ := 10

/-- The cost of each art book -/
def art_book_cost : ℕ := 20

/-- The amount spent on music books -/
def music_books_cost : ℕ := 160

/-- The total cost of all books -/
def total_cost : ℕ := 
  maths_book_cost * num_maths_books + 
  science_book_cost * (num_maths_books + 6) + 
  art_book_cost * (2 * num_maths_books) + 
  music_books_cost

theorem june_maths_books : 
  num_maths_books = 4 ∧ total_cost = total_amount :=
sorry

end june_maths_books_l3437_343719


namespace widget_carton_height_l3437_343788

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the packing configuration -/
structure PackingConfig where
  widgetsPerCarton : ℕ
  widgetsPerShippingBox : ℕ
  cartonDimensions : BoxDimensions
  shippingBoxDimensions : BoxDimensions

/-- The packing configuration for the Widget Factory -/
def widgetFactoryConfig : PackingConfig :=
  { widgetsPerCarton := 3
  , widgetsPerShippingBox := 300
  , cartonDimensions := 
    { width := 4
    , length := 4
    , height := 0  -- Unknown, to be determined
    }
  , shippingBoxDimensions := 
    { width := 20
    , length := 20
    , height := 20
    }
  }

/-- Theorem: The height of each carton in the Widget Factory configuration is 5 inches -/
theorem widget_carton_height (config : PackingConfig := widgetFactoryConfig) : 
  config.cartonDimensions.height = 5 := by
  sorry


end widget_carton_height_l3437_343788


namespace series_sum_equality_l3437_343782

/-- Given real numbers c and d satisfying a specific equation, 
    prove that the sum of a certain series equals a specific fraction. -/
theorem series_sum_equality (c d : ℝ) 
    (h : (c / d) / (1 - 1 / d) + (1 / d) / (1 - 1 / d) = 6) :
  c / (c + 2 * d) / (1 - 1 / (c + 2 * d)) = (6 * d - 7) / (8 * (d - 1)) := by
  sorry

end series_sum_equality_l3437_343782


namespace functional_equation_solutions_l3437_343762

def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x - f (x + y) = f (x^2 * f y + x)

theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x ≥ 0) →
  FunctionalEquation f →
  (∀ x, x > 0 → f x = 0) ∨ (∀ x, x > 0 → f x = 1 / x) :=
sorry

end functional_equation_solutions_l3437_343762


namespace tan_half_product_l3437_343754

theorem tan_half_product (a b : Real) :
  7 * (Real.sin a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 1) ∨ (Real.tan (a / 2) * Real.tan (b / 2) = -1) :=
by sorry

end tan_half_product_l3437_343754


namespace inverse_of_inverse_fourteen_l3437_343786

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x - 4

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x + 4) / 3

-- Theorem statement
theorem inverse_of_inverse_fourteen (h : ∀ x, g (g_inv x) = x) :
  g_inv (g_inv 14) = 10 / 3 := by
  sorry

end inverse_of_inverse_fourteen_l3437_343786


namespace game_lives_distribution_l3437_343733

/-- Given a game with initial players, players who quit, and total lives among remaining players,
    calculates the number of lives each remaining player has. -/
def lives_per_player (initial_players quitters total_lives : ℕ) : ℕ :=
  total_lives / (initial_players - quitters)

/-- Theorem stating that in a game with 13 initial players, 8 quitters, and 30 total lives,
    each remaining player has 6 lives. -/
theorem game_lives_distribution :
  lives_per_player 13 8 30 = 6 := by
  sorry


end game_lives_distribution_l3437_343733


namespace boys_average_score_l3437_343700

theorem boys_average_score (num_boys num_girls : ℕ) (girls_avg class_avg : ℝ) :
  num_boys = 12 →
  num_girls = 4 →
  girls_avg = 92 →
  class_avg = 86 →
  (num_boys * (class_avg * (num_boys + num_girls) - num_girls * girls_avg)) / (num_boys * (num_boys + num_girls)) = 84 :=
by sorry

end boys_average_score_l3437_343700


namespace inequality_proof_l3437_343792

theorem inequality_proof (x a b : ℝ) (h1 : x < a) (h2 : a < 0) (h3 : b = -a) :
  x^2 > b^2 ∧ b^2 > 0 := by
  sorry

end inequality_proof_l3437_343792


namespace problem_solution_l3437_343791

theorem problem_solution (m n : ℝ) 
  (h1 : m * n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 44 + n^4)
  (h4 : m^5 + 5 = 11) :
  m^9 + n = 38 := by
  sorry

end problem_solution_l3437_343791


namespace arrangement_equality_l3437_343777

theorem arrangement_equality (n : ℕ) (r₁ r₂ c₁ c₂ : ℕ) 
  (h₁ : n = r₁ * c₁)
  (h₂ : n = r₂ * c₂)
  (h₃ : n = 48)
  (h₄ : r₁ = 6)
  (h₅ : c₁ = 8)
  (h₆ : r₂ = 2)
  (h₇ : c₂ = 24) :
  Nat.factorial n = Nat.factorial n :=
by sorry

end arrangement_equality_l3437_343777


namespace denis_neighbors_l3437_343796

-- Define the students
inductive Student : Type
| Anya : Student
| Borya : Student
| Vera : Student
| Gena : Student
| Denis : Student

-- Define the line as a list of students
def Line := List Student

-- Define a function to check if two students are next to each other in the line
def next_to (s1 s2 : Student) (line : Line) : Prop :=
  ∃ i, (line.get? i = some s1 ∧ line.get? (i+1) = some s2) ∨
       (line.get? i = some s2 ∧ line.get? (i+1) = some s1)

-- Define the conditions
def valid_line (line : Line) : Prop :=
  (line.length = 5) ∧
  (line.head? = some Student.Borya) ∧
  (next_to Student.Vera Student.Anya line) ∧
  (¬ next_to Student.Vera Student.Gena line) ∧
  (¬ next_to Student.Anya Student.Borya line) ∧
  (¬ next_to Student.Anya Student.Gena line) ∧
  (¬ next_to Student.Borya Student.Gena line)

-- Theorem to prove
theorem denis_neighbors (line : Line) (h : valid_line line) :
  next_to Student.Denis Student.Anya line ∧ next_to Student.Denis Student.Gena line :=
sorry

end denis_neighbors_l3437_343796


namespace chip_sales_ratio_l3437_343772

/-- Represents the sales data for a convenience store's chip sales over a month. -/
structure ChipSales where
  total : ℕ
  first_week : ℕ
  third_week : ℕ
  fourth_week : ℕ

/-- Calculates the ratio of second week sales to first week sales. -/
def sales_ratio (sales : ChipSales) : ℚ :=
  let second_week := sales.total - sales.first_week - sales.third_week - sales.fourth_week
  (second_week : ℚ) / sales.first_week

/-- Theorem stating that given the specific sales conditions, the ratio of second week to first week sales is 3:1. -/
theorem chip_sales_ratio :
  ∀ (sales : ChipSales),
    sales.total = 100 ∧
    sales.first_week = 15 ∧
    sales.third_week = 20 ∧
    sales.fourth_week = 20 →
    sales_ratio sales = 3 := by
  sorry

end chip_sales_ratio_l3437_343772


namespace model_car_velocities_l3437_343766

/-- A model car on a closed circuit -/
structure ModelCar where
  circuit_length : ℕ
  uphill_length : ℕ
  flat_length : ℕ
  downhill_length : ℕ
  vs : ℕ  -- uphill velocity
  vp : ℕ  -- flat velocity
  vd : ℕ  -- downhill velocity

/-- The conditions of the problem -/
def satisfies_conditions (car : ModelCar) : Prop :=
  car.circuit_length = 600 ∧
  car.uphill_length = car.downhill_length ∧
  car.uphill_length + car.flat_length + car.downhill_length = car.circuit_length ∧
  car.vs < car.vp ∧ car.vp < car.vd ∧
  (car.uphill_length / car.vs + car.flat_length / car.vp + car.downhill_length / car.vd : ℚ) = 50

/-- The theorem to prove -/
theorem model_car_velocities (car : ModelCar) :
  satisfies_conditions car →
  ((car.vs = 7 ∧ car.vp = 12 ∧ car.vd = 42) ∨
   (car.vs = 8 ∧ car.vp = 12 ∧ car.vd = 24) ∨
   (car.vs = 9 ∧ car.vp = 12 ∧ car.vd = 18) ∨
   (car.vs = 10 ∧ car.vp = 12 ∧ car.vd = 15)) :=
by sorry

end model_car_velocities_l3437_343766
