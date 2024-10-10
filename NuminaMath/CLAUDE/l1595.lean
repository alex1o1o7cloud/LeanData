import Mathlib

namespace rectangle_width_l1595_159594

/-- Given a rectangle with length 5.4 cm and area 48.6 cm², prove its width is 9 cm -/
theorem rectangle_width (length : ℝ) (area : ℝ) (h1 : length = 5.4) (h2 : area = 48.6) :
  area / length = 9 := by
  sorry

end rectangle_width_l1595_159594


namespace discriminant_nonnegative_m_value_when_root_difference_is_two_l1595_159587

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - 4*m*x + 3*m^2

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (-4*m)^2 - 4*1*(3*m^2)

-- Theorem 1: The discriminant is always non-negative
theorem discriminant_nonnegative (m : ℝ) : discriminant m ≥ 0 := by
  sorry

-- Theorem 2: When m > 0 and the difference between roots is 2, m = 1
theorem m_value_when_root_difference_is_two (m : ℝ) 
  (h1 : m > 0) 
  (h2 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
                     quadratic_equation m x1 = 0 ∧ 
                     quadratic_equation m x2 = 0 ∧ 
                     x1 - x2 = 2) : 
  m = 1 := by
  sorry

end discriminant_nonnegative_m_value_when_root_difference_is_two_l1595_159587


namespace power_of_power_equals_power_product_l1595_159578

theorem power_of_power_equals_power_product (x : ℝ) : (x^2)^4 = x^8 := by
  sorry

end power_of_power_equals_power_product_l1595_159578


namespace xy_squared_minus_x_squared_y_equals_negative_two_sqrt_two_l1595_159533

theorem xy_squared_minus_x_squared_y_equals_negative_two_sqrt_two :
  ∀ x y : ℝ,
  x = Real.sqrt 3 + Real.sqrt 2 →
  y = Real.sqrt 3 - Real.sqrt 2 →
  x * y^2 - x^2 * y = -2 * Real.sqrt 2 := by
sorry

end xy_squared_minus_x_squared_y_equals_negative_two_sqrt_two_l1595_159533


namespace exp_inequality_equivalence_l1595_159596

theorem exp_inequality_equivalence (x : ℝ) : 1 < Real.exp x ∧ Real.exp x < 2 ↔ 0 < x ∧ x < Real.log 2 := by
  sorry

end exp_inequality_equivalence_l1595_159596


namespace quadratic_passes_through_point_l1595_159592

/-- A quadratic function passing through (-1, 0) given a - b + c = 0 -/
theorem quadratic_passes_through_point
  (a b c : ℝ) -- Coefficients of the quadratic function
  (h : a - b + c = 0) -- Given condition
  : let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c -- Definition of the quadratic function
    f (-1) = 0 := by sorry

end quadratic_passes_through_point_l1595_159592


namespace total_amount_is_105_l1595_159500

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : ShareDistribution) : Prop :=
  s.y = 0.45 * s.x ∧ s.z = 0.30 * s.x ∧ s.y = 27

/-- The theorem to prove -/
theorem total_amount_is_105 (s : ShareDistribution) :
  problem_conditions s → s.x + s.y + s.z = 105 := by sorry

end total_amount_is_105_l1595_159500


namespace simplify_polynomial_l1595_159581

theorem simplify_polynomial (r : ℝ) : (2*r^2 + 5*r - 7) - (r^2 + 9*r - 3) = r^2 - 4*r - 4 := by
  sorry

end simplify_polynomial_l1595_159581


namespace power_five_minus_self_divisible_by_five_l1595_159510

theorem power_five_minus_self_divisible_by_five (a : ℤ) : ∃ k : ℤ, a^5 - a = 5 * k := by
  sorry

end power_five_minus_self_divisible_by_five_l1595_159510


namespace sum_60_is_negative_120_l1595_159571

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_20 : (20 : ℚ) / 2 * (2 * a + 19 * d) = 200
  sum_50 : (50 : ℚ) / 2 * (2 * a + 49 * d) = 50

/-- The sum of the first 60 terms of the arithmetic progression is -120 -/
theorem sum_60_is_negative_120 (ap : ArithmeticProgression) :
  (60 : ℚ) / 2 * (2 * ap.a + 59 * ap.d) = -120 := by
  sorry

end sum_60_is_negative_120_l1595_159571


namespace special_factorization_of_630_l1595_159521

theorem special_factorization_of_630 : ∃ (a b x y z : ℕ), 
  (a + 1 = b) ∧ 
  (x + 1 = y) ∧ 
  (y + 1 = z) ∧ 
  (a * b = 630) ∧ 
  (x * y * z = 630) ∧ 
  (a + b + x + y + z = 75) := by
  sorry

end special_factorization_of_630_l1595_159521


namespace euler_conjecture_counterexample_l1595_159554

theorem euler_conjecture_counterexample : 133^5 + 110^5 + 84^5 + 27^5 = 144^5 := by
  sorry

end euler_conjecture_counterexample_l1595_159554


namespace bob_spending_theorem_l1595_159535

def spending_problem (initial_amount : ℚ) : ℚ :=
  let after_monday := initial_amount / 2
  let after_tuesday := after_monday - (after_monday / 5)
  let after_wednesday := after_tuesday - (after_tuesday * 3 / 8)
  after_wednesday

theorem bob_spending_theorem :
  spending_problem 80 = 20 := by sorry

end bob_spending_theorem_l1595_159535


namespace distance_between_points_on_line_l1595_159577

/-- Given a line with equation 2x - 3y + 6 = 0 and two points (p, q) and (r, s) on this line,
    the distance between these points is (√13/3)|r - p| -/
theorem distance_between_points_on_line (p r : ℝ) :
  let q := (2*p + 6)/3
  let s := (2*r + 6)/3
  (2*p - 3*q + 6 = 0) →
  (2*r - 3*s + 6 = 0) →
  Real.sqrt ((r - p)^2 + (s - q)^2) = (Real.sqrt 13 / 3) * |r - p| := by
sorry

end distance_between_points_on_line_l1595_159577


namespace simplify_expression_l1595_159547

theorem simplify_expression (w : ℝ) :
  2 * w + 3 - 4 * w - 5 + 6 * w + 7 - 8 * w - 9 = -4 * w - 4 := by
  sorry

end simplify_expression_l1595_159547


namespace expression_simplification_l1595_159569

theorem expression_simplification : (((3 + 6 + 9 + 12) / 3) + ((3 * 4 - 6) / 2)) = 13 := by
  sorry

end expression_simplification_l1595_159569


namespace complement_A_eq_three_four_l1595_159544

-- Define the set A
def A : Set ℕ := {x : ℕ | x^2 - 7*x + 10 ≥ 0}

-- Define the complement of A with respect to ℕ
def complement_A : Set ℕ := {x : ℕ | x ∉ A}

-- Theorem statement
theorem complement_A_eq_three_four : complement_A = {3, 4} := by sorry

end complement_A_eq_three_four_l1595_159544


namespace inequality_proof_l1595_159551

theorem inequality_proof (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end inequality_proof_l1595_159551


namespace n_power_37_minus_n_divisibility_l1595_159553

theorem n_power_37_minus_n_divisibility (n : ℤ) : 
  (∃ k : ℤ, n^37 - n = 91 * k) ∧ 
  (∃ m : ℤ, n^37 - n = 3276 * m) ∧
  (∀ l : ℤ, l > 3276 → ∃ p : ℤ, ¬ (∃ q : ℤ, p^37 - p = l * q)) :=
by sorry

end n_power_37_minus_n_divisibility_l1595_159553


namespace isosceles_right_triangle_area_l1595_159507

/-- An isosceles right triangle with an inscribed circle -/
structure IsoscelesRightTriangle where
  -- The length of a leg of the triangle
  leg : ℝ
  -- The center of the inscribed circle
  center : ℝ × ℝ
  -- The radius of the inscribed circle
  radius : ℝ
  -- The area of the inscribed circle is 9π
  circle_area : radius^2 * Real.pi = 9 * Real.pi

/-- The area of an isosceles right triangle with an inscribed circle of area 9π is 36 -/
theorem isosceles_right_triangle_area 
  (triangle : IsoscelesRightTriangle) : triangle.leg^2 = 36 := by
  sorry

end isosceles_right_triangle_area_l1595_159507


namespace root_sum_theorem_l1595_159538

theorem root_sum_theorem (a b : ℝ) : 
  (Complex.I * Real.sqrt 7 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 7 + 2) + b = 0 → 
  a + b = 39 := by
  sorry

end root_sum_theorem_l1595_159538


namespace percent_of_self_equal_sixteen_l1595_159597

theorem percent_of_self_equal_sixteen (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 16) : x = 40 := by
  sorry

end percent_of_self_equal_sixteen_l1595_159597


namespace pencil_difference_l1595_159570

theorem pencil_difference (price : ℚ) (jamar_count sharona_count : ℕ) : 
  price > 0.01 →
  price * jamar_count = 216/100 →
  price * sharona_count = 272/100 →
  sharona_count - jamar_count = 7 := by
sorry

end pencil_difference_l1595_159570


namespace fewest_tiles_needed_l1595_159572

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of a single tile -/
def tileDimensions : Dimensions := ⟨6, 2⟩

/-- The dimensions of the rectangular region in feet -/
def regionDimensionsFeet : Dimensions := ⟨3, 6⟩

/-- The dimensions of the rectangular region in inches -/
def regionDimensionsInches : Dimensions :=
  ⟨feetToInches regionDimensionsFeet.length, feetToInches regionDimensionsFeet.width⟩

/-- Calculates the number of tiles needed to cover a given area -/
def tilesNeeded (regionArea tileArea : ℕ) : ℕ :=
  (regionArea + tileArea - 1) / tileArea

theorem fewest_tiles_needed :
  tilesNeeded (area regionDimensionsInches) (area tileDimensions) = 216 := by
  sorry

end fewest_tiles_needed_l1595_159572


namespace keith_missed_four_games_l1595_159562

/-- The number of football games Keith missed, given the total number of games and the number of games he attended. -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Theorem stating that Keith missed 4 football games. -/
theorem keith_missed_four_games :
  let total_games : ℕ := 8
  let attended_games : ℕ := 4
  games_missed total_games attended_games = 4 := by
sorry

end keith_missed_four_games_l1595_159562


namespace function_decomposition_l1595_159539

open Function Real

theorem function_decomposition (f : ℝ → ℝ) : 
  ∃ (g h : ℝ → ℝ), 
    (∀ x, g (-x) = g x) ∧ 
    (∀ x, h (-x) = -h x) ∧ 
    (∀ x, f x = g x + h x) := by
  sorry

end function_decomposition_l1595_159539


namespace circle_center_distance_l1595_159520

/-- The distance between the center of the circle x^2 + y^2 = 4x + 6y + 3 and the point (5, -2) is √34 -/
theorem circle_center_distance :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 = 4*x + 6*y + 3
  let center : ℝ × ℝ := (2, 3)
  let point : ℝ × ℝ := (5, -2)
  (∃ x y, circle_eq x y) →
  Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) = Real.sqrt 34 := by
sorry

end circle_center_distance_l1595_159520


namespace rosy_age_l1595_159591

/-- Proves that Rosy's current age is 12 years, given the conditions about David's age -/
theorem rosy_age (rosy_age david_age : ℕ) 
  (h1 : david_age = rosy_age + 18)
  (h2 : david_age + 6 = 2 * (rosy_age + 6)) : 
  rosy_age = 12 := by
  sorry

#check rosy_age

end rosy_age_l1595_159591


namespace circle_diameter_ratio_l1595_159558

theorem circle_diameter_ratio (R S : Real) (harea : R^2 = 0.36 * S^2) :
  R = 0.6 * S := by
  sorry

end circle_diameter_ratio_l1595_159558


namespace speed_of_boat_in_still_water_l1595_159506

/-- Theorem: Speed of boat in still water
Given:
- The rate of the current is 15 km/hr
- The boat traveled downstream for 25 minutes
- The boat covered a distance of 33.33 km downstream

Prove that the speed of the boat in still water is approximately 64.992 km/hr
-/
theorem speed_of_boat_in_still_water
  (current_speed : ℝ)
  (travel_time : ℝ)
  (distance_covered : ℝ)
  (h1 : current_speed = 15)
  (h2 : travel_time = 25 / 60)
  (h3 : distance_covered = 33.33) :
  ∃ (boat_speed : ℝ), abs (boat_speed - 64.992) < 0.001 ∧
    distance_covered = (boat_speed + current_speed) * travel_time :=
by sorry

end speed_of_boat_in_still_water_l1595_159506


namespace segment_properties_l1595_159536

/-- Given two points A(1, 2) and B(9, 14), prove the distance between them and their midpoint. -/
theorem segment_properties : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (9, 14)
  let distance := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (distance = 16) ∧ (midpoint = (5, 8)) := by
  sorry

end segment_properties_l1595_159536


namespace congruent_face_tetrahedron_volume_l1595_159566

/-- A tetrahedron with congruent triangular faces -/
structure CongruentFaceTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  triangle_inequality_ab : a < b + c
  triangle_inequality_bc : b < a + c
  triangle_inequality_ca : c < a + b

/-- The volume of a tetrahedron with congruent triangular faces -/
noncomputable def volume (t : CongruentFaceTetrahedron) : ℝ :=
  (1 / (6 * Real.sqrt 2)) * Real.sqrt ((-t.a^2 + t.b^2 + t.c^2) * (t.a^2 - t.b^2 + t.c^2) * (t.a^2 + t.b^2 - t.c^2))

/-- Theorem: The volume of a tetrahedron with congruent triangular faces is given by the formula -/
theorem congruent_face_tetrahedron_volume (t : CongruentFaceTetrahedron) :
  ∃ V, V = volume t ∧ V > 0 := by
  sorry

end congruent_face_tetrahedron_volume_l1595_159566


namespace expression_simplification_l1595_159543

theorem expression_simplification (y : ℝ) (h : y ≠ 0) :
  (20 * y^3) * (7 * y^2) * (1 / (2*y)^3) = 17.5 * y^2 := by
  sorry

end expression_simplification_l1595_159543


namespace sequence_convergence_comparison_l1595_159582

theorem sequence_convergence_comparison
  (k : ℝ) (h_k : 0 < k ∧ k < 1/2)
  (a₀ b₀ : ℝ) (h_a₀ : 0 < a₀ ∧ a₀ < 1) (h_b₀ : 0 < b₀ ∧ b₀ < 1)
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_a : ∀ n, a (n + 1) = (a n + 1) / 2)
  (h_b : ∀ n, b (n + 1) = (b n) ^ k) :
  ∃ N, ∀ n ≥ N, a n < b n :=
sorry

end sequence_convergence_comparison_l1595_159582


namespace matrix_product_l1595_159567

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 4; 3, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, -7; 2, 3]

theorem matrix_product :
  A * B = !![8, 5; -4, -27] := by sorry

end matrix_product_l1595_159567


namespace square_divisibility_l1595_159527

theorem square_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : ∃ k : ℕ, a^2 + b^2 = k * (a * b + 1)) : 
  ∃ n : ℕ, (a^2 + b^2) / (a * b + 1) = n^2 := by
sorry

end square_divisibility_l1595_159527


namespace john_account_balance_l1595_159540

/-- Calculates the final balance after a deposit and withdrawal -/
def final_balance (initial_balance deposit withdrawal : ℚ) : ℚ :=
  initial_balance + deposit - withdrawal

/-- Theorem: Given the specified initial balance, deposit, and withdrawal,
    the final balance is 43.8 -/
theorem john_account_balance :
  final_balance 45.7 18.6 20.5 = 43.8 := by
  sorry

end john_account_balance_l1595_159540


namespace vasechkin_result_l1595_159518

def petrov_operation (x : ℚ) : ℚ := (x / 2) * 7 - 1001

def vasechkin_operation (x : ℚ) : ℚ := (x / 8)^2 - 1001

theorem vasechkin_result :
  ∃ x : ℚ, (∃ p : ℕ, Nat.Prime p ∧ petrov_operation x = ↑p) →
  vasechkin_operation x = 295 :=
sorry

end vasechkin_result_l1595_159518


namespace sinusoidal_amplitude_l1595_159517

/-- Given a sinusoidal function y = a * sin(bx + c) + d with positive constants a, b, c, and d,
    if the function oscillates between 5 and -3, then a = 4 -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) :
  a = 4 := by
  sorry

end sinusoidal_amplitude_l1595_159517


namespace right_triangle_side_length_l1595_159519

theorem right_triangle_side_length 
  (P Q R : ℝ × ℝ) 
  (is_right_triangle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0) 
  (tan_R : (R.2 - P.2) / (R.1 - P.1) = 4/3) 
  (PQ_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 3) : 
  Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 5 := by sorry

end right_triangle_side_length_l1595_159519


namespace range_of_m_for_false_proposition_l1595_159557

theorem range_of_m_for_false_proposition : 
  (∃ m : ℝ, ¬(∀ x : ℝ, x^2 - 2*x - m ≥ 0)) ↔ 
  (∃ m : ℝ, m > -1) :=
sorry

end range_of_m_for_false_proposition_l1595_159557


namespace midpoint_trajectory_l1595_159573

/-- The trajectory of the midpoint of a line segment with one end fixed and the other on a circle -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ a b : ℝ, (a^2 + b^2 = 16) ∧ 
              (x = (10 + a) / 2) ∧ 
              (y = b / 2)) → 
  (x - 5)^2 + y^2 = 4 := by
sorry

end midpoint_trajectory_l1595_159573


namespace first_group_size_l1595_159524

/-- The number of men in the first group -/
def M : ℕ := 42

/-- The number of days the first group takes to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def men_second_group : ℕ := 27

/-- The number of days the second group takes to complete the work -/
def days_second_group : ℕ := 28

/-- The work done by a group is inversely proportional to the number of days they take -/
axiom work_inverse_proportion (men days : ℕ) : men * days = men_second_group * days_second_group

theorem first_group_size : M = 42 := by
  sorry

end first_group_size_l1595_159524


namespace inscribed_squares_ratio_l1595_159580

/-- Right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13
  right_angle : a^2 + b^2 = c^2

/-- Square inscribed in the first triangle with vertex at right angle -/
def square_at_vertex (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x ≤ t.b ∧ x / t.a = x / t.b

/-- Square inscribed in the second triangle with side on hypotenuse -/
def square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.c ∧ t.a * y / t.c = y

/-- The main theorem -/
theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ) 
    (hx : square_at_vertex t1 x) (hy : square_on_hypotenuse t2 y) : 
    x / y = 144 / 221 := by
  sorry

end inscribed_squares_ratio_l1595_159580


namespace polygon_sides_diagonals_l1595_159511

theorem polygon_sides_diagonals : ∃ (n : ℕ), n > 2 ∧ 3 * n * (n * (n - 3)) = 300 := by
  use 10
  sorry

end polygon_sides_diagonals_l1595_159511


namespace class_average_problem_l1595_159513

theorem class_average_problem (group1_percent : Real) (group1_score : Real)
                              (group2_percent : Real)
                              (group3_percent : Real) (group3_score : Real)
                              (total_average : Real) :
  group1_percent = 0.25 →
  group1_score = 0.8 →
  group2_percent = 0.5 →
  group3_percent = 0.25 →
  group3_score = 0.9 →
  total_average = 0.75 →
  group1_percent + group2_percent + group3_percent = 1 →
  group1_percent * group1_score + group2_percent * (65 / 100) + group3_percent * group3_score = total_average :=
by
  sorry


end class_average_problem_l1595_159513


namespace rectangular_field_area_l1595_159555

/-- Calculates the area of a rectangular field given specific fencing conditions -/
theorem rectangular_field_area (uncovered_side : ℝ) (total_fencing : ℝ) : uncovered_side = 20 → total_fencing = 76 → uncovered_side * ((total_fencing - uncovered_side) / 2) = 560 := by
  sorry

end rectangular_field_area_l1595_159555


namespace curve_single_intersection_l1595_159514

/-- The curve (x+2y+a)(x^2-y^2)=0 intersects at a single point if and only if a = 0 -/
theorem curve_single_intersection (a : ℝ) : 
  (∃! p : ℝ × ℝ, (p.1 + 2 * p.2 + a) * (p.1^2 - p.2^2) = 0) ↔ a = 0 := by
  sorry

end curve_single_intersection_l1595_159514


namespace strategy_game_cost_l1595_159586

/-- The cost of Tom's video game purchases -/
def total_cost : ℚ := 35.52

/-- The cost of the football game -/
def football_cost : ℚ := 14.02

/-- The cost of the Batman game -/
def batman_cost : ℚ := 12.04

/-- The cost of the strategy game -/
def strategy_cost : ℚ := total_cost - football_cost - batman_cost

theorem strategy_game_cost :
  strategy_cost = 9.46 := by sorry

end strategy_game_cost_l1595_159586


namespace cone_slant_height_l1595_159516

/-- Given a cone with base radius 3 cm and curved surface area 141.3716694115407 cm²,
    prove that its slant height is 15 cm. -/
theorem cone_slant_height (r : ℝ) (csa : ℝ) (h1 : r = 3) (h2 : csa = 141.3716694115407) :
  csa / (Real.pi * r) = 15 := by
  sorry

end cone_slant_height_l1595_159516


namespace geometric_sequence_problem_l1595_159512

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  isGeometricSequence a →
  (a 4 + a 8 = -11) →
  (a 4 * a 8 = 9) →
  a 6 = -3 := by
sorry

end geometric_sequence_problem_l1595_159512


namespace eulers_formula_l1595_159523

/-- Euler's formula -/
theorem eulers_formula (a b : ℝ) :
  Complex.exp (a + Complex.I * b) = Complex.exp a * (Complex.cos b + Complex.I * Complex.sin b) := by
  sorry

end eulers_formula_l1595_159523


namespace fraction_multiplication_l1595_159537

theorem fraction_multiplication : (1 : ℚ) / 3 * (3 : ℚ) / 5 * (5 : ℚ) / 6 = (1 : ℚ) / 6 := by
  sorry

end fraction_multiplication_l1595_159537


namespace max_hubs_is_six_l1595_159552

/-- A structure representing a state with cities and roads --/
structure State where
  num_cities : ℕ
  num_roads : ℕ
  num_hubs : ℕ

/-- Definition of a valid state configuration --/
def is_valid_state (s : State) : Prop :=
  s.num_cities = 10 ∧
  s.num_roads = 40 ∧
  s.num_hubs ≤ s.num_cities ∧
  s.num_hubs * (s.num_hubs - 1) / 2 + s.num_hubs * (s.num_cities - s.num_hubs) ≤ s.num_roads

/-- Theorem stating that the maximum number of hubs in a valid state is 6 --/
theorem max_hubs_is_six :
  ∀ s : State, is_valid_state s → s.num_hubs ≤ 6 :=
by sorry

end max_hubs_is_six_l1595_159552


namespace unique_three_prime_product_l1595_159593

def isPrime (n : ℕ) : Prop := Nat.Prime n

def primeFactors (n : ℕ) : List ℕ := sorry

theorem unique_three_prime_product : 
  ∃! n : ℕ, 
    ∃ p1 p2 p3 : ℕ, 
      isPrime p1 ∧ isPrime p2 ∧ isPrime p3 ∧
      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
      n = p1 * p2 * p3 ∧
      p1 + p2 + p3 = (primeFactors 9271).sum := by sorry

end unique_three_prime_product_l1595_159593


namespace shaded_percentage_of_square_l1595_159548

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  sideLength : ℝ
  bottomLeft : Point

/-- Represents a shaded region -/
structure ShadedRegion where
  bottomLeft : Point
  topRight : Point

/-- Calculate the area of a square -/
def squareArea (s : Square) : ℝ := s.sideLength * s.sideLength

/-- Calculate the area of a shaded region -/
def shadedRegionArea (r : ShadedRegion) : ℝ :=
  (r.topRight.x - r.bottomLeft.x) * (r.topRight.y - r.bottomLeft.y)

/-- The main theorem -/
theorem shaded_percentage_of_square (EFGH : Square)
  (region1 region2 region3 : ShadedRegion) :
  EFGH.sideLength = 7 →
  EFGH.bottomLeft = ⟨0, 0⟩ →
  region1 = ⟨⟨0, 0⟩, ⟨1, 1⟩⟩ →
  region2 = ⟨⟨3, 0⟩, ⟨5, 5⟩⟩ →
  region3 = ⟨⟨6, 0⟩, ⟨7, 7⟩⟩ →
  (shadedRegionArea region1 + shadedRegionArea region2 + shadedRegionArea region3) /
    squareArea EFGH * 100 = 14 / 49 * 100 := by
  sorry

end shaded_percentage_of_square_l1595_159548


namespace f_properties_l1595_159501

noncomputable def f (x : ℝ) := Real.exp x * Real.cos x - x

theorem f_properties :
  let a := 0
  let b := Real.pi / 2
  ∃ (tangent_line : ℝ → ℝ),
    (∀ x, tangent_line x = 1) ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f a) ∧
    (∀ x ∈ Set.Icc a b, f b ≤ f x) ∧
    f a = 1 ∧
    f b = -Real.pi / 2 := by
  sorry

end f_properties_l1595_159501


namespace amiths_age_l1595_159515

theorem amiths_age (a d : ℕ) : 
  (a - 5 = 3 * (d - 5)) → 
  (a + 10 = 2 * (d + 10)) → 
  a = 50 := by
sorry

end amiths_age_l1595_159515


namespace flower_difference_l1595_159502

def white_flowers : ℕ := 555
def red_flowers : ℕ := 347
def blue_flowers : ℕ := 498
def yellow_flowers : ℕ := 425

theorem flower_difference : 
  (red_flowers + blue_flowers + yellow_flowers) - white_flowers = 715 := by
  sorry

end flower_difference_l1595_159502


namespace max_value_implies_m_l1595_159575

-- Define the function f(x) = x^2 - 2x + m
def f (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Define the interval [0, 3]
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem max_value_implies_m (m : ℝ) :
  (∀ x ∈ interval, f x m ≤ 1) ∧
  (∃ x ∈ interval, f x m = 1) →
  m = -2 :=
sorry

end max_value_implies_m_l1595_159575


namespace convex_polygon_27_sides_diagonals_l1595_159541

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 27 sides has 324 diagonals -/
theorem convex_polygon_27_sides_diagonals :
  num_diagonals 27 = 324 := by sorry

end convex_polygon_27_sides_diagonals_l1595_159541


namespace equal_area_segment_property_l1595_159563

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  base_difference : longer_base = shorter_base + 150
  midpoint_ratio : ℝ
  midpoint_area_ratio : midpoint_ratio = 3 / 4

/-- The length of the segment that divides the trapezoid into two equal-area regions -/
def equal_area_segment (t : Trapezoid) : ℝ :=
  t.shorter_base + 150

/-- Theorem stating the property of the equal area segment -/
theorem equal_area_segment_property (t : Trapezoid) :
  ⌊(equal_area_segment t)^3 / 1000⌋ = 142 := by
  sorry

#check equal_area_segment_property

end equal_area_segment_property_l1595_159563


namespace donation_problem_l1595_159589

/-- Calculates the total number of articles of clothing donated given the initial number of items set aside by Adam and the number of friends donating. -/
def total_donated_clothing (adam_pants : ℕ) (adam_jumpers : ℕ) (adam_pajama_sets : ℕ) (adam_tshirts : ℕ) (num_friends : ℕ) : ℕ := 
  let adam_initial := adam_pants + adam_jumpers + (2 * adam_pajama_sets) + adam_tshirts
  let friends_donation := num_friends * adam_initial
  let adam_final := adam_initial / 2
  adam_final + friends_donation

/-- Theorem stating that the total number of articles of clothing donated is 126, given the specific conditions of the problem. -/
theorem donation_problem : total_donated_clothing 4 4 4 20 3 = 126 := by
  sorry

end donation_problem_l1595_159589


namespace salt_calculation_l1595_159561

/-- Calculates the amount of salt Jack will have after water evaporation -/
def salt_after_evaporation (
  water_volume_day1 : ℝ)
  (water_volume_day2 : ℝ)
  (salt_concentration_day1 : ℝ)
  (salt_concentration_day2 : ℝ)
  (evaporation_rate_day1 : ℝ)
  (evaporation_rate_day2 : ℝ) : ℝ :=
  ((water_volume_day1 * salt_concentration_day1 +
    water_volume_day2 * salt_concentration_day2) * 1000)

theorem salt_calculation :
  salt_after_evaporation 4 4 0.18 0.22 0.30 0.40 = 1600 := by
  sorry

end salt_calculation_l1595_159561


namespace walter_work_hours_l1595_159576

/-- Walter's work schedule and earnings -/
structure WorkSchedule where
  days_per_week : ℕ
  hourly_rate : ℚ
  allocation_ratio : ℚ
  school_allocation : ℚ

/-- Calculate the daily work hours given a work schedule -/
def daily_work_hours (schedule : WorkSchedule) : ℚ :=
  schedule.school_allocation / (schedule.days_per_week * schedule.hourly_rate * schedule.allocation_ratio)

/-- Theorem: Walter works 4 hours a day -/
theorem walter_work_hours : 
  let walter_schedule : WorkSchedule := {
    days_per_week := 5,
    hourly_rate := 5,
    allocation_ratio := 3/4,
    school_allocation := 75
  }
  daily_work_hours walter_schedule = 4 := by
  sorry

end walter_work_hours_l1595_159576


namespace swimmers_pass_21_times_l1595_159530

/-- Represents the swimming pool setup and swimmer characteristics --/
structure SwimmingSetup where
  poolLength : ℝ
  swimmerASpeed : ℝ
  swimmerBSpeed : ℝ
  totalTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def calculatePassings (setup : SwimmingSetup) : ℕ :=
  sorry

/-- Theorem stating that the swimmers pass each other 21 times --/
theorem swimmers_pass_21_times :
  let setup : SwimmingSetup := {
    poolLength := 120,
    swimmerASpeed := 4,
    swimmerBSpeed := 3,
    totalTime := 15 * 60  -- 15 minutes in seconds
  }
  calculatePassings setup = 21 := by
  sorry

end swimmers_pass_21_times_l1595_159530


namespace value_after_two_years_approximation_l1595_159525

/-- Calculates the value after n years given an initial value and annual increase rate -/
def value_after_n_years (initial_value : ℝ) (increase_rate : ℝ) (n : ℕ) : ℝ :=
  initial_value * (1 + increase_rate) ^ n

/-- The problem statement -/
theorem value_after_two_years_approximation :
  let initial_value : ℝ := 64000
  let increase_rate : ℝ := 1 / 9
  let years : ℕ := 2
  let final_value := value_after_n_years initial_value increase_rate years
  abs (final_value - 79012.36) < 0.01 := by
  sorry

end value_after_two_years_approximation_l1595_159525


namespace ellas_raise_percentage_l1595_159564

/-- Calculates the percentage raise given the conditions of Ella's babysitting earnings and expenses. -/
theorem ellas_raise_percentage 
  (video_game_percentage : Real) 
  (last_year_video_game_expense : Real) 
  (new_salary : Real) 
  (h1 : video_game_percentage = 0.40)
  (h2 : last_year_video_game_expense = 100)
  (h3 : new_salary = 275) : 
  (new_salary - (last_year_video_game_expense / video_game_percentage)) / (last_year_video_game_expense / video_game_percentage) * 100 = 10 := by
  sorry

end ellas_raise_percentage_l1595_159564


namespace triangle_angle_A_l1595_159504

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_A (t : Triangle) : 
  t.a = 4 * Real.sqrt 3 → 
  t.c = 12 → 
  t.C = π / 3 → 
  t.A = π / 6 := by
  sorry


end triangle_angle_A_l1595_159504


namespace complex_roots_isosceles_triangle_l1595_159585

theorem complex_roots_isosceles_triangle (a b z₁ z₂ : ℂ) :
  z₁^2 + a*z₁ + b = 0 →
  z₂^2 + a*z₂ + b = 0 →
  z₂ = Complex.exp (Real.pi * Complex.I / 4) * z₁ →
  a^2 / b = 4 + 4 * Real.sqrt 2 :=
by sorry

end complex_roots_isosceles_triangle_l1595_159585


namespace triangle_angle_measure_l1595_159545

theorem triangle_angle_measure (a b c : ℝ) (A C : ℝ) (h : b = c * Real.cos A + Real.sqrt 3 * a * Real.sin C) :
  C = π / 6 := by
  sorry

end triangle_angle_measure_l1595_159545


namespace batsman_average_l1595_159532

def average (totalRuns : ℕ) (innings : ℕ) : ℚ :=
  (totalRuns : ℚ) / (innings : ℚ)

theorem batsman_average (totalRuns18 : ℕ) (totalRuns17 : ℕ) :
  average totalRuns18 18 = 18 →
  totalRuns18 = totalRuns17 + 1 →
  average totalRuns17 17 = 19 := by
sorry

end batsman_average_l1595_159532


namespace expression_value_l1595_159574

theorem expression_value (b : ℚ) (h : b = 1/3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 30 := by sorry

end expression_value_l1595_159574


namespace common_tangent_existence_l1595_159579

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 169/100

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 49/4

/-- Common tangent line L -/
def L (a b c : ℕ) (x y : ℝ) : Prop := a * x + b * y = c

theorem common_tangent_existence :
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (Nat.gcd a (Nat.gcd b c) = 1) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧
      L a b c x₁ y₁ ∧ L a b c x₂ y₂) ∧
    a + b + c = 52 :=
by sorry

end common_tangent_existence_l1595_159579


namespace relationship_between_sum_and_product_l1595_159509

theorem relationship_between_sum_and_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → (a * b > 1 → a + b > 1)) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b > 1 ∧ a * b ≤ 1) :=
by sorry

end relationship_between_sum_and_product_l1595_159509


namespace distance_to_origin_l1595_159583

theorem distance_to_origin (a : ℝ) : |a - 0| = 5 → (3 - a = -2 ∨ 3 - a = 8) := by
  sorry

end distance_to_origin_l1595_159583


namespace quadratic_roots_product_l1595_159565

/-- Given a quadratic equation x^2 + px + q = 0 with roots p and q, 
    the product pq is either 0 or -2 -/
theorem quadratic_roots_product (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) → 
  pq = 0 ∨ pq = -2 := by
sorry

end quadratic_roots_product_l1595_159565


namespace parallel_lines_a_eq_neg_one_l1595_159546

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + y = b₁ ↔ m₂ * x + y = b₂) ↔ m₁ = m₂

/-- The slope of a line ax + by + c = 0 is -a/b when b ≠ 0 -/
axiom line_slope {a b c : ℝ} (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = -a/b * x - c/b

theorem parallel_lines_a_eq_neg_one (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0 ↔ x + (a - 1) * y + a^2 - 1 = 0) →
  a = -1 :=
sorry

end parallel_lines_a_eq_neg_one_l1595_159546


namespace janice_initial_sentences_janice_started_with_258_l1595_159549

/-- Calculates the number of sentences Janice started with today -/
theorem janice_initial_sentences 
  (typing_speed : ℕ) 
  (total_typing_time : ℕ) 
  (erased_sentences : ℕ) 
  (final_sentence_count : ℕ) : ℕ :=
let typed_sentences := typing_speed * total_typing_time
let added_sentences := typed_sentences - erased_sentences
final_sentence_count - added_sentences

/-- Proves that Janice started with 258 sentences today -/
theorem janice_started_with_258 : 
  janice_initial_sentences 6 53 40 536 = 258 := by
sorry

end janice_initial_sentences_janice_started_with_258_l1595_159549


namespace binomial_10_2_l1595_159595

theorem binomial_10_2 : (10 : ℕ).choose 2 = 45 := by
  sorry

end binomial_10_2_l1595_159595


namespace max_b_minus_a_l1595_159584

/-- Given a function f and a constant a, finds the maximum value of b-a -/
theorem max_b_minus_a (a : ℝ) (f : ℝ → ℝ) (h1 : a > -1) 
  (h2 : ∀ x, f x = Real.exp x - a * x + (1/2) * x^2) 
  (h3 : ∀ x b, f x ≥ (1/2) * x^2 + x + b) :
  ∃ (b : ℝ), b - a ≤ 1 + Real.exp (-1) ∧ 
  (∀ c, (∀ x, f x ≥ (1/2) * x^2 + x + c) → c - a ≤ 1 + Real.exp (-1)) :=
sorry

end max_b_minus_a_l1595_159584


namespace S_n_perfect_square_iff_T_n_perfect_square_iff_l1595_159528

/-- Definition of S_n -/
def S_n (n : ℕ) : ℕ := n * (4 * n + 5)

/-- Definition of T_n -/
def T_n (n : ℕ) : ℕ := n * (3 * n + 2)

/-- Definition of is_perfect_square -/
def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k^2

/-- Pell's equation solution -/
def is_pell_solution (l m : ℕ) : Prop := l^2 - 3 * m^2 = 1

/-- Theorem for S_n -/
theorem S_n_perfect_square_iff (n : ℕ) : 
  is_perfect_square (S_n n) ↔ n = 1 :=
sorry

/-- Theorem for T_n -/
theorem T_n_perfect_square_iff (n : ℕ) : 
  is_perfect_square (T_n n) ↔ ∃ m : ℕ, n = 2 * m^2 ∧ ∃ l : ℕ, is_pell_solution l m :=
sorry

end S_n_perfect_square_iff_T_n_perfect_square_iff_l1595_159528


namespace inequality_not_hold_l1595_159534

theorem inequality_not_hold (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a < 1) :
  ¬(a * b < b^2 ∧ b^2 < 1) :=
by sorry

end inequality_not_hold_l1595_159534


namespace division_remainder_l1595_159556

theorem division_remainder (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : 
  ∃ (q : ℕ), 2 * x + 3 * u * y = q * y + (if 2 * v < y then 2 * v else 2 * v - y) := by
  sorry

end division_remainder_l1595_159556


namespace exists_same_acquaintance_count_exists_no_three_same_acquaintance_count_l1595_159550

/-- Represents a meeting with participants and their acquaintances -/
structure Meeting where
  participants : Finset ℕ
  acquaintances : ℕ → Finset ℕ
  valid : ∀ i ∈ participants, acquaintances i ⊆ participants ∧ i ∉ acquaintances i

/-- There exist at least two participants with the same number of acquaintances -/
theorem exists_same_acquaintance_count (m : Meeting) (h : 1 < m.participants.card) :
  ∃ i j, i ∈ m.participants ∧ j ∈ m.participants ∧ i ≠ j ∧
    (m.acquaintances i).card = (m.acquaintances j).card :=
  sorry

/-- There exists an arrangement of acquaintances such that no three participants have the same number of acquaintances -/
theorem exists_no_three_same_acquaintance_count (n : ℕ) (h : 1 < n) :
  ∃ m : Meeting, m.participants.card = n ∧
    ∀ i j k, i ∈ m.participants → j ∈ m.participants → k ∈ m.participants →
      i ≠ j → j ≠ k → i ≠ k →
        (m.acquaintances i).card ≠ (m.acquaintances j).card ∨
        (m.acquaintances j).card ≠ (m.acquaintances k).card ∨
        (m.acquaintances i).card ≠ (m.acquaintances k).card :=
  sorry

end exists_same_acquaintance_count_exists_no_three_same_acquaintance_count_l1595_159550


namespace inverse_proportion_increasing_l1595_159560

theorem inverse_proportion_increasing (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → (m + 3) / x₁ < (m + 3) / x₂) → 
  m < -3 := by
sorry

end inverse_proportion_increasing_l1595_159560


namespace probability_is_seven_ninety_sixths_l1595_159598

/-- Triangle PQR with given side lengths -/
structure Triangle :=
  (PQ : ℝ)
  (QR : ℝ)
  (PR : ℝ)

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { PQ := 7,
    QR := 24,
    PR := 25 }

/-- A point randomly selected inside the triangle -/
def S : Type := Unit

/-- The midpoint of side QR -/
def M (t : Triangle) : ℝ × ℝ := sorry

/-- Function to determine if a point is closer to M than to P or R -/
def closerToM (t : Triangle) (s : S) : Prop := sorry

/-- The probability of the event -/
def probability (t : Triangle) : ℝ := sorry

/-- The main theorem -/
theorem probability_is_seven_ninety_sixths :
  probability problemTriangle = 7 / 96 := by sorry

end probability_is_seven_ninety_sixths_l1595_159598


namespace john_reading_days_l1595_159522

/-- Given that John reads 4 books a day and 48 books in 6 weeks, prove that he reads on 2 days per week. -/
theorem john_reading_days 
  (books_per_day : ℕ) 
  (total_books : ℕ) 
  (total_weeks : ℕ) 
  (h1 : books_per_day = 4) 
  (h2 : total_books = 48) 
  (h3 : total_weeks = 6) : 
  (total_books / books_per_day) / total_weeks = 2 :=
by sorry

end john_reading_days_l1595_159522


namespace algebraic_expression_value_l1595_159568

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x * y = -2) 
  (h2 : x + y = 4) : 
  x^2 * y + x * y^2 = -8 := by sorry

end algebraic_expression_value_l1595_159568


namespace angle_A_measure_l1595_159542

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- State the theorem
theorem angle_A_measure (t : Triangle) 
  (h1 : t.C = 3 * t.B) 
  (h2 : t.B = 15) 
  (h3 : t.A + t.B + t.C = 180) : 
  t.A = 120 := by
  sorry

end angle_A_measure_l1595_159542


namespace train_passing_jogger_time_l1595_159588

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (initial_distance : ℝ)
  (train_length : ℝ)
  (h1 : jogger_speed = 9 * 1000 / 3600) -- 9 km/hr in m/s
  (h2 : train_speed = 45 * 1000 / 3600) -- 45 km/hr in m/s
  (h3 : initial_distance = 240)
  (h4 : train_length = 110) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 35 := by
sorry


end train_passing_jogger_time_l1595_159588


namespace chris_mixture_problem_l1595_159526

/-- Given the conditions of Chris's mixture of raisins and nuts, prove that the number of pounds of nuts is 4. -/
theorem chris_mixture_problem (raisin_pounds : ℝ) (nut_pounds : ℝ) (raisin_cost : ℝ) (nut_cost : ℝ) :
  raisin_pounds = 3 →
  nut_cost = 2 * raisin_cost →
  (raisin_pounds * raisin_cost) = (3 / 11) * (raisin_pounds * raisin_cost + nut_pounds * nut_cost) →
  nut_pounds = 4 := by
sorry

end chris_mixture_problem_l1595_159526


namespace number_of_pupils_in_class_number_of_pupils_in_class_is_correct_l1595_159529

/-- The number of pupils in a class, given an error in mark entry and its effect on the class average. -/
theorem number_of_pupils_in_class : ℕ :=
  let incorrect_mark : ℕ := 73
  let correct_mark : ℕ := 63
  let average_increase : ℚ := 1/2
  20

/-- Proof that the number of pupils in the class is correct. -/
theorem number_of_pupils_in_class_is_correct (n : ℕ) 
  (h1 : n = number_of_pupils_in_class)
  (h2 : (incorrect_mark - correct_mark : ℚ) / n = average_increase) : 
  n = 20 := by
  sorry

end number_of_pupils_in_class_number_of_pupils_in_class_is_correct_l1595_159529


namespace original_sales_tax_percentage_l1595_159503

/-- Proves that the original sales tax percentage was 3.5% given the conditions -/
theorem original_sales_tax_percentage
  (new_tax_rate : ℚ)
  (market_price : ℚ)
  (tax_difference : ℚ)
  (h1 : new_tax_rate = 10 / 3)
  (h2 : market_price = 6600)
  (h3 : tax_difference = 10.999999999999991)
  : ∃ (original_tax_rate : ℚ), original_tax_rate = 7 / 2 :=
sorry

end original_sales_tax_percentage_l1595_159503


namespace prime_factors_of_N_l1595_159531

def N : ℕ := (10^2011 - 1) / 9

theorem prime_factors_of_N (p : ℕ) (hp : p.Prime) (hdiv : p ∣ N) :
  ∃ j : ℕ, p = 4022 * j + 1 := by
  sorry

end prime_factors_of_N_l1595_159531


namespace minimum_dimes_needed_l1595_159590

def shoe_cost : ℚ := 45.50
def five_dollar_bills : ℕ := 4
def one_dollar_coins : ℕ := 10
def dime_value : ℚ := 0.10

theorem minimum_dimes_needed (n : ℕ) : 
  (five_dollar_bills * 5 + one_dollar_coins * 1 + n * dime_value ≥ shoe_cost) →
  n ≥ 155 := by
  sorry

end minimum_dimes_needed_l1595_159590


namespace min_reciprocal_sum_l1595_159599

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 1) : 1/x + 1/y + 1/z ≥ 9 :=
sorry

end min_reciprocal_sum_l1595_159599


namespace gcd_7163_209_l1595_159559

theorem gcd_7163_209 : Nat.gcd 7163 209 = 19 := by
  have h1 : 7163 = 209 * 34 + 57 := by sorry
  have h2 : 209 = 57 * 3 + 38 := by sorry
  have h3 : 57 = 38 * 1 + 19 := by sorry
  have h4 : 38 = 19 * 2 := by sorry
  sorry

#check gcd_7163_209

end gcd_7163_209_l1595_159559


namespace exactly_two_transformations_map_pattern_l1595_159508

/-- A pattern on a line consisting of alternating right-facing and left-facing triangles,
    followed by their vertically flipped versions, creating a symmetric, infinite, repeating pattern. -/
structure TrianglePattern where
  ℓ : Line

/-- Transformations that can be applied to the pattern -/
inductive Transformation
  | Rotate90 : Point → Transformation
  | TranslateParallel : Real → Transformation
  | Rotate120 : Point → Transformation
  | TranslatePerpendicular : Real → Transformation

/-- Predicate to check if a transformation maps the pattern onto itself -/
def maps_onto_self (t : Transformation) (p : TrianglePattern) : Prop :=
  sorry

theorem exactly_two_transformations_map_pattern (p : TrianglePattern) :
  ∃! (ts : Finset Transformation), ts.card = 2 ∧
    (∀ t ∈ ts, maps_onto_self t p) ∧
    (∀ t : Transformation, maps_onto_self t p → t ∈ ts) :=
  sorry

end exactly_two_transformations_map_pattern_l1595_159508


namespace arithmetic_sequence_problem_l1595_159505

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define a geometric sequence
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence a d →
  a 3 = 7 →
  geometric_sequence (λ n => a n - 1) →
  a 10 = 21 := by
  sorry


end arithmetic_sequence_problem_l1595_159505
