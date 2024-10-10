import Mathlib

namespace inequality_solution_set_l1700_170025

theorem inequality_solution_set (x : ℝ) :
  (-x^2 + 3*x - 2 > 0) ↔ (1 < x ∧ x < 2) :=
by sorry

end inequality_solution_set_l1700_170025


namespace sum_of_real_solutions_l1700_170084

theorem sum_of_real_solutions (b : ℝ) (h : b > 2) :
  ∃ y : ℝ, y ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + y)) = y ∧
  y = (Real.sqrt (4 * b - 3) - 1) / 2 := by
sorry

end sum_of_real_solutions_l1700_170084


namespace min_value_expression_l1700_170066

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (Real.sqrt 10 - 1)^2 ∧
  (∃ (a b c : ℝ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 5 ∧
    (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 = 4 * (Real.sqrt 10 - 1)^2) :=
by sorry

end min_value_expression_l1700_170066


namespace modulus_one_minus_i_to_eight_l1700_170016

theorem modulus_one_minus_i_to_eight : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by
  sorry

end modulus_one_minus_i_to_eight_l1700_170016


namespace john_toy_store_spending_l1700_170030

def weekly_allowance : ℚ := 240/100

theorem john_toy_store_spending (arcade_fraction : ℚ) (candy_store_amount : ℚ) 
  (h1 : arcade_fraction = 3/5)
  (h2 : candy_store_amount = 64/100) :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction)
  let toy_store_amount := remaining_after_arcade - candy_store_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by sorry

end john_toy_store_spending_l1700_170030


namespace merry_go_round_revolutions_l1700_170089

theorem merry_go_round_revolutions 
  (r₁ : ℝ) (r₂ : ℝ) (rev₁ : ℝ) 
  (h₁ : r₁ = 36) 
  (h₂ : r₂ = 12) 
  (h₃ : rev₁ = 18) : 
  ∃ rev₂ : ℝ, rev₂ * r₂ = rev₁ * r₁ ∧ rev₂ = 54 := by
  sorry

end merry_go_round_revolutions_l1700_170089


namespace angle_measure_in_special_triangle_l1700_170001

theorem angle_measure_in_special_triangle (a b c : ℝ) (h : b^2 + c^2 = a^2 + b*c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))
  A = π/3 := by
sorry

end angle_measure_in_special_triangle_l1700_170001


namespace inequality_properties_l1700_170091

theorem inequality_properties (x y : ℝ) (h : x > y) : 
  (x - 3 > y - 3) ∧ 
  (x / 3 > y / 3) ∧ 
  (x + 3 > y + 3) ∧ 
  (-3 * x < -3 * y) := by
  sorry

end inequality_properties_l1700_170091


namespace distance_from_origin_to_point_l1700_170079

theorem distance_from_origin_to_point : Real.sqrt ((-12)^2 + 9^2) = 15 := by
  sorry

end distance_from_origin_to_point_l1700_170079


namespace circle_intersection_range_l1700_170031

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}
def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

-- State the theorem
theorem circle_intersection_range (r : ℝ) :
  r > 0 ∧ M ∩ N r = N r ↔ r ∈ Set.Ioo 0 (2 - Real.sqrt 2) := by
  sorry

end circle_intersection_range_l1700_170031


namespace rectangles_in_5x5_grid_l1700_170078

/-- The number of different rectangles in a 5x5 grid -/
def num_rectangles (n : ℕ) : ℕ := (n.choose 2) ^ 2

/-- Theorem: The number of different rectangles with sides parallel to the grid
    that can be formed by connecting four dots in a 5x5 square array of dots is 100. -/
theorem rectangles_in_5x5_grid :
  num_rectangles 5 = 100 := by
  sorry

#eval num_rectangles 5  -- Should output 100

end rectangles_in_5x5_grid_l1700_170078


namespace r_value_when_n_is_2_l1700_170090

theorem r_value_when_n_is_2 (n : ℕ) (s : ℕ) (r : ℕ) 
  (h1 : s = 2^n + 1) 
  (h2 : r = 3^s - s) 
  (h3 : n = 2) : 
  r = 238 := by
  sorry

end r_value_when_n_is_2_l1700_170090


namespace roots_of_polynomial_l1700_170070

def polynomial (x : ℝ) : ℝ := (x^2 - 3*x + 2) * x * (x - 4)

theorem roots_of_polynomial :
  {x : ℝ | polynomial x = 0} = {0, 1, 2, 4} := by sorry

end roots_of_polynomial_l1700_170070


namespace two_bedroom_units_l1700_170088

theorem two_bedroom_units (total_units : ℕ) (one_bedroom_cost two_bedroom_cost : ℕ) (total_cost : ℕ)
  (h1 : total_units = 12)
  (h2 : one_bedroom_cost = 360)
  (h3 : two_bedroom_cost = 450)
  (h4 : total_cost = 4950)
  (h5 : ∃ (x y : ℕ), x + y = total_units ∧ x * one_bedroom_cost + y * two_bedroom_cost = total_cost) :
  ∃ (y : ℕ), y = 7 ∧ ∃ (x : ℕ), x + y = total_units ∧ x * one_bedroom_cost + y * two_bedroom_cost = total_cost :=
by
  sorry

end two_bedroom_units_l1700_170088


namespace tangent_line_sum_l1700_170075

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 2 / x

theorem tangent_line_sum (a b m : ℝ) : 
  (∀ x : ℝ, 3 * x + f a 1 = b) →  -- Tangent line equation
  (∀ x : ℝ, f a x = a * Real.log x + 2 / x) →  -- Function definition
  (f a 1 = m) →  -- Point of tangency
  a + b = 4 := by sorry

end tangent_line_sum_l1700_170075


namespace kabadi_players_count_l1700_170004

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 30

/-- The number of people who play both kabadi and kho kho -/
def both_games : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 40

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := total_players - kho_kho_only + both_games

theorem kabadi_players_count : kabadi_players = 10 := by
  sorry

end kabadi_players_count_l1700_170004


namespace existence_of_opposite_colors_l1700_170006

/-- Represents a piece on the circle -/
inductive Piece
| White
| Black

/-- Represents the circle with pieces placed on it -/
structure Circle :=
  (pieces : Fin 40 → Piece)
  (white_count : Nat)
  (black_count : Nat)
  (white_count_eq : white_count = 25)
  (black_count_eq : black_count = 15)
  (total_count : white_count + black_count = 40)

/-- Two points are diametrically opposite if their indices differ by 20 (mod 40) -/
def diametricallyOpposite (i j : Fin 40) : Prop :=
  (i.val + 20) % 40 = j.val ∨ (j.val + 20) % 40 = i.val

/-- Main theorem: There exist diametrically opposite white and black pieces -/
theorem existence_of_opposite_colors (c : Circle) :
  ∃ (i j : Fin 40), diametricallyOpposite i j ∧ 
    c.pieces i = Piece.White ∧ c.pieces j = Piece.Black :=
sorry

end existence_of_opposite_colors_l1700_170006


namespace min_value_complex_expression_l1700_170035

theorem min_value_complex_expression (p q r : ℤ) (ξ : ℂ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (h_fourth_root : ξ^4 = 1)
  (h_not_one : ξ ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 5 ∧ 
    (∀ (p' q' r' : ℤ) (h_distinct' : p' ≠ q' ∧ q' ≠ r' ∧ p' ≠ r'),
      Complex.abs (p' + q' * ξ + r' * ξ^3) ≥ m) ∧
    (∃ (p' q' r' : ℤ) (h_distinct' : p' ≠ q' ∧ q' ≠ r' ∧ p' ≠ r'),
      Complex.abs (p' + q' * ξ + r' * ξ^3) = m) :=
by sorry

end min_value_complex_expression_l1700_170035


namespace function_upper_bound_l1700_170026

/-- Given a function f(x) = ax - x ln x - a, prove that if f(x) ≤ 0 for all x ≥ 2, 
    then a ≤ 2ln 2 -/
theorem function_upper_bound (a : ℝ) : 
  (∀ x : ℝ, x ≥ 2 → a * x - x * Real.log x - a ≤ 0) → 
  a ≤ 2 * Real.log 2 :=
by sorry

end function_upper_bound_l1700_170026


namespace det_of_matrix_is_one_l1700_170072

theorem det_of_matrix_is_one : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 7; 2, 3]
  Matrix.det A = 1 := by
  sorry

end det_of_matrix_is_one_l1700_170072


namespace tissues_per_box_l1700_170062

theorem tissues_per_box (boxes : ℕ) (used : ℕ) (left : ℕ) : 
  boxes = 3 → used = 210 → left = 270 → (used + left) / boxes = 160 :=
by
  sorry

end tissues_per_box_l1700_170062


namespace door_replacement_cost_l1700_170005

/-- The total cost of replacing doors given the number of bedroom and outside doors,
    the cost of outside doors, and that bedroom doors cost half as much as outside doors. -/
def total_door_cost (num_bedroom_doors num_outside_doors outside_door_cost : ℕ) : ℕ :=
  num_outside_doors * outside_door_cost +
  num_bedroom_doors * (outside_door_cost / 2)

/-- Theorem stating that the total cost for replacing 3 bedroom doors and 2 outside doors
    is $70, given that outside doors cost $20 each and bedroom doors cost half as much. -/
theorem door_replacement_cost :
  total_door_cost 3 2 20 = 70 := by
  sorry


end door_replacement_cost_l1700_170005


namespace line_points_l1700_170049

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_points : 
  let p1 : Point := ⟨4, 8⟩
  let p2 : Point := ⟨1, 2⟩
  let p3 : Point := ⟨3, 6⟩
  let p4 : Point := ⟨2, 4⟩
  let p5 : Point := ⟨5, 10⟩
  collinear p1 p2 p3 ∧ collinear p1 p2 p4 ∧ collinear p1 p2 p5 := by
  sorry

end line_points_l1700_170049


namespace clock_hands_angle_at_7_l1700_170013

/-- The number of hour marks on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The hour we're interested in -/
def target_hour : ℕ := 7

/-- The angle between each hour mark on the clock -/
def hour_angle : ℕ := full_circle_degrees / clock_hours

/-- The smaller angle formed by the clock hands at 7 o'clock -/
def smaller_angle_at_7 : ℕ := target_hour * hour_angle

theorem clock_hands_angle_at_7 :
  smaller_angle_at_7 = 150 := by sorry

end clock_hands_angle_at_7_l1700_170013


namespace expression_simplification_l1700_170017

theorem expression_simplification (a : ℝ) (h : a = 2023) :
  (a^2 - 6*a + 9) / (a^2 - 2*a) / (1 - 1/(a - 2)) = 2020 / 2023 := by
  sorry

end expression_simplification_l1700_170017


namespace wednesday_rainfall_calculation_l1700_170077

/-- Calculates the rainfall on Wednesday given the conditions of the problem -/
def wednesday_rainfall (monday : ℝ) (tuesday_difference : ℝ) : ℝ :=
  2 * (monday + (monday - tuesday_difference))

/-- Theorem stating that given the specific conditions, Wednesday's rainfall is 2.2 inches -/
theorem wednesday_rainfall_calculation :
  wednesday_rainfall 0.9 0.7 = 2.2 := by
  sorry

end wednesday_rainfall_calculation_l1700_170077


namespace range_of_m_l1700_170061

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3| ≤ 5
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, p x ∧ ¬(q x m)) →
  (0 < m ∧ m ≤ 3) :=
sorry

end range_of_m_l1700_170061


namespace line_transformation_l1700_170082

open Matrix

/-- The matrix representing the linear transformation -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 0, 1]

/-- The original line equation: x + y + 2 = 0 -/
def original_line (x y : ℝ) : Prop := x + y + 2 = 0

/-- The transformed line equation: x + 2y + 2 = 0 -/
def transformed_line (x y : ℝ) : Prop := x + 2*y + 2 = 0

/-- Theorem stating that the linear transformation maps the original line to the transformed line -/
theorem line_transformation :
  ∀ (x y : ℝ), original_line x y → 
  ∃ (x' y' : ℝ), M.mulVec ![x', y'] = ![x, y] ∧ transformed_line x' y' := by
sorry

end line_transformation_l1700_170082


namespace genuine_product_probability_l1700_170003

theorem genuine_product_probability 
  (p_second : ℝ) 
  (p_third : ℝ) 
  (h1 : p_second = 0.03) 
  (h2 : p_third = 0.01) 
  : 1 - (p_second + p_third) = 0.96 := by
  sorry

end genuine_product_probability_l1700_170003


namespace divisibility_property_l1700_170029

theorem divisibility_property (A B : ℤ) 
  (h : ∀ k : ℤ, 1 ≤ k ∧ k ≤ 65 → (A + B) % k = 0) : 
  ((A + B) % 66 = 0) ∧ ¬(∀ C D : ℤ, (∀ k : ℤ, 1 ≤ k ∧ k ≤ 65 → (C + D) % k = 0) → (C + D) % 67 = 0) :=
by sorry

end divisibility_property_l1700_170029


namespace crease_lines_equivalence_l1700_170050

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the ellipse
structure Ellipse where
  focus1 : Point
  focus2 : Point
  majorAxis : ℝ

-- Define the set of points on crease lines
def CreaseLines (c : Circle) (a : Point) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ (a' : Point), a'.x^2 + a'.y^2 = c.radius^2 ∧ 
    (p.1 - (a.x + a'.x)/2)^2 + (p.2 - (a.y + a'.y)/2)^2 = ((a.x - a'.x)^2 + (a.y - a'.y)^2) / 4 }

-- Define the set of points not on the ellipse
def NotOnEllipse (e : Ellipse) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (Real.sqrt ((p.1 - e.focus1.x)^2 + (p.2 - e.focus1.y)^2) + 
                 Real.sqrt ((p.1 - e.focus2.x)^2 + (p.2 - e.focus2.y)^2)) ≠ e.majorAxis }

-- Theorem statement
theorem crease_lines_equivalence 
  (c : Circle) (a : Point) (e : Ellipse) 
  (h1 : (a.x - c.center.1)^2 + (a.y - c.center.2)^2 < c.radius^2)  -- A is inside the circle
  (h2 : e.focus1 = Point.mk c.center.1 c.center.2)  -- O is a focus of the ellipse
  (h3 : e.focus2 = a)  -- A is the other focus of the ellipse
  (h4 : e.majorAxis = c.radius) :  -- The major axis of the ellipse is R
  CreaseLines c a = NotOnEllipse e := by
  sorry

end crease_lines_equivalence_l1700_170050


namespace triangle_cosine_theorem_l1700_170020

theorem triangle_cosine_theorem (a b c : ℝ) (h1 : b^2 = a*c) (h2 : c = 2*a) :
  let cos_C := (a^2 + b^2 - c^2) / (2*a*b)
  cos_C = -Real.sqrt 2 / 4 := by sorry

end triangle_cosine_theorem_l1700_170020


namespace sufficient_condition_for_inequality_l1700_170010

theorem sufficient_condition_for_inequality (x : ℝ) :
  0 < x ∧ x < 2 → x^2 - 3*x < 0 := by
  sorry

end sufficient_condition_for_inequality_l1700_170010


namespace r_daily_earnings_l1700_170045

/-- Given the daily earnings of three individuals p, q, and r, prove that r earns 60 per day. -/
theorem r_daily_earnings (P Q R : ℚ) 
  (h1 : P + Q + R = 190) 
  (h2 : P + R = 120)
  (h3 : Q + R = 130) : 
  R = 60 := by
  sorry

end r_daily_earnings_l1700_170045


namespace max_abc_value_l1700_170063

theorem max_abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + c = (a + c) * (b + c)) :
  a * b * c ≤ 1 / 27 := by
sorry

end max_abc_value_l1700_170063


namespace geometric_sequence_ratio_l1700_170096

/-- The common ratio of a geometric sequence starting with 10, -20, 40, -80 is -2 -/
theorem geometric_sequence_ratio : ∀ (a : ℕ → ℤ), 
  a 0 = 10 ∧ a 1 = -20 ∧ a 2 = 40 ∧ a 3 = -80 → 
  (∀ n : ℕ, a (n + 1) = a n * (-2)) :=
by sorry

end geometric_sequence_ratio_l1700_170096


namespace unique_k_for_coplanarity_l1700_170087

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the origin and points
variable (O A B C D : V)

-- Define the condition for coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), (D - A) = a • (B - A) + b • (C - A) + c • (A - A)

-- State the theorem
theorem unique_k_for_coplanarity :
  ∃! k : ℝ, ∀ (A B C D : V),
    (4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = 0) →
    coplanar A B C D :=
by sorry

end unique_k_for_coplanarity_l1700_170087


namespace log_x2y2_value_l1700_170012

theorem log_x2y2_value (x y : ℝ) (hxy4 : Real.log (x * y^4) = 1) (hx3y : Real.log (x^3 * y) = 1) :
  Real.log (x^2 * y^2) = 10/11 := by
sorry

end log_x2y2_value_l1700_170012


namespace baker_remaining_cakes_l1700_170015

theorem baker_remaining_cakes 
  (initial_cakes : ℝ) 
  (additional_cakes : ℝ) 
  (sold_cakes : ℝ) 
  (h1 : initial_cakes = 62.5)
  (h2 : additional_cakes = 149.25)
  (h3 : sold_cakes = 144.75) :
  initial_cakes + additional_cakes - sold_cakes = 67 := by
sorry

end baker_remaining_cakes_l1700_170015


namespace product_of_five_consecutive_integers_l1700_170064

theorem product_of_five_consecutive_integers (n : ℕ) : 
  n = 3 → (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end product_of_five_consecutive_integers_l1700_170064


namespace circle_C_equation_and_OP_not_parallel_AB_l1700_170068

-- Define the circle M
def circle_M (r : ℝ) (x y : ℝ) : Prop := (x + 2)^2 + (y + 2)^2 = r^2

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define circle C
def circle_C (r : ℝ) (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = r^2

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the slope of line OP
def slope_OP : ℝ := 1

-- Define the slope of line AB
def slope_AB : ℝ := 0

theorem circle_C_equation_and_OP_not_parallel_AB (r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, circle_C r x y ↔ (x - 2)^2 + (y - 2)^2 = r^2) ∧ 
  slope_OP ≠ slope_AB :=
sorry

end circle_C_equation_and_OP_not_parallel_AB_l1700_170068


namespace prob_at_least_one_is_correct_l1700_170053

/-- The probability of success for each event -/
def p : ℝ := 0.7

/-- The probability of at least one success in two independent events -/
def prob_at_least_one (p : ℝ) : ℝ := 1 - (1 - p) * (1 - p)

/-- Theorem stating that the probability of at least one success is 0.91 -/
theorem prob_at_least_one_is_correct : prob_at_least_one p = 0.91 := by
  sorry

#eval prob_at_least_one p

end prob_at_least_one_is_correct_l1700_170053


namespace equal_chords_implies_tangential_l1700_170043

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  -- Add necessary fields

/-- A circle -/
structure Circle where
  -- Add necessary fields

/-- Represents the property that a circle intersects each side of a quadrilateral at two points forming equal chords -/
def has_equal_chords_intersection (q : ConvexQuadrilateral) (c : Circle) : Prop :=
  sorry

/-- A quadrilateral is tangential if it has an inscribed circle -/
def is_tangential (q : ConvexQuadrilateral) : Prop :=
  ∃ c : Circle, sorry -- c is inscribed in q

/-- If a convex quadrilateral has the property that a circle intersects each of its sides 
    at two points forming equal chords, then the quadrilateral is tangential -/
theorem equal_chords_implies_tangential (q : ConvexQuadrilateral) (c : Circle) :
  has_equal_chords_intersection q c → is_tangential q :=
by
  sorry

end equal_chords_implies_tangential_l1700_170043


namespace triangle_side_length_l1700_170094

theorem triangle_side_length (A B C : ℝ) (h1 : Real.cos (3*A - B) + Real.sin (A + B) = 2) 
  (h2 : 0 < A ∧ A < π) (h3 : 0 < B ∧ B < π) (h4 : 0 < C ∧ C < π) (h5 : A + B + C = π) 
  (h6 : (4 : ℝ) = 4 * Real.sin A / Real.sin C) : 
  4 * Real.sin B / Real.sin C = 2 * Real.sqrt (2 - Real.sqrt 2) := by
sorry


end triangle_side_length_l1700_170094


namespace equation_solutions_l1700_170099

def equation (x : ℝ) : Prop :=
  x ≠ 4 ∧ x ≠ 6 ∧
  (x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 5) * (x - 4) * (x - 3) /
  ((x - 4) * (x - 6) * (x - 4)) = 1

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2 :=
by sorry

end equation_solutions_l1700_170099


namespace right_triangular_prism_volume_l1700_170065

/-- Given a right triangular prism where:
    - The lateral edge is equal to the height of its base
    - The area of the cross-section passing through this lateral edge and the height of the base is Q
    Prove that the volume of the prism is Q √(3Q) -/
theorem right_triangular_prism_volume (Q : ℝ) (Q_pos : Q > 0) :
  ∃ (V : ℝ), V = Q * Real.sqrt (3 * Q) ∧
  (∃ (a h : ℝ) (a_pos : a > 0) (h_pos : h > 0),
    h = a * Real.sqrt 5 / 2 ∧
    Q = a * Real.sqrt 5 / 2 * h ∧
    V = Real.sqrt 3 / 4 * a^2 * h) :=
by sorry

end right_triangular_prism_volume_l1700_170065


namespace ralph_wild_animal_pictures_l1700_170060

/-- The number of pictures Derrick has -/
def derrick_pictures : ℕ := 34

/-- The difference between Derrick's and Ralph's picture count -/
def picture_difference : ℕ := 8

/-- The number of pictures Ralph has -/
def ralph_pictures : ℕ := derrick_pictures - picture_difference

theorem ralph_wild_animal_pictures : ralph_pictures = 26 := by sorry

end ralph_wild_animal_pictures_l1700_170060


namespace smallest_n_for_monochromatic_isosceles_trapezoid_l1700_170009

/-- A coloring of vertices using three colors -/
def Coloring (n : ℕ) := Fin n → Fin 3

/-- Checks if four vertices form an isosceles trapezoid in a regular n-gon -/
def IsIsoscelesTrapezoid (n : ℕ) (v1 v2 v3 v4 : Fin n) : Prop :=
  sorry

/-- Checks if a coloring contains four vertices of the same color forming an isosceles trapezoid -/
def HasMonochromaticIsoscelesTrapezoid (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (v1 v2 v3 v4 : Fin n),
    c v1 = c v2 ∧ c v2 = c v3 ∧ c v3 = c v4 ∧
    IsIsoscelesTrapezoid n v1 v2 v3 v4

theorem smallest_n_for_monochromatic_isosceles_trapezoid :
  (∀ (c : Coloring 17), HasMonochromaticIsoscelesTrapezoid 17 c) ∧
  (∀ (n : ℕ), n < 17 → ∃ (c : Coloring n), ¬HasMonochromaticIsoscelesTrapezoid n c) :=
sorry

end smallest_n_for_monochromatic_isosceles_trapezoid_l1700_170009


namespace initial_number_proof_l1700_170071

theorem initial_number_proof : ∃ x : ℕ, 
  (↑x + 5.000000000000043 : ℝ) % 23 = 0 ∧ x = 18 := by
  sorry

end initial_number_proof_l1700_170071


namespace negation_of_universal_proposition_l1700_170014

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by sorry

end negation_of_universal_proposition_l1700_170014


namespace birthday_crayons_l1700_170095

theorem birthday_crayons (given_away lost remaining : ℕ) 
  (h1 : given_away = 111)
  (h2 : lost = 106)
  (h3 : remaining = 223) :
  given_away + lost + remaining = 440 := by
  sorry

end birthday_crayons_l1700_170095


namespace tom_payment_multiple_l1700_170032

def original_price : ℝ := 3.00
def tom_payment : ℝ := 9.00

theorem tom_payment_multiple : tom_payment / original_price = 3 := by
  sorry

end tom_payment_multiple_l1700_170032


namespace chairs_per_rectangular_table_l1700_170098

theorem chairs_per_rectangular_table :
  let round_tables : ℕ := 2
  let rectangular_tables : ℕ := 2
  let chairs_per_round_table : ℕ := 6
  let total_chairs : ℕ := 26
  (total_chairs - round_tables * chairs_per_round_table) / rectangular_tables = 7 := by
sorry

end chairs_per_rectangular_table_l1700_170098


namespace campers_third_week_l1700_170046

/-- Proves the number of campers in the third week given conditions about three consecutive weeks of camping. -/
theorem campers_third_week
  (total : ℕ)
  (second_week : ℕ)
  (h_total : total = 150)
  (h_second : second_week = 40)
  (h_difference : second_week = (second_week - 10) + 10) :
  total - (second_week - 10) - second_week = 80 :=
by sorry

end campers_third_week_l1700_170046


namespace cross_pentominoes_fit_on_chessboard_l1700_170033

/-- A "cross" pentomino consists of 5 unit squares -/
def cross_pentomino_area : ℝ := 5

/-- The chessboard is 8x8 units -/
def chessboard_side : ℝ := 8

/-- The number of cross pentominoes to be cut -/
def num_crosses : ℕ := 9

/-- The area of half-rectangles between crosses -/
def half_rectangle_area : ℝ := 1

/-- The number of half-rectangles -/
def num_half_rectangles : ℕ := 8

/-- The maximum area of corner pieces -/
def max_corner_piece_area : ℝ := 1.5

/-- The number of corner pieces -/
def num_corner_pieces : ℕ := 4

theorem cross_pentominoes_fit_on_chessboard :
  (num_crosses : ℝ) * cross_pentomino_area +
  (num_half_rectangles : ℝ) * half_rectangle_area +
  (num_corner_pieces : ℝ) * max_corner_piece_area ≤ chessboard_side ^ 2 :=
sorry

end cross_pentominoes_fit_on_chessboard_l1700_170033


namespace school_trip_buses_l1700_170080

/-- The number of buses for a school trip, given the number of supervisors per bus and the total number of supervisors. -/
def number_of_buses (supervisors_per_bus : ℕ) (total_supervisors : ℕ) : ℕ :=
  total_supervisors / supervisors_per_bus

/-- Theorem stating that the number of buses is 7, given the conditions from the problem. -/
theorem school_trip_buses : number_of_buses 3 21 = 7 := by
  sorry

end school_trip_buses_l1700_170080


namespace arithmetic_sequence_property_l1700_170067

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property that m + n = p + q implies a_m + a_n = a_p + a_q -/
def SufficientCondition (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q

/-- The property that a_m + a_n = a_p + a_q does not always imply m + n = p + q -/
def NotNecessaryCondition (a : ℕ → ℝ) : Prop :=
  ∃ m n p q : ℕ, a m + a n = a p + a q ∧ m + n ≠ p + q

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  ArithmeticSequence a →
  SufficientCondition a ∧ NotNecessaryCondition a :=
sorry

end arithmetic_sequence_property_l1700_170067


namespace sum_of_reciprocals_l1700_170027

theorem sum_of_reciprocals (x y z a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : x * y / (x - y) = a)
  (h2 : x * z / (x - z) = b)
  (h3 : y * z / (y - z) = c) :
  1 / x + 1 / y + 1 / z = (1 / a + 1 / b + 1 / c) / 2 :=
by sorry

end sum_of_reciprocals_l1700_170027


namespace small_cheese_slices_l1700_170023

/-- The number of slices in a pizza order --/
structure PizzaOrder where
  small_cheese : ℕ
  large_pepperoni : ℕ
  eaten_per_person : ℕ
  left_per_person : ℕ

/-- Theorem: Given the conditions, the small cheese pizza has 8 slices --/
theorem small_cheese_slices (order : PizzaOrder)
  (h1 : order.large_pepperoni = 14)
  (h2 : order.eaten_per_person = 9)
  (h3 : order.left_per_person = 2)
  : order.small_cheese = 8 := by
  sorry

end small_cheese_slices_l1700_170023


namespace song_book_cost_l1700_170000

/-- The cost of the song book given the total amount spent and the cost of the trumpet -/
theorem song_book_cost (total_spent : ℚ) (trumpet_cost : ℚ) (h1 : total_spent = 151) (h2 : trumpet_cost = 145.16) :
  total_spent - trumpet_cost = 5.84 := by
  sorry

end song_book_cost_l1700_170000


namespace prob_basket_A_given_white_l1700_170055

/-- Represents a basket with white and black balls -/
structure Basket where
  white : ℕ
  black : ℕ

/-- The probability of choosing a specific basket -/
def choose_probability : ℚ := 1/2

/-- Calculates the probability of picking a white ball from a given basket -/
def white_probability (b : Basket) : ℚ :=
  b.white / (b.white + b.black)

/-- Theorem: Probability of choosing Basket A given a white ball was picked -/
theorem prob_basket_A_given_white 
  (basket_A basket_B : Basket)
  (h_A : basket_A = ⟨2, 3⟩)
  (h_B : basket_B = ⟨1, 3⟩) :
  let p_A := choose_probability
  let p_B := choose_probability
  let p_W_A := white_probability basket_A
  let p_W_B := white_probability basket_B
  let p_W := p_A * p_W_A + p_B * p_W_B
  p_A * p_W_A / p_W = 8/13 := by
    sorry

end prob_basket_A_given_white_l1700_170055


namespace elevator_height_after_20_seconds_l1700_170097

/-- Calculates the height of a descending elevator after a given time. -/
def elevatorHeight (initialHeight : ℝ) (descentSpeed : ℝ) (time : ℝ) : ℝ :=
  initialHeight - descentSpeed * time

/-- Theorem: An elevator starting at 120 meters above ground and descending
    at 4 meters per second will be at 40 meters after 20 seconds. -/
theorem elevator_height_after_20_seconds :
  elevatorHeight 120 4 20 = 40 := by
  sorry

end elevator_height_after_20_seconds_l1700_170097


namespace basketball_score_proof_l1700_170074

theorem basketball_score_proof (junior_score : ℕ) (percentage_increase : ℚ) : 
  junior_score = 260 → percentage_increase = 20/100 →
  junior_score + (junior_score + junior_score * percentage_increase) = 572 :=
by sorry

end basketball_score_proof_l1700_170074


namespace perpendicular_planes_line_l1700_170008

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_line 
  (a : Line) (α β : Plane) (l : Line)
  (h1 : perp_planes α β)
  (h2 : l = intersect α β)
  (h3 : perp_line_plane a β) :
  subset a α ∧ perp_lines a l :=
sorry

end perpendicular_planes_line_l1700_170008


namespace quadratic_inequalities_solutions_l1700_170007

-- Define the solution sets
def solution_set1 : Set ℝ := {x | x ≤ -5 ∨ x ≥ 2}
def solution_set2 : Set ℝ := {x | (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2}

-- State the theorem
theorem quadratic_inequalities_solutions :
  (∀ x : ℝ, x^2 + 3*x - 10 ≥ 0 ↔ x ∈ solution_set1) ∧
  (∀ x : ℝ, x^2 - 3*x - 2 ≤ 0 ↔ x ∈ solution_set2) := by
  sorry

end quadratic_inequalities_solutions_l1700_170007


namespace smallest_undefined_inverse_l1700_170083

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ b : ℕ, b < a → (Nat.gcd b 72 = 1 ∨ Nat.gcd b 45 = 1)) ∧ 
  Nat.gcd a 72 > 1 ∧ 
  Nat.gcd a 45 > 1 → 
  a = 3 := by
sorry

end smallest_undefined_inverse_l1700_170083


namespace f_equality_f_explicit_formula_l1700_170073

-- Define the function f
def f : ℝ → ℝ := fun x => 2 * x - x^2

-- State the theorem
theorem f_equality (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : 
  f (1 - Real.cos x) = Real.sin x ^ 2 := by
  sorry

-- Prove that f(x) = 2x - x^2 for 0 ≤ x ≤ 2
theorem f_explicit_formula (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  f x = 2 * x - x^2 := by
  sorry

end f_equality_f_explicit_formula_l1700_170073


namespace cos_alpha_plus_beta_l1700_170018

theorem cos_alpha_plus_beta (α β : Real) 
  (h1 : Real.sin (3 * Real.pi / 4 + α) = 5 / 13)
  (h2 : Real.cos (Real.pi / 4 - β) = 3 / 5)
  (h3 : 0 < α) (h4 : α < Real.pi / 4) (h5 : Real.pi / 4 < β) (h6 : β < 3 * Real.pi / 4) :
  Real.cos (α + β) = -33 / 65 := by
  sorry

end cos_alpha_plus_beta_l1700_170018


namespace complex_cube_root_problem_l1700_170051

theorem complex_cube_root_problem : ∃ (c : ℤ), (1 + 3*I : ℂ)^3 = -26 + c*I := by
  sorry

end complex_cube_root_problem_l1700_170051


namespace consecutive_numbers_with_lcm_660_l1700_170057

theorem consecutive_numbers_with_lcm_660 (a b c : ℕ) : 
  b = a + 1 ∧ c = b + 1 ∧ Nat.lcm (Nat.lcm a b) c = 660 → 
  a = 10 ∧ b = 11 ∧ c = 12 := by
sorry

end consecutive_numbers_with_lcm_660_l1700_170057


namespace valid_four_digit_numbers_l1700_170047

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (∀ d : ℕ, d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10] → is_prime d) ∧
  (∀ p : ℕ, p ∈ [n / 100, n / 10 % 100, n % 100] → is_prime p)

theorem valid_four_digit_numbers :
  {n : ℕ | is_valid_number n} = {2373, 3737, 5373, 7373} :=
by sorry

end valid_four_digit_numbers_l1700_170047


namespace g_geq_f_implies_t_leq_one_l1700_170058

noncomputable section

open Real

-- Define the functions f and g
def f (x : ℝ) : ℝ := exp x - x * log x
def g (t : ℝ) (x : ℝ) : ℝ := exp x - t * x^2 + x

-- State the theorem
theorem g_geq_f_implies_t_leq_one (t : ℝ) :
  (∀ x > 0, g t x ≥ f x) → t ≤ 1 := by
  sorry

end

end g_geq_f_implies_t_leq_one_l1700_170058


namespace abc_fraction_value_l1700_170092

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 2)
  (h2 : b * c / (b + c) = 5)
  (h3 : c * a / (c + a) = 7) :
  a * b * c / (a * b + b * c + c * a) = 140 / 59 := by
  sorry

end abc_fraction_value_l1700_170092


namespace prob_heads_win_value_l1700_170085

/-- The probability of getting heads in a fair coin flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting tails in a fair coin flip -/
def p_tails : ℚ := 1/2

/-- The number of consecutive heads needed to win -/
def heads_to_win : ℕ := 6

/-- The number of consecutive tails needed to lose -/
def tails_to_lose : ℕ := 3

/-- The probability of encountering a run of 6 heads before a run of 3 tails 
    when repeatedly flipping a fair coin -/
def prob_heads_win : ℚ := 32/63

/-- Theorem stating that the probability of encountering a run of 6 heads 
    before a run of 3 tails when repeatedly flipping a fair coin is 32/63 -/
theorem prob_heads_win_value : 
  prob_heads_win = 32/63 :=
sorry

end prob_heads_win_value_l1700_170085


namespace square_minus_triple_eq_five_implies_double_square_minus_sextuple_plus_one_eq_eleven_l1700_170039

theorem square_minus_triple_eq_five_implies_double_square_minus_sextuple_plus_one_eq_eleven
  (a b : ℝ) (h : a^2 - 3*b = 5) : 2*a^2 - 6*b + 1 = 11 := by
  sorry

end square_minus_triple_eq_five_implies_double_square_minus_sextuple_plus_one_eq_eleven_l1700_170039


namespace rectangle_width_length_ratio_l1700_170036

/-- Given a rectangle with width w, length 10, and perimeter 30, 
    prove that the ratio of width to length is 1:2 -/
theorem rectangle_width_length_ratio 
  (w : ℝ) -- width of the rectangle
  (h1 : w > 0) -- width is positive
  (h2 : 2 * w + 2 * 10 = 30) -- perimeter formula
  : w / 10 = 1 / 2 := by
  sorry

end rectangle_width_length_ratio_l1700_170036


namespace bridge_length_l1700_170044

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 140)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  train_speed_kmh * (1000 / 3600) * crossing_time - train_length = 235 :=
sorry

end bridge_length_l1700_170044


namespace mean_temperature_is_84_l1700_170054

def temperatures : List ℚ := [80, 79, 81, 85, 87, 89, 87]

theorem mean_temperature_is_84 :
  (temperatures.sum / temperatures.length : ℚ) = 84 := by
  sorry

end mean_temperature_is_84_l1700_170054


namespace dog_human_years_ratio_l1700_170002

theorem dog_human_years_ratio : 
  (∀ (dog_age human_age : ℝ), dog_age = 7 * human_age) → 
  (∃ (x : ℝ), x * 3 = 21 ∧ 7 / x = 7 / 6) :=
by sorry

end dog_human_years_ratio_l1700_170002


namespace parabola_properties_l1700_170037

/-- Parabola with vertex at origin and focus at (1,0) -/
structure Parabola where
  vertex : ℝ × ℝ := (0, 0)
  focus : ℝ × ℝ := (1, 0)

/-- Line passing through the focus of the parabola -/
structure Line (p : Parabola) where
  slope : ℝ

/-- Intersection points of the line with the parabola -/
def intersection_points (p : Parabola) (l : Line p) : Set (ℝ × ℝ) :=
  sorry

/-- Area of triangle formed by origin, focus, and two intersection points -/
def triangle_area (p : Parabola) (l : Line p) : ℝ :=
  sorry

theorem parabola_properties (p : Parabola) :
  (∀ x y : ℝ, (x, y) ∈ {(x, y) | y^2 = 4*x}) ∧
  (∃ min_area : ℝ, min_area = 2 ∧ 
    ∀ l : Line p, triangle_area p l ≥ min_area) :=
sorry

end parabola_properties_l1700_170037


namespace solution_set_2x_plus_y_eq_9_l1700_170024

theorem solution_set_2x_plus_y_eq_9 :
  {(x, y) : ℕ × ℕ | 2 * x + y = 9} = {(0, 9), (1, 7), (2, 5), (3, 3), (4, 1)} := by
  sorry

end solution_set_2x_plus_y_eq_9_l1700_170024


namespace extended_segment_endpoint_l1700_170059

-- Define points in 2D space
def Point := ℝ × ℝ

-- Define the given points
def A : Point := (-3, 5)
def B : Point := (9, -1)

-- Define vector addition
def vadd (p q : Point) : Point := (p.1 + q.1, p.2 + q.2)

-- Define scalar multiplication
def smul (k : ℝ) (p : Point) : Point := (k * p.1, k * p.2)

-- Define vector from two points
def vec (p q : Point) : Point := (q.1 - p.1, q.2 - p.2)

-- Theorem statement
theorem extended_segment_endpoint (C : Point) :
  vec A B = smul 3 (vec B C) → C = (15, -4) := by
  sorry

end extended_segment_endpoint_l1700_170059


namespace abc_sum_product_l1700_170069

theorem abc_sum_product (x : ℝ) : ∃ a b c : ℝ, a + b + c = 1 ∧ a * b + a * c + b * c = x := by
  sorry

end abc_sum_product_l1700_170069


namespace sin_2alpha_value_l1700_170056

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π / 4 - α) = 4 / 5) : 
  Real.sin (2 * α) = 7 / 25 := by
  sorry

end sin_2alpha_value_l1700_170056


namespace tangent_line_touches_both_curves_l1700_170081

noncomputable def curve1 (x : ℝ) : ℝ := x^2 - Real.log x

noncomputable def curve2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

noncomputable def tangent_line (x : ℝ) : ℝ := x

theorem tangent_line_touches_both_curves (a : ℝ) :
  (∀ x, x > 0 → curve1 x ≥ tangent_line x) ∧
  (curve1 1 = tangent_line 1) ∧
  (∀ x, curve2 a x ≥ tangent_line x) ∧
  (∃ x, curve2 a x = tangent_line x) →
  a = 1 := by sorry

end tangent_line_touches_both_curves_l1700_170081


namespace rhombus_diagonal_l1700_170086

theorem rhombus_diagonal (side_length square_area rhombus_area diagonal1 diagonal2 : ℝ) :
  square_area = side_length * side_length →
  rhombus_area = square_area →
  rhombus_area = (diagonal1 * diagonal2) / 2 →
  side_length = 8 →
  diagonal1 = 16 →
  diagonal2 = 8 :=
by
  sorry

end rhombus_diagonal_l1700_170086


namespace long_furred_brown_dogs_l1700_170076

theorem long_furred_brown_dogs
  (total : ℕ)
  (long_furred : ℕ)
  (brown : ℕ)
  (neither : ℕ)
  (h_total : total = 45)
  (h_long_furred : long_furred = 26)
  (h_brown : brown = 22)
  (h_neither : neither = 8) :
  long_furred + brown - (total - neither) = 11 := by
sorry

end long_furred_brown_dogs_l1700_170076


namespace remainder_problem_l1700_170021

theorem remainder_problem (k : ℕ+) (h : ∃ q : ℕ, 120 = k^2 * q + 12) :
  ∃ r : ℕ, 160 = k * (160 / k) + r ∧ r < k ∧ r = 4 := by
  sorry

end remainder_problem_l1700_170021


namespace point_on_line_not_perpendicular_to_y_axis_l1700_170042

-- Define a line l with equation x + my - 2 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := x + m * y - 2 = 0

-- Theorem stating that (2,0) always lies on line l
theorem point_on_line (m : ℝ) : line_l m 2 0 := by sorry

-- Theorem stating that line l is not perpendicular to the y-axis
theorem not_perpendicular_to_y_axis (m : ℝ) : m ≠ 0 := by sorry

end point_on_line_not_perpendicular_to_y_axis_l1700_170042


namespace inverse_inequality_iff_inequality_l1700_170028

theorem inverse_inequality_iff_inequality (a b : ℝ) (h : a * b > 0) :
  (1 / a < 1 / b) ↔ (a > b) := by sorry

end inverse_inequality_iff_inequality_l1700_170028


namespace abigail_cookies_l1700_170022

theorem abigail_cookies (grayson_boxes : ℚ) (olivia_boxes : ℕ) (cookies_per_box : ℕ) (total_cookies : ℕ) :
  grayson_boxes = 3/4 →
  olivia_boxes = 3 →
  cookies_per_box = 48 →
  total_cookies = 276 →
  (total_cookies - (grayson_boxes * cookies_per_box + olivia_boxes * cookies_per_box)) / cookies_per_box = 2 := by
  sorry

end abigail_cookies_l1700_170022


namespace geometric_sequence_minimum_l1700_170040

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k : ℕ, a k > 0) →  -- Positive sequence
  (∃ q : ℝ, q > 0 ∧ ∀ k : ℕ, a (k + 1) = q * a k) →  -- Geometric sequence
  a 2018 = a 2017 + 2 * a 2016 →  -- Given condition
  (a m * a n = 16 * (a 1)^2) →  -- Derived from √(a_m * a_n) = 4a_1
  (∀ i j : ℕ, i > 0 ∧ j > 0 ∧ a i * a j = 16 * (a 1)^2 → 1/i + 5/j ≥ 7/4) :=
by sorry

end geometric_sequence_minimum_l1700_170040


namespace function_value_implies_input_l1700_170038

/-- Given a function f(x) = (2x + 1) / (x - 1) and f(p) = 4, prove that p = 5/2 -/
theorem function_value_implies_input (f : ℝ → ℝ) (p : ℝ) 
  (h1 : ∀ x, x ≠ 1 → f x = (2 * x + 1) / (x - 1))
  (h2 : f p = 4) :
  p = 5/2 := by
sorry

end function_value_implies_input_l1700_170038


namespace cara_seating_arrangements_l1700_170019

theorem cara_seating_arrangements (n : ℕ) (k : ℕ) : n = 8 ∧ k = 2 → Nat.choose n k = 28 := by
  sorry

end cara_seating_arrangements_l1700_170019


namespace combined_population_is_8000_l1700_170034

/-- The total population of five towns -/
def total_population : ℕ := 120000

/-- The population of Gordonia -/
def gordonia_population : ℕ := total_population / 3

/-- The population of Toadon -/
def toadon_population : ℕ := (gordonia_population * 3) / 4

/-- The population of Riverbank -/
def riverbank_population : ℕ := toadon_population + (toadon_population * 2) / 5

/-- The combined population of Lake Bright and Sunshine Hills -/
def lake_bright_sunshine_hills_population : ℕ := 
  total_population - (gordonia_population + toadon_population + riverbank_population)

theorem combined_population_is_8000 : 
  lake_bright_sunshine_hills_population = 8000 := by
  sorry

end combined_population_is_8000_l1700_170034


namespace negation_equivalence_l1700_170052

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Teenager : U → Prop)
variable (Responsible : U → Prop)

-- State the theorem
theorem negation_equivalence :
  (∃ x, Teenager x ∧ ¬Responsible x) ↔ ¬(∀ x, Teenager x → Responsible x) :=
by sorry

end negation_equivalence_l1700_170052


namespace cooper_savings_l1700_170011

/-- Calculates the total savings for a given daily savings amount and number of days. -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Proves that saving $34 daily for 365 days results in a total savings of $12,410. -/
theorem cooper_savings :
  totalSavings 34 365 = 12410 := by
  sorry

end cooper_savings_l1700_170011


namespace characterization_of_satisfying_functions_l1700_170048

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y^2 * f x + x^2 * f y + x * y = x * y * f (x + y) + x^2 + y^2

/-- The main theorem stating the form of functions satisfying the equation -/
theorem characterization_of_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x + 1 := by
sorry

end characterization_of_satisfying_functions_l1700_170048


namespace sneakers_cost_proof_l1700_170093

/-- The cost of a wallet in dollars -/
def wallet_cost : ℝ := 50

/-- The cost of a backpack in dollars -/
def backpack_cost : ℝ := 100

/-- The cost of a pair of jeans in dollars -/
def jeans_cost : ℝ := 50

/-- The total amount spent by Leonard and Michael in dollars -/
def total_spent : ℝ := 450

/-- The number of pairs of sneakers bought -/
def num_sneakers : ℕ := 2

/-- The number of pairs of jeans bought -/
def num_jeans : ℕ := 2

/-- The cost of each pair of sneakers in dollars -/
def sneakers_cost : ℝ := 100

theorem sneakers_cost_proof :
  wallet_cost + num_sneakers * sneakers_cost + backpack_cost + num_jeans * jeans_cost = total_spent :=
by sorry

end sneakers_cost_proof_l1700_170093


namespace factor_polynomial_l1700_170041

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end factor_polynomial_l1700_170041
