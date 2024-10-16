import Mathlib

namespace NUMINAMATH_CALUDE_circles_intersect_l1751_175171

/-- The circles x^2 + y^2 = -4y and (x-1)^2 + y^2 = 1 are intersecting -/
theorem circles_intersect : ∃ (x y : ℝ),
  (x^2 + y^2 = -4*y) ∧ ((x-1)^2 + y^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_l1751_175171


namespace NUMINAMATH_CALUDE_overlap_area_is_one_l1751_175185

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  h_x : x < 3
  h_y : y < 3

/-- Represents a triangle on the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The two specific triangles on the grid -/
def triangle1 : GridTriangle := {
  p1 := ⟨0, 0, by norm_num, by norm_num⟩,
  p2 := ⟨2, 1, by norm_num, by norm_num⟩,
  p3 := ⟨1, 2, by norm_num, by norm_num⟩
}

def triangle2 : GridTriangle := {
  p1 := ⟨2, 2, by norm_num, by norm_num⟩,
  p2 := ⟨0, 1, by norm_num, by norm_num⟩,
  p3 := ⟨1, 0, by norm_num, by norm_num⟩
}

/-- Calculates the area of the overlapping region of two triangles -/
def overlapArea (t1 t2 : GridTriangle) : ℝ := sorry

/-- Theorem stating that the overlap area of the specific triangles is 1 -/
theorem overlap_area_is_one : overlapArea triangle1 triangle2 = 1 := by sorry

end NUMINAMATH_CALUDE_overlap_area_is_one_l1751_175185


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sum_of_sqrts_l1751_175188

theorem sqrt_sum_equals_sum_of_sqrts : 
  Real.sqrt (36 + 14 * Real.sqrt 6 + 14 * Real.sqrt 5 + 6 * Real.sqrt 30) = 
  Real.sqrt 15 + Real.sqrt 10 + Real.sqrt 8 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sum_of_sqrts_l1751_175188


namespace NUMINAMATH_CALUDE_all_but_one_are_sum_of_two_primes_l1751_175193

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = n

theorem all_but_one_are_sum_of_two_primes :
  ∀ k : ℕ, k > 0 → is_sum_of_two_primes (1 + 10 * k) :=
by sorry

end NUMINAMATH_CALUDE_all_but_one_are_sum_of_two_primes_l1751_175193


namespace NUMINAMATH_CALUDE_max_planes_eq_combinations_l1751_175111

/-- The number of points in space -/
def num_points : ℕ := 15

/-- A function that calculates the number of combinations of k items from n items -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- The maximum number of planes determined by the points -/
def max_planes : ℕ := combinations num_points 3

/-- Theorem stating that the maximum number of planes is equal to the number of combinations of 3 points from 15 points -/
theorem max_planes_eq_combinations : 
  max_planes = combinations num_points 3 := by sorry

end NUMINAMATH_CALUDE_max_planes_eq_combinations_l1751_175111


namespace NUMINAMATH_CALUDE_number_problem_l1751_175151

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 14 → (40/100 : ℝ) * N = 168 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1751_175151


namespace NUMINAMATH_CALUDE_wage_increase_hours_decrease_l1751_175195

theorem wage_increase_hours_decrease (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let new_wage := 1.5 * w
  let new_hours := h / 1.5
  let percent_decrease := 100 * (1 - 1 / 1.5)
  new_wage * new_hours = w * h ∧ 
  100 * (h - new_hours) / h = percent_decrease := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_hours_decrease_l1751_175195


namespace NUMINAMATH_CALUDE_line_equation_through_points_l1751_175103

/-- The equation of a line passing through two points (5, 0) and (2, -5) -/
theorem line_equation_through_points :
  ∃ (A B C : ℝ),
    (A * 5 + B * 0 + C = 0) ∧
    (A * 2 + B * (-5) + C = 0) ∧
    (A = 5 ∧ B = -3 ∧ C = -25) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l1751_175103


namespace NUMINAMATH_CALUDE_triangle_area_l1751_175158

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define a median
def isMedian (t : Triangle) (M : ℝ × ℝ) (X Y Z : ℝ × ℝ) : Prop :=
  M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2) ∨ 
  M = ((Y.1 + Z.1) / 2, (Y.2 + Z.2) / 2) ∨ 
  M = ((Z.1 + X.1) / 2, (Z.2 + X.2) / 2)

-- Define the intersection point O
def intersectionPoint (XM YN : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the right angle intersection
def isRightAngle (XM YN : ℝ × ℝ) (O : ℝ × ℝ) : Prop := sorry

-- Define the length of a line segment
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_area (t : Triangle) (M N O : ℝ × ℝ) :
  isMedian t M t.X t.Y t.Z →
  isMedian t N t.X t.Y t.Z →
  O = intersectionPoint M N →
  isRightAngle M N O →
  length t.X M = 18 →
  length t.Y N = 24 →
  area t = 288 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1751_175158


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l1751_175115

/-- Given a curve y = x^3 and a point (1, 1) on this curve, 
    the equation of the tangent line at this point is 3x - y - 2 = 0 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3) → -- The curve equation
  (1 = 1^3) → -- The point (1, 1) satisfies the curve equation
  (3*x - y - 2 = 0) -- The equation of the tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l1751_175115


namespace NUMINAMATH_CALUDE_steak_eaten_l1751_175146

theorem steak_eaten (original_weight : ℝ) (burn_ratio : ℝ) (eat_ratio : ℝ) : 
  original_weight = 30 ∧ 
  burn_ratio = 0.5 ∧ 
  eat_ratio = 0.8 → 
  original_weight * (1 - burn_ratio) * eat_ratio = 12 := by
sorry

end NUMINAMATH_CALUDE_steak_eaten_l1751_175146


namespace NUMINAMATH_CALUDE_group_size_proof_l1751_175107

theorem group_size_proof (n : ℕ) (D : ℝ) (h : D > 0) : 
  (n : ℝ) / 8 * D + (n : ℝ) / 10 * D = D → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l1751_175107


namespace NUMINAMATH_CALUDE_trailingZeros_100_factorial_l1751_175189

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailingZeros_100_factorial :
  trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trailingZeros_100_factorial_l1751_175189


namespace NUMINAMATH_CALUDE_parabola_point_order_l1751_175180

/-- Parabola function -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem parabola_point_order (a b c d : ℝ) : 
  f a = 2 → f b = 6 → f c = d → d < 1 → a < 0 → b > 0 → a < c ∧ c < b :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_order_l1751_175180


namespace NUMINAMATH_CALUDE_magnitude_unit_vector_times_vector_l1751_175163

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

/-- Given a unit vector e and a non-zero vector b, prove that |e|*b = b -/
theorem magnitude_unit_vector_times_vector (e b : n) 
  (h_unit : ‖e‖ = 1) (h_nonzero : b ≠ 0) : 
  ‖e‖ • b = b := by
  sorry

end NUMINAMATH_CALUDE_magnitude_unit_vector_times_vector_l1751_175163


namespace NUMINAMATH_CALUDE_regular_bike_wheels_count_l1751_175191

/-- The number of wheels on a regular bike -/
def regular_bike_wheels : ℕ := 2

/-- The number of regular bikes -/
def num_regular_bikes : ℕ := 7

/-- The number of children's bikes -/
def num_childrens_bikes : ℕ := 11

/-- The number of wheels on a children's bike -/
def childrens_bike_wheels : ℕ := 4

/-- The total number of wheels observed -/
def total_wheels : ℕ := 58

theorem regular_bike_wheels_count : 
  num_regular_bikes * regular_bike_wheels + 
  num_childrens_bikes * childrens_bike_wheels = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_regular_bike_wheels_count_l1751_175191


namespace NUMINAMATH_CALUDE_product_of_logs_l1751_175133

theorem product_of_logs (a b : ℕ+) : 
  (b - a = 870) →
  (Real.log b / Real.log a = 2) →
  (a + b : ℕ) = 930 := by
sorry

end NUMINAMATH_CALUDE_product_of_logs_l1751_175133


namespace NUMINAMATH_CALUDE_simplify_part1_simplify_part2_l1751_175170

-- Part 1
theorem simplify_part1 (x : ℝ) (h1 : 1 ≤ x) (h2 : x < 4) :
  Real.sqrt (1 - 2*x + x^2) + Real.sqrt (x^2 - 8*x + 16) = 3 := by sorry

-- Part 2
theorem simplify_part2 (x : ℝ) (h : 2 - x ≥ 0) :
  (Real.sqrt (2 - x))^2 - Real.sqrt (x^2 - 6*x + 9) = -1 := by sorry

end NUMINAMATH_CALUDE_simplify_part1_simplify_part2_l1751_175170


namespace NUMINAMATH_CALUDE_election_winner_votes_l1751_175145

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) :
  winner_percentage = 54/100 →
  vote_difference = 288 →
  ⌊(winner_percentage : ℝ) * total_votes⌋ - ⌊((1 - winner_percentage) : ℝ) * total_votes⌋ = vote_difference →
  ⌊(winner_percentage : ℝ) * total_votes⌋ = 1944 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1751_175145


namespace NUMINAMATH_CALUDE_binomial_coefficient_times_two_l1751_175166

theorem binomial_coefficient_times_two : 2 * (Nat.choose 30 3) = 8120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_times_two_l1751_175166


namespace NUMINAMATH_CALUDE_min_value_expression_l1751_175190

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / (2 * a) + 1 / b) ≥ Real.sqrt 2 + 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1751_175190


namespace NUMINAMATH_CALUDE_books_read_l1751_175101

theorem books_read (total_books : ℕ) (books_left : ℕ) (h : total_books = 19 ∧ books_left = 15) : 
  total_books - books_left = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_read_l1751_175101


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_power_last_two_digits_of_7_2017_l1751_175155

def last_two_digits (n : ℕ) : ℕ := n % 100

def power_pattern (k : ℕ) : ℕ :=
  match k % 4 with
  | 0 => 01
  | 1 => 07
  | 2 => 49
  | 3 => 43
  | _ => 0  -- This case should never occur

theorem last_two_digits_of_7_power (n : ℕ) :
  last_two_digits (7^n) = power_pattern n :=
sorry

theorem last_two_digits_of_7_2017 :
  last_two_digits (7^2017) = 07 :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_power_last_two_digits_of_7_2017_l1751_175155


namespace NUMINAMATH_CALUDE_rectangle_area_with_hole_l1751_175120

theorem rectangle_area_with_hole (x : ℝ) 
  (h : (3*x ≤ 2*x + 10) ∧ (x ≤ x + 3)) : 
  (2*x + 10) * (x + 3) - (3*x * x) = -x^2 + 16*x + 30 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_with_hole_l1751_175120


namespace NUMINAMATH_CALUDE_window_side_length_main_theorem_l1751_175168

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio_height_to_width : height = 3 * width

/-- Represents the dimensions of a square window -/
structure SquareWindow where
  pane : Pane
  border_width : ℝ
  side_length : ℝ
  pane_arrangement : side_length = 3 * pane.width + 4 * border_width

/-- Theorem: The side length of the square window is 24 inches -/
theorem window_side_length (w : SquareWindow) 
  (h1 : w.border_width = 3) : w.side_length = 24 := by
  sorry

/-- Main theorem combining all conditions -/
theorem main_theorem : ∃ (w : SquareWindow), 
  w.border_width = 3 ∧ w.side_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_window_side_length_main_theorem_l1751_175168


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1751_175136

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed : ℝ) (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ),
  current_speed = 6 →
  downstream_distance = 5.2 →
  downstream_time = 1/5 →
  (boat_speed + current_speed) * downstream_time = downstream_distance →
  boat_speed = 20 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1751_175136


namespace NUMINAMATH_CALUDE_sum_equals_four_l1751_175186

theorem sum_equals_four (x y : ℝ) (h : |x - 3| + |y + 2| = 0) : x + y + 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_four_l1751_175186


namespace NUMINAMATH_CALUDE_polynomial_square_l1751_175143

theorem polynomial_square (a b : ℚ) : 
  (∃ p q : ℚ, ∀ x, x^4 + x^3 - x^2 + a*x + b = (x^2 + p*x + q)^2) → 
  b = 25/64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_square_l1751_175143


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1751_175122

theorem complex_fraction_simplification : ∀ (x : ℝ),
  x = (3 * (Real.sqrt 3 + Real.sqrt 7)) / (4 * Real.sqrt (3 + Real.sqrt 2)) →
  x ≠ 3 * Real.sqrt 7 / 4 ∧
  x ≠ 9 * Real.sqrt 2 / 16 ∧
  x ≠ 3 * Real.sqrt 3 / 4 ∧
  x ≠ 15 / 8 ∧
  x ≠ 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1751_175122


namespace NUMINAMATH_CALUDE_similar_rectangles_l1751_175167

theorem similar_rectangles (w1 l1 w2 : ℝ) (hw1 : w1 = 25) (hl1 : l1 = 40) (hw2 : w2 = 15) :
  let l2 := w2 * l1 / w1
  let perimeter := 2 * (w2 + l2)
  let area := w2 * l2
  (l2 = 24 ∧ perimeter = 78 ∧ area = 360) := by sorry

end NUMINAMATH_CALUDE_similar_rectangles_l1751_175167


namespace NUMINAMATH_CALUDE_cube_surface_area_ratio_l1751_175194

theorem cube_surface_area_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (ratio : a = 7 * b) :
  (6 * a^2) / (6 * b^2) = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_ratio_l1751_175194


namespace NUMINAMATH_CALUDE_val_coin_ratio_l1751_175173

theorem val_coin_ratio :
  -- Define the number of nickels Val has initially
  let initial_nickels : ℕ := 20
  -- Define the value of a nickel in cents
  let nickel_value : ℕ := 5
  -- Define the value of a dime in cents
  let dime_value : ℕ := 10
  -- Define the total value in cents after finding additional nickels
  let total_value_after : ℕ := 900
  -- Define the function to calculate the number of additional nickels
  let additional_nickels (n : ℕ) : ℕ := 2 * n
  -- Define the function to calculate the total number of nickels after finding additional ones
  let total_nickels (n : ℕ) : ℕ := n + additional_nickels n
  -- Define the function to calculate the value of nickels in cents
  let nickel_value_cents (n : ℕ) : ℕ := n * nickel_value
  -- Define the function to calculate the value of dimes in cents
  let dime_value_cents (d : ℕ) : ℕ := d * dime_value
  -- Define the function to calculate the number of dimes
  let num_dimes (n : ℕ) : ℕ := (total_value_after - nickel_value_cents (total_nickels n)) / dime_value
  -- The ratio of dimes to nickels is 3:1
  num_dimes initial_nickels / initial_nickels = 3 := by
  sorry

end NUMINAMATH_CALUDE_val_coin_ratio_l1751_175173


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1751_175141

theorem cyclic_sum_inequality (a b c : ℝ) :
  |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| ≤ 
  (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2 := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1751_175141


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1751_175116

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the foci of the ellipse
def foci (F₁ F₂ : ℝ × ℝ) : Prop := ∃ c : ℝ, c^2 = 7 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the hyperbola
def on_hyperbola (P : ℝ × ℝ) : Prop := ∃ a b : ℝ, a^2 - b^2 = 1 ∧ (P.1^2/a^2) - (P.2^2/b^2) = 1

-- Define perpendicularity of PF₁ and PF₂
def perpendicular (P F₁ F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Define the product condition
def product_condition (P F₁ F₂ : ℝ × ℝ) : Prop :=
  ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 4

-- Theorem statement
theorem hyperbola_equation (P F₁ F₂ : ℝ × ℝ) :
  foci F₁ F₂ →
  on_hyperbola P →
  perpendicular P F₁ F₂ →
  product_condition P F₁ F₂ →
  ∃ x y : ℝ, P = (x, y) ∧ x^2/6 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1751_175116


namespace NUMINAMATH_CALUDE_imaginary_product_condition_l1751_175102

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem imaginary_product_condition (a : ℝ) : 
  (((1 : ℂ) + i) * ((1 : ℂ) + a * i)).re = 0 → a = 1 := by
  sorry

-- Note: We use .re to get the real part of the complex number, 
-- which should be 0 for a purely imaginary number.

end NUMINAMATH_CALUDE_imaginary_product_condition_l1751_175102


namespace NUMINAMATH_CALUDE_circle_condition_l1751_175178

/-- The equation x^2 + y^2 - 2x + m = 0 represents a circle if and only if m < 1 -/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + m = 0 ∧ ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - 2*x + m = 0) ↔ 
  m < 1 := by sorry

end NUMINAMATH_CALUDE_circle_condition_l1751_175178


namespace NUMINAMATH_CALUDE_slope_of_solutions_l1751_175121

/-- The equation that defines the relationship between x and y -/
def equation (x y : ℝ) : Prop := (4 / x) + (6 / y) = 0

/-- Theorem stating that the slope between any two distinct solutions of the equation is -3/2 -/
theorem slope_of_solutions (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : equation x₁ y₁) (h₂ : equation x₂ y₂) (h_dist : (x₁, y₁) ≠ (x₂, y₂)) :
  (y₂ - y₁) / (x₂ - x₁) = -3/2 := by
sorry

end NUMINAMATH_CALUDE_slope_of_solutions_l1751_175121


namespace NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l1751_175197

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the focus
def line_through_focus (m b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + b ∧ focus.2 = m * focus.1 + b

-- Define the intersection points of the line and the parabola
def intersection_points (m b : ℝ) : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line_through_focus m b p}

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (m b : ℝ) 
  (M N : ℝ × ℝ) 
  (h_M : M ∈ intersection_points m b) 
  (h_N : N ∈ intersection_points m b) 
  (h_distinct : M ≠ N) :
  let midpoint := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  midpoint.1 = 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l1751_175197


namespace NUMINAMATH_CALUDE_min_sum_squares_roots_l1751_175144

/-- The sum of squares of the roots of x^2 - (m+1)x + (m-1) = 0 is minimized when m = 0 -/
theorem min_sum_squares_roots (m : ℝ) : 
  let f : ℝ → ℝ := λ m => m^2 + 3
  let sum_squares := f m
  ∀ k : ℝ, f k ≥ f 0 := by sorry

end NUMINAMATH_CALUDE_min_sum_squares_roots_l1751_175144


namespace NUMINAMATH_CALUDE_min_black_cells_l1751_175169

/-- Represents a board configuration -/
def Board := Fin 2007 → Fin 2007 → Bool

/-- Checks if three cells form an L-trinome -/
def is_L_trinome (b : Board) (i j k : Fin 2007 × Fin 2007) : Prop :=
  sorry

/-- Checks if a board configuration is valid -/
def is_valid_configuration (b : Board) : Prop :=
  ∀ i j k, is_L_trinome b i j k → ¬(b i.1 i.2 ∧ b j.1 j.2 ∧ b k.1 k.2)

/-- Counts the number of black cells in a board configuration -/
def count_black_cells (b : Board) : Nat :=
  sorry

/-- The main theorem -/
theorem min_black_cells :
  ∃ (b : Board),
    is_valid_configuration b ∧
    count_black_cells b = (2007^2 / 3 : Nat) ∧
    ∀ (b' : Board),
      (∀ i j, b i j → b' i j) →
      count_black_cells b' > count_black_cells b →
      ¬is_valid_configuration b' :=
sorry

end NUMINAMATH_CALUDE_min_black_cells_l1751_175169


namespace NUMINAMATH_CALUDE_triangle_equation_solution_l1751_175192

/-- Definition of the triangle operator -/
def triangle (x y : ℝ) : ℝ := x * y + x + y

/-- Theorem stating that given 2 ▵ m = -16, m = -6 -/
theorem triangle_equation_solution :
  ∃ m : ℝ, triangle 2 m = -16 ∧ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_equation_solution_l1751_175192


namespace NUMINAMATH_CALUDE_linear_function_problem_l1751_175100

/-- A linear function satisfying specific conditions -/
def f (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x + b

/-- The theorem statement -/
theorem linear_function_problem (a b : ℝ) :
  (∀ x, f a b x = 3 * (f a b).invFun x ^ 2 + 5) →
  f a b 0 = 2 →
  f a b 3 = 3 * Real.sqrt 5 + 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_problem_l1751_175100


namespace NUMINAMATH_CALUDE_total_pages_to_read_l1751_175149

def pages_read : ℕ := 113
def days_left : ℕ := 5
def pages_per_day : ℕ := 59

theorem total_pages_to_read : pages_read + days_left * pages_per_day = 408 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_to_read_l1751_175149


namespace NUMINAMATH_CALUDE_no_valid_grid_exists_l1751_175183

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Checks if two positions in the grid are adjacent -/
def isAdjacent (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ (j.val + 1 = j'.val ∨ j'.val + 1 = j.val)) ∨
  (j = j' ∧ (i.val + 1 = i'.val ∨ i'.val + 1 = i.val))

/-- Checks if a grid satisfies the prime sum condition -/
def satisfiesPrimeSum (g : Grid) : Prop :=
  ∀ i j i' j' : Fin 3, isAdjacent i j i' j' → isPrime (g i j + g i' j')

/-- Checks if a grid contains all numbers from 1 to 9 exactly once -/
def containsAllNumbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃! i j : Fin 3, g i j = n.val + 1

/-- The main theorem stating that no valid grid exists -/
theorem no_valid_grid_exists : ¬∃ g : Grid, satisfiesPrimeSum g ∧ containsAllNumbers g := by
  sorry

end NUMINAMATH_CALUDE_no_valid_grid_exists_l1751_175183


namespace NUMINAMATH_CALUDE_g_100_value_l1751_175159

/-- A function satisfying the given property for all positive real numbers -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x * g y - y * g x = g (x / y) + x - y

/-- The main theorem stating the value of g(100) -/
theorem g_100_value (g : ℝ → ℝ) (h : SatisfiesProperty g) : g 100 = -99 / 2 := by
  sorry


end NUMINAMATH_CALUDE_g_100_value_l1751_175159


namespace NUMINAMATH_CALUDE_brian_initial_cards_l1751_175162

def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def cards_left : ℕ := 17

theorem brian_initial_cards : initial_cards = cards_taken + cards_left := by
  sorry

end NUMINAMATH_CALUDE_brian_initial_cards_l1751_175162


namespace NUMINAMATH_CALUDE_largest_blue_balls_l1751_175123

theorem largest_blue_balls (total : ℕ) (is_prime : ℕ → Prop) : 
  total = 72 →
  (∃ (red blue prime : ℕ), 
    red + blue = total ∧ 
    is_prime prime ∧ 
    red = blue + prime) →
  (∃ (max_blue : ℕ), 
    max_blue ≤ total ∧
    (∀ (blue : ℕ), 
      blue ≤ total →
      (∃ (red prime : ℕ), 
        red + blue = total ∧ 
        is_prime prime ∧ 
        red = blue + prime) →
      blue ≤ max_blue) ∧
    max_blue = 35) :=
by sorry

end NUMINAMATH_CALUDE_largest_blue_balls_l1751_175123


namespace NUMINAMATH_CALUDE_sum_of_a_values_for_single_solution_l1751_175150

theorem sum_of_a_values_for_single_solution (a : ℝ) : 
  let equation := fun (x : ℝ) ↦ 3 * x^2 + a * x + 12 * x + 16
  let discriminant := (a + 12)^2 - 4 * 3 * 16
  (∃! x, equation x = 0) → 
  (∃ a₁ a₂, a₁ + a₂ = -24 ∧ discriminant = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_a_values_for_single_solution_l1751_175150


namespace NUMINAMATH_CALUDE_min_distance_complex_l1751_175196

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ w, Complex.abs (z + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_l1751_175196


namespace NUMINAMATH_CALUDE_largest_number_l1751_175156

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 0.986) 
  (hb : b = 0.9851) 
  (hc : c = 0.9869) 
  (hd : d = 0.9807) 
  (he : e = 0.9819) : 
  max a (max b (max c (max d e))) = c := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1751_175156


namespace NUMINAMATH_CALUDE_perspective_square_area_l1751_175128

/-- A square whose perspective drawing is a parallelogram -/
structure PerspectiveSquare where
  /-- The side length of the parallelogram in the perspective drawing -/
  parallelogram_side : ℝ
  /-- The side length of the original square -/
  square_side : ℝ

/-- The theorem stating the possible areas of the square -/
theorem perspective_square_area (s : PerspectiveSquare) (h : s.parallelogram_side = 4) :
  s.square_side ^ 2 = 16 ∨ s.square_side ^ 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_perspective_square_area_l1751_175128


namespace NUMINAMATH_CALUDE_total_amount_received_l1751_175157

/-- The amount John won in the lottery -/
def lottery_winnings : ℚ := 155250

/-- The number of top students receiving money -/
def num_students : ℕ := 100

/-- The fraction of the winnings given to each student -/
def fraction_given : ℚ := 1 / 1000

theorem total_amount_received (lottery_winnings : ℚ) (num_students : ℕ) (fraction_given : ℚ) :
  (lottery_winnings * fraction_given) * num_students = 15525 :=
sorry

end NUMINAMATH_CALUDE_total_amount_received_l1751_175157


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1751_175176

theorem trigonometric_equation_solution (k : ℤ) : 
  let x : ℝ := -Real.arccos (-4/5) + (2 * k + 1 : ℝ) * Real.pi
  let y : ℝ := -1/2
  3 * Real.sin x - 4 * Real.cos x = 4 * y^2 + 4 * y + 6 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1751_175176


namespace NUMINAMATH_CALUDE_area_covered_by_overlapping_strips_l1751_175165

/-- Represents a rectangular strip with a given length and width of 1 unit -/
structure Strip where
  length : ℝ
  width : ℝ := 1

/-- Calculates the total area of overlaps between strips -/
def totalOverlapArea (strips : List Strip) : ℝ := sorry

/-- Theorem: Area covered by overlapping strips -/
theorem area_covered_by_overlapping_strips
  (strips : List Strip)
  (h_strips : strips = [
    { length := 8 },
    { length := 10 },
    { length := 12 },
    { length := 7 },
    { length := 9 }
  ])
  (h_overlap : totalOverlapArea strips = 16) :
  (strips.map (λ s => s.length * s.width)).sum - totalOverlapArea strips = 30 := by
  sorry

end NUMINAMATH_CALUDE_area_covered_by_overlapping_strips_l1751_175165


namespace NUMINAMATH_CALUDE_max_value_constraint_l1751_175172

theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 5 * y < 105) :
  x * y * (105 - 2 * x - 5 * y) ≤ 4287.5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1751_175172


namespace NUMINAMATH_CALUDE_second_jumper_height_l1751_175184

/-- The height of Ravi's jump in inches -/
def ravi_jump : ℝ := 39

/-- The height of the first next highest jumper in inches -/
def first_jumper : ℝ := 23

/-- The height of the third next highest jumper in inches -/
def third_jumper : ℝ := 28

/-- The ratio of Ravi's jump height to the average of the three next highest jumpers -/
def ravi_ratio : ℝ := 1.5

/-- The height of the second next highest jumper in inches -/
def second_jumper : ℝ := 27

theorem second_jumper_height :
  ravi_jump = ravi_ratio * (first_jumper + second_jumper + third_jumper) / 3 →
  second_jumper = 27 := by
  sorry

end NUMINAMATH_CALUDE_second_jumper_height_l1751_175184


namespace NUMINAMATH_CALUDE_area_between_parallel_chords_l1751_175140

/-- The area between two parallel chords in a circle -/
theorem area_between_parallel_chords (R : ℝ) (h : R > 0) :
  let circle_area := π * R^2
  let segment_60 := circle_area / 6 - R^2 * Real.sqrt 3 / 4
  let segment_120 := circle_area / 3 - R^2 * Real.sqrt 3 / 4
  circle_area - segment_60 - segment_120 = R^2 * (π + Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_area_between_parallel_chords_l1751_175140


namespace NUMINAMATH_CALUDE_max_distance_for_specific_car_l1751_175142

/-- Represents the lifespan of a set of tires in kilometers. -/
structure TireLifespan where
  km : ℕ

/-- Represents a car with front and rear tires. -/
structure Car where
  frontTires : TireLifespan
  rearTires : TireLifespan

/-- Calculates the maximum distance a car can travel with optimal tire swapping. -/
def maxDistance (car : Car) : ℕ :=
  sorry

/-- Theorem stating the maximum distance for a specific car configuration. -/
theorem max_distance_for_specific_car :
  let car := Car.mk (TireLifespan.mk 20000) (TireLifespan.mk 30000)
  maxDistance car = 24000 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_for_specific_car_l1751_175142


namespace NUMINAMATH_CALUDE_stone_slab_length_l1751_175125

theorem stone_slab_length (num_slabs : ℕ) (total_area : ℝ) (slab_length : ℝ) :
  num_slabs = 50 →
  total_area = 98 →
  num_slabs * (slab_length ^ 2) = total_area →
  slab_length = 1.4 :=
by
  sorry

#check stone_slab_length

end NUMINAMATH_CALUDE_stone_slab_length_l1751_175125


namespace NUMINAMATH_CALUDE_bill_problem_count_bill_composes_twenty_l1751_175110

theorem bill_problem_count : ℕ → Prop :=
  fun b : ℕ =>
    let r := 2 * b  -- Ryan's problem count
    let f := 3 * r  -- Frank's problem count
    let types := 4  -- Number of problem types
    let frank_per_type := 30  -- Frank's problems per type
    f = types * frank_per_type → b = 20

-- Proof
theorem bill_composes_twenty : ∃ b : ℕ, bill_problem_count b :=
  sorry

end NUMINAMATH_CALUDE_bill_problem_count_bill_composes_twenty_l1751_175110


namespace NUMINAMATH_CALUDE_omelets_per_person_l1751_175131

/-- Given 3 dozen eggs, 4 eggs per omelet, and 3 people, prove that each person gets 3 omelets when all eggs are used. -/
theorem omelets_per_person (total_eggs : ℕ) (eggs_per_omelet : ℕ) (num_people : ℕ) :
  total_eggs = 3 * 12 →
  eggs_per_omelet = 4 →
  num_people = 3 →
  (total_eggs / eggs_per_omelet) / num_people = 3 :=
by sorry

end NUMINAMATH_CALUDE_omelets_per_person_l1751_175131


namespace NUMINAMATH_CALUDE_farm_field_theorem_l1751_175161

/-- Represents the farm field and ploughing scenario -/
structure FarmField where
  totalArea : ℕ
  plannedRate : ℕ
  actualRate : ℕ
  extraDays : ℕ

/-- Calculates the area left to plough given the farm field scenario -/
def areaLeftToPlough (f : FarmField) : ℕ :=
  f.totalArea - f.actualRate * (f.totalArea / f.plannedRate + f.extraDays)

/-- Theorem stating that under the given conditions, 40 hectares are left to plough -/
theorem farm_field_theorem (f : FarmField) 
  (h1 : f.totalArea = 3780)
  (h2 : f.plannedRate = 90)
  (h3 : f.actualRate = 85)
  (h4 : f.extraDays = 2) :
  areaLeftToPlough f = 40 := by
  sorry

#eval areaLeftToPlough { totalArea := 3780, plannedRate := 90, actualRate := 85, extraDays := 2 }

end NUMINAMATH_CALUDE_farm_field_theorem_l1751_175161


namespace NUMINAMATH_CALUDE_carls_ride_distance_l1751_175108

/-- The distance between Carl's house and Ralph's house -/
def distance : ℝ := 10

/-- The time Carl spent riding to Ralph's house in hours -/
def time : ℝ := 5

/-- Carl's speed in miles per hour -/
def speed : ℝ := 2

/-- Theorem: The distance between Carl's house and Ralph's house is 10 miles -/
theorem carls_ride_distance : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_carls_ride_distance_l1751_175108


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_l1751_175106

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_l1751_175106


namespace NUMINAMATH_CALUDE_blue_twice_prob_octahedron_l1751_175181

/-- A regular octahedron with colored faces -/
structure ColoredOctahedron where
  blue_faces : ℕ
  red_faces : ℕ
  total_faces : ℕ
  is_regular : Prop
  face_sum : blue_faces + red_faces = total_faces

/-- The probability of an event occurring twice in independent trials -/
def independent_event_twice_prob (single_prob : ℚ) : ℚ :=
  single_prob * single_prob

/-- The probability of rolling a blue face twice in succession on a colored octahedron -/
def blue_twice_prob (o : ColoredOctahedron) : ℚ :=
  independent_event_twice_prob ((o.blue_faces : ℚ) / (o.total_faces : ℚ))

theorem blue_twice_prob_octahedron :
  ∃ (o : ColoredOctahedron),
    o.blue_faces = 5 ∧
    o.red_faces = 3 ∧
    o.total_faces = 8 ∧
    o.is_regular ∧
    blue_twice_prob o = 25 / 64 := by
  sorry

end NUMINAMATH_CALUDE_blue_twice_prob_octahedron_l1751_175181


namespace NUMINAMATH_CALUDE_max_self_intersections_l1751_175130

/-- A closed six-segment broken line with vertices on a circle -/
structure BrokenLine where
  vertices : Fin 6 → ℝ × ℝ
  on_circle : ∀ i, (vertices i).1^2 + (vertices i).2^2 = 1

/-- The number of self-intersections in a broken line -/
def num_self_intersections (bl : BrokenLine) : ℕ := sorry

/-- Theorem: The maximum number of self-intersections is 7 -/
theorem max_self_intersections (bl : BrokenLine) :
  num_self_intersections bl ≤ 7 := by sorry

end NUMINAMATH_CALUDE_max_self_intersections_l1751_175130


namespace NUMINAMATH_CALUDE_octagon_area_in_circle_l1751_175119

theorem octagon_area_in_circle (r : ℝ) (h : r = 2) :
  let octagon_area := 8 * r^2 * Real.sin (π / 4)
  octagon_area = 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_octagon_area_in_circle_l1751_175119


namespace NUMINAMATH_CALUDE_net_increase_is_86400_l1751_175153

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 8

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 6

/-- Calculates the net population increase in one day -/
def net_population_increase (birth_rate death_rate : ℚ) (seconds_per_day : ℕ) : ℚ :=
  (birth_rate - death_rate) / 2 * seconds_per_day

/-- Theorem stating that the net population increase in one day is 86400 -/
theorem net_increase_is_86400 :
  net_population_increase birth_rate death_rate seconds_per_day = 86400 := by
  sorry

end NUMINAMATH_CALUDE_net_increase_is_86400_l1751_175153


namespace NUMINAMATH_CALUDE_cassidy_grounded_days_l1751_175182

/-- The number of days Cassidy is grounded for lying about her report card -/
def days_grounded_for_lying (total_days : ℕ) (grades_below_b : ℕ) (extra_days_per_grade : ℕ) : ℕ :=
  total_days - (grades_below_b * extra_days_per_grade)

/-- Theorem stating that Cassidy was grounded for 14 days for lying about her report card -/
theorem cassidy_grounded_days : 
  days_grounded_for_lying 26 4 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_cassidy_grounded_days_l1751_175182


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1751_175135

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + 3*x + 17 + (2*x + 6) + (x + 24)) / 6 = 26 → x = 79/7 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1751_175135


namespace NUMINAMATH_CALUDE_infinite_prime_pairs_l1751_175187

theorem infinite_prime_pairs : 
  ∃ (S : Set (ℕ × ℕ)), 
    (∀ (p q : ℕ), (p, q) ∈ S → Nat.Prime p ∧ Nat.Prime q) ∧ 
    (∀ (p q : ℕ), (p, q) ∈ S → p ∣ (2^(q-1) - 1) ∧ q ∣ (2^(p-1) - 1)) ∧ 
    Set.Infinite S :=
by sorry

end NUMINAMATH_CALUDE_infinite_prime_pairs_l1751_175187


namespace NUMINAMATH_CALUDE_milk_problem_l1751_175126

/-- The original amount of milk in liters -/
def original_milk : ℝ := 1.15

/-- The fraction of milk grandmother drank -/
def grandmother_drank : ℝ := 0.4

/-- The amount of milk remaining in liters -/
def remaining_milk : ℝ := 0.69

theorem milk_problem :
  original_milk * (1 - grandmother_drank) = remaining_milk :=
by sorry

end NUMINAMATH_CALUDE_milk_problem_l1751_175126


namespace NUMINAMATH_CALUDE_sphere_deflation_radius_l1751_175179

theorem sphere_deflation_radius (r : ℝ) (h : r = 4) :
  let hemisphere_volume := (2/3) * Real.pi * r^3
  let original_sphere_volume := (4/3) * Real.pi * (((4 * Real.rpow 2 (1/3)) / Real.rpow 3 (1/3))^3)
  hemisphere_volume = (3/4) * original_sphere_volume :=
by sorry

end NUMINAMATH_CALUDE_sphere_deflation_radius_l1751_175179


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l1751_175174

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored) / (b.innings + 1)

theorem batsman_average_after_12th_innings
  (b : Batsman)
  (h1 : b.innings = 11)
  (h2 : newAverage b 65 = b.average + 3)
  : newAverage b 65 = 32 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l1751_175174


namespace NUMINAMATH_CALUDE_only_winning_lottery_is_random_l1751_175124

-- Define the type for events
inductive Event
  | WaterBoiling
  | WinningLottery
  | AthleteRunning
  | DrawingRedBall

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.WaterBoiling => false
  | Event.WinningLottery => true
  | Event.AthleteRunning => false
  | Event.DrawingRedBall => false

-- Theorem statement
theorem only_winning_lottery_is_random :
  ∀ e : Event, isRandomEvent e ↔ e = Event.WinningLottery :=
sorry

end NUMINAMATH_CALUDE_only_winning_lottery_is_random_l1751_175124


namespace NUMINAMATH_CALUDE_no_solution_exists_l1751_175104

theorem no_solution_exists (x y : ℕ+) : 3 * y^2 ≠ x^4 + x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1751_175104


namespace NUMINAMATH_CALUDE_sequence_sum_proof_l1751_175198

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℚ := (n + 1) / 2

-- Define the geometric sequence {b_n}
def b (n : ℕ) : ℚ := 2^(n-1)

-- Define the sum of the first n terms of {b_n}
def T (n : ℕ) : ℚ := 2^n - 1

theorem sequence_sum_proof :
  -- Given conditions
  (a 3 = 2) ∧
  ((a 1 + a 2 + a 3) = 9/2) ∧
  (b 1 = a 1) ∧
  (b 4 = a 15) →
  -- Conclusion
  ∀ n : ℕ, T n = 2^n - 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_proof_l1751_175198


namespace NUMINAMATH_CALUDE_middle_number_proof_l1751_175164

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : a + b = 12) (h4 : a + c = 17) (h5 : b + c = 19) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1751_175164


namespace NUMINAMATH_CALUDE_nested_expression_value_l1751_175114

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l1751_175114


namespace NUMINAMATH_CALUDE_problem_statement_l1751_175117

theorem problem_statement (A : ℤ) (h : A = 43^2011 - 2011^43) : 
  (3 ∣ A) ∧ (A % 11 = 7) ∧ (A % 35 = 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1751_175117


namespace NUMINAMATH_CALUDE_tangent_circle_min_radius_l1751_175134

noncomputable section

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P on curve C
def P (x₀ y₀ : ℝ) : Prop := C x₀ y₀ ∧ y₀ > 0

-- Define the line l tangent to C at P
def l (x₀ y₀ k : ℝ) (x y : ℝ) : Prop := y - y₀ = k * (x - x₀)

-- Define the circle M centered at (a, 0)
def M (a r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = r^2

-- Main theorem
theorem tangent_circle_min_radius (a x₀ y₀ k r : ℝ) :
  a > 2 →
  P x₀ y₀ →
  (∀ x y, C x y → l x₀ y₀ k x y → x = x₀ ∧ y = y₀) →
  (∃ x y, l x₀ y₀ k x y ∧ M a r x y) →
  (∀ r' : ℝ, (∃ x y, l x₀ y₀ k x y ∧ M a r' x y) → r ≤ r') →
  a - x₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_min_radius_l1751_175134


namespace NUMINAMATH_CALUDE_triangle_area_l1751_175175

theorem triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 12) (h3 : c = 5) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1751_175175


namespace NUMINAMATH_CALUDE_not_domain_zero_to_three_l1751_175138

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The theorem stating that [0, 3] cannot be the domain of f(x) given its value range is [1, 2] -/
theorem not_domain_zero_to_three :
  (∀ y ∈ Set.Icc 1 2, ∃ x, f x = y) →
  ¬(∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc 1 2) :=
by sorry

end NUMINAMATH_CALUDE_not_domain_zero_to_three_l1751_175138


namespace NUMINAMATH_CALUDE_aunt_angela_nephews_l1751_175129

theorem aunt_angela_nephews (total_jellybeans : ℕ) (jellybeans_per_child : ℕ) (num_nieces : ℕ) :
  total_jellybeans = 70 →
  jellybeans_per_child = 14 →
  num_nieces = 2 →
  total_jellybeans = (num_nieces + 3) * jellybeans_per_child :=
by sorry

end NUMINAMATH_CALUDE_aunt_angela_nephews_l1751_175129


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1751_175112

theorem quadratic_form_sum (x : ℝ) : ∃ b c : ℝ, 
  (∀ x, x^2 - 16*x + 64 = (x + b)^2 + c) ∧ b + c = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1751_175112


namespace NUMINAMATH_CALUDE_sqrt_sum_theorem_l1751_175139

theorem sqrt_sum_theorem (a b : ℝ) : 
  Real.sqrt ((a - b)^2) + (a - b)^(1/5) = 
    if a ≥ b then 2*(a - b) else 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_theorem_l1751_175139


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1751_175148

/-- Given two 2D vectors a and b, where a = (2, 1) and b = (m, -1),
    and a is parallel to b, prove that m = -2. -/
theorem parallel_vectors_m_value :
  ∀ (m : ℝ),
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![m, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) →
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1751_175148


namespace NUMINAMATH_CALUDE_residue_5_2023_mod_11_l1751_175154

theorem residue_5_2023_mod_11 : 5^2023 ≡ 4 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_residue_5_2023_mod_11_l1751_175154


namespace NUMINAMATH_CALUDE_test_subjects_count_l1751_175127

def choose (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def number_of_colors : ℕ := 5
def colors_per_code : ℕ := 2
def unidentified_subjects : ℕ := 6

theorem test_subjects_count : 
  choose number_of_colors colors_per_code + unidentified_subjects = 16 := by
  sorry

end NUMINAMATH_CALUDE_test_subjects_count_l1751_175127


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l1751_175118

theorem quiz_competition_participants (initial_participants : ℕ) : 
  (initial_participants * 40 / 100 * 1 / 4 = 30) →
  initial_participants = 300 := by
sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l1751_175118


namespace NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l1751_175137

theorem factor_81_minus_27x_cubed (x : ℝ) :
  81 - 27 * x^3 = 3 * (3 - x) * (81 + 27 * x + 9 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l1751_175137


namespace NUMINAMATH_CALUDE_chess_piece_arrangements_l1751_175147

def num_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial 2)^k

theorem chess_piece_arrangements :
  num_arrangements 6 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_chess_piece_arrangements_l1751_175147


namespace NUMINAMATH_CALUDE_find_a_l1751_175177

theorem find_a : ∃ a : ℝ, (4 : ℝ) = 4 ∧ 5 * (4 - 1) - 3 * a = -3 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1751_175177


namespace NUMINAMATH_CALUDE_cattle_problem_l1751_175109

/-- Represents the problem of determining the number of cattle that died --/
theorem cattle_problem (initial_cattle : ℕ) (initial_price : ℕ) (price_reduction : ℕ) (total_loss : ℕ) : 
  initial_cattle = 340 →
  initial_price = 204000 →
  price_reduction = 150 →
  total_loss = 25200 →
  ∃ (dead_cattle : ℕ), 
    dead_cattle = 57 ∧ 
    (initial_cattle - dead_cattle) * (initial_price / initial_cattle - price_reduction) = initial_price - total_loss := by
  sorry


end NUMINAMATH_CALUDE_cattle_problem_l1751_175109


namespace NUMINAMATH_CALUDE_iron_wire_length_l1751_175199

/-- The length of each cut-off part of the wire in centimeters. -/
def cut_length : ℝ := 10

/-- The original length of the iron wire in centimeters. -/
def original_length : ℝ := 110

/-- The length of the remaining part of the wire after cutting both ends. -/
def remaining_length : ℝ := original_length - 2 * cut_length

/-- Theorem stating that the original length of the iron wire is 110 cm. -/
theorem iron_wire_length :
  (remaining_length = 4 * (2 * cut_length) + 10) →
  original_length = 110 :=
by sorry

end NUMINAMATH_CALUDE_iron_wire_length_l1751_175199


namespace NUMINAMATH_CALUDE_number_problem_l1751_175105

theorem number_problem : ∃ x : ℝ, x / 100 = 31.76 + 0.28 ∧ x = 3204 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1751_175105


namespace NUMINAMATH_CALUDE_bills_equal_at_122_minutes_l1751_175152

/-- United Telephone pricing structure -/
def united_base_rate : ℝ := 8.00
def united_per_minute : ℝ := 0.25
def united_tax_rate : ℝ := 0.10
def united_regulatory_fee : ℝ := 1.00

/-- Atlantic Call pricing structure -/
def atlantic_base_rate : ℝ := 12.00
def atlantic_per_minute : ℝ := 0.20
def atlantic_tax_rate : ℝ := 0.15
def atlantic_compatibility_fee : ℝ := 1.50

/-- Calculate the bill for United Telephone -/
def united_bill (minutes : ℝ) : ℝ :=
  let subtotal := united_base_rate + united_per_minute * minutes
  subtotal + united_tax_rate * subtotal + united_regulatory_fee

/-- Calculate the bill for Atlantic Call -/
def atlantic_bill (minutes : ℝ) : ℝ :=
  let subtotal := atlantic_base_rate + atlantic_per_minute * minutes
  subtotal + atlantic_tax_rate * subtotal + atlantic_compatibility_fee

/-- Theorem stating that the bills are equal at 122 minutes -/
theorem bills_equal_at_122_minutes :
  united_bill 122 = atlantic_bill 122 := by sorry

end NUMINAMATH_CALUDE_bills_equal_at_122_minutes_l1751_175152


namespace NUMINAMATH_CALUDE_new_ratio_after_boarders_join_l1751_175132

theorem new_ratio_after_boarders_join (initial_boarders : ℕ) (new_boarders : ℕ) :
  initial_boarders = 60 →
  new_boarders = 15 →
  (2 : ℚ) / 5 = initial_boarders / (initial_boarders * 5 / 2) →
  (1 : ℚ) / 2 = (initial_boarders + new_boarders) / (initial_boarders * 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_new_ratio_after_boarders_join_l1751_175132


namespace NUMINAMATH_CALUDE_estimate_eight_minus_two_sqrt_seven_l1751_175160

theorem estimate_eight_minus_two_sqrt_seven :
  2 < 8 - 2 * Real.sqrt 7 ∧ 8 - 2 * Real.sqrt 7 < 3 := by
  sorry

end NUMINAMATH_CALUDE_estimate_eight_minus_two_sqrt_seven_l1751_175160


namespace NUMINAMATH_CALUDE_square_remainder_mod_nine_l1751_175113

theorem square_remainder_mod_nine (N : ℤ) :
  (N % 9 = 2 ∨ N % 9 = 7) → (N^2 % 9 = 4) := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_mod_nine_l1751_175113
