import Mathlib

namespace NUMINAMATH_CALUDE_circular_film_radius_l261_26181

/-- The radius of a circular film formed by a liquid --/
theorem circular_film_radius 
  (volume : ℝ) 
  (thickness : ℝ) 
  (radius : ℝ) 
  (h1 : volume = 320) 
  (h2 : thickness = 0.05) 
  (h3 : π * radius^2 * thickness = volume) : 
  radius = Real.sqrt (6400 / π) := by
sorry

end NUMINAMATH_CALUDE_circular_film_radius_l261_26181


namespace NUMINAMATH_CALUDE_interest_equality_problem_l261_26109

theorem interest_equality_problem (total : ℚ) (x : ℚ) : 
  total = 2743 →
  (x * 3 * 8) / 100 = ((total - x) * 5 * 3) / 100 →
  total - x = 1688 := by
  sorry

end NUMINAMATH_CALUDE_interest_equality_problem_l261_26109


namespace NUMINAMATH_CALUDE_book_distribution_l261_26170

theorem book_distribution (x : ℕ) : 
  (3 * x + 20 = 4 * x - 25) ↔ 
  (∃ (total_books : ℕ), 
    (total_books = 3 * x + 20) ∧ 
    (total_books = 4 * x - 25)) :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_l261_26170


namespace NUMINAMATH_CALUDE_exam_outcomes_count_l261_26127

/-- The number of possible outcomes for n people in a qualification exam -/
def exam_outcomes (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of possible outcomes for n people in a qualification exam is 2^n -/
theorem exam_outcomes_count (n : ℕ) : exam_outcomes n = 2^n := by
  sorry

end NUMINAMATH_CALUDE_exam_outcomes_count_l261_26127


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_endpoint_l261_26166

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 < 5) ∨
               (5 = y - 2 ∧ x + 3 < 5) ∨
               (x + 3 = y - 2 ∧ 5 < x + 3)}

-- Define a ray
def Ray (start : ℝ × ℝ) (dir : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * dir.1, start.2 + t * dir.2)}

-- Theorem statement
theorem T_is_three_rays_with_common_endpoint :
  ∃ (start : ℝ × ℝ) (dir1 dir2 dir3 : ℝ × ℝ),
    T = Ray start dir1 ∪ Ray start dir2 ∪ Ray start dir3 ∧
    dir1 ≠ dir2 ∧ dir1 ≠ dir3 ∧ dir2 ≠ dir3 :=
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_endpoint_l261_26166


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l261_26155

/-- Calculates the interest rate at which B lends money to C -/
theorem calculate_interest_rate (principal : ℝ) (rate_ab : ℝ) (time : ℝ) (gain : ℝ) : 
  principal = 4000 →
  rate_ab = 10 →
  time = 3 →
  gain = 180 →
  ∃ (rate_bc : ℝ), rate_bc = 11.5 ∧ 
    principal * (rate_bc / 100) * time = principal * (rate_ab / 100) * time + gain :=
by sorry


end NUMINAMATH_CALUDE_calculate_interest_rate_l261_26155


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l261_26182

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that given the collinearity of the points (4, -10), (-b + 4, 6), and (3b + 6, 4),
    the value of b must be -16/31 -/
theorem collinear_points_b_value :
  ∀ b : ℚ, collinear 4 (-10) (-b + 4) 6 (3*b + 6) 4 → b = -16/31 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l261_26182


namespace NUMINAMATH_CALUDE_largest_element_l261_26123

def S (a : ℝ) : Set ℝ := {-3*a, 2*a, 18/a, a^2, 1}

theorem largest_element (a : ℝ) (h : a = 3) : ∀ x ∈ S a, x ≤ a^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_element_l261_26123


namespace NUMINAMATH_CALUDE_vector_CQ_equals_2p_l261_26133

-- Define the space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points
variable (A B C P Q : V)

-- Define vector p
variable (p : V)

-- Conditions
variable (h1 : P ∈ interior (triangle A B C))
variable (h2 : A - P + 2 • (B - P) + 3 • (C - P) = 0)
variable (h3 : ∃ t : ℝ, Q = C + t • (P - C) ∧ Q ∈ line_through A B)
variable (h4 : C - P = p)

-- Theorem to prove
theorem vector_CQ_equals_2p : C - Q = 2 • p := by sorry

end NUMINAMATH_CALUDE_vector_CQ_equals_2p_l261_26133


namespace NUMINAMATH_CALUDE_complex_absolute_value_l261_26196

theorem complex_absolute_value (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z = (2*i)/(1+i) → Complex.abs (z - 2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l261_26196


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l261_26101

theorem quadratic_inequality_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 2*a ≥ 0) ↔ (-8 ≤ a ∧ a ≤ 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l261_26101


namespace NUMINAMATH_CALUDE_number_ratio_l261_26140

theorem number_ratio (x y z : ℝ) (k : ℝ) : 
  y = 2 * x →
  z = k * y →
  (x + y + z) / 3 = 165 →
  x = 45 →
  z / y = 4 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l261_26140


namespace NUMINAMATH_CALUDE_reeyas_average_score_l261_26108

def scores : List ℕ := [55, 67, 76, 82, 55]

theorem reeyas_average_score :
  (scores.sum : ℚ) / scores.length = 67 := by sorry

end NUMINAMATH_CALUDE_reeyas_average_score_l261_26108


namespace NUMINAMATH_CALUDE_michaels_class_size_l261_26132

theorem michaels_class_size (b : ℕ) : 
  (100 < b ∧ b < 200) ∧ 
  (∃ k : ℕ, b = 4 * k - 2) ∧ 
  (∃ l : ℕ, b = 5 * l - 3) ∧ 
  (∃ m : ℕ, b = 6 * m - 4) →
  (b = 122 ∨ b = 182) := by
sorry

end NUMINAMATH_CALUDE_michaels_class_size_l261_26132


namespace NUMINAMATH_CALUDE_cube_surface_area_l261_26184

/-- The surface area of a cube with edge length 5 cm is 150 cm². -/
theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 5) :
  6 * edge_length ^ 2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l261_26184


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l261_26121

def diamond (x y : ℕ) : ℕ := Nat.gcd x y

def oplus (x y : ℕ) : ℕ := Nat.lcm x y

theorem gcd_lcm_problem : 
  (oplus (oplus (diamond 24 36) (diamond 54 24)) (diamond (48 * 60) (72 * 48))) = 576 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l261_26121


namespace NUMINAMATH_CALUDE_complex_number_equality_l261_26141

theorem complex_number_equality (a : ℝ) : 
  (Complex.re ((2 * Complex.I - a) / Complex.I) = Complex.im ((2 * Complex.I - a) / Complex.I)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l261_26141


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l261_26143

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l261_26143


namespace NUMINAMATH_CALUDE_largest_c_value_l261_26110

theorem largest_c_value : ∃ (c_max : ℚ), 
  (∀ c : ℚ, (3 * c + 4) * (c - 2) = 9 * c → c ≤ c_max) ∧ 
  ((3 * c_max + 4) * (c_max - 2) = 9 * c_max) ∧
  c_max = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l261_26110


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l261_26193

theorem sin_2alpha_value (α : Real) (h : Real.tan (π/4 + α) = 2) : 
  Real.sin (2 * α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l261_26193


namespace NUMINAMATH_CALUDE_total_tickets_sold_l261_26179

def student_tickets : ℕ := 90
def non_student_tickets : ℕ := 60

theorem total_tickets_sold : student_tickets + non_student_tickets = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l261_26179


namespace NUMINAMATH_CALUDE_sock_pair_count_l261_26138

/-- The number of ways to choose a pair of socks from a drawer with specific conditions. -/
def sock_pairs (white brown blue : ℕ) : ℕ :=
  let total := white + brown + blue
  let same_color := Nat.choose white 2 + Nat.choose brown 2 + Nat.choose blue 2
  let not_blue := Nat.choose (white + brown) 2
  not_blue

/-- Theorem stating the number of valid sock pairs for the given problem. -/
theorem sock_pair_count :
  sock_pairs 5 5 2 = 45 := by
  sorry

#eval sock_pairs 5 5 2

end NUMINAMATH_CALUDE_sock_pair_count_l261_26138


namespace NUMINAMATH_CALUDE_least_k_for_convergence_l261_26151

def u : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 3 * u n - 5 * (u n)^2

def L : ℚ := 1/5

theorem least_k_for_convergence :
  ∀ k < 7, |u k - L| > 1/2^100 ∧ |u 7 - L| ≤ 1/2^100 := by sorry

end NUMINAMATH_CALUDE_least_k_for_convergence_l261_26151


namespace NUMINAMATH_CALUDE_construct_segment_a_construct_segment_b_l261_26146

-- Part a
theorem construct_segment_a (a : ℝ) (h : a = Real.sqrt 5) : ∃ b : ℝ, b = 1 := by
  sorry

-- Part b
theorem construct_segment_b (a : ℝ) (h : a = 7) : ∃ b : ℝ, b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_construct_segment_a_construct_segment_b_l261_26146


namespace NUMINAMATH_CALUDE_complex_equality_implies_a_equals_three_l261_26173

theorem complex_equality_implies_a_equals_three (a : ℝ) : 
  Complex.re ((1 + a * Complex.I) * (2 - Complex.I)) = Complex.im ((1 + a * Complex.I) * (2 - Complex.I)) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_implies_a_equals_three_l261_26173


namespace NUMINAMATH_CALUDE_triangle_height_equals_twice_rectangle_width_l261_26103

/-- Given a rectangle with dimensions a and b, and an isosceles triangle with base a and height h',
    if they have the same area, then the height of the triangle is 2b. -/
theorem triangle_height_equals_twice_rectangle_width
  (a b h' : ℝ) 
  (ha : a > 0)
  (hb : b > 0)
  (hh' : h' > 0)
  (h_area_eq : (1/2) * a * h' = a * b) :
  h' = 2 * b :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_equals_twice_rectangle_width_l261_26103


namespace NUMINAMATH_CALUDE_even_function_properties_l261_26178

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

-- Define what it means for f to be even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_properties (b : ℝ) 
  (h : is_even_function (f b)) :
  (b = 0) ∧ 
  (Set.Ioo 1 2 = {x | f b (x - 1) < x}) :=
by sorry

end NUMINAMATH_CALUDE_even_function_properties_l261_26178


namespace NUMINAMATH_CALUDE_root_sum_inverse_squares_l261_26150

theorem root_sum_inverse_squares (a b c : ℝ) : 
  a^3 - 12*a^2 + 20*a - 3 = 0 →
  b^3 - 12*b^2 + 20*b - 3 = 0 →
  c^3 - 12*c^2 + 20*c - 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 328/9 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_inverse_squares_l261_26150


namespace NUMINAMATH_CALUDE_chord_inclination_range_l261_26119

/-- The range of inclination angles for a chord through the focus of a parabola -/
theorem chord_inclination_range (x y : ℝ) (α : ℝ) : 
  (y^2 = 4*x) →                             -- Parabola equation
  (3*x^2 + 2*y^2 = 2) →                     -- Ellipse equation
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    y = (x - 1)*Real.tan α ∧               -- Chord passes through focus (1, 0)
    y^2 = 4*x ∧                            -- Chord intersects parabola
    (x₂ - x₁)^2 + ((x₂ - 1)*Real.tan α - (x₁ - 1)*Real.tan α)^2 ≤ 64) → -- Chord length ≤ 8
  (α ∈ Set.Icc (Real.pi/4) (Real.pi/3) ∪ Set.Icc (2*Real.pi/3) (3*Real.pi/4)) :=
by sorry

end NUMINAMATH_CALUDE_chord_inclination_range_l261_26119


namespace NUMINAMATH_CALUDE_probability_two_red_two_blue_eq_l261_26145

def total_marbles : ℕ := 27
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 12
def marbles_selected : ℕ := 4

def probability_two_red_two_blue : ℚ :=
  6 * (red_marbles.choose 2 * blue_marbles.choose 2) / total_marbles.choose marbles_selected

theorem probability_two_red_two_blue_eq :
  probability_two_red_two_blue = 154 / 225 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_two_blue_eq_l261_26145


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l261_26160

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 5
  let a : ℤ := 3
  let r : ℕ := 3
  let coeff : ℤ := (n.choose r) * a^(n-r) * (-1)^r
  coeff = -90 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l261_26160


namespace NUMINAMATH_CALUDE_range_of_m_l261_26131

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x - 6

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-10) (-6)) ∧
  (∀ y ∈ Set.Icc (-10) (-6), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l261_26131


namespace NUMINAMATH_CALUDE_probability_three_common_books_l261_26147

theorem probability_three_common_books (total_books : ℕ) (books_to_select : ℕ) (common_books : ℕ) :
  total_books = 12 →
  books_to_select = 7 →
  common_books = 3 →
  (Nat.choose total_books common_books * Nat.choose (total_books - common_books) (books_to_select - common_books) * Nat.choose (total_books - common_books) (books_to_select - common_books)) /
  (Nat.choose total_books books_to_select * Nat.choose total_books books_to_select) =
  3502800 / 627264 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_common_books_l261_26147


namespace NUMINAMATH_CALUDE_ernie_circles_problem_l261_26116

/-- Given a total number of boxes, the number of boxes Ali uses per circle,
    the number of circles Ali makes, and the number of boxes Ernie uses per circle,
    calculate the number of circles Ernie can make with the remaining boxes. -/
def ernie_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ali_circles : ℕ) (ernie_boxes_per_circle : ℕ) : ℕ :=
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle

/-- Theorem stating that given 80 boxes, if Ali uses 8 boxes per circle and makes 5 circles,
    and Ernie uses 10 boxes per circle, then Ernie can make 4 circles with the remaining boxes. -/
theorem ernie_circles_problem :
  ernie_circles 80 8 5 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ernie_circles_problem_l261_26116


namespace NUMINAMATH_CALUDE_intersection_A_B_l261_26156

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define set A
def A : Set ℝ := {x | f x < 0}

-- Define set B
def B : Set ℝ := {x | (deriv f) x > 0}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l261_26156


namespace NUMINAMATH_CALUDE_game_strategies_l261_26197

def game_state (n : ℕ) : Prop := n > 0

def player_A_move (n m : ℕ) : Prop := n ≤ m ∧ m ≤ n^2

def player_B_move (n m : ℕ) : Prop := ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = m * p^k

def A_wins (n : ℕ) : Prop := n = 1990

def B_wins (n : ℕ) : Prop := n = 1

def A_has_winning_strategy (n₀ : ℕ) : Prop := game_state n₀ ∧ n₀ ≥ 8

def B_has_winning_strategy (n₀ : ℕ) : Prop := game_state n₀ ∧ n₀ ≤ 5

def no_guaranteed_winning_strategy (n₀ : ℕ) : Prop := game_state n₀ ∧ (n₀ = 6 ∨ n₀ = 7)

theorem game_strategies :
  ∀ n₀ : ℕ, game_state n₀ →
    (A_has_winning_strategy n₀ ↔ n₀ ≥ 8) ∧
    (B_has_winning_strategy n₀ ↔ n₀ ≤ 5) ∧
    (no_guaranteed_winning_strategy n₀ ↔ (n₀ = 6 ∨ n₀ = 7)) :=
  sorry

end NUMINAMATH_CALUDE_game_strategies_l261_26197


namespace NUMINAMATH_CALUDE_jane_calculation_l261_26105

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 15) 
  (h2 : x - y - z = 7) : 
  x - y = 11 := by sorry

end NUMINAMATH_CALUDE_jane_calculation_l261_26105


namespace NUMINAMATH_CALUDE_no_equilateral_with_100_degree_angle_l261_26168

-- Define what an equilateral triangle is
def is_equilateral (a b c : ℝ) : Prop := a = b ∧ b = c

-- Define the sum of angles in a triangle
axiom triangle_angle_sum (a b c : ℝ) : a + b + c = 180

-- Theorem: An equilateral triangle cannot have an angle of 100 degrees
theorem no_equilateral_with_100_degree_angle (a b c : ℝ) :
  is_equilateral a b c → ¬(a = 100 ∨ b = 100 ∨ c = 100) :=
by sorry

end NUMINAMATH_CALUDE_no_equilateral_with_100_degree_angle_l261_26168


namespace NUMINAMATH_CALUDE_rational_expression_evaluation_l261_26126

theorem rational_expression_evaluation : 
  let x : ℝ := 8
  (x^4 - 18*x^2 + 81) / (x^2 - 9) = 55 := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_evaluation_l261_26126


namespace NUMINAMATH_CALUDE_line_through_P_perpendicular_to_given_line_l261_26195

-- Define the point P
def P : ℝ × ℝ := (4, -1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 3 * x - 4 * y + 6 = 0

-- Define the equation of the line we're looking for
def target_line (x y : ℝ) : Prop := 4 * x + 3 * y - 13 = 0

-- Theorem statement
theorem line_through_P_perpendicular_to_given_line :
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), m * x + y + b = 0 ↔ target_line x y) ∧
    (m * P.1 + P.2 + b = 0) ∧
    (m * 4 + 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_P_perpendicular_to_given_line_l261_26195


namespace NUMINAMATH_CALUDE_tan_product_30_60_l261_26129

theorem tan_product_30_60 : 
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (60 * π / 180)) = 2 + 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_30_60_l261_26129


namespace NUMINAMATH_CALUDE_exists_abs_less_than_one_l261_26162

def sequence_property (a : ℕ → ℝ) : Prop :=
  (a 1 * a 2 < 0) ∧
  (∀ n > 2, ∃ i j, 1 ≤ i ∧ i < j ∧ j < n ∧
    a n = a i + a j ∧
    ∀ k l, 1 ≤ k ∧ k < l ∧ l < n → |a i + a j| ≤ |a k + a l|)

theorem exists_abs_less_than_one (a : ℕ → ℝ) (h : sequence_property a) :
  ∃ i : ℕ, |a i| < 1 := by sorry

end NUMINAMATH_CALUDE_exists_abs_less_than_one_l261_26162


namespace NUMINAMATH_CALUDE_square_diagonal_perimeter_ratio_l261_26194

theorem square_diagonal_perimeter_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (a * Real.sqrt 2) / (b * Real.sqrt 2) = 5/2 → (4 * a) / (4 * b) = 5/2 := by
sorry

end NUMINAMATH_CALUDE_square_diagonal_perimeter_ratio_l261_26194


namespace NUMINAMATH_CALUDE_all_star_seating_arrangements_l261_26185

/-- Represents the number of ways to arrange All-Stars from different teams in a row --/
def allStarArrangements (total : Nat) (team1 : Nat) (team2 : Nat) (team3 : Nat) : Nat :=
  Nat.factorial 3 * Nat.factorial team1 * Nat.factorial team2 * Nat.factorial team3

/-- Theorem stating the number of arrangements for 8 All-Stars from 3 teams --/
theorem all_star_seating_arrangements :
  allStarArrangements 8 3 3 2 = 432 := by
  sorry

#eval allStarArrangements 8 3 3 2

end NUMINAMATH_CALUDE_all_star_seating_arrangements_l261_26185


namespace NUMINAMATH_CALUDE_f_even_l261_26169

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0
axiom functional_equation : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b

-- State the theorem to be proved
theorem f_even : ∀ x, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_f_even_l261_26169


namespace NUMINAMATH_CALUDE_complex_equation_solution_l261_26187

theorem complex_equation_solution (z : ℂ) : (1 + 2*I)*z = -3 + 4*I → z = 1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l261_26187


namespace NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_l261_26159

/-- Sum of digits of a three-digit number -/
def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- Theorem: Among 18 consecutive three-digit numbers, there is at least one divisible by its sum of digits -/
theorem exists_divisible_by_sum_of_digits (n : ℕ) (h : 100 ≤ n ∧ n ≤ 982) :
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % sumOfDigits k = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_l261_26159


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l261_26161

theorem exactly_one_positive_integer_solution : 
  ∃! (n : ℕ), n > 0 ∧ 25 - 5 * n > 15 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l261_26161


namespace NUMINAMATH_CALUDE_cloth_loss_per_meter_l261_26120

/-- Calculates the loss per meter of cloth given the total meters sold, total selling price, and cost price per meter. -/
def loss_per_meter (total_meters : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  let total_cost_price := total_meters * cost_price_per_meter
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_meters

/-- Theorem stating that for 400 meters of cloth sold at Rs. 18,000 with a cost price of Rs. 50 per meter, the loss per meter is Rs. 5. -/
theorem cloth_loss_per_meter :
  loss_per_meter 400 18000 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cloth_loss_per_meter_l261_26120


namespace NUMINAMATH_CALUDE_function_inverse_fraction_l261_26164

/-- Given a function f : ℝ \ {-1} → ℝ satisfying f((1-x)/(1+x)) = x for all x ≠ -1,
    prove that f(x) = (1-x)/(1+x) for all x ≠ -1 -/
theorem function_inverse_fraction (f : ℝ → ℝ) 
    (h : ∀ x ≠ -1, f ((1 - x) / (1 + x)) = x) :
    ∀ x ≠ -1, f x = (1 - x) / (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_function_inverse_fraction_l261_26164


namespace NUMINAMATH_CALUDE_power_product_equality_l261_26199

theorem power_product_equality (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l261_26199


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l261_26118

theorem unequal_gender_probability :
  let n : ℕ := 8  -- number of children
  let p : ℚ := 1/2  -- probability of each gender
  let total_outcomes : ℕ := 2^n
  let equal_outcomes : ℕ := n.choose (n/2)
  let unequal_outcomes : ℕ := total_outcomes - equal_outcomes
  (unequal_outcomes : ℚ) / total_outcomes = 93/128 := by
sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l261_26118


namespace NUMINAMATH_CALUDE_fish_filets_count_l261_26112

/-- The number of fish filets Ben and his family will have after their fishing trip -/
def fish_filets : ℕ :=
  let ben_fish := 4
  let judy_fish := 1
  let billy_fish := 3
  let jim_fish := 2
  let susie_fish := 5
  let small_fish := 3
  let filets_per_fish := 2
  let total_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let kept_fish := total_caught - small_fish
  kept_fish * filets_per_fish

/-- Theorem stating that the number of fish filets Ben and his family will have is 24 -/
theorem fish_filets_count : fish_filets = 24 := by
  sorry

end NUMINAMATH_CALUDE_fish_filets_count_l261_26112


namespace NUMINAMATH_CALUDE_cube_root_problem_l261_26171

theorem cube_root_problem :
  ∃ (a b : ℤ) (c : ℚ),
    (5 * a - 2 : ℚ) = -27 ∧
    b = Int.floor (Real.sqrt 22) ∧
    c = -(4 : ℚ)/25 ∧
    a = -5 ∧
    b = 4 ∧
    c = -(2 : ℚ)/5 ∧
    Real.sqrt ((4 : ℚ) * a * c + 7 * b) = 6 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_problem_l261_26171


namespace NUMINAMATH_CALUDE_sally_buttons_theorem_l261_26128

/-- The number of buttons needed for Sally's shirts -/
def buttons_needed (monday_shirts tuesday_shirts wednesday_shirts buttons_per_shirt : ℕ) : ℕ :=
  (monday_shirts + tuesday_shirts + wednesday_shirts) * buttons_per_shirt

/-- Theorem: Sally needs 45 buttons for all her shirts -/
theorem sally_buttons_theorem :
  buttons_needed 4 3 2 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sally_buttons_theorem_l261_26128


namespace NUMINAMATH_CALUDE_linear_function_and_inequality_l261_26157

-- Define the linear function f
def f : ℝ → ℝ := fun x ↦ x + 2

-- Define the function g
def g (a : ℝ) : ℝ → ℝ := fun x ↦ (1 - a) * x^2 - x

theorem linear_function_and_inequality (a : ℝ) :
  (∀ x, f (f x) = x + 4) →
  (∀ x₁ ∈ Set.Icc (1/4 : ℝ) 4, ∃ x₂ ∈ Set.Icc (-3 : ℝ) (1/3 : ℝ), g a x₁ ≥ f x₂) →
  (∀ x, f x = x + 2) ∧ a ∈ Set.Iic (3/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_and_inequality_l261_26157


namespace NUMINAMATH_CALUDE_unknown_bill_value_l261_26125

/-- Represents the contents of Ali's wallet -/
structure Wallet where
  five_dollar_bills : ℕ
  unknown_bill : ℕ
  total_amount : ℕ

/-- Theorem stating that given the conditions of Ali's wallet, the unknown bill is $10 -/
theorem unknown_bill_value (w : Wallet) 
  (h1 : w.five_dollar_bills = 7)
  (h2 : w.total_amount = 45) :
  w.unknown_bill = 10 := by
  sorry

#check unknown_bill_value

end NUMINAMATH_CALUDE_unknown_bill_value_l261_26125


namespace NUMINAMATH_CALUDE_f_is_mapping_from_A_to_B_l261_26130

def A : Set ℕ := {0, 1, 2, 4}
def B : Set ℚ := {1/2, 0, 1, 2, 6, 8}

def f (x : ℕ) : ℚ := 2^(x - 1)

theorem f_is_mapping_from_A_to_B : ∀ x ∈ A, f x ∈ B := by
  sorry

end NUMINAMATH_CALUDE_f_is_mapping_from_A_to_B_l261_26130


namespace NUMINAMATH_CALUDE_bill_face_value_l261_26107

/-- Calculates the face value of a bill given the true discount, interest rate, and time until due. -/
def face_value (true_discount : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  true_discount * (1 + interest_rate * time)

/-- Theorem: The face value of a bill with a true discount of 210, interest rate of 16% per annum, 
    and due in 9 months is 235.20. -/
theorem bill_face_value : 
  face_value 210 0.16 (9 / 12) = 235.20 := by
  sorry

end NUMINAMATH_CALUDE_bill_face_value_l261_26107


namespace NUMINAMATH_CALUDE_lego_problem_solution_l261_26134

def lego_problem (initial_pieces : ℕ) : ℕ :=
  let castle_pieces := initial_pieces / 4
  let after_castle := initial_pieces - castle_pieces
  let spaceship_pieces := (after_castle * 2) / 5
  let after_spaceship := after_castle - spaceship_pieces
  let lost_after_building := (after_spaceship * 15) / 100
  let after_loss := after_spaceship - lost_after_building
  let town_pieces := after_loss / 2
  let after_town := after_loss - town_pieces
  let final_loss := (after_town * 10) / 100
  after_town - final_loss

theorem lego_problem_solution :
  lego_problem 500 = 85 := by sorry

end NUMINAMATH_CALUDE_lego_problem_solution_l261_26134


namespace NUMINAMATH_CALUDE_pam_withdrawal_l261_26139

def initial_balance : ℕ := 400
def current_balance : ℕ := 950

def tripled_balance : ℕ := initial_balance * 3

def withdrawn_amount : ℕ := tripled_balance - current_balance

theorem pam_withdrawal : withdrawn_amount = 250 := by
  sorry

end NUMINAMATH_CALUDE_pam_withdrawal_l261_26139


namespace NUMINAMATH_CALUDE_expression_simplification_l261_26198

theorem expression_simplification (a b : ℝ) (h1 : a = 2) (h2 : b = -1) :
  2 * (-a^2 + 2*a*b) - 3 * (a*b - a^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l261_26198


namespace NUMINAMATH_CALUDE_bisecting_line_projection_ratio_l261_26188

/-- A convex polygon type -/
structure ConvexPolygon where
  -- Add necessary fields

/-- A line type -/
structure Line where
  -- Add necessary fields

/-- Represents the projection of a polygon onto a line -/
structure Projection where
  -- Add necessary fields

/-- Checks if a line bisects the area of a polygon -/
def bisects_area (l : Line) (p : ConvexPolygon) : Prop :=
  sorry

/-- Gets the projection of a polygon onto a line perpendicular to the given line -/
def get_perpendicular_projection (p : ConvexPolygon) (l : Line) : Projection :=
  sorry

/-- Gets the ratio of the segments created by a line on a projection -/
def projection_ratio (proj : Projection) (l : Line) : ℝ :=
  sorry

/-- The main theorem -/
theorem bisecting_line_projection_ratio 
  (p : ConvexPolygon) (l : Line) :
  bisects_area l p →
  projection_ratio (get_perpendicular_projection p l) l ≤ 1 + Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_projection_ratio_l261_26188


namespace NUMINAMATH_CALUDE_planted_fraction_is_seven_tenths_l261_26117

/-- Represents a right triangular field with an unplanted square at the right angle -/
structure RightTriangleField where
  leg1 : ℝ
  leg2 : ℝ
  square_to_hypotenuse : ℝ

/-- Calculates the fraction of the field that is planted -/
def planted_fraction (field : RightTriangleField) : ℝ :=
  sorry

/-- Theorem stating that the planted fraction is 7/10 for the given field -/
theorem planted_fraction_is_seven_tenths :
  let field : RightTriangleField := {
    leg1 := 5,
    leg2 := 12,
    square_to_hypotenuse := 3
  }
  planted_fraction field = 7/10 := by sorry

end NUMINAMATH_CALUDE_planted_fraction_is_seven_tenths_l261_26117


namespace NUMINAMATH_CALUDE_pants_discount_percentage_l261_26177

theorem pants_discount_percentage (cost : ℝ) (profit_percentage : ℝ) (marked_price : ℝ) :
  cost = 80 →
  profit_percentage = 0.3 →
  marked_price = 130 →
  let profit := cost * profit_percentage
  let selling_price := cost + profit
  let discount := marked_price - selling_price
  let discount_percentage := (discount / marked_price) * 100
  discount_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_pants_discount_percentage_l261_26177


namespace NUMINAMATH_CALUDE_cash_count_correction_l261_26167

/-- Represents the correction needed for a cash count error -/
def correction_needed (q d n c x : ℕ) : ℤ :=
  let initial_count := 25 * q + 10 * d + 5 * n + c
  let corrected_count := 25 * (q - x) + 10 * (d - x) + 5 * (n + x) + (c + x)
  corrected_count - initial_count

/-- 
Theorem: Given a cash count with q quarters, d dimes, n nickels, c cents,
and x nickels mistakenly counted as quarters and x dimes as cents,
the correction needed is to add 11x cents.
-/
theorem cash_count_correction (q d n c x : ℕ) :
  correction_needed q d n c x = 11 * x := by
  sorry

end NUMINAMATH_CALUDE_cash_count_correction_l261_26167


namespace NUMINAMATH_CALUDE_max_child_age_fraction_is_five_eighths_l261_26137

/-- The maximum fraction of Jane's age that a child she babysat could be -/
def max_child_age_fraction : ℚ :=
  let jane_current_age : ℕ := 34
  let years_since_stopped : ℕ := 10
  let jane_age_when_stopped : ℕ := jane_current_age - years_since_stopped
  let oldest_child_current_age : ℕ := 25
  let oldest_child_age_when_jane_stopped : ℕ := oldest_child_current_age - years_since_stopped
  (oldest_child_age_when_jane_stopped : ℚ) / jane_age_when_stopped

/-- Theorem stating that the maximum fraction of Jane's age that a child she babysat could be is 5/8 -/
theorem max_child_age_fraction_is_five_eighths :
  max_child_age_fraction = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_child_age_fraction_is_five_eighths_l261_26137


namespace NUMINAMATH_CALUDE_ratio_of_probabilities_l261_26175

/-- The number of rational terms in the expansion -/
def rational_terms : ℕ := 5

/-- The number of irrational terms in the expansion -/
def irrational_terms : ℕ := 4

/-- The total number of terms in the expansion -/
def total_terms : ℕ := rational_terms + irrational_terms

/-- The probability of having rational terms adjacent -/
def p : ℚ := (Nat.factorial rational_terms * Nat.factorial rational_terms) / Nat.factorial total_terms

/-- The probability of having no two rational terms adjacent -/
def q : ℚ := (Nat.factorial irrational_terms * Nat.factorial rational_terms) / Nat.factorial total_terms

theorem ratio_of_probabilities : p / q = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_probabilities_l261_26175


namespace NUMINAMATH_CALUDE_dartboard_probability_l261_26190

theorem dartboard_probability :
  -- Define the probabilities for each sector
  ∀ (prob_E prob_F prob_G prob_H prob_I : ℚ),
  -- Conditions
  prob_E = 1/5 →
  prob_F = 2/5 →
  prob_G = prob_H →
  prob_G = prob_I →
  -- Sum of all probabilities is 1
  prob_E + prob_F + prob_G + prob_H + prob_I = 1 →
  -- Conclusion: probability of landing on sector G is 2/15
  prob_G = 2/15 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_probability_l261_26190


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l261_26183

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) :
  A = 4 * Real.pi →
  A = Real.pi * r^2 →
  d = 2 * r →
  d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l261_26183


namespace NUMINAMATH_CALUDE_coefficient_of_x_l261_26135

theorem coefficient_of_x (x : ℝ) : 
  let expansion := (1 + x) * (x - 2/x)^3
  ∃ (a b c d e : ℝ), expansion = a*x^4 + b*x^3 + c*x^2 + (-6)*x + e
  := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l261_26135


namespace NUMINAMATH_CALUDE_truck_meeting_distance_difference_l261_26154

theorem truck_meeting_distance_difference 
  (initial_distance : ℝ) 
  (speed_a : ℝ) 
  (speed_b : ℝ) 
  (head_start : ℝ) :
  initial_distance = 855 →
  speed_a = 90 →
  speed_b = 80 →
  head_start = 1 →
  let relative_speed := speed_a + speed_b
  let meeting_time := (initial_distance - speed_a * head_start) / relative_speed
  let distance_a := speed_a * (meeting_time + head_start)
  let distance_b := speed_b * meeting_time
  distance_a - distance_b = 135 := by sorry

end NUMINAMATH_CALUDE_truck_meeting_distance_difference_l261_26154


namespace NUMINAMATH_CALUDE_perpendicular_planes_l261_26111

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (a b : Line) 
  (α β : Plane) 
  (hab : a ≠ b) 
  (hαβ : α ≠ β) 
  (hab_perp : perp_line a b) 
  (haα_perp : perp_line_plane a α) 
  (hbβ_perp : perp_line_plane b β) : 
  perp_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l261_26111


namespace NUMINAMATH_CALUDE_consortium_psychology_majors_l261_26189

theorem consortium_psychology_majors 
  (total : ℝ) 
  (college_A_percent : ℝ) 
  (college_B_percent : ℝ) 
  (college_C_percent : ℝ) 
  (college_A_freshmen : ℝ) 
  (college_B_freshmen : ℝ) 
  (college_C_freshmen : ℝ) 
  (college_A_liberal_arts : ℝ) 
  (college_B_liberal_arts : ℝ) 
  (college_C_liberal_arts : ℝ) 
  (college_A_psychology : ℝ) 
  (college_B_psychology : ℝ) 
  (college_C_psychology : ℝ) 
  (h1 : college_A_percent = 0.40) 
  (h2 : college_B_percent = 0.35) 
  (h3 : college_C_percent = 0.25) 
  (h4 : college_A_freshmen = 0.80) 
  (h5 : college_B_freshmen = 0.70) 
  (h6 : college_C_freshmen = 0.60) 
  (h7 : college_A_liberal_arts = 0.60) 
  (h8 : college_B_liberal_arts = 0.50) 
  (h9 : college_C_liberal_arts = 0.40) 
  (h10 : college_A_psychology = 0.50) 
  (h11 : college_B_psychology = 0.40) 
  (h12 : college_C_psychology = 0.30) : 
  (college_A_percent * college_A_freshmen * college_A_liberal_arts * college_A_psychology + 
   college_B_percent * college_B_freshmen * college_B_liberal_arts * college_B_psychology + 
   college_C_percent * college_C_freshmen * college_C_liberal_arts * college_C_psychology) * 100 = 16.3 := by
sorry

end NUMINAMATH_CALUDE_consortium_psychology_majors_l261_26189


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l261_26174

theorem quadratic_roots_expression (a b : ℝ) : 
  (a^2 - a - 1 = 0) → (b^2 - b - 1 = 0) → (3*a^2 + 2*b^2 - 3*a - 2*b = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l261_26174


namespace NUMINAMATH_CALUDE_prime_comparison_l261_26165

theorem prime_comparison (x y : ℕ) (hx : Prime x) (hy : Prime y) 
  (hlcm : Nat.lcm x y = 10) (heq : 2 * x + y = 12) : x > y := by
  sorry

end NUMINAMATH_CALUDE_prime_comparison_l261_26165


namespace NUMINAMATH_CALUDE_triplet_sum_not_two_l261_26106

theorem triplet_sum_not_two : ∃! (a b c : ℚ), 
  ((a, b, c) = (3/4, 1/2, 3/4) ∨ 
   (a, b, c) = (6/5, 1/5, 2/5) ∨ 
   (a, b, c) = (3/5, 7/10, 7/10) ∨ 
   (a, b, c) = (33/10, -8/5, 3/10) ∨ 
   (a, b, c) = (6/5, 1/5, 2/5)) ∧ 
  a + b + c ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_triplet_sum_not_two_l261_26106


namespace NUMINAMATH_CALUDE_catch_up_time_meeting_distance_l261_26136

def distance_AB : ℝ := 46
def speed_A : ℝ := 15
def speed_B : ℝ := 40
def time_difference : ℝ := 1

-- Time for Person B to catch up with Person A
theorem catch_up_time : 
  ∃ t : ℝ, speed_B * t = speed_A * (t + time_difference) ∧ t = 3/5 := by sorry

-- Distance from point B where they meet on Person B's return journey
theorem meeting_distance : 
  ∃ y : ℝ, 
    (distance_AB - y) / speed_A - (distance_AB + y) / speed_B = time_difference ∧ 
    y = 10 := by sorry

end NUMINAMATH_CALUDE_catch_up_time_meeting_distance_l261_26136


namespace NUMINAMATH_CALUDE_masking_tape_for_room_l261_26124

/-- Calculates the amount of masking tape needed for a room with given dimensions --/
def masking_tape_needed (wall_width1 : ℝ) (wall_width2 : ℝ) (window_width : ℝ) (door_width : ℝ) : ℝ :=
  2 * (wall_width1 + wall_width2) - (2 * window_width + door_width)

/-- Theorem stating that the amount of masking tape needed for the given room is 15 meters --/
theorem masking_tape_for_room : masking_tape_needed 4 6 1.5 2 = 15 := by
  sorry

#check masking_tape_for_room

end NUMINAMATH_CALUDE_masking_tape_for_room_l261_26124


namespace NUMINAMATH_CALUDE_intersection_count_7_intersection_count_21_l261_26100

-- Define the line equation
def line_equation (k : ℝ) (x y : ℝ) : Prop := k * x + y + k^3 = 0

-- Define the set of k values for the first case
def k_values_7 : Set ℝ := {0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9}

-- Define the set of k values for the second case
def k_values_21 : Set ℝ := {x : ℝ | ∃ n : ℤ, -10 ≤ n ∧ n ≤ 10 ∧ x = n / 10}

-- Define the function to count intersection points
noncomputable def count_intersections (k_values : Set ℝ) : ℕ := sorry

-- Theorem for the first case
theorem intersection_count_7 : 
  count_intersections k_values_7 = 11 := by sorry

-- Theorem for the second case
theorem intersection_count_21 :
  count_intersections k_values_21 = 110 := by sorry

end NUMINAMATH_CALUDE_intersection_count_7_intersection_count_21_l261_26100


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l261_26186

universe u

def U : Finset ℕ := {4,5,6,8,9}
def M : Finset ℕ := {5,6,8}

theorem complement_of_M_in_U :
  (U \ M) = {4,9} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l261_26186


namespace NUMINAMATH_CALUDE_family_divisors_characterization_l261_26114

/-- Represents a six-digit number and its family -/
def SixDigitFamily :=
  {A : ℕ // A ≥ 100000 ∧ A < 1000000}

/-- Generates the k-th member of the family for a six-digit number -/
def family_member (A : SixDigitFamily) (k : Fin 6) : ℕ :=
  let B := A.val / (10^k.val)
  let C := A.val % (10^k.val)
  10^(6-k.val) * C + B

/-- The set of numbers that divide all members of a six-digit number's family -/
def family_divisors (A : SixDigitFamily) : Set ℕ :=
  {x : ℕ | ∀ k : Fin 6, (family_member A k) % x = 0}

/-- The set of numbers we're proving to be the family_divisors -/
def target_set : Set ℕ :=
  {x : ℕ | x ≥ 1000000 ∨ 
           (∃ h : Fin 9, x = 111111 * (h.val + 1)) ∨
           999999 % x = 0}

/-- The main theorem stating that family_divisors is a subset of target_set -/
theorem family_divisors_characterization (A : SixDigitFamily) :
  family_divisors A ⊆ target_set := by
  sorry


end NUMINAMATH_CALUDE_family_divisors_characterization_l261_26114


namespace NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l261_26163

theorem smallest_perfect_square_multiplier : ∃ (n : ℕ), 
  (7 * n = 7 * 7) ∧ 
  (∃ (m : ℕ), m * m = 7 * n) ∧
  (∀ (k : ℕ), k < 7 → ¬∃ (m : ℕ), m * m = k * n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l261_26163


namespace NUMINAMATH_CALUDE_books_obtained_l261_26172

/-- Given an initial number of books and a final number of books,
    calculate the number of additional books obtained. -/
def additional_books (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that for the given initial and final book counts,
    the number of additional books is 23. -/
theorem books_obtained (initial : ℕ) (final : ℕ)
    (h1 : initial = 54)
    (h2 : final = 77) :
    additional_books initial final = 23 := by
  sorry

end NUMINAMATH_CALUDE_books_obtained_l261_26172


namespace NUMINAMATH_CALUDE_smallest_bob_number_l261_26115

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ n → p ∣ m)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), 
    has_all_prime_factors alice_number bob_number ∧
    (∀ m : ℕ, has_all_prime_factors alice_number m → bob_number ≤ m) ∧
    bob_number = 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l261_26115


namespace NUMINAMATH_CALUDE_periodic_even_function_theorem_l261_26158

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem periodic_even_function_theorem (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_defined : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end NUMINAMATH_CALUDE_periodic_even_function_theorem_l261_26158


namespace NUMINAMATH_CALUDE_factorial_nine_mod_eleven_l261_26102

theorem factorial_nine_mod_eleven : Nat.factorial 9 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_nine_mod_eleven_l261_26102


namespace NUMINAMATH_CALUDE_min_value_of_expression_l261_26180

theorem min_value_of_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 9) :
  ∃ (m : ℝ), m = 36 ∧ ∀ (a b : ℝ),
    (a = (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))^2 ∧
     b = (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))^2) →
    a - b ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l261_26180


namespace NUMINAMATH_CALUDE_shoe_company_earnings_l261_26149

/-- Proves that the current monthly earnings of a shoe company are $4000,
    given their annual goal and required monthly increase. -/
theorem shoe_company_earnings (annual_goal : ℕ) (monthly_increase : ℕ) (months_per_year : ℕ) :
  annual_goal = 60000 →
  monthly_increase = 1000 →
  months_per_year = 12 →
  (annual_goal / months_per_year - monthly_increase : ℕ) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_shoe_company_earnings_l261_26149


namespace NUMINAMATH_CALUDE_smallest_exceeding_day_l261_26176

def tea_intake (n : ℕ) : ℚ := (n * (n + 1) * (n + 2)) / 3

theorem smallest_exceeding_day : 
  (∀ k < 13, tea_intake k ≤ 900) ∧ tea_intake 13 > 900 := by sorry

end NUMINAMATH_CALUDE_smallest_exceeding_day_l261_26176


namespace NUMINAMATH_CALUDE_triangle_theorem_l261_26152

/-- Triangle ABC with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sqrt 2 * t.b * t.c = t.b^2 + t.c^2 - t.a^2) :
  t.A = π / 4 ∧ 
  (t.a = 2 * Real.sqrt 2 ∧ t.B = π / 3 → t.b = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l261_26152


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l261_26104

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 2 * y = 3) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 2 ∧ ∀ (a b : ℝ), a + 2 * b = 3 → 2^a + 4^b ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l261_26104


namespace NUMINAMATH_CALUDE_unique_solution_l261_26113

/-- Represents the number of vehicles of each type Jeff has -/
structure VehicleCounts where
  trucks : ℕ
  cars : ℕ
  motorcycles : ℕ
  buses : ℕ

/-- Checks if the given vehicle counts satisfy all the conditions -/
def satisfiesConditions (v : VehicleCounts) : Prop :=
  v.cars = 2 * v.trucks ∧
  v.motorcycles = 3 * v.cars ∧
  v.buses = v.trucks / 2 ∧
  v.trucks + v.cars + v.motorcycles + v.buses = 180

/-- The theorem stating that the given vehicle counts are the unique solution -/
theorem unique_solution : 
  ∃! v : VehicleCounts, satisfiesConditions v ∧ 
    v.trucks = 19 ∧ v.cars = 38 ∧ v.motorcycles = 114 ∧ v.buses = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l261_26113


namespace NUMINAMATH_CALUDE_monkey_bird_problem_l261_26144

theorem monkey_bird_problem (initial_monkeys initial_birds : ℕ) 
  (eating_monkeys : ℕ) (percentage_monkeys : ℚ) :
  initial_monkeys = 6 →
  initial_birds = 6 →
  eating_monkeys = 2 →
  percentage_monkeys = 60 / 100 →
  ∃ (birds_eaten : ℕ),
    birds_eaten * eating_monkeys = initial_birds - (initial_monkeys / percentage_monkeys - initial_monkeys) ∧
    birds_eaten = 1 :=
by sorry

end NUMINAMATH_CALUDE_monkey_bird_problem_l261_26144


namespace NUMINAMATH_CALUDE_student_percentage_theorem_l261_26191

theorem student_percentage_theorem (total : ℝ) (h_total_pos : total > 0) : 
  let third_year_percent : ℝ := 0.30
  let not_third_second_ratio : ℝ := 1/7
  let third_year : ℝ := third_year_percent * total
  let not_third_year : ℝ := total - third_year
  let second_year_not_third : ℝ := not_third_second_ratio * not_third_year
  let not_second_year : ℝ := total - second_year_not_third
  (not_second_year / total) * 100 = 90
:= by sorry

end NUMINAMATH_CALUDE_student_percentage_theorem_l261_26191


namespace NUMINAMATH_CALUDE_smallest_k_for_900_digit_sum_l261_26153

def digit_sum (n : ℕ) : ℕ := sorry

def repeated_7 (k : ℕ) : ℕ := (10^k - 1) / 9

theorem smallest_k_for_900_digit_sum : 
  ∀ k : ℕ, k > 0 → 
  (∀ j : ℕ, 0 < j ∧ j < k → digit_sum (9 * repeated_7 j) ≠ 900) ∧ 
  digit_sum (9 * repeated_7 k) = 900 → 
  k = 100 := by sorry

end NUMINAMATH_CALUDE_smallest_k_for_900_digit_sum_l261_26153


namespace NUMINAMATH_CALUDE_visitors_count_l261_26142

/-- Represents the cost per person based on the number of visitors -/
def cost_per_person (n : ℕ) : ℚ :=
  if n ≤ 30 then 100
  else max 72 (100 - 2 * (n - 30))

/-- The total cost for n visitors -/
def total_cost (n : ℕ) : ℚ := n * cost_per_person n

/-- Theorem stating that 35 is the number of visitors given the conditions -/
theorem visitors_count : ∃ (n : ℕ), n > 30 ∧ total_cost n = 3150 ∧ n = 35 := by
  sorry


end NUMINAMATH_CALUDE_visitors_count_l261_26142


namespace NUMINAMATH_CALUDE_alex_overall_score_l261_26148

def quiz_problems : ℕ := 30
def test_problems : ℕ := 50
def exam_problems : ℕ := 20

def quiz_score : ℚ := 75 / 100
def test_score : ℚ := 85 / 100
def exam_score : ℚ := 80 / 100

def total_problems : ℕ := quiz_problems + test_problems + exam_problems

def correct_problems : ℚ := 
  quiz_score * quiz_problems + test_score * test_problems + exam_score * exam_problems

theorem alex_overall_score : correct_problems / total_problems = 81 / 100 := by
  sorry

end NUMINAMATH_CALUDE_alex_overall_score_l261_26148


namespace NUMINAMATH_CALUDE_triangle_angle_c_is_right_angle_l261_26122

/-- Given a triangle ABC, if |sin A - 1/2| and (tan B - √3)² are opposite in sign, 
    then angle C is 90°. -/
theorem triangle_angle_c_is_right_angle 
  (A B C : ℝ) -- Angles of the triangle
  (h_triangle : A + B + C = PI) -- Sum of angles in a triangle is π radians (180°)
  (h_opposite_sign : (|Real.sin A - 1/2| * (Real.tan B - Real.sqrt 3)^2 < 0)) -- Opposite sign condition
  : C = PI / 2 := by -- C is π/2 radians (90°)
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_is_right_angle_l261_26122


namespace NUMINAMATH_CALUDE_right_triangle_area_l261_26192

/-- A right triangle ABC in the xy-plane with specific properties -/
structure RightTriangle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- The angle at C is a right angle -/
  right_angle_at_C : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  /-- The length of hypotenuse AB is 50 -/
  hypotenuse_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 50^2
  /-- The median through A lies along the line y = x - 2 -/
  median_A : ∃ (t : ℝ), A.2 = A.1 - 2 ∧ ((B.1 + C.1) / 2 = A.1 + t) ∧ ((B.2 + C.2) / 2 = A.2 + t)
  /-- The median through B lies along the line y = 3x + 1 -/
  median_B : ∃ (t : ℝ), B.2 = 3 * B.1 + 1 ∧ ((A.1 + C.1) / 2 = B.1 + t) ∧ ((A.2 + C.2) / 2 = B.2 + 3 * t)

/-- The area of a right triangle ABC with the given properties is 3750/59 -/
theorem right_triangle_area (t : RightTriangle) : 
  abs ((t.A.1 - t.C.1) * (t.B.2 - t.C.2) - (t.B.1 - t.C.1) * (t.A.2 - t.C.2)) / 2 = 3750 / 59 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l261_26192
