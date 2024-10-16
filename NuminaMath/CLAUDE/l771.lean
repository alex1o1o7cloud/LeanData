import Mathlib

namespace NUMINAMATH_CALUDE_max_points_in_specific_tournament_l771_77186

/-- Represents a football tournament with given number of teams. -/
structure Tournament where
  num_teams : ℕ
  points_per_win : ℕ
  points_per_draw : ℕ
  points_per_loss : ℕ

/-- The maximum number of points each team can achieve in the tournament. -/
def max_points_per_team (t : Tournament) : ℕ :=
  sorry

/-- Theorem stating the maximum points per team in a specific tournament setup. -/
theorem max_points_in_specific_tournament :
  ∃ (t : Tournament),
    t.num_teams = 10 ∧
    t.points_per_win = 3 ∧
    t.points_per_draw = 1 ∧
    t.points_per_loss = 0 ∧
    max_points_per_team t = 13 :=
  sorry

end NUMINAMATH_CALUDE_max_points_in_specific_tournament_l771_77186


namespace NUMINAMATH_CALUDE_fifth_fibonacci_is_eight_l771_77110

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | k + 2 => fibonacci k + fibonacci (k + 1)

theorem fifth_fibonacci_is_eight :
  fibonacci 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fifth_fibonacci_is_eight_l771_77110


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l771_77155

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; 6, 3]
  Matrix.det A = 33 := by sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l771_77155


namespace NUMINAMATH_CALUDE_shooting_guard_footage_l771_77166

/-- Represents the duration of footage for each player in seconds -/
structure PlayerFootage where
  pointGuard : ℕ
  shootingGuard : ℕ
  smallForward : ℕ
  powerForward : ℕ
  center : ℕ

/-- The total number of players -/
def totalPlayers : ℕ := 5

/-- The average footage duration per player in seconds -/
def averageFootage : ℕ := 120

theorem shooting_guard_footage (footage : PlayerFootage) 
  (h1 : footage.pointGuard = 130)
  (h2 : footage.smallForward = 85)
  (h3 : footage.powerForward = 60)
  (h4 : footage.center = 180)
  (h5 : totalPlayers * averageFootage = 
        footage.pointGuard + footage.shootingGuard + footage.smallForward + 
        footage.powerForward + footage.center) : 
  footage.shootingGuard = 145 := by
  sorry

#check shooting_guard_footage

end NUMINAMATH_CALUDE_shooting_guard_footage_l771_77166


namespace NUMINAMATH_CALUDE_parabola_vertex_sum_max_l771_77133

theorem parabola_vertex_sum_max (a T : ℤ) (h1 : T ≠ 0) : 
  (∃ b c : ℝ, ∀ x y : ℝ, 
    (y = a * x^2 + b * x + c ↔ 
      (x = 0 ∧ y = 0) ∨ 
      (x = 2 * T ∧ y = 0) ∨ 
      (x = 2 * T + 1 ∧ y = 36))) →
  (let N := T - a * T^2
   ∀ T' a' : ℤ, T' ≠ 0 → 
    (∃ b' c' : ℝ, ∀ x y : ℝ,
      (y = a' * x^2 + b' * x + c' ↔ 
        (x = 0 ∧ y = 0) ∨ 
        (x = 2 * T' ∧ y = 0) ∨ 
        (x = 2 * T' + 1 ∧ y = 36))) →
    T' - a' * T'^2 ≤ N) →
  N = -14 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_sum_max_l771_77133


namespace NUMINAMATH_CALUDE_min_v_value_l771_77191

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the translated function g
def g (x u v : ℝ) : ℝ := (x-u)^3 - 3*(x-u) - v

-- Theorem statement
theorem min_v_value (u : ℝ) (h : u > 0) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = g x₁ u v → f x₂ ≠ g x₂ u v) →
  v ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_v_value_l771_77191


namespace NUMINAMATH_CALUDE_height_difference_l771_77109

-- Define variables for heights
variable (h_A h_B h_D h_E h_F h_G : ℝ)

-- Define the conditions
def condition1 : Prop := h_A - h_D = 4.5
def condition2 : Prop := h_E - h_D = -1.7
def condition3 : Prop := h_F - h_E = -0.8
def condition4 : Prop := h_G - h_F = 1.9
def condition5 : Prop := h_B - h_G = 3.6

-- Theorem statement
theorem height_difference 
  (c1 : condition1 h_A h_D)
  (c2 : condition2 h_E h_D)
  (c3 : condition3 h_F h_E)
  (c4 : condition4 h_G h_F)
  (c5 : condition5 h_B h_G) :
  h_A > h_B :=
by sorry

end NUMINAMATH_CALUDE_height_difference_l771_77109


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l771_77137

/-- 
Given a quadratic equation x^2 - 4x - m = 0 with two equal real roots,
prove that m = -4.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x - m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y - m = 0 → y = x) → 
  m = -4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l771_77137


namespace NUMINAMATH_CALUDE_mayas_books_pages_l771_77115

/-- Proves that if Maya read 5 books last week, read twice as much this week, 
    and read a total of 4500 pages this week, then each book had 450 pages. -/
theorem mayas_books_pages (books_last_week : ℕ) (pages_this_week : ℕ) 
  (h1 : books_last_week = 5)
  (h2 : pages_this_week = 4500) :
  (pages_this_week / (2 * books_last_week) : ℚ) = 450 := by
  sorry

#check mayas_books_pages

end NUMINAMATH_CALUDE_mayas_books_pages_l771_77115


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l771_77124

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x - 5| = 3*x + 2) ↔ (x = 3/5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l771_77124


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l771_77111

theorem tan_sum_pi_twelfths : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 * Real.sqrt 2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l771_77111


namespace NUMINAMATH_CALUDE_vector_problems_l771_77126

def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

theorem vector_problems (x : ℝ) :
  (∃ k : ℝ, a x = k • b x → ‖a x - b x‖ = 2 ∨ ‖a x - b x‖ = 2 * Real.sqrt 5) ∧
  (0 < (a x).1 * (b x).1 + (a x).2 * (b x).2 → x ∈ Set.Ioo (-1) 0 ∪ Set.Ioo 0 3) ∧
  (‖a x‖ = 2 → ∃ c : ℝ × ℝ, ‖c‖ = 1 ∧ (a x).1 * c.1 + (a x).2 * c.2 = 0 ∧
    ((c.1 = Real.sqrt 3 / 2 ∧ c.2 = -1/2) ∨
     (c.1 = -Real.sqrt 3 / 2 ∧ c.2 = 1/2) ∨
     (c.1 = Real.sqrt 3 / 2 ∧ c.2 = 1/2) ∨
     (c.1 = -Real.sqrt 3 / 2 ∧ c.2 = -1/2))) :=
by sorry


end NUMINAMATH_CALUDE_vector_problems_l771_77126


namespace NUMINAMATH_CALUDE_calculate_expression_l771_77198

theorem calculate_expression (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (8 * y) * (1 / (4 * y)^3) = 9/4 * y := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l771_77198


namespace NUMINAMATH_CALUDE_sum_divides_10n_count_l771_77193

theorem sum_divides_10n_count : 
  (∃ (S : Finset ℕ), S.card = 5 ∧ 
    (∀ n : ℕ, n > 0 → (n ∈ S ↔ (10 * n) % ((n * (n + 1)) / 2) = 0))) :=
sorry

end NUMINAMATH_CALUDE_sum_divides_10n_count_l771_77193


namespace NUMINAMATH_CALUDE_fraction_equality_l771_77195

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l771_77195


namespace NUMINAMATH_CALUDE_tissues_boxes_calculation_l771_77180

/-- Given the number of tissues per box, the number of used tissues, and the number of remaining tissues,
    prove that the number of boxes bought is equal to the sum of used and remaining tissues
    divided by the number of tissues per box. -/
theorem tissues_boxes_calculation (tissues_per_box used_tissues remaining_tissues : ℕ)
    (h_tissues_per_box : tissues_per_box > 0) :
    (used_tissues + remaining_tissues) / tissues_per_box =
    (used_tissues + remaining_tissues) / tissues_per_box := by
  sorry

end NUMINAMATH_CALUDE_tissues_boxes_calculation_l771_77180


namespace NUMINAMATH_CALUDE_buddys_gym_class_size_l771_77151

theorem buddys_gym_class_size (group1 : ℕ) (group2 : ℕ) 
  (h1 : group1 = 34) (h2 : group2 = 37) : group1 + group2 = 71 := by
  sorry

end NUMINAMATH_CALUDE_buddys_gym_class_size_l771_77151


namespace NUMINAMATH_CALUDE_train_length_problem_l771_77100

/-- The length of two trains passing each other -/
theorem train_length_problem (speed_kmh : ℝ) (crossing_time : ℝ) : 
  speed_kmh = 18 ∧ crossing_time = 24 →
  ∃ (train_length : ℝ), train_length = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l771_77100


namespace NUMINAMATH_CALUDE_decimal_sum_l771_77105

theorem decimal_sum : (0.35 : ℚ) + 0.048 + 0.0072 = 0.4052 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l771_77105


namespace NUMINAMATH_CALUDE_sum_of_22_and_62_l771_77150

theorem sum_of_22_and_62 : 22 + 62 = 84 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_22_and_62_l771_77150


namespace NUMINAMATH_CALUDE_extended_equilateral_area_ratio_l771_77108

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Extends a triangle by a factor along each side -/
def extendTriangle (t : Triangle) (factor : ℝ) : Triangle := sorry

/-- Main theorem: The area of an extended equilateral triangle is 9 times the original -/
theorem extended_equilateral_area_ratio (t : Triangle) :
  isEquilateral t →
  area (extendTriangle t 3) = 9 * area t := by sorry

end NUMINAMATH_CALUDE_extended_equilateral_area_ratio_l771_77108


namespace NUMINAMATH_CALUDE_second_distribution_boys_l771_77199

theorem second_distribution_boys (total_amount : ℕ) (first_boys : ℕ) (difference : ℕ) : 
  total_amount = 5040 →
  first_boys = 14 →
  difference = 80 →
  ∃ (second_boys : ℕ), 
    (total_amount / first_boys = total_amount / second_boys + difference) ∧
    second_boys = 18 :=
by sorry

end NUMINAMATH_CALUDE_second_distribution_boys_l771_77199


namespace NUMINAMATH_CALUDE_max_value_theorem_l771_77194

theorem max_value_theorem (x y z : ℝ) 
  (hx : 0 < x ∧ x < Real.sqrt 5) 
  (hy : 0 < y ∧ y < Real.sqrt 5) 
  (hz : 0 < z ∧ z < Real.sqrt 5) 
  (h_sum : x^4 + y^4 + z^4 ≥ 27) :
  (x / (x^2 - 5)) + (y / (y^2 - 5)) + (z / (z^2 - 5)) ≤ -3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l771_77194


namespace NUMINAMATH_CALUDE_sinA_cosA_rational_l771_77153

/-- An isosceles triangle with integer base and height -/
structure IsoscelesTriangle where
  base : ℤ
  height : ℤ

/-- The sine of angle A in an isosceles triangle -/
def sinA (t : IsoscelesTriangle) : ℚ :=
  4 * t.base * t.height^2 / (4 * t.height^2 + t.base^2)

/-- The cosine of angle A in an isosceles triangle -/
def cosA (t : IsoscelesTriangle) : ℚ :=
  (4 * t.height^2 - t.base^2) / (4 * t.height^2 + t.base^2)

/-- Theorem: In an isosceles triangle with integer base and height, 
    both sin A and cos A are rational numbers -/
theorem sinA_cosA_rational (t : IsoscelesTriangle) : 
  (∃ q : ℚ, sinA t = q) ∧ (∃ q : ℚ, cosA t = q) := by
  sorry

end NUMINAMATH_CALUDE_sinA_cosA_rational_l771_77153


namespace NUMINAMATH_CALUDE_circle_symmetry_l771_77181

/-- Definition of the first circle C₁ -/
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

/-- Definition of the second circle C₂ -/
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

/-- Definition of the line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Function to check if two points are symmetric with respect to the line -/
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  symmetry_line ((x1 + x2) / 2) ((y1 + y2) / 2) ∧
  x2 - x1 = y2 - y1

/-- Theorem stating that C₂ is symmetric to C₁ with respect to the given line -/
theorem circle_symmetry :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle_C1 x1 y1 →
    circle_C2 x2 y2 →
    symmetric_points x1 y1 x2 y2 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_symmetry_l771_77181


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l771_77140

-- Define the polynomial
def P (k X : ℝ) : ℝ := X^4 + 2*X^3 + (2 + 2*k)*X^2 + (1 + 2*k)*X + 2*k

-- Define the theorem
theorem sum_of_squares_of_roots (k : ℝ) :
  (∃ r₁ r₂ : ℝ, P k r₁ = 0 ∧ P k r₂ = 0 ∧ r₁ * r₂ = -2013) →
  (∃ r₁ r₂ : ℝ, P k r₁ = 0 ∧ P k r₂ = 0 ∧ r₁^2 + r₂^2 = 4027) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l771_77140


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l771_77138

def x : ℕ := 5 * 27 * 64

theorem smallest_y_for_perfect_fourth_power (y : ℕ) : 
  y > 0 ∧ 
  (∀ z : ℕ, z > 0 ∧ z < y → ¬ ∃ n : ℕ, x * z = n^4) ∧
  (∃ n : ℕ, x * y = n^4) → 
  y = 1500 := by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_fourth_power_l771_77138


namespace NUMINAMATH_CALUDE_charlie_win_probability_l771_77122

/-- The probability of rolling a six on a standard six-sided die -/
def probSix : ℚ := 1 / 6

/-- The probability of not rolling a six on a standard six-sided die -/
def probNotSix : ℚ := 5 / 6

/-- The number of players in the game -/
def numPlayers : ℕ := 3

/-- The probability that Charlie (the third player) wins the dice game -/
def probCharlieWins : ℚ := 125 / 546

theorem charlie_win_probability :
  probCharlieWins = probSix * (probNotSix^numPlayers / (1 - probNotSix^numPlayers)) :=
sorry

end NUMINAMATH_CALUDE_charlie_win_probability_l771_77122


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l771_77147

theorem parabola_line_intersection (α : Real) :
  (∃! x, 3 * x^2 + 1 = 4 * Real.sin α * x) →
  0 < α ∧ α < π / 2 →
  α = π / 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l771_77147


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l771_77157

theorem smallest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  max a (max b c) = 120 →  -- The largest angle is 120°
  b / c = 3 / 2 →  -- Ratio of the other two angles is 3:2
  b > c →  -- Ensure b is the middle angle and c is the smallest
  c = 24 :=  -- The smallest angle is 24°
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l771_77157


namespace NUMINAMATH_CALUDE_min_prime_factorization_sum_l771_77163

theorem min_prime_factorization_sum (x y a b : ℕ+) (e f : ℕ) :
  5 * x^7 = 13 * y^11 →
  x = a^e * b^f →
  a.val.Prime ∧ b.val.Prime →
  a ≠ b →
  a + b + e + f = 25 :=
sorry

end NUMINAMATH_CALUDE_min_prime_factorization_sum_l771_77163


namespace NUMINAMATH_CALUDE_find_t_value_l771_77154

theorem find_t_value (s t : ℚ) 
  (eq1 : 15 * s + 7 * t = 210)
  (eq2 : t = 3 * s - 1) : 
  t = 205 / 12 := by
  sorry

end NUMINAMATH_CALUDE_find_t_value_l771_77154


namespace NUMINAMATH_CALUDE_total_sum_lent_l771_77179

/-- Proves that the total sum lent is Rs. 2743 given the problem conditions -/
theorem total_sum_lent (first_part second_part : ℕ) : 
  (first_part * 3 * 8 = second_part * 5 * 3) → -- Interest equality condition
  (second_part = 1688) →                      -- Second part value
  (first_part + second_part = 2743) :=        -- Total sum to prove
by
  sorry

#check total_sum_lent

end NUMINAMATH_CALUDE_total_sum_lent_l771_77179


namespace NUMINAMATH_CALUDE_root_of_quadratic_l771_77104

theorem root_of_quadratic (x : ℝ) : 
  x = (-25 + Real.sqrt 361) / 12 → 6 * x^2 + 25 * x + 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_of_quadratic_l771_77104


namespace NUMINAMATH_CALUDE_min_value_theorem_l771_77112

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (ht : t > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p * q * r * s = 16)
  (h2 : t * u * v * w = 25)
  (h3 : p * t = q * u)
  (h4 : p * t = r * v)
  (h5 : p * t = s * w) :
  (∀ x : ℝ, (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 80) ∧
  (∃ x : ℝ, (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 = 80) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l771_77112


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l771_77125

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℚ := sorry

/-- The first term of the arithmetic sequence -/
def a₁ : ℚ := sorry

/-- The common difference of the arithmetic sequence -/
def d : ℚ := sorry

/-- Properties of the arithmetic sequence -/
axiom sum_formula (n : ℕ) : S n = n * a₁ + (n * (n - 1) / 2) * d

/-- Given conditions -/
axiom condition_1 : S 10 = 16
axiom condition_2 : S 100 - S 90 = 24

/-- Theorem to prove -/
theorem arithmetic_sequence_sum : S 100 = 200 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l771_77125


namespace NUMINAMATH_CALUDE_tournament_games_played_21_teams_l771_77132

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- The number of games played in a single-elimination tournament. -/
def games_played (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 21 teams and no ties,
    20 games are played before a winner is declared. -/
theorem tournament_games_played_21_teams :
  ∀ t : Tournament, t.num_teams = 21 → t.no_ties = true →
    games_played t = 20 := by
  sorry


end NUMINAMATH_CALUDE_tournament_games_played_21_teams_l771_77132


namespace NUMINAMATH_CALUDE_decagon_perimeter_decagon_perimeter_is_35_l771_77172

theorem decagon_perimeter : ℕ → ℕ → ℕ → ℕ
  | n, a, b =>
    if n = 10 ∧ a = 3 ∧ b = 4
    then 5 * a + 5 * b
    else 0

theorem decagon_perimeter_is_35 :
  decagon_perimeter 10 3 4 = 35 :=
by sorry

end NUMINAMATH_CALUDE_decagon_perimeter_decagon_perimeter_is_35_l771_77172


namespace NUMINAMATH_CALUDE_pittsburgh_police_stations_count_l771_77128

/-- The number of police stations in Pittsburgh -/
def pittsburgh_police_stations : ℕ := 20

/-- The number of stores in Pittsburgh -/
def pittsburgh_stores : ℕ := 2000

/-- The number of hospitals in Pittsburgh -/
def pittsburgh_hospitals : ℕ := 500

/-- The number of schools in Pittsburgh -/
def pittsburgh_schools : ℕ := 200

/-- The total number of buildings in the new city -/
def new_city_total_buildings : ℕ := 2175

theorem pittsburgh_police_stations_count :
  pittsburgh_police_stations = 20 :=
by
  have new_city_stores : ℕ := pittsburgh_stores / 2
  have new_city_hospitals : ℕ := pittsburgh_hospitals * 2
  have new_city_schools : ℕ := pittsburgh_schools - 50
  have new_city_police_stations : ℕ := pittsburgh_police_stations + 5
  
  have : new_city_stores + new_city_hospitals + new_city_schools + new_city_police_stations = new_city_total_buildings :=
    by sorry
  
  sorry -- The proof goes here

end NUMINAMATH_CALUDE_pittsburgh_police_stations_count_l771_77128


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l771_77149

def team_size : ℕ := 13
def lineup_size : ℕ := 6

theorem starting_lineup_combinations : 
  (team_size * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4) * (team_size - 5)) = 1027680 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l771_77149


namespace NUMINAMATH_CALUDE_fireworks_display_l771_77187

theorem fireworks_display (year_digits : ℕ) (phrase_letters : ℕ) 
  (additional_boxes : ℕ) (fireworks_per_box : ℕ) (fireworks_per_letter : ℕ) 
  (total_fireworks : ℕ) :
  year_digits = 4 →
  phrase_letters = 12 →
  additional_boxes = 50 →
  fireworks_per_box = 8 →
  fireworks_per_letter = 5 →
  total_fireworks = 484 →
  ∃ (fireworks_per_digit : ℕ),
    year_digits * fireworks_per_digit + 
    phrase_letters * fireworks_per_letter + 
    additional_boxes * fireworks_per_box = total_fireworks ∧
    fireworks_per_digit = 6 :=
by sorry

end NUMINAMATH_CALUDE_fireworks_display_l771_77187


namespace NUMINAMATH_CALUDE_max_value_theorem_l771_77182

theorem max_value_theorem (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ (max : ℝ), max = 7/5 ∧ ∀ (x y : ℝ), 
    x + y - 2 ≥ 0 → y - x - 1 ≤ 0 → x ≤ 1 → 
    (x + 2*y) / (2*x + y) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l771_77182


namespace NUMINAMATH_CALUDE_boat_length_boat_length_is_four_l771_77119

/-- The length of a boat given specific conditions --/
theorem boat_length (breadth : ℝ) (sink_depth : ℝ) (man_mass : ℝ) 
                    (water_density : ℝ) (gravity : ℝ) : ℝ :=
  let boat_length := 4
  let volume_displaced := man_mass * gravity / (water_density * gravity)
  let calculated_length := volume_displaced / (breadth * sink_depth)
  
  -- Assumptions
  have h1 : breadth = 3 := by sorry
  have h2 : sink_depth = 0.01 := by sorry
  have h3 : man_mass = 120 := by sorry
  have h4 : water_density = 1000 := by sorry
  have h5 : gravity = 9.8 := by sorry
  
  -- Proof that the calculated length equals the given boat length
  have h6 : calculated_length = boat_length := by sorry
  
  boat_length

/-- Main theorem stating the boat length is 4 meters --/
theorem boat_length_is_four : 
  boat_length 3 0.01 120 1000 9.8 = 4 := by sorry

end NUMINAMATH_CALUDE_boat_length_boat_length_is_four_l771_77119


namespace NUMINAMATH_CALUDE_matrix_with_unequal_rank_and_square_rank_l771_77174

theorem matrix_with_unequal_rank_and_square_rank
  (n : ℕ)
  (h_n : n ≥ 2)
  (A : Matrix (Fin n) (Fin n) ℂ)
  (h_rank : Matrix.rank A ≠ Matrix.rank (A * A)) :
  ∃ (B : Matrix (Fin n) (Fin n) ℂ), B ≠ 0 ∧ A * B = 0 ∧ B * A = 0 ∧ B * B = 0 := by
sorry

end NUMINAMATH_CALUDE_matrix_with_unequal_rank_and_square_rank_l771_77174


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l771_77139

theorem partial_fraction_decomposition :
  ∃! (A B C D : ℚ),
    ∀ (x : ℝ), x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ -1 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 5) * (x + 1)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x + 1) ∧
      A = -5/9 ∧ B = 0 ∧ C = 4/9 ∧ D = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l771_77139


namespace NUMINAMATH_CALUDE_circle_c_equation_l771_77118

/-- A circle C with the following properties:
    1. Its center is on the line x - 3y = 0
    2. It is tangent to the negative half-axis of the y-axis
    3. The chord cut by C on the x-axis is 4√2 in length -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.1 - 3 * center.2 = 0
  tangent_to_negative_y : center.2 < 0 ∧ radius = -center.2
  chord_length : 4 * Real.sqrt 2 = 2 * Real.sqrt (2 * radius * center.1)

/-- The equation of circle C is (x + 3)² + (y + 1)² = 9 -/
theorem circle_c_equation (c : CircleC) : 
  ∀ x y : ℝ, (x + 3)^2 + (y + 1)^2 = 9 ↔ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_c_equation_l771_77118


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l771_77158

-- Define the function
def f (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem function_satisfies_conditions :
  (f 1 = 3) ∧ (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :=
by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l771_77158


namespace NUMINAMATH_CALUDE_divisors_of_40_and_72_l771_77159

theorem divisors_of_40_and_72 : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ 40 % n = 0 ∧ 72 % n = 0) ∧ 
  (∀ n : ℕ, n > 0 ∧ 40 % n = 0 ∧ 72 % n = 0 → n ∈ S) ∧
  Finset.card S = 4 := by
sorry

end NUMINAMATH_CALUDE_divisors_of_40_and_72_l771_77159


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l771_77176

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  sum_property : a 2 + a 8 = 15
  product_property : a 3 * a 7 = 36

/-- The theorem stating the possible values of a_19 / a_13 -/
theorem geometric_sequence_ratio 
  (seq : GeometricSequence) : 
  seq.a 19 / seq.a 13 = 1/4 ∨ seq.a 19 / seq.a 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l771_77176


namespace NUMINAMATH_CALUDE_population_difference_specific_population_difference_l771_77185

/-- The population difference between two cities with different volumes, given a constant population density. -/
theorem population_difference (density : ℕ) (volume1 volume2 : ℕ) :
  density * (volume1 - volume2) = density * volume1 - density * volume2 := by
  sorry

/-- The population difference between two specific cities. -/
theorem specific_population_difference :
  let density : ℕ := 80
  let volume1 : ℕ := 9000
  let volume2 : ℕ := 6400
  density * (volume1 - volume2) = 208000 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_specific_population_difference_l771_77185


namespace NUMINAMATH_CALUDE_max_stickers_for_one_player_l771_77196

theorem max_stickers_for_one_player (n : ℕ) (avg : ℕ) (min_stickers : ℕ) 
  (h1 : n = 22)
  (h2 : avg = 4)
  (h3 : min_stickers = 1) :
  ∃ (max_stickers : ℕ), max_stickers = n * avg - (n - 1) * min_stickers ∧ max_stickers = 67 := by
  sorry

end NUMINAMATH_CALUDE_max_stickers_for_one_player_l771_77196


namespace NUMINAMATH_CALUDE_games_lost_l771_77130

theorem games_lost (total_games won_games : ℕ) 
  (h1 : total_games = 16) 
  (h2 : won_games = 12) : 
  total_games - won_games = 4 := by
sorry

end NUMINAMATH_CALUDE_games_lost_l771_77130


namespace NUMINAMATH_CALUDE_parallel_vectors_result_obtuse_triangle_result_l771_77121

noncomputable section

def m (x : ℝ) : ℝ × ℝ := (Real.cos x, 1)
def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 / 2)

def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = (k * w.1, k * w.2)

def f (x : ℝ) : ℝ := (m x).1^2 + (m x).2^2 - ((n x).1^2 + (n x).2^2)

theorem parallel_vectors_result (x : ℝ) (h : parallel (m x) (n x)) :
  (Real.sin x + Real.sqrt 3 * Real.cos x) / (Real.sqrt 3 * Real.sin x - Real.cos x) = 3 * Real.sqrt 3 :=
sorry

theorem obtuse_triangle_result (A B : ℝ) (hA : A > π / 2) (hC : Real.sin A = 1 / 2) :
  f A = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_result_obtuse_triangle_result_l771_77121


namespace NUMINAMATH_CALUDE_wanda_initial_blocks_l771_77103

/-- The number of blocks Theresa gave to Wanda -/
def blocks_from_theresa : ℕ := 79

/-- The total number of blocks Wanda has after receiving blocks from Theresa -/
def total_blocks : ℕ := 83

/-- The number of blocks Wanda had initially -/
def initial_blocks : ℕ := total_blocks - blocks_from_theresa

theorem wanda_initial_blocks :
  initial_blocks = 4 :=
by sorry

end NUMINAMATH_CALUDE_wanda_initial_blocks_l771_77103


namespace NUMINAMATH_CALUDE_hotel_price_per_night_l771_77136

def car_value : ℕ := 30000
def house_value : ℕ := 4 * car_value
def total_value : ℕ := 158000

theorem hotel_price_per_night :
  ∃ (price_per_night : ℕ), 
    car_value + house_value + 2 * price_per_night = total_value ∧
    price_per_night = 4000 :=
by sorry

end NUMINAMATH_CALUDE_hotel_price_per_night_l771_77136


namespace NUMINAMATH_CALUDE_sin_cos_eq_one_solutions_l771_77131

theorem sin_cos_eq_one_solutions (x : Real) :
  x ∈ Set.Icc 0 Real.pi →
  (Real.sin x + Real.cos x = 1) ↔ (x = 0 ∨ x = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_eq_one_solutions_l771_77131


namespace NUMINAMATH_CALUDE_complex_division_equality_l771_77142

theorem complex_division_equality : (1 - Complex.I) / Complex.I = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equality_l771_77142


namespace NUMINAMATH_CALUDE_rectangle_D_leftmost_l771_77145

-- Define the structure for a rectangle
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the sum of side labels for a rectangle
def sum_labels (r : Rectangle) : Int :=
  r.w + r.x + r.y + r.z

-- Define the five rectangles
def rectangle_A : Rectangle := ⟨3, 2, 5, 8⟩
def rectangle_B : Rectangle := ⟨2, 1, 4, 7⟩
def rectangle_C : Rectangle := ⟨4, 9, 6, 3⟩
def rectangle_D : Rectangle := ⟨8, 6, 5, 9⟩
def rectangle_E : Rectangle := ⟨10, 3, 8, 1⟩

-- Theorem: Rectangle D has the highest sum of side labels
theorem rectangle_D_leftmost :
  sum_labels rectangle_D > sum_labels rectangle_A ∧
  sum_labels rectangle_D > sum_labels rectangle_B ∧
  sum_labels rectangle_D > sum_labels rectangle_C ∧
  sum_labels rectangle_D > sum_labels rectangle_E :=
by sorry

end NUMINAMATH_CALUDE_rectangle_D_leftmost_l771_77145


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l771_77102

theorem cloth_sale_calculation (selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  selling_price = 8925 ∧ 
  profit_per_meter = 10 ∧ 
  cost_price_per_meter = 95 →
  (selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l771_77102


namespace NUMINAMATH_CALUDE_exists_geometric_progression_shift_l771_77135

/-- Given a sequence {a_n} defined by a_n = q * a_{n-1} + d where q ≠ 1,
    there exists a constant c such that b_n = a_n + c forms a geometric progression. -/
theorem exists_geometric_progression_shift 
  (q d : ℝ) (hq : q ≠ 1) (a : ℕ → ℝ) 
  (ha : ∀ n : ℕ, a (n + 1) = q * a n + d) :
  ∃ c : ℝ, ∃ r : ℝ, ∀ n : ℕ, a n + c = r^n * (a 0 + c) :=
sorry

end NUMINAMATH_CALUDE_exists_geometric_progression_shift_l771_77135


namespace NUMINAMATH_CALUDE_homework_time_difference_prove_homework_time_difference_l771_77190

/-- The difference in time taken to finish homework between Sarah and Samuel is 48 minutes -/
theorem homework_time_difference : ℝ → Prop :=
  fun difference =>
    let samuel_time : ℝ := 30  -- Samuel's time in minutes
    let sarah_time : ℝ := 1.3 * 60  -- Sarah's time converted to minutes
    difference = sarah_time - samuel_time ∧ difference = 48

/-- Proof of the homework time difference theorem -/
theorem prove_homework_time_difference : ∃ (difference : ℝ), homework_time_difference difference := by
  sorry

end NUMINAMATH_CALUDE_homework_time_difference_prove_homework_time_difference_l771_77190


namespace NUMINAMATH_CALUDE_focus_of_specific_parabola_l771_77107

/-- The focus of a parabola defined by y = ax^2 + bx + c -/
def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  sorry

theorem focus_of_specific_parabola :
  parabola_focus 9 6 (-4) = (-1/3, -59/12) := by
  sorry

end NUMINAMATH_CALUDE_focus_of_specific_parabola_l771_77107


namespace NUMINAMATH_CALUDE_fisherman_pelican_difference_l771_77178

/-- The number of fish caught by the pelican -/
def pelican_fish : ℕ := 13

/-- The number of fish caught by the kingfisher -/
def kingfisher_fish : ℕ := pelican_fish + 7

/-- The total number of fish caught by the pelican and kingfisher -/
def total_fish : ℕ := pelican_fish + kingfisher_fish

/-- The number of fish caught by the fisherman -/
def fisherman_fish : ℕ := 3 * total_fish

theorem fisherman_pelican_difference :
  fisherman_fish - pelican_fish = 86 := by sorry

end NUMINAMATH_CALUDE_fisherman_pelican_difference_l771_77178


namespace NUMINAMATH_CALUDE_ellipse_and_point_G_l771_77141

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given an ellipse C and points on it, prove the equation and find point G -/
theorem ellipse_and_point_G (C : Ellipse) 
  (h_triangle_area : (1/2) * C.a * (C.a^2 - C.b^2).sqrt * C.a = 4 * Real.sqrt 3)
  (B : Point) (h_B_on_C : B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1)
  (h_B_nonzero : B.x * B.y ≠ 0)
  (A : Point) (h_A : A = ⟨0, 2 * Real.sqrt 3⟩)
  (D E : Point) (h_D : D.y = 0) (h_E : E.y = 0)
  (h_collinear_ABD : (A.y - D.y) / (A.x - D.x) = (B.y - D.y) / (B.x - D.x))
  (h_collinear_ABE : (A.y - E.y) / (A.x - E.x) = (B.y + E.y) / (B.x - E.x))
  (G : Point) (h_G : G.x = 0)
  (h_angle_equal : (G.y / D.x)^2 = (G.y / E.x)^2) :
  (C.a = 4 ∧ C.b = 2 * Real.sqrt 3) ∧ 
  (G.y = 4 ∨ G.y = -4) := by sorry

end NUMINAMATH_CALUDE_ellipse_and_point_G_l771_77141


namespace NUMINAMATH_CALUDE_x_squared_gt_one_necessary_not_sufficient_l771_77167

theorem x_squared_gt_one_necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_gt_one_necessary_not_sufficient_l771_77167


namespace NUMINAMATH_CALUDE_jorge_ticket_cost_l771_77162

/-- Calculates the total cost of tickets after all discounts --/
def total_cost_after_discounts (adult_tickets senior_tickets child_tickets : ℕ)
  (adult_price senior_price child_price : ℚ)
  (tier1_threshold tier2_threshold tier3_threshold : ℚ)
  (tier1_adult_discount tier1_senior_discount : ℚ)
  (tier2_adult_discount tier2_senior_discount : ℚ)
  (tier3_adult_discount tier3_senior_discount : ℚ)
  (extra_discount_per_50 max_extra_discount : ℚ) : ℚ :=
  sorry

/-- The theorem to be proved --/
theorem jorge_ticket_cost :
  total_cost_after_discounts 10 8 6 12 8 6 100 200 300
    0.1 0.05 0.2 0.1 0.3 0.15 0.05 0.15 = 161.16 := by
  sorry

end NUMINAMATH_CALUDE_jorge_ticket_cost_l771_77162


namespace NUMINAMATH_CALUDE_streamer_hourly_rate_l771_77165

/-- A streamer's weekly schedule and earnings --/
structure StreamerSchedule where
  daysOff : ℕ
  hoursPerStreamDay : ℕ
  weeklyEarnings : ℕ

/-- Calculate the hourly rate of a streamer --/
def hourlyRate (s : StreamerSchedule) : ℚ :=
  s.weeklyEarnings / ((7 - s.daysOff) * s.hoursPerStreamDay)

/-- Theorem stating that given the specific conditions, the hourly rate is $10 --/
theorem streamer_hourly_rate :
  let s : StreamerSchedule := {
    daysOff := 3,
    hoursPerStreamDay := 4,
    weeklyEarnings := 160
  }
  hourlyRate s = 10 := by
  sorry

end NUMINAMATH_CALUDE_streamer_hourly_rate_l771_77165


namespace NUMINAMATH_CALUDE_power_comparison_l771_77173

theorem power_comparison : (2 : ℝ)^30 < 10^10 ∧ 10^10 < 5^15 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l771_77173


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l771_77116

theorem complex_magnitude_equality (s : ℝ) (hs : s > 0) :
  Complex.abs (-3 + s * Complex.I) = 2 * Real.sqrt 10 → s = Real.sqrt 31 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l771_77116


namespace NUMINAMATH_CALUDE_three_digit_cubes_divisible_by_27_l771_77161

theorem three_digit_cubes_divisible_by_27 :
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ 
    (100 ≤ n^3 ∧ n^3 ≤ 999 ∧ n^3 % 27 = 0)) ∧
  (∃ s : Finset ℕ, (∀ n : ℕ, n ∈ s ↔ 
    (100 ≤ n^3 ∧ n^3 ≤ 999 ∧ n^3 % 27 = 0)) ∧ 
    s.card = 2) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_cubes_divisible_by_27_l771_77161


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l771_77184

theorem product_of_three_numbers (x y z : ℝ) : 
  x + y + z = 30 →
  x = 3 * ((y + z) - 2) →
  y = 4 * z - 1 →
  x * y * z = 294 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l771_77184


namespace NUMINAMATH_CALUDE_plates_for_matt_l771_77120

/-- The number of plates needed for a week under specific dining conditions -/
def plates_needed (days_with_two : Nat) (days_with_four : Nat) (plates_per_person_two : Nat) (plates_per_person_four : Nat) : Nat :=
  (days_with_two * 2 * plates_per_person_two) + (days_with_four * 4 * plates_per_person_four)

theorem plates_for_matt : plates_needed 3 4 1 2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_plates_for_matt_l771_77120


namespace NUMINAMATH_CALUDE_max_renovation_days_l771_77192

def turnkey_cost : ℕ := 50000
def materials_cost : ℕ := 20000
def husband_wage : ℕ := 2000
def wife_wage : ℕ := 1500

theorem max_renovation_days : 
  ∃ n : ℕ, n = 8 ∧ 
  n * (husband_wage + wife_wage) + materials_cost ≤ turnkey_cost ∧
  (n + 1) * (husband_wage + wife_wage) + materials_cost > turnkey_cost :=
sorry

end NUMINAMATH_CALUDE_max_renovation_days_l771_77192


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l771_77143

theorem fraction_equality_sum (P Q : ℚ) : 
  (5 : ℚ) / 7 = P / 63 ∧ (5 : ℚ) / 7 = 140 / Q → P + Q = 241 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l771_77143


namespace NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l771_77171

/-- A right triangle with specific leg relationship and area -/
structure SpecialRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  leg_relationship : longer_leg = 3 * shorter_leg - 3
  area_condition : (1 / 2) * shorter_leg * longer_leg = 84
  right_angle : shorter_leg ^ 2 + longer_leg ^ 2 = hypotenuse ^ 2

/-- The hypotenuse of the special right triangle is √505 -/
theorem special_right_triangle_hypotenuse (t : SpecialRightTriangle) : t.hypotenuse = Real.sqrt 505 := by
  sorry

#check special_right_triangle_hypotenuse

end NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l771_77171


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l771_77189

theorem opposite_of_negative_2023 : 
  (-(- 2023 : ℤ)) = (2023 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l771_77189


namespace NUMINAMATH_CALUDE_equation_solution_l771_77197

theorem equation_solution : ∃ x : ℚ, (5 * (x + 30) / 3 = (4 - 3 * x) / 7) ∧ (x = -519 / 22) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l771_77197


namespace NUMINAMATH_CALUDE_last_digit_largest_known_prime_l771_77123

/-- The last digit of 2^216091 - 1 is 7 -/
theorem last_digit_largest_known_prime : 
  (2^216091 - 1) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_largest_known_prime_l771_77123


namespace NUMINAMATH_CALUDE_point_distance_product_l771_77127

theorem point_distance_product : 
  ∀ y₁ y₂ : ℝ,
  (((1 : ℝ) - 5)^2 + (y₁ - 2)^2 = 12^2) →
  (((1 : ℝ) - 5)^2 + (y₂ - 2)^2 = 12^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -28 :=
by
  sorry

end NUMINAMATH_CALUDE_point_distance_product_l771_77127


namespace NUMINAMATH_CALUDE_linear_function_composition_l771_77188

theorem linear_function_composition (a b : ℝ) :
  (∀ x : ℝ, (fun x => 3 * ((fun x => a * x + b) x) + 2) = (fun x => 4 * x - 1)) →
  a + b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_linear_function_composition_l771_77188


namespace NUMINAMATH_CALUDE_kendall_driving_distance_l771_77101

/-- The distance Kendall drove with her father -/
def distance_with_father (total_distance mother_distance : ℝ) : ℝ :=
  total_distance - mother_distance

/-- Theorem: Kendall drove 0.50 miles with her father -/
theorem kendall_driving_distance :
  distance_with_father 0.67 0.17 = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_kendall_driving_distance_l771_77101


namespace NUMINAMATH_CALUDE_min_sum_with_gcd_conditions_l771_77144

theorem min_sum_with_gcd_conditions :
  ∃ (a b c : ℕ+),
    (Nat.gcd a b > 1) ∧
    (Nat.gcd b c > 1) ∧
    (Nat.gcd c a > 1) ∧
    (Nat.gcd a (Nat.gcd b c) = 1) ∧
    (a + b + c = 31) ∧
    (∀ (x y z : ℕ+),
      (Nat.gcd x y > 1) →
      (Nat.gcd y z > 1) →
      (Nat.gcd z x > 1) →
      (Nat.gcd x (Nat.gcd y z) = 1) →
      (x + y + z ≥ 31)) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_gcd_conditions_l771_77144


namespace NUMINAMATH_CALUDE_expression_evaluation_l771_77146

theorem expression_evaluation :
  let x : ℕ := 3
  let y : ℕ := 2
  5 * x^y + 2 * y^x + x^2 * y^2 = 97 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l771_77146


namespace NUMINAMATH_CALUDE_min_value_of_a_plus_4b_l771_77169

theorem min_value_of_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) : 
  a + 4*b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_plus_4b_l771_77169


namespace NUMINAMATH_CALUDE_fencing_length_l771_77168

/-- Given a rectangular field with area 400 sq. ft and one side of 20 feet,
    prove that the fencing required for three sides is 60 feet. -/
theorem fencing_length (area : ℝ) (side : ℝ) (h1 : area = 400) (h2 : side = 20) :
  2 * (area / side) + side = 60 := by
  sorry

end NUMINAMATH_CALUDE_fencing_length_l771_77168


namespace NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l771_77164

theorem consecutive_four_product_plus_one_is_square (x : ℤ) :
  ∃ y : ℤ, x * (x + 1) * (x + 2) * (x + 3) + 1 = y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l771_77164


namespace NUMINAMATH_CALUDE_mikey_leaves_total_l771_77129

theorem mikey_leaves_total (initial_leaves additional_leaves : Float) : 
  initial_leaves = 356.0 → additional_leaves = 112.0 → 
  initial_leaves + additional_leaves = 468.0 := by
  sorry

end NUMINAMATH_CALUDE_mikey_leaves_total_l771_77129


namespace NUMINAMATH_CALUDE_watch_cost_price_l771_77156

/-- Proves that the cost price of a watch is 2000, given specific selling conditions. -/
theorem watch_cost_price : 
  ∀ (cost_price : ℝ),
  (cost_price * 0.8 + 520 = cost_price * 1.06) →
  cost_price = 2000 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l771_77156


namespace NUMINAMATH_CALUDE_number_equality_l771_77114

theorem number_equality (x : ℝ) : (30 / 100 : ℝ) * x = (25 / 100 : ℝ) * 45 → x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l771_77114


namespace NUMINAMATH_CALUDE_tom_annual_cost_l771_77183

/-- Calculates the total annual cost for Tom's sleep medication and doctor visits -/
def annual_cost (daily_pills : ℕ) (pill_cost : ℚ) (insurance_coverage : ℚ) 
                (yearly_doctor_visits : ℕ) (doctor_visit_cost : ℚ) : ℚ :=
  let daily_medication_cost := daily_pills * pill_cost
  let daily_out_of_pocket := daily_medication_cost * (1 - insurance_coverage)
  let annual_medication_cost := daily_out_of_pocket * 365
  let annual_doctor_cost := yearly_doctor_visits * doctor_visit_cost
  annual_medication_cost + annual_doctor_cost

/-- Theorem stating that Tom's annual cost for sleep medication and doctor visits is $1530 -/
theorem tom_annual_cost : 
  annual_cost 2 5 (4/5) 2 400 = 1530 := by
  sorry

end NUMINAMATH_CALUDE_tom_annual_cost_l771_77183


namespace NUMINAMATH_CALUDE_prob_one_girl_l771_77117

theorem prob_one_girl (p_two_boys p_two_girls : ℚ) 
  (h1 : p_two_boys = 1/3) 
  (h2 : p_two_girls = 2/15) : 
  1 - (p_two_boys + p_two_girls) = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_girl_l771_77117


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l771_77134

theorem quadratic_inequality_solution_condition (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (c > 0 ∧ c < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l771_77134


namespace NUMINAMATH_CALUDE_projectile_trajectory_l771_77160

/-- Represents the trajectory of a projectile --/
def trajectory (c g : ℝ) (x y : ℝ) : Prop :=
  x^2 = (2 * c^2 / g) * y

/-- Theorem stating that a projectile follows a parabolic trajectory --/
theorem projectile_trajectory (c g : ℝ) (hc : c > 0) (hg : g > 0) :
  ∀ x y : ℝ, trajectory c g x y ↔ x^2 = (2 * c^2 / g) * y :=
sorry

end NUMINAMATH_CALUDE_projectile_trajectory_l771_77160


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l771_77152

theorem equilateral_triangle_area (p : ℝ) (h : p > 0) :
  let perimeter := 3 * p
  let side_length := perimeter / 3
  let area := (Real.sqrt 3 / 4) * side_length ^ 2
  area = (Real.sqrt 3 / 4) * p ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l771_77152


namespace NUMINAMATH_CALUDE_sin_cos_product_l771_77177

theorem sin_cos_product (a : ℝ) (h : Real.sin (Real.pi - a) = -2 * Real.sin (Real.pi / 2 + a)) :
  Real.sin a * Real.cos a = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_l771_77177


namespace NUMINAMATH_CALUDE_count_valid_pairs_l771_77148

def valid_pair (b c : ℕ) : Prop :=
  1 ≤ b ∧ b ≤ 5 ∧ 1 ≤ c ∧ c ≤ 5 ∧ 
  (b * b : ℤ) - 4 * c ≤ 0 ∧ 
  (c * c : ℤ) - 4 * b ≤ 0

theorem count_valid_pairs : 
  ∃ (S : Finset (ℕ × ℕ)), S.card = 15 ∧ ∀ p, p ∈ S ↔ valid_pair p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l771_77148


namespace NUMINAMATH_CALUDE_five_hundredth_barrel_is_four_l771_77113

/-- The labeling function for barrels based on their position in the sequence -/
def barrel_label (n : ℕ) : ℕ :=
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | 6 => 10 - 6
  | 7 => 10 - 7
  | 0 => 10 - 8
  | _ => 0  -- This case should never occur due to properties of modulo

/-- The theorem stating that the 500th barrel is labeled 4 -/
theorem five_hundredth_barrel_is_four :
  barrel_label 500 = 4 := by
  sorry


end NUMINAMATH_CALUDE_five_hundredth_barrel_is_four_l771_77113


namespace NUMINAMATH_CALUDE_parallel_line_slope_l771_77175

theorem parallel_line_slope (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  let m := -a / b
  (∀ x y, a * x + b * y = c) → m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l771_77175


namespace NUMINAMATH_CALUDE_monotonic_cubic_range_l771_77170

/-- Given a function f(x) = -x^3 + ax^2 - x - 1 that is monotonic on ℝ,
    the range of the real number a is [-√3, √3]. -/
theorem monotonic_cubic_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ 
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_range_l771_77170


namespace NUMINAMATH_CALUDE_teacher_age_l771_77106

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) :
  num_students = 30 →
  student_avg_age = 14 →
  total_avg_age = 15 →
  (num_students : ℝ) * student_avg_age + 45 = (num_students + 1 : ℝ) * total_avg_age :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l771_77106
