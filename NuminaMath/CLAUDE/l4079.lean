import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_l4079_407914

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + a*|x - 1|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x ≥ 4} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} :=
by sorry

-- Part II
theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f a x ≥ |x - 2|) → a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_l4079_407914


namespace NUMINAMATH_CALUDE_gcd_bound_for_special_fraction_l4079_407996

theorem gcd_bound_for_special_fraction (a b : ℕ+) 
  (h : ∃ (k : ℤ), (a.1 + 1 : ℚ) / b.1 + (b.1 + 1 : ℚ) / a.1 = k) : 
  Nat.gcd a.1 b.1 ≤ Real.sqrt (a.1 + b.1) := by
  sorry

end NUMINAMATH_CALUDE_gcd_bound_for_special_fraction_l4079_407996


namespace NUMINAMATH_CALUDE_at_least_two_thirds_covered_l4079_407941

/-- Represents a chessboard with dominoes -/
structure ChessboardWithDominoes where
  m : Nat
  n : Nat
  dominoes : Finset (Nat × Nat)
  m_ge_two : m ≥ 2
  n_ge_two : n ≥ 2
  valid_placement : ∀ (i j : Nat), (i, j) ∈ dominoes → 
    (i < m ∧ j < n) ∧ (
      ((i + 1, j) ∈ dominoes ∧ (i + 1) < m) ∨
      ((i, j + 1) ∈ dominoes ∧ (j + 1) < n)
    )
  no_overlap : ∀ (i j k l : Nat), (i, j) ∈ dominoes → (k, l) ∈ dominoes → 
    (i = k ∧ j = l) ∨ (i + 1 = k ∧ j = l) ∨ (i = k ∧ j + 1 = l) ∨
    (k + 1 = i ∧ j = l) ∨ (k = i ∧ l + 1 = j)
  no_more_addable : ∀ (i j : Nat), i < m → j < n → 
    (i, j) ∉ dominoes → (i + 1 < m → (i + 1, j) ∈ dominoes) ∧
    (j + 1 < n → (i, j + 1) ∈ dominoes)

/-- The main theorem stating that at least 2/3 of the chessboard is covered by dominoes -/
theorem at_least_two_thirds_covered (board : ChessboardWithDominoes) : 
  (2 : ℚ) / 3 * (board.m * board.n : ℚ) ≤ (board.dominoes.card * 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_thirds_covered_l4079_407941


namespace NUMINAMATH_CALUDE_min_a_over_x_l4079_407984

theorem min_a_over_x (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100)
  (h : y^2 - 1 = a^2 * (x^2 - 1)) :
  ∀ k : ℚ, (k : ℝ) = a / x → k ≥ 2 ∧ ∃ a₀ x₀ y₀ : ℕ,
    a₀ > 100 ∧ x₀ > 100 ∧ y₀ > 100 ∧
    y₀^2 - 1 = a₀^2 * (x₀^2 - 1) ∧
    (a₀ : ℝ) / x₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_a_over_x_l4079_407984


namespace NUMINAMATH_CALUDE_inequality_solution_l4079_407946

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 4) ≥ 1) ↔ 
  (x ∈ Set.Ioc (-4) (-2) ∪ Set.Ioc (-2) (Real.sqrt 8)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4079_407946


namespace NUMINAMATH_CALUDE_average_monthly_balance_l4079_407900

def monthly_balances : List ℕ := [200, 250, 300, 350, 400]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℚ) = 300 := by sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l4079_407900


namespace NUMINAMATH_CALUDE_isosceles_similar_triangle_perimeter_l4079_407981

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Defines similarity between two triangles -/
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t2.a = k * t1.a ∧
    t2.b = k * t1.b ∧
    t2.c = k * t1.c

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

theorem isosceles_similar_triangle_perimeter :
  ∀ (t1 t2 : Triangle),
    t1.a = 15 ∧ t1.b = 30 ∧ t1.c = 30 →
    similar t1 t2 →
    t2.a = 75 →
    perimeter t2 = 375 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_similar_triangle_perimeter_l4079_407981


namespace NUMINAMATH_CALUDE_ferry_tourists_count_l4079_407978

/-- Calculates the total number of tourists transported by a ferry -/
def total_tourists (trips : ℕ) (initial_tourists : ℕ) (decrease : ℕ) : ℕ :=
  trips * (2 * initial_tourists - (trips - 1) * decrease) / 2

/-- Proves that the total number of tourists transported is 798 -/
theorem ferry_tourists_count :
  total_tourists 7 120 2 = 798 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourists_count_l4079_407978


namespace NUMINAMATH_CALUDE_expression_evaluation_l4079_407950

theorem expression_evaluation :
  (5^500 + 6^501)^2 - (5^500 - 6^501)^2 = 24 * 30^500 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4079_407950


namespace NUMINAMATH_CALUDE_average_speed_is_69_l4079_407977

def speeds : List ℝ := [90, 30, 60, 120, 45]
def total_time : ℝ := 5

theorem average_speed_is_69 :
  (speeds.sum / total_time) = 69 := by sorry

end NUMINAMATH_CALUDE_average_speed_is_69_l4079_407977


namespace NUMINAMATH_CALUDE_watermelon_sharing_l4079_407968

/-- The number of people that can share one watermelon -/
def people_per_watermelon : ℕ := 8

/-- The number of watermelons available -/
def num_watermelons : ℕ := 4

/-- The total number of people that can share the watermelons -/
def total_people : ℕ := people_per_watermelon * num_watermelons

theorem watermelon_sharing :
  total_people = 32 :=
by sorry

end NUMINAMATH_CALUDE_watermelon_sharing_l4079_407968


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l4079_407965

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = -103 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l4079_407965


namespace NUMINAMATH_CALUDE_print_shop_Y_charge_l4079_407945

/-- The charge per color copy at print shop X -/
def charge_X : ℚ := 1.25

/-- The number of copies being compared -/
def num_copies : ℕ := 80

/-- The additional charge at print shop Y for the given number of copies -/
def additional_charge : ℚ := 120

/-- The charge per color copy at print shop Y -/
def charge_Y : ℚ := (charge_X * num_copies + additional_charge) / num_copies

theorem print_shop_Y_charge : charge_Y = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_Y_charge_l4079_407945


namespace NUMINAMATH_CALUDE_remainder_theorem_l4079_407902

theorem remainder_theorem (r : ℤ) : (r^11 - 3) % (r - 2) = 2045 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4079_407902


namespace NUMINAMATH_CALUDE_probability_even_and_less_equal_three_l4079_407955

def dice_sides : ℕ := 6

def prob_even_first_die : ℚ := 1 / 2

def prob_less_equal_three_second_die : ℚ := 1 / 2

theorem probability_even_and_less_equal_three (independence : True) :
  prob_even_first_die * prob_less_equal_three_second_die = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_and_less_equal_three_l4079_407955


namespace NUMINAMATH_CALUDE_min_lateral_perimeter_is_six_l4079_407948

/-- Represents a rectangular parallelepiped with a square base -/
structure Parallelepiped where
  base_side : ℝ
  height : ℝ

/-- The volume of a parallelepiped -/
def volume (p : Parallelepiped) : ℝ :=
  p.base_side^2 * p.height

/-- The perimeter of a lateral face of a parallelepiped -/
def lateral_perimeter (p : Parallelepiped) : ℝ :=
  2 * p.base_side + 2 * p.height

/-- Theorem: The minimum perimeter of a lateral face among all rectangular
    parallelepipeds with volume 4 and square bases is 6 -/
theorem min_lateral_perimeter_is_six :
  ∀ p : Parallelepiped, volume p = 4 → lateral_perimeter p ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_min_lateral_perimeter_is_six_l4079_407948


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l4079_407973

theorem absolute_value_inequality (x y : ℝ) (h : x < y ∧ y < 0) :
  abs x > (abs (x + y)) / 2 ∧ (abs (x + y)) / 2 > abs y := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l4079_407973


namespace NUMINAMATH_CALUDE_max_segment_length_l4079_407916

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4
def line (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 2 = 0

-- Define the condition PB ≥ 2PA
def condition (x y : ℝ) : Prop :=
  ((x - 4)^2 + y^2 - 4) ≥ 4 * (x^2 + y^2 - 1)

-- Theorem statement
theorem max_segment_length :
  ∃ (E F : ℝ × ℝ),
    line E.1 E.2 ∧ line F.1 F.2 ∧
    (∀ (P : ℝ × ℝ), line P.1 P.2 →
      (E.1 ≤ P.1 ∧ P.1 ≤ F.1) → condition P.1 P.2) ∧
    Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) = 2 * Real.sqrt 39 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_segment_length_l4079_407916


namespace NUMINAMATH_CALUDE_point_on_transformed_plane_l4079_407929

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem to be proved -/
theorem point_on_transformed_plane :
  let A : Point3D := { x := 4, y := 3, z := 1 }
  let a : Plane := { a := 3, b := -4, c := 5, d := -6 }
  let k : ℝ := 5/6
  pointOnPlane A (transformPlane a k) := by
  sorry

end NUMINAMATH_CALUDE_point_on_transformed_plane_l4079_407929


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l4079_407907

theorem like_terms_exponent_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 3 * x^(2*m) * y^3 = -2 * x^2 * y^n) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l4079_407907


namespace NUMINAMATH_CALUDE_tan_sum_45_deg_l4079_407969

theorem tan_sum_45_deg (A B : Real) (h : A + B = Real.pi / 4) :
  (1 + Real.tan A) * (1 + Real.tan B) = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_45_deg_l4079_407969


namespace NUMINAMATH_CALUDE_cistern_length_is_8_l4079_407971

/-- Represents a cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the total wet surface area of a cistern --/
def totalWetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: The length of the cistern is 8 meters --/
theorem cistern_length_is_8 (c : Cistern) 
    (h_width : c.width = 4)
    (h_depth : c.depth = 1.25)
    (h_area : c.wetSurfaceArea = 62) :
    c.length = 8 := by
  sorry

#check cistern_length_is_8

end NUMINAMATH_CALUDE_cistern_length_is_8_l4079_407971


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l4079_407980

/-- The number of different books in the 'crazy silly school' series -/
def num_books : ℕ := sorry

/-- The number of different movies in the 'crazy silly school' series -/
def num_movies : ℕ := 11

/-- The number of books you have read -/
def books_read : ℕ := 13

/-- The number of movies you have watched -/
def movies_watched : ℕ := 12

theorem crazy_silly_school_series :
  (books_read = movies_watched + 1) →
  (num_books = 12) :=
by sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l4079_407980


namespace NUMINAMATH_CALUDE_initial_boys_on_slide_l4079_407976

theorem initial_boys_on_slide (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  additional = 13 → total = 35 → initial + additional = total → initial = 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_boys_on_slide_l4079_407976


namespace NUMINAMATH_CALUDE_distance_between_squares_l4079_407949

theorem distance_between_squares (small_perimeter : ℝ) (large_area : ℝ) : 
  small_perimeter = 8 →
  large_area = 25 →
  let small_side := small_perimeter / 4
  let large_side := Real.sqrt large_area
  let horizontal := small_side + large_side
  let vertical := large_side - small_side
  Real.sqrt (horizontal ^ 2 + vertical ^ 2) = Real.sqrt 58 := by
  sorry

#check distance_between_squares

end NUMINAMATH_CALUDE_distance_between_squares_l4079_407949


namespace NUMINAMATH_CALUDE_determinant_of_roots_l4079_407906

/-- Given a, b, c are roots of x^3 + px^2 + qx + r = 0, 
    the determinant of [[a, c, b], [c, b, a], [b, a, c]] is -c^3 + b^2c -/
theorem determinant_of_roots (p q r a b c : ℝ) : 
  a^3 + p*a^2 + q*a + r = 0 →
  b^3 + p*b^2 + q*b + r = 0 →
  c^3 + p*c^2 + q*c + r = 0 →
  Matrix.det !![a, c, b; c, b, a; b, a, c] = -c^3 + b^2*c := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_roots_l4079_407906


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l4079_407956

def A (c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; c, d]

theorem matrix_is_own_inverse (c d : ℝ) :
  A c d * A c d = 1 ↔ c = 3/2 ∧ d = -2 := by sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l4079_407956


namespace NUMINAMATH_CALUDE_total_blue_balloons_l4079_407957

theorem total_blue_balloons (alyssa_balloons sandy_balloons sally_balloons : ℕ)
  (h1 : alyssa_balloons = 37)
  (h2 : sandy_balloons = 28)
  (h3 : sally_balloons = 39) :
  alyssa_balloons + sandy_balloons + sally_balloons = 104 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l4079_407957


namespace NUMINAMATH_CALUDE_figure3_turns_l4079_407924

/-- Represents a dot in the grid --/
inductive Dot
| Black : Dot
| White : Dot

/-- Represents a turn in the loop --/
inductive Turn
| Right : Turn

/-- Represents a grid with dots --/
structure Grid :=
(dots : List Dot)

/-- Represents a loop in the grid --/
structure Loop :=
(turns : List Turn)

/-- Function to check if a loop is valid for a given grid --/
def is_valid_loop (g : Grid) (l : Loop) : Prop := sorry

/-- Function to count the number of turns in a loop --/
def count_turns (l : Loop) : Nat := l.turns.length

/-- The specific grid configuration for Figure 3 --/
def figure3 : Grid := sorry

/-- Theorem stating that the valid loop for Figure 3 has 20 turns --/
theorem figure3_turns :
  ∃ (l : Loop), is_valid_loop figure3 l ∧ count_turns l = 20 := by sorry

end NUMINAMATH_CALUDE_figure3_turns_l4079_407924


namespace NUMINAMATH_CALUDE_trucks_left_l4079_407947

-- Define the initial number of trucks Sarah had
def initial_trucks : ℕ := 51

-- Define the number of trucks Sarah gave away
def trucks_given_away : ℕ := 13

-- Theorem to prove
theorem trucks_left : initial_trucks - trucks_given_away = 38 := by
  sorry

end NUMINAMATH_CALUDE_trucks_left_l4079_407947


namespace NUMINAMATH_CALUDE_johnny_work_hours_l4079_407964

/-- Given Johnny's hourly wage and total earnings, prove the number of hours he worked -/
theorem johnny_work_hours (hourly_wage : ℝ) (total_earnings : ℝ) (h1 : hourly_wage = 6.75) (h2 : total_earnings = 67.5) :
  total_earnings / hourly_wage = 10 := by
  sorry

end NUMINAMATH_CALUDE_johnny_work_hours_l4079_407964


namespace NUMINAMATH_CALUDE_jonas_sequence_l4079_407993

/-- Sequence of positive multiples of 13 in ascending order -/
def multiples_of_13 : ℕ → ℕ := λ n => 13 * (n + 1)

/-- The nth digit in the sequence of multiples of 13 -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Whether a number appears in the sequence of multiples of 13 -/
def appears_in_sequence (m : ℕ) : Prop := ∃ k : ℕ, multiples_of_13 k = m

theorem jonas_sequence :
  (nth_digit 2019 = 8) ∧ appears_in_sequence 2019 := by sorry

end NUMINAMATH_CALUDE_jonas_sequence_l4079_407993


namespace NUMINAMATH_CALUDE_ball_ratio_problem_l4079_407920

theorem ball_ratio_problem (white_balls red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 4 / 3 →
  white_balls = 12 →
  red_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_ball_ratio_problem_l4079_407920


namespace NUMINAMATH_CALUDE_oplus_two_one_l4079_407918

def oplus (x y : ℝ) : ℝ := x^3 - 3*x*y^2 + y^3

theorem oplus_two_one : oplus 2 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_oplus_two_one_l4079_407918


namespace NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l4079_407972

theorem halfway_between_one_eighth_and_one_third :
  (1/8 : ℚ) + ((1/3 : ℚ) - (1/8 : ℚ)) / 2 = 11/48 := by sorry

end NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l4079_407972


namespace NUMINAMATH_CALUDE_total_people_needed_l4079_407937

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := 2 * people_per_car

/-- The number of cars to be lifted -/
def num_cars : ℕ := 6

/-- The number of trucks to be lifted -/
def num_trucks : ℕ := 3

/-- Theorem stating the total number of people needed to lift the given vehicles -/
theorem total_people_needed : 
  num_cars * people_per_car + num_trucks * people_per_truck = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_people_needed_l4079_407937


namespace NUMINAMATH_CALUDE_division_and_addition_of_fractions_l4079_407940

theorem division_and_addition_of_fractions : 
  (2 : ℚ) / 3 / ((4 : ℚ) / 5) + (1 : ℚ) / 2 = (4 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_of_fractions_l4079_407940


namespace NUMINAMATH_CALUDE_f_periodic_l4079_407943

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan (x / 2) + 1

theorem f_periodic (a : ℝ) (h : f (-a) = 11) : f (2 * Real.pi + a) = -9 := by
  sorry

end NUMINAMATH_CALUDE_f_periodic_l4079_407943


namespace NUMINAMATH_CALUDE_min_workers_proof_l4079_407922

/-- The minimum number of workers in team A that satisfies the given conditions -/
def min_workers_A : ℕ := 153

/-- The number of workers team B transfers to team A -/
def workers_transferred : ℕ := (11 * min_workers_A - 1620) / 7

theorem min_workers_proof :
  (∀ a b : ℕ,
    (a ≥ min_workers_A) →
    (b + 90 = 2 * (a - 90)) →
    (a + workers_transferred = 6 * (b - workers_transferred)) →
    (workers_transferred > 0) →
    (∃ k : ℕ, a + 1 = 7 * k)) →
  (∀ a : ℕ,
    (a < min_workers_A) →
    (¬∃ b : ℕ,
      (b + 90 = 2 * (a - 90)) ∧
      (a + workers_transferred = 6 * (b - workers_transferred)) ∧
      (workers_transferred > 0))) :=
by sorry

end NUMINAMATH_CALUDE_min_workers_proof_l4079_407922


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l4079_407988

theorem divisibility_by_twelve (n : Nat) : n < 10 → (516 * 10 + n) % 12 = 0 ↔ n = 0 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l4079_407988


namespace NUMINAMATH_CALUDE_integer_solution_exists_l4079_407952

theorem integer_solution_exists : ∃ (x₁ x₂ y₁ y₂ y₃ y₄ : ℤ),
  (x₁ + x₂ = y₁ + y₂ + y₃ + y₄) ∧
  (x₁^2 + x₂^2 = y₁^2 + y₂^2 + y₃^2 + y₄^2) ∧
  (x₁^3 + x₂^3 = y₁^3 + y₂^3 + y₃^3 + y₄^3) ∧
  (abs x₁ > 2020) ∧ (abs x₂ > 2020) ∧
  (abs y₁ > 2020) ∧ (abs y₂ > 2020) ∧
  (abs y₃ > 2020) ∧ (abs y₄ > 2020) := by
  sorry

#print integer_solution_exists

end NUMINAMATH_CALUDE_integer_solution_exists_l4079_407952


namespace NUMINAMATH_CALUDE_towel_area_decrease_l4079_407925

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let new_length := 0.7 * L
  let new_breadth := 0.85 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.405 := by
sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l4079_407925


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4079_407998

/-- Given a hyperbola with the following properties:
  1. Its equation is of the form x²/a² - y²/b² = 1 where a > 0 and b > 0
  2. It has an asymptote parallel to the line x + 2y + 5 = 0
  3. One of its foci lies on the line x + 2y + 5 = 0
  Prove that its equation is x²/20 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ k, k ≠ 0 ∧ b / a = 1 / 2 * k) -- Asymptote parallel condition
  (h4 : ∃ x y, x + 2*y + 5 = 0 ∧ (x - a)^2 / a^2 + y^2 / b^2 = 1) -- Focus on line condition
  : a^2 = 20 ∧ b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4079_407998


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l4079_407912

def diamond (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 2

theorem diamond_equation_solution :
  ∀ X : ℝ, diamond X 6 = 35 → X = 51 / 4 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l4079_407912


namespace NUMINAMATH_CALUDE_height_difference_climbing_l4079_407989

/-- Proves that the difference in height climbed between two people with different climbing rates over a given time is equal to the product of the time and the difference in their climbing rates. -/
theorem height_difference_climbing (matt_rate jason_rate : ℝ) (time : ℝ) 
  (h1 : matt_rate = 6)
  (h2 : jason_rate = 12)
  (h3 : time = 7) :
  jason_rate * time - matt_rate * time = (jason_rate - matt_rate) * time :=
by sorry

/-- Calculates the actual height difference between Jason and Matt after 7 minutes of climbing. -/
def actual_height_difference (matt_rate jason_rate : ℝ) (time : ℝ) 
  (h1 : matt_rate = 6)
  (h2 : jason_rate = 12)
  (h3 : time = 7) : ℝ :=
jason_rate * time - matt_rate * time

#eval actual_height_difference 6 12 7 rfl rfl rfl

end NUMINAMATH_CALUDE_height_difference_climbing_l4079_407989


namespace NUMINAMATH_CALUDE_arctan_gt_arcsin_iff_in_open_interval_l4079_407991

theorem arctan_gt_arcsin_iff_in_open_interval (x : ℝ) :
  Real.arctan x > Real.arcsin x ↔ x ∈ Set.Ioo (-1 : ℝ) 0 :=
by
  sorry

end NUMINAMATH_CALUDE_arctan_gt_arcsin_iff_in_open_interval_l4079_407991


namespace NUMINAMATH_CALUDE_kamals_biology_marks_l4079_407994

def english_marks : ℕ := 76
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def total_subjects : ℕ := 5
def average_marks : ℕ := 75

theorem kamals_biology_marks :
  ∃ (biology_marks : ℕ),
    biology_marks = total_subjects * average_marks - (english_marks + mathematics_marks + physics_marks + chemistry_marks) :=
by sorry

end NUMINAMATH_CALUDE_kamals_biology_marks_l4079_407994


namespace NUMINAMATH_CALUDE_sum_n_value_l4079_407930

/-- An arithmetic sequence {a_n} satisfying given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  condition1 : a 3 * a 7 = -16
  condition2 : a 4 + a 6 = 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n : ℤ) * (seq.a 1 + seq.a n) / 2

/-- Theorem stating the possible values for the sum of the first n terms -/
theorem sum_n_value (seq : ArithmeticSequence) (n : ℕ) :
  sum_n seq n = n * (n - 9) ∨ sum_n seq n = -n * (n - 9) := by
  sorry


end NUMINAMATH_CALUDE_sum_n_value_l4079_407930


namespace NUMINAMATH_CALUDE_linear_equation_solution_l4079_407935

theorem linear_equation_solution (x y m : ℝ) 
  (hx : x = -1)
  (hy : y = 2)
  (hm : 5 * x + 3 * y = m) : 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l4079_407935


namespace NUMINAMATH_CALUDE_factorization_theorem_1_l4079_407975

theorem factorization_theorem_1 (x : ℝ) : 
  4 * (x - 2)^2 - 1 = (2*x - 3) * (2*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_1_l4079_407975


namespace NUMINAMATH_CALUDE_total_travel_time_l4079_407982

def station_distance : ℕ := 2 -- hours
def break_time : ℕ := 30 -- minutes

theorem total_travel_time :
  let travel_time_between_stations := station_distance * 60 -- convert hours to minutes
  let total_travel_time := 2 * travel_time_between_stations + break_time
  total_travel_time = 270 := by
sorry

end NUMINAMATH_CALUDE_total_travel_time_l4079_407982


namespace NUMINAMATH_CALUDE_system_solution_l4079_407905

theorem system_solution :
  ∃ (x y : ℚ), 
    (7 * x - 50 * y = 2) ∧ 
    (3 * y - x = 4) ∧ 
    (x = -206/29) ∧ 
    (y = -30/29) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4079_407905


namespace NUMINAMATH_CALUDE_circle_line_distance_l4079_407958

theorem circle_line_distance (m : ℝ) : 
  (∃ (A B C : ℝ × ℝ), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧ (C.1^2 + C.2^2 = 4) ∧
    (|A.2 + A.1 - m| / Real.sqrt 2 = 1) ∧
    (|B.2 + B.1 - m| / Real.sqrt 2 = 1) ∧
    (|C.2 + C.1 - m| / Real.sqrt 2 = 1)) →
  -Real.sqrt 2 ≤ m ∧ m ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_l4079_407958


namespace NUMINAMATH_CALUDE_room_tiles_proof_l4079_407953

/-- Calculates the least number of square tiles required to pave a rectangular floor -/
def leastSquareTiles (length width : ℕ) : ℕ :=
  let gcd := Nat.gcd length width
  (length * width) / (gcd * gcd)

theorem room_tiles_proof (length width : ℕ) 
  (h_length : length = 5000)
  (h_width : width = 1125) :
  leastSquareTiles length width = 360 := by
  sorry

#eval leastSquareTiles 5000 1125

end NUMINAMATH_CALUDE_room_tiles_proof_l4079_407953


namespace NUMINAMATH_CALUDE_min_correct_answers_quiz_problem_l4079_407908

/-- The minimum number of correctly answered questions to exceed 81 points in a quiz -/
theorem min_correct_answers (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (target_score : ℕ) : ℕ :=
  let min_correct := ((target_score + 1 + incorrect_points * total_questions) + (correct_points + incorrect_points) - 1) / (correct_points + incorrect_points)
  min_correct

/-- The specific quiz problem -/
theorem quiz_problem : min_correct_answers 22 4 2 81 = 21 := by
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_quiz_problem_l4079_407908


namespace NUMINAMATH_CALUDE_value_of_a_l4079_407974

theorem value_of_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 8) 
  (h3 : d = 4) : 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l4079_407974


namespace NUMINAMATH_CALUDE_difference_of_squares_l4079_407967

theorem difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4079_407967


namespace NUMINAMATH_CALUDE_cookies_per_guest_l4079_407970

theorem cookies_per_guest (total_cookies : ℕ) (num_guests : ℕ) (h1 : total_cookies = 38) (h2 : num_guests = 2) :
  total_cookies / num_guests = 19 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_guest_l4079_407970


namespace NUMINAMATH_CALUDE_pie_eating_contest_l4079_407934

theorem pie_eating_contest (student1 student2 student3 : ℚ) 
  (h1 : student1 = 5/6)
  (h2 : student2 = 7/8)
  (h3 : student3 = 2/3) :
  max student1 (max student2 student3) - min student1 (min student2 student3) = 5/24 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l4079_407934


namespace NUMINAMATH_CALUDE_inequality_proof_l4079_407963

theorem inequality_proof (a : ℝ) (h : a ≠ 1) : (1 + a + a^2)^2 < 3*(1 + a^2 + a^4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4079_407963


namespace NUMINAMATH_CALUDE_cos_diff_symmetric_angles_l4079_407923

/-- Two angles are symmetric with respect to the origin if their difference is an odd multiple of π -/
def symmetric_angles (α β : Real) : Prop :=
  ∃ k : Int, β = α + (2 * k - 1) * Real.pi

/-- 
If the terminal sides of angles α and β are symmetric with respect to the origin O,
then cos(α - β) = -1
-/
theorem cos_diff_symmetric_angles (α β : Real) 
  (h : symmetric_angles α β) : Real.cos (α - β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_diff_symmetric_angles_l4079_407923


namespace NUMINAMATH_CALUDE_rectangle_area_is_200_l4079_407901

/-- A rectangular region with three fenced sides and one wall -/
structure FencedRectangle where
  short_side : ℝ
  long_side : ℝ
  fence_length : ℝ
  wall_side : ℝ := long_side
  fenced_sides : ℝ := 2 * short_side + long_side
  area : ℝ := short_side * long_side

/-- The fenced rectangular region satisfying the problem conditions -/
def problem_rectangle : FencedRectangle where
  short_side := 10
  long_side := 20
  fence_length := 40

theorem rectangle_area_is_200 (r : FencedRectangle) :
  r.long_side = 2 * r.short_side →
  r.fence_length = 40 →
  r.area = 200 := by
  sorry

#check rectangle_area_is_200 problem_rectangle

end NUMINAMATH_CALUDE_rectangle_area_is_200_l4079_407901


namespace NUMINAMATH_CALUDE_ad_arrangement_count_l4079_407909

def num_commercial_ads : ℕ := 4
def num_public_service_ads : ℕ := 2
def total_ads : ℕ := 6

theorem ad_arrangement_count :
  (num_commercial_ads.factorial) * (num_public_service_ads.factorial) = 48 :=
sorry

end NUMINAMATH_CALUDE_ad_arrangement_count_l4079_407909


namespace NUMINAMATH_CALUDE_reverse_digit_integers_l4079_407919

theorem reverse_digit_integers (q r : ℕ) : 
  (q ≥ 10 ∧ q < 100) →  -- q is a two-digit number
  (r ≥ 10 ∧ r < 100) →  -- r is a two-digit number
  (∃ (a b : ℕ), q = 10 * a + b ∧ r = 10 * b + a) →  -- q and r have reversed digits
  (q > r → q - r < 30) →  -- positive difference less than 30
  (∀ (q' r' : ℕ), (q' ≥ 10 ∧ q' < 100) → (r' ≥ 10 ∧ r' < 100) → 
    (∃ (a' b' : ℕ), q' = 10 * a' + b' ∧ r' = 10 * b' + a') → 
    (q' > r' → q' - r' ≤ q - r)) →  -- q - r is the greatest possible difference
  (q - r = 27) →  -- greatest difference is 27
  (∃ (a b : ℕ), q = 10 * a + b ∧ r = 10 * b + a ∧ a - b = 3 ∧ a = 9 ∧ b = 6) :=
by sorry

end NUMINAMATH_CALUDE_reverse_digit_integers_l4079_407919


namespace NUMINAMATH_CALUDE_min_alpha_value_l4079_407986

/-- Definition of α-level quasi-periodic function -/
def is_alpha_quasi_periodic (f : ℝ → ℝ) (D : Set ℝ) (α : ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x ∈ D, α * f x = f (x + T)

/-- The function f on the domain [1,+∞) -/
noncomputable def f : ℝ → ℝ
| x => if 1 ≤ x ∧ x < 2 then 2^x * (2*x + 1) else 0  -- We define f only for [1,2) as given

/-- Theorem statement -/
theorem min_alpha_value :
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y) →  -- Monotonically increasing
  (∀ α, is_alpha_quasi_periodic f (Set.Ici 1) α → α ≥ 10/3) ∧
  (is_alpha_quasi_periodic f (Set.Ici 1) (10/3)) :=
by sorry

end NUMINAMATH_CALUDE_min_alpha_value_l4079_407986


namespace NUMINAMATH_CALUDE_linear_function_comparison_inverse_proportion_comparison_l4079_407938

-- Linear function
theorem linear_function_comparison (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = -2 * x₁ + 1) 
  (h2 : y₂ = -2 * x₂ + 1) 
  (h3 : x₁ < x₂) : 
  y₁ > y₂ := by sorry

-- Inverse proportion function
theorem inverse_proportion_comparison (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : x₁ < x₂) 
  (h4 : x₂ < 0) : 
  y₁ > y₂ := by sorry

end NUMINAMATH_CALUDE_linear_function_comparison_inverse_proportion_comparison_l4079_407938


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l4079_407921

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l4079_407921


namespace NUMINAMATH_CALUDE_field_division_l4079_407962

theorem field_division (total_area smaller_area larger_area : ℝ) : 
  total_area = 900 ∧
  smaller_area + larger_area = total_area ∧
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 405 := by
  sorry

end NUMINAMATH_CALUDE_field_division_l4079_407962


namespace NUMINAMATH_CALUDE_stamp_ratio_problem_l4079_407954

theorem stamp_ratio_problem (x : ℕ) 
  (h1 : x > 0)
  (h2 : 7 * x - 8 = (4 * x + 8) + 8) :
  (7 * x - 8) / (4 * x + 8) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_stamp_ratio_problem_l4079_407954


namespace NUMINAMATH_CALUDE_min_box_height_is_seven_l4079_407927

/-- Represents the side length of the square base of the box -/
def base_side : ℝ → ℝ := λ x => x

/-- Represents the height of the box -/
def box_height : ℝ → ℝ := λ x => x + 5

/-- Calculates the surface area of the box -/
def surface_area : ℝ → ℝ := λ x => 2 * x^2 + 4 * x * (x + 5)

/-- Theorem stating that the minimum height of the box satisfying the conditions is 7 -/
theorem min_box_height_is_seven :
  ∃ x : ℝ, x > 0 ∧ 
    surface_area x ≥ 120 ∧
    box_height x = 7 ∧
    ∀ y : ℝ, y > 0 ∧ surface_area y ≥ 120 → box_height y ≥ box_height x :=
by
  sorry


end NUMINAMATH_CALUDE_min_box_height_is_seven_l4079_407927


namespace NUMINAMATH_CALUDE_product_one_plus_minus_sqrt_three_l4079_407939

theorem product_one_plus_minus_sqrt_three : (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_product_one_plus_minus_sqrt_three_l4079_407939


namespace NUMINAMATH_CALUDE_plane_division_l4079_407933

/-- Given m parallel lines and n non-parallel lines on a plane,
    where no more than two lines pass through any single point,
    the number of regions into which these lines divide the plane
    is 1 + (n(n+1))/2 + m(n+1). -/
theorem plane_division (m n : ℕ) : ℕ := by
  sorry

#check plane_division

end NUMINAMATH_CALUDE_plane_division_l4079_407933


namespace NUMINAMATH_CALUDE_percentage_relationship_l4079_407926

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 0.4444444444444444)) :
  y = x * 1.8 := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l4079_407926


namespace NUMINAMATH_CALUDE_kyle_money_after_snowboarding_l4079_407960

theorem kyle_money_after_snowboarding (dave_money : ℕ) (kyle_initial_money : ℕ) 
  (h1 : dave_money = 46) 
  (h2 : kyle_initial_money = 3 * dave_money - 12) 
  (h3 : kyle_initial_money ≥ 12) : 
  kyle_initial_money - (kyle_initial_money / 3) = 84 := by
  sorry

end NUMINAMATH_CALUDE_kyle_money_after_snowboarding_l4079_407960


namespace NUMINAMATH_CALUDE_conic_eccentricity_l4079_407910

/-- Given that 4, m, 1 form a geometric sequence, 
    the eccentricity of x²/m + y² = 1 is √2/2 or √3 -/
theorem conic_eccentricity (m : ℝ) : 
  (4 * 1 = m^2) →  -- Geometric sequence condition
  (∃ (e : ℝ), (e = Real.sqrt 2 / 2 ∨ e = Real.sqrt 3) ∧
   ∀ (x y : ℝ), x^2 / m + y^2 = 1 → 
   (∃ (a b : ℝ), 
     (m > 0 → x^2 / a^2 + y^2 / b^2 = 1 ∧ e = Real.sqrt (1 - b^2 / a^2)) ∧
     (m < 0 → y^2 / a^2 - x^2 / b^2 = 1 ∧ e = Real.sqrt (1 + a^2 / b^2)))) :=
by sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l4079_407910


namespace NUMINAMATH_CALUDE_prob_red_then_black_standard_deck_l4079_407999

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- Definition of a standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    red_cards := 26,
    black_cards := 26 }

/-- Probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards * (d.black_cards : ℚ) / (d.total_cards - 1)

/-- Theorem stating the probability for a standard deck -/
theorem prob_red_then_black_standard_deck :
  prob_red_then_black standard_deck = 13 / 51 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_black_standard_deck_l4079_407999


namespace NUMINAMATH_CALUDE_nancys_apples_calculation_l4079_407915

/-- The number of apples Nancy ate -/
def nancys_apples : ℝ := 3.0

/-- The number of apples Mike picked -/
def mikes_apples : ℝ := 7.0

/-- The number of apples Keith picked -/
def keiths_apples : ℝ := 6.0

/-- The number of apples left -/
def apples_left : ℝ := 10.0

/-- Theorem: Nancy's apples equals the total picked by Mike and Keith minus the apples left -/
theorem nancys_apples_calculation : 
  nancys_apples = mikes_apples + keiths_apples - apples_left := by
  sorry

end NUMINAMATH_CALUDE_nancys_apples_calculation_l4079_407915


namespace NUMINAMATH_CALUDE_certain_number_proof_l4079_407983

theorem certain_number_proof (x y : ℕ) : 
  x + y = 24 → 
  x = 11 → 
  x ≤ y → 
  7 * x + 5 * y = 142 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l4079_407983


namespace NUMINAMATH_CALUDE_log_problem_l4079_407944

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_problem (x : ℝ) (h1 : x < 1) (h2 : (log10 x)^3 - log10 (x^3) = 125) :
  (log10 x)^4 - log10 (x^4) = 645 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l4079_407944


namespace NUMINAMATH_CALUDE_spring_work_l4079_407903

/-- Work done to stretch a spring -/
theorem spring_work (force : Real) (compression : Real) (stretch : Real) : 
  force = 10 →
  compression = 0.1 →
  stretch = 0.06 →
  (1/2) * (force / compression) * stretch^2 = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_spring_work_l4079_407903


namespace NUMINAMATH_CALUDE_unique_solution_on_sphere_l4079_407987

theorem unique_solution_on_sphere (x y : ℝ) :
  (x - 8)^2 + (y - 9)^2 + (x - y)^2 = 1/3 →
  x = 8 + 1/3 ∧ y = 8 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_on_sphere_l4079_407987


namespace NUMINAMATH_CALUDE_a_sum_cube_minus_product_l4079_407959

noncomputable def a (i : ℕ) (x : ℝ) : ℝ := ∑' n, (x ^ (3 * n + i)) / (Nat.factorial (3 * n + i))

theorem a_sum_cube_minus_product (x : ℝ) :
  (a 0 x) ^ 3 + (a 1 x) ^ 3 + (a 2 x) ^ 3 - 3 * (a 0 x) * (a 1 x) * (a 2 x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_sum_cube_minus_product_l4079_407959


namespace NUMINAMATH_CALUDE_prop_P_implies_t_range_prop_P_sufficient_not_necessary_implies_a_range_l4079_407979

/-- Represents the curve equation -/
def curve_equation (x y t : ℝ) : Prop :=
  x^2 / (4 - t) + y^2 / (t - 1) = 1

/-- Predicate for the curve being an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (t : ℝ) : Prop :=
  ∃ x y : ℝ, curve_equation x y t

/-- The inequality involving t and a -/
def inequality (t a : ℝ) : Prop :=
  t^2 - (a + 3) * t + (a + 2) < 0

/-- Proposition P implies the range of t -/
theorem prop_P_implies_t_range :
  ∀ t : ℝ, is_ellipse_x_foci t → 1 < t ∧ t < 5/2 :=
sorry

/-- Proposition P is sufficient but not necessary for Q implies the range of a -/
theorem prop_P_sufficient_not_necessary_implies_a_range :
  (∀ t a : ℝ, is_ellipse_x_foci t → inequality t a) ∧
  (∃ t a : ℝ, inequality t a ∧ ¬is_ellipse_x_foci t) →
  ∀ a : ℝ, a > 1/2 :=
sorry

end NUMINAMATH_CALUDE_prop_P_implies_t_range_prop_P_sufficient_not_necessary_implies_a_range_l4079_407979


namespace NUMINAMATH_CALUDE_prop_a_false_prop_b_false_prop_c_true_prop_d_false_false_propositions_l4079_407931

-- Proposition A
theorem prop_a_false : ¬(∀ x : ℝ, x^2 + 3 < 0) := by sorry

-- Proposition B
theorem prop_b_false : ¬(∀ x : ℕ, x^2 > 1) := by sorry

-- Proposition C
theorem prop_c_true : ∃ x : ℤ, x^5 < 1 := by sorry

-- Proposition D
theorem prop_d_false : ¬(∃ x : ℚ, x^2 = 3) := by sorry

-- Combined theorem
theorem false_propositions :
  (¬(∀ x : ℝ, x^2 + 3 < 0)) ∧
  (¬(∀ x : ℕ, x^2 > 1)) ∧
  (∃ x : ℤ, x^5 < 1) ∧
  (¬(∃ x : ℚ, x^2 = 3)) := by sorry

end NUMINAMATH_CALUDE_prop_a_false_prop_b_false_prop_c_true_prop_d_false_false_propositions_l4079_407931


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4079_407990

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2 * x - 6| = 3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4079_407990


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l4079_407951

theorem sum_of_three_numbers (a b c : ℝ) : 
  a + b = 35 ∧ b + c = 50 ∧ c + a = 60 → a + b + c = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l4079_407951


namespace NUMINAMATH_CALUDE_sum_of_squared_pairs_l4079_407942

theorem sum_of_squared_pairs (a b c d : ℝ) : 
  (a^4 - 24*a^3 + 50*a^2 - 35*a + 10 = 0) →
  (b^4 - 24*b^3 + 50*b^2 - 35*b + 10 = 0) →
  (c^4 - 24*c^3 + 50*c^2 - 35*c + 10 = 0) →
  (d^4 - 24*d^3 + 50*d^2 - 35*d + 10 = 0) →
  (a+b)^2 + (b+c)^2 + (c+d)^2 + (d+a)^2 = 541 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_pairs_l4079_407942


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l4079_407985

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 9) 
  (h2 : x * y = -6) : 
  x^2 + y^2 = 21 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l4079_407985


namespace NUMINAMATH_CALUDE_parallel_lines_equal_angles_plane_l4079_407904

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the relation for a line forming equal angles with a plane
variable (forms_equal_angles : Line → Plane → Prop)

-- Theorem statement
theorem parallel_lines_equal_angles_plane (a b : Line) (M : Plane) :
  (parallel a b → forms_equal_angles b M) ∧
  ¬(forms_equal_angles b M → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_equal_angles_plane_l4079_407904


namespace NUMINAMATH_CALUDE_solution_set_has_three_elements_l4079_407917

/-- A pair of positive integers representing the sides of a rectangle. -/
structure RectangleSides where
  a : ℕ+
  b : ℕ+

/-- The condition that the perimeter of a rectangle equals its area. -/
def perimeterEqualsArea (sides : RectangleSides) : Prop :=
  2 * (sides.a.val + sides.b.val) = sides.a.val * sides.b.val

/-- The set of all rectangle sides satisfying the perimeter-area equality. -/
def solutionSet : Set RectangleSides :=
  {sides | perimeterEqualsArea sides}

/-- The theorem stating that the solution set contains exactly three elements. -/
theorem solution_set_has_three_elements :
    solutionSet = {⟨3, 6⟩, ⟨6, 3⟩, ⟨4, 4⟩} := by sorry

end NUMINAMATH_CALUDE_solution_set_has_three_elements_l4079_407917


namespace NUMINAMATH_CALUDE_michelle_gas_usage_l4079_407936

theorem michelle_gas_usage (start_gas end_gas : Real) 
  (h1 : start_gas = 0.5)
  (h2 : end_gas = 0.16666666666666666) :
  start_gas - end_gas = 0.33333333333333334 := by
  sorry

end NUMINAMATH_CALUDE_michelle_gas_usage_l4079_407936


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4079_407932

theorem sqrt_equation_solution : 
  Real.sqrt (2 + Real.sqrt (3 + Real.sqrt (81/256))) = (2 + Real.sqrt (81/256)) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4079_407932


namespace NUMINAMATH_CALUDE_condition_relationship_l4079_407995

theorem condition_relationship (x : ℝ) :
  (1 / x > 1 → x < 1) ∧ ¬(x < 1 → 1 / x > 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l4079_407995


namespace NUMINAMATH_CALUDE_base6_to_base10_fraction_l4079_407928

/-- Converts a base-6 number to base-10 --/
def base6To10 (n : ℕ) : ℕ := sorry

/-- Determines if a natural number is a valid 3-digit base-10 number --/
def isValidBase10 (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- Extracts the hundreds digit from a 3-digit base-10 number --/
def hundredsDigit (n : ℕ) : ℕ := n / 100

/-- Extracts the ones digit from a 3-digit base-10 number --/
def onesDigit (n : ℕ) : ℕ := n % 10

theorem base6_to_base10_fraction (c d e : ℕ) :
  base6To10 532 = 100 * c + 10 * d + e →
  isValidBase10 (100 * c + 10 * d + e) →
  (c * e : ℚ) / 10 = 0 := by sorry

end NUMINAMATH_CALUDE_base6_to_base10_fraction_l4079_407928


namespace NUMINAMATH_CALUDE_wrong_height_calculation_l4079_407911

theorem wrong_height_calculation (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (correct_avg : ℝ) 
  (h1 : n = 35)
  (h2 : initial_avg = 185)
  (h3 : actual_height = 106)
  (h4 : correct_avg = 183) :
  ∃ wrong_height : ℝ, 
    wrong_height = n * initial_avg - (n * correct_avg - actual_height) := by
  sorry

end NUMINAMATH_CALUDE_wrong_height_calculation_l4079_407911


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4079_407966

/-- The eccentricity of a hyperbola with equation x²/2 - y² = 1 is √6/2 -/
theorem hyperbola_eccentricity : 
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/2 - y^2 = 1}
  ∃ e : ℝ, e = (Real.sqrt 6) / 2 ∧ 
    ∀ (a b c : ℝ), 
      (a^2 = 2 ∧ b^2 = 1 ∧ c^2 = a^2 + b^2) → 
      e = c / a :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4079_407966


namespace NUMINAMATH_CALUDE_team_points_l4079_407913

/-- Calculates the total points earned by a sports team based on their performance. -/
def total_points (wins losses ties : ℕ) : ℕ :=
  2 * wins + 0 * losses + 1 * ties

/-- Theorem stating that a team with 9 wins, 3 losses, and 4 ties earns 22 points. -/
theorem team_points : total_points 9 3 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_team_points_l4079_407913


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4079_407997

/-- Given a geometric sequence with first term a and common ratio r,
    prove that if the first three terms are of the form a, 3a+3, 6a+6,
    then the fourth term is -24. -/
theorem geometric_sequence_fourth_term
  (a r : ℝ) -- a is the first term, r is the common ratio
  (h1 : (3*a + 3) = a * r) -- second term = first term * r
  (h2 : (6*a + 6) = (3*a + 3) * r) -- third term = second term * r
  : a * r^3 = -24 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4079_407997


namespace NUMINAMATH_CALUDE_f_at_two_equals_three_l4079_407961

def f (x : ℝ) : ℝ := 5 * x - 7

theorem f_at_two_equals_three : f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_equals_three_l4079_407961


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l4079_407992

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_P_complement_Q : P ∩ (U \ Q) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l4079_407992
