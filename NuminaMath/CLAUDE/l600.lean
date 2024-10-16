import Mathlib

namespace NUMINAMATH_CALUDE_point_on_line_with_equal_distances_quadrant_l600_60079

/-- A point with coordinates (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line y = 2x + 3 -/
def lineEquation (p : Point) : Prop :=
  p.y = 2 * p.x + 3

/-- Equal distance to both coordinate axes -/
def equalDistanceToAxes (p : Point) : Prop :=
  abs p.x = abs p.y

/-- Second quadrant -/
def inSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Third quadrant -/
def inThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: A point on the line y = 2x + 3 with equal distances to both axes is in the second or third quadrant -/
theorem point_on_line_with_equal_distances_quadrant (p : Point) 
  (h1 : lineEquation p) (h2 : equalDistanceToAxes p) : 
  inSecondQuadrant p ∨ inThirdQuadrant p := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_with_equal_distances_quadrant_l600_60079


namespace NUMINAMATH_CALUDE_rotten_apples_percentage_l600_60036

theorem rotten_apples_percentage (total_apples : ℕ) (smelling_ratio : ℚ) (non_smelling_rotten : ℕ) :
  total_apples = 200 →
  smelling_ratio = 7/10 →
  non_smelling_rotten = 24 →
  (non_smelling_rotten : ℚ) / ((1 - smelling_ratio) * total_apples) = 2/5 :=
by
  sorry

end NUMINAMATH_CALUDE_rotten_apples_percentage_l600_60036


namespace NUMINAMATH_CALUDE_apple_tree_production_l600_60012

theorem apple_tree_production (first_season : ℕ) : 
  (first_season : ℝ) + 0.8 * first_season + 1.6 * first_season = 680 →
  first_season = 200 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_production_l600_60012


namespace NUMINAMATH_CALUDE_three_pairs_satisfy_l600_60037

/-- The set S of elements -/
inductive S
| A₀ : S
| A₁ : S
| A₂ : S

/-- The operation ⊕ on S -/
def op (x y : S) : S :=
  match x, y with
  | S.A₀, S.A₀ => S.A₀
  | S.A₀, S.A₁ => S.A₁
  | S.A₀, S.A₂ => S.A₂
  | S.A₁, S.A₀ => S.A₁
  | S.A₁, S.A₁ => S.A₂
  | S.A₁, S.A₂ => S.A₀
  | S.A₂, S.A₀ => S.A₂
  | S.A₂, S.A₁ => S.A₀
  | S.A₂, S.A₂ => S.A₁

/-- The theorem stating that there are exactly 3 pairs satisfying the equation -/
theorem three_pairs_satisfy :
  ∃! (pairs : List (S × S)), pairs.length = 3 ∧
    ∀ (x y : S), (op (op x y) x = S.A₀) ↔ (x, y) ∈ pairs :=
by sorry

end NUMINAMATH_CALUDE_three_pairs_satisfy_l600_60037


namespace NUMINAMATH_CALUDE_squirrel_stocks_l600_60005

structure Squirrel where
  mushrooms : ℕ
  hazelnuts : ℕ
  fir_cones : ℕ

def total_items (s : Squirrel) : ℕ := s.mushrooms + s.hazelnuts + s.fir_cones

theorem squirrel_stocks :
  ∃ (zrzecka pizizubka krivoousko : Squirrel),
    -- Each squirrel has 48 mushrooms
    zrzecka.mushrooms = 48 ∧ pizizubka.mushrooms = 48 ∧ krivoousko.mushrooms = 48 ∧
    -- Zrzečka has twice as many hazelnuts as Pizizubka
    zrzecka.hazelnuts = 2 * pizizubka.hazelnuts ∧
    -- Křivoouško has 20 more hazelnuts than Pizizubka
    krivoousko.hazelnuts = pizizubka.hazelnuts + 20 ∧
    -- Together, they have 180 fir cones and 180 hazelnuts
    zrzecka.fir_cones + pizizubka.fir_cones + krivoousko.fir_cones = 180 ∧
    zrzecka.hazelnuts + pizizubka.hazelnuts + krivoousko.hazelnuts = 180 ∧
    -- All squirrels have the same total number of items
    total_items zrzecka = total_items pizizubka ∧
    total_items pizizubka = total_items krivoousko ∧
    -- The correct distribution of items
    zrzecka = { mushrooms := 48, hazelnuts := 80, fir_cones := 40 } ∧
    pizizubka = { mushrooms := 48, hazelnuts := 40, fir_cones := 80 } ∧
    krivoousko = { mushrooms := 48, hazelnuts := 60, fir_cones := 60 } :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_stocks_l600_60005


namespace NUMINAMATH_CALUDE_product_in_first_quadrant_l600_60033

def complex_multiply (a b c d : ℝ) : ℂ :=
  Complex.mk (a * c - b * d) (a * d + b * c)

theorem product_in_first_quadrant :
  let z : ℂ := complex_multiply 1 3 3 (-1)
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_product_in_first_quadrant_l600_60033


namespace NUMINAMATH_CALUDE_investment_rate_problem_l600_60040

theorem investment_rate_problem (total_interest amount_invested_low rate_high : ℚ) 
  (h1 : total_interest = 520)
  (h2 : amount_invested_low = 2000)
  (h3 : rate_high = 5 / 100) : 
  ∃ (rate_low : ℚ), 
    amount_invested_low * rate_low + 4 * amount_invested_low * rate_high = total_interest ∧ 
    rate_low = 6 / 100 := by
sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l600_60040


namespace NUMINAMATH_CALUDE_zoo_bus_distribution_l600_60017

theorem zoo_bus_distribution (total_people : ℕ) (num_buses : ℕ) (h1 : total_people = 219) (h2 : num_buses = 3) :
  total_people % num_buses = 0 →
  total_people / num_buses = 73 := by
sorry

end NUMINAMATH_CALUDE_zoo_bus_distribution_l600_60017


namespace NUMINAMATH_CALUDE_all_permissible_triangles_in_final_set_l600_60085

/-- A permissible triangle for a prime p is represented by its angles as multiples of (180/p) degrees -/
structure PermissibleTriangle (p : ℕ) :=
  (a b c : ℕ)
  (sum_eq_p : a + b + c = p)
  (p_prime : Nat.Prime p)

/-- The set of all permissible triangles for a given prime p -/
def allPermissibleTriangles (p : ℕ) : Set (PermissibleTriangle p) :=
  {t : PermissibleTriangle p | true}

/-- A function that represents cutting a triangle into two different permissible triangles -/
def cutTriangle (p : ℕ) (t : PermissibleTriangle p) : Option (PermissibleTriangle p × PermissibleTriangle p) :=
  sorry

/-- The set of triangles resulting from repeated cutting until no more cuts are possible -/
def finalTriangleSet (p : ℕ) (initial : PermissibleTriangle p) : Set (PermissibleTriangle p) :=
  sorry

/-- The main theorem: the final set of triangles includes all possible permissible triangles -/
theorem all_permissible_triangles_in_final_set (p : ℕ) (hp : Nat.Prime p) (initial : PermissibleTriangle p) :
  finalTriangleSet p initial = allPermissibleTriangles p :=
sorry

end NUMINAMATH_CALUDE_all_permissible_triangles_in_final_set_l600_60085


namespace NUMINAMATH_CALUDE_x_squared_gt_4_necessary_not_sufficient_for_x_cubed_lt_neg_8_l600_60031

theorem x_squared_gt_4_necessary_not_sufficient_for_x_cubed_lt_neg_8 :
  (∀ x : ℝ, x^3 < -8 → x^2 > 4) ∧
  (∃ x : ℝ, x^2 > 4 ∧ x^3 ≥ -8) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_gt_4_necessary_not_sufficient_for_x_cubed_lt_neg_8_l600_60031


namespace NUMINAMATH_CALUDE_dad_steps_l600_60083

/-- Represents the number of steps taken by each person --/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between steps taken by Dad, Masha, and Yasha --/
def step_relation (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha ∧ 5 * s.masha = 3 * s.yasha

/-- The total number of steps taken by Masha and Yasha --/
def total_masha_yasha (s : Steps) : ℕ := s.masha + s.yasha

/-- Theorem stating that given the conditions, Dad took 90 steps --/
theorem dad_steps :
  ∀ s : Steps,
  step_relation s →
  total_masha_yasha s = 400 →
  s.dad = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_dad_steps_l600_60083


namespace NUMINAMATH_CALUDE_min_value_on_negative_interval_l600_60018

/-- Given positive real numbers a and b, and a function f with maximum value 4 on [0,1],
    prove that the minimum value of f on [-1,0] is -3/2 -/
theorem min_value_on_negative_interval
  (a b : ℝ) (f : ℝ → ℝ)
  (a_pos : 0 < a) (b_pos : 0 < b)
  (f_def : ∀ x, f x = a * x^3 + b * x + 2^x)
  (max_value : ∀ x ∈ Set.Icc 0 1, f x ≤ 4)
  (max_achieved : ∃ x ∈ Set.Icc 0 1, f x = 4) :
  ∀ x ∈ Set.Icc (-1) 0, f x ≥ -3/2 ∧ ∃ y ∈ Set.Icc (-1) 0, f y = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_negative_interval_l600_60018


namespace NUMINAMATH_CALUDE_exists_color_with_all_distances_l600_60014

-- Define a type for colors
inductive Color
| Yellow
| Red

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define a distance function between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem exists_color_with_all_distances :
  ∃ c : Color, ∀ x : ℝ, x > 0 → ∃ p1 p2 : Point,
    coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by sorry

end NUMINAMATH_CALUDE_exists_color_with_all_distances_l600_60014


namespace NUMINAMATH_CALUDE_bug_probability_after_8_steps_l600_60065

/-- Probability of being at vertex A after n steps -/
def P (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1 - P (n - 1)) / 3

/-- The probability of being at vertex A after 8 steps in a regular tetrahedron -/
theorem bug_probability_after_8_steps :
  P 8 = 547 / 2187 := by sorry

end NUMINAMATH_CALUDE_bug_probability_after_8_steps_l600_60065


namespace NUMINAMATH_CALUDE_triangle_rectangle_ratio_l600_60071

theorem triangle_rectangle_ratio : 
  ∀ (t w l : ℝ),
  (3 * t = 24) →  -- Perimeter of equilateral triangle
  (2 * l + 2 * w = 24) →  -- Perimeter of rectangle
  (l = 2 * w) →  -- Length is twice the width
  (t / w = 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_ratio_l600_60071


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l600_60049

theorem quadratic_equation_solution : 
  ∀ x : ℝ, (x - 2)^2 = 2*x - 4 ↔ x = 2 ∨ x = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l600_60049


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l600_60013

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (2/5 : ℂ) + (4/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (2/5 : ℂ) - (4/9 : ℂ) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l600_60013


namespace NUMINAMATH_CALUDE_sin_45_eq_neg_cos_135_l600_60038

theorem sin_45_eq_neg_cos_135 : Real.sin (π / 4) = - Real.cos (3 * π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_45_eq_neg_cos_135_l600_60038


namespace NUMINAMATH_CALUDE_binary_sum_equals_136_l600_60054

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary1 : List Bool := [true, false, true, false, true, false, true]
def binary2 : List Bool := [true, true, false, false, true, true]

theorem binary_sum_equals_136 :
  binary_to_decimal binary1 + binary_to_decimal binary2 = 136 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_136_l600_60054


namespace NUMINAMATH_CALUDE_diagonal_passes_at_least_length_squares_l600_60070

/-- Represents an irregular hexagon composed of unit squares -/
structure IrregularHexagon where
  total_squares : ℕ
  length : ℕ
  width1 : ℕ
  width2 : ℕ

/-- The minimum number of squares a diagonal passes through -/
def diagonal_squares_count (h : IrregularHexagon) : ℕ :=
  h.length

/-- Theorem stating that the diagonal passes through at least as many squares as the length -/
theorem diagonal_passes_at_least_length_squares (h : IrregularHexagon)
  (h_total : h.total_squares = 78)
  (h_length : h.length = 12)
  (h_width1 : h.width1 = 8)
  (h_width2 : h.width2 = 6) :
  diagonal_squares_count h ≥ h.length :=
sorry

end NUMINAMATH_CALUDE_diagonal_passes_at_least_length_squares_l600_60070


namespace NUMINAMATH_CALUDE_president_secretary_selection_l600_60063

theorem president_secretary_selection (n : ℕ) (h : n = 6) :
  (n * (n - 1) : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_president_secretary_selection_l600_60063


namespace NUMINAMATH_CALUDE_valid_numeral_count_l600_60096

def is_single_digit_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def count_valid_numerals : ℕ :=
  let three_digit_count := 4 * 10 * 4
  let four_digit_count := 4 * 10 * 10 * 4
  three_digit_count + four_digit_count

theorem valid_numeral_count :
  count_valid_numerals = 1760 :=
sorry

end NUMINAMATH_CALUDE_valid_numeral_count_l600_60096


namespace NUMINAMATH_CALUDE_c_is_positive_l600_60088

theorem c_is_positive (a b c d e f : ℤ) 
  (h1 : a * b + c * d * e * f < 0)
  (h2 : a < 0)
  (h3 : b < 0)
  (h4 : d < 0)
  (h5 : e < 0)
  (h6 : f < 0) :
  c > 0 := by
sorry

end NUMINAMATH_CALUDE_c_is_positive_l600_60088


namespace NUMINAMATH_CALUDE_sum_of_solutions_l600_60064

theorem sum_of_solutions (S : ℝ) : 
  ∃ (N₁ N₂ : ℝ), N₁ ≠ 0 ∧ N₂ ≠ 0 ∧ 
  (6 * N₁ + 2 / N₁ = S) ∧ 
  (6 * N₂ + 2 / N₂ = S) ∧ 
  (N₁ + N₂ = S / 6) := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l600_60064


namespace NUMINAMATH_CALUDE_multiple_value_l600_60039

theorem multiple_value (a b m : ℤ) : 
  a * b = m * (a + b) + 1 → 
  b = 7 → 
  b - a = 4 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_multiple_value_l600_60039


namespace NUMINAMATH_CALUDE_total_people_shook_hands_l600_60059

/-- The number of schools participating in the debate -/
def num_schools : ℕ := 5

/-- The number of students in the fourth school -/
def students_fourth : ℕ := 150

/-- The number of faculty members per school -/
def faculty_per_school : ℕ := 10

/-- The number of event staff per school -/
def event_staff_per_school : ℕ := 5

/-- Calculate the number of students in the third school -/
def students_third : ℕ := (3 * students_fourth) / 2

/-- Calculate the number of students in the second school -/
def students_second : ℕ := students_third + 50

/-- Calculate the number of students in the first school -/
def students_first : ℕ := 2 * students_second

/-- Calculate the number of students in the fifth school -/
def students_fifth : ℕ := students_fourth - 120

/-- Calculate the total number of students -/
def total_students : ℕ := students_first + students_second + students_third + students_fourth + students_fifth

/-- Calculate the total number of faculty and staff -/
def total_faculty_staff : ℕ := num_schools * (faculty_per_school + event_staff_per_school)

/-- The theorem to prove -/
theorem total_people_shook_hands : total_students + total_faculty_staff = 1305 := by
  sorry

end NUMINAMATH_CALUDE_total_people_shook_hands_l600_60059


namespace NUMINAMATH_CALUDE_sum_of_digits_5N_plus_2013_l600_60051

/-- Sum of digits function in base 10 -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The smallest positive integer with sum of digits 2013 -/
def N : ℕ := sorry

/-- Theorem stating the sum of digits of (5N + 2013) is 18 -/
theorem sum_of_digits_5N_plus_2013 :
  sum_of_digits (5 * N + 2013) = 18 ∧ 
  sum_of_digits N = 2013 ∧
  ∀ m : ℕ, m < N → sum_of_digits m ≠ 2013 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_5N_plus_2013_l600_60051


namespace NUMINAMATH_CALUDE_chinese_chess_probability_l600_60001

/-- The probability of player A winning a game of Chinese chess -/
def prob_A_win : ℝ := 0.2

/-- The probability of a draw between players A and B -/
def prob_draw : ℝ := 0.5

/-- The probability of player B winning a game of Chinese chess -/
def prob_B_win : ℝ := 1 - (prob_A_win + prob_draw)

theorem chinese_chess_probability :
  prob_B_win = 0.3 := by sorry

end NUMINAMATH_CALUDE_chinese_chess_probability_l600_60001


namespace NUMINAMATH_CALUDE_constant_term_expansion_l600_60062

/-- Given that the constant term in the expansion of (x + a/√x)^6 is 15, 
    prove that the positive value of a is 1. -/
theorem constant_term_expansion (a : ℝ) (h : a > 0) : 
  (∃ (x : ℝ), (x + a / Real.sqrt x)^6 = 15 + x * (1 + 1/x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l600_60062


namespace NUMINAMATH_CALUDE_pen_problem_solution_l600_60047

/-- Represents the number of pens of each color in Maria's desk drawer. -/
structure PenCounts where
  red : ℕ
  black : ℕ
  blue : ℕ

/-- The conditions of the pen problem. -/
def penProblem (p : PenCounts) : Prop :=
  p.red = 8 ∧
  p.black > p.red ∧
  p.blue = p.red + 7 ∧
  p.red + p.black + p.blue = 41

/-- The theorem stating the solution to the pen problem. -/
theorem pen_problem_solution (p : PenCounts) (h : penProblem p) : 
  p.black - p.red = 10 := by
  sorry

end NUMINAMATH_CALUDE_pen_problem_solution_l600_60047


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l600_60006

/-- A quadratic equation x^2 + bx + 16 has at least one real root if and only if b ∈ (-∞,-8] ∪ [8,∞) -/
theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 16 = 0) ↔ b ≤ -8 ∨ b ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l600_60006


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_cos_double_l600_60056

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- Define the acute angle between asymptotes
def asymptote_angle (α : ℝ) : Prop := 
  ∃ (x y : ℝ), hyperbola x y ∧ 
  (∀ (x' y' : ℝ), hyperbola x' y' → 
    α = Real.arctan (abs (y / x)) ∧ α > 0 ∧ α < Real.pi / 2)

-- Theorem statement
theorem hyperbola_asymptote_angle_cos_double :
  ∀ α : ℝ, asymptote_angle α → Real.cos (2 * α) = -7/25 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_cos_double_l600_60056


namespace NUMINAMATH_CALUDE_sum_of_squares_l600_60058

theorem sum_of_squares (x y z : ℝ) 
  (sum_eq : x + y + z = 13)
  (product_eq : x * y * z = 72)
  (sum_reciprocals_eq : 1/x + 1/y + 1/z = 3/4) :
  x^2 + y^2 + z^2 = 61 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l600_60058


namespace NUMINAMATH_CALUDE_unique_solution_mod_30_l600_60050

theorem unique_solution_mod_30 : 
  ∃! x : ℕ, x < 30 ∧ (x^4 + 2*x^3 + 3*x^2 - x + 1) % 30 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_mod_30_l600_60050


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l600_60089

theorem negation_of_universal_proposition :
  (¬ ∀ x y : ℝ, x < 0 → y < 0 → x + y ≤ -2 * Real.sqrt (x * y)) ↔
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x + y > -2 * Real.sqrt (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l600_60089


namespace NUMINAMATH_CALUDE_inequality_solution_set_l600_60008

-- Define the solution set
def solution_set := {x : ℝ | x < 5}

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | |x - 8| - |x - 4| > 2} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l600_60008


namespace NUMINAMATH_CALUDE_lucy_snowballs_l600_60081

theorem lucy_snowballs (charlie_snowballs : ℕ) (difference : ℕ) (lucy_snowballs : ℕ) : 
  charlie_snowballs = 50 → 
  difference = 31 → 
  charlie_snowballs = lucy_snowballs + difference → 
  lucy_snowballs = 19 := by
sorry

end NUMINAMATH_CALUDE_lucy_snowballs_l600_60081


namespace NUMINAMATH_CALUDE_emilia_berry_cobbler_l600_60044

/-- The number of cartons of berries needed for Emilia's berry cobbler -/
def total_cartons (strawberry_cartons blueberry_cartons additional_cartons : ℕ) : ℕ :=
  strawberry_cartons + blueberry_cartons + additional_cartons

/-- Theorem stating that the total number of cartons is 42 given the specific quantities -/
theorem emilia_berry_cobbler : total_cartons 2 7 33 = 42 := by
  sorry

end NUMINAMATH_CALUDE_emilia_berry_cobbler_l600_60044


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l600_60092

-- Define the conditions p and q
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l600_60092


namespace NUMINAMATH_CALUDE_solution_comparison_l600_60000

theorem solution_comparison (a a' b b' k : ℝ) 
  (ha : a ≠ 0) (ha' : a' ≠ 0) (hk : k > 0) :
  (-kb / a < -b' / a') ↔ (k * b * a' > a * b') :=
sorry

end NUMINAMATH_CALUDE_solution_comparison_l600_60000


namespace NUMINAMATH_CALUDE_max_value_three_ways_l600_60027

/-- A function representing the number of ways to draw balls with a specific maximum value -/
def num_ways_max_value (n : ℕ) (max_value : ℕ) (num_draws : ℕ) : ℕ :=
  sorry

/-- The number of balls in the box -/
def num_balls : ℕ := 3

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The maximum value we're interested in -/
def target_max : ℕ := 3

theorem max_value_three_ways :
  num_ways_max_value num_balls target_max num_draws = 19 := by
  sorry

end NUMINAMATH_CALUDE_max_value_three_ways_l600_60027


namespace NUMINAMATH_CALUDE_swimming_hours_per_month_l600_60021

/-- Calculate the required hours per month for freestyle and sidestroke swimming --/
theorem swimming_hours_per_month 
  (total_required : ℕ) 
  (completed : ℕ) 
  (months : ℕ) 
  (h1 : total_required = 1500) 
  (h2 : completed = 180) 
  (h3 : months = 6) :
  (total_required - completed) / months = 220 :=
by sorry

end NUMINAMATH_CALUDE_swimming_hours_per_month_l600_60021


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l600_60003

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt x + Real.sqrt (x + 6) = 12 → x = 529 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l600_60003


namespace NUMINAMATH_CALUDE_special_ap_ratio_l600_60090

/-- An arithmetic progression with the property that the sum of its first ten terms
    is four times the sum of its first five terms. -/
structure SpecialAP where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_condition : (10 * a + 45 * d) = 4 * (5 * a + 10 * d)

/-- The ratio of the first term to the common difference in a SpecialAP is 1:2. -/
theorem special_ap_ratio (ap : SpecialAP) : ap.a / ap.d = 1 / 2 := by
  sorry

#check special_ap_ratio

end NUMINAMATH_CALUDE_special_ap_ratio_l600_60090


namespace NUMINAMATH_CALUDE_max_abc_constrained_polynomial_l600_60016

/-- A polynomial of degree 4 with specific constraints on its coefficients. -/
structure ConstrainedPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  bound_a : a < 3
  bound_b : b < 3
  bound_c : c < 3
  p : ℝ → ℝ := λ x => x^4 + a*x^3 + b*x^2 + c*x + 1
  no_real_roots : ∀ x : ℝ, p x ≠ 0

/-- The maximum value of abc for polynomials satisfying the given constraints is 18.75. -/
theorem max_abc_constrained_polynomial (poly : ConstrainedPolynomial) :
  ∃ M : ℝ, M = 18.75 ∧ poly.a * poly.b * poly.c ≤ M ∧
  ∀ ε > 0, ∃ poly' : ConstrainedPolynomial, poly'.a * poly'.b * poly'.c > M - ε :=
sorry

end NUMINAMATH_CALUDE_max_abc_constrained_polynomial_l600_60016


namespace NUMINAMATH_CALUDE_original_number_is_22_l600_60032

theorem original_number_is_22 (N : ℕ) : 
  (∀ k < 6, ¬ (16 ∣ (N - k))) →  -- Condition 1: 6 is the least number
  (16 ∣ (N - 6)) →               -- Condition 2: N - 6 is divisible by 16
  N = 22 := by                   -- Conclusion: The original number is 22
sorry

end NUMINAMATH_CALUDE_original_number_is_22_l600_60032


namespace NUMINAMATH_CALUDE_smallest_angle_of_triangle_l600_60080

theorem smallest_angle_of_triangle (y : ℝ) (h : y + 40 + 70 = 180) :
  min (min 40 70) y = 40 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_of_triangle_l600_60080


namespace NUMINAMATH_CALUDE_second_class_size_l600_60098

/-- Given two classes of students, where:
    - The first class has 24 students with an average mark of 40
    - The second class has an unknown number of students with an average mark of 60
    - The average mark of all students combined is 53.513513513513516
    This theorem proves that the number of students in the second class is 50. -/
theorem second_class_size (n : ℕ) :
  let first_class_size : ℕ := 24
  let first_class_avg : ℝ := 40
  let second_class_avg : ℝ := 60
  let total_avg : ℝ := 53.513513513513516
  let total_size : ℕ := first_class_size + n
  (first_class_size * first_class_avg + n * second_class_avg) / total_size = total_avg →
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_second_class_size_l600_60098


namespace NUMINAMATH_CALUDE_hundred_million_scientific_notation_l600_60097

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem hundred_million_scientific_notation :
  toScientificNotation 100000000 = ScientificNotation.mk 1 8 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_hundred_million_scientific_notation_l600_60097


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l600_60002

theorem arithmetic_sequence_count (n : ℕ) (m : ℕ) (k : ℕ) (h1 : n = 2014) (h2 : m = 315) (h3 : k = 5490) :
  (∃ (sequences : Finset (Finset ℕ)),
    sequences.card = k ∧
    (∀ seq ∈ sequences,
      seq.card = m ∧
      (∃ d : ℕ, d > 0 ∧ d ≤ 6 ∧
        (∀ i j : ℕ, i < j → i ∈ seq → j ∈ seq →
          ∃ k : ℕ, j - i = k * d)) ∧
      1 ∈ seq ∧
      (∀ x ∈ seq, 1 ≤ x ∧ x ≤ n))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l600_60002


namespace NUMINAMATH_CALUDE_fish_in_third_tank_l600_60025

/-- The number of fish in the first tank -/
def first_tank : ℕ := 7 + 8

/-- The number of fish in the second tank -/
def second_tank : ℕ := 2 * first_tank

/-- The number of fish in the third tank -/
def third_tank : ℕ := second_tank / 3

theorem fish_in_third_tank : third_tank = 10 := by
  sorry

end NUMINAMATH_CALUDE_fish_in_third_tank_l600_60025


namespace NUMINAMATH_CALUDE_composite_expressions_l600_60091

theorem composite_expressions (p : ℕ) (hp : Nat.Prime p) : 
  (¬ Nat.Prime (p^2 + 35)) ∧ (¬ Nat.Prime (p^2 + 55)) := by
  sorry

end NUMINAMATH_CALUDE_composite_expressions_l600_60091


namespace NUMINAMATH_CALUDE_oliver_age_l600_60061

/-- Given the ages of Mark, Nina, and Oliver, prove that Oliver is 22 years old. -/
theorem oliver_age (m n o : ℕ) : 
  (m + n + o) / 3 = 12 →  -- Average age is 12
  o - 5 = 2 * n →  -- Five years ago, Oliver was twice Nina's current age
  m + 2 = (4 * (n + 2)) / 5 →  -- In 2 years, Mark's age will be 4/5 of Nina's
  m + 4 + n + 4 + o + 4 = 60 →  -- In 4 years, total age will be 60
  o = 22 := by
  sorry

end NUMINAMATH_CALUDE_oliver_age_l600_60061


namespace NUMINAMATH_CALUDE_feb_1_2015_was_sunday_l600_60007

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Theorem: If January 1, 2015 was a Thursday, then February 1, 2015 was a Sunday -/
theorem feb_1_2015_was_sunday :
  advanceDays DayOfWeek.Thursday 31 = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_feb_1_2015_was_sunday_l600_60007


namespace NUMINAMATH_CALUDE_total_distance_walked_l600_60075

/-- Given a pace of 2 miles per hour maintained for 8 hours, 
    the total distance walked is 16 miles. -/
theorem total_distance_walked 
  (pace : ℝ) 
  (duration : ℝ) 
  (h1 : pace = 2) 
  (h2 : duration = 8) : 
  pace * duration = 16 := by
sorry

end NUMINAMATH_CALUDE_total_distance_walked_l600_60075


namespace NUMINAMATH_CALUDE_stratified_sample_grade12_l600_60046

/-- Represents the number of students in a grade -/
structure GradePopulation where
  total : ℕ
  sampled : ℕ

/-- Represents the school population -/
structure SchoolPopulation where
  grade11 : GradePopulation
  grade12 : GradePopulation

/-- Checks if the sampling is stratified (same ratio across grades) -/
def isStratifiedSample (school : SchoolPopulation) : Prop :=
  school.grade11.sampled * school.grade12.total = school.grade12.sampled * school.grade11.total

/-- The main theorem -/
theorem stratified_sample_grade12 (school : SchoolPopulation) 
    (h1 : school.grade11.total = 500)
    (h2 : school.grade12.total = 450)
    (h3 : school.grade11.sampled = 20)
    (h4 : isStratifiedSample school) :
  school.grade12.sampled = 18 := by
  sorry

#check stratified_sample_grade12

end NUMINAMATH_CALUDE_stratified_sample_grade12_l600_60046


namespace NUMINAMATH_CALUDE_sum_of_divisors_930_l600_60026

/-- Sum of positive divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem sum_of_divisors_930 (i j : ℕ+) :
  sum_of_divisors (2^i.val * 5^j.val) = 930 → i.val + j.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_930_l600_60026


namespace NUMINAMATH_CALUDE_f_derivative_at_negative_one_l600_60055

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- State the theorem
theorem f_derivative_at_negative_one (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_negative_one_l600_60055


namespace NUMINAMATH_CALUDE_max_product_l600_60078

def Digits : Finset Nat := {3, 5, 8, 9, 1}

def valid_two_digit (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n / 10) ∈ Digits ∧ (n % 10) ∈ Digits ∧ (n / 10) ≠ (n % 10)

def valid_three_digit (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (n / 100) ∈ Digits ∧ ((n / 10) % 10) ∈ Digits ∧ (n % 10) ∈ Digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧ (n / 100) ≠ (n % 10) ∧ ((n / 10) % 10) ≠ (n % 10)

def valid_pair (a b : Nat) : Prop :=
  valid_two_digit a ∧ valid_three_digit b ∧
  (∀ d : Nat, d ∈ Digits → (d = (a / 10) ∨ d = (a % 10) ∨ d = (b / 100) ∨ d = ((b / 10) % 10) ∨ d = (b % 10)))

theorem max_product :
  ∀ a b : Nat, valid_pair a b → a * b ≤ 91 * 853 :=
sorry

end NUMINAMATH_CALUDE_max_product_l600_60078


namespace NUMINAMATH_CALUDE_books_per_shelf_l600_60043

theorem books_per_shelf 
  (mystery_shelves : ℕ) 
  (picture_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : mystery_shelves = 6) 
  (h2 : picture_shelves = 2) 
  (h3 : total_books = 72) : 
  total_books / (mystery_shelves + picture_shelves) = 9 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l600_60043


namespace NUMINAMATH_CALUDE_cost_price_correct_l600_60060

/-- The cost price of a product satisfying given conditions -/
def cost_price : ℝ := 90

/-- The marked price of the product -/
def marked_price : ℝ := 120

/-- The discount rate applied to the product -/
def discount_rate : ℝ := 0.1

/-- The profit rate relative to the cost price -/
def profit_rate : ℝ := 0.2

/-- Theorem stating that the cost price is correct given the conditions -/
theorem cost_price_correct : 
  cost_price * (1 + profit_rate) = marked_price * (1 - discount_rate) := by
  sorry

#eval cost_price -- Should output 90

end NUMINAMATH_CALUDE_cost_price_correct_l600_60060


namespace NUMINAMATH_CALUDE_mikes_total_payment_l600_60048

/-- Calculates the amount Mike needs to pay after insurance coverage for his medical tests. -/
def mikes_payment (xray_cost : ℚ) (blood_test_cost : ℚ) : ℚ :=
  let mri_cost := 3 * xray_cost
  let ct_scan_cost := 2 * mri_cost
  let xray_payment := xray_cost * (1 - 0.8)
  let mri_payment := mri_cost * (1 - 0.8)
  let ct_scan_payment := ct_scan_cost * (1 - 0.7)
  let blood_test_payment := blood_test_cost * (1 - 0.5)
  xray_payment + mri_payment + ct_scan_payment + blood_test_payment

/-- Theorem stating that Mike's payment after insurance coverage is $750. -/
theorem mikes_total_payment :
  mikes_payment 250 200 = 750 := by
  sorry

end NUMINAMATH_CALUDE_mikes_total_payment_l600_60048


namespace NUMINAMATH_CALUDE_factor_x8_minus_625_l600_60041

theorem factor_x8_minus_625 (x : ℝ) : 
  x^8 - 625 = (x^4 + 25) * (x^2 + 5) * (x + Real.sqrt 5) * (x - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_x8_minus_625_l600_60041


namespace NUMINAMATH_CALUDE_problem_1_l600_60028

theorem problem_1 : |(-6)| - 7 + (-3) = -4 := by sorry

end NUMINAMATH_CALUDE_problem_1_l600_60028


namespace NUMINAMATH_CALUDE_work_completion_time_l600_60084

theorem work_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a = 4 → b = 12 → 1 / (1 / a + 1 / b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l600_60084


namespace NUMINAMATH_CALUDE_students_playing_neither_l600_60068

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 36)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l600_60068


namespace NUMINAMATH_CALUDE_total_tickets_sold_l600_60076

/-- Represents the number of tickets sold for a theater performance --/
structure TheaterTickets where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total revenue from ticket sales --/
def totalRevenue (tickets : TheaterTickets) : ℕ :=
  12 * tickets.orchestra + 8 * tickets.balcony

/-- Theorem stating the total number of tickets sold given the conditions --/
theorem total_tickets_sold : 
  ∃ (tickets : TheaterTickets), 
    totalRevenue tickets = 3320 ∧ 
    tickets.balcony = tickets.orchestra + 140 ∧
    tickets.orchestra + tickets.balcony = 360 := by
  sorry

#check total_tickets_sold

end NUMINAMATH_CALUDE_total_tickets_sold_l600_60076


namespace NUMINAMATH_CALUDE_age_sum_problem_l600_60099

theorem age_sum_problem :
  ∀ (y k : ℕ+),
    y * (2 * y) * k = 72 →
    y + (2 * y) + k = 13 :=
by sorry

end NUMINAMATH_CALUDE_age_sum_problem_l600_60099


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l600_60052

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The main theorem -/
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 5 * a 7 = 2)
  (h_sum : a 2 + a 10 = 3) :
  a 12 / a 4 = 2 ∨ a 12 / a 4 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l600_60052


namespace NUMINAMATH_CALUDE_couch_money_calculation_l600_60086

theorem couch_money_calculation (quarters : ℕ) (pennies : ℕ) 
  (quarter_value : ℚ) (penny_value : ℚ) :
  quarters = 12 →
  pennies = 7 →
  quarter_value = 25 / 100 →
  penny_value = 1 / 100 →
  quarters * quarter_value + pennies * penny_value = 307 / 100 := by
sorry

end NUMINAMATH_CALUDE_couch_money_calculation_l600_60086


namespace NUMINAMATH_CALUDE_bicycle_inventory_decrease_is_58_l600_60069

/-- Calculates the decrease in bicycle inventory from January 1 to October 1 -/
def bicycle_inventory_decrease : ℕ :=
  let initial_inventory : ℕ := 200
  let feb_to_june_decrease : ℕ := 4 + 6 + 8 + 10 + 12
  let july_decrease : ℕ := 14
  let august_decrease : ℕ := 16 + 20  -- Including sales event
  let september_decrease : ℕ := 18
  let new_shipment : ℕ := 50
  (feb_to_june_decrease + july_decrease + august_decrease + september_decrease) - new_shipment

/-- Theorem stating that the bicycle inventory decrease from January 1 to October 1 is 58 -/
theorem bicycle_inventory_decrease_is_58 : bicycle_inventory_decrease = 58 := by
  sorry

#eval bicycle_inventory_decrease

end NUMINAMATH_CALUDE_bicycle_inventory_decrease_is_58_l600_60069


namespace NUMINAMATH_CALUDE_totalPaintingCost_l600_60034

/-- Calculates the sum of digits for a given range of an arithmetic sequence -/
def sumOfDigits (start : Nat) (diff : Nat) (count : Nat) : Nat :=
  sorry

/-- Calculates the total cost to paint house numbers on one side of the street -/
def sideCost (start : Nat) (diff : Nat) (count : Nat) : Nat :=
  sorry

/-- The total cost to paint all house numbers on the street -/
theorem totalPaintingCost : 
  let eastSideCost := sideCost 5 7 25
  let westSideCost := sideCost 6 8 25
  eastSideCost + westSideCost = 123 := by
  sorry

end NUMINAMATH_CALUDE_totalPaintingCost_l600_60034


namespace NUMINAMATH_CALUDE_monotonically_decreasing_interval_l600_60020

-- Define the function f(x) = x³ - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem monotonically_decreasing_interval :
  ∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_interval_l600_60020


namespace NUMINAMATH_CALUDE_undefined_rational_expression_l600_60011

theorem undefined_rational_expression (x : ℝ) :
  (x^2 - 16*x + 64 = 0) ↔ (x = 8) :=
by sorry

end NUMINAMATH_CALUDE_undefined_rational_expression_l600_60011


namespace NUMINAMATH_CALUDE_coronavirus_recoveries_day2_l600_60004

/-- Proves that the number of recoveries on day 2 is 50, given the conditions of the Coronavirus case problem. -/
theorem coronavirus_recoveries_day2 
  (initial_cases : ℕ) 
  (day2_increase : ℕ) 
  (day3_new_cases : ℕ) 
  (day3_recoveries : ℕ) 
  (total_cases_day3 : ℕ) 
  (h1 : initial_cases = 2000)
  (h2 : day2_increase = 500)
  (h3 : day3_new_cases = 1500)
  (h4 : day3_recoveries = 200)
  (h5 : total_cases_day3 = 3750) :
  ∃ (day2_recoveries : ℕ), 
    initial_cases + day2_increase - day2_recoveries + day3_new_cases - day3_recoveries = total_cases_day3 ∧ 
    day2_recoveries = 50 := by
  sorry

end NUMINAMATH_CALUDE_coronavirus_recoveries_day2_l600_60004


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l600_60030

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 2*x^3 = (x^2 + 7*x + 2) * q + (33*x^2 + 10*x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l600_60030


namespace NUMINAMATH_CALUDE_largest_fraction_l600_60053

theorem largest_fraction : (5 : ℚ) / 6 > 3 / 4 ∧ (5 : ℚ) / 6 > 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l600_60053


namespace NUMINAMATH_CALUDE_xiaogangSavings_l600_60029

/-- Represents the correct inequality for Xiaogang's savings plan -/
theorem xiaogangSavings (x : ℕ) (initialSavings : ℕ) (monthlySavings : ℕ) (targetAmount : ℕ) : 
  initialSavings = 50 → monthlySavings = 30 → targetAmount = 280 →
  (monthlySavings * x + initialSavings ≥ targetAmount ↔ 
   x ≥ (targetAmount - initialSavings) / monthlySavings) :=
by sorry

end NUMINAMATH_CALUDE_xiaogangSavings_l600_60029


namespace NUMINAMATH_CALUDE_third_bottle_volume_is_250ml_l600_60082

/-- Represents the volume of milk in a bottle -/
structure MilkBottle where
  volume : ℝ
  unit : String

/-- Converts liters to milliliters -/
def litersToMilliliters (liters : ℝ) : ℝ := liters * 1000

/-- Calculates the volume of the third milk bottle -/
def thirdBottleVolume (bottle1 : MilkBottle) (bottle2 : MilkBottle) (totalVolume : ℝ) : ℝ :=
  litersToMilliliters totalVolume - (litersToMilliliters bottle1.volume + bottle2.volume)

/-- Theorem: The third milk bottle contains 250 milliliters -/
theorem third_bottle_volume_is_250ml 
  (bottle1 : MilkBottle) 
  (bottle2 : MilkBottle) 
  (totalVolume : ℝ) :
  bottle1.volume = 2 ∧ 
  bottle1.unit = "liters" ∧
  bottle2.volume = 750 ∧ 
  bottle2.unit = "milliliters" ∧
  totalVolume = 3 →
  thirdBottleVolume bottle1 bottle2 totalVolume = 250 := by
  sorry

end NUMINAMATH_CALUDE_third_bottle_volume_is_250ml_l600_60082


namespace NUMINAMATH_CALUDE_dave_spent_29_dollars_l600_60057

/-- Represents the cost of rides for a day at the fair -/
structure DayAtFair where
  rides : List ℕ

/-- Calculates the total cost of rides for a day -/
def totalCost (day : DayAtFair) : ℕ :=
  day.rides.sum

/-- Represents Dave's two days at the fair -/
def davesFairDays : List DayAtFair := [
  { rides := [4, 5, 3, 2] },  -- First day
  { rides := [5, 6, 4] }     -- Second day
]

theorem dave_spent_29_dollars : 
  (davesFairDays.map totalCost).sum = 29 := by
  sorry

end NUMINAMATH_CALUDE_dave_spent_29_dollars_l600_60057


namespace NUMINAMATH_CALUDE_frog_safety_probability_l600_60009

/-- The probability of the frog reaching safety when starting from platform N -/
noncomputable def P (N : ℕ) : ℝ :=
  sorry

/-- The number of platforms -/
def num_platforms : ℕ := 9

/-- The starting platform of the frog -/
def start_platform : ℕ := 2

theorem frog_safety_probability :
  -- Conditions
  (∀ N : ℕ, 0 < N → N < num_platforms - 1 →
    P N = (N : ℝ) / 8 * P (N - 1) + (1 - (N : ℝ) / 8) * P (N + 1)) →
  P 0 = 0 →
  P (num_platforms - 1) = 1 →
  -- Theorem to prove
  P start_platform = 21 / 64 :=
sorry

end NUMINAMATH_CALUDE_frog_safety_probability_l600_60009


namespace NUMINAMATH_CALUDE_pen_pricing_gain_percentage_l600_60019

theorem pen_pricing_gain_percentage 
  (cost_price selling_price : ℝ) 
  (h : 20 * cost_price = 12 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 200 / 3 :=
by sorry

end NUMINAMATH_CALUDE_pen_pricing_gain_percentage_l600_60019


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l600_60042

/-- Represents the price and volume of orangeade on a given day -/
structure OrangeadeDay where
  orange_juice : ℝ
  water : ℝ
  price : ℝ

/-- The orangeade scenario over two days -/
def OrangeadeScenario (day1 day2 : OrangeadeDay) : Prop :=
  day1.orange_juice > 0 ∧
  day1.water = day1.orange_juice ∧
  day2.orange_juice = day1.orange_juice ∧
  day2.water = 2 * day1.water ∧
  day1.price = 0.5 ∧
  (day1.orange_juice + day1.water) * day1.price = (day2.orange_juice + day2.water) * day2.price

theorem orangeade_price_day2 (day1 day2 : OrangeadeDay) 
  (h : OrangeadeScenario day1 day2) : day2.price = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l600_60042


namespace NUMINAMATH_CALUDE_problem_solution_l600_60094

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 591 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l600_60094


namespace NUMINAMATH_CALUDE_equation_solution_l600_60066

theorem equation_solution : 
  ∃ (x₁ x₂ : ℚ), x₁ = 1/3 ∧ x₂ = 1/2 ∧ 
  (∀ x : ℚ, 6*x^2 - 3*x - 1 = 2*x - 2 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l600_60066


namespace NUMINAMATH_CALUDE_xy_range_l600_60095

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2/x + 3*y + 4/y = 10) : 1 ≤ x*y ∧ x*y ≤ 8/3 := by
  sorry

end NUMINAMATH_CALUDE_xy_range_l600_60095


namespace NUMINAMATH_CALUDE_weekend_haircut_price_l600_60035

theorem weekend_haircut_price (weekday_price : ℝ) (weekend_markup : ℝ) : 
  weekday_price = 18 → weekend_markup = 0.5 → weekday_price * (1 + weekend_markup) = 27 := by
  sorry

end NUMINAMATH_CALUDE_weekend_haircut_price_l600_60035


namespace NUMINAMATH_CALUDE_distribution_five_three_l600_60010

/-- The number of ways to distribute n distinct items among k distinct categories,
    where each item must be used exactly once. -/
def distributionCount (n k : ℕ) : ℕ :=
  k^n - (k * 1) - (Nat.choose k 2 * (2^n - 2))

/-- Theorem stating that distributing 5 distinct items among 3 distinct categories,
    where each item must be used exactly once, results in 150 possible distributions. -/
theorem distribution_five_three : distributionCount 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribution_five_three_l600_60010


namespace NUMINAMATH_CALUDE_rope_division_l600_60087

theorem rope_division (rope_length : ℚ) (n_parts : ℕ) (h1 : rope_length = 8/15) (h2 : n_parts = 3) :
  let part_fraction : ℚ := 1 / n_parts
  let part_length : ℚ := rope_length / n_parts
  (part_fraction = 1/3) ∧ (part_length = 8/45) := by
  sorry

end NUMINAMATH_CALUDE_rope_division_l600_60087


namespace NUMINAMATH_CALUDE_blue_balls_count_l600_60022

theorem blue_balls_count (total : ℕ) (prob : ℚ) (blue : ℕ) : 
  total = 15 →
  prob = 1 / 21 →
  (blue * (blue - 1)) / (total * (total - 1)) = prob →
  blue = 5 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l600_60022


namespace NUMINAMATH_CALUDE_untouched_shapes_after_game_l600_60074

-- Define the game state
structure GameState where
  triangles : Nat
  squares : Nat
  pentagons : Nat
  untouchedShapes : Nat
  turn : Nat

-- Define the initial game state
def initialState : GameState :=
  { triangles := 3
  , squares := 4
  , pentagons := 5
  , untouchedShapes := 12
  , turn := 0
  }

-- Define a move function for Petya
def petyaMove (state : GameState) : GameState :=
  { state with
    untouchedShapes := state.untouchedShapes - (if state.turn = 0 then 1 else 0)
    turn := state.turn + 1
  }

-- Define a move function for Vasya
def vasyaMove (state : GameState) : GameState :=
  { state with
    untouchedShapes := state.untouchedShapes - 1
    turn := state.turn + 1
  }

-- Define the final state after 10 turns
def finalState : GameState :=
  (List.range 5).foldl (fun state _ => vasyaMove (petyaMove state)) initialState

-- Theorem statement
theorem untouched_shapes_after_game :
  finalState.untouchedShapes = 6 := by sorry

end NUMINAMATH_CALUDE_untouched_shapes_after_game_l600_60074


namespace NUMINAMATH_CALUDE_school_problem_l600_60093

/-- Represents a school with a specific number of classes and students. -/
structure School where
  num_classes : Nat
  largest_class : Nat
  difference : Nat
  total_students : Nat

/-- Calculates the total number of students in the school. -/
def calculate_total (s : School) : Nat :=
  let series := List.range s.num_classes
  series.foldr (fun i acc => acc + s.largest_class - i * s.difference) 0

/-- Theorem stating the properties of the school in the problem. -/
theorem school_problem :
  ∃ (s : School),
    s.num_classes = 5 ∧
    s.largest_class = 32 ∧
    s.difference = 2 ∧
    s.total_students = 140 ∧
    calculate_total s = s.total_students :=
  sorry

end NUMINAMATH_CALUDE_school_problem_l600_60093


namespace NUMINAMATH_CALUDE_diophantine_equation_prime_divisor_l600_60073

theorem diophantine_equation_prime_divisor (x y n : ℕ) 
  (h1 : x ≥ 3) (h2 : n ≥ 2) (h3 : x^2 + 5 = y^n) :
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p ≡ 1 [MOD 4] := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_prime_divisor_l600_60073


namespace NUMINAMATH_CALUDE_disrespectful_quadratic_max_sum_at_one_l600_60072

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  t : ℝ
  k : ℝ

/-- The value of the polynomial at x -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  x^2 - p.t * x + p.k

/-- The composition of the polynomial with itself -/
def QuadraticPolynomial.compose (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.eval (p.eval x)

/-- A quadratic polynomial is disrespectful if p(p(x)) = 0 has exactly four real solutions -/
def QuadraticPolynomial.isDisrespectful (p : QuadraticPolynomial) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (∀ x : ℝ, p.compose x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

/-- The sum of coefficients of a quadratic polynomial -/
def QuadraticPolynomial.sumCoefficients (p : QuadraticPolynomial) : ℝ :=
  1 - p.t + p.k

/-- The theorem to be proved -/
theorem disrespectful_quadratic_max_sum_at_one :
  ∃ (p : QuadraticPolynomial),
    p.isDisrespectful ∧
    (∀ q : QuadraticPolynomial, q.isDisrespectful → p.sumCoefficients ≥ q.sumCoefficients) ∧
    p.eval 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_disrespectful_quadratic_max_sum_at_one_l600_60072


namespace NUMINAMATH_CALUDE_tetrahedron_altitude_volume_inequality_l600_60015

/-- A tetrahedron with volume and altitudes -/
structure Tetrahedron where
  volume : ℝ
  altitude : Fin 4 → ℝ

/-- Predicate to check if a tetrahedron is right-angled -/
def isRightAngled (t : Tetrahedron) : Prop := sorry

/-- Theorem stating the relationship between altitudes and volume of a tetrahedron -/
theorem tetrahedron_altitude_volume_inequality (t : Tetrahedron) :
  ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    t.altitude i * t.altitude j * t.altitude k ≤ 6 * t.volume ∧
    (t.altitude i * t.altitude j * t.altitude k = 6 * t.volume ↔ isRightAngled t) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_altitude_volume_inequality_l600_60015


namespace NUMINAMATH_CALUDE_beka_miles_l600_60077

/-- The number of miles Jackson flew -/
def jackson_miles : ℕ := 563

/-- The additional miles Beka flew compared to Jackson -/
def additional_miles : ℕ := 310

/-- Theorem: Given the conditions, Beka flew 873 miles -/
theorem beka_miles : jackson_miles + additional_miles = 873 := by
  sorry

end NUMINAMATH_CALUDE_beka_miles_l600_60077


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l600_60067

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l600_60067


namespace NUMINAMATH_CALUDE_completing_square_correct_transformation_l600_60024

theorem completing_square_correct_transformation :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 := by sorry

end NUMINAMATH_CALUDE_completing_square_correct_transformation_l600_60024


namespace NUMINAMATH_CALUDE_fifteen_to_binary_l600_60045

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem fifteen_to_binary :
  decimal_to_binary 15 = [1, 1, 1, 1] :=
by sorry

end NUMINAMATH_CALUDE_fifteen_to_binary_l600_60045


namespace NUMINAMATH_CALUDE_investment_interest_rate_l600_60023

/-- Proves that given the specified investment conditions, the annual interest rate of the second certificate is 8% -/
theorem investment_interest_rate 
  (initial_investment : ℝ)
  (first_rate : ℝ)
  (second_rate : ℝ)
  (final_value : ℝ)
  (h1 : initial_investment = 15000)
  (h2 : first_rate = 8)
  (h3 : final_value = 15612)
  (h4 : initial_investment * (1 + first_rate / 400) * (1 + second_rate / 400) = final_value) :
  second_rate = 8 := by
    sorry

#check investment_interest_rate

end NUMINAMATH_CALUDE_investment_interest_rate_l600_60023
