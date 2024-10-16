import Mathlib

namespace NUMINAMATH_CALUDE_unique_sums_count_l1165_116559

def set_A : Finset ℕ := {2, 3, 5, 8}
def set_B : Finset ℕ := {1, 4, 6, 7}

theorem unique_sums_count : 
  Finset.card ((set_A.product set_B).image (fun p => p.1 + p.2)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l1165_116559


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l1165_116591

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l1165_116591


namespace NUMINAMATH_CALUDE_probability_multiple_3_or_4_in_30_l1165_116544

def is_multiple_of_3_or_4 (n : ℕ) : Bool :=
  n % 3 = 0 || n % 4 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_3_or_4 |>.length

theorem probability_multiple_3_or_4_in_30 :
  count_multiples 30 / 30 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_3_or_4_in_30_l1165_116544


namespace NUMINAMATH_CALUDE_subset_union_existence_l1165_116592

theorem subset_union_existence (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 2^(n-1) + 1) :
  ∀ (A : Fin m → Set (Fin n)), 
    (∀ j, A j ≠ ∅) → 
    (∀ i j, i ≠ j → A i ≠ A j) → 
    ∃ i j k, A i ∪ A j = A k := by
  sorry

end NUMINAMATH_CALUDE_subset_union_existence_l1165_116592


namespace NUMINAMATH_CALUDE_f_properties_l1165_116536

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (2 * Real.sqrt 3 * Real.cos x - Real.sin x) + 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (x y : ℝ), -π/4 ≤ x ∧ x < y ∧ y ≤ π/6 → f x < f y) ∧
  (∀ (x y : ℝ), π/6 ≤ x ∧ x < y ∧ y ≤ π/4 → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1165_116536


namespace NUMINAMATH_CALUDE_pirate_treasure_l1165_116555

theorem pirate_treasure (x : ℕ) : x > 0 → (
  let paul_coins := x
  let pete_coins := x^2
  paul_coins + pete_coins = 12
) ↔ (
  -- Pete's coins follow the pattern 1, 3, 5, ..., (2x-1)
  pete_coins = x^2 ∧
  -- Paul receives x coins in total
  paul_coins = x ∧
  -- Pete has exactly three times as many coins as Paul
  pete_coins = 3 * paul_coins ∧
  -- All coins are distributed (implied by the other conditions)
  True
) := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_l1165_116555


namespace NUMINAMATH_CALUDE_probability_a_speaks_truth_l1165_116588

theorem probability_a_speaks_truth 
  (prob_b : ℝ) 
  (prob_both : ℝ) 
  (h1 : prob_b = 0.60)
  (h2 : prob_both = 0.33)
  (h3 : prob_both = prob_a * prob_b)
  : prob_a = 0.55 :=
by sorry

end NUMINAMATH_CALUDE_probability_a_speaks_truth_l1165_116588


namespace NUMINAMATH_CALUDE_marks_candy_bars_l1165_116511

def total_candy_bars (snickers mars butterfingers : ℕ) : ℕ :=
  snickers + mars + butterfingers

theorem marks_candy_bars : total_candy_bars 3 2 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_marks_candy_bars_l1165_116511


namespace NUMINAMATH_CALUDE_third_quiz_score_l1165_116590

theorem third_quiz_score (score1 score2 score3 : ℕ) : 
  score1 = 91 → 
  score2 = 90 → 
  (score1 + score2 + score3) / 3 = 91 → 
  score3 = 92 := by
sorry

end NUMINAMATH_CALUDE_third_quiz_score_l1165_116590


namespace NUMINAMATH_CALUDE_airplane_altitude_l1165_116564

theorem airplane_altitude (alice_bob_distance : ℝ) (alice_elevation : ℝ) (bob_elevation : ℝ) 
  (h_distance : alice_bob_distance = 15)
  (h_alice_elevation : alice_elevation = 25 * π / 180)
  (h_bob_elevation : bob_elevation = 45 * π / 180) :
  ∃ (altitude : ℝ), 3.7 < altitude ∧ altitude < 3.9 := by
  sorry


end NUMINAMATH_CALUDE_airplane_altitude_l1165_116564


namespace NUMINAMATH_CALUDE_shaded_square_ratio_l1165_116552

/-- The ratio of the area of a shaded square to the area of a large square in a specific grid configuration -/
theorem shaded_square_ratio : 
  ∀ (n : ℕ) (large_square_area shaded_square_area : ℝ),
  n = 5 →
  large_square_area = n^2 →
  shaded_square_area = 4 * (1/2) →
  shaded_square_area / large_square_area = 2/25 := by
sorry

end NUMINAMATH_CALUDE_shaded_square_ratio_l1165_116552


namespace NUMINAMATH_CALUDE_marble_groups_l1165_116553

theorem marble_groups (total_marbles : ℕ) (marbles_per_group : ℕ) (num_groups : ℕ) :
  total_marbles = 64 →
  marbles_per_group = 2 →
  num_groups * marbles_per_group = total_marbles →
  num_groups = 32 := by
  sorry

end NUMINAMATH_CALUDE_marble_groups_l1165_116553


namespace NUMINAMATH_CALUDE_point_d_coordinates_l1165_116525

/-- Given two points P and Q in the plane, and a point D on the line segment PQ such that
    PD = 2DQ, prove that D has specific coordinates. -/
theorem point_d_coordinates (P Q D : ℝ × ℝ) : 
  P = (-3, -2) →
  Q = (5, 10) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • P + t • Q) →
  (P.1 - D.1)^2 + (P.2 - D.2)^2 = 4 * ((Q.1 - D.1)^2 + (Q.2 - D.2)^2) →
  D = (3, 7) := by
sorry


end NUMINAMATH_CALUDE_point_d_coordinates_l1165_116525


namespace NUMINAMATH_CALUDE_no_triangle_with_special_side_ratios_l1165_116558

theorem no_triangle_with_special_side_ratios :
  ¬ ∃ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (a + b > c ∧ b + c > a ∧ a + c > b) ∧
    ((a = b / 2 ∧ a = c / 3) ∨ 
     (b = a / 2 ∧ b = c / 3) ∨ 
     (c = a / 2 ∧ c = b / 3)) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_with_special_side_ratios_l1165_116558


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1165_116533

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  2*x^3 - x^2 + 23*x - 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1165_116533


namespace NUMINAMATH_CALUDE_square_plus_one_positive_l1165_116534

theorem square_plus_one_positive (a : ℚ) : 0 < a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_positive_l1165_116534


namespace NUMINAMATH_CALUDE_original_numerator_proof_l1165_116570

theorem original_numerator_proof (n : ℚ) : 
  (n + 3) / 12 = 2 / 3 → n / 9 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_original_numerator_proof_l1165_116570


namespace NUMINAMATH_CALUDE_greatest_x_value_l1165_116571

theorem greatest_x_value (x : ℝ) : 
  (x^2 - x - 90) / (x - 9) = 4 / (x + 6) → x ≤ -7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1165_116571


namespace NUMINAMATH_CALUDE_center_is_five_l1165_116579

-- Define the grid type
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define the property of consecutive numbers sharing an edge
def ConsecutiveShareEdge (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, g i j = g k l + 1 →
    ((i = k ∧ j.val + 1 = l.val) ∨ (i = k ∧ j.val = l.val + 1) ∨
     (i.val + 1 = k.val ∧ j = l) ∨ (i.val = k.val + 1 ∧ j = l))

-- Define the sum of corner numbers
def CornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

-- Define the sum of numbers along one diagonal
def DiagonalSum (g : Grid) : Nat :=
  g 0 0 + g 1 1 + g 2 2

-- Theorem statement
theorem center_is_five (g : Grid) 
  (grid_nums : ∀ i j, g i j ∈ Finset.range 9)
  (consecutive_edge : ConsecutiveShareEdge g)
  (corner_sum : CornerSum g = 20)
  (diagonal_sum : DiagonalSum g = 15) :
  g 1 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_center_is_five_l1165_116579


namespace NUMINAMATH_CALUDE_sign_of_c_l1165_116575

theorem sign_of_c (a b c : ℝ) 
  (h1 : a * b / c < 0) 
  (h2 : a * b < 0) : 
  c > 0 := by sorry

end NUMINAMATH_CALUDE_sign_of_c_l1165_116575


namespace NUMINAMATH_CALUDE_M_on_line_l_line_l_equation_AB_length_l1165_116505

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Define point M
def point_M : ℝ × ℝ := (2, 1)

-- Define that M is on line l
theorem M_on_line_l : line_l point_M.1 point_M.2 := by sorry

-- Define that A and B are on the parabola
axiom A_on_parabola : ∃ (x y : ℝ), parabola x y ∧ line_l x y
axiom B_on_parabola : ∃ (x y : ℝ), parabola x y ∧ line_l x y

-- Define that M is the midpoint of AB
axiom M_midpoint_AB : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
  point_M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

-- Theorem 1: The equation of line l is 2x - y - 3 = 0
theorem line_l_equation : ∀ (x y : ℝ), line_l x y ↔ 2*x - y - 3 = 0 := by sorry

-- Theorem 2: The length of segment AB is √35
theorem AB_length : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    point_M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 35 := by sorry

end NUMINAMATH_CALUDE_M_on_line_l_line_l_equation_AB_length_l1165_116505


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_four_l1165_116515

theorem fraction_zero_implies_x_equals_four (x : ℝ) : 
  (16 - x^2) / (x + 4) = 0 ∧ x + 4 ≠ 0 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_four_l1165_116515


namespace NUMINAMATH_CALUDE_min_cubes_for_specific_block_l1165_116595

/-- The minimum number of cubes needed to create a hollow block -/
def min_cubes_hollow_block (length width depth : ℕ) : ℕ :=
  let total_cubes := length * width * depth
  let hollow_length := length - 2
  let hollow_width := width - 2
  let hollow_depth := depth - 2
  let hollow_cubes := hollow_length * hollow_width * hollow_depth
  total_cubes - hollow_cubes

/-- Theorem stating the minimum number of cubes needed for the specific block -/
theorem min_cubes_for_specific_block :
  min_cubes_hollow_block 4 10 7 = 200 := by
  sorry

#eval min_cubes_hollow_block 4 10 7

end NUMINAMATH_CALUDE_min_cubes_for_specific_block_l1165_116595


namespace NUMINAMATH_CALUDE_negative_three_cubed_equality_l1165_116578

theorem negative_three_cubed_equality : -3^3 = (-3)^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_cubed_equality_l1165_116578


namespace NUMINAMATH_CALUDE_short_bushes_count_l1165_116556

/-- The number of short bushes initially in the park -/
def initial_short_bushes : ℕ := 37

/-- The number of short bushes planted by workers -/
def planted_short_bushes : ℕ := 20

/-- The total number of short bushes after planting -/
def total_short_bushes : ℕ := 57

/-- Theorem stating that the initial number of short bushes plus the planted ones equals the total -/
theorem short_bushes_count : 
  initial_short_bushes + planted_short_bushes = total_short_bushes := by
  sorry

end NUMINAMATH_CALUDE_short_bushes_count_l1165_116556


namespace NUMINAMATH_CALUDE_square_root_fraction_simplification_l1165_116551

theorem square_root_fraction_simplification :
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (25 + 36)) = 17 / Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_simplification_l1165_116551


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l1165_116529

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The function f(x) = x^2 - |x + a| -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 - |x + a|

/-- If f(x) = x^2 - |x + a| is an even function, then a = 0 -/
theorem even_function_implies_a_zero (a : ℝ) : IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l1165_116529


namespace NUMINAMATH_CALUDE_jesses_friends_l1165_116504

theorem jesses_friends (bananas_per_friend : ℝ) (total_bananas : ℕ) 
  (h1 : bananas_per_friend = 21.0) 
  (h2 : total_bananas = 63) : 
  (total_bananas : ℝ) / bananas_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_jesses_friends_l1165_116504


namespace NUMINAMATH_CALUDE_sin_135_degrees_l1165_116500

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l1165_116500


namespace NUMINAMATH_CALUDE_square_diff_divided_by_24_l1165_116510

theorem square_diff_divided_by_24 : (145^2 - 121^2) / 24 = 266 := by sorry

end NUMINAMATH_CALUDE_square_diff_divided_by_24_l1165_116510


namespace NUMINAMATH_CALUDE_equal_squares_count_l1165_116593

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Cell

/-- Defines the specific coloring pattern of the grid -/
def initial_grid : Grid :=
  fun i j => 
    if (i = 2 ∧ j = 2) ∨ 
       (i = 1 ∧ j = 3) ∨ 
       (i = 3 ∧ j = 1) ∨ 
       (i = 3 ∧ j = 3) ∨ 
       (i = 3 ∧ j = 5) 
    then Cell.Black 
    else Cell.White

/-- Checks if a square in the grid has equal number of black and white cells -/
def has_equal_cells (g : Grid) (top_left_i top_left_j size : Nat) : Bool :=
  sorry

/-- Counts the number of squares with equal black and white cells -/
def count_equal_squares (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that there are exactly 16 squares with equal black and white cells -/
theorem equal_squares_count : count_equal_squares initial_grid = 16 :=
  sorry

end NUMINAMATH_CALUDE_equal_squares_count_l1165_116593


namespace NUMINAMATH_CALUDE_quadratic_m_value_l1165_116518

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  point_zero : ℝ
  point_five : ℝ

/-- The properties of the given quadratic function -/
def given_quadratic : QuadraticFunction where
  a := 0
  b := 0
  c := 0
  min_value := -10
  min_x := -1
  point_zero := 8
  point_five := 0  -- This is the m we want to prove

/-- The theorem stating the value of m -/
theorem quadratic_m_value (f : QuadraticFunction) (h1 : f.min_value = -10) 
    (h2 : f.min_x = -1) (h3 : f.point_zero = 8) : f.point_five = 638 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_m_value_l1165_116518


namespace NUMINAMATH_CALUDE_middle_number_problem_l1165_116530

theorem middle_number_problem (x y z : ℤ) 
  (sum_xy : x + y = 15)
  (sum_xz : x + z = 18)
  (sum_yz : y + z = 22) :
  y = (19 : ℚ) / 2 := by
sorry

end NUMINAMATH_CALUDE_middle_number_problem_l1165_116530


namespace NUMINAMATH_CALUDE_yannas_cookies_l1165_116522

/-- Yanna's cookie baking problem -/
theorem yannas_cookies
  (morning_butter_cookies : ℕ)
  (morning_biscuits : ℕ)
  (afternoon_butter_cookies : ℕ)
  (afternoon_biscuits : ℕ)
  (h1 : morning_butter_cookies = 20)
  (h2 : morning_biscuits = 40)
  (h3 : afternoon_butter_cookies = 10)
  (h4 : afternoon_biscuits = 20) :
  (morning_biscuits + afternoon_biscuits) - (morning_butter_cookies + afternoon_butter_cookies) = 30 :=
by sorry

end NUMINAMATH_CALUDE_yannas_cookies_l1165_116522


namespace NUMINAMATH_CALUDE_not_all_same_probability_l1165_116501

def roll_five_eight_sided_dice : ℕ := 8^5

def same_number_outcomes : ℕ := 8

theorem not_all_same_probability :
  (roll_five_eight_sided_dice - same_number_outcomes) / roll_five_eight_sided_dice = 4095 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_not_all_same_probability_l1165_116501


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l1165_116524

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l1165_116524


namespace NUMINAMATH_CALUDE_perfect_square_equation_l1165_116582

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_equation_l1165_116582


namespace NUMINAMATH_CALUDE_abs_eq_neg_iff_nonpos_l1165_116596

theorem abs_eq_neg_iff_nonpos (a : ℝ) : |a| = -a ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_abs_eq_neg_iff_nonpos_l1165_116596


namespace NUMINAMATH_CALUDE_min_ab_perpendicular_lines_l1165_116520

/-- Given two perpendicular lines and b > 0, the minimum value of ab is 2 -/
theorem min_ab_perpendicular_lines (b : ℝ) (a : ℝ) (h : b > 0) :
  (∃ x y, (b^2 + 1) * x + a * y + 2 = 0) ∧ 
  (∃ x y, x - b^2 * y - 1 = 0) ∧
  ((b^2 + 1) * (1 / b^2) = -1) →
  (∀ c, ab ≥ c → c ≤ 2) ∧ (∃ d, ab = d ∧ d = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_ab_perpendicular_lines_l1165_116520


namespace NUMINAMATH_CALUDE_opposite_of_a_is_two_l1165_116586

theorem opposite_of_a_is_two (a : ℝ) : -a = 2 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_a_is_two_l1165_116586


namespace NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l1165_116502

theorem proposition_false_iff_a_in_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l1165_116502


namespace NUMINAMATH_CALUDE_square_of_negative_triple_l1165_116523

theorem square_of_negative_triple (a : ℝ) : (-3 * a)^2 = 9 * a^2 := by sorry

end NUMINAMATH_CALUDE_square_of_negative_triple_l1165_116523


namespace NUMINAMATH_CALUDE_cone_volume_l1165_116587

/-- A cone with base area π and lateral surface in the shape of a semicircle has volume (√3 / 3)π -/
theorem cone_volume (r h l : ℝ) : 
  r > 0 → h > 0 → l > 0 →
  π * r^2 = π →
  π * l = 2 * π * r →
  h^2 + r^2 = l^2 →
  (1/3) * π * r^2 * h = (Real.sqrt 3 / 3) * π := by
sorry

end NUMINAMATH_CALUDE_cone_volume_l1165_116587


namespace NUMINAMATH_CALUDE_apprentice_production_rate_l1165_116508

/-- The number of parts the master processes per day -/
def master_parts_per_day : ℚ := 112.5

/-- The number of parts the apprentice processes per day -/
def apprentice_parts_per_day : ℚ := 45

/-- The total number of parts to be processed -/
def total_parts : ℕ := 765

/-- Theorem stating that the given conditions lead to the apprentice processing 45 parts per day -/
theorem apprentice_production_rate :
  (4 * master_parts_per_day + 7 * apprentice_parts_per_day = total_parts) ∧
  (6 * master_parts_per_day + 2 * apprentice_parts_per_day = total_parts) →
  apprentice_parts_per_day = 45 := by
  sorry

#check apprentice_production_rate

end NUMINAMATH_CALUDE_apprentice_production_rate_l1165_116508


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1165_116563

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {-2, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1165_116563


namespace NUMINAMATH_CALUDE_opaque_arrangements_count_l1165_116589

/-- Represents a glass piece with one painted triangular section -/
structure GlassPiece where
  rotation : Fin 4  -- 0, 1, 2, 3 representing 0°, 90°, 180°, 270°

/-- Represents a stack of glass pieces -/
def GlassStack := List GlassPiece

/-- Checks if a given stack of glass pieces is completely opaque -/
def is_opaque (stack : GlassStack) : Bool :=
  sorry

/-- Counts the number of opaque arrangements for 5 glass pieces -/
def count_opaque_arrangements : Nat :=
  sorry

/-- The main theorem stating the correct number of opaque arrangements -/
theorem opaque_arrangements_count :
  count_opaque_arrangements = 7200 :=
sorry

end NUMINAMATH_CALUDE_opaque_arrangements_count_l1165_116589


namespace NUMINAMATH_CALUDE_point_not_on_graph_l1165_116580

theorem point_not_on_graph : ¬ ∃ (y : ℝ), y = (-2 - 1) / (-2 + 2) ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l1165_116580


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1165_116516

theorem polynomial_factorization (m : ℝ) : 
  (∀ x, x^2 + m*x - 6 = (x - 2) * (x + 3)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1165_116516


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1165_116541

theorem other_root_of_quadratic (p : ℝ) : 
  (2 : ℝ)^2 + 4*2 - p = 0 → 
  ∃ (x : ℝ), x^2 + 4*x - p = 0 ∧ x = -6 := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1165_116541


namespace NUMINAMATH_CALUDE_triangle_problem_l1165_116568

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (c - Real.sqrt 3 * b * Real.sin A = (a^2 + c^2 - b^2) / (2 * c) - b) →
  (A = π / 3) ∧
  ((b = c / 4) → 
   (a * 2 * Real.sqrt 3 = b * c * Real.sin A) → 
   (a = 13)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1165_116568


namespace NUMINAMATH_CALUDE_three_color_cubes_l1165_116576

theorem three_color_cubes (total red blue green : ℕ) 
  (h_total : total = 100)
  (h_red : red = 80)
  (h_blue : blue = 85)
  (h_green : green = 75)
  (h_red_le : red ≤ total)
  (h_blue_le : blue ≤ total)
  (h_green_le : green ≤ total) :
  ∃ n : ℕ, 40 ≤ n ∧ n ≤ 75 ∧ n = total - ((total - red) + (total - blue) + (total - green)) :=
sorry

end NUMINAMATH_CALUDE_three_color_cubes_l1165_116576


namespace NUMINAMATH_CALUDE_negation_of_existence_square_gt_power_negation_l1165_116535

theorem negation_of_existence (p : ℕ → Prop) :
  (¬∃ n : ℕ, n > 1 ∧ p n) ↔ (∀ n : ℕ, n > 1 → ¬(p n)) := by sorry

theorem square_gt_power_negation :
  (¬∃ n : ℕ, n > 1 ∧ n^2 > 2^n) ↔ (∀ n : ℕ, n > 1 → n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_gt_power_negation_l1165_116535


namespace NUMINAMATH_CALUDE_equation_solution_l1165_116598

theorem equation_solution :
  let f (x : ℝ) := x + 3 = 4 / (x - 2)
  ∀ x : ℝ, x ≠ 2 → (f x ↔ (x = (-1 + Real.sqrt 41) / 2 ∨ x = (-1 - Real.sqrt 41) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1165_116598


namespace NUMINAMATH_CALUDE_triangle_side_length_l1165_116594

/-- Given a triangle ABC with sides a, b, and c, if b = 5, c = 4, and cos(B - C) = 31/32, then a = 6 -/
theorem triangle_side_length (a b c : ℝ) (B C : ℝ) : 
  b = 5 → c = 4 → Real.cos (B - C) = 31/32 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1165_116594


namespace NUMINAMATH_CALUDE_derivative_of_f_at_1_l1165_116531

def f (x : ℝ) := x^2

theorem derivative_of_f_at_1 : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_at_1_l1165_116531


namespace NUMINAMATH_CALUDE_product_tens_digit_is_nine_l1165_116513

theorem product_tens_digit_is_nine (x : ℤ) : 
  0 ≤ x ∧ x ≤ 9 → 
  ((200 + 10 * x + 7) * 39 ≡ 90 [ZMOD 100] ↔ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_product_tens_digit_is_nine_l1165_116513


namespace NUMINAMATH_CALUDE_system_no_solution_l1165_116514

theorem system_no_solution (n : ℝ) : 
  (∀ x y z : ℝ, nx + y + z ≠ 2 ∨ x + ny + z ≠ 2 ∨ x + y + nz ≠ 2) ↔ n = -1 :=
by sorry

end NUMINAMATH_CALUDE_system_no_solution_l1165_116514


namespace NUMINAMATH_CALUDE_ladies_walking_group_miles_l1165_116549

/-- Calculates the total miles walked by a group of ladies over a number of days -/
def totalMilesWalked (groupSize : ℕ) (daysWalked : ℕ) (groupMiles : ℕ) (jamieExtra : ℕ) (sueExtra : ℕ) : ℕ :=
  groupSize * groupMiles * daysWalked + (jamieExtra + sueExtra) * daysWalked

/-- Theorem stating the total miles walked by the group under given conditions -/
theorem ladies_walking_group_miles :
  ∀ d : ℕ, totalMilesWalked 5 d 3 2 1 = 18 * d :=
by
  sorry

#check ladies_walking_group_miles

end NUMINAMATH_CALUDE_ladies_walking_group_miles_l1165_116549


namespace NUMINAMATH_CALUDE_price_increase_ratio_l1165_116599

theorem price_increase_ratio (original_price : ℝ) 
  (h1 : original_price > 0)
  (h2 : original_price * 1.3 = 364) : 
  364 / original_price = 1.3 := by
sorry

end NUMINAMATH_CALUDE_price_increase_ratio_l1165_116599


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l1165_116577

/-- Given an infinite geometric series with common ratio -1/4 and sum 40,
    the second term of the series is -12.5 -/
theorem geometric_series_second_term :
  ∀ (a : ℝ),
    (∑' n, a * (-1/4)^n) = 40 →
    a * (-1/4) = -12.5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l1165_116577


namespace NUMINAMATH_CALUDE_f_is_smallest_not_on_board_l1165_116512

/-- The game function that represents the number left on the board after subtraction -/
def g (k : ℕ) (x : ℕ) : ℕ := x^2 - k

/-- The smallest integer a such that g_{2n}(a) - g_{2n}(a-1) ≥ 3 -/
def x (n : ℕ) : ℕ := 2*n + 2

/-- The function f(2n) representing the smallest positive integer not written on the board -/
def f (n : ℕ) : ℕ := (2*n + 1)^2 - 2*n

/-- Theorem stating that f(2n) is the smallest positive integer not written on the board -/
theorem f_is_smallest_not_on_board (n : ℕ) :
  f n = (2*n + 1)^2 - 2*n ∧
  ∀ m < f n, ∃ i ≤ x n, m = g (2*n) i ∨ m = g (2*n) (i+1) :=
sorry

end NUMINAMATH_CALUDE_f_is_smallest_not_on_board_l1165_116512


namespace NUMINAMATH_CALUDE_inverse_B_cubed_l1165_116560

theorem inverse_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, 7; -2, -4]) : 
  (B⁻¹)^3 = !![11, 17; -10, -18] := by
  sorry

end NUMINAMATH_CALUDE_inverse_B_cubed_l1165_116560


namespace NUMINAMATH_CALUDE_roots_of_equation_l1165_116581

theorem roots_of_equation (x : ℝ) : 
  x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1165_116581


namespace NUMINAMATH_CALUDE_existence_of_equal_elements_l1165_116537

theorem existence_of_equal_elements (n p q : ℕ) (x : ℕ → ℤ)
  (h_pos : 0 < n ∧ 0 < p ∧ 0 < q)
  (h_n_gt : n > p + q)
  (h_x_bounds : x 0 = 0 ∧ x n = 0)
  (h_x_diff : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (x i - x (i-1) = p ∨ x i - x (i-1) = -q)) :
  ∃ (i j : ℕ), i < j ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
by sorry

end NUMINAMATH_CALUDE_existence_of_equal_elements_l1165_116537


namespace NUMINAMATH_CALUDE_intersection_M_N_l1165_116539

def M : Set ℝ := {x | 4 < x ∧ x < 8}
def N : Set ℝ := {x | x^2 - 6*x < 0}

theorem intersection_M_N : M ∩ N = {x | 4 < x ∧ x < 6} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1165_116539


namespace NUMINAMATH_CALUDE_curve_transformation_l1165_116527

theorem curve_transformation (x : ℝ) : 
  Real.sin (x + π / 2) = Real.sin (2 * (x + π / 12) + 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l1165_116527


namespace NUMINAMATH_CALUDE_survey_theorem_l1165_116550

def survey_problem (total : ℕ) (high_bp : ℕ) (heart : ℕ) (diabetes : ℕ) 
                   (bp_heart : ℕ) (bp_diabetes : ℕ) (heart_diabetes : ℕ) 
                   (all_three : ℕ) : Prop :=
  let teachers_with_condition := 
    (high_bp - bp_heart - bp_diabetes + all_three) +
    (heart - bp_heart - heart_diabetes + all_three) +
    (diabetes - bp_diabetes - heart_diabetes + all_three) +
    (bp_heart - all_three) + (bp_diabetes - all_three) + 
    (heart_diabetes - all_three) + all_three
  let teachers_without_condition := total - teachers_with_condition
  (teachers_without_condition : ℚ) / (total : ℚ) * 100 = 50/3

theorem survey_theorem : 
  survey_problem 150 90 50 30 25 10 15 5 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_theorem_l1165_116550


namespace NUMINAMATH_CALUDE_cubic_equation_fraction_value_l1165_116506

theorem cubic_equation_fraction_value (a : ℝ) : 
  a^3 + 3*a^2 + a = 0 → 
  (2022*a^2) / (a^4 + 2015*a^2 + 1) = 0 ∨ (2022*a^2) / (a^4 + 2015*a^2 + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_fraction_value_l1165_116506


namespace NUMINAMATH_CALUDE_temperature_function_correct_and_linear_l1165_116526

/-- Represents the temperature change per kilometer of altitude increase -/
def temperature_change_per_km : ℝ := -6

/-- Represents the ground temperature in Celsius -/
def ground_temperature : ℝ := 20

/-- Represents the temperature y in Celsius at a height of x kilometers above the ground -/
def temperature (x : ℝ) : ℝ := temperature_change_per_km * x + ground_temperature

theorem temperature_function_correct_and_linear :
  (∀ x : ℝ, temperature x = temperature_change_per_km * x + ground_temperature) ∧
  (∃ m b : ℝ, ∀ x : ℝ, temperature x = m * x + b) :=
by sorry

end NUMINAMATH_CALUDE_temperature_function_correct_and_linear_l1165_116526


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1165_116561

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1165_116561


namespace NUMINAMATH_CALUDE_car_fuel_usage_l1165_116554

/-- Proves that a car traveling at 40 miles per hour for 5 hours, with a fuel efficiency
    of 1 gallon per 40 miles and starting with a full 12-gallon tank, uses 5/12 of its fuel. -/
theorem car_fuel_usage (speed : ℝ) (time : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) :
  speed = 40 →
  time = 5 →
  fuel_efficiency = 40 →
  tank_capacity = 12 →
  (speed * time / fuel_efficiency) / tank_capacity = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_usage_l1165_116554


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1165_116573

theorem intersection_of_lines (x y : ℚ) : 
  x = 155 / 67 ∧ y = 5 / 67 ↔ 
  11 * x - 5 * y = 40 ∧ 9 * x + 2 * y = 15 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1165_116573


namespace NUMINAMATH_CALUDE_base_6_conversion_l1165_116567

def base_6_to_decimal (a b c d e : ℕ) : ℕ := 
  a * (6^4) + b * (6^3) + c * (6^2) + d * (6^1) + e * (6^0)

theorem base_6_conversion (m : ℕ) : 
  base_6_to_decimal 3 m 5 0 2 = 4934 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_6_conversion_l1165_116567


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l1165_116545

theorem perfect_square_quadratic (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - a*x + 16 = (x - b)^2) → (a = 8 ∨ a = -8) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l1165_116545


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1165_116585

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and asymptote y = 2x,
    its eccentricity e is either √5 or √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → y = 2*x) →
  ∃ e : ℝ, (e = Real.sqrt 5 ∨ e = Real.sqrt 5 / 2) ∧
    ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 →
      e = Real.sqrt ((a^2 + b^2) / a^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1165_116585


namespace NUMINAMATH_CALUDE_inclination_angle_of_line_l1165_116509

/-- The inclination angle of a line with equation ax + by + c = 0 is the angle between the positive x-axis and the line. -/
def InclinationAngle (a b c : ℝ) : ℝ := sorry

/-- The line equation sqrt(3)x + y - 1 = 0 -/
def LineEquation (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 1 = 0

theorem inclination_angle_of_line :
  InclinationAngle (Real.sqrt 3) 1 (-1) = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_inclination_angle_of_line_l1165_116509


namespace NUMINAMATH_CALUDE_ned_trays_theorem_l1165_116557

/-- The number of trays Ned can carry at a time -/
def trays_per_trip : ℕ := 8

/-- The number of trips Ned made -/
def num_trips : ℕ := 4

/-- The number of trays Ned picked up from the second table -/
def trays_from_second_table : ℕ := 5

/-- The number of trays Ned picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * num_trips - trays_from_second_table

theorem ned_trays_theorem : trays_from_first_table = 27 := by
  sorry

end NUMINAMATH_CALUDE_ned_trays_theorem_l1165_116557


namespace NUMINAMATH_CALUDE_union_equality_iff_range_l1165_116597

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m + 1)*x + 2*m < 0}

theorem union_equality_iff_range (m : ℝ) : A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_iff_range_l1165_116597


namespace NUMINAMATH_CALUDE_shape_perimeter_l1165_116532

theorem shape_perimeter (total_area : ℝ) (num_squares : ℕ) (h1 : total_area = 196) (h2 : num_squares = 4) :
  let side_length := Real.sqrt (total_area / num_squares)
  let perimeter := (num_squares + 1) * side_length + 2 * num_squares * side_length
  perimeter = 91 := by
sorry

end NUMINAMATH_CALUDE_shape_perimeter_l1165_116532


namespace NUMINAMATH_CALUDE_bobs_speed_l1165_116503

theorem bobs_speed (initial_time : ℝ) (construction_time : ℝ) (construction_speed : ℝ) (total_distance : ℝ) :
  initial_time = 1.5 →
  construction_time = 2 →
  construction_speed = 45 →
  total_distance = 180 →
  ∃ (initial_speed : ℝ),
    initial_speed * initial_time + construction_speed * construction_time = total_distance ∧
    initial_speed = 60 :=
by sorry

end NUMINAMATH_CALUDE_bobs_speed_l1165_116503


namespace NUMINAMATH_CALUDE_count_divisible_integers_l1165_116566

theorem count_divisible_integers :
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (1764 : ℤ) ∣ (m^2 - 4)) ∧
    (∀ m : ℕ, m > 0 ∧ (1764 : ℤ) ∣ (m^2 - 4) → m ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l1165_116566


namespace NUMINAMATH_CALUDE_polynomial_coefficient_bound_l1165_116562

-- Define the polynomial function
def p (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem polynomial_coefficient_bound (a b c d : ℝ) :
  (∀ x : ℝ, |x| < 1 → |p a b c d x| ≤ 1) →
  |a| + |b| + |c| + |d| ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_bound_l1165_116562


namespace NUMINAMATH_CALUDE_morgan_change_l1165_116546

/-- Calculates the change received from a purchase given item costs and amount paid --/
def calculate_change (hamburger_cost onion_rings_cost smoothie_cost amount_paid : ℕ) : ℕ :=
  amount_paid - (hamburger_cost + onion_rings_cost + smoothie_cost)

/-- Theorem stating that Morgan receives $11 in change --/
theorem morgan_change : calculate_change 4 2 3 20 = 11 := by
  sorry

end NUMINAMATH_CALUDE_morgan_change_l1165_116546


namespace NUMINAMATH_CALUDE_prime_factor_congruence_l1165_116543

theorem prime_factor_congruence (p : ℕ) (h_prime : Prime p) :
  ∃ q : ℕ, Prime q ∧ q ∣ (p^p - 1) ∧ q ≡ 1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_prime_factor_congruence_l1165_116543


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l1165_116548

theorem middle_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 120)
  (h_prop : ∃ (x : ℚ), a = x ∧ b = (1/2) * x ∧ c = (1/4) * x)
  (h_sum : a + b + c = total) :
  b = 34 + 2/7 := by
  sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l1165_116548


namespace NUMINAMATH_CALUDE_prism_dimensions_l1165_116519

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Checks if the given dimensions satisfy the conditions of the problem -/
def satisfiesConditions (d : PrismDimensions) : Prop :=
  -- One edge is five times longer than another
  (d.length = 5 * d.width ∨ d.width = 5 * d.length ∨ d.length = 5 * d.height ∨
   d.height = 5 * d.length ∨ d.width = 5 * d.height ∨ d.height = 5 * d.width) ∧
  -- Increasing height by 2 increases volume by 90
  d.length * d.width * 2 = 90 ∧
  -- Changing height to half of (height + 2) makes volume three-fifths of original
  (d.height + 2) / 2 = 3 / 5 * d.height

/-- The theorem stating the only possible dimensions for the rectangular prism -/
theorem prism_dimensions :
  ∀ d : PrismDimensions,
    satisfiesConditions d →
    (d = ⟨0.9, 50, 10⟩ ∨ d = ⟨50, 0.9, 10⟩ ∨
     d = ⟨2, 22.5, 10⟩ ∨ d = ⟨22.5, 2, 10⟩ ∨
     d = ⟨3, 15, 10⟩ ∨ d = ⟨15, 3, 10⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_prism_dimensions_l1165_116519


namespace NUMINAMATH_CALUDE_root_product_sum_l1165_116540

theorem root_product_sum (p q r : ℂ) : 
  (6 * p^3 - 5 * p^2 + 20 * p - 10 = 0) →
  (6 * q^3 - 5 * q^2 + 20 * q - 10 = 0) →
  (6 * r^3 - 5 * r^2 + 20 * r - 10 = 0) →
  p * q + p * r + q * r = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l1165_116540


namespace NUMINAMATH_CALUDE_office_episodes_l1165_116572

theorem office_episodes (total_episodes : ℕ) (weeks : ℕ) (monday_episodes : ℕ) 
  (h1 : total_episodes = 201)
  (h2 : weeks = 67)
  (h3 : monday_episodes = 1) :
  ∃ wednesday_episodes : ℕ, 
    wednesday_episodes * weeks + monday_episodes * weeks = total_episodes ∧ 
    wednesday_episodes = 2 := by
  sorry

end NUMINAMATH_CALUDE_office_episodes_l1165_116572


namespace NUMINAMATH_CALUDE_farm_rent_calculation_l1165_116584

-- Define the constants
def rent_per_acre_per_month : ℝ := 60
def plot_length : ℝ := 360
def plot_width : ℝ := 1210
def square_feet_per_acre : ℝ := 43560

-- Define the theorem
theorem farm_rent_calculation :
  let plot_area : ℝ := plot_length * plot_width
  let acres : ℝ := plot_area / square_feet_per_acre
  let monthly_rent : ℝ := rent_per_acre_per_month * acres
  monthly_rent = 600 := by sorry

end NUMINAMATH_CALUDE_farm_rent_calculation_l1165_116584


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1165_116507

/-- Given two vectors a and b in ℝ², where a = (-5, 1) and b = (2, x),
    if a and b are perpendicular, then x = 10. -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (-5, 1)
  let b : ℝ × ℝ := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 10 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1165_116507


namespace NUMINAMATH_CALUDE_dot_product_of_vectors_l1165_116542

/-- Given two vectors a and b in ℝ², prove that their dot product is -12
    when a + b = (1, 3) and a - b = (3, 7). -/
theorem dot_product_of_vectors (a b : ℝ × ℝ) 
    (h1 : a + b = (1, 3)) 
    (h2 : a - b = (3, 7)) : 
  a.1 * b.1 + a.2 * b.2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_vectors_l1165_116542


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1165_116574

/-- Given two hyperbolas with equations (x²/9) - (y²/16) = 1 and (y²/25) - (x²/M) = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) →
  (∀ x y : ℝ, y = 4/3 * x ↔ y = 5/Real.sqrt M * x) →
  M = 225/16 := by
sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1165_116574


namespace NUMINAMATH_CALUDE_pencil_count_l1165_116565

theorem pencil_count (initial_pencils added_pencils : ℕ) : 
  initial_pencils = 2 → added_pencils = 3 → initial_pencils + added_pencils = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1165_116565


namespace NUMINAMATH_CALUDE_roots_are_eccentricities_l1165_116517

def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 4 * x + 1 = 0

def is_ellipse_eccentricity (e : ℝ) : Prop := 0 < e ∧ e < 1

def is_parabola_eccentricity (e : ℝ) : Prop := e = 1

theorem roots_are_eccentricities :
  ∃ (e₁ e₂ : ℝ),
    quadratic_equation e₁ ∧
    quadratic_equation e₂ ∧
    e₁ ≠ e₂ ∧
    ((is_ellipse_eccentricity e₁ ∧ is_parabola_eccentricity e₂) ∨
     (is_ellipse_eccentricity e₂ ∧ is_parabola_eccentricity e₁)) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_eccentricities_l1165_116517


namespace NUMINAMATH_CALUDE_constant_product_of_distances_l1165_116528

/-- Hyperbola type representing x^2 - y^2/4 = 1 -/
structure Hyperbola where
  x : ℝ
  y : ℝ
  eq : x^2 - y^2/4 = 1

/-- Line type representing a line passing through a point on the hyperbola -/
structure Line (h : Hyperbola) where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  passes_through : m * h.x + b = h.y

/-- Intersection point of a line with an asymptote -/
structure AsymptoteIntersection (h : Hyperbola) (l : Line h) where
  x : ℝ
  y : ℝ
  on_asymptote : y = 2*x ∨ y = -2*x
  on_line : y = l.m * x + l.b

/-- Theorem: Product of distances from origin to asymptote intersections is constant -/
theorem constant_product_of_distances (h : Hyperbola) (l : Line h) 
  (a b : AsymptoteIntersection h l) 
  (midpoint : h.x = (a.x + b.x)/2 ∧ h.y = (a.y + b.y)/2) :
  (a.x^2 + a.y^2) * (b.x^2 + b.y^2) = 25 := by sorry

end NUMINAMATH_CALUDE_constant_product_of_distances_l1165_116528


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l1165_116569

/-- Represents the dimensions of the triangular brownie pan -/
structure PanDimensions where
  base : ℕ
  height : ℕ

/-- Represents the dimensions of a single brownie piece -/
structure PieceDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of full brownie pieces that can be obtained from a triangular pan -/
def maxBrowniePieces (pan : PanDimensions) (piece : PieceDimensions) : ℕ :=
  (pan.base / piece.width) * (pan.height / piece.height)

/-- Theorem stating the maximum number of brownie pieces for the given dimensions -/
theorem brownie_pieces_count :
  let pan := PanDimensions.mk 30 24
  let piece := PieceDimensions.mk 3 4
  maxBrowniePieces pan piece = 60 := by
sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l1165_116569


namespace NUMINAMATH_CALUDE_sqrt_88200_simplification_l1165_116521

theorem sqrt_88200_simplification : Real.sqrt 88200 = 210 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_88200_simplification_l1165_116521


namespace NUMINAMATH_CALUDE_prime_power_sum_l1165_116538

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 840 →
  2*w + 3*x + 5*y + 7*z = 21 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1165_116538


namespace NUMINAMATH_CALUDE_field_division_proof_l1165_116547

theorem field_division_proof (total_area smaller_area larger_area certain_value : ℝ) : 
  total_area = 900 ∧ 
  smaller_area = 405 ∧ 
  larger_area = total_area - smaller_area ∧ 
  larger_area - smaller_area = (1 / 5) * certain_value →
  certain_value = 450 := by
sorry

end NUMINAMATH_CALUDE_field_division_proof_l1165_116547


namespace NUMINAMATH_CALUDE_y_intercept_of_line_y_intercept_specific_line_l1165_116583

/-- The y-intercept of a line with equation ax + by + c = 0 is -c/b when b ≠ 0 -/
theorem y_intercept_of_line (a b c : ℝ) (hb : b ≠ 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let y_intercept := {y : ℝ | (0, y) ∈ line}
  y_intercept = {-c/b} :=
by sorry

/-- The y-intercept of the line x + 2y + 1 = 0 is -1/2 -/
theorem y_intercept_specific_line :
  let line := {p : ℝ × ℝ | p.1 + 2 * p.2 + 1 = 0}
  let y_intercept := {y : ℝ | (0, y) ∈ line}
  y_intercept = {-1/2} :=
by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_y_intercept_specific_line_l1165_116583
