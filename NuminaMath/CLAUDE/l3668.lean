import Mathlib

namespace NUMINAMATH_CALUDE_bead_necklaces_sold_l3668_366860

/-- Proves that the number of bead necklaces sold is 4, given the conditions of the problem -/
theorem bead_necklaces_sold (gem_necklaces : ℕ) (price_per_necklace : ℕ) (total_earnings : ℕ) 
  (h1 : gem_necklaces = 3)
  (h2 : price_per_necklace = 3)
  (h3 : total_earnings = 21) :
  total_earnings - gem_necklaces * price_per_necklace = 4 * price_per_necklace :=
by sorry

end NUMINAMATH_CALUDE_bead_necklaces_sold_l3668_366860


namespace NUMINAMATH_CALUDE_largest_quantity_l3668_366805

theorem largest_quantity (a b c d e : ℝ) 
  (h : a - 1 = b + 2 ∧ a - 1 = c - 3 ∧ a - 1 = d + 4 ∧ a - 1 = e - 6) : 
  e = max a (max b (max c d)) :=
sorry

end NUMINAMATH_CALUDE_largest_quantity_l3668_366805


namespace NUMINAMATH_CALUDE_weight_of_b_l3668_366876

/-- Given three weights a, b, and c, prove that b = 31 under certain conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →  -- average of a, b, and c is 45
  (a + b) / 2 = 40 →      -- average of a and b is 40
  (b + c) / 2 = 43 →      -- average of b and c is 43
  b = 31 := by
    sorry


end NUMINAMATH_CALUDE_weight_of_b_l3668_366876


namespace NUMINAMATH_CALUDE_common_solution_conditions_l3668_366875

theorem common_solution_conditions (x y : ℝ) : 
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_conditions_l3668_366875


namespace NUMINAMATH_CALUDE_eric_pencils_l3668_366890

/-- The number of boxes of pencils Eric has -/
def num_boxes : ℕ := 12

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 17

/-- The total number of pencils Eric has -/
def total_pencils : ℕ := num_boxes * pencils_per_box

theorem eric_pencils : total_pencils = 204 := by
  sorry

end NUMINAMATH_CALUDE_eric_pencils_l3668_366890


namespace NUMINAMATH_CALUDE_april_rose_price_l3668_366806

/-- Calculates the price per rose given the initial number of roses, remaining roses, and total earnings -/
def price_per_rose (initial_roses : ℕ) (remaining_roses : ℕ) (total_earnings : ℕ) : ℚ :=
  (total_earnings : ℚ) / ((initial_roses - remaining_roses) : ℚ)

theorem april_rose_price : price_per_rose 13 4 36 = 4 := by
  sorry

end NUMINAMATH_CALUDE_april_rose_price_l3668_366806


namespace NUMINAMATH_CALUDE_fraction_simplification_l3668_366848

theorem fraction_simplification : (20 - 20) / (20 + 20) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3668_366848


namespace NUMINAMATH_CALUDE_complex_fourth_power_l3668_366837

theorem complex_fourth_power (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l3668_366837


namespace NUMINAMATH_CALUDE_michaels_pets_l3668_366845

theorem michaels_pets (total_pets : ℕ) (dog_percentage : ℚ) (cat_percentage : ℚ) :
  total_pets = 36 →
  dog_percentage = 25 / 100 →
  cat_percentage = 50 / 100 →
  ↑(total_pets : ℕ) * (1 - dog_percentage - cat_percentage) = 9 := by
  sorry

end NUMINAMATH_CALUDE_michaels_pets_l3668_366845


namespace NUMINAMATH_CALUDE_product_nine_consecutive_divisible_by_ten_l3668_366899

theorem product_nine_consecutive_divisible_by_ten (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7) * (n + 8)) = 10 * k :=
by sorry

end NUMINAMATH_CALUDE_product_nine_consecutive_divisible_by_ten_l3668_366899


namespace NUMINAMATH_CALUDE_power_value_l3668_366824

theorem power_value (m n : ℤ) (x : ℝ) (h1 : x^m = 3) (h2 : x = 2) : x^(2*m+n) = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_value_l3668_366824


namespace NUMINAMATH_CALUDE_one_sport_count_l3668_366895

/-- The number of members who play only one sport (badminton, tennis, or basketball) -/
def members_one_sport (total members badminton tennis basketball badminton_tennis badminton_basketball tennis_basketball all_three none : ℕ) : ℕ :=
  let badminton_only := badminton - badminton_tennis - badminton_basketball + all_three
  let tennis_only := tennis - badminton_tennis - tennis_basketball + all_three
  let basketball_only := basketball - badminton_basketball - tennis_basketball + all_three
  badminton_only + tennis_only + basketball_only

theorem one_sport_count :
  members_one_sport 150 65 80 60 20 15 25 10 12 = 115 := by
  sorry

end NUMINAMATH_CALUDE_one_sport_count_l3668_366895


namespace NUMINAMATH_CALUDE_paper_pieces_difference_paper_pieces_problem_l3668_366856

theorem paper_pieces_difference : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun initial_squares initial_corners final_pieces final_corners corner_difference =>
    initial_squares * 4 = initial_corners →
    ∃ (triangles pentagonals : ℕ),
      triangles + pentagonals = final_pieces ∧
      3 * triangles + 5 * pentagonals = final_corners ∧
      triangles - pentagonals = corner_difference

theorem paper_pieces_problem :
  paper_pieces_difference 25 100 50 170 30 := by
  sorry

end NUMINAMATH_CALUDE_paper_pieces_difference_paper_pieces_problem_l3668_366856


namespace NUMINAMATH_CALUDE_ivan_total_distance_l3668_366892

/-- Represents the distances Ivan ran on each day of the week -/
structure WeeklyRun where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The conditions of Ivan's running schedule -/
def validWeeklyRun (run : WeeklyRun) : Prop :=
  run.tuesday = 2 * run.monday ∧
  run.wednesday = run.tuesday / 2 ∧
  run.thursday = run.wednesday / 2 ∧
  run.friday = 2 * run.thursday ∧
  min run.monday (min run.tuesday (min run.wednesday (min run.thursday run.friday))) = 5

/-- The theorem stating that the total distance Ivan ran is 55 km -/
theorem ivan_total_distance (run : WeeklyRun) (h : validWeeklyRun run) :
  run.monday + run.tuesday + run.wednesday + run.thursday + run.friday = 55 := by
  sorry


end NUMINAMATH_CALUDE_ivan_total_distance_l3668_366892


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3668_366802

theorem min_value_theorem (a : ℝ) (h : a > 1) : (4 / (a - 1)) + a ≥ 6 := by
  sorry

theorem min_value_achieved (ε : ℝ) (h : ε > 0) : 
  ∃ a : ℝ, a > 1 ∧ (4 / (a - 1)) + a < 6 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3668_366802


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3668_366811

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3668_366811


namespace NUMINAMATH_CALUDE_toothpicks_150th_stage_l3668_366828

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  6 + (n - 1) * 4

/-- Theorem: The number of toothpicks in the 150th stage is 602 -/
theorem toothpicks_150th_stage : toothpicks 150 = 602 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_150th_stage_l3668_366828


namespace NUMINAMATH_CALUDE_sin_cos_equality_relation_l3668_366836

open Real

theorem sin_cos_equality_relation :
  (∃ (α β : ℝ), (sin α = sin β ∧ cos α = cos β) ∧ α ≠ β) ∧
  (∀ (α β : ℝ), α = β → (sin α = sin β ∧ cos α = cos β)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equality_relation_l3668_366836


namespace NUMINAMATH_CALUDE_circle_max_min_value_l3668_366804

theorem circle_max_min_value (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 4 →
  ∃ (S_max S_min : ℝ),
    (∀ S, S = 3*x - y → S ≤ S_max ∧ S ≥ S_min) ∧
    S_max = 5 + 2 * Real.sqrt 10 ∧
    S_min = 5 - 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_min_value_l3668_366804


namespace NUMINAMATH_CALUDE_value_of_expression_l3668_366855

theorem value_of_expression (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - z = 0)
  (eq2 : x + 4 * y - 15 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 3*x*y - y*z) / (y^2 + z^2) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3668_366855


namespace NUMINAMATH_CALUDE_negative_abs_opposite_double_negative_l3668_366859

-- Define the property of being opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- State the theorem
theorem negative_abs_opposite_double_negative :
  are_opposite (-|(-3 : ℝ)|) (-(-3)) :=
sorry

end NUMINAMATH_CALUDE_negative_abs_opposite_double_negative_l3668_366859


namespace NUMINAMATH_CALUDE_length_AB_with_equal_quarter_circles_l3668_366809

/-- The length of AB given two circles with equal quarter-circle areas --/
theorem length_AB_with_equal_quarter_circles 
  (r : ℝ) 
  (h_r : r = 4)
  (π_approx : ℝ) 
  (h_π : π_approx = 3) : 
  let quarter_circle_area := (1/4) * π_approx * r^2
  let total_shaded_area := 2 * quarter_circle_area
  let AB := total_shaded_area / (2 * r)
  AB = 6 := by sorry

end NUMINAMATH_CALUDE_length_AB_with_equal_quarter_circles_l3668_366809


namespace NUMINAMATH_CALUDE_all_statements_correct_l3668_366830

-- Define chemical elements and their atomic masses
def H : ℝ := 1
def O : ℝ := 16
def S : ℝ := 32
def N : ℝ := 14
def C : ℝ := 12

-- Define molecules and their molar masses
def H2SO4_mass : ℝ := 2 * H + S + 4 * O
def NO_mass : ℝ := N + O
def NO2_mass : ℝ := N + 2 * O
def O2_mass : ℝ := 2 * O
def O3_mass : ℝ := 3 * O
def CO_mass : ℝ := C + O
def CO2_mass : ℝ := C + 2 * O

-- Define the number of atoms in 2 mol of NO and NO2
def NO_atoms : ℕ := 2
def NO2_atoms : ℕ := 3

-- Theorem stating all given statements are correct
theorem all_statements_correct :
  (H2SO4_mass = 98) ∧
  (2 * NO_atoms ≠ 2 * NO2_atoms) ∧
  (∀ m : ℝ, m > 0 → m / O2_mass * 2 = m / O3_mass * 3) ∧
  (∀ n : ℝ, n > 0 → n * (CO_mass / C) = n * (CO2_mass / C)) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_correct_l3668_366830


namespace NUMINAMATH_CALUDE_symmetry_axis_l3668_366841

-- Define a function f with the given property
def f (x : ℝ) : ℝ := sorry

-- State the condition that f(x) = f(3 - x) for all x
axiom f_symmetry (x : ℝ) : f x = f (3 - x)

-- Define the concept of an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem stating that x = 1.5 is an axis of symmetry for f
theorem symmetry_axis : is_axis_of_symmetry 1.5 f := by sorry

end NUMINAMATH_CALUDE_symmetry_axis_l3668_366841


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_396_l3668_366835

def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_distinct_prime_factors_396 :
  sum_of_distinct_prime_factors 396 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_396_l3668_366835


namespace NUMINAMATH_CALUDE_solve_equation_l3668_366854

theorem solve_equation (x : ℝ) (h : Real.sqrt (3 / x + 5) = 2) : x = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3668_366854


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l3668_366850

theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ c₁ c₂ k : ℝ) 
  (h1 : a₁ ≠ 0) (h2 : a₂ ≠ 0) (h3 : c₁ ≠ 0) (h4 : c₂ ≠ 0)
  (h5 : a₁ * b₁ * c₁ = k) (h6 : a₂ * b₂ * c₂ = k)
  (h7 : a₁ / a₂ = 3 / 4) (h8 : b₁ = 2 * b₂) : 
  c₁ / c₂ = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l3668_366850


namespace NUMINAMATH_CALUDE_min_lines_inequality_l3668_366858

/-- Represents the minimum number of lines required to compute a function using only disjunctions and conjunctions -/
def M (n : ℕ) : ℕ := sorry

/-- The theorem states that for n ≥ 4, the minimum number of lines to compute f_n is at least 3 more than the minimum number of lines to compute f_(n-2) -/
theorem min_lines_inequality (n : ℕ) (h : n ≥ 4) : M n ≥ M (n - 2) + 3 := by
  sorry

end NUMINAMATH_CALUDE_min_lines_inequality_l3668_366858


namespace NUMINAMATH_CALUDE_initial_players_count_l3668_366863

/-- Represents a round-robin chess tournament. -/
structure ChessTournament where
  initial_players : ℕ
  matches_played : ℕ
  dropped_players : ℕ
  matches_per_dropped : ℕ

/-- Calculates the total number of matches in a round-robin tournament. -/
def total_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating the initial number of players in the tournament. -/
theorem initial_players_count (t : ChessTournament) 
  (h1 : t.matches_played = 84)
  (h2 : t.dropped_players = 2)
  (h3 : t.matches_per_dropped = 3) :
  t.initial_players = 15 := by
  sorry

/-- The specific tournament instance described in the problem. -/
def problem_tournament : ChessTournament := {
  initial_players := 15,  -- This is what we're proving
  matches_played := 84,
  dropped_players := 2,
  matches_per_dropped := 3
}

end NUMINAMATH_CALUDE_initial_players_count_l3668_366863


namespace NUMINAMATH_CALUDE_exists_close_vertices_l3668_366869

/-- A regular polygon with 2n+1 sides inscribed in a unit circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n+1) → ℝ × ℝ
  is_regular : ∀ i : Fin (2*n+1), norm (vertices i) = 1

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := norm (p.1 - q.1, p.2 - q.2)

/-- The theorem statement -/
theorem exists_close_vertices {n : ℕ} (polygon : RegularPolygon n) :
  ∃ (α : ℝ), α > 0 ∧
  ∀ (p : ℝ × ℝ), norm p < 1 →
  ∃ (i j : Fin (2*n+1)), i ≠ j ∧
    |distance p (polygon.vertices i) - distance p (polygon.vertices j)| < 1/n - α/n^3 :=
sorry

end NUMINAMATH_CALUDE_exists_close_vertices_l3668_366869


namespace NUMINAMATH_CALUDE_coronavirus_case_ratio_l3668_366801

/-- Proves that the ratio of new coronavirus cases in the second week to the first week is 1/4 -/
theorem coronavirus_case_ratio :
  let first_week : ℕ := 5000
  let third_week (second_week : ℕ) : ℕ := second_week + 2000
  let total_cases : ℕ := 9500
  ∀ second_week : ℕ,
    first_week + second_week + third_week second_week = total_cases →
    (second_week : ℚ) / first_week = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_coronavirus_case_ratio_l3668_366801


namespace NUMINAMATH_CALUDE_turtle_reaches_waterhole_in_28_minutes_l3668_366898

/-- Represents the scenario with two lion cubs and a turtle moving towards a watering hole -/
structure WaterholeProblem where
  /-- Distance of the first lion cub from the watering hole in minutes -/
  lion1_distance : ℝ
  /-- Speed multiplier of the second lion cub compared to the first -/
  lion2_speed_multiplier : ℝ
  /-- Distance of the turtle from the watering hole in minutes -/
  turtle_distance : ℝ

/-- Calculates the time it takes for the turtle to reach the watering hole after meeting the lion cubs -/
def timeToWaterhole (problem : WaterholeProblem) : ℝ :=
  sorry

/-- Theorem stating that given the specific problem conditions, the turtle reaches the watering hole 28 minutes after meeting the lion cubs -/
theorem turtle_reaches_waterhole_in_28_minutes :
  let problem : WaterholeProblem :=
    { lion1_distance := 5
      lion2_speed_multiplier := 1.5
      turtle_distance := 30 }
  timeToWaterhole problem = 28 :=
sorry

end NUMINAMATH_CALUDE_turtle_reaches_waterhole_in_28_minutes_l3668_366898


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3668_366847

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ∧ x = -48 / 23 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3668_366847


namespace NUMINAMATH_CALUDE_natalia_clip_sales_l3668_366865

/-- Natalia's clip sales problem -/
theorem natalia_clip_sales 
  (x : ℝ) -- number of clips sold to each friend in April
  (y : ℝ) -- number of clips sold in May
  (z : ℝ) -- total earnings in dollars
  (h1 : y = x / 2) -- y is half of x
  : (48 * x + y = 97 * x / 2) ∧ (z / (48 * x + y) = 2 * z / (97 * x)) := by
  sorry

end NUMINAMATH_CALUDE_natalia_clip_sales_l3668_366865


namespace NUMINAMATH_CALUDE_three_digit_divisibility_l3668_366808

theorem three_digit_divisibility (a b c : ℕ) (h : ∃ k : ℕ, 100 * a + 10 * b + c = 27 * k ∨ 100 * a + 10 * b + c = 37 * k) :
  ∃ m : ℕ, 100 * b + 10 * c + a = 27 * m ∨ 100 * b + 10 * c + a = 37 * m := by
sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_l3668_366808


namespace NUMINAMATH_CALUDE_compare_expressions_l3668_366883

theorem compare_expressions (x : ℝ) (h : x > 1) : x^3 + 6*x > x^2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l3668_366883


namespace NUMINAMATH_CALUDE_solve_equation_l3668_366871

/-- Given an equation 19(x + y) + z = 19(-x + y) - 21 where x = 1, prove that z = -59 -/
theorem solve_equation (y : ℝ) : 
  ∃ z : ℝ, 19 * (1 + y) + z = 19 * (-1 + y) - 21 ∧ z = -59 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3668_366871


namespace NUMINAMATH_CALUDE_word_transformations_l3668_366822

-- Define the alphabet
inductive Letter : Type
| x : Letter
| y : Letter
| t : Letter

-- Define a word as a list of letters
def Word := List Letter

-- Define the transformation rules
inductive Transform : Word → Word → Prop
| xy_yyx : Transform (Letter.x::Letter.y::w) (Letter.y::Letter.y::Letter.x::w)
| xt_ttx : Transform (Letter.x::Letter.t::w) (Letter.t::Letter.t::Letter.x::w)
| yt_ty  : Transform (Letter.y::Letter.t::w) (Letter.t::Letter.y::w)
| refl   : ∀ w, Transform w w
| symm   : ∀ v w, Transform v w → Transform w v
| trans  : ∀ u v w, Transform u v → Transform v w → Transform u w

-- Define the theorem
theorem word_transformations :
  (¬ ∃ (w : Word), Transform [Letter.x, Letter.y] [Letter.x, Letter.t]) ∧
  (¬ ∃ (w : Word), Transform [Letter.x, Letter.y, Letter.t, Letter.x] [Letter.t, Letter.x, Letter.y, Letter.t]) ∧
  (∃ (w : Word), Transform [Letter.x, Letter.t, Letter.x, Letter.y, Letter.y] [Letter.t, Letter.t, Letter.x, Letter.y, Letter.y, Letter.y, Letter.y, Letter.x])
  := by sorry

end NUMINAMATH_CALUDE_word_transformations_l3668_366822


namespace NUMINAMATH_CALUDE_ceiling_plus_x_eq_two_x_l3668_366844

theorem ceiling_plus_x_eq_two_x (x : ℝ) (h : ⌈x⌉ + ⌊x⌋ = 2 * x) : ⌈x⌉ + x = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_ceiling_plus_x_eq_two_x_l3668_366844


namespace NUMINAMATH_CALUDE_parallel_vectors_acute_angle_l3668_366880

/-- Given two vectors a and b that are parallel and α is an acute angle, prove that α = 45° -/
theorem parallel_vectors_acute_angle (α : Real) 
  (h_acute : 0 < α ∧ α < Real.pi / 2)
  (a : Fin 2 → Real)
  (b : Fin 2 → Real)
  (h_a : a = ![3/2, 1 + Real.sin α])
  (h_b : b = ![1 - Real.cos α, 1/3])
  (h_parallel : ∃ (k : Real), a = k • b) :
  α = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_acute_angle_l3668_366880


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l3668_366886

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l3668_366886


namespace NUMINAMATH_CALUDE_gray_area_calculation_l3668_366857

/-- Given two rectangles with dimensions 8x10 and 12x9, and an overlapping area of 37,
    the non-overlapping area in the second rectangle (gray part) is 65. -/
theorem gray_area_calculation (rect1_width rect1_height rect2_width rect2_height black_area : ℕ)
  (h1 : rect1_width = 8)
  (h2 : rect1_height = 10)
  (h3 : rect2_width = 12)
  (h4 : rect2_height = 9)
  (h5 : black_area = 37) :
  rect2_width * rect2_height - (rect1_width * rect1_height - black_area) = 65 := by
sorry

end NUMINAMATH_CALUDE_gray_area_calculation_l3668_366857


namespace NUMINAMATH_CALUDE_additional_investment_rate_problem_l3668_366882

/-- Calculates the rate of additional investment needed to achieve a target total rate --/
def additional_investment_rate (initial_investment : ℚ) (initial_rate : ℚ) 
  (additional_investment : ℚ) (target_total_rate : ℚ) : ℚ :=
  let total_investment := initial_investment + additional_investment
  let initial_interest := initial_investment * initial_rate
  let total_desired_interest := total_investment * target_total_rate
  let additional_interest_needed := total_desired_interest - initial_interest
  additional_interest_needed / additional_investment

theorem additional_investment_rate_problem 
  (initial_investment : ℚ) 
  (initial_rate : ℚ) 
  (additional_investment : ℚ) 
  (target_total_rate : ℚ) :
  initial_investment = 8000 →
  initial_rate = 5 / 100 →
  additional_investment = 4000 →
  target_total_rate = 6 / 100 →
  additional_investment_rate initial_investment initial_rate additional_investment target_total_rate = 8 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_additional_investment_rate_problem_l3668_366882


namespace NUMINAMATH_CALUDE_balance_proof_l3668_366813

def initial_balance : ℕ := 27004
def transferred_amount : ℕ := 69
def remaining_balance : ℕ := 26935

theorem balance_proof : initial_balance = transferred_amount + remaining_balance := by
  sorry

end NUMINAMATH_CALUDE_balance_proof_l3668_366813


namespace NUMINAMATH_CALUDE_set_A_representation_l3668_366894

def A : Set (ℤ × ℤ) := {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_representation : A = {(-1, 0), (0, -1), (1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_set_A_representation_l3668_366894


namespace NUMINAMATH_CALUDE_circles_intersection_triangle_similarity_l3668_366879

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the centers of the circles
variable (O₁ O₂ : Point)

-- Define the circles
variable (Γ₁ Γ₂ : Circle)

-- Define the intersection points
variable (X Y : Point)

-- Define point A on Γ₁
variable (A : Point)

-- Define point B as the intersection of AY and Γ₂
variable (B : Point)

-- Define the property of being on a circle
variable (on_circle : Point → Circle → Prop)

-- Define the property of two circles intersecting
variable (intersect : Circle → Circle → Point → Point → Prop)

-- Define the property of a point being on a line
variable (on_line : Point → Point → Point → Prop)

-- Define the property of triangle similarity
variable (similar_triangles : Point → Point → Point → Point → Point → Point → Prop)

-- State the theorem
theorem circles_intersection_triangle_similarity
  (h1 : on_circle O₁ Γ₁)
  (h2 : on_circle O₂ Γ₂)
  (h3 : intersect Γ₁ Γ₂ X Y)
  (h4 : on_circle A Γ₁)
  (h5 : A ≠ X)
  (h6 : A ≠ Y)
  (h7 : on_line A Y B)
  (h8 : on_circle B Γ₂) :
  similar_triangles X O₁ O₂ X A B :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_triangle_similarity_l3668_366879


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3668_366816

-- Define the second quadrant
def second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 2 * Real.pi + Real.pi / 2 < α ∧ α < k * 2 * Real.pi + Real.pi

-- Define the first quadrant
def first_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 2 * Real.pi < α ∧ α < n * 2 * Real.pi + Real.pi / 2

-- Define the third quadrant
def third_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 2 * Real.pi + Real.pi < α ∧ α < n * 2 * Real.pi + 3 * Real.pi / 2

-- Theorem statement
theorem half_angle_quadrant (α : Real) :
  second_quadrant α → first_quadrant (α / 2) ∨ third_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3668_366816


namespace NUMINAMATH_CALUDE_simplify_expression_l3668_366838

theorem simplify_expression (x : ℝ) : 
  (3 * x + 6 - 5 * x) / 3 = -2/3 * x + 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3668_366838


namespace NUMINAMATH_CALUDE_lisa_spoon_count_l3668_366818

/-- The number of spoons Lisa has after replacing her old cutlery -/
def total_spoons (num_children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) 
  (large_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

/-- Proof that Lisa has 39 spoons in total -/
theorem lisa_spoon_count : 
  total_spoons 4 3 2 10 15 = 39 := by
  sorry

end NUMINAMATH_CALUDE_lisa_spoon_count_l3668_366818


namespace NUMINAMATH_CALUDE_multiplication_problem_l3668_366819

theorem multiplication_problem : 7 * (1 / 11) * 33 = 21 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l3668_366819


namespace NUMINAMATH_CALUDE_lower_limit_of_a_l3668_366877

theorem lower_limit_of_a (a b : ℤ) (h1 : a < 15) (h2 : b > 6) (h3 : b < 21)
  (h4 : (a : ℝ) / 7 - (a : ℝ) / 20 = 1.55) : a ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_lower_limit_of_a_l3668_366877


namespace NUMINAMATH_CALUDE_four_digit_addition_l3668_366872

/-- Given four different natural numbers A, B, C, and D that satisfy the equation
    4A5B + C2D7 = 7070, prove that C = 2. -/
theorem four_digit_addition (A B C D : ℕ) 
  (diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (eq : 4000 * A + 50 * B + 1000 * C + 200 * D + 7 = 7070) : 
  C = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_addition_l3668_366872


namespace NUMINAMATH_CALUDE_inequality_holds_l3668_366833

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3668_366833


namespace NUMINAMATH_CALUDE_saras_money_theorem_l3668_366870

/-- Calculates Sara's remaining money after all expenses --/
def saras_remaining_money (hours_per_week : ℕ) (weeks : ℕ) (hourly_rate : ℚ) 
  (tax_rate : ℚ) (insurance_fee : ℚ) (misc_fee : ℚ) (tire_cost : ℚ) : ℚ :=
  let gross_pay := hours_per_week * weeks * hourly_rate
  let taxes := tax_rate * gross_pay
  let net_pay := gross_pay - taxes - insurance_fee - misc_fee - tire_cost
  net_pay

/-- Theorem stating that Sara's remaining money is $292 --/
theorem saras_money_theorem : 
  saras_remaining_money 40 2 (11.5) (0.15) 60 20 410 = 292 := by
  sorry

end NUMINAMATH_CALUDE_saras_money_theorem_l3668_366870


namespace NUMINAMATH_CALUDE_commission_breakpoint_l3668_366893

/-- Proves that for a sale of $800, if the commission is 20% of the first $X plus 25% of the remainder, 
    and the total commission is 21.875% of the sale, then X = $500. -/
theorem commission_breakpoint (X : ℝ) : 
  let total_sale := 800
  let commission_rate_1 := 0.20
  let commission_rate_2 := 0.25
  let total_commission_rate := 0.21875
  commission_rate_1 * X + commission_rate_2 * (total_sale - X) = total_commission_rate * total_sale →
  X = 500 :=
by sorry

end NUMINAMATH_CALUDE_commission_breakpoint_l3668_366893


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_implies_a_equals_one_l3668_366852

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) / (x + a)

theorem f_derivative_at_zero_implies_a_equals_one (a : ℝ) :
  (deriv (f a)) 0 = 1 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_implies_a_equals_one_l3668_366852


namespace NUMINAMATH_CALUDE_complex_cube_real_l3668_366840

theorem complex_cube_real (a b : ℝ) (hb : b ≠ 0) 
  (h : ∃ (r : ℝ), (Complex.mk a b)^3 = r) : b^2 = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_real_l3668_366840


namespace NUMINAMATH_CALUDE_odd_function_sum_l3668_366807

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_property : ∀ x, f (2 + x) + f (2 - x) = 0)
  (h_f1 : f 1 = 9) :
  f 2010 + f 2011 + f 2012 = -9 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l3668_366807


namespace NUMINAMATH_CALUDE_gumball_difference_l3668_366887

theorem gumball_difference (carolyn lew amanda tom : ℕ) (x : ℕ) :
  carolyn = 17 →
  lew = 12 →
  amanda = 24 →
  tom = 8 →
  14 ≤ (carolyn + lew + amanda + tom + x) / 7 →
  (carolyn + lew + amanda + tom + x) / 7 ≤ 32 →
  ∃ (x_min x_max : ℕ), x_min ≤ x ∧ x ≤ x_max ∧ x_max - x_min = 126 :=
by sorry

end NUMINAMATH_CALUDE_gumball_difference_l3668_366887


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l3668_366814

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 6*x - 5 < 3 - 2*x :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l3668_366814


namespace NUMINAMATH_CALUDE_aunts_gift_l3668_366817

theorem aunts_gift (grandfather_gift : ℕ) (bank_deposit : ℕ) (total_gift : ℕ) :
  grandfather_gift = 150 →
  bank_deposit = 45 →
  bank_deposit * 5 = total_gift →
  total_gift - grandfather_gift = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_aunts_gift_l3668_366817


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l3668_366868

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) :
  total_distance = 560 →
  total_time = 25 →
  first_half_speed = 21 →
  ∃ second_half_speed : ℝ,
    second_half_speed = 24 ∧
    (total_distance / 2) / first_half_speed + (total_distance / 2) / second_half_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l3668_366868


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l3668_366821

theorem quadratic_root_sum (a b c : ℤ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = a ∨ x = b) →
  a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l3668_366821


namespace NUMINAMATH_CALUDE_train_trip_probability_l3668_366815

theorem train_trip_probability : ∀ (p₁ p₂ p₃ p₄ : ℝ),
  p₁ = 0.3 →
  p₂ = 0.1 →
  p₃ = 0.4 →
  p₁ + p₂ + p₃ + p₄ = 1 →
  p₄ = 0.2 := by
sorry

end NUMINAMATH_CALUDE_train_trip_probability_l3668_366815


namespace NUMINAMATH_CALUDE_house_to_library_distance_l3668_366826

/-- Represents the distances between locations in miles -/
structure Distances where
  total : ℝ
  library_to_post_office : ℝ
  post_office_to_home : ℝ

/-- Calculates the distance from house to library -/
def distance_house_to_library (d : Distances) : ℝ :=
  d.total - d.library_to_post_office - d.post_office_to_home

/-- Theorem stating the distance from house to library is 0.3 miles -/
theorem house_to_library_distance (d : Distances) 
  (h1 : d.total = 0.8)
  (h2 : d.library_to_post_office = 0.1)
  (h3 : d.post_office_to_home = 0.4) : 
  distance_house_to_library d = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_house_to_library_distance_l3668_366826


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3668_366889

/-- Given a complex number z = (10-5ai)/(1-2i) where a is a real number,
    and the sum of its real and imaginary parts is 4,
    prove that its real part is negative and its imaginary part is positive. -/
theorem complex_number_in_second_quadrant (a : ℝ) :
  let z : ℂ := (10 - 5*a*Complex.I) / (1 - 2*Complex.I)
  (z.re + z.im = 4) →
  (z.re < 0 ∧ z.im > 0) :=
by sorry


end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3668_366889


namespace NUMINAMATH_CALUDE_range_of_a_l3668_366839

-- Define the set A
def A (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (a * x^2 + 2*(a-1)*x - 4)}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, A a = Set.Ici 0) ↔ Set.Ici 0 = {a : ℝ | 0 ≤ a} := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3668_366839


namespace NUMINAMATH_CALUDE_f_sin_cos_inequality_l3668_366846

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 2)

def f_on_interval (f : ℝ → ℝ) : Prop := ∀ x ∈ Set.Icc 3 4, f x = x - 2

theorem f_sin_cos_inequality 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_period : has_period_two f) 
  (h_interval : f_on_interval f) : 
  f (Real.sin 1) < f (Real.cos 1) := by
  sorry

end NUMINAMATH_CALUDE_f_sin_cos_inequality_l3668_366846


namespace NUMINAMATH_CALUDE_camera_profit_difference_l3668_366896

/-- Calculates the difference in profit between two camera sellers --/
theorem camera_profit_difference 
  (maddox_cameras : ℕ) (maddox_buy_price : ℚ) (maddox_sell_price : ℚ)
  (maddox_shipping : ℚ) (maddox_listing_fee : ℚ)
  (theo_cameras : ℕ) (theo_buy_price : ℚ) (theo_sell_price : ℚ)
  (theo_shipping : ℚ) (theo_listing_fee : ℚ)
  (h1 : maddox_cameras = 10)
  (h2 : maddox_buy_price = 35)
  (h3 : maddox_sell_price = 50)
  (h4 : maddox_shipping = 2)
  (h5 : maddox_listing_fee = 10)
  (h6 : theo_cameras = 15)
  (h7 : theo_buy_price = 30)
  (h8 : theo_sell_price = 40)
  (h9 : theo_shipping = 3)
  (h10 : theo_listing_fee = 15) :
  (maddox_cameras : ℚ) * maddox_sell_price - 
  (maddox_cameras : ℚ) * maddox_buy_price - 
  (maddox_cameras : ℚ) * maddox_shipping - 
  maddox_listing_fee -
  (theo_cameras : ℚ) * theo_sell_price + 
  (theo_cameras : ℚ) * theo_buy_price + 
  (theo_cameras : ℚ) * theo_shipping + 
  theo_listing_fee = 30 :=
by sorry

end NUMINAMATH_CALUDE_camera_profit_difference_l3668_366896


namespace NUMINAMATH_CALUDE_translation_left_proof_l3668_366832

def translate_left (x y : ℝ) (d : ℝ) : ℝ × ℝ :=
  (x - d, y)

theorem translation_left_proof :
  let A : ℝ × ℝ := (1, 2)
  let A₁ : ℝ × ℝ := translate_left A.1 A.2 1
  A₁ = (0, 2) := by sorry

end NUMINAMATH_CALUDE_translation_left_proof_l3668_366832


namespace NUMINAMATH_CALUDE_cupcake_cost_split_l3668_366884

theorem cupcake_cost_split (num_cupcakes : ℕ) (price_per_cupcake : ℚ) (num_people : ℕ) :
  num_cupcakes = 12 →
  price_per_cupcake = 3/2 →
  num_people = 2 →
  (num_cupcakes : ℚ) * price_per_cupcake / num_people = 9 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_cost_split_l3668_366884


namespace NUMINAMATH_CALUDE_remainder_problem_l3668_366897

theorem remainder_problem (n : ℤ) (h : n % 11 = 4) : (4 * n - 9) % 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3668_366897


namespace NUMINAMATH_CALUDE_sum_opposite_and_abs_l3668_366825

theorem sum_opposite_and_abs : -15 + |(-6)| = -9 := by
  sorry

end NUMINAMATH_CALUDE_sum_opposite_and_abs_l3668_366825


namespace NUMINAMATH_CALUDE_total_cost_l3668_366823

/-- Represents the price of an enchilada in dollars -/
def enchilada_price : ℚ := sorry

/-- Represents the price of a taco in dollars -/
def taco_price : ℚ := sorry

/-- Represents the price of a drink in dollars -/
def drink_price : ℚ := sorry

/-- The first price condition: one enchilada, two tacos, and a drink cost $3.20 -/
axiom price_condition1 : enchilada_price + 2 * taco_price + drink_price = 32/10

/-- The second price condition: two enchiladas, three tacos, and a drink cost $4.90 -/
axiom price_condition2 : 2 * enchilada_price + 3 * taco_price + drink_price = 49/10

/-- Theorem stating that the cost of four enchiladas, five tacos, and two drinks is $8.30 -/
theorem total_cost : 4 * enchilada_price + 5 * taco_price + 2 * drink_price = 83/10 := by sorry

end NUMINAMATH_CALUDE_total_cost_l3668_366823


namespace NUMINAMATH_CALUDE_omega_range_l3668_366831

open Real

theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∃! (r₁ r₂ r₃ : ℝ), 0 < r₁ ∧ r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ < π ∧
    (∀ x, sin (ω * x) - Real.sqrt 3 * cos (ω * x) = -1 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃)) →
  13/6 < ω ∧ ω ≤ 7/2 :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l3668_366831


namespace NUMINAMATH_CALUDE_distance_BC_l3668_366862

/-- Represents a point on the route --/
structure Point :=
  (position : ℝ)

/-- Represents the route with points A, B, and C --/
structure Route :=
  (A B C : Point)
  (speed : ℝ)
  (time : ℝ)
  (AC_distance : ℝ)

/-- The theorem statement --/
theorem distance_BC (route : Route) : 
  route.A.position = 0 ∧ 
  route.speed = 50 ∧ 
  route.time = 20 ∧ 
  route.AC_distance = 600 →
  route.C.position - route.B.position = 400 :=
sorry

end NUMINAMATH_CALUDE_distance_BC_l3668_366862


namespace NUMINAMATH_CALUDE_dad_steps_l3668_366878

/-- Represents the number of steps taken by each person --/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- The ratio of steps between Dad and Masha --/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- The ratio of steps between Masha and Yasha --/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- The total number of steps taken by Masha and Yasha --/
def masha_yasha_total (s : Steps) : Prop :=
  s.masha + s.yasha = 400

/-- The main theorem: Given the conditions, Dad took 90 steps --/
theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s) 
  (h2 : masha_yasha_ratio s) 
  (h3 : masha_yasha_total s) : 
  s.dad = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l3668_366878


namespace NUMINAMATH_CALUDE_extreme_value_condition_l3668_366873

/-- The function f(x) = ax + ln(x) has an extreme value -/
def has_extreme_value (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → a * x + Real.log x ≥ a * y + Real.log y) ∨
                   (∀ y : ℝ, y > 0 → a * x + Real.log x ≤ a * y + Real.log y)

/-- a ≤ 0 is a necessary but not sufficient condition for f(x) = ax + ln(x) to have an extreme value -/
theorem extreme_value_condition (a : ℝ) :
  has_extreme_value a → a ≤ 0 ∧ ∃ b : ℝ, b ≤ 0 ∧ ¬has_extreme_value b :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l3668_366873


namespace NUMINAMATH_CALUDE_place_values_in_9890_l3668_366829

theorem place_values_in_9890 : 
  ∃ (thousands hundreds tens : ℕ),
    9890 = thousands * 1000 + hundreds * 100 + tens * 10 + (9890 % 10) ∧
    thousands = 9 ∧
    hundreds = 8 ∧
    tens = 9 :=
by sorry

end NUMINAMATH_CALUDE_place_values_in_9890_l3668_366829


namespace NUMINAMATH_CALUDE_final_amount_correct_l3668_366843

/-- Represents the financial transactions in the boot-selling scenario -/
def boot_sale (original_price total_collected price_per_boot return_amount candy_cost actual_return : ℚ) : Prop :=
  -- Original intended price
  original_price = 25 ∧
  -- Total collected from selling two boots
  total_collected = 2 * price_per_boot ∧
  -- Price per boot
  price_per_boot = 12.5 ∧
  -- Amount to be returned per boot
  return_amount = 2.5 ∧
  -- Cost of candy Hans bought
  candy_cost = 3 ∧
  -- Actual amount returned to each customer
  actual_return = 1

/-- The theorem stating that the final amount Karl received is correct -/
theorem final_amount_correct 
  (original_price total_collected price_per_boot return_amount candy_cost actual_return : ℚ)
  (h : boot_sale original_price total_collected price_per_boot return_amount candy_cost actual_return) :
  total_collected - (2 * actual_return) = original_price - (2 * return_amount) :=
by sorry

end NUMINAMATH_CALUDE_final_amount_correct_l3668_366843


namespace NUMINAMATH_CALUDE_smallest_angle_triangle_range_l3668_366810

theorem smallest_angle_triangle_range (x : Real) : 
  (∀ y : Real, y = Real.sqrt 2 * Real.sin (x + π/4)) →
  (0 < x ∧ x ≤ π/3) →
  ∃ (a b : Real), a = 1 ∧ b = Real.sqrt 2 ∧ 
    (∀ y : Real, y = Real.sqrt 2 * Real.sin (x + π/4) → a < y ∧ y ≤ b) ∧
    (∀ z : Real, a < z ∧ z ≤ b → ∃ x₀ : Real, 0 < x₀ ∧ x₀ ≤ π/3 ∧ z = Real.sqrt 2 * Real.sin (x₀ + π/4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_triangle_range_l3668_366810


namespace NUMINAMATH_CALUDE_juggling_balls_count_l3668_366874

theorem juggling_balls_count (balls_per_juggler : ℕ) (number_of_jugglers : ℕ) (total_balls : ℕ) : 
  balls_per_juggler = 6 → 
  number_of_jugglers = 378 → 
  total_balls = balls_per_juggler * number_of_jugglers → 
  total_balls = 2268 := by
sorry

end NUMINAMATH_CALUDE_juggling_balls_count_l3668_366874


namespace NUMINAMATH_CALUDE_complex_symmetry_product_l3668_366881

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  (z₁.re = -z₂.re ∧ z₁.im = z₂.im) →  -- symmetry about imaginary axis
  z₁ = 1 + 2*I →                     -- given value of z₁
  z₁ * z₂ = -5 :=                    -- product equals -5
by
  sorry

end NUMINAMATH_CALUDE_complex_symmetry_product_l3668_366881


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3668_366812

theorem trigonometric_identity (x : Real) : 
  Real.sin x ^ 2 + Real.sin x * Real.cos (π / 6 + x) + Real.cos (π / 6 + x) ^ 2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3668_366812


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3668_366800

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem min_value_geometric_sequence (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 2) :
  ∃ (min_value : ℝ), 
    (∀ a₁ a₃, a 1 = a₁ ∧ a 3 = a₃ → a₁ + 2 * a₃ ≥ min_value) ∧
    (∃ a₁ a₃, a 1 = a₁ ∧ a 3 = a₃ ∧ a₁ + 2 * a₃ = min_value) ∧
    min_value = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3668_366800


namespace NUMINAMATH_CALUDE_charlie_area_is_72_l3668_366849

-- Define the total area to be painted
def total_area : ℝ := 360

-- Define the ratio of work done by each person
def allen_ratio : ℝ := 3
def ben_ratio : ℝ := 5
def charlie_ratio : ℝ := 2

-- Define the total ratio
def total_ratio : ℝ := allen_ratio + ben_ratio + charlie_ratio

-- Theorem to prove
theorem charlie_area_is_72 : 
  charlie_ratio / total_ratio * total_area = 72 := by
  sorry

end NUMINAMATH_CALUDE_charlie_area_is_72_l3668_366849


namespace NUMINAMATH_CALUDE_multiples_of_4_or_6_in_100_l3668_366827

theorem multiples_of_4_or_6_in_100 :
  let S := Finset.range 100
  (S.filter (fun n => n % 4 = 0 ∨ n % 6 = 0)).card = 33 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_4_or_6_in_100_l3668_366827


namespace NUMINAMATH_CALUDE_least_product_of_primes_over_50_l3668_366834

theorem least_product_of_primes_over_50 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧
    p > 50 ∧ q > 50 ∧
    p ≠ q ∧
    p * q = 3127 ∧
    ∀ r s : ℕ,
      r.Prime → s.Prime →
      r > 50 → s > 50 →
      r ≠ s →
      r * s ≥ 3127 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_primes_over_50_l3668_366834


namespace NUMINAMATH_CALUDE_highest_number_on_paper_l3668_366853

theorem highest_number_on_paper (n : ℕ) : 
  (1 : ℚ) / n = 0.010526315789473684 → n = 95 :=
by sorry

end NUMINAMATH_CALUDE_highest_number_on_paper_l3668_366853


namespace NUMINAMATH_CALUDE_tan_addition_special_case_l3668_366866

theorem tan_addition_special_case (x : Real) (h : Real.tan x = Real.sqrt 3) :
  Real.tan (x + π/3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_addition_special_case_l3668_366866


namespace NUMINAMATH_CALUDE_rectangle_to_square_area_ratio_l3668_366803

theorem rectangle_to_square_area_ratio :
  let large_square_side : ℝ := 50
  let grid_size : ℕ := 5
  let rectangle_rows : ℕ := 2
  let rectangle_cols : ℕ := 3
  let large_square_area : ℝ := large_square_side ^ 2
  let small_square_side : ℝ := large_square_side / grid_size
  let rectangle_area : ℝ := (rectangle_rows * small_square_side) * (rectangle_cols * small_square_side)
  rectangle_area / large_square_area = 6 / 25 := by
sorry

end NUMINAMATH_CALUDE_rectangle_to_square_area_ratio_l3668_366803


namespace NUMINAMATH_CALUDE_ray_walks_to_highschool_l3668_366891

/-- Represents the number of blocks Ray walks to the park -/
def blocks_to_park : ℕ := 4

/-- Represents the number of blocks Ray walks from the high school to home -/
def blocks_from_highschool_to_home : ℕ := 11

/-- Represents the number of times Ray walks his dog each day -/
def walks_per_day : ℕ := 3

/-- Represents the total number of blocks Ray's dog walks each day -/
def total_blocks_per_day : ℕ := 66

/-- Represents the number of blocks Ray walks to the high school -/
def blocks_to_highschool : ℕ := 7

theorem ray_walks_to_highschool :
  blocks_to_highschool = 7 ∧
  walks_per_day * (blocks_to_park + blocks_to_highschool + blocks_from_highschool_to_home) = total_blocks_per_day :=
by sorry

end NUMINAMATH_CALUDE_ray_walks_to_highschool_l3668_366891


namespace NUMINAMATH_CALUDE_inequality_solution_l3668_366820

def inequality (x : ℝ) : Prop :=
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 2

def solution_set (x : ℝ) : Prop :=
  (0 < x ∧ x ≤ 0.5) ∨ (x ≥ 6)

theorem inequality_solution : ∀ x : ℝ, inequality x ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3668_366820


namespace NUMINAMATH_CALUDE_incenter_orthocenter_collinearity_l3668_366864

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Triangle : Type :=
  (A B C : Point)

-- Define the necessary concepts
def isIncenter (I : Point) (t : Triangle) : Prop := sorry
def isOrthocenter (H : Point) (t : Triangle) : Prop := sorry
def isMidpoint (M : Point) (A B : Point) : Prop := sorry
def liesOn (P : Point) (A B : Point) : Prop := sorry
def intersectsAt (A B C D : Point) (K : Point) : Prop := sorry
def isCircumcenter (O : Point) (t : Triangle) : Prop := sorry
def areCollinear (A B C : Point) : Prop := sorry
def areaTriangle (A B C : Point) : ℝ := sorry

-- State the theorem
theorem incenter_orthocenter_collinearity 
  (t : Triangle) (I H B₁ C₁ B₂ C₂ K A₁ : Point) : 
  isIncenter I t → 
  isOrthocenter H t → 
  isMidpoint B₁ t.A t.C → 
  isMidpoint C₁ t.A t.B → 
  liesOn B₂ t.A t.B → 
  liesOn B₂ B₁ I → 
  B₂ ≠ t.B → 
  liesOn C₂ t.A C₁ → 
  liesOn C₂ C₁ I → 
  intersectsAt B₂ C₂ t.B t.C K → 
  isCircumcenter A₁ ⟨t.B, H, t.C⟩ → 
  (areCollinear t.A I A₁ ↔ areaTriangle t.B K B₂ = areaTriangle t.C K C₂) :=
sorry

end NUMINAMATH_CALUDE_incenter_orthocenter_collinearity_l3668_366864


namespace NUMINAMATH_CALUDE_rice_grains_difference_l3668_366842

def grains_on_square (k : ℕ) : ℕ := 2^k

def sum_of_first_n_squares (n : ℕ) : ℕ :=
  (List.range n).map grains_on_square |>.sum

theorem rice_grains_difference :
  grains_on_square 12 - sum_of_first_n_squares 10 = 2050 := by
  sorry

end NUMINAMATH_CALUDE_rice_grains_difference_l3668_366842


namespace NUMINAMATH_CALUDE_percentage_of_black_cats_l3668_366851

theorem percentage_of_black_cats 
  (total_cats : ℕ) 
  (white_cats : ℕ) 
  (grey_cats : ℕ) 
  (h1 : total_cats = 16) 
  (h2 : white_cats = 2) 
  (h3 : grey_cats = 10) :
  (((total_cats - white_cats - grey_cats) : ℚ) / total_cats) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_black_cats_l3668_366851


namespace NUMINAMATH_CALUDE_alyssa_ate_25_limes_l3668_366885

/-- The number of limes Mike picked -/
def mike_limes : ℝ := 32.0

/-- The number of limes left -/
def limes_left : ℝ := 7

/-- The number of limes Alyssa ate -/
def alyssa_limes : ℝ := mike_limes - limes_left

/-- Proof that Alyssa ate 25.0 limes -/
theorem alyssa_ate_25_limes : alyssa_limes = 25.0 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_ate_25_limes_l3668_366885


namespace NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l3668_366888

/-- The minimum distance from any point on the curve y = e^x to the line y = x - 1 is √2 -/
theorem min_distance_exp_curve_to_line : 
  ∀ (x₀ y₀ : ℝ), y₀ = Real.exp x₀ → 
  (∃ (d : ℝ), d = |y₀ - (x₀ - 1)| / Real.sqrt 2 ∧ 
   ∀ (x y : ℝ), y = Real.exp x → 
   d ≤ |y - (x - 1)| / Real.sqrt 2) → 
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  (∀ (x y : ℝ), y = Real.exp x → 
   d ≤ |y - (x - 1)| / Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l3668_366888


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3668_366867

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3668_366867


namespace NUMINAMATH_CALUDE_base5_123_to_base10_l3668_366861

/-- Converts a base-5 number represented as a list of digits to its base-10 equivalent -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Theorem: The base-10 representation of the base-5 number 123 is 38 -/
theorem base5_123_to_base10 :
  base5ToBase10 [3, 2, 1] = 38 := by
  sorry

#eval base5ToBase10 [3, 2, 1]

end NUMINAMATH_CALUDE_base5_123_to_base10_l3668_366861
