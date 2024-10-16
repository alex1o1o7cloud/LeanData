import Mathlib

namespace NUMINAMATH_CALUDE_age_group_problem_l1034_103406

theorem age_group_problem (n : ℕ) (A : ℝ) : 
  (n + 1) * (A + 7) = n * A + 39 →
  (n + 1) * (A - 1) = n * A + 15 →
  n = 3 := by sorry

end NUMINAMATH_CALUDE_age_group_problem_l1034_103406


namespace NUMINAMATH_CALUDE_quadratic_root_discriminant_square_relation_l1034_103410

theorem quadratic_root_discriminant_square_relation 
  (a b c t : ℝ) (h1 : a ≠ 0) (h2 : a * t^2 + b * t + c = 0) :
  b^2 - 4*a*c = (2*a*t + b)^2 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_discriminant_square_relation_l1034_103410


namespace NUMINAMATH_CALUDE_bottom_right_value_mod_2011_l1034_103422

/-- Represents a cell on the board -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the board configuration -/
structure Board where
  size : Nat
  markedCells : List Cell

/-- The value of a cell on the board -/
def cellValue (board : Board) (cell : Cell) : ℕ :=
  sorry

/-- Theorem stating that the bottom-right corner value is congruent to 2 modulo 2011 -/
theorem bottom_right_value_mod_2011 (board : Board) 
  (h1 : board.size = 2012)
  (h2 : ∀ c ∈ board.markedCells, c.row + c.col = 2011 ∧ c.row ≠ 1 ∧ c.col ≠ 1)
  (h3 : ∀ c, c.row = 1 ∨ c.col = 1 → cellValue board c = 1)
  (h4 : ∀ c ∈ board.markedCells, cellValue board c = 0)
  (h5 : ∀ c, c.row > 1 ∧ c.col > 1 ∧ c ∉ board.markedCells → 
    cellValue board c = cellValue board {row := c.row - 1, col := c.col} + 
                        cellValue board {row := c.row, col := c.col - 1}) :
  cellValue board {row := 2012, col := 2012} ≡ 2 [MOD 2011] :=
sorry

end NUMINAMATH_CALUDE_bottom_right_value_mod_2011_l1034_103422


namespace NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l1034_103400

theorem prop_a_necessary_not_sufficient (h : ℝ) (h_pos : h > 0) :
  (∀ a b : ℝ, (|a - 1| < h ∧ |b - 1| < h) → |a - b| < 2*h) ∧
  (∃ a b : ℝ, |a - b| < 2*h ∧ ¬(|a - 1| < h ∧ |b - 1| < h)) :=
by sorry

end NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l1034_103400


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1034_103413

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² + 5x = 0 -/
def f (x : ℝ) : ℝ := x^2 + 5*x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1034_103413


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1034_103444

theorem consecutive_integers_average (n m : ℤ) : 
  (n > 0) →
  (m = (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7) →
  ((m + (m+1) + (m+2) + (m+3) + (m+4) + (m+5) + (m+6)) / 7 = n + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1034_103444


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1034_103487

/-- The area of the region within a rectangle of dimensions 5 by 6 units,
    but outside three semicircles with radii 2, 3, and 2.5 units, 
    is equal to 30 - 14.625π square units. -/
theorem shaded_area_calculation : 
  let rectangle_area : ℝ := 5 * 6
  let semicircle_area (r : ℝ) : ℝ := (1/2) * Real.pi * r^2
  let total_semicircle_area : ℝ := semicircle_area 2 + semicircle_area 3 + semicircle_area 2.5
  rectangle_area - total_semicircle_area = 30 - 14.625 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1034_103487


namespace NUMINAMATH_CALUDE_aarti_work_completion_l1034_103436

/-- Given that Aarti can complete a piece of work in 9 days, 
    this theorem proves that she will complete 3 times the same work in 27 days. -/
theorem aarti_work_completion :
  ∀ (work : ℕ) (days : ℕ),
    days = 9 →  -- Aarti can complete the work in 9 days
    (27 : ℚ) / days = 3 -- The ratio of 27 days to the original work duration is 3
    :=
by
  sorry

end NUMINAMATH_CALUDE_aarti_work_completion_l1034_103436


namespace NUMINAMATH_CALUDE_simplify_expression_l1034_103476

theorem simplify_expression (x : ℝ) : (3 * x + 8) + (50 * x + 25) = 53 * x + 33 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1034_103476


namespace NUMINAMATH_CALUDE_integer_fraction_pairs_l1034_103494

theorem integer_fraction_pairs : 
  ∀ a b : ℕ+, 
    (((a : ℤ)^3 * (b : ℤ) - 1) % ((a : ℤ) + 1) = 0 ∧ 
     ((b : ℤ)^3 * (a : ℤ) + 1) % ((b : ℤ) - 1) = 0) ↔ 
    ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_pairs_l1034_103494


namespace NUMINAMATH_CALUDE_min_score_11_l1034_103466

def basketball_problem (scores : List ℕ) (score_11 : ℕ) : Prop :=
  let total_10 := scores.sum + 15 + 22 + 18
  let avg_10 := total_10 / 10
  let avg_7 := (total_10 - (15 + 22 + 18)) / 7
  let total_11 := total_10 + score_11
  let avg_11 := total_11 / 11
  (scores.length = 7) ∧
  (avg_10 > avg_7) ∧
  (avg_11 > 20) ∧
  (∀ s : ℕ, s < score_11 → (total_10 + s) / 11 ≤ 20)

theorem min_score_11 (scores : List ℕ) :
  basketball_problem scores 33 → 
  ∀ n : ℕ, n < 33 → ¬(basketball_problem scores n) :=
by sorry

end NUMINAMATH_CALUDE_min_score_11_l1034_103466


namespace NUMINAMATH_CALUDE_product_of_valid_bases_l1034_103488

theorem product_of_valid_bases : ∃ (S : Finset ℕ), 
  (∀ b ∈ S, b ≥ 2 ∧ 
    (∃ (P : Finset ℕ), (∀ p ∈ P, Nat.Prime p) ∧ 
      Finset.card P = b ∧
      (b^6 - 1) / (b - 1) = Finset.prod P id)) ∧
  Finset.prod S id = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_valid_bases_l1034_103488


namespace NUMINAMATH_CALUDE_a4_value_l1034_103493

theorem a4_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5) →
  a₄ = -5 := by
sorry

end NUMINAMATH_CALUDE_a4_value_l1034_103493


namespace NUMINAMATH_CALUDE_better_scores_seventh_grade_l1034_103440

/-- Represents the grade level of students -/
inductive Grade
  | Seventh
  | Eighth

/-- Represents statistical measures for a set of scores -/
structure ScoreStatistics where
  mean : ℝ
  median : ℝ
  mode : ℝ
  variance : ℝ

/-- The test scores for a grade -/
def scores (g : Grade) : List ℝ :=
  match g with
  | Grade.Seventh => [96, 85, 90, 86, 93, 92, 95, 81, 75, 81]
  | Grade.Eighth => [68, 95, 83, 93, 94, 75, 85, 95, 95, 77]

/-- The statistical measures for a grade -/
def statistics (g : Grade) : ScoreStatistics :=
  match g with
  | Grade.Seventh => ⟨87.4, 88, 81, 43.44⟩
  | Grade.Eighth => ⟨86, 89, 95, 89.2⟩

/-- Maximum possible score -/
def maxScore : ℝ := 100

theorem better_scores_seventh_grade :
  (statistics Grade.Seventh).median = 88 ∧
  (statistics Grade.Eighth).mode = 95 ∧
  (statistics Grade.Seventh).mean > (statistics Grade.Eighth).mean ∧
  (statistics Grade.Seventh).variance < (statistics Grade.Eighth).variance :=
by sorry

end NUMINAMATH_CALUDE_better_scores_seventh_grade_l1034_103440


namespace NUMINAMATH_CALUDE_solve_equation_l1034_103414

theorem solve_equation (q r x : ℚ) : 
  (5 / 6 : ℚ) = q / 90 ∧ 
  (5 / 6 : ℚ) = (q + r) / 102 ∧ 
  (5 / 6 : ℚ) = (x - r) / 150 → 
  x = 135 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1034_103414


namespace NUMINAMATH_CALUDE_triangle_third_side_length_triangle_third_side_length_proof_l1034_103433

/-- Given a triangle with perimeter 160 and two sides of lengths 40 and 50,
    the length of the third side is 70. -/
theorem triangle_third_side_length : ℝ → ℝ → ℝ → Prop :=
  fun (perimeter side1 side2 : ℝ) =>
    perimeter = 160 ∧ side1 = 40 ∧ side2 = 50 →
    ∃ (side3 : ℝ), side3 = 70 ∧ perimeter = side1 + side2 + side3

/-- Proof of the theorem -/
theorem triangle_third_side_length_proof :
  triangle_third_side_length 160 40 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_triangle_third_side_length_proof_l1034_103433


namespace NUMINAMATH_CALUDE_magician_marbles_problem_l1034_103447

theorem magician_marbles_problem (initial_red : ℕ) (initial_blue : ℕ) 
  (red_taken : ℕ) (blue_taken_multiplier : ℕ) :
  initial_red = 20 →
  initial_blue = 30 →
  red_taken = 3 →
  blue_taken_multiplier = 4 →
  (initial_red - red_taken) + (initial_blue - (blue_taken_multiplier * red_taken)) = 35 :=
by sorry

end NUMINAMATH_CALUDE_magician_marbles_problem_l1034_103447


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l1034_103423

-- Define the capacities of the pitchers
def pitcher1_capacity : ℚ := 800
def pitcher2_capacity : ℚ := 700

-- Define the fractions of orange juice in each pitcher
def pitcher1_juice_fraction : ℚ := 1/4
def pitcher2_juice_fraction : ℚ := 1/3

-- Calculate the amount of orange juice in each pitcher
def pitcher1_juice : ℚ := pitcher1_capacity * pitcher1_juice_fraction
def pitcher2_juice : ℚ := pitcher2_capacity * pitcher2_juice_fraction

-- Calculate the total amount of orange juice
def total_juice : ℚ := pitcher1_juice + pitcher2_juice

-- Calculate the total volume of the mixture
def total_volume : ℚ := pitcher1_capacity + pitcher2_capacity

-- Define the fraction of orange juice in the large container
def juice_fraction : ℚ := total_juice / total_volume

-- Theorem to prove
theorem orange_juice_fraction :
  juice_fraction = 433.33 / 1500 := by sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l1034_103423


namespace NUMINAMATH_CALUDE_power_multiplication_l1034_103478

theorem power_multiplication (a : ℝ) : a^2 * a^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1034_103478


namespace NUMINAMATH_CALUDE_gdp_equality_l1034_103407

/-- Represents the GDP value in billions of yuan -/
def gdp_billions : ℝ := 4504.5

/-- Represents the same GDP value in scientific notation -/
def gdp_scientific : ℝ := 4.5045 * (10 ^ 12)

/-- Theorem stating that the GDP value in billions is equal to its scientific notation representation -/
theorem gdp_equality : gdp_billions * (10 ^ 9) = gdp_scientific := by sorry

end NUMINAMATH_CALUDE_gdp_equality_l1034_103407


namespace NUMINAMATH_CALUDE_inequality_proof_l1034_103402

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 2 * (x * y + y * z + z * x) = x * y * z) :
  (1 / ((x - 2) * (y - 2) * (z - 2))) + (8 / ((x + 2) * (y + 2) * (z + 2))) ≤ 1 / 32 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1034_103402


namespace NUMINAMATH_CALUDE_relationship_abc_l1034_103419

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.rpow 0.8 0.8)
  (hb : b = Real.rpow 0.8 1.2)
  (hc : c = Real.rpow 1.2 0.8) : 
  c > a ∧ a > b :=
sorry

end NUMINAMATH_CALUDE_relationship_abc_l1034_103419


namespace NUMINAMATH_CALUDE_factorization_equality_l1034_103464

theorem factorization_equality (a b c : ℤ) :
  -14 * a * b * c - 7 * a * b + 49 * a * b^2 * c = -7 * a * b * (2 * c + 1 - 7 * b * c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1034_103464


namespace NUMINAMATH_CALUDE_minimum_loaves_needed_l1034_103442

def slices_per_loaf : ℕ := 20
def regular_sandwich_slices : ℕ := 2
def double_meat_sandwich_slices : ℕ := 3
def triple_decker_sandwich_slices : ℕ := 4
def club_sandwich_slices : ℕ := 5
def regular_sandwiches : ℕ := 25
def double_meat_sandwiches : ℕ := 18
def triple_decker_sandwiches : ℕ := 12
def club_sandwiches : ℕ := 8

theorem minimum_loaves_needed : 
  ∃ (loaves : ℕ), 
    loaves * slices_per_loaf = 
      regular_sandwiches * regular_sandwich_slices +
      double_meat_sandwiches * double_meat_sandwich_slices +
      triple_decker_sandwiches * triple_decker_sandwich_slices +
      club_sandwiches * club_sandwich_slices ∧
    loaves = 10 := by
  sorry

end NUMINAMATH_CALUDE_minimum_loaves_needed_l1034_103442


namespace NUMINAMATH_CALUDE_arithmetic_progression_duality_l1034_103456

theorem arithmetic_progression_duality 
  (x y z k p n : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hk : k > 0) (hp : p > 0) (hn : n > 0)
  (h_arith : ∃ (a d : ℝ), x = a + d * (k - 1) ∧ 
                          y = a + d * (p - 1) ∧ 
                          z = a + d * (n - 1)) :
  ∃ (a' d' : ℝ), 
    (k = a' + d' * (x - 1) ∧
     p = a' + d' * (y - 1) ∧
     n = a' + d' * (z - 1)) ∧
    (∃ (d : ℝ), d * d' = 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_duality_l1034_103456


namespace NUMINAMATH_CALUDE_rain_on_tuesday_l1034_103427

theorem rain_on_tuesday (rain_monday : ℝ) (rain_both : ℝ) (no_rain : ℝ)
  (h1 : rain_monday = 0.7)
  (h2 : rain_both = 0.4)
  (h3 : no_rain = 0.2) :
  ∃ rain_tuesday : ℝ,
    rain_tuesday = 0.5 ∧
    rain_monday + rain_tuesday - rain_both = 1 - no_rain :=
by
  sorry

end NUMINAMATH_CALUDE_rain_on_tuesday_l1034_103427


namespace NUMINAMATH_CALUDE_hcf_problem_l1034_103445

theorem hcf_problem (a b : ℕ+) (h1 : max a b = 414) 
  (h2 : Nat.lcm a b = Nat.gcd a b * 13 * 18) : Nat.gcd a b = 23 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l1034_103445


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l1034_103448

theorem sufficient_condition_range (a : ℝ) : 
  (a > 0) →
  (∀ x : ℝ, (|x - 4| > 6 → x^2 - 2*x + 1 - a^2 > 0)) →
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 > 0 ∧ |x - 4| ≤ 6) →
  (0 < a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l1034_103448


namespace NUMINAMATH_CALUDE_equation_solutions_l1034_103429

theorem equation_solutions : 
  ∀ (m n : ℕ), 3 * 2^m + 1 = n^2 ↔ (m = 0 ∧ n = 2) ∨ (m = 3 ∧ n = 5) ∨ (m = 4 ∧ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1034_103429


namespace NUMINAMATH_CALUDE_symmetric_with_x_minus_y_factor_implies_squared_factor_l1034_103435

-- Define a symmetric polynomial
def is_symmetric (p : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, p x y = p y x

-- Define what it means for (x - y) to be a factor
def has_x_minus_y_factor (p : ℝ → ℝ → ℝ) : Prop :=
  ∃ q : ℝ → ℝ → ℝ, ∀ x y, p x y = (x - y) * q x y

-- Define what it means for (x - y)^2 to be a factor
def has_x_minus_y_squared_factor (p : ℝ → ℝ → ℝ) : Prop :=
  ∃ r : ℝ → ℝ → ℝ, ∀ x y, p x y = (x - y)^2 * r x y

-- The theorem to be proved
theorem symmetric_with_x_minus_y_factor_implies_squared_factor
  (p : ℝ → ℝ → ℝ)
  (h_sym : is_symmetric p)
  (h_factor : has_x_minus_y_factor p) :
  has_x_minus_y_squared_factor p :=
sorry

end NUMINAMATH_CALUDE_symmetric_with_x_minus_y_factor_implies_squared_factor_l1034_103435


namespace NUMINAMATH_CALUDE_max_threes_in_selection_l1034_103491

/-- Represents the count of each card type in the selection -/
structure CardSelection :=
  (threes : ℕ)
  (fours : ℕ)
  (fives : ℕ)

/-- The problem constraints -/
def isValidSelection (s : CardSelection) : Prop :=
  s.threes + s.fours + s.fives = 8 ∧
  3 * s.threes + 4 * s.fours + 5 * s.fives = 27 ∧
  s.threes ≤ 10 ∧ s.fours ≤ 10 ∧ s.fives ≤ 10

/-- The theorem statement -/
theorem max_threes_in_selection :
  ∃ (s : CardSelection), isValidSelection s ∧
    (∀ (t : CardSelection), isValidSelection t → t.threes ≤ s.threes) ∧
    s.threes = 6 :=
sorry

end NUMINAMATH_CALUDE_max_threes_in_selection_l1034_103491


namespace NUMINAMATH_CALUDE_shopkeeper_milk_ounces_l1034_103438

/-- Calculates the total amount of milk in fluid ounces bought by a shopkeeper -/
theorem shopkeeper_milk_ounces (packets : ℕ) (ml_per_packet : ℕ) (ml_per_fl_oz : ℕ) 
    (h1 : packets = 150)
    (h2 : ml_per_packet = 250)
    (h3 : ml_per_fl_oz = 30) : 
  (packets * ml_per_packet) / ml_per_fl_oz = 1250 := by
  sorry


end NUMINAMATH_CALUDE_shopkeeper_milk_ounces_l1034_103438


namespace NUMINAMATH_CALUDE_mean_cat_weight_l1034_103468

def cat_weights : List ℝ := [87, 90, 93, 95, 95, 98, 104, 106, 106, 107, 109, 110, 111, 112]

theorem mean_cat_weight :
  let n : ℕ := cat_weights.length
  let sum : ℝ := cat_weights.sum
  sum / n = 101.64 := by sorry

end NUMINAMATH_CALUDE_mean_cat_weight_l1034_103468


namespace NUMINAMATH_CALUDE_sector_angle_l1034_103489

theorem sector_angle (r : ℝ) (α : ℝ) 
  (h1 : 2 * r + α * r = 4)  -- circumference of sector is 4
  (h2 : (1 / 2) * α * r^2 = 1)  -- area of sector is 1
  : α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1034_103489


namespace NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_unique_perpendicular_plane_l1034_103484

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (passes_through : Plane → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_perpendicular
  (a b : Line) (α : Plane) :
  parallel a α → perpendicular_line_plane b α →
  perpendicular_line_line a b :=
sorry

-- Theorem 2
theorem unique_perpendicular_plane
  (a b : Line) :
  perpendicular_line_line a b →
  ∃! p : Plane, passes_through p b ∧ perpendicular_line_plane a p :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_unique_perpendicular_plane_l1034_103484


namespace NUMINAMATH_CALUDE_distance_to_nearest_town_l1034_103490

theorem distance_to_nearest_town (d : ℝ) :
  (¬ (d ≥ 6)) ∧ (¬ (d ≤ 5)) ∧ (¬ (d ≤ 4)) → 5 < d ∧ d < 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_nearest_town_l1034_103490


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_seven_l1034_103421

/-- Given two nonconstant geometric sequences with first term k and different common ratios p and r,
    if a₃ - b₃ = 7(a₂ - b₂), then p + r = 7. -/
theorem sum_of_common_ratios_is_seven
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  k * p^2 - k * r^2 = 7 * (k * p - k * r) → p + r = 7 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_common_ratios_is_seven_l1034_103421


namespace NUMINAMATH_CALUDE_polynomial_identity_l1034_103480

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (2 - Real.sqrt 3 * x)^8 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  (a₀ + a₂ + a₄ + a₆ + a₈)^2 - (a₁ + a₃ + a₅ + a₇)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1034_103480


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1034_103409

theorem complex_equation_sum (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I →
  a + b = 4 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1034_103409


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l1034_103405

/-- The intersection points of the parabolas y = (x + 2)^2 and x + 2 = (y - 1)^2 lie on a circle with r^2 = 2 -/
theorem intersection_points_on_circle : ∃ (c : ℝ × ℝ) (r : ℝ),
  (∀ x y : ℝ, y = (x + 2)^2 ∧ x + 2 = (y - 1)^2 →
    (x - c.1)^2 + (y - c.2)^2 = r^2) ∧
  r^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l1034_103405


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l1034_103472

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle ABC is (4,3,2) -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (4, 3, 2) :=
by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l1034_103472


namespace NUMINAMATH_CALUDE_washing_machine_payment_l1034_103465

theorem washing_machine_payment (remaining_payment : ℝ) (remaining_percentage : ℝ) 
  (part_payment_percentage : ℝ) (h1 : remaining_payment = 3683.33) 
  (h2 : remaining_percentage = 85) (h3 : part_payment_percentage = 15) : 
  (part_payment_percentage / 100) * (remaining_payment / (remaining_percentage / 100)) = 649.95 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_payment_l1034_103465


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1034_103495

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1 : ℤ) * d

theorem arithmetic_sequence_problem :
  ∃ n : ℕ, arithmetic_sequence (-1) 2 n = 15 ∧ n = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1034_103495


namespace NUMINAMATH_CALUDE_train_crossing_time_l1034_103469

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  train_speed_kmh = 54 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1034_103469


namespace NUMINAMATH_CALUDE_emily_remaining_toys_l1034_103459

/-- The number of toys Emily started with -/
def initial_toys : ℕ := 7

/-- The number of toys Emily sold -/
def sold_toys : ℕ := 3

/-- The number of toys Emily has left -/
def remaining_toys : ℕ := initial_toys - sold_toys

theorem emily_remaining_toys : remaining_toys = 4 := by
  sorry

end NUMINAMATH_CALUDE_emily_remaining_toys_l1034_103459


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l1034_103481

/-- Four real numbers are in arithmetic sequence -/
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r

/-- The sum of the first and last terms equals the sum of the middle terms -/
def sum_property (a b c d : ℝ) : Prop :=
  a + d = b + c

theorem arithmetic_sequence_condition (a b c d : ℝ) :
  (is_arithmetic_sequence a b c d → sum_property a b c d) ∧
  ¬(sum_property a b c d → is_arithmetic_sequence a b c d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l1034_103481


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_21_l1034_103411

/-- A set of consecutive positive integers -/
def ConsecutiveSet (start : ℕ) (length : ℕ) : Set ℕ :=
  {n | start ≤ n ∧ n < start + length}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

/-- Predicate for a valid set of consecutive integers summing to 21 -/
def ValidSet (start : ℕ) (length : ℕ) : Prop :=
  length ≥ 2 ∧ ConsecutiveSum start length = 21

theorem unique_consecutive_sum_21 :
  ∃! p : ℕ × ℕ, ValidSet p.1 p.2 := by sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_21_l1034_103411


namespace NUMINAMATH_CALUDE_selling_to_buying_price_ratio_l1034_103415

theorem selling_to_buying_price_ratio 
  (natasha_money : ℕ) 
  (natasha_carla_ratio : ℕ) 
  (carla_cosima_ratio : ℕ) 
  (profit : ℕ) 
  (h1 : natasha_money = 60)
  (h2 : natasha_carla_ratio = 3)
  (h3 : carla_cosima_ratio = 2)
  (h4 : profit = 36) :
  let carla_money := natasha_money / natasha_carla_ratio
  let cosima_money := carla_money / carla_cosima_ratio
  let total_money := natasha_money + carla_money + cosima_money
  let selling_price := total_money + profit
  ∃ (a b : ℕ), a = 7 ∧ b = 5 ∧ selling_price * b = total_money * a :=
by sorry

end NUMINAMATH_CALUDE_selling_to_buying_price_ratio_l1034_103415


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1034_103420

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1034_103420


namespace NUMINAMATH_CALUDE_parabola_vertex_l1034_103403

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 3 * (x - 2)^2 - 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, -5)

/-- Theorem: The vertex of the parabola y = 3(x-2)^2 - 5 is (2, -5) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1034_103403


namespace NUMINAMATH_CALUDE_square_difference_equality_l1034_103443

theorem square_difference_equality : (36 + 9)^2 - (9^2 + 36^2) = 648 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1034_103443


namespace NUMINAMATH_CALUDE_find_divisor_l1034_103452

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 13787)
  (h2 : quotient = 89)
  (h3 : remainder = 14)
  (h4 : dividend = quotient * 155 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 155 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1034_103452


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l1034_103457

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) : 
  total_caps = 35 → num_groups = 7 → caps_per_group = total_caps / num_groups → caps_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l1034_103457


namespace NUMINAMATH_CALUDE_domino_game_strategy_l1034_103426

/-- Represents the players in the game -/
inductive Player
| Alice
| Bob

/-- Represents the outcome of the game -/
inductive Outcome
| Win
| Lose

/-- Represents a grid in the domino game -/
structure Grid :=
  (n : ℕ)
  (m : ℕ)

/-- Determines if a player has a winning strategy on a given grid -/
def has_winning_strategy (player : Player) (grid : Grid) : Prop :=
  match player with
  | Player.Alice => 
      (grid.n % 2 = 0 ∧ grid.m % 2 = 1) ∨
      (grid.n % 2 = 1 ∧ grid.m % 2 = 0)
  | Player.Bob => 
      (grid.n % 2 = 0 ∧ grid.m % 2 = 0)

/-- Theorem stating the winning strategies for the domino game -/
theorem domino_game_strategy (grid : Grid) :
  (grid.n % 2 = 0 ∧ grid.m % 2 = 0 → has_winning_strategy Player.Bob grid) ∧
  (grid.n % 2 = 0 ∧ grid.m % 2 = 1 → has_winning_strategy Player.Alice grid) :=
sorry

end NUMINAMATH_CALUDE_domino_game_strategy_l1034_103426


namespace NUMINAMATH_CALUDE_cos_330_degrees_l1034_103496

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l1034_103496


namespace NUMINAMATH_CALUDE_membership_change_l1034_103460

theorem membership_change (initial_members : ℝ) (h : initial_members > 0) :
  let fall_increase := 0.04
  let spring_decrease := 0.19
  let fall_members := initial_members * (1 + fall_increase)
  let spring_members := fall_members * (1 - spring_decrease)
  let total_change := (spring_members - initial_members) / initial_members
  total_change = -0.1576 := by
sorry

end NUMINAMATH_CALUDE_membership_change_l1034_103460


namespace NUMINAMATH_CALUDE_geoffrey_game_cost_l1034_103453

theorem geoffrey_game_cost (initial_money : ℕ) : 
  initial_money + 20 + 25 + 30 = 125 → 
  ∃ (game_cost : ℕ), 
    game_cost * 3 = 125 - 20 ∧ 
    game_cost = 35 := by
  sorry

end NUMINAMATH_CALUDE_geoffrey_game_cost_l1034_103453


namespace NUMINAMATH_CALUDE_sum_of_leading_digits_of_roots_l1034_103474

/-- A function that returns the leading digit of a positive real number -/
def leadingDigit (x : ℝ) : ℕ :=
  sorry

/-- The number M, which is a 303-digit number consisting only of 5s -/
def M : ℕ := sorry

/-- The function g that returns the leading digit of the r-th root of M -/
def g (r : ℕ) : ℕ :=
  leadingDigit (M ^ (1 / r : ℝ))

/-- Theorem stating that the sum of g(2) to g(6) is 10 -/
theorem sum_of_leading_digits_of_roots :
  g 2 + g 3 + g 4 + g 5 + g 6 = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_leading_digits_of_roots_l1034_103474


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_ten_l1034_103416

theorem sqrt_sum_equals_ten : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_ten_l1034_103416


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l1034_103492

theorem easter_egg_hunt (kevin bonnie cheryl george : ℕ) 
  (h1 : kevin = 5)
  (h2 : bonnie = 13)
  (h3 : cheryl = 56)
  (h4 : cheryl = kevin + bonnie + george + 29) :
  george = 9 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l1034_103492


namespace NUMINAMATH_CALUDE_sphere_tangent_angle_theorem_l1034_103417

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Check if three lines are parallel -/
def areParallel (l1 l2 l3 : Line3D) : Prop := sorry

/-- Check if a line is tangent to a sphere -/
def isTangent (l : Line3D) (s : Sphere) : Prop := sorry

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point3D) : ℝ := sorry

theorem sphere_tangent_angle_theorem (O K L M : Point3D) (s : Sphere) (l1 l2 l3 : Line3D) :
  s.center = O →
  s.radius = 5 →
  areParallel l1 l2 l3 →
  isTangent l1 s →
  isTangent l2 s →
  isTangent l3 s →
  triangleArea O K L = 12 →
  triangleArea K L M > 30 →
  angle K M L = Real.arccos (3/5) := by
  sorry

end NUMINAMATH_CALUDE_sphere_tangent_angle_theorem_l1034_103417


namespace NUMINAMATH_CALUDE_debt_payment_proof_l1034_103485

theorem debt_payment_proof (x : ℝ) : 
  (20 * x + 20 * (x + 65)) / 40 = 442.5 → x = 410 := by
  sorry

end NUMINAMATH_CALUDE_debt_payment_proof_l1034_103485


namespace NUMINAMATH_CALUDE_work_completion_workers_work_completion_workers_proof_l1034_103471

/-- Given a work that can be finished in 12 days by an initial group of workers,
    and is finished in 9 days after 10 more workers join,
    prove that the total number of workers after the addition is 40. -/
theorem work_completion_workers : ℕ → Prop :=
  λ initial_workers =>
    (initial_workers * 12 = (initial_workers + 10) * 9) →
    initial_workers + 10 = 40
  
#check work_completion_workers

/-- Proof of the theorem -/
theorem work_completion_workers_proof : ∃ n : ℕ, work_completion_workers n := by
  sorry

end NUMINAMATH_CALUDE_work_completion_workers_work_completion_workers_proof_l1034_103471


namespace NUMINAMATH_CALUDE_movie_deal_savings_l1034_103441

theorem movie_deal_savings : 
  let deal_price : ℚ := 20
  let movie_price : ℚ := 8
  let popcorn_price : ℚ := movie_price - 3
  let drink_price : ℚ := popcorn_price + 1
  let candy_price : ℚ := drink_price / 2
  let total_price : ℚ := movie_price + popcorn_price + drink_price + candy_price
  total_price - deal_price = 2 := by sorry

end NUMINAMATH_CALUDE_movie_deal_savings_l1034_103441


namespace NUMINAMATH_CALUDE_circle_radius_relation_l1034_103432

theorem circle_radius_relation (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  (π * x^2 = π * y^2) → (2 * π * x = 20 * π) → y / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_relation_l1034_103432


namespace NUMINAMATH_CALUDE_min_cost_trees_l1034_103404

/-- The cost function for purchasing trees -/
def cost_function (x : ℕ) : ℕ := 20 * x + 12000

/-- The constraint on the number of cypress trees -/
def cypress_constraint (x : ℕ) : Prop := x ≥ 3 * (150 - x)

/-- The total number of trees to be purchased -/
def total_trees : ℕ := 150

/-- The theorem stating the minimum cost and optimal purchase -/
theorem min_cost_trees :
  ∃ (x : ℕ), 
    x ≤ total_trees ∧
    cypress_constraint x ∧
    (∀ (y : ℕ), y ≤ total_trees → cypress_constraint y → cost_function x ≤ cost_function y) ∧
    x = 113 ∧
    cost_function x = 14260 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_trees_l1034_103404


namespace NUMINAMATH_CALUDE_candy_bar_price_is_correct_l1034_103475

/-- The price of a candy bar in dollars -/
def candy_bar_price : ℝ := 2

/-- The price of a bag of chips in dollars -/
def chips_price : ℝ := 0.5

/-- The number of students -/
def num_students : ℕ := 5

/-- The total amount needed for all students in dollars -/
def total_amount : ℝ := 15

/-- The number of candy bars each student gets -/
def candy_bars_per_student : ℕ := 1

/-- The number of bags of chips each student gets -/
def chips_per_student : ℕ := 2

theorem candy_bar_price_is_correct : 
  candy_bar_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_price_is_correct_l1034_103475


namespace NUMINAMATH_CALUDE_unique_solution_system_l1034_103497

theorem unique_solution_system (x y z : ℝ) :
  (x - 1) * (y - 1) * (z - 1) = x * y * z - 1 ∧
  (x - 2) * (y - 2) * (z - 2) = x * y * z - 2 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1034_103497


namespace NUMINAMATH_CALUDE_woods_length_l1034_103425

/-- Given a rectangular area of woods with width 8 miles and total area 24 square miles,
    prove that the length of the woods is 3 miles. -/
theorem woods_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 8 → area = 24 → area = width * length → length = 3 := by sorry

end NUMINAMATH_CALUDE_woods_length_l1034_103425


namespace NUMINAMATH_CALUDE_log_ratio_simplification_l1034_103498

theorem log_ratio_simplification (x : ℝ) 
  (h1 : 5 * x^3 > 0) (h2 : 7 * x - 3 > 0) : 
  (Real.log (Real.sqrt (7 * x - 3)) / Real.log (5 * x^3)) / Real.log (7 * x - 3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_simplification_l1034_103498


namespace NUMINAMATH_CALUDE_bus_distance_traveled_l1034_103486

theorem bus_distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 67 → time = 3 → distance = speed * time → distance = 201 := by sorry

end NUMINAMATH_CALUDE_bus_distance_traveled_l1034_103486


namespace NUMINAMATH_CALUDE_largest_valid_number_nine_zero_nine_nine_is_valid_nine_zero_nine_nine_is_largest_l1034_103446

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n / 10) % 10 = (n / 1000) % 10 + (n / 100) % 10 ∧
  n % 10 = (n / 100) % 10 + (n / 10) % 10

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 9099 :=
by sorry

theorem nine_zero_nine_nine_is_valid :
  is_valid_number 9099 :=
by sorry

theorem nine_zero_nine_nine_is_largest :
  ∀ n : ℕ, is_valid_number n → n = 9099 ∨ n < 9099 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_nine_zero_nine_nine_is_valid_nine_zero_nine_nine_is_largest_l1034_103446


namespace NUMINAMATH_CALUDE_dataset_statistics_l1034_103463

def dataset : List ℝ := [158, 149, 155, 157, 156, 162, 155, 168]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def mode (l : List ℝ) : ℝ := sorry

theorem dataset_statistics :
  median dataset = 155.5 ∧
  mean dataset = 157.5 ∧
  mode dataset = 155 := by
  sorry

end NUMINAMATH_CALUDE_dataset_statistics_l1034_103463


namespace NUMINAMATH_CALUDE_sin_sqrt3_over_2_solution_set_l1034_103479

theorem sin_sqrt3_over_2_solution_set (θ : ℝ) : 
  Real.sin θ = (Real.sqrt 3) / 2 ↔ 
  ∃ k : ℤ, θ = π / 3 + 2 * k * π ∨ θ = 2 * π / 3 + 2 * k * π :=
sorry

end NUMINAMATH_CALUDE_sin_sqrt3_over_2_solution_set_l1034_103479


namespace NUMINAMATH_CALUDE_square_root_of_16_l1034_103430

theorem square_root_of_16 : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_16_l1034_103430


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1034_103401

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17)
  (h2 : a * b + c + d = 85)
  (h3 : a * d + b * c = 180)
  (h4 : c * d = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 934 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1034_103401


namespace NUMINAMATH_CALUDE_periodic_sum_implies_constant_l1034_103450

/-- A function is periodic with period a if f(x + a) = f(x) for all x --/
def IsPeriodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a ≠ 0 ∧ ∀ x, f (x + a) = f x

theorem periodic_sum_implies_constant
  (f g : ℝ → ℝ) (a b : ℝ)
  (hfa : IsPeriodic f a)
  (hgb : IsPeriodic g b)
  (ha_rat : ℚ)
  (hb_irrat : Irrational b)
  (h_sum_periodic : ∃ c, IsPeriodic (f + g) c) :
  (∃ k, ∀ x, f x = k) ∨ (∃ k, ∀ x, g x = k) := by
  sorry

end NUMINAMATH_CALUDE_periodic_sum_implies_constant_l1034_103450


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1034_103424

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 17 + (3 * x) + 15 + (3 * x + 6)) / 5 = 26 → x = 82 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1034_103424


namespace NUMINAMATH_CALUDE_expression_factorization_l1034_103431

theorem expression_factorization (x : ℝ) :
  (12 * x^4 + 34 * x^3 + 45 * x - 6) - (3 * x^4 - 7 * x^3 + 8 * x - 6) = x * (9 * x^3 + 41 * x^2 + 37) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1034_103431


namespace NUMINAMATH_CALUDE_inequality_problem_l1034_103483

theorem inequality_problem (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  ¬(1 / (a - b) > 1 / a) := by
sorry

end NUMINAMATH_CALUDE_inequality_problem_l1034_103483


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l1034_103454

-- Define the line ax - by - 2 = 0
def line (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y - 2 = 0

-- Define the curve y = x^3
def curve (x y : ℝ) : Prop := y = x^3

-- Define the point P(1, 1)
def point_P : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line to the curve at P
def tangent_slope_curve : ℝ := 3

-- Define the condition that the tangent lines are mutually perpendicular
def perpendicular_tangents (a b : ℝ) : Prop :=
  (a / b) * tangent_slope_curve = -1

theorem perpendicular_tangents_ratio (a b : ℝ) :
  line a b point_P.1 point_P.2 →
  curve point_P.1 point_P.2 →
  perpendicular_tangents a b →
  a / b = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l1034_103454


namespace NUMINAMATH_CALUDE_man_rowing_speed_l1034_103458

/-- The speed of a man rowing in still water, given his speeds with wind influence -/
theorem man_rowing_speed 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (wind_speed : ℝ) 
  (h1 : upstream_speed = 25) 
  (h2 : downstream_speed = 65) 
  (h3 : wind_speed = 5) : 
  (upstream_speed + downstream_speed) / 2 = 45 := by
sorry


end NUMINAMATH_CALUDE_man_rowing_speed_l1034_103458


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l1034_103482

/-- Calculates the sampling interval for systematic sampling. -/
def samplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The sampling interval for a population of 1500 and sample size of 30 is 50. -/
theorem systematic_sampling_interval :
  samplingInterval 1500 30 = 50 := by
  sorry

#eval samplingInterval 1500 30

end NUMINAMATH_CALUDE_systematic_sampling_interval_l1034_103482


namespace NUMINAMATH_CALUDE_digital_earth_sharing_l1034_103428

/-- Represents the concept of Digital Earth -/
structure DigitalEarth where
  technology : Type
  data : Type
  sharing_method : Type

/-- Represents the internet as a sharing method -/
def Internet : Type := Unit

/-- Axiom: Digital Earth involves digital technology and Earth-related data -/
axiom digital_earth_components : ∀ (de : DigitalEarth), de.technology × de.data

/-- Theorem: Digital Earth can only achieve global information sharing through the internet -/
theorem digital_earth_sharing (de : DigitalEarth) : 
  de.sharing_method = Internet :=
sorry

end NUMINAMATH_CALUDE_digital_earth_sharing_l1034_103428


namespace NUMINAMATH_CALUDE_distance_covered_l1034_103439

/-- Proves that the total distance covered is 6 km given the specified conditions -/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 2.25)
  (h4 : (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time)
  : total_distance = 6 :=
by
  sorry

#check distance_covered

end NUMINAMATH_CALUDE_distance_covered_l1034_103439


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l1034_103434

theorem student_multiplication_problem (x : ℚ) : 
  (63 * x) - 142 = 110 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l1034_103434


namespace NUMINAMATH_CALUDE_simplify_expression_l1034_103461

theorem simplify_expression : (3 * 2 + 4 + 6) / 3 - 2 / 3 = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1034_103461


namespace NUMINAMATH_CALUDE_fixed_point_and_equal_intercept_line_l1034_103470

/-- The fixed point through which all lines of the form ax + y - a - 2 = 0 pass -/
def fixed_point : ℝ × ℝ := (1, 2)

/-- The line equation with parameter a -/
def line_equation (a x y : ℝ) : Prop := a * x + y - a - 2 = 0

/-- A line with equal intercepts on both axes passing through a point -/
def equal_intercept_line (p : ℝ × ℝ) (x y : ℝ) : Prop := x + y = p.1 + p.2

theorem fixed_point_and_equal_intercept_line :
  (∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ (x, y) = fixed_point) ∧
  equal_intercept_line fixed_point = λ x y => x + y = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_and_equal_intercept_line_l1034_103470


namespace NUMINAMATH_CALUDE_line_segment_ratio_l1034_103418

/-- Given points P, Q, R, and S on a straight line in that order,
    with PQ = 3, QR = 7, and PS = 20, prove that PR:QS = 1 -/
theorem line_segment_ratio (P Q R S : ℝ) 
  (h_order : P < Q ∧ Q < R ∧ R < S)
  (h_PQ : Q - P = 3)
  (h_QR : R - Q = 7)
  (h_PS : S - P = 20) :
  (R - P) / (S - Q) = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l1034_103418


namespace NUMINAMATH_CALUDE_sequence_matches_l1034_103477

/-- The sequence defined by a_n = 2^n - 1 -/
def a (n : ℕ) : ℕ := 2^n - 1

/-- The first four terms of the sequence match 1, 3, 7, 15 -/
theorem sequence_matches : 
  (a 1 = 1) ∧ (a 2 = 3) ∧ (a 3 = 7) ∧ (a 4 = 15) := by
  sorry

#eval a 1  -- Expected: 1
#eval a 2  -- Expected: 3
#eval a 3  -- Expected: 7
#eval a 4  -- Expected: 15

end NUMINAMATH_CALUDE_sequence_matches_l1034_103477


namespace NUMINAMATH_CALUDE_apple_pricing_l1034_103408

/-- The price function for apples -/
noncomputable def price (l q x : ℝ) (k : ℝ) : ℝ :=
  if k ≤ x then l * k else l * x + q * (k - x)

theorem apple_pricing (l q x : ℝ) : 
  (price l q x 33 = 11.67) →
  (price l q x 36 = 12.48) →
  (price l q x 10 = 3.62) →
  (x = 30) := by
sorry

end NUMINAMATH_CALUDE_apple_pricing_l1034_103408


namespace NUMINAMATH_CALUDE_ratio_equality_solutions_l1034_103455

theorem ratio_equality_solutions :
  {x : ℝ | (x + 3) / (3 * x + 3) = (3 * x + 4) / (6 * x + 4)} = {0, 1/3} := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_solutions_l1034_103455


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_22_l1034_103467

/-- The repeating decimal 0.454545... --/
def repeating_decimal : ℚ := 5 / 11

theorem product_of_repeating_decimal_and_22 :
  repeating_decimal * 22 = 10 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_22_l1034_103467


namespace NUMINAMATH_CALUDE_bathroom_flooring_area_l1034_103499

/-- The total area of hardwood flooring installed in Nancy's bathroom -/
def total_area (central_length central_width hallway_length hallway_width : ℝ) : ℝ :=
  central_length * central_width + hallway_length * hallway_width

/-- Proof that the total area of hardwood flooring installed in Nancy's bathroom is 124 square feet -/
theorem bathroom_flooring_area :
  total_area 10 10 6 4 = 124 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_flooring_area_l1034_103499


namespace NUMINAMATH_CALUDE_walk_distance_theorem_l1034_103412

/-- Calculates the total distance walked when a person walks at a given speed for a certain time in one direction and then returns along the same path. -/
def totalDistanceWalked (speed : ℝ) (time : ℝ) : ℝ :=
  2 * speed * time

/-- Theorem stating that walking at 2 miles per hour for 3 hours in one direction and returning results in a total distance of 12 miles. -/
theorem walk_distance_theorem :
  totalDistanceWalked 2 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_walk_distance_theorem_l1034_103412


namespace NUMINAMATH_CALUDE_gumball_difference_l1034_103437

/-- The number of gumballs Hector purchased -/
def total_gumballs : ℕ := 45

/-- The number of gumballs Hector gave to Todd -/
def todd_gumballs : ℕ := 4

/-- The number of gumballs Hector gave to Alisha -/
def alisha_gumballs : ℕ := 2 * todd_gumballs

/-- The number of gumballs Hector had remaining -/
def remaining_gumballs : ℕ := 6

/-- The number of gumballs Hector gave to Bobby -/
def bobby_gumballs : ℕ := total_gumballs - todd_gumballs - alisha_gumballs - remaining_gumballs

theorem gumball_difference : 
  4 * alisha_gumballs - bobby_gumballs = 5 := by sorry

end NUMINAMATH_CALUDE_gumball_difference_l1034_103437


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1034_103462

def U : Set Int := {x | (x + 1) * (x - 3) ≤ 0}

def A : Set Int := {0, 1, 2}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1034_103462


namespace NUMINAMATH_CALUDE_exists_always_last_card_l1034_103473

/-- Represents a card with a unique natural number -/
structure Card where
  number : ℕ
  unique : ℕ

/-- Represents the circular arrangement of cards -/
def CardArrangement := Vector Card 1000

/-- Simulates the card removal process -/
def removeCards (arrangement : CardArrangement) (startIndex : Fin 1000) : Card :=
  sorry

/-- Checks if a card is the last remaining for all starting positions except its own -/
def isAlwaysLast (arrangement : CardArrangement) (cardIndex : Fin 1000) : Prop :=
  ∀ i : Fin 1000, i ≠ cardIndex → removeCards arrangement i = arrangement.get cardIndex

/-- Main theorem: There exists a card arrangement where one card is always the last remaining -/
theorem exists_always_last_card : ∃ (arrangement : CardArrangement), ∃ (i : Fin 1000), isAlwaysLast arrangement i :=
  sorry

end NUMINAMATH_CALUDE_exists_always_last_card_l1034_103473


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1034_103449

theorem fraction_subtraction : (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1034_103449


namespace NUMINAMATH_CALUDE_expression_simplification_l1034_103451

/-- Given that |2+y|+(x-1)^2=0, prove that 5x^2*y-[3x*y^2-2(3x*y^2-7/2*x^2*y)] = 16 -/
theorem expression_simplification (x y : ℝ) 
  (h : |2 + y| + (x - 1)^2 = 0) : 
  5*x^2*y - (3*x*y^2 - 2*(3*x*y^2 - 7/2*x^2*y)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1034_103451
