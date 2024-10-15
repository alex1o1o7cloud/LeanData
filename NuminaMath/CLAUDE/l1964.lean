import Mathlib

namespace NUMINAMATH_CALUDE_addition_to_reach_target_l1964_196419

theorem addition_to_reach_target : (1250 / 50) + 7500 = 7525 := by
  sorry

end NUMINAMATH_CALUDE_addition_to_reach_target_l1964_196419


namespace NUMINAMATH_CALUDE_b_share_is_1000_l1964_196443

/-- Given a partnership with investment ratios A:B:C as 2:2/3:1 and a total profit,
    calculate B's share of the profit. -/
def calculate_B_share (total_profit : ℚ) : ℚ :=
  let a_ratio : ℚ := 2
  let b_ratio : ℚ := 2/3
  let c_ratio : ℚ := 1
  let total_ratio : ℚ := a_ratio + b_ratio + c_ratio
  (b_ratio / total_ratio) * total_profit

/-- Theorem stating that given the investment ratios and a total profit of 5500,
    B's share of the profit is 1000. -/
theorem b_share_is_1000 :
  calculate_B_share 5500 = 1000 := by
  sorry

#eval calculate_B_share 5500

end NUMINAMATH_CALUDE_b_share_is_1000_l1964_196443


namespace NUMINAMATH_CALUDE_sales_after_reduction_profit_after_optimal_reduction_l1964_196494

/-- Represents a store's sales and pricing strategy -/
structure Store where
  initial_sales : ℕ
  initial_profit : ℝ
  sales_increase : ℝ
  min_profit : ℝ

/-- Calculates the new sales quantity after a price reduction -/
def new_sales (s : Store) (price_reduction : ℝ) : ℝ :=
  s.initial_sales + s.sales_increase * price_reduction

/-- Calculates the new profit per item after a price reduction -/
def new_profit_per_item (s : Store) (price_reduction : ℝ) : ℝ :=
  s.initial_profit - price_reduction

/-- Calculates the total daily profit after a price reduction -/
def daily_profit (s : Store) (price_reduction : ℝ) : ℝ :=
  new_sales s price_reduction * new_profit_per_item s price_reduction

/-- The store's initial conditions -/
def my_store : Store :=
  { initial_sales := 20
  , initial_profit := 40
  , sales_increase := 2
  , min_profit := 25 }

theorem sales_after_reduction (s : Store) :
  new_sales s 3 = 26 :=
sorry

theorem profit_after_optimal_reduction (s : Store) :
  ∃ (x : ℝ), x = 10 ∧ 
    daily_profit s x = 1200 ∧ 
    new_profit_per_item s x ≥ s.min_profit :=
sorry

end NUMINAMATH_CALUDE_sales_after_reduction_profit_after_optimal_reduction_l1964_196494


namespace NUMINAMATH_CALUDE_scientific_notation_digits_l1964_196403

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log n 10).succ

/-- Conversion from scientific notation to standard form -/
def scientific_to_standard (mantissa : ℚ) (exponent : ℤ) : ℚ :=
  mantissa * (10 : ℚ) ^ exponent

theorem scientific_notation_digits :
  let mantissa : ℚ := 721 / 100
  let exponent : ℤ := 11
  let standard_form := scientific_to_standard mantissa exponent
  num_digits (Nat.floor standard_form) = 12 := by
sorry

end NUMINAMATH_CALUDE_scientific_notation_digits_l1964_196403


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_l1964_196447

theorem imaginary_part_of_one_minus_i :
  Complex.im (1 - Complex.I) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_l1964_196447


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1964_196450

theorem inequality_solution_set (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let S := {x : ℝ | (x - a) * (x + a - 1) < 0}
  (0 ≤ a ∧ a < 1/2 → S = Set.Ioo a (1 - a)) ∧
  (a = 1/2 → S = ∅) ∧
  (1/2 < a ∧ a ≤ 1 → S = Set.Ioo (1 - a) a) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1964_196450


namespace NUMINAMATH_CALUDE_game_points_percentage_l1964_196486

theorem game_points_percentage (samanta mark eric : ℕ) : 
  samanta = mark + 8 →
  eric = 6 →
  samanta + mark + eric = 32 →
  (mark - eric : ℚ) / eric * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_game_points_percentage_l1964_196486


namespace NUMINAMATH_CALUDE_section_area_of_specific_pyramid_l1964_196411

/-- Regular quadrilateral pyramid with square base -/
structure RegularQuadPyramid where
  base_side : ℝ
  height : ℝ

/-- Plane intersecting the pyramid -/
structure IntersectingPlane where
  angle_with_base : ℝ

/-- The area of the section created by the intersecting plane -/
noncomputable def section_area (p : RegularQuadPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

theorem section_area_of_specific_pyramid :
  let p : RegularQuadPyramid := ⟨8, 9⟩
  let plane : IntersectingPlane := ⟨Real.arctan (3/4)⟩
  section_area p plane = 45 := by sorry

end NUMINAMATH_CALUDE_section_area_of_specific_pyramid_l1964_196411


namespace NUMINAMATH_CALUDE_heartsuit_three_five_l1964_196460

-- Define the ♥ operation
def heartsuit (x y : ℤ) : ℤ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_three_five : heartsuit 3 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_five_l1964_196460


namespace NUMINAMATH_CALUDE_abs_T_equals_1024_l1964_196427

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^19 - (1 - i)^19

-- Theorem statement
theorem abs_T_equals_1024 : Complex.abs T = 1024 := by
  sorry

end NUMINAMATH_CALUDE_abs_T_equals_1024_l1964_196427


namespace NUMINAMATH_CALUDE_max_time_at_8_l1964_196467

noncomputable def y (t : ℝ) : ℝ := -1/8 * t^3 - 3/4 * t^2 + 36*t - 629/4

theorem max_time_at_8 :
  ∃ (t_max : ℝ), t_max = 8 ∧
  ∀ (t : ℝ), 6 ≤ t ∧ t ≤ 9 → y t ≤ y t_max :=
by sorry

end NUMINAMATH_CALUDE_max_time_at_8_l1964_196467


namespace NUMINAMATH_CALUDE_cubic_inequality_l1964_196483

theorem cubic_inequality (a b c : ℝ) (h : ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  ∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = x ∨ t = y ∨ t = z) : 
  2*a^3 + 9*c ≤ 7*a*b ∧ 
  (2*a^3 + 9*c = 7*a*b ↔ ∃ r : ℝ, r > 0 ∧ ∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = r) :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1964_196483


namespace NUMINAMATH_CALUDE_scientific_notation_of_240000_l1964_196465

theorem scientific_notation_of_240000 : 
  240000 = 2.4 * (10 ^ 5) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_240000_l1964_196465


namespace NUMINAMATH_CALUDE_golden_ratio_product_ab_pq_minus_n_l1964_196435

/-- The golden ratio is the positive root of x^2 + x - 1 = 0 -/
theorem golden_ratio : ∃ x : ℝ, x > 0 ∧ x^2 + x - 1 = 0 ∧ x = (-1 + Real.sqrt 5) / 2 := by sorry

/-- Given a^2 + ma = 1 and b^2 - 2mb = 4, ab = 2 -/
theorem product_ab (m a b : ℝ) (h1 : a^2 + m*a = 1) (h2 : b^2 - 2*m*b = 4) (h3 : b ≠ -2*a) : a * b = 2 := by sorry

/-- Given p^2 + np - 1 = q and q^2 + nq - 1 = p, pq - n = 0 -/
theorem pq_minus_n (n p q : ℝ) (h1 : p^2 + n*p - 1 = q) (h2 : q^2 + n*q - 1 = p) (h3 : p ≠ q) : p * q - n = 0 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_product_ab_pq_minus_n_l1964_196435


namespace NUMINAMATH_CALUDE_product_53_57_l1964_196488

theorem product_53_57 (h : 2021 = 43 * 47) : 53 * 57 = 3021 := by
  sorry

end NUMINAMATH_CALUDE_product_53_57_l1964_196488


namespace NUMINAMATH_CALUDE_race_probability_inconsistency_l1964_196444

-- Define the probabilities for each car to win
def prob_X_wins : ℚ := 1/2
def prob_Y_wins : ℚ := 1/4
def prob_Z_wins : ℚ := 1/3

-- Define the total probability of one of them winning
def total_prob : ℚ := 1.0833333333333333

-- Theorem stating the inconsistency of the given probabilities
theorem race_probability_inconsistency :
  prob_X_wins + prob_Y_wins + prob_Z_wins = total_prob ∧
  total_prob > 1 := by sorry

end NUMINAMATH_CALUDE_race_probability_inconsistency_l1964_196444


namespace NUMINAMATH_CALUDE_triangle_properties_l1964_196402

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 7 →
  b = 2 →
  A = π / 3 →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (a / Real.sin A = b / Real.sin B) →
  (a / Real.sin A = c / Real.sin C) →
  c = 3 ∧
  Real.sin B = Real.sqrt 21 / 7 ∧
  π * (a / (2 * Real.sin A))^2 = 7 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1964_196402


namespace NUMINAMATH_CALUDE_system_solution_l1964_196420

theorem system_solution (x y : ℝ) : 
  (x^2 + 3*x*y = 12 ∧ x*y = 16 + y^2 - x*y - x^2) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l1964_196420


namespace NUMINAMATH_CALUDE_investment_difference_l1964_196434

def initial_investment : ℕ := 10000

def alice_multiplier : ℕ := 3
def bob_multiplier : ℕ := 7

def alice_final : ℕ := initial_investment * alice_multiplier
def bob_final : ℕ := initial_investment * bob_multiplier

theorem investment_difference : bob_final - alice_final = 40000 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_l1964_196434


namespace NUMINAMATH_CALUDE_conditional_probability_of_longevity_l1964_196476

theorem conditional_probability_of_longevity 
  (p_20 : ℝ) 
  (p_25 : ℝ) 
  (h1 : p_20 = 0.8) 
  (h2 : p_25 = 0.4) : 
  p_25 / p_20 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_of_longevity_l1964_196476


namespace NUMINAMATH_CALUDE_double_first_triple_second_row_l1964_196472

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]

theorem double_first_triple_second_row (A : Matrix (Fin 2) (Fin 2) ℝ) :
  N • A = !![2 * A 0 0, 2 * A 0 1; 3 * A 1 0, 3 * A 1 1] := by sorry

end NUMINAMATH_CALUDE_double_first_triple_second_row_l1964_196472


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1964_196456

theorem polynomial_factorization (x : ℤ) :
  3 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (3 * x^2 + 58 * x + 231) * (x + 7) * (x + 11) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1964_196456


namespace NUMINAMATH_CALUDE_inequality_proof_l1964_196489

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1964_196489


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_60_l1964_196442

theorem largest_multiple_of_8_less_than_60 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 60 → n ≤ 56 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_60_l1964_196442


namespace NUMINAMATH_CALUDE_sin_780_degrees_l1964_196413

theorem sin_780_degrees : Real.sin (780 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_780_degrees_l1964_196413


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l1964_196433

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  paintedSquaresPerFace : Nat
  paintedFaces : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpaintedUnitCubes (cube : PaintedCube) : Nat :=
  cube.totalUnitCubes - paintedUnitCubes cube
where
  /-- Calculates the number of painted unit cubes, accounting for overlaps -/
  paintedUnitCubes (cube : PaintedCube) : Nat :=
    let totalPaintedSquares := cube.paintedSquaresPerFace * cube.paintedFaces
    let edgeOverlap := 12 * 2  -- 12 edges, each counted twice
    let cornerOverlap := 8 * 2  -- 8 corners, each counted thrice (so subtract 2)
    totalPaintedSquares - edgeOverlap - cornerOverlap

/-- The theorem to be proved -/
theorem unpainted_cubes_in_6x6x6 :
  let cube : PaintedCube := {
    size := 6,
    totalUnitCubes := 216,
    paintedSquaresPerFace := 13,
    paintedFaces := 6
  }
  unpaintedUnitCubes cube = 210 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l1964_196433


namespace NUMINAMATH_CALUDE_closest_fraction_to_japan_medals_l1964_196464

theorem closest_fraction_to_japan_medals :
  let japan_fraction : ℚ := 25 / 120
  let fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]
  (1/5 : ℚ) = fractions.argmin (fun x => |x - japan_fraction|) := by
  sorry

end NUMINAMATH_CALUDE_closest_fraction_to_japan_medals_l1964_196464


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l1964_196409

theorem distinct_prime_factors_of_90 : Finset.card (Nat.factors 90).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l1964_196409


namespace NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l1964_196484

theorem fourth_root_over_sixth_root_of_seven (x : ℝ) :
  (7 : ℝ)^(1/4) / (7 : ℝ)^(1/6) = (7 : ℝ)^(1/12) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l1964_196484


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l1964_196438

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l1964_196438


namespace NUMINAMATH_CALUDE_carole_wins_iff_n_odd_l1964_196407

/-- The game interval -/
def GameInterval (n : ℕ) := Set.Icc (0 : ℝ) n

/-- Predicate for a valid move -/
def ValidMove (prev : Set ℝ) (x : ℝ) : Prop :=
  ∀ y ∈ prev, |x - y| ≥ 1.5

/-- The game state -/
structure GameState (n : ℕ) where
  chosen : Set ℝ
  current_player : Bool -- true for Carole, false for Leo

/-- The game result -/
inductive GameResult
  | CaroleWins
  | LeoWins

/-- Optimal strategy -/
def OptimalStrategy (n : ℕ) : GameState n → GameResult :=
  sorry

/-- The main theorem -/
theorem carole_wins_iff_n_odd (n : ℕ) (h : n > 10) :
  OptimalStrategy n { chosen := ∅, current_player := true } = GameResult.CaroleWins ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_carole_wins_iff_n_odd_l1964_196407


namespace NUMINAMATH_CALUDE_mike_marks_short_l1964_196468

def passing_threshold (max_marks : ℕ) : ℕ := (30 * max_marks) / 100

theorem mike_marks_short (max_marks mike_score : ℕ) 
  (h1 : max_marks = 760) 
  (h2 : mike_score = 212) : 
  passing_threshold max_marks - mike_score = 16 := by
  sorry

end NUMINAMATH_CALUDE_mike_marks_short_l1964_196468


namespace NUMINAMATH_CALUDE_clock_correction_l1964_196458

/-- The daily gain of the clock in minutes -/
def daily_gain : ℚ := 13 / 4

/-- The number of hours between 8 A.M. on April 10 and 3 P.M. on April 19 -/
def total_hours : ℕ := 223

/-- The negative correction in minutes to be subtracted from the clock -/
def m : ℚ := (daily_gain * total_hours) / 24

theorem clock_correction : m = 30 + 13 / 96 := by
  sorry

end NUMINAMATH_CALUDE_clock_correction_l1964_196458


namespace NUMINAMATH_CALUDE_mean_score_of_all_students_l1964_196449

theorem mean_score_of_all_students
  (avg_score_group1 : ℝ)
  (avg_score_group2 : ℝ)
  (ratio_students : ℚ)
  (h1 : avg_score_group1 = 90)
  (h2 : avg_score_group2 = 75)
  (h3 : ratio_students = 2/5) :
  let total_score := avg_score_group1 * (ratio_students * s) + avg_score_group2 * s
  let total_students := ratio_students * s + s
  total_score / total_students = 79 :=
by
  sorry

#check mean_score_of_all_students

end NUMINAMATH_CALUDE_mean_score_of_all_students_l1964_196449


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1964_196474

theorem simplify_complex_fraction :
  1 / ((2 / (Real.sqrt 2 + 2)) + (3 / (Real.sqrt 3 - 2)) + (4 / (Real.sqrt 5 + 1))) =
  (Real.sqrt 2 + 3 * Real.sqrt 3 - Real.sqrt 5 + 5) / 27 := by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1964_196474


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l1964_196432

/-- Given a rectangle with length x^2 and width x + 5, prove that if its area
    equals three times its perimeter, then x = 3. -/
theorem rectangle_area_perimeter_relation (x : ℝ) : 
  (x^2 * (x + 5) = 3 * (2 * x^2 + 2 * (x + 5))) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l1964_196432


namespace NUMINAMATH_CALUDE_quadratic_coefficient_b_l1964_196480

theorem quadratic_coefficient_b (a b c y₁ y₂ y₃ : ℝ) : 
  y₁ = a + b + c →
  y₂ = a - b + c →
  y₃ = 4*a + 2*b + c →
  y₁ - y₂ = 8 →
  y₃ = y₁ + 2 →
  b = 4 := by
sorry


end NUMINAMATH_CALUDE_quadratic_coefficient_b_l1964_196480


namespace NUMINAMATH_CALUDE_haley_small_gardens_l1964_196401

/-- The number of small gardens Haley had -/
def num_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Theorem stating that Haley had 7 small gardens -/
theorem haley_small_gardens : 
  num_small_gardens 56 35 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_haley_small_gardens_l1964_196401


namespace NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l1964_196478

theorem smallest_solution_for_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 10 → x ≤ y) ∧
  ⌊x^2⌋ - x * ⌊x⌋ = 10 ∧
  x = 131 / 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l1964_196478


namespace NUMINAMATH_CALUDE_inequality_equivalence_system_of_inequalities_equivalence_l1964_196487

theorem inequality_equivalence (x : ℝ) :
  (1 - (x - 3) / 6 > x / 3) ↔ (x < 3) :=
sorry

theorem system_of_inequalities_equivalence (x : ℝ) :
  (x + 1 ≥ 3 * (x - 3) ∧ (x + 2) / 3 - (x - 1) / 4 > 1) ↔ (1 < x ∧ x ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_system_of_inequalities_equivalence_l1964_196487


namespace NUMINAMATH_CALUDE_possible_value_of_n_l1964_196415

theorem possible_value_of_n : ∃ n : ℕ, 
  3 * 3 * 5 * 5 * 7 * 9 = 3 * 3 * 7 * n * n ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_possible_value_of_n_l1964_196415


namespace NUMINAMATH_CALUDE_matrix_power_1000_l1964_196405

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_1000 :
  A^1000 = !![1, 0; 2000, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_1000_l1964_196405


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l1964_196448

theorem more_girls_than_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 345 →
  boys = 138 →
  girls > boys →
  total = girls + boys →
  girls - boys = 69 :=
by sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l1964_196448


namespace NUMINAMATH_CALUDE_range_of_expression_l1964_196418

def line_equation (x : ℝ) : ℝ := -2 * x + 8

theorem range_of_expression (x₁ y₁ : ℝ) :
  y₁ = line_equation x₁ →
  x₁ ∈ Set.Icc 2 5 →
  (y₁ + 1) / (x₁ + 1) ∈ Set.Icc (-1/6) (5/3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l1964_196418


namespace NUMINAMATH_CALUDE_unique_number_property_l1964_196481

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l1964_196481


namespace NUMINAMATH_CALUDE_hidden_faces_sum_l1964_196422

def standard_die_sum : ℕ := 21

def visible_faces : List ℕ := [1, 1, 2, 2, 3, 3, 4, 5, 6, 6]

def total_faces : ℕ := 24

theorem hidden_faces_sum (num_dice : ℕ) (h1 : num_dice = 4) :
  num_dice * standard_die_sum - visible_faces.sum = 51 := by
  sorry

end NUMINAMATH_CALUDE_hidden_faces_sum_l1964_196422


namespace NUMINAMATH_CALUDE_total_pies_sold_l1964_196479

/-- Represents the daily pie sales for a week -/
structure WeekSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculates the total sales for a week -/
def totalSales (sales : WeekSales) : ℕ :=
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday + sales.friday + sales.saturday + sales.sunday

/-- The actual sales data for the week -/
def actualSales : WeekSales := {
  monday := 8,
  tuesday := 12,
  wednesday := 14,
  thursday := 20,
  friday := 20,
  saturday := 20,
  sunday := 20
}

/-- Theorem: The total number of pies sold in the week is 114 -/
theorem total_pies_sold : totalSales actualSales = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_pies_sold_l1964_196479


namespace NUMINAMATH_CALUDE_simplify_expression_l1964_196461

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^3 - b^3) / (a * b^2) - (a * b^2 - b^3) / (a * b^2 - a^3) = (a^3 - a * b^2 + b^4) / (a * b^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1964_196461


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_4_between_34_and_135_l1964_196452

def sumOfMultiplesOf4 (lower upper : ℕ) : ℕ :=
  let firstMultiple := (lower + 3) / 4 * 4
  let lastMultiple := upper / 4 * 4
  let n := (lastMultiple - firstMultiple) / 4 + 1
  n * (firstMultiple + lastMultiple) / 2

theorem sum_of_multiples_of_4_between_34_and_135 :
  sumOfMultiplesOf4 34 135 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_4_between_34_and_135_l1964_196452


namespace NUMINAMATH_CALUDE_factorization_equality_l1964_196470

theorem factorization_equality (a b : ℝ) : 4*a - a*b^2 = a*(2+b)*(2-b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1964_196470


namespace NUMINAMATH_CALUDE_smallest_n_divides_l1964_196491

theorem smallest_n_divides (n : ℕ) : n = 90 ↔ 
  (n > 0 ∧ 
   (315^2 - n^2) ∣ (315^3 - n^3) ∧ 
   ∀ m : ℕ, m > 0 ∧ m < n → ¬((315^2 - m^2) ∣ (315^3 - m^3))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divides_l1964_196491


namespace NUMINAMATH_CALUDE_betty_age_l1964_196446

theorem betty_age (carol alice betty : ℝ) 
  (h1 : carol = 5 * alice)
  (h2 : carol = 2 * betty)
  (h3 : alice = carol - 12) :
  betty = 7.5 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l1964_196446


namespace NUMINAMATH_CALUDE_polynomial_equality_l1964_196406

theorem polynomial_equality (x : ℝ) : 
  (x - 2/3) * (x + 1/2) = x^2 - (1/6)*x - 1/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1964_196406


namespace NUMINAMATH_CALUDE_largest_n_squared_sum_largest_n_exists_largest_n_is_three_l1964_196496

theorem largest_n_squared_sum (n : ℕ+) : 
  (∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18) →
  n ≤ 3 :=
by sorry

theorem largest_n_exists : 
  ∃ (x y z : ℕ+), 3^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18 :=
by sorry

theorem largest_n_is_three : 
  (∃ (n : ℕ+), (∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18) ∧
  (∀ (m : ℕ+), (∃ (a b c : ℕ+), m^2 = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 6*a + 6*b + 6*c - 18) → m ≤ n)) →
  (∃ (x y z : ℕ+), 3^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 18) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_squared_sum_largest_n_exists_largest_n_is_three_l1964_196496


namespace NUMINAMATH_CALUDE_raghu_investment_l1964_196424

theorem raghu_investment (total_investment : ℝ) (vishal_investment : ℝ → ℝ) (trishul_investment : ℝ → ℝ) :
  total_investment = 7225 ∧
  (∀ r, vishal_investment r = 1.1 * (trishul_investment r)) ∧
  (∀ r, trishul_investment r = 0.9 * r) →
  ∃ r, r = 2500 ∧ r + trishul_investment r + vishal_investment r = total_investment :=
by sorry

end NUMINAMATH_CALUDE_raghu_investment_l1964_196424


namespace NUMINAMATH_CALUDE_weight_difference_is_correct_l1964_196493

/-- The difference in grams between the total weight of oranges and apples -/
def weight_difference : ℝ :=
  let apple_weight_oz : ℝ := 27.5
  let apple_unit_weight_oz : ℝ := 1.5
  let orange_count_dozen : ℝ := 5.5
  let orange_unit_weight_g : ℝ := 45
  let oz_to_g_conversion : ℝ := 28.35

  let apple_weight_g : ℝ := apple_weight_oz * oz_to_g_conversion
  let orange_count : ℝ := orange_count_dozen * 12
  let orange_weight_g : ℝ := orange_count * orange_unit_weight_g

  orange_weight_g - apple_weight_g

theorem weight_difference_is_correct :
  weight_difference = 2190.375 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_is_correct_l1964_196493


namespace NUMINAMATH_CALUDE_race_head_start_l1964_196453

/-- Proves that Cristina gave Nicky a 12-second head start in a 100-meter race -/
theorem race_head_start (race_distance : ℝ) (cristina_speed nicky_speed : ℝ) (catch_up_time : ℝ) :
  race_distance = 100 →
  cristina_speed = 5 →
  nicky_speed = 3 →
  catch_up_time = 30 →
  (catch_up_time - (nicky_speed * catch_up_time) / cristina_speed) = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_head_start_l1964_196453


namespace NUMINAMATH_CALUDE_xyz_product_is_27_l1964_196469

theorem xyz_product_is_27 
  (x y z : ℂ) 
  (h1 : x * y + 3 * y = -9)
  (h2 : y * z + 3 * z = -9)
  (h3 : z * x + 3 * x = -9) :
  x * y * z = 27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_is_27_l1964_196469


namespace NUMINAMATH_CALUDE_expression_evaluation_l1964_196445

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 10)
  (h2 : b = a + 2)
  (h3 : a = 4)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 2 ≠ 0)
  (h6 : c + 6 ≠ 0) :
  (a + 2) / (a + 1) * (b - 1) / (b - 2) * (c + 8) / (c + 6) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1964_196445


namespace NUMINAMATH_CALUDE_ratio_equality_l1964_196477

theorem ratio_equality (p q r u v w : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
  (pos_u : 0 < u) (pos_v : 0 < v) (pos_w : 0 < w)
  (sum_pqr : p^2 + q^2 + r^2 = 49)
  (sum_uvw : u^2 + v^2 + w^2 = 64)
  (dot_product : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1964_196477


namespace NUMINAMATH_CALUDE_point_b_coordinates_l1964_196495

/-- Given point A (-1, 5) and vector a (2, 3), if vector AB = 3 * vector a, 
    then the coordinates of point B are (5, 14). -/
theorem point_b_coordinates 
  (A : ℝ × ℝ) 
  (a : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (h1 : A = (-1, 5)) 
  (h2 : a = (2, 3)) 
  (h3 : B.1 - A.1 = 3 * a.1 ∧ B.2 - A.2 = 3 * a.2) : 
  B = (5, 14) := by
sorry


end NUMINAMATH_CALUDE_point_b_coordinates_l1964_196495


namespace NUMINAMATH_CALUDE_non_pine_trees_count_l1964_196440

/-- Given a park with 350 trees, where 70% are pine trees, prove that 105 trees are not pine trees. -/
theorem non_pine_trees_count (total_trees : ℕ) (pine_percentage : ℚ) : 
  total_trees = 350 → pine_percentage = 70 / 100 →
  (total_trees : ℚ) - (pine_percentage * total_trees) = 105 := by
  sorry

end NUMINAMATH_CALUDE_non_pine_trees_count_l1964_196440


namespace NUMINAMATH_CALUDE_geometric_sequence_q_eq_one_l1964_196408

/-- A positive geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a (n + 1) = q * a n

theorem geometric_sequence_q_eq_one
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_prod : a 2 * a 6 = 16)
  (h_sum : a 4 + a 8 = 8) :
  q = 1 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_q_eq_one_l1964_196408


namespace NUMINAMATH_CALUDE_houses_traded_l1964_196492

theorem houses_traded (x y z : ℕ) (h : x + y ≥ z) : ∃ t : ℕ, x - t + y = z :=
sorry

end NUMINAMATH_CALUDE_houses_traded_l1964_196492


namespace NUMINAMATH_CALUDE_radio_cost_price_l1964_196412

/-- The cost price of a radio given its selling price and loss percentage -/
def cost_price (selling_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  selling_price / (1 - loss_percentage / 100)

/-- Theorem: The cost price of a radio sold for 1245 with 17% loss is 1500 -/
theorem radio_cost_price :
  cost_price 1245 17 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_radio_cost_price_l1964_196412


namespace NUMINAMATH_CALUDE_roots_not_real_l1964_196426

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z m : ℂ) : Prop :=
  5 * z^2 - 7 * i * z - m = 0

-- State the theorem
theorem roots_not_real (m : ℂ) :
  ∃ (z₁ z₂ : ℂ), quadratic_equation z₁ m ∧ quadratic_equation z₂ m ∧
  z₁ ≠ z₂ ∧ ¬(z₁.im = 0) ∧ ¬(z₂.im = 0) := by
  sorry

end NUMINAMATH_CALUDE_roots_not_real_l1964_196426


namespace NUMINAMATH_CALUDE_vegan_soy_free_fraction_l1964_196451

theorem vegan_soy_free_fraction (total_dishes : ℕ) (vegan_dishes : ℕ) (soy_vegan_dishes : ℕ) :
  vegan_dishes = total_dishes / 4 →
  vegan_dishes = 6 →
  soy_vegan_dishes = 5 →
  (vegan_dishes - soy_vegan_dishes : ℚ) / total_dishes = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_vegan_soy_free_fraction_l1964_196451


namespace NUMINAMATH_CALUDE_intersection_area_of_bisected_octahedron_l1964_196471

-- Define a regular octahedron
structure RegularOctahedron :=
  (side_length : ℝ)

-- Define the intersection polygon
structure IntersectionPolygon :=
  (octahedron : RegularOctahedron)
  (is_parallel : Bool)
  (is_bisecting : Bool)

-- Define the area of the intersection polygon
def intersection_area (p : IntersectionPolygon) : ℝ :=
  sorry

-- Theorem statement
theorem intersection_area_of_bisected_octahedron 
  (o : RegularOctahedron) 
  (p : IntersectionPolygon) 
  (h1 : o.side_length = 2) 
  (h2 : p.octahedron = o) 
  (h3 : p.is_parallel = true) 
  (h4 : p.is_bisecting = true) : 
  intersection_area p = 9 * Real.sqrt 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_intersection_area_of_bisected_octahedron_l1964_196471


namespace NUMINAMATH_CALUDE_brick_wall_problem_l1964_196400

/-- Represents a brick wall with a specific structure -/
structure BrickWall where
  rows : Nat
  total_bricks : Nat
  bottom_row_bricks : Nat
  row_difference : Nat

/-- Calculates the sum of bricks in all rows of the wall -/
def sum_of_bricks (wall : BrickWall) : Nat :=
  wall.rows * wall.bottom_row_bricks - (wall.rows * (wall.rows - 1) * wall.row_difference) / 2

/-- Theorem stating the properties of the specific brick wall -/
theorem brick_wall_problem : ∃ (wall : BrickWall), 
  wall.rows = 5 ∧ 
  wall.total_bricks = 200 ∧ 
  wall.row_difference = 1 ∧
  sum_of_bricks wall = wall.total_bricks ∧
  wall.bottom_row_bricks = 42 := by
  sorry

end NUMINAMATH_CALUDE_brick_wall_problem_l1964_196400


namespace NUMINAMATH_CALUDE_roots_product_l1964_196425

theorem roots_product (a b : ℝ) : 
  a^2 + a - 2020 = 0 → b^2 + b - 2020 = 0 → (a - 1) * (b - 1) = -2018 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_l1964_196425


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_ratio_l1964_196499

/-- Given an arithmetic progression where the k-th, n-th, and p-th terms form three consecutive terms
    of a geometric progression, the common ratio of the geometric progression is (n-p)/(k-n). -/
theorem arithmetic_geometric_progression_ratio
  (a : ℕ → ℝ) -- The arithmetic progression
  (k n p : ℕ) -- Indices of the terms
  (d : ℝ) -- Common difference of the arithmetic progression
  (h1 : ∀ i, a (i + 1) = a i + d) -- Definition of arithmetic progression
  (h2 : ∃ q : ℝ, a n = a k * q ∧ a p = a n * q) -- Geometric progression condition
  : ∃ q : ℝ, q = (n - p) / (k - n) ∧ a n = a k * q ∧ a p = a n * q :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_ratio_l1964_196499


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1964_196497

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1964_196497


namespace NUMINAMATH_CALUDE_log_xy_value_l1964_196437

theorem log_xy_value (x y : ℝ) 
  (h1 : Real.log (x^2 * y^5) = 2) 
  (h2 : Real.log (x^3 * y^2) = 2) : 
  Real.log (x * y) = 8 / 11 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l1964_196437


namespace NUMINAMATH_CALUDE_tuition_calculation_l1964_196490

/-- Given the total cost and the difference between tuition and room and board,
    calculate the tuition fee. -/
theorem tuition_calculation (total_cost room_and_board tuition : ℕ) : 
  total_cost = tuition + room_and_board ∧ 
  tuition = room_and_board + 704 ∧
  total_cost = 2584 →
  tuition = 1644 := by
  sorry

#check tuition_calculation

end NUMINAMATH_CALUDE_tuition_calculation_l1964_196490


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_l1964_196430

def M : Set ℝ := {x | -x^2 - 5*x + 6 > 0}

def N : Set ℝ := {x | |x + 1| < 1}

theorem M_intersect_N_eq : M ∩ N = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_l1964_196430


namespace NUMINAMATH_CALUDE_amusement_park_admission_fee_l1964_196475

theorem amusement_park_admission_fee (child_fee : ℝ) (total_people : ℕ) (total_fee : ℝ) (num_children : ℕ) :
  child_fee = 1.5 →
  total_people = 315 →
  total_fee = 810 →
  num_children = 180 →
  ∃ (adult_fee : ℝ), adult_fee = 4 ∧ 
    child_fee * num_children + adult_fee * (total_people - num_children) = total_fee :=
by
  sorry

end NUMINAMATH_CALUDE_amusement_park_admission_fee_l1964_196475


namespace NUMINAMATH_CALUDE_solve_sandwich_problem_l1964_196462

/-- Represents the sandwich eating problem over two days -/
def sandwich_problem (total : ℕ) (first_day_fraction : ℚ) (remaining : ℕ) : Prop :=
  let first_day := (total : ℚ) * first_day_fraction
  let second_day := (total : ℕ) - first_day.floor - remaining
  first_day.floor - second_day = 2

/-- The theorem representing the sandwich problem -/
theorem solve_sandwich_problem :
  sandwich_problem 12 (1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_sandwich_problem_l1964_196462


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l1964_196421

theorem square_garden_perimeter (a p : ℝ) (h1 : a > 0) (h2 : p > 0) (h3 : a = 2 * p + 14.25) : p = 38 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l1964_196421


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l1964_196482

/-- The unit price of a volleyball in yuan -/
def volleyball_price : ℝ := 50

/-- The unit price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := 80

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 11

/-- The minimum number of soccer balls to be purchased -/
def min_soccer_balls : ℕ := 2

/-- The cost function for purchasing x volleyballs -/
def cost_function (x : ℝ) : ℝ := -30 * x + 880

/-- The theorem stating the minimum cost of purchasing the balls -/
theorem minimum_cost_theorem :
  ∃ (x : ℝ), 
    0 ≤ x ∧ 
    x ≤ total_balls - min_soccer_balls ∧
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ total_balls - min_soccer_balls → 
      cost_function x ≤ cost_function y ∧
      cost_function x = 610 :=
sorry

end NUMINAMATH_CALUDE_minimum_cost_theorem_l1964_196482


namespace NUMINAMATH_CALUDE_dawn_monthly_savings_l1964_196485

theorem dawn_monthly_savings (annual_salary : ℝ) (months_per_year : ℕ) (savings_rate : ℝ) : 
  annual_salary = 48000 ∧ 
  months_per_year = 12 ∧ 
  savings_rate = 0.1 → 
  (annual_salary / months_per_year) * savings_rate = 400 := by
sorry

end NUMINAMATH_CALUDE_dawn_monthly_savings_l1964_196485


namespace NUMINAMATH_CALUDE_soup_weight_after_four_days_l1964_196429

/-- The weight of soup remaining after four days of reduction -/
def remaining_soup_weight (initial_weight : ℝ) (day1_reduction day2_reduction day3_reduction day4_reduction : ℝ) : ℝ :=
  initial_weight * (1 - day1_reduction) * (1 - day2_reduction) * (1 - day3_reduction) * (1 - day4_reduction)

/-- Theorem stating the remaining weight of soup after four days -/
theorem soup_weight_after_four_days :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |remaining_soup_weight 80 0.40 0.35 0.55 0.50 - 7.02| < ε :=
sorry

end NUMINAMATH_CALUDE_soup_weight_after_four_days_l1964_196429


namespace NUMINAMATH_CALUDE_circle_config_exists_l1964_196463

-- Define the type for our circle configuration
def CircleConfig := Fin 8 → Fin 8

-- Define a function to check if two numbers are connected in our configuration
def isConnected (i j : Fin 8) : Prop :=
  (i.val = j.val + 1 ∧ i.val % 2 = 0) ∨
  (j.val = i.val + 1 ∧ j.val % 2 = 0) ∨
  (i.val = j.val + 2 ∧ i.val % 4 = 0) ∨
  (j.val = i.val + 2 ∧ j.val % 4 = 0)

-- Define the property that the configuration satisfies the problem conditions
def validConfig (c : CircleConfig) : Prop :=
  (∀ i : Fin 8, c i ≠ 0) ∧
  (∀ i j : Fin 8, i ≠ j → c i ≠ c j) ∧
  (∀ d : Fin 7, ∃! (i j : Fin 8), isConnected i j ∧ |c i - c j| = d + 1)

-- State the theorem
theorem circle_config_exists : ∃ c : CircleConfig, validConfig c := by
  sorry

end NUMINAMATH_CALUDE_circle_config_exists_l1964_196463


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_one_zero_l1964_196439

/-- A circle with center (a, a) and radius r -/
structure Circle where
  a : ℝ
  r : ℝ

/-- The circle is tangent to the x-axis at (1, 0) -/
def isTangentAtOneZero (c : Circle) : Prop :=
  c.r = 1 ∧ c.a = 1

/-- The equation of the circle -/
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y - c.a)^2 = c.r^2

theorem circle_tangent_to_x_axis_at_one_zero :
  ∀ c : Circle, isTangentAtOneZero c →
  ∀ x y : ℝ, circleEquation c x y ↔ (x - 1)^2 + (y - 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_one_zero_l1964_196439


namespace NUMINAMATH_CALUDE_problem_solution_l1964_196454

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem problem_solution :
  (∀ x : ℝ, a = 1 → (p x a ∧ q x) ↔ x ∈ Set.Ioo 2 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ a ∈ Set.Icc 1 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1964_196454


namespace NUMINAMATH_CALUDE_negation_of_existential_quadratic_l1964_196441

theorem negation_of_existential_quadratic (p : Prop) : 
  (p ↔ ∃ x : ℝ, x^2 + 2*x + 2 = 0) → 
  (¬p ↔ ∀ x : ℝ, x^2 + 2*x + 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_quadratic_l1964_196441


namespace NUMINAMATH_CALUDE_sunglasses_cap_probability_l1964_196466

theorem sunglasses_cap_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (total_hats : ℕ) 
  (prob_cap_and_sunglasses : ℚ) 
  (h1 : total_sunglasses = 120) 
  (h2 : total_caps = 84) 
  (h3 : total_hats = 60) 
  (h4 : prob_cap_and_sunglasses = 3 / 7) : 
  (prob_cap_and_sunglasses * total_caps) / total_sunglasses = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_sunglasses_cap_probability_l1964_196466


namespace NUMINAMATH_CALUDE_derivative_property_l1964_196455

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem derivative_property (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_property_l1964_196455


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1964_196431

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def N : Set ℝ := {x | -2 < x ∧ x ≤ 4}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1964_196431


namespace NUMINAMATH_CALUDE_sine_cosine_shift_l1964_196459

theorem sine_cosine_shift (ω : ℝ) (h_ω : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 8)
  let g : ℝ → ℝ := λ x ↦ Real.cos (ω * x)
  (∀ x : ℝ, f (x + π / ω) = f x) →
  ∃ k : ℝ, k = 3 * π / 16 ∧ ∀ x : ℝ, g x = f (x + k) :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_shift_l1964_196459


namespace NUMINAMATH_CALUDE_trig_sum_thirty_degrees_l1964_196428

theorem trig_sum_thirty_degrees :
  let tan30 := Real.sqrt 3 / 3
  let sin30 := 1 / 2
  let cos30 := Real.sqrt 3 / 2
  tan30 + 4 * sin30 + 2 * cos30 = (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_thirty_degrees_l1964_196428


namespace NUMINAMATH_CALUDE_double_discount_price_l1964_196473

-- Define the original price
def original_price : ℝ := 33.78

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Define the function to apply a discount
def apply_discount (price : ℝ) : ℝ := price * (1 - discount_rate)

-- Theorem statement
theorem double_discount_price :
  apply_discount (apply_discount original_price) = 19.00125 := by
  sorry

end NUMINAMATH_CALUDE_double_discount_price_l1964_196473


namespace NUMINAMATH_CALUDE_internal_curve_convexity_l1964_196423

-- Define a curve as a function from ℝ to ℝ × ℝ
def Curve := ℝ → ℝ × ℝ

-- Define convexity for a curve
def IsConvex (c : Curve) : Prop := sorry

-- Define the r-neighborhood of a curve
def RNeighborhood (c : Curve) (r : ℝ) : Set (ℝ × ℝ) := sorry

-- Define what it means for a curve to bound a set
def Bounds (c : Curve) (s : Set (ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem internal_curve_convexity 
  (K : Curve) (r : ℝ) (C : Curve) 
  (h_K_convex : IsConvex K) 
  (h_r_pos : r > 0) 
  (h_C_bounds : Bounds C (RNeighborhood K r)) : 
  IsConvex C := by
  sorry

end NUMINAMATH_CALUDE_internal_curve_convexity_l1964_196423


namespace NUMINAMATH_CALUDE_square_perimeter_increase_l1964_196436

theorem square_perimeter_increase (s : ℝ) : 
  (s + 2) * 4 - s * 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_increase_l1964_196436


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1964_196416

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (c * (b + c)) + b^2 / (a * (c + a)) + c^2 / (b * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1964_196416


namespace NUMINAMATH_CALUDE_floor_cube_negative_fraction_l1964_196414

theorem floor_cube_negative_fraction : ⌊(-7/4)^3⌋ = -6 := by
  sorry

end NUMINAMATH_CALUDE_floor_cube_negative_fraction_l1964_196414


namespace NUMINAMATH_CALUDE_longest_segment_in_quarter_pie_l1964_196410

theorem longest_segment_in_quarter_pie (d : ℝ) (h : d = 20) : 
  let r := d / 2
  let l := 2 * r * Real.sin (π / 4)
  l^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_quarter_pie_l1964_196410


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l1964_196404

theorem no_linear_term_condition (x m : ℝ) : 
  (∀ a b c : ℝ, (x - m) * (x - 3) = a * x^2 + c → m = -3) ∧
  (m = -3 → ∃ a c : ℝ, (x - m) * (x - 3) = a * x^2 + c) :=
sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l1964_196404


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1964_196498

/-- Given that 3y varies inversely as the square of x, prove that y = 5/9 when x = 6, 
    given the initial condition y = 5 when x = 2 -/
theorem inverse_variation_problem (k : ℝ) :
  (∀ x y : ℝ, x ≠ 0 → 3 * y = k / (x^2)) →  -- Inverse variation relationship
  (3 * 5 = k / (2^2)) →                     -- Initial condition
  ∃ y : ℝ, 3 * y = k / (6^2) ∧ y = 5/9      -- Conclusion for x = 6
  := by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1964_196498


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l1964_196457

/-- Represents a seating arrangement for cousins in a van --/
structure SeatingArrangement where
  row1 : Fin 4 → Fin 7
  row2 : Fin 4 → Option (Fin 7)

/-- Represents a pair of cousins --/
inductive CousinPair
  | Pair1
  | Pair2
  | Pair3

/-- Returns the pair that a cousin belongs to, if any --/
def cousinPair (cousin : Fin 7) : Option CousinPair := sorry

/-- Checks if a seating arrangement is valid according to the rules --/
def isValidArrangement (arr : SeatingArrangement) : Prop := sorry

/-- Counts the number of valid seating arrangements --/
def countValidArrangements : Nat := sorry

/-- Theorem stating that the number of valid seating arrangements is 240 --/
theorem valid_arrangements_count :
  countValidArrangements = 240 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l1964_196457


namespace NUMINAMATH_CALUDE_f_of_2_eq_0_l1964_196417

/-- The function f(x) = x^3 - 3x^2 + 2x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- Theorem: f(2) = 0 -/
theorem f_of_2_eq_0 : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_eq_0_l1964_196417
